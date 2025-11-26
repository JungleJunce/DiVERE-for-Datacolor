from PySide6.QtCore import QObject, Signal, QRunnable, Slot, QThreadPool, QTimer
from PySide6.QtGui import QPixmapCache
from typing import Optional, List, Tuple
import numpy as np
from pathlib import Path
from colour.temperature import CCT_to_xy_CIE_D

from .data_types import ImageData, ColorGradingParams, Preset, CropInstance, PresetBundle, CropPresetEntry, InputTransformationDefinition, MatrixDefinition, CurveDefinition, PipelineConfig, UIStateConfig, ContactsheetProfile, CropAddDirection, PreviewConfig
from .image_manager import ImageManager
from .color_space import ColorSpaceManager
from .the_enlarger import TheEnlarger
from .film_type_controller import FilmTypeController
from .folder_navigator import FolderNavigator
from ..utils.auto_preset_manager import AutoPresetManager
from ..utils.enhanced_config_manager import enhanced_config_manager
from . import color_science
from ..i18n import tr


class _PreviewWorkerSignals(QObject):
    result = Signal(ImageData)
    error = Signal(str)
    finished = Signal()


class _PreviewWorker(QRunnable):
    def __init__(self, image: ImageData, params: ColorGradingParams, the_enlarger: TheEnlarger,
                 color_space_manager: ColorSpaceManager, convert_to_monochrome_in_idt: bool = False,
                 orientation: int = 0, idt_gamma: float = 1.0, custom_colorspace_def: dict = None):
        super().__init__()
        self.image = image
        self.params = params
        self.the_enlarger = the_enlarger
        self.color_space_manager = color_space_manager
        self.convert_to_monochrome_in_idt = convert_to_monochrome_in_idt
        self.orientation = orientation
        self.idt_gamma = idt_gamma
        self.custom_colorspace_def = custom_colorspace_def
        self.signals = _PreviewWorkerSignals()

    @Slot()
    def run(self):
        print("[DEBUG] PreviewWorker.run() 开始执行", flush=True)
        # —— 关键点：把大对象搬到局部变量，再把 self 上的引用清掉 ——
        image = self.image
        params = self.params
        orientation = self.orientation
        idt_gamma = self.idt_gamma
        custom_colorspace_def = self.custom_colorspace_def
        self.image = None
        self.params = None
        self.idt_gamma = None
        self.custom_colorspace_def = None
        print(f"[DEBUG] PreviewWorker.run(): 图像尺寸={image.width}x{image.height}, orientation={orientation}", flush=True)

        try:
            # === Step A: 应用 IDT Gamma（在旋转之前）===
            if abs(idt_gamma - 1.0) > 1e-6:
                print(f"[DEBUG] PreviewWorker.run(): 应用 IDT gamma={idt_gamma}", flush=True)
                image.array = self.the_enlarger.pipeline_processor.math_ops.apply_power(
                    image.array, idt_gamma, use_optimization=True
                )

            # === Step B: 注册自定义色彩空间（如果需要）===
            if custom_colorspace_def:
                try:
                    cs_name = custom_colorspace_def.get('name')
                    primaries_xy = np.array(custom_colorspace_def.get('primaries_xy'), dtype=float)
                    white_point_xy = np.array(custom_colorspace_def.get('white_point_xy'), dtype=float)
                    gamma = float(custom_colorspace_def.get('gamma', 1.0))

                    print(f"[DEBUG] PreviewWorker.run(): 注册自定义色彩空间 {cs_name}", flush=True)
                    self.color_space_manager.register_custom_colorspace(
                        name=cs_name,
                        primaries_xy=primaries_xy,
                        white_point_xy=white_point_xy,
                        gamma=gamma
                    )
                except Exception as e:
                    # 注册失败不应导致预览失败，记录错误并继续
                    print(f"[WARNING] PreviewWorker.run(): 注册自定义色彩空间失败: {e}", flush=True)

            # === Step C: 色彩空间转换 ===
            print(f"[DEBUG] PreviewWorker.run(): 应用色彩空间转换 {params.input_color_space_name}", flush=True)
            image = self.color_space_manager.set_image_color_space(
                image, params.input_color_space_name
            )
            image = self.color_space_manager.convert_to_working_space(
                image, skip_gamma_inverse=True
            )

            # === Step D: Apply rotation if needed ===
            if orientation % 360 != 0:
                k = (orientation // 90) % 4
                if k != 0:
                    print(f"[DEBUG] PreviewWorker.run(): 应用旋转 k={k} (orientation={orientation})", flush=True)
                    image.array = np.rot90(image.array, k=int(k))
                    print(f"[DEBUG] PreviewWorker.run(): 旋转后图像尺寸={image.width}x{image.height}", flush=True)

            print("[DEBUG] PreviewWorker.run(): 准备monochrome_converter", flush=True)
            monochrome_converter = None
            if self.convert_to_monochrome_in_idt:
                monochrome_converter = self.color_space_manager.convert_to_monochrome
                print("[DEBUG] PreviewWorker.run(): 将使用monochrome转换", flush=True)

            print(f"[DEBUG] PreviewWorker.run(): params.enable_density_curve={params.enable_density_curve}, curve_name={params.density_curve_name}", flush=True)
            print(f"[DEBUG] PreviewWorker.run(): 曲线点数: RGB={len(params.curve_points) if params.curve_points else 0}, R={len(params.curve_points_r) if params.curve_points_r else 0}", flush=True)
            print("[DEBUG] PreviewWorker.run(): 开始apply_full_pipeline...", flush=True)
            result_image = self.the_enlarger.apply_full_pipeline(
                image,
                params,
                convert_to_monochrome_in_idt=self.convert_to_monochrome_in_idt,
                monochrome_converter=monochrome_converter,
            )
            print(f"[DEBUG] PreviewWorker.run(): apply_full_pipeline完成，结果尺寸={result_image.width}x{result_image.height}", flush=True)

            print("[DEBUG] PreviewWorker.run(): 开始convert_to_display_space...", flush=True)
            result_image = self.color_space_manager.convert_to_display_space(
                result_image, "DisplayP3"
            )
            print("[DEBUG] PreviewWorker.run(): convert_to_display_space完成", flush=True)

            print("[DEBUG] PreviewWorker.run(): 发射result信号", flush=True)
            self.signals.result.emit(result_image)
            print("[DEBUG] PreviewWorker.run(): result信号已发射", flush=True)

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            error_msg = f"{e}\n{tb}"
            print(f"[ERROR] PreviewWorker.run(): 处理失败: {error_msg}", flush=True)
            self.signals.error.emit(error_msg)

        finally:
            # 不再依赖 del，只发 finished，实际引用已经在上面提前断掉了
            print("[DEBUG] PreviewWorker.run(): 发射finished信号", flush=True)
            self.signals.finished.emit()
            print("[DEBUG] PreviewWorker.run() 执行完成", flush=True)


class ApplicationContext(QObject):
    """
    应用上下文，作为单一数据源 (Single Source of Truth)。
    管理应用状态、核心业务逻辑和与UI的交互。
    """
    # =================
    # 信号 (Signals)
    # =================
    image_loaded = Signal()
    preview_updated = Signal(ImageData)
    params_changed = Signal(ColorGradingParams)
    status_message_changed = Signal(str)
    autosave_requested = Signal()
    # 请求清空预览
    preview_clear_requested = Signal()  # <<< 新增：请求清空预览
    # 裁剪变化（None 或 (x,y,w,h) 归一化）
    crop_changed = Signal(object)
    # 胶片类型变化信号
    film_type_changed = Signal(str)
    # curves配置重载信号
    curves_config_reloaded = Signal()
    # 旋转完成信号
    rotation_completed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        
        # =================
        # 核心服务实例
        # =================
        self.image_manager = ImageManager()
        self.color_space_manager = ColorSpaceManager()
        
        # 从配置读取proxy尺寸设置并创建PreviewConfig
        proxy_max_size = enhanced_config_manager.get_ui_setting("proxy_max_size", 2000)
        preview_config = PreviewConfig(proxy_max_size=proxy_max_size)
        self.the_enlarger = TheEnlarger(preview_config=preview_config)
        
        self.film_type_controller = FilmTypeController()
        self.folder_navigator = FolderNavigator(self.image_manager)
        self.auto_preset_manager = AutoPresetManager()

        # 注意：ColorSpaceManager 不需要 context 引用
        # CCMOptimizer 通过 app_context 参数直接传递，不需要通过 color_space_manager.context
        # 移除此赋值以避免循环引用：ApplicationContext ↔ ColorSpaceManager
        # self.color_space_manager.context = self  # 已移除

        # =================
        # 状态变量
        # =================
        self._current_image: Optional[ImageData] = None
        self._current_proxy: Optional[ImageData] = None # 应用输入变换和工作空间变换后的代理
        self._current_params: ColorGradingParams = self._create_default_params()
        # 当前胶片类型
        self._current_film_type: str = "color_negative_c41"
        
        # =================
        # 后台处理
        # =================
        self._preview_busy: bool = False
        self._preview_pending: bool = False
        self._loading_image: bool = False  # 图像加载状态标志

        # 进程隔离配置（用于解决 macOS heap 内存不归还问题）
        # 参考：PROCESS_ISOLATION_ANALYSIS.md
        self._use_process_isolation: bool = self._should_use_process_isolation()

        if self._use_process_isolation:
            # 进程模式相关字段
            self._preview_worker_process = None  # PreviewWorkerProcess 实例
            self._proxy_shared_memory = None  # shared_memory.SharedMemory 实例
            self._proxy_needs_reload = False  # Proxy 是否需要重载到 worker（优化标志）
            self._result_poll_timer = QTimer()
            self._result_poll_timer.timeout.connect(self._poll_preview_result)

            # 注册 atexit handler 确保程序退出时清理 worker 进程
            import atexit
            atexit.register(self._atexit_cleanup)
        else:
            # 线程模式相关字段
            self.thread_pool: QThreadPool = QThreadPool.globalInstance()
            self.thread_pool.setMaxThreadCount(1)
            # 设置线程栈大小为8MB以避免macOS ARM版本打包后的栈溢出问题
            # 这主要解决OpenBLAS在numpy.linalg.solve中的栈空间不足问题
            self.thread_pool.setStackSize(8 * 1024 * 1024)  # 8MB

        # AI自动校色迭代状态
        self._auto_color_iterations = 0
        self._get_preview_for_auto_color_callback = None

        # 中性点自动增益迭代状态
        self._neutral_point_iterations = 0
        self._neutral_point_norm = None  # (x, y) 归一化坐标
        self._neutral_point_callback = None
        self._neutral_point_sample_size = 5

        # 色卡优化状态管理
        self._ccm_optimization_active = False
        
        # Reference Color缓存（用于确保Preview和Optimizer数据一致性）
        self._cached_reference_colors = None
        self._reference_colorspace = None
        self._reference_filename = None

        # 防抖自动保存
        self._autosave_timer = QTimer()
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(50) # 500ms delay for autosave
        self._autosave_timer.timeout.connect(self.autosave_requested.emit)

        # 应用集中式"默认预设"（config/defaults/default.json 或内置回退）
        try:
            from divere.utils.defaults import load_default_preset
            default_preset = load_default_preset()
            # 仅在无图像初始状态或每次启动时应用到 contactsheet 基线
            self.load_preset(default_preset)
        except Exception:
            pass
        
        # 裁剪状态
        self._crops: list = []  # list[CropInstance]
        self._active_crop_id: Optional[str] = None
        self._crop_focused: bool = False
        
        # 预设 Profile：contactsheet 与 per-crop 参数集
        self._current_profile_kind: str = 'contactsheet'  # 'contactsheet' | 'crop'
        self._contactsheet_profile: ContactsheetProfile = ContactsheetProfile(
            params=self._current_params.shallow_copy(),  # 优化：只读初始化，使用 shallow_copy()
            orientation=0,
            crop_rect=None
        )
        self._per_crop_params: dict = {}
        
        # 连接文件夹导航信号
        self.folder_navigator.file_changed.connect(self.load_image)

    def _should_use_process_isolation(self) -> bool:
        """根据配置和平台决定是否使用进程隔离

        进程隔离用于解决 macOS heap 内存不归还问题。
        配置项位于 config/app_settings.json 的 ui.use_process_isolation
        参考文档：PROCESS_ISOLATION_ANALYSIS.md

        Returns:
            bool: 是否使用进程隔离
        """
        import platform

        # 从配置文件读取（config/app_settings.json）
        config_value = enhanced_config_manager.get_ui_setting(
            "use_process_isolation", "never"  # 默认禁用，等稳定后改为 "auto"
        ).lower()

        if config_value == "never":
            return False
        elif config_value == "always":
            return True
        else:  # "auto"
            # macOS/Linux: 默认启用（内存问题严重）
            # Windows: 默认禁用（multiprocessing 复杂，需要 if __name__ == '__main__' 保护）
            return platform.system() in ['Darwin', 'Linux']

    def _create_default_params(self) -> ColorGradingParams:
        params = ColorGradingParams()
        params.density_gamma = 2.6
        params.density_matrix = self.the_enlarger.pipeline_processor.get_density_matrix_array("Cineon_States_M_to_Print_Density")
        params.density_matrix_name = "Cineon_States_M_to_Print_Density"
        params.enable_density_matrix = True

        # 默认曲线改由 default preset 决定；此处不再强行加载硬编码曲线

        return params

    def ensure_thread_pool(self) -> QThreadPool:
        """确保线程池已初始化并返回

        在进程隔离模式下，某些功能（如CCM优化）仍需要使用线程池。
        此方法按需创建线程池，避免在__init__中条件创建导致的兼容问题。

        Returns:
            QThreadPool: 全局线程池实例
        """
        if not hasattr(self, 'thread_pool') or self.thread_pool is None:
            self.thread_pool = QThreadPool.globalInstance()
            self.thread_pool.setMaxThreadCount(1)
            self.thread_pool.setStackSize(8 * 1024 * 1024)  # 8MB
        return self.thread_pool

    def _load_smart_default_preset(self, file_path: str):
        """使用智能预设加载器加载默认预设"""
        try:
            from divere.utils.smart_preset_loader import SmartPresetLoader
            loader = SmartPresetLoader()
            preset = loader.get_smart_default_preset(file_path)
            
            if preset:
                self.load_preset(preset)
                self.status_message_changed.emit(tr("app_context.preset_messages.auto_classification_applied"))
            else:
                # 回退到通用默认
                self._load_generic_default_preset()
                
        except Exception as e:
            # 回退到通用默认
            self._load_generic_default_preset()
            self.status_message_changed.emit(tr("app_context.preset_messages.auto_classification_failed", error=e))
    
    def _load_generic_default_preset(self):
        """加载通用默认预设"""
        try:
            from divere.utils.path_manager import get_default_preset_path
            default_preset_path = get_default_preset_path("default.json")
            if default_preset_path:
                with open(default_preset_path, "r", encoding="utf-8") as f:
                    import json
                    data = json.load(f)
                    from divere.core.data_types import Preset
                    preset = Preset.from_dict(data)
                    self.load_preset(preset)
            else:
                raise FileNotFoundError("找不到默认预设文件")
        except Exception:
            self._current_params = self._create_default_params()
            self._contactsheet_params = self._current_params.shallow_copy()  # 优化：只读备份（死代码），使用 shallow_copy()
            self.params_changed.emit(self._current_params)

    # =================
    # 属性访问器 (Getters)
    # =================
    def get_current_image(self) -> Optional[ImageData]:
        return self._current_image
        
    def get_current_params(self) -> ColorGradingParams:
        return self._current_params

    def get_input_color_space(self) -> str:
        return self._current_params.input_color_space_name

    def get_contactsheet_crop_rect(self) -> Optional[tuple[float, float, float, float]]:
        """获取接触印相/原图的单张裁剪矩形（归一化），可能为 None。"""
        return self._contactsheet_profile.crop_rect

    # =================
    # 裁剪（Crops）API - 支持新的CropInstance模型
    # =================
    def set_single_crop(self, rect_norm: tuple[float, float, float, float], orientation: int = None, preserve_focus: bool = True) -> None:
        """设置/替换单一裁剪，并设为激活。
        
        Args:
            rect_norm: 归一化裁剪坐标
            orientation: crop的orientation（None时使用默认值0）
            preserve_focus: 是否保持当前的crop_focused状态（默认True）
        """
        try:
            x, y, w, h = [float(max(0.0, min(1.0, v))) for v in rect_norm]
            # 规范到图像范围内
            w = max(0.0, min(1.0 - x, w))
            h = max(0.0, min(1.0 - y, h))
            
            # 使用新的CropInstance模型
            # 如果没有指定orientation，使用默认值0（不继承全局orientation）
            if orientation is None:
                orientation = 0  # 默认无旋转，crop独立于全局orientation
            
            crop_instance = CropInstance(
                id="default",
                name="默认裁剪",
                rect_norm=(x, y, w, h),
                orientation=orientation
            )
            
            # 更新内部状态：直接存储CropInstance
            self._crops = [crop_instance]  # 存储CropInstance对象而不是字典
            self._active_crop_id = crop_instance.id
            
            # 保持crop_focused状态（重要修复！）
            if not preserve_focus:
                self._crop_focused = False
            # 如果preserve_focus=True，保持当前的_crop_focused状态不变
            
            # 发送crop变化信号
            self.crop_changed.emit(crop_instance.rect_norm)
            # 触发自动保存（裁剪变化也应持久化）
            self._autosave_timer.start()
        except Exception:
            pass

    def get_active_crop(self) -> Optional[tuple[float, float, float, float]]:
        """获取激活crop的坐标（向后兼容）"""
        crop_instance = self.get_active_crop_instance()
        return crop_instance.rect_norm if crop_instance else None
    
    def get_active_crop_instance(self) -> Optional[CropInstance]:
        """获取激活的CropInstance对象"""
        if not self._active_crop_id or not self._crops:
            return None
        
        # 直接从存储的CropInstance对象获取
        for crop_instance in self._crops:
            if isinstance(crop_instance, CropInstance) and crop_instance.id == self._active_crop_id:
                return crop_instance
            elif isinstance(crop_instance, dict) and crop_instance.get('id') == self._active_crop_id:
                # 兼容旧格式（字典）
                return CropInstance(
                    id=crop_instance.get('id', 'default'),
                    name=crop_instance.get('name', '默认裁剪'),
                    rect_norm=crop_instance.get('rect', (0, 0, 1, 1)),
                    orientation=0  # 旧格式默认无旋转
                )
        return None
    
    def get_all_crops(self) -> List[CropInstance]:
        """获取所有裁剪实例列表"""
        return self._crops.copy()
    
    def get_active_crop_id(self) -> Optional[str]:
        """获取当前活跃的裁剪ID"""
        return self._active_crop_id
    
    def get_current_profile_kind(self) -> str:
        """获取当前Profile类型 'contactsheet' 或 'crop'"""
        return self._current_profile_kind

    def clear_crop(self) -> None:
        self._crops = []
        self._active_crop_id = None
        self._crop_focused = False
        self._contactsheet_profile.crop_rect = None
        self.crop_changed.emit(None)
        self._autosave_timer.start()
        # 若无任何裁剪，回到 contactsheet（single 语义）
        self._current_profile_kind = 'contactsheet'

    def focus_on_active_crop(self) -> None:
        if self.get_active_crop() is None or not self._current_image:
            return
        self._crop_focused = True
        print("[DEBUG] Call for _prepare_proxy: focus_on_active_crop()"); self._prepare_proxy()
        # 模式切换后，worker 会在预览时自动 reload proxy（热重载）
        print("[DEBUG] Trigger for preview_update: focus_on_active_crop()"); self._trigger_preview_update()
        self._autosave_timer.start()

    def restore_crop_preview(self) -> None:
        if not self._current_image:
            return
        self._crop_focused = False
        print("[DEBUG] Call for _prepare_proxy: restore_crop_preview()"); self._prepare_proxy()
        # 模式切换后，worker 会在预览时自动 reload proxy（热重载）
        print("[DEBUG] Trigger for preview_update: restore_crop_preview()");self._trigger_preview_update()
        self._autosave_timer.start()

    def focus_on_contactsheet_crop(self) -> None:
        """在原图/接触印相模式下聚焦 contactsheet 裁剪（若存在）。"""
        try:
            if not self._current_image:
                return
            if self._active_crop_id is not None:
                return  # 仅在原图/接触印相模式下允许
            if self._contactsheet_profile.crop_rect is None:
                return
            self._crop_focused = True
            print("[DEBUG] Call for _prepare_proxy: focus_on_contactsheet_crop()"); self._prepare_proxy()
            # 模式切换后，worker 会在预览时自动 reload proxy（热重载）
            self._trigger_preview_update()
            self._autosave_timer.start()
        except Exception:
            pass

    # =================
    # Profile 切换与裁剪管理（Bundle 支持）
    # =================
    def switch_to_contactsheet(self) -> None:
        """切换到原图 Profile（不自动聚焦）- 无orientation同步"""
        try:
            self._current_profile_kind = 'contactsheet'
            self._current_params = self._contactsheet_profile.params.shallow_copy()  # 优化：只读加载，使用 shallow_copy()
            self._crop_focused = False
            self._active_crop_id = None
            # 移除orientation同步 - UI将直接读取contactsheet的orientation
            # 发送参数变更信号
            self.params_changed.emit(self._current_params)
            print("[DEBUG] Call for _prepare_proxy: switch_to_contactsheet()")
            self._prepare_proxy(); self._trigger_preview_update()
        except Exception:
            pass

    def switch_to_crop(self, crop_id: str) -> None:
        """切换到指定裁剪的 Profile（不自动聚焦）- 无orientation同步"""
        try:
            self._current_profile_kind = 'crop'
            self._active_crop_id = crop_id
            params = self._per_crop_params.get(crop_id)
            if params is None:
                # 如果不存在，使用 contactsheet 复制一份初始化
                params = self._contactsheet_profile.params.shallow_copy()  # 优化：只读初始化，使用 shallow_copy()
                self._per_crop_params[crop_id] = params
            self._current_params = params.shallow_copy()  # 优化：只读加载，使用 shallow_copy()
            self._crop_focused = False
            # 移除orientation同步 - UI将直接读取crop的orientation
            # 发送参数变更信号
            self.params_changed.emit(self._current_params)
            print("[DEBUG] Call for _prepare_proxy: switch_to_crop()")
            self._prepare_proxy(); self._trigger_preview_update()
        except Exception:
            pass

    def switch_to_crop_focused(self, crop_id: str) -> None:
        """一次性切换到指定裁剪并进入聚焦模式（单次预览更新，避免先显示原图再聚焦的闪烁）。"""
        try:
            self._current_profile_kind = 'crop'
            self._active_crop_id = crop_id
            # 参数集
            params = self._per_crop_params.get(crop_id)
            if params is None:
                # 优先继承接触印相设置
                if self._contactsheet_profile.params:
                    params = self._contactsheet_profile.params.shallow_copy()  # 优化：只读初始化，使用 shallow_copy()
                else:
                    # 没有接触印相设置时，使用智能分类默认
                    if self._current_image:
                        self._load_smart_default_preset(self._current_image.file_path)
                        params = self._current_params.shallow_copy()  # 优化：只读初始化，使用 shallow_copy()
                    else:
                        # 没有图像时，使用通用默认
                        self._load_generic_default_preset()
                        params = self._current_params.shallow_copy()  # 优化：只读初始化，使用 shallow_copy()

                self._per_crop_params[crop_id] = params
            self._current_params = params.shallow_copy()  # 优化：只读加载，使用 shallow_copy()
            # 移除orientation同步 - UI将直接读取crop的orientation
            # 直接进入聚焦
            self._crop_focused = True
            # 一次性刷新
            self.params_changed.emit(self._current_params)
            print("[DEBUG] Call for _prepare_proxy: switch_to_crop_focused()"); self._prepare_proxy()
            # 模式切换后，worker 会在预览时自动 reload proxy（热重载）
            self._trigger_preview_update()
            self._autosave_timer.start()
        except Exception:
            pass


    def _calculate_visual_aspect_ratio(self, orientation: int) -> float:
        """计算考虑orientation的视觉宽高比（用户看到的宽高比）
        
        Args:
            orientation: contact sheet的orientation角度
            
        Returns:
            视觉宽高比（视觉宽度/视觉高度）
        """
        if not self._current_image or self._current_image.array is None:
            return 1.0
            
        h, w = self._current_image.array.shape[:2]
        
        # 根据orientation计算视觉宽高比
        if orientation % 180 == 90:  # 90°或270°
            # 宽高互换
            return h / w if w > 0 else 1.0
        else:  # 0°或180°
            # 保持原始宽高比
            return w / h if h > 0 else 1.0


    def smart_add_crop(self, direction: CropAddDirection = CropAddDirection.DOWN_RIGHT) -> str:
        """智能添加裁剪：根据现有裁剪自动计算最佳位置（仅使用direction mapping处理orientation）
        
        Args:
            direction: 指定的添加方向（用户视觉方向）
        """
        try:
            from divere.utils.crop_layout_manager import CropLayoutManager
            
            # 获取contact sheet的orientation
            cs_orientation = self._contactsheet_profile.orientation
            
            # 直接使用现有crops的显示坐标（不做坐标变换）
            existing_crops_display = [crop.rect_norm for crop in self._crops]
            
            # 在显示坐标系中计算新位置，依靠direction mapping处理方向转换
            layout_manager = CropLayoutManager()
            new_rect_display = layout_manager.find_next_position(
                existing_crops=existing_crops_display,
                template_size=None,  # 使用最后一个裁剪的尺寸或默认值
                direction=direction,  # 用户视觉方向
                orientation=cs_orientation  # 仅用于direction mapping
            )
            
            # 直接使用计算结果（已经是显示坐标系）
            return self.add_crop(new_rect_display, cs_orientation)
            
        except Exception as e:
            print(f"智能添加裁剪失败: {e}")
            # 失败时使用默认位置
            cs_orientation = self._contactsheet_profile.orientation
            return self.add_crop((0.1, 0.1, 0.25, 0.25), cs_orientation)
    
    def add_crop(self, rect_norm: Tuple[float, float, float, float], orientation: int) -> str:
        """新增一个裁剪：复制 contactsheet 的参数作为初始值，返回 crop_id。"""
        try:
            # 规范 rect
            x, y, w, h = [float(max(0.0, min(1.0, v))) for v in rect_norm]
            w = max(0.0, min(1.0 - x, w)); h = max(0.0, min(1.0 - y, h))
            crop_id = f"crop_{len(self._crops) + 1}"
            crop = CropInstance(id=crop_id, name=f"裁剪 {len(self._crops) + 1}", rect_norm=(x, y, w, h), orientation=int(orientation) % 360)
            self._crops.append(crop)
            self._active_crop_id = crop_id
            # 初始化该裁剪的参数集（shallow_copy contactsheet）
            self._per_crop_params[crop_id] = self._contactsheet_profile.params.shallow_copy()  # 优化：只读初始化，使用 shallow_copy()
            # 添加crop不切换profile，保持在contactsheet（避免不必要的prepare_proxy）
            # 发信号
            self.crop_changed.emit(crop.rect_norm)
            self._autosave_timer.start()
            return crop_id
        except Exception:
            return ""
    
    def update_active_crop(self, rect_norm: tuple[float, float, float, float]) -> None:
        """更新当前活跃裁剪的区域"""
        try:
            crop_instance = self.get_active_crop_instance()
            if crop_instance:
                # 规范 rect
                x, y, w, h = [float(max(0.0, min(1.0, v))) for v in rect_norm]
                w = max(0.0, min(1.0 - x, w)); h = max(0.0, min(1.0 - y, h))
                crop_instance.rect_norm = (x, y, w, h)
                # 发信号
                self.crop_changed.emit(crop_instance.rect_norm)
                # 如果当前处于聚焦状态，需要重新准备proxy
                if self._crop_focused:
                    print("[DEBUG] Call for _prepare_proxy: update_active_crop()"); self._prepare_proxy()
                    # Crop rect调整后，worker 会在预览时自动 reload proxy（热重载）
                    self._trigger_preview_update()
                self._autosave_timer.start()
        except Exception as e:
            print(f"更新裁剪失败: {e}")

    def delete_crop(self, crop_id: str) -> None:
        """删除指定裁剪；维护active_id与预览状态。"""
        try:
            if not crop_id:
                return
            kept: list[CropInstance] = []
            removed = False
            for crop in self._crops:
                if isinstance(crop, CropInstance) and crop.id == crop_id:
                    removed = True
                    continue
                kept.append(crop)
            if not removed:
                return
            self._crops = kept
            # 清理该裁剪的独立参数集
            try:
                if crop_id in self._per_crop_params:
                    del self._per_crop_params[crop_id]
            except Exception:
                pass
            # 维护 active id
            if self._active_crop_id == crop_id:
                self._active_crop_id = kept[0].id if kept else None
            # 退出聚焦（只有之前在聚焦状态才需要prepare_proxy）
            was_focused = self._crop_focused
            self._crop_focused = False
            # 触发预览刷新（只有从聚焦状态退出才需要prepare_proxy）
            if was_focused:
                print("[DEBUG] Call for _prepare_proxy: delete_crop()"); self._prepare_proxy()
            self._trigger_preview_update()
            # 发裁剪变化信号（传当前active或None）
            try:
                active_rect = self.get_active_crop()
                self.crop_changed.emit(active_rect)
            except Exception:
                pass
            # 自动保存
            self._autosave_timer.start()
        except Exception as e:
            print(f"删除裁剪失败: {e}")

    def apply_contactsheet_to_active_crop(self) -> None:
        """将接触印相（contactsheet）的参数复制到当前活跃裁剪的参数集。"""
        try:
            print(f"DEBUG: apply_contactsheet_to_active_crop开始执行")
            active_id = self._active_crop_id
            print(f"DEBUG: active_id = {active_id}")
            if not active_id:
                print("DEBUG: 没有active_crop_id，直接返回")
                return
            
            # 检查contactsheet参数是否存在
            if not self._contactsheet_profile.params:
                print("DEBUG: contactsheet_params为空")
                return
                
            # 复制参数
            cs_params = self._contactsheet_profile.params.shallow_copy()  # 优化：只读复制，使用 shallow_copy()
            print(f"DEBUG: 复制的contactsheet参数数量: {len(vars(cs_params))}")
            self._per_crop_params[active_id] = cs_params
            print("DEBUG: 参数已复制到per_crop_params")

            # 若当前正聚焦该裁剪，同步当前参数并刷新预览
            if self._crop_focused:
                print("DEBUG: 当前处于crop聚焦状态，同步参数")
                self._current_params = cs_params.shallow_copy()  # 优化：只读加载，使用 shallow_copy()
                self.params_changed.emit(self._current_params)
                print("DEBUG: params_changed信号已发射")
                self._trigger_preview_update()  # 只是参数复制，不需要prepare_proxy
                print("DEBUG: 预览更新已触发")
            else:
                print("DEBUG: 当前未处于crop聚焦状态")
                
            self._autosave_timer.start()
            print("DEBUG: apply_contactsheet_to_active_crop执行完成")
        except Exception as e:
            print(f"沿用接触印相设置失败: {e}")
            import traceback
            traceback.print_exc()

    def apply_active_crop_to_contactsheet(self) -> None:
        """将当前活跃裁剪的参数复制到接触印相（contactsheet）的参数集。"""
        try:
            print(f"DEBUG: apply_active_crop_to_contactsheet开始执行")
            active_id = self._active_crop_id
            print(f"DEBUG: active_id = {active_id}")
            
            # 获取当前crop的参数
            crop_params = None
            if self._crop_focused and active_id:
                # 如果当前在crop聚焦状态，使用当前参数
                crop_params = self._current_params.shallow_copy()  # 优化：只读复制，使用 shallow_copy()
                print("DEBUG: 从当前参数获取crop参数（聚焦状态）")
            elif active_id and active_id in self._per_crop_params:
                # 如果不在聚焦状态，从per_crop_params获取
                crop_params = self._per_crop_params[active_id].shallow_copy()  # 优化：只读复制，使用 shallow_copy()
                print("DEBUG: 从per_crop_params获取crop参数（非聚焦状态）")
            else:
                print("DEBUG: 没有有效的crop参数可复制")
                return

            if not crop_params:
                print("DEBUG: crop参数为空")
                return

            print(f"DEBUG: 复制的crop参数数量: {len(vars(crop_params))}")
            # 复制参数到contactsheet
            self._contactsheet_profile.params = crop_params.shallow_copy()  # 优化：只读保存，使用 shallow_copy()
            print("DEBUG: 参数已复制到contactsheet_params")

            # 如果当前处于contactsheet模式，同步当前参数并刷新预览
            if self._current_profile_kind == 'contactsheet':
                print("DEBUG: 当前处于contactsheet模式，同步参数")
                self._current_params = crop_params.shallow_copy()  # 优化：只读加载，使用 shallow_copy()
                self.params_changed.emit(self._current_params)
                print("DEBUG: params_changed信号已发射")
                self._trigger_preview_update()  # 只是参数复制，不需要prepare_proxy
                print("DEBUG: 预览更新已触发")
            else:
                print("DEBUG: 当前未处于contactsheet模式")
                
            self._autosave_timer.start()
            print("DEBUG: apply_active_crop_to_contactsheet执行完成")
        except Exception as e:
            print(f"应用当前设置到接触印相失败: {e}")
            import traceback
            traceback.print_exc()

    def export_preset_bundle(self) -> PresetBundle:
        """导出 Bundle：contactsheet + 各裁剪条目（每个带独立Preset）。"""
        # 1) contactsheet preset
        cs_preset = self._create_preset_from_params(self._contactsheet_profile.params, name="contactsheet")
        cs_preset.orientation = self._contactsheet_profile.orientation  # 保存原图的 orientation
        # 写入原图文件名，满足 v3 metadata.raw_file 必填
        try:
            if self._current_image and getattr(self._current_image, 'file_path', ''):
                cs_preset.raw_file = Path(self._current_image.file_path).name
        except Exception:
            pass
        # 写入 contactsheet 裁剪（旧字段，用于兼容）
        if self._contactsheet_profile.crop_rect is not None:
            cs_preset.crop = tuple(self._contactsheet_profile.crop_rect)
        # 2) crops
        entries: list[CropPresetEntry] = []
        for crop in self._crops:
            params = self._per_crop_params.get(crop.id, self._contactsheet_profile.params)
            crop_preset = self._create_preset_from_params(params, name=crop.name)
            # 将 crop 的 orientation 写入 crop 或 preset（preset.orientation 用于 BWC）
            crop_preset.orientation = crop.orientation
            entries.append(CropPresetEntry(crop=crop, preset=crop_preset))
        active_id = self._active_crop_id if self._active_crop_id in [c.id for c in self._crops] else None
        return PresetBundle(contactsheet=cs_preset, crops=entries, active_crop_id=active_id)

    def export_single_preset(self) -> Preset:
        """导出当前图像的单预设（single）。
        使用 contactsheet 参数作为单预设的基础；包含 raw_file、orientation 与可选的 contactsheet 裁剪。
        """
        try:
            preset = self._create_preset_from_params(self._contactsheet_profile.params, name="single")
            # orientation 与裁剪（若存在）
            preset.orientation = self._contactsheet_profile.orientation
            if self._contactsheet_profile.crop_rect is not None:
                preset.crop = tuple(self._contactsheet_profile.crop_rect)
            # 写入文件名（v3 必填）
            if self._current_image and getattr(self._current_image, 'file_path', ''):
                preset.raw_file = Path(self._current_image.file_path).name
            return preset
        except Exception:
            # 回退：最小可用结构
            preset = Preset(name="single")
            if self._current_image and getattr(self._current_image, 'file_path', ''):
                preset.raw_file = Path(self._current_image.file_path).name
            preset.grading_params = self._contactsheet_profile.params.to_dict()
            return preset

    def _create_preset_from_params(self, params: ColorGradingParams, name: str = "Preset") -> Preset:
        """将当前参数打包为 Preset（用于 Bundle 导出）。"""
        preset = Preset(name=name, film_type=self._current_film_type)
        # input transformation（保存名称与参数：gamma/white/primaries）
        cs_name = params.input_color_space_name
        cs_def = self.color_space_manager.get_color_space_definition(cs_name)
        if cs_def:
            preset.input_transformation = InputTransformationDefinition(name=cs_name, definition=cs_def)
        else:
            preset.input_transformation = InputTransformationDefinition(name=cs_name, definition={})
        # grading params（从 params.to_dict 获取 UI 相关字段）
        preset.grading_params = params.to_dict()
        # density matrix（镜像冗余）
        if params.density_matrix is not None:
            preset.density_matrix = MatrixDefinition(name=params.density_matrix_name, values=params.density_matrix.tolist())
        else:
            preset.density_matrix = MatrixDefinition(name=params.density_matrix_name, values=None)
        # 曲线（若命名为 custom，直接写 points）
        preset.density_curve = CurveDefinition(name=params.density_curve_name, points=params.curve_points)
        # orientation 由上层（contactsheet/crop）决定
        return preset

    # =================
    # 核心业务逻辑 (Actions)
    # =================
    def load_image(self, file_path: str):
        try:
            self._loading_image = True  # 设置加载标志，延迟预览更新
            self.status_message_changed.emit(tr("app_context.image_loading.loading_image", file_path=file_path))

            # 1. 根据内存报告决定是轻量清理还是重度清理
            try:
                report = self.get_memory_usage_report()
                total_mb = report.get("total_estimated_mb", 0)
            except Exception:
                total_mb = 0
            print(f"[MEM] total_estimated_mb={total_mb:.1f}MB，执行 _clear_all_caches()")
            self._clear_all_caches()
            if total_mb > 2000:  # 阈值你自己定，比如 2GB
                print(f"[MEM] total_estimated_mb={total_mb:.1f}MB，执行 _clear_all_caches()")
                self._clear_all_caches()
            else:
                # 只清当前图像数据
                self._clear_current_image_data()

            # 先让 UI 清空当前预览，释放 PreviewWidget 里那一份大图像
            self.preview_clear_requested.emit()

            # 显式释放旧图像以防止内存泄漏
            if self._current_image is not None:
                if hasattr(self._current_image, 'array') and self._current_image.array is not None:
                    self._current_image.array = None  # 释放 numpy 数组
                self._current_image = None

            # 显式释放旧代理图像（关键修复：之前缺失此步骤）
            if self._current_proxy is not None:
                if hasattr(self._current_proxy, 'array') and self._current_proxy.array is not None:
                    self._current_proxy.array = None  # 释放代理的 numpy 数组（~17MB）
                self._current_proxy = None

            # 清理色卡缓存（新图片新缓存，避免旧图片数据残留）
            self._cached_reference_colors = None
            self._reference_colorspace = None
            self._reference_filename = None

            # ============ 进程隔离：智能管理 worker 进程 ============
            if self._use_process_isolation:
                # 检查是否需要重启 worker（内存监控）
                if self._should_restart_worker():
                    print("[WORKER] Restarting worker process due to memory threshold or death")
                    self._shutdown_preview_worker_process()
                # 否则 worker 会在预览时自动 reload proxy（热重载）
            # =========================================================

            self._current_image = self.image_manager.load_image(file_path)
            
            # 更新文件夹导航状态
            self.folder_navigator.update_folder(file_path)
            
            # 检测单/双通道图像，自动切换黑白模式
            if self._current_image.is_monochrome_source:
                # 根据现有胶片类型智能选择黑白类型
                if self._current_film_type == "color_reversal":
                    target_film_type = "b&w_reversal"
                else:
                    target_film_type = "b&w_negative"  # 默认选择
                
                print(f"[ApplicationContext] 检测到{self._current_image.original_channels}通道图像，自动切换为{target_film_type}")
                self.set_current_film_type(target_film_type, apply_defaults=True)
                self.status_message_changed.emit(tr("app_context.image_loading.monochrome_detected"))
            
            # 重置裁剪（orientation延迟重置，让预设系统先处理）
            self._contactsheet_profile.crop_rect = None
            self._crops = []
            self._active_crop_id = None
            self._crop_focused = False
            self.crop_changed.emit(None)
            
            # 检查并应用自动预设
            self.auto_preset_manager.set_active_directory(str(Path(file_path).parent))
            bundle = self.auto_preset_manager.get_bundle_for_image(file_path)
            if bundle:
                self.load_preset_bundle(bundle)
                self.status_message_changed.emit(tr("app_context.preset_messages.auto_preset_bundle_loaded"))
                # Bundle内部已触发预览；此处不再重复
            else:
                preset = self.auto_preset_manager.get_preset_for_image(file_path)
                if preset:
                    self.load_preset(preset)
                    self.status_message_changed.emit(tr("app_context.preset_messages.auto_preset_loaded", name=preset.name))
                    
                    # NOTE: Do not apply film type overrides when loading presets from file
                    # The preset's values should be preserved as-is
                    # self._apply_film_type_override_if_needed()
                    
                    # DEBUG信息
                    try:
                        print(f"[DEBUG] after load_preset(user): input={self._current_params.input_color_space_name}, gamma={self._current_params.density_gamma}, dmax={self._current_params.density_dmax}, rgb={self._current_params.rgb_gains}")
                    except Exception:
                        pass
                else:
                    # 如果没有预设，则使用智能分类器选择默认预设
                    try:
                        self._load_smart_default_preset(file_path)
                    except Exception:
                        # 智能分类失败时，回退到通用默认
                        try:
                            self._load_generic_default_preset()
                            self.status_message_changed.emit(tr("app_context.preset_messages.preset_not_found_default"))
                        except Exception:
                            self.reset_params()
                            self.status_message_changed.emit(tr("app_context.preset_messages.preset_not_found_fallback"))


            # 清除加载标志并触发最终预览更新
            print(f"[DEBUG] load_image(): 清除_loading_image标志，准备最终预览更新", flush=True)
            print(f"[DEBUG] load_image(): 当前参数 enable_density_curve={self._current_params.enable_density_curve}, curve_name={self._current_params.density_curve_name}", flush=True)
            print(f"[DEBUG] load_image(): 曲线点数: RGB={len(self._current_params.curve_points) if self._current_params.curve_points else 0}, R={len(self._current_params.curve_points_r) if self._current_params.curve_points_r else 0}", flush=True)
            self._loading_image = False
            if self._current_image:
                print(f"[DEBUG] load_image(): 调用_prepare_proxy()...", flush=True)
                print("[DEBUG] Call for _prepare_proxy: load_image()"); self._prepare_proxy()
                print(f"[DEBUG] load_image(): 调用_trigger_preview_update()...", flush=True)
                self._trigger_preview_update()

            # 通知UI：图像已加载完成
            self.image_loaded.emit()

        except Exception as e:
            self._loading_image = False  # 确保异常情况下也清除标志
            import traceback
            print(traceback.format_exc())
            self.status_message_changed.emit(tr("app_context.image_loading.load_image_failed", error=e))

    def load_preset(self, preset: Preset, preserve_film_type: bool = False):
        """从Preset对象加载状态 - 使用新的CropInstance模型
        
        Args:
            preset: 要加载的预设
            preserve_film_type: 是否保留当前胶片类型，不被预设覆盖
        """
        # 1. 清理当前状态
        # self.clear_crop() # 不清理裁剪，允许预设仅更新参数
        
        # Load film type (optionally preserve current film type)  
        old_film_type = self._current_film_type
        is_loading_bw_preset = False
        if not preserve_film_type:
            is_loading_bw_preset = self.film_type_controller.is_monochrome_type(preset.film_type)
            # Set film type but do NOT apply defaults when loading a preset
            # The preset's values should take precedence
            self.set_current_film_type(preset.film_type, apply_defaults=False)
        else:
            # Keep current film type - don't change it
            pass
        
        # 2/3. 合并更新参数：先从 grading_params 构造，再应用 input_transformation（若有）
        new_params = ColorGradingParams.from_dict(preset.grading_params or {})
        if preset.input_transformation and preset.input_transformation.name:
            # 同步输入色彩空间名称
            new_params.input_color_space_name = preset.input_transformation.name
            # 将预设中的 idt 数据写入 ColorSpaceManager（内存覆盖，不持久化）
            try:
                cs_def = preset.input_transformation.definition or {}
                cs_name = preset.input_transformation.name
                
                # 检查是否包含完整的色彩空间定义
                if 'primitives' in cs_def and 'white' in cs_def:
                    # 检查是否已存在预定义色彩空间，避免覆盖丢失type等属性
                    existing_info = self.color_space_manager.get_color_space_info(cs_name)
                    if existing_info and existing_info.get('type'):
                        # 如果是预定义色彩空间，只更新gamma，不覆盖整个定义
                        print(f"Debug: 跳过注册预定义色彩空间 {cs_name}，仅更新gamma")
                        gamma = cs_def.get('gamma', 1.0)
                        self.color_space_manager.update_color_space_gamma(cs_name, gamma)
                    else:
                        # 转换格式：从preset格式转换为register_custom_colorspace所需格式
                        primitives = cs_def['primitives']  # {"r": {"x": ..., "y": ...}, ...}
                        white = cs_def['white']           # {"x": ..., "y": ...}
                        gamma = cs_def.get('gamma', 1.0)
                        
                        # 转换为numpy数组格式
                        primaries_xy = np.array([
                            [primitives['r']['x'], primitives['r']['y']],
                            [primitives['g']['x'], primitives['g']['y']], 
                            [primitives['b']['x'], primitives['b']['y']]
                        ])
                        white_point_xy = np.array([white['x'], white['y']])
                        
                        # 注册完整的自定义色彩空间（仅对真正的自定义色彩空间）
                        print(f"Debug: 注册自定义色彩空间 {cs_name}: primaries={primaries_xy.tolist()}, white={white_point_xy.tolist()}, gamma={gamma}")
                        self.color_space_manager.register_custom_colorspace(
                            name=cs_name,
                            primaries_xy=primaries_xy,
                            white_point_xy=white_point_xy,
                            gamma=gamma
                        )
                elif 'gamma' in cs_def:
                    # 只有gamma，仅更新gamma
                    self.color_space_manager.update_color_space_gamma(cs_name, float(cs_def['gamma']))
            except Exception as e:
                print(f"处理input_transformation失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 兼容处理矩阵和曲线
        if preset.density_matrix:
            new_params.density_matrix_name = preset.density_matrix.name
            if preset.density_matrix.values:
                new_params.density_matrix = np.array(preset.density_matrix.values)
                # 预设显式包含矩阵数值时，自动启用密度矩阵
                new_params.enable_density_matrix = True
            elif preset.density_matrix.name:
                matrix = self.the_enlarger.pipeline_processor.get_density_matrix_array(preset.density_matrix.name)
                new_params.density_matrix = matrix
                # 预设指定了矩阵名称（且可解析）时，自动启用密度矩阵
                if matrix is not None:
                    new_params.enable_density_matrix = True

        if preset.density_curve:
            # 设置曲线名称
            new_params.density_curve_name = preset.density_curve.name
            print(f"[DEBUG] load_preset(): 设置密度曲线名称={preset.density_curve.name}, enable_density_curve={new_params.enable_density_curve}", flush=True)
            print(f"[DEBUG] load_preset(): 预设中的曲线点数: RGB={len(new_params.curve_points) if new_params.curve_points else 0}, R={len(new_params.curve_points_r) if new_params.curve_points_r else 0}", flush=True)

            # 如果曲线名称不是custom，且预设中没有完整的曲线数据，则从曲线文件重新加载
            # 这是因为预设文件通常只保存曲线名称，不保存完整的曲线点数据
            if preset.density_curve.name and preset.density_curve.name != "custom":
                # 检查是否需要重新加载（R通道只有2个点说明没有真实数据）
                needs_reload = (not new_params.curve_points_r or len(new_params.curve_points_r) <= 2)
                if needs_reload:
                    print(f"[DEBUG] load_preset(): 预设中缺少完整曲线数据，从曲线文件重新加载...", flush=True)
                    curve_data = self._load_density_curve_points_by_name(preset.density_curve.name)
                    if curve_data:
                        # 加载所有通道的曲线点
                        if 'RGB' in curve_data:
                            new_params.curve_points = curve_data['RGB']
                        if 'R' in curve_data:
                            new_params.curve_points_r = curve_data['R']
                        if 'G' in curve_data:
                            new_params.curve_points_g = curve_data['G']
                        if 'B' in curve_data:
                            new_params.curve_points_b = curve_data['B']
                        print(f"[DEBUG] load_preset(): 从曲线文件重新加载完成，曲线点数: RGB={len(new_params.curve_points)}, R={len(new_params.curve_points_r)}", flush=True)
                    else:
                        print(f"[WARNING] load_preset(): 无法从曲线文件加载曲线 '{preset.density_curve.name}'", flush=True)

            # 如果预设中包含了曲线启用标志，from_dict已经设置了
            # 这里不需要额外处理

        # 在更新参数前先设置orientation，避免预览闪烁
        try:
            self._contactsheet_profile.orientation = preset.orientation
        except Exception:
            pass

        # 加载预设只是参数修改，不需要prepare_proxy（worker会处理）

        self.update_params(new_params)
        self._contactsheet_profile.params = self._current_params.shallow_copy()  # 优化：只读保存，使用 shallow_copy()

        # 4. 加载crop和orientation（完全分离模型）
        try:

            # 再加载crop（使用新的CropInstance接口）
            crop_instances = preset.get_crop_instances()
            if crop_instances:
                # 当前阶段：仅支持单crop，取第一个
                crop_instance = crop_instances[0]
                # 保持crop的独立orientation
                self.set_single_crop(crop_instance.rect_norm, crop_instance.orientation, preserve_focus=False)
                self._crop_focused = False
            elif preset.crop and len(preset.crop) == 4:
                # 回退：旧字段作为 contactsheet 临时裁剪
                try:
                    x, y, w, h = [float(max(0.0, min(1.0, v))) for v in tuple(preset.crop)]
                    w = max(0.0, min(1.0 - x, w)); h = max(0.0, min(1.0 - y, h))
                    self._contactsheet_profile.crop_rect = (x, y, w, h)
                    self._crop_focused = False
                    self.crop_changed.emit(self._contactsheet_profile.crop_rect)
                except Exception:
                    self._contactsheet_profile.crop_rect = None
        except Exception:
            # orientation 已在前面设置，这里无需重复
            pass
        
        # 切换到 contactsheet profile
        # self._current_profile_kind = 'contactsheet'
        
        # NOTE: Do not apply film type overrides when loading presets
        # The preset's values should be preserved as-is
        # self._apply_film_type_override_if_needed()
    
    def _apply_film_type_override_if_needed(self):
        """Apply film type hierarchical override system"""
        # This triggers the UI-based override system that was implemented in MainWindow
        # The signal will be caught by MainWindow.on_context_film_type_changed
        # which will then apply the B&W neutralization if needed
        if hasattr(self, 'film_type_changed'):
            self.film_type_changed.emit(self._current_film_type)

    # === 新增：Bundle 加载/保存 API ===
    def load_preset_bundle(self, bundle: PresetBundle):
        """加载预设集合：设置 contactsheet 与 per-crop 参数集，并默认切换到 contactsheet。"""
        try:
            # 加载 contactsheet preset
            self.load_preset(bundle.contactsheet)
            # 构建 per-crop 参数与 crop list
            self._per_crop_params.clear()
            self._crops = []
            self._active_crop_id = None
            for entry in bundle.crops:
                crop = entry.crop
                # 保证 id 存在
                cid = crop.id or f"crop_{len(self._crops)+1}"
                crop.id = cid
                self._crops.append(crop)
                # 解析该裁剪的参数
                params = ColorGradingParams.from_dict(entry.preset.grading_params or {})
                # 同步 input colorspace 与显式矩阵等
                if entry.preset.input_transformation and entry.preset.input_transformation.name:
                    params.input_color_space_name = entry.preset.input_transformation.name
                    # 将 per-crop 预设中的 idt 数据写入 ColorSpaceManager（内存覆盖）
                    try:
                        cs_def = entry.preset.input_transformation.definition or {}
                        cs_name = entry.preset.input_transformation.name
                        
                        # 检查是否包含完整的色彩空间定义
                        if 'primitives' in cs_def and 'white' in cs_def:
                            # 检查是否已存在预定义色彩空间，避免覆盖丢失type等属性
                            existing_info = self.color_space_manager.get_color_space_info(cs_name)
                            if existing_info and existing_info.get('type'):
                                # 如果是预定义色彩空间，只更新gamma，不覆盖整个定义
                                print(f"Debug: 跳过注册预定义色彩空间 {cs_name} (crop)，仅更新gamma")
                                gamma = cs_def.get('gamma', 1.0)
                                self.color_space_manager.update_color_space_gamma(cs_name, gamma)
                            else:
                                # 转换格式：从preset格式转换为register_custom_colorspace所需格式
                                primitives = cs_def['primitives']  # {"r": {"x": ..., "y": ...}, ...}
                                white = cs_def['white']           # {"x": ..., "y": ...}
                                gamma = cs_def.get('gamma', 1.0)
                                
                                # 转换为numpy数组格式
                                primaries_xy = np.array([
                                    [primitives['r']['x'], primitives['r']['y']],
                                    [primitives['g']['x'], primitives['g']['y']], 
                                    [primitives['b']['x'], primitives['b']['y']]
                                ])
                                white_point_xy = np.array([white['x'], white['y']])
                                
                                # 注册完整的自定义色彩空间（仅对真正的自定义色彩空间）
                                print(f"Debug: 注册自定义色彩空间 {cs_name} (crop): primaries={primaries_xy.tolist()}, white={white_point_xy.tolist()}, gamma={gamma}")
                                self.color_space_manager.register_custom_colorspace(
                                    name=cs_name,
                                    primaries_xy=primaries_xy,
                                    white_point_xy=white_point_xy,
                                    gamma=gamma
                                )
                        elif 'gamma' in cs_def:
                            # 只有gamma，仅更新gamma
                            self.color_space_manager.update_color_space_gamma(cs_name, float(cs_def['gamma']))
                    except Exception as e:
                        print(f"处理per-crop input_transformation失败: {e}")
                        import traceback
                        traceback.print_exc()
                if entry.preset.density_matrix:
                    params.density_matrix_name = entry.preset.density_matrix.name
                    if entry.preset.density_matrix.values:
                        params.density_matrix = np.array(entry.preset.density_matrix.values)
                        params.enable_density_matrix = True
                    elif entry.preset.density_matrix.name:
                        matrix = self.the_enlarger.pipeline_processor.get_density_matrix_array(entry.preset.density_matrix.name)
                        params.density_matrix = matrix
                        if matrix is not None:
                            params.enable_density_matrix = True
                if entry.preset.density_curve:
                    # 只设置曲线名称，所有曲线数据已经通过 from_dict 正确加载
                    # 原则：预设中有什么就加载什么，不做任何额外判断或按名称重新加载
                    params.density_curve_name = entry.preset.density_curve.name

                self._per_crop_params[cid] = params
            # 活跃裁剪（仅记录，不自动聚焦）
            self._active_crop_id = bundle.active_crop_id
            self._crop_focused = False
            self._current_profile_kind = 'contactsheet'
            # 加载bundle不需要prepare_proxy，调用方（如load_image）会处理
        except Exception as e:
            print(f"加载Bundle失败: {e}")

    # === 名称解析辅助：按名字加载曲线点 ===
    def _load_density_curve_points_by_name(self, curve_name: str):
        """根据曲线名称从配置文件加载曲线点。
        返回 dict: {'RGB': [...], 'R': [...], 'G': [...], 'B': [...]} 或 None。
        """
        if not curve_name:
            return None
        
        # 处理修改状态的曲线名（带*前缀）
        original_curve_name = curve_name
        if curve_name.startswith('*'):
            original_curve_name = curve_name[1:]  # 去掉*前缀，使用原始名称查找
        try:
            from divere.utils.enhanced_config_manager import enhanced_config_manager
            def _norm(s: str) -> str:
                return " ".join(str(s).strip().lower().replace('_', ' ').split())

            target = _norm(original_curve_name)
            for json_path in enhanced_config_manager.get_config_files("curves"):
                try:
                    data = enhanced_config_manager.load_config_file(json_path)
                    if data is None:
                        continue
                    name_in_file = data.get("name") or json_path.stem
                    if _norm(name_in_file) != target and _norm(json_path.stem) != target:
                        continue
                    # 统一输出结构
                    result = { 'RGB': None, 'R': None, 'G': None, 'B': None }
                    if isinstance(data.get("curves"), dict):
                        curves = data["curves"]
                        # 兼容键名
                        if "RGB" in curves:
                            result['RGB'] = curves.get('RGB')
                        result['R'] = curves.get('R')
                        result['G'] = curves.get('G')
                        result['B'] = curves.get('B')
                    elif isinstance(data.get("points"), list):
                        result['RGB'] = data.get('points')
                    # 若至少有一条曲线，返回
                    if any(result.values()):
                        # 规范化为 float tuple 列表
                        def _normalize(lst):
                            if not lst:
                                return None
                            out = []
                            for p in lst:
                                if isinstance(p, (list, tuple)) and len(p) >= 2:
                                    out.append((float(p[0]), float(p[1])))
                            return out if out else None
                        return {
                            'RGB': _normalize(result['RGB']),
                            'R': _normalize(result['R']),
                            'G': _normalize(result['G']),
                            'B': _normalize(result['B']),
                        }
                except Exception:
                    continue
        except Exception:
            return None
        return None
        
    def update_params(self, new_params: ColorGradingParams):
        """由UI调用以更新参数"""
        print(f"[DEBUG] update_params(): enable_density_curve={new_params.enable_density_curve}, curve_name={new_params.density_curve_name}", flush=True)
        print(f"[DEBUG] update_params(): 曲线点数: RGB={len(new_params.curve_points) if new_params.curve_points else 0}, R={len(new_params.curve_points_r) if new_params.curve_points_r else 0}", flush=True)
        self._current_params = new_params
        # 同步到当前 profile 存根
        try:
            if self._current_profile_kind == 'contactsheet':
                self._contactsheet_profile.params = new_params.shallow_copy()  # 优化：只读保存，使用 shallow_copy()
            elif self._current_profile_kind == 'crop' and self._active_crop_id is not None:
                self._per_crop_params[self._active_crop_id] = new_params.shallow_copy()  # 优化：只读保存，使用 shallow_copy()
        except Exception:
            pass
        self.params_changed.emit(self._current_params)
        self._trigger_preview_update()
        self._autosave_timer.start() # 每次参数变更都启动自动保存计时器
    
    def set_input_color_space(self, space_name: str):
        self._current_params.input_color_space_name = space_name
        self.params_changed.emit(self._current_params)
        if self._current_image:
            #self._prepare_proxy()
            self._trigger_preview_update()
    
    def set_current_film_type(self, film_type: str, apply_defaults: bool = True, 
                             force_apply_defaults: bool = False):
        """
        设置当前胶片类型并可选地应用对应配置
        
        Args:
            film_type: 要设置的胶片类型
            apply_defaults: 是否应用胶片类型的默认值
            force_apply_defaults: 是否强制应用默认值（覆盖现有值）
        """
        old_film_type = self._current_film_type
        self._current_film_type = film_type
        
        # Emit signal if film type changed
        if old_film_type != film_type:
            self.film_type_changed.emit(film_type)
            
            # 对于黑白类型，强制延迟触发UI状态更新以确保正确禁用控件
            if self.is_monochrome_type():
                from PySide6.QtCore import QTimer
                # 直接传递方法引用，避免 lambda 闭包内存泄漏
                QTimer.singleShot(0, self._ensure_ui_state_sync_for_monochrome)
        
        # 根据参数决定是否应用胶片类型的默认配置
        if apply_defaults:
            self._current_params = self.film_type_controller.apply_film_type_defaults(
                self._current_params, film_type, force_apply=force_apply_defaults
            )
        
        # 触发参数更新和预览刷新
        self.params_changed.emit(self._current_params)
        if self._current_image and not getattr(self, '_loading_image', False):
            print("[DEBUG] Trigger for preview_update: set_current_film_type()"); self._trigger_preview_update()  # 只是参数修改，不需要prepare_proxy
    
    def convert_to_black_and_white_mode(self, show_dialog: bool = True):
        """
        Convert the current photo to black & white mode.
        This should only be called when explicitly converting a photo to B&W mode,
        not when loading an existing B&W preset.
        
        Args:
            show_dialog: Whether to show confirmation dialog (handled by UI layer)
        """
        if self.is_monochrome_type():
            return  # Already in B&W mode
            
        # Determine the appropriate B&W film type based on current type
        if self._current_film_type == "color_reversal":
            new_film_type = "b&w_reversal"
        else:
            new_film_type = "b&w_negative"
        
        # Set the film type without applying defaults (preserves IDT parameters)
        self.set_current_film_type(new_film_type, apply_defaults=False)
        
        # Apply only the B&W-specific changes (RGB gains, curve, pipeline settings)
        # This preserves IDT gamma, dmax, and other user settings
        self._current_params = self.film_type_controller.apply_black_and_white_conversion(
            self._current_params, new_film_type
        )
        
        # Update the UI and trigger preview refresh
        self.params_changed.emit(self._current_params)
        if self._current_image:
            print("[DEBUG] Trigger for preview_update: convert_to_black_and_white_mode()"); self._trigger_preview_update()  # 只是参数修改，不需要prepare_proxy

        self.status_message_changed.emit(tr("app_context.image_loading.converted_to_bw", film_type=self.film_type_controller.get_film_type_display_name(new_film_type)))
    
    def get_current_film_type(self) -> str:
        """获取当前胶片类型"""
        return self._current_film_type
    
    def get_pipeline_config(self) -> 'PipelineConfig':
        """获取当前胶片类型的pipeline配置"""
        return self.film_type_controller.get_pipeline_config(self._current_film_type)
    
    def get_ui_state_config(self) -> 'UIStateConfig':
        """获取当前胶片类型的UI状态配置"""
        return self.film_type_controller.get_ui_state_config(self._current_film_type)
    
    def should_convert_to_monochrome(self) -> bool:
        """判断当前胶片类型是否需要转换为monochrome"""
        return self.film_type_controller.should_convert_to_monochrome(self._current_film_type)
    
    def is_monochrome_type(self) -> bool:
        """判断当前是否为黑白胶片类型"""
        return self.film_type_controller.is_monochrome_type(self._current_film_type)
    
    def load_film_type_default_preset(self, film_type: Optional[str] = None):
        """加载指定胶片类型的默认预设"""
        if film_type is None:
            film_type = self._current_film_type
            
        try:
            from divere.utils.defaults import load_film_type_default_preset
            preset = load_film_type_default_preset(film_type)
            if preset:
                self.load_preset(preset)
                self.status_message_changed.emit(tr("app_context.preset_messages.film_type_preset_loaded", film_type=film_type))
            else:
                raise ValueError(f"无法加载胶片类型 '{film_type}' 的默认预设")
        except Exception as e:
            self.status_message_changed.emit(tr("app_context.preset_messages.film_type_preset_failed", error=e))
            # 回退到通用默认预设
            try:
                from divere.utils.defaults import load_default_preset
                self.load_preset(load_default_preset())
                self.status_message_changed.emit(tr("app_context.preset_messages.fallback_to_default"))
            except Exception as fallback_error:
                self.status_message_changed.emit(tr("app_context.preset_messages.load_default_failed", error=fallback_error))

    def reset_params(self):
        """重置参数：根据当前图像类型选择智能默认预设"""
        # 保存当前orientation，避免被预设重置
        saved_orientation = self._contactsheet_profile.orientation
        
        if self._current_image:
            # 有图像时，使用智能分类器选择默认预设
            try:
                self._load_smart_default_preset(self._current_image.file_path)
                self.status_message_changed.emit(tr("app_context.preset_messages.reset_to_auto_classification"))
            except Exception:
                # 智能分类失败时，回退到通用默认
                try:
                    from divere.utils.defaults import load_default_preset
                    self.load_preset(load_default_preset())
                    self.status_message_changed.emit(tr("app_context.preset_messages.reset_to_default"))
                except Exception:
                    self._current_params = self._create_default_params()
                    self._contactsheet_profile.params = self._current_params.shallow_copy()  # 优化：只读保存，使用 shallow_copy()
                    self.params_changed.emit(self._current_params)
                    self.status_message_changed.emit(tr("app_context.preset_messages.reset_to_internal"))
        else:
            # 没有图像时，使用通用默认
            try:
                self._load_generic_default_preset()
                self.status_message_changed.emit(tr("app_context.preset_messages.reset_to_default"))
            except Exception:
                self._current_params = self._create_default_params()
                self._contactsheet_profile.params = self._current_params.shallow_copy()  # 优化：只读保存，使用 shallow_copy()
                self.params_changed.emit(self._current_params)
                self.status_message_changed.emit(tr("app_context.preset_messages.reset_to_internal"))
        
        # 恢复保存的orientation
        self._contactsheet_profile.orientation = saved_orientation

        # 重新更新预览（只是参数修改，不需要prepare_proxy）
        if self._current_image:
            self._trigger_preview_update()

    def set_current_as_folder_default(self):
        """将当前参数设置为文件夹默认设置"""
        if not self._current_image:
            self.status_message_changed.emit(tr("app_context.image_loading.please_load_image"))
            return
        
        try:
            from pathlib import Path
            
            # 获取当前图像的目录
            image_path = Path(self._current_image.file_path)
            
            # 获取当前色彩空间的完整信息
            cs_info = self.color_space_manager.get_color_space_info(self._current_params.input_color_space_name)
            if not cs_info:
                cs_info = {"gamma": 1.0}
            
            # 提取当前的idt数据
            idt_data = {
                "name": self._current_params.input_color_space_name,
                "gamma": cs_info.get("gamma", 1.0),
            }
            
            # 添加白点和基色信息（如果存在）
            if "white_point" in cs_info:
                white_point = cs_info["white_point"]
                if isinstance(white_point, np.ndarray):
                    idt_data["white"] = {"x": float(white_point[0]), "y": float(white_point[1])}

            if "primaries" in cs_info:
                primaries = cs_info["primaries"]
                if isinstance(primaries, np.ndarray):
                    idt_data["primitives"] = {
                        "r": {"x": float(primaries[0][0]), "y": float(primaries[0][1])},
                        "g": {"x": float(primaries[1][0]), "y": float(primaries[1][1])},
                        "b": {"x": float(primaries[2][0]), "y": float(primaries[2][1])}
                    }
            
            # 提取当前的cc_params数据
            cc_params_data = {
                "density_gamma": self._current_params.density_gamma,
                "density_dmax": self._current_params.density_dmax,
                "rgb_gains": list(self._current_params.rgb_gains),
                "density_matrix": {
                    "name": self._current_params.density_matrix_name,
                    "values": self._current_params.density_matrix.tolist() if self._current_params.density_matrix is not None else None
                },
                "density_curve": {
                    "name": self._current_params.density_curve_name,
                    "points": {
                        "rgb": self._current_params.curve_points,
                        "r": self._current_params.curve_points_r,
                        "g": self._current_params.curve_points_g,
                        "b": self._current_params.curve_points_b
                    }
                },
                "screen_glare_compensation": self._current_params.screen_glare_compensation,
                # 保存 channel_gamma 参数
                "channel_gamma": {
                    "r": self._current_params.channel_gamma_r,
                    "b": self._current_params.channel_gamma_b
                },
                # 保存pipeline控制状态
                "enable_density_matrix": self._current_params.enable_density_matrix,
                "enable_density_curve": self._current_params.enable_density_curve,
                "enable_rgb_gains": self._current_params.enable_rgb_gains,
                "enable_density_inversion": self._current_params.enable_density_inversion
            }
            
            # 通过AutoPresetManager保存folder_default
            self.auto_preset_manager.set_active_directory(str(image_path.parent))
            self.auto_preset_manager.save_folder_default(idt_data, cc_params_data)

            self.status_message_changed.emit(tr("app_context.preset_messages.folder_default_saved"))

        except Exception as e:
            self.status_message_changed.emit(tr("app_context.preset_messages.folder_default_save_failed", error=e))

    def run_auto_color_correction(self, get_preview_callback):
        """执行AI自动白平衡"""
        # 黑白模式下跳过RGB gains调整
        if self.is_monochrome_type():
            pipeline_config = self.film_type_controller.get_pipeline_config(self._current_film_type)
            if not pipeline_config.enable_rgb_gains:
                self.status_message_changed.emit(tr("app_context.auto_color.bw_mode_disabled"))
                return

        preview_image = get_preview_callback()
        if preview_image is None or preview_image.array is None:
            self.status_message_changed.emit(tr("app_context.auto_color.no_preview_image"))
            return

        try:
            gains_t = self.the_enlarger.calculate_auto_gain_learning_based(preview_image)
            gains = np.array(gains_t[:3])
            # rgb gain的调整量与gamma是耦合的：最终的增量 = 原始建议增量 × (gamma / 2)
            gamma = float(self._current_params.density_gamma)
            scale = gamma / 2.0
            delta = gains * scale
            
            current_gains = np.array(self._current_params.rgb_gains)
            new_gains = np.clip(current_gains + delta, -2.0, 2.0)
            # 低侵入：改用统一入口，确保写入当前 profile 存根并触发 autosave
            new_params = self._current_params.shallow_copy()  # 优化：仅修改 rgb_gains (tuple)，使用 shallow_copy()
            new_params.rgb_gains = tuple(new_gains)
            self.update_params(new_params)
            
            # 根据图像类型显示不同的消息
            if self.is_monochrome_type():
                # 黑白图像：只显示灰度增益
                self.status_message_changed.emit(
                    tr("app_context.auto_color.complete_mono", gray=delta[0])
                )
            else:
                # 彩色图像：显示RGB增益
                self.status_message_changed.emit(
                    tr("app_context.auto_color.complete_color", r=delta[0], g=delta[1], b=delta[2])
                )
        except Exception as e:
            self.status_message_changed.emit(tr("app_context.auto_color.failed", error=e))

    def run_iterative_auto_color(self, get_preview_callback, max_iterations=10):
        """执行迭代式AI自动白平衡"""
        # 黑白模式下跳过RGB gains调整
        if self.is_monochrome_type():
            pipeline_config = self.film_type_controller.get_pipeline_config(self._current_film_type)
            if not pipeline_config.enable_rgb_gains:
                self.status_message_changed.emit(tr("app_context.auto_color.iterative_disabled"))
                return
        
        self._auto_color_iterations = max_iterations
        self._get_preview_for_auto_color_callback = get_preview_callback
        self._perform_auto_color_iteration() # Start the first iteration

    def _perform_auto_color_iteration(self):
        if self._auto_color_iterations <= 0 or not self._get_preview_for_auto_color_callback:
            self._get_preview_for_auto_color_callback = None # Clean up
            return
        
        # 黑白模式下中止迭代
        if self.is_monochrome_type():
            pipeline_config = self.film_type_controller.get_pipeline_config(self._current_film_type)
            if not pipeline_config.enable_rgb_gains:
                self.status_message_changed.emit(tr("app_context.auto_color.iterative_stopped"))
                self._auto_color_iterations = 0
                self._get_preview_for_auto_color_callback = None
                return

        preview_image = self._get_preview_for_auto_color_callback()
        if preview_image is None or preview_image.array is None:
            self.status_message_changed.emit(tr("app_context.auto_color.iterative_no_preview"))
            self._get_preview_for_auto_color_callback = None # Clean up
            return

        try:
            gains_t = self.the_enlarger.calculate_auto_gain_learning_based(preview_image)
            gains = np.array(gains_t[:3])
            gamma = float(self._current_params.density_gamma)
            scale = gamma / 2.0
            delta = gains * scale

            current_gains = np.array(self._current_params.rgb_gains)
            new_gains = np.clip(current_gains + delta, -3.0, 3.0)
            
            self._auto_color_iterations -= 1
            
            # If gains are very small, stop iterating
            if np.allclose(current_gains, new_gains, atol=1e-3):
                self.status_message_changed.emit(tr("app_context.auto_color.converged"))
                self._auto_color_iterations = 0
                self._get_preview_for_auto_color_callback = None
                self._autosave_timer.start() # Save on convergence
                return

            # 低侵入：改用统一入口，确保写入当前 profile 存根并触发 autosave
            new_params = self._current_params.shallow_copy()  # 优化：仅修改 rgb_gains (tuple)，使用 shallow_copy()
            new_params.rgb_gains = tuple(new_gains)
            self.update_params(new_params)  # 将触发预览；_on_preview_result 会调度下一次迭代
            self.status_message_changed.emit(tr("app_context.auto_color.iterations_remaining", count=self._auto_color_iterations))

        except Exception as e:
            self.status_message_changed.emit(tr("app_context.auto_color.iterative_failed", error=e))
            self._auto_color_iterations = 0
            self._get_preview_for_auto_color_callback = None

    def _sample_preview_region(self, preview_image: ImageData, norm_x: float, norm_y: float, sample_size: int) -> Tuple[float, float, float]:
        """
        从preview图像采样指定区域的RGB均值

        Args:
            preview_image: Preview图像（DisplayP3空间）
            norm_x: 归一化X坐标 (0-1)
            norm_y: 归一化Y坐标 (0-1)
            sample_size: 采样区域大小（n×n像素）

        Returns:
            (r_mean, g_mean, b_mean): RGB均值
        """
        img_array = preview_image.array
        height, width = img_array.shape[:2]

        # 转换为像素坐标
        x_pixel = int(norm_x * width)
        y_pixel = int(norm_y * height)

        # 计算采样区域
        half_size = sample_size // 2
        x_start = max(0, x_pixel - half_size)
        x_end = min(width, x_pixel + half_size + 1)
        y_start = max(0, y_pixel - half_size)
        y_end = min(height, y_pixel + half_size + 1)

        # 提取区域并计算均值
        region = img_array[y_start:y_end, x_start:x_end, :]
        r_mean = float(np.mean(region[:, :, 0]))
        g_mean = float(np.mean(region[:, :, 1]))
        b_mean = float(np.mean(region[:, :, 2]))

        return r_mean, g_mean, b_mean

    def _perform_neutral_point_iteration(self):
        """执行一次中性点迭代（从DisplayP3 preview采样并调整gains）"""

        # 检查是否应该停止
        if self._neutral_point_iterations <= 0 or not self._neutral_point_callback:
            self._neutral_point_callback = None
            self._neutral_point_norm = None
            self.status_message_changed.emit(tr("app_context.neutral_color.complete"))
            self._autosave_timer.start()
            return

        # 获取当前preview图像（DisplayP3空间）
        preview_image = self._neutral_point_callback()
        if preview_image is None or preview_image.array is None:
            self.status_message_changed.emit(tr("app_context.neutral_color.no_preview"))
            self._neutral_point_callback = None
            self._neutral_point_norm = None
            return

        try:
            # 使用保存的色温值
            white_point = self._neutral_point_white_point

            # 将Kelvin色温转换为Display P3空间的RGB比值
            # Step 1: Kelvin → xy色度坐标 (CIE 1931)
            xy = CCT_to_xy_CIE_D(white_point+1000) # 以5500K为白点

            # Step 2: xy → XYZ (Y=1)
            xyz = color_science.xy_to_XYZ_unitY(xy)

            # Step 3: XYZ → Display P3 Linear RGB
            # 注意：Display P3的白点是D65 (6500K)
            # 我们不做色适应，因为我们要的就是white_point在Display P3中的实际RGB表示
            target_rgb = color_science.xyz_to_display_p3_linear_rgb(xyz)
            # Step 3.5: 应用 Display P3 的 gamma 编码（2.2）                                                                                                                                                             │ │
            # 因为preview图像已经过gamma编码，我们需要将线性target_rgb也编码                                                                                                                                             │ │
            display_p3_gamma = 2.2
            target_rgb = np.power(np.clip(target_rgb, 0, 1), 1.0 / display_p3_gamma)

            # Step 4: 归一化为比值（以G通道为基准）
            # 避免除以零
            if target_rgb[1] < 1e-10:
                target_rgb[1] = 1e-10
            target_r_ratio = target_rgb[0] / target_rgb[1]
            target_g_ratio = 1.0
            target_b_ratio = target_rgb[2] / target_rgb[1]

            # 从DisplayP3 preview采样5x5区域
            norm_x, norm_y = self._neutral_point_norm
            r_mean, g_mean, b_mean = self._sample_preview_region(
                preview_image, norm_x, norm_y, self._neutral_point_sample_size
            )

            # 计算当前采样点的RGB比值（以G通道为基准）
            # 避免除以零
            if g_mean < 1e-10:
                g_mean = 1e-10
            current_r_ratio = r_mean / g_mean
            current_g_ratio = 1.0
            current_b_ratio = b_mean / g_mean

            # 检查收敛（比较RGB比值差异）
            r_ratio_diff = abs(current_r_ratio - target_r_ratio)
            b_ratio_diff = abs(current_b_ratio - target_b_ratio)
            max_ratio_diff = max(r_ratio_diff, b_ratio_diff)

            if max_ratio_diff < 0.001:  # 比值差异阈值0.001
                self.status_message_changed.emit(tr("app_context.neutral_color.converged", diff=max_ratio_diff))
                self._neutral_point_iterations = 0
                self._neutral_point_callback = None
                self._neutral_point_norm = None
                self._autosave_timer.start()
                return

            # 计算调整量（梯度下降，基于比值误差）
            learning_rate = 0.7*g_mean
            r_error = current_r_ratio - target_r_ratio
            b_error = current_b_ratio - target_b_ratio

            current_gains = np.array(self._current_params.rgb_gains)
            new_r_gain = np.clip(current_gains[0] - learning_rate * r_error, -3.0, 3.0)
            new_b_gain = np.clip(current_gains[2] - learning_rate * b_error, -3.0, 3.0)

            # 更新参数并触发预览（会自动调用下一次迭代）
            new_params = self._current_params.shallow_copy()  # 优化：仅修改 rgb_gains (tuple)，使用 shallow_copy()
            new_params.rgb_gains = (new_r_gain, current_gains[1], new_b_gain)

            self._neutral_point_iterations -= 1
            self.update_params(new_params)
            self.status_message_changed.emit(
                tr("app_context.neutral_color.iterating", count=self._neutral_point_iterations, white_point=white_point, diff=max_ratio_diff)
            )

        except Exception as e:
            self.status_message_changed.emit(tr("app_context.neutral_color.failed", error=e))
            self._neutral_point_iterations = 0
            self._neutral_point_callback = None
            self._neutral_point_norm = None

    def calculate_neutral_point_auto_gain(self, norm_x: float, norm_y: float, get_preview_callback, white_point: int = 5500):
        """
        启动中性点自动增益迭代（通过preview更新循环）

        Args:
            norm_x: 选择点的归一化X坐标 (0-1)
            norm_y: 选择点的归一化Y坐标 (0-1)
            get_preview_callback: 获取预览图像的回调函数
            white_point: 中性色的色温 (Kelvin), 默认5500K
        """
        # 黑白模式下跳过RGB gains调整
        if self.is_monochrome_type():
            pipeline_config = self.film_type_controller.get_pipeline_config(self._current_film_type)
            if not pipeline_config.enable_rgb_gains:
                self.status_message_changed.emit(tr("app_context.neutral_color.rgb_adjustment_disabled"))
                return

        # 初始化迭代状态
        self._neutral_point_iterations = 8  # 最多8次
        self._neutral_point_norm = (norm_x, norm_y)
        self._neutral_point_callback = get_preview_callback
        self._neutral_point_sample_size = 5
        self._neutral_point_white_point = white_point  # 保存色温参数

        self.status_message_changed.emit(tr("app_context.neutral_color.start_iteration", white_point=white_point))

        # 开始第一次迭代
        self._perform_neutral_point_iteration()

    def _prepare_proxy(self):
        """准备proxy图像：根据模式生成适当质量的proxy

        两种模式：
        1. Contactsheet模式：生成完整的downsampled原图
        2. Crop focused模式：先crop再downsample（保证质量）

        这样做的好处：
        - Crop模式下长边始终保持proxy_max_size（1500px），质量好
        - 其他变换（IDT gamma、color transform、rotate）在worker中处理
        - 只有模式切换和crop rect调整需要重建worker

        重建时机：
        - 图片切换：需要重建
        - 模式切换（contactsheet ↔ crop focused）：需要重建
        - Crop rect调整：需要重建
        - IDT/Rotate/Color修改：不需要重建（在worker中处理）
        """
        print("[DEBUG] _prepare_proxy() 开始执行", flush=True)

        if not self._current_image:
            print("[WARNING] _prepare_proxy(): 没有当前图像，提前返回", flush=True)
            return

        # 源图
        src_image = self._current_image
        orig_h, orig_w = src_image.height, src_image.width
        print(f"[DEBUG] _prepare_proxy(): 源图尺寸={orig_w}x{orig_h}, crop_focused={self._crop_focused}", flush=True)

        # === 模式判断：是否需要预先crop ===
        if self._crop_focused:
            print("[DEBUG] _prepare_proxy(): Crop focused模式，准备crop后的proxy", flush=True)
            # Crop focused模式：先crop再downsample（保证质量）
            crop_instance = self.get_active_crop_instance()
            if crop_instance and crop_instance.rect_norm and src_image.array is not None:
                print("[DEBUG] _prepare_proxy(): 多裁剪聚焦模式", flush=True)
                try:
                    x, y, w, h = crop_instance.rect_norm
                    x0 = int(round(x * orig_w))
                    y0 = int(round(y * orig_h))
                    x1 = int(round((x + w) * orig_w))
                    y1 = int(round((y + h) * orig_h))
                    x0 = max(0, min(orig_w - 1, x0))
                    x1 = max(x0 + 1, min(orig_w, x1))
                    y0 = max(0, min(orig_h - 1, y0))
                    y1 = max(y0 + 1, min(orig_h, y1))
                    cropped_arr = src_image.array[y0:y1, x0:x1, :].copy()
                    src_image = src_image.copy_with_new_array(cropped_arr)
                    print(f"[DEBUG] _prepare_proxy(): crop完成，新尺寸={(x1-x0)}x{(y1-y0)}", flush=True)
                except Exception as e:
                    print(f"[ERROR] _prepare_proxy(): crop失败: {e}", flush=True)
            # 接触印相聚焦：无激活 crop，但存在 contactsheet 裁剪矩形
            elif (self._active_crop_id is None and
                  self._contactsheet_profile.crop_rect is not None and
                  src_image.array is not None):
                print("[DEBUG] _prepare_proxy(): 单张裁剪聚焦模式", flush=True)
                try:
                    x, y, w, h = self._contactsheet_profile.crop_rect
                    x0 = int(round(x * orig_w))
                    y0 = int(round(y * orig_h))
                    x1 = int(round((x + w) * orig_w))
                    y1 = int(round((y + h) * orig_h))
                    x0 = max(0, min(orig_w - 1, x0))
                    x1 = max(x0 + 1, min(orig_w, x1))
                    y0 = max(0, min(orig_h - 1, y0))
                    y1 = max(y0 + 1, min(orig_h, y1))
                    cropped_arr = src_image.array[y0:y1, x0:x1, :].copy()
                    src_image = src_image.copy_with_new_array(cropped_arr)
                    print(f"[DEBUG] _prepare_proxy(): contactsheet crop完成，新尺寸={(x1-x0)}x{(y1-y0)}", flush=True)
                except Exception as e:
                    print(f"[ERROR] _prepare_proxy(): contactsheet crop失败: {e}", flush=True)
        else:
            print("[DEBUG] _prepare_proxy(): Contactsheet模式（非聚焦），使用完整原图", flush=True)

        # 生成downsampled proxy（基于crop后的图像，或完整图像）
        print("[DEBUG] _prepare_proxy(): 开始生成proxy...", flush=True)
        try:
            proxy = self.image_manager.generate_proxy(
                src_image,
                self.the_enlarger.preview_config.get_proxy_size_tuple()
            )
            print(f"[DEBUG] _prepare_proxy(): proxy生成成功，尺寸={proxy.width}x{proxy.height}", flush=True)
        except Exception as e:
            print(f"[ERROR] _prepare_proxy(): proxy生成失败: {e}", flush=True)
            return

        # 保存原图尺寸到metadata（供UI层使用）
        proxy.metadata['source_wh'] = (int(orig_w), int(orig_h))

        # 释放旧proxy并保存新proxy
        if self._current_proxy is not None:
            print("[DEBUG] _prepare_proxy(): 释放旧proxy", flush=True)
            del self._current_proxy
        self._current_proxy = proxy

        # 标记 proxy 需要重载到 worker（进程模式专用）
        if self._use_process_isolation:
            self._proxy_needs_reload = True
            print("[DEBUG] _prepare_proxy(): 标记 proxy 需要重载", flush=True)

        print("[DEBUG] _prepare_proxy() 执行完成，proxy已更新", flush=True)

    def get_current_idt_gamma(self) -> float:
        """读取当前输入色彩空间的IDT Gamma（无则返回1.0）。"""
        try:
            cs_name = self._current_params.input_color_space_name
            cs_info = self.color_space_manager.get_color_space_info(cs_name) or {}
            return float(cs_info.get("gamma", 1.0))
        except Exception:
            return 1.0

    def get_current_idt_primaries(self) -> np.ndarray:
        """读取当前输入色彩空间的IDT Primaries（无则返回sRGB）。
        
        Returns:
            形状为 (3, 2) 的 numpy 数组，按 R、G、B 顺序为 xy 坐标
        """
        try:
            cs_name = self._current_params.input_color_space_name
            cs_info = self.color_space_manager.get_color_space_info(cs_name) or {}
            
            # 获取 primaries，期望格式为 (3, 2) 数组
            primaries = cs_info.get("primaries")
            if primaries is not None:
                primaries_array = np.asarray(primaries, dtype=np.float64)
                if primaries_array.shape == (3, 2):
                    return primaries_array
                else:
                    # 记录格式问题但继续使用默认值
                    if hasattr(self, '_verbose_logs') and self._verbose_logs:
                        print(f"警告: 色彩空间 {cs_name} 的 primaries 格式不正确: {primaries_array.shape}")
        except Exception as e:
            # 记录异常但继续使用默认值
            if hasattr(self, '_verbose_logs') and self._verbose_logs:
                print(f"警告: 获取当前IDT primaries失败: {e}")
        
        # 默认返回 sRGB primaries (与原始硬编码值保持一致)
        return np.array([0.64, 0.33, 0.30, 0.60, 0.15, 0.06], dtype=np.float64).reshape(3, 2)

    def _trigger_preview_update(self):
        """触发预览更新（根据配置选择进程或线程模式）"""
        print("[DEBUG] _trigger_preview_update() 开始执行", flush=True)

        if not self._current_proxy:
            print("[WARNING] _trigger_preview_update(): _current_proxy为None，提前返回", flush=True)
            return

        # 如果正在加载图片，延迟预览
        if self._loading_image:
            print("[WARNING] _trigger_preview_update(): 正在加载图片（_loading_image=True），提前返回", flush=True)
            return

        # 根据配置选择模式
        print(f"[DEBUG] _trigger_preview_update(): 使用{'进程' if self._use_process_isolation else '线程'}模式", flush=True)
        if self._use_process_isolation:
            print("[DEBUG] _trigger_preview_update(): 调用_trigger_preview_with_process()", flush=True)
            self._trigger_preview_with_process()
        else:
            print("[DEBUG] _trigger_preview_update(): 调用_trigger_preview_with_thread()", flush=True)
            self._trigger_preview_with_thread()
        print("[DEBUG] _trigger_preview_update() 执行完成", flush=True)

    def _on_preview_result(self, result_image: ImageData):
        if result_image is not None:
            print(f"[DEBUG] _on_preview_result(): 收到预览结果，尺寸={result_image.width}x{result_image.height}", flush=True)
            print("[DEBUG] _on_preview_result(): 发射preview_updated信号", flush=True)
            self.preview_updated.emit(result_image)
            print("[DEBUG] _on_preview_result(): preview_updated信号已发射", flush=True)
            # If an iterative auto color is in progress, trigger the next step
            if self._auto_color_iterations > 0 and self._get_preview_for_auto_color_callback:
                print("[DEBUG] _on_preview_result(): 触发下一次auto_color迭代", flush=True)
                QTimer.singleShot(0, self._perform_auto_color_iteration)
            # If neutral point iteration is in progress, trigger the next step
            if self._neutral_point_iterations > 0 and self._neutral_point_callback:
                print("[DEBUG] _on_preview_result(): 触发下一次neutral_point迭代", flush=True)
                QTimer.singleShot(0, self._perform_neutral_point_iteration)
        else:
            pass # print("[ERROR] _on_preview_result(): result_image is None")


    def _on_preview_error(self, message: str):
        # 如果是 proxy 重载失败，直接把进程干掉，避免后面用坏掉的 proxy 再跑一次
        if "Failed to reload proxy" in message:
            print("[DEBUG] _on_preview_error(): 检测到 proxy 重载失败，重启 worker 进程", flush=True)
            try:
                self._shutdown_preview_worker_process()
            except Exception as e:
                print(f"[DEBUG] _on_preview_error(): shutdown worker 失败: {e}", flush=True)

        # 在色卡优化期间不发送预览错误消息，避免覆盖优化状态
        if not self._ccm_optimization_active:
            self.status_message_changed.emit(tr("app_context.preview.update_failed", message=message))
        else:
            print(f"[DEBUG] 色卡优化期间忽略预览错误: {message}")
        self._auto_color_iterations = 0 # Stop iteration on error
        self._get_preview_for_auto_color_callback = None
        
    def set_ccm_optimization_active(self, active: bool):
        """设置色卡优化状态"""
        self._ccm_optimization_active = active
        if active:
            print(f"[DEBUG] CCM优化已激活，将忽略预览错误消息")
        else:
            print(f"[DEBUG] CCM优化已结束，恢复正常状态消息")

    def _on_preview_finished(self):
        """预览处理完成的回调

        执行关键的内存清理工作：
        1. 断开 worker signals 的所有连接，允许 worker 对象被垃圾回收
        2. 释放 worker 持有的图像数据副本（通过允许 GC 回收 worker）
        3. 重置 busy 标志
        4. 触发 pending 的预览请求

        修复说明：
        - 解决 Preview Worker 累积性内存泄漏问题
        - 每个 worker 完成后立即释放其信号连接
        - 防止 worker 对象和其持有的大型数据（~17MB/worker）累积
        """
        print("[DEBUG] _on_preview_finished(): Preview worker完成", flush=True)
        # 获取发出 finished 信号的 signals 对象（来自刚完成的 worker）
        sender_signals = self.sender()

        # 断开该 worker 的所有信号连接，这是防止内存泄漏的关键步骤
        # 如果不断开，signals 对象会持有对 ApplicationContext 方法的引用，
        # 导致 worker 对象无法被 GC，其持有的图像副本也无法释放
        if sender_signals:
            print("[DEBUG] _on_preview_finished(): 断开worker信号连接", flush=True)
            try:
                # 为每个信号独立处理断开，确保即使某个失败也不影响其他
                try:
                    sender_signals.result.disconnect(self._on_preview_result)
                except (RuntimeError, TypeError):
                    # 连接可能已经断开或不存在，忽略异常
                    pass

                try:
                    sender_signals.error.disconnect(self._on_preview_error)
                except (RuntimeError, TypeError):
                    pass

                try:
                    sender_signals.finished.disconnect(self._on_preview_finished)
                except (RuntimeError, TypeError):
                    pass

            except Exception as e:
                # 记录意外异常，但不中断预览流程
                print(f"[WARNING] 清理 preview worker 信号连接时出错: {e}", flush=True)

        # 重置预览忙碌状态
        print("[DEBUG] _on_preview_finished(): 重置busy标志", flush=True)
        self._preview_busy = False

        # 如果有 pending 的预览请求，触发它
        # 这确保了高频更新时的防抖机制正常工作
        if self._preview_pending:
            print("[DEBUG] _on_preview_finished(): 有pending请求，触发preview更新", flush=True)
            self._preview_pending = False
            self._trigger_preview_update()
        else:
            print("[DEBUG] _on_preview_finished(): 没有pending请求", flush=True)

    # =================
    # 方向与旋转（UI调用）
    # =================
    def get_current_orientation(self) -> int:
        """根据当前profile返回对应的orientation"""
        if self._current_profile_kind == 'contactsheet':
            return self._contactsheet_profile.orientation
        elif self._active_crop_id:
            crop = self.get_active_crop_instance()
            return crop.orientation if crop else 0
        return 0

    def set_orientation(self, degrees: int):
        """设置当前profile的orientation"""
        try:
            print(f"[DEBUG] set_orientation() 开始执行, degrees={degrees}", flush=True)

            deg = int(degrees) % 360
            # 规范到 0/90/180/270
            choices = [0, 90, 180, 270]
            normalized = min(choices, key=lambda x: abs(x - deg))
            print(f"[DEBUG] set_orientation(): 规范化后的角度={normalized}, profile_kind={self._current_profile_kind}", flush=True)

            # 直接写入对应的数据源
            if self._current_profile_kind == 'contactsheet':
                print(f"[DEBUG] set_orientation(): 设置contactsheet orientation={normalized}", flush=True)
                self._contactsheet_profile.orientation = normalized
            elif self._active_crop_id:
                crop = self.get_active_crop_instance()
                print(f"[DEBUG] set_orientation(): crop模式, crop={'存在' if crop else '不存在'}", flush=True)
                if crop:
                    print(f"[DEBUG] set_orientation(): 设置crop orientation={normalized}", flush=True)
                    crop.orientation = normalized

            # 触发预览更新（orientation在worker中处理，不需要prepare_proxy）
            if self._current_image:
                print("[DEBUG] set_orientation(): 触发预览更新", flush=True)
                self._trigger_preview_update()
                self._autosave_timer.start()
            else:
                print("[WARNING] set_orientation(): 没有当前图像，跳过预览更新", flush=True)

            print("[DEBUG] set_orientation() 执行完成", flush=True)
        except Exception as e:
            import traceback
            error_msg = tr("app_context.errors.set_orientation_failed", error=str(e), traceback=traceback.format_exc())
            print(f"[ERROR] {error_msg}", flush=True)
            self.status_message_changed.emit(error_msg)
    
    def _ensure_ui_state_sync_for_monochrome(self):
        """确保黑白模式下UI状态正确同步（延迟调用以避免时序问题）"""
        try:
            # 再次发射film_type_changed信号，确保UI层收到并处理
            self.film_type_changed.emit(self._current_film_type)
            print(f"[ApplicationContext] 强制同步黑白模式UI状态: {self._current_film_type}")
        except Exception as e:
            print(f"[ApplicationContext] UI状态同步失败: {e}")

    def rotate(self, direction: int):
        """direction: 1=左旋+90°, -1=右旋-90°
        纯净的旋转逻辑：crop和全局orientation完全分离
        """
        try:
            print(f"[DEBUG] rotate() 开始执行, direction={direction}", flush=True)

            # 验证当前状态
            if not self._current_image:
                print("[WARNING] rotate(): 没有当前图像，取消旋转操作", flush=True)
                return

            step = 90 if int(direction) >= 0 else -90
            print(f"[DEBUG] rotate(): step={step}, crop_focused={self._crop_focused}, profile_kind={self._current_profile_kind}", flush=True)

            if self._crop_focused or self._current_profile_kind == 'crop':
                # 聚焦或裁剪Profile下：只旋转当前crop的orientation
                crop_instance = self.get_active_crop_instance()
                print(f"[DEBUG] rotate(): crop模式, crop_instance={'存在' if crop_instance else '不存在'}", flush=True)
                if crop_instance:
                    old_orientation = crop_instance.orientation
                    new_orientation = (crop_instance.orientation + step) % 360
                    print(f"[DEBUG] rotate(): 更新crop orientation: {old_orientation} -> {new_orientation}", flush=True)
                    # 仅更新当前裁剪的方向，保留其它裁剪
                    self.update_active_crop_orientation(new_orientation)
                    print("[DEBUG] rotate(): 触发预览更新（orientation在worker中处理）...", flush=True)
                    self._trigger_preview_update()
                    print("[DEBUG] rotate(): 启动自动保存计时器", flush=True)
                    self._autosave_timer.start()
                else:
                    print("[WARNING] rotate(): crop模式下没有活动crop实例", flush=True)
            else:
                # 非聚焦状态：只旋转contactsheet orientation（不影响crop）
                current_orientation = self.get_current_orientation()
                new_deg = (current_orientation + step) % 360
                print(f"[DEBUG] rotate(): contactsheet模式, orientation: {current_orientation} -> {new_deg}", flush=True)
                self.set_orientation(new_deg)
                # 注意：不同步crop的orientation，保持完全分离

            # 发射旋转完成信号，让MainWindow知道需要fit to window
            print("[DEBUG] rotate(): 发射rotation_completed信号", flush=True)
            self.rotation_completed.emit()
            print("[DEBUG] rotate() 执行完成", flush=True)
        except Exception as e:
            import traceback
            error_msg = tr("app_context.errors.rotate_failed", error=str(e), traceback=traceback.format_exc())
            print(f"[ERROR] {error_msg}", flush=True)
            self.status_message_changed.emit(error_msg)

    def update_active_crop_orientation(self, orientation: int) -> None:
        """仅更新当前活跃裁剪的 orientation，保留所有裁剪与激活状态。"""
        try:
            print(f"[DEBUG] update_active_crop_orientation() 开始执行, orientation={orientation}", flush=True)

            crop_instance = self.get_active_crop_instance()
            if crop_instance:
                old_orientation = crop_instance.orientation
                new_orientation = int(orientation) % 360
                crop_instance.orientation = new_orientation
                print(f"[DEBUG] update_active_crop_orientation(): 已更新 {old_orientation} -> {new_orientation}", flush=True)
            else:
                print("[WARNING] update_active_crop_orientation(): 没有活动的crop实例", flush=True)

            print("[DEBUG] update_active_crop_orientation() 执行完成", flush=True)
        except Exception as e:
            import traceback
            error_msg = tr("app_context.errors.update_crop_orientation_failed", error=str(e), traceback=traceback.format_exc())
            print(f"[ERROR] {error_msg}", flush=True)
            self.status_message_changed.emit(error_msg)

    # ==== 单张裁剪：仅记录在 contactsheet，不创建正式 crop ====
    def set_contactsheet_crop(self, rect_norm: tuple[float, float, float, float]) -> None:
        try:
            x, y, w, h = [float(max(0.0, min(1.0, v))) for v in rect_norm]
            w = max(0.0, min(1.0 - x, w))
            h = max(0.0, min(1.0 - y, h))
            self._contactsheet_profile.crop_rect = (x, y, w, h)
            # 不改变 profile 与聚焦，仅发出 overlay 的变更
            self.crop_changed.emit(self._contactsheet_profile.crop_rect)
            self._autosave_timer.start()
        except Exception:
            pass

    def reload_curves_config(self):
        """重新加载curves配置"""
        try:
            # 发出curves配置重载信号，通知所有相关UI组件刷新
            self.curves_config_reloaded.emit()
        except Exception as e:
            self.status_message_changed.emit(tr("app_context.config.reload_curves_failed", error=e))

    def reload_all_configs(self):
        """重新加载所有配置文件"""
        try:
            # 重新加载色彩空间配置
            self.color_space_manager.reload_config()
            
            # 重新加载矩阵配置
            self.the_enlarger.pipeline_processor.reload_matrices()
            
            # 重新加载curves配置
            self.reload_curves_config()

            self.status_message_changed.emit(tr("app_context.config.reloaded"))
        except Exception as e:
            self.status_message_changed.emit(tr("app_context.config.reload_failed", error=e))
    
    def get_reference_colors(self, filename: str):
        """
        获取ColorChecker参考色彩数据，确保Preview和Optimizer使用相同的数据
        
        Args:
            filename: ColorChecker JSON文件名
            
        Returns:
            Dict[str, List[float]]: 色块ID到工作空间RGB值的映射
        """
        from typing import Dict, List
        
        try:
            # 获取当前工作空间
            current_workspace = self.color_space_manager.get_current_working_space()
            
            # 检查缓存是否有效
            cache_valid = (self._cached_reference_colors is not None and 
                          self._reference_colorspace == current_workspace and
                          self._reference_filename == filename)
            
            if cache_valid:
                print(f"[DEBUG] 使用cached reference colors: {filename} @ {current_workspace}")
                return self._cached_reference_colors
            else:
                print(f"[DEBUG] 重新加载reference colors: {filename} @ {current_workspace}")
                print(f"[DEBUG] 缓存状态: colors={self._cached_reference_colors is not None}, "
                      f"space_match={self._reference_colorspace == current_workspace}, "
                      f"file_match={self._reference_filename == filename}")
                
                # 重新加载并缓存
                from divere.utils.colorchecker_loader import load_colorchecker_reference
                
                self._cached_reference_colors = load_colorchecker_reference(
                    filename, 
                    current_workspace,
                    self.color_space_manager
                )
                self._reference_colorspace = current_workspace
                self._reference_filename = filename
                
                print(f"[DEBUG] 成功缓存reference colors: {len(self._cached_reference_colors)} patches")
                
            return self._cached_reference_colors
            
        except Exception as e:
            print(f"获取reference colors失败: {e}")
            import traceback
            print(f"[DEBUG] 错误详情: {traceback.format_exc()}")
            return {}
    
    def clear_reference_color_cache(self):
        """清除reference color缓存（当工作空间或ColorChecker文件变化时调用）"""
        if self._cached_reference_colors is not None:
            print(f"[DEBUG] 清除reference color缓存: {self._reference_filename} @ {self._reference_colorspace}")
        self._cached_reference_colors = None
        self._reference_colorspace = None
        self._reference_filename = None
    
    def update_proxy_max_size(self, size: int):
        """更新代理图像最大尺寸设置"""
        try:
            # 保存到配置
            enhanced_config_manager.set_ui_setting("proxy_max_size", size)
            
            # 更新the_enlarger的preview_config
            self.the_enlarger.preview_config.proxy_max_size = size
            
            # 触发预览更新以应用新的proxy尺寸
            self._trigger_preview_update()

            self.status_message_changed.emit(tr("app_context.config.proxy_size_updated", size=size))
        except Exception as e:
            self.status_message_changed.emit(tr("app_context.config.proxy_size_update_failed", error=e))

    # =================
    # 状态备份/恢复方法（用于批量保存）
    # =================
    def backup_state(self) -> dict:
        """备份当前Context状态，用于批量保存时的临时状态切换"""
        try:
            return {
                # 核心状态
                'current_image': self._current_image,
                'current_proxy': self._current_proxy,
                'current_params': self._current_params.shallow_copy() if self._current_params else None,  # 优化：只读备份，使用 shallow_copy()
                'current_film_type': self._current_film_type,

                # 裁剪相关状态
                'crops': [crop for crop in self._crops],  # 浅拷贝CropInstance列表
                'active_crop_id': self._active_crop_id,
                'crop_focused': self._crop_focused,
                'current_profile_kind': self._current_profile_kind,
                'contactsheet_profile': self._contactsheet_profile.copy() if hasattr(self._contactsheet_profile, 'copy') else self._contactsheet_profile,
                'per_crop_params': {k: v.shallow_copy() if v else None for k, v in self._per_crop_params.items()},  # 优化：只读备份，使用 shallow_copy()
                
                # 其他状态
                'loading_image': self._loading_image,
                'preview_busy': self._preview_busy,
                'preview_pending': self._preview_pending,
            }
        except Exception as e:
            print(f"备份状态失败: {e}")
            return {}

    def restore_state(self, backup: dict):
        """恢复Context状态"""
        try:
            if not backup:
                return

            # 显式释放旧对象以防止内存泄漏
            if self._current_image is not None:
                del self._current_image
            if self._current_proxy is not None:
                del self._current_proxy

            # 恢复核心状态
            self._current_image = backup.get('current_image')
            self._current_proxy = backup.get('current_proxy')
            if backup.get('current_params'):
                self._current_params = backup['current_params'].shallow_copy()  # 优化：只读恢复，使用 shallow_copy()
            self._current_film_type = backup.get('current_film_type', 'color_negative_c41')
            
            # 恢复裁剪相关状态
            self._crops = backup.get('crops', [])
            self._active_crop_id = backup.get('active_crop_id')
            self._crop_focused = backup.get('crop_focused', False)
            self._current_profile_kind = backup.get('current_profile_kind', 'contactsheet')
            if backup.get('contactsheet_profile'):
                self._contactsheet_profile = backup['contactsheet_profile']
            self._per_crop_params = backup.get('per_crop_params', {})
            
            # 恢复其他状态
            self._loading_image = backup.get('loading_image', False)
            self._preview_busy = backup.get('preview_busy', False)
            self._preview_pending = backup.get('preview_pending', False)

            print("Context状态已恢复")

            # 状态恢复后触发预览更新，确保UI显示正确
            # 修复：导出crops后preview显示错误的bug

            # 强制清除加载标志，允许预览更新
            # 修复：恢复状态后preview被永久锁定的bug
            # 原因：backup可能包含_loading_image=True，恢复后会阻止所有预览更新
            self._loading_image = False

            # 重新生成proxy，确保与恢复的状态匹配
            # 修复：批量导出后proxy对象不匹配导致preview锁定的bug
            # 原因：backup的proxy对象引用指向旧图像，与恢复的_current_image等状态不匹配
            # _prepare_proxy() 会根据恢复的状态（_crop_focused、_active_crop_id等）生成正确的proxy
            # 并自动设置_proxy_needs_reload=True（进程模式下）
            print("[DEBUG] Call for _prepare_proxy: restore_state()")
            self._prepare_proxy()

            # 发送参数变更信号，更新UI参数面板
            if self._current_params:
                self.params_changed.emit(self._current_params)

            # 触发预览更新，刷新PreviewWidget显示
            self._trigger_preview_update()

        except Exception as e:
            print(f"恢复状态失败: {e}")

    # =================
    # 内存管理辅助方法
    # =================
    def _clear_current_image_data(self):
        """智能清理：清除当前图像数据（保留缓存以提高性能）

        清理内容：
        - _current_image: 当前原始图像（~50-200MB）
        - _current_proxy: 当前代理图像（~17MB）

        不清理内容（保留以提高性能）：
        - ImageManager._proxy_cache: 代理缓存（LRU 自动管理）
        - LUTProcessor._lut_cache: LUT 缓存（LRU 自动管理）
        - ColorSpaceManager._convert_cache: 转换缓存（LRU 自动管理）
        """
        # 清理当前原始图像
        if self._current_image is not None:
            if hasattr(self._current_image, 'array') and self._current_image.array is not None:
                self._current_image.array = None  # 释放 numpy 数组
            self._current_image = None

        # 清理当前代理图像
        if self._current_proxy is not None:
            if hasattr(self._current_proxy, 'array') and self._current_proxy.array is not None:
                self._current_proxy.array = None  # 释放代理的 numpy 数组
            self._current_proxy = None

    def _clear_all_caches(self):
        """激进清理：清空所有缓存和当前图像数据

        用于"完全从文件重新加载"的场景，会清空所有缓存。
        警告：这会导致性能下降，因为后续操作需要重新加载和计算。

        清理内容：
        - 当前图像数据（_current_image, _current_proxy）
        - ImageManager 代理缓存（~170MB）
        - LUTProcessor LUT 缓存（~20MB）
        - ColorSpaceManager 转换缓存（~0.02MB）
        - 触发垃圾回收
        """
        # 1. 清理当前图像数据
        self._clear_current_image_data()

        # 2. 清理 ImageManager 缓存
        if hasattr(self, 'image_manager') and self.image_manager:
            self.image_manager.clear_cache()

        # 3. 清理 LUTProcessor 缓存
        if (hasattr(self, 'the_enlarger') and self.the_enlarger and
            hasattr(self.the_enlarger, 'lut_processor') and self.the_enlarger.lut_processor):
            self.the_enlarger.lut_processor.clear_cache()

        # 4. 清理 ColorSpaceManager 转换缓存
        if hasattr(self, 'color_space_manager') and self.color_space_manager:
            self.color_space_manager.clear_convert_cache()

        # 5. 新增：清 Qt 的全局 pixmap 缓存
        try:
            QPixmapCache.clear()
        except Exception as e:
            print("[MEM] QPixmapCache.clear() failed:", e)

        # 6. 手动触发垃圾回收（确保内存立即释放）
        import gc
        gc.collect()

    def get_memory_usage_report(self) -> dict:
        """获取详细的内存使用报告

        返回各个组件的内存使用情况，用于诊断和监控。

        Returns:
            包含以下键的字典：
            - current_image: 当前原始图像大小（字节）
            - current_proxy: 当前代理图像大小（字节）
            - proxy_cache: 代理缓存信息
            - lut_cache: LUT 缓存信息
            - colorspace_cache: 色彩空间转换缓存信息
            - total_estimated_mb: 总估计内存使用（MB）
        """
        import sys

        report = {}

        # 1. 当前原始图像
        current_image_size = 0
        if self._current_image is not None:
            current_image_size = sys.getsizeof(self._current_image)
            if hasattr(self._current_image, 'array') and self._current_image.array is not None:
                # numpy 数组的实际大小
                current_image_size += self._current_image.array.nbytes
        report['current_image_bytes'] = current_image_size
        report['current_image_mb'] = round(current_image_size / (1024 * 1024), 2)

        # 2. 当前代理图像
        current_proxy_size = 0
        if self._current_proxy is not None:
            current_proxy_size = sys.getsizeof(self._current_proxy)
            if hasattr(self._current_proxy, 'array') and self._current_proxy.array is not None:
                current_proxy_size += self._current_proxy.array.nbytes
        report['current_proxy_bytes'] = current_proxy_size
        report['current_proxy_mb'] = round(current_proxy_size / (1024 * 1024), 2)

        # 3. ImageManager 代理缓存
        proxy_cache_count = 0
        proxy_cache_max = 0
        if hasattr(self, 'image_manager') and self.image_manager:
            proxy_cache_count = len(self.image_manager._proxy_cache)
            proxy_cache_max = self.image_manager._max_cache_size
        report['proxy_cache'] = {
            'count': proxy_cache_count,
            'max': proxy_cache_max,
            'estimated_mb': round(proxy_cache_count * 17, 2)  # 假设每个代理约 17MB
        }

        # 4. LUTProcessor LUT 缓存
        lut_cache_count = 0
        lut_cache_max = 0
        if (hasattr(self, 'the_enlarger') and self.the_enlarger and
            hasattr(self.the_enlarger, 'lut_processor') and self.the_enlarger.lut_processor):
            lut_cache_count = len(self.the_enlarger.lut_processor._lut_cache)
            lut_cache_max = self.the_enlarger.lut_processor._max_cache_size
        report['lut_cache'] = {
            'count': lut_cache_count,
            'max': lut_cache_max,
            'estimated_mb': round(lut_cache_count * 1, 2)  # 假设每个 LUT 约 1MB
        }

        # 5. ColorSpaceManager 转换缓存
        colorspace_cache_count = 0
        colorspace_cache_max = 0
        if hasattr(self, 'color_space_manager') and self.color_space_manager:
            colorspace_cache_count = len(self.color_space_manager._convert_cache)
            colorspace_cache_max = self.color_space_manager._convert_cache_max_size
        report['colorspace_cache'] = {
            'count': colorspace_cache_count,
            'max': colorspace_cache_max,
            'estimated_mb': round(colorspace_cache_count * 0.0001, 4)  # 非常小
        }

        # 6. 总估计内存
        total_mb = (
            report['current_image_mb'] +
            report['current_proxy_mb'] +
            report['proxy_cache']['estimated_mb'] +
            report['lut_cache']['estimated_mb'] +
            report['colorspace_cache']['estimated_mb']
        )
        report['total_estimated_mb'] = round(total_mb, 2)

        return report

    def print_memory_usage_report(self):
        """打印内存使用报告到控制台（便于调试）"""
        report = self.get_memory_usage_report()

        print("\n" + "="*60)
        print("内存使用报告 (Memory Usage Report)")
        print("="*60)
        print(f"当前原始图像:       {report['current_image_mb']:>8.2f} MB")
        print(f"当前代理图像:       {report['current_proxy_mb']:>8.2f} MB")
        print("-"*60)
        print(f"代理缓存:           {report['proxy_cache']['count']:>3d} / {report['proxy_cache']['max']:>3d} 个  (~{report['proxy_cache']['estimated_mb']:>6.2f} MB)")
        print(f"LUT 缓存:           {report['lut_cache']['count']:>3d} / {report['lut_cache']['max']:>3d} 个  (~{report['lut_cache']['estimated_mb']:>6.2f} MB)")
        print(f"色彩空间缓存:       {report['colorspace_cache']['count']:>3d} / {report['colorspace_cache']['max']:>3d} 个  (~{report['colorspace_cache']['estimated_mb']:>6.4f} MB)")
        print("-"*60)
        print(f"总估计内存使用:     {report['total_estimated_mb']:>8.2f} MB")
        print("="*60 + "\n")

    # =================
    # 进程隔离相关方法（用于解决 macOS heap 内存不归还问题）
    # 参考文档：PROCESS_ISOLATION_ANALYSIS.md
    # =================

    def _should_restart_worker(self) -> bool:
        """判断是否需要重启 worker 进程（内存监控）

        Returns:
            bool: True 表示需要重启，False 表示可以复用
        """
        if not hasattr(self, '_preview_worker_process') or self._preview_worker_process is None:
            print("[WORKER] No worker process, no need to restart")
            return False  # 还没创建，不需要重启

        if not self._preview_worker_process.is_alive():
            print("[WORKER] Worker process is dead, need to restart")
            return True  # 进程已死，需要重启

        # 检查内存使用量（从配置文件读取阈值）
        try:
            # 获取 PID 用于调试
            worker_pid = self._preview_worker_process.process.pid if self._preview_worker_process.process else None

            # 从配置文件读取内存阈值（默认 4GB）
            from divere.utils.enhanced_config_manager import enhanced_config_manager
            threshold_mb = enhanced_config_manager.get_ui_setting("worker_memory_threshold_mb", 4000)

            mem_mb = self._preview_worker_process.get_memory_usage()
            if mem_mb is not None:
                print(f"[WORKER] PID {worker_pid} memory usage: {mem_mb:.1f} MB (threshold: {threshold_mb} MB)")
                if mem_mb > threshold_mb:
                    print(f"[WORKER] ⚠️  Memory threshold exceeded ({mem_mb:.1f} MB > {threshold_mb} MB), will restart")
                    return True
                else:
                    print(f"[WORKER] ✓ Memory OK, will reuse worker")
            else:
                print(f"[WORKER] ⚠️  Failed to get memory usage for PID {worker_pid} (psutil not available?), will reuse worker")
        except Exception as e:
            print(f"[WORKER] ⚠️  Exception getting memory usage: {e}, will reuse worker")
            import traceback
            traceback.print_exc()

        return False  # 内存正常，可以复用

    def _shutdown_preview_worker_process(self):
        """销毁 worker 进程并清理 shared memory（幂等操作）"""
        if not hasattr(self, '_preview_worker_process'):
            return

        # 销毁 worker 进程
        if self._preview_worker_process is not None:
            try:
                self._preview_worker_process.shutdown()
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to shutdown worker process: {e}")
            finally:
                self._preview_worker_process = None

        # 清理 shared memory
        if self._proxy_shared_memory is not None:
            try:
                self._proxy_shared_memory.close()
                self._proxy_shared_memory.unlink()
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to cleanup shared memory: {e}")
            finally:
                self._proxy_shared_memory = None

        # 停止轮询定时器
        if hasattr(self, '_result_poll_timer') and self._result_poll_timer is not None:
            self._result_poll_timer.stop()

    def _create_preview_worker_process(self):
        """创建 worker 进程并传递 proxy（Lazy 创建）

        失败时自动回退到线程模式
        """
        if not self._current_proxy:
            return

        try:
            from multiprocessing import shared_memory
            from divere.core.preview_worker_process import PreviewWorkerProcess

            # 1. 生成 proxy（如果还没有）
            proxy = self._current_proxy

            # 2. 创建 shared memory 并写入 proxy
            shm = shared_memory.SharedMemory(create=True, size=proxy.array.nbytes)
            shm_array = np.ndarray(proxy.array.shape, dtype=proxy.array.dtype,
                                   buffer=shm.buf)
            np.copyto(shm_array, proxy.array)

            # 3. 创建 worker 进程
            self._preview_worker_process = PreviewWorkerProcess(
                proxy_shm_name=shm.name,
                proxy_shape=proxy.array.shape,
                proxy_dtype=str(proxy.array.dtype),
                init_config={},  # 可选配置
            )
            self._preview_worker_process.start()

            # 4. 验证进程启动成功
            import time
            time.sleep(0.1)
            if not self._preview_worker_process.is_alive():
                raise RuntimeError("Worker process failed to start")

            # 5. 保存 shared memory 引用并重置标志
            self._proxy_shared_memory = shm
            self._proxy_needs_reload = False  # Worker 已加载 proxy

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Process isolation failed, falling back to thread mode: {e}")

            # 清理失败的资源
            if hasattr(self, '_proxy_shared_memory') and self._proxy_shared_memory:
                try:
                    self._proxy_shared_memory.close()
                    self._proxy_shared_memory.unlink()
                except:
                    pass
                self._proxy_shared_memory = None

            # 自动回退到线程模式
            self._use_process_isolation = False
            self._preview_worker_process = None

            # 提示用户
            self.status_message_changed.emit(
                tr("app_context.preview.process_fallback")
            )

            # 使用线程模式重新触发预览
            self._trigger_preview_with_thread()

    def _reload_worker_proxy(self):
        """重新加载 worker 的 proxy（不重启进程，热重载）

        用于切换图片时复用 worker 进程，避免重新 import 模块的开销
        """
        if not self._current_proxy:
            return

        if not hasattr(self, '_preview_worker_process') or self._preview_worker_process is None:
            return

        try:
            from multiprocessing import shared_memory

            proxy = self._current_proxy

            # 1. 保存旧的 shared memory 引用（延迟删除）
            old_shm = self._proxy_shared_memory

            # 2. 创建新的 shared memory 并写入新 proxy
            shm = shared_memory.SharedMemory(create=True, size=proxy.array.nbytes)
            shm_array = np.ndarray(proxy.array.shape, dtype=proxy.array.dtype,
                                   buffer=shm.buf)
            np.copyto(shm_array, proxy.array)

            # 3. 通知 worker 重新加载 proxy
            self._preview_worker_process.reload_proxy(
                proxy_shm_name=shm.name,
                proxy_shape=proxy.array.shape,
                proxy_dtype=str(proxy.array.dtype)
            )

            # 4. 更新 shared memory 引用
            self._proxy_shared_memory = shm

            # 5. 重置标志：proxy 已成功重载
            self._proxy_needs_reload = False

            print(f"[WORKER] Reloaded proxy via hot-reload (shape={proxy.array.shape})")

            # 6. 事件驱动的 shared memory 清理（替代延迟等待）
            # 当 worker 成功 reload 后，会触发 proxy_reloaded 信号，然后清理旧 shared memory
            if old_shm is not None:
                def cleanup_old_shm_callback():
                    """事件驱动的 shared memory 清理（在 worker 确认 reload 后执行）"""
                    import threading
                    def do_cleanup():
                        try:
                            old_shm.close()
                            old_shm.unlink()
                        except Exception as e:
                            import logging
                            logger = logging.getLogger(__name__)
                            logger.debug(f"Failed to cleanup old shared memory (may already be closed): {e}")

                    # 使用后台线程执行清理（避免阻塞事件循环）
                    cleanup_thread = threading.Thread(target=do_cleanup, daemon=True)
                    cleanup_thread.start()

                # 注册回调：在 worker 成功 reload 后清理
                self._preview_worker_process.set_on_proxy_reloaded_callback(cleanup_old_shm_callback)

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to reload worker proxy: {e}")
            # 重载失败，回退到重启 worker
            self._shutdown_preview_worker_process()

    def _trigger_preview_with_process(self):
        """使用进程模式触发预览"""
        if not self._current_proxy:
            return

        # 进程模式也用 busy/pending
        if self._preview_busy:
            self._preview_pending = True
            return
        self._preview_busy = True

        # Lazy 创建或热重载 worker 进程
        if self._preview_worker_process is None:
            # 首次创建 worker 进程
            self._create_preview_worker_process()
        elif self._preview_worker_process.is_alive():
            # Worker 存活，检查是否需要重载 proxy
            if self._proxy_needs_reload:
                print("[DEBUG] _trigger_preview_with_process(): proxy 已改变，重载 proxy", flush=True)
                self._reload_worker_proxy()
            else:
                print("[DEBUG] _trigger_preview_with_process(): proxy 未改变，跳过重载", flush=True)
        else:
            # Worker 已死，需要重新创建
            self._shutdown_preview_worker_process()
            self._create_preview_worker_process()

        # 如果创建失败（已回退到线程模式），直接返回
        if self._preview_worker_process is None:
            return

        # Worker不需要crop（主进程在_prepare_proxy()中已处理）
        # - Crop focused模式：proxy已经是crop后的高质量图
        # - Contactsheet模式：proxy是完整的原图
        crop_rect_norm = None

        # 构建显示状态元数据（供UI正确渲染overlay和旋转）
        display_metadata = {}

        # Crop focused状态
        display_metadata['crop_focused'] = self._crop_focused

        # Crop overlay信息（仅在非focused模式下传递）
        if not self._crop_focused:
            crop_instance = self.get_active_crop_instance()
            if crop_instance and crop_instance.rect_norm:
                display_metadata['crop_overlay'] = crop_instance.rect_norm

        # 原图尺寸（用于UI计算overlay位置）
        if self._current_image:
            display_metadata['source_wh'] = (self._current_image.width, self._current_image.height)

        # 检测是否需要传递自定义色彩空间定义给worker进程
        custom_colorspace_def = None
        cs_name = self._current_params.input_color_space_name
        if self.color_space_manager.is_custom_color_space(cs_name):
            # 获取自定义色彩空间的完整定义（primaries, white_point, gamma）
            custom_colorspace_def = self.color_space_manager.get_color_space_definition(cs_name)
            if custom_colorspace_def:
                # 添加色彩空间名称
                custom_colorspace_def['name'] = cs_name

        # 发送预览请求（非阻塞），传递完整的proxy准备参数和显示状态
        print("[DEBUG] _trigger_preview_with_process: 发送预览请求（非阻塞）")
        self._preview_worker_process.request_preview(
            self._current_params,
            crop_rect_norm=crop_rect_norm,  # 始终为None
            orientation=self.get_current_orientation(),
            idt_gamma=self.get_current_idt_gamma(),
            convert_to_monochrome=self.should_convert_to_monochrome(),
            display_metadata=display_metadata,
            custom_colorspace_def=custom_colorspace_def
        )

        # 启动结果轮询定时器
        if not self._result_poll_timer.isActive():
            self._result_poll_timer.start(16)  # ~60 FPS

    def _trigger_preview_with_thread(self):
        """使用线程模式触发预览（原有实现）"""
        print("[DEBUG] _trigger_preview_with_thread() 开始执行", flush=True)

        if not self._current_proxy:
            print("[WARNING] _trigger_preview_with_thread(): _current_proxy为None，提前返回", flush=True)
            return

        if self._preview_busy:
            print(f"[WARNING] _trigger_preview_with_thread(): preview忙碌中，设置pending标志", flush=True)
            self._preview_pending = True
            return

        print("[DEBUG] _trigger_preview_with_thread(): 设置busy标志，准备创建worker", flush=True)
        self._preview_busy = True

        # 使用 view() 和 shallow_copy() 避免深拷贝
        try:
            proxy_view = self._current_proxy.view()
            params_view = self._current_params.shallow_copy()
            print(f"[DEBUG] _trigger_preview_with_thread(): proxy和params复制完成", flush=True)
        except Exception as e:
            print(f"[ERROR] _trigger_preview_with_thread(): 复制proxy/params失败: {e}", flush=True)
            self._preview_busy = False
            return

        # 获取 IDT gamma
        idt_gamma = self.get_current_idt_gamma()
        print(f"[DEBUG] _trigger_preview_with_thread(): IDT gamma={idt_gamma}", flush=True)

        # 检测是否需要传递自定义色彩空间定义
        custom_colorspace_def = None
        cs_name = self._current_params.input_color_space_name
        if self.color_space_manager.is_custom_color_space(cs_name):
            # 获取自定义色彩空间的完整定义（primaries, white_point, gamma）
            custom_colorspace_def = self.color_space_manager.get_color_space_definition(cs_name)
            if custom_colorspace_def:
                # 添加色彩空间名称
                custom_colorspace_def['name'] = cs_name
                print(f"[DEBUG] _trigger_preview_with_thread(): 自定义色彩空间={cs_name}", flush=True)

        print("[DEBUG] _trigger_preview_with_thread(): 创建PreviewWorker...", flush=True)
        worker = _PreviewWorker(
            image=proxy_view,
            params=params_view,
            the_enlarger=self.the_enlarger,
            color_space_manager=self.color_space_manager,
            convert_to_monochrome_in_idt=self.should_convert_to_monochrome(),
            orientation=self.get_current_orientation(),
            idt_gamma=idt_gamma,
            custom_colorspace_def=custom_colorspace_def
        )
        worker.signals.result.connect(self._on_preview_result)
        worker.signals.error.connect(self._on_preview_error)
        worker.signals.finished.connect(self._on_preview_finished)
        print("[DEBUG] _trigger_preview_with_thread(): Worker创建完成，信号已连接", flush=True)

        # 使用 ensure_thread_pool() 统一处理线程池创建
        print("[DEBUG] _trigger_preview_with_thread(): 提交worker到线程池", flush=True)
        try:
            self.ensure_thread_pool().start(worker)
            print("[DEBUG] _trigger_preview_with_thread(): Worker已提交到线程池", flush=True)
        except Exception as e:
            print(f"[ERROR] _trigger_preview_with_thread(): 提交worker失败: {e}", flush=True)
            self._preview_busy = False

    def _poll_preview_result(self):
        """定期轮询结果队列（~60 FPS）"""
        # print("[DEBUG] _poll_preview_result: 轮询结果定时触发")
        if self._preview_worker_process is None:
            self._result_poll_timer.stop()
            return

        result = self._preview_worker_process.try_get_result()

        if result is not None:
            if isinstance(result, Exception):
                # 错误处理
                self._on_preview_error(str(result))
            else:
                # 正常结果
                self._on_preview_result(result)

        # 不管是正常还是异常，这一轮 preview 算是结束了
        self._preview_busy = False

        if isinstance(result, Exception):
            self._on_preview_error(str(result))
        else:
            self._on_preview_result(result)

        # 看看有没有 pending 的请求
        if self._preview_pending:
            self._preview_pending = False
            self._trigger_preview_update()

    def _atexit_cleanup(self):
        """程序退出时的清理函数（atexit handler）

        确保 worker 进程和 shared memory 被正确清理
        """
        try:
            if hasattr(self, '_preview_worker_process'):
                self._shutdown_preview_worker_process()
        except:
            # atexit handler 中不应该抛出异常
            pass

    def cleanup(self):
        """清理 ApplicationContext 的资源，防止内存泄漏

        停止定时器，清理缓存，等待线程池完成
        这个方法应该在 ApplicationContext 不再使用时调用（如窗口关闭时）
        """
        print("[DEBUG] ApplicationContext.cleanup: 开始清理资源")

        # 0. 清理 worker 进程（如果使用进程隔离）
        if self._use_process_isolation:
            try:
                self._shutdown_preview_worker_process()
                print("[DEBUG] preview_worker_process 清理完成")
            except Exception as e:
                print(f"[WARNING] preview_worker_process 清理失败: {e}")

        # 1. 停止自动保存定时器
        try:
            if hasattr(self, '_autosave_timer') and self._autosave_timer:
                self._autosave_timer.stop()
                self._autosave_timer.deleteLater()
                self._autosave_timer = None
            print("[DEBUG] _autosave_timer 清理完成")
        except Exception as e:
            print(f"[WARNING] _autosave_timer 清理失败: {e}")

        # 2. 等待线程池完成当前任务
        try:
            if hasattr(self, 'thread_pool') and self.thread_pool:
                # 最多等待1秒，避免阻塞过久
                self.thread_pool.waitForDone(1000)
            print("[DEBUG] thread_pool 等待完成")
        except Exception as e:
            print(f"[WARNING] thread_pool 清理失败: {e}")

        # 3. 清理 ImageManager 缓存
        try:
            if hasattr(self, 'image_manager') and self.image_manager:
                self.image_manager.clear_cache()
            print("[DEBUG] image_manager 缓存清理完成")
        except Exception as e:
            print(f"[WARNING] image_manager 缓存清理失败: {e}")

        # 4. 清理 LUTProcessor 缓存
        try:
            if (hasattr(self, 'the_enlarger') and self.the_enlarger and
                hasattr(self.the_enlarger, 'lut_processor') and self.the_enlarger.lut_processor):
                self.the_enlarger.lut_processor.clear_cache()
            print("[DEBUG] lut_processor 缓存清理完成")
        except Exception as e:
            print(f"[WARNING] lut_processor 缓存清理失败: {e}")

        # 5. 清理 ColorSpaceManager 转换缓存
        try:
            if hasattr(self, 'color_space_manager') and self.color_space_manager:
                self.color_space_manager.clear_convert_cache()
            print("[DEBUG] color_space_manager 缓存清理完成")
        except Exception as e:
            print(f"[WARNING] color_space_manager 缓存清理失败: {e}")

        # 6. 显式释放大型对象引用
        try:
            if hasattr(self, '_current_image') and self._current_image:
                self._current_image = None
            if hasattr(self, '_current_proxy') and self._current_proxy:
                self._current_proxy = None
            print("[DEBUG] 大型对象引用清理完成")
        except Exception as e:
            print(f"[WARNING] 对象引用清理失败: {e}")

        print("[DEBUG] ApplicationContext.cleanup: 清理完成")

    def __del__(self):
        """析构函数：确保资源被清理

        作为最后一道防线，即使 cleanup() 没有被显式调用
        注意：__del__ 的调用时机不确定，不应该依赖它进行关键清理
        """
        try:
            # 只清理关键资源，避免在析构时做复杂操作
            if hasattr(self, '_autosave_timer') and self._autosave_timer:
                self._autosave_timer.stop()
        except Exception:
            # 析构函数中不应该抛出异常
            pass
