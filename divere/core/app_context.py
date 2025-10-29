from PySide6.QtCore import QObject, Signal, QRunnable, Slot, QThreadPool, QTimer
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


class _PreviewWorkerSignals(QObject):
    result = Signal(ImageData)
    error = Signal(str)
    finished = Signal()


class _PreviewWorker(QRunnable):
    def __init__(self, image: ImageData, params: ColorGradingParams, the_enlarger: TheEnlarger, 
                 color_space_manager: ColorSpaceManager, convert_to_monochrome_in_idt: bool = False):
        super().__init__()
        self.image = image
        self.params = params
        self.the_enlarger = the_enlarger
        self.color_space_manager = color_space_manager
        self.convert_to_monochrome_in_idt = convert_to_monochrome_in_idt
        self.signals = _PreviewWorkerSignals()

    @Slot()
    def run(self):
        try:
            # 传递monochrome转换参数
            monochrome_converter = None
            if self.convert_to_monochrome_in_idt:
                monochrome_converter = self.color_space_manager.convert_to_monochrome

            result_image = self.the_enlarger.apply_full_pipeline(
                self.image, self.params,
                convert_to_monochrome_in_idt=self.convert_to_monochrome_in_idt,
                monochrome_converter=monochrome_converter
            )
            result_image = self.color_space_manager.convert_to_display_space(result_image, "DisplayP3")
            self.signals.result.emit(result_image)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            self.signals.error.emit(f"{e}\n{tb}")
        finally:
            # 显式释放worker持有的对象以防止内存泄漏
            if hasattr(self, 'image') and self.image is not None:
                del self.image
            if hasattr(self, 'params') and self.params is not None:
                del self.params
            self.signals.finished.emit()


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
        
        # 为ColorSpaceManager添加ApplicationContext引用，以便CCMOptimizer访问共享数据
        self.color_space_manager.context = self

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
            params=self._current_params.copy(),
            orientation=0,
            crop_rect=None
        )
        self._per_crop_params: dict = {}
        
        # 连接文件夹导航信号
        self.folder_navigator.file_changed.connect(self.load_image)

    def _create_default_params(self) -> ColorGradingParams:
        params = ColorGradingParams()
        params.density_gamma = 2.6
        params.density_matrix = self.the_enlarger.pipeline_processor.get_density_matrix_array("Cineon_States_M_to_Print_Density")
        params.density_matrix_name = "Cineon_States_M_to_Print_Density"
        params.enable_density_matrix = True
        
        # 默认曲线改由 default preset 决定；此处不再强行加载硬编码曲线
            
        return params
    
    def _load_smart_default_preset(self, file_path: str):
        """使用智能预设加载器加载默认预设"""
        try:
            from divere.utils.smart_preset_loader import SmartPresetLoader
            loader = SmartPresetLoader()
            preset = loader.get_smart_default_preset(file_path)
            
            if preset:
                self.load_preset(preset)
                self.status_message_changed.emit(f"已应用智能分类默认设置")
            else:
                # 回退到通用默认
                self._load_generic_default_preset()
                
        except Exception as e:
            # 回退到通用默认
            self._load_generic_default_preset()
            self.status_message_changed.emit(f"智能分类失败，已应用通用默认设置: {e}")
    
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
            self._contactsheet_params = self._current_params.copy()
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
        self._prepare_proxy()
        self._trigger_preview_update()
        self._autosave_timer.start()

    def restore_crop_preview(self) -> None:
        if not self._current_image:
            return
        self._crop_focused = False
        self._prepare_proxy()
        self._trigger_preview_update()
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
            self._prepare_proxy()
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
            self._current_params = self._contactsheet_profile.params.copy()
            self._crop_focused = False
            self._active_crop_id = None
            # 移除orientation同步 - UI将直接读取contactsheet的orientation
            # 发送参数变更信号
            self.params_changed.emit(self._current_params)
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
                params = self._contactsheet_profile.params.copy()
                self._per_crop_params[crop_id] = params
            self._current_params = params.copy()
            self._crop_focused = False
            # 移除orientation同步 - UI将直接读取crop的orientation
            # 发送参数变更信号
            self.params_changed.emit(self._current_params)
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
                    params = self._contactsheet_profile.params.copy()
                else:
                    # 没有接触印相设置时，使用智能分类默认
                    if self._current_image:
                        self._load_smart_default_preset(self._current_image.file_path)
                        params = self._current_params.copy()
                    else:
                        # 没有图像时，使用通用默认
                        self._load_generic_default_preset()
                        params = self._current_params.copy()
                
                self._per_crop_params[crop_id] = params
            self._current_params = params.copy()
            # 移除orientation同步 - UI将直接读取crop的orientation
            # 直接进入聚焦
            self._crop_focused = True
            # 一次性刷新
            self.params_changed.emit(self._current_params)
            self._prepare_proxy(); self._trigger_preview_update()
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
            # 初始化该裁剪的参数集（深拷贝 contactsheet）
            self._per_crop_params[crop_id] = self._contactsheet_profile.params.copy()
            # 切到该裁剪 Profile（不聚焦）
            self.switch_to_crop(crop_id)
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
                    self._prepare_proxy()
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
            # 退出聚焦
            self._crop_focused = False
            # 触发预览刷新
            self._prepare_proxy(); self._trigger_preview_update()
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
            cs_params = self._contactsheet_profile.params.copy()
            print(f"DEBUG: 复制的contactsheet参数数量: {len(vars(cs_params))}")
            self._per_crop_params[active_id] = cs_params
            print("DEBUG: 参数已复制到per_crop_params")
            
            # 若当前正聚焦该裁剪，同步当前参数并刷新预览
            if self._crop_focused:
                print("DEBUG: 当前处于crop聚焦状态，同步参数")
                self._current_params = cs_params.copy()
                self.params_changed.emit(self._current_params)
                print("DEBUG: params_changed信号已发射")
                self._prepare_proxy(); self._trigger_preview_update()
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
                crop_params = self._current_params.copy()
                print("DEBUG: 从当前参数获取crop参数（聚焦状态）")
            elif active_id and active_id in self._per_crop_params:
                # 如果不在聚焦状态，从per_crop_params获取
                crop_params = self._per_crop_params[active_id].copy()
                print("DEBUG: 从per_crop_params获取crop参数（非聚焦状态）")
            else:
                print("DEBUG: 没有有效的crop参数可复制")
                return
                
            if not crop_params:
                print("DEBUG: crop参数为空")
                return
                
            print(f"DEBUG: 复制的crop参数数量: {len(vars(crop_params))}")
            # 复制参数到contactsheet
            self._contactsheet_profile.params = crop_params.copy()
            print("DEBUG: 参数已复制到contactsheet_params")
            
            # 如果当前处于contactsheet模式，同步当前参数并刷新预览
            if self._current_profile_kind == 'contactsheet':
                print("DEBUG: 当前处于contactsheet模式，同步参数")
                self._current_params = crop_params.copy()
                self.params_changed.emit(self._current_params)
                print("DEBUG: params_changed信号已发射")
                self._prepare_proxy(); self._trigger_preview_update()
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
            self.status_message_changed.emit(f"正在加载图像: {file_path}...")
            # 显式释放旧图像以防止内存泄漏
            if self._current_image is not None:
                del self._current_image
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
                self.status_message_changed.emit(f"检测到单色图像，已自动切换为黑白模式")
            
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
                self.status_message_changed.emit("已为图像加载自动预设（Bundle）")
                # Bundle内部已触发预览；此处不再重复
            else:
                preset = self.auto_preset_manager.get_preset_for_image(file_path)
                if preset:
                    self.load_preset(preset)
                    self.status_message_changed.emit(f"已为图像加载自动预设: {preset.name}")
                    
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
                            self.status_message_changed.emit("未找到预设，已应用通用默认预设")
                        except Exception:
                            self.reset_params()
                            self.status_message_changed.emit("未找到预设，已应用默认参数（回退）")


            # 清除加载标志并触发最终预览更新
            self._loading_image = False
            if self._current_image:
                self._prepare_proxy()
                self._trigger_preview_update()
            
            # 通知UI：图像已加载完成
            self.image_loaded.emit()

        except Exception as e:
            self._loading_image = False  # 确保异常情况下也清除标志
            import traceback
            print(traceback.format_exc())
            self.status_message_changed.emit(f"无法加载图像: {e}")

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
            new_params.density_curve_name = preset.density_curve.name
            # 若预设自带点，但为单位曲线 [(0,0),(1,1)]，按“无点”处理并尝试按名称加载
            def _is_identity_curve(points):
                try:
                    return (
                        isinstance(points, (list, tuple)) and len(points) == 2 and
                        float(points[0][0]) == 0.0 and float(points[0][1]) == 0.0 and
                        float(points[1][0]) == 1.0 and float(points[1][1]) == 1.0
                    )
                except Exception:
                    return False

            has_points = bool(preset.density_curve.points)
            if has_points and not _is_identity_curve(preset.density_curve.points):
                new_params.curve_points = preset.density_curve.points
                new_params.enable_density_curve = True
            else:
                # 无点或单位曲线：尝试根据名称加载实际曲线
                try:
                    loaded = self._load_density_curve_points_by_name(preset.density_curve.name)
                    if loaded:
                        if 'RGB' in loaded and loaded['RGB']:
                            new_params.curve_points = loaded['RGB']
                        if 'R' in loaded and loaded['R']:
                            new_params.curve_points_r = loaded['R']
                        if 'G' in loaded and loaded['G']:
                            new_params.curve_points_g = loaded['G']
                        if 'B' in loaded and loaded['B']:
                            new_params.curve_points_b = loaded['B']
                        new_params.enable_density_curve = True
                except Exception:
                    pass
        
        # 在更新参数前先设置orientation，避免预览闪烁
        try:
            self._contactsheet_profile.orientation = preset.orientation
        except Exception:
            pass
        
        # 确保在更新参数前先准备好proxy，避免使用旧proxy导致预览暗淡
        if self._current_image:
            self._prepare_proxy()
        
        self.update_params(new_params)
        self._contactsheet_profile.params = self._current_params.copy()

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
                    params.density_curve_name = entry.preset.density_curve.name
                    def _is_identity_curve(points):
                        try:
                            return (
                                isinstance(points, (list, tuple)) and len(points) == 2 and
                                float(points[0][0]) == 0.0 and float(points[0][1]) == 0.0 and
                                float(points[1][0]) == 1.0 and float(points[1][1]) == 1.0
                            )
                        except Exception:
                            return False
                    has_points = bool(entry.preset.density_curve.points)
                    if has_points and not _is_identity_curve(entry.preset.density_curve.points):
                        params.curve_points = entry.preset.density_curve.points
                        params.enable_density_curve = True
                    else:
                        try:
                            loaded = self._load_density_curve_points_by_name(entry.preset.density_curve.name)
                            if loaded:
                                if 'RGB' in loaded and loaded['RGB']:
                                    params.curve_points = loaded['RGB']
                                if 'R' in loaded and loaded['R']:
                                    params.curve_points_r = loaded['R']
                                if 'G' in loaded and loaded['G']:
                                    params.curve_points_g = loaded['G']
                                if 'B' in loaded and loaded['B']:
                                    params.curve_points_b = loaded['B']
                                params.enable_density_curve = True
                        except Exception:
                            pass

                self._per_crop_params[cid] = params
            # 活跃裁剪（仅记录，不自动聚焦）
            self._active_crop_id = bundle.active_crop_id
            self._crop_focused = False
            self._current_profile_kind = 'contactsheet'
            self._prepare_proxy(); self._trigger_preview_update()
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
        self._current_params = new_params
        # 同步到当前 profile 存根
        try:
            if self._current_profile_kind == 'contactsheet':
                self._contactsheet_profile.params = new_params.copy()
            elif self._current_profile_kind == 'crop' and self._active_crop_id is not None:
                self._per_crop_params[self._active_crop_id] = new_params.copy()
        except Exception:
            pass
        self.params_changed.emit(self._current_params)
        self._trigger_preview_update()
        self._autosave_timer.start() # 每次参数变更都启动自动保存计时器
    
    def set_input_color_space(self, space_name: str):
        self._current_params.input_color_space_name = space_name
        self.params_changed.emit(self._current_params)
        if self._current_image:
            self._prepare_proxy()
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
                QTimer.singleShot(0, lambda: self._ensure_ui_state_sync_for_monochrome())
        
        # 根据参数决定是否应用胶片类型的默认配置
        if apply_defaults:
            self._current_params = self.film_type_controller.apply_film_type_defaults(
                self._current_params, film_type, force_apply=force_apply_defaults
            )
        
        # 触发参数更新和预览刷新
        self.params_changed.emit(self._current_params)
        if self._current_image and not getattr(self, '_loading_image', False):
            self._prepare_proxy()
            self._trigger_preview_update()
    
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
            self._prepare_proxy()
            self._trigger_preview_update()
        
        self.status_message_changed.emit(f"已转换为黑白模式: {self.film_type_controller.get_film_type_display_name(new_film_type)}")
    
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
                self.status_message_changed.emit(f"已加载 {film_type} 胶片类型的默认预设")
            else:
                raise ValueError(f"无法加载胶片类型 '{film_type}' 的默认预设")
        except Exception as e:
            self.status_message_changed.emit(f"加载胶片类型默认预设失败: {e}")
            # 回退到通用默认预设
            try:
                from divere.utils.defaults import load_default_preset
                self.load_preset(load_default_preset())
                self.status_message_changed.emit("已回退到通用默认预设")
            except Exception as fallback_error:
                self.status_message_changed.emit(f"加载默认预设失败: {fallback_error}")

    def reset_params(self):
        """重置参数：根据当前图像类型选择智能默认预设"""
        # 保存当前orientation，避免被预设重置
        saved_orientation = self._contactsheet_profile.orientation
        
        if self._current_image:
            # 有图像时，使用智能分类器选择默认预设
            try:
                self._load_smart_default_preset(self._current_image.file_path)
                self.status_message_changed.emit("参数已重置为智能分类默认预设")
            except Exception:
                # 智能分类失败时，回退到通用默认
                try:
                    from divere.utils.defaults import load_default_preset
                    self.load_preset(load_default_preset())
                    self.status_message_changed.emit("参数已重置为通用默认预设")
                except Exception:
                    self._current_params = self._create_default_params()
                    self._contactsheet_profile.params = self._current_params.copy()
                    self.params_changed.emit(self._current_params)
                    self.status_message_changed.emit("参数已重置（回退内部默认）")
        else:
            # 没有图像时，使用通用默认
            try:
                self._load_generic_default_preset()
                self.status_message_changed.emit("参数已重置为通用默认预设")
            except Exception:
                self._current_params = self._create_default_params()
                self._contactsheet_profile.params = self._current_params.copy()
                self.params_changed.emit(self._current_params)
                self.status_message_changed.emit("参数已重置（回退内部默认）")
        
        # 恢复保存的orientation
        self._contactsheet_profile.orientation = saved_orientation
        
        # 重新更新预览，确保使用正确的orientation
        if self._current_image:
            self._prepare_proxy()
            self._trigger_preview_update()

    def set_current_as_folder_default(self):
        """将当前参数设置为文件夹默认设置"""
        if not self._current_image:
            self.status_message_changed.emit("请先加载图像")
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
                # 新增：保存pipeline控制状态，这些对于folder_default至关重要
                "enable_density_matrix": self._current_params.enable_density_matrix,
                "enable_density_curve": self._current_params.enable_density_curve,
                "enable_rgb_gains": self._current_params.enable_rgb_gains,
                "enable_density_inversion": self._current_params.enable_density_inversion
            }
            
            # 通过AutoPresetManager保存folder_default
            self.auto_preset_manager.set_active_directory(str(image_path.parent))
            self.auto_preset_manager.save_folder_default(idt_data, cc_params_data)
            
            self.status_message_changed.emit(f"已将当前设置保存为文件夹默认设置")
            
        except Exception as e:
            self.status_message_changed.emit(f"保存文件夹默认设置失败: {e}")

    def run_auto_color_correction(self, get_preview_callback):
        """执行AI自动白平衡"""
        # 黑白模式下跳过RGB gains调整
        if self.is_monochrome_type():
            pipeline_config = self.film_type_controller.get_pipeline_config(self._current_film_type)
            if not pipeline_config.enable_rgb_gains:
                self.status_message_changed.emit("黑白胶片模式下RGB自动校色功能已禁用")
                return
        
        preview_image = get_preview_callback()
        if preview_image is None or preview_image.array is None:
            self.status_message_changed.emit("自动校色失败：无预览图像")
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
            new_params = self._current_params.copy()
            new_params.rgb_gains = tuple(new_gains)
            self.update_params(new_params)
            
            # 根据图像类型显示不同的消息
            if self.is_monochrome_type():
                # 黑白图像：只显示灰度增益
                self.status_message_changed.emit(
                    f"AI自动校色完成. ΔGain(×γ/2): 灰度={delta[0]:.2f}"
                )
            else:
                # 彩色图像：显示RGB增益
                self.status_message_changed.emit(
                    f"AI自动校色完成. ΔGains(×γ/2): R={delta[0]:.2f}, G={delta[1]:.2f}, B={delta[2]:.2f}"
                )
        except Exception as e:
            self.status_message_changed.emit(f"AI自动校色失败: {e}")

    def run_iterative_auto_color(self, get_preview_callback, max_iterations=10):
        """执行迭代式AI自动白平衡"""
        # 黑白模式下跳过RGB gains调整
        if self.is_monochrome_type():
            pipeline_config = self.film_type_controller.get_pipeline_config(self._current_film_type)
            if not pipeline_config.enable_rgb_gains:
                self.status_message_changed.emit("黑白胶片模式下RGB迭代校色功能已禁用")
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
                self.status_message_changed.emit("黑白胶片模式下RGB迭代校色已中止")
                self._auto_color_iterations = 0
                self._get_preview_for_auto_color_callback = None
                return

        preview_image = self._get_preview_for_auto_color_callback()
        if preview_image is None or preview_image.array is None:
            self.status_message_changed.emit("自动校色迭代中止：无预览图像")
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
                self.status_message_changed.emit("AI自动校色收敛，已停止")
                self._auto_color_iterations = 0
                self._get_preview_for_auto_color_callback = None
                self._autosave_timer.start() # Save on convergence
                return

            # 低侵入：改用统一入口，确保写入当前 profile 存根并触发 autosave
            new_params = self._current_params.copy()
            new_params.rgb_gains = tuple(new_gains)
            self.update_params(new_params)  # 将触发预览；_on_preview_result 会调度下一次迭代
            self.status_message_changed.emit(f"AI校色迭代剩余: {self._auto_color_iterations}")

        except Exception as e:
            self.status_message_changed.emit(f"AI自动校色迭代失败: {e}")
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
            self.status_message_changed.emit("中性色定义完成")
            self._autosave_timer.start()
            return

        # 获取当前preview图像（DisplayP3空间）
        preview_image = self._neutral_point_callback()
        if preview_image is None or preview_image.array is None:
            self.status_message_changed.emit("中性色定义中止：无preview图像")
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
                self.status_message_changed.emit(f"中性色已收敛 (比值差异={max_ratio_diff:.4f})")
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
            new_params = self._current_params.copy()
            new_params.rgb_gains = (new_r_gain, current_gains[1], new_b_gain)

            self._neutral_point_iterations -= 1
            self.update_params(new_params)
            self.status_message_changed.emit(
                f"中性色迭代中... 剩余{self._neutral_point_iterations}次 (目标={white_point}K, 比值差异={max_ratio_diff:.4f})"
            )

        except Exception as e:
            self.status_message_changed.emit(f"中性色定义迭代失败: {e}")
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
                self.status_message_changed.emit("黑白胶片模式下RGB增益调整功能已禁用")
                return

        # 初始化迭代状态
        self._neutral_point_iterations = 8  # 最多8次
        self._neutral_point_norm = (norm_x, norm_y)
        self._neutral_point_callback = get_preview_callback
        self._neutral_point_sample_size = 5
        self._neutral_point_white_point = white_point  # 保存色温参数

        self.status_message_changed.emit(f"开始中性色定义迭代 (色温={white_point}K)...")

        # 开始第一次迭代
        self._perform_neutral_point_iteration()

    def _prepare_proxy(self):
        """准备proxy图像，应用标准变换链：裁剪 → 旋转"""
        if not self._current_image:
            return
        
        # 源图（用于变换）
        src_image = self._current_image
        orig_h, orig_w = src_image.height, src_image.width

        # === 标准变换链：Step 1 - 裁剪（若聚焦） ===
        if self._crop_focused:
            crop_instance = self.get_active_crop_instance()
            if crop_instance and crop_instance.rect_norm and src_image.array is not None:
                try:
                    x, y, w, h = crop_instance.rect_norm
                    x0 = int(round(x * orig_w)); y0 = int(round(y * orig_h))
                    x1 = int(round((x + w) * orig_w)); y1 = int(round((y + h) * orig_h))
                    x0 = max(0, min(orig_w - 1, x0)); x1 = max(x0 + 1, min(orig_w, x1))
                    y0 = max(0, min(orig_h - 1, y0)); y1 = max(y0 + 1, min(orig_h, y1))
                    cropped_arr = src_image.array[y0:y1, x0:x1, :].copy()
                    src_image = src_image.copy_with_new_array(cropped_arr)
                except Exception:
                    pass
            # 接触印相聚焦：无激活 crop，但存在 contactsheet 裁剪矩形
            elif (self._active_crop_id is None and self._contactsheet_profile.crop_rect is not None and src_image.array is not None):
                try:
                    x, y, w, h = self._contactsheet_profile.crop_rect
                    x0 = int(round(x * orig_w)); y0 = int(round(y * orig_h))
                    x1 = int(round((x + w) * orig_w)); y1 = int(round((y + h) * orig_h))
                    x0 = max(0, min(orig_w - 1, x0)); x1 = max(x0 + 1, min(orig_w, x1))
                    y0 = max(0, min(orig_h - 1, y0)); y1 = max(y0 + 1, min(orig_h, y1))
                    cropped_arr = src_image.array[y0:y1, x0:x1, :].copy()
                    src_image = src_image.copy_with_new_array(cropped_arr)
                except Exception:
                    pass

        # === 标准变换链：Step 2 - 生成代理 ===
        proxy = self.image_manager.generate_proxy(
            src_image,
            self.the_enlarger.preview_config.get_proxy_size_tuple()
        )

        # === 标准变换链：Step 3 - 前置IDT Gamma（幂次变换） ===
        idt_gamma = self.get_current_idt_gamma()
        if abs(idt_gamma - 1.0) > 1e-6 and proxy.array is not None:
            try:
                proxy_array = self.the_enlarger.pipeline_processor.math_ops.apply_power(
                    proxy.array, idt_gamma, use_optimization=True
                )
                proxy = proxy.copy_with_new_array(proxy_array)
            except Exception:
                pass

        # === 标准变换链：Step 4 - 色彩空间转换（跳过逆伽马） ===
        proxy = self.color_space_manager.set_image_color_space(
            proxy, self._current_params.input_color_space_name
        )
        # 显式释放旧的proxy对象以防止内存泄漏
        if self._current_proxy is not None:
            del self._current_proxy
        self._current_proxy = self.color_space_manager.convert_to_working_space(
            proxy, skip_gamma_inverse=True
        )

        # Monochrome转换现在在pipeline processor的IDT阶段进行

        # === 标准变换链：Step 5 - 旋转 ===
        # 分离的旋转逻辑：crop focused时使用crop的orientation，否则使用contactsheet orientation
        effective_orientation = self.get_current_orientation()  # 获取当前profile的orientation
        
        if self._crop_focused:
            # 聚焦状态：使用crop的独立orientation
            crop_instance = self.get_active_crop_instance()
            if crop_instance:
                effective_orientation = crop_instance.orientation
        
        # 应用旋转到代理图像
        if (self._current_proxy and self._current_proxy.array is not None and
            effective_orientation % 360 != 0):
            try:
                import numpy as np
                k = (effective_orientation // 90) % 4
                if k != 0:
                    rotated = np.rot90(self._current_proxy.array, k=int(k))
                    # 保存旧对象引用以便释放
                    old_proxy = self._current_proxy
                    self._current_proxy = old_proxy.copy_with_new_array(rotated)
                    # 显式释放旧对象以防止内存泄漏
                    del old_proxy
            except Exception:
                pass
        
        # === 标准变换链：Step 6 - 注入裁剪可视化元数据（供PreviewWidget绘制） ===
        try:
            md = self._current_proxy.metadata
            md['source_wh'] = (int(orig_w), int(orig_h))
            md['orientation'] = int(effective_orientation) % 360  # Add orientation to metadata
            
            # 传递CropInstance信息到UI层
            crop_instance = self.get_active_crop_instance()
            if crop_instance:
                md['crop_overlay'] = crop_instance.rect_norm
                md['crop_instance'] = crop_instance  # 传递完整的crop实例
            else:
                # 无正式裁剪时，传递 contactsheet 裁剪（若有）
                md['crop_overlay'] = self._contactsheet_profile.crop_rect
                # 若此时处于"接触印相聚焦"，为坐标换算提供一个临时的 CropInstance
                if self._crop_focused and self._contactsheet_profile.crop_rect is not None:
                    try:
                        md['crop_instance'] = CropInstance(
                            id='contactsheet_focus',
                            name='接触印相聚焦',
                            rect_norm=self._contactsheet_profile.crop_rect,
                            orientation=int(self._contactsheet_profile.orientation) % 360
                        )
                    except Exception:
                        md['crop_instance'] = None
                else:
                    md['crop_instance'] = None
                
            md['crop_focused'] = bool(self._crop_focused)
            md['active_crop_id'] = self._active_crop_id
            md['global_orientation'] = int(self._contactsheet_profile.orientation)  # contactsheet orientation
        except Exception:
            pass

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
        if not self._current_proxy:
            return

        if self._preview_busy:
            self._preview_pending = True
            return

        self._preview_busy = True
        
        
        from copy import deepcopy
        proxy_copy = self._current_proxy.copy()
        params_copy = deepcopy(self._current_params)

        worker = _PreviewWorker(
            image=proxy_copy,
            params=params_copy,
            the_enlarger=self.the_enlarger,
            color_space_manager=self.color_space_manager,
            convert_to_monochrome_in_idt=self.should_convert_to_monochrome()
        )
        worker.signals.result.connect(self._on_preview_result)
        worker.signals.error.connect(self._on_preview_error)
        worker.signals.finished.connect(self._on_preview_finished)
        self.thread_pool.start(worker)

    def _on_preview_result(self, result_image: ImageData):
        self.preview_updated.emit(result_image)
        # If an iterative auto color is in progress, trigger the next step
        if self._auto_color_iterations > 0 and self._get_preview_for_auto_color_callback:
            QTimer.singleShot(0, self._perform_auto_color_iteration)
        # If neutral point iteration is in progress, trigger the next step
        if self._neutral_point_iterations > 0 and self._neutral_point_callback:
            QTimer.singleShot(0, self._perform_neutral_point_iteration)


    def _on_preview_error(self, message: str):
        # 在色卡优化期间不发送预览错误消息，避免覆盖优化状态
        if not self._ccm_optimization_active:
            self.status_message_changed.emit(f"预览更新失败: {message}")
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
        self._preview_busy = False
        if self._preview_pending:
            self._preview_pending = False
            self._trigger_preview_update()

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
            deg = int(degrees) % 360
            # 规范到 0/90/180/270
            choices = [0, 90, 180, 270]
            normalized = min(choices, key=lambda x: abs(x - deg))
            
            # 直接写入对应的数据源
            if self._current_profile_kind == 'contactsheet':
                self._contactsheet_profile.orientation = normalized
            elif self._active_crop_id:
                crop = self.get_active_crop_instance()
                if crop:
                    crop.orientation = normalized
            
            # 触发预览更新
            if self._current_image:
                self._prepare_proxy()
                self._trigger_preview_update()
                self._autosave_timer.start()
        except Exception:
            pass
    
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
            step = 90 if int(direction) >= 0 else -90
            
            if self._crop_focused or self._current_profile_kind == 'crop':
                # 聚焦或裁剪Profile下：只旋转当前crop的orientation
                crop_instance = self.get_active_crop_instance()
                if crop_instance:
                    new_orientation = (crop_instance.orientation + step) % 360
                    # 仅更新当前裁剪的方向，保留其它裁剪
                    self.update_active_crop_orientation(new_orientation)
                    self._prepare_proxy()
                    self._trigger_preview_update()
                    self._autosave_timer.start()
            else:
                # 非聚焦状态：只旋转contactsheet orientation（不影响crop）
                current_orientation = self.get_current_orientation()
                new_deg = (current_orientation + step) % 360
                self.set_orientation(new_deg)
                # 注意：不同步crop的orientation，保持完全分离
            
            # 发射旋转完成信号，让MainWindow知道需要fit to window
            self.rotation_completed.emit()
        except Exception:
            pass

    def update_active_crop_orientation(self, orientation: int) -> None:
        """仅更新当前活跃裁剪的 orientation，保留所有裁剪与激活状态。"""
        try:
            crop_instance = self.get_active_crop_instance()
            if crop_instance:
                crop_instance.orientation = int(orientation) % 360
        except Exception:
            pass

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
            self.status_message_changed.emit(f"重新加载curves配置失败: {e}")

    def reload_all_configs(self):
        """重新加载所有配置文件"""
        try:
            # 重新加载色彩空间配置
            self.color_space_manager.reload_config()
            
            # 重新加载矩阵配置
            self.the_enlarger.pipeline_processor.reload_matrices()
            
            # 重新加载curves配置
            self.reload_curves_config()
            
            self.status_message_changed.emit("配置文件已重新加载")
        except Exception as e:
            self.status_message_changed.emit(f"重新加载配置失败: {e}")
    
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
            
            self.status_message_changed.emit(f"Proxy长边尺寸已更新为: {size}")
        except Exception as e:
            self.status_message_changed.emit(f"更新Proxy尺寸失败: {e}")

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
                'current_params': self._current_params.copy() if self._current_params else None,
                'current_film_type': self._current_film_type,
                
                # 裁剪相关状态
                'crops': [crop for crop in self._crops],  # 浅拷贝CropInstance列表
                'active_crop_id': self._active_crop_id,
                'crop_focused': self._crop_focused,
                'current_profile_kind': self._current_profile_kind,
                'contactsheet_profile': self._contactsheet_profile.copy() if hasattr(self._contactsheet_profile, 'copy') else self._contactsheet_profile,
                'per_crop_params': {k: v.copy() if v else None for k, v in self._per_crop_params.items()},
                
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
                self._current_params = backup['current_params'].copy()
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
            
        except Exception as e:
            print(f"恢复状态失败: {e}")
