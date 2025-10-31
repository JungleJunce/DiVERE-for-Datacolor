"""
主窗口界面
"""

import sys
import json
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QMenuBar, QToolBar, QStatusBar, QFileDialog, QMessageBox,
    QSplitter, QLabel, QDockWidget, QDialog, QApplication, QPushButton,
    QInputDialog
)
from PySide6.QtCore import Qt, QTimer, QObject, Signal, QRunnable, Slot, QThreadPool
from PySide6.QtGui import QAction, QKeySequence
import numpy as np

from divere.core.app_context import ApplicationContext
from divere.core.data_types import ImageData, ColorGradingParams, Preset, InputTransformationDefinition, MatrixDefinition, CurveDefinition, CropAddDirection
from divere.utils.enhanced_config_manager import enhanced_config_manager
from divere.utils.preset_manager import PresetManager, apply_preset_to_params
from divere.utils.auto_preset_manager import AutoPresetManager
from divere.utils.spectral_sharpening import run as run_spectral_sharpening

from .preview_widget import PreviewWidget
from .save_dialog import SaveImageDialog
from .parameter_panel import ParameterPanel
from .theme import apply_theme, current_theme
from .cmaes_progress_dialog import CMAESProgressDialog
from .shortcuts import ShortcutsBinder, install_ime_brackets_fallback
from .shortcut_help_dialog import ShortcutHelpDialog


class MainWindow(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()

        # 初始化核心组件
        # 使用 parent=None 而不是 self，避免循环引用 MainWindow ↔ ApplicationContext
        # ApplicationContext 不需要访问 parent()，所以不设置parent也不影响功能
        self.context = ApplicationContext(parent=None)

        # 导航相关属性
        self.image_file_list = []  # 当前文件夹中的图片文件列表
        self.current_image_index = -1  # 当前图片在列表中的索引
        self.current_image_path = None  # 当前图片的路径

        # 中性点色温参数
        self._neutral_point_white_point = 5500  # 默认色温5500K

        # 设置窗口
        self.setWindowTitle("DiVERE - 数字彩色放大机")
        self.setGeometry(100, 100, 1400, 900)
        
        # 创建界面
        self._create_ui()
        self._create_menus()
        self._create_toolbar()
        self._create_statusbar()
        self._connect_context_signals()
        self._connect_panel_signals()

        # 主题：启动时应用上次选择
        try:
            app = QApplication.instance()
            saved_theme = enhanced_config_manager.get_ui_setting("theme", "dark")
            apply_theme(app, saved_theme)
        except Exception as _:
            pass
        
        # 初始化默认色彩空间 - 逻辑迁移到 Context
        # self._initialize_color_space_info()
        
        # 实时预览更新 - 逻辑迁移到 Context
        # self.preview_timer = QTimer()
        # self.preview_timer.timeout.connect(self._update_preview)
        # self.preview_timer.setSingleShot(True)
        # self.preview_timer.setInterval(10)  # 10ms延迟，超快响应
        
        # 拖动状态跟踪
        self.is_dragging = False
        # 首次加载后在首帧预览到达时适应窗口
        self._fit_after_next_preview: bool = False
        # 控制fit时机的新标志
        self._should_fit_after_image_load: bool = False
        self._should_fit_after_rotation: bool = False
        
        # 预览后台线程池 - 逻辑迁移到 Context
        # self.thread_pool: QThreadPool = QThreadPool.globalInstance()
        # 限制为1，防止堆积；配合"忙碌/待处理"标志实现去抖
        try:
            self.context.thread_pool.setMaxThreadCount(1)
        except Exception:
            pass
        self._preview_busy: bool = False
        self._preview_pending: bool = False
        self._preview_seq_counter: int = 0

        # CCM 优化相关的临时引用（避免闭包内存泄漏）
        self._ccm_worker: Optional[QRunnable] = None
        self._ccm_progress_dialog: Optional[object] = None  # CMAESProgressDialog

        # 最后，初始化参数面板的默认值
        self.parameter_panel.initialize_defaults(self.context.get_current_params())
        
        # 自动加载测试图像（可选）
        # self._load_demo_image()

        ## 快捷键支持
        # 一行完成：注册快捷键 + 安装【】兜底
        self._binder = ShortcutsBinder(self)
        self._binder.setup_default_shortcuts()
        self._ime_filter = install_ime_brackets_fallback(self._binder, install_on_app=True)
        
    def _apply_crop_and_rotation_for_export(self, src_image: ImageData, rect_norm: Optional[tuple], orientation_deg: int) -> ImageData:
        """按导出标准链路应用裁剪与旋转：先裁剪再旋转。"""
        try:
            out = src_image
            # 裁剪
            if rect_norm and out and out.array is not None:
                x, y, w, h = rect_norm
                H, W = out.height, out.width
                x0 = int(round(x * W)); y0 = int(round(y * H))
                x1 = int(round((x + w) * W)); y1 = int(round((y + h) * H))
                x0 = max(0, min(W - 1, x0)); x1 = max(x0 + 1, min(W, x1))
                y0 = max(0, min(H - 1, y0)); y1 = max(y0 + 1, min(H, y1))
                cropped = out.array[y0:y1, x0:x1, :].copy()
                out = out.copy_with_new_array(cropped)
            # 旋转（逆时针）
            deg = int(orientation_deg) % 360
            if deg != 0 and out and out.array is not None:
                k = (deg // 90) % 4
                if k:
                    out = out.copy_with_new_array(np.rot90(out.array, k=int(k)))
            return out
        except Exception:
            return src_image

    def _convert_to_grayscale_if_bw_mode(self, image: ImageData) -> ImageData:
        """Convert image to grayscale if current film type is B&W mode."""
        try:
            # Check if current film type is monochrome
            current_film_type = self.context.get_current_film_type()
            if not self.context.film_type_controller.is_monochrome_type(current_film_type):
                return image  # Not B&W mode, return unchanged
            
            if image.array is None or image.array.ndim != 3 or image.array.shape[2] != 3:
                return image  # Not RGB format, return unchanged
            
            # Convert RGB to grayscale using ITU-R BT.709 weights
            # Same formula as used in preview_widget.py
            luminance = (0.2126 * image.array[:, :, 0] + 
                        0.7152 * image.array[:, :, 1] + 
                        0.0722 * image.array[:, :, 2])
            
            # Create single-channel grayscale array
            grayscale_array = luminance[:, :, np.newaxis]
            
            # Return new ImageData with grayscale array
            return image.copy_with_new_array(grayscale_array)
            
        except Exception as e:
            print(f"Grayscale conversion failed: {e}")
            return image  # Return original on error
    
    def _connect_context_signals(self):
        """连接 ApplicationContext 的信号到UI槽函数"""
        self.context.preview_updated.connect(self._on_preview_updated)
        self.context.status_message_changed.connect(self.statusBar().showMessage)
        self.context.image_loaded.connect(self._on_image_loaded)
        self.context.autosave_requested.connect(self._on_autosave_requested)
        # 连接curves配置重载信号
        self.context.curves_config_reloaded.connect(self._on_curves_config_reloaded)
        # 连接旋转完成信号
        self.context.rotation_completed.connect(self._on_rotation_completed)
        try:
            # 使用命名方法替代 lambda 闭包，避免内存泄漏
            self.context.preview_updated.connect(self._on_preview_updated_for_contactsheet)
        except Exception:
            pass
        try:
            self.preview_widget.request_focus_contactsheet.connect(self._on_request_focus_contactsheet)
        except Exception:
            pass

    def _connect_panel_signals(self):
        """连接ParameterPanel的信号"""
        self.parameter_panel.auto_color_requested.connect(self._on_auto_color_requested)
        self.parameter_panel.auto_color_iterative_requested.connect(self._on_auto_color_iterative_requested)
        self.parameter_panel.pick_neutral_point_requested.connect(self._on_pick_neutral_point_requested)
        self.parameter_panel.apply_neutral_color_requested.connect(self._on_apply_neutral_color_requested)
        self.parameter_panel.neutral_white_point_changed.connect(self._on_neutral_white_point_changed)
        self.parameter_panel.ccm_optimize_requested.connect(self._on_ccm_optimize_requested)
        self.parameter_panel.save_custom_colorspace_requested.connect(self._on_save_custom_colorspace_requested)
        self.parameter_panel.save_density_matrix_requested.connect(self._on_save_density_matrix_requested)
        self.parameter_panel.save_colorchecker_colors_requested.connect(self._on_save_colorchecker_colors_requested)
        self.parameter_panel.toggle_color_checker_requested.connect(self.preview_widget.toggle_color_checker)
        # 色卡变换信号连接
        self.parameter_panel.cc_flip_horizontal_requested.connect(self.preview_widget.flip_colorchecker_horizontal)
        self.parameter_panel.cc_flip_vertical_requested.connect(self.preview_widget.flip_colorchecker_vertical)
        self.parameter_panel.cc_rotate_left_requested.connect(self.preview_widget.rotate_colorchecker_left)
        self.parameter_panel.cc_rotate_right_requested.connect(self.preview_widget.rotate_colorchecker_right)
        # 色卡类型变化信号连接
        self.parameter_panel.colorchecker_changed.connect(self.preview_widget.on_colorchecker_changed)
        # 清除ApplicationContext的reference color缓存以确保数据一致性
        self.parameter_panel.colorchecker_changed.connect(self.context.clear_reference_color_cache)
        # 屏幕反光补偿交互信号连接
        self.parameter_panel.glare_compensation_interaction_started.connect(self._on_glare_compensation_interaction_started)
        self.parameter_panel.glare_compensation_interaction_ended.connect(self._on_glare_compensation_interaction_ended)
        # Proxy尺寸变化信号连接
        self.parameter_panel.proxy_size_changed.connect(self.context.update_proxy_max_size)
        self.parameter_panel.glare_compensation_realtime_update.connect(self._on_glare_compensation_realtime_update)
        # 当 UCS 三角拖动结束：注册/切换到一个临时 custom 输入空间，触发代理重建与预览
        self.parameter_panel.custom_primaries_changed.connect(self._on_custom_primaries_changed)
        # LUT导出信号
        self.parameter_panel.lut_export_requested.connect(self._on_lut_export_requested)
        # 预览裁剪交互
        self.preview_widget.crop_committed.connect(self._on_crop_committed)
        # 中性点选择
        self.preview_widget.neutral_point_selected.connect(self._on_neutral_point_selected)
        # 单张裁剪（不创建正式crop项）
        try:
            self.preview_widget.single_crop_committed.connect(self._on_single_crop_committed)
        except Exception:
            pass
        self.preview_widget.crop_updated.connect(self._on_crop_updated)
        self.preview_widget.request_focus_crop.connect(self._on_request_focus_crop)
        self.preview_widget.request_restore_crop.connect(self._on_request_restore_crop)
        # 裁剪选择条信号
        try:
            self.preview_widget.request_switch_profile.connect(self._on_request_switch_profile)
            self.preview_widget.request_new_crop.connect(self._on_request_new_crop)
            self.preview_widget.request_delete_crop.connect(self._on_request_delete_crop)
        except Exception:
            pass
        # Context → UI：裁剪改变后刷新 overlay
        try:
            self.context.crop_changed.connect(self.preview_widget.set_crop_overlay)
        except Exception:
            pass

    def _create_ui(self):
        """创建用户界面"""
        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 创建分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧参数面板
        self.parameter_panel = ParameterPanel(self.context)
        self.parameter_panel.parameter_changed.connect(self.on_parameter_changed)
        self.parameter_panel.input_colorspace_changed.connect(self.on_input_colorspace_changed)
        self.parameter_panel.film_type_changed.connect(self.on_film_type_changed)
        
        # Connect ApplicationContext signals
        self.context.film_type_changed.connect(self.on_context_film_type_changed)
        parameter_dock = QDockWidget("调色参数", self)
        parameter_dock.setWidget(self.parameter_panel)
        parameter_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, parameter_dock)
        
        # 中央预览区域
        self.preview_widget = PreviewWidget(self.context)
        self.preview_widget.image_rotated.connect(self._on_image_rotated)
        splitter.addWidget(self.preview_widget)
        
        # 设置分割器比例
        splitter.setSizes([300, 800])
        

    
    def _create_menus(self):
        """创建菜单栏"""
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件")
        
        # 打开图像
        open_action = QAction("打开图像", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self._open_image)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()

        # 加载预设
        load_preset_action = QAction("导入预设...", self)
        load_preset_action.setToolTip("从文件导入预设并应用到当前图像")
        load_preset_action.triggered.connect(self._load_preset)
        file_menu.addAction(load_preset_action)

        # 保存预设
        save_preset_action = QAction("导出预设...", self)
        save_preset_action.setToolTip("将当前参数导出为预设文件")
        save_preset_action.triggered.connect(self._save_preset)
        file_menu.addAction(save_preset_action)
        
        file_menu.addSeparator()
        
        # 选择输入色彩变换
        colorspace_action = QAction("设置输入色彩变换", self)
        colorspace_action.triggered.connect(self._select_input_color_space)
        file_menu.addAction(colorspace_action)
        
        # 设置工作色彩空间
        working_space_action = QAction("设置工作色彩空间", self)
        working_space_action.triggered.connect(self._select_working_color_space)
        file_menu.addAction(working_space_action)
        
        file_menu.addSeparator()
        
        # 保存图像
        save_action = QAction("保存图像", self)
        save_action.setShortcut(QKeySequence.StandardKey.Save)
        save_action.triggered.connect(self._save_image)
        file_menu.addAction(save_action)

        # 保存图像副本
        save_as_action = QAction("另存为...", self)
        save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs)
        save_as_action.triggered.connect(self._save_image_as)
        file_menu.addAction(save_as_action)
        
        file_menu.addSeparator()
        
        # 退出
        exit_action = QAction("退出", self)
        # 移除快捷键，避免与 Channel Gamma (Ctrl+Q) 冲突
        # 用户可通过菜单或系统快捷键 (Cmd+W/Alt+F4) 退出
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 编辑菜单
        edit_menu = menubar.addMenu("编辑")
        
        # 重置参数
        reset_action = QAction("重置参数", self)
        reset_action.triggered.connect(self._reset_parameters)
        edit_menu.addAction(reset_action)
        
        # 视图菜单
        view_menu = menubar.addMenu("视图")
        
        # 显示原始图像
        show_original_action = QAction("显示原始图像", self)
        show_original_action.setCheckable(True)
        show_original_action.triggered.connect(self._toggle_original_view)
        view_menu.addAction(show_original_action)
        
        view_menu.addSeparator()
        
        # 视图控制
        reset_view_action = QAction("重置视图", self)
        reset_view_action.setShortcut(QKeySequence("0"))
        reset_view_action.triggered.connect(self._reset_view)
        view_menu.addAction(reset_view_action)

        # 主题切换
        view_menu.addSeparator()
        dark_action = QAction("暗黑模式", self)
        dark_action.setCheckable(True)
        try:
            dark_action.setChecked(current_theme(QApplication.instance()) == "dark")
        except Exception:
            dark_action.setChecked(True)
        dark_action.toggled.connect(self._toggle_dark_mode)
        view_menu.addAction(dark_action)
        
        # 工具菜单
        tools_menu = menubar.addMenu("工具")
        
        # 直接添加分隔符，移除估算胶片类型功能
        
        # 文件分类规则管理器
        file_classification_action = QAction("文件分类规则管理器", self)
        file_classification_action.setToolTip("管理文件分类规则和默认预设文件")
        file_classification_action.triggered.connect(self._open_file_classification_manager)
        tools_menu.addAction(file_classification_action)
        
        # 精确通道分离IDT计算工具
        idt_calculator_action = QAction("精确通道分离IDT计算工具", self)
        idt_calculator_action.setToolTip("通过三张光谱分离图片计算精确的IDT色彩空间")
        idt_calculator_action.triggered.connect(self._open_idt_calculator)
        tools_menu.addAction(idt_calculator_action)

        # 配置管理
        tools_menu.addSeparator()
        config_manager_action = QAction("配置管理器", self)
        config_manager_action.triggered.connect(self._open_config_manager)
        tools_menu.addAction(config_manager_action)

        # 启用预览Profiling
        tools_menu.addSeparator()
        profiling_action = QAction("启用预览Profiling", self)
        profiling_action.setCheckable(True)
        profiling_action.toggled.connect(self._toggle_profiling)
        tools_menu.addAction(profiling_action)
        
        # LUT数学一致性验证功能已移除
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助")
        
        # 快捷键参考
        shortcuts_action = QAction("快捷键参考", self)
        shortcuts_action.setShortcut(QKeySequence("F1"))
        shortcuts_action.setToolTip("显示所有可用的键盘快捷键")
        shortcuts_action.triggered.connect(self._show_shortcuts_help)
        help_menu.addAction(shortcuts_action)
        
        help_menu.addSeparator()
        
        # 关于
        about_action = QAction("关于", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _create_toolbar(self):
        """创建工具栏"""
        toolbar = QToolBar()
        toolbar.setObjectName("mainToolBar")
        self.addToolBar(toolbar)
        
        
        # 打开图像
        open_action = QAction("打开", self)
        open_action.triggered.connect(self._open_image)
        toolbar.addAction(open_action)
        
        # 保存图像
        save_action = QAction("保存", self)
        save_action.triggered.connect(self._save_image)
        toolbar.addAction(save_action)
        
        toolbar.addSeparator()
        
        # 重置参数
        reset_action = QAction("粘贴默认", self)
        reset_action.triggered.connect(self._reset_parameters)
        toolbar.addAction(reset_action)
        
        # 设为当前文件夹默认
        set_folder_default_action = QAction("复制为默认", self)
        set_folder_default_action.setToolTip("将当前参数设置保存为当前文件夹的默认设置")
        set_folder_default_action.triggered.connect(self._set_folder_default)
        toolbar.addAction(set_folder_default_action)
        
        # 沿用接触印相设置（只在聚焦裁剪时可用）
        apply_contactsheet_action = QAction("沿用接触印相设置", self)
        apply_contactsheet_action.setToolTip("将接触印相的调色参数复制到当前裁剪")
        apply_contactsheet_action.triggered.connect(self._on_apply_contactsheet_to_crop)
        toolbar.addAction(apply_contactsheet_action)
        self._apply_contactsheet_action = apply_contactsheet_action
        # 初始禁用，进入聚焦模式后启用
        try:
            self._apply_contactsheet_action.setEnabled(False)
        except Exception:
            pass

        # 应用当前设置到接触印相（只在聚焦裁剪时可用）
        apply_to_contactsheet_action = QAction("应用当前设置到接触印相", self)
        apply_to_contactsheet_action.setToolTip("将当前裁剪的调色参数复制到接触印相")
        apply_to_contactsheet_action.triggered.connect(self._on_apply_to_contactsheet)
        toolbar.addAction(apply_to_contactsheet_action)
        self._apply_to_contactsheet_action = apply_to_contactsheet_action
        # 初始禁用，进入聚焦模式后启用
        try:
            self._apply_to_contactsheet_action.setEnabled(False)
        except Exception:
            pass
        

    
    def _create_statusbar(self):
        """创建状态栏"""
        self.statusBar().showMessage("就绪")

    def _apply_theme_and_refresh(self, theme: str):
        try:
            app = QApplication.instance()
            apply_theme(app, theme)
            enhanced_config_manager.set_ui_setting("theme", theme)
            # 将主题传递给曲线编辑器的自绘颜色
            try:
                self.parameter_panel.curve_editor.curve_edit_widget.apply_palette(app.palette(), theme)
            except Exception:
                pass
        except Exception as e:
            print(f"应用主题失败: {e}")

    def _toggle_dark_mode(self, enabled: bool):
        theme = "dark" if enabled else "light"
        self._apply_theme_and_refresh(theme)
    
    def _open_image(self):
        """打开图像文件"""
        # 获取上次打开的目录
        last_directory = enhanced_config_manager.get_directory("open_image")
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "打开图像文件",
            last_directory,
            "图像文件 (*.jpg *.jpeg *.png *.tiff *.tif *.bmp *.webp *.fff)"
        )
        
        if file_path:
            # 保存当前目录（写入父目录）
            enhanced_config_manager.set_directory("open_image", file_path)
            self.context.load_image(file_path)

    def _on_image_loaded(self):
        self._fit_after_next_preview = True
        self._should_fit_after_image_load = True  # 设置图像加载后fit标志
        # 图像加载后刷新裁剪选择条（基于 Context 内部列表）
        try:
            crops = getattr(self.context, '_crops', [])
            active_id = getattr(self.context, '_active_crop_id', None)
            self.preview_widget.refresh_crop_selector(crops, active_id)
        except Exception:
            pass
        
        # 更新文件夹导航状态
        try:
            self.preview_widget.update_navigation_state()
        except Exception:
            pass

        # 切换图片时清除中性点标记并禁用应用按钮
        self.preview_widget.clear_neutral_point()
        self.parameter_panel.enable_apply_neutral_button(False)

    def _on_autosave_requested(self):
        """处理来自Context的自动保存请求"""
        current_image = self.context.get_current_image()
        if not current_image or not current_image.file_path:
            return

        # 根据是否存在裁剪，决定保存 single 还是 contactsheet
        crops = self.context.get_all_crops()
        if not crops:
            # 无裁剪：保存为 v3 single
            preset = self.context.export_single_preset()
            self.context.auto_preset_manager.save_preset_for_image(current_image.file_path, preset)
        else:
            # 有裁剪：保存为 v3 contactsheet
            bundle = self.context.export_preset_bundle()
            self.context.auto_preset_manager.save_bundle_for_image(current_image.file_path, bundle)
        
        preset_file_path = self.context.auto_preset_manager.get_current_preset_file_path()
        if preset_file_path:
            self.statusBar().showMessage(f"参数已自动保存到: {preset_file_path.name}")

    def _on_curves_config_reloaded(self):
        """响应curves配置重载信号"""
        try:
            # 刷新parameter_panel中的curve_editor组件
            if hasattr(self.parameter_panel, 'curve_editor') and self.parameter_panel.curve_editor:
                self.parameter_panel.curve_editor.reload_curves_config()
        except Exception as e:
            print(f"Failed to reload curves config in UI: {e}")

    def _on_request_switch_profile(self, kind: str, crop_id: object):
        """处理切换Profile请求"""
        try:
            if kind == 'contactsheet':
                # 切换到原图模式
                self.context.switch_to_contactsheet()
                self.context.restore_crop_preview()
                is_focused = False
                # 恢复原图需要等预览更新完成再适应窗口
                self._fit_after_next_preview = True
            elif kind == 'crop' and isinstance(crop_id, str):
                # 一次性切换到裁剪并聚焦，避免闪烁
                self.context.switch_to_crop_focused(crop_id)
                is_focused = True
                # 聚焦需要等预览更新完成再适应窗口
                self._fit_after_next_preview = True
            else:
                is_focused = False
                
            # 刷新选择条状态
            crops = self.context.get_all_crops()
            active_id = self.context.get_active_crop_id()
            self.preview_widget.refresh_crop_selector(crops, active_id, is_focused)
            # 同步参数面板
            self.parameter_panel.initialize_defaults(self.context.get_current_params())
            # 更新工具可见性/可用性
            self._update_apply_contactsheet_enabled()
        except Exception as e:
            print(f"切换Profile失败: {e}")

    def _on_request_new_crop(self, direction: CropAddDirection):
        """响应新增裁剪请求
        - 若已有 >=1 个裁剪：智能新增（复制相同大小并布局），不进入鼠标框选
        - 若没有裁剪：保持现有逻辑，由预览进入手动框选
        
        Args:
            direction: 指定的添加方向
        """
        try:
            crops = self.context.get_all_crops()
            if isinstance(crops, list) and len(crops) >= 1:
                # 调用智能新增：复制尺寸、按长宽比布局
                new_id = self.context.smart_add_crop(direction)
                if new_id:
                    # 切回原图显示所有裁剪（不聚焦），并显示编号按钮
                    self.context.switch_to_contactsheet()
                    try:
                        self.preview_widget._hide_single_crop_selector = False
                    except Exception:
                        pass
                    self.preview_widget.refresh_crop_selector(
                        self.context.get_all_crops(),
                        self.context.get_active_crop_id(),
                        is_focused=False
                    )
                    # 同步参数面板
                    self.parameter_panel.initialize_defaults(self.context.get_current_params())
                return
        except Exception as e:
            print(f"智能新增裁剪失败: {e}")
        # 无裁剪时，保持原逻辑：预览组件会进入手动框选
        return

    def _on_request_delete_crop(self, crop_id: str):
        try:
            self.context.delete_crop(crop_id)
            # 刷新选择条
            crops = self.context.get_all_crops()
            active_id = self.context.get_active_crop_id()
            self.preview_widget.refresh_crop_selector(crops, active_id, is_focused=False)
            # 更新工具可用性
            self._update_apply_contactsheet_enabled()
        except Exception as e:
            print(f"删除裁剪失败: {e}")

    def _on_apply_contactsheet_to_crop(self):
        print("DEBUG: 沿用接触印相设置按钮被点击")
        try:
            print(f"DEBUG: 当前active_crop_id: {getattr(self.context, '_active_crop_id', None)}")
            print(f"DEBUG: 当前crop_focused: {getattr(self.context, '_crop_focused', False)}")
            print(f"DEBUG: contactsheet_params存在: {bool(getattr(self.context._contactsheet_profile, 'params', None))}")
            
            self.context.apply_contactsheet_to_active_crop()
            print("DEBUG: apply_contactsheet_to_active_crop调用完成")
            
            # 同步参数面板
            current_params = self.context.get_current_params()
            print(f"DEBUG: 获取到当前参数: {bool(current_params)}")
            self.parameter_panel.initialize_defaults(current_params)
            print("DEBUG: 参数面板同步完成")
        except Exception as e:
            print(f"沿用接触印相设置失败: {e}")
            import traceback
            traceback.print_exc()

    def _on_apply_to_contactsheet(self):
        print("DEBUG: 应用当前设置到接触印相按钮被点击")
        try:
            print(f"DEBUG: 当前active_crop_id: {getattr(self.context, '_active_crop_id', None)}")
            print(f"DEBUG: 当前crop_focused: {getattr(self.context, '_crop_focused', False)}")
            print(f"DEBUG: current_params存在: {bool(self.context.get_current_params())}")
            
            self.context.apply_active_crop_to_contactsheet()
            print("DEBUG: apply_active_crop_to_contactsheet调用完成")
            
            # 如果当前在contactsheet模式，同步参数面板
            if self.context.get_current_profile_kind() == 'contactsheet':
                current_params = self.context.get_current_params()
                print(f"DEBUG: 获取到contactsheet当前参数: {bool(current_params)}")
                self.parameter_panel.initialize_defaults(current_params)
                print("DEBUG: contactsheet参数面板同步完成")
        except Exception as e:
            print(f"应用当前设置到接触印相失败: {e}")
            import traceback
            traceback.print_exc()

    def _on_auto_color_requested(self):
        self.context.run_auto_color_correction(self.preview_widget.get_current_image_data)

    def _on_auto_color_iterative_requested(self):
        self.context.run_iterative_auto_color(self.preview_widget.get_current_image_data)

    def _on_pick_neutral_point_requested(self):
        """进入中性点选择模式（不立即迭代）"""
        self.preview_widget.enter_neutral_point_selection_mode()

    def _on_neutral_point_selected(self, norm_x: float, norm_y: float):
        """用户选择了中性点，启用应用按钮"""
        # 启用应用按钮
        self.parameter_panel.enable_apply_neutral_button(True)
        # 显示提示信息
        self.statusBar().showMessage("中性点已选择，点击'应用中性色'开始调色", 3000)

    def _on_apply_neutral_color_requested(self, white_point: int):
        """应用中性色迭代调整"""
        # 检查是否已选择点
        if self.preview_widget.neutral_point_norm is None:
            self.statusBar().showMessage("请先点击'取点'按钮选择中性点", 3000)
            return

        # 保存色温并开始迭代
        self._neutral_point_white_point = white_point
        norm_x, norm_y = self.preview_widget.neutral_point_norm
        self.context.calculate_neutral_point_auto_gain(
            norm_x, norm_y,
            self.preview_widget.get_current_image_data,
            white_point
        )

    def _on_neutral_white_point_changed(self, white_point: int):
        """色温spinbox变化时，如果有中性点则自动重新定义"""
        # 更新保存的色温值
        self._neutral_point_white_point = white_point

        # 如果preview上有中性点标记，自动重新定义
        if self.preview_widget.neutral_point_norm is not None:
            norm_x, norm_y = self.preview_widget.neutral_point_norm
            self.context.calculate_neutral_point_auto_gain(
                norm_x, norm_y,
                self.preview_widget.get_current_image_data,
                white_point
            )

    # _apply_preset logic is now in ApplicationContext
    # def _apply_preset(self, preset: Preset): ...

    def _load_preset(self):
        """加载预设文件并应用"""
        last_directory = enhanced_config_manager.get_directory("preset")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "加载预设", last_directory, "预设文件 (*.json)"
        )
        if not file_path:
            return

        enhanced_config_manager.set_directory("preset", str(Path(file_path).parent))

        try:
            preset = PresetManager.load_preset(file_path)
            if not preset:
                raise ValueError("加载预设返回空值")

            # 0. 检查raw_file是否匹配
            current_image = self.context.get_current_image()
            if preset.raw_file and current_image:
                current_filename = Path(current_image.file_path).name
                if preset.raw_file != current_filename:
                    from PySide6.QtWidgets import QMessageBox
                    reply = QMessageBox.question(self, "预设警告", 
                        f"预设应用于 '{preset.raw_file}',\n"
                        f"当前图像是 '{current_filename}'.\n\n"
                        "确定要应用吗？",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No)
                    if reply == QMessageBox.StandardButton.No:
                        return

            self.context.load_preset(preset)
            
            # 如果有图像，触发预览更新
            if current_image:
                self.context._prepare_proxy()
                self.context._trigger_preview_update()

        except (IOError, ValueError, FileNotFoundError) as e:
            QMessageBox.critical(self, "加载预设失败", str(e))

    def _create_preset_from_current_state(self, name: str) -> Preset:
        """从当前应用状态创建Preset对象"""
        params = self.context.get_current_params()
        
        # 构造 InputTransformationDefinition (名称+数值冗余)
        cs_name = self.context.get_input_color_space()
        cs_def = None
        definition = self.context.color_space_manager.get_color_space_definition(cs_name)
        if definition:
            cs_def = InputTransformationDefinition(name=cs_name, definition=definition)
        
        # 构造 MatrixDefinition
        matrix_display_name = self.parameter_panel.matrix_combo.currentText()
        matrix_values = None
        if params.density_matrix is not None:
            matrix_values = params.density_matrix.tolist()

        matrix_def = None
        if matrix_display_name:
            clean_matrix_name = matrix_display_name.replace("preset: ", "")
            # 如果UI显示是"自定义"，则在预设中记录为 "custom"
            if clean_matrix_name == "自定义":
                clean_matrix_name = "custom"
            matrix_def = MatrixDefinition(name=clean_matrix_name, values=matrix_values)

        # 构造Curve Definition
        curve_def = None
        curve_display_name = self.parameter_panel.curve_editor.curve_combo.currentText()
        curve_points = params.curve_points
        
        if curve_display_name:
            clean_curve_name = curve_display_name.replace("preset: ", "")
            if clean_curve_name == "自定义":
                clean_curve_name = "custom"
            curve_def = CurveDefinition(name=clean_curve_name, points=curve_points)

        # 构造文件名、裁切和方向
        current_image = self.context.get_current_image()
        raw_file = Path(current_image.file_path).name if current_image else None
        # 裁切：优先使用当前激活的 crop；否则使用 contactsheet 的单裁剪（BWC）
        crop_rect = self.context.get_active_crop() or self.context.get_contactsheet_crop_rect()

        # 获取当前crop实例（包含独立orientation）
        crop_instance = self.context.get_active_crop_instance()
        
        return Preset(
            name=name,
            # Metadata
            raw_file=raw_file,
            orientation=self.context.get_current_orientation(),  # 全局orientation
            crop=crop_rect,  # 向后兼容（single/原图裁剪）
            film_type=self.parameter_panel.get_current_film_type(),  # 胶片类型
            # 新的多裁剪结构（包含crop的独立orientation）
            crops=(
                [crop_instance.to_dict()]
                if crop_instance is not None else None
            ),
            active_crop_id=(crop_instance.id if crop_instance is not None else None),
            # Input Transformation
            input_transformation=cs_def,
            # Grading Parameters
            grading_params=params.to_dict(),
            density_matrix=matrix_def,
            density_curve=curve_def,
        )

    def _save_preset(self):
        """保存当前设置为预设文件"""
        if not self.context.get_current_image():
            QMessageBox.warning(self, "请先打开一张图片", "无法保存预设，因为需要基于当前状态创建。")
            return

        last_directory = enhanced_config_manager.get_directory("preset")
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存预设", last_directory, "预设文件 (*.json)"
        )
        if not file_path:
            return

        # 确保文件名以.json结尾
        if not file_path.lower().endswith('.json'):
            file_path += '.json'
        
        enhanced_config_manager.set_directory("preset", str(Path(file_path).parent))

        try:
            # 弹窗让用户输入预设名称
            from PySide6.QtWidgets import QInputDialog
            preset_name, ok = QInputDialog.getText(self, "预设名称", "请输入预设名称:")
            if not ok or not preset_name:
                preset_name = Path(file_path).stem

            # 创建Preset对象
            preset = self._create_preset_from_current_state(preset_name)

            # 保存预设
            PresetManager.save_preset(preset, file_path)
            self.statusBar().showMessage(f"预设已保存: {preset.name}")

        except (IOError, KeyError) as e:
            QMessageBox.critical(self, "保存预设失败", str(e))

    def _select_input_color_space(self):
        """选择输入色彩变换"""
        from PySide6.QtWidgets import QInputDialog
        
        # 获取可用的色彩空间列表
        available_spaces = self.context.color_space_manager.get_available_color_spaces()
        
        # 显示选择对话框
        color_space, ok = QInputDialog.getItem(
            self, 
            "选择输入色彩变换", 
            "请选择图像的输入色彩变换:", 
            available_spaces, 
            available_spaces.index(self.context.get_input_color_space()) if self.context.get_input_color_space() in available_spaces else 0, 
            False
        )
        
        if ok and color_space:
            try:
                self.context.set_input_color_space(color_space)
                # 更新状态栏
                self.statusBar().showMessage(f"已设置输入色彩变换: {color_space}")
                # 如果已经有图像，重新处理
                if self.context.get_current_image():
                    self.context._reload_with_color_space()
            except Exception as e:
                QMessageBox.critical(self, "错误", f"设置色彩空间失败: {str(e)}")
    
    def _select_working_color_space(self):
        """选择工作色彩空间（全局设置）"""
        # 获取可用的工作色彩空间列表
        working_spaces = self.context.color_space_manager.get_working_color_spaces()
        
        if not working_spaces:
            QMessageBox.information(self, "提示", "没有可用的工作色彩空间")
            return
        
        # 获取当前工作空间
        current_working_space = self.context.color_space_manager.get_current_working_space()
        
        # 显示选择对话框
        selected_space, ok = QInputDialog.getItem(
            self, 
            "选择工作色彩空间", 
            "请选择应用程序的工作色彩空间:", 
            working_spaces, 
            working_spaces.index(current_working_space) if current_working_space in working_spaces else 0, 
            False
        )
        
        if ok and selected_space and selected_space != current_working_space:
            try:
                # 设置新的工作空间
                success = self.context.color_space_manager.set_working_space(selected_space)
                if success:
                    # 清除reference color缓存，确保下次使用新工作空间
                    self.context.clear_reference_color_cache()
                    
                    # 更新状态栏
                    self.statusBar().showMessage(f"工作色彩空间已切换至: {selected_space}")
                    
                    # 如果有图像加载，重新处理预览
                    if self.context.get_current_image():
                        self._reload_with_color_space()
                else:
                    QMessageBox.warning(self, "警告", f"无法设置工作空间: {selected_space}")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"设置工作色彩空间失败: {str(e)}")
    
    def _reload_with_icc(self):
        """使用新的ICC配置文件重新加载图像"""
        if not self.context.get_current_image():
            return
            
        try:
            # 简单触发参数更新，让系统重新处理图像和ICC配置
            current_params = self.context.get_current_params()
            self.context.update_params(current_params)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"重新处理图像失败: {str(e)}")
    
    def _reload_with_color_space(self):
        """使用新的色彩空间重新加载图像"""
        if not self.context.get_current_image():
            return
        
        try:
            # 简单触发参数更新，让系统重新处理图像和工作空间
            current_params = self.context.get_current_params()
            self.context.update_params(current_params)
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"重新加载图像失败: {str(e)}")

    # ===== Spectral Sharpening Hooks =====

    def _on_ccm_optimize_requested(self):
        """根据色卡执行光谱锐化（硬件校正）优化（后台），更新输入色彩空间与参数。"""
        current_image = self.context.get_current_image()
        if not (current_image and current_image.array is not None):
            QMessageBox.warning(self, "提示", "请先打开一张图片")
            return
        
        # 使用原图进行优化
        source_image = self.context._current_image  # 直接访问原图
        if not (source_image and source_image.array is not None):
            QMessageBox.warning(self, "提示", "无法获取源图像数据")
            return
            
        # 获取当前输入空间 gamma（若取不到，退化为1.0）
        cs_name = self.context.get_input_color_space()
        cs_info = self.context.color_space_manager.get_color_space_info(cs_name) or {}
        input_gamma = float(cs_info.get("gamma", 1.0))

        # 取色卡角点（归一化坐标）
        cc_corners_norm = getattr(self.preview_widget, 'cc_corners_norm', None)
        if not cc_corners_norm or len(cc_corners_norm) != 4:
            QMessageBox.information(self, "提示", "请在预览中启用色卡选择器并设置四角点")
            return
            
        # 直接使用归一化坐标转换为原图像素坐标
        cc_corners_source = self.preview_widget._norm_to_source_coords(cc_corners_norm)
        if not cc_corners_source:
            QMessageBox.warning(self, "错误", "无法获取源图像坐标")
            return

        # 获取光谱锐化（硬件校正）配置
        sharpening_config = self.parameter_panel.get_spectral_sharpening_config()
        
        # 取当前密度校正矩阵
        params = self.context.get_current_params()
        use_mat = bool(params.enable_density_matrix)
        
        # 如果启用密度矩阵优化，不传递当前矩阵（避免双重应用）
        # 优化器将从单位矩阵开始优化
        if sharpening_config.optimize_density_matrix:
            corr_mat = None
        else:
            corr_mat = params.density_matrix if (use_mat and params.density_matrix is not None) else None

        # 创建并显示进度对话框
        progress_dialog = CMAESProgressDialog(self)
        progress_dialog.set_max_iterations(sharpening_config.max_iter)
        progress_dialog.start_optimization()  # 立即启动优化状态
        progress_dialog.show()
        progress_dialog.raise_()  # 确保对话框在前台
        
        # 激活优化状态，避免其他状态消息干扰
        self.context.set_ccm_optimization_active(True)
        self.statusBar().showMessage("正在进行光谱锐化（硬件校正）优化...")
        
        # 获取UI当前参数作为优化初值
        current_params = self.context.get_current_params()
        ui_params = {
            'gamma': current_params.density_gamma,
            'dmax': current_params.density_dmax,
            'r_gain': current_params.rgb_gains[0],
            'b_gain': current_params.rgb_gains[2],
            'density_matrix': current_params.density_matrix if current_params.density_matrix is not None else np.eye(3)
        }
        
        # 始终获取当前色彩空间的primaries（无论是否优化IDT）
        # 这样确保优化器使用正确的初值，而不是硬编码的sRGB
        cs_name = self.context.get_input_color_space()
        cs_info = self.context.color_space_manager.get_color_space_info(cs_name)
        if cs_info and 'primaries' in cs_info:
            primaries = cs_info['primaries']
            # 兼容两种primaries格式
            if isinstance(primaries, dict):
                # 字典格式：{'R': [x, y], 'G': [x, y], 'B': [x, y]}
                ui_params['primaries_xy'] = np.array([
                    [primaries['R'][0], primaries['R'][1]],
                    [primaries['G'][0], primaries['G'][1]],
                    [primaries['B'][0], primaries['B'][1]]
                ])
            else:
                # numpy数组格式：shape (3, 2)
                ui_params['primaries_xy'] = np.array(primaries)

        # 在UI线程直接调用会卡顿。这里用QRunnable封装，复用全局线程池。
        class _CCMWorker(QRunnable):
            def __init__(self, image_array, corners, gamma, use_mat, mat, config, ui_params, status_callback=None, color_space_manager=None, working_colorspace=None, app_context=None):
                super().__init__()
                self.image_array = image_array
                self.corners = corners
                self.gamma = gamma
                self.use_mat = use_mat
                self.mat = mat
                self.config = config
                self.ui_params = ui_params
                self.status_callback = status_callback
                self.color_space_manager = color_space_manager
                self.working_colorspace = working_colorspace
                self.app_context = app_context
                self.result = None
                self.error = None
            @Slot()
            def run(self):
                try:
                    self.result = run_spectral_sharpening(
                        self.image_array,
                        self.corners,
                        self.gamma,
                        self.use_mat,
                        self.mat,
                        optimizer_max_iter=self.config.max_iter,
                        optimizer_tolerance=self.config.tolerance,
                        reference_file=self.config.reference_file,
                        sharpening_config=self.config,
                        ui_params=self.ui_params,
                        status_callback=self.status_callback,
                        color_space_manager=self.color_space_manager,
                        working_colorspace=self.working_colorspace,
                        app_context=self.app_context,
                    )
                except Exception as e:
                    import traceback
                    self.error = f"{e}\n{traceback.format_exc()}"

        # 创建线程安全的状态回调，使用信号/槽机制
        def thread_safe_status_callback(message: str):
            print(f"[DEBUG] 收到状态回调: '{message}'")

            # 使用线程安全的信号机制更新进度对话框
            # 这确保所有UI更新都在主线程中执行
            try:
                progress_dialog.request_update_progress(message)
                print(f"[DEBUG] 通过信号更新进度对话框")
            except Exception as e:
                print(f"[DEBUG] 信号更新失败，使用QTimer备用方案: {e}")
                # 备用方案：使用 functools.partial 避免闭包内存泄漏
                from functools import partial
                update_func = partial(self._ccm_update_progress_fallback, progress_dialog, message)
                QTimer.singleShot(0, update_func)

        worker = _CCMWorker(
            source_image.array,
            cc_corners_source,
            input_gamma,
            use_mat,
            corr_mat,
            sharpening_config,
            ui_params,
            thread_safe_status_callback,
            color_space_manager=self.context.color_space_manager,
            working_colorspace=self.context.color_space_manager.get_current_working_space(),
            app_context=self.context
        )

        # 注意：原来的 _on_done 和 _poll 闭包已经移动到类方法：
        # _ccm_on_optimization_done 和 _ccm_poll_worker_completion

        print(f"[DEBUG] 启动Worker线程")

        # 存储 worker 和 progress_dialog 的引用（用于类方法访问）
        self._ccm_worker = worker
        self._ccm_progress_dialog = progress_dialog

        self.context.thread_pool.start(worker)
        print(f"[DEBUG] 启动轮询检查")
        # 使用类方法替代闭包，避免内存泄漏
        QTimer.singleShot(150, self._ccm_poll_worker_completion)

    def _ccm_update_progress_fallback(self, progress_dialog, message):
        """CCM 优化进度更新的备用方法（替代闭包）

        用于 QTimer.singleShot 的备用方案，避免闭包内存泄漏
        """
        try:
            if progress_dialog and progress_dialog.isVisible():
                progress_dialog.request_update_progress(message)
        except Exception as ex:
            print(f"[DEBUG] 备用方案更新进度失败: {ex}")

    def _ccm_poll_worker_completion(self):
        """轮询检测 CCM 优化任务完成（替代闭包）

        这是一个类方法，用于替代 _poll 闭包，避免内存泄漏
        """
        if not self._ccm_worker:
            print(f"[DEBUG] _ccm_poll: worker 已清理，停止轮询")
            return

        has_result = getattr(self._ccm_worker, 'result', None) is not None
        has_error = getattr(self._ccm_worker, 'error', None) is not None
        print(f"[DEBUG] _ccm_poll: has_result={has_result}, has_error={has_error}")

        if has_result or has_error:
            print(f"[DEBUG] Worker完成，调用_ccm_on_optimization_done")
            self._ccm_on_optimization_done()
            return

        # 继续轮询
        QTimer.singleShot(150, self._ccm_poll_worker_completion)

    def _ccm_on_optimization_done(self):
        """CCM 优化完成的处理（替代闭包）

        这是一个类方法，用于替代 _on_done 闭包，避免内存泄漏
        """
        worker = self._ccm_worker
        progress_dialog = self._ccm_progress_dialog

        if not worker or not progress_dialog:
            print(f"[DEBUG] _ccm_on_optimization_done: worker 或 progress_dialog 为 None")
            return

        try:
            if worker.error:
                # 结束优化状态
                self.context.set_ccm_optimization_active(False)
                progress_dialog.finish_optimization(False)
                QMessageBox.critical(self, "优化失败", worker.error)
                self.statusBar().showMessage("光谱锐化（硬件校正）优化失败")
                return

            res = worker.result or {}
            params_dict = res.get('parameters', {})

            # 获取当前输入空间名称（用于后续处理）
            cs_name = self.context.get_input_color_space()

            # 获取光谱锐化配置（从 worker 的 config 属性）
            sharpening_config = worker.config

            # 只有启用IDT优化时才处理primaries结果
            if sharpening_config.optimize_idt_transformation:
                primaries_xy = np.asarray(params_dict.get('primaries_xy'), dtype=float)
                if primaries_xy is None or primaries_xy.shape != (3, 2):
                    # 结束优化状态
                    self.context.set_ccm_optimization_active(False)
                    progress_dialog.finish_optimization(False)
                    QMessageBox.warning(self, "结果无效", "未获得有效的基色坐标")
                    self.statusBar().showMessage("光谱锐化（硬件校正）优化完成但结果无效")
                    return

                # 注册并切换到自定义输入色彩空间
                base_name = cs_name.replace("_custom", "").replace("_preset", "")
                custom_name = f"{base_name}_custom"
                self.context.color_space_manager.register_custom_colorspace(custom_name, primaries_xy, None, gamma=1.0)
                # 使用专用入口以便重建代理
                self.context.set_input_color_space(custom_name)

                # 更新UCS widget显示优化后的primaries
                from divere.core.color_space import xy_to_uv
                coords_uv = {}
                for i, key in enumerate(['R', 'G', 'B']):
                    if i < len(primaries_xy):
                        x, y = primaries_xy[i]
                        u, v = xy_to_uv(x, y)
                        coords_uv[key] = (u, v)

                if len(coords_uv) == 3:
                    self.parameter_panel.ucs_widget.set_uv_coordinates(coords_uv)

            # 应用其他参数更新
            new_params = self.context.get_current_params().copy()
            # 密度参数与RB对数增益
            new_params.density_gamma = float(params_dict.get('gamma', new_params.density_gamma))
            new_params.density_dmax = float(params_dict.get('dmax', new_params.density_dmax))
            r_gain = float(params_dict.get('r_gain', new_params.rgb_gains[0]))
            b_gain = float(params_dict.get('b_gain', new_params.rgb_gains[2]))
            new_params.rgb_gains = (r_gain, new_params.rgb_gains[1], b_gain)

            # 应用优化的density_matrix（如果有）
            if 'density_matrix' in params_dict and params_dict['density_matrix'] is not None:
                new_params.density_matrix = params_dict['density_matrix']
                new_params.enable_density_matrix = True
                new_params.density_matrix_name = "optimized_custom"

            # 强制重置 channel_gamma 为 1.0（色卡优化不包含主观分层反差调整）
            new_params.channel_gamma_r = 1.0
            new_params.channel_gamma_b = 1.0

            self.context.update_params(new_params)
            final_log_rmse = float(res.get('rmse', 0.0))

            # 结束优化状态
            self.context.set_ccm_optimization_active(False)
            progress_dialog.finish_optimization(True, final_log_rmse)

            completion_message = f"光谱锐化（硬件校正）完成：最终log-RMSE={final_log_rmse:.4f}"
            self.statusBar().showMessage(completion_message)
            print(f"[DEBUG] 优化完成，显示消息: '{completion_message}'")
        finally:
            # 清理临时引用，释放内存
            self._ccm_worker = None
            self._ccm_progress_dialog = None

    def _on_save_custom_colorspace_requested(self, primaries_dict: dict):
        """保存 UCS 三角的基色坐标为输入色彩变换 JSON（项目config目录）。"""
        try:
            # 弹出保存对话框，让用户输入名称和描述
            from PySide6.QtWidgets import QInputDialog, QMessageBox
            
            # 获取基础名称作为默认值
            name_base = self.context.get_input_color_space().replace("_custom", "").replace("_preset", "")
            default_name = f"{name_base}_custom"
            
            # 输入自定义名称
            save_name, ok = QInputDialog.getText(
                self, 
                "保存自定义色彩空间", 
                "请输入色彩空间名称:",
                text=default_name
            )
            
            if not ok or not save_name.strip():
                return
                
            # 输入描述（可选）
            description, ok = QInputDialog.getText(
                self, 
                "保存自定义色彩空间", 
                "请输入描述（可选）:",
                text="用户自定义色彩空间"
            )
            
            if not ok:
                description = "用户自定义色彩空间"
            
            # 构建保存数据
            data = {
                "name": save_name,
                "type": ["IDT"],
                "description": description,
                "primaries": {
                    "R": [float(primaries_dict['R'][0]), float(primaries_dict['R'][1])],
                    "G": [float(primaries_dict['G'][0]), float(primaries_dict['G'][1])],
                    "B": [float(primaries_dict['B'][0]), float(primaries_dict['B'][1])],
                },
                # 采用D65与 gamma=1.0（扫描线性）
                "white_point": [0.3127, 0.3290],
                "gamma": 1.0,
            }
            
            # 保存到项目config目录
            ok = enhanced_config_manager.save_user_config("colorspace", save_name, data)
            if ok:
                self.statusBar().showMessage(f"已保存自定义色彩空间到项目配置: {save_name}.json")
                QMessageBox.information(self, "保存成功", f"自定义色彩空间已保存: {save_name}")
                
                # 重新加载色彩空间配置并刷新UI，然后自动应用刚保存的色彩空间
                try:
                    self.context.reload_all_configs()
                    self.parameter_panel._refresh_colorspace_combo()
                    # 自动选择并应用刚保存的色彩空间
                    self.context.set_input_color_space(save_name)
                    self.parameter_panel._refresh_colorspace_combo()  # 再次刷新以显示选中状态
                except AttributeError:
                    # 如果没有刷新方法，用户需要重启应用来看到新色彩空间
                    pass
            else:
                QMessageBox.warning(self, "保存失败", "无法保存到项目配置目录")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存自定义色彩空间失败: {str(e)}")

    def _on_save_density_matrix_requested(self, matrix: np.ndarray, current_name: str):
        """保存密度矩阵到文件"""
        try:
            from PySide6.QtWidgets import QInputDialog, QMessageBox
            
            # 输入矩阵名称
            save_name, ok = QInputDialog.getText(
                self, 
                "保存密度矩阵", 
                "请输入矩阵名称:",
                text=current_name if current_name else "custom_matrix"
            )
            
            if not ok or not save_name.strip():
                return
                
            # 输入描述（可选）
            description, ok = QInputDialog.getText(
                self, 
                "保存密度矩阵", 
                "请输入描述（可选）:",
                text="用户自定义密度校正矩阵"
            )
            
            if not ok:
                description = "用户自定义密度校正矩阵"
            
            # 构建保存数据
            data = {
                "name": save_name,
                "description": description,
                "version": 1,
                "matrix": matrix.tolist()  # 转换为列表以便JSON序列化
            }
            
            # 保存到项目config目录的matrices子目录
            ok = enhanced_config_manager.save_user_config("matrices", save_name, data)
            if ok:
                self.statusBar().showMessage(f"已保存密度矩阵到项目配置: {save_name}.json")
                QMessageBox.information(self, "保存成功", f"密度矩阵已保存: {save_name}")
                
                # 重新加载矩阵列表并自动应用刚保存的矩阵
                try:
                    # 1. 重新加载所有配置
                    self.context.reload_all_configs()
                    # 2. 刷新parameter_panel的矩阵下拉列表，确保新保存的矩阵出现在列表中
                    self.parameter_panel._refresh_matrix_combo()
                    # 3. 更新当前参数以应用刚保存的矩阵
                    new_params = self.context.get_current_params().copy()
                    new_params.density_matrix = matrix
                    new_params.density_matrix_name = save_name
                    new_params.enable_density_matrix = True
                    # 4. 更新参数，这会触发UI更新，包括下拉框选择状态
                    self.context.update_params(new_params)
                except AttributeError:
                    # 如果没有刷新方法，用户需要重启应用来看到新矩阵
                    pass
            else:
                QMessageBox.critical(self, "保存失败", "无法保存密度矩阵到配置目录")
                
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存密度矩阵时出错：\n{str(e)}")
    
    def _on_save_colorchecker_colors_requested(self):
        """保存色卡颜色到JSON文件"""
        try:
            from PySide6.QtWidgets import QInputDialog, QMessageBox, QFileDialog
            import json
            from datetime import datetime
            
            # 检查色卡选择器是否启用
            if not self.preview_widget.cc_enabled:
                QMessageBox.warning(self, "警告", "请先启用色卡选择器")
                return
            
            # 从原图像中直接读取色卡颜色数据（假设为ACEScg）
            rgb_patches = self._extract_original_colorchecker_patches()
            if rgb_patches is None or rgb_patches.shape[0] != 24:
                QMessageBox.warning(self, "警告", "无法读取色卡数据，请检查色卡选择器位置是否正确")
                return
            
            # 输入文件名
            save_name, ok = QInputDialog.getText(
                self,
                "保存色卡颜色",
                "请输入色卡文件名称:",
                text="custom_colorchecker"
            )
            
            if not ok or not save_name.strip():
                return
            
            save_name = save_name.strip()
            
            # 输入描述信息
            description, ok = QInputDialog.getText(
                self,
                "色卡描述",
                "请输入色卡描述信息:",
                text=f"从图像中提取的色卡数据 - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            )
            
            if not ok:
                description = "用户自定义色卡数据"
            
            # 获取当前working colorspace名称
            current_working_space = self.context.color_space_manager.get_current_working_space()
            
            # 构造保存数据
            colorchecker_data = {
                "description": description,
                "type": "DensityExp",
                "required_working_colorspace": current_working_space,
                "data": {}
            }
            
            # 将24个色块数据按A1-D6格式整理
            patch_ids = []
            for row in ['A', 'B', 'C', 'D']:
                for col in range(1, 7):
                    patch_ids.append(f"{row}{col}")
            
            for i, patch_id in enumerate(patch_ids):
                if i < rgb_patches.shape[0]:
                    r, g, b = rgb_patches[i]
                    colorchecker_data["data"][patch_id] = [float(r), float(g), float(b)]
            
            # 选择保存位置
            try:
                from divere.utils.app_paths import resolve_data_path
                default_dir = str(resolve_data_path("config", "colorchecker"))
            except Exception:
                from pathlib import Path
                default_dir = str(Path(__file__).parent.parent / "config" / "colorchecker")
            
            filename = f"{save_name}_cc24data.json"
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "保存色卡颜色文件",
                f"{default_dir}/{filename}",
                "JSON文件 (*.json)"
            )
            
            if file_path:
                # 确保文件扩展名正确
                if not file_path.endswith('.json'):
                    file_path += '.json'
                
                # 写入JSON文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(colorchecker_data, f, indent=4, ensure_ascii=False)
                
                QMessageBox.information(self, "保存成功", f"色卡颜色已保存到：\n{file_path}")
                
                # 刷新色卡下拉列表
                try:
                    self.parameter_panel._populate_colorchecker_combo()
                except AttributeError:
                    pass
                    
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存色卡颜色时出错：\n{str(e)}")

    def _apply_rotation_to_point(self, x, y, orig_w, orig_h, orientation):
        """应用旋转变换到点坐标

        将原图坐标系中的点坐标变换到旋转后坐标系中。

        Args:
            x, y: 原图坐标系中的点坐标（像素）
            orig_w, orig_h: 原图尺寸（宽，高）
            orientation: 旋转角度（0, 90, 180, 270度）

        Returns:
            (x_rot, y_rot): 旋转后坐标系中的点坐标（像素）
        """
        k = (orientation // 90) % 4
        if k == 0:  # 无旋转
            return x, y
        elif k == 1:  # 左旋90度
            # 原图的(x, y) -> 旋转后图像中的(y, orig_w - 1 - x)
            # 像素坐标从0开始，需要 -1
            return y, orig_w - 1 - x
        elif k == 2:  # 旋转180度
            return orig_w - 1 - x, orig_h - 1 - y
        elif k == 3:  # 右旋90度（左旋270度）
            return orig_h - 1 - y, x
        return x, y

    def _rotate_normalized_coords(self, x_norm, y_norm, orientation):
        """旋转归一化坐标（0-1范围）

        Args:
            x_norm, y_norm: 原图归一化坐标 [0, 1]
            orientation: 旋转角度（0, 90, 180, 270度）

        Returns:
            (x_rot_norm, y_rot_norm): 旋转后的归一化坐标 [0, 1]
        """
        k = (orientation // 90) % 4
        if k == 0:  # 无旋转
            return x_norm, y_norm
        elif k == 1:  # 左旋90度
            # 归一化坐标变换：(x, y) → (y, 1-x)
            return y_norm, 1.0 - x_norm
        elif k == 2:  # 旋转180度
            return 1.0 - x_norm, 1.0 - y_norm
        elif k == 3:  # 右旋90度
            return 1.0 - y_norm, x_norm
        return x_norm, y_norm

    def _extract_original_colorchecker_patches(self):
        """从pipeline处理后的图像中读取色卡区域的working space RGB值"""
        try:
            import numpy as np
            import cv2
            from copy import deepcopy
            
            # 检查必要条件
            if not (self.preview_widget.cc_enabled and 
                    self.preview_widget.cc_corners_norm and 
                    self.context.get_current_image()):
                return None
            
            # 获取当前proxy图像和参数
            proxy_image = self.context._current_proxy
            if proxy_image is None:
                return None
                
            current_params = deepcopy(self.context.get_current_params())
            
            # 通过pipeline处理图像，获取working space RGB
            processed_image = self.context.the_enlarger.apply_full_pipeline(
                proxy_image, 
                current_params,
                convert_to_monochrome_in_idt=self.context.should_convert_to_monochrome(),
                monochrome_converter=self.context.color_space_manager.convert_to_monochrome if self.context.should_convert_to_monochrome() else None
            )
            
            # 获取处理后的图像数组
            processed_array = processed_image.array
            if processed_array is None:
                return None
            
            # 确保数据为float32格式
            if processed_array.dtype == np.uint8:
                processed_array = processed_array.astype(np.float32) / 255.0
            else:
                processed_array = np.clip(processed_array, 0.0, 1.0).astype(np.float32)
            
            # 获取处理后（已旋转）的图像尺寸
            H_img, W_img = processed_array.shape[:2]

            # 获取色卡角点的归一化坐标（相对原图）
            cc_corners_norm = self.preview_widget.cc_corners_norm

            # 获取当前旋转角度
            global_orientation = self.context.get_current_orientation()

            # 关键修复：proxy_image 已经被旋转，processed_array 也已旋转
            # 需要将归一化坐标从原图坐标系旋转到processed_array坐标系
            # 然后直接乘以processed_array的尺寸
            corners_img = []
            for x_norm, y_norm in cc_corners_norm:
                # 先旋转归一化坐标到旋转后的坐标系
                if global_orientation % 360 != 0:
                    x_norm_rot, y_norm_rot = self._rotate_normalized_coords(
                        x_norm, y_norm, global_orientation
                    )
                else:
                    x_norm_rot, y_norm_rot = x_norm, y_norm

                # 再乘以旋转后图像的尺寸得到像素坐标
                x_img = x_norm_rot * W_img
                y_img = y_norm_rot * H_img

                corners_img.append([x_img, y_img])
            
            # 计算透视变换矩阵
            src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
            dst = np.array(corners_img, dtype=np.float32)
            H = cv2.getPerspectiveTransform(src, dst)
            
            # 提取24个色块
            margin = 0.18  # 边距，避免边界污染
            rgb_list = []
            
            # 按4行6列提取色块
            for r in range(4):
                for c in range(6):
                    # 计算色块在归一化坐标系中的位置
                    gx0 = c / 6.0
                    gx1 = (c + 1) / 6.0
                    gy0 = r / 4.0
                    gy1 = (r + 1) / 4.0
                    
                    # 应用边距
                    sx0 = gx0 + margin * (gx1 - gx0)
                    sx1 = gx1 - margin * (gx1 - gx0)
                    sy0 = gy0 + margin * (gy1 - gy0)
                    sy1 = gy1 - margin * (gy1 - gy0)
                    
                    # 变换到图像坐标系
                    rect = np.array([[sx0, sy0], [sx1, sy0], [sx1, sy1], [sx0, sy1]], dtype=np.float32)
                    rect_h = np.hstack([rect, np.ones((4, 1), dtype=np.float32)])
                    poly = (H @ rect_h.T).T
                    poly = poly[:, :2] / poly[:, 2:3]
                    
                    # 转换为整数坐标并创建掩码
                    poly_int = np.round(poly).astype(np.int32)
                    mask = np.zeros((H_img, W_img), dtype=np.uint8)
                    cv2.fillPoly(mask, [poly_int], 255)
                    
                    # 提取色块的平均RGB值（pipeline处理后的working space RGB）
                    m = mask.astype(bool)
                    if not np.any(m):
                        rgb = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    else:
                        rgb = processed_array[m].reshape(-1, processed_array.shape[2]).mean(axis=0)
                    
                    rgb_list.append(rgb.astype(np.float32))
            
            return np.stack(rgb_list, axis=0)
            
        except Exception as e:
            print(f"提取pipeline处理后色卡色块失败: {e}")
            return None
    
    def _initialize_color_space_info(self):
        """初始化色彩空间信息"""
        try:
            # 验证默认色彩空间
            if self.context.color_space_manager.validate_color_space(self.context.get_input_color_space()):
                self.statusBar().showMessage(f"已设置默认输入色彩变换: {self.context.get_input_color_space()}")
            else:
                # 如果默认色彩空间无效，使用第一个可用的
                available_spaces = self.context.color_space_manager.get_available_color_spaces()
                if available_spaces:
                    self.context.set_input_color_space(available_spaces[0])
                    self.statusBar().showMessage(f"默认色彩空间无效，使用: {self.context.get_input_color_space()}")
                else:
                    print("错误: 没有可用的色彩空间")
        except Exception as e:
            print(f"初始化色彩空间信息失败: {str(e)}")
    
    def _save_image(self):
        """保存图像"""
        if not self.context.get_current_image():
            QMessageBox.warning(self, "警告", "没有可保存的图像")
            return
        
        # 检测当前是否为B&W模式
        current_film_type = self.context.get_current_film_type()
        is_bw_mode = self.context.film_type_controller.is_monochrome_type(current_film_type)
        
        # 根据B&W模式获取合适的色彩空间列表
        if is_bw_mode:
            available_spaces = self.context.color_space_manager.get_grayscale_colorspaces()
            # 如果没有灰度色彩空间，退回到所有色彩空间
            if not available_spaces:
                available_spaces = self.context.color_space_manager.get_available_color_spaces()
        else:
            available_spaces = self.context.color_space_manager.get_color_colorspaces()
            # 如果没有彩色色彩空间，退回到所有色彩空间
            if not available_spaces:
                available_spaces = self.context.color_space_manager.get_available_color_spaces()
        
        # 打开保存设置对话框
        save_dialog = SaveImageDialog(self, None, is_bw_mode, self.context.color_space_manager, self.context)
        if save_dialog.exec() != QDialog.DialogCode.Accepted:
            return
        
        # 获取保存设置
        settings = save_dialog.get_settings()
        
        self._execute_save(settings, force_dialog=True) # 强制弹出另存为
    
    def _save_image_as(self):
        """"另存为"图像"""
        if not self.context.get_current_image():
            QMessageBox.warning(self, "警告", "没有可保存的图像")
            return
        
        # 检测当前是否为B&W模式
        current_film_type = self.context.get_current_film_type()
        is_bw_mode = self.context.film_type_controller.is_monochrome_type(current_film_type)
        
        # 根据B&W模式获取合适的色彩空间列表
        if is_bw_mode:
            available_spaces = self.context.color_space_manager.get_grayscale_colorspaces()
            # 如果没有灰度色彩空间，退回到所有色彩空间
            if not available_spaces:
                available_spaces = self.context.color_space_manager.get_available_color_spaces()
        else:
            available_spaces = self.context.color_space_manager.get_color_colorspaces()
            # 如果没有彩色色彩空间，退回到所有色彩空间
            if not available_spaces:
                available_spaces = self.context.color_space_manager.get_available_color_spaces()
        
        save_dialog = SaveImageDialog(self, None, is_bw_mode, self.context.color_space_manager, self.context)
        if save_dialog.exec() != QDialog.DialogCode.Accepted:
            return
            
        settings = save_dialog.get_settings()
        self._execute_save(settings, force_dialog=True)

    def _execute_save(self, settings: dict, force_dialog: bool = False, target_file_path: str = None):
        """执行保存操作
        
        Args:
            settings: 保存设置
            force_dialog: 是否强制显示文件对话框
            target_file_path: 直接指定保存路径（用于批量保存），如果提供则跳过对话框
        """
        current_image = self.context.get_current_image()
        file_path = current_image.file_path if current_image else None
        
        # 如果提供了目标路径，直接使用，跳过对话框
        if target_file_path:
            file_path = target_file_path
        elif force_dialog or not file_path:
            extension = ".tiff" if settings["format"] == "tiff" else ".jpg"
            filter_str = "TIFF文件 (*.tiff *.tif)" if settings["format"] == "tiff" else "JPEG文件 (*.jpg *.jpeg)"
            original_filename = Path(current_image.file_path).stem if current_image and current_image.file_path else "untitled"

            # 模式判断：single / contactsheet(single crop) / contactsheet(all)
            save_mode = settings.get("save_mode", "single")
            base_dir = str(Path(current_image.file_path).parent) if current_image and current_image.file_path else ""

            # 计算编号/命名
            crops = self.context.get_all_crops()
            active_id = self.context.get_active_crop_id()
            has_contactsheet_single = (self.context.get_contactsheet_crop_rect() is not None and (not crops or active_id is None))

            def _default_name_single():
                # single 模式：CC-原文件名
                return f"CC-{original_filename}{extension}"

            def _default_name_contactsheet_all():
                # contactsheet"保存所有"：基名（没有编号），后续批量时加 -[两位数]
                return f"CC-{original_filename}{extension}"

            def _default_name_contactsheet_single():
                # contactsheet 的接触印相：固定中文后缀
                if active_id and crops:
                    # 若为正式裁剪聚焦，仍按编号命名
                    for i, c in enumerate(crops, start=1):
                        if getattr(c, 'id', None) == active_id:
                            return f"CC-{original_filename}-{i:02d}{extension}"
                # 非正式单裁剪：接触印相
                return f"CC-{original_filename}-接触印相{extension}"

            if save_mode == 'all':
                # 仅选择"目录"和"基名"，不真正返回 file_path（批量保存时逐个拼接）
                base_choice = QFileDialog.getExistingDirectory(self, "选择保存目录", base_dir)
                if not base_choice:
                    return
                # 询问基名（可选），默认 CC-原文件名
                default_basename = f"CC-{original_filename}"
                from PySide6.QtWidgets import QInputDialog
                basename, ok = QInputDialog.getText(self, "保存所有", "文件基名（将自动加 -[编号] 与扩展名）:", text=default_basename)
                if not ok or not basename:
                    basename = default_basename
                # 执行批量保存
                if "selected_files" in settings:
                    # 新的选择性批量保存
                    self._execute_selective_batch_save(settings, base_choice, basename, extension)
                else:
                    # 旧的批量保存（保存所有裁剪）
                    self._execute_batch_save(settings, base_choice, basename, extension)
                return
            else:
                # 保存单张
                if not crops:
                    # 没有crops：single模式，不加"接触印相"
                    default_filename = _default_name_single()
                elif active_id:
                    # 有crops且有激活crop：按编号命名
                    default_filename = _default_name_contactsheet_single()
                elif has_contactsheet_single:
                    # 有crops但没激活crop：接触印相命名
                    default_filename = _default_name_contactsheet_single()
                else:
                    default_filename = _default_name_single()

                default_path = str(Path(base_dir) / default_filename) if base_dir else default_filename
                file_path, _ = QFileDialog.getSaveFileName(self, "保存图像", default_path, filter_str)
        
        if not file_path:
            return
            
        # 只有非批量保存时才更新配置目录
        if not target_file_path:
            enhanced_config_manager.set_directory("save_image", str(Path(file_path).parent))
            
        try:
            # 根据保存模式确定裁剪与方向
            save_mode = settings.get("save_mode", "single")
            crop_instance = self.context.get_active_crop_instance()
            rect_norm = None
            orientation = self.context.get_current_orientation()
            if save_mode == 'single':
                if crop_instance is not None:
                    rect_norm = crop_instance.rect_norm
                    orientation = crop_instance.orientation
                elif self.context.get_contactsheet_crop_rect() is not None:
                    rect_norm = self.context.get_contactsheet_crop_rect()
                    orientation = self.context.get_current_orientation()
            # 应用裁剪与旋转
            final_image = self._apply_crop_and_rotation_for_export(current_image, rect_norm, orientation)

            # 重要：将原图转换到工作色彩空间，保持与预览一致
            print(f"导出前的色彩空间转换:")
            print(f"  原始图像色彩空间: {final_image.color_space}")
            print(f"  输入色彩变换设置: {self.context.get_input_color_space()}")
            
            # 先设置输入色彩变换
            working_image = self.context.color_space_manager.set_image_color_space(
                final_image, self.context.get_input_color_space()
            )
            # 前置IDT Gamma（导出走高精度pow）
            try:
                cs_name = self.context.get_input_color_space()
                cs_info = self.context.color_space_manager.get_color_space_info(cs_name) or {}
                idt_gamma = float(cs_info.get("gamma", 1.0))
            except Exception:
                idt_gamma = 1.0
            if abs(idt_gamma - 1.0) > 1e-6 and working_image.array is not None:
                arr = self.context.the_enlarger.pipeline_processor.math_ops.apply_power(
                    working_image.array, idt_gamma, use_optimization=False
                )
                working_image = working_image.copy_with_new_array(arr)
            # 转到工作色彩空间（ACEScg），跳过逆伽马
            working_image = self.context.color_space_manager.convert_to_working_space(
                working_image, skip_gamma_inverse=True
            )
            print(f"  转换后工作色彩空间: {working_image.color_space}")
            
            # 导出模式：提升为float64精度，确保全程高精度计算
            if working_image.array is not None:
                working_image.array = working_image.array.astype(np.float64)
                working_image.dtype = np.float64
            
            # 应用调色参数到工作空间的图像（根据设置决定是否包含曲线）
            # 导出必须使用全精度（禁用低精度LUT）+ 分块并行
            result_image = self.context.the_enlarger.apply_full_pipeline(
                working_image,
                self.context.get_current_params(),
                include_curve=settings["include_curve"],
                for_export=True
            )
            
            # 转换到输出色彩空间
            result_image = self.context.color_space_manager.convert_to_display_space(
                result_image, settings["color_space"]
            )
            
            # Convert to grayscale for B&W film types
            result_image = self._convert_to_grayscale_if_bw_mode(result_image)
            
            # 根据扩展名与设置计算"有效位深"
            ext = str(Path(file_path).suffix).lower()
            requested_bit_depth = int(settings.get("bit_depth", 8))
            if ext in [".jpg", ".jpeg"]:
                effective_bit_depth = 8
            elif ext in [".png", ".tif", ".tiff"]:
                effective_bit_depth = 16 if requested_bit_depth == 16 else 8
            else:
                effective_bit_depth = requested_bit_depth

            # 保存图像
            self.context.image_manager.save_image(
                result_image,
                file_path,
                bit_depth=effective_bit_depth,
                quality=settings.get("jpeg_quality", 95),
                export_color_space=settings.get("color_space")
            )
            
            self.statusBar().showMessage(
                f"图像已保存: {Path(file_path).name} "
                f"({effective_bit_depth}bit, {settings['color_space']})"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存图像失败: {str(e)}")

    def _execute_batch_save(self, settings: dict, target_dir: str, basename: str, extension: str):
        """执行批量保存（保存所有裁剪）
        - 命名：{basename}-[两位编号]{extension}
        - 顺序：按当前 crops 列表顺序
        - 若没有任何裁剪但存在 contactsheet 单裁剪，视为一张（编号01）
        """
        try:
            crops = self.context.get_all_crops()
            if crops:
                for i, crop in enumerate(crops, start=1):
                    # 切到该裁剪 Profile（不聚焦，避免视图闪烁），并以该裁剪的 orientation 导出
                    self.context.switch_to_crop(crop.id)
                    # 处理图像（复用 _execute_save 的核心管道，但不弹对话框）
                    filename = f"{basename}-{i:02d}{extension}"
                    file_path = str(Path(target_dir) / filename)
                    # 复用单张保存流程：构造一个"强制路径"，跳过另存弹窗
                    tmp_settings = dict(settings)
                    # 临时将 force_dialog 置 False 并直接走保存
                    # 下面直接复制 _execute_save 后半段的处理流程：
                    current_image = self.context.get_current_image()
                    crop_instance = self.context.get_active_crop_instance()
                    rect_norm = crop_instance.rect_norm if crop_instance is not None else None
                    orientation = crop_instance.orientation if crop_instance is not None else self.context.get_current_orientation()
                    final_image = self._apply_crop_and_rotation_for_export(current_image, rect_norm, orientation)
                    working_image = self.context.color_space_manager.set_image_color_space(
                        final_image, self.context.get_input_color_space()
                    )
                    # 前置IDT Gamma
                    try:
                        cs_name = self.context.get_input_color_space()
                        cs_info = self.context.color_space_manager.get_color_space_info(cs_name) or {}
                        idt_gamma = float(cs_info.get("gamma", 1.0))
                    except Exception:
                        idt_gamma = 1.0
                    if abs(idt_gamma - 1.0) > 1e-6 and working_image.array is not None:
                        arr = self.context.the_enlarger.pipeline_processor.math_ops.apply_power(
                            working_image.array, idt_gamma, use_optimization=False
                        )
                        working_image = working_image.copy_with_new_array(arr)
                    working_image = self.context.color_space_manager.convert_to_working_space(
                        working_image, skip_gamma_inverse=True
                    )
                    result_image = self.context.the_enlarger.apply_full_pipeline(
                        working_image,
                        self.context.get_current_params(),
                        include_curve=settings["include_curve"],
                        for_export=True
                    )
                    result_image = self.context.color_space_manager.convert_to_display_space(
                        result_image, settings["color_space"]
                    )
                    # Convert to grayscale for B&W film types
                    result_image = self._convert_to_grayscale_if_bw_mode(result_image)
                    # 有效位深
                    ext = extension.lower()
                    requested_bit_depth = int(settings.get("bit_depth", 8))
                    if ext in [".jpg", ".jpeg"]:
                        effective_bit_depth = 8
                    elif ext in [".png", ".tif", ".tiff"]:
                        effective_bit_depth = 16 if requested_bit_depth == 16 else 8
                    else:
                        effective_bit_depth = requested_bit_depth
                    # 保存
                    self.context.image_manager.save_image(
                        result_image,
                        file_path,
                        bit_depth=effective_bit_depth,
                        quality=settings.get("jpeg_quality", 95),
                        export_color_space=settings.get("color_space")
                    )
                self.statusBar().showMessage(f"已保存所有裁剪到: {target_dir}")
            else:
                # 无正式裁剪：若存在 contactsheet 单裁剪，视为一张（编号01）
                if self.context.get_contactsheet_crop_rect() is not None:
                    filename = f"{basename}-01{extension}"
                    file_path = str(Path(target_dir) / filename)
                    # 直接复用 _execute_save：构造一次弹窗路径
                    # 为保持简单，这里复用保存单张路径
                    # 保存图像（复制单张保存处理）：
                    current_image = self.context.get_current_image()
                    rect_norm = self.context.get_contactsheet_crop_rect()
                    orientation = self.context.get_current_orientation()
                    final_image = self._apply_crop_and_rotation_for_export(current_image, rect_norm, orientation)
                    working_image = self.context.color_space_manager.set_image_color_space(
                        final_image, self.context.get_input_color_space()
                    )
                    # 前置IDT Gamma
                    try:
                        cs_name = self.context.get_input_color_space()
                        cs_info = self.context.color_space_manager.get_color_space_info(cs_name) or {}
                        idt_gamma = float(cs_info.get("gamma", 1.0))
                    except Exception:
                        idt_gamma = 1.0
                    if abs(idt_gamma - 1.0) > 1e-6 and working_image.array is not None:
                        arr = self.context.the_enlarger.pipeline_processor.math_ops.apply_power(
                            working_image.array, idt_gamma, use_optimization=False
                        )
                        working_image = working_image.copy_with_new_array(arr)
                    working_image = self.context.color_space_manager.convert_to_working_space(
                        working_image, skip_gamma_inverse=True
                    )
                    result_image = self.context.the_enlarger.apply_full_pipeline(
                        working_image,
                        self.context.get_current_params(),
                        include_curve=settings["include_curve"],
                        for_export=True
                    )
                    result_image = self.context.color_space_manager.convert_to_display_space(
                        result_image, settings["color_space"]
                    )
                    # Convert to grayscale for B&W film types
                    result_image = self._convert_to_grayscale_if_bw_mode(result_image)
                    # 有效位深
                    ext = extension.lower()
                    requested_bit_depth = int(settings.get("bit_depth", 8))
                    if ext in [".jpg", ".jpeg"]:
                        effective_bit_depth = 8
                    elif ext in [".png", ".tif", ".tiff"]:
                        effective_bit_depth = 16 if requested_bit_depth == 16 else 8
                    else:
                        effective_bit_depth = requested_bit_depth
                    # 保存
                    self.context.image_manager.save_image(
                        result_image,
                        file_path,
                        bit_depth=effective_bit_depth,
                        quality=settings.get("jpeg_quality", 95),
                        export_color_space=settings.get("color_space")
                    )
                    self.statusBar().showMessage(f"已保存: {Path(file_path).name}")
                else:
                    self.statusBar().showMessage("没有需要保存的裁剪")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存所有失败: {e}")

    def _execute_selective_batch_save(self, settings: dict, target_dir: str, basename: str, extension: str):
        """执行选择性批量保存 - 通过状态切换+复用单张保存逻辑确保结果一致"""
        from PySide6.QtWidgets import QProgressDialog
        from PySide6.QtCore import Qt
        
        selected_files = settings.get("selected_files", [])
        if not selected_files:
            QMessageBox.information(self, "信息", "没有选择要导出的文件")
            return
            
        # 备份当前Context状态
        backup = self.context.backup_state()
        
        # 创建进度条
        progress = QProgressDialog("正在批量导出...", "取消", 0, len(selected_files), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        
        try:
            # 获取当前图像目录和预设管理器
            original_image = self.context.get_current_image()
            if not original_image:
                return
                
            current_dir = Path(original_image.file_path).parent
            auto_preset_manager = self.context.auto_preset_manager
            auto_preset_manager.set_active_directory(str(current_dir))
            
            # 获取预设数据
            presets = auto_preset_manager.get_all_presets()
            bundles = auto_preset_manager.get_all_bundles()
            
            # 按树结构顺序排序选择的文件
            ordered_files = self._sort_selected_files_by_tree_order(selected_files, presets, bundles)
            
            saved_count = 0
            
            for i, file_entry in enumerate(ordered_files):
                if progress.wasCanceled():
                    break
                
                file_type, filename, crop_id = file_entry
                
                # 显示名称
                if file_type == 'crop':
                    display_name = f"{filename}#{crop_id}"
                else:
                    display_name = filename
                    
                progress.setValue(i)
                progress.setLabelText(f"正在导出: {display_name}")
                
                try:
                    # 生成输出文件路径
                    output_filename = f"{basename}-{saved_count + 1:02d}{extension}"
                    output_path = str(Path(target_dir) / output_filename)
                    
                    # 根据文件类型进行状态切换和保存
                    success = False
                    if file_type == 'single':
                        success = self._export_with_context_switch_single(filename, presets, current_dir, output_path, settings)
                    elif file_type == 'contactsheet':
                        success = self._export_with_context_switch_contactsheet(filename, bundles, current_dir, output_path, settings)
                    elif file_type == 'crop':
                        success = self._export_with_context_switch_crop(filename, crop_id, bundles, current_dir, output_path, settings)
                    
                    if success:
                        saved_count += 1
                    
                except Exception as e:
                    print(f"导出 {display_name} 失败: {e}")
                    continue
            
            progress.setValue(len(selected_files))
            
            if saved_count > 0:
                self.statusBar().showMessage(f"已成功导出 {saved_count} 个文件到: {target_dir}")
            else:
                self.statusBar().showMessage("没有文件被导出")
                
        except Exception as e:
            QMessageBox.critical(self, "错误", f"批量导出失败: {e}")
        finally:
            # 恢复原始Context状态
            self.context.restore_state(backup)
            progress.close()

    def _sort_selected_files_by_tree_order(self, selected_files: set, presets: dict, bundles: dict) -> list:
        """按树结构顺序排序选中的文件：Single -> ContactSheet -> Crops"""
        ordered_files = []
        
        # 1. 首先添加所有Single文件（按文件名排序）
        single_files = []
        for filename in presets.keys():
            if filename in selected_files:
                single_files.append(('single', filename, None))
        single_files.sort(key=lambda x: x[1])  # 按文件名排序
        ordered_files.extend(single_files)
        
        # 2. 然后添加ContactSheet文件及其Crops（按文件名排序）
        contactsheet_files = []
        for filename in bundles.keys():
            contactsheet_files.append(filename)
        contactsheet_files.sort()  # 按文件名排序
        
        for filename in contactsheet_files:
            # 添加ContactSheet本身（如果被选中）
            if filename in selected_files:
                ordered_files.append(('contactsheet', filename, None))
            
            # 添加该ContactSheet的所有Crops（按crop_id顺序）
            bundle = bundles[filename]
            crop_entries = []
            for crop_entry in bundle.crops:
                crop_key = f"{filename}#{crop_entry.crop.id}"
                if crop_key in selected_files:
                    crop_entries.append(('crop', filename, crop_entry.crop.id))
            # Crops按照它们在bundle中的顺序
            ordered_files.extend(crop_entries)
        
        return ordered_files

    def _export_with_context_switch_single(self, filename: str, presets: dict, current_dir: Path, output_path: str, settings: dict) -> bool:
        """导出Single预设：临时切换Context状态并调用单张保存"""
        try:
            # 检查文件存在
            file_path = current_dir / filename
            if not file_path.exists():
                return False
                
            # 临时切换Context状态
            self.context.load_image(str(file_path))  # 加载图像
            self.context.load_preset(presets[filename])  # 加载预设
            
            # 调用单张保存逻辑
            self._execute_save(settings, target_file_path=output_path)
            return True
            
        except Exception as e:
            print(f"导出Single预设失败 {filename}: {e}")
            return False

    def _export_with_context_switch_contactsheet(self, filename: str, bundles: dict, current_dir: Path, output_path: str, settings: dict) -> bool:
        """导出ContactSheet预设：临时切换Context状态并调用单张保存"""
        try:
            # 检查文件存在
            file_path = current_dir / filename
            if not file_path.exists():
                return False
                
            # 临时切换Context状态
            self.context.load_image(str(file_path))  # 加载图像
            self.context.load_preset_bundle(bundles[filename])  # 加载预设Bundle
            
            # 调用单张保存逻辑
            self._execute_save(settings, target_file_path=output_path)
            return True
            
        except Exception as e:
            print(f"导出ContactSheet预设失败 {filename}: {e}")
            return False

    def _export_with_context_switch_crop(self, filename: str, crop_id: str, bundles: dict, current_dir: Path, output_path: str, settings: dict) -> bool:
        """导出Crop项目：临时切换Context状态并调用单张保存"""
        try:
            # 检查文件存在
            file_path = current_dir / filename
            if not file_path.exists():
                return False
                
            if filename not in bundles:
                return False
                
            bundle = bundles[filename]
            
            # 找到对应的crop
            target_crop = None
            for crop_entry in bundle.crops:
                if crop_entry.crop.id == crop_id or crop_entry.crop.name == crop_id:
                    target_crop = crop_entry
                    break
                    
            if not target_crop:
                print(f"未找到crop: {crop_id} in bundle {filename}")
                return False
                
            # 临时切换Context状态
            self.context.load_image(str(file_path))  # 加载图像
            self.context.load_preset_bundle(bundle)  # 加载预设Bundle
            self.context.switch_to_crop_focused(crop_id)  # 使用focused版本确保正确切换
            
            # 验证crop状态
            current_crop = self.context.get_active_crop_instance()
            if current_crop is None or current_crop.id != crop_id:
                print(f"Crop状态切换失败: 期望{crop_id}, 实际{current_crop.id if current_crop else 'None'}")
                return False
                
            print(f"[DEBUG] Crop导出: {filename}#{crop_id}")
            print(f"[DEBUG] 活动crop: {current_crop.id if current_crop else 'None'}")
            print(f"[DEBUG] 裁剪坐标: {current_crop.rect_norm if current_crop else 'None'}")
            print(f"[DEBUG] 旋转角度: {current_crop.orientation if current_crop else 'None'}")
            
            # 强制设置为single模式，确保使用crop参数
            crop_settings = settings.copy()
            crop_settings["save_mode"] = "single"  # 确保使用crop参数
            
            # 调用单张保存逻辑
            self._execute_save(crop_settings, target_file_path=output_path)
            return True
            
        except Exception as e:
            print(f"导出Crop项目失败 {filename}#{crop_id}: {e}")
            import traceback
            print(traceback.format_exc())
            return False

    
    def _reset_parameters(self):
        """重置调色参数"""
        self.context.reset_params()
    
    def _set_folder_default(self):
        """设为当前文件夹默认"""
        self.context.set_current_as_folder_default()
    
    def _load_default_curves(self):
        """加载默认曲线（Kodak Endura Paper）"""
        # 逻辑已迁移或将在Context中重新实现
        pass
    
    def _toggle_original_view(self, checked: bool):
        """切换原始图像视图"""
        if checked:
            # 显示原始图像
            if self.context.get_current_image():
                # TODO: 需要从Context获取原始代理图像
                # self.preview_widget.set_image(self.current_proxy)
                pass
        else:
            # 显示调色后的图像
            self.context._trigger_preview_update()
    
    def _reset_view(self):
        """重置预览视图"""
        self.preview_widget.reset_view()
        self.statusBar().showMessage("视图已重置")
    

    
    def _open_file_classification_manager(self):
        """打开文件分类规则管理器"""
        try:
            from divere.standalone_tools.launcher import launch_file_classification_manager
            self.file_classification_manager = launch_file_classification_manager(self)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法打开文件分类规则管理器: {str(e)}")
    
    def _open_idt_calculator(self):
        """打开精确通道分离IDT计算工具"""
        try:
            from divere.standalone_tools.launcher import launch_idt_calculator
            self.idt_calculator = launch_idt_calculator(self)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法打开IDT计算工具: {str(e)}")
    
    def _show_shortcuts_help(self):
        """显示快捷键帮助对话框"""
        try:
            dialog = ShortcutHelpDialog(self)
            dialog.exec()
        except Exception as e:
            print(f"显示快捷键帮助时出错: {e}")
            QMessageBox.warning(
                self,
                "错误",
                f"无法显示快捷键帮助：{str(e)}"
            )
    
    def _show_about(self):
        """显示关于对话框"""
        QMessageBox.about(
            self,
            "关于 DiVERE",
            "DiVERE - 数字彩色放大机\n\n"
            "版本: 0.1.0\n"
            "基于ACEScg Linear工作流的数字化胶片后期处理\n\n"
            "© 2025 V7"
        )
    
    def _update_preview(self):
        """此方法现在由Context的信号触发，或直接调用Context的方法"""
        self.context._trigger_preview_update()

    def _on_preview_updated(self, result_image: ImageData):
        self.preview_widget.set_image(result_image)
        # 图像加载后自动适应窗口（保持旧逻辑兼容）
        if self._fit_after_next_preview:
            self._fit_after_next_preview = False
            self._schedule_fit_to_window()

        # 新的精确控制fit时机
        if self._should_fit_after_image_load or self._should_fit_after_rotation:
            self._should_fit_after_image_load = False
            self._should_fit_after_rotation = False
            self._schedule_fit_to_window()

        # 更新工具可用性
        try:
            self._update_apply_contactsheet_enabled()
        except Exception:
            pass

    def _schedule_fit_to_window(self):
        """延迟执行 fit_to_window，确保图像设置完成后再适应窗口

        使用 QTimer.singleShot(0) 将调用推迟到事件循环的下一轮
        这避免了在图像设置过程中立即调整窗口，确保布局完成
        """
        try:
            QTimer.singleShot(0, self.preview_widget.fit_to_window)
        except Exception as e:
            print(f"[WARNING] _schedule_fit_to_window 失败: {e}")

    def _on_preview_updated_for_contactsheet(self, _image_data):
        """预览更新时同步更新接触印相按钮状态

        这是一个命名方法，用于替代 lambda 闭包，避免内存泄漏
        忽略 _image_data 参数，因为只需要更新按钮状态
        """
        try:
            self._update_apply_contactsheet_enabled()
        except Exception:
            pass

    def _update_apply_contactsheet_enabled(self):
        """仅在 contact sheet 模式下、进入单张 crop 聚焦时才显示并启用这两个按钮。"""
        try:
            focused = bool(getattr(self.context, '_crop_focused', False))
            kind = self.context.get_current_profile_kind()
            # 检查是否有contactsheet参数可以沿用
            has_contactsheet_params = bool(getattr(self.context._contactsheet_profile, 'params', None))
            # 检查是否有active crop参数可以应用到contactsheet
            has_active_crop = bool(getattr(self.context, '_active_crop_id', None))
            
            # 只有当 kind == 'crop' 且聚焦且有contactsheet参数时，显示并启用"沿用接触印相设置"
            should_enable_apply_from = (kind == 'crop' and focused and has_contactsheet_params)
            # 只有当 kind == 'crop' 且聚焦且有active crop时，显示并启用"应用当前设置到接触印相"
            should_enable_apply_to = (kind == 'crop' and focused and has_active_crop)
            
            try:
                self._apply_contactsheet_action.setVisible(should_enable_apply_from)
                self._apply_contactsheet_action.setEnabled(should_enable_apply_from)
            except Exception:
                pass
                
            try:
                self._apply_to_contactsheet_action.setVisible(should_enable_apply_to)
                self._apply_to_contactsheet_action.setEnabled(should_enable_apply_to)
            except Exception:
                pass
        except Exception:
            pass

    def _open_config_manager(self):
        """打开配置管理器"""
        from divere.ui.config_manager_dialog import ConfigManagerDialog
        dialog = ConfigManagerDialog(self)
        dialog.exec()

    # ===== 裁剪：UI协调槽 =====
    def _on_crop_committed(self, rect_norm: tuple):
        """处理新建裁剪"""
        try:
            # 不论当前 Profile，点击"+"后的裁剪都视为"新增 crop"
            orientation = self.context.get_current_orientation()
            crop_id = self.context.add_crop(rect_norm, orientation)
            if crop_id:
                # 切换到该 crop 的 profile（不自动聚焦）
                self.context.switch_to_crop(crop_id)
                # 刷新裁剪选择条（强制显示编号）
                try:
                    self.preview_widget._hide_single_crop_selector = False
                except Exception:
                    pass
                crops = self.context.get_all_crops()
                active_id = self.context.get_active_crop_id()
                self.preview_widget.refresh_crop_selector(crops, active_id)
                # 更新参数面板
                self.parameter_panel.on_context_params_changed(self.context.get_current_params())
        except Exception as e:
            print(f"创建裁剪失败: {e}")
    
    def _on_single_crop_committed(self, rect_norm: tuple):
        """处理单张裁剪：不创建正式crop项，仅在 contactsheet 上记录裁剪并显示 overlay。"""
        try:
            # 设置 contactsheet 裁剪（新方法）
            if hasattr(self.context, 'set_contactsheet_crop'):
                self.context.set_contactsheet_crop(rect_norm)
            else:
                # 回退：直接通过 crop_changed 信号驱动 overlay
                self.preview_widget.set_crop_overlay(rect_norm)
            # 刷新选择条：当仅存在 single 裁剪时隐藏编号
            try:
                self.preview_widget._hide_single_crop_selector = True
                crops = self.context.get_all_crops()
                active_id = self.context.get_active_crop_id()
                self.preview_widget.refresh_crop_selector(crops, active_id, is_focused=False)
            except Exception:
                pass
        except Exception as e:
            print(f"创建单张裁剪失败: {e}")
    
    def _on_crop_updated(self, crop_id_or_rect, rect_norm=None):
        """处理更新现有裁剪"""
        try:
            # 兼容两种调用方式
            if isinstance(crop_id_or_rect, str):
                # 新格式: (crop_id, rect_norm)
                crop_id = crop_id_or_rect
                if rect_norm:
                    # 更新指定裁剪
                    for crop in self.context._crops:
                        if crop.id == crop_id:
                            crop.rect_norm = rect_norm
                            break
                    self.context._autosave_timer.start()
            else:
                # 旧格式: (rect_norm)
                self.context.update_active_crop(crop_id_or_rect)
        except Exception as e:
            print(f"更新裁剪失败: {e}")

    def _on_request_focus_crop(self, crop_id=None):
        """处理聚焦裁剪请求"""
        try:
            if crop_id:
                # 一次性切到指定裁剪并聚焦，避免先显示原图再聚焦的闪烁
                self.context.switch_to_crop_focused(crop_id)
            else:
                self.context.focus_on_active_crop()
            # 刷新UI
            crops = self.context.get_all_crops()
            active_id = self.context.get_active_crop_id()
            self.preview_widget.refresh_crop_selector(crops, active_id, is_focused=True)
            # 进入聚焦后等预览更新完成再适应窗口
            self._fit_after_next_preview = True
            # 更新工具可用性
            try:
                self._update_apply_contactsheet_enabled()
            except Exception:
                pass
        except Exception as e:
            print(f"聚焦裁剪失败: {e}")

    def _on_request_restore_crop(self):
        try:
            self.context.restore_crop_preview()
            # 恢复到原图模式后，刷新选择条与显示状态
            crops = self.context.get_all_crops()
            active_id = self.context.get_active_crop_id()
            self.preview_widget.refresh_crop_selector(crops, active_id, is_focused=False)
            # 恢复到原图需要等预览更新完成再适应窗口
            self._fit_after_next_preview = True
            # 更新工具可见性/可用性
            self._update_apply_contactsheet_enabled()
        except Exception:
            pass

    def _on_request_focus_contactsheet(self):
        """进入接触印相/单张裁剪聚焦。"""
        try:
            # 若 Context 尚未记录 contactsheet 裁剪，但元数据里有 overlay，则先回写一份
            try:
                img = self.preview_widget.get_current_image_data()
                if img and img.metadata:
                    rect = img.metadata.get('crop_overlay')
                    if rect and getattr(self.context, '_contactsheet_crop_rect', None) is None:
                        if hasattr(self.context, 'set_contactsheet_crop'):
                            self.context.set_contactsheet_crop(tuple(rect))
            except Exception:
                pass

            self.context.focus_on_contactsheet_crop()
            # 刷新UI
            crops = self.context.get_all_crops()
            active_id = self.context.get_active_crop_id()
            self.preview_widget.refresh_crop_selector(crops, active_id, is_focused=True)
            self._fit_after_next_preview = True
            self._update_apply_contactsheet_enabled()
        except Exception as e:
            print(f"接触印相聚焦失败: {e}")

    def _on_custom_primaries_changed(self, primaries_xy: dict):
        """当用户在 UCS 三角拖动完成后，基于 primaries_xy 注册并切换到临时输入空间。
        遵循单向数据流：通过 Context 的 set_input_color_space 触发代理重建与预览更新。
        """
        try:
            # 规范输入为 (3,2) 数组顺序 R,G,B
            arr = np.array([primaries_xy['R'], primaries_xy['G'], primaries_xy['B']], dtype=float)
            base_name = self.context.get_input_color_space().replace("_custom", "").replace("_preset", "")
            temp_name = f"{base_name}_custom"
            # 注册/覆盖临时空间（gamma=1.0，白点D65）
            self.context.color_space_manager.register_custom_colorspace(temp_name, arr, None, gamma=1.0)
            # 切换输入色彩变换（Context 内部会重建代理并刷新预览）
            self.context.set_input_color_space(temp_name)
            # 不修改其他调色参数，仅切换输入空间
        except Exception as e:
            try:
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "提示", f"应用临时基色失败: {e}")
            except Exception:
                pass
        
    def _toggle_profiling(self, enabled: bool):
        """切换预览Profiling"""
        self.context.the_enlarger.set_profiling_enabled(enabled)
        self.context.color_space_manager.set_profiling_enabled(enabled)
        self.statusBar().showMessage("预览Profiling已开启" if enabled else "预览Profiling已关闭")
    
    def on_parameter_changed(self):
        """参数改变时的回调"""
        new_params = self.parameter_panel.get_current_params()
        self.context.update_params(new_params)
    
    def on_input_colorspace_changed(self, space_name: str):
        """输入色彩空间改变时的回调 - 需要特殊处理以重建代理"""
        self.context.set_input_color_space(space_name)
    
    def on_film_type_changed(self, film_type: str):
        """胶片类型改变时的回调"""
        # Check if transitioning from color to B&W (data loss warning)
        old_film_type = self.context.get_current_film_type()
        old_is_mono = self.context.film_type_controller.is_monochrome_type(old_film_type)
        new_is_mono = self.context.film_type_controller.is_monochrome_type(film_type)
        
        # Show warning dialog if transitioning from color to B&W
        if not old_is_mono and new_is_mono:
            if not self._confirm_color_to_bw_transition():
                # User cancelled - revert film type selection in UI
                self.parameter_panel.set_film_type(old_film_type)
                return
        
        # Proceed with film type change
        self.context.set_current_film_type(film_type)

    def on_context_film_type_changed(self, film_type: str):
        """ApplicationContext胶片类型改变时的回调 - 同步UI"""
        # Update film type dropdown
        self.parameter_panel.set_film_type(film_type)
        
        # Apply neutralization for B&W film types immediately
        # This prevents the preview flash issue when loading B&W presets
        self._apply_bw_neutralization_if_needed(film_type)
    
    def _apply_bw_neutralization_if_needed(self, film_type: str):
        """Apply neutral values for B&W film types"""
        # Check if this is a B&W film type
        if not self.context.film_type_controller.is_monochrome_type(film_type):
            return
        
        # Get current parameters
        params = self.context.get_current_params()
        
        # Set neutral values for B&W mode
        # 1. Set IDT color transformation to identity
        params.input_color_space_name = "Identity"  # This should result in identity transform
        
        # 2. Set RGB gains to (0.0, 0.0, 0.0)
        params.rgb_gains = (0.0, 0.0, 0.0)
        
        # 3. Set density matrix to identity
        params.density_matrix = np.eye(3)
        params.density_matrix_name = "Identity"
        
        # 4. Set RGB curves to Ilford Multigrade 2 (proper B&W curve)
        # Load the curve from the curve manager if available, otherwise use linear fallback
        try:
            # Get the curve from configuration
            from divere.utils.enhanced_config_manager import enhanced_config_manager
            curve_files = enhanced_config_manager.get_config_files("curves")
            ilford_curve = None
            
            for curve_file in curve_files:
                if "Ilford MGFB 2" in str(curve_file) or "Ilford_MGFB_2" in str(curve_file):
                    curve_data = enhanced_config_manager.load_config_file(curve_file)
                    if curve_data and "curves" in curve_data:
                        # Use RGB curve if available
                        if "RGB" in curve_data["curves"]:
                            ilford_curve = curve_data["curves"]["RGB"]
                        break
            
            # Apply Ilford curve if found, otherwise use linear
            if ilford_curve:
                params.curve_points = ilford_curve
                params.curve_points_r = curve_data["curves"].get("R", [(0.0, 0.0), (1.0, 1.0)])
                params.curve_points_g = curve_data["curves"].get("G", [(0.0, 0.0), (1.0, 1.0)])
                params.curve_points_b = curve_data["curves"].get("B", [(0.0, 0.0), (1.0, 1.0)])
                params.density_curve_name = "Ilford MGFB 2"
            else:
                # Fallback to linear
                linear_curve = [(0.0, 0.0), (1.0, 1.0)]
                params.curve_points_r = linear_curve
                params.curve_points_g = linear_curve
                params.curve_points_b = linear_curve
                params.curve_points = linear_curve
                params.density_curve_name = "linear"
                
        except Exception as e:
            print(f"Warning: Could not load Ilford Multigrade 2 curve, using linear: {e}")
            # Fallback to linear
            linear_curve = [(0.0, 0.0), (1.0, 1.0)]
            params.curve_points_r = linear_curve
            params.curve_points_g = linear_curve
            params.curve_points_b = linear_curve
            params.curve_points = linear_curve
            params.density_curve_name = "linear"
        
        # Apply the neutralized parameters
        self.context.update_params(params)
        
        # Trigger auto-save after B&W override
        self.context.autosave_requested.emit()

    def _confirm_color_to_bw_transition(self) -> bool:
        """Confirm color to B&W transition with user"""
        from PySide6.QtWidgets import QMessageBox
        
        reply = QMessageBox.question(
            self, 
            "胶片类型变更确认", 
            "切换到黑白模式将覆盖当前的彩色校正设置：\n"
            "• IDT色彩变换将设为单位矩阵\n"
            "• 密度矩阵将设为单位矩阵\n" 
            "• RGB增益将设为 (0, 0, 0)\n"
            "• RGB曲线将设为 Ilford Multigrade 2\n\n"
            "当前的彩色校正信息将丢失。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        return reply == QMessageBox.StandardButton.Yes

    def get_current_params(self) -> ColorGradingParams:
        """获取当前调色参数"""
        return self.context.get_current_params()
    
    def set_current_params(self, params: ColorGradingParams):
        """设置当前调色参数"""
        self.context.update_params(params)
    
    def _on_image_rotated(self, direction):
        """处理图像旋转：委托给 ApplicationContext 维护朝向与预览更新"""
        try:
            self.context.rotate(int(direction))
            self._should_fit_after_rotation = True  # 设置旋转后fit标志
        except Exception:
            pass
    
    def _on_rotation_completed(self):
        """处理旋转完成信号：设置fit标志"""
        self._should_fit_after_rotation = True
    
    def _on_lut_export_requested(self, lut_type: str, file_path: str, size: int):
        """处理LUT导出请求"""
        try:
            # 获取当前参数
            current_params = self.context.get_current_params()
            
            if lut_type == "input_transform":
                success = self._export_input_transform_lut(current_params, file_path, size)
            elif lut_type == "color_correction":
                success = self._export_color_correction_lut(current_params, file_path, size)
            elif lut_type == "density_curve":
                success = self._export_density_curve_lut(current_params, file_path, size)
            else:
                success = False
                
            if success:
                self.statusBar().showMessage(f"{lut_type} LUT已导出到: {file_path}")
            else:
                self.statusBar().showMessage(f"{lut_type} LUT导出失败")
                
        except Exception as e:
            print(f"LUT导出失败: {e}")
            self.statusBar().showMessage(f"LUT导出失败: {e}")
    
    def _export_input_transform_lut(self, params, file_path: str, size: int) -> bool:
        """导出输入设备转换LUT (3D) - 包含IDT Gamma + 色彩空间转换"""
        try:
            from divere.utils.lut_generator.interface import DiVERELUTInterface
            
            # 获取当前IDT Gamma值
            try:
                cs_name = self.context.get_input_color_space()
                cs_info = self.context.color_space_manager.get_color_space_info(cs_name) or {}
                idt_gamma = float(cs_info.get("gamma", 1.0))
            except Exception:
                idt_gamma = 1.0
            
            # 创建专门的输入设备转换配置
            idt_config = {
                "idt_gamma": idt_gamma,
                "context": self.context,
                "input_colorspace_name": params.input_color_space_name
            }
            
            lut_interface = DiVERELUTInterface()
            return lut_interface.generate_input_device_transform_lut(
                idt_config, file_path, size
            )
            
        except Exception as e:
            print(f"导出输入转换LUT失败: {e}")
            return False
    
    def _export_color_correction_lut(self, params, file_path: str, size: int) -> bool:
        """导出反相校色LUT (3D, 不含密度曲线)"""
        try:
            from divere.utils.lut_generator.interface import DiVERELUTInterface
            
            # 创建不含密度曲线的参数副本
            color_params = params.copy()
            color_params.enable_density_curve = False
            color_params.curve_points = [(0.0, 0.0), (1.0, 1.0)]
            color_params.curve_points_r = [(0.0, 0.0), (1.0, 1.0)]
            color_params.curve_points_g = [(0.0, 0.0), (1.0, 1.0)]
            color_params.curve_points_b = [(0.0, 0.0), (1.0, 1.0)]
            
            # 管线配置
            pipeline_config = {
                "params": color_params,
                "context": self.context,
                "the_enlarger": self.context.the_enlarger
            }
            
            lut_interface = DiVERELUTInterface()
            return lut_interface.generate_pipeline_lut(
                pipeline_config, file_path, "3D", size
            )
            
        except Exception as e:
            print(f"导出反相校色LUT失败: {e}")
            return False
    
    def _export_density_curve_lut(self, params, file_path: str, size: int) -> bool:
        """导出密度曲线LUT (1D) - 在密度空间应用曲线"""
        try:
            from divere.utils.lut_generator.interface import DiVERELUTInterface
            
            # 提取曲线数据
            curves = {
                'R': params.curve_points_r or [(0.0, 0.0), (1.0, 1.0)],
                'G': params.curve_points_g or [(0.0, 0.0), (1.0, 1.0)],
                'B': params.curve_points_b or [(0.0, 0.0), (1.0, 1.0)]
            }
            
            # 如果有RGB通用曲线，使用它
            if params.curve_points and params.curve_points != [(0.0, 0.0), (1.0, 1.0)]:
                curves['R'] = curves['G'] = curves['B'] = params.curve_points
            
            # 使用密度曲线专用方法（包含屏幕反光补偿）
            lut_interface = DiVERELUTInterface()
            return lut_interface.generate_density_curve_lut(
                curves, file_path, size, params.screen_glare_compensation
            )
            
        except Exception as e:
            print(f"导出密度曲线LUT失败: {e}")
            return False
    
    def _on_glare_compensation_interaction_started(self, compensation_value: float):
        """处理屏幕反光补偿交互开始"""
        try:
            # 启用预览中的black cut-off显示
            self.preview_widget.set_black_cutoff_display(True, compensation_value)
        except Exception as e:
            print(f"启用black cut-off显示失败: {e}")
    
    def _on_glare_compensation_interaction_ended(self):
        """处理屏幕反光补偿交互结束"""
        try:
            # 关闭预览中的black cut-off显示
            self.preview_widget.set_black_cutoff_display(False)
        except Exception as e:
            print(f"关闭black cut-off显示失败: {e}")
    
    def _on_glare_compensation_realtime_update(self, compensation_value: float):
        """处理屏幕反光补偿实时更新"""
        try:
            # 实时更新cut-off显示的补偿值
            self.preview_widget.update_cutoff_compensation(compensation_value)
        except Exception as e:
            print(f"实时更新cut-off显示失败: {e}")
    
    def _show_status_message(self, message: str, timeout: int = 2000):
        """在状态栏显示消息"""
        try:
            self.statusBar().showMessage(message, timeout)
        except Exception:
            pass

    def closeEvent(self, event):
        """窗口关闭时的清理工作

        Qt 会自动断开对象销毁时的信号连接，但我们需要：
        1. 停止所有定时器
        2. 停止后台线程池
        3. 清理缓存资源
        4. 显式断开全局对象的信号连接（ApplicationContext）
        """
        print("[DEBUG] MainWindow.closeEvent: 开始清理资源")

        # 1. 停止 ApplicationContext 的后台处理
        try:
            # 优先使用 ApplicationContext 的 cleanup() 方法（统一清理逻辑）
            if hasattr(self.context, 'cleanup'):
                self.context.cleanup()
            else:
                # 向后兼容：如果 cleanup 方法不存在，使用旧逻辑
                if hasattr(self.context, '_autosave_timer') and self.context._autosave_timer:
                    self.context._autosave_timer.stop()
                if hasattr(self.context, 'thread_pool') and self.context.thread_pool:
                    self.context.thread_pool.waitForDone(1000)

            print("[DEBUG] ApplicationContext 清理完成")
        except Exception as e:
            print(f"[WARNING] ApplicationContext 清理失败: {e}")

        # 2. 清理 PreviewWidget 的定时器
        try:
            if hasattr(self, 'preview_widget') and self.preview_widget:
                self.preview_widget.cleanup()
            print("[DEBUG] PreviewWidget 定时器清理完成")
        except Exception as e:
            print(f"[WARNING] PreviewWidget 清理失败: {e}")

        # 3. 缓存清理（已由 ApplicationContext.cleanup() 处理）
        # 注意：缓存清理现在集成在 ApplicationContext.cleanup() 中
        # 保留这段代码作为向后兼容（如果 cleanup() 方法不存在或失败）
        try:
            # 只有在 ApplicationContext 没有 cleanup 方法时才执行这里的清理
            if not hasattr(self.context, 'cleanup'):
                if hasattr(self.context, 'image_manager') and self.context.image_manager:
                    self.context.image_manager.clear_cache()
                if (hasattr(self.context, 'the_enlarger') and self.context.the_enlarger and
                    hasattr(self.context.the_enlarger, 'lut_processor')):
                    self.context.the_enlarger.lut_processor.clear_cache()
                if hasattr(self.context, 'color_space_manager') and self.context.color_space_manager:
                    self.context.color_space_manager.clear_convert_cache()
                print("[DEBUG] 缓存清理完成（向后兼容路径）")
        except Exception as e:
            print(f"[WARNING] 缓存清理失败: {e}")

        # 4. 显式断开 ApplicationContext 的信号连接（高风险连接）
        # 注意：这一步在大多数情况下不是必需的，因为 Qt 会自动处理
        # 但为了保险起见，我们显式断开全局对象的连接
        try:
            # 使用列表收集所有信号，避免在断开时出错
            signals_to_disconnect = [
                (self.context.preview_updated, self),
                (self.context.status_message_changed, self),
                (self.context.image_loaded, self),
                (self.context.autosave_requested, self),
                (self.context.curves_config_reloaded, self),
                (self.context.rotation_completed, self),
                (self.context.crop_changed, self),
                (self.context.film_type_changed, self),
            ]

            for signal, receiver in signals_to_disconnect:
                try:
                    # 尝试断开与指定接收者相关的所有连接
                    # disconnect() 不带参数会断开所有连接，我们只想断开与 self 相关的
                    # 但 PySide6 的 disconnect 不支持 receiver 参数，所以使用 try-except
                    signal.disconnect(self)
                except (RuntimeError, TypeError):
                    # 信号可能已经断开或无连接
                    pass

            print("[DEBUG] ApplicationContext 信号断开完成")
        except Exception as e:
            print(f"[WARNING] 信号断开失败: {e}")

        # 5. 调用父类的 closeEvent
        super().closeEvent(event)
        print("[DEBUG] MainWindow.closeEvent: 清理完成")

# 移除 Worker 相关类定义
# class _PreviewWorkerSignals(QObject): ...
# class _PreviewWorker(QRunnable): ...
