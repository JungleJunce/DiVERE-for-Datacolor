"""
参数面板
包含所有调色参数的控件
"""

from typing import Optional, Tuple
import numpy as np
import json
from pathlib import Path

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QSlider, QDoubleSpinBox, QSpinBox, QComboBox,
    QGroupBox, QPushButton, QCheckBox, QTabWidget,
    QScrollArea, QMessageBox, QInputDialog, QFileDialog,
    QApplication
)
from PySide6.QtCore import Qt, Signal, QSize, QTimer
from PySide6.QtGui import QIcon

from divere.core.data_types import ColorGradingParams, Preset
from divere.core.app_context import ApplicationContext
from divere.ui.curve_editor_widget import CurveEditorWidget
from divere.ui.ucs_triangle_widget import UcsTriangleWidget
from divere.core.color_space import xy_to_uv, uv_to_xy
from divere.utils.enhanced_config_manager import enhanced_config_manager
from divere.utils.colorchecker_loader import validate_colorchecker_workspace_compatibility


class PrecisionSlider(QSlider):
    """支持Command/Ctrl键精细调整的自定义滑块"""
    
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)
        self._precision_mode = False
        self._accumulated_delta = 0.0  # 累积的鼠标移动
        self._last_mouse_pos = None
        self._precision_ratio = 5  # 5:1的精细比例
        
    def mousePressEvent(self, event):
        modifiers = QApplication.keyboardModifiers()
        self._precision_mode = bool(modifiers & Qt.ControlModifier)
        
        if self._precision_mode:
            # 精细模式：记录起始位置，不调用父类
            self._last_mouse_pos = event.pos().x()
            self._accumulated_delta = 0.0
            # 在精细模式下，我们仍然需要处理点击位置来移动滑块
            # 但是要阻止默认的跳转行为
            self.setSliderDown(True)
        else:
            # 正常模式：完全交给父类处理
            super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        if self._precision_mode and self._last_mouse_pos is not None:
            # 精细模式：自定义移动逻辑
            current_pos = event.pos().x()
            mouse_delta = current_pos - self._last_mouse_pos
            
            # 累积鼠标移动（10:1比例）
            self._accumulated_delta += mouse_delta / self._precision_ratio
            
            # 当累积移动超过1个slider单位时才实际移动
            if abs(self._accumulated_delta) >= 1.0:
                slider_delta = int(self._accumulated_delta)
                new_value = self.value() + slider_delta
                new_value = max(self.minimum(), min(self.maximum(), new_value))
                self.setValue(new_value)
                self._accumulated_delta -= slider_delta  # 保留小数部分
            
            self._last_mouse_pos = current_pos
        else:
            # 正常模式：完全交给父类处理
            super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        if self._precision_mode:
            # 清理精细模式状态
            self._precision_mode = False
            self._accumulated_delta = 0.0
            self._last_mouse_pos = None
            self.setSliderDown(False)
        super().mouseReleaseEvent(event)


class ParameterPanel(QWidget):
    """参数面板 (重构版)"""
    
    parameter_changed = Signal()
    auto_color_requested = Signal()
    auto_color_iterative_requested = Signal()
    pick_neutral_point_requested = Signal()  # 请求进入中性点选择模式（取点）
    apply_neutral_color_requested = Signal(int)  # 请求应用中性色迭代，参数为色温(K)
    neutral_white_point_changed = Signal(int)  # 中性点色温变化信号，参数为新色温(K)
    input_colorspace_changed = Signal(str)
    film_type_changed = Signal(str)
    colorchecker_changed = Signal(str)  # 色卡类型变化信号，参数为文件名
    monochrome_preview_changed = Signal(bool)  # 黑白预览状态变化信号

    # Signals for complex actions requiring coordination
    ccm_optimize_requested = Signal()
    save_custom_colorspace_requested = Signal(dict)
    save_density_matrix_requested = Signal(np.ndarray, str)  # 保存密度矩阵信号(矩阵, 当前名称)
    toggle_color_checker_requested = Signal(bool)
    proxy_size_changed = Signal(int)  # Proxy长边尺寸变化信号
    # 色卡变换信号
    cc_flip_horizontal_requested = Signal()
    cc_flip_vertical_requested = Signal()
    cc_rotate_left_requested = Signal()
    cc_rotate_right_requested = Signal()
    # 新增：基色(primaries)改变（拖动结束时触发，负担轻）
    custom_primaries_changed = Signal(dict)
    # LUT导出信号
    lut_export_requested = Signal(str, str, int)  # (lut_type, file_path, size)
    # 屏幕反光补偿交互信号
    glare_compensation_interaction_started = Signal(float)  # 当前补偿值
    glare_compensation_interaction_ended = Signal()
    glare_compensation_realtime_update = Signal(float)  # 实时更新补偿值
    # 读取并保存色卡颜色信号
    save_colorchecker_colors_requested = Signal()
    
    def __init__(self, context: ApplicationContext):
        super().__init__()
        self.context = context
        self.current_params = self.context.get_current_params().copy()
        self.current_film_type = "color_negative_c41"  # Default film type
        self.selected_colorchecker_file = "original_color_cc24data.json"  # Default reference

        self._is_updating_ui = False
        self.context.params_changed.connect(self.on_context_params_changed)
        self.context.image_loaded.connect(self.on_image_loaded)

        # 长按重复触发的定时器
        self._repeat_timer = None
        self._repeat_callback = None

        self._create_ui()
        self._connect_signals()
        
    def on_context_params_changed(self, params: ColorGradingParams):
        """当Context中的参数改变时，更新UI"""
        self.current_params = params.copy()
        self.update_ui_from_params()
    
    def on_image_loaded(self):
        """当新图片加载时的处理"""
        # 清理不属于当前预设的修改状态项目
        # 这样每次切换图片时，下拉框会重新开始，只保留当前需要的状态
        self._cleanup_stale_modified_items()

    def initialize_defaults(self, initial_params: ColorGradingParams):
        """由主窗口调用，在加载图像后设置并应用默认参数"""
        self.current_params = initial_params.copy()
        self.update_ui_from_params()
        
        # 设置proxy尺寸控件的初始值
        proxy_size = enhanced_config_manager.get_ui_setting("proxy_max_size", 2000)
        self.proxy_size_spinbox.setValue(proxy_size)

    def _create_ui(self):
        layout = QVBoxLayout(self)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        
        tab_widget = QTabWidget()
        tab_widget.addTab(self._create_basic_tab(), "输入色彩科学")
        tab_widget.addTab(self._create_density_tab(), "密度与矩阵")
        tab_widget.addTab(self._create_rgb_tab(), "RGB曝光")
        tab_widget.addTab(self._create_curve_tab(), "密度曲线")
        tab_widget.addTab(self._create_debug_tab(), "管线控制")
        
        content_layout.addWidget(tab_widget)
        content_layout.addStretch()

        # 初始隐藏所有辅助调整控件
        # 在所有tabs创建完成后调用，确保UI层次结构已完全建立
        for widget in self.matrix_helper_widgets:
            widget.setVisible(False)

        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)

    def _create_basic_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Film Type Group (added above colorspace group)
        film_type_group = self._create_film_type_group()
        layout.addWidget(film_type_group)
        
        colorspace_group = QGroupBox("输入色彩变换")
        colorspace_layout = QGridLayout(colorspace_group)
        # IDT Gamma（在下拉菜单上方）
        self.idt_gamma_slider = PrecisionSlider(Qt.Orientation.Horizontal)
        self.idt_gamma_spinbox = QDoubleSpinBox()
        self._setup_slider_spinbox(self.idt_gamma_slider, self.idt_gamma_spinbox, 500, 2800, 0.5, 2.8, 0.005)
        colorspace_layout.addWidget(QLabel("IDT Gamma:"), 0, 0)
        colorspace_layout.addWidget(self.idt_gamma_slider, 0, 1)
        colorspace_layout.addWidget(self.idt_gamma_spinbox, 0, 2)
        self.input_colorspace_combo = QComboBox()
        spaces = self.context.color_space_manager.get_idt_color_spaces()
        for space in spaces:
            self.input_colorspace_combo.addItem(space, space)
        colorspace_layout.addWidget(QLabel("IDT 基色:"), 1, 0)
        colorspace_layout.addWidget(self.input_colorspace_combo, 1, 1, 1, 2)
        layout.addWidget(colorspace_group)
        
        # --- Spectral Sharpening Section ---
        self.enable_scanner_spectral_checkbox = QCheckBox("扫描仪光谱锐化（串扰校正）")
        layout.addWidget(self.enable_scanner_spectral_checkbox)

        self.ucs_widget = UcsTriangleWidget()
        self.ucs_widget.setVisible(False)
        layout.addWidget(self.ucs_widget)

        # 色卡选择器和变换按钮的水平布局
        cc_selector_layout = QHBoxLayout()
        self.cc_selector_checkbox = QCheckBox("色卡选择器")
        self.cc_selector_checkbox.setVisible(False)
        cc_selector_layout.addWidget(self.cc_selector_checkbox)
        
        # 色卡变换按钮
        self.cc_flip_h_button = QPushButton("↔")
        self.cc_flip_h_button.setToolTip("水平翻转色卡选择器")
        self.cc_flip_h_button.setFixedWidth(30)
        self.cc_flip_h_button.setVisible(False)
        cc_selector_layout.addWidget(self.cc_flip_h_button)
        
        self.cc_flip_v_button = QPushButton("↕")
        self.cc_flip_v_button.setToolTip("竖直翻转色卡选择器")
        self.cc_flip_v_button.setFixedWidth(30)
        self.cc_flip_v_button.setVisible(False)
        cc_selector_layout.addWidget(self.cc_flip_v_button)
        
        self.cc_rotate_l_button = QPushButton("↶")
        self.cc_rotate_l_button.setToolTip("左旋转色卡选择器")
        self.cc_rotate_l_button.setFixedWidth(30)
        self.cc_rotate_l_button.setVisible(False)
        cc_selector_layout.addWidget(self.cc_rotate_l_button)
        
        self.cc_rotate_r_button = QPushButton("↷")
        self.cc_rotate_r_button.setToolTip("右旋转色卡选择器")
        self.cc_rotate_r_button.setFixedWidth(30)
        self.cc_rotate_r_button.setVisible(False)
        cc_selector_layout.addWidget(self.cc_rotate_r_button)
        
        cc_selector_layout.addStretch()  # 推到左边
        
        # 色卡类型选择下拉菜单
        self.colorchecker_combo = QComboBox()
        self.colorchecker_combo.setToolTip("选择色卡参考类型")
        self.colorchecker_combo.setVisible(False)
        self.colorchecker_combo.setMinimumWidth(150)
        self._populate_colorchecker_combo()
        cc_selector_layout.addWidget(self.colorchecker_combo)
        
        layout.addLayout(cc_selector_layout)

        # 读取并保存色卡颜色按钮
        self.save_colorchecker_colors_button = QPushButton("读取并保存色卡密度")
        self.save_colorchecker_colors_button.setToolTip("从当前色卡选择器中读取24个色块的平均密度，转换为透过率并保存为JSON文件")
        self.save_colorchecker_colors_button.setVisible(False)
        self.save_colorchecker_colors_button.setEnabled(False)
        layout.addWidget(self.save_colorchecker_colors_button)

        # 光谱锐化（硬件校正）优化配置开关
        spectral_config_layout = QHBoxLayout()
        self.optimize_idt_checkbox = QCheckBox("优化IDT线性变换基色（光源-传感器串扰校正）")
        self.optimize_idt_checkbox.setToolTip("优化IDT线性变换基色，消除光源-传感器串扰")
        self.optimize_idt_checkbox.setChecked(False)  # 默认禁用
        self.optimize_idt_checkbox.setVisible(False)
        spectral_config_layout.addWidget(self.optimize_idt_checkbox)
        
        self.optimize_density_matrix_checkbox = QCheckBox("优化数字Mask（密度计-负片串扰校正）")
        self.optimize_density_matrix_checkbox.setToolTip("优化数字Mask消除扫描仪-负片染料引入的串扰")
        self.optimize_density_matrix_checkbox.setChecked(False)  # 默认禁用
        self.optimize_density_matrix_checkbox.setVisible(False)
        spectral_config_layout.addWidget(self.optimize_density_matrix_checkbox)
        
        layout.addLayout(spectral_config_layout)

        self.ccm_optimize_button = QPushButton("开始优化")
        self.ccm_optimize_button.setToolTip("从色卡选择器读取24个颜色并优化参数")
        self.ccm_optimize_button.setVisible(False)
        self.ccm_optimize_button.setEnabled(False)
        layout.addWidget(self.ccm_optimize_button)

        self.save_input_colorspace_button = QPushButton("保存IDT基色结果")
        self.save_input_colorspace_button.setToolTip("将当前UCS三角形对应的基色与白点保存为JSON文件")
        self.save_input_colorspace_button.setVisible(False)
        layout.addWidget(self.save_input_colorspace_button)

        self.save_matrix_button_afterOpt = QPushButton("保存数字Mask结果")
        self.save_matrix_button_afterOpt.setToolTip("将当前数字Mask保存到文件")
        self.save_matrix_button_afterOpt.setVisible(False)
        layout.addWidget(self.save_matrix_button_afterOpt)


        layout.addStretch()
        return widget

    def _create_film_type_group(self) -> QGroupBox:
        """创建胶片类型选择组件"""
        film_type_group = QGroupBox("胶片类型")
        film_type_layout = QGridLayout(film_type_group)
        
        self.film_type_combo = QComboBox()
        # Add film type options with Chinese display names and English values
        film_types = [
            ("彩色负片C41", "color_negative_c41"),
            ("彩色电影负片ECN2", "color_negative_ecn2"), 
            ("彩色反转片", "color_reversal"),
            ("黑白负片", "b&w_negative"),
            ("黑白反转片", "b&w_reversal"),
            ("数字", "digital")
        ]
        
        for display_name, value in film_types:
            self.film_type_combo.addItem(display_name, value)
        
        film_type_layout.addWidget(QLabel("胶片类型（TODO）:"), 0, 0)
        film_type_layout.addWidget(self.film_type_combo, 0, 1, 1, 2)
        
        return film_type_group

    def _populate_colorchecker_combo(self):
        """填充色卡类型下拉菜单"""
        try:
            from divere.utils.app_paths import resolve_data_path
            colorchecker_dir = resolve_data_path("config", "colorchecker")
        except Exception:
            colorchecker_dir = Path(__file__).parent.parent / "config" / "colorchecker"
        
        if not colorchecker_dir.exists():
            return
            
        # 扫描色卡文件（两种格式）
        chart_files = []
        for file_path in colorchecker_dir.glob("*_cc24data.json"):
            chart_files.append(file_path.name)
        for file_path in colorchecker_dir.glob("*_colorchecker_format.json"):
            chart_files.append(file_path.name)
        
        # 排序并添加到下拉菜单
        chart_files.sort()
        for filename in chart_files:
            display_name = self._format_colorchecker_display_name(filename)
            self.colorchecker_combo.addItem(display_name, filename)
            
        # 设置默认选择
        default_index = self.colorchecker_combo.findData("original_color_cc24data.json")
        if default_index >= 0:
            self.colorchecker_combo.setCurrentIndex(default_index)
    
    def _format_colorchecker_display_name(self, filename: str) -> str:
        """将文件名格式化为显示名称"""
        # 移除后缀
        name = filename.replace("_cc24data.json", "").replace("_colorchecker_format.json", "")
        
        # 处理特殊情况
        if name == "colorchecker_acescg":
            return "标准色卡 (ACEScg)"
        
        # 将下划线替换为空格，首字母大写
        parts = name.split("_")
        formatted_parts = []
        for part in parts:
            if part.upper() in ["RGB", "ACEScg", "D50", "D60"]:
                formatted_parts.append(part.upper())
            else:
                formatted_parts.append(part.capitalize())
        
        return " ".join(formatted_parts)
    
    def get_selected_colorchecker_file(self) -> str:
        """获取当前选择的色卡参考文件"""
        return self.selected_colorchecker_file

    def revert_colorchecker_selection(self):
        """回滚到默认或上一个有效的色卡选择"""
        try:
            # 首先尝试回滚到默认选择
            default_index = self.colorchecker_combo.findData("original_color_cc24data.json")
            if default_index >= 0:
                self.colorchecker_combo.setCurrentIndex(default_index)
                self.selected_colorchecker_file = "original_color_cc24data.json"
                print("已回滚到默认色卡: original_color_cc24data.json")
            else:
                # 如果没有默认选择，选择第一个可用的
                if self.colorchecker_combo.count() > 0:
                    self.colorchecker_combo.setCurrentIndex(0)
                    self.selected_colorchecker_file = self.colorchecker_combo.itemData(0)
                    print(f"已回滚到第一个可用色卡: {self.selected_colorchecker_file}")
        except Exception as e:
            print(f"回滚色卡选择失败: {e}")

    def _create_density_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        inversion_group = QGroupBox("密度反相")
        inversion_layout = QGridLayout(inversion_group)
        self.density_gamma_slider = PrecisionSlider(Qt.Orientation.Horizontal)
        self.density_gamma_spinbox = QDoubleSpinBox()
        self.density_dmax_slider = PrecisionSlider(Qt.Orientation.Horizontal)
        self.density_dmax_spinbox = QDoubleSpinBox()
        self._setup_slider_spinbox(self.density_gamma_slider, self.density_gamma_spinbox, 100, 4000, 0.1, 4.0, 0.005)
        self._setup_slider_spinbox(self.density_dmax_slider, self.density_dmax_spinbox, 0, 4800, 0.0, 4.8, 0.005)
        inversion_layout.addWidget(QLabel("密度反差:"), 0, 0)
        inversion_layout.addWidget(self.density_gamma_slider, 0, 1)
        inversion_layout.addWidget(self.density_gamma_spinbox, 0, 2)
        inversion_layout.addWidget(QLabel("最大密度:"), 1, 0)
        inversion_layout.addWidget(self.density_dmax_slider, 1, 1)
        inversion_layout.addWidget(self.density_dmax_spinbox, 1, 2)
        layout.addWidget(inversion_group)

        matrix_group = QGroupBox("数字Mask（密度校正矩阵）")
        matrix_layout = QVBoxLayout(matrix_group)
        self.matrix_editor_widgets = []
        matrix_grid = QGridLayout()
        for i in range(3):
            row = []
            for j in range(3):
                spinbox = QDoubleSpinBox()
                spinbox.setRange(-10.0, 10.0); spinbox.setSingleStep(0.01); spinbox.setDecimals(4); spinbox.setFixedWidth(80)
                matrix_grid.addWidget(spinbox, i, j)
                row.append(spinbox)
            self.matrix_editor_widgets.append(row)
        
        combo_layout = QHBoxLayout()
        combo_layout.addWidget(QLabel("预设:"))
        self.matrix_combo = QComboBox()
        self.matrix_combo.addItem("自定义", "custom")
        available = self.context.the_enlarger.pipeline_processor.get_available_matrices()
        for matrix_id in available:
            data = self.context.the_enlarger.pipeline_processor.get_matrix_data(matrix_id)
            if data: self.matrix_combo.addItem(data.get("name", matrix_id), matrix_id)
        combo_layout.addWidget(self.matrix_combo)
        combo_layout.addStretch()
        
        # 保存矩阵按钮
        self.save_matrix_button = QPushButton("保存矩阵")
        self.save_matrix_button.setToolTip("将当前密度矩阵保存到文件")
        combo_layout.addWidget(self.save_matrix_button)
        
        matrix_layout.addLayout(combo_layout)
        matrix_layout.addLayout(matrix_grid)

        # === 矩阵辅助调整控件 ===
        # 添加控制辅助调整显示的复选框
        self.aesthetic_matrix_adjustment_checkbox = QCheckBox("美学式调整数字Mask")
        self.aesthetic_matrix_adjustment_checkbox.setToolTip("显示/隐藏矩阵辅助调整按钮")
        self.aesthetic_matrix_adjustment_checkbox.setChecked(False)  # 默认不显示
        matrix_layout.addWidget(self.aesthetic_matrix_adjustment_checkbox)

        helper_grid = QGridLayout()
        helper_grid.setSpacing(5)

        # 设置列宽比例：缩小标签列宽度，让按钮列占据更多空间
        # 偶数列（0,2,4）是标签列，设为较小宽度
        # 奇数列（1,3,5）是按钮列，设为较大宽度
        for i in range(6):  # 总共6列：3组 x 2列
            if i % 2 == 0:  # 标签列
                helper_grid.setColumnStretch(i, 0)  # 最小宽度
            else:  # 按钮列
                helper_grid.setColumnStretch(i, 1)  # 自动扩展

        # 保存按钮引用以便状态管理
        self.matrix_helper_buttons = []
        # 保存所有辅助调整相关的控件以便切换可见性
        self.matrix_helper_widgets = []

        channel_names = ["红通道", "绿通道", "蓝通道"]

        for col in range(3):
            # 列标题
            channel_label = QLabel(channel_names[col])
            channel_label.setAlignment(Qt.AlignCenter)
            helper_grid.addWidget(channel_label, 0, col * 2, 1, 2)
            self.matrix_helper_widgets.append(channel_label)

            # 纯度标签
            purity_label = QLabel("纯度:")
            purity_label.setFixedWidth(35)  # 限制宽度
            purity_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)  # 右对齐
            helper_grid.addWidget(purity_label, 1, col * 2)
            self.matrix_helper_widgets.append(purity_label)

            # 纯度按钮
            purity_layout = QHBoxLayout()
            purity_layout.setSpacing(2)
            purity_minus = QPushButton("-")
            purity_minus.setFixedWidth(30)
            purity_minus.setToolTip(f"减少{channel_names[col]}纯度（主元素-0.01，辅元素各+0.005）")
            purity_plus = QPushButton("+")
            purity_plus.setFixedWidth(30)
            purity_plus.setToolTip(f"增加{channel_names[col]}纯度（主元素+0.01，辅元素各-0.005）")
            purity_layout.addWidget(purity_minus)
            purity_layout.addWidget(purity_plus)
            purity_layout.setContentsMargins(0, 0, 0, 0)
            helper_grid.addLayout(purity_layout, 1, col * 2 + 1)

            # 色相标签
            hue_label = QLabel("色相:")
            hue_label.setFixedWidth(35)  # 限制宽度
            hue_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)  # 右对齐
            helper_grid.addWidget(hue_label, 2, col * 2)
            self.matrix_helper_widgets.append(hue_label)

            # 色相按钮
            hue_layout = QHBoxLayout()
            hue_layout.setSpacing(2)
            hue_left = QPushButton("<")
            hue_left.setFixedWidth(30)
            hue_left.setToolTip(f"{channel_names[col]}色相调整（上辅元素+0.01，下辅元素-0.01）")
            hue_right = QPushButton(">")
            hue_right.setFixedWidth(30)
            hue_right.setToolTip(f"{channel_names[col]}色相调整（下辅元素+0.01，上辅元素-0.01）")
            hue_layout.addWidget(hue_left)
            hue_layout.addWidget(hue_right)
            hue_layout.setContentsMargins(0, 0, 0, 0)
            helper_grid.addLayout(hue_layout, 2, col * 2 + 1)

            # 保存按钮引用
            self.matrix_helper_buttons.extend([purity_minus, purity_plus, hue_left, hue_right])
            self.matrix_helper_widgets.extend([purity_minus, purity_plus, hue_left, hue_right])

            # 连接信号
            # 单击触发
            purity_minus.clicked.connect(lambda checked=False, c=col: self._adjust_purity(c, increase=False))
            purity_plus.clicked.connect(lambda checked=False, c=col: self._adjust_purity(c, increase=True))
            hue_left.clicked.connect(lambda checked=False, c=col: self._adjust_hue(c, increase_down=False))
            hue_right.clicked.connect(lambda checked=False, c=col: self._adjust_hue(c, increase_down=True))

            # 长按重复触发
            purity_minus.pressed.connect(lambda c=col: self._start_button_repeat(lambda: self._adjust_purity(c, increase=False)))
            purity_minus.released.connect(self._stop_button_repeat)
            purity_plus.pressed.connect(lambda c=col: self._start_button_repeat(lambda: self._adjust_purity(c, increase=True)))
            purity_plus.released.connect(self._stop_button_repeat)
            hue_left.pressed.connect(lambda c=col: self._start_button_repeat(lambda: self._adjust_hue(c, increase_down=False)))
            hue_left.released.connect(self._stop_button_repeat)
            hue_right.pressed.connect(lambda c=col: self._start_button_repeat(lambda: self._adjust_hue(c, increase_down=True)))
            hue_right.released.connect(self._stop_button_repeat)

        matrix_layout.addLayout(helper_grid)
        layout.addWidget(matrix_group)

        # === 分层反差组（新增） ===
        channel_gamma_group = QGroupBox("分层反差 (Channel Gamma)")
        channel_gamma_layout = QVBoxLayout(channel_gamma_group)

        # R通道
        r_layout = QHBoxLayout()
        r_layout.addWidget(QLabel("R Gamma:"))
        self.channel_gamma_r_slider = QSlider(Qt.Horizontal)
        self.channel_gamma_r_slider.setRange(500, 2000)  # 0.5-2.0, *1000
        self.channel_gamma_r_slider.setValue(1000)       # 默认1.0
        r_layout.addWidget(self.channel_gamma_r_slider)

        self.channel_gamma_r_spinbox = QDoubleSpinBox()
        self.channel_gamma_r_spinbox.setRange(0.5, 2.0)
        self.channel_gamma_r_spinbox.setSingleStep(0.001)
        self.channel_gamma_r_spinbox.setDecimals(3)
        self.channel_gamma_r_spinbox.setValue(1.0)
        self.channel_gamma_r_spinbox.setFixedWidth(80)
        r_layout.addWidget(self.channel_gamma_r_spinbox)
        channel_gamma_layout.addLayout(r_layout)

        # B通道
        b_layout = QHBoxLayout()
        b_layout.addWidget(QLabel("B Gamma:"))
        self.channel_gamma_b_slider = QSlider(Qt.Horizontal)
        self.channel_gamma_b_slider.setRange(500, 2000)  # 0.5-2.0, *1000
        self.channel_gamma_b_slider.setValue(1000)       # 默认1.0
        b_layout.addWidget(self.channel_gamma_b_slider)

        self.channel_gamma_b_spinbox = QDoubleSpinBox()
        self.channel_gamma_b_spinbox.setRange(0.5, 2.0)
        self.channel_gamma_b_spinbox.setSingleStep(0.001)
        self.channel_gamma_b_spinbox.setDecimals(3)
        self.channel_gamma_b_spinbox.setValue(1.0)
        self.channel_gamma_b_spinbox.setFixedWidth(80)
        b_layout.addWidget(self.channel_gamma_b_spinbox)
        channel_gamma_layout.addLayout(b_layout)

        # 添加工具提示
        channel_gamma_group.setToolTip(
            "分层反差 - 模拟扫描仪的非线性通道响应\n"
            "调整R/B通道的密度缩放，G通道固定为1.0\n"
            "仅在启用密度矩阵时生效"
        )

        layout.addWidget(channel_gamma_group)
        layout.addStretch()
        return widget

    def _create_rgb_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        rgb_group = QGroupBox("RGB曝光调整")
        rgb_layout = QGridLayout(rgb_group)
        self.red_gain_slider = PrecisionSlider(Qt.Orientation.Horizontal)
        self.red_gain_spinbox = QDoubleSpinBox()
        self.green_gain_slider = PrecisionSlider(Qt.Orientation.Horizontal)
        self.green_gain_spinbox = QDoubleSpinBox()
        self.blue_gain_slider = PrecisionSlider(Qt.Orientation.Horizontal)
        self.blue_gain_spinbox = QDoubleSpinBox()
        self._setup_slider_spinbox(self.red_gain_slider, self.red_gain_spinbox, -3000, 3000, -3.0, 3.0, 0.005)
        self._setup_slider_spinbox(self.green_gain_slider, self.green_gain_spinbox, -3000, 3000, -3.0, 3.0, 0.005)
        self._setup_slider_spinbox(self.blue_gain_slider, self.blue_gain_spinbox, -3000, 3000, -3.0, 3.0, 0.005)
        rgb_layout.addWidget(QLabel("R:"), 0, 0); rgb_layout.addWidget(self.red_gain_slider, 0, 1); rgb_layout.addWidget(self.red_gain_spinbox, 0, 2)
        rgb_layout.addWidget(QLabel("G:"), 1, 0); rgb_layout.addWidget(self.green_gain_slider, 1, 1); rgb_layout.addWidget(self.green_gain_spinbox, 1, 2)
        rgb_layout.addWidget(QLabel("B:"), 2, 0); rgb_layout.addWidget(self.blue_gain_slider, 2, 1); rgb_layout.addWidget(self.blue_gain_spinbox, 2, 2)
        self.auto_color_single_button = QPushButton("AI自动校色（单次）")
        self.auto_color_multi_button = QPushButton("AI自动校色（多次）")
        rgb_layout.addWidget(self.auto_color_single_button, 3, 1)
        rgb_layout.addWidget(self.auto_color_multi_button, 3, 2)

        # 中性色定义按钮组（取点图标按钮 + 应用文本按钮）
        neutral_button_layout = QHBoxLayout()
        neutral_button_layout.setSpacing(4)  # 按钮间距

        # 取点图标按钮
        self.pick_neutral_point_button = QPushButton()
        icon_path = Path(__file__).parent / "asset" / "dropper-icon.png"
        self.pick_neutral_point_button.setIcon(QIcon(str(icon_path)))
        self.pick_neutral_point_button.setIconSize(QSize(20, 20))  # 图标大小
        self.pick_neutral_point_button.setFixedSize(28, 28)  # 按钮固定大小（正方形）
        self.pick_neutral_point_button.setToolTip("取点：在预览图上选择中性色点")

        # 应用中性色按钮
        self.apply_neutral_color_button = QPushButton("应用中性色")
        self.apply_neutral_color_button.setEnabled(False)  # 初始状态禁用

        neutral_button_layout.addWidget(self.pick_neutral_point_button, 0)  # 图标按钮不拉伸
        neutral_button_layout.addWidget(self.apply_neutral_color_button, 1)  # 文本按钮拉伸填充
        rgb_layout.addLayout(neutral_button_layout, 4, 1, 1, 1)  # 占据第1列

        self.neutral_white_point_spinbox = QSpinBox()
        self.neutral_white_point_spinbox.setMinimum(2000)
        self.neutral_white_point_spinbox.setMaximum(7000)
        self.neutral_white_point_spinbox.setSingleStep(100)
        self.neutral_white_point_spinbox.setValue(5500)
        self.neutral_white_point_spinbox.setSuffix(" K")
        self.neutral_white_point_spinbox.setToolTip("中性色的色温 (Kelvin)")
        rgb_layout.addWidget(self.neutral_white_point_spinbox, 4, 2)
        layout.addWidget(rgb_group)
        layout.addStretch()
        return widget

    def _create_curve_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.curve_editor = CurveEditorWidget()
        layout.addWidget(self.curve_editor)
        
        # 屏幕反光补偿控件
        glare_group = QGroupBox("屏幕反光补偿")
        glare_layout = QGridLayout(glare_group)
        
        self.glare_compensation_slider = PrecisionSlider(Qt.Orientation.Horizontal)
        self.glare_compensation_spinbox = QDoubleSpinBox()
        self._setup_slider_spinbox(self.glare_compensation_slider, self.glare_compensation_spinbox, 0, 5000, 0.0, 5.0, 0.005)
        self.glare_compensation_spinbox.setSuffix("%")
        
        glare_layout.addWidget(QLabel("补偿强度:"), 0, 0)
        glare_layout.addWidget(self.glare_compensation_slider, 0, 1)
        glare_layout.addWidget(self.glare_compensation_spinbox, 0, 2)
        
        layout.addWidget(glare_group)
        return widget

    def _create_debug_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Proxy尺寸设置组
        proxy_group = QGroupBox("Proxy设置")
        proxy_layout = QHBoxLayout(proxy_group)
        proxy_layout.addWidget(QLabel("Proxy长边尺寸:"))
        self.proxy_size_spinbox = QSpinBox()
        self.proxy_size_spinbox.setMinimum(500)
        self.proxy_size_spinbox.setMaximum(8000)
        self.proxy_size_spinbox.setValue(2000)
        self.proxy_size_spinbox.setSuffix(" px")
        self.proxy_size_spinbox.valueChanged.connect(self.proxy_size_changed.emit)
        proxy_layout.addWidget(self.proxy_size_spinbox)
        proxy_layout.addStretch()
        layout.addWidget(proxy_group)
        
        pipeline_group = QGroupBox("管线控制")
        pipeline_layout = QVBoxLayout(pipeline_group)
        self.enable_density_inversion_checkbox = QCheckBox("启用密度反相")
        self.enable_density_matrix_checkbox = QCheckBox("启用密度矩阵")
        self.enable_rgb_gains_checkbox = QCheckBox("启用RGB增益")
        self.enable_density_curve_checkbox = QCheckBox("启用密度曲线")
        pipeline_layout.addWidget(self.enable_density_inversion_checkbox)
        pipeline_layout.addWidget(self.enable_density_matrix_checkbox)
        pipeline_layout.addWidget(self.enable_rgb_gains_checkbox)
        pipeline_layout.addWidget(self.enable_density_curve_checkbox)
        layout.addWidget(pipeline_group)

        # 预览设置组
        preview_settings_group = QGroupBox("预览设置")
        preview_settings_layout = QVBoxLayout(preview_settings_group)
        self.monochrome_preview_checkbox = QCheckBox("黑白预览")
        self.monochrome_preview_checkbox.setToolTip("仅影响预览显示，不改变处理管线")
        preview_settings_layout.addWidget(self.monochrome_preview_checkbox)
        layout.addWidget(preview_settings_group)

        # LUT导出组
        lut_group = QGroupBox("LUT导出")
        lut_layout = QVBoxLayout(lut_group)
        
        # 输入设备转换LUT (3D)
        input_lut_layout = QHBoxLayout()
        input_lut_layout.addWidget(QLabel("输入设备转换LUT (3D):"))
        input_lut_layout.addStretch()
        self.input_lut_size_combo = QComboBox()
        self.input_lut_size_combo.addItems(["16", "32", "64", "128"])
        self.input_lut_size_combo.setCurrentText("64")
        input_lut_layout.addWidget(self.input_lut_size_combo)
        self.export_input_lut_btn = QPushButton("导出")
        self.export_input_lut_btn.clicked.connect(self._on_export_input_lut)
        input_lut_layout.addWidget(self.export_input_lut_btn)
        lut_layout.addLayout(input_lut_layout)
        
        # 反相校色LUT (3D, 不含密度曲线)
        color_lut_layout = QHBoxLayout()
        color_lut_layout.addWidget(QLabel("反相校色LUT (3D):"))
        color_lut_layout.addStretch()
        self.color_lut_size_combo = QComboBox()
        self.color_lut_size_combo.addItems(["16", "32", "64", "128"])
        self.color_lut_size_combo.setCurrentText("64")
        color_lut_layout.addWidget(self.color_lut_size_combo)
        self.export_color_lut_btn = QPushButton("导出")
        self.export_color_lut_btn.clicked.connect(self._on_export_color_lut)
        color_lut_layout.addWidget(self.export_color_lut_btn)
        lut_layout.addLayout(color_lut_layout)
        
        # 密度曲线LUT (1D)
        curve_lut_layout = QHBoxLayout()
        curve_lut_layout.addWidget(QLabel("密度曲线LUT (1D):"))
        curve_lut_layout.addStretch()
        self.curve_lut_size_combo = QComboBox()
        self.curve_lut_size_combo.addItems(["2048", "4096", "8192", "16384", "32768", "65536"])
        self.curve_lut_size_combo.setCurrentText("4096")
        curve_lut_layout.addWidget(self.curve_lut_size_combo)
        self.export_curve_lut_btn = QPushButton("导出")
        self.export_curve_lut_btn.clicked.connect(self._on_export_curve_lut)
        curve_lut_layout.addWidget(self.export_curve_lut_btn)
        lut_layout.addLayout(curve_lut_layout)
        
        layout.addWidget(lut_group)
        layout.addStretch()
        return widget
    
    def _setup_slider_spinbox(self, slider, spinbox, s_min, s_max, sp_min, sp_max, sp_step):
        slider.setRange(s_min, s_max)
        spinbox.setRange(sp_min, sp_max)
        spinbox.setSingleStep(sp_step)
        spinbox.setDecimals(3)
        # 设置slider步长：singleStep对应0.05，pageStep对应0.1
        slider.setSingleStep(50)   # 0.05 * 1000 = 50
        slider.setPageStep(100)    # 0.1 * 1000 = 100

    def _connect_signals(self):
        self.film_type_combo.currentTextChanged.connect(self._on_film_type_changed)
        self.input_colorspace_combo.currentTextChanged.connect(self._on_input_colorspace_changed)
        # IDT Gamma联动
        self.idt_gamma_slider.valueChanged.connect(self._on_idt_gamma_slider_changed)
        self.idt_gamma_spinbox.valueChanged.connect(self._on_idt_gamma_spinbox_changed)
        
        self.density_gamma_slider.valueChanged.connect(self._on_gamma_slider_changed)
        self.density_gamma_spinbox.valueChanged.connect(self._on_gamma_spinbox_changed)
        self.density_dmax_slider.valueChanged.connect(self._on_dmax_slider_changed)
        self.density_dmax_spinbox.valueChanged.connect(self._on_dmax_spinbox_changed)
        
        self.red_gain_slider.valueChanged.connect(self._on_red_gain_slider_changed)
        self.red_gain_spinbox.valueChanged.connect(self._on_red_gain_spinbox_changed)
        self.green_gain_slider.valueChanged.connect(self._on_green_gain_slider_changed)
        self.green_gain_spinbox.valueChanged.connect(self._on_green_gain_spinbox_changed)
        self.blue_gain_slider.valueChanged.connect(self._on_blue_gain_slider_changed)
        self.blue_gain_spinbox.valueChanged.connect(self._on_blue_gain_spinbox_changed)
        
        # 屏幕反光补偿信号连接
        self.glare_compensation_slider.valueChanged.connect(self._on_glare_compensation_slider_changed)
        self.glare_compensation_spinbox.valueChanged.connect(self._on_glare_compensation_spinbox_changed)
        # 安装事件过滤器以检测交互开始/结束
        self.glare_compensation_slider.installEventFilter(self)
        self.glare_compensation_spinbox.installEventFilter(self)

        self.matrix_combo.currentIndexChanged.connect(self._on_matrix_combo_changed)
        self.save_matrix_button.clicked.connect(self._on_save_matrix_clicked)
        for i in range(3):
            for j in range(3):
                # 统一由专用槽处理：必要时自动勾选"启用密度矩阵"，并触发参数变更
                self.matrix_editor_widgets[i][j].valueChanged.connect(self._on_matrix_value_changed)

        # 分层反差信号连接（新增）
        self.channel_gamma_r_slider.valueChanged.connect(self._on_channel_gamma_r_slider_changed)
        self.channel_gamma_r_spinbox.valueChanged.connect(self._on_channel_gamma_r_spinbox_changed)
        self.channel_gamma_b_slider.valueChanged.connect(self._on_channel_gamma_b_slider_changed)
        self.channel_gamma_b_spinbox.valueChanged.connect(self._on_channel_gamma_b_spinbox_changed)

        # 曲线编辑器发出 (curve_name, points)，使用专用槽以丢弃参数并统一触发
        self.curve_editor.curve_changed.connect(self._on_curve_changed)
        
        for checkbox in [self.enable_density_inversion_checkbox, self.enable_density_matrix_checkbox,
                         self.enable_rgb_gains_checkbox, self.enable_density_curve_checkbox]:
            checkbox.toggled.connect(self._on_debug_step_changed)

        # 美学式调整数字Mask复选框信号连接
        self.aesthetic_matrix_adjustment_checkbox.toggled.connect(self._on_aesthetic_matrix_adjustment_toggled)

        # 黑白预览信号连接
        self.monochrome_preview_checkbox.toggled.connect(self._on_monochrome_preview_toggled)

        self.auto_color_single_button.clicked.connect(self.auto_color_requested.emit)
        self.auto_color_multi_button.clicked.connect(self.auto_color_iterative_requested.emit)
        self.pick_neutral_point_button.clicked.connect(self.pick_neutral_point_requested.emit)
        self.apply_neutral_color_button.clicked.connect(lambda: self.apply_neutral_color_requested.emit(self.neutral_white_point_spinbox.value()))
        self.neutral_white_point_spinbox.valueChanged.connect(self.neutral_white_point_changed.emit)

        # Spectral sharpening signals
        self.enable_scanner_spectral_checkbox.toggled.connect(self._on_scanner_spectral_toggled)
        # dragFinished(dict) → 触发参数变更，同时发出 primaries 改变
        self.ucs_widget.dragFinished.connect(lambda coords: self._on_ucs_drag_finished(coords))
        self.ucs_widget.resetPointRequested.connect(self._on_reset_point)
        self.cc_selector_checkbox.toggled.connect(self._on_cc_selector_toggled)
        self.cc_flip_h_button.clicked.connect(self._on_cc_flip_horizontal)
        self.cc_flip_v_button.clicked.connect(self._on_cc_flip_vertical)
        self.cc_rotate_l_button.clicked.connect(self._on_cc_rotate_left)
        self.cc_rotate_r_button.clicked.connect(self._on_cc_rotate_right)
        self.ccm_optimize_button.clicked.connect(self.ccm_optimize_requested.emit)
        self.save_input_colorspace_button.clicked.connect(self._on_save_input_colorspace_clicked)
        self.save_matrix_button_afterOpt.clicked.connect(self._on_save_matrix_clicked)
        self.save_colorchecker_colors_button.clicked.connect(self.save_colorchecker_colors_requested.emit)
        self.colorchecker_combo.currentTextChanged.connect(self._on_colorchecker_changed)

    def enable_apply_neutral_button(self, enabled: bool):
        """启用或禁用'应用中性色'按钮"""
        self.apply_neutral_color_button.setEnabled(enabled)

    def update_ui_from_params(self):
        self._is_updating_ui = True
        try:
            params = self.current_params
            
            # Sync film type dropdown
            self._sync_combo_box(self.film_type_combo, self.current_film_type)
            
            self._sync_combo_box(self.input_colorspace_combo, params.input_color_space_name)
            # 读取当前输入空间的gamma
            try:
                info = self.context.color_space_manager.get_color_space_info(params.input_color_space_name) or {}
                g = float(info.get('gamma', 1.0))
            except Exception:
                g = 1.0
            self.idt_gamma_slider.setValue(int(g * 1000))
            self.idt_gamma_spinbox.setValue(g)
            
            self.density_gamma_slider.setValue(int(params.density_gamma * 1000))
            self.density_gamma_spinbox.setValue(params.density_gamma)
            self.density_dmax_slider.setValue(int(params.density_dmax * 1000))
            self.density_dmax_spinbox.setValue(params.density_dmax)
            
            self.red_gain_slider.setValue(int(params.rgb_gains[0] * 1000))
            self.red_gain_spinbox.setValue(params.rgb_gains[0])
            self.green_gain_slider.setValue(int(params.rgb_gains[1] * 1000))
            self.green_gain_spinbox.setValue(params.rgb_gains[1])
            self.blue_gain_slider.setValue(int(params.rgb_gains[2] * 1000))
            self.blue_gain_spinbox.setValue(params.rgb_gains[2])
            
            # 屏幕反光补偿参数同步 (0.0-0.05 -> 0-500)
            self.glare_compensation_slider.setValue(int(params.screen_glare_compensation * 100000.0))
            self.glare_compensation_spinbox.setValue(params.screen_glare_compensation * 100.0)

            # 分层反差参数同步（新增）
            self.channel_gamma_r_slider.setValue(int(params.channel_gamma_r * 1000))
            self.channel_gamma_r_spinbox.setValue(params.channel_gamma_r)
            self.channel_gamma_b_slider.setValue(int(params.channel_gamma_b * 1000))
            self.channel_gamma_b_spinbox.setValue(params.channel_gamma_b)

            matrix = params.density_matrix if params.density_matrix is not None else np.eye(3)
            print(f"Debug: update_ui_from_params - matrix_name={params.density_matrix_name}, matrix shape={matrix.shape}")
            print(f"Debug: matrix values: {matrix.tolist()}")
            for i in range(3):
                for j in range(3):
                    value = float(matrix[i,j])
                    if not np.isfinite(value):
                        print(f"Warning: invalid matrix value at [{i},{j}]: {value}, using 0.0")
                        value = 0.0
                    self.matrix_editor_widgets[i][j].setValue(value)
            self._sync_combo_box(self.matrix_combo, params.density_matrix_name)
            print(f"Debug: matrix UI update completed for {params.density_matrix_name}")

            curves = {'RGB': params.curve_points, 'R': params.curve_points_r, 'G': params.curve_points_g, 'B': params.curve_points_b}
            # 避免在拖动过程中反复重置内部曲线与选择状态：当曲线内容未变化时跳过写回
            try:
                current_curves = self.curve_editor.get_all_curves()
                if not self._curves_equal(current_curves, curves):
                    self.curve_editor.set_all_curves(curves)
            except Exception:
                self.curve_editor.set_all_curves(curves)
            self._sync_combo_box(self.curve_editor.curve_combo, params.density_curve_name)
            
            self.enable_density_inversion_checkbox.setChecked(params.enable_density_inversion)
            self.enable_density_matrix_checkbox.setChecked(params.enable_density_matrix)
            self.enable_rgb_gains_checkbox.setChecked(params.enable_rgb_gains)
            self.enable_density_curve_checkbox.setChecked(params.enable_density_curve)
            
            # 根据checkbox状态动态控制控件的enabled状态
            self._update_controls_enabled_state()
            
            # 更新UCS Diagram以反映当前色彩空间的基色
            try:
                print(f"Debug: UCS更新 - 色彩空间: {params.input_color_space_name}")
                cs_info = self.context.color_space_manager.get_color_space_info(params.input_color_space_name) or {}
                print(f"Debug: UCS更新 - cs_info keys: {list(cs_info.keys())}")
                
                if 'primaries' in cs_info:
                    primaries = cs_info['primaries']
                    print(f"Debug: UCS更新 - primaries: {primaries}")
                    print(f"Debug: UCS更新 - primaries type: {type(primaries)}")
                    
                    # 转换xy坐标到uv坐标
                    coords_uv = {}
                    for i, key in enumerate(['R', 'G', 'B']):
                        if i < len(primaries):
                            x, y = primaries[i]
                            u, v = xy_to_uv(x, y)
                            coords_uv[key] = (u, v)
                            print(f"Debug: UCS更新 - {key}: xy=({x:.4f}, {y:.4f}) -> uv=({u:.4f}, {v:.4f})")
                    
                    # 更新UCS Diagram
                    if len(coords_uv) == 3:
                        print(f"Debug: UCS更新 - 更新坐标: {coords_uv}")
                        self.ucs_widget.set_uv_coordinates(coords_uv)
                        print("Debug: UCS更新 - 更新完成")
                    else:
                        print(f"Debug: UCS更新 - 坐标不完整: {len(coords_uv)}/3")
                else:
                    print("Debug: UCS更新 - 没有primaries数据")
            except Exception as e:
                print(f"更新UCS控件失败: {e}")
                import traceback
                traceback.print_exc()
        finally:
            self._is_updating_ui = False

    def _update_controls_enabled_state(self):
        """根据checkbox状态动态更新控件的enabled状态"""
        # RGB增益控件：只受"启用RGB增益"checkbox控制
        rgb_gains_checked = self.enable_rgb_gains_checkbox.isChecked()
        self.red_gain_slider.setEnabled(rgb_gains_checked)
        self.red_gain_spinbox.setEnabled(rgb_gains_checked)
        self.green_gain_slider.setEnabled(rgb_gains_checked)
        self.green_gain_spinbox.setEnabled(rgb_gains_checked)
        self.blue_gain_slider.setEnabled(rgb_gains_checked)
        self.blue_gain_spinbox.setEnabled(rgb_gains_checked)
        self.auto_color_single_button.setEnabled(rgb_gains_checked)
        self.auto_color_multi_button.setEnabled(rgb_gains_checked)
        self.pick_neutral_point_button.setEnabled(rgb_gains_checked)
        # apply_neutral_color_button的状态由是否选择了点来控制，不受checkbox影响
        
        # 密度曲线相关控件：只受"启用密度曲线"checkbox控制
        density_curve_checked = self.enable_density_curve_checkbox.isChecked()
        # 注意：曲线编辑器和屏幕反光补偿都属于密度曲线功能
        if hasattr(self, 'curve_editor'):
            self.curve_editor.setEnabled(density_curve_checked)
        self.glare_compensation_slider.setEnabled(density_curve_checked)
        self.glare_compensation_spinbox.setEnabled(density_curve_checked)

        # 分层反差控件：跟随密度矩阵开关（新增）
        matrix_enabled = self.enable_density_matrix_checkbox.isChecked()
        self.channel_gamma_r_slider.setEnabled(matrix_enabled)
        self.channel_gamma_r_spinbox.setEnabled(matrix_enabled)
        self.channel_gamma_b_slider.setEnabled(matrix_enabled)
        self.channel_gamma_b_spinbox.setEnabled(matrix_enabled)

    def _curves_equal(self, a: dict, b: dict) -> bool:
        try:
            keys = ('RGB','R','G','B')
            for k in keys:
                pa = a.get(k, []) if isinstance(a, dict) else []
                pb = b.get(k, []) if isinstance(b, dict) else []
                if len(pa) != len(pb):
                    return False
                for (xa, ya), (xb, yb) in zip(pa, pb):
                    if abs(float(xa) - float(xb)) > 1e-6 or abs(float(ya) - float(yb)) > 1e-6:
                        return False
            return True
        except Exception:
            return False

    def _sync_combo_box(self, combo: QComboBox, name: str):
        # 首先尝试精确匹配
        for i in range(combo.count()):
            if combo.itemData(i) == name:
                combo.setCurrentIndex(i)
                return
        
        # 如果是曲线下拉框，使用其专门的同步机制
        if combo == self.curve_editor.curve_combo:
            # 对于曲线，让curve_editor自己处理同步，避免重复添加*项目
            self._sync_curve_combo_by_name(name)
            return
        
        # 对于其他下拉框，尝试按显示名匹配
        for i in range(combo.count()):
            if combo.itemText(i).strip('*') == name:
                combo.setCurrentIndex(i)
                return
        
        # 只有当确实需要时才添加*项目（例如非曲线的情况）
        display_name = f"*{name}"
        combo.insertItem(0, display_name, name)
        combo.setCurrentIndex(0)
    
    def _sync_curve_combo_by_name(self, name: str):
        """专门用于同步曲线下拉框的方法"""
        combo = self.curve_editor.curve_combo
        
        # 首先尝试精确匹配
        for i in range(combo.count()):
            if combo.itemData(i) == name:
                combo.blockSignals(True)
                combo.setCurrentIndex(i)
                combo.blockSignals(False)
                return
        
        # 如果name以*开头，说明是修改状态曲线
        if name.startswith('*'):
            original_name = name[1:]  # 去掉*前缀
            
            # 检查是否已经有这个修改状态项目（精确匹配）
            for i in range(combo.count()):
                if combo.itemData(i) == name:
                    combo.blockSignals(True)
                    combo.setCurrentIndex(i)
                    combo.blockSignals(False)
                    # 确保curve_editor状态正确
                    self.curve_editor.current_curve_name = name
                    self.curve_editor.is_modified = True
                    return
            
            # 如果修改状态项目不存在，说明这可能是预设文件中保存的修改状态
            # 需要恢复这个状态，但这应该由curve_editor的预设加载逻辑处理
            # 参数同步不应该主动创建修改状态，只应该恢复已存在的状态
            
            # 尝试切换到对应的原始曲线，如果存在的话
            for i in range(combo.count()):
                if combo.itemData(i) in self.curve_editor.preset_curves:
                    curve_data = self.curve_editor.preset_curves[combo.itemData(i)]
                    if curve_data["name"] == original_name:
                        combo.blockSignals(True)
                        combo.setCurrentIndex(i)
                        combo.blockSignals(False)
                        # 设置状态但不创建修改选项（这应该由用户编辑触发）
                        self.curve_editor.current_curve_name = curve_data["name"]
                        self.curve_editor.original_curve_name = curve_data["name"]
                        self.curve_editor.original_curve_key = combo.itemData(i)
                        self.curve_editor.is_modified = False
                        return
        else:
            # 普通曲线名称，尝试按显示名匹配
            for i in range(combo.count()):
                if combo.itemData(i) in self.curve_editor.preset_curves:
                    curve_data = self.curve_editor.preset_curves[combo.itemData(i)]
                    if curve_data["name"] == name:
                        combo.blockSignals(True)
                        combo.setCurrentIndex(i)
                        combo.blockSignals(False)
                        
                        # 更新curve_editor的状态
                        self.curve_editor.current_curve_name = name
                        self.curve_editor.original_curve_name = name
                        self.curve_editor.original_curve_key = combo.itemData(i)
                        self.curve_editor.is_modified = False
                        return
    
    def _cleanup_stale_modified_items(self):
        """清理不属于当前状态的修改状态项目，并修复被污染的原始预设项目"""
        combo = self.curve_editor.curve_combo
        current_curve_name = self.current_params.density_curve_name
        
        # 收集需要处理的项目
        items_to_remove = []
        items_to_fix = []
        
        for i in range(combo.count()):
            item_data = combo.itemData(i)
            item_text = combo.itemText(i)
            
            # 检查是否为修改状态项目（显示文本以*开头）
            if item_text.startswith('*'):
                # 如果data不以*开头，说明这是被污染的原始预设项目
                if isinstance(item_data, str) and not item_data.startswith('*'):
                    # 修复被污染的原始预设项目：恢复原始显示名称
                    original_name = item_text[1:]  # 去掉*前缀
                    items_to_fix.append((i, original_name))
                else:
                    # 这是真正的修改状态项目
                    # 如果不是当前正在使用的修改状态，则标记为移除
                    if item_data != current_curve_name and item_text != current_curve_name:
                        items_to_remove.append(i)
        
        # 修复被污染的原始预设项目
        combo.blockSignals(True)
        for i, original_name in items_to_fix:
            combo.setItemText(i, original_name)
        
        # 移除不相关的修改状态项目
        for i in reversed(items_to_remove):
            combo.removeItem(i)
        combo.blockSignals(False)
        
        # 如果当前曲线的修改状态被移除了，重置curve_editor的修改状态
        if not current_curve_name.startswith('*'):
            self.curve_editor.is_modified = False
            self.curve_editor.modified_item_index = -1

    def get_current_params(self) -> ColorGradingParams:
        params = ColorGradingParams()
        params.input_color_space_name = self.input_colorspace_combo.currentData() or self.input_colorspace_combo.currentText().strip('*')
        params.density_gamma = self.density_gamma_spinbox.value()
        params.density_dmax = self.density_dmax_spinbox.value()
        params.rgb_gains = (self.red_gain_spinbox.value(), self.green_gain_spinbox.value(), self.blue_gain_spinbox.value())
        
        matrix = np.zeros((3, 3), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                matrix[i, j] = self.matrix_editor_widgets[i][j].value()
        params.density_matrix = matrix
        params.density_matrix_name = self.matrix_combo.currentData() or self.matrix_combo.currentText().strip('*')
        
        all_curves = self.curve_editor.get_all_curves()
        params.curve_points = all_curves.get('RGB', [])
        params.curve_points_r = all_curves.get('R', [])
        params.curve_points_g = all_curves.get('G', [])
        params.curve_points_b = all_curves.get('B', [])
        params.density_curve_name = self.curve_editor.curve_combo.currentData() or self.curve_editor.curve_combo.currentText().strip('*')
        
        # 屏幕反光补偿参数 (UI显示0-20% -> 实际值0.0-0.2)
        params.screen_glare_compensation = self.glare_compensation_spinbox.value() / 100.0

        # 分层反差参数（新增）
        params.channel_gamma_r = self.channel_gamma_r_spinbox.value()
        params.channel_gamma_b = self.channel_gamma_b_spinbox.value()

        params.enable_density_inversion = self.enable_density_inversion_checkbox.isChecked()
        params.enable_density_matrix = self.enable_density_matrix_checkbox.isChecked()
        params.enable_rgb_gains = self.enable_rgb_gains_checkbox.isChecked()
        params.enable_density_curve = self.enable_density_curve_checkbox.isChecked()
        return params
    
    def get_current_film_type(self) -> str:
        """获取当前选择的胶片类型"""
        return self.current_film_type
    
    def set_film_type(self, film_type: str):
        """设置胶片类型（用于从预设加载时）"""
        self.current_film_type = film_type
        self._is_updating_ui = True
        self._sync_combo_box(self.film_type_combo, film_type)
        self._is_updating_ui = False
        # 应用对应的UI状态
        self._apply_ui_state_for_film_type(film_type)

    def get_monochrome_preview_enabled(self) -> bool:
        """获取黑白预览状态"""
        return self.monochrome_preview_checkbox.isChecked()

    def set_monochrome_preview_enabled(self, enabled: bool):
        """设置黑白预览状态（用于未来的自动联动）"""
        self._is_updating_ui = True
        self.monochrome_preview_checkbox.setChecked(enabled)
        self._is_updating_ui = False

    def _apply_ui_state_for_film_type(self, film_type: str):
        """根据胶片类型应用UI状态配置"""
        ui_config = self.context.film_type_controller.get_ui_state_config(film_type)
        
        # 应用密度反相控件状态
        self.enable_density_inversion_checkbox.setEnabled(ui_config.density_inversion_enabled)
        self.enable_density_inversion_checkbox.setVisible(ui_config.density_inversion_visible)
        
        # 应用密度矩阵控件状态
        self._set_density_matrix_ui_state(ui_config.density_matrix_enabled, ui_config.density_matrix_visible)
        
        # 应用RGB增益控件状态
        self._set_rgb_gains_ui_state(ui_config.rgb_gains_enabled, ui_config.rgb_gains_visible)
        
        # 应用密度曲线控件状态
        self._set_density_curve_ui_state(ui_config.density_curve_enabled, ui_config.density_curve_visible)
        
        # 应用色彩空间控件状态
        self._set_color_space_ui_state(ui_config.color_space_enabled, ui_config.color_space_visible)
        
        # 设置工具提示
        if ui_config.disabled_tooltip:
            self._set_disabled_tooltips(ui_config.disabled_tooltip)
    
    def _set_density_matrix_ui_state(self, enabled: bool, visible: bool):
        """设置密度矩阵控件组状态"""
        # 控制矩阵编辑器
        for i in range(3):
            for j in range(3):
                self.matrix_editor_widgets[i][j].setEnabled(enabled)

        # 控制矩阵下拉菜单
        self.matrix_combo.setEnabled(enabled)

        # 控制启用复选框
        self.enable_density_matrix_checkbox.setEnabled(enabled)
        self.enable_density_matrix_checkbox.setVisible(visible)

        # 控制矩阵辅助调整按钮
        for button in self.matrix_helper_buttons:
            button.setEnabled(enabled)
    
    def _set_rgb_gains_ui_state(self, enabled: bool, visible: bool):
        """设置RGB增益控件组状态"""
        # 控制滑块和输入框
        self.red_gain_slider.setEnabled(enabled)
        self.red_gain_spinbox.setEnabled(enabled)
        self.green_gain_slider.setEnabled(enabled)
        self.green_gain_spinbox.setEnabled(enabled)
        self.blue_gain_slider.setEnabled(enabled)
        self.blue_gain_spinbox.setEnabled(enabled)

        # 控制自动校色按钮
        self.auto_color_single_button.setEnabled(enabled)
        self.auto_color_multi_button.setEnabled(enabled)
        self.pick_neutral_point_button.setEnabled(enabled)
        # apply_neutral_color_button的状态由是否选择了点来控制

        # 控制启用复选框
        self.enable_rgb_gains_checkbox.setEnabled(enabled)
        self.enable_rgb_gains_checkbox.setVisible(visible)
    
    def _set_color_space_ui_state(self, enabled: bool, visible: bool):
        """设置色彩空间控件组状态"""
        # 控制输入色彩空间下拉菜单
        self.input_colorspace_combo.setEnabled(enabled)
        
        # 控制IDT Gamma控件
        self.idt_gamma_slider.setEnabled(enabled)
        self.idt_gamma_spinbox.setEnabled(enabled)
        
        # 控制光谱锐化（硬件校正）相关控件（如果启用）
        if hasattr(self, 'enable_scanner_spectral_checkbox'):
            self.enable_scanner_spectral_checkbox.setEnabled(enabled)
    
    def _set_density_curve_ui_state(self, enabled: bool, visible: bool):
        """设置密度曲线控件组状态（包括屏幕反光补偿）"""
        # 控制启用复选框
        self.enable_density_curve_checkbox.setEnabled(enabled)
        self.enable_density_curve_checkbox.setVisible(visible)
        
        # 控制屏幕反光补偿控件
        self.glare_compensation_slider.setEnabled(enabled)
        self.glare_compensation_spinbox.setEnabled(enabled)
    
    def _mark_as_modified(self, combo: QComboBox):
        """给combo box的当前选中项添加星号标记"""
        if self._is_updating_ui:
            return
            
        current_text = combo.currentText()
        current_data = combo.currentData()
        
        # 如果已经有星号，不需要再次添加
        if current_text.startswith('*'):
            return
            
        # 添加星号标记
        modified_text = f"*{current_text}"
        
        # 更新当前项的显示文本，但保持data不变
        current_index = combo.currentIndex()
        if current_index >= 0:
            combo.setItemText(current_index, modified_text)
    
    def _is_matrix_modified(self) -> bool:
        """检查当前matrix是否与原始预设不同"""
        try:
            current_matrix_name = self.matrix_combo.currentData() or self.matrix_combo.currentText().strip('*')
            
            # 如果是自定义，则认为是修改的
            if current_matrix_name in ("custom", "自定义"):
                return True
                
            # 获取原始matrix数据
            original_matrix = self.context.the_enlarger.pipeline_processor.get_density_matrix_array(current_matrix_name)
            if original_matrix is None:
                return True
                
            # 获取当前UI中的matrix数据
            current_matrix = np.zeros((3, 3), dtype=np.float32)
            for i in range(3):
                for j in range(3):
                    current_matrix[i, j] = self.matrix_editor_widgets[i][j].value()
            
            # 比较矩阵是否相同（使用较小的容差）
            return not np.allclose(original_matrix, current_matrix, atol=1e-6)
        except Exception:
            # 出现异常，认为是修改的（安全的fallback）
            return True
    
    def _is_curve_modified(self) -> bool:
        """检查当前curves是否与原始预设不同"""
        try:
            current_curve_name = self.curve_editor.curve_combo.currentData() or self.curve_editor.curve_combo.currentText().strip('*')
            current_display_name = self.curve_editor.curve_combo.currentText()
            
            # 如果是custom，或者显示名称以*开头，则认为是修改的
            if current_curve_name == "custom" or current_display_name.startswith('*'):
                return True
            
            # 获取原始curve数据
            if hasattr(self.curve_editor, 'preset_curves') and current_curve_name in self.curve_editor.preset_curves:
                original_curve_data = self.curve_editor.preset_curves[current_curve_name]
                
                # 获取当前curves
                current_curves = self.curve_editor.get_all_curves()
                
                # 比较RGB曲线（主要的）
                current_rgb_points = current_curves.get('RGB', [])
                original_rgb_points = original_curve_data.get('points', {}).get('rgb', [])
                
                # 简单比较点的数量和值
                if len(current_rgb_points) != len(original_rgb_points):
                    return True
                    
                for i, (current_point, original_point) in enumerate(zip(current_rgb_points, original_rgb_points)):
                    if not (abs(current_point[0] - original_point[0]) < 1e-6 and 
                           abs(current_point[1] - original_point[1]) < 1e-6):
                        return True
            else:
                # 找不到原始数据，认为是修改的
                return True
        except Exception:
            # 出现异常，认为是修改的（安全的fallback）
            return True
            
        return False
    
    def _is_idt_modified(self) -> bool:
        """检查当前IDT参数是否与原始预设不同"""
        try:
            current_space_name = self.input_colorspace_combo.currentData() or self.input_colorspace_combo.currentText().strip('*')
            current_gamma = self.idt_gamma_spinbox.value()
            
            # 获取原始色彩空间信息
            original_info = self.context.color_space_manager.get_color_space_info(current_space_name) or {}
            original_gamma = float(original_info.get('gamma', 1.0))
            
            # 比较gamma值是否相同
            return not abs(current_gamma - original_gamma) < 1e-6
        except Exception:
            # 出现异常，认为是修改的（安全的fallback）
            return True

    def _update_colorchecker_for_film_type(self, film_type: str):
        """根据胶片类型更新colorchecker参考文件选择"""
        from divere.core.data_types import FILM_TYPE_COLORCHECKER_MAPPING
        
        # 获取对应的参考文件
        reference_file = FILM_TYPE_COLORCHECKER_MAPPING.get(
            film_type, 
            "original_color_cc24data.json"
        )
        
        # 更新选中的参考文件
        self.selected_colorchecker_file = reference_file
        
        # 如果有colorchecker下拉菜单，同步选择
        if hasattr(self, 'colorchecker_combo'):
            try:
                # 查找对应的菜单项并选择
                for i in range(self.colorchecker_combo.count()):
                    if self.colorchecker_combo.itemData(i) == reference_file:
                        self._is_updating_ui = True
                        self.colorchecker_combo.setCurrentIndex(i)
                        self._is_updating_ui = False
                        break
            except Exception as e:
                print(f"警告：无法同步colorchecker选择: {e}")
    
    def _set_disabled_tooltips(self, tooltip: str):
        """为禁用的控件设置工具提示"""
        # 为主要的禁用控件设置工具提示
        disabled_widgets = []
        
        # 根据当前状态确定哪些控件被禁用
        if not self.enable_density_inversion_checkbox.isEnabled():
            disabled_widgets.extend([
                self.density_gamma_slider, self.density_gamma_spinbox,
                self.density_dmax_slider, self.density_dmax_spinbox
            ])
        
        if not self.matrix_combo.isEnabled():
            disabled_widgets.append(self.matrix_combo)
            disabled_widgets.extend([widget for row in self.matrix_editor_widgets for widget in row])
        
        if not self.input_colorspace_combo.isEnabled():
            disabled_widgets.extend([
                self.input_colorspace_combo, self.idt_gamma_slider, self.idt_gamma_spinbox
            ])
        
        # 应用工具提示
        for widget in disabled_widgets:
            widget.setToolTip(tooltip)

    # --- Internal sync slots ---
    def _on_gamma_slider_changed(self, value: int):
        if self._is_updating_ui: return
        self.density_gamma_spinbox.blockSignals(True)
        self.density_gamma_spinbox.setValue(value / 1000.0)
        self.density_gamma_spinbox.blockSignals(False)
        self.parameter_changed.emit()

    def _on_gamma_spinbox_changed(self, value: float):
        if self._is_updating_ui: return
        self.density_gamma_slider.blockSignals(True)
        self.density_gamma_slider.setValue(int(value * 1000))
        self.density_gamma_slider.blockSignals(False)
        self.parameter_changed.emit()

    def _on_dmax_slider_changed(self, value: int):
        if self._is_updating_ui: return
        self.density_dmax_spinbox.blockSignals(True)
        self.density_dmax_spinbox.setValue(value / 1000.0)
        self.density_dmax_spinbox.blockSignals(False)
        self.parameter_changed.emit()

    def _on_dmax_spinbox_changed(self, value: float):
        if self._is_updating_ui: return
        self.density_dmax_slider.blockSignals(True)
        self.density_dmax_slider.setValue(int(value * 1000))
        self.density_dmax_slider.blockSignals(False)
        self.parameter_changed.emit()

    def _on_red_gain_slider_changed(self, value: int):
        if self._is_updating_ui: return
        self.red_gain_spinbox.blockSignals(True)
        self.red_gain_spinbox.setValue(value / 1000.0)
        self.red_gain_spinbox.blockSignals(False)
        self.parameter_changed.emit()

    def _on_red_gain_spinbox_changed(self, value: float):
        if self._is_updating_ui: return
        self.red_gain_slider.blockSignals(True)
        self.red_gain_slider.setValue(int(value * 1000))
        self.red_gain_slider.blockSignals(False)
        self.parameter_changed.emit()

    def _on_green_gain_slider_changed(self, value: int):
        if self._is_updating_ui: return
        self.green_gain_spinbox.blockSignals(True)
        self.green_gain_spinbox.setValue(value / 1000.0)
        self.green_gain_spinbox.blockSignals(False)
        self.parameter_changed.emit()

    def _on_green_gain_spinbox_changed(self, value: float):
        if self._is_updating_ui: return
        self.green_gain_slider.blockSignals(True)
        self.green_gain_slider.setValue(int(value * 1000))
        self.green_gain_slider.blockSignals(False)
        self.parameter_changed.emit()

    def _on_blue_gain_slider_changed(self, value: int):
        if self._is_updating_ui: return
        self.blue_gain_spinbox.blockSignals(True)
        self.blue_gain_spinbox.setValue(value / 1000.0)
        self.blue_gain_spinbox.blockSignals(False)
        self.parameter_changed.emit()

    def _on_blue_gain_spinbox_changed(self, value: float):
        if self._is_updating_ui: return
        self.blue_gain_slider.blockSignals(True)
        self.blue_gain_slider.setValue(int(value * 1000))
        self.blue_gain_slider.blockSignals(False)
        self.parameter_changed.emit()

    def _on_glare_compensation_slider_changed(self, value: int):
        if self._is_updating_ui: return
        self.glare_compensation_spinbox.blockSignals(True)
        self.glare_compensation_spinbox.setValue(value / 1000.0)  # Slider 0-5000 -> SpinBox 0-5.0
        self.glare_compensation_spinbox.blockSignals(False)
        self.parameter_changed.emit()
        # 发送实时更新信号（用于cut-off显示）
        compensation_value = value / 100000.0  # 转换为0.0-0.05范围
        self.glare_compensation_realtime_update.emit(compensation_value)

    def _on_glare_compensation_spinbox_changed(self, value: float):
        if self._is_updating_ui: return
        self.glare_compensation_slider.blockSignals(True)
        self.glare_compensation_slider.setValue(int(value * 1000.0))  # SpinBox 0-5.0 -> Slider 0-5000
        self.glare_compensation_slider.blockSignals(False)
        self.parameter_changed.emit()
        # 发送实时更新信号（用于cut-off显示）
        compensation_value = value / 100.0  # 转换为0.0-0.05范围
        self.glare_compensation_realtime_update.emit(compensation_value)

    # --- Action slots ---
    def _on_film_type_changed(self, display_name: str):
        """当胶片类型改变时"""
        if self._is_updating_ui:
            return
        
        # Get the actual film type value from the combo box data
        film_type_value = self.film_type_combo.currentData()
        if film_type_value:
            self.current_film_type = film_type_value
            
            # 应用UI状态配置
            self._apply_ui_state_for_film_type(film_type_value)
            
            # 更新colorchecker参考文件选择（如果存在相关UI组件）
            self._update_colorchecker_for_film_type(film_type_value)
            
            # 发出信号
            self.film_type_changed.emit(film_type_value)
            self.parameter_changed.emit()
    
    def _on_input_colorspace_changed(self, space_name: str):
        """当输入色彩空间改变时，更新UCS Diagram和IDT Gamma"""
        try:
            # 移除星号标记
            clean_name = space_name.strip('*')
            
            # 更新IDT Gamma
            cs_info = self.context.color_space_manager.get_color_space_info(clean_name) or {}
            gamma = float(cs_info.get("gamma", 1.0))
            
            # 更新UI（避免触发信号循环）
            self._is_updating_ui = True
            self.idt_gamma_slider.setValue(int(gamma * 1000))
            self.idt_gamma_spinbox.setValue(gamma)
            self._is_updating_ui = False
            
            # 更新UCS Diagram以反映新的色彩空间基色
            if 'primaries' in cs_info:
                primaries = cs_info['primaries']
                # 转换xy坐标到uv坐标
                coords_uv = {}
                for i, key in enumerate(['R', 'G', 'B']):
                    if i < len(primaries):
                        x, y = primaries[i]
                        u, v = xy_to_uv(x, y)
                        coords_uv[key] = (u, v)
                
                # 更新UCS Diagram
                if len(coords_uv) == 3:
                    self.ucs_widget.set_uv_coordinates(coords_uv)
                    
        except Exception as e:
            print(f"更新色彩空间失败: {e}")
            pass
        
        # 发出色彩空间特定变更信号，触发专用处理路径
        clean_name = space_name.strip('*')
        self.input_colorspace_changed.emit(clean_name)

    def _on_matrix_combo_changed(self, index: int):
        if self._is_updating_ui: return
        matrix_id = self.matrix_combo.itemData(index)
        print(f"Debug: matrix combo changed to index={index}, matrix_id={matrix_id}")
        if matrix_id and matrix_id != "custom":
            matrix = self.context.the_enlarger.pipeline_processor.get_density_matrix_array(matrix_id)
            print(f"Debug: loaded matrix: {matrix is not None}, shape={matrix.shape if matrix is not None else 'None'}")
            if matrix is not None:
                self._is_updating_ui = True
                try:
                    print(f"Debug: updating matrix spinboxes with values: {matrix.tolist()}")
                    for i in range(3):
                        for j in range(3):
                            value = float(matrix[i, j])
                            if not np.isfinite(value):
                                print(f"Warning: invalid matrix value at [{i},{j}]: {value}, using 0.0")
                                value = 0.0
                            self.matrix_editor_widgets[i][j].setValue(value)
                    # 选择了预设矩阵，默认自动启用
                    self.enable_density_matrix_checkbox.setChecked(True)
                    print("Debug: matrix spinboxes updated successfully")
                finally:
                    self._is_updating_ui = False
            else:
                print(f"Warning: failed to load matrix for matrix_id={matrix_id}")
        self.parameter_changed.emit()

    # --- IDT Gamma slots ---
    def _on_idt_gamma_slider_changed(self, value: int):
        if self._is_updating_ui: return
        self.idt_gamma_spinbox.blockSignals(True)
        self.idt_gamma_spinbox.setValue(value / 1000.0)
        self.idt_gamma_spinbox.blockSignals(False)
        self._apply_idt_gamma_to_colorspace()

    def _on_idt_gamma_spinbox_changed(self, value: float):
        if self._is_updating_ui: return
        self.idt_gamma_slider.blockSignals(True)
        self.idt_gamma_slider.setValue(int(value * 1000))
        self.idt_gamma_slider.blockSignals(False)
        self._apply_idt_gamma_to_colorspace()

    def _apply_idt_gamma_to_colorspace(self):
        """将UI中的IDT Gamma写入当前输入色彩空间的内存定义，并重建代理。"""
        try:
            g = float(self.idt_gamma_spinbox.value())
            g = max(0.5, min(2.8, g))
            space_name = self.input_colorspace_combo.currentData() or self.input_colorspace_combo.currentText().strip('*')
            self.context.color_space_manager.update_color_space_gamma(space_name, g)
            
            # 检查是否修改，并添加星号标记
            if self._is_idt_modified():
                self._mark_as_modified(self.input_colorspace_combo)
            
            # 触发Context按当前色彩空间重建proxy（内部会skip逆伽马，并应用前置幂次）
            self.context._prepare_proxy(); self.context._trigger_preview_update()
        except Exception:
            pass
    
    def _on_scanner_spectral_toggled(self, checked: bool):
        self.ucs_widget.setVisible(checked)
        self.cc_selector_checkbox.setVisible(checked)
        self.cc_flip_h_button.setVisible(checked)
        self.cc_flip_v_button.setVisible(checked)
        self.cc_rotate_l_button.setVisible(checked)
        self.cc_rotate_r_button.setVisible(checked)
        self.colorchecker_combo.setVisible(checked)
        self.save_colorchecker_colors_button.setVisible(checked)
        self.optimize_idt_checkbox.setVisible(checked)
        self.optimize_density_matrix_checkbox.setVisible(checked)
        self.ccm_optimize_button.setVisible(checked)
        self.save_input_colorspace_button.setVisible(checked)
        self.save_matrix_button_afterOpt.setVisible(checked)
        if checked:
            self._on_reset_point('R')
            self._on_reset_point('G')
            self._on_reset_point('B')
            
    def _apply_ucs_coords(self):
        if self._is_updating_ui: return
        self.parameter_changed.emit()

    def _on_ucs_drag_finished(self, coords_uv: dict):
        """UCS拖动结束：发出无参参数变更，并将 primaries (xy) 以字典形式广播给上层。"""
        try:
            if not isinstance(coords_uv, dict):
                coords_uv = self.ucs_widget.get_uv_coordinates()
            # 转换到 xy
            primaries_xy = {}
            for key in ("R", "G", "B"):
                if key in coords_uv:
                    u, v = coords_uv[key]
                    x, y = uv_to_xy(u, v)
                    primaries_xy[key] = (float(x), float(y))
            
            # 检查是否修改IDT，并添加星号标记（primaries修改也算IDT修改）
            self._mark_as_modified(self.input_colorspace_combo)
            
            # 发出两类信号：UI参数变更 + 基色更新
            self.parameter_changed.emit()
            if len(primaries_xy) == 3:
                self.custom_primaries_changed.emit(primaries_xy)
        except Exception:
            # 即使转换失败，也保持无参的参数变更触发
            self.parameter_changed.emit()

    def _on_reset_point(self, key: str):
        space = self.input_colorspace_combo.currentText().strip('*')
        info = self.context.color_space_manager.get_color_space_info(space)
        if info and 'primaries' in info:
            prim = np.array(info['primaries'], dtype=float)
            idx = {'R': 0, 'G': 1, 'B': 2}.get(key, 0)
            u, v = xy_to_uv(prim[idx, 0], prim[idx, 1])
            self.ucs_widget.set_uv_coordinates({key: (u, v)})

    def _on_cc_selector_toggled(self, checked: bool):
        self.ccm_optimize_button.setEnabled(checked)
        self.save_colorchecker_colors_button.setEnabled(checked)
        self.toggle_color_checker_requested.emit(checked)

    def _on_save_input_colorspace_clicked(self):
        coords_uv = self.ucs_widget.get_uv_coordinates()
        if not all(k in coords_uv for k in ("R", "G", "B")):
            QMessageBox.warning(self, "警告", "没有可保存的基色坐标。")
            return
            
        primaries = {k: uv_to_xy(*v) for k, v in coords_uv.items()}
        self.save_custom_colorspace_requested.emit(primaries)

    def _on_cc_flip_horizontal(self):
        """水平翻转色卡选择器"""
        self.cc_flip_horizontal_requested.emit()
    
    def _on_cc_flip_vertical(self):
        """竖直翻转色卡选择器"""
        self.cc_flip_vertical_requested.emit()
    
    def _on_cc_rotate_left(self):
        """左旋转色卡选择器"""
        self.cc_rotate_left_requested.emit()
    
    def _on_cc_rotate_right(self):
        """右旋转色卡选择器"""
        self.cc_rotate_right_requested.emit()
    
    def _on_colorchecker_changed(self, display_name: str):
        """色卡类型选择改变时"""
        if self._is_updating_ui:
            return
        filename = self.colorchecker_combo.currentData()
        if filename:
            # 验证工作空间兼容性
            if not self._validate_colorchecker_workspace_compatibility(filename):
                return  # 验证失败，已回滚选择
            
            self.selected_colorchecker_file = filename
            self.colorchecker_changed.emit(filename) # 发出色卡类型变化信号

    def _validate_colorchecker_workspace_compatibility(self, filename: str) -> bool:
        """
        验证ColorChecker文件与当前工作空间的兼容性
        如果验证失败，显示对话框并回滚dropdown选择
        
        Returns:
            True: 验证通过
            False: 验证失败，已回滚选择
        """
        try:
            # 获取真正的工作色彩空间（不是输入色彩空间）
            working_colorspace = self.context.color_space_manager.get_current_working_space()
            
            
            # 验证兼容性
            is_compatible, error_message = validate_colorchecker_workspace_compatibility(
                filename, working_colorspace
            )
            
            if not is_compatible:
                # 显示错误对话框
                QMessageBox.warning(
                    self, 
                    "工作空间不兼容", 
                    error_message or "ColorChecker文件与当前工作空间不兼容"
                )
                
                # 回滚dropdown选择到上一个选项
                self._rollback_colorchecker_selection()
                return False
            
            return True
            
        except Exception as e:
            # 处理验证过程中的异常
            QMessageBox.warning(
                self,
                "验证失败",
                f"无法验证ColorChecker兼容性: {e}"
            )
            self._rollback_colorchecker_selection()
            return False

    def _rollback_colorchecker_selection(self):
        """回滚ColorChecker下拉菜单选择到上一个有效选项"""
        try:
            # 查找当前记录的有效文件名
            previous_filename = self.selected_colorchecker_file
            
            # 在下拉菜单中找到对应的项目并选择
            for i in range(self.colorchecker_combo.count()):
                if self.colorchecker_combo.itemData(i) == previous_filename:
                    self._is_updating_ui = True
                    self.colorchecker_combo.setCurrentIndex(i)
                    self._is_updating_ui = False
                    break
            else:
                # 如果找不到上一个选项，回滚到默认选项
                default_index = self.colorchecker_combo.findData("original_color_cc24data.json")
                if default_index >= 0:
                    self._is_updating_ui = True
                    self.colorchecker_combo.setCurrentIndex(default_index)
                    self._is_updating_ui = False
                    self.selected_colorchecker_file = "original_color_cc24data.json"
                    
        except Exception as e:
            print(f"回滚ColorChecker选择失败: {e}")
    
    def get_spectral_sharpening_config(self):
        """获取当前的光谱锐化（硬件校正）配置（优先使用用户选择的参考文件）"""
        from divere.core.data_types import SpectralSharpeningConfig, FILM_TYPE_COLORCHECKER_MAPPING
        
        # 优先使用用户在dropdown中的选择
        reference_file = self.selected_colorchecker_file
        
        # 如果没有用户选择，回退到胶片类型映射
        if not reference_file:
            film_type = self.get_current_film_type()
            reference_file = FILM_TYPE_COLORCHECKER_MAPPING.get(
                film_type, 
                "original_color_cc24data.json"  # 默认值
            )
        
        return SpectralSharpeningConfig(
            optimize_idt_transformation=self.optimize_idt_checkbox.isChecked(),
            optimize_density_matrix=self.optimize_density_matrix_checkbox.isChecked(),
            reference_file=reference_file
        )

    def _on_save_matrix_clicked(self):
        """保存当前密度矩阵"""
        # 获取当前矩阵数据
        matrix = np.zeros((3, 3), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                matrix[i, j] = self.matrix_editor_widgets[i][j].value()
        
        # 获取当前矩阵名称
        current_name = self.matrix_combo.currentText().strip('*')
        if current_name == "自定义":
            current_name = ""
        
        # 发射保存信号
        self.save_density_matrix_requested.emit(matrix, current_name)

    def _refresh_matrix_combo(self):
        """刷新矩阵下拉列表（保存后调用）"""
        # 保存当前选择
        current_data = self.matrix_combo.currentData()
        
        # 清空并重新填充
        self.matrix_combo.clear()
        self.matrix_combo.addItem("自定义", "custom")
        available = self.context.the_enlarger.pipeline_processor.get_available_matrices()
        for matrix_id in available:
            data = self.context.the_enlarger.pipeline_processor.get_matrix_data(matrix_id)
            if data: 
                self.matrix_combo.addItem(data.get("name", matrix_id), matrix_id)
        
        # 恢复选择
        index = self.matrix_combo.findData(current_data)
        if index >= 0:
            self.matrix_combo.setCurrentIndex(index)

    def _refresh_colorspace_combo(self):
        """刷新色彩空间下拉列表（保存后调用）"""
        # 保存当前选择
        current_data = self.input_colorspace_combo.currentData()
        
        # 清空并重新填充，只显示IDT色彩空间
        self.input_colorspace_combo.clear()
        spaces = self.context.color_space_manager.get_idt_color_spaces()
        for space in spaces:
            self.input_colorspace_combo.addItem(space, space)
        
        # 恢复选择
        index = self.input_colorspace_combo.findData(current_data)
        if index >= 0:
            self.input_colorspace_combo.setCurrentIndex(index)

    def refresh_all_combos(self):
        """刷新所有下拉列表"""
        try:
            self._refresh_colorspace_combo()
        except AttributeError:
            pass
        try:
            self._refresh_matrix_combo()
        except AttributeError:
            pass

    def _on_curve_changed(self, curve_name, points):
        if self._is_updating_ui: return
        
        # 修改状态管理完全由curve_editor_widget负责
        # parameter_panel只负责响应变化，不主动管理修改状态
        # 移除_mark_as_modified调用以避免污染原始预设项目
        
        self.parameter_changed.emit()

    def _on_matrix_value_changed(self, *args):
        """矩阵单元格改动：必要时自动勾选启用，并触发参数变更。"""
        if self._is_updating_ui:
            return
        if not self.enable_density_matrix_checkbox.isChecked():
            # 用户开始编辑矩阵时，自动启用矩阵
            self.enable_density_matrix_checkbox.setChecked(True)

        # 检查是否修改，并添加星号标记
        if self._is_matrix_modified():
            self._mark_as_modified(self.matrix_combo)

        self.parameter_changed.emit()

    def _adjust_purity(self, col_index: int, increase: bool = True):
        """调整矩阵列的纯度

        纯度调整分为两步：
        1. 主列操作：主元素增加步长，两个辅元素各减少步长的一半
        2. 行归一化：调整所有元素使每行的和保持不变

        Args:
            col_index: 列索引 (0=红通道, 1=绿通道, 2=蓝通道)
            increase: True=增加纯度（主元素增加），False=减少纯度
        """
        if self._is_updating_ui:
            return

        # 保存调整前每行的和
        row_sums_before = [
            sum(self.matrix_editor_widgets[row][col].value() for col in range(3))
            for row in range(3)
        ]

        step = 0.02 if increase else -0.02
        half_step = step / 2.0
        main_idx = col_index  # 主元素索引（对角线元素）

        # 读取当前列的所有值
        col_values = [self.matrix_editor_widgets[i][col_index].value() for i in range(3)]

        # 步骤1: 调整主列
        col_values[main_idx] += step

        # 调整辅元素（两个辅元素各减少步长的一半）
        for i in range(3):
            if i != main_idx:
                col_values[i] -= half_step

        # 更新UI
        self._is_updating_ui = True
        try:
            for i in range(3):
                self.matrix_editor_widgets[i][col_index].setValue(col_values[i])
        finally:
            self._is_updating_ui = False

        # 步骤2: 行归一化，保持每行的和不变
        self._normalize_rows_to_preserve_sum(row_sums_before)

        # 自动启用矩阵并标记为修改
        if not self.enable_density_matrix_checkbox.isChecked():
            self.enable_density_matrix_checkbox.setChecked(True)
        if self._is_matrix_modified():
            self._mark_as_modified(self.matrix_combo)

        # 触发参数更新
        self.parameter_changed.emit()

    def _adjust_hue(self, col_index: int, increase_down: bool = True):
        """调整矩阵列的色相

        色相调整分为两步：
        1. 主列操作：改变两个辅元素的相对比例，主元素保持不变
        2. 行归一化：调整所有元素使每行的和保持不变

        Args:
            col_index: 列索引 (0=红通道, 1=绿通道, 2=蓝通道)
            increase_down: True=增下减上（>按钮），False=增上减下（<按钮）
        """
        if self._is_updating_ui:
            return

        # 保存调整前每行的和
        row_sums_before = [
            sum(self.matrix_editor_widgets[row][col].value() for col in range(3))
            for row in range(3)
        ]

        step = 0.01
        main_idx = col_index  # 主元素索引（对角线元素）

        # 计算辅元素索引（按行索引排序）
        aux_indices = [i for i in range(3) if i != main_idx]
        upper_idx = aux_indices[0]  # 行索引较小的（上辅助）
        lower_idx = aux_indices[1]  # 行索引较大的（下辅助）

        # 读取当前值
        upper_val = self.matrix_editor_widgets[upper_idx][col_index].value()
        lower_val = self.matrix_editor_widgets[lower_idx][col_index].value()

        # 步骤1: 调整辅元素
        # 红通道(0)和蓝通道(2)的左右按钮行为与绿通道(1)相反
        if col_index in [0, 2]:
            increase_down = not increase_down

        if increase_down:
            # > 按钮：增下减上
            lower_val += step
            upper_val -= step
        else:
            # < 按钮：增上减下
            upper_val += step
            lower_val -= step

        # 更新UI
        self._is_updating_ui = True
        try:
            self.matrix_editor_widgets[upper_idx][col_index].setValue(upper_val)
            self.matrix_editor_widgets[lower_idx][col_index].setValue(lower_val)
        finally:
            self._is_updating_ui = False

        # 步骤2: 行归一化，保持每行的和不变
        self._normalize_rows_to_preserve_sum(row_sums_before)

        # 自动启用矩阵并标记为修改
        if not self.enable_density_matrix_checkbox.isChecked():
            self.enable_density_matrix_checkbox.setChecked(True)
        if self._is_matrix_modified():
            self._mark_as_modified(self.matrix_combo)

        # 触发参数更新
        self.parameter_changed.emit()

    def _normalize_rows_to_preserve_sum(self, row_sums_before):
        """行归一化：使每行的和恢复到调整前的值

        在矩阵列调整后，对每一行应用缩放系数，使该行的和保持不变。

        Args:
            row_sums_before: 调整前每行的和（长度为3的列表）
        """
        self._is_updating_ui = True
        try:
            for row_i in range(3):
                # 计算当前行的和
                row_sum_after = sum(
                    self.matrix_editor_widgets[row_i][col].value() for col in range(3)
                )

                # 计算缩放系数并应用
                if abs(row_sum_after) > 1e-12:  # 避免除零
                    scale = row_sums_before[row_i] / row_sum_after

                    # 该行所有元素乘以scale
                    for col_j in range(3):
                        old_val = self.matrix_editor_widgets[row_i][col_j].value()
                        new_val = old_val * scale
                        self.matrix_editor_widgets[row_i][col_j].setValue(new_val)
        finally:
            self._is_updating_ui = False

    # === 分层反差槽函数（新增） ===
    def _on_channel_gamma_r_slider_changed(self, value: int):
        """R通道分层反差滑条变化"""
        if self._is_updating_ui:
            return
        gamma_value = value / 1000.0
        self._is_updating_ui = True
        self.channel_gamma_r_spinbox.setValue(gamma_value)
        self._is_updating_ui = False
        self.parameter_changed.emit()

    def _on_channel_gamma_r_spinbox_changed(self, value: float):
        """R通道分层反差数值框变化"""
        if self._is_updating_ui:
            return
        slider_value = int(value * 1000)
        self._is_updating_ui = True
        self.channel_gamma_r_slider.setValue(slider_value)
        self._is_updating_ui = False
        self.parameter_changed.emit()

    def _on_channel_gamma_b_slider_changed(self, value: int):
        """B通道分层反差滑条变化"""
        if self._is_updating_ui:
            return
        gamma_value = value / 1000.0
        self._is_updating_ui = True
        self.channel_gamma_b_spinbox.setValue(gamma_value)
        self._is_updating_ui = False
        self.parameter_changed.emit()

    def _on_channel_gamma_b_spinbox_changed(self, value: float):
        """B通道分层反差数值框变化"""
        if self._is_updating_ui:
            return
        slider_value = int(value * 1000)
        self._is_updating_ui = True
        self.channel_gamma_b_slider.setValue(slider_value)
        self._is_updating_ui = False
        self.parameter_changed.emit()

    def _on_debug_step_changed(self):
        if self._is_updating_ui: return
        # 当checkbox状态改变时，更新控件的enabled状态
        self._update_controls_enabled_state()
        self.parameter_changed.emit()

    def _on_monochrome_preview_toggled(self, checked: bool):
        """黑白预览开关切换时"""
        if self._is_updating_ui:
            return
        # 通过信号通知MainWindow
        self.monochrome_preview_changed.emit(checked)

    def _on_aesthetic_matrix_adjustment_toggled(self, checked: bool):
        """美学式调整数字Mask复选框切换时"""
        # 切换所有辅助调整控件的可见性
        for widget in self.matrix_helper_widgets:
            widget.setVisible(checked)

    def _on_auto_color_single_clicked(self):
        if self._is_updating_ui: return
        self.auto_color_requested.emit()

    def _on_auto_color_correct_clicked(self):
        if self._is_updating_ui: return
        self.auto_color_iterative_requested.emit()
    
    def _on_export_input_lut(self):
        """导出输入设备转换LUT"""
        from PySide6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出输入设备转换LUT", "", "LUT文件 (*.cube)"
        )
        if file_path:
            if not file_path.endswith('.cube'):
                file_path += '.cube'
            size = int(self.input_lut_size_combo.currentText())
            self.lut_export_requested.emit("input_transform", file_path, size)
    
    def _on_export_color_lut(self):
        """导出反相校色LUT（不含密度曲线）"""
        from PySide6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出反相校色LUT", "", "LUT文件 (*.cube)"
        )
        if file_path:
            if not file_path.endswith('.cube'):
                file_path += '.cube'
            size = int(self.color_lut_size_combo.currentText())
            self.lut_export_requested.emit("color_correction", file_path, size)
    
    def _on_export_curve_lut(self):
        """导出密度曲线LUT"""
        from PySide6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出密度曲线LUT", "", "LUT文件 (*.cube)"
        )
        if file_path:
            if not file_path.endswith('.cube'):
                file_path += '.cube'
            size = int(self.curve_lut_size_combo.currentText())
            self.lut_export_requested.emit("density_curve", file_path, size)
    
    def eventFilter(self, obj, event):
        """事件过滤器：检测屏幕反光补偿控件的交互"""
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QMouseEvent
        
        if obj in (self.glare_compensation_slider, self.glare_compensation_spinbox):
            if event.type() == QEvent.Type.MouseButtonPress:
                # 获取当前补偿值
                current_value = self.glare_compensation_slider.value() / 100000.0
                self.glare_compensation_interaction_started.emit(current_value)
            elif event.type() == QEvent.Type.MouseButtonRelease:
                self.glare_compensation_interaction_ended.emit()
            elif event.type() == QEvent.Type.FocusIn:
                # spinbox获得焦点时也开始交互
                current_value = self.glare_compensation_slider.value() / 100000.0
                self.glare_compensation_interaction_started.emit(current_value)
            elif event.type() == QEvent.Type.FocusOut:
                # spinbox失去焦点时结束交互
                self.glare_compensation_interaction_ended.emit()

        return super().eventFilter(obj, event)

    # === 按钮长按重复触发辅助方法 ===
    def _start_button_repeat(self, callback):
        """启动按钮长按重复触发

        Args:
            callback: 要重复调用的函数
        """
        # 停止现有定时器
        self._stop_button_repeat()

        # 保存回调函数
        self._repeat_callback = callback

        # 创建定时器
        self._repeat_timer = QTimer(self)
        self._repeat_timer.timeout.connect(self._on_repeat_triggered)

        # 1秒后首次触发，然后每100ms重复触发
        self._repeat_timer.setSingleShot(True)
        self._repeat_timer.start(1000)

    def _stop_button_repeat(self):
        """停止按钮长按重复触发"""
        if self._repeat_timer is not None:
            self._repeat_timer.stop()
            self._repeat_timer.deleteLater()
            self._repeat_timer = None
        self._repeat_callback = None

    def _on_repeat_triggered(self):
        """定时器触发时的处理"""
        if self._repeat_callback is not None:
            # 执行回调
            self._repeat_callback()

            # 转换为重复定时器（100ms间隔）
            if self._repeat_timer is not None:
                self._repeat_timer.setSingleShot(False)
                self._repeat_timer.start(100)
