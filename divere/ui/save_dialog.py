"""
保存图像对话框
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGroupBox,
    QRadioButton, QComboBox, QCheckBox, QPushButton,
    QLabel, QGridLayout, QDialogButtonBox, QSplitter,
    QTreeWidget, QTreeWidgetItem, QProgressDialog, QSlider
)
from PySide6.QtCore import Qt, QThread, Signal
from pathlib import Path
from typing import Dict, List, Optional, Set

from divere.i18n import tr


class SaveImageDialog(QDialog):
    """保存图像对话框"""
    
    # JPEG质量等级映射（个位数1-10到实际质量值）
    QUALITY_MAPPING = {
        1: 60, 2: 65, 3: 70, 4: 75, 5: 80,
        6: 85, 7: 90, 8: 93, 9: 95, 10: 100
    }
    
    def __init__(self, parent=None, color_spaces=None, is_bw_mode=False, color_space_manager=None, app_context=None):
        super().__init__(parent)
        self.setWindowTitle(tr("save_dialog.title"))
        self.setModal(True)
        self.setMinimumWidth(800)  # 增加宽度以容纳左右布局
        self._save_mode = 'single'  # 'single' | 'all'
        self._is_bw_mode = is_bw_mode
        self._color_space_manager = color_space_manager
        self._app_context = app_context
        self._selected_files: Set[str] = set()  # 选中的文件名集合
        
        # 导入配置管理器
        from divere.utils.enhanced_config_manager import enhanced_config_manager
        self._config_manager = enhanced_config_manager
        
        # 可用的色彩空间
        if color_spaces is None and color_space_manager:
            # 使用过滤后的regular色彩空间（有ICC文件的）
            self.color_spaces = color_space_manager.get_regular_color_spaces_with_icc()
        else:
            self.color_spaces = color_spaces or ["sRGB", "AdobeRGB", "ProPhotoRGB"]
        
        # 创建UI
        self._create_ui()
        
        # 设置默认值
        self._set_defaults()
    
    def _quality_level_to_jpeg(self, level: int) -> int:
        """将质量等级(1-10)转换为JPEG质量值"""
        return self.QUALITY_MAPPING.get(level, 95)
    
    def _jpeg_to_quality_level(self, jpeg_quality: int) -> int:
        """将JPEG质量值转换为质量等级(1-10)"""
        # 找到最接近的质量等级
        best_level = 9  # 默认等级9(95质量)
        min_diff = float('inf')
        for level, quality in self.QUALITY_MAPPING.items():
            diff = abs(quality - jpeg_quality)
            if diff < min_diff:
                min_diff = diff
                best_level = level
        return best_level
    
    def _update_quality_label(self):
        """更新质量标签显示"""
        level = self.jpeg_quality_slider.value()
        jpeg_quality = self._quality_level_to_jpeg(level)
        self.jpeg_quality_label.setText(f"{level} ({jpeg_quality}%)")
        
    def _create_ui(self):
        """创建用户界面"""
        layout = QVBoxLayout(self)
        
        # 创建左右分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        
        # 左侧：现有的保存设置
        left_widget = self._create_left_panel()
        splitter.addWidget(left_widget)
        
        # 右侧：文件树和批量选择
        right_widget = self._create_right_panel()
        splitter.addWidget(right_widget)
        
        # 设置分割器比例 (左侧:右侧 = 1:1)
        splitter.setSizes([400, 400])
        
        # 按钮
        button_box = QDialogButtonBox()
        # 标准"保存单张"按钮
        ok_btn = button_box.addButton(QDialogButtonBox.StandardButton.Ok)
        ok_btn.setText(tr("save_dialog.button_save_single"))
        # 自定义"保存所有"按钮
        self.save_all_btn = QPushButton(tr("save_dialog.button_save_all"))
        button_box.addButton(self.save_all_btn, QDialogButtonBox.ButtonRole.AcceptRole)
        # 取消
        cancel_btn = button_box.addButton(QDialogButtonBox.StandardButton.Cancel)
        
        def _on_ok():
            self._save_mode = 'single'
            self.accept()
        def _on_save_all():
            self._save_mode = 'all'
            self.accept()
        def _on_cancel():
            self.reject()
        ok_btn.clicked.connect(_on_ok)
        self.save_all_btn.clicked.connect(_on_save_all)
        cancel_btn.clicked.connect(_on_cancel)
        
        layout.addWidget(button_box)
        
        # 连接信号
        self.tiff_16bit_radio.toggled.connect(self._on_format_changed)
        self.jpeg_8bit_radio.toggled.connect(self._on_format_changed)
        self.jpeg_quality_slider.valueChanged.connect(self._update_quality_label)
        
        # 初始化文件树
        self._load_file_tree()
        
    def _set_defaults(self):
        """设置默认值"""
        self.tiff_16bit_radio.setChecked(True)
        self._load_saved_color_space()
        
    def _on_format_changed(self):
        """格式选择改变时更新默认色彩空间和JPEG质量组显示"""
        self._load_saved_color_space()
        
        # 控制JPEG质量组的显示
        is_jpeg = self.jpeg_8bit_radio.isChecked()
        self.jpeg_quality_group.setVisible(is_jpeg)
        
    def _load_saved_color_space(self):
        """加载保存的色彩空间选择"""
        if self._is_bw_mode:
            # B&W mode: prioritize grayscale color spaces
            preferred = ["Gray_Gamma_2_2", "Gray Gamma 2.2", "Grayscale", "sRGB"]
            for name in preferred:
                if name in self.color_spaces:
                    self.colorspace_combo.setCurrentText(name)
                    break
        else:
            # 从配置读取保存的色彩空间
            if self.tiff_16bit_radio.isChecked():
                saved_color_space = self._config_manager.get_default_setting("output_color_space_16bit", "DisplayP3")
            else:
                saved_color_space = self._config_manager.get_default_setting("output_color_space_8bit", "sRGB")
            
            # 检查保存的色彩空间是否在可用列表中
            if saved_color_space in self.color_spaces:
                self.colorspace_combo.setCurrentText(saved_color_space)
            else:
                # 如果保存的色彩空间不可用，使用默认的fallback逻辑
                if self.tiff_16bit_radio.isChecked():
                    fallback = ["DisplayP3", "AdobeRGB", "sRGB"]
                else:
                    fallback = ["sRGB", "DisplayP3", "AdobeRGB"]
                
                for name in fallback:
                    if name in self.color_spaces:
                        self.colorspace_combo.setCurrentText(name)
                        break
        
        # 加载保存的JPEG质量设置
        # 先尝试加载新格式（质量等级）
        saved_quality_level = self._config_manager.get_default_setting("jpeg_quality_level", None)
        if saved_quality_level is not None:
            # 使用新格式的质量等级
            try:
                level = int(saved_quality_level)
                level = max(1, min(10, level))  # 确保在有效范围内
                self.jpeg_quality_slider.setValue(level)
            except ValueError:
                self.jpeg_quality_slider.setValue(9)  # 默认等级9
        else:
            # 向后兼容：尝试加载旧格式（实际质量值）并转换为等级
            saved_jpeg_quality = self._config_manager.get_default_setting("jpeg_quality", "95")
            try:
                jpeg_quality = int(saved_jpeg_quality)
                level = self._jpeg_to_quality_level(jpeg_quality)
                self.jpeg_quality_slider.setValue(level)
            except ValueError:
                self.jpeg_quality_slider.setValue(9)  # 默认等级9
        
        # 更新质量标签显示
        self._update_quality_label()
    
    def _create_left_panel(self):
        """创建左侧面板：现有的保存设置"""
        left_widget = QGroupBox(tr("save_dialog.groups.save_settings"))
        layout = QVBoxLayout(left_widget)
        
        # 文件格式选择
        format_group = QGroupBox(tr("save_dialog.groups.file_format"))
        format_layout = QVBoxLayout(format_group)

        self.tiff_16bit_radio = QRadioButton(tr("save_dialog.format_16bit_tiff"))
        self.jpeg_8bit_radio = QRadioButton(tr("save_dialog.format_8bit_jpeg"))
        
        format_layout.addWidget(self.tiff_16bit_radio)
        format_layout.addWidget(self.jpeg_8bit_radio)
        
        layout.addWidget(format_group)
        
        # 色彩空间选择
        colorspace_group = QGroupBox(tr("save_dialog.groups.color_management"))
        colorspace_layout = QGridLayout(colorspace_group)

        colorspace_layout.addWidget(QLabel(tr("save_dialog.output_color_space_label")), 0, 0)
        self.colorspace_combo = QComboBox()
        self.colorspace_combo.addItems(self.color_spaces)
        colorspace_layout.addWidget(self.colorspace_combo, 0, 1)
        
        layout.addWidget(colorspace_group)
        
        # JPEG质量设置
        self.jpeg_quality_group = QGroupBox(tr("save_dialog.groups.jpeg_quality"))
        jpeg_quality_layout = QGridLayout(self.jpeg_quality_group)

        jpeg_quality_layout.addWidget(QLabel(tr("save_dialog.quality_label")), 0, 0)
        self.jpeg_quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.jpeg_quality_slider.setRange(1, 10)
        self.jpeg_quality_slider.setValue(9)  # 默认等级9(对应95质量)
        self.jpeg_quality_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.jpeg_quality_slider.setTickInterval(1)
        jpeg_quality_layout.addWidget(self.jpeg_quality_slider, 0, 1)
        
        self.jpeg_quality_label = QLabel("9 (95%)")
        jpeg_quality_layout.addWidget(self.jpeg_quality_label, 0, 2)
        
        # 初始隐藏质量设置组
        self.jpeg_quality_group.setVisible(False)
        
        layout.addWidget(self.jpeg_quality_group)
        
        # 处理选项
        options_group = QGroupBox(tr("save_dialog.groups.processing_options"))
        options_layout = QVBoxLayout(options_group)

        self.include_curve_checkbox = QCheckBox(tr("save_dialog.include_density_curves"))
        self.include_curve_checkbox.setChecked(True)
        options_layout.addWidget(self.include_curve_checkbox)
        
        layout.addWidget(options_group)
        
        return left_widget
        
    def _create_right_panel(self):
        """创建右侧面板：文件树和批量选择"""
        right_widget = QGroupBox(tr("save_dialog.groups.batch_export"))
        layout = QVBoxLayout(right_widget)

        # 说明标签
        info_label = QLabel(tr("save_dialog.batch_files_label"))
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # 文件树
        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabels([tr("save_dialog.table_headers.file"), tr("save_dialog.table_headers.type")])
        self.file_tree.itemChanged.connect(self._on_tree_item_changed)
        layout.addWidget(self.file_tree)
        
        # 快速选择按钮
        button_layout = QHBoxLayout()

        select_all_btn = QPushButton(tr("save_dialog.button_select_all"))
        select_all_btn.clicked.connect(self._select_all)
        button_layout.addWidget(select_all_btn)

        select_none_btn = QPushButton(tr("save_dialog.button_deselect_all"))
        select_none_btn.clicked.connect(self._select_none)
        button_layout.addWidget(select_none_btn)

        select_default_btn = QPushButton(tr("save_dialog.button_default_selection"))
        select_default_btn.clicked.connect(self._select_default)
        button_layout.addWidget(select_default_btn)
        
        layout.addLayout(button_layout)
        
        return right_widget

    def _load_file_tree(self):
        """加载文件树"""
        if not self._app_context:
            return
            
        self.file_tree.clear()
        self._selected_files.clear()
        
        try:
            # 获取当前目录的预设管理器
            auto_preset_manager = self._app_context.auto_preset_manager
            if not auto_preset_manager:
                return
                
            # 获取当前图像的目录
            current_image = self._app_context.get_current_image()
            if not current_image:
                return
                
            current_dir = Path(current_image.file_path).parent
            
            # 设置预设管理器的活动目录
            auto_preset_manager.set_active_directory(str(current_dir))
            
            # 获取所有预设
            presets = auto_preset_manager.get_all_presets()
            bundles = auto_preset_manager.get_all_bundles()
            
            # 添加Single预设（平铺在根层）
            for filename, preset in presets.items():
                file_path = current_dir / filename
                if file_path.exists():
                    item = QTreeWidgetItem(self.file_tree, [filename, "Single"])
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(0, Qt.CheckState.Checked)  # 默认选中
                    item.setData(0, Qt.ItemDataRole.UserRole, {
                        'filename': filename,
                        'type': 'single'
                    })
                    self._selected_files.add(filename)
            
            # 添加ContactSheet预设和其crops（平铺在根层，crops作为子项）
            for filename, bundle in bundles.items():
                file_path = current_dir / filename  
                if file_path.exists():
                    # ContactSheet父项
                    cs_item = QTreeWidgetItem(self.file_tree, [filename, "ContactSheet"])
                    cs_item.setFlags(cs_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    cs_item.setCheckState(0, Qt.CheckState.Unchecked)  # 默认不选中
                    cs_item.setData(0, Qt.ItemDataRole.UserRole, {
                        'filename': filename,
                        'type': 'contactsheet'
                    })
                    
                    # 添加该bundle的crops作为子项
                    for crop_entry in bundle.crops:
                        crop_display_name = crop_entry.crop.name or crop_entry.crop.id
                        crop_item = QTreeWidgetItem(cs_item, [crop_display_name, "Crop"])
                        crop_item.setFlags(crop_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                        crop_item.setCheckState(0, Qt.CheckState.Checked)  # 默认选中
                        # 存储完整信息以便后续处理
                        crop_key = f"{filename}#{crop_entry.crop.id}"  # 使用#避免与文件名冲突
                        crop_item.setData(0, Qt.ItemDataRole.UserRole, {
                            'filename': filename,
                            'crop_id': crop_entry.crop.id,
                            'type': 'crop',
                            'key': crop_key
                        })
                        self._selected_files.add(crop_key)
            
            # 展开所有分组
            self.file_tree.expandAll()
            
            # 更新保存所有按钮状态
            self._update_save_all_button()
            
        except Exception as e:
            print(f"加载文件树失败: {e}")

    def _on_tree_item_changed(self, item, column):
        """树项目变化时的处理"""
        if column != 0:  # 只处理第一列的勾选变化
            return
            
        user_data = item.data(0, Qt.ItemDataRole.UserRole)
        if not user_data:
            return
            
        item_type = user_data.get('type')
        
        if item_type == 'single':
            # Single文件
            filename = user_data['filename']
            if item.checkState(0) == Qt.CheckState.Checked:
                self._selected_files.add(filename)
            else:
                self._selected_files.discard(filename)
                
        elif item_type == 'contactsheet':
            # ContactSheet文件
            filename = user_data['filename']
            if item.checkState(0) == Qt.CheckState.Checked:
                self._selected_files.add(filename)
            else:
                self._selected_files.discard(filename)
                
        elif item_type == 'crop':
            # Crop项
            crop_key = user_data['key']
            if item.checkState(0) == Qt.CheckState.Checked:
                self._selected_files.add(crop_key)
            else:
                self._selected_files.discard(crop_key)
                
        self._update_save_all_button()

    def _select_all(self):
        """全选"""
        self._selected_files.clear()
        
        def check_items(item):
            if item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                item.setCheckState(0, Qt.CheckState.Checked)
                user_data = item.data(0, Qt.ItemDataRole.UserRole)
                if user_data:
                    item_type = user_data.get('type')
                    if item_type == 'single':
                        self._selected_files.add(user_data['filename'])
                    elif item_type == 'contactsheet':
                        self._selected_files.add(user_data['filename'])
                    elif item_type == 'crop':
                        self._selected_files.add(user_data['key'])
            
            for i in range(item.childCount()):
                check_items(item.child(i))
                
        root = self.file_tree.invisibleRootItem()
        for i in range(root.childCount()):
            check_items(root.child(i))
            
        self._update_save_all_button()

    def _select_none(self):
        """全不选"""
        self._selected_files.clear()
        
        def uncheck_items(item):
            if item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                item.setCheckState(0, Qt.CheckState.Unchecked)
            
            for i in range(item.childCount()):
                uncheck_items(item.child(i))
                
        root = self.file_tree.invisibleRootItem()
        for i in range(root.childCount()):
            uncheck_items(root.child(i))
            
        self._update_save_all_button()

    def _select_default(self):
        """默认选择：所有Single和Crops，ContactSheet不选"""
        self._selected_files.clear()
        
        def default_select_items(item):
            if item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                user_data = item.data(0, Qt.ItemDataRole.UserRole)
                if user_data:
                    item_type = user_data.get('type')
                    if item_type in ['single', 'crop']:
                        item.setCheckState(0, Qt.CheckState.Checked)
                        if item_type == 'single':
                            self._selected_files.add(user_data['filename'])
                        elif item_type == 'crop':
                            self._selected_files.add(user_data['key'])
                    else:  # contactsheet
                        item.setCheckState(0, Qt.CheckState.Unchecked)
            
            for i in range(item.childCount()):
                default_select_items(item.child(i))
                
        root = self.file_tree.invisibleRootItem()
        for i in range(root.childCount()):
            default_select_items(root.child(i))
            
        self._update_save_all_button()

    def _update_save_all_button(self):
        """更新保存所有按钮的状态和文本"""
        count = len(self._selected_files)
        if count == 0:
            self.save_all_btn.setText(tr("save_dialog.button_save_all"))
            self.save_all_btn.setEnabled(False)
        else:
            self.save_all_btn.setText(f"{tr('save_dialog.button_save_all')} ({count})")
            self.save_all_btn.setEnabled(True)

    def get_settings(self):
        """获取保存设置"""
        current_color_space = self.colorspace_combo.currentText()
        
        # 保存当前选择的色彩空间到配置
        if not self._is_bw_mode:
            if self.tiff_16bit_radio.isChecked():
                self._config_manager.set_default_setting("output_color_space_16bit", current_color_space)
            else:
                self._config_manager.set_default_setting("output_color_space_8bit", current_color_space)
        
        # 保存JPEG质量设置到配置（保存等级值）
        self._config_manager.set_default_setting("jpeg_quality_level", str(self.jpeg_quality_slider.value()))
        
        # 获取实际的JPEG质量值
        jpeg_quality = self._quality_level_to_jpeg(self.jpeg_quality_slider.value()) if self.jpeg_8bit_radio.isChecked() else 95
        
        settings = {
            "format": "tiff" if self.tiff_16bit_radio.isChecked() else "jpeg",
            "bit_depth": 16 if self.tiff_16bit_radio.isChecked() else 8,
            "color_space": current_color_space,
            "include_curve": self.include_curve_checkbox.isChecked(),
            "save_mode": self._save_mode,
            "jpeg_quality": jpeg_quality
        }
        
        # 如果是批量保存模式，添加选中的文件列表
        if self._save_mode == 'all':
            settings["selected_files"] = list(self._selected_files)
            
        return settings