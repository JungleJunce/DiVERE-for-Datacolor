"""
精确通道分离IDT计算工具主窗口

提供图形化界面用于：
- 加载三张光谱分离的图片
- 选择目标工作空间
- 执行CMA-ES优化
- 显示计算结果
- 保存色彩空间配置文件
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QTextEdit, QGroupBox,
    QFileDialog, QMessageBox, QProgressBar, QLineEdit,
    QSplitter, QFrame, QScrollArea, QSpinBox, QDoubleSpinBox
)
from PySide6.QtCore import Qt, Signal, QThread, QTimer
from PySide6.QtGui import QFont, QPixmap, QIcon
from pathlib import Path
import json
import traceback
from typing import Optional, Dict, Any

from .idt_calculation_engine import IDTCalculationEngine
from .idt_optimizer import IDTOptimizer

# 尝试导入DiVERE的组件
try:
    from divere.core.color_space import ColorSpaceManager
    from divere.utils.path_manager import PathManager
    from divere.utils.enhanced_config_manager import enhanced_config_manager
    from divere.ui.cmaes_progress_dialog import CMAESProgressDialog
except ImportError as e:
    print(f"警告：无法导入DiVERE组件: {e}")
    ColorSpaceManager = None
    PathManager = None
    enhanced_config_manager = None
    CMAESProgressDialog = None


class OptimizationWorker(QThread):
    """优化工作线程"""
    
    progress_updated = Signal(str)  # 进度更新信号
    optimization_completed = Signal(dict)  # 优化完成信号
    optimization_failed = Signal(str)  # 优化失败信号
    
    def __init__(self, engine: IDTCalculationEngine, target_colorspace: str, 
                 color_space_manager=None):
        super().__init__()
        self.engine = engine
        self.target_colorspace = target_colorspace
        self.color_space_manager = color_space_manager
        self.optimizer = IDTOptimizer(status_callback=self._status_callback)
        
    def _status_callback(self, message: str):
        """状态回调函数"""
        self.progress_updated.emit(message)
    
    def run(self):
        """执行优化"""
        try:
            # 获取初始RGB矩阵
            initial_rgb = self.engine.get_initial_rgb_matrix()
            if initial_rgb is None:
                self.optimization_failed.emit("缺少RGB数据，请确保已加载三张图片")
                return
            
            # 执行优化
            result = self.optimizer.optimize(
                initial_rgb, max_iter=1000, tolerance=1e-8, 
                target_colorspace=self.target_colorspace
            )
            
            if result['success']:
                # 设置优化后的CCM矩阵
                self.engine.set_optimized_ccm(result['ccm_matrix'])
                
                # 计算最终的色彩空间信息
                colorspace_info = self.engine.calculate_final_colorspace(
                    self.target_colorspace, self.color_space_manager
                )
                
                # 合并结果
                final_result = {
                    **result,
                    'colorspace_info': colorspace_info
                }
                
                self.optimization_completed.emit(final_result)
            else:
                self.optimization_failed.emit(result.get('error', '优化失败'))
                
        except Exception as e:
            error_msg = f"优化过程出错: {str(e)}\n{traceback.format_exc()}"
            self.optimization_failed.emit(error_msg)


class IDTCalculatorWindow(QMainWindow):
    """精确通道分离IDT计算工具主窗口"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("光源-传感器串扰计算工具")
        self.setMinimumSize(800, 600)
        
        # 初始化组件
        self.engine = IDTCalculationEngine()
        self.color_space_manager = ColorSpaceManager() if ColorSpaceManager else None
        self.optimization_worker = None
        
        # 数据存储
        self.loaded_images = {'red': None, 'green': None, 'blue': None}
        self.optimization_result = None
        
        # 初始化UI
        self._init_ui()
        self._update_ui_state()
    
    def _init_ui(self):
        """初始化用户界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建分割器
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 左侧控制面板
        control_panel = self._create_control_panel()
        splitter.addWidget(control_panel)
        
        # 右侧结果显示面板
        result_panel = self._create_result_panel()
        splitter.addWidget(result_panel)
        
        # 设置分割器比例
        splitter.setSizes([400, 400])
        
        # 状态栏
        self.statusBar().showMessage("准备就绪")
    
    def _create_control_panel(self) -> QWidget:
        """创建控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 图片加载组
        image_group = QGroupBox("图片加载")
        image_layout = QGridLayout(image_group)
        
        # 红光图片
        image_layout.addWidget(QLabel("红光图片:"), 0, 0)
        self.red_path_label = QLabel("未选择")
        self.red_path_label.setStyleSheet("QLabel { border: 1px solid gray; padding: 5px; }")
        image_layout.addWidget(self.red_path_label, 0, 1)
        self.red_button = QPushButton("选择")
        self.red_button.clicked.connect(lambda: self._load_image('red'))
        image_layout.addWidget(self.red_button, 0, 2)
        
        # 绿光图片
        image_layout.addWidget(QLabel("绿光图片:"), 1, 0)
        self.green_path_label = QLabel("未选择")
        self.green_path_label.setStyleSheet("QLabel { border: 1px solid gray; padding: 5px; }")
        image_layout.addWidget(self.green_path_label, 1, 1)
        self.green_button = QPushButton("选择")
        self.green_button.clicked.connect(lambda: self._load_image('green'))
        image_layout.addWidget(self.green_button, 1, 2)
        
        # 蓝光图片
        image_layout.addWidget(QLabel("蓝光图片:"), 2, 0)
        self.blue_path_label = QLabel("未选择")
        self.blue_path_label.setStyleSheet("QLabel { border: 1px solid gray; padding: 5px; }")
        image_layout.addWidget(self.blue_path_label, 2, 1)
        self.blue_button = QPushButton("选择")
        self.blue_button.clicked.connect(lambda: self._load_image('blue'))
        image_layout.addWidget(self.blue_button, 2, 2)
        
        layout.addWidget(image_group)
        
        # 目标工作空间选择组
        workspace_group = QGroupBox("目标工作空间")
        workspace_layout = QVBoxLayout(workspace_group)
        
        self.workspace_combo = QComboBox()
        self._populate_workspace_combo()
        workspace_layout.addWidget(self.workspace_combo)
        
        layout.addWidget(workspace_group)
        
        # 优化参数组
        params_group = QGroupBox("优化参数")
        params_layout = QGridLayout(params_group)
        
        params_layout.addWidget(QLabel("最大迭代次数:"), 0, 0)
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(100, 5000)
        self.max_iter_spin.setValue(1000)
        params_layout.addWidget(self.max_iter_spin, 0, 1)
        
        params_layout.addWidget(QLabel("收敛容差:"), 1, 0)
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(1e-12, 1e-3)
        self.tolerance_spin.setValue(1e-8)
        self.tolerance_spin.setDecimals(12)
        self.tolerance_spin.setSingleStep(1e-9)
        params_layout.addWidget(self.tolerance_spin, 1, 1)
        
        layout.addWidget(params_group)
        
        # 操作按钮组
        button_group = QGroupBox("操作")
        button_layout = QVBoxLayout(button_group)
        
        self.optimize_button = QPushButton("优化求解")
        self.optimize_button.clicked.connect(self._start_optimization)
        button_layout.addWidget(self.optimize_button)
        
        self.save_button = QPushButton("保存IDT基色")
        self.save_button.clicked.connect(self._save_colorspace)
        self.save_button.setEnabled(False)
        button_layout.addWidget(self.save_button)
        
        self.clear_button = QPushButton("清除数据")
        self.clear_button.clicked.connect(self._clear_data)
        button_layout.addWidget(self.clear_button)
        
        layout.addWidget(button_group)
        
        # 添加弹性空间
        layout.addStretch()
        
        return panel
    
    def _create_result_panel(self) -> QWidget:
        """创建结果显示面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # RGB值显示组
        rgb_group = QGroupBox("提取的RGB值")
        rgb_layout = QVBoxLayout(rgb_group)
        
        self.rgb_display = QTextEdit()
        self.rgb_display.setMaximumHeight(120)
        self.rgb_display.setReadOnly(True)
        rgb_layout.addWidget(self.rgb_display)
        
        layout.addWidget(rgb_group)
        
        # 优化结果显示组
        result_group = QGroupBox("优化结果")
        result_layout = QVBoxLayout(result_group)
        
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        result_layout.addWidget(self.result_display)
        
        layout.addWidget(result_group)
        
        # 日志显示组
        log_group = QGroupBox("优化日志")
        log_layout = QVBoxLayout(log_group)
        
        self.log_display = QTextEdit()
        self.log_display.setMaximumHeight(150)
        self.log_display.setReadOnly(True)
        log_layout.addWidget(self.log_display)
        
        layout.addWidget(log_group)
        
        return panel
    
    def _populate_workspace_combo(self):
        """填充工作空间下拉框"""
        self.workspace_combo.clear()
        
        if self.color_space_manager:
            try:
                # 获取所有working_space类型的色彩空间
                working_spaces = []
                if hasattr(self.color_space_manager, 'get_working_color_spaces'):
                    working_spaces = self.color_space_manager.get_working_color_spaces()
                elif hasattr(self.color_space_manager, '_color_spaces'):
                    # 手动筛选working_space类型的色彩空间
                    for name, info in self.color_space_manager._color_spaces.items():
                        if 'type' in info and 'working_space' in info['type']:
                            working_spaces.append(name)
                    working_spaces.sort()
                self.workspace_combo.addItems(working_spaces)
                
                # 设置默认选择为ACEScg
                if "ACEScg" in working_spaces:
                    index = working_spaces.index("ACEScg")
                    self.workspace_combo.setCurrentIndex(index)
                    
            except Exception as e:
                self._log_message(f"获取工作空间列表失败: {e}")
                # 添加一些默认选项
                self.workspace_combo.addItems(["ACEScg", "sRGB", "Kodak2383"])
        else:
            # 如果没有ColorSpaceManager，添加默认选项
            self.workspace_combo.addItems(["ACEScg", "sRGB", "Kodak2383"])
    
    def _load_image(self, channel: str):
        """加载图片"""
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName(
            self, 
            f"选择{channel}光图片",
            "",
            "图片文件 (*.jpg *.jpeg *.png *.tiff *.tif *.bmp);;所有文件 (*)"
        )
        
        if file_path:
            try:
                # 使用计算引擎提取RGB值
                success = self.engine.load_and_extract_rgb(file_path, channel)
                
                if success:
                    self.loaded_images[channel] = file_path
                    
                    # 更新UI显示
                    label = getattr(self, f"{channel}_path_label")
                    label.setText(Path(file_path).name)
                    
                    self._update_rgb_display()
                    self._update_ui_state()
                    
                    self._log_message(f"成功加载{channel}光图片: {Path(file_path).name}")
                else:
                    QMessageBox.warning(self, "加载失败", f"无法加载{channel}光图片")
                    
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载图片时出错: {str(e)}")
                self._log_message(f"加载{channel}光图片失败: {str(e)}")
    
    def _update_rgb_display(self):
        """更新RGB值显示"""
        rgb_text = ""
        for channel in ['red', 'green', 'blue']:
            if channel in self.engine.rgb_values:
                rgb = self.engine.rgb_values[channel]
                rgb_text += f"{channel.upper()}通道: R={rgb[0]:.6f}, G={rgb[1]:.6f}, B={rgb[2]:.6f}\n"
        
        self.rgb_display.setText(rgb_text)
    
    def _update_ui_state(self):
        """更新UI状态"""
        # 检查是否加载了所有三张图片
        all_loaded = all(self.loaded_images[ch] is not None for ch in ['red', 'green', 'blue'])
        self.optimize_button.setEnabled(all_loaded)
        
        # 检查是否有优化结果
        has_result = self.optimization_result is not None
        self.save_button.setEnabled(has_result)
    
    def _start_optimization(self):
        """开始优化"""
        try:
            # 检查数据完整性
            initial_rgb = self.engine.get_initial_rgb_matrix()
            if initial_rgb is None:
                QMessageBox.warning(self, "数据不完整", "请先加载所有三张图片")
                return
            
            # 获取目标工作空间
            target_workspace = self.workspace_combo.currentText()
            if not target_workspace:
                QMessageBox.warning(self, "参数错误", "请选择目标工作空间")
                return
            
            # 清除之前的日志和结果
            self.log_display.clear()
            self.result_display.clear()
            
            # 禁用优化按钮
            self.optimize_button.setEnabled(False)
            self.optimize_button.setText("优化中...")
            
            # 创建并启动优化工作线程
            self.optimization_worker = OptimizationWorker(
                self.engine, target_workspace, self.color_space_manager
            )
            self.optimization_worker.progress_updated.connect(self._on_progress_updated)
            self.optimization_worker.optimization_completed.connect(self._on_optimization_completed)
            self.optimization_worker.optimization_failed.connect(self._on_optimization_failed)
            self.optimization_worker.start()
            
            self._log_message("开始优化计算...")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动优化时出错: {str(e)}")
            self._reset_optimize_button()
    
    def _on_progress_updated(self, message: str):
        """处理进度更新"""
        self._log_message(message)
    
    def _on_optimization_completed(self, result: Dict):
        """处理优化完成"""
        try:
            self.optimization_result = result
            self._display_optimization_result(result)
            self._update_ui_state()
            self._log_message("✓ 优化完成！")
            
        except Exception as e:
            self._log_message(f"处理优化结果时出错: {str(e)}")
        finally:
            self._reset_optimize_button()
    
    def _on_optimization_failed(self, error_message: str):
        """处理优化失败"""
        self._log_message(f"✗ 优化失败: {error_message}")
        QMessageBox.critical(self, "优化失败", error_message)
        self._reset_optimize_button()
    
    def _reset_optimize_button(self):
        """重置优化按钮状态"""
        self.optimize_button.setEnabled(True)
        self.optimize_button.setText("优化求解")
    
    def _display_optimization_result(self, result: Dict):
        """显示优化结果"""
        try:
            text = "优化结果:\n"
            text += "=" * 50 + "\n\n"
            
            # 基本信息
            text += f"迭代次数: {result.get('iterations', 'N/A')}\n"
            text += f"最终MSE: {result.get('mse', 'N/A'):.8f}\n"
            text += f"验证MSE: {result.get('verification_mse', 'N/A'):.8f}\n\n"
            
            # CCM矩阵
            if 'ccm_matrix' in result:
                ccm = result['ccm_matrix']
                text += "计算得到的CCM矩阵:\n"
                for i in range(3):
                    row_text = "[" + ", ".join([f"{ccm[i,j]:8.6f}" for j in range(3)]) + "]\n"
                    text += row_text
                text += "\n"
            
            # 色彩空间信息
            if 'colorspace_info' in result:
                cs_info = result['colorspace_info']
                text += "计算出的原始色彩空间:\n"
                
                primaries = cs_info.get('primaries', {})
                text += f"红色原色: x={primaries.get('R', [0,0])[0]:.6f}, y={primaries.get('R', [0,0])[1]:.6f}\n"
                text += f"绿色原色: x={primaries.get('G', [0,0])[0]:.6f}, y={primaries.get('G', [0,0])[1]:.6f}\n"
                text += f"蓝色原色: x={primaries.get('B', [0,0])[0]:.6f}, y={primaries.get('B', [0,0])[1]:.6f}\n"
                
                white_point = cs_info.get('white_point', [0, 0])
                text += f"白点: x={white_point[0]:.6f}, y={white_point[1]:.6f}\n"
            
            self.result_display.setText(text)
            
        except Exception as e:
            self.result_display.setText(f"显示结果时出错: {str(e)}")
    
    def _save_colorspace(self):
        """保存色彩空间配置文件"""
        if not self.optimization_result:
            QMessageBox.warning(self, "无数据", "没有可保存的优化结果")
            return

        try:
            # 获取色彩空间名称
            name, ok = self._get_colorspace_name()
            if not ok or not name:
                return

            # 构建色彩空间配置
            colorspace_info = self.optimization_result['colorspace_info']
            config = {
                "name": name,
                "type": ["IDT"],
                "primaries": colorspace_info['primaries'],
                "white_point": colorspace_info['white_point'],
                "gamma": 1.0
            }

            # 检查文件是否已存在
            if enhanced_config_manager:
                # 使用 enhanced_config_manager 获取配置目录
                config_dir = Path(enhanced_config_manager.app_config_dir) / "colorspace"
                config_file = config_dir / f"{name}.json"

                if config_file.exists():
                    reply = QMessageBox.question(
                        self, "文件已存在",
                        f"色彩空间 '{name}' 已存在，是否覆盖？",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No
                    )
                    if reply != QMessageBox.Yes:
                        return

                # 使用 enhanced_config_manager 保存配置
                success = enhanced_config_manager.save_user_config("colorspace", name, config)

                if not success:
                    raise Exception("enhanced_config_manager.save_user_config() 返回失败")

                config_file = config_dir / f"{name}.json"  # 更新 config_file 路径用于显示
            else:
                # 回退方案：使用相对路径
                config_dir = Path(__file__).parent.parent.parent.parent / "config" / "colorspace"
                config_dir.mkdir(parents=True, exist_ok=True)
                config_file = config_dir / f"{name}.json"

                # 检查文件是否已存在
                if config_file.exists():
                    reply = QMessageBox.question(
                        self, "文件已存在",
                        f"色彩空间 '{name}' 已存在，是否覆盖？",
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.No
                    )
                    if reply != QMessageBox.Yes:
                        return

                # 保存文件
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=4, ensure_ascii=False)

            # 通知ColorSpaceManager重新加载（如果存在）
            if self.color_space_manager:
                try:
                    self.color_space_manager.reload_config()
                except Exception as e:
                    self._log_message(f"重新加载色彩空间时出错: {e}")

            QMessageBox.information(self, "保存成功", f"色彩空间已保存到:\n{config_file}")
            self._log_message(f"色彩空间已保存: {config_file}")

        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存色彩空间时出错: {str(e)}")
            self._log_message(f"保存失败: {str(e)}")
    
    def _get_colorspace_name(self) -> tuple:
        """获取色彩空间名称"""
        from PySide6.QtWidgets import QInputDialog
        
        name, ok = QInputDialog.getText(
            self, "色彩空间名称", 
            "请输入色彩空间名称:",
            text="CustomIDT"
        )
        
        if ok and name:
            # 验证名称合法性
            if not name.replace('_', '').replace('-', '').isalnum():
                QMessageBox.warning(self, "无效名称", "名称只能包含字母、数字、下划线和横线")
                return "", False
        
        return name, ok
    
    def _clear_data(self):
        """清除所有数据"""
        reply = QMessageBox.question(
            self, "确认清除", 
            "确定要清除所有数据吗？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 清除引擎数据
            self.engine.clear_data()
            
            # 清除UI显示
            self.loaded_images = {'red': None, 'green': None, 'blue': None}
            self.optimization_result = None
            
            self.red_path_label.setText("未选择")
            self.green_path_label.setText("未选择")
            self.blue_path_label.setText("未选择")
            
            self.rgb_display.clear()
            self.result_display.clear()
            self.log_display.clear()
            
            self._update_ui_state()
            self._log_message("数据已清除")
    
    def _log_message(self, message: str):
        """添加日志消息"""
        self.log_display.append(message)
        # 滚动到底部
        scrollbar = self.log_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.optimization_worker and self.optimization_worker.isRunning():
            reply = QMessageBox.question(
                self, "确认关闭", 
                "优化正在进行中，确定要关闭窗口吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.optimization_worker.terminate()
                self.optimization_worker.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()