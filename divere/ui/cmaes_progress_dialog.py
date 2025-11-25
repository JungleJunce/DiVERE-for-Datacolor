"""
CMA-ES 优化进度对话框
显示优化迭代进度、log-RMSE 值和其他相关信息
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QTextEdit, QGroupBox
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QFont
import re
from divere.i18n import tr


class CMAESProgressDialog(QDialog):
    """CMA-ES优化进度对话框"""
    
    # 线程安全的信号，用于从worker线程更新UI
    update_progress_signal = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("cmaes_dialog.title"))
        self.setModal(True)
        self.resize(500, 400)
        
        # 优化状态
        self.is_running = False
        self.current_iteration = 0
        self.max_iterations = 300  # 默认值
        self.best_log_rmse = float('inf')
        self.current_log_rmse = float('inf')
        
        self._setup_ui()
        
        # 连接信号到槽函数，确保线程安全
        self.update_progress_signal.connect(self._update_progress_slot)
        
    def _setup_ui(self):
        """设置用户界面"""
        layout = QVBoxLayout(self)
        
        # 标题
        title_label = QLabel(tr("cmaes_dialog.message"))
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(12)
        title_label.setFont(title_font)
        layout.addWidget(title_label)
        
        # 进度信息组
        progress_group = QGroupBox(tr("cmaes_dialog.groups.progress"))
        progress_layout = QVBoxLayout(progress_group)
        
        # 迭代进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        # 进度信息标签
        info_layout = QHBoxLayout()
        self.iteration_label = QLabel(tr("cmaes_dialog.iteration_label", current=0, total=300))
        self.log_rmse_label = QLabel(tr("cmaes_dialog.current_rmse_label", value="--"))
        self.best_log_rmse_label = QLabel(tr("cmaes_dialog.best_rmse_label", value="--"))
        
        info_layout.addWidget(self.iteration_label)
        info_layout.addStretch()
        info_layout.addWidget(self.log_rmse_label)
        info_layout.addStretch()
        info_layout.addWidget(self.best_log_rmse_label)
        
        progress_layout.addLayout(info_layout)
        layout.addWidget(progress_group)
        
        # 详细日志
        log_group = QGroupBox(tr("cmaes_dialog.groups.detailed_log"))
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        
        layout.addWidget(log_group)
        
        # 按钮
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.cancel_button = QPushButton(tr("cmaes_dialog.button_cancel"))
        self.cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_button)

        self.close_button = QPushButton(tr("cmaes_dialog.button_close"))
        self.close_button.clicked.connect(self.accept)
        self.close_button.setEnabled(False)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
        
    def set_max_iterations(self, max_iter: int):
        """设置最大迭代次数"""
        self.max_iterations = max_iter
        self.iteration_label.setText(tr("cmaes_dialog.iteration_label", current=self.current_iteration, total=max_iter))
        
    def start_optimization(self):
        """开始优化"""
        self.is_running = True
        self.current_iteration = 0
        self.best_log_rmse = float('inf')
        self.current_log_rmse = float('inf')
        
        self.cancel_button.setEnabled(True)
        self.close_button.setEnabled(False)
        
        self.progress_bar.setValue(0)
        self.log_text.clear()
        self.add_log_message(tr("cmaes_dialog.messages.starting"))
        
    def finish_optimization(self, success: bool, final_log_rmse: float = None):
        """完成优化"""
        self.is_running = False
        self.cancel_button.setEnabled(False)
        self.close_button.setEnabled(True)
        
        if success:
            if final_log_rmse is not None:
                self.add_log_message(tr("cmaes_dialog.messages.success", final_log_rmse=final_log_rmse))
            else:
                self.add_log_message(tr("cmaes_dialog.messages.success", final_log_rmse=self.best_log_rmse))
            self.progress_bar.setValue(100)
        else:
            self.add_log_message(tr("cmaes_dialog.messages.failed"))
            
    def request_update_progress(self, message: str):
        """线程安全的进度更新请求接口"""
        # 发射信号，让槽函数在主线程中处理
        self.update_progress_signal.emit(message)
    
    @Slot(str)
    def _update_progress_slot(self, message: str):
        """槽函数：在主线程中更新进度信息"""
        print(f"[DEBUG] CMAESProgressDialog._update_progress_slot: '{message}'")
        print(f"[DEBUG] is_running: {self.is_running}")
        
        if not self.is_running:
            print(f"[DEBUG] 优化未运行，启动优化状态")
            self.start_optimization()
            
        # 解析CMA-ES消息
        if "迭代" in message:
            print(f"[DEBUG] 检测到迭代消息，解析中...")
            self._parse_iteration_message(message)
        elif "优化成功完成" in message:
            print(f"[DEBUG] 检测到优化完成消息")
            self.finish_optimization(True)
        elif "开始优化" in message:
            print(f"[DEBUG] 检测到开始优化消息")
            self.start_optimization()
        
        # 添加到日志
        self.add_log_message(message)
    
    # 为了向后兼容，保留原方法名但重定向到线程安全版本
    def update_progress(self, message: str):
        """更新进度信息（向后兼容方法）"""
        self.request_update_progress(message)
        
    def _parse_iteration_message(self, message: str):
        """解析迭代消息"""
        # 尝试提取迭代数和log-RMSE值
        patterns = [
            r'迭代\s*(\d+)\s*:\s*log-RMSE=([\d.]+)',
            r'迭代\s+(\d+)\s*:\s*log-RMSE=([\d.]+)',
            r'迭代.*?(\d+).*?log-RMSE=(\d*\.?\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message)
            if match and len(match.groups()) >= 2:
                try:
                    iteration = int(match.group(1))
                    log_rmse = float(match.group(2))
                    
                    self.current_iteration = iteration
                    self.current_log_rmse = log_rmse
                    
                    if log_rmse < self.best_log_rmse:
                        self.best_log_rmse = log_rmse
                    
                    # 更新界面
                    self.iteration_label.setText(tr("cmaes_dialog.iteration_label", current=iteration, total=self.max_iterations))
                    self.log_rmse_label.setText(tr("cmaes_dialog.current_rmse_label", value=f"{log_rmse:.6f}"))
                    self.best_log_rmse_label.setText(tr("cmaes_dialog.best_rmse_label", value=f"{self.best_log_rmse:.6f}"))
                    
                    # 更新进度条
                    if self.max_iterations > 0:
                        progress = min(100, int(iteration * 100 / self.max_iterations))
                        self.progress_bar.setValue(progress)
                    
                    return
                except (ValueError, IndexError):
                    continue
                    
        # 如果无法解析，尝试只提取迭代数
        iteration_match = re.search(r'迭代.*?(\d+)', message)
        if iteration_match:
            try:
                iteration = int(iteration_match.group(1))
                self.current_iteration = iteration
                self.iteration_label.setText(tr("cmaes_dialog.iteration_label", current=iteration, total=self.max_iterations))
                
                if self.max_iterations > 0:
                    progress = min(100, int(iteration * 100 / self.max_iterations))
                    self.progress_bar.setValue(progress)
            except (ValueError, IndexError):
                pass
                
    def add_log_message(self, message: str):
        """添加日志消息"""
        self.log_text.append(message)
        # 自动滚动到底部
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum()
        )
        
    def closeEvent(self, event):
        """关闭事件处理"""
        if self.is_running:
            # 如果优化正在进行，询问是否取消
            from PySide6.QtWidgets import QMessageBox
            reply = QMessageBox.question(
                self, tr("main_window.dialogs.confirm"), tr("cmaes_dialog.messages.confirm_cancel"),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.reject()
            else:
                event.ignore()
                return

        super().closeEvent(event)