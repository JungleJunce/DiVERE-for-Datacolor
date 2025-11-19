"""
可视化曲线编辑器组件
支持拖拽编辑控制点，实时预览曲线效果
"""

import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                            QPushButton, QComboBox, QGroupBox, QDoubleSpinBox,
                            QGridLayout, QSizePolicy, QFileDialog, QInputDialog, QMessageBox)
from PySide6.QtCore import Qt, QPointF, Signal, QRectF
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QPolygonF, QPainterPath
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from ..core.data_types import Curve


class CurveEditWidget(QWidget):
    """曲线编辑画布"""
    
    curve_changed = Signal(list)  # 当曲线改变时发出信号，传递控制点列表
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 460)  # 增加最小高度以容纳正方形绘图区域
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # 曲线控制点 [(x, y), ...]
        # 支持多通道曲线：RGB, R, G, B
        self.curves = {
            'RGB': [(0.0, 0.0), (1.0, 1.0)],
            'R': [(0.0, 0.0), (1.0, 1.0)],
            'G': [(0.0, 0.0), (1.0, 1.0)],
            'B': [(0.0, 0.0), (1.0, 1.0)]
        }
        self.current_channel = 'RGB'  # 当前编辑的通道
        self.control_points = self.curves[self.current_channel]  # 向后兼容
        self.selected_point = -1
        self.dragging = False
        
        # 样式设置 - 使用可更新的调色板变量
        self.grid_color = QColor(220, 220, 220)
        self.curve_color = QColor(80, 80, 80)
        self.point_color = QColor(100, 100, 100)
        self.selected_point_color = QColor(60, 60, 60)
        self.background_color = QColor(250, 250, 250)
        self.text_color = QColor(20, 20, 20)
        self.frame_color = QColor(0, 0, 0)
        
        # 通道曲线颜色
        self.channel_colors = {
            'RGB': QColor(80, 80, 80),      # 深灰色
            'R': QColor(200, 80, 80, 128),   # 半透明红色
            'G': QColor(80, 200, 80, 128),   # 半透明绿色
            'B': QColor(80, 80, 200, 128)    # 半透明蓝色
        }
        
        self.dmax = 4.0  # 默认最大密度值
        self.gamma = 1.0 # 默认gamma值
        
        # 曲线分辨率
        self.curve_resolution = 256
        
        self.setMouseTracking(True)
    
    def set_dmax(self, dmax: float):
        """设置最大密度值用于坐标轴显示"""
        self.dmax = dmax
        self.update()

    def set_gamma(self, gamma: float):
        """设置gamma值用于坐标轴显示"""
        self.gamma = gamma
        self.update()
    
    def set_current_channel(self, channel: str):
        """设置当前编辑的通道"""
        if channel in self.curves:
            self.current_channel = channel
            self.control_points = self.curves[channel]
            self.selected_point = -1
            self.update()
            self.curve_changed.emit(self.control_points)
    
    def set_curve_points(self, points: List[Tuple[float, float]], channel: str = None, emit_signal: bool = True):
        """设置曲线控制点"""
        if channel is None:
            channel = self.current_channel

        # 规范化channel键名：统一转换为大写，避免大小写不匹配问题
        channel = channel.upper() if channel else self.current_channel

        if channel in self.curves:
            # 确保所有点都是tuple格式，并按x坐标排序
            normalized_points = []
            for point in points:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    normalized_points.append((float(point[0]), float(point[1])))

            self.curves[channel] = sorted(normalized_points, key=lambda p: p[0])
            if channel == self.current_channel:
                self.control_points = self.curves[channel]
                self.selected_point = -1
                self.update()
                if emit_signal:
                    self.curve_changed.emit(self.control_points)
    
    def get_curve_points(self, channel: str = None) -> List[Tuple[float, float]]:
        """获取曲线控制点"""
        if channel is None:
            channel = self.current_channel
        return self.curves.get(channel, [(0.0, 0.0), (1.0, 1.0)])
    
    def get_all_curves(self) -> Dict[str, List[Tuple[float, float]]]:
        """获取所有通道的曲线"""
        return self.curves.copy()
    
    def set_all_curves(self, curves: Dict[str, List[Tuple[float, float]]], emit_signal: bool = True):
        """设置所有通道的曲线"""
        for channel, points in curves.items():
            # 规范化channel键名：统一转换为大写，避免大小写不匹配问题
            channel_upper = channel.upper()
            if channel_upper in self.curves:
                # 确保所有点都是tuple格式，并按x坐标排序
                normalized_points = []
                for point in points:
                    if isinstance(point, (list, tuple)) and len(point) >= 2:
                        normalized_points.append((float(point[0]), float(point[1])))

                self.curves[channel_upper] = sorted(normalized_points, key=lambda p: p[0])
        
        # 更新当前显示的曲线
        if self.current_channel in self.curves:
            self.control_points = self.curves[self.current_channel]
            self.selected_point = -1
            self.update()
            if emit_signal:
                self.curve_changed.emit(self.control_points)
    
    def add_point(self, x: float, y: float):
        """添加控制点"""
        # 确保坐标在[0,1]范围内
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        
        # 检查是否过于接近现有点
        min_distance = 0.05  # 最小距离阈值
        for i, (px, py) in enumerate(self.control_points):
            if abs(px - x) < min_distance:
                # 如果太接近现有点，就选中现有点而不是添加新点
                self.selected_point = i
                self.update()
                return
        
        # 插入到正确位置保持x坐标排序
        inserted = False
        for i, (px, py) in enumerate(self.control_points):
            if x < px:
                self.control_points.insert(i, (x, y))
                self.selected_point = i
                inserted = True
                break
        
        if not inserted:
            self.control_points.append((x, y))
            self.selected_point = len(self.control_points) - 1
        
        self.update()
        self.curve_changed.emit(self.control_points)
    
    def remove_selected_point(self):
        """删除选中的控制点"""
        if (self.selected_point >= 0 and 
            self.selected_point < len(self.control_points) and
            len(self.control_points) > 2):
            # 不允许删除第一个和最后一个点
            if 0 < self.selected_point < len(self.control_points) - 1:
                del self.control_points[self.selected_point]
                self.selected_point = -1
                self.update()
                self.curve_changed.emit(self.control_points)
    
    def _get_draw_rect(self) -> QRectF:
        """获取绘制区域的矩形（考虑边距，保持1:1长宽比）"""
        rect = self.rect()
        left_margin = 40
        top_margin = 20
        right_margin = 20
        bottom_margin = 40  # 减小底部边距以节省空间

        # 计算可用空间
        available_width = rect.width() - left_margin - right_margin
        available_height = rect.height() - top_margin - bottom_margin

        # 取较小值作为正方形边长，保持1:1长宽比
        size = available_width

        # 居中对齐
        x_offset = left_margin + (available_width - size) / 2
        y_offset = top_margin + (available_height - size) / 2

        return QRectF(x_offset, y_offset, size, size)

    def _widget_to_curve_coords(self, widget_x: int, widget_y: int) -> Tuple[float, float]:
        """将组件坐标转换为曲线坐标(0-1)"""
        draw_rect = self._get_draw_rect()

        curve_x = (widget_x - draw_rect.left()) / draw_rect.width()
        curve_y = 1.0 - (widget_y - draw_rect.top()) / draw_rect.height()

        return max(0.0, min(1.0, curve_x)), max(0.0, min(1.0, curve_y))

    def _curve_to_widget_coords(self, curve_x: float, curve_y: float) -> Tuple[int, int]:
        """将曲线坐标转换为组件坐标"""
        draw_rect = self._get_draw_rect()

        widget_x = draw_rect.left() + curve_x * draw_rect.width()
        widget_y = draw_rect.top() + (1.0 - curve_y) * draw_rect.height()

        return int(widget_x), int(widget_y)
    
    def _find_point_near(self, x: int, y: int) -> int:
        """查找鼠标附近的控制点"""
        for i, (px, py) in enumerate(self.control_points):
            wx, wy = self._curve_to_widget_coords(px, py)
            if abs(wx - x) <= 8 and abs(wy - y) <= 8:
                return i
        return -1
    
    def _interpolate_curve(self, points: List[Tuple[float, float]] = None) -> List[Tuple[float, float]]:
        """插值生成平滑曲线 - 使用单调三次插值（类似Photoshop）"""
        if points is None:
            points = self.control_points
            
        if len(points) < 2:
            return points
        
        if len(points) == 2:
            # 只有两个点时使用线性插值
            curve_points = []
            for i in range(self.curve_resolution + 1):
                t = i / self.curve_resolution
                x = points[0][0] + t * (points[1][0] - points[0][0])
                y = points[0][1] + t * (points[1][1] - points[0][1])
                curve_points.append((x, y))
            return curve_points
        
        # 使用单调三次样条插值（更接近Photoshop的行为）
        curve_points = []
        
        # 按x坐标对控制点排序
        sorted_points = sorted(points, key=lambda p: p[0])
        
        # 生成插值点
        for i in range(self.curve_resolution + 1):
            x = i / self.curve_resolution
            y = self._monotonic_cubic_interpolate(x, sorted_points)
            curve_points.append((x, y))
        
        return curve_points
    
    def _monotonic_cubic_interpolate(self, x: float, points: List[Tuple[float, float]]) -> float:
        """单调三次插值（类似Photoshop的曲线插值）"""
        if not points:
            return x  # 默认线性
        
        # 找到x所在的区间
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            
            if x1 <= x <= x2:
                if x2 - x1 == 0:
                    return y1
                
                # 计算局部切线斜率（避免过冲）
                if i == 0:
                    # 第一段：使用当前段的斜率
                    m1 = (y2 - y1) / (x2 - x1)
                else:
                    # 中间段：使用相邻段的平均斜率
                    x0, y0 = points[i - 1]
                    m1 = ((y1 - y0) / (x1 - x0) + (y2 - y1) / (x2 - x1)) * 0.5
                
                if i == len(points) - 2:
                    # 最后一段：使用当前段的斜率
                    m2 = (y2 - y1) / (x2 - x1)
                else:
                    # 中间段：使用相邻段的平均斜率
                    x3, y3 = points[i + 2]
                    m2 = ((y2 - y1) / (x2 - x1) + (y3 - y2) / (x3 - x2)) * 0.5
                
                # Hermite插值（更平滑，类似Photoshop）
                t = (x - x1) / (x2 - x1)
                t2 = t * t
                t3 = t2 * t
                
                h00 = 2*t3 - 3*t2 + 1
                h10 = t3 - 2*t2 + t
                h01 = -2*t3 + 3*t2
                h11 = t3 - t2
                
                result = (h00 * y1 + h10 * (x2 - x1) * m1 + h01 * y2 + h11 * (x2 - x1) * m2)
                # 限制结果在[0,1]范围内
                return max(0.0, min(1.0, result))
        
        # 如果x在范围外，返回最近端点的y值
        if x <= points[0][0]:
            return points[0][1]
        else:
            return points[-1][1]
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            x, y = event.position().x(), event.position().y()
            point_index = self._find_point_near(x, y)
            
            if point_index >= 0:
                # 选中已存在的点
                self.selected_point = point_index
                self.dragging = True
            else:
                # 添加新点
                curve_x, curve_y = self._widget_to_curve_coords(x, y)
                self.add_point(curve_x, curve_y)
                self.dragging = True
            
            self.update()
        
        elif event.button() == Qt.MouseButton.RightButton:
            # 右键删除点
            x, y = event.position().x(), event.position().y()
            point_index = self._find_point_near(x, y)
            if point_index >= 0:
                self.selected_point = point_index
                self.remove_selected_point()
                self.update()
    
    def mouseMoveEvent(self, event):
        if self.dragging and self.selected_point >= 0:
            x, y = event.position().x(), event.position().y()
            curve_x, curve_y = self._widget_to_curve_coords(x, y)
            
            # 限制Y坐标在[0,1]范围内
            curve_y = max(0.0, min(1.0, curve_y))
            
            # 允许编辑端部点，但限制X坐标在[0,1]范围内
            curve_x = max(0.0, min(1.0, curve_x))
            
            # 对于中间点，限制在相邻点之间（保持顺序）
            if 0 < self.selected_point < len(self.control_points) - 1:
                left_x = self.control_points[self.selected_point - 1][0]
                right_x = self.control_points[self.selected_point + 1][0]
                curve_x = max(left_x + 0.01, min(right_x - 0.01, curve_x))
            
            # 直接更新选中点，不重新排序
            self.control_points[self.selected_point] = (curve_x, curve_y)
            
            self.update()
            self.curve_changed.emit(self.control_points)
    
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete:
            self.remove_selected_point()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        draw_rect = self._get_draw_rect()
        
        # 绘制背景
        painter.fillRect(draw_rect, self.background_color)
        
        # 绘制网格（以密度为单位）
        painter.setPen(QPen(self.grid_color, 1))
        
        # 横向网格线（对应Y轴密度值）
        # Y轴从上到下：2.33到-13.67密度，每整数一格
        y_density_min = -13.67
        y_density_max = 2.33
        y_density_range = y_density_max - y_density_min

        for i in range(-13, 3):  # -13, -12, ..., 0, 1, 2
            density = float(i)
            # 将密度映射到归一化坐标[0,1]，其中0对应顶部，1对应底部
            norm_y = (y_density_max - density) / y_density_range
            y = draw_rect.top() + norm_y * draw_rect.height()
            painter.drawLine(int(draw_rect.left()), int(y),
                           int(draw_rect.right()), int(y))
        
        # 垂直网格线（对应X轴密度值）
        # X轴是固定宽度的滑动窗口：左边界 = dmax - 4.816，右边界 = dmax
        x_density_range = np.log10(65536)  # ≈ 4.816，固定窗口宽度
        x_density_min = self.dmax - x_density_range
        x_density_max = self.dmax
        x_density_step = 0.3

        # 找到窗口范围内的第一个和最后一个0.3倍数
        first_grid = int(np.ceil(x_density_min / x_density_step)) * x_density_step
        last_grid = int(np.floor(x_density_max / x_density_step)) * x_density_step

        # 绘制所有在窗口内的网格线
        density = first_grid
        while density <= last_grid:
            # 将密度映射到归一化坐标[0,1]
            norm_x = (density - x_density_min) / x_density_range
            x = draw_rect.left() + norm_x * draw_rect.width()
            painter.drawLine(int(x), int(draw_rect.top()),
                           int(x), int(draw_rect.bottom()))
            density += x_density_step
        
        # 绘制边框
        painter.setPen(QPen(self.frame_color, 2))
        painter.drawRect(draw_rect)
        
        # 绘制曲线
        # 如果在RGB模式，先绘制暗淡的R、G、B曲线
        if self.current_channel == 'RGB':
            for channel in ['R', 'G', 'B']:
                points = self.curves.get(channel, [])
                if len(points) >= 2:
                    curve_points = self._interpolate_curve(points)
                    if curve_points:
                        polygon = QPolygonF()
                        for x, y in curve_points:
                            wx, wy = self._curve_to_widget_coords(x, y)
                            polygon.append(QPointF(wx, wy))
                        
                        # 使用半透明的通道颜色
                        painter.setPen(QPen(self.channel_colors[channel], 1.0))
                        painter.drawPolyline(polygon)
        
        # 绘制当前通道的曲线（主曲线）
        if len(self.control_points) >= 2:
            curve_points = self._interpolate_curve(self.control_points)
            
            if curve_points:
                polygon = QPolygonF()
                for x, y in curve_points:
                    wx, wy = self._curve_to_widget_coords(x, y)
                    polygon.append(QPointF(wx, wy))
                
                # 使用当前通道的颜色，更粗一些
                color = self.channel_colors.get(self.current_channel, self.curve_color)
                if self.current_channel != 'RGB':
                    # 单通道模式下使用不透明的颜色
                    color = QColor(color.red(), color.green(), color.blue(), 255)
                painter.setPen(QPen(color, 2.0))  # 主曲线更粗
                painter.drawPolyline(polygon)
        
        # 绘制控制点
        for i, (x, y) in enumerate(self.control_points):
            wx, wy = self._curve_to_widget_coords(x, y)
            
            color = self.selected_point_color if i == self.selected_point else self.point_color
            painter.setPen(QPen(color, 1))  # 更细的边框
            painter.setBrush(QBrush(color))
            painter.drawEllipse(int(wx - 3), int(wy - 3), 6, 6)  # 更小的控制点
        
        # 绘制坐标标签
        painter.setPen(QPen(self.text_color, 1))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)

        # 绘制Y轴标签（密度值）
        y_density_min = -13.67
        y_density_max = 2.33
        y_density_range = y_density_max - y_density_min

        for i in range(-13, 3):  # -13, -12, ..., 0, 1, 2
            density = float(i)
            # 将密度映射到归一化坐标[0,1]，其中0对应顶部，1对应底部
            norm_y = (y_density_max - density) / y_density_range
            y = draw_rect.top() + norm_y * draw_rect.height()
            # 右对齐y轴标签
            text = f"{int(density)}"
            text_width = painter.fontMetrics().horizontalAdvance(text)
            x = int(draw_rect.left()) - text_width - 5  # 右对齐到绘图区域左边界，留5像素间距
            painter.drawText(x, int(y) + 4, text)

        # 绘制X轴标签（密度值）
        # 使用与网格线相同的滑动窗口范围
        x_density_range = np.log10(65536)  # ≈ 4.816，固定窗口宽度
        x_density_min = self.dmax - x_density_range
        x_density_max = self.dmax
        x_density_step = 0.3

        # 找到窗口范围内的第一个和最后一个0.3倍数
        first_label = int(np.ceil(x_density_min / x_density_step)) * x_density_step
        last_label = int(np.floor(x_density_max / x_density_step)) * x_density_step

        # 绘制所有在窗口内且非负的标签
        density = first_label
        while density <= last_label:
            if density >= 0:  # 只显示非负值标签
                # 将密度映射到归一化坐标[0,1]
                norm_x = (density - x_density_min) / x_density_range
                wx = draw_rect.left() + norm_x * draw_rect.width()
                text_width = painter.fontMetrics().horizontalAdvance(f"{density:.1f}")
                painter.drawText(int(wx) - text_width // 2, int(draw_rect.bottom()) + 15, f"{density:.1f}")
            density += x_density_step
        
        # 绘制轴标题
        font.setPointSize(10)
        painter.setFont(font)
        
        # X轴标题
        x_title = "输入密度"
        x_title_width = painter.fontMetrics().horizontalAdvance(x_title)
        painter.setPen(QPen(self.text_color, 1))
        painter.drawText(int(draw_rect.center().x() - x_title_width // 2), 
                        int(draw_rect.bottom()) + 35, x_title)
        
        # Y轴标题（垂直绘制）
        painter.save()
        painter.translate(15, int(draw_rect.center().y()))
        painter.rotate(-90)
        y_title = "SDR EV值"
        y_title_width = painter.fontMetrics().horizontalAdvance(y_title)
        painter.setPen(QPen(self.text_color, 1))
        painter.drawText(-y_title_width // 2, -5, y_title)
        painter.restore()

    def apply_palette(self, palette, theme: str):
        """根据主题/调色板更新绘制颜色（不更改业务逻辑）。"""
        try:
            if (theme or "dark").lower() == "dark":
                self.background_color = QColor(45, 45, 45)
                self.grid_color = QColor(150, 150, 150)  # 更亮的网格线
                self.curve_color = QColor(220, 220, 220)
                self.point_color = QColor(210, 210, 210)
                self.selected_point_color = QColor(255, 255, 255)
                self.text_color = QColor(230, 230, 230)
                self.frame_color = QColor(200, 200, 200)  # 更亮的边框
                self.channel_colors = {
                    'RGB': QColor(220, 220, 220),
                    'R': QColor(255, 120, 120, 180),
                    'G': QColor(120, 255, 120, 180),
                    'B': QColor(120, 120, 255, 180),
                }
            else:
                self.background_color = palette.window().color() if hasattr(palette, 'window') else QColor(250, 250, 250)
                self.grid_color = QColor(160, 160, 160)  # 更暗的网格线，提高对比度
                self.curve_color = QColor(80, 80, 80)
                self.point_color = QColor(100, 100, 100)
                self.selected_point_color = QColor(60, 60, 60)
                self.text_color = QColor(20, 20, 20)
                self.frame_color = QColor(0, 0, 0)
                self.channel_colors = {
                    'RGB': QColor(80, 80, 80),
                    'R': QColor(200, 80, 80, 128),
                    'G': QColor(80, 200, 80, 128),
                    'B': QColor(80, 80, 200, 128),
                }
            self.update()
        except Exception:
            pass


class CurveEditorWidget(QWidget):
    """完整的曲线编辑器组件"""
    
    curve_changed = Signal(str, list)  # 曲线名称和控制点
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.preset_curves = {}
        self.current_curve_name = "custom"
        self.original_curve_name = None  # 记录用户选择的原始曲线名称
        self.original_curve_key = None   # 记录用户选择的原始曲线key
        self.is_modified = False         # 标记当前曲线是否被修改
        self.modified_item_index = -1    # 记录修改状态选项的索引
        self._loading_preset = False     # 标记是否正在加载预设（防止误触发修改状态）
        
        self._load_preset_curves()
        self._setup_ui()
        self._connect_signals()
    
    def _load_preset_curves(self):
        """从curves目录加载已保存的曲线（支持用户配置优先）"""
        self.preset_curves = {}
        
        try:
            from divere.utils.enhanced_config_manager import enhanced_config_manager
            
            # 获取所有配置文件（用户配置优先）
            config_files = enhanced_config_manager.get_config_files("curves")
            
            for json_file in config_files:
                try:
                    data = enhanced_config_manager.load_config_file(json_file)
                    if data is None:
                        continue
                    
                    # 支持新旧两种格式
                    if 'curves' in data and isinstance(data['curves'], dict):
                        # 新格式（多通道）
                        self.preset_curves[json_file.stem] = {
                            "name": data.get("name", json_file.stem),
                            "description": data.get("description", ""),
                            "curves": data["curves"]
                        }
                    elif 'points' in data:
                        # 旧格式（单曲线）- 转换为新格式
                        points = data["points"]
                        self.preset_curves[json_file.stem] = {
                            "name": data.get("name", json_file.stem),
                            "description": data.get("description", ""),
                            "curves": {
                                "RGB": points,
                                "R": [(0.0, 0.0), (1.0, 1.0)],
                                "G": [(0.0, 0.0), (1.0, 1.0)],
                                "B": [(0.0, 0.0), (1.0, 1.0)]
                            }
                        }
                    
                    # 标记是否为用户配置
                    if json_file.parent == enhanced_config_manager.user_curves_dir:
                        print(f"加载用户曲线: {json_file.stem}")
                    else:
                        print(f"加载内置曲线: {json_file.stem}")
                        
                except Exception as e:
                    print(f"加载曲线文件 {json_file} 时出错: {e}")
                    
        except ImportError:
            # 如果增强配置管理器不可用，使用原来的方法
            curves_dir = Path("config/curves")
            
            if curves_dir.exists():
                for json_file in curves_dir.glob("*.json"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                            # 支持新旧两种格式
                            if 'curves' in data and isinstance(data['curves'], dict):
                                # 新格式（多通道）
                                self.preset_curves[json_file.stem] = {
                                    "name": data.get("name", json_file.stem),
                                    "description": data.get("description", ""),
                                    "curves": data["curves"]
                                }
                            elif 'points' in data:
                                # 旧格式（单曲线）- 转换为新格式
                                points = data["points"]
                                self.preset_curves[json_file.stem] = {
                                    "name": data.get("name", json_file.stem),
                                    "description": data.get("description", ""),
                                    "curves": {
                                        "RGB": points,
                                        "R": [(0.0, 0.0), (1.0, 1.0)],
                                        "G": [(0.0, 0.0), (1.0, 1.0)],
                                        "B": [(0.0, 0.0), (1.0, 1.0)]
                                    }
                                }
                    except Exception as e:
                        print(f"加载曲线文件 {json_file} 时出错: {e}")
    
    def _setup_ui(self):
        """设置用户界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 通道选择（放在曲线控件上方）
        channel_layout = QHBoxLayout()
        channel_layout.addWidget(QLabel("通道:"))
        self.channel_combo = QComboBox()
        self.channel_combo.addItem("RGB", "RGB")
        self.channel_combo.addItem("红", "R")
        self.channel_combo.addItem("绿", "G")
        self.channel_combo.addItem("蓝", "B")
        self.channel_combo.setMaximumWidth(150)
        channel_layout.addWidget(self.channel_combo)
        channel_layout.addStretch()
        
        layout.addLayout(channel_layout)
        
        # 曲线编辑画布
        self.curve_edit_widget = CurveEditWidget()
        layout.addWidget(self.curve_edit_widget, 1)
        
        # 曲线控制（已保存曲线和操作按钮）
        control_layout = QHBoxLayout()
        
        # 已保存曲线选择
        control_layout.addWidget(QLabel("已保存曲线:"))
        self.curve_combo = QComboBox()
        # 不添加固定的custom选项，只添加实际的预设曲线
        for curve_key, curve_data in sorted(self.preset_curves.items(), key=lambda x: x[1]["name"]):
            self.curve_combo.addItem(curve_data["name"], curve_key)
        self.curve_combo.setMaximumWidth(200)
        control_layout.addWidget(self.curve_combo)
        
        # 设置默认选择为"Kodak Endura Paper"
        kodak_index = self.curve_combo.findText("Kodak Endura Paper")
        if kodak_index >= 0:
            self.curve_combo.setCurrentIndex(kodak_index)
        
        control_layout.addStretch()
        
        # 操作按钮
        self.reset_button = QPushButton("重置为线性")
        self.save_button = QPushButton("保存曲线")
        control_layout.addWidget(self.reset_button)
        control_layout.addWidget(self.save_button)
        
        layout.addLayout(control_layout)
        
        # 使用说明
        help_label = QLabel(
            "使用说明：左键点击添加/选择控制点，拖拽移动点，右键删除点，Delete键删除选中点"
        )
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(help_label)
    
    def _connect_signals(self):
        """连接信号"""
        self.channel_combo.currentTextChanged.connect(self._on_channel_changed)
        self.curve_combo.currentTextChanged.connect(self._on_preset_curve_changed)
        self.reset_button.clicked.connect(self._reset_to_linear)
        self.save_button.clicked.connect(self._save_curve)
        
        self.curve_edit_widget.curve_changed.connect(self._on_curve_changed)
    
    def _on_channel_changed(self):
        """通道选择改变"""
        channel = self.channel_combo.currentData()
        self.curve_edit_widget.set_current_channel(channel)
    
    def _on_preset_curve_changed(self):
        """预设曲线改变"""
        curve_key = self.curve_combo.currentData()
        
        # 处理custom选项
        if curve_key == "custom":
            self.current_curve_name = "custom"
            self.original_curve_name = None
            self.original_curve_key = None
            self.is_modified = False
            return
        
        # 处理修改状态选项（用户切换回修改状态）
        if isinstance(curve_key, str) and curve_key.startswith("*"):
            # 用户选择了修改状态选项，需要加载对应的原始曲线数据并保持修改状态显示
            original_name = curve_key[1:]  # 去掉*前缀
            self._load_original_curve_as_modified(original_name)
            return
        
        # 处理原始预设曲线选择
        if curve_key in self.preset_curves:
            # 设置加载标志，防止触发修改状态
            self._loading_preset = True
            try:
                curve_data = self.preset_curves[curve_key]
                self.current_curve_name = curve_data["name"]
                # 记录原始曲线信息
                self.original_curve_name = curve_data["name"]
                self.original_curve_key = curve_key
                
                # 切换到原始曲线时，重置修改状态但保留修改选项
                # 用户期望修改状态和原始状态可以共存，自由切换
                self.is_modified = False
                
                # 加载所有通道的曲线，不触发信号避免跳转到修改状态
                if "curves" in curve_data:
                    self.curve_edit_widget.set_all_curves(curve_data["curves"], emit_signal=False)
                else:
                    # 兼容旧格式
                    points = curve_data.get("points", [(0.0, 0.0), (1.0, 1.0)])
                    self.curve_edit_widget.set_curve_points(points, emit_signal=False)
                
                # 手动触发曲线改变信号，通知主窗口更新预览
                self.curve_changed.emit(self.current_curve_name, self.curve_edit_widget.get_curve_points())
            finally:
                # 确保标志被重置
                self._loading_preset = False
    
    def _reset_to_linear(self):
        """重置为线性曲线"""
        linear_points = [(0.0, 0.0), (1.0, 1.0)]
        # 重置所有通道为线性曲线
        all_curves = {
            'RGB': linear_points,
            'R': linear_points,
            'G': linear_points,
            'B': linear_points
        }
        self.curve_edit_widget.set_all_curves(all_curves)
        
        # 清理修改状态
        if self.is_modified:
            self._remove_modified_option()
        
        # 动态添加custom选项并选中
        self.curve_combo.blockSignals(True)
        # 检查是否已经有custom选项
        custom_index = self.curve_combo.findData("custom")
        if custom_index == -1:
            # 没有custom选项，添加一个
            self.curve_combo.addItem("custom", "custom")
            custom_index = self.curve_combo.count() - 1
        
        self.curve_combo.setCurrentIndex(custom_index)
        self.curve_combo.blockSignals(False)
        
        # 清除所有状态信息
        self.current_curve_name = "custom"
        self.original_curve_name = None
        self.original_curve_key = None
        self.is_modified = False
        self.modified_item_index = -1
    
    def _save_curve(self):
        """保存当前曲线到文件"""
        # 获取曲线名称
        # 如果current_curve_name带星号，取原始名称作为默认值
        default_name = ""
        if self.current_curve_name and self.current_curve_name != "custom":
            if self.current_curve_name.startswith('*'):
                default_name = self.current_curve_name[1:]  # 去掉星号
            else:
                default_name = self.current_curve_name
        
        name, ok = QInputDialog.getText(
            self, 
            "保存曲线", 
            "请输入曲线名称:",
            text=default_name
        )
        
        if not ok or not name:
            return
        
        # 获取描述
        description, ok = QInputDialog.getText(
            self, 
            "保存曲线", 
            "请输入曲线描述（可选）:"
        )
        
        if not ok:
            return
        
        # 准备数据
        all_curves = self.curve_edit_widget.get_all_curves()
        curve_data = {
            "name": name,
            "description": description,
            "version": 2,
            "curves": all_curves
        }
        
        # 生成文件名（去除特殊字符）
        safe_filename = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_filename = safe_filename.replace(' ', '_')
        
        # 打开文件保存对话框
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "保存曲线文件",
            f"config/curves/{safe_filename}.json",
            "JSON文件 (*.json)"
        )
        
        if file_path:
            try:
                # 确保文件有.json扩展名
                if not file_path.endswith('.json'):
                    file_path += '.json'
                
                # 保存文件
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(curve_data, f, indent=2, ensure_ascii=False)
                
                QMessageBox.information(self, "成功", f"曲线已保存到：\n{file_path}")
                
                # 重新加载曲线列表并自动应用刚保存的曲线
                self._load_preset_curves()
                self._refresh_curve_combo()
                # 自动选择并应用刚保存的曲线
                self._apply_saved_curve(safe_filename)
                
            except Exception as e:
                QMessageBox.critical(self, "错误", f"保存曲线时出错：\n{str(e)}")
    
    def reload_curves_config(self):
        """重新加载curves配置（响应配置文件变化）"""
        # 重新加载preset curves
        self._load_preset_curves()
        # 刷新UI
        self._refresh_curve_combo()

    def _refresh_curve_combo(self):
        """刷新曲线下拉列表"""
        # 保存当前状态
        current_data = self.curve_combo.currentData()
        was_modified = self.is_modified
        
        # 清空并重新填充，不添加固定的custom选项
        self.curve_combo.clear()
        for curve_key, curve_data in sorted(self.preset_curves.items(), key=lambda x: x[1]["name"]):
            self.curve_combo.addItem(curve_data["name"], curve_key)
        
        # 恢复修改状态选项
        if was_modified and self.original_curve_name:
            modified_name = f"*{self.original_curve_name}"
            modified_data = modified_name  # 保持一致性
            self.curve_combo.insertItem(0, modified_name, modified_data)
            self.curve_combo.setCurrentIndex(0)
            self.modified_item_index = 0
        else:
            # 恢复普通选择
            index = self.curve_combo.findData(current_data)
            if index >= 0:
                self.curve_combo.setCurrentIndex(index)
            elif current_data == "custom":
                # 如果之前选择的是custom，重新添加custom选项
                self.curve_combo.addItem("custom", "custom")
                self.curve_combo.setCurrentIndex(self.curve_combo.count() - 1)

    def _apply_saved_curve(self, curve_key: str):
        """自动应用刚保存的曲线"""
        try:
            # 在下拉列表中找到对应的曲线
            index = self.curve_combo.findData(curve_key)
            if index >= 0:
                self.curve_combo.setCurrentIndex(index)
                # 触发曲线应用
                self._on_preset_curve_selected()
        except Exception as e:
            print(f"自动应用保存的曲线失败: {e}")
    
    def _on_curve_changed(self, points):
        """曲线改变时的处理"""
        # 如果正在加载预设，不要触发修改状态
        if self._loading_preset:
            return
            
        # 如果有原始曲线，添加修改状态选项
        if self.original_curve_name and self.original_curve_key and not self.is_modified:
            self._add_modified_option()
        elif not self.original_curve_name:
            # 没有原始曲线信息（例如已经是custom状态）
            self.current_curve_name = "custom"

        # 发出曲线改变信号，包含名称和点
        self.curve_changed.emit(self.current_curve_name, points)
    
    def _add_modified_option(self):
        """添加修改状态选项"""
        if not self.original_curve_name or self.is_modified:
            return
        
        # 生成修改后的名称，UI显示和数据保持一致
        modified_name = f"*{self.original_curve_name}"
        modified_data = modified_name  # 保持一致性，而不是 f"modified_{self.original_curve_key}"
        
        # 在下拉框顶部添加修改状态选项
        self.curve_combo.blockSignals(True)
        self.curve_combo.insertItem(0, modified_name, modified_data)
        self.curve_combo.setCurrentIndex(0)
        self.curve_combo.blockSignals(False)
        
        # 更新状态
        self.is_modified = True
        self.modified_item_index = 0
        self.current_curve_name = modified_name
    
    def _remove_modified_option(self):
        """移除修改状态选项"""
        if not self.is_modified or self.modified_item_index == -1:
            return
        
        self.curve_combo.blockSignals(True)
        self.curve_combo.removeItem(self.modified_item_index)
        self.curve_combo.blockSignals(False)
        
        # 重置状态
        self.is_modified = False
        self.modified_item_index = -1
    
    def cleanup_modified_curve_items(self):
        """清理所有修改状态的曲线项目（用于图片切换时重置状态）"""
        # 收集所有以*开头的项目索引
        items_to_remove = []
        for i in range(self.curve_combo.count()):
            item_data = self.curve_combo.itemData(i)
            item_text = self.curve_combo.itemText(i)
            if (isinstance(item_data, str) and item_data.startswith('*')) or item_text.startswith('*'):
                items_to_remove.append(i)
        
        # 倒序删除，避免索引变化
        self.curve_combo.blockSignals(True)
        for i in reversed(items_to_remove):
            self.curve_combo.removeItem(i)
        self.curve_combo.blockSignals(False)
        
        # 重置修改状态
        self.is_modified = False
        self.modified_item_index = -1
        self.original_curve_name = None
        self.original_curve_key = None
    
    def _load_original_curve_as_modified(self, original_name: str):
        """加载原始曲线数据但保持修改状态显示"""
        # 查找原始曲线数据
        for curve_key, curve_data in self.preset_curves.items():
            if curve_data["name"] == original_name:
                # 更新状态信息
                self.original_curve_name = original_name
                self.original_curve_key = curve_key
                self.current_curve_name = f"*{original_name}"
                
                # 注意：不加载曲线数据，保持当前修改后的曲线
                # 这是用户从修改状态切换回修改状态，应该保持当前的修改
                break
    
    def set_curve(self, points: List[Tuple[float, float]]):
        """设置曲线"""
        # 数据验证和保护
        if not isinstance(points, list) or not all(isinstance(p, tuple) and len(p) == 2 for p in points):
            print(f"警告: set_curve 收到无效的数据格式: {points}，将重置为线性。")
            points = [(0.0, 0.0), (1.0, 1.0)]
        
        self.curve_edit_widget.set_curve_points(points)
    
    def get_curve_points(self) -> List[Tuple[float, float]]:
        """获取曲线控制点"""
        return self.curve_edit_widget.get_curve_points()
    
    def get_all_curves(self) -> Dict[str, List[Tuple[float, float]]]:
        """获取所有通道的曲线"""
        return self.curve_edit_widget.get_all_curves()
    
    def set_all_curves(self, curves: Dict[str, List[Tuple[float, float]]]):
        """设置所有通道的曲线"""
        for channel, points in curves.items():
            self.curve_edit_widget.set_curve_points(points, channel)