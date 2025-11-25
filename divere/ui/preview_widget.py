"""
预览组件
用于显示图像预览
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from enum import Enum

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea, QHBoxLayout, QPushButton, QCheckBox
from PySide6.QtWidgets import QFrame, QApplication, QComboBox
from PySide6.QtCore import Qt, QTimer, Signal, QPoint, QPointF, QRect, QEvent
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QKeySequence, QCursor, QPolygonF, QAction

from divere.core.data_types import ImageData, CropInstance, CropAddDirection
from divere.i18n import tr


class CropEditMode(Enum):
    """裁剪框编辑模式"""
    NONE = 0
    DRAG_TOP = 1
    DRAG_BOTTOM = 2
    DRAG_LEFT = 3
    DRAG_RIGHT = 4
    DRAG_TOP_LEFT = 5
    DRAG_TOP_RIGHT = 6
    DRAG_BOTTOM_LEFT = 7
    DRAG_BOTTOM_RIGHT = 8


class PreviewCanvas(QLabel):
    """自绘制画布：基于 pan/zoom 稳定绘制，不改变自身尺寸"""
    def __init__(self):
        super().__init__()
        self._source_pixmap: Optional[QPixmap] = None
        self._zoom: float = 1.0
        self._pan_x: float = 0.0
        self._pan_y: float = 0.0
        # 简单缩放缓存以提升性能
        self._scaled_pixmap: Optional[QPixmap] = None
        self._scaled_zoom: float = 1.0
        # 叠加绘制回调
        self.overlay_drawer = None

    def set_source_pixmap(self, pixmap: QPixmap) -> None:
        """设置新的源 pixmap 并释放旧的资源

        执行关键的内存管理工作：
        1. 显式释放旧的 pixmap 对象（source 和 scaled）
        2. 设置新的 pixmap
        3. 清理缓存状态
        4. 触发重绘

        修复说明：
        - 使用 `= None` 而不是 `del` 来释放引用，更清晰地表达意图
        - `del` 只删除变量名，不保证对象被释放
        - `= None` 释放引用后，如果没有其他引用，对象会被 GC 回收
        - 对于 Qt 对象，释放引用后会被 Qt 的内存管理系统处理
        """
        # 释放旧的 source pixmap 引用
        # 注意：如果 paintEvent 正在使用这个 pixmap，Qt 会保持引用直到绘制完成
        if self._source_pixmap is not None:
            self._source_pixmap = None

        # 释放旧的 scaled pixmap 缓存
        if self._scaled_pixmap is not None:
            self._scaled_pixmap = None

        # 设置新的 source pixmap
        self._source_pixmap = pixmap

        # 清空文本避免覆盖图像
        self.setText("")

        # 源变更需重建缩放缓存
        self._scaled_pixmap = None
        self._scaled_zoom = 1.0

        # 触发重绘，显示新的 pixmap
        self.update()

    def set_view(self, zoom: float, pan_x: float, pan_y: float) -> None:
        self._zoom = float(zoom)
        self._pan_x = float(pan_x)
        self._pan_y = float(pan_y)
        self.update()

    def _ensure_scaled_cache(self) -> None:
        if self._source_pixmap is None:
            self._scaled_pixmap = None
            return
        # 仅在缩放不为1时缓存
        if abs(self._zoom - 1.0) < 1e-6:
            self._scaled_pixmap = None
            self._scaled_zoom = 1.0
            return
        # 使用不取整的绘制缩放路径，暂不生成缓存，避免滚轮缩放时跳动
        self._scaled_pixmap = None
        return
        # 超大尺寸时放弃缓存，避免内存/卡顿（直接使用绘制时缩放）
        if target_w <= 0 or target_h <= 0:
            self._scaled_pixmap = None
            return
        if target_w > 4096 or target_h > 4096 or target_w * target_h > 16_000_000:
            self._scaled_pixmap = None
            return
        if self._scaled_pixmap is None or self._scaled_zoom != self._zoom:
            self._scaled_pixmap = self._source_pixmap.scaled(
                target_w, target_h,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self._scaled_zoom = self._zoom

    def paintEvent(self, event):
        # 先让 QLabel 按样式绘制背景
        super().paintEvent(event)
        if self._source_pixmap is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.translate(QPointF(self._pan_x, self._pan_y))
        # 为避免滚轮缩放时因缓存缩放尺寸取整导致的锚点漂移，这里统一走绘制时缩放路径
        painter.scale(self._zoom, self._zoom)
        painter.drawPixmap(0, 0, self._source_pixmap)
        if self.overlay_drawer is not None:
            try:
                self.overlay_drawer(painter)
            except Exception as e:
                print(f"overlay绘制失败: {e}")
        painter.end()


class PreviewWidget(QWidget):
    """图像预览组件"""
    
    # 发送图像旋转信号，参数为旋转方向：1=左旋，-1=右旋
    image_rotated = Signal(int)
    
    # 裁剪相关信号
    crop_committed = Signal(tuple)  # 新建裁剪
    single_crop_committed = Signal(tuple)  # 单张裁剪（不创建正式crop项）
    crop_updated = Signal(str, tuple)    # 更新现有裁剪 (crop_id, rect_norm)
    request_focus_crop = Signal(str)  # 请求聚焦某个裁剪
    request_restore_crop = Signal()  # 返回原图模式
    request_smart_add_crop = Signal()  # 智能添加裁剪
    request_focus_contactsheet = Signal()  # 单张/接触印相的裁剪聚焦

    # 中性色定义相关信号
    neutral_point_selected = Signal(float, float)  # (norm_x, norm_y) 选中的中性点归一化坐标

    def __init__(self, context=None):
        super().__init__()
        
        self.context = context  # ApplicationContext for checking film type
        self.current_image: Optional[ImageData] = None
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        
        # 拖动相关状态
        self.dragging = False
        self.last_mouse_pos = None
        self.drag_start_pos = None
        self.original_pan_pos = None
        
        # 平滑拖动相关
        self.smooth_drag_timer = QTimer()
        self.smooth_drag_timer.timeout.connect(self._smooth_drag_update)
        self.smooth_drag_timer.setInterval(16)  # ~60fps
        
        self._create_ui()
        self._create_crop_selector()
        self._setup_mouse_controls()
        self._setup_keyboard_controls()
        self._setup_navigation_controls()
        # 捕获应用级别的修饰键变化（确保光标能及时更新）
        try:
            QApplication.instance().installEventFilter(self)
        except Exception:
            pass

        # 视图边界（避免留白）的开关（按需限制）。
        # 根据需求改为默认关闭：允许自由拖放，不做边界约束。
        self._enable_pan_clamp = False

        # 缩放上下限
        self._min_zoom: float = 0.05
        self._max_zoom: float = 16.0

        # 旋转锚点相关代码已移除，简化旋转逻辑

        # 色卡选择器状态
        self.cc_enabled: bool = False
        self.cc_corners_norm = None  # [(x_norm, y_norm) * 4] 原图归一化坐标 [0,1]
        self.cc_drag_idx: int | None = None
        self.cc_handle_radius_disp: float = 8.0
        self.cc_ref_qcolors = None  # 24项，参考色（QColor）

        # 中性色定义状态
        self.neutral_point_selection_mode: bool = False  # 是否处于中性点选择模式
        self.neutral_point_norm: Optional[Tuple[float, float]] = None  # 选定的中性点归一化坐标 [0,1]

        # 裁剪交互与显示
        self._crop_mode: bool = False  # 是否在框选新裁剪模式
        self._crop_dragging: bool = False
        self._crop_start_norm = None  # (x,y) 原图归一化坐标
        self._crop_current_norm = None  # (x,y) 原图归一化坐标
        self._crop_overlay_norm = None  # (x,y,w,h) 当前活跃裁剪框
        
        # 多裁剪显示
        self._all_crops: List[Dict] = []  # 所有裁剪 [{id, rect_norm, active}, ...]
        self._show_all_crops: bool = True  # 是否显示所有裁剪框
        self._hovering_crop_id: Optional[str] = None  # 悬停的裁剪 ID
        self._dragging_crop_id: Optional[str] = None  # 正在拖动的裁剪 ID
        self._drag_offset: Tuple[float, float] = (0, 0)  # 拖动偏移
        self._double_click_timer = QTimer()
        self._double_click_timer.setSingleShot(True)
        self._double_click_timer.setInterval(300)  # 300ms 双击间隔
        self._pending_click_crop_id: Optional[str] = None
        
        # 裁剪框编辑状态
        self._crop_edit_mode: CropEditMode = CropEditMode.NONE
        self._crop_edit_start_rect: Optional[Tuple[float, float, float, float]] = None
        self._crop_edit_hover_mode: CropEditMode = CropEditMode.NONE
        
        # marching ants 动画
        self._ants_phase: float = 0.0
        self._ants_timer = QTimer()
        self._ants_timer.setInterval(100)
        self._ants_timer.timeout.connect(self._advance_ants)

        # 编辑热键状态（Cmd/Ctrl）
        self._edit_modifier_down: bool = False

        # 多裁剪编辑：当前正在编辑的裁剪ID（用于边/角编辑）
        self._active_edit_crop_id: Optional[str] = None
        # 新建裁剪意图：'add' 来自加号按钮；'single' 来自 Cmd/Ctrl 直接框选
        self._crop_creation_intent: Optional[str] = None
        # UI：当仅存在"单张裁剪"时隐藏下方 [1] 按钮（未按加号）
        self._hide_single_crop_selector: bool = False

        # 聚焦模式：整体移动当前裁剪框
        self._moving_overlay_active: bool = False
        self._moving_overlay_offset_norm: Tuple[float, float] = (0.0, 0.0)
        
        # 屏幕反光补偿：black cut-off检测与显示
        self._show_black_cutoff: bool = False
        self._cutoff_compensation: float = 0.0
        self._cutoff_pixels: Optional[np.ndarray] = None  # 布尔mask，标记cut-off像素

    # ============ 内部工具 ============
    def _get_viewport_size(self):
        """获取可视区域尺寸（viewport 尺寸）"""
        if hasattr(self, 'scroll_area') and self.scroll_area is not None:
            return self.scroll_area.viewport().size()
        return self.size()

    def _get_scaled_image_size(self):
        """返回当前缩放下图像尺寸 (Wi, Hi)"""
        if not self.current_image or self.current_image.array is None:
            return 0, 0
        h, w = self.current_image.array.shape[:2]
        Wi = int(round(w * float(self.zoom_factor)))
        Hi = int(round(h * float(self.zoom_factor)))
        return Wi, Hi

    def _clamp_pan(self):
        """根据 viewport 与缩放后图像尺寸对 pan 进行边界约束"""
        if not self._enable_pan_clamp:
            return
        Wv = self._get_viewport_size().width()
        Hv = self._get_viewport_size().height()
        Wi, Hi = self._get_scaled_image_size()

        # 当图像大于视口：允许范围 [Wv - Wi, 0]
        # 当图像小于视口：允许范围 [0, Wv - Wi]，以便可实现居中
        if Wi >= Wv:
            min_tx, max_tx = Wv - Wi, 0
        else:
            min_tx, max_tx = 0, Wv - Wi
        if Hi >= Hv:
            min_ty, max_ty = Hv - Hi, 0
        else:
            min_ty, max_ty = 0, Hv - Hi

        if self.pan_x < min_tx:
            self.pan_x = int(min_tx)
        elif self.pan_x > max_tx:
            self.pan_x = int(max_tx)
        if self.pan_y < min_ty:
            self.pan_y = int(min_ty)
        elif self.pan_y > max_ty:
            self.pan_y = int(max_ty)
    
    def _create_ui(self):
        """创建用户界面"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 按钮组
        button_layout = QHBoxLayout()
        
        # 旋转按钮
        self.rotate_left_btn = QPushButton(tr("preview_widget.buttons.rotate_left"))
        self.rotate_right_btn = QPushButton(tr("preview_widget.buttons.rotate_right"))
        self.rotate_left_btn.setMaximumWidth(80)
        self.rotate_right_btn.setMaximumWidth(80)
        self.rotate_left_btn.clicked.connect(self.rotate_left)
        self.rotate_right_btn.clicked.connect(self.rotate_right)
        
        # 视图控制按钮
        self.fit_window_btn = QPushButton(tr("preview_widget.buttons.fit_window"))
        self.center_btn = QPushButton(tr("preview_widget.buttons.center"))
        self.fit_window_btn.setMaximumWidth(80)
        self.center_btn.setMaximumWidth(80)
        self.fit_window_btn.clicked.connect(self.fit_to_window)
        self.center_btn.clicked.connect(self.center_image)
        # 色卡选择器（默认隐藏，由参数面板联动显示/控制）
        self.cc_checkbox = QCheckBox(tr("preview_widget.buttons.colorchecker"))
        self.cc_checkbox.toggled.connect(self._on_cc_toggled)
        self.cc_checkbox.setVisible(False)
        # 裁剪按钮组（已移除，使用底部选择条）
        
        # 添加按钮到布局
        button_layout.addWidget(self.rotate_left_btn)
        button_layout.addWidget(self.rotate_right_btn)
        button_layout.addWidget(self.fit_window_btn)
        button_layout.addWidget(self.center_btn)
        button_layout.addWidget(self.cc_checkbox)
        button_layout.addStretch()  # 弹性空间
        
        # 文件夹导航控件
        self._create_navigation_controls(button_layout)
        layout.addLayout(button_layout)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        
        self.image_label = PreviewCanvas()
        self.image_label.setObjectName('imageCanvas')
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        self.image_label.setText(tr("preview_widget.labels.load_image"))
        
        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area)
        # 统一overlay绘制器
        self.image_label.overlay_drawer = self._draw_overlays

    # ====== 裁剪选择条 ======
    def _create_crop_selector(self):
        """在下方创建简单的裁剪选择条：[原图] [1] [2] ... [+]"""
        try:
            # 容器
            row = QHBoxLayout()
            row.setContentsMargins(8, 4, 8, 4)
            row.setSpacing(8)
            # 左侧：裁剪聚焦开关（默认隐藏）
            self.btn_focus_toggle = QPushButton(tr("preview_widget.buttons.crop_focus"))
            self.btn_focus_toggle.setCheckable(True)
            self.btn_focus_toggle.setVisible(False)
            self.btn_focus_toggle.setChecked(False)
            self.btn_focus_toggle.toggled.connect(self._on_focus_toggle_toggled)
            row.addWidget(self.btn_focus_toggle)

            # 添加拉伸以居中
            row.addStretch()
            
            # 原图按钮
            self.btn_contactsheet = QPushButton(tr("preview_widget.buttons.original"))
            self.btn_contactsheet.setMaximumWidth(56)
            self.btn_contactsheet.setCheckable(True)
            self.btn_contactsheet.setChecked(True)  # 默认选中原图
            self.btn_contactsheet.clicked.connect(lambda: self._emit_switch_profile('contactsheet', None))
            row.addWidget(self.btn_contactsheet)
            
            # 占位：动态添加crop按钮
            self._crop_buttons_container = QHBoxLayout()
            self._crop_buttons_container.setSpacing(6)
            row.addLayout(self._crop_buttons_container)
            
            # 添加拉伸将"+"按钮推到最右侧
            row.addStretch()
            
            # 新增按钮（固定在最右侧）
            self.btn_add_crop = QPushButton(tr("preview_widget.buttons.add_crop"))
            self.btn_add_crop.setMaximumWidth(32)
            self.btn_add_crop.setToolTip(tr("preview_widget.tooltips.add_crop"))
            self.btn_add_crop.clicked.connect(self._on_add_crop_clicked)
            row.addWidget(self.btn_add_crop)
            
            # 方向选择下拉框
            self.combo_direction = QComboBox()
            self.combo_direction.setMaximumWidth(100)
            self.combo_direction.setToolTip(tr("preview_widget.tooltips.select_direction"))
            
            # 添加8个方向选项
            directions = [
                ("↓→", CropAddDirection.DOWN_RIGHT, tr("preview_widget.crop_directions.down_right")),
                ("↓←", CropAddDirection.DOWN_LEFT, tr("preview_widget.crop_directions.down_left")),
                ("→↓", CropAddDirection.RIGHT_DOWN, tr("preview_widget.crop_directions.right_down")),
                ("→↑", CropAddDirection.RIGHT_UP, tr("preview_widget.crop_directions.right_up")),
                ("↑←", CropAddDirection.UP_LEFT, tr("preview_widget.crop_directions.up_left")),
                ("↑→", CropAddDirection.UP_RIGHT, tr("preview_widget.crop_directions.up_right")),
                ("←↑", CropAddDirection.LEFT_UP, tr("preview_widget.crop_directions.left_up")),
                ("←↓", CropAddDirection.LEFT_DOWN, tr("preview_widget.crop_directions.left_down"))
            ]
            
            for display_text, direction_value, tooltip in directions:
                self.combo_direction.addItem(display_text)
                # 设置每个项目的用户数据为对应的方向值
                self.combo_direction.setItemData(self.combo_direction.count() - 1, direction_value)
            
            # 默认选择第一个方向（DOWN_RIGHT）
            self.combo_direction.setCurrentIndex(0)
            row.addWidget(self.combo_direction)
            
            # 放到底部
            self.layout().addLayout(row)
        except Exception as e:
            print(f"创建裁剪选择条失败: {e}")
            
    def _get_selected_direction(self) -> CropAddDirection:
        """获取当前选择的添加方向"""
        try:
            current_index = self.combo_direction.currentIndex()
            direction = self.combo_direction.itemData(current_index)
            return direction if direction is not None else CropAddDirection.DOWN_RIGHT
        except AttributeError:
            # 如果combo_direction还没有初始化，返回默认方向
            return CropAddDirection.DOWN_RIGHT
            
    def _on_add_crop_clicked(self):
        """处理+按钮点击，使用选择的方向添加裁剪"""
        selected_direction = self._get_selected_direction()
        self._emit_request_new_crop(selected_direction)

    def refresh_crop_selector(self, crops: list, active_crop_id: str | None, is_focused: bool = False):
        """根据 Context 的裁剪列表刷新按钮。crops: list[CropInstance]"""
        try:
            # 清现有
            while self._crop_buttons_container.count():
                item = self._crop_buttons_container.takeAt(0)
                w = item.widget()
                if w:
                    w.setParent(None)
            
            # 文案：原图/接触印相
            has_crops = isinstance(crops, list) and len(crops) > 0
            try:
                self.btn_contactsheet.setText(tr("preview_widget.buttons.contactsheet") if has_crops else tr("preview_widget.buttons.original"))
            except Exception:
                pass

            # 是否隐藏"单裁剪"的编号按钮（仅一项，且由 Cmd/Ctrl 框选产生）
            hide_single = (not is_focused) and self._hide_single_crop_selector and isinstance(crops, list) and (len(crops) == 1)

            # 根据是否有活跃的crop来设置原图按钮状态（单裁剪隐藏时，原图按钮保持选中）
            if hide_single:
                self.btn_contactsheet.setChecked(True)
            else:
                self.btn_contactsheet.setChecked(active_crop_id is None and not is_focused)

            # 裁剪聚焦开关：仅在"原图/接触印相"模式（active_crop_id 为 None）显示
            try:
                show_focus_toggle = (active_crop_id is None)
                self.btn_focus_toggle.setVisible(bool(show_focus_toggle))
                # 同步选中态到是否处于聚焦
                self.btn_focus_toggle.blockSignals(True)
                self.btn_focus_toggle.setChecked(bool(is_focused))
                self.btn_focus_toggle.blockSignals(False)
                # 可用性：当有可聚焦的 overlay（本地或来自 Context）才可用
                enable_focus = False
                if show_focus_toggle:
                    if self._crop_overlay_norm is not None:
                        enable_focus = True
                    else:
                        try:
                            if self.current_image and self.current_image.metadata:
                                enable_focus = self.current_image.metadata.get('crop_overlay') is not None
                        except Exception:
                            enable_focus = False
                self.btn_focus_toggle.setEnabled(bool(enable_focus))
            except Exception:
                pass
            
            # 更新内部裁剪列表
            self._all_crops = []
            for idx, crop in enumerate(crops, start=1):
                cid = getattr(crop, 'id', f'crop_{idx}')
                rect_norm = getattr(crop, 'rect_norm', (0, 0, 1, 1))
                self._all_crops.append({
                    'id': cid,
                    'index': idx,
                    'rect_norm': rect_norm,
                    'active': cid == active_crop_id
                })
            
            # 添加按钮（当只有一个且要求隐藏时，不渲染编号按钮）
            if not hide_single:
                for idx, crop in enumerate(crops, start=1):
                    btn = QPushButton(str(idx))
                    btn.setMaximumWidth(32)
                    btn.setCheckable(True)
                    cid = getattr(crop, 'id', f'crop_{idx}')
                    btn.setChecked(active_crop_id == cid)
                    btn.setToolTip(getattr(crop, 'name', tr("preview_widget.tooltips.crop_name", idx=idx)))
                    btn.clicked.connect(lambda checked, c_id=cid: self._emit_switch_profile('crop', c_id))
                    self._crop_buttons_container.addWidget(btn)
            
            # 设置显示模式
            self._show_all_crops = not is_focused
            
            # 刷新显示
            self.image_label.update()
            
        except Exception as e:
            print(f"刷新裁剪选择条失败: {e}")

    def _on_focus_toggle_toggled(self, checked: bool):
        """左侧裁剪聚焦开关：仅在原图/接触印相模式出现。"""
        try:
            if checked:
                self.request_focus_contactsheet.emit()
            else:
                self.request_restore_crop.emit()
        except Exception:
            pass

    # 供 MainWindow 连接到 Context 的信号
    request_switch_profile = Signal(str, object)  # kind, crop_id
    request_new_crop = Signal(object)  # direction (CropAddDirection)
    request_delete_crop = Signal(str)  # 删除指定裁剪

    def _emit_switch_profile(self, kind: str, crop_id: object):
        # 进入切换时清空交互临时态
        self._clear_crop_hover_state()
        self._crop_mode = False
        self._crop_dragging = False
        self._crop_start_norm = None
        self._crop_current_norm = None
        self.clear_neutral_point()  # 切换crop/single时清除中性点标记
        self.request_switch_profile.emit(kind, crop_id)

    def _emit_request_new_crop(self, direction: CropAddDirection):
        # 若已存在至少一个正式裁剪，则交给主窗口执行"智能新增"，不进入手动框选
        try:
            if isinstance(self._all_crops, list) and len(self._all_crops) >= 1:
                self.request_new_crop.emit(direction)
                return
        except Exception:
            pass
        # 否则进入手动框选模式，让用户画第一张裁剪
        # 指示为"正式添加"意图
        self._crop_creation_intent = 'add'
        # 进入正式多裁剪流程，显示编号按钮
        self._hide_single_crop_selector = False
        self._toggle_crop_mode(True)

    # ===== 色卡选择器：UI与绘制 =====
    
    # ===== ColorChecker坐标转换方法 =====
    def _get_source_image_size(self) -> Optional[Tuple[int, int]]:
        """获取原图尺寸"""
        if not self.current_image or not self.current_image.metadata:
            return None
        source_wh = self.current_image.metadata.get('source_wh')
        if source_wh:
            return int(source_wh[0]), int(source_wh[1])
        return None

    def _norm_to_display_coords(self, norm_corners) -> Optional[List[Tuple[float, float]]]:
        """归一化坐标 → 显示像素坐标"""
        if not norm_corners or not self.current_image:
            return None
            
        source_size = self._get_source_image_size()
        if not source_size:
            return None
            
        src_w, src_h = source_size
        
        # 先转换为原图像素坐标
        source_corners = []
        for x_norm, y_norm in norm_corners:
            source_x = x_norm * src_w
            source_y = y_norm * src_h
            source_corners.append((source_x, source_y))
        
        # 再转换为显示坐标
        display_corners = []
        for source_x, source_y in source_corners:
            # 使用现有的转换逻辑（逆向）
            md = self.current_image.metadata or {}
            focused = md.get('crop_focused', False)
            disp_h, disp_w = self.current_image.array.shape[:2]
            
            if focused:
                # 聚焦模式：原图坐标 → crop坐标 → 显示坐标
                crop_instance = md.get('crop_instance')
                if crop_instance:
                    x, y, w, h = crop_instance.rect_norm
                    crop_left = x * src_w
                    crop_top = y * src_h
                    crop_width = w * src_w
                    crop_height = h * src_h
                    
                    # 原图坐标 → crop坐标
                    # 修复：缩放时需要考虑crop的orientation
                    if crop_instance.orientation % 180 == 0:  # 0°或180°
                        crop_x = (source_x - crop_left) * disp_w / crop_width
                        crop_y = (source_y - crop_top) * disp_h / crop_height
                    else:  # 90°或270°（宽高交换）
                        crop_x = (source_x - crop_left) * disp_h / crop_width
                        crop_y = (source_y - crop_top) * disp_w / crop_height

                    # 处理crop的独立orientation
                    if crop_instance.orientation % 360 != 0:
                        # 修复：传入未旋转的crop尺寸
                        # 缩放后坐标在未旋转的crop坐标系中
                        # 对于90°/270°，未旋转尺寸是(disp_h, disp_w)
                        if crop_instance.orientation % 180 == 0:
                            rot_crop_w, rot_crop_h = disp_w, disp_h
                        else:
                            rot_crop_w, rot_crop_h = disp_h, disp_w

                        crop_x, crop_y = self._apply_rotate_point(
                            crop_x, crop_y, rot_crop_w, rot_crop_h, crop_instance.orientation
                        )
                    
                    display_corners.append((crop_x, crop_y))
                else:
                    display_corners.append((source_x, source_y))
            else:
                # 非聚焦模式：原图坐标 → 显示坐标
                global_orientation = md.get('global_orientation', 0)

                if global_orientation % 180 == 0:  # 0°或180°
                    img_x = source_x * disp_w / src_w
                    img_y = source_y * disp_h / src_h
                else:  # 90°或270°（宽高交换）
                    img_x = source_x * disp_h / src_w
                    img_y = source_y * disp_w / src_h

                if global_orientation % 360 != 0:
                    # 修复：传入未旋转的图像尺寸作为输入坐标系
                    # 对于90°/270°，缩放后坐标在(disp_h, disp_w)坐标系中
                    if global_orientation % 180 == 0:
                        rot_img_w, rot_img_h = disp_w, disp_h
                    else:
                        rot_img_w, rot_img_h = disp_h, disp_w

                    img_x, img_y = self._apply_rotate_point(
                        img_x, img_y, rot_img_w, rot_img_h, global_orientation
                    )

                display_corners.append((img_x, img_y))
        
        return display_corners

    def _display_to_norm_coords(self, display_point: QPoint) -> Optional[Tuple[float, float]]:
        """显示像素坐标 → 归一化坐标"""
        orig_point = self._display_to_original_point(display_point)
        if not orig_point:
            return None
            
        source_size = self._get_source_image_size()
        if not source_size:
            return None
            
        src_w, src_h = source_size
        norm_x = orig_point[0] / src_w
        norm_y = orig_point[1] / src_h
        return (norm_x, norm_y)

    def _norm_to_source_coords(self, norm_corners) -> Optional[List[Tuple[float, float]]]:
        """归一化坐标 → 原图像素坐标"""
        if not norm_corners:
            return None
            
        source_size = self._get_source_image_size()
        if not source_size:
            return None
            
        src_w, src_h = source_size
        source_corners = []
        for x_norm, y_norm in norm_corners:
            source_x = x_norm * src_w
            source_y = y_norm * src_h
            source_corners.append((source_x, source_y))
        
        return source_corners

    def _on_cc_toggled(self, checked: bool):
        self.cc_enabled = bool(checked)
        if self.cc_enabled and self.current_image and self.current_image.array is not None:
            if not self.cc_corners_norm:
                self._init_default_colorchecker()
            self._ensure_cc_reference_colors()
        
        # 切换色卡模式时清理悬停状态
        if self.cc_enabled:
            self._clear_crop_hover_state()
        
        # overlay_drawer 始终使用统一的 _draw_overlays
        self._update_display()

    def _init_default_colorchecker(self):
        # cc_corners_norm 存储原图归一化坐标 [0,1]
        # 默认在原图中心60%区域放置色卡选择器
        margin = 0.2  # 距离边缘20%
        # 色卡比例为1.5:1 (6:4)，在可用区域内按比例调整
        available_width = 1.0 - 2 * margin
        available_height = 1.0 - 2 * margin
        
        # 按色卡比例计算实际尺寸
        target_width = min(available_width, available_height * 1.5)
        target_height = target_width / 1.5
        
        # 居中放置
        center_x = 0.5
        center_y = 0.5
        x0 = center_x - target_width / 2.0
        y0 = center_y - target_height / 2.0
        x1 = center_x + target_width / 2.0
        y1 = center_y + target_height / 2.0
        
        # 存储归一化坐标：左上、右上、右下、左下
        self.cc_corners_norm = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    def _draw_colorchecker_overlay(self, painter: QPainter):
        """绘制色卡选择器的UI覆盖层：边框、角点、网格等"""
        if not (self.cc_enabled and self.cc_corners_norm):
            return
        
        # 转换归一化坐标到显示坐标
        display_corners = self._norm_to_display_coords(self.cc_corners_norm)
        if not display_corners:
            return
        
        # 绘制边框和角点
        pts = [QPointF(*p) for p in display_corners]
        painter.setPen(QPen(QColor(255, 255, 0, 220), 2))
        painter.setBrush(Qt.NoBrush)
        
        # 绘制边框线条
        for i in range(4):
            painter.drawLine(pts[i], pts[(i+1)%4])
        
        # 绘制角点标记
        painter.setBrush(QColor(255, 255, 0, 200))
        for p in pts:
            painter.drawEllipse(p, 5, 5)
        
        # 绘制4x6色块网格（内嵌参考色块）
        self._draw_colorchecker_grid(painter)
        
        # 可选：绘制参考色块
        if self.cc_ref_qcolors:
            self._draw_reference_colors(painter)

    def _draw_colorchecker_grid(self, painter: QPainter):
        """绘制4x6色块网格，内嵌参考色块"""
        if not self.cc_corners_norm or not self.current_image or not self.current_image.array is not None:
            return
        
        try:
            import numpy as np
            import cv2
            
            # 转换归一化坐标到显示坐标
            display_corners = self._norm_to_display_coords(self.cc_corners_norm)
            if not display_corners:
                return
            
            # 计算透视变换矩阵
            src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
            dst = np.array(display_corners, dtype=np.float32)
            H = cv2.getPerspectiveTransform(src, dst)
            
            # 网格参数
            margin = 0.18  # 色块边距
            grid_color = QColor(255, 255, 0, 150)  # 半透明黄色，稍微透明以便看到参考色块
            painter.setPen(QPen(grid_color, 1))
            painter.setBrush(Qt.NoBrush)
            
            # 绘制4x6网格，内嵌参考色块
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
                    
                    # 转换为QPointF
                    qpoly = [QPointF(float(x), float(y)) for x, y in poly]
                    
                    # 先绘制参考色块（如果可用）
                    if self.cc_ref_qcolors:
                        idx = r * 6 + c
                        if idx < len(self.cc_ref_qcolors):
                            ref_color = self.cc_ref_qcolors[idx]
                            # 使用不透明的参考色块，比网格小一点
                            painter.setBrush(QColor(ref_color.red(), ref_color.green(), ref_color.blue(), 255))
                            # 缩小参考色块，留出网格边距
                            inner_margin = 0.25  # 参考色块比网格小25%
                            inner_sx0 = sx0 + inner_margin * (sx1 - sx0)
                            inner_sx1 = sx1 - inner_margin * (sx1 - sx0)
                            inner_sy0 = sy0 + inner_margin * (sy1 - sy0)
                            inner_sy1 = sy1 - inner_margin * (sy1 - sy0)
                            
                            # 计算内缩后的多边形
                            inner_rect = np.array([[inner_sx0, inner_sy0], [inner_sx1, inner_sy0], [inner_sx1, inner_sy1], [inner_sx0, inner_sy1]], dtype=np.float32)
                            inner_rect_h = np.hstack([inner_rect, np.ones((4, 1), dtype=np.float32)])
                            inner_poly = (H @ inner_rect_h.T).T
                            inner_poly = inner_poly[:, :2] / inner_poly[:, 2:3]
                            
                            # 绘制内缩的参考色块（无边框）
                            inner_qpoly = [QPointF(float(x), float(y)) for x, y in inner_poly]
                            painter.setPen(Qt.NoPen)  # 移除边框
                            painter.drawPolygon(inner_qpoly)
                    
                    # 再绘制网格线（确保在参考色块之上）
                    painter.setBrush(Qt.NoBrush)
                    painter.setPen(QPen(grid_color, 1))  # 恢复网格线绘制
                    painter.drawPolygon(qpoly)
                        
        except Exception as e:
            print(f"绘制色卡网格失败: {e}")
            pass



    def _ensure_cc_reference_colors(self):
        """生成24个参考颜色（QColor，按A1..D6），从当前选择的色卡文件读取"""
        try:
            # 获取当前选择的色卡文件
            from divere.ui.main_window import MainWindow
            main_window = self.window()
            if hasattr(main_window, 'parameter_panel'):
                colorchecker_file = main_window.parameter_panel.get_selected_colorchecker_file()
            else:
                colorchecker_file = "original_color_cc24data.json"
            
            # 读取色卡JSON文件
            self.cc_ref_qcolors = self._load_colorchecker_colors(colorchecker_file)
            
        except Exception as e:
            print(f"生成参考颜色失败: {e}")
            # 降级到默认色卡
            try:
                from divere.core.color_science import colorchecker_display_p3_qcolors
                self.cc_ref_qcolors = colorchecker_display_p3_qcolors()
            except Exception:
                self.cc_ref_qcolors = None

    def _load_colorchecker_colors(self, filename: str):
        """
        使用ApplicationContext共享的reference color数据，确保与优化器数据一致
        避免重复加载和XYZ转换，直接复用已缓存的工作空间RGB值
        """
        try:
            from divere.utils.colorchecker_loader import WorkspaceValidationError
            
            # 从ApplicationContext获取已转换的reference colors
            if self.context and hasattr(self.context, 'get_reference_colors'):
                reference_data = self.context.get_reference_colors(filename)
                working_colorspace = self.context.color_space_manager.get_current_working_space()
            else:
                # fallback到旧方法
                return self._load_colorchecker_colors_legacy(filename)
            
            # 提取24个色块的RGB值（按A1..D6顺序）
            patch_order = [
                'A1', 'A2', 'A3', 'A4', 'A5', 'A6',
                'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 
                'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
                'D1', 'D2', 'D3', 'D4', 'D5', 'D6'
            ]
            
            rgb_values = []
            for patch_id in patch_order:
                if patch_id in reference_data:
                    rgb_values.append(reference_data[patch_id])
                else:
                    # 如果某个色块缺失，使用默认颜色
                    rgb_values.append([0.5, 0.5, 0.5])
            
            # 数据已经在working colorspace中，直接转换到显示空间
            return self._convert_colorchecker_to_display_colors(rgb_values, working_colorspace)
            
        except WorkspaceValidationError as e:
            # 专门处理工作空间验证错误，弹出对话框并回滚
            return self._handle_workspace_validation_error(e, filename)
            
        except Exception as e:
            print(f"加载色卡颜色失败: {e}")
            # 如果新方法失败，尝试fallback到旧方法
            return self._load_colorchecker_colors_legacy(filename)

    def _convert_colorchecker_to_display_colors(self, rgb_values: list, source_color_space: str) -> list:
        """使用现有的色彩空间管理器转换色卡颜色到显示空间"""
        try:
            from divere.core.data_types import ImageData
            
            # 将RGB值转换为numpy数组 (24, 3)
            rgb_array = np.array(rgb_values, dtype=np.float32)
            
            # 创建虚拟的ImageData对象，模拟24x1像素的图像
            # 重塑为 (24, 1, 3) 以便处理
            rgb_array_reshaped = rgb_array.reshape(24, 1, 3)
            
            # 使用传入的source_color_space作为输入色彩空间
            # 这样确保从正确的working colorspace转换到显示空间
            input_space = source_color_space
            
            # 创建虚拟ImageData对象
            virtual_image = ImageData(
                array=rgb_array_reshaped,
                width=1,
                height=24,
                channels=3,
                dtype=np.float32,
                color_space=input_space,
                metadata={},
                file_path=""
            )
            
            # 使用现有的色彩空间管理器转换到DisplayP3
            from divere.ui.main_window import MainWindow
            main_window = self.window()
            if hasattr(main_window, 'context') and hasattr(main_window.context, 'color_space_manager'):
                color_space_manager = main_window.context.color_space_manager
                converted_image = color_space_manager.convert_to_display_space(virtual_image, "DisplayP3")
                
                # 提取转换后的RGB值并转换为QColor
                converted_rgb = converted_image.array.reshape(24, 3)
                colors = []
                for rgb in converted_rgb:
                    r = int(np.clip(rgb[0], 0, 1) * 255 + 0.5)
                    g = int(np.clip(rgb[1], 0, 1) * 255 + 0.5)
                    b = int(np.clip(rgb[2], 0, 1) * 255 + 0.5)
                    colors.append(QColor(r, g, b))
                
                return colors
            else:
                # 如果无法获取色彩空间管理器，降级到简单处理
                return self._simple_rgb_to_qcolor(rgb_values)
                
        except Exception as e:
            print(f"色彩空间转换失败: {e}")
            # 降级到简单处理
            return self._simple_rgb_to_qcolor(rgb_values)

    def _simple_rgb_to_qcolor(self, rgb_values: list) -> list:
        """简单的RGB到QColor转换（降级方案）"""
        colors = []
        for rgb in rgb_values:
            r = int(np.clip(rgb[0], 0, 1) * 255 + 0.5)
            g = int(np.clip(rgb[1], 0, 1) * 255 + 0.5)
            b = int(np.clip(rgb[2], 0, 1) * 255 + 0.5)
            colors.append(QColor(r, g, b))
        return colors

    def _load_colorchecker_colors_legacy(self, filename: str):
        """旧版色卡加载方法（fallback），使用独立的colorchecker_loader调用"""
        try:
            from divere.utils.colorchecker_loader import load_colorchecker_reference, WorkspaceValidationError
            from divere.ui.main_window import MainWindow
            
            # 获取ColorSpaceManager和工作空间信息
            main_window = self.window()
            if hasattr(main_window, 'context'):
                color_space_manager = main_window.context.color_space_manager
                working_colorspace = color_space_manager.get_current_working_space()
            else:
                color_space_manager = None
                working_colorspace = 'ACEScg'  # fallback
            
            # 独立调用colorchecker_loader（不使用缓存）
            reference_data = load_colorchecker_reference(
                filename, 
                working_colorspace,
                color_space_manager
            )
            
            # 提取24个色块的RGB值（按A1..D6顺序）
            patch_order = [
                'A1', 'A2', 'A3', 'A4', 'A5', 'A6',
                'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 
                'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
                'D1', 'D2', 'D3', 'D4', 'D5', 'D6'
            ]
            
            rgb_values = []
            for patch_id in patch_order:
                if patch_id in reference_data:
                    rgb_values.append(reference_data[patch_id])
                else:
                    # 如果某个色块缺失，使用默认颜色
                    rgb_values.append([0.5, 0.5, 0.5])
            
            # 数据是在working colorspace中，转换到显示空间
            return self._convert_colorchecker_to_display_colors(rgb_values, working_colorspace)
            
        except Exception as e:
            print(f"Legacy加载色卡颜色失败: {e}")
            # 最终fallback：使用简单的RGB转换
            return self._simple_rgb_to_qcolor([[0.5, 0.5, 0.5]] * 24)

    def _handle_workspace_validation_error(self, error: Exception, filename: str):
        """
        处理工作空间验证错误，显示对话框并回滚到之前的选择
        """
        from PySide6.QtWidgets import QMessageBox
        from divere.ui.main_window import MainWindow
        
        # 显示错误对话框
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(tr("preview_widget.dialogs.workspace_incompatible_title"))
        msg_box.setIcon(QMessageBox.Icon.Warning)
        msg_box.setText(tr("preview_widget.dialogs.workspace_incompatible_message", filename=filename))
        msg_box.setDetailedText(str(error))
        msg_box.setInformativeText(
            "EnduraDensityExp类型的色卡需要KodakEnduraPremier工作空间。\n"
            + tr("preview_widget.dialogs.workspace_incompatible_detail")
        )
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        
        # 显示对话框
        msg_box.exec()
        
        # 回滚到上一个有效的色卡选择
        try:
            main_window = self.window()
            if hasattr(main_window, 'parameter_panel'):
                # 触发参数面板回滚到默认或上一个有效选择
                main_window.parameter_panel.revert_colorchecker_selection()
        except Exception as e:
            print(f"回滚色卡选择失败: {e}")
        
        # 返回默认色卡显示
        try:
            from divere.core.color_science import colorchecker_display_p3_qcolors
            return colorchecker_display_p3_qcolors()
        except Exception:
            return self._simple_rgb_to_qcolor([[0.5, 0.5, 0.5]] * 24)

    def on_colorchecker_changed(self, filename: str):
        """当色卡类型改变时，重新生成参考色块"""
        if self.cc_enabled:
            self._ensure_cc_reference_colors()
            # 触发重绘以显示新的参考色块
            self._update_display()

    def _create_navigation_controls(self, parent_layout):
        """创建文件夹导航控件"""
        # 导航控件容器
        nav_layout = QHBoxLayout()
        nav_layout.setSpacing(4)
        
        # 上一张按钮
        self.nav_prev_btn = QPushButton("◀")
        self.nav_prev_btn.setMaximumWidth(32)
        self.nav_prev_btn.setToolTip(tr("preview_widget.tooltips.nav_prev"))
        self.nav_prev_btn.setEnabled(False)

        # 文件选择下拉框
        self.nav_file_combo = QComboBox()
        self.nav_file_combo.setMinimumWidth(120)
        self.nav_file_combo.setMaximumWidth(200)
        self.nav_file_combo.setToolTip(tr("preview_widget.tooltips.file_combo"))

        # 下一张按钮
        self.nav_next_btn = QPushButton("▶")
        self.nav_next_btn.setMaximumWidth(32)
        self.nav_next_btn.setToolTip(tr("preview_widget.tooltips.nav_next"))
        self.nav_next_btn.setEnabled(False)
        
        # 添加到导航布局
        nav_layout.addWidget(self.nav_prev_btn)
        nav_layout.addWidget(self.nav_file_combo)
        nav_layout.addWidget(self.nav_next_btn)
        
        # 将导航布局添加到父布局
        parent_layout.addLayout(nav_layout)

    def _setup_mouse_controls(self):
        """设置鼠标控制"""
        # 将鼠标事件绑定到image_label而不是scroll_area
        self.image_label.wheelEvent = self._wheel_event
        self.image_label.mousePressEvent = self._mouse_press_event
        self.image_label.mouseMoveEvent = self._mouse_move_event
        self.image_label.mouseReleaseEvent = self._mouse_release_event
        self.image_label.contextMenuEvent = self._context_menu_event
    def _context_menu_event(self, event):
        """右键菜单事件：按住 Cmd/Ctrl 时可删除命中的裁剪（仅原图模式）"""
        try:
            # 增强条件检查：确保在原图模式且有裁剪可操作
            if (not self._show_all_crops or not self._edit_modifier_down or 
                not hasattr(self, '_all_crops') or not self._all_crops):
                return
            
            cid = self._get_crop_at_position(event.pos())
            if cid and cid.strip():  # 确保crop_id不为空
                self.request_delete_crop.emit(cid)
                event.accept(); return
        except Exception as e:
            print(f"右键菜单处理出错: {e}")
        
        # 设置鼠标样式
        self.image_label.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        
    def _setup_keyboard_controls(self):
        """设置键盘控制"""
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def _setup_navigation_controls(self):
        """设置导航控件事件连接"""
        # 连接按钮点击事件
        self.nav_prev_btn.clicked.connect(self._on_nav_previous)
        self.nav_next_btn.clicked.connect(self._on_nav_next)
        
        # 连接下拉框选择变化事件
        self.nav_file_combo.currentIndexChanged.connect(self._on_nav_file_selected)

    def _on_nav_previous(self):
        """处理上一张按钮点击"""
        if self.context and hasattr(self.context, 'folder_navigator'):
            self.context.folder_navigator.navigate_previous()

    def _on_nav_next(self):
        """处理下一张按钮点击"""
        if self.context and hasattr(self.context, 'folder_navigator'):
            self.context.folder_navigator.navigate_next()

    def _on_nav_file_selected(self, index):
        """处理文件选择下拉框变化"""
        if self.context and hasattr(self.context, 'folder_navigator'):
            # 避免循环触发
            if index >= 0:
                self.context.folder_navigator.navigate_to_index(index)

    def update_navigation_state(self):
        """更新导航控件状态"""
        if not self.context or not hasattr(self.context, 'folder_navigator'):
            # 禁用所有导航控件
            self.nav_prev_btn.setEnabled(False)
            self.nav_next_btn.setEnabled(False)
            self.nav_file_combo.clear()
            return
        
        navigator = self.context.folder_navigator
        nav_info = navigator.get_navigation_info()
        
        # 更新按钮状态
        self.nav_prev_btn.setEnabled(nav_info['can_previous'])
        self.nav_next_btn.setEnabled(nav_info['can_next'])
        
        # 更新下拉框内容
        current_index = nav_info['current_index']
        file_list = nav_info['file_list']
        
        # 暂时断开信号连接以避免循环触发
        self.nav_file_combo.blockSignals(True)
        self.nav_file_combo.clear()
        
        # 添加文件列表
        for filename in file_list:
            self.nav_file_combo.addItem(filename)
        
        # 设置当前选中项
        if 0 <= current_index < len(file_list):
            self.nav_file_combo.setCurrentIndex(current_index)
        
        # 重新连接信号
        self.nav_file_combo.blockSignals(False)

    def eventFilter(self, obj, event):
        # 全局监听修饰键变化与鼠标移动，以便更新编辑模式下的光标
        try:
            if event.type() in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
                if hasattr(event, 'key'):
                    key = event.key()
                else:
                    key = None
                if key in (Qt.Key.Key_Control, Qt.Key.Key_Meta):
                    is_down = (event.type() == QEvent.Type.KeyPress)
                    if self._edit_modifier_down != is_down:
                        self._edit_modifier_down = is_down
                        # 根据是否按下编辑键更新光标形态
                        self._update_global_edit_cursor()
            elif event.type() == QEvent.Type.MouseMove:
                # 鼠标移动时也根据当前编辑键状态调整
                self._update_global_edit_cursor()
        except Exception:
            pass
        return super().eventFilter(obj, event)

    def _update_global_edit_cursor(self):
        # 编辑热键按下时：优先显示边/角缩放光标；否则框内显示抓手；否则十字光标
        try:
            if self._edit_modifier_down:
                pos = self.image_label.mapFromGlobal(QCursor.pos())
                if self._show_all_crops:
                    cid, zone = self._get_crop_edit_target_at_position(pos)
                    if zone != CropEditMode.NONE:
                        self._update_crop_cursor(zone)
                        return
                    # Fallback：当无多裁剪条目但存在 overlay 时，允许对 overlay 进行边/角编辑
                    if (not self._all_crops) and self._crop_overlay_norm:
                        zone = self._get_crop_interaction_zone(pos)
                        if zone != CropEditMode.NONE:
                            self._update_crop_cursor(zone)
                            return
                        # 未命中边/角：若在 overlay 内，显示抓手
                        orig_point = self._display_to_original_point(pos)
                        if orig_point and self.current_image and self.current_image.metadata:
                            source_wh = self.current_image.metadata.get('source_wh')
                            if source_wh:
                                src_w, src_h = source_wh
                                nx = orig_point[0] / src_w
                                ny = orig_point[1] / src_h
                                ox, oy, ow, oh = self._crop_overlay_norm
                                if ox <= nx <= ox + ow and oy <= ny <= oy + oh:
                                    self.image_label.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
                                    return
                    # 未命中边/角，则如果在某个裁剪框内部，显示抓手
                    inside_id = self._get_crop_at_position(pos)
                    if inside_id:
                        self.image_label.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
                        return
                    # 默认十字
                    self.image_label.setCursor(QCursor(Qt.CursorShape.CrossCursor))
                    return
                # 聚焦模式
                zone = self._get_crop_interaction_zone(pos)
                if zone != CropEditMode.NONE:
                    self._update_crop_cursor(zone)
                    return
                # 判断是否在当前overlay内
                if self._crop_overlay_norm:
                    orig_point = self._display_to_original_point(pos)
                    if orig_point and self.current_image and self.current_image.metadata:
                        source_wh = self.current_image.metadata.get('source_wh')
                        if source_wh:
                            src_w, src_h = source_wh
                            nx = orig_point[0] / src_w
                            ny = orig_point[1] / src_h
                            ox, oy, ow, oh = self._crop_overlay_norm
                            if ox <= nx <= ox + ow and oy <= ny <= oy + oh:
                                self.image_label.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
                                return
                # 默认十字
                self.image_label.setCursor(QCursor(Qt.CursorShape.CrossCursor))
            else:
                # 根据悬停状态恢复
                pos = self.image_label.mapFromGlobal(QCursor.pos())
                self._handle_crop_hover_detection(pos)
        except Exception:
            pass
    
    def toggle_color_checker(self, checked: bool):
        """Public method to toggle the color checker selector"""
        self.cc_checkbox.setChecked(checked)

    def set_image(self, image_data: ImageData):
        """设置显示的图像"""
        # 保存当前cut-off显示状态
        was_showing_cutoff = self._show_black_cutoff
        current_compensation = self._cutoff_compensation

        # 如果传进来的就是当前这张图，不要把它的 array 清掉
        same_object = (image_data is self.current_image)

        # 显式释放旧的ImageData对象及其numpy array以防止内存泄漏
        if self.current_image is not None and not same_object:
            if hasattr(self.current_image, "array") and self.current_image.array is not None:
                self.current_image.array = None
            self.current_image = None

        # 挂上新图（同一对象就等于“重绘”）
        self.current_image = image_data

        # 若图像元数据中已有裁剪（来自Preset或Context），并且本地未显式覆盖，则创建/同步虚线框
        try:
            md = getattr(image_data, 'metadata', {}) or {}
            crop_md = md.get('crop_overlay', None)
            # 过滤掉全幅裁剪 (0,0,1,1) 等无效框
            def _is_full_rect(r):
                try:
                    x, y, w, h = [float(v) for v in r]
                    return (abs(x - 0.0) < 1e-6 and abs(y - 0.0) < 1e-6 and
                            abs(w - 1.0) < 1e-6 and abs(h - 1.0) < 1e-6)
                except Exception:
                    return False
            if self._crop_overlay_norm is None and crop_md is not None and not _is_full_rect(crop_md):
                self._crop_overlay_norm = tuple(float(v) for v in crop_md)
                self._ensure_ants_timer(True)
        except Exception:
            pass
        self._update_display()
        
        # 恢复cut-off显示状态（如果之前在显示）
        if was_showing_cutoff:
            self._show_black_cutoff = True
            self._cutoff_compensation = current_compensation
            self._detect_black_cutoff_pixels()
            
        self.image_label.update()
        # 不再在每次set_image时自动适应窗口
        # fit_to_window的调用已移至MainWindow中精确控制时机

    def get_current_image_data(self) -> Optional[ImageData]:
        """返回当前显示的ImageData对象"""
        return self.current_image
    
    def _update_display(self):
        """更新显示"""
        if not self.current_image or self.current_image.array is None:
            self.image_label.setText(tr("preview_widget.labels.load_image"))
            return
        
        try:
            # 设置源图与视图参数，自绘中按 pan/zoom 绘制
            pixmap = self._array_to_pixmap(self.current_image.array)
            self._clamp_pan()
            self.image_label.set_source_pixmap(pixmap)
            self.image_label.set_view(self.zoom_factor, self.pan_x, self.pan_y)
            
            # 如果正在显示cut-off，重新检测像素以同步最新图像数据
            if self._show_black_cutoff:
                self._detect_black_cutoff_pixels()
            
        except Exception as e:
            print(f"更新显示失败: {e}")
            self.image_label.setText(tr("preview_widget.labels.display_error", error=str(e)))
    
    def _array_to_pixmap(self, array: np.ndarray) -> QPixmap:
        """将 numpy 数组转换为 QPixmap

        执行关键的内存安全转换：
        1. 处理黑白胶片的灰度转换
        2. 转换为 uint8 格式
        3. 创建 QImage 并确保数据独立性
        4. 应用色彩空间（DisplayP3）
        5. 转换为 QPixmap

        修复说明：
        - 使用 qimage.copy() 确保 QImage 数据独立于 numpy array
        - 防止 array 被释放后 QImage 访问无效内存
        - 只增加少量内存开销（~17MB for 2000×3000 RGB）
        - 拷贝时间 ~10-20ms，不影响交互体验

        注意：
        - Monochrome conversion现在统一由MainWindow中的checkbox机制控制
        - 移除了旧的基于film type的自动conversion，避免双重转换
        """
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 1)
            array = np.round(array * 255).astype(np.uint8)

        # 确保数组是连续的内存布局（旋转后可能不连续）
        if not array.flags['C_CONTIGUOUS']:
            array = np.ascontiguousarray(array)

        height, width = array.shape[:2]

        # 创建 QImage，此时 QImage 引用 numpy array 的数据指针
        if len(array.shape) == 3:
            channels = array.shape[2]
            if channels == 3:
                qimage = QImage(array.data, width, height, width * 3, QImage.Format.Format_RGB888)
            elif channels == 4:
                qimage = QImage(array.data, width, height, width * 4, QImage.Format.Format_RGBA8888)
            else: # Fallback for other channel counts
                array = array[:, :, :3]
                if not array.flags['C_CONTIGUOUS']:
                    array = np.ascontiguousarray(array)
                qimage = QImage(array.data, width, height, width * 3, QImage.Format.Format_RGB888)
        else:
            qimage = QImage(array.data, width, height, width, QImage.Format.Format_Grayscale8)

        # 为 DisplayP3 图像应用 Qt 内置色彩管理
        if hasattr(self, 'current_image') and self.current_image and self.current_image.color_space == "DisplayP3":
            from PySide6.QtGui import QColorSpace
            # 创建色彩空间（DisplayP3）
            displayp3_space = QColorSpace(QColorSpace.NamedColorSpace.DisplayP3)
            # 应用色彩空间到 QImage
            qimage.setColorSpace(displayp3_space)

        # 关键修复：使用 copy() 创建独立的 QImage
        # 这确保 QImage 拥有自己的数据副本，不依赖于 numpy array 的生命周期
        # 性能影响：~10-20ms 拷贝时间，但完全消除数据共享风险
        qimage_independent = qimage.copy()

        # 从独立的 QImage 创建 QPixmap
        # QPixmap.fromImage() 会再次拷贝数据到 GPU 兼容格式
        return QPixmap.fromImage(qimage_independent)

    # ===== 屏幕反光补偿：black cut-off检测 =====
    def set_black_cutoff_display(self, show: bool, compensation: float = 0.0):
        """设置是否显示black cut-off像素"""
        self._show_black_cutoff = show
        self._cutoff_compensation = compensation
        if show:
            self._detect_black_cutoff_pixels()
        else:
            self._cutoff_pixels = None
        self.image_label.update()
    
    def update_cutoff_compensation(self, compensation: float):
        """实时更新补偿值（用于滑块拖动过程中）"""
        if self._show_black_cutoff:
            self._cutoff_compensation = compensation
            self._detect_black_cutoff_pixels()
            self.image_label.update()
    
    def _detect_black_cutoff_pixels(self):
        """检测因屏幕反光补偿导致的black cut-off像素"""
        # 释放旧的mask以避免累积（~6MB if active）
        if self._cutoff_pixels is not None:
            self._cutoff_pixels = None

        if not self.current_image or self.current_image.array is None:
            self._cutoff_pixels = None
            return
        
        try:
            if self._cutoff_compensation <= 0.0:
                self._cutoff_pixels = None
                return
            
            # 获取图像数据 - 这里使用代理图像进行检测以提高性能
            image_array = self.current_image.array
            
            # 检查是否需要转换为黑白（与显示逻辑一致）
            should_convert_to_mono = False
            if self.context and hasattr(self.context, 'should_convert_to_monochrome'):
                should_convert_to_mono = self.context.should_convert_to_monochrome()
            
            # 如果需要转换为黑白，先进行转换（与_array_to_pixmap逻辑一致）
            if should_convert_to_mono and len(image_array.shape) == 3 and image_array.shape[2] >= 3:
                # Convert RGB to luminance using ITU-R BT.709 weights
                luminance = (0.2126 * image_array[:, :, 0] + 
                            0.7152 * image_array[:, :, 1] + 
                            0.0722 * image_array[:, :, 2])
                # 对于cut-off检测，我们只需要单通道的luminance值
                image_array = luminance
            
            # 为了提高性能，如果图像太大，进行降采样检测
            h, w = image_array.shape[:2] if len(image_array.shape) >= 2 else (image_array.shape[0], 1)
            max_size = 1000  # 最大检测尺寸
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                
                # 使用OpenCV或numpy进行降采样
                try:
                    import cv2
                    if len(image_array.shape) == 3:
                        image_array = cv2.resize(image_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    else:
                        image_array = cv2.resize(image_array, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                except ImportError:
                    # 如果没有OpenCV，使用简单的降采样
                    step_h = max(1, h // new_h)
                    step_w = max(1, w // new_w) 
                    image_array = image_array[::step_h, ::step_w]
                    new_h, new_w = image_array.shape[:2] if len(image_array.shape) >= 2 else (image_array.shape[0], 1)
                
                h, w = new_h, new_w
            
            # 使用固定0.0阈值检测cut-off像素：
            # 模拟补偿效果，检测应用补偿后会变成负值的像素
            cutoff_mask = np.zeros((h, w), dtype=bool)
            
            if len(image_array.shape) == 3:
                # RGB图像：检查所有通道
                for c in range(image_array.shape[2]):
                    channel = image_array[:, :, c].astype(np.float64)
                    # 模拟补偿：如果 pixel - compensation < 0.0，则会被cut-off
                    channel_cutoff = (channel - self._cutoff_compensation) < 0.0
                    cutoff_mask |= channel_cutoff
            else:
                # 灰度图或黑白模式转换后的单通道
                channel = image_array.astype(np.float64)
                # 模拟补偿：如果 pixel - compensation < 0.0，则会被cut-off
                cutoff_mask = (channel - self._cutoff_compensation) < 0.0
            
            self._cutoff_pixels = cutoff_mask
            
        except Exception as e:
            print(f"检测black cut-off像素失败: {e}")
            self._cutoff_pixels = None
    
    def _draw_black_cutoff_overlay(self, painter: QPainter):
        """绘制black cut-off像素覆盖层"""
        if not self._show_black_cutoff or self._cutoff_pixels is None:
            return
        
        try:
            if not self.current_image or self.current_image.array is None:
                return
            
            # 获取cut-off mask的尺寸
            mask_h, mask_w = self._cutoff_pixels.shape
            
            # 获取当前显示的图像尺寸（考虑代理图像）
            image_h, image_w = self.current_image.array.shape[:2]
            
            # 计算从mask坐标到图像坐标的映射比例
            # 如果cut-off检测时进行了降采样，需要恢复到原图像尺寸
            scale_x = image_w / mask_w
            scale_y = image_h / mask_h
            
            # 设置绘制参数：半透明红色覆盖
            painter.save()
            painter.setPen(QPen(QColor(255, 0, 0, 100), 0))  # 红色边框
            painter.setBrush(QColor(255, 0, 0, 80))  # 半透明红色填充
            
            # 根据当前缩放级别动态调整绘制密度，避免性能问题
            zoom = self.zoom_factor
            if zoom < 0.25:
                step = 8  # 缩放很小时，大幅降采样
            elif zoom < 0.5:
                step = 4
            elif zoom < 1.0:
                step = 2
            else:
                step = 1  # 放大时精细绘制
            
            # 在图像坐标系中绘制像素
            # painter已经应用了transform（translate + scale），所以这里直接用图像坐标
            for mask_y in range(0, mask_h, step):
                for mask_x in range(0, mask_w, step):
                    if self._cutoff_pixels[mask_y, mask_x]:
                        # 将mask坐标映射回图像坐标
                        img_x = int(mask_x * scale_x)
                        img_y = int(mask_y * scale_y)
                        
                        # 计算像素块大小（考虑步长和缩放比例）
                        pixel_w = max(1, int(scale_x * step))
                        pixel_h = max(1, int(scale_y * step))
                        
                        # 直接在图像坐标系中绘制，让painter的transform处理缩放和平移
                        painter.drawRect(img_x, img_y, pixel_w, pixel_h)
            
            painter.restore()
            
        except Exception as e:
            print(f"绘制black cut-off覆盖层失败: {e}")

    # ===== 裁剪 overlay 与交互 =====
    def _draw_crop_overlay(self, painter: QPainter):
        """绘制裁剪框覆盖层"""
        # 在原图模式下显示所有裁剪框
        if self._show_all_crops and self._all_crops:
            self._ensure_ants_timer(True)
            for crop_info in self._all_crops:
                rect_norm = crop_info['rect_norm']
                crop_id = crop_info['id']
                index = crop_info['index']
                is_active = crop_info['active']
                is_hovering = crop_id == self._hovering_crop_id
                
                # 选择颜色
                if is_active:
                    color = QColor(255, 200, 0, 220)  # 活跃裁剪用黄色
                elif is_hovering:
                    color = QColor(0, 255, 255, 220)  # 悬停用青色
                else:
                    color = QColor(255, 255, 255, 220)  # 普通用白色
                
                # 绘制虚线框
                img_rect = self._norm_to_image_rect(rect_norm)
                if img_rect:
                    x, y, w, h = img_rect
                    pen = QPen(color, 2)
                    pen.setStyle(Qt.DashLine)
                    pen.setDashPattern([6, 4])
                    pen.setDashOffset(self._ants_phase)
                    painter.setPen(pen)
                    painter.setBrush(Qt.NoBrush)
                    painter.drawRect(x, y, w, h)
                    
                    # 绘制序号标签
                    self._draw_crop_label(painter, x, y, str(index), color)
        else:
            # 聚焦模式或没有裁剪时的逻辑
            # 需求：在单图聚焦模式下不显示虚线框
            is_focused = False
            try:
                if self.current_image and self.current_image.metadata:
                    is_focused = bool(self.current_image.metadata.get('crop_focused', False))
            except Exception:
                pass
            if is_focused:
                # 聚焦：不画overlay，也关闭蚂蚁线动画
                self._ensure_ants_timer(False)
            else:
                rect_norm = self._crop_overlay_norm
                try:
                    if rect_norm is None and self.current_image and self.current_image.metadata:
                        rect_norm = self.current_image.metadata.get('crop_overlay', None)
                except Exception:
                    pass
                if rect_norm is not None:
                    self._ensure_ants_timer(True)
                    img_rect = self._norm_to_image_rect(rect_norm)
                    if img_rect is not None:
                        pen = QPen(QColor(255, 255, 255, 220), 2)
                        pen.setStyle(Qt.DashLine)
                        pen.setDashPattern([6, 4])
                        pen.setDashOffset(self._ants_phase)
                        painter.setPen(pen)
                        painter.setBrush(Qt.NoBrush)
                        x, y, w, h = img_rect
                        painter.drawRect(x, y, w, h)
                else:
                    self._ensure_ants_timer(False)

        # 绘制临时框选（绿框）
        if self._crop_mode and self._crop_start_norm and self._crop_current_norm:
            self._ensure_ants_timer(True)
            # 绿框：临时框选，直接使用归一化坐标
            x0, y0 = self._crop_start_norm
            x1, y1 = self._crop_current_norm
            temp_norm = (min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0))
            
            # 直接将归一化坐标转换为显示坐标
            temp_display = self._norm_to_image_rect(temp_norm)
            if temp_display is not None:
                x, y, w, h = temp_display
                pen = QPen(QColor(0, 255, 0, 220), 2)
                pen.setStyle(Qt.DashLine)
                pen.setDashPattern([6, 4])
                pen.setDashOffset(self._ants_phase)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(x, y, w, h)

    def _draw_crop_label(self, painter: QPainter, x: int, y: int, text: str, color: QColor):
        """绘制裁剪框的序号标签"""
        # 设置更大的字体
        font = painter.font()
        font.setPixelSize(18)  # 从14增加到18像素
        font.setBold(True)
        painter.setFont(font)
        
        metrics = painter.fontMetrics()
        text_rect = metrics.boundingRect(text)
        padding = 5  # 从3增加到5像素，增强视觉效果
        
        # 将标号放在crop框内部的左上角
        label_rect = QRect(
            x + padding,  # 在crop框内部，左边留padding
            y + padding,  # 在crop框内部，上边留padding
            text_rect.width() + padding * 2,
            text_rect.height() + padding * 2
        )
        
        # 绘制背景
        painter.fillRect(label_rect, QColor(0, 0, 0, 180))
        
        # 绘制文字
        painter.setPen(color)
        painter.drawText(label_rect, Qt.AlignCenter, text)
    
    def _draw_overlays(self, painter: QPainter):
        try:
            self._draw_colorchecker_overlay(painter)
        except Exception:
            pass
        try:
            self._draw_crop_overlay(painter)
        except Exception:
            pass
        try:
            self._draw_black_cutoff_overlay(painter)
        except Exception:
            pass
        try:
            self._draw_neutral_point_overlay(painter)
        except Exception:
            pass

    def _ensure_ants_timer(self, on: bool):
        if on:
            if not self._ants_timer.isActive():
                self._ants_timer.start()
        else:
            if self._ants_timer.isActive():
                self._ants_timer.stop()

    def _advance_ants(self):
        self._ants_phase = (self._ants_phase + 1.0) % 1000.0
        self.image_label.update()

    def set_crop_overlay(self, rect_norm_or_none):
        self._crop_overlay_norm = rect_norm_or_none
        has_crop = rect_norm_or_none is not None
        # 按钮已移除，使用底部选择条
        # 立即启动蚂蚁虚线动画并重绘覆盖层，必要时也刷新底图以确保立刻可见
        self._ensure_ants_timer(bool(has_crop))
        # 刷新显示，确保 paintEvent 立即走一遍
        self._update_display()
        self.image_label.update()

    def _on_focus_clicked(self):
        """点击关注：清理临时裁剪交互状态，避免叠加绿色框，并请求上层聚焦。"""
        # 清理临时交互框（绿色）
        self._crop_mode = False
        self._crop_dragging = False
        self._crop_start_norm = None
        self._crop_current_norm = None
        
        # 清理编辑状态
        self._reset_crop_editing_state()
        self.image_label.update()
        # 请求上层执行聚焦（Context 生成新proxy并回传overlay）
        self.request_focus_crop.emit()

    def _on_restore_clicked(self):
        """点击恢复：清理临时裁剪交互状态，避免叠加绿色框，并请求上层恢复。"""
        self._crop_mode = False
        self._crop_dragging = False
        self._crop_start_norm = None
        self._crop_current_norm = None
        
        # 清理编辑状态
        self._reset_crop_editing_state()
        self.image_label.update()
        self.request_restore_crop.emit()

    def _toggle_crop_mode(self, enabled: bool = None):
        if enabled is not None:
            self._crop_mode = enabled
        else:
            self._crop_mode = not self._crop_mode
        if not self._crop_mode:
            self._crop_dragging = False
            self._crop_start_norm = None
            self._crop_current_norm = None
        
        # 切换模式时清理悬停状态
        self._clear_crop_hover_state()
        
        # 进入裁剪模式：立即切换光标样式，给予交互反馈
        if self._crop_mode:
            self.image_label.setCursor(QCursor(Qt.CursorShape.CrossCursor))
        else:
            self.image_label.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        self._update_display()

    def _reset_crop_editing_state(self):
        """统一重置所有裁剪编辑相关状态，避免状态不一致"""
        self._crop_edit_mode = CropEditMode.NONE
        self._crop_edit_start_rect = None
        self._crop_edit_hover_mode = CropEditMode.NONE
        self._active_edit_crop_id = None
        self._dragging_crop_id = None
        self._drag_offset = (0, 0)
        
        # 重置悬停状态
        self._clear_crop_hover_state()
        
        # 恢复默认光标
        self.image_label.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    def _handle_crop_double_click(self, crop_id: str) -> bool:
        """处理裁剪框双击事件，返回True表示处理了双击"""
        if (not self._edit_modifier_down and 
            self._pending_click_crop_id == crop_id and 
            self._double_click_timer.isActive()):
            self._double_click_timer.stop()
            self._pending_click_crop_id = None
            self.request_focus_crop.emit(crop_id)
            return True
        return False

    def _start_crop_click_tracking(self, crop_id: str):
        """开始跟踪裁剪框点击，用于双击检测"""
        self._pending_click_crop_id = crop_id
        self._double_click_timer.start()

    def _norm_to_image_rect(self, rect_norm):
        """将原图归一化坐标转换为当前显示图像的像素坐标（UI显示用）"""
        try:
            if not self.current_image or self.current_image.array is None:
                return None
            x, y, w, h = rect_norm
            md = self.current_image.metadata or {}
            src_wh = md.get('source_wh', None)
            focused = bool(md.get('crop_focused', False))
            
            if not src_wh:
                # 元数据缺失时优雅退化
                return (x * self.current_image.width, y * self.current_image.height, 
                       w * self.current_image.width, h * self.current_image.height)
            
            src_w, src_h = src_wh
            disp_h, disp_w = self.current_image.array.shape[:2]
            
            if focused:
                # === 聚焦模式：显示crop内的相对位置 ===
                crop_instance = md.get('crop_instance')
                if not crop_instance:
                    return None
                
                cx, cy, cw, ch = crop_instance.rect_norm
                
                # 将要显示的区域转换为相对于crop的坐标
                rel_x = (x - cx) / cw if cw > 0 else 0
                rel_y = (y - cy) / ch if ch > 0 else 0
                rel_w = w / cw if cw > 0 else 0
                rel_h = h / ch if ch > 0 else 0
                
                # 处理crop的独立orientation
                if crop_instance.orientation % 360 != 0:
                    # 应用crop的旋转变换
                    rel_x, rel_y, rel_w, rel_h = self._apply_orientation_to_rect(
                        rel_x, rel_y, rel_w, rel_h, crop_instance.orientation
                    )
                
                # 映射到当前显示像素坐标
                px = rel_x * disp_w
                py = rel_y * disp_h
                pw = rel_w * disp_w
                ph = rel_h * disp_h
                
                return (px, py, pw, ph)
            else:
                # === 非聚焦模式：显示完整图像上的位置 ===
                global_orientation = md.get('global_orientation', 0)
                
                # Step 1: 原图归一化坐标 → 像素坐标
                orig_px = x * src_w
                orig_py = y * src_h
                orig_pw = w * src_w
                orig_ph = h * src_h
                
                # Step 2: 应用全局orientation变换
                if global_orientation % 360 != 0:
                    # 应用旋转变换到矩形
                    disp_x, disp_y, disp_w_rotated, disp_h_rotated = self._apply_orientation_to_rect(
                        orig_px / src_w, orig_py / src_h, orig_pw / src_w, orig_ph / src_h, global_orientation
                    )
                else:
                    # 无旋转
                    disp_x = orig_px / src_w
                    disp_y = orig_py / src_h
                    disp_w_rotated = orig_pw / src_w
                    disp_h_rotated = orig_ph / src_h
                
                # Step 3: 映射到当前显示像素坐标
                px = disp_x * disp_w
                py = disp_y * disp_h
                pw = disp_w_rotated * disp_w
                ph = disp_h_rotated * disp_h
                
                return (px, py, pw, ph)
                
        except Exception:
            return None
    
    def _apply_orientation_to_rect(self, x, y, w, h, orientation):
        """应用旋转变换到归一化矩形（0-1坐标系）"""
        if orientation % 360 == 0:
            return (x, y, w, h)
        
        k = (orientation // 90) % 4
        # 统一方向与 np.rot90 保持一致：k 表示逆时针 CCW 次数
        if k == 1:  # 90° 逆时针（CCW）
            # 变换：(x,y,w,h) → (y, 1-x-w, h, w)
            return (y, 1.0 - x - w, h, w)
        elif k == 2:  # 180度
            # 变换：(x,y,w,h) → (1-x-w, 1-y-h, w, h)
            return (1.0 - x - w, 1.0 - y - h, w, h)
        elif k == 3:  # 270° 逆时针（即 90° 顺时针）
            # 变换：(x,y,w,h) → (1-y-h, x, h, w)
            return (1.0 - y - h, x, h, w)
        else:
            return (x, y, w, h)
    
    def _wheel_event(self, event):
        """鼠标滚轮事件 - 缩放"""
        if not self.current_image: 
            event.accept()
            return
            
        delta = event.angleDelta().y()
        
        # 获取鼠标在 label 坐标系中的位置（使用浮点坐标保证精度）
        m = event.position()

        # 计算围绕鼠标的缩放：保持鼠标下像素不动
        zoom_factor_change = 1.05 if delta > 0 else 1/1.05
        old_zoom = float(self.zoom_factor)
        new_zoom = float(self.zoom_factor * zoom_factor_change)
        new_zoom = max(self._min_zoom, min(self._max_zoom, new_zoom))

        # 图像坐标 p（以当前 pan, zoom 映射）
        # p = (m - t) / s
        p_x = (float(m.x()) - float(self.pan_x)) / old_zoom
        p_y = (float(m.y()) - float(self.pan_y)) / old_zoom

        # 新平移 t' = m - p * s'
        new_pan_x = float(m.x()) - p_x * new_zoom
        new_pan_y = float(m.y()) - p_y * new_zoom

        self.zoom_factor = new_zoom
        # 使用浮点 pan，避免取整引入的锚点跳变
        self.pan_x = new_pan_x
        self.pan_y = new_pan_y

        # 边界约束（基于新 zoom）
        self._clamp_pan()

        # 更新显示
        self._update_display()
        
        event.accept()
    
    def _get_crop_at_position(self, pos: QPoint) -> Optional[str]:
        """获取鼠标位置下的裁剪框ID"""
        if not self._show_all_crops or not self._all_crops:
            return None
        
        # 转换为归一化坐标
        try:
            display_point = (pos.x(), pos.y())
            norm_point = self._display_to_original_point(display_point)
            if not norm_point:
                return None
            
            # 归一化
            if self.current_image and self.current_image.metadata:
                source_wh = self.current_image.metadata.get('source_wh')
                if source_wh:
                    src_w, src_h = source_wh
                    norm_x = norm_point[0] / src_w
                    norm_y = norm_point[1] / src_h
                    
                    # 检查每个裁剪框
                    for crop_info in self._all_crops:
                        x, y, w, h = crop_info['rect_norm']
                        if x <= norm_x <= x + w and y <= norm_y <= y + h:
                            return crop_info['id']
        except Exception:
            pass
        
        return None
    
    def _mouse_press_event(self, event):
        """鼠标按下事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            # 优先级0：中性点选择模式（最高优先级）
            if self.neutral_point_selection_mode:
                # 直接从preview图像取色，不转换到原图坐标
                # 这样点击位置、光标位置、取色位置始终一致，不受裁剪/旋转/聚焦影响
                if self.current_image and self.current_image.array is not None:
                    try:
                        # 显示坐标 → preview图像坐标
                        mx, my = float(event.position().x()), float(event.position().y())
                        img_x = (mx - self.pan_x) / self.zoom_factor
                        img_y = (my - self.pan_y) / self.zoom_factor

                        # preview图像尺寸
                        img_h, img_w = self.current_image.array.shape[:2]

                        # 转换为归一化坐标（基于preview图像）
                        norm_x = img_x / img_w
                        norm_y = img_y / img_h

                        # 边界检查
                        if 0 <= norm_x <= 1 and 0 <= norm_y <= 1:
                            # 保存选定的中性点
                            self.neutral_point_norm = (norm_x, norm_y)

                            # 退出选择模式
                            self.exit_neutral_point_selection_mode()

                            # 发射信号，通知已选择中性点
                            self.neutral_point_selected.emit(norm_x, norm_y)

                            event.accept()
                            return
                    except Exception as e:
                        print(f"中性点选择失败: {e}")

            # 记录点击时间，用于双击检测
            current_time = self._double_click_timer.remainingTime()

            # 检查是否点击在裁剪框内（原图模式，多裁剪）
            if self._show_all_crops:
                clicked_crop_id = self._get_crop_at_position(event.pos())
                if clicked_crop_id:
                    # 双击聚焦（仅在未按编辑修饰键时生效）
                    if self._handle_crop_double_click(clicked_crop_id):
                        event.accept(); return
                    
                    # 记录点击，用于后续双击判定
                    self._start_crop_click_tracking(clicked_crop_id)
                    
                    if self._edit_modifier_down:
                        # 优先边/角编辑
                        cid, zone = self._get_crop_edit_target_at_position(event.pos())
                        if zone != CropEditMode.NONE and cid is not None:
                            self._active_edit_crop_id = cid
                            for info in self._all_crops:
                                if info['id'] == cid:
                                    self._crop_edit_start_rect = info['rect_norm']
                                    break
                            self._crop_edit_mode = zone
                            self._update_crop_cursor(zone)
                            event.accept(); return
                        # 否则框内拖动移动该裁剪框
                        self._dragging_crop_id = clicked_crop_id
                        # 记录拖动偏移
                        orig_point = self._display_to_original_point(event.position())
                        if orig_point and self.current_image and self.current_image.metadata:
                            source_wh = self.current_image.metadata.get('source_wh')
                            if source_wh:
                                src_w, src_h = source_wh
                                norm_x = orig_point[0] / src_w
                                norm_y = orig_point[1] / src_h
                                for crop_info in self._all_crops:
                                    if crop_info['id'] == clicked_crop_id:
                                        cx, cy, cw, ch = crop_info['rect_norm']
                                        self._drag_offset = (norm_x - cx, norm_y - cy)
                                        break
                        event.accept(); return

            # 原图模式下：若按下编辑修饰键且当前不存在多裁剪列表，但存在 overlay，则允许直接边/角编辑 overlay
            if self._show_all_crops and self._edit_modifier_down and (not self._all_crops) and self._crop_overlay_norm:
                zone = self._get_crop_interaction_zone(event.pos())
                if zone != CropEditMode.NONE:
                    self._active_edit_crop_id = None
                    self._crop_edit_start_rect = self._crop_overlay_norm
                    self._crop_edit_mode = zone
                    self._update_crop_cursor(zone)
                    event.accept(); return
            
            # 按住 Cmd/Ctrl 且未命中任何裁剪框时：直接进入"新建裁剪"框选（支持所有模式，包括聚焦模式）
            if self._edit_modifier_down:
                try:
                    # 再次确认当前位置未命中任何现有裁剪（避免与移动/编辑冲突）
                    if not self._get_crop_at_position(event.pos()):
                        orig_point = self._display_to_original_point(event.position())
                        if orig_point is not None and self.current_image and self.current_image.metadata:
                            source_wh = self.current_image.metadata.get('source_wh')
                            if source_wh:
                                src_w, src_h = source_wh
                                norm_x = orig_point[0] / src_w
                                norm_y = orig_point[1] / src_h
                                # 进入框选模式并设定起点
                                self._crop_creation_intent = 'single'
                                self._crop_mode = True
                                self._crop_dragging = True
                                self._crop_start_norm = (norm_x, norm_y)
                                self._crop_current_norm = (norm_x, norm_y)
                                self.image_label.setCursor(QCursor(Qt.CursorShape.CrossCursor))
                                event.accept(); return
                except Exception:
                    pass

            # 优先级1：裁剪框编辑模式（聚焦模式）
            if (not self._crop_mode and 
                self._crop_overlay_norm and 
                self._crop_edit_mode == CropEditMode.NONE and
                not self._show_all_crops):
                # 聚焦模式下禁用裁剪编辑与整体移动
                pass
            
            # 优先级2：裁剪选择模式
            if self._crop_mode:
                # 直接转换为原图归一化坐标，确保与显示逻辑一致
                orig_point = self._display_to_original_point(event.position())
                if orig_point is not None:
                    # 获取原图尺寸进行归一化
                    md = self.current_image.metadata or {} if self.current_image else {}
                    src_wh = md.get('source_wh')
                    if src_wh:
                        src_w, src_h = src_wh
                        norm_x = orig_point[0] / src_w
                        norm_y = orig_point[1] / src_h
                        self._crop_dragging = True
                        self._crop_start_norm = (norm_x, norm_y)
                        self._crop_current_norm = (norm_x, norm_y)
                        self.image_label.setCursor(QCursor(Qt.CursorShape.CrossCursor))
                        event.accept(); return
            # 色卡角点命中测试（优先）
            if self.cc_enabled and self.cc_corners_norm:
                # 转换归一化坐标到显示坐标进行命中测试
                display_corners = self._norm_to_display_coords(self.cc_corners_norm)
                if display_corners:
                    m = event.position()
                    for idx, (ix, iy) in enumerate(display_corners):
                        dx = (ix * float(self.zoom_factor) + float(self.pan_x)) - float(m.x())
                        dy = (iy * float(self.zoom_factor) + float(self.pan_y)) - float(m.y())
                        if (dx*dx + dy*dy) ** 0.5 <= self.cc_handle_radius_disp:
                            self.cc_drag_idx = idx
                            event.accept(); return

            # 如果按下了编辑修饰键（Cmd/Ctrl），不应该进入平移模式
            # 编辑键用于框选/编辑裁剪，与平移互斥
            if self._edit_modifier_down:
                event.accept()
                return

            self.dragging = True
            self.last_mouse_pos = event.pos()
            self.drag_start_pos = event.pos()
            
            # 记录开始拖动时的平移位置（使用浮点坐标）
            self.original_pan_pos = QPointF(float(self.pan_x), float(self.pan_y))
            
            # 改变鼠标样式
            self.image_label.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            
        event.accept()
    
    def _mouse_move_event(self, event):
        """鼠标移动事件"""
        # === 优先级0：悬停检测（独立于其他交互） ===
        self._handle_crop_hover_detection(event.pos())
        
        # === 优先级1：拖动裁剪框（原图模式） ===
        if self._dragging_crop_id and self._show_all_crops:
            orig_point = self._display_to_original_point(event.position())
            if orig_point and self.current_image and self.current_image.metadata:
                source_wh = self.current_image.metadata.get('source_wh')
                if source_wh:
                    src_w, src_h = source_wh
                    norm_x = orig_point[0] / src_w
                    norm_y = orig_point[1] / src_h
                    # 计算新位置
                    new_x = norm_x - self._drag_offset[0]
                    new_y = norm_y - self._drag_offset[1]
                    # 更新裁剪框位置
                    for crop_info in self._all_crops:
                        if crop_info['id'] == self._dragging_crop_id:
                            x, y, w, h = crop_info['rect_norm']
                            # 限制边界
                            new_x = max(0, min(1 - w, new_x))
                            new_y = max(0, min(1 - h, new_y))
                            crop_info['rect_norm'] = (new_x, new_y, w, h)
                            # 发出更新信号（增强验证）
                            if self._dragging_crop_id and self._dragging_crop_id.strip():
                                self.crop_updated.emit(self._dragging_crop_id, (new_x, new_y, w, h))
                            self.image_label.update()
                            break
            event.accept()
            return
        
        # === 优先级2：正在进行的交互 ===
        if self._crop_edit_mode != CropEditMode.NONE:
            if self._active_edit_crop_id is not None:
                self._apply_crop_edit_multi(event.pos())
            else:
                self._apply_crop_edit(event.pos())
            event.accept()
            return
            
        # === 优先级2：裁剪选择模式 ===  
        if self._crop_mode:
            # 转换为原图归一化坐标
            orig_point = self._display_to_original_point(event.position())
            if orig_point is not None:
                md = self.current_image.metadata or {} if self.current_image else {}
                src_wh = md.get('source_wh')
                if src_wh:
                    src_w, src_h = src_wh
                    norm_x = orig_point[0] / src_w
                    norm_y = orig_point[1] / src_h
                    
                    if not self._crop_dragging:
                        self._crop_dragging = True
                        self._crop_start_norm = (norm_x, norm_y)
                        self._crop_current_norm = (norm_x, norm_y)
                    else:
                        # 支持 Shift 限定为正方形选区
                        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                            sx, sy = self._crop_start_norm
                            dx = norm_x - sx; dy = norm_y - sy
                            side = max(abs(dx), abs(dy))
                            norm_x = sx + (side if dx >= 0 else -side)
                            norm_y = sy + (side if dy >= 0 else -side)
                        self._crop_current_norm = (norm_x, norm_y)
                    self._update_display(); event.accept(); return
        if (self.cc_enabled and self.cc_drag_idx is not None and 
            self.cc_corners_norm and 0 <= self.cc_drag_idx < len(self.cc_corners_norm)):
            # 将显示坐标转换为归一化坐标
            norm_coords = self._display_to_norm_coords(event.position())
            if norm_coords:
                self.cc_corners_norm[self.cc_drag_idx] = norm_coords
                self._update_display(); event.accept(); return
        # 聚焦模式：整体移动裁剪框（禁用；仅在原图多裁剪下允许）
        if self._moving_overlay_active and self._crop_overlay_norm and self._show_all_crops:
            orig_point = self._display_to_original_point(event.position())
            if orig_point and self.current_image and self.current_image.metadata:
                source_wh = self.current_image.metadata.get('source_wh')
                if source_wh:
                    src_w, src_h = source_wh
                    norm_x = orig_point[0] / src_w
                    norm_y = orig_point[1] / src_h
                    off_x, off_y = self._moving_overlay_offset_norm
                    _, _, w, h = self._crop_overlay_norm
                    new_x = norm_x - off_x
                    new_y = norm_y - off_y
                    new_x = max(0.0, min(1.0 - w, new_x))
                    new_y = max(0.0, min(1.0 - h, new_y))
                    self._crop_overlay_norm = (new_x, new_y, w, h)
                    self.image_label.update()
            event.accept(); return
        if self.dragging and self.last_mouse_pos:
            delta = event.pos() - self.last_mouse_pos
            
            # 直接更新平移偏移
            # 使用浮点累加，避免取整抖动
            self.pan_x += float(delta.x())
            self.pan_y += float(delta.y())

            # 边界约束（M3）
            self._clamp_pan()
            
            # 更新显示
            self._update_display()
            
            self.last_mouse_pos = event.pos()
            
        event.accept()
    
    def _mouse_release_event(self, event):
        """鼠标释放事件"""
        if event.button() == Qt.MouseButton.LeftButton:
            # 优先级1：结束裁剪框拖动
            if self._dragging_crop_id:
                self._dragging_crop_id = None
                self._drag_offset = (0, 0)
                event.accept()
                return
            
            # 优先级2：裁剪框编辑结束
            if self._crop_edit_mode != CropEditMode.NONE:
                try:
                    if self._active_edit_crop_id is not None:
                        # 多裁剪：边/角编辑结束，按当前rect_norm发送更新
                        if self._all_crops:  # 边界检查：确保裁剪列表不为空
                            for info in self._all_crops:
                                if info and info.get('id') == self._active_edit_crop_id:
                                    rect_norm = info.get('rect_norm')
                                    if rect_norm:
                                        self.crop_updated.emit(self._active_edit_crop_id, rect_norm)
                                    break
                    else:
                        # 聚焦模式：按overlay发送更新
                        if self._crop_overlay_norm and self._all_crops:  # 边界检查：确保两者都存在
                            active_id = None
                            for crop_info in self._all_crops:
                                if crop_info and crop_info.get('active'):
                                    active_id = crop_info.get('id')
                                    break
                            if active_id:
                                self.crop_updated.emit(active_id, self._crop_overlay_norm)
                except Exception:
                    pass

                # 重置编辑状态（仅重置编辑相关，不重置拖动状态）
                self._crop_edit_mode = CropEditMode.NONE
                self._crop_edit_start_rect = None
                self._active_edit_crop_id = None

                # 检查当前悬停状态并更新光标
                hover_mode = self._get_crop_interaction_zone(event.pos()) if self._edit_modifier_down else CropEditMode.NONE
                self._crop_edit_hover_mode = hover_mode
                self._update_crop_cursor(hover_mode)

                event.accept()
                return

            # 聚焦模式：结束整体移动裁剪框（禁用；仅在原图多裁剪下允许）
            if self._moving_overlay_active and self._show_all_crops:
                self._moving_overlay_active = False
                try:
                    if self._crop_overlay_norm:
                        active_id = None
                        for crop_info in self._all_crops:
                            if crop_info['active']:
                                active_id = crop_info['id']
                                break
                        if active_id:
                            self.crop_updated.emit(active_id, self._crop_overlay_norm)
                except Exception:
                    pass
                event.accept()
                return
            
            self.cc_drag_idx = None
            self.dragging = False
            self.last_mouse_pos = None
            self.drag_start_pos = None
            self.original_pan_pos = None
            
            # 优先级2：提交新裁剪
            if self._crop_mode and self._crop_dragging and self._crop_start_norm and self._crop_current_norm:
                # 直接使用归一化坐标，无需转换
                x0, y0 = self._crop_start_norm
                x1, y1 = self._crop_current_norm
                x = min(x0, x1); y = min(y0, y1)
                w = abs(x1 - x0); h = abs(y1 - y0)
                
                # 边界约束（归一化坐标）
                x = max(0.0, min(1.0, x))
                y = max(0.0, min(1.0, y))
                w = max(0.01, min(1.0 - x, w))  # 最小宽度1%
                h = max(0.01, min(1.0 - y, h))  # 最小高度1%
                
                rect_norm = (x, y, w, h)
                try:
                    # 先在本地创建/更新虚线框（立即可见）
                    self._crop_overlay_norm = rect_norm
                    self._ensure_ants_timer(True)
                    self.image_label.update()
                    # 再发信号交由上层持久化与刷新代理
                    if self._crop_creation_intent == 'add':
                        # 正式新增 crop（验证rect_norm有效性）
                        if rect_norm and len(rect_norm) == 4:
                            self.crop_committed.emit(rect_norm)
                    else:
                        # 单张裁剪（不新增正式 crop）
                        if rect_norm and len(rect_norm) == 4:
                            self.single_crop_committed.emit(rect_norm)
                except Exception:
                    pass
                self._crop_dragging = False
                self._crop_mode = False
                # 重置意图
                self._crop_creation_intent = None

            # 恢复鼠标样式
            self.image_label.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
            
        event.accept()
    
    def _smooth_drag_update(self):
        """平滑拖动更新（预留功能）"""
        # 这里可以实现更平滑的拖动效果
        pass
    
    def reset_view(self):
        """重置视图"""
        self.zoom_factor = 1.0
        self.center_image()

    def fit_to_window(self):
        """适应窗口大小"""
        if not self.current_image: 
            return
            
        # 使用viewport的可用区域尺寸，避免滚动条/边距带来的误差
        widget_size = self.scroll_area.viewport().size()
        image_size = self.current_image.array.shape[:2][::-1]
        scale_x = widget_size.width() / image_size[0]
        scale_y = widget_size.height() / image_size[1]
        # 聚焦裁剪时允许放大以填满窗口；非聚焦时不放大
        focused = False
        try:
            if self.current_image and self.current_image.metadata:
                focused = bool(self.current_image.metadata.get('crop_focused', False))
        except Exception:
            focused = False
        self.zoom_factor = min(scale_x, scale_y) if focused else min(scale_x, scale_y, 1.0)

        # 居中平移
        Wi, Hi = self._get_scaled_image_size()
        Wv, Hv = widget_size.width(), widget_size.height()
        self.pan_x = int(round((Wv - Wi) / 2))
        self.pan_y = int(round((Hv - Hi) / 2))
        self._clamp_pan()

        self._update_display()
    
    def center_image(self):
        """居中显示图像"""
        if not self.current_image:
            return

        widget_size = self._get_viewport_size()
        Wi, Hi = self._get_scaled_image_size()
        Wv, Hv = widget_size.width(), widget_size.height()
        self.pan_x = int(round((Wv - Wi) / 2))
        self.pan_y = int(round((Hv - Hi) / 2))
        self._clamp_pan()

        self._update_display()

    # ===== 色卡选择器：读取24色块平均色（XYZ） =====
    def read_colorchecker_xyz(self):
        """读取当前色卡选择器24块的平均颜色（XYZ），返回 ndarray (24, 3)。行序为4行x6列。"""
        if not (self.cc_enabled and self.cc_corners_norm and self.current_image and self.current_image.array is not None):
            return None
        
        try:
            # 使用新的提取方法获取RGB值
            rgb_array = self._extract_colorchecker_patches()
            if rgb_array is None:
                return None
            
            # 转换为XYZ
            from colour import RGB_COLOURSPACES, RGB_to_XYZ
            xyz_list = []
            
            for rgb in rgb_array:
                try:
                    # 假设预览为 DisplayP3
                    cs = RGB_COLOURSPACES.get('Display P3')
                    XYZ = RGB_to_XYZ(rgb, cs.whitepoint, cs.whitepoint, cs.matrix_RGB_to_XYZ)
                except Exception:
                    # 如果转换失败，使用原始RGB值
                    XYZ = rgb
                
                xyz_list.append(XYZ.astype(np.float32))
            
            return np.stack(xyz_list, axis=0)
            
        except Exception as e:
            print(f"读取色卡XYZ失败: {e}")
            return None

    def _extract_colorchecker_patches(self):
        """提取色卡选择器中的24个色块数据，返回RGB值数组"""
        if not (self.cc_enabled and self.cc_corners_norm and self.current_image and self.current_image.array is not None):
            return None
        
        try:
            import numpy as np
            import cv2
            
            # 获取图像数组
            arr = self.current_image.array
            if arr.dtype == np.uint8:
                arr = arr.astype(np.float32) / 255.0
            else:
                arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
            
            # 获取图像尺寸
            H_img, W_img = arr.shape[:2]
            
            # 转换归一化坐标到显示坐标
            display_corners = self._norm_to_display_coords(self.cc_corners_norm)
            if not display_corners:
                return None
            
            # 计算透视变换矩阵
            src = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
            dst = np.array(display_corners, dtype=np.float32)
            H = cv2.getPerspectiveTransform(src, dst)
            
            # 色块参数
            margin = 0.18
            rgb_list = []
            
            # 提取4x6色块
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
                    
                    # 提取色块的平均RGB值
                    m = mask.astype(bool)
                    if not np.any(m):
                        rgb = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                    else:
                        rgb = arr[m].reshape(-1, arr.shape[2]).mean(axis=0)
                    
                    rgb_list.append(rgb.astype(np.float32))
            
            return np.stack(rgb_list, axis=0)
            
        except Exception as e:
            print(f"提取色卡色块失败: {e}")
            return None

    # ===== 色卡选择器变换操作 =====
    def flip_colorchecker_horizontal(self):
        """水平翻转色卡选择器"""
        if not (self.cc_enabled and self.cc_corners_norm and self.current_image and self.current_image.array is not None):
            return
        
        try:
            # 在归一化空间进行水平翻转: x_norm' = 1.0 - x_norm, y_norm' = y_norm
            new_corners = []
            for (x_norm, y_norm) in self.cc_corners_norm:
                new_x_norm = 1.0 - x_norm
                new_corners.append((new_x_norm, y_norm))
            self.cc_corners_norm = new_corners
            self._update_display()
        except Exception as e:
            print(f"水平翻转色卡失败: {e}")

    def flip_colorchecker_vertical(self):
        """竖直翻转色卡选择器"""
        if not (self.cc_enabled and self.cc_corners_norm and self.current_image and self.current_image.array is not None):
            return
        
        try:
            # 在归一化空间进行竖直翻转: x_norm' = x_norm, y_norm' = 1.0 - y_norm
            new_corners = []
            for (x_norm, y_norm) in self.cc_corners_norm:
                new_y_norm = 1.0 - y_norm
                new_corners.append((x_norm, new_y_norm))
            self.cc_corners_norm = new_corners
            self._update_display()
        except Exception as e:
            print(f"竖直翻转色卡失败: {e}")

    def rotate_colorchecker_left(self):
        """左旋转色卡选择器90度"""
        if not (self.cc_enabled and self.cc_corners_norm and self.current_image and self.current_image.array is not None):
            return
        
        try:
            # 在归一化空间进行左旋90度: x_norm' = y_norm, y_norm' = 1.0 - x_norm
            new_corners = []
            for (x_norm, y_norm) in self.cc_corners_norm:
                new_x_norm = y_norm
                new_y_norm = 1.0 - x_norm
                new_corners.append((new_x_norm, new_y_norm))
            self.cc_corners_norm = new_corners
            self._update_display()
        except Exception as e:
            print(f"左旋转色卡失败: {e}")

    def rotate_colorchecker_right(self):
        """右旋转色卡选择器90度"""
        if not (self.cc_enabled and self.cc_corners_norm and self.current_image and self.current_image.array is not None):
            return
        
        try:
            # 在归一化空间进行右旋90度: x_norm' = 1.0 - y_norm, y_norm' = x_norm
            new_corners = []
            for (x_norm, y_norm) in self.cc_corners_norm:
                new_x_norm = 1.0 - y_norm
                new_y_norm = x_norm
                new_corners.append((new_x_norm, new_y_norm))
            self.cc_corners_norm = new_corners
            self._update_display()
        except Exception as e:
            print(f"右旋转色卡失败: {e}")

    # ===== 中性色定义相关方法 =====

    def enter_neutral_point_selection_mode(self):
        """进入中性点选择模式"""
        self.neutral_point_selection_mode = True
        # 设置十字光标
        self.image_label.setCursor(QCursor(Qt.CursorShape.CrossCursor))

        self._update_display()

    def exit_neutral_point_selection_mode(self):
        """退出中性点选择模式"""
        self.neutral_point_selection_mode = False
        # 恢复默认光标
        self.image_label.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        self._update_display()

    def clear_neutral_point(self):
        """清除选定的中性点"""
        self.neutral_point_norm = None
        self._update_display()

    def _draw_neutral_point_overlay(self, painter: QPainter):
        """绘制中性点标记（靶心样式）"""
        if not self.neutral_point_norm or not self.current_image or self.current_image.array is None:
            return

        try:
            # preview归一化坐标 → 图像坐标
            # painter已经应用了transform（translate + scale），直接使用图像坐标
            norm_x, norm_y = self.neutral_point_norm
            img_h, img_w = self.current_image.array.shape[:2]

            # 直接使用图像坐标，让painter的transform处理缩放和平移
            x = norm_x * img_w
            y = norm_y * img_h

            # 绘制靶心标记
            painter.save()

            # 外圈（半透明白色）
            painter.setPen(QPen(QColor(255, 255, 255, 180), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(x, y), 12, 12)

            # 中圈（半透明红色）
            painter.setPen(QPen(QColor(255, 0, 0, 200), 2))
            painter.drawEllipse(QPointF(x, y), 8, 8)

            # 内圈（实心红色）
            painter.setBrush(QColor(255, 0, 0, 180))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(x, y), 3, 3)

            # 十字线
            painter.setPen(QPen(QColor(255, 255, 255, 200), 1))
            painter.drawLine(QPointF(x - 15, y), QPointF(x - 5, y))  # 左
            painter.drawLine(QPointF(x + 5, y), QPointF(x + 15, y))  # 右
            painter.drawLine(QPointF(x, y - 15), QPointF(x, y - 5))  # 上
            painter.drawLine(QPointF(x, y + 5), QPointF(x, y + 15))  # 下

            painter.restore()

        except Exception as e:
            print(f"绘制中性点标记失败: {e}")

    # ===== 旋转锚点支持代码已移除 =====
    # 简化旋转逻辑，旋转后自动适应窗口显示
    # cc_corners_norm 保持相对物理原图不变，显示/提取时根据 orientation 自动应用旋转

    def rotate_left(self):
        """左旋90度"""
        if self.current_image and self.current_image.array is not None:
            # 直接通知主窗口执行旋转，不使用锚点机制
            # cc_corners_norm 保持相对物理原图不变，显示时会自动应用旋转
            self.image_rotated.emit(1)  # 1表示左旋

    def rotate_right(self):
        """右旋90度"""
        if self.current_image and self.current_image.array is not None:
            # 直接通知主窗口执行旋转，不使用锚点机制
            # cc_corners_norm 保持相对物理原图不变，显示时会自动应用旋转
            self.image_rotated.emit(-1)  # -1表示右旋
            
    def keyPressEvent(self, event):
        """键盘事件处理"""
        if event.key() in (Qt.Key.Key_Control, Qt.Key.Key_Meta):
            if not self._edit_modifier_down:
                self._edit_modifier_down = True
                # 立即刷新光标/悬停
                try:
                    pos = self.image_label.mapFromGlobal(QCursor.pos())
                    self._handle_crop_hover_detection(pos)
                except Exception:
                    pass
        elif event.key() == Qt.Key.Key_Left:
            self.rotate_left()
        elif event.key() == Qt.Key.Key_Right:
            self.rotate_right()
        elif event.key() == Qt.Key.Key_0:  # 数字0键重置视图
            self.reset_view()
        elif event.key() == Qt.Key.Key_C:  # C键居中
            self.center_image()
        elif event.key() == Qt.Key.Key_Escape:
            # 退出裁剪模式
            if self._crop_mode or self._crop_dragging:
                self._crop_mode = False
                self._crop_dragging = False
                self._crop_start_norm = None
                self._crop_current_norm = None
                self.image_label.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
                self._update_display()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        try:
            if event.key() in (Qt.Key.Key_Control, Qt.Key.Key_Meta):
                if self._edit_modifier_down:
                    self._edit_modifier_down = False
                    # 退出编辑修饰键，复位相关状态
                    self._active_edit_crop_id = None
                    if self._moving_overlay_active:
                        # 松开修饰键也结束移动
                        self._moving_overlay_active = False
                    self._clear_crop_hover_state()
                    try:
                        pos = self.image_label.mapFromGlobal(QCursor.pos())
                        self._handle_crop_hover_detection(pos)
                    except Exception:
                        pass
                return
        except Exception:
            pass
        super().keyReleaseEvent(event)

    # ===== 信号：裁剪交互 =====


    def _image_rect_to_norm(self, img_rect):
        """将显示图像矩形转换为原图归一化坐标（基于原图逻辑坐标系）"""
        try:
            if not self.current_image or self.current_image.array is None:
                return None
            x, y, w, h = img_rect
            if w <= 0 or h <= 0:
                return None
            
            # 关键思想：通过鼠标坐标直接获取原图归一化坐标
            # 将矩形的四个角点转换为原图坐标，然后计算边界框
            
            # 矩形的四个角点（显示坐标）
            points = [
                QPoint(int(x), int(y)),           # 左上
                QPoint(int(x + w), int(y)),       # 右上  
                QPoint(int(x), int(y + h)),       # 左下
                QPoint(int(x + w), int(y + h))    # 右下
            ]
            
            # 转换到原图坐标
            orig_points = []
            for pt in points:
                orig_pt = self._display_to_original_point(pt)
                if orig_pt is None:
                    return None
                orig_points.append(orig_pt)
            
            # 计算原图坐标系下的边界框
            xs = [pt[0] for pt in orig_points]
            ys = [pt[1] for pt in orig_points]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            
            # 获取原图尺寸进行归一化
            md = self.current_image.metadata or {}
            src_wh = md.get('source_wh', None)
            if not src_wh:
                return None
            src_w, src_h = src_wh
            
            # 归一化到[0,1]
            norm_x = min_x / src_w
            norm_y = min_y / src_h
            norm_w = (max_x - min_x) / src_w
            norm_h = (max_y - min_y) / src_h
            
            return (norm_x, norm_y, norm_w, norm_h)
            
        except Exception:
            return None

    def _clamp_img_rect(self, img_rect):
        """将图像坐标矩形 clamp 到当前图像边界，并应用最小尺寸（如8px）。"""
        try:
            if not self.current_image or self.current_image.array is None:
                return None
            x, y, w, h = img_rect
            img_h, img_w = self.current_image.array.shape[:2]
            # 归一化坐标 → clamp 算子在像素域
            x = max(0.0, min(float(img_w), float(x)))
            y = max(0.0, min(float(img_h), float(y)))
            w = max(0.0, min(float(img_w) - x, float(w)))
            h = max(0.0, min(float(img_h) - y, float(h)))
            # 最小尺寸（8px）
            MIN_PIX = 8.0
            if w < MIN_PIX or h < MIN_PIX:
                return None
            return (x, y, w, h)
        except Exception:
            return None

    # ===== 裁剪框编辑功能 =====
    def _translate_edit_mode_for_rotation(self, mode: CropEditMode, orientation: int) -> CropEditMode:
        """
        Translate display-space CropEditMode to original-space CropEditMode based on image orientation.
        
        IMPORTANT: orientation represents how much the image has been rotated FROM the original.
        - orientation = 90 means image is currently 90° CCW from original
        - This means what user sees as "top" is actually the original "right" edge
        
        Args:
            mode: CropEditMode detected in display coordinates (what user visually sees)
            orientation: Final orientation angle in degrees (0, 90, 180, 270)
            
        Returns:
            CropEditMode translated for original image coordinates (what gets semantically edited)
        """
        print(f"🔧 TRANSLATE DEBUG: Input mode={mode}, orientation={orientation}")
        
        if orientation % 360 == 0:
            print(f"🔧 TRANSLATE DEBUG: No rotation, returning original mode={mode}")
            return mode
        
        # Normalize rotation to 0, 90, 180, 270
        k = (orientation // 90) % 4
        
        print(f"🔧 TRANSLATE DEBUG: k={k} (rotation case)")
        
        if k == 1:  # Image is 90° CCW from original (user clicked "left rotate")
            print(f"🔧 TRANSLATE DEBUG: Using k=1 (90° CCW) mappings")
            # Visual → Semantic mapping (VERIFIED with np.rot90 test):
            # After 90° CCW: Original LEFT→Visual TOP, Original TOP→Visual RIGHT, etc.
            # User sees "top" → actually original "left" edge
            # User sees "right" → actually original "top" edge  
            # User sees "bottom" → actually original "right" edge
            # User sees "left" → actually original "bottom" edge
            translation_map = {
                CropEditMode.DRAG_TOP: CropEditMode.DRAG_RIGHT,        # Visual top → Original right
                CropEditMode.DRAG_RIGHT: CropEditMode.DRAG_BOTTOM,       # Visual right → Original bottom
                CropEditMode.DRAG_BOTTOM: CropEditMode.DRAG_LEFT,    # Visual bottom → Original left
                CropEditMode.DRAG_LEFT: CropEditMode.DRAG_TOP,     # Visual left → Original top
                CropEditMode.DRAG_TOP_LEFT: CropEditMode.DRAG_TOP_RIGHT,     # Visual top-left → Original top-right
                CropEditMode.DRAG_TOP_RIGHT: CropEditMode.DRAG_BOTTOM_RIGHT,       # Visual top-right → Original bottom-right
                CropEditMode.DRAG_BOTTOM_RIGHT: CropEditMode.DRAG_BOTTOM_LEFT,   # Visual bottom-right → Original bottom-left
                CropEditMode.DRAG_BOTTOM_LEFT: CropEditMode.DRAG_TOP_LEFT, # Visual bottom-left → Original top-left
            }
        elif k == 2:  # Image is 180° from original
            print(f"🔧 TRANSLATE DEBUG: Using k=2 (180°) mappings")
            # Visual → Semantic mapping:
            # User sees "top" → actually original "bottom" edge
            # User sees "right" → actually original "left" edge
            # User sees "bottom" → actually original "top" edge
            # User sees "left" → actually original "right" edge
            translation_map = {
                CropEditMode.DRAG_TOP: CropEditMode.DRAG_BOTTOM,      # Visual top → Original bottom
                CropEditMode.DRAG_RIGHT: CropEditMode.DRAG_LEFT,      # Visual right → Original left
                CropEditMode.DRAG_BOTTOM: CropEditMode.DRAG_TOP,      # Visual bottom → Original top
                CropEditMode.DRAG_LEFT: CropEditMode.DRAG_RIGHT,      # Visual left → Original right
                CropEditMode.DRAG_TOP_LEFT: CropEditMode.DRAG_BOTTOM_RIGHT,    # Visual top-left → Original bottom-right
                CropEditMode.DRAG_TOP_RIGHT: CropEditMode.DRAG_BOTTOM_LEFT,    # Visual top-right → Original bottom-left  
                CropEditMode.DRAG_BOTTOM_RIGHT: CropEditMode.DRAG_TOP_LEFT,    # Visual bottom-right → Original top-left
                CropEditMode.DRAG_BOTTOM_LEFT: CropEditMode.DRAG_TOP_RIGHT,    # Visual bottom-left → Original top-right
            }
        elif k == 3:  # Image is 270° CCW from original (equivalent to 90° CW, user clicked "right rotate")
            print(f"🔧 TRANSLATE DEBUG: Using k=3 (90° CW) mappings")
            # Visual → Semantic mapping (VERIFIED with np.rot90 test):
            # After 90° CW: Original RIGHT→Visual TOP, Original BOTTOM→Visual RIGHT, etc.
            # User sees "top" → actually original "right" edge
            # User sees "right" → actually original "bottom" edge
            # User sees "bottom" → actually original "left" edge
            # User sees "left" → actually original "top" edge
            translation_map = {
                CropEditMode.DRAG_TOP: CropEditMode.DRAG_LEFT,       # Visual top → Original left
                CropEditMode.DRAG_RIGHT: CropEditMode.DRAG_TOP,    # Visual right → Original top
                CropEditMode.DRAG_BOTTOM: CropEditMode.DRAG_RIGHT,     # Visual bottom → Original right
                CropEditMode.DRAG_LEFT: CropEditMode.DRAG_BOTTOM,        # Visual left → Original bottom
                CropEditMode.DRAG_TOP_LEFT: CropEditMode.DRAG_BOTTOM_LEFT,       # Visual top-left → Original bottom-left
                CropEditMode.DRAG_TOP_RIGHT: CropEditMode.DRAG_TOP_LEFT,   # Visual top-right → Original top-left
                CropEditMode.DRAG_BOTTOM_RIGHT: CropEditMode.DRAG_TOP_RIGHT, # Visual bottom-right → Original top-right
                CropEditMode.DRAG_BOTTOM_LEFT: CropEditMode.DRAG_BOTTOM_RIGHT,     # Visual bottom-left → Original bottom-right
            }
        else:
            print(f"🔧 TRANSLATE DEBUG: Unknown k value, returning original mode={mode}")
            return mode
        
        translated_result = translation_map.get(mode, mode)
        print(f"🔧 TRANSLATE DEBUG: Final result: {mode} → {translated_result}")
        return translated_result

    def _get_crop_interaction_zone(self, mouse_pos: QPoint, rect_norm_override: Optional[Tuple[float, float, float, float]] = None) -> CropEditMode:
        """检测鼠标位置对应的裁剪交互区域。可选传入指定裁剪框rect进行检测。"""
        if not self.current_image:
            return CropEditMode.NONE
            
        # 获取裁剪框在图像像素坐标系中的位置
        target_rect_norm = rect_norm_override if rect_norm_override is not None else self._crop_overlay_norm
        if not target_rect_norm:
            return CropEditMode.NONE
        img_rect = self._norm_to_image_rect(target_rect_norm)
        if not img_rect:
            return CropEditMode.NONE
            
        x, y, w, h = img_rect
        
        # 转换鼠标坐标到图像坐标系（考虑zoom和pan）
        try:
            # 鼠标在canvas上的位置
            mx, my = mouse_pos.x(), mouse_pos.y()
            
            # 反向变换：从canvas坐标到图像坐标
            # canvas坐标 = (图像坐标 * zoom) + pan
            # 图像坐标 = (canvas坐标 - pan) / zoom
            img_x = (mx - self.pan_x) / self.zoom_factor
            img_y = (my - self.pan_y) / self.zoom_factor
            
            # 检测容差（固定屏幕像素大小，确保可靠的命中检测）
            EDGE_TOLERANCE = 12.0  # 边缘检测容差（屏幕像素）
            CORNER_TOLERANCE = 16.0  # 角点检测容差（屏幕像素）
            
            # 转换为图像坐标系下的容差
            EDGE_TOLERANCE_IMG = EDGE_TOLERANCE / self.zoom_factor
            CORNER_TOLERANCE_IMG = CORNER_TOLERANCE / self.zoom_factor
            
            # 判断是否在裁剪框附近
            left, right = x, x + w
            top, bottom = y, y + h
            
            # Detect interaction zone in display coordinates
            detected_mode = CropEditMode.NONE
            
            # 先检测角点（优先级最高）
            if (abs(img_x - left) <= CORNER_TOLERANCE_IMG and 
                abs(img_y - top) <= CORNER_TOLERANCE_IMG):
                detected_mode = CropEditMode.DRAG_TOP_LEFT
            elif (abs(img_x - right) <= CORNER_TOLERANCE_IMG and 
                  abs(img_y - top) <= CORNER_TOLERANCE_IMG):
                detected_mode = CropEditMode.DRAG_TOP_RIGHT
            elif (abs(img_x - left) <= CORNER_TOLERANCE_IMG and 
                  abs(img_y - bottom) <= CORNER_TOLERANCE_IMG):
                detected_mode = CropEditMode.DRAG_BOTTOM_LEFT
            elif (abs(img_x - right) <= CORNER_TOLERANCE_IMG and 
                  abs(img_y - bottom) <= CORNER_TOLERANCE_IMG):
                detected_mode = CropEditMode.DRAG_BOTTOM_RIGHT
            # 再检测边缘
            elif (left <= img_x <= right and 
                  abs(img_y - top) <= EDGE_TOLERANCE_IMG):
                detected_mode = CropEditMode.DRAG_TOP
            elif (left <= img_x <= right and 
                  abs(img_y - bottom) <= EDGE_TOLERANCE_IMG):
                detected_mode = CropEditMode.DRAG_BOTTOM
            elif (top <= img_y <= bottom and 
                  abs(img_x - left) <= EDGE_TOLERANCE_IMG):
                detected_mode = CropEditMode.DRAG_LEFT
            elif (top <= img_y <= bottom and 
                  abs(img_x - right) <= EDGE_TOLERANCE_IMG):
                detected_mode = CropEditMode.DRAG_RIGHT
            
            # If no interaction detected, return NONE
            if detected_mode == CropEditMode.NONE:
                return CropEditMode.NONE
            
            # Apply rotation translation to get semantically correct edit mode
            # Get orientation from image metadata
            md = self.current_image.metadata or {}
            focused = md.get('crop_focused', False)
            
            if focused:
                # In focused mode, use crop-specific orientation
                crop_instance = md.get('crop_instance')
                orientation = crop_instance.orientation if crop_instance else 0
            else:
                # In contact sheet mode, use global orientation
                orientation = md.get('orientation', 0)
            
            # Translate the detected mode based on rotation
            translated_mode = self._translate_edit_mode_for_rotation(detected_mode, orientation)
            return translated_mode
            
        except Exception:
            return CropEditMode.NONE

    def _handle_crop_hover_detection(self, mouse_pos: QPoint):
        """独立的裁剪框悬停检测，不受其他交互状态影响"""
        # 中性点选择模式优先级最高，不改变光标
        if self.neutral_point_selection_mode:
            return

        # 原图模式下的多裁剪悬停
        if self._show_all_crops:
            old_hovering = self._hovering_crop_id
            if self._edit_modifier_down:
                # 编辑修饰键按下：优先检测边/角
                cid, zone = self._get_crop_edit_target_at_position(mouse_pos)
                if zone != CropEditMode.NONE and cid is not None:
                    self._hovering_crop_id = cid
                    if zone != self._crop_edit_hover_mode:
                        self._crop_edit_hover_mode = zone
                        self._update_crop_cursor(zone)
                    return
                # 未命中边/角：检测是否在某个裁剪框内部
                self._crop_edit_hover_mode = CropEditMode.NONE
                self._hovering_crop_id = self._get_crop_at_position(mouse_pos)
                if self._hovering_crop_id:
                    self.image_label.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
                else:
                    # 编辑态的默认光标
                    self.image_label.setCursor(QCursor(Qt.CursorShape.CrossCursor))
                return
            # 未按编辑键：更新悬停与光标（悬停框显示抓手）
            self._crop_edit_hover_mode = CropEditMode.NONE
            self._hovering_crop_id = self._get_crop_at_position(mouse_pos)
            if old_hovering != self._hovering_crop_id:
                self.image_label.update()
            if not (self._crop_mode or self.dragging):
                if self._hovering_crop_id:
                    self.image_label.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
                else:
                    self.image_label.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        # 聚焦模式：不显示虚线框也不响应边缘编辑
        elif (not self._show_all_crops):
            if not (self._crop_mode or self.dragging or self.cc_enabled):
                self._update_crop_cursor(CropEditMode.NONE)
        else:
            # 没有裁剪框或正在编辑时，确保光标为默认状态
            if self._crop_edit_hover_mode != CropEditMode.NONE:
                self._crop_edit_hover_mode = CropEditMode.NONE
                # 只有在非交互状态下才重置光标
                if not (self._crop_mode or self.dragging or self.cc_enabled):
                    self._update_crop_cursor(CropEditMode.NONE)

    def _update_crop_cursor(self, edit_mode: CropEditMode):
        """根据编辑模式更新光标样式，考虑优先级和图像旋转"""
        
        # 如果正在其他交互中，不改变光标
        if self._crop_mode:
            return  # 保持裁剪模式的十字光标
        
        if self.dragging:
            return  # 保持拖拽时的光标
            
        if self.cc_enabled and self.cc_drag_idx is not None:
            return  # 保持色卡拖拽时的光标
        
        # 获取旋转角度用于光标调整
        display_edit_mode = self._get_display_cursor_mode(edit_mode)
        
        # 应用裁剪编辑的光标（基于显示后的视觉效果）
        cursor_map = {
            CropEditMode.NONE: Qt.CursorShape.ArrowCursor,
            CropEditMode.DRAG_TOP: Qt.CursorShape.SizeVerCursor,
            CropEditMode.DRAG_BOTTOM: Qt.CursorShape.SizeVerCursor,
            CropEditMode.DRAG_LEFT: Qt.CursorShape.SizeHorCursor,
            CropEditMode.DRAG_RIGHT: Qt.CursorShape.SizeHorCursor,
            CropEditMode.DRAG_TOP_LEFT: Qt.CursorShape.SizeFDiagCursor,
            CropEditMode.DRAG_BOTTOM_RIGHT: Qt.CursorShape.SizeFDiagCursor,
            CropEditMode.DRAG_TOP_RIGHT: Qt.CursorShape.SizeBDiagCursor,
            CropEditMode.DRAG_BOTTOM_LEFT: Qt.CursorShape.SizeBDiagCursor,
        }
        
        cursor = cursor_map.get(display_edit_mode, Qt.CursorShape.ArrowCursor)
        self.image_label.setCursor(QCursor(cursor))
    
    def _get_display_cursor_mode(self, semantic_edit_mode: CropEditMode) -> CropEditMode:
        """
        Convert semantic edit mode to display cursor mode based on image rotation.
        
        IMPORTANT: This is the INVERSE of _translate_edit_mode_for_rotation.
        - Semantic mode: what actually gets edited in original coordinates  
        - Display mode: what cursor the user should see visually after rotation
        
        Example: If semantic mode is DRAG_RIGHT (edit original right edge),
        and image is 90° CCW from original, the user visually sees this as the bottom edge,
        so we return DRAG_BOTTOM for proper cursor display.
        
        Args:
            semantic_edit_mode: The CropEditMode for original image coordinates
            
        Returns:
            The CropEditMode for display cursor (what user visually sees)
        """
        if not self.current_image or semantic_edit_mode == CropEditMode.NONE:
            return semantic_edit_mode
            
        # Get orientation from image metadata
        md = self.current_image.metadata or {}
        focused = md.get('crop_focused', False)
        
        if focused:
            # In focused mode, use crop-specific orientation
            crop_instance = md.get('crop_instance')
            orientation = crop_instance.orientation if crop_instance else 0
        else:
            # In normal mode, use image orientation
            orientation = md.get('orientation', 0)
        
        # Normalize rotation to 0, 90, 180, 270
        k = (orientation // 90) % 4
        
        if k == 0:
            return semantic_edit_mode
        elif k == 1:  # Image is 90° CCW from original - INVERSE mapping
            # Semantic → Visual mapping (inverse of corrected forward mapping):
            # Original "left" → User sees "top" (inverse of visual top → original left)
            # Original "top" → User sees "right" (inverse of visual right → original top)
            # Original "right" → User sees "bottom" (inverse of visual bottom → original right)
            # Original "bottom" → User sees "left" (inverse of visual left → original bottom)
            reverse_map = {
                CropEditMode.DRAG_LEFT: CropEditMode.DRAG_TOP,            # Original left → Visual top
                CropEditMode.DRAG_TOP: CropEditMode.DRAG_RIGHT,           # Original top → Visual right  
                CropEditMode.DRAG_RIGHT: CropEditMode.DRAG_BOTTOM,        # Original right → Visual bottom
                CropEditMode.DRAG_BOTTOM: CropEditMode.DRAG_LEFT,         # Original bottom → Visual left
                CropEditMode.DRAG_BOTTOM_LEFT: CropEditMode.DRAG_TOP_LEFT,        # Original bottom-left → Visual top-left
                CropEditMode.DRAG_TOP_LEFT: CropEditMode.DRAG_TOP_RIGHT,          # Original top-left → Visual top-right
                CropEditMode.DRAG_TOP_RIGHT: CropEditMode.DRAG_BOTTOM_RIGHT,      # Original top-right → Visual bottom-right
                CropEditMode.DRAG_BOTTOM_RIGHT: CropEditMode.DRAG_BOTTOM_LEFT,    # Original bottom-right → Visual bottom-left
            }
        elif k == 2:  # Image is 180° from original - INVERSE mapping  
            # Semantic → Visual mapping:
            # Original "bottom" → User sees "top" (because image rotated 180°)
            # Original "left" → User sees "right"
            # Original "top" → User sees "bottom" 
            # Original "right" → User sees "left"
            reverse_map = {
                CropEditMode.DRAG_BOTTOM: CropEditMode.DRAG_TOP,          # Original bottom → Visual top
                CropEditMode.DRAG_LEFT: CropEditMode.DRAG_RIGHT,          # Original left → Visual right
                CropEditMode.DRAG_TOP: CropEditMode.DRAG_BOTTOM,          # Original top → Visual bottom
                CropEditMode.DRAG_RIGHT: CropEditMode.DRAG_LEFT,          # Original right → Visual left
                CropEditMode.DRAG_BOTTOM_RIGHT: CropEditMode.DRAG_TOP_LEFT,       # Original bottom-right → Visual top-left
                CropEditMode.DRAG_BOTTOM_LEFT: CropEditMode.DRAG_TOP_RIGHT,       # Original bottom-left → Visual top-right  
                CropEditMode.DRAG_TOP_LEFT: CropEditMode.DRAG_BOTTOM_RIGHT,       # Original top-left → Visual bottom-right
                CropEditMode.DRAG_TOP_RIGHT: CropEditMode.DRAG_BOTTOM_LEFT,       # Original top-right → Visual bottom-left
            }
        elif k == 3:  # Image is 270° CCW (90° CW) from original - INVERSE mapping
            # Semantic → Visual mapping (inverse of corrected forward mapping):
            # Original "right" → User sees "top" (inverse of visual top → original right)
            # Original "bottom" → User sees "right" (inverse of visual right → original bottom)
            # Original "left" → User sees "bottom" (inverse of visual bottom → original left)
            # Original "top" → User sees "left" (inverse of visual left → original top)
            reverse_map = {
                CropEditMode.DRAG_RIGHT: CropEditMode.DRAG_TOP,           # Original right → Visual top
                CropEditMode.DRAG_BOTTOM: CropEditMode.DRAG_RIGHT,        # Original bottom → Visual right
                CropEditMode.DRAG_LEFT: CropEditMode.DRAG_BOTTOM,         # Original left → Visual bottom
                CropEditMode.DRAG_TOP: CropEditMode.DRAG_LEFT,            # Original top → Visual left
                CropEditMode.DRAG_TOP_RIGHT: CropEditMode.DRAG_TOP_LEFT,          # Original top-right → Visual top-left
                CropEditMode.DRAG_BOTTOM_RIGHT: CropEditMode.DRAG_TOP_RIGHT,      # Original bottom-right → Visual top-right
                CropEditMode.DRAG_BOTTOM_LEFT: CropEditMode.DRAG_BOTTOM_RIGHT,    # Original bottom-left → Visual bottom-right
                CropEditMode.DRAG_TOP_LEFT: CropEditMode.DRAG_BOTTOM_LEFT,        # Original top-left → Visual bottom-left
            }
        else:
            return semantic_edit_mode
            
        return reverse_map.get(semantic_edit_mode, semantic_edit_mode)

    def _clear_crop_hover_state(self):
        """清理裁剪悬停状态，在适当时机调用"""
        if self._crop_edit_hover_mode != CropEditMode.NONE:
            self._crop_edit_hover_mode = CropEditMode.NONE
            # 仅在非交互状态下重置光标
            if not (self._crop_mode or self.dragging or self.cc_enabled):
                self.image_label.setCursor(QCursor(Qt.CursorShape.ArrowCursor))

    # ===== 多裁剪：边/角命中检测 =====
    def _get_crop_edit_target_at_position(self, mouse_pos: QPoint) -> Tuple[Optional[str], CropEditMode]:
        """返回鼠标位置命中的裁剪及边/角编辑模式。(crop_id, mode)"""
        if not (self._show_all_crops and self._all_crops):
            return (None, CropEditMode.NONE)
        try:
            # 优先检测当前悬停框，提高连贯性
            candidate_ids: List[str] = []
            if self._hovering_crop_id:
                candidate_ids.append(self._hovering_crop_id)
            for info in self._all_crops:
                cid = info['id']
                if cid not in candidate_ids:
                    candidate_ids.append(cid)
            for cid in candidate_ids:
                rect_norm = None
                for info in self._all_crops:
                    if info['id'] == cid:
                        rect_norm = info['rect_norm']
                        break
                if rect_norm is None:
                    continue
                zone = self._get_crop_interaction_zone(mouse_pos, rect_norm)
                if zone != CropEditMode.NONE:
                    return (cid, zone)
        except Exception:
            pass
        return (None, CropEditMode.NONE)

    # ===== 标准坐标转换函数（支持CropInstance模型） =====
    def _get_current_crop_context(self) -> Optional[CropInstance]:
        """获取当前crop上下文"""
        if self.current_image and self.current_image.metadata:
            return self.current_image.metadata.get('crop_instance')
        return None

    def _display_to_original_point(self, display_point: QPoint) -> Optional[Tuple[float, float]]:
        """显示坐标 → 原图像素坐标（纯净的坐标转换，无变换概念）"""
        if not self.current_image:
            return None
            
        try:
            # Step 1: Canvas坐标 → 当前显示图像像素坐标
            # 兼容 QPoint/QPointF/tuple
            try:
                mx, my = float(display_point.x()), float(display_point.y())
            except Exception:
                try:
                    mx, my = float(display_point[0]), float(display_point[1])
                except Exception:
                    return None
            img_x = (mx - self.pan_x) / self.zoom_factor
            img_y = (my - self.pan_y) / self.zoom_factor
            
            # Step 2: 获取metadata
            md = self.current_image.metadata or {}
            source_wh = md.get('source_wh')
            if not source_wh:
                # 无原图信息，直接返回当前坐标（假设就是原图）
                return (img_x, img_y)
            
            src_w, src_h = source_wh
            focused = md.get('crop_focused', False)
            
            if focused:
                # === 聚焦状态：当前显示的是某个crop区域 ===
                crop_instance = md.get('crop_instance')
                if not crop_instance:
                    return (img_x, img_y)
                
                # 获取crop区域在原图中的像素范围
                x, y, w, h = crop_instance.rect_norm
                crop_left = x * src_w
                crop_top = y * src_h
                crop_width = w * src_w
                crop_height = h * src_h
                
                # 当前显示图像的尺寸
                disp_h, disp_w = self.current_image.array.shape[:2]
                
                # 处理crop的独立orientation
                if crop_instance.orientation % 360 != 0:
                    # 将显示坐标映射到crop前的坐标（逆旋转）
                    crop_x, crop_y = self._reverse_rotate_point(
                        img_x, img_y, disp_w, disp_h, crop_instance.orientation
                    )
                    # 映射回原图：crop坐标 → 原图坐标
                    orig_x = crop_left + (crop_x * crop_width / disp_w)
                    orig_y = crop_top + (crop_y * crop_height / disp_h)
                else:
                    # 无旋转：直接映射
                    orig_x = crop_left + (img_x * crop_width / disp_w)
                    orig_y = crop_top + (img_y * crop_height / disp_h)
                
                return (orig_x, orig_y)
            else:
                # === 非聚焦状态：当前显示的是完整图像（可能旋转） ===
                global_orientation = md.get('global_orientation', 0)
                disp_h, disp_w = self.current_image.array.shape[:2]
                
                if global_orientation % 360 != 0:
                    # 逆旋转到原图坐标
                    orig_x, orig_y = self._reverse_rotate_point(
                        img_x, img_y, disp_w, disp_h, global_orientation
                    )
                    # 缩放到原图尺寸
                    if global_orientation % 180 == 0:  # 0°或180°
                        orig_x = orig_x * src_w / disp_w
                        orig_y = orig_y * src_h / disp_h
                    else:  # 90°或270°（宽高交换）
                        orig_x = orig_x * src_w / disp_h
                        orig_y = orig_y * src_h / disp_w
                    return (orig_x, orig_y)
                else:
                    # 无旋转：直接缩放
                    orig_x = img_x * src_w / disp_w
                    orig_y = img_y * src_h / disp_h
                    return (orig_x, orig_y)
                
        except Exception:
            pass
        return None

    def _apply_rotate_point(self, x: float, y: float, img_w: int, img_h: int, orientation: int) -> Tuple[float, float]:
        """正向旋转点坐标"""
        if orientation % 360 == 0:
            return (x, y)
            
        # 标准化旋转角度
        k = (orientation // 90) % 4
        if k == 0:
            return (x, y)
        elif k == 1:  # 正向90°旋转（CCW 90°）
            # 原图(W_o,H_o) → 显示(H_o,W_o): (x_d, y_d) = (y_o, W_o - 1 - x_o)
            return (y, img_w - 1 - x)
        elif k == 2:  # 正向180°旋转
            return (img_w - 1 - x, img_h - 1 - y)
        elif k == 3:  # 正向270°旋转（CCW 270°/CW 90°）
            # 原图(W_o,H_o) → 显示(H_o,W_o): (x_d, y_d) = (H_o - 1 - y_o, x_o)
            return (img_h - 1 - y, x)
        else:
            return (x, y)

    def _reverse_rotate_point(self, x: float, y: float, img_w: int, img_h: int, orientation: int) -> Tuple[float, float]:
        """逆向旋转点坐标"""
        if orientation % 360 == 0:
            return (x, y)
            
        # 标准化旋转角度
        k = (orientation // 90) % 4
        if k == 0:
            return (x, y)
        elif k == 1:  # 逆向90°旋转（对应正向CCW 90°）
            # 原图(W_o,H_o) → 显示(H_o,W_o): (x_d, y_d) = (y_o, W_o - 1 - x_o)
            # 逆变换：x_o = W_o - 1 - y_d, y_o = x_d
            return (img_h - 1 - y, x)
        elif k == 2:  # 逆向180°旋转  
            return (img_w - 1 - x, img_h - 1 - y)
        elif k == 3:  # 逆向270°旋转（对应正向CCW 270°/CW 90°）
            # 原图(W_o,H_o) → 显示(H_o,W_o): (x_d, y_d) = (H_o - 1 - y_o, x_o)
            # 逆变换：x_o = y_d, y_o = H_o - 1 - x_d
            return (y, img_w - 1 - x)
        else:
            return (x, y)

    def _original_to_norm_rect(self, pixel_rect: Tuple[float, float, float, float]) -> Optional[Tuple[float, float, float, float]]:
        """原始图像像素坐标 → 归一化坐标"""
        if not self.current_image:
            return None
            
        try:
            md = self.current_image.metadata or {}
            source_wh = md.get('source_wh')
            if source_wh:
                src_w, src_h = source_wh
                x, y, w, h = pixel_rect
                return (x / src_w, y / src_h, w / src_w, h / src_h)
        except Exception:
            pass
        return None

    def _apply_crop_edit(self, mouse_pos: QPoint):
        """根据鼠标位置和当前编辑模式，实时更新裁剪框（支持CropInstance模型）"""
        if (self._crop_edit_mode == CropEditMode.NONE or 
            not self._crop_edit_start_rect or 
            not self.current_image):
            return
            
        try:
            # 检查是否在crop focused模式下，如果是则使用特殊处理
            md = self.current_image.metadata or {}
            if md.get('crop_focused', False):
                self._apply_crop_edit_focused(mouse_pos)
                return
            # 使用新的坐标转换：鼠标 → 原始图像坐标
            original_point = self._display_to_original_point(mouse_pos)
            if not original_point:
                return
            
            orig_x, orig_y = original_point
            
            # 获取原始图像尺寸
            md = self.current_image.metadata or {}
            source_wh = md.get('source_wh')
            if not source_wh:
                return
            src_w, src_h = source_wh
            
            # 获取初始裁剪框（归一化坐标）
            start_x, start_y, start_w, start_h = self._crop_edit_start_rect
            
            # 转换为原始图像像素坐标
            start_px = start_x * src_w
            start_py = start_y * src_h  
            start_pw = start_w * src_w
            start_ph = start_h * src_h
            
            # 根据编辑模式计算新的裁剪框（在原始图像坐标系中）
            new_px, new_py, new_pw, new_ph = start_px, start_py, start_pw, start_ph
            
            # 最小尺寸约束（原始图像像素）
            MIN_SIZE = 20.0
            
            # Apply rotation translation to edit mode
            current_orientation = md.get('orientation', 0)
            k = current_orientation // 90 if current_orientation in [0, 90, 180, 270] else 0
            translated_edit_mode = self._translate_edit_mode_for_rotation(self._crop_edit_mode, k)
            
            if translated_edit_mode == CropEditMode.DRAG_TOP:
                new_py = max(0, min(orig_y, start_py + start_ph - MIN_SIZE))
                new_ph = start_py + start_ph - new_py
            elif translated_edit_mode == CropEditMode.DRAG_BOTTOM:
                new_ph = max(MIN_SIZE, min(src_h - start_py, orig_y - start_py))
            elif translated_edit_mode == CropEditMode.DRAG_LEFT:
                new_px = max(0, min(orig_x, start_px + start_pw - MIN_SIZE))
                new_pw = start_px + start_pw - new_px
            elif translated_edit_mode == CropEditMode.DRAG_RIGHT:
                new_pw = max(MIN_SIZE, min(src_w - start_px, orig_x - start_px))
            elif translated_edit_mode == CropEditMode.DRAG_TOP_LEFT:
                new_px = max(0, min(orig_x, start_px + start_pw - MIN_SIZE))
                new_py = max(0, min(orig_y, start_py + start_ph - MIN_SIZE))
                new_pw = start_px + start_pw - new_px
                new_ph = start_py + start_ph - new_py
            elif translated_edit_mode == CropEditMode.DRAG_TOP_RIGHT:
                new_py = max(0, min(orig_y, start_py + start_ph - MIN_SIZE))
                new_pw = max(MIN_SIZE, min(src_w - start_px, orig_x - start_px))
                new_ph = start_py + start_ph - new_py
            elif translated_edit_mode == CropEditMode.DRAG_BOTTOM_LEFT:
                new_px = max(0, min(orig_x, start_px + start_pw - MIN_SIZE))
                new_pw = start_px + start_pw - new_px
                new_ph = max(MIN_SIZE, min(src_h - start_py, orig_y - start_py))
            elif translated_edit_mode == CropEditMode.DRAG_BOTTOM_RIGHT:
                new_pw = max(MIN_SIZE, min(src_w - start_px, orig_x - start_px))
                new_ph = max(MIN_SIZE, min(src_h - start_py, orig_y - start_py))
            
            # 转换回归一化坐标（相对于原始图像）
            new_rect_norm = self._original_to_norm_rect((new_px, new_py, new_pw, new_ph))
            if new_rect_norm:
                self._crop_overlay_norm = new_rect_norm
                # 立即重绘以显示变化
                self.image_label.update()
                
        except Exception as e:
            print(f"裁剪编辑计算错误: {e}")

    def _apply_crop_edit_multi(self, mouse_pos: QPoint):
        """原图模式下：对多裁剪中的某个框进行边/角编辑，直接修改 self._all_crops 对应项。"""
        if (self._crop_edit_mode == CropEditMode.NONE or 
            not self._crop_edit_start_rect or 
            not self.current_image or 
            self._active_edit_crop_id is None):
            return
        try:
            # 鼠标 → 原图像素坐标
            original_point = self._display_to_original_point(mouse_pos)
            if not original_point:
                return
            orig_x, orig_y = original_point
            md = self.current_image.metadata or {}
            source_wh = md.get('source_wh')
            if not source_wh:
                return
            src_w, src_h = source_wh
            sx, sy, sw, sh = self._crop_edit_start_rect
            start_px = sx * src_w
            start_py = sy * src_h
            start_pw = sw * src_w
            start_ph = sh * src_h
            new_px, new_py, new_pw, new_ph = start_px, start_py, start_pw, start_ph
            MIN_SIZE = 20.0
            
            # Apply rotation translation to edit mode
            current_orientation = md.get('orientation', 0)
            k = current_orientation // 90 if current_orientation in [0, 90, 180, 270] else 0
            translated_edit_mode = self._translate_edit_mode_for_rotation(self._crop_edit_mode, k)
            
            if translated_edit_mode == CropEditMode.DRAG_TOP:
                new_py = max(0, min(orig_y, start_py + start_ph - MIN_SIZE))
                new_ph = start_py + start_ph - new_py
            elif translated_edit_mode == CropEditMode.DRAG_BOTTOM:
                new_ph = max(MIN_SIZE, min(src_h - start_py, orig_y - start_py))
            elif translated_edit_mode == CropEditMode.DRAG_LEFT:
                new_px = max(0, min(orig_x, start_px + start_pw - MIN_SIZE))
                new_pw = start_px + start_pw - new_px
            elif translated_edit_mode == CropEditMode.DRAG_RIGHT:
                new_pw = max(MIN_SIZE, min(src_w - start_px, orig_x - start_px))
            elif translated_edit_mode == CropEditMode.DRAG_TOP_LEFT:
                new_px = max(0, min(orig_x, start_px + start_pw - MIN_SIZE))
                new_py = max(0, min(orig_y, start_py + start_ph - MIN_SIZE))
                new_pw = start_px + start_pw - new_px
                new_ph = start_py + start_ph - new_py
            elif translated_edit_mode == CropEditMode.DRAG_TOP_RIGHT:
                new_py = max(0, min(orig_y, start_py + start_ph - MIN_SIZE))
                new_pw = max(MIN_SIZE, min(src_w - start_px, orig_x - start_px))
                new_ph = start_py + start_ph - new_py
            elif translated_edit_mode == CropEditMode.DRAG_BOTTOM_LEFT:
                new_px = max(0, min(orig_x, start_px + start_pw - MIN_SIZE))
                new_pw = start_px + start_pw - new_px
                new_ph = max(MIN_SIZE, min(src_h - start_py, orig_y - start_py))
            elif translated_edit_mode == CropEditMode.DRAG_BOTTOM_RIGHT:
                new_pw = max(MIN_SIZE, min(src_w - start_px, orig_x - start_px))
                new_ph = max(MIN_SIZE, min(src_h - start_py, orig_y - start_py))
            # 回写到目标crop
            new_rect_norm = self._original_to_norm_rect((new_px, new_py, new_pw, new_ph))
            if new_rect_norm:
                for i, info in enumerate(self._all_crops):
                    if info['id'] == self._active_edit_crop_id:
                        self._all_crops[i]['rect_norm'] = new_rect_norm
                        break
                # 即时刷新
                self.image_label.update()
        except Exception as e:
            print(f"多裁剪编辑计算错误: {e}")
    
    def _apply_crop_edit_focused(self, mouse_pos: QPoint):
        """Crop focused模式下的编辑：在crop的局部坐标系中操作"""
        try:
            # Step 1: 鼠标 → 当前显示图像坐标
            mx, my = mouse_pos.x(), mouse_pos.y()
            img_x = (mx - self.pan_x) / self.zoom_factor
            img_y = (my - self.pan_y) / self.zoom_factor
            
            # Step 2: 当前显示图像的尺寸
            disp_h, disp_w = self.current_image.array.shape[:2]
            
            # Step 3: 转换为crop内的归一化坐标(0-1)
            local_x = img_x / disp_w
            local_y = img_y / disp_h
            
            # Step 4: 获取原始图像和crop信息
            md = self.current_image.metadata or {}
            crop_instance = md.get('crop_instance')
            source_wh = md.get('source_wh')
            if not crop_instance or not source_wh:
                return
                
            src_w, src_h = source_wh
            original_crop_rect = crop_instance.rect_norm  # 原始crop位置
            
            # Step 5: 获取编辑前的crop区域（这是我们要修改的）
            start_x, start_y, start_w, start_h = self._crop_edit_start_rect
            
            # Step 6: 在crop局部坐标系中计算新边界
            new_x, new_y, new_w, new_h = start_x, start_y, start_w, start_h
            
            # 最小尺寸约束（归一化）
            MIN_SIZE_NORM = 0.01  # 1%
            
            # Apply rotation translation to edit mode
            current_orientation = md.get('orientation', 0)
            k = current_orientation // 90 if current_orientation in [0, 90, 180, 270] else 0
            translated_edit_mode = self._translate_edit_mode_for_rotation(self._crop_edit_mode, k)
            
            # 关键：根据用户拖动的视觉边缘来调整crop区域
            if translated_edit_mode == CropEditMode.DRAG_TOP:
                # 用户拖动视觉上的"上边缘"
                # 在crop坐标系中，这对应调整crop的上边界
                delta_y = (local_y - 0.5) * start_h  # 相对于crop中心的偏移
                new_top = start_y + delta_y
                new_bottom = start_y + start_h
                new_y = max(0, min(new_top, new_bottom - MIN_SIZE_NORM))
                new_h = new_bottom - new_y
                
            elif translated_edit_mode == CropEditMode.DRAG_BOTTOM:
                delta_y = (local_y - 0.5) * start_h
                new_bottom = start_y + start_h + delta_y
                new_h = max(MIN_SIZE_NORM, min(1.0 - start_y, new_bottom - start_y))
                
            elif translated_edit_mode == CropEditMode.DRAG_LEFT:
                delta_x = (local_x - 0.5) * start_w
                new_left = start_x + delta_x
                new_right = start_x + start_w
                new_x = max(0, min(new_left, new_right - MIN_SIZE_NORM))
                new_w = new_right - new_x
                
            elif translated_edit_mode == CropEditMode.DRAG_RIGHT:
                delta_x = (local_x - 0.5) * start_w
                new_right = start_x + start_w + delta_x
                new_w = max(MIN_SIZE_NORM, min(1.0 - start_x, new_right - start_x))
                
            # 简化：暂时只支持边缘编辑，角点编辑逻辑类似但更复杂
            
            # Step 7: 更新crop overlay
            self._crop_overlay_norm = (new_x, new_y, new_w, new_h)
            # 立即重绘
            self.image_label.update()
            
        except Exception as e:
            print(f"Focused模式裁剪编辑错误: {e}")

    def clear_preview(self):
        """彻底清空当前预览及相关状态"""
        # 停止 black cutoff 显示
        self._show_black_cutoff = False
        self._cutoff_compensation = 0.0
        self._cutoff_pixels = None

        # 释放当前图像
        if self.current_image is not None:
            if hasattr(self.current_image, "array") and self.current_image.array is not None:
                self.current_image.array = None
                print("Debug: preview清空，删除Array")
            self.current_image = None
            print("Debug: preview清空，current_image")

        # 清空 UI
        self.image_label.clear()
        # 如果有其它 per-image 状态（裁剪 overlay 等），也可以在这里顺手清掉
        self._crop_overlay_norm = None
        self._focused_crop_id = None
        self._all_crops = []
        print("Debug: preview清空完毕")

    def cleanup(self):
        """清理资源，防止内存泄漏

        停止所有 QTimer 实例，防止在对象销毁后仍然触发回调
        这个方法应该在 MainWindow.closeEvent 中调用
        """
        try:
            if hasattr(self, 'smooth_drag_timer') and self.smooth_drag_timer:
                self.smooth_drag_timer.stop()
                self.smooth_drag_timer.deleteLater()
                self.smooth_drag_timer = None
        except Exception as e:
            print(f"[WARNING] smooth_drag_timer 清理失败: {e}")

        try:
            if hasattr(self, '_double_click_timer') and self._double_click_timer:
                self._double_click_timer.stop()
                self._double_click_timer.deleteLater()
                self._double_click_timer = None
        except Exception as e:
            print(f"[WARNING] _double_click_timer 清理失败: {e}")

        try:
            if hasattr(self, '_ants_timer') and self._ants_timer:
                self._ants_timer.stop()
                self._ants_timer.deleteLater()
                self._ants_timer = None
        except Exception as e:
            print(f"[WARNING] _ants_timer 清理失败: {e}")

    def __del__(self):
        """析构函数：确保资源被清理

        作为最后一道防线，即使 cleanup() 没有被显式调用
        注意：__del__ 的调用时机不确定，不应该依赖它进行关键清理
        """
        try:
            # 只清理关键资源，避免在析构时做复杂操作
            self.cleanup()
        except Exception:
            # 析构函数中不应该抛出异常
            pass
