# shortcuts.py
from PySide6.QtCore import Qt, QObject, QEvent
from PySide6.QtGui import QShortcut, QKeySequence
from PySide6.QtWidgets import QApplication

import sys


class ShortcutsBinder(QObject):
    """
    统一注册快捷键并内置动作逻辑。
    依赖 host 提供以下属性/方法：
        host.context.rotate(deg)
        host.context.get_current_params() -> params or None
        host.context.update_params(params)
        host.preview_widget.context.folder_navigator.navigate_previous()
        host.preview_widget.context.folder_navigator.navigate_next()
        host._show_status_message(str)
        host._reset_parameters()
        host._set_folder_default()
    """
    def __init__(self, host):
        super().__init__(host)
        self.host = host
        self._shortcuts = []  # 持有引用，避免被GC

    # ---------- 公共入口 ----------
    def setup_default_shortcuts(self):
        add = self._add

        # ========== 导航功能 ==========
        add(Qt.Key_Left,  self._act_go_prev)        # 上一张照片
        add(Qt.Key_Right, self._act_go_next)        # 下一张照片

        # ========== 裁剪功能 ==========
        add(Qt.Key_Up,   self._act_cycle_crops_up)  # 向上切换裁剪
        add(Qt.Key_Down, self._act_cycle_crops_down)# 向下切换裁剪
        add("Ctrl+=", self._act_add_crop)           # 添加新裁剪

        # ========== 旋转功能 ==========
        add(Qt.Key_BracketLeft,  self._act_rotate_left)  # 左旋转 90°
        add(Qt.Key_BracketRight, self._act_rotate_right) # 右旋转 90°

        # ========== AI校色功能 ==========
        add(Qt.Key_Space, self._act_auto_color)     # AI校色一次
        add("Shift+Space", self._act_auto_color_multi) # AI校色多次

        # ========== 参数管理 ==========
        add("Ctrl+V", self._act_reset_parameters)  # 重置所有参数
        add("Ctrl+C", self._act_set_folder_default) # 设为文件夹默认

        # ========== 参数调节功能 ==========
        # R通道 (红色/青色)
        add(Qt.Key_Q, self._act_R_down)         # Q: R通道-0.01 (增青)
        add("Shift+Q", self._act_R_down_fine)   # Shift+Q: R通道-0.001 (精细)
        add(Qt.Key_E, self._act_R_up)           # E: R通道+0.01 (增红)
        add("Shift+E", self._act_R_up_fine)     # Shift+E: R通道+0.001 (精细)

        # B通道 (蓝色/黄色)
        add(Qt.Key_A, self._act_B_down)         # A: B通道-0.01 (增黄)
        add("Shift+A", self._act_B_down_fine)   # Shift+A: B通道-0.001 (精细)
        add(Qt.Key_D, self._act_B_up)           # D: B通道+0.01 (增蓝)
        add("Shift+D", self._act_B_up_fine)     # Shift+D: B通道+0.001 (精细)

        # 最大密度 (整体曝光)
        add(Qt.Key_W, self._act_dmax_down)      # W: 最大密度-0.01 (提亮)
        add("Shift+W", self._act_dmax_down_fine) # Shift+W: 最大密度-0.001 (精细)
        add(Qt.Key_S, self._act_dmax_up)        # S: 最大密度+0.01 (压暗)
        add("Shift+S", self._act_dmax_up_fine)  # Shift+S: 最大密度+0.001 (精细)

        # 密度反差 (对比度)
        add(Qt.Key_R, self._act_gamma_up)       # R: 密度反差+0.01 (增对比)
        add("Shift+R", self._act_gamma_up_fine) # Shift+R: 密度反差+0.001 (精细)
        add(Qt.Key_F, self._act_gamma_down)     # F: 密度反差-0.01 (降对比)
        add("Shift+F", self._act_gamma_down_fine) # Shift+F: 密度反差-0.001 (精细)

        # 分层反差 - R通道 (亮部红青平衡)
        add("Ctrl+E", self._act_channel_gamma_r_up)          # Ctrl+E: R Gamma升高 (亮部变红)
        add("Ctrl+Shift+E", self._act_channel_gamma_r_up_fine)   # 精细调整
        add("Ctrl+Q", self._act_channel_gamma_r_down)        # Ctrl+Q: R Gamma降低 (亮部变青)
        add("Ctrl+Shift+Q", self._act_channel_gamma_r_down_fine) # 精细调整

        # 分层反差 - B通道 (亮部蓝黄平衡)
        add("Ctrl+D", self._act_channel_gamma_b_up)          # Ctrl+D: B Gamma升高 (亮部变蓝)
        add("Ctrl+Shift+D", self._act_channel_gamma_b_up_fine)   # 精细调整
        add("Ctrl+A", self._act_channel_gamma_b_down)        # Ctrl+A: B Gamma降低 (亮部变黄)
        add("Ctrl+Shift+A", self._act_channel_gamma_b_down_fine) # 精细调整

        # ========== 取色点、中性色功能 ==========
        # 取色点
        add(Qt.Key_I, self._act_pickNeutralPoint)
        add(Qt.Key_N, self._act_applyNeutralColor)

    # ---------- 内部：工具 ----------
    def _add(self, seq, slot, context=Qt.ApplicationShortcut):
        sc = QShortcut(QKeySequence(seq), self.host)
        sc.setContext(context)
        sc.activated.connect(slot)
        self._shortcuts.append(sc)
        return sc


    def _with_params(self, fn):
        params = self.host.context.get_current_params()
        if params is None:
            return False
        fn(params)
        self.host.context.update_params(params)
        return True

    # ---------- 内置动作（不再依赖主窗实现） ----------
    # 导航
    def _act_go_prev(self):
        self.host.preview_widget.context.folder_navigator.navigate_previous()
        self.host._show_status_message("已切换到上一张照片")

    def _act_go_next(self):
        self.host._show_status_message("⏳正在切换到下一张照片...")
        self.host.preview_widget.context.folder_navigator.navigate_next()
        self.host._show_status_message("已切换到下一张照片")

    # 旋转
    def _act_rotate_left(self):
        self.host.context.rotate(90)
        self.host._show_status_message("左旋转 90°")

    def _act_rotate_right(self):
        self.host.context.rotate(-90)
        self.host._show_status_message("右旋转 90°")

    # 参数重置/默认
    def _act_reset_parameters(self):
        self.host._reset_parameters()
        self.host._show_status_message("参数已重置")

    def _act_set_folder_default(self):
        self.host._set_folder_default()
        self.host._show_status_message("已设为文件夹默认")

    # R通道
    def _act_R_down(self):
        def op(p):
            r, g, b = p.rgb_gains
            p.rgb_gains = (max(-3.0, min(3.0, r - 0.01)), g, b)
            self.host._show_status_message(f"R通道: {p.rgb_gains[0]:.3f}")
        self._with_params(op)

    def _act_R_down_fine(self):
        def op(p):
            r, g, b = p.rgb_gains
            p.rgb_gains = (max(-3.0, min(3.0, r - 0.001)), g, b)
            self.host._show_status_message(f"R通道: {p.rgb_gains[0]:.3f}")
        self._with_params(op)

    def _act_R_up(self):
        def op(p):
            r, g, b = p.rgb_gains
            p.rgb_gains = (max(-3.0, min(3.0, r + 0.01)), g, b)
            self.host._show_status_message(f"R通道: {p.rgb_gains[0]:.3f}")
        self._with_params(op)

    def _act_R_up_fine(self):
        def op(p):
            r, g, b = p.rgb_gains
            p.rgb_gains = (max(-3.0, min(3.0, r + 0.001)), g, b)
            self.host._show_status_message(f"R通道: {p.rgb_gains[0]:.3f}")
        self._with_params(op)

    # B通道
    def _act_B_down(self):
        def op(p):
            r, g, b = p.rgb_gains
            p.rgb_gains = (r, g, max(-3.0, min(3.0, b - 0.01)))
            self.host._show_status_message(f"B通道: {p.rgb_gains[2]:.3f}")
        self._with_params(op)

    def _act_B_down_fine(self):
        def op(p):
            r, g, b = p.rgb_gains
            p.rgb_gains = (r, g, max(-3.0, min(3.0, b - 0.001)))
            self.host._show_status_message(f"B通道: {p.rgb_gains[2]:.3f}")
        self._with_params(op)

    def _act_B_up(self):
        def op(p):
            r, g, b = p.rgb_gains
            p.rgb_gains = (r, g, max(-3.0, min(3.0, b + 0.01)))
            self.host._show_status_message(f"B通道: {p.rgb_gains[2]:.3f}")
        self._with_params(op)

    def _act_B_up_fine(self):
        def op(p):
            r, g, b = p.rgb_gains
            p.rgb_gains = (r, g, max(-3.0, min(3.0, b + 0.001)))
            self.host._show_status_message(f"B通道: {p.rgb_gains[2]:.3f}")
        self._with_params(op)

    # dmax
    def _act_dmax_down(self):
        def op(p):
            p.density_dmax = max(0.0, min(4.8, p.density_dmax - 0.01))
            self.host._show_status_message(f"最大密度: {p.density_dmax:.3f}")
        self._with_params(op)

    def _act_dmax_down_fine(self):
        def op(p):
            p.density_dmax = max(0.0, min(4.8, p.density_dmax - 0.001))
            self.host._show_status_message(f"最大密度: {p.density_dmax:.3f}")
        self._with_params(op)

    def _act_dmax_up(self):
        def op(p):
            p.density_dmax = max(0.0, min(4.8, p.density_dmax + 0.01))
            self.host._show_status_message(f"最大密度: {p.density_dmax:.3f}")
        self._with_params(op)

    def _act_dmax_up_fine(self):
        def op(p):
            p.density_dmax = max(0.0, min(4.8, p.density_dmax + 0.001))
            self.host._show_status_message(f"最大密度: {p.density_dmax:.3f}")
        self._with_params(op)

    # gamma
    def _act_gamma_up(self):
        def op(p):
            p.density_gamma = max(0.1, min(4.0, p.density_gamma + 0.01))
            self.host._show_status_message(f"密度反差: {p.density_gamma:.3f}")
        self._with_params(op)

    def _act_gamma_up_fine(self):
        def op(p):
            p.density_gamma = max(0.1, min(4.0, p.density_gamma + 0.001))
            self.host._show_status_message(f"密度反差: {p.density_gamma:.3f}")
        self._with_params(op)

    def _act_gamma_down(self):
        def op(p):
            p.density_gamma = max(0.1, min(4.0, p.density_gamma - 0.01))
            self.host._show_status_message(f"密度反差: {p.density_gamma:.3f}")
        self._with_params(op)

    def _act_gamma_down_fine(self):
        def op(p):
            p.density_gamma = max(0.1, min(4.0, p.density_gamma - 0.001))
            self.host._show_status_message(f"密度反差: {p.density_gamma:.3f}")
        self._with_params(op)

    # Channel Gamma R (亮部分层反差 - 红青平衡)
    def _act_channel_gamma_r_up(self):
        def op(p):
            p.channel_gamma_r = max(0.5, min(2.0, p.channel_gamma_r + 0.01))
            self.host._show_status_message(f"R Gamma: {p.channel_gamma_r:.3f} (亮部变红)")
        self._with_params(op)

    def _act_channel_gamma_r_up_fine(self):
        def op(p):
            p.channel_gamma_r = max(0.5, min(2.0, p.channel_gamma_r + 0.001))
            self.host._show_status_message(f"R Gamma: {p.channel_gamma_r:.3f} (亮部变红)")
        self._with_params(op)

    def _act_channel_gamma_r_down(self):
        def op(p):
            p.channel_gamma_r = max(0.5, min(2.0, p.channel_gamma_r - 0.01))
            self.host._show_status_message(f"R Gamma: {p.channel_gamma_r:.3f} (亮部变青)")
        self._with_params(op)

    def _act_channel_gamma_r_down_fine(self):
        def op(p):
            p.channel_gamma_r = max(0.5, min(2.0, p.channel_gamma_r - 0.001))
            self.host._show_status_message(f"R Gamma: {p.channel_gamma_r:.3f} (亮部变青)")
        self._with_params(op)

    # Channel Gamma B (亮部分层反差 - 蓝黄平衡)
    def _act_channel_gamma_b_up(self):
        def op(p):
            p.channel_gamma_b = max(0.5, min(2.0, p.channel_gamma_b + 0.01))
            self.host._show_status_message(f"B Gamma: {p.channel_gamma_b:.3f} (亮部变蓝)")
        self._with_params(op)

    def _act_channel_gamma_b_up_fine(self):
        def op(p):
            p.channel_gamma_b = max(0.5, min(2.0, p.channel_gamma_b + 0.001))
            self.host._show_status_message(f"B Gamma: {p.channel_gamma_b:.3f} (亮部变蓝)")
        self._with_params(op)

    def _act_channel_gamma_b_down(self):
        def op(p):
            p.channel_gamma_b = max(0.5, min(2.0, p.channel_gamma_b - 0.01))
            self.host._show_status_message(f"B Gamma: {p.channel_gamma_b:.3f} (亮部变黄)")
        self._with_params(op)

    def _act_channel_gamma_b_down_fine(self):
        def op(p):
            p.channel_gamma_b = max(0.5, min(2.0, p.channel_gamma_b - 0.001))
            self.host._show_status_message(f"B Gamma: {p.channel_gamma_b:.3f} (亮部变黄)")
        self._with_params(op)

    def _act_auto_color(self):
        self.host._on_auto_color_requested()
        self.host._show_status_message("校色一次")

    def _act_auto_color_multi(self):
        self.host._on_auto_color_iterative_requested()
        self.host._show_status_message("校色一次")

    def _act_pickNeutralPoint(self):
        self.host.preview_widget.enter_neutral_point_selection_mode()

    def _act_applyNeutralColor(self):
        self.host.parameter_panel.apply_neutral_color_requested.emit(self.host.parameter_panel.neutral_white_point_spinbox.value())

    def _act_add_crop(self):
        """添加新裁剪（相当于点击+按钮）"""
        self.host.preview_widget._on_add_crop_clicked()
        self.host._show_status_message("添加新裁剪")

    # 裁剪循环切换
    def _act_cycle_crops_up(self):
        """向上循环切换crops (contactsheet -> 最后一个crop -> ... -> crop2 -> crop1 -> contactsheet)"""
        try:
            # 获取所有crops
            all_crops = self.host.context.get_all_crops()
            
            # 获取当前状态
            current_kind = getattr(self.host.context, '_current_profile_kind', 'contactsheet')
            current_crop_id = self.host.context.get_active_crop_id()
            
            if not all_crops:
                # 没有crops，保持在contactsheet
                self.host.context.switch_to_contactsheet()
                self.host._show_status_message("没有裁剪区域")
                return
            
            if current_kind == 'contactsheet':
                # 从contactsheet切换到最后一个crop (反向)
                last_crop = all_crops[-1]
                self.host.context.switch_to_crop_focused(last_crop.id)
                self.host._show_status_message(f"切换到 {last_crop.name}")
            else:
                # 当前在某个crop上，找到上一个 (反向)
                current_index = -1
                for i, crop in enumerate(all_crops):
                    if crop.id == current_crop_id:
                        current_index = i
                        break
                
                if current_index >= 0:
                    if current_index > 0:
                        # 切换到上一个crop (反向)
                        prev_crop = all_crops[current_index - 1]
                        self.host.context.switch_to_crop_focused(prev_crop.id)
                        self.host._show_status_message(f"切换到 {prev_crop.name}")
                    else:
                        # 第一个crop，回到contactsheet
                        self.host.context.switch_to_contactsheet()
                        self.host.context.restore_crop_preview()
                        self.host._show_status_message("切换到接触印相")
                else:
                    # 找不到当前crop，回到contactsheet
                    self.host.context.switch_to_contactsheet()
                    self.host.context.restore_crop_preview()
                    self.host._show_status_message("切换到接触印相")
        except Exception as e:
            self.host._show_status_message(f"切换失败: {str(e)}")

    def _act_cycle_crops_down(self):
        """向下循环切换crops (contactsheet -> crop1 -> crop2 -> crop3 -> ... -> contactsheet)"""
        try:
            # 获取所有crops
            all_crops = self.host.context.get_all_crops()
            
            # 获取当前状态
            current_kind = getattr(self.host.context, '_current_profile_kind', 'contactsheet')
            current_crop_id = self.host.context.get_active_crop_id()
            
            if not all_crops:
                # 没有crops，保持在contactsheet
                self.host.context.switch_to_contactsheet()
                self.host._show_status_message("没有裁剪区域")
                return
            
            if current_kind == 'contactsheet':
                # 从contactsheet切换到第一个crop (正向)
                first_crop = all_crops[0]
                self.host.context.switch_to_crop_focused(first_crop.id)
                self.host._show_status_message(f"切换到 {first_crop.name}")
            else:
                # 当前在某个crop上，找到下一个 (正向)
                current_index = -1
                for i, crop in enumerate(all_crops):
                    if crop.id == current_crop_id:
                        current_index = i
                        break
                
                if current_index >= 0:
                    if current_index < len(all_crops) - 1:
                        # 切换到下一个crop (正向)
                        next_crop = all_crops[current_index + 1]
                        self.host.context.switch_to_crop_focused(next_crop.id)
                        self.host._show_status_message(f"切换到 {next_crop.name}")
                    else:
                        # 最后一个crop，回到contactsheet
                        self.host.context.switch_to_contactsheet()
                        self.host.context.restore_crop_preview()
                        self.host._show_status_message("切换到接触印相")
                else:
                    # 找不到当前crop，回到contactsheet
                    self.host.context.switch_to_contactsheet()
                    self.host.context.restore_crop_preview()
                    self.host._show_status_message("切换到接触印相")
        except Exception as e:
            self.host._show_status_message(f"切换失败: {str(e)}")


class ImeBracketFilter(QObject):
    """
    输入法全角括号兜底过滤器：捕捉【】并调用旋转动作。
    """
    def __init__(self, binder: ShortcutsBinder):
        super().__init__(binder.host)
        self._binder = binder

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress:
            t = event.text()
            if t == "【":
                self._binder._act_rotate_left()
                return True
            elif t == "】":
                self._binder._act_rotate_right()
                return True
        return False


def install_ime_brackets_fallback(binder: ShortcutsBinder, install_on_app=True):
    """
    安装【】兜底过滤器到 QApplication（默认）或 host。
    返回过滤器对象（需在外部持有引用）。
    """
    filt = ImeBracketFilter(binder)
    if install_on_app:
        QApplication.instance().installEventFilter(filt)
    else:
        binder.host.installEventFilter(filt)
    return filt