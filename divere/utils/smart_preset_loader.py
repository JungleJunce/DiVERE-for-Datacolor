from typing import Optional
from divere.core.data_types import Preset
import json

# Import debug logger
try:
    from .debug_logger import debug, info, warning, error, log_file_operation
except ImportError:
    # Fallback if debug logger is not available
    def debug(msg, module=None): pass
    def info(msg, module=None): pass
    def warning(msg, module=None): pass
    def error(msg, module=None): pass
    def log_file_operation(op, path, success=True, err=None, module=None): pass

class SmartPresetLoader:
    """智能预设加载器 - 完全解耦的独立模块"""
    
    def __init__(self):
        pass
    
    def load_preset_by_name(self, preset_name: str) -> Optional[Preset]:
        """根据预设文件名加载预设"""
        info(f"Loading preset by name: '{preset_name}'", "SmartPresetLoader")
        
        try:
            from divere.utils.path_manager import resolve_path
            info(f"Using path_manager.resolve_path() to find: '{preset_name}'", "SmartPresetLoader")
            
            preset_path = resolve_path(preset_name)
            if not preset_path:
                error(f"Path resolution failed for preset: '{preset_name}'", "SmartPresetLoader")
                raise FileNotFoundError(f"找不到预设文件: {preset_name}")
            
            info(f"Resolved preset path: {preset_path}", "SmartPresetLoader")
            log_file_operation("Load preset file", preset_path, True, None, "SmartPresetLoader")
                
            with open(preset_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            preset = Preset.from_dict(data)
            info(f"Successfully loaded preset: '{preset_name}' -> {preset.name}", "SmartPresetLoader")
            return preset
        except Exception as e:
            error_msg = str(e)
            error(f"Failed to load preset '{preset_name}': {error_msg}", "SmartPresetLoader")
            log_file_operation("Load preset file", preset_name, False, error_msg, "SmartPresetLoader")
            return None
    
    def get_smart_default_preset(self, file_path: str) -> Optional[Preset]:
        """获取文件的智能默认预设"""
        info(f"Getting smart default preset for file: {file_path}", "SmartPresetLoader")
        
        try:
            # 首先检查文件夹默认设置
            folder_default = self._load_folder_default(file_path)
            if folder_default:
                info(f"Found folder default for: {file_path}", "SmartPresetLoader")
                return folder_default
            
            # 如果没有文件夹默认，使用智能分类
            from divere.utils.smart_file_classifier import SmartFileClassifier
            classifier = SmartFileClassifier()
            preset_file = classifier.classify_file(file_path)
            
            info(f"File classification result: '{preset_file}'", "SmartPresetLoader")
            
            result = self.load_preset_by_name(preset_file)
            if result:
                info(f"Smart classification successful: {file_path} -> {preset_file}", "SmartPresetLoader")
            return result
            
        except Exception as e:
            error_msg = str(e)
            error(f"Smart classification failed for {file_path}: {error_msg}", "SmartPresetLoader")
            # 回退到通用默认
            info("Falling back to default.json", "SmartPresetLoader")
            return self.load_preset_by_name("default.json")

    def _load_folder_default(self, file_path: str) -> Optional[Preset]:
        """从文件所在目录的divere_presets.json中加载文件夹默认设置"""
        try:
            from pathlib import Path
            from divere.utils.auto_preset_manager import AutoPresetManager
            from divere.core.data_types import InputTransformationDefinition
            
            # 获取文件所在目录
            image_path = Path(file_path)
            
            # 使用AutoPresetManager加载文件夹默认设置
            auto_manager = AutoPresetManager()
            auto_manager.set_active_directory(str(image_path.parent))
            folder_default = auto_manager.load_folder_default()
            if not folder_default:
                return None
            
            info(f"Loading folder default from: {image_path.parent}", "SmartPresetLoader")
            
            # 构建Preset对象
            preset = Preset()
            preset.name = "文件夹默认设置"
            
            # 设置input_transformation
            idt_data = folder_default.get('idt', {})
            if idt_data:
                preset.input_transformation = InputTransformationDefinition(
                    name=idt_data.get('name', ''),
                    definition=idt_data
                )
            
            # 设置grading_params
            cc_params_data = folder_default.get('cc_params', {})
            if cc_params_data:
                # 处理嵌套的 density_matrix 结构，避免字典被错误转换为0维数组
                processed_params = cc_params_data.copy()
                if 'density_matrix' in processed_params and isinstance(processed_params['density_matrix'], dict):
                    matrix_def = processed_params['density_matrix']
                    # 将嵌套结构扁平化为 ColorGradingParams 期望的格式
                    processed_params['density_matrix_name'] = matrix_def.get('name', 'custom')
                    if matrix_def.get('values'):
                        processed_params['density_matrix'] = matrix_def['values']  # 只传递values数组
                    else:
                        processed_params.pop('density_matrix', None)  # 移除空的matrix字段
                
                # 处理密度曲线：智能处理线性曲线数据，保留真实的单通道曲线
                if 'density_curve' in processed_params and isinstance(processed_params['density_curve'], dict):
                    curve_data = processed_params['density_curve']
                    if curve_data.get('name'):
                        curve_name = curve_data['name']
                        
                        # 特殊处理：修改版曲线（以*开头）应该保留实际曲线数据，不要触发按名称加载
                        if curve_name.startswith('*'):
                            debug(f"Detected modified curve '{curve_name}', preserving actual curve data", "SmartPresetLoader")
                            processed_params['density_curve_name'] = curve_name
                            # 保留所有曲线点数据，确保修改版曲线完整恢复
                            points_data = curve_data.get('points', {})
                            for src_key, dst_key in [('rgb', 'curve_points'), ('r', 'curve_points_r'), ('g', 'curve_points_g'), ('b', 'curve_points_b')]:
                                if src_key in points_data and points_data[src_key]:
                                    processed_params[dst_key] = points_data[src_key]
                                    debug(f"Preserved modified curve data {src_key} -> {dst_key}: {len(points_data[src_key])} points", "SmartPresetLoader")
                        else:
                            # 原有逻辑：处理普通曲线名
                            rgb_points = curve_data.get('points', {}).get('rgb', [])
                            if self._is_identity_curve(rgb_points):
                                # RGB是线性的，但需要检查单通道曲线是否包含真实数据
                                points_data = curve_data.get('points', {})
                                has_real_channel_curves = any([
                                    not self._is_identity_curve(points_data.get(ch, []))
                                    for ch in ['r', 'g', 'b']
                                    if points_data.get(ch)  # 只检查存在的通道
                                ])
                                
                                if has_real_channel_curves:
                                    # 保留单通道曲线，只清空RGB线性数据
                                    new_points = points_data.copy()
                                    new_points['rgb'] = []  # 只清空RGB，保留r/g/b真实曲线
                                    debug(f"Preserving channel curves for '{curve_data['name']}', clearing only RGB linear data", "SmartPresetLoader")
                                    processed_params['density_curve'] = {
                                        'name': curve_data['name'],
                                        'points': new_points
                                    }
                                    # 关键修复：将单通道曲线数据映射到ColorGradingParams期望的格式
                                    processed_params['density_curve_name'] = curve_data['name']
                                    for src_key, dst_key in [('r', 'curve_points_r'), ('g', 'curve_points_g'), ('b', 'curve_points_b')]:
                                        if src_key in points_data and points_data[src_key]:
                                            processed_params[dst_key] = points_data[src_key]
                                            debug(f"Mapped {src_key} curve data to {dst_key}: {len(points_data[src_key])} points", "SmartPresetLoader")
                                else:
                                    # 全是线性曲线，按名称重新加载
                                    debug(f"All curves are linear for '{curve_data['name']}', clearing all points to trigger name-based loading", "SmartPresetLoader")
                                    processed_params['density_curve'] = {
                                        'name': curve_data['name'],
                                        'points': {'rgb': []}  # 触发按名称加载
                                    }
                                    processed_params['density_curve_name'] = curve_data['name']

                # 处理 channel_gamma：将嵌套结构展开为扁平字段
                if 'channel_gamma' in processed_params and isinstance(processed_params['channel_gamma'], dict):
                    gamma_data = processed_params['channel_gamma']
                    processed_params['channel_gamma_r'] = gamma_data.get('r', 1.0)
                    processed_params['channel_gamma_b'] = gamma_data.get('b', 1.0)
                    del processed_params['channel_gamma']  # 删除嵌套结构

                preset.grading_params = processed_params
                
            info(f"Successfully loaded folder default preset", "SmartPresetLoader")
            return preset
            
        except Exception as e:
            error(f"Failed to load folder default for {file_path}: {e}", "SmartPresetLoader")
            return None
    
    def _is_identity_curve(self, points):
        """检查曲线点是否为单位曲线（线性曲线）"""
        try:
            return (
                isinstance(points, (list, tuple)) and len(points) == 2 and
                float(points[0][0]) == 0.0 and float(points[0][1]) == 0.0 and
                float(points[1][0]) == 1.0 and float(points[1][1]) == 1.0
            )
        except Exception:
            return False
