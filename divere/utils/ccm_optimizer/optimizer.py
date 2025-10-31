#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCM参数优化器

使用 CMA-ES 优化 DiVERE 管线的关键参数：
- primaries_xy: 输入色彩变换的RGB基色
- gamma: 密度反差参数
- dmax: 最大密度参数
- r_gain: R通道增益
- b_gain: B通道增益

目标：最小化24个 ColorChecker 色块的 log-RMSE 损失
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
import sys

# 项目根目录（源码情形下可用），冻结环境下可能无效，但不强依赖
project_root = Path(__file__).parent.parent.parent

# 兼容导入（冻结/源码/包内）
try:
    from divere.utils.ccm_optimizer.pipeline import DiVEREPipelineSimulator  # type: ignore
    from divere.utils.ccm_optimizer.extractor import extract_colorchecker_patches  # type: ignore
    from divere.utils.ccm_optimizer.log_rmse_loss import calculate_log_rmse, calculate_colorchecker_log_rmse  # type: ignore
except Exception:
    try:
        from .pipeline import DiVEREPipelineSimulator  # type: ignore
        from .extractor import extract_colorchecker_patches  # type: ignore
        from .log_rmse_loss import calculate_log_rmse, calculate_colorchecker_log_rmse  # type: ignore
    except Exception:
        try:
            from utils.ccm_optimizer.pipeline import DiVEREPipelineSimulator  # type: ignore
            from utils.ccm_optimizer.extractor import extract_colorchecker_patches  # type: ignore
            from utils.ccm_optimizer.log_rmse_loss import calculate_log_rmse, calculate_colorchecker_log_rmse  # type: ignore
        except Exception as e:
            raise ImportError(f"无法导入CCM优化器依赖: {e}")

class CCMOptimizer:
    """CCM参数优化器"""
    
    def __init__(self, reference_file: str = "original_color_cc24data.json",
                 weights_config_path: Optional[str] = None,
                 bounds_config_path: Optional[str] = None,
                 sharpening_config: Optional['SpectralSharpeningConfig'] = None,
                 status_callback: Optional[callable] = None,
                 app_context: Any = None):
        """
        初始化优化器
        
        Args:
            reference_file: 参考RGB值文件路径
            weights_config_path: 权重配置文件路径
            bounds_config_path: 参数边界配置文件路径
            sharpening_config: 光谱锐化（硬件校正）配置，决定哪些参数参与优化
            status_callback: 状态回调函数
            app_context: ApplicationContext实例（必需），用于获取统一的reference color数据
        """
        # 导入配置类（避免循环导入）
        if sharpening_config is None:
            try:
                from divere.core.data_types import SpectralSharpeningConfig
                sharpening_config = SpectralSharpeningConfig()
            except ImportError:
                # 向后兼容：创建一个简单的配置对象
                class DefaultConfig:
                    optimize_idt_transformation = True
                    optimize_density_matrix = False
                sharpening_config = DefaultConfig()
        
        self.sharpening_config = sharpening_config
        
        # 检查必需参数
        if app_context is None:
            raise ValueError(
                "app_context 是必需参数。CCMOptimizer 需要 ApplicationContext "
                "以确保与 Preview 显示使用相同的 reference color 数据。"
            )
        
        if not hasattr(app_context, 'get_reference_colors'):
            raise ValueError(
                "app_context 必须具有 get_reference_colors 方法。"
                "请传入有效的 ApplicationContext 实例。"
            )
        
        self.app_context = app_context
        self.reference_file = reference_file
        
        # 获取当前工作空间信息
        working_colorspace = "ACEScg"  # 默认值
        color_space_manager = None
        if app_context and hasattr(app_context, 'color_space_manager'):
            color_space_manager = app_context.color_space_manager
            if hasattr(color_space_manager, 'get_current_working_space'):
                working_colorspace = color_space_manager.get_current_working_space()
        
        self.pipeline = DiVEREPipelineSimulator(
            verbose=False, 
            working_colorspace=working_colorspace,
            color_space_manager=color_space_manager
        )
        self.reference_values = self._load_reference_values(reference_file)
        # 加载权重配置（可选）。默认使用内置的 config/colorchecker/weights.json
        self._weights_config = self._load_weights_config(weights_config_path)
        self._patch_weight_map = self._build_patch_weight_map(self._weights_config)
        # 加载参数边界配置
        self._bounds_config = self._load_optimization_bounds(bounds_config_path)
        
        # 状态回调函数
        self.status_callback = status_callback
        
        # 根据配置构建参数映射
        self._build_parameter_mapping()
        
    def _build_parameter_mapping(self):
        """根据配置动态构建参数边界、初始值和索引映射"""
        self.bounds = {}
        self.initial_params = {}
        self._param_indices = {}  # 参数名到数组索引的映射
        
        current_idx = 0
        
        # 从配置文件读取边界设置
        bounds_cfg = self._bounds_config
        
        # IDT transformation参数（始终包含gamma, dmax, r_gain, b_gain）
        gamma_cfg = bounds_cfg.get('gamma', {'min': 1.0, 'max': 4.0, 'default': 2.0})
        dmax_cfg = bounds_cfg.get('dmax', {'min': 0.5, 'max': 4.0, 'default': 2.0})
        r_gain_cfg = bounds_cfg.get('r_gain', {'min': -2.0, 'max': 2.0, 'default': 0.0})
        b_gain_cfg = bounds_cfg.get('b_gain', {'min': -2.0, 'max': 2.0, 'default': 0.0})
        
        self.bounds['gamma'] = (gamma_cfg['min'], gamma_cfg['max'])
        self.bounds['dmax'] = (dmax_cfg['min'], dmax_cfg['max'])
        self.bounds['r_gain'] = (r_gain_cfg['min'], r_gain_cfg['max'])
        self.bounds['b_gain'] = (b_gain_cfg['min'], b_gain_cfg['max'])
        
        self.initial_params['gamma'] = gamma_cfg['default']
        self.initial_params['dmax'] = dmax_cfg['default']
        self.initial_params['r_gain'] = r_gain_cfg['default']
        self.initial_params['b_gain'] = b_gain_cfg['default']
        
        self._param_indices['gamma'] = current_idx
        self._param_indices['dmax'] = current_idx + 1
        self._param_indices['r_gain'] = current_idx + 2  
        self._param_indices['b_gain'] = current_idx + 3
        current_idx += 4
        
        # primaries_xy（如果启用IDT优化）
        if self.sharpening_config.optimize_idt_transformation:
            prim_cfg = bounds_cfg.get('primaries_xy', {})
            r_x_cfg = prim_cfg.get('r_x', {'min': 0.0, 'max': 1.0, 'default': 0.64})
            r_y_cfg = prim_cfg.get('r_y', {'min': 0.0, 'max': 1.0, 'default': 0.33})
            g_x_cfg = prim_cfg.get('g_x', {'min': 0.0, 'max': 1.0, 'default': 0.30})
            g_y_cfg = prim_cfg.get('g_y', {'min': 0.0, 'max': 1.0, 'default': 0.60})
            b_x_cfg = prim_cfg.get('b_x', {'min': 0.0, 'max': 1.0, 'default': 0.15})
            b_y_cfg = prim_cfg.get('b_y', {'min': 0.0, 'max': 1.0, 'default': 0.06})
            
            self.bounds['primaries_xy'] = [
                (r_x_cfg['min'], r_x_cfg['max']), (r_y_cfg['min'], r_y_cfg['max']),
                (g_x_cfg['min'], g_x_cfg['max']), (g_y_cfg['min'], g_y_cfg['max']),
                (b_x_cfg['min'], b_x_cfg['max']), (b_y_cfg['min'], b_y_cfg['max'])
            ]
            self.initial_params['primaries_xy'] = np.array([
                r_x_cfg['default'], r_y_cfg['default'],
                g_x_cfg['default'], g_y_cfg['default'],
                b_x_cfg['default'], b_y_cfg['default']
            ])
            self._param_indices['primaries_xy'] = slice(current_idx, current_idx + 6)
            current_idx += 6
        
        # density_matrix（如果启用density matrix优化）  
        if self.sharpening_config.optimize_density_matrix:
            # 现在所有9个参数都可以优化（包括左上角）
            matrix_cfg = bounds_cfg.get('density_matrix', {})
            m00_cfg = matrix_cfg.get('m00', {'min': 0.5, 'max': 1.5, 'default': 1.0})
            m01_cfg = matrix_cfg.get('m01', {'min': -0.5, 'max': 0.5, 'default': 0.0})
            m02_cfg = matrix_cfg.get('m02', {'min': -0.1, 'max': 0.1, 'default': 0.0})
            m10_cfg = matrix_cfg.get('m10', {'min': -0.5, 'max': 0.5, 'default': 0.0})
            m11_cfg = matrix_cfg.get('m11', {'min': 0.5, 'max': 1.5, 'default': 1.0})
            m12_cfg = matrix_cfg.get('m12', {'min': -0.5, 'max': 0.5, 'default': 0.0})
            m20_cfg = matrix_cfg.get('m20', {'min': -0.1, 'max': 0.1, 'default': 0.0})
            m21_cfg = matrix_cfg.get('m21', {'min': -0.5, 'max': 0.5, 'default': 0.0})
            m22_cfg = matrix_cfg.get('m22', {'min': 0.5, 'max': 1.5, 'default': 1.0})
            
            self.bounds['density_matrix'] = [
                (m00_cfg['min'], m00_cfg['max']),  # 现在包含左上角(0,0)
                (m01_cfg['min'], m01_cfg['max']), (m02_cfg['min'], m02_cfg['max']),
                (m10_cfg['min'], m10_cfg['max']), (m11_cfg['min'], m11_cfg['max']), (m12_cfg['min'], m12_cfg['max']),
                (m20_cfg['min'], m20_cfg['max']), (m21_cfg['min'], m21_cfg['max']), (m22_cfg['min'], m22_cfg['max'])
            ]
            # 初始为单位矩阵的9个元素
            self.initial_params['density_matrix'] = np.array([
                m00_cfg['default'],                                    # (0,0) - 不再固定
                m01_cfg['default'], m02_cfg['default'],                # 第一行剩余: (0,1), (0,2)
                m10_cfg['default'], m11_cfg['default'], m12_cfg['default'],  # 第二行: (1,0), (1,1), (1,2)
                m20_cfg['default'], m21_cfg['default'], m22_cfg['default']   # 第三行: (2,0), (2,1), (2,2)
            ])
            self._param_indices['density_matrix'] = slice(current_idx, current_idx + 9)
            current_idx += 9
        
        self._total_params = current_idx
    
    def _clamp_to_bounds(self, value: float, param_config: Dict) -> float:
        """将数值调整到配置的边界内"""
        min_val = param_config['min']
        max_val = param_config['max']
        return max(min_val, min(max_val, value))
    
    def _update_initial_params_from_ui(self, ui_params: Dict):
        """根据UI当前参数更新优化初值，自动调整超界值"""
        bounds_cfg = self._bounds_config
        
        # 更新基础参数，并确保在边界内
        if 'gamma' in ui_params:
            gamma_cfg = bounds_cfg.get('gamma', {'min': 1.0, 'max': 4.0})
            self.initial_params['gamma'] = self._clamp_to_bounds(float(ui_params['gamma']), gamma_cfg)
        if 'dmax' in ui_params:
            dmax_cfg = bounds_cfg.get('dmax', {'min': 0.5, 'max': 4.0})
            self.initial_params['dmax'] = self._clamp_to_bounds(float(ui_params['dmax']), dmax_cfg)
        if 'r_gain' in ui_params:
            r_gain_cfg = bounds_cfg.get('r_gain', {'min': -2.0, 'max': 2.0})
            self.initial_params['r_gain'] = self._clamp_to_bounds(float(ui_params['r_gain']), r_gain_cfg)
        if 'b_gain' in ui_params:
            b_gain_cfg = bounds_cfg.get('b_gain', {'min': -2.0, 'max': 2.0})
            self.initial_params['b_gain'] = self._clamp_to_bounds(float(ui_params['b_gain']), b_gain_cfg)
            
        # 更新primaries_xy（如果启用优化）
        if 'primaries_xy' in self._param_indices and 'primaries_xy' in ui_params:
            prim_cfg = bounds_cfg.get('primaries_xy', {})
            ui_primaries = np.array(ui_params['primaries_xy']).flatten()
            
            # 对每个基色坐标进行边界检查
            param_names = ['r_x', 'r_y', 'g_x', 'g_y', 'b_x', 'b_y']
            clamped_primaries = []
            for i, param_name in enumerate(param_names):
                param_cfg = prim_cfg.get(param_name, {'min': 0.0, 'max': 1.0})
                clamped_value = self._clamp_to_bounds(ui_primaries[i], param_cfg)
                clamped_primaries.append(clamped_value)
            
            self.initial_params['primaries_xy'] = np.array(clamped_primaries)
            
        # 更新density_matrix（如果启用优化）
        if 'density_matrix' in self._param_indices and 'density_matrix' in ui_params:
            matrix_cfg = bounds_cfg.get('density_matrix', {})
            ui_matrix = ui_params['density_matrix']
            
            # 处理3x3矩阵，对每个元素进行边界检查
            if ui_matrix is not None:
                if hasattr(ui_matrix, 'shape') and ui_matrix.shape == (3, 3):
                    # 扁平化为9个元素并应用边界约束
                    matrix_elements = ['m00', 'm01', 'm02', 'm10', 'm11', 'm12', 'm20', 'm21', 'm22']
                    clamped_matrix = []
                    for i, elem_name in enumerate(matrix_elements):
                        row, col = i // 3, i % 3
                        elem_cfg = matrix_cfg.get(elem_name, {'min': -1.0, 'max': 1.0})
                        clamped_value = self._clamp_to_bounds(float(ui_matrix[row, col]), elem_cfg)
                        clamped_matrix.append(clamped_value)
                    
                    self.initial_params['density_matrix'] = np.array(clamped_matrix)
    
    def _load_reference_values(self, reference_file: str) -> Dict[str, List[float]]:
        """
        加载参考RGB值，使用ApplicationContext的统一数据源
        确保与Preview显示的reference color完全一致
        包括type=XYZ时的工作空间转换逻辑
        """
        try:
            # 使用ApplicationContext的统一reference color数据
            # 这确保了与Preview显示完全相同的数据（包括所有转换逻辑）
            reference_data = self.app_context.get_reference_colors(reference_file)
            
            if not reference_data:
                raise ValueError(f"无法从 ApplicationContext 获取 reference colors: {reference_file}")
            
            # 转换为优化器需要的格式 Dict[str, List[float]]
            result = {}
            for patch_id, rgb_tuple in reference_data.items():
                if isinstance(rgb_tuple, (list, tuple)) and len(rgb_tuple) >= 3:
                    result[patch_id] = [float(rgb_tuple[0]), float(rgb_tuple[1]), float(rgb_tuple[2])]
                else:
                    raise ValueError(f"无效的RGB数据格式 (patch {patch_id}): {type(rgb_tuple)}")
            
            if not result:
                raise ValueError(f"未找到有效的色块数据: {reference_file}")
            
            return result
            
        except Exception as e:
            raise RuntimeError(
                f"从 ApplicationContext 加载 reference colors 失败: {e}\n"
                f"文件: {reference_file}\n"
                f"请检查 ApplicationContext 和 ColorChecker 文件是否正常。"
            )

    # ===== 权重：加载与查询 =====
    def _load_weights_config(self, weights_config_path: Optional[str]) -> Dict[str, Any]:
        """加载色块权重配置；失败时返回默认等权。"""
        # 默认路径：使用统一的资源解析入口
        if weights_config_path:
            path = Path(weights_config_path)
        else:
            try:
                from divere.utils.app_paths import resolve_data_path
                path = resolve_data_path("config", "colorchecker", "weights.json")
            except Exception:
                path = project_root / "divere" / "config" / "colorchecker" / "weights.json"
        try:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data
        except Exception as e:
            print(f"警告：无法加载权重配置 {path}: {e}")
        # 回退：等权配置
        return {
            "weights": {"grayscale_weight": 1.0, "skin_weight": 1.0, "color_weight": 1.0},
            "patch_categories": {
                "grayscale": [], "skin": [], "color": []
            }
        }

    def _build_patch_weight_map(self, cfg: Dict[str, Any]) -> Dict[str, float]:
        """根据配置生成每个色块到权重的映射。
        新版本：直接从 cfg['weights'] 读取色块ID到权重的映射。
        向后兼容：如果是旧格式（包含patch_categories），使用旧逻辑。
        """
        mapping: Dict[str, float] = {}
        try:
            weights = cfg.get("weights", {}) or {}
            
            # 检查是否为新格式（直接的色块权重映射）
            # 新格式的weights字段包含类似"A1": 1.0的映射
            if weights and any(key.upper() in ['A1', 'A2', 'B1', 'C1', 'D1'] for key in weights.keys()):
                # 新格式：直接读取色块权重
                for patch_id, weight in weights.items():
                    mapping[str(patch_id)] = float(weight)
            else:
                # 旧格式：使用原有的分组逻辑（向后兼容）
                cats = cfg.get("patch_categories", {}) or {}
                # 首先根据类别设置权重
                for cat, ids in cats.items():
                    weight_key = f"{cat}_weight"
                    wv = float(weights.get(weight_key, 1.0))
                    for pid in ids or []:
                        mapping[str(pid)] = wv
                
                # 然后应用个别权重覆盖（优先级更高）
                individual_weights = cfg.get("individual_weights", {}) or {}
                for patch_id, weight in individual_weights.items():
                    mapping[str(patch_id)] = float(weight)
        except Exception:
            pass
        return mapping

    def _get_patch_weight(self, patch_id: str) -> float:
        return float(self._patch_weight_map.get(patch_id, 1.0))

    def _print_patch_details(self, input_patches: Dict[str, Tuple[float, float, float]], 
                           optimal_params: Dict, correction_matrix: Optional[np.ndarray] = None):
        """打印所有色块的详细误差信息"""
        
        # 加载色块信息
        try:
            from divere.utils.app_paths import resolve_data_path
            weights_path = resolve_data_path("config", "colorchecker", "weights.json")
        except Exception:
            weights_path = Path(__file__).parent.parent.parent / "config" / "colorchecker" / "weights.json"
        
        # 读取色块说明信息
        patch_info = {}
        try:
            import json
            if weights_path.exists():
                with open(weights_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    patch_info = config.get('info', {})
        except Exception:
            pass
        
        # 运行管线获取输出结果
        output_patches = self.pipeline.simulate_full_pipeline(
            input_patches,
            primaries_xy=optimal_params['primaries_xy'],
            gamma=optimal_params['gamma'],
            dmax=optimal_params['dmax'],
            r_gain=optimal_params['r_gain'],
            b_gain=optimal_params['b_gain'],
            correction_matrix=correction_matrix if correction_matrix is not None else optimal_params.get('density_matrix'),
        )
        
        # 首先打印优化参数结果
        print("\n" + "="*80)
        print("优化参数结果")
        print("="*80)
        
        print("\n基础参数:")
        print(f"  gamma: {optimal_params['gamma']:.6f}")
        print(f"  dmax: {optimal_params['dmax']:.6f}")
        print(f"  r_gain: {optimal_params['r_gain']:.6f}")
        print(f"  b_gain: {optimal_params['b_gain']:.6f}")
        
        # 打印IDT基色坐标（如果存在）
        if 'primaries_xy' in optimal_params and optimal_params['primaries_xy'] is not None:
            primaries = optimal_params['primaries_xy']
            print(f"\nIDT基色坐标:")
            print(f"  红色 (R): x={primaries[0,0]:.6f}, y={primaries[0,1]:.6f}")
            print(f"  绿色 (G): x={primaries[1,0]:.6f}, y={primaries[1,1]:.6f}")
            print(f"  蓝色 (B): x={primaries[2,0]:.6f}, y={primaries[2,1]:.6f}")
        
        # 打印密度校正矩阵（如果存在）
        if 'density_matrix' in optimal_params and optimal_params['density_matrix'] is not None:
            matrix = optimal_params['density_matrix']
            print(f"\n密度校正矩阵:")
            for i in range(3):
                row_str = "  [" + ", ".join([f"{matrix[i,j]:8.6f}" for j in range(3)]) + "]"
                print(row_str)
        
        print("\n" + "="*80)
        print("ColorChecker 24色块详细误差报告")
        print("="*80)
        
        # 按A1-D6顺序打印
        all_patches = [
            'A1', 'A2', 'A3', 'A4', 'A5', 'A6',
            'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 
            'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
            'D1', 'D2', 'D3', 'D4', 'D5', 'D6'
        ]
        
        for patch_id in all_patches:
            if patch_id in self.reference_values and patch_id in output_patches:
                ref_rgb = np.array(self.reference_values[patch_id])
                out_rgb = np.array(output_patches[patch_id])
                
                # 计算log-RMSE
                log_rmse = calculate_log_rmse(ref_rgb, out_rgb)
                
                # 获取权重和说明
                weight = self._get_patch_weight(patch_id)
                info = patch_info.get(patch_id, "")
                
                print(f"\n{patch_id} ({info}):")
                print(f"  目标RGB: [{ref_rgb[0]:.6f}, {ref_rgb[1]:.6f}, {ref_rgb[2]:.6f}]")
                print(f"  结果RGB: [{out_rgb[0]:.6f}, {out_rgb[1]:.6f}, {out_rgb[2]:.6f}]")
                print(f"  log-RMSE: {log_rmse:.8f}")
                print(f"  权重: {weight}")
                
                # 如果误差较大，额外标注
                if log_rmse > 1e-3:
                    print(f"  ⚠️  误差较大")
                elif log_rmse < 1e-6:
                    print(f"  ✓  误差极小")
        
        print("\n" + "="*80)

    
    def _load_optimization_bounds(self, bounds_config_path: Optional[str]) -> Dict[str, Any]:
        """加载优化参数边界配置"""
        # 默认路径：使用统一的资源解析入口
        if bounds_config_path:
            path = Path(bounds_config_path)
        else:
            try:
                from divere.utils.app_paths import resolve_data_path
                path = resolve_data_path("config", "colorchecker", "optimization_bounds.json")
            except Exception:
                path = project_root / "divere" / "config" / "colorchecker" / "optimization_bounds.json"
        
        try:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('parameter_bounds', {})
        except Exception as e:
            print(f"警告：无法加载边界配置 {path}: {e}")
        
        # 回退：使用硬编码边界
        return self._get_default_bounds()
    
    def _get_default_bounds(self) -> Dict[str, Any]:
        """获取默认的参数边界配置（回退方案）"""
        return {
            "gamma": {"min": 1.0, "max": 4.0, "default": 2.0},
            "dmax": {"min": 0.5, "max": 4.0, "default": 2.0},
            "r_gain": {"min": -2.0, "max": 2.0, "default": 0.0},
            "b_gain": {"min": -2.0, "max": 2.0, "default": 0.0},
            "primaries_xy": {
                "r_x": {"min": 0.0, "max": 1.0, "default": 0.64},
                "r_y": {"min": 0.0, "max": 1.0, "default": 0.33},
                "g_x": {"min": 0.0, "max": 1.0, "default": 0.30},
                "g_y": {"min": 0.0, "max": 1.0, "default": 0.60},
                "b_x": {"min": 0.0, "max": 1.0, "default": 0.15},
                "b_y": {"min": 0.0, "max": 1.0, "default": 0.06}
            },
            "density_matrix": {
                "m00": {"min": 0.5, "max": 1.5, "default": 1.0},
                "m01": {"min": -0.5, "max": 0.5, "default": 0.0},
                "m02": {"min": -0.1, "max": 0.1, "default": 0.0},
                "m10": {"min": -0.5, "max": 0.5, "default": 0.0},
                "m11": {"min": 0.5, "max": 1.5, "default": 1.0},
                "m12": {"min": -0.5, "max": 0.5, "default": 0.0},
                "m20": {"min": -0.1, "max": 0.1, "default": 0.0},
                "m21": {"min": -0.5, "max": 0.5, "default": 0.0},
                "m22": {"min": 0.5, "max": 1.5, "default": 1.0}
            }
        }
    
    def _params_to_dict(self, params: np.ndarray) -> Dict[str, np.ndarray]:
        """将优化参数数组转换为参数字典（动态映射版本）"""
        result = {}
        
        # 基础参数（始终存在）
        result['gamma'] = params[self._param_indices['gamma']]
        result['dmax'] = params[self._param_indices['dmax']]
        result['r_gain'] = params[self._param_indices['r_gain']]
        result['b_gain'] = params[self._param_indices['b_gain']]
        
        # primaries_xy（如果启用IDT优化）
        if 'primaries_xy' in self._param_indices:
            idx_slice = self._param_indices['primaries_xy']
            result['primaries_xy'] = params[idx_slice].reshape(3, 2)
        else:
            # 从当前管线读取IDT变换（当不优化IDT时）
            if self.app_context and hasattr(self.app_context, 'get_current_idt_primaries'):
                result['primaries_xy'] = self.app_context.get_current_idt_primaries()
            else:
                # 后备方案：使用sRGB基色（向后兼容）
                result['primaries_xy'] = np.array([0.64, 0.33, 0.30, 0.60, 0.15, 0.06]).reshape(3, 2)
        
        # density_matrix（如果启用优化）
        if 'density_matrix' in self._param_indices:
            idx_slice = self._param_indices['density_matrix']
            # 重建3x3矩阵：现在所有9个元素都来自优化参数
            matrix_params = params[idx_slice]  # 9个参数
            matrix = np.zeros((3, 3), dtype=float)
            matrix[0, 0] = matrix_params[0]  # (0,0) - 不再固定
            matrix[0, 1] = matrix_params[1]  # (0,1)
            matrix[0, 2] = matrix_params[2]  # (0,2)
            matrix[1, 0] = matrix_params[3]  # (1,0)
            matrix[1, 1] = matrix_params[4]  # (1,1)
            matrix[1, 2] = matrix_params[5]  # (1,2)
            matrix[2, 0] = matrix_params[6]  # (2,0)
            matrix[2, 1] = matrix_params[7]  # (2,1)
            matrix[2, 2] = matrix_params[8]  # (2,2)
            result['density_matrix'] = matrix
        else:
            result['density_matrix'] = None
        
        return result
    
    def _dict_to_params(self, params_dict: Dict) -> np.ndarray:
        """将参数字典转换为优化参数数组（动态映射版本）"""
        params = np.zeros(self._total_params, dtype=float)
        
        # 基础参数（始终存在），使用默认值填充缺少的参数
        params[self._param_indices['gamma']] = params_dict.get('gamma', self.initial_params['gamma'])
        params[self._param_indices['dmax']] = params_dict.get('dmax', self.initial_params['dmax'])
        params[self._param_indices['r_gain']] = params_dict.get('r_gain', self.initial_params['r_gain'])
        params[self._param_indices['b_gain']] = params_dict.get('b_gain', self.initial_params['b_gain'])
        
        # primaries_xy（如果启用优化）
        if 'primaries_xy' in self._param_indices:
            idx_slice = self._param_indices['primaries_xy']
            if 'primaries_xy' in params_dict:
                params[idx_slice] = params_dict['primaries_xy'].flatten()
            
        # density_matrix（如果启用优化）
        if 'density_matrix' in self._param_indices:
            idx_slice = self._param_indices['density_matrix']
            if 'density_matrix' in params_dict and params_dict['density_matrix'] is not None:
                matrix = params_dict['density_matrix']
                # 如果matrix已经是1D数组（从initial_params来），直接使用
                if matrix.ndim == 1 and len(matrix) == 9:
                    params[idx_slice] = matrix
                else:
                    # 如果是3x3矩阵，提取所有9个元素
                    matrix_params = np.array([
                        matrix[0, 0], matrix[0, 1], matrix[0, 2],  # 第一行3个（包含左上角）
                        matrix[1, 0], matrix[1, 1], matrix[1, 2],  # 第二行3个
                        matrix[2, 0], matrix[2, 1], matrix[2, 2]   # 第三行3个
                    ])
                    params[idx_slice] = matrix_params
            
        return params
    
    def objective_function(self, params: np.ndarray, 
                          input_patches: Dict[str, Tuple[float, float, float]]) -> float:
        """
        目标函数：计算 log-RMSE 损失
        
        Args:
            params: 优化参数数组
            input_patches: 输入色块RGB值
            
        Returns:
            加权平均 log-RMSE 值
        """
        # 兼容旧签名：默认不使用校正矩阵
        return self.compute_rmse(params, input_patches, correction_matrix=None)

    def compute_rmse(self, params: np.ndarray,
                     input_patches: Dict[str, Tuple[float, float, float]],
                     correction_matrix: Optional[np.ndarray] = None) -> float:
        """计算给定参数与可选密度校正矩阵下的全局 log-RMSE 损失。"""
        try:
            params_dict = self._params_to_dict(params)
            
            # 确定实际使用的correction_matrix
            actual_correction_matrix = correction_matrix
            if params_dict['density_matrix'] is not None:
                # 如果参数中包含优化的density_matrix，优先使用它
                actual_correction_matrix = params_dict['density_matrix']
            
            output_patches = self.pipeline.simulate_full_pipeline(
                input_patches,
                primaries_xy=params_dict['primaries_xy'],
                gamma=params_dict['gamma'],
                dmax=params_dict['dmax'],
                r_gain=params_dict['r_gain'],
                b_gain=params_dict['b_gain'],
                correction_matrix=actual_correction_matrix,
            )
            
            # 使用 log-RMSE 计算损失
            weights_dict = {patch_id: self._get_patch_weight(patch_id) 
                           for patch_id in self.reference_values.keys()}
            
            weighted_avg_log_rmse, _ = calculate_colorchecker_log_rmse(
                self.reference_values,
                output_patches,
                weights_dict
            )
            
            return weighted_avg_log_rmse
        except Exception as e:
            print(f"目标函数计算错误: {e}")
            return float('inf')
    
    def optimize(self, input_patches: Dict[str, Tuple[float, float, float]],
                 method: str = 'CMA-ES',
                 max_iter: int = 1000,
                 tolerance: float = 1e-8,
                 correction_matrix: Optional[np.ndarray] = None,
                 ui_params: Optional[Dict] = None,
                 status_callback: Optional[callable] = None) -> Dict:
        """
        执行优化
        
        Args:
            input_patches: 输入色块RGB值
            method: 优化方法
            max_iter: 最大迭代次数
            tolerance: 收敛容差
            correction_matrix: 密度校正矩阵（如果不优化density matrix时使用）
            ui_params: 来自UI的当前参数，用作优化初值
            
        Returns:
            优化结果字典
        """
        # 使用传入的回调或实例回调
        callback = status_callback or self.status_callback
        
        # 如果提供了UI参数，使用它们作为初值
        if ui_params:
            if callback:
                callback(f"使用UI参数作为初值: {ui_params}")
            # 强制设置分层反差为默认值（新增）
            # 原因：优化器优化的是硬件固有特性，分层反差是后期主观调整
            ui_params_copy = ui_params.copy()
            ui_params_copy['channel_gamma_r'] = 1.0
            ui_params_copy['channel_gamma_b'] = 1.0
            self._update_initial_params_from_ui(ui_params_copy)
            if callback:
                callback(f"更新后的初始参数: {self.initial_params}")
        else:
            if callback:
                callback("使用默认初始参数（未提供UI参数）")
            
        if callback:
            callback(f"开始优化，目标：最小化24个色块的 log-RMSE 损失")
            callback(f"优化方法: {method}")
            callback(f"最大迭代: {max_iter}")
            callback(f"收敛容差: {tolerance}")
        
        # 统一使用 CMA-ES
        return self._optimize_cma(input_patches, max_iter=max_iter, tolerance=tolerance, correction_matrix=correction_matrix, status_callback=callback)
    
    def evaluate_parameters(self, params_dict: Dict,
                           input_patches: Dict[str, Tuple[float, float, float]],
                           correction_matrix: Optional[np.ndarray] = None) -> Dict:
        """
        评估给定参数的性能
        
        Args:
            params_dict: 参数字典
            input_patches: 输入色块RGB值
            
        Returns:
            评估结果字典
        """
        # 确定实际使用的correction_matrix
        actual_correction_matrix = correction_matrix
        if 'density_matrix' in params_dict and params_dict['density_matrix'] is not None:
            # 如果参数中包含优化的density_matrix，优先使用它
            actual_correction_matrix = params_dict['density_matrix']
        
        # 运行管线
        output_patches = self.pipeline.simulate_full_pipeline(
            input_patches,
            primaries_xy=params_dict['primaries_xy'],
            gamma=params_dict['gamma'],
            dmax=params_dict['dmax'],
            r_gain=params_dict['r_gain'],
            b_gain=params_dict['b_gain'],
            correction_matrix=actual_correction_matrix,
        )
        
        # 计算每个色块的误差
        patch_errors: Dict[str, Any] = {}
        total_log_rmse = 0.0
        total_weighted_log_rmse = 0.0
        total_weight = 0.0
        valid_patches = 0
        
        for patch_id in self.reference_values.keys():
            if patch_id in output_patches:
                ref_rgb = np.array(self.reference_values[patch_id])
                out_rgb = np.array(output_patches[patch_id])
                
                # 计算 log-RMSE 损失
                log_rmse = calculate_log_rmse(ref_rgb, out_rgb)
                
                # 为了兼容性，也计算传统的RGB误差
                error = ref_rgb - out_rgb
                mse = float(np.mean(error ** 2))
                rmse = float(np.sqrt(mse))
                
                w = self._get_patch_weight(patch_id)
                patch_errors[patch_id] = {
                    'reference': ref_rgb.tolist(),
                    'output': out_rgb.tolist(),
                    'error': error.tolist(),
                    'mse': float(mse),
                    'rmse': rmse,
                    'log_rmse': float(log_rmse),
                    'weight': float(w)
                }
                
                # 使用 log-RMSE 进行优化目标计算
                total_log_rmse += log_rmse
                total_weighted_log_rmse += float(w) * float(log_rmse)
                total_weight += float(w)
                valid_patches += 1
        
        avg_log_rmse = total_log_rmse / valid_patches if valid_patches > 0 else float('inf')
        weighted_avg_log_rmse = (total_weighted_log_rmse / total_weight) if total_weight > 0 else float('inf')
        
        return {
            'average_log_rmse': avg_log_rmse,
            'weighted_average_log_rmse': weighted_avg_log_rmse,
            'patch_errors': patch_errors,
            'valid_patches': valid_patches
        }

    # ===== CMA-ES 实现 =====
    def _build_bounds_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """生成 (lb, ub, span) 数组，按动态参数映射顺序"""
        lb_list = []
        ub_list = []
        
        # 按照 _build_parameter_mapping 中的顺序构建边界
        # 基础参数：gamma, dmax, r_gain, b_gain
        lb_list.append(self.bounds['gamma'][0])
        ub_list.append(self.bounds['gamma'][1])
        lb_list.append(self.bounds['dmax'][0])
        ub_list.append(self.bounds['dmax'][1])
        lb_list.append(self.bounds['r_gain'][0])
        ub_list.append(self.bounds['r_gain'][1])
        lb_list.append(self.bounds['b_gain'][0])
        ub_list.append(self.bounds['b_gain'][1])
        
        # primaries_xy（如果启用）
        if 'primaries_xy' in self.bounds:
            prim_bounds = self.bounds['primaries_xy']
            for bound in prim_bounds:
                lb_list.append(bound[0])
                ub_list.append(bound[1])
        
        # density_matrix（如果启用）
        if 'density_matrix' in self.bounds:
            matrix_bounds = self.bounds['density_matrix']
            for bound in matrix_bounds:
                lb_list.append(bound[0])
                ub_list.append(bound[1])
        
        lb = np.array(lb_list, dtype=float)
        ub = np.array(ub_list, dtype=float)
        span = np.maximum(ub - lb, 1e-6)
        return lb, ub, span

    def _optimize_cma(self,
                      input_patches: Dict[str, Tuple[float, float, float]],
                      max_iter: int = 1000,
                      tolerance: float = 1e-8,
                      correction_matrix: Optional[np.ndarray] = None,
                      status_callback: Optional[callable] = None) -> Dict:
        try:
            import cma
        except Exception as e:
            raise RuntimeError(f"请先安装 cma: pip install cma ({e})")

        x0 = self._dict_to_params(self.initial_params)
        lb, ub, span = self._build_bounds_arrays()

        sigma0 = 0.3  # 增大初始步长以提高搜索范围
        opts = {
            'bounds': [lb.tolist(), ub.tolist()],
            'scaling_of_variables': span.tolist(),
            'maxiter': int(max_iter),
            'ftarget': max(float(tolerance), 1e-8),  # 防止收敛条件过严
            'verb_disp': 0,  # 禁用CMA-ES内置显示
            'verbose': -1,   # 禁用详细输出
            'popsize': max(int(8 + 4 * np.log(len(x0))), 50),  # 增加最小种群大小
            'tolfun': 1e-12,  # 设置函数值变化容差
            'tolx': 1e-12,    # 设置解向量变化容差
        }

        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        best_log_rmse = float('inf')
        while not es.stop():
            xs = es.ask()
            fs = [float(self.compute_rmse(x, input_patches, correction_matrix=correction_matrix)) for x in xs]
            es.tell(xs, fs)
            # es.disp()  # 禁用CMA-ES内置显示，使用我们自己的状态回调
            gen_best = float(np.min(fs))
            if gen_best < best_log_rmse:
                best_log_rmse = gen_best
            message = f"迭代 {es.countiter:3d}: log-RMSE={gen_best:.6f}  (累计最优={best_log_rmse:.6f})"
            if status_callback:
                status_callback(message)
            else:
                print(message)

        res = es.result  # type: ignore[attr-defined]
        xbest = np.array(res.xbest, dtype=float)
        fbest = float(res.fbest)
        nit = int(es.countiter)
        optimal_params = self._params_to_dict(xbest)
        
        completion_message = "✓ 优化成功完成"
        final_message = f"最终 log-RMSE: {fbest:.6f}"
        iterations_message = f"迭代次数: {nit}"
        
        if status_callback:
            status_callback(completion_message)
            status_callback(final_message)
            status_callback(iterations_message)
        else:
            print(completion_message)
            print(final_message)
            print(iterations_message)
        
        # 打印详细的色块误差信息
        self._print_patch_details(input_patches, optimal_params, correction_matrix)
            
        return {
            'success': True,
            'rmse': fbest,  # 保持字段名为rmse以兼容现有代码，但实际是log-RMSE
            'iterations': nit,
            'parameters': optimal_params,
            'raw_result': res,
        }

def optimize_from_image(image_array: np.ndarray,
                       corners: List[Tuple[float, float]],
                       reference_file: str = "original_color_cc24data.json",
                       **optimizer_kwargs) -> Dict:
    """
    从图像直接优化的便捷函数
    
    Args:
        image_array: 图像数组
        corners: ColorChecker四角点坐标
        reference_file: 参考文件路径
        **optimizer_kwargs: 传递给优化器的参数
        
    Returns:
        优化结果字典
    """
    # 提取色块
    status_callback = optimizer_kwargs.get('status_callback', None)
    
    message = "提取ColorChecker色块..."
    if status_callback:
        status_callback(message)
    else:
        print(message)
        
    input_patches = extract_colorchecker_patches(image_array, corners)
    
    if not input_patches:
        raise ValueError("无法提取色块数据")
    
    message = f"成功提取 {len(input_patches)} 个色块"
    if status_callback:
        status_callback(message)
    else:
        print(message)
    
    # 创建优化器并执行优化
    status_callback = optimizer_kwargs.pop('status_callback', None)
    optimizer = CCMOptimizer(reference_file, status_callback=status_callback)
    result = optimizer.optimize(input_patches, **optimizer_kwargs)
    
    # 评估最终结果
    evaluation = optimizer.evaluate_parameters(result['parameters'], input_patches)
    result['evaluation'] = evaluation
    
    return result

if __name__ == "__main__":
    # 测试代码
    print("CCM优化器测试")
    
    # 创建测试数据
    test_patches = {
        'A1': (0.1, 0.1, 0.1),
        'D6': (0.9, 0.9, 0.9),
        'B3': (0.4, 0.1, 0.1)
    }
    
    optimizer = CCMOptimizer()
    
    # 测试目标函数
    test_params = optimizer._dict_to_params(optimizer.initial_params)
    log_rmse = optimizer.objective_function(test_params, test_patches)
    print(f"初始参数log-RMSE: {log_rmse:.6f}")
    
    # 测试优化
    print("\n开始测试优化...")
    result = optimizer.optimize(test_patches, max_iter=10)
    print(f"优化结果: {result['success']}")
    print(f"最终log-RMSE: {result['rmse']:.6f}")
