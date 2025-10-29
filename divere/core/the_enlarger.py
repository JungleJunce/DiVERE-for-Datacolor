"""
胶片放大机引擎
负责所有胶片图像处理操作 - 重构版本
使用分离的数学操作和管线处理器
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_dilation
import json
from pathlib import Path
import time
import traceback

from .data_types import ImageData, ColorGradingParams, LUT3D, PreviewConfig
from .math_ops import FilmMathOps
from .pipeline_processor import FilmPipelineProcessor

# 尝试导入深度学习白平衡相关模块（启动时静默，除非显式开启详细日志）
try:
    from ..models.deep_wb_wrapper import create_deep_wb_wrapper
    DEEP_WB_AVAILABLE = True
except ImportError:
    try:
        import os
        _VERBOSE = bool(int(os.environ.get('DIVERE_VERBOSE', '0')))
    except Exception:
        _VERBOSE = False
    if _VERBOSE:
        print("Failed to import deep_wb_wrapper (optional dependency)")
        traceback.print_exc()
    DEEP_WB_AVAILABLE = False



class TheEnlarger:
    """胶片放大机引擎，负责所有图像处理操作 - 重构版本"""

    def __init__(self, preview_config: Optional[PreviewConfig] = None):
        self.preview_config = preview_config or PreviewConfig()
        
        # 核心处理组件
        self.pipeline_processor = FilmPipelineProcessor(
            preview_config=self.preview_config
        )
        
        # GPU加速器
        self.gpu_accelerator = self.pipeline_processor.gpu_accelerator
        
        # 深度白平衡相关
        self._deep_wb_wrapper = None
        
        # 性能分析
        self._profiling_enabled: bool = False
        
        # Deep WB 可选，不可用时保持静默，避免启动期噪声
        if not DEEP_WB_AVAILABLE:
            pass

    def set_profiling_enabled(self, enabled: bool) -> None:
        """启用/关闭预览管线Profiling"""
        self._profiling_enabled = bool(enabled)
        self.pipeline_processor.set_profiling_enabled(enabled)

    def is_profiling_enabled(self) -> bool:
        return self._profiling_enabled

    # =======================
    # 主要处理接口
    # =======================

    def apply_full_pipeline(self, image: ImageData, params: ColorGradingParams, 
                           include_curve: bool = True,
                           for_export: bool = False,
                           chunked: Optional[bool] = None,
                           convert_to_monochrome_in_idt: bool = False,
                           monochrome_converter: Optional[callable] = None) -> ImageData:
        """
        应用完整处理管线（保持向后兼容的接口）
        
        Args:
            image: 输入图像
            params: 处理参数
            include_curve: 是否包含曲线处理
            for_export: 是否用于导出
            chunked: 是否使用分块处理
            convert_to_monochrome_in_idt: 是否在IDT阶段转换为单色
            monochrome_converter: 单色转换函数
            
        Returns:
            处理后的图像
        """
        if image is None:
            return None
            
        # 使用新的全精度管线处理器
        # 导出时强制全精度：禁用LUT优化 + 启用分块
        use_optimization = not for_export
        chunked_arg = True if for_export else chunked
        return self.pipeline_processor.apply_full_precision_pipeline(
            image, params,
            include_curve=include_curve,
            use_optimization=use_optimization,
            chunked=chunked_arg,
            convert_to_monochrome_in_idt=convert_to_monochrome_in_idt,
            monochrome_converter=monochrome_converter
        )

    def apply_preview_pipeline(self, image: ImageData, params: ColorGradingParams,
                              include_curve: bool = True,
                              convert_to_monochrome_in_idt: bool = False,
                              monochrome_converter: Optional[callable] = None) -> ImageData:
        """
        应用预览管线（新接口）
        
        Args:
            image: 输入图像
            params: 处理参数
            include_curve: 是否包含曲线处理
            convert_to_monochrome_in_idt: 是否在IDT阶段转换为单色
            monochrome_converter: 单色转换函数
            
        Returns:
            处理后的预览图像
        """
        if image is None:
            return None
            
        return self.pipeline_processor.apply_preview_pipeline(
            image, params, 
            include_curve=include_curve,
            convert_to_monochrome_in_idt=convert_to_monochrome_in_idt,
            monochrome_converter=monochrome_converter
        )

    def apply_density_inversion(self, image: ImageData, gamma: float, dmax: float) -> ImageData:
        """应用密度反转（保持向后兼容的接口）"""
        if image.array is None: 
            return image
        
        # This now directly calls the pipeline processor's math_ops
        result_array = self.pipeline_processor.math_ops.density_inversion(
            image.array, gamma, dmax, use_optimization=True
        )
        
        return image.copy_with_new_array(result_array)

    # =======================
    # 缓存管理
    # =======================
    
    def clear_caches(self) -> None:
        """清空内部缓存（调试用）"""
        self.pipeline_processor.math_ops.clear_caches()

    # =======================
    # 自动白平衡
    # =======================

    # legacy 方法 calculate_auto_gain_legacy 已移除（请使用 calculate_auto_gain_learning_based）

    def calculate_auto_gain_learning_based(self, image: ImageData) -> Tuple[float, float, float, float, float, float]:
        """
        使用深度学习模型计算自动白平衡的RGB增益。 cr: https://github.com/mahmoudnafifi/Deep_White_Balance/tree/master
        
        返回: (r_gain, g_gain, b_gain, r_illuminant, g_illuminant, b_illuminant)
        """
        if not DEEP_WB_AVAILABLE:
            return (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

        if image.array is None or image.array.size == 0:
            return (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

        try:
            # 确保图像数据在 [0, 255] 范围内
            t0 = time.time()
            img_uint8 = image.array.copy()
            if img_uint8.max() <= 1.0:
                img_uint8 = (img_uint8 * 255).astype(np.uint8)

            # 使用深度学习模型进行白平衡
            # 缓存与复用模型，避免每次加载
            if self._deep_wb_wrapper is None:
                # 优先尝试GPU
                try:
                    deep_wb_wrapper = create_deep_wb_wrapper(device='cuda')
                except Exception:
                    deep_wb_wrapper = create_deep_wb_wrapper(device='cpu')
                self._deep_wb_wrapper = deep_wb_wrapper
            deep_wb_wrapper = self._deep_wb_wrapper
            if deep_wb_wrapper is None:
                return (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

            # 应用深度学习白平衡
            # 降低推理输入最大边长以提升速度（例如512），保持效果可用
            inference_size = 128
            t1 = time.time()
            result = deep_wb_wrapper.process_image(img_uint8, max_size=inference_size)
            t2 = time.time()
            
            if result is None:
                return (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

            # 计算增益（原始图像与校正后图像的比值）
            original_mean = np.mean(img_uint8, axis=(0, 1))
            corrected_mean = np.mean(result, axis=(0, 1))
            
            # 避免除以零
            corrected_mean = np.maximum(corrected_mean, 1e-10)
            
            # 计算增益
            gains = np.log10(original_mean / corrected_mean)
            
            # 裁剪增益值，并计算相对于G通道的调整
            gains = -np.clip(gains, -3.0, 3.0) + gains[1]
            
            # 计算光源估计（归一化的原始均值）
            illuminant = original_mean / np.sum(original_mean)
            
            t3 = time.time()
            if self._profiling_enabled:
                print(f"AI自动校色耗时: 预处理={(t1 - t0)*1000:.1f}ms, 推理={(t2 - t1)*1000:.1f}ms, 统计/收尾={(t3 - t2)*1000:.1f}ms, 总={(t3 - t0)*1000:.1f}ms")
            
            return (gains[0], gains[1], gains[2], illuminant[0], illuminant[1], illuminant[2])
            
        except Exception as e:
            if self._profiling_enabled:
                print(f"Deep White Balance error: {e}")
            return (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)

    # =======================
    # 中性点自动增益（已废弃，移至ApplicationContext的迭代模式）
    # =======================
    # 之前的 calculate_auto_gain_by_selected_neutral 方法已移除
    # 新实现采用异步迭代模式，在 ApplicationContext 中通过
    # _perform_neutral_point_iteration 实现，每次迭代都基于
    # 实际渲染的 DisplayP3 preview 图像

    # =======================
    # 矩阵管理 (DEPRECATED - MOVED TO FilmPipelineProcessor)
    # =======================

    # =======================
    # LUT生成
    # =======================
    
    def generate_3d_lut(self, params: ColorGradingParams, lut_size: int = 64,
                       include_curve: bool = True, use_optimization: bool = True) -> np.ndarray:
        """
        生成3D LUT用于外部应用
        
        Args:
            params: 处理参数
            lut_size: LUT大小（每个维度）
            include_curve: 是否包含曲线
            use_optimization: 是否使用LUT优化
            
        Returns:
            3D LUT数组 [lut_size, lut_size, lut_size, 3]
        """
        return self.pipeline_processor.generate_3d_lut(params, lut_size, include_curve, use_optimization)

    # =======================
    # 向后兼容的Legacy方法（标记为弃用）
    # =======================
    
    # legacy 方法 _process_in_density_space 已移除（请使用 pipeline_processor.apply_full_precision_pipeline）
