"""
胶片处理管线
包含预览版本和全精度版本的管线处理
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import cv2

from .data_types import ImageData, ColorGradingParams, PreviewConfig
from .math_ops import FilmMathOps
from ..utils.enhanced_config_manager import enhanced_config_manager
from pathlib import Path


class FilmPipelineProcessor:
    """胶片处理管线处理器"""
    
    def __init__(self, math_ops: Optional[FilmMathOps] = None, 
                 preview_config: Optional[PreviewConfig] = None):
        self.math_ops = math_ops or FilmMathOps()
        
        # 预览配置（统一管理）
        self.preview_config = preview_config or PreviewConfig()
        
        # GPU加速器（共享math_ops的实例）
        self.gpu_accelerator = self.math_ops.gpu_accelerator
        
        # 性能监控
        self._profiling_enabled = False
        self._last_profile: Dict[str, float] = {}

        # 矩阵管理
        self._density_matrices: Dict[str, Any] = {}
        self._load_default_matrices()

        # 全精度管线分块参数（自动阈值与默认tile设置）
        # 当图像像素数超过该阈值时，自动启用分块处理以降低峰值内存并提高吞吐
        self.full_pipeline_chunk_threshold: int = 4096 * 4096  # 约16MP
        self.full_pipeline_tile_size: Tuple[int, int] = (2048, 2048)
        self.full_pipeline_max_workers: int = self.math_ops.num_threads
    
    def _load_default_matrices(self):
        """加载默认的校正矩阵"""
        config_files = enhanced_config_manager.get_config_files("matrices")
        for matrix_file in config_files:
            try:
                data = enhanced_config_manager.load_config_file(matrix_file)
                if data:
                    matrix_key = matrix_file.stem
                    self._density_matrices[matrix_key] = data
            except Exception as e:
                print(f"Failed to load matrix {matrix_file}: {e}")

    def get_available_matrices(self) -> List[str]:
        return sorted(list(self._density_matrices.keys()))

    def get_matrix_data(self, key: str) -> Optional[Dict[str, Any]]:
        return self._density_matrices.get(key)
        
    def get_density_matrix_array(self, key: str) -> Optional[np.ndarray]:
        """获取校正矩阵的numpy数组"""
        matrix_data = self.get_matrix_data(key)
        print(f"Debug: get_density_matrix_array({key}) - data found: {matrix_data is not None}")
        if matrix_data:
            print(f"Debug: matrix_data keys: {list(matrix_data.keys())}")
            print(f"Debug: matrix_space: {matrix_data.get('matrix_space')}")
            
            # 检查matrix_space（兼容没有该字段的旧格式）
            matrix_space = matrix_data.get("matrix_space", "density")  # 默认为density
            if matrix_space == "density":
                try:
                    matrix_array = matrix_data.get("matrix")
                    if matrix_array is not None:
                        result = np.array(matrix_array, dtype=np.float64)
                        print(f"Debug: converted matrix shape: {result.shape}, values: {result.tolist()}")
                        # 验证矩阵有效性
                        if result.shape != (3, 3):
                            print(f"Error: invalid matrix shape {result.shape}, expected (3,3)")
                            return None
                        if not np.isfinite(result).all():
                            print(f"Warning: matrix contains invalid values, cleaning up")
                            result = np.where(np.isfinite(result), result, 0.0)
                        return result
                    else:
                        print(f"Error: no 'matrix' field in matrix_data for {key}")
                except Exception as e:
                    print(f"Error: failed to convert matrix for {key}: {e}")
                    return None
            else:
                print(f"Debug: skipping matrix {key} - wrong matrix_space: {matrix_space}")
        return None

    def reload_matrices(self):
        self._density_matrices.clear()
        self._load_default_matrices()

    def set_profiling_enabled(self, enabled: bool) -> None:
        """启用/关闭性能分析"""
        self._profiling_enabled = enabled
    
    def _get_cv2_interpolation(self) -> int:
        """根据预览质量设置获取OpenCV插值方法"""
        quality_map = {
            'nearest': cv2.INTER_NEAREST,
            'linear': cv2.INTER_LINEAR,
            'cubic': cv2.INTER_CUBIC
        }
        return quality_map.get(self.preview_config.preview_quality, cv2.INTER_LINEAR)
    
    def get_last_profile(self) -> Dict[str, float]:
        """获取最后一次处理的性能分析"""
        return self._last_profile.copy()
    
    # =======================
    # 预览版本管线
    # =======================
    
    def apply_preview_pipeline(self, image: ImageData, params: ColorGradingParams,
                              input_colorspace_transform: Optional[np.ndarray] = None,
                              output_colorspace_transform: Optional[np.ndarray] = None,
                              include_curve: bool = True,
                              convert_to_monochrome_in_idt: bool = False,
                              monochrome_converter: Optional[callable] = None) -> ImageData:
        """
        预览版本管线（优化版）：
        原图 -> 早期降采样 -> 输入色彩科学 -> dmax/gamma调整（图片级别） -> 
        套LUT（密度校正矩阵 -> RGB曝光 -> 曲线 -> 转线性） -> 输出色彩转换
        
        关键优化：更早进行降采样，减少后续所有操作的像素数量
        """
        if image is None or image.array is None:
            return image
            
        profile = {}
        t_start = time.time()
        
        # 1. 早期降采样（在色彩管理之前）- 关键优化！
        t0 = time.time()
        proxy_array, scale_factor = self._create_preview_proxy(image.array)
        profile['early_downsample_ms'] = (time.time() - t0) * 1000.0
        
        # 2. 输入色彩科学（在较小的图像上）
        t1 = time.time()
        if input_colorspace_transform is not None:
            proxy_array = self._apply_colorspace_transform(proxy_array, input_colorspace_transform)
        profile['input_colorspace_ms'] = (time.time() - t1) * 1000.0
        
        # 2.5 IDT阶段monochrome转换（移除 - 现在在显示阶段处理）
        profile['idt_monochrome_ms'] = 0.0
        
        # 3. 图片级别的dmax/gamma调整（使用LUT优化）
        t2 = time.time()
        if params.enable_density_inversion:
            proxy_array = self.math_ops.density_inversion(
                proxy_array, params.density_gamma, params.density_dmax, use_optimization=True
            )
        profile['gamma_dmax_ms'] = (time.time() - t2) * 1000.0
        
        # 4. 套LUT（完整数学管线的其余部分，强制禁用并行）
        t3 = time.time()
        lut_profile = {}
        proxy_array = self._apply_preview_lut_pipeline_optimized(proxy_array, params, include_curve, lut_profile)
        profile['lut_pipeline_ms'] = (time.time() - t3) * 1000.0
        profile.update({f"lut/{k}": v for k, v in lut_profile.items()})
        
        # 5. 输出色彩转换
        t4 = time.time()
        if output_colorspace_transform is not None:
            proxy_array = self._apply_colorspace_transform(proxy_array, output_colorspace_transform)
        profile['output_colorspace_ms'] = (time.time() - t4) * 1000.0
        
        # 记录总时间和性能分析
        profile['total_preview_ms'] = (time.time() - t_start) * 1000.0
        profile['scale_factor'] = scale_factor
        self._last_profile = profile
        
        if self._profiling_enabled:
            self._print_preview_profile(profile)
        
        return image.copy_with_new_array(proxy_array)
    
    def _create_preview_proxy(self, image_array: np.ndarray) -> Tuple[np.ndarray, float]:
        """创建预览代理图像（优化版）"""
        h, w = image_array.shape[:2]
        max_dim = max(h, w)
        
        if max_dim <= self.preview_config.preview_max_size:
            return image_array, 1.0
            
        # 计算缩放因子
        scale_factor = self.preview_config.preview_max_size / max_dim
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        
        # 优化：对于大幅度缩放，使用分步降采样以提高质量和速度
        if scale_factor < 0.5:
            # 大幅度缩放：先用INTER_AREA快速降采样，再用INTER_LINEAR精细调整
            intermediate_factor = 0.5
            intermediate_h = int(h * intermediate_factor)
            intermediate_w = int(w * intermediate_factor)
            
            # 第一步：快速降采样
            temp_proxy = cv2.resize(image_array, (intermediate_w, intermediate_h), 
                                  interpolation=cv2.INTER_LINEAR)
            
            # 第二步：精细调整到目标尺寸
            proxy = cv2.resize(temp_proxy, (new_w, new_h), 
                             interpolation=cv2.INTER_LINEAR)
        else:
            # 小幅度缩放：直接使用线性插值
            proxy = cv2.resize(image_array, (new_w, new_h), 
                             interpolation=cv2.INTER_LINEAR)
        
        return proxy, scale_factor
    
    def _apply_preview_lut_pipeline_optimized(self, proxy_array: np.ndarray, params: ColorGradingParams,
                                            include_curve: bool, profile: Dict[str, float]) -> np.ndarray:
        """
        优化版预览LUT管线 - 强制禁用并行处理，针对小图像优化
        """
        if profile is not None:
            profile.clear()
            
        # 转为密度空间
        t0 = time.time()
        density_array = self.math_ops.linear_to_density(proxy_array)
        profile['to_density_ms'] = (time.time() - t0) * 1000.0
        
        # 密度校正矩阵（强制禁用并行）
        if params.enable_density_matrix:
            t1 = time.time()
            matrix = self._get_density_matrix_from_params(params)
            if matrix is not None and not np.allclose(matrix, np.eye(3)):
                density_array = self.math_ops.apply_density_matrix(
                    density_array, matrix, params.density_dmax,
                    channel_gamma_r=params.channel_gamma_r,
                    channel_gamma_b=params.channel_gamma_b,
                    use_parallel=False
                )
            profile['density_matrix_ms'] = (time.time() - t1) * 1000.0
        
        # RGB曝光调整（强制禁用并行）
        if params.enable_rgb_gains:
            t2 = time.time()
            density_array = self.math_ops.apply_rgb_gains(
                density_array, params.rgb_gains, use_parallel=False
            )
            profile['rgb_gains_ms'] = (time.time() - t2) * 1000.0
        
        # 密度曲线调整（强制禁用并行）
        if include_curve and params.enable_density_curve:
            t3 = time.time()
            # RGB通用曲线
            curve_points = params.curve_points if not self._is_default_curve(params.curve_points) else None
            
            # 单通道曲线
            channel_curves = {}
            if not self._is_default_curve(params.curve_points_r):
                channel_curves['r'] = params.curve_points_r
            if not self._is_default_curve(params.curve_points_g):
                channel_curves['g'] = params.curve_points_g
            if not self._is_default_curve(params.curve_points_b):
                channel_curves['b'] = params.curve_points_b
            
            if curve_points or channel_curves:
                # apply_density_curve现在包含了转线性和屏幕反光补偿
                result_array = self.math_ops.apply_density_curve(
                    density_array, curve_points, channel_curves, use_parallel=False,
                    use_optimization=True,  # 预览模式使用LUT优化
                    screen_glare_compensation=params.screen_glare_compensation
                )
            else:
                # 没有曲线时，需要手动转换到线性空间并应用屏幕反光补偿
                result_array = self.math_ops.density_to_linear(density_array)
                if params.screen_glare_compensation > 0.0:
                    result_array = np.maximum(0.0, result_array - params.screen_glare_compensation)
            profile['density_curves_ms'] = (time.time() - t3) * 1000.0
        
        return result_array
    
    def _apply_preview_lut_pipeline(self, proxy_array: np.ndarray, params: ColorGradingParams,
                                   include_curve: bool, profile: Dict[str, float]) -> np.ndarray:
        """
        应用预览LUT管线（不包含密度反相，那已经在图片级别做了）
        包含：密度校正矩阵 -> RGB曝光 -> 曲线 -> 转线性
        """
        if profile is not None:
            profile.clear()
            
        # 转为密度空间
        t0 = time.time()
        density_array = self.math_ops.linear_to_density(proxy_array)
        profile['to_density_ms'] = (time.time() - t0) * 1000.0
        
        # 密度校正矩阵
        if params.enable_density_matrix:
            t1 = time.time()
            matrix = self._get_density_matrix_from_params(params)
            if matrix is not None and not np.allclose(matrix, np.eye(3)):
                density_array = self.math_ops.apply_density_matrix(
                    density_array, matrix, params.density_dmax,
                    channel_gamma_r=params.channel_gamma_r,
                    channel_gamma_b=params.channel_gamma_b,
                    use_parallel=False
                )
            profile['density_matrix_ms'] = (time.time() - t1) * 1000.0

        # RGB曝光调整
        if params.enable_rgb_gains:
            t2 = time.time()
            density_array = self.math_ops.apply_rgb_gains(
                density_array, params.rgb_gains, use_parallel=False
            )
            profile['rgb_gains_ms'] = (time.time() - t2) * 1000.0
        
        # 密度曲线调整
        if include_curve and params.enable_density_curve:
            t3 = time.time()
            # RGB通用曲线
            curve_points = params.curve_points if not self._is_default_curve(params.curve_points) else None

            # 单通道曲线
            channel_curves = {}
            if not self._is_default_curve(params.curve_points_r):
                channel_curves['r'] = params.curve_points_r
            if not self._is_default_curve(params.curve_points_g):
                channel_curves['g'] = params.curve_points_g
            if not self._is_default_curve(params.curve_points_b):
                channel_curves['b'] = params.curve_points_b
            
            if curve_points or channel_curves:
                # apply_density_curve现在包含了转线性和屏幕反光补偿
                result_array = self.math_ops.apply_density_curve(
                    density_array, curve_points, channel_curves, use_parallel=False,
                    use_optimization=True,  # 预览模式使用LUT优化
                    screen_glare_compensation=params.screen_glare_compensation
                )
            else:
                # 没有曲线时，需要手动转换到线性空间并应用屏幕反光补偿
                result_array = self.math_ops.density_to_linear(density_array)
                if params.screen_glare_compensation > 0.0:
                    result_array = np.maximum(0.0, result_array - params.screen_glare_compensation)
            profile['density_curves_ms'] = (time.time() - t3) * 1000.0
        
        return result_array
    
    # =======================
    # 全精度版本管线
    # =======================
    
    def apply_full_precision_pipeline(self, image: ImageData, params: ColorGradingParams,
                                     input_colorspace_transform: Optional[np.ndarray] = None,
                                     output_colorspace_transform: Optional[np.ndarray] = None,
                                     include_curve: bool = True,
                                     use_optimization: bool = True,
                                     chunked: Optional[bool] = None,
                                     tile_size: Optional[Tuple[int, int]] = None,
                                     max_workers: Optional[int] = None,
                                     convert_to_monochrome_in_idt: bool = False,
                                     monochrome_converter: Optional[callable] = None) -> ImageData:
        """
        全精度版本管线：完整数学过程套在原图上
        
        Args:
            image: 输入图像
            params: 处理参数
            input_colorspace_transform: 输入色彩变换变换矩阵
            output_colorspace_transform: 输出色彩空间变换矩阵
            include_curve: 是否包含曲线处理
            use_optimization: 是否使用优化版本
            chunked: 是否使用分块处理
            tile_size: 分块大小
            max_workers: 最大工作线程数
            convert_to_monochrome_in_idt: 是否在IDT阶段转换为单色
            monochrome_converter: 单色转换函数
            
        Returns:
            处理后的全精度图像
        """
        if image is None or image.array is None:
            return image
            
        profile = {}
        t_start = time.time()
        
        # 是否启用分块（若未指定，则按阈值自动决定）
        if chunked is None:
            h, w = image.height, image.width
            chunked = (h * w) > self.full_pipeline_chunk_threshold

        tile_h, tile_w = tile_size or self.full_pipeline_tile_size
        workers = max_workers or self.full_pipeline_max_workers

        if not chunked:
            # 1. 输入色彩科学
            t0 = time.time()
            working_array = image.array.copy()
            if input_colorspace_transform is not None:
                working_array = self._apply_colorspace_transform(working_array, input_colorspace_transform)
            profile['input_colorspace_ms'] = (time.time() - t0) * 1000.0
            
            # 1.5 IDT阶段monochrome转换（移除 - 现在在显示阶段处理）
            profile['idt_monochrome_ms'] = 0.0
            
            # 2. 应用完整数学管线
            t1 = time.time()
            math_profile = {}
            
            # 注入矩阵获取函数到数学操作中（临时解决方案）
            # original_get_matrix = self.math_ops._get_density_matrix
            self.math_ops._get_density_matrix = lambda p: self._get_density_matrix_from_params(p)
            
            try:
                working_array = self.math_ops.apply_full_math_pipeline(
                    working_array, params, include_curve, 
                    params.enable_density_inversion, use_optimization, math_profile
                )
            finally:
                # 恢复原函数
                # self.math_ops._get_density_matrix = original_get_matrix
                pass
                
            profile['math_pipeline_ms'] = (time.time() - t1) * 1000.0
            profile.update({f"math/{k}": v for k, v in math_profile.items()})
            
            # 3. 输出色彩转换
            t2 = time.time()
            if output_colorspace_transform is not None:
                working_array = self._apply_colorspace_transform(working_array, output_colorspace_transform)
            profile['output_colorspace_ms'] = (time.time() - t2) * 1000.0
        else:
            # 分块并行路径
            h, w = image.height, image.width
            working_array = np.empty_like(image.array, dtype=image.array.dtype)

            # 预先拷贝（以便在块内做原地/独立处理）
            src_array = image.array

            # 生成所有块坐标
            tiles: Tuple[Tuple[int,int,int,int], ...] = tuple(
                (y, min(y + tile_h, h), x, min(x + tile_w, w))
                for y in range(0, h, tile_h)
                for x in range(0, w, tile_w)
            )

            # 为累积Profile做粗略计时
            t_input_total = 0.0
            t_math_total = 0.0
            t_output_total = 0.0

            def process_tile(tile_coords: Tuple[int, int, int, int]) -> Tuple[Tuple[int,int,int,int], np.ndarray, Dict[str, float]]:
                sh, eh, sw, ew = tile_coords
                block = src_array[sh:eh, sw:ew, :].copy()

                prof_local: Dict[str, float] = {}

                # 输入色彩
                t0_local = time.time()
                if input_colorspace_transform is not None:
                    block = self._apply_colorspace_transform(block, input_colorspace_transform)
                prof_local['input_ms'] = (time.time() - t0_local) * 1000.0
                
                # IDT阶段monochrome转换（移除 - 现在在显示阶段处理）
                prof_local['monochrome_ms'] = 0.0

                # 完整数学管线（块级）
                t1_local = time.time()
                math_profile_local: Dict[str, float] = {}

                # original_get_matrix = self.math_ops._get_density_matrix
                self.math_ops._get_density_matrix = lambda p: self._get_density_matrix_from_params(p)
                try:
                    block = self.math_ops.apply_full_math_pipeline(
                        block, params, include_curve,
                        params.enable_density_inversion, use_optimization, math_profile_local
                    )
                finally:
                    # self.math_ops._get_density_matrix = original_get_matrix
                    pass

                prof_local['math_ms'] = (time.time() - t1_local) * 1000.0

                # 输出色彩
                t2_local = time.time()
                if output_colorspace_transform is not None:
                    block = self._apply_colorspace_transform(block, output_colorspace_transform)
                prof_local['output_ms'] = (time.time() - t2_local) * 1000.0

                return (sh, eh, sw, ew), block, prof_local

            # 并行执行块
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(process_tile, tile) for tile in tiles]
                for fut in as_completed(futures):
                    (sh, eh, sw, ew), block_out, prof_local = fut.result()
                    working_array[sh:eh, sw:ew, :] = block_out
                    t_input_total += prof_local.get('input_ms', 0.0)
                    t_math_total += prof_local.get('math_ms', 0.0)
                    t_output_total += prof_local.get('output_ms', 0.0)

            # 汇总Profile（仅粗略参考）
            profile['input_colorspace_ms'] = t_input_total
            profile['math_pipeline_ms'] = t_math_total
            profile['output_colorspace_ms'] = t_output_total
        
        # 记录总时间和性能分析
        profile['total_full_precision_ms'] = (time.time() - t_start) * 1000.0
        self._last_profile = profile
        
        if self._profiling_enabled:
            self._print_full_precision_profile(profile)
        
        return image.copy_with_new_array(working_array)
    
    # =======================
    # 辅助方法
    # =======================
    
    def _apply_colorspace_transform(self, image_array: np.ndarray, 
                                   transform_matrix: np.ndarray) -> np.ndarray:
        """应用色彩空间变换"""
        if transform_matrix is None:
            return image_array
            
        # 多通道图像，正常处理
        original_shape = image_array.shape
        reshaped = image_array.reshape(-1, image_array.shape[-1])
        if image_array.shape[-1] == 3:
            # RGB图像，直接应用变换
            transformed = np.dot(reshaped, transform_matrix.T)
            result = transformed.reshape(original_shape)
        else:
            # 其他通道数，仅处理前3个通道
            rgb_part = reshaped[:, :3]
            transformed_rgb = np.dot(rgb_part, transform_matrix.T)
            transformed = reshaped.copy()
            transformed[:, :3] = transformed_rgb
            result = transformed.reshape(original_shape)
        
        # 裁剪到有效范围
        result = np.clip(result, 0.0, 1.0)
        
        return result
    
    def _is_default_curve(self, points: list) -> bool:
        """检查曲线是否为默认直线"""
        return points == [(0.0, 0.0), (1.0, 1.0)] or not points

    def _get_density_matrix_from_params(self, params: ColorGradingParams) -> Optional[np.ndarray]:
        """从参数中获取校正矩阵（重构后简化）"""
        if params.density_matrix is not None:
            return params.density_matrix
        # Fallback to file-based for older preset compatibility if needed
        # but the new flow should ensure density_matrix is always populated.
        return None
    
    def set_matrix_loader(self, loader_func):
        """设置矩阵加载器函数"""
        # Deprecated: matrix loading is now internal.
        pass
    
    def _print_preview_profile(self, profile: Dict[str, float]) -> None:
        """打印预览性能分析"""
        print(
            f"预览管线Profile (缩放={profile.get('scale_factor', 1.0):.2f}): "
            f"输入色彩={profile.get('input_colorspace_ms', 0.0):.1f}ms, "
            f"降采样={profile.get('downsample_ms', 0.0):.1f}ms, "
            f"Gamma/Dmax={profile.get('gamma_dmax_ms', 0.0):.1f}ms, "
            f"LUT管线={profile.get('lut_pipeline_ms', 0.0):.1f}ms "
            f"(密度转换={profile.get('lut/to_density_ms', 0.0):.1f}ms, "
            f"矩阵={profile.get('lut/density_matrix_ms', 0.0):.1f}ms, "
            f"RGB增益={profile.get('lut/rgb_gains_ms', 0.0):.1f}ms, "
            f"曲线={profile.get('lut/density_curves_ms', 0.0):.1f}ms, "
            f"转线性={profile.get('lut/to_linear_ms', 0.0):.1f}ms), "
            f"输出色彩={profile.get('output_colorspace_ms', 0.0):.1f}ms, "
            f"总计={profile.get('total_preview_ms', 0.0):.1f}ms"
        )
    
    def _print_full_precision_profile(self, profile: Dict[str, float]) -> None:
        """打印全精度性能分析"""
        print(
            f"全精度管线Profile: "
            f"输入色彩={profile.get('input_colorspace_ms', 0.0):.1f}ms, "
            f"数学管线={profile.get('math_pipeline_ms', 0.0):.1f}ms "
            f"(密度反相={profile.get('math/density_inversion_ms', 0.0):.1f}ms, "
            f"密度转换={profile.get('math/to_density_ms', 0.0):.1f}ms, "
            f"矩阵={profile.get('math/density_matrix_ms', 0.0):.1f}ms, "
            f"RGB增益={profile.get('math/rgb_gains_ms', 0.0):.1f}ms, "
            f"曲线={profile.get('math/density_curves_ms', 0.0):.1f}ms, "
            f"转线性={profile.get('math/to_linear_ms', 0.0):.1f}ms), "
            f"输出色彩={profile.get('output_colorspace_ms', 0.0):.1f}ms, "
            f"总计={profile.get('total_full_precision_ms', 0.0):.1f}ms"
        )
    
    # =======================
    # LUT生成（用于外部LUT导出）
    # =======================
    
    def generate_3d_lut(self, params: ColorGradingParams, lut_size: int = 64,
                       include_curve: bool = True, use_optimization: bool = True) -> np.ndarray:
        """
        生成3D LUT用于外部应用
        
        Args:
            params: 处理参数
            lut_size: LUT大小（每个维度）
            include_curve: 是否包含曲线
            
        Returns:
            3D LUT数组 [lut_size, lut_size, lut_size, 3]
        """
        # 生成输入网格
        coords = np.linspace(0.0, 1.0, lut_size, dtype=np.float32)
        r_coords, g_coords, b_coords = np.meshgrid(coords, coords, coords, indexing='ij')
        
        # 重塑为 [N, 3]
        input_colors = np.stack([r_coords.ravel(), g_coords.ravel(), b_coords.ravel()], axis=1)
        
        # 应用数学管线（不包含密度反相，因为LUT通常用于已经反相的图像）
        # 注入矩阵获取函数
        # original_get_matrix = self.math_ops._get_density_matrix
        self.math_ops._get_density_matrix = lambda p: self._get_density_matrix_from_params(p)
        
        try:
            output_colors = self.math_ops.apply_full_math_pipeline(
                input_colors.reshape(lut_size, lut_size, lut_size, 3),
                params, include_curve, enable_density_inversion=False, use_optimization=use_optimization
            )
        finally:
            # self.math_ops._get_density_matrix = original_get_matrix
            pass
        
        return output_colors
