"""
图像管理模块
负责图像的导入、代理生成、缓存管理
"""

import os
import hashlib
from typing import Optional, Dict, Tuple
from pathlib import Path
from collections import OrderedDict
import numpy as np
import cv2
from PIL import Image
import imageio

from .data_types import ImageData
from ..utils.app_paths import get_data_dir

# 配置PIL的图像大小限制
# 默认情况下PIL限制为178,956,970像素以防止decompression bomb攻击
# 对于胶片扫描，我们需要处理更大的图像（如228M像素的扫描图）
# 设置为None表示无限制，或设置为更大的值（如500M像素）
#PIL_MAX_PIXELS = 500_000_000  # 500M像素，足够大但仍有安全边界
Image.MAX_IMAGE_PIXELS = None
#print(f"[ImageManager] PIL maximum image pixels set to: {PIL_MAX_PIXELS:,} ({PIL_MAX_PIXELS/1_000_000:.1f}M pixels)")
print(f"[ImageManager] PIL maximum image pixels set to: Infinite pixels)")


class ImageManager:
    """图像管理器"""

    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._proxy_cache: "OrderedDict[str, ImageData]" = OrderedDict()  # 使用 OrderedDict 实现正确的 LRU
        self._max_cache_size = 10  # 最大缓存图像数量
        
    def load_image(self, file_path: str) -> ImageData:
        """加载图像文件"""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"图像文件不存在: {file_path}")
        ext = file_path.suffix.lower()
        
        # 优先使用PIL处理TIFF以正确识别通道顺序（如RGBA/ARGB/CMYK等）
        if ext in [".tif", ".tiff", ".fff"]:
            try:
                pil_image = Image.open(file_path)
                mode = pil_image.mode  # 例如: 'RGB', 'RGBA', 'CMYK', 'I;16', 'F', 'LA' 等
                bands = pil_image.getbands()  # 例如: ('R','G','B','A') 或 ('A','R','G','B') 或 ('C','M','Y','K')

                # 若为CMYK或其它非RGB空间，先转换到RGB
                if mode in ["CMYK", "YCbCr", "LAB"]:
                    pil_image = pil_image.convert("RGB")
                    mode = pil_image.mode
                    bands = pil_image.getbands()

                image = np.array(pil_image)

                # 归一化到[0,1]
                if image.dtype != np.float32:
                    image = image.astype(np.float32)
                    if image.max() > 1.0:
                        # 根据位深度推断范围
                        if image.max() > 255:
                            image /= 65535.0
                        else:
                            image /= 255.0

                # 灰度转为单通道形状 (H,W,1)
                if image.ndim == 2:
                    image = image[:, :, np.newaxis]

                # 处理4通道的通道顺序：优先识别Alpha；否则启发式识别红外IR通道（更“平滑/平均”）并移到最后
                if image.ndim == 3 and image.shape[2] == 4:
                    print(f"[ImageManager] 检测到4通道TIFF: {file_path.name}, mode={mode}, bands={bands}")
                    handled = False
                    # 1) 若bands可用且包含A，按RGBA重排
                    if bands is not None:
                        try:
                            band_list = list(bands)
                            if 'A' in band_list:
                                alpha_idx = band_list.index('A')
                                if set(['R','G','B']).issubset(set(band_list)):
                                    r_idx = band_list.index('R')
                                    g_idx = band_list.index('G')
                                    b_idx = band_list.index('B')
                                    image = image[..., [r_idx, g_idx, b_idx, alpha_idx]]
                                    print(f"[ImageManager] 通过bands识别Alpha通道(index={alpha_idx})，已重排为RGBA顺序")
                                else:
                                    # 仅将Alpha放末尾
                                    rgb_indices = [i for i in range(4) if i != alpha_idx]
                                    image = image[..., rgb_indices + [alpha_idx]]
                                    print(f"[ImageManager] 通过bands识别Alpha通道(index={alpha_idx})，已将Alpha移至最后")
                                handled = True
                        except Exception:
                            handled = False
                    
                    if not handled:
                        # 2) 启发式识别IR通道：其方差/Laplacian方差明显更低（更平滑/平均）
                        # 采样以提速
                        sample = image[::8, ::8, :]
                        H, W, _ = sample.shape
                        ch = sample.reshape(-1, 4)
                        var_spatial = ch.var(axis=0)
                        # 简单梯度能量（近似边缘能量）
                        gx = np.diff(sample, axis=1, prepend=sample[:, :1, :])
                        gy = np.diff(sample, axis=0, prepend=sample[:1, :, :])
                        edge_energy = (gx**2 + gy**2).mean(axis=(0, 1))
                        score = var_spatial + 0.5 * edge_energy
                        candidate = int(np.argmin(score))
                        # 与次小值比较，确保明显更低
                        sorted_scores = np.sort(score)
                        if sorted_scores[0] < 0.5 * sorted_scores[1]:
                            # 将IR放到最后，其余通道保持原相对顺序
                            order = [i for i in range(4) if i != candidate] + [candidate]
                            image = image[..., order]
                            print(f"[ImageManager] 通过启发式识别IR通道(index={candidate})，score={score.round(6).tolist()}，已重排为RGB+IR")
                        else:
                            print(f"[ImageManager] 启发式无法明确识别IR通道，保持原通道顺序，score={score.round(6).tolist()}")

                    # 3) 若存在Alpha通道，则在导入时直接丢弃Alpha，避免影响后续流程
                    drop_alpha = False
                    alpha_index = None
                    # 明确的bands包含A
                    if bands is not None and 'A' in list(bands):
                        # 若前面按bands重排过，此时A应在末位；稳妥起见再次定位索引
                        alpha_index = list(bands).index('A')
                        # 若已重排至RGBA，alpha_index应为3；否则按当前位置删除
                        drop_alpha = True
                    else:
                        # 启发式判断Alpha（近乎常量且接近0或1）
                        sample = image[::8, ::8, :]
                        ch = sample.reshape(-1, 4)
                        vars_ = ch.var(axis=0)
                        means_ = ch.mean(axis=0)
                        for idx in range(4):
                            if vars_[idx] < 1e-6 and (means_[idx] < 0.01 or means_[idx] > 0.99):
                                alpha_index = idx
                                drop_alpha = True
                                break
                    if drop_alpha and alpha_index is not None and 0 <= alpha_index < 4:
                        image = np.delete(image, alpha_index, axis=2)
                        print(f"[ImageManager] 检测到Alpha通道(index={alpha_index})，已在导入时移除。当前shape={image.shape}")

                # 检测单/双通道图像并标记
                original_channels = image.shape[2] if image.ndim == 3 else 1
                is_monochrome_source = original_channels <= 2
                
                # 创建ImageData并返回
                image_data = ImageData(
                    array=image,
                    color_space=None,
                    file_path=str(file_path),
                    is_proxy=False,
                    proxy_scale=1.0,
                    original_channels=original_channels,
                    is_monochrome_source=is_monochrome_source
                )
                return image_data
            except Exception as e:
                # 回退到OpenCV路径
                print(f"[ImageManager] PIL TIFF loading failed, falling back to OpenCV: {type(e).__name__}: {str(e)[:200]}")
                pass
        
        # 尝试使用OpenCV加载（非TIFF优先走此分支）
        opencv_error = None
        try:
            print(f"[ImageManager] Attempting to load image with OpenCV: {file_path.name}")
            image = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError("OpenCV无法加载图像")
            print(f"[ImageManager] OpenCV successfully loaded image: {image.shape}, dtype={image.dtype}")
            
            # OpenCV使用BGR(A)格式，转换为RGB(A)
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif image.shape[2] == 4:
                    print(f"[ImageManager] 检测到4通道图像(BGRA→RGBA): {file_path.name}")
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
                    # 对OpenCV读取的4通道图也进行IR启发式识别（若不存在明显Alpha）
                    sample = image[::8, ::8, :].astype(np.float32) / (255.0 if image.dtype!=np.float32 and image.max()>1.0 else 1.0)
                    ch = sample.reshape(-1, 4)
                    var_spatial = ch.var(axis=0)
                    gx = np.diff(sample, axis=1, prepend=sample[:, :1, :])
                    gy = np.diff(sample, axis=0, prepend=sample[:1, :, :])
                    edge_energy = (gx**2 + gy**2).mean(axis=(0, 1))
                    score = var_spatial + 0.5 * edge_energy
                    candidate = int(np.argmin(score))
                    sorted_scores = np.sort(score)
                    # 若第4通道近似透明掩码（极低方差且均值接近0或1），放行不改。否则按IR处理。
                    means = ch.mean(axis=0)
                    is_alpha_like = (var_spatial[candidate] < 1e-6) and (means[candidate] < 0.01 or means[candidate] > 0.99)
                    if (sorted_scores[0] < 0.5 * sorted_scores[1]) and not is_alpha_like:
                        order = [i for i in range(4) if i != candidate] + [candidate]
                        image = image[..., order]
                        print(f"[ImageManager] 通过启发式识别IR通道(index={candidate})，score={score.round(6).tolist()}，已重排为RGB+IR")
                    else:
                        print(f"[ImageManager] 4通道保持顺序，score={score.round(6).tolist()}，alpha_like={bool(is_alpha_like)}")

                    # 导入时移除Alpha（若存在）
                    # 再次采用启发式：近乎常量且接近0/1的通道视为Alpha
                    sample2 = image[::8, ::8, :].astype(np.float32) / (255.0 if image.dtype!=np.float32 and image.max()>1.0 else 1.0)
                    ch2 = sample2.reshape(-1, 4)
                    vars2 = ch2.var(axis=0)
                    means2 = ch2.mean(axis=0)
                    alpha_idx = None
                    for idx in range(4):
                        if vars2[idx] < 1e-6 and (means2[idx] < 0.01 or means2[idx] > 0.99):
                            alpha_idx = idx
                            break
                    if alpha_idx is not None:
                        image = np.delete(image, alpha_idx, axis=2)
                        print(f"[ImageManager] 检测到Alpha通道(index={alpha_idx})，已在导入时移除。当前shape={image.shape}")
            
            # 转换为float32并归一化到[0,1]
            if image.dtype != np.float32:
                original_dtype = image.dtype
                image = image.astype(np.float32)
                
                # 根据原始数据类型进行正确的归一化
                if original_dtype == np.uint8:
                    # 8bit图像: 0-255 -> 0-1
                    image /= 255.0
                elif original_dtype == np.uint16:
                    # 16bit图像: 0-65535 -> 0-1
                    image /= 65535.0
                elif original_dtype == np.uint32:
                    # 32bit图像: 0-4294967295 -> 0-1
                    image /= 4294967295.0
                elif original_dtype == np.int16:
                    # 16bit有符号: -32768-32767 -> 0-1
                    image = (image + 32768) / 65535.0
                elif original_dtype == np.int32:
                    # 32bit有符号: -2147483648-2147483647 -> 0-1
                    image = (image + 2147483648) / 4294967295.0
                elif image.max() > 1.0:
                    # 其他情况，使用最大值归一化
                    image /= image.max()
            
        except Exception as e:
            # 如果OpenCV失败，尝试使用PIL
            opencv_error = e
            print(f"[ImageManager] OpenCV loading failed: {type(e).__name__}: {str(e)[:200]}")
            max_pixels_str = f"{Image.MAX_IMAGE_PIXELS:,}" if Image.MAX_IMAGE_PIXELS else "Unlimited"
            print(f"[ImageManager] Attempting to load image with PIL (MAX_IMAGE_PIXELS={max_pixels_str})")
            try:
                pil_image = Image.open(file_path)
                original_mode = pil_image.mode
                print(f"[ImageManager] PIL successfully opened image: size={pil_image.size}, mode={original_mode}")
                image = np.array(pil_image, dtype=np.float32)
                
                # 根据PIL模式进行正确的归一化
                if original_mode in ['L', 'RGB', 'RGBA']:
                    # 8bit图像
                    if image.max() > 1.0:
                        image /= 255.0
                elif original_mode in ['I', 'I;16']:
                    # 16bit图像
                    if image.max() > 1.0:
                        image /= 65535.0
                elif original_mode in ['F']:
                    # 32bit浮点图像，通常已经是0-1范围
                    pass
                elif image.max() > 1.0:
                    # 其他情况，使用最大值归一化
                    image /= image.max()
                
                # 处理灰度图像
                if len(image.shape) == 2:
                    image = image[:, :, np.newaxis]

            except Exception as e2:
                # 生成详细的错误报告
                error_msg = f"无法加载图像文件: {file_path.name}\n\n"
                error_msg += f"图像信息:\n"
                error_msg += f"  文件路径: {file_path}\n"
                error_msg += f"  文件大小: {file_path.stat().st_size / (1024*1024):.1f} MB\n"
                error_msg += f"\nOpenCV错误: {type(opencv_error).__name__}: {str(opencv_error)[:150]}\n"
                error_msg += f"PIL错误: {type(e2).__name__}: {str(e2)[:150]}\n"

                # 检查是否是像素限制问题
                if "pixels" in str(opencv_error).lower() or "pixels" in str(e2).lower():
                    error_msg += f"\n可能的原因: 图像尺寸超过了限制\n"
                    error_msg += f"  PIL最大像素限制: {Image.MAX_IMAGE_PIXELS:,} ({Image.MAX_IMAGE_PIXELS/1_000_000:.1f}M)\n"
                    error_msg += f"\n解决方案:\n"
                    error_msg += f"  1. 尝试用其他工具缩小图像尺寸\n"
                    error_msg += f"  2. 检查图像文件是否损坏\n"
                    error_msg += f"  3. 如果是Windows系统，可能需要更新OpenCV版本\n"

                print(f"[ImageManager] FATAL: {error_msg}")
                raise RuntimeError(error_msg)
        
        # 检测单/双通道图像并标记
        if len(image.shape) == 3:
            original_channels = image.shape[2]
        else:
            original_channels = 1
        is_monochrome_source = original_channels <= 2
        
        # 如果存在Alpha通道，暂不丢弃，但在后续色彩空间转换时会自动忽略
        # 创建ImageData对象（默认不设具体空间，由上层默认预设或用户选择决定）
        image_data = ImageData(
            array=image,
            color_space=None,
            file_path=str(file_path),
            is_proxy=False,
            proxy_scale=1.0,
            original_channels=original_channels,
            is_monochrome_source=is_monochrome_source
        )

        # 输出加载成功的信息
        h, w = image.shape[:2]
        total_pixels = h * w
        print(f"[ImageManager] Successfully loaded image: {file_path.name}")
        print(f"[ImageManager]   Size: {w}x{h} ({total_pixels:,} pixels = {total_pixels/1_000_000:.1f}M)")
        print(f"[ImageManager]   Channels: {original_channels}, Monochrome: {is_monochrome_source}")
        print(f"[ImageManager]   Data range: [{image.min():.4f}, {image.max():.4f}]")

        return image_data
    
    def generate_proxy(self, image: ImageData, max_size: Tuple[int, int] = (2000, 2000)) -> ImageData:
        """生成代理图像"""
        # 移除这个检查，允许对代理图像进行进一步缩放
        # if image.is_proxy:
        #     return image
        
        # 处理单/双通道图像转换为3通道用于pipeline兼容
        source_array = image.array.copy()
        if image.is_monochrome_source and image.array is not None:
            if image.array.ndim == 2:
                # 2D灰度图像 → 3通道
                source_array = np.stack([image.array, image.array, image.array], axis=2)
                print(f"[ImageManager] 单通道图像转换为3通道代理: {image.array.shape} → {source_array.shape}")
            elif image.array.ndim == 3 and image.original_channels == 1:
                # 3D单通道 → 3通道（复制）
                gray_channel = image.array[:, :, 0]
                source_array = np.stack([gray_channel, gray_channel, gray_channel], axis=2)
                print(f"[ImageManager] 单通道图像转换为3通道代理: {image.array.shape} → {source_array.shape}")
            elif image.array.ndim == 3 and image.original_channels == 2:
                # 双通道（L+IR）→ 4通道（L,L,L,IR）
                gray_channel = image.array[:, :, 0]
                ir_channel = image.array[:, :, 1]
                source_array = np.stack([gray_channel, gray_channel, gray_channel, ir_channel], axis=2)
                print(f"[ImageManager] 双通道图像转换为4通道代理: {image.array.shape} → {source_array.shape}")
        
        # 计算缩放比例
        h, w = source_array.shape[:2]
        max_w, max_h = max_size
        
        scale_w = max_w / w
        scale_h = max_h / h
        scale = min(scale_w, scale_h, 1.0)  # 不放大图像
        
        if scale >= 1.0:
            # 图像已经足够小，但仍需要通道转换
            proxy_array = source_array
        else:
            # 计算新的尺寸
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # 使用OpenCV进行高质量缩放
            proxy_array = cv2.resize(
                source_array, 
                (new_w, new_h), 
                interpolation=cv2.INTER_LINEAR
            )
        
        # 创建代理ImageData
        proxy_data = ImageData(
            array=proxy_array,
            color_space=image.color_space,
            icc_profile=image.icc_profile,
            metadata=image.metadata,
            file_path=image.file_path,
            is_proxy=True,
            proxy_scale=scale,
            original_channels=image.original_channels,
            is_monochrome_source=image.is_monochrome_source
        )
        
        return proxy_data
    
    def get_cached_proxy(self, image_id: str) -> Optional[ImageData]:
        """获取缓存的代理图像（更新LRU顺序）

        当缓存命中时，将该项移到末尾，标记为最近使用
        """
        proxy = self._proxy_cache.get(image_id)
        if proxy is not None:
            self._proxy_cache.move_to_end(image_id)  # 移到末尾（最近使用）
        return proxy
    
    def cache_proxy(self, image_id: str, proxy: ImageData):
        """缓存代理图像（正确的LRU实现）

        使用 OrderedDict 的 LRU 特性：
        - move_to_end() 将最近使用的项移到末尾
        - popitem(last=False) 移除最旧的项（队首）

        Proxy 图像可能很大（~17MB），确保旧图像被正确释放
        """
        self._proxy_cache[image_id] = proxy
        self._proxy_cache.move_to_end(image_id)  # 移到末尾（最近使用）

        if len(self._proxy_cache) > self._max_cache_size:
            # 移除最旧的缓存项（队首）
            oldest_key, old_proxy = self._proxy_cache.popitem(last=False)
            # 显式释放 ImageData 对象以防止内存泄漏
            if old_proxy is not None:
                # ImageData 持有 numpy array（最大内存占用）
                if old_proxy.array is not None:
                    old_proxy.array = None  # 释放 numpy 数组（~17MB）
                # 释放 ImageData 对象引用
                old_proxy = None
    
    def clear_cache(self):
        """清空缓存并释放资源

        确保所有缓存的 Proxy 图像及其持有的 numpy 数组被正确释放
        Proxy 图像可能很大（每个 ~17MB），总计可达 ~170MB
        """
        # 显式释放所有 Proxy 图像
        for key in list(self._proxy_cache.keys()):
            proxy = self._proxy_cache[key]
            if proxy is not None:
                # 释放 ImageData 持有的 numpy array（最大内存占用）
                if proxy.array is not None:
                    proxy.array = None
                # 释放 ImageData 对象引用
                self._proxy_cache[key] = None
        # 清空字典
        self._proxy_cache.clear()
    
    def get_image_id(self, file_path: str) -> str:
        """生成图像唯一标识符"""
        # 使用文件路径和修改时间的哈希作为ID
        stat = os.stat(file_path)
        content = f"{file_path}_{stat.st_mtime}_{stat.st_size}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def save_image(self, image_data: ImageData, output_path: str, quality: int = 95, bit_depth: int = 8, export_color_space: str = None):
        """保存图像
        - 统一归一化通道形状（灰度 squeeze，JPEG 限制为3通道）
        - 检查 imwrite 返回值，失败抛出异常
        - 根据 bit_depth 决定 8/16 位量化
        - 单色源图像自动还原为单通道并使用灰度ICC
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 确保图像数据在[0,1]范围内
        image_array = np.clip(image_data.array, 0, 1)
        
        # 处理单色源图像：还原为单通道
        force_grayscale_icc = False
        if image_data.is_monochrome_source:
            print(f"[ImageManager] 单色源图像导出：还原为单通道，原始{image_data.original_channels}通道")
            # 取第一个通道作为灰度数据
            if image_array.ndim == 3 and image_array.shape[2] >= 1:
                image_array = image_array[:, :, 0]  # 取第一通道
                force_grayscale_icc = True
                print(f"[ImageManager] 从3通道还原为单通道: {image_array.shape}")
        
        # 覆盖color space为灰度
        if force_grayscale_icc:
            export_color_space = "Gray Gamma 2.2"
        
        # 根据位深度和文件格式选择保存类型
        ext = output_path.suffix.lower()
        
        if bit_depth == 16 and ext in ['.png', '.tiff', '.tif']:
            # 16bit保存
            image_array = (image_array * 65535).astype(np.uint16)
        else:
            # 8bit保存
            image_array = (image_array * 255).astype(np.uint8)
        
        # 根据文件扩展名选择保存格式
        ext = output_path.suffix.lower()

        # 若需要嵌入 ICC，优先使用 Pillow 保存（仅限 JPEG/TIFF），否则走 OpenCV
        def _read_bytes(path: Path) -> Optional[bytes]:
            try:
                with open(path, 'rb') as f:
                    data = f.read()
                    print(f"[ImageManager] Successfully read ICC file: {path} ({len(data)} bytes)")
                    return data
            except FileNotFoundError:
                print(f"[ImageManager] ICC file not found: {path}")
                return None
            except PermissionError:
                print(f"[ImageManager] Permission denied reading ICC file: {path}")
                return None
            except Exception as e:
                print(f"[ImageManager] Error reading ICC file {path}: {e}")
                return None


        def _get_icc_bytes_for_export(image: ImageData, export_cs_name: Optional[str]) -> Optional[bytes]:
            try:
                if image and getattr(image, 'icc_profile', None):
                    print(f"[ImageManager] Using ICC profile from image data")
                    return image.icc_profile
            except Exception:
                pass
            
            if not export_cs_name:
                print(f"[ImageManager] No export color space specified")
                return None
                
            # First try to get ICC filename from ColorSpaceManager
            try:
                from divere.core.color_space import ColorSpaceManager
                # Try to get a global instance or create one
                if hasattr(self, '_color_space_manager'):
                    csm = self._color_space_manager
                else:
                    csm = ColorSpaceManager()
                
                cs_info = csm.get_color_space_info(export_cs_name)
                if cs_info and cs_info.get('icc_profile'):
                    icc_filename = cs_info['icc_profile']
                    print(f"[ImageManager] Found ICC filename in JSON config: {icc_filename}")
                    
                    # Resolve the full path
                    try:
                        icc_dir = get_data_dir("config").joinpath("colorspace", "icc")
                    except Exception as e:
                        icc_dir = Path("config").joinpath("colorspace", "icc")
                        print(f"[ImageManager] Using fallback ICC directory: {icc_dir} (get_data_dir failed: {e})")
                    
                    icc_path = icc_dir / icc_filename
                    print(f"[ImageManager] Resolved ICC path: {icc_path}")
                    
                    if icc_path.exists():
                        return _read_bytes(icc_path)
                    else:
                        print(f"[ImageManager] ⚠️  ICC file referenced in JSON but not found: {icc_path}")
                        print(f"[ImageManager] Please check if the ICC file exists or update the JSON configuration")
                elif cs_info:
                    print(f"[ImageManager] ℹ️  Color space '{export_cs_name}' found but no 'icc_profile' field in JSON configuration")
                    print(f"[ImageManager] To enable ICC embedding, add 'icc_profile': 'filename.icc' to the JSON file")
                else:
                    print(f"[ImageManager] ❌ Color space '{export_cs_name}' not found in configuration")
            except Exception as e:
                print(f"[ImageManager] Error accessing ColorSpaceManager: {e}")
            
            # No hardcoded fallback - purely configuration driven  
            print(f"[ImageManager] 📋 ICC embedding summary for '{export_cs_name}': No ICC profile available")
            print(f"[ImageManager] Image will be saved without embedded ICC profile")
            return None

        # 归一化形状：灰度 squeeze 到 (H,W)，JPEG 只允许 3 通道
        def _normalize_shape_for_saving(arr: np.ndarray, expect_rgb: bool) -> np.ndarray:
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr[:, :, 0]
            if expect_rgb:
                # 若不是3通道，尝试从灰度扩展到3通道；若是4通道仅取前3通道
                if arr.ndim == 2:
                    arr = np.stack([arr, arr, arr], axis=2)
                elif arr.ndim == 3 and arr.shape[2] >= 3:
                    arr = arr[:, :, :3]
                elif arr.ndim != 3 or arr.shape[2] != 3:
                    raise ValueError("JPEG 保存需要3通道或灰度图像")
            else:
                # 非JPEG格式允许 1,3,4 通道
                if arr.ndim == 3 and arr.shape[2] not in (1, 3, 4):
                    # 其他通道数不支持，取前三通道作为回退
                    arr = arr[:, :, :3]
            return arr

        if ext in ['.jpg', '.jpeg']:
            # JPEG格式：限制为3通道，使用BGR
            image_jpeg = _normalize_shape_for_saving(image_array, expect_rgb=True)
            icc_bytes = _get_icc_bytes_for_export(image_data, export_color_space)
            if icc_bytes is not None:
                # 使用 Pillow 保存并嵌入 ICC
                pil_img = Image.fromarray(image_jpeg)
                try:
                    pil_img.save(str(output_path), format='JPEG', quality=int(quality), subsampling=0, optimize=True, icc_profile=icc_bytes)
                    print(f"[ImageManager] JPEG saved successfully with ICC profile: {output_path}")
                except Exception as e:
                    print(f"[ImageManager] Failed to save JPEG with ICC: {e}")
                    raise RuntimeError(f"保存JPEG失败(ICC): {e}")
            else:
                print(f"[ImageManager] Saving JPEG without ICC profile (no ICC data available)")
                bgr_image = cv2.cvtColor(image_jpeg, cv2.COLOR_RGB2BGR)
                ok = cv2.imwrite(str(output_path), bgr_image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
                if not ok:
                    raise RuntimeError(f"保存JPEG失败: {output_path}")
                print(f"[ImageManager] JPEG saved successfully without ICC: {output_path}")
        
        elif ext in ['.png']:
            image_png = _normalize_shape_for_saving(image_array, expect_rgb=False)
            if image_png.ndim == 3 and image_png.shape[2] in (3, 4):
                # OpenCV 期望BGR(A)
                code = cv2.COLOR_RGB2BGR if image_png.shape[2] == 3 else cv2.COLOR_RGBA2BGRA
                bgr_or_bgra = cv2.cvtColor(image_png, code)
                ok = cv2.imwrite(str(output_path), bgr_or_bgra)
            else:
                ok = cv2.imwrite(str(output_path), image_png)
            if not ok:
                raise RuntimeError(f"保存PNG失败: {output_path}")
        
        elif ext in ['.tiff', '.tif']:
            image_tif = _normalize_shape_for_saving(image_array, expect_rgb=False)
            icc_bytes = _get_icc_bytes_for_export(image_data, export_color_space)
            if icc_bytes is not None:
                # 强制使用 tifffile 写入 16-bit/多通道 TIFF 并附带 ICC（无回退）
                try:
                    import tifffile as tiff
                except Exception as e:
                    print(f"[ImageManager] tifffile not available: {e}")
                    raise RuntimeError("需要安装 'tifffile' 以在 TIFF 中嵌入 ICC")

                try:
                    arr = image_tif
                    # 仅保留最多3通道
                    if arr.ndim == 3 and arr.shape[2] > 3:
                        arr = arr[:, :, :3]
                    # photometric
                    if arr.ndim == 2:
                        photometric = 'minisblack'
                    elif arr.ndim == 3 and arr.shape[2] == 3:
                        photometric = 'rgb'
                    else:
                        photometric = 'minisblack'
                    extratags = [(34675, 'B', len(icc_bytes), icc_bytes, True)]
                    # 使用 LZW 压缩以减小体积
                    tiff.imwrite(
                        str(output_path),
                        arr,
                        photometric=photometric,
                        extratags=extratags,
                        compression='lzw'
                    )
                    print(f"[ImageManager] TIFF saved successfully with ICC profile: {output_path}")
                except Exception as e:
                    print(f"[ImageManager] Failed to save TIFF with ICC: {e}")
                    raise RuntimeError(f"保存TIFF失败(ICC): {e}")
            else:
                print(f"[ImageManager] Saving TIFF without ICC profile (no ICC data available)")
                if image_tif.ndim == 3 and image_tif.shape[2] in (3, 4):
                    code = cv2.COLOR_RGB2BGR if image_tif.shape[2] == 3 else cv2.COLOR_RGBA2BGRA
                    bgr_or_bgra = cv2.cvtColor(image_tif, code)
                    ok = cv2.imwrite(str(output_path), bgr_or_bgra)
                else:
                    ok = cv2.imwrite(str(output_path), image_tif)
                if not ok:
                    raise RuntimeError(f"保存TIFF失败: {output_path}")
                print(f"[ImageManager] TIFF saved successfully without ICC: {output_path}")
        
        else:
            # 默认使用PIL保存（尽量兼容）
            pil_image = Image.fromarray(image_array if image_array.ndim != 3 or image_array.shape[2] != 1 else image_array[:, :, 0])
            pil_kwargs = {}
            if ext in ['.jpg', '.jpeg']:
                pil_kwargs['quality'] = int(quality)
            pil_image.save(output_path, **pil_kwargs)
    
    def get_supported_formats(self) -> list:
        """获取支持的图像格式"""
        return [
            '.jpg', '.jpeg', '.png', '.tiff', '.tif', '.fff', 
            '.bmp', '.webp', '.exr', '.hdr'
        ]
    
    def is_supported_format(self, file_path: str) -> bool:
        """检查文件格式是否支持"""
        ext = Path(file_path).suffix.lower()
        return ext in self.get_supported_formats() 