"""
色彩空间管理模块
处理色彩空间转换和管理
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
from collections import OrderedDict
import colour
import json
import os

from .data_types import ImageData


def uv_to_xy(u_prime: Any, v_prime: Any) -> Tuple[Any, Any]:
    """
    将 CIE 1976 UCS 坐标 (u', v') 转换为 CIE 1931 xy 色度坐标。

    公式:
        x = 9 u' / (6 u' - 16 v' + 12)
        y = 4 v' / (6 u' - 16 v' + 12)

    支持标量与 NumPy 数组，返回的类型与输入一致（标量返回float，数组返回ndarray）。
    """
    u = np.asarray(u_prime)
    v = np.asarray(v_prime)
    denom = 6.0 * u - 16.0 * v + 12.0
    x = np.divide(9.0 * u, denom, out=np.full_like(u, np.nan, dtype=float), where=denom != 0)
    y = np.divide(4.0 * v, denom, out=np.full_like(v, np.nan, dtype=float), where=denom != 0)

    if np.isscalar(u_prime) and np.isscalar(v_prime):
        return float(x), float(y)
    return x, y


def xy_to_uv(x: Any, y: Any) -> Tuple[Any, Any]:
    """
    将 CIE 1931 xy 色度坐标转换为 CIE 1976 UCS 坐标 (u', v')。

    公式:
        u' = 4 x / (-2 x + 12 y + 3)
        v' = 9 y / (-2 x + 12 y + 3)

    支持标量与 NumPy 数组，返回的类型与输入一致（标量返回float，数组返回ndarray）。
    """
    xx = np.asarray(x)
    yy = np.asarray(y)
    denom = -2.0 * xx + 12.0 * yy + 3.0
    u = np.divide(4.0 * xx, denom, out=np.full_like(xx, np.nan, dtype=float), where=denom != 0)
    v = np.divide(9.0 * yy, denom, out=np.full_like(yy, np.nan, dtype=float), where=denom != 0)

    if np.isscalar(x) and np.isscalar(y):
        return float(u), float(v)
    return u, v


class ColorSpaceManager:
    """色彩空间管理器"""
    
    def __init__(self):
        # 从JSON文件加载色彩空间定义
        self._color_spaces = {}
        # 轻量调试开关（通过环境变量启用详细加载日志）——必须在加载前就初始化
        try:
            import os
            self._verbose_logs: bool = bool(int(os.environ.get('DIVERE_VERBOSE', '0')))
        except Exception:
            self._verbose_logs = False
        self._load_colorspaces_from_json()
        
        # 设置monochrome色彩空间
        self.setup_monochrome_color_space()
        
        # 不再需要预计算转换矩阵，使用在线计算
        # 增加一个简单的转换缓存，加速重复转换
        self._convert_cache: "OrderedDict[Any, Tuple[np.ndarray, np.ndarray]]" = OrderedDict()  # 使用 OrderedDict 实现正确的 LRU
        self._convert_cache_max_size: int = 100  # 限制缓存大小，防止无限增长
        # Profiling 开关
        self._profiling_enabled: bool = False
        
        # 工作空间管理
        self._working_space: str = "ACEScg"  # 默认工作空间
        self._load_working_space_from_config()
    def set_profiling_enabled(self, enabled: bool) -> None:
        self._profiling_enabled = bool(enabled)

    def is_profiling_enabled(self) -> bool:
        return self._profiling_enabled
    
    def _load_colorspaces_from_json(self):
        """从JSON文件加载色彩空间定义（支持用户配置优先）"""
        try:
            from divere.utils.enhanced_config_manager import enhanced_config_manager
            
            # 获取所有配置文件（用户配置优先）
            config_files = enhanced_config_manager.get_config_files("colorspace")
            
            for json_file in config_files:
                try:
                    data = enhanced_config_manager.load_config_file(json_file)
                    if data is None:
                        continue
                    
                    # 将primaries转换为numpy数组格式
                    if "primaries" in data and isinstance(data["primaries"], dict):
                        primaries = np.array([
                            data["primaries"]["R"],
                            data["primaries"]["G"],
                            data["primaries"]["B"]
                        ])
                        data["primaries"] = primaries
                    
                    # 将white_point转换为numpy数组
                    if "white_point" in data:
                        data["white_point"] = np.array(data["white_point"])
                    
                    # 使用文件名（不含扩展名）作为色彩空间名称
                    colorspace_name = json_file.stem
                    self._color_spaces[colorspace_name] = data
                    
                    # 标记是否为用户配置
                    if self._verbose_logs:
                        if json_file.parent == enhanced_config_manager.user_colorspace_dir:
                            print(f"加载用户色彩空间: {colorspace_name}")
                        else:
                            print(f"加载内置色彩空间: {colorspace_name}")
                    
                except Exception as e:
                    print(f"加载色彩空间配置文件 {json_file} 时出错: {e}")
                    
        except ImportError:
            # 如果增强配置管理器不可用，使用原来的方法
            colorspace_dir = Path("config/colorspace")
            if not colorspace_dir.exists():
                if self._verbose_logs:
                    print(f"警告：色彩空间配置目录 {colorspace_dir} 不存在")
                return
                
            for json_file in colorspace_dir.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # 将primaries转换为numpy数组格式
                    if "primaries" in data and isinstance(data["primaries"], dict):
                        primaries = np.array([
                            data["primaries"]["R"],
                            data["primaries"]["G"],
                            data["primaries"]["B"]
                        ])
                        data["primaries"] = primaries
                        
                    # 将white_point转换为numpy数组
                    if "white_point" in data:
                        data["white_point"] = np.array(data["white_point"])
                        
                    # 使用文件名（不含扩展名）作为色彩空间名称
                    colorspace_name = json_file.stem
                    self._color_spaces[colorspace_name] = data
                    
                except Exception as e:
                    if self._verbose_logs:
                        print(f"加载色彩空间配置文件 {json_file} 时出错: {e}")
    
    def _build_conversion_matrices(self):
        """构建色彩空间转换矩阵"""
        # 现在使用在线计算，不再预计算所有矩阵
        # 只在需要时动态计算转换矩阵和增益向量
        pass

    # --- 自定义色彩空间注册 ---
    def register_custom_colorspace(self, name: str, primaries_xy: np.ndarray, white_point_xy: Optional[np.ndarray] = None, gamma: float = 1.0) -> None:
        """
        动态注册或更新一个自定义色彩空间，用于临时预览或交互。

        Args:
            name: 色彩空间名称（将作为键存储）。
            primaries_xy: 形状为 (3, 2) 的数组，按 R、G、B 顺序为 xy 坐标。
            white_point_xy: 长度为 2 的数组，xy 白点。缺省使用 D65。
            gamma: 色彩空间的 gamma，扫描线性数据通常为 1.0。
        """
        if white_point_xy is None:
            white_point_xy = np.array([0.3127, 0.3290], dtype=float)  # D65
        data = {
            "name": name,
            "primaries": np.array(primaries_xy, dtype=float),
            "white_point": np.array(white_point_xy, dtype=float),
            "gamma": float(gamma),
        }
        self._color_spaces[name] = data
        # 该空间相关的转换结果需要失效
        self._invalidate_convert_cache_for(name)

    def is_custom_color_space(self, name: str) -> bool:
        """检查一个色彩空间是否是自定义的（基于命名约定）"""
        return "_custom" in name or "_preset" in name

    def get_color_space_definition(self, name: str) -> Optional[Dict[str, Any]]:
        """获取自定义色彩空间的定义，用于保存"""
        space = self._color_spaces.get(name)
        if not space:
            return None
        return {
            "primaries_xy": space["primaries"].tolist(),
            "white_point_xy": space["white_point"].tolist(),
            "gamma": space.get("gamma", 1.0)
        }

    def clear_convert_cache(self) -> None:
        """清空转换缓存并释放资源

        确保所有缓存的转换矩阵和增益向量被正确释放
        """
        for key in list(self._convert_cache.keys()):
            entry = self._convert_cache.get(key)
            if entry:
                # 释放 numpy 数组
                matrix, vector = entry
                if matrix is not None:
                    matrix = None
                if vector is not None:
                    vector = None
                self._convert_cache[key] = None
        self._convert_cache.clear()

    def _invalidate_convert_cache_for(self, space_name: str) -> None:
        """使包含指定色彩空间的缓存项失效

        当色彩空间定义更新时，需要清除所有相关的转换缓存
        确保 numpy 数组被正确释放
        """
        try:
            keys_to_delete = [k for k in self._convert_cache.keys() if space_name in k]
            for k in keys_to_delete:
                entry = self._convert_cache[k]
                if entry:
                    # 释放缓存的 numpy 数组
                    matrix, vector = entry
                    if matrix is not None:
                        matrix = None
                    if vector is not None:
                        vector = None
                    self._convert_cache[k] = None
                del self._convert_cache[k]
        except Exception:
            # 失败时清空整个缓存（调用统一方法）
            self.clear_convert_cache()
    
    def calculate_color_space_conversion(self, src_space_name: str, dst_space_name: str) -> tuple[np.ndarray, np.ndarray]:
        """
        计算色彩空间转换矩阵和增益向量
        
        Args:
            src_space_name: 源色彩空间名称
            dst_space_name: 目标色彩空间名称
            
        Returns:
            tuple: (转换矩阵, 增益向量) - 3x3矩阵和长度为3的向量
        """
        if src_space_name == dst_space_name:
            return np.eye(3), np.array([1.0, 1.0, 1.0])
        
        # 缓存命中（更新LRU顺序）
        cache_key = (src_space_name, dst_space_name)
        cached = self._convert_cache.get(cache_key)
        if cached is not None:
            self._convert_cache.move_to_end(cache_key)  # 移到末尾（最近使用）
            return cached

        # 获取源和目标色彩空间信息
        src_space = self._color_spaces.get(src_space_name)
        dst_space = self._color_spaces.get(dst_space_name)
        
        if src_space is None or dst_space is None:
            if self._verbose_logs:
                print(f"警告: 未找到色彩空间定义，使用单位矩阵")
            return np.eye(3), np.array([1.0, 1.0, 1.0])
        
        try:
            # 计算源色彩空间的RGB到XYZ矩阵
            src_matrix = self._calculate_rgb_to_xyz_matrix(src_space)
            
            # 计算目标色彩空间的RGB到XYZ矩阵
            dst_matrix = self._calculate_rgb_to_xyz_matrix(dst_space)
            
            # 计算XYZ到目标RGB的矩阵（目标矩阵的逆）
            dst_matrix_inv = np.linalg.inv(dst_matrix)
            
            # 计算从源RGB到目标RGB的转换矩阵
            conversion_matrix = np.dot(dst_matrix_inv, src_matrix)
            
            # 计算白点适应增益向量
            gain_vector = self._calculate_white_point_adaptation(src_space, dst_space)

            # 缓存结果（带LRU管理）
            self._convert_cache[cache_key] = (conversion_matrix.astype(np.float64), gain_vector.astype(np.float64))
            self._convert_cache.move_to_end(cache_key)  # 移到末尾（最近使用）

            # LRU驱逐：如果缓存超过限制，移除最旧的项
            if len(self._convert_cache) > self._convert_cache_max_size:
                oldest_key, (old_matrix, old_vector) = self._convert_cache.popitem(last=False)
                # 显式释放numpy数组以防止内存泄漏
                if old_matrix is not None:
                    old_matrix = None
                if old_vector is not None:
                    old_vector = None

            return self._convert_cache[cache_key]
            
        except Exception as e:
            if self._verbose_logs or self._profiling_enabled:
                print(f"色彩空间转换计算失败: {e}")
            return np.eye(3), np.array([1.0, 1.0, 1.0])
    
    def _calculate_rgb_to_xyz_matrix(self, color_space: dict) -> np.ndarray:
        """
        根据RGB基色和白点计算RGB到XYZ的转换矩阵
        
        Args:
            color_space: 色彩空间信息字典，包含primaries和white_point
            
        Returns:
            3x3的RGB到XYZ转换矩阵
        """
        primaries = color_space['primaries']  # [[Rx,Ry], [Gx,Gy], [Bx,By]]
        white_point = color_space['white_point']  # [Wx, Wy]
        
        # 将xy色度坐标转换为XYZ坐标（假设Y=1）
        def xy_to_XYZ(xy):
            x, y = xy
            if y == 0:
                return np.array([0, 0, 0])
            X = x / y
            Y = 1.0
            Z = (1 - x - y) / y
            return np.array([X, Y, Z])
        
        # 计算RGB基色的XYZ坐标
        R_XYZ = xy_to_XYZ(primaries[0])  # 红色基色
        G_XYZ = xy_to_XYZ(primaries[1])  # 绿色基色
        B_XYZ = xy_to_XYZ(primaries[2])  # 蓝色基色
        
        # 计算白点的XYZ坐标
        W_XYZ = xy_to_XYZ(white_point)
        
        # 构建基色矩阵 [Rx Gx Bx; Ry Gy By; Rz Gz Bz]
        primaries_matrix = np.column_stack([R_XYZ, G_XYZ, B_XYZ])
        
        # 求解标量因子，使得 primaries_matrix * [Sr, Sg, Sb]^T = W_XYZ
        try:
            # 检查矩阵条件数，避免数值不稳定
            cond_num = np.linalg.cond(primaries_matrix)
            if cond_num > 1e12:  # 条件数过大，使用伪逆
                if self._verbose_logs:
                    print(f"警告: 基色矩阵条件数过大 ({cond_num:.2e})，使用伪逆求解")
                scaling_factors = np.linalg.pinv(primaries_matrix) @ W_XYZ
            else:
                scaling_factors = np.linalg.solve(primaries_matrix, W_XYZ)
        except (np.linalg.LinAlgError, RuntimeError, MemoryError) as e:
            if self._verbose_logs:
                print(f"警告: 矩阵求解失败 ({type(e).__name__}: {e})，使用默认缩放因子")
            scaling_factors = np.array([1.0, 1.0, 1.0])
        
        # 构建最终的RGB到XYZ转换矩阵
        rgb_to_xyz_matrix = primaries_matrix * scaling_factors[np.newaxis, :]
        
        return rgb_to_xyz_matrix
    
    def _calculate_white_point_adaptation(self, src_space: dict, dst_space: dict) -> np.ndarray:
        """
        计算白点适应增益向量
        
        Args:
            src_space: 源色彩空间信息
            dst_space: 目标色彩空间信息
            
        Returns:
            长度为3的增益向量
        """
        src_white = src_space['white_point']
        dst_white = dst_space['white_point']
        
        # 简化的白点适应：基于白点XYZ坐标的比值
        def xy_to_XYZ_normalized(xy):
            x, y = xy
            if y == 0:
                return np.array([1, 1, 1])
            X = x / y
            Y = 1.0
            Z = (1 - x - y) / y
            return np.array([X, Y, Z])
        
        src_white_XYZ = xy_to_XYZ_normalized(src_white)
        dst_white_XYZ = xy_to_XYZ_normalized(dst_white)
        
        # 计算白点适应增益（避免除零）
        gain_vector = np.divide(dst_white_XYZ, src_white_XYZ, 
                               out=np.ones(3), where=src_white_XYZ!=0)
        
        # 限制增益范围，避免极端值
        gain_vector = np.clip(gain_vector, 0.1, 10.0)
        
        return gain_vector
    
    def _get_colour_space_name(self, space: dict) -> str:
        """获取colour库中的色彩空间名称"""
        # 映射自定义名称到colour库名称
        if 'primaries' in space:
            primaries = space['primaries']
            # sRGB
            if np.allclose(primaries[0], [0.6400, 0.3300], atol=1e-3):
                return 'sRGB'
            # ACEScg  
            elif np.allclose(primaries[0], [0.7130, 0.2930], atol=1e-3):
                return 'ACEScg'
            # Adobe RGB
            elif np.allclose(primaries[0], [0.6400, 0.3300], atol=1e-3) and \
                 np.allclose(primaries[1], [0.2100, 0.7100], atol=1e-3):
                return 'Adobe RGB (1998)'
        
        return 'sRGB'  # 默认
    

    
    def get_available_color_spaces(self) -> list:
        """获取可用的色彩空间列表"""
        return sorted(list(self._color_spaces.keys()))
    
    def get_idt_color_spaces(self) -> list:
        """获取IDT色彩空间列表（type包含'IDT'的色彩空间）"""
        idt_spaces = []
        for name, space in self._color_spaces.items():
            space_type = space.get("type", [])
            if isinstance(space_type, str):
                space_type = [space_type]
            if "IDT" in space_type:
                idt_spaces.append(name)
        return sorted(idt_spaces)
    
    def get_regular_color_spaces_with_icc(self) -> list:
        """获取regular色彩空间列表（type包含'regular'且有icc_profile的色彩空间）"""
        regular_spaces = []
        for name, space in self._color_spaces.items():
            space_type = space.get("type", [])
            if isinstance(space_type, str):
                space_type = [space_type]
            icc_profile = space.get("icc_profile", "")
            if "regular" in space_type and icc_profile:
                regular_spaces.append(name)
        return sorted(regular_spaces)
    
    def get_working_color_spaces(self) -> list:
        """获取可用的工作色彩空间（type包含'working_space'的色彩空间）"""
        working_spaces = []
        for name, space in self._color_spaces.items():
            space_type = space.get("type", [])
            if isinstance(space_type, str):
                space_type = [space_type]
            if "working_space" in space_type:
                working_spaces.append(name)
        return sorted(working_spaces)
    
    def get_current_working_space(self) -> str:
        """获取当前工作空间"""
        return self._working_space
    
    def get_working_space_white_point(self) -> str:
        """
        获取当前工作空间的白点字符串
        
        Returns:
            白点字符串 ("D50", "D55", "D60", "D65")
        """
        return self.get_colorspace_white_point(self._working_space)
    
    def get_colorspace_white_point(self, colorspace_name: str) -> str:
        """
        获取指定色彩空间的白点字符串
        
        Args:
            colorspace_name: 色彩空间名称
            
        Returns:
            白点字符串 ("D50", "D55", "D60", "D65")
        """
        if colorspace_name not in self._color_spaces:
            # 默认返回D65（最常见的白点）
            return "D65"
        
        space = self._color_spaces[colorspace_name]
        white_point_xy = space.get('white_point')
        
        if white_point_xy is None:
            return "D65"
        
        # 将xy坐标转换为标准光源名称
        return self._xy_to_illuminant_name(white_point_xy)
    
    def _xy_to_illuminant_name(self, white_point_xy: np.ndarray) -> str:
        """
        将xy白点坐标转换为标准光源名称
        
        Args:
            white_point_xy: [x, y] 坐标
            
        Returns:
            最接近的标准光源名称
        """
        from divere.core.color_science import STANDARD_ILLUMINANTS
        
        # 将标准光源的XYZ转换为xy坐标进行比较
        def xyz_to_xy(xyz):
            total = xyz.sum()
            if total == 0:
                return np.array([0.3127, 0.3290])  # D65 fallback
            return xyz[:2] / total
        
        min_distance = float('inf')
        closest_illuminant = "D65"
        
        for name, xyz in STANDARD_ILLUMINANTS.items():
            standard_xy = xyz_to_xy(xyz)
            distance = np.linalg.norm(white_point_xy - standard_xy)
            if distance < min_distance:
                min_distance = distance
                closest_illuminant = name
        
        return closest_illuminant
    
    def set_working_space(self, space_name: str) -> bool:
        """设置工作空间（验证类型）"""
        if space_name in self.get_working_color_spaces():
            if self._working_space != space_name:
                self._working_space = space_name
                self.clear_convert_cache()  # 清空转换缓存（使用统一方法）
                self._save_working_space_to_config(space_name)  # 保存到配置文件
                if self._verbose_logs:
                    print(f"工作空间已切换到: {space_name}")
            return True
        if self._verbose_logs:
            print(f"无效的工作空间: {space_name}")
        return False
    
    def _load_working_space_from_config(self):
        """从配置文件加载工作空间设置"""
        try:
            from divere.utils.enhanced_config_manager import enhanced_config_manager
            working_space = enhanced_config_manager.get_default_setting("working_color_space", "ACEScg")
            if self.set_working_space(working_space):
                return
        except Exception:
            pass
        
        # 回退到默认值
        self._working_space = "ACEScg"
        if self._verbose_logs:
            print(f"使用默认工作空间: {self._working_space}")
    
    def _save_working_space_to_config(self, space_name: str) -> bool:
        """保存工作空间设置到配置文件"""
        try:
            from divere.utils.enhanced_config_manager import enhanced_config_manager
            enhanced_config_manager.set_default_setting("working_color_space", space_name)
            
            if self._verbose_logs:
                print(f"工作空间配置已保存: {space_name}")
            return True
        except Exception as e:
            if self._verbose_logs:
                print(f"保存配置失败: {e}")
            return False
    
    def reload_config(self):
        """重新加载色彩空间配置文件"""
        self._color_spaces.clear()
        self.clear_convert_cache()  # 清空转换缓存（使用统一方法）
        self._load_colorspaces_from_json()
        self.setup_monochrome_color_space()
    
    def validate_color_space(self, color_space_name: str) -> bool:
        """验证色彩空间名称是否有效"""
        return color_space_name in self._color_spaces
    
    def get_color_space_info(self, color_space_name: str) -> Optional[dict]:
        """获取色彩空间详细信息"""
        return self._color_spaces.get(color_space_name, None)
    
    def is_grayscale_colorspace(self, color_space_name: str) -> bool:
        """判断色彩空间是否为灰度空间"""
        if not color_space_name:
            return False
        
        # 检查色彩空间配置中的type字段
        color_space_info = self.get_color_space_info(color_space_name)
        if color_space_info and color_space_info.get("type") == "grayscale":
            return True
        
        # 基于名称模式的检测（后备方案）
        name_lower = color_space_name.lower()
        grayscale_patterns = ["gray", "grey", "grayscale", "greyscale", "mono", "monochrome"]
        return any(pattern in name_lower for pattern in grayscale_patterns)
    
    def get_color_colorspaces(self) -> list:
        """获取色彩空间列表（排除灰度空间）"""
        all_spaces = self.get_available_color_spaces()
        return [space for space in all_spaces if not self.is_grayscale_colorspace(space)]
    
    def get_grayscale_colorspaces(self) -> list:
        """获取灰度色彩空间列表"""
        all_spaces = self.get_available_color_spaces()
        return [space for space in all_spaces if self.is_grayscale_colorspace(space)]
    
    def set_image_color_space(self, image: ImageData, color_space: str) -> ImageData:
        """设置图像的色彩空间"""
        if not self.validate_color_space(color_space):
            if self._verbose_logs:
                print(f"无效的色彩空间: {color_space}，使用默认值")
            try:
                from divere.utils.defaults import load_default_preset
                color_space = load_default_preset().input_transformation.name or "KodakEnduraPremier"
            except Exception:
                color_space = "KodakEnduraPremier"
        
        # 创建新的图像数据对象，更新色彩空间信息
        new_image = ImageData(
            array=image.array.copy(),
            width=image.width,
            height=image.height,
            channels=image.channels,
            color_space=color_space,
            file_path=image.file_path,
            is_proxy=image.is_proxy,
            proxy_scale=image.proxy_scale,
            metadata=image.metadata  # 保留并传递元数据（如 source_wh/crop_overlay 等）
        )
        if self._verbose_logs or self._profiling_enabled:
            print(f"设置图像色彩空间: {image.color_space} -> {color_space}")
        return new_image
    
    def convert_to_working_space(self, image: ImageData, source_profile: str = None,
                                 skip_gamma_inverse: bool = False) -> ImageData:
        """转换到工作色彩空间
        Args:
            image: 输入图像
            source_profile: 源空间名（默认使用 image.color_space）
            skip_gamma_inverse: 为 True 时跳过逆伽马线性化，仅做矩阵与白点变换
        """
        if image.color_space == self._working_space:
            return image
        
        # 如果指定了source_profile参数，使用它；否则使用图像的color_space
        source_space = source_profile if source_profile else image.color_space
        
        import time
        t0 = time.time()
        if skip_gamma_inverse:
            # 跳过逆伽马：直接使用原图作为“线性”输入（假定上游已前置幂次）
            linear_image = ImageData(
                array=image.array.copy(),
                width=image.width,
                height=image.height,
                channels=image.channels,
                dtype=image.dtype,
                color_space=image.color_space,
                icc_profile=image.icc_profile,
                metadata=image.metadata,
                file_path=image.file_path,
                is_proxy=image.is_proxy,
                proxy_scale=image.proxy_scale
            )
            t1 = time.time()
        else:
            # 先转换到线性空间（通过逆伽马）
            linear_image = self._convert_to_linear(image, source_space)
            t1 = time.time()
        
        # 然后转换到工作空间
        if source_space != self._working_space:
            # 使用在线计算的转换矩阵和增益向量
            t2 = time.time()
            conversion_matrix, gain_vector = self.calculate_color_space_conversion(source_space, self._working_space)
            t3 = time.time()
            linear_image.array = self._apply_color_conversion(linear_image.array, conversion_matrix, gain_vector)
            t4 = time.time()
            if self._profiling_enabled:
                msg = (
                    f"到工作空间Profiling: "
                    f"gamma逆变换={(t1 - t0)*1000:.1f}ms, 计算矩阵={(t3 - t2)*1000:.1f}ms, 应用矩阵={(t4 - t3)*1000:.1f}ms"
                )
                print(msg)
        
        linear_image.color_space = self._working_space
        return linear_image
    
    def convert_to_display_space(self, image: ImageData, target_space: str = "sRGB") -> ImageData:
        """转换到显示色彩空间"""
        import time
        
        if image.color_space == target_space:
            return image
        
        # 从工作空间转换到目标空间
        if image.color_space == self._working_space:
            # 使用在线计算的转换矩阵和增益向量
            t0 = time.time()
            conversion_matrix, gain_vector = self.calculate_color_space_conversion(self._working_space, target_space)
            t1 = time.time()
            image.array = self._apply_color_conversion(image.array, conversion_matrix, gain_vector)
            t2 = time.time()
        
        # 应用gamma校正
        t3 = time.time()
        image.array = self._apply_gamma(image.array, self._color_spaces[target_space]["gamma"])
        t4 = time.time()
        image.color_space = target_space
        
        if self._profiling_enabled:
            print(
                f"显示空间转换Profiling: 计算矩阵={(t1 - t0)*1000 if 't1' in locals() else 0:.1f}ms, 应用矩阵={(t2 - t1)*1000 if 't2' in locals() else 0:.1f}ms, gamma={(t4 - t3)*1000:.1f}ms"
            )

        return image
    
    def _convert_to_linear(self, image: ImageData, source_space: str) -> ImageData:
        """转换到线性空间"""
        gamma = self._color_spaces.get(source_space, {}).get("gamma", 2.2)
        
        # 应用gamma校正（从非线性到线性）
        linear_array = self._apply_gamma(image.array, gamma, inverse=True)
        
        linear_image = ImageData(
            array=linear_array,
            width=image.width,
            height=image.height,
            channels=image.channels,
            dtype=image.dtype,
            color_space=f"{source_space}_Linear",
            icc_profile=image.icc_profile,
            metadata=image.metadata,
            file_path=image.file_path,
            is_proxy=image.is_proxy,
            proxy_scale=image.proxy_scale
        )
        
        return linear_image
    
    def _apply_gamma(self, image_array: np.ndarray, gamma: float, inverse: bool = False) -> np.ndarray:
        """应用gamma校正"""
        # 确保图像数据在[0,1]范围内
        image_array = np.clip(image_array, 0, 1)
        
        if inverse:
            # 从非线性到线性：I_linear = I_nonlinear^gamma
            return np.power(image_array, gamma)
        else:
            # 从线性到非线性：I_nonlinear = I_linear^(1/gamma)
            return np.power(image_array, 1.0 / gamma)
    
    def _apply_color_conversion(self, image_array: np.ndarray, matrix: np.ndarray, gain_vector: np.ndarray) -> np.ndarray:
        """应用色彩矩阵变换和增益校正"""
        # 重塑图像为2D数组以便矩阵乘法
        original_shape = image_array.shape
        
        # 处理单通道图像
        if len(original_shape) == 2:
            # 2D灰度图像，复制为3通道处理
            mono_array = image_array[..., np.newaxis]
            rgb_array = np.repeat(mono_array, 3, axis=-1)
            h, w, c = rgb_array.shape
            rgb = rgb_array.reshape(-1, 3)
            # 应用矩阵与白点增益
            transformed = np.dot(rgb, matrix.T)
            transformed *= gain_vector[np.newaxis, :]
            # 转换回单通道（取绿色通道）
            result_rgb = transformed.reshape(h, w, 3)
            return result_rgb[..., 1]  # 返回2D灰度图
        elif len(original_shape) == 3:
            h, w, c = original_shape
            if c == 1:
                # 单通道图像，复制为3通道处理
                rgb_array = np.repeat(image_array, 3, axis=-1)
                rgb = rgb_array.reshape(-1, 3)
                # 应用矩阵与白点增益
                transformed = np.dot(rgb, matrix.T)
                transformed *= gain_vector[np.newaxis, :]
                # 转换回单通道（取绿色通道）
                result_rgb = transformed.reshape(h, w, 3)
                return result_rgb[..., 1:2]  # 保持单通道维度
            elif c >= 3:
                # 多通道图像，仅对前3个通道做矩阵变换（忽略Alpha）
                rgb = image_array[..., :3].reshape(-1, 3)
                # 应用矩阵与白点增益
                transformed = np.dot(rgb, matrix.T)
                transformed *= gain_vector[np.newaxis, :]
                result = image_array.copy()
                result[..., :3] = transformed.reshape(h, w, 3)
                return result
            else:
                # 其他通道数，直接返回
                return image_array
        else:
            return image_array
    
    def _apply_color_matrix(self, image_array: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """应用色彩矩阵变换（保留用于向后兼容）"""
        return self._apply_color_conversion(image_array, matrix, np.array([1.0, 1.0, 1.0]))
    
    def convert_xyz_to_working_space_rgb(self, xyz: np.ndarray, working_colorspace: str) -> np.ndarray:
        """
        将XYZ值转换为工作色彩空间RGB
        
        Args:
            xyz: XYZ值 (shape可以是(3,)或(N,3))
            working_colorspace: 工作色彩空间名称
            
        Returns:
            RGB值，shape与输入相同
        """
        # 获取工作色彩空间定义
        workspace_info = self._color_spaces.get(working_colorspace)
        if workspace_info is None:
            if self._verbose_logs:
                print(f"警告: 未找到工作色彩空间定义 {working_colorspace}，使用sRGB")
            workspace_info = self._color_spaces.get("sRGB")
            if workspace_info is None:
                raise ValueError(f"无法找到工作色彩空间定义: {working_colorspace}")
        
        try:
            # 计算工作空间的RGB到XYZ矩阵
            rgb_to_xyz_matrix = self._calculate_rgb_to_xyz_matrix(workspace_info)
            
            # 计算XYZ到RGB矩阵（RGB到XYZ矩阵的逆）
            xyz_to_rgb_matrix = np.linalg.inv(rgb_to_xyz_matrix)
            
            # 准备XYZ数据进行矩阵运算
            xyz = np.asarray(xyz, dtype=np.float64)
            original_shape = xyz.shape
            
            if len(original_shape) == 1:
                # 单个XYZ值 (3,)
                rgb = xyz_to_rgb_matrix @ xyz
                return rgb
            elif len(original_shape) == 2 and original_shape[1] == 3:
                # 多个XYZ值 (N, 3)
                rgb = (xyz_to_rgb_matrix @ xyz.T).T
                return rgb
            else:
                raise ValueError(f"不支持的XYZ数据形状: {original_shape}")
                
        except Exception as e:
            if self._verbose_logs:
                print(f"XYZ到RGB转换失败: {e}")
            # 作为最后手段，返回输入值（假设已经是RGB）
            return np.asarray(xyz, dtype=np.float64)
    
    def get_default_color_space(self) -> str:
        """获取默认色彩空间（集中读取 default preset）"""
        try:
            from divere.utils.defaults import load_default_preset
            return load_default_preset().input_transformation.name or "KodakEnduraPremier"
        except Exception:
            return "KodakEnduraPremier"
    

    
    def estimate_source_gamma(self, image: ImageData) -> float:
        """估算源图像的gamma值"""
        # 简单的gamma估算方法
        # 基于图像直方图的分布特征
        
        if len(image.array.shape) == 3:
            channels = image.array.shape[2]
            if channels >= 3:
                # 使用RGB亮度加权
                gray = np.dot(image.array[..., :3], [0.299, 0.587, 0.114])
            elif channels == 1:
                # 单通道图像，直接使用
                gray = image.array[..., 0]
            elif channels == 2:
                # 双通道图像，使用第一个通道
                gray = image.array[..., 0]
            else:
                # 其他情况，使用第一个通道
                gray = image.array[..., 0]
        else:
            gray = image.array
        
        # 计算累积分布函数
        hist, bins = np.histogram(gray.flatten(), bins=256, range=(0, 1))
        cdf = np.cumsum(hist) / np.sum(hist)
        
        # 找到50%分位数
        mid_point = np.argmax(cdf >= 0.5) / 256.0
        
        # 基于中值估算gamma
        if mid_point > 0:
            estimated_gamma = np.log(0.5) / np.log(mid_point)
            return np.clip(estimated_gamma, 1.0, 3.0)
        
        return 2.2  # 默认值
    
    def apply_white_balance(self, image: ImageData, temperature: float, tint: float) -> ImageData:
        """应用白平衡校正"""
        # 简化的白平衡实现
        # temperature: 色温 (K)
        # tint: 色调偏移
        
        # 计算色温转换矩阵
        # 这里使用简化的转换，实际应用中需要更复杂的算法
        if temperature != 6500:  # 6500K为标准白点
            # 简化的色温调整
            ratio = 6500 / temperature
            matrix = np.array([
                [ratio, 0, 0],
                [0, 1, 0], 
                [0, 0, 1/ratio]
            ])
            
            image.array = self._apply_color_matrix(image.array, matrix)
        
        return image
    
    # =================
    # Monochrome Support for Black & White Films
    # =================
    
    def convert_to_monochrome(self, image: ImageData, preserve_ir: bool = True) -> ImageData:
        """
        将彩色图像转换为单色（黑白）图像

        Args:
            image: 输入图像
            preserve_ir: 是否保留红外通道（如果存在）

        Returns:
            转换后的单色图像
        """
        if image is None or image.array is None:
            return image

        array = image.array.copy()
        original_shape = array.shape
        
        # 处理不同的通道数
        if len(original_shape) == 2:
            # 已经是单色图像
            return image
        elif len(original_shape) == 3:
            height, width, channels = original_shape
            
            if channels == 1:
                # 已经是单色
                return image
            elif channels >= 3:
                # RGB转换为luminance，使用ITU-R BT.709权重
                luminance = self._rgb_to_luminance(array[..., :3])

                if channels == 3:
                    # RGB → 3个相同的monochrome通道（保持3通道用于管线兼容性）
                    result_array = np.stack([luminance, luminance, luminance], axis=2)
                elif channels == 4 and preserve_ir:
                    # RGBIR → 3个相同的Luminance通道 + IR
                    ir_channel = array[..., 3:4]  # 保持维度
                    mono_channels = np.stack([luminance, luminance, luminance], axis=2)
                    result_array = np.concatenate([mono_channels, ir_channel], axis=2)
                else:
                    # 其他情况，3个相同的luminance通道
                    result_array = np.stack([luminance, luminance, luminance], axis=2)
                
                return ImageData(
                    array=result_array,
                    file_path=image.file_path,
                    color_space="Monochrome"
                )
        
        return image
    
    def _rgb_to_luminance(self, rgb_array: np.ndarray) -> np.ndarray:
        """
        使用ITU-R BT.709权重将RGB转换为luminance
        
        Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
        """
        if len(rgb_array.shape) == 3 and rgb_array.shape[2] >= 3:
            # 使用ITU-R BT.709 luminance权重
            weights = np.array([0.2126, 0.7152, 0.0722])
            return np.dot(rgb_array[..., :3], weights)
        else:
            # 如果不是RGB格式，返回第一个通道
            return rgb_array[..., 0] if len(rgb_array.shape) == 3 else rgb_array
    
    def is_monochrome_image(self, image: ImageData) -> bool:
        """
        判断图像是否为单色图像
        """
        if image is None or image.array is None:
            return False
        
        shape = image.array.shape
        if len(shape) == 2:
            return True
        elif len(shape) == 3:
            channels = shape[2]
            if channels == 1:
                return True
            elif channels == 2:
                # 可能是 Luminance + IR
                return True
            elif channels >= 3:
                # 检查是否所有RGB通道都相等
                rgb = image.array[..., :3]
                # 安全检查：确保rgb确实有3个通道
                if rgb.shape[-1] >= 3:
                    return np.allclose(rgb[..., 0], rgb[..., 1]) and np.allclose(rgb[..., 1], rgb[..., 2])
                elif rgb.shape[-1] == 1:
                    # 如果实际只有1个通道，则认为是单色
                    return True
                else:
                    # 2个通道的情况，假设是单色
                    return True
        
        return False
    
    def get_monochrome_color_space_info(self) -> Dict[str, Any]:
        """
        获取monochrome色彩空间信息
        """
        return {
            "name": "Monochrome",
            "type": "monochrome",
            "gamma": 1.0,
            "white_point": np.array([0.3127, 0.3290]),  # D65白点
            "primaries": None,  # 单色没有primaries
            "description": "Monochrome (Black & White) color space"
        }
    
    def setup_monochrome_color_space(self):
        """
        设置monochrome色彩空间定义
        """
        self._color_spaces["Monochrome"] = self.get_monochrome_color_space_info()

    # --- 颜色空间属性更新 ---
    def update_color_space_gamma(self, space_name: str, gamma: float) -> None:
        """更新内存中的色彩空间gamma参数（不持久化）。"""
        space = self._color_spaces.get(space_name)
        if space is None:
            return
        try:
            space["gamma"] = float(gamma)
            # 失效相关转换缓存
            self._invalidate_convert_cache_for(space_name)
        except Exception:
            pass