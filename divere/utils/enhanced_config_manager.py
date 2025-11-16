#!/usr/bin/env python3
"""
增强配置管理器
支持用户配置目录和配置文件优先级管理
"""

import json
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Any
import platform
import time

# Import debug logger
try:
    from .debug_logger import debug, info, warning, error, log_path_search, log_file_operation
except ImportError:
    # Fallback if debug logger is not available
    def debug(msg, module=None): pass
    def info(msg, module=None): pass
    def warning(msg, module=None): pass
    def error(msg, module=None): pass
    def log_path_search(desc, paths, found=None, module=None): pass
    def log_file_operation(op, path, success=True, err=None, module=None): pass


class EnhancedConfigManager:
    """增强配置管理器"""
    
    def __init__(self):
        """初始化配置管理器"""
        self.app_name = "DiVERE"
        
        # 获取用户配置目录
        self.user_config_dir = self._get_user_config_dir()
        self.user_config_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建用户配置子目录
        self.user_colorspace_dir = self.user_config_dir / "config" / "colorspace"
        self.user_curves_dir = self.user_config_dir / "config" / "curves"
        self.user_matrices_dir = self.user_config_dir / "config" / "matrices"
        self.user_models_dir = self.user_config_dir / "models"
        self.user_logs_dir = self.user_config_dir / "logs"
        
        # 创建目录
        for dir_path in [self.user_colorspace_dir, self.user_curves_dir, 
                        self.user_matrices_dir, self.user_models_dir, self.user_logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 应用内置配置目录（统一入口）：优先二进制旁顶层 config，回退到包内 divere/config
        info("Resolving app config directory", "EnhancedConfigManager")
        try:
            from .app_paths import get_data_dir
            self.app_config_dir = get_data_dir("config")
            info(f"Using app_paths.get_data_dir() -> {self.app_config_dir}", "EnhancedConfigManager")
        except Exception as e:
            self.app_config_dir = Path("config")
            warning(f"app_paths.get_data_dir() failed: {e}, using fallback: {self.app_config_dir}", "EnhancedConfigManager")
        
        debug(f"app_config_dir exists: {self.app_config_dir.exists()}", "EnhancedConfigManager")
        
        # 应用设置文件（从软件config目录读取，与colorspace、curves等保持一致）
        self.app_settings_file = self.app_config_dir / "app_settings.json"
        debug(f"app_settings_file path: {self.app_settings_file}", "EnhancedConfigManager")
        debug(f"app_settings_file exists: {self.app_settings_file.exists()}", "EnhancedConfigManager")
        self.app_settings = self._load_app_settings()
    
    def _get_user_config_dir(self) -> Path:
        """获取用户配置目录"""
        system = platform.system()
        
        if system == "Darwin":  # macOS
            return Path.home() / "Library" / "Application Support" / self.app_name
        elif system == "Windows":
            return Path.home() / "AppData" / "Local" / self.app_name
        elif system == "Linux":
            return Path.home() / ".config" / self.app_name
        else:
            # 默认使用当前目录
            return Path.cwd() / "user_config"
    
    def _load_app_settings(self) -> Dict[str, Any]:
        """加载应用设置"""
        default_settings = {
            "directories": {
                "open_image": "",
                "save_image": "",
                "save_lut": "",
                "save_matrix": ""
            },
            "ui": {
                "window_size": [1200, 800],
                "window_position": [100, 100],
                "proxy_max_size": 2000,
                "use_process_isolation": "auto"
            },
            "defaults": {
                "input_color_space": "sRGB",
                "output_color_space_16bit": "DisplayP3",
                "output_color_space_8bit": "sRGB"
            },
            "config": {
                "show_user_config_dir": True,
                "auto_backup_config": True
            }
        }
        
        if self.app_settings_file.exists():
            try:
                log_file_operation("read", self.app_settings_file, success=True, module="EnhancedConfigManager")
                with open(self.app_settings_file, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)
                    return self._merge_configs(default_settings, loaded_settings)
            except Exception as e:
                log_file_operation("read", self.app_settings_file, success=False, err=str(e), module="EnhancedConfigManager")
                error(f"加载应用设置失败: {e}", "EnhancedConfigManager")
                return default_settings
        else:
            info(f"app_settings.json不存在，将创建默认配置: {self.app_settings_file}", "EnhancedConfigManager")
            # 保存默认设置
            self._save_app_settings(default_settings)
            return default_settings
    
    def _save_app_settings(self, settings: Dict[str, Any] = None):
        """保存应用设置"""
        if settings is None:
            settings = self.app_settings
        
        try:
            # 确保目录存在
            self.app_settings_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.app_settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=2, ensure_ascii=False)
            log_file_operation("write", self.app_settings_file, success=True, module="EnhancedConfigManager")
        except Exception as e:
            log_file_operation("write", self.app_settings_file, success=False, err=str(e), module="EnhancedConfigManager")
            error(f"保存应用设置失败: {e}", "EnhancedConfigManager")
    
    def _merge_configs(self, default: Dict[str, Any], loaded: Dict[str, Any]) -> Dict[str, Any]:
        """递归合并配置"""
        result = default.copy()
        
        for key, value in loaded.items():
            if key in result and isinstance(value, dict) and isinstance(result[key], dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get_config_files(self, config_type: str) -> List[Path]:
        """
        获取指定类型的配置文件列表（仅从项目配置目录）
        
        Args:
            config_type: 配置类型 ("colorspace", "curves", "matrices")
            
        Returns:
            配置文件路径列表，仅包含项目配置目录中的文件
        """
        info(f"Getting config files for type: {config_type}", "EnhancedConfigManager")
        
        app_dir = self.app_config_dir / config_type
        
        debug(f"App dir: {app_dir} (exists: {app_dir.exists()})", "EnhancedConfigManager")
        
        config_files = []
        search_paths = [str(app_dir)]
        
        # 仅从项目配置目录加载配置文件
        if app_dir.exists():
            app_files = list(app_dir.glob("*.json"))
            debug(f"Found {len(app_files)} app config files", "EnhancedConfigManager")
            for json_file in app_files:
                config_files.append(json_file)
                debug(f"Added app config: {json_file}", "EnhancedConfigManager")
        
        log_path_search(f"get_config_files({config_type})", search_paths, f"{len(config_files)} files found", "EnhancedConfigManager")
        
        return config_files
    
    def load_config_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """加载单个配置文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载配置文件失败 {file_path}: {e}")
            return None
    
    def save_user_config(self, config_type: str, name: str, data: Dict[str, Any]) -> bool:
        """
        保存用户配置文件
        
        Args:
            config_type: 配置类型 ("colorspace", "curves", "matrices")
            name: 配置名称（文件名，不含扩展名）
            data: 配置数据
            
        Returns:
            是否保存成功
        """
        # 所有配置都保存到项目config目录
        save_dir = self.app_config_dir / config_type
        
        file_path = save_dir / f"{name}.json"
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"配置已保存: {file_path}")
            return True
        except Exception as e:
            print(f"保存配置失败: {e}")
            return False
    
    def delete_user_config(self, config_type: str, name: str) -> bool:
        """删除配置文件（从项目配置目录）"""
        config_dir = self.app_config_dir / config_type
        file_path = config_dir / f"{name}.json"
        
        if file_path.exists():
            try:
                file_path.unlink()
                print(f"配置文件已删除: {file_path}")
                return True
            except Exception as e:
                print(f"删除配置文件失败: {e}")
                return False
        return False
    
    def copy_default_to_user(self, config_type: str, name: str) -> bool:
        """此方法已废弃，所有配置都在项目配置目录中"""
        # 由于所有配置都在项目目录中，不需要复制操作
        print("注意：所有配置都已统一在项目配置目录中，无需复制操作")
        return True
    
    def get_user_config_dir_path(self) -> str:
        """获取项目配置目录路径（用于显示给用户）"""
        return str(self.app_config_dir)
    
    def open_user_config_dir(self):
        """打开项目配置目录"""
        try:
            if platform.system() == "Darwin":  # macOS
                os.system(f"open '{self.app_config_dir}'")
            elif platform.system() == "Windows":
                os.system(f"explorer '{self.app_config_dir}'")
            elif platform.system() == "Linux":
                os.system(f"xdg-open '{self.app_config_dir}'")
        except Exception as e:
            print(f"打开配置目录失败: {e}")
    
    def backup_user_configs(self) -> bool:
        """备份项目配置"""
        backup_dir = self.app_config_dir.parent / "backup" / f"backup_{int(time.time())}"
        
        try:
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(self.app_config_dir, backup_dir / "config", dirs_exist_ok=True)
            print(f"项目配置已备份到: {backup_dir}")
            return True
        except Exception as e:
            print(f"备份项目配置失败: {e}")
            return False
    
    # 应用设置相关方法
    def get_directory(self, directory_type: str) -> str:
        """获取指定类型的目录路径"""
        directory = self.app_settings.get("directories", {}).get(directory_type, "")
        
        if directory and Path(directory).exists():
            return directory
        else:
            return ""
    
    def set_directory(self, directory_type: str, path: str):
        """设置指定类型的目录路径"""
        if "directories" not in self.app_settings:
            self.app_settings["directories"] = {}
        
        path_obj = Path(path)
        if path_obj.is_file():
            path = str(path_obj.parent)
        elif path_obj.is_dir():
            path = str(path_obj)
        else:
            parent = path_obj.parent
            if parent.exists() or parent.mkdir(parents=True, exist_ok=True):
                path = str(parent)
            else:
                return
        
        self.app_settings["directories"][directory_type] = path
        self._save_app_settings()
    
    def get_ui_setting(self, key: str, default=None):
        """获取UI设置"""
        return self.app_settings.get("ui", {}).get(key, default)
    
    def set_ui_setting(self, key: str, value):
        """设置UI设置"""
        if "ui" not in self.app_settings:
            self.app_settings["ui"] = {}
        self.app_settings["ui"][key] = value
        self._save_app_settings()
    
    def get_default_setting(self, key: str, default=None):
        """获取默认设置"""
        return self.app_settings.get("defaults", {}).get(key, default)
    
    def set_default_setting(self, key: str, value):
        """设置默认设置"""
        if "defaults" not in self.app_settings:
            self.app_settings["defaults"] = {}
        self.app_settings["defaults"][key] = value
        self._save_app_settings()


# 全局增强配置管理器实例
enhanced_config_manager = EnhancedConfigManager()
