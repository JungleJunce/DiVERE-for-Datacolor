"""
多语言支持模块 (Internationalization - i18n)
提供文件驱动的多语言支持，使用 JSON 格式存储翻译。
"""

import json
import os
import locale
from pathlib import Path
from typing import Dict, Any, List, Optional
from divere.utils.path_manager import path_manager


class I18nManager:
    """
    多语言管理器

    特性：
    - JSON 文件存储翻译（文件驱动）
    - 嵌套键访问（点分隔符）
    - 格式化参数支持
    - 自动回退机制
    - 与 PathManager 集成
    """

    def __init__(self):
        self._current_lang = "zh_CN"  # 默认语言
        self._fallback_lang = "zh_CN"  # 回退语言
        self._translations: Dict[str, Dict[str, Any]] = {}  # {lang_code: translations_dict}
        self._available_languages: List[Dict[str, str]] = []  # [{code, name}]

        # 初始化：发现可用语言
        self._discover_languages()

        # 加载默认语言
        self._load_language(self._current_lang)

    def _discover_languages(self):
        """
        发现所有可用的语言文件

        在以下位置查找 *.json 文件：
        - 开发环境：divere/assets/i18n/*.json
        - 打包环境：通过 PathManager 查找
        """
        # 方法1：直接从assets/i18n目录查找
        try:
            # 从当前模块目录（divere/i18n/）向上，然后到assets/i18n
            i18n_dir = Path(__file__).parent.parent / "assets" / "i18n"
            lang_files = list(i18n_dir.glob("*.json")) if i18n_dir.exists() else []

            for lang_file in lang_files:
                lang_code = lang_file.stem  # zh_CN, en_US 等

                # 读取语言元信息
                try:
                    with open(lang_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        lang_name = data.get("meta", {}).get("language", lang_code)
                        self._available_languages.append({
                            "code": lang_code,
                            "name": lang_name,
                            "path": str(lang_file)
                        })
                except Exception as e:
                    print(f"[i18n] 无法读取语言文件 {lang_file}: {e}")

        except Exception as e:
            print(f"[i18n] 语言发现失败: {e}")

        # 方法2：通过 PathManager 查找（打包环境）
        # 为 i18n 添加路径类别
        if hasattr(path_manager, '_paths') and 'i18n' not in path_manager._paths:
            # 添加 i18n 路径到 PathManager
            try:
                # 获取当前模块的可能路径
                possible_paths = [
                    os.path.join(path_manager._get_project_root(), "divere", "assets", "i18n"),
                ]

                # 打包环境路径
                if path_manager._is_pyinstaller_bundle():
                    bundle_dir = path_manager._get_pyinstaller_root()
                    possible_paths.extend([
                        os.path.join(bundle_dir, "divere", "assets", "i18n"),
                        os.path.join(bundle_dir, "i18n"),
                        os.path.join(bundle_dir, "_internal", "divere", "assets", "i18n"),
                        os.path.join(bundle_dir, "_internal", "i18n")
                    ])

                # macOS app bundle 路径
                elif path_manager._is_macos_app_bundle():
                    executable_dir = path_manager._get_app_bundle_root()
                    bundle_contents = os.path.dirname(executable_dir)
                    possible_paths.extend([
                        os.path.join(executable_dir, "divere", "assets", "i18n"),
                        os.path.join(executable_dir, "assets", "i18n"),
                        os.path.join(bundle_contents, "Resources", "divere", "assets", "i18n"),
                        os.path.join(bundle_contents, "Resources", "assets", "i18n")
                    ])

                # 将路径添加到 PathManager
                for p in possible_paths:
                    if os.path.exists(p):
                        path_manager.add_path("i18n", p)

            except Exception as e:
                print(f"[i18n] 添加 i18n 路径到 PathManager 失败: {e}")

        # 如果没有发现任何语言，添加默认的中文和英文占位
        if not self._available_languages:
            self._available_languages = [
                {"code": "zh_CN", "name": "简体中文", "path": ""},
                {"code": "en_US", "name": "English", "path": ""}
            ]

    def _load_language(self, lang_code: str) -> bool:
        """
        加载指定语言的翻译文件

        Args:
            lang_code: 语言代码，如 zh_CN, en_US

        Returns:
            是否成功加载
        """
        # 如果已加载，直接返回
        if lang_code in self._translations:
            return True

        # 查找语言文件
        lang_file = None
        for lang_info in self._available_languages:
            if lang_info["code"] == lang_code:
                lang_file = lang_info.get("path")
                break

        if not lang_file or not os.path.exists(lang_file):
            # 尝试通过 PathManager 查找
            filename = f"{lang_code}.json"
            lang_file = path_manager.find_file(filename, "i18n")

            if not lang_file:
                # 尝试直接从assets/i18n目录查找
                try:
                    i18n_dir = Path(__file__).parent.parent / "assets" / "i18n"
                    lang_file = i18n_dir / filename
                    if not lang_file.exists():
                        lang_file = None
                except Exception:
                    lang_file = None

        if not lang_file:
            print(f"[i18n] 找不到语言文件: {lang_code}")
            return False

        # 加载 JSON 文件
        try:
            with open(lang_file, 'r', encoding='utf-8') as f:
                translations = json.load(f)
                self._translations[lang_code] = translations
                print(f"[i18n] 成功加载语言: {lang_code} ({lang_file})")
                return True

        except Exception as e:
            print(f"[i18n] 加载语言文件失败 {lang_code}: {e}")
            return False

    def _get_nested_value(self, data: Dict[str, Any], key: str) -> Optional[Any]:
        """
        从嵌套字典中获取值（支持点分隔符）

        Args:
            data: 嵌套字典
            key: 键，支持点分隔符，如 "main_window.menu.file"

        Returns:
            找到的值，如果不存在返回 None
        """
        keys = key.split('.')
        current = data

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None

        return current

    def tr(self, key: str, **kwargs) -> str:
        """
        翻译文本（支持嵌套键和格式化）

        Args:
            key: 翻译键，支持点分隔符，如 "main_window.menu.file"
            **kwargs: 格式化参数，如 tr("status.value", value=1.234)

        Returns:
            翻译后的文本，如果找不到则返回键名

        示例:
            tr("main_window.title")
            tr("status.rgb_value", channel="R", value=1.234)
        """
        # 从当前语言获取翻译
        translation = None
        if self._current_lang in self._translations:
            translation = self._get_nested_value(self._translations[self._current_lang], key)

        # 如果找不到，尝试回退语言
        if translation is None and self._fallback_lang != self._current_lang:
            if self._fallback_lang in self._translations:
                translation = self._get_nested_value(self._translations[self._fallback_lang], key)

        # 如果还是找不到，返回键名
        if translation is None:
            translation = key

        # 格式化（如果提供了参数）
        if kwargs:
            try:
                translation = translation.format(**kwargs)
            except (KeyError, ValueError) as e:
                print(f"[i18n] 格式化失败 '{key}': {e}")

        return translation

    def set_language(self, lang_code: str) -> bool:
        """
        切换语言

        Args:
            lang_code: 语言代码，如 zh_CN, en_US

        Returns:
            是否成功切换
        """
        # 加载语言（如果尚未加载）
        if lang_code not in self._translations:
            if not self._load_language(lang_code):
                print(f"[i18n] 无法切换到语言: {lang_code}")
                return False

        self._current_lang = lang_code
        print(f"[i18n] 已切换语言: {lang_code}")

        # 保存语言偏好到配置文件
        self._save_language_preference(lang_code)

        return True

    def get_current_language(self) -> str:
        """获取当前语言代码"""
        return self._current_lang

    def get_available_languages(self) -> List[Dict[str, str]]:
        """
        获取所有可用语言列表

        Returns:
            [{"code": "zh_CN", "name": "简体中文"}, ...]
        """
        return self._available_languages

    def detect_system_language(self) -> str:
        """
        检测系统语言

        Returns:
            语言代码，如 zh_CN, en_US
        """
        try:
            # 获取系统 locale
            system_locale, _ = locale.getdefaultlocale()

            if system_locale:
                # 标准化语言代码
                # zh_CN.UTF-8 -> zh_CN
                # en_US.UTF-8 -> en_US
                lang_code = system_locale.split('.')[0]

                # 检查是否支持该语言
                available_codes = [lang["code"] for lang in self._available_languages]
                if lang_code in available_codes:
                    return lang_code

                # 尝试匹配语言前缀（zh -> zh_CN, en -> en_US）
                lang_prefix = lang_code.split('_')[0]
                for code in available_codes:
                    if code.startswith(lang_prefix):
                        return code

        except Exception as e:
            print(f"[i18n] 系统语言检测失败: {e}")

        # 默认返回中文
        return "zh_CN"

    def _save_language_preference(self, lang_code: str):
        """
        保存语言偏好到配置文件

        Args:
            lang_code: 语言代码
        """
        try:
            # 尝试使用 enhanced_config_manager
            from divere.utils.enhanced_config_manager import enhanced_config_manager
            enhanced_config_manager.set_ui_setting("language", lang_code)

        except Exception as e:
            print(f"[i18n] 保存语言偏好失败: {e}")

    def load_language_preference(self) -> Optional[str]:
        """
        从配置文件加载语言偏好

        Returns:
            语言代码，如果没有保存则返回 None
        """
        try:
            from divere.utils.enhanced_config_manager import enhanced_config_manager
            lang_code = enhanced_config_manager.get_ui_setting("language")
            return lang_code

        except Exception as e:
            print(f"[i18n] 加载语言偏好失败: {e}")
            return None

    def initialize_language(self):
        """
        初始化语言设置（在应用启动时调用）

        优先级：
        1. 配置文件中保存的语言
        2. 系统语言（如果支持）
        3. 默认语言（zh_CN）
        """
        # 尝试加载保存的语言偏好
        saved_lang = self.load_language_preference()
        if saved_lang:
            available_codes = [lang["code"] for lang in self._available_languages]
            if saved_lang in available_codes:
                self.set_language(saved_lang)
                return

        # 尝试检测系统语言
        system_lang = self.detect_system_language()
        if system_lang != self._current_lang:
            self.set_language(system_lang)
            return

        # 使用默认语言（zh_CN）
        print(f"[i18n] 使用默认语言: {self._current_lang}")


# 全局单例
_i18n_manager = I18nManager()

# 便捷函数
def tr(key: str, **kwargs) -> str:
    """
    翻译文本（全局便捷函数）

    Args:
        key: 翻译键，支持点分隔符
        **kwargs: 格式化参数

    Returns:
        翻译后的文本

    示例:
        from divere.i18n import tr

        label = QLabel(tr("main_window.menu.file"))
        status = tr("status.rgb_value", channel="R", value=1.234)
    """
    return _i18n_manager.tr(key, **kwargs)


def set_language(lang_code: str) -> bool:
    """
    切换语言（全局便捷函数）

    Args:
        lang_code: 语言代码

    Returns:
        是否成功切换
    """
    return _i18n_manager.set_language(lang_code)


def get_current_language() -> str:
    """获取当前语言代码"""
    return _i18n_manager.get_current_language()


def get_available_languages() -> List[Dict[str, str]]:
    """获取所有可用语言列表"""
    return _i18n_manager.get_available_languages()


def initialize_language():
    """初始化语言设置（在应用启动时调用）"""
    _i18n_manager.initialize_language()


# 导出
__all__ = [
    'I18nManager',
    'tr',
    'set_language',
    'get_current_language',
    'get_available_languages',
    'initialize_language'
]
