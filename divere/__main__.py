"""
DiVERE 主应用程序入口
"""

import sys
import os
import platform
from pathlib import Path

# Windows控制台附加：检测是否从命令行运行
if platform.system() == 'Windows':
    import ctypes
    
    # 尝试附加到父进程的控制台（如果从cmd/PowerShell运行）
    # ATTACH_PARENT_PROCESS = -1
    kernel32 = ctypes.windll.kernel32
    if kernel32.AttachConsole(-1):
        # 成功附加到父控制台，重定向标准输出
        import io
        sys.stdout = io.TextIOWrapper(open('CONOUT$', 'wb'), encoding='utf-8')
        sys.stderr = io.TextIOWrapper(open('CONOUT$', 'wb'), encoding='utf-8')
        # 输出空行以确保从新行开始
        print()

# 将工作目录切换到可执行文件所在目录（适配 .app/Contents/MacOS 与独立二进制）
try:
    executable_dir = Path(sys.argv[0]).resolve().parent
    os.chdir(executable_dir)
except Exception:
    pass

# 配置数值计算库，防止多线程冲突和栈溢出
# 必须在导入numpy之前设置
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

# 取消OpenCV图像像素限制，允许加载超大图像
# 设置为0表示无限制（OpenCV默认限制约1.79亿像素）
# 注意：此环境变量在某些Windows系统上可能不生效，但image_manager.py会额外配置PIL限制
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = '0'

# 验证环境变量设置（可选的调试信息）
if '--debug' in sys.argv or '-v' in sys.argv:
    print(f"[DiVERE] OpenCV pixel limit env var: {os.environ.get('OPENCV_IO_MAX_IMAGE_PIXELS')}")
    print(f"[DiVERE] Platform: {platform.system()} {platform.release()}")

# 添加项目根目录到Python路径（开发环境下使用）
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
import multiprocessing

from divere.ui.main_window import MainWindow


def main():
    """主函数"""
    # Windows multiprocessing 支持（PyInstaller 打包必需）
    # 必须在创建任何 Process 之前调用，macOS/Linux 会自动忽略
    multiprocessing.freeze_support()

    # 创建Qt应用
    app = QApplication(sys.argv)
    app.setApplicationName("DiVERE")
    # 获取应用版本号
    try:
        from divere import __version__
        app.setApplicationVersion(__version__)
    except ImportError:
        app.setApplicationVersion("0.1.27")
    app.setOrganizationName("DiVERE Team")
    
    # 设置应用程序图标（如果有的话）
    # app.setWindowIcon(QIcon("icons/app_icon.png"))
    
    # 创建主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec())


if __name__ == "__main__":
    main() 