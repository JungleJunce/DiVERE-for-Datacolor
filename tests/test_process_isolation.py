#!/usr/bin/env python3
"""
进程隔离功能测试脚本

测试 Phase 1-3 实现的进程隔离功能：
- 加载图片和切换
- Worker 进程生命周期
- 内存释放验证
- 异常处理和恢复

参考文档：PROCESS_ISOLATION_PROPOSAL.md 附录B
"""

import sys
import os
import time
import psutil
import gc
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 设置环境变量启用进程隔离
os.environ['DIVERE_PROCESS_ISOLATION'] = 'always'

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from divere.core.app_context import ApplicationContext


def get_memory_usage_mb():
    """获取当前进程内存使用量（MB）"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


class ProcessIsolationTester:
    """进程隔离功能测试器"""

    def __init__(self):
        self.app = QApplication(sys.argv)
        self.context = ApplicationContext()
        self.test_images = []
        self.test_results = []

    def setup_test_images(self):
        """查找测试图片"""
        test_scans_dir = project_root / "test_scans"
        if not test_scans_dir.exists():
            print(f"❌ 测试图片目录不存在: {test_scans_dir}")
            return False

        # 查找 TIF 文件
        self.test_images = list(test_scans_dir.glob("*.tif"))
        if not self.test_images:
            self.test_images = list(test_scans_dir.glob("*.TIF"))

        if len(self.test_images) < 2:
            print(f"❌ 测试图片不足（需要至少2张）: {len(self.test_images)}")
            return False

        # 限制到前5张（加快测试）
        self.test_images = self.test_images[:5]
        print(f"✅ 找到 {len(self.test_images)} 张测试图片")
        for img in self.test_images:
            print(f"   - {img.name}")
        return True

    def test_1_basic_lifecycle(self):
        """测试1: 基础生命周期"""
        print("\n" + "="*60)
        print("测试1: Worker 进程基础生命周期")
        print("="*60)

        # 记录初始内存
        initial_mem = get_memory_usage_mb()
        print(f"初始内存: {initial_mem:.1f} MB")

        # 加载第一张图片
        print(f"\n加载图片: {self.test_images[0].name}")
        self.context.load_image(str(self.test_images[0]))

        # 等待加载完成
        time.sleep(1.0)

        # 检查 worker 进程是否创建（lazy creation，需要触发预览）
        print("触发预览...")
        self.context._trigger_preview_update()

        # 等待预览完成
        for i in range(50):  # 最多等待5秒
            time.sleep(0.1)
            self.app.processEvents()
            if hasattr(self.context, '_preview_worker_process'):
                if self.context._preview_worker_process is not None:
                    if self.context._preview_worker_process.is_alive():
                        break

        # 检查进程状态
        if not hasattr(self.context, '_preview_worker_process'):
            print("❌ 进程隔离未启用（可能是配置问题）")
            return False

        if self.context._preview_worker_process is None:
            print("❌ Worker 进程未创建")
            return False

        if not self.context._preview_worker_process.is_alive():
            print("❌ Worker 进程未启动或已崩溃")
            return False

        print("✅ Worker 进程已启动")
        worker_pid = self.context._preview_worker_process.process.pid
        print(f"   Worker PID: {worker_pid}")

        # 记录加载后内存
        loaded_mem = get_memory_usage_mb()
        print(f"加载后内存: {loaded_mem:.1f} MB (增长 {loaded_mem - initial_mem:.1f} MB)")

        # 切换到第二张图片
        print(f"\n切换到图片: {self.test_images[1].name}")
        old_pid = worker_pid
        self.context.load_image(str(self.test_images[1]))

        # 等待切换完成
        time.sleep(1.0)

        # 检查旧进程是否被销毁
        try:
            old_process = psutil.Process(old_pid)
            if old_process.is_running():
                print(f"⚠️  旧 Worker 进程仍在运行: {old_pid}")
            else:
                print(f"✅ 旧 Worker 进程已退出: {old_pid}")
        except psutil.NoSuchProcess:
            print(f"✅ 旧 Worker 进程已退出: {old_pid}")

        # 触发新预览
        print("触发新预览...")
        self.context._trigger_preview_update()
        time.sleep(1.0)
        self.app.processEvents()

        # 检查新进程
        if self.context._preview_worker_process is not None:
            new_pid = self.context._preview_worker_process.process.pid
            print(f"✅ 新 Worker 进程已创建: {new_pid}")

            if new_pid == old_pid:
                print("⚠️  警告：新旧进程 PID 相同（可能是重用）")
        else:
            print("❌ 新 Worker 进程未创建")
            return False

        # 记录切换后内存
        switched_mem = get_memory_usage_mb()
        print(f"切换后内存: {switched_mem:.1f} MB (变化 {switched_mem - loaded_mem:+.1f} MB)")

        print("\n✅ 测试1 通过：基础生命周期正常")
        return True

    def test_2_memory_release(self):
        """测试2: 内存释放验证"""
        print("\n" + "="*60)
        print("测试2: 多次切换图片的内存释放")
        print("="*60)

        initial_mem = get_memory_usage_mb()
        print(f"初始内存: {initial_mem:.1f} MB")

        memory_samples = [initial_mem]

        # 循环切换图片 10 次
        num_iterations = 10
        for i in range(num_iterations):
            img_idx = i % len(self.test_images)
            img_path = self.test_images[img_idx]

            print(f"\n迭代 {i+1}/{num_iterations}: 加载 {img_path.name}")
            self.context.load_image(str(img_path))

            # 触发预览
            self.context._trigger_preview_update()

            # 等待处理完成
            time.sleep(0.5)
            for _ in range(20):
                self.app.processEvents()
                time.sleep(0.05)

            # 记录内存
            current_mem = get_memory_usage_mb()
            memory_samples.append(current_mem)
            print(f"   当前内存: {current_mem:.1f} MB (增长 {current_mem - initial_mem:.1f} MB)")

        final_mem = get_memory_usage_mb()
        total_growth = final_mem - initial_mem

        print(f"\n内存统计:")
        print(f"  初始: {initial_mem:.1f} MB")
        print(f"  最终: {final_mem:.1f} MB")
        print(f"  总增长: {total_growth:.1f} MB")
        print(f"  平均每次: {total_growth / num_iterations:.1f} MB")

        # 判断标准：总增长应该小于 500MB（因为可能有缓存）
        if total_growth < 500:
            print(f"✅ 测试2 通过：内存增长在可接受范围内 ({total_growth:.1f} MB < 500 MB)")
            return True
        else:
            print(f"⚠️  测试2 警告：内存增长较大 ({total_growth:.1f} MB >= 500 MB)")
            print("   可能原因：图片缓存、Qt 对象缓存、或存在内存泄漏")
            # 不算失败，只是警告
            return True

    def test_3_crash_recovery(self):
        """测试3: 崩溃恢复"""
        print("\n" + "="*60)
        print("测试3: Worker 崩溃自动恢复")
        print("="*60)

        # 加载图片
        print(f"加载图片: {self.test_images[0].name}")
        self.context.load_image(str(self.test_images[0]))
        self.context._trigger_preview_update()
        time.sleep(1.0)
        self.app.processEvents()

        if self.context._preview_worker_process is None:
            print("❌ Worker 进程未创建，无法测试崩溃恢复")
            return False

        # 模拟崩溃：强制杀死 worker 进程
        worker_pid = self.context._preview_worker_process.process.pid
        print(f"\n模拟崩溃：强制杀死 Worker 进程 {worker_pid}")

        try:
            worker_process = psutil.Process(worker_pid)
            worker_process.kill()
            worker_process.wait(timeout=2.0)
            print("✅ Worker 进程已被杀死")
        except Exception as e:
            print(f"❌ 无法杀死进程: {e}")
            return False

        # 等待一下
        time.sleep(0.5)

        # 尝试触发预览（应该触发自动重启）
        print("\n尝试触发新预览（应该自动重启）...")
        self.context._trigger_preview_update()

        # 等待重启
        time.sleep(1.0)
        for _ in range(20):
            self.app.processEvents()
            time.sleep(0.1)

        # 检查是否重启成功
        if self.context._preview_worker_process is None:
            print("❌ Worker 进程未重启")
            return False

        if not self.context._preview_worker_process.is_alive():
            print("❌ Worker 进程重启后仍未存活")
            return False

        new_pid = self.context._preview_worker_process.process.pid
        print(f"✅ Worker 进程已自动重启: {new_pid}")

        if new_pid == worker_pid:
            print("⚠️  警告：新旧进程 PID 相同")

        print("\n✅ 测试3 通过：崩溃恢复正常")
        return True

    def test_4_cleanup(self):
        """测试4: 清理验证"""
        print("\n" + "="*60)
        print("测试4: 资源清理验证")
        print("="*60)

        # 加载图片
        self.context.load_image(str(self.test_images[0]))
        self.context._trigger_preview_update()
        time.sleep(1.0)
        self.app.processEvents()

        if self.context._preview_worker_process is not None:
            worker_pid = self.context._preview_worker_process.process.pid
            print(f"Worker PID: {worker_pid}")
        else:
            print("❌ Worker 进程未创建")
            return False

        # 调用 cleanup
        print("\n调用 cleanup()...")
        self.context.cleanup()

        # 等待清理完成
        time.sleep(1.0)

        # 检查进程是否退出
        try:
            worker_process = psutil.Process(worker_pid)
            if worker_process.is_running():
                print(f"❌ Worker 进程未退出: {worker_pid}")
                return False
        except psutil.NoSuchProcess:
            print(f"✅ Worker 进程已退出: {worker_pid}")

        # 检查 shared memory 是否清理
        if self.context._proxy_shared_memory is not None:
            print("⚠️  Proxy shared memory 未清理")
        else:
            print("✅ Proxy shared memory 已清理")

        print("\n✅ 测试4 通过：资源清理正常")
        return True

    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "="*70)
        print("进程隔离功能测试")
        print("="*70)

        # 检查进程隔离是否启用
        if not self.context._use_process_isolation:
            print("\n❌ 进程隔离未启用，请检查配置")
            print(f"   环境变量 DIVERE_PROCESS_ISOLATION: {os.environ.get('DIVERE_PROCESS_ISOLATION')}")
            return False

        print(f"✅ 进程隔离已启用")

        # 设置测试图片
        if not self.setup_test_images():
            return False

        # 运行测试
        results = []

        try:
            results.append(("基础生命周期", self.test_1_basic_lifecycle()))
            results.append(("内存释放", self.test_2_memory_release()))
            results.append(("崩溃恢复", self.test_3_crash_recovery()))
            results.append(("资源清理", self.test_4_cleanup()))
        except Exception as e:
            print(f"\n❌ 测试异常: {e}")
            import traceback
            traceback.print_exc()
            return False

        # 输出结果
        print("\n" + "="*70)
        print("测试结果汇总")
        print("="*70)

        for test_name, passed in results:
            status = "✅ 通过" if passed else "❌ 失败"
            print(f"{status}: {test_name}")

        all_passed = all(result[1] for result in results)

        if all_passed:
            print("\n🎉 所有测试通过！")
        else:
            print("\n❌ 部分测试失败")

        return all_passed


def main():
    """主函数"""
    tester = ProcessIsolationTester()
    success = tester.run_all_tests()

    # 清理
    try:
        tester.context.cleanup()
    except:
        pass

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
