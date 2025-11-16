# 进程隔离方案：彻底解决 Heap 内存不归还问题

## 执行摘要

**方案核心**：为每张图片的预览处理创建独立的 worker 进程，在切换图片时销毁旧进程，从而 100% 释放 heap 内存回操作系统。

**可行性**：✅ **技术可行，架构合理，无后效性风险**

**关键收益**：
- ✅ **100% 内存释放** - 进程终止时操作系统强制回收所有 heap
- ✅ **自然的清理时机** - 图片切换是用户可感知的操作边界
- ✅ **无参数调优** - 无需设置"N 次预览后清理"等魔法数字
- ✅ **崩溃隔离** - worker 崩溃不影响主程序稳定性

**实施成本**：8-10 个工作日

**风险等级**：中等（有完备的回退方案，可渐进式实施）

---

## 1. 问题根源

### 1.1 macOS Heap 内存不归还

参考 `memory_analysis_report.md`，核心问题：

```
Preview 1: 分配 400MB → heap 扩展到 600MB
Preview 2: 分配 420MB → heap 扩展到 800MB (新的峰值)
Preview 3: 分配 380MB → heap 保持 800MB (复用空闲块)
...
Preview N: 分配 500MB → heap 扩展到 1.2GB (又一个峰值)

结果：heap 呈阶梯式单向增长，即使 Python 对象已释放
```

**根本原因**：
- macOS 的 `malloc` (system allocator) 倾向于保留 heap 作为 free blocks
- Python 的 `del` 和 `gc.collect()` 只释放 Python 对象，不归还 heap
- `malloc_zone_pressure_relief()` 效果有限（10-30%）
- jemalloc 更好，但仍无法**保证** 100% 归还

**唯一的 100% 解决方案**：销毁进程，让操作系统回收所有资源

---

## 2. 为什么切换图片是最佳时机

### 2.1 生命周期分析

```
加载图片 A (load_image)
├─> 生成 proxy (2000x2000, ~48MB)
├─> 预览 1 (density_gamma=2.4)   → 分配 ~400MB
├─> 预览 2 (density_gamma=2.5)   → 分配 ~420MB
├─> 预览 3 (rgb_gains=[0.1,0,0]) → 分配 ~380MB
├─> 预览 4-20 ...                  → heap 累积增长到 1.2GB
└─> [单张图片的处理生命周期结束]

切换到图片 B (navigate_to_index / load_image)  ← 自然边界！
├─> 销毁 worker 进程 A            → heap 100% 归还给 OS
├─> 创建新 worker 进程 B          → 从干净状态开始
└─> 预览 1-20 ...                  → heap 从零开始增长
```

### 2.2 为什么这是最优边界

1. **用户可感知的操作** - 切换图片时有短暂延迟（200-500ms）是可以接受的
2. **无需频繁重启** - 不像"每 50 次预览重启 worker"那样影响用户体验
3. **逻辑清晰** - 一张图对应一个进程，状态隔离，易于理解和调试
4. **代码已有清理逻辑** - `load_image()` 已经执行缓存清理，是插入进程销毁的理想位置

---

## 3. 当前架构分析

### 3.1 现有 Preview Worker (线程模式)

**文件**：`divere/core/app_context.py`

```python
class _PreviewWorker(QRunnable):
    """在主进程的线程池中运行"""
    def __init__(self, image, params, the_enlarger, color_space_manager, ...):
        self.image = image          # 共享主进程的对象
        self.the_enlarger = the_enlarger
        ...

    def run(self):
        result = self.the_enlarger.apply_full_pipeline(self.image, self.params, ...)
        result = self.color_space_manager.convert_to_display_space(result, "DisplayP3")
        self.signals.result.emit(result)  # Qt Signal

class ApplicationContext(QObject):
    def _trigger_preview_update(self):
        worker = _PreviewWorker(...)
        self.thread_pool.start(worker)  # 在主进程的线程中运行
```

**问题**：
- ✅ 实现简单，Qt 集成方便
- ❌ **所有内存分配在主进程** - heap 不归还
- ⚠️ 共享对象可能导致竞态条件（虽然目前用 view() 缓解）

### 3.2 图片切换流程

**调用链**：
```
UI 操作 (按键/鼠标)
  └─> FolderNavigator.navigate_to_index(i)
      └─> FolderNavigator.file_changed.emit(file_path)  # Qt Signal
          └─> ApplicationContext.load_image(file_path)
              ├─> _clear_all_caches()  或 _clear_current_image_data()
              ├─> preview_clear_requested.emit()
              ├─> self._current_image = None
              ├─> self._current_proxy = None
              ├─> 加载新图片
              └─> _trigger_preview_update()
```

**关键点**：`load_image()` 已经在清理旧数据，是插入进程销毁逻辑的**完美位置**。

---

## 4. 进程隔离方案设计

### 4.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│ ApplicationContext (主进程 / Qt GUI)                         │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ 图片 A 的生命周期                                        │ │
│ │ ├─ ImageData (原图)                                     │ │
│ │ ├─ Proxy (2000x2000, 存储在 shared_memory)             │ │
│ │ └─ PreviewWorkerProcess ────────────────────┐          │ │
│ │    ├─ multiprocessing.Process (独立进程)    │          │ │
│ │    ├─ queue_request  (主→worker: params)    │          │ │
│ │    ├─ queue_result   (worker→主: result)    │          │ │
│ │    └─ shared_memory  (proxy array)          │          │ │
│ └─────────────────────────────────────────────┼──────────┘ │
│                                                │            │
│ 切换到图片 B (load_image)                      │            │
│   ├─> worker_process.shutdown()  ──────────→ X (进程终止) │
│   │   ├─> process.join(timeout=2.0)           ↓            │
│   │   └─> 清理 queue 和 shared_memory    heap 100% 归还   │
│   │                                                         │
│   └─> 创建新 worker 进程 B (lazy, 首次预览时)              │
│       └─ PreviewWorkerProcess (图片 B)                      │
│          └─ 独立进程，干净的 heap 空间                      │
└─────────────────────────────────────────────────────────────┘

Worker 进程 (独立地址空间)
┌───────────────────────────────────────┐
│ _worker_main_loop()                    │
│ ├─ 初始化 TheEnlarger                  │
│ ├─ 初始化 ColorSpaceManager            │
│ ├─ 从 shared_memory attach proxy       │
│ └─ while True:                         │
│    ├─ params = queue_request.get()     │
│    ├─ if params is None: break (停止) │
│    ├─ result = process_pipeline(...)   │
│    └─> queue_result.put(result_info)   │
└───────────────────────────────────────┘
```

### 4.2 数据传递策略

| 数据类型 | 大小 | 传递方式 | 原因 |
|---------|------|---------|------|
| Proxy Image (输入) | ~48MB | `shared_memory` (一次性传递) | 避免拷贝，进程创建时传递 |
| ColorGradingParams | ~1KB | `pickle` via `Queue` | 小对象，每次预览传递 |
| Result Image | ~48MB | `shared_memory` (每次返回) | 避免拷贝，返回时创建新 shm |
| 配置/元数据 | <1KB | `pickle` via `Queue` | 简单，随参数传递 |

**关键优化**：
- Proxy 只在进程创建时通过 shared_memory 传递一次
- 每次预览只传递小的参数对象（Queue）
- 结果通过新的 shared_memory 返回（主进程拷贝后立即释放 shm）

### 4.3 生命周期管理

#### Phase 1: 加载图片时销毁旧进程

```python
# app_context.py

def load_image(self, file_path: str):
    try:
        self._loading_image = True

        # ============ 新增：销毁旧 worker 进程 ============
        if self._preview_worker_process is not None:
            self._preview_worker_process.shutdown()  # 发送停止信号
            self._preview_worker_process.join(timeout=2.0)  # 等待退出
            if self._preview_worker_process.is_alive():
                self._preview_worker_process.terminate()  # 强制终止
            self._preview_worker_process = None
            # ✅ 此时旧进程的 heap 100% 归还给 OS

        # 清理旧 shared memory（防止泄漏）
        if self._proxy_shared_memory is not None:
            self._proxy_shared_memory.close()
            self._proxy_shared_memory.unlink()
            self._proxy_shared_memory = None
        # ================================================

        # 原有的清理逻辑
        self._clear_all_caches()
        self.preview_clear_requested.emit()

        # 释放旧图像
        if self._current_image is not None:
            self._current_image.array = None
            self._current_image = None

        if self._current_proxy is not None:
            self._current_proxy.array = None
            self._current_proxy = None

        # 加载新图片
        self._current_image = self.image_manager.load_image(file_path)
        # ... 其他初始化逻辑 ...

        # 注意：不立即创建 worker 进程，等首次预览时再创建 (Lazy)

    finally:
        self._loading_image = False
```

#### Phase 2: 首次预览时 Lazy 创建进程

```python
def _trigger_preview_update(self):
    if self._loading_image:
        return

    # ============ 新增：Lazy 创建 worker 进程 ============
    if self._use_process_isolation:
        # Lazy 创建：首次预览时才创建进程
        if self._preview_worker_process is None:
            self._create_preview_worker_process()

        # 发送预览请求（非阻塞）
        self._preview_worker_process.request_preview(self._current_params)

        # 启动结果轮询定时器
        if not self._result_poll_timer.isActive():
            self._result_poll_timer.start(16)  # ~60 FPS
    else:
        # 回退到线程模式（兼容性）
        self._trigger_preview_with_thread()  # 原有实现

def _create_preview_worker_process(self):
    """创建 worker 进程并传递 proxy"""
    # 1. 生成 proxy
    proxy = self.image_manager.generate_proxy(self._current_image)

    # 2. 创建 shared memory 并写入 proxy
    shm = shared_memory.SharedMemory(create=True, size=proxy.array.nbytes)
    shm_array = np.ndarray(proxy.array.shape, dtype=proxy.array.dtype,
                           buffer=shm.buf)
    np.copyto(shm_array, proxy.array)

    # 3. 创建 worker 进程
    self._preview_worker_process = PreviewWorkerProcess(
        proxy_shm_name=shm.name,
        proxy_shape=proxy.array.shape,
        proxy_dtype=str(proxy.array.dtype),
        # 传递配置（序列化）
        color_space_config=self.color_space_manager.get_config_dict(),
        pipeline_config=self.the_enlarger.get_config_dict(),
    )
    self._preview_worker_process.start()
    self._proxy_shared_memory = shm
```

#### Phase 3: 轮询结果（替代 Qt Signal）

```python
def __init__(self):
    # ... 其他初始化 ...

    # 结果轮询定时器（Qt Signal 不能跨进程）
    self._result_poll_timer = QTimer()
    self._result_poll_timer.timeout.connect(self._poll_preview_result)
    # 不自动启动，只在有 worker 时启动

def _poll_preview_result(self):
    """定期轮询结果队列（~60 FPS）"""
    if self._preview_worker_process is None:
        self._result_poll_timer.stop()
        return

    result = self._preview_worker_process.try_get_result()  # 非阻塞

    if result is not None:
        if isinstance(result, Exception):
            self.status_message_changed.emit(f"预览失败: {result}")
        else:
            # 正常结果，发射信号更新 UI
            self.preview_updated.emit(result)
```

---

## 5. 核心实现：PreviewWorkerProcess

### 5.1 类接口设计

**新文件**：`divere/core/preview_worker_process.py`

```python
import multiprocessing
from multiprocessing import shared_memory, Queue, Process
import queue
import numpy as np
from typing import Optional
import time
import traceback

class PreviewWorkerProcess:
    """管理一个独立的预览处理进程"""

    def __init__(self, proxy_shm_name: str, proxy_shape: tuple, proxy_dtype: str,
                 color_space_config: dict, pipeline_config: dict):
        """初始化但不启动进程

        Args:
            proxy_shm_name: shared memory 名称（proxy 已写入）
            proxy_shape: proxy 数组形状
            proxy_dtype: proxy 数据类型
            color_space_config: 色彩空间配置（序列化）
            pipeline_config: 管线配置（序列化）
        """
        self.proxy_shm_name = proxy_shm_name
        self.proxy_shape = proxy_shape
        self.proxy_dtype = proxy_dtype
        self.color_space_config = color_space_config
        self.pipeline_config = pipeline_config

        # IPC 组件
        self.queue_request = Queue(maxsize=2)  # 参数队列（限制大小避免积压）
        self.queue_result = Queue(maxsize=2)   # 结果队列

        self.process: Optional[Process] = None

    def start(self):
        """启动 worker 进程"""
        self.process = Process(
            target=_worker_main_loop,
            args=(
                self.queue_request,
                self.queue_result,
                self.proxy_shm_name,
                self.proxy_shape,
                self.proxy_dtype,
                self.color_space_config,
                self.pipeline_config,
            )
        )
        self.process.start()

    def request_preview(self, params: ColorGradingParams):
        """请求预览（非阻塞）

        只保留最新请求，丢弃旧的未处理请求（预览去重）
        """
        # 清空旧请求
        while not self.queue_request.empty():
            try:
                self.queue_request.get_nowait()
            except queue.Empty:
                break

        # 发送新请求
        self.queue_request.put({
            'action': 'preview',
            'params': params.to_dict(),
            'timestamp': time.time()
        })

    def try_get_result(self) -> Optional[ImageData]:
        """尝试获取结果（非阻塞）

        Returns:
            ImageData: 预览结果
            Exception: 处理出错
            None: 暂无结果
        """
        try:
            result_info = self.queue_result.get_nowait()
        except queue.Empty:
            return None

        if result_info['status'] == 'error':
            return Exception(result_info['message'])

        # 从 shared memory 读取结果
        shm = shared_memory.SharedMemory(name=result_info['shm_name'])
        result_array = np.ndarray(
            tuple(result_info['shape']),
            dtype=result_info['dtype'],
            buffer=shm.buf
        )

        # 拷贝数据到主进程（重要！）
        result_image = ImageData(
            array=result_array.copy(),  # 拷贝到主进程内存
            metadata=result_info['metadata']
        )

        # 立即清理 shared memory
        shm.close()
        shm.unlink()

        return result_image

    def shutdown(self):
        """优雅停止进程"""
        # 发送停止信号
        try:
            self.queue_request.put(None, timeout=0.5)
        except:
            pass

        # 等待退出
        if self.process is not None:
            self.process.join(timeout=2.0)

            # 如果超时，强制终止
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1.0)

            self.process = None

        # 清理残留资源
        self._cleanup_queues()

    def _cleanup_queues(self):
        """清理队列和残留的 shared memory"""
        # 清空请求队列
        while not self.queue_request.empty():
            try:
                self.queue_request.get_nowait()
            except:
                break

        # 清空结果队列（并释放 shared memory）
        while not self.queue_result.empty():
            try:
                result = self.queue_result.get_nowait()
                if isinstance(result, dict) and 'shm_name' in result:
                    try:
                        shm = shared_memory.SharedMemory(name=result['shm_name'])
                        shm.close()
                        shm.unlink()
                    except:
                        pass
            except:
                break
```

### 5.2 Worker 主循环

```python
def _worker_main_loop(queue_request, queue_result, proxy_shm_name,
                      proxy_shape, proxy_dtype, color_space_config, pipeline_config):
    """Worker 进程的主循环（在独立进程中运行）

    这个函数运行在独立进程中，不能访问主进程的对象。
    所有依赖对象都需要在这里重新初始化。
    """
    try:
        # ============ Step 1: 初始化（在 worker 进程中） ============
        from divere.core.the_enlarger import TheEnlarger
        from divere.core.color_space import ColorSpaceManager
        from divere.core.data_types import ImageData, ColorGradingParams

        # 重新创建对象（不能共享主进程的对象）
        the_enlarger = TheEnlarger.from_config_dict(pipeline_config)
        color_space_manager = ColorSpaceManager.from_config_dict(color_space_config)

        # ============ Step 2: 加载 proxy 从 shared memory ============
        shm = shared_memory.SharedMemory(name=proxy_shm_name)
        proxy_array = np.ndarray(proxy_shape, dtype=proxy_dtype, buffer=shm.buf)

        # 拷贝到 worker 进程内存（避免依赖主进程的 shm）
        proxy_image = ImageData(
            array=proxy_array.copy(),
            metadata={},
        )

        shm.close()  # 不 unlink，主进程负责清理

        # ============ Step 3: 主循环：处理预览请求 ============
        while True:
            # 3.1 接收请求
            request = queue_request.get()  # 阻塞等待

            # 3.2 停止信号
            if request is None:
                break

            # 3.3 解析参数
            action = request.get('action')
            if action != 'preview':
                continue

            params = ColorGradingParams.from_dict(request['params'])

            # 3.4 处理预览
            try:
                result_image = the_enlarger.apply_full_pipeline(
                    proxy_image,
                    params,
                    workspace=None,  # 注意：可选使用 workspace buffer pool
                )
                result_image = color_space_manager.convert_to_display_space(
                    result_image, "DisplayP3"
                )

                # 3.5 通过 shared memory 返回结果
                result_shm = shared_memory.SharedMemory(
                    create=True,
                    size=result_image.array.nbytes
                )
                result_shm_array = np.ndarray(
                    result_image.array.shape,
                    result_image.array.dtype,
                    buffer=result_shm.buf
                )
                np.copyto(result_shm_array, result_image.array)

                # 3.6 发送结果元数据
                queue_result.put({
                    'status': 'success',
                    'shm_name': result_shm.name,
                    'shape': list(result_image.array.shape),
                    'dtype': str(result_image.array.dtype),
                    'metadata': result_image.metadata
                })

            except Exception as e:
                # 发送错误
                queue_result.put({
                    'status': 'error',
                    'message': str(e),
                    'traceback': traceback.format_exc()
                })

    except Exception as e:
        # Worker 初始化失败
        queue_result.put({
            'status': 'error',
            'message': f"Worker initialization failed: {e}",
            'traceback': traceback.format_exc()
        })
```

---

## 6. 无后效性保证

### 6.1 配置开关

```python
# divere/config/app_settings.json 或环境变量
{
  "preview": {
    "use_process_isolation": "auto"  // "auto" | "always" | "never"
  }
}

# app_context.py
class ApplicationContext:
    def __init__(self):
        # ... 其他初始化 ...

        # 决定是否使用进程隔离
        self._use_process_isolation = self._should_use_process_isolation()

        if self._use_process_isolation:
            self._preview_worker_process = None
            self._proxy_shared_memory = None
            self._result_poll_timer = QTimer()
            self._result_poll_timer.timeout.connect(self._poll_preview_result)
        # else: 使用现有的线程模式

    def _should_use_process_isolation(self) -> bool:
        """根据配置和平台决定是否使用进程隔离"""
        import platform
        from divere.utils.enhanced_config_manager import enhanced_config_manager

        config = enhanced_config_manager.get_ui_setting(
            "use_process_isolation", "auto"
        )

        if config == "never":
            return False
        elif config == "always":
            return True
        else:  # "auto"
            # macOS/Linux: 默认启用（内存问题严重）
            # Windows: 默认禁用（multiprocessing 复杂）
            return platform.system() in ['Darwin', 'Linux']
```

### 6.2 自动回退机制

```python
def _create_preview_worker_process(self):
    """创建 worker 进程，失败时自动回退到线程模式"""
    try:
        # 尝试创建进程
        proxy = self.image_manager.generate_proxy(self._current_image)
        shm = shared_memory.SharedMemory(create=True, size=proxy.array.nbytes)
        shm_array = np.ndarray(proxy.array.shape, dtype=proxy.array.dtype,
                               buffer=shm.buf)
        np.copyto(shm_array, proxy.array)

        self._preview_worker_process = PreviewWorkerProcess(
            proxy_shm_name=shm.name,
            proxy_shape=proxy.array.shape,
            proxy_dtype=str(proxy.array.dtype),
            color_space_config=self.color_space_manager.get_config_dict(),
            pipeline_config=self.the_enlarger.get_config_dict(),
        )
        self._preview_worker_process.start()

        # 验证进程启动成功
        time.sleep(0.1)
        if not self._preview_worker_process.process.is_alive():
            raise RuntimeError("Worker process failed to start")

        self._proxy_shared_memory = shm

    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Process isolation failed, falling back to thread mode: {e}")

        # 清理失败的资源
        if hasattr(self, '_proxy_shared_memory') and self._proxy_shared_memory:
            try:
                self._proxy_shared_memory.close()
                self._proxy_shared_memory.unlink()
            except:
                pass
            self._proxy_shared_memory = None

        # 自动回退到线程模式
        self._use_process_isolation = False
        self._preview_worker_process = None

        # 提示用户
        self.status_message_changed.emit(
            "进程隔离启动失败，已回退到线程模式（内存优化受限）"
        )

        # 使用线程模式重新触发预览
        self._trigger_preview_with_thread()
```

### 6.3 代码隔离（无侵入）

**文件结构**：
```
divere/core/
├── app_context.py          # 主要修改，添加 if/else 分支
├── preview_worker_process.py  # 新文件，完全独立
└── ...
```

**分支选择**：
```python
# app_context.py

def _trigger_preview_update(self):
    if self._loading_image:
        return

    if self._use_process_isolation:
        # ============ 新代码：进程模式 ============
        self._trigger_preview_with_process()
    else:
        # ============ 旧代码：线程模式 ============
        self._trigger_preview_with_thread()

def _trigger_preview_with_thread(self):
    """旧的线程模式实现（保持不变）"""
    worker = _PreviewWorker(...)
    self.thread_pool.start(worker)

def _trigger_preview_with_process(self):
    """新的进程模式实现"""
    if self._preview_worker_process is None:
        self._create_preview_worker_process()
    self._preview_worker_process.request_preview(self._current_params)
    if not self._result_poll_timer.isActive():
        self._result_poll_timer.start(16)
```

**回滚策略**：
1. 设置配置 `use_process_isolation = "never"`
2. 或者删除 `preview_worker_process.py` 文件
3. 旧代码完全不受影响，可以安全回滚

---

## 7. 与现有优化的关系

### 7.1 多层次内存优化策略

```
┌─────────────────────────────────────────────────────────┐
│ 层次化内存优化                                           │
├─────────────────────────────────────────────────────────┤
│ Layer 1: PreviewWorkspace 缓冲池                        │
│   └─> 减少 50-60% 临时分配 (已实现)                    │
│   └─> 作用：同一张图片内的多次预览                      │
│                                                          │
│ Layer 2: jemalloc 替代 system allocator                 │
│   └─> 减少 30-50% heap 增长 (已实现)                   │
│   └─> 作用：更好的 heap 管理和部分归还                  │
│                                                          │
│ Layer 3: malloc_zone_pressure_relief                    │
│   └─> 周期性释放 10-30% (已实现)                       │
│   └─> 作用：触发系统内存压力缓解                        │
│                                                          │
│ Layer 4: 进程隔离（切换图片时）                         │
│   └─> 100% 释放 heap (本方案)                          │
│   └─> 作用：彻底重置 worker 内存状态                    │
└─────────────────────────────────────────────────────────┘
```

### 7.2 组合效果

**同一张图片内（预览 1-20 次）**：
- Layer 1-3 优化生效
- 内存稳定在 200-400MB（proxy + 临时分配）
- heap 增长较小（workspace buffer pool 减少分配）

**切换图片时**：
- Layer 4 生效
- 销毁 worker 进程 → heap 归零
- 创建新进程 → 从干净状态开始

**长时间使用（100+ 张图片）**：
- 每张图片内存稳定在 200-400MB
- 切换时归零
- **总体内存占用不会无限增长**

### 7.3 性能影响

| 操作 | 线程模式 | 进程模式 | 差异 |
|------|---------|---------|-----|
| 首次预览（冷启动） | ~100ms | ~300-500ms | +200-400ms (进程创建) |
| 后续预览（同一图片） | ~50-100ms | ~50-100ms | 无差异 |
| 切换图片 | ~200ms | ~400-600ms | +200-400ms (进程销毁+创建) |
| 内存占用（100 张图） | 10-20GB | 4-6GB | **-50% ~ -70%** |

**结论**：
- 延迟增加可接受（图片切换时用户已有心理预期）
- 内存占用显著降低（长时间使用收益巨大）

---

## 8. 实施计划

### 8.1 Phase 1: 基础架构（3-4 天）

**任务**：
- [ ] 创建 `preview_worker_process.py`
  - [ ] `PreviewWorkerProcess` 类
  - [ ] `_worker_main_loop()` 函数
  - [ ] Shared memory 管理
- [ ] 修改 `app_context.py`
  - [ ] 添加配置开关逻辑
  - [ ] 添加 `_create_preview_worker_process()`
  - [ ] 修改 `load_image()` - 销毁旧进程
  - [ ] 修改 `_trigger_preview_update()` - 分支选择
  - [ ] 添加 `_poll_preview_result()`
- [ ] 数据类型支持序列化
  - [ ] `ColorGradingParams.to_dict()` / `from_dict()`
  - [ ] `TheEnlarger.get_config_dict()` / `from_config_dict()`
  - [ ] `ColorSpaceManager.get_config_dict()` / `from_config_dict()`

**验收标准**：
- ✅ 能够启动/停止 worker 进程
- ✅ 能够通过 shared memory 传递 proxy
- ✅ 能够通过 queue 传递参数和结果
- ✅ 简单预览流程可以工作

### 8.2 Phase 2: 稳定性和优化（2-3 天）

**任务**：
- [ ] 异常处理
  - [ ] Worker 崩溃检测和自动重启
  - [ ] 超时处理（worker 无响应）
  - [ ] Shared memory 泄漏检测和清理
  - [ ] 主进程退出时的 cleanup
- [ ] 性能优化
  - [ ] 预览请求去重（只保留最新）
  - [ ] 队列大小调优
  - [ ] 减少不必要的拷贝
- [ ] 自动回退机制
  - [ ] 进程启动失败时回退到线程模式
  - [ ] 用户提示

**验收标准**：
- ✅ Worker 崩溃后能自动恢复
- ✅ 无 shared memory 泄漏
- ✅ 快速连续预览不会卡顿
- ✅ 启动失败时能优雅降级

### 8.3 Phase 3: 集成测试（2-3 天）

**任务**：
- [ ] 全流程测试
  - [ ] 加载 → 预览 → 切换 → 预览 → 导出
  - [ ] 快速连续切换图片（10 次）
  - [ ] 长时间运行（100+ 张图片）
  - [ ] Activity Monitor 监控内存占用
- [ ] 边缘情况测试
  - [ ] 加载图片但不预览
  - [ ] 切换图片时正在预览
  - [ ] 色卡优化流程（多次快速预览）
  - [ ] 导出时切换图片
- [ ] 平台兼容性
  - [ ] macOS 测试
  - [ ] Linux 测试（可选）
  - [ ] Windows 测试（可选）

**验收标准**：
- ✅ 所有测试用例通过
- ✅ 内存占用符合预期（100 张图 < 6GB）
- ✅ 无崩溃、无卡顿
- ✅ 用户体验可接受

### 8.4 总时间估算

| Phase | 估算时间 | 风险缓冲 | 总计 |
|-------|---------|---------|------|
| Phase 1: 基础架构 | 3-4 天 | +1 天 | 4-5 天 |
| Phase 2: 稳定性优化 | 2-3 天 | +1 天 | 3-4 天 |
| Phase 3: 集成测试 | 2-3 天 | +0.5 天 | 2.5-3.5 天 |
| **总计** | **7-10 天** | **+2.5 天** | **9.5-12.5 天** |

**保守估算**：**10 个工作日**

---

## 9. 风险分析与缓解

### 9.1 技术风险

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|---------|
| **进程启动延迟** | 首次预览慢 200-500ms | 高 | Lazy 创建 + loading 状态提示 |
| **Shared memory 泄漏** | 内存占用累积 | 中 | 严格 cleanup + atexit handler + 泄漏检测 |
| **Worker 崩溃** | 预览失败，用户体验差 | 低 | 自动重启 + 错误提示 + 回退 |
| **序列化失败** | 无法传递参数 | 低 | 全面测试 + 回退到线程模式 |
| **IPC 开销** | 预览变慢 | 低 | Shared memory（不拷贝大数组）|
| **Qt 兼容性问题** | UI 无响应 | 低 | 轮询模式 + 充分测试 |

### 9.2 用户体验风险

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| **切换图片变慢** | 延迟 200-500ms | 显示 loading 动画，用户可接受 |
| **快速切换卡顿** | 频繁创建/销毁进程 | 预览请求去重 + 异步处理 |
| **色卡优化变慢** | 多次预览延迟累积 | 色卡优化时可选禁用进程隔离 |

### 9.3 平台兼容性

| 平台 | 风险 | 缓解措施 |
|------|------|---------|
| **macOS** | 低 | 主要目标平台，充分测试 |
| **Linux** | 低 | multiprocessing 兼容，测试验证 |
| **Windows** | 中 | 需要 `if __name__ == '__main__'` 保护，默认禁用 |

---

## 10. 决策建议

### 10.1 是否实施？

**推荐**：✅ **实施**

**理由**：
1. **效果确定** - 100% 解决 heap 不归还问题（操作系统保证）
2. **风险可控** - 有完备回退方案，可配置开关
3. **收益显著** - 长时间使用内存占用减少 50-70%
4. **互补优化** - 与 jemalloc/workspace 互补，而非替代
5. **无后效性** - 可以安全启用/禁用，不影响现有功能

### 10.2 实施策略

**建议顺序**：

1. **立即**：测试 jemalloc + workspace 效果（已实现）
   - 监控实际使用场景的内存占用
   - 收集用户反馈

2. **如果 jemalloc 足够**：暂缓进程隔离
   - 如果内存稳定在可接受范围（<6GB），进程隔离优先级降低
   - 节省 10 天开发时间

3. **如果内存仍然问题**：实施进程隔离
   - 分阶段实施：Phase 1 → 测试 → Phase 2 → 测试 → Phase 3
   - 每个 Phase 后评估效果

### 10.3 决策树

```
测试 jemalloc + workspace 效果
│
├─ 内存稳定 (<6GB，100 张图)
│  └─> ✅ 暂不实施进程隔离，继续监控
│
└─ 内存仍增长 (>8GB，100 张图)
   │
   ├─ 用户可接受 200-500ms 切换延迟？
   │  ├─ 是 → ✅ 实施进程隔离
   │  └─ 否 → ⚠️ 考虑进程池优化版本（复用进程）
   │
   └─ 无法接受任何延迟？
      └─> ❌ 不实施，建议用户增加 RAM
```

---

## 11. 总结

### 11.1 方案优势

1. **彻底解决 heap 不归还** - 操作系统保证 100% 回收
2. **自然的清理时机** - 图片切换是用户可感知的操作边界
3. **崩溃隔离** - worker 崩溃不影响主程序
4. **无参数调优** - 无需设置"N 次预览后清理"等魔法数字
5. **无后效性** - 可配置开关，安全回退

### 11.2 实施成本

- **开发时间**：8-10 个工作日
- **风险等级**：中等（有完备缓解措施）
- **用户体验影响**：小（切换图片延迟 200-500ms）

### 11.3 预期效果

**内存占用**：
- 同一张图片：200-400MB（稳定）
- 100 张图片：4-6GB（vs 当前 10-20GB）
- **改善幅度**：50-70%

**用户体验**：
- 首次预览：+200-400ms
- 后续预览：无影响
- 切换图片：+200-400ms
- **总体**：可接受

### 11.4 与现有方案的关系

| 方案 | 内存释放 | 实施成本 | 用户体验 | 推荐 |
|------|---------|---------|---------|------|
| jemalloc + workspace | 70-80% | 已完成 | 优秀 | ✅ 基线 |
| malloc_zone_pressure_relief | 10-30% | 已完成 | 优秀 | ✅ 补充 |
| **进程隔离** | **100%** | **10 天** | **良好** | ✅ **长期方案** |

**组合策略**：
- **Layer 1-3**（已实现）：减少内存分配，改善 heap 管理
- **Layer 4**（本方案）：彻底重置 heap，防止长期累积

---

## 12. 下一步行动

### 12.1 立即行动（今天）

1. **测试 jemalloc 效果**：
   ```bash
   # 运行应用，监控内存
   python -m divere
   # 使用 Activity Monitor 或 top 监控
   # 加载 100 张图片，每张预览 10-20 次
   # 记录最终内存占用
   ```

2. **评估是否需要进程隔离**：
   - 如果内存 <6GB：暂不需要
   - 如果内存 >8GB：需要进程隔离

### 12.2 如果决定实施（本周）

1. **创建 feature branch**：
   ```bash
   git checkout -b feature/process-isolation
   ```

2. **实施 Phase 1**（基础架构）：
   - 创建 `preview_worker_process.py`
   - 修改 `app_context.py`
   - 数据类型序列化支持
   - 单元测试

3. **中期评估**：
   - Phase 1 完成后测试效果
   - 测量内存占用和延迟
   - 决定是否继续 Phase 2

### 12.3 文档更新

- [ ] 更新 `MEMORY_SOLUTION.md` 添加进程隔离方案
- [ ] 更新 `CLAUDE.md` 添加架构说明
- [ ] 创建 `docs/process_isolation.md` 详细设计文档

---

## 附录 A: 关键代码示例

### A.1 完整的序列化支持

```python
# data_types.py

class ColorGradingParams:
    def to_dict(self) -> dict:
        """序列化为字典（用于进程间传递）"""
        return {
            'density_gamma': self.density_gamma,
            'density_matrix': self.density_matrix.tolist() if self.density_matrix is not None else None,
            'density_matrix_name': self.density_matrix_name,
            'enable_density_matrix': self.enable_density_matrix,
            'rgb_gains': list(self.rgb_gains),
            # ... 其他字段
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ColorGradingParams':
        """从字典反序列化"""
        params = cls()
        params.density_gamma = data['density_gamma']
        if data['density_matrix'] is not None:
            params.density_matrix = np.array(data['density_matrix'])
        params.density_matrix_name = data['density_matrix_name']
        params.enable_density_matrix = data['enable_density_matrix']
        params.rgb_gains = tuple(data['rgb_gains'])
        # ... 其他字段
        return params
```

### A.2 配置文件示例

```json
// config/app_settings.json
{
  "preview": {
    "use_process_isolation": "auto",  // "auto" | "always" | "never"
    "process_isolation": {
      "lazy_creation": true,
      "shutdown_timeout_seconds": 2.0,
      "result_poll_interval_ms": 16,
      "queue_max_size": 2
    }
  }
}
```

---

## 实施清单

### Phase 1: 基础架构 ⏳

#### 1.1 数据类型序列化支持
- [ ] `ColorGradingParams.to_dict()` / `from_dict()`
- [ ] `ImageData` 元数据序列化
- [ ] 测试 pickle 兼容性

#### 1.2 创建 PreviewWorkerProcess 模块
- [ ] 创建 `divere/core/preview_worker_process.py`
- [ ] 实现 `_worker_main_loop()` 函数
- [ ] 实现 `PreviewWorkerProcess` 类
  - [ ] `__init__()` - 初始化
  - [ ] `start()` - 启动进程
  - [ ] `request_preview()` - 发送预览请求
  - [ ] `try_get_result()` - 获取结果（非阻塞）
  - [ ] `shutdown()` - 优雅停止
  - [ ] `_cleanup_queues()` - 清理资源
- [ ] Shared memory 管理工具

#### 1.3 修改 ApplicationContext
- [ ] 添加配置字段
  - [ ] `_use_process_isolation`
  - [ ] `_preview_worker_process`
  - [ ] `_proxy_shared_memory`
  - [ ] `_result_poll_timer`
- [ ] 实现配置开关逻辑
  - [ ] `_should_use_process_isolation()`
- [ ] 修改 `load_image()`
  - [ ] 销毁旧 worker 进程
  - [ ] 清理 shared memory
- [ ] 实现进程模式预览
  - [ ] `_create_preview_worker_process()` - Lazy 创建
  - [ ] `_trigger_preview_with_process()` - 进程模式
  - [ ] `_trigger_preview_with_thread()` - 线程模式（重构）
  - [ ] `_poll_preview_result()` - 轮询结果
- [ ] 修改 `_trigger_preview_update()` - 分支选择

#### 1.4 配置文件
- [ ] 添加配置项到 `config/app_settings.json`
- [ ] 环境变量支持

#### 1.5 基础测试
- [ ] 进程启动/停止测试
- [ ] Shared memory 创建/销毁测试
- [ ] 简单预览流程测试
- [ ] 序列化/反序列化测试

---

### Phase 2: 稳定性和优化 ⏳

#### 2.1 异常处理
- [ ] Worker 崩溃检测
- [ ] Worker 自动重启机制
- [ ] 超时处理（worker 无响应）
- [ ] Shared memory 泄漏检测
- [ ] 主进程退出时 cleanup
- [ ] atexit handler

#### 2.2 性能优化
- [ ] 预览请求去重（只保留最新）
- [ ] 队列大小调优
- [ ] 减少不必要的拷贝

#### 2.3 自动回退机制
- [ ] 进程启动失败检测
- [ ] 自动回退到线程模式
- [ ] 用户友好的错误提示

---

### Phase 3: 集成测试 ⏳

#### 3.1 功能测试
- [ ] 加载 → 预览 → 切换 → 预览 → 导出
- [ ] 快速连续切换图片（10 次）
- [ ] 长时间运行（100+ 张图片）
- [ ] 色卡优化流程测试

#### 3.2 边缘情况测试
- [ ] 加载图片但不预览
- [ ] 切换图片时正在预览
- [ ] 导出时切换图片
- [ ] Worker 崩溃恢复

#### 3.3 内存测试
- [ ] Activity Monitor 监控内存占用
- [ ] 内存泄漏测试（100 次切换）
- [ ] 验证 heap 归还（进程终止后）

#### 3.4 性能测试
- [ ] 测量首次预览延迟
- [ ] 测量后续预览延迟
- [ ] 测量切换图片延迟
- [ ] 对比线程模式 vs 进程模式

#### 3.5 平台兼容性
- [ ] macOS 测试
- [ ] Linux 测试（可选）
- [ ] Windows 配置禁用测试

---

### Phase 4: 文档和发布 ⏳

#### 4.1 代码文档
- [ ] 代码注释完善
- [ ] 类型提示完善
- [ ] Docstring 完善

#### 4.2 用户文档
- [ ] 更新 `CLAUDE.md` 架构说明
- [ ] 创建配置说明文档
- [ ] 添加故障排查指南

#### 4.3 发布准备
- [ ] 代码审查
- [ ] 最终测试
- [ ] 创建 PR
- [ ] 更新 CHANGELOG

---

### 状态跟踪

| Phase | 状态 | 开始日期 | 完成日期 | 备注 |
|-------|------|---------|---------|------|
| Phase 1: 基础架构 | ✅ 完成 | 2025-11-16 | 2025-11-16 | 所有功能实现并测试通过 |
| Phase 2: 稳定性优化 | ⏸️ 待开始 | - | - | - |
| Phase 3: 集成测试 | ⏸️ 待开始 | - | - | - |
| Phase 4: 文档发布 | ⏸️ 待开始 | - | - | - |

---

**文档版本**：1.1
**创建日期**：2025-11-16
**最后更新**：2025-11-16
**状态**：实施中
**作者**：基于 `memory_analysis_report.md` 和代码库分析
