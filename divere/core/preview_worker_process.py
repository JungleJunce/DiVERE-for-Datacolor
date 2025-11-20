"""
Preview Worker Process - 独立进程预览处理

这个模块实现了进程隔离的预览处理架构，用于彻底解决 macOS heap 内存不归还问题。

核心思路：
- 为每张图片创建一个独立的 worker 进程
- 所有预览处理（density conversion, matrix, curves 等）在 worker 进程中进行
- 切换图片时销毁旧进程，操作系统强制回收所有 heap 内存
- 通过 shared_memory 传递大数组（proxy, result），通过 Queue 传递参数

无后效性保证：
- 配置开关可以禁用进程隔离
- 进程启动失败时自动回退到线程模式
- 不影响现有功能

参考文档：PROCESS_ISOLATION_ANALYSIS.md
"""

import multiprocessing
from multiprocessing import shared_memory, Queue, Process
import queue
import numpy as np
from typing import Optional, Dict, Any
import time
import traceback
import logging

logger = logging.getLogger(__name__)


def _load_proxy_from_shm(shm_name: str, shape: tuple, dtype: str):
    """从 shared memory 加载 proxy 图像

    Args:
        shm_name: shared memory 名称
        shape: 数组形状
        dtype: 数据类型

    Returns:
        ImageData: proxy 图像
    """
    from divere.core.data_types import ImageData

    shm = shared_memory.SharedMemory(name=shm_name)
    proxy_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    # 拷贝到 worker 进程内存（避免依赖主进程的 shm）
    proxy_image = ImageData(
        array=proxy_array.copy(),
        metadata={},
    )

    shm.close()  # 不 unlink，主进程负责清理
    return proxy_image


def _worker_main_loop(
    queue_request: Queue,
    queue_result: Queue,
    proxy_shm_name: str,
    proxy_shape: tuple,
    proxy_dtype: str,
    init_config: dict,
):
    """Worker 进程的主循环（在独立进程中运行）

    这个函数运行在独立进程中，不能访问主进程的对象。
    所有依赖对象都需要在这里重新初始化。

    Args:
        queue_request: 请求队列（主进程 → worker）
        queue_result: 结果队列（worker → 主进程）
        proxy_shm_name: proxy 图片的 shared memory 名称
        proxy_shape: proxy 数组形状
        proxy_dtype: proxy 数据类型
        init_config: 初始化配置（色彩空间、管线配置等）
    """
    try:
        # ============ Step 1: 初始化（在 worker 进程中） ============
        from divere.core.the_enlarger import TheEnlarger
        from divere.core.color_space import ColorSpaceManager
        from divere.core.data_types import ImageData, ColorGradingParams

        # 重新创建对象（不能共享主进程的对象）
        the_enlarger = TheEnlarger()
        color_space_manager = ColorSpaceManager()

        # ============ Step 2: 加载 proxy 从 shared memory ============
        proxy_image = _load_proxy_from_shm(proxy_shm_name, proxy_shape, proxy_dtype)

        # ============ Step 3: 主循环：处理预览请求 ============
        while True:
            # 3.1 接收请求
            request = queue_request.get()  # 阻塞等待

            # 3.2 停止信号
            if request is None:
                break

            # 3.3 解析请求
            action = request.get('action')

            # === 处理 reload_proxy 请求 ===
            if action == 'reload_proxy':
                try:
                    new_shm_name = request['proxy_shm_name']
                    new_shape = request['proxy_shape']
                    new_dtype = request['proxy_dtype']

                    # 释放旧 proxy 内存
                    if proxy_image is not None and hasattr(proxy_image, 'array'):
                        proxy_image.array = None

                    # 重新加载 proxy
                    proxy_image = _load_proxy_from_shm(new_shm_name, new_shape, new_dtype)

                    queue_result.put({'status': 'proxy_reloaded'})
                except Exception as e:
                    queue_result.put({
                        'status': 'error',
                        'message': f"Failed to reload proxy: {e}",
                        'traceback': traceback.format_exc()
                    })
                continue

            # === 处理 get_memory 请求 ===
            if action == 'get_memory':
                try:
                    import psutil
                    import os
                    process = psutil.Process(os.getpid())
                    mem_info = process.memory_info()
                    queue_result.put({
                        'status': 'memory_info',
                        'rss_mb': mem_info.rss / 1024 / 1024,
                        'vms_mb': mem_info.vms / 1024 / 1024
                    })
                except ImportError:
                    queue_result.put({
                        'status': 'error',
                        'message': 'psutil not available'
                    })
                except Exception as e:
                    queue_result.put({
                        'status': 'error',
                        'message': f"Failed to get memory: {e}"
                    })
                continue

            # === 处理 preview 请求 ===
            if action != 'preview':
                # 未知 action，忽略
                continue

            params = ColorGradingParams.from_dict(request['params'])
            crop_rect_norm = request.get('crop_rect_norm')
            orientation = request.get('orientation', 0)
            idt_gamma = request.get('idt_gamma', 1.0)
            convert_to_monochrome = request.get('convert_to_monochrome', False)
            display_metadata = request.get('display_metadata', {})
            custom_colorspace_def = request.get('custom_colorspace_def')

            # 3.4 动态准备proxy（每次根据请求参数执行完整变换链）
            try:
                # === Step A: Crop ===
                working_image = ImageData(
                    array=proxy_image.array.copy(),
                    metadata=proxy_image.metadata.copy()
                )

                if crop_rect_norm:
                    x, y, w, h = crop_rect_norm
                    h_orig, w_orig = working_image.array.shape[:2]
                    x0 = int(round(x * w_orig))
                    y0 = int(round(y * h_orig))
                    x1 = int(round((x + w) * w_orig))
                    y1 = int(round((y + h) * h_orig))
                    x0 = max(0, min(w_orig - 1, x0))
                    x1 = max(x0 + 1, min(w_orig, x1))
                    y0 = max(0, min(h_orig - 1, y0))
                    y1 = max(y0 + 1, min(h_orig, y1))
                    working_image.array = working_image.array[y0:y1, x0:x1, :].copy()

                # === Step B: IDT Gamma ===
                if abs(idt_gamma - 1.0) > 1e-6:
                    working_image.array = the_enlarger.pipeline_processor.math_ops.apply_power(
                        working_image.array, idt_gamma, use_optimization=True
                    )

                # === Step B.5: 注册自定义色彩空间（如果需要）===
                if custom_colorspace_def:
                    try:
                        cs_name = custom_colorspace_def.get('name')
                        primaries_xy = np.array(custom_colorspace_def.get('primaries_xy'), dtype=float)
                        white_point_xy = np.array(custom_colorspace_def.get('white_point_xy'), dtype=float)
                        gamma = float(custom_colorspace_def.get('gamma', 1.0))

                        # 在worker进程的ColorSpaceManager中注册自定义色彩空间
                        color_space_manager.register_custom_colorspace(
                            name=cs_name,
                            primaries_xy=primaries_xy,
                            white_point_xy=white_point_xy,
                            gamma=gamma
                        )
                    except Exception as e:
                        # 注册失败不应导致预览失败，记录错误并继续
                        logger.warning(f"Failed to register custom colorspace in worker: {e}")

                # === Step C: Color Transform ===
                working_image = color_space_manager.set_image_color_space(
                    working_image, params.input_color_space_name
                )
                working_image = color_space_manager.convert_to_working_space(
                    working_image, skip_gamma_inverse=True
                )

                # === Step D: Rotate ===
                if orientation % 360 != 0:
                    k = (orientation // 90) % 4
                    if k != 0:
                        working_image.array = np.rot90(working_image.array, k=int(k))

                # === Step E: Pipeline处理 ===
                monochrome_converter = None
                if convert_to_monochrome:
                    monochrome_converter = color_space_manager.convert_to_monochrome

                result_image = the_enlarger.apply_full_pipeline(
                    working_image,
                    params,
                    convert_to_monochrome_in_idt=convert_to_monochrome,
                    monochrome_converter=monochrome_converter,
                )
                result_image = color_space_manager.convert_to_display_space(
                    result_image, "DisplayP3"
                )

                # 添加orientation到metadata供UI使用（用于正确显示crop overlay）
                result_image.metadata['orientation'] = orientation
                result_image.metadata['global_orientation'] = orientation

                # 合并主进程传递的display_metadata（包含crop_focused, crop_overlay等显示状态）
                result_image.metadata.update(display_metadata)

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


class PreviewWorkerProcess:
    """管理一个独立的预览处理进程

    生命周期：
    1. 创建时初始化队列和配置，但不启动进程
    2. start() 启动进程
    3. request_preview() 发送预览请求（非阻塞）
    4. try_get_result() 获取结果（非阻塞）
    5. shutdown() 优雅停止进程

    内存管理：
    - proxy 图片通过 shared_memory 传递（一次性）
    - 参数通过 pickle + Queue 传递（每次预览）
    - 结果通过 shared_memory 返回（每次预览）

    无后效性：
    - 进程启动失败不会影响主程序
    - shutdown() 总是安全的（幂等）
    """

    def __init__(
        self,
        proxy_shm_name: str,
        proxy_shape: tuple,
        proxy_dtype: str,
        init_config: Optional[Dict[str, Any]] = None
    ):
        """初始化但不启动进程

        Args:
            proxy_shm_name: shared memory 名称（proxy 已写入）
            proxy_shape: proxy 数组形状
            proxy_dtype: proxy 数据类型
            init_config: 初始化配置（可选）
        """
        self.proxy_shm_name = proxy_shm_name
        self.proxy_shape = proxy_shape
        self.proxy_dtype = proxy_dtype
        self.init_config = init_config or {}

        # IPC 组件
        self.queue_request = Queue(maxsize=2)  # 参数队列（限制大小避免积压）
        self.queue_result = Queue(maxsize=2)   # 结果队列

        self.process: Optional[Process] = None

        # 崩溃检测和自动重启
        self._restart_count = 0
        self._max_restart_attempts = 3  # 最多自动重启 3 次
        self._last_request_time = 0.0
        self._request_timeout = 300.0  # 300 秒超时

        # Shared memory 泄漏追踪
        self._active_result_shm = set()  # 追踪未清理的 result shared memory

    def start(self):
        """启动 worker 进程"""
        if self.process is not None and self.process.is_alive():
            logger.warning("Worker process already running, skipping start")
            return

        self.process = Process(
            target=_worker_main_loop,
            args=(
                self.queue_request,
                self.queue_result,
                self.proxy_shm_name,
                self.proxy_shape,
                self.proxy_dtype,
                self.init_config,
            )
        )
        self.process.start()

    def is_alive(self) -> bool:
        """检查 worker 进程是否存活"""
        return self.process is not None and self.process.is_alive()

    def reload_proxy(self, proxy_shm_name: str, proxy_shape: tuple, proxy_dtype: str):
        """重新加载 proxy（不重启进程）

        用于切换图片时复用 worker 进程，避免重新 import 模块的开销

        Args:
            proxy_shm_name: 新 proxy 的 shared memory 名称
            proxy_shape: 新 proxy 数组形状
            proxy_dtype: 新 proxy 数据类型
        """
        if not self.is_alive():
            logger.warning("Worker process not alive, cannot reload proxy")
            return

        # 更新 proxy 元数据
        self.proxy_shm_name = proxy_shm_name
        self.proxy_shape = proxy_shape
        self.proxy_dtype = proxy_dtype

        # 发送 reload_proxy 请求
        request_dict = {
            'action': 'reload_proxy',
            'proxy_shm_name': proxy_shm_name,
            'proxy_shape': proxy_shape,
            'proxy_dtype': proxy_dtype,
        }

        try:
            self.queue_request.put(request_dict, timeout=1.0)
        except queue.Full:
            logger.warning("Request queue full, cannot reload proxy")

    def get_memory_usage(self) -> Optional[float]:
        """获取 worker 进程内存使用量（MB）

        macOS 上 psutil 的 RSS 严重低估内存（因为内存压缩），
        改用 `ps` 命令直接读取物理内存占用。

        Returns:
            float: 内存使用量（单位 MB），如果获取失败返回 None
        """
        if not self.is_alive():
            logger.debug("Worker not alive, cannot get memory usage")
            return None

        try:
            import platform
            import subprocess

            if self.process is None:
                logger.debug("Worker process object is None")
                return None

            pid = self.process.pid

            # macOS: 使用 ps 命令读取真实物理内存
            # RSS 字段在 macOS 上低估内存，需要直接解析 ps 输出
            if platform.system() == 'Darwin':
                try:
                    # 使用 ps 命令获取进程内存（KB）
                    # -o rss= 返回实际物理内存（虽然还是可能低估，但比 psutil 准确）
                    result = subprocess.run(
                        ['ps', '-p', str(pid), '-o', 'rss='],
                        capture_output=True,
                        text=True,
                        timeout=1.0
                    )

                    if result.returncode == 0:
                        rss_kb = int(result.stdout.strip())
                        mem_mb = rss_kb / 1024

                        #
                        return mem_mb
                    else:
                        logger.debug(f"ps command failed: {result.stderr}")
                        return None

                except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError) as e:
                    logger.debug(f"Failed to run ps command: {e}")
                    # 降级到 psutil
                    pass

            # Linux/Windows 或 macOS fallback：使用 psutil
            try:
                import psutil
                proc = psutil.Process(pid)
                mem_info = proc.memory_info()
                mem_mb = mem_info.rss / 1024 / 1024
                logger.debug(f"Worker PID {pid} memory (psutil): {mem_mb:.1f} MB")
                return mem_mb
            except ImportError:
                print(f"[WORKER] ⚠️  psutil not installed, cannot monitor memory. Install with: pip install psutil")
                return None

        except Exception as e:
            # 其他异常也输出，方便调试
            print(f"[WORKER] ⚠️  Failed to get memory usage: {e}")
            logger.debug(f"Failed to get memory usage: {e}")
            import traceback
            traceback.print_exc()
            return None

    def request_preview(self, params,
                        crop_rect_norm=None,
                        orientation: int = 0,
                        idt_gamma: float = 1.0,
                        convert_to_monochrome: bool = False,
                        display_metadata: dict = None,
                        custom_colorspace_def: dict = None):
        """请求预览（非阻塞）

        只保留最新请求，丢弃旧的未处理请求（预览去重）

        Args:
            params: ColorGradingParams 实例
            crop_rect_norm: 归一化裁剪区域 (x, y, w, h) 或 None
            orientation: 旋转角度（0/90/180/270）
            idt_gamma: IDT gamma 值
            convert_to_monochrome: 是否转换为单色
            display_metadata: 显示状态元数据（crop_focused, crop_overlay等）
            custom_colorspace_def: 自定义色彩空间定义（用于动态注册的primaries）
        """
        # 检查 worker 是否存活，如果崩溃则尝试重启
        if not self.is_alive():
            if self._try_restart():
                logger.info("Worker process restarted successfully")
            else:
                logger.error("Worker process not alive and restart failed")
                return

        # 清空旧请求（只保留最新）
        while not self.queue_request.empty():
            try:
                self.queue_request.get_nowait()
            except queue.Empty:
                break

        # 构建请求字典
        request_dict = {
            'action': 'preview',
            'params': params.to_full_dict(),
            'crop_rect_norm': crop_rect_norm,
            'orientation': orientation,
            'idt_gamma': idt_gamma,
            'convert_to_monochrome': convert_to_monochrome,
            'display_metadata': display_metadata or {},
            'custom_colorspace_def': custom_colorspace_def,
            'timestamp': time.time()
        }

        # 发送新请求
        try:
            self.queue_request.put(request_dict, timeout=0.1)
            self._last_request_time = time.time()
        except queue.Full:
            logger.warning("Request queue full, dropping old request")
            # 再次尝试清空并发送
            try:
                self.queue_request.get_nowait()
                self.queue_request.put(request_dict, timeout=0.1)
                self._last_request_time = time.time()
            except:
                pass

    def _try_restart(self) -> bool:
        """尝试重启 worker 进程

        Returns:
            bool: 重启是否成功
        """
        if self._restart_count >= self._max_restart_attempts:
            logger.error(f"Worker restart failed: exceeded max attempts ({self._max_restart_attempts})")
            return False

        logger.warning(f"Worker process died, attempting restart ({self._restart_count + 1}/{self._max_restart_attempts})")

        try:
            # 清理旧进程
            if self.process is not None:
                try:
                    self.process.join(timeout=0.5)
                except:
                    pass
                self.process = None

            # 重新启动
            self.start()
            time.sleep(0.1)

            # 验证启动成功
            if self.is_alive():
                self._restart_count += 1
                return True
            else:
                logger.error("Worker process failed to start after restart")
                return False

        except Exception as e:
            logger.error(f"Worker restart exception: {e}")
            return False

    def try_get_result(self):
        """尝试获取结果（非阻塞）

        Returns:
            ImageData: 预览结果
            Exception: 处理出错
            None: 暂无结果
        """
        # 超时检测：如果请求发出后很久没有响应，可能 worker 卡住了
        if self._last_request_time > 0:
            elapsed = time.time() - self._last_request_time
            if elapsed > self._request_timeout:
                pass
                #logger.warning(f"Worker timeout detected ({elapsed:.1f}s > {self._request_timeout}s)")
                # 不自动重启，只记录警告（避免频繁重启）
                # 用户下次操作时会触发重启

        try:
            result_info = self.queue_result.get_nowait()
        except queue.Empty:
            return None

        # 处理错误消息
        if result_info['status'] == 'error':
            return Exception(result_info['message'])

        # 过滤内部消息（不是预览结果）
        if result_info['status'] in ('proxy_reloaded', 'memory_info'):
            # 内部状态消息，忽略并继续等待预览结果
            return None

        # 预览结果必须包含 'shm_name'
        if result_info['status'] != 'success':
            logger.warning(f"Unexpected result status: {result_info['status']}")
            return None

        # 从 shared memory 读取结果
        shm_name = result_info['shm_name']
        try:
            from divere.core.data_types import ImageData

            # 追踪 shared memory（用于泄漏检测）
            self._active_result_shm.add(shm_name)

            shm = shared_memory.SharedMemory(name=shm_name)
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

            # 从追踪集合中移除
            self._active_result_shm.discard(shm_name)

            return result_image

        except Exception as e:
            logger.error(f"Failed to read result from shared memory: {e}")
            # 尝试清理泄漏的 shared memory
            try:
                shm = shared_memory.SharedMemory(name=shm_name)
                shm.close()
                shm.unlink()
                self._active_result_shm.discard(shm_name)
            except:
                pass
            return Exception(f"Failed to read result: {e}")

    def shutdown(self):
        """优雅停止进程（幂等操作）"""
        if self.process is None:
            return

        # 发送停止信号
        try:
            self.queue_request.put(None, timeout=0.5)
        except:
            pass

        # 等待退出
        if self.process.is_alive():
            self.process.join(timeout=2.0)

            # 如果超时，强制终止
            if self.process.is_alive():
                logger.warning("Worker process did not exit gracefully, terminating")
                self.process.terminate()
                self.process.join(timeout=1.0)

                # 最后手段：kill
                if self.process.is_alive():
                    logger.error("Worker process did not terminate, killing")
                    self.process.kill()
                    self.process.join(timeout=0.5)

        self.process = None

        # 清理残留资源
        self._cleanup_queues()
        self._cleanup_leaked_shm()

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

    def _cleanup_leaked_shm(self):
        """清理泄漏的 shared memory（从追踪集合）"""
        if not self._active_result_shm:
            return

        logger.warning(f"Cleaning up {len(self._active_result_shm)} leaked shared memory blocks")

        for shm_name in list(self._active_result_shm):
            try:
                shm = shared_memory.SharedMemory(name=shm_name)
                shm.close()
                shm.unlink()
                logger.info(f"Cleaned up leaked shared memory: {shm_name}")
            except Exception as e:
                logger.debug(f"Failed to cleanup shared memory {shm_name}: {e}")

        self._active_result_shm.clear()
