# 允许类型注解使用字符串形式的前向引用
from __future__ import annotations

# 类型注解工具
from typing import Any, Callable, Dict, Tuple

# init_logger 创建模块 logger
from minisgl.utils import init_logger

# 创建模块级 logger
logger = init_logger(__name__)


# 在独立 CUDA stream 上精确测量函数的 GPU 执行时间（可选 CUDA Graph 加速重复调用）。
# init_stream 是否用专用 stream；repetitions 计时时重复次数；
# cuda_graph_repetitions 图内打包次数（None 表示不用图）
def perf_cuda(
    f: Callable[[], Any],
    *,
    init_stream: bool = True,
    repetitions: int = 10,
    cuda_graph_repetitions: int | None = 10,
) -> float:
    # 函数内导入 torch，保持模块导入轻量
    import torch

    # 重复次数必须为正
    assert repetitions > 0
    # 起始计时事件（GPU 侧精确计时）
    tic = torch.cuda.Event(enable_timing=True)
    # 结束计时事件
    toc = torch.cuda.Event(enable_timing=True)
    # 先建一个临时 stream
    stream = torch.cuda.Stream()
    # 等待 GPU 全部空闲，保证计时起点干净
    torch.cuda.synchronize()
    # 按需选择 stream
    if init_stream:
        # 用新建的专用 stream
        stream = torch.cuda.Stream()
    else:
        # 沿用当前 stream
        stream = torch.cuda.current_stream()

    # 切换到目标 stream 执行准备工作与图捕获
    with torch.cuda.stream(stream):
        # 先裸跑一次：触发内核编译/JIT/缓存分配等一次性开销
        f()
        # 需要用 CUDA Graph 时把 N 次调用打包进图，之后一次 replay 顶 N 次
        if N := cuda_graph_repetitions:
            # 创建图对象
            g = torch.cuda.CUDAGraph()
            # 进入图捕获模式
            with torch.cuda.graph(g):
                # 捕获 N 次调用
                for _ in range(N):
                    # 记录 kernel 启动到图中
                    f()
            # 用 replay 句柄代替原函数
            replay = g.replay
            # 删除图对象本身（replay 句柄仍可用）
            del g
        else:
            # 不用图：直接重复调用原函数
            replay = f
            # 单次即一次
            N = 1

        # 再次同步，确保预热完成
        torch.cuda.synchronize()

        # 再跑一次 replay，确保进入稳定状态
        replay()
        # 记录起始事件
        tic.record()
        # 连续执行 repetitions 次 replay
        for _ in range(repetitions):
            # 反复回放
            replay()
        # 记录结束事件
        toc.record()
        # 等待结束事件完成
        toc.synchronize()
        # 两个事件间的 GPU 时间（毫秒）
        dur = tic.elapsed_time(toc)
        # 返回单次调用的平均毫秒数
        return dur / (N * repetitions)


# 对比基线实现与自研实现的带宽/延迟，打印结果并返回两者带宽。
# memory_footprint 是该操作读写的总字节数（折算带宽用）
def compare_memory_kernel_perf(
    *,
    baseline: Callable[[], Any],
    our_impl: Callable[[], Any],
    memory_footprint: int,  # in bytes
    description: str = " ",
    extra_kwargs: Dict[str, Any] | None = None,
    need_latency: bool = True,
) -> Tuple[float, float]:
    # 未传则用空字典
    extra_kwargs = extra_kwargs or {}

    # 测量基线耗时（毫秒）
    dur = perf_cuda(baseline, **extra_kwargs)
    # 折算基线带宽：字节 / 毫秒 => GB/s
    bandwidth_0 = memory_footprint / (dur * 1e6)  # GB/s
    # 按需拼延迟片段
    latency_msg = f"{dur:8.3f} ms | " if need_latency else ""
    # 组装基线结果字符串
    message_0 = f"Baseline: {latency_msg}{bandwidth_0:8.3f} GB/s"

    # 测量自研实现耗时
    dur = perf_cuda(our_impl, **extra_kwargs)
    # 折算自研实现带宽
    bandwidth_1 = memory_footprint / (dur * 1e6)  # GB/s
    # 按需拼延迟片段
    latency_msg = f"{dur:8.3f} ms | " if need_latency else ""
    # 一行打印基线与自研实现的对比
    logger.info(f"{description}{message_0} | Our Impl: {latency_msg}{bandwidth_1:8.3f} GB/s")
    # 返回两者带宽供程序化比较
    return bandwidth_0, bandwidth_1
