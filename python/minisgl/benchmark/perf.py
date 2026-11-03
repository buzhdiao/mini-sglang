from __future__ import annotations  # 启用 PEP 563 延迟注解求值

from typing import Any, Callable, Dict, Tuple  # 导入类型注解工具

from minisgl.utils import init_logger  # 导入日志初始化工具

logger = init_logger(__name__)  # 创建本模块 logger


def perf_cuda(
    f: Callable[[], Any],  # 被测函数（无参可调用体）
    *,
    init_stream: bool = True,  # 是否使用独立 CUDA stream 计时
    repetitions: int = 10,  # 实际计时的重复次数
    cuda_graph_repetitions: int | None = 10,  # CUDA Graph 内捕获的重复次数（None/0 表示不用 Graph）
) -> float:
    """用 CUDA Event 测量函数平均耗时：可选 CUDA Graph 捕获以消除 kernel launch 开销，返回单次执行平均毫秒数。"""
    import torch  # 函数内延迟导入 torch，避免模块加载即依赖 CUDA 环境

    assert repetitions > 0  # 重复次数必须为正数
    tic = torch.cuda.Event(enable_timing=True)  # 起点事件（开启计时功能）
    toc = torch.cuda.Event(enable_timing=True)  # 终点事件（开启计时功能）
    stream = torch.cuda.Stream()  # 预创建一个独立 stream（后续可能被覆盖）
    torch.cuda.synchronize()  # 等待设备上全部工作完成，保证计时起点干净
    if init_stream:  # 使用独立 stream
        stream = torch.cuda.Stream()  # 新建 stream（与其他工作隔离，计时更纯净）
    else:  # 使用当前 stream
        stream = torch.cuda.current_stream()  # 沿用当前上下文（如调用方已自行设置 stream，测试中常见）

    with torch.cuda.stream(stream):  # 在目标 stream 上执行后续全部操作
        f()  # 先跑一次：触发初始化/编译/缓存等一次性开销，不计入测量
        if N := cuda_graph_repetitions:  # 海象表达式：N 非 None 且非 0 时启用 CUDA Graph
            g = torch.cuda.CUDAGraph()  # 创建 CUDA Graph 对象
            with torch.cuda.graph(g):  # 进入图捕获模式（期间的操作不会真正执行）
                for _ in range(N):  # 把 N 次调用捕获为一张图
                    f()
            replay = g.replay  # 之后以整图回放代替逐次调用
            del g  # 删除 Python 侧引用（回放句柄仍有效，顺便释放多余对象）
        else:  # 不使用 CUDA Graph 时
            replay = f  # 直接以原函数作为回放体
            N = 1  # 单次调用等效于 N=1

        torch.cuda.synchronize()  # 同步：确保捕获完成且设备空闲后再开始计时

        replay()  # 预热一次回放（激活图/缓存路径）
        tic.record()  # 在当前 stream 记录起点事件
        for _ in range(repetitions):  # 连续回放 repetitions 次
            replay()
        toc.record()  # 记录终点事件
        toc.synchronize()  # CPU 阻塞等待终点事件在 GPU 上完成，保证时间有效
        dur = tic.elapsed_time(toc)  # 两个事件之间的 GPU 时间（毫秒）
        return dur / (N * repetitions)  # 平均单次 f() 耗时 = 总时长 / (每图 N 次 × 图重复次数)


def compare_memory_kernel_perf(
    *,
    baseline: Callable[[], Any],  # 基线实现（参考对照）
    our_impl: Callable[[], Any],  # 我们自己的实现
    memory_footprint: int,  # in bytes  # 该操作读写的总字节数（用于换算带宽）
    description: str = " ",  # 日志前缀描述（如 "BS=4 | "）
    extra_kwargs: Dict[str, Any] | None = None,  # 透传给 perf_cuda 的额外参数
    need_latency: bool = True,  # 是否在输出中包含延迟数据
) -> Tuple[float, float]:
    """对比基线与自家实现的带宽：各测一次 perf_cuda，日志输出对比并返回 (基线带宽, 自家带宽)，单位 GB/s。"""
    extra_kwargs = extra_kwargs or {}  # None 时回退为空字典

    dur = perf_cuda(baseline, **extra_kwargs)  # 测量基线实现耗时（毫秒）
    bandwidth_0 = memory_footprint / (dur * 1e6)  # GB/s  # 带宽 = 字节数 / (毫秒×1e6)（等价于 字节/纳秒→GB/s）
    latency_msg = f"{dur:8.3f} ms | " if need_latency else ""  # 按需构造延迟字段文本
    message_0 = f"Baseline: {latency_msg}{bandwidth_0:8.3f} GB/s"  # 组装基线结果文本

    dur = perf_cuda(our_impl, **extra_kwargs)  # 测量自家实现耗时
    bandwidth_1 = memory_footprint / (dur * 1e6)  # GB/s  # 同样方式换算带宽
    latency_msg = f"{dur:8.3f} ms | " if need_latency else ""  # 延迟字段
    logger.info(f"{description}{message_0} | Our Impl: {latency_msg}{bandwidth_1:8.3f} GB/s")  # 输出两者对比结果
    return bandwidth_0, bandwidth_1  # 返回两组带宽供调用方进一步分析（如计算加速比）
