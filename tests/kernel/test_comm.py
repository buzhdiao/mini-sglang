import os  # 导入 os，设置分布式初始化所需的环境变量（MASTER_ADDR/PORT）
import time  # 导入 time，用于模拟某 rank 滞后的 sleep
import torch  # 导入 torch，张量与 CUDA/分布式操作
from minisgl.distributed import set_tp_info  # 导入 TP 信息设置函数（通信层依赖全局 rank/size）
import minisgl.kernel as kernel  # 导入 kernel 模块，使用其中的 pynccl 通信封装
from tqdm import tqdm  # 导入进度条（graph 捕获与逐步计时循环的显示）

from minisgl.utils import init_logger  # 导入日志初始化工具


logger = init_logger(__name__)  # 创建本模块 logger


@torch.no_grad()  # 禁用梯度追踪（纯通信测试无需反向传播）
def run(tp_size: int, tp_rank: int):
    """单 rank 的测试主体：初始化 pynccl → all-reduce 正确性校验 → 性能压测 → all-gather 校验。"""
    torch.cuda.set_device(tp_rank)  # 绑定本进程使用的 GPU（rank 与设备一一对应）
    torch.cuda.set_stream(torch.cuda.Stream(tp_rank))  # type: ignore  # 使用独立 stream，避免与默认流互相干扰
    stream = torch.cuda.current_stream()  # 记录当前 stream 备用（逐步执行路径使用）
    set_tp_info(tp_rank, tp_size)  # 设置全局 TP rank/size 信息

    # cpu group  # 初始化 CPU 进程组（gloo 后端，用于进程间协调与 NCCL 通信器关联）
    torch.distributed.init_process_group(
        world_size=tp_size,  # 全局进程数
        rank=tp_rank,  # 本进程 rank
        backend="gloo",  # CPU 通信后端
    )

    # use default cpu group  # 直接使用默认 WORLD 进程组
    tp_cpu_group = torch.distributed.group.WORLD  # 取 WORLD 组作为 TP CPU 组
    assert tp_cpu_group is not None, "CPU group should not be None"  # 断言进程组存在
    dtype = torch.float16  # 测试使用的张量精度 fp16

    K = 512  # 消息规模系数（总元素数 = 8192 × K）
    USE_SYMM = 0  # 是否启用对称内存注册缓冲（0=否，走普通 NCCL 路径）

    comm = kernel.init_pynccl(  # 初始化 pynccl 通信器封装
        tp_rank=tp_rank,  # 本 rank 编号
        tp_size=tp_size,  # 总 rank 数
        tp_cpu_group=tp_cpu_group,  # 关联的 CPU 进程组
        max_size_bytes=8192 * K * dtype.itemsize if USE_SYMM else 0,  # SYMM 模式下预注册缓冲大小，否则传 0
    )

    def bench_performance(f, use_graph=False):
        """压测 all-reduce 性能：可选 CUDA Graph 捕获，统计平均延迟（微秒）与带宽（GB/s）。"""
        import gc  # 函数内导入 gc，用于控制垃圾回收

        gc.collect()  # 先触发一次完整回收
        gc.disable()  # 计时期间禁用 GC，避免回收抖动影响测量

        N = 1024  # 每轮重复次数
        M = 16  # 计时轮数
        x = torch.zeros(8192 * K, dtype=dtype, device=f"cuda:{tp_rank}")  # 被归约的输入缓冲（全 0）
        f(x)  # 预热两次（触发 NCCL 通道/资源初始化）
        f(x)

        pbar = tqdm(list(range(N)), desc="Capturing cuda graph", disable=tp_rank > 0)  # 进度条仅 rank0 显示，避免多进程刷屏

        torch.cuda.synchronize()  # 同步后再开始捕获/计时
        if use_graph:  # CUDA Graph 路径
            g = torch.cuda.CUDAGraph()  # 创建图对象
            graph = torch.cuda.graph(g)  # 获取图捕获上下文
            with graph:
                for _ in pbar:  # 捕获 N 次通信操作进图
                    f(x)
            cur_stream = graph.capture_stream  # 记录捕获所用 stream（回放需在其上）
        else:  # 逐步执行路径
            nonlocal stream  # 引用并更新外层 stream 变量
            f(x)  # 再预热两次
            f(x)
            cur_stream = stream  # 使用当前 stream

        tic = torch.cuda.Event(enable_timing=True)  # 起点 GPU 事件
        toc = torch.cuda.Event(enable_timing=True)  # 终点 GPU 事件
        with torch.cuda.stream(cur_stream):  # 在目标 stream 上计时
            tic.record(cur_stream)  # 记录起点
            if use_graph:  # 图回放路径：M 轮 × 每轮整图（内含 N 次）
                for _ in range(M):
                    g.replay()  # type: ignore  # 整图回放
            else:  # 逐步路径：M 轮 × N 次
                for _ in range(M):
                    for _ in pbar:
                        f(x)
            toc.record(cur_stream)  # 记录终点
        gc.enable()  # 恢复垃圾回收
        toc.synchronize()  # CPU 等待终点事件完成，保证时长有效
        elapsed_time = tic.elapsed_time(toc)  # GPU 总耗时（毫秒）
        avg_time = elapsed_time * 1000 / (M * N)  # 平均单次耗时（毫秒→微秒）
        logger.info(f"Rank {tp_rank} all-reduce avg time: {avg_time: .4f} us")  # 打印平均延迟
        bandwidth = (8192 * K * dtype.itemsize) / (avg_time * 1e3)  # in GB/s  # 带宽 = 字节数 / 纳秒数
        logger.info(f"Rank {tp_rank} all-reduce bandwidth: {bandwidth:.2f} GB/s")  # 打印带宽
        # print memory usage  # 打印显存占用
        mem_usage = torch.cuda.memory_allocated() / (1024 * 1024)  # 当前已分配显存（MB）
        logger.info(f"Rank {tp_rank} memory usage: {mem_usage:.2f} MB")
        torch.cuda.empty_cache()  # 释放缓存的显存块
        torch.cuda.reset_peak_memory_stats()  # 重置峰值统计，避免影响后续测量

    def test_correctness(f):
        """all-reduce 正确性校验：三组不同输入模式的结果断言 + rank 滞后同步性检查。"""
        N = 4  # 连续归约次数
        x = torch.ones(8192 * K, dtype=dtype, device=f"cuda:{tp_rank}")  # 输入全 1
        for _ in range(N):  # 连续归约 4 次（每次全体求和）
            f(x)
        ans = pow(tp_size, N)  # 期望值：每次全加 tp_size 份、共 N 次 → tp_size^N
        y = torch.full((8192 * K,), ans, dtype=dtype, device=f"cuda:{tp_rank}")  # 期望张量

        assert torch.allclose(x, y), f"Rank {tp_rank} failed: {x} != {y}"  # 校验多次归约结果正确

        x = torch.full((8192 * K,), tp_rank, dtype=dtype, device=f"cuda:{tp_rank}")  # 输入为本 rank 编号
        # sanity check: if one TP rank lags behind, the others should wait  # 同步性检查：某 rank 滞后时其他 rank 应等待（集合通信阻塞语义）
        if tp_rank == 0:  # 仅 rank0 故意延迟
            torch.cuda.synchronize()  # 先确保 rank0 的数据写入完成
            time.sleep(1)  # 滞后 1 秒再参与归约
        f(x)  # 归约一次：结果为所有 rank 编号之和
        ans = (tp_size * (tp_size - 1)) // 2  # 期望值 = 0+1+...+(tp_size-1)（等差数列求和）
        y = torch.full((8192 * K,), ans, dtype=dtype, device=f"cuda:{tp_rank}")  # 期望张量
        assert torch.allclose(x, y), f"Rank {tp_rank} failed: {x} != {y}"  # 校验（同时验证了跨 rank 同步语义）

        # to prevent overflow, we use a smaller value for this test  # 防止 fp16 溢出，改用 0/1 小值输入
        x = torch.cat(  # 前半 0、后半 1
            [
                torch.zeros((8192 * K // 2,), dtype=dtype, device=f"cuda:{tp_rank}"),
                torch.ones((8192 * K // 2,), dtype=dtype, device=f"cuda:{tp_rank}"),
            ]
        )
        f(x)  # 归约一次
        y = torch.cat(  # 期望：前半 0、后半 tp_size
            [
                torch.zeros((8192 * K // 2,), dtype=dtype, device=f"cuda:{tp_rank}"),
                torch.full((8192 * K // 2,), tp_size, dtype=dtype, device=f"cuda:{tp_rank}"),
            ]
        )
        assert torch.allclose(x, y), f"Rank {tp_rank} failed: {x} != {y}"  # 校验分段结果正确

        if N % 2 != 0:  # N 为奇数时
            f(x)  # 再归约一次，使后续总操作次数为偶数（保持各 rank 状态一致）

        logger.info(f"Correctness check for rank {tp_rank} passed")  # 打印通过信息

    test_correctness(lambda x: comm.all_reduce(x, "sum"))  # 无图模式的正确性校验
    bench_performance(lambda x: comm.all_reduce(x, "sum"))  # 无图模式的性能压测
    test_correctness(lambda x: comm.all_reduce(x, "sum"))  # 压测后再验一次正确性（确认通信状态未被破坏）

    # test all gather  # 测试 all-gather
    src = torch.full((K,), tp_rank, dtype=dtype, device=f"cuda:{tp_rank}")  # 每个 rank 提供自己编号的 K 元向量
    torch.cuda.synchronize()  # 同步后再发通信
    dst = torch.empty((K * tp_size,), dtype=dtype, device=f"cuda:{tp_rank}")  # 输出缓冲：K × tp_size
    comm.all_gather(dst, src)  # 执行 all-gather（各 rank 的 src 按 rank 序拼接）
    torch.cuda.synchronize()  # 等待通信完成
    expected = torch.arange(tp_size, dtype=dtype, device=f"cuda:{tp_rank}")  # 期望基础序列：[0..tp_size-1]
    expected = expected.repeat_interleave(K)  # 每个编号重复 K 次
    assert torch.allclose(dst, expected), f"Rank {tp_rank} all-gather failed"  # 校验拼接顺序与内容正确
    torch.distributed.destroy_process_group()  # 销毁进程组，清理分布式资源


if __name__ == "__main__":  # 直接运行本文件时启动多进程测试
    import multiprocessing as mp  # 主入口内导入多进程模块

    tp_size = 4  # TP 度：共 4 个进程
    mp.set_start_method("spawn", force=True)  # spawn 启动方式（CUDA 子进程必需）
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # 分布式初始化的主节点地址（本机回环）
    os.environ["MASTER_PORT"] = "12355"  # 主节点端口
    p_list = []  # 子进程列表
    for i in range(tp_size):  # 为每个 rank 创建进程
        p = mp.Process(target=run, args=(tp_size, i))  # 进程目标：run(tp_size, rank)
        p_list.append(p)  # 收集
    try:
        for p in p_list:  # 先全部启动
            p.start()
        for p in p_list:  # 再全部等待结束
            p.join()
    except BaseException:  # 任意异常（含 Ctrl+C）时
        for p in p_list:  # 终止所有子进程
            p.terminate()
        raise  # 并重新抛出异常
