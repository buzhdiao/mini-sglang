from __future__ import annotations  # 启用 PEP 563 延迟注解求值

from minisgl.benchmark.perf import compare_memory_kernel_perf  # 导入带宽对比压测工具
import torch  # 导入 torch，张量构造与 GPU 操作
from minisgl.kernel import store_cache  # 导入被测的 KV cache 写入 kernel
from minisgl.utils import call_if_main  # 导入"仅主模块执行"装饰器


@call_if_main(__name__)  # 装饰器：仅本文件作为主模块运行时执行
def test_store_cache():
    """store_cache 正确性与性能测试：随机槽位写入 K/V 后校验，并与 torch.compile 的索引赋值基线比带宽。"""
    HEAD_SIZE = 128  # 每个注意力头的维度
    NUM_TOKENS = 1048576  # 1M  # cache 总槽位数（100 万 token）
    stream = torch.cuda.Stream()  # 创建独立 CUDA stream
    torch.cuda.set_stream(stream)  # 切换到该 stream
    kv_cache = torch.randn((NUM_TOKENS, 2, HEAD_SIZE), device="cuda", dtype=torch.float16)  # KV cache 缓冲（每槽 K/V 各 128 维）
    k_cache = kv_cache[:, 0, :]  # K 部分视图（非连续：行步长跨 2×HEAD_SIZE）
    v_cache = kv_cache[:, 1, :]  # V 部分视图（同样非连续）

    for bs in [2**n for n in range(0, 16)]:  # batch size 从 1 到 32768 逐档测试
        # NOTE: we cannot tolerate duplicate indices in this test  # 注意：本测试不能容忍重复索引（并发写同一槽位会互相覆盖，导致断言不稳定）
        indices = torch.randperm(NUM_TOKENS, device="cuda")[:bs].to(torch.int32)  # 无放回随机抽取 bs 个不重复槽位
        qkv = torch.randn((bs, HEAD_SIZE * 4), device="cuda", dtype=torch.float16)  # 模拟 QKV 投影输出（4×HEAD_SIZE 列）
        k = qkv[:, :HEAD_SIZE]  # 取 K 部分（切片视图）
        v = qkv[:, HEAD_SIZE : HEAD_SIZE * 2]  # 取 V 部分（切片视图）
        store_cache(  # 执行自定义写入 kernel
            k_cache,
            v_cache,
            indices,
            k,
            v,
        )

        assert torch.all(k_cache[indices] == k), bs  # 校验 K 已正确写入对应槽位
        assert torch.all(v_cache[indices] == v), bs  # 校验 V 已正确写入对应槽位

        # 2 = k + v  # 数据量 = bs × HEAD_SIZE × (K+V 两份) × 2 字节
        MEM = bs * HEAD_SIZE * 2 * kv_cache.element_size()

        k = k.contiguous()  # 转为连续内存（保证与基线对比时输入布局一致、公平）
        v = v.contiguous()

        @torch.compile()  # JIT 编译基线函数，尽量消除 Python/调度开销
        def baseline():
            k_cache[indices] = k  # torch 高级索引写 K（scatter 语义）
            v_cache[indices] = v  # torch 高级索引写 V

        compare_memory_kernel_perf(
            our_impl=lambda: store_cache(k_cache, v_cache, indices, k, v),  # 自家 kernel
            baseline=baseline,  # 编译后的 torch 基线
            memory_footprint=MEM,  # 数据量
            description=f"BS={bs:6d} | ",
            extra_kwargs={"init_stream": False},  # 沿用外层已设置的 stream（不另建）
        )
