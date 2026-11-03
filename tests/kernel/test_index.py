from __future__ import annotations  # 启用 PEP 563 延迟注解求值
from typing import Tuple  # 导入 Tuple 类型注解
import torch  # 导入 torch，张量构造与 GPU 操作
import torch.nn.functional as F  # 导入 F，用 F.embedding 作为参考基线实现

from minisgl.benchmark.perf import compare_memory_kernel_perf  # 导入带宽对比压测工具
from minisgl.kernel import indexing  # 导入被测的自定义 embedding 索引 kernel
from minisgl.utils import call_if_main, init_logger  # 导入"仅主模块执行"装饰器与日志工具

logger = init_logger(__name__)  # 创建本模块 logger


def ref_indexing(
    weights: torch.Tensor,  # 词嵌入权重矩阵 [vocab, embed]
    indices: torch.Tensor,  # token 索引 [bs]
    *,
    vocab_range: Tuple[int, int] | None = None,  # (start, length)  # 本 rank 负责的词表区间（TP 切分词表场景）
) -> torch.Tensor:
    """参考实现：用 F.embedding 查表；支持 vocab_range 模式——越界（不属于本分片）的索引输出 0 向量。"""
    if vocab_range is not None:  # 启用词表区间（掩码）模式
        start, length = vocab_range  # 区间起点与长度
        assert length <= weights.shape[0]  # 区间长度不能超过权重行数
        indices = indices - start  # 全局索引平移为区间局部索引
        indices_mask = (indices < 0) | (indices >= length)  # 标记越界索引（不在本分片内）
        indices[indices_mask] = 0  # set out-of-vocab indices to zero  # 越界索引临时置 0（避免查表越界）
        result = F.embedding(indices, weights)  # 常规查表
        result[indices_mask] = 0  # 越界位置的结果行清零（模拟"不属于本 rank"）
        return result  # 返回带掩码的结果
    else:  # 无区间限制
        return F.embedding(indices, weights)  # 直接常规查表


@call_if_main(__name__)  # 装饰器：仅本文件作为主模块运行时执行（被 import 时不执行）
def test_indexing():
    """正确性 + 性能测试（无掩码）：batch size 从 1 到 2^15 逐档，与 F.embedding 对比。"""
    EMBED_SIZE = 4096  # 嵌入向量维度
    NUM_TOKENS = 131072  # 词表大小（2^17）
    stream = torch.cuda.Stream()  # 创建独立 CUDA stream
    torch.cuda.set_stream(stream)  # 切换到该 stream（隔离计时）
    weights = torch.randn((NUM_TOKENS, EMBED_SIZE), device="cuda", dtype=torch.float16)  # 随机初始化权重矩阵

    for bs in [2**n for n in range(0, 16)]:  # batch size = 1, 2, 4, ..., 32768 逐档测试
        indices = torch.randint(0, NUM_TOKENS, (bs,), device="cuda", dtype=torch.int32)  # 随机生成合法索引

        # first test the correctness  # 先测正确性
        result = indexing(  # 自定义 kernel 的输出
            weights,
            indices,
        )
        expected = ref_indexing(  # 参考实现的输出
            weights,
            indices,
        )
        assert torch.all(result == expected), f"Mismatch for BS={bs}"  # 逐元素完全一致才通过

        # test the perf  # 再测性能（带宽对比）
        MEM = bs * EMBED_SIZE * weights.element_size()  # 读取数据量 = bs 行 × 维度 × 2 字节
        compare_memory_kernel_perf(
            our_impl=lambda: indexing(weights, indices),  # 自家实现
            baseline=lambda: ref_indexing(weights, indices),  # 基线 F.embedding
            memory_footprint=MEM,  # 数据量用于换算带宽
            description=f"BS={bs:6d} | ",  # 日志前缀标注当前 batch size
        )


@call_if_main(__name__)  # 装饰器：仅主模块运行时执行
def test_indexing_with_mask():
    """带 vocab_range 掩码的索引测试：模拟 TP=4 切分词表后本 rank 只持有 1/4 区间，其余索引输出 0 向量。"""
    EMBED_SIZE = 4096  # 嵌入维度
    NUM_TOKENS = 131072  # 全局词表大小
    TP = 4  # 模拟的张量并行度
    stream = torch.cuda.Stream()  # 独立 stream
    torch.cuda.set_stream(stream)  # 切换 stream
    weights = torch.randn((NUM_TOKENS, EMBED_SIZE), device="cuda", dtype=torch.float16)  # 完整权重（真实场景本 rank 只持有分片，这里用全量对比正确性）

    assert TP > 1  # 掩码测试需要 TP > 1 才有意义
    MASK_LENGTH = NUM_TOKENS // TP  # 本 rank 持有的词表长度（1/4）
    MASK_RANGE = (MASK_LENGTH, MASK_LENGTH)  # start, length  # 区间：起点与长度均为 1/4 词表（即第 2 个分片）

    for bs in [2**n for n in range(0, 16)]:  # 各 batch size 档位
        indices = torch.randint(0, NUM_TOKENS, (bs,), device="cuda", dtype=torch.int32)  # 索引覆盖全词表（包含本分片之外的部分）

        # first test the correctness  # 先测正确性
        result = indexing(  # 自定义 kernel（带掩码）
            weights,
            indices,
            vocab_range=MASK_RANGE,
        )
        expected = ref_indexing(  # 参考实现（带掩码）
            weights,
            indices,
            vocab_range=MASK_RANGE,
        )
        assert torch.all(result == expected), f"Mismatch for BS={bs}"  # 一致才通过

        # test the perf  # 性能对比
        MEM = bs * EMBED_SIZE * weights.element_size()  # 数据量
        compare_memory_kernel_perf(
            our_impl=lambda: indexing(weights, indices),  # 自家实现
            baseline=lambda: ref_indexing(weights, indices),  # 基线
            memory_footprint=MEM,
            description=f"BS={bs:6d} | ",
            extra_kwargs={"init_stream": False},  # 不另建 stream：沿用外层已设置的 stream（计时在同一流上）
        )
