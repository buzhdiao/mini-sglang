# 延迟类型注解：注解在运行时不求值
from __future__ import annotations

from typing import List

import torch
# F.linear 完成实际的矩阵乘
import torch.nn.functional as F
# TP 通信器与并行信息：行并行需要 all_reduce
from minisgl.distributed import DistributedCommunicator, get_tp_info
# 整除工具：把维度均分到各 TP rank
from minisgl.utils import div_even

from .base import BaseOP


class _LinearTPImpl(BaseOP):
    """Real implementation of a linear layer with tensor parallelism."""

    # TP 线性层的公共实现：按"全局/本地"输入输出维分别记录形状并分配权重分片
    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        has_bias: bool,
    ):
        # 全局输入维（完整模型的）
        self.full_input_size = full_isize
        # 全局输出维
        self.full_output_size = full_osize
        # 本 rank 的输入分片维
        self.local_input_size = local_isize
        # 本 rank 的输出分片维
        self.local_output_size = local_osize
        # 权重按本地分片形状分配：(本地输出, 本地输入)
        self.weight = torch.empty(local_osize, local_isize)
        # 偏置只按输出维分配
        self.bias = torch.empty(local_osize) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 默认前向：本地矩阵乘，不做通信（列并行/复制式的语义）
        return F.linear(x, self.weight, self.bias)


class LinearReplicated(_LinearTPImpl):
    """
    Linear layer where weights are replicated (not sharded) across all TP ranks.
    Each GPU holds the full weight matrix.
    """

    # 复制式线性层：每张卡持有完整权重，不做任何切分与通信（如 MoE 的路由门控）
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        super().__init__(
            full_isize=input_size,
            full_osize=output_size,
            local_isize=input_size,
            local_osize=output_size,
            has_bias=has_bias,
        )


class LinearColParallelMerged(_LinearTPImpl):
    # 列并行（按输出维切分）：每个 rank 拿到输出的一个分片，输入各 rank 相同，
    # 前向只需本地 GEMM 无需通信。支持把多个输出合并成一次 GEMM（如 gate+up 投影）
    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        has_bias: bool,
    ):
        # check that all output sizes are divisible by tp_size
        # 各输出段分别均分到各 rank（除不尽直接报错）
        tp_info = get_tp_info()
        tp_output_sizes = [div_even(size, tp_info.size) for size in output_sizes]
        # 全局输出维 = 各段之和
        output_size = sum(output_sizes)
        # 本地输出维 = 各段分片之和
        tp_output_size = sum(tp_output_sizes)
        super().__init__(input_size, output_size, input_size, tp_output_size, has_bias)


class LinearQKVMerged(_LinearTPImpl):
    # 合并的 QKV 投影（列并行特化）：把 Q、K、V 三个投影合成一次 GEMM，
    # 按"每头"粒度切分到各 rank，避免三次小矩阵乘的开销
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()

        # 本 rank 分到的 Q 头数
        local_num_qo = div_even(num_qo_heads, tp_info.size)
        # 本 rank 分到的 KV 头数（GQA 下 KV 头少，允许跨 rank 复制）
        local_num_kv = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        # 全局输入维 = 隐藏维
        full_isize = hidden_size
        # 全局输出维 = Q + K + V 全部头维之和
        full_osize = (num_qo_heads + 2 * num_kv_heads) * head_dim
        # 输入不做切分（列并行每 rank 拿全量输入）
        local_isize = hidden_size
        # 本地输出维 = 本地 Q/K/V 头维之和
        local_osize = (local_num_qo + 2 * local_num_kv) * head_dim
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)


class LinearOProj(_LinearTPImpl):
    # 注意力输出投影（行并行特化）：注意力输出按头维已切在各 rank 上，
    # 本层按输入维切分权重，各 rank 算出部分和后再 all_reduce 求和
    def __init__(self, input_size: int, output_size: int, has_bias: bool):
        tp_info = get_tp_info()
        # 全局输入维 = 所有头的总维度
        full_isize = input_size
        # 全局输出维 = 隐藏维
        full_osize = output_size
        # 本 rank 的输入分片（对应本 rank 的 Q 头维度）
        local_isize = div_even(input_size, tp_info.size)
        # 行并行下输出不切分，每 rank 都算出完整维度的部分和
        local_osize = output_size
        # 行并行需要通信器
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 本地部分和计算
        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            # 行并行的核心：all_reduce 把各 rank 的部分和累加成完整输出
            y = self._comm.all_reduce(y)
        return y


class LinearRowParallel(_LinearTPImpl):
    # 通用行并行线性层：按输入维切分权重（如 MLP 的 down_proj），
    # 输入需要上游按相同方式切分，输出经 all_reduce 求和
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()
        # 本 rank 的输入分片维
        local_input_size = div_even(input_size, tp_info.size)
        # 输出维不切分
        local_output_size = output_size
        # 行并行需要通信器
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(input_size, output_size, local_input_size, local_output_size, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 本地部分和计算
        y = F.linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            # all_reduce 累加各 rank 部分和，还原完整输出
            y = self._comm.all_reduce(y)
        return y
