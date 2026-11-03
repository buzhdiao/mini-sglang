from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解中使用前向引用

from typing import List  # 类型注解：List 用于标注多输出尺寸列表

import torch  # PyTorch 主库：张量存储
import torch.nn.functional as F  # 函数式接口：F.linear 做实际矩阵乘
from minisgl.distributed import DistributedCommunicator, get_tp_info  # TP 通信器与进程组信息
from minisgl.utils import div_even  # 偶数切分工具：尺寸除以 tp_size，可选允许复制

from .base import BaseOP  # 算子基类：提供 state_dict/load_state_dict 协议


class _LinearTPImpl(BaseOP):
    """张量并行线性层的统一实现：权重按 [local_osize, local_isize] 分片存储，乘法本地完成。"""

    def __init__(
        self,
        full_isize: int,  # 全局输入维度（切分前的完整值）
        full_osize: int,  # 全局输出维度
        local_isize: int,  # 本 rank 持有的输入维度（行并行时 = full/tp，列并行时 = full）
        local_osize: int,  # 本 rank 持有的输出维度（列并行时 = full/tp，行并行时 = full）
        has_bias: bool,  # 是否带偏置（偏置按输出维度分片）
    ):
        self.full_input_size = full_isize  # 记录全局形状（供上层/调试参考）
        self.full_output_size = full_osize  # 记录全局输出维度
        self.local_input_size = local_isize  # 记录本地输入维度
        self.local_output_size = local_osize  # 记录本地输出维度
        self.weight = torch.empty(local_osize, local_isize)  # 本 rank 权重分片，形状 [局部输出, 局部输入]（F.linear 的 W 定义：y = xW^T）
        self.bias = torch.empty(local_osize) if has_bias else None  # 偏置形状与局部输出一致；无偏置时为 None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)  # 本地矩阵乘：x [tokens, local_isize] @ W^T -> [tokens, local_osize]（通信由子类按切分方式补充）


class LinearReplicated(_LinearTPImpl):
    """
    Linear layer where weights are replicated (not sharded) across all TP ranks.
    Each GPU holds the full weight matrix.
    """

    def __init__(
        self,
        input_size: int,  # 输入维度
        output_size: int,  # 输出维度
        has_bias: bool,  # 是否带偏置
    ):
        super().__init__(
            full_isize=input_size,  # 全局输入维度
            full_osize=output_size,  # 全局输出维度
            local_isize=input_size,  # 复制模式：本地维度 = 全局维度（不切分）
            local_osize=output_size,  # 同上
            has_bias=has_bias,  # 透传偏置开关
        )


class LinearColParallelMerged(_LinearTPImpl):
    """合并的列并行（按输出维切分）线性层：一次乘法同时算多个子层（如 gate_proj + up_proj）。

    每个输出子层各自除以 tp_size，拼成一个大权重 [局部总输出, input_size]。
    列并行（weight sharding on output dim）各 rank 独立计算，forward 无需通信。
    """

    def __init__(
        self,
        input_size: int,  # 输入维度（列并行不切输入，本地 = 全局）
        output_sizes: List[int],  # 各子层的输出尺寸列表（如 MLP 的 [ffn, ffn] 对应 gate/up）
        has_bias: bool,  # 是否带偏置
    ):
        # check that all output sizes are divisible by tp_size
        tp_info = get_tp_info()  # 获取 TP 进程组信息
        tp_output_sizes = [div_even(size, tp_info.size) for size in output_sizes]  # 每个子层输出均分到各 rank（不整除会在此报错）
        output_size = sum(output_sizes)  # 全局总输出维度 = 各子层尺寸之和
        tp_output_size = sum(tp_output_sizes)  # 本 rank 局部总输出 = 各子层局部尺寸之和
        super().__init__(input_size, output_size, input_size, tp_output_size, has_bias)  # 列并行：输入不切（local_isize=full），输出切分


class LinearQKVMerged(_LinearTPImpl):
    """QKV 合并投影的列并行线性层：把 hidden 一次映射到本 rank 的 q/k/v 三段拼接。

    与 LinearColParallelMerged 的区别：按"头"而非按尺寸切分——Q 头均分，
    KV 头不足时允许跨 rank 复制（GQA/MQA 场景），保证每 rank 头数配比正确。
    """

    def __init__(
        self,
        hidden_size: int,  # 输入隐藏维度
        head_dim: int,  # 每个注意力头的维度
        num_qo_heads: int,  # 全局 Q 头数
        num_kv_heads: int,  # 全局 KV 头数（GQA 时少于 Q 头）
        has_bias: bool,  # 是否带偏置
    ):
        tp_info = get_tp_info()  # 获取 TP 并行度

        local_num_qo = div_even(num_qo_heads, tp_info.size)  # 本 rank 分到的 Q 头数（必须整除）
        local_num_kv = div_even(num_kv_heads, tp_info.size, allow_replicate=True)  # 本 rank 的 KV 头数；不足整除时允许复制副本
        full_isize = hidden_size  # 列并行：全局输入 = 本地输入 = hidden_size
        full_osize = (num_qo_heads + 2 * num_kv_heads) * head_dim  # 全局输出 = (Q头 + 2×KV头) × 头维（k、v 各占 KV头 份）
        local_isize = hidden_size  # 本地输入维度（列并行不切输入）
        local_osize = (local_num_qo + 2 * local_num_kv) * head_dim  # 本地输出 = 本 rank 的 (q + k + v) 拼接维度，与 AttentionLayer 的切分尺寸对应
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)  # 用全局/本地尺寸初始化权重分片


class LinearOProj(_LinearTPImpl):
    """注意力输出投影（o_proj）：行并行（按输入维切分）。

    输入 x 已按头切分到各 rank（attention 输出），本地乘法只得到部分和，
    需要 all_reduce 跨 rank 求和得到完整输出。
    """

    def __init__(self, input_size: int, output_size: int, has_bias: bool):
        tp_info = get_tp_info()  # 获取 TP 进程组信息
        full_isize = input_size  # 全局输入维度（= num_heads * head_dim）
        full_osize = output_size  # 全局输出维度（= hidden_size）
        local_isize = div_even(input_size, tp_info.size)  # 行并行：输入按 rank 均分
        local_osize = output_size  # 行并行：输出不切，每 rank 都是完整尺寸
        self._comm = DistributedCommunicator()  # TP 通信器：forward 中做 all_reduce
        self._tp_size = tp_info.size  # 缓存并行度，forward 判断是否需要通信（下划线开头不进 state_dict）
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)  # 初始化权重分片 [output_size, local_isize]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)  # 本地部分和：x [tokens, local_isize] @ W^T -> [tokens, hidden]（每 rank 只算了一部分）
        if self._tp_size > 1:  # 多卡时才需要跨 rank 汇总
            y = self._comm.all_reduce(y)  # all_reduce 求各 rank 部分和 -> 完整输出（潜在问题：bias 未切分，每 rank 各加一次完整偏置，all_reduce 后被重复累加 tp_size 次，行并行时通常应传 has_bias=False）
        return y  # 返回完整输出 [tokens, hidden]


class LinearRowParallel(_LinearTPImpl):
    """通用行并行（按输入维切分）线性层：与 LinearOProj 同构，语义上用于 MLP 的 down_proj 等。

    forward 中本地矩阵乘 + all_reduce 汇总部分和，是 Megatron 行并行的标准实现。
    """

    def __init__(
        self,
        input_size: int,  # 全局输入维度（将被切分）
        output_size: int,  # 输出维度（不切分）
        has_bias: bool,  # 是否带偏置
    ):
        tp_info = get_tp_info()  # 获取 TP 进程组信息
        local_input_size = div_even(input_size, tp_info.size)  # 输入按 rank 均分
        local_output_size = output_size  # 输出不切，每 rank 完整
        self._comm = DistributedCommunicator()  # TP 通信器：forward 中做 all_reduce
        self._tp_size = tp_info.size  # 缓存并行度用于条件通信
        super().__init__(input_size, output_size, local_input_size, local_output_size, has_bias)  # 初始化权重分片 [output_size, local_isize]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)  # 本地部分和计算
        if self._tp_size > 1:  # 单卡无需通信
            y = self._comm.all_reduce(y)  # 汇总各 rank 部分和得到完整输出（bias 同样会被重复累加 tp_size 次，行并行建议 has_bias=False）
        return y  # 返回 [tokens, output_size]
