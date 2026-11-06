# 延迟类型注解：注解在运行时不求值
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
# 全局上下文：提供 batch 信息与注意力后端
from minisgl.core import get_global_ctx
# TP 并行信息：用于切分注意力头
from minisgl.distributed import get_tp_info
# 整除工具：把头数均分到各 TP rank
from minisgl.utils import div_even

from .base import StateLessOP
# RoPE 构造函数
from .rotary import get_rope

if TYPE_CHECKING:
    from minisgl.layers import RMSNorm
    from minisgl.models import RotaryConfig


class AttentionLayer(StateLessOP):
    # 注意力层（无参数算子）：权重 QKV/OProj 由外部 Linear 持有，
    # 这里只负责 QKV 切分、Q/K 归一化、RoPE 与调用注意力后端
    def __init__(
        self,
        layer_id: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        rotary_config: RotaryConfig,
        q_norm: RMSNorm | None = None,
        k_norm: RMSNorm | None = None,
    ):
        # GQA 约束：query 组数必须是 KV 头数的整数倍（每组共享一个 KV 头）
        assert num_qo_heads % num_kv_heads == 0
        # 记录层号：注意力后端用它索引该层的 KV cache
        self.layer_id = layer_id
        # 单头维度
        self.head_dim = head_dim
        # 取 TP rank 数
        tp_size = get_tp_info().size
        # 把 Q 头均分到本 rank（除不尽则报错）
        self.num_qo_heads = div_even(num_qo_heads, tp_size)
        # 把 KV 头均分到本 rank；KV 头太少时允许跨 rank 复制
        self.num_kv_heads = div_even(num_kv_heads, tp_size, allow_replicate=True)
        # 本 rank 的 Q 总维度（头数×头维），用于从 QKV 拼接张量里切分
        self.qo_attn_dim = self.num_qo_heads * head_dim
        # 本 rank 的 K/V 总维度
        self.kv_attn_dim = self.num_kv_heads * head_dim
        # 构造本层的 RoPE：位置编码的旋转矩阵按 (头维, 旋转维, 基数, 缩放) 参数化
        self.rotary = get_rope(
            head_dim=head_dim,
            rotary_dim=rotary_config.rotary_dim,
            max_position=rotary_config.max_position,
            base=rotary_config.base,
            rope_scaling=tuple(rotary_config.scaling.items()) if rotary_config.scaling else None,
        )
        # 可选的 Q/K 归一化（QK-Norm，如 Qwen3 使用，稳定注意力数值）
        self.q_norm = q_norm
        self.k_norm = k_norm

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        # 取全局上下文：batch（含每 token 的位置）与注意力后端
        ctx = get_global_ctx()
        # 把 QKV 投影结果按最后一维切成 Q / K / V 三段
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)
        if self.q_norm is not None:
            # 对 Q 按头归一化：reshape 成 (token, 头, 头维) 后原地计算
            self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))
        if self.k_norm is not None:
            # 对 K 按头归一化（原地）
            self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))
        # 应用 RoPE：按每个 token 的绝对位置旋转 Q/K，让注意力感知相对位置
        q, k = self.rotary.forward(ctx.batch.positions, q, k)
        # 把 Q 恢复成 (token, 头, 头维) 供注意力后端使用
        q = q.view(-1, self.num_qo_heads, self.head_dim)
        # 调用注意力后端（如 FlashAttention）：内部会读写本层 KV cache，
        # 完成"追加本批 KV + 对上下文做注意力"两步
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)
        # 输出摊平成 (token, q 维)，交给后续的 O 投影
        return o.view(-1, self.qo_attn_dim)
