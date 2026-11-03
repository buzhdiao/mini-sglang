from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解中使用前向引用（如 RMSNorm | None）

from typing import TYPE_CHECKING  # 类型检查开关：运行时为 False，用于仅静态检查的 import

import torch  # PyTorch 主库：张量运算（split/view）
from minisgl.core import get_global_ctx  # 全局上下文：forward 时获取当前 batch（positions）与 attention 后端
from minisgl.distributed import get_tp_info  # TP 进组信息：确定当前 rank 分到的头数
from minisgl.utils import div_even  # 偶数切分工具：头数除以 tp_size，可选允许跨 rank 复制

from .base import StateLessOP  # 无状态算子基类：本层自身不持有可加载权重（q/k/v 投影由外部 LinearQKVMerged 完成）
from .rotary import get_rope  # RoPE 工厂：按配置创建（并缓存）旋转位置编码模块

if TYPE_CHECKING:  # 仅类型检查时导入，避免运行时循环依赖
    from minisgl.layers import RMSNorm  # q/k 归一化层类型注解（如 Qwen3 的 per-head norm）
    from minisgl.models import RotaryConfig  # RoPE 配置类型注解


class AttentionLayer(StateLessOP):
    """注意力层封装：把 QKV 合并投影拆成 q/k/v，做（可选）q/k 归一化与 RoPE，再交给 attention 后端。

    自身无权重（StateLessOP）：QKV 投影在模型层外用 LinearQKVMerged 完成，
    本层只负责"拆分 -> 归一化 -> 位置编码 -> 调后端"的流程编排。
    """

    def __init__(
        self,
        layer_id: int,  # transformer 层编号：attention 后端按它定位该层的 KV Cache 页表
        num_qo_heads: int,  # 全局的 query 头数（qo = query only，GQA/MQA 中不含 KV 头）
        num_kv_heads: int,  # 全局的 key/value 头数（GQA 时 < qo 头数，MQA 时为 1）
        head_dim: int,  # 每个头的维度
        rotary_config: RotaryConfig,  # RoPE 配置（rotary_dim / base / max_position / scaling）
        q_norm: RMSNorm | None = None,  # 可选的 query 每头归一化（如 Qwen3）
        k_norm: RMSNorm | None = None,  # 可选的 key 每头归一化
    ):
        assert num_qo_heads % num_kv_heads == 0  # GQA 约束：每组 KV 头被固定数量的 Q 头共享，必须整除
        self.layer_id = layer_id  # 保存层编号，forward 时传给 attention 后端
        self.head_dim = head_dim  # 保存每头维度
        tp_size = get_tp_info().size  # TP 并行度
        self.num_qo_heads = div_even(num_qo_heads, tp_size)  # Q 头按 rank 均分（必须整除，否则报错）
        self.num_kv_heads = div_even(num_kv_heads, tp_size, allow_replicate=True)  # KV 头均分；不足时允许跨 rank 复制（如 MQA 单头时每 rank 持副本）
        self.qo_attn_dim = self.num_qo_heads * head_dim  # 本 rank 的 q 总维度，用于从 QKV 拼接中切分
        self.kv_attn_dim = self.num_kv_heads * head_dim  # 本 rank 的 k（或 v）总维度
        self.rotary = get_rope(
            head_dim=head_dim,  # 每头维度
            rotary_dim=rotary_config.rotary_dim,  # 参与旋转的维度（可以小于 head_dim）
            max_position=rotary_config.max_position,  # 最大位置：决定 cos/sin 缓存表长度
            base=rotary_config.base,  # 频率基数 theta
            rope_scaling=tuple(rotary_config.scaling.items()) if rotary_config.scaling else None,  # 缩放配置转 tuple 以便被 @functools.cache 哈希缓存
        )
        self.q_norm = q_norm  # 保存可选 q 归一化层（None 表示不启用）
        self.k_norm = k_norm  # 保存可选 k 归一化层

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        """前向：QKV 拼接 [tokens, qo_dim+2*kv_dim] -> 注意力输出 [tokens, qo_dim]。"""
        ctx = get_global_ctx()  # 取全局上下文：拿到当前 batch（含 positions）与 attention 后端
        q, k, v = qkv.split([self.qo_attn_dim, self.kv_attn_dim, self.kv_attn_dim], dim=-1)  # 沿最后一维切成 q/k/v 三段（视图切分，不拷贝）
        if self.q_norm is not None:  # 启用了 q 归一化（如 Qwen3）
            self.q_norm.forward_inplace(q.view(-1, self.num_qo_heads, self.head_dim))  # 变形为 [tokens, 头数, 头维] 后原地归一化（结果写回 q 的存储）
        if self.k_norm is not None:  # 启用了 k 归一化
            self.k_norm.forward_inplace(k.view(-1, self.num_kv_heads, self.head_dim))  # 同上，对 k 每头原地归一化
        q, k = self.rotary.forward(ctx.batch.positions, q, k)  # 对 q/k 原地施加旋转位置编码（v 不需要位置信息）
        q = q.view(-1, self.num_qo_heads, self.head_dim)  # q 变形为 [tokens, 头数, 头维]：attention 后端要求的三维布局
        o = ctx.attn_backend.forward(q, k, v, self.layer_id, ctx.batch)  # 调用 attention 后端（FlashInfer/PagedAttention）：按 KV Cache 页表做自注意力，输出 [tokens, qo_dim]
        return o.view(-1, self.qo_attn_dim)  # 把后端输出整理为 [tokens, 本rank的q维度]，交由后续 o_proj
