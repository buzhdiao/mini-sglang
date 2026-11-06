# 延迟类型注解：注解在运行时不求值
from __future__ import annotations

from typing import TYPE_CHECKING

# 复用各基础层组装通用模块（多种模型共享）
from minisgl.layers import (
    AttentionLayer,
    BaseOP,
    LinearColParallelMerged,
    LinearOProj,
    LinearQKVMerged,
    LinearReplicated,
    LinearRowParallel,
    MoELayer,
    RMSNorm,
    gelu_and_mul,
    silu_and_mul,
)
# 模型配置类型（仅作注解）
from minisgl.models import ModelConfig
# NVTX 性能打点
from minisgl.utils import nvtx_annotate

if TYPE_CHECKING:
    import torch


class GatedMLP(BaseOP):
    # 通用门控 FFN（SwiGLU/GLU）：gate+up 合并投影 -> 激活门控 -> down 投影
    def __init__(self, config: ModelConfig):
        # 列并行的合并投影：一次 GEMM 同时算出 gate 与 up 两份中间结果，
        # 列并行前向无需通信
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            has_bias=False,
        )

        # 按配置选择门控激活函数
        FN_MAP = {"silu": silu_and_mul, "gelu": gelu_and_mul}
        act_fn = FN_MAP.get(config.hidden_act, None)
        if act_fn is None:
            raise ValueError(f"Unsupported activation function: {config.hidden_act}")
        self.act_fn = act_fn
        # 行并行降维投影：按中间维切分输入，输出部分和经 all_reduce 汇总
        self.down_proj = LinearRowParallel(
            config.intermediate_size,
            config.hidden_size,
            has_bias=False,
        )

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 升维：一次 GEMM 得到 [gate|up] 拼接
        gate_up = self.gate_up_proj.forward(x)
        # 及时释放输入引用，降低显存峰值
        del x
        # 融合门控激活：silu(gate)*up（或 gelu 版本）
        y = self.act_fn(gate_up)
        # 释放中间结果
        del gate_up
        # 降维回隐藏空间（行并行含 all_reduce）
        return self.down_proj.forward(y)


class MoEMLP(BaseOP):
    # MoE 版 FFN：路由门控（每个专家一个 logit）+ 稀疏专家计算
    def __init__(self, config: ModelConfig):
        # 专家层：持有全部专家的 w1/w2 权重分片
        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
        )
        # 路由门控：hidden -> 每专家一个分数；权重极小且所有 rank 需一致路由，
        # 故用复制式（不做 TP 切分）
        self.gate = LinearReplicated(
            config.hidden_size,
            config.num_experts,
            has_bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 记录原始形状，MoE 内核要求输入是二维
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # 计算路由 logits（每 token 一份专家分数分布）
        router_logits = self.gate.forward(hidden_states)
        # 专家计算：内核取 top-k 专家、分组 GEMM、按路由权重合并
        final_hidden_states = self.experts.forward(
            hidden_states=hidden_states, router_logits=router_logits
        )
        # 恢复原始形状
        final_hidden_states = final_hidden_states.view(num_tokens, hidden_dim)
        return final_hidden_states


class RopeAttn(BaseOP):
    # 通用"带 RoPE 的自注意力"组装：QKV 投影（可带 QK-Norm/bias）+ 注意力 + O 投影
    def __init__(
        self,
        config: ModelConfig,
        layer_id: int,
        *,
        has_attn_bias: bool = False,
        has_qk_norm: bool = False,
    ):
        # 单头维度
        head_dim = config.head_dim
        # 合并的 QKV 投影（列并行、按头切分）；has_attn_bias 控制 QKV 偏置（Qwen2 有）
        self.qkv_proj = LinearQKVMerged(
            hidden_size=config.hidden_size,
            head_dim=config.head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            has_bias=has_attn_bias,
        )
        # 是否启用 QK-Norm
        self.has_qk_norm = has_qk_norm
        if has_qk_norm:
            # Q/K 按头 RMSNorm（Qwen3 使用），稳定注意力 logits 的量级
            self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)
        else:
            # Llama/Qwen2 无 QK-Norm
            self.q_norm = None
            self.k_norm = None
        # 注意力层：RoPE 切分、KV cache 读写与后端注意力计算
        self.attn = AttentionLayer(
            layer_id=layer_id,
            head_dim=head_dim,
            num_qo_heads=config.num_qo_heads,
            num_kv_heads=config.num_kv_heads,
            rotary_config=config.rotary_config,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
        )
        # 输出投影（行并行）：把多头拼接结果映回隐藏维
        self.o_proj = LinearOProj(
            head_dim * config.num_qo_heads,
            config.hidden_size,
            has_bias=False,
        )

    @nvtx_annotate("MHA")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 合并 QKV 投影，一次 GEMM 出三份结果
        qkv = self.qkv_proj.forward(x)
        # 释放输入引用
        del x
        # 注意力计算（切分/RoPE/KV cache/后端调用）
        o = self.attn.forward(qkv)
        # 输出投影回隐藏维（行并行含 all_reduce）
        return self.o_proj.forward(o)


__all__ = ["GatedMLP", "RopeAttn", "MoEMLP"]
