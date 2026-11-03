from __future__ import annotations  # 启用延迟注解求值，允许类型注解使用前向引用

from typing import TYPE_CHECKING  # 类型检查开关：仅静态分析时导入 torch

from minisgl.layers import (  # 导入本文件用到的所有层组件与算子
    AttentionLayer,  # 注意力层封装：内部集成 paged attention kernel 与 RoPE、QK Norm
    BaseOP,  # 算子基类：所有可组合模块的公共父类
    LinearColParallelMerged,  # 列切分合并线性层：一次矩阵乘算出多个并列投影（如 gate+up、q+k+v），按输出列做 TP 切分
    LinearOProj,  # 注意力输出投影：按输入维度行切分的行并行线性层（特化版）
    LinearQKVMerged,  # QKV 合并投影：一次算出 Q/K/V 三组（GQA 下 K/V 头更少），按输出列切分
    LinearReplicated,  # 不切分线性层：各 TP rank 持有完整权重（如 MoE router，输入输出都小）
    LinearRowParallel,  # 行切分线性层：按输入行做 TP 切分，各 rank 算部分和后 all-reduce（如 MLP down_proj）
    MoELayer,  # MoE 计算层：封装专家权重栈与 fused MoE kernel 调用
    RMSNorm,  # 普通 RMSNorm（不融合残差）：用于 Q/K 逐头归一化
    gelu_and_mul,  # 融合激活：gelu(x[..., :d]) * x[..., d:]（门控后半乘前半）
    silu_and_mul,  # 融合激活：silu(gate) * up，SwiGLU 的核心算子
)
from minisgl.models import ModelConfig  # 模型配置类（从包顶层导入，注意与"from .config"不同以避免循环导入）
from minisgl.utils import nvtx_annotate  # NVTX 性能标注装饰器

if TYPE_CHECKING:  # 仅类型检查时导入 torch（运行时 torch 由 minisgl.layers 间接加载，本文件只做注解）
    import torch


class GatedMLP(BaseOP):
    """门控 MLP（SwiGLU/GeGLU）：LLaMA 系模型的稠密 FFN。

    计算流程：down_proj( act(gate_up_proj(x)) )
    - gate_up_proj: [hidden] → 2×[intermediate]（gate 与 up 两个投影合并成一次 GEMM）
    - act: silu(gate) * up（逐元素门控相乘）
    - down_proj: [intermediate] → [hidden]
    """

    def __init__(self, config: ModelConfig):
        self.gate_up_proj = LinearColParallelMerged(  # 合并列并行：把 gate_proj 与 up_proj 拼成一个大权重一次算完（减少 kernel 启动与显存碎片）
            config.hidden_size,  # 输入维度
            [config.intermediate_size, config.intermediate_size],  # 两个输出块的宽度列表：各为 intermediate_size（合并后总输出 2×intermediate）
            has_bias=False,  # LLaMA 系 MLP 无偏置
        )

        FN_MAP = {"silu": silu_and_mul, "gelu": gelu_and_mul}  # 激活函数名 → 融合算子映射表
        act_fn = FN_MAP.get(config.hidden_act, None)  # 按 config（如 config.json 的 hidden_act）查找对应算子
        if act_fn is None:  # 未注册的激活函数（如 gelu_new）：
            raise ValueError(f"Unsupported activation function: {config.hidden_act}")  # 尽早报错，避免跑到一半才发现不支持
        self.act_fn = act_fn  # 保存激活算子供 forward 使用
        self.down_proj = LinearRowParallel(  # 行并行下投影：输入按行切分在各 rank，输出需 all-reduce 求和
            config.intermediate_size,  # 输入维度（激活后的门控向量宽度）
            config.hidden_size,  # 输出维度（回到 hidden）
            has_bias=False,  # 无偏置
        )

    @nvtx_annotate("MLP")  # 性能标注：profiler 中显示为 "MLP" 区间
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向：[num_tokens, hidden] → 门控升维 → 激活 → 降维回 [num_tokens, hidden]。"""
        gate_up = self.gate_up_proj.forward(x)  # 一次 GEMM 同时得 gate 与 up：[num_tokens, 2×intermediate]
        del x  # 显式释放输入引用，降低显存峰值（推理显存紧张时每一点都有用）
        y = self.act_fn(gate_up)  # 融合激活：silu(gate) * up → [num_tokens, intermediate]
        del gate_up  # 释放升维中间结果（2×intermediate 较大，尽早回收）
        return self.down_proj.forward(y)  # 降维回 hidden 维度，内部含 TP all-reduce


class MoEMLP(BaseOP):
    """混合专家 FFN：router 计算 top-k 专家，token 只流经被选中的专家。

    结构 = gate(路由器线性层) + experts(专家权重栈 + fused MoE kernel)。
    与稠密 MLP 相比参数量大、FLOPs 少（每 token 只激活 top-k 个专家）。
    """

    def __init__(self, config: ModelConfig):
        self.experts = MoELayer(  # 专家层：持有 E 个专家的 gate_up/down 权重栈（形状 [E, ...]）及 fused kernel
            num_experts=config.num_experts,  # 专家总数 E
            top_k=config.num_experts_per_tok,  # 每 token 激活的专家数 k
            hidden_size=config.hidden_size,  # 专家输入/输出维度（= hidden）
            intermediate_size=config.moe_intermediate_size,  # 每个专家 FFN 的中间宽度
            renormalize=config.norm_topk_prob,  # 是否对 top-k 路由权重重新归一化（和为 1）
        )
        self.gate = LinearReplicated(  # 路由器：不切分（各 rank 都有完整权重），输出每专家得分
            config.hidden_size,  # 输入维度（hidden）
            config.num_experts,  # 输出维度（E 个专家各自的 router logit）
            has_bias=False,  # 路由器无偏置
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """前向：[num_tokens, hidden] → 路由 → top-k 专家加权组合 → [num_tokens, hidden]。"""
        num_tokens, hidden_dim = hidden_states.shape  # 记录原始形状（token 数与隐藏维度），后面恢复用
        hidden_states = hidden_states.view(-1, hidden_dim)  # 展平为二维（防御性：容忍多余维度输入）
        router_logits = self.gate.forward(hidden_states)  # 路由打分：[num_tokens, E]，每个专家一个 logit
        final_hidden_states = self.experts.forward(  # fused MoE：按 router_logits 选 top-k 专家并加权计算
            hidden_states=hidden_states, router_logits=router_logits  # 输入 token 表示 [num_tokens, hidden]；路由得分（内部做 topk_softmax）
        )
        final_hidden_states = final_hidden_states.view(num_tokens, hidden_dim)  # 恢复原始形状后返回
        return final_hidden_states  # [num_tokens, hidden]


class RopeAttn(BaseOP):
    """通用 RoPE 自注意力层：QKV 合并投影（+可选 QK Norm）→ RoPE + paged attention → O 投影。

    通过 has_qk_norm / has_attn_bias 两个开关适配不同模型家族：
    - Llama/Mistral: 均关
    - Qwen2: 仅 bias
    - Qwen3: 仅 QK Norm
    """

    def __init__(
        self,
        config: ModelConfig,  # 模型配置
        layer_id: int,  # 层编号：定位该层对应的 KV Cache 池
        *,
        has_attn_bias: bool = False,  # QKV 投影是否带偏置（Qwen2 为 True）
        has_qk_norm: bool = False,  # 是否对 Q/K 做逐头 RMSNorm（Qwen3 为 True）
    ):
        head_dim = config.head_dim  # 每头维度，后续多处使用
        self.qkv_proj = LinearQKVMerged(  # QKV 合并投影：一次 GEMM 同时产出 Q、K、V 三组（省两次 kernel 调用）
            hidden_size=config.hidden_size,  # 输入维度
            head_dim=config.head_dim,  # 每头维度
            num_qo_heads=config.num_qo_heads,  # Q 头数（GQA 下大于 KV 头数）
            num_kv_heads=config.num_kv_heads,  # KV 头数（GQA：多 Q 头共享少量 KV 头，省 KV Cache 显存）
            has_bias=has_attn_bias,  # 偏置开关（模型家族差异）
        )
        self.has_qk_norm = has_qk_norm  # 记录开关（forward 内部不直接用，主要供调试/检查）
        if has_qk_norm:  # Qwen3 风格：Q/K 逐头 RMSNorm，稳定注意力的数值尺度
            self.q_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)  # Q 归一化：在每头维度 head_dim 上做 RMSNorm
            self.k_norm = RMSNorm(head_dim, eps=config.rms_norm_eps)  # K 归一化：同上
        else:
            self.q_norm = None  # 无 QK Norm 时置 None（如 Llama/Mistral/Qwen2）
            self.k_norm = None  # 同上
        self.attn = AttentionLayer(  # 注意力核心：RoPE + paged attention（decode 用 FlashDecoding 类 kernel，prefill 用变长 attention）
            layer_id=layer_id,  # 层编号：读写对应层的 KV Cache
            head_dim=head_dim,  # 每头维度
            num_qo_heads=config.num_qo_heads,  # Q 头数
            num_kv_heads=config.num_kv_heads,  # KV 头数
            rotary_config=config.rotary_config,  # RoPE 配置（theta 基数/最大位置/缩放策略）
            q_norm=self.q_norm,  # 传入 Q Norm（None 则跳过）；注意在 RoPE 之前对 QK 归一化
            k_norm=self.k_norm,  # 传入 K Norm（None 则跳过）
        )
        self.o_proj = LinearOProj(  # 输出投影：把多头拼接结果映射回 hidden 维度
            head_dim * config.num_qo_heads,  # 输入维度 = 每头维度 × Q 头数（多头 concat 后宽度）
            config.hidden_size,  # 输出维度 = hidden
            has_bias=False,  # O 投影无偏置（各家族一致）
        )

    @nvtx_annotate("MHA")  # 性能标注：profiler 中显示为 "MHA" 区间
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向：[num_tokens, hidden] → QKV 投影 → attention → O 投影 → [num_tokens, hidden]。"""
        qkv = self.qkv_proj.forward(x)  # 一次投影得拼接的 QKV：[num_tokens, (q_heads + 2*kv_heads) × head_dim]
        del x  # 尽早释放输入，降低显存峰值
        o = self.attn.forward(qkv)  # RoPE + QK Norm + paged attention：输出各头拼接 [num_tokens, q_heads × head_dim]
        return self.o_proj.forward(o)  # 输出投影回 hidden 维度（行并行，内部含 all-reduce）


__all__ = ["GatedMLP", "RopeAttn", "MoEMLP"]  # 模块对外暴露三个可复用组件：稠密 MLP / 通用注意力 / MoE FFN
