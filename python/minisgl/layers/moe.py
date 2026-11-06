import torch
# 全局上下文：提供 MoE 后端
from minisgl.core import get_global_ctx
# TP 通信器与并行信息：MoE 专家按中间维切分，输出需要 all_reduce
from minisgl.distributed import DistributedCommunicator, get_tp_info
# 整除工具：把专家中间维均分到各 rank
from minisgl.utils import div_even

from .base import BaseOP


class MoELayer(BaseOP):
    # MoE 层：持有全部专家的权重分片，路由 softmax 已由上游算出（router_logits），
    # 专家计算交给 moe_backend（fused MoE kernel）
    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        renormalize: bool = True,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
    ):
        super().__init__()

        # 专家总数
        self.num_experts = num_experts
        # 每个 token 激活的专家数
        self.top_k = top_k
        # 隐藏维
        self.hidden_size = hidden_size
        # 单专家 FFN 中间维
        self.intermediate_size = intermediate_size
        # TP 通信器
        self._comm = DistributedCommunicator()

        # 取 TP 规模
        tp_info = get_tp_info()
        self.tp_size = tp_size = tp_info.size
        # 是否对 top-k 路由权重重新归一化（softmax 后再按 top-k 归一）
        self.renormalize = renormalize
        # 专家 FFN 激活函数类型
        self.activation = activation
        # 为 true 时路由权重乘在专家输入侧（部分模型如 GLM 的约定），否则乘在输出侧
        self.apply_router_weight_on_input = apply_router_weight_on_input
        # 每个专家的中间维按 TP 切分（专家内并行，类似 MLP 行/列并行）
        intermediate_size_per_partition = div_even(intermediate_size, tp_size)
        # w1 权重：每专家 (2*本地中间维, 隐藏维)，2 倍是 gate+up 两份（SwiGLU 结构）
        self.gate_up_proj = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
        )
        # w2 权重：每专家 (隐藏维, 本地中间维)，降维回隐藏空间
        self.down_proj = torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
        )

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        # 取全局上下文以访问 MoE 后端
        ctx = get_global_ctx()
        # 调用融合 MoE 内核：完成路由 top-k 选择、按专家分组的 token 排列、
        # 两次 GEMM（w1 升维+激活+门控、w2 降维）与路由加权合并
        final_hidden_states = ctx.moe_backend.forward(
            hidden_states=hidden_states,
            w1=self.gate_up_proj,
            w2=self.down_proj,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
        )
        if self.tp_size > 1:
            # 专家中间维被切分：每个 rank 只算了一部分中间通道，
            # all_reduce 把各 rank 的部分输出累加成完整 MoE 结果
            final_hidden_states = self._comm.all_reduce(final_hidden_states)
        return final_hidden_states
