# 抽象基类支持
from abc import ABC, abstractmethod

import torch


class BaseMoeBackend(ABC):
    # MoE 后端抽象基类：封装"路由选择 -> 专家分组 GEMM -> 加权合并"的完整流程
    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        activation: str,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor: ...
    # 参数说明：
    # hidden_states: (token 数, 隐藏维) 的输入
    # w1/w2: 堆叠的专家权重 (专家数, 2*中间维, 隐藏维) 与 (专家数, 隐藏维, 中间维)
    # gating_output: 每 token 的路由 logits (token 数, 专家数)
    # topk: 每个激活的专家数; renormalize: 是否重归一化路由权重
    # activation: 门控激活; apply_router_weight_on_input: 路由权重乘在输入侧还是输出侧
