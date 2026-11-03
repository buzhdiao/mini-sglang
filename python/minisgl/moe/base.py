from abc import ABC, abstractmethod  # 抽象基类支持：强制子类实现 forward

import torch  # PyTorch：张量类型注解（本文件只有抽象签名，无实际计算）


class BaseMoeBackend(ABC):
    """MoE 后端抽象基类：统一不同实现（目前仅 fused Triton 版）的调用接口。

    后端负责"路由得分 → top-k 选择 → 专家计算 → 加权合并"的完整流程。
    """

    @abstractmethod  # 抽象方法：子类（如 FusedMoe）必须实现，否则实例化报错
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入 token 表示，形状 [num_tokens, hidden_size]
        w1: torch.Tensor,  # 第一层专家权重栈（gate_up），形状 [num_experts, 2*inter, hidden]
        w2: torch.Tensor,  # 第二层专家权重栈（down），形状 [num_experts, hidden, inter]
        gating_output: torch.Tensor,  # 路由器 logits，形状 [num_tokens, num_experts]
        topk: int,  # 每 token 激活的专家数 k
        renormalize: bool,  # 是否对 top-k 路由权重重新归一化（和为 1）
        activation: str,  # 激活函数名（"silu"/"gelu"），作用于 gate_up 输出
        apply_router_weight_on_input: bool,  # True：路由权重乘在第一层输入上；False：乘在第二层输出上
    ) -> torch.Tensor: ...  # 返回与输入同形状的输出 [num_tokens, hidden_size]
