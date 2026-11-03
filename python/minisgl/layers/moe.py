import torch  # PyTorch 主库：MoE 专家权重张量存储
from minisgl.core import get_global_ctx  # 全局上下文：forward 时通过它拿到 moe_backend
from minisgl.distributed import DistributedCommunicator, get_tp_info  # TP 通信器与进程组信息
from minisgl.utils import div_even  # 偶数切分工具：专家中间维按 rank 均分

from .base import BaseOP  # 算子基类：提供 state_dict/load_state_dict 协议


class MoELayer(BaseOP):
    """混合专家层（Mixture of Experts）：路由器选 top-k 专家，仅激活被选专家的 FFN 权重。

    权重按专家堆叠为三维张量 [num_experts, ...]，TP 时沿专家中间维（FFN 隐层）切分，
    每 rank 持有每个专家的一个分片；具体路由/分组/GEMM 由 moe_backend 的融合 kernel 完成。
    """

    def __init__(
        self,
        num_experts: int,  # 专家总数
        top_k: int,  # 每 token 激活的专家数
        hidden_size: int,  # 隐藏维度（专家 FFN 的输入/输出维度）
        intermediate_size: int,  # 全局 FFN 中间维（TP 时会被切分）
        renormalize: bool = True,  # 是否对 top-k 路由权重做 renormalize（和归一）
        activation: str = "silu",  # 专家 FFN 的激活函数名（由后端 kernel 解释）
        apply_router_weight_on_input: bool = False,  # True: 路由权重乘在输入上；False: 乘在专家输出合并时
    ):
        super().__init__()  # 调用基类构造

        self.num_experts = num_experts  # 保存专家数
        self.top_k = top_k  # 保存每 token 激活的专家数
        self.hidden_size = hidden_size  # 保存隐藏维度
        self.intermediate_size = intermediate_size  # 保存全局中间维
        self._comm = DistributedCommunicator()  # TP 通信器：forward 中做 all_reduce（下划线开头不进 state_dict）

        tp_info = get_tp_info()  # 获取 TP 进程组信息
        self.tp_size = tp_size = tp_info.size  # 缓存 TP 并行度（局部变量 tp_size 供下方切分使用）
        self.renormalize = renormalize  # 保存路由权重归一化开关，透传给后端
        self.activation = activation  # 保存激活函数名，透传给后端
        self.apply_router_weight_on_input = apply_router_weight_on_input  # 保存路由权重施加位置开关
        intermediate_size_per_partition = div_even(intermediate_size, tp_size)  # 每 rank 的专家中间维 = 中间维均分（专家维不切、FFN 隐层切）
        self.gate_up_proj = torch.empty(
            num_experts,  # 第一维按专家堆叠
            2 * intermediate_size_per_partition,  # gate 与 up 合并输出（SwiGLU 双份中间维），已按 TP 切分
            hidden_size,  # 输入为完整 hidden_size（等价列并行，本地乘法无需通信）
        )
        self.down_proj = torch.empty(
            num_experts,  # 按专家堆叠
            hidden_size,  # 输出回投影到完整 hidden_size
            intermediate_size_per_partition,  # 输入是切分后的中间维（等价行并行，合并后需 all_reduce）
        )

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        """前向：hidden [tokens, hidden] + 路由得分 [tokens, num_experts] -> 混合专家输出 [tokens, hidden]。"""
        ctx = get_global_ctx()  # 取全局上下文中的 MoE 后端
        final_hidden_states = ctx.moe_backend.forward(
            hidden_states=hidden_states,  # 输入 hidden，形状 [tokens, hidden_size]
            w1=self.gate_up_proj,  # gate/up 合并权重 [E, 2*ffn_tp, hidden]
            w2=self.down_proj,  # down 投影权重 [E, hidden, ffn_tp]
            gating_output=router_logits,  # 路由器输出的每专家得分
            topk=self.top_k,  # 每 token 选 top-k 专家
            renormalize=self.renormalize,  # 是否归一化 top-k 权重
            activation=self.activation,  # 激活函数（如 silu，配合 gate/up 使用）
            apply_router_weight_on_input=self.apply_router_weight_on_input,  # 路由权重施加位置
        )
        if self.tp_size > 1:  # 多卡时 down_proj 的行并行部分和需要跨 rank 汇总
            final_hidden_states = self._comm.all_reduce(final_hidden_states)  # all_reduce 求和得到完整专家混合输出
        return final_hidden_states  # 返回 [tokens, hidden_size]
