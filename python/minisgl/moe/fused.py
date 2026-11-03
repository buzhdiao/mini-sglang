import functools  # 标准库：partial 用于固定 try_get_optimal_moe_config 的前几个参数（权重形状等）
from typing import Dict, Tuple  # 类型注解：字典与元组

import torch  # PyTorch：张量操作与显存分配
from minisgl.moe import BaseMoeBackend  # MoE 后端抽象基类
from minisgl.utils import div_ceil  # 整数向上取整除法：(a + b - 1) // b，用于算分块数


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_token_non_padded: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """融合 top-k 路由：对 router logits 做 softmax + top-k 选择（CUDA kernel 加速）。

    参数：
    - hidden_states: [M, hidden]，仅用其行数 M（token 数）与设备信息
    - gating_output: [M, num_experts] 路由器输出的 logits
    - topk: 每 token 选几个专家
    - renormalize: 是否把 top-k 权重重新归一化（和为 1）
    - num_token_non_padded: 实际有效 token 数（标量张量）；CUDA Graph 模式下
      batch 按最大长度 padding，超出部分的路由结果要作废（id 置 -1）

    返回：(topk_weights [M, topk] float32, topk_ids [M, topk] int32)
    """
    from sgl_kernel import topk_softmax  # 延迟导入 sgl-kernel 的高性能融合算子（softmax+topk 一次完成）

    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"  # 校验 token 数一致，防止路由错位
    M, _ = hidden_states.shape  # M = 当前 token 数
    topk_weights = torch.empty(M, topk, dtype=torch.float32, device=hidden_states.device)  # 预分配输出：每 token 的 top-k 路由权重（fp32 保证数值精度）
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)  # 预分配输出：每 token 选中的专家 id（int32，供 kernel 索引用）
    topk_softmax(topk_weights, topk_ids, gating_output.float(), renormalize)  # 融合 kernel：logits → softmax → top-k，就地写入两个输出缓冲
    if renormalize:  # kernel 的 renormalize 只在部分版本生效，这里兜底再归一化一次
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)  # 沿专家维求和后相除；+1e-8 防止除零
    if num_token_non_padded is not None:  # CUDA Graph padding 场景：真实 token 之外的行需要作废
        indices = torch.arange(0, topk_ids.shape[0], device=topk_ids.device)  # 生成行号 [0, 1, ..., M-1]
        topk_ids[indices >= num_token_non_padded, :] = -1  # padding 行（行号 >= 有效数）的专家 id 全部置 -1（后续 kernel 跳过）
    return topk_weights, topk_ids  # 返回路由权重与专家 id


def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.
    将 token 在各专家间的分布按块大小对齐，使后续分块矩阵乘维度整齐。

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Parameters（参数说明）:
    - topk_ids: [总 token 数, top_k]，每个 token 的 top-k 专家 id
    - block_size: 分块矩阵乘使用的块大小（对应 Triton kernel 的 BLOCK_SIZE_M）
    - num_experts: 专家总数

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    Returns（返回值说明）:
    - sorted_token_ids: 按专家分组排序后的 token 索引（含填充占位）
    - expert_ids: 每个块对应的专家 id（第 i 块由哪个专家处理）
    - num_tokens_post_padded: 填充后的总 token 数（保证能被 block_size 整除）

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.
    本函数把每个专家需处理的 token 数补齐到 block_size 的整数倍，
    确保分块矩阵乘时各维度对齐。

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.

    示例：4 个 token 各选 3 个专家，共 12 个 (token,专家) 对，每专家 3 个 token；
    block_size=4 时每专家补 1 个占位 token（id=12，不存在，kernel 直接忽略），
    排序后每专家占满一个整块，矩阵乘的 M 维整齐无需特殊处理。
    """
    from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size  # 延迟导入 sgl-kernel 的融合实现（单 kernel 完成排序+计数）

    max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)  # 填充上界：所有 (token,专家) 对 + 每专家最多补 block_size-1 个
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)  # 预分配排序后的 token id 数组（含占位）
    max_num_m_blocks = div_ceil(max_num_tokens_padded, block_size)  # M 维总块数 = 填充上界向上取整除以块大小
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)  # 预分配每块专家 id 数组
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)  # 预分配标量：实际填充后的总 token 数（kernel 写入）
    cumsum_buffer = torch.empty((num_experts + 2,), dtype=torch.int32, device=topk_ids.device)  # 预分配前缀和缓冲：各专家 token 数的累计（含哨兵位）
    sgl_moe_align_block_size(  # 融合 kernel：统计各专家 token 数 → 前缀和 → 按专家稳定排序 → 写占位
        topk_ids,  # 输入：[tokens, topk] 专家 id
        num_experts + 1,  # 专家数 +1：额外一个桶收纳无效 id（如 -1）
        block_size,  # 对齐块大小
        sorted_ids,  # 输出：排序+填充后的 token id
        expert_ids,  # 输出：每块对应的专家 id
        num_tokens_post_pad,  # 输出：填充后总长
        cumsum_buffer,  # 工作缓冲：专家计数前缀和
        True,  # pad_sorted：是否在排序结果中写入块对齐占位
    )
    return sorted_ids, expert_ids, num_tokens_post_pad  # 三元组直接作为 fused_moe_kernel 的调度元数据


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
) -> Dict[str, int]:
    """选择 Triton fused MoE kernel 的分块参数（tuning config）。

    M=token 数、E=专家数、N/K=权重维度、topk=每 token 专家数。
    策略很简单：token 少（M<=E）时用小块（减少空转块），否则用默认大块。
    """
    # 默认分块：M 维 64 / N 维 64 / K 维 32，M 维按 8 组分组调度（提升 L2 命中）
    config = {
        "BLOCK_SIZE_M": 64,  # M 维块大小：每块处理的 token-(expert) 对数
        "BLOCK_SIZE_N": 64,  # N 维块大小：每块计算的输出列数
        "BLOCK_SIZE_K": 32,  # K 维块大小：每次内积累加的输入深度
        "GROUP_SIZE_M": 8,  # M 维分组数：相邻块共享 N 切片，改善缓存局部性
    }
    if M <= E:  # decode 场景：token 数通常远小于专家数，小块减少 padding 浪费
        config = {
            "BLOCK_SIZE_M": 16,  # 小 M 块：避免大量空块
            "BLOCK_SIZE_N": 32,  # 相应缩小 N 块
            "BLOCK_SIZE_K": 64,  # 加深 K 维单次累加
            "GROUP_SIZE_M": 1,  # 不分组（块太少，分组无收益）
        }
    return config  # 返回分块参数字典，直接传给 Triton kernel


def try_get_optimal_moe_config(
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    top_k: int,
    M: int,
) -> Dict[str, int]:
    """尝试获取最优分块配置（当前实现只是回落到默认配置）。

    原版 vLLM 会按 (E, N, device) 查 tuning 表，本简化版直接用启发式默认值。
    """
    E, _, N = w2_shape  # 从 w2 形状解出专家数 E 与专家输出维度 N（w2: [E, hidden, inter]）
    config = get_default_config(M, E, N, w1_shape[2], top_k)  # w1_shape[2] 即 hidden 维 K，套用启发式配置
    return config  # 返回配置字典


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    """fused MoE 主体：两次分组 GEMM（Triton）+ 中间激活 + top-k 加权求和。

    数据流：x [M, hidden]
      → kernel1: 每 (token, 专家) 对算 x @ w1ᵀ → cache1 [M, topk, 2N]
      → 激活: silu(gate)*up → cache2 [M*topk, N]
      → kernel2: @ w2ᵀ → cache3 [M, topk, hidden]（同时乘路由权重）
      → 归约: top-k 维求和 → 输出 [M, hidden]
    """
    from minisgl.kernel import fused_moe_kernel_triton, moe_sum_reduce_triton  # 本项目的两个 Triton kernel：分组 GEMM 与 top-k 求和
    from minisgl.layers import gelu_and_mul, silu_and_mul  # 融合激活算子

    padded_size = 0  # 预留的 per-token 对齐填充（当前未启用，恒为 0）
    assert hidden_states.shape[1] == w1.shape[2] - padded_size, "Hidden size mismatch"  # 校验输入维度与 w1 的 K 维匹配
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"  # 权重与 id 必须同为 [M, topk]
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"  # Triton kernel 要求连续内存（按指针线性寻址）
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"  # 同上
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"  # 同上
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]  # 仅支持三种常用精度（kernel 模板限制）
    num_tokens, _ = hidden_states.shape  # M：token 数
    E, N, _ = w1.shape  # E：专家数；N：w1 输出维度（=2×专家中间宽度，gate/up 拼一起）
    M = num_tokens  # 别名，与 vLLM 原版记号对齐
    get_config_func = functools.partial(  # 固定"权重形状/topk"参数，之后只需传 M 即可取配置
        try_get_optimal_moe_config,
        w1.shape,  # w1 形状 [E, 2N, hidden]
        (w2.shape[0], w2.shape[1], w2.shape[2] - padded_size),  # w2 形状（扣除未启用的 padding 维）
        topk_ids.shape[1],  # topk
    )
    config = get_config_func(M)  # 按当前 token 数取分块配置

    cache = torch.empty(  # 一次性分配大缓冲，cache1/cache3 从中切片（省一次显存分配且利于复用）
        M * topk_ids.shape[1] * max(N, w2.shape[1]),  # 容量取两种中间结果的最大需求
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache1 = cache[: M * topk_ids.shape[1] * N].view(  # cache1：kernel1 输出（gate_up 未激活），容量 [M, topk, N]
        (M, topk_ids.shape[1], N),
    )
    intermediate_cache2 = torch.empty(  # cache2：激活后结果 [M*topk, N/2]（2N 拆 gate/up 后减半），独立分配（与 cache1 生命周期重叠）
        (M * topk_ids.shape[1], N // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    intermediate_cache3 = cache[: M * topk_ids.shape[1] * w2.shape[1]].view(  # cache3：kernel2 输出，与 cache1 共享同一缓冲（复用显存）
        (M, topk_ids.shape[1], w2.shape[1]),
    )
    compute_type = hidden_states.dtype  # GEMM 计算精度：跟随输入 dtype

    out_hidden_states = hidden_states  # 输出复用输入缓冲（就地累加，省一份显存）
    curr_hidden_states = hidden_states  # 当前参与 GEMM 的输入（分块处理时按段切换）
    tokens_num, _ = curr_hidden_states.shape  # 当前段 token 数（本实现未分块，等于 M）
    begin_token_idx, end_token_idx = 0, num_tokens  # 处理区间：全部 token（分块框架预留的起止下标）

    intermediate_cache1 = intermediate_cache1[:tokens_num]  # 按实际 token 数裁剪视图（去掉 padding 部分）
    intermediate_cache2 = intermediate_cache2[: tokens_num * topk_ids.shape[1]]  # 同上（cache2 是平铺的 [M*topk, N/2]）
    intermediate_cache3 = intermediate_cache3[:tokens_num]  # 同上
    config = get_config_func(tokens_num)  # 用实际 token 数重取配置（与上面 M 相同，冗余但无害）

    curr_topk_ids = topk_ids[begin_token_idx:end_token_idx]  # 当前段的专家 id [M, topk]
    curr_topk_weights = topk_weights[begin_token_idx:end_token_idx]  # 当前段的路由权重 [M, topk]

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(  # 对齐排序：把 (token,专家) 对按专家分组并补齐成整块
        curr_topk_ids, config["BLOCK_SIZE_M"], E  # 块大小取配置中的 BLOCK_SIZE_M（GEMM 的 M 维 tile）
    )

    fused_moe_kernel_triton(  # 第一轮分组 GEMM：对每个 (token, 专家) 对计算 x @ w1[e]ᵀ（gate_up 升维）
        curr_hidden_states,  # 输入 x [M, hidden]
        w1,  # 专家权重栈 [E, 2N, hidden]
        intermediate_cache1,  # 输出缓冲 [M, topk, 2N]
        curr_topk_weights,  # 路由权重（apply_router_weight_on_input=True 时在此乘入）
        curr_topk_ids,  # 专家 id
        sorted_token_ids,  # 对齐排序后的调度索引
        expert_ids,  # 每块对应的专家 id
        num_tokens_post_padded,  # 填充后总长（kernel 据此忽略越界块）
        apply_router_weight_on_input,  # 是否在输入端乘路由权重
        topk_ids.shape[1],  # topk：kernel 展开每 token 的多个专家
        config,  # 分块配置
        compute_type=compute_type,  # 计算精度
    )
    FN_MAP = {"silu": silu_and_mul, "gelu": gelu_and_mul}  # 激活函数名 → 融合算子映射
    FN_MAP[activation](intermediate_cache1.view(-1, N), intermediate_cache2)  # 融合激活：silu(gate)*up，[M*topk, 2N] → [M*topk, N]
    fused_moe_kernel_triton(  # 第二轮分组 GEMM：降维 x @ w2[e]ᵀ，并把路由权重乘到输出
        intermediate_cache2,  # 输入：激活后的隐状态 [M*topk, N/2]
        w2,  # 专家权重栈 [E, hidden, N/2]
        (intermediate_cache3),  # 输出缓冲 [M, topk, hidden]
        curr_topk_weights,  # 路由权重（apply_router_weight_on_input=False 时在此乘入）
        curr_topk_ids,  # 专家 id
        sorted_token_ids,  # 同一轮调度索引（两轮用同一套分组）
        expert_ids,  # 每块专家 id
        num_tokens_post_padded,  # 填充总长
        not apply_router_weight_on_input,  # 与第一轮互斥：路由权重只在其中一轮乘入（避免乘两次）
        1,  # topk 传 1：此轮每个输入行已对应唯一的 (token,专家) 对
        config,  # 分块配置
        compute_type=compute_type,  # 计算精度
    )

    moe_sum_reduce_triton(  # top-k 归约：把 [M, topk, hidden] 沿专家维求和写回输出
        intermediate_cache3,  # 输入：每专家的输出 [M, topk, hidden]
        out_hidden_states[begin_token_idx:end_token_idx],  # 输出：就地写入输入缓冲的对应段 [M, hidden]
    )
    return out_hidden_states  # 返回最终 MoE 输出（与输入共享存储）


class FusedMoe(BaseMoeBackend):
    """fused MoE 后端：路由（fused_topk）+ 专家计算（fused_experts_impl）的完整实现。"""

    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入 token 表示 [num_tokens, hidden]
        w1: torch.Tensor,  # 专家第一层权重栈 [E, 2*inter, hidden]
        w2: torch.Tensor,  # 专家第二层权重栈 [E, hidden, inter]
        gating_output: torch.Tensor,  # 路由器 logits [num_tokens, E]
        topk: int,  # 每 token 激活专家数
        renormalize: bool,  # 是否归一化路由权重
        activation: str = "silu",  # 激活函数名
        apply_router_weight_on_input: bool = False,  # 路由权重乘在输入端还是输出端
    ) -> torch.Tensor:
        """实现 BaseMoeBackend 接口：先路由再计算，返回与输入同形状的结果。"""
        topk_weights, topk_ids = fused_topk(  # 第一步：softmax + top-k 路由，得每 token 的权重与专家 id
            hidden_states=hidden_states,  # 借用其形状/设备
            gating_output=gating_output,  # 路由 logits
            topk=topk,  # top-k
            renormalize=renormalize,  # 归一化开关
        )
        return fused_experts_impl(  # 第二步：分组 GEMM ×2 + 激活 + 加权求和
            hidden_states,  # 输入
            w1,  # 升维权重栈
            w2,  # 降维权重栈
            topk_weights,  # 路由权重 [M, topk]
            topk_ids,  # 专家 id [M, topk]
            activation,  # 激活函数
            apply_router_weight_on_input=apply_router_weight_on_input,  # 路由权重作用位置
        )
