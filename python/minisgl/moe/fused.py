# functools 用于固化 GEMM 分块配置函数的部分参数
import functools
from typing import Dict, Tuple

import torch
# MoE 后端抽象基类
from minisgl.moe import BaseMoeBackend
# 向上取整除法
from minisgl.utils import div_ceil


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_token_non_padded: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # 融合 top-k 路由：对每 token 的专家 logits 做 softmax 并选出 top-k 个专家
    from sgl_kernel import topk_softmax

    # token 数必须一致
    assert hidden_states.shape[0] == gating_output.shape[0], "Number of tokens mismatch"
    # token 数
    M, _ = hidden_states.shape
    # 输出：每 token 的 top-k 路由权重（fp32 保证精度）
    topk_weights = torch.empty(M, topk, dtype=torch.float32, device=hidden_states.device)
    # 输出：每 token 选中的专家编号
    topk_ids = torch.empty(M, topk, dtype=torch.int32, device=hidden_states.device)
    # GPU 核一次完成 softmax + top-k 选择
    topk_softmax(topk_weights, topk_ids, gating_output.float(), renormalize)
    if renormalize:
        # 对选中的 top-k 权重重归一化，使概率和为 1（Qwen3-MoE 等模型的约定）
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-8)
    if num_token_non_padded is not None:
        # CUDA Graph 模式下 batch 被 pad 到固定大小：把填充 token 的路由置 -1，
        # 后续对齐排序会跳过它们，保证只算真实 token
        indices = torch.arange(0, topk_ids.shape[0], device=topk_ids.device)
        topk_ids[indices >= num_token_non_padded, :] = -1
    return topk_weights, topk_ids


def moe_align_block_size(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

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
    """
    # MoE 分组 GEMM 的对齐准备：把每个 token*topk 的行按专家排序，
    # 并把每个专家的 token 数 pad 到 block_size 整数倍，
    # 这样每个 GEMM 分块内只含一个专家，块尺寸恒定，可直接用固定分块的 Triton 核
    from sgl_kernel import moe_align_block_size as sgl_moe_align_block_size

    # 排序后数组的上界：token*topk 总数 + 每个专家最多补 (block_size-1) 个填充
    max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)
    # 预分配排序结果缓冲（填充位置写入超大哨兵值）
    sorted_ids = torch.empty((max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device)
    # GEMM 的 M 方向块数上界
    max_num_m_blocks = div_ceil(max_num_tokens_padded, block_size)
    # 每个 M 块对应的专家编号
    expert_ids = torch.empty((max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device)
    # 实际 pad 后的总 token 数（单元素张量，供核函数在 GPU 上读取）
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    # 专家前缀和缓冲，用于 GPU 上并行计数排序
    cumsum_buffer = torch.empty((num_experts + 2,), dtype=torch.int32, device=topk_ids.device)
    # 调用 sgl-kernel 的一次完成"计数排序 + 分块填充"的核
    sgl_moe_align_block_size(
        topk_ids,
        num_experts + 1,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
        cumsum_buffer,
        True,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
) -> Dict[str, int]:

    # 默认 GEMM 分块配置：适合 token 数较多的一般情形
    config = {
        "BLOCK_SIZE_M": 64,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
    }
    if M <= E:
        # token 数不多于专家数（解码小 batch）：每个专家的 token 很少，
        # 用小 M 块减少 padding 浪费、提高张量核利用率
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
    return config


def try_get_optimal_moe_config(
    w1_shape: Tuple[int, ...],
    w2_shape: Tuple[int, ...],
    top_k: int,
    M: int,
) -> Dict[str, int]:
    # 选定分块配置的入口：当前实现按启发式默认值（可扩展为按形状调优表）
    E, _, N = w2_shape
    config = get_default_config(M, E, N, w1_shape[2], top_k)
    return config


def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    apply_router_weight_on_input: bool = False,
) -> torch.Tensor:
    # 融合专家计算的完整流水：w1 升维 GEMM -> 门控激活 -> w2 降维 GEMM -> top-k 求和归并
    from minisgl.kernel import fused_moe_kernel_triton, moe_sum_reduce_triton
    from minisgl.layers import gelu_and_mul, silu_and_mul

    padded_size = 0
    # 各输入形状/内存布局合法性检查
    assert hidden_states.shape[1] == w1.shape[2] - padded_size, "Hidden size mismatch"
    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.is_contiguous(), "Expert weights1 must be contiguous"
    assert w2.is_contiguous(), "Expert weights2 must be contiguous"
    assert hidden_states.dtype in [torch.float32, torch.float16, torch.bfloat16]
    # token 数
    num_tokens, _ = hidden_states.shape
    # w1: (专家数 E, 2N, K)，N 为升维中间维（含 gate+up 两份）
    E, N, _ = w1.shape
    M = num_tokens
    # 固化形状参数，只留 M 待定的配置选择函数
    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.shape,
        (w2.shape[0], w2.shape[1], w2.shape[2] - padded_size),
        topk_ids.shape[1],
    )
    config = get_config_func(M)

    # 一次性分配大缓冲，再切出各中间结果（减少 allocator 往返与显存碎片）
    cache = torch.empty(
        M * topk_ids.shape[1] * max(N, w2.shape[1]),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    # cache1：第一次 GEMM 输出，(M, topk, 2N) —— 每 token*topk 行一份 gate|up
    intermediate_cache1 = cache[: M * topk_ids.shape[1] * N].view(
        (M, topk_ids.shape[1], N),
    )
    # cache2：门控激活输出，(M*topk, N/2)
    intermediate_cache2 = torch.empty(
        (M * topk_ids.shape[1], N // 2),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    # cache3：第二次 GEMM 输出，(M, topk, hidden)，复用 cache 的前段
    intermediate_cache3 = cache[: M * topk_ids.shape[1] * w2.shape[1]].view(
        (M, topk_ids.shape[1], w2.shape[1]),
    )
    # 计算精度：与输入一致（fp16/bf16）
    compute_type = hidden_states.dtype

    # 输出直接复用输入缓冲（原地更新，省一次分配）
    out_hidden_states = hidden_states
    # 当前处理的隐藏态
    curr_hidden_states = hidden_states
    tokens_num, _ = curr_hidden_states.shape
    # 当前处理的 token 区间 [0, num_tokens)（预留分段处理能力）
    begin_token_idx, end_token_idx = 0, num_tokens

    # 按实际 token 数裁剪各中间缓冲
    intermediate_cache1 = intermediate_cache1[:tokens_num]
    intermediate_cache2 = intermediate_cache2[: tokens_num * topk_ids.shape[1]]
    intermediate_cache3 = intermediate_cache3[:tokens_num]
    # 按真实 token 数重新选定分块配置
    config = get_config_func(tokens_num)

    # 取本段的 top-k 专家编号与权重
    curr_topk_ids = topk_ids[begin_token_idx:end_token_idx]
    curr_topk_weights = topk_weights[begin_token_idx:end_token_idx]

    # 对齐排序：每个专家的 token pad 到 BLOCK_SIZE_M 倍数，产出 GEMM 需要的索引
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        curr_topk_ids, config["BLOCK_SIZE_M"], E
    )

    # 第一次 GEMM：x @ w1^T（升维）。apply_router_weight_on_input 为真时
    # 在此乘路由权重（乘在输入侧的模型约定）
    fused_moe_kernel_triton(
        curr_hidden_states,
        w1,
        intermediate_cache1,
        curr_topk_weights,
        curr_topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        apply_router_weight_on_input,
        topk_ids.shape[1],
        config,
        compute_type=compute_type,
    )
    # 按配置选择门控激活函数
    FN_MAP = {"silu": silu_and_mul, "gelu": gelu_and_mul}
    # 融合门控激活：silu(gate)*up，把 2N 压回 N
    FN_MAP[activation](intermediate_cache1.view(-1, N), intermediate_cache2)
    # 第二次 GEMM：h @ w2^T（降维）。默认（乘在输出侧）在此乘路由权重
    fused_moe_kernel_triton(
        intermediate_cache2,
        w2,
        (intermediate_cache3),
        curr_topk_weights,
        curr_topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        not apply_router_weight_on_input,
        1,
        config,
        compute_type=compute_type,
    )

    # 归并：把每 token 的 topk 份专家输出按元素求和，写回输出缓冲
    moe_sum_reduce_triton(
        intermediate_cache3,
        out_hidden_states[begin_token_idx:end_token_idx],
    )
    return out_hidden_states


class FusedMoe(BaseMoeBackend):
    # 融合 MoE 后端：路由 + 专家计算全流程
    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
    ) -> torch.Tensor:
        # 第一步：softmax + top-k 选专家，得到路由权重与专家编号
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
        )
        # 第二步：专家分组 GEMM 流水，产出加权重合并后的隐藏态
        return fused_experts_impl(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )
