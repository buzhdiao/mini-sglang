# 类型注解工具
from typing import Any, Dict

# torch 提供张量与 dtype
import torch


# Triton 实现的 MoE 融合 GEMM 内核的 Python 启动封装。
# A (M,K)：按 expert 排序后的 token 激活；B (E,N,K)：各 expert 权重；C：输出缓冲；
# topk_weights：每 token 每 expert 的路由权重；topk_ids：每 token 选中的 expert id；
# sorted_token_ids：排序+填充后的 token 索引（分块对齐）；expert_ids：每块对应的 expert；
# num_tokens_post_padded：填充后的实际 token 数；mul_routed_weight：是否乘路由权重；
# top_k：每 token 的 top-k 数；config：Triton 调优配置；compute_type：计算精度
def fused_moe_kernel_triton(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    top_k: int,
    config: Dict[str, Any],
    compute_type: torch.dtype,
) -> None:
    # 延迟导入 triton
    import triton
    import triton.language as tl

    # 导入预定义的 Triton kernel
    from .triton.fused_moe import fused_moe_kernel

    # 路由权重需按行连续访问
    assert topk_weights.stride(1) == 1
    # token 索引需连续
    assert sorted_token_ids.stride(0) == 1
    # 本实现不支持额外填充
    padded_size = 0
    # 网格：M 维块数 x N 维块数（META 运行时读取块大小）
    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),
    )
    # K 维大小（扣除填充）
    K = B.shape[2] - padded_size
    # K 能否被 BLOCK_SIZE_K 整除（决定 kernel 内是否处理尾块）
    if K % config["BLOCK_SIZE_K"] == 0:
        # K 整除，无尾块
        even_Ks = True
    else:
        # 存在尾块
        even_Ks = False
    # 把 torch dtype 映射为 triton dtype
    dtype = tl.bfloat16 if compute_type == torch.bfloat16 else tl.float16
    # 启动融合 MoE kernel
    fused_moe_kernel[grid](
        A,
        B,
        C,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        B.shape[1],
        B.shape[2] - padded_size,
        sorted_token_ids.shape[0],
        topk_ids.numel(),
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(2),
        B.stride(1),
        C.stride(1),
        C.stride(2),
        MUL_ROUTED_WEIGHT=mul_routed_weight,  # type: ignore
        top_k=top_k,  # type: ignore
        compute_type=dtype,  # type: ignore
        even_Ks=even_Ks,  # type: ignore
        **config,
    )


# Triton 归约内核的启动封装：把 (token, topk, hidden) 沿 topk 维求和回 (token, hidden)
def moe_sum_reduce_triton(input: torch.Tensor, output: torch.Tensor) -> None:
    # 延迟导入 triton
    import triton

    # 导入预定义的归约 kernel
    from .triton.fused_moe import moe_sum_reduce_kernel

    # 输入必须连续
    assert input.is_contiguous()
    # 输出必须连续
    assert output.is_contiguous()

    # 取形状：token 数、topk 数、隐藏维度
    token_num, topk_num, hidden_dim = input.shape
    # 输出形状需与 (token, hidden) 匹配
    assert output.shape[0] == token_num and output.shape[1] == hidden_dim

    # 每个 block 处理 1 个 token
    BLOCK_M = 1
    # 每个 block 处理 2048 个隐藏维元素
    BLOCK_DIM = 2048
    # 不做流水线分级
    NUM_STAGE = 1
    # 每块 8 个 warp
    num_warps = 8

    # 网格：token 维块数 x 隐藏维块数
    grid = (
        triton.cdiv(token_num, BLOCK_M),
        triton.cdiv(hidden_dim, BLOCK_DIM),
    )

    # 启动归约 kernel（stride 展开传入）
    moe_sum_reduce_kernel[grid](
        input,
        *input.stride(),
        output,  # type: ignore
        *output.stride(),
        token_num=token_num,
        topk_num=topk_num,
        hidden_dim=hidden_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_DIM=BLOCK_DIM,
        NUM_STAGE=NUM_STAGE,
        num_warps=num_warps,  # type: ignore
    )
