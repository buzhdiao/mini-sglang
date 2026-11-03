from typing import Any, Dict  # 类型注解：config 字典的值类型（任意）与字典本身

import torch  # PyTorch：张量与 dtype


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
    """启动 MoE fused GEMM Triton kernel 的 Python 封装。

    对按专家分组并 padding 后的 token（sorted_token_ids）执行分块矩阵乘：
    C[token, expert_slot, :] = A[token, :] @ B[expert, :]^T，
    可选乘上路由权重 topk_weights。结果原地写入 C。
    """
    import triton  # Triton 运行时（延迟导入，缩短模块加载时间）

    import triton.language as tl  # Triton 语言内建（bfloat16/float16 dtype、cdiv 等）

    from .triton.fused_moe import fused_moe_kernel  # 实际的 Triton kernel 定义

    assert topk_weights.stride(1) == 1  # 要求 topk_weights 最后一维连续，保证 kernel 内按 1 步长加载
    assert sorted_token_ids.stride(0) == 1  # 要求 sorted_token_ids 一维连续
    padded_size = 0  # 本实现假定 K 维不做 padding，故恒为 0（保留参数位以便扩展）
    # 网格大小 = M 方向 tile 数 x N 方向 tile 数；BLOCK_SIZE_M/N 由 config 在 META 中回传
    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])  # M 方向：padding 后的 token 总数除以每块行数，向上取整
        * triton.cdiv(B.shape[1], META["BLOCK_SIZE_N"]),  # N 方向：输出特征维度除以每块列数，向上取整
    )
    K = B.shape[2] - padded_size  # 输入特征维度 K（无 padding 时即 B 的最后一维）
    if K % config["BLOCK_SIZE_K"] == 0:  # K 能被块大小整除 -> 循环内无需 K 掩码（更快路径）
        even_Ks = True
    else:  # 否则最后一个 K 块需要掩码防止越界
        even_Ks = False
    dtype = tl.bfloat16 if compute_type == torch.bfloat16 else tl.float16  # 把 torch dtype 映射为 Triton dtype（累加结果写回时的目标类型）
    fused_moe_kernel[grid](  # 以 grid 启动 Triton kernel，以下按位置传入各参数
        A,  # 输入激活 (num_tokens, K)
        B,  # 专家权重 (E, N, K)
        C,  # 输出缓存 (M, topk, N)
        topk_weights,  # 每个 (token, expert槽) 的路由权重
        sorted_token_ids,  # 按专家分组、padding 后的 token 索引数组
        expert_ids,  # 每个 M 块对应的专家 id
        num_tokens_post_padded,  # 设备端标量：padding 后的有效 token 总数
        B.shape[1],  # N：输出特征维度
        B.shape[2] - padded_size,  # K：输入特征维度
        sorted_token_ids.shape[0],  # EM：padding 后数组总长
        topk_ids.numel(),  # 有效 token 数上限（原始 token 数 x topk），用于掩码
        A.stride(0),  # A 行步长 stride_am
        A.stride(1),  # A 列步长 stride_ak
        B.stride(0),  # B 专家维步长 stride_be
        B.stride(2),  # B 的 K 维步长 stride_bk（注意 B 布局为 (E, N, K)，K 在最内维）
        B.stride(1),  # B 的 N 维步长 stride_bn
        C.stride(1),  # C 的 token 槽位步长 stride_cm
        C.stride(2),  # C 的 N 维步长 stride_cn
        MUL_ROUTED_WEIGHT=mul_routed_weight,  # type: ignore  # 是否在累加后乘路由权重（第一层 GEMM 不乘，第二层乘）
        top_k=top_k,  # type: ignore  # 每个 token 的 topk 专家数，用于 offs_token//top_k 还原真实 token 行号
        compute_type=dtype,  # type: ignore  # 输出写回的数值类型
        even_Ks=even_Ks,  # type: ignore  # K 维是否整齐（决定加载掩码分支）
        **config,  # 展开 BLOCK_SIZE_M/N/K、GROUP_SIZE_M、num_warps/num_stages 等调优参数
    )


def moe_sum_reduce_triton(input: torch.Tensor, output: torch.Tensor) -> None:
    """对 MoE 输出按 topk 维求和归约：output[t, :] = sum_i input[t, i, :]。

    将 (token_num, topk_num, hidden_dim) 的中间结果压缩为
    (token_num, hidden_dim)，原地写入 output。
    """
    import triton  # Triton 运行时（延迟导入）

    from .triton.fused_moe import moe_sum_reduce_kernel  # 归约 kernel 定义

    assert input.is_contiguous()  # 输入必须连续，步长展开才正确
    assert output.is_contiguous()  # 输出必须连续

    token_num, topk_num, hidden_dim = input.shape  # 解包形状：token 数、topk 数、隐藏维度
    assert output.shape[0] == token_num and output.shape[1] == hidden_dim  # 校验输出形状与归约结果一致

    BLOCK_M = 1  # 每个 program 处理 1 个 token（token 维不并行分块，靠 grid 维度并行）
    BLOCK_DIM = 2048  # 每个 program 处理 2048 个隐藏维度元素
    NUM_STAGE = 1  # 软件流水级数为 1（不做多级流水）
    num_warps = 8  # 每个 program 使用 8 个 warp（256 线程）

    # 二维网格：(token 数 / BLOCK_M) x (隐藏维 / BLOCK_DIM)，均向上取整
    grid = (
        triton.cdiv(token_num, BLOCK_M),
        triton.cdiv(hidden_dim, BLOCK_DIM),
    )

    moe_sum_reduce_kernel[grid](  # 启动归约 kernel
        input,
        *input.stride(),  # 展开输入的三个步长
        output,  # type: ignore  # 输出张量
        *output.stride(),  # 展开输出的两个步长
        token_num=token_num,  # token 总数（用于边界判断）
        topk_num=topk_num,  # topk 数（内层循环次数）
        hidden_dim=hidden_dim,  # 隐藏维（用于边界判断）
        BLOCK_M=BLOCK_M,  # token 方向块大小
        BLOCK_DIM=BLOCK_DIM,  # 隐藏维块大小
        NUM_STAGE=NUM_STAGE,  # 流水级数
        num_warps=num_warps,  # type: ignore  # 启动参数：每块 warp 数
    )
