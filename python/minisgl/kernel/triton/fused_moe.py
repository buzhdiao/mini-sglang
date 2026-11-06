# 导入 Triton 与其语言接口，用于编写在 GPU 上并行执行的核函数
import triton
import triton.language as tl


# Triton JIT 装饰器：把下面的 Python 函数在首次调用时编译成 GPU 核函数
# 该核函数做 MoE 输出归并：每个 token 被 topk 个专家各算一次，得到
# (token_num, topk_num, hidden_dim) 的中间结果，本核沿 topk 维按元素求和，
# 归并成最终 (token_num, hidden_dim) 的输出
@triton.jit
def moe_sum_reduce_kernel(
    input_ptr,
    input_stride_0,
    input_stride_1,
    input_stride_2,
    output_ptr,
    output_stride_0,
    output_stride_1,
    token_num: int,
    topk_num: int,
    hidden_dim: int,
    BLOCK_M: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    NUM_STAGE: tl.constexpr,
):
    # 参数说明：input_ptr/output_ptr 为输入/输出首地址；
    # input_stride_0/1/2 与 output_stride_0/1 为两张量各维步长（以元素为单位）；
    # token_num/topk_num/hidden_dim 为逻辑形状；BLOCK_M/BLOCK_DIM 为二维网格分块大小；
    # NUM_STAGE 为沿 topk 维循环的软流水级数
    # 把涉及大维度的步长转成 int64，防止大 batch 时 32 位地址偏移计算溢出
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    # 网格是二维的：第 0 维按 token 分块，第 1 维按 hidden 维分块
    token_block_id = tl.program_id(0)
    dim_block_id = tl.program_id(1)

    # 计算本 program 负责的 token 起止下标（左闭右开区间）
    token_start = token_block_id * BLOCK_M
    token_end = min((token_block_id + 1) * BLOCK_M, token_num)

    # 计算本 program 负责的 hidden 维起止下标
    dim_start = dim_block_id * BLOCK_DIM
    dim_end = min((dim_block_id + 1) * BLOCK_DIM, hidden_dim)

    # 生成本块覆盖的 hidden 维下标向量
    offs_dim = dim_start + tl.arange(0, BLOCK_DIM)

    # 逐 token 归并：把该 token 在 topk 个专家上的输出按元素累加
    for token_index in range(token_start, token_end):
        # 用 fp32 累加器累加，避免多专家求和时精度损失
        accumulator = tl.zeros((BLOCK_DIM,), dtype=tl.float32)
        # 定位该 token 在输入中本 hidden 分块的地址（带上 hidden 维偏移）
        input_t_ptr = input_ptr + token_index * input_stride_0 + offs_dim
        # 沿 topk 维循环累加，num_stages 让 Triton 对载入做软流水预取
        for i in tl.range(0, topk_num, num_stages=NUM_STAGE):
            # 载入第 i 个专家对当前 token 的输出，hidden 越界位置填 0
            tmp = tl.load(input_t_ptr + i * input_stride_1, mask=offs_dim < dim_end, other=0.0)
            # 累加进 fp32 累加器
            accumulator += tmp
        # 定位该 token 在输出中本 hidden 分块的写入地址
        store_t_ptr = output_ptr + token_index * output_stride_0 + offs_dim
        # 把 fp32 结果转回输入元素类型并写出，掩码屏蔽 hidden 越界部分
        tl.store(
            store_t_ptr,
            accumulator.to(input_ptr.dtype.element_ty),
            mask=offs_dim < dim_end,
        )


# Triton JIT 装饰器：编译下述 MoE 融合 GEMM 核函数
# 原理：MoE 层的每个 token 只与被路由到的少数专家权重相乘。为高效执行，
# 先把所有 token 按"专家编号 × token"排序并 pad 到 BLOCK_SIZE_M 的整数倍（总行数 EM），
# 本核按分块读 sorted_token_ids/expert_ids，对每个 (BLOCK_SIZE_M, BLOCK_SIZE_N) 输出块
# 执行 C = A @ B^T：A 为 token 特征 (有效行, K)，B 为专家权重 (E, N, K)，
# 可选地乘上路由权重（MUL_ROUTED_WEIGHT），用于 MoE 的两次 GEMM（w1 升维 / w2 降维）
@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    even_Ks: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.

    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # 把一维 program id 映射成输出 C 的二维块坐标 (pid_m, pid_n)；
    # 采用"分组"映射：相邻 GROUP_SIZE_M 行块编成一组，组内横向遍历 N 方向，
    # 这样同组 program 加载的 A 分块相同、B 分块连续，可显著提升 L2 缓存命中率
    pid = tl.program_id(axis=0)
    # M 方向（pad 后的 token 行）共有多少个块
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    # N 方向共有多少个块
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    # 一个分组覆盖的 program 数（组内行块数 × N 方向块数）
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    # 当前 pid 属于第几个分组
    group_id = pid // num_pid_in_group
    # 该分组起始的 M 方向块号
    first_pid_m = group_id * GROUP_SIZE_M
    # 末尾分组可能不满，用 min 处理实际组内行块数
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    # 组内先沿 M 方向排，再折回下一列 N：得到当前 pid 的 M 方向块号
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    # 得到当前 pid 的 N 方向块号
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # 读取 pad 排序后真实的 token 行总数（由 host 侧动态计算）
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    # 本 M 块整体落在 padding 区：没有任何有效 token，直接退出
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    # 本块覆盖的"排序后行"下标（行数恰为 BLOCK_SIZE_M，可能含 padding）
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # 查表得到每行对应的原始 token*topk 展开下标；padding 位置填的是超大哨兵值
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    # 转 int64，避免大张量时地址偏移溢出
    offs_token = offs_token.to(tl.int64)
    # 有效 token 掩码：只有真实 token 才参与计算与写回
    token_mask = offs_token < num_valid_tokens

    # B 的 N 方向列下标；对 N 取模是为了尾块越界时回绕读安全地址（结果随后被掩码丢弃）
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    # K 方向块内下标 0..BLOCK_SIZE_K-1
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    # A 块指针：每行取 sorted_token_ids//top_k 还原成 token 行号再乘行步长
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)

    # 查表得到本 M 块使用的专家编号（同一块的行都属于同一专家）
    off_experts = tl.load(expert_ids_ptr + pid_m)
    # B 块指针：定位到该专家权重 (N, K) 的对应 (BLOCK_SIZE_K, BLOCK_SIZE_N) 子块
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    # 用 fp32 累加器存放 (BLOCK_SIZE_M, BLOCK_SIZE_N) 分块结果，保证精度，最后再转回低精度
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # 沿 K 维循环做分块矩阵乘，共 cdiv(K, BLOCK_SIZE_K) 次
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        # K 维恰好整除时无需 K 方向掩码，省掉一次比较
        if even_Ks:
            # 载入 A 块：只需 token 掩码，padding 行读到的地址是哨兵值但结果会被丢弃
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None],
                other=0.0,
            )
            # B 块无 K 越界，直接整块载入
            b = tl.load(b_ptrs)
        else:
            # K 不整除时，A 需同时屏蔽 padding 行与 K 方向越界列
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                other=0.0,
            )
            # B 同样屏蔽 K 方向越界部分，其余填 0（不影响累加）
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        # We accumulate along the K dimension.

        # 张量核心做 (M,K)@(K,N) 分块乘并累加进 fp32 累加器
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        # A/B 指针沿 K 方向推进一个块，进入下轮迭代
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # 第二次 GEMM（w2 降维）时需要乘上路由权重：每个 token*topk 行乘其路由 softmax 概率
    if MUL_ROUTED_WEIGHT:
        # 按原始 token*topk 下标取出该行的路由权重，padding 行取 0
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        # 权重按行广播相乘
        accumulator = accumulator * moe_weight[:, None]

    # 把 fp32 累加结果转成输出计算类型（如 fp16/bf16）
    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    # 写回：C 按 (EM, N) 布局，行下标就是排序后的 offs_token，从而每个 token*topk 各占一行
    # 输出块内 N 方向列下标
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # 计算输出块的首地址（行用 offs_token，列用 offs_cn）
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    # 写掩码：屏蔽 padding 行与 N 方向越界列
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    # 把分块结果写出到 C
    tl.store(c_ptrs, accumulator, mask=c_mask)
