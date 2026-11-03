import triton  # Triton 编译/运行时框架：@triton.jit 装饰器与网格启动接口
import triton.language as tl  # Triton 设备端语言：张量切片、加载/存储、矩阵乘 dot 等 GPU 内建操作


@triton.jit  # 标记为 Triton JIT kernel：编译为 GPU 代码，按 grid 启动
def moe_sum_reduce_kernel(
    input_ptr,  # 输入指针：形状 (token_num, topk_num, hidden_dim) 的 MoE 中间结果
    input_stride_0,  # 输入第 0 维（token）步长：跨一个 token 需要移动的元素数
    input_stride_1,  # 输入第 1 维（topk）步长：跨一个专家槽位移动的元素数
    input_stride_2,  # 输入第 2 维（hidden）步长：跨一个隐藏元素移动的元素数（最内维通常为 1）
    output_ptr,  # 输出指针：形状 (token_num, hidden_dim) 的归约结果
    output_stride_0,  # 输出第 0 维（token）步长
    output_stride_1,  # 输出第 1 维（hidden）步长
    token_num: int,  # token 总数（用于网格越界判断）
    topk_num: int,  # 每个token 的专家数（内层循环次数）
    hidden_dim: int,  # 隐藏维度（用于隐藏维掩码）
    BLOCK_M: tl.constexpr,  # 每个 program 处理的 token 数（编译期常量，本封装中为 1）
    BLOCK_DIM: tl.constexpr,  # 每个 program 处理的隐藏维元素数（编译期常量，本封装中为 2048）
    NUM_STAGE: tl.constexpr,  # 内层循环的软件流水级数（编译期常量）
):
    # 把前两个步长转成 int64：token 数 x hidden 很大时，32 位偏移会溢出，必须用 64 位地址运算
    input_stride_0 = tl.cast(input_stride_0, dtype=tl.int64)
    input_stride_1 = tl.cast(input_stride_1, dtype=tl.int64)
    output_stride_0 = tl.cast(output_stride_0, dtype=tl.int64)

    token_block_id = tl.program_id(0)  # 当前 program 在网格第 0 维的编号（负责哪些 token）
    dim_block_id = tl.program_id(1)  # 当前 program 在网格第 1 维的编号（负责哪段隐藏维）

    token_start = token_block_id * BLOCK_M  # 本 program 负责的 token 区间起点
    token_end = min((token_block_id + 1) * BLOCK_M, token_num)  # 区间终点（不越过 token 总数）

    dim_start = dim_block_id * BLOCK_DIM  # 本 program 负责的隐藏维区间起点
    dim_end = min((dim_block_id + 1) * BLOCK_DIM, hidden_dim)  # 区间终点（不越过隐藏维）

    offs_dim = dim_start + tl.arange(0, BLOCK_DIM)  # 长度 BLOCK_DIM 的隐藏维偏移向量（本块负责的列下标）

    for token_index in range(token_start, token_end):  # 逐 token 处理（BLOCK_M=1 时只循环一次）
        accumulator = tl.zeros((BLOCK_DIM,), dtype=tl.float32)  # fp32 累加器：比原精度更高，避免多次累加损失精度
        input_t_ptr = input_ptr + token_index * input_stride_0 + offs_dim  # 指向该 token 的隐藏维切片基址
        for i in tl.range(0, topk_num, num_stages=NUM_STAGE):  # 遍历 topk 个专家输出（可硬件流水化）
            # 加载该 token 第 i 个专家在隐藏维切片上的值；越界列用 0 填充
            tmp = tl.load(input_t_ptr + i * input_stride_1, mask=offs_dim < dim_end, other=0.0)
            accumulator += tmp  # 累加到寄存器中的向量（对 topk 求和）
        store_t_ptr = output_ptr + token_index * output_stride_0 + offs_dim  # 指向输出中该 token 的切片
        tl.store(  # 把 fp32 累加结果转回输入元素类型后写回输出
            store_t_ptr,
            accumulator.to(input_ptr.dtype.element_ty),  # 类型收窄为输入 dtype（如 bf16/fp16）
            mask=offs_dim < dim_end,  # 掩码写回，防止最后一个块越过 hidden_dim 边界
        )


@triton.jit  # 标记为 Triton JIT kernel：MoE fused 分组 GEMM
def fused_moe_kernel(
    # Pointers to matrices  # 各矩阵的设备端指针
    a_ptr,  # 输入激活 A 指针，逻辑形状 (*, K)（* 为批维，这里按 token 索引）
    b_ptr,  # 专家权重 B 指针，逻辑形状 (E, N, K)：E 个专家，输出维 N，输入维 K
    c_ptr,  # 输出 C 指针，逻辑形状 (M, topk, N)：每个 token 在其 topk 个专家槽位各有一行输出
    topk_weights_ptr,  # 路由权重指针：每个 (token, 专家槽位) 的加权系数
    sorted_token_ids_ptr,  # 按专家分组排序并 padding 后的 token 索引数组（长度 EM）
    expert_ids_ptr,  # 每个 M 方向 tile 对应的专家 id（长度 = M 方向 tile 数）
    num_tokens_post_padded_ptr,  # 设备端标量：padding 后的有效 token 总数（用于提前退出空块）
    # Matrix dimensions  # 矩阵维度
    N,  # B 的输出特征维度（C 的列数）
    K,  # B 的输入特征维度（A 的列数，即规约维）
    EM,  # sorted_token_ids 的总长度（padding 后）
    num_valid_tokens,  # 有效 token 数上限（原始 token 数 x topk），越界 token 用掩码丢弃
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).  # 步长含义：沿某维移动 1 个元素时指针的增量；如 stride_am 是 A 下一行的偏移
    stride_am,  # A 的 token 维步长
    stride_ak,  # A 的 K 维步长
    stride_be,  # B 的专家维步长
    stride_bk,  # B 的 K 维步长（K 是 B 最内维）
    stride_bn,  # B 的 N 维步长
    stride_cm,  # C 的槽位维步长（槽位 = token x topk 展开）
    stride_cn,  # C 的 N 维步长
    # Meta-parameters  # 编译期元参数（决定 tile 尺寸与行为分支）
    BLOCK_SIZE_M: tl.constexpr,  # M 方向 tile 高度：每块处理的 token（槽位）数
    BLOCK_SIZE_N: tl.constexpr,  # N 方向 tile 宽度：每块计算的输出列数
    BLOCK_SIZE_K: tl.constexpr,  # K 方向 tile 深度：每步内积累加的长度
    GROUP_SIZE_M: tl.constexpr,  # L2 分组大小：连续多少个 M tile 共享 N 遍历以提升缓存命中
    MUL_ROUTED_WEIGHT: tl.constexpr,  # 是否乘路由权重（编译期开关：第二层 GEMM 为 True）
    top_k: tl.constexpr,  # 每个token 的专家数：用于从槽位 id 除法还原真实 token 行号
    compute_type: tl.constexpr,  # 输出写回的目标 dtype（fp16/bf16）
    even_Ks: tl.constexpr,  # K 是否能被 BLOCK_SIZE_K 整除（决定是否需要 K 维掩码）
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.  # 用 token 矩阵与专家权重矩阵完成 MoE 的 fused 分块矩阵乘

    Key Parameters:  # 关键参数说明（原文保留，附中文摘要）
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.  # 输入激活 (num_tokens, K)
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.  # 专家权重 (E, N, K)，K 为最内维
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of
        times each token is repeated, and N is the output feature dimension.  # 输出 (M, topk, N)
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.  # 按(专家, token)排序、重复 topk 次、按 BLOCK_SIZE_M padding 的 token 索引
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.  # 每个 M 块使用哪个专家的权重

    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    # 本 kernel 按 expert_ids 为每个块选择专家权重做乘法；先按专家排序再 padding 到
    # BLOCK_SIZE_M 的整数倍，保证同一专家处理的各块 GEMM 行为一致。
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # 把一维 pid 映射为二维 tile 坐标 (pid_m, pid_n)；按 GROUP_SIZE_M 分组遍历，
    # 让同一组内的 tile 共享 A 的行数据，提升 L2 缓存命中率。
    pid = tl.program_id(axis=0)  # 当前 program 在一维网格中的编号
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)  # M 方向 tile 总数（padding 后长度除以块高，向上取整）
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)  # N 方向 tile 总数
    num_pid_in_group = GROUP_SIZE_M * num_pid_n  # 每组包含的 tile 数（GROUP_SIZE_M 行 tile x 所有 N tile）
    group_id = pid // num_pid_in_group  # 当前 pid 属于第几组
    first_pid_m = group_id * GROUP_SIZE_M  # 本组第一个 M tile 的行号
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)  # 本组实际 M tile 数（末组可能不满）
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)  # 组内蛇形/列优先映射出 M tile 坐标
    pid_n = (pid % num_pid_in_group) // group_size_m  # 组内映射出 N tile 坐标

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate  # 构造 A/B 首块的指针张量，之后沿 K 方向推进并累加
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers  # a_ptrs 是 [BLOCK_M, BLOCK_K] 的指针块
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers  # b_ptrs 是 [BLOCK_K, BLOCK_N] 的指针块
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)  # 从显存读出 padding 后有效长度（CPU 端不确定，故放设备端）
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:  # 本块完全落在 padding 区域，无有效 token
        return  # 直接退出，避免无谓计算
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # 本块在 sorted 数组中的槽位下标
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)  # 查表得到真实的 token 槽位 id（padding 槽为大数）
    offs_token = offs_token.to(tl.int64)  # 转 int64：token 数大时 32 位偏移会溢出
    token_mask = offs_token < num_valid_tokens  # 有效 token 掩码：padding 槽位（超出有效数）置 False

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N  # N 方向列偏移（取模回绕：越界列读回开头，配合掩码写回被丢弃）
    offs_k = tl.arange(0, BLOCK_SIZE_K)  # K 方向块内偏移 [0, BLOCK_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak)  # A 指针块：槽位 id 整除 top_k 还原真实 token 行号；广播成 [BLOCK_M, BLOCK_K]

    off_experts = tl.load(expert_ids_ptr + pid_m)  # 本 M 块负责的专家 id（每个 M 块共用一个专家）
    b_ptrs = (  # B 指针块：先定位专家矩阵，再按 [BLOCK_K, BLOCK_N] 广播展开
        b_ptr
        + off_experts * stride_be  # 选中的专家矩阵基址
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)  # K 行 x N 列的二维偏移
    )

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    # 沿 K 维循环计算输出块；用 fp32 累加保证数值精度，循环结束后再转回目标 dtype
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)  # fp32 累加器 [BLOCK_M, BLOCK_N]

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):  # K 维分块循环，共 cdiv(K, BLOCK_K) 步
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.  # 加载下一块 A/B；K 不整齐时按剩余长度生成掩码
        if even_Ks:  # K 整除 BLOCK_K：无需 K 掩码（更快路径）
            a = tl.load(  # 加载 A 块；无效 token 行用 0 填充
                a_ptrs,
                mask=token_mask[:, None],  # 只对有效 token 行加载
                other=0.0,
            )
            b = tl.load(b_ptrs)  # 加载 B 块（offs_bn 取模保证不越界，无需掩码）
        else:  # K 不整齐：最后一个 K 块需掩码
            a = tl.load(  # A 掩码 = 有效 token 行 且 K 列未越界
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                other=0.0,
            )
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)  # B 掩码 = K 行未越界

        # We accumulate along the K dimension.  # 沿 K 维做张量核心乘加并累加

        accumulator += tl.dot(a, b)  # 硬件矩阵乘 (BLOCK_M,BLOCK_K)x(BLOCK_K,BLOCK_N) 累加进 fp32
        # Advance the ptrs to the next K block.  # 指针推进到下一个 K 块
        a_ptrs += BLOCK_SIZE_K * stride_ak  # A 沿 K 方向前进一块
        b_ptrs += BLOCK_SIZE_K * stride_bk  # B 沿 K 方向前进一块

    if MUL_ROUTED_WEIGHT:  # 第二层 GEMM 需要乘上路由权重
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)  # 按槽位 id 取每个输出的路由权重（无效槽位为 0）
        accumulator = accumulator * moe_weight[:, None]  # 按行广播乘权重

    accumulator = accumulator.to(compute_type)  # fp32 累加结果收窄回输出 dtype（fp16/bf16）
    # -----------------------------------------------------------
    # Write back the block of the output  # 把输出块写回 C
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # 输出 N 方向的真实列下标（不取模）
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]  # C 指针块：行由槽位 id、列由 offs_cn 决定
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)  # 写回掩码：无效 token 行与越界列都不写
    tl.store(c_ptrs, accumulator, mask=c_mask)  # 掩码写回输出块
