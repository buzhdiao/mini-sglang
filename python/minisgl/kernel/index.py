# 允许类型注解使用字符串形式的前向引用
from __future__ import annotations

# functools.cache 缓存 JIT 编译结果，同一配置只编译一次
import functools
# 类型注解工具
from typing import TYPE_CHECKING, Tuple

# JIT 编译工具：内核配置、加载器与 C++ 模板参数拼接
from .utils import KernelConfig, load_jit, make_cpp_args

# 仅类型检查时导入 torch 与 Module
if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module

# index 内核默认配置：128 线程/块、占用率 1、不启用 PDL
DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)


# 按元素字节数等参数 JIT 编译 index 内核模块（结果按参数缓存）。
# num_splits：一行数据切几份并行处理；config：内核启动配置
@functools.cache
def _jit_index_module(
    element_size: int,
    *,
    num_splits: int = 1,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    # 把模板参数拼成 C++ 模板实参字符串
    args = make_cpp_args(element_size, num_splits, *config)
    # 编译并加载 index.cu，暴露 launch 入口
    return load_jit(
        "index",
        *args,
        cuda_files=["index.cu"],
        cuda_wrappers=[("launch", f"IndexKernel<{args}>::run")],
    )


# 按行索引 gather：weights[indices[i]] -> output[i]（用于 embedding 查表/取 KV 行等）。
# output 为可选的预分配输出；vocab_range=(start, length) 限定索引有效区间，越界行写 0
def indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    vocab_range: Tuple[int, int] | None = None,  # (start, length)
) -> torch.Tensor:
    # 未给输出则按行数 x 列数分配
    if output is None:
        # 与 weights 同 dtype 的空输出
        output = weights.new_empty(indices.shape[0], weights.shape[1])

    # 一行的字节数：决定向量化宽度与 split 策略
    element_size = weights.shape[1] * weights.element_size()
    # 行足够宽（>=4 段 2048B 对齐）时用 4 个 split 提升并行度
    if element_size % 2048 == 0:
        # 4 路切分
        num_splits = 4
    elif element_size % 1024 == 0:
        # 2 路切分
        num_splits = 2
    else:
        # 无法对齐切分则单路处理
        num_splits = 1
    # 取（或编译）对应配置的内核模块
    module = _jit_index_module(element_size, num_splits=num_splits)
    # 启动内核执行 gather
    module.launch(weights, indices, output, vocab_range)
    # 返回结果
    return output
