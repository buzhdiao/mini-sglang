from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解中使用前向引用（如 torch.Tensor | None）

import functools  # 提供 functools.cache 装饰器，用于缓存 JIT 编译结果，避免同一配置重复编译
from typing import TYPE_CHECKING, Tuple  # TYPE_CHECKING：仅静态检查时 import；Tuple：词表范围类型注解

from .utils import KernelConfig, load_jit, make_cpp_args  # 导入 kernel 工具：启动配置、JIT 编译入口、C++ 模板参数转换

if TYPE_CHECKING:  # 仅在类型检查（mypy/pyright）时导入，运行时不产生依赖开销
    import torch  # 张量库，仅用于类型注解
    from tvm_ffi import Module  # tvm-ffi 编译产出的动态库模块类型

# 默认 kernel 启动配置：128 线程/block、限制 1 倍最大占用率、不使用 PDL（Programmatic Dependent Launch）
DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)


@functools.cache  # 缓存编译结果：相同 (element_size, num_splits, config) 只编译一次，后续直接命中缓存
def _jit_index_module(
    element_size: int,
    *,
    num_splits: int = 1,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    """JIT 编译 index kernel 的动态库。

    按每行字节数 element_size 与 split 数量实例化 C++ 模板 IndexKernel<...>，
    相同模板参数只编译一次（functools.cache 缓存）。
    """
    # 把整型模板参数转成 C++ 模板实参字符串列表，如 ["2048", "4", "128", "1", "false"]
    args = make_cpp_args(element_size, num_splits, *config)
    # 以 "index" 为名做 JIT inline 编译：源码为 jit/index.cu，并生成导出包装函数 launch -> IndexKernel<args>::run
    return load_jit(
        "index",  # 模块名（会参与生成唯一库名，避免不同模板实例的缓存冲突）
        *args,  # 模板参数，同样拼进模块名保证唯一性
        cuda_files=["index.cu"],  # CUDA 源文件（位于 kernel/jit 目录）
        cuda_wrappers=[("launch", f"IndexKernel<{args}>::run")],  # 生成 TVM_FFI 导出：launch 函数绑定到模板静态方法 run
    )


def indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    vocab_range: Tuple[int, int] | None = None,  # (start, length)
) -> torch.Tensor:
    """按索引收集（gather）权重行：output[i] = weights[indices[i]]。

    用于 embedding 查表 / logprob 提取等场景；支持可选词表范围偏移，
    且根据每行字节数自动选择 split 数以提升并行度。
    """
    if output is None:  # 未传入输出张量时，按索引数量 x 权重行宽分配新的输出
        output = weights.new_empty(indices.shape[0], weights.shape[1])

    # 每行的字节数 = 行内元素数 x 单元素字节数，作为模板参数让 kernel 按字节向量宽度展开
    element_size = weights.shape[1] * weights.element_size()
    if element_size % 2048 == 0:  # 行宽能被 2048 字节整除 -> 每行拆 4 份并行处理
        num_splits = 4
    elif element_size % 1024 == 0:  # 行宽能被 1024 字节整除 -> 每行拆 2 份
        num_splits = 2
    else:  # 否则不拆分（保证每份是完整向量对齐，避免越界/非对齐访问）
        num_splits = 1
    module = _jit_index_module(element_size, num_splits=num_splits)  # 获取（或编译并缓存）对应模板实例的模块
    module.launch(weights, indices, output, vocab_range)  # 调用编译好的 CUDA kernel：按 indices 从 weights 拷贝行到 output
    return output  # 返回结果张量（外部传入 output 时即原对象）
