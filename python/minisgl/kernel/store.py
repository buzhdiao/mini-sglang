# 允许类型注解使用字符串形式的前向引用
from __future__ import annotations

# functools.cache 缓存 JIT 编译结果
import functools
# 类型注解工具
from typing import TYPE_CHECKING

# JIT 编译工具
from .utils import KernelConfig, load_jit, make_cpp_args

# 仅类型检查时导入
if TYPE_CHECKING:
    import torch
    from tvm_ffi import Module

# store 内核默认配置：128 线程/块、占用率 1、不启用 PDL
DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)


# 按 KV 行字节数等参数 JIT 编译 store 内核模块（结果按参数缓存）。
# config：内核启动配置
@functools.cache
def _jit_store_module(
    element_size: int,
    *,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    # 把模板参数拼成 C++ 模板实参字符串
    args = make_cpp_args(element_size, *config)
    # 编译并加载 store.cu，暴露 launch 入口
    return load_jit(
        "store",
        *args,
        cuda_files=["store.cu"],
        cuda_wrappers=[("launch", f"StoreKernel<{args}>::run")],
    )


# 把本层新产生的 K/V 写入分页 KV cache：indices 指定每个 token 的目标槽位
def store_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    # cache 的总 token 行数
    num_tokens = k_cache.shape[0]
    # 把 cache 摊平成 (tokens, 每行字节数/元素) 视图，方便按行寻址（不拷贝）
    k_cache = k_cache.view(num_tokens, -1)
    # V 同样摊平
    v_cache = v_cache.view(num_tokens, -1)
    # 单行 K 的字节数，决定向量化宽度
    element_size = k_cache.shape[1] * k_cache.element_size()
    # 取（或编译）对应配置的内核模块
    module = _jit_store_module(element_size)
    # 启动内核：按 indices 把 k/v 的每一行散写进 cache
    module.launch(k_cache, v_cache, indices, k, v)
