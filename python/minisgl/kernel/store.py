from __future__ import annotations  # 启用 PEP 563 延迟注解求值，支持前向引用类型注解

import functools  # 提供 functools.cache，缓存 JIT 编译结果
from typing import TYPE_CHECKING  # 仅静态类型检查时导入重类型

from .utils import KernelConfig, load_jit, make_cpp_args  # kernel 工具：启动配置、JIT 编译入口、C++ 模板参数转换

if TYPE_CHECKING:  # 仅类型检查时导入，运行时无开销
    import torch  # 张量库，仅用于注解
    from tvm_ffi import Module  # tvm-ffi 动态库模块类型

# 默认 kernel 启动配置：128 线程/block、限制 1 倍最大占用率、不使用 PDL（与 index kernel 保持一致）
DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)


@functools.cache  # 缓存编译结果：相同 (element_size, config) 的模板实例只编译一次
def _jit_store_module(
    element_size: int,
    *,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    """JIT 编译 store kernel 的动态库。

    按每行字节数 element_size 实例化 C++ 模板 StoreKernel<...>，
    相同模板参数只编译一次（functools.cache 缓存）。
    """
    # 把整型模板参数转成 C++ 模板实参字符串列表
    args = make_cpp_args(element_size, *config)
    # 以 "store" 为名做 JIT inline 编译：源码为 jit/store.cu，导出包装 launch -> StoreKernel<args>::run
    return load_jit(
        "store",  # 模块名（参与生成唯一库名）
        *args,  # 模板参数拼进模块名避免不同实例缓存冲突
        cuda_files=["store.cu"],  # CUDA 源文件（位于 kernel/jit 目录）
        cuda_wrappers=[("launch", f"StoreKernel<{args}>::run")],  # 生成 TVM_FFI 导出绑定
    )


def store_cache(
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indices: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """将当前 token 的 K/V 写入 paged KV cache：cache[indices[i]] = k[i]/v[i]。

    cache 视作 (num_tokens, -1) 的二维布局，按行索引写入（scatter）。
    """
    num_tokens = k_cache.shape[0]  # cache 的槽数（页数 x 页大小）
    k_cache = k_cache.view(num_tokens, -1)  # 把 K cache 摊平成二维：每行是一个槽位的完整 KV 数据
    v_cache = v_cache.view(num_tokens, -1)  # 同样摊平 V cache
    element_size = k_cache.shape[1] * k_cache.element_size()  # 每行字节数，作为模板参数让 kernel 向量化拷贝
    module = _jit_store_module(element_size)  # 获取（或编译并缓存）对应模板实例
    module.launch(k_cache, v_cache, indices, k, v)  # 启动 CUDA kernel：按 indices 把 k/v 行 scatter 写入 cache
