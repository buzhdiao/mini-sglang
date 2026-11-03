from .index import indexing  # 导入索引 kernel：按行索引（gather）权重矩阵，用于 embedding 查表等场景
from .moe_impl import fused_moe_kernel_triton, moe_sum_reduce_triton  # 导入 MoE 相关 Triton kernel 的 Python 封装： fused 分组 GEMM 与 topk 求和归约
from .pynccl import PyNCCLCommunicator, init_pynccl  # 导入 NCCL 通信封装：初始化 pynccl 通信器与对应类型（仅类型检查时可见）
from .radix import fast_compare_key  # 导入 radix cache 用的快速 key 比较函数（C++ 实现，比较两个 1-D int 张量是否相等）
from .store import store_cache  # 导入 KV Cache 写入 kernel：将当前 token 的 K/V 按索引写入 paged cache
from .tensor import test_tensor  # 导入张量测试函数：对两个张量做比较（多用于测试/验证 FFI 通路）

__all__ = [  # 模块对外导出的公开 API 列表（from minisgl.kernel import * 时可见）
    "indexing",  # 索引（行 gather）kernel
    "fast_compare_key",  # radix key 快速比较
    "store_cache",  # KV cache 写入
    "test_tensor",  # 张量 FFI 测试
    "init_pynccl",  # NCCL 通信器初始化
    "PyNCCLCommunicator",  # NCCL 通信器类型（仅静态检查用，运行时为 Any）
    "fused_moe_kernel_triton",  # MoE fused GEMM 封装
    "moe_sum_reduce_triton",  # MoE topk 输出求和归约封装
]
