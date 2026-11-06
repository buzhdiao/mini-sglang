# 延迟求值类型注解：注解在运行时不求值
from __future__ import annotations

# functools.cache：进程内缓存检测结果（设备能力不会变）
import functools
# 类型注解工具
from typing import Tuple


@functools.cache
def _get_torch_cuda_version() -> Tuple[int, int] | None:
    # 函数内导入：只在首次调用时加载 torch
    import torch
    import torch.version

    if not torch.cuda.is_available() or not torch.version.cuda:
        # 无 CUDA 环境（CPU-only 或 CUDA 未编译）：返回 None 表示不支持
        return None
    # 返回当前 GPU 的算力版本 (major, minor)，如 (9, 0) 表示 Hopper SM90
    return torch.cuda.get_device_capability()


def is_arch_supported(major: int, minor: int = 0) -> bool:
    # 取设备算力版本
    arch = _get_torch_cuda_version()
    if arch is None:
        # 无 CUDA：一律不支持
        return False
    # 元组比较：设备算力 >= 要求的最低算力才支持
    return arch >= (major, minor)


def is_sm90_supported() -> bool:
    # Hopper（H100 等，SM90）是否支持：决定 FlashAttention3/TRT-LLM 等内核可用性
    return is_arch_supported(9, 0)


def is_sm100_supported() -> bool:
    # Blackwell（B200 等，SM100）是否支持
    return is_arch_supported(10, 0)
