from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解中使用前向引用

import functools  # 提供 cache 装饰器，用于缓存 GPU 能力探测结果
from typing import Tuple  # 类型注解：元组


@functools.cache  # 结果缓存：设备计算能力在进程生命周期内不变，只需探测一次
def _get_torch_cuda_version() -> Tuple[int, int] | None:
    """探测当前 CUDA 设备的计算能力 (major, minor)，如 (9, 0) 表示 SM90；无可用 GPU/CUDA 时返回 None"""
    import torch  # 延迟导入 torch，加快模块导入速度
    import torch.version  # 显式导入子模块，确保 torch.version.cuda 属性可用

    if not torch.cuda.is_available() or not torch.version.cuda:  # 没有 GPU，或 PyTorch 是非 CUDA 构建
        return None  # 返回 None 表示架构检测不可用
    return torch.cuda.get_device_capability()  # 返回 (主版本号, 次版本号)，如 H100 返回 (9, 0)


def is_arch_supported(major: int, minor: int = 0) -> bool:
    """判断当前 GPU 计算能力是否 >= 指定架构（如 SM90）"""
    arch = _get_torch_cuda_version()  # 取缓存的设备计算能力
    if arch is None:  # 没有 CUDA 设备，一律视为不支持
        return False
    return arch >= (major, minor)  # 元组按字典序比较：先比主版本，再比次版本


def is_sm90_supported() -> bool:
    """是否支持 SM90（Hopper，如 H100）——用于决定是否启用 FlashInfer 等新特性"""
    return is_arch_supported(9, 0)  # 要求计算能力 >= 9.0


def is_sm100_supported() -> bool:
    """是否支持 SM100（Blackwell，如 B200）"""
    return is_arch_supported(10, 0)  # 要求计算能力 >= 10.0
