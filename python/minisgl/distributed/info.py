# 允许类型注解使用字符串形式的前向引用
from __future__ import annotations

# dataclass 声明信息类
from dataclasses import dataclass


# TP 进程组内本进程的信息（frozen：创建后不可改）
@dataclass(frozen=True)
class DistributedInfo:  # should not export from here
    # 本进程在 TP 组中的序号（0 起始）
    rank: int
    # TP 组的进程总数
    size: int

    # 构造后校验合法性
    def __post_init__(self):
        # rank 必须落在 [0, size) 内
        assert 0 <= self.rank < self.size

    # 是否为主进程（rank 0）：负责日志/分发等单例职责
    def is_primary(self) -> bool:
        # rank 0 即主
        return self.rank == 0


# 模块级单例：当前进程的 TP 信息
_TP_INFO: DistributedInfo | None = None


# 设置 TP 信息（每个进程只能设置一次）
def set_tp_info(rank: int, size: int) -> None:
    # 声明修改模块级变量
    global _TP_INFO
    # 重复设置视为编程错误
    if _TP_INFO is not None:
        # 抛出运行时错误
        raise RuntimeError("TP info has been set")
    # 构造并保存
    _TP_INFO = DistributedInfo(rank, size)


# 读取 TP 信息（必须已设置）
def get_tp_info() -> DistributedInfo:
    # 未设置时直接报错，避免下游拿到 None
    if _TP_INFO is None:
        # 抛出运行时错误
        raise RuntimeError("TP info has not been set")
    # 返回单例
    return _TP_INFO


# 尝试读取 TP 信息（允许未设置，返回 None）
def try_get_tp_info() -> DistributedInfo | None:
    # 原样返回（可能为 None）
    return _TP_INFO


# 声明模块公共 API
__all__ = ["DistributedInfo", "set_tp_info", "get_tp_info", "try_get_tp_info"]
