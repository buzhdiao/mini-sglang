from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解中使用前向引用

from dataclasses import dataclass  # 数据类装饰器：自动生成 __init__/__repr__/__eq__


@dataclass(frozen=True)  # 冻结数据类：创建后字段不可修改，保证 TP 信息全局一致性
class DistributedInfo:  # should not export from here  TP 并行信息（rank/size），不应在本模块之外导出或流转实例
    rank: int  # 当前进程在 TP 组内的序号（从 0 开始）
    size: int  # TP 组的总进程数（张量并行度）

    def __post_init__(self):
        """构造后校验：rank 必须落在 [0, size) 区间内，尽早暴露非法配置"""
        assert 0 <= self.rank < self.size

    def is_primary(self) -> bool:
        """是否为主 rank（rank0）——通常只有 rank0 负责打印日志、接收客户端请求"""
        return self.rank == 0


_TP_INFO: DistributedInfo | None = None  # 模块级全局单例：保存当前进程的 TP 信息（每个进程一份）


def set_tp_info(rank: int, size: int) -> None:
    """初始化全局 TP 信息（每个进程只允许设置一次，重复设置视为逻辑 bug）"""
    global _TP_INFO  # 声明修改模块级变量
    if _TP_INFO is not None:  # 已设置过说明初始化流程重复执行
        raise RuntimeError("TP info has been set")
    _TP_INFO = DistributedInfo(rank, size)  # 构造并写入全局单例（__post_init__ 会校验合法性）


def get_tp_info() -> DistributedInfo:
    """读取全局 TP 信息；未初始化时抛错（调用方需保证初始化已完成）"""
    if _TP_INFO is None:  # 尚未初始化
        raise RuntimeError("TP info has not been set")
    return _TP_INFO  # 返回全局单例


def try_get_tp_info() -> DistributedInfo | None:
    """尝试读取全局 TP 信息：未初始化时返回 None 而非抛错（用于可选场景，如日志格式化）"""
    return _TP_INFO  # 直接返回当前值，可能为 None


__all__ = ["DistributedInfo", "set_tp_info", "get_tp_info", "try_get_tp_info"]  # 模块公开接口
