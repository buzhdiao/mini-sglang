from .impl import DistributedCommunicator, destroy_distributed, enable_pynccl_distributed  # TP 通信门面：插件栈管理与两种通信后端
from .info import DistributedInfo, get_tp_info, set_tp_info, try_get_tp_info  # TP 并行信息（rank/size）的全局存取

__all__ = [  # 模块公开接口列表（`from minisgl.distributed import *` 时只导出这些名字）
    "DistributedInfo",  # 冻结的 TP 信息数据类（rank/size）
    "get_tp_info",  # 读取全局 TP 信息（未初始化则抛错）
    "set_tp_info",  # 初始化全局 TP 信息（仅允许设置一次）
    "enable_pynccl_distributed",  # 启用 PyNCCL 通信后端（压入插件栈）
    "DistributedCommunicator",  # 通信门面类：all_reduce / all_gather
    "try_get_tp_info",  # 尝试读取 TP 信息（未初始化返回 None，不抛错）
    "destroy_distributed",  # 销毁全部通信插件（退出前清理）
]
