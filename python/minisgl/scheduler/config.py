from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解中使用前向引用

from dataclasses import dataclass, field  # dataclass 装饰器与 field：声明数据类及带默认工厂的字段

from minisgl.engine import EngineConfig  # 引擎配置基类：含 tp_info、page_size、max_running_req 等通用字段


def _get_pid_suffix() -> str:
    # 生成进程唯一后缀 ".pid=<pid>"：避免同机多实例的 IPC 文件路径冲突
    import os  # 函数内延迟导入：仅构造配置时才需要

    return f".pid={os.getpid()}"  # 用当前进程 PID 保证唯一性


@dataclass(frozen=True)  # frozen=True：实例不可变，配置创建后禁止修改（防运行期误改）
class SchedulerConfig(EngineConfig):
    """调度器配置：prefill 预算、缓存类型、运行模式与各 ZMQ IPC 端点地址"""

    max_extend_tokens: int = 8192  # 单次 prefill 前向最多计算的 token 数（同时作为 prefill 预算与单次前向上限）
    cache_type: str = "radix"  # 前缀缓存实现类型（radix tree）
    offline_mode: bool = False  # 离线模式：不走 ZMQ 消息收发（由上层驱动直接调用）

    # networking config  # 网络配置
    _unique_suffix: str = field(default_factory=_get_pid_suffix)  # IPC 路径唯一后缀，默认按 PID 生成

    @property
    def zmq_backend_addr(self) -> str:
        # tokenizer -> scheduler 的请求接收地址（PULL 端点，由主 rank bind）
        return "ipc:///tmp/minisgl_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        # scheduler -> detokenizer 的结果发送地址（PUSH 端点）
        return "ipc:///tmp/minisgl_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        # rank0 -> 其他 TP rank 的请求广播地址（PUB/SUB 端点）
        return "ipc:///tmp/minisgl_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        # 单次前向的最大 token 数：直接复用 prefill 扩展预算
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        # 调度器侧负责创建（bind）与 detokenizer 的 ZMQ 连接
        return True
