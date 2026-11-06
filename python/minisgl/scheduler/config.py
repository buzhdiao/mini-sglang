# 延迟求值类型注解：注解在运行时不求值
from __future__ import annotations

# dataclass 用于生成配置类，field 用于带工厂的默认值
from dataclasses import dataclass, field

# 继承引擎配置基类，保持与 Engine 的配置协议一致
from minisgl.engine import EngineConfig


def _get_pid_suffix() -> str:
    # 函数内导入：只在真正构造配置时才需要 os，延迟导入加快模块加载
    import os

    # 返回当前进程号的字符串后缀，用于让多组调度器进程互不占用同一 IPC 通道
    return f".pid={os.getpid()}"


# 调度器配置：frozen=True 保证创建后不可变（多线程/多阶段读取安全）
@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    # 单次 forward 允许的最大"扩展"token 数（prefill chunk 大小上界）
    max_extend_tokens: int = 8192
    # 前缀缓存类型：默认基数树（radix），可复用跨请求的公共前缀 KV
    cache_type: str = "radix"
    # 离线模式：不走 ZMQ/多进程通信，直接进程内调用（便于单机测试/压测）
    offline_mode: bool = False

    # networking config
    # 网络配置：进程号后缀作为 IPC 地址的唯一化成分
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        # 调度器接收后端消息（用户请求等）的 IPC 地址；pid 后缀避免多个实例冲突
        return "ipc:///tmp/minisgl_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        # 调度器把生成的 token 推给反解码进程的 IPC 地址
        return "ipc:///tmp/minisgl_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        # TP>1 时 rank0 调度器向其余 rank 广播消息的 IPC 地址
        return "ipc:///tmp/minisgl_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        # 单次 forward 的最大 token 长度即扩展预算（prefill 分块大小）
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        # 调度器侧（后端）负责创建到反解码进程的 ZMQ 连接端点
        return True
