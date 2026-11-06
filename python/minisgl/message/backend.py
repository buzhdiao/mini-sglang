# 延迟类型注解：注解在运行时不求值
from __future__ import annotations

# dataclass 自动生成 __init__/__repr__ 等结构体方法
from dataclasses import dataclass
from typing import Dict, List

import torch
# 采样参数：随用户请求一起传输
from minisgl.core import SamplingParams

# 通用 dataclass <-> JSON 序列化工具
from .utils import deserialize_type, serialize_type


@dataclass
class BaseBackendMsg:
    # 发往后端（调度器）的消息基类：实例方法编码，类方法解码
    def encoder(self) -> Dict:
        # 把自身序列化成 JSON 兼容字典
        return serialize_type(self)

    @staticmethod
    def decoder(json: Dict) -> BaseBackendMsg:
        # 从字典反序列化：在当前模块 globals() 里按类型名查找并重建子类实例
        return deserialize_type(globals(), json)


@dataclass
class BatchBackendMsg(BaseBackendMsg):
    # 批量消息：把多条消息打包成一个 ZMQ 帧传输，减少消息条数与系统调用开销
    data: List[BaseBackendMsg]


@dataclass
class ExitMsg(BaseBackendMsg):
    # 退出消息：通知后端进程结束（空载荷，类型即语义）
    pass


@dataclass
class UserMsg(BaseBackendMsg):
    # 用户请求消息：分词后的输入与采样参数
    uid: int
    input_ids: torch.Tensor  # CPU 1D int32 tensor
    sampling_params: SamplingParams


@dataclass
class AbortBackendMsg(BaseBackendMsg):
    # 中止消息：按 uid 取消一个正在处理的请求
    uid: int
