from __future__ import annotations  # 启用延迟注解求值，允许前向引用类型

from dataclasses import dataclass  # dataclass 装饰器：自动生成样板方法
from typing import Dict, List  # 类型注解工具

import torch  # PyTorch：UserMsg 携带 CPU 张量形式的 input_ids
from minisgl.core import SamplingParams  # 采样参数：随用户消息一起传给调度器

from .utils import deserialize_type, serialize_type  # 序列化工具：对象 <-> 可 JSON 传输的字典


@dataclass  # 数据类：后端侧消息基类
class BaseBackendMsg:
    """后端侧消息基类：实例方法 encoder 把自身序列化，静态 decoder 反序列化"""

    def encoder(self) -> Dict:  # 实例方法：序列化自身
        return serialize_type(self)  # 借助工具函数把对象字段转为带 __type__ 标记的字典

    @staticmethod  # 静态方法：不依赖实例，作为类级反序列化入口
    def decoder(json: Dict) -> BaseBackendMsg:
        return deserialize_type(globals(), json)  # 用本模块全局命名空间做类名->类 的查找表还原对象


@dataclass  # 数据类：批量包装
class BatchBackendMsg(BaseBackendMsg):
    data: List[BaseBackendMsg]  # 多条后端消息打成一个包，减少通信轮次


@dataclass  # 数据类：退出信号
class ExitMsg(BaseBackendMsg):
    pass  # 无字段：仅表示"结束进程/服务"这一事件本身


@dataclass  # 数据类：用户请求消息
class UserMsg(BaseBackendMsg):
    uid: int  # 请求唯一 id
    input_ids: torch.Tensor  # CPU 1D int32 tensor  # 已分词的输入 id（CPU 一维 int32 张量，避免传文本）
    sampling_params: SamplingParams  # 该请求的采样参数


@dataclass  # 数据类：中止请求消息
class AbortBackendMsg(BaseBackendMsg):
    uid: int  # 要中止的请求 id
