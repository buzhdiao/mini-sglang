# 延迟类型注解：注解在运行时不求值
from __future__ import annotations

# dataclass 自动生成结构体方法
from dataclasses import dataclass
from typing import Dict, List

# 通用序列化工具
from .utils import deserialize_type, serialize_type


@dataclass
class BaseFrontendMsg:
    # 发往前端（对外服务）的消息基类：与 backend 侧类似但用静态方法编码
    @staticmethod
    def encoder(msg: BaseFrontendMsg) -> Dict:
        # 把消息序列化成 JSON 兼容字典
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseFrontendMsg:
        # 从字典反序列化：在当前模块 globals() 里按类型名重建子类实例
        return deserialize_type(globals(), json)


@dataclass
class BatchFrontendMsg(BaseFrontendMsg):
    # 批量消息：把多条回复打包成一个 ZMQ 帧传输，降低通信开销
    data: List[BaseFrontendMsg]


@dataclass
class UserReply(BaseFrontendMsg):
    # 用户回复消息：增量文本 + 是否生成结束
    uid: int
    incremental_output: str
    finished: bool
