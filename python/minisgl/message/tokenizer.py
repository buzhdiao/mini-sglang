# 延迟类型注解：注解在运行时不求值
from __future__ import annotations

# dataclass 自动生成结构体方法
from dataclasses import dataclass
from typing import Dict, List

# 采样参数：随 tokenize 请求一起传输
from minisgl.core import SamplingParams

# 通用序列化工具
from .utils import deserialize_type, serialize_type


@dataclass
class BaseTokenizerMsg:
    # 分词器进程消息基类：分词器夹在前端与后端之间，
    # 负责把文本编码成 token（发给后端）、把后端的 token 解码回文本（发给前端）
    @staticmethod
    def encoder(msg: BaseTokenizerMsg) -> Dict:
        # 把消息序列化成 JSON 兼容字典
        return serialize_type(msg)

    @staticmethod
    def decoder(json: Dict) -> BaseTokenizerMsg:
        # 从字典反序列化：在当前模块 globals() 里按类型名重建子类实例
        return deserialize_type(globals(), json)


@dataclass
class BatchTokenizerMsg(BaseTokenizerMsg):
    # 批量消息：多条消息打包成一个 ZMQ 帧传输，减少通信次数
    data: List[BaseTokenizerMsg]


@dataclass
class DetokenizeMsg(BaseTokenizerMsg):
    # 解码结果消息：后端产出的一个新 token，由分词器转成文本后下发前端
    uid: int
    next_token: int
    finished: bool


@dataclass
class TokenizeMsg(BaseTokenizerMsg):
    # 编码请求消息：text 可以是纯文本，也可以是对话模板消息列表
    uid: int
    text: str | List[Dict[str, str]]
    sampling_params: SamplingParams


@dataclass
class AbortMsg(BaseTokenizerMsg):
    # 中止消息：按 uid 取消一个请求（分词器转发给后端）
    uid: int
