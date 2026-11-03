from .backend import AbortBackendMsg, BaseBackendMsg, BatchBackendMsg, ExitMsg, UserMsg  # 后端(调度器/引擎)侧消息：用户请求、中止、退出及批量包装
from .frontend import BaseFrontendMsg, BatchFrontendMsg, UserReply  # 前端(API 服务)侧消息：用户回复及批量包装
from .tokenizer import AbortMsg, BaseTokenizerMsg, BatchTokenizerMsg, DetokenizeMsg, TokenizeMsg  # 分词器侧消息：编码请求、解码结果、中止及批量包装

__all__ = [  # 声明模块公开 API：按侧归类导出全部消息类型
    "AbortMsg",  # 分词器侧：中止请求
    "AbortBackendMsg",  # 后端侧：中止请求
    "BaseBackendMsg",  # 后端侧消息基类（提供编解码方法）
    "BatchBackendMsg",  # 后端侧：批量消息包装
    "ExitMsg",  # 后端侧：退出信号
    "UserMsg",  # 后端侧：用户请求（token id + 采样参数）
    "BaseTokenizerMsg",  # 分词器侧消息基类
    "BatchTokenizerMsg",  # 分词器侧：批量消息包装
    "DetokenizeMsg",  # 分词器侧：解码结果（新 token）
    "TokenizeMsg",  # 分词器侧：编码请求
    "BaseFrontendMsg",  # 前端侧消息基类
    "BatchFrontendMsg",  # 前端侧：批量消息包装
    "UserReply",  # 前端侧：发给用户的增量回复
]
