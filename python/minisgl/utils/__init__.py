from .arch import is_arch_supported, is_sm90_supported, is_sm100_supported  # GPU 架构（SM 版本）支持检测
from .hf import cached_load_hf_config, download_hf_weight, load_tokenizer  # HuggingFace 配置/权重/分词器加载与下载
from .logger import init_logger  # 带彩色格式的日志初始化
from .misc import UNSET, Unset, align_ceil, align_down, call_if_main, div_ceil, div_even  # 杂项工具：哨兵值、整除/对齐运算、main 守卫装饰器
from .mp import (  # 基于 ZMQ + msgpack 的多进程消息队列封装
    ZmqAsyncPullQueue,  # 异步拉取队列（PULL 端，配合 asyncio）
    ZmqAsyncPushQueue,  # 异步推送队列（PUSH 端，配合 asyncio）
    ZmqPubQueue,  # 发布队列（PUB 端，一对多广播）
    ZmqPullQueue,  # 同步拉取队列（PULL 端，阻塞接收）
    ZmqPushQueue,  # 同步推送队列（PUSH 端，一对一发送）
    ZmqSubQueue,  # 订阅队列（SUB 端，接收广播）
)
from .registry import Registry  # 通用注册器：按名字注册/查找可插拔组件（类或函数）
from .torch_utils import nvtx_annotate, torch_dtype  # PyTorch 工具：默认精度上下文、NVTX 性能分析标注

__all__ = [  # 模块公开接口列表（`from minisgl.utils import *` 时只导出这些名字）
    "cached_load_hf_config",  # 读取并缓存 HF 模型配置
    "download_hf_weight",  # 下载模型权重到本地缓存
    "load_tokenizer",  # 加载分词器（含 Mistral 聊天模板兼容处理）
    "init_logger",  # 初始化彩色 logger
    "is_arch_supported",  # 判断 GPU 架构是否 >= 指定 SM 版本
    "is_sm90_supported",  # 是否支持 SM90（Hopper，如 H100）
    "is_sm100_supported",  # 是否支持 SM100（Blackwell，如 B200）
    "call_if_main",  # 仅主模块运行时执行函数的装饰器
    "div_even",  # 要求整除的除法（支持 KV 头复制语义）
    "div_ceil",  # 向上取整除法
    "align_ceil",  # 向上对齐到 b 的整数倍
    "align_down",  # 向下对齐到 b 的整数倍
    "UNSET",  # 全局唯一"未设置"哨兵实例
    "Unset",  # "未设置"哨兵类
    "torch_dtype",  # 临时切换 PyTorch 默认精度的上下文管理器
    "nvtx_annotate",  # NVTX 性能分析区间标注装饰器
    "Registry",  # 通用名字注册器
    "ZmqPushQueue",  # 同步 PUSH 队列
    "ZmqPullQueue",  # 同步 PULL 队列
    "ZmqPubQueue",  # 发布（广播）队列
    "ZmqSubQueue",  # 订阅队列
    "ZmqAsyncPushQueue",  # 异步 PUSH 队列
    "ZmqAsyncPullQueue",  # 异步 PULL 队列
]
