from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解中使用前向引用（ChunkedReq | None）

from dataclasses import dataclass  # dataclass 装饰器：自动生成 __init__ 等样板方法
from typing import TYPE_CHECKING, List  # 类型注解工具：TYPE_CHECKING（仅类型检查期导入）、List

import torch  # PyTorch：input_ids 的张量类型

if TYPE_CHECKING:  # 仅静态类型检查时导入，避免运行时循环依赖
    from minisgl.core import SamplingParams  # 采样参数类型

    from .prefill import ChunkedReq  # 分块请求类型


@dataclass
class PendingReq:
    """调度器内部的待处理请求：用户消息的中间表示，附带分块续接状态"""

    uid: int  # 请求唯一 id：用于消息路由与中止匹配
    input_ids: torch.Tensor  # 完整输入 token 序列（CPU）
    sampling_params: SamplingParams  # 该请求的采样参数
    chunked_req: ChunkedReq | None = None  # 上轮被分块截断的 Req：非 None 时本轮优先续接 prefill

    @property
    def input_len(self) -> int:
        return len(self.input_ids)  # 输入 token 总数

    @property
    def output_len(self) -> int:
        return self.sampling_params.max_tokens  # 期望生成的输出 token 数


@dataclass
class ScheduleResult:
    """调度结果（预留的数据结构）：本轮接纳的请求列表及其输出位置索引"""

    reqs: List[PendingReq]  # 本轮接纳的待处理请求
    output_indices: List[torch.Tensor]  # 各请求的输出 token 写入索引
