# 延迟求值类型注解：注解在运行时不求值
from __future__ import annotations

# dataclass 生成数据类
from dataclasses import dataclass
# 类型注解工具
from typing import TYPE_CHECKING, List

import torch

# 仅类型检查导入，避免运行时循环依赖
if TYPE_CHECKING:
    # 采样参数类型
    from minisgl.core import SamplingParams

    # 分块请求类型
    from .prefill import ChunkedReq


# 等待调度的用户请求（调度器内部的轻量表示，区别于正式的 Req）
@dataclass
class PendingReq:
    # 请求唯一 id
    uid: int
    # 完整输入 token 序列（CPU 张量）
    input_ids: torch.Tensor
    # 采样参数
    sampling_params: SamplingParams
    # 分块续传引用：若上一轮 prefill 没喂完，指向挂起的 ChunkedReq
    chunked_req: ChunkedReq | None = None

    @property
    def input_len(self) -> int:
        # 输入 token 总数
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        # 期望生成的最大输出 token 数
        return self.sampling_params.max_tokens


# 一次调度的结果（预留结构：请求列表与对应的输出槽位索引）
@dataclass
class ScheduleResult:
    # 参与调度的请求
    reqs: List[PendingReq]
    # 每个请求输出 token 要写入的槽位索引
    output_indices: List[torch.Tensor]
