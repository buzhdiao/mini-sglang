# 延迟类型注解：注解在运行时不求值
from __future__ import annotations

# dataclass 生成请求状态结构
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
# 采样参数定义
from minisgl.core import SamplingParams
# 单卡离线模式的分布式信息
from minisgl.distributed import DistributedInfo
# 前后端消息类型：用户请求与逐 token 解码结果
from minisgl.message import (
    BaseBackendMsg,
    DetokenizeMsg,
    UserMsg,
)
# 调度器及其配置：LLM 类直接继承调度器
from minisgl.scheduler import Scheduler, SchedulerConfig


class RequestAllFinished(Exception):
    # 离线模式下所有请求都完成时抛出的控制流异常，用于跳出调度主循环
    pass


@dataclass
class RequestStatus:
    # 单个请求的跟踪状态：输入 token、已生成的输出 token
    uid: int
    input_ids: List[int]
    output_ids: List[int]


class LLM(Scheduler):
    # 离线批推理入口：复用调度器主循环，但用内存队列替代网络收发，
    # 提供类似 OpenAI/HF 的 generate() 同步接口
    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, **kwargs):
        # 构造调度器配置：单进程单卡（rank 0 / tp 1）、离线模式
        config = SchedulerConfig(
            model_path=model_path,
            tp_info=DistributedInfo(0, 1),
            dtype=dtype,
            offline_mode=True,
            **kwargs,
        )
        # 初始化底层调度器（加载模型、KV cache、tokenize 环境等）
        super().__init__(config)
        # 待提交请求队列：(prompt 文本或 token 列表, 采样参数)
        self.pending_requests: List[Tuple[List[int] | str, SamplingParams]] = []
        # uid -> 请求状态映射
        self.status_map: Dict[int, RequestStatus] = {}
        # 全局 uid 计数器
        self.counter = 0

    def _tokenize_one(self, prompt: List[int] | str) -> torch.Tensor:
        # 把单条 prompt 统一成 int32 的 token 张量
        if isinstance(prompt, str):
            # 文本输入：编码成 token 并转 int32
            return self.tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)
        else:
            # 已是 token id 列表：直接转张量
            return torch.tensor(prompt, dtype=torch.int32, device="cpu")

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        # 离线模式的"收消息"：从内部队列取出一批请求喂给调度器
        if blocking and len(self.pending_requests) == 0:
            # 阻塞语义下队列空即视为全部完成，向上抛异常结束主循环
            raise RequestAllFinished()
        results: List[BaseBackendMsg] = []
        # added：本轮提交的请求数；sum_input_len：累计输入 token 数
        added, sum_input_len = 0, 0
        for tokens_or_prompt, sampling_params in self.pending_requests:
            if sum_input_len >= self.prefill_budget:
                # 超过单次预填充预算就停下，剩余请求下一轮再提交（模拟分批调度）
                break
            # tokenize 得到输入 token
            input_ids = self._tokenize_one(tokens_or_prompt)
            sum_input_len += len(input_ids)
            # 分配连续 uid
            uid, added = self.counter + added, added + 1
            # 包装成调度器能识别的用户消息
            results.append(UserMsg(uid=uid, input_ids=input_ids, sampling_params=sampling_params))
            # 登记该请求的跟踪状态
            self.status_map[uid] = RequestStatus(
                uid=uid,
                input_ids=(
                    input_ids.tolist() if isinstance(tokens_or_prompt, str) else tokens_or_prompt
                ),
                output_ids=[],
            )
        # 推进计数器并从队列里移除已提交的请求
        self.counter += added
        self.pending_requests = self.pending_requests[added:]
        return results

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        # 离线模式的"发结果"：接收调度器产出的逐 token 消息，累计到对应请求状态
        for msg in reply:
            status = self.status_map[msg.uid]
            if not (msg.finished and msg.next_token == self.eos_token_id):
                # 结束消息若是 EOS 则不计入输出（EOS 只是终止信号，不是生成内容）
                status.output_ids.append(msg.next_token)

    def generate(
        self,
        prompts: List[str] | List[List[int]],
        sampling_params: List[SamplingParams] | SamplingParams,
    ) -> List[Dict[str, str | List[int]]]:
        # 同步批量生成：提交全部请求、驱动调度循环直到所有请求完成、再汇总结果
        self.pending_requests = []
        self.status_map = {}
        self.counter = 0
        if isinstance(sampling_params, SamplingParams):
            # 传入单个采样参数时对每条 prompt 复制一份
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.pending_requests.append((prompt, sp))
        try:
            # 运行调度器主循环，内部反复调用 offline_receive/offline_send_result
            self.run_forever()
        except RequestAllFinished:
            # 全部请求生成完毕：正常退出循环
            pass
        results: List[Dict[str, str | List[int]]] = []
        for i in range(len(prompts)):
            # 按 uid 顺序收集每个请求的输出
            status = self.status_map[i]
            # 把输出 token 解码回文本
            output_text = self.tokenizer.decode(status.output_ids)
            results.append({"text": output_text, "token_ids": status.output_ids})
        return results
