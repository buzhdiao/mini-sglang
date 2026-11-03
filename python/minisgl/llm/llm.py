from __future__ import annotations  # 启用延迟注解求值，允许前向引用类型

from dataclasses import dataclass  # dataclass 装饰器：自动生成样板方法
from typing import Dict, List, Tuple  # 类型注解工具

import torch  # PyTorch：张量构造（token id）
from minisgl.core import SamplingParams  # 采样参数（温度/top-k/top-p 等）
from minisgl.distributed import DistributedInfo  # 分布式信息：离线模式下固定为单卡 (rank=0, size=1)
from minisgl.message import (  # 消息类型：离线模式下直接在进程内构造/消费，不走网络
    BaseBackendMsg,  # 后端消息基类
    DetokenizeMsg,  # 解码结果消息（每步新 token）
    UserMsg,  # 用户请求消息
)
from minisgl.scheduler import Scheduler, SchedulerConfig  # 调度器基类与配置：LLM 继承 Scheduler 复用其事件循环


class RequestAllFinished(Exception):
    """所有请求都已完成的信号异常：用于终止 Scheduler 的 run_forever 循环"""
    pass  # 无需携带额外信息，仅作控制流标记


@dataclass  # 数据类：单个请求的生成状态记录
class RequestStatus:
    uid: int  # 请求唯一 id
    input_ids: List[int]  # 输入 token id 序列（原始保存，便于回显）
    output_ids: List[int]  # 已生成的输出 token id 序列（逐步追加）


class LLM(Scheduler):
    """离线批量推理接口：继承调度器，把消息收发改写为进程内函数调用，对外暴露 generate()"""

    def __init__(self, model_path: str, dtype: torch.dtype = torch.bfloat16, **kwargs):
        config = SchedulerConfig(  # 构造离线模式的调度器配置
            model_path=model_path,  # 模型路径
            tp_info=DistributedInfo(0, 1),  # 固定单进程：rank=0，world_size=1
            dtype=dtype,  # 计算精度，默认 bfloat16
            offline_mode=True,  # 标记离线模式：调度器将调用下面的 offline_* 钩子而非网络消息
            **kwargs,  # 其余参数透传（如 max_running_req、page_size 等）
        )
        super().__init__(config)  # 调用调度器初始化（加载模型/KV Cache/启动工作线程等）
        self.pending_requests: List[Tuple[List[int] | str, SamplingParams]] = []  # 待提交请求队列：(prompt, 采样参数) 对
        self.status_map: Dict[int, RequestStatus] = {}  # uid -> 状态记录，用于生成结束后回查输出
        self.counter = 0  # 已分配的 uid 计数器（单调递增）

    def _tokenize_one(self, prompt: List[int] | str) -> torch.Tensor:
        """把单个 prompt 规范化为 CPU 上的 int32 token id 张量"""
        if isinstance(prompt, str):  # 字符串输入：需要分词
            return self.tokenizer.encode(prompt, return_tensors="pt").view(-1).to(torch.int32)  # 编码并展平为 int32 一维张量
        else:  # 已经是 token id 列表：直接构造张量
            return torch.tensor(prompt, dtype=torch.int32, device="cpu")  # CPU int32 张量

    def offline_receive_msg(self, blocking: bool = False) -> List[BaseBackendMsg]:
        """离线消息源：从 pending 队列取请求打包为 UserMsg（受 prefill 预算约束）"""
        if blocking and len(self.pending_requests) == 0:  # 阻塞模式下队列已空
            raise RequestAllFinished()  # 抛出完成信号，结束 run_forever 循环
        results: List[BaseBackendMsg] = []  # 本轮要提交给调度器的消息列表
        added, sum_input_len = 0, 0  # added：本轮实际接纳的请求数；sum_input_len：累计输入长度
        for tokens_or_prompt, sampling_params in self.pending_requests:  # 顺序遍历待提交请求
            if sum_input_len >= self.prefill_budget:  # 累计输入长度达到 prefill 预算
                break  # 剩余请求留到下一轮，避免单批过长
            input_ids = self._tokenize_one(tokens_or_prompt)  # 分词/规范化为张量
            sum_input_len += len(input_ids)  # 累加输入长度
            uid, added = self.counter + added, added + 1  # 为该请求分配 uid；同时递增本地计数
            results.append(UserMsg(uid=uid, input_ids=input_ids, sampling_params=sampling_params))  # 打包成用户消息
            self.status_map[uid] = RequestStatus(  # 登记状态，供结束后取结果
                uid=uid,  # 请求 id
                input_ids=(  # 保存原始输入 id（字符串输入转为列表，数字输入原样保存）
                    input_ids.tolist() if isinstance(tokens_or_prompt, str) else tokens_or_prompt
                ),
                output_ids=[],  # 输出从空开始
            )
        self.counter += added  # 全局计数器推进（注意：与局部 uid 计算方式一致，见下）
        self.pending_requests = self.pending_requests[added:]  # 移除已接纳的请求，保留剩余
        return results  # 返回本轮提交的消息

    def offline_send_result(self, reply: List[DetokenizeMsg]) -> None:
        """离线结果汇：接收解码消息，把新 token 追加到对应请求的输出列表"""
        for msg in reply:  # 遍历本批解码消息
            status = self.status_map[msg.uid]  # 按 uid 找到状态记录
            if not (msg.finished and msg.next_token == self.eos_token_id):  # 结束时若最后是 EOS 则不记录该 token
                status.output_ids.append(msg.next_token)  # 追加新 token 到输出序列

    def generate(
        self,
        prompts: List[str] | List[List[int]],  # 提示列表：字符串或 token id 列表
        sampling_params: List[SamplingParams] | SamplingParams,  # 采样参数：可每请求不同，也可整体共享
    ) -> List[Dict[str, str | List[int]]]:
        """批量生成主入口：填队列 -> 跑事件循环 -> 按提交顺序收集结果"""
        self.pending_requests = []  # 清空残留队列（保证多次调用 generate 互不污染）
        self.status_map = {}  # 清空状态表
        self.counter = 0  # uid 从 0 重新计数（保证结果可按索引 0..n-1 对齐）
        if isinstance(sampling_params, SamplingParams):  # 传入了单个参数对象
            sampling_params = [sampling_params] * len(prompts)  # 复制给每个请求，使两者一一对应
        for prompt, sp in zip(prompts, sampling_params):  # 逐对入队
            self.pending_requests.append((prompt, sp))  # 待提交队列
        try:
            self.run_forever()  # 启动调度器事件循环（prefill/decode 交替），直到所有请求完成
        except RequestAllFinished:  # 阻塞接收时抛出的完成信号
            pass  # 属预期退出路径，吞掉异常
        results: List[Dict[str, str | List[int]]] = []  # 结果列表
        for i in range(len(prompts)):  # 按提交顺序回查（uid 恰好是 0..n-1，见上面对齐逻辑）
            status = self.status_map[i]  # 取该请求的状态
            output_text = self.tokenizer.decode(status.output_ids)  # 把输出 token 解码回文本
            results.append({"text": output_text, "token_ids": status.output_ids})  # 同时返回文本与 token id
        return results  # 返回每个请求的结果字典
