from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许在类型注解中使用未定义/前向引用的类型（如 Req | None）

from dataclasses import dataclass, field  # 导入 dataclass 装饰器与 field，用于声明数据类及带默认工厂的字段
from typing import Iterable, Set  # 导入类型注解：Iterable（可迭代）与 Set（集合）

from minisgl.core import Batch, Req  # 导入核心数据结构：Batch（一次调度出的一批请求）与 Req（单个用户请求）


@dataclass  # 数据类装饰器，自动生成 __init__、__repr__、__eq__ 等方法
class DecodeManager:
    """解码阶段管理器：维护运行中的请求集合，并负责构建每轮 decode 批次"""

    page_size: int  # KV Cache 页大小（每页容纳的 token 数），用于估算页内预留开销
    running_reqs: Set[Req] = field(default_factory=set)  # 当前处于解码阶段的请求集合（用 set 支持高效增删查）

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        # 将传入的请求并入运行集合，再过滤掉不可解码（can_decode=False）的请求，整体重建集合
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        self.running_reqs.discard(req)  # 从运行集合移除指定请求；不存在时不抛异常

    def abort_req(self, uid: int) -> Req | None:
        for req in self.running_reqs:  # 线性遍历运行中的请求
            if req.uid == uid:  # 找到与目标 uid 匹配的请求
                self.running_reqs.remove(req)  # 从运行集合中移除（即中止该请求的解码）
                return req  # 返回被中止的请求，交由上层后续处理（如释放 KV Cache）
        return None  # 未找到匹配 uid，返回 None 表示中止失败/无需处理

    @property
    def inflight_tokens(self) -> int:
        tokens_reserved = (self.page_size - 1) * len(self.running_reqs)  # 1 page reserved  # 每请求预留 1 页，最坏情况下页内有 page_size-1 个 token 浪费
        return sum(req.remain_len for req in self.running_reqs) + tokens_reserved  # 在途 token 总量 = 所有请求剩余待生成 token 数 + 页内预留浪费，用于内存预估与准入控制

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:  # 运行集合为空，没有可解码的请求
            return None  # 返回 None，本轮调度不产生解码批次
        # 按 uid 升序排序后再构建批次：set 迭代顺序不稳定，排序可保证各 TP rank 看到一致的请求顺序（#113 修复）
        return Batch(reqs=sorted(self.running_reqs, key=lambda req: req.uid), phase="decode")

    @property
    def runnable(self) -> bool:
        return len(self.running_reqs) > 0  # 运行集合非空即表示还有可调度的解码请求
