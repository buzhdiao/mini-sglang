# 延迟求值类型注解：注解在运行时不求值
from __future__ import annotations

# dataclass 生成管理器类，field 提供可变默认值工厂
from dataclasses import dataclass, field
# 类型注解工具
from typing import Iterable, Set

# Batch：送入引擎的一个批次；Req：请求对象
from minisgl.core import Batch, Req


# Decode 管理器：维护正在持续解码（每步生成 1 个 token）的请求集合
@dataclass
class DecodeManager:
    # KV cache 分页大小（用于预留页计算）
    page_size: int
    # 当前在解码中的请求集合（set 便于 O(1) 增删）
    running_reqs: Set[Req] = field(default_factory=set)

    def filter_reqs(self, reqs: Iterable[Req]) -> None:
        # 用本轮 forward 参与的请求刷新运行集合：并集后只保留"还能继续解码"的请求
        # （输出已达上限的请求自然被淘汰；Req 用 id 做相等性，按身份去重）
        self.running_reqs = {req for req in self.running_reqs.union(reqs) if req.can_decode}

    def remove_req(self, req: Req) -> None:
        # 从运行集合移除指定请求（discard：不存在也不报错，容忍重复释放）
        self.running_reqs.discard(req)

    def abort_req(self, uid: int) -> Req | None:
        # 按 uid 查找并中止一个在解码的请求
        for req in self.running_reqs:
            # 找到目标请求
            if req.uid == uid:
                # 直接从集合中移除（set.remove 要求元素存在，此处遍历保证存在）
                self.running_reqs.remove(req)
                # 返回被中止的请求，交由上层释放其资源
                return req
        # 没找到：该请求可能还在 prefill 队列或不存在
        return None

    @property
    def inflight_tokens(self) -> int:
        # 估计所有在解码请求未来还要占用的 token 槽位总数（用于 prefill 预留显存）
        # 每个请求额外预留 (page_size-1) 个槽位：decode 追加 token 时可能把当前页
        # 写满后跨到新页，整页分配会多占最多 page_size-1 个槽位
        tokens_reserved = (self.page_size - 1) * len(self.running_reqs)  # 1 page reserved
        # 总在途 token = 各请求剩余生成数之和 + 页粒度预留
        return sum(req.remain_len for req in self.running_reqs) + tokens_reserved

    def schedule_next_batch(self) -> Batch | None:
        if not self.runnable:
            # 没有可解码请求：本步不产出 decode 批
            return None
        # 按 uid 排序保证批内顺序确定（可复现、利于 CUDA Graph 键稳定）
        return Batch(reqs=sorted(self.running_reqs, key=lambda req: req.uid), phase="decode")

    @property
    def runnable(self) -> bool:
        # 只要有在解码的请求就可产出批次
        return len(self.running_reqs) > 0
