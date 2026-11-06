# 延迟求值类型注解：注解在运行时不求值
from __future__ import annotations

# dataclass 生成管理器类；field 提供默认工厂
from dataclasses import dataclass, field
# 类型注解工具
from typing import TYPE_CHECKING, List, Tuple

import torch
# Batch/Req：批次与请求对象（Req 会被子类化）
from minisgl.core import Batch, Req
# 日志工具
from minisgl.utils import init_logger

# PendingReq：等待调度的用户请求（含分块续传信息）
from .utils import PendingReq

# 仅类型检查导入：避免运行时循环依赖
if TYPE_CHECKING:
    from minisgl.kvcache import BaseCacheHandle
    from minisgl.message import UserMsg

    from .cache import CacheManager
    from .decode import DecodeManager
    from .table import TableManager

# 本模块日志器
logger = init_logger(__name__)


# 分块请求：超长输入被切块后"未完成部分"的占位请求
class ChunkedReq(Req):
    def append_host(self, next_token: torch.Tensor) -> None:
        # 分块请求不参与采样（它还没有把输入喂完），不应走到这里
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        # 恒为 False：防止分块请求被误加入 decode 运行集合
        return False  # avoid being added to decode manager


# Prefill 加法器：在给定 token 预算内逐个尝试接纳待调度请求
@dataclass
class PrefillAdder:
    # 本轮 prefill 还可消耗的新 token 预算（控制单批计算量）
    token_budget: int
    # 需为在途 decode 请求预留的 token 槽位数（防止 decode 被 prefill 挤饿死）
    reserved_size: int
    # KV cache 管理器（页分配/前缀匹配）
    cache_manager: CacheManager
    # 请求槽位管理器（table_idx 分配）
    table_manager: TableManager

    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        # 请求槽位耗尽：无法接纳新请求
        if self.table_manager.available_size == 0:
            return None

        # TODO: consider host cache match case
        # 前缀匹配：找出输入中已在前缀缓存里的部分（可跳过重算）
        handle = self.cache_manager.match_req(req).cuda_handle
        # 命中的缓存长度
        cached_len = handle.cached_len
        # TODO: better estimate policy
        # 本次需要实际计算（扩展）的 token 数 = 输入长 - 缓存命中
        extend_len = req.input_len - cached_len
        # 估算整个生命周期要占的槽位 = 扩展部分 + 未来的输出 token
        estimated_len = extend_len + req.output_len

        # 预估占用加预留超过当前可用容量：不接纳（第一次检查，粗筛）
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            return None
        # 先锁定匹配到的前缀节点，防止检查期间被其他请求驱逐
        self.cache_manager.lock(handle)
        if estimated_len + self.reserved_size > self.cache_manager.available_size:
            # 锁定后复检：其他并发路径可能刚好改变了可用量；仍不足则解锁并放弃
            return self.cache_manager.unlock(handle)

        # 为该请求分配一个页表行（table_idx）
        table_idx = self.table_manager.allocate()
        if cached_len > 0:  # NOTE: set the cached part
            # 有缓存命中：先把命中部分"挂"到该请求的页表行上
            # 目标 GPU 槽位行（前 cached_len 个）
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]
            # 对应的页表行
            page_entry = self.table_manager.page_table[table_idx][:cached_len]
            # 把命中的输入 token 异步拷到 token 池（kernel 要读 token id）
            device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
            # 把匹配到的 KV 槽位下标写进页表，注意力 kernel 才能读到复用的 KV
            page_entry.copy_(handle.get_matched_indices())

        # 返回缓存句柄与分配到的槽位号
        return handle, table_idx

    def _add_one_req(
        self,
        pending_req: PendingReq,
        cache_handle: BaseCacheHandle,
        table_idx: int,
        cached_len: int,
    ) -> Req:
        # 还需喂进模型的输入长度
        remain_len = pending_req.input_len - cached_len
        # 本轮最多喂 min(预算, 剩余) 个 token：超出部分留给下一轮（chunked prefill）
        chunk_size = min(self.token_budget, remain_len)
        # 切块标志：预算不够一次性喂完
        is_chunked = chunk_size < remain_len
        # 被切块的请求用 ChunkedReq 类标记（不可采样/解码）
        CLS = ChunkedReq if is_chunked else Req
        # 扣减本轮 token 预算
        self.token_budget -= chunk_size
        # 把该请求未来的占用量计入预留（后续 prefill 决策要避开）
        self.reserved_size += remain_len + pending_req.output_len
        # NOTE: update the tokens ids only; new pages will be allocated in the scheduler
        # 本轮要写入的 token 区间 [cached_len, cached_len+chunk_size)
        _slice = slice(cached_len, cached_len + chunk_size)
        # token 池中对应的目标槽位
        device_ids = self.table_manager.token_pool[table_idx, _slice]
        # 把这批输入 token 从 CPU 锁页内存异步拷到 GPU
        device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)
        # 构造正式的 Req（或 ChunkedReq）：新页的分配推迟到 _prepare_batch 统一做
        return CLS(
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,
            cached_len=cached_len,
            output_len=pending_req.output_len,
            uid=pending_req.uid,
            cache_handle=cache_handle,
            sampling_params=pending_req.sampling_params,
        )

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        # 预算耗尽：本批不再接纳任何请求
        if self.token_budget <= 0:
            return None

        # 分块续传：该请求上一轮没喂完，沿用旧的句柄/槽位/已缓存长度
        if chunked_req := pending_req.chunked_req:
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
            )

        # 全新请求：先尝试分配资源，成功才构造 Req
        if resource := self._try_allocate_one(pending_req):
            # 解包资源
            cache_handle, table_idx = resource
            return self._add_one_req(
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,
            )

        # 资源不足：本请求本轮无法调度
        return None


# Prefill 管理器：维护等待 prefill 的请求队列并产出 prefill 批次
@dataclass
class PrefillManager:
    # KV cache 管理器
    cache_manager: CacheManager
    # 槽位管理器
    table_manager: TableManager
    # decode 管理器（用于读取在途 token 量做预留）
    decode_manager: DecodeManager
    # 等待调度的请求队列（保持到达顺序，先来先服务）
    pending_list: List[PendingReq] = field(default_factory=list)

    def add_one_req(self, req: UserMsg) -> None:
        # 用户消息转成内部 PendingReq 入队（含 uid/输入/采样参数）
        self.pending_list.append(PendingReq(req.uid, req.input_ids, req.sampling_params))

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        # 队列为空：无事可做
        if len(self.pending_list) == 0:
            return None

        # estimated offset due to in-flight decode
        # 组装加法器：预留量取 decode 在途 token 的估计值，
        # 保证 prefill 抢占资源时不会把正在解码的请求挤到 OOM
        adder = PrefillAdder(
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )
        # 本批接纳的请求
        reqs: List[Req] = []
        # 本轮没喂完、下轮要继续的分块请求队列
        chunked_list: List[PendingReq] = []
        for pending_req in self.pending_list:
            # 逐个尝试接纳
            if req := adder.try_add_one(pending_req):
                # 先清掉旧的分块续传引用
                pending_req.chunked_req = None
                if isinstance(req, ChunkedReq):
                    # 本轮又被切块：把新的 ChunkedReq 挂回请求上，并让它排到队首
                    pending_req.chunked_req = req
                    chunked_list.append(pending_req)
                # 加入本批
                reqs.append(req)
            else:
                # 资源/预算不足：直接停止（不跳过后面的请求，保持 FIFO 公平性）
                break  # We cannot add more requests
        if len(reqs) == 0:
            # 一个都没接纳：本步不产出 prefill 批
            return None
        # 更新队列：没喂完的分块请求放最前面优先续传，其余从未接纳处继续排队
        self.pending_list = chunked_list + self.pending_list[len(reqs) :]
        # 产出 prefill 批次
        return Batch(reqs=reqs, phase="prefill")

    def abort_req(self, uid: int) -> Req | None:
        # 在等待队列中查找要中止的请求
        for i, req in enumerate(self.pending_list):
            # 按 uid 匹配
            if req.uid == uid:
                # 从队列移除
                self.pending_list.pop(i)
                # 若它已有未完成的 ChunkedReq，返回它交由上层释放资源
                return req.chunked_req
        # 不在 prefill 队列中
        return None

    @property
    def runnable(self) -> bool:
        # 队列非空即有待 prefill 的请求
        return len(self.pending_list) > 0
