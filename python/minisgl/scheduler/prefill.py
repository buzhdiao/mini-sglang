from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解中使用前向引用

from dataclasses import dataclass, field  # dataclass 装饰器与 field：声明数据类及带默认工厂的字段
from typing import TYPE_CHECKING, List, Tuple  # 类型注解工具：TYPE_CHECKING、List、Tuple

import torch  # PyTorch：张量拷贝（CPU->GPU 异步填充 token 池）
from minisgl.core import Batch, Req  # 核心结构：批次与请求
from minisgl.utils import init_logger  # 日志工具

from .utils import PendingReq  # 调度器内部待处理请求

if TYPE_CHECKING:  # 仅静态类型检查时导入，避免运行时循环依赖
    from minisgl.kvcache import BaseCacheHandle  # 前缀缓存句柄类型
    from minisgl.message import UserMsg  # 用户请求消息类型

    from .cache import CacheManager  # KV 缓存管理器类型
    from .decode import DecodeManager  # 解码管理器类型
    from .table import TableManager  # 请求槽位管理器类型

logger = init_logger(__name__)  # 模块级日志器


class ChunkedReq(Req):
    """被分块截断（chunked prefill）的请求：只参与 prefill，禁止进入解码阶段"""

    def append_host(self, next_token: torch.Tensor) -> None:
        # 分块请求不会被采样，理论上不存在"新 token"；防御性报错暴露逻辑错误
        raise NotImplementedError("ChunkedReq should not be sampled")

    @property
    def can_decode(self) -> bool:
        return False  # avoid being added to decode manager  # 恒为 False：避免被加入解码管理器（其 prefill 尚未完成）


@dataclass
class PrefillAdder:
    """预填充添加器：在 token 预算与显存预留约束下，本轮尽可能多地接纳待处理请求"""

    token_budget: int  # 本轮可消费的 prefill token 预算（每接纳一个请求按其块大小扣减）
    reserved_size: int  # 已预留量：初值为在途解码 token 数，随接纳请求累加（未来轮次还要消耗显存）
    cache_manager: CacheManager  # 缓存管理器：前缀匹配、锁定与可用容量查询
    table_manager: TableManager  # 槽位管理器：请求槽位分配与 GPU token 池

    def _try_allocate_one(self, req: PendingReq) -> Tuple[BaseCacheHandle, int] | None:
        # 尝试为单个待处理请求分配资源（前缀匹配+锁定+槽位）；资源不足返回 None
        if self.table_manager.available_size == 0:  # 无空闲请求槽位
            return None

        # TODO: consider host cache match case  # TODO：考虑主机侧（CPU）缓存命中的场景
        handle = self.cache_manager.match_req(req).cuda_handle  # 前缀匹配，取 GPU 侧缓存句柄
        cached_len = handle.cached_len  # 命中前缀长度（可复用、无需重算的 token 数）
        # TODO: better estimate policy  # TODO：更精细的容量预估策略
        extend_len = req.input_len - cached_len  # 本轮需要新计算的输入 token 数
        estimated_len = extend_len + req.output_len  # 预估总增量 = 新算输入 + 期望输出

        if estimated_len + self.reserved_size > self.cache_manager.available_size:  # 预估需求超出现有可用容量
            return None  # 拒绝接纳（避免请求中途 OOM）
        self.cache_manager.lock(handle)  # 锁定命中前缀：防止后续被并发驱逐
        if estimated_len + self.reserved_size > self.cache_manager.available_size:  # 锁定操作可能触发驱逐回收，需复核容量
            return self.cache_manager.unlock(handle)  # 仍不足：解锁并返回 None（unlock 返回 None，等效拒绝）

        table_idx = self.table_manager.allocate()  # 分配一个请求槽位（token 池/页表的行号）
        if cached_len > 0:  # NOTE: set the cached part  # NOTE：回填命中前缀部分的数据
            device_ids = self.table_manager.token_pool[table_idx][:cached_len]  # GPU token 池中该槽位前 cached_len 格
            page_entry = self.table_manager.page_table[table_idx][:cached_len]  # 页表中对应的位置区间
            device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)  # 异步拷贝 token id 到 GPU
            page_entry.copy_(handle.get_matched_indices())  # 写入命中前缀的 KV 物理地址（复用已有缓存）

        return handle, table_idx  # 返回句柄与槽位，供 _add_one_req 构造请求

    def _add_one_req(
        self,
        pending_req: PendingReq,  # 待处理请求
        cache_handle: BaseCacheHandle,  # 已锁定的前缀缓存句柄
        table_idx: int,  # 已分配的槽位号
        cached_len: int,  # 已缓存（命中前缀或已处理）长度
    ) -> Req:
        # 把一个待处理请求转成 Req 加入本批；输入超出预算时用 ChunkedReq 截断
        remain_len = pending_req.input_len - cached_len  # 尚未处理的输入长度
        chunk_size = min(self.token_budget, remain_len)  # 本批实际处理的输入 token 数（受预算限制）
        is_chunked = chunk_size < remain_len  # 输入被截断 => 需要分块，后续轮次续接
        CLS = ChunkedReq if is_chunked else Req  # 截断请求用 ChunkedReq 类（不可解码）
        self.token_budget -= chunk_size  # 扣减本轮 token 预算
        self.reserved_size += remain_len + pending_req.output_len  # 累加预留：剩余输入 + 期望输出（后续轮次要占显存）
        # NOTE: update the tokens ids only; new pages will be allocated in the scheduler  # NOTE：此处只回填 token id；新 KV 页由调度器统一分配
        _slice = slice(cached_len, cached_len + chunk_size)  # 本批处理的输入区间
        device_ids = self.table_manager.token_pool[table_idx, _slice]  # GPU token 池对应区间视图
        device_ids.copy_(pending_req.input_ids[_slice].pin_memory(), non_blocking=True)  # 异步拷贝到 GPU token 池
        return CLS(  # 构造请求对象（input_ids 截到本批处理完的位置）
            input_ids=pending_req.input_ids[: cached_len + chunk_size],
            table_idx=table_idx,  # 槽位号
            cached_len=cached_len,  # 已缓存长度（命中的前缀）
            output_len=pending_req.output_len,  # 期望输出长度
            uid=pending_req.uid,  # 请求唯一 id
            cache_handle=cache_handle,  # 缓存句柄
            sampling_params=pending_req.sampling_params,  # 采样参数
        )

    def try_add_one(self, pending_req: PendingReq) -> Req | None:
        # 尝试接纳一个请求：预算耗尽或资源不足返回 None；分块请求优先续接上轮状态
        if self.token_budget <= 0:  # 本轮 token 预算已耗尽
            return None

        if chunked_req := pending_req.chunked_req:  # 海象运算符：该请求上一轮被分块截断过
            return self._add_one_req(  # 续接分块：复用其句柄/槽位/已处理长度
                pending_req=pending_req,
                cache_handle=chunked_req.cache_handle,
                table_idx=chunked_req.table_idx,
                cached_len=chunked_req.cached_len,
            )

        if resource := self._try_allocate_one(pending_req):  # 尝试分配资源（前缀匹配+容量校验+锁定+槽位）
            cache_handle, table_idx = resource  # 解包分配结果
            return self._add_one_req(  # 正常构造请求加入本批
                pending_req=pending_req,
                cache_handle=cache_handle,
                table_idx=table_idx,
                cached_len=cache_handle.cached_len,  # 以命中前缀长度作为已缓存长度
            )

        return None  # 资源不足，无法接纳


@dataclass
class PrefillManager:
    """预填充管理器：维护 FIFO 待处理队列，按预算调度 prefill 批次，并处理中止"""

    cache_manager: CacheManager  # 缓存管理器
    table_manager: TableManager  # 槽位管理器
    decode_manager: DecodeManager  # 解码管理器（取在途 token 数作为显存预留基线）
    pending_list: List[PendingReq] = field(default_factory=list)  # FIFO 待处理队列

    def add_one_req(self, req: UserMsg) -> None:
        # 用户消息入队：转为调度器内部中间表示 PendingReq
        self.pending_list.append(PendingReq(req.uid, req.input_ids, req.sampling_params))

    def schedule_next_batch(self, prefill_budget: int) -> Batch | None:
        # 调度下一批 prefill：按队列顺序尽量多接纳；被截断的请求保持在队首优先续接
        if len(self.pending_list) == 0:  # 队列为空
            return None

        # estimated offset due to in-flight decode  # 在途解码造成的显存预留偏移
        adder = PrefillAdder(  # 构造本轮添加器：预算为配置上限，预留为在途解码 token 总量
            token_budget=prefill_budget,
            reserved_size=self.decode_manager.inflight_tokens,
            cache_manager=self.cache_manager,
            table_manager=self.table_manager,
        )
        reqs: List[Req] = []  # 本批接纳的请求
        chunked_list: List[PendingReq] = []  # 本轮仍被截断的请求（下轮需优先续接）
        for pending_req in self.pending_list:  # 按队列顺序逐个尝试接纳
            if req := adder.try_add_one(pending_req):  # 成功接纳一个
                pending_req.chunked_req = None  # 先清空旧分块标记
                if isinstance(req, ChunkedReq):  # 本轮再次被截断
                    pending_req.chunked_req = req  # 记录新的分块状态
                    chunked_list.append(pending_req)  # 加入分块列表（保持队首）
                reqs.append(req)  # 加入批次
            else:
                break  # We cannot add more requests  # 一旦失败即停止：FIFO 严格顺序，防止队首饥饿
        if len(reqs) == 0:  # 一个都没接纳（预算/资源耗尽）
            return None
        self.pending_list = chunked_list + self.pending_list[len(reqs) :]  # 更新队列：分块请求置首 + 未处理的队尾
        return Batch(reqs=reqs, phase="prefill")  # 构建 prefill 批次

    def abort_req(self, uid: int) -> Req | None:
        # 按 uid 中止等待中的请求：移出队列并返回其分块中的 Req（若在 prefill 中途，需上层释放资源）
        for i, req in enumerate(self.pending_list):  # 线性查找队列
            if req.uid == uid:
                self.pending_list.pop(i)  # 移出队列
                return req.chunked_req  # 返回分块请求（可能为 None 表示无需释放）
        return None  # 队列中无此 uid

    @property
    def runnable(self) -> bool:
        return len(self.pending_list) > 0  # 队列非空即可调度
