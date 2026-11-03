from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解中使用前向引用

from contextlib import contextmanager  # contextmanager 装饰器：把生成器函数包装为 with 上下文管理器
from typing import TYPE_CHECKING, List, Tuple  # 类型注解工具：TYPE_CHECKING（仅类型检查期导入）、List、Tuple

import torch  # PyTorch：张量运算（空闲页管理、批量索引写）
from minisgl.core import Req  # 核心请求结构
from minisgl.kvcache import BaseCacheHandle, MatchResult, create_prefix_cache  # 前缀缓存：句柄类型、匹配结果类型与缓存工厂函数
from minisgl.utils import div_ceil  # 向上取整除法工具：token 数换算页数

if TYPE_CHECKING:  # 仅静态类型检查时导入，避免运行时循环依赖
    from .utils import PendingReq  # 待处理请求（调度器内部中间表示）


class CacheManager:
    """KV Cache 管理器：负责页级显存分配/回收、前缀缓存匹配与插入，并校验页守恒"""

    def __init__(self, num_pages: int, page_size: int, page_table: torch.Tensor, type: str):
        # The `_free_slots` follows a page-aligned manner. For example, if page_size = 2,
        # the `_free_slots` may look like [0, 2, 4, 6, ...], and each slot represents a page.
        # `_free_slots` 按页对齐管理空闲显存：page_size=2 时形如 [0,2,4,6,...]，每个槽位代表一页（存页首物理地址）
        device = page_table.device  # 从页表推断设备，保证分配的张量与页表同设备
        self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size  # 初始全部空闲：页号×页大小=页首地址
        self.prefix_cache = create_prefix_cache(device=device, type=type)  # 创建前缀缓存（如 radix tree）：复用相同前缀的 KV
        self.device = device  # 记录设备（后续构造偏移张量用）
        self.num_pages = num_pages  # 总页数（完整性校验用）
        self.page_table = page_table  # 页表引用：与引擎/槽位管理器共享同一张量
        self.page_size = page_size  # 页大小（页/token 换算用）

    def match_req(self, req: PendingReq) -> MatchResult:
        # 在前缀缓存中查找该请求可复用的最长 KV 前缀
        input_len = req.input_len  # 输入 token 总数
        assert input_len > 0, "Input length must be greater than 0."  # 空输入无意义，直接断言失败
        # 只匹配前 input_len-1 个 token：至少留 1 个 token 供本次前向计算（避免 extend_len=0 的空前向）
        return self.prefix_cache.match_prefix(req.input_ids[: input_len - 1])

    @property
    def available_size(self) -> int:
        # 可用 token 容量 = 可驱逐缓存中的 token 数 + 空闲页可容纳的 token 数（准入控制依据）
        return self.prefix_cache.size_info.evictable_size + len(self.free_slots) * self.page_size

    def lock(self, handle: BaseCacheHandle) -> None:
        # 锁定句柄对应的前缀缓存节点：防止正被使用的 KV 被驱逐回收
        self.prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        # 解锁句柄：允许其缓存页在显存紧张时被驱逐复用
        self.prefix_cache.lock_handle(handle, unlock=True)

    def allocate_paged(self, reqs: List[Req]) -> None:
        # 为一批请求按页分配新增 KV 空间，并把分配到的物理地址批量写入页表
        needed_pages = 0  # 本批新增页总需求
        allocation_info: List[Tuple[int, int, int]] = []  # 每请求的分配信息：(槽位号, 起始页, 结束页)
        for req in reqs:
            first_page = div_ceil(req.cached_len, self.page_size)  # 已缓存部分占用的页数（向上取整）
            last_page = div_ceil(req.device_len, self.page_size)  # 本轮前向后需要的总页数
            if last_page > first_page:  # 需要新增页才分配（页对齐下多数 decode 步无需新页）
                needed_pages += last_page - first_page  # 累加新增页数
                allocation_info.append((req.table_idx, first_page, last_page))  # 记录该请求的页区间
        if needed_pages > 0:  # 有实际需求才执行分配与页表写入
            allocated = self._page_to_token(self._allocate(needed_pages))  # 按页分配，再展开为逐 token 物理地址
            _write_page_table(self.page_table, allocated, allocation_info, self.page_size)  # 批量写入页表

    def cache_req(self, req: Req, *, finished: bool) -> None:
        # ==================================== valid cache region ====================================
        # [0, req.cached_len)                       This part is valid for attention kernel read/write.
        # [0, old_handle.cached_len)                This part is in the prefix cache before prefill.
        # [old_handle.cached_len, req.cached_len)   This part is allocated by cache manager for this request.
        # ================================== allocated cache region ==================================
        # [old_handle.cached_len, cached_len)       This part was not in the prefix cache when prefill,
        #                                           but later cached by other requests.
        #                                           We must free them to avoid memory leak.
        # [cached_len, new_handle.cached_len)       This part is newly inserted into the prefix cache.
        # [new_handle.cached_len, req.cached_len)   This part is tailing part that can not inserted into the prefix cache.
        #                                           We should free it if the request has finished.
        # 把请求已计算的 KV 段插入前缀缓存（中文注解上述区间）：
        # [0, cached_len) 为注意力核可读的有效区；[0, old) 是 prefill 前已命中的缓存；
        # [old, cached_len) 中"后来被其他请求缓存过"的旧物理页必须释放以防泄漏；
        # [new, cached_len) 是因页对齐未能入缓存的尾部，请求已结束则需释放
        insert_ids = req.input_ids[: req.cached_len]  # 取已缓存段的 token 序列作为缓存键
        page_indices = self.page_table[req.table_idx, : req.cached_len]  # 该段 token 对应的 KV 物理地址
        old_handle = req.cache_handle  # 保存旧句柄（prefill 时锁定的命中前缀）
        cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)  # 插入缓存：返回实际入缓存长度与新句柄
        # unlock until all operations on handle is done  # 句柄相关操作全部完成后才解锁
        self.unlock(old_handle)  # 解锁旧句柄：其前缀此后可被正常驱逐
        # this part is already in the prefix cache, free it  # 这段物理页已被其他请求缓存（重复占用），释放防泄漏
        self._free(page_indices[old_handle.cached_len : cached_len])  # 释放 [old, cached_len) 中重复的旧页
        if finished:  # this tail part should be freed  # 请求已结束：页对齐产生的尾段一并释放
            self._free(page_indices[new_handle.cached_len :])  # 释放 [new_handle.cached_len, cached_len) 尾部
        else:  # keep the tail part, update the handle  # 请求继续运行：保留尾段并切换到新句柄
            req.cache_handle = new_handle  # 请求改用新句柄（覆盖新插入的前缀）
            self.lock(new_handle)  # 锁定新句柄，防止正被使用的 KV 被驱逐

    def check_integrity(self) -> None:
        # 一致性自检：空闲页 + 缓存页必须等于总页数（页不可凭空产生或丢失，用于尽早发现泄漏）
        self.prefix_cache.check_integrity()  # 先校验前缀缓存内部结构一致性
        cache_pages = self.prefix_cache.size_info.total_size // self.page_size  # 缓存占用 token 数换算为页数
        if len(self.free_slots) + cache_pages != self.num_pages:  # 空闲+缓存 != 总数 => 存在页泄漏
            raise RuntimeError(  # 抛异常并给出各项计数，便于定位
                "CacheManager integrity check failed:"
                f" free_pages({len(self.free_slots)}) +"
                f" cache_pages({cache_pages}) != num_pages({self.num_pages})"
            )
        if self.page_size > 1:  # 页大小>1 时，额外校验空闲槽位仍保持页对齐
            assert torch.all(self.free_slots % self.page_size == 0)

    @contextmanager  # 上下文管理器：with 块内的 _free 调用被替换为"延迟收集"版本
    def lazy_free_region(self):
        # 延迟释放区域：与在途 GPU 前向可能仍引用的页存在竞态，故先收集、退出时统一归还
        def lazy_free(indices: torch.Tensor) -> None:
            # 只取页首地址（步长 page_size 采样：一页只需记录一个槽位），收集而不立即归还
            lazy_free_list.append(indices[:: self.page_size])

        lazy_free_list: List[torch.Tensor] = []  # 延迟释放的页首地址列表
        try:
            self._free = lazy_free  # 以实例属性临时遮蔽类方法 _free，替换为延迟版本
            yield  # 挂起点：with 块内的释放调用都被延迟收集
        finally:
            del self._free  # 删除实例属性，恢复使用类上的原 _free 方法
            self.free_slots = torch.cat([self.free_slots] + lazy_free_list)  # 退出时把收集的页一次性归还

    def _allocate(self, needed_pages: int) -> torch.Tensor:
        # 分配指定数量的页；空闲页不足时先从前缀缓存驱逐（只驱逐未锁定的节点）
        if needed_pages > (free_pages := len(self.free_slots)):  # 海象运算符：同时求值并绑定 free_pages
            evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)  # 驱逐足够 token 数的缓存
            self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])  # 驱逐结果按页首步长采样后并入空闲列表
            assert len(self.free_slots) >= needed_pages, "Eviction did not free enough space."  # 驱逐后仍不足 => 显存超卖，断言失败
        allocated = self.free_slots[:needed_pages]  # 取前 needed_pages 个页首地址
        self.free_slots = self.free_slots[needed_pages:]  # 从空闲列表移除已分配的页
        return allocated  # 返回按页粒度的物理地址（与 allocation_info 顺序对应）

    def _free(self, indices: torch.Tensor) -> None:
        # 释放一组 token 物理地址：按页首步长采样合并进空闲列表（空张量则跳过）
        if len(indices) > 0:
            self.free_slots = torch.cat([self.free_slots, indices[:: self.page_size]])

    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        # 把页首地址展开为逐 token 地址：页 N -> [N, N+1, ..., N+page_size-1]（供页表逐 token 写入）
        if self.page_size == 1:  # 页大小为 1 时页地址即 token 地址，无需展开
            return pages
        # [X * page_size] -> [X * page_size, ..., X * page_size + page_size - 1]
        offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)  # 页内偏移 [0..page_size-1]
        return (pages.unsqueeze(1) + offsets).flatten()  # 广播相加成 [n_pages, page_size] 再展平为 token 地址序列


def _write_page_table(
    page_table: torch.Tensor,  # 页表张量（GPU）：[槽数, 最大长度] -> KV 物理地址
    allocated: torch.Tensor,  # 分配到的逐 token 物理地址（GPU），顺序与 allocation_info 展开后一致
    allocation_info: List[Tuple[int, int, int]],  # 每请求的 (槽位号, 起始页, 结束页)
    page_size: int,
) -> None:
    # 批量写页表：先在 CPU 锁页内存中填充索引，再异步拷贝到 GPU 做一次高级索引写（减少同步开销）
    needed_tokens = len(allocated)  # 总新增 token 数 = 各请求新增 token 数之和
    table_idx_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)  # 行索引缓冲：每个 token 属于哪个槽位
    positions_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)  # 列索引缓冲：每个 token 的序列位置
    offset = 0  # 当前填充偏移
    for table_idx, first_page, last_page in allocation_info:  # 逐请求填充其页区间对应的索引
        first_pos, last_pos = first_page * page_size, last_page * page_size  # 页区间换算为 token 位置区间
        length = last_pos - first_pos  # 该请求新增 token 数
        table_idx_host[offset : offset + length].fill_(table_idx)  # 填充行索引（槽位号重复 length 次）
        torch.arange(first_pos, last_pos, out=positions_host[offset : offset + length])  # 填充列索引 [first_pos, last_pos)
        offset += length  # 前进偏移
    assert offset == needed_tokens, "Mismatch in allocated tokens and filled tokens."  # 填充总量必须与分配量一致
    table_idxs = table_idx_host.to(page_table.device, non_blocking=True)  # 异步 H2D 拷贝（锁页内存支持 non_blocking）
    offsets = positions_host.to(page_table.device, non_blocking=True)  # 同上
    page_table[table_idxs, offsets] = allocated  # 一次高级索引写：页表[槽位, 位置] = 物理地址
