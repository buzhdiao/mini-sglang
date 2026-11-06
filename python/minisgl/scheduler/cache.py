# 延迟求值类型注解：注解在运行时不求值，避免循环导入与不必要的运行时开销
from __future__ import annotations

# contextmanager 用于把"惰性释放区域"封装成 with 上下文
from contextlib import contextmanager
# 类型注解工具
from typing import TYPE_CHECKING, List, Tuple

import torch
# Req：调度器侧的请求对象（含 table_idx/cached_len 等生命周期字段）
from minisgl.core import Req
# KV cache 相关：句柄基类、前缀匹配结果、按类型创建前缀缓存（radix 等）
from minisgl.kvcache import BaseCacheHandle, MatchResult, create_prefix_cache
# div_ceil：向上取整除法，用于把 token 数换算成页数
from minisgl.utils import div_ceil

# 仅类型检查时导入，避免运行时循环依赖
if TYPE_CHECKING:
    from .utils import PendingReq


# KV cache 管理器：负责物理 KV 槽位（页）的分配/释放与前缀缓存的匹配/插入/驱逐
class CacheManager:
    def __init__(self, num_pages: int, page_size: int, page_table: torch.Tensor, type: str):
        # The `_free_slots` follows a page-aligned manner. For example, if page_size = 2,
        # the `_free_slots` may look like [0, 2, 4, 6, ...], and each slot represents a page.
        # _free_slots 以页为粒度管理空闲槽位：每个元素是一个页的起始 token 下标，
        # 例如 page_size=2 时形如 [0, 2, 4, 6, ...]，一个元素代表一整页
        # 取页表所在设备，保证空闲列表与页表在同一 GPU 上
        device = page_table.device
        # 初始所有页都空闲：第 i 页的起始槽位即 i * page_size
        self.free_slots = torch.arange(num_pages, dtype=torch.int32, device=device) * page_size
        # 创建前缀缓存实例（如基数树），用于跨请求复用已计算的 KV 前缀
        self.prefix_cache = create_prefix_cache(device=device, type=type)
        # 记录设备，后续分配 offset 张量等要用
        self.device = device
        # 物理页总数（显存容量上限，以页计）
        self.num_pages = num_pages
        # 全局页表（行=请求槽位，列=位置，值=KV 池中的 token 槽位）
        self.page_table = page_table
        # 每页包含的 token 数（页大小）
        self.page_size = page_size

    def match_req(self, req: PendingReq) -> MatchResult:
        # 取输入长度
        input_len = req.input_len
        # 输入长度必须为正，否则无法做任何前缀匹配
        assert input_len > 0, "Input length must be greater than 0."
        # 只用前 input_len-1 个 token 做前缀匹配：最后一个 token 的 KV 必须由本次
        # forward 现算（注意力需要它作为 query），不能从缓存复用
        return self.prefix_cache.match_prefix(req.input_ids[: input_len - 1])

    @property
    def available_size(self) -> int:
        # 当前可用 token 容量 = 前缀缓存中可驱逐部分的 token 数 + 空闲页折算的 token 数
        return self.prefix_cache.size_info.evictable_size + len(self.free_slots) * self.page_size

    def lock(self, handle: BaseCacheHandle) -> None:
        # 锁定句柄对应的缓存节点：防止其他请求的驱逐操作把它释放掉
        self.prefix_cache.lock_handle(handle, unlock=False)

    def unlock(self, handle: BaseCacheHandle) -> None:
        # 解锁句柄：请求对该段前缀的操作已全部完成，允许其重新进入可驱逐状态
        self.prefix_cache.lock_handle(handle, unlock=True)

    def allocate_paged(self, reqs: List[Req]) -> None:
        # 统计本批所有请求需要新增的页数
        needed_pages = 0
        # 记录每个请求的页表写入信息：(请求槽位号, 起始页, 结束页)
        allocation_info: List[Tuple[int, int, int]] = []
        for req in reqs:
            # 已缓存部分占用的页数（向上取整）
            first_page = div_ceil(req.cached_len, self.page_size)
            # 本轮 forward 后总长度占用的页数（向上取整）
            last_page = div_ceil(req.device_len, self.page_size)
            # 只有末页未满/未分配时才需要新页：整页分配保证同页不跨请求共享
            if last_page > first_page:
                # 累计需要分配的页数
                needed_pages += last_page - first_page
                # 记录该请求要填的页范围
                allocation_info.append((req.table_idx, first_page, last_page))
        if needed_pages > 0:
            # 统一分配页，再把页起始槽位展开成逐 token 槽位
            allocated = self._page_to_token(self._allocate(needed_pages))
            # 把分配到的 token 槽位批量写进全局页表
            _write_page_table(self.page_table, allocated, allocation_info, self.page_size)

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
        # 取请求已写入 KV 的那段输入 token（用于插入前缀缓存）
        insert_ids = req.input_ids[: req.cached_len]
        # 取该请求在页表中对应的 token 槽位（与 insert_ids 一一对应）
        page_indices = self.page_table[req.table_idx, : req.cached_len]
        # 保存旧句柄（prefill 前匹配到的前缀），插入后要解锁它
        old_handle = req.cache_handle
        # 把该请求的 KV 插入前缀缓存：返回实际被缓存接纳的长度与新句柄
        # （页对齐约束下，末尾不满一页的尾巴插不进去）
        cached_len, new_handle = self.prefix_cache.insert_prefix(insert_ids, page_indices)
        # unlock until all operations on handle is done
        # 全部读取旧句柄的操作完成后才解锁，避免中途被别人驱逐
        self.unlock(old_handle)
        # this part is already in the prefix cache, free it
        # 旧句柄之后、新缓存边界之前的槽位：别的请求已把这段算过并入了缓存，
        # 本请求的这份是重复副本，必须释放防止显存泄漏
        self._free(page_indices[old_handle.cached_len : cached_len])
        if finished:  # this tail part should be freed
            # 请求已结束：页对齐插不进缓存的尾部槽位也没有保留价值，直接释放
            self._free(page_indices[new_handle.cached_len :])
        else:  # keep the tail part, update the handle
            # 请求还在跑（decode 中）：尾部槽位仍要继续追加 token，保留并换用新句柄
            req.cache_handle = new_handle
            # 锁定新句柄，防止这段前缀在请求存活期间被驱逐
            self.lock(new_handle)

    def check_integrity(self) -> None:
        # 先让前缀缓存自检（节点引用计数、父子关系等）
        self.prefix_cache.check_integrity()
        # 前缀缓存当前占用的页数（总 token 数折算成页）
        cache_pages = self.prefix_cache.size_info.total_size // self.page_size
        # 不变量：空闲页数 + 缓存占用页数 必须等于总页数，否则说明有页丢失或重复释放
        if len(self.free_slots) + cache_pages != self.num_pages:
            raise RuntimeError(
                "CacheManager integrity check failed:"
                f" free_pages({len(self.free_slots)}) +"
                f" cache_pages({cache_pages}) != num_pages({self.num_pages})"
            )
        if self.page_size > 1:
            # 空闲槽位必须仍是页对齐的（都是 page_size 的整数倍）
            assert torch.all(self.free_slots % self.page_size == 0)

    @contextmanager
    def lazy_free_region(self):
        # 惰性释放区域：区域内调用 _free 只把槽位暂存到列表，不出上下文不真正回收。
        # 目的：释放的槽位可能仍被当前在跑的 kernel/下一次 forward 引用，
        # 立即复用会造成数据竞争（use-after-free），推迟到本批结果处理完再回收
        def lazy_free(indices: torch.Tensor) -> None:
            # 按页粒度抽稀（每页只留起始槽位）后加入待释放列表
            lazy_free_list.append(indices[:: self.page_size])

        # 暂存延迟释放的页起始槽位列表
        lazy_free_list: List[torch.Tensor] = []
        try:
            # 运行期间把实例的 _free 方法偷换成惰性版本
            self._free = lazy_free
            # 执行 with 体内的代码（通常是本批请求的资源回收）
            yield
        finally:
            # 无论是否异常都恢复真正的 _free 方法
            del self._free
            # 此时本批计算已同步完成，被延迟释放的页可以安全回收
            self.free_slots = torch.cat([self.free_slots] + lazy_free_list)

    def _allocate(self, needed_pages: int) -> torch.Tensor:
        # 用海象表达式先取当前空闲页数
        if needed_pages > (free_pages := len(self.free_slots)):
            # 空闲页不够：从前缀缓存中按 LRU 驱逐足够的 token 数（换算成 token 计）
            evicted = self.prefix_cache.evict((needed_pages - free_pages) * self.page_size)
            # 驱逐出来的槽位按页粒度抽稀后并入空闲列表
            self.free_slots = torch.cat([self.free_slots, evicted[:: self.page_size]])
            # 驱逐后仍不够说明缓存中没有足够可驱逐内容，属于严重错误
            assert len(self.free_slots) >= needed_pages, "Eviction did not free enough space."
        # 从空闲列表头部切出所需页（头部是最久未用的，利于 LRU）
        allocated = self.free_slots[:needed_pages]
        # 剩余部分继续留在空闲列表
        self.free_slots = self.free_slots[needed_pages:]
        # 返回本批分配到的页起始槽位
        return allocated

    def _free(self, indices: torch.Tensor) -> None:
        if len(indices) > 0:
            # 把 token 槽位按页粒度抽稀（每页取起始下标）后归还空闲列表
            self.free_slots = torch.cat([self.free_slots, indices[:: self.page_size]])

    def _page_to_token(self, pages: torch.Tensor) -> torch.Tensor:
        if self.page_size == 1:
            # 页大小为 1 时页槽位即 token 槽位，无需展开
            return pages
        # [X * page_size] -> [X * page_size, ..., X * page_size + page_size - 1]
        # 把每页的起始槽位展开成该页内所有 token 槽位：
        # offsets 是 [0, 1, ..., page_size-1]，广播相加后展平
        offsets = torch.arange(self.page_size, device=self.device, dtype=torch.int32)
        return (pages.unsqueeze(1) + offsets).flatten()


def _write_page_table(
    page_table: torch.Tensor,
    allocated: torch.Tensor,
    allocation_info: List[Tuple[int, int, int]],
    page_size: int,
) -> None:
    # 本批需要写入页表的 token 槽位总数（= 分配到的 token 槽位数）
    needed_tokens = len(allocated)
    # 在 CPU 侧用锁页内存组装行号数组（锁页内存支持 GPU 异步拷贝，速度快）
    table_idx_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    # 在 CPU 侧组装位置（列号）数组
    positions_host = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    # 当前填充到的偏移
    offset = 0
    for table_idx, first_page, last_page in allocation_info:
        # 该请求要写入的起始/结束位置（列号，按 token 计）
        first_pos, last_pos = first_page * page_size, last_page * page_size
        # 该请求要写入的槽位个数
        length = last_pos - first_pos
        # 这段所有行的行号都是该请求的 table_idx，整段填充
        table_idx_host[offset : offset + length].fill_(table_idx)
        # 列号是连续区间 [first_pos, last_pos)，直接生成等差序列写入
        torch.arange(first_pos, last_pos, out=positions_host[offset : offset + length])
        # 推进偏移到下一段
        offset += length
    # 校验填充总量与分配量严格一致，防止页表写错位
    assert offset == needed_tokens, "Mismatch in allocated tokens and filled tokens."
    # 行号数组异步拷贝到 GPU（non_blocking：不阻塞 CPU）
    table_idxs = table_idx_host.to(page_table.device, non_blocking=True)
    # 列号数组异步拷贝到 GPU
    offsets = positions_host.to(page_table.device, non_blocking=True)
    # 一次性批量写页表：page_table[行号, 列号] = 分配到的 KV token 槽位
    page_table[table_idxs, offsets] = allocated
