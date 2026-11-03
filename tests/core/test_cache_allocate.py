"""
Test that CacheManager._allocate correctly handles eviction with page_size > 1.
测试 CacheManager._allocate 在 page_size > 1 时能否正确处理逐出（eviction）：
分配结果与空闲表必须保持页对齐，且互不重叠。
"""

from __future__ import annotations  # 启用 PEP 563 延迟注解求值

import pytest  # 导入 pytest 测试框架（fixture、断言增强等）
import torch  # 导入 torch，用于构造测试用张量数据

import minisgl.core as core  # 导入 core 模块本身，以便访问/替换其全局变量 _GLOBAL_CTX
from minisgl.scheduler.cache import CacheManager  # 导入被测对象：缓存管理器（radix cache + 页式分配）


@pytest.fixture(autouse=True)  # 自动生效的 fixture：本文件每个测试前后都会执行，无需显式传参
def reset_global_ctx():
    """Reset global context before and after each test.  每个测试前后重置全局上下文，避免测试间相互污染。"""
    old_ctx = core._GLOBAL_CTX  # 保存当前全局上下文
    core._GLOBAL_CTX = None  # 置空，让被测代码从干净状态重新初始化
    yield  # 挂起点：此处执行测试函数本体
    core._GLOBAL_CTX = old_ctx  # 测试结束后恢复原上下文


def _make_cache_manager(num_pages: int, page_size: int) -> CacheManager:
    """Helper to create a CacheManager with radix cache on CPU.  在 CPU 上构造带 radix cache 的 CacheManager 辅助函数。"""
    page_table = torch.empty((1,))  # 占位 page_table 张量（本测试只关心分配逻辑，不使用其内容）
    ctx = core.Context(page_size=page_size)  # 创建指定页大小的上下文对象
    core.set_global_ctx(ctx)  # 注册为全局上下文（CacheManager 内部依赖它）
    return CacheManager(num_pages, page_size, page_table, type="radix")  # 构造被测的缓存管理器（radix 缓存类型）


def _insert_evictable(cm: CacheManager, input_ids: torch.Tensor, indices: torch.Tensor):
    """Insert a prefix into the radix cache so it becomes evictable.  向 radix cache 插入前缀，使其成为可被逐出的缓存数据。"""
    cm.prefix_cache.insert_prefix(input_ids, indices)  # 插入前缀：token ids 与其在 KV cache 中的槽位索引


def _assert_all_page_aligned(tensor: torch.Tensor, page_size: int, label: str = ""):
    """Assert every element in tensor is a multiple of page_size.  断言张量中每个元素都是 page_size 的整数倍（页对齐）。"""
    if len(tensor) == 0:  # 空张量无需检查
        return  # 直接通过
    misaligned = tensor[tensor % page_size != 0]  # 布尔索引筛出所有未对齐的元素
    assert (
        len(misaligned) == 0
    ), f"{label} contains non-page-aligned values: {misaligned.tolist()}, page_size={page_size}"  # 存在未对齐值即失败，并打印具体值便于定位


def _assert_no_overlap(pages: torch.Tensor, page_size: int):
    """Assert that page-aligned starts, when expanded, produce no overlapping ranges.  断言页起始槽位展开成完整页后互不重叠。"""
    if len(pages) <= 1:  # 少于 2 页不可能重叠
        return  # 直接通过
    expanded = set()  # 已占用的 token 槽位集合
    for p in pages.tolist():  # 逐页检查
        token_range = set(range(p, p + page_size))  # 该页覆盖的 token 槽位区间 [p, p+page_size)
        overlap = expanded & token_range  # 与已占用集合求交集
        assert len(overlap) == 0, f"Overlapping tokens: {overlap}"  # 有交集说明分配重叠，立即失败
        expanded.update(token_range)  # 并入已占用集合


class TestAllocateEvictPageAlignment:
    """Tests for _allocate handling eviction with page_size > 1.  针对 _allocate 逐出行为与页对齐的测试组。"""

    def test_allocate_after_evict_returns_page_aligned(self):
        """Allocated pages after eviction must be page-aligned.  逐出后分配到的页必须是页对齐的。"""
        page_size = 4  # 用小页大小便于构造与验证
        num_pages = 4  # 总共 4 页
        cm = _make_cache_manager(num_pages, page_size)  # 构造缓存管理器

        # Exhaust all free pages  # 耗尽全部空闲页
        cm._allocate(num_pages)  # 一次性分配全部 4 页
        assert len(cm.free_slots) == 0  # 此时空闲槽表应为空

        # Insert 2 pages worth of data into the cache (evictable)  # 向缓存插入 2 页数据（成为可逐出对象）
        input_ids = torch.arange(page_size * 2, dtype=torch.int32)  # 8 个 token id：[0..7]
        # Simulate page table entries: page 0 = [0,1,2,3], page 1 = [4,5,6,7]  # 模拟页表项：第 0 页槽位 0-3，第 1 页槽位 4-7
        indices = torch.arange(page_size * 2, dtype=torch.int32)  # 槽位索引 [0..7]，天然按页对齐
        _insert_evictable(cm, input_ids, indices)  # 插入 radix cache

        # Allocate 1 page — triggers eviction  # 再分配 1 页——此时无空闲页，将触发逐出逻辑
        allocated = cm._allocate(1)
        _assert_all_page_aligned(allocated, page_size, "allocated")  # 断言分配结果页对齐
        _assert_all_page_aligned(cm.free_slots, page_size, "_free_slots after evict")  # 逐出后回填的空闲槽也必须页对齐

    def test_consecutive_allocations_after_evict_no_overlap(self):
        """Multiple allocations after eviction must not produce overlapping pages.  逐出后连续多次分配不得产生重叠页。"""
        page_size = 4
        num_pages = 4
        cm = _make_cache_manager(num_pages, page_size)  # 构造缓存管理器

        # Exhaust all free pages  # 耗尽全部空闲页
        cm._allocate(num_pages)

        # Insert 2 pages into cache  # 插入 2 页可逐出数据
        input_ids = torch.arange(page_size * 2, dtype=torch.int32)
        indices = torch.arange(page_size * 2, dtype=torch.int32)
        _insert_evictable(cm, input_ids, indices)

        # Allocate 2 pages one by one  # 逐次分配 2 页（每次都各自触发逐出）
        page_a = cm._allocate(1)
        page_b = cm._allocate(1)
        all_pages = torch.cat([page_a, page_b])  # 合并两次分配结果统一校验

        _assert_all_page_aligned(all_pages, page_size, "all_pages")  # 两次分配都应对齐
        _assert_no_overlap(all_pages, page_size)  # 两页展开后不得有任何重叠槽位

    def test_free_slots_stay_page_aligned_after_evict(self):
        """_free_slots must remain page-aligned after eviction refills them.  逐出回填后空闲槽表必须保持页对齐。"""
        page_size = 8
        num_pages = 8
        cm = _make_cache_manager(num_pages, page_size)  # 构造 8 页 × 每页 8 token

        # Exhaust all free pages  # 耗尽全部 8 页
        cm._allocate(num_pages)

        # Insert 4 pages worth of data (4 * 8 = 32 tokens)  # 插入 4 页（共 32 token）可逐出数据
        n_tokens = page_size * 4
        input_ids = torch.arange(n_tokens, dtype=torch.int32)
        # Indices: page starts at 0, 8, 16, 24  # 槽位索引 0..31，对应页起点 0/8/16/24
        indices = torch.arange(n_tokens, dtype=torch.int32)
        _insert_evictable(cm, input_ids, indices)

        # Allocate 1 page — evicts and refills _free_slots  # 分配 1 页——触发逐出，并把释放的槽位回填到空闲表
        cm._allocate(1)

        # All remaining free slots must be page-aligned  # 其余全部空闲槽位必须仍保持页对齐
        _assert_all_page_aligned(cm.free_slots, page_size, "_free_slots")

    def test_allocate_exact_pages_needed_from_evict(self):
        """When exactly N pages are needed, eviction must provide at least N pages.  一次需要 N 页时，逐出必须一次凑够至少 N 页。"""
        page_size = 4
        num_pages = 4
        cm = _make_cache_manager(num_pages, page_size)

        # Exhaust all  # 耗尽全部页
        cm._allocate(num_pages)

        # Insert 3 pages into cache  # 插入 3 页可逐出数据（足以满足 2 页需求）
        n_tokens = page_size * 3
        input_ids = torch.arange(n_tokens, dtype=torch.int32)
        indices = torch.arange(n_tokens, dtype=torch.int32)
        _insert_evictable(cm, input_ids, indices)

        # Allocate 2 pages at once — needs eviction of at least 2 pages  # 一次分配 2 页——需要逐出至少 2 页才能满足
        allocated = cm._allocate(2)
        assert len(allocated) == 2  # 必须足额分配 2 页
        _assert_all_page_aligned(allocated, page_size, "allocated")  # 且页对齐
        _assert_no_overlap(allocated, page_size)  # 且互不重叠

    def test_page_to_token_expansion_correct_after_evict(self):
        """_page_to_token on eviction-allocated pages must produce correct consecutive ranges.  逐出分配的页经 _page_to_token 展开须得到正确的连续槽位区间。"""
        page_size = 4
        num_pages = 4
        cm = _make_cache_manager(num_pages, page_size)

        cm._allocate(num_pages)  # 耗尽全部页

        # Insert 2 pages: tokens [0..7] with indices [0..7]  # 插入 2 页可逐出数据
        input_ids = torch.arange(page_size * 2, dtype=torch.int32)
        indices = torch.arange(page_size * 2, dtype=torch.int32)
        _insert_evictable(cm, input_ids, indices)

        # Allocate 2 pages via eviction  # 通过逐出分配 2 页
        pages = cm._allocate(2)
        tokens = cm._page_to_token(pages)  # 页起始槽位展开成完整 token 槽位序列

        # Each page should expand to page_size consecutive tokens  # 每页应展开为 page_size 个连续槽位
        assert len(tokens) == 2 * page_size  # 总数 = 2 × 4 = 8
        for i, page_start in enumerate(pages.tolist()):  # 逐页验证展开正确性
            chunk = tokens[i * page_size : (i + 1) * page_size].tolist()  # 取该页对应的展开片段
            expected = list(range(page_start, page_start + page_size))  # 期望值：从页起点开始的连续整数
            assert chunk == expected, f"Page {page_start}: got {chunk}, expected {expected}"  # 不一致即失败

    def test_check_integrity_passes_after_evict_cycle(self):
        """CacheManager integrity check should pass after allocate-free-evict cycles.  经历分配-入缓存-逐出完整循环后，完整性检查应通过。"""
        page_size = 4
        num_pages = 8
        cm = _make_cache_manager(num_pages, page_size)

        # Allocate 2 pages (simulating a request using them)  # 分配 2 页（模拟请求占用 KV cache）
        pages_for_cache = cm._allocate(2)
        # free_slots: 6 pages remaining  # 此时剩余 6 个空闲页

        # Insert those 2 pages into radix cache (simulating a finished request)  # 把这 2 页插入 radix cache（模拟请求结束、页进入缓存可复用）
        # The token indices come from _page_to_token expansion of the allocated pages  # token 槽位由分配页展开得到
        token_indices = cm._page_to_token(pages_for_cache)
        n_tokens = len(token_indices)
        input_ids = torch.arange(n_tokens, dtype=torch.int32)
        _insert_evictable(cm, input_ids, token_indices)
        # Now: free=6, cache=2, total=8  # 现状：空闲 6 页、缓存 2 页、总计 8 页

        # This should not raise  # 此时完整性检查应通过（不抛异常）
        cm.check_integrity()

        # Now exhaust free slots and trigger eviction  # 耗尽全部空闲页并触发逐出
        cm._allocate(6)
        assert len(cm.free_slots) == 0  # 空闲页清零

        # Allocate 1 more page — must evict from cache  # 再分配 1 页——此时必须从缓存逐出
        allocated = cm._allocate(1)
        _assert_all_page_aligned(allocated, page_size, "allocated after evict")  # 逐出分配的结果仍要页对齐
        _assert_all_page_aligned(cm.free_slots, page_size, "_free_slots after evict")  # 回填的空闲槽同样要对齐


if __name__ == "__main__":  # 直接运行本文件时
    pytest.main([__file__])  # 用 pytest 执行本文件内全部测试
