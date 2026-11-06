# 延迟类型注解：注解在运行时不求值
from __future__ import annotations

# heapq 用于驱逐时按 LRU 挑选叶子节点（最小时间戳堆）
import heapq
# time 用于生成 LRU 时间戳
import time
# dataclass 生成不可变句柄类
from dataclasses import dataclass
# 类型注解工具
from typing import Any, Callable, Dict, List, Tuple, TypeAlias

import torch
# 全局上下文：读取 page_size 等调度参数
from minisgl.core import get_global_ctx
# 向下对齐工具：按页大小取整
from minisgl.utils import align_down

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo

# 节点索引键函数类型：把一段 token 序列映射成可哈希的字典键
KEY_FN: TypeAlias = Callable[[torch.Tensor], Any]


class RadixTreeNode:
    # 基数树节点：每条边携带一段连续 token（key）与其 KV 槽位下标（value）。
    # 多个请求只要共享相同前缀，就能沿树复用同一批 KV cache 槽位，
    # 从而跳过重复的预填充（prefill）计算
    counter: int = 0

    def __init__(self, key_fn: KEY_FN, tic: int | None = None) -> None:
        # 保存索引键函数（按页粒度取键）
        self.key_fn = key_fn
        # 子节点字典：键为首页 token 的可哈希值，值为子节点
        self.children: Dict[Any, RadixTreeNode] = {}
        # 父节点引用
        self._parent: RadixTreeNode | None = None
        # 引用计数：>0 表示正被某些在跑请求锁定，不可驱逐
        self.ref_count: int = 0
        # 全局唯一 id，便于调试
        self.uuid = RadixTreeNode.counter
        RadixTreeNode.counter += 1
        # LRU 时间戳：最近被访问的时间，越小越先被驱逐
        self.timestamp = tic or time.monotonic_ns()

        # these fields should be updated later
        # 以下字段在 set_key_value 时才填充：
        # 该节点承载的 token 序列（页对齐）
        self._key: torch.Tensor
        # 对应 token 在 KV 池中的槽位下标
        self._value: torch.Tensor
        # token 数
        self._length: int

    def set_key_value(self, key: torch.Tensor, value: torch.Tensor) -> None:
        # 设置节点内容：token 序列与其槽位下标必须一一对应
        assert len(key) == len(value)
        self._key = key
        self._value = value
        self._length = len(key)

    def set_parent(self, parent: RadixTreeNode) -> None:
        # 挂到父节点下，并在父节点的 children 字典里按首页 token 建立索引
        self._parent = parent
        parent.children[self.key_fn(self._key)] = self

    @property
    def length(self) -> int:
        # 节点承载的 token 数
        return self._length

    @property
    def parent(self) -> RadixTreeNode:
        # 取父节点（根节点没有父，取根的 parent 是错误用法）
        assert self._parent is not None
        return self._parent

    @property
    def value(self) -> torch.Tensor:
        # 取该节点对应的 KV 槽位下标
        return self._value

    def is_root(self) -> bool:
        # 根节点没有父
        return self._parent is None

    def is_leaf(self) -> bool:
        # 叶子节点没有子节点，是驱逐的最小单位
        return len(self.children) == 0

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        # 计算节点 key 与输入序列的最长公共前缀长度（GPU 核并行比较，找到第一个不同处）
        from minisgl.kernel import fast_compare_key

        # compare key and input_ids, find the first diff
        return fast_compare_key(self._key, input_ids)

    def split_at(self, pos: int) -> RadixTreeNode:
        # 在 pos 处把本节点一分为二：前 pos 个 token 留给新父节点，
        # 这是基数树插入/匹配路径不整段命中时的标准操作
        assert 0 < pos < self.length
        # 先记下原父节点
        parent = self.parent

        # 新建节点承接前缀段（继承时间戳，保持 LRU 语义）
        new_node = RadixTreeNode(self.key_fn, self.timestamp)
        # 前缀段的 key/value 给新节点
        new_node.set_key_value(self._key[:pos], self._value[:pos])
        # 新节点接在原父节点下
        new_node.set_parent(parent)
        # 引用计数随前缀段转移（锁定它的人引用的是这段前缀）
        new_node.ref_count = self.ref_count

        # 本节点改持剩余后缀段
        self.set_key_value(self._key[pos:], self._value[pos:])
        # 并改挂到新节点下
        self.set_parent(new_node)

        return new_node

    def __lt__(self, other: RadixTreeNode) -> bool:
        # 供堆比较：时间戳小者优先被驱逐
        return self.timestamp < other.timestamp


# 冻结句柄：指向匹配到的树节点，cached_len 为匹配前缀长度
@dataclass(frozen=True)
class RadixCacheHandle(BaseCacheHandle):
    node: RadixTreeNode

    def get_matched_indices(self) -> torch.Tensor:
        # 自底向上收集路径上每个节点的槽位下标，拼成整段前缀的槽位序列
        node = self.node
        value_list: List[torch.Tensor] = []
        # 沿父指针走到根，沿途收集 value
        while not node.is_root():
            value_list.append(node.value)
            node = node.parent
        # 反转成从树根到叶的顺序再拼接
        value_list.reverse()
        return torch.cat(value_list)


class RadixPrefixCache(BasePrefixCache):
    # 基数树前缀缓存：按页（page_size）粒度组织前缀，支持跨请求 KV 复用与 LRU 驱逐
    def __init__(self, device: torch.device):
        super().__init__()
        # 记录设备
        self.device = device
        # 从全局上下文取页大小：树按键按页对齐，槽位以页为单位复用
        self.page_size = get_global_ctx().page_size
        # 依据页大小生成节点索引键函数
        self.key_fn = _get_key_fn(self.page_size)
        # 空槽位张量，用于 evict(0) 等空返回
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)
        # 可驱逐的 token 数（未被锁定的部分）
        self.evictable_size = 0
        # 受保护（被锁定）的 token 数
        self.protected_size = 0
        # 创建树根：根节点不承载 token，只是遍历入口
        self.root_node = RadixTreeNode(self.key_fn)
        self.root_node.ref_count = 1  # root is always protected

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        # 锁定/解锁句柄：沿匹配路径给每个节点增减引用计数，并同步两类容量统计
        assert isinstance(handle, RadixCacheHandle)
        node = handle.node
        if unlock:
            # 解锁：自底向上逐节点减引用
            while not node.is_root():
                node.ref_count -= 1
                assert node.ref_count >= 0
                # 引用归零的节点从"受保护"转为"可驱逐"
                if node.ref_count == 0:
                    self.evictable_size += node.length
                    self.protected_size -= node.length
                node = node.parent
        else:
            # 加锁：自底向上逐节点加引用
            while not node.is_root():
                # 首次从 0 变 1 的节点转为受保护状态
                if node.ref_count == 0:
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1
                node = node.parent

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        # 沿树匹配输入序列的最长已缓存前缀，返回句柄（供后续加锁与取槽位）
        node, prefix_len = self._tree_walk(input_ids)
        return MatchResult(RadixCacheHandle(prefix_len, node))

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        # 把一条序列的 KV 插入缓存树；只缓存完整页，尾部的半页丢弃
        insert_len = align_down(len(input_ids), self.page_size)
        input_ids, indices = input_ids[:insert_len], indices[:insert_len]
        # 先走一遍树，找出已存在的公共前缀
        node, prefix_len = self._tree_walk(input_ids)
        if prefix_len != insert_len:  # NOTE: prefix_len < insert_len
            # 存在未缓存的后缀：新建节点承接剩余段
            new_node = RadixTreeNode(self.key_fn)
            # 槽位下标必须 clone：外部缓冲可能被复用，树内要持独立副本
            new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:].clone())
            # 挂到匹配到的节点下
            new_node.set_parent(node)
            # 新增部分默认可驱逐
            self.evictable_size += new_node.length
            node = new_node
        # 返回：prefix_len 为插入前已缓存长度（对应槽位调用方可释放），handle 指向新内容
        return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))

    def evict(self, size: int) -> torch.Tensor:
        # 驱逐至少 size 个 token 的可缓存前缀，返回被释放的槽位下标供池回收
        if size == 0:
            # 驱逐 0 恒安全，返回空张量
            return self.empty_tensor
        assert (
            size <= self.evictable_size
        ), f"Cannot evict {size}, only {self.evictable_size} is evictable"

        # 收集当前所有可驱逐叶子节点
        leave_nodes = self._collect_leave_nodes_for_evict()
        # 建最小堆：时间戳最早（最久未用）的叶子先驱逐，即 LRU 策略
        heapq.heapify(leave_nodes)
        evicted_indices: List[torch.Tensor] = []
        evicted_size = 0

        while evicted_size < size:
            assert (
                leave_nodes
            ), f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"
            # 取最久未用的叶子
            node = heapq.heappop(leave_nodes)
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()
            # 累计驱逐量（按整页粒度，可能略超请求量）
            evicted_size += node.length
            # 记录该节点的槽位下标
            evicted_indices.append(node.value)
            # 同步可驱逐统计
            self.evictable_size -= node.length
            # 从父节点中摘除该叶子
            parent = node.parent
            del parent.children[self.key_fn(node._key)]
            # NOTE: root is always protected, so won't be evicted
            # 父节点因此变成叶子且未被锁定时，也进入候选堆（可继续驱逐）
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, parent)

        # 拼接所有被驱逐的槽位下标返回
        return torch.cat(evicted_indices)

    def reset(self) -> None:
        # 清空整棵树暂未实现
        raise NotImplementedError("RadixManager.reset is not implemented")

    @property
    def size_info(self) -> SizeInfo:
        # 返回容量统计
        return SizeInfo(
            evictable_size=self.evictable_size,
            protected_size=self.protected_size,
        )

    def check_integrity(self) -> None:
        # 完整性校验暂为空实现
        pass

    def _collect_leave_nodes_for_evict(self) -> List[RadixTreeNode]:
        # 深度优先遍历整棵树，收集所有"未加锁的叶子节点"作为驱逐候选
        nodes: List[RadixTreeNode] = [self.root_node]
        leave_nodes: List[RadixTreeNode] = []

        # 用栈做迭代式 DFS
        while len(nodes) > 0:
            node = nodes.pop()
            if node.is_leaf():
                # 叶子才可直接摘除；被锁定的不能驱逐
                if node.ref_count == 0:
                    leave_nodes.append(node)
            else:
                # 非叶子继续下钻
                for child in node.children.values():
                    nodes.append(child)

        return leave_nodes

    def _tree_walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        # 沿树逐页匹配输入序列，返回（最终停留节点, 已匹配 token 数）
        prefix_len = 0
        indice_len = len(input_ids)
        node = self.root_node
        # 本次访问的时间戳，用于刷新沿途节点的 LRU 时钟
        tic = time.monotonic_ns()

        # 还没匹配完就继续向下走
        while prefix_len < indice_len:
            # 用"从当前位置起的首页 token"作为键查子节点
            child_node = node.children.get(self.key_fn(input_ids[prefix_len:]))
            if child_node is None:
                # 没有匹配的子节点：停在当前节点，匹配结束
                return node, prefix_len
            node = child_node  # walk to child node

            # NOTE: at least 1 page is matched, so match_len >= page_size
            # 计算该节点 key 与剩余输入的最长公共长度
            match_len = node.get_match_len(input_ids[prefix_len:])
            # 再向下取整到页边界：半页命中不可复用（页是缓存的最小单位）
            match_len = align_down(match_len, self.page_size)
            prefix_len += match_len

            # need to split the node if not fully matched
            # 只命中节点的一部分：把节点在命中处分裂，前缀段独立出来供本请求引用
            if match_len != node.length:
                node = node.split_at(match_len)
                node.timestamp = tic
                return node, prefix_len

            # update timestamp for accessed node
            # 整段命中：刷新节点 LRU 时间戳
            node.timestamp = tic

        return node, prefix_len


def _get_key_fn(page_size: int) -> KEY_FN:
    # 生成节点索引键函数：把 token 序列映射成可哈希值（子字典的键）
    if page_size == 1:
        # 页大小为 1 时直接用首个 token 的标量值
        return lambda x: x[0].item()
    # 否则取前 page_size 个 token 组成元组作为键
    return lambda x: tuple(x[:page_size].tolist())
