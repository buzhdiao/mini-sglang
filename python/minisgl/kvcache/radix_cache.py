from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解中使用前向引用

import heapq  # 最小堆：淘汰时按 LRU（时间戳最早优先）选出节点
import time  # 时间源：monotonic_ns 作 LRU 时间戳
from dataclasses import dataclass  # dataclass 装饰器
from typing import Any, Callable, Dict, List, Tuple, TypeAlias  # 类型注解工具（TypeAlias：类型别名声明）

import torch  # PyTorch：key/value 张量
from minisgl.core import get_global_ctx  # 全局上下文（读取 page_size）
from minisgl.utils import align_down  # 向下对齐工具：按页粒度截断

from .base import BaseCacheHandle, BasePrefixCache, InsertResult, MatchResult, SizeInfo  # 前缀缓存接口与数据结构

KEY_FN: TypeAlias = Callable[[torch.Tensor], Any]  # 类型别名：key 提取函数——从 token 序列提取可哈希的字典键


class RadixTreeNode:
    """radix tree 节点：每节点持有一段连续的 token 序列（key）与其在 KV Cache 中的物理位置（value）。

    与普通前缀树的区别：一个节点可包含多个 token（路径压缩），匹配不满时在中间分裂节点。
    ref_count > 0 表示正被请求使用（受保护，不可淘汰）；ref_count = 0 表示仅作前缀复用（可淘汰）。
    """

    counter: int = 0  # 类级计数器：为每个节点生成全局唯一 uuid（调试/标识用）

    def __init__(self, key_fn: KEY_FN, tic: int | None = None) -> None:
        self.key_fn = key_fn  # 保存 key 提取函数（按 page_size 生成字典键）
        self.children: Dict[Any, RadixTreeNode] = {}  # 子节点字典：键为首 token（或首页 token 元组），值为子节点
        self._parent: RadixTreeNode | None = None  # 父节点引用（根节点为 None）
        self.ref_count: int = 0  # 引用计数：>0 表示被请求锁定（不可淘汰）
        self.uuid = RadixTreeNode.counter  # 分配全局唯一 id
        RadixTreeNode.counter += 1  # 计数器递增
        self.timestamp = tic or time.monotonic_ns()  # LRU 时间戳：优先继承父节点时间戳（保持相对新旧），否则取当前单调时钟

        # these fields should be updated later  # 以下字段需后续通过 set_key_value 设置
        self._key: torch.Tensor  # 本节点持有的 token 序列（路径压缩的边标签）
        self._value: torch.Tensor  # 对应 token 在 KV Cache 中的物理位置索引
        self._length: int  # key 长度（缓存避免重复求 len）

    def set_key_value(self, key: torch.Tensor, value: torch.Tensor) -> None:
        # 设置节点的 key/value/length 三元组（key 与 value 必须一一对应）
        assert len(key) == len(value)
        self._key = key  # token 序列
        self._value = value  # cache 物理位置序列
        self._length = len(key)  # 缓存长度

    def set_parent(self, parent: RadixTreeNode) -> None:
        # 建立双向父子关系：本节点记父指针，父节点的 children 字典以本节点 key 的字典键登记
        self._parent = parent
        parent.children[self.key_fn(self._key)] = self  # key_fn 决定字典键粒度（单 token 或首页）

    @property  # 属性：节点长度
    def length(self) -> int:
        return self._length  # 返回缓存的 key 长度

    @property  # 属性：父节点
    def parent(self) -> RadixTreeNode:
        assert self._parent is not None  # 根节点没有父节点，访问即视为逻辑错误
        return self._parent

    @property  # 属性：物理位置序列
    def value(self) -> torch.Tensor:
        return self._value  # 返回本段 token 对应的 cache 位置

    def is_root(self) -> bool:
        # 无父节点即为根
        return self._parent is None

    def is_leaf(self) -> bool:
        # 无子节点即为叶
        return len(self.children) == 0

    def get_match_len(self, input_ids: torch.Tensor) -> int:
        """计算本节点 key 与输入序列的最长公共前缀长度（匹配长度）。"""
        from minisgl.kernel import fast_compare_key  # 延迟导入 GPU 加速的比较 kernel

        # compare key and input_ids, find the first diff  # 逐元素比较，找到第一个不同处即为匹配长度
        return fast_compare_key(self._key, input_ids)

    def split_at(self, pos: int) -> RadixTreeNode:
        """在位置 pos 处把本节点一分为二：前 pos 个 token 迁入新节点（作为本节点的父）。

        用于匹配进行到节点中间的场景：新节点承接已匹配的前缀（原 ref_count 一并迁移），
        本节点保留剩余后缀并改挂到新节点下。返回新节点（即"前缀节点"）。
        """
        assert 0 < pos < self.length  # 分裂点必须在节点内部（否则无意义）
        parent = self.parent  # 先记下原父节点

        new_node = RadixTreeNode(self.key_fn, self.timestamp)  # 创建前缀节点（继承时间戳，避免分裂影响 LRU 语义）
        new_node.set_key_value(self._key[:pos], self._value[:pos])  # 前 pos 个 token/位置归新节点
        new_node.set_parent(parent)  # 新节点挂到原父节点下
        new_node.ref_count = self.ref_count  # 引用计数迁给前缀节点（原有使用者依赖的正是这段前缀）

        self.set_key_value(self._key[pos:], self._value[pos:])  # 本节点裁剪为剩余后缀
        self.set_parent(new_node)  # 改挂到新节点下（同时在新节点 children 中登记）

        return new_node  # 返回前缀节点

    def __lt__(self, other: RadixTreeNode) -> bool:
        # 定义节点大小比较：按时间戳，供 heapq 实现 LRU（越旧越优先淘汰）
        return self.timestamp < other.timestamp


@dataclass(frozen=True)  # 不可变数据类：句柄创建后其字段不可变
class RadixCacheHandle(BaseCacheHandle):
    """radix 缓存句柄：记录命中路径的最深节点，可回溯整条链取出全部命中位置"""

    node: RadixTreeNode  # 命中路径的最深节点（沿 parent 链回溯即得完整前缀）

    def get_matched_indices(self) -> torch.Tensor:
        # 从最深节点向上收集每段的物理位置，再反转为前缀顺序拼接
        node = self.node
        value_list: List[torch.Tensor] = []  # 各节点 value 的收集列表（自底向上）
        while not node.is_root():  # 根节点不持有数据，跳过
            value_list.append(node.value)  # 收集本段位置
            node = node.parent  # 向上一层
        value_list.reverse()  # 反转：从根方向到叶方向（恢复前缀顺序）
        return torch.cat(value_list)  # 拼成一维索引张量（形状 [cached_len]）


class RadixPrefixCache(BasePrefixCache):
    """基于 radix tree 的前缀缓存：跨请求复用相同前缀的 KV（如多轮对话、few-shot 公共前缀）。

    核心机制：
    - 匹配：沿树逐节点比较最长公共前缀，必要时在节点中间分裂；
    - 引用计数：被请求使用的路径整链上锁（protected），空闲部分可按 LRU 淘汰；
    - 淘汰：每次弹出一个最旧的空闲叶节点，父节点若因此变叶且空闲则入堆继续淘汰。
    """

    def __init__(self, device: torch.device):
        super().__init__()  # 调用 ABC 基类初始化
        self.device = device  # 记录设备
        self.page_size = get_global_ctx().page_size  # 页大小：匹配/插入均按页粒度对齐（避免页内部分命中导致物理不连续）
        self.key_fn = _get_key_fn(self.page_size)  # 构造字典键函数（page_size=1 用单 token，否则用首页元组）
        self.empty_tensor = torch.empty(0, dtype=torch.int32, device=device)  # 共享空张量（evict(0) 等场景返回）
        self.evictable_size = 0  # 可淘汰 token 数（ref_count=0 的节点长度之和）
        self.protected_size = 0  # 受保护 token 数（ref_count>0 的节点长度之和）
        self.root_node = RadixTreeNode(self.key_fn)  # 创建根节点（不持有数据）
        self.root_node.ref_count = 1  # root is always protected  # 根节点永受保护，不会被淘汰

    def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
        """沿命中链逐节点上锁/解锁（引用计数 ±1），并同步维护容量统计。

        上锁：从叶到根，第一个从 0 变正的节点段长度由 evictable 转入 protected；
        解锁：从叶到根逐层 -1，节点归零时其长度由 protected 转回 evictable。
        """
        assert isinstance(handle, RadixCacheHandle)  # 类型校验（接口收到的是基类引用）
        node = handle.node  # 从最深节点开始
        if unlock:  # 解锁：引用计数递减
            while not node.is_root():  # 根节点不参与计数
                node.ref_count -= 1  # 减引用
                assert node.ref_count >= 0  # 不允许减到负数（配对性检查）
                if node.ref_count == 0:  # 恰好归零：该段从受保护转为可淘汰
                    self.evictable_size += node.length
                    self.protected_size -= node.length
                node = node.parent  # 继续向上
        else:  # 上锁：引用计数递增
            while not node.is_root():
                if node.ref_count == 0:  # 从 0 变 1：该段从可淘汰转为受保护
                    self.evictable_size -= node.length
                    self.protected_size += node.length
                node.ref_count += 1  # 增引用
                node = node.parent  # 继续向上

    def match_prefix(self, input_ids: torch.Tensor) -> MatchResult:
        """在树中匹配最长已知前缀（不修改树结构，也不改变引用计数）。"""
        node, prefix_len = self._tree_walk(input_ids)  # 沿树行走得到最深命中节点与命中长度
        return MatchResult(RadixCacheHandle(prefix_len, node))  # 打包为句柄返回（调用方需先 lock 再用索引）

    def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> InsertResult:
        """把一段新前缀插入树中，返回插入前已命中的长度与插入后句柄。

        上层可据 cached_len 释放本次重复分配的 cache 槽位（命中部分无需新分配）。
        """
        insert_len = align_down(len(input_ids), self.page_size)  # 按页粒度向下截断：页内不足一页的尾部不插入（保证下次整页匹配）
        input_ids, indices = input_ids[:insert_len], indices[:insert_len]  # 截断后的 key 与物理位置
        node, prefix_len = self._tree_walk(input_ids)  # 先行走：找到已存在的最长前缀
        if prefix_len != insert_len:  # NOTE: prefix_len < insert_len  # 存在未覆盖的新增部分，需要新建节点挂载
            new_node = RadixTreeNode(self.key_fn)  # 新建节点
            new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:].clone())  # 仅存新增段；clone 物理位置（避免与外部张量共享生命周期）
            new_node.set_parent(node)  # 挂到行走终点下（该处的 key 前缀与新节点互补，天然构成正确字典键）
            self.evictable_size += new_node.length  # 新节点暂无请求引用，计入可淘汰量
            node = new_node  # 句柄指向新节点（含新增段的最深处）
        return InsertResult(prefix_len, RadixCacheHandle(insert_len, node))  # 返回命中长度与句柄（cached_len=insert_len）

    def evict(self, size: int) -> torch.Tensor:
        """按 LRU 淘汰约 size 个空闲 token，返回被淘汰 token 的物理位置（供槽位池回收）。

        算法：收集所有空闲叶节点建最小堆（时间戳最旧优先），循环弹出整节点淘汰；
        节点删除后父节点若变为空闲叶节点则入堆，直到淘汰量达标。
        """
        if size == 0:  # 约定 evict(0) 恒安全、不做事
            return self.empty_tensor
        assert (
            size <= self.evictable_size
        ), f"Cannot evict {size}, only {self.evictable_size} is evictable"  # 请求量不可超过可淘汰量

        leave_nodes = self._collect_leave_nodes_for_evict()  # 收集当前全部"空闲叶节点"
        heapq.heapify(leave_nodes)  # 原地建最小堆（按 __lt__ 即时间戳）
        evicted_indices: List[torch.Tensor] = []  # 被淘汰各段的物理位置收集
        evicted_size = 0  # 已淘汰总量

        while evicted_size < size:  # 未达标则继续淘汰
            assert (
                leave_nodes
            ), f"Cannot evict enough cache, need {size}, only {evicted_size} evicted"  # 堆空仍未达标（理论上不应发生，防御性检查）
            node = heapq.heappop(leave_nodes)  # 弹出最旧的空闲叶节点
            assert node.ref_count == 0 and node.is_leaf() and not node.is_root()  # 必须是空闲叶节点（根永不淘汰）
            evicted_size += node.length  # 累计淘汰量
            evicted_indices.append(node.value)  # 记录其物理位置
            self.evictable_size -= node.length  # 从可淘汰量中扣除
            parent = node.parent  # 取父节点
            del parent.children[self.key_fn(node._key)]  # 从父节点的 children 中摘除本节点
            # NOTE: root is always protected, so won't be evicted  # 父节点若因此变成叶且空闲，则可继续淘汰
            if parent.is_leaf() and parent.ref_count == 0:
                heapq.heappush(leave_nodes, parent)  # 父节点入堆参与后续淘汰

        return torch.cat(evicted_indices)  # 拼接返回全部被淘汰的物理位置

    def reset(self) -> None:
        raise NotImplementedError("RadixManager.reset is not implemented")  # 尚未实现：清空树需遍历释放引用，暂不支持

    @property  # 属性：容量信息
    def size_info(self) -> SizeInfo:
        return SizeInfo(  # 打包当前两个容量计数
            evictable_size=self.evictable_size,
            protected_size=self.protected_size,
        )

    def check_integrity(self) -> None:
        pass  # 一致性自检暂未实现（恒通过）

    def _collect_leave_nodes_for_evict(self) -> List[RadixTreeNode]:
        """深度优先遍历整棵树，收集所有 ref_count==0 的叶节点（可淘汰候选）。"""
        nodes: List[RadixTreeNode] = [self.root_node]  # DFS 显式栈（从根开始）
        leave_nodes: List[RadixTreeNode] = []  # 候选叶节点列表

        while len(nodes) > 0:  # 栈非空则继续遍历
            node = nodes.pop()  # 弹出栈顶
            if node.is_leaf():  # 叶节点
                if node.ref_count == 0:  # 且空闲
                    leave_nodes.append(node)  # 加入候选
            else:
                for child in node.children.values():  # 内部节点：所有子节点入栈
                    nodes.append(child)

        return leave_nodes  # 返回候选列表

    def _tree_walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
        """沿树匹配 input_ids 的最长已知前缀（行走中可能触发节点分裂）。

        返回 (最深命中节点, 命中长度)。匹配按页粒度对齐：页内不满一页的部分不计入命中。
        """
        prefix_len = 0  # 已匹配长度
        indice_len = len(input_ids)  # 目标总长
        node = self.root_node  # 从根出发
        tic = time.monotonic_ns()  # 本次访问时间戳（LRU 更新用）

        while prefix_len < indice_len:  # 还有未匹配的输入
            child_node = node.children.get(self.key_fn(input_ids[prefix_len:]))  # 以"下一个 token（页）"为键查子节点
            if child_node is None:  # 无对应子节点：匹配终止于当前节点
                return node, prefix_len
            node = child_node  # walk to child node  # 下潜到子节点

            # NOTE: at least 1 page is matched, so match_len >= page_size  # 字典键已匹配（至少一个页/首 token 相同），这里算完整段内匹配长度
            match_len = node.get_match_len(input_ids[prefix_len:])  # 与该段 key 逐 token 比较
            match_len = align_down(match_len, self.page_size)  # 按页向下取整：不完整页不算命中（保证物理连续性）
            prefix_len += match_len  # 累计命中长度

            # need to split the node if not fully matched  # 只匹配了段内一部分：在匹配点分裂节点
            if match_len != node.length:
                node = node.split_at(match_len)  # 前缀部分独立成新节点（即新的命中终点）
                node.timestamp = tic  # 更新其时间戳（被访问过）
                return node, prefix_len  # 输入已被此段耗尽或用尽，直接返回

            # update timestamp for accessed node  # 整段全匹配：更新时间戳后继续向下一层
            node.timestamp = tic

        return node, prefix_len  # 输入全部匹配完毕


def _get_key_fn(page_size: int) -> KEY_FN:
    """按页大小生成字典键函数：子节点查找键必须与节点登记键使用同一套规则。

    page_size=1 时键为首个 token 的标量（.item() 转成 Python 数）；
    否则键为首页 token 的元组（可哈希），保证同页 token 前缀可被一次查中。
    """
    if page_size == 1:
        return lambda x: x[0].item()  # 单 token 标量键
    return lambda x: tuple(x[:page_size].tolist())  # 首页元组键
