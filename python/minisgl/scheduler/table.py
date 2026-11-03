import torch  # PyTorch：张量初始化（token 池）


class TableManager:
    """请求槽位管理器：管理最大并发请求数对应的槽位分配/回收，并持有 token 池与页表"""

    def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
        self._max_running_reqs = max_running_reqs  # 最大并发请求数（= 槽位总数）
        self._free_slots = list(range(max_running_reqs))  # 空闲槽位列表：初始全部空闲
        self.page_table = page_table  # 页表引用（与引擎共享同一张量）：[槽位, 位置] -> KV 物理地址
        # NOTE: dummy request also use this pool to get the input ids, so we need to
        # make sure the token pool is initialized with valid values (token_id = 0).
        # NOTE：dummy（补齐）请求也会从 token 池读取输入 id，因此必须初始化为合法值（token_id=0），避免读到未定义数据
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)  # GPU token 池：形状同页表，存每个请求各位置的 token id

    @property
    def available_size(self) -> int:
        return len(self._free_slots)  # 当前可用（空闲）槽位数

    def allocate(self) -> int:
        return self._free_slots.pop()  # 从尾部弹出一个空闲槽位（O(1)），供新请求占用

    def free(self, slot: int) -> None:
        self._free_slots.append(slot)  # 归还槽位到空闲列表，供后续请求复用
