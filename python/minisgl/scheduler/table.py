import torch


# 请求槽位管理器：管理"页表的行号"（每条请求占一行，行内列=token 位置）
class TableManager:
    def __init__(self, max_running_reqs: int, page_table: torch.Tensor) -> None:
        # 最大并发请求数（即页表行数上限）
        self._max_running_reqs = max_running_reqs
        # 空闲行号栈：用列表实现，pop 取尾部、append 还尾部，O(1)
        self._free_slots = list(range(max_running_reqs))
        # 共享的全局页表（行=请求槽位，列=token 位置，值=KV 池槽位下标）
        self.page_table = page_table
        # NOTE: dummy request also use this pool to get the input ids, so we need to
        # make sure the token pool is initialized with valid values (token_id = 0).
        # token 池与页表同形状：存放每个请求每个位置的 token id。
        # 必须初始化为 0：CUDA Graph 回放时 dummy（padding）请求也会从这里读
        # 输入 token，未初始化内存可能含非法值导致数值异常
        self.token_pool = torch.zeros_like(page_table, dtype=torch.int32)

    @property
    def available_size(self) -> int:
        # 还能接纳的请求数 = 剩余空闲行数
        return len(self._free_slots)

    def allocate(self) -> int:
        # 分配一个槽位行号（弹出栈尾）
        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        # 归还槽位行号（压回栈尾，供后续请求复用）
        self._free_slots.append(slot)
