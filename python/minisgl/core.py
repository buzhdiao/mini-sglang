# 允许类型注解使用字符串形式的前向引用
from __future__ import annotations

# contextmanager 定义上下文管理器
from contextlib import contextmanager
# dataclass/field 声明数据类
from dataclasses import dataclass, field
# 类型注解工具
from typing import TYPE_CHECKING, List, Literal

# torch 提供张量
import torch

# 仅类型检查时导入，避免运行时循环依赖
if TYPE_CHECKING:
    from minisgl.attention import BaseAttnBackend, BaseAttnMetadata
    from minisgl.kvcache import BaseCacheHandle, BaseKVCachePool
    from minisgl.moe import BaseMoeBackend


# 采样参数：控制解码的随机性与停止条件
@dataclass
class SamplingParams:
    # 温度：0 表示贪心，越大越随机
    temperature: float = 0.0
    # top-k 截断：-1 表示不启用
    top_k: int = -1
    # top-p 截断（核采样）：1.0 表示不截断
    top_p: float = 1.0
    # 是否忽略 EOS 强制生成到 max_tokens（压测常用）
    ignore_eos: bool = False
    # 最大生成 token 数
    max_tokens: int = 1024

    # 判断是否为纯贪心解码（无随机性，可用于优化采样路径）
    @property
    def is_greedy(self) -> bool:
        # 温度为 0 或 top_k=1 都等价于取 argmax，且要求 top_p 不截断
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


# 一条请求：输入、KV cache 表项与生命周期进度（eq=False 用 id 做相等性，避免张量比较）
@dataclass(eq=False)
class Req:
    # CPU 上的输入 token 序列（含逐步追加的输出 token）
    input_ids: torch.Tensor  # cpu tensor
    # 在全局页表中的行号（该请求的 KV 槽位映射）
    table_idx: int
    # 已存在于 KV cache 的长度（radix cache 命中/已写入部分）
    cached_len: int
    # 允许的最大输出 token 数
    output_len: int
    # 请求唯一 id
    uid: int
    # 该请求的采样参数
    sampling_params: SamplingParams
    # KV cache 池的句柄（管理该请求的槽位）
    cache_handle: BaseCacheHandle

    # 构造后立即派生长度字段并做一致性校验
    def __post_init__(self) -> None:
        # 输入必须在 CPU（调度/统计在 CPU 侧做）
        assert self.input_ids.is_cpu
        # 当前在 GPU KV cache 中的长度，初始为输入长度
        self.device_len = len(self.input_ids)
        # 全生命周期的最大长度 = 输入 + 输出
        self.max_device_len = len(self.input_ids) + self.output_len
        # 校验长度关系：缓存命中 < 当前长度 <= 最大长度
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len

    # 还需生成的 token 数
    @property
    def remain_len(self) -> int:
        # 最大长度减当前长度
        return self.max_device_len - self.device_len

    # 本次 forward 需要计算的新 token 数（总长 - 已缓存）
    @property
    def extend_len(self) -> int:
        # 当前长度减缓存命中长度
        return self.device_len - self.cached_len

    # decode 一步完成：新 token 正式进入 KV cache
    def complete_one(self) -> None:
        # 原来的"未落盘"部分变为已缓存
        self.cached_len = self.device_len
        # 总长度加 1（刚生成的 token）
        self.device_len += 1

    # 把新 token 追加到 CPU 侧输入序列（保留完整历史）
    def append_host(self, next_token: torch.Tensor) -> None:
        # 拼接张量生成新序列
        self.input_ids = torch.cat([self.input_ids, next_token])

    # 是否还能继续 decode（未到输出上限）
    @property
    def can_decode(self) -> bool:
        # 还有剩余额度即可继续
        return self.remain_len > 0

    # 调试输出：打印各长度字段
    def __repr__(self) -> str:
        # 组装可读字符串
        return (
            f"{type(self)}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )


# 一个调度批次：一组请求 + 阶段标记 + 由调度器/后端填充的张量字段
@dataclass
class Batch:
    # 本批的请求列表
    reqs: List[Req]
    # 批次阶段：prefill（整段新输入）或 decode（每请求 1 个新 token）
    phase: Literal["prefill", "decode"]
    # these fields should be set by scheduler
    # 以下字段由调度器填充（init=False 表示构造时不传）
    # 拼接后的输入 token（packed 一维）
    input_ids: torch.Tensor = field(init=False)
    # 每个 token 的位置 id（RoPE 用）
    positions: torch.Tensor = field(init=False)
    # 每个新 token 在 KV cache 中的写入槽位
    out_loc: torch.Tensor = field(init=False)
    # 填充对齐后的请求列表（CUDA Graph 需要固定批量）
    padded_reqs: List[Req] = field(init=False)
    # this field should be set by attention backend
    # 由注意力后端填充的元数据
    attn_metadata: BaseAttnMetadata = field(init=False)

    # 是否为 prefill 批
    @property
    def is_prefill(self) -> bool:
        # 比较阶段标记
        return self.phase == "prefill"

    # 是否为 decode 批
    @property
    def is_decode(self) -> bool:
        # 比较阶段标记
        return self.phase == "decode"

    # 真实请求数
    @property
    def size(self) -> int:
        # 请求列表长度
        return len(self.reqs)

    # 填充后的批量大小（图回放时 >= 真实请求数）
    @property
    def padded_size(self) -> int:
        # 填充请求列表长度
        return len(self.padded_reqs)


# 全局上下文：一个进程内共享的页表/后端/KV cache 与"当前批次"
@dataclass
class Context:
    # KV cache 分页大小
    page_size: int
    # NOTE: this table always treat page_size = 1
    # 全局页表：行=请求，列=槽位序号；始终按 token 粒度（等效 page_size=1）存储
    page_table: torch.Tensor = field(init=False)
    # 选定的注意力后端
    attn_backend: BaseAttnBackend = field(init=False)
    # 选定的 MoE 后端
    moe_backend: BaseMoeBackend = field(init=False)
    # KV cache 池
    kv_cache: BaseKVCachePool = field(init=False)
    # 当前正在 forward 的批次（无则为 None）
    _batch: Batch | None = field(default=None, init=False)

    # 获取当前批次，必须存在
    @property
    def batch(self) -> Batch:
        # 不允许在无批次时访问
        assert self._batch is not None, "No active batch in context"
        # 返回当前批次
        return self._batch

    # 上下文管理器：进入时登记当前批次，退出时清空（保证不嵌套）
    @contextmanager
    def forward_batch(self, batch: Batch):
        # 不允许嵌套 forward
        assert self._batch is None, "Nested forward_batch is not allowed"
        # 保证异常时也能清理
        try:
            # 登记当前批次
            self._batch = batch
            # 让出控制权给 forward 主体
            yield
        finally:
            # 无论成败都清空登记
            self._batch = None


# 模块级全局上下文单例
_GLOBAL_CTX: Context | None = None


# 设置全局上下文（只允许设置一次）
def set_global_ctx(ctx: Context):
    # 声明修改模块级变量
    global _GLOBAL_CTX
    # 重复设置视为编程错误
    assert _GLOBAL_CTX is None, "Global context is already set"
    # 保存上下文
    _GLOBAL_CTX = ctx


# 获取全局上下文（必须已设置）
def get_global_ctx() -> Context:
    # 未设置时直接暴露错误
    assert _GLOBAL_CTX is not None, "Global context is not set"
    # 返回单例
    return _GLOBAL_CTX
