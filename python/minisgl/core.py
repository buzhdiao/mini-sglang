from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解中使用前向引用（如 Context | None）

from contextlib import contextmanager  # contextmanager 装饰器：把生成器函数包装为 with 上下文管理器
from dataclasses import dataclass, field  # dataclass 装饰器与 field：声明数据类及带特殊行为的字段
from typing import TYPE_CHECKING, List, Literal  # 类型注解工具：TYPE_CHECKING（仅类型检查期导入）、List、Literal（字面量类型）

import torch  # PyTorch 主库：张量与 GPU 计算

if TYPE_CHECKING:  # 仅静态类型检查时导入，避免运行时循环依赖
    from minisgl.attention import BaseAttnBackend, BaseAttnMetadata  # 注意力后端与其元数据（仅注解用）
    from minisgl.kvcache import BaseCacheHandle, BaseKVCachePool  # KV 缓存句柄与缓存池（仅注解用）
    from minisgl.moe import BaseMoeBackend  # MoE（专家混合）后端（仅注解用）


@dataclass  # 数据类装饰器：自动生成 __init__、__repr__ 等样板方法
class SamplingParams:
    """采样参数：控制模型生成 token 时的随机性策略"""

    temperature: float = 0.0  # 温度系数：0 表示贪心取最大概率；>0 时按温度缩放 logits 引入随机性
    top_k: int = -1  # Top-K 采样：只在概率最高的 k 个 token 中采样；-1 表示不启用
    top_p: float = 1.0  # 核采样：只在累积概率达到 top_p 的最小 token 集合中采样；1.0 表示不启用
    ignore_eos: bool = False  # 是否忽略 EOS 结束符：True 时即使采样到 EOS 也继续生成
    max_tokens: int = 1024  # 最多生成的输出 token 数上限

    @property  # 只读属性：像字段一样访问
    def is_greedy(self) -> bool:
        # 是否为贪心解码：温度<=0 或 top_k==1（二者都等价于只取 argmax），且未启用 top_p
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


@dataclass(eq=False)  # eq=False：不生成 __eq__，保留对象身份比较，使 Req 可按"同一对象"语义放入 set/dict
class Req:
    """单个用户请求的运行时状态：跟踪序列长度与 KV Cache 占用的演进"""

    input_ids: torch.Tensor  # cpu tensor  # 完整 token 序列（存于 CPU 主存，含 prompt 与已生成部分）
    table_idx: int  # 该请求在 token 池/页表中占用的槽位行号
    cached_len: int  # 已在 KV Cache 中（已计算过）的前缀长度
    output_len: int  # 期望生成的输出 token 数上限
    uid: int  # 请求唯一标识符：用于消息路由、中止与排序
    sampling_params: SamplingParams  # 该请求的采样参数
    cache_handle: BaseCacheHandle  # 前缀缓存句柄：记录其在 radix cache 中命中的前缀与锁定状态

    def __post_init__(self) -> None:  # dataclass 生成的 __init__ 之后自动调用的钩子：派生字段 + 不变量校验
        assert self.input_ids.is_cpu  # 约束 input_ids 必须在 CPU（GPU 侧 token 池由调度器另行拷贝）
        self.device_len = len(self.input_ids)  # 设备（GPU KV Cache）侧当前序列长度，初始等于输入长度
        self.max_device_len = len(self.input_ids) + self.output_len  # 序列长度上界 = 输入 + 最大输出
        assert 0 <= self.cached_len < self.device_len <= self.max_device_len  # 长度不变量：cached < device <= max

    @property
    def remain_len(self) -> int:
        # 还需生成的 token 数 = 长度上界 - 当前设备侧长度
        return self.max_device_len - self.device_len

    @property
    def extend_len(self) -> int:
        # 本轮前向需新计算的 token 数 = 设备侧长度 - 已缓存长度（prefill/chunk 时 >1，decode 时恰为 1）
        return self.device_len - self.cached_len

    def complete_one(self) -> None:
        # 完成一步解码：当前全部长度标记为已缓存，设备长度 +1（新采样 token 尚未进 KV Cache）
        self.cached_len = self.device_len
        self.device_len += 1

    def append_host(self, next_token: torch.Tensor) -> None:
        # 把新采样出的 token 追加到 CPU 侧序列记录（主机侧的完整对话历史）
        self.input_ids = torch.cat([self.input_ids, next_token])

    @property
    def can_decode(self) -> bool:
        # 只要还有剩余生成长度就可继续解码（分块请求通过覆写此属性恒返回 False）
        return self.remain_len > 0

    def __repr__(self) -> str:
        # 自定义调试输出：突出关键长度字段，便于日志排查
        return (
            f"{type(self)}(table_idx={self.table_idx}, "
            f"cached_len={self.cached_len}, device_len={self.device_len}, "
            f"max_device_len={self.max_device_len})"
        )


@dataclass
class Batch:
    """一次调度产生的一批请求：携带前向计算所需的全部输入张量与元数据"""

    reqs: List[Req]  # 本批次的真实请求列表
    phase: Literal["prefill", "decode"]  # 批次阶段：prefill（预填充）或 decode（解码）
    # these fields should be set by scheduler  # 以下字段由调度器填充（init=False：不出现在 __init__ 参数中）
    input_ids: torch.Tensor = field(init=False)  # 输入 token id 张量（由 token 池按索引取出）
    positions: torch.Tensor = field(init=False)  # 每个 token 的位置编码索引
    out_loc: torch.Tensor = field(init=False)  # 新 KV 写入缓存池的物理位置（查页表得到）
    padded_reqs: List[Req] = field(init=False)  # 补齐后的请求列表（可能含 dummy req，用于对齐 CUDA graph）
    # this field should be set by attention backend  # 此字段由注意力后端设置
    attn_metadata: BaseAttnMetadata = field(init=False)  # 注意力核函数所需的元数据

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"  # 是否为 prefill 批次

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"  # 是否为 decode 批次

    @property
    def size(self) -> int:
        return len(self.reqs)  # 真实请求数

    @property
    def padded_size(self) -> int:
        return len(self.padded_reqs)  # 补齐后的请求数（>= size，差值即 dummy req）


@dataclass
class Context:
    """引擎全局上下文：持有页表、各计算后端与 KV 缓存池，并跟踪当前活跃批次"""

    page_size: int  # KV Cache 页大小（每页容纳的 token 数）
    # NOTE: this table always treat page_size = 1  # 注意：此页表始终按 page_size=1（逐 token）粒度存储映射
    page_table: torch.Tensor = field(init=False)  # 页表 [槽数, 最大长度]：逻辑位置 -> KV 物理地址
    attn_backend: BaseAttnBackend = field(init=False)  # 注意力计算后端
    moe_backend: BaseMoeBackend = field(init=False)  # MoE 计算后端
    kv_cache: BaseKVCachePool = field(init=False)  # KV 缓存池（GPU 显存中的 K/V 张量）
    _batch: Batch | None = field(default=None, init=False)  # 当前正在前向的批次（内部状态）

    @property
    def batch(self) -> Batch:
        # 取当前活跃批次；不存在时断言失败（防止在无批次上下文中误用）
        assert self._batch is not None, "No active batch in context"
        return self._batch

    @contextmanager  # 把生成器函数包装为上下文管理器：进入时执行到 yield，退出时执行 finally
    def forward_batch(self, batch: Batch):
        # 登记当前前向批次的上下文；禁止嵌套（一次只允许一个批次在前向）
        assert self._batch is None, "Nested forward_batch is not allowed"
        try:
            self._batch = batch  # 进入上下文：登记批次
            yield  # 挂起点：with 块内的代码在此执行
        finally:
            self._batch = None  # 退出上下文：无论是否异常都清空批次


_GLOBAL_CTX: Context | None = None  # 模块级全局上下文单例（每进程一个）


def set_global_ctx(ctx: Context):
    # 设置全局上下文：只允许设置一次（重复设置视为编程错误）
    global _GLOBAL_CTX  # 声明修改模块级变量而非局部
    assert _GLOBAL_CTX is None, "Global context is already set"
    _GLOBAL_CTX = ctx


def get_global_ctx() -> Context:
    # 获取全局上下文：未初始化时断言失败（提示调用顺序错误）
    assert _GLOBAL_CTX is not None, "Global context is not set"
    return _GLOBAL_CTX
