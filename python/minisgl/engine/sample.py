# 允许类型注解使用字符串形式的前向引用
from __future__ import annotations

# dataclass 声明采样参数/采样器
from dataclasses import dataclass
# TYPE_CHECKING 仅类型检查时导入；List 用于注解
from typing import TYPE_CHECKING, List

# torch 提供张量与 NVTX 范围
import torch
# 工具：架构判断与 NVTX 性能标注
from minisgl.utils import is_sm90_supported, nvtx_annotate

# 仅类型检查时导入 Batch
if TYPE_CHECKING:
    from minisgl.core import Batch


# 一个批次的采样参数（GPU 张量，逐请求一组）
@dataclass
class BatchSamplingArgs:
    # 温度张量；None 表示整批都是贪心解码（走 argmax 快路径）
    temperatures: torch.Tensor | None
    # 逐请求 top-k；None 表示整批无 top-k 截断
    top_k: torch.Tensor | None = None
    # 逐请求 top-p；None 表示整批无 top-p 截断
    top_p: torch.Tensor | None = None


# 从 CPU 数据创建设备张量：经 pinned 内存异步上卡，隐藏 H2D 延迟
def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    # 先建 pinned CPU 张量，再 non_blocking 拷到 GPU
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)


# 用 FlashInfer 内核执行采样：按 top_k/top_p 的组合选择不同 kernel
def sample_impl(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor | int | None,
    top_p: torch.Tensor | float | None,
) -> torch.Tensor:
    # 延迟导入 flashinfer 采样模块
    import flashinfer.sampling as sampling

    # 先按温度做 softmax 得到概率分布（SM90 启用 PDL 重叠执行）
    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())
    # 无任何截断：直接从概率分布采样
    if top_k is None and top_p is None:
        # 从 probs 按行采样
        return sampling.sampling_from_probs(probs)

    # 仅 top-k 截断
    if top_p is None:
        # 此时 top_k 必须存在
        assert top_k is not None
        # 执行 top-k 采样 kernel
        return sampling.top_k_sampling_from_probs(probs, top_k)

    # 仅 top-p 截断
    if top_k is None:
        # 此时 top_p 必须存在
        assert top_p is not None
        # 执行 top-p 采样 kernel
        return sampling.top_p_sampling_from_probs(probs, top_p)

    # 同时启用 top-k 与 top-p
    assert top_k is not None and top_p is not None
    # 执行联合截断采样 kernel
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)


# 采样器：管理设备并把批次参数转成 GPU 张量后执行采样
@dataclass
class Sampler:
    # 采样所在设备
    device: torch.device
    # 词表大小（top_k 未启用时用词表大小兜底）
    vocab_size: int

    # 把批内每条请求的采样参数整理成 GPU 张量；全贪心时走快路径
    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        # 取每条请求的采样参数
        params = [r.sampling_params for r in batch.reqs]
        # 整批都是贪心：无需温度/top-k/top-p 张量
        if all(p.is_greedy for p in params):
            # 温度为 None 标记贪心路径
            return BatchSamplingArgs(temperatures=None)

        # 温度与 top-p 的最小保护值，避免除零/空分布
        MIN_P = MIN_T = 1e-6
        # 贪心请求温度按 0 处理，再 clamp 到 MIN_T（softmax 需要正温度）
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]
        # top_k<1 视为不启用，用词表大小代替（等于不截断）
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]
        # top_p clamp 到 [MIN_P, 1.0]
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]
        # 温度上卡（float32）
        temperatures = make_device_tensor(ts, torch.float32, self.device)
        # 先假定都不启用
        top_k, top_p = None, None
        # 只要有一个请求启用了 top-k 才上卡该张量
        if any(k != self.vocab_size for k in top_ks):
            # 构造 top-k 张量（int32）
            top_k = make_device_tensor(top_ks, torch.int32, self.device)
        # 只要有一个请求的 top-p <1 才上卡
        if any(p < 1.0 for p in top_ps):
            # 构造 top-p 张量（float32）
            top_p = make_device_tensor(top_ps, torch.float32, self.device)
        # 返回整理好的批量参数
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p)

    # 对 logits 做采样得到 next token（贪心走 argmax，否则走 FlashInfer）
    @nvtx_annotate("Sampler")
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        # NVTX 范围便于 nsight 分析
        with torch.cuda.nvtx.range("Sampler"):
            # 贪心：直接取 logits 最大的下标
            if args.temperatures is None:  # greedy sampling
                # 沿词表维度 argmax
                return torch.argmax(logits, dim=-1)
            # 非贪心：float32 精度下执行概率采样
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)
