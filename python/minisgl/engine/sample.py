from __future__ import annotations  # 启用延迟注解求值，允许前向引用类型

from dataclasses import dataclass  # dataclass 装饰器：自动生成样板方法
from typing import TYPE_CHECKING, List  # 类型注解工具；TYPE_CHECKING 避免运行时循环导入

import torch  # PyTorch：张量与设备管理
from minisgl.utils import is_sm90_supported, nvtx_annotate  # 工具：GPU 架构判断、NVTL 性能标注装饰器

if TYPE_CHECKING:  # 仅类型检查时导入
    from minisgl.core import Batch  # 批次类型（仅作注解）


@dataclass  # 数据类：批级采样参数容器
class BatchSamplingArgs:
    temperatures: torch.Tensor | None  # 每请求温度张量；None 表示全部走贪心（greedy）解码
    top_k: torch.Tensor | None = None  # 每请求 top-k 张量；None 表示该批无人启用 top-k
    top_p: torch.Tensor | None = None  # 每请求 top-p 张量；None 表示该批无人启用 top-p


def make_device_tensor(data: List, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(data, dtype=dtype, pin_memory=True).to(device, non_blocking=True)  # 锁页内存 + 异步拷贝，隐藏 H2D 传输延迟


def sample_impl(
    logits: torch.Tensor,  # 原始 logits（未归一化）
    temperatures: torch.Tensor,  # 每请求温度
    top_k: torch.Tensor | int | None,  # 每请求 top-k（张量/整数/无）
    top_p: torch.Tensor | float | None,  # 每请求 top-p（张量/浮点/无）
) -> torch.Tensor:
    """调用 flashinfer 内核执行采样；按是否启用 top-k/top-p 选择不同算子路径"""
    import flashinfer.sampling as sampling  # 函数内延迟导入：仅在实际采样时才加载该依赖

    probs = sampling.softmax(logits, temperatures, enable_pdl=is_sm90_supported())  # 温度缩放后 softmax 得到概率分布（Hopper 上启用 PDL 加速）
    if top_k is None and top_p is None:  # 两者都未启用：纯随机采样
        return sampling.sampling_from_probs(probs)  # 直接从完整分布采样

    if top_p is None:  # 只启用 top-k
        assert top_k is not None  # 类型收窄断言
        return sampling.top_k_sampling_from_probs(probs, top_k)  # 先截断到前 k 再采样

    if top_k is None:  # 只启用 top-p
        assert top_p is not None  # 类型收窄断言
        return sampling.top_p_sampling_from_probs(probs, top_p)  # 核采样（累计概率截断）

    assert top_k is not None and top_p is not None  # 两者都启用
    return sampling.top_k_top_p_sampling_from_probs(probs, top_k, top_p)  # 联合 top-k+top-p 采样


@dataclass  # 数据类：采样器（device/vocab_size 为构造参数）
class Sampler:
    device: torch.device  # 采样在哪个设备执行
    vocab_size: int  # 词表大小（用于 top-k 退化为全词表）

    def prepare(self, batch: Batch) -> BatchSamplingArgs:
        """把批内各请求的采样参数整理为设备端张量；能省则省（全贪心/全员默认值时不分配）"""
        params = [r.sampling_params for r in batch.reqs]  # 收集每个请求的采样参数
        if all(p.is_greedy for p in params):  # 整批都是贪心解码
            return BatchSamplingArgs(temperatures=None)  # 温度置 None，采样时直接 argmax

        MIN_P = MIN_T = 1e-6  # 温度/top-p 的数值下限，防止除零或空分布
        ts = [max(0.0 if p.is_greedy else p.temperature, MIN_T) for p in params]  # 贪心请求温度记 0（仍截到下限）
        top_ks = [p.top_k if p.top_k >= 1 else self.vocab_size for p in params]  # top_k<1（如 -1）视为禁用：用全词表大小代替
        top_ps = [min(max(p.top_p, MIN_P), 1.0) for p in params]  # 把 top-p 夹到 [MIN_P, 1.0] 合法区间
        temperatures = make_device_tensor(ts, torch.float32, self.device)  # 温度列表转 GPU 张量
        top_k, top_p = None, None  # 默认不启用两个过滤
        if any(k != self.vocab_size for k in top_ks):  # 只要有一人真正启用 top-k 才分配张量
            top_k = make_device_tensor(top_ks, torch.int32, self.device)  # top-k 转 GPU int32 张量
        if any(p < 1.0 for p in top_ps):  # 只要有一人真正启用 top-p 才分配张量
            top_p = make_device_tensor(top_ps, torch.float32, self.device)  # top-p 转 GPU 浮点张量
        return BatchSamplingArgs(temperatures, top_k=top_k, top_p=top_p)  # 打包为批级参数

    @nvtx_annotate("Sampler")  # 装饰器：在 Nsight profiler 时间线上标注此段为 "Sampler"
    def sample(self, logits: torch.Tensor, args: BatchSamplingArgs) -> torch.Tensor:
        with torch.cuda.nvtx.range("Sampler"):  # 手动 NVTX 范围（与装饰器配合便于性能分析）
            if args.temperatures is None:  # greedy sampling  # 温度为 None 即整批贪心
                return torch.argmax(logits, dim=-1)  # 直接取最大 logit 的下标作为输出 token
            return sample_impl(logits.float(), args.temperatures, args.top_k, args.top_p)  # 随机采样路径（logits 先升到 fp32 保证数值精度）
