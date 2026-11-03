from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解中使用前向引用

from abc import ABC, abstractmethod  # 抽象基类：定义通信接口规范，强制子类实现
from dataclasses import dataclass  # 数据类装饰器：自动生成 __init__/__repr__ 等
from typing import TYPE_CHECKING, List  # 类型工具

import torch  # 张量库
import torch.distributed as dist  # PyTorch 官方分布式通信模块（NCCL/gloo 后端）

if TYPE_CHECKING:  # 仅类型检查时导入，避免运行时循环依赖
    from minisgl.distributed import DistributedInfo
    from minisgl.kernel import PyNCCLCommunicator


@dataclass
class DistributedImpl(ABC):
    """分布式通信实现的抽象基类：规定 TP 并行所需的两个核心集合通信原语"""

    @abstractmethod  # 抽象方法：子类必须实现，否则实例化报错
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor: ...  # 全规约：所有 rank 的 x 求和并广播（TP 中用于合并各卡的部分和，如 logits）

    @abstractmethod  # 抽象方法：子类必须实现
    def all_gather(self, x: torch.Tensor) -> torch.Tensor: ...  # 全收集：各 rank 的 x 沿第 0 维拼接（TP 中用于汇总切分后的张量）


@dataclass
class TorchDistributedImpl(DistributedImpl):
    """基于 torch.distributed 的默认通信实现：通用性最好，作为保底后端"""

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()  # 当前进程组的世界大小（即 TP 并行度）
        if tp_size == 1:  # 单卡无通信需求，直接返回省开销
            return x
        dist.all_reduce(x, op=dist.ReduceOp.SUM)  # 原地全规约求和：完成后每个 rank 的 x 都变为全局和
        return x  # 返回同一块被原地修改的张量

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        tp_size = dist.get_world_size()  # 世界大小
        if tp_size == 1:  # 单卡直接返回
            return x
        shape = list(x.shape)  # 计算输出形状
        shape[0] = shape[0] * tp_size  # 第 0 维放大 tp_size 倍（各 rank 数据按 rank 序首尾拼接）
        out = torch.empty(shape, dtype=x.dtype, device=x.device)  # 预分配输出缓冲区
        dist.all_gather_into_tensor(out, x)  # 高效全收集：单张量接口（比逐条 all_gather 列表版少一次拷贝）
        return out


@dataclass
class PyNCCLDistributedImpl(DistributedImpl):
    """基于自研 PyNCCL 绑定的通信实现：更细粒度控制 NCCL 行为，性能通常更优"""

    comm: PyNCCLCommunicator  # C++/CUDA 侧通信器句柄（由 kernel 模块初始化）

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        self.comm.all_reduce(x, "sum")  # 调用 PyNCCL 的原地求和全规约
        return x  # 返回原地修改后的张量

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        from .info import get_tp_info  # 局部导入，避免模块加载期的循环依赖

        world_size = get_tp_info().size  # 从全局 TP 信息取并行度（而非 dist.get_world_size）
        output_shape = list(x.shape)  # 计算输出形状
        output_shape[0] *= world_size  # 第 0 维放大 world_size 倍
        result = x.new_empty(output_shape)  # 按 x 的 dtype/device 分配输出
        self.comm.all_gather(result, x)  # PyNCCL 全收集
        return result


class DistributedCommunicator:
    """TP 通信门面（facade）：对外统一 all_reduce/all_gather 接口，内部维护插件栈。
    新后端通过 append 压栈，通信总是走栈顶（最后启用的）实现。
    注意：plugins 是类属性，所有实例共享同一份插件栈。
    """

    plugins: List[DistributedImpl] = [TorchDistributedImpl()]  # 类属性：实现插件栈，默认只有 torch.distributed 保底实现

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_reduce(x)  # 委托给栈顶（最新启用）的实现

    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        return self.plugins[-1].all_gather(x)  # 委托给栈顶实现


def enable_pynccl_distributed(
    tp_info: DistributedInfo, tp_cpu_group: torch.distributed.ProcessGroup, max_bytes: int
) -> None:
    """
    Enable PyNCCL-based distributed communication for tensor parallelism.
    启用 PyNCCL 通信后端：初始化通信器并压入插件栈，此后所有集合通信走 PyNCCL 路径。
    """
    if tp_info.size == 1:  # 单卡没有通信需求，无需初始化 NCCL
        return
    from minisgl.kernel import init_pynccl  # 运行时导入 kernel 模块（内含 CUDA 扩展，导入较重）

    comm = init_pynccl(  # 创建 PyNCCL 通信器
        tp_rank=tp_info.rank,  # 当前进程在 TP 组内的 rank
        tp_size=tp_info.size,  # TP 并行度
        tp_cpu_group=tp_cpu_group,  # CPU 进程组：NCCL 初始化阶段的引导/同步依赖它
        max_size_bytes=max_bytes,  # 预分配 NCCL 通信缓冲区上限（字节）
    )

    DistributedCommunicator.plugins.append(PyNCCLDistributedImpl(comm))  # 压栈：此后通信走栈顶的 PyNCCL 实现


def destroy_distributed() -> None:
    """
    Destroy all the distributed communication plugins.
    销毁所有通信插件：直接清空插件栈（仅用于进程退出前清理；
    注意清空后并未恢复默认实现，之后再通信会因 plugins[-1] 越界报错）。
    """
    DistributedCommunicator.plugins = []  # 重置为空列表，让 NCCL 相关资源可被回收
