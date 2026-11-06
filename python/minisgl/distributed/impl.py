# 允许类型注解使用字符串形式的前向引用
from __future__ import annotations

# ABC/abstractmethod 定义通信实现的抽象接口
from abc import ABC, abstractmethod
# dataclass 声明实现类
from dataclasses import dataclass
# TYPE_CHECKING 仅类型检查时导入；List 用于注解
from typing import TYPE_CHECKING, List

# torch 提供张量
import torch
# torch.distributed 提供进程组通信原语
import torch.distributed as dist

# 仅类型检查时导入，避免运行时循环依赖
if TYPE_CHECKING:
    from minisgl.distributed import DistributedInfo
    from minisgl.kernel import PyNCCLCommunicator


# 分布式通信实现的抽象基类：TP 并行只依赖这两个集合通信原语
@dataclass
class DistributedImpl(ABC):
    # 张量并行中：行并行 Linear 输出按 rank 分片，需 all_reduce 求和还原完整结果
    @abstractmethod
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor: ...

    # 序列并行/按 head 分片时：把各 rank 的分片拼接成完整张量
    @abstractmethod
    def all_gather(self, x: torch.Tensor) -> torch.Tensor: ...


# 基于 torch.distributed(NCCL) 的默认实现
@dataclass
class TorchDistributedImpl(DistributedImpl):
    # 对 x 做跨 TP 组的求和归约
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        # 取 TP 组的进程数
        tp_size = dist.get_world_size()
        # 单卡 TP 时无需通信，直接返回
        if tp_size == 1:
            # 原样返回
            return x
        # 原地执行 NCCL all_reduce 求和（各 rank 得到相同的完整结果）
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        # 返回归约后的张量
        return x

    # 把各 rank 的分片在 0 维上拼接成完整张量
    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        # 取 TP 组的进程数
        tp_size = dist.get_world_size()
        # 单卡无需通信
        if tp_size == 1:
            # 原样返回
            return x
        # 复制形状准备输出
        shape = list(x.shape)
        # 0 维按进程数扩展（每 rank 一份分片）
        shape[0] = shape[0] * tp_size
        # 预分配输出缓冲
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
        # 用 NCCL 的 gather-into-tensor 原语：比逐 rank gather 更高效
        dist.all_gather_into_tensor(out, x)
        # 返回拼接结果
        return out


# 基于 minisgl 自研 PyNCCL 绑定的实现（支持自定义 allreduce 算法）
@dataclass
class PyNCCLDistributedImpl(DistributedImpl):
    # 持有的 PyNCCL 通信器
    comm: PyNCCLCommunicator

    # 用自研通道做 all_reduce（对中小消息有更低延迟的定制实现）
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        # 调用 C++/CUDA 侧实现，原地求和
        self.comm.all_reduce(x, "sum")
        # 返回结果
        return x

    # 用自研通道做 all_gather
    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        # 函数内导入，避免循环依赖
        from .info import get_tp_info

        # 取 TP 进程数
        world_size = get_tp_info().size
        # 计算输出形状
        output_shape = list(x.shape)
        # 0 维按进程数扩展
        output_shape[0] *= world_size
        # 在同设备同类型上预分配输出
        result = x.new_empty(output_shape)
        # 执行 all_gather
        self.comm.all_gather(result, x)
        # 返回拼接结果
        return result


# 对外统一的通信入口：通过插件栈转发，最后注册的实现优先
class DistributedCommunicator:
    # 插件列表：默认只有 torch.distributed 实现
    plugins: List[DistributedImpl] = [TorchDistributedImpl()]

    # 委派给最新注册的插件（PyNCCL 启用后覆盖默认实现）
    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        # 取栈顶插件执行
        return self.plugins[-1].all_reduce(x)

    # 同上，委派 all_gather
    def all_gather(self, x: torch.Tensor) -> torch.Tensor:
        # 取栈顶插件执行
        return self.plugins[-1].all_gather(x)


# 初始化 PyNCCL 通信并注册为新插件（TP>1 才有意义）
def enable_pynccl_distributed(
    tp_info: DistributedInfo, tp_cpu_group: torch.distributed.ProcessGroup, max_bytes: int
) -> None:
    """
    Enable PyNCCL-based distributed communication for tensor parallelism.
    """
    # 单卡无通信需求
    if tp_info.size == 1:
        # 直接返回
        return
    # 延迟导入内核层，保持依赖最小
    from minisgl.kernel import init_pynccl

    # 用 TP 的 rank/size 与 CPU 进程组（用于 bootstrap/同步）创建通信器
    comm = init_pynccl(
        tp_rank=tp_info.rank,
        tp_size=tp_info.size,
        tp_cpu_group=tp_cpu_group,
        max_size_bytes=max_bytes,
    )

    # 追加到插件栈末尾：此后所有 all_reduce/all_gather 都走 PyNCCL
    DistributedCommunicator.plugins.append(PyNCCLDistributedImpl(comm))


# 销毁全部通信插件（进程退出时调用，释放 NCCL 资源）
def destroy_distributed() -> None:
    """
    Destroy all the distributed communication plugins.
    """
    # 清空插件列表，释放底层通信器引用
    DistributedCommunicator.plugins = []
