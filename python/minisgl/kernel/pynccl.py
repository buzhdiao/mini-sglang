# 允许类型注解使用字符串形式的前向引用
from __future__ import annotations

# functools.cache 缓存模块加载与 FFI 类注册
import functools
# 类型注解工具
from typing import TYPE_CHECKING, Any, Literal

# 环境配置（PyNCCL 最大缓冲上限）
from minisgl.env import ENV

# AOT 编译加载工具
from .utils import load_aot

# 仅类型检查时导入：声明 PyNCCLCommunicator 的接口形状
if TYPE_CHECKING:
    from abc import abstractmethod

    import torch
    from tvm_ffi import Module

    # FFI 通信器的静态接口描述（仅供类型检查）
    class PyNCCLCommunicator:
        # 原地 all_reduce（当前支持 sum）
        @abstractmethod
        def all_reduce(self, input: torch.Tensor, op: Literal["sum"]) -> None: ...
        # 把 input 聚合进 output（0 维拼接）
        @abstractmethod
        def all_gather(self, output: torch.Tensor, input: torch.Tensor) -> None: ...
        # 返回内部缓冲地址（供事件/流互操作）
        @abstractmethod
        def get_buffer(self) -> int: ...

else:
    # 运行时用 Any 代替，绕开 FFI 对象的类型检查
    PyNCCLCommunicator = Any


# AOT 编译并加载 pynccl.cu（链接 NCCL 库），只编译一次
@functools.cache
def _load_nccl_module() -> Module:
    # 编译 pynccl.cu 并额外链接 -lnccl
    return load_aot("pynccl", cuda_files=["pynccl.cu"], extra_ldflags=["-lnccl"])


# 把 C++ 侧的 minisgl.NCCLWrapper 注册为可从 Python 构造的 tvm-ffi 对象类
@functools.cache
def _get_pynccl_wrapper_cls():
    # 导入 tvm-ffi 主体
    import tvm_ffi

    # 注册 FFI 对象类型：Python 侧类与 C++ 侧 NCCLWrapper 一一对应
    @tvm_ffi.register_object("minisgl.NCCLWrapper")
    class PyNCCLImpl(tvm_ffi.Object):
        # 构造时转调 C++ 侧的初始化
        def __init__(self, *args):
            # 调用 FFI 生成的初始化
            self.__ffi_init__(*args)

    # 返回该类
    return PyNCCLImpl


# 初始化 PyNCCL 通信器：rank0 生成 NCCL unique id 并广播给全部 TP rank。
# tp_rank/tp_size：本 rank 在 TP 组中的序号与组大小；tp_cpu_group：CPU 进程组（广播 bootstrap 信息）；
# max_size_bytes：allreduce 自定义缓冲的最大字节数
def init_pynccl(
    *,
    tp_rank: int,
    tp_size: int,
    tp_cpu_group: torch.distributed.ProcessGroup,
    max_size_bytes: int = 0,
) -> PyNCCLCommunicator:
    # 运行时导入 torch
    import torch

    # 缓冲上限受环境变量约束（防止自定义算法在大消息上退化）
    max_size_bytes = min(max_size_bytes, ENV.PYNCCL_MAX_BUFFER_SIZE.value)

    # 加载（或编译）NCCL 封装模块
    module = _load_nccl_module()
    # 取注册好的 FFI 包装类
    cls = _get_pynccl_wrapper_cls()

    # rank0 生成 NCCL unique id 并广播；其余 rank 只接收
    if tp_rank == 0:
        # rank0 创建唯一 id
        id_list = [module.create_nccl_uid()]
        # 通过 CPU 进程组广播给全部 rank
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )
    else:
        # 非 rank0 先占位
        id_list = [None]
        # 接收 rank0 广播的 id
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )

    # 取出 NCCL unique id
    nccl_id = id_list[0]
    # 必须成功拿到 id 才能建通信器
    assert not nccl_id is None, f"Failed to get NCCL unique ID on {tp_rank = }"

    # bypass type checking for the FFI object
    # 用 id/规模/缓冲上限构造 C++ 侧通信器
    return cls(tp_rank, tp_size, max_size_bytes, nccl_id)  # type: ignore
