from __future__ import annotations  # 启用 PEP 563 延迟注解求值，支持前向引用类型注解

import functools  # 提供 functools.cache，缓存编译结果与 FFI 包装类
from typing import TYPE_CHECKING, Any, Literal  # TYPE_CHECKING：仅静态检查导入；Any：运行时占位类型；Literal：字面量类型

from minisgl.env import ENV  # 全局环境变量配置对象，读取 NCCL 缓冲区大小上限

from .utils import load_aot  # AOT 预编译库加载工具

if TYPE_CHECKING:  # 仅静态检查时生效：声明通信器接口，供类型注解使用（运行时无此定义）
    from abc import abstractmethod  # 抽象方法装饰器

    import torch  # 张量库
    from tvm_ffi import Module  # tvm-ffi 动态库模块类型

    class PyNCCLCommunicator:  # NCCL 通信器的类型存根：仅描述接口，不含实现
        @abstractmethod  # 抽象方法标记（静态检查用）
        def all_reduce(self, input: torch.Tensor, op: Literal["sum"]) -> None: ...  # 全局规约（当前仅支持求和）
        @abstractmethod  # 抽象方法标记（静态检查用）
        def all_gather(self, output: torch.Tensor, input: torch.Tensor) -> None: ...  # 全收集
        @abstractmethod  # 抽象方法标记（静态检查用）
        def get_buffer(self) -> int: ...  # 获取内部通信缓冲区（返回地址/句柄）

else:
    PyNCCLCommunicator = Any  # 运行时该名字只是 Any，实际对象由 FFI 包装类创建


@functools.cache  # 缓存加载结果：pynccl 动态库整个进程只加载一次
def _load_nccl_module() -> Module:
    """加载 AOT 预编译的 pynccl 动态库（链接 NCCL C++ 库）。"""
    # 加载 csrc 中预编译产物 pynccl.cu 对应的库，链接 -lnccl
    return load_aot("pynccl", cuda_files=["pynccl.cu"], extra_ldflags=["-lnccl"])


@functools.cache  # 缓存类定义：FFI 包装类只注册一次
def _get_pynccl_wrapper_cls():
    """构造并返回 NCCLWrapper 的 tvm-ffi FFI 包装类（进程内仅构造一次）。"""
    import tvm_ffi  # tvm-ffi 运行时，用于把 C++ 对象暴露给 Python

    @tvm_ffi.register_object("minisgl.NCCLWrapper")  # 将该 Python 类注册为 C++ 端 minisgl.NCCLWrapper 类型的反射包装
    class PyNCCLImpl(tvm_ffi.Object):  # 继承 FFI 对象基类，实例由 C++ 侧持有
        def __init__(self, *args):
            self.__ffi_init__(*args)  # 调用 C++ 构造函数（按位置传参：rank, size, max_bytes, nccl_id）

    return PyNCCLImpl  # 返回包装类，供 init_pynccl 实例化


def init_pynccl(
    *,
    tp_rank: int,
    tp_size: int,
    tp_cpu_group: torch.distributed.ProcessGroup,
    max_size_bytes: int = 0,
) -> PyNCCLCommunicator:
    """初始化 pynccl 通信器：rank 0 生成 NCCL unique ID 并广播给整个 TP 组。

    所有 rank 用同一个 ID 加入同一 communicator，随后即可执行 all_reduce/all_gather。
    """
    import torch  # 延迟导入 torch，用到其分布式广播接口

    # 用环境变量上限截断用户传入的最大缓冲区字节数，防止通信缓冲区过大占用显存
    max_size_bytes = min(max_size_bytes, ENV.PYNCCL_MAX_BUFFER_SIZE.value)

    module = _load_nccl_module()  # 获取（并缓存）pynccl 动态库模块
    cls = _get_pynccl_wrapper_cls()  # 获取（并缓存）FFI 包装类

    if tp_rank == 0:  # rank 0 负责生成唯一 ID 并作为广播源
        id_list = [module.create_nccl_uid()]  # C++ 侧生成 NCCL unique ID（加入同一通信器的"门票"）
        torch.distributed.broadcast_object_list(  # 通过 CPU 进程组把 ID 列表对象广播出去
            id_list,
            src=0,  # 源 rank 为全局/组内 0 号
            group=tp_cpu_group,  # 用 CPU 上的 gloo/mpi 进程组做引导通信（NCCL 尚未就绪）
        )
    else:  # 其余 rank 准备空列表接收广播
        id_list = [None]  # 占位，广播后会被 rank 0 的值覆盖
        torch.distributed.broadcast_object_list(  # 接收来自 rank 0 的 ID 列表
            id_list,
            src=0,  # 广播源为 0 号
            group=tp_cpu_group,  # 同样走 CPU 引导进程组
        )

    nccl_id = id_list[0]  # 取出广播得到的 NCCL unique ID
    assert not nccl_id is None, f"Failed to get NCCL unique ID on {tp_rank = }"  # 非 0 rank 必须收到 ID，否则初始化失败

    # bypass type checking for the FFI object
    return cls(tp_rank, tp_size, max_size_bytes, nccl_id)  # type: ignore  # 用 4 个参数调用 C++ 构造函数，创建通信器对象
