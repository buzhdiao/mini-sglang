from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许注解中使用前向引用类型

import pathlib  # 路径处理：定位 kernel 的 csrc 源码目录
from typing import TYPE_CHECKING, List, NamedTuple, Tuple, TypeAlias, Union  # 类型工具：NamedTuple 定义配置、TypeAlias 定义类型别名

if TYPE_CHECKING:  # 仅静态检查时导入，运行时无开销
    from tvm_ffi import Module  # tvm-ffi 动态库模块类型（仅作返回值注解）

KERNEL_PATH = pathlib.Path(__file__).parent / "csrc"  # kernel C++/CUDA 源码根目录（minisgl/kernel/csrc）
DEFAULT_INCLUDE = [str(KERNEL_PATH / "include")]  # 默认头文件搜索路径（csrc/include）
DEFAULT_CFLAGS = ["-std=c++20", "-O3"]  # C++ 编译默认参数：C++20 标准 + O3 优化
DEFAULT_CUDA_CFLAGS = ["-std=c++20", "-O3", "--expt-relaxed-constexpr"]  # CUDA 编译默认参数：额外允许 host 代码调用 constexpr 设备函数（常量表达式跨端使用）
DEFAULT_LDFLAGS = []  # 默认链接参数（暂无）
CPP_TEMPLATE_TYPE: TypeAlias = Union[int, float, bool]  # 允许作为 C++ 模板实参的 Python 类型集合


class CppArgList(list[str]):  # 继承 list[str]：存放 C++ 模板实参字符串的列表
    def __str__(self) -> str:
        return ", ".join(self)  # 序列化为逗号分隔串，可直接填入 C++ 模板尖括号（如 "2048,4,128,1,false"）


class KernelConfig(NamedTuple):  # kernel 启动配置（NamedTuple：不可变、可解包、可当模板参数）
    num_threads: int  # 每个 block 的线程数（传入 C++ 模板）
    max_occupancy: int  # 允许的最大占用率倍数（限制 SM 上并发 block 数）
    use_pdl: bool  # 是否启用 PDL（Programmatic Dependent Launch，重叠相邻 kernel 启动延迟）

    @property  # 把配置序列化为 C++ 模板实参字符串的计算属性
    def template_args(self) -> str:
        pdl = "true" if self.use_pdl else "false"  # bool 转 C++ 字面量
        return f"{self.num_threads},{self.max_occupancy},{pdl}"  # 形如 "128,1,false"


def _make_name(*args: str) -> str:
    """把模块名与模板参数拼成全局唯一的库名，避免不同模板实例的编译缓存冲突。"""
    return "minisgl__" + "_".join(str(arg) for arg in args)  # 形如 "minisgl__index_2048_4_128_1_false"


def _make_wrapper(tup: Tuple[str, str]) -> str:
    """生成一行 C++ 导出宏：把指定 kernel 的静态 run 方法以指定名字暴露给 Python。"""
    export_name, kernel_name = tup  # tup = (Python 侧函数名, C++ 模板实例方法)
    return f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({export_name}, ({kernel_name}));"  # tvm-ffi 的 DLL 导出宏，生成 FFI 入口符号


def make_cpp_args(*args: CPP_TEMPLATE_TYPE) -> CppArgList:
    """把 Python 标量转成 C++ 模板实参字符串列表（bool -> true/false，数值 -> 十进制串）。"""

    def _convert(arg: CPP_TEMPLATE_TYPE) -> str:
        if isinstance(arg, bool):  # 注意必须先判 bool（bool 是 int 的子类，顺序反了会输出 1/0）
            return "true" if arg else "false"  # C++ 布尔字面量
        if isinstance(arg, (int, float)):  # 数值直接转字符串
            return str(arg)
        raise TypeError(f"Unsupported argument type for cpp template: {type(arg)}")  # 其他类型不支持作模板参数

    return CppArgList(_convert(arg) for arg in args)  # 惰性转换并包装成 CppArgList


def load_aot(
    *args: str,
    cpp_files: List[str] | None = None,
    cuda_files: List[str] | None = None,
    extra_cflags: List[str] | None = None,
    extra_cuda_cflags: List[str] | None = None,
    extra_ldflags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    build_directory: str | None = None,
) -> Module:
    """加载 AOT（ahead-of-time）预编译的 kernel 动态库（首次调用时若未编译则触发构建）。"""
    from tvm_ffi.cpp import load  # tvm-ffi 的库加载入口（延迟导入，避免拖慢模块加载）

    cpp_files = cpp_files or []  # None 统一归一为空列表
    cuda_files = cuda_files or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    extra_include_paths = extra_include_paths or []

    # 把源文件名解析为 csrc/src 下的绝对路径字符串
    cpp_files = [str((KERNEL_PATH / "src" / f).resolve()) for f in cpp_files]
    cuda_files = [str((KERNEL_PATH / "src" / f).resolve()) for f in cuda_files]

    return load(  # 调 tvm-ffi 加载/构建动态库，返回可调用 Module
        _make_name(*args),  # 唯一库名（含模板参数），命中编译缓存时直接复用
        cpp_files=cpp_files,  # C++ 源文件绝对路径列表
        cuda_files=cuda_files,  # CUDA 源文件绝对路径列表
        extra_cflags=DEFAULT_CFLAGS + extra_cflags,  # 默认 + 额外的 C++ 编译参数
        extra_cuda_cflags=DEFAULT_CUDA_CFLAGS + extra_cuda_cflags,  # 默认 + 额外的 CUDA 编译参数
        extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,  # 默认 + 额外的链接参数（如 -lnccl）
        extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,  # 默认 + 额外的头文件路径
        build_directory=build_directory,  # 构建产物目录（None 用 tvm-ffi 默认缓存目录）
    )


def load_jit(
    *args: str,
    cpp_files: List[str] | None = None,
    cuda_files: List[str] | None = None,
    cpp_wrappers: List[Tuple[str, str]] | None = None,
    cuda_wrappers: List[Tuple[str, str]] | None = None,
    extra_cflags: List[str] | None = None,
    extra_cuda_cflags: List[str] | None = None,
    extra_ldflags: List[str] | None = None,
    extra_include_paths: List[str] | None = None,
    build_directory: str | None = None,
) -> Module:
    """JIT inline 编译：用生成的 #include + 导出宏源码现场编译模板实例化的 kernel。"""
    from tvm_ffi.cpp import load_inline  # tvm-ffi 的 inline 编译入口（延迟导入）

    cpp_files = cpp_files or []  # None 统一归一为空列表
    cuda_files = cuda_files or []
    cpp_wrappers = cpp_wrappers or []
    cuda_wrappers = cuda_wrappers or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    extra_include_paths = extra_include_paths or []

    # include cpp files  # 构造 C++ 编译单元：仅包含头文件式的 .cu/.cpp 源 + 导出宏
    cpp_paths = [(KERNEL_PATH / "jit" / f).resolve() for f in cpp_files]  # jit 源码位于 csrc/jit 目录
    cpp_sources = [f'#include "{path}"' for path in cpp_paths]  # 生成 #include 语句把源码并入编译单元
    cpp_sources += [_make_wrapper(tup) for tup in cpp_wrappers]  # 追加 TVM_FFI 导出宏，生成 Python 可调用的入口

    # include cuda files  # 同样方式构造 CUDA 编译单元
    cuda_paths = [(KERNEL_PATH / "jit" / f).resolve() for f in cuda_files]  # CUDA jit 源码路径
    cuda_sources = [f'#include "{path}"' for path in cuda_paths]  # #include 并入源码
    cuda_sources += [_make_wrapper(tup) for tup in cuda_wrappers]  # 追加导出宏（如 launch -> IndexKernel<...>::run）

    return load_inline(  # 现场编译并加载，返回 FFI Module
        _make_name(*args),  # 唯一库名：相同模板参数命中缓存不重编
        cpp_sources=cpp_sources,  # 生成的 C++ 源码行列表
        cuda_sources=cuda_sources,  # 生成的 CUDA 源码行列表
        extra_cflags=DEFAULT_CFLAGS + extra_cflags,  # 编译参数同 load_aot
        extra_cuda_cflags=DEFAULT_CUDA_CFLAGS + extra_cuda_cflags,
        extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
        extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
        build_directory=build_directory,  # 构建产物目录
    )
