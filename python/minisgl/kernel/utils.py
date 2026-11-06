# 延迟类型注解：让注解在运行时不求值，避免为注解而导入重模块
from __future__ import annotations

# pathlib 用于拼接内核源码目录的路径
import pathlib
# 类型注解工具；TYPE_CHECKING 块内的导入只在静态类型检查时生效
from typing import TYPE_CHECKING, List, NamedTuple, Tuple, TypeAlias, Union

# tvm_ffi.Module 只作为返回类型注解使用，运行时无需导入
if TYPE_CHECKING:
    from tvm_ffi import Module

# 内核 C++/CUDA 源码根目录：<包目录>/kernel/csrc
KERNEL_PATH = pathlib.Path(__file__).parent / "csrc"
# 默认头文件搜索路径（csrc/include），编译扩展时传给编译器
DEFAULT_INCLUDE = [str(KERNEL_PATH / "include")]
# 默认 C++ 编译选项：C++20 标准 + O3 最高优化
DEFAULT_CFLAGS = ["-std=c++20", "-O3"]
# 默认 CUDA 编译选项：--expt-relaxed-constexpr 允许 device 代码调用 host constexpr 函数
DEFAULT_CUDA_CFLAGS = ["-std=c++20", "-O3", "--expt-relaxed-constexpr"]
# 默认链接选项为空
DEFAULT_LDFLAGS = []
# C++ 模板实参支持的 Python 标量类型别名：int/float/bool
CPP_TEMPLATE_TYPE: TypeAlias = Union[int, float, bool]


# C++ 实参列表：继承 list[str]，重写 __str__ 方便直接拼成逗号分隔的函数实参串
class CppArgList(list[str]):
    def __str__(self) -> str:
        return ", ".join(self)


# 内核启动配置（NamedTuple 不可变元组）：线程数、最大占用块数、是否启用 PDL
class KernelConfig(NamedTuple):
    num_threads: int
    max_occupancy: int
    use_pdl: bool

    @property
    def template_args(self) -> str:
        # 把配置序列化成 C++ 模板实参串，例如 "256,2,true"
        pdl = "true" if self.use_pdl else "false"
        return f"{self.num_threads},{self.max_occupancy},{pdl}"


def _make_name(*args: str) -> str:
    # 生成全局唯一的扩展模块名（如 minisgl__jit__xxx），既作缓存键也防符号冲突
    return "minisgl__" + "_".join(str(arg) for arg in args)


def _make_wrapper(tup: Tuple[str, str]) -> str:
    # 生成 TVM FFI 导出宏代码：把 C++ 模板实例注册为 Python 可直接调用的函数
    # export_name 为对外函数名，kernel_name 为 C++ 模板实例化表达式
    export_name, kernel_name = tup
    return f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({export_name}, ({kernel_name}));"


def make_cpp_args(*args: CPP_TEMPLATE_TYPE) -> CppArgList:
    # 把 Python 标量转成 C++ 模板实参字符串列表，用于按模板参数编译特化内核
    def _convert(arg: CPP_TEMPLATE_TYPE) -> str:
        # bool 必须先于 int 判断（bool 是 int 子类），转成 C++ 的 true/false
        if isinstance(arg, bool):
            return "true" if arg else "false"
        # int/float 直接转字符串
        if isinstance(arg, (int, float)):
            return str(arg)
        # 其余类型不支持模板传参，直接报错
        raise TypeError(f"Unsupported argument type for cpp template: {type(arg)}")

    return CppArgList(_convert(arg) for arg in args)


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
    # AOT 方式加载内核：源文件预先写好在 csrc/src 下，这里只负责编译装载
    from tvm_ffi.cpp import load

    # 各可选参数缺省时置为空列表，统一后续拼接逻辑
    cpp_files = cpp_files or []
    cuda_files = cuda_files or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    extra_include_paths = extra_include_paths or []

    # 把相对文件名解析成 csrc/src 下的绝对路径
    cpp_files = [str((KERNEL_PATH / "src" / f).resolve()) for f in cpp_files]
    cuda_files = [str((KERNEL_PATH / "src" / f).resolve()) for f in cuda_files]

    # 交给 tvm-ffi 编译并加载共享库，返回可直接调用的 Module
    return load(
        _make_name(*args),
        cpp_files=cpp_files,
        cuda_files=cuda_files,
        extra_cflags=DEFAULT_CFLAGS + extra_cflags,
        extra_cuda_cflags=DEFAULT_CUDA_CFLAGS + extra_cuda_cflags,
        extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
        extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
        build_directory=build_directory,
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
    # JIT 方式加载内核：运行时拼接源码字符串（include + 导出宏）再即时编译
    from tvm_ffi.cpp import load_inline

    # 各可选参数缺省时置为空列表
    cpp_files = cpp_files or []
    cuda_files = cuda_files or []
    cpp_wrappers = cpp_wrappers or []
    cuda_wrappers = cuda_wrappers or []
    extra_cflags = extra_cflags or []
    extra_cuda_cflags = extra_cuda_cflags or []
    extra_ldflags = extra_ldflags or []
    extra_include_paths = extra_include_paths or []

    # include cpp files
    # 解析 csrc/jit 下的 C++ 源文件绝对路径
    cpp_paths = [(KERNEL_PATH / "jit" / f).resolve() for f in cpp_files]
    # 用 #include 方式把源文件内联进翻译单元
    cpp_sources = [f'#include "{path}"' for path in cpp_paths]
    # 追加导出宏，把模板实例注册为可调用符号
    cpp_sources += [_make_wrapper(tup) for tup in cpp_wrappers]

    # include cuda files
    # 同样解析并内联 CUDA 源文件
    cuda_paths = [(KERNEL_PATH / "jit" / f).resolve() for f in cuda_files]
    cuda_sources = [f'#include "{path}"' for path in cuda_paths]
    cuda_sources += [_make_wrapper(tup) for tup in cuda_wrappers]

    # 交给 tvm-ffi 即时编译并加载，返回可调用的 Module
    return load_inline(
        _make_name(*args),
        cpp_sources=cpp_sources,
        cuda_sources=cuda_sources,
        extra_cflags=DEFAULT_CFLAGS + extra_cflags,
        extra_cuda_cflags=DEFAULT_CUDA_CFLAGS + extra_cuda_cflags,
        extra_ldflags=DEFAULT_LDFLAGS + extra_ldflags,
        extra_include_paths=DEFAULT_INCLUDE + extra_include_paths,
        build_directory=build_directory,
    )
