from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解中使用前向引用

import os  # 读取环境变量
from functools import partial  # 偏函数：批量生成 EnvVar 的特化版本（固定 fn/default_value 参数）
from typing import Callable, Generic, TypeVar  # 泛型工具：环境变量值的类型标注


class BaseEnv:
    """环境变量包装器的基类：约定 _init 接口，供 EnvClassSingleton 统一扫描初始化"""

    def _init(self, name: str) -> None:
        """用名为 name 的环境变量初始化自身；子类必须实现"""
        raise NotImplementedError


T = TypeVar("T")  # 环境变量值的类型占位符


class EnvVar(BaseEnv, Generic[T]):
    """单个环境变量的强类型包装器：默认值 + 自定义解析函数。
    用法：EnvVar[int](0, int) 表示读取 int 型环境变量，未设置时默认为 0。
    """

    def __init__(self, default_value: T, fn: Callable[[str], T]):
        self.value = default_value  # 先持有默认值（环境变量未设置/解析失败时生效）
        self.fn = fn  # 字符串 -> 目标类型 的解析函数
        super().__init__()

    def _init(self, name: str) -> None:
        """读取环境变量 name 并解析覆盖默认值；未设置或解析失败都保持默认值"""
        env_value = os.getenv(name)  # 读取环境变量（未设置返回 None）
        if env_value is not None:  # 已设置才尝试覆盖默认值
            try:
                self.value = self.fn(env_value)  # 用解析函数做类型转换（如 int("8") -> 8）
            except Exception:
                pass  # 解析失败（格式非法）时静默忽略、回退默认值——避免环境变量写错导致进程崩溃

    def __bool__(self):
        """让实例可直接用于 if 判断，语义上等于判断 value 的真值"""
        return self.value  # 注意：value 非 bool 型（如 int/None）时，此处可能触发运行时 TypeError

    def __str__(self):
        """打印实例时直接显示当前值（调试友好）"""
        return str(self.value)


_TO_BOOL = lambda x: x.lower() in ("1", "true", "yes")  # 布尔解析：1/true/yes（不区分大小写）为真，其余一律为假


def _PARSE_MEM_BYTES(mem: str) -> int:
    """内存容量字符串 -> 字节数：支持 "1G" / "512M" / "64K" / "100B" / 裸数字等写法"""
    mem = mem.strip().upper()  # 去空白并统一大写，简化单位解析
    if not mem[-1].isalpha():  # 末尾是数字：视为纯字节数
        return int(mem)
    if mem.endswith("B"):  # 兼容 "KB/MB/GB" 中的 B 后缀
        mem = mem[:-1]  # 去掉 B，只剩单位字母 K/M/G
    UNIT_MAP = {"K": 1024, "M": 1024**2, "G": 1024**3}  # 二进制单位换算表
    return int(float(mem[:-1]) * UNIT_MAP[mem[-1]])  # 数值部分 * 单位；float 支持小数写法（如 "1.5G"）


MINISGL_ENV_PREFIX = "MINISGL_"  # 本项目环境变量的统一前缀（配置项 SHELL_TOP_K 对应 MINISGL_SHELL_TOP_K）
EnvInt = partial(EnvVar[int], fn=int)  # 特化：int 型环境变量
EnvFloat = partial(EnvVar[float], fn=float)  # 特化：float 型环境变量
EnvBool = partial(EnvVar[bool], fn=_TO_BOOL)  # 特化：bool 型环境变量
EnvOption = partial(EnvVar[bool | None], fn=_TO_BOOL, default_value=None)  # 特化：三态开关（None=未设置 / True / False）
EnvMem = partial(EnvVar[int], fn=_PARSE_MEM_BYTES)  # 特化：内存容量型环境变量（解析为字节数 int）


class EnvClassSingleton:
    """环境变量配置单例：类属性即全部配置项；实例化时统一扫描并按 MINISGL_ 前缀读环境变量"""

    _instance: EnvClassSingleton | None = None  # 单例引用

    # shell
    # ---- 交互式客户端（shell）相关配置 ----
    SHELL_MAX_TOKENS = EnvInt(2048)  # 单次生成 token 数上限（默认 2048）
    SHELL_TOP_K = EnvInt(-1)  # top-k 采样（-1 表示禁用，不做 k 截断）
    SHELL_TOP_P = EnvFloat(1.0)  # top-p 采样（1.0 表示禁用，保留全部概率质量）
    SHELL_TEMPERATURE = EnvFloat(0.6)  # 采样温度（越低越确定，越高越随机）

    # backend runtime
    # ---- 后端运行时相关配置 ----
    FLASHINFER_USE_TENSOR_CORES = EnvOption()  # FlashInfer 是否用 Tensor Core（三态：None=交给后端自行决定）
    DISABLE_OVERLAP_SCHEDULING = EnvBool(False)  # 是否关闭调度与 GPU 执行的重叠（overlap）优化（调试排障用）
    PYNCCL_MAX_BUFFER_SIZE = EnvMem(1024**3)  # PyNCCL 通信缓冲区上限（默认 1GiB）

    def __new__(cls):
        # single instance  单例模式：全局只允许存在一个配置实例
        if cls._instance is None:  # 首次创建
            cls._instance = super().__new__(cls)  # 创建并记住实例
        return cls._instance  # 之后所有调用都返回同一实例

    def __init__(self) -> None:
        """遍历所有公开类属性，逐个用对应的 MINISGL_XXX 环境变量初始化"""
        for attr_name in dir(self):  # 枚举实例及类上的全部属性名（含继承）
            if attr_name.startswith("_"):  # 跳过内部属性（如 _instance）
                continue
            attr_value = getattr(self, attr_name)  # 取属性值
            assert isinstance(attr_value, BaseEnv)  # 防呆：公开属性必须是环境变量包装器（误加普通属性会立即暴露）
            attr_value._init(f"{MINISGL_ENV_PREFIX}{attr_name}")  # 用 "MINISGL_属性名" 对应的环境变量初始化该配置项


ENV = EnvClassSingleton()  # 模块级唯一配置实例：其他模块通过 `from minisgl.env import ENV` 读取配置
