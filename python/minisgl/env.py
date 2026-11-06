# 允许类型注解使用字符串形式的前向引用
from __future__ import annotations

# os 读取进程环境变量
import os
# partial 预绑定参数生成特化类型
from functools import partial
# 类型注解工具
from typing import Callable, Generic, TypeVar


# 环境变量条目的基类：仅约定 _init 接口
class BaseEnv:
    # 子类必须实现：按环境变量名完成自身初始化
    def _init(self, name: str) -> None:
        # 基类不实现
        raise NotImplementedError


# 泛型类型参数
T = TypeVar("T")


# 带类型转换的环境变量包装器：存默认值，读到环境变量时用 fn 转换
class EnvVar(BaseEnv, Generic[T]):
    # 保存默认值与解析函数
    def __init__(self, default_value: T, fn: Callable[[str], T]):
        # 初始值 = 默认值
        self.value = default_value
        # 字符串 -> 目标类型的解析函数
        self.fn = fn
        # 完成基类初始化
        super().__init__()

    # 按环境变量名读取并解析；非法值静默保留默认
    def _init(self, name: str) -> None:
        # 读取环境变量
        env_value = os.getenv(name)
        # 设置了才覆盖默认值
        if env_value is not None:
            # 解析失败不崩溃
            try:
                # 用注册的函数转换
                self.value = self.fn(env_value)
            except Exception:
                # 忽略非法值
                pass

    # 让对象可直接当布尔用（if ENV.XXX:）
    def __bool__(self):
        # 值本身真值
        return self.value

    # 打印时展示当前值
    def __str__(self):
        # 转字符串
        return str(self.value)


# 布尔解析："1/true/yes"（忽略大小写）为真，其余为假
_TO_BOOL = lambda x: x.lower() in ("1", "true", "yes")


# 内存容量解析：支持 "1G"、"512M"、"1024KB" 或纯字节数
def _PARSE_MEM_BYTES(mem: str) -> int:
    # 去空白并统一大写
    mem = mem.strip().upper()
    # 结尾不是字母：纯数字按字节
    if not mem[-1].isalpha():
        # 直接转整数
        return int(mem)
    # 以 B 结尾则去掉单位里的 B（K/M/G）
    if mem.endswith("B"):
        # 去掉最后的 B
        mem = mem[:-1]
    # 单位 -> 字节倍率
    UNIT_MAP = {"K": 1024, "M": 1024**2, "G": 1024**3}
    # 数值部分乘以单位倍率
    return int(float(mem[:-1]) * UNIT_MAP[mem[-1]])


# 环境变量统一前缀
MINISGL_ENV_PREFIX = "MINISGL_"
# 各类型特化：int/float/bool/三态 bool/内存容量
EnvInt = partial(EnvVar[int], fn=int)
EnvFloat = partial(EnvVar[float], fn=float)
EnvBool = partial(EnvVar[bool], fn=_TO_BOOL)
EnvOption = partial(EnvVar[bool | None], fn=_TO_BOOL, default_value=None)
EnvMem = partial(EnvVar[int], fn=_PARSE_MEM_BYTES)


# 全局环境配置单例：类属性即全部受支持的环境变量
class EnvClassSingleton:
    # 唯一实例引用
    _instance: EnvClassSingleton | None = None

    # shell
    # minisgl shell 交互的最大生成 token 数
    SHELL_MAX_TOKENS = EnvInt(2048)
    # shell 的 top-k（-1 不启用）
    SHELL_TOP_K = EnvInt(-1)
    # shell 的 top-p
    SHELL_TOP_P = EnvFloat(1.0)
    # shell 的采样温度
    SHELL_TEMPERATURE = EnvFloat(0.6)

    # backend runtime
    # 强制指定 FlashInfer 是否用 tensor core 内核（None=自动按 GQA 比例决定）
    FLASHINFER_USE_TENSOR_CORES = EnvOption()
    # 关闭调度与 forward 的重叠流水
    DISABLE_OVERLAP_SCHEDULING = EnvBool(False)
    # PyNCCL 自定义 allreduce 的最大缓冲字节数（超过则回退 NCCL 原生）
    PYNCCL_MAX_BUFFER_SIZE = EnvMem(1024**3)

    # 单例：多次构造返回同一实例
    def __new__(cls):
        # single instance
        # 尚未创建过才真正分配
        if cls._instance is None:
            # 调用父类创建实例
            cls._instance = super().__new__(cls)
        # 总是返回同一实例
        return cls._instance

    # 构造时扫描所有类属性，逐一按 MINISGL_<名字> 读取环境变量
    def __init__(self) -> None:
        # 遍历实例可见属性
        for attr_name in dir(self):
            # 跳过私有/魔法属性
            if attr_name.startswith("_"):
                continue
            # 取属性值
            attr_value = getattr(self, attr_name)
            # 必须都是环境变量包装器
            assert isinstance(attr_value, BaseEnv)
            # 用 属性名 拼出完整环境变量名并初始化
            attr_value._init(f"{MINISGL_ENV_PREFIX}{attr_name}")


# 模块级单例：整个进程共享这份环境配置
ENV = EnvClassSingleton()
