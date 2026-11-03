from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解中使用前向引用


def call_if_main(name: str = "__main__", discard: bool | None = None):
    """Decorator to ensure a function will call when the script is run as main.
    装饰器：仅当以主模块运行（name == "__main__"）时，在装饰的瞬间立即执行函数；
    否则按 discard 选项决定"丢弃函数"还是"原样保留函数"。
    """
    if name != "__main__":  # 非主模块运行（本脚本被 import 或作为子进程入口调用）
        discard = False if discard is None else discard  # 未指定时默认不丢弃：保留函数本身可调用
        if discard:
            return lambda _: None  # 丢弃模式：返回"吃掉任意函数、恒返回 None"的 lambda，函数体永不执行
        else:
            return lambda f: f  # 保留模式：恒等装饰器，函数原样通过
    else:
        discard = True if discard is None else discard  # 主模块运行时默认"丢弃"：装饰后立即执行且装饰结果为 None
        if discard:
            return lambda f: (f() or True) and None  # 立即执行 f，无论返回什么都折叠成 None（f 的返回值被丢弃）
        else:
            return lambda f: (f() and None) or f  # 立即执行 f，同时把函数本身作为装饰结果保留（f 返回 None 时不影响）


def div_even(a: int, b: int, allow_replicate: bool = False) -> int:
    """Divides two integers. If allow_replicate=True, allows b > a when b % a == 0, returning 1.
    强制整除的除法：要求 a 能被 b 整除，返回 a//b（常用于 TP 切分、head 数切分等必须整除的维度）。
    allow_replicate=True（GQA 中 KV 头数少于 Q 头数的复制场景）：若 b > a 且 b 能被 a 整除，返回 1。
    """
    if allow_replicate and b > a:  # KV 头数不足、需要复制补充时的特殊路径
        assert b % a == 0, f"{b = } must be divisible by {a = } for KV head replication"  # b 必须是 a 的整数倍，否则无法均匀复制
        return 1  # 复制语义下每个 KV 头视作 1 份
    assert a % b == 0, f"{a = } must be divisible by {b = }"  # 常规路径：必须整除，否则断言失败（快速暴露配置错误）
    return a // b  # 返回整除结果


def div_ceil(a: int, b: int) -> int:
    """Divides two integers, rounding up  向上取整除法"""
    return (a + b - 1) // b  # 经典整数技巧：加 b-1 再整除实现向上取整，避免浮点误差


def align_ceil(a: int, b: int) -> int:
    """Aligns a to the next multiple of b  把 a 向上对齐到 b 的整数倍"""
    return div_ceil(a, b) * b  # 先向上取整除，再乘回 b


def align_down(a: int, b: int) -> int:
    """Aligns a to the previous multiple of b  把 a 向下对齐到 b 的整数倍"""
    return (a // b) * b  # 整除后乘回 b，抹掉不足 b 的余数部分


class Unset:
    """哨兵类：用于区分"参数未传"与"显式传了 None"两种语义（None 可能是合法值）"""

    pass


UNSET = Unset()  # 模块级唯一哨兵实例（单例），配合 `is UNSET` 判断使用
