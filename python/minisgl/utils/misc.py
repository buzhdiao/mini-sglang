# 延迟求值类型注解：注解在运行时不求值
from __future__ import annotations


# 装饰器：让函数只在脚本作为主入口时被自动调用（默认并丢弃返回值）
def call_if_main(name: str = "__main__", discard: bool | None = None):
    """Decorator to ensure a function will call when the script is run as main."""
    if name != "__main__":
        # 非主入口：默认不丢弃返回值（函数原样保留）
        discard = False if discard is None else discard
        if discard:
            # 丢弃模式：返回吞掉函数的占位（调用返回 None）
            return lambda _: None
        else:
            # 保留模式：函数原样通过
            return lambda f: f
    else:
        # 主入口：默认丢弃返回值（模块导入时函数自动执行且不留返回值）
        discard = True if discard is None else discard
        if discard:
            # 返回包装器：调用 f() 并把结果抹成 None（f() or True 保证表达式成立）
            return lambda f: (f() or True) and None
        else:
            # 返回包装器：f() 返回真值则执行即返回 None，否则返回 f 本身
            return lambda f: (f() and None) or f


def div_even(a: int, b: int, allow_replicate: bool = False) -> int:
    """Divides two integers. If allow_replicate=True, allows b > a when b % a == 0, returning 1."""
    if allow_replicate and b > a:
        # KV 头数复制场景：b(查询头数) > a(KV 头数) 且整除时，
        # 返回 1（每组 KV 头只对应 1 份复制前的划分）
        assert b % a == 0, f"{b = } must be divisible by {a = } for KV head replication"
        return 1
    # 必须整除（返回精确商，不取整）
    assert a % b == 0, f"{a = } must be divisible by {b = }"
    return a // b


def div_ceil(a: int, b: int) -> int:
    """Divides two integers, rounding up"""
    # 向上取整除法：先加 b-1 再整除（如把 token 数换算成页数）
    return (a + b - 1) // b


def align_ceil(a: int, b: int) -> int:
    """Aligns a to the next multiple of b"""
    # 向上对齐到 b 的倍数
    return div_ceil(a, b) * b


def align_down(a: int, b: int) -> int:
    """Aligns a to the previous multiple of b"""
    # 向下对齐到 b 的倍数
    return (a // b) * b


# 哨兵类型：区分"没传这个参数"与"传了 None"两种情况
class Unset:
    pass


# 全局唯一哨兵实例（用 `is UNSET` 判断）
UNSET = Unset()
