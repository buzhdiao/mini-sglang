from __future__ import annotations  # 启用 PEP 563 延迟注解求值，允许类型注解中使用前向引用

from functools import partial  # 偏函数：预先绑定 _which 参数，动态生成 *_rank0 方法
from typing import TYPE_CHECKING  # 仅类型检查时导入的开关，避免运行时循环依赖

_LOG_LEVEL = None  # 模块级日志级别缓存：首次 init_logger 时解析一次，之后所有 logger 复用


def init_logger(
    name: str,  # logger 名称（通常传模块名或进程标识）
    suffix: str = "",  # 显示在时间戳后的后缀（如脚本文件名、pid、rank）
    *,
    strip_file: bool = True,  # 是否只保留 suffix 的文件名部分（去掉目录路径）
    level: str | None = None,  # 日志级别字符串；为空则回退读环境变量 LOG_LEVEL
    use_pid: bool | None = None,  # 是否在日志中显示进程 PID；为空则读环境变量 LOG_PID
    use_tp_rank: bool | None = None,  # 是否在日志中显示 TP rank；False 表示强制关闭
):
    """Initialize the logger for the module with colors and pretty formatting.
    为模块初始化带 ANSI 颜色、带时间戳后缀的 logger，并附加 *_rank0 系列"仅 rank0 打日志"方法。
    """
    import logging  # 延迟导入标准库 logging，加快模块导入速度
    import os  # 读环境变量、取 PID、处理路径
    import sys  # 日志输出流（stdout）

    global _LOG_LEVEL  # 声明修改模块级缓存变量
    if _LOG_LEVEL is None:  # 只在第一次调用时解析级别，避免重复解析
        LEVEL_MAP = {  # 字符串级别名 -> logging 数值级别 的映射表
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }

        level = level or os.getenv("LOG_LEVEL", "").upper()  # 优先用显式参数，否则读环境变量并统一转大写
        _LOG_LEVEL = LEVEL_MAP.get(level, logging.INFO)  # 未知/空级别一律回退到 INFO

    if strip_file:  # 需要去路径时
        suffix = os.path.basename(suffix)  # 只取路径最后一段文件名（如 "a/b/entry.py" -> "entry.py"）

    if suffix:  # 后缀非空时
        suffix = f"|{suffix}"  # 加 "|" 分隔符，最终拼进时间戳后缀

    if use_pid is None:  # 未显式指定时由环境变量 LOG_PID 决定
        use_pid = os.getenv("LOG_PID", "0").lower() in ("1", "true", "yes")  # 支持 1/true/yes 三种写法

    if use_pid:  # 需要显示 PID 时（多进程场景用于区分日志来源）
        pid = os.getpid()  # 取当前进程 ID
        suffix = f"|pid={pid}{suffix}"  # PID 拼在最前，如 "|pid=1234|entry.py"

    tp_info = None  # 局部缓存 TP 信息；首次用到时才通过 distributed 模块懒加载

    # Color formatter class
    class ColorFormatter(logging.Formatter):  # 定义在函数内部：闭包捕获 suffix/use_tp_rank/tp_info
        """Formatter with colors and pretty output
        带颜色的格式化器：输出形如 `[时间戳] 级别 消息`，级别按颜色高亮"""

        # ANSI color codes
        COLORS = {  # 各日志级别对应的 ANSI 前景色码
            "DEBUG": "\033[36m",  # Cyan
            "INFO": "\033[32m",  # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",  # Red
            "CRITICAL": "\033[35m",  # Magenta
        }
        RESET = "\033[0m"  # ANSI 重置码：结束颜色作用范围
        BOLD = "\033[1m"  # ANSI 加粗码：用于时间戳

        def format(self, record):
            """把一条日志记录格式化成带颜色的一行文本"""
            from minisgl.distributed import try_get_tp_info  # 运行时才导入，避免 utils 与 distributed 循环依赖

            # Format timestamp like SGLang: [YYYY-MM-DD|HH:MM:SS|pid=1234]
            # 先用 {suffix} 占位符生成时间戳骨架，稍后再填充真正的后缀
            timestamp = self.formatTime(record, "[%Y-%m-%d|%H:%M:%S{suffix}]")
            nonlocal tp_info  # 允许修改外层函数缓存中的 tp_info
            tp_info = tp_info or try_get_tp_info()  # 懒加载：只有尚未拿到时才查询（TP 未初始化则返回 None）
            if tp_info is not None and use_tp_rank is not False:  # TP 已初始化且未被显式关闭时，追加显示 rank
                real_suffix = f"{suffix}|core|rank={tp_info.rank}"  # 追加 "|core|rank=N"
            else:
                real_suffix = suffix  # 沿用原始后缀（可能为空串）
            timestamp = timestamp.format(suffix=real_suffix)  # 把真实后缀填进时间戳模板

            # Get color for log level
            level_color = self.COLORS.get(record.levelname, "")  # 查级别对应颜色；未知级别用空串（无色）

            # Format the message
            colored_level = f"{level_color}{record.levelname:<8}{self.RESET}"  # 级别名左对齐补足 8 字符并上色
            message = record.getMessage()  # 把带 %s/%d 参数的日志消息展开成最终字符串

            # Pretty format: [timestamp] LEVEL message
            return f"{self.BOLD}{timestamp}{self.RESET} {colored_level} {message}"  # 加粗时间戳 + 彩色级别 + 消息正文

    logger = logging.getLogger(name)  # 按名字获取（或创建）全局单例 logger
    logger.setLevel(_LOG_LEVEL)  # 应用解析好的全局日志级别

    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()  # 清空旧 handler：重复调用 init_logger 时避免同一条消息输出多遍

    handler = logging.StreamHandler(sys.stdout)  # 新建输出到 stdout 的流处理器
    formatter = ColorFormatter()  # 实例化彩色格式化器
    handler.setFormatter(formatter)  # 绑定格式化器
    logger.addHandler(handler)  # 挂载到 logger

    # Prevent propagation to root logger
    logger.propagate = False  # 禁止向 root logger 冒泡，避免消息被根日志重复打印

    def _call_rank0(msg, *args, _which, **kwargs):
        """内部实现：仅 TP rank0 进程真正打印日志，其余 rank 静默丢弃"""
        from minisgl.distributed import get_tp_info  # 运行时才导入，避免循环依赖

        nonlocal tp_info  # 复用并回填外层缓存的 TP 信息
        tp_info = tp_info or get_tp_info()  # 懒加载；TP 未初始化时此处直接抛 RuntimeError
        assert tp_info is not None, "TP info not set yet"  # 防御性断言：必须已初始化 TP 信息
        if tp_info.is_primary():  # 只有 rank0 才执行打印
            getattr(logger, _which)(msg, *args, **kwargs)  # 按 _which 反射调用 logger.info/debug/...

    if TYPE_CHECKING:  # 仅静态类型检查走此分支：给 IDE/类型检查器提供精确签名
        class WrapperLogger(logging.Logger):  # 声明带 *_rank0 方法的 logger 类型
            """Custom logger to handle the color formatter."""

            def info_rank0(self, msg, *args, **kwargs): ...  # 仅 rank0 输出 info
            def warning_rank0(self, msg, *args, **kwargs): ...  # 仅 rank0 输出 warning
            def debug_rank0(self, msg, *args, **kwargs): ...  # 仅 rank0 输出 debug
            def critical_rank0(self, msg, *args, **kwargs): ...  # 仅 rank0 输出 critical

        return WrapperLogger(name)  # 类型检查器看到的返回类型（运行时不执行）
    else:
        # 运行时走此分支：用 partial 动态挂载 *_rank0 方法，无需真正定义子类
        logger.info_rank0 = partial(_call_rank0, _which="info")  # 调用 info_rank0(msg) 等价于 _call_rank0(msg, _which="info")
        logger.debug_rank0 = partial(_call_rank0, _which="debug")  # 同上，对应 debug 级别
        logger.critical_rank0 = partial(_call_rank0, _which="critical")  # 同上，对应 critical 级别
        logger.warning_rank0 = partial(_call_rank0, _which="warning")  # 同上，对应 warning 级别
        return logger  # 返回挂载好扩展方法的 logger 实例
