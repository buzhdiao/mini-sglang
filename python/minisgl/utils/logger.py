# 延迟求值类型注解：注解在运行时不求值
from __future__ import annotations

# partial：把 _call_rank0 绑定成各类 rank0 便捷方法
from functools import partial
# 类型注解工具
from typing import TYPE_CHECKING

# 全局日志级别缓存：只解析一次环境变量
_LOG_LEVEL = None


def init_logger(
    name: str,
    suffix: str = "",
    *,
    strip_file: bool = True,
    level: str | None = None,
    use_pid: bool | None = None,
    use_tp_rank: bool | None = None,
):
    """Initialize the logger for the module with colors and pretty formatting."""
    # 函数内导入：加快模块加载（logging 较重且并非每个模块都要立刻用）
    import logging
    import os
    import sys

    # 引用全局级别缓存
    global _LOG_LEVEL
    if _LOG_LEVEL is None:
        # 首次初始化：建立级别名 -> 级别常量的映射
        LEVEL_MAP = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }

        # 优先用显式传入的级别，否则读 LOG_LEVEL 环境变量
        level = level or os.getenv("LOG_LEVEL", "").upper()
        # 未识别/未设置时默认 INFO
        _LOG_LEVEL = LEVEL_MAP.get(level, logging.INFO)

    if strip_file:
        # 后缀若是文件路径，只保留文件名（日志更紧凑）
        suffix = os.path.basename(suffix)

    if suffix:
        # 给后缀加分隔符
        suffix = f"|{suffix}"

    if use_pid is None:
        # 默认由 LOG_PID 环境变量决定是否在日志中显示进程号
        use_pid = os.getenv("LOG_PID", "0").lower() in ("1", "true", "yes")

    if use_pid:
        # 把进程号拼进后缀：多进程（调度器/tokenizer）日志区分来源
        pid = os.getpid()
        suffix = f"|pid={pid}{suffix}"

    # 缓存本模块的 TP 拓扑信息（格式化时避免每次查询）
    tp_info = None

    # Color formatter class
    # 带颜色的格式化器
    class ColorFormatter(logging.Formatter):
        """Formatter with colors and pretty output"""

        # ANSI color codes
        # 各日志级别对应的 ANSI 前景色
        COLORS = {
            "DEBUG": "\033[36m",  # Cyan
            "INFO": "\033[32m",  # Green
            "WARNING": "\033[33m",  # Yellow
            "ERROR": "\033[31m",  # Red
            "CRITICAL": "\033[35m",  # Magenta
        }
        # ANSI 颜色重置码
        RESET = "\033[0m"
        # ANSI 加粗码
        BOLD = "\033[1m"

        def format(self, record):
            from minisgl.distributed import try_get_tp_info

            # Format timestamp like SGLang: [YYYY-MM-DD|HH:MM:SS|pid=1234]
            # 先按 SGLang 风格生成时间戳模板（{suffix} 留待填充）
            timestamp = self.formatTime(record, "[%Y-%m-%d|%H:%M:%S{suffix}]")
            nonlocal tp_info
            tp_info = tp_info or try_get_tp_info()
            if tp_info is not None and use_tp_rank is not False:
                # TP 进程：后缀里带上 rank 号，便于区分并行日志
                real_suffix = f"{suffix}|core|rank={tp_info.rank}"
            else:
                # 非 TP 或不显示 rank：用普通后缀
                real_suffix = suffix
            # 填充后缀生成最终时间戳
            timestamp = timestamp.format(suffix=real_suffix)

            # Get color for log level
            # 取当前级别对应的颜色（未知级别不着色）
            level_color = self.COLORS.get(record.levelname, "")

            # Format the message
            # 级别名着色并左对齐到 8 字符
            colored_level = f"{level_color}{record.levelname:<8}{self.RESET}"
            # 取原始消息文本
            message = record.getMessage()

            # Pretty format: [timestamp] LEVEL message
            # 最终格式：加粗时间戳 + 彩色级别 + 消息
            return f"{self.BOLD}{timestamp}{self.RESET} {colored_level} {message}"

    # 取（或创建）指定名字的 logger
    logger = logging.getLogger(name)
    # 设置过滤级别
    logger.setLevel(_LOG_LEVEL)

    # Clear existing handlers to avoid duplicates
    # 清空既有 handler：重复 init 时避免一条日志打多遍
    logger.handlers.clear()

    # 输出到 stdout（而非 stderr，便于与训练日志合并查看）
    handler = logging.StreamHandler(sys.stdout)
    # 创建彩色格式化器
    formatter = ColorFormatter()
    # 挂上格式化器
    handler.setFormatter(formatter)
    # 注册 handler
    logger.addHandler(handler)

    # Prevent propagation to root logger
    # 关闭向根 logger 传播：防止重复输出
    logger.propagate = False

    def _call_rank0(msg, *args, _which, **kwargs):
        from minisgl.distributed import get_tp_info

        nonlocal tp_info
        tp_info = tp_info or get_tp_info()
        # TP 拓扑必须已初始化才能判断 rank
        assert tp_info is not None, "TP info not set yet"
        if tp_info.is_primary():
            # 只有 rank0 真正打日志：避免 TP 并行时同一消息重复 N 遍
            getattr(logger, _which)(msg, *args, **kwargs)

    if TYPE_CHECKING:

        # 类型检查期：给 IDE 提供带 rank0 方法的完整类型
        class WrapperLogger(logging.Logger):
            """Custom logger to handle the color formatter."""

            def info_rank0(self, msg, *args, **kwargs): ...
            def warning_rank0(self, msg, *args, **kwargs): ...
            def debug_rank0(self, msg, *args, **kwargs): ...
            def critical_rank0(self, msg, *args, **kwargs): ...

        return WrapperLogger(name)
    else:
        # 运行期：用 partial 把 _call_rank0 动态挂成四个 rank0 方法
        logger.info_rank0 = partial(_call_rank0, _which="info")
        logger.debug_rank0 = partial(_call_rank0, _which="debug")
        logger.critical_rank0 = partial(_call_rank0, _which="critical")
        logger.warning_rank0 = partial(_call_rank0, _which="warning")
        # 返回增强后的 logger
        return logger
