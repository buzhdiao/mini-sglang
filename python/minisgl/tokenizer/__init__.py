# 对外只暴露 tokenizer 子进程入口函数
from .server import tokenize_worker

# 导出清单
__all__ = ["tokenize_worker"]
