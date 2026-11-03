from .llm import LLM  # 离线批量推理入口类：面向单进程、类 HuggingFace generate 的简易接口

__all__ = ["LLM"]  # 声明模块公开 API
