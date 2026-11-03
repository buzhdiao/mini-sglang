from .launch import launch_server  # 导入服务器启动入口函数（解析参数、拉起各子进程并启动 API 服务）

__all__ = ["launch_server"]  # 声明包对外暴露的公共接口，供 `from minisgl.server import *` 使用
