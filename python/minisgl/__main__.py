from .server import launch_server  # 导入服务器启动入口（支持 `python -m minisgl` 方式启动）

assert __name__ == "__main__"  # 防呆断言：本文件只能作为主模块运行；若被 import 则立即失败，避免误触发启动

launch_server()  # 启动推理服务器（阻塞运行，直到手动停止）
