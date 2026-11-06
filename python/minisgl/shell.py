# 复用 server 包的启动入口
from .server import launch_server

if __name__ == "__main__":
    # 直接运行本文件即以 shell（交互终端）模式启动
    launch_server(run_shell=True)
