from .server import launch_server  # 导入服务器启动入口

if __name__ == "__main__":  # 仅直接运行本文件时执行（`python -m minisgl.shell`）
    launch_server(run_shell=True)  # 启动服务器并进入交互式 shell 模式（单进程内同时充当服务端与交互客户端）
