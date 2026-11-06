# 导入服务端启动函数
from .server import launch_server

# 该模块只应通过 `python -m minisgl` 执行，作为库导入属于误用
assert __name__ == "__main__"

# 启动 OpenAI 兼容的推理服务
launch_server()
