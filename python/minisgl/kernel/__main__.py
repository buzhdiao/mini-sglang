assert __name__ == "__main__"  # 断言本模块只作为脚本直接运行（python -m minisgl.kernel），被 import 时会直接失败


def generate_clangd():
    """生成 .clangd 配置文件：为 IDE 的 clangd 语言服务器提供 CUDA 相关的编译参数，让 C++/CUDA 源码（csrc 目录）获得正确的代码补全与跳转"""

    import os  # 用于检查 .clangd 文件是否已存在
    import subprocess  # 用于调用 nvidia-smi 查询 GPU 算力

    from minisgl.kernel.utils import DEFAULT_INCLUDE  # kernel 头文件默认 include 路径（csrc/include）
    from minisgl.utils import init_logger  # 日志初始化工具
    from tvm_ffi.libinfo import find_dlpack_include_path, find_include_path  # 获取 tvm-ffi 与 dlpack 的头文件路径

    logger = init_logger(__name__)  # 创建本模块的 logger
    logger.info("Generating .clangd file...")  # 提示开始生成
    # 汇总所有 include 路径：tvm-ffi 自带头文件 + dlpack 头文件 + 仓库内 kernel 头文件
    include_paths = [find_include_path(), find_dlpack_include_path()] + DEFAULT_INCLUDE
    # 调用 nvidia-smi 查询 GPU 计算能力（compute capability），输出形如 "8.9" 的 CSV
    status = subprocess.run(
        args=["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True,  # 捕获 stdout/stderr 而非直接打印
        check=True,  # 命令失败（非零返回码）时抛异常
    )
    # 取第一块 GPU 的算力字符串并去掉首尾空白（如 "8.9"）
    compute_cap = status.stdout.decode("utf-8").strip().split("\n")[0]
    major, minor = compute_cap.split(".")  # 拆分主/次版本号，用于拼出 sm_XX 架构名
    # 拼接 clangd 需要的编译 flag：CUDA 语言模式、目标 GPU 架构、C++20、开启告警，以及所有 -isystem 头文件路径
    compile_flags = ",\n    ".join(
        [
            "-xcuda",  # 按 CUDA 源码处理
            f"--cuda-gpu-arch=sm_{major}{minor}",  # 指定 GPU 架构（如 sm_89）
            "-std=c++20",  # C++20 标准
            "-Wall",  # 开启常见告警
            "-Wextra",  # 开启额外告警
        ]
        + [f"-isystem{path}" for path in include_paths]  # 把每个 include 目录作为系统头文件路径加入
    )
    # 按 clangd 的 YAML 配置格式生成文件内容，把上面 flags 写入 CompileFlags.Add 列表
    clangd_content = f"""
CompileFlags:
  Add: [
    {compile_flags}
  ]
"""
    if os.path.exists(".clangd"):  # 工作目录下已存在 .clangd 文件时不覆盖
        logger.warning(".clangd file already exists, nothing done.")  # 提示未做任何修改
        logger.warning(f"suggested content: {clangd_content}")  # 打印建议内容供用户手动参考
    else:
        with open(".clangd", "w") as f:  # 在当前工作目录创建 .clangd 文件
            f.write(clangd_content)  # 写入配置内容
        logger.info(".clangd file generated.")  # 提示生成成功


generate_clangd()  # 模块加载时（作为 __main__ 运行时）立即执行生成逻辑
