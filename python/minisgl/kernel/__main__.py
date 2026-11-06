# 该模块只应通过 `python -m minisgl.kernel` 执行
assert __name__ == "__main__"


# 生成 .clangd 配置文件，让编辑器按 CUDA/C++20 语法正确索引本包源码
def generate_clangd():
    # os 检查文件存在性
    import os
    # subprocess 调用 nvidia-smi
    import subprocess

    # 内核源码的默认 include 目录
    from minisgl.kernel.utils import DEFAULT_INCLUDE
    # 日志工具
    from minisgl.utils import init_logger
    # tvm-ffi 提供的 dlpack/tvm 头文件路径查找
    from tvm_ffi.libinfo import find_dlpack_include_path, find_include_path

    # 创建 logger
    logger = init_logger(__name__)
    # 提示开始生成
    logger.info("Generating .clangd file...")
    # 汇总 include 路径：tvm-ffi 与 dlpack 的头文件 + 项目默认路径
    include_paths = [find_include_path(), find_dlpack_include_path()] + DEFAULT_INCLUDE
    # 用 nvidia-smi 查询本机 GPU 的 compute capability
    status = subprocess.run(
        args=["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
        capture_output=True,
        check=True,
    )
    # 解析出形如 "8.9" 的算力字符串（取第一块 GPU）
    compute_cap = status.stdout.decode("utf-8").strip().split("\n")[0]
    # 拆出主/次版本号
    major, minor = compute_cap.split(".")
    # 组装 clangd 编译参数：按 CUDA 解析、目标本机 sm 架构、C++20、开警告，再附上全部 include
    compile_flags = ",\n    ".join(
        [
            "-xcuda",
            f"--cuda-gpu-arch=sm_{major}{minor}",
            "-std=c++20",
            "-Wall",
            "-Wextra",
        ]
        + [f"-isystem{path}" for path in include_paths]
    )
    # 生成 .clangd 的 YAML 内容
    clangd_content = f"""
CompileFlags:
  Add: [
    {compile_flags}
  ]
"""
    # 已存在则不覆盖，只给出建议内容
    if os.path.exists(".clangd"):
        # 提示已存在
        logger.warning(".clangd file already exists, nothing done.")
        # 打印建议内容
        logger.warning(f"suggested content: {clangd_content}")
    else:
        # 写入 .clangd 文件
        with open(".clangd", "w") as f:
            # 写入内容
            f.write(clangd_content)
        # 提示生成成功
        logger.info(".clangd file generated.")


# 脚本入口：直接执行生成
generate_clangd()
