from __future__ import annotations  # 启用 PEP 563 延迟注解求值

from minisgl.kernel import test_tensor  # 导入被测 kernel：基于 TVM FFI TensorMatcher 的张量属性校验函数
from minisgl.utils import call_if_main  # 导入"仅主模块执行"装饰器
import torch  # 导入 torch，构造测试张量


@call_if_main()  # 装饰器：仅本文件作为主模块运行时执行（默认检查 __name__）
def main():
    """验证 C++ 侧 TensorMatcher 的校验逻辑：非连续 CPU 输入 + 指定 GPU 的目标张量能通过形状/dtype/设备检查。"""
    x = torch.empty((12, 2048), dtype=torch.int32, device="cpu")[:, :1024]  # CPU 非连续视图（列切片：形状 (12,1024) 但行步长仍为 2048）
    y = torch.empty((12, 1024), dtype=torch.int64, device="cuda:1")  # GPU:1 上的 int64 连续目标张量
    test_tensor(x, y)  # 调用 kernel：内部按约束校验（x 任意行步长/int32 或 float/CPU；y 连续/int32 或 int64/CUDA:1；N%4==0），不匹配则抛异常
