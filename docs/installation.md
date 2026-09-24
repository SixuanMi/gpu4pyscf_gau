# 安装与环境

## 需要准备什么

- Linux计算节点、NVIDIA GPU及驱动；Unix domain socket需要节点本地可用。
- Python 3.10以上。接口本身只用标准库；worker需要numpy、PySCF、GPU4PySCF、CuPy和B3LYP/D3BJ所需依赖。
- 用户自行安装且有权使用的Gaussian 16，包含g16及formchk。

历史验证组合是GPU4PySCF 1.8.1、PySCF 2.8.0、CuPy 13.6.0、Gaussian 16 B.01。它是复现参考，不代表其他版本自动兼容。CUDA、驱动、CuPy wheel和cuTENSOR的匹配由用户环境负责；本项目不通过pip自动升级它们。

使用GPU环境的Python执行：

```bash
python -c 'import numpy,pyscf,gpu4pyscf,cupy; print(pyscf.__version__,gpu4pyscf.__version__,cupy.__version__); print(cupy.cuda.runtime.getDeviceCount())'
python -m pip install -e .
gpu-gau --help
```

也可不安装接口，在仓库根目录运行`PYTHONPATH=src python -m gpu4pyscf_gau ...`。不同Python通过`runtime.worker_python`选择；worker导入路径会指向同一套接口代码。

## Gaussian配置

`gaussian.executable`是完整g16文件路径，不是其目录；`gaussian.exedir`对应GAUSS_EXEDIR；`gaussian.formchk`指向formchk。未指定exedir且executable是路径时，默认取它的父目录。

单任务的`%nprocshared`默认1，`%mem`默认32GB。`gaussian.memory`和`gpu.memory_mb`都是主机内存设置，不是显存上限。工作区默认使用节点临时目录；可用`runtime.scratch_dir`指定本地高速盘。Unix socket始终在短的临时路径下，避免长共享目录超出系统路径限制。

配置文件中的相对程序路径相对于配置文件本身；`run --xyz`相对当前工作目录；batch中xyz相对manifest。支持路径中的空格，支持程序路径的`~`和环境变量展开。

formchk默认启用。将其设为null可跳过格式化与final.xyz导出，适合只读Gaussian原始结果的用户；此时优化后的结构需要自行从Gaussian检查点提取，不能拿最后一个worker试探点当最终结构。

## CUDA环境隔离

优先在已正确配置的环境运行。如果GPU库需要特定动态库路径或预加载，可仅配置给worker：

```json
{
  "runtime": {
    "worker_environment": {
      "CUDA_PATH": "/path/to/cuda",
      "LD_LIBRARY_PATH": "/path/to/cuda/lib:/path/to/environment/lib",
      "LD_PRELOAD": "/path/to/environment/lib/libcusolver.so.11"
    }
  }
}
```

以上是可选示意，路径和库名应以自己的环境为准，不要照抄不存在的库。Gaussian子进程会移除LD_PRELOAD；轻量External客户端也移除它，并使用`python -S`避免每次回调载入GPU库。Gaussian额外环境可用`gaussian.environment`指定。

提交节点上可运行`check-config`，它不导入CUDA，也不证明GPU计算可用。数值验证需在分配到GPU的计算节点完成。
