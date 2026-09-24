# GPU4PySCF + Gaussian External

用GPU4PySCF计算能量和导数，用Gaussian进行几何优化和IRC步进。适合已安装Gaussian 16、GPU4PySCF及兼容CUDA环境的Linux GPU计算节点。

Gaussian每次给出结构，常驻GPU worker完成SCF并按需返回梯度/Hessian；Gaussian据此产生下一步结构。worker复用上一个成功点的密度作为初猜，回调客户端只使用Python标准库。

## 快速开始：直接运行

先在已有GPU4PySCF环境安装本项目。此命令只安装接口，不自动选择或升级CUDA、CuPy和GPU4PySCF：

```bash
python -m pip install -e .
cp examples/config.json config.local.json
```

编辑`config.local.json`中的三个Gaussian路径：`executable`为g16可执行文件，`exedir`为Gaussian程序目录，`formchk`为格式化检查点工具。已有g16配置在PATH中时，也可以使用相应命令名。CUDA环境需由用户预先配置，详见[安装说明](docs/installation.md)。

```bash
gpu-gau check-config --config config.local.json
gpu-gau run --config config.local.json --xyz examples/water.xyz \
  --task sp --charge 0 --multiplicity 1 --output runs/water-sp
gpu-gau run --config config.local.json --xyz examples/water.xyz \
  --task opt --charge 0 --multiplicity 1 --output runs/water-opt
```

成功输出位于`runs/water-opt/opt/`，包括`gaussian.log`、`gaussian.chk`、`gaussian.fchk`和`final.xyz`。根目录`summary.json`记录是否真正完成及计时。已有输出目录不会被覆盖。

直接运行入口只在当前节点计算，适合用户自行接入Slurm等调度系统；本项目不提供这些平台的提交脚本。请先申请所需GPU资源，再执行命令。

## qzcli 提交

计算入口相同，只由qzcli负责申请节点和启动。用户在共享目录准备本项目、输入及配置，并填写`examples/qzcli.json`的副本：

```bash
cp examples/qzcli.json qzcli.local.json
# 先编辑个人平台配置；路径须在计算容器内可见。
gpu-gau qz-submit --platform qzcli.local.json -- \
  run --config /shared/project/config.local.json \
  --xyz /shared/project/examples/water.xyz \
  --task sp --charge 0 --multiplicity 1 --output /shared/project/runs/water-sp
```

默认只预览命令。确认路径和资源后，在`--platform`之后加`--submit`实际提交。默认CLI模式由qzcli管理认证；使用浏览器登录的平台可显式选择console兼容模式，仅在提交时读取本机qzcli登录状态，绝不写入运行配置或仓库。[完整说明](docs/qzcli.md)。

## 支持范围

| 任务 | Gaussian控制 | GPU计算 |
|---|---|---|
| `sp` | 单点 | 能量 |
| `opt` | 普通结构优化 | 能量、梯度；按请求阶数决定是否需要Hessian |
| `tsopt` | TS优化，默认CalcFC | 能量、梯度、初始Hessian |
| `irc` | 默认局部IRC | 能量、梯度、按请求计算Hessian |
| `freq` | 振动频率分析 | 能量、梯度、Hessian |
| `force` | 单点力/梯度 | 能量、梯度 |

默认方法是B3LYP-D3BJ/def2-SVP，DF辅助基组def2-universal-jkfit，无溶剂。XYZ采用Å，电荷和自旋多重度必须显式提供。

**已知限制：当前库版本在部分大体系Hessian上会OOM；本次整理没有加入Hessian优化。** 零占位的电响应数据不能用于IR/Raman强度。TSOPT正常结束不自动等于正确过渡态，IRC局部路径不自动证明反应物/产物连通性。详见[限制与验证](docs/limitations.md)。

## 文档

- [安装与环境](docs/installation.md)
- [联算原理、参数和具体操作](docs/workflow.md)
- [qzcli配置与提交](docs/qzcli.md)
- [已知限制与排错](docs/limitations.md)
- [历史基准和实测显存摘要](docs/benchmarks.md)
- [仓库结构、验证及旧数据归档](docs/maintenance.md)

CPU接口测试：`python -m unittest discover -s tests -v`。GitHub CI只跑接口测试，不包含专有Gaussian程序或GPU数值测试。

本仓库不附带Gaussian二进制、许可证、GPU依赖源码、个人平台配置或原始大分子数据。发布前请由项目所有者选择本项目许可证；目前未擅自指定授权条款。
