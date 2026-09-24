# GPU4PySCF + Gaussian External

用GPU4PySCF计算能量和导数，用Gaussian进行几何优化和IRC步进。适合已安装Gaussian 16、GPU4PySCF及兼容CUDA环境的Linux GPU计算节点。

Gaussian每次给出结构，常驻GPU worker完成SCF并按需返回梯度/Hessian；Gaussian据此产生下一步结构。worker复用上一个成功点的密度作为初猜，回调客户端只使用Python标准库。

## 快速开始：直接运行

先在已有GPU4PySCF环境安装本项目。此命令只安装接口，不自动选择或升级CUDA、CuPy和GPU4PySCF：

```bash
python -m pip install -e .
cp examples/config.yaml config.local.yaml
```

YAML支持注释，旧JSON配置仍可通过`--config`读取。Hessian响应默认使用SCF主网格和`conv_tol_cpscf: 1e-10`；示例注释写明原默认值及恢复方法。

编辑`config.local.yaml`中的三个Gaussian路径：`executable`为g16可执行文件，`exedir`为Gaussian程序目录，`formchk`为格式化检查点工具。已有g16配置在PATH中时，也可以使用相应命令名。CUDA环境需由用户预先配置，详见[安装说明](docs/installation.md)。

```bash
gpu-gau check-config --config config.local.yaml
gpu-gau run --config config.local.yaml --xyz examples/water.xyz \
  --task sp --charge 0 --multiplicity 1 --output runs/water-sp
gpu-gau run --config config.local.yaml --xyz examples/water.xyz \
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
  run --config /shared/project/config.local.yaml \
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

零占位的电响应数据不能用于IR/Raman强度。TSOPT正常结束不自动等于正确过渡态，IRC局部路径不自动证明反应物/产物连通性。详见[限制与验证](docs/limitations.md)。

## cuTENSOR与Hessian显存

**已验证：启用cuTENSOR后，H100 80GB上的62原子（709 AO）和78原子（843 AO）算例，关闭本项目额外分块也能完成完整Hessian。** 此前回退到CuPy时，这两个算例的未修改路径曾OOM。因此不能把历史OOM概括为“这些体系必然需要本项目分块”。

cuTENSOR是推荐的可选张量收缩后端。本次审计的CuPy回退路径会为部分收缩分配额外中间结果，即使传入`out`也未必原地计算；启用cuTENSOR能改变这部分工作区需求。应在实际worker环境确认后端生效，而不只检查pip安装记录：

```bash
python -c 'from gpu4pyscf_gau.hessian_memory import backend_diagnostics; import json; print(json.dumps(backend_diagnostics(), indent=2))'
```

输出中的`effective_backend`应为`cutensor`。已审计版本通过成功导入库自动选择后端，不支持手动设置`CONTRACT_ENGINE=cutensor`。实测组合为GPU4PySCF 1.8.1、PySCF 2.8.0、CuPy 13.6.0、cuTENSOR 2.2.0，具体安装和动态库配置见[安装说明](docs/installation.md)。

| 已完成的H100验证 | 本项目额外分块 | 响应设置与结论 |
|---|---|---|
| 62/78原子完整原库Hessian | `off` | 原CPHF设置：1e-6＋SG1辅助网格；cuTENSOR下均完成、未OOM |
| 62/78原子高精度导数一致性检查 | `conservative` | 1e-10＋SCF主网格，配合稳定DF梯度及更严格SCF；已测方向通过1e-5 Eh/Bohr²阈值 |
| 新cuTENSOR镜像的Gaussian External水分子SP/OPT/FREQ | `conservative` | YAML新参数传递及真实运行通过；SP/普通OPT没有额外Hessian计算 |

**上述未分块成功记录采用原CPHF设置，不能直接视为新高精度设置关闭分块后的显存保证。** 也尚未验证111原子在cuTENSOR下关闭本项目分块，不能将62/78原子的结果直接推广到更大基组或显存更少的设备。基函数数、响应设置和收缩后端都会影响内存需求；原子数不能单独决定能否计算。完整对照见[Hessian精度与cuTENSOR复核](docs/hessian-accuracy-followup.md)。

`gpu.hessian_memory.policy`默认`"off"`，保留上游内存规划与已有分块算法；它并不表示上游完全不分块。需要额外限制中间张量大小时，可显式设为`conservative`。本项目策略仅支持已审计源码、单GPU串行worker，会校验源码并在计算结束后恢复原函数；它不能保证所有体系不OOM，也可能增加耗时。上游梯度的`lowmem`不是此处的Hessian开关。详见[显存策略](docs/hessian-memory.md)。

## Hessian精度参数与稳定DF梯度

当前外层YAML默认如下，完整示例及旧值注释见[config.yaml](examples/config.yaml)：

```yaml
gpu:
  conv_tol_cpscf: 1e-10  # 原受测上游默认1e-6；null表示沿用所安装上游的默认阈值
  cphf_grid: scf        # 原默认default：SG1(50,194)辅助网格
  df_gradient_metric: original  # 稳定DF梯度需显式启用solve
  hessian_memory:
    policy: "off"      # 额外显存限制需显式启用conservative
```

`cphf_grid: scf`在Hessian计算前设置`mf.cphf_grids = mf.grids`，复用SCF主网格；本项目默认主网格为99×590、nwchem裁剪。设为`default`可恢复原辅助网格。这两个响应参数仅在Hessian请求时生效，不给SP或只请求梯度的OPT增加二阶导数计算；已测62/78原子的完整Hessian采用高精度响应设置时，单次耗时相对原响应设置增加约39%/42%，不能将此比例套用于整个优化流程。

`df_gradient_metric: solve`是独立的、默认关闭的稳定性修正：把DF梯度中的显式逆矩阵作用改为直接求解等价线性方程，不改变泛函、基组或删减响应项。当前仅对通过源码校验的单GPU闭壳层DF开放，开壳层梯度会明确拒绝启用。显存分块对照与梯度/Hessian一致性是两项独立验收；仅启用cuTENSOR或新CPHF默认值，不等于已包含这项梯度修正。

62/78原子的方向差分验证使用`solve`、SCF阈值1e-11/1e-9、高精度CPHF及`conservative`；只覆盖指定方向和步长，未重跑完整TSOPT/IRC轨迹。应用配置及精度范围见[导数一致性说明](docs/gradient-hessian-consistency.md)。


## 文档

- [安装与环境](docs/installation.md)
- [联算原理、参数和具体操作](docs/workflow.md)
- [qzcli配置与提交](docs/qzcli.md)
- [已知限制与排错](docs/limitations.md)
- [历史基准和实测显存摘要](docs/benchmarks.md)
- [Hessian显存策略与兼容性](docs/hessian-memory.md)
- [GPU梯度/Hessian一致性参数与修正](docs/gradient-hessian-consistency.md)
- [H100分块、精度与cuTENSOR实测](docs/hessian-accuracy-followup.md)
- [YAML配置与新cuTENSOR镜像运行验证](docs/yaml-release-validation.md)
- [仓库结构、验证及旧数据归档](docs/maintenance.md)

CPU接口测试：`python -m unittest discover -s tests -v`。GitHub CI只跑接口测试，不包含专有Gaussian程序或GPU数值测试。

本仓库不附带Gaussian二进制、许可证、GPU依赖源码、个人平台配置或原始大分子数据。发布前请由项目所有者选择本项目许可证；目前未擅自指定授权条款。
