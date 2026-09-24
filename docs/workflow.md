# 联算流程与操作

## 每个优化步发生什么

Gaussian通过External协议发送原子序数、Bohr坐标、电荷、多重度和导数请求阶数。轻量客户端将请求发给节点本地的常驻worker。worker执行SCF，确认收敛后按请求返回能量、梯度或Hessian；Gaussian用这些数据继续优化。

请求0只算能量，1算能量和梯度，2再加Hessian。当前接口直接使用GPU4PySCF的SCF对象，不调用高层driver的with_grad/with_hess默认开关，因此不会因为driver默认值而给SP多算梯度/Hessian。

连续几何点将上一个成功点的密度投影到新几何，再按电子数归一化。梯度/Hessian也成功后才提交新状态；失败结果不会作为成功结果送回Gaussian。单重态使用限制性对象，其他多重度使用非限制性对象。多重度为2S+1，PySCF spin为多重度减1。

## 默认电子结构参数

| 配置 | 默认值 | 解释 |
|---|---|---|
| method / dispersion | b3lyp / d3bj | 能量及导数由GPU4PySCF负责 |
| basis / auxbasis | def2-svp / def2-universal-jkfit | 主基组/DF辅助基组 |
| density_fit | true | 开启DF |
| atom_grid / pruning | [99,590] / nwchem | 数值积分网格 |
| conv_tol | 1e-10 | SCF能量收敛阈值 |
| conv_tol_grad | 1e-7 | SCF轨道梯度阈值，不是核梯度开关 |
| conv_tol_cpscf | 1e-10 | Hessian响应阈值；原上游默认1e-6，null可保留上游默认 |
| cphf_grid | scf | Hessian响应复用SCF主网格；default恢复原SG1辅助网格 |
| direct_scf_tol | 1e-14 | 积分筛选阈值 |
| max_cycle | 100 | SCF最大循环，不是Gaussian几何步数 |
| threads / memory_mb | 1 / 32000 | worker线程数/CPU内存预算 |
| reuse_guess | true | 优化步间复用成功密度 |

完整可修改键见`examples/config.yaml`及`config.py`。未知键会报错，避免把拼错的参数当作已生效。DF梯度响应和DFT网格响应启用；Hessian显存分块是单独的显式选项，默认关闭。YAML中的科学计数法（如1e-10）按浮点数读取，JSON兼容保留。响应网格和阈值仅用于Hessian，不额外给SP/普通OPT计算二阶导数。

## Gaussian几何设置

方法与基组来自GPU配置，Gaussian route只负责外部计算与几何任务。默认均加NoSymm：

| task | 几何关键词 |
|---|---|
| sp | 无 |
| opt | Opt=(NoMicro,Redundant,MaxCycles=100) |
| tsopt | Opt=(TS,CalcFC,NoEigenTest,NoMicro,Redundant,MaxCycles=100) |
| irc | IRC=(CalcFC,HPC,MaxPoints=10,StepSize=10) |
| freq | Freq |
| force | Force |

默认OPT没有Tight，沿用已验证的普通优化阈值。需要Tight或其他步数时，在配置`routes`中覆盖对应字符串，并检查生成的input.gjf。改变参数后重新验证，避免直接拿不同设置的耗时做加速比。

```json
{"routes":{"opt":"Opt=(NoMicro,Redundant,Tight,MaxCycles=150)"}}
```

## 单点、优化与TS/IRC

```bash
gpu-gau run --config config.local.yaml --xyz molecule.xyz --task sp --charge 0 --multiplicity 1 --output runs/sp
gpu-gau run --config config.local.yaml --xyz molecule.xyz --task opt --charge 0 --multiplicity 1 --output runs/opt
gpu-gau run --config config.local.yaml --xyz ts_guess.xyz --task tsopt --charge 0 --multiplicity 1 --output runs/tsopt
```

TSOPT成功后，先对`runs/tsopt/tsopt/final.xyz`做独立freq，确认存在合适的单虚频且振动模式符合目标反应，再将这一结构作为IRC输入：

```bash
gpu-gau run --config config.local.yaml --xyz runs/tsopt/tsopt/final.xyz --task freq --charge 0 --multiplicity 1 --output runs/ts-freq
gpu-gau run --config config.local.yaml --xyz runs/tsopt/tsopt/final.xyz --task irc --charge 0 --multiplicity 1 --output runs/irc
```

这些步骤不自动串联，也不自动证明TS或反应连通性。频率中的IR/Raman强度不可用，见限制文档。大体系的TSOPT/freq/IRC可能在Hessian处失败。

## 批量与热启动

```bash
gpu-gau batch --config config.local.yaml --manifest examples/batch.json --output runs/batch
```

manifest是独立任务列表。它们串行共享常驻worker、CUDA启动成本；每个任务开始重置密度状态，避免不同分子相互污染。单个OPT内部复用成功密度。SP→OPT跨任务完整电子态缓存尚未实现，batch示例里的OPT重新从同一XYZ开始。失败后关闭worker，下一个任务用新进程；任一失败则整批退出码非零。

## 结果、失败和时间

- `summary.json`：每例completed、Gaussian正常终止/优化收敛标志、成功回调数和计时。
- `TASK/input.gjf`及`gaussian.log`：真实Gaussian输入输出。
- `TASK/final.xyz`：成功时由Gaussian格式化检查点导出的接受几何，不是最后一个试探几何；formchk为null时不生成。
- `worker_NNN/evaluations/`：每点请求、SCF日志、结果与检查点。
- `config.resolved.json`、`jobs.json`、worker版本：本次实际配置、输入与环境。

退出码0表示要求的计算判据通过；1表示至少一例计算失败；2表示输入/配置错误；130表示中断。OPT/TSOPT还必须出现优化完成标志，单纯Normal termination不够。freq正常结束并不自动满足某个虚频数量；IRC正常结束只表示局部任务完成。

SCF/梯度/Hessian分项累加只包含成功返回的回调，失败回调的计算时间可能不在这些分项内。Gaussian墙钟包括其控制及回调等待；worker首次启动单列，formchk和归档不混入Gaussian计算墙钟。诊断计时与平台排队、整批历时必须区分。

## 与Gaussian原生计算比较

原生参考可使用`B3LYP/Def2SVP EmpiricalDispersion=GD3BJ Int=UltraFine SCF=(Conver=10,MaxCycle=100) 5D 7F NoSymm`，配合自身OPT/TSOPT/IRC关键词。原生线程数、初始结构、电荷、自旋、收敛阈值均应明确记录。

GPU默认有DF，原生参考未使用DF；不能期待能量逐位相同。OPT时间还取决于轨迹、电子态和SCF循环数。当前仓库提供联算运行器，不把历史专用的原生批处理调度脚本继续作为公共接口维护。
