# QMER HDF5批量反应流程

本功能位于 `codex/qmer-batch-resume-20261009` 分支，独立于主仓库工作目录。它读取QMER7 combined HDF5的TS，调用已有Gaussian16＋GPU4PySCF入口：

```
TSOPT → 独立TS FREQ → 双向IRC → reverse endpoint OPT → forward endpoint OPT
```

TSOPT必须有Gaussian正常结束与优化完成标志，且未发生Cartesian fallback；FREQ中低于-20 cm⁻¹的频率必须恰好一个（可通过 `--imaginary-threshold` 调整）。IRC必须正常结束，且格式化检查点中包含正、负反应坐标对应的有限、接受几何；从最远正负点分别启动endpoint OPT。两个端点都优化收敛才标记complete。失败阶段阻止后续阶段。

这继承现有仓库的科学方法和Gaussian routes。默认IRC仍为 `IRC=(CalcFC,HPC,MaxPoints=10,StepSize=10)`，是有限长度的局部双向路径；需要更长路径时在运行前显式覆盖 `routes.irc`。endpoint OPT收敛不等于已确认与数据集R/P对应，本入口保留forward/reverse命名，不未经结构检查就给它们贴R/P标签。

## 数据与分块

已检查输入 `selected.h5` 有100,360条记录，TS坐标Å，原子顺序由 `offsets/atom` 和 `atoms/atomic_numbers` 定义，根属性charge=0、multiplicity=1。输入TS来自xTB，DFT下不一定仍是合格过渡态，因此本流程单独做频率验收。

准备仅生成8个小型JSONL清单，不复制约10GB的HDF5，也不加载原始IRC轨迹/巨大Hessian数组：

```bash
python -m pip install -e '.[qmer]'
qmer-gau prepare --h5 /shared/data/selected.h5 \
  --output /shared/qmer/manifests --shards 8 --chunk-size 100
```

分配规则为selected文件原始行号 `index % 8`；每份正好12,545条。每份按清单顺序分成100条一块，最后一块45条。保存稳定反应hash、原id_num、原始行号和原子数。所有反应恰好分配一次；8份没有重叠。按行号交错也避免把按大小排列的数据连续切成8份后负载明显偏斜，但不是精确耗时平衡。

结果目录：

```
OUTPUT/shard_00/chunk_00000/000000_REACTION_HASH/
    ts_input.xyz / input.json / config.json / state.json
    forward.xyz / reverse.xyz
    stages/tsopt/attempt_001/...
    stages/ts_freq/attempt_001/...
    stages/irc/attempt_001/...
    stages/endpoint_opt_reverse/attempt_001/...
    stages/endpoint_opt_forward/attempt_001/...
```

每份保存run.json、progress.json及summary.json；反应级state.json保存每阶段指纹、attempt、状态、耗时边界及验证数据。数据集版本、路径、大小、修改时间、清单SHA256和解析后的计算配置绑定到该运行。源数据应只读；这些身份校验不是对10GB数据内容的全文件哈希。

## 每卡运行

先复制并填写 `examples/config.yaml`，推荐本次smoke使用的闭壳层设置：

```yaml
gpu:
  df_gradient_metric: solve
  conv_tol: 1e-11
  conv_tol_grad: 1e-9
  max_cycle: 200
  conv_tol_cpscf: 1e-10
  cphf_grid: scf
  hessian_memory:
    policy: "off"
```

实际GPU环境需确认cuTENSOR生效；显存不足时可在第一次运行前选择已审计的conservative策略。源代码不匹配时补丁拒绝启用。Gaussian路径、运行镜像和资源配置使用个人配置，不写入公开代码。

在8个单GPU任务中分别指定shard 0至7，共用manifest目录和结果根目录：

```bash
qmer-gau run --config /shared/config.local.yaml \
  --manifests /shared/qmer/manifests --shard 0 \
  --output /shared/qmer/results
```

一个8卡节点上，另行设置每个进程的 `CUDA_VISIBLE_DEVICES`，每个进程只看一张卡。例如卡3对应 `CUDA_VISIBLE_DEVICES=3` 和 `--shard 3`。8个独立单卡节点都可使用设备0，shard仍分别0至7。提供的 `examples/qmer-run.sh` 从SHARD_ID等环境变量读取参数，启动平台任务时使用可见的绝对路径。

可限制单块、前若干条或特定记录进行排错：

```bash
qmer-gau run --config /shared/config.local.yaml \
  --manifests /shared/qmer/manifests --shard 0 --chunk 0 --limit 5 \
  --output /shared/qmer/results
```

`--indices`是selected.h5的零基行号，必须属于所选shard。smoke结果与大规模正式运行建议使用不同output根目录。

## 断点与失败重试

重复同一命令即可恢复。完整反应直接跳过；未完成反应复用已成功阶段。崩溃前runner已生成completed=true及接受几何、但state尚未提交时，会恢复该阶段。中断阶段另开新attempt；这是阶段边界续算，不承诺恢复Gaussian中断时某一步的内存波函数或IRC轨迹。

失败/频率不合格默认跳过。只有显式 `--retry-failed` 才再次尝试；每阶段默认最多2个attempt，可通过 `--max-attempts`调整。已成功前置阶段不重算，失败attempt不覆盖。改变科学参数或几何时必须用新的output根目录，避免混用历史结果。每shard使用进程锁，同一份清单不能被两个进程同时计算；8个不同shard可独立并行。

成功阶段删除逐步GPU SCF检查点以减少磁盘量，保留Gaussian最终chk/fchk、接受结构和全部日志/参数记录；失败阶段保留诊断文件。每阶段调用已有runner并单独启动worker，阶段内几何步继续复用成功密度；阶段之间目前不复用worker或密度。短任务有重复启动开销，但隔离和恢复边界明确。

退出码0表示本次所选记录均通过或已经完成；1表示存在失败/频率拒绝；2表示输入、配置或锁错误；130表示中断。一个反应失败后继续处理清单中的其他反应。平台进程成功与科学验收分别记录，以state.json为准。

## smoke与测试

23项接口/批量测试已通过，包括8份完整覆盖、分块、阶段恢复、配置变化拒绝及从带符号IRC数据提取接受端点。真实5例smoke选取7/10/12/14/16原子各一例，每例结束后再执行相同命令并核对Gaussian日志修改时间，验证续算没有重新计算。结果与运行边界在smoke完成后补充。
