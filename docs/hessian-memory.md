# Hessian 显存策略

此功能是显式启用的、针对已审计GPU4PySCF构建的兼容补丁，默认关闭。它处理气相DF Hessian的J/K导数分块和CPHF轨道响应批次，不把上游名称中带lowmem的其他功能当作Hessian开关。

这是实验功能。显存与分块一致性测试、独立GPU梯度差分测试的结果须分别看待。后续对照已定位原先62原子的主要差分偏差来自DF梯度的度量矩阵运算及响应精度；完整未分块Hessian也有该偏差。修正方式见[GPU内部导数一致性配置](gradient-hessian-consistency.md)，分层验收数据见[精度复核](hessian-accuracy-followup.md)。

## 启用与关闭

在已有配置的`gpu`节加入：

```json
{
  "gpu": {
    "hessian_memory": {
      "policy": "conservative",
      "memory_fraction": 0.6,
      "reserve_gib": 8,
      "aux_batch_cap": 256,
      "aux_block_cap": 128,
      "integral_block_cap": 128,
      "response_aux_cap": 256,
      "response_dm_cap": 8,
      "h1_aux_cap": 256
    }
  }
}
```

所有参数均可省略，只设置`policy`即可采用上述默认值。设置`policy: "off"`或删除整个配置恢复原行为。需要`density_fit: true`，当前仅支持单GPU、隔离的串行worker。

| 参数 | 含义 |
|---|---|
| memory_fraction | DF Hessian局部规划使用上游报告可用显存的比例，范围(0,0.8] |
| reserve_gib | 规划时至少扣除的安全余量；不是系统强制显存上限 |
| aux_batch_cap | 辅助外层块上限，正16倍数 |
| aux_block_cap | 原子/占据轨道导数张量辅助子块上限，正8倍数 |
| integral_block_cap | 稠密积分处理块上限，正8倍数 |
| response_aux_cap | CPHF J/K辅助基组块上限，正8倍数 |
| response_dm_cap | CPHF每批响应密度数量上限，正整数 |
| h1_aux_cap | 构造一阶有效势的辅助积分批次上限，正8倍数，运行时必须至少容纳最大辅助基壳层 |

预算取`min(可用量×memory_fraction, 可用量−reserve_gib)`，保留上游各调用点对内存池的计入/排除规则，并在现有算法选出的块大小上再取上限；不会扩大已有小块或尾块。该策略为临时副本留余量，但不是整个Hessian峰值的严格上界，其他导数和响应阶段仍可能耗费显存。较小的块也可能增加计算时间。

## 修改在哪里

磁盘上的代码改动在本项目`hessian_memory.py`、`worker.py`、`config.py`，不覆盖环境的site-packages文件。

仅在worker调用解析Hessian期间，策略临时替换五个运行时函数：

| GPU4PySCF内部函数 | 修改内容 |
|---|---|
| `gpu4pyscf.df.hessian.rhf._jk_energy_per_atom` | 闭壳层J/K导数的显存预算、辅助外层块、辅助子块及积分块 |
| `gpu4pyscf.df.hessian.uhf._jk_energy_per_atom` | 开壳层对应的规划 |
| `gpu4pyscf.df.hessian.rhf._get_veff` | 闭壳层一阶有效势（make_h1）的预算、辅助积分批次和积分子块 |
| `gpu4pyscf.df.hessian.uhf._get_veff` | 开壳层一阶有效势对应的规划 |
| `gpu4pyscf.df.hessian.rhf._get_jk` | 闭壳层和开壳层共用的CPHF J/K响应预算、辅助块及响应密度批次 |

原函数的算术、循环累加、响应项和导数设置保留。RKS/UKS的原入口继续调用这些函数。成功或异常退出都会恢复原函数，SP和梯度不进入该作用域。第二处响应补丁是完整Hessian测试发现的必要改动：仅修正最初的J/K导数收缩，仍可能在后续CPHF中产生过大的临时输出而OOM。

111原子测试还发现一阶有效势构造阶段的相同问题，因此第三版补丁覆盖`_get_veff`。`h1_aux_cap`控制辅助积分批次；该阶段的稠密积分子块继续使用`integral_block_cap`。它没有重写一阶有效势公式，也没有强制减少占据轨道数量。

CPHF预算缩小也会使原有的CDERI转移到主机内存的判据更保守。本策略不在一次张量收缩OOM后原地重试：当前CuPy回退实现可能先修改`out`再申请临时量，盲目重试可能导致重复缩放或累加。需要重新计算时，应从干净的请求/进程开始。

补丁从实际安装函数生成AST，只包裹白名单规划赋值；完整源文件SHA256必须匹配内置审计值，且所有预期类别都必须命中。所有函数准备成功后才进行替换。发生版本或源码差异时明确失败，不能仅凭GPU4PySCF显示相同版本号就绕过检查。

已审计构建标记为GPU4PySCF 1.8.1、PySCF 2.8.0、CuPy 13.6.0；不同来源的同版本文件也可能不同。哈希和具体匹配范围见`src/gpu4pyscf_gau/hessian_memory.py`。版本升级时需要重新审计/验证，而不是自动添加新哈希。不能在同一进程并发运行其他Hessian调用。

数值验证覆盖B3LYP-D3BJ/def2-SVP、def2-universal-jkfit、闭壳层与开壳层。其他方法和基组仍需独立验证；减小批次不能消除所有随体系规模增长的固定张量。

## lowmem、cuTENSOR与CPU内存

上游有不同的低内存功能，包括低内存SCF及PCM相关实现；它们不等于当前气相DF Hessian J/K路径的统一低内存开关。本补丁没有修改SCF/gradient的lowmem选项，也没有减少网格、改变辅助基组或关闭导数响应。

验证镜像中`CONTRACT_ENGINE`未设置，两个cuTENSOR导入均因缺少可加载的`libcutensor.so.2`而失败，因此实际后端为CuPy。新策略继续使用这个后端，以单独验证分块效果。没有自动安装/升级CuPy、CUDA或cuTENSOR。

更换cuTENSOR可以另作环境优化，但其版本需要与CuPy匹配，也可能有自身workspace；参见[上游安装说明](https://github.com/pyscf/gpu4pyscf#installation)。这份已审计代码不接受`CONTRACT_ENGINE=cutensor`字符串，导入成功且未强制其他后端时才自动选用cuTENSOR。

Gaussian `%mem`和`gpu.memory_mb`是主机内存设置，不能控制这里的GPU临时副本。大块CUDA分配可能绕过CuPy内存池，不能只靠池上限保证整个调用不OOM。

## 结果与证据

每个请求Hessian的evaluation目录新增`hessian_memory.json`，包含实际后端、配置、是否应用、安装源码路径/哈希、每次原始及限制后的规划值。失败时同样尝试保存；硬性SIGKILL无法保证最终落盘。

项目仍先检查原始Hessian的有限性和对称性，再对称化。不能把“没有OOM”替代数值正确性验证。测试工具位于`tools/hessian_probe.py`和`tools/hessian_suite.py`：使用独立进程与约20ms NVML采样，保留请求、数值结果和阶段标记，不依赖Gaussian。完整External协议回归还需要另跑Gaussian。

本分支的实测结果和适用限制记录于[验证记录](hessian-memory-validation.md)；原始运行输入及个人资源配置留在忽略的`runs/`目录中。`tools/hessian_finite_difference.py`可在已完成的固定几何案例上，用两种步长的梯度中心差分检查一列解析Hessian。
