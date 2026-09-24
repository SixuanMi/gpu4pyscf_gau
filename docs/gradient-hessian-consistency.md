# GPU 梯度与解析 Hessian 一致性

本页的验收对象是 GPU 计算自身的方向导数一致性，不要求匹配另一套 CPU 软件的绝对结果。

H100 复核定位出两个需要分开处理的问题：原 DF 梯度显式构造逆矩阵的算术路径造成主要差分偏差；默认 CPHF 响应阈值和辅助网格又在部分方向留下较小偏差。Hessian 显存分块本身在同轨道完整原库对照中的误差只有约 `1e-9 Eh/Bohr²`。证据、范围和各轮数据见[精度复核记录](hessian-accuracy-followup.md)。

## 精度配置

对本次审计和验证的闭壳层、气相、DF B3LYP-D3BJ体系，配置片段为：

```json
{
  "gpu": {
    "df_gradient_metric": "solve",
    "conv_tol": 1e-11,
    "conv_tol_grad": 1e-9,
    "max_cycle": 200,
    "conv_tol_cpscf": 1e-10,
    "cphf_grid": "scf",
    "hessian_memory": {"policy": "conservative"}
  }
}
```

最后一项是独立的显存策略；有足够显存时可使用 `off`。装好cuTENSOR后，本次62/78原子的未分块原库也能完整计算。精度修正不依赖开启显存分块。

- `df_gradient_metric` 默认 `original`，新选项 `solve` 显式启用。它只作用于请求阶数1/2的梯度，不增加SP的导数计算。
- `conv_tol_cpscf` 默认 `null`，保留上游默认阈值；设为正数时只在Hessian计算前传给 `mf.conv_tol_cpscf`。它是上游的输入控制值，上游可能按响应批次规模缩放求解停止准则。
- `cphf_grid` 默认 `default`，保留上游辅助响应网格；`scf` 让 `mf.cphf_grids=mf.grids`，与SCF主网格匹配。仅对DFT有效，不改变已有SCF网格本身。
- 不应只打开 `solve` 就假设所有方向都通过：62原子的Ni坐标和最负曲率方向还需要上述响应精度设置。

这些选项均保持原默认行为，未自动写入个人生产配置。科学方法、基组、辅助基组和响应项保持不变；响应网格与迭代精度的提高会增加Hessian耗时，比较速度时也需固定这组设置。

## 修正在哪里

`src/gpu4pyscf_gau/gradient_metric.py` 只在梯度调用期间替换已审计的 `gpu4pyscf.df.grad.rhf._jk_energy_per_atom`。原实现的度量作用可以写成：

```
M = C solve(J, C.T)
D = M B
```

修正后的计算次序为：

```
Y = solve(J, C.T B)
D = C Y
```

两者在精确算术中相同。新次序避免显式形成逆矩阵后再与另一组张量相乘；保留原Cholesky/特征分解选择、线性相关阈值、双精度、完整辅助基组与所有梯度响应项。没有通过放宽差分验收阈值、删减响应项或降低基组来通过检查。

`worker.py` 包裹梯度计算并记录 `gradient_metric.json`；成功/异常退出均恢复上游函数。Hessian计算前单独设置CPHF控制，并在 `hessian_memory.json.response_accuracy` 记录实际参数。磁盘上的GPU4PySCF安装文件不被覆盖。

补丁校验整个梯度源文件SHA256：

```
1b5655cf302c8f3206e7cd264d0022153592857bd34304524507a31ce50cb172
```

源文件不同会拒绝启用，不应仅凭版本号相同跳过审计。当前作用域补丁仅开放给单GPU、串行worker、restricted DF；开壳层尚未验证，显式设置 `solve` 后请求开壳层梯度会报错，不会悄悄套用闭壳层补丁。纯SP不执行此补丁。

## 如何验证

对同一几何，使用已收敛中心密度分别初始化正负位移点，计算

```
[g(R + h u) - g(R - h u)] / (2h)  与  H(R) u
```

其中 `u` 可取单个笛卡尔坐标，也可取最负曲率特征向量。使用多个步长检查误差和收敛趋势；只验证某一列不代表独立验证了全矩阵，更不等于完整TSOPT/IRC已验收。

已生成解析Hessian的案例目录可运行：

```bash
python tools/hessian_finite_difference.py \
  --case RUN_ROOT/case --output NEW_FD_DIRECTORY \
  --axis 3 --steps .002 .001 \
  --conv-tol 1e-11 --conv-tol-grad 1e-9 \
  --df-gradient-metric solve
```

案例中的 `config.json`、`request.json` 和 `derivatives.npz` 必须对应正确的解析Hessian参考及配置。本工具检查最大元素误差 `1e-5 Eh/Bohr²`，未通过时保存证据并以非零状态退出。GPU运行、谱方向检查和完整External接口回归属于不同层次的验证，分别报告。
