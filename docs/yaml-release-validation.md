# YAML配置与cuTENSOR镜像运行验证（2026-09-24）

验证实现提交：`787054b`，功能分支 `codex/h100-hessian-memory`。构建wheel后安装到新的独立目录，通过实际安装包执行CLI，未使用源码目录替代wheel中的实现。

## 用户配置

主示例为 `examples/config.yaml`，JSON仍兼容。两个外层配置项为：

```yaml
gpu:
  # 新默认1e-10；原受测上游默认1e-6，原接口null表示沿用上游。
  conv_tol_cpscf: 1e-10
  # 新默认scf，复用SCF主网格；原默认default为SG1(50,194)辅助网格。
  cphf_grid: scf
```

显式设为 `conv_tol_cpscf: 1e-6` 和 `cphf_grid: default` 可恢复原测试设置；null允许保留所安装上游版本自己的阈值。没有DFT网格的HF配置在未指定网格时保持default。SCF主网格本身仍由 `atom_grid` / `pruning` 控制，未改动其默认值。

YAML通过安全解析器读取，裸科学计数法 `1e-10` 被识别为浮点数。自定义Python对象标签被拒绝。PyYAML>=6.0是接口新增的配置依赖，不自动升级CUDA/CuPy/GPU4PySCF。

## 验证环境与结果

用户提供镜像：`docker.sii.shaipower.online/inspire-studio/gpu4pyscf-fast:20260924`。

实际设备为H100 80GB HBM3；GPU4PySCF 1.8.1、PySCF 2.8.0、CuPy 13.6.0、CUDA runtime 12.9，安装的配置依赖PyYAML 6.0.3。`backend_diagnostics()`确认实际收缩后端为cuTENSOR，两个cuTENSOR扩展导入均成功；本轮不再借用原H100隔离overlay。

20项轻量测试全部通过，覆盖原External接口、配置缺省值、JSON兼容、YAML科学计数法、旧配置覆盖、非法/不安全YAML、HF兼容及作用域恢复。有效wheel元数据包含PyYAML依赖。

真实Gaussian External采用水分子、B3LYP-D3BJ/def2-SVP、DF def2-universal-jkfit。显式启用闭壳层 `df_gradient_metric: solve` 和 `hessian_memory.policy: conservative`，验证这两个分支已有的作用域补丁与新配置同时工作。它们的公共默认值仍分别是original/off。

| 用例 | 实际导数请求 | 结果 |
|---|---|---|
| YAML SP | 0 | Gaussian正常结束；梯度/Hessian计时均为0 |
| YAML OPT | 1, 1, 1 | 正常结束且优化收敛；导出接受几何；无Hessian计算 |
| YAML FREQ，新设置 | 2 | 正常结束；worker记录CPHF为1e-10、scf，cuTENSOR实际生效 |
| YAML FREQ，恢复旧设置 | 2 | 正常结束；worker记录CPHF为1e-6、default |

新设置FREQ的三个振动频率为1599.4385、3904.2312、4009.9435 cm⁻¹；该用例读取原始水分子XYZ，是接口运行检查，不以它代替优化后频率或TS验收。显式梯度补丁在全部梯度请求中确认已应用。批量SP/OPT/FREQ约31.62秒，此数包含冷启动影响，不用于新旧精度设置的加速比比较。

## 证据与边界

私有完整记录位于Git忽略目录 `runs/yaml_cutensor_release_20260924/`，包括实际镜像、提交与wheel哈希、后端探测、配置、Gaussian输出、worker参数记录和 `validation.json`（passed=true）。初次调度因启动命令使用相对路径而取消，改用绝对路径后的最终作业成功；数值结论仅基于最终作业。

本轮验收外层YAML到真实Gaussian/GPU worker的完整传参和运行路径，未新增大体系精度扫描、TSOPT或IRC轨迹测试。此前大体系显存和导数一致性结果见对应历史文档，不把本次小分子通过推广为所有体系均已验证。
