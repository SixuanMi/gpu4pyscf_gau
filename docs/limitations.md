# 限制、结果验证与排错

## 科学范围

当前支持孤立分子气相计算；不支持ONIOM/MM点电荷、溶剂、周期体系或破缺自旋对称单重态的显式选择。重元素/ECP需要自行验证，当前配置没有专门ECP映射。电荷、自旋由用户提供；程序不会凭原子数自动判定基态。

Hessian协议中未实现的电响应数据用零占位，不能据此解释IR/Raman强度。可检查振动频率与模式，但过渡态还需要目标模式和IRC连通性验证。原始Hessian会检查有限性与非对称误差，再做对称化。

## 已知Hessian OOM

在已验证的库版本中，H100 80GB上41/62/78/111原子体系均曾在解析Hessian路径触发CUDA内存分配失败。独立进程重测确认主要涉及工作区和张量收缩临时副本。即使监控曲线未到100%，后续单次申请大于剩余显存仍会失败。当前仓库整理没有包含Hessian显存优化。

不要用调高Gaussian `%mem`解决GPU显存不足。不要把SCF/梯度可算误认为Hessian也一定可算。源码和依赖应固定版本；若后续优化分块或更换cuTENSOR，需独立验证数值和显存。

## 排错顺序

1. `check-config`报错：检查未知键、路径、正数阈值与JSON语法。
2. worker初始化失败：查看`worker_NNN/worker.log`，检查GPU可见性、GPU Python和动态库。
3. SCF不收敛：查看每点`pyscf.log`，核查结构、电荷、自旋与初猜。可显式提高max_cycle，但它不保证落到正确电子态。
4. Gaussian失败：查看任务`gaussian.log`和回调消息。OPT正常退出与正确驻点是不同的判据；程序也识别“negligible forces”完成语句。
5. formchk失败：检查工具路径、Gaussian环境和检查点是否完整。Gaussian成功但结果导出失败仍会报告失败。
6. 退出码不为0：保留`summary.json`和日志。失败不会生成可继续使用的成功结果。中断会尝试停止Gaussian/worker并归档现有证据；强制SIGKILL或节点故障可能无法执行归档，应保留本地scratch供恢复。

运行器生成的Unix socket只用于同一节点；不能把worker放在另一个节点后仍使用同一个socket路径。
