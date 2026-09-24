# 本次仓库整理的验证

2026-09-24。验证针对通用运行器及打包，不是新的性能基准，也未实施Hessian优化。

## Python安装与接口

Python 3.10隔离环境完成editable安装，`gpu-gau --help`和`check-config`可运行。CPU测试覆盖External协议、Hessian打包、fchk解析、优化完成判据、真实Unix socket回调、带空格路径、worker复用、失败返回、输出保护和qzcli配置构造。

GitHub CI只运行这些CPU接口测试，不要求Gaussian许可证或GPU。

## H100 + Gaussian真实验证

使用原GPU4PySCF 1.8.1 / PySCF 2.8.0 / CuPy 13.6.0环境和Gaussian 16，通过新qz-submit的console兼容模式启动同一个直接batch入口。

| 算例 | 结果 | External导数请求 | 能量 / Eh |
|---|---|---|---:|
| 示例水分子SP | 正常结束，结果导出成功 | 0 | −76.35873332533 |
| 示例水分子OPT | Gaussian优化完成，结果导出成功 | 1、1、1 | −76.35889960653 |

两例共用一个常驻worker。SP梯度/Hessian耗时均为0，OPT Hessian耗时为0；优化结构由formchk结果导出。没有把模型Hessian或原始SCF猜测当作请求的解析Hessian。

qzcli的CLI模式参数与引用经过测试，但本部署未配置其OpenAPI Token，因此没有实际通过CLI模式提交。console模式使用本机qzcli登录状态完成真实提交。两者的认证区别已在qzcli文档说明。

新打包入口本次实际复测的是SP和OPT；TSOPT、IRC、freq的原数值worker沿用历史实现，没有宣称完成新的大体系Hessian验证。历史能力及失败范围见benchmarks和limitations。

## 数据保留

真实验证日志、配置与个人平台回执保存在本部署的仓库外归档中，位置见被Git忽略的`local_archive.json`。公共仓库只保留上述无个人路径的摘要。
