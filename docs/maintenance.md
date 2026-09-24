# 仓库维护与旧资料迁移

## 当前结构

```text
README.md                 快速开始
pyproject.toml            安装元数据与gpu-gau命令
src/gpu4pyscf_gau/         唯一维护的接口、worker、协议与运行器
examples/                 水分子、公开配置模板、批量模板
docs/                     统一操作、限制与基准文档
tests/                    CPU协议与Unix socket集成测试
.github/workflows/        不依赖GPU/Gaussian的CI
```

仓库不存放大体积运行结果。默认输出写入被忽略的runs目录；个人配置使用`*.local.json`。`local_archive.json`是本部署机器上的归档位置索引，不上传GitHub。

## 合并规则

旧`code/`、`code_four_*`、`code_optimized_*`不再并列作为入口。经过验证的worker/协议逻辑归入src；专用平台ID、绝对环境路径和历史批次控制器从公共入口移除。

历史README、测试报告、运行记录中可重复使用的说明合入installation/workflow/qzcli；结果和失败结论合入benchmarks/limitations。smoke、阶段性日志、旧版本代码及完整原始结果整体移至仓库外归档，未把唯一证据直接删除。旧文件记录的绝对路径保持原样，可按归档索引映射；它们不应继续作为新作业启动脚本使用。

已讨论的Hessian优化方案属于未来工作，不进入当前发布版，也不宣称已修复OOM。

## 验证与发布

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
gpu-gau check-config --config examples/config.json
```

本地CPU测试包含协议格式、Hessian打包、失败判据、路径空格、真实Unix socket回调、worker复用和失败归档。mock Gaussian测试验证接口，不替代数值计算。真实GPU/Gaussian验证单独记录在validation.md。

发布前检查`git status`和`git ls-files`，确保没有个人配置、日志、检查点、登录状态或平台账户信息。用户数据和大分子结构未纳入公共仓库。该项目未包含Gaussian二进制或第三方库源码。

许可证应由项目所有者决定；目前未指定本项目代码的开源授权，不将依赖许可证误用作本仓库许可证。添加远端URL和推送由所有者按实际GitHub仓库操作；整理过程不会自动创建公开仓库或推送。
