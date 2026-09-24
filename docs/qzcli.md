# qzcli 运行方式

默认CLI模式使用已核对的`qzcli create`接口（平台客户端版本0.1.0），认证交给qzcli。qzcli由使用者按所在平台安装、登录；先用`qzcli create --help`核对版本接口。

注意：该版本的GPU `create`调用OpenAPI Token认证；浏览器Cookie登录不能直接替代它。若所在平台仅使用qzcli保存的浏览器登录，选择下述console兼容模式。

## 配置

将`examples/qzcli.json`复制为`qzcli.local.json`，填写workspace、project、compute_group、spec、image、python和repository。资源规格应提供单GPU节点；repository、输入、输出和科学配置路径必须能被计算容器访问。适配只申请一个实例，不做分布式或多节点计算。

`python`是计算镜像内已安装GPU4PySCF的Python；`repository`是共享存储中的本仓库绝对路径。提交机的`qzcli`可填写命令名或可执行文件绝对路径。shm_gib默认64、priority默认6，实际需符合所在空间资源条件。

平台个人配置、镜像名称和ID不随公共仓库分发。命名为`*.local.json`会被.gitignore忽略；登录状态继续由qzcli自己的目录保存。

## 预览与提交

```bash
gpu-gau qz-submit --platform qzcli.local.json -- \
  batch --config /shared/project/config.local.yaml \
  --manifest /shared/project/examples/batch.json --output /shared/project/runs/batch-001
```

这条命令只打印将调用的qzcli命令。正式提交：

```bash
gpu-gau qz-submit --platform qzcli.local.json --submit -- \
  batch --config /shared/project/config.local.yaml \
  --manifest /shared/project/examples/batch.json --output /shared/project/runs/batch-001
```

适配生成的是`env PYTHONPATH=<repository>/src <python> -m gpu4pyscf_gau batch ...`，参数经shell引用，不依赖提交节点当前工作目录。run/batch参数请使用计算容器中的绝对路径。

适配不自动重试提交。CLI模式的去重行为由qzcli决定；console模式在最近200个同空间任务中检查同名任务，但这不是覆盖全部历史的幂等保证。提交超时且结果不明确时先检查是否已创建，避免重复占用资源；每次用新的任务名和输出目录。状态和取消操作交给qzcli，具体命令参考其帮助。

## console兼容模式

复制`examples/qzcli-console.json`为个人配置，将`resource_template`指向自己平台导出的单GPU资源模板。`examples/resource-template.json`只展示字段结构，真实规格应从自己的有效资源配置获取；不能只改ID而保留不匹配的CPU/GPU参数。

此模式沿用同一个`qz-submit`命令，通过训练控制台API提交，并在实际提交时读取`~/.qzcli/.cookie`。可用`cookie_file`指定本机登录状态文件、`console_url`指定HTTPS控制台地址。预览不读取登录状态；Cookie只进入请求头，不会保存到结果、脚本或仓库。

缺失或过期登录时，使用qzcli登录/更新Cookie后重试；不要把凭据放进平台JSON或发到聊天中。本次真实H100验证使用此兼容模式。CLI模式的参数拼接已测试，但当前部署没有配置OpenAPI认证，不能宣称两种认证均已完成实际提交。

## 非qz平台

直接执行README中的`gpu-gau run`或`batch`即可。用户自行在Slurm等调度器中分配GPU、载入环境并调用该命令。本仓库不生成这些平台的提交脚本。
