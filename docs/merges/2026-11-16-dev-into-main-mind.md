# dev 合入 main-mind：试合并说明

本分支用于评审合并结果，不改变现有 `main-mind`、`dev` 或部署 pin。

## 分支比较

| 项目 | revision / 结果 |
| --- | --- |
| 目标 `main-mind` | `e9c0dacf7ff9ecba350cc16e5f3db6cc842d445c` |
| 来源 `dev` | `d16c26d54a46a7d0027e95a360159ef67f79613f` |
| merge base | `d697c7b1891842dd448eb3f307cae8175a881890` |
| 分叉后的提交数 | main-mind 27；dev 522 |
| dev 相对共同祖先 | 1,315 文件，180,693 行新增、33,905 行删除 |
| main-mind 相对共同祖先 | 88 文件，21,824 行新增、1,027 行删除 |
| 双方都修改的文件 | 70 |
| Git 报告的冲突文件 | 45 |
| 试合并分支 | `codex/merge-dev-into-main-mind-20261116` |

`dev` 包含大批上游模型、训练、checkpoint、数据处理、文档和依赖更新，尤其是
ModelConfig/ModelBuilder 与 Hybrid 构建路径。它不是只多几项 fork 修复。
`main-mind` 的增量主要是部署所需的 LoRA、FP8 转换、加载性能和 DSV4.1/GLM 能力。

采用普通 merge 保留两条历史。没有逐个 cherry-pick 522 个提交，也没有把冲突文件
整体替换为任意一侧。`9695307ce` 和 `cc4b022d1` 的 targeted parameter LoRA 增删内容
相同，整合后只保留一次实现。

## 整合原则和保留的行为

- 保留 dev 的 builder、Hybrid、packed/vision/TP mapping、layer-wise optimizer、
  MFSDP v2 `no_sync`、callbacks、checkpoint sidecars 和逐 DP RNG 存储。
- 保留 main-mind 的 batched/pool LoRA、split gate/up、split QKV、受保护的 in-place
  opt-in、shared-expert EP finalize 同步和 GTP/HDO checkpoint 支持。
- DSv4 采用 dev builder，并保留 checkpoint 声明的 FP8 block grid。32×32 scale
  不会被当作默认 128×128；同时保留 dev 对缺失 gate bias 的显式补零。
- DSV4.1 保留本地 HF config 注册、直接 FP8 写入/重定量化、Engram shard readahead
  和 phase profiling。GLM5 保留部署使用的 TileLang 默认 indexer。
- GLM53 保留 dev 的 builder/packed/vision 路径，恢复 main 的直接 FP8 导入。
  保持 main 的默认 MTP 关闭策略；显式配置仍可启用 MTP，相关 round-trip 测试保留。
- 恢复 GLM53 的 `output_processor`、context 和原有 temperature 接口；processor 的
  任意返回值直接透传，不被新 packed-supervision 返回格式再次包装。
- **在线 adapter 导出保持紧凑布局**：`export_adapter_weights` 不重新开放
  `expand_shared_outer` 参数，传入它仍报 `TypeError`。离线 `save_hf_adapter` 直接调用
  下层 streamer 做显式展开，并保留 dev 的 rank0-only materialization 和错误传播。
- 保留配置 target 校验、无 native recompute boundary 时的兼容处理，以及 FSDP
  factory 不作为 `isinstance` 类型传入的保护。
- 保留普通 blockwise 缺 dtype 时的跳过语义，但显式 MXFP8 metadata 必须通过
  dtype validator；不能让旧分支静默掩盖损坏的 MXFP8 metadata。

自动合并还产生了同名函数、方法、dataclass 字段和常量的重复定义。有些后面的旧定义
会覆盖前面的新实现，例如 QKV scale 的 GPU index 修复和兼容初始化 helper。
这些重复定义已核对、去重；没有只以“冲突标记消失”作为合并完成标准。
DSW 正常包回归还抓到一个遗漏的公开兼容函数引用；已恢复
`apply_run_config_backward_compat`，并增加真实 YAML/旧 dataclass 和校验顺序回归。

## 依赖影响

Core gitlink 随 dev 从 `41b60e7a` 更新为
`902feb8eea1839d60b312cb7b6ec3b2b0724cc6d`。dev 同时更新 Transformers、PEFT、TE、
FlashInfer、cuDNN frontend 和 TileLang 的依赖约束，具体版本以合并后的
`pyproject.toml`、`uv.lock` 为准。

| 依赖 | main-mind | dev / 本次整合 |
| --- | --- | --- |
| Transformers | `>=5.8,<=5.12.1` | `>=5.15,<5.21` |
| PEFT | `>=0.18.1` | `>=0.20.0,<=0.21.0` |
| ModelOpt | `0.46.0rc1` | `0.48.0rc1` |
| FlashInfer | `0.6.8.post1` | `0.7.2.post1` |
| TE git revision | `4329ff84` | `131157b5` |

DSW 起初继承的 ModelOpt 为 `0.44.0`，缺少 dev 使用的两个新接口；按声明版本
安装隔离的 `0.48.0rc1` 后，完整 `test_model_bridge.py` 104 项通过。没有通过
删除 sparse-slot 或 quantizer rotation 检查兼容旧版本。

这次不是旧环境上的纯 Python 小补丁。保留若干旧 Core 导入保护不代表新增 builder
在每条旧 Core 线上都可完整运行。Core902 本身没有 native `deepseek_v41`，DSV4.1
实际构建仍要求包含该能力的配套 Core；普通包导入和不支持能力的明确报错应分别验证。

## 验证

DSW 正常 package 导入的综合回归与定向补测结果：

| 结果类别 | 数量 / 说明 |
| --- | --- |
| 正常预算通过 | 1,507 |
| CPU 环境跳过 | 14，包含真实依赖 CUDA buffer/RNG 的 canonical forward |
| 原有功能失败 | 1，两个父提交均可复现的 LoRA wildcard/bias promotion |
| 原 60 秒预算超时 | 1，PP placeholder epochs 的双进程测试 |
| 该 PP 用例的独立诊断 | 不改仓库断言，临时采用 180 秒预算后通过；测试耗时约 70 秒，不能记作原 60 秒门槛通过 |
| 两进程真实 Gloo | 两个 EP gradient-sync 用例，每 rank 均通过 |

综合结果按 testcase 唯一名称汇总最后一次定向结果，不累加重复运行。11 项 TE fused
测试未纳入 CPU 选择；完整 NVRx checkpoint suite 也未运行：当前 DSW 是 Python3.13，
没有 lock 中的 NVRx0.7 cp312 构建。GPU 正被其他作业使用，未抢占。

综合运行前核对 69 个 fork 源码/测试文件 SHA256。验证时 Torch2.14.0+cu130、
Transformers5.19.0、TE2.20.2、ModelOpt0.48.0rc1；CUDA 未初始化。此环境用于源代码
CPU 回归，不代表完整锁定的 Python3.12/CUDA 环境已验收。

其他完成的检查：

- 全部源码与 unit test Python AST 解析；冲突标记、重复定义和 `git diff --check` 检查。
- 冲突解决涉及的 Python 文件 Ruff 和格式检查。
- 66 项既有量化与 safetensors state CPU 测试，执行真实模块；在 macOS 使用
  namespace-only package loading 绕过 GPU registry 导入。
- 9 项 PEFT、8 项 GLM 和 GTP/checkpoint/config reader 的实际源码 CPU 语义检查。
  这些检查有明确的隔离边界，不算完整 package / CUDA 验证。

macOS 普通 pytest collection 被 Core 的 Triton 导入阻断，因此另建 DSW 隔离 CPU
环境补做正常 package 回归。没有改共享源码、全局环境或重启 Ray。

DSW 验证目录：
`/vePFS-Mindverse/user/songlin/logs/bridge_merge_dev_20261116.d2p1lnrm`。

该目录的 `checks/integration-validation-summary.json` 汇总 testcase 去重后的结果；
`final-cpu-attempt2.xml`、`final-three-cpu.xml` 和 `pp-placeholder-cpu-180s.xml`
保留综合运行及定向补测结果。`ep-gloo-attempt2.log` 保留真实双进程验证日志。

已确认的原有失败，未借此次合并修复：

- `test_load_profile.py` 的两个文本断言在原始 main-mind 上也失败；当前 profiler 输出
  包含 `(gpu … / host …)`，断言仍按旧格式匹配。
- `TestLoRA.test_lora_wildcard_matching` 在两个父提交的实际 parameter-promotion
  helper 上均能复现：宽泛 module wildcard 会命中已有 Linear 的一维 bias 并报错。
  三份 helper AST 相同，未将这个旧缺陷混入合并修复，也没有删除断言或标记 xfail。
  复现证据是 DSW `checks/peft_wildcard_baseline.json`。
- 未修改的 `peft/batched` 目录存在原有 Ruff 问题；这不包含在“冲突解决文件 Ruff
  通过”的范围中。

DSW 四节点 GPU 在检查时均有其他任务运行，未抢占 GPU。因此本分支尚不代表
真实模型 CUDA forward/backward、FP8 TE 集成或完整训练已经通过。

## 评审和部署前检查

优先评审 builder/provider 衔接、量化 sidecar/direct-write、serving/offline adapter
导出边界、shared-expert finalize 与 checkpoint 状态恢复。在实际配套 Core/TE 上补做
小模型 GPU 前反向、分布式 LoRA 梯度和 adapter/checkpoint round-trip，再修改部署 pin。
已有 native DSV4.1 能力的 Core 与上游 Core902 应作为不同验证配置记录。
