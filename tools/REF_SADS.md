# 冻结 WeDetect-Ref 的 SADS-inspired 小规模实验

本入口只做冻结推理和事后分析，不包含训练。它实现 `results/ref_sads_plan_20260923/EXPERIMENT_PLAN.md` 中的单头先导方案。代码构建与离线测试不代表真实 checkpoint 的一致性检查已通过，也不预设存在 re-sinking 或抑制收益。

## 文件

| 文件 | 作用 |
|---|---|
| `config/ref_sads_pilot_v1.json` | 固定样本量、层、种子、GMM 规则及来源 SHA |
| `tools/ref_sads.py` | prepare/check/collect/calibrate/select/intervene 分阶段入口 |
| `tools/ref_sads_core.py` | token 分区、实际 Q/K 采集、分块统计、o_proj 前 head gate |
| `tools/ref_sads_stats.py` | 无标签校准、回退、选头与匹配随机对照 |
| `tools/ref_sads_analysis.py` | 仅在最后读取效果集 GT，输出逐样本指标和报告 |
| `tools/ref_sads_io.py` | 来源校验、阶段依赖、独占写入与中断恢复 |
| `tools/test_ref_sads*.py` | 合成数据与 CPU 测试模型；不加载真实 checkpoint |

原模型、原分类头、原评估脚本、历史结果及 object hidden-state 缓存均不改动。统计只来自新的完整前向；不读取旧 hidden-state 缓存反推 head。

## 固定协议与实现假设

保持 36 层、冻结权重、BF16、batch=1、原 prompt/预处理、100 个候选及原顺序；raw-logit 首个 argmax 决定预测。没有早退或 Top-3 重排。checkpoint、处理器、模型来源、候选、图片、代码及配置均被冻结并逐阶段校验。

论文的最大视觉注意力与非视觉注意力熵被用于两阶段分类：视觉注意力较高为 vision；较低者再按熵分为 sinkG（高熵）和 sinkS（低熵）。下列 WeDetect 适配明确属于 **SADS-inspired 实现假设**，不称为论文逐项复现：

- query 使用所有非 padding 行。主要视觉统计为 `max_key(mean_query(A))`，同时保存其他聚合诊断与 object-query 辅助统计，不据辅助结果重新选规则。
- G=全图图像 token，O=候选对象 token，T=普通文本，S=system 内容，R=模板结构 token，P=padding。V=G∪O，I=T∪S∪R；当前模板的 S 可以为空。实际 tokenizer 角色头解析失败会停止。
- 非视觉熵先对每个有效 query 在 I 上重归一化，再对 query 平均、计算自然对数熵 H；GMM 使用 `e=H/log(|I|)`，同时保留 H。因果 mask 后质量为零的行不伪造熵；有效 query 比例不足 0.99 则该统计无效。
- 在实际 RoPE 后的 Q/K 上重算所需注意力统计。FP32 softmax、双精度归约、query 分块；32 个输出 heads 与 8 个 KV heads 按 GQA 展开。只记录目标层统计，不保留全模型注意力矩阵。
- 两个 GMM 均用固定 5 次初始化，要求 BIC 增益≥10、每成分权重≥0.1、真实双峰及内部谷点，至少 200 个观测/20 张原始图。20 次 image bootstrap 至少 16 次有效，谷点 IQR≤原统计 IQR 的 0.1。bootstrap 重拟合两阶段；无稳定阈值则 unknown/no-op，不强行按分位数划分类别。
- 每个 sample×layer 从 sinkS 中均匀选一个 head，始终保留 h0（0-based 首个 shared head）。随机对照从该层其余 31 个 heads 均匀选择，种子 11/29/47，允许与 sinkS 重合。无可选 sinkS 时，所有匹配对照均 k=0。
- 只干预第 28/32/36 层（1-based），每次一个 head、一个层；gate=0 或 0.5，位置在拼接后的 attention 输出进入 o_proj 之前。全部 gate=1 也执行真实乘法，以检验接入误差。

使用历史 1,000 条 dev 中固定种子抽取的 100 条 calibration 与 200 条 evaluation，按图片分离。抽样不读取是否答错、候选覆盖率或 GT。已有 2,573 条 validation 不用于校准或选参数；此结果仍是开发集证据。

每条效果样本有 25 个逻辑结果：1 个无干预基线 + 3 层 × 2 个 gate ×（sinkS + 3 个随机种子）。重复的相同 head/gate 操作可复用物理前向，所有逻辑臂仍完整输出。k=0 引用基线。FA2 方案最多 5,146 次计划前向（含 6 次 warm-up、40 次一致性前向和 300 次统计前向）；eager 方案另采 FA2 参考基线，保守预算上限 5,392 次。预算不代表算子裁剪或实际加速。

## 顺序运行

以下是有真实 checkpoint 与图片的原 Linux GPU 环境上的命令示例。代码构建期间不执行这些命令。使用原 `wedetect_ref` 环境，生产入口严格检查历史 torch/transformers/torchvision/flash-attn 版本及 HF 源码 SHA，禁止自动下载权重。需要该环境已有 NumPy 和 Pillow；无需新增训练框架。

```bash
conda activate wedetect_ref
cd /media/data6/chengz/WeDetect
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
RUN=results/ref_sads_pilot_v1_20260924

# 只冻结和校验文件，0 次模型前向；同名目录已存在则拒绝覆盖。
python -B tools/ref_sads.py prepare --config config/ref_sads_pilot_v1.json --out "$RUN"

# 必须先通过：原路径 A、重复 A2、全 gate=1 的 B、带统计采集的 C。
python -B tools/ref_sads.py check --run "$RUN" --n 10

python -B tools/ref_sads.py collect --run "$RUN" --split calibration
python -B tools/ref_sads.py calibrate --run "$RUN"
python -B tools/ref_sads.py collect --run "$RUN" --split evaluation
python -B tools/ref_sads.py select --run "$RUN"
python -B tools/ref_sads.py intervene --run "$RUN"
python -B tools/ref_sads_analysis.py --run "$RUN"
```

若同一批权重或图片在其他路径，prepare 可显式传 `--checkpoint /actual/path/WeDetect-Ref-4B --images /actual/path/coco2014`；SHA 仍必须相同。保持默认 FA2 时仅目标层添加 hook，不切换后端。若确需 eager，必须用新的输出目录，在 prepare 时传 `--backend eager`；全部一致性/校准/干预基线都使用 eager，且 intervene 额外生成原 FA2 的 200 条无干预参考结果。禁止中途更换后端或把跨后端差值归因于干预。

check 从校准样本中按实际 token 长度取 10 条，逐候选比较原 logits（atol=rtol=1e-5）并要求 Top-1 一致，验证实际位置编码与输入不变、36 层均执行、分块统计与直接参考统计相符。失败不生成通过标记，后续阶段不能运行。check 根据实测普通/collector 耗时外推预算；超过固定 7,200 秒预算则停止，不能通过缩短层数、改精度、筛样本绕过。

建议预留单张 48 GB GPU、1–2 小时和约 2 GB 输出空间，最终以 check 的实测值为准；没有真实 GPU 测量前不保证时间或峰值显存。流程不自动租 GPU、提交远程任务或调参。

## 输出与分析口径

- `manifest.json`、`config.json`、`inputs.jsonl`：来源与无标签输入。`eval_targets.jsonl` 只含 200 条效果集 GT；校准和选头不读取它的内容。
- `checks/consistency.json`：逐样本 A/A2/B/C 误差、Top-1、一致性噪声阈值 eps 和预算判断。
- `stats/*_heads.jsonl`：所有选定层的逐样本逐 head 统计。`calibration.json` 保留各 GMM、稳定性和回退原因；`selections.jsonl` 保留完整分类、连续分数、实际选头及随机 heads。
- `predictions.jsonl`：所有逻辑臂的原始候选 logits、预测框/索引、head/gate/seed、物理前向引用；`eval_predictions.jsonl` 加入事后 IoU、损失、修复/破坏标签。
- `resources.jsonl`：每个已记录物理前向的 CUDA 时间、forward wall、从输入准备到 logits 校验结束的 request wall、输入审计开销、峰值 allocated/reserved 显存。collector 与普通前向分开；重复逻辑引用不重复计时。
- `timings/*.json`：各 CLI 阶段实际墙钟与成功状态；包含加载/校验/落盘等开销，不能与 forward 时间再相加。硬中断可能使最后一次未落盘测量丢失；恢复报告不能声称这些测量完整。
- `metrics.json`、`report.md`：Top-1、F、H、F−H、任务损失变化、随机种子分布、类别比较、连续分数预测性、按图片配对 bootstrap 区间及资源汇总。

Top-1 沿用 IoU≥0.5。任务损失沿用原候选平均 sigmoid focal loss（alpha=0.25、gamma=2，目标为 IoU>0.5 时的 soft IoU，否则 0）。损失定义和 Top-1 边界有意保留原口径。B=基线损失−干预损失，B>eps 记为有害 head；eps 只从无标签一致性误差得到并冻结。

主检验 gate=0：C 为 sinkS 有害频率减三个随机种子均值；D 为 sinkS 的 Δloss 减随机均值。只用随机单头臂检查固定 SADS 连续分数能否预测 B（AUROC/Spearman），避免仅在已按分数选择的 sinkS 上循环论证。同时报告全样本与 eligible 子集、逐层与表达内等权跨层结果；不选最佳随机 seed。无双峰、无可估区间、阴性结果都按实际结果输出，工程 PASS 不表示科学假设成立。

## 恢复与离线验证

每一步校验父阶段标记及文件 SHA，修改代码/配置/权重后必须新建 run。未完成阶段需先检查 `failures/`，再对同一命令加 `--resume`（报告入口也支持）；只复用 manifest 与内容完整校验通过的记录。并发写同阶段会被内核锁拒绝，进程死亡会释放锁。已完成阶段拒绝覆盖；损坏或不同内容的部分文件也不自动删除。prepare 的不完整目录保留，重新 prepare 使用新目录。

```bash
# CPU 合成测试；不需要图片或 checkpoint，不执行真实实验。
python -B tools/test_ref_sads.py

# 示例：仅在该阶段曾中断且来源不变时使用。
python -B tools/ref_sads.py intervene --run "$RUN" --resume
python -B tools/ref_sads_analysis.py --run "$RUN" --resume
```

测试覆盖 token 分区、32Q/8KV 分组注意力、chunk/reference 统计、head 维度与 gate=1/0/0.5、hook 清理、GMM 失败回退、GT 隔离、匹配随机矩阵、指标边界、真实文件阶段链、破损记录拒绝与中断恢复。CPU 测试使用的 HF 版本可与生产不同；生产 RoPE/attention 源码检查不会因此放宽，真实 FA2/BF16 等价性仍由 GPU 上的 check 决定。

2026-09-24 构建验证：58 项 unittest 全部通过（33.961 秒），另有上述 core 的 3 组数值检查通过。验证环境为 Python 3.12.14、torch 2.13.0+cpu、transformers 4.57.6、NumPy 2.5.1；这是现有本地 CPU 环境，不是生产依赖升级。9 个 Python 文件语法检查通过，原模型/预处理/评估相关的 6 个来源文件 LF 标准化 SHA 与历史签名一致。未进行真实 checkpoint 或 GPU 前向，没有训练或实验效果结论。
