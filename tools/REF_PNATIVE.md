# P-native：冻结原分类头的逐层读取

依赖已通过的 E0，默认 RefCOCOg validation。只验证原头在不同深度能读出什么；
不训练 probe、不改主干、不删除 tokens、不进行真实提前退出。原 E0 文件与结果不修改。

## 固定协议

- 作者提供的候选框前 100 个、原顺序；不插入 GT、不乘 objectness。
- 复用 E0 的 prompt、图像处理、BF16、FlashAttention2、cache 设置。
- Top-1，无分数过滤、无额外 NMS；IoU≥0.5 为正确。
- 主排名沿用 BF16 sigmoid → FP32，分数相同时取候选原顺序中第一个。
- 每层使用一次原 final RMSNorm 与同一个原分类头，全部冻结。

实际 L 从模型读取，位置为 `{0, ceil(L/4), ceil(L/2), ceil(2L/3), ceil(5L/6), L}`。
已核对的 36 层模型对应 **0、9、18、24、30、36**。
深度 k<L 从零基第 k 个 block 的输入读取，已经完成前 k 个 block 及 DeepStack；
L 从 final RMSNorm 输入读取。h0 已包含视觉/ROI/位置投影成本，不是无图像特征。

每条表达两次完整前向：一次无 hook 参考、一次同时采集全部选定深度。
逐条检查 hook 不改输出、末层完整/紧凑重建一致、全部 decoder 恰好执行一次。
与 E0 重叠的样本检查输入、图像哈希、logits、分数、Top-1；首对同图 query 检查 h0。
只要一致性失败即停止并保存 FAILED.json，不能把失败当成性能下降。
浅层准确率低不是实现失败，只有末层与基线一致性属于停止条件。

## 文件与依赖

新增并同步到服务器 tools/：

- `ref_pnative.py`
- `ref_pnative_core.py`
- `ref_pnative_analysis.py`
- `test_ref_pnative.py`
- `REF_PNATIVE.md`

还需此前已有的 `ref_e0.py`、`ref_e0_core.py`、`ref_e0_data.py`、
`humanref_pipeline.py` 和 `test_ref_e0.py`。不需要重新安装服务器依赖。
读取本地真实权重，不下载其他模型。单卡普通 python，不使用多进程 torchrun。

## 运行顺序

```bash
cd /media/data6/chengz/WeDetect
conda activate wedetect_ref
python -B tools/test_ref_pnative.py --transformers
```

6 条 smoke（日志在输出目录外；输出目录不能预先创建）：

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 P_N=6 P_OUT=results/ref_pnative_refcocog_smoke6 \
  python -u -B tools/ref_pnative.py 2>&1 | tee ref_pnative_refcocog_smoke6.log
```

通过后跑与 E0 相同的 60 条：

```bash
CUDA_VISIBLE_DEVICES=0 P_N=60 P_OUT=results/ref_pnative_refcocog_val60 \
  python -u -B tools/ref_pnative.py 2>&1 | tee ref_pnative_refcocog_val60.log
```

确认末层一致性后，扩大到固定 500 条 validation 表达。采样与 E0 相同种子/算法，
保持前 60 条不变，不按任何方法结果挑选或剔除样本：

```bash
CUDA_VISIBLE_DEVICES=0 P_N=500 P_OUT=results/ref_pnative_refcocog_val500 \
  python -u -B tools/ref_pnative.py 2>&1 | tee ref_pnative_refcocog_val500.log
```

路径环境变量（相对路径以仓库根目录为工作目录）：

```text
P_ANN        wedetect_ref/eval_grounding/eval_refcoco/refcocog_validation.json
P_PROPOSALS  wedetect_ref/eval_grounding/eval_refcoco/refcoco_proposals_all.json
P_IMAGES     data/coco2014
P_REF        checkpoints/WeDetect-Ref-4B
P_E0         results/ref_e0_refcocog_val60
```

图像根目录与注释 image 字符串拼成实际文件路径，规则同 E0。
如改变了数据集或权重，须先运行相应 E0 并设置 P_E0，不能跳过协议检查。
会核查 E0 数据/权重/核心源码/关键版本与数值设置；不同路径可以，内容不同不可以。
输出独占创建、不覆盖、不自动续跑；重试用新 P_OUT。

## 输出

| 文件 | 内容 |
|---|---|
| `summary.md` / `summary.json` | 所有一致性检查通过后生成；逐层结果 |
| `curve.csv` | 深度、准确率、差值、索引差异、伤害/恢复及条件分母 |
| `paired_errors.json` | 每层“基线对→该层错”和反方向样本 ID |
| `query_pairs.json` | 同图不同 query、GT 相互 IoU<0.5 的配对检查 |
| `samples/00000.json` | 全部候选、GT、每层 logits/分数/决策、数值检查 |
| `samples/00000.pt` | 每个图像-query 对的特征及对齐信息 |
| `native_head.pt` | 原 Norm 参数、epsilon 和原 head 权重，用于离线复核 |
| `manifest.json` / `selection.json` | 版本、协议、权重/数据/源码哈希、固定样本 |
| `FAILED.json` | 实现一致性失败或输入问题；失败运行不用于曲线结论 |

特征形状 `[depth, candidate, hidden]`，每份缓存明确保存 `depths`；本模型通常
`[6,100,2560]`，BF16，final Norm 之前的状态。还保存候选坐标与顺序哈希、GT、
input/position IDs、object 位置、BF16 模型框输入、原输出、每层原头 logits。
每份张量缓存还包含 manifest、共享 head 和对应逐样本 JSON 的 SHA256，便于离线检查来源。
`max_gt_iou` 与 `iou_soft_labels` 仅供以后核查；软标签为 IoU>0.5 时取 IoU，否则 0，
与评价的 IoU≥0.5 区别明确。无 GT 候选补入，本阶段不消费这些标签训练。
这些是 validation 缓存，**以后不能直接拿去当训练集**。

特征主体约 3.072 MB/表达，60 条约 184 MB，500 条约 1.536 GB，1000 条约 3.072 GB。
不保存所有 token 的逐层状态或 KV；全序列末层只临时用于 E0 式重建检查。
可信的自生成张量文件可用 `torch.load(..., weights_only=True)` 读取。

## 如何解释指标

- `accuracy`：主协议 Top-1 正确率；不是与基线的一致率。
- `index_disagreement_rate`：选的候选不同；两个框可能都正确。
- `harmed_count`：基线正确、该层错误；`recovered_count`：反方向。
- `harm_rate_all` 分母是全部表达，`harm_rate_given_baseline_correct` 分母仅为基线正确数。
- 保证 `该层正确数 - 基线正确数 = recovered - harmed`。
- `accuracy_given_covered` 分母为候选覆盖 GT 的表达数；无分母返回 null，不伪造 0。
- `raw_logit_accuracy`、并列数与饱和计数是附加数值诊断。浅层 BF16 sigmoid 可能
  将不同 logits 压成相同分数；主曲线不静默换成 raw-logit 排名来改善结果。
- query pair 同时看“选择随 query 改变”和“两条都正确”；前者单独不证明正确理解。
  GT IoU 筛选仅用于离线诊断，不影响输入候选；配对之间相关，不能按独立样本估计显著性。

P-native 仍完整执行所有层，**不报告层退出速度**。原头在浅层表现差，不等于该层
没有可学习的信息；下一阶段 P-linear 需独立训练数据，并包含最后层同规格重训头。
V 仍要另做 split/resume 一致性检查，这里并未实现或验证它。

## 本地验证记录

2026-09-20：标准库统计/报表测试、真实 Transformers 4.57.1 小型随机 Qwen3-VL
在 CPU FP32/BF16 下逐层重放与采集对照、异常清理及张量缓存存取测试通过。
原 E0 与 HumanRef 回归测试通过，E0 三个核心脚本哈希仍与服务器成功运行记录一致。
本机没有真实权重与 CUDA，尚未验证生产模型上的 P-native 前向，以服务器 smoke 为准。
