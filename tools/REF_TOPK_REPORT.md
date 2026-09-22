# P-linear 候选排序诊断（只读，不重新训练）

目标：检查浅深共同错误中，合格候选是否仍位于第 2/3 名，并区分候选缺失、
短名单不足、几何阈值问题。**高 Top-k 只说明 oracle 重排空间，不证明消歧模块能学会选择。**

## 固定口径

- 第 24 层：三个已在 dev 选定的线性头全部报告，不挑 validation 最好 seed。
- 第 30 层：原头 BF16 **raw logits**，不使用 sigmoid 排序。
- 第 36 层：原完整模型 BF16 raw logits，不是重新训练的第 36 层头。
- 与现有评估严格一致：IoU **>= 0.5**；不是严格 `> 0.5`。
  使用缓存中的 FP32 IoU 决定成功，另用框坐标核验几何一致性。
- k=1/2/3/5/10/全部；k 大于候选数时使用全部候选。并列按原候选索引稳定排序，
  首名与现有 argmax 一致，输出 Top-1 及截断边界并列计数。
- 每个浅层头各自与原 36 层组成四组：both_correct / deep_only / shallow_only / both_wrong。
  分组只依据原始 Top-1，去重后不重新分组。空组比例为 null/NA，不伪造为 0。
- 同时保留 raw 与保守去重诊断：按分数贪心，和已保留框 IoU >= 0.9 时抑制。
  不使用 GT 决定保留框，不改变正式评估。高重叠不等于同一语义对象；同时报告去重
  导致全部合格候选丢失、以及合格框被不合格框抑制的风险。
- 错误分两条轴，不能把两轴计数相加：
  - 合格候选位置：rank2-5 / rank6-10 / outside_top10 / candidate_miss。
  - 选中框几何：IoU [0.45,0.5) / [0.1,0.45) / <0.1。
  这些只是诊断分箱，不是语义标签或调参。IoU=0.49 不足以认定同一对象，低 IoU 也不能
  自动认定语义对象选错。语义错误、同对象定位、表达/标注歧义都留给人工复核。

## 运行

同步新增的 `ref_topk_core.py`、`ref_topk_report.py`、`test_ref_topk.py` 到服务器 `tools/`。
不改已有提取/训练/评测脚本，不覆盖任何已有结果。使用已有 wedetect_ref 环境：

```bash
cd /media/data6/chengz/WeDetect
conda activate wedetect_ref
python -B tools/test_ref_topk.py

set -o pipefail
CUDA_VISIBLE_DEVICES=0 \
PL_OUT=results/ref_plinear_refcocog_uni_full_20260921_183958_868480 \
python -u -B tools/ref_topk_report.py \
2>&1 | tee ref_topk_diagnostics_v1.log
```

默认 GPU 仅重算小线性头，不加载 WeDetect/Ref 大模型，不重新提取特征、不训练。
按原评测 batch=16 流式读取 validation 的 `.pt`，不读取训练特征。
缓存有多个 GB，耗时可能主要在读盘。无需图像文件、无需下载 `.pt` 到本地。

没空闲 GPU 可添加 `PL_DEVICE=cpu`，统计核心本来就在 CPU。
CPU/换卡可能改变极接近的浮点分数；脚本逐表达核对所有报告头的 Top-1 索引、
正确性、IoU、并列数与原 `evaluation/predictions.json`。
若不一致，写 `CONSISTENCY_FAILURE.json` 并停止，不悄悄重分组或放宽门槛。
保留失败目录，用原评测设备和新的输出目录复查，例如
`DIAG_OUT=results/ref_plinear_refcocog_uni_full_20260921_183958_868480/topk_diagnostics_v2`。

注意：线性头的完整分数是在原冻结特征上重建的，而不是原 evaluation 保存的完整分数；
原 evaluation 只保存 Top-1。来源、设备、版本、输入哈希均写入报告。

## 只下载小报告目录

默认输出 `PL_OUT/topk_diagnostics_v1/`：

- `summary.md`：总体和四组 Top-k，**共同错误第一合格框名次直方图**，几何/重复框计数。
- `summary.json`：精确分子/分母、条件比例、各组直方图、并列、去重风险与来源。
- `per_expression.jsonl`：所有表达的分组、首个合格名次、Top-10 索引/分数/IoU、几何标志。
- `review_cases.jsonl`：按组/名次/近阈值/重复框风险确定性分层抽取，附 query、GT、
  原图路径和相关候选坐标（原图像素 xyxy，候选索引从 0 起，名次从 1 起）。
  包含即使在 Top-10 之外的第一个合格候选，便于检查。
- `review.csv`：人工复核清单，语义/同对象定位/歧义/不确定字段初始为空。
  每张表达可能涉及多个头，填写 arm_under_review；需要时复制该行记录另一个头。
  这是富集诊断样本，不能用其中错误占比直接估计整个数据集的错误占比。
- `COMPLETE.json`：只有全部校验通过才生成，并包含导出文件的哈希。

人工复核需要看原图，JSON 只是坐标与文本，不包含图像。先将小报告下载回来，
再按清单选择性获取图像即可，不必搬整套特征缓存。

当前统计来自已用于方法开发的 validation，不能据此宣称独立测试集结果，
也不会自动选择 NMS、IoU 阈值或训练新机制。
