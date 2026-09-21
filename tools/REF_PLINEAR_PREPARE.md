# P-linear 前置：RefCOCOg UMD 标注转换与服务器输入核查

**已完成audit_v1并发现训练候选不足后，转到 `REF_PLINEAR_UNI.md`。**
旧audit仍按全训练源报告NEEDS_INPUTS，保留用于诊断；新版P-linear只要求冻结抽样涉及的候选，
不再需要将旧audit刷成全训练源READY。下文描述旧audit的原有行为。

本步骤不训练、不使用GPU、不改旧评测文件，也不生成候选框。
新增 `ref_plinear_prepare.py`、`test_ref_plinear_prepare.py` 和本说明。
依赖此前的 P-linear/E0 公共工具，只有 Python 标准库；无需安装新库。

## 固定数据规则

- 只读取 `data/refcocog/refs(umd).p` 与 `instances.json`，不使用Google划分。
- UMD train展开全部表达：42,226个目标、80,512条表达、21,899张图。
- UMD val每个目标取第一条表达：2,573条、1,300张图，与此前作者评测协议一致。
- test保留不输出训练标注。原始test的9,602条表达不用于抽样、选择checkpoint或训练。
- `sentence.sent` 为处理后表达，不替换为 `raw`，不额外改大小写或标点。
- 用 `image_id` 查 `instances.images.file_name`，不使用带目标编号的 `ref.file_name`。
- 图片字段固定为 `train2014/COCO_train2014_XXXXXXXXXXXX.jpg`。
- GT从COCO `xywh` 转为 `xyxy`；不预先裁剪，沿用后续原推理流程裁剪。
- train的ID为 `refcocog_train_{原始sent_id}`，稳定可追溯；另保留ref_id/ann_id/sent_id。
- 转换后的 `conversations` 是当前探针加载器的存储格式，不代表执行Stage1/2训练。

转换时逐条核对先前完整验证结果的ID、顺序、图片、表达及GT框。
还校验旧 `samples/*.json` 与 `sample_hashes.json`；不修改这些结果。
pickle只允许原始数据操作码，拒绝GLOBAL/REDUCE等可执行构造操作；不直接信任任意pickle对象。

## 路径匹配

服务器图片路径已确认应为：

```text
/media/data6/chengz/WeDetect/data/coco2014/train2014/COCO_train2014_000000581921.jpg
```

所以 `PREP_IMAGES` 和后续 `PL_IMAGES` 都应指向 **data/coco2014**，
不是它下面的train2014目录；否则会重复拼接 `train2014/train2014`。
某个最大编号图片存在不能证明其他所需图片齐全，audit会逐张检查。

## 先测试，然后转换

同步新增三个文件及上一轮P-linear工具到服务器。如果直接同步已经转换好的目录，
须保证转换脚本字节不变（勿改变换行符），否则来源哈希会拒绝；最简单是服务器重新转换一次。

```bash
cd /media/data6/chengz/WeDetect
conda activate wedetect_ref
python -B tools/test_ref_plinear_prepare.py
PREP_STAGE=convert python -u -B tools/ref_plinear_prepare.py
```

默认需要：

```text
data/refcocog/refs(umd).p
data/refcocog/instances.json
results/ref_full_d30_refcocog_validation/manifest.json
results/ref_full_d30_refcocog_validation/summary.json
results/ref_full_d30_refcocog_validation/sample_hashes.json
results/ref_full_d30_refcocog_validation/samples/*.json
```

默认生成全新的目录：

```text
data/refcocog_plinear_umd_v1/
  refcocog_train.json                # 80,512条，PL_TRAIN_ANN
  refcocog_validation_first.json     # 2,573条，仅供核对
  image_inventory.json              # 所需train/val图片和既有val图片哈希
  conversion.json                   # 来源、转换规则、产物哈希及核对结果
```

**不要将 `refcocog_validation_first.json` 替换作者原 validation.json。**
它虽然语义逐条一致，JSON字节/附加字段不同，不能绕过旧baseline文件哈希约束。
后续P-linear仍使用原先的作者验证文件。

若目录已经存在，脚本拒绝覆盖。确认其中conversion是PASSED后可直接audit；
确需重跑，设置新的 `PREP_OUT`，随后所有audit都沿用该路径，不删除旧结果。

## 服务器audit：检查图片及现有训练候选覆盖率

```bash
export PREP_IMAGES=/media/data6/chengz/WeDetect/data/coco2014
export PREP_PROPOSALS=wedetect_ref/eval_grounding/eval_refcoco/refcoco_proposals_all.json
export PREP_AUDIT_OUT=results/ref_plinear_data_audit_v1
set -o pipefail
PREP_STAGE=audit python -u -B tools/ref_plinear_prepare.py
```

默认检查全部21,899张train图和1,300张val图：路径存在且文件非空，
val图片SHA还必须与旧评测一致。**不在此阶段解码每张train JPEG**，坏图解码仍由后续提取直接报错。

候选核查支持原文件的裸框列表和 `[boxes, scores]` 两种格式；
允许将 `COCO_train2014_...jpg`、纯数字COCO文件名统一映射到 `train2014/...jpg`。
同图出现多个别名且候选内容不一致时直接报错，不擅自选一个。
只统一键名，不改变框、顺序、scores、不截断、不乘objectness，不插入GT。
报告还列出少于/等于/多于100框的图片数量，之后仍由原P-linear取前100。

输出：

```text
results/ref_plinear_data_audit_v1/
  summary.md
  summary.json
  missing_images.json
  changed_validation_images.json
  missing_train_proposals.json
  proposal_coverage.json
  planned_selection.json           # 固定seed的5000train+1000dev预览
  train_proposals.json             # 仅全量训练候选覆盖时才生成
```

`summary.json` 的状态：

- `READY`：所需文件及结构检查通过，可以做P-linear preflight。
- `NEEDS_INPUTS`：缺图、验证图字节变化或缺候选；报告保存后以退出码2结束，这是有意的阻断。

`READY` 不证明候选生成器来源；普通JSON无法推断它来自哪套Uni权重或NMS。
复用时应确认是原作者候选或另行固定并记录生成协议，不能将GT框伪装成候选。

现有P-linear加载器先检查整个train标注涉及的候选，因此本版audit的READY要求全训练源覆盖。
同时报告最终5000+1000条表达涉及的候选覆盖数，便于决定下一步提取范围。
**缺候选时先看报告，不直接删掉缺候选的表达，也不要马上启动全量GPU提取。**
后续可根据真实缺口确定按已冻结选择提取的方案；本轮不混用未经审核的新Uni生成设置。

## 只有READY后才继续

```bash
export PL_TRAIN_ANN=data/refcocog_plinear_umd_v1/refcocog_train.json
export PL_TRAIN_PROPOSALS=results/ref_plinear_data_audit_v1/train_proposals.json
export PL_IMAGES=/media/data6/chengz/WeDetect/data/coco2014
export PL_OUT=results/ref_plinear_refcocog_smoke
export PL_TRAIN_N=16
export PL_DEV_N=8
export PL_VAL_N=6
PL_STAGE=preflight python -u -B tools/ref_plinear.py
```

通过后再按 `REF_PLINEAR.md` 执行cache/train/cache_val/eval。此处先不占GPU。
如果audit是NEEDS_INPUTS，请先提供 `summary.json`，据此补图片或构建固定Uni候选提取。
