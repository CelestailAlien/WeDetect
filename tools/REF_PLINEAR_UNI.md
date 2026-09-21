# 固定1663图的 Uni 候选提取 → P-linear

本文件是 audit_v1 之后的操作入口，优先于旧文档要求“补齐整个训练源”的步骤。
不再为21,899张train图全部提取；只对冻结的5000条train、1000条dev涉及的1663图统一重提取。
现成131图的候选不混入新文件；作者2573条validation及其候选完全不变。

## 同步代码

新增至服务器 `tools/`：

- `ref_plinear_inputs.py`
- `ref_plinear_uni_core.py`
- `ref_plinear_proposals.py`
- `test_ref_plinear_proposals.py`
- 本说明 `REF_PLINEAR_UNI.md`

更新：`ref_plinear.py`、`test_ref_plinear.py`；其余旧工具继续保留。
`generate_proposal.py` 和 `vis.py` 使用仓库原文件，未修改。
标注转换器 `ref_plinear_prepare.py` 未修改，不需要重新转换，不覆盖旧audit目录。

## 本轮生成协议

- Uni **Base**，默认 `checkpoints/wedetect_base_uni.pth`；不根据文件名猜模型结构。
- 单GPU、单图batch，FP32，无autocast、TF32关闭；不初始化分布式。
- 原 `generate_proposal.SimpleYOLOWorldDetector`，256个学习到的prompt，维度768。
- 原640×640 RGB letterbox、双线性缩放、114填充、除255、原坐标还原及裁剪。
- 原分数阈值0，pre-NMS top30000，**按学习到的prompt标签分组**的batched NMS，IoU=.7。
- 保留原NMS输出前100；不追加class-agnostic NMS，不额外重排并列分数。
- 少于100框时按实际数量记录，0框报错；不插GT，不重复框凑数，不换样本。
- 记录checkpoint、源码、依赖、GPU型号、主机、逐图字节哈希以及冻结划分哈希。
- 不声明该生成协议就是作者未公开的训练候选协议。这是本轮固定训练输入；所有探针层使用相同候选。

生成器后处理常量还会通过源码结构检查，避免本地被修改后仍误报旧NMS参数。
只有原分类backbone尾部4项可为unexpected keys；任何missing/mismatched key均阻断。

## 1. 无GPU测试

```bash
cd /media/data6/chengz/WeDetect
conda activate wedetect_ref
python -B tools/test_ref_plinear_proposals.py
python -B tools/test_ref_plinear.py
```

测试覆盖随机张量、原分数过滤函数、权重键映射、冻结划分、小样本截取、缺候选拒绝、
损坏文件拒绝、恢复记录验证及原P-linear合成训练链路。
若真实转换数据/audit存在，也会核对真实1663图清单；这不代表已经运行Uni模型。

## 2. 先提取6张图验证Uni工程链路

选择已分配的空闲GPU；若调度系统设置了CUDA_VISIBLE_DEVICES，请沿用。
下例假定可以自行使用GPU0。模型仅为Uni，不会同时加载4B Ref。

```bash
export CUDA_VISIBLE_DEVICES=0
export UNI_IMAGES=/media/data6/chengz/WeDetect/data/coco2014
export UNI_CHECKPOINT=checkpoints/wedetect_base_uni.pth
set -o pipefail
UNI_N=6 UNI_OUT=results/ref_plinear_uni_k100_smoke6 \
python -u -B tools/ref_plinear_proposals.py 2>&1 | tee ref_plinear_uni_k100_smoke6.log
```

默认读取：

```text
data/refcocog_plinear_umd_v1/{conversion,refcocog_train,refcocog_validation_first,image_inventory}.json
results/ref_plinear_data_audit_v1/planned_selection.json
```

`UNI_DATA`、`UNI_SELECTION` 可改路径；完整输入内容仍需通过校验。
6张图是冻结图像清单按文件名排序的前6张，不用结果挑图。
**这个输出不能直接用于P-linear训练**，因为未覆盖完整固定划分。

## 3. 全部1663图

确认上一条命令退出码0，`summary.json` 为PASSED后运行：

```bash
UNI_N=0 UNI_OUT=results/ref_plinear_uni_k100_v1 \
python -u -B tools/ref_plinear_proposals.py 2>&1 | tee ref_plinear_uni_k100_v1.log
```

输出：

```text
results/ref_plinear_uni_k100_v1/
  manifest.json
  loading_info.json
  samples/00000.json ...              # 每张图完成即保存
  train_proposals.json                # {image: [xyxy框列表, Uni分数列表]}
  summary.json                       # 数量、train/dev候选覆盖率
  sample_hashes.json
  COMPLETE.json                      # 完整结束后才发布
```

GT仅用于提取结束后的覆盖率诊断，坐标裁剪与Ref输入一致；覆盖率低也不会补GT或删除表达。
Uni分数用于原候选排序并保存，不乘入Ref分数。

中断恢复（同GPU型号、同节点、同权重/源码/依赖/图片/选择）：

```bash
UNI_RESUME=1 UNI_N=0 UNI_OUT=results/ref_plinear_uni_k100_v1 \
python -u -B tools/ref_plinear_proposals.py 2>&1 | tee ref_plinear_uni_k100_v1_resume.log
```

恢复时会逐条校验已有记录；不覆盖它们。若换GPU型号/节点，使用新UNI_OUT从头提取，
不把旧卡结果拼接进来。原NMS同分排序不承诺跨硬件逐位一致。
中断恰好损坏某条JSON时会明确报错，不自动删除/修复；保留诊断证据后开新目录最安全。
完成的全量目录也可验证式重跑，最终产物必须与已有内容一致。

## 4. P-linear smoke：固定划分内截取，先检查候选来源

新版预检先验证**完整训练标注**与val不重叠、固定划分与audit一致，
然后只要求所选图像有候选。不会因为候选不足重新抽样或改train/dev归属。

```bash
export PL_TRAIN_ANN=data/refcocog_plinear_umd_v1/refcocog_train.json
export PL_TRAIN_PROPOSALS=results/ref_plinear_uni_k100_v1/train_proposals.json
export PL_UNI_RUN=results/ref_plinear_uni_k100_v1
export PL_SELECTION=results/ref_plinear_data_audit_v1/planned_selection.json
export PL_IMAGES=/media/data6/chengz/WeDetect/data/coco2014
export PL_OUT=results/ref_plinear_refcocog_smoke_uni_v1
export PL_TRAIN_N=16
export PL_DEV_N=8
export PL_VAL_N=6
PL_STAGE=preflight python -u -B tools/ref_plinear.py
```

必须保留上述 `PL_SELECTION`：16条train和8条dev分别取固定5000/1000划分的前缀，
不是用dev=8重新划分。`PL_UNI_RUN` 校验COMPLETE、全量标记、候选哈希和选择/标注来源。
原作者 `PL_VAL_ANN`、`PL_VAL_PROPOSALS` 维持既有默认值，**不要指向新生成文件**。

只有preflight通过再启动Ref：

```bash
PL_STAGE=cache python -u -B tools/ref_plinear.py 2>&1 | tee "$PL_OUT/cache_fit.log"
python -u -B tools/ref_plinear_train.py 2>&1 | tee "$PL_OUT/train.log"
PL_STAGE=cache_val python -u -B tools/ref_plinear.py 2>&1 | tee "$PL_OUT/cache_validation.log"
python -u -B tools/ref_plinear_eval.py 2>&1 | tee "$PL_OUT/evaluation.log"
```

Uni进程结束后GPU显存已释放，再启动Ref；不要并行占用同一张卡。
每条命令成功再执行下一条，smoke不作性能结论。

## 5. 正式训练

沿用上面的数据来源与锁定划分环境变量，换新输出目录：

```bash
export PL_OUT=results/ref_plinear_refcocog_uni_v1
unset PL_TRAIN_N PL_DEV_N PL_VAL_N
PL_STAGE=preflight python -u -B tools/ref_plinear.py
```

再执行相同cache/train/cache_val/eval四步。此时为5000train、1000dev、原2573validation。
检查 `plan.json` 的计数与 `uni_provenance` 后再继续。修改了预检源码，所以旧preflight/cache计划
不能与新版源码混用；保留旧目录，不覆盖或手工改hash。

本轮没有修改任何旧E0/P-native/full-eval脚本或作者模型，也没有改变旧实验结果。
