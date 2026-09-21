# P-linear：冻结 Ref，只训练各层共享线性读出头

本阶段回答：**第9/18/24/30层是否已包含可以被线性读出的指代表达信息？**
不训练主干、不修改作者模型文件、不做 LoRA/MLP/路由器、不重新调 HumanRef 阈值。
沿用已验证的 object-token 边界、原 final RMSNorm、固定候选与 prompt。

## 固定实验对照

| 对照 | 深度 | 更新参数 | 用途 |
|---|---|---|---|
| 原始完整模型 | 36 | 无 | 真正的现有 baseline；始终保留 |
| P-native BF16 | 9、18、24、30、36 | 无 | 复核原头直接读出的表现；同时报 raw/sigmoid |
| P-native FP32 | 相同五层 | 无，只将头转 FP32 | 隔离头精度变化，不将消除饱和并列算成训练收益 |
| P-linear FP32 | 相同五层 | 每层独立 `w[2560], b` | 浅层可读性；新36层头是同监督、同训练预算对照 |

五个头都是跨候选共享的二分类线性头，**不是为100个候选位置分别建头**。
每个头只有2561个参数。36层原头已经有了，不需要覆盖或重训它；新36层头作为另一个实验分支。
若新36层头比原头差，不能用弱化的新头替代原 baseline 来夸大浅层优势。

特征为 `original_frozen_final_RMSNorm(h_depth)`，BF16保存，读出头FP32。
Norm的权重、epsilon以及BF16计算顺序不变；没有可学习Norm。
使用原代码的 sigmoid focal loss（alpha=.25, gamma=2），标签 `IoU > .5 ? IoU : 0`。
评测正确标准仍为 `IoU >= .5`，两者边界有意不同，沿用作者实现。
先在每条表达内部对有效候选取均值，再对表达取均值；padding不参与loss。
无候选覆盖GT的表达仍保留、全部label为0，不能当作真正的无目标指令。

主比较采用 **FP32原始logit Top-1**，并列取原候选最前位置；无分数阈值、无额外NMS。
原BF16 full36 的 raw-logit 和 sigmoid 两个历史口径都单列报告，不能跨口径混报提升。
compact原头与full-shape原头可能有BF16 GEMM舍入差异，逐条存储差异与赢家变化；
原完整模型baseline始终取原full-shape输出，不用compact36替代它。

## 必需数据：先确认，不能拿 validation 顶替

本地当前只有验证结果，**未确认服务器是否有 RefCOCOg 训练标注和训练图像的固定候选**。
不会自动把已有验证缓存变为训练缓存，也不会因为候选缺失补GT。

需要：

1. 真正的 RefCOCOg 训练标注，处理成仓库的REC JSON列表格式：

   ```json
   [{"id":"refcocog_train_0","image":"COCO_train2014_000000000123.jpg",
     "conversations":[{"from":"human","value":"<image>"},{"from":"gpt","value":"the person on the left"}],
     "bounding_boxes":[[10,20,100,200]]}]
   ```

   上述仅展示格式，不是真实样本。query从 `conversations[1].value` 读取；
   框须为像素坐标xyxy，单GT。原始REFER pickle、COCO xywh、stage3 `class_name` 格式不能直接传入。
2. 训练图像候选JSON，以相同image字符串为键，值为xyxy框列表或 `[boxes, objectness]`。
   至少覆盖输入训练标注涉及的所有图像。采用文件中前100框及其顺序，不乘objectness。
   `refcoco_proposals_all.json` 的 `all` **不能证明它覆盖train**，脚本会真实检查缺失键。
   若缺少训练候选，需要单独准备：固定Uni checkpoint/输入/NMS/top-K提取；不得由GT生成或补入。
   本轮代码不包含未验证的新训练集下载器、REFER转换器或Uni候选生成器。
3. 训练/验证图像，以及已有作者验证标注、验证候选、Ref checkpoint。
4. 已通过的 `results/ref_full_d30_refcocog_validation/{manifest,summary}.json`，用于冻结来源。

默认数据目录、模型路径沿用完整验证脚本；可用 `PL_IMAGES`、`PL_REF`、
`PL_VAL_ANN`、`PL_VAL_PROPOSALS`、`PL_REFERENCE` 覆盖路径，内容仍校验。

划分规则：从**训练标注**随机排列图像组，先保留1000条dev，再取5000条train。
最后一个图像组可截断表达数，未选表达丢弃，绝不会放进另一划分。
同一图像所有表达只能属于一个划分，COCO文件名别名及图像字节哈希也检查。
整个训练源必须与作者验证集按图像不相交；不按GT难度或候选覆盖率抽样。
图像与表达列表、数据/候选/源码/模型哈希保存在plan和manifest中。
原 validation 已参与之前的方法探索，只能称验证集，最终论文仍需未参与选择的test。

## 文件与环境

同步新增七个文件至服务器 `tools/`：

- `ref_plinear.py`、`ref_plinear_core.py`、`ref_plinear_data.py`
- `ref_plinear_train.py`、`ref_plinear_eval.py`
- `test_ref_plinear.py`、`REF_PLINEAR.md`

复用现有E0/P-native工具和目标模型，不修改旧实验文件，不增加第三方依赖。
提取特征用原 `wedetect_ref` 环境、BF16、FlashAttention2、Transformers4.57.1；单GPU运行。
离线训练只需要PyTorch，可用GPU或 `PL_DEVICE=cpu`，不加载4B模型。
固定CPU线程4，GPU读出禁用TF32；训练使用确定性算法及CUBLAS工作区设置。

先运行CPU测试，无需数据和模型下载：

```bash
cd /media/data6/chengz/WeDetect
conda activate wedetect_ref
python -B tools/ref_plinear_core.py
python -B tools/test_ref_plinear.py
```

测试包含：官方focal函数数值对齐、维度/冻结梯度、候选置换、padding屏蔽、可实现信号拟合、
图像泄漏拒绝、损坏缓存拒绝、合成特征真实train→dev选头→eval流程、不覆盖旧输出。
**合成数据只用于单元测试，不是有效实验结果。**

## 运行顺序

先设置真实路径。下列训练路径是**需替换的占位符，不代表这些文件已经存在**：

```bash
export PL_TRAIN_ANN=/绝对路径/真实的refcocog_train.json
export PL_TRAIN_PROPOSALS=/绝对路径/真实的train_proposals.json
export PL_OUT=results/ref_plinear_refcocog_smoke
export PL_TRAIN_N=16
export PL_DEV_N=8
export PL_VAL_N=6
set -o pipefail
PL_STAGE=preflight python -u -B tools/ref_plinear.py
```

preflight不加载CUDA模型。缺数据、候选键缺失、图像泄漏都会明确停止；先解决输入，勿放宽断言。
确认当前节点上实际可用的GPU；若调度器已指定 `CUDA_VISIBLE_DEVICES` 则沿用，不手动覆盖。
下例仅适用于自行选择的空闲GPU0：

```bash
export CUDA_VISIBLE_DEVICES=0
PL_STAGE=cache python -u -B tools/ref_plinear.py 2>&1 | tee "$PL_OUT/cache_fit.log"
python -u -B tools/ref_plinear_train.py 2>&1 | tee "$PL_OUT/train.log"
PL_STAGE=cache_val python -u -B tools/ref_plinear.py 2>&1 | tee "$PL_OUT/cache_validation.log"
python -u -B tools/ref_plinear_eval.py 2>&1 | tee "$PL_OUT/evaluation.log"
```

需要每条命令退出码为0再执行下一条。脚本还会用COMPLETE标记拒绝不完整上游。
smoke完成只证明工程链路，不解释这6条上的性能。

正式5000/1000训练开发集 + 全验证集，沿用上面两个真实训练路径：

```bash
export PL_OUT=results/ref_plinear_refcocog_v1
unset PL_TRAIN_N PL_DEV_N PL_VAL_N
PL_STAGE=preflight python -u -B tools/ref_plinear.py
PL_STAGE=cache python -u -B tools/ref_plinear.py 2>&1 | tee "$PL_OUT/cache_fit.log"
python -u -B tools/ref_plinear_train.py 2>&1 | tee "$PL_OUT/train.log"
PL_STAGE=cache_val python -u -B tools/ref_plinear.py 2>&1 | tee "$PL_OUT/cache_validation.log"
python -u -B tools/ref_plinear_eval.py 2>&1 | tee "$PL_OUT/evaluation.log"
```

只把某条训练/评测命令前加 `PL_DEVICE=cpu` 即可离线CPU运行；特征提取仍需GPU。
三种seed 42/43/44，每种20epochs、AdamW lr=.001、wd=.0001、batch16表达。
每seed五层同初始权重（std=.01，bias=0）、相同样本顺序和预算；不同seed分别运行。
每层在训练源划出的dev上按最高raw Top-1、再低loss、再早epoch选头。
不挑验证分数最高的seed，不自动选择层数，完整报告全部五层三seed。
固定学习率只是预先约定的第一轮优化预算，不是保证每层达到最优的结论。

## 存储、换卡与结果

每表达最多 `5×100×2560×2` 字节≈2.56MB特征。
5000train +1000dev +2573validation的特征约21.95GB，加上元数据略增；先准备至少约25GB可用空间。
按batch流式读取，不一次把全部特征搬进RAM/显存；3seed×20epochs会重复读盘，建议本地SSD。
每表达提取时做原始前向和只读hook前向，核对完整及compact读出；**不用于计时**。

缓存完成后即可释放大模型GPU。离线头训练可换卡/CPU；输入特征字节不变。
train/dev与validation提取若换GPU，会记录两者型号；权重/依赖/特征协议仍要求一致。
原模型baseline在validation同一次提取中重算，不以历史跨卡logit逐位相等作为门槛。
当前不支持半次提取/训练断点续跑，避免不经审核混合不同来源；
中断目录保留诊断证据，换一个 `PL_OUT` 重新做该轮。不要删除既有成功结果或覆盖文件。
源码在preflight后改动也会触发拒绝，需要明确开新轮。

结果结构：

```text
PL_OUT/
  plan.json                      # 固定图像分组、样本和协议
  cache_fit/                     # train/dev特征；COMPLETE才可训练
  train/
    protocol.json
    seed*_epoch*.json             # 每epoch dev与在线train loss
    seed*_history.json            # 含epoch0的优化诊断
    heads_seed*.pt                # 各层在dev选定的头，含选中epoch
    COMPLETE.json
  cache_validation/              # 训练完成后才允许提取
  evaluation/
    summary.md                   # 对照表
    summary.json                 # 各seed/层指标、seed均值/样本标准差
    predictions.json             # 逐表达配对正确性、IoU、赢家与并列数
```

下一步先看：浅层linear比同层native-FP32改善多少；与原36层差多少；与新36层差多少；
3seed是否稳定、是否收敛、收益/损失集中在哪些表达。seed标准差不是数据采样置信区间。
浅层提升只说明当前监督/线性探针可读性改善；若很差，也不能排除优化预算、数据量和线性容量限制。
本轮全部缓存来自36层完整前向，**不等于训练后真实早退速度验证**。有希望后再接真实截断路径，
重新做一致性与效率测试；不拿离线头运行时间充当整模加速比。
