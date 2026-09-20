# E0：先证明读取正确，不训练、不改 Ref

本阶段只构建 E0。原模型、候选框、预处理、prompt、精度和 attention 后端保持固定。
不会先铺开逐层 probe、视觉 token 删除或动态退出器。E0 通过不等于早退方案有效。

## 范围与通过标准

默认使用 **RefCOCOg validation**，60 条固定工程样本：先选同图不同表达的一对，
优先选择 GT 不同的表达，再按固定种子抽取剩余样本。不是完整数据集的准确率估计，
不用于训练。本版本替换之前的 HumanRef E0 入口；原 HumanRef 评测管线不变。

候选来源是作者提供的 `refcoco_proposals_all.json`，按注释中的 image 字段查找，
截前 100、裁剪边界；绝不插入 GT、重排候选或乘上 proposal objectness。
注释读取 `conversations[1]["value"]` 和 `bounding_boxes`，匹配仓库官方 REC loader。
支持裸框列表及 `[boxes, scores]`，会区分“恰好两个裸框”，不凭长度猜格式。

**每条表达仅选择最高 Ref 分数的一个候选，不设 Ref 分数阈值、不做额外 NMS。**
IoU≥0.5 是固定的 GT 匹配评价标准，不是预测分数阈值。候选生成时可能已有后处理，
此处固定作者候选文件及其哈希，并非声称整个系统从来没有 NMS。
沿用现有 BF16 sigmoid 后转 FP32 的分数，最高分相等时取原候选顺序中的第一个，
两条路径使用相同规则。与官方 `torch.topk` 的并列顺序、旧脚本 BF16 输出坐标舍入
可能存在差异，因此 E0 是内部一致性协议，不宣称与历史论文分数逐位一致。

每条样本检查：

1. 无 hook 重复前向 logits 稳定。
2. 只读采集 hook 不改变 object logits。
3. 最终 Norm 输入经过原 Norm + 原 head，复现原 logits。
4. 同一表示经过独立纯张量 RMSNorm + linear 实现，复现原 logits。
5. 仅 object rows 的紧凑读取通过数值及 Top-1 候选索引一致性检查。
6. 单独计时的 hooks 不改变 logits。
7. 首对同图不同 query 的图像输入相同、候选相同，h0 应保持一致。

严格检查 `atol=rtol=1e-5`。紧凑 BF16 GEMM 因矩阵形状不同，单独使用
`atol=0.03125, rtol=0.01`，仍要求 Top-1 候选索引相同。
两种误差都会记录，不能仅凭“数值接近”忽略预测改变。失败时先查数据与数值路径，
不要为了通过测试直接放宽容差。非有限张量、字段缺失、错维度、缺权重直接失败。

每条记录另存所选框 IoU、是否正确、候选是否覆盖 GT、最高分并列数量；不同候选可能
都正确，不能将索引改变直接解释为任务退化。但 E0 是严格的实现一致性检查，索引改变
仍需审查。`summary.json` 中的 Top-1/覆盖率只标记为工程子集诊断，不作为论文成绩。

只保存 h0 和 hL 的 object states，不做中间层 sweep。h0 取首个 decoder block 输入；
hL 取最终文本 RMSNorm 输入，避免最后一次 DeepStack 注入前后混淆。**h0 已包含
视觉编码、ROI 与位置投影的成本，不是未经视觉编码的初始 token。**

“切开 decoder 后无干预续算”暂不实现：那是 V 阶段干预前的独立门槛。
本阶段无任何续算器，不能把完整 forward 的 hook 检查声称为 split/resume 已通过。

## 同步文件

新增以下文件，和现有 `tools/humanref_pipeline.py` 一起放到服务器对应目录：

- `ref_e0.py`：固定输入、加载模型、检查、计时、保存证据。
- `ref_e0_core.py`：纯张量原头读取、只读 hook 和计时 hook。
- `ref_e0_data.py`：样本选择与候选决策比较。
- `test_ref_e0.py`：人工随机数据、失败路径及真实 tiny Qwen text decoder 测试。
- `REF_E0.md`：本说明。

不需要重新安装你的服务器环境。要求现有 PyTorch、torchvision、Transformers 4.57.1、
flash-attn、Pillow 和模型原有依赖。固定 BF16 + flash_attention_2，不静默回退后端。
模型从本地加载，缺文件直接失败，不自动下载替代权重。

## 服务器命令

先运行不需要 GPU 或真实权重的测试（最后一项使用小尺寸随机 Qwen3-VL decoder）：

```bash
cd /media/data6/chengz/WeDetect
conda activate wedetect_ref
python -B tools/test_ref_e0.py --transformers
```

然后做 6 条真实模型 smoke。是**单卡、普通 python**，不需要 torchrun。
终端日志放在输出目录之外，因为脚本拒绝已有输出目录。

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 E0_N=6 E0_OUT=results/ref_e0_refcocog_smoke6 \
  python -u -B tools/ref_e0.py 2>&1 | tee ref_e0_refcocog_smoke6.log
```

smoke 全部通过之后，再跑 60 条：

```bash
CUDA_VISIBLE_DEVICES=0 E0_N=60 E0_OUT=results/ref_e0_refcocog_val60 \
  python -u -B tools/ref_e0.py 2>&1 | tee ref_e0_refcocog_val60.log
```

默认路径：

```text
E0_ANN        wedetect_ref/eval_grounding/eval_refcoco/refcocog_validation.json
E0_PROPOSALS  wedetect_ref/eval_grounding/eval_refcoco/refcoco_proposals_all.json
E0_IMAGES     data/coco2014
E0_REF        checkpoints/WeDetect-Ref-4B
```

路径不同时可在命令前设置 `E0_ANN=... E0_PROPOSALS=... E0_IMAGES=... E0_REF=...`。
图像根目录必须与注释中的 image 字符串拼成真实路径。例如 image 是
`train2014/COCO_train2014_...jpg` 时根目录指向 `data/coco2014`；若 image 只有文件名，
根目录应指向 `data/coco2014/train2014`。不会静默搜索其他数据目录。
如果你指的是 **RefCOCO 而非 RefCOCOg**，可将 `E0_ANN` 设为同目录的
`refcoco_validation.json`，并另用一个输出目录；同样支持 `refcocoplus_validation.json`。
默认只读一个 validation 文件，不混入 test/testA/testB。

仅路径、输出目录
和样本数用环境变量；实验超参数固定在脚本顶部并写入 manifest。不覆盖输出、不自动续跑，
失败重试请选择新 `E0_OUT`，保留失败现场。权重 SHA256 需要读完整文件，启动可能较慢，
这不计入模型耗时。BF16 使用与现有管线相同的 sigmoid 后转 FP32 顺序。

研究顺序：先 E0，再在 validation 上做逐层诊断及后续模块选择；训练 probe 另用
image-disjoint 训练数据。确定方案后再做完整 test 和 RefCOCO/+/g 跨集检验，最后
补 HumanRef 的多目标与拒识检验。REC 无需调分数/NMS 阈值，不等于训练超参数、
退出层选择也无需开发集，更不代表 REC 的结果自动适用于多目标/负查询。

## 输出与计时解释

| 文件 | 内容 |
|---|---|
| `summary.md` / `summary.json` | 仅全部通过才生成；误差与耗时概览 |
| `FAILED.json` | 失败原因与 traceback；不产生成功报告 |
| `manifest.json` | 权重/源码/数据/图像哈希、实际 L/d、版本、精度、后端、协议 |
| `selection.json` | 固定样本、同图 query 对、采样规则 |
| `loading_info.json` / `model_config.json` | 权重加载核查、实际模型配置 |
| `samples/0000.json` | 输入候选、GT、原/重建 logits、检查结果、计时 |
| `samples/0000.pt` | h0、hL-preNorm、input/position IDs、BF16 输入框、原 logits |

`.pt` 是自己生成的实验工件，后续读取可使用 `torch.load(..., weights_only=True)`。
不会长期保存全序列 hL 或 KV cache。约 60 条、N=100、d=2560、BF16 时，两组
object states 总计约 61 MB，实际以记录的候选数与隐藏维度为准。

每条样本先预热 2 次，无 hook 计时 3 次；采集/重建不计入无 hook 计时。
另外运行一次 CUDA event 插桩，区分视觉编码、ROI+输入组装、LLM、head 等区间。
ROI+输入组装包含多尺度卷积、ROIAlign、投影、scatter、位置准备，不谎称是纯 projector。
LLM 包含内部位置处理和 final Norm。CUDA event 区间可能含 CPU 提交造成的空隙，
不是 profiler 的逐 kernel 时间；逐区间相加应近似整个 event span。

图像读取/预处理/CPU 选框检查单独记录。预处理可能包含首次初始化；这些工程诊断计时
不能当稳定 benchmark FPS。未在线执行 Uni，**不能报告 Uni→Ref 端到端加速**，
也不能拿它与论文 FPS 比较。未来需在相同软硬件、相同缓存范围下专门测最终方法。

## 验证边界

标准库测试可在没有 torch 的本机运行；`--torch` 增加 FP32/BF16 纯函数和合成 hook
测试；`--transformers` 额外验证真正的 4.57.1 text decoder、DeepStack、末层读取。
这些只使用随机权重，不替代服务器的 BF16 + FlashAttention + ROI + 真实 checkpoint 测试。

核查依据：仓库 `wedetect_ref/models/qwen3vl_referring.py` 与
[Transformers 4.57.1 Qwen3-VL 源码](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py)。
若服务器版本或模型结构不同，先重新审查，不自动兼容未知边界。

本地构建验证（2026-09-20）：在独立临时环境、PyTorch 2.5.1+cpu /
Transformers 4.57.1 下，标准库、FP32/BF16 读取、合成 hook、真实 tiny text decoder
测试通过；计时 hook 仅用模拟 events 验证区间组织与异常清理。原 HumanRef 管线
回归测试通过。**尚未验证真实权重、CUDA event 计时或 FlashAttention2 前向**，
以服务器 smoke 结果为准，不将 CPU 测试通过记为真实 E0 通过。
