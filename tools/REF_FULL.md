# 完整 RefCOCOg 验证集：36 层 vs 30 层真实退出

本阶段只评测准确率，固定此前选定的第 30 层，不扫描深度、不训练、不重新计时。
完整对象是作者提供的 `refcocog_validation.json` 中的所有表达，**不是**
RefCOCO/RefCOCO+/RefCOCOg 三套数据，也不是 RefCOCOg 的 train/test 合并。
它仍是参与方法开发的验证集，不应写成独立测试集结果。

## 相比旧脚本的变化

- `FULL_N=0` 遍历标注文件的全部行，保持原顺序，自动读取总数，不硬编码样本数。
- `FULL_N=10` 仅用于 smoke，报告显式标记 `smoke_only`；不调用 E0 的抽样/GT 配对逻辑。
- 不生成或依赖全量 P-native 隐藏状态缓存，不要求重新跑 E0/P-native。
- 每条表达各做一次完整前向和真实 30 层退出；首条另加无 hook 对照和 36 层 sham。
- 每条均验证实际层执行、DeepStack、退出边界、原 full-shape norm/head 重建，以及
  两套排名的赢家与同次完整运行的第 30 层读出一致。数值门槛不放宽。
- 验证前向带只读 hooks，**不是计时实验**；不把总运行时间换算成加速比。
- 使用已经成功的 500 条 static-exit 记录冻结权重、数据、源码、关键依赖和协议。
  对历史 logits 的差异只做诊断，不因换卡后的数值漂移中止。

原来的 E0、P-native、static-exit 脚本、checkpoint 和结果均不修改。

## 固定协议

原 BF16、FlashAttention2、cache 设置、图像处理、prompt 全部沿用。
作者候选前 100 个、原顺序、按图像边界裁剪；不插入 GT、不乘 objectness。
同一个 frozen 模型、同一 GPU、同一份输入分别评估 full-36 和 exit-30。
每次独立 forward，不重用 KV。退出后仍用原 final RMSNorm 和原分类头。
所有 token 保留，尾层权重仍驻留显存，不声称节省权重显存。

两套预先固定的 Top-1 规则应用于双方：

1. `bf16_sigmoid`：原协议，BF16 sigmoid 后转 FP32，最高分并列取原候选第一个。
2. `raw_logit`：按 BF16 logits 直接排序，并列也取原候选第一个；
   转 FP32 仅用于保存，**不是 FP32 模型推理**，也不能消除原 logits 的 BF16 并列。

无 Ref 分数阈值、无额外 NMS。正确标准为 IoU≥0.5。
另报 IoU≥0.75、平均 IoU、并列数、候选覆盖率与覆盖条件下准确率作为诊断。
候选未覆盖 GT 的样本仍进入总分母，不自动剔除。

## 同步文件

将五个新文件同步到服务器项目的 `tools/`：

- `ref_full.py`
- `ref_full_core.py`
- `ref_full_analysis.py`
- `test_ref_full.py`
- `REF_FULL.md`

依赖服务器上原有 E0、P-native、static-exit 公共工具及原模型环境；不需要安装新库。
默认参考目录是已经成功的：
`results/ref_exit_d30_refcocog_a6000_val500`。
保留其中 `manifest.json`、`summary.json`、`samples/*.json`；不需要任何隐藏状态 `.pt`。
不能用之前在第 10 条失败的目录作为参考。

## 运行顺序

在有可用 GPU 的服务器节点，先确认当前分配的设备。若调度系统已经设置
`CUDA_VISIBLE_DEVICES`，沿用分配，不覆盖。下面假设手动选择的空闲卡是 0。

```bash
cd /media/data6/chengz/WeDetect
conda activate wedetect_ref
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
export CUDA_VISIBLE_DEVICES=0
set -o pipefail
python -B tools/test_ref_full.py --transformers
```

通过后先跑标注原顺序的前 10 条（不保证是旧 smoke 的那 10 条）：

```bash
FULL_N=10 \
FULL_REFERENCE=results/ref_exit_d30_refcocog_a6000_val500 \
FULL_OUT=results/ref_full_d30_refcocog_smoke10 \
python -u -B tools/ref_full.py \
2>&1 | tee ref_full_d30_refcocog_smoke10.log
```

smoke 通过后，跑完整验证集：

```bash
FULL_N=0 \
FULL_REFERENCE=results/ref_exit_d30_refcocog_a6000_val500 \
FULL_OUT=results/ref_full_d30_refcocog_validation \
python -u -B tools/ref_full.py \
2>&1 | tee ref_full_d30_refcocog_validation.log
```

单卡普通 python，不使用 torchrun。输出目录不能预先创建；不覆盖、不续跑。
中断或失败后用新的 `FULL_OUT` 重跑，并保留原诊断文件。
每条一般 2 次前向，而旧脚本每条包含 24 次验证/热身/计时前向；
全量表达数更大，实际耗时由标注数量和当前 GPU 决定，不承诺固定运行时间。

路径变量（省略时采用以下默认值）：

```text
FULL_ANN       wedetect_ref/eval_grounding/eval_refcoco/refcocog_validation.json
FULL_PROPOSALS wedetect_ref/eval_grounding/eval_refcoco/refcoco_proposals_all.json
FULL_IMAGES    data/coco2014
FULL_REF       checkpoints/WeDetect-Ref-4B
FULL_REFERENCE results/ref_exit_d30_refcocog_a6000_val500
FULL_N         0
FULL_OUT       results/ref_full_d30_refcocog_validation
```

没有 `FULL_DEPTH` 调参入口，本轮固定 30；不改图像分辨率、候选数量或精度。
只允许路径迁移，数据/权重/已验证源码和关键依赖仍需与参考协议匹配。

## 跨卡与历史对比

允许参考目录来自 A6000、本次运行使用另一张受支持 GPU。启动时记录型号、容量、
计算能力、CUDA/库版本与当前数值后端设置。当前完整模型和退出模型始终在同一设备配对。
`history_drift=True` 表示与旧缓存有差异，不表示当前退出错误。

历史输入、图像内容、候选顺序、权重和协议不一致仍会停止；
同次运行的退出边界/输出/执行检查失败也立即停止。绝不能把任意失败都解释为换卡。
每次运行使用独立目录；不支持在中途换卡后续写或混合多卡结果。

## 输出和解读

| 文件 | 内容 |
|---|---|
| `summary.md` / `summary.json` | 总体及分组的两套排名准确率、伤害/恢复、数值检查、历史差异 |
| `paired_errors.json` | 两套排名的伤害/恢复表达 ID |
| `samples/*.json` | 两边的 logits、scores、选框、GT IoU、执行轨迹、逐条一致性及历史差异 |
| `sample_hashes.json` | 逐样本结果 SHA256；生成汇总前重新读取全部样本验证覆盖 |
| `manifest.json` / `selection.json` | 权重/数据/源码/参考结果哈希、完整 ID 集、选取顺序与运行环境 |
| `model_config.json` / `loading_info.json` | 模型配置与权重加载检查 |
| `FAILED.json` | 失败信息；有此文件的运行不得当作完整成功实验 |

汇总分组：

- `all_evaluated`：本次全部表达；正式运行应有 `full_split=true`。
- `previous_expressions`：与旧 500 条重叠的表达。
- `new_expressions`：其余表达，可能仍与旧表达来自同一张图。
- `new_images`：来自旧 500 条未出现图像的表达；是 `new_expressions` 的子集，
  两者不能相加，也不能称为官方独立测试集。

完成条件是所有选中样本全部通过且文件齐全。全量时核对 ID 顺序与整个标注文件
完全一致；漏项、重复、额外文件或错误决策统计都不允许产生有效汇总。
`PASSED` 只证明工程一致性和覆盖完整，不表示退出准确率更高。

## 本地验证

标准库测试覆盖完整/部分选取、漏项/重复/乱序、分组分母、候选未覆盖、
历史分数变化不阻塞、历史输入变化阻塞、双排序统计、独占写盘和结果覆盖。
重算本地 A6000 500 条旧记录，两套排名指标与保存的汇总一致。
随机小型 Transformers 4.57.1 Qwen3-VL 的 FP32/BF16、eager/SDPA、cache 开/关
8 种组合验证新的配对执行函数、首条控制、形状断言与 hooks 恢复。
本机没有完整数据集、真实 4B checkpoint 或 CUDA，尚未运行真实全量评测；
服务器 smoke 是必要的下一步，不能用 CPU 测试替代。
