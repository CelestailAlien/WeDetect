# 真正的第 30 层退出：等价性与配对计时

这是 P-native 后的最小工程验证，不训练，不实现动态路由，不改 V 路径，
也不扩大到完整数据集。原 E0、P-native、模型源码、checkpoint 和结果保持不变。

## 实际执行

当前 Transformers 4.57.1 的 Qwen3VLTextModel 遍历 `self.layers`，
完成相应 DeepStack 加法后执行原 final RMSNorm。新脚本暂时让这个层列表只包含
原来的前 30 层，继续调用原模型的完整 forward、原 norm 和原分类头。
其余 6 层确实不执行；并非完整计算后的中间结果读取。

不更改模型 config、位置编码、注意力实现、cache 设置或任何参数；
每次 forward 使用新请求，不重用 KV。上下文退出或报错时恢复原层列表。
仅用于单线程/单进程推理，不可在切换期间保存模型。尾层参数仍驻留显存，
不声称节省权重显存。视觉编码、ROI、所有 token 都保留。

## 验证与停止条件

逐表达检查：

1. 数据、权重、图像、核心源码、关键依赖和协议与成功的 P-native 一致。
2. 新的完整输出与保存的完整输出一致；采集 hooks 不改输出。
3. 相同层列表替换机制保留全部 36 层时，与原完整模型一致（sham）。
4. 验证阶段给所有原 block 注册计数 hooks，真实退出时只执行 0..29，
   final norm 和 head 各执行一次。DeepStack 数量必须不超过退出深度。
5. 退出时完整序列的 pre-norm hidden state，与完整运行时第 30 层边界一致。
6. 退出 logits 与边界的同形状 norm/head 重建一致。
7. 紧凑 object-token 读取与 P-native 缓存一致。完整序列与紧凑 GEMM 采用此前
   已固定的 BF16 容差，同时要求两种排序下的 Top-1 都与缓存一致。
8. 每次计时输出也需与当前已验证输出一致。失败不产出 PASSED 汇总。

严格容差复用 E0：atol=rtol=1e-5；紧凑 GEMM 容差 atol=.03125、rtol=.01。
不可为通过测试随意放宽。`FAILED.json` 和失败样本记录用于诊断，不用于性能结论。

## 两套排名始终并列保留

- `bf16_sigmoid`：原协议，BF16 sigmoid 后转 FP32，最高分并列取原候选中第一个。
- `raw_logit`：直接按已计算的 BF16 logits 排序，转 FP32 仅供保存，
  **不是 FP32 模型前向**。两边都使用同样规则，并列仍取原顺序。

无分数阈值、无 NMS，GT 不进入输入或候选；IoU≥0.5 仅用于离线评价。
每套均报告 full/exit 正确率、伤害/恢复、选框变化、并列数、平均 IoU。
原始 logits 仍可能在 BF16 下并列；不声称已消除全部量化误差。
数值协议修正不是训练或协同机制创新，也不把当前验证子集当独立测试。

## 计时口径

在验证通过之后进行另一组无采集/计数 hooks 的运行，每样本、每种模式、
每个口径热身 2 次并重复测量 3 次。full/exit 顺序按样本、口径、重复轮次交替。
每次前后 CUDA synchronize，使用 wall-clock；检查和报告生成均不计时。

| 口径 | 包含 | 不包含 |
|---|---|---|
| forward | 已准备 GPU 输入后的原 Ref 完整前向 | 读图、输入预处理、GPU 输出回传及选框 |
| request | 读图/RGB 解码、预处理/H2D、Ref、候选分数 D2H、两套 Top-1 选择 | Uni、已加载候选的准备/裁剪、GT 指标、哈希、日志/结果写盘 |

request 使用已被访问过的图像，是**热文件缓存的固定候选 Ref 单请求**，
不是 Uni→Ref 全系统延迟，也不是冷启动/吞吐量基准。保留 checkpoint cache 设置。
两种模式使用相同口径；静态层列表切换位于计时之外（相当于部署时选定深度）。
统计先取各样本各模式重复运行的中位数，再在样本间取均值，并报告均值比值及
配对速度比中位数。保留所有原始毫秒值。勿把 6/36 层直接换算成系统加速比。

## 同步文件

只需新增以下五个文件到服务器 `tools/`：

- `ref_exit.py`
- `ref_exit_core.py`
- `ref_exit_analysis.py`
- `test_ref_exit.py`
- `REF_EXIT.md`

依赖之前已有的 E0/P-native/HumanRef 公共工具。无需重新安装环境。
P-native 的 `manifest.json`、`summary.json`、`selection.json`、`samples/*.json`
必须在服务器保留；本实验不需要其隐藏状态 `.pt` 文件。

## 运行

先运行 CPU 单元测试（其中模型为随机小型 Qwen3-VL，不下载 checkpoint）：

```bash
cd /media/data6/chengz/WeDetect
conda activate wedetect_ref
python -B tools/test_ref_exit.py --transformers
```

6 条真实模型 smoke，与 P-native 500 的前 6 条对齐：

```bash
set -o pipefail
CUDA_VISIBLE_DEVICES=0 EXIT_N=6 \
EXIT_PNATIVE=results/ref_pnative_refcocog_val500 \
EXIT_OUT=results/ref_exit_d30_refcocog_smoke6 \
python -u -B tools/ref_exit.py 2>&1 | tee ref_exit_d30_refcocog_smoke6.log
```

确认 PASS 后，运行已有的相同 500 条（不是扩大样本量）：

```bash
CUDA_VISIBLE_DEVICES=0 EXIT_N=500 \
EXIT_PNATIVE=results/ref_pnative_refcocog_val500 \
EXIT_OUT=results/ref_exit_d30_refcocog_val500 \
python -u -B tools/ref_exit.py 2>&1 | tee ref_exit_d30_refcocog_val500.log
```

每条表达包含 4 次验证前向和 20 次热身/计时请求，明显慢于之前 P-native 的
两次前向；这里用重复运行换取更可信的耗时比较。请在空闲 GPU 上运行，
避免其他任务抢占。先看 smoke 耗时再估算 500 条，不承诺固定完成时间。

输出目录不能预先创建；失败后使用新的 `EXIT_OUT`，不覆盖、不自动续跑。
主结果为 `summary.md`、`summary.json`，另有 `paired_errors.json`、逐样本 JSON、
manifest、selection、model_config、loading_info。无需再次保存 GB 级隐藏状态。
`PASS` 只说明执行一致性通过，**不保证准确率更高或加速幅度值得采用**。

可覆盖的路径与规模变量：

```text
EXIT_PNATIVE  results/ref_pnative_refcocog_val500
EXIT_ANN      wedetect_ref/eval_grounding/eval_refcoco/refcocog_validation.json
EXIT_PROPOSALS wedetect_ref/eval_grounding/eval_refcoco/refcoco_proposals_all.json
EXIT_IMAGES   data/coco2014
EXIT_REF      checkpoints/WeDetect-Ref-4B
EXIT_N        6
EXIT_DEPTH    30
EXIT_OUT      results/ref_exit_d30_refcocog_val6
```

退出深度必须在已验证 P-native 缓存中出现，且完整覆盖 DeepStack 注入。
当前建议固定 30，不在本轮扫描深度。请求更多样本时，必须已存在对应 P-native 记录。

## 本地验证记录

新增的统计、并列排序、报表独占写入、非法输入检查通过。
随机小型 Transformers 4.57.1 Qwen3-VL 在 CPU FP32/BF16、eager/SDPA、
cache 开/关的 8 种组合中，均通过 3/4/5/6 层前缀与完整运行边界对齐，
完整深度 sham、DeepStack、参数/config 不变和异常清理检查。
已有 E0、P-native、HumanRef 回归测试通过。

本机未运行真实 4B checkpoint/CUDA/FlashAttention2 实验；
这些检查不能替代服务器 smoke，也没有产生真实准确率或加速测量。
