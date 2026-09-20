# HumanRef baseline 核查与后续比较协议

状态：作者配置核查记录 + 当前探索协议 + 后续研究比较草案。
本文件不将 HumanRef 上选出的阈值认定为作者设置，也不将后处理收益视为模型创新。

## 1. 作者公开设置核查

本轮检查了官方 README、Ref README、单图推理脚本、HumanRef 评测入口及论文
arXiv v1（可访问版本）。v2 HTML 本轮访问失败，不能声称已核查全部版本或作者私有流程。

| 来源 | 可确认内容 | 无法推出的结论 |
|---|---|---|
| [论文 §9](https://arxiv.org/html/2512.12309v1#S9) | 一般性描述训练/评测使用 Base-Uni top-100 proposals | 未找到 HumanRef 专用分数阈值与 Ref NMS 完整配置 |
| [Ref README](https://github.com/WeChatCV/WeDetect/blob/main/wedetect_ref/README.md) | 提供 RefCOCO 命令，num_select=20 | 不是 HumanRef 完整复现命令 |
| [主 README](https://github.com/WeChatCV/WeDetect/blob/main/README.md) | Ref 单图 demo 示例使用 score_thre=0.3 | demo 阈值不是 HumanRef benchmark 设置的证明 |
| [infer_wedetect_ref.py](https://github.com/WeChatCV/WeDetect/blob/main/infer_wedetect_ref.py) | Uni 生成100候选，Ref 按分数或 Top1 选择；无额外 Ref NMS | 不能说明论文 HumanRef 是否采用额外 NMS |
| [eval.py](https://github.com/WeChatCV/WeDetect/blob/main/wedetect_ref/eval_grounding/eval.py) | HumanRef 读取 candidate_boxes；--nms 默认关闭，启用时 IoU=0.7 | 代码默认值不等于作者产生论文数字时的命令 |

论文概括与 HumanRef 代码候选来源之间的差异尚未解决。准确表述是：
**在以上检查范围内未找到真实 Uni→Ref 的 HumanRef 完整后处理设置。**
如后续获得作者说明，应保存原文、来源、版本，再更新协议；不能仅因数字接近而认定一致。

## 2. 当前自建端到端评测协议

- 官方 Base-Uni + Ref-4B 权重，记录文件哈希。
- Uni FP32、原实现内置 NMS=0.7、最多100候选；无额外 objectness 阈值。
- Ref 主体 BF16，固定官方提示词；模型框输入 BF16，保存/评测几何坐标 FP32。
- Ref sigmoid 分数使用严格大于阈值；额外 Ref NMS 按 Ref 分数排序，平分按原候选序号。
- 使用原始 HumanRef 标注和官方指标实现；保留 candidate_boxes 字段供原指标计算。
- A 使用数据集候选；B 使用真实 Uni 候选；C 复用 B 分数并追加 Ref NMS=0.7。
- 已运行的 NMS=0.5 属于另一项探索对照。所有历史结果均保留，不覆盖。
- 当前全量 HumanRef 已参与方法诊断和参数探索；不能再描述为完全未见的最终测试。

## 3. 小范围单因素敏感性

锚点：Ref 阈值0.35，额外 Ref NMS IoU=0.5，均为已有探索设置。

| 观察因素 | 取值 | 固定项 |
|---|---|---|
| Ref 分数阈值 | 0.25、0.35、0.45 | NMS=0.5 |
| 额外 Ref NMS | 关闭、0.4、0.5、0.6、0.7 | Ref 阈值=0.35 |

共7个独立配置（锚点复用），不是全组合搜索。其余因素保持不变。
运行脚本只报告所有结果，不自动选择最优或最接近论文的配置。
报告两组曲线/表格及 Recall、Precision、DF1、Rejection 和误检构成。
NMS 本身不会将非空集合变为空，因此固定分数阈值时拒识结果应不变。

```bash
cd /media/data6/chengz/WeDetect
conda activate wedetect_ref
python -B tools/humanref_sensitivity.py \
  --annotations data/HumanRef/annotations.jsonl \
  --ref-dir results/humanref_abc_k100/ref \
  --output results/humanref_abc_k100/sensitivity_v1
```

输出 `plan.json`（缓存来源及哈希）、`summary.md/csv/json`、每个配置的日志及完整分析目录。
不需要图像或 GPU。需要服务器已有的 annotations.jsonl 和指标依赖；本地仅有推理缓存无法重算官方指标。

## 4. 后续公平比较草案

在新方法最终实验前确定并保存以下内容：

1. 训练、开发、最终测试的样本ID清单与版本。按图像划分，同图多表达必须在同一组；
   检查重复图像和近重复。不能把已看过的完整 HumanRef 重新切分后称为全新独立测试。
   RefCOCO 验证集可以支持对应任务调参，但不自动具有多目标/拒识开发样本覆盖。
   最终独立数据来源尚待根据研究方向落实，不在这里虚构已拥有的划分。
2. 参数选择目标在开发前固定。例如主任务为严格定位时用 DF1@0.5:0.95，若关注拒识则
   预先规定拒识约束或联合目标；不得看完测试结果后更换主指标。
3. baseline 和新方法允许不同数值阈值，但使用同一开发集、目标、相同搜索预算和并列处理规则。
   新方法改变分数分布时，强制相同数值未必公平。搜索范围需覆盖各自分数尺度，次数一致。
4. 固定候选上限、图像处理、初始化权重与训练预算。若研究对象就是候选或计算分配，
   单独列为变化项，并用等计算预算对照、计算—性能曲线报告。
5. 端到端计入 Uni、Ref、额外裁剪/通信/重跑和后处理；固定硬件与精度，独立测量延迟。
   当前缓存时间不能直接用于在线 FPS 的公平比较。
6. 参数选定后冻结，在最终测试评测。所有方法报告一致的指标、错误定义与阈值选择来源。
   有训练的方法可用相同种子组重复；置信区间可按图像配对重采样，避免同图表达被视为独立。

## 5. 按科学问题选择诊断

| 研究目标 | 应优先检查 | 最小有效对照 |
|---|---|---|
| 关系/否定理解 | 去重后仍选错人的关系表达，先确认正确候选确实存在 | 固定候选/后处理/预算；等量普通上下文、关系对应打乱 |
| 定位/框选择 | 候选已有高IoU框，但实际高分或最终保留框定位较差 | 语义分数、Uni质量分数/简单组合；GT oracle仅作诊断上限 |
| 实例去重 | NMS抑制错误、不同人物重叠、同一实例多种框 | 普通NMS等强简单对照；112条TP受损样本适用于此方向 |
| 候选补救 | 自然漏检、原候选不存在可用目标框 | 固定追加检测、低置信度触发；检查真实补救上限 |

112条样本不是所有协同研究的通用入口。先确认想验证的机制，再决定抽样与改动。
