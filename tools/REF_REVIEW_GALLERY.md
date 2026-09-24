# 离线人工复核图册

图册只使用 `topk_diagnostics_v1/review_cases.jsonl` 已选定的表达，不再挑最好种子。
这批数据为113条表达、107张不同的COCO train2014图片。默认显示 seed42，可切换其他头。

## 本地直接下载并生成

只依赖 Python 标准库，不需要 GPU、PyTorch、Pillow 或服务器登录。

```powershell
cd D:\code\study\WeDetect
python -B tools/test_ref_review_gallery.py
$env:REVIEW_DOWNLOAD = '1'
python -u -B tools/ref_review_gallery.py
```

公开图片来源：`http://images.cocodataset.org/train2014/`。
原图每张必须通过既有实验 `plan.json` 中的 SHA-256 校验；不会关闭 TLS 验证，
不会用相似图替代，不缩放、重编码或裁剪 JPEG。下载源 HTTPS 存在域名证书不匹配，
因此使用公开 HTTP 加已固定内容哈希的方式，下载不携带账号、密码或实验内容。

默认输出 `results/ref_topk_review_v1/`：

- `index.html`：双击即可离线复核。
- `images/train2014/*.jpg`：107张原图，去重存放。
- `images_manifest.json`：来源、哈希、尺寸、大小。
- `BUILD.json`、`COMPLETE.json`：输入绑定和完成检查。

路径可用 `REVIEW_DIAG`、`REVIEW_OUT` 修改；已完成的图册不覆盖。
网络中断后以相同代码、同一命令重跑，可复用已通过哈希校验的图片。
如果文件已损坏会停止，不会自动删除或替换。换 `REVIEW_OUT` 可重新生成。

## 服务器无网络模式

```bash
REVIEW_IMAGES=/media/data6/chengz/WeDetect/data/coco2014 \
REVIEW_OUT=results/ref_topk_review_v1 \
python -u -B tools/ref_review_gallery.py
```

此模式只复制待复核原图，无需下载整个数据集；将输出目录整体传回本地即可。
必须将 Python 脚本和 `ref_review_template.html` 一起同步。

## 如何复核

1. 先看左侧原图和表达，判断目标与歧义；点击“显示预测与GT”再看结果。
2. 可切换 GT、浅层Top-3、深层Top-1 叠加，200%/300%放大或在新窗口看原图。
3. 选错语义对象 / 同对象框质量问题 / 表达或标注歧义 / 无法确定分别填写；填写具体依据。
4. 点“标记本条已复核”。每个表达与模型组合独立保存，不把不同种子重复计作独立证据。
5. 定期“导出标注JSON”；也可导出CSV供查看。后续把JSON交回即可继续统计。

浏览器本地存储只是便利功能，清理缓存、移动文件或换浏览器后不保证恢复。
JSON 是可靠的手动备份格式，可导入到同一图册；冲突的标注不会静默覆盖。
浏览器不会直接修改原 `review.csv`，所有实验结果保持冻结。
复核集是富集样本，不能用其错误比例代表全体，也不应把这些 validation 样例混入训练。
