"""Export a portable, offline HTML review of expressions harmed by Ref NMS."""
import argparse
import base64
import html
import io
import json
from pathlib import Path

from humanref_pipeline import (digest, image_path, load_annotations, load_shards,
                               diagnose, iou, select_indices, save_json)


def suppression_pairs(boxes, scores, threshold, nms_iou):
    kept, removed = [], []
    for index in sorted(select_indices(boxes, scores, threshold), key=lambda i: (-scores[i], i)):
        suppressor = next((j for j in kept if iou(boxes[index], boxes[j]) > nms_iou), None)
        if suppressor is None:
            kept.append(index)
        else:
            removed.append((index, suppressor))
    assert kept == select_indices(boxes, scores, threshold, nms_iou)
    return kept, removed


def panel(uri, width, height, overlays):
    marks = []
    for box, label, color in overlays:
        x1, y1, x2, y2 = box
        marks.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" '
                     f'fill="none" stroke="{color}" stroke-width="2" vector-effect="non-scaling-stroke"/>'
                     f'<text x="{x1}" y="{max(20,y1+20)}" fill="{color}" '
                     f'font-size="{max(width/45,14)}" stroke="black" stroke-width="0.8" '
                     f'paint-order="stroke">{html.escape(label)}</text>')
    return (f'<svg viewBox="0 0 {width} {height}"><image href="{uri}" width="{width}" '
            f'height="{height}"/>{"".join(marks)}</svg>')


STYLE = '''<meta charset="utf-8"><style>
body{font-family:system-ui;margin:24px;background:#f4f6f8;color:#18202b}
.panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
svg{width:100%;background:#111}article{background:white;padding:12px;border-radius:8px}
table{border-collapse:collapse;background:white}td,th{padding:8px;border:1px solid #cbd1da}
a{color:#1558b0}code{background:#e5e9ed;padding:2px}button{padding:8px}
@media(max-width:900px){.panels{grid-template-columns:1fr}}
</style>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--ref-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--score-threshold', type=float, default=.35)
    parser.add_argument('--nms-iou', type=float, default=.5)
    parser.add_argument('--iou', type=float, default=.5)
    parser.add_argument('--max-cases', type=int, default=0, help='0 exports all TP-decrease cases')
    args = parser.parse_args()
    assert 0 <= args.score_threshold <= 1 and 0 < args.nms_iou <= 1 and 0 < args.iou <= 1
    assert args.max_cases >= 0
    output = Path(args.output)
    assert not output.exists(), 'Choose a new output directory'
    refs, meta, hashes = load_shards(args.ref_dir, 'ref')
    assert meta['annotation_sha256'] == digest(args.annotations)
    annotations = {r['id']: r for r in load_annotations(args.annotations, 0)}
    assert set(refs) == set(meta['ids']) and set(refs) <= set(annotations)
    cases = []
    for sample_id, r in refs.items():
        ann = annotations[sample_id]
        assert r['image_name'] == ann['image_name']
        before = diagnose(ann['answer_boxes'], r['boxes'], r['scores'], args.score_threshold, args.iou)
        after = diagnose(ann['answer_boxes'], r['boxes'], r['scores'], args.score_threshold, args.iou, args.nms_iou)
        if after['one_to_one_tp'] < before['one_to_one_tp']:
            cases.append(dict(id=sample_id, domain=ann['domain'], query=ann['referring'],
                              image_name=ann['image_name'], tp_before=before['one_to_one_tp'],
                              tp_after=after['one_to_one_tp'], fp_before=before['one_to_one_fp'],
                              fp_after=after['one_to_one_fp'],
                              geometric_targets_lost=before['recovered_targets']-after['recovered_targets']))
    cases.sort(key=lambda c: (-(c['tp_before']-c['tp_after']), str(c['id'])))
    selected = cases[:args.max_cases] if args.max_cases else cases
    # Fail before export if an image is missing.
    paths = {c['id']: image_path(args.images, c['image_name']) for c in selected}
    from PIL import Image, ImageOps
    save_json(output/'cases.json', dict(settings=vars(args), total_cases=len(cases),
                                      exported_cases=len(selected), cases=cases,
                                      ref_meta=meta, ref_shard_sha256=hashes))
    links = []
    for number, c in enumerate(selected):
        ann, r = annotations[c['id']], refs[c['id']]
        boxes, scores, gt = r['boxes'], r['scores'], ann['answer_boxes']
        kept, pairs = suppression_pairs(boxes, scores, args.score_threshold, args.nms_iou)
        with Image.open(paths[c['id']]) as original:
            image = original.convert('RGB')  # Same orientation policy as inference.
            width, height = image.size
            image = ImageOps.contain(image, (1400, 1400))
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=85)
        uri = 'data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode('ascii')
        gt_marks = [(g, f'GT{j}', '#62ff8a') for j, g in enumerate(gt)]
        before_marks = [(boxes[j], f'#{j} {scores[j]:.3f}', '#ffb74a')
                        for j in select_indices(boxes, scores, args.score_threshold)]
        after_marks = [(boxes[j], f'#{j} {scores[j]:.3f}', '#54d8ff') for j in kept]
        rows = []
        for deleted, winner in pairs:
            overlap = [j for j,g in enumerate(gt) if iou(g, boxes[deleted]) >= args.iou]
            winner_overlap = [j for j,g in enumerate(gt) if iou(g, boxes[winner]) >= args.iou]
            rows.append(f'<tr><td>#{deleted} ({scores[deleted]:.3f})</td><td>#{winner} ({scores[winner]:.3f})</td>'
                        f'<td>{iou(boxes[deleted],boxes[winner]):.3f}</td><td>{overlap}</td><td>{winner_overlap}</td>'
                        f'<td>{max((iou(g,boxes[deleted]) for g in gt),default=0):.3f}</td>'
                        f'<td>{max((iou(g,boxes[winner]) for g in gt),default=0):.3f}</td></tr>')
        name = f'case_{number:03d}.html'
        page = STYLE + f'<a href="index.html">返回目录</a><h1>ID {html.escape(str(c["id"]))} · {html.escape(c["domain"])}</h1>'
        page += f'<h2>{html.escape(c["query"])}</h2><p>{html.escape(c["image_name"])}</p>'
        page += f'<p>TP {c["tp_before"]} → {c["tp_after"]}；FP {c["fp_before"]} → {c["fp_after"]}；几何覆盖目标损失 {c["geometric_targets_lost"]}</p>'
        page += '<p>候选编号在三个面板中保持一致。点击任意面板可全屏放大。</p><div class="panels">'
        for title, marks in [('GT（绿）',gt_marks),('NMS 前（橙）',before_marks),('NMS 后（蓝）',after_marks)]:
            page += f'<article onclick="this.requestFullscreen()"><h3>{title}</h3>{panel(uri,width,height,marks)}</article>'
        page += '</div><h2>被删除框 → 抑制它的保留框</h2><table><tr><th>删除框（Ref 分数）</th><th>保留框（Ref 分数）</th><th>框间 IoU</th><th>删除框覆盖 GT</th><th>保留框覆盖 GT</th><th>删除框最大 GT IoU</th><th>保留框最大 GT IoU</th></tr>'
        page += ''.join(rows) + '</table><p>重叠 GT 集合不是一对一匹配结果；最大 GT IoU 可能对应不同人物。需结合图像判断。</p>'
        (output/name).write_text(page, encoding='utf-8')
        links.append(f'<tr><td><a href="{name}">{html.escape(str(c["id"]))}</a></td><td>{html.escape(c["domain"])}</td>'
                     f'<td>{html.escape(c["query"])}</td><td>{c["tp_before"]} → {c["tp_after"]}</td>'
                     f'<td>{c["fp_before"]} → {c["fp_after"]}</td><td>{c["geometric_targets_lost"]}</td></tr>')
    index = STYLE + f'<h1>HumanRef NMS 受损样本</h1><p>共 {len(cases)} 条 TP 下降，导出 {len(selected)} 条。点击 ID 查看。图片已嵌入页面，可离线浏览。</p>'
    index += '<p>检查：①不同人物被抑制；②同一人物保留了定位较差的框；③覆盖仍在但一对一匹配冲突。不要只凭模型分数判定。</p>'
    index += '<table><tr><th>ID</th><th>Domain</th><th>Query</th><th>TP</th><th>FP</th><th>几何覆盖目标损失</th></tr>' + ''.join(links) + '</table>'
    (output/'index.html').write_text(index, encoding='utf-8')
    print(f'Found {len(cases)} harmed expressions; exported {len(selected)}. Open {output / "index.html"}')


if __name__ == '__main__':
    main()
