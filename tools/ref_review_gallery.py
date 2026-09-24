"""Download ONLY audited review images and build a portable offline HTML gallery.

Standard library only. REVIEW_DOWNLOAD=1 explicitly enables COCO HTTP downloads;
every JPEG must match the SHA-256 recorded BEFORE the original experiment.
Alternatively set REVIEW_IMAGES to the server's coco2014 image root (no network).
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/ref_plinear_refcocog_uni_full_20260921_183958_868480'
COCO_ORIGIN = 'http://images.cocodataset.org/'
WORKERS = 4
DEEP = 'original_full36_raw_bf16'


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, value) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def forward_algorithm(cases: list[dict], plan_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Pure join/validation: review expressions -> unique pinned images + UI rows."""
    assert cases and len({r['id'] for r in cases}) == len(cases)
    assert len({r['id'] for r in plan_rows}) == len(plan_rows)
    plan = {r['id']: r for r in plan_rows}
    images, rows = {}, []
    for case in cases:
        original = plan[case['id']]
        assert original['split'] == 'validation'
        assert case['query'] == original['referring'] and case['image_key'] == original['image_key']
        name = original['image_name']
        assert case['image_name'] == name
        assert re.fullmatch(r'train2014/COCO_train2014_\d{12}\.jpg', name), name
        pinned = original['image_sha256']
        assert re.fullmatch(r'[0-9a-f]{64}', pinned)
        item = dict(image_name=name, file='images/'+name, sha256=pinned)
        if name in images:
            assert images[name] == item, 'Conflicting image identity'
        images[name] = item
        boxes = case['candidate_boxes']
        for box in list(boxes.values()) + case['gt']:
            assert len(box) == 4 and all(math.isfinite(v) for v in box)
            assert box[0] <= box[2] and box[1] <= box[3]
        assert len(case['gt']) == 1
        arms = {}
        for arm, diag in case['arms'].items():
            raw = diag['modes']['raw']
            indices, scores, ious = raw['top10'], raw['top10_scores'], raw['top10_ious']
            assert 0 < len(indices) == len(scores) == len(ious) <= 10
            assert diag['winner'] == indices[0] and diag['top1_iou'] == ious[0]
            top = []
            for i, score, iou in zip(indices[:3], scores[:3], ious[:3]):
                assert math.isfinite(score) and 0 <= iou <= 1
                top.append(dict(index=i, score=score, iou=iou, box=boxes[str(i)]))
            first = raw['first_qualified_index']
            arms[arm] = dict(top=top, geometry=diag['geometry'], availability=diag['availability'],
                first_rank=raw['first_qualified_rank'], best_iou=diag['best_iou'],
                best_box=boxes[str(diag['best_iou_index'])],
                first_box=None if first is None else boxes[str(first)])
        assert DEEP in arms and all(a in arms for a in case['groups'])
        rows.append(dict(id=case['id'], image_key=case['image_key'], query=case['query'],
            image_name=name, image_file=item['file'], image_sha256=pinned,
            gt=case['gt'], groups=case['groups'], arms=arms, strata=case['review_strata']))
    return list(images.values()), rows


def jpeg_size(data: bytes) -> tuple[int, int]:
    """Read JPEG SOF dimensions without decoding/re-encoding the original image."""
    assert data[:2] == b'\xff\xd8', 'Expected original JPEG'
    pos = 2
    while pos < len(data):
        assert data[pos] == 255, 'Malformed JPEG marker'
        while pos < len(data) and data[pos] == 255:
            pos += 1
        assert pos < len(data)
        marker = data[pos]; pos += 1
        if marker in (0xd8, 0x01) or 0xd0 <= marker <= 0xd7:
            continue
        assert marker not in (0xd9, 0xda), 'JPEG SOF missing before image data'
        assert pos + 2 <= len(data)
        length = int.from_bytes(data[pos:pos+2], 'big')
        assert length >= 2 and pos + length <= len(data)
        if marker in (0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf):
            assert length >= 8
            height, width = int.from_bytes(data[pos+3:pos+5], 'big'), int.from_bytes(data[pos+5:pos+7], 'big')
            assert width > 0 and height > 0
            return width, height
        pos += length
    raise AssertionError('No JPEG dimensions')


def obtain_image(item: dict, out: Path, image_root: Path | None, download: bool) -> dict:
    target = out / item['file']
    if target.exists():
        data = target.read_bytes()
        origin = 'verified_existing_file'
    elif image_root is not None:
        source = (image_root / item['image_name']).resolve()
        assert source.is_relative_to(image_root.resolve())
        data = source.read_bytes()
        origin = str(source)
    else:
        assert download, 'Set REVIEW_IMAGES or explicitly enable REVIEW_DOWNLOAD=1'
        origin = COCO_ORIGIN + item['image_name']
        for attempt in range(3):
            try:
                with urllib.request.urlopen(origin, timeout=35) as response:
                    assert response.geturl().startswith(COCO_ORIGIN), 'Unexpected download redirect'
                    data = response.read(20_000_001)
                assert len(data) < 20_000_001, 'Unexpectedly large JPEG'
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
                if attempt == 2:
                    raise RuntimeError(f'Image download failed: {origin}') from error
                time.sleep(attempt + 1)
    assert sha256(data) == item['sha256'], f'Original image hash mismatch: {item["image_name"]}; refusing substitute'
    width, height = jpeg_size(data)
    if not target.exists():
        # Validate BEFORE writing. An interrupted write is an explicit hash failure on resume.
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
    return dict(item, width=width, height=height, bytes=len(data), origin=origin)


def run() -> None:
    assert __debug__, 'Do not use python -O'
    diag = Path(os.environ.get('REVIEW_DIAG', RUN/'topk_diagnostics_v1')).resolve()
    out = Path(os.environ.get('REVIEW_OUT', ROOT/'results/ref_topk_review_v1')).resolve()
    assert out != diag and not out.is_relative_to(diag), 'Keep gallery separate from frozen diagnostics'
    complete = read_json(diag/'COMPLETE.json')
    assert complete['status'] == 'PASSED'
    for filename, pinned in complete['files'].items():
        assert Path(filename).name == filename
        assert sha256((diag/filename).read_bytes()) == pinned, f'Changed diagnostic: {filename}'
    summary = read_json(diag/'summary.json')
    plan_path = diag.parent/'plan.json'
    assert sha256(plan_path.read_bytes()) == summary['inputs']['plan.json']
    cases = [json.loads(line) for line in (diag/'review_cases.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(cases) == summary['review_n']
    images, rows = forward_algorithm(cases, read_json(plan_path)['rows'])
    template = Path(__file__).with_name('ref_review_template.html').read_text(encoding='utf-8')
    identity = dict(plan_sha256=sha256(plan_path.read_bytes()), diagnostic_sha256=sha256((diag/'COMPLETE.json').read_bytes()),
                    source_sha256=sha256(Path(__file__).read_bytes()), template_sha256=sha256(template.encode()))
    if out.exists():
        assert read_json(out/'BUILD.json') == identity, 'Different inputs/code: choose a new REVIEW_OUT'
        assert not (out/'COMPLETE.json').exists(), 'Gallery already completed; refusing overwrite'
    else:
        out.mkdir(parents=True)
        write_json(out/'BUILD.json', identity)
    (out/'images').mkdir(exist_ok=True)
    source = Path(os.environ['REVIEW_IMAGES']).resolve() if 'REVIEW_IMAGES' in os.environ else None
    download = os.environ.get('REVIEW_DOWNLOAD', '0') == '1'
    print(f'{len(rows)} expressions / {len(images)} unique original images -> {out}', flush=True)
    fetched = {}
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(obtain_image, item, out, source, download) for item in images]
        for future in as_completed(futures):
            item = future.result()
            fetched[item['image_name']] = item
            print(f'Image {len(fetched)}/{len(images)} SHA256 PASS {item["image_name"]}', flush=True)
    for row in rows:
        item = fetched[row['image_name']]
        row['width'], row['height'] = item['width'], item['height']
        boxes = row['gt'] + [c['box'] for a in row['arms'].values() for c in a['top']]
        assert all(0 <= b[0] <= b[2] <= item['width'] and 0 <= b[1] <= b[3] <= item['height'] for b in boxes)
    review_id = sha256(json.dumps(identity, sort_keys=True).encode())
    payload = dict(review_id=review_id, rows=rows, deep=DEEP, identity=identity)
    # Escape '<' so even an untrusted query containing </script> stays inert JSON.
    embedded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    assert template.count('__REVIEW_DATA__') == 1
    with (out/'index.html').open('x', encoding='utf-8') as stream:
        stream.write(template.replace('__REVIEW_DATA__', embedded))
    write_json(out/'images_manifest.json', [fetched[i['image_name']] for i in images])
    with (out/'使用说明.txt').open('x', encoding='utf-8') as stream:
        stream.write('双击 index.html 打开，无需联网。保留同目录 images 文件夹。\n'
            '先看原图和表达，再点击“显示预测与GT”检查框。默认复核 seed42，可切换其他头。\n'
            '每条表达、每个头分别记录；页面存储只是便利功能，请定期导出 JSON 备份。\n'
            'JSON 可重新导入，CSV 用于表格查看。浏览器不能直接修改原 review.csv。\n'
            '该113条为富集诊断样例，不代表整体错误比例；请勿将 validation 复核样例混入训练。\n'
            f'所有{len(images)}张原图按实验 plan.json 的 SHA-256 验证，未缩放、裁剪或重编码。\n')
    write_json(out/'COMPLETE.json', dict(status='PASSED', expressions=len(rows), images=len(images), review_id=review_id,
        image_bytes=sum(i['bytes'] for i in fetched.values()),
        files={p.name: sha256(p.read_bytes()) for p in sorted(out.iterdir()) if p.is_file()}))
    print(f'Completed offline gallery: {out / "index.html"}', flush=True)


if __name__ == '__main__':
    run()
