"""Pure randomized joins and offline file-pipeline tests; never downloads images."""
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import tempfile
from unittest.mock import patch

import ref_review_gallery as gallery


def must_fail(fn, errors=(AssertionError, KeyError)):
    try:
        fn()
    except errors:
        return
    raise AssertionError('Expected invalid input to fail')


def fixtures():
    rng = random.Random(23)
    # SOF fixture only: tests dimensions without requiring Pillow or image generation.
    header = b'\xff\xd8\xff\xc0\x00\x0b\x08\x00\xf0\x01\x40\x01\x01\x11\x00\xff\xd9'
    cases, plan = [], []
    for i in range(5):
        name=f'train2014/COCO_train2014_{i%3:012d}.jpg'
        boxes={str(j): [rng.uniform(0,50),rng.uniform(0,50),rng.uniform(100,200),rng.uniform(100,200)] for j in range(4)}
        query='synthetic </script><img onerror=alert(1)> " & test'
        row=dict(id=f'test_{i}',split='validation',image_name=name,image_key=i%3,
                 referring=query,image_sha256=gallery.sha256(header))
        plan.append(row)
        arms={}
        for a in ['linear_d24_seed42_raw_fp32',gallery.DEEP]:
            arms[a]=dict(winner=0,top1_iou=.1,geometry='low_overlap_lt010',availability='rank2_5',
                best_iou=.9,best_iou_index=2,modes=dict(raw=dict(top10=[0,1,2],top10_scores=[3.,2.,1.],
                top10_ious=[.1,.3,.9],first_qualified_index=2,first_qualified_rank=3)))
        cases.append(dict(id=row['id'],image_key=row['image_key'],image_name=name,query=query,
            candidate_boxes=boxes,gt=[boxes['2']],arms=arms,
            groups={'linear_d24_seed42_raw_fp32':'both_wrong'},review_strata=['synthetic']))
    return cases,plan,header


def test_core():
    cases,plan,header=fixtures()
    images,rows=gallery.forward_algorithm(cases,plan)
    assert len(images)==3 and len(rows)==5
    assert all(len(a['top'])==3 and all(len(c['box'])==4 for c in a['top']) for r in rows for a in r['arms'].values())
    assert gallery.jpeg_size(header)==(320,240)
    assert all(r['query']==cases[i]['query'] for i,r in enumerate(rows))
    bad=deepcopy(cases);bad[0]['candidate_boxes']['0']=[1,2,3]
    must_fail(lambda:gallery.forward_algorithm(bad,plan))
    bad=deepcopy(cases);bad[0]['query']='changed'
    must_fail(lambda:gallery.forward_algorithm(bad,plan))
    badplan=deepcopy(plan);badplan[0]['image_name']='../../private.jpg'
    bad=deepcopy(cases);bad[0]['image_name']='../../private.jpg'
    must_fail(lambda:gallery.forward_algorithm(bad,badplan))
    badplan=deepcopy(plan);badplan[3]['image_sha256']='a'*64
    must_fail(lambda:gallery.forward_algorithm(cases,badplan))
    must_fail(lambda:gallery.jpeg_size(b'not an image'))


def test_pipeline():
    cases,plan,header=fixtures()
    with tempfile.TemporaryDirectory(prefix='ref-gallery-test-') as temp:
        root=Path(temp);diag=root/'diagnostics';diag.mkdir();image_root=root/'coco';(image_root/'train2014').mkdir(parents=True)
        for name in {r['image_name'] for r in plan}:
            (image_root/name).write_bytes(header)
        gallery.write_json(root/'plan.json',dict(rows=plan))
        (diag/'review_cases.jsonl').write_text('\n'.join(json.dumps(c) for c in cases),encoding='utf-8')
        gallery.write_json(diag/'summary.json',dict(review_n=len(cases),inputs={'plan.json':gallery.sha256((root/'plan.json').read_bytes())}))
        gallery.write_json(diag/'COMPLETE.json',dict(status='PASSED',files={p.name:gallery.sha256(p.read_bytes()) for p in diag.iterdir()}))
        env=dict(REVIEW_DIAG=str(diag),REVIEW_OUT=str(root/'gallery'),REVIEW_IMAGES=str(image_root),REVIEW_DOWNLOAD='0')
        with patch.dict(os.environ,env):
            gallery.run()
            done=gallery.read_json(root/'gallery/COMPLETE.json')
            assert done['images']==3 and done['expressions']==5 and done['status']=='PASSED'
            html=(root/'gallery/index.html').read_text(encoding='utf-8')
            assert 'synthetic </script>' not in html and '\\u003c/script>' in html
            assert '__REVIEW_DATA__' not in html and 'images/train2014/' in html
            must_fail(gallery.run)
        items,_=gallery.forward_algorithm(cases,plan)
        # No fallback when source is missing, and corrupt bytes never reach output.
        must_fail(lambda:gallery.obtain_image(items[0],root/'other',None,False))
        (image_root/items[0]['image_name']).write_bytes(b'corrupt')
        must_fail(lambda:gallery.obtain_image(items[0],root/'other',image_root,False))
        assert not (root/'other'/items[0]['file']).exists()
        # A interrupted gallery may reuse exactly matching images, not changed images.
        cached=gallery.obtain_image(items[0],root/'gallery',None,False)
        assert cached['sha256']==items[0]['sha256']


if __name__=='__main__':
    assert __debug__
    test_core()
    print('Random shape/join checks / image dedup / coordinate validation / path safety / JPEG dimensions: PASS')
    test_pipeline()
    print('Offline gallery / pinned image hashes / injection escaping / overwrite guard / missing inputs: PASS')
