"""CPU tests for fixed selection, Uni outputs, provenance and resume artifacts."""
import ast
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from humanref_pipeline import ROOT, digest, save_json
from ref_plinear_inputs import load_rec_rows, select_rows, load_selected_candidates, verify_uni_artifact
from ref_plinear_uni_core import forward_algorithm, remap_checkpoint
import ref_plinear_proposals as uni


def fails(fn, exception=AssertionError):
    try:
        fn()
    except exception:
        return
    raise AssertionError('Bad input was accepted')


def test_core():
    torch.manual_seed(32)
    a = torch.rand(100, 2) * 100
    boxes = torch.cat((a, a + torch.rand(100, 2) * 10), 1)
    scores = torch.rand(100).sort(descending=True).values
    x, y = forward_algorithm(boxes, scores, 640, 480)
    assert x.shape == (100, 4) and y.shape == (100,)
    assert torch.equal(x, boxes) and torch.equal(y, scores)
    fails(lambda: forward_algorithm(boxes[:, :3], scores, 640, 480))
    fails(lambda: forward_algorithm(boxes, scores.flip(0), 640, 480))
    fails(lambda: forward_algorithm(boxes[:0], scores[:0], 640, 480))
    fails(lambda: forward_algorithm(boxes, scores, 1, 1))
    ckpt = {'backbone.image_model.model.stages.0.weight': torch.randn(2),
            'bbox_head.head_module.cls_preds.0.0.conv.weight': torch.randn(2),
            'embeddings': torch.randn(256, 768)}
    mapped = remap_checkpoint({'state_dict': ckpt})
    assert set(mapped) == {'backbone.stages.0.weight', 'bbox_head.cls_preds.0.0.weight', 'embeddings'}
    assert mapped['embeddings'] is ckpt['embeddings']
    fails(lambda: remap_checkpoint({'backbone.x': torch.ones(1), 'backbone.image_model.model.x': torch.ones(1)}))
    # Real author's score filtering, no NMS replacement; threshold=0, prompt-label multi-label ranking.
    tree = ast.parse((ROOT / 'generate_proposal.py').read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'filter_scores_and_topk')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<original-Uni-filter>', 'exec'), namespace)
    values = torch.tensor([[.2, .9], [.8, 0.]])
    s, labels, candidates, _ = namespace['filter_scores_and_topk'](values, 0., 30000)
    assert torch.equal(s, torch.tensor([.9, .8, .2]))
    assert labels.tolist() == [1, 0, 0] and candidates.tolist() == [0, 1, 0]
    uni.check_generator_protocol(ROOT / 'generate_proposal.py')
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'changed_generator.py'
        path.write_text((ROOT / 'generate_proposal.py').read_text(encoding='utf-8').replace('labels, 0.7)', 'labels, 0.6)'), encoding='utf-8')
        fails(lambda: uni.check_generator_protocol(path))
    print('Random tensor shapes / clipping / score order / checkpoint remap / original score filter: PASS')


def test_selection_and_provenance():
    with tempfile.TemporaryDirectory(prefix='plinear-uni-test-') as directory:
        root = Path(directory)
        raw = [dict(id=f'refcocog_train_{i}', image=f'COCO_train2014_{i//2:012d}.jpg',
                    conversations=[dict(value='<image>'), dict(value=f'query {i}')],
                    bounding_boxes=[[0, 0, 10, 10]]) for i in range(40)]
        ann = root / 'train.json'
        save_json(ann, raw)
        rows = load_rec_rows(ann)
        val = [dict(id='refcocog_val_0', image_name='999.jpg', referring='q', answer_boxes=[[0, 0, 1, 1]])]
        chosen = select_rows(rows, val, 12, 6, 42)
        lock = root / 'selection.json'
        save_json(lock, dict(seed=42, rows=chosen))
        smoke = select_rows(rows, val, 3, 2, 42, lock)
        assert smoke == dict(train=chosen['train'][:3], dev=chosen['dev'][:2])
        assert select_rows(rows, val, 12, 6, 42, lock) == chosen
        fails(lambda: select_rows(rows, val, 12, 6, 43, lock))
        changed = deepcopy(rows)
        changed[next(i for i, r in enumerate(rows) if r['id'] == chosen['train'][0]['id'])]['referring'] = 'changed'
        fails(lambda: select_rows(changed, val, 12, 6, 42, lock))
        names = sorted({r['image_name'] for records in chosen.values() for r in records})
        data = {name: [[[0, 0, 2, 2], [3, 3, 4, 4]], [.9, .5]] for name in names}
        candidate_path = root / 'train_proposals.json'
        save_json(candidate_path, data)
        assert len(load_selected_candidates(candidate_path, names)) == len(names) < len({r['image_name'] for r in rows})
        fails(lambda: load_selected_candidates(candidate_path, names + ['missing.jpg']))
        manifest = dict(smoke_only=False, images=names, selection_sha256=digest(lock), train_annotation_sha256=digest(ann))
        save_json(root / 'manifest.json', manifest)
        save_json(root / 'COMPLETE.json', dict(status='PASSED', artifacts={
            'manifest.json': digest(root / 'manifest.json'), 'train_proposals.json': digest(candidate_path)}))
        assert verify_uni_artifact(root, candidate_path, smoke, ann, lock)['manifest_sha256'] == digest(root / 'manifest.json')
        record = dict(image_name=names[0], image_sha256='sha', width=10, height=10, boxes=data[names[0]][0], scores=data[names[0]][1])
        assert uni.validate_record(record, names[0], 'sha', 10, 10) == data[names[0]]
        fails(lambda: uni.validate_record(record, names[0], 'changed', 10, 10))
        rpath = root / 'resume.json'
        with patch.object(uni, 'RESUME', False):
            uni.write_or_verify(rpath, record)
            fails(lambda: uni.write_or_verify(rpath, record))
        with patch.object(uni, 'RESUME', True):
            uni.write_or_verify(rpath, record)
            fails(lambda: uni.write_or_verify(rpath, dict(record, width=11)))
        # Damaging an exported candidate file must fail provenance validation.
        candidate_path.write_text('{}', encoding='utf-8')
        fails(lambda: verify_uni_artifact(root, candidate_path, smoke, ann, lock))
        report = uni.summarize_coverage(chosen, data, {n: dict(width=640, height=480) for n in names})
        assert report['train']['evaluated_expressions'] == 12 and report['train']['hits_iou50'] == 0
        assert report['dev']['evaluated_expressions'] == 6  # no drops despite zero candidate recall
    print('Frozen split / smoke prefixes / selected-only candidates / provenance / resume / no GT repair: PASS')


def test_real_selection():
    if (uni.DATA / 'conversion.json').is_file() and uni.SELECTION.is_file():
        selected, names, _ = uni.frozen_inputs(uni.DATA, uni.SELECTION)
        assert len(names) == 1663 and len(selected['train']) == 5000 and len(selected['dev']) == 1000
        print('Real converted data + prior audit: frozen 1663 image set PASS (no model inference)')


def test_extraction_io():
    """Actual orchestration with EXPLICITLY MOCKED inference; no GPU/model claim."""
    from PIL import Image
    with tempfile.TemporaryDirectory(prefix='uni-io-test-') as directory:
        root = Path(directory)
        names = [f'train2014/COCO_train2014_{i:012d}.jpg' for i in range(2)]
        inventory = {n: dict(width=8, height=8, split='train') for n in names}
        selected = {split: [dict(id=f'refcocog_train_{i}', image_name=names[i],
                     referring='synthetic', answer_boxes=[[0, 0, 1, 1]])]
                    for i, split in enumerate(('train', 'dev'))}
        for i, name in enumerate(names):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (8, 8), (i*30, 60, 90)).save(path)
        for name in ('conversion.json', 'refcocog_train.json', 'refcocog_validation_first.json', 'selection.json'):
            save_json(root / name, {'fixture_only': True})
        checkpoint = root / 'synthetic.pth'
        torch.save({'synthetic': torch.ones(1)}, checkpoint)
        model = MagicMock()
        for method in ('cuda', 'float', 'eval', 'requires_grad_'):
            getattr(model, method).return_value = model
        model.img_size, model.num_proposals = (640, 640), 100
        model.parameters.return_value = [torch.nn.Parameter(torch.ones(1), requires_grad=False)]
        model.load_state_dict.return_value = SimpleNamespace(missing_keys=[], unexpected_keys=[])
        model.side_effect = lambda images: [dict(bboxes=torch.tensor([[2., 2., 4., 4.], [4., 4., 8., 8.]]), scores=torch.tensor([.9, .7]))]
        out = root / 'output'
        with patch.multiple(uni, DATA=root, SELECTION=root / 'selection.json', IMAGES=root,
                            CHECKPOINT=checkpoint, OUTPUT=out, LIMIT=0, RESUME=False), \
             patch.object(uni, 'frozen_inputs', return_value=(selected, names, inventory)), \
             patch.object(uni, 'package_version', return_value='synthetic-test'), \
             patch.dict('sys.modules', {'generate_proposal': SimpleNamespace(SimpleYOLOWorldDetector=lambda **kwargs: model),
                                       'torchvision': SimpleNamespace(__version__='synthetic-test')}), \
             patch.object(torch.cuda, 'is_available', return_value=True), \
             patch.object(torch.cuda, 'get_device_name', return_value='SYNTHETIC-NO-GPU'), \
             patch.object(torch.cuda, 'get_device_capability', return_value=(0, 0)):
            uni.run()
            assert model.call_count == 2 and (out / 'COMPLETE.json').is_file()
            complete_sha = digest(out / 'COMPLETE.json')
            with patch.object(uni, 'RESUME', True):
                uni.run()
                assert model.call_count == 2 and digest(out / 'COMPLETE.json') == complete_sha
                with patch.object(torch.cuda, 'get_device_name', return_value='DIFFERENT-GPU'):
                    fails(uni.run)
            smoke_out = root / 'smoke'
            with patch.multiple(uni, OUTPUT=smoke_out, LIMIT=1):
                uni.run()
                fails(lambda: verify_uni_artifact(smoke_out, smoke_out / 'train_proposals.json', selected,
                                                  root / 'refcocog_train.json', root / 'selection.json'))
    print('MOCKED extraction -> export -> verified resume / GPU change rejection / smoke rejection: PASS')


if __name__ == '__main__':
    assert __debug__
    torch.set_num_threads(2)
    test_core()
    test_selection_and_provenance()
    test_extraction_io()
    test_real_selection()
