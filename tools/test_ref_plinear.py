"""CPU-only math, leakage, corruption and synthetic offline integration tests.

Synthetic data are generated ONLY here, never as a fallback in experiment code.
"""
import ast
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch
import torch.nn.functional as F

from humanref_pipeline import ROOT, digest, save_json
from ref_e0 import write_tensor
from ref_e0_core import forward_algorithm as native_readout
from ref_plinear import DEPTHS, source_hashes
from ref_plinear_core import compute_loss, forward_algorithm, frozen_norm
from ref_plinear_data import image_key, open_cache, read_sample, split_train_dev


def must_fail(fn, kinds=(AssertionError, KeyError)):
    try:
        fn()
    except kinds:
        return
    raise AssertionError('Expected invalid input to fail')


def test_math():
    torch.manual_seed(123)
    h = torch.randn(11, 16).bfloat16()
    scale = torch.randn(16).bfloat16()
    w, b = torch.randn(1, 16).bfloat16(), torch.randn(1).bfloat16()
    z = frozen_norm(h, scale, 1e-6)
    assert z.shape == h.shape and z.dtype == torch.bfloat16
    assert torch.equal(F.linear(z, w, b), native_readout(h, scale, 1e-6, w, b))
    x = torch.randn(3, 5, 7, 16)
    weights = torch.randn(5, 16, requires_grad=True)
    bias = torch.randn(5, requires_grad=True)
    y = torch.rand(3, 7)
    y[0] = 0  # candidate-miss expressions still have legitimate all-zero labels
    mask = torch.ones(3, 7, dtype=torch.bool)
    logits = forward_algorithm(x, weights, bias)
    assert logits.shape == (3, 5, 7)
    loss = compute_loss(logits, y, mask)
    # Execute the author's actual standalone focal function, without importing
    # the full GPU model or third-party vision stack into this CPU test.
    tree = ast.parse((ROOT / 'wedetect_ref/models/qwen3vl_referring.py').read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'sigmoid_focal_loss')
    namespace = {'torch': torch, 'F': F}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<official-focal>', 'exec'), namespace)
    for k in range(5):
        expected = namespace['sigmoid_focal_loss'](logits[:, k], y, 1)
        assert torch.allclose(loss[k], expected, atol=1e-7)
    loss[2].backward()
    assert weights.grad[2].abs().sum() > 0
    assert torch.count_nonzero(weights.grad[[0, 1, 3, 4]]) == 0
    assert x.grad is None and z.grad is None
    # Candidate padding must not change the query-balanced loss.
    mask[1, 3:] = False
    a = compute_loss(logits.detach(), y, mask)
    changed, yc = logits.detach().clone(), y.clone()
    changed[1, :, 3:] = 1000
    yc[1, 3:] = .99
    assert torch.equal(a, compute_loss(changed, yc, mask))
    expected = torch.stack([namespace['sigmoid_focal_loss'](logits[i, 0, mask[i]], y[i, mask[i]], 1)
                            for i in range(3)]).mean()
    assert torch.allclose(a[0], expected)
    must_fail(lambda: forward_algorithm(x, weights[:, :3], bias))
    must_fail(lambda: compute_loss(logits, y[:, :2], mask))
    must_fail(lambda: compute_loss(logits, y, torch.zeros_like(mask)))
    # Shared heads respect a permutation of candidate order; no position-specific weights.
    perm = torch.randperm(7)
    assert torch.allclose(forward_algorithm(x[:, :, perm], weights, bias), logits[:, :, perm])
    # Tiny realizable signal should fit; protects loss sign and gradient flow.
    features = torch.randn(12, 1, 6, 8)
    labels = (features[:, 0, :, 0] > 0).float()
    wt = torch.zeros(1, 8, requires_grad=True)
    bt = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([wt, bt], lr=.1)
    valid = torch.ones_like(labels, dtype=torch.bool)
    before = float(compute_loss(forward_algorithm(features, wt, bt), labels, valid).detach())
    for _ in range(60):
        opt.zero_grad()
        objective = compute_loss(forward_algorithm(features, wt, bt), labels, valid).sum()
        objective.backward()
        opt.step()
    after = float(compute_loss(forward_algorithm(features, wt, bt), labels, valid).detach())
    assert after < before * .3, (before, after)


def test_splits():
    rows = [dict(id=f'refcocog_train_{i}', image_name=f'COCO_train2014_{i//3:012d}.jpg') for i in range(60)]
    val = [dict(id='refcocog_val_0', image_name='000000000999.jpg')]
    split = split_train_dev(rows, val, 13, 7, 123)
    assert split == split_train_dev(list(reversed(rows)), val, 13, 7, 123)
    assert len(split['train']) == 13 and len(split['dev']) == 7
    assert not ({image_key(r['image_name']) for r in split['train']} &
                {image_key(r['image_name']) for r in split['dev']})
    assert image_key('train2014/COCO_train2014_000000000123.jpg') == image_key('123.jpg')
    must_fail(lambda: split_train_dev(rows, [dict(id='x', image_name='000000000001.jpg')], 13, 7, 123))
    must_fail(lambda: split_train_dev(rows, val, 1000, 7, 123))
    must_fail(lambda: split_train_dev([dict(id='refcocog_val_0', image_name='1.jpg')], val, 1, 1, 1))


def test_preflight():
    import ref_plinear as pipeline
    with tempfile.TemporaryDirectory(prefix='ref-plinear-preflight-') as directory:
        root = Path(directory)
        annotations, validation, proposals, hashes = [], [], {}, {}
        for i in range(16):
            name = f'COCO_train2014_{i:012d}.jpg'
            path = root / name
            # Preflight checks presence/hashes, not image decoding. Only this
            # explicitly synthetic test writes placeholder image bytes.
            path.write_bytes(f'SYNTHETIC_PREFLIGHT_IMAGE_{i}'.encode())
            hashes[str(path)] = digest(path)
            row = dict(id=f'refcocog_{"train" if i < 14 else "val"}_{i}', image=name,
                       conversations=[dict(value='<image>'), dict(value='synthetic query')],
                       bounding_boxes=[[0, 0, 10, 10]])
            (annotations if i < 14 else validation).append(row)
            proposals[name] = [[0, 0, 10, 10], [20, 20, 30, 30]]
        ann, val, props = root / 'train.json', root / 'validation.json', root / 'proposals.json'
        save_json(ann, annotations)
        save_json(val, validation)
        save_json(props, proposals)
        # Only SELECTED training images have proposals. Unselected images must
        # not block preflight, and availability must not alter the split.
        from ref_plinear_inputs import load_rec_rows, select_rows
        chosen = select_rows(load_rec_rows(ann), load_rec_rows(val), 6, 3, pipeline.SPLIT_SEED)
        names = {r['image_name'] for rows in chosen.values() for r in rows}
        partial = root / 'selected_proposals.json'
        save_json(partial, {n: proposals[n] for n in names})
        ref = root / 'reference'
        save_json(ref / 'summary.json', dict(status='PASSED'))
        save_json(ref / 'manifest.json', dict(full_split=True, num_layers=36,
            annotation_sha256=digest(val), proposals_sha256=digest(props),
            sample_ids=[r['id'] for r in validation], image_sha256=hashes))
        with patch.multiple(pipeline, OUTPUT=root / 'run', REFERENCE=ref, VAL_ANN=val,
                            VAL_PROPOSALS=props, IMAGES=root, TRAIN_N=6, DEV_N=3, VAL_N=0,
                            SELECTION=None, UNI_RUN=None):
            with patch.dict('os.environ', {'PL_TRAIN_ANN': str(ann), 'PL_TRAIN_PROPOSALS': str(partial)}):
                pipeline.preflight()
                plan = pipeline.read_json(root / 'run/plan.json')
                assert plan['counts'] == dict(train=6, dev=3, validation=2)
                assert plan['smoke_only']
                must_fail(pipeline.preflight)


def test_integration():
    import ref_plinear_train as train
    import ref_plinear_eval as evaluate
    with tempfile.TemporaryDirectory(prefix='ref-plinear-test-') as directory:
        output = Path(directory)
        rows = [dict(id=f'synthetic_{i}', split='train' if i < 8 else ('dev' if i < 12 else 'validation'),
                     image_key=i) for i in range(16)]
        plan = dict(depths=DEPTHS, rows=rows, counts=dict(train=8, dev=4, validation=4),
                    sources=source_hashes(), smoke_only=True)
        save_json(output / 'plan.json', plan)
        torch.manual_seed(42)
        signature = dict(hidden_size=8, sources=source_hashes())
        # Create only fit first; real validation extraction is similarly gated
        # until heads have been selected. This fixture skips the expensive model.
        train.OUTPUT = evaluate.OUTPUT = output
        train.DEVICE = evaluate.DEVICE = 'cpu'
        train.EPOCHS, train.SEEDS, train.BATCH_SIZE = 2, [42, 43], 4
        for stage in ('fit', 'validation'):
            if stage == 'validation':
                train.run()
            root = output / ('cache_fit' if stage == 'fit' else 'cache_validation')
            chosen = [r for r in rows if (r['split'] == 'validation') == (stage == 'validation')]
            manifest = dict(stage=stage, depths=DEPTHS,
                plan_sha256=digest(output / 'plan.json'), sample_ids=[r['id'] for r in chosen],
                model_signature=signature, gpu='SYNTHETIC-CPU')
            if stage == 'validation':
                manifest['frozen_train_complete_sha256'] = digest(output / 'train/COMPLETE.json')
            save_json(root / 'manifest.json', manifest)
            (root / 'samples').mkdir()
            entries = []
            for i, row in enumerate(chosen):
                n = 3 + i % 3
                features = torch.randn(5, n, 8).bfloat16()
                overlaps = torch.zeros(n)
                overlaps[i % n] = .9
                overlaps[(i + 1) % n] = .5  # evaluation positive but training target zero
                sample = dict(row, depths=DEPTHS, features=features, overlaps=overlaps,
                    labels=torch.where(overlaps > .5, overlaps, 0), query='SYNTHETIC TEST',
                    native_bf16=torch.randn(5, n).bfloat16(), native_fp32=torch.randn(5, n),
                    baseline=torch.randn(n).bfloat16())
                file = root / 'samples' / f'{i:05d}.pt'
                write_tensor(file, sample, torch)
                entries.append(dict(row, file=f'samples/{i:05d}.pt', sha256=digest(file)))
            save_json(root / 'index.json', entries)
            save_json(root / 'COMPLETE.json', dict(status='PASSED', stage=stage, samples=len(entries),
                manifest_sha256=digest(root / 'manifest.json'), index_sha256=digest(root / 'index.json')))
            open_cache(root, stage)
            damaged = dict(entries[0], sha256='not-the-hash')
            must_fail(lambda: read_sample(root, damaged))
        # Exercise the real train and evaluation loops, not a test reimplementation.
        evaluate.run()
        summary = evaluate.read_json(output / 'evaluation/summary.json')
        assert summary['status'] == 'PASSED' and summary['n'] == 4 and summary['smoke_only']
        assert len(summary['trained_seed_summary']) == 5
        assert len(summary['curve']) == 2 + 5 * 3 + 5 * 2
        must_fail(train.run)  # refuse to overwrite frozen heads
        must_fail(evaluate.run)


if __name__ == '__main__':
    assert __debug__
    torch.set_num_threads(2)
    test_math()
    print('Math / official loss parity / frozen gradients / padding / overfit: PASS')
    test_splits()
    print('Image-group split / alias leakage / insufficient data: PASS')
    test_preflight()
    print('CPU preflight / author schema / frozen plan / reference image hashes: PASS')
    test_integration()
    print('Synthetic fit -> checkpoint selection -> paired validation / corruption / overwrite gates: PASS')
