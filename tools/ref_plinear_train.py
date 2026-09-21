"""Offline P-linear fitting: train/dev only, no model or official validation load."""
import os
from pathlib import Path
import random

from humanref_pipeline import digest, save_json
from ref_e0 import write_tensor
from ref_plinear import DEPTHS, OUTPUT, source_hashes
from ref_plinear_data import batches, open_cache, read_json

SEEDS = [42, 43, 44]
EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
INIT_STD = .01
DEVICE = os.environ.get('PL_DEVICE', 'cuda')


def check_fit(root, plan):
    manifest, entries = open_cache(root, 'fit')
    assert manifest['depths'] == DEPTHS == plan['depths']
    assert manifest['plan_sha256'] == digest(OUTPUT / 'plan.json')
    assert manifest['model_signature']['sources'] == source_hashes() == plan['sources']
    expected = [r for r in plan['rows'] if r['split'] != 'validation']
    assert [(e['id'], e['split'], e['image_key']) for e in entries] == [
        (r['id'], r['split'], r['image_key']) for r in expected]
    train = [e for e in entries if e['split'] == 'train']
    dev = [e for e in entries if e['split'] == 'dev']
    assert len(train) == plan['counts']['train'] and len(dev) == plan['counts']['dev']
    assert not ({e['image_key'] for e in train} & {e['image_key'] for e in dev})
    return manifest, train, dev


def score_heads(root, entries, weight, bias, device):
    import torch
    from ref_plinear_core import compute_loss, forward_algorithm
    loss_sum = torch.zeros(len(DEPTHS), device=device)
    correct = torch.zeros(len(DEPTHS), device=device, dtype=torch.long)
    count = 0
    with torch.no_grad():
        for x, y, mask, overlaps, _ in batches(root, entries, BATCH_SIZE, device):
            logits = forward_algorithm(x, weight, bias)
            loss_sum += compute_loss(logits, y, mask) * x.shape[0]
            picks = logits.masked_fill(~mask[:, None], -torch.inf).argmax(-1)
            selected_iou = overlaps.gather(1, picks)
            assert selected_iou.shape == (x.shape[0], len(DEPTHS))
            correct += (selected_iou >= .5).sum(0)
            count += x.shape[0]
    assert count == len(entries)
    return dict(correct=correct.cpu().tolist(), loss=(loss_sum / count).cpu().tolist(), n=count)


def run():
    import torch
    from ref_plinear_core import compute_loss, forward_algorithm
    assert __debug__ and int(os.environ.get('WORLD_SIZE', '1')) == 1
    assert DEVICE == 'cpu' or (DEVICE == 'cuda' and torch.cuda.is_available())
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    plan = read_json(OUTPUT / 'plan.json')
    root, out = OUTPUT / 'cache_fit', OUTPUT / 'train'
    assert not out.exists(), f'Refusing to overwrite {out}'
    manifest, train, dev = check_fit(root, plan)
    save_json(out / 'protocol.json', dict(depths=DEPTHS, seeds=SEEDS, epochs=EPOCHS,
        batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        optimizer='AdamW; separate independent head slices; sum of depth losses',
        initialization=f'same N(0,{INIT_STD}^2) weight across depths per seed; zero bias',
        loss='original sigmoid focal alpha=.25 gamma=2; IoU if >.5 else 0; query-balanced mean',
        selection='best dev raw-logit Top1 count; then lower dev focal loss; then earlier epoch',
        precision='FP32 heads on frozen BF16 original-final-norm features; TF32 disabled',
        trainable='one weight[D] and bias per depth; no backbone or norm gradients',
        plan_sha256=digest(OUTPUT / 'plan.json'), cache_complete_sha256=digest(root / 'COMPLETE.json'),
        source_sha256={Path(__file__).name: digest(__file__), **source_hashes()},
        device=DEVICE, gpu=torch.cuda.get_device_name(0) if DEVICE == 'cuda' else None,
        torch=str(torch.__version__), cuda=torch.version.cuda, smoke_only=plan['smoke_only']))
    d = manifest['model_signature']['hidden_size']
    checkpoints = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        initial = torch.randn(1, d) * INIT_STD
        weight = initial.expand(len(DEPTHS), d).clone().to(DEVICE).requires_grad_()
        bias = torch.zeros(len(DEPTHS), device=DEVICE, requires_grad=True)
        optimizer = torch.optim.AdamW([weight, bias], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        best_keys = [None] * len(DEPTHS)
        best_w, best_b, best_epochs = torch.empty_like(weight), torch.empty_like(bias), [None] * len(DEPTHS)
        history = [dict(epoch=0, dev=score_heads(root, dev, weight, bias, DEVICE))]
        for epoch in range(1, EPOCHS + 1):
            order = list(train)
            random.Random(seed * 100000 + epoch).shuffle(order)
            losses = torch.zeros(len(DEPTHS), device=DEVICE)
            seen = 0
            for x, y, valid, _, _ in batches(root, order, BATCH_SIZE, DEVICE):
                optimizer.zero_grad(set_to_none=True)
                per_depth = compute_loss(forward_algorithm(x, weight, bias), y, valid)
                per_depth.sum().backward()
                assert weight.grad is not None and bias.grad is not None
                assert torch.isfinite(weight.grad).all() and torch.isfinite(bias.grad).all()
                optimizer.step()
                losses += per_depth.detach() * x.shape[0]
                seen += x.shape[0]
            assert seen == len(train)
            metrics = score_heads(root, dev, weight, bias, DEVICE)
            for k in range(len(DEPTHS)):
                key = (metrics['correct'][k], -metrics['loss'][k], -epoch)
                if best_keys[k] is None or key > best_keys[k]:
                    best_keys[k], best_epochs[k] = key, epoch
                    best_w[k], best_b[k] = weight[k].detach(), bias[k].detach()
            row = dict(epoch=epoch, online_train_loss=(losses / seen).cpu().tolist(), dev=metrics)
            history.append(row)
            save_json(out / f'seed{seed}_epoch{epoch:02d}.json', row)
            print(f'P-linear seed={seed} epoch={epoch}/{EPOCHS} dev={metrics["correct"]}/{metrics["n"]}', flush=True)
        save_json(out / f'seed{seed}_history.json', history)
        path = out / f'heads_seed{seed}.pt'
        write_tensor(path, dict(depths=DEPTHS, seed=seed, weight=best_w.detach().cpu(), bias=best_b.detach().cpu(),
            selected_epochs=best_epochs, selected_dev_correct=[key[0] for key in best_keys],
            protocol_sha256=digest(out / 'protocol.json'), plan_sha256=digest(OUTPUT / 'plan.json'),
            model_signature=manifest['model_signature']), torch)
        checkpoints.append(dict(file=path.name, sha256=digest(path), seed=seed, selected_epochs=best_epochs))
    save_json(out / 'COMPLETE.json', dict(status='PASSED', checkpoints=checkpoints,
        protocol_sha256=digest(out / 'protocol.json'), plan_sha256=digest(OUTPUT / 'plan.json')))
    print('All heads frozen. Next: PL_STAGE=cache_val, then ref_plinear_eval.py.', flush=True)


if __name__ == '__main__':
    # Required by deterministic CUDA BLAS; set before creating CUDA context.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    run()
