"""No weights/GPU needed. Stdlib tests always run; --torch adds tensor tests.

--transformers also exercises the real pinned Qwen3-VL text decoder, initialized
randomly and made tiny. Synthetic tests do NOT validate the user's checkpoint.
"""
import ast
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace

from ref_e0_data import canonical_hash, choose_samples, compare_decisions, load_rec_annotations


def expect_failure(function, error_type):
    try:
        function()
    except error_type:
        return
    raise AssertionError(f'Expected {error_type.__name__}')


def test_data():
    rng = random.Random(7)
    rows = [dict(id=i, image_name=f'image_{i // 2}', referring=f'query_{i}',
                 answer_boxes=[] if i % 3 == 0 else [[rng.randrange(10), 0, 20, 20]],
                 domain='rejection' if i % 3 == 0 else 'positive', sub_domain=str(i % 2))
            for i in range(40)]
    sample = choose_samples(rows, 12, 42)
    assert sample == choose_samples(rows, 12, 42)
    assert len(sample) == len({row['id'] for row in sample}) == 12
    assert sample[0]['image_name'] == sample[1]['image_name']
    assert sample[0]['referring'] != sample[1]['referring']
    assert sample[0]['answer_boxes'] != sample[1]['answer_boxes']
    assert {row['domain'] for row in sample} == {'rejection', 'positive'}
    assert len(choose_samples(rows, len(rows), 42)) == len(rows)
    expect_failure(lambda: choose_samples(rows, 41, 42), AssertionError)
    expect_failure(lambda: choose_samples(rows[::2], 5, 42), AssertionError)
    invalid = [dict(row) for row in rows]
    del invalid[0]['referring']
    expect_failure(lambda: choose_samples(invalid, 4, 42), KeyError)
    boxes = [[0, 0, 10, 10], [20, 0, 30, 10]]
    exact = compare_decisions(boxes, [.2, .8], [.2, .8], [boxes[1]])
    assert exact['top1_equal'] and exact['baseline_correct'] and exact['actual_correct']
    crossed = compare_decisions(boxes, [.34999, .8], [.35001, .8], [boxes[1]])
    assert crossed['top1_equal']  # crossing .35 is irrelevant in REC Top-1
    low = compare_decisions(boxes, [.001, .002], [.001, .002], [boxes[1]])
    assert low['baseline_indices'] == [1] and low['baseline_correct']  # never reject
    switched = compare_decisions(boxes, [.8, .9], [.9, .8], [boxes[1]])
    assert not switched['top1_equal'] and switched['baseline_correct'] and not switched['actual_correct']
    both_good = compare_decisions([boxes[0], boxes[0]], [.8, .9], [.9, .8], [boxes[0]])
    assert not both_good['top1_equal'] and both_good['baseline_correct'] and both_good['actual_correct']
    tied = compare_decisions(boxes, [.8, .8], [.8, .8], [boxes[0]])
    assert tied['baseline_indices'] == [0] and tied['baseline_top1_ties'] == 2
    missed = compare_decisions(boxes, [.8, .9], [.8, .9], [[80, 80, 90, 90]])
    assert not missed['candidate_covers_gt'] and not missed['baseline_correct']
    expect_failure(lambda: compare_decisions(boxes, [.8, float('nan')], [.8, .9], [boxes[0]]), AssertionError)
    assert canonical_hash(boxes) != canonical_hash(boxes[::-1])
    assert canonical_hash({'b': 2, 'a': 1}) == canonical_hash({'a': 1, 'b': 2})
    for name in ('ref_e0.py', 'ref_e0_core.py', 'ref_e0_data.py', 'test_ref_e0.py'):
        ast.parse(Path(__file__).with_name(name).read_text(encoding='utf-8'))
    print('E0 stdlib tests passed: selection, pairing, missing fields, decision gates, syntax.')


def test_rec_loader():
    rng = random.Random(15)
    boxes = [[x, 0, x + 10, 10] for x in (rng.randrange(5), 20)]
    annotations = [dict(id=i, image='train2014/test.jpg', bounding_boxes=[boxes[i]],
                        conversations=[dict(value='unused'), dict(value=f'object {i}')])
                   for i in range(2)]
    with tempfile.TemporaryDirectory() as temporary:
        annotation_path, proposal_path = Path(temporary) / 'val.json', Path(temporary) / 'proposals.json'

        def write_fixture(anns, proposals):
            annotation_path.write_text(json.dumps(anns), encoding='utf-8')
            proposal_path.write_text(json.dumps(proposals), encoding='utf-8')

        for entry in (boxes, [boxes, [.2, .1]]):
            write_fixture(annotations, {'train2014/test.jpg': entry})
            rows, proposals = load_rec_annotations(annotation_path, proposal_path)
            assert len(rows) == 2 and rows[1]['referring'] == 'object 1'
            assert rows[0]['answer_boxes'] == [boxes[0]]
            assert proposals['train2014/test.jpg']['boxes'] == boxes  # includes TWO-bare-box case
            assert rows[0]['image_name'] == 'train2014/test.jpg'
        write_fixture(annotations, {})
        expect_failure(lambda: load_rec_annotations(annotation_path, proposal_path), KeyError)
        write_fixture(annotations, {'train2014/test.jpg': [boxes, [.1]]})
        expect_failure(lambda: load_rec_annotations(annotation_path, proposal_path), AssertionError)
        write_fixture(annotations + annotations, {'train2014/test.jpg': boxes})
        expect_failure(lambda: load_rec_annotations(annotation_path, proposal_path), AssertionError)
        annotations[0]['bounding_boxes'] = []
        write_fixture(annotations, {'train2014/test.jpg': boxes})
        expect_failure(lambda: load_rec_annotations(annotation_path, proposal_path), AssertionError)
    print('E0 REC loader tests passed: author schema, bare/scored proposals, missing data, single GT.')


def test_tensors():
    import torch
    from ref_e0_core import (capture_boundaries, compare_tensors, forward_algorithm,
                             object_logits, test_algorithm)
    test_algorithm()
    torch.manual_seed(123)
    hidden = torch.randn(1, 9, 16)
    weight, bias = torch.randn(1, 16), torch.randn(1)
    scale = torch.randn(16)
    expect_failure(lambda: forward_algorithm(hidden, scale[:2], 1e-6, weight, bias), AssertionError)
    expect_failure(lambda: forward_algorithm(hidden, scale, 1e-6, weight, bias.repeat(2)), AssertionError)
    expect_failure(lambda: forward_algorithm(hidden, scale, -1, weight, bias), AssertionError)
    dirty = hidden.clone()
    dirty[0, 0, 0] = float('nan')
    expect_failure(lambda: forward_algorithm(dirty, scale, 1e-6, weight, bias), AssertionError)
    expect_failure(lambda: compare_tensors(hidden, hidden[:, :2], 0, 0), AssertionError)
    expect_failure(lambda: object_logits(hidden, torch.zeros(1, 9, dtype=torch.bool)), AssertionError)

    # Minimal synthetic hook target, not a claim of equivalence to Qwen attention.
    lm = torch.nn.Module()
    lm.layers = torch.nn.ModuleList([torch.nn.Linear(16, 16), torch.nn.Linear(16, 16)])
    lm.norm = torch.nn.Identity()
    lm.config = SimpleNamespace(hidden_size=16)
    positions = torch.zeros(1, 9, dtype=torch.bool)
    positions[0, [5, 8]] = True
    visual = torch.zeros_like(positions)
    visual[0, :2] = True
    deepstack = [torch.randn(2, 16), torch.randn(2, 16)]

    def fixture_forward(inputs_embeds, visual_pos_masks, deepstack_visual_embeds, position_ids):
        value = inputs_embeds
        for layer, delta in zip(lm.layers, deepstack_visual_embeds):
            value = layer(value)
            value = value.clone()
            value[visual_pos_masks] += delta
        return lm.norm(value)

    lm.forward = fixture_forward
    wrapper = SimpleNamespace(model=SimpleNamespace(language_model=lm))
    inputs = dict(inputs_embeds=hidden, visual_pos_masks=visual,
                  deepstack_visual_embeds=deepstack, position_ids=torch.arange(9).view(1, 9))
    with torch.inference_mode():
        baseline = lm(**inputs)
        with capture_boundaries(wrapper, positions) as captured:
            hooked = lm(**inputs)
        assert torch.equal(baseline, hooked)
        assert torch.equal(captured['h0'], hidden[positions])
        assert torch.equal(captured['hL_full'], baseline)
        assert captured['hL'].shape == (2, 16)
        assert captured['num_visual_tokens'] == 2 and captured['deepstack_count'] == 2
        snapshot = captured['hL_full'].clone()
        hooked.add_(3)
        assert torch.equal(snapshot, captured['hL_full']), 'Captured tensor aliases live tensor'
    assert not lm._forward_pre_hooks and not lm.norm._forward_pre_hooks
    assert not lm.layers[0]._forward_pre_hooks

    def injected_failure():
        with capture_boundaries(wrapper, positions):
            raise RuntimeError('Intentional: verify hook cleanup')

    expect_failure(injected_failure, RuntimeError)
    assert not lm._forward_pre_hooks and not lm.norm._forward_pre_hooks
    assert not lm.layers[0]._forward_pre_hooks

    def overlap_failure():
        with capture_boundaries(wrapper, visual):
            lm(**inputs)

    expect_failure(overlap_failure, AssertionError)
    assert not lm._forward_pre_hooks
    print('E0 tensor/hook tests passed, including error cleanup and invalid inputs.')


def test_real_text_decoder():
    import torch
    import transformers
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    from ref_e0_core import capture_boundaries, forward_algorithm
    assert transformers.__version__ == '4.57.1'
    torch.manual_seed(123)
    config = Qwen3VLTextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        rope_scaling={'rope_type': 'default', 'mrope_section': [1, 1, 2]},
        use_cache=False, attention_dropout=0.0)
    config._attn_implementation = 'eager'
    lm = Qwen3VLTextModel(config).eval()
    head = torch.nn.Linear(32, 1).eval()
    wrapper = SimpleNamespace(model=SimpleNamespace(language_model=lm))
    embeddings = torch.randn(1, 12, 32)
    visual = torch.zeros(1, 12, dtype=torch.bool)
    visual[0, [1, 2, 3]] = True
    objects = torch.zeros_like(visual)
    objects[0, [9, 10]] = True
    inputs = dict(inputs_embeds=embeddings, visual_pos_masks=visual,
        deepstack_visual_embeds=[torch.randn(3, 32) for _ in range(3)],
        position_ids=torch.arange(12).view(1, 1, 12).expand(3, 1, 12),
        attention_mask=torch.ones(1, 12, dtype=torch.long), use_cache=False)
    with torch.inference_mode():
        baseline = lm(**inputs).last_hidden_state
        with capture_boundaries(wrapper, objects) as captured:
            hooked = lm(**inputs).last_hidden_state
        torch.testing.assert_close(baseline, hooked, atol=0, rtol=0)
        rebuilt = forward_algorithm(captured['hL_full'], lm.norm.weight,
                                     lm.norm.variance_epsilon, head.weight, head.bias)
        torch.testing.assert_close(rebuilt, head(hooked), atol=1e-6, rtol=1e-6)
        # Actual model, changed query embedding, identical object/image inputs.
        changed = dict(inputs)
        changed['inputs_embeds'] = embeddings.clone()
        changed['inputs_embeds'][:, 6] += torch.randn(32)
        with capture_boundaries(wrapper, objects) as other:
            lm(**changed)
        assert torch.equal(captured['h0'], other['h0'])
        assert not torch.equal(captured['hL'], other['hL'])
    print('E0 real Transformers 4.57.1 tiny Qwen3-VL decoder test passed (CPU FP32, random weights).')


def test_profile_hooks():
    """Mock CUDA events to test intervals/cleanup only; not a latency measurement."""
    from itertools import count
    from unittest.mock import patch
    import torch
    from ref_e0_core import profile_forward
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.visual = torch.nn.Linear(8, 8)
    model.model.language_model = torch.nn.Linear(8, 8)
    model.out_proj = torch.nn.Linear(8, 1)
    model.forward = lambda value: model.out_proj(model.model.language_model(model.model.visual(value)))
    ticks = count()

    def event(enable_timing):
        assert enable_timing
        value = SimpleNamespace()
        value.record = lambda: setattr(value, 'tick', next(ticks))
        value.elapsed_time = lambda other: float(other.tick - value.tick)
        return value

    value = torch.randn(1, 3, 8)
    with patch('torch.cuda.Event', event), patch('torch.cuda.synchronize'):
        output, spans = profile_forward(model, {'value': value})
        assert torch.equal(output, model(value=value))
        assert sum(v for k, v in spans.items() if k != 'total_event_ms') == spans['total_event_ms']
        expect_failure(lambda: profile_forward(model, {'wrong_argument': value}), TypeError)
    for module in (model.model.visual, model.model.language_model, model.out_proj):
        assert not module._forward_pre_hooks and not module._forward_hooks
    print('E0 profiling-hook structure/cleanup tests passed (mock events; no GPU timing verified).')


if __name__ == '__main__':
    assert __debug__, 'Tests require assertions; do not use -O'
    assert set(sys.argv[1:]) <= {'--torch', '--transformers'}
    test_data()
    test_rec_loader()
    if '--torch' in sys.argv or '--transformers' in sys.argv:
        test_tensors()
        test_profile_hooks()
    else:
        print('NOT RUN: tensor tests. Run --torch in wedetect_ref; this is not an E0 model PASS.')
    if '--transformers' in sys.argv:
        test_real_text_decoder()
