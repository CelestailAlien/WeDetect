"""Run stdlib tests; add --transformers for CPU tensor + real tiny decoder tests."""
import ast
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from ref_e0_data import canonical_hash
from ref_pnative_analysis import layer_result, query_pairs, selected_depths, summarize
from test_ref_e0 import expect_failure


def test_statistics():
    assert selected_depths(36) == [0, 9, 18, 24, 30, 36]
    assert selected_depths(1) == [0, 1]
    expect_failure(lambda: selected_depths(0), AssertionError)
    a, b = [0, 0, 10, 10], [30, 0, 40, 10]
    boxes = [a, b, a]

    def record(identifier, gt, baseline, shallow):
        layers = {str(k): dict(decision=layer_result(boxes, [gt], baseline, score, score))
                  for k, score in ((0, shallow), (36, baseline))}
        return dict(id=identifier, image_name='same.jpg', query=identifier, answer_boxes=[gt],
                    boxes_sha256=canonical_hash(boxes), passed=True, layers=layers)

    records = [record('harm', a, [.8, .1, .2], [.1, .8, .2]),
               record('recover', b, [.8, .1, .2], [.1, .8, .2]),
               record('different_but_correct', a, [.8, .1, .2], [.1, .2, .8])]
    report, errors = summarize(records, [0, 36])
    row = report['curve'][0]
    assert row['correct'] == row['baseline_correct'] == 2
    assert row['harmed_count'] == row['recovered_count'] == 1 and row['delta_pp'] == 0
    assert row['harm_rate_all'] == 1/3 and row['harm_rate_given_baseline_correct'] == .5
    assert row['different_index_both_correct'] == 1 and row['index_disagreement_rate'] == 1
    assert errors['0'] == dict(harmed=['harm'], recovered=['recover'])
    assert report['curve'][-1]['harmed_count'] == report['curve'][-1]['recovered_count'] == 0
    pairs = query_pairs(records, [0, 36])
    assert pairs['pair_count'] == 2  # same-GT pair excluded
    assert pairs['curve'][0]['both_correct_rate'] == .5
    changed = deepcopy(records)
    changed[1]['boxes_sha256'] = 'changed'
    expect_failure(lambda: query_pairs(changed, [0, 36]), AssertionError)
    expect_failure(lambda: summarize(records + records, [0, 36]), AssertionError)
    changed = deepcopy(records)
    del changed[0]['layers']['0']
    expect_failure(lambda: summarize(changed, [0, 36]), AssertionError)
    saturated = layer_result(boxes, [a], [.8, .1, .2], [1., 1., 1.], [20., 30., 21.])
    assert saturated['actual_top1'] == 0 and saturated['raw_logit_top1'] == 1
    assert saturated['actual_top1_ties'] == 3 and saturated['sigmoid_vs_logit_top1_differs']
    assert saturated['actual_correct'] and not saturated['raw_logit_correct']
    empty_denominator = record('miss', [90, 90, 100, 100], [.8, .1, .2], [.1, .8, .2])
    zero, _ = summarize([empty_denominator], [0, 36])
    assert zero['curve'][0]['accuracy_given_covered'] is None
    assert zero['curve'][0]['harm_rate_given_baseline_correct'] is None
    assert query_pairs([empty_denominator], [0, 36])['curve'][0]['both_correct_rate'] is None
    for name in ('ref_pnative.py', 'ref_pnative_analysis.py', 'ref_pnative_core.py', 'test_ref_pnative.py'):
        ast.parse(Path(__file__).with_name(name).read_text(encoding='utf-8'))
    # Load only stdlib E0 manifest guard, no model or GPU initialization.
    from ref_pnative import check_e0_protocol
    import json
    e0_path = Path(__file__).resolve().parents[1] / 'results/ref_e0_refcocog_val60/manifest.json'
    if e0_path.is_file():
        reference = json.loads(e0_path.read_text(encoding='utf-8'))
        check_e0_protocol(reference, reference)
        wrong = deepcopy(reference)
        wrong['checkpoint_use_cache'] = not reference['checkpoint_use_cache']
        expect_failure(lambda: check_e0_protocol(wrong, reference), AssertionError)
        wrong = deepcopy(reference)
        key = next(iter(wrong['source_sha256']))
        wrong['source_sha256'][key] = 'modified'
        expect_failure(lambda: check_e0_protocol(wrong, reference), AssertionError)
    # Exercise actual JSON/CSV/Markdown reporting and exclusive creation.
    import ref_pnative
    for item in records:
        item['checks'] = dict(test_check=dict(max_abs=0.))
    with tempfile.TemporaryDirectory() as temporary:
        with patch.object(ref_pnative, 'OUTPUT', Path(temporary)):
            ref_pnative.write_reports(records, [0, 36])
            saved = json.loads((Path(temporary) / 'summary.json').read_text(encoding='utf-8'))
            assert saved['curve'][0]['harmed_count'] == 1
            assert (Path(temporary) / 'curve.csv').read_text(encoding='utf-8').count('\n') == 3
            assert (Path(temporary) / 'summary.md').is_file()
            expect_failure(lambda: ref_pnative.write_reports(records, [0, 36]), FileExistsError)
    print('P-native stdlib tests passed: layer plan, paired conservation, query pairs, saturation, invalid input.')


def test_real_boundaries():
    import torch
    import transformers
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    from ref_e0_core import forward_algorithm, test_algorithm
    from ref_pnative_core import capture_depths, read_depths
    assert transformers.__version__ == '4.57.1'
    torch.set_num_threads(1)  # small CPU fixtures; does not alter the inference process
    test_algorithm()
    torch.manual_seed(31)
    for dtype in (torch.float32, torch.bfloat16):
        config = Qwen3VLTextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
            num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            rope_scaling={'rope_type': 'default', 'mrope_section': [1, 1, 2]},
            use_cache=False, attention_dropout=0.)
        config._attn_implementation = 'eager'
        lm = Qwen3VLTextModel(config).to(dtype).eval()
        head = torch.nn.Linear(32, 1).to(dtype).eval()
        wrapper = SimpleNamespace(model=SimpleNamespace(language_model=lm))
        objects = torch.zeros(1, 12, dtype=torch.bool)
        objects[0, [9, 10]] = True
        visual = torch.zeros_like(objects)
        visual[0, [1, 2, 3]] = True
        embeddings = torch.randn(1, 12, 32).to(dtype)
        inputs = dict(inputs_embeds=embeddings, visual_pos_masks=visual,
            deepstack_visual_embeds=[torch.randn(3, 32).to(dtype) for _ in range(3)],
            position_ids=torch.arange(12).view(1, 1, 12).expand(3, 1, 12),
            attention_mask=torch.ones(1, 12, dtype=torch.long), use_cache=False)
        # Independently replay each actual decoder block, adding DeepStack explicitly.
        # This verifies off-by-one boundaries, including the first three additions.
        saved_kwargs = {}

        def remember(module, args, kwargs):
            saved_kwargs.update(kwargs)

        handle = lm.layers[0].register_forward_pre_hook(remember, with_kwargs=True)
        with torch.inference_mode():
            baseline = lm(**inputs).last_hidden_state
        handle.remove()
        depths = list(range(7))  # test all early injection boundaries, not only production points
        with torch.inference_mode():
            with capture_depths(wrapper, objects, depths) as captured:
                hooked = lm(**inputs).last_hidden_state
            torch.testing.assert_close(baseline, hooked, atol=0, rtol=0)
            hidden = embeddings.clone()
            expected = {0: hidden[objects].clone()}
            for index, block in enumerate(lm.layers):
                hidden = block(hidden, **saved_kwargs)
                if index < 3:
                    hidden = hidden.clone()
                    hidden[visual] += inputs['deepstack_visual_embeds'][index]
                expected[index + 1] = hidden[objects].clone()
            for k in depths:
                torch.testing.assert_close(captured['states'][k], expected[k], atol=0, rtol=0)
                assert captured['states'][k].shape == (2, 32)
            torch.testing.assert_close(captured['hL_full'], hidden, atol=0, rtol=0)
            outputs = read_depths(captured['states'], lm.norm.weight, lm.norm.variance_epsilon,
                                  head.weight, head.bias)
            with tempfile.TemporaryDirectory() as temporary:
                cache = Path(temporary) / 'sample.pt'
                torch.save(dict(depths=depths, hidden_pre_norm=torch.stack([captured['states'][k] for k in depths])), cache)
                restored = torch.load(cache, weights_only=True)
                assert restored['depths'] == depths and restored['hidden_pre_norm'].shape == (7, 2, 32)
                assert torch.equal(restored['hidden_pre_norm'][3], expected[3])
            full = forward_algorithm(captured['hL_full'], lm.norm.weight,
                                     lm.norm.variance_epsilon, head.weight, head.bias)
            torch.testing.assert_close(full, head(hooked), atol=0, rtol=0)
            torch.testing.assert_close(outputs[6], head(hooked)[objects][:, 0], atol=.03125, rtol=.01)
            old = captured['states'][3].clone()
            hidden.add_(99)
            assert torch.equal(old, captured['states'][3]), 'Cache aliases a live tensor'
            changed = dict(inputs)
            changed['inputs_embeds'] = embeddings.clone()
            changed['inputs_embeds'][:, 6] += torch.randn(32).to(dtype)
            with capture_depths(wrapper, objects, depths) as other:
                lm(**changed)
            assert torch.equal(captured['states'][0], other['states'][0])
            assert not torch.equal(captured['states'][6], other['states'][6])

        def intentional_failure():
            with capture_depths(wrapper, objects, depths):
                raise RuntimeError('cleanup test')

        expect_failure(intentional_failure, RuntimeError)
        assert all(not block._forward_pre_hooks for block in lm.layers)
        assert not lm._forward_pre_hooks and not lm.norm._forward_pre_hooks
        def invalid_depths():
            with capture_depths(wrapper, objects, [0, 2, 2, 6]):
                lm(**inputs)
        expect_failure(invalid_depths, AssertionError)
        bad = dict(captured['states'])
        bad[0] = bad[0][:1]
        expect_failure(lambda: read_depths(bad, lm.norm.weight, lm.norm.variance_epsilon,
                                         head.weight, head.bias), AssertionError)
        print(f'P-native real tiny Qwen3-VL boundaries/readout/cleanup passed: {dtype}')


if __name__ == '__main__':
    assert __debug__ and set(sys.argv[1:]) <= {'--transformers'}
    test_statistics()
    if '--transformers' in sys.argv:
        test_real_boundaries()
    else:
        print('NOT RUN: tensor/decoder tests; add --transformers. This is not a model experiment PASS.')
