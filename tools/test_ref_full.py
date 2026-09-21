"""No-download stdlib tests; --transformers adds random CPU model execution."""
import ast
from copy import deepcopy
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

from humanref_pipeline import iou, save_json
from ref_exit_analysis import decisions
from ref_full_analysis import select_rows, history_diagnostic, metrics, summarize, markdown
from ref_pnative import read_json
from test_ref_e0 import expect_failure


def fixture(identifier, image, gt, full, early, first=False):
    boxes = [[0, 0, 10, 10], [20, 0, 30, 10]]
    logits = dict(full=full[0], exit=early[0])
    scores = dict(full=full[1], exit=early[1])
    check = dict(passed=True, max_abs=0.)
    checks = {key: dict(check) for key in ('exit_boundary', 'exit_readout')}
    if first:
        checks.update({key: dict(check) for key in ('unhooked_full', 'full_depth_sham')})
    return dict(id=identifier, image_name=image, query=identifier, boxes=boxes, answer_boxes=[gt],
        logits=logits, scores=scores, decisions={arm: decisions(boxes, [gt], logits[arm], scores[arm]) for arm in logits},
        candidate_covers_gt=any(iou(b, gt) >= .5 for b in boxes), history=None,
        executed_full=list(range(36)), executed_exit=list(range(30)), norm_calls_exit=1, head_calls_exit=1,
        checks=checks, boundary_ranking_equal=dict(raw_logit=True, bf16_sigmoid=True),
        control_checked=first, passed=True)


def test_statistics():
    from ref_full import write_reports, load_reference
    rows = [dict(id=k) for k in ('a', 'b', 'c')]
    assert select_rows(rows, 0) == rows and select_rows(rows, 1) == rows[:1]
    assert select_rows(rows, 3) == rows
    expect_failure(lambda: select_rows(rows, 4), AssertionError)
    expect_failure(lambda: select_rows(rows, -1), AssertionError)
    expect_failure(lambda: select_rows(rows + rows, 0), AssertionError)
    gt = [20, 0, 30, 10]
    records = [fixture('a', 'old.jpg', gt, ([0., 1.], [.5, .73]), ([10.5, 14.875], [1., 1.]), True),
               fixture('b', 'old.jpg', gt, ([2., 1.], [.88, .73]), ([0., 3.], [.5, .95])),
               fixture('c', 'new.jpg', [50, 50, 60, 60], ([0., 1.], [.5, .73]), ([0., 1.], [.5, .73]))]
    previous = deepcopy(records[0])
    previous['logits']['exit'] = [15., 14.875]  # Historical winner differs; not a local failure.
    previous['decisions']['exit'] = decisions(previous['boxes'], previous['answer_boxes'],
                                             previous['logits']['exit'], previous['scores']['exit'])
    records[0]['history'] = history_diagnostic(records[0], previous)
    history = records[0]['history']['exit']
    assert history['max_abs'] == 4.5 and history['protocols']['raw_logit']['index_changed']
    assert records[0]['passed'], 'Historical drift must not rewrite same-run validity'
    bad = deepcopy(previous)
    bad['query'] = 'changed input'
    expect_failure(lambda: history_diagnostic(records[0], bad), AssertionError)
    ids, prior, images = ['a', 'b', 'c'], ['a'], ['old.jpg']
    report, errors = summarize(records, ids, ids, prior, images)
    assert report['full_split'] and report['evaluation_scope'] == 'full_validation'
    assert not report['timing_measured'] and 'timing' not in report
    assert report['accuracy']['bf16_sigmoid']['harmed'] == report['accuracy']['bf16_sigmoid']['recovered'] == 1
    assert report['accuracy']['raw_logit']['full_correct'] == 1
    assert report['accuracy']['raw_logit']['exit_correct'] == 2
    assert report['candidate_coverage'] == 2/3
    assert report['groups']['previous_expressions']['samples'] == 1
    assert report['groups']['new_expressions']['samples'] == 2
    assert report['groups']['new_images']['samples'] == 1
    assert report['groups']['new_images']['accuracy']['raw_logit']['exit_accuracy_given_covered'] is None
    assert report['history']['arms']['exit']['index_changes']['raw_logit'] == 1
    assert errors['bf16_sigmoid'] == dict(harmed=['a'], recovered=['b'])
    smoke, _ = summarize(records[:1], ['a'], ids, prior, images)
    assert not smoke['full_split'] and smoke['evaluation_scope'] == 'smoke_only'
    assert smoke['groups']['new_expressions']['accuracy'] is None
    for bad_records in (records[:2], records[::-1], records + records[:1]):
        expect_failure(lambda: summarize(bad_records, ids, ids, prior, images), AssertionError)
    for key, value in (('passed', False), ('executed_exit', list(range(31))), ('candidate_covers_gt', False)):
        bad_records = deepcopy(records)
        bad_records[0][key] = value
        expect_failure(lambda: summarize(bad_records, ids, ids, prior, images), AssertionError)
    bad_records = deepcopy(records)
    bad_records[0]['checks']['exit_boundary']['passed'] = False
    expect_failure(lambda: summarize(bad_records, ids, ids, prior, images), AssertionError)
    bad_records = deepcopy(records)
    bad_records[0]['decisions']['exit']['raw_logit']['index'] = 0
    expect_failure(lambda: summarize(bad_records, ids, ids, prior, images), AssertionError)
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary)
        (output / 'samples').mkdir()
        for index, record in enumerate(records):
            save_json(output / 'samples' / f'{index:05d}.json', record)
        written = write_reports(output, records, ids, ids, prior, images)
        assert written == read_json(output / 'summary.json')
        assert read_json(output / 'paired_errors.json') == errors
        assert len(read_json(output / 'sample_hashes.json')) == 3
        assert 'No latency measurement' in (output / 'summary.md').read_text(encoding='utf-8')
        expect_failure(lambda: write_reports(output, records, ids, ids, prior, images), FileExistsError)
    assert 'smoke_only' in markdown(smoke)
    # Read actual earlier results as a regression fixture, if available locally.
    directory = Path(__file__).resolve().parents[1] / 'results/ref_exit_d30_refcocog_a6000_val500'
    if directory.is_dir():
        _, reference, _ = load_reference(directory)
        actual = [dict(r, candidate_covers_gt=any(iou(b, r['answer_boxes'][0]) >= .5 for b in r['boxes']))
                  for r in reference.values()]
        result = metrics(actual)
        expected = read_json(directory / 'summary.json')['accuracy']
        for protocol, expected_values in expected.items():
            for key, value in expected_values.items():
                assert math.isclose(result['accuracy'][protocol][key], value, abs_tol=1e-12), (protocol, key)
        print(f'Recomputed prior {len(actual)} records: both ranking protocols match stored summary.')
    for name in ('ref_full.py', 'ref_full_core.py', 'ref_full_analysis.py', 'test_ref_full.py'):
        ast.parse(Path(__file__).with_name(name).read_text(encoding='utf-8'))
    print('Full-eval selection/coverage/grouping/history/report/invalid-input tests passed.')


def test_tensors():
    import torch
    import transformers
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    from ref_full_core import evaluate_pair
    from ref_e0_core import test_algorithm
    assert transformers.__version__ == '4.57.1'
    torch.set_num_threads(1)
    test_algorithm()
    torch.manual_seed(97)
    for dtype in (torch.float32, torch.bfloat16):
        for backend in ('eager', 'sdpa'):
            for cache in (False, True):
                config = Qwen3VLTextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                    num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                    rope_scaling={'rope_type': 'default', 'mrope_section': [1, 1, 2]},
                    use_cache=cache, attention_dropout=0.)
                config._attn_implementation = backend
                lm = Qwen3VLTextModel(config).to(dtype).eval().requires_grad_(False)
                head = torch.nn.Linear(32, 1).to(dtype).eval().requires_grad_(False)

                def model(**kwargs):
                    return SimpleNamespace(logits=head(lm(**kwargs).last_hidden_state))

                model.model, model.out_proj = SimpleNamespace(language_model=lm), head
                objects = torch.zeros(1, 12, dtype=torch.bool)
                objects[0, [9, 10]] = True
                visual = torch.zeros_like(objects)
                visual[0, [1, 2, 3]] = True
                inputs = dict(inputs_embeds=torch.randn(1, 12, 32).to(dtype), visual_pos_masks=visual,
                    deepstack_visual_embeds=[torch.randn(3, 32).to(dtype) for _ in range(3)],
                    position_ids=torch.arange(12).view(1, 1, 12).expand(3, 1, 12),
                    attention_mask=torch.ones(1, 12, dtype=torch.long), use_cache=cache)
                original = lm.layers
                with torch.inference_mode():
                    baseline = model(**inputs).logits[objects][:, 0]
                    values, info = evaluate_pair(model, inputs, objects, 4, 1e-5, 1e-5, control=True)
                    assert all(v.shape == (2,) and v.dtype == dtype for v in values.values())
                    assert info['passed'] and info['control_checked']
                    assert all(c['exact'] for c in info['checks'].values())
                    assert info['executed_full'] == list(range(6)) and info['executed_exit'] == list(range(4))
                    torch.testing.assert_close(values['full'], baseline, atol=0, rtol=0)
                    other, minimal = evaluate_pair(model, inputs, objects, 4, 1e-5, 1e-5)
                    assert minimal['passed'] and set(minimal['checks']) == {'exit_boundary', 'exit_readout'}
                    torch.testing.assert_close(values['exit'], other['exit'], atol=0, rtol=0)
                    expect_failure(lambda: evaluate_pair(model, inputs, objects[:, :11], 4, 1e-5, 1e-5), AssertionError)
                assert lm.layers is original
                assert all(not m._forward_pre_hooks and not m._forward_hooks for m in (head, *lm.modules()))
                print(f'Full-eval real paired runner passed: {dtype}, {backend}, cache={cache}')


if __name__ == '__main__':
    assert __debug__ and set(sys.argv[1:]) <= {'--transformers'}
    test_statistics()
    if '--transformers' in sys.argv:
        test_tensors()
    else:
        print('NOT RUN: tensor/decoder tests; add --transformers. Not a production evaluation PASS.')
