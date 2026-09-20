"""CPU checks. --transformers runs actual tiny Qwen3-VL prefix/sham forwards."""
import ast
from copy import deepcopy
from pathlib import Path
import sys
import tempfile

from ref_exit_analysis import decisions, summarize, markdown
from test_ref_e0 import expect_failure


def test_statistics():
    boxes = [[0, 0, 10, 10], [20, 0, 30, 10]]
    gt = [boxes[1]]
    full = decisions(boxes, gt, [0., 1.], [.5, .73])
    early = decisions(boxes, gt, [10.5, 14.875], [1., 1.])
    assert full['bf16_sigmoid']['correct'] and full['raw_logit']['correct']
    assert not early['bf16_sigmoid']['correct'] and early['raw_logit']['correct']
    assert early['bf16_sigmoid']['top_ties'] == 2 and early['raw_logit']['top_ties'] == 1
    record = dict(id='a', image_name='a.jpg', passed=True, decisions=dict(full=full, exit=early),
                  checks={'example': {'max_abs': 0}}, timing={})
    for scope in ('forward', 'request'):
        record['timing'][scope] = [dict(arm=arm, wall_ms=ms)
                                  for ms in (10., 20., 30.) for arm in ('full', 'exit')]
        for row in record['timing'][scope]:
            if row['arm'] == 'exit':
                row['wall_ms'] /= 2
    summary, errors = summarize([record])
    assert summary['accuracy']['bf16_sigmoid']['harmed'] == 1
    assert summary['accuracy']['raw_logit']['harmed'] == 0
    assert errors['bf16_sigmoid']['harmed'] == ['a']
    assert summary['timing']['forward']['speedup_ratio'] == 2
    assert summary['timing']['request']['latency_reduction_percent'] == 50
    assert 'NOT an accuracy gain' in markdown(summary, 30)
    # Real exclusive JSON/Markdown output, without producing an experiment PASS.
    from humanref_pipeline import save_json
    from ref_pnative import read_json
    with tempfile.TemporaryDirectory() as temporary:
        target = Path(temporary) / 'summary.json'
        save_json(target, summary)
        assert read_json(target) == summary
        expect_failure(lambda: save_json(target, summary), FileExistsError)
        with (Path(temporary) / 'summary.md').open('x', encoding='utf-8') as stream:
            stream.write(markdown(summary, 30))
        assert '| raw_logit |' in (Path(temporary) / 'summary.md').read_text(encoding='utf-8')
    expect_failure(lambda: summarize([]), AssertionError)
    expect_failure(lambda: summarize([record, record]), AssertionError)
    broken = deepcopy(record)
    broken['timing']['forward'].pop()
    expect_failure(lambda: summarize([broken]), AssertionError)
    broken = deepcopy(record)
    broken['timing']['forward'][0]['wall_ms'] = float('nan')
    expect_failure(lambda: summarize([broken]), AssertionError)
    expect_failure(lambda: decisions(boxes, gt, [1.], [1.]), AssertionError)
    expect_failure(lambda: decisions(boxes, gt, [1., float('nan')], [1., 1.]), AssertionError)
    expect_failure(lambda: decisions([[10, 0, 0, 1], boxes[1]], gt, [1., 2.], [.5, .6]), AssertionError)
    for name in ('ref_exit.py', 'ref_exit_core.py', 'ref_exit_analysis.py', 'test_ref_exit.py'):
        ast.parse(Path(__file__).with_name(name).read_text(encoding='utf-8'))
    print('Static-exit paired statistics, saturation, timing, invalid-input tests passed.')


def test_transformers():
    import torch
    import transformers
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    from ref_exit_core import audit_execution, decoder_prefix, forward_algorithm, test_algorithm
    assert transformers.__version__ == '4.57.1'
    torch.set_num_threads(1)
    torch.manual_seed(79)
    test_algorithm()
    for dtype in (torch.float32, torch.bfloat16):
        for backend in ('eager', 'sdpa'):
            for use_cache in (False, True):
                config = Qwen3VLTextConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                    num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                    rope_scaling={'rope_type': 'default', 'mrope_section': [1, 1, 2]},
                    use_cache=use_cache, attention_dropout=0.)
                config._attn_implementation = backend
                lm = Qwen3VLTextModel(config).to(dtype).eval().requires_grad_(False)
                head = torch.nn.Linear(32, 1).to(dtype).eval().requires_grad_(False)
                objects = torch.zeros(1, 12, dtype=torch.bool)
                objects[0, [9, 10]] = True
                visual = torch.zeros_like(objects)
                visual[0, [1, 2, 3]] = True
                inputs = dict(inputs_embeds=torch.randn(1, 12, 32).to(dtype), visual_pos_masks=visual,
                    deepstack_visual_embeds=[torch.randn(3, 32).to(dtype) for _ in range(3)],
                    position_ids=torch.arange(12).view(1, 1, 12).expand(3, 1, 12),
                    attention_mask=torch.ones(1, 12, dtype=torch.long), use_cache=use_cache)
                layers = lm.layers
                config_before = deepcopy(lm.config.to_dict())
                state_before = {name: value.clone() for name, value in lm.state_dict().items()}
                with torch.inference_mode():
                    full = head(lm(**inputs).last_hidden_state)
                    assert full.shape == (1, 12, 1)
                    for depth in (3, 4, 5, 6):
                        with audit_execution(lm, head, depth, 6) as ref:
                            captured_full = head(lm(**inputs).last_hidden_state)
                        torch.testing.assert_close(full, captured_full, atol=0, rtol=0)
                        expected = forward_algorithm(ref['boundary_full'], lm.norm.weight,
                            lm.norm.variance_epsilon, head.weight, head.bias)
                        with audit_execution(lm, head, depth, depth) as audit:
                            with decoder_prefix(lm, depth, 3):
                                assert len(lm.layers) == depth and lm.config.num_hidden_layers == 6
                                actual = head(lm(**inputs).last_hidden_state)
                        assert audit['blocks'] == list(range(depth))
                        assert audit['deepstack_count'] == 3
                        assert audit['head_calls'] == audit['norm_calls'] == 1
                        torch.testing.assert_close(ref['boundary_full'], audit['final_full'], atol=0, rtol=0)
                        torch.testing.assert_close(expected, actual, atol=0, rtol=0)
                        assert actual[objects].shape == (2, 1)
                        assert lm.layers is layers
                        torch.testing.assert_close(head(lm(**inputs).last_hidden_state), full, atol=0, rtol=0)

                def intentional_failure():
                    with audit_execution(lm, head, 3, 3):
                        with decoder_prefix(lm, 3, 3):
                            raise RuntimeError('intentional cleanup failure')

                expect_failure(intentional_failure, RuntimeError)
                assert lm.layers is layers and lm.config.to_dict() == config_before
                assert all(not m._forward_hooks and not m._forward_pre_hooks for m in (head, *lm.modules()))
                assert state_before.keys() == lm.state_dict().keys()
                assert all(torch.equal(state_before[name], value) for name, value in lm.state_dict().items())

                def invalid_depth(depth):
                    with decoder_prefix(lm, depth, 3):
                        raise RuntimeError('Must reject before entering')

                for depth in (0, 2, 7):
                    expect_failure(lambda: invalid_depth(depth), AssertionError)
                print(f'Tiny Qwen3-VL prefix/sham/DeepStack/cleanup passed: {dtype}, {backend}, cache={use_cache}')


if __name__ == '__main__':
    assert __debug__ and set(sys.argv[1:]) <= {'--transformers'}
    test_statistics()
    if '--transformers' in sys.argv:
        test_transformers()
    else:
        print('NOT RUN: actual decoder tests; add --transformers. No production CUDA claim.')
