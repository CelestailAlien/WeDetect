"""Offline tests, including actual CPU tensor forwards through 36 tiny blocks.

No checkpoint/data downloads, GPU allocation, original experiment writes, or
production model forwards. Requires torch for the numerical tests. Run directly
with --stdlib-only only to check role partitioning and source syntax.
"""
import ast
from copy import deepcopy
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

from ref_sads_core import HeadIntervention, attention_statistics, partition_tokens


def fails(call, error=ValueError):
    try:
        call()
    except error:
        return
    raise AssertionError(f'Expected {error.__name__}')


class FakeTokenizer:
    unk_token_id = -1
    all_special_ids = [1, 2, 3, 4, 5, 6]
    pieces = {0: '<pad>', 1: '<|im_start|>', 2: '<|im_end|>',
              3: '<|vision_start|>', 4: '<|vision_end|>', 5: '<image>', 6: '<object>',
              7: 'user', 8: 'assistant', 9: 'system', 10: '\n',
              11: 'detect blue car', 12: 'system content', 13: 'tool', 14: 'user\n',
              15: 'user\ndetect'}

    def convert_tokens_to_ids(self, token):
        return next((k for k, v in self.pieces.items() if v == token), self.unk_token_id)

    def decode(self, ids, **kwargs):
        return ''.join(self.pieces[i] for i in ids)


def test_partition():
    tokenizer = FakeTokenizer()
    ids = [1, 7, 10, 3, 5, 5, 4, 11, 2, 10, 1, 8, 10, 6, 6, 2, 10, 0]
    mask = [1] * 17 + [0]
    actual = partition_tokens([ids], [mask], tokenizer, 5, 6)
    assert actual['groups'] == list('RRRRGGRTRRRRROORRP')
    assert actual['counts'] == dict(G=2, O=2, T=1, S=0, R=12, P=1)
    assert actual['roles'] == ['user', 'assistant']
    prefix = [1, 9, 10, 12, 2, 10]
    with_system = partition_tokens(prefix + ids, [1] * len(prefix) + mask, tokenizer, 5, 6)
    assert with_system['counts']['S'] == 1 and with_system['roles'][0] == 'system'
    merged = [1, 14] + ids[3:]
    assert partition_tokens(merged, [1] * (len(merged) - 1) + [0], tokenizer, 5, 6)['counts']['G'] == 2
    fails(lambda: partition_tokens([1, 13, 10] + ids[3:], mask, tokenizer, 5, 6))
    fails(lambda: partition_tokens([1, 15] + ids[3:], mask[1:], tokenizer, 5, 6))
    fails(lambda: partition_tokens(ids[:-3], mask[:-3], tokenizer, 5, 6))
    fails(lambda: partition_tokens([ids, ids], [mask, mask], tokenizer, 5, 6))
    bad = ids.copy()
    bad[13] = 11
    fails(lambda: partition_tokens(bad, mask, tokenizer, 5, 6))
    for filename in ('ref_sads_core.py', 'test_ref_sads_core.py'):
        ast.parse(Path(__file__).with_name(filename).read_text(encoding='utf-8'))
    print('Role parsing, missing system, padding, malformed template rejection: PASS')


def test_statistics():
    import torch
    torch.manual_seed(217)
    zero = torch.zeros(1, 4, 4, 2)
    records = attention_statistics(zero, zero[:, :2], list('RTGO'), 1., chunk_size=1, reference=True)
    entropy = -.625 * math.log(.625) - .375 * math.log(.375)
    row_entropy_mean = 3 * math.log(2) / 4
    for row in records:
        assert row['valid'] and row['query_count'] == row['valid_query_count'] == 4
        assert abs(row['x_G'] - (1 / 3 + .25) / 4) < 1e-7
        assert abs(row['x_O'] - .25 / 4) < 1e-7
        assert abs(row['H'] - entropy) < 1e-7
        assert abs(row['H'] - row_entropy_mean) > .1
        assert abs(row['e'] - entropy / math.log(2)) < 1e-7
        assert abs(sum(row['masses'].values()) - 1) < 1e-7
        assert row['object_queries']['query_count'] == 1
        assert abs(row['object_queries']['e'] - 1) < 1e-7
        assert row['diagnostics']['reference_max_abs'] < 1e-7
    q = torch.randn(1, 32, 19, 8)
    k = torch.randn(1, 8, 19, 8)
    groups = list('RRTT') + ['G'] * 5 + ['O'] * 6 + list('RRPP')
    assert len(groups) == 19
    chunked = attention_statistics(q, k, groups, 8 ** -.5, chunk_size=3, reference=True)
    full = attention_statistics(q, k, groups, 8 ** -.5, chunk_size=100)
    repeated = attention_statistics(q, k.repeat_interleave(4, 1), groups, 8 ** -.5, chunk_size=5)
    for a, b, c in zip(chunked, full, repeated):
        for key in ('x', 'x_G', 'x_O', 'H', 'e', 'mean_query_max_visual'):
            assert abs(a[key] - b[key]) < 2e-7 and abs(a[key] - c[key]) < 2e-7, key
        assert a['diagnostics']['masked_probability_max'] == 0
        assert a['valid_query_count'] == 17
    # Padded Q/K changes cannot affect active rows, even if padded keys are huge.
    changed_q, changed_k = q.clone(), k.clone()
    changed_q[:, :, -2:] = 1e4
    changed_k[:, :, -2:] = -1e4
    padded = attention_statistics(changed_q, changed_k, groups, 8 ** -.5)
    assert all(abs(a['H'] - b['H']) < 2e-7 for a, b in zip(full, padded))
    early_visual = attention_statistics(zero, zero[:, :2], list('GROT'), 1.)
    assert all(not r['valid'] and r['x_valid'] and not r['entropy_valid'] and
               r['valid_query_count'] == 3 and r['H'] is None for r in early_visual)
    one_nv = attention_statistics(zero, zero[:, :2], list('RGGO'), 1.)
    assert all(not r['valid'] and 'nonvisual_key_count_le_one' in r['reason'] for r in one_nv)
    invalid_q = zero.clone()
    invalid_q[:, 2, 1] = float('nan')
    invalid = attention_statistics(invalid_q, zero[:, :2], list('RTGO'), 1.)
    assert not invalid[2]['valid'] and invalid[2]['x'] is None and invalid[0]['valid']
    # Max(mean attention) must not become mean(max attention).
    q_peak, k_peak = zero.clone(), zero[:, :2].clone()
    q_peak[:, :, 2, 0], q_peak[:, :, 3, 1] = 1., 1.
    k_peak[:, :, 2, 0], k_peak[:, :, 3, 1] = 10., 10.
    peak = attention_statistics(q_peak, k_peak, list('RTGO'), 1.)[0]
    assert peak['mean_query_max_visual'] > 1.9 * peak['x']
    fails(lambda: attention_statistics(q, k[:, :3], groups, 1.))
    print('GQA, causal/padding, normalized entropy order, max order, invalid rows, chunk/reference: PASS')


def fake_model():
    import torch
    from torch import nn

    def rope(q, k, cos, sin, unsqueeze_dim=1):
        def rotate(x):
            return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), -1)
        cos, sin = cos.unsqueeze(unsqueeze_dim), sin.unsqueeze(unsqueeze_dim)
        return q * cos + rotate(q) * sin, k * cos + rotate(k) * sin

    class RecordingLinear(nn.Linear):
        def forward(self, value):
            self.last_input = value.detach().clone()
            return super().forward(value)

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(num_attention_heads=4, num_key_value_heads=2)
            self.head_dim, self.scaling, self.is_causal = 4, .5, True
            self.q_proj, self.k_proj, self.v_proj = nn.Linear(12, 16), nn.Linear(12, 8), nn.Linear(12, 8)
            self.q_norm, self.k_norm = nn.LayerNorm(4), nn.LayerNorm(4)
            self.o_proj = RecordingLinear(16, 12)

        def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, cache_position=None):
            b, s, _ = hidden_states.shape
            q = self.q_norm(self.q_proj(hidden_states).view(b, s, 4, 4)).transpose(1, 2)
            k = self.k_norm(self.k_proj(hidden_states).view(b, s, 2, 4)).transpose(1, 2)
            v = self.v_proj(hidden_states).view(b, s, 2, 4).transpose(1, 2)
            q, k = rope(q, k, *position_embeddings)
            scores = torch.matmul(q, k.repeat_interleave(2, 1).transpose(-1, -2)) * self.scaling
            allowed = torch.arange(s)[None] <= torch.arange(s)[:, None]
            if attention_mask is not None:
                allowed = allowed & attention_mask.bool()
            probability = scores.masked_fill(~allowed[None, None], -torch.inf).softmax(-1)
            output = torch.matmul(probability, v.repeat_interleave(2, 1)).transpose(1, 2).reshape(b, s, 16)
            return self.o_proj(output)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attention()

        def forward(self, hidden_states, **kwargs):
            return hidden_states + .03 * self.self_attn(hidden_states, **kwargs)

    class Language(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList(Block() for _ in range(36))
            self.config = SimpleNamespace(num_hidden_layers=36)

        def forward(self, inputs_embeds, position_ids, attention_mask=None, past_key_values=None):
            s = inputs_embeds.shape[1]
            angles = position_ids[0].float()[..., None].expand(1, s, 4) / 10
            value = inputs_embeds
            for layer in self.layers:
                value = layer(value, position_embeddings=(angles.cos(), angles.sin()),
                    attention_mask=attention_mask, cache_position=torch.arange(s), past_key_values=past_key_values)
            return value

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.language_model = Language()
            self.out_proj = nn.Linear(12, 1)

        def forward(self, **kwargs):
            return self.out_proj(self.model.language_model(**kwargs))

    return Model().eval().requires_grad_(False), rope


def test_hooks():
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(931)
    model, rope = fake_model()
    lm = model.model.language_model
    groups = list('RRTGGGOO')
    inputs = dict(inputs_embeds=torch.randn(1, 8, 12), position_ids=torch.arange(8).view(1, 1, 8).expand(3, 1, 8),
                  attention_mask=torch.ones(1, 8, dtype=torch.long))
    initial = deepcopy(model.state_dict())
    with torch.inference_mode():
        baseline = model(**inputs)
        original_concat = lm.layers[-1].self_attn.o_proj.last_input.clone().reshape(1, 8, 4, 4)
        with HeadIntervention(model, [28, 32, 36], groups=groups) as sham:
            actual = model(**inputs)
        torch.testing.assert_close(baseline, actual, atol=0, rtol=0)
        assert sham.executed_layers == list(range(1, 37))
        assert sham.diagnostics['gate_calls'] == {28: 1, 32: 1, 36: 1}
        assert sham.diagnostics['position_ids']['values'] == inputs['position_ids'].tolist()
        for amount in (0., .5):
            with HeadIntervention(model, [28, 32, 36], groups=groups, gates={36: {2: amount}}):
                changed = model(**inputs)
            concat = lm.layers[-1].self_attn.o_proj.last_input.reshape(1, 8, 4, 4)
            torch.testing.assert_close(concat[:, :, 0], original_concat[:, :, 0], atol=0, rtol=0)
            torch.testing.assert_close(concat[:, :, 2], original_concat[:, :, 2] * amount, atol=0, rtol=0)
            assert not torch.equal(changed, baseline)
        with patch('ref_sads_core._validated_rope', return_value=(rope, {'fake_test_only': True})):
            with HeadIntervention(model, [28, 32, 36], groups=groups, collect=True, reference=True, chunk_size=3) as captured:
                collector_output = model(**inputs)
        torch.testing.assert_close(collector_output, baseline, atol=0, rtol=0)
        assert set(captured.statistics) == {28, 32, 36}
        assert all(len(rows) == 4 and all(r['valid'] for r in rows) for rows in captured.statistics.values())
        assert all(d['q_shape'] == [1, 4, 8, 4] and d['k_shape'] == [1, 2, 8, 4]
                   for d in captured.diagnostics['layers'].values())
        assert all(d['position_embeddings_sha256'] for d in captured.diagnostics['layers'].values())

        def explode(module, args):
            raise RuntimeError('intentional model failure')

        handle = lm.layers[30].register_forward_pre_hook(explode)
        try:
            def interrupted():
                with patch('ref_sads_core._validated_rope', return_value=(rope, {})):
                    with HeadIntervention(model, [28, 32, 36], groups=groups, collect=True) as broken:
                        model(**inputs)
            fails(interrupted, RuntimeError)
        finally:
            handle.remove()
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
        assert all(torch.equal(initial[key], value) for key, value in model.state_dict().items())
        torch.testing.assert_close(model(**inputs), baseline, atol=0, rtol=0)

        def forbidden_gate():
            with HeadIntervention(model, [28, 32, 36], gates={36: {0: 0}}):
                model(**inputs)
        fails(forbidden_gate)

        def twice():
            with HeadIntervention(model, [28, 32, 36]):
                model(**inputs)
                model(**inputs)
        fails(twice, RuntimeError)

        def cached():
            with HeadIntervention(model, [28, 32, 36]):
                model(**inputs, past_key_values=SimpleNamespace(get_seq_length=lambda: 8))
        fails(cached)
        assert all(not m._forward_hooks and not m._forward_pre_hooks for m in model.modules())
    print('Actual fake 36-layer forwards: gate=1/0/.5, 4096-vs-hidden-style split, collector, shared head, cache rejection, exception cleanup: PASS')


def run_all():
    test_partition()
    test_statistics()
    test_hooks()


if __name__ == '__main__':
    assert __debug__ and set(sys.argv[1:]) <= {'--stdlib-only'}
    if '--stdlib-only' in sys.argv:
        test_partition()
        print('NOT RUN: torch numerical/fake-module tests. No production consistency claim.')
    else:
        run_all()
        print('Offline core tests passed. No real checkpoint or GPU forward was run.')
