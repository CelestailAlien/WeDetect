"""Read-only Q/K statistics and temporary output-head gates for frozen Ref.

This module never loads a checkpoint and never reads labels. The production
collector is pinned to the audited Transformers 4.57.1 Qwen3-VL implementation.
Only scalar statistics and positional audit information survive the context.
"""
import ast
from collections import Counter
import hashlib
import inspect
import json
import math
import textwrap


GROUPS = 'GOTSRP'


def _sequence(value):
    value = value.detach().cpu().tolist() if hasattr(value, 'detach') else value
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError('Only batch size 1 is supported')
        value = value[0]
    return list(value)


def _groups(value):
    result = list(value['groups'] if isinstance(value, dict) else value)
    if not result or any(g not in GROUPS for g in result):
        raise ValueError('Expected an exhaustive G/O/T/S/R/P partition')
    return result


def partition_tokens(input_ids, attention_mask, tokenizer, image_token_id, object_token_id):
    """Parse actual chat roles; padding and structural tokens have priority.

    Role header text (including its terminating newline) is structural. Ordinary
    user text is T; ordinary system text is S. The present Ref assistant payload
    must contain only object tokens and whitespace. Unknown layouts fail closed.
    No system message is inserted or inferred from an absolute token position.
    """
    ids, mask = _sequence(input_ids), _sequence(attention_mask)
    if not ids or len(ids) != len(mask) or any(x not in (0, 1, False, True) for x in mask):
        raise ValueError('Expected aligned token IDs and binary attention mask')
    start = tokenizer.convert_tokens_to_ids('<|im_start|>')
    end = tokenizer.convert_tokens_to_ids('<|im_end|>')
    if start == end or start is None or end is None or start == tokenizer.unk_token_id or end == tokenizer.unk_token_id:
        raise ValueError('Qwen chat boundary tokens are required')
    if len({start, end, image_token_id, object_token_id}) != 4:
        raise ValueError('Content and structural token IDs must be distinct')
    specials = set(tokenizer.all_special_ids) - {image_token_id, object_token_id}
    specials.update((start, end))

    def decode(tokens):
        return tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)

    result, roles = [], []
    role, header = None, None
    for token, active in zip(ids, mask):
        if not active:
            result.append('P')
            continue
        if token == start:
            if role is not None or header is not None:
                raise ValueError('Nested or unterminated chat message')
            header = []
            result.append('R')
            continue
        if header is not None:
            if token in specials or token in (image_token_id, object_token_id):
                raise ValueError('Special token inside chat role header')
            header.append(token)
            text = decode(header)
            result.append('R')
            if '\n' in text:
                name, tail = text.split('\n', 1)
                if name not in ('user', 'assistant', 'system') or tail:
                    raise ValueError('Unknown role or header/content token boundary')
                role, header = name, None
                roles.append(name)
            continue
        if token == end:
            if role is None:
                raise ValueError('Unmatched chat end token')
            role = None
            result.append('R')
        elif token == image_token_id:
            if role != 'user':
                raise ValueError('Expected image content inside user message')
            result.append('G')
        elif token == object_token_id:
            if role != 'assistant':
                raise ValueError('Expected object content inside assistant message')
            result.append('O')
        elif token in specials:
            result.append('R')
        elif role == 'user':
            result.append('T')
        elif role == 'system':
            result.append('S')
        elif decode([token]).strip() == '':
            result.append('R')
        else:
            raise ValueError('Unexpected ordinary text outside user/system content')
    if header is not None or role is not None:
        raise ValueError('Incomplete chat template')
    counts = {g: result.count(g) for g in GROUPS}
    if not counts['G'] or not counts['O'] or not counts['T']:
        raise ValueError('Expected image, object and user text tokens')
    assert sum(counts.values()) == len(ids)
    return dict(groups=result, counts=counts, roles=roles, visual='GO', nonvisual='TSR')


def _statistics_for_queries(q, k, groups, scale, query_indices, chunk_size, min_valid_fraction, reference):
    import torch

    _, heads, length, _ = q.shape
    device = q.device
    keys = torch.arange(length, device=device)
    masks = {g: torch.tensor([v == g for v in groups], device=device) for g in GROUPS}
    visual = masks['G'] | masks['O']
    nonvisual = masks['T'] | masks['S'] | masks['R']
    nv_count, q_count = int(nonvisual.sum()), len(query_indices)
    average = torch.zeros(heads, length, device=device, dtype=torch.float64)
    conditional = torch.zeros(heads, length, device=device, dtype=torch.float64)
    maximum_sum = torch.zeros(heads, device=device, dtype=torch.float64)
    valid_count = torch.zeros(heads, device=device, dtype=torch.long)
    row_error = torch.zeros(heads, device=device)
    nonfinite = torch.zeros(heads, device=device, dtype=torch.bool)
    reference_error = torch.zeros(heads, device=device)
    repeated_k = k[0].float().repeat_interleave(heads // k.shape[1], dim=0)
    selected = torch.tensor(query_indices, dtype=torch.long, device=device)
    # The reference uses the SAME Q/K and only a bounded subset of query rows.
    # It is never another model forward and never allocates an S by S tensor.
    reference_rows = selected[:min(8, q_count)] if reference else selected[:0]
    reference_probs = None
    if reference_rows.numel():
        logits = torch.einsum('hqd,hkd->hqk', q[0, :, reference_rows].float(), repeated_k) * scale
        allowed = (keys[None, :] <= reference_rows[:, None]) & ~masks['P'][None, :]
        reference_probs = logits.masked_fill(~allowed[None], -torch.inf).softmax(-1)

    for begin in range(0, q_count, chunk_size):
        indices = selected[begin:begin + chunk_size]
        logits = torch.matmul(q[0, :, indices].float(), repeated_k.transpose(-1, -2)) * scale
        allowed = (keys[None, :] <= indices[:, None]) & ~masks['P'][None, :]
        probabilities = logits.masked_fill(~allowed[None], -torch.inf).softmax(-1)
        finite = torch.isfinite(probabilities).all(-1)
        nonfinite |= ~finite.all(-1)
        probabilities = torch.where(torch.isfinite(probabilities), probabilities, 0.)
        if probabilities.masked_select(~allowed[None].expand_as(probabilities)).count_nonzero():
            raise AssertionError('Causal/padding masked probability is nonzero')
        error = (probabilities.sum(-1) - 1).abs()
        row_error = torch.maximum(row_error, error.max(-1).values)
        if (error[finite] > 2e-6).any():
            raise AssertionError('Attention row does not normalize to one')
        average += probabilities.double().sum(1)
        if visual.any():
            maximum_sum += probabilities[..., visual].max(-1).values.double().sum(-1)
        denominator = probabilities[..., nonvisual].sum(-1)
        visible_nv = (allowed & nonvisual[None]).any(-1)
        valid = finite & visible_nv[None] & (denominator > 1e-12)
        valid_count += valid.sum(-1)
        renormalized = probabilities / denominator.clamp_min(1e-12)[..., None]
        conditional += (renormalized * valid[..., None] * nonvisual[None, None]).double().sum(1)
        if reference_probs is not None and begin < len(reference_rows):
            size = min(len(indices), len(reference_rows) - begin)
            delta = (probabilities[:, :size] - reference_probs[:, begin:begin + size]).abs()
            reference_error = torch.maximum(reference_error, delta.flatten(1).max(-1).values)

    results = []
    for head in range(heads):
        reasons = []
        if not q_count:
            reasons.append('no_queries')
        if not visual.any():
            reasons.append('no_visual_keys')
        if nv_count <= 1:
            reasons.append('nonvisual_key_count_le_one')
        if bool(nonfinite[head]):
            reasons.append('nonfinite_attention')
        count = int(valid_count[head])
        if q_count and count / q_count < min_valid_fraction:
            reasons.append('insufficient_valid_query_fraction')
        avg = average[head] / max(q_count, 1)
        p = conditional[head] / max(count, 1)
        entropy = -(p[p > 0] * p[p > 0].log()).sum()
        entropy_good = not any(r != 'no_visual_keys' for r in reasons)
        if entropy_good and (not torch.isfinite(entropy) or float(entropy) < -1e-7 or float(entropy) > math.log(nv_count) + 1e-6):
            reasons.append('invalid_entropy')
            entropy_good = False
        xg = float(avg[masks['G']].max()) if masks['G'].any() and q_count else None
        xo = float(avg[masks['O']].max()) if masks['O'].any() and q_count else None
        chosen = max(((xg, 'G'), (xo, 'O')), key=lambda pair: -1 if pair[0] is None else pair[0])
        trustworthy = q_count > 0 and not bool(nonfinite[head])
        x_valid = bool(trustworthy and visual.any())
        ref_error = float(reference_error[head]) if reference else None
        if ref_error is not None and ref_error > 2e-6:
            raise AssertionError(f'Q/K reference mismatch: {ref_error}')
        results.append(dict(head=head, x=chosen[0] if trustworthy else None,
            x_G=xg if trustworthy else None, x_O=xo if trustworthy else None,
            max_group=chosen[1] if chosen[0] is not None and trustworthy else None,
            mean_query_max_visual=float(maximum_sum[head] / q_count) if trustworthy and visual.any() else None,
            H=float(entropy) if entropy_good else None, e=float(entropy) / math.log(nv_count) if entropy_good else None,
            valid=bool(x_valid and entropy_good), x_valid=x_valid, entropy_valid=entropy_good,
            reason=';'.join(reasons) if reasons else None,
            masses={g: float(avg[masks[g]].sum()) if trustworthy else None for g in 'GOTSR'},
            query_count=q_count, valid_query_count=count, nonvisual_key_count=nv_count,
            diagnostics=dict(row_sum_max_abs_error=float(row_error[head]), masked_probability_max=0.,
                reference_max_abs=ref_error, reference_query_count=len(reference_rows))))
    return results


def attention_statistics(q, k, groups, scale, chunk_size=128, min_valid_fraction=.99, reference=False):
    """Streaming FP32 causal probabilities; entropy after mean conditional rows.

    Q/K must already include the model's Q/K norm and actual RoPE. GQA repeats
    each KV head contiguously. Main queries are all nonpadding positions;
    object-only statistics are diagnostic and never replace the main values.
    """
    groups = _groups(groups)
    if q.ndim != 4 or k.ndim != 4 or q.shape[0] != 1 or k.shape[0] != 1:
        raise ValueError('Expected Q/K [1,H,S,D]')
    if q.shape[2:] != k.shape[2:] or q.shape[2] != len(groups) or q.shape[1] % k.shape[1]:
        raise ValueError('Incompatible sequence, head dimension or GQA mapping')
    if q.device != k.device or not q.is_floating_point() or not k.is_floating_point():
        raise ValueError('Expected colocated floating point Q/K')
    if chunk_size < 1 or not 0 < min_valid_fraction <= 1 or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid statistics parameters')
    import torch
    with torch.no_grad():
        rows = _statistics_for_queries(q, k, groups, scale,
            [i for i, g in enumerate(groups) if g != 'P'], chunk_size, min_valid_fraction, reference)
        objects = _statistics_for_queries(q, k, groups, scale,
            [i for i, g in enumerate(groups) if g == 'O'], chunk_size, min_valid_fraction, reference)
    for row, obj in zip(rows, objects):
        row['object_queries'] = obj
    return rows


def _body(source):
    node = ast.parse(textwrap.dedent(source)).body[0]
    body = node.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body = body[1:]
    return ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)


def _validated_rope():
    """Validate the installed implementation against the reviewed HF release."""
    import transformers
    if transformers.__version__ != '4.57.1':
        raise RuntimeError('SADS collector requires audited Transformers 4.57.1')
    from transformers.models.qwen3_vl import modeling_qwen3_vl as hf
    expected = '''def apply_rotary_pos_emb():
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed
    '''
    rotate = '''def rotate_half():
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    '''
    if _body(inspect.getsource(hf.apply_rotary_pos_emb)) != _body(expected) or _body(inspect.getsource(hf.rotate_half)) != _body(rotate):
        raise RuntimeError('Installed Qwen3-VL rotary implementation differs from audited source')
    source = inspect.getsource(hf.Qwen3VLTextAttention.forward)
    required = ('self.q_norm(self.q_proj(hidden_states).view(hidden_shape))',
                'self.k_norm(self.k_proj(hidden_states).view(hidden_shape))',
                'apply_rotary_pos_emb(query_states, key_states, cos, sin)',
                'attn_output.reshape(*input_shape, -1).contiguous()', 'self.o_proj(attn_output)')
    if not all(text in source for text in required):
        raise RuntimeError('Installed Qwen3-VL attention layout differs from audited source')
    path = inspect.getfile(hf)
    with open(path, 'rb') as stream:
        digest = hashlib.sha256(stream.read()).hexdigest()
    return hf.apply_rotary_pos_emb, dict(transformers='4.57.1', source=path,
        source_sha256=digest, rotary_body_verified=True, attention_layout_verified=True)


def _tensor_audit(tensor):
    if tensor is None:
        return None
    record = dict(shape=list(tensor.shape), dtype=str(tensor.dtype), values=tensor.detach().cpu().tolist())
    record['sha256'] = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return record


def _tensor_sha256(tensor):
    import torch
    digest = hashlib.sha256(str((tuple(tensor.shape), str(tensor.dtype))).encode())
    digest.update(bytes(tensor.detach().contiguous().cpu().view(torch.uint8).reshape(-1).tolist()))
    return digest.hexdigest()


class HeadIntervention:
    """Temporary gates and optional fresh-prefill collector, exactly one forward.

    Layer IDs are one based; head IDs are zero based. Every requested layer
    passes through the multiplication even when all gates equal one. The first
    shared head is always retained. Entry/exit does not alter weights or config.
    """
    def __init__(self, model, layers, groups=None, gates=None, collect=False, chunk_size=128, reference=False):
        self.model = model
        self.lm = model.model.language_model
        self.layers = list(layers)
        self.groups = _groups(groups) if groups is not None else None
        self.gates = {} if gates is None else gates
        self.collect, self.chunk_size, self.reference = collect, chunk_size, reference
        self.statistics, self.diagnostics, self.executed_layers = {}, {}, []
        self._handles, self._pending = [], {}
        self._entered, self._calls = False, 0
        self._gate_calls = Counter()

    @staticmethod
    def _arguments(module, args, kwargs):
        return inspect.signature(module.forward).bind_partial(*args, **kwargs).arguments

    def _language_pre(self, module, args, kwargs):
        import torch
        bound = self._arguments(module, args, kwargs)
        self._calls += 1
        if self._calls != 1:
            raise RuntimeError('One context permits exactly one fresh prefill')
        past = bound.get('past_key_values')
        if past is not None and past.get_seq_length() != 0:
            raise ValueError('Past cache reuse is forbidden')
        hidden = bound.get('inputs_embeds')
        tokens = bound.get('input_ids')
        shape = hidden.shape[:2] if hidden is not None else tokens.shape
        if len(shape) != 2 or shape[0] != 1 or shape[1] < 2:
            raise ValueError('Expected batch-one fresh prefill')
        if self.groups is not None and len(self.groups) != shape[1]:
            raise ValueError('Token partition and model input length disagree')
        mask = bound.get('attention_mask')
        if mask is not None:
            if mask.ndim != 2 or tuple(mask.shape) != tuple(shape) or not ((mask == 0) | (mask == 1)).all():
                raise ValueError('Original language input mask must be binary [1,S]')
            if self.groups is not None and mask[0].bool().tolist() != [g != 'P' for g in self.groups]:
                raise ValueError('Padding partition disagrees with actual input mask')
        elif self.groups is not None and 'P' in self.groups:
            raise ValueError('Padding requires the actual attention mask')
        position_ids = bound.get('position_ids')
        if self.collect and position_ids is None:
            raise ValueError('Collector requires actual language position_ids')
        self.diagnostics.update(position_ids=_tensor_audit(position_ids),
            attention_mask=_tensor_audit(mask), sequence_length=int(shape[1]),
            fresh_prefill=True, cache_reuse=False)

    def _attention_pre(self, layer, module, args, kwargs):
        import torch
        bound = self._arguments(module, args, kwargs)
        cos, sin = bound['position_embeddings']
        if layer in self._pending or layer in self.statistics:
            raise RuntimeError('Attention layer repeated in a single prefill')
        if not module.is_causal:
            raise ValueError('Only causal text attention is supported')
        length = len(self.groups)
        if cos.shape != sin.shape or tuple(cos.shape) != (1, length, module.head_dim):
            raise ValueError('Unexpected rotary cos/sin shape')
        position = bound.get('cache_position')
        if position is not None and (position.ndim != 1 or len(position) != length or
                not torch.equal(position, torch.arange(length, device=position.device))):
            raise ValueError('Collector requires fresh contiguous cache_position')
        past = bound.get('past_key_values')
        if past is not None and past.get_seq_length(layer - 1) != 0:
            raise ValueError('Target layer already has cached keys')
        mask = bound.get('attention_mask')
        active = torch.tensor([g != 'P' for g in self.groups], device=cos.device)
        if mask is not None:
            if mask.ndim == 2:
                if tuple(mask.shape) != (1, length) or not torch.equal(mask[0].bool(), active):
                    raise ValueError('Attention padding mask differs from partition')
            elif mask.ndim == 4:
                if tuple(mask.shape) != (1, 1, length, length):
                    raise ValueError('Unsupported attention mask shape')
                allowed = mask[0, 0] if mask.dtype == torch.bool else mask[0, 0] == 0
                causal = torch.arange(length, device=cos.device)[None] <= torch.arange(length, device=cos.device)[:, None]
                expected = causal & active[None]
                if not torch.equal(allowed[active], expected[active]):
                    raise ValueError('Actual attention mask has nonstandard visibility')
            else:
                raise ValueError('Unsupported attention mask rank')
        elif not active.all():
            raise ValueError('Attention lost its padding mask')
        self._pending[layer] = dict(cos=cos.detach(), sin=sin.detach())
        self.diagnostics.setdefault('layers', {})[layer] = dict(
            causal=True, actual_mask_verified=True, cache_position=_tensor_audit(position),
            position_embeddings_sha256=hashlib.sha256(json.dumps(
                [_tensor_sha256(cos), _tensor_sha256(sin)]).encode()).hexdigest())

    def _norm_capture(self, layer, name, output):
        if layer not in self._pending or name in self._pending[layer]:
            raise RuntimeError('Unexpected Q/K norm call order')
        attention = self.lm.layers[layer - 1].self_attn
        expected_heads = (attention.config.num_attention_heads if name == 'q'
                          else attention.config.num_key_value_heads)
        if tuple(output.shape) != (1, len(self.groups), expected_heads, attention.head_dim):
            raise ValueError('Actual Q/K norm output disagrees with attention configuration')
        self._pending[layer][name] = output.detach()

    def _gate(self, layer, module, args):
        import torch
        if len(args) != 1:
            raise ValueError('Expected original positional o_proj input')
        value = args[0]
        attention = self.lm.layers[layer - 1].self_attn
        heads, dim = attention.config.num_attention_heads, attention.head_dim
        if value.ndim != 3 or value.shape[0] != 1 or value.shape[-1] != heads * dim:
            raise ValueError('Head concat shape changed; do not use hidden_size/heads')
        if self.collect:
            saved = self._pending.pop(layer)
            q, k = self._rope(saved['q'].transpose(1, 2), saved['k'].transpose(1, 2), saved['cos'], saved['sin'])
            self.statistics[layer] = attention_statistics(q, k, self.groups,
                attention.scaling, self.chunk_size, reference=self.reference)
            self.diagnostics['layers'][layer]['q_shape'] = list(q.shape)
            self.diagnostics['layers'][layer]['k_shape'] = list(k.shape)
            self.diagnostics['layers'][layer]['gqa_groups'] = heads // k.shape[1]
            self.diagnostics['layers'][layer]['row_sum_max_abs_error'] = max(
                r['diagnostics']['row_sum_max_abs_error'] for r in self.statistics[layer])
            del saved, q, k
        weights = torch.ones(heads, device=value.device, dtype=value.dtype)
        for head, gate in self.gates.get(layer, {}).items():
            weights[head] = gate
        assert weights[0] == 1
        self._gate_calls[layer] += 1
        # Deliberately no all-ones shortcut: the sham must exercise this path.
        result = (value.reshape(*value.shape[:-1], heads, dim) * weights[None, None, :, None]).reshape_as(value)
        return (result,)

    def __enter__(self):
        if self._entered:
            raise RuntimeError('HeadIntervention contexts cannot be reused')
        self._entered = True
        if len(self.lm.layers) != 36 or self.lm.config.num_hidden_layers != 36:
            raise ValueError('This protocol requires all 36 decoder layers')
        if not self.layers or len(set(self.layers)) != len(self.layers) or any(type(x) is not int or not 1 <= x <= 36 for x in self.layers):
            raise ValueError('Layer IDs must be unique and one based in [1,36]')
        if set(self.gates) - set(self.layers):
            raise ValueError('A gate refers to a layer without an installed interface')
        if any(m.training for m in self.model.modules()) or any(p.requires_grad for p in self.model.parameters()):
            raise ValueError('Intervention requires eval mode and frozen weights')
        if self.collect and self.groups is None:
            raise ValueError('Collector requires the actual token partition')
        if self.chunk_size < 1:
            raise ValueError('chunk_size must be positive')
        for layer in self.layers:
            heads = self.lm.layers[layer - 1].self_attn.config.num_attention_heads
            for head, gate in self.gates.get(layer, {}).items():
                if type(head) is not int or not 0 <= head < heads or gate not in (0, .5, 1):
                    raise ValueError('Invalid query head index or gate')
                if head == 0 and gate != 1:
                    raise ValueError('Shared head 0 must always be retained')
        try:
            if self.collect:
                self._rope, audit = _validated_rope()
                self.diagnostics['implementation'] = audit
            self._handles.append(self.lm.register_forward_pre_hook(self._language_pre, with_kwargs=True))
            for index, block in enumerate(self.lm.layers, 1):
                def trace(module, args, layer=index):
                    self.executed_layers.append(layer)
                self._handles.append(block.register_forward_pre_hook(trace))
            for layer in self.layers:
                attention = self.lm.layers[layer - 1].self_attn
                if self.collect:
                    def before(module, args, kwargs, layer=layer):
                        self._attention_pre(layer, module, args, kwargs)
                    self._handles.append(attention.register_forward_pre_hook(before, with_kwargs=True))
                    for name in ('q', 'k'):
                        def capture(module, args, output, layer=layer, name=name):
                            self._norm_capture(layer, name, output)
                        self._handles.append(getattr(attention, name + '_norm').register_forward_hook(capture))
                def gate(module, args, layer=layer):
                    return self._gate(layer, module, args)
                self._handles.append(attention.o_proj.register_forward_pre_hook(gate))
            return self
        except BaseException:
            self._cleanup()
            raise

    def _cleanup(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._pending.clear()

    def __exit__(self, exc_type, exc, tb):
        self._cleanup()
        self.diagnostics['executed_layers'] = list(self.executed_layers)
        self.diagnostics['gate_calls'] = dict(self._gate_calls)
        if exc_type is None:
            if self._calls != 1 or self.executed_layers != list(range(1, 37)):
                raise AssertionError('The complete model did not execute exactly once')
            if any(self._gate_calls[layer] != 1 for layer in self.layers):
                raise AssertionError('A target layer did not execute its real gate')
            if self.collect and set(self.statistics) != set(self.layers):
                raise AssertionError('Statistics missing for a target layer')
        return False
