# HumanRef: real Uni proposals -> Ref -> error attribution

Run from the repository root, in the existing `wedetect_ref` environment.
This is a new evaluation protocol alongside the official dataset-proposal protocol.
It supports HumanRef only. It does not train any model or insert GT proposals.

## Smoke test and complete run

```bash
cd /media/data6/chengz/WeDetect
conda activate wedetect_ref
# First 16 expressions: check loading, checkpoint conversion, saved scores and metrics.
CUDA_VISIBLE_DEVICES=0 GPUS=1 LIMIT=16 OUT=results/humanref_real_uni_smoke \
  bash tools/run_humanref_pipeline.sh
# All expressions, images deduplicated for Uni inference:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 GPUS=8 \
  OUT=results/humanref_real_uni_k100 bash tools/run_humanref_pipeline.sh
```

Override `ANN`, `IMAGES`, `UNI`, `REF`, `K`, `THRESHOLD` if paths/settings differ.
Defaults: Base-Uni, Ref-4B, 100 proposals, Ref score > 0.35, no additional Ref NMS.
0.35 is an exploratory setting from previous tests, not a verified official threshold.
Uni uses the repository's FP32 inference, built-in score > 0, NMS IoU 0.7,
and top K. Ref uses BF16 and the same prompt/processor as `eval.py`.
No extra Uni score threshold is applied. Ref geometry inputs are BF16; saved
evaluation boxes retain original FP32 coordinates (the old eval.py rounds its
output boxes through BF16, so very small metric differences are possible).
The image path resolver supports Unicode normalization as needed by HumanRef.

## Outputs

- `uni/uni.rankNNN.json`: image names, original-coordinate xyxy boxes, objectness
  scores, timings and provenance. One complete shard per process.
- `ref/ref.rankNNN.json`: every expression ID, actual input candidates and their
  raw sigmoid Ref scores, including below-threshold candidates.
- `analysis/predictions.jsonl`: official metric prediction format.
- `analysis/official/comparison.md` and `comparison.json`: existing HumanRef
  P/R/DensityF1/rejection tables, including IoU 0.5:0.95.
- `analysis/diagnostics.json`: overall and per-domain error counts and rates.
- `analysis/samples.json`: per-expression attribution and unmatched prediction count.
- `analysis/provenance.json`: Ref metadata and input shard hashes.

Original annotation candidate_boxes remain untouched for metric calculation:
the official DensityF1 penalty uses their count as the number of persons.
They are NOT sent to Ref. Ground truth is only read for offline analysis.

Different expressions for the same image share Uni inference but receive separate
Ref scores. No process-group communication is needed: torchrun assigns ranks,
each process saves its shard, and the next command starts only after all exit.
Analysis checks full rank coverage, identical metadata, unique IDs and exact
annotation/prediction coverage. Failed workers must not be treated as empty predictions.
Outputs cannot overwrite existing files. For retry, choose a new directory or
rerun individual failed ranks with exactly the same settings; no automatic resume.

## Offline threshold analysis (no GPU)

```bash
python tools/humanref_pipeline.py analyze \
  --annotations data/HumanRef/annotations.jsonl \
  --ref-dir results/humanref_real_uni_k100/ref \
  --output results/humanref_real_uni_k100/analysis_t050 \
  --score-threshold 0.5
```

Use the same `--limit` as the producing run for smoke-test analysis.
Use plain python for analysis, not torchrun. `--skip-official-metrics` computes
only dependency-free diagnostics; it does not report official benchmark metrics.

## Diagnostic definitions

At IoU >= `--iou` (default 0.5), a target is covered if at least one candidate
overlaps it, and recovered if at least one *selected* candidate overlaps it.
This is geometric coverage, not one-to-one matching. The official metric still
uses its own matching implementation; do not substitute diagnostic recall for it.

For positive targets the decomposition is exact:

`missed targets = proposal_missed_targets + ref_missed_covered_targets`.

`ref_missed_covered_targets` includes threshold/calibration/selection effects;
it does not prove a semantic reasoning failure. Adding boxes can add false positives,
so also inspect official Precision and `unmatched_selected_boxes`.

Mutually exclusive sample groups:

1. `no_target_covered`: no GT target has any usable candidate.
2. `partial_target_coverage`: only some GT targets have usable candidates.
3. `covered_but_ref_missed`: all targets covered, but selection misses at least one.
4. `all_targets_recovered`: all recovered, possibly with extra false positives.
5. `correct_rejection`: negative expression, no selected boxes.
6. `false_positive_rejection`: negative expression, at least one selected box.

Groups 1/2 describe coverage first; group 2 may also contain Ref selection errors.
Use target-level counts to expose both sources. Rates with no denominator are null.
The main decision statistics are `target_coverage`, `fully_covered_sample_rate`,
`recovery_given_covered_target`, `all_targets_recovered_given_full_coverage`,
and `rejection_accuracy`. Multi-target full recovery is not exact-set accuracy.

## Scope and timing

This is cached end-to-end *accuracy* evaluation, not online end-to-end latency.
Uni times include image loading/preprocessing/forward/postprocessing. Ref times
include prompt preprocessing and forward, but exclude image file loading. Both
include cold calls, exclude weight loading, and synchronize CUDA. Do not compare
them directly to paper FPS. Preserve raw timings for later dedicated benchmarking.

No GPU inference was validated on the development Windows machine. Run the smoke
test on the server before the full benchmark. CPU tests:

```bash
python tools/test_humanref_pipeline.py
```
