# AdaGC implementation and validation

AdaGC is opt-in; global clipping and optimized execution remain the defaults.
The implementation uses clipped historical norms, exactly 100 successful warmup steps,
positive-history bootstrap, staged history commits, packed state, native scale dispatch,
and versioned checkpoints. See the README for API and engineering extensions.

## Verified properties

- Final complete installed-wheel test suite: **79 passed**, plus 15 subtests; see `results/adagc/full-suite.log`.
- CPU-only CI collection with CUDA hidden: **50 passed, 29 deselected**, plus 15 subtests.
- Six independent CUDA integration groups passed, including bitwise-equal fresh-process
  AdaGC resumes before, at and after warmup; see `integration-final.log` and `integration-final/`.
- Numerical oracle, FP16/BF16 master weights, uint8 and FP32 momentum, AMP overflow skips,
  uneven accumulation, noncontiguous gradients, nondefault streams, tiny values, invalid
  checkpoint rejection without mutation, and dtype-preserving checkpoint restore tested.
- Ruff, wheel/sdist build, strict metadata checks, isolated wheel import and CPU usage passed.
- Independent core and measurement-tool reviews completed; identified defects were fixed.

Hardware: Quadro RTX 6000 (SM75, 24 GB), Python 3.12.3, NVIDIA PyTorch
2.13.0a0+9186a08b2c.nv26.07, CUDA runtime 13.3, bitsandbytes 0.50.2.
The bnb loader used its CUDA 13.2 binary fallback. BF16 numerical tests do not establish
native BF16 throughput on this GPU. No FP8 or multi-GPU claim is made.

## Matched application study

Exploratory SHARK log-Mel model on the preserved real MeerKAT audio subset:
256 training / 64 validation clips, custom 12-label clip-presence BCE, batch 8,
1,100 updates, seeds 17/29/43, identical initial tensors and sampled data order per seed,
FP32 parameters with FP16 autocast and GradScaler. Learning rate .0003, betas .95/.98,
weight decay .001, cosine schedule, absolute threshold 1. AdaGC uses .99/1.04/100.
The first 150 updates are excluded from throughput. Diagnostic reductions are measured
separately after synchronizing queued training work. Validation occurs after training.
All nine runs completed all updates without overflow skips or nonfinite losses/metrics.
No loss spike was detected using the preceding 1,000 observations and a 10-standard-deviation
upper threshold; only 100 observations per run were eligible, so this does not demonstrate
long-horizon stability. Raw updates include norms, scales and AdaGC threshold histories.

| Strategy / momentum bits | Validation BCE, seeds 17 / 29 / 43 | Mean examples/s | Max allocated MiB |
| --- | --- | --- | --- |
| global / 8 | 0.342537, 0.342984, 0.336741 | 389.32 | 140.19 |
| adagc / 8 | 0.344836, 0.343835, 0.343391 | 437.29 | 140.19 |
| adagc / 32 | 0.343951, 0.343999, 0.341077 | 380.83 | 140.42 |

AdaGC uint8 versus global uint8 paired relative validation BCE:
mean **+0.965%**, two-sided 95% Student-t interval
**[-1.272%, +3.201%]** (3 pairs, df=2).
Prespecified noninferiority margin: +2%. Result: **inconclusive**.
The interval does not establish noninferiority. The FP32-momentum control is reported
separately; differing training trajectories are not a matched-gradient sign oracle.
Production-task quality remains unresolved regardless of this subset result.

## Performance and memory

One million parameter elements: 4 x 262,144 or 256 x 4,096, FP32 parameters,
seed 0, 150 warmup and 100 timed updates. Identical replay distributions use no injected
spikes, localized spikes, or widespread spikes. Raw JSON includes every wall/CUDA sample,
p95, allocated/reserved peaks and profiler synchronization counts; traces are retained.
Application timings vary across seeds/runs and have no throughput confidence interval.
Timing is host-inclusive and synchronized; these short microbenchmarks have no multi-run
confidence interval. They are not production speed predictions.

| Layout / spikes | Global median ms | AdaGC median ms | Ratio | Extra peak allocated bytes |
| --- | --- | --- | --- | --- |
| few-large / none | 0.805 | 1.457 | 1.81x | 5632 |
| many-small / none | 28.645 | 35.271 | 1.23x | 30208 |
| few-large / localized | 0.807 | 1.418 | 1.76x | 5632 |
| many-small / localized | 28.088 | 34.746 | 1.24x | 30208 |
| few-large / widespread | 0.826 | 1.829 | 2.21x | 5632 |
| many-small / widespread | 32.946 | 33.894 | 1.03x | 30208 |

AdaGC is slower in this matrix. It remains opt-in; there is no speedup claim.
Profiler traces record one stream synchronization for packed readback across both tensor
counts, plus device synchronization events from the benchmark/profiler harness. There is
no per-tensor coefficient `.item()` readback. FP16-parameter/master-weight cases are also
retained in `bench-*-float16.json`. No native BF16 performance test was run on SM75.
Some earlier raw benchmark files use `first_peak_allocated`; that field covers the entire
warmup. The tool now names it `warmup_peak_allocated`.

Persistent ordinary history is FP32. **Design deviation:** small vector metadata arithmetic
uses FP64 throughout for conservative overflow/underflow handling. Stable active indices
and the packed readback buffer are reused; other vector temporaries remain allocator-managed.
This is not a fully fused implementation. FP64 exception histories and materialized strict
clipping can require an extra readback. Real memory includes metadata/scratch, allocator
rounding, native state and optional FP32 master weights; it is not simply four bytes/tensor.

## Reproducibility and raw evidence

Run from the repository root with the compatible CUDA PyTorch environment:

```bash
python -m pip install -e '.[dev]'
python -m pytest --require-cuda -q
python tools/cuda_integration.py
python examples/train_regression.py --device cuda --clipping adagc
python -m ruff check .
python -m ruff format --check .
python -m build
python -m twine check --strict dist/*
```

This machine used isolated tool dependencies in `/tmp/lionplus-adagc-dev`, bnb in
`/abyss/home/ml/lion_8bit/verification_20261111/deps`, and source under this checkout's `src`
on PYTHONPATH. The final full test run instead resolves the built wheel from
`/tmp/lionplus-adagc-wheel-final` to verify packaged behavior. CPU checks set
`CUDA_VISIBLE_DEVICES=''`. Raw logs record the actual environment and results.

Exact study/benchmark invocations and frozen configuration are in
`results/adagc/run_campaign.py`, `run_benchmarks.py`, and `study_config.json`:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# Add the preserved application_sources directory to PYTHONPATH for models.BaselineModel.
python results/adagc/run_campaign.py
python results/adagc/run_benchmarks.py
```

Use fresh output directories: tools refuse overwrite. The data cache and supporting model
sources are external to the package; paths and SHA-256 hashes are recorded in the config
and `model_source_hashes.json`. Original data preparation and split manifest remain in
`lion_plus_revised/study_release/`. The normal project dataset/loader is not reproduced here.

The raw bundle `lionplus8bit_adagc_evidence.zip` contains JSON, logs, traces, campaign scripts,
configuration, build artifacts and model/optimizer checkpoints. `SHA256SUMS` records hashes.
The first study's quality results are retained under `study-timing-v1/`; its throughput is
invalid and excluded because queued backward work was mistakenly subtracted as diagnostics.
An initially overlapping benchmark sample is retained as `overlap-discarded-*` and excluded.
The entire study was repeated after the timing fix. Original snapshots and first-run evidence
were not edited.

## Momentum sign diagnostic

An independent Python AdaGC norm-history oracle clipped 100 replayed gradients for
4,353 coordinates with 10 warmup steps. Native uint8 and FP32 momentum consumed
identical clipped gradients. Before updates, 4,546 of
435,300 sign decisions disagreed (**1.044%**).
This measures quantization behavior on one replay, not an accuracy guarantee or a defect
threshold. Raw margins/errors are retained in `sign_replay.json`; rerun with
`python results/adagc/replay_signs.py`.

## Failures and unresolved questions

- The quality noninferiority gate is inconclusive; do not claim a quality improvement.
- Optimizer-only latency increases in the tested matrix; default global clipping is faster.
- Fully device-side scales/fused updates, distributed logical-tensor reductions, paging,
  graph capture, compilation and FP8 remain outside the verified support envelope.
- Original globally clipped Lion+ theoretical guarantees do not automatically transfer.
- Longer training, other models/tasks, changing batch sizes, threshold bootstrap ablations,
  and threshold/bootstrap sensitivity remain research work.
- Existing CUDA tests cover numerical BF16 behavior, not native BF16 hardware performance.
- A backend failure partway through updates cannot roll back earlier parameters; restart
  from a known checkpoint. Failed finite-input validation is checked before updates.

Local CPU/CUDA CI-equivalent commands were executed. Workflow definitions include the new
tests and AdaGC examples; a fresh hosted Actions result is not claimed here.

An initial final-integration invocation used an unsupported `--require-cuda` CLI option.
Its rejection log is retained; the corrected unittest runner invocation passed all six gates.
The pytest suite separately used `--require-cuda`.
