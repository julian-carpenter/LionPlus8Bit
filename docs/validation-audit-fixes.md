# Corrective validation after the fresh audit

This implements the ten findings in [the fresh audit](audits/2026-11-11-fresh-audit.md), based on revision `64205431d701415de6aaa8355bcc536f81626cd5`. The original audit, raw evidence archive, and first failure results are preserved. Corrections are on branch `fix/fresh-audit-findings`.

| Finding | Correction and regression coverage |
| --- | --- |
| F01 | Per-tensor tiny flags; an auxiliary scalar no longer makes unrelated matrices permanently strict. Strict persistence remains conservative for the affected tensor and existing checkpoints. |
| F02 | The guard also detects exact nonzero inputs. FP64 norm recovery is restricted to suspect nonzero/overflowing tensors; exact zeros do not replay unrelated matrices. |
| F03 | Optimizer-owned effective override merging; saved settings override destination module/index defaults, including lazy state, while later explicit parameter overrides remain possible. |
| F04 | Restored native configuration dictionaries have independent ownership. |
| F05 | Deferred checkpoint state migrates before guards inspect it; CUDA tests compare continuation for both momentum formats and clipping modes. |
| F06 | Strict momentum scratch is chunked at 65,536 elements and normalization is independent for each 256-element block, including partial blocks. Three-step heterogeneous-range tests retain the small block and correct signs. |
| F07 | CPU preparation permits optimizer norm recovery before rejecting nonfinite reductions; true NaN/Inf inputs still fail without numerical state updates. |
| F08 | Pending module precision defaults are resolved for added parameters before allocation and saved even before their first gradient. |
| F09 | Save/load hooks run once around complete native and AdaGC state; replacement dictionaries and prepend order are tested. |
| F10 | Representability checks use the intended clipping bound rather than a reconstructed product or rounded scale. Tests cover the minimum FP32 bound, its invalid lower neighbor, warmup/bootstrap/adaptive phases, and CUDA. |

Additional changes implement the audit's practical performance and validation recommendations:

- Mixed contiguous/strided gradients retain the fused guard for eligible tensors. CPU reference and CUDA dtype/length/device-limit/uint8 parity tests cover the vector decisions.
- A bounded eight-entry LRU reuses alternating active-index sets. Parameter-sized strict momentum scratch is eliminated; clipping or flattening a strided gradient can still require a full gradient temporary.
- Live initialized dtype/device changes, cross-device inactive groups, overlapping parameters and parameter/gradient storage sharing are rejected before updates. Valid disjoint contiguous parameter views remain supported.
- Host `last_step_stats` records active/clipped/tiny/strict/recovered tensor counts. AMP overflow skips preserve diagnostics, weights, histories and counters; loading clears stale diagnostics.
- CUDA allocator-stress tests compare ordinary/native scale, no-scale, masters, strict quantization, and strided gradients against serialized execution under both clipping policies.
- Benchmarks restore exact replay inputs, reuse events, expose zero/tiny/recovery/strided/variable-active cases and nondefault streams, and include actual FP16-autocast/GradScaler training plus an unclipped native bitsandbytes control. Outputs contain raw updates, conditional spike/path distributions, timing phases, repetitions, persistent unique storage and temporary allocated peaks. A balanced campaign driver rotates arm position and pairs seeds.
- CPU CI now includes Python 3.10 with PyTorch 2.5.0 alongside current-version Python 3.11/3.12 jobs. This job is configured; hosted execution is not asserted by local validation.

## Verification results

- Full real-CUDA suite: **172 passed**, plus **15 subtests**, 16 recorded warnings.
- Installed-wheel CPU suite: **113 passed**, 59 CUDA tests deselected, plus **15 subtests**.
- Independent CUDA integration: all **six groups passed**, including separate-process controlled restart.
- Independent CUDA AMP scale/unscale oracle and all eight CPU-to-CUDA load variants passed.
- Six separate-process CPU AMP restart splits and twelve same-dtype precision restart controls remained bitwise equal.
- Wheel/sdist build, strict twine metadata checks, Ruff lint and formatting passed locally.
- Balanced benchmark matrix: **72 process runs**, **1,800 timed updates**, zero failed runs or overflow skips. Every AdaGC zero-gradient run reported zero recovered norms. Two repetitions per condition and short segments verify paths/telemetry; they do not provide a throughput confidence interval.
- Fresh independent review found no blocking defect; its CPU chunk-boundary, negative-tiny-block and overflow-state probes passed. CUDA ownership also passed allocator-stress tests.

The repeated application study completed all nine 1,100-update arms with no overflow skips. All initial/data-order hashes matched within each seed. Every validation loss exactly equaled its original counterpart. AdaGC8 versus global8 mean relative loss remains **+0.965%**, 95% paired interval **−1.272% to +3.201%**: still **inconclusive** at the declared +2% margin. This verifies continuity on that subset, not a convergence or noninferiority guarantee.

## Measurement scope

Local verification uses the same Quadro RTX 6000 SM75, Python 3.12.3, NVIDIA PyTorch 2.13 alpha, CUDA 13.3 and bitsandbytes 0.50.2/CUDA 13.2 binary as the audit. It does not establish native BF16 throughput or other backend/driver coverage.

The first corrected real AMP measurements used FP32 `Linear(1024,1024)` weights, actual uint8 momentum, FP16 autocast and GradScaler. After initialization, 12 synchronized optimizer/scaler steps per condition gave the following temporary allocated peaks:

| Auxiliary-gradient condition | Before | Corrected |
| --- | ---: | ---: |
| Ordinary | 6,656 bytes | 7,168 bytes |
| Exactly zero | 8,394,752 bytes | 7,168 bytes |
| Ordinary after one tiny value | 20,975,616 bytes | 7,168 bytes |

Only the auxiliary scalar is strict after the tiny-value trigger. In that run, corrected median step/scaler times were approximately 1.515 / 1.497 / 1.573 ms for ordinary / zero / recovered cases. Ordinary metadata overhead increased slightly; the improvements address fallback propagation and temporary peaks, rather than promising every ordinary step is faster. These are small single-run measurements, not confidence-bounded throughput estimates. Every condition's injected overflow left the complete optimizer state and parameters unchanged.

Momentum scratch is bounded by chunk size, but chunking adds launches for a large tensor that really requires strict arithmetic. Existing checkpoints with broad strict flags retain them for numerical continuity. No automatic flag reset is claimed to preserve arbitrary previous strict trajectories.

## Reproduction and evidence

The original archive remains at [audit evidence](evidence/audit-2026-11-11/README.md). Corrective raw logs and commands are recorded separately in `/abyss/home/ml/lion_8bit/audit_fixes_20261111`; the [portable corrective archive and reproduction guide](evidence/audit-fixes-2026-11-11/README.md) include the logs and complete commands. All commands run from the repository root with a hardware-compatible PyTorch and `bitsandbytes==0.50.2` installed.

```bash
python -m pytest --require-cuda -q
CUDA_VISIBLE_DEVICES='' python -m pytest -m 'not cuda' -q
LION_RESULTS=results/audit-fix-integration python tools/cuda_integration.py
python -m build
python -m twine check --strict dist/*
python -m ruff check .
python -m ruff format --check .

python tools/benchmark_clipping.py \
  --strategy adagc --layout few-large --mode amp --case zero \
  --pattern none --warmup 100 --steps 20 --repeats 3 \
  --output results/audit-amp-zero.json
```

The matched application study repeats the original three seeds and 1,100 updates for global8, AdaGC8 and AdaGC32, rotating arm order by seed. Initial weights and minibatch order hashes must match within each seed. It remains an exploratory real-audio subset with a custom loss, not production validation. Cached data/model dependencies remain outside the package, with their original hashes in the study configuration.

## Support decisions

The audit's suggested device-only scale ABI, fused preparation/update kernels, distributed logical-tensor semantics, paging, graph capture and compilation are larger backend/integration designs. This corrective pass retains the tested eager AMP contract and default-stream ownership. DDP and paging remain unverified; FSDP/ZeRO gradient shards and CUDA graphs remain unsupported. Tests and measurements must precede enabling those configurations. The memory and host improvements here do not imply that those integrations have been implemented.
