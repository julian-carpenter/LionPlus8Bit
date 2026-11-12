# Training integration validation

This report separates verified behavior from observed failures and unanswered questions. The implementation baseline is `0fb6a5742c6b27e4445e05b401410d6083c45190`. The [approved design](superpowers/specs/2026-11-12-training-integrations-design.md) and [plan](superpowers/plans/2026-11-12-training-integrations.md) define the acceptance criteria. [Portable raw evidence, source snapshots and checksums](evidence/integrations-2026-11-12/README.md) accompany this report. Earlier evidence archives remain unchanged.

## Results at a glance

| Area | Verdict | Practical boundary |
| --- | --- | --- |
| Standard DDP + FP16 AMP + AdaGC | Passed on two/four GPUs | One process and optimizer per GPU; replicated, synchronized gradients and matching update boundaries |
| Channels-last/3d and dense permutations | Passed numerical, storage and stream tests | Temporary staging; no overlapping or stepped views |
| Native managed paging | Unsupported | Eligible requests fail before updates; pinned allocator ownership/lifetime is insufficient |
| Native Inductor FP32 model | Passed focused numerical fixtures | Short two/four-rank DDP smoke and one-GPU cache-recreation recovery; no speed claim |
| Native Inductor FP16 model | Failed fixed numerical gates | Eager optimizer replay of the actual compiled gradients still matches |
| Optimizer compilation | Partial Dynamo capture passed | Counted GraphModule backend; fullgraph and CUDA graphs unsupported |
| Trainer/DataLoader recovery | Passed controlled rollback fixtures | Fixed world size, committed checkpoints, replayable data; no live worker hot replacement |
| Native FP8 | Hardware blocked | SM75 cannot run the positive qualification; backend availability alone is insufficient |
| Paired contiguous performance | Five bounds passed; one inconclusive | Three repetitions, one-sided upper 95% slowdown bound of +10% |

## Qualified environment

Four Quadro RTX 6000 GPUs, 24 GiB each, compute capability 7.5, driver 580.178.04. Python 3.12.3; NVIDIA PyTorch `2.13.0a0+9186a08b2c.nv26.07`, CUDA runtime 13.3, NCCL 2.30.7; bitsandbytes 0.50.2 selected its CUDA 13.2 binary. The optional environment contained torchdata 0.11.0, Triton 3.7.1, and development builds of Transformer Engine 2.17 and torchao 0.18. Those optional installations do not qualify their FP8 backends.

The passing distributed runs explicitly set `NCCL_P2P_DISABLE=1` and used SHM transport. GPUs have peer access but no NVLink. An initial direct-transport run was interrupted without a verdict; this does not diagnose a transport defect or establish direct-P2P qualification. Record and validate transport settings for a different node.

Deterministic recovery fixtures disable TF32 and cuDNN benchmarking, enable deterministic algorithms and set `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Rank/device mappings, GPU UUIDs, source hashes and selected environment settings are recorded in manifests. GPU experiments ran sequentially under one coordinator.

## Final regression and packaging checks

| Check | Recorded result |
| --- | --- |
| Complete suite, four GPUs/CUDA required | 401 passed, 15 subtests passed |
| Installed-wheel CPU suite from `/tmp`, CUDA hidden | 334 passed, 67 deselected, 15 subtests passed |
| Original independent CUDA runner | All six groups passed, including fresh-process resume |
| Dedicated layout CUDA matrix | 130 passed |
| Focused final recovery intersections | Seven CPU tests passed; separate CUDA intersections passed |
| Wheel and sdist | Built successfully; strict Twine metadata check passed |
| Ruff and patch whitespace | Passed |

The frozen baseline passed 172 tests/15 subtests with CUDA and 113 tests/15 subtests on CPU (59 CUDA tests deselected). Its first CPU run had a missing distribution-metadata import path; a separately retained rerun corrects the environment without editing baseline source. The installed-wheel import is checked outside the checkout and every package source hash matches the tested implementation. Nonfatal dependency/fork deprecation warnings are preserved in raw logs.

## Distributed numerical evidence

Two- and four-rank contiguous models and a four-rank channels-last Conv2d model each completed four cases with 204 attempts per case. FP16 AMP cases committed 201 updates and collectively skipped three overflow attempts; FP32 cases committed 204. Cases cover AdaGC warmup 0/3/100, accumulation 1/3, bucket views on/off, both zero-grad modes, auxiliary zero/tiny gradients and original gradient preservation. Optimizer state, native integer momentum codes, history, model weights and complete public scaler state matched across ranks and against reduced-gradient replay bitwise.

Pre-reduction Inf injected on one rank causes coordinated skips before initialization and around warmup boundaries. Parameters, momentum, history, counters and scheduler remain unchanged on skips. Mismatched initial scaler state and rank-local nonfinite gradients introduced *after* reduction are negative cases: the checked driver refuses all updates. The optimizer itself has no process group and cannot repair a rank-local training-loop error. Validation agreement scans/collectives are outside the timed benchmark.

Independent full-global-batch gradient comparisons use fixed tolerances: FP32 `rtol=2e-5, atol=1e-7`; FP16 backward `rtol=2e-3, atol=2e-5`. FP32 maximum errors were 2.33e-10 to 3.98e-10, with no sign differences. FP16 maximum errors were 7.75e-7, 4.92e-7 and 2.50e-6 for two-rank linear, four-rank linear and four-rank convolution fixtures; sign differences were 2/4/70, with zero minimum reference margins. These are reduction/backward differences, not full-batch bitwise equivalence. Actual reduced-gradient replay remains exact. Independent norm/momentum equations rule out multiplying clipping norms by an erroneous square root of world size. Uneven per-rank sample counts use global-mean loss weighting.

Replay preserves gradient strides and alignment: cloning a bucket view to offset zero changed an FP32 foreach norm by one ULP (1.49e-8 in history). Preserving alignment resolved the oracle discrepancy without changing optimizer arithmetic or relaxing assertions. Intermediate failed replay logs are retained.

Additional two-/four-rank tests use `find_unused_parameters=True`, FP16 AMP, AdaGC warmup three, bucket views and accumulation one/three. Each case completes 12 updates. A branch used only by rank zero receives the synchronized gradient on every replica. A shared branch first activates at step five, then alternates activity; absent gradients freeze its history and the first positive gradient bootstraps it. A permanently dormant 4,097-element parameter retains no momentum and unseeded history. A separate actual iterator-exhaustion probe stops every rank before another forward/update and verifies unchanged state rather than continuing with `join()` and stale optimizer histories.

No FSDP/ZeRO shards, changed-world-size continuation or one optimizer spanning devices is qualified. Parameters excluded from DDP at construction require coordinated reconstruction before training them.

## Layout and checkpoint contracts

Supported weights are dense, nonoverlapping contiguous permutations: ordinary contiguous, channels-last Conv2d, channels-last-3d Conv3d, transposes and dimension permutations. Expanded/internal-overlap views, distinct overlapping parameter storage and stepped slices with holes fail before updates. Disjoint dense views remain allowed.

Native state, quantization blocks and master weights use canonical logical row-major order. A noncontiguous weight stages one tensor at a time; copy-back preserves the original data pointer, shape, strides, storage offset and gradient object/values. FP32 weights have no persistent FP32 shadow. FP16/BF16 primary weights retain the documented optional FP32 master. An initialized layout cannot change in place; use a load boundary or fresh optimizer.

The dedicated CUDA layout matrix passed 130 tests; CPU layout controls passed 128 with three CUDA skips. Cases include uint8/FP32 momentum, both clipping policies, ordinary/strict paths, partial blocks, low-precision masters, versioned checkpoint continuation, native overrides/hooks, and nondefault-stream allocator churn. Four-rank channels-last DDP, four-rank channels-last AMP checkpoint rollback and two-rank transposed AMP restart provide end-to-end intersections. Other layout/backend combinations need their own checks.

Checkpoint schema v3 records `state_order="logical-row-major-v1"` and parameter shapes, supporting continuation between accepted physical layouts. Schemas v1/v2 still load. Group/order preservation remains necessary: swapped unnamed same-shaped parameters cannot be detected automatically.

Three rotated measurement repetitions used 100 warmup and 300 timed updates on two 524,288-element weights. Median contiguous/staged times were 0.732/0.811 ms with uint8 momentum and 0.720/0.811 ms with FP32 momentum. Individual paired staging overheads ranged from 2.1% to 14.2%. Persistent allocation matched; transient peak was 2,099,200 bytes staged versus 3,072 bytes contiguous. This is a workload-specific copy cost, not a universal performance prediction. The PyTorch control has different temporary-memory behavior and is not an equivalent optimizer speed comparison. These results do not justify a new stride-aware kernel in this change.

## Managed paging investigation

The pinned bitsandbytes `get_paged` path wraps a raw `cudaMallocManaged` pointer using NumPy/ctypes and `torch.frombuffer`. It provides no storage deleter/release contract and the process-wide page manager retains strong references. Its tensors are CPU-labelled, which also prevents treating them as ordinary CUDA tensors in PyTorch guard/strict arithmetic. Inspection of exported symbols in all seven supplied CUDA binaries found no supported release interface.

Consequently `is_paged=True` is refused for CUDA parameters with at least 100,000 elements, before any update or eligible state allocation. CPU and smaller allocations use ordinary storage and are not called managed paging. Immediate load, deferred first-step transfer and forged/legacy paging metadata have focused negative tests. Saving copies legacy attributed tensors to plain values; restoring a Python attribute does not certify managed allocation. No unsafe guessed `cudaFree`, fake paged restore or allocator rewrite is introduced.

GPU migration pressure, residency, repeated managed-allocation lifecycle and DDP managed-state recovery remain unqualified because the storage ownership gate fails first. Investigating an owned backend allocator is necessary before those experiments. Eight-bit momentum compression works independently with default `is_paged=False`.

## Compilation verdicts

Actual Inductor FP32 model fixtures passed with global/AdaGC, native uint8 state, changing LR, checkpoint loads and sign/state comparisons. Short two-/four-rank compiled-global FP32 DDP smoke runs passed four updates each. A separate one-GPU FP32 trainer restart deletes/recreates compiler caches in each launch, captures a real graph and reproduces restored state/RNG and continuation bitwise. These are focused compatibility results, not throughput or long-run stability qualification.

Actual Inductor FP16 model fixtures **failed** the fixed gradient/sign gates: maximum gradient errors ranged from about 6.1e-5 to 2.5e-4, with one/four sign differences at steps two/three. The raw failure is preserved in `runtime/cuda-compile-final`. Eager optimizer replay of the actual compiled gradients still matched bitwise. No tolerances were relaxed and no eager fallback was substituted for this result. FP16 model-fusion drift needs a separate investigation before promoting that combination.

Dynamo optimizer probes captured 25/34/35 graphs for global/AdaGC/channels-last cases with exact replay, changing LR/active sets, checkpoints and strict paths. They use a counted GraphModule interpreter backend, **not Inductor code generation**, and do not establish a speedup. Host/storage introspection and scalar readbacks break fullgraph capture. CUDA graphs and a fully device-only optimizer are unsupported; removing safeguards to obtain capture is not an acceptable fix.

## Recovery and checkpoint review

CPU fixtures exercise six trainer fault points: before forward, inside accumulation, after backward, after a step before commit, during checkpoint writing and after manifest publication. Real loader-child SIGKILL cases cover two/four workers, prefetch one/two and persistent/nonpersistent workers. Two-rank Gloo rollback also passes. A separate torchdata 0.11.0 StatefulDataLoader iterable fixture replays with zero/two/four workers; it does not qualify arbitrary external streams or the full cross-product of failures and datasets.

CUDA fixtures pass four-rank trainer failure after a step with two loader workers, two-rank loader failure with four workers/prefetch two, two-rank AMP recovery with transposed weights, and four-rank AMP recovery with channels-last convolution weights. Exact comparisons include consumed sample IDs/augmentation, native optimizer state, scaler/scheduler, distinct logical-rank buffers and next Python/NumPy/CPU/CUDA random draws. The AMP fixture consumes 12 batches, completes five successful updates and skips one overflow per rank. Prefetched data does not advance the consumed cursor; an AMP skip does.

Physical GPU remapping from visible devices 0/1 to 2/3 passes UUID-change checks and exact restored checkpoint/data/RNG comparisons. Continuation is compared with an independent control on the *new placement*, not with an unconditional old-topology bitwise trajectory. World size remains fixed. Native compiled-cache recreation passes separately in FP32.

Reference checkpoint generations contain per-rank model and RNG/data state, optimizer/scaler/scheduler, checksums and a final atomic manifest. Generation directory entries are fsynced before manifest publication. Model unwrapping recognizes actual framework wrappers rather than arbitrary children named `module`; a red-to-green regression preserves other parameters/buffers. Component validation/load failures cause collective rollback of training components using CPU snapshots. The independent review and resolution notes are included in the archive.

These are process-failure checkpoint/replay guarantees at quiescent boundaries. Mid-accumulation failure rolls back and replays; loading an arbitrary unfinished accumulation, live worker replacement, unreplayable streams, changed worker-count stateful datasets, changed world size and recovery of an unfinished CUDA kernel remain outside the contract. Native paging recovery is blocked by allocator ownership.

## Performance and quality

The final balanced campaign completed **54/54 fresh processes**, three repetitions, 100 warmup plus 300 timed updates each: 16,200 timed updates plus 5,400 warmup updates. It covers strong/weak scaling at world sizes one/two/four, paired baseline/current contiguous arms, and descriptive current channels-last arms. Ordinary, zero, tiny auxiliary and recovery segments are timed; replay hashes and correctness collectives are excluded. Reports contain slowest-rank latency, aggregate samples/second, per-phase successful-update counts and allocated/reserved memory. No AMP updates were skipped in timed arms. The initial campaign's incompatible baseline channels-last attempt is preserved separately; the corrected campaign pairs only layouts supported by the baseline.

The gate is a one-sided upper 95% paired log-ratio t bound against +10% slowdown, using three repetitions and explicit distributional assumptions:

| World size | Scaling | Estimate | Upper 95% slowdown | Verdict |
| ---: | --- | ---: | ---: | --- |
| 1 | Strong | -11.25% | +18.20% | Inconclusive |
| 1 | Weak | +0.89% | +3.66% | Passed |
| 2 | Strong | +1.07% | +4.80% | Passed |
| 2 | Weak | -0.06% | +6.30% | Passed |
| 4 | Strong | -0.07% | +0.83% | Passed |
| 4 | Weak | -0.14% | +0.73% | Passed |

The first row is noisy and cannot establish the declared bound. More independent repetitions are needed; a favorable point estimate is insufficient. Channels-last costs are descriptive because the old implementation rejects that layout. Native managed bytes are zero because paging was not requested. Migration costs and compiler/FP8 speedups are not measured by this campaign.

A three-seed, 200-update AdaGC contiguous CPU replay matches baseline bitwise, including strict arithmetic and AdaGC warmup. This change does not introduce a new contiguous numerical algorithm or a new application-quality claim. The prior three-seed AdaGC/global application study remains **inconclusive** at its declared +2% quality margin (mean +0.965%, 95% interval -1.272% to +3.201%); see [its report](validation-audit-fixes.md). The synthetic scaling campaign does not establish application convergence.

## FP8 boundary and remaining work

SM75 native-FP8 probes reject before FP8 model/optimizer allocation. `--require-fp8` exits nonzero; dependency presence, emulation and float8 casts cannot pass. A mock positive-capability test checks the public Transformer Engine availability API but is not hardware evidence. Even on a capable machine, the probe reports `available-unqualified` until the full numerical/cache/checkpoint/DDP suite exists; the required flag is deliberately not a hardware-only success switch.

The optional Transformer Engine example illustrates FP8 compute with FP32 primary parameters, ordinary dequantized/unscaled FP32 gradients, fused `main_grad` disabled and backend extra state in model checkpoints. It is not a qualified adapter. FP8 cached-weight refresh, scale/amax evolution across AMP skips, matched-gradient replay, two/four-rank scaling, resume and paired quality/memory/throughput need compatible hardware. torchao tensor subclasses/recipes need their own adapter and tests; no torchao support is claimed. FP8 primary parameters remain rejected.

## Reproduce

Install CUDA PyTorch for the machine, then `python -m pip install '.[dev]'`. Optional torchdata/Transformer Engine versions require separate compatible installations. Run from the checkout with fresh result directories. The portable archive includes exact historical commands, including local dependency paths; the following commands use an ordinary installed package.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
python tools/training_integration.py --suite capabilities --output results/capabilities
python -m pytest --require-gpus 4 --require-cuda -q
CUDA_VISIBLE_DEVICES='' python -m pytest -m 'not cuda' -q
LIONPLUS_LAYOUT_DEVICE=cuda python -m pytest tests/test_weight_layouts.py --require-cuda -q
LION_RESULTS=results/independent-cuda python tools/cuda_integration.py

# Explicit transport setting used on this node; validate on other topologies.
export NCCL_P2P_DISABLE=1
python -m torch.distributed.run --standalone --nproc-per-node=2 tools/ddp_integration.py --output results/ddp2
python -m torch.distributed.run --standalone --nproc-per-node=4 tools/ddp_integration.py --output results/ddp4
python -m torch.distributed.run --standalone --nproc-per-node=4 tools/ddp_integration.py --channels-last --output results/layout4
python -m torch.distributed.run --standalone --nproc-per-node=4 tools/ddp_unused_integration.py --output results/unused4
python -m torch.distributed.run --standalone --nproc-per-node=4 tools/ddp_integration.py --suite participation --output results/exhausted4

python tools/training_integration.py --suite paging --device 0 --output results/paging
python tools/training_integration.py --suite compile --compile-precision fp32 --output results/compile-fp32
# Expected failed numerical gates on the recorded runtime:
python tools/training_integration.py --suite compile --compile-precision fp16 --output results/compile-fp16
# Expected nonzero unsupported/unqualified native FP8:
python tools/training_integration.py --suite fp8 --require-fp8 --output results/fp8-required

python tools/restart_integration.py --device cuda --world-size 4 --loader-workers 2 --fault-point after-step --output results/restart4
python tools/restart_integration.py --device cuda --world-size 2 --loader-workers 4 --case loader --prefetch-factor 2 --output results/loader2
python tools/restart_integration.py --device cuda --world-size 4 --layout channels-last --loader-workers 2 --fault-point after-step --output results/channels-last-restart
python tools/restart_integration.py --device cuda --world-size 2 --layout transposed --fault-point after-step --output results/transposed-restart
CUDA_VISIBLE_DEVICES=0,1 python tools/restart_integration.py --device cuda --world-size 2 --resume-visible-devices 2,3 --fault-point after-step --output results/remapped-restart
python tools/restart_integration.py --device cuda --world-size 1 --compiled-backend inductor --precision fp32 --fault-point after-step --output results/compiled-restart

# Unpack baseline source into a fresh directory before the matched campaign.
mkdir /tmp/lionplus-baseline
git archive 0fb6a57 | tar -x -C /tmp/lionplus-baseline
python tools/benchmark_integrations.py --baseline-root /tmp/lionplus-baseline --scaling strong weak --repetitions 3 --warmup 100 --steps 300 --output results/scaling
python -m build
python -m twine check --strict dist/*
```

The manual [integration workflow](../.github/workflows/integrations.yml) requires a trusted four-GPU self-hosted runner, checks the device count and uploads per-rank evidence on failure. It covers regression/layout/DDP/unused-branch/restart cases. It is configured rather than evidence that a self-hosted runner has been registered or that its dispatch has run. Hosted CPU build/test CI is separate.
