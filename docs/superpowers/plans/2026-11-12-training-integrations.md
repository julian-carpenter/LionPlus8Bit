# Training Integrations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Qualify DDP with AMP/AdaGC on four GPUs, then support useful weight layouts and establish measured paging, compilation, restart and FP8 integration contracts.

**Architecture:** Preserve the single-device eager optimizer as the reference. Add focused layout/paging helpers only behind tested paths and independent integration runners; keep training recovery separate from optimizer state. Promote support statements only after their corresponding gates pass.

**Tech Stack:** PyTorch/NCCL, bitsandbytes 0.50.2, optional torchdata/Transformer Engine/torchao, pytest and manual GPU CI.

**Spec:** [Training integration design](../specs/2026-11-12-training-integrations-design.md). Read it before executing; it defines invariants and the proposed scope decisions.

Status: executed against baseline `0fb6a57`; see [results and limitations](../../validation-integrations.md) and [execution ledger](../../integration-progress.md). Four 24-GiB SM75 GPUs were available. Native FP8/BF16 throughput is hardware-blocked, managed paging fails its ownership prerequisite, and FP16 Inductor did not pass its numerical gate. Task completion means a recorded verdict within those boundaries, not universal positive qualification.

## Global Constraints

- Core dependency bounds remain Python >=3.10, torch >=2.5,<3 and bitsandbytes==0.50.2. Qualify optional feature versions separately.
- One optimizer per device remains the public boundary. DDP uses one process per GPU with full replicated parameters; no FSDP/ZeRO shard support is implied.
- Preserve clipping equations, successful-step counters, original gradients, strict safeguards and independent checkpoint policy ownership.
- Preserve prior evidence. New runners refuse overwrites and record failures/timeouts/unsupported cases explicitly.
- GPU experiments run sequentially under one coordinator; CPU review can run in parallel. Set local CUDA device before NCCL initialization and optimizer allocation.
- Initial recovery requires fixed world size and checkpointable/replayable data. FP8 positive qualification requires compatible hardware.
- Contiguous eager defaults stay available. Do not enable a new default solely because a smoke test passes.

## Review Focus

1. Rank-local overflow or mismatched loss scales must not silently produce divergent optimizer histories: Task 1 tests standard synchronized skips and unsafe post-reduction failures.
2. Layout staging must write the original weight and preserve aliases/strides: Task 2 checks data pointers, gradients and canonical quantization blocks.
3. A saved paging flag must restore real managed allocation and the correct device: Task 3 checks storage, ownership and continuation on each GPU.
4. Prefetched data is not necessarily consumed data; an AMP skip still consumes a batch: Task 5 checks exact sample/RNG replay and separate counters.
5. Compiler fallback or FP8 emulation must not masquerade as positive support: Tasks 4/6 record captured graphs/capability reasons and require real-kernel evidence.

## File map

Existing core files to modify only as needed: `src/lionplus8bit/_optimizer.py`, `_clipping.py`, `_numerical_guard.py`, `_strict.py`; tests and README remain consistent with qualified behavior.

New internal modules: `_layout.py` for canonical contiguous work and overlap checks; `_paging.py` for native managed allocation/restore/lifecycle. Keep their responsibilities separate.

Integration infrastructure: `tools/_integration_common.py`, `tools/training_integration.py` (single-rank suites), `tools/ddp_integration.py` (torchrun worker), `tools/restart_integration.py` (failure controller), `tools/benchmark_integrations.py` (balanced measured runs). Existing `tools/cuda_integration.py` and benchmark tools remain regression controls.

Reference trainer: `examples/train_ddp.py` and `examples/_training_state.py`. Dataset/RNG/DataLoader snapshots belong to this layer, not to `LionPlus8bit.state_dict()`.

New focused tests: `test_ddp_integration.py`, `test_weight_layouts.py`, `test_paging.py`, `test_compile_integration.py`, `test_training_restart.py`, `test_fp8_integration.py`. Reuse current regression helpers without sharing the numerical implementation with every oracle.

## Task 0 — Freeze baseline and build the capability/launch harness

**Files:** create `tools/_integration_common.py`, `tools/training_integration.py`; modify `tests/conftest.py`, `pyproject.toml`; create launcher tests.

**Interfaces:** `tools/training_integration.py --suite capabilities|layouts|paging|compile|fp8 --output DIR [--require-fp8]`; `tests/conftest.py` adds `--require-gpus N`, and `multigpu`, `paging`, `compile`, `fp8`, `slow` markers. These are planned interfaces, not current commands.

- [ ] Add launcher tests asserting insufficient GPUs/unsupported required FP8 exits nonzero, subprocess errors propagate and existing output directories are refused. Verify they fail before adding the new harness.
- [ ] Implement capability JSON: source/version hashes, GPU UUID/capability/memory/topology, actual backend availability, NCCL/version/transport when launched, optional imports and explicit unsupported reasons. Do not import optional GPU backends unconditionally.
- [ ] Run the current complete suite, installed-wheel CPU check and six CUDA integration groups in fresh directories. Expect at least the existing 172 tests and 15 subtests to pass; record actual updated counts rather than freezing collection to 172.
- [ ] Add a minimal NCCL smoke to Task 1's worker; collect 2-/4-rank device mapping and collective results before optimizer cases. Use process-group timeout 180 seconds and controller case timeout 300 seconds; kill only child groups on timeout.
- [ ] Commit the harness/baseline record. No README support promotion yet.

## Task 1 — Qualify standard DDP with FP16 AMP and AdaGC first

**Files:** create `tools/ddp_integration.py`, `tests/test_ddp_integration.py`, `examples/train_ddp.py`; modify core only for reproduced defects.

**Interfaces:** `torchrun --standalone --nnodes=1 --nproc-per-node=N tools/ddp_integration.py --suite smoke|core|amp|buckets --output DIR`, for N=1/2/4. Shared harness creates per-rank files and rank-0 summary; only that summary may mark the distributed case passed.

- [ ] Write red/coverage-gap tests for agreed device mapping, AdaGC equality on all ranks and an independently computed global norm that must not gain a sqrt(world-size) factor. Include real uint8 state (4097 elements) and small FP32 state.
- [ ] Implement the runner with `torch.cuda.set_device(LOCAL_RANK)`, NCCL, identical policy/scaler initialization and full replicated parameters. No distributed collectives inside the core optimizer by default.
- [ ] Run global/AdaGC controls at world sizes 1/2/4, FP32 and FP16 autocast, warmup=0/3 and crossing step 100/101, `gradient_as_bucket_view` on/off, `zero_grad(set_to_none)` true/false and accumulation 1/3 with `no_sync()` covering forward/backward. Use targeted cases/pairwise coverage rather than every combination.
- [ ] Test a zero auxiliary tensor, a tiny coordinate, mixed active/None patterns, identical branch-activation schedules, and one rank's unused branch with `find_unused_parameters=True`. Check that DDP's synchronized gradient availability, not local forward usage, determines optimizer history. Do not add parameters after constructing DDP or claim true unfreezing for parameters excluded from its initial trainable set; coordinated DDP rebuild is a separate feature.
- [ ] Inject NaN/Inf before reduction on one rank: all ranks must skip with identical complete public GradScaler state, including its growth tracker, and unchanged parameters/momentum/history/counters/scheduler. Test skips before initialization and on both sides of warmup. Test inconsistent scaler startup and post-reduction one-rank Inf as negative cases: the checked reference driver verifies scaler agreement before backward, checks gradient finiteness collectively after synchronized backward/unscale, and stops every rank before an update. No private AMP fields. Keep these extra diagnostic scans/collectives explicit in timings; they are not an automatic cost or guarantee of the core optimizer.
- [ ] Check uneven per-rank sample counts with correct global-mean weighting; separately make a rank exhaust its input and require bounded fail-fast termination. Do not enable `join()` continuation with stale optimizer state.
- [ ] Compare identical-gradient replay bitwise across ranks and a single-rank replica; compare real DDP gradients to a full-global-batch reference using FP32 reference tolerance rtol=2e-5/atol=1e-7 and FP16-backward tolerance rtol=2e-3/atol=2e-5. Fix seeds/input scales before outcomes. For sign-sensitive discrepancies, report margins and replay actual reduced gradients; do not relax tolerances to conceal wrong clipping. Hash integer momentum and metadata, and check decoded momentum against an independent recurrence where appropriate.
- [ ] Run 200 successful updates including warmup on 2/4 GPUs and three matched application seeds if numerical paths change. Deliver DDP example, a first support statement restricted to tested standard training, and commit.

## Task 2 — Support channels-last and safe noncontiguous weights

**Files:** create `src/lionplus8bit/_layout.py`, `tests/test_weight_layouts.py`; modify `_optimizer.py`, `_strict.py`, allocation/checkpoint methods and the single-rank runner.

**Interfaces:** `_layout.py` exposes internal `validate_parameter_layouts(params)`, `canonical_work(tensor)` and copy-back helpers chosen during implementation; all native work/state uses logical row-major order. New checkpoint v3 records `state_order='logical-row-major-v1'` and shape mapping; continue reading v1/v2.

- [ ] Add tests that fail under the current contiguity rejection: Conv2d channels-last, Conv3d channels-last-3d, dense transpose/permutation, size-one dimensions and partial quantization blocks. Test real 8-bit state, FP32 state, FP32 weights and FP16/BF16 masters.
- [ ] Add overlap tests before loosening validation: expanded/overlapping views and distinct overlapping parameters fail before updates; disjoint contiguous views still work. Stepped-slice support is a separate investigated case, not an accidental consequence of removing the check.
- [ ] Implement canonical state/master allocation and per-tensor work staging/copy-back. Never call the backend branch that replaces original noncontiguous `p.data`. Preserve original data pointer, shape, strides, storage offset, tied-object identity and `.grad` identity/values.
- [ ] Compare at least three updates with contiguous logical copies under both clipping policies, ordinary/clipped/strict paths and FP32-control/uint8 momentum. Assert no persistent FP32 shadow for FP32 weights and no scratch retained for the sum of all weights.
- [ ] Check v1/v2 resume and v3 cross-layout continuation in both directions; check native overrides/hooks, deferred load and layout-change rejection after initialization. Keep master/state ordering canonical on load.
- [ ] Run default/nondefault stream allocator churn and 2-/4-rank channels-last DDP. Compare full model forward/backward output quality, not only manually assigned gradients. Record staging/copy-back peak and timing versus contiguous eager and a PyTorch control; use the result to decide whether stride-aware kernels are justified.
- [ ] Commit supported layouts and document rejected overlap/slice boundaries.

## Task 3 — Investigate, repair and qualify native paging

**Files:** create `src/lionplus8bit/_paging.py`, `tests/test_paging.py`; modify allocation/load/guard/strict paths only where needed; extend the single-rank and DDP runners.

**Interfaces:** `_paging.py` owns managed state allocation and reconstruction, exposing internal allocation/restore/release operations. Do not serialize raw pointers or treat a restored `.is_paged` attribute as proof of valid allocation.

- [ ] Write storage assertions and continuation tests at 99,999/100,000/100,001 elements plus a block tail. Demonstrate actual managed allocations through native metadata/pointer attributes, and no such allocation for ineligible small state. Test uint8 and FP32 momentum.
- [ ] Probe CPU-labelled managed tensors through fused/eager guards and strict arithmetic. Prefetch before first device access where required; ensure guards do not inadvertently pull FP32 state to CPU or create full-sized shadows. Preserve default/caller stream lifetimes and native per-device targeting.
- [ ] Test every local device ordinal, then two/four rank processes with `is_paged=True`. Check page_deviceid and shared-manager ownership; native allocations on GPU0 must not capture another rank's state.
- [ ] Implement checkpoint value snapshots and real paged reconstruction, including v2 migration and default/deferred loading onto different physical devices. Strip stale paging attributes; verify resumed allocation and numerical continuation, not only saved policy equality.
- [ ] Test repeated create/use/delete and failure cleanup with native ownership accounting, NVML/process RSS and GPU work completion. Investigate native allocation/free guarantees before changing lifetime management; no raw-pointer freeing while views/kernels still reference it.
- [ ] Compare paged/nonpaged identical-gradient numerical trajectories and normal/tiny/strict cases; then force controlled CPU eviction and page-in. Run an isolated oversubscription stress only when free device/host memory budgets permit it. Use an otherwise idle selected GPU. Nonmanaged pressure must leave at least 2 GiB device headroom. Cap the combined process allocation target at 110% of its initially free device capacity, managed test storage at 4 GiB, and host usage below 50% of MemAvailable. Compute the budgets before allocating; if other jobs or existing usage preclude isolation, skip pressure stress. If this cannot safely force eviction, report a skipped pressure qualification rather than a pass.
- [ ] Report resident-fit versus migration-heavy performance, actual managed memory/residency and guard/prefetch stalls. Add no default paging or speed claim without evidence. Commit fixes and scoped support statement; if the pinned allocator cannot meet the ownership contract, keep that case explicitly unsupported and report the blocker.

## Task 4 — Qualify compilation at honest support levels

**Files:** create `tests/test_compile_integration.py`; extend `tools/training_integration.py`, `tools/ddp_integration.py`; extract a small internal tensor helper only if the experiment justifies it.

- [ ] Add compiled-model/eager-optimizer parity tests with AMP, global/AdaGC, uint8 state, zero/tiny/strict paths, changing LR/active sets, checkpoint load and channels-last. Test one GPU before 2/4 DDP; use the documented DDP-before-compile order first.
- [ ] Collect `torch._dynamo.explain`/compiler logs, cold compile time and warm steady-state timing for `torch.compile(opt.step, fullgraph=False)`. Count captured graphs, breaks, recompiles and fully skipped frames; a whole-step eager fallback is not a compiled-optimizer pass.
- [ ] Test changing scheduler LR explicitly to catch frozen Python scalars. Compare optimizer/history/scaler skip behavior and next-step sign decisions with eager, preserving checks/hooks. Attempt `fullgraph=True` as an expected capability probe, not a requirement to bypass host safeguards.
- [ ] If a pure-tensor helper offers a measurable captured region, extract it behind the same reference oracle and rerun strict/rare-path tests; otherwise deliver compiled-model compatibility and document partial/full optimizer capture results without a performance promise. Do not start a device-only backend rewrite in this task.
- [ ] Commit the bounded integration and separately label model compilation, partial optimizer compilation, and fullgraph/CUDA-graph status.

## Task 5 — Demonstrate trainer and DataLoader worker recovery

**Files:** create `examples/_training_state.py`, `tools/restart_integration.py`, `tests/test_training_restart.py`; extend `examples/train_ddp.py` and integration runners. Optional torchdata dependency remains in an integration extra/qualified environment, not core imports.

**Interfaces:** reference helper `save_training_checkpoint(path, *, model, optimizer, scaler, scheduler, data_state, process_group=None)` and matching load return trainer counters/data/RNG state. CLI: `tools/restart_integration.py --world-size N --loader-workers W --case trainer|loader|checkpoint-write --output DIR`. Exact public packaging of helpers is not required; examples are the initial contract.

- [ ] Start with failing comparisons of uninterrupted and restart trajectories, saving per-rank RNG, consumed data, optimizer-success counts and partial accumulation policy. Test successful and overflow-skipped attempt boundaries.
- [ ] Implement atomic multi-component checkpoint generations with a final committed manifest and checksums. Save base module state independent of DDP/compile wrapper names. Give sampler/shuffle and loader worker seeding separate explicit generators, snapshot/reconstruct their data state, and finish model/optimizer/loader/iterator reconstruction before restoring trainer CPU/CUDA RNG. Iterator base-seed creation must not consume the trainer's default CPU RNG. Verify next CPU/CUDA random draws and a stochastic model step after rollback, including a prefetching loader, against uninterrupted execution. Checkpoint at quiescent boundaries with state-producing CUDA work complete; do not start asynchronous save/next-step overlap in the initial implementation. Save rank-local model buffers rather than assuming DDP synchronizes all buffer values at checkpoint time.
- [ ] Inject one-time controlled failures at: before forward, inside accumulation, after backward, after an optimizer step but before checkpoint commit, during checkpoint writing, and just after manifest publication. Use a durable failure marker so elastic retries do not fail forever. Restart the full torchrun group from the last committed generation; timeout each case after 600 seconds and record expected failure/recovery separately.
- [ ] Kill a DataLoader child with workers 2/4, let the trainer/full torchrun group roll back to the common checkpoint, and recreate the loader; test prefetch_factor 1/2 and persistent workers on/off. Compare consumed sample IDs and stateless augmentation outputs with uninterrupted execution. Prove that the cursor counts consumed batches even when AMP skips, and does not count merely prefetched batches.
- [ ] Add StatefulDataLoader with per-rank loader snapshots for a restartable stateful dataset/iterable. Test/validate its worker-count restrictions; reject uncheckpointable external streams and changed-world-size exact replay clearly. Detect corrupt/incomplete checkpoints without partially loading an optimizer.
- [ ] For fixed world size and placement, require bitwise numerical/data continuation on a deterministic fixture: deterministic algorithms on; cuDNN benchmark off/deterministic on; TF32 off; CUBLAS_WORKSPACE_CONFIG=:4096:8; fixed DDP buckets and recorded NCCL algorithm/protocol/topology. Compare per-rank model buffers too. For remapped physical GPU IDs, require bitwise restored state/data/RNG and continuation against a control on that placement, not an unconditional old-topology bitwise trajectory. Mid-accumulation failures roll back/replay; arbitrary mid-group checkpoint loading remains unsupported unless gradients/scale state are implemented and tested.
- [ ] Run intersections: DDP+AMP+AdaGC restart, channels-last restart, actual paging reconstruction, compiled-model cache recreation. Commit trainer examples and exact guarantees/limitations.

## Task 6 — Investigate FP8 with a hardware-gated positive suite

**Files:** create `tests/test_fp8_integration.py`; add `--suite fp8` to the single-rank runner and an FP8 option to DDP/restart runners; add an optional `examples/train_fp8.py`. Do not add mandatory optional-backend imports to `lionplus8bit`.

- [ ] On SM75, test an explicit unsupported-native-FP8 result before model/optimizer state allocation. `--require-fp8` must fail clearly rather than silently run FP16. Dependency presence, casting and emulation are not positive hardware evidence.
- [ ] Pin qualified optional backend versions in the result manifest. First adapter: Transformer Engine with FP32 parameters/ordinary FP32 gradients, FP8 compute context and fused `.main_grad` accumulation disabled. Check actual backend capability as well as device architecture; preserve existing rejection of actual FP8 primary parameters.
- [ ] On a compatible runner, test FP8 forward/backward with global/AdaGC and uint8/FP32 momentum. Verify gradients entering clipping are dequantized and loss-unscaled; compare matched-gradient replay independently of FP8 model approximation. Check clipped/raw norms, dtype/stride ownership, update signs, backend cached-weight refresh and scale/amax extra state.
- [ ] Test overflow skips without falsely requiring FP8 forward metadata to remain unchanged. Test checkpoint resume including backend extra state and two-/four-GPU FP8 DDP where hardware exists, coordinating backend scaling groups consistently with DDP. Record untested GPU counts explicitly.
- [ ] Evaluate torchao as a second backend with its own tensor-subclass/recipe constraints; it must pass its own tests before a support label. Fullgraph/FP8-primary-weight or specialized `.main_grad` adapters require a separate contract if encountered.
- [ ] Run paired FP8/high-precision quality and memory/throughput studies only on capable hardware. Until then, deliver capability handling, example code and pending positive commands; status remains hardware-blocked on this node, never "FP8 verified".

## Task 7 — Combined regression, benchmarks, documentation and CI

**Files:** create `tools/benchmark_integrations.py`, `docs/validation-integrations.md`; modify `README.md`, `.github/workflows/cuda.yml` or create a separate manual multi-GPU workflow. Keep historical evidence drivers unchanged.

- [ ] Run the original suite plus new focused tests, then selected intersections: 4-GPU DDP+AMP+AdaGC+channels-last; DDP+paging; compiled model+DDP; restart with each supported storage mode. Positive FP8 intersections stay hardware-gated. Negative cases must terminate with all child processes cleaned up.
- [ ] Implement balanced performance campaigns: 3 independent repetitions, 100+ warmup and 300+ timed updates per reported case; ordinary/zero/tiny/recovery segments; strong scaling at fixed global batch and weak scaling at fixed per-rank batch. Record actual samples per condition, slowest-rank latency/aggregate throughput, numerical errors/sign margins, allocated/reserved/native-managed memory and migration/communication costs. Time profiling/diagnostic runs separately.
- [ ] Compare unchanged contiguous eager baselines against this revision, with a prespecified upper 95% paired slowdown bound of +10%; use more repetitions or label inconclusive rather than claiming a pass from noisy data. New modes report costs without a forced speedup assumption. Reuse the three-seed +2% quality-margin protocol where update numerics change; an inconclusive quality interval remains inconclusive.
- [ ] Add manual single-/2-/4-GPU CI with fail-on-insufficient-device flags, bounded timeouts and per-rank artifacts. FP8-positive jobs require an explicitly capable runner; SM75 negative tests are separate. Preserve CPU Python3.10/torch2.5 CI and narrower optional version matrices; do not assume a self-hosted runner is already registered merely because GPUs exist.
- [ ] Replace broad "Unverified" labels with precise tested configurations, expected errors, unsupported hardware and unresolved cases. Publish reproducible commands/raw manifests/source hashes and independent final review. Commit each qualified stage; publish only when requested.

## Planned execution commands

These are interfaces to build, not commands claimed to work in the current revision. Use fresh output directories and run GPU suites sequentially.

```bash
python tools/training_integration.py --suite capabilities --output results/integrations/capabilities
python -m pytest --require-cuda -q
CUDA_VISIBLE_DEVICES='' python -m pytest -m 'not cuda' -q

torchrun --standalone --nnodes=1 --nproc-per-node=2 \
  tools/ddp_integration.py --suite core --output results/integrations/ddp2
torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  tools/ddp_integration.py --suite amp --output results/integrations/ddp4-amp
python -m pytest --require-gpus 4 -m multigpu -q

python tools/training_integration.py --suite layouts --output results/integrations/layouts
python tools/training_integration.py --suite paging --output results/integrations/paging
python tools/training_integration.py --suite compile --output results/integrations/compile
python tools/restart_integration.py --world-size 4 --loader-workers 2 \
  --case trainer --output results/integrations/restart4
python tools/training_integration.py --suite fp8 --require-fp8 \
  --output results/integrations/fp8-required
```

The last command must report unsupported hardware on this SM75 node. Run it successfully only on a backend-supported FP8 runner. CLI defaults must not silently replace required CUDA/NCCL/FP8 with a weaker path.

## Completion ledger

- [x] Task 0 baseline/capability evidence.
- [x] Task 1 standard DDP qualified before changing layouts.
- [x] Task 2 layout staging and checkpoint v3 qualified.
- [x] Task 3 actual paging storage/lifecycle/migration verdicts.
- [x] Task 4 compilation levels measured and honestly labeled.
- [x] Task 5 trainer/DataLoader failure recovery proved within contract.
- [x] Task 6 SM75 negative capability tests; capable-hardware positive results or explicit hardware block.
- [x] Task 7 intersection evidence, performance/quality verdicts and support matrix.
