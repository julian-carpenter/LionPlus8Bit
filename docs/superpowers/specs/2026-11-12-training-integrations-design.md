# Training integrations: design and validation contract

Status: proposed for review; no implementation or training validation has run for this proposal. Baseline: `0fb6a5742c6b27e4445e05b401410d6083c45190`.

## Intent and order

Qualify standard DDP with FP16 AMP and AdaGC on this four-GPU node first. Then add safe channels-last/other noncontiguous weight layouts, investigate and repair native paging, qualify compilation in distinct levels, and demonstrate recovery from trainer and DataLoader worker failures. Investigate FP8 as a separate compute integration; do not confuse it with this optimizer's uint8 momentum encoding.

Each stage should produce a separately reviewable change and an evidence-backed support statement. Testing a feature may show that no optimizer change is necessary. A failed experiment is a recorded result, not permission to disable numerical safeguards or silently fall back and claim support.

## Environment actually inspected

- Four Quadro RTX 6000 GPUs, each 24 GiB, compute capability 7.5 (Turing); all four visible to PyTorch.
- Driver 580.178.04; PCIe PIX paths between GPUs, no NVLink links reported. CUDA peer-access queries returned true for every distinct pair; actual NCCL transport is still to be tested.
- Python 3.12.3; NVIDIA PyTorch `2.13.0a0+9186a08b2c.nv26.07`, CUDA 13.3; NCCL available, version 2.30.7.
- bitsandbytes 0.50.2 from the verified dependency directory, selecting its CUDA 13.2 binary.
- Installed optional package metadata: torchao `0.18.0+git18278f9b`, Transformer Engine `2.17.0+2e559f06`, torchdata 0.11.0, Triton `3.7.1+gitf797708c.nv26.7`. Presence does not establish working imports or compatibility.
- Native BF16 and native FP8 training performance cannot be qualified on SM75. Use FP16 autocast with FP32 parameters here. The FP8 positive gate needs a backend-supported Ada/Hopper/Blackwell GPU and a qualified software tuple. [Transformer Engine hardware scope](https://docs.nvidia.com/deeplearning/transformer-engine/getting_started/index.html).

Only repository/runtime inspection and capability queries were performed while preparing this design. No dependencies were installed and no distributed training, paging-pressure, compile, restart or FP8 experiments were run.

## Approach

Recommended: retain the eager single-device optimizer as the numerical reference and add independent integration harnesses. Change core execution only where a failing integration test demonstrates the need. Keep distributed process/data recovery in a reference training layer rather than teaching `optimizer.step()` to own datasets or process groups.

Alternatives considered:

1. Rewrite the optimizer around device-only fused kernels and a generic distributed trainer. This could remove host synchronization but combines unrelated changes to update numerics, public APIs and recovery semantics; it is too broad for initial qualification.
2. Remove the current rejections or trust third-party defaults without dedicated tests. This risks updating temporary weight copies, miscounting distributed norms, restoring nonpaged state as if paged, or claiming compilation that ran entirely eagerly.

This recommendation and the scope choices below are proposed engineering decisions, not prior user approvals of detailed implementation choices.

## Common invariants

- Keep existing global/AdaGC equations, successful-step warmup counting, zero/None distinction, per-tensor strict persistence, FP32 masters and quantized-state block size 256.
- Preserve original parameter objects, weight storage/layout and gradient objects/values across successful steps, apart from the documented AMP unscale performed by GradScaler outside the optimizer.
- No sparse-gradient, FP64/complex-parameter, FSDP/ZeRO shard or one-optimizer-across-devices support is implied.
- Core dependency bounds remain Python >=3.10, torch >=2.5,<3 and bitsandbytes==0.50.2. Optional integrations may have a narrower tested software tuple, documented separately; do not silently raise the core floor or upgrade dependencies in the user's base environment.
- Retain CPU CI and existing CUDA regressions. Preserve old snapshots/evidence, and write fresh immutable result directories with commands, source/dependency hashes, environment and explicit pass/fail/skip/block reasons.

## 1. Standard DDP and AMP contract

One process and one optimizer per GPU, replicated full parameters, NCCL gradient reduction, equal update boundaries on ranks. Test one-, two- and four-GPU cases. All ranks must start with the same model, optimizer policy, GradScaler state and LR schedule.

DDP has already reduced/averaged gradients when synchronized backward completes. Compute Lion+ global norms and AdaGC tensor norms from those gradients once per replica. Do not sum replica norms again across ranks. `no_sync()` must cover forward and backward on accumulation-only microbatches; clipping occurs only after the final synchronized backward. [DDP semantics](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html).

Test normal/zero/None/tiny gradients, active-set changes, threshold crossing, momentum sizes below/above 4096, bucket gradient views, gradient preservation, accumulation, warmup boundaries and rank-local loss overflow before reduction. Injecting overflow before the reducer must make every rank skip, keep optimizer state unchanged, update the scaler consistently and leave the scheduler unchanged.

Explicit boundaries:

- A nonfinite mutation introduced on one rank after DDP reduction can cause local GradScaler decisions to diverge. Include it as a negative test. The checked reference driver must detect and stop all ranks before updates; ordinary DDP qualification must not imply arbitrary rank-local postprocessing is safe. An optional coordinated skip implementation can be added only with public, tested AMP semantics, including the growth tracker; no private `found_inf` patching.
- Unequal batch sizes at the same update boundary need explicit sample-count weighting. Test this separately from one rank finishing its training loop early.
- DDP `join()` can reconcile model replicas after uneven input exhaustion without reconciling optimizer/AdaGC/scaler histories. Initial examples require equal update counts and use fail-fast early termination. Arbitrary uneven-join continuation stays outside the support claim unless state synchronization is designed and tested.
- Construct DDP with all parameters that may receive gradients already registered as trainable. Test identical branch-activation schedules. Parameters created or excluded with `requires_grad=False` at DDP construction need a separately coordinated DDP rebuild before true unfreezing; that rebuild is outside the initial contract.

Use two oracles: bitwise rank/replay equality with identical already-reduced gradients; and a real distributed forward/backward comparison against a same-global-batch reference. For the latter, reduction/backward order can change gradients near sign boundaries; report gradient error, sign margins and disagreements rather than hiding differences behind a loose parameter tolerance.

## 2. Weight-layout contract

Support channels-last Conv2d, channels-last-3d Conv3d, and dense nonoverlapping transposes/permutations first. Investigate stepped slices as a separate case; keep them rejected until internal-overlap and shared-storage checks are reliable. Expanded/overlapping views and distinct overlapping parameters remain rejected. Disjoint contiguous parameter views remain supported.

Define optimizer momentum and master-weight storage in logical row-major parameter order, independent of the weight's physical strides. Native kernels receive contiguous work tensors. For a noncontiguous FP32 weight, stage a temporary contiguous work weight, update it and copy back into the original parameter. With an FP32 master, keep the master canonical and copy back into the original weight layout. Preserve data pointer, storage offset, stride and gradient identity. Never invoke the backend path that assigns `p.data = p.data.contiguous()` on a noncontiguous parameter.

No persistent FP32 shadow for every FP32 parameter. Scratch is per updated tensor, not the sum of all noncontiguous parameters. Layout overhead must appear in benchmark results. Keep a stride-aware-kernel alternative as a measured follow-up if staging is too costly.

Introduce checkpoint schema v3 for explicit `logical-row-major-v1` state order and parameter shape mapping in both clipping modes. Load v1/v2 through their existing rules; v2 weights were contiguous so their native state already has this order. Permit same-shaped v3 checkpoint continuation into contiguous or supported noncontiguous targets. Validate mappings before mutations; layout can change at a fresh load boundary, not during an initialized trajectory.

## 3. Paging contract

`is_paged=True` must represent actual native managed-memory state on eligible buffers, not only a saved flag. Pinned backend source pages tensors with at least 100,000 elements, independently of the 4096-element quantization threshold. Its managed tensor has a CPU PyTorch device label and `is_paged`/`page_deviceid` attributes, so `.device` and `torch.cuda.memory_allocated()` alone cannot validate residency or usage.

Investigate guard access, strict arithmetic, device selection, prefetch ordering, default/caller stream dependencies, lifecycle and repeated optimizer creation/deletion. Test uint8 and FP32 state, 99,999/100,000/100,001 elements, a block tail, masters and each GPU ordinal. Preserve ownership and correct numerical state without a full GPU shadow of paged momentum.

Checkpoints save values and allocation policy, never ownership of a raw pointer. On load, reconstruct eligible managed buffers for the current local device. Do not trust deserialized transient paging attributes or retain a stale source GPU ordinal. Default and deferred migration both need true paged continuation tests. Profile guards before prefetch: otherwise a correctness guard may itself cause page migration.

Separate residency/page-in correctness from oversubscription performance. Start with controlled CPU eviction/prefetch; run pressure tests only in isolated subprocesses on one GPU with detected free-VRAM/host-RAM budgets, a timeout and cleanup. Use NVML/device-level memory plus process RSS and migration tracing when available; CUDA allocator statistics miss native managed allocations. Do not promise universal speed or a resident-memory reduction when all pages fit on GPU. [Native paging explanation](https://huggingface.co/docs/bitsandbytes/explanations/optimizers).

## 4. Compilation contract

Report three distinct levels:

1. Compiled model forward/backward with eager LionPlus8bit: first compatibility target, on one GPU then DDP.
2. `torch.compile(optimizer.step, fullgraph=False)`: investigate actual captured graphs and graph breaks, numerical parity, warmup compilation time, recompilations and steady-state performance.
3. Fully captured optimizer/CUDA graphs: not promised by this plan. Host `.cpu().tolist()`, Python decisions and the native host-float scale ABI are concrete blockers; a device-only backend redesign needs a separate design if required.

A successful call with the whole optimizer skipped by Dynamo counts as eager compatibility, not a compiled optimizer. Do not disable guards, freeze changing LR values or bypass AMP skips to make a trace succeed. First test the PyTorch-recommended DDP-before-compile wrapper order, then document any version-specific alternative. [DDP compilation ordering](https://docs.pytorch.org/docs/2.15/notes/ddp.html), [optimizer compilation recipe](https://docs.pytorch.org/tutorials/recipes/compiling_optimizer.html).

Test ordinary/clipped/strict paths, changing active sets and LR schedules, checkpoint loads and supported layouts. Keep compiler-generated caches out of portable checkpoints. Any extracted tensor helper remains optional until its independent oracle passes and measured benefit justifies maintaining it.

## 5. Recovery contract

Cover two failure types: trainer/DDP-rank termination and DataLoader subprocess termination. Initial guarantee: fixed world size, known deterministic/restartable datasets and recovery from the last fully committed checkpoint. A crash can occur at any instruction; recovery replays work after that checkpoint. It does not resurrect a half-finished CUDA kernel or partially written optimizer step.

Save model/buffers, optimizer policy/state/masters/history, scaler, scheduler, per-logical-rank Python/NumPy/CPU/CUDA RNG, sampler epoch/order and committed consumed-data position. Preserve rank-local model buffers when the model can have them; replicated parameter equality does not prove all buffers are identical. Consumed batches advance even when AMP skips an update; successful optimizer/scheduler counters advance separately. For mid-accumulation crashes, replay the entire unfinished accumulation group from the prior checkpoint. Do not promise arbitrary mid-group checkpoint restoration without also saving partial gradients and matching scale state.

Use temporary files, complete per-rank components, checksums and an atomically published committed manifest. Ignore incomplete checkpoints. On a rank failure, restart the full worker group and reload one common committed generation; physical GPU IDs may change. A changed world size is a separate statistical-resume contract and initially fails clearly. [Torchrun recovery responsibilities](https://docs.pytorch.org/docs/main/distributed/elastic/run.html).

Bitwise end-to-end replay is qualified on a deterministic fixture with fixed GPU placement and software/collective settings: deterministic algorithms enabled, cuDNN benchmarking disabled, cuDNN deterministic mode enabled, TF32 disabled, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, fixed DDP bucket configuration and recorded NCCL algorithm/protocol/topology. RNG restoration alone cannot guarantee a bitwise trajectory for arbitrary kernels. On physical-device remapping, require bitwise restored checkpoint values/data/RNG and correct continuation against a control on that new placement; do not promise bitwise equality to an uninterrupted run on a different collective topology.

A DataLoader-child failure initially causes the parent trainer to fail and the full trainer group to roll back to the common checkpoint, recreating its loaders. In-place hot replacement of one child while keeping queues and trainer state live is outside the first contract.

Give sampler/shuffle and DataLoader worker seeding independent generators; iterator creation must not consume the trainer's default CPU RNG. Capture/reconstruct data-generator state through the chosen loader contract, and restore trainer CPU/CUDA RNG after model, loader and iterator reconstruction. Verify the next CPU/CUDA random draws and a stochastic model step against the uninterrupted control. Saving RNG before rebuilding an iterator is insufficient.

For map-style datasets, use replayable sample IDs and augmentation seeds derived from logical sample/epoch/occurrence rather than worker arrival order. Test workers 0/2/4, prefetch and persistent workers. Use optional torchdata StatefulDataLoader for stateful datasets, saving each rank's loader state; it does not coordinate ranks for us. Stateful iterable recovery may require the same worker count. External streams or side effects without checkpoint/replay support remain outside the guarantee. [Stateful loader contract](https://meta-pytorch.org/data/main/stateful_dataloader_tutorial.html).

## 6. FP8 compute integration

Use Transformer Engine first with high-precision FP32 trainable parameters, ordinary FP32 `.grad`, and its FP8 compute context. Disable specialized fused `.main_grad` accumulation for the first contract. Torchao is a second explicitly qualified backend, not an assumed equivalent. Optional dependencies remain optional.

Optimizer uint8 momentum is not an FP8 tensor. The optimizer must receive dequantized, AMP-unscaled numerical gradients, not raw FP8 codes or backend scale factors. No `fp8=True` optimizer flag and no silent cast of actual FP8 primary parameters are planned.

On this SM75 node, verify capability checks and a clear unsupported-device result before any FP8 training allocation. Emulated casts may test a contract but cannot qualify acceleration or native FP8 numerics. Positive FP8 forward/backward, single-/multi-GPU and performance tests require suitable hardware; both architecture and the backend's own capability check must pass.

On a capable runner, compare FP8 versus that backend's high-precision mode, then matched-gradient uint8 versus FP32 momentum. Check FP8 scale/amax state, cached weight refresh after optimizer updates, clipping histories and restart. FP8 metadata can advance during forward even if GradScaler skips the optimizer; assert optimizer/scaler behavior against backend semantics instead of incorrectly demanding all model metadata remain unchanged. Save backend extra state through its supported checkpoint API. [Transformer Engine FP8 scaling](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/fp8_current_scaling/fp8_current_scaling.html), [torchao training](https://docs.pytorch.org/ao/stable/workflows/training.html).

## Evidence and release criteria

Keep numerical correctness, compatibility, throughput and task quality as separate verdicts. No failures, expected skips or unsupported hardware may silently turn into green positive-feature claims. Core single-device behavior must not regress. Add intersection tests after individual stages; do not claim the entire Cartesian product from isolated successes.

Use 3 independent paired performance repetitions, at least 100 warmup and 300 timed updates for reported latency, balanced arm order and per-rank timings. Report the slowest-rank step time and aggregate examples/s, optimizer-only timing separately, cold initialization/compile cost, persistent/transient memory and readback/transfer counts. Small smoke runs establish correctness only. Prespecify a +10% maximum upper confidence bound for regressions in unchanged contiguous baseline workloads; investigate or retain an opt-in path when that gate fails. New staging/paging/compile modes have mode-specific costs and should not be advertised as faster without measured evidence.

Final deliverables: qualified support matrix, launch examples, version/hardware requirements, raw JSON/logs/checkpoints and code hashes, plus explicit unresolved items. Initial FP8 status on this node must remain hardware-blocked, and fullgraph optimizer/changed-world-size recovery must not acquire an unsupported support claim.
