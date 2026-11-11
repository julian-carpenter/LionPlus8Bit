# Fresh code audit: LionPlus8Bit, with emphasis on AMP and AdaGC

Date: 2026-11-11. Audited revision: **`64205431d701415de6aaa8355bcc536f81626cd5`**. This is an audit and implementation handoff; no optimizer, test, benchmark, or CI code was changed. Findings below remain unfixed in that revision.

The ordinary AMP path passed the new controls: FP32 parameters with FP16 autocast and GradScaler reached the same parameters and optimizer state as an unscaled reference, including uint8 momentum. Overflow skips preserved native state, AdaGC history, and counters. However, the implementation has substantial fallback costs and several conditional correctness defects. The most useful next work is to contain numerical fallback to affected tensors, avoid replaying normal tensors because another tensor is zero, and make checkpoint override ownership and precedence reliable.

The [evidence bundle](../evidence/audit-2026-11-11/README.md) contains original probes, raw results, eight CUDA profiler traces, independent review notes, source hashes, and commands. Some probes deliberately reproduce failures and exit successfully after recording them. **A zero process exit code is not a claim that every case passed.**

## Scope and evidence standard

Reviewed all package modules, tests, examples, CI workflows, benchmark/study code, and the relevant bitsandbytes 0.50.2 optimizer, CPU/CUDA backend, and module-override code. Three independent reviews examined numerical/AMP behavior, checkpoint/state behavior, and performance/stream assumptions. CUDA measurements were run sequentially by the primary reviewer to avoid contention between audit workers.

Fresh execution used Python 3.12.3, PyTorch `2.13.0a0+9186a08b2c.nv26.07`, CUDA runtime 13.3, bitsandbytes 0.50.2, and a Quadro RTX 6000, SM75, 24 GiB, driver 580.178.04. bitsandbytes selected its CUDA 13.2 binary. This is one environment, not validation of every declared dependency version or GPU architecture. SM75 does not establish native BF16 performance.

The existing suite was rerun: **79 passed, 15 subtests passed, 16 warnings**, with `--require-cuda`. Additional independent probes found the failures below. Existing passing tests therefore do not close the uncovered branches.

Priorities describe recommended engineering order: **A** means address in the next corrective pass; **B** means a confirmed conditional defect or compatibility gap; **C** means a low-frequency boundary. They are not claims that every training job fails.

| ID | Priority | Type | Confirmed finding | Execution evidence |
| --- | --- | --- | --- | --- |
| F01 | A | Performance/memory | One tiny value permanently selects strict arithmetic for every active tensor | CPU, CUDA, real AMP |
| F02 | A | Performance/memory | One exact-zero gradient causes FP64 norm recovery over all active gradients | CPU, CUDA, real AMP |
| F03 | A | Correctness | Saved native policy loses to destination overrides; saving can omit effective overrides | CPU, separate-process resume |
| F04 | B | Correctness | Loaded overrides alias the caller checkpoint and other resumed optimizers | CPU |
| F05 | B | Correctness | Deferred CPU-to-CUDA checkpoint migration fails before migration | CUDA, both clipping modes and state dtypes |
| F06 | B | Numerical correctness | Strict whole-tensor normalization erases independently representable quantization blocks | CPU and CUDA |
| F07 | B | Numerical correctness | CPU AdaGC rejects finite gradients before FP64 norm recovery | CPU, FP32/BF16 |
| F08 | B | Integration | Dynamically added StableEmbedding loses its FP32 momentum policy | CPU |
| F09 | B | Compatibility | Public load-state hooks are accepted but never called | CPU; inherited backend behavior |
| F10 | C | Numerical boundary | Exactly representable minimum FP32 clipping bound can be rejected | CPU |

## F01 — A single tiny value permanently changes all active tensor updates

Locations: `_numerical_guard.py:78–110`, `_clipping.py:45–52`, `_optimizer.py:292–299`, `487–493`, `516–554`. The global policy has the same aggregate guard behavior.

The guard returns one scalar flag across every active gradient and momentum state. AdaGC broadcasts that flag into every packed row. Every active parameter then gets `strict_math=True`, which persists on later ordinary steps and in checkpoints. The code comment mentions FP32 state, but the persistent flag also affects uint8 momentum. A tiny auxiliary scalar therefore changes execution of an otherwise normal large matrix.

The strict uint8 path constructs int64 indices, decoded FP32 momentum, direction/update temporaries, and a normalized tensor before requantization. Persistent momentum remains uint8; the problem is the temporary peak and repeated work, not a permanent FP32 shadow allocation.

In a replay with a 1,048,576-element FP32 parameter and one auxiliary scalar, a single `1e-25` scalar gradient set **both** strict flags. After restoring ordinary gradients, AdaGC retained **20,975,104 bytes** of additional peak allocated memory, versus **6,144 bytes** on the ordinary path. Median step latency was 1.891 versus 1.447 ms. The global policy retained 25,167,872 versus 2,560 bytes. A real FP16-autocast/GradScaler training probe independently reproduced the propagation and approximately 20 MiB temporary peak.

Recommended change: return per-tensor guard decisions and restrict strict arithmetic to affected tensors. Specify whether persistence is numerically required for each state format; do not clear flags arbitrarily or disable protection. A blockwise or chunked strict update should avoid full int64 decoding and large simultaneous scratch tensors. If old checkpoints already mark every tensor strict, define a safe migration/reassessment policy.

Acceptance: normal → tiny auxiliary tensor → normal → checkpoint resume must preserve numerical behavior while keeping unrelated tensors on their normal path. Include uint8 and FP32 momentum, genuine subnormals, and per-tensor clipping scales. Record dispatch/strict counts and temporary peaks. Large normal tensors must not acquire parameter-sized strict scratch because of another tensor.

Evidence: `root/profile_fallbacks.py`, `root/fallback-profile.json`, `root/*-recovered.trace.json`, `root/probe_real_amp.{py,json,log}`; CPU control in `perf/cpu_probes.json`.

## F02 — An exact-zero gradient causes full FP64 norm replay

Location: `_optimizer.py:261–274`.

The fallback predicate includes every norm below `1e-18`, including exact zero. If one scalar gradient is zero, the optimizer recomputes **all** active norms in FP64, repeats clipping preparation, scans every gradient/state again, and performs another packed readback. A separate finite-check boolean adds a third readback point. Exact-zero gradients are ordinary in masked branches and auxiliary losses.

In the same 1,048,576-element replay, one zero scalar increased median AdaGC step latency from **1.447 to 2.292 ms** and temporary allocated peak from **6,144 to 8,394,240 bytes**. The profiler shows an **8,388,608-byte cast of the unrelated large gradient**. Optimizer stream synchronizations increased from one to three; trace-level device synchronizations also include the measurement harness and must not all be attributed to the optimizer. No strict flags were set by zero itself.

Recommended change: distinguish an exact-zero tensor from a nonzero tensor whose FP32 norm underflowed, and recover only suspect tensor norms. A simple `norm > 0` condition is insufficient: it would miss nonzero gradients whose initial reduction became zero. Preserve robust recovery for finite reduction overflow and genuinely tiny gradients without replaying unrelated normal tensors.

Acceptance: mixed normal/zero, all-zero, truly underflowed nonzero, finite-overflow, and genuinely nonfinite gradients. Active zero tensors must retain their momentum, decay, history, and step semantics. Replacing zero gradients with `None` is not an equivalent workaround. The ordinary matrix plus zero auxiliary scalar must avoid a matrix-sized FP64 temporary and duplicate scans of that matrix.

Evidence: `root/fallback-profile.json`, `root/adagc-zero.trace.json`, `root/probe_real_amp.json`, `perf/cpu_probes.{py,json}`.

## F03 — Checkpoint native overrides are neither authoritative nor fully serialized

Locations: `_optimizer.py:343–348`, `466–474`, and first-step `check_overrides()` calls at `175–176` and `284–286`. Pinned backend: `optim/optimizer.py:286–303`, `343–365`.

There are three related failures:

1. The backend merges index overrides and then parameter-ID overrides. Loading an empty saved parameter override does not remove a destination index override. A checkpoint with LR `.1` resumed with a destination's registered `.7` LR; its next parameter became `.2` instead of `.8`.
2. Registered module overrides run again on the first resumed step. A saved `.125` LR was visible immediately after loading, then became `.7` when stepping.
3. Serialization chooses the parameter override **or** the index override instead of merging their effective values. An optimizer operating at LR `.7` saved an empty override. In a clean process, the next update used `.1`: the parameter became approximately `.1` instead of the uninterrupted `-.5`.

These reproductions use public `GlobalOptimManager.override_config`, `register_parameters`/module registration, and normal save/load APIs. They do not require manually corrupting checkpoint fields or unrelated manager entries. The README's saved-policy precedence statement is therefore false for these cases. Similar policy resolution affects betas and allocation settings, although the reproduced numerical discrepancies use LR.

Recommended change: define one effective override policy owned by the optimizer. Serialize the resolved merge consistently, and restore saved settings without leaving matching destination index/module settings able to replace them. Do not clear unrelated optimizers' process-wide manager state. Deliberate changes made through supported APIs after loading should remain possible.

Acceptance: conflicting destination index and module overrides; an empty saved override; a parameter allocated lazily after loading; a second save and clean-process reload; multiple optimizers sharing the manager. Compare effective settings and actual continuation, not just checkpoint dictionary equality.

Evidence: `state/probe_state.py`, `state/probe_state_results.json`, `state/probe_split_override_resume.py`, `state/split_override_resume_results.json`.

## F04 — Loaded override dictionaries share ownership with checkpoints

Location: `_optimizer.py:466–468`.

The wrapper installs the caller's override dictionary directly into the global manager. The backend's later deepcopy does not protect this earlier assignment. Loading one checkpoint into two optimizers gives both parameter IDs the same dictionary. Changing one optimizer's native LR from `.125` to `.5` also changed the input checkpoint and the second optimizer. Both parameters updated to `.5`; the independent second result should have been `.875`.

Recommended change: take independent ownership of restored override data before publishing it. Deep-copy supported nested configuration values as necessary. Keep this fix separate from F03's policy-precedence logic so both properties are tested.

Acceptance: load one checkpoint into two fresh optimizers, change one override, and assert that the checkpoint, source optimizer, other optimizer, and its next update stay unchanged. Evidence: `state/probe_state_results.json`, key `override_alias`.

## F05 — `move_to_device=False` breaks CPU-checkpoint-to-CUDA resume

Locations: `_optimizer.py:231` versus `235`, and `262` versus `286`; `_numerical_guard.py` state-pointer inputs.

After `torch.load(..., map_location="cpu")`, `load_state_dict(..., move_to_device=False)` intentionally leaves native state on CPU. The inherited backend normally migrates it on the first step. This wrapper scans the state **before** `to_gpu()`. The Triton guard receives a CPU state pointer and raises:

```text
ValueError: Pointer argument cannot be accessed from Triton (cpu tensor?)
```

All four combinations of global/AdaGC and 17/4097 elements failed. All four default `move_to_device=True` controls passed. Failed steps left parameters unchanged. This is a failure of the explicitly exposed deferred-loading option; it does not invalidate the default README resume example.

Recommended change: place required state tensors on the appropriate device before any guard consumes them, while preserving validation ordering and failure semantics. Alternatively reject the option explicitly if deferred loading is intentionally unsupported; silently accepting it and failing later is inadequate.

Acceptance: both state dtypes and clipping modes, fresh CPU-loaded checkpoint to CUDA, default and deferred options, Triton and eager guards, plus a nondefault caller stream. Check continuation against default migration. Evidence: `root/probe_cuda_resume.{py,json,log}`.

## F06 — Strict requantization destroys a representable small block

Location: `_optimizer.py:524–554`, especially `536–538` and `554`.

The strict path divides the entire momentum tensor by one global peak before blockwise quantization. A sufficiently small independent block underflows during this division, although that block's FP32 scale and uniform endpoint codes could retain its momentum. This is additional cross-block loss, beyond ordinary within-block 8-bit quantization error.

Reproduction: 4096 FP32 elements, uint8 momentum, `max_grad_norm=inf`, `betas=(.9,.9)`, LR `.1`, no decay. One 256-element block has gradient `1e10`; another has `1e-39` on step 1, `-1e-40` on step 2, and zero on step 3. Use either clipping mode, with AdaGC warmup zero. An independent reference normalizes each block separately using the same codebook.

On CPU and CUDA, the first update agreed but the small block's momentum scale became zero instead of approximately `9.9999461e-41`. Step 2 produced **256/256 wrong signs** in that block: actual parameter `0.0`, reference `-0.2`. Step 3 remained `0.0` versus `-0.3`. The extreme magnitude range is deliberate; this is not evidence that ordinary training generally diverges.

Recommended change: preserve each block's representable range during strict requantization. Evaluate block-local scaling or a safe blockwise implementation that also limits scratch memory. Simply routing the case back to native arithmetic can reintroduce the flush-to-zero failure this fallback was intended to prevent.

Acceptance: three-step sign and state comparisons against an independent per-block reference; positive/negative tiny blocks, zero blocks, partial final blocks, CPU/CUDA, and nondefault-stream lifetimes. Retain all-tiny-tensor tests as a distinct case. Evidence: `perf/cross_block_probe.{py,json,log}` and `root/cross_block_cuda.{py,json,log}`.

## F07 — CPU AdaGC rejects finite reduction overflow before recovery

Locations: `_clipping.py:74–75`; `_optimizer.py:187–194`, `261–274`.

`prepare_clipping` immediately rejects nonfinite norms on CPU before `_adagc_step` can recompute them in FP64. With `p.grad=[1e20,1e20]`, every gradient value is finite, the FP32 norm is infinity, and the FP64 norm is `1.4142135907151756e20`. AdaGC raises `RuntimeError: Non-finite AdaGC gradient norm`; global mode succeeds and stores clipped momentum approximately `[.70710677,.70710677]` with zero betas and threshold one.

Confirmed for FP32/BF16 storage, both norm implementations, and warmup lengths zero and one. This is a CPU path defect. GradScaler's elementwise finite checks would not themselves reject these finite values.

Recommended change: recover suspect norms before a helper can reject them, then distinguish true nonfinite input from finite reduction overflow. Acceptance: the above combinations, established AdaGC history, preserved original gradients, and NaN/Inf rejection without parameter/history/native-state updates. Evidence: `amp/probe_amp.py`, `amp/probe_results.json`, `finite_norm_overflow` cases.

## F08 — Dynamically added StableEmbedding ignores its precision override

Locations: `_optimizer.py:118–138`, `175–176`, `284–287`; backend `nn/modules.py:99`.

`bnb.nn.StableEmbedding(64,128)` registers `optim_bits=32`. Included at optimizer construction, its momentum is FP32. Added using `add_param_group` after an earlier optimizer step, the same module gets uint8 momentum: initialization is already marked complete, so pending module overrides are never resolved. The size is 8192 elements, so this is not the small-tensor rule.

Recommended change: resolve module registrations for new parameters before their first state allocation. Coordinate this with F03 so fresh module defaults cannot overwrite a resumed saved policy.

Acceptance: global/AdaGC, delayed first gradient, checkpoint after group addition but before first use, preserved old histories, and an initially unseeded new history slot. Evidence: `state/probe_state_results.json`, key `dynamic_module_override`.

## F09 — Public checkpoint hooks do not surround the complete operation

Locations: `_optimizer.py:329–361`, `363–474`; pinned backend `optim/optimizer.py:192–271`.

Registering PyTorch `load_state_dict` pre- and post-hooks succeeds, but loading a valid checkpoint calls neither: the observed event list is `[]`, expected `['pre','post']`. The backend custom loader omits the hooks. Also, a `state_dict` post-hook sees only native `state` and `param_groups`; the wrapper adds LionPlus metadata afterward.

This inherited compatibility gap matters to integrations that adapt or validate checkpoint mappings through the public API. It is separate from normal save/load continuation, which passed controls.

Recommended change: honor hooks once around the complete wrapper operation, including pre-hooks before wrapper validation and post-hooks after native and clipping state are restored. Avoid double execution if backend behavior changes. If supporting this API is outside scope, explicitly reject/document it instead of silently ignoring registered callbacks.

Acceptance: registration/prepend order, returned replacement dictionaries, complete AdaGC metadata visibility, and exactly-once invocation. Evidence: `state/probe_state_results.json`, `load_hooks` and `state_dict_post_hook_keys`.

## F10 — Minimum FP32 clipping threshold has a rounding false rejection

Locations: `_clipping.py:95–102`, `_optimizer.py:277–279`.

For a scalar gradient `49.`, threshold `2**-149`, and zero betas, AdaGC reconstructs the clipped norm as `1.4012984643248169e-45`, one FP64 rounding step below the requested `1.401298464324817e-45`. It rejects the threshold although actual FP64 clipping followed by FP32 conversion produces the correct nonzero minimum FP32 value. Global mode succeeds. Warmup, bootstrap, and established minimum-valued history reproduce it. A truly below-minimum threshold remains correctly rejected.

Recommended change: validate the intended clipping bound and actual representable work result without treating reconstruction rounding as an unrepresentable request. Acceptance: scalar 49, valid neighboring bounds, a genuinely invalid bound, all three history phases, CPU and CUDA. Only CPU execution is established for this finding. Evidence: `amp/probe_results.json`, `minimum_threshold_boundary` cases.

## AMP and memory measurements

The replay below uses one large FP32 parameter with 1,048,576 elements and one scalar, eight ordinary warmup updates, and 20 synchronized timed steps. The large parameter has actual uint8 momentum. Timings exclude profiler instrumentation. Additional peak is `max_memory_allocated - memory_allocated` measured after live state initialization; it is not total training VRAM or reserved-memory usage.

| Policy / condition | Median ms | Additional peak bytes | Strict flags, large/scalar |
| --- | ---: | ---: | --- |
| Global / ordinary | 0.593 | 2,560 | false / false |
| Global / zero scalar | 0.662 | 2,560 | false / false |
| Global / tiny scalar | 1.267 | 25,167,872 | true / true |
| Global / restored ordinary scalar | 1.263 | 25,167,872 | true / true |
| AdaGC / ordinary | 1.447 | 6,144 | false / false |
| AdaGC / zero scalar | 2.292 | 8,394,240 | false / false |
| AdaGC / tiny scalar | 2.797 | 20,975,104 | true / true |
| AdaGC / restored ordinary scalar | 1.891 | 20,975,104 | true / true |

The independent real AMP probe uses `Linear(1024,1024,bias=False)`, FP32 weights, FP16 autocast, GradScaler, an auxiliary FP32 scalar loss, and actual uint8 momentum. Twelve measured successful updates per condition follow initialization. Timing includes `scaler.step`, `scaler.update`, and synchronization, excluding forward/backward. The recovered condition first injects one tiny auxiliary gradient, then restores its ordinary coefficient.

| AdaGC AMP condition | Median ms | Median additional peak bytes | Strict flags |
| --- | ---: | ---: | --- |
| Ordinary | 1.370 | 6,656 | false / false |
| Zero auxiliary gradient | 2.090 | 8,394,752 | false / false |
| Ordinary after one tiny auxiliary gradient | 1.746 | 20,975,616 | true / true |

Every condition passed an additional injected-overflow check: weights and complete optimizer state stayed unchanged and the scaler decreased from 32 to 16. These small, single-run measurements demonstrate paths and allocation sizes; they are not confidence-bounded throughput estimates or convergence studies. The large scratch allocations recur per affected tensor update; do not sum them across tensors as if every temporary necessarily lives simultaneously.

Ordinary GradScaler also makes its own finite-check decision before invoking the optimizer. Consequently, AdaGC's ordinary one packed readback is an optimizer-only property, not a one-synchronization guarantee for the full AMP iteration. See the [PyTorch AMP contract](https://docs.pytorch.org/docs/2.15/notes/amp_examples.html#working-with-unscaled-gradients); the executed runtime's implementation was also inspected in `perf/backend_notes.txt`.

## Verified properties and remaining limits

The following passed in this audit, within the stated runtime and probe inputs:

- Existing suite: 79 tests plus 15 subtests with real CUDA required.
- Independent CUDA AMP scale control: FP32 `Linear(128,64)` weights and uint8 momentum, scales 16/1024, explicit/implicit unscale, compared with an unscaled autocast reference. Parameters and complete optimizer state were bitwise equal after six attempts, including skips before initialization and at the warmup boundary. Power-of-two inputs isolate optimizer semantics from FP16 backward-rounding differences.
- Independent CPU BF16-autocast AMP controls: scale invariance, uneven/partial accumulation, active zeros, inactive parameters, closures, bootstrap/rebootstrap, and overflow state preservation.
- Six separate-process CPU AMP restart splits, with uint8 and FP32 momentum, an initially inactive parameter, a dynamic group, scheduler, scaler, and two overflow skips. Continuation matched uninterrupted execution. Only absent versus false inherited `differentiable` metadata was normalized.
- Twelve same-dtype restart cases: FP32/FP16/BF16 × 17/4097 elements × two split points. Parameters and complete optimizer state matched; histories and low-precision master weights stayed FP32.
- Default CPU-checkpoint-to-CUDA migration passed all four tested combinations. No new stream-lifetime defect was established.

The existing AdaGC AMP unit test uses `Linear(8,4)`, so its parameters are too small to exercise uint8 momentum. Several replay controls reuse the package's own update implementation and cannot independently validate strict requantization. Passing separate tests for AMP, uint8, strided gradients, and streams does not establish their complete cross-product.

The adaptive rule is consistent with the intended previous-history threshold and clipped-norm EMA in ordinary cases. Warmup counting, positive bootstrap, zero handling, and tiny-history exceptions are deliberate package extensions; they should not be mistaken for a proof that the original globally clipped Lion+ theory covers this variant. The [AdaGC paper](https://arxiv.org/html/2502.11034v4) supports the algorithmic context, not the correctness or performance of these implementation-specific paths.

Still unresolved or outside tested support:

- DDP rank consistency with unused parameters, FSDP/ZeRO logical-tensor reductions, distributed overflow decisions, paging, CUDA graphs, and compiled optimizers. Gradient shards and graphs are already explicitly unsupported. The audit did not test additional GPUs/backends.
- Stream lifetime under aggressive allocator reuse. Current waits and `record_stream` calls are consistent with native kernels using the default stream. Stress ordinary scaling, masters, strict quantization, and strided copies against a serialized reference before altering those dependencies.
- Live dtype/device changes after state initialization. A CPU BF16 → FP32 → BF16 probe reused a stale master and produced `.80078125` instead of approximately `.69921875`. Same-dtype resume passed; live transitions are not an established supported workflow. Specify or reject them.
- Overlapping parameter/gradient storage. A CPU `p.grad=p.detach()` probe changes the supposedly preserved gradient and makes strict momentum consume modified storage. Ordinary autograd normally provides separate storage. Explicitly reject overlap or define copy semantics if supporting this custom use case; object-identity checks alone do not establish storage independence.
- Long-run convergence after strict fallback, large-model OOM thresholds, native BF16 throughput, and the frequency of tiny values in target applications. The measured allocations imply a scaling risk; no new application OOM or convergence failure was demonstrated.

## Further performance and validation work

These are opportunities or coverage limitations, not additional proven numerical failures:

1. **Keep the fused guard for eligible tensors.** One noncontiguous gradient, or absent Triton, sends all tensors through eager absolute-value/mask/reduction work. CPU instrumentation counted 3,076 guard dispatches for 256 FP32 tensors with FP32 state, excluding probe setup; those are not CUDA timing measurements. Partition layouts or add a stride-aware guard. Directly compare AdaGC device-limit flags with an independent reference across dtypes, scales, and uint8 block states.
2. **Reduce host and metadata overhead after fixing correctness.** The implementation rebuilds lists/maps, uses several FP64 metadata temporaries, copies five columns, converts them to Python lists, and loops over tensors. Only the most recent active-index pattern is cached; alternating `grad=None` patterns recreate CPU-origin device indices. Profile dynamic active sets before choosing a bounded cache, reusable index buffer, or fused preparation/commit kernel.
3. **Optimize strict scratch by blocks/chunks.** Avoid materializing int64 codes and repeated full-length block scales. Tie this work to F06 so the memory optimization preserves independent block ranges. Measure both peak scratch and numerical sign parity; a persistent dequantized shadow would change the package's memory tradeoff.
4. **Do not remove stream bridges without an alternative.** Moving the entire safe update phase to one backend stream may reduce per-tensor events, but requires explicit input readiness, output ordering, and allocator-lifetime tests. A device-only scale ABI or custom fused optimizer could avoid host decisions, but this is a larger backend change, not a flag switch.
5. **Measure actual AMP training and conditional costs.** Add zero/tiny/recovery, variable-active, strided, and nondefault-stream workloads; report clipping/strict counts, initialization, spike/nonspike distributions, persistent storage, and temporary peaks. Distinguish `optimizer.step` from scaler and scheduler time. Balance run order and repeat paired trials before throughput claims.
6. **Close CI's declared-version gap.** Hosted CI covers Python 3.11/3.12 with current CPU PyTorch; GPU CI is manual. Python 3.10 and the torch 2.5 floor are not enforced. The `_foreach_norm(dtype=...)` schema does exist at torch 2.5, so no specific incompatibility was found for that API. Add a minimum-version job or narrow the supported range on evidence.

Reanalysis of existing benchmark samples found only four injected spikes in 100 timed updates. Their cost can fall above p95 and disappear from summary metrics. For one stored FP16/global case, spike median was **1.269 ms**, overall median **1.063 ms**, and p95 **1.081 ms**. This does not invalidate the reported overall median; it limits what it says about spike handling. Details are in `perf/benchmark_reanalysis.json`.

The existing application study's inconclusive three-seed quality result is accurately disclosed. This audit did not rerun that study or reinterpret it as a failure. Its instrumented, synchronized loop is not a universal production-throughput measurement, and its ordinary-path memory figures do not bound F01/F02.

## Suggested implementation sequence

1. Add focused regressions from F03–F05, F07, and F08; fix override ownership/resolution, state migration ordering, and early norm rejection. Keep normal AMP behavior and existing checkpoint compatibility intact.
2. Add zero/tiny per-tensor path and memory checks; implement F01/F02 with an explicit per-tensor recovery policy. Do not trade away subnormal correctness for benchmark numbers.
3. Add the independent heterogeneous-block oracle and repair F06 with bounded scratch. Validate nondefault streams before changing quantization internals.
4. Resolve the hook contract and minimum-threshold boundary. Decide documented handling of storage aliases and live dtype changes.
5. Rerun the original suite and independent AMP/resume controls, then the targeted branch/performance matrix. Only afterward repeat longer matched application comparisons if update numerics changed. Record changed signs and quality intervals rather than asserting universal equivalence.

Each fix should identify the exact failed case it closes, retain its raw before/after evidence, and report any changed numerical policy. Full commands and original results are in the accompanying [reproduction guide](../evidence/audit-2026-11-11/README.md).
