# Training integration evidence

Read the [validation report](../../validation-integrations.md) for exact support boundaries, measurements and unresolved questions. [`integration-evidence.zip`](integration-evidence.zip) contains raw results and source snapshots; verify its checksum with [`SHA256SUMS`](SHA256SUMS). The archive's `MANIFEST.sha256` verifies every included file. Earlier evidence archives are preserved separately.

## Contents

- `source/`: implementation, tests, tools, examples, workflows, package metadata, README, report and plan/ledger. `SOURCE_BASE` identifies the baseline. Source hashes let results be checked independently of remote branch history.
- `baseline-source-0fb6a57.tar.gz`: frozen pre-integration source. No baseline source is modified for benchmark compatibility.
- `raw/baseline-*`: original CPU/CUDA runs, including the first CPU metadata-path failure and the corrected environment run.
- `raw/final-expanded-cuda-tests.log`, `raw/final-expanded-wheel-cpu.log`: final 401/334-test suites and 15 subtests each. `raw/release-wheel-identity.json` records the installed import path and matching implementation hashes.
- `raw/final-independent-cuda*`: six original independent CUDA groups and their fresh-process controls.
- `raw/ddp2-aligned`, `raw/ddp4-aligned`, `raw/ddp4-channels-last-aligned`: 204-attempt numerical/replay cases, independent global-batch oracles and unsafe-driver negative cases.
- `raw/ddp-unused2`, `raw/ddp-unused4`, `raw/ddp-exhausted2`, `raw/ddp-exhausted4`: unused/late/dormant parameter and actual iterator-exhaustion checks.
- `raw/layouts`: layout regressions, stream/lifetime checks, raw repeated timing/memory results and measurement producer.
- `raw/paging`, `raw/release-paging-cuda`: pinned Python allocator code, symbol inspection, ordinary-storage controls and explicit ownership refusal. These are not managed-allocation qualification.
- `raw/runtime`: CPU/CUDA compiler, trainer/loader failure, physical remap, transposed restart and fresh native compiler-cache recovery. `raw/restart-channels-last4` adds four-rank channels-last AMP recovery with native uint8 momentum and preserved strides.
- `raw/compiled-ddp*-fp32`: short actual Inductor FP32 DDP smoke results.
- `raw/fp8-required`: real SM75 refusal, including nonzero required-FP8 outcome. No native positive FP8 result exists.
- `raw/benchmarks-final`: 54 successful fresh processes, exact command schedule, per-rank/phase raw results and paired bounds. Five bounds pass, one remains inconclusive. `raw/benchmarks` retains the earlier incompatible baseline-layout attempt.
- `raw/contiguous_replay.py`, `raw/contiguous-*.pt`: three-seed, 200-step AdaGC CPU replay against frozen baseline, including strict arithmetic and warmup.
- `raw/final-review.md`, `raw/final-review-resolution.md`: independent review and checked resolutions. Initial red tests, interrupted transport/replay debugging, FP16 Inductor numerical failure and installed-wheel test-harness failure are retained alongside final verdicts.
- `raw/publish-*`, `raw/release-*`: build/metadata/identity/smoke logs and final capability manifests. Compiler caches, bytecode caches and wheels themselves are omitted from the portable archive; raw producer checkpoints and synthetic replay tensors are included.

An intermediate output is not a final pass merely because it exists. Use the named final logs and report; failed/unsupported/inconclusive outcomes remain separate. Per-run manifests identify the source used in that run. Harness-only additions after performance runs do not retroactively change those manifests.

## Reproduce

Install a CUDA PyTorch suited to the host, then `python -m pip install '.[dev]'`. See the [report's commands](../../validation-integrations.md#reproduce) for DDP, layouts, compile, paging, FP8, restart and scaling. All result directories must be fresh. `--require-cuda`, `--require-gpus 4` and required-FP8 gates refuse silent fallback.

The local dependency override used for source runs was:

```bash
export PYTHONPATH=/tmp/lionplus-adagc-dev:/abyss/home/ml/lion_8bit/verification_20261111/deps:src
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

Distributed runs used `NCCL_P2P_DISABLE=1`; some also set `NCCL_IB_DISABLE=1` on this single-node host. Confirm transport settings independently on another topology. Installed-wheel CPU runs instead started in `/tmp`, hid CUDA and put the isolated wheel target ahead of dependencies without `src` on `PYTHONPATH`. Optional torchdata 0.11.0 was present; those tests skip when it is absent.

The complete local evidence directory is `/abyss/home/ml/lion_8bit/integrations_20261112`. Extract the archive into a new directory before rerunning producers; runners refuse overwrites. The source snapshot, baseline archive and per-run hashes provide reproducibility without access to temporary directories on this node. Hardware-sensitive timings and reduction differences need fresh measurements on another host.
