# Evidence and reproduction: fresh 2026-11-11 audit

Read the [consolidated report](../../audits/2026-11-11-fresh-audit.md) for findings, priorities, limits, and acceptance criteria. The audited source revision is `64205431d701415de6aaa8355bcc536f81626cd5`. No implementation fixes are included.

[`audit-evidence.zip`](audit-evidence.zip) preserves the original probes and outputs. [`SHA256SUMS`](SHA256SUMS) checks the archive. Inside the archive, `MANIFEST.sha256` checks individual files. The evidence is self-contained except for Python/PyTorch/bitsandbytes/Triton and suitable hardware. No application audio or private dataset is required.

| Archive path | Contents |
| --- | --- |
| `environment.json` | Runtime versions, GPU, driver, backend hashes, audited revision |
| `tracked_source_sha256.json` | Hashes of tracked files at the audited revision |
| `audited-code.tar.gz` | Original package, tests, tools, examples, workflows, packaging, README and license from that revision |
| `root/baseline-tests.log` | Fresh complete suite: 79 passed, 15 subtests, 16 warnings |
| `root/fallback-profile.json`, eight `root/*.trace.json` | Raw timing samples, allocation peaks, selected profiler events, complete Chrome traces |
| `root/probe_real_amp.*` | Real FP16 autocast + GradScaler + uint8 momentum, zero/tiny recovery costs, overflow state checks |
| `root/probe_amp_scale_reference.*` | CUDA scaled versus unscaled control with quantized momentum |
| `root/probe_cuda_resume.*` | Default/deferred CPU-checkpoint-to-CUDA migration results and tracebacks |
| `root/cross_block_cuda.*` | CUDA heterogeneous-block momentum/sign failure |
| `amp/` | Independent CPU AMP, norm overflow, minimum-threshold, half-storage, accumulation and zero/history probes |
| `state/` | Override failures, dynamic module precision, hooks, separate-process resumes, precision matrix, synthetic checkpoints |
| `perf/` | CPU control-flow/dispatch probes, independent blockwise oracle, old benchmark reanalysis, source review |

The independent review notes in `amp/report.md`, `state/report.txt`, and `perf/report.txt` provide additional source details. Their GPU statements are explicitly limited to results communicated by the primary reviewer; the actual CUDA logs are under `root/`.

## Preserve the captured evidence

From the repository root:

```bash
(cd docs/evidence/audit-2026-11-11 && sha256sum -c SHA256SUMS)
```

Extract a copy elsewhere to inspect all raw files, then run `sha256sum -c MANIFEST.sha256` from the extraction directory. Do not run probes in the captured evidence directory: scripts write their result files beside themselves. Some failure probes intentionally catch exceptions and record them, so inspect result fields rather than treating exit zero as a verdict.

## Prepare a separate reproduction directory

Use the audited commit or unpack `audited-code.tar.gz` into a separate checkout. Install a hardware-compatible PyTorch first, then install the package/development dependencies:

```bash
python -m pip install '.[dev]'
```

For the exact local runtime, the original audit used:

```bash
export PYTHONPATH=/tmp/lionplus-adagc-dev:/abyss/home/ml/lion_8bit/verification_20261111/deps:/abyss/home/ml/lion_8bit/LionPlus8Bit/src
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Those dependency directories are machine-specific. On another host, use the installed package and record actual versions; the NVIDIA PyTorch prerelease and timings are not promised to be reproducible with an arbitrary stable build. The original dependency hashes and selected CUDA binary are recorded in `environment.json`.

The following copies only reproduction scripts into a new directory. Four state probes originally used an absolute output directory; the copy replaces only that `pathlib.Path(...)` expression with the script's own directory. The captured scripts and results remain unchanged.

```bash
export LION_AUDIT_RUN="$(mktemp -d /tmp/lionplus-audit.XXXXXX)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONDONTWRITEBYTECODE=1

python - "$LION_AUDIT_RUN" <<'PY'
from pathlib import Path
import sys
import zipfile

target = Path(sys.argv[1])
archive = Path("docs/evidence/audit-2026-11-11/audit-evidence.zip")
old = "pathlib.Path('/abyss/home/ml/lion_8bit/audit_20261111_fresh/state')"
new = "pathlib.Path(__file__).resolve().parent"
with zipfile.ZipFile(archive) as bundle:
    for entry in bundle.infolist():
        if entry.filename.endswith(".py"):
            destination = target / entry.filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.read(entry).decode()
            destination.write_text(source.replace(old, new))
print(target)
PY
```

## Commands

Run the baseline from the source checkout. CUDA is required; the flag prevents a CPU-only green result:

```bash
python -m pytest --require-cuda -q
```

Run CUDA probes sequentially, without other GPU workloads:

```bash
python "$LION_AUDIT_RUN/root/profile_fallbacks.py"
python "$LION_AUDIT_RUN/root/probe_real_amp.py"
python "$LION_AUDIT_RUN/root/probe_amp_scale_reference.py"
python "$LION_AUDIT_RUN/root/probe_cuda_resume.py"
python "$LION_AUDIT_RUN/root/cross_block_cuda.py"
```

Run CPU probes with CUDA hidden. The state producer must precede its separate-process consumer; `probe_state.py` creates the override checkpoint consumed by `probe_split_override_resume.py`.

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python "$LION_AUDIT_RUN/amp/probe_amp.py"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python "$LION_AUDIT_RUN/state/probe_state.py"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python "$LION_AUDIT_RUN/state/probe_split_override_resume.py"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python "$LION_AUDIT_RUN/state/probe_amp_resume.py" produce
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python "$LION_AUDIT_RUN/state/probe_amp_resume.py" resume
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python "$LION_AUDIT_RUN/state/probe_precision.py"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python "$LION_AUDIT_RUN/perf/cpu_probes.py"
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python "$LION_AUDIT_RUN/perf/cross_block_probe.py"
```

Expected original observations: deferred CUDA loads fail while default loads pass; the cross-block probe intentionally asserts that the old implementation loses a block and changes 256 signs; state probes record the silent LR/alias failures; ordinary AMP/reference and supported resume controls pass. After fixes, convert the defect probes into positive regression assertions rather than retaining assertions that require a bug.

The raw timing JSON includes every sample. The profile also contains explicit harness synchronization, so total trace synchronization counts are not all optimizer overhead. Peak figures are allocated-memory deltas above live initialized state; they are distinct from total VRAM and allocator-reserved memory. Original benchmarks were small single runs and do not provide throughput confidence intervals.
