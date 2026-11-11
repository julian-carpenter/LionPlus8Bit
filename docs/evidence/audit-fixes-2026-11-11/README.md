# Corrective evidence for the fresh audit

See [the corrective report](../../validation-audit-fixes.md) for implementation decisions, test outcomes, measured memory effects and remaining support boundaries. The [original audit archive](../audit-2026-11-11/README.md) is preserved separately.

[`corrective-evidence.zip`](corrective-evidence.zip) contains the raw corrective evidence; verify it with [`SHA256SUMS`](SHA256SUMS). `MANIFEST.sha256` inside checks every included file.

- `source/`: final Python implementation, tests, tools, workflows, package metadata, README and corrective report, plus source hashes. This captures the working tree relative to base `6420543` without relying on an unpublished remote branch.
- `full-suite-final.log`: 172 CUDA-required tests and 15 subtests passed.
- `wheel-cpu-final.log`, `build.log`, `twine.log`: installed-wheel CPU tests and packaging verification.
- `integration/`, `integration.log`: six independent CUDA groups and their separate-process controlled resumes.
- `probe_*`, `fallback-profile.json`, `*.trace.json`: repeated original probes, before/after-compatible measurements, and CUDA traces. Original before-results remain in the original audit archive.
- `state/`: six CPU AMP resume splits, twelve precision controls, state implementation notes and regression logs; synthetic producer checkpoints are included.
- `final_review.md`: independent final review and CPU boundary/overflow control results.
- `study/`, `campaign.log`, `run_campaign.py`: all nine 1,100-update application arms, per-update results, balanced arm order, paired interval and comparison with the original study. Trained application checkpoints and batch-order tensor files are omitted from this portable archive; original raw files remain in the workspace.
- `benchmarks/`, `benchmarks.log`: 72 independent process runs, rotated condition/arm order, two matched repetitions, every raw timing/path/phase sample, aggregate dispersion and exact command manifests.

Initial red regressions and intermediate integration logs are retained. Early full-suite attempts overlapped the benchmark's test-first development and consequently recorded temporarily missing benchmark APIs. The files named `*-final.log` are the final verdicts.

## Reproduce

Install a compatible PyTorch for the hardware, then from the source snapshot or checkout:

```bash
python -m pip install '.[dev]'
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python -m pytest --require-cuda -q
CUDA_VISIBLE_DEVICES='' python -m pytest -m 'not cuda' -q
LION_RESULTS=results/corrective-integration python tools/cuda_integration.py
python -m build
python -m twine check --strict dist/*
python -m ruff check .
python -m ruff format --check .

python tools/run_benchmark_campaign.py \
  --output-dir results/corrective-benchmarks \
  --cases ordinary zero tiny recovery strided variable-active \
  --modes replay amp --strategies global adagc native \
  --repeats 2 --warmup 100 --steps 25
```

Use a new output directory. Benchmark campaigns refuse overwrites and detect source changes. `--quick` gives a six-run smoke matrix; increasing repetitions and timed steps is appropriate for estimating performance rather than exercising paths. The native backend baseline is unclipped Lion8bit, so its update semantics differ from Lion+.

The exact local dependency override was:

```bash
export PYTHONPATH=/tmp/lionplus-adagc-dev:/abyss/home/ml/lion_8bit/verification_20261111/deps:/abyss/home/ml/lion_8bit/LionPlus8Bit/src
```

For the study, append `/abyss/home/ml/lion_8bit/verification_20261111/application_sources` to that import path. The captured `run_campaign.py` runs from the repository root and reads the original `docs/evidence/adagc/study_config.json`. Cached application tensors/model dependencies and their hashes are recorded in that configuration; they are not included here. On another host, provide those dependencies or use the synthetic tests and benchmarks, which need no application data. The standalone CUDA probes write beside their scripts; extract a copy to a fresh directory before rerunning them.

Local runtime: Quadro RTX 6000 SM75, Python 3.12.3, NVIDIA PyTorch 2.13 alpha/CUDA 13.3, bitsandbytes 0.50.2 selecting its CUDA 13.2 binary. The Python 3.10/PyTorch 2.5 minimum CI job is configured but was not executed in this local runtime. No native BF16 throughput is claimed.
