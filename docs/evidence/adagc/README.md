# AdaGC validation evidence

These files record the local validation described in [the report](../../validation-adagc.md).
They contain measurements and commands, not application audio or trained checkpoints.

- `full-suite.log`, `cpu-suite-final.log`, and `integration-final.log`: final test output.
- `integration-final/`: independent CUDA gate summaries, including fresh-process resume.
- `bench-*.json`: every measured timing sample, memory peaks and profiler event counts for the replay matrix. Large Chrome traces remain in the original local archive.
- `aggregate.json` and `<strategy>-<bits>-seed<seed>/result.json`: paired application results.
- `<strategy>-<bits>-seed<seed>/raw_updates.json.gz`: loss, timing, norm and clipping telemetry for each of the 1,100 updates. These are the corrected timing runs.
- `sign_replay.json`: matched clipped-gradient sign diagnostics against FP32 momentum.
- `study_config.json`, `model_source_hashes.json`, and `state_inventory.json`: frozen input configuration/hashes and verified momentum allocation.
- `SOURCE_COMMIT`: optimizer source revision used for validation. Later documentation commits do not change those tested sources.
- `SHA256SUMS`: checksums for this evidence directory, excluding the checksum file itself.

From this directory, verify checksums with `sha256sum -c SHA256SUMS`. From the repository root, inspect compressed update records with:

```bash
python - <<'PY'
import gzip
import json

with gzip.open("docs/evidence/adagc/adagc-8-seed17/raw_updates.json.gz", "rt") as stream:
    updates = json.load(stream)
print("Updates:", len(updates))
print("First update:", updates[0])
PY
```

To regenerate synthetic measurements from the repository root after installing `.[dev]`:

```bash
mkdir -p results/adagc
python docs/evidence/adagc/run_benchmarks.py
python docs/evidence/adagc/replay_signs.py
```

Use fresh output paths: the tools refuse overwrite. The benchmark script generates 14 CUDA cases and profiler traces; the sign replay writes `results/adagc/sign_replay.json`. Native CUDA is required. Timing varies with the machine and runtime.

The application campaign needs external SHARK model sources and the prepared MeerKAT tensor cache. Original absolute paths in the frozen configuration identify the measurement environment; they are not installation instructions. To reproduce it, obtain those inputs, verify their hashes, place the model module on `PYTHONPATH`, and copy the configuration to `results/adagc/study_config.json`, updating paths for your machine. Then run:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python docs/evidence/adagc/run_campaign.py
```

The study used a subset and a custom objective, so it cannot establish quality on the original full training task. Earlier throughput measurements had a timing-accounting error; the published application records here come from the corrected rerun. The original local archive retains the superseded runs, profiler traces, binary checkpoints and failure logs for audit.
