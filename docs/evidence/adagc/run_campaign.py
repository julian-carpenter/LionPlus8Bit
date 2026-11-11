import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

base = Path("results/adagc")
for seed in [17, 29, 43]:
    for strategy, bits in [("global", 8), ("adagc", 8), ("adagc", 32)]:
        name = f"{strategy}-{bits}-seed{seed}"
        with (base / f"{name}.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "tools/study_clipping.py",
                    "--config",
                    str(base / "study_config.json"),
                    "--strategy",
                    strategy,
                    "--momentum-bits",
                    str(bits),
                    "--seed",
                    str(seed),
                    "--output",
                    str(base / name),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        print(name, "completed", flush=True)
rows = [
    json.loads((base / f"{strategy}-{bits}-seed{seed}" / "result.json").read_text())
    for seed in [17, 29, 43]
    for strategy, bits in [("global", 8), ("adagc", 8), ("adagc", 32)]
]
deltas = []
for seed in [17, 29, 43]:
    rs = [r for r in rows if r["seed"] == seed]
    assert len({r["initial_sha256"] for r in rs}) == len({r["order_sha256"] for r in rs}) == 1
    a, b, c = rs
    deltas.append((b["validation_bce"] - a["validation_bce"]) / a["validation_bce"])
mean = statistics.mean(deltas)
half = 4.3026527299 * statistics.stdev(deltas) / math.sqrt(3)
result = dict(
    results=rows,
    paired_relative_bce=deltas,
    mean=mean,
    ci95=[mean - half, mean + half],
    quality_margin=0.02,
    quality_status="pass"
    if mean + half <= 0.02
    else "fail"
    if mean - half > 0.02
    else "inconclusive",
    production_status="unresolved: substitute subset task",
)
(base / "aggregate.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "results"}), flush=True)
