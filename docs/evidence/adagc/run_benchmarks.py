import subprocess
import sys
from pathlib import Path

base = Path("results/adagc")
for pattern in ["none", "localized", "widespread"]:
    for layout in ["few-large", "many-small"]:
        for strategy in ["global", "adagc"]:
            name = f"bench-{strategy}-{layout}-{pattern}"
            with (base / f"{name}.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "tools/benchmark_clipping.py",
                        "--strategy",
                        strategy,
                        "--layout",
                        layout,
                        "--pattern",
                        pattern,
                        "--steps",
                        "100",
                        "--warmup",
                        "150",
                        "--seed",
                        "0",
                        "--profile",
                        "--output",
                        str(base / f"{name}.json"),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            print(name, "completed", flush=True)
for strategy in ["global", "adagc"]:
    name = f"bench-{strategy}-few-large-float16"
    with (base / f"{name}.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "tools/benchmark_clipping.py",
                "--strategy",
                strategy,
                "--layout",
                "few-large",
                "--dtype",
                "float16",
                "--steps",
                "100",
                "--warmup",
                "150",
                "--seed",
                "0",
                "--profile",
                "--output",
                str(base / f"{name}.json"),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print(name, "completed", flush=True)
