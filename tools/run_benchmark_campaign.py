"""Run matched benchmark arms with rotated order and preserved per-process evidence.

Default: few-large replay, ordinary/zero/recovery, global/AdaGC, three paired
repetitions. --quick reduces this to six short smoke runs. Add AMP or the
unclipped native dispatch control with --modes replay amp and
--strategies global adagc native. Historical evidence drivers are unchanged.
"""

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
CASES = ("ordinary", "zero", "tiny", "recovery", "strided", "variable-active")


def build_jobs(args):
    conditions = list(itertools.product(args.layouts, args.cases, args.modes))
    jobs = []
    for repetition in range(args.repeats):
        offset = repetition % len(conditions)
        for index in list(range(len(conditions)))[offset:] + list(range(len(conditions)))[:offset]:
            layout, case, mode = conditions[index]
            first = (repetition + index) % len(args.strategies)
            order = args.strategies[first:] + args.strategies[:first]
            for strategy in order:
                name = f"r{repetition}-{layout}-{case}-{mode}-{strategy}"
                output = args.output_dir / f"{name}.json"
                command = [
                    sys.executable,
                    str(ROOT / "tools" / "benchmark_clipping.py"),
                    "--strategy",
                    strategy,
                    "--layout",
                    layout,
                    "--case",
                    case,
                    "--mode",
                    mode,
                    "--stream",
                    args.stream,
                    "--pattern",
                    args.pattern,
                    "--steps",
                    str(args.steps),
                    "--warmup",
                    str(args.warmup),
                    "--seed",
                    str(args.seed + repetition),
                    "--output",
                    str(output),
                ]
                if args.profile:
                    command.append("--profile")
                jobs.append(
                    dict(
                        repetition=repetition,
                        layout=layout,
                        case=case,
                        mode=mode,
                        strategy=strategy,
                        seed=args.seed + repetition,
                        output=str(output),
                        log=str(args.output_dir / f"{name}.log"),
                        command=command,
                        status="planned",
                    )
                )
    return jobs


def aggregate(jobs):
    groups = defaultdict(list)
    global_times = {}
    for job in jobs:
        if job["status"] != "completed":
            continue
        condition = (job["layout"], job["case"], job["mode"])
        groups[(*condition, job["strategy"])].append(job)
        if job["strategy"] == "global":
            global_times[(*condition, job["repetition"])] = job["median_wall_ms"]
    rows = []
    for (layout, case, mode, strategy), runs in groups.items():
        medians = [run["median_wall_ms"] for run in runs]
        ratios = [
            run["median_wall_ms"] / global_times[(layout, case, mode, run["repetition"])]
            for run in runs
            if (layout, case, mode, run["repetition"]) in global_times
        ]
        rows.append(
            dict(
                layout=layout,
                case=case,
                mode=mode,
                strategy=strategy,
                run_count=len(runs),
                run_medians_wall_ms=medians,
                mean_median_wall_ms=statistics.mean(medians),
                stdev_median_wall_ms=statistics.stdev(medians) if len(medians) > 1 else None,
                paired_global_ratios=ratios,
                mean_paired_global_ratio=statistics.mean(ratios) if ratios else None,
                stdev_paired_global_ratio=statistics.stdev(ratios) if len(ratios) > 1 else None,
            )
        )
    return rows


def fingerprints():
    names = [
        "tools/benchmark_clipping.py",
        "tools/run_benchmark_campaign.py",
        "src/lionplus8bit/_optimizer.py",
        "src/lionplus8bit/_clipping.py",
        "src/lionplus8bit/_numerical_guard.py",
        "src/lionplus8bit/_strict.py",
    ]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--quick", action="store_true", help="Six smoke runs by default; not a performance study"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Write the concrete run plan without using CUDA"
    )
    parser.add_argument(
        "--cases", nargs="+", choices=CASES, default=["ordinary", "zero", "recovery"]
    )
    parser.add_argument(
        "--layouts", nargs="+", choices=["few-large", "many-small"], default=["few-large"]
    )
    parser.add_argument("--modes", nargs="+", choices=["replay", "amp"], default=["replay"])
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["global", "adagc", "native"],
        default=["global", "adagc"],
    )
    parser.add_argument("--stream", choices=["default", "nondefault"], default="default")
    parser.add_argument(
        "--pattern", choices=["none", "localized", "widespread"], default="localized"
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    args.steps = args.steps if args.steps is not None else (2 if args.quick else 100)
    args.warmup = args.warmup if args.warmup is not None else (100 if args.quick else 150)
    args.repeats = args.repeats if args.repeats is not None else (1 if args.quick else 3)
    if args.steps < 1 or args.repeats < 1 or args.warmup < 100:
        parser.error("steps/repeats must be positive and warmup must be >=100")
    for field in ("cases", "layouts", "modes", "strategies"):
        if len(set(getattr(args, field))) != len(getattr(args, field)):
            parser.error(f"duplicate {field} would overwrite evidence")
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        parser.error("output directory already exists; use a fresh evidence directory")
    jobs = build_jobs(args)
    manifest = dict(
        arguments=vars(args) | {"output_dir": str(args.output_dir)},
        jobs=jobs,
        source_sha256=fingerprints(),
        notes="Rotated arm/condition order; matched seed per repetition. Native is an unclipped dispatch control. Dispersion is across independent processes, not a confidence interval.",
    )
    args.output_dir.mkdir(parents=True)

    def save():
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (args.output_dir / "aggregate.json").write_text(
            json.dumps(aggregate(jobs), indent=2) + "\n"
        )

    save()
    print(f"Planned {len(jobs)} runs in {args.output_dir}", flush=True)
    if args.dry_run:
        return
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    for job in jobs:
        if fingerprints() != manifest["source_sha256"]:
            raise SystemExit("Source changed during campaign; preserved completed results")
        job["status"] = "running"
        save()
        with Path(job["log"]).open("w") as log:
            completed = subprocess.run(
                job["command"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
        job["returncode"] = completed.returncode
        job["status"] = "completed" if completed.returncode == 0 else "failed"
        if completed.returncode == 0:
            result = json.loads(Path(job["output"]).read_text())
            job["median_wall_ms"] = result["median_wall_ms"]
            job["transient_peak_delta"] = result["transient_peak_delta"]
        save()
        print(f"{Path(job['output']).stem}: {job['status']}", flush=True)
        if completed.returncode:
            raise SystemExit(f"Benchmark failed; see {job['log']}")
        if fingerprints() != manifest["source_sha256"]:
            raise SystemExit(
                "Source changed during campaign; do not combine this run with earlier results"
            )


if __name__ == "__main__":
    main()
