"""CPU checks for balanced campaigns; these never qualify GPU performance."""

import importlib.util
from pathlib import Path
import sys

import pytest


def runner():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "tools"))
    spec = importlib.util.spec_from_file_location(
        "benchmark_integrations", root / "tools/benchmark_integrations.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_campaign_rotates_condition_order_and_pairs_revisions():
    rows = runner().build_schedule(
        [1, 2, 4],
        ["contiguous", "channels-last"],
        ["strong"],
        3,
        candidate_root=Path("/candidate"),
        baseline_root=Path("/baseline"),
    )
    assert len(rows) == 27
    for repetition in range(3):
        grouped = [row for row in rows if row["repetition"] == repetition]
        conditions = {(row["world_size"], row["layout"], row["scaling"]) for row in grouped}
        assert len(conditions) == 6
        for condition in conditions:
            pair = [
                row
                for row in grouped
                if (row["world_size"], row["layout"], row["scaling"]) == condition
            ]
            expected = {"baseline", "candidate"} if condition[1] == "contiguous" else {"candidate"}
            assert {row["revision"] for row in pair} == expected
    assert rows[0]["world_size"] != rows[9]["world_size"] or rows[0]["layout"] != rows[9]["layout"]


def test_paired_slowdown_bound_requires_qualified_sample_counts():
    module = runner()
    accepted = module.paired_slowdown_verdict(
        [1.0, 1.0, 1.0], [1.05, 1.05, 1.05], warmup=100, steps=300
    )
    assert accepted["status"] == "passed"
    assert accepted["upper_95_slowdown"] == pytest.approx(0.05)
    short = module.paired_slowdown_verdict([1.0, 1.0, 1.0], [1.05, 1.05, 1.05], warmup=1, steps=3)
    assert short["status"] == "inconclusive"
    noisy = module.paired_slowdown_verdict([1.0, 1.0, 1.0], [0.8, 1.1, 1.5], warmup=100, steps=300)
    assert noisy["status"] == "inconclusive"
    assert noisy["upper_95_slowdown"] > 0.1


def test_dry_run_is_planned_and_refuses_existing_output(tmp_path):
    import json
    import subprocess

    root = Path(__file__).resolve().parents[1]
    destination = tmp_path / "campaign"
    command = [
        sys.executable,
        str(root / "tools/benchmark_integrations.py"),
        "--dry-run",
        "--output",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    report = json.loads((destination / "manifest.json").read_text())
    assert report["status"] == "planned"
    assert len(report["schedule"]) == 18
    assert not (destination / "result.json").exists()
    assert subprocess.run(command, capture_output=True, text=True).returncode != 0


def test_missing_baseline_source_cannot_fall_back_to_installed_candidate(tmp_path):
    import subprocess

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "tools/benchmark_integrations.py"),
            "--dry-run",
            "--output",
            str(tmp_path / "campaign"),
            "--baseline-root",
            str(tmp_path / "missing"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "lionplus8bit" in result.stderr


def test_new_layout_costs_have_no_unchanged_baseline_comparison():
    rows = runner().build_schedule(
        [1, 2, 4],
        ["contiguous", "channels-last"],
        ["strong", "weak"],
        3,
        candidate_root=Path("/candidate"),
        baseline_root=Path("/baseline"),
    )
    assert len(rows) == 54
    assert all(row["layout"] == "contiguous" for row in rows if row["revision"] == "baseline")
    assert len([row for row in rows if row["layout"] == "channels-last"]) == 18
