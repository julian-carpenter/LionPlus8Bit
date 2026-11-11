import importlib.util
from pathlib import Path
from types import SimpleNamespace


def load_campaign():
    path = Path(__file__).resolve().parents[1] / "tools" / "run_benchmark_campaign.py"
    spec = importlib.util.spec_from_file_location("run_benchmark_campaign", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_campaign_balances_order_and_matches_seeds_within_each_pair(tmp_path):
    campaign = load_campaign()
    args = SimpleNamespace(
        layouts=["few-large"],
        cases=["ordinary", "zero"],
        modes=["replay"],
        strategies=["global", "adagc"],
        repeats=2,
        seed=17,
        steps=3,
        warmup=100,
        stream="default",
        pattern="none",
        output_dir=tmp_path,
        profile=False,
    )
    jobs = campaign.build_jobs(args)
    assert len(jobs) == 8
    for case in args.cases:
        groups = [
            [j for j in jobs if j["case"] == case and j["repetition"] == repeat]
            for repeat in range(2)
        ]
        assert [j["strategy"] for j in groups[0]] == [j["strategy"] for j in groups[1]][::-1]
        assert {j["seed"] for j in groups[0]} == {17}
        assert {j["seed"] for j in groups[1]} == {18}
    assert len({j["output"] for j in jobs}) == 8


def test_campaign_aggregate_preserves_paired_ratios():
    campaign = load_campaign()
    jobs = []
    for repeat, base in enumerate([2.0, 4.0]):
        for strategy, time in [("global", base), ("adagc", base * 1.5)]:
            jobs.append(
                dict(
                    layout="few-large",
                    case="zero",
                    mode="replay",
                    strategy=strategy,
                    repetition=repeat,
                    status="completed",
                    median_wall_ms=time,
                )
            )
    summary = campaign.aggregate(jobs)
    row = next(v for v in summary if v["strategy"] == "adagc")
    assert row["run_count"] == 2
    assert row["paired_global_ratios"] == [1.5, 1.5]
    assert row["mean_paired_global_ratio"] == 1.5
