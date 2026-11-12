"""Capability-gated integration entry point; outputs never overwrite evidence."""

import argparse
from pathlib import Path
from _integration_common import capabilities, output_directory, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", choices=["capabilities", "layouts", "paging", "compile", "fp8"], required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-fp8", action="store_true")
    parser.add_argument("--device", type=int)
    parser.add_argument("--compile-precision", choices=["fp32", "fp16"], default="fp16")
    args = parser.parse_args()
    out = output_directory(args.output)
    write_json(out / "environment.json", capabilities())
    if args.suite == "capabilities":
        result = dict(status="passed", suite="capabilities", details=capabilities())
    elif args.suite == "compile":
        from _compile_integration import run_compile_suite

        result = run_compile_suite(out, precision=args.compile_precision)
    elif args.suite == "fp8":
        from _fp8_integration import run_fp8_suite

        result = run_fp8_suite(out, require_fp8=args.require_fp8)
    elif args.suite == "paging":
        from _paging_integration import run_paging_suite

        result = run_paging_suite(out, device=args.device)
    else:
        import os
        import subprocess
        import sys

        env = dict(os.environ)
        if args.device is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(args.device)
            env["LIONPLUS_LAYOUT_DEVICE"] = "cuda"
        with (out / "tests.log").open("w") as log:
            test = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/test_weight_layouts.py", "-q"],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        result = dict(suite="layouts", status="passed" if test.returncode == 0 else "failed")
    write_json(out / "result.json", result)
    print(result)
    if result.get("exit_code", 0):
        raise SystemExit(result["exit_code"])
    if result.get("status") == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
