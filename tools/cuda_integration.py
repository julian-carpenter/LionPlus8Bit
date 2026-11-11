"""Independent CUDA gates. All numerical tolerances are fixed before execution."""

import copy
import json
import math
import os
import random
import subprocess
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
import bitsandbytes as bnb
from lionplus8bit import LionPlus8bit

OUT = Path(os.environ["LION_RESULTS"])
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
assert torch.cuda.is_available(), "CPU fallback forbidden"
torch.cuda.set_device(0)


def cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().clone()
    if isinstance(x, dict):
        return {k: cpu(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return type(x)(cpu(v) for v in x)
    return x


def exact(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            exact(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            exact(x, y)
    else:
        assert a == b, (a, b)


def note(name, data):
    print(name, json.dumps(data), flush=True)
    (OUT / (name + ".json")).write_text(json.dumps(data, indent=2) + "\n")


def worker(mode, kind, path):
    path = Path(path)
    random.seed(137)
    np.random.seed(137)
    torch.manual_seed(137)
    torch.cuda.manual_seed_all(137)
    torch.use_deterministic_algorithms(True)
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 64), torch.nn.Dropout(0.2), torch.nn.Linear(64, 8)
    ).cuda()
    cfg = dict(
        lr=0.003, betas=(0.7, 0.91), weight_decay=0.02, max_grad_norm=0.7, min_8bit_size=4096
    )
    opt = LionPlus8bit(model.parameters(), **cfg)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.9)
    scaler = torch.amp.GradScaler("cuda", init_scale=32, growth_interval=2)
    start = 0
    losses = []
    if mode == "resume":
        ck = torch.load(path / "split.pt", weights_only=False)
        assert ck["config"] == cfg and ck["manager_overrides"] == {}
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        random.setstate(ck["rng"]["python"])
        np.random.set_state(ck["rng"]["numpy"])
        torch.set_rng_state(ck["rng"]["torch"])
        torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        start = ck["data_position"]
        losses = ck["losses"]
    for t in range(start, 5 if mode == "split" else 12):
        opt.zero_grad(set_to_none=True)
        if kind == "replay":
            for p in model.parameters():
                p.grad = torch.randn_like(p) * (0.1 if t % 3 else 10)
            # Exercise scaler serialization even when no scaler is applicable to supplied gradients.
            opt.step()
            losses.append(None)
        else:
            x = torch.randn(8, 128, device="cuda") * float(np.random.uniform(0.8, 1.2))
            y = torch.randn(8, 8, device="cuda") * random.uniform(0.8, 1.2)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = (model(x) - y).square().mean()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        sched.step()
    record = dict(
        model=cpu(model.state_dict()),
        optimizer=cpu(opt.state_dict()),
        scheduler=sched.state_dict(),
        scaler=scaler.state_dict(),
        rng=dict(
            python=random.getstate(),
            numpy=np.random.get_state(),
            torch=torch.get_rng_state(),
            cuda=torch.cuda.get_rng_state_all(),
        ),
        data_position=5 if mode == "split" else 12,
        config=cfg,
        manager_overrides={},
        losses=losses,
    )
    torch.save(record, path / (mode + ".pt"))
    print(
        json.dumps(
            dict(
                pid=os.getpid(),
                mode=mode,
                kind=kind,
                position=record["data_position"],
                quantized=sum(
                    s.get("state1", torch.empty(0)).dtype == torch.uint8 for s in opt.state.values()
                ),
            )
        ),
        flush=True,
    )


class Gates(unittest.TestCase):
    def test_01_fp64_equations_global_groups(self):
        ps = [
            torch.nn.Parameter(torch.tensor(v, device="cuda"))
            for v in [[0.2, -0.4, 0.7], [0.1, -0.8], [0.6]]
        ]
        groups = [
            dict(params=[p], lr=lr, weight_decay=wd, betas=(0.6, 0.8))
            for p, lr, wd in zip(ps, [0.03, 0.07, 0.02], [0.1, 0, 0.2])
        ]
        opt = LionPlus8bit(groups, max_grad_norm=2.5, min_8bit_size=4)
        xs = [p.detach().cpu().double() for p in ps]
        ms = [torch.zeros_like(x) for x in xs]
        errors = []
        margin = math.inf
        seq = [
            [[3, 1, -2], [4, -1], None],
            [[-20, 5, 8], [0, 0], None],
            [[2, -3, -4], [-2, 1], [1]],
            [[0, 0, 0], None, [-2]],
        ]
        for raw in seq:
            flat = [float(v) for row in raw if row is not None for v in row]
            norm = math.sqrt(sum(v * v for v in flat))
            scale = min(1.0, 2.5 / norm) if norm else 1.0
            refs = []
            for i, (p, row) in enumerate(zip(ps, raw)):
                p.grad = (
                    None if row is None else torch.tensor(row, device="cuda", dtype=torch.float32)
                )
                refs.append(p.grad)
                if row is None:
                    continue
                g = torch.tensor(row, dtype=torch.float64) * scale
                c = 0.6 * ms[i] + 0.4 * g
                nz = c.abs()[c != 0]
                margin = min(margin, nz.min().item() if nz.numel() else math.inf)
                xs[i] = (1 - groups[i]["lr"] * groups[i]["weight_decay"]) * xs[i] - groups[i][
                    "lr"
                ] * c.sign()
                ms[i] = 0.8 * ms[i] + 0.2 * g
            opt.step()
            for i, p in enumerate(ps):
                self.assertIs(p.grad, refs[i])
                torch.testing.assert_close(p.cpu().double(), xs[i], atol=2e-7, rtol=2e-6)
                if p in opt.state:
                    torch.testing.assert_close(
                        opt.state[p]["state1"].cpu().double(), ms[i], atol=2e-7, rtol=2e-6
                    )
                    errors.append(
                        (opt.state[p]["state1"].cpu().double() - ms[i]).abs().max().item()
                    )
        note(
            "fp64_equations",
            dict(
                max_momentum_abs_error=max(errors),
                minimum_nonzero_sign_margin=margin,
                atol=2e-7,
                rtol=2e-6,
                reason="FP32 rounding over four updates; sign mixtures deliberately away from zero",
            ),
        )

    def test_02_allocation_native_replay_and_nonfinite(self):
        inventory = []
        for n in [255, 256, 257, 4095, 4096, 4097, 8192, 10003, 1048577]:
            for threshold in [0, 4096] if n < 4096 else [4096]:
                p = torch.nn.Parameter(torch.linspace(-0.5, 0.5, n, device="cuda"))
                q = torch.nn.Parameter(torch.ones(7, device="cuda"))
                rp = torch.nn.Parameter(p.detach().clone())
                rq = torch.nn.Parameter(q.detach().clone())

                def groups(a, b):
                    return [
                        dict(params=[a], lr=0.01, weight_decay=0.1),
                        dict(params=[b], lr=0.02, weight_decay=0),
                    ]

                a = LionPlus8bit(
                    groups(p, q),
                    betas=(0.7, 0.91),
                    max_grad_norm=0.4,
                    min_8bit_size=threshold,
                    native_gradient_scale=False,
                    batched_norm=False,
                )
                b = bnb.optim.Lion8bit(groups(rp, rq), betas=(0.7, 0.91), min_8bit_size=threshold)
                for t in range(6):
                    gs = [
                        torch.sin(torch.arange(n, device="cuda") * 0.13 + t)
                        * ([1, 100, 0, 0.1, 2, 1][t]),
                        torch.full_like(q, float(t)),
                    ]
                    norms = [torch.linalg.vector_norm(g.float()) for g in gs]
                    norm = torch.linalg.vector_norm(torch.stack(norms))
                    coeff = (
                        torch.ones_like(norm) if norm.item() <= 0.4 else norm.new_tensor(0.4) / norm
                    )
                    originals = []
                    for x, y, g in zip([p, q], [rp, rq], gs):
                        x.grad = g
                        originals.append(g)
                        y.grad = g * coeff
                    a.step()
                    b.step()
                    for x, y, g in zip([p, q], [rp, rq], originals):
                        exact(cpu(x), cpu(y))
                        exact(cpu(a.state[x]), cpu(b.state[y]))
                        self.assertIs(x.grad, g)
                        exact(cpu(x.grad), cpu(g))
                state = a.state[p]
                quant = n >= threshold
                self.assertEqual(state["state1"].dtype, torch.uint8 if quant else torch.float32)
                if quant:
                    self.assertEqual(state["absmax1"].numel(), math.ceil(n / 256))
                    self.assertEqual(state["qmap1"].dtype, torch.float32)
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        self.assertEqual(value.device.type, "cuda")
                before = (cpu([p, q]), cpu(a.state_dict()))
                for bad in [float("nan"), float("inf"), -float("inf")]:
                    p.grad = torch.ones_like(p)
                    q.grad = torch.full_like(q, bad)
                    with self.assertRaisesRegex(RuntimeError, "non-finite"):
                        a.step()
                    exact(cpu([p, q]), before[0])
                    exact(cpu(a.state_dict()), before[1])
                    self.assertIsNone(a._clip_scale)
                inventory.append(
                    dict(
                        n=n,
                        threshold=threshold,
                        dtype=str(state["state1"].dtype),
                        scale_count=state.get("absmax1", torch.empty(0)).numel(),
                    )
                )
        note(
            "allocation_replay",
            dict(
                cases=inventory,
                updates_per_case=6,
                parameter_and_state_discrepancy=0,
                nonfinite_cases=3,
                comparison="exact",
            ),
        )

    def test_03_direct_low_precision(self):
        results = []
        for dtype in [torch.float16, torch.bfloat16]:
            p = torch.nn.Parameter(torch.ones(8193, device="cuda", dtype=dtype))
            q = torch.nn.Parameter(p.detach().clone())
            a = LionPlus8bit(
                [p], lr=0.01, max_grad_norm=1, betas=(0, 0), weight_decay=0, master_weights=False
            )
            b = bnb.optim.Lion8bit([q], lr=0.01, betas=(0, 0), weight_decay=0)
            g = torch.full_like(p, 2)
            norm = torch.linalg.vector_norm(g, dtype=torch.float32)
            scaled = (g.float() * (norm.new_tensor(1) / norm)).to(dtype)
            p.grad = g
            q.grad = scaled
            a.step()
            b.step()
            exact(cpu(p), cpu(q))
            exact(cpu(a.state[p]), cpu(b.state[q]))
            self.assertIs(p.grad, g)
            self.assertEqual(p.dtype, dtype)
            self.assertEqual(p.device.type, "cuda")
            results.append(
                dict(
                    dtype=str(dtype),
                    native_parity="exact",
                    native_bf16_hardware=torch.cuda.is_bf16_supported(including_emulation=False),
                )
            )
        note("direct_precision", results)

    def test_04_amp_overflow_scheduler(self):
        results = []
        for dtype in [torch.float16, torch.bfloat16]:
            torch.manual_seed(9)
            m = torch.nn.Linear(128, 64).cuda()
            opt = LionPlus8bit(m.parameters(), lr=0.001)
            sch = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.9)
            s = torch.amp.GradScaler("cuda", init_scale=16, growth_interval=100)
            calls = []
            opt.register_step_post_hook(lambda *args: calls.append(1))
            steps = []
            for t in range(3):
                opt.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=dtype):
                    loss = m(torch.randn(4, 128, device="cuda")).square().mean()
                s.scale(loss).backward()
                if t == 1:
                    next(m.parameters()).grad.view(-1)[0] = float("inf")
                s.unscale_(opt)
                grads = [p.grad for p in m.parameters()]
                values = cpu(grads)
                before = (cpu(m.state_dict()), cpu(opt.state_dict()))
                old = s.get_scale()
                count = len(calls)
                s.step(opt)
                s.update()
                success = len(calls) > count
                if success:
                    sch.step()
                if t == 1:
                    self.assertFalse(success)
                    exact(cpu(m.state_dict()), before[0])
                    exact(cpu(opt.state_dict()), before[1])
                    self.assertLess(s.get_scale(), old)
                else:
                    self.assertTrue(success)
                    for p, g, v in zip(m.parameters(), grads, values):
                        self.assertIs(p.grad, g)
                        exact(cpu(g), v)
                steps.append(
                    dict(success=success, scale=s.get_scale(), scheduler_epoch=sch.last_epoch)
                )
            self.assertEqual(sch.last_epoch, 2)
            self.assertEqual(opt.state[next(m.parameters())]["step"], 2)
            results.append(dict(dtype=str(dtype), steps=steps))
        note("amp", results)

    def test_05_accumulation(self):
        torch.manual_seed(33)
        m = torch.nn.Linear(32, 16, bias=True).cuda()
        n = copy.deepcopy(m)
        a = LionPlus8bit(m.parameters(), lr=0.003, betas=(0.7, 0.91), max_grad_norm=0.3)
        b = LionPlus8bit(n.parameters(), lr=0.003, betas=(0.7, 0.91), max_grad_norm=0.3)
        discrepancies = []
        for t in range(5):
            x = torch.randn(16, 32, device="cuda")
            y = torch.randn(16, 16, device="cuda")
            a.zero_grad()
            b.zero_grad()
            (m(x) - y).square().mean().backward()
            for xx, yy in zip(x.chunk(4), y.chunk(4)):
                ((n(xx) - yy).square().mean() / 4).backward()
            for p, q in zip(m.parameters(), n.parameters()):
                torch.testing.assert_close(p.grad, q.grad, atol=1e-7, rtol=2e-5)
            a.step()
            b.step()
            for p, q in zip(m.parameters(), n.parameters()):
                torch.testing.assert_close(p, q, atol=2e-7, rtol=2e-6)
                torch.testing.assert_close(
                    a.state[p]["state1"], b.state[q]["state1"], atol=1e-7, rtol=2e-5
                )
                discrepancies.append(
                    (a.state[p]["state1"] - b.state[q]["state1"]).abs().max().item()
                )
        note(
            "accumulation",
            dict(
                max_momentum_abs_error=max(discrepancies),
                updates=5,
                microbatches=4,
                batch=16,
                atol=1e-7,
                rtol=2e-5,
                reason="FP32 GEMM and reduction order differ between full and micro batches; unquantized small-state test isolates accumulation",
            ),
        )

    def test_06_fresh_process_checkpoint(self):
        for kind in ["replay", "training"]:
            path = OUT / kind
            path.mkdir(exist_ok=True)
            for mode in ["control", "split", "resume"]:
                with (path / (mode + ".log")).open("w") as log:
                    subprocess.run(
                        [sys.executable, __file__, "worker", mode, kind, str(path)],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            a = torch.load(path / "control.pt", weights_only=False)
            b = torch.load(path / "resume.pt", weights_only=False)
            # NumPy RNG tuple contains an ndarray; compare it separately.
            np.testing.assert_array_equal(a["rng"]["numpy"][1], b["rng"]["numpy"][1])
            a["rng"]["numpy"] = tuple(x for i, x in enumerate(a["rng"]["numpy"]) if i != 1)
            b["rng"]["numpy"] = tuple(x for i, x in enumerate(b["rng"]["numpy"]) if i != 1)
            exact(a, b)
        note(
            "checkpoint",
            dict(
                replay="bitwise equal",
                training="bitwise equal",
                updates=12,
                split_after=5,
                restored=[
                    "model",
                    "optimizer",
                    "scheduler",
                    "scaler",
                    "python_rng",
                    "numpy_rng",
                    "torch_cpu_rng",
                    "cuda_rng",
                    "data_position",
                    "constructor_config",
                    "manager_overrides",
                ],
                sampler="generated indexed stream; no DataLoader workers",
            ),
        )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(*sys.argv[2:])
    else:
        unittest.main(verbosity=2)
