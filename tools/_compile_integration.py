"""Measure compilation boundaries without counting whole-frame eager fallback."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import time

import torch


class CaptureCounter:
    """Dynamo backend recording actual captured graph modules.

    Executes GraphModule.forward intentionally: this is capture compatibility
    evidence, not an Inductor throughput measurement or native fusion claim.
    """

    def __init__(self, backend="graphmodule"):
        self.backend = backend
        self.graph_count = 0
        self.operation_count = 0
        self.graph_fingerprints = set()

    def __call__(self, graph_module, example_inputs):
        self.graph_count += 1
        self.graph_fingerprints.add(graph_module.code)
        self.operation_count += sum(
            node.op.startswith("call_") for node in graph_module.graph.nodes
        )
        if self.backend == "inductor":
            return torch._inductor.compile(graph_module, example_inputs)
        return graph_module.forward


def compilation_verdict(graph_count, *, parity):
    return "failed" if not parity else "partial-capture" if graph_count else "eager-fallback"


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(_equal(a, b) for a, b in zip(left, right))
    return left == right


def _model_case(device, clipping, backend_name="inductor", amp=None):
    from lionplus8bit import LionPlus8bit

    torch.manual_seed(309)
    model = torch.nn.Sequential(
        torch.nn.Linear(65, 65), torch.nn.Tanh(), torch.nn.Linear(65, 2)
    ).to(device)
    candidate = copy.deepcopy(model)
    replay = copy.deepcopy(model)
    eager = LionPlus8bit(model.parameters(), lr=0.01, weight_decay=0, clipping=clipping)
    tested = LionPlus8bit(candidate.parameters(), lr=0.01, weight_decay=0, clipping=clipping)
    replay_optimizer = LionPlus8bit(replay.parameters(), lr=0.01, weight_decay=0, clipping=clipping)
    backend = CaptureCounter(backend=backend_name)
    compiled = torch.compile(candidate, backend=backend)
    inputs = torch.randn(4, 65, device=device)
    initial = time.perf_counter()
    exact_states = []
    matched_gradient_states = []
    diagnostics = []
    amp = device == "cuda" if amp is None else bool(amp)
    rtol, atol = (2e-3, 2e-5) if amp else (2e-5, 1e-7)
    for step in range(4):
        outputs = []
        gradients = []
        deltas = []
        for module, optimizer in [(model, eager), (compiled, tested)]:
            optimizer.zero_grad(set_to_none=True)
            optimizer.param_groups[0]["lr"] = 0.01 / (step + 1)
            before = [parameter.detach().clone() for parameter in module.parameters()]
            with torch.autocast(
                device, dtype=torch.float16 if amp else torch.bfloat16, enabled=amp
            ):
                output = module(inputs)
                loss = output.square().mean()
            outputs.append(output.detach().float())
            loss.backward()
            gradients.append([parameter.grad.detach().clone() for parameter in module.parameters()])
            optimizer.step()
            deltas.append(
                [old - parameter.detach() for old, parameter in zip(before, module.parameters())]
            )
        replay_optimizer.param_groups[0]["lr"] = tested.param_groups[0]["lr"]
        for parameter, gradient in zip(replay.parameters(), gradients[1]):
            parameter.grad = gradient.clone()
        replay_optimizer.step()
        matched_gradient_states.append(
            _equal(replay.state_dict(), candidate.state_dict())
            and _equal(replay_optimizer.state_dict(), tested.state_dict())
        )
        exact_states.append(
            _equal(model.state_dict(), candidate.state_dict())
            and _equal(eager.state_dict(), tested.state_dict())
        )
        diagnostics.append(
            {
                "step": step,
                "output_max_abs_error": float((outputs[0] - outputs[1]).abs().max()),
                "gradient_max_abs_error": max(
                    float((a - b).abs().max()) for a, b in zip(*gradients)
                ),
                "gradients_within_tolerance": all(
                    torch.allclose(a, b, rtol=rtol, atol=atol) for a, b in zip(*gradients)
                ),
                "update_sign_disagreements": sum(
                    int((a.sign() != b.sign()).sum()) for a, b in zip(*deltas)
                ),
                "minimum_gradient_sign_margin": min(float(a.abs().min()) for a in gradients[0]),
            }
        )
        if step == 1:
            tested.load_state_dict(copy.deepcopy(tested.state_dict()))
            replay_optimizer.load_state_dict(copy.deepcopy(replay_optimizer.state_dict()))
    if device == "cuda":
        torch.cuda.synchronize()
    parity = all(matched_gradient_states) and all(
        row["gradients_within_tolerance"] and row["update_sign_disagreements"] == 0
        for row in diagnostics
    )
    return {
        "name": f"compiled-model-{clipping}",
        "status": "passed" if parity else "failed",
        "graphs": backend.graph_count,
        "operations": backend.operation_count,
        "elapsed_seconds": time.perf_counter() - initial,
        "backend": backend_name,
        "amp": amp,
        "performance_qualified": False,
        "bitwise_model_optimizer_parity": all(exact_states),
        "matched_gradient_optimizer_bitwise_parity": all(matched_gradient_states),
        "gradient_tolerance": {"rtol": rtol, "atol": atol},
        "diagnostics": diagnostics,
        "uint8_state": any(
            state.get("state1", torch.empty(0)).dtype == torch.uint8
            for state in tested.state.values()
        ),
    }


def _optimizer_case(device, clipping="global", channels_last=False):
    from lionplus8bit import LionPlus8bit

    torch._dynamo.reset()
    values = torch.ones((17, 17, 1, 17) if channels_last else (4097,), device=device)
    if channels_last:
        values = values.to(memory_format=torch.channels_last)
    parameter = torch.nn.Parameter(values)
    auxiliary = torch.nn.Parameter(torch.ones(3, device=device))
    control_auxiliary = torch.nn.Parameter(auxiliary.detach().clone())
    control = torch.nn.Parameter(parameter.detach().clone())
    eager = LionPlus8bit([control, control_auxiliary], lr=0.01, weight_decay=0, clipping=clipping)
    optimizer = LionPlus8bit([parameter, auxiliary], lr=0.01, weight_decay=0, clipping=clipping)
    backend = CaptureCounter()
    compiled = torch.compile(optimizer.step, backend=backend, fullgraph=False)
    before = dict(torch._dynamo.utils.counters["frames"])
    timings = []
    parity = True
    checkpoint_roundtrip = False
    for index, scale in enumerate([1.0, 0.0, 1e-30, 8.0]):
        for target, opt in [(parameter, optimizer), (control, eager)]:
            target.grad = torch.full_like(target, scale)
            opt.param_groups[0]["lr"] = 0.01 / (index + 1)
        auxiliary.grad = None if index % 2 else torch.full_like(auxiliary, scale)
        control_auxiliary.grad = None if auxiliary.grad is None else auxiliary.grad.clone()
        begin = time.perf_counter()
        compiled()
        if device == "cuda":
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - begin)
        eager.step()
        parity &= (
            torch.equal(parameter, control)
            and torch.equal(auxiliary, control_auxiliary)
            and _equal(optimizer.state_dict(), eager.state_dict())
        )
        if index == 1:
            optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            eager.load_state_dict(copy.deepcopy(eager.state_dict()))
            checkpoint_roundtrip = True
    graphs = backend.graph_count
    measured_frames = dict(torch._dynamo.utils.counters["frames"])
    guard_failures = {
        f"{code.co_filename}:{code.co_firstlineno}:{code.co_name}": list(reasons)
        for code, reasons in torch._dynamo.utils.guard_failures.items()
        if reasons
    }
    try:
        explanation = torch._dynamo.explain(optimizer.step)()
        explained = {
            "graph_count": explanation.graph_count,
            "graph_break_count": explanation.graph_break_count,
            "break_reasons": [str(reason) for reason in explanation.break_reasons],
        }
    except Exception as error:
        explained = {"error": f"{type(error).__name__}: {error}"}
    try:
        torch.compile(optimizer.step, backend=CaptureCounter(), fullgraph=True)()
        fullgraph = {"status": "captured"}
    except Exception as error:
        fullgraph = {"status": "unsupported", "reason": f"{type(error).__name__}: {error}"[:2000]}
    after = dict(torch._dynamo.utils.counters["frames"])
    return {
        "name": f"optimizer-step-{clipping}" + ("-channels-last" if channels_last else ""),
        "status": compilation_verdict(graphs, parity=parity),
        "clipping": clipping,
        "channels_last": channels_last,
        "checkpoint_roundtrip": checkpoint_roundtrip,
        "changing_active_sets": True,
        "strict_path": bool(optimizer.state[parameter].get("strict_math", False)),
        "graphs": graphs,
        "operations": backend.operation_count,
        "parity": parity,
        "cold_call_seconds": timings[0],
        "warm_call_seconds": timings[1:],
        "frame_counter_delta": {
            key: measured_frames.get(key, 0) - before.get(key, 0) for key in measured_frames
        },
        "probe_frame_counter_delta": {
            key: after.get(key, 0) - measured_frames.get(key, 0) for key in after
        },
        "guard_failures": guard_failures,
        "repeated_identical_graph_captures": graphs - len(backend.graph_fingerprints),
        "fully_skipped_frames": "not separately exposed by this Dynamo version; frame totals recorded",
        "explain": explained,
        "fullgraph": fullgraph,
        "backend": "counted-graphmodule",
        "performance_qualified": False,
    }


def run_compile_suite(output, precision="fp16"):
    if precision not in ("fp16", "fp32"):
        raise ValueError("compile precision must be fp16 or fp32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cases = []
    for clipping in ("global", "adagc"):
        try:
            cases.append(
                _model_case(device, clipping, amp=device == "cuda" and precision == "fp16")
            )
        except Exception as error:
            cases.append(
                {
                    "name": f"compiled-model-{clipping}",
                    "status": "failed",
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
    for clipping, channels_last in [("global", False), ("adagc", False), ("adagc", True)]:
        try:
            cases.append(_optimizer_case(device, clipping=clipping, channels_last=channels_last))
        except Exception as error:
            cases.append(
                {
                    "name": f"optimizer-step-{clipping}",
                    "status": "failed",
                    "channels_last": channels_last,
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
    result = {
        "suite": "compile",
        "device": device,
        "cases": cases,
        "model_precision": precision if device == "cuda" else "fp32",
        "status": "failed" if any(c["status"] == "failed" for c in cases) else "passed",
        "performance_qualified": False,
        "cuda_graphs": "unqualified",
    }
    Path(output, "compile.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result
