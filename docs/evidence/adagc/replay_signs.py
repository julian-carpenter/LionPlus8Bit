import json
import math
from pathlib import Path
import torch
from lionplus8bit import LionPlus8bit
from lionplus8bit.diagnostics import compare_signs

torch.manual_seed(17)
p = torch.nn.Parameter(torch.ones(4353, device="cuda"))
q = torch.nn.Parameter(p.detach().clone())
a = LionPlus8bit([p], max_grad_norm=math.inf, min_8bit_size=0, weight_decay=0)
b = LionPlus8bit([q], max_grad_norm=math.inf, min_8bit_size=100000, weight_decay=0)
for parameter, opt in [(p, a), (q, b)]:
    parameter.grad = torch.zeros_like(parameter)
    opt.step()
history = None
rows = []
for step in range(100):
    g = torch.randn_like(p) * (0.01 if step % 17 else 1.0)
    n = float(torch.linalg.vector_norm(g, dtype=torch.float64))
    h = min(1.0, 1.0 / n) if step < 10 else min(1.0, 1.04 * history / n)
    clipped = g * h
    observed = float(torch.linalg.vector_norm(clipped, dtype=torch.float64))
    history = (
        observed
        if history is None
        else min(history, observed)
        if step < 10
        else 0.99 * history + 0.01 * observed
    )
    diagnostic = compare_signs(a, b, [clipped])[0]
    rows.append(dict(step=step + 1, norm=n, scale=h, history=history, **diagnostic))
    p.grad = clipped
    q.grad = clipped.clone()
    a.step()
    b.step()
result = dict(
    scope="Identical AdaGC-preprocessed FP32 gradient replay; independent Python norm-history oracle, 10-step warmup; next-sign disagreement measures quantized momentum only",
    rows=rows,
    total_elements=sum(r["elements"] for r in rows),
    sign_disagreements=sum(r["sign_disagreements"] for r in rows),
)
result["disagreement_fraction"] = result["sign_disagreements"] / result["total_elements"]
Path("results/adagc/sign_replay.json").write_text(json.dumps(result, indent=2) + "\n")
print({k: v for k, v in result.items() if k != "rows"})
