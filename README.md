# LionPlus8Bit

[![CI](https://github.com/julian-carpenter/LionPlus8Bit/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/julian-carpenter/LionPlus8Bit/actions/workflows/ci.yml)

**Lion+ for PyTorch, with native 8-bit momentum and a choice of global or AdaGC gradient clipping.**

LionPlus8Bit combines Lion's signed updates, gradient clipping before momentum, and bitsandbytes' blockwise momentum quantization. Optimized execution is the default. Global clipping is the default policy; AdaGC is an option for clipping each parameter tensor against its own recent gradient history.

The package is an early release. It saves optimizer-state memory, but task quality and speed depend on your workload. The [corrective validation](docs/validation-audit-fixes.md) passed 172 tests, 15 subtests and six independent CUDA integration groups, including new AMP, checkpoint and numerical fallback regressions. AdaGC's three-seed application quality comparison was inconclusive, and it increased optimizer-only latency in the tested replay matrix. See [validation](#validation-and-limitations) before choosing a policy.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Lion, Lion+, and AdaGC explained](#lion-lion-and-adagc-explained)
- [How 8-bit momentum saves memory](#how-8-bit-momentum-saves-memory)
- [Choosing hyperparameters](#choosing-hyperparameters)
- [Constructor reference](#constructor-reference)
- [Mixed precision and gradient accumulation](#mixed-precision-and-gradient-accumulation)
- [Parameter groups and FP32 controls](#parameter-groups-and-fp32-controls)
- [Checkpoints and resuming](#checkpoints-and-resuming)
- [Diagnostics and benchmarking](#diagnostics-and-benchmarking)
- [Validation and limitations](#validation-and-limitations)
- [Development and CI](#development-and-ci)
- [References and license](#references-and-license)

## Installation

Requires Python 3.10+ and PyTorch `>=2.5,<3`. Install a PyTorch build suited to your hardware using the [official installation guide](https://pytorch.org/get-started/locally/), then install this package from GitHub:

```bash
python -m pip install "git+https://github.com/julian-carpenter/LionPlus8Bit.git"
```

For a local checkout:

```bash
git clone https://github.com/julian-carpenter/LionPlus8Bit.git
cd LionPlus8Bit
python -m venv .venv
source .venv/bin/activate
# Install the appropriate PyTorch build in this environment first.
python -m pip install .
```

For reproducible experiments, pin a commit: `git+https://github.com/julian-carpenter/LionPlus8Bit.git@COMMIT_SHA`. The distribution and import name are both `lionplus8bit`; the optimizer class is `LionPlus8bit`.

Version 0.1.0 pins `bitsandbytes==0.50.2` because it depends on that backend's state layout and update interface. CUDA use also requires a compatible NVIDIA driver and bitsandbytes binary. The optional numerical guard uses Triton when available and otherwise uses an eager PyTorch implementation. CPU execution supports the numerical tests and small examples; the GPU measurements below apply to the stated CUDA environment.

There is no PyPI release at present. Use the GitHub or checkout installation commands above.

## Quick start

This complete example fits a linear model to generated data. Move the model to its device before constructing the optimizer and keep its parameters in FP32:

```python
import torch
from lionplus8bit import LionPlus8bit


torch.manual_seed(7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inputs = torch.randn(128, 16, device=device)
true_weights = torch.randn(16, 4, device=device)
targets = inputs @ true_weights
model = torch.nn.Linear(16, 4).to(device)
optimizer = LionPlus8bit(
    model.parameters(),
    lr=0.01,  # Chosen for this small demonstration, not a general training recipe.
    weight_decay=0.0,
    max_grad_norm=1.0,
)

for step in range(300):
    optimizer.zero_grad(set_to_none=True)
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
    loss.backward()
    optimizer.step()  # Includes clipping before both Lion momentum mixtures.

print(f"Final MSE: {loss.item():.6f}")
```

For your own model, replace the generated data and loss with your training loop. Avoid an additional `clip_grad_norm_` call: this optimizer already clips, and a second clipping stage changes the algorithm.

Select AdaGC when constructing the optimizer:

```python
optimizer = LionPlus8bit(
    model.parameters(),
    lr=5e-5,
    clipping="adagc",
    max_grad_norm=1.0,  # Absolute threshold during warmup and cold-start bootstrap.
    adagc_beta=0.99,
    adagc_factor=1.04,
    adagc_warmup_steps=100,
)
```

To run the repository's regression examples:

```bash
python examples/train_regression.py --device cpu
python examples/train_regression.py --device cpu --clipping adagc
python examples/train_regression.py --device cuda --clipping adagc
```

The CUDA command requires a GPU and refuses a CPU fallback. These examples check basic use; they do not predict convergence on your task.

## Lion, Lion+, and AdaGC explained

### Lion: use momentum to choose a direction

A gradient estimates how changing each model weight would change the loss. Momentum averages recent gradients to reduce noise. Lion stores one momentum vector, mixes it with the current gradient, and takes the **sign** of that mixture to choose each weight's update direction. A second mixture updates the stored momentum. Unlike Adam, Lion has no second-moment vector. [Lion paper, Algorithm 2](https://arxiv.org/html/2302.06675#A1).

Let $m$ be the previous momentum, $g$ the gradient supplied to Lion, $\theta$ the weights, $\eta$ the learning rate, and $\lambda$ the weight decay:

$$
c_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
$$

$$
\theta_t = (1-\eta_t\lambda)\theta_{t-1}
           - \eta_t\operatorname{sign}(c_t)
$$

$$
m_t = \beta_2 m_{t-1} + (1-\beta_2)g_t.
$$

Both mixtures use the old momentum. `sign` returns +1, -1, or 0; there is no bias correction. Weight decay shrinks the weights separately from the gradient, which makes it *decoupled* weight decay.

### Lion+: clip the gradient before it enters momentum

A rare, large gradient can distort momentum for several updates. Lion+ adds global norm clipping before both mixtures. If $g$ is the vector of gradients across all active parameters and $M$ is the threshold:

$$
\widehat g = g\min\left(1,\frac{M}{\lVert g\rVert_2}\right).
$$

Use scale one for a zero norm and substitute $\widehat g$ for $g$ in the Lion equations. Global clipping scales the whole vector together, preserving its direction in exact arithmetic. The Lion+ paper studies this approach under heavy-tailed gradient noise. [Lion+ paper, Appendix C.2](https://arxiv.org/html/2506.04192v4#A3.SS2).

For example, `(3, 4)` has length 5. A threshold of 2.5 produces `(1.5, 2)`. With zero initial momentum, the signed directions remain positive: clipping halves the gradient entering momentum, but does not halve the first signed weight update. You still need to tune the learning rate.

### AdaGC: give each tensor a changing threshold

Global clipping can shrink every tensor because one tensor has a spike. AdaGC tracks each tensor's own gradient history and clips it independently. Here a tensor means one parameter, such as an entire weight matrix or bias vector.

After warmup, let $n_i=\lVert g_i\rVert_2$, $\gamma_i$ be the previous history, $r$ the relative threshold, and $b$ the history smoothing coefficient:

$$
h_i=\min\left(1,\frac{r\gamma_i}{n_i}\right),\qquad
\widehat g_i=h_i g_i,
$$

$$
\gamma_i^{new}=b\gamma_i+(1-b)\lVert\widehat g_i\rVert_2.
$$

The previous history determines the threshold, and the **clipped** norm updates history. There is no EMA bias correction. Use scale one for zero gradients. This is history-based gradient clipping, distinct from AGC methods that compare gradients with parameter lengths. [AdaGC paper, Section 4 and Algorithm 1](https://arxiv.org/html/2502.11034v4).

With histories `(2, 10)` and `adagc_factor=1.04`, the ceilings are `(2.08, 10.4)`. Current norms `(10, 5)` clip only the first tensor. With `adagc_beta=0.99`, the histories become `(2.0008, 9.95)`.

The first 100 completed steps use global clipping and record the minimum positive clipped norm per tensor. Step 101 switches to AdaGC. Overflow-skipped calls and calls with no active gradients do not advance the counter. This clipping warmup is separate from learning-rate warmup. Afterward, there is no second global cap.

### Practical extensions to AdaGC

This package defines edge cases beyond the paper's equations:

- Zero warmup norms do not reduce history to zero, which would otherwise suppress future gradients permanently.
- A tensor first activated after warmup receives one per-tensor absolute bootstrap clip, then seeds its history from the clipped norm. This also defines zero-length warmup.
- `grad=None` leaves that tensor's history unchanged. An active zero gradient can still trigger momentum and weight-decay updates. Established histories decay on zero observations and become unseeded below FP32 work representability.
- An explicit zero branch replaces a fixed denominator epsilon. Tiny and extreme values use per-tensor numerical fallback paths.

The linked Paddle and author PyTorch reference implementations differ in warmup counting, raw versus clipped warmup history, and epsilon. This implementation uses exactly 100 steps by default and clipped history with the zero recovery above. See the [design and implementation record](docs/superpowers/plans/2026-11-11-adagc.md) for those decisions.

AdaGC changes the globally clipped Lion+ algorithm. The original global-bound theory does not automatically cover this adaptive policy, 8-bit momentum, or arbitrary floating-point behavior.

## How 8-bit momentum saves memory

FP32 uses four bytes per value. bitsandbytes represents eligible momentum tensors with one-byte codes, a codebook, and block scale factors. Its kernels reconstruct approximate momentum, calculate updates, and encode the next momentum. Blocks limit how far one large value affects other values. [8-bit optimizer paper](https://arxiv.org/abs/2110.02861).

For an eligible tensor with $N$ elements, the pinned backend uses approximately

$$
N+4\lceil N/256\rceil\text{ bytes}
$$

for momentum codes and scales, plus its share of a codebook. FP32 momentum uses $4N$ bytes. Tensors smaller than 4,096 elements retain FP32 momentum by default.

| Component | FP32 momentum | Eligible 8-bit momentum |
| --- | ---: | ---: |
| Model weights, with FP32 parameters | $4N$ bytes | $4N$ bytes |
| Momentum payload | $4N$ bytes | About $1.016N$ bytes + codebook |

One million eligible elements therefore use about 1.02 MB of momentum payload instead of 4 MB. Gradients, model weights, activations, temporary buffers, and allocator reserves still occupy memory. A roughly 75% momentum reduction does not mean 75% less total training VRAM.

With actual FP16/BF16 parameters, the default FP32 master weights add another $4N$ bytes for those parameters. FP32 parameters with autocast avoid this extra master copy. AdaGC adds packed FP32 history per tensor, masks, cached indices and readback storage; small metadata vectors use FP64 arithmetic and rare histories use FP64 exceptions. There is no ordinary parameter-sized FP32 shadow momentum for quantized tensors. Strict arithmetic is selected per affected tensor and remains conservative across checkpoints. Strict uint8 updates decode/requantize momentum in chunks of at most 65,536 elements with independent 256-element block scales. This bounds momentum scratch; clipping and flattening a strided gradient can still allocate a full gradient temporary. Old checkpoints with conservative `strict_math` flags retain those flags for numerical continuity.

Quantization introduces error, including possible sign changes near zero. Use FP32 controls and your own validation metric to decide whether the memory savings suit your task.

## Choosing hyperparameters

| Option | Default | What it controls |
| --- | --- | --- |
| `lr` | `5e-5` | Magnitude of each nonzero signed update |
| `betas` | `(0.95, 0.98)` | Direction mixing, then momentum storage |
| `weight_decay` | `1e-2` | Decoupled weight shrinkage |
| `max_grad_norm` | `4.0` | Global threshold; AdaGC warmup/bootstrap threshold |
| `clipping` | `"global"` | Global clipping or opt-in `"adagc"` |
| `adagc_beta` | `0.99` | Smoothing of clipped norm history |
| `adagc_factor` | `1.04` | Adaptive ceiling relative to previous history |
| `adagc_warmup_steps` | `100` | Completed global-clipping steps before AdaGC |
| `min_8bit_size` | `4096` | Minimum tensor size eligible for 8-bit momentum |

**Moving from AdamW.** The Lion paper suggests trying learning rates about 3–10 times smaller and adjusting weight decay to preserve the intended `lr * weight_decay` product. Treat this as a tuning guide. Apply a chosen LR ratio consistently across the schedule. [Lion paper, Section 5](https://arxiv.org/html/2302.06675#S5).

**Learning rate and clipping have different effects.** Lowering a clipping threshold changes the gradient entering momentum; a nonzero signed weight update still has magnitude `lr`. If updates oscillate, examine the learning rate and schedule as well as clipping.

**Keep the batch policy fixed while comparing settings.** Loss reduction, effective batch size and gradient accumulation all affect gradient norms. Start a global-versus-AdaGC comparison with identical LR, decay, initialization, data order and training budget. Declare your validation metric and acceptable quality change first.

**Tune Lion and AdaGC betas separately.** Lion's second beta controls momentum memory: its approximate horizon is `1 / (1 - beta2)` updates. AdaGC's beta controls gradient-length history. Higher values smooth more but react more slowly.

**Understand AdaGC's growth limit.** With beta .99 and factor 1.04, a consistently clipped history grows by at most 0.04% per update. A sustained distribution change can outpace it. Larger factors clip less; factor one prevents ceilings from increasing. The factor must be finite and at least one, beta must remain below one in the working precision, and warmup must be a nonnegative integer.

**Choose the absolute threshold explicitly.** AdaGC does not change the constructor's `max_grad_norm=4.0` default. The AdaGC examples use 1.0 to match the paper's reported absolute setting. In global mode, `max_grad_norm=float("inf")` disables clipping while retaining gradient validation. In AdaGC mode, infinity disables the absolute warmup/bootstrap cap; adaptive clipping still operates after history initialization.

The package creates no learning-rate schedule. Use a PyTorch scheduler and advance it after successful optimizer updates, including when using accumulation and AMP.

## Constructor reference

```python
LionPlus8bit(
    params,
    lr=5e-5,
    betas=(0.95, 0.98),
    weight_decay=1e-2,
    max_grad_norm=4.0,
    min_8bit_size=4096,
    is_paged=False,
    master_weights=True,
    native_gradient_scale=True,
    batched_norm=True,
    # The following options are keyword-only:
    clipping="global",
    adagc_beta=0.99,
    adagc_factor=1.04,
    adagc_warmup_steps=100,
)
```

`params` accepts parameters or PyTorch parameter-group dictionaries. All registered parameters must share one device; active parameters must use contiguous storage; noncontiguous gradients are supported. Groups can have different Lion LR, betas and decay, but must share the clipping policy/settings and absolute threshold.

Parameter and gradient storage must be separate, and distinct parameters must not overlap. Changing a parameter's dtype or device after stepping requires a fresh optimizer; move/cast the model before constructing it. Disjoint contiguous parameter views are accepted.

Optimized defaults batch norm calculations and pass scale values to native kernels when the precision guard permits it. The pinned native API accepts host floats, so AdaGC reads back a packed coefficient vector rather than one GPU scalar per tensor. It is an eager implementation with host synchronization, not a fully fused device-only optimizer.

`master_weights=True` maintains FP32 copies for actual FP16/BF16 parameters. Keep it enabled unless you deliberately accept rounding of small weight updates. `is_paged=True` exposes native paging, which remains unverified here.

For legacy arithmetic comparisons, set both `native_gradient_scale=False` and `batched_norm=False`. Safety checks remain active. Nonfinite gradients fail before updates; extreme norms/scales use FP64 work, and tiny values use strict arithmetic to avoid native flush-to-zero loss. A positive clipping request below FP32 representability fails when clipping is needed. A backend error partway through updates can leave partially updated parameters: restore a known checkpoint before continuing.

## Mixed precision and gradient accumulation

Prefer FP32 parameters with autocast. Ordinary `GradScaler.step(optimizer)` unscales gradients before calling the optimizer and skips that call on overflow, leaving AdaGC histories and counters unchanged. Clip once after accumulating the full effective batch. [PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).

This complete CUDA example handles a partial final accumulation group and advances the scheduler after successful updates:

```python
import math
from itertools import islice

import torch
from torch.utils.data import DataLoader, TensorDataset

from lionplus8bit import LionPlus8bit


if not torch.cuda.is_available():
    raise SystemExit("This mixed-precision example requires CUDA")

torch.manual_seed(7)
model = torch.nn.Linear(16, 4).cuda()  # Keep parameters FP32.
dataset = TensorDataset(torch.randn(259, 16), torch.randn(259, 4))
loader = DataLoader(dataset, batch_size=16, shuffle=False)
accumulation_steps = 4
optimizer = LionPlus8bit(model.parameters(), clipping="adagc", max_grad_norm=1.0)
scaler = torch.amp.GradScaler("cuda")
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=math.ceil(len(loader) / accumulation_steps)
)

iterator = iter(loader)
while batches := list(islice(iterator, accumulation_steps)):
    optimizer.zero_grad(set_to_none=True)
    sample_count = sum(len(inputs) for inputs, _ in batches)
    for inputs, targets in batches:
        inputs, targets = inputs.cuda(), targets.cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = torch.nn.functional.mse_loss(model(inputs), targets)
            loss = loss * len(inputs) / sample_count
        scaler.scale(loss).backward()

    previous_scale = scaler.get_scale()
    scaler.step(optimizer)  # Unscale, check overflow, then clip and update.
    scaler.update()
    if scaler.get_scale() >= previous_scale:
        scheduler.step()  # Standard GradScaler reduces its scale on overflow.
```

The sample weighting assumes a mean loss with the same output shape per example. Adapt normalization for token counts, masks or other loss reductions. Keep the loss scale constant within an accumulation group. Explicit `scaler.unscale_(optimizer)` is needed only if you want to inspect unscaled gradients before `scaler.step`; call it once after accumulation.

Actual FP16 parameters differ from FP16 autocast. Ordinary PyTorch GradScaler rejects FP16 gradients, and internal FP32 master weights do not bypass that restriction. Use already unscaled gradients or a separately validated integration for actual FP16 parameters. BF16 autocast usually uses no scaler; check hardware support. Numerical BF16 execution does not imply native BF16 acceleration.

After a successful step, `.grad` retains your original, unscaled, **unclipped** gradient values. The optimizer uses a scale or temporary internally; it does not replace your gradients with the clipped values.

## Parameter groups and FP32 controls

For example, exclude one-dimensional parameters from decay:

```python
decay, no_decay = [], []
for parameter in model.parameters():
    if parameter.requires_grad:
        (no_decay if parameter.ndim < 2 else decay).append(parameter)

optimizer = LionPlus8bit(
    [
        {"params": decay, "weight_decay": 1e-2},
        {"params": no_decay, "weight_decay": 0.0},
    ],
    lr=5e-5,
    clipping="adagc",
    max_grad_norm=1.0,
)
```

Check that this decay policy suits your architecture. A zero gradient is active; `grad=None` skips that parameter.

To force FP32 momentum for every parameter while retaining the same clipping implementation:

```python
parameters = list(model.parameters())
optimizer = LionPlus8bit(
    parameters,
    clipping="adagc",
    max_grad_norm=1.0,
    min_8bit_size=1 + max(parameter.numel() for parameter in parameters),
)
```

You can also use the native manager to override selected parameters before their first update:

```python
import bitsandbytes as bnb


manager = bnb.optim.GlobalOptimManager.get_instance()
manager.override_config(model.weight, "optim_bits", 32)  # For the Linear model above.
```

The manager is process-wide. Each optimizer owns its resolved effective overrides, including module precision defaults for dynamically added groups. A later parameter-specific `manager.override_config(...)` call updates that optimizer's policy. Avoid unrelated index registrations in controlled comparisons and inspect actual state dtypes after initialization. Overrides such as `skip_zeros` and `max_unorm` change the update contract and are rejected.

## Checkpoints and resuming

Save model and optimizer state, plus the scheduler and scaler if used:

```python
torch.save(
    {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
    },
    "checkpoint.pt",
)
```

On restart, reconstruct the model and parameter groups in the same order and move the model to its device. Construct the optimizer and scheduler before loading their states, then:

```python
checkpoint = torch.load("checkpoint.pt", map_location="cpu", weights_only=True)
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
scheduler.load_state_dict(checkpoint["scheduler"])
scaler.load_state_dict(checkpoint["scaler"])
```

The versioned optimizer checkpoint includes clipping policy, AdaGC warmup/history, precision policy, master weights and native overrides. Saved policy takes precedence over destination index and module defaults on resume; loaded overrides are independently owned. Public save/load hooks run once around the complete native and clipping checkpoint. The optional `optimizer.load_state_dict(state, move_to_device=False)` defers native state migration until the first step. Old checkpoints without the AdaGC schema restore global clipping. Construct a fresh optimizer to change clipping strategy. Rebuild identical group/parameter order: same-shaped unnamed parameters swapped within a group cannot be detected automatically. Standalone historical `exp_avg` checkpoints are not migrated.

A complete training restart also needs RNG states, data/sampler position and any partial accumulated gradients. The small example above saves optimizer-related state, not those application-specific pieces. The integration suite verifies controlled fresh-process continuation around the warmup boundary; arbitrary DataLoader worker restarts remain unverified.

## Diagnostics and benchmarking

Inspect live initialized momentum after representative updates:

```python
for parameter, state in optimizer.state.items():
    if "state1" in state:
        print(tuple(parameter.shape), state["state1"].dtype)

# AdaGC diagnostics describe the most recent completed step.
if optimizer.last_clipping is not None:
    scales = optimizer.last_clipping["scales"]
    print("Clipped tensors:", sum(scale < 1.0 for scale in scales), "of", len(scales))
```

Unused parameters may have no state yet. AdaGC diagnostics contain raw norms, applied scales, clipped-norm histories and step/warmup metadata. Check for an AMP skip before logging a new update, since a skipped call leaves the previous diagnostic intact.

`lionplus8bit.diagnostics.compare_signs(candidate, control, clipped_gradients)` compares momentum error and the next sign decision. Use matching parameter order and replay history, with identical **already-clipped** gradients. The diagnostic allocates decoded temporaries and synchronizes; keep it outside timed intervals. Freely evolving training runs do not isolate quantization because their gradients differ.

`optimizer.last_step_stats` describes the latest completed update with active, clipped, tiny, strict, and recovered-norm tensor counts. AMP skips leave these diagnostics unchanged.

For a reproducible synthetic CUDA benchmark:

```bash
python tools/benchmark_clipping.py \
  --strategy global --layout many-small --pattern localized \
  --dtype float32 --warmup 150 --steps 500 --seed 0 \
  --output results/global.json

python tools/benchmark_clipping.py \
  --strategy adagc --layout many-small --pattern localized \
  --dtype float32 --warmup 150 --steps 500 --seed 0 \
  --output results/adagc.json
```

For balanced repeated comparisons that preserve every command and rotate arm order:

```bash
python tools/run_benchmark_campaign.py \
  --cases ordinary zero recovery --modes replay amp \
  --strategies global adagc native --repeats 3 \
  --output-dir results/benchmark-campaign
```

Use `--quick` for six short replay smoke runs. The native control has no Lion+ clipping, and synthetic measurements do not establish task quality.

Use a fresh output path; the tool refuses overwrite. Add `--profile` for a Chrome trace. It reports synchronized wall/CUDA samples, median/p95, and allocated/reserved memory peaks. `--warmup` is timing warmup and must be at least 100; it differs from `adagc_warmup_steps`. Measure end-to-end training as well as optimizer-only time, and distinguish initialization from steady state. The benchmark also supports `--case zero|tiny|recovery|strided|variable-active`, `--stream nondefault`, and `--repeats 3`. Outputs include spike/nonspike and path-conditioned distributions, each raw update, persistent state bytes, and temporary allocated peaks.

To benchmark FP32 parameters with actual FP16 autocast and GradScaler:

```bash
python tools/benchmark_clipping.py \
  --strategy adagc --layout few-large --mode amp --dtype float16 \
  --case recovery --pattern localized --repeats 3 \
  --warmup 150 --steps 500 --output results/amp-recovery.json
```

AMP output separates forward/backward, unscale, finite stress-input conditioning, optimizer and scaler-update timing. Conditioning deliberately injects the selected finite test pattern after unscale; this is an instrumented stress benchmark. `--strategy native` measures unclipped bitsandbytes Lion8bit as a backend baseline, which has different clipping semantics. Native BF16 performance mode requires hardware support.

## Validation and limitations

The initial validation run used a Quadro RTX 6000 (SM75), Python 3.12.3, NVIDIA PyTorch 2.13 alpha, CUDA runtime 13.3 and bitsandbytes 0.50.2. The loader selected its CUDA 13.2 binary. These results describe that environment, not every supported PyTorch/backend combination.

| Check | Observed result |
| --- | --- |
| Final installed-wheel regression suite | 79 tests passed, plus 15 subtests |
| CPU-only collection | 50 passed, 29 CUDA tests deselected |
| Independent CUDA integration | Six groups passed, including bitwise fresh-process AdaGC resume |
| Build and packaging | Wheel/sdist, strict metadata checks and isolated-wheel usage passed |
| AdaGC vs global application quality | Inconclusive against a 2% relative validation-loss margin |
| AdaGC optimizer-only replay latency | About 1.03–2.21 times global-clipping latency across tested FP32 cases |
| Additional peak allocated memory in those cases | 5,632–30,208 bytes |
| Matched-gradient uint8 vs FP32 momentum replay | 1.044% sign disagreement on one 100-update replay |

The application comparison used three paired seeds, 1,100 updates per arm and a small real-audio subset with a custom training loop. AdaGC's mean relative validation loss was **+0.965%**, with a 95% interval of **−1.272% to +3.201%**. That interval does not establish noninferiority at the declared +2% margin. Short runs without loss spikes do not establish long-term stability. Neither the quality study nor the microbenchmarks support a universal quality or speed claim.

Read the [full validation report](docs/validation-adagc.md) and [portable raw evidence](docs/evidence/adagc/README.md) for methods, commands, corrected timing results and limitations. The repository does not include application audio, model-source dependencies or trained checkpoints; synthetic tests and benchmarks need no application dataset.

| Configuration | Status |
| --- | --- |
| Single-device CUDA; FP32 parameters; uint8/FP32 momentum | Tested locally |
| FP16 autocast, AMP overflow skips, accumulation, controlled resume | Tested locally |
| Actual FP16/BF16 parameters with master weights | Numerical tests; observe the GradScaler restriction above |
| CPU | Numerical/API tests, examples and build CI |
| Native BF16 throughput on SM75 | Unavailable; no performance claim |
| Sparse gradients, FP64/complex parameters, noncontiguous parameters | Rejected |
| Multiple devices in one optimizer | Rejected |
| DDP, paging, compiled optimizer, arbitrary worker restart | Unverified |
| FSDP/ZeRO gradient shards, CUDA graphs | Unsupported |
| FP8 integration | Unverified |

Flattening parameters changes AdaGC's tensor granularity. Distributed shards require logical-tensor norm reductions and shared history semantics before use. Clipping cannot fix upstream numerical drift, infrastructure errors or nonfinite inputs.

## Development and CI

```bash
python -m pip install -e '.[dev]'
python -m pytest -m 'not cuda'
python -m pytest --require-cuda
LION_RESULTS=results/integration python tools/cuda_integration.py
python -m build
python -m twine check --strict dist/*
python -m ruff check .
python -m ruff format --check .
```

[CI](https://github.com/julian-carpenter/LionPlus8Bit/actions/workflows/ci.yml) runs on pushes and pull requests for Python 3.11/3.12 with current CPU PyTorch, plus Python 3.10 with PyTorch 2.5.0 to enforce the declared minimum versions. It builds distributions, installs the wheel, checks imports outside the checkout, runs CPU tests and both clipping examples, and uploads test/build artifacts. The badge covers this workflow. No PyPI publishing step is configured.

[CUDA validation](https://github.com/julian-carpenter/LionPlus8Bit/actions/workflows/cuda.yml) is a manual workflow for a trusted Linux x64 runner labeled `gpu` with CUDA PyTorch installed. It requires real CUDA, runs the regression suite, independent integration checks and examples, and uploads logs. `pytest` skips CUDA-marked tests without a GPU unless you supply `--require-cuda`, which fails instead. The unittest integration tool performs its own CUDA check and does not accept that pytest flag.

## References and license

- Chen et al., [Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675): Lion.
- Sfyraki and Wang, [Lions and Muons: Optimization via Stochastic Frank-Wolfe under Heavy-Tailed Noise](https://arxiv.org/abs/2506.04192): Lion+; Appendix C.2 is Algorithm 8 in v4 and Algorithm 7 in v1.
- Wang et al., [AdaGC: Enhancing LLM Pretraining Stability via Adaptive Gradient Clipping](https://arxiv.org/html/2502.11034v4): adaptive historical-gradient clipping.
- Dettmers et al., [8-bit Optimizers via Block-wise Quantization](https://arxiv.org/abs/2110.02861).
- [bitsandbytes Lion API](https://huggingface.co/docs/bitsandbytes/en/reference/optim/lion) and [implementation](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/optim/lion.py).
- [PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).

This package extends bitsandbytes' native optimizer. It is an independent implementation and has no affiliation with the paper authors or bitsandbytes. Code uses the [MIT license](LICENSE); dependencies retain their own licenses.
