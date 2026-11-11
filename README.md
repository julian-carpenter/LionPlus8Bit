# LionPlus8Bit

[![CI](https://github.com/julian-carpenter/LionPlus8Bit/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/julian-carpenter/LionPlus8Bit/actions/workflows/ci.yml)

A PyTorch optimizer that combines **Lion+ gradient clipping** with **bitsandbytes 8-bit momentum storage**. Optimized mode is the default: the package batches gradient-norm calculations and supplies the clipping scale to native update kernels when possible.

Use it when optimizer state takes a meaningful share of your training memory and you can evaluate a change of optimizer on your task. Keep model weights in FP32 and use autocast for mixed-precision training as a starting point.

**Status:** early release. This repository is private; cloning, installing from GitHub, and viewing CI require repository access. The badge covers automatic CPU tests and package builds. CUDA validation is a separate, manually dispatched workflow for a trusted GPU runner.

This package quantizes **optimizer state**. It does not quantize your model to 8 bits, automatically choose training hyperparameters, or promise faster training. See [validation and limits](#validation-and-limits) for the evidence and its scope.

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [The ideas behind the optimizer](#the-ideas-behind-the-optimizer)
- [How 8-bit storage saves memory](#how-8-bit-storage-saves-memory)
- [Choose hyperparameters](#choose-hyperparameters)
- [Adaptive clipping with AdaGC](#adaptive-clipping-with-adagc)
- [Constructor reference](#constructor-reference)
- [Mixed precision and accumulated gradients](#mixed-precision-and-accumulated-gradients)
- [Parameter groups and selective precision](#parameter-groups-and-selective-precision)
- [Checkpoints and resuming](#checkpoints-and-resuming)
- [Diagnostics and memory accounting](#diagnostics-and-memory-accounting)
- [Validation and limits](#validation-and-limits)
- [Development and CI](#development-and-ci)
- [References and license](#references-and-license)

## Install

Use Python 3.10 or later. Install a compatible PyTorch build first, especially if you need NVIDIA CUDA: use the [official PyTorch selector](https://pytorch.org/get-started/locally/). A Python package cannot install or repair your GPU driver.

For this private repository, authenticate Git without putting a token in a URL or source file:

```bash
gh auth login
gh auth setup-git
git clone https://github.com/julian-carpenter/LionPlus8Bit.git
cd LionPlus8Bit

python -m venv .venv
source .venv/bin/activate
# Install the appropriate PyTorch build here, following its official selector.
python -m pip install .
```

Once Git authentication works, a direct Git installation is also possible:

```bash
python -m pip install "git+https://github.com/julian-carpenter/LionPlus8Bit.git"
```

For repeatable experiments, replace the branch reference with a commit: `git+https://github.com/julian-carpenter/LionPlus8Bit.git@COMMIT_SHA`.

The distribution is named `lionplus8bit`; the import is `lionplus8bit`. Version 0.1.0 pins `bitsandbytes==0.50.2` because this implementation relies on its state layout and native update interface. The optional fused numerical guard uses Triton when it is available. Otherwise the package uses an eager PyTorch guard with the same purpose. Installing a CUDA PyTorch build commonly provides Triton on supported Linux systems; the package does not install an arbitrary extra compiler version over your PyTorch environment.

There is no PyPI release configured. `pip install lionplus8bit` from a public index is not the installation method for this repository.

## Quick start

Move the model to its device **before** constructing the optimizer:

```python
import torch
from lionplus8bit import LionPlus8bit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.nn.Linear(128, 64).to(device)  # FP32 weights
optimizer = LionPlus8bit(
    model.parameters(),
    lr=5e-5,
    betas=(0.95, 0.98),
    weight_decay=1e-2,
    max_grad_norm=4.0,
)

for inputs, targets in training_loader:  # your DataLoader
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad(set_to_none=True)
    loss = loss_function(model(inputs), targets)
    loss.backward()
    optimizer.step()  # includes global gradient clipping
```

You supply the model, data and loss function. Do not call `clip_grad_norm_` as an additional clipping stage unless you intend to change the algorithm.

For a complete synthetic example that requires no dataset:

```bash
python examples/train_regression.py --device cpu
python examples/train_regression.py --device cuda
```

The CUDA command fails if CUDA is unavailable. The example checks a small regression problem and prints the actual momentum dtypes; it is a smoke test, not evidence about your model's convergence.

## The ideas behind the optimizer

### Start with a gradient

A gradient tells you how changing each weight would change the current loss. Plain gradient descent subtracts a small multiple of that vector. The learning rate controls the size of that move.

A single minibatch gives a noisy estimate. Momentum smooths estimates over time, like a moving average that remembers recent directions. An optimizer stores this memory as **state**, separate from the model weights and the current gradients.

### Lion chooses a direction using momentum

Lion maintains one momentum vector. It mixes the old momentum with the current gradient, takes the sign of that mixture, and moves each weight in the chosen direction. It then updates momentum with a second mixture. The two beta coefficients serve different roles; Lion does not store Adam's second-moment estimate. [Chen et al., Algorithm 2](https://arxiv.org/html/2302.06675#A1).

`sign(x)` is +1 for a positive value, -1 for a negative value, and 0 at zero. A larger gradient therefore does not directly produce a larger signed step. Learning rate matters even when you clip the gradient.

### Lion+ clips before updating momentum

Lion+ adds a limit to the gradient's Euclidean length before either momentum mixture. The length includes **all active parameters in all groups**. With clipping threshold $M$, a gradient whose length exceeds $M$ is scaled down as one vector. Its relative component directions stay the same in exact arithmetic. The paper introduces this to address heavy-tailed noise, where unusually large gradient samples can distort momentum. [Sfyraki and Wang, Appendix C.2](https://arxiv.org/html/2506.04192v4#A3.SS2).

### The full update

Let $g_t$ be the gradient, $m_{t-1}$ the stored momentum, $\theta_{t-1}$ the weights, $\eta_t$ the learning rate, and $\lambda$ the weight decay:

$$
\bar g_t = g_t \min\left(1,\frac{M}{\lVert g_t\rVert_2}\right)
$$

For a zero gradient, use a scale of 1. Then compute:

$$
c_t = \beta_1 m_{t-1} + (1-\beta_1)\bar g_t
$$

$$
\theta_t = (1-\eta_t\lambda)\theta_{t-1} - \eta_t\operatorname{sign}(c_t)
$$

$$
m_t = \beta_2 m_{t-1} + (1-\beta_2)\bar g_t
$$

Both mixtures use the **old** momentum and the same clipped gradient. There is no bias correction. Weight decay shrinks the weights separately from the gradient; this is *decoupled* weight decay.

For a concrete first-step example, take $g=(3,4)$, whose length is 5. With $M=2.5$, the clipped gradient is $(1.5,2)$. If momentum starts at zero, both sign directions are still positive. Clipping changes the stored momentum's magnitude and its influence on later updates; it does not halve the first signed step. This follows from the equations above.

The optimizer in this package approximates $m_t$ with native quantized storage for eligible tensors. The paper's mathematical guarantees do not automatically extend to that quantization, arbitrary floating-point arithmetic or arbitrary training settings.

## How 8-bit storage saves memory

An FP32 number occupies four bytes. A uint8 value occupies one byte but can represent only 256 codes. bitsandbytes stores momentum as codes plus a codebook and per-block scale factors. During an update, it reconstructs an approximate momentum, does floating-point arithmetic, then encodes the new momentum. Blocks limit how far one large value affects the representation of other values. [8-bit optimizer paper](https://arxiv.org/abs/2110.02861), [bnb explanation](https://huggingface.co/docs/bitsandbytes/en/explanations/optimizers).

For the pinned backend, an eligible tensor with $N$ elements uses approximately

$$
N + 4\lceil N/256\rceil \text{ bytes}
$$

for codes and block scales, plus its share of a codebook. Compare this with $4N$ bytes for one FP32 momentum vector. The default allocation threshold keeps tensors with fewer than 4,096 elements in FP32.

| Storage with FP32 model weights | Model weights | Momentum, approximately |
|---|---:|---:|
| Lion with FP32 momentum | $4N$ bytes | $4N$ bytes |
| This optimizer, eligible large tensor | $4N$ bytes | $1.016N$ bytes + codebook |

For one million eligible elements, momentum payload is about 1.02 MB instead of 4 MB, using decimal MB. Model weights, gradients, activations, allocator reserves, temporary buffers and metadata still occupy memory. A roughly 75% momentum reduction is **not** a 75% reduction in total training VRAM.

Quantization introduces error. A small change near zero can change a sign decision, so 8-bit and FP32-state trajectories need not be identical. Keep selected states in FP32 if your own validation shows a problem.

## Choose hyperparameters

Treat the defaults as a starting point. They follow the Lion+ paper's language-model experiment, not a universal recommendation for every task.

| Parameter | Default | Practical meaning |
|---|---:|---|
| `lr` | `5e-5` | Size of each nonzero signed update, before accounting for decay |
| `betas` | `(0.95, 0.98)` | First coefficient controls direction mixing; second controls momentum storage |
| `weight_decay` | `1e-2` | Decoupled shrinkage strength |
| `max_grad_norm` | `4.0` | Global input-gradient norm limit |
| `min_8bit_size` | `4096` | Small tensors retain FP32 state |

**Transferring from AdamW.** The original Lion paper suggests trying learning rates around 3–10 times smaller. It also adjusts decay to keep the intended product `lr * weight_decay`; that is a tuning guide, not a rule to multiply settings blindly. Apply a chosen LR ratio consistently to the schedule's start, peak and end values. [Lion, Section 5](https://arxiv.org/html/2302.06675#S5).

**Learning rate and clipping solve different problems.** If updates oscillate or validation quality worsens, reduce the LR and inspect the schedule. Lowering `max_grad_norm` controls gradients entering momentum, but the signed update still has per-coordinate magnitude `lr` when the sign is nonzero.

**Tune the threshold with the batch policy fixed.** Keep effective batch size, loss reduction and gradient accumulation consistent. Inspect pre-clipping norms and clipping frequency. A threshold that clips almost every step creates a different momentum history from one that rarely clips. `max_grad_norm=float("inf")` disables clipping while retaining gradient validation.

**Choose betas deliberately.** Larger `beta2` remembers gradients longer. Roughly, the averaging horizon is $1/(1-\beta_2)$ updates: about 50 at 0.98 and 100 at 0.99. This is an interpretation of exponential averaging, not a guarantee of training quality. Very large betas that round to 1 in the native FP32 interface are rejected.

**Use an LR schedule.** The package does not create warmup or cosine decay for you. Use a PyTorch scheduler and advance it after successful optimizer updates. Tune schedules over update counts, taking gradient accumulation into account.

**Compare fairly.** Keep initialization, data order, precision, training budget, effective batch size and evaluation fixed across optimizer comparisons. Use several seeds when affordable. Declare the validation metric and acceptable quality change before viewing the result. Do not choose based only on training loss or memory use.

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
    clipping="global",
    adagc_beta=0.99,
    adagc_factor=1.04,
    adagc_warmup_steps=100,
)
```

- `params` accepts model parameters or PyTorch parameter-group dictionaries. All active parameters must use one device and contiguous storage. Noncontiguous gradients are supported.
- `native_gradient_scale=True` uses native FP32 gradient scaling when the numerical guard permits it, avoiding a clipped-gradient temporary.
- `batched_norm=True` batches norm calculations. Reduction/scaling rounding can differ from the earlier implementation.
- `master_weights=True` maintains an FP32 weight copy **only for direct FP16/BF16 parameters**, so small cumulative updates survive weight rounding. That adds $4N$ bytes for those parameters and is separate from momentum. FP32 model weights with autocast do not need this extra copy.
- `is_paged=False` keeps the ordinary nonpaged path. Paging is exposed but unverified here; it is not recommended as a default memory remedy.

To reproduce the earlier arithmetic policy explicitly:

```python
optimizer = LionPlus8bit(
    model.parameters(),
    native_gradient_scale=False,
    batched_norm=False,
)
```

The revised safety checks remain active. Neither mode promises arbitrary-precision behavior. Nonfinite gradients fail before native updates. Exceptional norms/scales use FP64 temporary work, and tiny-value cases use a strict arithmetic path to avoid native flush-to-zero loss. A clipping threshold below the smallest FP32 subnormal fails when clipping is required. Backend exceptions can leave partial parameter/state updates; `step()` is not transactional.

## Mixed precision and accumulated gradients

Prefer FP32 model weights with autocast. For FP16 autocast on CUDA, scale the loss, accumulate gradients, **unscale once**, then let this optimizer clip and update. [PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).

```python
scaler = torch.amp.GradScaler("cuda")
accumulation_steps = 4
optimizer.zero_grad(set_to_none=True)

# One effective batch consists of exactly four equal-sized microbatches here.
for inputs, targets in microbatches:
    inputs, targets = inputs.cuda(), targets.cuda()
    with torch.autocast("cuda", dtype=torch.float16):
        loss = loss_function(model(inputs), targets) / accumulation_steps
    scaler.scale(loss).backward()

scaler.unscale_(optimizer)
old_scale = scaler.get_scale()
scaler.step(optimizer)
scaler.update()
if scaler.get_scale() >= old_scale:  # default GradScaler: overflow reduces scale
    scheduler.step()  # if you use a scheduler
optimizer.zero_grad(set_to_none=True)
```

Here `microbatches` must contain exactly `accumulation_steps` equal-sized batches for that one update. Normalize uneven or partial batches by their actual sample counts. Repeat the block for subsequent effective batches.

GradScaler can skip an update on overflow; advancing a scheduler on that skipped update would change the schedule. The optimizer normally returns `None`, so its return value does not identify a skip. The scale comparison above assumes the standard scaler policy. BF16 autocast often uses no scaler; use your hardware's supported precision policy. Direct FP16 parameters with FP32 master weights do not automatically bypass GradScaler's restrictions on unscaling FP16 gradients.

After a step, `.grad` still refers to your original gradients. If you explicitly unscaled first, those are unscaled, **unclipped** gradients. Log the clipping coefficient separately rather than interpreting post-step `.grad` as the values supplied to the native kernel.

## Parameter groups and selective precision

You can give groups different learning rates, betas and decay. They must share one clipping threshold:

```python
decay, no_decay = [], []
for name, parameter in model.named_parameters():
    if not parameter.requires_grad:
        continue
    (no_decay if parameter.ndim < 2 else decay).append(parameter)

optimizer = LionPlus8bit(
    [
        {"params": decay, "weight_decay": 1e-2},
        {"params": no_decay, "weight_decay": 0.0},
    ],
    lr=5e-5,
    max_grad_norm=4.0,
)
```

Dimension-based decay exclusion is an example policy; check that it fits your architecture. A zero gradient remains an active gradient: momentum and decay can still move the weight. `grad=None` skips that parameter.

To retain FP32 momentum for a selected parameter, use the pinned backend's manager **before the first update**:

```python
import bitsandbytes as bnb

manager = bnb.optim.GlobalOptimManager.get_instance()
manager.override_config(model.output_layer.weight, "optim_bits", 32)
```

Use a real attribute from your model. Verify actual state dtypes after initialization. The optimizer saves these overrides by parameter-group order. Overrides such as `skip_zeros` or `max_unorm` change the Lion+ update contract and are rejected.

For an all-FP32-state control with identical clipping logic:

```python
parameters = list(model.parameters())
threshold = 1 + max(p.numel() for p in parameters)
optimizer = LionPlus8bit(parameters, min_8bit_size=threshold)
```

This assumes no override forces 8-bit state. Check the states instead of relying on the class name.

## Checkpoints and resuming

Save `state_dict()` values rather than pickling the optimizer object:

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

On restart, reconstruct the same model and parameter-group **order**, move the model to its device, construct the optimizer and scheduler, then load their saved states. Construct the scheduler before loading the optimizer to avoid replacing restored learning rates. Load only checkpoints you trust; these contain Python structures as well as tensors.

```python
checkpoint = torch.load("checkpoint.pt", map_location="cpu", weights_only=False)
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
scheduler.load_state_dict(checkpoint["scheduler"])
scaler.load_state_dict(checkpoint["scaler"])
```

A current optimizer checkpoint includes versioned constructor settings, precision policy, master weights and native overrides. This matters when a previously unused parameter first receives a gradient after restart. For older checkpoints lacking configuration metadata, supply the original constructor settings yourself. Historical standalone Lion checkpoints using `exp_avg` are not automatically migrated.

A complete training restart also needs Python/NumPy/PyTorch RNG state, CUDA RNG state, data/sampler position and any partial accumulated gradients. The example checkpoint above omits those application-specific pieces; it is not a full restart guarantee. The CUDA integration tool tests complete continuation on a controlled generated stream in genuinely fresh processes, not arbitrary worker-based DataLoaders.

## Diagnostics and memory accounting

After representative updates, inspect initialized state:

```python
for parameter, state in optimizer.state.items():
    if "state1" in state:
        print(parameter.shape, state["state1"].dtype)
```

State is lazy: an unused parameter may have no entry yet. Count logical optimizer tensors once by underlying storage because codebooks can be shared. For actual training VRAM, measure both allocated and reserved GPU peaks over a defined, synchronized interval; include forward/backward and distinguish initialization from steady state.

`lionplus8bit.diagnostics.compare_signs(candidate, control, clipped_gradients)` compares next-step sign decisions and momentum errors. Use matching group order, matching replay history and the **same already-clipped gradients**. It allocates decoded temporary tensors and synchronizes. Keep it outside timing intervals. Two freely evolving training runs generally have different gradients, so their difference does not isolate quantization alone.

For timing, warm up first, synchronize at measurement boundaries, and report optimizer-step and end-to-end training times separately. Native dispatch and numerical checks have real costs. The optional Triton guard reduces guard overhead; it does not establish a speed advantage on every workload.

## Validation and limits

The prior local revision passed 40 tests, complete CUDA verification and six independent integration groups on a Quadro RTX 6000 with NVIDIA PyTorch 2.13 alpha/CUDA 13.3 and bnb 0.50.2. That wheel selected its CUDA 13.2 binary. This package changes optimized-mode defaults and tests those defaults separately from explicit legacy replay.

An exploratory three-seed real-audio subset study met its predefined loss margin and measured about 73% less logical optimizer-state storage than a forced-FP32-state control. It used a smaller model and custom training loop; it does not validate arbitrary application quality. In the final small-tensor microbenchmark, optimized mode was roughly 3.6 ms per clipped update versus 3.4 ms for the original wrapper. Numerical safety adds overhead; there is no general speedup claim. The private repository contains code and synthetic tests, not the original audio, datasets or application checkpoints.

| Configuration | Scope |
|---|---|
| Single-device CUDA, FP32 weights, quantized and FP32 states | Covered by local validation and manual GPU workflow |
| FP16 autocast, overflow skipping, accumulation, controlled fresh-process restart | Covered by CUDA integration fixtures |
| CPU | Automatic CI for numerical/API tests and wheel builds; no CUDA certification |
| BF16 | Numerical execution tested on the prior GPU; native BF16 acceleration was absent |
| Sparse gradients, FP64/complex parameters, noncontiguous parameters | Rejected |
| Multiple devices in one optimizer | Rejected |
| DDP and paging | Unverified |
| FSDP/ZeRO gradient shards | Unsupported: global norm requires distributed shard coordination |
| CUDA graphs | Unsupported by the eager host-synchronizing implementation |
| Compiled optimizer and arbitrary loader/worker restart | Unverified |

GlobalOptimManager is process-wide; avoid unrelated overrides when comparing optimizers. The package pins bnb rather than treating a minimum version as evidence that future native internals are compatible.

## Development and CI

```bash
python -m pip install -e '.[dev]'
python -m pytest -m 'not cuda'
python -m pytest --require-cuda
LION_RESULTS=/absolute/new/results/path python tools/cuda_integration.py
python -m build
python -m twine check --strict dist/*
ruff check .
ruff format --check .
```

`pytest` skips CUDA-marked tests when no GPU is available. `--require-cuda` fails before testing if CUDA is unavailable. The compatibility suite runs explicit legacy arithmetic; the other tests exercise optimized defaults, numerical defects, checkpoint policy and master weights.

[CI](https://github.com/julian-carpenter/LionPlus8Bit/actions/workflows/ci.yml) runs on push and pull request for Python 3.11 and 3.12, builds wheel and source distributions, installs the wheel, tests an import outside the checkout, and runs CPU tests and the example. It uploads test results and the built distributions. It does not publish to PyPI.

[CUDA validation](https://github.com/julian-carpenter/LionPlus8Bit/actions/workflows/cuda.yml) is manual. Register a trusted, current Linux x64 self-hosted runner with the `gpu` label and an installed CUDA PyTorch environment before dispatching it. This workflow uses real CUDA, runs the regression and independent integration tests, and uploads logs. Merely enabling the workflow does not provide a GPU runner. Private-repository badge visibility follows GitHub access; do not embed an access token in the README to make it public.

## References and license

- Chen et al., [Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675): Lion.
- Sfyraki and Wang, [Lions and Muons: Optimization via Stochastic Frank-Wolfe under Heavy-Tailed Noise](https://arxiv.org/abs/2506.04192): Lion+; Appendix C.2 is Algorithm 8 in v4 and Algorithm 7 in v1.
- Dettmers et al., [8-bit Optimizers via Block-wise Quantization](https://arxiv.org/abs/2110.02861): optimizer-state quantization.
- [bitsandbytes Lion API](https://huggingface.co/docs/bitsandbytes/en/reference/optim/lion) and [implementation](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/optim/lion.py).
- [PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html).

This package extends the native bnb optimizer; it is not an official implementation from the paper authors and is not affiliated with bitsandbytes. Code uses the [MIT license](LICENSE). Dependencies retain their own licenses.

## Adaptive clipping with AdaGC

Global clipping remains the default. To select AdaGC:

```python
optimizer = LionPlus8bit(
    model.parameters(),
    clipping="adagc",
    max_grad_norm=1.0,
    adagc_beta=0.99,
    adagc_factor=1.04,
    adagc_warmup_steps=100,
)
```

Global clipping scales every gradient together when their combined length exceeds a fixed limit. A spike in one layer can therefore shrink all layers' gradients. AdaGC gives each parameter tensor its own changing limit, based on that tensor's recent gradient lengths. A tensor here means a complete parameter such as a weight matrix or bias vector, rather than each scalar weight.

After warmup, for tensor `i` with gradient length `n`, previous history `gamma`, relative factor `r` and smoothing coefficient `b`:

```text
scale = min(1, r * gamma / n)
clipped_gradient = scale * gradient
new_gamma = b * gamma + (1 - b) * length(clipped_gradient)
```

Use scale one for a zero gradient. The old history determines clipping; the current spike cannot increase its own threshold before being clipped. Both Lion's sign decision and momentum update consume the clipped gradient. Decoupled weight decay keeps its existing behavior. This method comes from [AdaGC](https://arxiv.org/html/2502.11034v4); it differs from AGC methods that compare gradients with parameter lengths.

For example, two tensors with histories 2 and 10 have ceilings 2.08 and 10.4 with factor 1.04. Current lengths 10 and 5 cause only the first tensor to be clipped. Their next histories become 2.0008 and 9.95 with smoothing 0.99.

### Warmup and tuning

The first 100 completed optimizer steps use global clipping and track the smallest positive **clipped** length for each tensor. Step 101 switches to AdaGC. AMP-skipped calls and calls with no gradients do not advance this counter. This clipping warmup is separate from learning-rate warmup.

| Option | Default | Meaning |
| --- | --- | --- |
| `clipping` | `"global"` | Choose `"global"` or `"adagc"` |
| `max_grad_norm` | `4.0` | Global threshold; AdaGC uses it during warmup and cold-start bootstrap |
| `adagc_beta` | `0.99` | History smoothing; higher values adapt more slowly |
| `adagc_factor` | `1.04` | Ceiling relative to history; must be finite and at least one |
| `adagc_warmup_steps` | `100` | Number of completed global-clipping steps; zero is allowed |

The example explicitly uses the paper's absolute threshold 1.0; selecting AdaGC does not silently change the package's existing 4.0 default. After warmup there is no additional global cap. Start comparisons with the same learning rate, weight decay, batch size and schedule as your global-clipping run. AdaGC's beta is separate from Lion's two betas.

A larger factor clips less aggressively. With beta .99 and factor 1.04, a consistently clipped history grows by at most 0.04% per update. This can track sustained increases slowly, such as after a batch-size or loss-scale convention change. A factor of one prevents the ceiling from increasing. Evaluate validation quality and clipping behavior rather than choosing settings from training loss alone.

### Zero gradients and late activation

This package deliberately extends the paper's edge-case handling. Zero observations do not set a warmup minimum to zero, which would otherwise suppress all later positive gradients permanently. A tensor first activated after warmup receives one per-tensor absolute bootstrap clip and seeds its history from that result. The same rule applies when warmup is zero. `grad=None` does not change that tensor's history; a zero gradient can still cause momentum and weight-decay updates. Established histories decay on zero observations and become unseeded if they fall below FP32 work representability. No fixed denominator epsilon is added.

The linked Paddle and author PyTorch reference implementations differ in warmup counting, raw-versus-clipped warmup history, and epsilon. This package follows exactly 100 steps and clipped histories, with the documented zero recovery above. These details matter for reproducibility.

### AMP, memory and checkpoints

Use FP32 parameters with autocast and ordinary `scaler.step(optimizer)`. Clipping runs inside the optimizer after unscaling, once after all microbatches have accumulated. Avoid an additional external clipping call. Overflow skips leave AdaGC history and counters unchanged.

Actual FP16 parameters are different from FP16 autocast: ordinary PyTorch GradScaler refuses FP16 gradients. Internal FP32 master weights do not remove this restriction. Actual FP16 parameters require already unscaled gradients or a separately validated training integration. BF16 parameters can use the unscaled path; FP32 parameters with BF16 autocast are also appropriate.

History is stored in packed FP32 buffers, with rare FP64 exceptions for extreme values. Its payload scales with the number of tensors, not the number of individual weights. Masks, scratch vectors, allocator overhead, and optional FP32 master weights add memory beyond the history payload. Native bitsandbytes kernels take host scale values, so AdaGC batches coefficient readback rather than synchronizing separately for every parameter. A device-vector fused update kernel is not included.

Optimizer checkpoints save clipping policy, warmup progress and history. Loading old checkpoints restores global clipping. Reconstruct the same parameter-group order; same-shaped unnamed parameters swapped within a group cannot be identified automatically. Use a fresh optimizer when changing clipping strategy. On a native backend error after partial updates, restore a known checkpoint rather than continuing from partially updated state.

AdaGC changes the original globally clipped Lion+ algorithm, so the original convergence assumptions do not automatically apply. Distributed shards, flattened logical tensors, CUDA graph capture, and FP8 require separate integration work. See [AdaGC validation](docs/validation-adagc.md) for measured results and limitations.
