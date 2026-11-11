# Validation notes for 0.1.0

The package defaults to `native_gradient_scale=True` and `batched_norm=True`. Legacy bitwise native replay is a separate compatibility configuration, with both flags set to false.

## Local installed-wheel checks

The build produced a wheel and source distribution; strict metadata/README checks passed. The wheel was installed outside the checkout and tested without adding `src` to the Python path.

- CPU-only selection: 28 tests passed, 15 CUDA tests deselected.
- CUDA-required selection: all 43 tests passed; CUDA availability was mandatory.
- Independent CUDA integration: all six groups passed, including matched native replay, state allocation boundaries, FP64 equation comparison, AMP overflow/skipping, accumulation and continuation across fresh processes.
- The CPU and CUDA synthetic regression examples both met their loss-reduction assertion and used actual uint8 weight momentum with FP32 bias momentum.
- Ruff lint and formatting checks passed.
- The source distribution includes the test fixtures, example and independent CUDA tool. The wheel contains the importable package, type marker and license.

Local hardware/software: Quadro RTX 6000 (SM75), Python 3.12.3, NVIDIA PyTorch 2.13 alpha with CUDA 13.3, bitsandbytes 0.50.2 selecting its CUDA 13.2 binary. The environment supports numerical BF16 execution but lacks native BF16 hardware acceleration.

Hosted CI passed for Python 3.11 and 3.12 on the initial complete package commit: [run 42998523496](https://github.com/julian-carpenter/LionPlus8Bit/actions/runs/42998523496).

The README CI status badge represents hosted CPU/build checks. A separate manual workflow needs a trusted self-hosted CUDA runner. Check the actual Actions run for hosted results; a local pass is not evidence that a remote job ran.

## Review changes

The packaging review identified and resolved two issues: an optimized-versus-legacy test had become a comparison of identical defaults, and the first source distribution omitted the shared pytest fixtures/example/CUDA tool. The final comparison uses an explicit legacy baseline and nonuniform clipped gradients. MANIFEST.in includes the documented source-distribution tools.

## Limits

This work does not add a general convergence or speed guarantee. A prior exploratory real-audio subset study used the previous arithmetic policy; its results are not a new representative application study of the package defaults. DDP, paging, compiled optimizers and arbitrary DataLoader-worker resume remain unverified. Gradient shards and CUDA graph capture remain unsupported. The repository contains synthetic tests, not the original application data or checkpoints.


The opt-in AdaGC implementation has separate [validation evidence and limitations](validation-adagc.md). The earlier results above describe the original global-clipping package.
