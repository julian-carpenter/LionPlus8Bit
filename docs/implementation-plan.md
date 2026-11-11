# LionPlus8Bit packaging plan

User-approved scope: private GitHub repository julian-carpenter/LionPlus8Bit; installable revised optimizer with optimized defaults; approachable theory and usage README; tests, CI and badge.

- [x] Build a src-layout package from the preserved revision, keeping native_gradient_scale=True and batched_norm=True by default. Export LionPlus8bit and version metadata; use package-relative imports.
- [x] Migrate exact replay tests to explicit compatibility mode; test optimized defaults separately, with CPU tests for hosted CI and CUDA regression/integration tests requiring real hardware.
- [x] Build wheel/sdist, test installed package outside checkout, run CPU and CUDA checks locally.
- [x] Write installation, math/theory, usage, AMP/accumulation, checkpoint, hyperparameter, memory/precision and validation documentation. Explain optimized-mode rounding and measured performance without claims beyond evidence.
- [x] Configure automatic CPU/build CI and a manual self-hosted CUDA workflow; badge points to automatic CI.
- [ ] Independent review; resolve material findings; create private repo and push; verify privacy, remote contents and actual CI outcome.

Keep original handoff, review evidence and revised snapshot unchanged. Do not publish datasets, checkpoints, local credentials or raw environment dumps. Current task authorizes repository creation and push. No PyPI publishing is requested. No writing-style service is available; documentation follows the requested audience and supplied prose constraints.
