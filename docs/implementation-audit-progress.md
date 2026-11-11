# Audit implementation ledger

Specification: docs/audits/2026-11-11-fresh-audit.md (F01–F10 and further recommendations).
Branch: fix/fresh-audit-findings; original base 64205431d701415de6aaa8355bcc536f81626cd5.
Original evidence remains under docs/evidence/audit-2026-11-11.

Implemented: all F01–F10; per-tensor guards and selective norms, bounded blockwise strict momentum updates, optimizer-owned checkpoint policies/hooks/migration, lazy module precision, safety preflight, active-set LRU, telemetry, expanded AMP/replay/native benchmarks, balanced repeated campaign, minimum-version CI, README and corrective report.

Verified:172 CUDA-required tests plus 15 subtests; 113 installed-wheel CPU tests plus 15 subtests; six independent CUDA integration groups; six fresh-process CPU AMP resume splits; twelve precision resumes; independent scale and overflow oracles; nine matched 1,100-update application arms. Application validation metrics exactly match original counterparts; quality remains inconclusive at the 2% margin. Fresh independent review found no blocker. Packaging and static checks pass. All 72 balanced benchmark process runs passed (1,800 timed updates; zero overflow skips).

Rulings: preserve default-stream dependencies; strict persistence remains per affected tensor until a numerically justified exit exists. Legacy checkpoint strict flags remain conservative. Reject initialized live dtype/device transitions, cross-device inactive groups, and overlapping parameter/gradient storage before updates; disjoint contiguous views remain valid. Moment scratch is chunk bounded; gradient clipping/strided flatten and FP64 recovery may still allocate gradient-sized work. Distributed/paged/graph/compile support and device-only native ABI/fused kernels require separate backend designs and validation, and remain documented boundaries rather than enabled speculative functionality.

Final evidence and benchmark results: docs/validation-audit-fixes.md and docs/evidence/audit-fixes-2026-11-11/README.md.
