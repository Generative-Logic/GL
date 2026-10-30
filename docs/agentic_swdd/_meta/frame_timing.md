<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Frame timing `[DRAFT]`

## Contract

Every `main.py` run creates . `main.py` configures the ledger through `frame_timing.py` and exports its absolute path as `GL_FRAME_TIMING_PATH`; native subprocesses append to the same file. Each compact JSON object has exactly six fields: `stage`, `parent`, `batch`, `seconds`, `count`, and `excluded`.

`stage` and `parent` form an inclusive hierarchy. The `run.total` root contains startup, unit gates, and `python.pipeline`. Each native `<tag>` scope contains setup, the start-up stages of the prover call (`native.extent_purge`, `native.ce_filter`, `native.batch_preparation`; [D-533](../40_decisions.md#d-533)), the explicitly excluded prover and compressor boundaries, save/orchestration, raw proof generation, and telemetry. Raw proof generation is subdivided into GL-binary export, chapter-index scan, outer `buildStack` calls, serialization, reload release, and the remaining exclusive work.

Instrumentation is outside the statified burst paths. Python uses `frame_timing.py` and native code measures only parent-owned orchestration/export boundaries in `run_modes.cpp · fullRun` and `proof/proof_graph_writer.cpp · ProofGraphWriter::generateRawProofGraph`; no per-firing timer or allocation enters `performElemPhase1`, `performElemPhase2`, or `performElemPhase3`.

## Report

`.scripts/frame_timing_report.py <ledger>` ranks inclusive in-scope stages, prints excluded boundaries separately, and calculates residual time for every measured parent. The acceleration campaign requires each positive residual to be no more than one percent before the inventory is considered complete; a larger residual returns status 2 so the owning parent is subdivided instead of guessed about.

WSL measurement runs use `.scripts/wsl_snapshot.py`, which performs the direct working-tree copy, rebuilds, invokes `main.py --run-descriptor <tag>_<mode>`, and writes the paired SHA-256 proof-artifact manifest under the snapshot's `.debug/` directory. The workflow never clones.

## Weaknesses

### Known and tracked

- Wall-time scopes are intentionally coarse and low-overhead. Function-level attribution inside the selected largest stage comes from a targeted profiler run through `main.py`, not permanent hot-path timers.

### Not exercised by tests

- Cross-process append ordering is not an interface. Records are aggregated by stage, parent, and batch; consumers must never infer execution order from JSON-lines order.

## Concurrent finalization

`run_modes.py` and `_finish_full_graphs` start a child process that arranges the
incubator book (`python.processed_graph.incubator`) while the parent processes the
main graph (`python.processed_graph.main`); both record under `python.finalization`.
After joining it, the parent launches the incubator output process, which renders and
verifies incubator chapters while the parent writes the Lean export
(`python.lean_export.main`, the build excluded), renders main HTML, verifies main
chapters and finally waits for the background kernel check
(`python.lean_kernel_wait.main`: only the part of the build the parent waits for).
Each process owns its renderer state and output tree. The parent joins every child
and the kernel check on success and failure, retains the child logs, and combines
both verifier counts only after successful child execution. During the batches,
`python.incubator_theorem_list` (per tag) is the incubator's theorem-list write.

`python.finalization` is the elapsed wall-time scope under `python.pipeline`.
Its main and incubator child scopes overlap, so their sum is not elapsed time;
a negative child-sum residual at this parent represents overlap. Descendant stage
records retain their individual costs. Use the measured parent wall duration in
whole-run attribution. See `D-345`, `D-540` and `D-537`.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
