<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Memory investigation playbook

The maintainer invokes this workflow by name ("memory playbook"). It is the memory counterpart of the [runtime investigation playbook](rt_investigation_playbook.md) and is repeated iteratively: measure, report, suggest, wait for approval, implement, measure again. Diagnosis does not authorize a fix. Keep the task's root `ledger/` record current; its latest approval boundary governs implementation.

## What counts as a fix

In the maintainer's order of preference:

1. **Less memory needed at all** — fewer bytes for the same proof state: removed waste (page padding, freed pages parked inside held blocks, derived indexes sized for traffic that has gone), a denser representation, removed duplication.
2. **Less RAM or VRAM residency** — the same bytes, but on SSD while not in use (the deload), or a device reservation sized to demand. The SSD is treated as unlimited.

A fix may cost some runtime when the saving is worth it. Prefer non-functional changes; a slightly functional change is acceptable only when the architecture stays, the data representation alone differs, determinism holds and every theorem is kept.

## Establish the reference

1. Work on the branch the maintainer names, in its own worktree (Rule 34), with a Rule-32 ledger holding identity, machine (RAM, page file, GPU and VRAM) and the reference numbers.
2. Reuse the newest completed dump-free run whose tree matches:  carries the per-batch pool peaks and the census below. A saved run is a saved observation; say so when quoting it.
3. The instrumented run is a full `main.py --GPU --run-descriptor <descriptor>` through the queue controller, with the sacred dump targeted at an impossible key (Rule 14) and `MEM_MEASUREMENT` set to 1 on the investigation branch only (reset to 0 before any landing). A processor-route run is needed only when the question concerns the SSD pager's behavior.

## Read the census top-down

Every figure names its unit and denominator; never add figures that do not share one ([memory measurement](memory_measurement.md)).

1. **Process.** `[process-memory]` and the "Process memory" tables in : the operating system's lifetime peak working set and private bytes per batch process, the pool reservations (address space) and the pool bytes committed (the carve high-waters; a block is committed when first carved). `[process-phase]` gives, per prover step, the maximum working set at the step's seams and the resident memory outside the pools (an estimate: working set minus touched pool bytes). Start with the batch whose peak working set is highest.
2. **Pools.** `[pool-purpose]`: which consumer held the pool at its lifetime high-water (logic-block state, string scratch, container scratch, sealed pages, CUDA host arenas, logic-block shells, unattributed). `[pool-phase]`: in which prover step the main pool peaked and its purpose split then. A peak dominated by transient purposes (scratch, sealed, GPU host) is a sizing question of one step; a peak dominated by logic-block state is a content question.
3. **Logic-block state.** `.rt/_memory_<anchor>.log` (peak iteration of the batch): the physical split of the sampled logic blocks' pinned blocks into live pages (content, derived indexes, page padding), free pages parked inside held blocks, and other block bytes; then held pages per container and the per-structure content table. Compare a container's held pages with its content to find padding; compare index with content to find stale indexes.
4. **CUDA route.** `[gpu-memory]`: the device total, the fixed Phase 2 reservation, and the per-chunk host arenas (projection, projection shards, task, schedules, sealing, pinned staging).
5. **Geometry.** `[lb-geometry]`: the `Memory` shell size and logic-block store slots per block, which size the LB-body pool.
6. **Heap.** The per-step "Heap committed" and "Heap live (C++)" columns: live is what the program holds, committed minus live is allocator retention. For the owners, rerun with `GL_HEAP_PROFILE_KIB=256` in the pipeline step's environment and read the "Heap live by allocation site" tables (heap peak and largest barrier) — see [memory_measurement.md](memory_measurement.md#the-heap-census-live-bytes-and-allocation-sites).

## Classify and propose

For each large consumer, say which of these it is, with the measured bytes:

- **Waste** — bytes that hold nothing: page padding, whole-block granularity, parked free pages, oversized derived indexes, fixed-capacity reservations above demand.
- **Representation** — bytes per logical item larger than the item needs (fixed-width rows, strings where ids suffice).
- **Duplication** — the same content held twice (copies of rows in several columns, projection shards next to the merged projection).
- **Residency** — content not used in the current step that could live on SSD.
- **Legitimate growth** — more proof state; not a leak.

Propose fixes ranked by memory saved, first-choice fixes first. For each: the saving (measured or bounded from the census), the runtime risk, whether it is functional, the determinism argument, the tests and the acceptance gate. Implement nothing before approval (Rule 8 applies to any change of semantics, and pool, block, page or window sizes are capacity changes that need approval).

## After an approved fix

Rebuild, run the unit tests, then one full `--GPU` run with the same instrumentation. Report the before/after census for the affected batches, the theorem lists (`files/theorems/theorems.txt` byte comparison), the verifier result and the runtime of the affected batches. Determinism is checked as the task requires (two sequential runs when proof bytes could move). Record the result in the ledger and update the memory figures in the SwDD chapters the change touches.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
