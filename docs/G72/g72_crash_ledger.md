<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


<a id="g-72"></a>
# G-72 crash investigation ledger

Living task document (Rule 32). Branch created 2026-09-03 from `main` tip
. Owning reference: [G-72](../agentic_swdd/50_gotchas.md#g-72).

## Objective and acceptance

Find the cold, hidden defect behind the intermittent abort
`removeOwnerFromRun: the key is absent` (rule-owner removal, `prover.hpp`),
which fires in some runs and not in others on identical proof state, on the
shortcut and on the Peano main batch, without trying to reproduce it: read the
section, place traps that decide hypotheses structurally, run shortcut and full
pipelines with the traps armed, and let a firing (if any) land on instrumented
code. Acceptance: either the root cause is proved by a trap line and fixed with
the assert untouched, or every candidate mechanism in the section is
demonstrated absent by trap evidence and the remaining candidates are named.

## Hard constraints and approvals

- Sandbox branch: traps and measurements are allowed at will; production
 semantics untouched; every trap is `#if CRASH_TRAPS` (Rule 30 temporary,
 Rule 31 output to ). The permanent G-72 diagnostics stay.
- Do not hunt a reproduction; assume the abort will not fire.
- Never hard-kill a prover process that may hold a CUDA context (driver
 aftermath); long runs launch detached with a done-file monitor.

## Current measured state (2026-09-03)

| Run | Build | Result | Trap lines (beyond `[TRAP-INIT]`) |
|---|---|---|---|
| shortcut 1 | production, CUDA | 9217 checks / 0 failures, prover 42.1 s | none |
| full 1 | production, CUDA | verifier 16912 / 0, theorem artifacts byte-identical to `main` | 33 benign `[TRAP-SLOT]` (registry init walk); 7109 `[TRAP-LEDGER]` incubator noise from the trap's own gating asymmetry (fixed) |
| shortcut 2 | RT_MEASUREMENT + PHASE13_DEEP_TIMING, CUDA | 9217 / 0, prover 42.4 s | none |
| full 2 | RT_MEASUREMENT + PHASE13_DEEP_TIMING, CUDA | 141136 / 0 (Peano 16912 / 0), overall 286 s, theorem artifacts unchanged | 30 benign `[TRAP-SLOT]` (registry init walk); ledger covered all seven batches: zero lines |
| shortcut 3 (gate, fixes landed) | RT build, CUDA | 9217 / 0, prover 41.6 s; log content identical to shortcut 2 (per-burst swept / skipped counts equal, only wall times differ) | none |
| full 3 (gate, fixes landed) | RT build, CUDA | 141136 / 0, overall 287 s, artifacts unchanged; log content identical to full 2 apart from CE-filter wall times | 28 benign `[TRAP-SLOT]` (registry init walk) |
| full loop 1–10 (maintainer-directed, `run_full_10x.py`, sequential, traps armed) | RT build, CUDA | every run 141136 / 0, wall 276–288 s, artifacts unchanged, logs content-identical to each other after stripping wall times | none beyond the benign init walk in every run |

**Fixes landed (maintainer-approved, commit,
[D-337](../agentic_swdd/40_decisions.md#d-337)):**
the gen-scratch registry now has the reserved single-threaded slot
(`initGenScratchArenas(logicalCores + 1)`), and `claimAndLoadForWork`'s
`WorkerOwned` early return is phase-2-only — outside phase 2 the caller must
be the holder thread (`Memory::claimHolder`, assert). The `[TRAP-CLAIM]` trap
is superseded by that assert and removed; the other traps stay armed.

No firing during the instrumented campaign; the abort fired on the first
full run after the traps were removed (see Outcome). One full-run attempt was
hard-killed by the tool harness mid-Peano
(); it is not a data point.

## Section analysis (what the code says)

- The removal (`removeRuleFromHashMemory`) re-runs `addToHashMemory` under
 `RuleIndexOp::Remove`; the enumeration is a pure function of the rule text,
 the LB's `nameMap` / `ruleInterner` ids, and the permutation table. Single-
 threaded divergence between install and removal is therefore excluded by
 construction; every observed firing had the whole key and owner PRESENT and
 the `(argSet, NormKey)` edge absent in `remainingArgsOwners`.
- Timing dependence with per-LB deterministic state points at a cross-thread
 interaction. Two structural facts found:
 1. The gen-scratch registry has NO reserved slot
 (`initGenScratchArenas(logicalCores)` versus the string registry's
 `logicalCores + 1`), so every `g_currentCoreId == -1` fallback resolves
 to worker `logicalCores - 1`'s arena AND its rule-index staging pool. The
 unit tests exercise exactly that sharing between the main thread and a
 steward IO thread on gen slot 31 (`[TRAP-SLOT]` in the unit-test log).
 2. `MemorySteward::claimAndLoadForWork` returns immediately on
 `WorkerOwned` in EVERY phase, not only for phase-2 split siblings, so a
 second claimant would silently co-own an LB.
- In the crashing configuration the batch configs run `allow_ssd_deload=false`
 (resident-only steward): no prefetch loads, no evictions, so the IO threads
 never reach scratch; the CUDA projection shard threads use no scratch. Both
 structural facts are latent there — the traps test them anyway.
- All `linkChild` LB-creation sites live in `addTheoremToMemory` (seam); the
 `SimpleMapStore` single-threaded-writer invariant holds on paper.

## Trap set (commit, refined)

1. `[TRAP-SLOT]` — `ScratchArenaRegistry::forSlot` records thread, pool
 generation and thread kind per slot; SAME-WINDOW-HANDOVER = two threads on
 one slot inside one pool window; NONWORKER-TOUCH = a steward / io /
 projection thread on any slot.
2. `[TRAP-CLAIM]` — foreign `WorkerOwned` early return in phases 1 / 3 / 4.
3. `[TRAP-FLUSH]` — every staged (key, owner) read back through the derived
 index after each owner-section flush.
4. `[TRAP-LEDGER]` — per-LB ledger of every remaining-args edge hash the
 install staged into `overallHashMemory`; the removal's enumeration is
 checked against it, so a firing reports "never installed" versus "lost
 after install" with rule, scope and LB chain.
5. `[TRAP-ERASE]` — owner-less erase count audit.

## Hypotheses

- Shared reserved gen slot between a worker and a non-worker thread: NOT
 observed in production configuration (zero NONWORKER-TOUCH lines in four
 runs). Latent defect nonetheless (fact 1 above).
- Foreign claim / two writers on one LB: NOT observed (zero `[TRAP-CLAIM]`).
- Flush merge or owner-less erase losing an entry: NOT observed.
- Install / removal enumeration drift: NOT observed in any batch (zero ledger
 lines in full 2, incubator batches included).
- Still open: a mechanism outside the instrumented seams (phase-2 device
 path output, CUDA driver aftermath of hard-killed processes, an
 uninitialized read whose value depends on thread stack reuse).

## Outcome — ROOT CAUSE FOUND AND FIXED (2026-09-03)

The abort fired on the first full run after the traps were removed (Peano
main, hash burst 10, `__contradiction__(=[7,8]) <- (preorder[1,4,8,7]) <-
(preorder[1,4,7,8]) <- AnchorPeano`, rule
`(>[]!(=[u_it_0_lev_3_40,u_2])(existence3[u_1,u_it_0_lev_3_40,u_3]))`). The
permanent key-family dump () showed the
failing edge key as `[-842150451 × 10, 4, 42]` — `0xCDCDCDCD`, the arena
poison. Cause: `remArgsEdgeKeyInto` built the edge key on the slot's
gen-scratch arena and returned only a length; its four callers read the key
back at a cursor mark captured before the allocation, and `LbArena::alloc`
pads to the next block when a request would straddle the current block. On
a wrap the mark addressed the poisoned tail of the previous block, so the
install staged garbage (or the removal looked up garbage) and the rule's
true edge was later absent. The wrap position depends on the slot's cursor
history, i.e. on scheduling — the whole intermittency. Fix: the key is
built into a caller-owned stack buffer (`kMaxRemArgsEdgeKeyBytes`) at all
four sites (commit follows this entry), regression test
`rem_args_edge_key_independent_of_block_wrap`. Decision
`D-336`, invariant
`I-222`, G-72 gotcha updated. Evidence:
, .
Fix gate (production build, CUDA): shortcut 9217 / 0 and full 141136 / 0,
theorem artifacts unchanged, log content identical to the pre-fix runs.
The two earlier fixes stay as latent-door closures. **Document frozen
2026-09-03; the branch is squashed to `main`.**

## Exact next action

Gate the fix (unit tests, shortcut, full), then squash the branch to `main`
per the playbook (renumber the three pending slugs). Traps are removed
(Rule 30); RT gates are 0.

## Blockers

None. The launch path that survives the tool harness is
`scratchpad/launch_full.py` (Python `Popen` with job-breakaway flags).

## Evidence log

- 2026-09-03 17:18 build 1 (production + traps), 1624/1624 unit tests.
- 2026-09-03 17:21 shortcut 1 clean, traps silent.
- 2026-09-03 17:32 full 1 clean (detached relaunch), traps: benign only.
- 2026-09-03 build 2 (RT + traps, ledger gate removed), 1624/1624.
- 2026-09-03 shortcut 2 clean, traps silent.
- 2026-09-03 full 2 clean (141136 / 0, 286 s), traps: benign init walk only,
 ledger silent across all batches.
- 2026-09-03 fixes committed, build clean, 1625/1625 unit tests
 (new `gen_scratch_registry_has_reserved_slot`, extended claim handshake test).
- 2026-09-03 shortcut 3 gate clean, content-identical to shortcut 2; full 3 gate
 clean (141136 / 0, 287 s), content-identical to full 2; traps silent.
- 2026-09-03 ten sequential full runs (,
 ): all clean, traps silent. Tally on this
 branch: 15 full + 3 shortcut instrumented runs, zero firings.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
