<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Runtime ideas — the open acceleration levers `[PLAN]`

> Status: opened 2026-08-21, after the
> mandatory-containment fold (`D-298`) took the
> FTA shortcut from 972 s of prover time to 132 s. This file collects the
> levers that survived that campaign's triage, in the order the maintainer
> ranked them. Ideas 1 and 2 have since shipped; idea 3 is open. Two candidates
> were considered and REJECTED — they are recorded at the end so nobody
> re-proposes them.

Where the time goes now, re-measured on the gated shortcut run after ideas 1
and 2 shipped (`.rt/_aggregate_FTA.log`, 2026-08-22, 1095.4 burst-seconds
attributed, 6657 bursts; pipeline 122.4 s of which prover 117.7 s, re-timed on a
healthy host after a power-supply fault was found and fixed — the `iter` counts
are exact, the seconds columns come from the faulty window and are indicative of
shares only):

| Region | burst-s | % | calls | per call |
|---|---|---|---|---|
| Folded batch 2+3 → grow search | 887.4 | 81.0 | 6504 | 136 ms |
| Batch 1 → pairing merge | 129.0 | 11.8 | 4693 | 27 ms |
| Batch 1 → grow search | 64.2 | 5.9 | 4698 | 14 ms |
| Batch 2+3 → filter + sort | 5.9 | 0.5 | 6504 | — |
| Request firing (`STATIC_REQGEN_FIRE_EVAL`) | 2.1 | 0.19 | 294,686 fired | — |
| Batches 4 + 5, every section together | 2.2 | 0.2 | — | — |

Outside the prover, the same run used to spend 36.4 s in
`native.raw.build_stack` — 20% of the pipeline. Idea 2 below measured it, then
closed it: the stage is now **0.017 s**, and the pipeline is prover-bound
again.

---

## 1. Build each grow candidate from its predecessor *(SHIPPED 2026-08-21)*

**Where.** The grow DFS inside `generateEncodedRequestsStatic`.

**The waste.** Inside one stack pop the first k premises never change — only the
appended one walks the positions. The loop nevertheless redid all of it per
position: the premise pointer array, the normalized key, and the request gates.

**Measured before.** An ETW sampled profile of the gated shortcut run put
`gl_quick.exe` at 1559.6 core-seconds, of which `generateEncodedRequestsStatic`
is ~92% with `performElem2` as its only caller. Split of the generator's own
inclusive time (butterfly parent percent; the core-second column cross-checks
against the flat per-function self times and agrees within 2%):

| per-node stage | % of generator | ≈ core-s | ≈ % of process |
|---|---:|---:|---:|
| the two `ownerKeyAccepts` probes | 56.87 | 815 | 52 |
| `requestGatesPass` (self 89 + `productsOfRecursionIds.contains` 74) | 13.93 | 200 | 13 |
| `makeIntNormalizedKeyFromEncoded` | 10.41 | 146 | 9.3 |
| `candPtrs` refill (`IntStmtView::operator[]`) | 7.57 | 109 | 7.0 |
| phase-4 merge `stable_sort` (batch 1 only — idea 3 deletes it) | 3.98 | 57 | 3.7 |
| generator self | 2.60 | 46 | 3.0 |
| scope fold (`ancContains` + `deeperOf`) | 2.17 | 31 | 2.0 |
| filter + emit | 0.70 | 10 | 0.6 |

The three prefix rebuilds are the middle rows: ~455 core-seconds, ~29% of prover
CPU. The owner probes are the other 57% and this idea does not touch them — a
whole key is hashed and byte-compared once per node however it was built.

**What shipped.** Three commits, each verified
byte-identical against the branch baseline:

| step | what moved | prover on the shortcut |
|---|---|---:|
| baseline | — | 151.55 s |
| premise pointers hoisted out of the position loop | pure code motion | 139.24 s |
| normalized key resumed from the prefix | `NormKeyBuildState` | 134.19 s |
| request gates folded one premise at a time | `RequestGateState` | 129.13 s |

**−22.4 s, −14.8%**, with the proof artifacts and the verifier's 9635 checks /
0 failures unchanged at every step.

**Why it is sound.** Both the key and the gates are monotone under a suffix
append. The key renumbers arguments in first-appearance order, so the key over
`e0..e(n-1)` is a strict PREFIX of the key over `e0..e(n-1),en` — appending can
only mint slots the prefix had not used. The gates are order-free conjunctions,
so each reduces to a summary a premise folds into: the hypothesis scope, a
saturating count of the distinct scopes not covered by the main-scope anchor
exemption, and the distinct secondary variables. The prefix is prepared once per
pop; each position rewinds the counters and folds in its own premise.

**How it is factored.** One per-expression body each
(`appendExprToIntNormalizedKey`, `foldExprIntoRequestGates`), used by both the
whole-candidate entry and the resume path, so the two cannot drift.
`makeIntNormalizedKeyFromEncoded` and `requestGatesPass` keep their signatures
and delegate. Contract: [I-202](30_invariants.md#i-202).
Decision: [D-296](40_decisions.md#d-296).

**Rejected on the way.** Storing the prefix state in the DFS stack item so a
child inherits its parent's. It saves nothing extra — the per-pop rebuild
already amortizes over the whole position loop — and would grow each frontier
entry from about 50 bytes to over two kilobytes on the byte-bump arena tier.

**What is left here.** The recovered share is depth-dependent: the per-node
saving is roughly (count − 1)/count of the prefix walk, zero at two premises and
75% at four, and a pruned DFS's node population is heaviest at the shallow
levels. Measured recovery was about 40% of the addressable region. Squeezing the
rest means fewer nodes, not cheaper nodes — and candidate restriction per node
is already rejected below.

---

## 2. The chapter walker — `buildStack` *(measured and CLOSED 2026-08-21)*

**Where.** `visualizer.cpp`, stage `native.raw_proof`. Was 36.4 s of a 181.8 s
gated shortcut pipeline, essentially all of it `native.raw.build_stack`;
serialization 0.03 s, LB release 2.2 µs, the whole Python tail ~1 s.

**Outcome.** Two commits — the `covered` insertion journal (lever A) and the
discard-aware fallback (lever B) — took the stage from 36.4 s to 0.017 s and
the walk from 4,803,594 recursive calls to 5,501, with every chapter file
byte-identical and the verifier unchanged at 9635 checks / 0 failures. The
5,000,000-call abort tripwire went from 96% consumed to 0.11%. Levers C and D
were never needed and stay on the shelf.

**Why it surfaced now.** It did not grow. It was always this size and used to
be a few percent of a prover-dominated run; with the prover at 139 s it is 20%.

### What the dissection found

A one-off dissection split the walk: one RT tracker per top-level chapter
walk, leaf-only `BS_*` interior scopes around each kind of work, plus a
per-chapter counter row carrying calls / depth / candidates / backtracks /
peak `covered`. It was investigation instrumentation and was removed once the
fix shipped (Rule 30); the numbers below are its record. Measured on the
instrumented FTA shortcut (39.7 s; the instrumentation itself cost ~3.3 s at
this call count, so the uninstrumented figure is 36.4 s):

| Section | as found | after A | after A+B | hits (as found → now) |
|---|---:|---:|---:|---|
| `BS_COVERED_SNAPSHOT` | 15.25 | *(retired)* | — | 4,472,837 → — |
| `EXPORT_CHAPTER_WALK` (self) | 11.24 | 2.52 | ~0 | 101 → 101 |
| `BS_LIFT_DEPS` | 2.27 | 2.02 | ~0 | 5,110,074 → 5,510 |
| `BS_ORIGIN_DECODE` | 2.24 | 1.99 | ~0 | 4,795,288 → 5,407 |
| `BS_COVERED_INSERT` | 1.87 | 1.66 | ~0 | 9,755,657 → 8,815 |
| `BS_PATH_SET` | 1.78 | 1.44 | ~0 | 9,590,576 → 10,814 |
| `BS_COVERED_RESTORE` | 0.91 | 1.12 | ~0 | 157,378 → 370 |
| `BS_LIFT_ENTRY` | 1.22 | 0.99 | ~0 | 4,803,594 → 5,501 |
| `BS_EMIT_ROW` | 0.91 | 0.79 | ~0 | 4,775,995 → 5,179 |
| `BS_ORIGIN_PROBE` | 0.77 | 0.72 | ~0 | 4,803,594 → 5,501 |
| `BS_PATH_PROBE` | 0.79 | 0.71 | ~0 | 4,806,916 → 5,468 |
| `BS_USAGE_REACHES` | 0.18 | 0.16 | ~0 | 304,090 → 253 |
| `BS_CONTRA_FALLBACK` | 0.14 | 0.13 | ~0 | 303,170 → 424 |
| `BS_ENTRY_RELOAD` | 0.12 | 0.11 | ~0 | 4,803,594 → 5,501 |
| **total** | **39.71** | **14.36** | **0.017** | |

Read the two steps differently. **A changed no hit count at all** — the walk
performed exactly the same search, it just stopped copying a set to do it (the
self-time collapse from 11.24 s to 2.52 s is the copy's *destructor*, which ran
outside the measured scope). **B changed nothing but hit counts** — the search
itself shrank by three orders of magnitude while emitting the same 2,925 rows.

Three findings, in order of consequence.

**One chapter is 98.4% of the export, and it enumerates a complete binary
tree.** Of 101 top-level walks, chapter 91 — a `check_induction_condition`
walk — takes 39.2 s; the other 100 take 0.65 s together. Its counters are exact
powers of two, which settles the diagnosis on its own:

| counter | value | |
|---|---:|---|
| recursive calls | 4,605,957 | |
| candidates tried | 4,605,953 | ~1.00 per call |
| cyclic skips | 131,072 | = 2^17 |
| backtracks | 131,071 | = 2^17 − 1 |
| fallback emits | 262,143 | = 2^18 − 1 |
| max depth | 38 | |
| peak `covered` | 311 | |
| **rows emitted** | **297** | |

4.6 M calls to produce 297 rows over 311 distinct nodes — each node re-derived
on the order of 15,000 times. The powers of two say the walk explores a
complete binary decision tree exhaustively rather than converging: at each of
~18 levels a choice fails, and nothing remembers that the subtree below it was
already settled. That is chronological backtracking with no memory — rolling
`covered` back to its snapshot discards every derivation the failed candidate's
subtree had established, and the next attempt re-derives all of it.

The counters are deterministic: two consecutive instrumented runs produced
identical values in every column (only wall-clock differs).

**The export is at 96% of its own abort cap.** `buildStack` carries a hard
5,000,000-call tripwire that `std::abort`s the process. The run makes
4,803,594. This is now a liveness hazard, not only a speed one: one more FTA
rung of this shape aborts the export.

**Cold-LB reloads are zero.** `BS_ENTRY_RELOAD` is 0.12 s across 4.8 M calls
and the CSV records zero reloads for every chapter. The export never left RAM.

### What the numbers rule out

**Chapter-level parallelism is dead.** One chapter is 98.4% of the work, so
distributing chapters over the worker pool takes the export from 39.7 s to
39.1 s. The three blockers it would need solved first — the usage graph
mutating between chapters ([I-190](30_invariants.md#i-190)), the shared reload
sink, LB residency counting — buy nothing. Do not build it.

**Loading the working set from SSD in one sweep is dead for this run.** There
is nothing to load: reloads measure zero. The idea stays open for the full
pipeline, where `native.raw.release` is 0.15 s on `IncubatorPeano1` — still
small.

**The path-cycle filter and the admissibility probe are not the cost.**
`BS_PATH_SET` + `BS_PATH_PROBE` together are 2.6 s (6.5%) and
`BS_USAGE_REACHES` is 0.18 s. Packing their keys to integers, or precomputing
the usage graph's transitive closure per chapter, is real but small — a second
pass, not a first one.

### What the numbers point at

**A. The `covered` snapshot — SHIPPED, 39.7 s to 14.4 s.** Every candidate
deep-copied the whole `covered` set before trying, then copied it back on
failure: 4.47 M snapshots against 157 k restores, so **97% of the copies were
never used**, and the cost was charged twice because the snapshot's destructor
ran outside the measured scope. Replaced by an insertion journal
(`g_coveredJournal`, a thread-local vector beside `g_buildStackPath`): a
candidate records the journal length as its mark and rollback erases exactly
the keys inserted since. Exact because `covered` is insert-only inside the walk
and membership-only outside it — no call site iterates it — so the rollback
reproduces the copy's set. Chapters byte-identical, verifier unchanged.

**B. Discarded fallback work — SHIPPED, 14.4 s to 0.017 s.** In the dominant
chapter `cyclic_skips` was 2^17 and `backtracks` 2^17 − 1 — every backtrack one
cycle-filtered candidate — while `fallback_emits` was 2^18 − 1, twice as many.
The dominant shape: a node's only candidate is rejected by the cycle filter;
the node then runs the whole last-resort `front` emit *and recurses over its
entire dependency subtree*; and finally returns false — at which point the
caller's `stack.resize` and `covered` rollback throw all of it away.

`buildStack` now carries a defaulted `rowsSurviveOnFailure` and skips that
block when it is clear. Exactly one call site passes `false` — the
candidate-loop recursion, the only caller that truncates on a false verdict.
The two contradiction tail calls and the fallback's own dependency walk inherit
it. Output-preserving because the skipped block always ended in `return false`
and its three side effects (rows, `covered`, path set) are respectively
truncated by the parent, rolled back by the parent's journal, and inserted-then-
erased symmetrically inside the block.

Measured: 4,803,594 recursive calls → 5,501; backtracks 157,378 → 370; fallback
emits 303,158 → 42; the same 2,925 rows across the same 101 walks; chapters
byte-identical. See [D-297](40_decisions.md#d-297)
and [I-201](30_invariants.md#i-201).

**C. Remembering derivations — NOT NEEDED, kept on the shelf.** Record a
node's successful derivation (its emitted row run) the first time and replay it
on later visits, guarded by the two conditions that make replay exact: the
recorded node set must not intersect the current `covered` (or rows would be
dropped) and must not intersect `g_buildStackPath` (or the cycle filter would
have rejected it). Guard fails → fall back to the real search. The classic
memoized-search-with-validity-guard. After B the walk makes 5,501 calls for
2,925 rows, so there is nothing left for it to memoize; revisit only if a
future ladder rung brings the re-derivation back.

**D. Replacing the search — NOT NEEDED, and the most expensive option.**
Settle nodes bottom-up in nondecreasing cost order so a settled node's chosen
origin cites only already-settled nodes — acyclic by construction, no path
stack, no backtracking. Recorded with its history: tried on 2026-05-08 and
rejected ([D-51](40_decisions.md#d-51)) because every chapter goal carried a
zero-cost `theorem` origin that won over the real derivation, and because
contradiction records still lived at ancestor LBs. D-51 itself removed both
causes, so that rejection no longer stands on its original reasoning — but
chapter content changes unless the cost tie-break reproduces the D-49
preference order, so it costs a verifier re-baseline that A, B and C do not.
With the stage at 0.017 s there is no case for paying it.

---

## 2b. The export outside the shortcut — the canonical loader *(measured and fixed 2026-08-21)*

Idea 2's two fixes bought the FTA shortcut everything and the full pipeline
almost nothing (21.61 s → 21.22 s across seven batches). The shortcut's cost
was ONE chapter exploding through the last-resort fallback; the incubator
batches are a different shape entirely and needed their own root cause.

**The shape.** IncubatorPeano1 exports 946 chapters in ~11 s, IncubatorGauss1
191 in ~10 s, and every other batch is under 0.02 s. The search wastes
nothing there — 1.0–1.1 recursive calls per emitted row, 2 backtracks and 0
fallback emits across 35 k calls — so it is pure per-call constant factor, not
exploration.

**Where it went.** A leaf-only dissection put 99.2–99.8% of both batches in one
place: the post-loop contradiction-LB probe, specifically
`contradictionLbHoldsRecord`'s `ensureLoadedForRead` on a discharged
`__contradiction__` twin. 14.2 ms per reload on IncubatorPeano1 (837 of them),
80.1 ms on IncubatorGauss1 (132). The string builds and the ancestor-chain
`findChild` walk beside it measured 0.004 s and 0.000 s. Every probe RESOLVES
(837/837, 132/132), so the reload is a load the chapter genuinely needs — there
are no failed probes to memoize away.

**Root cause, two layers.**

1. *Why the export reloads at all.* Origin history lives on the per-LB
 DELOADABLE arena ([I-121](30_invariants.md#i-121)). By export time every
 contributing LB is discharged and cold, so the walk must reconstitute a
 whole LB to read one map — and the per-theorem release ([G-53](50_gotchas.md#g-53))
 throws it away before the next chapter. Contrast `intToBeProved`, put on a
 never-deloaded persistent pool ([I-108](30_invariants.md#i-108)) precisely
 so it stays readable while the arena is cold.
2. *Why each reload was so expensive.* `lb_deload.cpp::loadLbMemory` read the
 image with an `istreambuf_iterator<char>` pair into an unreserved
 `std::vector<char>` — a virtual call per byte plus geometric reallocation.
 It was the only such pattern in `memory_infra/`; the raw loader beside it
 always used `seekg` + a block `read`.

**Fixed.** The canonical loader now sizes the file and does one block read.
Export 19.86 s → **8.55 s**; per reload ~14.2 → ~3.6 ms (Peano) and ~80 → ~41 ms
(Gauss). Byte-transparent — same bytes, read differently — and it is on the
prover's own canonical reload path too, though this machine's ±90 s prover
variance across runs is far too wide to show that.

A second, smaller fix rides with it: the read-only door no longer re-derives
the four `ReverseArgsIndex` membership indexes, which only request generation
reads ([I-204](30_invariants.md#i-204)).
Worth 19.86 → 21.22 s measured on its own, i.e. ~6% — correct, but not the
lever. It was proposed on the theory that the rebuild dominated the reload;
the measurement refuted that and pointed at the reader instead.

**What is left.** ~8.5 s, and it is now the genuine container restore: the
walk touches exactly two members of a reloaded LB (`exprOriginMap`,
`originInterner`, tags 451–454 plus the interner) while `loadLbMemory`
restores every tag — LbMemory's own 0–50, four `HashMemory` bands spanning
51–450, cold mail at 455/505, 655, 705. The v3 file already carries a per-tag
directory, so a FILTERED load is an extension of the existing loader rather
than a new format. The obstacle is [I-111](30_invariants.md#i-111): `LbArena`
models residency as one binary flag with asserts on every `resolve`, and a
partially-loaded LB is a state it does not have. That is a maintainer decision,
not a patch.

Chapter-level parallelism is also viable HERE, unlike on the shortcut — no walk
exceeds 1.6% of its batch and a greedy 32-way schedule spans 0.33 s / 0.37 s
against the pre-fix 21 s. It stays unbuilt: it would run 32 concurrent reloads
against one static pool, so the pool budget and LB residency counting become
the hard part, and the reader fix took most of the prize without touching any
of that.

---

## 3. Remove the surviving obligatory stumps — batches 1, 4, 5 *(SHIPPED 2026-08-22, )*

COMPLETE. All three batches are converted, the obligatory-stump machinery is
deleted, and so are the minus-one / minus-two owner maps
([D-295](40_decisions.md#d-295)). **Measured, on a clean like-for-like pair after the power fix:** shortcut prover
117.664 s -> 102.048 s, **-13.3%**; overall pipeline 122.393 -> 107.565 s.
`theorems.txt` byte-identical (one md5 across all nine captures, branch-point run
included). The per-step timings taken during the fault are discarded; only the
endpoints are measured cleanly. Corroboration that survives a drifting clock:
680,490 (base candidate x stump) merge pairings per run eliminated, grow
enumeration unchanged within 0.7%. Batches 4 and 5 pay a wider filter now that
`alsoAcceptFullKeys` is true for them, which is inside the 13.3%.

**Where.** `performElem2` (`prover.cpp`), the three remaining
`stumpLen = 1` call sites, and the merge phase inside
`generateEncodedRequestsStatic` that only they still reach.

**The state after step 1.** The fold retired the obligatory stump for batches
2 and 3 only. Batch 1 still builds one-element stumps over
`intLocalEncodedStatements` and still runs the pairing merge — **9.4% of burst
time, the second-largest region in the profile**, and it is the same raw
(base candidate × stump) cross product the fold was built to kill. Batches 4
and 5 are the same shape at negligible cost.

**Why they were not folded into 2+3.** Each reads a DIFFERENT rule registry —
`workingMemory` (batch 1), `localHashMemory` (batch 4),
`localHashMemoryDelta` (batch 5) — and `intMemory` drives `maxKeyLength`, the
target map selection, the filter's subkey gate, and every owner probe. They can
never share one call with the `overallHashMemory` batches. Each keeps its own
invocation; only its stump becomes a term.

**The conversion.** Mechanical, one call site at a time:

| Batch | today | becomes |
|---|---|---|
| 1 | stump over `intLocalEncodedStatements` | one term: `{ intLocalEncodedStatements }` |
| 4 | stump over `intExternalStatements` | one term: `{ intExternalStatements }` |
| 5 | stump over `intEncodedStatements` | one term: `{ intEncodedStatements }` — the whole universe, so the term is **vacuous** and the call becomes a plain full-depth grow |

**What it unlocks.** With no `stumpLen > 0` caller left, four things retire
together: the seed phase, the merge phase, `makeMandatoryEncodedStatementLists1Static`,
and — once `produceExpressionStumps`' terminal-retention probe drops them —
`normalizedEncodedSubkeysMinusOne` and `normalizedEncodedSubkeysMinusTwo`,
including their construction in `addToHashMemory` and their deload facets.
[I-155](30_invariants.md#i-155) then collapses from two mandatory-ingredient
modes to one.

**Watch out for.** Batch 1's own grow currently stops at `maxKeyLength - 1`;
converted, it runs to full depth. Batch 5's vacuous term means the containment
prune cannot fire at all there, so its node count is whatever the full lattice
over the broad universe costs — measure that call before and after rather than
assuming the batch stays negligible.

---

## Rejected — do not re-propose

**Candidate restriction per grow node** (skip positions that provably cannot
extend the current prefix). Rejected by the maintainer 2026-08-21. The reason
is structural: GL generates requests for **all rules at once**, never for one
rule, so "what can extend this prefix" is the union over every rule owning the
prefix's subkey. At depth 0 that is every name; at depth 1 on a common premise
the owner set is large and the union approaches the whole universe; and the set
only grows as rules are installed. The lever is weakest exactly where the nodes
are.

**One chapter per worker in the export** (fan the chapter walks over the
thread pool). Rejected by measurement 2026-08-21, not by design: a single
`check_induction_condition` walk is 98.4% of the export, so perfect
parallelism over 101 chapters takes 39.7 s to 39.1 s. Revisit only if a future
profile shows the export time spread across many chapters — re-adding the
dissection is the check.

**Preloading the export's LB working set from SSD in one sweep.** Rejected by
measurement 2026-08-21 for the shortcut run: the export performs zero reloads
and `BS_ENTRY_RELOAD` totals 0.12 s across 4.8 M calls. Nothing to prefetch.
Still open in principle for the full pipeline, where the release stage is
0.15 s on `IncubatorPeano1` — also small.

**Collapsing the equivalence-class duality in the filtered universe** (one row
per class instead of the theorem term plus one representative). Rejected by the
maintainer 2026-08-21: *"we need them to be seen as diff for important key
firings."* The two rows normalize to the same key and are indistinguishable to
the growth and record probes, but they bind differently at firing and that
difference is load-bearing. The duality is deliberate.

---

## Weaknesses

### Known & tracked

- Idea 1 shipped and its win is measured end to end (151.55 s -> 129.13 s of
 prover time), but the per-step attribution is wall clock on one machine, not
 an RT-scope ledger, so the individual -12.3 / -5.1 / -5.1 splits carry that
 machine's run-to-run variance. The depth distribution of visited nodes is
 still unmeasured, so why the recovery landed near 40% of the addressable
 region rather than higher is inferred from the (count-1)/count shape, not
 observed. A per-depth node counter in the DFS would settle both.
- Idea 1's soundness rests on the rewind being total, which is written down as
 [I-202](30_invariants.md#i-202)
 and pinned by two twin unit tests. A future field added to either carried
 record and not added to the rewind would break it silently — wrong keys and
 wrong gate verdicts, not a crash.
- Idea 2's numbers come from single gated shortcut runs, and the profile it
 found turned on ONE theorem's origin graph. A different ladder rung could
 have a different shape. The abort-cap headroom (now 0.11% consumed) is the
 number to watch; re-deriving the per-chapter counters means re-adding the
 dissection, which is not standing instrumentation.
- Lever B's correctness rests on the candidate-loop rollback being TOTAL. That
 is written down as [I-201](30_invariants.md#i-201),
 but it is an invariant a plausible future optimisation (a memo keyed on a
 node a failed candidate visited, say) would break silently — the symptom
 would be a chapter quietly missing its degraded rows, not a crash.

### Not exercised by tests

- No regression test pins the export stage's runtime, so idea 2's win (or a
 future regression) would be visible only in the frame-timing ledger's
 `native.raw.build_stack` row.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
