<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# A20 campaign — state and runtime investigation

Append-only, newest last.

## 2026-08-05 — A20 proved 5/5 first run, squashed to main

A20 (positivity forms) entered the shortcut pool as rows 24–28 and proved 5/5 on the first run (2546 checks / 0
failures airtight). The two forward rows export as or-theorem compacts
(`a=0 ∨ 1≤a`, `a=0 ∨ a=1 ∨ 2≤a`) via `constructOrTheorem`'s
negated-premise fold. Squashed to main as. A false-negative
prove-status check (literal-row grep against `theorems.txt` instead of
the or base form `!(&!A!B...)`) was made and corrected in-session.

## 2026-08-05 — runtime explosion investigation 

**Symptom.** The A20 run took 352.5 s against the A18 baseline of
25.0 s — 14× — with the SAME 45 hash bursts. Burst 8 alone: 323.9 s
(92 % of the run).

**Offending LB (exact chain, innermost first):**

```text
(in2[rec0,9,3]) <- !(=[6,9]) <- !(=[2,9]) <- (in[9,1]) <- (AnchorFTA[1,2,3,4,5,6,7,8]) <- <root>
```

The induction successor-step block (hypothesis `9 = s(rec0)`) inside
the A20b-forward row's premise tree (`a ∈ N`, `a ≠ 0`, `a ≠ 1`,
goal `2 ≤ a`). A second block of the same shape under the A19 chain
(`(in2[rec1,9,3])`) is a minor contributor (~6 s phase-2 parts).

**Measurement funnel** (per-run logs in `.debug/`, evidence lines
quoted in the commit messages):

| Layer | Result |
|---|---|
| Per-burst `dt=` | burst 8 = 323.9 s; all other bursts ≤ 3.7 s |
| RTTracker (`RT_MEASUREMENT 1`, 120 s trigger) | `.rt/` EMPTY — no single phase-2 hashburst call over 120 s |
| `[RT-PHASES]` per-phase split | p1 = 0.015 s, p2 = 3.35 s, **p3 = 332.0 s** — all in the one LB above |
| `[RT-P3sec]` phase-3 split | stdProc = 343.8 s, reactToHypo ≈ 0, sanitize = 0.07 s |
| `[RT-SPsec]` standardProcessing split | absorb = 0.010 s, **equiApply = 330.4 s**, cleanup = 0.012 s, discharge = 0.002 s, fillMailOut ≈ 0 |
| `[RT-AEC]` applyEquiClasses counters | secs = 330.4, rounds = 2, deltaClasses0 = 16, stmts 1,522 → 9,352, pairs1 = 149,632, pairs2 = 24,051 |

**Root cause.** `applyEquiClasses` (Pass 1: delta classes ×
ALL registry statements from index 0) inside the induction
successor-step scope. At burst 8 the block holds 16 delta equivalence
classes — the successor hypothesis `9 = s(rec0)` plus the equalities
the induction-hypothesis instantiation and the corpus successor laws
mint around it (`s(rec0) = rec0 + 1`, witness forms) — and
back-applies them across the inherited statement population. The
rewrites are fertile: ~7,830 new statements commit in this single call
(registry 1,522 → 9,352, 6.1×), and each commit pays the full
kernel-chain cost (Site-F ancestor dedup, level merge, origin history,
per-commit discharge probe, admission-map equi-hooks). The measured
average is ≈ 42 ms per committed rewrite / ≈ 1.9 ms per pair visit —
the pair count (174 k) is modest; the per-commit constant dominates.

**Classification: architecture / scale behavior, not a coding bug.**
Every stage does its contracted work; the blow-up is the product of
(a) an induction scheduled on the premise variable of a row whose
scope already carries two negated equalities and the numeral 2,
(b) Pass-1 back-application semantics (a changed class re-applies to
the whole registry), and (c) a high per-commit constant. The goal
itself proved regardless — the explosion is collateral machinery, not
a stall.

**Open sub-question.** The ≈ 42 ms per-commit constant is itself
suspiciously high and was not decomposed further (would need
per-commit sub-timers inside `applyEquivalenceClass` /
`addExprToMemoryBlock`). Candidate contributors, unverified: the
per-commit `dischargeToBeProved` probe, or-convergence
`mergeProducts`, admission-map equi-hooks walking cold blobs.

**Candidate repair directions (maintainer decision — none
implemented, Rule 8):**

1. Decompose the per-commit constant first (one more trap layer) —
 if one contributor dominates, a targeted fix may recover most of
 the 330 s without touching equi-class semantics.
2. Scope-gate Pass-1 back-application inside recursion/induction
 sub-blocks (e.g. per-class start indices seeded to class-creation
 time rather than 0 in these scopes) — an equivalence-class
 semantics change, needs explicit approval and soundness argument.
3. Unfair-advantage route: none needed for provability (A20 proved);
 only relevant if a pool lemma could pre-empt the induction
 scheduling on rows of this shape.

**Rule-30 status.** All instrumentation lives ONLY; `main` is untouched. Remaining active traps
on the branch, reported per Rule 30 (diagnosis-only session, no fix
yet): `RT_MEASUREMENT 1` (parameters.hpp), `disable_lb_split: true`
(ConfigFTA.json), `[RT-TEMP]`-tagged timers in `prover.cpp::proveKernel`
/ `performElemPhase3` and `prover.hpp::standardProcessing` /
`applyEquiClasses`, and the `rtChainOf` helper. Remove all of them
(and restore the two switches) before any production run from this
branch or at fix time.

**Run evidence files:**  (baseline
explosion), `run_a18_first_shortcut.log` (25 s baseline),
`run_a20_rt_instrumented.log` (empty `.rt/`),
`run_a20_rt_phases.log`, `run_a20_rt_p3sec.log`,
`run_a20_rt_spsec.log`, `run_a20_rt_aec.log`. All instrumented runs
reproduced 2546 / 0 airtight with the full 28-row pool — observation
did not perturb results.

## 2026-08-05 — per-commit constant decomposed (maintainer direction 1)

Sixth trap layer (`run_a20_rt_decomp.log`): thread-local cumulative
sub-timers split the hot `applyEquiClasses` call as

```text
secs=333.9  helpers=0.012s  apply=333.8s (map=188.7s deposit=144.4s)  merge=0.023s
```

- **Exonerated:** the five per-class admission/rejected equi-hooks
 (0.012 s) and the or-convergence `mergeProducts` walk (0.023 s) —
 the earlier open sub-question's candidate list (discharge probe,
 or-merge, admission hooks) is measured dead.
- **Deposit walk = 144.4 s ≈ 18 ms per surviving commit** — the
 maintainer's statification-byproduct hypothesis is CONFIRMED for
 this half: the walk is where the per-commit cold-container work
 lives (the `equality1` origin `addOriginEncoded` sorted in-place RMW
 into the `TypedColdBlobMap`, `intStatementLevelsMap` CSR
 `assignSetRanges` splices, registry append bookkeeping). A
 per-commit cost-vs-registry-size trace (not yet run) would separate
 O(container) splice growth from a flat constant.
- **Mapping/substitution phase = 188.7 s ≈ 1.1 ms averaged per pair
 visit**, paid before any commit, surviving or not — the candidate-
 construction engine (member-occurrence scan, parallel-substitution
 row building, level merges, per-call scratch copies). Not previously
 on the suspect list; now the single largest bucket. Its interior has
 not been decomposed further.

Updated repair-direction picture: the two real targets are (1) the
mapping phase's per-visit constant (interior split pending) and
(2) the deposit walk's per-commit cold-container RMW cost (splice-
growth trace pending). Both are measurement-ready with one more trap
layer each; neither is a semantics change.

## 2026-08-05 — content inspection: the burst is a refuted-branch grind

Maintainer-ordered pivot from timing to CONTENT ("maybe there is some
crazy shit going on" — correct). The sacred hashburst dump was
retargeted (Rule 14, explicit consent) at the hot LB, and a temporary
`[RT-AEC-CLASS]` trap printed the 16 delta classes' decoded members on
the slow call. Findings (`run_a20_rt_dump.log`, trace snapshot
`hashburst_a20_hotlb.txt`, 43 MB / 490k lines):

1. **The burst grinds inside an already-refuted or-branch.** The
 delta classes live at
 `main_boundary_ordis_(or1[rec,2,1,3])_((=[rec,2]))` — the
 `rec = 0` disjunct of a predecessor case split on the witness
 `rec`. Under the block's hypothesis `9 = s(rec)`, that assumption
 forces `9 = 1`, contradicting the row's main-scope premise
 `9 ≠ 1` — the branch is FALSE. The contradiction surfaces as an
 equality-class COLLAPSE instead of a fast retirement: the dominant
 class is `{0, 1, 9, 9_copy, rec, x2, x6, it_0_lev_0_30,
 it_0_lev_4_24, it_0_lev_4_8}` — zero, one, and the induction
 variable all "equal". Dead-branch retirement
 (`drainDeadOrBranches`) runs in phase 3's sanitize block AFTER the
 `standardProcessing` that hosts `applyEquiClasses`, so the whole
 330-s apply happens inside the doomed branch first; from burst 9
 the LB is back to ~2.5 s (branch retired). 79,662 of the trace's
 490k lines belong to that one scope.
2. **Eleven duplicate delta-class entries.** Of the 16 delta entries,
 ~11 are the SAME (validity, member set) class snapshot repeated —
 each triggers a FULL Pass-1 back-application over the registry
 from index 0, producing only downstream-dedup'd duplicates.
3. **Factorial mapping enumeration on the collapsed class.** With a
 10-member class, `enumerateEqClassRewrites`'s `allMappingsAna`
 lookup iterates factorially many injective mappings per pair (cap
 shape `(5,12)` → up to 95,040 per pair visit) — the degenerate
 class maximizes exactly this.

The three layers multiply into the 324–334 s. The proof output stays
sound and airtight throughout — the grind is pure waste inside a
vacuous branch, which also explains why the timing levers (inlining
−20 s; arg-id cache + reduceEqClassIds hoist — measured NO gain, kept
for cleanliness) could not touch the bulk: the work itself is
semantically pointless, not inefficiently executed.

**Candidate repairs (maintainer decision, none implemented — all
Rule 8):**
1. **Distinct-numerals tripwire:** a class merge that equates two
 distinct anchor numerals (0 = 1) proves the scope's premise set
 inconsistent — flag the branch refuted at the merge site and
 suppress equivalence-class application at/below that scope; the
 existing retirement drain reaps it at end of burst. Kills
 essentially the whole 330 s.
2. **Dedup `changedClassesThisStep`** by (validity, member set) per
 step — kills the ~11× redundant sweeps independent of the
 contradiction (likely a much broader win than this one LB).
3. (Considered, riskier) run dead-branch staging BEFORE the phase-3
 apply — an I-63 ordering change; the tripwire in 1 achieves the
 effect without reordering the seam.

**Lever-2 A/B for the record:** apply 316.9 s (map 191.8 / deposit
124.4) vs lever-1 313.1 s (map 189.2 / deposit 123.1) — no gain,
byte-clean; the cache-coherence assert also caught (and the fix
handles) mid-pass registry compaction shifting row indexes
(owner-identity keyed cache, commit c6d3057d).

## 2026-08-05 — ETW sampled profile (Windows WPR, maintainer-run capture)

Maintainer captured an elevated `wpr -start CPU -filemode` trace around
a full `main.py --shortcut` run; decoded non-elevated with
`xperf -a profile -detail` against the Release PDB
(, 5.4 GB — delete when done; decoded table
). Prover process total: 486.6 s CPU.
Top functions (per-function self time; MSVC ICF folds identical
template instantiations, so individual template names are
representative of their FAMILY, not exact):

```text
51.8 s  10.6%  enumerateEqClassRewrites<...applyEquivalenceClass lambda>   (substitution enumeration core)
51.7 s  10.6%  PagedVector<char>::dataVidOf                                (paged-address resolve per access)
48.8 s  10.0%  HashMap<BytesKeyStore,EmptyValueStore>::lookup              (cold byte-key set probes)
46.3 s   9.5%  HashMap<BytesKeyStore,BlobCsrValueStore>::lookup            (cold blob-map probes)
43.7 s   9.0%  vcruntime140.dll (memset/memcpy/memmove intrinsics)
31.7 s   6.5%  PagedVector<char>::contiguousRun
29.8 s   6.1%  PagedVector<AmSortRow-family>::operator[]                   (ICF-folded paged element access)
23.3 s   4.8%  PagedHashIndex::slotPtr
18.8 s   3.9%  BytesKeyStore::decodeAt
10.8 s   2.2%  BytesKeyStore::equalStored
10.7 s   2.2%  makeIntNormalizedKeyFromEncoded
```

**Refined root cause.** The maintainer's statification hypothesis is
confirmed with a sharper mechanism than blob-splice growth: ~55–60 %
of the prover's CPU is **cold-substrate primitives** — byte-key hash
probes (~100 s across the two HashMap families + PagedHashIndex),
paged-address resolution and byte access (`dataVidOf` +
`contiguousRun` + element access ≈ 115 s), byte decode/compare/copy
(~75 s incl. the vcruntime intrinsics) — invoked per member × per
statement × per occurrence by the equi-class machinery
(`enumerateEqClassRewrites` is the one big named algorithm frame at
51.8 s). The per-call constants are small; the CALL COUNT (~10⁸-scale)
inside the burst-8 apply is what detonates. Notably `dataVidOf`
appearing as a standalone 51.7 s frame means the per-access resolve is
NOT inlined on this build — a pure build/inlining observation worth
checking before any structural work.

**Candidate levers (maintainer decision, none implemented):**
1. Inline/flatten the hot paged-access primitives (`dataVidOf`,
 `contiguousRun`, element `operator[]`) — build-level, no semantics.
2. Reduce cold-primitive call count in the equi-class apply path:
 cache decoded class members / statement bytes across the pair loop
 instead of re-probing cold containers per pair (the mapping loop
 re-resolves the same bytes per visit).
3. The scope-gating idea from the first report (limit Pass-1
 back-application in induction sub-blocks) remains valid as a
 work-avoidance lever independent of the substrate cost.

## 2026-08-06 — dead-scope equi-skip experiment: explosion eliminated, 331.6 s → 55.8 s

Maintainer-ordered trap experiment ("trap debug if dead branch can be
disabled before explosion"): can the refuted branch be cut off BEFORE
the phase-3 apply grinds inside it? Answer: **yes, with machinery the
prover already has.**

**Timeline verification first (trace analysis, no new semantics).**
The 43 MB hot-LB trace pinned the order of events inside the slow
burst-8 call (dump pair #11):

1. At ENTRY the refuted `rec = 0` branch scope holds only its
 assumption; the refutation `!(=[rec,2])` exists ONLY at the
 sibling `existence11` branch scope — not an ancestor — so the
 branch is not yet retirable and the drain was right not to fire.
2. During that burst's OWN phase 2, ordinary main-scope hash rules
 derive `!(=[2,rec])` at `v=main` twice over, from pure main-scope
 premises (`(in2[2,6,3])` = 1 = s(0) with `(preorder[1,4,6,rec])`;
 and the corpus row `1≤a → a≠0`) — fully independent of the dead
 branch.
3. Phase-3 `standardProcessing` absorbs those deposits in the 7 ms
 `absorb` step BEFORE `applyEquiClasses`. So when the 294-s apply
 starts, the branch's asserted disjunct is already refuted at an
 ancestor — the exact staging predicate `ordisMerge` uses for
 `pendingDeadOrBranches`. The registry of dead scopes the repair
 needs ALREADY EXISTS; the equi part just never consults it, and
 the drain runs after `standardProcessing`.

**The `[RT-TEMP]` gate.** In `applyEquiClasses`' Pass-1 delta loop
(RT branch only, Rule 30): for each delta class, walk its validity
chain; at an `_ordis_` disintegration scope run the ordisMerge
refutation probe (negate(asserted disjunct) `known` at scope or
ancestor, non-minting); on a hit print `[RT-DEADSKIP]`, mint the
branch into `pendingDeadOrBranches` (the drain's own inbox, so the
existing end-of-burst retirement reaps it), and skip the class.

**Result (`run_a20_rt_deadskip.log`, trace snapshot
`hashburst_a20_deadskip.txt`):**

| Metric | ungated (dump run) | gated |
|---|---|---|
| overall runtime | 331.6 s | **55.8 s** |
| burst 8 `dt` | 296.9 s | 4.6 s |
| burst 8 p3 | 294.0 s | 0.26 s |
| verifier | 2546 / 0 airtight | 2546 / 0 airtight |

The gate fired exactly on target: the 14 dead-scope delta entries
skipped in both Pass-1 rounds at burst 8; the two benign `x6`-cohort
classes (k=3, k=9) untouched. Result identity is strong: verifier
tallies byte-identical per category, statement counts at EVERY dump
boundary identical between runs (EXIT #11: 1263 both; EXIT #12: 1544
and `toBeProved=0` both), final branch-scope populations identical
(44 rows surviving `existence11` scope, 2-row residue at the wiped
`rec=0` scope, both runs). Only origin-history VOLUME differs (7,068
vs 14,542 at EXIT #11) — the dead-branch `equality1` history lines
that were never minted; Rule-16 process documentation, not proof
input. Two additional burst-10 `[RT-DEADSKIP]` hits at the
`existence11` scope were semantically vacuous in both runs: the
cohort had already resolved, so `drainDeadOrBranches` takes its
defined `orDisjunctCount == 0` skip either way (the exact known-flag
producer for that scope's negation was not pinned down — known-only
rows are invisible in the registry dump — but the same state exists
in both runs, boundary-count-identical through #12).

**Conclusion.** The maintainer's proposed repair — contra detection +
a dead-scope shortcut registry consulted by the equi part — is
CONFIRMED feasible and sufficient: it kills the whole explosion, with
`pendingDeadOrBranches` as the registry and the existing refutation
predicate as the detector, no seam reorder needed. Production shape
(this per-class gate, and/or the merge-site distinct-numerals
tripwire, and/or `changedClassesThisStep` dedup) remains the
maintainer's Rule-8 decision; the gate stays `[RT-TEMP]` until then.

## 2026-08-06 — production: gate + dedup + lever 1 landed, battery airtight

Maintainer decisions: the per-class gate goes production; the
`changedClassesThisStep` dedup goes production; the distinct-numerals
tripwire is REJECTED (where such collapses matter, provided lemmas
will cover them); lever 1 rides along. New branch from `main`; the RT branch's traps stay behind.

**Commits.**
1. — dead-scope gate: `refutedOrBranchAtOrAbove`
 (D-262, I-181),
 consulted by `applyEquiClasses` Pass 1; a skip always stages the
 branch for the same burst's drain.
2. + — keep-last delta dedup
 (D-261). The FIRST form compared entry pairs
 (O(n²) views per round); the full-run battery caught it grinding
 the Peano incubator (`AnchorIncubator3` bursts 42–124 s, stage
 unfinished after 26 minutes, run stopped by the maintainer; memory
 pressure ruled out — ~1 GiB of 12.9 GiB, zero pager evictions).
 Replaced by ONE linear descending pre-pass per round
 (`fillDeltaKeepLastFlags`, packed `(validity, members)` keys into a
 per-round scratch set) — identical verdicts.
3. — lever 1 cherry-pick (GL_FORCEINLINE on the paged leaf
 primitives).

**Verification battery (all pass).**

| Gate | Result |
|---|---|
| unit tests | 1379/1379 (1377 + gate test + dedup test) |
| shortcut run | 28/28 rows, 2546 / 0 airtight, **40.3 s** (burst 8: 6.3 s with the LB split active) |
| determinism pair | second shortcut run 38.9 s; `theorems.txt` + `global_theorem_list.txt` byte-identical, verifier tallies identical |
| full standard run | 1425 s, **137,436 checks / 0 failures airtight**; both Gauss summation CE copies proved; IncubatorPeano1 stage max burst 19.98 s vs the July-30 baseline's 20.41 s (the grind is gone); heavy-burst profile in-family (max 283 s vs baseline-era 264 s, main Peano batch) |

**Open comparison note.** No current-era `main` full-run artifact
exists on disk, so the strict theorem-SET diff against `main` was not
run; the incubator save counts differ from the July-30 baseline only
at era level (1036/156/204/49/8 vs 1036/157/210/49/8 — the baseline
predates the A15 clean-start and the A16–A20 corpus). A one-off
`main` full run would close it exactly if wanted.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
