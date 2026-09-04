<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Part C shortcut RT investigation — diagnosis, 2026-08-20

**Question.** The shortcut prover runtime rose to ~890 s (~3x over the ~210–260 s
B10/B11-era baselines). Suspects: C12 (pool row 60) or C14 (rows 61/65). Find the
culprit LBs, dissect the worst one's runtime by phase and content. Diagnosis only —
no fixes.

**Evidence files.**  (instrumented timing run,
prover 883 s),  (dump run, prover 876 s),
 (540 MB sacred-dump snapshot, 22 ENTRY/EXIT
pairs of the worst LB). Instrumentation: temporary per-LB per-phase wall-time
ledger in `proveKernel` (`[BURST-PH]` / `[RT-LB]` lines, Rule 30 — still in tree).

## Culprit LBs (full chains, run-total CPU seconds, phase 2 unless noted)

| CPU-s | max/burst | LB (leaf ← … ← root) | owner row |
|---|---|---|---|
| 2610 | 528 | `(in2[rec2,9,3]) ← (preorder[1,4,6,9]) ← (preorder[1,5,11,13]) ← (in3[9,12,13,5]) ← (in3[9,10,11,5]) ← AnchorFTA` | **row 61 = C14-reverse** |
| 2367 | 726 | `(in2[rec0,10,3]) ← (preorder[1,4,11,13]) ← (in3[12,9,13,5]) ← (in3[10,9,11,5]) ← (preorder[1,4,6,9]) ← AnchorFTA` | row 42 (old) |
| 650 | 330 | `(in2[rec0,10,3]) ← (in3[9,10,11,5]) ← (preorder[1,4,6,10]) ← (preorder[1,4,7,9]) ← AnchorFTA` | row 32 (old) |
| 536+363+264 | — | rows 63/64 step blocks + `__contradiction__!(=[9,10])` twin | **C13a/b** |
| 393 | — | `(in2[rec0,10,3])` under row 43's chain | row 43 (old, B10 monster) |
| 245 | — | `(in2[rec1,9,3])` under row 60's chain | **C12 — cheap** |

Phase split is unambiguous: **phase 2 (hashburst) owns >99 %** of every heavy
burst (e.g. burst 7: ph2 = 2056 CPU-s vs ph1 = 3.1 s, finalize = 0.07 s,
ph3 = 3.4 s). Within phase 2 the `work=` submatch tallies identify the
request-generation grow search (same signature the B10 investigation measured at
99.7 % in `GENERATE_ENCODED_REQUESTS_STATIC`). Splitting works for these LBs
(their buckets parallelize; wall ≈ total CPU / cores), so wall-time relief must
come from reducing total work, not from better dealing.

## Root cause (trap-proven on the worst LB, C14-reverse's rec2/9 step block)

**Not an `_ordis_` explosion.** Zero `_ordis_` occurrences in the 540 MB trace;
the LB carries its goal (`toBeProved = (preorder[1,5,10,12])`, i.e. d | e) from
first to last burst. The ordis hypothesis is refuted for this LB.

The registry trajectory (ENTRY/EXIT headers): statements 1 → 3,669, rules
3 → 342 (→ 779 late, broadcast arrivals), origin rows → 62 k+. At the heavy
bursts ~60 % of statements are `existence1[1,a,b,op]` closure rows. Those are
premise-inert (no rule cites an existence compact as premise — maintainer
confirmation 2026-08-20); the damage is what they cascade into:

1. **Totality pairing.** The corpus rule
 `(>[1](in[1,u_1])(>[2](in[2,u_1])(existence1[u_1,1,2,u_5])))` (and its `u_4`
 twin) fires over every ordered pair of `in[x,1]`-typed terms, for both + and ·.
2. **Witness minting.** Each `existence1[1,a,b,op]` head disintegrates and mints
 an `it_` witness: `(in3[a,b,it,op])` + `(in[it,1])` (origin chains verbatim in
 the trace).
3. **Re-entry.** The fresh `in[it,1]` re-enters the totality rule next burst —
 the term set grows and pairing is quadratic. It converges only when the I-171
 generation cap bites: 35 terms (2,636 of 4,052 closure-slot occupations are
 witnesses), statements plateau ~3,100 around LB-burst 14–16 — and the plateau
 population is re-scanned by the grow search every remaining burst.
4. **The premise-live fuel.** At EXIT #12: 283 `in3` rows (188 multiplicative,
 84 % witness-citing) and ~530 `preorder` occurrences — **divisibility rows
 (`preorder[1,5,…]`, 303) now match/exceed the ≤ rows (`preorder[1,4,…]`,
 228)**. Part C doubled the relational population over the same witness set and
 added 15 divisibility rules; 40 of 330 rules chain ≥ 2 `in3` premises. The
 grow search over (multi-`in3` ∧ preorder) rules × this population is the
 submatch work (per-LB 50–300 k per burst, grid totals 1.5–2.4 M).
5. **Duplicate witnesses.** ≥ 23 (a,b,op) products carry 2+ distinct witnesses at
 `main`; 11·rec had three (`it_0_lev_5_132` levels {-1}, `it_0_lev_5_3128`
 {0,1,4,5}, `it_1_lev_5_658` {0,1,5}). Uniqueness rules then derive equalities
 between the duplicates and equality1 rewrites churn the registry (62 k origin
 rows) — the B10 "witness-pair subsumption" lever, now measured live.

**Why row 61 (C14-reverse) is the worst.** Its premise set is the densest
multiplicative seed in the pool: two multiplication premises sharing the
induction variable (9·10 = 11, 9·12 = 13, induction on 9 = c), plus a
divisibility premise **between the two products** (11 | 13, which disintegrates
into a third multiplication witness 11·f = 13). Every induction context re-runs
the cascade against `rec` (the trace shows 11·rec, 13·rec, … witnesses minted per
stage). C12's step block, by contrast, costs only 245 CPU-s.

**Why the old rows inflated (the other half of the 3x).** Rows 42/32/43/40/44's
step blocks run the same totality-pairing machinery over their own witnesses;
Part C's broadcast divisibility theorems doubled the per-term relational row
count and rule count grid-wide, so every multiplication-rich step block pays
roughly double-to-triple its B-era work (row 42: 726 CPU-s in one burst,
work 596 k).

## Verdict

Architecture-level cost concentration, not a coding bug: undirected pairwise
totality closure over minted witnesses × the (new) doubled relation family.
C14-reverse (row 61) + C13a/b are the direct new-row cost (~4.5 k CPU-s);
the grid-wide relational doubling explains the old rows' inflation.

## Candidate repairs (maintainer's pick — nothing implemented)

1. **Witness subsumption at the existence-disintegration mint:** before minting a
 fresh `it_` for `a op b`, probe for an existing `(in3[a,b,x,op])` row at the
 scope and reuse `x`. Kills the duplicate-witness equality churn; deterministic
 and local to the mint site.
2. **Demand-gate the totality pairing:** fire the closure rule only when some
 rule premise demands `in3[a,b,?,op]` (the C8 demand-key machinery is the
 in-tree precedent), or exclude witness×witness pairs from the undirected
 pairing (they are 65 % of closure-slot occupations).
3. **Tighter generation cap for closure-only witnesses** (I-171 family): terms
 whose only role is being a closure product could stop re-entering the pairing
 a generation earlier, cutting the quadratic transient.
4. **Pool-side restatement of row 61** (helper lemma or a pre-split guard-variant
 route as used for C13) — permission-gated per the unfair-advantage precedent.

## Row bisection (2026-08-20, maintainer-directed) — CULPRIT: row 62 = C15

Cumulative pool bisection over the six rows appended
(rows 60–65), one `main.py --shortcut` run per step on the instrumented
main-tip-algorithm binary, early-stop at 2x base (driver
, results ,
logs ):

| pool | prover RT |
|---|---|
| rows 1–59 (pre-`FTA_C_rest`) | 280.9 s |
| +60 (C12) | 294.0 s |
| +61 (C14-reverse) | 336.5 s |
| **+62 (C15, product of divisors)** | **740.9 s — culprit, driver stopped** |

Findings that supersede the LB-level attribution above:

- The base pool on current code runs at B-era speed — the 3x is **not
 code-caused** (`FTA_C_rest`'s or-elimination / tier / drain changes are
 exonerated).
- C12 and C14-reverse are individually near-flat (+13 s / +43 s).
- **C15 (`d | a ∧ e | b ⟹ d·e | a·b`) adds +404 s by APPLICATION** (the
 maintainer's mechanism call): its own proof and LBs are cheap; once
 broadcast, the rule's four premises — two divisibility premises spanning
 FOUR independent variables plus two products — force the grow search to
 pair every divisibility row against every divisibility row (~300 × ~300 in
 a hot block) before the product premises discriminate: the widest rule
 fan-out in the pool. The hosts are other rows' witness-rich induction step
 blocks (row 42's rec0/10, row 61's rec2/9 — the "old-row inflation" the
 first analysis measured but misattributed to Part C broadly).
- Caveat: cumulative bisection — C15's jump is measured with C12 + C14-rev
 present; a base+62-only run would separate pure effect from interaction.
- Open: whether the `1≤c`-shaped premises unleash an `_ordis_` cohort inside
 the host blocks (maintainer hypothesis) — the row-61 rec2/9 block is
 dump-proven ordis-free, but the C15-era hosts (e.g. row 42's rec0/10) are
 uncleared; needs a dump retarget in the +62 configuration.

## Mechanism closed (2026-08-20, originMap + population evidence)

The maintainer's "difficult to satisfy" intuition is CORRECT and the earlier
firing-flood / self-feeding-monoid stories are REFUTED as the RT driver:

- **C15 barely fires.** The worst host LB's cumulative `exprOriginMap` (final
 EXIT, 70,080 entries) contains 9,563 `implication` history lines; only
 **105 cite C15's rule (64 distinct heads)** — on par with row 59 (156) and
 C14-forward (106). No conclusion flood.
- **The proof state barely grows.** +61-pool vs +62-pool runs: per-burst
 `total_exprs` near-identical (burst 7: 78,474 in BOTH; peak delta +3.4 %)
 while the same bursts' wall time triples (36.8→100.3 s, 29.3→113.3 s,
 56.3→124.5 s). Same statements, 3x the time.
- **Therefore the +404 s is pure request-generation search interior:** C15's
 key carries two divisibility premises with no shared variable, so the grow
 search enumerates every ordered pair of the ~300 divisibility rows in a
 witness-rich block, every burst, with almost no completions. The price is
 paid trying, not firing. Hosts: the same step blocks in both runs (row 61's
 rec2/9: 397→2,464 CPU-s; row 42's rec0/10: 1,231→2,122; row 32's: 265→689);
 C15's own LBs never enter the top five.

Repair ranking under the confirmed mechanism (maintainer-corrected 2026-08-20):

A grow-search join-reorder is REJECTED (maintainer): the canonical name-sorted
subkey order IS the hash engine's shared index structure (owner-set maps keyed
by sorted subkey prefixes, stumps naming nodes of the one canonical
enumeration, the split partition on top — I-155/I-156/I-77); a per-rule
linkage-aware order would fracture it. The engine cannot be extended like
that. Surviving options:

1. **Quarantine row 62** (drop, or proved-not-broadcast tier with per-batch
 lift) — immediate ~400 s relief, zero prover change. Supporting fact: C15
 is derivable pointwise with linked premises at every step (C14-forward
 twice + C5 transitivity: d|a ⟹ e·d|e·a, e|b ⟹ a·e|a·b, chain
 d·e | a·e | a·b), so Part E consumers may never need the wide rule
 registered.
2. **Shrink the divisibility-row population in hot blocks** (the N of the N²
 interior; ~80 % of the rows are witness-borne): demand-driven closure
 minting, witness subsumption at the mint. Same machinery, fewer rows.

## Quantified attribution (2026-08-20, maintainer-directed counters)

Maintainer challenged two points; both measured on the 62-row culprit pool
(, prover 852 s incl. instrumentation):

- **Proliferation of C15's fired heads: NONE.** The 64 heads have 292 direct
 children (max 19); the transitive descendant cone converges after 9
 generations at 1,628 of 70,080 originMap expressions (2.3 %), and the +61
 run reaches near-identical statement totals without C15 — the cone members
 mostly exist anyway via other derivation routes.
- **C15's submatch share in the worst host LB: 57.3 % of the entire
 candidate-enumeration interior.** Per-burst `[C15-SUB]` counters at the
 `preEvaluateFromEncoded` probe: 3.036 billion attempts over the run, of
 which 1.739 billion contain an unlinked positive divisibility pair —
 servable by C15 alone in this pool. Peak bursts: 638.8 M attempts / 72.9 %
 unlinked; 568.1 M / 67.6 %; 565.2 M / 54.5 %. Accepted among the unlinked:
 11,623 — an accept rate of 7×10⁻⁶. Onset signature: the six bursts before
 C15's rule circulates show ~0 unlinked attempts, then 130 k → 686 k →
 1.1 M → 129.8 M in one burst.

Verdict, final: C15 detonates the shortcut by forcing the request
generator to try ~1.7 billion divisibility-row pairings that almost never
complete — "difficult to satisfy" IS the cost. No firing flood, no
proliferation, no ordis involvement anywhere in the traced host.

## Hit-log dissection + the alias anomaly (2026-08-20, maintainer-directed)

Maintainer asked for the actual successful C15 submatches and per-phase batch
shares. Results (62-row pool, ,
 — 11,623 hits with full decoded premise
multisets):

- **Batch-wide phase shares** (all LBs, all iterations, [BURST-PH] sums;
 identical in the 65- and 62-row runs): **phase 2 = 98.9 %**, phase 3 =
 0.7–0.8 %, phase 1 = 0.37 %, finalize = 0.01 %.
- **Divisibility-row producers in the host** (final EXIT originMap): C7 242,
 row 59 (b | a·b) 176, C5 159, C15-self 110, C3 61, C1 49, C2 14, plus 342
 equality1 rewrites. **C14-reverse's rule produces ZERO rows** (its head is
 the block's own goal); C14-forward is absent from this pool. But C14-reverse
 IS nutritious as SEED: of 479 distinct divisibility heads, 71 (15 %) have no
 implication line — the premise deposits (11 | 13 and its rec-stage variants,
 hypo-scope goal assumptions, diagonals) — and the closure rules amplify that
 seed ~7x into the other 85 %.
- **Hit anatomy:** 7,107 distinct combinations of 11,623 hits (39 % identical
 re-enumerations across bursts); 4-premise interior prefixes + 5-premise full
 keys; no diagonal x | x members; X | 0 fodder in 11 % of hits.
- **THE ANOMALY — equality-known witness aliases never collapse.** The host
 knows SEVEN term-alias equalities at main (both orientations): 2=zero-alias,
 6=one-alias, 9/10/11/12 = c/d/n/e aliases, and rec = int_lev_4_1. Both sides
 keep parallel row populations (rec 14 divisibility rows, its alias 14); the
 hits' top facts are alias artifacts (`int_lev_4_1 | rec` 613 hits,
 `it_0_lev_4_158 | 9` 535); the zero-ALIAS form carries 269 hits while the
 canonical numeral form carries none. Rewriting the registry under just these
 seven equalities: distinct divisibility rows **246 → 132 (46 % duplicate)**,
 all statements **4,852 → 2,667 (45 % duplicate)**. The pair interior pays
 the SQUARE: ~3.5x more C15 pairings than a canonical census. Same family as
 the C6-era door-side canonicalization gap (addStatement eqfilter swallowing
 class-first non-canonical arrivals) and the B10 witness-subsumption lever —
 now measured as a 3.5x multiplier inside the worst host.

Repair implication: an alias-collapse fix (door-side canonicalization /
witness subsumption at the mint) cuts C15's interior ~3.5x in this LB without
touching C15 or the engine's enumeration order — compounding with (not
replacing) the row-62 quarantine option.

## The externalStatements leak (named finding, 2026-08-20 — FIX IMPLEMENTED on: `canonicalizeMailArrival` at the absorb drain before both consumers, D-301)

**One sentence:** every mailed statement is consumed twice — a class-disciplined
registry offer AND a raw, canonicalization-free staging into
`intExternalStatements` that request generation pairs directly — so
non-canonical equivalence-class members mailed from an ancestor enter the
phase-2 hashburst verbatim for one burst per arrival, bypassing the registry's
one-representative collapse.

**Class-discipline context (confirmed, burst-12 census of the worst host).**
The registry enforces the maintainer's participation rule — per class, the
theorem term plus ONE representative (the `int_` member when present, else the
decoded-lex-first `it_`; I-88 canonical-first-of-tier) — in 11 of 14 main-scope
classes exactly: e.g. the rec class {rec 404 rows, int_lev_4_1 304 rows, three
`it_` members at 8/8/8 defining rows with 0 hits}. The measured "45 % duplicate
census" is therefore mostly the SANCTIONED term+representative duality, not
runaway aliasing. The violations are mail-borne:

- class [8]: `it_0_lev_4_186` — **0 registry rows, 34 C15 hits** in burst 12;
- class [9]: `it_0_lev_5_252` — **5 registry rows, 209 C15 hits**.

**Mechanism (code sites, file + symbol).** The absorb door's mail-statements
drain (in [`prover.hpp`](../../GL_Quick_VS/GL_Quick/src/prover.hpp), the
`status == 3` "External-mail staging" branch inside the sorted per-row drain)
pushes `encodeExpression(statement, vName, nameMap)` verbatim into
`memoryBlock.intExternalStatements` — the in-tree comment near the
`disintegrateExpr2` caller states the container "holds the raw mailIn
expressions 1:1". No equivalence-class application on this path. The readers
are `performElem2`'s request-generation batches 3 and 4
(`IntStmtView(body.intExternalStatements)` as the mail side); the container is
cleared per burst in the phase-2 region, so the leak window is exactly the
arrival burst. The parallel registry offer (`addExprToMemoryBlock` via the
`addStatement` door) IS class-filtered — including the C6-era eqfilter gap
that swallows a class-first non-canonical arrival WITHOUT a canonical
substitute. Two faces of one missing piece: **mail arrivals are never
class-canonicalized at the door.**

**Fix direction for the follow-up session (Rule 8 — maintainer has signalled
intent to fix):** canonicalize each mailed statement against the recipient's
equivalence classes once, at the absorb drain, BEFORE both consumers — the
registry offer and the external staging — substituting the canonical form
rather than swallowing (this also closes the C6 Gauss-fold loss). Care points:
scope-correct class selection (exact validity + ancestors), the drain's sorted
deposit boundary and determinism (I-84 / I-102), levels-row semantics
unchanged (I-136 / I-182), Rule 16 history lines (`equality1`) for rewritten
arrivals, and I-26's main-only assert on the staging branch stays.

**Acceptance sketch:** re-run the burst hit log — zero hits citing class
members with no registry row; C6's lost Gauss fold recovers; shortcut and full
gates airtight; determinism pair byte-identical.

**Evidence files:**  (per-burst hits),
 (registry + class censuses; burst-12
sections), .

## State left in tree (Rule 30 report)

- The C_RT per-LB timing trap in `proveKernel` (`[BURST-PH]`/`[RT-LB]`) is still
 in — remove before any production gate on this branch.
- The sacred dump is retargeted at the C14-reverse rec2/9 chain (retarget is
 Rule-14-sanctioned and stays until the next investigation needs it elsewhere).
- No pool, config, or prover-semantics change was made.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
