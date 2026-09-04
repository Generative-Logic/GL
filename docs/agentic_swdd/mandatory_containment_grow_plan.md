<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Mandatory-containment grow — retiring the pairing merge `[IMPLEMENTED]`

> Status: design written 2026-08-21, maintainer direction given ("no merge
> at all — grow, and control in the read candidate whether the mandatory
> stuff is in; pair filtering is part of pairing and falls away too").
> **Step 1 implemented the same day on ** —
> batches 2 and 3 folded into one containment-controlled grow. See
> `D-298` for the landed decision and
> `I-155` for the rewritten contract. Step 2 (batches 1 / 4 / 5) is still
> open.

## Why

The C_RT investigation pinned the shortcut RT explosion on the request
generator's pairing merge. The RT aggregate (permanent instrumentation,
`.rt/_aggregate_<tag>.log`, shortcut measurement run 2026-08-21,
13340.6 burst-seconds attributed, 0.14 unattributed) measured:

| Region | burst-s | % |
|---|---|---|
| Batch 3 (local × mail) → pairing merge | 10746.2 | 80.6 |
| Batch 2 (local delta) → grow search | 1295.9 | 9.7 |
| Batch 3 → grow search | 577.8 | 4.3 |
| Batch 2 → pairing merge | 465.0 | 3.5 |
| Batch 1 → pairing + grow | 225.0 | 1.7 |
| Batches 4/5 | ~0.1 | 0.0 |
| Request firing (`STATIC_REQGEN_FIRE_EVAL`) | 4.6 | 0.03 |
| Stump building (`MAKE_MANDATORY_*`) | 15.5 | 0.1 |

The merge is a raw cross product — every recorded base candidate visited
against every mandatory stump (batch 3: 6.8M bases × pair lists in the
tens of thousands, 1.58 ms per base versus batch 2's 69 µs) — and its
interior (duplicate scan, `base ++ stump` sort, `preEvaluateFromEncoded`,
emit-dedup decode) is pure connect cost: firing is measured free.

## The design

**One enumeration, no merge.** Every batch runs the full-key grow the
empty-stump path (CE mode) already uses: canonical name-sorted DFS over
the batch's statement universe, `normalizedEncodedSubkeys` as the growth
probe, `normalizedEncodedKeys` as the record probe, emission inside the
search. The obligatory-stump machinery
(`makeMandatoryEncodedStatementLists1/2Static`), the pair viability
pre-filter, and the merge phase retire. The minus-one / minus-two
owner-set maps lose their consumer; their construction in
`addToHashMemory` is retired in a follow-up once confirmed
consumer-free.

**The containment control.** Per batch, the mandatory ("new
ingredient") sets are the existing views. Membership is precomputed as
bits per filtered statement; the DFS tracks which mandatory views the
growing candidate still misses and prunes a node when the remaining
depth cannot cover the uncovered views; the record probe accepts only
fully-covered candidates.

**Batches 2 + 3 fold into one grow.** Both read `overallHashMemory`
over the same universe; their mandatory sets are disjoint faces of one
delta rule (a request must contain something new this burst):

- Batch 2's newness: fresh local statements
 (`intLocalEncodedStatementsDelta`; mail arrivals never enter it —
 the status-3 drain registers them non-local).
- Batch 3's newness: fresh mail arrivals (`intExternalStatements`),
 paired with ≥ 1 local statement because a purely-external
 combination already existed — and fired — at the ancestor that
 mailed it (mail comes only from direct ancestors, I-57).

Folded control: accept a candidate that covers the local delta, OR
covers an external AND ≥ 1 local. One enumeration also removes today's
sanctioned cross-batch duplicate emissions (a request containing both a
fresh local and a fresh arrival is currently emitted by both batches;
the per-call `seen` dedup never sees across batches).

Batches 1, 4, 5 keep their own generator calls initially — their
newness is on the RULE side (recovered mail rules `workingMemory`,
`localHashMemory`, `localHashMemoryDelta`), not the statement side —
and together cost ~5% of burst time.

**Gates unchanged.** `requestGatesPass`, `ownerKeyAccepts` (D-105/D-120
comparability + u_-literal satisfiability), scope fold via `deeperOf` —
the same functions at every node; the gates are closed downward under
subsets, so intermediate acceptance is implied exactly as today.

**Emission-set equivalence.** A whole key K is emitted today iff some
viable (local, external) pair sits inside it with the remainder recorded
by the minus-two map. By downward closure that is exactly "K contains
≥ 1 local and ≥ 1 external and K passes the full-key probes" — the new
control's acceptance. The ascending-order search gives each subset one
path (the I-156 partition argument), so the dedup-collapsed request SET
matches.

**LB split survives untouched.** The split's expression stumps name
search-prefix nodes (I-156) — orthogonal to the obligatory stump, valid
at full-key depth. Only the phase-2 seed-deal
(`i % splitStump.total == ordinal`) disappears with the obligatory
stumps. I-155 is rewritten (dispatch by mandatory-view sets, not stump
length); I-156/I-157/I-158 keep their meaning.

## Determinism surface

Emission order changes, so submatch positions, doom-line coordinates
(I-191), and split statistics (I-160) renumber. The run stays fully
deterministic (a pure function of proof state), but the proof path may
shift where an early-exit winner changes.

## Gates

- Unit tests (`gl_quick.exe --unit-tests`), full rebuild.
- Shortcut run: same theorem set (`theorems.txt` byte-compare), verifier
 0 failures; RT aggregate before/after comparison.
- Full pipeline run: `theorems.txt` main + incubator byte-compare against
 `main`, verifier category-identical. Byte-identity is the target; any
 deviation is root-caused, never waved through.

## Staging

1. **Step 1:** fold batches 2 + 3 into the containment grow (84.1% of
 burst time measured). Batches 1/4/5 unchanged.
2. **Step 2** — DONE 2026-08-22
 ([D-295](40_decisions.md#d-295)):
 batches 1/4/5 moved onto view-controls (batch 5 onto an EMPTY term
 list — its source was the whole search universe), then the stump
 parameters, builder, seed and merge phases deleted, then the
 minus-one / minus-two maps deleted with their writers, wipe,
 deload facets and dump sections. `theorems.txt` byte-identical at
 every step, and identical to a fresh branch-point run (one md5 across all
 nine captures); shortcut prover 117.664 s -> 102.048 s, -13.3%, measured as a
 clean pair after a power-supply fault on the host was fixed.
3. RT instrumentation stays on during the campaign and measures the
 before/after directly; `RT_MEASUREMENT` restored to 0 at campaign
 end.

## Open questions — resolved during step 1

- **The all-external redundancy argument needed no proof.** Keeping the
 ≥ 1-local requirement inside term 2 reproduces today's emission set
 exactly, and the code already covers the all-external case elsewhere:
 batch 4 pairs arrivals against `localHashMemory`, this LB's OWN rules,
 which the ancestor does not have. Only all-external combinations against
 the INHERITED rule set are the redundant ones, and those are precisely
 what term 2 excludes.
- **The filter needed no variant.** The grow universe was already
 `body.intEncodedStatements` in every batch — the generator takes it
 internally — so the folded call filters exactly what batches 2 and 3
 filtered, with `alsoAcceptFullKeys` true because the stump length is 0.
 Only the stump SOURCE views differed between the two batches, which is
 what the terms now carry.
- **Minus-map retirement is still step 2.** Their last consumers are the
 three stumped batches' record probe and `produceExpressionStumps`'
 terminal-retention probe, which now also probes `normalizedEncodedKeys`
 so the folded call's recordable frontier nodes survive the split.

## Open questions — new, from step 1

- **The chapter walker is now 19% of the shortcut pipeline** (34.3 s of
 180.7 s, `native.raw.build_stack`). It did not grow; the prover shrank
 around it. Levers and the caveat that none of them is measured yet are
 in `10_pipeline/04_prover.md`'s weaknesses.
- **Step 2 sizing.** Batch 5's mandatory source is the whole statement
 universe, so its term is vacuous and converting it turns the call into a
 plain full-depth grow. Batches 1 and 4 keep real terms. All three read a
 narrower rule registry than the folded call, so each keeps its own
 invocation.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
