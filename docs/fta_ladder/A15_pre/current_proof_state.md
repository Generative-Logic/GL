<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# A15 — current proof state and the handoff design for the next session

> **Status (2026-08-04 —): the residual RT floor
> is RESOLVED (D-257). Bisect: the
> negated-compound integration commit alone is 11.1 s → 44.9 s; the
> traced culprit is the compound-relation `wrN == 1` marker template
> (`(preorder[1,4,11,marker])` in the addition-cancellation grid's
> recursion LB, re-instantiated per minted witness → the
> totality-existence pump, 13,224 statements / 90% `existence1`). Fix:
> compound relations never install marker templates — fully-bound
> both-polarities integration retained (A15's reductio needs the
> positive half of its dual pair; a negated-only qualification gate was
> tried and rejected for regressing A15). Decisive run: 52 s → 13.5 s,
> 17 rows incl. A15, verifier 2876/0, unit tests 1369/1369.**

> **Status (2026-08-03, coding session —,
> from the squashed ): the three design blocks below are
> IMPLEMENTED AND VERIFIED (D-252,
> I-177, I-178).
> Six commits: ordisOnly plumbing → reader discipline → the ordis
> qualification route (row 34 installs tagged markers) → `rejectedMapOrdis`
> + equi hook + dump section → the two-route opening / park / starter /
> mail revival (with the maintainer-approved un-know) → SwDD + docs.
> Maintainer decisions taken during planning: the ≥4 count INCLUDES the
> head (row 34 boundary pass); the candidate-at-head-input-slot filter IS
> adopted; revival is un-know + mail (the parked statement is a known
> local statement — `resetParkedOrStatementRegistries` is the documented
> I-85 exception). Decisive shortcut run: `theorems.txt` BYTE-IDENTICAL
> (17 rows incl. A15 — the m-cohort admits via row 34's `ordisOnly`
> demand and revives from the park), verifier 2876/0 airtight, runtime
> 155 s → 56.3 s (45.5 s is the pre-ordis floor); dump shows parked
> cohorts shrinking 5 → 2, the survivors parked forever. Unit tests
> 1369/1369. Deferred by design: integration-side probe, negated
> disjuncts, validity widening, branch-quiescence trigger; the head-gate
> stays the tightening knob. The full standard `main.py` run is
> DEFERRED to the next session (maintainer, session close) and remains
> blocked on the pre-existing Peano chapter-export crash (below);
> the on-disk `files/theorems/theorems.txt` baseline for that run's
> count comparison is 36 rows.**

> **Status (2026-08-03 —): BIG PLANNING
> SESSION, no code — coding is the NEXT session. Three design blocks are
> discussed, maintainer-reviewed, and recorded below: (1) admission-KEY
> CREATION, (2) ordis cohort OPENING + SEQUENCED RELEASE, (3)
> `rejectedMapOrdis` parking/revival. The maintainer's framing: the plan
> is a starting point and will grow over time — GL's way is to add
> functionality on demand; planning everything in advance is
> impossible.**

> **Status (2026-08-02, session close —,
> tip ): A15 PROVED and stable; the NEXT SESSION implements the
> maintainer-designed ordis ADMISSION/REJECTION machinery (design below).**

## Session-close summary — what is landed and agreed (2026-08-02)

**Proved and verified.** A15 is in `files/shortcut/theorems/theorems.txt`
(pool 17 = the 16 baseline lemmas + A15), verifier airtight on the final
run (2876 checks / 0 failures), unit tests 1356/1356. The head-switched
totality twin is NO LONGER a separate row: the alpha-variant mirror skip
(below) stops scheduling it, so the earlier pool-18 state (A15 + its
mirror both proved) collapsed to the single canonical row.

**Landed this session (12 commits, `abff0d1f..cac8aebf`).** The sequenced
or-disintegration machinery (substrate + wiring + the three chain fixes:
bootstrap-stages-instead-of-mailing, branch-contradiction refutation
emission, `ConfigFTA` iteration cap 35 → 45); the level-UNION fix in both
contradiction emissions plus the digit-arg membership exemption at the
induction promotion; the I-16-consented verifier extension for the
scope-carrying four-field reductio `contradiction` rows; the Rule-30
removal of the previous session's five leftover debug traps; the Peano
config-gate crash fix (demand-managed compound relations are config-backed
compacts only); and the mirror-pre-emit ALPHA-VARIANT SKIP
(`mirrorIsAlphaVariant` — normalize-recognize-skip, maintainer-designed):
a head-switched mirror that is the same statement up to bound-variable
renaming (the totality shape) is never scheduled as its own grid.

**RT measurements (trap-free, identical restored inputs).** Shortcut
pipeline: baseline ( + cleanups) **45.5 s / 16 theorems /
2605 checks** vs branch tip **~145–149 s / 17 theorems / 2876 checks** —
3.2×. Decomposition: the WHOLE gap sits in iterations 3–4 (38 s → 136 s);
the extra iterations 36–45 are quiescent and free. Per-grid: two pre-emit
mirror grids carry it — `preorder ∧ ¬strictOrder → =` (28 → 78 s) and the
successor-lemma mirror (10 → 58 s); both are genuinely different
statements (predicate swaps), so the alpha skip does not remove them.
Standard batches (incubators + Peano prove stage) show NO measurable
branch cost — identical theorem counts and timings up to the shared
pre-existing crash. Three RT attempts measured USELESS and were reverted
by the maintainer: the `it_`/`int_` witness filter, the primed-LB
exclusion (both ~no RT change); the alpha skip landed for the
duplicate-statement fix but is also RT-neutral. The maintainer's verdict:
3× RT for one or-family is unacceptable — eager cohort admission is the
wrong polarity.

**Pre-existing full-pipeline blockers found (NOT from this branch).**
(a) Peano crash `cfg && "updateAdmissionMap: compound-relation core must
have a config"` — FIXED on the branch: recognition now
engages only for config-backed relation compacts; spontaneous compacts
keep the pre-recognition consumption. (b) Peano chapter-export crash
`buildStack: no origin found` on the `__contradiction__!(=[7,2])` LB —
OPEN, reproduces identically at the handoff commit; blocks
EVERY full standard `main.py` run and therefore the branch's full-run
DoD. The killed full run also wiped `files/GL_binaries` +
`files/simple_facts`; both were restored from the `GL_official` snapshot
(2026-07-31) and the shortcut pipeline runs cleanly on them.

## Design decisions — admissionMap KEY CREATION (2026-08-03, maintainer-reviewed)

Scope: admission-key creation only. The disintegration step and
`rejectedMapOrdis` are a separate, not-yet-held discussion — nothing about
them is decided by this list.

1. **Input-marker keys are never cleaned up.** `cleanAdmissionMap`'s
 output-only consumption is the intended contract, not an accident: an
 output is heuristically well-defined (one value — a computation demand
 is satisfied once and forever), while an input can have multiple
 variants. Keys with the marker at an input slot stay live indefinitely.
2. **Ordis reads never consume.** A match by the ordis probe must not
 clean up or consume an admission key — ordis is hypothetical. No
 `consumedAdmissionKeys` involvement from ordis reads.
3. **Regular Writer A stays byte-identical.** The runtime marker-rule
 route (`makeNormalizedKeysForAdmission`) keeps output-slot candidates
 and the existing qualification gates exactly as they are. The
 mid-discussion idea of widening the regular tier to input slots is
 superseded by decision 4.
4. **New ordis-only qualification route** in
 `makeNormalizedKeysForAdmission`, additive to the existing gates. A
 rule that fails the (A) classic / (B) long-key / (C) local-u gates
 still generates admission keys FOR ORDIS iff: every non-anchor
 expression is an operator application (core in the `operators`
 registry — no equalities, no typing atoms, no compacts), and the rule
 is at least 4 operator expressions long excluding the anchor. Count
 includes the head (corpus row 34 = 3 premises + head = 4, passes at
 the boundary) — reading still flagged for maintainer confirmation.
5. **Input-slot marker candidates exist only in the ordis route.**
 Candidate = a variable at a config input slot of a premise, with
 per-candidate conditions: confinement (the candidate appears in no
 other premise; the head is allowed) and concreteness (every other
 argument of the marked premise is bound by the subkey, so the fired
 key is fully instantiated). Open micro-decision: additionally require
 the candidate at an input slot of the head (row 34 satisfies it; cheap
 extra garbage filter).
6. **`ordisOnly` tag.** Keys fired by the ordis route carry an
 `ordisOnly` flag in the admission value. General Pass-B admission
 (`isAdmitted`) skips tagged keys — general witness generation sees
 nothing new. The ordis probe reads both tagged and untagged keys.
 Merge rule: a key with both a regular and an ordis-route producer is
 regular — untagged wins, permanently.
7. **The ordis probe consults the admission maps only — never I-6.**
 `isAllowedAsOperatorInput` (the I-6 single-input-operator shape rule)
 stays what it is: the unconditional mint permission inside Pass B. It
 must not appear in the ordis probe — it would admit every
 exists-predecessor cohort everywhere and kill parking. Deliberate
 asymmetry: "may this witness get a name" (I-6, cheap) versus "is this
 case split worth opening" (demand evidence only).
8. **Head-gate rejected as default, kept as a knob.** The intermediate
 design — register a premise demand only when the rule's instantiated
 marked head is already an admission key (demand back-propagation) — is
 not part of the current design; the shape gate of decision 4 is the
 chosen, deliberately looser filter. If the A/B shows it too loose
 (heavy grids re-admitting), the head-gate is the documented tightening
 option.
9. **Integration side effect accepted.** Ordis-route marker firings ride
 the shared staging, so they also register the degenerate single-atomic
 integration templates (`prepareIntegration` replay) like every marker
 firing today. Accepted, not filtered.

**Verified findings underpinning the above (trace/code-confirmed
2026-08-03):**

- Two admissionMap writers exist: Writer A (runtime marker-rule firings
 via `admissionKeysAlgebra` and its drain; marker always at the
 premise's OUTPUT slot) and Writer B (`updateAdmissionMap` install-time;
 marker at the single free argument — input-only for config-backed
 compound relations, position-blind for atomic operators; equalities,
 typing atoms, and spontaneous compacts are non-participants).
- Theorems arrive with zero `u_` arguments, so gate (C) (local-u) never
 qualifies theorem installs; corpus row 34 fails (A)/(B)/(C) and today
 produces NO admission keys at all.
- Row 34 (`s(b)+pred(a) = a+b`,
 `files/shortcut/theorems/externally_provided_theorems.txt` line 34) is
 the unique current-corpus carrier for A15's predecessor witness p:
 candidate `11` confined to `(in2[11,7,3])`, subkey = the IH-witness
 commutativity mirror `(in3[m,rec,9,4])` plus the induction chain fact
 `(in2[rec,10,3])`, fired key = `(in2[marker,m,3])`, head =
 `(in3[10,p,9,4])` — the goal witness `b+p=a` itself. The Peano
 `+`-axiom cannot carry it: its predecessor variable occurs in three
 premises and fails confinement.
- The goal's own witness demand `(in3[u_10,marker,u_9,u_4])` is
 registered in `admissionMapIntegration` at main from the start
 (goal-driven integration; confirmed in
 ) — the same mechanism that
 minted m for the induction hypothesis.

## Design decisions — ordis cohort OPENING and SEQUENCED RELEASE (2026-08-03, second block, maintainer-reviewed)

Scope: cohort opening (admission) and the release policy.
`rejectedMapOrdis` (parking payload, container, revival, equi-class hook)
is the next discussion; nothing about it is decided here.

1. **Two-route cohort opening.** An or statement reaching the cohort-mint
 site opens its cohort iff:
 - **Route (a), demand-driven** (corpus-fired or heads): some
 disjunct's product template is a key in the algebra `admissionMap` —
 for now algebra only, the integration probe deferred. The probe
 reads BOTH regular and `ordisOnly` keys.
 - **Route (b), product-of-disintegration** (the D-32 criterion — the
 firing rule carries `u_` args): opens unconditionally, as in rungs
 1+2. The sequenced branch had set `allowOrDisintegration`
 unconditionally true, so the productOfDis distinction must be
 re-read at the or consumption — a restoration; the signal still
 travels on firing records and mail.
 - Neither route → park (next discussion).
2. **Sequenced release RETAINED, for both routes.** RT is the main enemy
 and the release policy is the only RT lever on route (b), which
 bypasses the admission filter by design and is the DOMINANT branching
 source of the FTA campaign: prime-definition unfolding fires
 productOfDis or heads throughout Parts E/G, and the future
 sequence/product definitions are or-based (empty-or-head), firing
 throughout Parts F/H/I. One live branch per cohort; the existing
 deterministic ranking orders the queue (non-equalities by descending
 complexity, bare equalities, anchor-argument equalities dead last).
 The mid-discussion all-at-once idea (both the route-(b) simplification
 and the general pivot) was examined and withdrawn on this argument.
3. **Route (a) starter = the admitted disjunct.** The demand names not
 just THAT the split is relevant but WHICH branch carries the
 relevance — it opens first. Tie rule when several disjuncts are
 admitted: the deterministic ranking among the admitted ones. Route (b)
 has no admitted disjunct; it starts at the top of the ranking.
 Structural guarantee: equality disjuncts can never be the starter on
 route (a) — no writer produces equality-shaped admission keys.
4. **Closure trigger, precise semantics.** The next branch releases when
 the live branch RESOLVES: (i) **goal-reached** — the branch derives,
 at its own scope, a statement matching a `toBeProved` goal anywhere on
 its parent chain (theorem goals at main and subproof-scope goals —
 `<goal>_subproof_` / implication scopes — work identically; the probe
 is parent-chain-wide); or (ii) **retired refuted** (D-242). The
 branch-local "close" is the resolution event plus convergence
 bookkeeping; the parent goal itself discharges only via convergence
 promotion (all branches, or the D-242-reduced count) — I-164
 untouched.
5. **OPEN — proposed third trigger: branch quiescence.** Release also
 when the live branch has no work left (the D-194 quiescent signals,
 `hasWork`/`mailPeek`). Would close the merge-first hole (decision 6)
 while keeping one-live-branch RT control. The maintainer deliberately
 does NOT adopt it yet — kept as the recorded candidate fix, decision
 deferred.
6. **Documented robustness limits of the pure goal/refute closure** (why
 decision 5 is recorded): merge-first deadlock — a branch whose
 contribution is a convergence FACT (not a parent-chain goal, not a
 refutation) stalls the sequence forever, while convergence needs all
 branches (circular wait by construction); goal-coupling brittleness
 (what counts as "progress" depends on the dynamic `toBeProved`
 population); iteration-cap pressure from serial resolutions (the
 observed 35 → 45 raise).
7. **Single ordis layer is the contract.** GL does not support nested
 ordis: `max_or_depth = 1` in every config (default likewise), enforced
 at the cohort-mint site; an or statement arriving inside a branch
 consumes flat — K mutual-exclusion implications still emit at every
 depth, so no soundness loss. Subproof scopes do not count toward the
 depth. The shortlist's lemma granularity is the structural guarantee:
 nesting unrolls across the ladder (one split per lemma; a proved lemma
 returns to the pool as a flat rule the next lemma splits on top of),
 not across scopes. The whole admission/sequencing design surface is
 therefore single-layer.

**Verified findings underpinning this block (code/config-confirmed
2026-08-03):**

- `forceDeep = (status == 0)`: every FACT disintegrates in forceDeep
 mode, where the existence decomposition mints the `int_` witness
 UNCONDITIONALLY (the "Unconditionally generate int_ path" branch of
 `disintegrateExprCore2`) and only the `it_`-twin products are filtered
 out; no admission pass runs. Pass B admission (`isAdmitted` / I-6 /
 `isAdmittedIntegration`, rejection parking) applies only to goal-status
 disintegration. This corrects the earlier assumption that branch
 witnesses ride I-6 — a released branch is self-sufficient: its seed
 decomposes and mints with no admission involvement.
- Consequently the admission maps have exactly ONE ordis touchpoint: the
 cohort-opening probe.
- At the or consumption no branch scope is minted: leaves are queued
 (`orPendingBranches`), K rules emit flat, and the branch scope mints at
 release time (seed on `sameIterationInternalMail`, absorbed through the
 full kernel pipeline).
- Prime disintegration produces productOfDis or-rules
 (`d | p ⟹ d = 1 ∨ d = p` as a local `u_` rule), confirming route (b)
 fires throughout the upper ladder, not only in rungs 1+2.

## Design decisions — `rejectedMapOrdis` PARKING and REVIVAL (2026-08-03, third block, maintainer-reviewed)

Starting-point design (maintainer: "OK-ish to start with, will grow over
time — GL's way is to add functionality on demand; planning everything
in advance is impossible").

**The mirror contract (the one-line summary):** `rejectedMapOrdis` is
subject to everything `rejectedMap` is subject to — container, key
space, revival seams and mechanics, equi-class discipline, scope wipe,
deload, dump — differing only in producer site, value record, and what
revival re-runs.

1. **Park trigger:** a theorem-based (route (a), not productOfDis) or
 head whose cohort probe is a TOTAL miss — no disjunct product
 template is an admission key. Route-(b) heads never park (they open
 unconditionally).
2. **Keys:** one entry per OPERATOR-based disjunct, under its marked
 product template (`(in2[marker,m,3])`, `(in3[a,marker,b,4])`, …), in
 the packed `(templateId, validityId)` admission key space
 (`templateInterner`, I-89). Equality disjuncts contribute no key (no
 writer can ever produce a matching demand); negated disjuncts skipped
 until that case emerges. Same key space as `admissionMap` → revival
 is a direct O(1) probe at key-gain, no sweeps.
3. **Value:** the original or expression + its validity + its levels run
 — the complete reopening context (the classic rejection buffer
 captures levels at buffer time for the same reason).
4. **Revival = mail-based, 1:1 with `revisitRejected2`:** hooked at the
 same key-gain seams (`drainAdmissionKeysAlgebra` step (5) and the
 inline Writer-B insert; `ordisOnly` keys land through the same drain
 and trigger it too). On a key match, the parked or statement is
 re-deposited on `sameIterationInternalMail`; the absorb re-runs the
 normal or consumption — K rules re-emit (deduped), and the probe runs
 FRESH, so the starter falls out of the standing tie rule over ALL
 demands that arrived meanwhile, not just the triggering key. A
 revival may legitimately re-park.
5. **Double-open safety is free:** a cohort parked under several keys
 whose demands all arrive — the first revival opens it and mints the
 `orDisjunctCount` row; later revivals hit the fresh-cohort guard and
 no-op. Stale sibling entries are harmless; cleanup lazy.
6. **Equi-class hook = I-37 verbatim** (drop + re-mail the rewritten or
 statement; re-insertion only through the normal consumption
 downstream of absorb). The first block's drop-and-rekey deviation is
 RETIRED — it existed because the parked value was assumed to be a
 cohort reference; the value being the or statement itself restores
 the uniform rejected-map discipline.
7. **Lifecycle:** `TypedColdBlobMap` in the `HashMemory` tag band
 (I-99), canonical-run RMW (which also dedups re-parks of the same
 statement), `survivesDischarge`, deload facets, enrollment in the
 cold scope-wipe sweep (`eraseBlobIf(coldScopeWipe)` — parked cohorts
 of wiped subproof scopes must not leak), and a dump section mirroring
 the existing rejected sections.
8. **The three deliberate asymmetries vs `rejectedMap`** (forced by what
 is parked, not discipline deviations): producer site (or-consumption
 probe miss, not Pass B); value record (or statement + validity +
 levels, not constituent + siblings + compact); revival product (the
 whole consumption re-runs — probe → starter → bootstrap — because
 what was parked is a decision, not a fact).
9. **Deliberately deferred, add-on-demand:** the integration-side probe
 and revival source (`admissionMapIntegration` — block 2's "for now
 only algebra"); the negated-disjunct case; any validity-comparability
 widening beyond exact-validity matching (start exact, extend when a
 trace shows the need).

## The A15 worked case — the one application rule, end to end (2026-08-03)

The single concrete rule + case the whole design is validated against.
Setting: the A15 induction-step triad (chain `root → (AnchorFTA[1..8]) →
(in[9,1]) → (in[10,1]) →!(preorder[1,4,9,10]) → (in2[rec,10,3])`; LB
names: 9 = a, 10 = b, rec = pred(b), m = the IH witness, an `int_lev`
name).

**The rule: corpus row 34**
(`files/shortcut/theorems/externally_provided_theorems.txt`):

```text
(>[1,2,3,4,5,6](AnchorPeano[1,2,3,4,5,6])(>[7,8,9](in3[7,8,9,4])(>[10](in2[8,10,3])(>[11](in2[11,7,3])(in3[10,11,9,4])))))
```

Reading (3 = s, 4 = +): 7+8=9 ∧ s(8)=10 ∧ s(11)=7 ⟹ 10+11=9 — the
successor shuffle `s(b) + pred(a) = a + b`. (The rule's internal
variables 7–11 are its own binder names, not the LB's 9/10.)

**Why only the new ordis route serves it:** row 34 fails all three
existing gates — (A) one `in3` premise where more than
`minNumOperatorsKey` = 2 are required; (B) 3 elements < `minLenLongKey`
= 5; (C) a theorem arrives with zero `u_` args, so the local-u path
never applies. Today it produces NO admission keys. Under the ordis
route: only-operator expressions ✓, 4 of them excluding the anchor ✓
(exactly at the boundary), input-slot candidate `11` in `(in2[11,7,3])`
— confined ✓ (only there and in the head), concrete ✓ (partner arg `7`
bound via premise 1), head-relevant ✓ (`11` at the head's input slot 1).
Marker rule installs, tagged `ordisOnly`: marked premise
`(in2[marker,u_7,3])`, subkey `{(in3[u_7,u_8,u_9,4]),
(in2[u_8,u_10,3])}`.

**Premises versus LB facts at the step (the one-missing-fact table):**

| Rule premise | Reading | LB fact | Status |
|---|---|---|---|
| `(in3[u_7,u_8,u_9,4])` | m + rec = a | `(in3[m,rec,9,4])` — commutativity mirror of the IH witness `(in3[rec,m,9,4])` | present |
| `(in2[u_8,u_10,3])` | s(rec) = b | `(in2[rec,10,3])` — induction chain element | present |
| `(in2[u_11,u_7,3])` | s(p) = m | — | MISSING |
| head `(in3[u_10,u_11,u_9,4])` | b + p = a | the goal witness for `(preorder[1,4,10,9])` | produced on firing |

The head is not merely useful — the goal `(preorder[1,4,10,9])` (b ≤ a)
is definitionally ∃k: b+k=a, so the head IS the witness fact that closes
it. No other corpus rule can admit p in this situation (checked
exhaustively: rows 44/45 need facts the triad lacks, the Gauss rows need
interval/limit facts, the Peano `+`-axiom has the predecessor in three
premises and fails confinement).

**The expected end-to-end chain under the new design:**

1. The IH fires: `(preorder[1,4,rec,9])` with witness facts
 `(in3[rec,m,9,4])` + `(in[m,1])`, m fresh.
2. The corpus zero-or-successor rule fires on `(in[m,1])`:
 `(or0[1,m,3,2])` — "m = 0 ∨ ∃p: s(p)=m" — reaches the cohort-mint
 site at the triad's main. Route (a) probe: `(=[m,2])` → no key
 possible; `(existence3[1,m,3])` → `(in2[marker,m,3])` → MISS (the
 demand is not registered yet) → cohort PARKS in `rejectedMapOrdis`
 under `(in2[marker,m,3])` at main, value = the or0 statement +
 validity + levels. K rules emitted flat.
3. Commutativity fires on the IH witness: `(in3[m,rec,9,4])` lands.
4. Row 34's ordis marker rule fires — subkey matched (u_7=m, u_8=rec,
 u_9=9, u_10=10) — staging the concrete `ordisOnly` key
 `(in2[marker,m,3])` on `admissionKeysAlgebra`.
5. The post-fixpoint drain inserts the key into `admissionMap`;
 `revisitRejectedOrdis` probes `rejectedMapOrdis` at the same packed
 key → HIT → re-deposits the parked or0 on
 `sameIterationInternalMail`.
6. Re-consumption: the probe now hits → the cohort OPENS; starter = the
 existence disjunct (the admitted one); the equality branch queues.
7. The branch seed `(existence3[1,m,3])` is a fact → forceDeep
 disintegration mints p (`int_lev`) unconditionally:
 `(in2[p,m,3])` + `(in[p,1])` at the branch scope.
8. Row 34's MAIN rule fires (all premises present): head
 `(in3[10,p,9,4])` — b+p=a — at the branch scope.
9. The preorder definition integrates it: `(preorder[1,4,10,9])` derives
 at the branch scope → the goal probe sees the parent-chain
 `toBeProved` reached → branch resolved → the equality branch
 `(=[m,2])` releases.
10. The equality branch refutes from ambient facts (m=0 → rec=a →
 b=s(a) → a ≤ b collides with the theorem premise) → D-242 retires
 it, count 2 → 1.
11. Reduced-count convergence promotes `(preorder[1,4,10,9])` to main →
 `toBeProved` empties → induction promotion (level-union +
 digit-arg exemption, already landed) → **A15 proves**.

**Acceptance checks for the coding session:** (i) the `ordisOnly` key
`(in2[marker,m,3])` visible in the admissionMap dump section at the
triad's main; (ii) the park-then-revival visible in the
`rejectedMapOrdis` dump section across bursts; (iii) A15 present in
`files/shortcut/theorems/theorems.txt`, verifier 0 failures, no pool row
lost; (iv) shortcut RT back toward the 45 s baseline (the heavy mirror
grids' cohorts park forever — no demand ever registers for them); (v)
one full standard `main.py` run with theorem count not below baseline
(route (b) preserves the rung-1+2 or-proofs).

## NEXT SESSION — ordis admission/rejection, 1:1 with algebra/integration (maintainer-designed, agreed)

Eager cohort admission is replaced by the same park-by-default,
admit-on-demand polarity the algebra and integration admission machinery
use — "the same 1:1":

1. **Admission side = the EXISTING maps (maintainer refinement at session
 close): the algebra `admissionMap` and the integration
 `admissionMapIntegration` serve as the ordis admission source,
 unchanged — no new admission map and no new inserters.** The ordis
 probe reads them exactly as `isAdmitted` / `isAdmittedIntegration` do
 (template-space packed keys, non-minting probes). The ONLY new
 container is **`rejectedMapOrdis`** — same container form and key
 discipline as the existing rejected maps (I-89/I-99 canonical-run RMW,
 new facets appended in the HashMemory tag band, a dump section
 mirroring the existing rejected sections).
2. **Demand arises exactly as today.** The marker-fired sites and drains
 keep inserting keys into the existing admission maps only; ordis adds
 no write path there — it consumes the demands the machinery already
 registers.
3. **Candidate flow 1:1.** An or statement reaching the disintegration
 gate computes each disjunct's PRODUCT TEMPLATE — an existence disjunct
 → its definition body's witness fact with the witness slot in marker
 form (for A15's `existence3[1,m,3]`: `(in2[marker,m,3])`, i.e. the
 maintainer's "we need some s(x)=m for m because it unblocks an
 algebraic transformation"); an atomic disjunct → itself — and probes
 the EXISTING admission maps (`isAdmittedOrdis` = a read-only twin over
 `admissionMap` + `admissionMapIntegration`). Admitted → bootstrap +
 stage exactly as the sequenced machinery does today. Not admitted →
 PARK in `rejectedMapOrdis` under each product template at the cohort's
 validity, value = the cohort's bootstrap payload. K rules still emit
 flat on arrival either way.
4. **`revisitRejectedOrdis`** hooks the seams where the existing
 admission maps GAIN keys (the marker-fired inserts and the admission
 drains, the same places `revisitRejected2` runs): a new key revives
 matching parked cohorts → bootstrap + stage.
5. **Equivalence-class hook for the new map only** — an
 `applyEquivalenceClassToRejectedMapOrdis` with the same drop-and-rekey
 discipline as the existing rejected-map hooks; the admission side
 keeps its existing hooks untouched.

**Expected effect.** The heavy grids' bound-variable cohorts have no
marker demanding any disjunct product — parked forever, RT back toward
the 45 s baseline; A15's m-cohort admits through the successor-law marker
demand for `(in2[·,m,3])`.

**First acceptance check before coding.** Verify in the proving run's
trace that the `(in2[·,m,·])`-family admission demand actually lands (the
admissionMap sections / the marker that inserts it) at the triad's main —
if it registers only later, admission happens via the revival seam, which
the design covers.

**Still queued (unchanged from the earlier handoff).** Single-theorem
or-construction ("one theorem is enough" — reaffirmed this session; the
alpha skip solved only the duplicate-copy half; the construction change +
the shortcut-mode or-emission wiring remain open, including the
single-parent `or theorem` verifier row). The Peano export crash (b)
above. The full standard-run DoD, blocked on that crash.

> **Status (2026-08-02, night —): A15 IS
> PROVED, airtight.** The confirming shortcut run proves BOTH totality
> directions — row 14 `(>[]!(preorder[1,4,9,10])(preorder[1,4,10,9]))` (A15)
> and row 13 its head-switched mirror — pool 16 → 18, verifier 3154 checks /
> 0 failures end-to-end, runtime ≈ 149 s. Beyond the sequenced-ordis
> machinery (previous entry), closure needed: the level-UNION fix in both
> contradiction emissions (`dischargeContradictionScopes` + the ordis
> branch-contradiction refutation now union the two antecedents' level
> runs); the maintainer-approved digit-arg membership exemption at the
> induction completion (a recursion sub-LB completion whose only missing
> level is the digit-arg's own `(in[<digitArg>,…])` chain element counts as
> all-levels-involved); and the I-16-consented verifier extension for the
> scope-carrying four-field reductio `contradiction` rows (checker + trace +
> the contradiction-scope seed's `task formulation`). The remaining
> branch-level DoD item is the full standard `main.py` regression run.

> **Status (2026-08-02, evening —):** the
> sequenced or-disintegration architecture is IMPLEMENTED, VERIFIED WORKING,
> and closes the A15 step-block goal end-to-end (trace
> ): the m case split
> `(or0[1,int_lev_4_335,3,2])` at the triad's main opens sequenced, the
> existence branch derives the goal, the goal probe releases the equality
> branch, the equality branch derives `(preorder[1,4,9,10])` against the
> theorem premise, the new branch-contradiction emission produces
> `!(=[m,2])` at main with a two-antecedent `contradiction` origin, D-242
> retires the branch (count 2 → 1), and the reduced-count convergence
> promotes `(preorder[1,4,10,9])` to main — `toBeProved` empties and the
> step LB discharges. **A15 still does not reach `theorems.txt`:** the
> induction promotion is refused by the pre-existing `allLevelsInvolved`
> discharge gate (`dischargeToBeProved` → `updateGlobal`). The promoted goal
> carries `levels={0,1,4}` on a level-4 LB (needs `{0,1,2,3,4}`, or four
> levels without 0). Two independent causes: (a) `dischargeContradictionScopes`
> emits its product with only the in-scope statement's levels — the ambient
> collision partner's level (the theorem premise, level 3) is dropped, a
> fixable propagation gap that also affects the new branch-contradiction
> emission; (b) the found proof genuinely never consumes `(in[10,1])`
> (level 2, b ∈ N — its content arrives via the level-4 hypothesis
> `(in2[rec,10,3])`), so even with (a) fixed the set is `{0,1,3,4}` and the
> gate still refuses. Resolving (b) is a prover-semantics decision
> (Rule 8): e.g. exempting the induction digit-arg's own membership level at
> the induction promotion — the machinery re-establishes b ∈ N in both
> induction cases by construction. AWAITING MAINTAINER DECISION.
> Verifier airtight on every run (2606 checks / 0 failures); pool stable at
> 16; shortcut runtime 46.5 s (pre-change) → ~146 s (sequenced branches
> live); `ConfigFTA.json` `maxIterationNumberProof` raised 35 → 45 (the
> step goal closed exactly at the old cap).

> **Status (2026-08-02, morning):** A15 (totality implication) does NOT prove yet.
> The contradiction-based integration of negated compound premises is LANDED
> and verified working end-to-end; the one remaining gap is the `m ≠ 0` case
> split, and the agreed way to close it is the **sequenced or-disintegration
> architecture change** described at the end — deliberately deferred to its
> own session. Branch:.

## The conjecture

```text
(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9](in[9,1])(>[10](in[10,1])(>[]!(preorder[1,4,9,10])(preorder[1,4,10,9])))))
```

∀a, b ∈ ℕ: ¬(a ≤ b) ⟹ b ≤ a — row 16 of
`files/shortcut/theorems/conjectures.txt`. The human-notation proof (GL
induction orientation, rec = predecessor) is in `human_proof.md` next to this
file. The hashburst dump currently targets the induction triad
`root → (AnchorFTA[1..8]) → (in[9,1]) → (in[10,1]) →!(preorder[1,4,9,10]) →
(in2[rec0,10,3])`.

## What is LANDED and verified working (dump-confirmed)

All, tip commit "Contradiction-scope integration WORKS
end-to-end". 1348/1348 unit tests; shortcut run 2606 checks / 0 failures;
pool unchanged at 16 theorems.

1. **Recognition** (`updateAdmissionMap`): compound (non-atomic) relations
 enter the demand machinery — negation-stripped core resolved via
 `compiledEntity`. A premise with every argument unchangeable (`wrN == 0`)
 and recursion-owned goes to immediate integration preparation; one free
 input arg (`wrN == 1`, positives only) takes the marker route.
2. **Polarity contract** (`prepareIntegrationCore`): a negated compound never
 instantiates the positive template (was a silent bug).
3. **Contradiction-based integration** (`prepareIntegration` intercept +
 `prepareNegatedCompoundContradiction`): a negated compound premise
 (category ≠ atomic, ≠ or) primes a child scope
 `<parent>_boundary_contradiction_!(X[args])` seeded with the POSITIVE
 compact (status 0, task-formulation origin). Standard consumption
 decomposes the seed; gates: product `registered` at scope/main, payload
 already primed (`integrationPrepared`).
4. **Discharge** (`dischargeContradictionScopes`, hooked after
 `dischargeToBeProved`): a delta statement EXACTLY at a `contradiction_`
 scope whose negation is known on the chain up to main fires the
 refutation — the product is emitted at the scope's parent on internal
 mail with a two-antecedent `contradiction` origin row, and the scope is
 queued for wipe (`intValidityNamesToFilter` + `pendingWipeScopes`, the
 latter doubling as the once-per-step guard). The exact-scope condition is
 the soundness guard: a statement at a deeper `_ordis_`/`orint_`/subproof
 node is conditional on that node's own assumption — its contradiction
 refutes the branch (D-242), never the reductio seed.
5. **Proven in the run:** the scope assumed `(preorder[1,4,9,rec])`, derived
 `(preorder[1,4,9,10])` inside (leaner than the human proof's
 successor-law lift), collided with the theorem premise at main →
 `!(preorder[1,4,9,rec])` emitted at main → **the induction hypothesis
 FIRED**: `(preorder[1,4,rec,9])` (rec ≤ a) is a statement at main, with
 its witness `(in3[rec,int_lev_4_249,9,4])` + `(in[int_lev_4_249,1])`.

## The remaining gap — the m ≠ 0 case split

`(or0[1,int_lev_4_249,3,2])` ("m = 0 ∨ ∃pred(m)") sits at main but is
consumed FLAT: the D-32 or-admission gate requires the firing rule to be a
product of disintegration, and the corpus or0 rule is not — so no `_ordis_`
branches are minted. Flat consumption does install the per-witness K-rule
`(>[]!(=[u_m,u_2])(existence3[u_1,u_m,u_3]))` (m ≠ 0 ⟹ ∃pred, concrete,
`productOfDis=1`) — which waits forever for the negated-atom fact
`!(=[m,2])` that nothing produces.

**Attempted and REVERTED (do not retry as-is):** extending the
contradiction-scope route to negated equality premises (u_-only trigger,
positive-known gate). It exploded the runtime — total expressions
5,337 → 25,414 → 75,548 within three bursts — because an assumed equality at
a near-main scope feeds the equivalence-class machinery, which replicates
the comparable statement population into the scope, one near-copy of main
per primed witness. This is the same pathology that got `_ordis_` branches
banned from main scope historically. The revert was explicit
(maintainer-directed `git reset --hard`); the tree is at the last verified
state.

## THE PLANNED CHANGE — sequenced or-disintegration (next session, maintainer design)

Maintainer's architecture idea, agreed as the way forward — a session-sized
change:

**Or-disintegration branches are not all released at once but in sequence:
most structurally complicated disjunct first. When a branch RESOLVES, the
next branch is released. `=[a,b]`-shaped disjuncts go last — equalities with
an anchor-slot argument (e.g. `=[m,2]`, 2 = the zero slot) dead last.**

Why this kills the explosion (three stacked effects):
- At most ONE branch per cohort is live at a time.
- The expensive equality shape opens only after the productive branch
 already resolved — a free relevance gate: a useless split never reaches
 its equality branch.
- When the equality branch does open, in descent contexts it refutes within
 a few bursts and retires + wipes — one bounded rewrite pulse, not a
 standing population.

How A15 closes under it: or0 at main disintegrates sequenced → branch 1 =
∃pred(m) (cheap witness facts) → m = s(p) → b + p = a → the goal
`(preorder[1,4,10,9])` derives under the branch → branch resolved → branch 2
= `(=[m,2])` released → rec = a → b = s(a) → a ≤ b collides with the ambient
premise → D-242 retires the branch → cohort resolved at reduced count →
convergence proves the goal at main. **No negated-equality fact and no new
reductio machinery needed** — the m ≠ 0 step becomes the existing
refuted-branch path. The landed contradiction-scope machinery stays (it
produces the IH premise); the two mechanisms compose.

Design points settled in discussion (the next session starts here):

1. **Release trigger = branch RESOLVED (converged ∨ retired), not
 goal-reached only.** A refuted first branch must still release the next,
 or the cohort deadlocks. An undecided branch stalls the sequence — safe
 degradation to today's flat behavior (no explosion, no contribution).
2. **Deterministic ordering** (I-84 discipline): compounds/existence first
 by descending structural complexity (e.g. token count), bare equalities
 after, anchor-arg equalities dead last, byte-lex ties. A pure function of
 the disjunct set.
3. **The D-32 admission gate is the real lever and the risky decision.**
 Today's flat consumption of corpus-fired or heads is what blocks the
 split; sequencing is the argument for relaxing the gate (sequenced
 disintegration where only flat consumption happens today). How far —
 everywhere, main-only, per-cohort capped — must be decided deliberately
 (the D-31 → D-32 history is the cautionary precedent).
4. **Bookkeeping:** the cohort needs a pending-branch queue (ordered
 disjunct list + next-to-release cursor) in deload-safe LB state
 (`lbStateInterner` space, I-93 discipline), drained at an end-of-burst
 seam. The existing `orDisjunctCount` / `orBookkeeping` / convergence
 counting survives unchanged — the branch-mint loop becomes a scheduler.
 The K upfront mutual-exclusion implications stay upfront (cheap rules);
 only the branch SCOPES are sequenced.
5. **Latency:** convergence arrives K sequential resolutions later instead
 of one parallel wave — acceptable, bounded by existing caps.
6. **Verifier likely untouched:** it validates final chapter state; staggered
 emission timing is invisible. (Shrinks the I-16 surface.)

## Other open items on this branch (queued from the approved plan)

- **Or-construction, single-theorem + `>[]` insight:** `(>[X](A)(>[](!B)(C)))`
 with the EMPTY inner binder is already A → (B ∨ C) (GL discharge is
 classical); construct `or<N>` from every single proved theorem whose
 rightmost negated premise has an empty bound-vars list — the check already
 exists in `headSwitchOne` (its `std::get<1>(chain[i]).empty` condition);
 drop the pair requirement. The or-compact SHOULD still be emitted
 (maintainer decision): downstream shortlist proofs (B8) consume totality
 as a case-split trigger via or-disintegration. Also: the OR-construction
 block in `run_modes.cpp::fullRun` sits in the `!skipCompression` branch —
 it must also run under the shortcut's `skip_compression` (gated so
 incubator batches keep current behavior).
- **Verifier rows (I-16 consent given):** the `contradiction` origin row
 emitted by `dischargeContradictionScopes` (two antecedents, scope-carrying)
 and single-parent `or theorem` provenance. Iterate against the actual
 emitted rows once A15 proves.
- **Temporary traps still in tree (Rule 30, remove before final verification):**
 `[TRIAD-DBG]` ×3 in `prover.cpp::addTheoremToMemory` (digit-arg loop,
 triad-creation branch, auxyImplication print), `[NEGINT-DBG]` ×2
 (`prover.hpp::updateAdmissionMap`, `prepareIntegrationCore`).
- **SwDD (Rules 10/15/20):** `D-pending-<slug>` for contradiction-based
 integration + the sequenced-ordis decision, `I-pending-<slug>` for the
 polarity contract and the exact-scope discharge guard, verifier chapter,
 user-SwDD prover page. None written yet.
- **DoD (maintainer-set):** shortcut run — A15 present in
 `files/shortcut/theorems/theorems.txt`, verifier 0 failures, no pool row
 lost; PLUS one full standard `main.py` run — theorem count in
 `files/theorems/theorems.txt` must NOT go down vs. baseline, verifier 0
 failures.

## Artifact pointers (volatile, .debug/ is not in git)

Run logs from this campaign: `run_a15_impl*.log`, `run_a15_contra.log`
(the working-machinery run), `run_a15_negeq.log` (the explosion, stopped).
Trace snapshots: `hashburst_a15_stall.txt` (premise LB, pre-machinery),
`hashburst_a15_triad.txt` (triad, subproof era), `hashburst_a15_contra.txt`
(triad, working contradiction-scope era — the reference dump for the
"what works" claims above).

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
