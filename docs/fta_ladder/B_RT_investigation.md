<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
Contributions require CLA — see CONTRIBUTING.md.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# FTA-shortcut runtime investigation — the B-theorem blow-up

, 2026-08-11. Diagnosis only; nothing
fixed. All run artifacts referenced below are local, gitignored files under
`.debug/` (prefix-bisection logs `run_bisect38..42.log`, instrumented logs
`run_rtinv39..42.log` / `run_rttrack42.log`, RT-tracker tables
`rt_rttrack42/`, hashburst dump `hashburst_dump41_B7rec1.txt`).

## Question

The FTA-shortcut runtime grew from ~90 s to 615–710 s while the last few
B rows were added to the 42-row pool. Which theorem caused the first jump
above 50 %, which logic block suffered most, which kernel phase blew up,
and why? (The full-pipeline main run did not change — this is shortcut-only.)

## Method

1. Prefix bisection on identical current-`main` code: five `--shortcut` runs
 with `conjectures.txt` truncated to its first 38, 39, 40, 41, 42 rows.
2. Temporary caller-side wall-clock probes (all removed after the diagnosis):
 per-iteration phase-boundary times in `proveKernel`, per-LB per-phase
 accumulation around the five phase call sites, stage timers around
 `compileAndRegister` / `buildGrid` / `broadcastTheorems` /
 `deactivateRecursively`.
3. One run with the permanent RT tracker enabled (`RT_MEASUREMENT 1`,
 trigger 5 s) for in-burst section attribution.
4. One hashburst-dump run with the dump retargeted at the worst LB
 (B7's induction sub-block) to inventory its contents. The retarget
 remains in the tree, as retargets do.

## Findings

### 1. First >50 % jumper: B5 (row 40). Each later multiplicative theorem repeats the pattern.

Wall seconds on identical code:

| pool rows | added theorem | wall | delta |
|---|---|---|---|
| 38 | (B3-era pool) | 103 s | — |
| 39 | B4 monotone `·` (direct proof) | 99 s | ~0 % |
| 40 | **B5 strict monotone `·`** | **197 s** | **+99 %** |
| 41 | B7 strict growth | 395 s | +100 % |
| 42 | B8 cancellation of `·` | 593 s | +50 % |

The cost is additive per theorem, not a pool-wide slowdown: every
pre-existing LB's cost stays flat across the five runs (for example row 33's
induction sub-block costs 633/616/608/626 s CPU at N=39/40/41/42), and each
new theorem's own grid adds a cost comparable to the whole prior total.

### 2. The suffering LBs: each new theorem's own induction sub-blocks.

Per-LB CPU (summed over executor parts, whole run):

- N=40 adds B5's `(strictOrder[1,4,9,10]) -> (preorder[1,4,6,11]) ->
 (in3[9,11,12,5]) -> (in3[10,11,13,5]) -> (in2[rec2,9,3])` — 1624 s CPU,
 plus its rec0/rec1 siblings ~530 s.
- N=41 adds B7's `(preorder[1,4,6,9]) -> (preorder[1,4,7,10]) ->
 (in3[9,10,11,5]) -> (in2[rec1,9,3])` — 3443 s CPU, the single worst LB of
 the investigation, plus rec0 ~557 s.
- N=42 adds B8's `(preorder[1,4,6,9]) -> (in3[10,9,11,5]) ->
 (in3[12,9,11,5]) -> (in2[rec0/rec1/rec2,·,3])` family — ~4600 s CPU.

B4 (row 39) proves direct — no induction, no cost. B5/B7/B8 all need
induction over a product, and it is exactly their recursion sub-blocks
that explode.

### 3. The phase: phase 2, inside request generation — not an LB-count or global-statement explosion.

- Kernel phase sums (N=39/40/41/42): ph1 = 0.7/1.1/1.2/1.9 s,
 **ph2 = 93.8/174.9/375.9/564.0 s**, ph3 = 2.9/7.5/8.8/16.7 s,
 post-join = ~1 s. Phases 1 and 3 and the post-join collectors are
 negligible everywhere.
- Non-kernel stages are exonerated: CE filter 0.001 s, `buildGrid` 0.004 s,
 `broadcastExternals` 0.003 s, `compileAndRegister` 0.1 s,
 `deactivateRecursively` ~0 s total.
- Inside phase 2 the RT-tracker section tables put **99.8 % of every heavy
 burst in `GENERATE_ENCODED_REQUESTS_STATIC`** — the grow search of request
 generation. The firing/fixpoint loop does not even register. This holds
 for B5's, B7's and B8's sub-blocks alike.
- The cost is super-linear in the LB's statement count: B7's rec1 goes
 3233 → 4055 statements (+25 %) while its burst goes 9 s → 175 s (+1800 %);
 the same hockey-stick shows for B5's rec2 (3721 stmts/59 s →
 10821/444 s) and B8's rec0 (3655/42 s → 7939/319 s).

### 4. What piles up (hashburst dump of B7's rec1, final entry).

At its last dumped burst the LB holds 7010 statements, 1761 installed hash
rules, 37287 origin rows, `startIntPi=3376`:

- **2473 `existence1` statements** and 5402 of 7010 statements carrying
 minted iteration witnesses (`it_·_lev_·` names); 2074 statements carry
 two distinct witnesses — a pairwise family (e.g.
 `(or5[it_0_lev_0_30, it_1_lev_4_244, x6,1,4,7])` for every witness pair).
- **~1900 or-compact statements** (`or0/or1/or2/or3/or5`).
- **62 % of all statements are duplicated into two `_ordis_` branch scopes**
 (`(or2[rec,2,1,3])` and `(or2[x6,2,1,3])` predecessor case-splits running
 inside the induction sub-block): 2677 + 1646 branch-scope copies versus
 2308 at main.
- On the rule side, 1320 of 1761 rules mention minted witnesses, **610 carry
 two or more distinct witnesses**, and 934 involve negated equalities —
 the `strictOrder = preorder ∧ ¬=` machinery propagating `¬=` between
 witness pairs is over half the registry.

Mechanism chain: a multiplicative order theorem needs induction; each
induction step expands `a·s(k)` through addition, minting successor /
addition / multiplication existence witnesses; the predecessor or-cohorts
split on the recursion variables and copy the population into branch
scopes; the strict-order machinery instantiates negated-equality and
or-compact families over witness *pairs* (quadratic); the grow search then
explores candidate premise sets over thousands of same-shape statements
against hundreds of subkeys — the super-linear fan-out that is 99.8 % of
the runtime.

### 5. Parallelization pathology (secondary, wall-clock relevant).

The straggler split fails exactly on these LBs at many bursts: the stump
producer returns **`stumps=0`**, and the classify falls back to one unsplit
part on the assumption "no stumps → the real burst would generate nothing
either" — demonstrably false here (those serial bursts then do 40k–200k
submatches, 10–100+ s each, pinning the phase-2 barrier). At other bursts
the LB does split into 32 buckets but stays flagged `[SPLIT] ineffective`
(one bucket above the fair share). The heavy work is reachable through a
request mode the stump space does not cover.

## Proposed fixes (not implemented — Rule 8, maintainer's pick)

1. **Cap or subsume the witness-pair families.** The quadratic
 `¬=`/or-compact instantiation over `it_·_lev` witness pairs is the fuel.
 A subsumption or relevance gate on pairs of *minted* witnesses (as
 opposed to conjecture variables) would cut both statement and rule
 populations at the source. Witness-depth capping already exists (I-171);
 this is the pairwise axis, not the depth axis.
2. **Branch-scope sharing instead of copying.** 62 % of the statement
 population is `_ordis_` branch-scope copies inside the induction
 sub-block. Letting branch scopes read ancestor-scope statements without
 materialized copies would shrink the search universe by ~2.6×
 (architectural — touches scope semantics).
3. **Grow-search pruning.** `GENERATE_ENCODED_REQUESTS_STATIC` explores
 candidate sets discriminated by operator/name only; with thousands of
 same-operator witness statements the branching factor explodes.
 Argument-level discrimination (e.g. indexing candidates by first
 argument id, or a candidate-count-aware cutoff with staged deepening)
 attacks the exponent directly.
4. **Fix the `stumps=0` fallback.** Make `produceExpressionStumps` cover the
 request mode that actually carries these bursts (its filter probes
 `normalizedEncodedSubkeys` with `alsoAcceptFullKeys=false` only), or
 bucket by base candidate when stump enumeration is empty — the current
 fallback assumption ("no stumps → cheap burst") is falsified and costs
 the barrier tens of seconds per iteration.
5. **Unfair-advantage lemma route (pool-side, zero machinery).** The
 standing precedent: if the multiplicative rungs' induction grids stay
 this expensive, adding true pool lemmas that let B-style theorems prove
 directly (as B4 did) sidesteps the explosive induction entirely.

## Addendum — B5 rec2 statement-necessity analysis (dump `hashburst_dump40_B5rec2.txt`, 40-row pool)

Final state of B5's `(in2[rec2,9,3])` block: 12 861 statements, 2 020 rules,
103 626 origin rows. B5's proof method in the artifacts is **`direct`** and
the theorem emits at grid burst 11; the induction block is wiped right after.
**Nothing the block built enters B5's proof — the induction route raced the
direct route and lost.** Deactivation itself reacted promptly (wipe at the
emission burst); the cost is the speculation, not stale cleanup.

Population classes (overlapping):
- existence closure 52 % (6 670 `existence1/11` rows; 3 736 over ≥2 minted
 names; witness set is only 68 names — the mass is the pair closure, each
 firing minting the next witness);
- cross-scope copies ~37 % (2 451 texts byte-identical at main AND a branch;
 the predecessor cohort opened on the MINTED witness `int_lev_5_391` is 82 %
 copies of main);
- negated equalities 1 060 rows, 17 % ever cited;
- globally only 39.8 % of live statements are ever cited as a source of any
 derivation.

The three `_ordis_` cohorts are live, not refuted — D-242 dead-branch
removal cannot erase them; the levers are not-open (gate cohort opening on
conjecture-variable terms, not minted witnesses), not-copy (ancestor-scope
visibility instead of materialized branch copies), not-feed (demand-driven
existence instead of witness×witness totality closure). Caveat: B7 proved
BY induction, so for B7 the block is needed and only the in-block hygiene
applies; the race-scheduling lever applies to B5-class rows.

## Handoff spec — ancestor-known dedup (maintainer-approved direction; a LATER session codes)

Maintainer directive: an expression offered at a scope whose parent scope
already knows it must not be added. Diagnosis result: **that contract already
exists in the code** — Site F in `addExprToMemoryBlock` scans
`ancestorsOf[valId]` (self + every strict prefix) against
`intKnownStatements` and refuses ancestor-known deposits, and the ordis
branch scopes DO register `main` as their ancestor (`ancestorsOf` =
`{1, branchVid}` in the dump's nameMap table). The duplicates are therefore
VIOLATIONS of the existing contract, not a missing feature:

- Registry-level census (B5 rec2, last dump call): 8 614 origIds known at
 main, 6 370 at an ordis branch, **3 125 known at BOTH** (mostly
 `existence1` over minted witnesses, or-compacts, negated strictOrder).
 Not list bloat: zero duplicate (text, scope) rows inside
 `intEncodedStatements`.
- Arrival order at statement-list level: 1 987 of 2 671 duplicated texts are
 main-FIRST (branch row admitted in a later burst while `(origId, main)`
 was already registered — Site F should have refused), 211 branch-first,
 473 same-burst ties (two deposits in one burst each checking before the
 other's commit — a door-level race Site F cannot see).

Steps for the coding session:

1. **Name the violating writer first (one trap run).** Temporary trap at the
 `intKnownStatements` insert: when inserting `(origId, vid)` with `vid`
 non-main and `(origId, anc)` already known for an ancestor `anc`, print
 the writer context (call path / state / mail origin tag). Prime suspects:
 the revival un-know resets that deliberately clear known rows so Site F
 will not drop a re-deposit (the two commented sites in
 `revisitRejected*` machinery, I-178's documented un-know), and any
 deposit path that skips the `addExprToMemoryBlock` door; the equality
 doors are unlikely (the duplicate shapes are not equality-shaped).
2. **Fix the named path** — either route it through the door, or re-run the
 ancestor check after a revival un-know re-admission.
3. **Close the same-burst tie window** (473 rows): an end-of-burst sweep
 (single-threaded seam, I-50 style) dropping branch-scope rows whose text
 became main-known within the burst, via the existing removal door
 `removeExpressionFromMemoryBlock`.
4. Open design questions recorded earlier still apply: disposition of the
 dropped branch row's levels (drop = conservative, matches I-164; merge
 risks over-claiming for `allLevelsInvolved`), branch-refutation
 ancestor-awareness (D-242 path), delta-container purge on removal.

Expected effect on B5's rec2 block: −3 125 registry keys / −~4 700 list rows
(−37 % statements); the grow search is super-linear in the statement
universe, so the burst-time cut compounds; double firings on twin copies and
their origin-log volume disappear as well. The same hygiene applies to B7's
and B8's blocks (B7 proves BY induction, so there it matters for a needed
route).

## Post-fix measurement — B5 repeated after the ancestor-known dedup squash

Same protocol as the original B5 dump run (40-row pool, dump on
`(in2[rec2,9,3])`), on the squashed branch tip (trace
):

| metric | pre-fix | post-fix |
|---|---|---|
| 40-row shortcut wall | 197 s | **112 s (−43 %)** |
| B5 rec2 final statements | 12 861 | **7 700 (−40 %)** |
| texts at both main and an ordis branch | 2 451 | **0** |
| ordis branch-scope rows | 6 801 | 1 830 |
| statements ever cited as an origin source | 39.8 % | **64.7 %** |
| theorems | 40/40, B5 `direct` | 40/40, B5 `direct` |

The contract now holds exactly (zero cross-scope duplicates; the sweep even
shrinks the population inside the last burst, 7 990 → 7 700). Main is
unchanged at 5 675 rows — confirming the erased branch mass was copies.
The residual branch rows are genuine branch-local derivations.

What remains expensive, in post-fix numbers: bursts still carry ~all the
wall (109 s of 112 across 45 bursts; top bursts 22/19/15 s), the
population is now dominated by the **witness existence closure at main**
(4 134 of 7 700 rows = 54 % `existence1/11`), and the **`stumps=0` serial
fallback still pins the phase-2 barrier** (`[SPLIT] ineffective … buckets=1
stumps=0` with work ≈ 34–37 k on the rec blocks). The remaining levers are
the three still-open proposals, now ranked by this data: witness-pair
existence subsumption (attacks the 54 % family), the `stumps=0` fallback
fix (attacks the serial 15–22 s barrier bursts), grow-search
argument-level discrimination (attacks the exponent).

## Disproof-twin removal in shortcut (ConfigFTA `try_contradiction: false`)

Maintainer decision: the shortcut pool is declared all-true, so the
positive-head contradiction twin ("assumes the head verbatim and can only
ever disprove it") can never fire there. Disabled for the shortcut config
only; the reductio twin, Peano/Gauss, and the incubator are untouched.
Trade-off accepted: a mis-authored false row now stalls at the burst cap
instead of being actively disproved.

Acceptance (42-row pool, run ): 42/42 proved,
verifier **5 132 checks / 0 failures — airtight, the exact pre-change check
count** (the twins contributed zero rows to any proof, confirming
"structurally dead"); grid shrinks 234 → 193 LBs at warm-up. Wall
264 → 257 s (−2.7 %): small, because post-dedup the twins ran in parallel
off the critical path — the wall is still pinned by the serial rec-block
bursts (253.6 s of 257 s in bursts). The next wall-relevant levers remain
the `stumps=0` fallback and the existence closure.

## Rule-registry audit (B5 rec2, final dump of the twin-free run)

1 884 rules collapse to **438 witness-abstracted shapes**; 89 % of instances
cite concrete minted witnesses; 94 % are single-premise. CORRECTED
provenance framing (maintainer, 2026-08-11): these are NOT gap detectors and
NOT memoized partial matches — the true gap-detector population (admission
machinery) is small exactly as designed (`admissionMap` 37,
`admissionStatusMap` 254, `admissionMapIntegration` 5, `rejectedMap` 1 263
parked). The 1 664 witness-citing single-premise rules are **derived
implication-shaped statements promoted to installed rules** through the
implication-compilation door — something derives ground conditional
implications per witness pair (shapes match the equality-necessity /
weak-variable / equivalence-class expansion machinery; mint site
unverified, one-trap question).
Every family exists as a perfect mirror pair — original + contrapositive
(167/167 strict-order bridge, 51/51 antisymmetry curry, 45/45 and 44/44
nonzero/predecessor, 24/24 equality congruence) — so the registry carries
~2× rules for the same logical content.

Verdict: **no semantically wrong rule found** — every family is a valid
partial consumption (strict-order bridge `a≠b → a<b`, antisymmetry curry
`a≤b → a=b` given the reverse known, `a≠0 → ∃pred` / `a≠0 → 1≤a`,
congruence pairs). The pathology is retention economics: quadratic ground
instantiation over the witness set, doubled by contrapositives, and mostly
speculative (the ¬=-between-witness statements that would fire the largest
family are ~17 %-used).

**One true anomaly: the x-copy parallel family — CONFIRMED A BUG by the
maintainer (2026-08-11).** 282 of 1 884 rules (15 %) and 553 of 7 700
statements (7 %) cite `x6` — the axed x-copy of numeral one — including
byte-twin rule families duplicated verbatim with `u_6 → u_x6` (44+44 in one
shape alone). Maintainer verdict: **x-copies exist solely to enable
theorems for anchor numerals and must stay confined to the anchor; an
x-name in any derived statement or rule outside the anchor context should
not exist.** A dedicated session owns the fix.

Starting hypothesis for that session: the axed-variable containment at the
`addExprToMemoryBlock` door drops deposits carrying an x-name in an
argument slot, yet 553 such statements are live in this sub-block — so
either the sub-block's own `intAxedVariables` set is empty (the axed set
may not propagate from the anchor/batch context to descendant LBs), or the
x-names enter through a path that bypasses the door (rule instantiation /
broadcast into sub-LBs mints `u_x6` rule slots without any statement-door
check). Quantified expectation: the fix removes ~15 % of the rule registry
and ~7 % of the statement population of the monster blocks for free.

Audit-derived proposals (undecided): fix the x-leak; skip contrapositive
minting for witness-citing ground residuals (halves the ground families);
park witness-citing residuals rejectedMap-style until their trigger
polarity exists instead of admitting them speculatively.

## Cleanup status

All temporary `[RTINV]` probes and the RT-tracker enablement are removed;
`parameters.hpp` is back at `RT_MEASUREMENT 0` / trigger 120. The hashburst
dump remains retargeted at B5's `(in2[rec2,9,3])` chain (retargets persist
by convention; the earlier B7 target was superseded for the addendum's
necessity analysis). The 42-row `conjectures.txt` is restored
byte-identically.

## X-copy leak — FIXED (2026-08-11, )

**Mechanism (trap-proven).** Hypothesis 1 was right, with a sharper edge:
`prehandleAnchor` early-returned on `isPartOfRecursion` LBs ABOVE both the
`intAxedVariables` mint and the child recursion, so induction subtrees kept
permanently empty axed sets and the containment door was a no-op there. A
route-attribution trap at the door showed exactly ONE external-mail seed —
the parent's `(AnchorFTA[1,2,3,4,5,x6,7,8])` statement riding the delta →
`fillMailOut` → absorb at warm-up — and 704 purely internal implication
registrations breeding from it (553 statements / 282 rules at the end).

**The fix (three maintainer decisions, one session).**

1. **Set-only containment** (D-272): recursion
 subtrees mint the axed names (door armed; a descendant's trace-accumulated
 set is a superset of its ancestors', so the mailed ancestor form is
 refused) but never register the axed-anchor statement. A first
 uniform-treatment attempt (statement + armed door everywhere) was
 implemented and abandoned same-day: no runtime benefit (263–300 s vs
 257 s baseline) and it unmasked the provenance gap below.
2. **Anchor-numeral exception**: a block-#1 recursion root whose goal (the
 theorem head) cites a `(1)`-typed anchor slot value leaves its subtree
 untouched — pure set-only un-proved exactly B3 (unit product, pool rows
 37/38, the only anchor-numeral-headed rows; 25/26 in-batch), whose
 pre-fix induction chapters cite ZERO x-content: the x-machinery is
 search-dynamics scaffolding there, so only full pre-fix dynamics restore
 the proof (a statement-with-armed-door middle ground is provably
 near-inert). Block-#2 `_induction_` side-chains always stay set-only.
3. **Mailed-rule origin maintenance** (D-274): the
 leak had MASKED a pre-existing provenance-transport gap — or-disintegration
 rule families (K mutual-exclusion + D-269 subset-exclusion) are flat hash
 rules that never ride the statement delta, and their `disintegration`
 origin rows had no paired `addMailOutOrigin` mirror; a descendant
 receiving a fired head by mail held a citing row it could not resolve
 (`buildStack: no origin found`, three distinct crash sites across the
 variants). Fixed producer-side with the missing paired writes, plus a
 receiver-side `reconstructed ⟵ expansion | carrier` row at the status-3
 carrier install.

**Shortcut acceptance (run_accept42, production path):** 42-row pool,
verifier **5 004 checks / 0 failures — airtight, all proof graphs
verified**; 26 proven in-batch (B3 restored); prover wall **119.1 s vs
257.3 s baseline (−54 %)**. B5 rec2 final dump: 4 853 statements (was
7 700), **872 rules (was 1 884, −54 %)**, 47 902 origins (was 93 488);
statements and rules sections **strictly x-free** (armed
`intAxedVariables (1)`; the only x6 mentions are mail-merged history rows —
documentation per Rule 16/I-44). The audit's quantified expectation
(~15 % rules / ~7 % statements) was an underestimate: killing the breeding
also killed its downstream witness/negated-equality fan-out.

**Full-pipeline regression gate + determinism (runs `run_xfix_full2` /
`run_xfix_det2`):** verifier **137 403 checks / 0 failures — airtight**
(fix_1 full baseline 137 387; the +16 delta is the fix's derivation-row
change); `files/theorems/theorems.txt` **byte-identical** to the standing
pre-fix full-run baseline (,
stable through the B8 and fix_1 acceptances); determinism rerun reproduces
the check count and `theorems.txt` + `global_theorem_list.txt`
byte-identically. Trap audit of the whole `main..HEAD` lineage before the
squash: zero residual instrumentation (each phase removed its own traps;
the Rule-14 dump retarget stays by convention).

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
