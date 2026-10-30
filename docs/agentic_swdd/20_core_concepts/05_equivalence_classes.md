<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Core concept — Equivalence classes `[DRAFT]`

> When the prover derives `(=[a,b])`, it records that `a` and `b` are equal in the current scope. That fact is not just stored — it propagates. Subsequent expressions mentioning `a` are admitted with `b` substituted, and vice versa. Inequality propagation is the dual — `!(=[a,b])` with an existing equality class expands into sibling inequalities.

---

## The data structure — `EquivalenceClass`

Defined at [`memory.hpp`](../../../GL_Quick_VS/GL_Quick/src/memory.hpp). Fields:

| Field | Type | Role |
|---|---|---|
| `memberIds` | `vector<NameId>` (32-bit names) | The members of this class — all known-equal — as NameMap ids, sorted by DECODED name (never by id): the first member of a tier is that tier's lex-min, the canonical-selection order ([D-134](../40_decisions.md#d-134), [I-88](../30_invariants.md#i-88)). Writers keep the order structurally: `setMembersFromNames` (creation) and `unionMemberIdsByName` (merge). |
| `intEqualityLevelsMap` | `map<int64_t, levels>` | Per-pair admission levels, keyed by `packEqPairKey` (unordered id pair). |
| `equalityOriginMap` | `IdOriginMap` (packed origin-interner keys → `(OriginTag, packed deps)` lines) | Provenance — which equalities produced this class. Id form per `D-131`, same `Memory::originInterner` space as `exprOriginMap`, so class↔body history copies are pure id operations. |

The weak-variable record is the packed `Memory::intWeakVariables` (`packStatementKey(variableId, validityId)` keys; the former `ExpressionWithValidity` string set is gone), written at `updateWeakVariables`, probed by `reduceEqClassIds`, filtered at `wipeSubtree` by the low-32-bits validity predicate, swapped at the `destroyGrid` capacity release; its dump section derives byte-identical output by decode + lex-sort. One reset rule for the [name caches](#name-classification--classifyname--scanspecialtokens--eqclassnamecaches): `destroyGrid` resets `nameMap`, so the caches reset with it — ids re-bind there.

Classes are owned per-scope on each `Memory`: `Memory::equivalenceClassesMap` is the cold `TypedColdBlobMap<NameId, EquivalenceClass>` (Batch 3, one canonical blob per class keyed by validity id) — a scope can hold several disjoint classes. Read paths decode via the non-minting `Memory::decodeClassesAt(validityName)` / `decodeClassesById(validityId)` (empty vector = no classes, the defined absent state), or read a single class blob in place through the zero-copy `EquivalenceClassView` (the transient-statification read form — layout-coupled to `serializeEquivalenceClass`, no heap decode; see [I-128](../30_invariants.md#i-128)); write paths (`updateEquivalenceClasses`, the `addEquality` registration site) serialize through `assignClassesById`. Order-sensitive walks — `reactToHypo`, `applyEquiClasses` pass 2, the hashburst dump section — iterate decoded-name lex-sorted snapshots, never id order ([I-84](../30_invariants.md#i-84)). `changedClassesThisStep` carries `(validityId, class copy)`; `eqClassSttmntIndexMapMap` is `(validityId, memberIds)`-keyed (flattened onto a byte-key `ColdHashMap<BytesKeyStore, int>` since Batch 2 — `encodeEqClassKey` packs the validity id + member ids into the byte key; point lookups only, never iterated). Different scopes can have different classes — an equality asserted under a hypothesis does not bleed into the parent unless the hypothesis discharges.

### Complete target-scope class view

Canonical parked-source and completion-record maintenance is delegated to
[`logical_block/rejected_source_memory.hpp`](../../../GL_Quick_VS/GL_Quick/src/logical_block/rejected_source_memory.hpp),
`RejectedSourceMemory`. It uses canonical expression and template interfaces while
the owning block retains receipt, source-lookup and dependency-index storage.
Existing affected-key selection, decoded-key ordering and completion retirement
are unchanged; no canonical rewrite is deferred to a new phase.


`Canonicalizer::canonicalFormAtScope` also owns expression canonicalization,
including equality-form classification and preservation of the applied-equality
justifier tail. Its class-membership, live-representative, local-equality and token
resolution helpers are private. Direct helper tests use a test-only fixture;
production callers request the resulting canonical form through the public operation.
`CanonicalForm`, `ResolvedToken` and `ResolvedStatementTokens` have separate headers
under `logical_block/`. They remain passive arena/span records with the same field
order, initialization and visitor order; no stored pointer lifetime changes.


[`logical_block/canonical_template_matcher.hpp`](../../../GL_Quick_VS/GL_Quick/src/logical_block/canonical_template_matcher.hpp),
`CanonicalTemplateMatcher`, owns canonical admission text and chain construction,
source-template projections and target-scope matching. Its only peer is
`Canonicalizer`; the supplied block retains derived indices and source records.
The public operations retain existing target-scope conditions, key spellings,
projection invalidation and result ordering. Shared chain-renaming operations live
in `DefinitionExpansion`; their heap/span bodies and scratch lifetimes are unchanged.


The scope-building algorithm is owned by
[`logical_block/canonicalizer.hpp`](../../../GL_Quick_VS/GL_Quick/src/logical_block/canonicalizer.hpp),
`Canonicalizer`. It borrows the existing parameters and the integration-task interface for root rechecks; the target `Memory`
retains both class stores, representative records and pending rechecks. Callers
request scope initialization/rebuilding, representative selection, class merging
and held-run flushing through this interface. Inline bodies and execution order
are preserved.

[`logical_block/canonical_run_overlay.hpp`](../../../GL_Quick_VS/GL_Quick/src/logical_block/canonical_run_overlay.hpp),
`CanonicalRunOverlay`, owns the private temporary record columns and scope index.
Its public operations add/find/copy runs and report entry count, scope and record
count. These readers expose no mutable columns. `CanonicalMemberChange` is a
separate passive record; neither extraction changes its layout or scratch lifetime.


`Memory::canonicalEquivalenceClassesMap` is a second, independently owned cold class store. `Canonicalizer::rebuildCanonicalScope` combines the target's application classes with its parent's complete view, closes every overlap transitively, and retains independent ancestor components. Local pair-level precedence and original equality citation scopes are preserved. Previously recorded paths are retained when a component is rebuilt; identical complete runs are not rewritten.

`initializeCanonicalScope` initializes parent scopes first. After ordinary equality, weak-variable and history maintenance, `refreshCanonicalClasses` excludes scopes refuted by `refutedOrBranchAtOrAbove`, then rebuilds the changed live scope and initialized live descendants in depth and decoded-name order — but only when the equality changed the membership of a class at its scope (the class watcher, [D-417](../40_decisions.md#d-417)). An equality between members already in one class records its pair level and origin in the scope's own class and leaves every complete view, representative and weak mark as it was (weak marks are a function of membership); its evidence reaches the complete views at their next rebuild. The rebuilt runs of one refresh are held (`CanonicalRunOverlay`, descendants read their parent's new run from it) and written with one scratch rebuild of the class store (`flushCanonicalRuns`) instead of one pool-tail move per scope; the result is byte-identical. No canonical copy enters the application-class store, its change buffer or its processing positions.

`LbMemory::canonicalMemberRepresentatives` records the last normal and special-name representative for each `(member, scope)`. The separate `canonicalClassChanges` records identity, membership, evidence and special-name-selection changes with bits 1, 2, 4 and 8 respectively. This snapshot detects representative changes even when weak-variable metadata changes before the complete view is rebuilt. Unchanged components contribute no pending changes. Both maps are independent of ordinary multiplication events.

Within `refreshCanonicalClasses`, parked-source and watch identity refresh, together with settled ancestor-negation import, select only members carrying change bit 1: their keys and imported values depend on canonical text. New class members also carry that bit. Evidence-only changes retain these identities without rebuilding unchanged source dependencies or re-importing unchanged negations. Ancestor import retains the original complete member list while statement arrivals await the watch drain, so a newly registered negation is imported at the same class-update seam even when its representative is unchanged. `refreshCanonicalSourceUses` still synchronizes authoritative source writes even when its identity list is empty; `recanonicalizeWatchLedgers` likewise synchronizes goal dependencies before its empty-identity return. All member versions and class changes are still published; processed/branched OR identities and completion receipts retain their existing complete change list, while an OR matching entry is dirtied only by change bit 1 or change bit 16 ([D-498](../40_decisions.md#d-498)). Equality levels, locality and provenance remain available to their original consumers.

Watch identity maintenance ([I-269](../30_invariants.md#i-269)) selects exact `(original expression, effective scope)` records once from `canonicalUses.watches`, then resolves their actual ledger runs. Selection across cores is shared; no full-ledger or per-record linear membership scan is needed at an exact-scope refresh. Selected records retain ledger-key order followed by decoded original-expression order, so canonical identifier minting is unchanged. Each touched run is rebuilt once in its original position. The index's `contains` reader distinguishes live records from retained retired keys; scope wipes, clear/replace and reconstruction keep this membership synchronized with the ledger.

`importAncestorNegationsForClass` ensures the watch index is built, then checks its live original/scope key before canonicalizing an already-imported record. Those records have just passed identity refresh at this boundary. Pending-arrival selection still uses the full component member set, preserving same-boundary import of new negations. `canonicalWatchIdAt` calls the shared `canonicalFormAtScope` chooser and substitution walk in text-only mode: it builds neither pair-level unions nor equality justifiers/locality/citation data. The full evidence mode remains the default for proof-producing consumers, including `canonicalWatchFormAt`; evidence versions, supporting levels and history are maintained in their original owners. OR entry creation and refresh use this text-only mode with the OR equality and inequality opt-in (`OrImplicationProcessor::canonicalOrFormId`). A firing's selection reads a spelling's rewrite locality through `canonicalWatchFormAt`'s lookup mode (`OrImplicationProcessor::rewriteLocalAt`): `canonicalFormAtScope` retains identical justifier text and equality locality while skipping level unions, and the watch adapter skips citation-scope lookup/copies. Its level buffer must be null and empty; returned citation spans are empty. Selected OR firing still uses full proof evidence, and the existing locality/version invalidation rules are unchanged.

`MergeClassAccum` keeps an exact pair-key-to-zero-based-position index on its own page-tier scratch arena ([I-273](../30_invariants.md#i-273)). `levelIndex` probes that derived map and asserts that a hit still names the same key in the flat rows. `setLevel`, `mergeLevelsBBase` and `mergeLevelsFromAccum` register each newly appended row; replacements retain its position, including empty level runs. Shared-key merges retain the accumulator's existing levels (A-wins replacement), while missing keys append in the same source order. No index traversal controls serialization, no new persistent or deloaded state is introduced, and origins are unchanged. The original flat key/start/length/value columns and sorted serializer remain authoritative.

The complete class store and its two small companion maps participate in clone, scope wipe, deload/reload and aggregate release. Their facets use tags 839–846 on the existing main pool. See [I-254](../30_invariants.md#i-254).

`importAncestorNegationsForClass` rejects positive statements and negated equalities while visiting member postings, before allocating candidate keys and sorting them. These are the same immutable shape exclusions applied by the import path; eligible negatives retain their original `(expression, source scope)` decoded order and authoritative ownership. Anchor/watch-shape checks, canonicalization and publication remain in that ordered path. No statement or dependency posting is removed from its shared index.

Canonical identity readers use the complete view at the target scope: expression entry, compact carriers, OR and watch identities, admission templates and values, construction receipts, goals and representative filtering. The existing application store continues to drive multiplication and weak-variable bookkeeping.

`CanonicalUseIndex` stores original/current-name and value dependencies under stable logical record keys. Equality versions and posting revisions select affected records; unchanged components retain their entries. `CanonicalTemplateIndex` projects only requested expression shapes and source scopes into a target scope. Matching returns original owning keys, including when ancestor-owned records are compared in a descendant. It neither rewrites an ancestor through child equalities nor admits a new scope route. Literal goal prefixes in subproof payloads are preserved. In `matchingCanonicalTemplates`, identical original and canonical text share one dependency-indexing pass; different forms retain both passes so every original/current token still invalidates its dependent projection.

Comparison refresh is queued at mutation boundaries. `rebuildCanonicalScope` sends each new or changed normal representative (change bit 1) to `CanonicalUseIndexes::invalidateComparisonMember`; each comparison store uses its original/current-token postings to enqueue affected `(original, target scope)` projections. A new source key or requested shape queues its previously unseen projections directly. The derived pending store is an `OrPostingIndex` keyed by target scope, so `matchingCanonicalTemplates` visits only that scope's queued identities and rebuilds them in the same decoded-original order. It no longer scans every class member before and after each lookup or scans a global pending set. Evidence-only changes retain the comparison forms, whose contract depends on canonical text; evidence consumers keep their independent versioned updates. OR canonical entries follow the same rule and store no evidence: `refreshCanonicalClasses` dirties an entry only for members with change bit 1 or change bit 16 (the recorded locality of the member's rewrite equality changed, which a firing then reads afresh) at the entry's scope ([D-498](../40_decisions.md#d-498), [D-547](../40_decisions.md#d-547)). Invalidation coalesces repeated changes, including a representative that changes back before the next query. Canonical release clears the queues with the other comparisons. Scope closure retires only closed target projections and closed source-shape postings; surviving forms, token dependencies, versions and queued invalidations remain warm. Raw eviction preserves them and queue mutation dirties the raw cache image. The authoritative map still filters retired source owners at lookup.

Each dependency edge carries its insertion revision under a logical `(token, record)` key. When equality is unchanged, selection visits only the new posting prefix, including edges reinserted for replaced records; an erasure alone schedules no surviving record. Equality-version changes still select all live dependents. `OrPostingIndex::forEachWhile` preserves newest-insertion order across erasure and permits stopping at the first previously observed edge. Edge revisions use the same derived pool storage and raw-image dirtiness/lifecycle as the other index state; they add one 20-byte key/value payload per distinct recorded edge, plus the existing cold-table overhead.

Boolean presence reads first test the original and target-canonical text against the exact source owner. A positive key answers without constructing comparison projections. Pure algebra admission probes use an exact canonical key only when its existing generation, secondary-variable and demand-kind gates pass; an ineligible exact hit still searches equivalent originals. Pure integration probes retain typing and negated-equality exclusions. Committed reads continue to process every matching original and its values.

`updateAdmissionMap` distinguishes fixed arguments from numbered placeholders before canonicalizing a chain. In a mixed rule chain, `u_` identifies fixed arguments; after renaming, `remainingArgs` identifies them. Only these arguments use complete-class representatives. The existing `renamingChainScratch` then separates placeholders from the selected fixed names, including when a representative has the same spelling as a placeholder. An already renamed chain may retain fixed-argument metadata absent from its remaining premises: these names stay in the admission value but do not constrain local alpha-renaming. A scope equality must not bind a rule placeholder or erase its marker demand.

`canonicalAdmissionChain` supplies the same binding distinction to stored algebra-value maintenance, target-scope comparison and recursion replay. Bare nonnumeric names remain concrete; numbered names outside the fixed set remain placeholders and do not contribute equality-dependency postings. It reports the existing Anchor exclusion without changing those values. Recursion and integration replay identify each marker group from the incoming expression before normalizing its fixed arguments. Already shared occurrences remain one group; equality normalization cannot capture another argument into that group. Applied integration values still use complete scoped identities. The task ledger records validation and runtime-gate evidence.

Construction completion is keyed by `(canonical_S(E), S)`. Algebra and integration completion bits remain distinct. Actual queued raw carriers retain pending and carrier-mode bits, with an exact index from canonical identity and scope to those carriers. Watches own their dependency index independently of expression-row retention. These derived indexes use the existing main pool, survive raw eviction, and are cleared at canonical image load/release and discharge. `CanonicalUseIndexes::eraseScopes`, called by `Memory::wipeSubtrees`, retires closed-scope records, edges, observations, pending carriers and comparisons. Surviving revisions and initialized-ledger flags remain valid; authoritative map synchronization rechecks keys after compaction without decoding unchanged records. Private token and shape names remain interned until full release. Their private dirty flags also participate in the raw-image reuse check: a cache-only change requires a fresh raw snapshot without dirtying the logical canonical image. Flags are acknowledged only after the raw dump completes.

`queueCanonicalWake` coalesces wake kinds by original template and target scope during the canonical consumer pass. `applyEquivalenceClassToCompactImplications` collects original carrier keys while removing their ownership. Its dependency selection preserves `isCompactImplicationSpan` eligibility: the shared ownership store also contains other expansion carriers, which this hook does not process. `finishCompactRemovals` finishes one batched removal across the complete pass; `installCanonicalCompacts` then installs replacements in decoded expression/scope order. `drainCanonicalWakes` first matches newly parked sources against settled demands, then releases wakes in decoded template/scope order after those replacements exist. Outside this pass the ordinary immediate wake path remains. Before matching parked algebra or integration sources, `hasPotentialParkedDemand` rejects only impossible demand routes. Algebra retains local/ancestor demand and the single-input operator route; integration retains its exact-scope prefixed instruction map and admission set. Positive candidates still execute all ordinary source-specific generation, secondary-variable, completion and history checks. Deferred wakes are queued before this filter, so unsettled canonical maintenance cannot lose a later demand. Anchor-excluded values survive another value's same-key rewrite. Deferred source and OR rewrite history cites equalities at their original scope through `findEqualityCiteScope`.

`CANONICAL_CLASS_VIEW`, `CANONICAL_DEPENDENT_IDENTITIES`, `CANONICAL_DEPENDENT_RECORDS` and `CANONICAL_MATCH_REBUILD` separate class calculation, dependent work and cache rebuilding in the existing runtime tracker. The task ledger owns validation status and the current-main runtime gate.

### Name classification — `classifyName` / `scanSpecialTokens` / `EqClassNameCaches` ([D-134](../40_decisions.md#d-134))

Canonical selection uses four classes: ordinary names, forced witnesses, ordinary integration witnesses, then algebraic witnesses. Within integration names, numeric depth is compared first; all remaining ties use decoded lexical order. `memory_infra/witness_names.hpp` / `compareWitnessNames` is the common ordering authority. `memory.hpp` / `classifyName` and the special-token cache retain forced and integration kinds separately while sharing the disintegration-token run. `firstSpecialMemberId` uses the same preference among special names; `chooseCanonicalId` includes ordinary names and excludes scope-weak members. Goal sanitization uses the same special-name comparator. Class members remain stored in decoded lexical order, so choosing the shallowest integration name requires a comparison scan rather than taking the first lexical member. No interner identifier determines rank.

The caches are pure functions of immutable NameMap strings and survive scope teardown. The shared token walk still distinguishes substring scanning from whole-argument substitution, preserving prefix and copy-suffix boundaries. Cache fills remain confined to block-exclusive phases.

---

**OR equality refuters.** `canonicalFormAtScope` and `canonicalWatchFormAt` preserve standalone equalities and inequalities by default. The OR matching entries explicitly include those shapes: a refuting fact and an equality-valued OR leaf must receive the same substitutions at the same effective scope. Only a selected firing asks `writeWatchTwinCanonical` to materialize that equality or inequality view with an `equality1` bridge. Equality registration and the ordinary negation/goal watch paths retain their existing defaults. See `logical_block/or_implication_processor.hpp` / `ensureOrCanonicalCache` and `logical_block/or_implication_processor.hpp` / `evaluateOrImplication`.

## Registration — when a class forms

`addStatement` (see [prover chapter](../10_pipeline/04_prover.md#addstatement-central-ingestion)) handles equality ingestion. When `(=[a,b])` arrives:

1. `addEquality` registers the original statement; its mirror branch runs only when `a!= b`. A reflexive equality `(=[a,a])` keeps its statement row and its history line, but it only joins an existing class: when no application class at its scope or a strict ancestor holds `a` (`Memory::applicationClassHoldsName`), `updateEquivalenceClasses` returns right after the history write and founds no class ([I-295](../30_invariants.md#i-295), [D-477](../40_decisions.md#d-477)). Every stored class therefore has two or more members; the application write-back in `updateEquivalenceClasses`, the complete-view rebuild in `rebuildCanonicalScope` and the storage door `Memory::assignClassesById` assert it (an empty or one-member class has no use anywhere, test fixtures included), and the alias probes (`checkNecessityForEquality`, `transferUnaryIntegrationDemands`) rely on it.
2. If neither `a` nor `b` is in an existing class, create a new class `{a, b}`.
3. If one of `a` or `b` is in an existing class `C`, add the other to `C`.
4. If both are in existing classes `C_a` and `C_b`, merge them — the union becomes the new class.

Per-class levels are maintained to respect the admission-map gating — a fact derivable through equality at level `L` is itself at level `L`.

### Origin tracking and the mail-sync rule

Each class carries `equalityOriginMap` — provenance for the equalities the class represents. The map is populated from three sources:

- The incoming equality's `origin` parameter, recorded by `updateEquivalenceClasses` at [`logical_block/equivalence_processor.hpp::updateEquivalenceClasses`](../../../GL_Quick_VS/GL_Quick/src/logical_block/equivalence_processor.hpp).
- Cross-pair `equality2` records emitted by `mergeTwoEquivalenceClasses` ([`canonicalizer.hpp`](../../../GL_Quick_VS/GL_Quick/src/logical_block/canonicalizer.hpp), `Canonicalizer::mergeTwoEquivalenceClasses`) — see *Cross-pair `equality2` emission* below.
- **Mail-bulk-merge sync** in the absorb recipe of [`logical_block/logic_block_operations.hpp::standardProcessing`](../../../GL_Quick_VS/GL_Quick/src/logical_block/logic_block_operations.hpp). Immediately after the LB's bulk-merge of the drained mail's `exprOriginMap` into `body.exprOriginMap`, every positive 2-arg equality entry `(=[a,b]) @ V` whose vars are already class-bound at `V` has its mail origins additionally registered in that class's `equalityOriginMap` (string mail record encoded via `addOriginEncoded`). Equalities whose vars are not yet class-bound are picked up later via `updateEquivalenceClasses`'s seed when their own absorption fires. An ancestor's merges arrive this way: the class-canonical sweep drops a merging equality from the outgoing delta, and `EqualityPropagation::shipDroppedMerge` mails it as a statement (experiment [D-544](../40_decisions.md#d-544)).

The mail-sync exists because the class's `equalityOriginMap` is the source of truth that the merge cross-pair logic consults: without sync, mail-arrived derivations are invisible to the class, and the merge re-derives them via `equality2` transitive routes — producing cycles when multiple bridges exist. See [03_mail_system.md](03_mail_system.md#the-cycle-boundary-protocol) for the producer side and [50_gotchas.md](../50_gotchas.md) for the historical chapter-22 / theorem-12 cycle that motivated this rule.

---

## Propagation — the mirror emission

When `(=[a,b])` is admitted, the prover automatically emits `(=[b,a])` (guarded by `a!= b`, see [I-9](../30_invariants.md#i-9)). Equality is thereby made symmetric *by construction* at the statement level, without requiring a symmetry rule in every implication table. This mirror operation does not establish reflexivity.

Argument substitution (the `equality1` tag) materializes encoded statement variants in `logical_block/equivalence_processor.hpp` and `applyEquiClasses` for signed cores used as premises. Unused encoded rows remain stored without class updates. Goals and negated statements are watched through the canonical ledgers; `materializeWatchHits` supplies the exact goal or contradiction twin before discharge ([I-246](../30_invariants.md#i-246)). Request generation consumes the resulting encoded rows rather than performing an additional equivalence-substitution pass.

---

## Transitivity

`equality2` (transitivity of equality) is captured through the class structure itself. If `(=[a,b])` and `(=[b,c])` are both admitted, both `a` and `c` end up in the same class — the transitivity conclusion `(=[a,c])` is a direct class-membership query, not a derived rule.

In the processed proof graph, the verifier still expects an explicit `equality2` tag row when transitivity is the justification for a derivation step. The checker walks the origin chain to verify the two equality premises that produced the transitive conclusion.

---

## Negated equality — the asymmetric expansion

`addStatement` has special handling for `!(=[a,b])`. When this arrives at scope `V` with existing equivalence class(es) touching either arg, `applyEquivalenceClassToNegatedEquality` ([`logical_block/equality_propagation.hpp`](../../../GL_Quick_VS/GL_Quick/src/logical_block/equality_propagation.hpp)) emits sibling inequalities:

- For each `c ∈ class(a) \ {a}` — emit `!(=[c, b])`.
- For each `d ∈ class(b) \ {b}` — emit `!(=[a, d])`.

**The symmetric cross-product is deliberately not emitted.** That is, the prover does *not* additionally emit `!(=[c, d])` for `c ∈ class(a)`, `d ∈ class(b)` — even though that combination is semantically derivable. See [I-12](../30_invariants.md#i-12).

Why skip the cross-product: it would blow up combinatorially (|class(a)| × |class(b)| emissions per input). The one-sided expansions are sufficient — anything the cross-product would derive is derivable by two one-sided steps, one for each arg.

**Two expansion moments ([D-226](../40_decisions.md#d-226)).** The expansion runs when the negation arrives (`addStatement`, against classes existing at that instant) AND when a delta class touching either arg later forms or grows (`applyEquiClasses` Pass 1 routes pre-existing negated equalities to the same expander; Pass 2 still skips them — a new statement always had its arrival-time expansion). Before the second moment existed, a negation inserted before its variable's class formed was never expanded: the `_orint_` peer negations arrive at branch creation, the interval witness-binding equalities arrive bursts later, and the K mutual-exclusion rules keyed to the witness form could then never fire — the FTA rung-2 stall. Pass 1's pre-gate is the same three-way comparability the positive-statement pass uses (own scope, strict ancestor, strict descendant — [D-271](../40_decisions.md#d-271)).

**Scope symmetry ([D-271](../40_decisions.md#d-271)).** The expander's class walk covers all three scope directions, mirroring `applyEquivalenceClass`'s D-33 comparability rule. Classes at the negation's own scope and at strict ancestors emit the sibling inequalities at the negation's scope (the two historical directions). Classes at strict DESCENDANTS of the negation's scope — walked in decoded-lex order over `equivalenceClassesMap`'s bucket validities, never id order ([I-84](../30_invariants.md#i-84)) — emit at the CLASS's scope, i.e. `deeperOf(negation scope, class scope)`, exactly where `applyEquivalenceClass` deposits its descendant-direction rewrites. Before this direction existed, an ancestor-scope negation was invisible to a deeper class: only a branch-scope COPY of the negation triggered branch-scope expansion, which made those copies load-bearing and blocked the ancestor-known dedup contract (a dropped branch copy would have silently lost the branch's expanded variants).

**Depth-first continuations and refuted scopes ([I-274](../30_invariants.md#i-274)).** `applyEquivalenceClassToNegatedEquality` stores suspended expansions in a page-backed `PagedVector` on the existing worker general scratch arena. Each frame keeps its scope/class/member cursors, source levels, owned source text and post-child publication. `addStatement` registers the next variant through its existing door and reports whether it reached the expansion point; the loop then pushes a child frame. It completes that child before publishing the parent's output row and resuming the parent scan. Self scopes, ancestors, lexically ordered descendants, classes and one-sided members retain their existing order. A large live closure consumes arena frames rather than native call-stack depth. Class-blob scratch stays live until its frame finishes; string scratch rewinds after its child and publication finish.

The existing `refutedOrBranchAtOrAbove` predicate runs before a target scope is scanned and again before its variant is admitted, including pending-dead ancestors. Canonical refresh uses the same predicate when choosing descendants. These guards do not retire scopes or remove proof history: `standardProcessing` still reaches the existing retirement drain, and its history-preservation rules remain in place. The phase-entry ignored-scope snapshot alone does not describe retirement staged during the current step.

**Citation scope of the justifying equality ([D-227](../40_decisions.md#d-227)).** Every equivalence-class history line cites each justifying equality at the scope where that equality's `exprOriginMap` row actually lives, never blindly at the emission or class scope. `applyEquivalenceClass` probes non-minting at the class scope first and then its strict ancestors deepest-first because a descendant class can absorb ancestor-class content. The negated-equality expander uses the same order, additionally considering the negation's own scope when it lies between a descendant class and their shared ancestors. `mergeTwoEquivalenceClasses` applies the rule independently to both bridge dependencies of each `equality2` row, so a branch-local bridge half and an inherited ancestor half retain their different scopes. The chapter walker resolves a dependency at its cited validity and its ancestor lift refuses to cross an `_orint_`/`_ordis_` boundary, so a branch-scope citation of an above-boundary equality could never resolve. The verifier accepts each equality dependency at a strict ancestor of the derived row. A probe that misses everywhere asserts at the emission (Rule 19).

---

## Per-step cleanup placement and `intKnownStatements` immortality ([D-93](../40_decisions.md#d-93), [I-58](../30_invariants.md#i-58))

The expression-side cleanup `logical_block/equality_propagation.hpp::cleanUpExpressions` runs **deferred**, from the Step-4b block at `logical_block/logic_block_operations.hpp::standardProcessing` immediately after `applyEquiClasses(memoryBlock)`: one call per step sweeps every affected scope collected from complete-class changes and application deltas, including affected descendants, in first-occurrence order ([D-405](../40_decisions.md#d-405)). The two pre-fix inline call sites — `updateEquivalenceClasses` post-merge and `addStatement` isEquality-suffix — are retired; comments at those former sites point at the Step-4b block.

The Step-4b block runs `cleanUpExpressions` alone, once per step over the unique validities. The admission maps need no Step-4b sweep: since [D-106](../40_decisions.md#d-106) both admission hooks canonicalise inline (drop the changed key, insert the canonical form the moment a class rewrite touches it), so no non-canonical admission key survives to Step-4b. The former sweep pair `cleanUpAdmissionMap` / `cleanUpAdmissionMapIntegration` was deleted as dead code (2026-07-02, user-authorized).

Why deferred. Pre-fix, the inline `cleanUpExpressions` call inside `updateEquivalenceClasses` ran **before** `applyEquiClasses` had a chance to read the non-canonical source row in `intEncodedStatements` and emit the canonical-form rewrite. Result: the source row was dropped without a canonical replacement, and any rule whose firing depended on having a same-witness pair across two `in3` (or similar) rows lost the input it needed. The right-cancellation theorem `(>[1..6](AnchorPeano[1..6])(>[7,8,9](in3[7,8,9,4])(>[10](in3[10,8,9,4])(=[7,10]))))` was the canonical regression — `(in3[10,rec,it_0_lev_3_8,4])` got dropped before `(in3[10,rec,it_0_lev_3_32,4])` could be generated. Deferring the cleanup lets `applyEquiClasses` see the source row, emit the canonical rewrite, *then* the sweep drops the now-redundant non-canonical original. See [D-93](../40_decisions.md#d-93) for the full reproducer.

What `cleanUpExpressions` modifies (one pass per call over all scheduled scopes: each registry is walked once to group the rows of the scheduled scopes by scope; each scope's complete view (`canonicalEquivalenceClassesMap`) is viewed once and its own rows judged in registry order: a row is dropped iff one of its scanned special tokens is a member of a class of the view other than that class's first special member — the verdict of `filterIterations` over the view's classes, read token by token from the special half of `canonicalMemberRepresentatives` ([I-287](../30_invariants.md#i-287)); the first dropping class (class order) is located only for a dropped delta row. A row is judged only by its own scope's view, so the kept rows, their order and the mail order of dropped delta histories equal one full-registry sweep per scope in the same order; the rebuild and the waterline repair then run once per call):

- `mb.intLocalEncodedStatements` + `mb.intLocalEncodedStatementsDelta` + the packed-key `mb.intLocalEncodedStatementsSet` (rebuilt from the kept rows) — filtered (drop where the row carries a special member of a class of its scope's view other than that class's first special member, the id-overload `filterIterations` verdict, read from the recorded representatives).
- `mb.intEncodedStatements` — same filter.

What `cleanUpExpressions` deliberately does **not** modify ([I-58](../30_invariants.md#i-58)):

- `mb.intKnownStatements` — kept immortal so that `addExprToMemoryBlock`'s **Site F** ancestor-scan duplicate-suppression gate (the `known` bit) catches re-arrivals of the same statement via mail. Without this, `addEquality`'s own `registered`-bit skip lets the runtime containers stay empty while the registration record is non-empty, and `addStatement`'s post-loop `intStatementLevelsMap` assert fires with a row that's registered but absent from the runtime containers.
- `mb.intStatementLevelsMap` — a dropped statement's levels row is kept for the same reason ([D-247](../40_decisions.md#d-247)): the `registered` bit makes every registration path skip row re-creation, and the negated-equality expansion's emit gate reads the row as its permanent already-emitted memory. An erased row with a surviving `registered` bit is re-emittable but never re-registrable — the class expansion regenerates dropped sibling variants without bound (stack overflow), and a dropped equality's re-arrival trips the assert above.

Only `Memory::wipeSubtree` (scope wholesale teardown), `memory.cpp::eradicateImplicationFromLB` (specific-implication wholesale teardown), and `logical_block/equivalence_admission_rewriter.hpp::resetResentExpressionRegistries` (single resent-compound teardown, [D-106](../40_decisions.md#d-106)) may legitimately erase rows from `intKnownStatements` — each retires a whole item, removing both membership bits at once (the CE teardown `releaseCEBatchMemory` additionally clears only the `registered` membership). Any new caller of `intKnownStatements.erase(...)` outside those sites is a bug; see [I-58](../30_invariants.md#i-58)'s *How to spot* / *How to fix on violation* clauses.

The newStatements-pair filter (the third operation pre-fix `cleanUpExpressions` performed — dropping non-canonical pairs from the per-equality return vector) is retired: with the L5 id-form flip `cleanUpExpressions` no longer takes a `newStatements` parameter at all (its Step-4b caller passed an empty throwaway and ignored the return, so the branch was already dead). Its live work is the `intEncodedStatements` / local-registry canonicalization (the levels map is deliberately untouched, see above).

---

## Per-deposit emission block

Every emission from `applyEquivalenceClassToNegatedEquality` goes through the standard local-deposit block inside `addStatement`:

```
intStatementLevelsMap[intKey] = levels
upsertStatementKey(intKnownStatements, intKey, ...)   // registered / known bits
intEncodedStatements.push_back(intRow)
if (local) {
    intLocalEncodedStatementsSet.insert(intKey)
    intLocalEncodedStatements{,Delta}.push_back(intRow)
}
newStatements.push_back(encoded)
if (validityName == "main")
    mailOut.statements.push_back(expr, levels)
```

So each sibling inequality enters the same admission pipeline as a freshly-derived expression — it is not short-circuited.

**Precondition assert:** entry requires `isNegatedEquality(expr)`. Callers must gate on the same check. Violating this assert has been a historical bug-source; the user has explicitly required the assert stay intact (memory: ).

---

## Equivalence-class visibility

Per memory file — on the migration and forward — equivalence classes defined in scope `S` become visible in every scope deeper than `S`. A class registered in `"main"` is visible everywhere; a class registered in an OR-branch scope is visible only in that branch and its sub-scopes.

The classification algorithm uses `comparable(scope, classScope)` — see [`20_core_concepts/04_validity_stack.md`](04_validity_stack.md) for the depth-comparison mechanics.

---

## Cross-scope application — bidirectional ([D-33](../40_decisions.md#d-33))

Equivalence-class application now runs in both directions whenever the class's validity scope and the expression's validity scope are comparable:

| Class scope | Expression scope | Deposit scope | Direction |
|---|---|---|---|
| `S` | `S` | `S` | same-NS (legacy) |
| `S_a` (strict ancestor of `S_d`) | `S_d` | `S_d` | class shallower (legacy) |
| `S_d` (strict descendant of `S_a`) | `S_a` | `S_d` | class deeper (NEW — D-33) |

The deposit scope is `deeperOf(class.scope, expr.scope)` — see [`memory.hpp::NameMap::deeperOf`](../../../GL_Quick_VS/GL_Quick/src/memory.hpp). The new direction is sound under descendant-inheritance: a fact at `S_a` is observably true at every descendant of `S_a`, including `S_d`. A class registered at `S_d` can therefore legitimately rewrite the inherited fact and produce a fact at `S_d`.

The original at `S_a` is **never overwritten** — the rewrite is purely additive. Both copies coexist:

| | Lives at | Touched by class application? |
|---|---|---|
| Original `(P[a]) @ S_a` | `S_a` only | No |
| Rewrite `(P[b]) @ S_d` | `S_d` only | Yes — newly inserted |

This is what unblocked FTA-rung-1 §9b: a class registered at an OR-disintegration branch scope (descendant of `main`) rewrites a ground `preorder` fact at `main` and deposits the result at the branch.

### Trigger sites

Equivalence-class application fires from a **single driver, `applyEquiClasses`** ([`logical_block/logic_block_operations.hpp`](../../../GL_Quick_VS/GL_Quick/src/logical_block/logic_block_operations.hpp)), called by `standardProcessing` after mail absorption in phases 1 and 3. It replaces the three former inline sites (the `updateEquivalenceClasses` merge-postscan, the `addStatement` per-statement block, and the `addStatement` fixpoint), which were consolidated into it: `updateEquivalenceClasses` still performs the class MERGE and records each changed class in `Memory::changedClassesThisStep`, but the per-statement APPLICATION loop moved out of the add path into the driver (recorded in the `addStatement` in-code comment).

Once per step, the driver runs its fixpoint rounds. Each round first runs the hooks (the rejected / admission / compact reconcilers) for every live delta class — the classes whose membership changed during the just-finished mail-absorb pass, keep-last deduplicated, dead-scope skipped — and for every non-delta class, class-major as before; then ONE statement-major batch replaces the former per-class registry sweeps ([D-334](../40_decisions.md#d-334)):

1. **Class table** (`buildEquiClassTable`, built once per apply — the class state is fixed inside an apply, no path in it admits a positive equality): every class of the LB with its zero-copy blob view, its strong-member run (`reduceEqClassIds` once per class, not once per pair), an application rank (delta entries in delta order, then the former Pass-2 order), a due-from position (the delta entry's `startIdx`, or the persistent waterline of `eqClassSttmntIndexMapMap`), and the token postings `member id → classes containing it` (`EquiClassTable`, the `ReverseArgsIndex` chain idiom). A delta entry Pass 1 skipped as dead-scope is left out.
2. **Affected rows** (`collectEquiAffectedRows`): one pass over the registry's int columns — a row is due when a comparable class (same scope, strict ancestor, strict descendant) is due for one of its `argId`s; anchors and positive equalities are skipped, negated equalities go to their own list.
3. **Closure per row** (`stageEquiClosure`): the row and its own staged products form a queue; every comparable class containing one of an item's argument ids runs `enumerateEqClassRewrites` on it, in rank order; the deposit scope is the deeper of class and item scope; the former sink's filters run in order — already staged at that scope, known at that scope, known at a strict ancestor (in the registry or among the products staged so far; the I-174 signal survives the refusal), the iteration cap, the `countPatternOccurrences` cap, an axed variable. A surviving product joins the queue, so the closure is the fixpoint the registry-wide loop reached, confined to one statement. Levels are the item's run ∪ the rewrite's pair levels (`mergeLevelRuns`). Nothing in the closure mints the NameMap (asserted). The products ride `StagedEquiProducts` — the temporary list keyed by `(scope id ++ text)`, O(1) append, nothing shifts.
4. **Drain** (`drainStagedEquiProducts`): the staged products commit in staging order through the same doors as before (levels row, known + local bits, local set, the three registry pushes), each with one `equality1` row under the I-34 gate — source = the immediate source (registry row or earlier product) at its scope, the class's pairs cited at `findEqualityCiteScope` — then one `ordisMerge` pass over the committed rows in their existing commit order.
5. Every class's due-from position then moves to the registry size; the negated equalities a delta class is due for run `applyEquivalenceClassToNegatedEquality` once per row (it covers every scope direction itself and self-dedups; the former loop ran it once per (delta class, row) pair). A round that grew the registry is followed by another; at the end of the apply every table class's waterline in `eqClassSttmntIndexMapMap` is set to the final registry size.

**Subtree cleanup preserves processed positions.** `Memory::wipeSubtrees` records the old positions removed from `intEncodedStatements` during its stable compaction and calls `StatementRegistryOperations::repairEqClassWaterlines` once. Every surviving class boundary decreases by the number of erased positions strictly before it. Local-list compaction does not adjust it again. A fact appended after a scope closes must remain beyond the processed prefix and receive the ordinary equality closure, including independent substitutions into repeated argument positions.

The three scope directions of a class × statement pair, the deposit scope being the deeper of the two ([D-33](../40_decisions.md#d-33)):

1. **same-NS** — the class scope equals the statement's scope.
2. **shallower (ancestor)** — the class scope is a strict ancestor of the statement's; the deposit is keyed at the statement's (deeper) scope.
3. **deeper (descendant)** — the class scope is a strict descendant of the statement's; the deposit lands at the class's deeper scope.

Products commit inline in the drain, never through the door ([I-217](../30_invariants.md#i-217) §5); the negated-equality expansions route into the single `newStatements` channel (below). The retired pairwise driver `applyEquivalenceClass` survives as the test oracle `applyEquivalenceClassOracle` (`tests/test_equi_reshuffle.cpp`), and the batch is gated against it on twin LBs.

**Delta-tracker lifecycle ([I-215](../30_invariants.md#i-215), [D-323](../40_decisions.md#d-323)).** `changedClassesThisStep` is cleared ONLY at `standardProcessing`'s tail (Step 7), after its three consumers — the driver's Pass 1, the deferred `cleanUpExpressions` sweep, and the [I-194](../30_invariants.md#i-194) recursion-products closure — have read it. There is no entry clear anywhere, so a delta minted OUTSIDE a `standardProcessing` call (LB seeding via `addTheoremToMemory` — variable-copy axioms, equality premises — and the post-join barrier deposits) survives to the LB's next call and is back-applied to every pre-existing statement by Pass 1. This matters because the Pass-2 waterline of a fresh class is seeded to the then-current registry size: Pass 1 is the only path that ever multiplies a class into the statements that preceded it. The one sanctioned in-call loss: a class minted after Step 4 by `dischargeContradiction`'s head deposit is wiped unconsumed by the same call's Step 7 — harmless, since every reacting role deactivates the LB in the same act ([I-112](../30_invariants.md#i-112)).

**Keep-last delta dedup ([D-261](../40_decisions.md#d-261)).** `changedClassesThisStep` holds one snapshot per class CHANGE, so a class updated several times in one step appears several times. Pass 1 consumes the delta keep-last: an entry is skipped when a later entry carries the same (validity id, member-id run). The verdicts come from one linear per-round pre-pass — `fillDeltaKeepLastFlags` ([`logical_block/equivalence_processor.hpp`](../../../GL_Quick_VS/GL_Quick/src/logical_block/equivalence_processor.hpp)), a descending walk interning packed `(validity, members)` keys into a per-round scratch set (a pairwise O(n²) first form ground the incubator's chatty-class steps and was replaced on sight). The surviving last snapshot carries the most complete pair-level state, and the buffer itself (push site, deload bytes) is untouched.

**Dead-scope skip ([D-262](../40_decisions.md#d-262), [I-181](../30_invariants.md#i-181)).** Before applying a delta class, Pass 1 resolves the class's validity through `refutedOrBranchAtOrAbove` ([`logical_block/case_processor.hpp`](../../../GL_Quick_VS/GL_Quick/src/logical_block/case_processor.hpp)): a class at or below an `_ordis_` branch whose asserted disjunct is already refuted at the branch scope or an ancestor (the [D-242](../40_decisions.md#d-242) `ordisMerge` staging predicate, probed non-minting) is skipped, and the refuted branch is staged into `pendingDeadOrBranches` so the same burst's `drainDeadOrBranches` retires it. Everything such a class would rewrite is ex-falso content inside a subtree the burst wipes anyway; back-applying it across the registry was the A20 runtime explosion (~294 s of one burst inside a doomed `rec = 0` case split). The skip never suppresses application in a live branch — a nonzero verdict requires the machine's own refutation evidence, and every skip stages the branch it skipped for unless an earlier drain already retired it (its scope is in `intValidityNamesToFilter`; staging it again would retire it twice, [I-172](../30_invariants.md#i-172)).

### Single-channel deposit ([I-25](../30_invariants.md#i-25))

Every deposit — same-scope, shallower-class, deeper-class — flows into the **single** `newStatements` channel: since the L5 id-form flip a caller-owned `PagedVector<IntEncodedExpr>&` out-param (`addStatement` returns `void` — PagedVector is non-movable, so it cannot be returned) on the per-slot `genScratchArenas`, holding one id-form row per deposit whose decoded `validityId` is the deposit's actual scope (see [I-25](../30_invariants.md#i-25)). There is no separate sink for cross-scope deposits.

The kernel's post-`addStatement` loop ([`logical_block/statement_memory.hpp`](../../../GL_Quick_VS/GL_Quick/src/logical_block/statement_memory.hpp)) consumes the buffer in emission order, decodes each row's `(originalId, validityId)` at the edge, and uses the decoded scope for the `intStatementLevelsMap` lookup, admission-map updates, `toBeProved` discharge, and validity-name promotion. Cross-scope deposits therefore go through the same discharge logic as same-scope deposits — the only difference is which scope drives the lookup.

This routing replaced an earlier WIP design (the original D-33 commit, prior to the routing fix) where cross-scope deposits went into separate sinks (`crossScopeSink`, `descendantSink`, `ancestorSink`) to avoid tripping the kernel's same-scope assert. That design caused cross-scope deposits to bypass `toBeProved` discharge entirely — facts were inserted at the right key but matching `toBeProved` entries never fired. The single per-deposit-scope channel closes that gap.

### What the rejected-integration map does NOT do

`applyEquivalenceClassToRejectedMapIntegration` does NOT admit the descendant direction (class strictly deeper than rmi entry). The rmi map is a deferred-match registry — its entries represent auxy integrations whose preconditions live in the world visible at the entry's scope. A class at a strictly deeper scope is invisible there, so it carries no information for that entry's deferred match; rewriting and erasing the entry would destroy the original revival path without compensating at the descendant. See [D-33](../40_decisions.md#d-33).

---

## Cross-scope class merge ([D-44](../40_decisions.md#d-44))

`updateEquivalenceClasses` ([`logical_block/equivalence_processor.hpp`](../../../GL_Quick_VS/GL_Quick/src/logical_block/equivalence_processor.hpp), `updateEquivalenceClasses`) absorbs same-NS classes that overlap `eqArgs` (the new equality's args) into a fresh `mergedClass` at `validityName`, then **also iterates strict ancestors** of `validityName` and absorbs any ancestor classes that overlap `eqArgs`. The ancestor classes themselves stay UNCHANGED at their scope ([I-31](../30_invariants.md#i-31)).

| Class scope | Equality scope | Resulting merged class scope | Ancestor class touched? |
|---|---|---|---|
| `S` | `S` | `S` (replaces same-NS originals) | n/a — same-NS |
| `S_a` (strict ancestor of `S_d`) | `S_d` | `S_d` (additive — ancestor's vars/levels/origins copied in) | **No** — `equivalenceClassesMap[S_a]` and `eqClassSttmntIndexMapMap[S_a]` untouched |
| `S_d` (strict descendant of `S_a`) | `S_a` | not merged (out of scope) | n/a — descendant class invisible at `S_a` |

Soundness. A class at `V_a` is observably visible at every descendant of `V_a` (every pair `var ≡ var'` in the ancestor class holds at every descendant). An equality `(=[a,b])` admitted at `V` (descendant) with `a ∈ C_a` therefore propagates `b ≡ all of C_a` at `V`. Adding `C_a`'s members to `mergedClass @ V` is locally sound at `V`. The original `C_a @ V_a` retains its semantics — the new equality is invisible at `V_a` (same-or-deeper visibility rule), so `V_a`'s class must not be modified.

Mirrors the additive principle from [D-33](../40_decisions.md#d-33) (cross-scope deposits never overwrite the original at the ancestor) and [D-43](../40_decisions.md#d-43) (rmi rewrites are additive at the new key). All three — apply-side ([D-33](../40_decisions.md#d-33)), rmi ([D-43](../40_decisions.md#d-43)), and merge-side ([D-44](../40_decisions.md#d-44)) — share one rule: ancestor scope state is never written by descendant-scope work.

The descendant direction (class strictly deeper than `validityName`) is **not** admitted, by symmetry with [D-33](../40_decisions.md#d-33)'s rmi exclusion: a class at a deeper scope is invisible at the equality's scope, so it carries no information for an equality admitted at the ancestor.

The ancestor pass uses the same primitive — `mergeTwoEquivalenceClasses` ([`canonicalizer.hpp`](../../../GL_Quick_VS/GL_Quick/src/logical_block/canonicalizer.hpp), `Canonicalizer::mergeTwoEquivalenceClasses`) — that the same-NS pass uses. The function takes `classB` by `const&`, so the ancestor class is read-only by construction. The bridge invariant (`commonArg ∈ eqArgs ∩ classA ∩ classB`) is preserved because the overlap test is against `eqArgs` directly.

---

## Cross-substitution `equality1` emission gating ([D-48](../40_decisions.md#d-48), [I-34](../30_invariants.md#i-34)) — producer-side redundancy guard

> **Status note.** D-48's gate is a producer-side redundancy guard, not the system-level cycle-resolution mechanism. The actual cycle-resolution is **[D-49](../40_decisions.md#d-49) / [I-35](../30_invariants.md#i-35)** (`addOrigin`'s cap-full preference replacement plus the bulk-merge routing change). This subsection describes D-48's gate verbatim; see *Origin selection at cap* below for D-49.

`applyEquivalenceClass` rewrites a non-equality expression by substituting equivalence-class peers into its arguments. For each rewrite, it (a) deposits the rewritten expression into `memoryBlock.intEncodedStatements` and (b) emits an `equality1` origin record naming the source expression + the justifying equalities.

**Gate.** The emission only fires when the target rewrite is *not already in* `memoryBlock.exprOriginMap`. The lambda lives inside the `trackHistory` block:

```cpp
auto alreadyHasOrigin = [&](const ExpressionWithValidity& ev) -> bool {
    auto it = memoryBlock.exprOriginMap.find(ev);
    return it != memoryBlock.exprOriginMap.end() && !it->second.empty();
};
```

The deposit (intStatementLevelsMap, intEncodedStatements, newStatements, mailOut.statements) is *not* gated — only the origin emission is. (A code drift had gated the deposit on the same origin probe; corrected by [D-321](../40_decisions.md#d-321) — under the canonical door an origin row no longer implies a registered statement, [G-70](../50_gotchas.md#g-70).) The previously-required populate of `exprOriginMapLocal[rewrittenExpr] = eqs` inside `enumerateEqClassRewrites`'s callback also stays unconditional, so any *first* emission for a target carries its full `setEqualities` justifier list (verifier `check_equality1` rejects `len(rest) < 4`).

**What this gate covers.** It suppresses redundant equality1 emissions inside `applyEquivalenceClass` when the target is already established **inside this LB**. The canonical case where the gate fires usefully: a rewrite produces a target whose origin was already recorded by an earlier non-equality emission in the same hashburst — emitting another equality1 origin would be a parallel record adding nothing.

**What this gate does not cover.** Two known cases:

1. **Swap-cycle across syntactically distinct keys.** When `applyEquivalenceClass` emits in both directions (e.g. source `(in2[2,X,8])` → target `(in2[X,2,8])` and source `(in2[X,2,8])` → target `(in2[2,X,8])` via the simultaneous slot-0/slot-1 swap mapping), each target's exprOriginMap entry is empty when its emission fires, so the gate does not fire. Cycle vector remains.
2. **Mail-imported cycles.** Multiple producer LBs may each emit one half of a cycle; `smashMail` then aggregates them into a receiver's `mailIn.exprOriginMap` uncapped. The gate is producer-local; it cannot suppress what arrives at a different receiver.

Both cases are resolved at the receiver via [D-49](../40_decisions.md#d-49)'s preference replacement at `addOrigin`.

**Site coverage.** Of the three `equality1` emission sites in `logical_block/equality_propagation.hpp` — `applyEquivalenceClass`, `applyEquivalenceClassToNegatedEquality`, `emitIntegrationRevivalToInternalMailIn` — only `applyEquivalenceClass` is gated by [I-34](../30_invariants.md#i-34). The other two have internal early-exits that prevent the same cycle shape (`intStatementLevelsMap` dedup at site 2; mail absorbance at site 3 routes through site 1's gate downstream). With D-49 in place, the receiver-side preference replacement handles any residual cycle vectors from any of these sites.

**Producer-side sync — not needed at the producer.** The gate consults only `body.exprOriginMap`, which is already kept current by the bulk-merge of `body.mailIn.exprOriginMap`. (D-49 changes how the bulk-merge writes to `body.exprOriginMap`, but D-48's gate predicate is unaffected.)

---

## Same-scope fixpoint after the ancestor pass ([D-322](../40_decisions.md#d-322), [I-218](../30_invariants.md#i-218))

`updateEquivalenceClasses` merges in three passes: (1) same-scope classes overlapping the equality's arguments; (2) ancestor classes overlapping the arguments, absorbed into the merged class (the ancestor stays, I-31); (3) a fixpoint over the same scope — every class not merged in (1) that shares a member with the grown merged class is merged too, with the shared member as bridge and no equality levels, until nothing overlaps. Without (3) a class that absorbed an ancestor class before that class grew survives beside the new merged class and the scope holds two classes with common members.

## Origin selection at cap ([D-325](../40_decisions.md#d-325), [I-216](../30_invariants.md#i-216))

Every origin writer keeps a key's first `max_origin_per_expr` rows in insertion order; once the cap is reached a newcomer is dropped and nothing is displaced. The `updateEquivalenceClasses` origin sync (`exprOriginRunReplace`, called per class key from `updateEquivalenceClasses`) writes the body's existing lines first, verbatim, and dedup-appends the merged class's lines while below the cap.

**Why insertion order.** A row's dependencies were recorded before the row, so the surviving prefix is well-founded. The earlier tag-based policy ([D-49](../40_decisions.md#d-49), retired — `equality1` / `equality2` treated as disposable "convenience" rows) removed the class merge's `equality2` transitivity line, the one non-circular origin of a class-pair equality, in favour of cancellation firings that were circular under the canonical door; the chapter goal then had no acyclic derivation and `buildStack` ran into its call tripwire.

**Composition with other invariants.** [I-32](../30_invariants.md#i-32) (cross-pair `equality2` gate) and [I-34](#cross-substitution-equality1-emission-gating-d-48-i-34--producer-side-redundancy-guard) (cross-substitution `equality1` gate) decide whether a row is written at all; the cap decides only how many rows a key keeps.

---

## Cross-pair `equality2` emission gating ([D-46](../40_decisions.md#d-46), [I-32](../30_invariants.md#i-32))

`mergeTwoEquivalenceClasses` builds a `mergedClass` from two overlapping classes (the in-flight `classA` and the existing `classB`), bridged through a `commonArg`. The body's history block iterates every `(varA ∈ classA \ {commonArg}, varB ∈ classB \ {commonArg})` pair and records an `equality2` cross-pair origin for the new equality `(=[varA, varB])` and its mirror — documenting that the equality is derivable through the bridge.

**Gate.** The push only fires when the target equality is *not* already established. The gate consults three sources:

| Source | Covers |
|---|---|
| `mergedOriginMap` | Origins from prior cross-pair pushes within the same `updateEquivalenceClasses` call (multiple existing classes overlap `eqArgs` and get folded sequentially). |
| `classB.equalityOriginMap` | Origins on the absorbed class — about to fold into `mergedClass` via the post-loop overwrite. Treat as already-known. |
| `memoryBlock.exprOriginMap` | LB-level origins, including mail-arrived ones synced into the class via the bulk-merge + mail-sync at [`prover.cpp::performElementaryLogicalStep`](../../../GL_Quick_VS/GL_Quick/src/prover.cpp). |

If any source has a non-empty entry for the target, all three `addOrigin` calls (`mergedOriginMap`, `memoryBlock.exprOriginMap`, `memoryBlock.mailOut.exprOriginMap`) are skipped.

**Bridge citation scopes.** When the target is new, each of its two bridge equalities is cited at the scope where its body origin row lives: probe the merge scope first, then strict ancestors deepest-first. Cross-scope merge can combine one branch-local equality with one equality inherited from `main`; assigning the merge scope to both would strand the ancestor dependency below the OR boundary during chapter export. A complete miss asserts at the producer.

**Why.** When a clique of equalities arrives via mail (e.g. parent-scope contradiction-cascade), each equality's mail origin lands in `body.exprOriginMap` and the class's `equalityOriginMap`. Without the gate, subsequent merges iterate possible bridges and produce `equality2` cross-pair records via DIFFERENT bridges that point at each other:

```text
(=[v1,v2]) ← equality2 | (=[v1,i1]) (=[i1,v2])    -- bridge i1
(=[v1,i1]) ← equality2 | (=[v1,v2]) (=[v2,i1])    -- bridge v2
```

Both records survive to the chapter; the verifier's origin-chain DFS detects the cycle. The gate suppresses the redundant transitive-closure records — they add zero deductive content when the target is already established by a separate path. See [G-38](../50_gotchas.md#g-38) for the historical chapter-22 / theorem-12 case.

---

## Worked example — class expansion

Setup (all in scope `"main"`):

```text
(=[a,b])    → class {a, b}
(=[b,c])    → class grows to {a, b, c}
```

Now `!(=[a,d])` arrives. `applyEquivalenceClassToNegatedEquality` emits:

```text
!(=[b,d])       — one-sided substitution on a's position
!(=[c,d])       — one-sided substitution on a's position
```

Does *not* emit (skipped intentionally):

```text
!(=[b,e])  if d had been in class {d,e}
```

That would require the symmetric cross-product. Two one-sided steps reach it: `!(=[a,e])` first (by substituting `d → e`), then `!(=[b,e])` (by substituting `a → b`).

---

## Weaknesses

### Known & tracked

- **Complete-view rebuilding still follows class growth across initialized descendants.** Indexed level lookup removes the quadratic absence scans in pair copying, but it does not make reconstruction incremental or defer equality visibility. Components that later compare unchanged still have their retained histories merged, serialized and published before that comparison. 

- **Large live inequality closure remains proportional to its generated pairs.** Explicit continuations remove native recursion and refuted scopes are excluded, but neither change bounds the number of distinct live variants. Runtime investigation must measure that work separately from canonical maintenance.

- **Reflexive equalities still arise.** GL is meant never to hold `x=x` (it makes variable copies instead), yet the FTA batch registers reflexive equalities by the thousand. They are born in `expansion` history lines of compact expressions whose arguments collide (`and23[11,1,11]` expands to `(&(in[11,1])(=[11,11]))`; `!(strictOrder[1,4,11,11])` yields the rule `preorder(11,11) → 11=11`; `implication1602[rec,rec,10,10]`), in `equality1` rewrites of an equality whose two sides share a class, in `equality2` class paths, and in the case scopes of colliding OR instances such as `or8[1,11,12,11,10]`. [I-295](../30_invariants.md#i-295) keeps them out of the class store only; the producers of the colliding instances are open. 
- **[I-12](../30_invariants.md#i-12) — one-sided-only is deliberate.** Anyone questioning the design should first profile alternatives.
- **Class-visibility confinement to deeper-scopes.** A fact derived via equality in scope `S` is admitted at scope `S`, not promoted to `"main"`. If `S` eventually discharges, the class itself goes with it (scope-scoped).

### Suspected fragility

- **Substring special tokens in the registry filter.** The scan behind `filterIterationsCore` is a substring search, so a statement holding a longer normal name such as `it_0_lev_1_7_copy` also yields the special token `it_0_lev_1_7`. The canonical door drops such fragments (`resolveStatementTokens`, whole bracket tokens only); the filter still judges them, so a class holding both names could make the filter refuse or keep a statement for a token it does not contain. Not observed; not changed.
- **`applyEquivalenceClassToNegatedEquality` precondition assert.** A caller that mis-gates (passes a non-negated-equality) crashes the prover. The guard is deliberate, but adding a new call site without reviewing the guard is a regression risk.
- **Class-merge level computation.** When two classes merge, the new class's per-pair level is the maximum of the two — or is it? Not inspected for this chapter; worth confirming.
- **Per-scope class-map lookup cost.** On each `addStatement`, the per-validity class map is queried. For deeply-nested scope stacks, this is O(depth) per emission. Not currently a hot path but warrants measurement before FTA scale.
- **rmi memory growth under [D-43](../40_decisions.md#d-43) keep-old.** `rmi` now grows monotonically under class application instead of moving in place. Bounded per-call by class fan-out × overlap with `varsInRejectedMapIntegrationKeys`, but cumulative growth at FTA scale (richer fact bases, larger classes) is unmeasured. The cache short-circuit suppresses additions for non-overlapping classes; overlapping classes can still produce K1+K2 pairs at every iteration. Watch peak `rmi.size` during FTA-rung runs; if it exceeds 5× the pre-D-43 baseline, profile.
- **Chained ancestor merges under [D-44](../40_decisions.md#d-44).** The ancestor-pass overlap test in `updateEquivalenceClasses` is against `eqArgs` (the new equality's two args), not against the growing `mergedClass`. If a same-NS class adds var `x` to `mergedClass` and an ancestor class contains `x` but not `a` or `b`, the ancestor class is not absorbed by the merge. The transitivity is still recoverable at runtime via the apply machinery (`mergedClass` substitutes `a ↔ x`; the ancestor class substitutes `x ↔ y` via [D-33](../40_decisions.md#d-33)'s class-shallower direction), so no soundness loss — but the descendant scope's stored class does not explicitly carry `y`. If a downstream rule needs `y` in the same class, the apply chain delivers it; if direct class-membership lookup of `y` is needed, it would miss. Not currently a known failure mode; flagged for future work.

### Not exercised by tests

- **Class-merge origin tracking.** Is the origin of a merged class correctly a combination of the origins of the two input classes? Believed yes, but no targeted test verifies.
- **Class cleanup on scope discharge.** When a scope closes, its class should be discarded. If it persists, stale classes would influence later unrelated scopes. Believed correctly handled; not targeted-tested.

---

## Extension — `rejectedMapIntegration` (source re-keying)

`logical_block/equivalence_admission_rewriter.hpp` / `applyEquivalenceClassToRejectedMapIntegration` delegates to `rewriteParkedExistences`. Both rejected maps now retain source existences rather than concrete witness products. The class's normal canonical chooser rewrites the marked key and source arguments at the class scope. Source arguments absent from the key are checked too. Source generation and processing flags remain, and the levels of applied equality pairs join the source's stored levels.

Each applied substitution appends a concrete before/after/equality triple to the source record. No witness or history is allocated for an undemanded rewritten source. A qualifying demand queues the rewritten existence on internal mail; selected disintegration later writes fresh witness history. Source/scope completion records follow canonical source spellings. Admission remains independent of origin-map contents.

The integration writer maintains its argument-key cache and excludes typing keys. Scope buckets refresh when re-keying adds keys; the two source hooks share the same record layout and transformation contract.

## Extension — `admissionMapIntegration` (integration-side equi-class hook)

As of [D-65](../40_decisions.md#d-65) (2026-05-13, ), `applyEquivalenceClassToAdmissionMapIntegration` is the integration-side equi-class hook on `admissionMapIntegration`. Lives in `logical_block/equivalence_admission_rewriter.hpp` next to the algebra `applyEquivalenceClassToAdmissionMap` ([D-57](../40_decisions.md#d-57)). Called once per class at all four sites where the other equi-class hooks run — same-NS loop, ancestor-NS loop, descendant-NS loop, and the `applyClassesFrom` fixpoint — immediately after each `applyEquivalenceClassToRejectedMapIntegration` call.

The new hook closes a long-standing asymmetry: when a class formed on the algebra side, [D-57](../40_decisions.md#d-57)'s `applyEquivalenceClassToAdmissionMap` inserted a canonical-form K' into `admissionMap`, but the parallel `admissionMapIntegration[K']` was never inserted. The integration side could not match on the canonical form even though the underlying class made it admissible.

**What it does.** For each entry in `HashMemory::admissionMapIntegration`:

1. **Short-circuit on `HashMemory::varsInAdmissionMapIntegrationKeys`** — new bare-name cache mirroring `varsInAdmissionMapKeys` for the integration side. Populated at every `admissionMapIntegration` insert (the existing site in `disintegrateExprCore2`'s C1 block, plus this hook's post-loop apply).

2. **Same-scope only.** Apply only to entries whose `validityName` is identical to the class's — same rule as the algebra hooks (no ancestor/descendant directions). The deposit scope is that shared validity. See [D-106](../40_decisions.md#d-106).

3. **Strip u_ prefix** from the key to obtain a bare-form key (`removeUPrefixFromArguments(K)`). Class members are bare names; `enumerateEqClassRewrites` enumerates bare-name rewrites. Restoring u_ on the rewritten key happens in step 5.

4. **Pick one canonical representative for the whole class** via `chooseCanonical` (normal > `int_` > `it_`, lex-smallest, strong members) and rewrite the bare-form key once with the single replacement map. Skip the entry if the key is unchanged, or if its base operator is `Anchor*` (anchor predicates are scope identities, never substituted — [D-82](../40_decisions.md#d-82)). There is **no enumeration and no arg-equalization filter**: the class collapses to one representative and slot collapses are allowed, because both sides canonicalise identically and [I-36](../30_invariants.md#i-36)'s positional-collision preservation is no longer load-bearing.

5. **Restore u_ prefix** on each non-marker bare arg in the rewritten key — mirrors the u_-restore step in the pre-revision rmi hook. The marker slot stays `"marker"`; bare args (rare in admissionMapIntegration but possible if a non-u_ arg ever lands in the key) stay bare.

6. **Substitute the `map<Instruction, set<string>>` value** with an augmented substMap. The augmented substMap includes both bare → canon AND u_<bare> → u_<canon> pairs:
 - Bare pairs substitute the `set<string>` applied-vars history (the per-Instruction applied-vars are bare names per the construction sites in `updateAdmissionMapIntegration` and `isAdmittedIntegration`).
 - u_-prefixed pairs substitute the u_-form strings inside `Instruction.data[*].signature`, `Instruction.data[*].elements`, and `Instruction.markedGoal`.

7. **Drop the changed key K and insert the canonical K'** (the u_-form rewritten key) — a re-key, not an additive add (supersedes the [I-42](../30_invariants.md#i-42) additive rule). There is no `admissionStatusMap` on the integration side, so nothing to move. Erases and inserts are queued and applied post-loop. Value-merge at K' uses `map<Instruction, set<string>>::operator[]` semantics — multiple inserts at the same (K', Instruction) merge applied-vars via `set::insert`. K' is canonical, so it is never itself one of the dropped keys.

8. **Populate the cache** for K' — non-marker args from the bare-form rewritten key are inserted into `varsInAdmissionMapIntegrationKeys`.

9. **Fire `revisitRejectedIntegration2(bareK', mb, depositValidity)`** for each new K' — note **bareK'**, because `rejectedMapIntegration` keys are bare-marker form, not u_-form. `revisitRejectedIntegration2` walks the unchanged `rejectedMapIntegration[bareK']`, mails any matching rejection cohort via `sameIterationInternalMail`, erases the rmi entry, and consumes no admission entry. The integration admission template at K' persists for future revivals.

**Why never touch `rejectedMapIntegration` directly.** Same provenance rule as algebra. Per the revised [I-37](../30_invariants.md#i-37): substituted constituents inserted directly into `rejectedMapIntegration[K']` would lack post-substitution `disintegration` provenance recorded in `exprOriginMap`. The drop+mail rmi hook ([D-64](../40_decisions.md#d-64)) routes the rewritten compound through `sameIterationInternalMail`; the kernel's natural disintegration re-emits provenance at production site.

**Iteration safety.** Inserts, cache populates, and `revisitRejectedIntegration2` calls are queued in `toInsert` and applied after the outer admissionMapIntegration walk. Mirror of the algebra hook's pattern.

The `applyEquivalenceClassToAdmissionMapIntegration` hook is gated on `!parameters.skip_eq_classes` at the same outer site as the other equi-class machinery.

### No post-class-update sweep ([D-106](../40_decisions.md#d-106); supersedes [D-66](../40_decisions.md#d-66) / [I-43](../30_invariants.md#i-43))

The hook now drops the changed key **inline** (the re-key in step 7), so there is no separate integration-admission sweep. `cleanUpAdmissionMapIntegration` is no longer called from the Step-4b deferred-cleanup block in `logical_block/logic_block_operations.hpp::standardProcessing`; the inline drop subsumes every key it removed. The `cleanUpAdmissionMapIntegration` definition was deleted as dead code together with its algebra sibling `cleanUpAdmissionMap` (2026-07-02, user-authorized). Only `cleanUpExpressions` still runs at Step-4b.

### The [I-22](../30_invariants.md#i-22) exception — no on-hit closure

> **Moot since [D-299](../40_decisions.md#d-299) (2026-08-22).** The algebra on-hit cleanup this section carved integration out of is deleted; neither side consumes an admission entry now ([I-200](../30_invariants.md#i-200)). The prohibition still stands — do not add a consumption-driven erase to either side — and the reasoning below is why.

[D-62](../40_decisions.md#d-62) had two halves: point 1 (post-class-update sweep, `cleanUpAdmissionMap`) and point 2 (on-hit canonicalization closure, `cleanAdmissionMap`'s `markerIsOutput` branch). The integration mirror replicates point 1 ONLY. **There is no integration analog of point 2.**

[I-22](../30_invariants.md#i-22) says: integration admission templates are reusable — a single template can admit multiple distinct `int_` witnesses over the proof lifetime. Cleaning canon-equivalent integration admission entries on hit would silently exhaust the template before its remaining witnesses fire. The user's framing from this session: *"with exception of cleanUp after hit - i think integration can follow algebra playbook with adjustments for integration data types"* — the one exception is exactly the on-hit closure.

The `mb.overallHashMemory.admissionMapIntegration.erase(evKey)` calls that once sat inside `cleanAdmissionMap`'s `markerIsOutput` branch were key-form no-ops (algebra bare-marker key vs integration u_-form key never match) and went with that function.

### Shared inner-loop helper

`applyEquivalenceClass` routes its per-mapping substitution through the template helper `enumerateEqClassRewrites` in `logical_block/equality_propagation.hpp`. (Historically the rmi hook routed here too; since the [D-106](../40_decisions.md#d-106) drop-and-rekey rework `applyEquivalenceClassToRejectedMapIntegration` re-keys via the single `chooseCanonical` representative instead, leaving one caller. The helper stays factored out because it owns the substitution arithmetic.)

The helper takes the reduced member list as decoded strings and ids in lockstep (`eqList` / `eqListIds`, produced by `reduceEqClassIds` plus per-id decode — decoded-lex storage order, never id order). It gates argument positions by id membership (`NameMap::lookup`, non-minting: a never-interned arg cannot be a member), iterates `allMappingsAna[(|indices|, |eqList|)]`, reads per-pair levels from the packed `intEqualityLevelsMap` twin (`packEqPairKey`), and emits each rewrite via a callback as an `EqClassRewrite` struct (`rewrittenExpr`, `setEqualities`, `extraLevels`, `substMap`, `isIdentity`). All policy stays at the call site, with one exception — the [anchor-exclusion gate](#anchor-exclusion-gate-d-82) lives at the helper level because it is universal:

- **Anchor-exclusion gate** (helper-level — universal) — `enumerateEqClassRewrites` returns immediately when `baseExpr.rfind("Anchor", 0) == 0`. No rewrite is emitted for any expression whose base operator is an `Anchor*`. See [D-82](../40_decisions.md#d-82).
- **Wrappers** — caller chooses `wrapLeft` / `wrapRight` and the `baseExpr` (handles both `(...)` and `!(...)`).
- **Scope-direction admission** — call-site policy; `applyEquivalenceClass` admits all three directions (same / class-shallower / class-deeper, [D-33](../40_decisions.md#d-33)).
- **Downstream emission** — the call site emits to `intEncodedStatements` / origin map (`equality1`); the helper emits nothing.

Any change to the substitution algorithm itself (mapping enumeration, equality-string format `"(=[from,to])"`, levels-set merging) lives in one place; policy lives at the call site — except the anchor-exclusion gate, which is logically universal and therefore lives once at the helper entry.

### Anchor-exclusion gate ([D-82](../40_decisions.md#d-82))

Anchor predicates (`AnchorPeano[…]`, `AnchorGauss[…]`, `AnchorIncubator[…]`) are the positional scope identities of their LB. Their argument slots carry conjecturer-assigned canonical names (e.g. `N`, `i0`, `s`, `+`, `*`, `i1` on Peano), not equivalence-class members; the chapter exporter and every downstream consumer treat an anchor predicate as the immutable contract of the LB's scope. Two gates jointly ensure that no equi-class substitution writes a modified anchor to any equi-class-touched container:

1. **Helper-level gate** — `enumerateEqClassRewrites` early-returns when the base operator is an `Anchor*`. Prevents *generation* of anchor-variant strings for `applyEquivalenceClass` (algebra additive emission), `applyEquivalenceClassToRejectedMapIntegration` (integration-side drop+mail), and any future caller.

2. **Admission-map value-loop gate** — `applyEquivalenceClassToAdmissionMap`'s value-key substitution loop tracks an `anchorChanged` flag and refuses the entire admission update if any element with prefix `(Anchor` would be rewritten. Prevents *insertion* of a modified-anchor admission template even when the substitution path bypassed the helper (the value-key loop iterates `AdmissionMapValue::key` directly with `ce::replaceKeysInString`, not through `enumerateEqClassRewrites`).

**What this protects.** A class with `(=[i0, i0_copy])` in scope would, without the gate, enumerate `AnchorPeano[N,i0_copy,s,+,*,i1]` as a "valid" rewrite of `AnchorPeano[N,i0,s,+,*,i1]`, deposit it as a statement, and key admission templates by the rewritten string. Downstream the chapter exporter sees two competing anchor predicates for the same scope; the verifier's chapter-row checks have no canonical-choice rule and emit failures. With the gates, the original concrete anchor is preserved (additive zero variants) and the admission map's anchor-position contract is never mutated.

**What this does NOT do.** It does not forbid equi-classes from *containing* members that happen to coincide with anchor-slot names — it forbids equi-class substitution from *writing* through an anchor-predicate string. An equality `(=[i0, j])` is still a valid class member; it propagates through every non-anchor expression normally.

**Code.** Both gates live in `logical_block/equality_propagation.hpp` — `enumerateEqClassRewrites` (top-of-function early return) and `applyEquivalenceClassToAdmissionMap` (value-loop `anchorChanged` flag). The integration mirror `applyEquivalenceClassToAdmissionMapIntegration` inherits the helper-level gate but is **not** itself guarded at the value-loop level — its values are `map<Instruction, set<string>>` content, where anchor predicates do not appear as full-key elements; only the helper-level coverage is needed there.

---

## Extension — `admissionMap` (algebra-side equi-class hook)

As of [D-57](../40_decisions.md#d-57), equivalence-class application has a third hook on the algebra side: `applyEquivalenceClassToAdmissionMap` in `logical_block/equivalence_admission_rewriter.hpp`. Called once per class at all four sites where the rmi hook runs — same-NS loop, ancestor-NS loop, descendant-NS loop, and the fixpoint re-iter lambda (`applyClassesFrom`) — immediately after each `applyEquivalenceClassToRejectedMapIntegration` call.

> **Distinct from the hashburst admission tail.** This hook is the equivalence-class *apply* path and runs inside `applyEquiClasses`. The other algebra-`admissionMap` writer — the new-key discovery in the hashburst marker branch (`memory.cpp::checkLocalEncodedMemoryStatic`) — was lifted out of the fixpoint loop on: it stages `AdmissionKeyAlgebraRecord`s on `Memory::admissionKeysAlgebra` and replays them post-loop via `drainAdmissionKeysAlgebra` ([D-103](../40_decisions.md#d-103) / [I-68](../30_invariants.md#i-68)). This hook is unaffected and still applies inline within the apply pass.

**What it does.** For each entry in `HashMemory::admissionMap`:

1. **Short-circuit on `HashMemory::varsInAdmissionMapKeys`** — symmetric to `varsInRejectedMapIntegrationKeys`. Monotonically-growing set of non-marker args appearing in any admission key. Populated at every admission insert (`logical_block/integration_processor.hpp::prepareIntegration`, `memory.cpp::makeMandatoryEncodedStatementLists1Static`, `logical_block/admission_memory.cpp::updateAdmissionMapRecursion`, and the hook itself's post-loop insert). Lets the hook early-exit for classes whose `clss.variables` have zero overlap with this set.

2. **Pick one canonical representative for the whole class** via `chooseCanonical` (`logical_block/canonicalizer.hpp::chooseCanonical`): the highest-priority *strong* member (drop weak names with `reduceEqClass`), preferring ordinary names (including `repl_`), then forced witnesses, then integration witnesses by numeric depth, then algebraic witnesses, with lexical ties. Build one replacement map sending every other class member to that representative. An empty pick (empty / all-weak class) returns with no work.

3. **Same-scope only.** Apply only to entries whose `validityName` is identical to the class's. Cross-scope equalities are folded into the class beforehand by the ancestor-absorbing merge ([D-44](../40_decisions.md#d-44)), so key application is same-`validityName` only — no ancestor/descendant directions. The deposit scope is that shared validity. See [D-106](../40_decisions.md#d-106).

4. **Rewrite the marker key once** with the single replacement map. If the key is unchanged, skip the entry. Anchor keys (`baseExpr` begins `Anchor`) are skipped — anchor predicates are scope identities, never substituted ([D-82](../40_decisions.md#d-82)). There is **no enumeration and no arg-equalization filter**: the class collapses to one representative, so a key changes in at most one way, and slot collapses are allowed (repetitions permitted). The positional-collision preservation that [I-36](../30_invariants.md#i-36) enforced on the old enumerate path is no longer needed here, because the companion rejected-key hook collapses identically — both sides of the admission/rejection pair land on the same representative. See [D-106](../40_decisions.md#d-106).

5. **Substitute the AdmissionMapValue contents**: each `AdmissionMapValue.key` element and each member of `remainingArgs` gets `ce::replaceKeysInString` with the same replacement map. `standardMaxAdmissionDepth`, `standardMaxSecondaryNumber`, and `flag` are copied unchanged. `u_`-prefixed args are naturally untouched (replacement keys are bare class members). **Anchor-changed refuse** ([D-82](../40_decisions.md#d-82)): if any `(Anchor`-prefixed value element would change under the map, that value is refused.

6. **Drop the changed key K and insert the canonical K'** — a re-key, not an additive add. K is erased from `admissionMap` and the parallel `admissionStatusMap`; K' is inserted with K's status moved onto it (unless K' already has a status entry). K' is canonical, so it is never itself one of the dropped keys. Erases and inserts are queued and applied post-loop to avoid mid-iteration mutation.

7. **Populate `varsInAdmissionMapKeys`** for K', then **fire `revisitRejected2(K', mb, depositValidity)`** — it walks the unchanged `rejectedMap[K']` and mail-emits any matching rejection cohort via `emitIntegrationRevivalToInternalMailIn`. K' is bare-marker form (admissionMap keys carry no u_ prefix), directly usable as a rejectedMap key.

**Why never touch `rejectedMap`.** Per [I-37](../30_invariants.md#i-37): `rejectedMap` holds real disintegration products whose `disintegration` origins were recorded at production site by `logical_block/disintegration_processor.cpp::disintegrateExprCore2::trackExpansionHistory`. The companion rejected-key hook ([rejectedMap section](#extension--rejectedmap-algebra-side-equi-class-hook-drop--mail)) canonicalises rejected keys the same way and mails the rewritten compound; the kernel regenerates `rejectedMap[K']` with proper provenance. Admission never writes `rejectedMap` directly — it re-keys its own map and lets `revisitRejected2` match against `rejectedMap[K']`.

**Iteration safety.** Erases, inserts, status moves, cache populates, and `revisitRejected2` calls are queued and applied after the outer admissionMap walk.

The `applyEquivalenceClassToAdmissionMap` hook is gated on `!parameters.skip_eq_classes` at the same outer site as the other equi-class machinery — incubator batches (which set `skip_eq_classes = true`) bypass it.

### No post-class-update sweep ([D-106](../40_decisions.md#d-106); supersedes [D-62](../40_decisions.md#d-62) / [I-40](../30_invariants.md#i-40) on the algebra admission path)

The hook now drops the changed key **inline** (the re-key in step 6), so there is no separate algebra-admission sweep. `cleanUpAdmissionMap` is no longer called from the Step-4b deferred-cleanup block in `logical_block/logic_block_operations.hpp::standardProcessing`. The inline drop subsumes every key the sweep removed: the sweep dropped keys with a non-canonical `int_`/`it_` member; the inline re-key drops every key that changes under the canonical map — a superset (it also canonicalises non-canonical *normal* members). The hooks run to fixpoint inside `applyEquiClasses` and Pass 2 re-fires every class, so by the time Step-4b would have run, the admission map is already canonical. Only `cleanUpExpressions` still runs at Step-4b. The `cleanUpAdmissionMap` definition was deleted as dead code together with its integration sibling `cleanUpAdmissionMapIntegration` (2026-07-02, user-authorized).

### Cleanup-on-hit closure ([D-62](../40_decisions.md#d-62), [I-41](../30_invariants.md#i-41))

> **Retired by [D-299](../40_decisions.md#d-299) (2026-08-22).** `cleanAdmissionMap` and the helper `canonicalizeUnderClasses` are deleted, and with them the consumed-key set. An admission key is now removed only by the canonicalization re-key above and by the subtree wipe ([I-200](../30_invariants.md#i-200)). The rest of this section describes the retired closure.

`cleanAdmissionMap` (`prover.hpp::cleanAdmissionMap`) fires when a disintegration head consumes K — specifically when `marker` is in the operator's output slot. Before this branch, the cleanup erased only the consumed key. The closure extension propagates the cleanup to every admissionMap entry at the same validity that admits the same fact under the current classes.

**Mechanism.** Option 3 of three considered (1-hop / iterative fix-point / canonicalization):

- For each entry K' at `validity`, compute `canonicalizeUnderClasses(K'.original, classes_at_validity)` — the helper substitutes every class member with its class's canonical (forced before integration by numeric depth before algebraic, with lexical ties) per the same rule `filterIterations` and `updateWeakVariables` use.
- Erase every K' whose canonical form equals `canon(consumed K)` — across all four state structures `cleanAdmissionMap` already touches: `admissionMap`, `admissionStatusMap`, `admissionMapIntegration`, `consumedAdmissionKeys`.

Canonicalization-based closure naturally covers transitive equivalence (a fact reachable from K via two different classes composed together still has the same canonical form). 1-hop closure (enumerate one class application at a time) would leave 2-hop entries stranded: with classes `C₁ = {a, b}`, `C₂ = {c, d}`, and admission keys `K = (p[a, c, marker])`, `K1 = (p[b, c, marker])`, `K2 = (p[a, d, marker])`, `K12 = (p[b, d, marker])`, a 1-hop sweep on K erases K1 and K2 but leaves K12 stale. Iterative fix-point would catch K12 but at the cost of multiple passes. Option 3 collapses the closure into one O(|admissionMap_at_v|) walk with deterministic semantics.

**Performance.** `cleanAdmissionMap` only does work when `markerIsOutput` (operator output slot consumed) — a constrained subset of admission probes. The closure walks all admissionMap entries at `validity` and computes canon per entry, O(|admissionMap_at_v| × |args|) per cleanup. No additional caches needed; the existing `varsInAdmissionMapKeys` short-circuit (no class member overlaps any admission key) skips the whole closure block.

**Same-validity pin.** Cross-scope K' (deposited at `deeperOf(class.scope, entry.scope)` by D-57's hook from the descendant or ancestor direction) is **not** chased by the closure. Each scope's `cleanAdmissionMap` triggers independently; the closure operates only on `admissionMap` entries at the consumed K's validity. Consistent with the principle that consumption at validity V does not necessarily speak to admission state at a different scope.

### Algebra `revisitRejected2` source delivery

`logical_block/rejected_source_replay.cpp` / `revisitRejected2` resolves exact and strict-ancestor source parks. `queueParkedExistence` checks current demand and budgets, deduplicates pending/completed source constructions at the target scope and mails the existence itself. Normal absorption re-expands the source with temporary slots and mints only after admission. Descendant history records `zoom in` for the source before its fresh disintegration chain; ancestor witnesses are not reused.

## Extension — `rejectedMap` (source re-keying)

`applyEquivalenceClassToRejectedMap` uses the same `rewriteParkedExistences` body as integration. Marked keys retain algebra qualification and the normal generation/secondary-variable budgets. Undemanded source rewrites remain parked with deferred provenance. A known source alone does not mean its witnesses were admitted: source/scope completion state, rather than ordinary statement presence, prevents duplicate minting.

The parked-OR maps retain their separate key-posting and drop/mail behavior through `applyEquivalenceClassToRejectedOrdisCore` (I-243). Their cohort semantics are unchanged by the witness-source representation.

## Extension — `rejectedMapSubproof` (key-only re-keying)

Parked implication subproofs ([D-406](../40_decisions.md#d-406)) keep a canonical key and a verbatim value. There is no hook inside `applyEquiClasses`: `rebuildCanonicalScope` stages every scope whose complete view changed (`stageSubproofRecheck`, only while parks exist), and the step-4b drain (`drainSubproofArrivals`) moves every park at or below a staged scope to its canonical key (`rekeySubproofPark`: records merged verbatim through the read-modify-write) and rechecks it. The recheck matters on its own: a merge can make a gate known without any new row, when a known row's member joins the gate's class. Reopening happens only in the drain, never inside the class application, because it opens scopes.

## Extension — `toBeProved` ([D-67](../40_decisions.md#d-67))

`sanitizeToBeProved` (`logical_block/equality_propagation.hpp`) closes the gap that pending goals in `Memory::toBeProved` were not rewritten on equi-class formation. A goal whose body referenced a now-renamed variable previously stayed in the map under its old form and could not be discharged. The function runs once per burst at the END of `performElementaryLogicalStep`, walks `toBeProved`, and rewrites every entry whose `it_/int_` args are now downprioritized under the active equi-classes. The entry's `validityName` is preserved verbatim — the goal's namespace never changes.

**Special-name sanitization.** Forced, integration and algebraic arguments can be replaced by a strictly preferred special peer under `compareWitnessNames`. Ordinary and `repl_` names do not participate in this special-only goal sanitization; the complete statement canonicalizer retains its ordinary-first policy.

**Why this constrained design — and why the catalogue prototype was retired.** An earlier prototype (additive insert via `applyEquivalenceClassToToBeProved` + per-class sweep `cleanUpToBeProved` + position catalogue `Memory::tbpAllowedIndices` + cascade-discharge of equivalent siblings + propagation logic + three call sites) was tried first. It exploded the Peano theorem count from baseline 58 to 160 until the catalogue gate was added. Once we constrained replacements to `it_/int_` → `it_/int_` only:

- **Plain Peano goals see zero substitution** because no class peer can win the priority test against a non-`it_/int_` argument. The catalogue's "skip if no it_/int_" guard becomes structural.
- **Cross-iteration substitution-surface widening cannot happen** because the rule only maps `it_/int_` → `it_/int_`; a rewrite never introduces an `it_/int_` at a previously-non-`it_/int_` position. The catalogue's "frozen at first insertion" rule becomes vacuously satisfied.
- **Equivalent siblings collapse naturally.** If two entries differ only at `it_/int_` positions, sanitize maps both to the same canonical form; the second insert is a no-op duplicate-drop. A separate cascade-discharge is unnecessary.

Net result: one function, one call site, zero side-data structures. No `tbpAllowedIndices` map, no `catalogueAllowedIndices` helper, no `cascadeDischargeTBP`, no `cleanUpToBeProved`, no `applyEquivalenceClassToToBeProved`.

**Scope rule.** The class peer must come from the entry's own scope or a strict ancestor. The helper walks `mb.nameMap.strictAncestorNames(keyTBP.validityName)` plus the entry's own scope and picks the best peer across all those equi-classes. Class-deeper-than-goal is rejected structurally: a class at a descendant scope is invisible at the entry's scope, so it never contributes a peer.

**No origin map emission at sanitize time.** Goals are exempt from the equality-line emission rule per the project conventions / [I-44](../30_invariants.md#i-44) — a goal has no upstream history line to extend. (The bridge-on-discharge history line that closes the gap between the source implication's `expansion for integration` line and the renamed goal lives elsewhere — see TBP-discharge bridge logic for that emission.)

**`auxies` are disjoint from `it_/int_` goals.** Auxies (the `set<int>` first slot of the TBP value tuple) are induction-discharge bookkeeping. A TBP entry whose body has `it_/int_` args is class-rewritable by sanitize and never participates in induction discharge — sanitize would rewrite the body while a parallel auxy expects the pre-rename form, breaking the induction discharge contract. An assert at the `addExprToMemoryBlock` `status==2` insertion site enforces this: if the body matches `it_/int_` regex, `auxyIndex` must be `< 0`.

**Call site.** The end of `performElementaryLogicalStep`; compact rules use their separate canonical maintenance path.

## Extension — `applyEquivalenceClass` compiled-implication branch ([D-68](../40_decisions.md#d-68), RETIRED)

A prototype split `applyEquivalenceClass`'s per-rewrite deposit loop by source-expression shape (`srcIsCompiledImpl` regex on `expr2.original`) and routed compiled-implication rewrites exclusively through `mb.sameIterationInternalMail.statements`. The prototype was reverted in the same squash — see [D-68](../40_decisions.md#d-68). The case is subsumed by the `expandedImplications` index + `sanitizeHashMemory` design described below: when `sanitizeHashMemory` rewrites + eradicates an implication on the rule side, the mailed rewrite re-enters via `sameIterationInternalMail` and the kernel's absorb chain populates `intEncodedStatements` (fact side) and `addToHashMemory` (rule side) together with proper `disintegration` provenance, so the fact-side gap that motivated the prototype closes structurally.

## Extension — the canonical gate at the rule-install loop ([D-312](../40_decisions.md#d-312))

**Only the canonical form of a compact implication enters hash memory.** The install door's per-implication body (`installExpandedImplication`, `logical_block/compact_rule_processor.hpp`; the one place the mailed compact and the or / existence cohort compact share, because a cohort's compacts are disintegrated inside the parent's disintegration and reach the door only through the implication channel's source column) opens with the gate:

1. **Canonical predicate.** For a compact carrier (`isCompactImplicationSpan`), `canonicalizeCompact` maps every eligible argument to `chooseCanonicalId` of its complete component at the rule's own scope (normal > `int_` > `it_`, lex-min in the tier, weak members at that target scope skipped). The rewrite is `replaceKeysScratch` over the fired pairs; the pairs and their admission levels (`forEachLevel(packEqPairKey(from, to))`, D-280) come back with it. Unchanged text = canonical.
2. **A canonical carrier** installs exactly as before (`overallHashMemory` always, `workingMemory` for a status-3 recovery, else `localHashMemory` + `localHashMemoryDelta`; the D-274 row; the `expandedImplications` pair and the carrier-index record).
3. **A non-canonical carrier K keeps no rule.** Its canonical form K* is expanded instead, once (`expandCanonicalCompact`: skipped when K* is already a carrier here or known at a strict ancestor; otherwise K*'s own disintegration, its implications through the same gate, and K* registered at the scope with the `equality1` history row `K* ← K, (=[from,to])…` — each equality cited where its origin row lives, the `applyEquivalenceClass` commit's walk — and the level run = the deposit's levels ∪ the fired pairs' levels). Attached implication products from that disintegration are also registered through `addStatement`, with their already-recorded derivations and the same supporting levels, including pair-free re-expansion. Each installed auxiliary rule therefore retains its own compact statement. K itself stays in the historical known-statement registry. Its encoded row remains stored while unused; ordinary class processing is enabled when its core becomes used.

**What it does not cover.** A compact expanded BEFORE its arguments joined a class keeps its rules until the class forms; that is the hook below, which runs the same reconcile from the class side. Gated on `!parameters.skip_eq_classes`.

## Extension — the canonical door ([D-326](../40_decisions.md#d-326), [I-217](../30_invariants.md#i-217))

**Only the class-canonical form of a deposit is admitted.** `addExprToMemoryBlock` (`logical_block/statement_memory.cpp`), after Site F on the raw text and Site H, runs `canonicalFormAtScope` (`logical_block/canonicalizer.hpp`) on every deposit that is not a goal (status 2), not a positive or negated equality, and not an anchor predicate:

1. **The scope's complete view.** The classes consulted are the deposit scope's complete canonical view (`canonicalEquivalenceClassesMap` at that validity: its own application classes merged with every visible ancestor class, maintained by `rebuildCanonicalScope`). The scope name and the induction-hypothesis identity keep the raw spelling they were minted from.
2. **Chooser.** `chooseCanonicalId` — ordinary > forced > integration by numeric depth > algebraic, weak members at the scope skipped, lexical ties — the same representative the admission / rejected key hooks collapse onto. Special-tier tokens come from the token scan, normal names from `collectExprTokens`, both resolved to NameMap ids once per call and walked once with the per-token rule of the core `forEachNonCanonicalMemberTokenCore`; each member token's representative is read from the normal half of `canonicalMemberRepresentatives` (`liveCanonicalMemberAt`, re-chosen live only when the recorded member turned weak, [I-287](../30_invariants.md#i-287)) instead of walking every class of the view. `filterIterationsCore` (the registry filter) is the core's `firstSpecialMemberId` / no-normal-source instance over the scan-and-lookup token source (byte-identical probe order, I-134). The two choosers agree on every reachable class (weak = the complement of `firstSpecialMemberId`'s pick at the same scope; a normal-name canonical carries no special token), so a door deposit is never refused by the registry filter.
3. **Skip on presence.** A changed deposit whose canonical form is known at the scope or a strict ancestor ends at the door — nothing registers, no history line for the new raw spelling, both the raw and the canonical ids run `stageOrReleaseForRefusedDeposit` (I-174). The maintainer's rule: a known canonical form that was not disintegrated means the raw one would not be either; retrospective admission (the rejected-map revival) disintegrates parked compounds independently of this loop.
4. **Locality, history and levels.** A rewrite flips the deposit to a LOCAL derivation iff at least one applied equality is a local statement of this LB (`appliedEqualityIsLocal` — the direct row or its mirror at the class scope or a strict ancestor; for a transitively joined pair, any local equality between two class members): it is then this LB's class product, the `equality1` rewrite `applyEquivalenceClass` would have made, and the canonical form registers local (the per-step delta the main-goal discharge and `fillMailOut` read; it is not staged external). A rewrite licensed only by mailed equalities keeps the arrival's locality. Without the flip a mailed fact that canonicalizes onto a main goal's literal text sits in the registry non-local and the goal never closes (the eight-theorem loss of the first landing). Under `trackHistory` the raw text keeps its producer line under the raw key (asserted present — a rewritten origin-less deposit would leave a bridge no chapter can resolve) and ships that row to `mailOut` at once whenever the row is local (the D-286 ship: the raw form is never a delta row). The canonical form's origin is one `equality1` bridge — raw form at the deposit scope, then each applied `(=[member,canonical])` cited at `findEqualityCiteScope` (the D-227 walk, shared with `applyEquivalenceClass`) — written only when the canonical form has no origin row (I-34). The level run is the deposit's ∪ the applied pairs' class levels (`{-1}` transparent, I-182); no union happens on the skip (a known canonical form with fewer levels means the arrival's extra premise was not needed — the over-premised-lemma ruling).
5. **Downstream unchanged.** `applyEquiClasses` multiplies the registered canonical form into its full orbit (products commit inline, never through the door), so a literal non-equality goal closes from the product in the same `standardProcessing` call. Equalities merge classes at arrival and negated equalities get the one-sided I-12 expansion as before. Rule carriers, negated compact existences and or compacts are canonicalized like any statement: the rule install's gate (D-312) then sees a canonical carrier, and an or cohort opens on the canonical signature.

The absorb drain offers every arrival raw and stages the status-3 `intExternalStatements` row from the door's returned canonical row; the drain-side `canonicalizeMailArrival` (I-203) is deleted.

**Or compacts get one gate MORE than the door** (the or-uniqueness gate, [I-219](../30_invariants.md#i-219), ): the door only collapses spellings whose classes exist at deposit time — an `(or<N>[…])` deposit that is a NEW text but an equi variant of an or already fully processed at the scope (the classes merged after the first processing) registers as a passive statement and is multiplied, and everything else — disintegration, K compacts, cohort, ordis parks — is suppressed. See the or-branching chapter's §1 for the ledger and probe mechanics.

## Extension — compact implications (the class-side reconcile hook)

`applyEquivalenceClassToCompactImplications` (`logical_block/compact_rule_processor.hpp`) runs after the admission hook for every class visit of both `applyEquiClasses` passes, with the D-307 round skip (`cbHitsPrevRound`). Its per-apply snapshot (`RejectedValidityBuckets::ensureBuiltFrom` over the compact rows of `intEncodedStatements`, decoded through the NameMap) posts `(validity, argument) → compact keys`; the candidates are the compacts at the class validity or a strict descendant of it naming a non-canonical member. Per candidate: a carrier removed earlier in the apply is skipped; one unchanged under the class and installed is done; otherwise the full canonical form under every visible class (`canonicalizeCompact`) decides — changed: its rules leave (`removeCompactExpansion`, staged) and K* is expanded (`expandCanonicalCompact`, status 1, generation 0, the statement's levels ∪ the fired pairs' levels); unchanged but unexpanded (a weak-member shift): expanded as is. The removals finish ONCE per apply after the fixpoint (`finishCompactRemovals`: the removed carrier keys erased in one pass, every instance's LMV compaction over the drop set + owner-less erasure) — before the next hashburst. Contract: [I-210](../30_invariants.md#i-210).

## Hypothesis reaction before class application

`logical_block/logic_block_operations.hpp` and `LogicBlockOperations::standardProcessing` call `reactToHypo` after absorbing and clearing input mail, before `applyEquiClasses`. Qualifying hypothetical classes emit parent-scope copy equalities through the ordinary deposit door. Their class deltas reach application, recursion-product enrichment and expression cleanup before the sole delta clear; `fillMailOut` publishes their history in the same call. Both CPU phases use this order. Earlier qualification may change theorem coverage; there is no compatibility condition or extra processing pass.

## Extension — hashMem rule registry via end-of-burst sanitize ([D-69](../40_decisions.md#d-69), RETIRED)

> **Retired ([D-312](../40_decisions.md#d-312)).** `sanitizeHashMemory` and `eradicateImplicationFromLB` are deleted; the rule registry stays canonical through the canonical gate and the compact hook above. `expandedImplications` survives as the hashburst dump's section source. The text below describes the retired mechanism.

`sanitizeHashMemory` (`memory_infra/cold_hash_map.hpp`) closed the rule-side gap: head LMVs in `encodedMap` whose `originalImplication` references a class-renamed variable kept firing under stale names and never produced canonical-form conclusions through the rule path. The fix lives at end-of-burst, on a per-LB index of installed implications, with eradication + drop+mail through the canonical absorb chain.

**Architecture (single call site, single per-LB index).** Two new symbols in `logical_block/logic_block_operations.cpp` — `sanitizeHashMemory` and `eradicateImplicationFromLB` — plus the per-LB set `Memory::expandedImplications` and its mail companion `Mail::expandedImplications`. The call site is exactly one: `logical_block/logic_block_operations.cpp::performElementaryLogicalStep`, immediately after `reactToHypo(body)` and before the EXIT trap. Equi-class machinery has stabilised by this point in the burst; running once at end-of-burst keeps the expensive eradication off the hot fact-absorption path.

**The `expandedImplications` index.** Every call to `addToHashMemory` from `addExprToMemoryBlock`'s implication branch inserts the implication's `(original, validityName)` into both `memoryBlock.expandedImplications` (local index, surveyed by the next end-of-burst sanitize) and `memoryBlock.mailOut.expandedImplications` (broadcast to children via `sendMail`). At the start of every burst, `body.mailIn.expandedImplications` is merged into `body.expandedImplications` and `mailIn` is cleared. Result: every LB whose hash registry actually holds a particular implication has that implication's signature in its own index, ready for sanitize.

**Sanitize walk.** For each entry in `body.expandedImplications`, the helper checks every `it_*_lev_*_*` / `int_lev_*_*` arg against the equi-classes visible at the entry's scope (the entry's own `validityName` plus every ancestor scope via `mb.nameMap.strictAncestorNames`). If any arg has a strictly-higher-priority `it_/int_` peer in any visible equi-class, the substitution is scheduled; otherwise the entry is skipped. The retired mechanism preferred its integration names to algebraic names with lexical ties and excluded ordinary names. Current witness ordering is described above under special-name sanitization.

**Apply: mail rewrite, eradicate original.** For each scheduled rewrite:

- The rewritten implication is mailed onto `mb.sameIterationInternalMail.statements` at the entry's `validityName` (with that LB's `intStatementLevelsMap` entry for the old form as the level set, or empty if absent).
- The `equality1` history line is written into `mb.sameIterationInternalMail.exprOriginMap` — origin tag `equality1`, with the old form followed by one `(=[old,new])` ExpressionWithValidity per substitution, gated by `parameters.trackHistory`.
- `eradicateImplicationFromLB(mb, oldImpl)` removes the old form from every per-LB registry that would otherwise dedup the re-push: `intStatementLevelsMap`, `intKnownStatements` (the whole row — both membership bits), `intEncodedStatements`, `intLocalEncodedStatements{,Delta}`, `intLocalEncodedStatementsSet`, every `encodedMap` LMV whose `originalImplication` matches across `overallHashMemory` + `localHashMemory` + `localHashMemoryDelta`, the underlying `originals` chain when fully orphaned (orphan check scans all three hashMems), and `mb.expandedImplications` itself.

The kernel's natural absorb-via-`addStatement` → `addExprToMemoryBlock` → disintegration → `addToHashMemory` chain on the next burst re-disintegrates the rewritten implication and re-installs both fact AND rule sides with `disintegration` origins for every chain element via `trackExpansionHistory`. Same drop+mail pattern as [D-63](../40_decisions.md#d-63) (rejectedMap) and [D-64](../40_decisions.md#d-64) (rejectedMapIntegration).

**Why drop+mail, not direct `addToHashMemory` reinstall.** A prototype that called `addToHashMemory(rewrittenImpl, …)` directly was built and ran. It bypassed disintegration, leaving chain-element premises without `disintegration` origins at the deposit scope, and `proof/proof_reconstructor.cpp::buildStack` asserted `"no origin found"` for premises like `(in[7,1])` during chapter export. The drop+mail design routes the rewritten implication through `addStatement`, which triggers disintegration, which emits the proper provenance via `trackExpansionHistory`.

**Why `sameIterationInternalMail`, not top-level mail.** The Site F / `registered`-bit dedup gates at `addExprToMemoryBlock`'s entry (the `intKnownStatements` membership bits) would block a top-level re-push of the rewritten implication. `sameIterationInternalMail` bypasses that gate; kernel processing of mailed statements runs disintegration regardless of the registration record.

**Why a per-LB index (vs walking `encodedMap` directly).** An earlier prototype walked `encodedMap` head LMVs by `remainingArgs ∩ class.variables` per class-rewrite candidate, paired with a two-phase sweep (`applyEquivalenceClassToHashMemoryOriginals` + `cleanUpHashMemoryOriginals`, both retired). The index keys directly on the implication's textual form — exactly what `eradicateImplicationFromLB` needs to remove the old form from every storage surface — and reduces the per-LMV walk to a single match-on-`originalImplication` pass per scheduled rewrite. The catalogue / cleanup / `tbpAllowedIndices` infrastructure (also retired) becomes unnecessary under the `it_/int_` → `it_/int_` restriction.

**Origin map emission.** Mandatory but indirect. The `equality1` line goes to `mb.sameIterationInternalMail.exprOriginMap`; the kernel's mail-bulk-merge routes it through `addOrigin` into `body.exprOriginMap` with cap-full preference per [D-49](../40_decisions.md#d-49) / [I-35](../30_invariants.md#i-35). Per the project conventions / [I-44](../30_invariants.md#i-44), originMap maintenance is mandatory but never drives proof decisions.

**Not touched.** `IntNormalizedKey`, `normalizedEncodedKeys`, `normalizedEncodedSubkeys*`, `remainingArgsNormalizedEncodedMap` — these carry only integer IDs with no concrete `u_/it_/int_` payload. Equi-class rewrites cannot invalidate them.

**`multiplyImplication` is not relevant here.** That fan-out fires only when disintegration is banned; in that mode `it_/int_` variables cannot be minted at all. The equi-class rewrite path targets `it_/int_` entries exclusively.

### Weaknesses

- **Mail-lag window for children.** A child LB receives an implication's index entry from `mailOut.expandedImplications` on burst N, then runs its own burst N+1 with its own visible equi-classes. If the equi-class admission and the implication install land in the same parent burst, the child sees the old form's index entry but not the equi-class until a subsequent burst — its first sanitize after inheriting the index is a no-op. The next sanitize then rewrites. Eventual consistency holds; throughput dips by one burst.
- **Mailed rewrite is local-only.** `sanitizeHashMemory` mails the rewrite to its own LB's `sameIterationInternalMail`, not to `mailOut`. Children of the LB still hold the old implication's index entry (from earlier `mailOut.expandedImplications` propagation) and rely on their own sanitize to drop+rewrite. Soundness holds because each LB's `eradicateImplicationFromLB` only touches its own LB; an empty registry at a child is a no-op erase.

## Extension — `checkForEquivalence` gates on the `fullyDisintegrated` flag

`intKnownStatements` is a map `packStatementKey(originalId, validityId) → StatementFlags`, where
`StatementFlags { bool local; bool fullyDisintegrated; }` (memory.hpp; row presence is membership per [I-85](../30_invariants.md#i-85)):

- `local` — `1` iff the expression was added to `localEncoded` (a status 0/1 local
 derivation — set as `local` in `addStatement` / `addEquality` /
 `addNegatedEquality`, and `true` on the unconditional-`localEncoded` paths:
 `prehandleAnchor`, `disintegrateExprHypothetically`, the
 `status==4` CE-fact load, and the equi-rewrite commit); `0` for a mail-origin
 (status 3) statement registered but never disintegrated. Set at insert, immortal
 in the map (survives the `localEncoded` sanitize). Preserved for provenance and
 future flag consumers; no longer the cFE gate.
- `fullyDisintegrated` — `1` iff the expression **entered `disintegrateExpr2` and
 came back fully disintegrated**: it has **no existence inside** (nothing to
 witness — e.g. a plain `in3[...]`), OR **every existence inside it got at least
 one admitted witness** (one `it_`/`int_` witness in `admittedVars` after Pass B +
 cascade — one is enough). Default `0`; only the `addExprToMemoryBlock` call site
 that consumed a `true` from `disintegrateExpr2` flips it on `expr`'s own entry
 only if admission already created that entry; the marker creates no row.

`checkForEquivalence` compares stored and incoming expressions through the complete classes at the exact disintegration scope. A matching original row must have `fullyDisintegrated` set. The index retains original statement keys and reads their current flags; ancestor and sibling completion does not suppress this scope. Direct literal completion is a fast hit, and other identities use the shared shape/scope projection cache. No Cartesian family of hypothetical expressions is enumerated.

**Why full disintegration, not locality.** Locality is too weak. The Gauss
fold-`n+1` step stalled on the dead product `9·7`: the local existence
`existence1[1,it_0_lev_4_0,7,5]` (`it_0_lev_4_0 ≡ 9`) disintegrates but its witness
is *rejected* (admission holds `in3[9,7,marker,5]`, never the `it_0_lev_4_0`-form),
so it is local yet **not** fully disintegrated. Under the step-1 local gate it
still suppressed the canonical `existence1[1,9,7,5]` — whose witness IS admitted —
and the two equivalent existences mutually blocked, so `in3[9,7,…,5]` never landed.
Gating on `fullyDisintegrated` leaves the rejecting twin unflagged (`0`), so it no
longer suppresses the canonical; the canonical disintegrates, its `it_` witness
admits against `in3[9,7,marker,5]`, and the Gauss summation closes.

**Computing the flag** (inside `disintegrateExpr2`, non-`forceDeep` path). Group
the witness vars in `newVarMap` by their `makeMarkedExpr` signature — the `it_` and
`int_` spawned from one existence share an identical marker body (the witness name
becomes `marker`), so they fall in one group; a group is "covered" when any of its
vars is in `admittedVars`. `fullDisintegrationHappened = every group covered`; an
empty group set (no existence) is vacuously true. The `forceDeep` path has no
admission pass, so it reports `newVarMap.empty` (atomic → true, any existence →
false). Erring toward `false` is always safe (cFE just disintegrates); only a wrong
`true` re-introduces the mirror bug. See
[I-72](../30_invariants.md#i-72)
and [D-108](../40_decisions.md#d-108).

Outside cFE every reader stays a presence check (`find`/`count`/`erase`); the
sacred hashburst dump's two iterations touch only `kv.first`, so the value-type
change is invisible to them.

## See also

- [`10_pipeline/04_prover.md`](../10_pipeline/04_prover.md) — `addStatement`, Pass B, and equivalence handling.
- [`20_core_concepts/04_validity_stack.md`](04_validity_stack.md) — scope-based visibility rules.
- [`20_core_concepts/03_mail_system.md`](03_mail_system.md) — `sameIterationInternalMail` is the emission target of revival matches.
- [I-9](../30_invariants.md#i-9), [I-12](../30_invariants.md#i-12), [I-21](../30_invariants.md#i-21), [I-22](../30_invariants.md#i-22), [I-37](../30_invariants.md#i-37), [I-36](../30_invariants.md#i-36), [I-40](../30_invariants.md#i-40), [I-41](../30_invariants.md#i-41).
- [D-19](../40_decisions.md#d-19), [D-57](../40_decisions.md#d-57), [D-62](../40_decisions.md#d-62), [D-63](../40_decisions.md#d-63), [G-31](../50_gotchas.md#g-31), [G-32](../50_gotchas.md#g-32).
-.
- Verifier tags: `equality1`, `equality2`, `symmetry of equality`, `symmetry of inequality` — see [`20_core_concepts/08_proof_tags.md`](08_proof_tags.md).

## Dormant encoded rows, `usedAsPremise`, and the watch ledgers

`memory.hpp` and `usedAsPremise` classify an encoded row as used when its
SIGNED core occurs in any of the block's four current premise indexes
(`usedAsPremiseCorePairs`, rebuilt by `refreshUsedAsPremise` at every rule
staging close from `forEachPremiseCore`); equality, negated equality and
anchors retain their dedicated handling. Nothing else makes a row used —
not a goal, not a negated statement, not a contradiction seed — and the same
signed membership is the only device-upload criterion. Those rows are
WATCHED instead: `Memory::negationLedger` mirrors `intKnownStatements` for
the known negated non-equality, non-anchor rows and `Memory::goalLedger` is
derived from `intToBeProved`, both keyed `(scope, unsigned core)` with sorted
`(canonical id, original id)` runs, where the canonical id is the original
text under every class bucket visible at the scope (`canonicalWatchFormAt`,
root → scope to a fixpoint). The class-merge seam re-keys the merged scope
and its descendants, stages the moved negation entries for a re-check, and
imports the ancestor negated rows naming a member into the merged scope's
run; a new negated row imports itself into the descendant class scopes that
can rewrite it. Once per `standardProcessing`, after the class application
and before the two discharge sweeps, `materializeWatchHits` turns every
canonical contradiction or goal pair into the exact pair the sweeps fire on:
positive arrivals probe the negation runs, negated arrivals / re-checks /
goals probe the known registry when their peer's signed core is used as a
premise and otherwise a per-step batch sweep, live `_ordis_` branches probe
their assumption's negation — and a hit writes the canonical twin of each
side at the probing scope (`commitDerivedLocalRow` with one `equality1`
bridge) plus, for a goal, its literal spelling. See
[D-380](../40_decisions.md#d-380)
and [I-246](../30_invariants.md#i-246).

The deposit door still processes the own-scope canonical replacement. A rejected
original spelling gets no new known entry or levels row; its producer history
remains when an equality bridge cites it. `canonicalizeUnusedDeposit` supplies
the same admission rule to statement products. A stored unused row is retained
as admitted: subsequent class changes neither rewrite it nor remove it through
`cleanUpExpressions`. Its known entry, levels, locality and ordinary list
membership remain. Scope wipes and block discharge still remove rows normally.

`applyEquiClasses` uses newly eligible signed cores to revisit stored rows below
the class waterlines. The statement-major closure generates required forms under
comparable classes, with products at the deeper scope. Older rows reactivated
without a class delta receive cleanup only after the closure completes; delta
scopes keep the normal `standardProcessing` cleanup order. Per-call completed
core coverage prevents reopening the same core in every fixpoint round. No
suppression set or restoration scan exists. A negated arrival activates
nothing; the watch drain pairs it. For a newly used row, affected-row
collection continues through applicable classes after its first hit so a
child delta cannot hide the older source scope's cleanup requirement.

`LbMemory` stores current signed premise eligibility and completed signed
class coverage in tags 822 and 824 (tag 823, the former retained unsigned
usage, is retired; tag 825 too). The two sets deload and clear on discharge.
The watch ledgers and the two pending lists ride the persistent arena (never
deloaded, reclaimed at discharge); fact-template clones copy the negation
ledger alongside the facts. GPU upload is controlled exclusively by signed
premise cores.

See [I-225](../30_invariants.md#i-225) and
[D-361](../40_decisions.md#d-361).

### Why the excluded shapes need separate handling

Core usage is insufficient by itself for shapes
with dedicated consumers. Equality substitution remains logically valid; the
exclusions preserve the current registration, propagation, and context contracts.

- **Equalities:** `logical_block/statement_memory.hpp` and `addStatement` register the original pair
 through `addEquality` and then update the class. Once that class contains both
 arguments, canonicalizing the pair can turn `(=[a,b])` into `(=[a,a])`.
 `fillMailOut` sends the local delta rows, not the class store, so replacing that
 delta would remove the nontrivial relation from outgoing statement mail. A
 canonical-only equality list therefore needs an explicit way to preserve
 class-building information for recipients and its existing proof history.
- **Negated equalities:** `applyEquivalenceClassToNegatedEquality` runs at
 arrival and after relevant class changes, including comparable scopes.
 `dischargeContradiction` scans encoded rows and probes the exact negated
 spelling in known facts. A canonical-only extension must coordinate those
 consumers, mirrored registration, and the equal-endpoint case; removing the
 shape gate alone does not establish that contract.
- **Anchors:** the runtime treats their argument tuples as positional context.
 `canonicalFormAtScope` and `collectEquiAffectedRows` exclude them, and the
 admission hooks independently refuse substitutions that change anchor slots.
 Anchor predicates can occur in rule premises; those signed cores are already
 eligible under the ordinary premise test. Canonicalizing an unused anchor row
 would still require checking its context and export consumers.

### Ordering of OR deposits

`ordisMerge` consumes each owning statement buffer in its existing order:
post-registration emissions, committed equivalence products, canonicalized
registry rows, and negated-equality products. No extra expression sort runs at
these four consumers. Class traversal and hash-engine ordering retain their
own contracts.

The first matching deposit selects any capped contradiction witness. A different
repeatable order can therefore change proof bytes without being nondeterministic.
The direct production-path regression in `tests/test_equi_reshuffle.cpp` checks
that reversed p/q input selects the corresponding first witness and repeated
identical input repeats that selection. Full and shortcut repeated-run acceptance
is recorded in `ordis_merge_sorting_plan.md`. See
[D-343](../40_decisions.md#d-343).

Canonical-only history bridges follow the execution mode: ordinary proof
processing requires the source history and keeps the assertion at the rewrite
site. Counterexample-filter clones intentionally omit fact origins and never
export proof certificates, so their canonical-only replacements update state
without producing these bridges. Admission first applies the existing raw
axed-variable and ancestor-known refusals; rewritten rows also pass the target
axed-variable and ancestor-known gates.

### Open question: replay resets and active class boundaries

`EquivalenceAdmissionRewriter::resetResentExpressionRegistries` (in the ordis hook the batched pair `resetResentRegistryKeys` and `eraseResentRows`, one sweep per hook call) and `RejectedSourceReplay::resetParkedOrStatementRegistries` erase statement rows without the position repair used by subtree cleanup and ordinary removal. Audit those replay paths together with the transient class table inside a running application before claiming universal erase coverage. This is separate from the repaired subtree-wipe path.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
