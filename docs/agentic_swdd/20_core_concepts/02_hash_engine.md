<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Core concept — Hash engine `[DRAFT]`

> The central bet of GL's architecture: **inference is memory access**. Every proof step reduces to a hash lookup on the structural signature of known expressions. This chapter walks the data structures and the query/fire cycle that make that bet real.

---

## Request execution owners
`HashRequestGenerator` owns deterministic grow enumeration and stump generation. Its private gate helpers query the existing hash-rule index. Each `BurstSink` passes dependency-ready requests to `HashRuleEvaluator`, which writes the same sealed firing records and applies them at the existing finalize boundary. Neither algorithm borrows the prover coordinator. Counter readers preserve the separate per-part, worker-task and batch reset scopes; they do not introduce an admission or stopping policy.

## The core idea

Traditional theorem provers explore a search tree of tactics, rewrites, or unifications. GL takes the opposite direction: treat inference as a *lookup* problem. Every implication rule `premise_pattern → conclusion_template` is hashed by its premise pattern into a per-LB index. When an LB has a registered statement (an `intEncodedStatements` row) that matches the premise pattern (after parameter binding), the rule fires and the conclusion is emitted.

Two invariants make this work:

1. **Expressions are structurally canonical.** Two theorems that differ only in bound-variable names must hash identically. This is what `reshuffleTheorems`, `precompileStructuralOperators`, and normalisation routines ensure.
2. **Hash keys are integer-encoded.** The string form is canonical but slow; every hot-path lookup uses the `IntNormalizedKey` form.

Result: each step is O(1) amortised. Exploration happens by breadth — multiple LBs fire in the same cycle, and the mail system carries derivations across the grid between cycles.

---

## Algorithm interfaces above hash storage

`inference/hash_key_builder.hpp` / `HashKeyBuilder` owns normalized integer-key
construction. Whole-key and reverse-map operations preserve their existing byte
language and inline implementations. Incremental clients request `buildPrefix`,
`append`, `checkpoint`, `restore` and `length`; the two fixed scratch arrays and
three counters are private. A checkpoint contains only the counters, so trying
another candidate retains the existing prefix mappings without copying or clearing
the arrays. Output buffers remain caller-owned and the same capacity assertions apply.

`logical_block/admission_policy.hpp` / `AdmissionPolicy` owns the classic, long-key,
local-unchangeable, OR and relation qualification queries. It borrows only const
`CompiledDefinitionRegistry` and `ProverParameters` references. Classification
mutates neither logical-block storage nor configuration. Existing rule installation
and request generation ask these components for decisions and key construction.
Hash containers, rule ownership, deferred drains, scope conditions and thresholds
retain their existing contracts.

`logical_block/admission_marker_builder.hpp` / `AdmissionMarkerBuilder` owns
qualification-driven marker generation, including the independent demand passes
and private per-element variant installer. It borrows definitions, admission policy,
parameters and the existing permutation table. The vector and span operation
interfaces retain the same arguments, default values and staging-window ownership.

`inference/hash_rule_index.hpp` / `HashRuleIndex` owns deterministic owner/index
insertion, removal and staged flush operations. Higher algorithms request these
operations on the supplied `HashMemory`; the four physical stores and their arenas
remain owned by the logical block. Private flush planning and section merge helpers
preserve first-touch identifiers, sorted-unique runs, mid-window binding and the
close-window drain sequence. Existing inline implementations remain inline.
`StagedSortedUnique` has its own header and keeps traversal state private; its
replayable merge iterator uses the same caller-owned scratch arena.

`inference/implication_multiplier.hpp` / `ImplicationMultiplier` owns the ordered
variable-binding copy expansion used by rule installation and removal. It borrows
const definition and parameter references plus the existing counterexample-phase
flag. `BindingPartitions` owns the two arena partition-enumeration overloads;
`CopyRef` is a separate passive arena-reference record. The multiplier's numeric
ordering helper is private. Caller-owned output arenas, partition limits, anchor
restrictions and copy order are unchanged; the heap/span twin tests exercise the
component directly.

`inference/hash_rule_classifier.hpp` / `HashRuleClassifier` owns OR-introduction
and subset-exclusion shape queries plus instantiated OR-leaf enumeration. It borrows
only the compiled-definition registry. The shared scratch-negation operation lives
beside its heap twin in `DefinitionExpansion`; it still returns either an input
slice or caller-arena bytes. Rule classification and expression transformations
retain their existing conditions, bounded traversal and allocation behavior.

Rule installation delegates unary-source registration to
`logical_block/unary_rule_memory.hpp` / `UnaryRuleMemory`. Source indices and
demand enrollment remain in their existing logical-block stores. The interface
borrows only the compiled registry and accepts the target block explicitly. It
preserves all registration and queueing order; composition drains retain their
existing execution points.

Canonical admission/source matching uses
`logical_block/canonical_template_matcher.hpp` / `CanonicalTemplateMatcher`,
which delegates scope initialization to `Canonicalizer`. The matcher owns text
projection and matching algorithms; Memory still owns comparison indices and source
records. Shared chain renderers remain inline in `DefinitionExpansion`, and the
existing string-prefix helper now shares `str_ops.hpp` with its span twin.

`logical_block/rejected_source_memory.hpp` / `RejectedSourceMemory` owns parked
source registration, canonical source indices, pending/completed existence receipts
and retirement of completed parked sources. It borrows parameters, `Canonicalizer`
and `CanonicalTemplateMatcher`; every record remains in the supplied block's
existing stores. Receipt erasure, pending-index maintenance and single-key source
refresh are private helpers. Public operations preserve replay-state values,
registration order, canonical source identity and removal order.

## The main data structure — `HashMemory`

Defined at [`memory_infra/hash_memory.hpp`](../../../GL_Quick_VS/GL_Quick/src/memory_infra/hash_memory.hpp), `HashMemory`. The four instances share resident immutable key payload through `LbMemory::sharedNativeKeys`; their memberships, identifiers and owner runs remain independent. Fields:

| Field | Type | Role |
|---|---|---|
| `encodedMap` | `TypedColdBlobMap<NormKey, LocalMemoryValue>` (cold) | The actual rule registry. Key = premise-pattern's integer-normalised hash; value = the run of rules with that same premise shape. Cold blob map ([I-98](../30_invariants.md#i-98)). |
| `admissionMap` | `TypedColdBlobMap<int64_t, AdmissionMapValue>` (cold) | Populated by `updateAdmissionMap` from consumer-side registrations during Pass B, keyed at the installing rule's scope (never `main` for a branch-scope rule — [D-369](../40_decisions.md#d-369)). Consulted by `isAdmitted`. Key shape: **`u_`-prefixed** template; cold-stored ([I-99](../30_invariants.md#i-99)). |
| `admissionMapIntegration` | `TypedColdBlobMap<int64_t, IntegrationEntry>` (cold) | Integration-side analog. Populated at `logical_block/admission_memory.hpp` from `le.signature`. Consulted by `isAdmittedIntegration`. Key shape: **`u_`-prefixed** template (same as `admissionMap`). Value is a nested `map<IntInstruction, ValueIdSet>` flattened one `IntegrationEntry` blob per inner entry; cold-stored ([I-99](../30_invariants.md#i-99)). |
| `admissionSetIntegration` | bookkeeping | Secondary integration-admission set. Populated at `compiler/definition_expansion.cpp` via `removeUPrefixFromArguments`. Key shape: **bare concrete** with `repl_` preserved. Consulted by Pass B `int_` fallback and by `applyEquivalenceClassToRejectedMapIntegration`. |
| `admissionMapIntegrationDouble` | `TypedColdMap<int64_t, int64_t>` (cold) | Double-marker integration demands ([D-492](../40_decisions.md#d-492), [I-302](../30_invariants.md#i-302)). Key = packed `(templateId, validityId)` of an OR disjunct of a flattened root integration goal that has two distinct open arguments: the earlier open position `marker1`, the later `marker2`, every other argument `u_`-prefixed. Value = 0, carrying no information (the map shape stays because the Rule-14 hashburst dump reads it); no instruction. Filed by `AdmissionMemory::updateAdmissionMapIntegrationDouble`; probed by `AdmissionMemory::isAdmittedIntegrationDouble` (admission route 5) at the probe scope and every strict ancestor through the derived class-aware index `CanonicalUseIndex::integrationDoubleMatches`. Scope-wiped beside `admissionSetIntegration`, deloaded at HashMemory band tags 98/99. |
| `rejectedMap` | `TypedColdBlobMap<int32_t, RejectedMapValue>` (cold) | Algebra-side rejection buffer — `it_` vars that failed Pass B admission. Consulted by `revisitRejected2` when a new admission key arrives. Packed `(templateId, validityId)` int32 key, **bare concrete** template form (Pass B's `makeMarkedExpr` output); cold-stored ([I-99](../30_invariants.md#i-99)). |
| `rejectedMapIntegration` | `TypedColdBlobMap<int32_t, RejectedMapIntegrationValue>` (cold) | Integration-side rejection buffer — `int_` vars that failed Pass B admission ([D-19](../40_decisions.md#d-19)). Consulted by `revisitRejectedIntegration2` and rewritten by `applyEquivalenceClassToRejectedMapIntegration`. Key shape: **bare concrete** (same as `rejectedMap`); cold-stored ([I-99](../30_invariants.md#i-99)). |
| `varsInRejectedMapIntegrationKeys` | `TypedColdSet<NameId>` (cold cache) | Monotonically-growing set of non-marker args appearing in any `rejectedMapIntegration` key. Used by the equi-class-rewrite helper for fast class-overlap short-circuit; cold-stored, never wiped. `NameId` is the current 32-bit identifier type. |
| `admissionMapOrdis2` | `TypedColdBlobMap<int64_t, AdmissionMapOrdis2Value>` (cold) | The admission half of the ordis2 demand/park PAIR ([D-267](../40_decisions.md#d-267), pure ordis — no machinery shared with the algebra or integration maps): key = packed `(templateId, validityId)` whose template half is a fully-GROUND compound premise text (polarity verbatim, I-175) — the one missing premise of a rule that matched everything else. Populated at the post-fixpoint drain when an `ordis2Demand`-tagged marker variant fires; its drain wakes `rejectedMapOrdis2` under the SAME key language, and `consumeOrLeavesCohort`'s route (c) probe consumes the demand on cohort open. A demand fired by a goal near-miss variant (`Ordis2DemandRecord::goalNearMiss`) never lands here: the drain hands it to the goal-driven rule search ([D-574](../40_decisions.md#d-574)). Value = `{sourceImplId, levels}` audit context; facets base+63..66; scope-wiped like the other rejection buffers; equi hook `applyEquivalenceClassToAdmissionMapOrdis2` (drop-and-rekey) at both `applyEquiClasses` seams. |
| `rejectedMapOrdis2` | `TypedColdBlobMap<int64_t, RejectedMapOrdisValue>` (cold) | The rejected half of the ordis2 pair: the SAME or heads that park in `rejectedMapOrdis` additionally file here under each ELIGIBLE disjunct's clean GROUND text (compiled non-atomic entity — predicates included, `operators` not consulted; arity ≥ `kOrdis2DemandMinArity`; polarity verbatim) at the cohort parent's validity — the demand map's key language, so `drainAdmissionKeysOrdis2`'s wake (`revisitRejectedOrdis2`: snapshot → erase key → un-know → re-mail) is a plain key rendezvous. Value family REUSED from `RejectedMapOrdisValue`; facets base+67..70 + the dedicated `ordis2RevisitInProgress` guard at base+71; scope-wiped; equi hook `applyEquivalenceClassToRejectedMapOrdis2` (I-37 drop-and-mail) at both seams. The algebra key-gain seams never touch it (different key language — a cross-probe would double-wake operator-application disjuncts). |
| `rejectedMapSubproof` | `TypedColdBlobMap<int64_t, IntegrationEntry>` (cold) | Parked implication subproofs ([D-406](../40_decisions.md#d-406)): an implication element of an and whose sibling non-implication element is not yet known at the preparing scope is kept here instead of opening its boundary subproof. Key = packed `(templateId, validityId)` of that element's clean ground text normalized through the preparing scope's classes (canonical; class-equal gates share one key). The value keeps the instruction's own text (the and's integration rule expects it). Value family REUSED from `admissionMapIntegration` (`IntegrationEntry`, canonical run under `DecodedInstructionLess`, RMW through `ArenaIntegrationMap`): one stored instruction per parked implication — entity 0 the implication (signature, elements, head last), entities 1..k its parent ands with their gating elements, `markedGoal` the embedded root goal. Facets base+88..91 + the dedicated `subproofRevisitInProgress` guard at base+92; scope-wiped; never cloned into a counterexample LB (asserted empty). Written by `parkImplicationSubproof`; revived by `drainSubproofArrivals` → `revisitRejectedSubproof` (comparison index `CanonicalUseIndexes::subproofParkMatches`). |
| `normalizedEncodedKeys` | cold owner-run blob map | `TypedColdBlobMap<NormKey, RuleOwnerRec>`: every whole key → the sorted-unique run of the rules that installed it (`RuleOwner` = packed `(originalImplicationId, validityId)`, added by `addWholeKeyOwner` → `addOwnerToRun` at the install sites). The request generator's emission gate: `wholeKeyPresent` is a bare lookup (the owner run is never read on the hot path). A rule's removal deletes its owner; an owner-less key is erased at the end of the apply (`eraseEmptyRuns`); never scope-wiped — a key whose rules left through a subtree wipe keeps their owners, its request reaches the firing site, whose wiped `encodedMap` misses it ([I-49](../30_invariants.md#i-49)). Five facets: key Lengths + Bytes (5/6), owner-run RunStarts + BlobStarts + BlobPool (72..74). See [D-71](../40_decisions.md#d-71), [I-100](../30_invariants.md#i-100), [D-312](../40_decisions.md#d-312). |
| `normalizedEncodedSubkeys` | cold record blob map | `TypedColdBlobMap<NormKey, OwnerSet>`: every growable normalized subkey maps to one record. One- and two-premise keys store a nine-byte empty frame, with zero loose flag, signatures and owners; presence alone permits growth. `ensureEmptySubkeyRecord` installs directly, `stageEmptySubkeyRecord` stages at most one item per key per window, and `flushSubkeySection` emits the empty frame. These short keys survive rule removal and ownerless cleanup until hash-memory reset. From three premises onward, `mergeSubkeySignatures` retains the loose flag, distinct u_ signatures and exact owner/signature pairs; removal deletes one owner and cleanup erases an ownerless long key. Later whole-key and encoded-rule lookups prevent a retained short prefix from firing a removed rule. No subkey map is scope-wiped. See [I-49](../30_invariants.md#i-49) and [I-228](../30_invariants.md#i-228). |
| `remainingArgsOwners` | cold owner-run blob map | `TypedColdBlobMap<IdVecKey, RuleOwnerRec>`: one owner run per `(arg-set, NormKey)` EDGE of `remainingArgsNormalizedEncodedMap`, keyed by the edge's bytes (`remArgsEdgeKeyInto`: the arg-set id run followed by the NormKey blob's id run). Written by `insertRemainingArgsNormKeyBatch` for every distinct edge of a batch, new or already present in the forward run; the forward run stays a pure NormKey run and the derived `ReverseArgsIndex` is untouched ([I-154](../30_invariants.md#i-154)). The last owner leaving an edge removes the NormKey from the forward run and the reverse-index edge. Five facets at 78..82. |
| `copyOwners` | cold owner-run blob map | `TypedColdBlobMap<IdVecKey, RuleOwnerRec>`: one run per multiplication COPY (key = the packed copy owner `(copy text's ruleInterner id, scope)` as eight bytes) naming the RULES `(recorded text's ruleInterner id, scope)` whose install produced that copy. `addToHashMemory` lists the rule on every copy it installs; under `RuleIndexOp::Remove` it delists the rule first and touches the copy's index entries and LMVs only when no other rule owns the copy — under `allow_multiplication` two different rules can multiply into one copy text (heads differing only in the order of the equated variables), and the copy's index owner is the copy, not the rule ([I-49](../30_invariants.md#i-49)). Five facets at 83..87. |
| `premisePatterns` | cold owner-run blob map | `TypedColdBlobMap<IdVecKey, RuleOwnerRec>` read by the request statement filter ([I-300](../30_invariants.md#i-300)). Three key families: a premise SHAPE key (`[1, name, polarity, first-appearance argument numbers]`) → the distinct `u_` slot masks seen for it; a PATTERN key (`[2, name, polarity, per slot (literal id, 1) or (variable number, 0)]`) → the scope-major owners `(install scope, copy id)`; a CLOSED-SCOPE key (`[3, scope]`) → one record while a subtree wipe has closed the scope and no install reopened it. Installs stage one mask and one owner per copy and premise (staging section 8); a rule removal deletes its owner, the end-of-apply sweep erases an emptied key; the wipe deletes no owner. Bound to the block's shared byte pool like `copyOwners`. Five facets at base+93..97. |
| `originals` | cold owner-run blob map | `TypedColdBlobMap<IdVecKey, RuleOwnerRec>`: every installed implication chain (`premises ++ head` as `ruleInterner` ids) → the run of the rules that installed it (the same chain text at two scopes is two owners). Read by the integration premise matcher, the necessity check and the dump; a rule removal (`removeOwnerFromRun`) rewrites the owner run in place and keeps the key until the end-of-apply sweep `eraseOwnerlessEntries` erases every owner-less chain (renumbering the survivors — the derived head index is marked stale there). Five facets: key Lengths + Bytes (51/52), owner run 75..77. |
| `originalsHeadIndex` | derived side-index (no facets) | `OriginalsHeadIndex`: head operator name → the ascending list of `originals` chain ids whose head carries it, and masked head key (`maskedHeadKey`: the sign of a negated head, name + arguments with the one constant slot, or the two constant slots beside at least one other argument, masked) → the ascending list of chains the necessity check can accept, probed once per deposit slot and once per pair of deposit slots with the deposit's sign ([D-483](../40_decisions.md#d-483)); built and extended lazily through `IntegrationProcessor::ensureOriginalsHeadIndexBuilt` (the one builder, called by `checkNecessityForEquality`), cleared at every teardown, stale at the reload doors and after an owner-less sweep that erased a chain, never deloaded ([I-242](../30_invariants.md#i-242), [D-373](../40_decisions.md#d-373), [D-415](../40_decisions.md#d-415)). The resident scalar `goalScanChainMark` beside it is the goal-rescan floor: every goal has been visited against the chains up to it ([I-272](../30_invariants.md#i-272)). The resident scalar `contradictionChainMark` is the contradiction-scope demand's floor: every registered contradiction scope has been matched against the chains up to it, remapped by the owner-less sweep the same way ([I-331](../30_invariants.md#i-331)). |

**Three key shapes, five maps.** Admission MAPS (`admissionMap`, `admissionMapIntegration`) are u_-prefixed templates; the double-marker map `admissionMapIntegrationDouble` uses the same template form with two markers, and its probe builds that form from Pass B's bare single-marker constituent (`AdmissionMemory::makeDoubleMarkerQueryScratch`). The admission SET (`admissionSetIntegration`) and rejection MAPS (`rejectedMap`, `rejectedMapIntegration`) are bare concrete instances. Lookups must transform at the boundary — u_-prefix when querying the maps, strip u_ when querying the set/rejection side. Storage keying: the algebra group (`admissionMap`, `admissionStatusMap`, `revisitInProgress`) is packed `(templateId, validityId)` via the dedicated `Memory::templateInterner` ([D-132](../40_decisions.md#d-132)) — write sites `mintTemplateKey`, probes the non-minting `lookupTemplateKey` (a never-interned template is a definitive miss); the key-SHAPE contract above is unchanged, the shapes are the interned strings. The algebra group + `varsInAdmissionMapKeys` AND the integration twins (`admissionMapIntegration` / `rejectedMapIntegration` / `varsIn{Admission,Rejected}MapIntegrationKeys`) are now COLD-stored on the cold-map family (`admissionMap` / `rejectedMap` / `admissionMapIntegration` / `rejectedMapIntegration` → `TypedColdBlobMap`, the satellites → `TypedColdSet` / `TypedColdMap`; [I-99](../30_invariants.md#i-99)). The admission SET side (`admissionSetIntegration` / `triggersForAdmissionSetIntegration`) stays on the heap. See [`10_pipeline/04_prover.md`](../10_pipeline/04_prover.md#admission-rejection-map-key-shapes) for the full table and [G-32](../50_gotchas.md#g-32) for the silent-miss pitfall.

A `LocalMemoryValue` ([`memory.hpp`](../../../GL_Quick_VS/GL_Quick/src/memory.hpp)) packages each rule entry. Id form per `D-133`: every string field is an id in the owning LB's `Memory::ruleInterner` (int32), minted at the single-threaded install sites; the parallel hashburst only decodes (array-index const refs).

| Field | Role |
|---|---|
| `valueId` | The normalised conclusion template — the "what to emit". For head-implication LMVs (insert path 1 below) this is the implication's conclusion; for marker LMVs (insert path 2 below) this is the marker-form expression with `"marker"` substituted at the output-arg slot. The firing path substitutes on the decoded template. |
| `levels` | `std::set<int>` — the rule's installation level set: LB depths whose state contributed to the rule being installed here. Propagates into the firing's derived statement via `union(rule.levels, premise.levels)` at fire time. Must be empty for rules deposited through mail (so only the receiver's premise side determines the final union) — see [I-51](../30_invariants.md#i-51) and [`02_glossary.md::levels`](../02_glossary.md#levels). |
| `originalImplicationId` | The full MPL of the original implication — for provenance (decoded into the firing's origin pair). |
| `justification` | The closed `RuleJustification` enum (`ruleJustificationFromString` asserts on an unknown string). `implication` on every installed rule; `none` on marker LMVs. |
| `keyIds` | The marker-key sub-key chain (`vector<int32_t>`, built from premises that don't reference the output arg). Carried ONLY by marker LMVs (insert path 2); head-implication LMVs leave it empty — the premise pattern they key on lives in `IntNormalizedKey`. |
| `remainingArgIds` | Arguments not yet bound — consulted during binding. Decoded-lex sorted id storage (set-iteration encode order). |
| `validityId` | The NameMap id of the scope in which this rule is active, minted at install (the same encode that feeds the owner record); the firing path's scope-comparability check reads it with no per-firing lookup. |
| `isMarker` | Install-time classification: the head template contains `"marker"`. Replaces the per-firing substring scan that used to branch head vs marker firing. |
| `productOfDisintegration` | (D-32, 2026-05-01) Marks rules whose chain has at least one premise with an arg starting with `"u_"` — the reserved bound-variable-placeholder prefix introduced by `prefixArgumentsWithU` during disintegration. Stamped only on head-implication LMVs (insert path 1); marker LMVs default to `false`. Consumed by `checkLocalEncodedMemoryStatic` to gate OR-disintegration via the `allowOrDisintegration` parameter threaded into `disintegrateExpr2`. See [D-32](../40_decisions.md#d-32) and [`07_or_branching.md`](07_or_branching.md#1-or-disintegration). |

One key can point to multiple `LocalMemoryValue`s — several different rules may happen to normalise to the same premise-pattern signature. When an LB matches the key, *all* pointed-to rules are candidate fires.

---

## The query-fire cycle

Per LB, per cycle, the elementary-step hashburst ([`proof/phase2_executor.cpp`](../../../GL_Quick_VS/GL_Quick/src/proof/phase2_executor.cpp), `performElemPhase2` → `performElem2`):

1. **Generate hash requests.** One function does this — `HashRequestGenerator::generateEncodedRequestsStatic` ([`inference/hash_request_generator.cpp`](../../../GL_Quick_VS/GL_Quick/src/inference/hash_request_generator.cpp)), and every caller now drives it the same way: with a list of **mandatory-containment terms**. The candidate is grown to a whole key and emitted only when it holds a statement from every view of at least one term. An **empty term list is a first-class input**, not the absence of one — it says no ingredient is mandatory, so every whole key is a request; the CE filter and batch 5 both pass it. The obligatory stump, its builder and the pairing merge are deleted ([D-295](../40_decisions.md#d-295)), so the target map is always `normalizedEncodedKeys`, the grow depth is always `maxKeyLength`, and there is no seed and no merge phase. Every install raises that bound to the premise count of each whole key it writes, ordinary or admission marker ([I-337](../30_invariants.md#i-337)); a key above it would never be requested.

 It emits `IntNormalizedKey` objects that correspond to each candidate match.


 Under the LB split's second dimension the call also carries a **bucket of split stumps** ([D-203](../40_decisions.md#d-203)). The search then runs once per stump in the bucket over one shared filtered statement list. The stump IS the growing candidate: the run seeds its depth-first stack with that node and grows it onward, so it owns exactly that node's subtree ([D-283](../40_decisions.md#d-283)). The stump alone is probed once before the search, because unsplit that candidate is recorded in the loop of the candidate one level up, which a stumped sub-part never runs.
 **Each candidate is built from its predecessor ([D-296](../40_decisions.md#d-296)).** Inside one stack pop of the grow search the first k premises are fixed and only the appended one walks the candidate positions, and both per-node computations are monotone under that suffix append. The normalized key renumbers arguments in first-appearance order, so the prefix's key is a strict PREFIX of the child's and appending can only mint slots the prefix had not used; the request gates are order-free conjunctions, so each collapses to a summary a premise folds into one at a time. The search therefore prepares the prefix once per pop — `buildIntNormalizedKeyPrefix` into the shared key buffer, `foldPrefixIntoRequestGates` into the gate summary — and each position rewinds the carried counters and folds in its own premise. One per-expression body serves both the whole-candidate entries (`makeIntNormalizedKeyFromEncoded`, `requestGatesPass`) and the resume path, so the key language has no second implementation to drift. The rewind must cover every carried field or the keys silently stop matching a full rebuild ([I-202](../30_invariants.md#i-202)). The two owner-set probes below are untouched by this: a whole key is hashed and byte-compared once per node however it was built.

2. **Look up.** Query `overallHashMemory.encodedMap[key]`. If the key is in `normalizedEncodedKeys`, there is a hit.

3. **Iterate `LocalMemoryValue` list.** For each pointed-to rule, attempt to bind the remaining arguments. If binding succeeds and admission holds (`isAdmitted` / `isAllowedAsOperatorInput`), fire.

4. **Fire = emit.** Build the conclusion expression by applying the binding to `value`. Compute the derived statement's `levels` as `union(rule.levels, premise[0].levels, premise[1].levels, …)` — every premise that participated in the binding contributes its level set, and the rule itself contributes its installation level set ([`LocalMemoryValue::levels`](#localmemoryvalue)). Call `addStatement` ([`logical_block/statement_memory.hpp`](../../../GL_Quick_VS/GL_Quick/src/logical_block/statement_memory.hpp)) with the expression, target memory block, computed levels, validity, local flag. Record origin entry into `exprOriginMap`. The derived levels then sit in `body.intStatementLevelsMap` (packed `(originalId, validityId)` key) and are read by the `allLevelsInvolved` registration verdict at `logical_block/goal_processor.hpp::dischargeToBeProved` whenever the derived expression matches an `intToBeProved` goal at v=main; an oversized level set silently swallows the registration (the goal still closes — see [`02_glossary.md::levels`](../02_glossary.md#levels), [G-46](../50_gotchas.md#g-46)).

5. **Broadcast.** If the new statement is at `validityName == "main"`, also push into `mailOut.statements` for other LBs to consume next cycle.

**Deposits captured + sorted, not applied inline ([D-117](../40_decisions.md#d-117)).** In the current code the fire step does not mutate the deposit containers inline. `checkLocalEncodedMemoryStatic` appends a `FiringRecord` per firing; after the request pass, `applyFiringRecords` sorts the records by a canonical content key and then applies the deposits (statements / origins / `disintegrationSignals` / the two staged admission+integration vectors). The sort makes the order-sensitive deposits (the origin cap, the `disintegrationSignals` last-write, the staged-vector drain order) independent of request-evaluation order — the property the LB-split rule-partition merge relies on ([I-77](../30_invariants.md#i-77)). The fire step does no deactivation at all — deactivation and discharge run only in phase 3's post-burst `standardProcessing` ([I-66](../30_invariants.md#i-66)).

**Validity-comparability prune at request generation ([D-105](../40_decisions.md#d-105)) — RETIRED.** The prune and its predicate `ownerSetHasComparable` are deleted ([D-304](../40_decisions.md#d-304)): no probe reads an owner scope. The firing gate `checkLocalEncodedMemoryStatic` enforces scope comparability exactly, which is all the prune ever relied on.

**Rule-partition filter at request generation ([D-119](../40_decisions.md#d-119)) — RETIRED.** `partitionAccepts`, the composite owner ids and the `g_splitProcessID` / `g_splitCount` thread-locals are deleted ([D-304](../40_decisions.md#d-304)); production never left `g_splitCount == 1`, and the expression (bucket) split is the only split dimension.

**u_ literal prune at request generation ([D-120](../40_decisions.md#d-120)).** At the subkey growth probe alone, from `kSubkeyUCheckMinElements` premises on (the whole-key record probes read key presence only since [D-305](../40_decisions.md#d-305), and the whole-key record carries no signature at all since [D-304](../40_decisions.md#d-304)), `gl::ownerSetUSatisfied(OwnerSet, exprs, count)` drops a request whose concrete argument values can satisfy no owner's u_ (unchangeable) literals. The two `normalizedEncoded*` maps are `ignoreU=false`, which normalizes every argument to a positional id and so **erases** a rule's u_ literal values — only the repetition pattern survives — so a structurally-matched request may be doomed by a u_ literal it can never match, a mismatch the firing gate (`encodedMap`, `ignoreU=true`) would otherwise catch only later. `recordUSignature` caches each owner's u_ signature — the `(linear-arg-slot, argFullId)` list of its unchangeable args, where `argFullId == NameMap::encode(arg[1])`, exactly the value the firing gate matches against the request's `argFullId` — at insert in `OwnerSet::uSignatures`. An owner with zero u_ args (or, conservatively, a u_ literal not yet interned at insert) sets `OwnerSet::hasLooseOwner`, which short-circuits the check to *keep* (the "0 u_, nothing to check" fast path). At match the request's flattened `argFullId` is compared against the cached signatures and the request is dropped only when no owner can be satisfied. The literal id is read via the non-minting `NameMap::lookup` at insert and never `encode`d at match, so the cache build mints nothing (byte-identical) and the match is read-only (safe on the shared LB the split executors read in parallel). Like D-105 it never changes which rules fire — the firing gate's `encodedMap` lookup enforces the exact u_ literal match — so it is a sound over-approximation ([I-79](../30_invariants.md#i-79)). Cost is O(1) on the common all-loose key; the subkey map is never scope-wiped, so a signature can go stale (a stale entry only weakens the prune, never makes it unsound).

**Premise-pattern statement filter ([D-485](../40_decisions.md#d-485)).** The statement filter that feeds the request window (`HashRequestGenerator::filterIntEncodedStatements`, oldest `maxNumberFilteredStatements` survivors) checks, after the shape keys and the iteration cap, that some live rule premise of the block accepts the row (`premisePatternAdmits`): the row's shape key selects the `u_` masks recorded in `premisePatterns`, the row's pattern key under each mask (its own argument at every masked slot) finds the premises with exactly those literals, and one of their owners must sit at a scope comparable with the row's that no wipe closed. The check is the union of the four hash memories, because the GPU route prefilters each logical block's uploaded rows once for all its filter calls (`Phase2ProjectionArena::appendLogicalBlock`) and both routes must keep the same row list. Each part is necessary for the row to take part in a firing request, so the filter stays an over-approximation of the firing gate; what changes is which rows the fixed window holds.

---

## Locality semantics: a local rule can fire against entirely non-local premises

A rule that lives in `body.localHashMemory{,Delta}` is "local" — this LB owns it. The premises that fire it can be drawn from anywhere visible at this LB, including parent-scope rows the LB sees only via `nm.comparable`. **A local rule firing on 100 % non-local premises is correct rule semantics**, not a bug.

**Both ends of the firing pipeline use `nm.comparable` for scope inheritance ([D-55](../40_decisions.md#d-55) Part C, 2026-05-09).** Request generation has always honoured ancestor/descendant inheritance via the NameMap ancestry reader `nm.comparable` (see the four-call fan-out below). The firing site (`checkLocalEncodedMemoryStatic`'s head-LMV scope check) was on a stricter strict-eq + main-mask rule until `D-55`'s Part C: pre-fix, a rule at scope `S` could only fire on facts at scope `S` or at `"main"`, even though the matching request had already been emitted under the broader comparable rule. The mismatch was masked while every load-bearing rule of this shape was a v=main universal (chapter-6 / chapter-11 mirror reformulations). When `D-55` Part B removed those parents from `theorems.txt` and the K-impls took over the role at the OR's non-main scope, the strict-eq mismatch surfaced and rejected legitimate cross-scope firings (rung-1 stalled at burst 5 toBeProved=4). Part C realigned the firing-site check to `nm.comparable`, matching the request-generation rule. Both ends now agree on what scopes are mutually visible.

This is encoded in the fan-out at request generation. `performElem2` ([`proof/phase2_executor.cpp`](../../../GL_Quick_VS/GL_Quick/src/proof/phase2_executor.cpp)) calls the one generator `generateEncodedRequestsStatic` four times, each call pairing a rule pool with a mandatory-element source, so that every emitted request contains at least one mandatory element and the mandatory source decides whether the request can be local-free.

The former batches 2 and 3 are now ONE call under the mandatory-containment control (`D-298`): they always read the same rule pool and the same statement universe and differed only in what they obliged a request to contain, which the terms now state directly. The other three batches are on the same control (`D-295`) but keep their own calls: each reads a DIFFERENT rule pool, and the rule pool decides the key length, the target map and every owner probe.

| # | Rule pool | Mandatory ingredient | Local mandatory? |
|---|---|---|---|
| 1 | `body.workingMemory` | `body.intLocalEncodedStatements` | ✓ |
| 2+3 | `body.overallHashMemory` | `body.intLocalEncodedStatementsDelta`, OR (`body.intExternalStatements` AND `body.intLocalEncodedStatements`) | mixed |
| 4 | `body.localHashMemory` | `body.intExternalStatements` | ✗ mail (v=main) |
| 5 | `body.localHashMemoryDelta` | none — empty term list | ✗ |

Batch 5 is the one with no mandatory ingredient at all. Its rule pool is what is new, and a rule that lands this burst has to meet every visible statement, not only the ones that arrived with it — so every whole key of that pool is a request. Naming the whole statement universe as a term would say the same thing and then pay a per-candidate mask test that can never fail.

Batches 4 and 5 are the asymmetric ones — **deliberately** non-local:

- **Batch 5** ([`inference/hash_request_generator.cpp`](../../../GL_Quick_VS/GL_Quick/src/inference/hash_request_generator.cpp)). When a fresh rule lands in `localHashMemoryDelta` this cycle, it must fire against every pre-existing visible statement, not only against statements that arrived this cycle. Hence it names no mandatory ingredient at all.
- **Batch 4** ([`inference/hash_request_generator.cpp`](../../../GL_Quick_VS/GL_Quick/src/inference/hash_request_generator.cpp)). Symmetric for the existing `localHashMemory` against fresh mail: every rule already at this LB needs to be tested against new mail rows.

Empirical confirmation: in the `IncubatorGauss1` run, 36 of 6398 firings at the SE2 LB carry **zero** premises in the LB's local statement set (then the string-keyed `localEncodedStatementsSet`; today's packed-key `intLocalEncodedStatementsSet` carries the same membership). The matching Peano universal-x rules have premise `(in[X, 1]) | v=main` among their chain. One example whose mathematical content is uncontroversially sound:

- `(>[1](in[1,u_1])!(in2[1,u_2,u_3]))` — Peano P4 lifted: for any `v ∈ N`, `s(v) ≠ 0` (`0` is not a successor of any natural). One chain element, head is the negation of the in2-successor relation.

The β template that fires at Step 3 of the FTA ladder ([`docs/fta_ladder/rung1/current_proof_state.md`](../../fta_ladder/rung1/current_proof_state.md) §"Step 3 — Peano β template fires at Branch A") has shape `!(=[u_p, u_i0]) → (existence0[u_N, u_p, u_s])` — the negated-equality premise gates predecessor existence. An earlier draft of this paragraph cited a single-`(in[X, N])`-premise version of the β template; that shape, taken literally, would assert *every* `n ∈ N` has a predecessor (false for `n = 0`), so it cannot be the actual installed rule. The original measurement was likely seeing the multi-premise mirror reformulation (anchor + `(in[v, N])` + `!(=[v, i0])`) but only counting the `(in[X, N])` chain element as the "single premise" because the anchor and equality premises had been satisfied via inheritance from v=main / the LB's ancestor scope. The "zero local premises" claim survives that re-reading; the load-bearing point ([D-29 clause 2](../40_decisions.md#d-29) — block disintegration when none of the matched premises is local) does not depend on the exact β-template shape.

The premises are anchor-deposited at v=main; the SE2 LB sees them via `body.intEncodedStatements` but never deposits them into its own `intLocalEncodedStatementsSet`.

**What this implies for disintegration.** Emission of these firings is fine. **Disintegration** of them at a non-anchor LB mints fresh `it_*` / `int_*` variables and historically exhausted the former 16-bit NameMap in incubator + `!ban_disintegration` mode. Name identifiers are now 32-bit, so that particular ceiling is gone, but the uncontrolled non-local disintegration remains a semantic and runtime fan-out error. The D-29 clause-2 gate is the architectural answer — block disintegration when none of the matched premises is in the LB's `intLocalEncodedStatementsSet` (an O(1) packed-key probe on the request's integer rows since D-129). The gate is *not* redundant with request generation; it corrects for Batches 4/5 at the disintegration site rather than tightening request emission.

---

## Writing rules — `addToHashMemory`

Defined at [`inference/hash_rule_installer.cpp`](../../../GL_Quick_VS/GL_Quick/src/inference/hash_rule_installer.cpp), `HashRuleInstaller::addToHashMemory` (extracted from `inference/implication_multiplier.cpp`, 2026-05-04). Takes a `(chain, head)` pair, normalises, keys by the chain's integer hash, and installs into `encodedMap`. Also invokes `multiplyImplication` (still in `inference/implication_multiplier.cpp`, called cross-TU) to install Bell-partition equalised copies (see [prover chapter](../10_pipeline/04_prover.md#multiplyimplication-bell-partition-equalisation)).

Every theorem-load path ends at `addToHashMemory`:

- Initial external-theorem load → `addTheoremToMemory` → `addToHashMemory`.
- Per-conjecture registration → `addTheoremToMemory` (with `proved=false`) → `addToHashMemory` for the premises.
- Broadcast receipt → `addStatement` (for a pure statement) or `addTheoremToMemory` (for a derived implication).
- Compressor Phase 1 → direct `addToHashMemory` for every rule in the pool.

**Rules that are never written ([D-596](../40_decisions.md#d-596)).** The totality rule of an anchor operator (x ∈ N ∧ y ∈ N → ∃ z ∈ N: (x, y, z) ∈ +) would fire for every pair of typed numbers. `CompactRuleProcessor::installExpandedImplication` does not call `addToHashMemory` for a rule [`logical_block/demand_built_rules.hpp`](../../../GL_Quick_VS/GL_Quick/src/logical_block/demand_built_rules.hpp) / `analyzeDemandBuiltRule` accepts; its instances are built from the demand side (prover chapter, *Demand-built rules*). The same holds for a typed rule with an OR head, such as a ≤ b ∨ b ≤ a over every pair of typed numbers ([D-597](../40_decisions.md#d-597)). Its instance is built when a fact refutes a disjunct, a case on a disjunct is demanded, or the instance is a goal ([I-347](../30_invariants.md#i-347)). `addToHashMemory` asserts that no rule it writes has a premise that can match such a head's instances ([I-346](../30_invariants.md#i-346)).

### The install writes are staged per absorb window (D-333)

Every record an install produces for the seven index maps — the head and marker LMVs of `encodedMap`, the whole-key / originals / copy-owner / remaining-args-owner runs, the subkey `OwnerSet` records, the remaining-args `NormKey` runs — goes through a staging twin (`stageLmvIdsRecord`, `stageOwnerToRun`, `stageWholeKeyOwner`, `stageEmptySubkeyRecord` / `stageSubkeySignatures`, `stageRemainingArgsNormKeyBatch`) into the worker slot's staging bound to the instance (`memory_infra/rule_index_staging.hpp`, `ruleStagings.acquire`), not into the cold map. The window's closer — the end of the two mail absorbs in `standardProcessing`, the end of `applyEquiClasses`, or the exit of the `addToHashMemory` that closes its LB's outermost window (`Memory::ruleStagingArmDepth` back to 0: LB seeding, CE facts), or the end of each compressor setup window in `Compressor::runPhase1` — calls `HashRuleIndex::closeRuleIndexStaging`, which flushes the LB's four instances (`flushRuleIndexStaging`; a window's nested rounds may have staged into any of them) and releases their bindings. The flush rebuilds each touched map once in key-id order — in place and backwards from the end (`HashMap::assignAllRunsGenerated` → `BlobCsrValueStore::rebuildRunsInPlace`: one emit pass buffers the produced runs, then only the tail behind the first produced key is rewritten; carried-over stretches move once as ranges) — with every key's run merged as the sequential doors would have merged it, then mints the new keys in first-touch order. A removal (`removeRuleFromHashMemory`) and the end-of-apply compaction (`finishRuleRemovals`) flush first and use the direct doors; the slot's stagings are asserted empty at every seam a reader reaches (the phase-2 producer, the gen-arena releases, the owner-less sweep). The cold bytes after a flush are byte-identical to the sequential result — the direct doors survive as the twin oracles the differential tests run (`test_rule_index_staging.cpp`).

**Premise orderings of an install ([D-543](../40_decisions.md#d-543)).** A rule's whole keys and subkeys are built only for premise orderings whose operator names never decrease. The four install loops (`HashRuleInstaller::addToHashMemory` whole keys and subkeys, `AdmissionMarkerBuilder::installAdmissionMarkerVariants` whole keys and subkeys) reach them through `HashKeyBuilder::forEachSortedOrdering` and `forEachSortedPrefix`: a depth-first walk over the unused premise indices in ascending order that drops a branch at the first decreasing name. The whole-key loops get exactly the sorted permutations of the permutation table, in table order; the subkey loops get every sorted prefix once, in the order a scan of the table first reaches it, the normalized key of a prefix extending its parent's by one premise (`HashKeyBuilder::checkpoint` on entering, `restore` on leaving). The table entry for the premise count still gates the install (more premises than the table holds: no whole keys, no subkeys). A scan of the table met each prefix once per permutation that starts with it; the repeats staged contributions the flush's sorted-unique merge dropped, so the records and their first-touch order are unchanged.

Per multiplied copy, `KeyEncodeMemo` (memory.hpp) encodes each premise text once, lazily at the first read (the NameMap first-touch order is unchanged); every permutation, every prefix and every admission derivative reads it, and each permutation's normalized key is built once incrementally (`appendExprToIntNormalizedKey`) with the prefix keys taken as the slots written so far. A status-3 install (the incubator's external-mail absorb) writes `overallHashMemory` and the per-burst `workingMemory` from ONE such build: `addToHashMemory` takes the working instance as an optional `secondTarget` and stages every record to both instances' stagings (the second target's admission triggers and admission passes run against its own maps, after the first's, per copy).

### Two `encodedMap` insertion sites — head LMVs vs marker LMVs

`encodedMap` carries entries from **two structurally distinct insertion paths**, both writing into the same map. Distinguishing them matters when you want to thread metadata through the LMV (e.g. `productOfDisintegration` for D-32 — only meaningful on head LMVs).

**Path 1 — head-implication insert** (`memory.cpp`, inside `addToHashMemory`'s per-permutation loop). For each Bell-partition copy of the implication and each permutation of the chain, build an LMV whose `.value` is the head's normalised template (with the chain's normalisation map applied) and push into `encodedMap[intIgnoredKey]`. This is the rule registry entry — the LMV that fires when an LB hash-matches the chain.

**Path 2 — derivative LMV insert** (`memory.cpp`, inside `makeNormalizedKeysForAdmission` called from `addToHashMemory`). For each selectable non-anchor chain element, build an LMV keyed by a derivative sub-key and carrying either the marker-form missing premise (regular and `ordisOnly`) or the premise verbatim (`ordis2Demand`). When the actual rule has an `Anchor*` premise, every derivative sub-key retains it, including the regular route when the anchor mentions the selected output argument. The anchor is never selected as the missing premise. Regular, `ordisOnly`, and `ordis2Demand` qualification remains based on the same non-anchor premises; only the installed key gains the anchor context. Normalization is unchanged and runs over the complete derivative key, so equality to a particular anchor slot survives alpha-renaming ([I-193](../30_invariants.md#i-193)).

**Regular output-marker qualification.** In `logical_block/admission_policy.hpp` / `AdmissionPolicy`, both forms of `implicationIsQualified` use `baselineClassicQualifies` for the shared classic verdict. Its arithmetic route requires more than `minNumOperatorsKey` positive `in3` premises and an `in3` head. Its separate long-key exception requires all three conditions: at least `minLenLongKey` premises, no `Anchor*` premise, and a product of disintegration identified by an argument-position `u_` prefix (`hasUPrefixInKey`, the existing product classification). Length alone never qualifies an anchored theorem or an unanchored rule without that classification. In particular, the strict multiplication-monotonicity theorem's anchor plus four substantive premises does not qualify through the five-premise exception. A fourth route, operator-relation ([D-494](../40_decisions.md#d-494)), qualifies single premise slots instead of the whole rule: `implicationIsQualified` route (D) holds when some slot meets `AdmissionPolicy::operatorRelationSlotQualifies`. The rule needs more than `minNumOperatorsKey` positive `in3` premises, the missing one included, beside at least one relation premise (`isRelationPremise`: a positive compiled non-atomic predicate whose configuration has no output slot), with any head. The slot must be a positive `in3` whose arguments other than the output all occur in other key elements, anchor included, while the output occurs in no other key element and among the head's arguments. The regular loop of `makeNormalizedKeysForAdmission` installs the output-marker variant of exactly those slots; classic per-element acceptance stays tied to `baselineClassicQualifies`, and the local-u narrowing guard does not apply to them. A fifth route, compound-operator ([D-569](../40_decisions.md#d-569)), also qualifies single slots: route (E) holds when some slot meets `AdmissionPolicy::compoundOperatorSlotQualifies`. The rule has exactly `ExecutionParameters::kCompoundOperatorRoutePremises` (4) positive operator premises (a configuration with an output slot), exactly `kCompoundOperatorRouteCompounds` (2) of them compound (a compiled non-atomic entity), and a positive compound head; the slot is the rule's only compound operator premise meeting the missing-premise conditions of route (D): its other arguments occur in other key elements, its output in none and among the head's arguments. A rule with two such premises (FTA pool row 87, two fold values named only by its in3 head) does not qualify. FTA2 row 16 is the consumer: its missing `appendToSequence` premise (an operator in `ConfigFTA.json`, output g) demands the appended sequence. The regular loop installs exactly those slots' output markers, outside the local-u narrowing guard. The independent local-u criterion and tagged OR/input-slot demand routes remain separate. Qualification installs a demand template; ordinary demand firing and witness admission still apply their own scope, purity and generation/secondary gates.

**Admission purity is scope-independent.** `memory.cpp` / `checkLocalEncodedMemoryStatic` rejects ordinary marker firings when any iteration-bearing premise argument is absent from `productsOfRecursionIds`, including inside OR-integration branches. Ordinary compound demands use the same guard; integration-justified `ordis2Demand` keeps its separate D-357 exemption, and the two goal-driven kinds justified `goalDemand` (a flat root rule's near-miss demand and the marked-head implication of the goal-driven rule search) are exempt as well ([D-574](../40_decisions.md#d-574)). `gpu/phase2_cuda.cu` applies the identical verdict in both output sizing and emission. This prevents an algebraic witness from demanding further constructions through ordinary rule premises; it does not prohibit ordinary inference using already available witnesses.

**Goal-driven operator demand.** A root goal's flat integration rule installs as `IntegrationFlat` and its admission build adds near-miss demand variants (`AdmissionPolicy::goalNearMissSlotQualifies`). When one fires, the missing ground expression E goes to `IntegrationProcessor::serveGoalNearMiss`: `forEachGoalDemandVariant` binds the head of every rule concluding E's operator to E and reports each operator premise whose output the head leaves open and four positive premises without that output fix; `HashRuleInstaller::installGoalDemandVariant` installs those premises → the operator premise with `marker` in its output slot into overall, local and delta at the near-miss scope, once per identity and scope (`LbMemory::goalDemandInstalls`). Its firing is the ordinary algebra demand of the output ([D-574](../40_decisions.md#d-574), [I-335](../30_invariants.md#i-335)). The supplied Peano successor-cancellation conjecture can provide a witness-free sum conclusion after it has been proved, rather than requiring a fresh intermediate sum in enumeration interval proofs.

The two paths share the map but populate disjoint metadata. Head LMVs carry full `originalImplication` / `justification` / `validityName` for provenance; derivative LMVs leave those fields default-empty (the entry is structural — it represents the actual rule with one non-anchor premise missing). When adding new fields to `LocalMemoryValue`, ask: do I want this on head firings, on derivative probes, or both?

For D-32, `productOfDisintegration` is set only on path 1 (head implications). Marker LMVs (path 2) inherit the default `false` — they don't drive OR-disintegration; only head firings do, and only head firings are the call shape that `checkLocalEncodedMemoryStatic` evaluates.

---

## Integer encoding — the hot-path form

Every string expression has an `EncodedExpression` form ([`memory.hpp`](../../../GL_Quick_VS/GL_Quick/src/memory.hpp)):

| Field | Role |
|---|---|
| `name` | The head operator name (e.g. `"in3"`). |
| `negation` | `true` if the expression is `!(...)` at top level. |
| `arguments` | `vector<Argument>` with iteration/level metadata per arg. |
| `maxIterationNumber` | Cap on iteration index used during hash generation. |
| `original` | The original string form — for debug output + round-trip. |
| `validityName` | The scope this expression is asserted in. |

Integer encoding uses `NameMap::encode` to collapse every name into a 32-bit `NameId`. The resulting `IntNormalizedKey` is a packed array of `(nameId, argumentIds)` — cheap to hash, fast to compare.

**The int row is the only stored statement form** ([D-127](../40_decisions.md#d-127)). `encodeExpression` (`memory.hpp`) converts an `EncodedExpression` into the flat all-32-bit `IntEncodedExpr` — interning the WHOLE `original` text as `originalId` and the scope name as `validityId` — and `decodeExpression` is its exact inverse: it looks both strings up and lets `EncodedExpression`'s parsing constructor re-derive every other field, so the round trip is lossless (arity above `MAX_ARITY` asserts at encode). `Memory::intEncodedStatements` and the local / delta / external int registries are therefore the single source of truth; `EncodedExpression` values exist only transiently — built at insert sites, decoded on demand at boundaries (diagnostic dump, mail fill, visualizer, equivalence-class rewriting). Boundary reads use the non-minting `NameMap::lookup` and copy decoded references before any mint ([I-3](../30_invariants.md#i-3)).

`precompileStructuralOperators` ([I-1](../30_invariants.md#i-1)) is the transformation that ensures raw `!(&...)` / `!(>...)` *always* become compiled `or<N>` / `existence<N>` names before they hit the encoder. Otherwise the encoded key would carry the structural operator as its name, and the hash index would treat semantically-identical expressions as distinct.

---

## Admission — when does an emission actually happen

Not every hash hit fires. Between the key match and the `addStatement` call, the admission check gates which freshly-minted variables are permitted in the conclusion.

Two paths (see [prover chapter — Disintegration](../10_pipeline/04_prover.md#disintegration)):

1. **`isAdmitted`** ([`logical_block/admission_memory.hpp`](../../../GL_Quick_VS/GL_Quick/src/logical_block/admission_memory.hpp)) — consults `admissionMap`, the authoritative path.
2. **`isAllowedAsOperatorInput`** ([`logical_block/admission_policy.hpp`](../../../GL_Quick_VS/GL_Quick/src/logical_block/admission_policy.hpp)) — single-input-operator fallback; guardrail against RT explosion. See [I-6](../30_invariants.md#i-6).

First-match-wins. Rejected variables are buffered in `pendingRejections`; the outer "commit rejections only for never-admitted vars" guard decides whether they stay rejected or get admitted later.

---

## Why hash-based and not SAT / congruence closure

The architectural bet is that the cost of search-based methods (SAT, proof-term reconstruction, tactic scheduling) is avoided if the *definition* of each rule provides its hash key structurally. A SAT solver has to discover what to prove; a GL LB has to discover that it already knows what was asked.

Consequences:

- **Determinism.** The same input always produces the same set of fires in the same order. LBs are stepped across real worker threads, but the result is thread-schedule-independent: no LB observes another's mid-cycle state and every cross-LB effect is drained in content-sorted order after the join ([I-28](../30_invariants.md#i-28)).
- **No backtracking.** Once an expression is admitted, it is never unadmitted. Branches are handled by scope (OR disintegration creates scoped children), not by speculative execution.
- **Latency is memory-access.** Optimising the prover is optimising the hash index's locality. `intNormalizedKey` + `unordered_map` is the current structure; more specialised structures (tries, Bloom filters over premise classes, cache-line-sized buckets) are explored candidates for the next RT campaign.

---

## Weaknesses

### Known & tracked

- **One subkey's merged owner record must fit one arena block.** `inference/hash_rule_index.hpp` / `flushSubkeySection` stores all distinct signatures and owner/signature pairs for one normalized subkey in one `OwnerSet` blob and assembles that blob with one `LbArena::alloc` call. Its size is nine header bytes, four bytes plus eight bytes per argument constraint for each signature, and twelve bytes per owner/signature pair. The surrounding paged containers do not remove this contiguous scratch limit. The arena-backed `HashMap::peekBlobContiguous` read path also assembles a straddling blob in one allocation, so changing only the writer would leave the same size boundary on reads. An observed 319,401-byte merged record exceeds the configured 262,144-byte block; the original allocator assertion remains the failure signal.
- **Enabled implication multiplication can expand installation work.** [`inference/implication_multiplier.hpp`](../../../GL_Quick_VS/GL_Quick/src/inference/implication_multiplier.hpp), `ImplicationMultiplier::multiplyImplication`, returns only the original implication when `allow_multiplication` is false and counterexample filtering is inactive. Where enabled, partition copies of eligible variables can increase installation work. The FTA configuration disables ordinary multiplication; its workload cannot be diagnosed from an unconditional partition-count estimate.
- **Historical counterexample-filter timings are not current bottleneck evidence.** The earlier claim that counterexample filtering consumed 80% of Gauss time described an older implementation. Current costs require the actual run's phase measurements; see the runtime-measurement chapter.
- **Statement-window completeness.** Request generation retains the first `maxNumberFilteredStatements` eligible statements before sorting on both processor and CUDA routes (default 16384; the earlier fixed limit was 8192). A later proved premise can therefore remain unavailable to an installed rule, and removing earlier rows can bring it back inside the window (observed with the former finished-task wipe on the slowest forward-F3 block, [D-443](../40_decisions.md#d-443)). The singleton Fold proof supplies an observed case; see the prover chapter's *Known & tracked* section. The soundness of the `u_` literal prune does not make this truncation complete.

### Suspected fragility

- **`admissionMap` consistency across nested scopes.** The admission map is per-LB, but nested-scope rules (hypothesis, OR branch) need their own entries. A scoping bug here would silently widen or narrow admission.
- **`IntNormalizedKey` collision assumption.** The normalisation is designed so that distinct semantic expressions produce distinct keys. Not formally proven — if two structurally-distinct expressions normalise to the same key, the hit would fire both rules on the wrong premises. Historically not observed, but the guarantee is implicit, not asserted.
- **`rejectedMap` cleanup.** Entries are added when admission fails; unclear whether all scope-discharge paths clean them up fully. A rejection that persists past its scope would prevent later admission of a legitimate expression.
- **Parked-source scale remains workload-dependent.** Integration parks store source records, and derived canonical indexes select affected sources and matching demands for maintenance and replay. A whole-map scan per class is no longer the general model. See *Parked existences and normal replay* in the [prover chapter](../10_pipeline/04_prover.md) for the current representation and lifecycle.

### Not exercised by tests

- **Hash-key-collision monitoring.** No instrumentation counts `encodedMap[key].size > 1` events or tracks whether multiple rules share a key. If a batch ever had catastrophic collisions, the symptom would be wall-clock degradation with no clear signal.

---

## See also

- [`10_pipeline/04_prover.md`](../10_pipeline/04_prover.md) — prover operation overview.
- [`20_core_concepts/01_logic_blocks.md`](01_logic_blocks.md) — the LB owning the hash memory.
- [`20_core_concepts/03_mail_system.md`](03_mail_system.md) — cross-LB propagation.
- [I-1](../30_invariants.md#i-1) — precompile before key encoding.
- [I-6](../30_invariants.md#i-6) — Pass B admission gate.
-, — RT campaign history.


### Owner-removal snapshot size

[`inference/hash_rule_index.hpp`](../../../GL_Quick_VS/GL_Quick/src/inference/hash_rule_index.hpp) and `HashRuleIndex::removeOwnerFromRun` snapshot surviving owners in a paged vector, then replace the original run with `assignRunGeneratedAtId`. Record order, owner-less keys awaiting cleanup, neighboring runs and key identifiers match the contiguous writer. The snapshot may exceed one arena block; each individual owner record remains fixed-size. This is transient storage only, with no index or deload-format change.

## Shared immutable expression records

`shared_expression_list.hpp`, `SharedExpressionStore` and `SharedExpressionList`, give each logic block one exact-byte record store and four independent ordered reference lists. A 32-bit identifier addresses an immutable `IntEncodedExpr`; full identity includes scope, flags and all argument fields. Reference counts preserve list lifetimes independently, freed slots are reused, and the last release frees the entire store. Partial free slots remain allocated and count against measured savings. The derived hash table resolves collisions by full-record comparison.

The main, local, newly-local and external lists preserve their existing order, duplicates, insertion and removal behavior. Copies between blocks re-intern full records in the destination. No proof predicate uses a storage identifier. `IntStmtView` reads both shared lists and ordinary scratch vectors. Temporary generated rows remain ordinary `PagedVector` storage.

Canonical deload tags 0 through 3 still stream the original full records in list order; reload rebuilds sharing. Raw eviction retains records, lookup buckets, reference counts and list identifiers as one arena image. Arena compaction changes only physical page bindings. The physical census counts `sharedExpressions` once and each list's reference pages separately. The canonical stream contains no new sharing metadata. See I-283.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
