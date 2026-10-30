/* Generative Logic : A deterministic reasoning and knowledge generation engine.
 Copyright(C) 2025-2026 Generative Logic UG(haftungsbeschränkt)

 This program is free software : you can redistribute it and /or modify
 it under the terms of the GNU Affero General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with this program.If not, see < https://www.gnu.org/licenses/>.

 ------------------------------------------------------------------------------

 This software is also available under a commercial license.For details,
 see: https://generative-logic.com/license

 Contributions to this project must be made under the terms of the
 Contributor License Agreement(CLA).See the project's CONTRIBUTING.md file.*/
#pragma once
#include "../project_paths.hpp"
#include "../infra/hashburst_dump.hpp"
#include "../memory.hpp"
#include "../filter.hpp"
#include "../compilation_state.hpp"
#include "../compiler/expression_parser.hpp"
#include "../compiler/compilation_normalizer.hpp"
#include "../compiler/compilation_persistence.hpp"
#include "../compiler/compiled_definition_registry.hpp"
#include "../inference/hash_key_builder.hpp"
#include "../inference/implication_multiplier.hpp"
#include "../inference/binding_partitions.hpp"
#include "../inference/hash_rule_index.hpp"
#include "../inference/hash_rule_classifier.hpp"
#include "../inference/hash_request_generator.hpp"
#include "../inference/hash_rule_evaluator.hpp"
#include "../inference/burst_sink.hpp"
#include "../logical_block/logic_block_identity.hpp"
#include "../logical_block/statement_knowledge.hpp"
#include "../logical_block/deferred_ancestor_admission.hpp"
#include "../logical_block/update_global_single_chapter_record.hpp"
#include "../logical_block/or_implication_refuter.hpp"
#include "../logical_block/or_implication_selection.hpp"
#include "../logical_block/watch_bridge.hpp"
#include "../logical_block/canonical_form_cache.hpp"
#include "../logical_block/compact_canon_pair.hpp"
#include "../logical_block/numeric_less.hpp"
#include "../logical_block/hypo_scope_parse.hpp"
#include "../logical_block/implication_subproof_outcome.hpp"
#include "../logical_block/collected_record.hpp"
#include "../logical_block/collected_arena.hpp"
#include "../logical_block/new_variable_store.hpp"
#include "../logical_block/witness_meta_store.hpp"
#include "../logical_block/rejection_record.hpp"
#include "../logical_block/rejection_store.hpp"
#include "../logical_block/scanned_token_runs.hpp"
#include "../logical_block/equivalence_span_pair.hpp"
#include "../logical_block/equivalence_rewrite.hpp"
#include "../logical_block/admission_policy.hpp"
#include "../logical_block/scope_type_resolver.hpp"
#include "../logical_block/canonicalizer.hpp"
#include "../logical_block/canonical_template_matcher.hpp"
#include "../logical_block/rejected_source_memory.hpp"
#include "../logical_block/integration_instruction_builder.hpp"
#include "../logical_block/unary_rule_memory.hpp"
#include "../logical_block/admission_marker_builder.hpp"
#include "../compiler/existence_rule_compiler.hpp"
#include "../compiler/implication_builder.hpp"
#include "../compiler/or_rule_compiler.hpp"
#include "../compiler/structural_compiler.hpp"
#include "../compiler/supporting_rule_compiler.hpp"
#include "../compiler/compilation_reuse.hpp"
#include "../compiler/definition_expansion.hpp"
#include "../proof/worker_context.hpp"
#include "../infra/rt_tracker.hpp"
#include "../infra/hashburst_dump.hpp"
#include "../memory_infra/steward.hpp"
#include "../memory_infra/lb_store.hpp"
#include "../mail_log.hpp"
#include "../simple_map_store.hpp"
#include "../memory_infra/or_implication_match_index.hpp"
#include "../memory_infra/temporary_witness_store.hpp"
#include "../logical_block/phase13_timing_slot.hpp"
#include "../logical_block/watch_ledger.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <unordered_set>
#include <utility>
#include <vector>
namespace gl {
class AdmissionMemory;
namespace tests { struct HashRuleInstallerFixture; }
/// @brief Hash Rule Installer operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class HashRuleInstaller {
friend struct tests::HashRuleInstallerFixture;
public:
/// @brief Bind inputs for hash rule installer.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param admissionMarkers Borrowed AdmissionMarkerBuilder interface or state; must outlive this component.
/// @param allPermutationsAna Borrowed std::map interface or state; must outlive this component.
/// @param hashRuleClassifier Borrowed HashRuleClassifier interface or state; must outlive this component.
/// @param implicationMultiplier Borrowed ImplicationMultiplier interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param unaryMemory Borrowed UnaryRuleMemory interface or state; must outlive this component.
/// @param admissionMemory Borrowed AdmissionMemory interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
HashRuleInstaller(AdmissionMarkerBuilder& admissionMarkers, const std::map<int, std::vector<std::vector<int> > >& allPermutationsAna, HashRuleClassifier& hashRuleClassifier, ImplicationMultiplier& implicationMultiplier, const ProverParameters& parameters, UnaryRuleMemory& unaryMemory, AdmissionMemory& admissionMemory);

public:
// ==== Rule-index staging (D-333) ==============
        //
        // The install doors above (addOwnerToRun, appendLmvIdsRecord,
        // ensureEmptySubkeyRecord / mergeSubkeySignatures,
        // insertRemainingArgsNormKeyBatch) write one record into a dense CSR
        // pool per call — O(pool tail) each. The install path calls the
        // staging twins below instead; the flush rewrites each touched map ONCE
        // (HashMap::assignAllRunsGenerated) with every key's run merged exactly
        // as the sequential doors would have merged it. The removal path keeps
        // the direct doors and flushes first.























        /// @brief Split an expanded implication text into the spans
        ///        `addToHashMemory` takes — the premise chain, the head and the
        ///        sorted-unique remaining-arg run — the one decomposition the
        ///        install door and the removal share.
        ///
        /// @details
        /// `extractKeyValueKeyScratch` copies the key half onto @p kvArena (the
        /// remaining-arg slices point into that copy, so the caller frames
        /// @p kvArena with a `ScratchScope` that outlives every use of
        /// @p remKeyRun); `extractRemainingArgs` yields the args, inserted
        /// sorted-unique; `disintegrateImplicationSpans` slices the chain
        /// elements and the head straight out of @p impStr (caller-stable).
        /// Zero heap.
        ///
        /// @param impStr     The expanded implication text (stable for the call).
        /// @param kvArena    The string-tier scratch arena for the key copy.
        /// @param chainRun   Out: the premise spans (`MAX_ADMISSION_KEY_ELEMENTS` cap).
        /// @param chainRunN  Out: their number.
        /// @param headSpan   Out: the head span.
        /// @param remKeyRun  Out: the sorted-unique remaining-arg spans
        ///                   (`MAX_ADMISSION_REM_ARGS` cap).
        /// @param remKeyRunN Out: their number.
        /// @see `addToHashMemory`, `removeRuleFromHashMemory`,
        ///      `addExprToMemoryBlock` (the install door).
void decomposeImplicationForInstall(StrSpan impStr, ScratchArena& kvArena,
                                            StrSpan* chainRun, int32_t& chainRunN,
                                            StrSpan& headSpan,
                                            StrSpan* remKeyRun, int32_t& remKeyRunN);

public:
/// @brief Remove one installed rule's ownership from every index of
        ///        one `HashMemory` instance and stage its LMVs for the batched
        ///        drop — the removal entry point.
        ///
        /// @details
        /// Decomposes the expanded implication exactly as the install door did
        /// and re-runs `addToHashMemory` under `RuleIndexOp::Remove`: every
        /// whole key, subkey, remaining-args edge and chain the install
        /// visited loses this rule's owner (first visit only, through the
        /// caller-owned @p visited set, reset here). The enumeration first
        /// delists the rule from each copy's owner run (`copyOwners`): a copy
        /// another rule still owns keeps its entries and LMVs; a copy left
        /// rule-less loses this owner everywhere and its
        /// `(originalImplicationId, validityId)` pair is minted into
        /// @p dropSet — the install's own multiplication decides the copies,
        /// so several texts stage several pairs and a multiplication that
        /// emitted no copy (the trivial-head skip) stages nothing, exactly as
        /// it installed nothing; the caller finishes the apply with
        /// `finishRuleRemovals`, which drops the LMVs of every staged rule in
        /// one compaction and erases the owner-less entries. Both scratch
        /// sets MUST live on an arena other than the enumeration's per-slot
        /// gen-scratch arena (the enumeration frames that arena with
        /// mark/pop and would free their pages). Single-threaded seam only.
        ///
        /// The replay passes the install's own route arguments, because the
        /// admission-key passes depend on them (the ordis2 demand variants
        /// need fewer premises for an implication than for an integration
        /// rule): @p installedAs `implication` replays the install door
        /// (`addExprToMemoryBlock`: the configured admission depth,
        /// secondary number and minimum operator count), `integration`
        /// replays the integration installs
        /// (`CompactRuleProcessor::installIntegrationCompact`: all zero).
        ///
        /// @param mb          The LB.
        /// @param hm          The instance the rule was installed into.
        /// @param impStr      The expanded implication text (the LMV's original).
        /// @param impValidity The install scope name.
        /// @param visited     The per-removal visited-entry set (reset here).
        /// @param dropSet     The apply's staged LMV drop set (packed owners).
        /// @param onlyIfPresent Probe first: a transient instance may have
        ///                    been reset since the install; absent means
        ///                    nothing to remove.
        /// @param installedAs The install route: `implication` (the door) or
        ///                    `integration`; asserted to be one of the two.
        /// @param goalNearMiss The rule was installed as a root goal's flat
        ///                    integration rule: the replay also removes its
        ///                    near-miss demand variants
        ///                    (D-574); only
        ///                    with @p installedAs `integration` (asserted).
        /// @return `true` when the removal ran; `false` when @p onlyIfPresent
        ///         found the rule absent from the instance.
        /// @see `addToHashMemory`, `RuleIndexOp`, `finishRuleRemovals`.
bool removeRuleFromHashMemory(Memory& mb, HashMemory& hm,
                                      StrSpan impStr, StrSpan impValidity,
                                      ColdHashSet<BytesKeyStore>& visited,
                                      ColdHashSet<PodKeyStore<int64_t>>& dropSet,
                                      bool onlyIfPresent = false,
                                      RuleJustification installedAs = RuleJustification::implication,
                                      bool goalNearMiss = false);

public:
/// @brief Finish an apply's rule removals on one instance: drop the
        ///        LMVs of every staged rule in ONE `encodedMap` compaction and
        ///        erase the owner-less index entries.
        ///
        /// @param hm      The instance.
        /// @param dropSet The staged `(originalImplicationId, validityId)` owners.
        /// @param gArena  Per-slot gen-scratch arena.
        /// @see `removeRuleFromHashMemory`, `eradicateEncodedMapWhere`,
        ///      `eraseOwnerlessEntries`.
static void finishRuleRemovals(HashMemory& hm,
                                       const ColdHashSet<PodKeyStore<int64_t>>& dropSet,
                                       ScratchArena& gArena);

private:
/// @brief Install a short subkey with one fixed empty record.
        ///
        /// @details
        /// The nine-byte empty Codec<OwnerSet> frame stores a zero loose byte
        /// and zero signature and owner counts. Repeated installs preserve it.
        /// Short keys survive rule removal until their hash memory is reset
        /// (I-228).
        ///
        /// @param map The subkey record map.
        /// @param numberExpressions The premise count, one or two.
        /// @param data The normalized key's NameId payload.
        /// @param len The payload length.
        /// @return Nothing.
        /// @invariant Presence alone accepts a short subkey; no owner is stored.
        /// @see stageEmptySubkeyRecord, subkeyUSatisfied.
static void ensureEmptySubkeyRecord(TypedColdBlobMap<NormKey, OwnerSet>& map,
                                            int32_t numberExpressions,
                                            const NameId* data, int32_t len);

private:
/// @brief Record a whole key with one owner — the install door of
        ///        `normalizedEncodedKeys`.
        ///
        /// @details
        /// The key bytes are built on the stack via `encodeNormKeyInto`
        /// (byte-exact to `Codec<NormKey>::encode`, the bytes `wholeKeyPresent`
        /// probes) and the owner is added to the key's owner run through
        /// `addOwnerToRun` (a new key gets the one-owner run, a present key
        /// gains the owner, a repeated owner is a no-op). The key is the
        /// request generator's emission gate; it leaves the map only when its
        /// last owner is removed ([I-49](../30_invariants.md#i-49)) or the
        /// map is reset — never by a scope wipe. Single-threaded install site
        /// only ([I-83](../30_invariants.md#i-83)).
        ///
        /// @param map               The whole-key owner map (`normalizedEncodedKeys`).
        /// @param numberExpressions The key's leading count field.
        /// @param data              The key's NameId payload.
        /// @param len               The payload length.
        /// @param owner             The installing rule's packed owner.
        /// @param gArena            Per-slot gen-scratch arena for the run merge.
        /// @see `wholeKeyPresent` — the probe; `ensureEmptySubkeyRecord` — the
        ///      subkey map's short-key door; `addOwnerToRun`.
static void addWholeKeyOwner(TypedColdBlobMap<NormKey, RuleOwnerRec>& map,
                                     int32_t numberExpressions,
                                     const NameId* data, int32_t len,
                                     RuleOwner owner, ScratchArena& gArena);

private:
/// @brief Merge one owner's u_ signature into a subkey record — the
        ///        blob-native install door of `normalizedEncodedSubkeys` for
        ///        subkeys of at least `kSubkeyUCheckMinElements` premises.
        ///
        /// @details
        /// Fully blob-native — NEITHER an `OwnerSet` NOR a `serialize` blob is
        /// materialized. The key bytes are built on the stack via
        /// `encodeNormKeyInto` (byte-exact to `Codec<NormKey>::encode`); the run
        /// is probed / written through the raw `inner().lookup` / `inner().assignRun`
        /// doors. The existing record is READ zero-copy through `OwnerSetBlob`
        /// (`peekRecordBytes` arena straddle overload), the owner's u_ signature is
        /// produced by `buildUSignatureRunInto`, and the MERGED blob is assembled
        /// directly on the gen byte-bump tier (reached internally via
        /// `gl::WorkerContext::coreId()`, per-registry fallback): byte 0
        /// `existingLoose || !hasUArg`; `uSignatures` the existing lex-ascending
        /// sigs with the new one inserted at its
        /// `std::vector<std::pair>::operator<` position (idempotent — skipped if
        /// byte-equal). The result is byte-identical to `Codec<OwnerSet>::serialize`
        /// of the decode-merge-reencode oracle (`recordUSignature` +
        /// `Codec<OwnerSet>`, retained as the differential twin). The blob is
        /// assembled in a DISJOINT buffer BEFORE the run-replacing `assignRun`, so
        /// the peeked existing bytes never dangle (09b pitfall 6). A subkey record
        /// is exactly what its growth probe reads (`subkeyUSatisfied`, D-120); the
        /// subkey map has no owner accounting and is never scope-wiped
        /// ([I-49](../30_invariants.md#i-49)). Below the threshold the install
        /// sites call `ensureEmptySubkeyRecord` instead. Single-threaded install
        /// site only ([I-83](../30_invariants.md#i-83)).
        ///
        /// @param map              The subkey map (`normalizedEncodedSubkeys`).
        /// @param numberExpressions The key's leading count field.
        /// @param data             The key's NameId payload.
        /// @param len              The payload length.
        /// @param encList          The subkey's pre-encoded premises.
        /// @param count            Their number (`>= kSubkeyUCheckMinElements`).
        /// @param owner            The installing rule's packed owner — appended
        ///                         to the record's owner list with the index of
        ///                         its signature (-1 when loose); a repeated
        ///                         owner is a no-op.
        /// @see `ensureEmptySubkeyRecord`, `subkeyUSatisfied`,
        ///      `buildUSignatureRunInto`, `OwnerSetBlob`, `encodeNormKeyInto`,
        ///      `recordSubkeyOwner` (the heap oracle), `Codec<OwnerSet>` (the
        ///      blob oracle).
static void mergeSubkeySignatures(TypedColdBlobMap<NormKey, OwnerSet>& map,
                                          int32_t numberExpressions, const NameId* data,
                                          int32_t len,
                                          const IntEncodedExpr* encList, NameId count,
                                          RuleOwner owner);

public:
/// @brief Install one or more `LocalMemoryValue` records into a `HashMemory`,
    /// keyed on the canonical normalization of `key`.
    ///
    /// @details
    /// Path 1 of the hash-engine rule installation: head-implication LMV inserts.
    /// Each call corresponds to one head-implication that the prover wants to
    /// register with the hash engine for later firing.
    ///
    /// The body unfolds in three logical phases:
    ///
    /// 1. **multiplyImplication.** The producer implication is expanded via
    ///    `multiplyImplication(originalImplication)` into one or more copies that
    ///    differ in how their bound variables and remaining-args are partitioned.
    ///    Copy 0 reuses the caller-supplied `key`/`value`; copies 1..N synthesize
    ///    their own from `multiplyImplication`'s output. Provenance for the
    ///    multiplied copies is recorded in `mb.exprOriginMap` (and the mail-out
    ///    mirror) under the `"multiplied from"` tag, capped per `max_origin_per_expr`
    ///    (`compressor_max_origins_per_expr` in compressor mode).
    /// 2. **Per-copy normalization + dedupe.** For each copy, build the
    ///    `IntNormalizedKey` from the NameId-encoded key bytes; consult
    ///    `normalizedEncodedKeys` and `normalizedEncodedSubkeys*` for fast
    ///    rejection; if novel, allocate on the gen scratch arena byte-bump tier so the key
    ///    pointer stays stable for the rest of the LB's lifetime.
    /// 3. **LMV emit + admission.** Construct a `LocalMemoryValue` carrying
    ///    the value, levels, justification, originalImplication, key,
    ///    remainingArgs, validityName. Set `productOfDisintegration` per
    ///    [D-32](../../docs/agentic_swdd/40_decisions.md#d-32) when the producing
    ///    implication has at least one `u_`-prefixed premise arg (this gates
    ///    later OR-disintegration on the head-firing path). Push into
    ///    `targetIntMemory.encodedMap[normKey]`. When
    ///    `performAdmissionMapUpdate == true`, also call into the admission
    ///    side via `updateAdmissionMap3` so the `(expression, scope)` pair
    ///    is registered in `targetIntMemory.admissionMap`.
    ///
    /// @param key                     Pre-built normalization key (string form;
    ///                                NameId mirror is computed inside).
    /// @param value                   Head expression text (what the rule produces).
    /// @param remainingArgs           Argument names not consumed by the key
    ///                                that must travel with the head.
    /// @param mb                      Owning `Memory` (used for `exprOriginMap`,
    ///                                `mailOut`, `ruleInterner`).
    /// @param secondTarget            Optional second instance receiving every
    ///                                staged record of the same build (the
    ///                                incubator's `workingMemory` beside
    ///                                `overallHashMemory`); install-only,
    ///                                never the first target.
    /// @param targetIntMemory         The `HashMemory` to install into. Usually
    ///                                `mb.localHashMemory` for fresh installs;
    ///                                deltas go to `mb.localHashMemoryDelta`.
    /// @param levels                  Ascending-unique level run — LB levels at
    ///                                which the rule is admissible.
    /// @param levelCount              The level count (`>= 0`).
    /// @param originalImplication     Full producer implication text, kept for
    ///                                provenance and for `multiplyImplication`.
    /// @param maxAdmissionDepth       Standard depth budget passed to the
    ///                                admission map.
    /// @param maxSecondaryNumber      Standard secondary-iteration budget.
    /// @param admissionFlags          Flag byte of the admission entries the
    ///                                install files (0, 1 or 3):
    ///                                `kAdmissionRecursionFlag` for a
    ///                                recursion-owned producer, with
    ///                                `kAdmissionInductionHypothesisFlag` for
    ///                                the step block's induction hypothesis
    ///                                (D-519).
    /// @param minNumOperatorsKey      Minimum-operator threshold used by some
    ///                                downstream filters; carried through.
    /// @param justification           Proof tag string emitted on hit
    ///                                (e.g. `"hash"`, `"head"`, `"equality1"`).
    /// @param performAdmissionMapUpdate If true, call `updateAdmissionMap3`
    ///                                in lockstep with the LMV install.
    /// @param originalImplicationClean De-renamed form used as the antecedent
    ///                                in `"multiplied from"` origin records.
    /// @param validityName            Scope at which the rule fires. `"main"`
    ///                                or a `_boundary_`-bearing form minted via
    ///                                `NameMap::encodePush` per
    ///                                [I-2](../../docs/agentic_swdd/30_invariants.md#i-2).
    /// @param goalNearMiss            The rule is a root goal's flat
    ///                                integration rule: its admission build
    ///                                adds the near-miss demand variants
    ///                                (D-574);
    ///                                a removal replays the same value.
    /// @param weakHypothesisInstall   The rule is a weak step block's induction
    ///                                hypothesis: its admission build may file
    ///                                the two-marker start of a chain that
    ///                                cannot start eating
    ///                                (`AdmissionMemory::updateAdmissionMap`,
    ///                                D-586).
    ///
    /// @pre  `mb.lbMemory.manager` is resident — the cold `encodedMap` and the
    ///       two `normalizedEncoded*` owner maps install onto it. Keys are owning
    ///       (`NormKey`); the former keyArena-backed `IntNormalizedKey`
    ///       persistence is retired.
    /// @post `targetIntMemory.encodedMap` and the related `normalizedEncoded*`
    ///       sets are extended with the new LMV(s). Origin entries are added
    ///       under the `"multiplied from"` tag for each non-zero copy.
    /// @invariant [I-2](../../docs/agentic_swdd/30_invariants.md#i-2) — non-`"main"`
    ///            validityName must be minted via `NameMap::encodePush`.
    /// @invariant [I-3](../../docs/agentic_swdd/30_invariants.md#i-3) — when this function
    ///            calls `nm.decode()` / reads `idToSub[]`, the result is
    ///            copied before any nested mint (re-entrance through
    ///            `multiplyImplication` may grow the underlying vectors).
    /// @invariant [I-25](../../docs/agentic_swdd/30_invariants.md#i-25) — cross-scope
    ///            deposits keep `validityName` end-to-end via
    ///            `ExpressionWithValidity` pairs.
    /// @see `inference/hash_rule_installer.hpp::checkLocalEncodedMemoryStatic` — consumer of the
    ///      LMVs installed here.
    /// @see `inference/hash_rule_installer.hpp::updateAdmissionMap3` — admission-side counterpart.
    /// @see `multiplyImplication` (`inference/hash_rule_installer.cpp`) — generator of the per-copy
    ///      key/value pairs.
void addToHashMemory(const StrSpan* keyRun, int32_t keyN,
            StrSpan valueSpan,
            const StrSpan* remRun, int32_t remN,
            Memory& mb,
            HashMemory& targetIntMemory,
            const int* levels, int32_t levelCount,
            StrSpan originalImplicationSpan,
            int maxAdmissionDepth,
            int maxSecondaryNumber,
            uint8_t admissionFlags,
            int minNumOperatorsKey,
            StrSpan justificationSpan,
            bool performAdmissionMapUpdate,
            StrSpan originalImplicationCleanSpan,
            StrSpan validityNameSpan = StrSpan("main", 4),
            const RuleIndexOp& op = RuleIndexOp{},
            HashMemory* secondTarget = nullptr,
            bool goalNearMiss = false,
            bool weakHypothesisInstall = false);

/// @brief Install a goal-driven marked-head implication: its present
///        premises imply the operator premise with `marker` in the output
///        slot (D-574).
///
/// @details
/// The goal-driven rule search (`IntegrationProcessor::forEachGoalDemandVariant`)
/// hands one qualifying operator premise of a rule whose head it bound to a
/// goal's missing expression: @p key holds the rule's premises that do not
/// name the operator premise's output, then the operator premise last. The
/// implication is installed as one admission-marker variant family
/// (`AdmissionMarkerBuilder::installAdmissionMarkerVariants`: every element
/// but the last is the key, @p markedArg the marked argument) into the
/// three instances of the integration route — overall, local and delta, so
/// the new rule meets every visible statement once — with justification
/// `goalDemand`, which the marker branch of the firing site exempts from
/// the purity gate. A firing instantiates the head into the ordinary algebra
/// demand of the output (`drainAdmissionKeysAlgebra`), which wakes an
/// existence parked under the same key.
///
/// The install is a staging window of its own; inside an enclosing window
/// that window's closer flushes it.
///
/// @param mb        The owning LB (single-threaded seam).
/// @param key       The present premises, then the operator premise
///                  (`u_` form); spans stable across the call.
/// @param keyN      Their count; at least two, at most
///                  `MAX_ADMISSION_KEY_ELEMENTS + 1`.
/// @param markedArg The operator premise's output argument: once in the
///                  operator premise and in no present premise (asserted).
/// @param identity  The implication's identity text; with @p validity the
///                  owner of every index entry and LMV of the install.
/// @param validity  The install scope, an existing scope name.
/// @return Nothing.
/// @invariant Writes no originals chain; mints only through the marker
///            door; 0% heap.
/// @see AdmissionMarkerBuilder::installAdmissionMarkerVariants,
///      IntegrationProcessor::serveGoalNearMiss.
void installGoalDemandVariant(Memory& mb, const StrSpan* key, int32_t keyN,
            StrSpan markedArg, StrSpan identity, StrSpan validity);

private:
/// @brief Collect the `u_`-prefixed "remaining" argument names of an
///        expression, with the `u_` prefix stripped, as zero-copy slices.
///
/// @details Scans every `[...]` argument block NOT immediately preceded by `>`
///          (a `>[...]` block is the forbidden zone for `u_` and is only
///          asserted clean). Each comma-separated, trimmed token beginning with
///          `u_` yields one output slice `token[2:]` — a `StrSpan` into the
///          caller's `str` bytes, so the slices live exactly as long as `str`.
///          Duplicates are emitted in scan order; the caller dedups (typically
///          by inserting into a `std::set<std::string>`), which is why this
///          routine owns no container and is 0% heap. The lexical scan is the
///          byte-exact twin of the former `std::sregex_iterator` over
///          `\[([^\]]*)\]` (see `extract_remaining_args_matches_regex_oracle`),
///          and the forbidden-zone / additional-`u_` invariants are asserted
///          exactly as before.
///
/// @param str Expression to scan (borrowed; slices point into it).
/// @param out Caller-provided array receiving the `u_`-stripped slices.
/// @param cap Capacity of `out`; overflow asserts (Rule 19 tripwire).
/// @return Number of slices written to `out[0..return)` (duplicates possible).
///
/// @invariant No token inside a `>[...]` block starts with `u_`; a `u_` token
///            contains no further `u_` substring — both asserted.
/// @see The caller `addExprToMemoryBlock` builds the deduped `std::set` for
///      `addToHashMemory`.
int32_t extractRemainingArgs(StrSpan str, StrSpan* out, int32_t cap);
private:

AdmissionMarkerBuilder& admissionMarkers;
const std::map<int, std::vector<std::vector<int> > >& allPermutationsAna;
HashRuleClassifier& hashRuleClassifier;
ImplicationMultiplier& implicationMultiplier;
const ProverParameters& parameters;
UnaryRuleMemory& unaryMemory;
AdmissionMemory& admissionMemory;
};
}
#include "../logical_block/admission_memory.hpp"
namespace gl {

// ==== Rule-index staging (D-333) ==============
        //
        // The install doors above (addOwnerToRun, appendLmvIdsRecord,
        // ensureEmptySubkeyRecord / mergeSubkeySignatures,
        // insertRemainingArgsNormKeyBatch) write one record into a dense CSR
        // pool per call — O(pool tail) each. The install path calls the
        // staging twins below instead; the flush rewrites each touched map ONCE
        // (HashMap::assignAllRunsGenerated) with every key's run merged exactly
        // as the sequential doors would have merged it. The removal path keeps
        // the direct doors and flushes first.























        /// @brief Split an expanded implication text into the spans
        ///        `addToHashMemory` takes — the premise chain, the head and the
        ///        sorted-unique remaining-arg run — the one decomposition the
        ///        install door and the removal share.
        ///
        /// @details
        /// `extractKeyValueKeyScratch` copies the key half onto @p kvArena (the
        /// remaining-arg slices point into that copy, so the caller frames
        /// @p kvArena with a `ScratchScope` that outlives every use of
        /// @p remKeyRun); `extractRemainingArgs` yields the args, inserted
        /// sorted-unique; `disintegrateImplicationSpans` slices the chain
        /// elements and the head straight out of @p impStr (caller-stable).
        /// Zero heap.
        ///
        /// @param impStr     The expanded implication text (stable for the call).
        /// @param kvArena    The string-tier scratch arena for the key copy.
        /// @param chainRun   Out: the premise spans (`MAX_ADMISSION_KEY_ELEMENTS` cap).
        /// @param chainRunN  Out: their number.
        /// @param headSpan   Out: the head span.
        /// @param remKeyRun  Out: the sorted-unique remaining-arg spans
        ///                   (`MAX_ADMISSION_REM_ARGS` cap).
        /// @param remKeyRunN Out: their number.
        /// @see `addToHashMemory`, `removeRuleFromHashMemory`,
        ///      `addExprToMemoryBlock` (the install door).
inline void HashRuleInstaller::decomposeImplicationForInstall(StrSpan impStr, ScratchArena& kvArena,
                                            StrSpan* chainRun, int32_t& chainRunN,
                                            StrSpan& headSpan,
                                            StrSpan* remKeyRun, int32_t& remKeyRunN) {
            const ScratchString kyS = ce::extractKeyValueKeyScratch(impStr, kvArena);
            StrSpan remainingArgsSpans[ExecutionParameters::MAX_KEY_SLOTS];
            const int32_t remainingArgsN = extractRemainingArgs(
                StrSpan(kyS), remainingArgsSpans, ExecutionParameters::MAX_KEY_SLOTS);
            remKeyRunN = 0;
            for (int32_t i = 0; i < remainingArgsN; ++i) {
                const StrSpan v = remainingArgsSpans[i];
                int32_t lo = 0, hi = remKeyRunN; bool dup = false;
                while (lo < hi) {
                    const int32_t mid = (lo + hi) / 2;
                    const int c = compareSpans(remKeyRun[mid], v);
                    if (c == 0) { dup = true; break; }
                    if (c < 0) lo = mid + 1; else hi = mid;
                }
                if (dup) continue;
                assert(remKeyRunN < ExecutionParameters::MAX_ADMISSION_REM_ARGS
                    && "decomposeImplicationForInstall: rem run exceeds cap");
                for (int32_t z = remKeyRunN; z > lo; --z) remKeyRun[z] = remKeyRun[z - 1];
                remKeyRun[lo] = v; ++remKeyRunN;
            }
            chainRunN = 0;
            ce::disintegrateImplicationSpans(impStr, headSpan,
                [chainRun, &chainRunN](StrSpan keySpan, const StrSpan*, int32_t) {
                    assert(chainRunN < ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                        && "decomposeImplicationForInstall: chain run exceeds cap");
                    chainRun[chainRunN++] = keySpan;
                });
        }

/// @brief Remove one installed rule's ownership from every index of
        ///        one `HashMemory` instance and stage its LMVs for the batched
        ///        drop — the removal entry point.
        ///
        /// @details
        /// Decomposes the expanded implication exactly as the install door did
        /// and re-runs `addToHashMemory` under `RuleIndexOp::Remove`: every
        /// whole key, subkey, remaining-args edge and chain the install
        /// visited loses this rule's owner (first visit only, through the
        /// caller-owned @p visited set, reset here). The enumeration first
        /// delists the rule from each copy's owner run (`copyOwners`): a copy
        /// another rule still owns keeps its entries and LMVs; a copy left
        /// rule-less loses this owner everywhere and its
        /// `(originalImplicationId, validityId)` pair is minted into
        /// @p dropSet — the install's own multiplication decides the copies,
        /// so several texts stage several pairs and a multiplication that
        /// emitted no copy (the trivial-head skip) stages nothing, exactly as
        /// it installed nothing; the caller finishes the apply with
        /// `finishRuleRemovals`, which drops the LMVs of every staged rule in
        /// one compaction and erases the owner-less entries. Both scratch
        /// sets MUST live on an arena other than the enumeration's per-slot
        /// gen-scratch arena (the enumeration frames that arena with
        /// mark/pop and would free their pages). Single-threaded seam only.
        ///
        /// The replay passes the install's own route arguments, because the
        /// admission-key passes depend on them (the ordis2 demand variants
        /// need fewer premises for an implication than for an integration
        /// rule): @p installedAs `implication` replays the install door
        /// (`addExprToMemoryBlock`: the configured admission depth,
        /// secondary number and minimum operator count), `integration`
        /// replays the integration installs
        /// (`CompactRuleProcessor::installIntegrationCompact`: all zero).
        ///
        /// @param mb          The LB.
        /// @param hm          The instance the rule was installed into.
        /// @param impStr      The expanded implication text (the LMV's original).
        /// @param impValidity The install scope name.
        /// @param visited     The per-removal visited-entry set (reset here).
        /// @param dropSet     The apply's staged LMV drop set (packed owners).
        /// @param onlyIfPresent Probe first: a transient instance may have
        ///                    been reset since the install; absent means
        ///                    nothing to remove.
        /// @param installedAs The install route: `implication` (the door) or
        ///                    `integration`; asserted to be one of the two.
        /// @param goalNearMiss The rule was installed as a root goal's flat
        ///                    integration rule: the replay also removes its
        ///                    near-miss demand variants
        ///                    (D-574); only
        ///                    with @p installedAs `integration` (asserted).
        /// @return `true` when the removal ran; `false` when @p onlyIfPresent
        ///         found the rule absent from the instance.
        /// @see `addToHashMemory`, `RuleIndexOp`, `finishRuleRemovals`.
inline bool HashRuleInstaller::removeRuleFromHashMemory(Memory& mb, HashMemory& hm,
                                      StrSpan impStr, StrSpan impValidity,
                                      ColdHashSet<BytesKeyStore>& visited,
                                      ColdHashSet<PodKeyStore<int64_t>>& dropSet,
                                      bool onlyIfPresent,
                                      RuleJustification installedAs,
                                      bool goalNearMiss) {
            assert((installedAs == RuleJustification::implication
                    || installedAs == RuleJustification::integration)
                && "removeRuleFromHashMemory: a rule is installed by the door or by integration");
            assert((!goalNearMiss || installedAs == RuleJustification::integration)
                && "removeRuleFromHashMemory: only an integration rule carries near-miss variants");
            // A removal reads and rewrites the index runs directly: every
            // staged install of this instance materializes first.
            gl::HashRuleIndex::flushRuleIndexStagingFor(hm);
            const unsigned kvSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : scratchArenas().slotCount() - 1;
            ScratchArena& kvArena = scratchArenas().forSlot(kvSlot);
            ScratchScope kvScope(kvArena);
            StrSpan chainRun[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
            int32_t chainRunN = 0;
            StrSpan headSpan;
            StrSpan remKeyRun[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
            int32_t remKeyRunN = 0;
            decomposeImplicationForInstall(impStr, kvArena, chainRun, chainRunN,
                                           headSpan, remKeyRun, remKeyRunN);
            if (onlyIfPresent) {
                // A transient instance (localHashMemoryDelta / workingMemory)
                // is emptied per burst: the rule is there only if installed
                // since the last reset — then it is listed on its FIRST
                // multiplication copy in the instance's copyOwners. The
                // install's own multiplication decides the copies (a rewritten
                // text, or no copy at all: nothing installed, nothing to
                // remove); the rule's own listing, not the copy's presence,
                // decides — another rule may own the same copy here.
                const int32_t ruleId = mb.ruleInterner.lookup(impStr);
                const NameId vid = mb.nameMap.lookup(impValidity);
                if (ruleId == 0 || vid == 0) return false;
                const unsigned cSlot = (gl::WorkerContext::coreId() >= 0)
                    ? static_cast<unsigned>(gl::WorkerContext::coreId())
                    : genScratchArenas().slotCount() - 1;
                ScratchArena& cArena = genScratchArenas().forSlot(cSlot);
                const ArenaOffset cMark = cArena.cursor();
                int32_t copyId = 0;
                {
                    DirtyState cDirty = DirtyState::Clean;
                    PagedVector<CopyRef> copies(&cArena, &cDirty);
                    implicationMultiplier.multiplyImplication(impStr, kvArena, copies);
                    if (copies.size() > 0) {
                        const StrSpan first(reinterpret_cast<const char*>(
                            kvArena.resolve(copies[0].off)), copies[0].len);
                        copyId = mb.ruleInterner.lookup(first);
                    }
                }
                cArena.popTo(cMark);
                if (copyId == 0) return false;
                const RuleOwner copyOwner = packRuleOwner(copyId, vid);
                char copyKey[sizeof(RuleOwner)];
                std::memcpy(copyKey, &copyOwner, sizeof(RuleOwner));
                const int32_t cid = hm.copyOwners.inner().lookup(
                    StrSpan(copyKey, static_cast<int32_t>(sizeof(RuleOwner))));
                if (cid == 0) return false;
                const RuleOwner ruleOwner = packRuleOwner(ruleId, vid);
                bool present = false;
                for (int32_t j = 0; j < hm.copyOwners.runLen(cid) && !present; ++j) {
                    int32_t bl = 0;
                    const char* bp = hm.copyOwners.peekRecordBytes(cid, j, bl, cArena);
                    RuleOwner o;
                    std::memcpy(&o, bp, sizeof(RuleOwner));
                    present = (o == ruleOwner);
                }
                cArena.popTo(cMark);
                if (!present) return false;
            }
            visited.resetToFresh();
            RuleIndexOp op;
            op.kind = RuleIndexOp::Kind::Remove;
            op.visited = &visited;
            op.dropSet = &dropSet;
            const bool integration = installedAs == RuleJustification::integration;
            this->addToHashMemory(chainRun, chainRunN, headSpan, remKeyRun, remKeyRunN,
                mb, hm, nullptr, 0, impStr,
                integration ? 0 : parameters.maxIterationNumberVariable,
                integration ? 0 : parameters.standardMaxSecondaryNumber, false,
                integration ? 0 : parameters.minNumOperatorsKey,
                integration ? StrSpan("integration", 11) : StrSpan("implication", 11), false,
                impStr, impValidity, op, nullptr, goalNearMiss);
            return true;
        }

/// @brief Finish an apply's rule removals on one instance: drop the
        ///        LMVs of every staged rule in ONE `encodedMap` compaction and
        ///        erase the owner-less index entries.
        ///
        /// @param hm      The instance.
        /// @param dropSet The staged `(originalImplicationId, validityId)` owners.
        /// @param gArena  Per-slot gen-scratch arena.
        /// @see `removeRuleFromHashMemory`, `eradicateEncodedMapWhere`,
        ///      `eraseOwnerlessEntries`.
inline void HashRuleInstaller::finishRuleRemovals(HashMemory& hm,
                                       const ColdHashSet<PodKeyStore<int64_t>>& dropSet,
                                       ScratchArena& gArena) {
            // The compaction reads and rewrites the runs: every staged install
            // of this instance (the apply's canonical expansions) materializes
            // first, so a key an install re-owned survives the sweep exactly as
            // it does on the sequential path.
            gl::HashRuleIndex::flushRuleIndexStagingFor(hm);
            if (dropSet.count() > 0) {
                eradicateEncodedMapWhere(hm.encodedMap, [&](const LmvBlobView& view) {
                    return dropSet.lookup(packRuleOwner(view.originalImplicationId(),
                                                        view.validityId())) != 0;
                }, gArena);
            }
            gl::HashRuleIndex::eraseOwnerlessEntries(hm, gArena);
        }

/// @brief Install a short subkey with one fixed empty record.
        ///
        /// @details
        /// The nine-byte empty Codec<OwnerSet> frame stores a zero loose byte
        /// and zero signature and owner counts. Repeated installs preserve it.
        /// Short keys survive rule removal until their hash memory is reset
        /// (I-228).
        ///
        /// @param map The subkey record map.
        /// @param numberExpressions The premise count, one or two.
        /// @param data The normalized key's NameId payload.
        /// @param len The payload length.
        /// @return Nothing.
        /// @invariant Presence alone accepts a short subkey; no owner is stored.
        /// @see stageEmptySubkeyRecord, subkeyUSatisfied.
inline void HashRuleInstaller::ensureEmptySubkeyRecord(TypedColdBlobMap<NormKey, OwnerSet>& map,
                                            int32_t numberExpressions,
                                            const NameId* data, int32_t len) {
            assert(numberExpressions > 0
                && numberExpressions < gl::HashRuleIndex::kSubkeyUCheckMinElements);
            char keyBuf[ExecutionParameters::kMaxNormKeyBytes];
            const int32_t kn = encodeNormKeyInto(numberExpressions, data, len,
                keyBuf, ExecutionParameters::kMaxNormKeyBytes);
            const StrSpan keyView(keyBuf, kn);
            if (map.inner().lookup(keyView) == 0) {
                const char emptyRecord[9]{};
                const int32_t recordLength = sizeof(emptyRecord);
                map.inner().assignRun(keyView, emptyRecord, &recordLength, 1);
            }
        }

/// @brief Record a whole key with one owner — the install door of
        ///        `normalizedEncodedKeys`.
        ///
        /// @details
        /// The key bytes are built on the stack via `encodeNormKeyInto`
        /// (byte-exact to `Codec<NormKey>::encode`, the bytes `wholeKeyPresent`
        /// probes) and the owner is added to the key's owner run through
        /// `addOwnerToRun` (a new key gets the one-owner run, a present key
        /// gains the owner, a repeated owner is a no-op). The key is the
        /// request generator's emission gate; it leaves the map only when its
        /// last owner is removed ([I-49](../30_invariants.md#i-49)) or the
        /// map is reset — never by a scope wipe. Single-threaded install site
        /// only ([I-83](../30_invariants.md#i-83)).
        ///
        /// @param map               The whole-key owner map (`normalizedEncodedKeys`).
        /// @param numberExpressions The key's leading count field.
        /// @param data              The key's NameId payload.
        /// @param len               The payload length.
        /// @param owner             The installing rule's packed owner.
        /// @param gArena            Per-slot gen-scratch arena for the run merge.
        /// @see `wholeKeyPresent` — the probe; `ensureEmptySubkeyRecord` — the
        ///      subkey map's short-key door; `addOwnerToRun`.
inline void HashRuleInstaller::addWholeKeyOwner(TypedColdBlobMap<NormKey, RuleOwnerRec>& map,
                                     int32_t numberExpressions,
                                     const NameId* data, int32_t len,
                                     RuleOwner owner, ScratchArena& gArena) {
            char keyBuf[ExecutionParameters::kMaxNormKeyBytes];
            const int32_t kn = encodeNormKeyInto(numberExpressions, data, len,
                keyBuf, ExecutionParameters::kMaxNormKeyBytes);
            gl::HashRuleIndex::addOwnerToRun(map, StrSpan(keyBuf, kn), owner, gArena);
        }

/// @brief Merge one owner's u_ signature into a subkey record — the
        ///        blob-native install door of `normalizedEncodedSubkeys` for
        ///        subkeys of at least `kSubkeyUCheckMinElements` premises.
        ///
        /// @details
        /// Fully blob-native — NEITHER an `OwnerSet` NOR a `serialize` blob is
        /// materialized. The key bytes are built on the stack via
        /// `encodeNormKeyInto` (byte-exact to `Codec<NormKey>::encode`); the run
        /// is probed / written through the raw `inner().lookup` / `inner().assignRun`
        /// doors. The existing record is READ zero-copy through `OwnerSetBlob`
        /// (`peekRecordBytes` arena straddle overload), the owner's u_ signature is
        /// produced by `buildUSignatureRunInto`, and the MERGED blob is assembled
        /// directly on the gen byte-bump tier (reached internally via
        /// `gl::WorkerContext::coreId()`, per-registry fallback): byte 0
        /// `existingLoose || !hasUArg`; `uSignatures` the existing lex-ascending
        /// sigs with the new one inserted at its
        /// `std::vector<std::pair>::operator<` position (idempotent — skipped if
        /// byte-equal). The result is byte-identical to `Codec<OwnerSet>::serialize`
        /// of the decode-merge-reencode oracle (`recordUSignature` +
        /// `Codec<OwnerSet>`, retained as the differential twin). The blob is
        /// assembled in a DISJOINT buffer BEFORE the run-replacing `assignRun`, so
        /// the peeked existing bytes never dangle (09b pitfall 6). A subkey record
        /// is exactly what its growth probe reads (`subkeyUSatisfied`, D-120); the
        /// subkey map has no owner accounting and is never scope-wiped
        /// ([I-49](../30_invariants.md#i-49)). Below the threshold the install
        /// sites call `ensureEmptySubkeyRecord` instead. Single-threaded install
        /// site only ([I-83](../30_invariants.md#i-83)).
        ///
        /// @param map              The subkey map (`normalizedEncodedSubkeys`).
        /// @param numberExpressions The key's leading count field.
        /// @param data             The key's NameId payload.
        /// @param len              The payload length.
        /// @param encList          The subkey's pre-encoded premises.
        /// @param count            Their number (`>= kSubkeyUCheckMinElements`).
        /// @param owner            The installing rule's packed owner — appended
        ///                         to the record's owner list with the index of
        ///                         its signature (-1 when loose); a repeated
        ///                         owner is a no-op.
        /// @see `ensureEmptySubkeyRecord`, `subkeyUSatisfied`,
        ///      `buildUSignatureRunInto`, `OwnerSetBlob`, `encodeNormKeyInto`,
        ///      `recordSubkeyOwner` (the heap oracle), `Codec<OwnerSet>` (the
        ///      blob oracle).
inline void HashRuleInstaller::mergeSubkeySignatures(TypedColdBlobMap<NormKey, OwnerSet>& map,
                                          int32_t numberExpressions, const NameId* data,
                                          int32_t len,
                                          const IntEncodedExpr* encList, NameId count,
                                          RuleOwner owner) {
            using Pair = std::pair<int32_t, NameId>;
            assert(count >= gl::HashRuleIndex::kSubkeyUCheckMinElements
                && "mergeSubkeySignatures: a shorter subkey records presence only");
            // Single-threaded install (I-83): reach the gen byte-bump tier via
            // the worker slot (workers publish gl::WorkerContext::coreId(); -1 -> the
            // reserved last slot). Per-registry fallback derived from the
            // genScratchArenas registry (G-57).
            const unsigned slot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            ScratchArena& gArena = genScratchArenas().forSlot(slot);
            const ArenaOffset mark = gArena.cursor();

            char keyBuf[ExecutionParameters::kMaxNormKeyBytes];
            const int32_t kn = encodeNormKeyInto(numberExpressions, data, len,
                keyBuf, ExecutionParameters::kMaxNormKeyBytes);
            const StrSpan keyView(keyBuf, kn);

            // 1. The owner's u_ signature run (slot-ascending pairs) on the arena
            //    — the byte-exact producer twin of recordUSignature.
            const int32_t sigCap = static_cast<int32_t>(count)
                * static_cast<int32_t>(ExecutionParameters::MAX_ARITY);
            Pair* newSig = reinterpret_cast<Pair*>(gArena.resolve(gArena.alloc(
                sigCap * static_cast<int32_t>(sizeof(Pair)),
                static_cast<int32_t>(alignof(Pair)))));
            bool hasUArg = false;
            int32_t newSigN = 0;
            {
                RT_SCOPE_HERE("SKM_SIGNATURE");
                newSigN = buildUSignatureRunInto(encList, count, newSig, sigCap, hasUArg);
            }

            // 2. Peek the existing record (empty when the key is new). The
            //    arena straddle overload assembles a page-straddling blob onto
            //    gArena with a FRESH alloc (no rewind), so the peeked pointer
            //    survives every later alloc below and never dangles (09b pitfall
            //    6 — the merged blob is assembled in a DISJOINT buffer before the
            //    run-replacing assignRun).
            int32_t id = 0;
            OwnerSetBlob ob{ nullptr, 0 };
            {
            RT_SCOPE_HERE("SKM_LOOKUP_PEEK");
            id = map.inner().lookup(keyView);
            if (id != 0) {
                int32_t blen = 0;
                const char* bp = map.peekRecordBytes(id, 0, blen, gArena);
                ob = OwnerSetBlob{ bp, blen };
            }
            }
            const bool haveExisting = (id != 0);
            const bool existingLoose = haveExisting && ob.hasLooseOwner();
            const int32_t existingSigCount = haveExisting ? ob.uSigCount() : 0;
            const int32_t existingOwnersOff = haveExisting ? ob.ownersOffset() : 0;
            const int32_t existingSigsBytes = haveExisting
                ? (existingOwnersOff - ob.firstSigOffset()) : 0;
            const int32_t existingOwnerCount = haveExisting
                ? ob.ownerCount(existingOwnersOff) : 0;

            // Signature comparator: newSig (run) vs the existing sig at byte
            // offset off — reproduces std::vector<std::pair>::operator< EXACTLY
            // (element-wise, pair compares .first then .second; a shorter run is a
            // prefix and sorts first). Returns <0 when newSig < existing.
            const auto cmpNewVsSig = [&](int32_t off) -> int {
                const int32_t epc = ob.sigPairCount(off);
                const int32_t common = std::min(newSigN, epc);
                for (int32_t k = 0; k < common; ++k) {
                    const int32_t ef = ob.sigPairFirstAt(off, k);
                    if (newSig[k].first != ef)
                        return newSig[k].first < ef ? -1 : 1;
                    const NameId es = ob.sigPairSecondAt(off, k);
                    if (newSig[k].second != es)
                        return newSig[k].second < es ? -1 : 1;
                }
                if (newSigN != epc) return newSigN < epc ? -1 : 1;
                return 0;
            };

            // 3. Signature insert decision: recordUSignature records the sig only
            //    for a non-loose owner (hasUArg), and the std::set insert is
            //    idempotent (skip an existing byte-equal sig).
            //    newSigPos is the new signature's index in the merged sorted
            //    list: the index of the byte-equal signature when one exists,
            //    else the number of existing signatures that sort below it.
            char* out = nullptr;
            int32_t mergedLen = 0;
            {
            RT_SCOPE_HERE("SKM_MERGE");
            RT_NOTE_ITERATIONS_HERE(existingOwnerCount + existingSigCount);   // existing records walked
            bool foundEqualSig = false;
            int32_t newSigPos = 0;
            if (hasUArg && haveExisting) {   // firstSigOffset derefs ob.p
                int32_t off = ob.firstSigOffset();
                for (int32_t s = 0; s < existingSigCount; ++s) {
                    const int c = cmpNewVsSig(off);
                    if (c == 0) { foundEqualSig = true; newSigPos = s; break; }
                    if (c < 0) { newSigPos = s; break; }
                    off += ob.sigBytes(off);
                    newSigPos = s + 1;
                }
            }
            const bool sigInsert = hasUArg && !foundEqualSig;
            const int32_t sigCountOut = existingSigCount + (sigInsert ? 1 : 0);
            const int32_t newSigBytes = sigInsert ? (4 + 8 * newSigN) : 0;
            const int32_t newOwnerSigIndex = hasUArg ? newSigPos : -1;

            // 3b. Owner-pair insert decision. The owner section is a run of
            //     (owner, signature index) PAIRS sorted by owner then index: one
            //     rule reaches the same subkey through several premise prefixes
            //     (its permutations) and those prefixes can carry different u_
            //     literals, so one owner may hold several signatures. A repeated
            //     pair (same owner, same signature) is a no-op; an existing
            //     pair's index is read AFTER the renumbering the new signature's
            //     insert implies, so the comparison and the write agree.
            const auto renumbered = [&](int32_t idx) -> int32_t {
                return (sigInsert && idx >= newSigPos) ? idx + 1 : idx;
            };
            bool pairPresent = false;
            int32_t ownerInsertAt = existingOwnerCount;
            for (int32_t i = 0; i < existingOwnerCount; ++i) {
                const RuleOwner o = ob.ownerAt(existingOwnersOff, i);
                const int32_t idx = renumbered(ob.ownerSigIndexAt(existingOwnersOff, i));
                if (o == owner && idx == newOwnerSigIndex) {
                    pairPresent = true;
                    break;
                }
                if (ownerInsertAt == existingOwnerCount
                    && (o > owner || (o == owner && idx > newOwnerSigIndex))) {
                    ownerInsertAt = i;
                }
            }
            if (pairPresent) {
                assert(!sigInsert
                    && "mergeSubkeySignatures: a present owner pair brings a new signature");
                gArena.popTo(mark);
                return;
            }
            const int32_t ownerCountOut = existingOwnerCount + 1;

            // 4. Merged blob length (Rule-19 exact-length fill below).
            mergedLen = 1 + 4 + existingSigsBytes + newSigBytes
                + 4 + OwnerSetBlob::kOwnerRecBytes * ownerCountOut;
            out = reinterpret_cast<char*>(gArena.resolve(gArena.alloc(
                mergedLen, 1)));
            int32_t at = 0;
            const auto wr32 = [&](int32_t v) {
                std::memcpy(out + at, &v, sizeof(int32_t)); at += 4;
            };
            const auto writeNewSig = [&]() {
                wr32(newSigN);
                for (int32_t k = 0; k < newSigN; ++k) {
                    wr32(newSig[k].first); wr32(newSig[k].second);
                }
            };
            const auto wrOwner = [&](RuleOwner o, int32_t sigIndex) {
                std::memcpy(out + at, &o, sizeof(RuleOwner)); at += 8;
                wr32(sigIndex);
            };

            // byte 0: hasLooseOwner = existing || (owner has no u_ arg) — the
            // decode-then-recordUSignature loose path, byte-for-byte.
            out[at++] = static_cast<char>((existingLoose || !hasUArg) ? 1 : 0);

            // uSignatures: existing sigs (lex-ascending) with newSig inserted at
            // its lex position (skipped when !sigInsert).
            wr32(sigCountOut);
            {
                bool sigWritten = !sigInsert;
                int32_t off = haveExisting ? ob.firstSigOffset() : 0;
                for (int32_t s = 0; s < existingSigCount; ++s) {
                    const int32_t sb = ob.sigBytes(off);
                    if (!sigWritten && cmpNewVsSig(off) < 0) {
                        writeNewSig();
                        sigWritten = true;
                    }
                    std::memcpy(out + at, ob.p + off, static_cast<std::size_t>(sb));
                    at += sb;
                    off += sb;
                }
                if (!sigWritten) writeNewSig();
            }

            // owner pairs: the existing pairs (ascending by owner, then index)
            // with the new pair inserted at its position; an existing pair's
            // signature index moves up by one when the new signature was
            // inserted at or below it.
            wr32(ownerCountOut);
            for (int32_t i = 0; i < existingOwnerCount; ++i) {
                if (i == ownerInsertAt) wrOwner(owner, newOwnerSigIndex);
                wrOwner(ob.ownerAt(existingOwnersOff, i),
                        renumbered(ob.ownerSigIndexAt(existingOwnersOff, i)));
            }
            if (ownerInsertAt == existingOwnerCount) wrOwner(owner, newOwnerSigIndex);
            assert(at == mergedLen
                && "mergeSubkeySignatures: merged subkey blob length drift");
            } // RT_SCOPE SKM_MERGE

            // 5. One raw whole-run replace (M == 1), byte-identical to
            //    Codec<OwnerSet>::serialize of the decode-merge-reencode oracle.
            {
                RT_SCOPE_HERE("SKM_WRITE");
                map.inner().assignRun(keyView, out, &mergedLen, 1);
            }
            gArena.popTo(mark);
        }
}
