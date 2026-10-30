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
#include "../compiler/compact_expression_syntax.hpp"
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
#include "phase13_timing_slot.hpp"
#include "watch_ledger.hpp"
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
class DisintegrationProcessor;
class HashRuleInstaller;
class RejectedSourceReplay;
class StatementMemory;
namespace tests { struct CompactRuleProcessorFixture; }
/// @brief Compact Rule Processor operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class CompactRuleProcessor {
friend struct tests::CompactRuleProcessorFixture;
public:
/// @brief Bind inputs for compact rule processor.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param canonicalizer Borrowed Canonicalizer interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param disintegrationProcessor Borrowed DisintegrationProcessor interface or state; must outlive this component.
/// @param hashRuleInstaller Borrowed HashRuleInstaller interface or state; must outlive this component.
/// @param statementMemory Borrowed StatementMemory interface or state; must outlive this component.
/// @param integrationBuilder Borrowed IntegrationInstructionBuilder interface or state; must outlive this component.
/// @param rejectedSourceReplay Borrowed RejectedSourceReplay (the demand-built rule decision); must outlive this component.
/// @return None; construction binds the supplied references.
CompactRuleProcessor(Canonicalizer& canonicalizer, const ProverParameters& parameters, DisintegrationProcessor& disintegrationProcessor, HashRuleInstaller& hashRuleInstaller, StatementMemory& statementMemory, IntegrationInstructionBuilder& integrationBuilder, RejectedSourceReplay& rejectedSourceReplay);

public:
/// @brief Install one filled integration compact at a scope: expand it,
        ///        install its rule through the integration route and record
        ///        the compact as the rule's carrier
        ///        (D-503).
        ///
        /// @details
        /// The compact is an entry of an entity's integration set whose owner
        /// tokens were filled with the instance's `u_`-prefixed arguments. It
        /// is expanded first (`IntegrationInstructionBuilder::
        /// expandIntegrationCompactScratch`, fresh witness names) and the
        /// expansion is returned through @p premises / @p head in every case.
        /// A compact already expanded at @p validity (its carrier entry
        /// exists) installs nothing. Otherwise,
        /// with history tracked, the compact (`u_`-stripped) takes the row
        /// @p tag citing @p deps and both forms of its rule take an
        /// `expansion` row citing the compact, each also on the mail-out;
        /// the rule is installed into the overall, local and delta instances
        /// with justification `integration`, and the carrier index
        /// records `(compact, validity) → (rule, validity)` with install kind
        /// `Integration`. The compact itself is never a statement and never
        /// mailed. Single-threaded seam only.
        ///
        /// @param mb         The LB.
        /// @param compact    The filled compact in `u_`-prefixed form.
        /// @param validity   The install scope (span stable across the call).
        /// @param tag        The compact's history tag.
        /// @param deps       The compact's history citations.
        /// @param depN       Their number.
        /// @param arena      String-tier scratch arena receiving the expansion.
        /// @param premises   Out: the expanded premises (`u_` form).
        /// @param premiseCap Capacity of @p premises.
        /// @param premiseN   Out: their number.
        /// @param head       Out: the expanded head (`u_` form).
        /// @param kind       `Integration` (default) or `IntegrationFlat`, a
        ///                   root goal's flat variant whose install adds the
        ///                   near-miss demand variants and whose carrier
        ///                   record keeps the kind for its removal and its
        ///                   canonical re-installation
        ///                   (D-574).
        /// @return `true` when the rule was installed, `false` when the compact
        ///         was already expanded at @p validity.
        /// @invariant 0% heap.
        /// @see expandIntegrationCompactScratch, appendCompactExpansion,
        ///      removeCompactExpansion, expandCanonicalIntegrationCompact.
bool installIntegrationCompact(Memory& mb, StrSpan compact, StrSpan validity,
                                       OriginTag tag, const OriginDep* deps, int32_t depN,
                                       ScratchArena& arena, StrSpan* premises, int32_t premiseCap,
                                       int32_t& premiseN, StrSpan& head,
                                       RuleInstallKind kind = RuleInstallKind::Integration);

public:
/// @brief The install half of `installIntegrationCompact`: install an
        ///        integration compact already expanded into @p premises /
        ///        @p head (D-503).
        ///
        /// @details
        /// For a caller whose history citation depends on the expansion (a
        /// disjunct rule cites its expanded head). A compact already expanded
        /// at @p validity installs nothing; otherwise the history rows, the
        /// three integration installs
        /// and the carrier record follow, exactly as `installIntegrationCompact`
        /// documents.
        ///
        /// @param mb        The LB.
        /// @param compact   The filled compact in `u_`-prefixed form.
        /// @param validity  The install scope (span stable across the call).
        /// @param tag       The compact's history tag.
        /// @param deps      The compact's history citations.
        /// @param depN      Their number.
        /// @param arena     String-tier scratch arena.
        /// @param premises  The expanded premises (`u_` form).
        /// @param premiseN  Their number.
        /// @param head      The expanded head (`u_` form).
        /// @param kind      `Integration` (default) or `IntegrationFlat`
        ///                  (asserted one of the two), as for
        ///                  `installIntegrationCompact`.
        /// @return `true` when the rule was installed, `false` when the compact
        ///         was already expanded at @p validity.
        /// @invariant 0% heap.
        /// @see installIntegrationCompact.
bool installExpandedIntegrationCompact(Memory& mb, StrSpan compact, StrSpan validity,
                                               OriginTag tag, const OriginDep* deps, int32_t depN,
                                               ScratchArena& arena, const StrSpan* premises, int32_t premiseN,
                                               StrSpan head,
                                               RuleInstallKind kind = RuleInstallKind::Integration);

public:
/// @brief What one carrier's records hold.
        ///
        /// @details `statement`: an expansion installed through the door (install
        /// kind `Local` / `External`); `integration`: an integration rule (kind
        /// `Integration` or `IntegrationFlat`, at most one per carrier);
        /// `integrationFlat`: that rule is a root goal's flat variant. The same
        /// compact text at one scope can carry both a statement and an
        /// integration record.
struct CarrierKinds {
    bool statement = false;
    bool integration = false;
    bool integrationFlat = false;
};

public:
/// @brief The record kinds of the carrier `(text, validity)`.
        ///
        /// @details
        /// An absent carrier (an unminted name, no index entry) holds nothing.
        /// An integration carrier holds exactly one integration-kind record,
        /// `Integration` or `IntegrationFlat` (asserted): the compact is
        /// expanded as an integration rule once per scope; `integrationFlat`
        /// reports a root goal's flat variant
        /// (D-574). Reads only: non-minting lookups, record peeks on the per-slot
        /// gen-scratch arena (rewound).
        ///
        /// @param mb         The LB.
        /// @param carrierKey The carrier's packed `(originalId, validityId)` key.
        /// @return The kinds; all `false` for an absent carrier.
        /// @see installIntegrationCompact, applyEquivalenceClassToCompactImplications,
        ///      expandCanonicalCompact.
static CarrierKinds carrierKinds(Memory& mb, int64_t carrierKey);

/// @brief `carrierKinds` of the carrier named by text and scope.
        ///
        /// @param mb       The LB.
        /// @param text     The compact text (`u_`-stripped).
        /// @param validity Its scope.
        /// @return The kinds; all `false` when either name is unminted.
        /// @see carrierKinds(Memory&, int64_t).
static CarrierKinds carrierKinds(Memory& mb, StrSpan text, StrSpan validity);

private:
/// @brief The `equality1` citations of a canonical compact K*: the
        ///        carrier K at its scope, then every fired pair as
        ///        `(=[from,to])` cited where its own origin row lives.
        ///
        /// @details
        /// The class scope first, else the nearest ancestor holding the
        /// equality's row (asserted found) — the citation walk of the
        /// `applyEquivalenceClass` commit. The equality texts and copied
        /// scope names ride @p strArena.
        ///
        /// @param mb       The LB.
        /// @param carrier  The non-canonical compact K.
        /// @param validity Its scope.
        /// @param pairs    The fired pairs.
        /// @param pairN    Their number (at most `MAX_ARITY`).
        /// @param strArena String-tier scratch arena.
        /// @param deps     Out: at least `1 + MAX_ARITY` entries.
        /// @return The number of citations written.
        /// @see expandCanonicalCompact, expandCanonicalIntegrationCompact.
int32_t canonicalEqualityDeps(Memory& mb, StrSpan carrier, StrSpan validity,
                                      const CompactCanonPair* pairs, int32_t pairN,
                                      ScratchArena& strArena, OriginDep* deps);

private:
/// @brief Re-form an integration rule after a class merge: install the
        ///        canonical form K* of its carrier K
        ///        (D-503).
        ///
        /// @details
        /// The integration twin of `expandCanonicalCompact`, with no statement
        /// registration. Nothing happens when K* is already expanded at @p
        /// validity. Otherwise K* (`u_`-prefixed) goes through
        /// `installIntegrationCompact` with the history row `K* ← equality1 ← K + (=[from,to])…`
        /// (`canonicalEqualityDeps`); its rule takes `expansion ← K*`.
        ///
        /// @param mb       The LB.
        /// @param kstar    The canonical form (`u_`-stripped, caller-stable).
        /// @param carrier  The non-canonical compact K.
        /// @param validity The scope.
        /// @param pairs    The fired pairs.
        /// @param pairN    Their number.
        /// @param kind     K's install kind, `Integration` or `IntegrationFlat`:
        ///                 K* is installed with the same kind
        ///                 (D-574).
        /// @see installCanonicalCompacts, installIntegrationCompact.
void expandCanonicalIntegrationCompact(Memory& mb, StrSpan kstar, StrSpan carrier,
                                               StrSpan validity,
                                               const CompactCanonPair* pairs, int32_t pairN,
                                               RuleInstallKind kind);

private:
/// @brief Record in the carrier index that @p carrier (a statement at
        ///        @p validity) was expanded into the implication @p implText
        ///        installed at @p implScope — the install door's write.
        ///
        /// @details
        /// Key = the carrier's packed `(originalId, validityId)` statement
        /// key; the record holds the `lbStateInterner` ids of the expanded
        /// text and its scope (the pair `expandedImplications` holds) and the
        /// install kind. A new carrier opens an empty run; the record is
        /// appended through the raw run door (nine bytes, no codec heap). A
        /// repeated `(text, scope)` for one carrier is asserted — the door
        /// expands a carrier once. Single-threaded seam only.
        ///
        /// @param mb        The LB.
        /// @param carrier   The carrier statement text.
        /// @param validity  Its scope.
        /// @param implText  The expanded implication text.
        /// @param implScope The expanded implication's scope.
        /// @param kind      Which instances the install reached.
        /// @see `CompactExpansionRec`, `removeCompactExpansion`.
static void appendCompactExpansion(Memory& mb, StrSpan carrier, StrSpan validity,
                                           StrSpan implText, StrSpan implScope,
                                           RuleInstallKind kind);

private:
/// @brief Remove every rule a carrier was expanded into — from every
        ///        instance that still holds it — and stage the carrier's index
        ///        entry for erasure.
        ///
        /// @details
        /// Walks the carrier's records: the persistent instances by install
        /// kind (`overallHashMemory` always, `localHashMemory` for a local
        /// derivation — both asserted present), the transient pair
        /// (`localHashMemoryDelta`, `workingMemory`) only when the per-burst
        /// reset has not yet emptied them of the rule (`onlyIfPresent`); each
        /// `(text, scope)` pair leaves `expandedImplications` (the dump's
        /// source). An `Integration` record's rule leaves through the
        /// integration route instead — overall and local, the delta instance
        /// if still present, justification `integration`; it has no
        /// `expandedImplications` pair
        /// (D-503). The carrier key is minted into @p removedCarriers;
        /// `finishCompactRemovals` erases the entries and finishes every
        /// instance. The decoded text spans alias `lbStateInterner` cold
        /// bytes, which nothing on this path mints. Single-threaded seam only.
        ///
        /// @param mb              The LB.
        /// @param carrierKey      The carrier's packed statement key (asserted present).
        /// @param visited         The per-removal visited-entry set (caller-owned arena).
        /// @param dropSet         The apply's staged LMV drop set.
        /// @param removedCarriers The apply's staged carrier-key erasure set.
        /// @see `appendCompactExpansion`, `removeRuleFromHashMemory`,
        ///      `finishCompactRemovals`.
void removeCompactExpansion(Memory& mb, int64_t carrierKey,
                                    ColdHashSet<BytesKeyStore>& visited,
                                    ColdHashSet<PodKeyStore<int64_t>>& dropSet,
                                    ColdHashSet<PodKeyStore<int64_t>>& removedCarriers);

public:
/// @brief Finish an apply's compact removals: erase the removed
        ///        carriers' index entries in one pass and finish every
        ///        hash-memory instance (LMV compaction + owner-less erasure).
        ///
        /// @param mb              The LB.
        /// @param dropSet         The apply's staged LMV drop set.
        /// @param removedCarriers The apply's staged carrier keys.
        /// @param gArena          Per-slot gen-scratch arena.
        /// @see `removeCompactExpansion`, `finishRuleRemovals`.
static void finishCompactRemovals(Memory& mb,
                                          const ColdHashSet<PodKeyStore<int64_t>>& dropSet,
                                          const ColdHashSet<PodKeyStore<int64_t>>& removedCarriers,
                                          ScratchArena& gArena);

public:


public:


private:
/// @brief The canonical form of a compact at a scope — every argument
        ///        replaced by the canonical representative of the class that
        ///        holds it in the maintained complete view at @p validity.
        ///
        /// @details
        /// The maintained canonical bucket includes the target's own classes
        /// and every visible ancestor class, with overlaps already consolidated.
        /// `chooseCanonicalId` selects at the target scope: normal > `int_` >
        /// `it_`, lex-min in the tier, weak target-scope members skipped (a
        /// class with no strong member maps
        /// nothing). The rewrite is `replaceKeysScratch` over the fired
        /// pairs (token-boundary, order-free); the pairs are returned for
        /// the `equality1` citation, and every fired pair's admission levels
        /// (D-280, `forEachLevel(packEqPairKey(from, to))`) are unioned into
        /// @p levelRun. Reads only: non-minting NameMap lookups, class-blob
        /// peeks, the name-kind memo. Zero heap; the peeks and the rewritten
        /// text ride the two scratch arenas the caller frames.
        ///
        /// @param mb        The LB.
        /// @param text      The compact text.
        /// @param validity  Its scope.
        /// @param strArena  String-tier scratch for the rewritten text.
        /// @param gArena    Gen-scratch for the class-blob peeks.
        /// @param kstarOut  Out: the rewritten text (only when `true` is returned).
        /// @param pairs     Out: the fired pairs (`MAX_ARITY` cap).
        /// @param pairN     Out: their number.
        /// @param levelRun  In/out: ascending-unique level run the pairs' levels join.
        /// @param levelN    In/out: its length.
        /// @param levelCap  Its capacity.
        /// @return `true` when the text changed (the compact is not canonical).
        /// @see `chooseCanonicalId`, `installExpandedImplication`,
        ///      `applyEquivalenceClassToCompactImplications`.
bool canonicalizeCompact(Memory& mb, StrSpan text, StrSpan validity,
                                 ScratchArena& strArena, ScratchArena& gArena,
                                 ScratchString& kstarOut,
                                 CompactCanonPair* pairs, int32_t& pairN,
                                 int* levelRun, int32_t& levelN, int32_t levelCap);

private:
/// @brief Whether a compact's rules are installed here — the carrier
        ///        index has an entry for `(text, validity)`.
        ///
        /// @param mb       The LB.
        /// @param text     The compact text.
        /// @param validity Its scope.
        /// @return `true` when the carrier has recorded expansions at this LB.
static bool compactExpandedHere(const Memory& mb, StrSpan text, StrSpan validity);

private:
/// @brief Expand the canonical form K* of a non-canonical compact K
        ///        into hash memory and register it — the "only canonical
        ///        enters hash memory" step both sites share.
        ///
        /// @details
        /// Nothing happens when K* is already expanded here or known at a
        /// strict ancestor (its rules are visible from there). Otherwise K*
        /// is disintegrated (`disintegrateExpr2`, the compact's own
        /// expansion — `trackExpansionHistory` records `rule ← expansion ←
        /// K*`), every produced implication is installed through
        /// `installExpandedImplication` (K* is its own canonical carrier, so
        /// the gate passes), and K* is registered as a statement at @p
        /// validity with the `equality1` history row `K* ← K + (=[from,to])…`
        /// (each equality cited at the scope where its own origin row lives
        /// — the class scope or the nearest ancestor holding it) and the
        /// level run @p levelRun (the deposit's levels ∪ the fired pairs'
        /// levels, D-280); a K* already known at this scope keeps its row and
        /// gains the history row only. Zero heap; the disintegration products
        /// ride the per-slot gen-scratch arena.
        ///
        /// @param mb        The LB.
        /// @param kstar     The canonical form (caller-stable text).
        /// @param carrier   The non-canonical compact K it came from.
        /// @param validity  The scope (K's, K*'s and the rules').
        /// @param pairs     The fired pairs.
        /// @param pairN     Their number.
        /// @param levelRun  The level run for K* and its rules.
        /// @param levelN    Its length.
        /// @param status    The deposit status (decides local / external).
        /// @param iteration The deposit iteration.
        /// @see `canonicalizeCompact`, `installExpandedImplication`.
void expandCanonicalCompact(Memory& mb, StrSpan kstar, StrSpan carrier,
                                    StrSpan validity,
                                    const CompactCanonPair* pairs, int32_t pairN,
                                    const int* levelRun, int32_t levelN,
                                    int status, int iteration);

public:
/// @brief The class-side reconcile of the canonical gate — the
        ///        `applyEquiClasses` hook over the registered compact
        ///        implications: a compact that a class made non-canonical
        ///        loses its rules and its canonical form is expanded; a
        ///        registered canonical compact without rules is expanded.
        ///
        /// @details
        /// Mirrors the rejected-map hooks' walk (D-307): the per-apply
        /// snapshot @p buckets is built at the first use from the compact
        /// rows of `intEncodedStatements` (`isCompactImplicationSpan`, keys in
        /// @p compactKeys) with `(validity, argument) → keys` postings decoded
        /// through the NameMap; the candidates are the compacts at the class
        /// validity OR a strict descendant of it (a class is visible below its
        /// scope) that name a non-canonical member. Per candidate: a carrier
        /// already removed in this apply is skipped; one whose text is
        /// unchanged under the class and whose rules are installed is done;
        /// otherwise the full canonical form under every visible class
        /// (`canonicalizeCompact`) decides — changed: its rules leave
        /// (`removeCompactExpansion`, staged) and its logical carrier key queues
        /// for replacement — an integration carrier in
        /// @p pendingIntegrationCompacts (D-503),
        /// any other in @p pendingCompacts. Unchanged but unexpanded carriers queue as well.
        /// The caller finishes all obsolete removals with `finishCompactRemovals`
        /// before `installCanonicalCompacts` exposes replacements. One batched
        /// compaction covers the complete consumer pass.
        /// The decoded candidate text is copied to the string arena before the
        /// reconcile mints the NameMap (I-3). Gated on `!skip_eq_classes` by
        /// the caller.
        ///
        /// @tparam ClassT `EquivalenceClass` or `EquivalenceClassView`.
        /// @param clss            The class (delta or registry).
        /// @param memoryBlock     The LB.
        /// @param validityName    The class scope.
        /// @param buckets         The per-apply compact index (built here on first use).
        /// @param compactKeys     The per-apply snapshot of compact statement keys.
        /// @param visited         The per-removal visited-entry set.
        /// @param dropSet         The apply's staged LMV drop set.
        /// @param removedCarriers The apply's staged carrier keys.
        /// @param pendingCompacts Logical carrier keys to install after obsolete removals finish.
        /// @param pendingIntegrationCompacts Integration carrier keys, each with
        ///        its install kind (`Integration` or `IntegrationFlat`, as a
        ///        byte), installed after obsolete removals finish.
        /// @param selected Optional complete-class dependency selection.
        /// @return The number of hits (removals + expansions).
        /// @see `installExpandedImplication` (the door-side gate),
        ///      `canonicalizeCompact`, `removeCompactExpansion`,
        ///      `expandCanonicalCompact`, `finishCompactRemovals`.
template <class ClassT>
        int32_t applyEquivalenceClassToCompactImplications(
                const ClassT& clss, Memory& memoryBlock, StrSpan validityName,
                RejectedValidityBuckets& buckets,
                PagedVector<int64_t>& compactKeys,
                ColdHashSet<BytesKeyStore>& visited,
                ColdHashSet<PodKeyStore<int64_t>>& dropSet,
                ColdHashSet<PodKeyStore<int64_t>>& removedCarriers,
                TypedColdSet<int64_t>& pendingCompacts,
                TypedColdMap<int64_t, uint8_t>& pendingIntegrationCompacts,
            const TypedColdSet<int64_t>* selected = nullptr);

public:
/// @brief Install canonical compact replacements after obsolete rules leave.
        /// @details Logical carrier keys are sorted by decoded expression and scope.
        /// Each replacement uses the current complete class view, original carrier
        /// levels and equality evidence. No vector compaction occurs during installs.
        /// Integration carriers follow in the same order, each re-formed by
        /// `expandCanonicalIntegrationCompact` with its recorded install kind.
        /// @param mb Owning block with completed rule removals.
        /// @param pending Coalesced original carrier keys, emptied on completion.
        /// @param pendingIntegration Integration carrier keys with their install
        ///        kinds, emptied on completion.
        /// @return Nothing.
        /// @invariant All hash memories have finished owner-less erasure before entry.
inline void installCanonicalCompacts(Memory& mb,TypedColdSet<int64_t>& pending,
                                     TypedColdMap<int64_t, uint8_t>& pendingIntegration);

public:
/// @brief Install one expanded implication into the instances its
        ///        status selects, record it in the two indexes — after the
        ///        canonical gate: a rule whose carrier is a non-canonical
        ///        compact is NOT installed; its canonical form is expanded
        ///        instead (once).
        ///
        /// @details
        /// The install door's per-implication body. Gate first: a compact
        /// carrier (`isCompactImplicationSpan`) is canonicalized at the
        /// rule's scope (`canonicalizeCompact`); when it changes, the rule
        /// of the non-canonical K is skipped and `expandCanonicalCompact`
        /// installs K*'s rules (if hash memory lacks them) — "only the
        /// canonical form of a compact enters hash memory". Otherwise the
        /// rule is installed as before: `overallHashMemory` always,
        /// `workingMemory` for an external-mail carrier (status 3), else
        /// `localHashMemory` + `localHashMemoryDelta` (D-176); the D-274
        /// `expansion` row for a status-3 recovery; the `expandedImplications`
        /// pair (the dump's source) and the carrier index record. Skipped
        /// entirely under `skip_eq_classes` (the incubator batches, no
        /// classes) — the gate, not the install.
        ///
        /// @param mb                 The LB.
        /// @param impStr             The expanded implication text (channel span, stable).
        /// @param impValidity        Its scope.
        /// @param carrier            The statement whose disintegration produced it.
        /// @param status             The deposit status.
        /// @param iteration          The deposit iteration.
        /// @param involvedLevels     The deposit's level run.
        /// @param involvedLevelCount Its length.
        /// @param doorExpr           The deposited expression; an inherited OR's rules cite their compact carrier instead.
        /// @param doorValidity       Its scope.
        /// @see `addExprToMemoryBlock`, `expandCanonicalCompact`, `appendCompactExpansion`.
void installExpandedImplication(Memory& mb, StrSpan impStr, StrSpan impValidity,
                                        StrSpan carrier, int status, int iteration,
                                        const int* involvedLevels, int32_t involvedLevelCount,
                                        StrSpan doorExpr, StrSpan doorValidity);
private:

Canonicalizer& canonicalizer;
const ProverParameters& parameters;
DisintegrationProcessor& disintegrationProcessor;
HashRuleInstaller& hashRuleInstaller;
StatementMemory& statementMemory;
IntegrationInstructionBuilder& integrationBuilder;
RejectedSourceReplay& rejectedSourceReplay;
};
}
#include "../logical_block/disintegration_processor.hpp"
#include "../inference/hash_rule_installer.hpp"
#include "../logical_block/statement_memory.hpp"
#include "../logical_block/rejected_source_replay.hpp"
namespace gl {

/// @brief Record in the carrier index that @p carrier (a statement at
        ///        @p validity) was expanded into the implication @p implText
        ///        installed at @p implScope — the install door's write.
        ///
        /// @details
        /// Key = the carrier's packed `(originalId, validityId)` statement
        /// key; the record holds the `lbStateInterner` ids of the expanded
        /// text and its scope (the pair `expandedImplications` holds) and the
        /// install kind. A new carrier opens an empty run; the record is
        /// appended through the raw run door (nine bytes, no codec heap). A
        /// repeated `(text, scope)` for one carrier is asserted — the door
        /// expands a carrier once. Single-threaded seam only.
        ///
        /// @param mb        The LB.
        /// @param carrier   The carrier statement text.
        /// @param validity  Its scope.
        /// @param implText  The expanded implication text.
        /// @param implScope The expanded implication's scope.
        /// @param kind      Which instances the install reached.
        /// @see `CompactExpansionRec`, `removeCompactExpansion`.
inline void CompactRuleProcessor::appendCompactExpansion(Memory& mb, StrSpan carrier, StrSpan validity,
                                           StrSpan implText, StrSpan implScope,
                                           RuleInstallKind kind) {
            const int64_t key = packStatementKey(mb.nameMap.encode(carrier),
                                                 mb.nameMap.encode(validity));
            char blob[Codec<CompactExpansionRec>::kBytes];
            const int32_t textId = mb.lbStateInterner.encode(implText);
            const int32_t scopeId = mb.lbStateInterner.encode(implScope);
            std::memcpy(blob, &textId, 4);
            std::memcpy(blob + 4, &scopeId, 4);
            blob[8] = static_cast<char>(kind);
            int32_t id = mb.compactExpansions.lookup(key);
            if (id == 0) {
                id = mb.compactExpansions.inner().assignRun(key, nullptr, nullptr, 0);
            } else {
                // A carrier disintegrated again (a re-delivered compact, an
                // internal-mail re-send) re-installs the same rule: the owner
                // runs and expandedImplications take that as a no-op, so does
                // the index — the identical record is not appended twice.
                const unsigned slot = (gl::WorkerContext::coreId() >= 0)
                    ? static_cast<unsigned>(gl::WorkerContext::coreId())
                    : genScratchArenas().slotCount() - 1;
                ScratchArena& gArena = genScratchArenas().forSlot(slot);
                const ArenaOffset mark = gArena.cursor();
                bool present = false;
                for (int32_t j = 0; j < mb.compactExpansions.runLen(id) && !present; ++j) {
                    int32_t bl = 0;
                    const char* bp = mb.compactExpansions.peekRecordBytes(id, j, bl, gArena);
                    present = (bl == Codec<CompactExpansionRec>::kBytes
                               && std::memcmp(bp, blob, Codec<CompactExpansionRec>::kBytes) == 0);
                }
                gArena.popTo(mark);
                if (present) return;
            }
            mb.compactExpansions.inner().appendBlobToRun(id, blob,
                Codec<CompactExpansionRec>::kBytes);
            // One more carrier of this rule (two compacts may expand into one
            // rule text; the rule leaves only with its last carrier).
            const int64_t ruleKey = packInt32Pair(textId, scopeId);
            const int32_t cid = mb.expansionCarrierCount.lookup(ruleKey);
            if (cid == 0) mb.expansionCarrierCount.insert(ruleKey, 1);
            else mb.expansionCarrierCount.setValueAt(cid, *mb.expansionCarrierCount.find(ruleKey) + 1);
        }

/// @brief Remove every rule a carrier was expanded into — from every
        ///        instance that still holds it — and stage the carrier's index
        ///        entry for erasure.
        ///
        /// @details
        /// Walks the carrier's records: the persistent instances by install
        /// kind (`overallHashMemory` always, `localHashMemory` for a local
        /// derivation — both asserted present), the transient pair
        /// (`localHashMemoryDelta`, `workingMemory`) only when the per-burst
        /// reset has not yet emptied them of the rule (`onlyIfPresent`); each
        /// `(text, scope)` pair leaves `expandedImplications` (the dump's
        /// source). An `Integration` record's rule leaves through the
        /// integration route instead — overall and local, the delta instance
        /// if still present, justification `integration`; it has no
        /// `expandedImplications` pair
        /// (D-503). The carrier key is minted into @p removedCarriers;
        /// `finishCompactRemovals` erases the entries and finishes every
        /// instance. The decoded text spans alias `lbStateInterner` cold
        /// bytes, which nothing on this path mints. Single-threaded seam only.
        ///
        /// @param mb              The LB.
        /// @param carrierKey      The carrier's packed statement key (asserted present).
        /// @param visited         The per-removal visited-entry set (caller-owned arena).
        /// @param dropSet         The apply's staged LMV drop set.
        /// @param removedCarriers The apply's staged carrier-key erasure set.
        /// @see `appendCompactExpansion`, `removeRuleFromHashMemory`,
        ///      `finishCompactRemovals`.
inline void CompactRuleProcessor::removeCompactExpansion(Memory& mb, int64_t carrierKey,
                                    ColdHashSet<BytesKeyStore>& visited,
                                    ColdHashSet<PodKeyStore<int64_t>>& dropSet,
                                    ColdHashSet<PodKeyStore<int64_t>>& removedCarriers) {
            const int32_t id = mb.compactExpansions.lookup(carrierKey);
            assert(id != 0 && "removeCompactExpansion: the carrier has no expansion");
            const unsigned slot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            ScratchArena& gArena = genScratchArenas().forSlot(slot);
            const int32_t M = mb.compactExpansions.runLen(id);
            for (int32_t j = 0; j < M; ++j) {
                const ArenaOffset mark = gArena.cursor();
                int32_t bl = 0;
                const char* bp = mb.compactExpansions.peekRecordBytes(id, j, bl, gArena);
                CompactExpansionRec rec;
                std::memcpy(&rec.implTextId, bp, 4);
                std::memcpy(&rec.scopeId, bp + 4, 4);
                rec.kind = static_cast<RuleInstallKind>(static_cast<uint8_t>(bp[8]));
                gArena.popTo(mark);
                // The rule leaves only with its last carrier.
                const int64_t ruleKey = packInt32Pair(rec.implTextId, rec.scopeId);
                const int32_t cid = mb.expansionCarrierCount.lookup(ruleKey);
                assert(cid != 0 && *mb.expansionCarrierCount.find(ruleKey) > 0
                    && "removeCompactExpansion: a recorded expansion without a carrier count");
                const int32_t left = *mb.expansionCarrierCount.find(ruleKey) - 1;
                mb.expansionCarrierCount.setValueAt(cid, left);
                if (left > 0) continue;
                const StrSpan text = mb.lbStateInterner.decodeView(rec.implTextId);
                const StrSpan scope = mb.lbStateInterner.decodeView(rec.scopeId);
                if (isIntegrationInstallKind(rec.kind)) {
                    RT_SCOPE_HERE("RCE_REMOVE_INTEGRATION_RULE");
                    // The integration route, replayed exactly as installed:
                    // a flat root rule takes its near-miss variants along.
                    const bool flat = rec.kind == RuleInstallKind::IntegrationFlat;
                    hashRuleInstaller.removeRuleFromHashMemory(mb, mb.overallHashMemory, text, scope,
                        visited, dropSet, /*onlyIfPresent=*/false, RuleJustification::integration, flat);
                    hashRuleInstaller.removeRuleFromHashMemory(mb, mb.localHashMemory, text, scope,
                        visited, dropSet, /*onlyIfPresent=*/false, RuleJustification::integration, flat);
                    hashRuleInstaller.removeRuleFromHashMemory(mb, mb.localHashMemoryDelta, text, scope,
                        visited, dropSet, /*onlyIfPresent=*/true, RuleJustification::integration, flat);
                    continue;
                }
                {
                    RT_SCOPE_HERE("RCE_REMOVE_RULES");
                    hashRuleInstaller.removeRuleFromHashMemory(mb, mb.overallHashMemory, text, scope,
                                             visited, dropSet);
                    if (rec.kind == RuleInstallKind::Local) {
                        hashRuleInstaller.removeRuleFromHashMemory(mb, mb.localHashMemory, text, scope,
                                                 visited, dropSet);
                        hashRuleInstaller.removeRuleFromHashMemory(mb, mb.localHashMemoryDelta, text, scope,
                                                 visited, dropSet, /*onlyIfPresent=*/true);
                    } else {
                        hashRuleInstaller.removeRuleFromHashMemory(mb, mb.workingMemory, text, scope,
                                                 visited, dropSet, /*onlyIfPresent=*/true);
                    }
                }
                {
                    RT_SCOPE_HERE("RCE_EXPANDED_ERASE");
                    // Deferred inside an apply: one eraseIf at
                    // finishCompactRemovals instead of a full column slide plus
                    // index rebuild per rule. A pair an install revived meanwhile
                    // is erased for real here, so the final pass (which keeps
                    // revived pairs) leaves the column as the immediate sequence
                    // would.
                    DeferredExpandedErase* const deferred = mb.deferredExpandedErase;
                    if (deferred == nullptr || deferred->revived.lookup(ruleKey) != 0) {
                        mb.expandedImplications.erase(LbStatePairKey{ rec.implTextId, rec.scopeId });
                    } else {
                        deferred->pending.mint(ruleKey);
                    }
                }
            }
            removedCarriers.mint(carrierKey);
        }

/// @brief Finish an apply's compact removals: erase the removed
        ///        carriers' index entries in one pass and finish every
        ///        hash-memory instance (LMV compaction + owner-less erasure).
        ///
        /// @param mb              The LB.
        /// @param dropSet         The apply's staged LMV drop set.
        /// @param removedCarriers The apply's staged carrier keys.
        /// @param gArena          Per-slot gen-scratch arena.
        /// @see `removeCompactExpansion`, `finishRuleRemovals`.
inline void CompactRuleProcessor::finishCompactRemovals(Memory& mb,
                                          const ColdHashSet<PodKeyStore<int64_t>>& dropSet,
                                          const ColdHashSet<PodKeyStore<int64_t>>& removedCarriers,
                                          ScratchArena& gArena) {
            if (removedCarriers.count() > 0) {
                mb.compactExpansions.eraseBlobIf([&](int64_t k) {
                    return removedCarriers.lookup(k) != 0;
                });
            }
            // The apply's deferred expanded-pair erases: one compaction pass
            // over the column, keeping the pairs an install revived.
            if (DeferredExpandedErase* const deferred = mb.deferredExpandedErase;
                deferred != nullptr && deferred->pending.count() > 0) {
                RT_SCOPE_HERE("FINISH_EXPANDED_ERASE");
                const ColdHashSet<PodKeyStore<int64_t>>& pending = deferred->pending;
                const ColdHashSet<PodKeyStore<int64_t>>& revived = deferred->revived;
                mb.expandedImplications.eraseIf([&](const LbStatePairKey& k) {
                    const int64_t packed = packInt32Pair(k.high, k.low);
                    return pending.lookup(packed) != 0 && revived.lookup(packed) == 0;
                });
                deferred->pending.resetToFresh();
                deferred->revived.resetToFresh();
            }
            gl::HashRuleInstaller::finishRuleRemovals(mb.overallHashMemory, dropSet, gArena);
            gl::HashRuleInstaller::finishRuleRemovals(mb.localHashMemory, dropSet, gArena);
            gl::HashRuleInstaller::finishRuleRemovals(mb.localHashMemoryDelta, dropSet, gArena);
            gl::HashRuleInstaller::finishRuleRemovals(mb.workingMemory, dropSet, gArena);
        }

/// @brief The record kinds of the carrier `(text, validity)`.
        ///
        /// @details
        /// An absent carrier (an unminted name, no index entry) holds nothing.
        /// An integration carrier holds exactly one integration-kind record,
        /// `Integration` or `IntegrationFlat` (asserted): the compact is
        /// expanded as an integration rule once per scope; `integrationFlat`
        /// reports a root goal's flat variant
        /// (D-574). Reads only: non-minting lookups, record peeks on the per-slot
        /// gen-scratch arena (rewound).
        ///
        /// @param mb         The LB.
        /// @param carrierKey The carrier's packed `(originalId, validityId)` key.
        /// @return The kinds; all `false` for an absent carrier.
        /// @see installIntegrationCompact, applyEquivalenceClassToCompactImplications,
        ///      expandCanonicalCompact.
inline CompactRuleProcessor::CarrierKinds CompactRuleProcessor::carrierKinds(Memory& mb, int64_t carrierKey) {
            CarrierKinds kinds;
            const int32_t id = mb.compactExpansions.lookup(carrierKey);
            if (id == 0) return kinds;
            const unsigned slot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            ScratchArena& gArena = genScratchArenas().forSlot(slot);
            for (int32_t j = 0; j < mb.compactExpansions.runLen(id); ++j) {
                const ArenaOffset mark = gArena.cursor();
                int32_t bl = 0;
                const char* bp = mb.compactExpansions.peekRecordBytes(id, j, bl, gArena);
                assert(bl == Codec<CompactExpansionRec>::kBytes && "carrierKinds: a carrier record of the wrong size");
                const RuleInstallKind kind = static_cast<RuleInstallKind>(static_cast<uint8_t>(bp[8]));
                gArena.popTo(mark);
                if (!isIntegrationInstallKind(kind)) {
                    kinds.statement = true;
                    continue;
                }
                assert(!kinds.integration && "carrierKinds: an integration carrier holds one integration record");
                kinds.integration = true;
                kinds.integrationFlat = kind == RuleInstallKind::IntegrationFlat;
            }
            return kinds;
        }

/// @brief `carrierKinds` of the carrier named by text and scope.
        ///
        /// @param mb       The LB.
        /// @param text     The compact text (`u_`-stripped).
        /// @param validity Its scope.
        /// @return The kinds; all `false` when either name is unminted.
        /// @see carrierKinds(Memory&, int64_t).
inline CompactRuleProcessor::CarrierKinds CompactRuleProcessor::carrierKinds(Memory& mb, StrSpan text, StrSpan validity) {
            const NameId oid = mb.nameMap.lookup(text);
            const NameId vid = mb.nameMap.lookup(validity);
            if (oid == 0 || vid == 0) return CarrierKinds{};
            return carrierKinds(mb, packStatementKey(oid, vid));
        }

/// @brief Install one filled integration compact at a scope: expand it,
        ///        install its rule through the integration route and record
        ///        the compact as the rule's carrier
        ///        (D-503).
        ///
        /// @details
        /// The compact is an entry of an entity's integration set whose owner
        /// tokens were filled with the instance's `u_`-prefixed arguments. It
        /// is expanded first (`IntegrationInstructionBuilder::
        /// expandIntegrationCompactScratch`, fresh witness names) and the
        /// expansion is returned through @p premises / @p head in every case.
        /// A compact already expanded at @p validity (its carrier entry
        /// exists) installs nothing. Otherwise,
        /// with history tracked, the compact (`u_`-stripped) takes the row
        /// @p tag citing @p deps and both forms of its rule take an
        /// `expansion` row citing the compact, each also on the mail-out;
        /// the rule is installed into the overall, local and delta instances
        /// with justification `integration`, and the carrier index
        /// records `(compact, validity) → (rule, validity)` with install kind
        /// `Integration`. The compact itself is never a statement and never
        /// mailed. Single-threaded seam only.
        ///
        /// @param mb         The LB.
        /// @param compact    The filled compact in `u_`-prefixed form.
        /// @param validity   The install scope (span stable across the call).
        /// @param tag        The compact's history tag.
        /// @param deps       The compact's history citations.
        /// @param depN       Their number.
        /// @param arena      String-tier scratch arena receiving the expansion.
        /// @param premises   Out: the expanded premises (`u_` form).
        /// @param premiseCap Capacity of @p premises.
        /// @param premiseN   Out: their number.
        /// @param head       Out: the expanded head (`u_` form).
        /// @param kind       `Integration` or `IntegrationFlat`, a root goal's
        ///                   flat variant whose install adds the near-miss
        ///                   demand variants and whose carrier record keeps the
        ///                   kind for its removal and its canonical
        ///                   re-installation
        ///                   (D-574).
        /// @return `true` when the rule was installed, `false` when the compact
        ///         was already expanded at @p validity.
        /// @invariant 0% heap.
        /// @see expandIntegrationCompactScratch, appendCompactExpansion,
        ///      removeCompactExpansion, expandCanonicalIntegrationCompact.
inline bool CompactRuleProcessor::installIntegrationCompact(Memory& mb, StrSpan compact, StrSpan validity,
                                       OriginTag tag, const OriginDep* deps, int32_t depN,
                                       ScratchArena& arena, StrSpan* premises, int32_t premiseCap,
                                       int32_t& premiseN, StrSpan& head, RuleInstallKind kind) {
            premiseN = integrationBuilder.expandIntegrationCompactScratch(arena, compact, mb,
                premises, premiseCap, head);
            return installExpandedIntegrationCompact(mb, compact, validity, tag, deps, depN, arena,
                premises, premiseN, head, kind);
        }

/// @brief The install half of `installIntegrationCompact`: install an
        ///        integration compact already expanded into @p premises /
        ///        @p head (D-503).
        ///
        /// @details
        /// For a caller whose history citation depends on the expansion (a
        /// disjunct rule cites its expanded head). A compact already expanded
        /// at @p validity installs nothing; otherwise the history rows, the
        /// three integration installs
        /// and the carrier record follow, exactly as `installIntegrationCompact`
        /// documents.
        ///
        /// @param mb        The LB.
        /// @param compact   The filled compact in `u_`-prefixed form.
        /// @param validity  The install scope (span stable across the call).
        /// @param tag       The compact's history tag.
        /// @param deps      The compact's history citations.
        /// @param depN      Their number.
        /// @param arena     String-tier scratch arena.
        /// @param premises  The expanded premises (`u_` form).
        /// @param premiseN  Their number.
        /// @param head      The expanded head (`u_` form).
        /// @param kind      `Integration` or `IntegrationFlat` (asserted one of
        ///                  the two), as for `installIntegrationCompact`.
        /// @return `true` when the rule was installed, `false` when the compact
        ///         was already expanded at @p validity.
        /// @invariant 0% heap.
        /// @see installIntegrationCompact.
inline bool CompactRuleProcessor::installExpandedIntegrationCompact(Memory& mb, StrSpan compact, StrSpan validity,
                                       OriginTag tag, const OriginDep* deps, int32_t depN,
                                       ScratchArena& arena, const StrSpan* premises, int32_t premiseN,
                                       StrSpan head, RuleInstallKind kind) {
            assert(isIntegrationInstallKind(kind)
                && "installExpandedIntegrationCompact: an integration compact installs as an integration kind");
            const ScratchString clean = gl::DefinitionExpansion::removeUPrefixScratch(arena, compact);
            if (carrierKinds(mb, StrSpan(clean), validity).integration) return false;
            ScratchString historyS;
            ScratchString hashS;
            integrationBuilder.buildIntegrationInstructionScratch(arena, premises, premiseN, head, historyS, hashS);
            if (parameters.trackHistory) {
                const int cap = parameters.compressor_mode
                    ? parameters.compressor_max_origins_per_expr
                    : parameters.max_origin_per_expr;
                addOriginEncoded(mb.exprOriginMap, mb.originInterner, StrSpan(clean), validity, tag, deps, depN, cap);
                mb.addMailOutOrigin(StrSpan(clean), validity, tag, deps, depN, cap);
                const OriginDep ruleDeps[1] = { { StrSpan(clean), validity } };
                addOriginEncoded(mb.exprOriginMap, mb.originInterner, StrSpan(historyS), validity,
                    OriginTag::expansion, ruleDeps, 1, cap);
                mb.addMailOutOrigin(StrSpan(historyS), validity, OriginTag::expansion, ruleDeps, 1, cap);
                addOriginEncoded(mb.exprOriginMap, mb.originInterner, StrSpan(hashS), validity,
                    OriginTag::expansion, ruleDeps, 1, cap);
                mb.addMailOutOrigin(StrSpan(hashS), validity, OriginTag::expansion, ruleDeps, 1, cap);
            }
            StrSpan unchangeables[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
            const int32_t unchangeableN = integrationBuilder.collectUnchangeables(head, premises, premiseN,
                unchangeables, ExecutionParameters::MAX_ADMISSION_REM_ARGS);
            HashMemory* const targets[3] = {
                &mb.overallHashMemory, &mb.localHashMemory, &mb.localHashMemoryDelta };
            for (HashMemory* target : targets) {
                hashRuleInstaller.addToHashMemory(premises, premiseN,
                    head,
                    unchangeables, unchangeableN,
                    mb,
                    *target,
                    nullptr, 0,
                    StrSpan(hashS),
                    0, 0, false, 0,
                    StrSpan("integration", 11),
                    false,
                    StrSpan(historyS),
                    validity, RuleIndexOp{}, nullptr,
                    kind == RuleInstallKind::IntegrationFlat);
            }
            appendCompactExpansion(mb, StrSpan(clean), validity, StrSpan(hashS), validity, kind);
            return true;
        }

/// @brief The `equality1` citations of a canonical compact K*: the
        ///        carrier K at its scope, then every fired pair as
        ///        `(=[from,to])` cited where its own origin row lives.
        ///
        /// @details
        /// The class scope first, else the nearest ancestor holding the
        /// equality's row (asserted found) — the citation walk of the
        /// `applyEquivalenceClass` commit. The equality texts and copied
        /// scope names ride @p strArena.
        ///
        /// @param mb       The LB.
        /// @param carrier  The non-canonical compact K.
        /// @param validity Its scope.
        /// @param pairs    The fired pairs.
        /// @param pairN    Their number (at most `MAX_ARITY`).
        /// @param strArena String-tier scratch arena.
        /// @param deps     Out: at least `1 + MAX_ARITY` entries.
        /// @return The number of citations written.
        /// @see expandCanonicalCompact, expandCanonicalIntegrationCompact.
inline int32_t CompactRuleProcessor::canonicalEqualityDeps(Memory& mb, StrSpan carrier, StrSpan validity,
                                      const CompactCanonPair* pairs, int32_t pairN,
                                      ScratchArena& strArena, OriginDep* deps) {
            assert(pairN <= ExecutionParameters::MAX_ARITY);
            int32_t depN = 0;
            deps[depN++] = OriginDep{ carrier, validity };
            for (int32_t p = 0; p < pairN; ++p) {
                const StrSpan from = mb.nameMap.decodeView(pairs[p].fromId);
                const StrSpan to = mb.nameMap.decodeView(pairs[p].toId);
                const int32_t eqLen = 3 + from.len + 1 + to.len + 2;
                char* eq = strArena.allocBytes(eqLen);
                int32_t at = 0;
                std::memcpy(eq + at, "(=[", 3); at += 3;
                std::memcpy(eq + at, from.ptr, static_cast<std::size_t>(from.len)); at += from.len;
                eq[at++] = ',';
                std::memcpy(eq + at, to.ptr, static_cast<std::size_t>(to.len)); at += to.len;
                eq[at++] = ']'; eq[at++] = ')';
                const StrSpan equality(eq, eqLen);
                const StrSpan classScopeView = mb.nameMap.decodeView(pairs[p].classVid);
                const ScratchString classScope = ScratchString::copyFrom(
                    strArena, classScopeView.ptr, classScopeView.len);
                const auto originRowAt = [&](StrSpan scope) {
                    int64_t pk = 0;
                    if (!lookupOriginKey(mb.originInterner, equality, scope, pk)) return false;
                    const int32_t oid = mb.exprOriginMap.lookup(pk);
                    return oid != 0 && mb.exprOriginMap.runLen(oid) > 0;
                };
                StrSpan citeScope = StrSpan(classScope);
                bool citeFound = originRowAt(citeScope);
                if (!citeFound) {
                    StrSpan classAncestors[ExecutionParameters::MAX_SCOPE_DEPTH];
                    const int32_t classAncestorCount = mb.nameMap.strictAncestorSpans(
                        StrSpan(classScope), classAncestors,
                        ExecutionParameters::MAX_SCOPE_DEPTH);
                    for (int32_t ai = classAncestorCount - 1; ai >= 0 && !citeFound; --ai) {
                        if (originRowAt(classAncestors[ai])) {
                            citeFound = true;
                            const ScratchString c = ScratchString::copyFrom(
                                strArena, classAncestors[ai].ptr, classAncestors[ai].len);
                            citeScope = StrSpan(c);
                        }
                    }
                }
                assert(citeFound
                    && "canonicalEqualityDeps: justifying equality has no origin row at the class scope or any of its ancestors");
                deps[depN++] = OriginDep{ equality, citeScope };
            }
            return depN;
        }

/// @brief Re-form an integration rule after a class merge: install the
        ///        canonical form K* of its carrier K
        ///        (D-503).
        ///
        /// @details
        /// The integration twin of `expandCanonicalCompact`, with no statement
        /// registration. Nothing happens when K* is already expanded at @p
        /// validity. Otherwise K* (`u_`-prefixed) goes through
        /// `installIntegrationCompact` with the history row `K* ← equality1 ← K + (=[from,to])…`
        /// (`canonicalEqualityDeps`); its rule takes `expansion ← K*`.
        ///
        /// @param mb       The LB.
        /// @param kstar    The canonical form (`u_`-stripped, caller-stable).
        /// @param carrier  The non-canonical compact K.
        /// @param validity The scope.
        /// @param pairs    The fired pairs.
        /// @param pairN    Their number.
        /// @param kind     K's install kind, `Integration` or `IntegrationFlat`:
        ///                 K* is installed with the same kind
        ///                 (D-574).
        /// @see installCanonicalCompacts, installIntegrationCompact.
inline void CompactRuleProcessor::expandCanonicalIntegrationCompact(Memory& mb, StrSpan kstar, StrSpan carrier,
                                               StrSpan validity,
                                               const CompactCanonPair* pairs, int32_t pairN,
                                               RuleInstallKind kind) {
            if (carrierKinds(mb, kstar, validity).integration) return;
            const unsigned strSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : scratchArenas().slotCount() - 1;
            ScratchArena& strArena = scratchArenas().forSlot(strSlot);
            ScratchScope strScope(strArena);
            OriginDep deps[1 + ExecutionParameters::MAX_ARITY];
            const int32_t depN = canonicalEqualityDeps(mb, carrier, validity, pairs, pairN, strArena, deps);
            const ScratchString kstarU = gl::DefinitionExpansion::prefixArgumentsWithUScratch(strArena, kstar);
            StrSpan premises[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
            int32_t premiseN = 0;
            StrSpan head;
            const bool installed = installIntegrationCompact(mb, StrSpan(kstarU), validity, OriginTag::equality1,
                deps, depN, strArena, premises, ExecutionParameters::MAX_INSTRUCTION_ELEMENTS, premiseN, head,
                kind);
            assert(installed && "expandCanonicalIntegrationCompact: K* is not expanded here, yet nothing was installed");
            (void)installed;
        }





/// @brief The canonical form of a compact at a scope — every argument
        ///        replaced by the canonical representative of the class that
        ///        holds it in the maintained complete view at @p validity.
        ///
        /// @details
        /// The maintained canonical bucket includes the target's own classes
        /// and every visible ancestor class, with overlaps already consolidated.
        /// `chooseCanonicalId` selects at the target scope: normal > `int_` >
        /// `it_`, lex-min in the tier, weak target-scope members skipped (a
        /// class with no strong member maps
        /// nothing). The rewrite is `replaceKeysScratch` over the fired
        /// pairs (token-boundary, order-free); the pairs are returned for
        /// the `equality1` citation, and every fired pair's admission levels
        /// (D-280, `forEachLevel(packEqPairKey(from, to))`) are unioned into
        /// @p levelRun. Reads only: non-minting NameMap lookups, class-blob
        /// peeks, the name-kind memo. Zero heap; the peeks and the rewritten
        /// text ride the two scratch arenas the caller frames.
        ///
        /// @param mb        The LB.
        /// @param text      The compact text.
        /// @param validity  Its scope.
        /// @param strArena  String-tier scratch for the rewritten text.
        /// @param gArena    Gen-scratch for the class-blob peeks.
        /// @param kstarOut  Out: the rewritten text (only when `true` is returned).
        /// @param pairs     Out: the fired pairs (`MAX_ARITY` cap).
        /// @param pairN     Out: their number.
        /// @param levelRun  In/out: ascending-unique level run the pairs' levels join.
        /// @param levelN    In/out: its length.
        /// @param levelCap  Its capacity.
        /// @return `true` when the text changed (the compact is not canonical).
        /// @see `chooseCanonicalId`, `installExpandedImplication`,
        ///      `applyEquivalenceClassToCompactImplications`.
inline bool CompactRuleProcessor::canonicalizeCompact(Memory& mb, StrSpan text, StrSpan validity,
                                 ScratchArena& strArena, ScratchArena& gArena,
                                 ScratchString& kstarOut,
                                 CompactCanonPair* pairs, int32_t& pairN,
                                 int* levelRun, int32_t& levelN, int32_t levelCap) {
            pairN = 0;
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t argN = getArgsSpans(text, args, ExecutionParameters::MAX_ARITY);
            if (argN == 0) return false;
            const StrSpan scopes[1] = {validity};
            const int32_t scopeN = 1;
            const ArenaOffset mark = gArena.cursor();
            StrReplacement subst[ExecutionParameters::MAX_ARITY];
            int32_t substN = 0;
            for (int32_t a = 0; a < argN; ++a) {
                const NameId id = mb.nameMap.lookup(args[a]);
                if (id == 0) continue;                       // not a name: never a member
                bool seen = false;
                for (int32_t p = 0; p < pairN && !seen; ++p) seen = (pairs[p].fromId == id);
                if (seen) continue;                          // a repeated argument
                // The complete target-scope class holding the argument.
                const char* bestP = nullptr;
                int32_t bestLen = 0;
                int32_t bestDepth = -1;
                int32_t bestScope = -1;
                NameId bestVid = 0;
                for (int32_t s = 0; s < scopeN; ++s) {
                    const NameId vid = mb.nameMap.lookup(scopes[s]);
                    if (vid == 0) continue;
                    const int32_t bucket = mb.canonicalEquivalenceClassesMap.lookup(vid);
                    if (bucket == 0) continue;
                    const int32_t depth = mb.nameMap.ancLen(vid);
                    if (depth <= bestDepth) continue;
                    const int32_t rl = mb.canonicalEquivalenceClassesMap.runLen(bucket);
                    for (int32_t j = 0; j < rl; ++j) {
                        int32_t blen = 0;
                        const char* bp = mb.canonicalEquivalenceClassesMap.peekRecordBytes(
                            bucket, j, blen, gArena);
                        const EquivalenceClassView cls{ bp, blen };
                        if (!classHasMember(cls, id)) continue;
                        bestP = bp; bestLen = blen; bestDepth = depth;
                        bestScope = s; bestVid = vid;
                        break;                               // one class per scope holds it
                    }
                }
                if (bestP == nullptr) continue;
                const EquivalenceClassView cls{ bestP, bestLen };
                const NameId canon = canonicalizer.chooseCanonicalId(cls, mb, scopes[bestScope]);
                if (canon == 0 || canon == id) continue;
                assert(pairN < ExecutionParameters::MAX_ARITY
                    && "canonicalizeCompact: fired pairs exceed MAX_ARITY");
                pairs[pairN] = CompactCanonPair{ id, canon, bestVid };
                ++pairN;
                subst[substN].key = args[a];
                subst[substN].value = mb.nameMap.decodeView(canon);
                ++substN;
                cls.forEachLevel(packEqPairKey(id, canon), [&](int lv) {
                    levelN = insertLevelSorted(levelRun, levelN, lv, levelCap);
                });
            }
            if (pairN == 0) {
                gArena.popTo(mark);
                return false;
            }
            kstarOut = replaceKeysScratch(strArena, text, subst, substN);
            gArena.popTo(mark);
            assert(!equalSpans(StrSpan(kstarOut), text)
                && "canonicalizeCompact: fired pairs left the text unchanged");
            return true;
        }

/// @brief Whether a compact's rules are installed here — the carrier
        ///        index has an entry for `(text, validity)`.
        ///
        /// @param mb       The LB.
        /// @param text     The compact text.
        /// @param validity Its scope.
        /// @return `true` when the carrier has recorded expansions at this LB.
inline bool CompactRuleProcessor::compactExpandedHere(const Memory& mb, StrSpan text, StrSpan validity) {
            const NameId oid = mb.nameMap.lookup(text);
            const NameId vid = mb.nameMap.lookup(validity);
            if (oid == 0 || vid == 0) return false;
            return mb.compactExpansions.lookup(packStatementKey(oid, vid)) != 0;
        }

/// @brief Expand the canonical form K* of a non-canonical compact K
        ///        into hash memory and register it — the "only canonical
        ///        enters hash memory" step both sites share.
        ///
        /// @details
        /// Nothing happens when K* is already expanded here or known at a
        /// strict ancestor (its rules are visible from there). Otherwise K*
        /// is disintegrated (`disintegrateExpr2`, the compact's own
        /// expansion — `trackExpansionHistory` records `rule ← expansion ←
        /// K*`), every produced implication is installed through
        /// `installExpandedImplication` (K* is its own canonical carrier, so
        /// the gate passes), and K* is registered as a statement at @p
        /// validity with the `equality1` history row `K* ← K + (=[from,to])…`
        /// (each equality cited at the scope where its own origin row lives
        /// — the class scope or the nearest ancestor holding it) and the
        /// level run @p levelRun (the deposit's levels ∪ the fired pairs'
        /// levels, D-280); a K* already known at this scope keeps its row and
        /// gains the history row only. Zero heap; the disintegration products
        /// ride the per-slot gen-scratch arena.
        ///
        /// @param mb        The LB.
        /// @param kstar     The canonical form (caller-stable text).
        /// @param carrier   The non-canonical compact K it came from.
        /// @param validity  The scope (K's, K*'s and the rules').
        /// @param pairs     The fired pairs.
        /// @param pairN     Their number.
        /// @param levelRun  The level run for K* and its rules.
        /// @param levelN    Its length.
        /// @param status    The deposit status (decides local / external).
        /// @param iteration The deposit iteration.
        /// @see `canonicalizeCompact`, `installExpandedImplication`.
inline void CompactRuleProcessor::expandCanonicalCompact(Memory& mb, StrSpan kstar, StrSpan carrier,
                                    StrSpan validity,
                                    const CompactCanonPair* pairs, int32_t pairN,
                                    const int* levelRun, int32_t levelN,
                                    int status, int iteration) {
            // Only the statement role counts: an integration record of the
            // same text leaves the statement's rules to be expanded.
            if (carrierKinds(mb, kstar, validity).statement) return;
            const NameId kstarId = mb.nameMap.encode(kstar);
            const NameId vid = mb.nameMap.encode(validity);
            if (mb.phase3IgnoresScope(vid)) return;
            if (gl::StatementKnowledge::ancestorKnown(mb, kstarId, vid, /*includeSelf=*/false)) return;

            const unsigned strSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : scratchArenas().slotCount() - 1;
            ScratchArena& strArena = scratchArenas().forSlot(strSlot);
            ScratchScope strScope(strArena);
            const unsigned genSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            ScratchArena& gArena = genScratchArenas().forSlot(genSlot);
            const ArenaOffset gMark = gArena.cursor();

            OriginDep deps[1 + ExecutionParameters::MAX_ARITY];
            int32_t depN = 0;
            {
            RT_SCOPE_HERE("CANON_EXPAND_DEPS");
            depN = canonicalEqualityDeps(mb, carrier, validity, pairs, pairN, strArena, deps);
            } // RT_SCOPE CANON_EXPAND_DEPS

            // K*'s own disintegration: the compact's rule(s).
            DisintProducts out(&gArena);
            bool full = false;
            {
                RT_SCOPE_HERE("CANON_EXPAND_DISINT");
                full = disintegrationProcessor.disintegrateExpr2(kstar, mb, iteration, status == 0,
                    validity, out, /*trackHistoryLocal=*/true,
                    /*allowOrDisintegration=*/false, levelRun, levelN,
                    /*allowOrProbe=*/false);
            }
            assert(full && "expandCanonicalCompact: a compact did not fully disintegrate");
            // A compact implication yields rules and no witness, so no eat of
            // the induction hypothesis opens a scope here
            // (D-575).
            assert(out.hypothesisDemands.count() == 0
                && "expandCanonicalCompact: a compact's disintegration opened an induction scope");
            const bool local = (status == 0 || status == 1 || status == 5);
            // Attached implications are ordinary statement products of K*.
            // Disintegration already recorded their checked derivations. Keep
            // every product registered alongside the rule it owns, including
            // on the pair-free re-expansion path.
            out.statements.forEachSorted([&](StrSpan product, StrSpan scope) {
                if (equalSpans(product, kstar) && equalSpans(scope, validity)) return;
                assert(gl::CompactExpressionSyntax::isCompactImplicationSpan(product) && equalSpans(scope, validity));
                DirtyState productDirty = DirtyState::Clean;
                PagedVector<IntEncodedExpr> added(&gArena, &productDirty);
                statementMemory.addStatement(product, mb, local, levelRun, levelN,
                    TransientOrigin{}, scope, added);
            });
            {
                RT_SCOPE_HERE("CANON_EXPAND_INSTALLS");
                out.implications.forEachSortedWithSource(
                    [&](StrSpan impStr, StrSpan impValidity, StrSpan impCarrier) {
                        this->installExpandedImplication(mb, impStr, impValidity, impCarrier,
                            status, iteration, levelRun, levelN, kstar, validity);
                    });
            }

            // Register K* (with its equality1 row) unless this scope knows it.
            // A pair-free call is the as-is expansion of a registered canonical
            // compact whose rules were missing (a weak-member shift made it
            // canonical after its registration): K* == K, nothing to register,
            // no history row to add.
            if (pairN == 0) {
                assert(equalSpans(kstar, carrier)
                    && "expandCanonicalCompact: no fired pair but a rewritten text");
                gArena.popTo(gMark);
                return;
            }
            RT_SCOPE_HERE("CANON_EXPAND_REGISTER");
            if (lookupStatementFlags(mb.intKnownStatements, mb.nameMap, kstar, validity) == nullptr) {
                const TransientOrigin origin{ true, OriginTag::equality1, deps, depN };
                DirtyState nsDirty = DirtyState::Clean;
                PagedVector<IntEncodedExpr> newStatements(&gArena, &nsDirty);
                statementMemory.addStatement(kstar, mb, local, levelRun, levelN, origin, validity,
                                   newStatements);
            } else if (parameters.trackHistory) {
                addOriginEncoded(mb.exprOriginMap, mb.originInterner, kstar, validity,
                    OriginTag::equality1, deps, depN,
                    (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr
                                                : parameters.max_origin_per_expr));
            }
            gArena.popTo(gMark);
        }

/// @brief The class-side reconcile of the canonical gate — the
        ///        `applyEquiClasses` hook over the registered compact
        ///        implications: a compact that a class made non-canonical
        ///        loses its rules and its canonical form is expanded; a
        ///        registered canonical compact without rules is expanded.
        ///
        /// @details
        /// Mirrors the rejected-map hooks' walk (D-307): the per-apply
        /// snapshot @p buckets is built at the first use from the compact
        /// rows of `intEncodedStatements` (`isCompactImplicationSpan`, keys in
        /// @p compactKeys) with `(validity, argument) → keys` postings decoded
        /// through the NameMap; the candidates are the compacts at the class
        /// validity OR a strict descendant of it (a class is visible below its
        /// scope) that name a non-canonical member. Per candidate: a carrier
        /// already removed in this apply is skipped; one whose text is
        /// unchanged under the class and whose rules are installed is done;
        /// otherwise the full canonical form under every visible class
        /// (`canonicalizeCompact`) decides — changed: its rules leave
        /// (`removeCompactExpansion`, staged) and its logical carrier key queues
        /// for replacement — an integration carrier in
        /// @p pendingIntegrationCompacts (D-503),
        /// any other in @p pendingCompacts. Unchanged but unexpanded carriers queue as well.
        /// The caller finishes all obsolete removals with `finishCompactRemovals`
        /// before `installCanonicalCompacts` exposes replacements. One batched
        /// compaction covers the complete consumer pass.
        /// The decoded candidate text is copied to the string arena before the
        /// reconcile mints the NameMap (I-3). Gated on `!skip_eq_classes` by
        /// the caller.
        ///
        /// @tparam ClassT `EquivalenceClass` or `EquivalenceClassView`.
        /// @param clss            The class (delta or registry).
        /// @param memoryBlock     The LB.
        /// @param validityName    The class scope.
        /// @param buckets         The per-apply compact index (built here on first use).
        /// @param compactKeys     The per-apply snapshot of compact statement keys.
        /// @param visited         The per-removal visited-entry set.
        /// @param dropSet         The apply's staged LMV drop set.
        /// @param removedCarriers The apply's staged carrier keys.
        /// @param pendingCompacts Logical carrier keys to install after obsolete removals finish.
        /// @param pendingIntegrationCompacts Integration carrier keys, each with
        ///        its install kind (`Integration` or `IntegrationFlat`, as a
        ///        byte), installed after obsolete removals finish.
        /// @param selected Optional complete-class dependency selection.
        /// @return The number of hits (removals + expansions).
        /// @see `installExpandedImplication` (the door-side gate),
        ///      `canonicalizeCompact`, `removeCompactExpansion`,
        ///      `expandCanonicalCompact`, `finishCompactRemovals`.
template <class ClassT>
inline int32_t CompactRuleProcessor::applyEquivalenceClassToCompactImplications(
                const ClassT& clss, Memory& memoryBlock, StrSpan validityName,
                RejectedValidityBuckets& buckets,
                PagedVector<int64_t>& compactKeys,
                ColdHashSet<BytesKeyStore>& visited,
                ColdHashSet<PodKeyStore<int64_t>>& dropSet,
                ColdHashSet<PodKeyStore<int64_t>>& removedCarriers,
                TypedColdSet<int64_t>& pendingCompacts,
                TypedColdMap<int64_t, uint8_t>& pendingIntegrationCompacts,
            const TypedColdSet<int64_t>* selected) {
            if (selected == nullptr && memoryBlock.intEncodedStatements.empty()) return 0;
            const NameId canonId = canonicalizer.chooseCanonicalId(clss, memoryBlock, validityName);
            if (canonId == 0) return 0;
            const unsigned slot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            ScratchArena& gArena = genScratchArenas().forSlot(slot);
            ScratchArena& strArena = scratchArenas().forSlot(slot);
            DirtyState dirty = DirtyState::Clean;

            if (selected == nullptr && !buckets.built()) {
                const std::size_t n = memoryBlock.intEncodedStatements.size();
                for (std::size_t s = 0; s < n; ++s) {
                    const IntEncodedExpr& row = memoryBlock.intEncodedStatements[s];
                    if (!gl::CompactExpressionSyntax::isCompactImplicationSpan(memoryBlock.nameMap.decodeView(row.originalId)))
                        continue;
                    compactKeys.push_back(packStatementKey(row.originalId, row.validityId));
                }
                buckets.ensureBuiltFrom(compactKeys.size(),
                    [&](int32_t id) { return compactKeys[id - 1]; },
                    [&](NameId orig) { return memoryBlock.nameMap.decodeView(orig); },
                    memoryBlock.nameMap);
            }
            if (selected == nullptr && buckets.countAtBuild() == 0) return 0;

            // The class substitution (every non-canonical member -> the canonical).
            const StrSpan canonSpan = memoryBlock.nameMap.decodeView(canonId);
            const int32_t memberN = clss.memberCount();
            const ArenaOffset mark = gArena.cursor();
            StrReplacement* subPairs = reinterpret_cast<StrReplacement*>(gArena.resolve(
                gArena.alloc((memberN > 0 ? memberN : 1) * static_cast<int32_t>(sizeof(StrReplacement)),
                             static_cast<int32_t>(alignof(StrReplacement)))));
            int32_t subCount = 0;
            for (int32_t mi = 0; mi < memberN; ++mi) {
                const NameId mid = clss.memberId(mi);
                if (mid == canonId) continue;
                subPairs[subCount].key = memoryBlock.nameMap.decodeView(mid);
                subPairs[subCount].value = canonSpan;
                ++subCount;
            }
            if (subCount == 0) { gArena.popTo(mark); return 0; }
            const NameId classVid = memoryBlock.nameMap.lookup(validityName);
            assert(classVid != 0 && "compact hook: a class validity is always a minted name");

            // Candidates belong to this exact target scope and name a member
            // displaced by its complete class representative.
            TypedColdSet<int64_t> candidateKeys(&gArena, &dirty);
            if (selected != nullptr) {
                for (int32_t i = 1; i <= selected->count(); ++i) candidateKeys.mint(selected->keyAt(i));
            } else {
                for (int32_t mi = 0; mi < memberN; ++mi) {
                    const NameId member = clss.memberId(mi);
                    for (int32_t j = buckets.postBegin(classVid, member), end = buckets.postEnd(classVid, member); j < end; ++j)
                        candidateKeys.mint(buckets.pkAt(buckets.postAt(j)));
                }
            }
            const auto less = [&](int32_t a, int32_t b) {
                return compareSpans(memoryBlock.nameMap.decodeView(Codec<StatementKey>::decode(candidateKeys.keyAt(a + 1)).orig),
                    memoryBlock.nameMap.decodeView(Codec<StatementKey>::decode(candidateKeys.keyAt(b + 1)).orig)) < 0;
            };
            ChunkSortedOrdinals<decltype(less)> order(gArena, candidateKeys.count(), 0, less);
            int32_t hits = 0;
            for (int32_t c = 0; c < candidateKeys.count(); ++c) {
                const int64_t pk = candidateKeys.keyAt(order.next() + 1);
                if (removedCarriers.lookup(pk) != 0) continue;   // removed earlier this apply
                const StatementKey sk = Codec<StatementKey>::decode(pk);
                ScratchScope candScope(strArena);
                // Owned copies: the reconcile below mints the NameMap (I-3).
                const StrSpan textView = memoryBlock.nameMap.decodeView(sk.orig);
                const ScratchString text = ScratchString::copyFrom(strArena, textView.ptr, textView.len);
                const StrSpan vView = memoryBlock.nameMap.decodeView(sk.validity);
                const ScratchString vspan = ScratchString::copyFrom(strArena, vView.ptr, vView.len);
                assert(gl::CompactExpressionSyntax::isCompactImplicationSpan(StrSpan(text))
                    && "compact hook: the snapshot holds a non-compact row");
                const bool changedUnderClass = !equalSpans(
                    StrSpan(replaceKeysScratch(strArena, StrSpan(text), subPairs, subCount)),
                    StrSpan(text));
                // Two roles per carrier text: a statement compact (expanded
                // through the door) and an integration compact (its rule
                // installed by integration, never a statement). A carrier
                // without an integration record is a statement compact, as
                // is one whose text is also a statement here.
                const CarrierKinds kinds = carrierKinds(memoryBlock, pk);
                const bool expandedHere = kinds.statement || kinds.integration;
                const bool statementCompact = !kinds.integration || kinds.statement
                    || lookupStatementFlags(memoryBlock.intKnownStatements, memoryBlock.nameMap,
                                            StrSpan(text), StrSpan(vspan)) != nullptr;
                if (!changedUnderClass && (!statementCompact || kinds.statement)) continue;   // canonical here, installed

                // The statement's own levels, then the fired pairs' levels.
                int levelRun[256];
                int32_t levelN = 0;
                {
                    const int32_t lvId = memoryBlock.intStatementLevelsMap.lookup(pk);
                    if (lvId != 0) {
                        levelN = coldIntRunAt(memoryBlock.intStatementLevelsMap, lvId, levelRun, 256);
                        if (levelN == 1 && levelRun[0] == -1) levelN = 0;   // the non-derived tier
                    }
                }
                ScratchString kstar;
                CompactCanonPair pairs[ExecutionParameters::MAX_ARITY];
                int32_t pairN = 0;
                bool changed = false;
                {
                    RT_SCOPE_HERE("HOOK_CANONICALIZE");
                    changed = canonicalizeCompact(memoryBlock, StrSpan(text), StrSpan(vspan),
                        strArena, gArena, kstar, pairs, pairN, levelRun, levelN, 256);
                }
                if (changed) {
                    if (expandedHere) {
                        RT_SCOPE_HERE("HOOK_REMOVE_EXPANSION");
                        removeCompactExpansion(memoryBlock, pk, visited, dropSet, removedCarriers);
                        ++hits;
                    }
                    if (kinds.integration) {
                        // K* keeps K's install kind: a flat root rule stays flat.
                        pendingIntegrationCompacts.upsert(pk, static_cast<uint8_t>(kinds.integrationFlat
                            ? RuleInstallKind::IntegrationFlat : RuleInstallKind::Integration));
                    }
                    if (statementCompact) pendingCompacts.mint(pk);
                    ++hits;
                } else if (statementCompact && !kinds.statement) {
                    pendingCompacts.mint(pk);
                    ++hits;
                }
            }
            gArena.popTo(mark);
            return hits;
        }

/// @brief Install canonical compact replacements after obsolete rules leave.
        /// @details Logical carrier keys are sorted by decoded expression and scope.
        /// Each replacement uses the current complete class view, original carrier
        /// levels and equality evidence. No vector compaction occurs during installs.
        /// Integration carriers follow in the same order, each re-formed by
        /// `expandCanonicalIntegrationCompact` with its recorded install kind.
        /// @param mb Owning block with completed rule removals.
        /// @param pending Coalesced original carrier keys, emptied on completion.
        /// @param pendingIntegration Integration carrier keys with their install
        ///        kinds, emptied on completion.
        /// @return Nothing.
        /// @invariant All hash memories have finished owner-less erasure before entry.
inline void CompactRuleProcessor::installCanonicalCompacts(Memory& mb,TypedColdSet<int64_t>& pending,
                                     TypedColdMap<int64_t, uint8_t>& pendingIntegration) {
            if (!pending.empty()) {
                assert(!mb.overallHashMemory.ownerlessPending && !mb.localHashMemory.ownerlessPending
                    && !mb.localHashMemoryDelta.ownerlessPending && !mb.workingMemory.ownerlessPending);
                const unsigned slot=gl::WorkerContext::coreId()>=0 ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount()-1;
                ScratchArena& arena=genScratchArenas().forSlot(slot);
                ScratchArena& strings=scratchArenas().forSlot(slot);
                ScratchScope frame(arena),textFrame(strings);
                const auto less=[&](int32_t a,int32_t b) {
                    const auto x=Codec<StatementKey>::decode(pending.keyAt(a+1));
                    const auto y=Codec<StatementKey>::decode(pending.keyAt(b+1));
                    const int c=compareSpans(mb.nameMap.decodeView(x.orig),mb.nameMap.decodeView(y.orig));
                    return c!=0 ? c<0 : compareSpans(mb.nameMap.decodeView(x.validity),mb.nameMap.decodeView(y.validity))<0;
                };
                ChunkSortedOrdinals<decltype(less)> order(arena,pending.count(),0,less);
                for(int32_t i=0;i<pending.count();++i) {
                    ScratchScope rowFrame(strings);
                    const int64_t packed=pending.keyAt(order.next()+1);
                    const auto key=Codec<StatementKey>::decode(packed);
                    if(mb.phase3IgnoresScope(key.validity)) continue;
                    const StrSpan original=mb.nameMap.decodeView(key.orig);
                    const ScratchString carrier=ScratchString::copyFrom(strings,original.ptr,original.len);
                    const StrSpan scopeView=mb.nameMap.decodeView(key.validity);
                    const ScratchString scope=ScratchString::copyFrom(strings,scopeView.ptr,scopeView.len);
                    int levels[256],levelCount=0;
                    const int32_t levelId=mb.intStatementLevelsMap.lookup(packed);
                    if(levelId!=0) {
                        levelCount=coldIntRunAt(mb.intStatementLevelsMap,levelId,levels,256);
                        if(levelCount==1 && levels[0]==-1) levelCount=0;
                    }
                    CompactCanonPair pairs[ExecutionParameters::MAX_ARITY];
                    int32_t pairCount=0; ScratchString canonical;
                    const bool changed=canonicalizeCompact(mb,StrSpan(carrier),StrSpan(scope),strings,arena,
                        canonical,pairs,pairCount,levels,levelCount,256);
                    RT_SCOPE_HERE("HOOK_EXPAND_CANONICAL");
                    expandCanonicalCompact(mb,changed ? StrSpan(canonical) : StrSpan(carrier),StrSpan(carrier),StrSpan(scope),
                        pairs,pairCount,levels,levelCount,1,-1);
                }
                pending.resetToFresh();
            }
            if (pendingIntegration.empty()) return;
            assert(!mb.overallHashMemory.ownerlessPending && !mb.localHashMemory.ownerlessPending
                && !mb.localHashMemoryDelta.ownerlessPending && !mb.workingMemory.ownerlessPending);
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
            ScratchArena& arena = genScratchArenas().forSlot(slot);
            ScratchArena& strings = scratchArenas().forSlot(slot);
            ScratchScope frame(arena), textFrame(strings);
            const auto less = [&](int32_t a, int32_t b) {
                const auto x = Codec<StatementKey>::decode(pendingIntegration.keyAt(a + 1));
                const auto y = Codec<StatementKey>::decode(pendingIntegration.keyAt(b + 1));
                const int c = compareSpans(mb.nameMap.decodeView(x.orig), mb.nameMap.decodeView(y.orig));
                return c != 0 ? c < 0 : compareSpans(mb.nameMap.decodeView(x.validity), mb.nameMap.decodeView(y.validity)) < 0;
            };
            ChunkSortedOrdinals<decltype(less)> order(arena, pendingIntegration.count(), 0, less);
            for (int32_t i = 0; i < pendingIntegration.count(); ++i) {
                ScratchScope rowFrame(strings);
                const int32_t id = order.next() + 1;
                const int64_t packed = pendingIntegration.keyAt(id);
                const auto key = Codec<StatementKey>::decode(packed);
                if (mb.phase3IgnoresScope(key.validity)) continue;
                const StrSpan original = mb.nameMap.decodeView(key.orig);
                const ScratchString carrier = ScratchString::copyFrom(strings, original.ptr, original.len);
                const StrSpan scopeView = mb.nameMap.decodeView(key.validity);
                const ScratchString scope = ScratchString::copyFrom(strings, scopeView.ptr, scopeView.len);
                int levels[256], levelCount = 0;
                CompactCanonPair pairs[ExecutionParameters::MAX_ARITY];
                int32_t pairCount = 0; ScratchString canonical;
                const bool changed = canonicalizeCompact(mb, StrSpan(carrier), StrSpan(scope), strings, arena,
                    canonical, pairs, pairCount, levels, levelCount, 256);
                assert(changed && "installCanonicalCompacts: an integration carrier queued while canonical");
                (void)changed;
                RT_SCOPE_HERE("HOOK_EXPAND_CANONICAL_INTEGRATION");
                const RuleInstallKind kind = static_cast<RuleInstallKind>(pendingIntegration.valueAt(id));
                assert(isIntegrationInstallKind(kind)
                    && "installCanonicalCompacts: an integration carrier queued with another kind");
                expandCanonicalIntegrationCompact(mb, StrSpan(canonical), StrSpan(carrier), StrSpan(scope),
                    pairs, pairCount, kind);
            }
            pendingIntegration.resetToFresh();
        }

/// @brief Install one expanded implication into the instances its
        ///        status selects, record it in the two indexes — after the
        ///        canonical gate: a rule whose carrier is a non-canonical
        ///        compact is NOT installed; its canonical form is expanded
        ///        instead (once).
        ///
        /// @details
        /// The install door's per-implication body. Gate first: a compact
        /// carrier (`isCompactImplicationSpan`) is canonicalized at the
        /// rule's scope (`canonicalizeCompact`); when it changes, the rule
        /// of the non-canonical K is skipped and `expandCanonicalCompact`
        /// installs K*'s rules (if hash memory lacks them) — "only the
        /// canonical form of a compact enters hash memory". Otherwise the
        /// rule is installed as before: `overallHashMemory` always,
        /// `workingMemory` for an external-mail carrier (status 3), else
        /// `localHashMemory` + `localHashMemoryDelta` (D-176); the D-274
        /// `expansion` row for a status-3 recovery; the `expandedImplications`
        /// pair (the dump's source) and the carrier index record. Skipped
        /// entirely under `skip_eq_classes` (the incubator batches, no
        /// classes) — the gate, not the install.
        ///
        /// @param mb                 The LB.
        /// @param impStr             The expanded implication text (channel span, stable).
        /// @param impValidity        Its scope.
        /// @param carrier            The statement whose disintegration produced it.
        /// @param status             The deposit status.
        /// @param iteration          The deposit iteration.
        /// @param involvedLevels     The deposit's level run.
        /// @param involvedLevelCount Its length.
        /// @param doorExpr           The deposited expression; an inherited OR's rules cite their compact carrier instead.
        /// @param doorValidity       Its scope.
        /// @see `addExprToMemoryBlock`, `expandCanonicalCompact`, `appendCompactExpansion`.
inline void CompactRuleProcessor::installExpandedImplication(Memory& mb, StrSpan impStr, StrSpan impValidity,
                                        StrSpan carrier, int status, int iteration,
                                        const int* involvedLevels, int32_t involvedLevelCount,
                                        StrSpan doorExpr, StrSpan doorValidity) {
            assert(carrier.len > 0
                && "installExpandedImplication: an implication without its carrier statement");
            if (!parameters.skip_eq_classes && gl::CompactExpressionSyntax::isCompactImplicationSpan(carrier)) {
                RT_SCOPE_HERE("INSTALL_CANON_GATE");
                const unsigned strSlot = (gl::WorkerContext::coreId() >= 0)
                    ? static_cast<unsigned>(gl::WorkerContext::coreId())
                    : scratchArenas().slotCount() - 1;
                ScratchArena& strArena = scratchArenas().forSlot(strSlot);
                ScratchScope strScope(strArena);
                const unsigned genSlot = (gl::WorkerContext::coreId() >= 0)
                    ? static_cast<unsigned>(gl::WorkerContext::coreId())
                    : genScratchArenas().slotCount() - 1;
                ScratchArena& gArena = genScratchArenas().forSlot(genSlot);
                ScratchString kstar;
                CompactCanonPair pairs[ExecutionParameters::MAX_ARITY];
                int32_t pairN = 0;
                int levelRun[256];
                int32_t levelN = 0;
                for (int32_t i = 0; i < involvedLevelCount; ++i)
                    levelN = insertLevelSorted(levelRun, levelN, involvedLevels[i], 256);
                if (canonicalizeCompact(mb, carrier, impValidity, strArena, gArena,
                                        kstar, pairs, pairN, levelRun, levelN, 256)) {
                    // K is not canonical: its rule stays out of hash memory;
                    // K* (the canonical form) gets expanded once. K itself
                    // stays a registered statement (hash fodder) — the door's
                    // statement loop registers it.
                    {
                    RT_SCOPE_HERE("INSTALL_CANON_EXPAND");
                    expandCanonicalCompact(mb, StrSpan(kstar), carrier, impValidity,
                                           pairs, pairN, levelRun, levelN,
                                           status, iteration);
                    }
                    return;
                }
            }

            // A demand-built rule (an anchor operator's totality rule) is not
            // installed: the block records that it holds the rule, and an
            // instance is built when a demand for its witness appears
            // (RejectedSourceReplay::buildDemandedInstances,
            // D-596). Its history row and
            // its installed-rule index entry are written as for any rule.
            const bool demandBuilt = rejectedSourceReplay.deferDemandBuiltRule(mb, impStr, carrier);
            if (!demandBuilt) {
            const unsigned kvSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : scratchArenas().slotCount() - 1;
            ScratchArena& kvArena = scratchArenas().forSlot(kvSlot);
            ScratchScope kvScope(kvArena);
            StrSpan remKeyRun[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
            int32_t remKeyRunN = 0;
            StrSpan chainRun[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
            int32_t chainRunN = 0;
            StrSpan headSpan;
            {
            RT_SCOPE_HERE("INSTALL_DECOMPOSE");
            hashRuleInstaller.decomposeImplicationForInstall(impStr, kvArena,
                chainRun, chainRunN, headSpan, remKeyRun, remKeyRunN);
            }

            // Distribute the rule by status (ASIC 0.1 reshuffle): overall
            // always; an external-mail absorb (status 3) is not a local
            // derivation and goes to the per-burst workingMemory instead of
            // the local pair (D-176) — as the second target of the SAME
            // build (one copy / permutation / key computation, every record
            // staged to both instances).
            {
            RT_SCOPE_HERE("INSTALL_HASHMEM_OVERALL");
            hashRuleInstaller.addToHashMemory(chainRun, chainRunN, headSpan, remKeyRun, remKeyRunN,
                mb, mb.overallHashMemory,
                involvedLevels, involvedLevelCount, impStr,
                parameters.maxIterationNumberVariable, parameters.standardMaxSecondaryNumber, false,
                parameters.minNumOperatorsKey, StrSpan("implication", 11), true, impStr, impValidity,
                RuleIndexOp{}, (status == 3) ? &mb.workingMemory : nullptr);
            }
            if (status != 3) {
                {
                RT_SCOPE_HERE("INSTALL_HASHMEM_LOCAL");
                hashRuleInstaller.addToHashMemory(chainRun, chainRunN, headSpan, remKeyRun, remKeyRunN,
                    mb, mb.localHashMemory,
                    involvedLevels, involvedLevelCount, impStr,
                    parameters.maxIterationNumberVariable, parameters.standardMaxSecondaryNumber, false,
                    parameters.minNumOperatorsKey, StrSpan("implication", 11), true, impStr, impValidity);
                }
                {
                RT_SCOPE_HERE("INSTALL_HASHMEM_DELTA");
                hashRuleInstaller.addToHashMemory(chainRun, chainRunN, headSpan, remKeyRun, remKeyRunN,
                    mb, mb.localHashMemoryDelta,
                    involvedLevels, involvedLevelCount, impStr,
                    parameters.maxIterationNumberVariable, parameters.standardMaxSecondaryNumber, false,
                    parameters.minNumOperatorsKey, StrSpan("implication", 11), true, impStr, impValidity);
                }
            }
            }

            // Rule-16 origin maintenance at the mailed-carrier install
            // (D-274): a rule recovered from an external-mail carrier
            // (status 3) is installed and fireable at THIS LB, so this LB's
            // map must hold a history row for it; local installs keep their
            // disintegration-path history writes.
            if (status == 3 && parameters.trackHistory) {
                RT_SCOPE_HERE("INSTALL_ORIGIN_ROW");
                // Attached implication products expand their own compact,
                // including those reached through an inherited source family.
                // Preserve that intermediate statement in the proof chain.
                const bool compactProduct = gl::CompactExpressionSyntax::isOrCompactSpan(doorExpr)
                    || gl::CompactExpressionSyntax::isCompactImplicationSpan(carrier);
                const OriginDep carrierDep[1] = {
                    { compactProduct ? carrier : doorExpr,
                      compactProduct ? impValidity : doorValidity } };
                addOriginEncoded(mb.exprOriginMap, mb.originInterner, impStr, impValidity,
                    OriginTag::expansion, carrierDep, 1,
                    (parameters.compressor_mode
                         ? parameters.compressor_max_origins_per_expr
                         : parameters.max_origin_per_expr));
            }

            // The dump's index of installed implications (Rule 14 section
            // source) and the carrier index the canonicalization removes
            // rules through.
            {
            RT_SCOPE_HERE("INSTALL_INDEX_PAIR");
            const int32_t pairTextId = mb.lbStateInterner.encode(impStr);
            const int32_t pairScopeId = mb.lbStateInterner.encode(impValidity);
            const LbStatePairKey pairKey{ pairTextId, pairScopeId };
            // A pair the running apply retired but has not erased yet: erase the
            // stale entry for real before re-minting, so the key lands at the
            // column's end exactly as an immediate erase + mint would, and mark
            // it revived so the apply's final pass keeps it.
            DeferredExpandedErase* const deferred = mb.deferredExpandedErase;
            if (deferred != nullptr) {
                const int64_t packed = packInt32Pair(pairTextId, pairScopeId);
                if (deferred->pending.lookup(packed) != 0) {
                    mb.expandedImplications.erase(pairKey);
                    deferred->revived.mint(packed);
                }
            }
            const bool newlyHeld = mb.expandedImplications.lookup(pairKey) == 0;
            mb.expandedImplications.mint(pairKey);
            // A deferred OR head newly held here: the facts that refuted a
            // disjunct before it arrived become events now
            // (D-597).
            if (demandBuilt && newlyHeld) rejectedSourceReplay.stageExistingDeferredOrEvents(mb, impStr);
            }
            // The carrier index removes installed rules from hash memory; a
            // demand-built rule has none.
            if (!demandBuilt) {
            RT_SCOPE_HERE("INSTALL_INDEX_COMPACT");
            appendCompactExpansion(mb, carrier, impValidity, impStr, impValidity,
                status == 3 ? RuleInstallKind::External : RuleInstallKind::Local);
            }
        }
}

namespace gl {
/// @brief Bind inputs for compact rule processor.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param canonicalizer Borrowed Canonicalizer interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param disintegrationProcessor Borrowed DisintegrationProcessor interface or state; must outlive this component.
/// @param hashRuleInstaller Borrowed HashRuleInstaller interface or state; must outlive this component.
/// @param statementMemory Borrowed StatementMemory interface or state; must outlive this component.
/// @param integrationBuilder Borrowed IntegrationInstructionBuilder interface or state; must outlive this component.
/// @param rejectedSourceReplay Borrowed RejectedSourceReplay (the demand-built rule decision); must outlive this component.
/// @return None; construction binds the supplied references.
inline CompactRuleProcessor::CompactRuleProcessor(Canonicalizer& canonicalizer, const ProverParameters& parameters, DisintegrationProcessor& disintegrationProcessor, HashRuleInstaller& hashRuleInstaller, StatementMemory& statementMemory, IntegrationInstructionBuilder& integrationBuilder, RejectedSourceReplay& rejectedSourceReplay) : canonicalizer(canonicalizer), parameters(parameters), disintegrationProcessor(disintegrationProcessor), hashRuleInstaller(hashRuleInstaller), statementMemory(statementMemory), integrationBuilder(integrationBuilder), rejectedSourceReplay(rejectedSourceReplay) {}
} // namespace gl
