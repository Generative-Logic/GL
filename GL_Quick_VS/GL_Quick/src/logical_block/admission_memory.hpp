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
#include "../logical_block/derivation_tree.hpp"
#include "../logical_block/canonicalizer.hpp"
#include "../logical_block/canonical_template_matcher.hpp"
#include "../logical_block/forced_products.hpp"
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
class EquivalenceQueries;
class IntegrationProcessor;
class RejectedSourceReplay;
namespace tests { struct AdmissionMemoryFixture; }

/// @brief The own scopes one committed witness admission opened: induction
///        scopes (routes 1 and 5) or int scopes (route 3).
/// @details A route-1 admission through an admission value flagged "admitted
/// by induction hypothesis" (`kAdmissionInductionHypothesisFlag`) eats a
/// premise of the induction hypothesis: each eaten premise of each
/// admitting chain opens its own scope `<scope>_boundary_is_<chain>_<witness>`
/// (D-519), and the advanced chain is filed there. A
/// route-3 commit that first fills a premise-bound integration copy opens
/// the copy's int scope `<scope>_boundary_int_scope_<premise>_<witness>` and
/// continues the copy there (D-578). The
/// disintegration publishes the admitted witness subtree in every scope listed,
/// in commit order, instead of its own scope, unless `publishedHere` is set.
/// @invariant Holds distinct minted scope ids; overflow asserts.
/// @see AdmissionMemory::isAdmitted, AdmissionMemory::isAdmittedIntegration,
///      DisintegrationProcessor::disintegrateExpr2.
struct InductionScopeSink {
    static constexpr int32_t kCapacity = 8;
    NameId scopes[kCapacity] = {};
    int32_t count = 0;
    /// Set by a route-3 commit when an instruction that opens no int scope
    /// (no premise binding, or a continuation inside an int scope) accepted
    /// the witness: the witness is then published at the offering scope as
    /// before, and the int scopes see it from there
    /// (D-578).
    bool publishedHere = false;
    /// The disintegration's channel for the hypothesis's missing expressions
    /// in the scopes the eats open (`DisintProducts::hypothesisDemands`); an
    /// eat of the induction hypothesis needs it (asserted)
    /// (D-575).
    DisintPairChannel* hypothesisDemands = nullptr;

    /// @brief Record one opened induction scope.
    /// @details A scope already listed is not listed twice: the same eat
    /// reached through two equal records opens one scope (`encodePush` returns
    /// the existing id).
    /// @param scope A minted induction scope id.
    /// @return Void.
    /// @invariant At most `kCapacity` distinct scopes (asserted).
    void add(NameId scope) {
        assert(scope != 0 && "InductionScopeSink::add: an unminted scope");
        for (int32_t i = 0; i < count; ++i)
            if (scopes[i] == scope) return;
        assert(count < kCapacity && "InductionScopeSink::add: one admission opens too many induction scopes");
        scopes[count++] = scope;
    }
};

/// @brief The names in `rec`'s role and in the induction variable's role for
///        one integration instruction of a weak step block.
///
/// @details
/// Every integration of a weak step block is tied to the induction through
/// `rec` and the inducted digit n, the arguments of the block's hypothesis
/// `(in2[rec,n,s])`. A copy bound to the second premise of a twin conjecture
/// integrates that premise at the twin partner's predecessor instead, named
/// `rec2` by the copy `(=[q,rec2])`: there `rec2` takes `rec`'s role and the
/// partner l takes n's (D-589,
/// D-581). The views alias the block's NameMap
/// (I-3): a caller that mints copies them first.
struct InductionRoles {
    StrSpan rec;
    StrSpan variable;
};

/// @brief The induction roles of an integration instruction of a weak step
///        block, by the premise level it is bound to.
///
/// @details
/// A copy bound to the second twin premise (its premise level is the block's
/// `Memory::twinSecondLevel`) has `rec2`, the name of the twin partner's
/// predecessor (`Memory::twinPredecessorId`), in `rec`'s role and the partner
/// (`Memory::twinPartnerId`) in the induction variable's; every other
/// instruction, an unbound one included, has `rec` and the inducted digit,
/// the second argument of the block's hypothesis
/// (D-589).
///
/// @param mb           A step block with its recursion hypothesis
///                     `(in2[rec,<variable>,<successor>])` (asserted).
/// @param premiseLevel The instruction's premise level, 0 when unbound.
/// @return The two names, as views into the block's NameMap.
/// @invariant Mints nothing. A copy bound to the second twin premise exists
///            only once its predecessor is recorded (asserted).
/// @see stepKeyConnected, premiseCopyRefuses,
///      AdmissionMemory::fileTwinSecondGoal.
inline InductionRoles inductionRolesOf(const Memory& mb, int32_t premiseLevel) {
    if (premiseLevel != 0 && premiseLevel == mb.twinSecondLevel) {
        assert(mb.twinPredecessorId != 0 && mb.twinPartnerId != 0
            && "inductionRolesOf: a second-twin copy exists once its predecessor is recorded");
        return { mb.nameMap.decodeView(mb.twinPredecessorId), mb.nameMap.decodeView(mb.twinPartnerId) };
    }
    assert(mb.recursionHypothesisId != 0 && "inductionRolesOf: a step block with its recursion hypothesis");
    StrSpan args[ExecutionParameters::MAX_ARITY];
    const int32_t argN = getArgsSpans(mb.nameMap.decodeView(mb.recursionHypothesisId), args,
                                      ExecutionParameters::MAX_ARITY);
    assert(argN == 3 && equalSpans(args[0], StrSpan("rec", 3))
        && "inductionRolesOf: the recursion hypothesis is (in2[rec,<variable>,<successor>])");
    return { StrSpan("rec", 3), args[1] };
}

/// @brief Whether a premise-bound integration instruction copy refuses an
///        arriving expression, from the expression's precomputed signature.
///
/// @details
/// Only a copy bound to one premise of an identical-goal group
/// (`IntegrationForbiddenSets::premiseLevel` set) refuses. It refuses an
/// arriving expression whose signature names one of its forbidden decimal
/// variables or a forced witness of one of its forbidden levels
/// (D-579). Decided at a scope that is
/// neither an int scope nor below one, where every premise-bound copy is a
/// root copy whose fill would start its int scope, it also refuses an
/// arriving expression with a free argument that is not a number or `rec`
/// (D-578, the initiation gate); for a copy bound to
/// the second twin premise `rec2`, the name of the twin partner's
/// predecessor, stands in `rec`'s place (D-589). The route-3 probe and commit, the
/// trial gates, the enclosing fill, the park marks and the statement fill
/// decide through this one predicate.
///
/// @param nm                 The name map owning the validity forest.
/// @param scopeId            The minted scope the route is decided at.
/// @param forbidden          The instruction's split binding.
/// @param arrivingVariables  The arriving expression's decimal variables
///                           (`signatureTokenSets`).
/// @param arrivingLevels     The arriving expression's forced-witness levels.
/// @param arrivingNumbersOnly Whether every free argument of the arriving
///                           expression is a number or the name in `rec`'s
///                           role for this copy (`hasOnlyNumberFreeArguments`,
///                           `inductionRolesOf`).
/// @return True iff the copy refuses the arriving expression.
/// @invariant Pure; mints nothing.
/// @see signatureTokenSets, hasOnlyNumberFreeArguments,
///      ScopeSyntax::atOrBelowIntScope.
inline bool premiseCopyRefuses(const NameMap& nm, NameId scopeId,
        const IntegrationForbiddenSets& forbidden,
        uint64_t arrivingVariables, uint64_t arrivingLevels, bool arrivingNumbersOnly) {
    assert(scopeId > 0 && "premiseCopyRefuses: expected a minted scope");
    if (forbidden.premiseLevel == 0) {
        assert(forbidden.variables == 0 && forbidden.levels == 0
            && "premiseCopyRefuses: forbidden sets without a premise binding");
        return false;
    }
    if ((forbidden.variables & arrivingVariables) != 0 || (forbidden.levels & arrivingLevels) != 0) return true;
    return !arrivingNumbersOnly && !gl::ScopeSyntax::atOrBelowIntScope(nm, scopeId);
}

/// @brief Whether a premise-bound integration instruction copy refuses an
///        arriving expression, read from the expression text.
///
/// @details
/// The text twin of the signature overload: computes the expression's
/// signature masks (`signatureTokenSets`) and its numbers-only verdict
/// (`hasOnlyNumberFreeArguments`, with the copy's name in `rec`'s role,
/// `inductionRolesOf`; D-589) and decides as that
/// overload does. An instruction without a premise binding never refuses and
/// reads nothing (D-578).
///
/// @param mb        The block (its NameMap owns the validity forest; its
///                  twin fields name the second twin premise's roles).
/// @param scopeId   The minted scope the route is decided at.
/// @param forbidden The instruction's split binding.
/// @param arriving  The arriving expression (the existence being
///                  disintegrated, or a parked source).
/// @return True iff the copy refuses @p arriving.
/// @invariant Pure; mints nothing.
/// @see premiseCopyRefuses (signature overload).
inline bool premiseCopyRefuses(const Memory& mb, NameId scopeId,
        const IntegrationForbiddenSets& forbidden, StrSpan arriving) {
    if (forbidden.premiseLevel == 0) return premiseCopyRefuses(mb.nameMap, scopeId, forbidden, 0, 0, true);
    uint64_t variables = 0;
    uint64_t levels = 0;
    signatureTokenSets(arriving, variables, levels);
    // Only the name in `rec`'s role matters here; a copy not bound to the
    // second twin premise keeps `rec`.
    const StrSpan recRole = forbidden.premiseLevel == mb.twinSecondLevel
        ? inductionRolesOf(mb, forbidden.premiseLevel).rec : StrSpan("rec", 3);
    return premiseCopyRefuses(mb.nameMap, scopeId, forbidden, variables, levels,
                              hasOnlyNumberFreeArguments(arriving, recRole));
}

/// @brief The key of the ancestor logic block at one level of a block's
///        chain: the original premise a hypothesis premise came from.
///
/// @details
/// A step block's induction hypothesis is its conjecture's chain at `rec`;
/// chain element k came from the conjecture premise that keys the ancestor
/// block at level k + 1 (D-579). The walk
/// follows `parentMemory` to that level. The view reads the never-deloaded
/// skeleton interner, which no prover phase mints into while it is held
/// (I-97).
///
/// @param mb    The step block.
/// @param level The ancestor's level, below the block's own.
/// @return The ancestor's key.
/// @invariant Mints nothing; the ancestor exists (asserted).
/// @see AdmissionMemory::updateAdmissionMap.
inline StrSpan ancestorPremiseKeyView(const Memory& mb, int32_t level) {
    assert(level > 0 && level < mb.level
        && "ancestorPremiseKeyView: a premise level below the step block's own");
    const Memory* ancestor = mb.parentMemory;
    while (ancestor != nullptr && ancestor->level > level) ancestor = ancestor->parentMemory;
    assert(ancestor != nullptr && ancestor->level == level
        && "ancestorPremiseKeyView: no ancestor block at a premise's level");
    return ancestor->exprKeyView();
}

/// @brief The forbidden sets of the copy bound to one premise of a twin pair.
///
/// @details
/// A twin conjecture's two premises (the original premises at block levels
/// @p firstLevel and @p secondLevel, one core expression) are integrated by
/// two goals of one group: the first at `rec`, the second at the twin
/// partner's predecessor (D-589). As for an
/// identical-goal group (`premiseForbiddenSets`), the copy bound to premise
/// @p member forbids the other premise's own decimal variables and level, and
/// carries its own premise's level.
///
/// @param mb          The weak step block below both premises.
/// @param firstLevel  The first twin premise's block level.
/// @param secondLevel The second twin premise's block level.
/// @param member      0 for the first premise's copy, 1 for the second's.
/// @param out         Receives the copy's binding.
/// @return Void.
/// @invariant Pure; reads the skeleton interner only (I-97).
/// @see premiseForbiddenSets, AdmissionMemory::fileTwinSecondGoal.
inline void twinPremiseForbiddenSets(const Memory& mb, int32_t firstLevel, int32_t secondLevel,
        int32_t member, IntegrationForbiddenSets& out) {
    assert(firstLevel > 0 && secondLevel > 0 && firstLevel != secondLevel && (member == 0 || member == 1)
        && "twinPremiseForbiddenSets: two distinct premise levels and one of them assigned");
    const StrSpan premises[2] = { ancestorPremiseKeyView(mb, firstLevel), ancestorPremiseKeyView(mb, secondLevel) };
    const int32_t levels[2] = { firstLevel, secondLevel };
    assert(equalSpans(extractExpressionUniversalSpan(premises[0]), extractExpressionUniversalSpan(premises[1]))
        && "twinPremiseForbiddenSets: twin premises share their core expression");
    premiseForbiddenSets(premises, levels, 2, member, out.variables, out.levels);
    out.premiseLevel = levels[member];
}

/// @brief The key a route-5 admission probe accepted, with its orientation.
/// @details Filled by `AdmissionMemory::isAdmittedIntegrationDouble` on a hit:
/// the packed key and the two argument positions of the constituent that
/// stood for the key's markers — the admitted witness at `markerAt`, the
/// second open argument at `otherAt`. The earlier of the two positions is the
/// key's `marker1`.
/// @see AdmissionMemory::isAdmittedIntegrationDouble,
///      AdmissionMemory::makeDoubleMarkerQueryScratch.
struct DoubleMarkerHit {
    int64_t key = 0;
    int32_t markerAt = -1;
    int32_t otherAt = -1;
};

/// @brief Which induction demand of a strong step block a double-marker key
///        is (D-502, D-566).
/// @details A strong step block files two: the bound premise `m + x = n` of
/// its first hypothesis (`Memory::inductionDoubleKey`) and the bound premise
/// `x · m = n` of its multiplication catcher (`Memory::inductionProductKey`).
/// Every other double-marker key is an ordinary demand.
/// @see AdmissionMemory::updateAdmissionMapIntegrationDouble.
enum class InductionDemand : uint8_t {
    none = 0,
    sum = 1,
    product = 2,
};

/// @brief Admission Memory operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class AdmissionMemory {
friend struct tests::AdmissionMemoryFixture;
public:
/// @brief Bind inputs for admission memory.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param admissionPolicy Borrowed AdmissionPolicy interface or state; must outlive this component.
/// @param anchorInfo Borrowed ce::AnchorInfo interface or state; must outlive this component.
/// @param canonicalTemplates Borrowed CanonicalTemplateMatcher interface or state; must outlive this component.
/// @param deferredAncestorAdmissionsMutex Borrowed std::mutex interface or state; must outlive this component.
/// @param deferredAncestorPages Borrowed std::optional interface or state; must outlive this component.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param scopeTypes Borrowed ScopeTypeResolver interface or state; must outlive this component.
/// @param steward Borrowed std::optional interface or state; must outlive this component.
/// @param integrationProcessor Borrowed IntegrationProcessor interface or state; must outlive this component.
/// @param rejectedSourceReplay Borrowed RejectedSourceReplay interface or state; must outlive this component.
/// @param equivalenceQueries Borrowed EquivalenceQueries interface (the known-statement probe of the gates); must outlive this component.
/// @return None; construction binds the supplied references.
AdmissionMemory(AdmissionPolicy& admissionPolicy, const ce::AnchorInfo& anchorInfo, CanonicalTemplateMatcher& canonicalTemplates, std::mutex& deferredAncestorAdmissionsMutex, std::optional<SealedPageSet>& deferredAncestorPages, const CompiledDefinitionRegistry& definitions, const ProverParameters& parameters, ScopeTypeResolver& scopeTypes, std::unique_ptr<MemorySteward>& steward, IntegrationProcessor& integrationProcessor, RejectedSourceReplay& rejectedSourceReplay, EquivalenceQueries& equivalenceQueries);

public:
/// @brief Seed the LB's admission map from a renamed key chain —
        ///        span-run-native, zero-heap interior.
        ///
        /// @details
        /// Renames the key chain via `renamingChainScratch`, and for each
        /// operator element with exactly one distinct non-remaining argument —
        /// and, in a recursion-owned chain, each positive compound element so
        /// placed that is not a two-input relation
        /// (D-413) — integrates the
        /// marker-substituted form and records the admission entry. Every
        /// string arrives as a `StrSpan` run over stable caller
        /// input; `key`/`remainingArgs` are sorted-unique runs (the former
        /// `std::vector` / `std::set` iteration orders) and `remainingArgs`
        /// MUST be sorted-unique — the frozen sequence, the `withoutRemaining`
        /// set-difference membership, and `prepareIntegration`'s sorted
        /// `unchangeableArgsSorted` all depend on it.
        ///
        /// @param key               The admission key chain, as spans — the
        ///                          rule's PREMISES only, never its head: a
        ///                          premise-completion demand asks for a
        ///                          missing premise instance; the head is
        ///                          the product the rule derives and can
        ///                          never be demanded (a head marker such as
        ///                          `!(in3[q,d,t,marker])` on the `×` slot is
        ///                          the D-379 bug).
        /// @param keyN              Element count of `key`.
        /// @param remainingArgs     Sorted-unique remaining-argument run.
        /// @param remN              Element count of `remainingArgs`.
        /// @param maxAdmissionDepth Recursion admission depth cap.
        /// @param maxSecondaryNumber Recursion secondary-iteration cap.
        /// @param admissionFlags    The seeded entry's flag byte:
        ///                          `kAdmissionRecursionFlag` marks it
        ///                          recursion-owned, with
        ///                          `kAdmissionInductionHypothesisFlag` it is the
        ///                          induction hypothesis's chain (0, 1 or 3). A
        ///                          hypothesis chain filed in an induction scope
        ///                          wakes the parked sources its new demands can
        ///                          eat (`revisitRejected2` at that scope), so a
        ///                          source parked at an ancestor before the chain
        ///                          arrived reaches it
        ///                          (D-519).
        /// @param validityName      Validity scope name for the packed key.
        /// @param recursionStatus   Files each written key's status as
        ///                          recursion although the value stays
        ///                          ordinary (`admissionFlags == 0`): a witness
        ///                          admitted through the key becomes a product
        ///                          of recursion, pure for the marker gate,
        ///                          while the value keeps the ordinary
        ///                          admission gate, which admits for originals
        ///                          with witness arguments too (I-285 reads the
        ///                          value's flag only). The forced successor
        ///                          demand's setting
        ///                          (D-571).
        /// @param weakHypothesisInstall True only when a weak step block
        ///                          installs its induction hypothesis's chain:
        ///                          if no element of the chain has an ordinary
        ///                          demand (the hypothesis cannot start
        ///                          eating), every element with exactly two
        ///                          unbound arguments demands an integration
        ///                          with the two markers `marker1` and
        ///                          `marker2`
        ///                          (D-586).
        /// @invariant The interner-touch sequence per admitted element is
        ///            FROZEN, in this order: (non-atomic only)
        ///            `nameMap.encode(removed)`; `valueInterner.encode` per
        ///            remainingArg in SET order (remIds); `valueInterner.encode`
        ///            per renamedKey element in CHAIN order (keyIds);
        ///            `mintTemplateKey(removed, validityName)`;
        ///            `templateInterner.encode` per non-`"marker"` arg of
        ///            `removed` in getArgs order. The remainingArgs-before-key
        ///            order (remIds before keyIds) is load-bearing — pinned by
        ///            `admission_value_ctor_mint_order_frozen`. This span
        ///            conversion swaps argument TYPES at existing calls, never
        ///            the call order (`renamingChainScratch` mints nothing).
        /// @see renamingChainScratch, prepareIntegration, mintTemplateKey,
        ///      insertAdmissionIdsBlob, classifyChainElement.
inline void updateAdmissionMap(Memory& mb,
            const StrSpan* key, int32_t keyN,
            const StrSpan* remainingArgs, int32_t remN,
            int maxAdmissionDepth,
            int maxSecondaryNumber,
            uint8_t admissionFlags,
            StrSpan validityName,
            bool recursionStatus = false,
            bool weakHypothesisInstall = false);

        /// @brief How one element of an admission key chain takes part in the
        ///        install-time demands of `updateAdmissionMap`.
        enum class ChainElementDemand : uint8_t {
            none,              ///< No demand: no participant, or its unbound arguments do not qualify.
            fullyBoundPremise, ///< A recursion-owned compound premise with every argument fixed.
            singleMarker,      ///< Exactly one unbound argument: the marker route.
            twoUnbound         ///< Exactly two unbound arguments of a positive element the marker route takes.
        };

        /// @brief Classify one element of an admission key chain for the
        ///        install-time demands of `updateAdmissionMap`.
        ///
        /// @details
        /// The element participates when it is a functional operator or a
        /// config-backed compound relation (a non-atomic compiled entity with a
        /// core configuration); anything else has no demand. A config-less
        /// compound is a spontaneous runtime-minted compact (implication<N> /
        /// existence<N> / and<N>), not a relation the demand machinery manages.
        /// Its unbound arguments are its distinct arguments outside
        /// @p remainingArgs, written sorted-unique to @p unbound (the former
        /// `std::set` set-difference, byte for byte).
        ///
        /// An operator with one unbound argument takes the marker route. A
        /// compound relation with none is a fully bound premise of a
        /// recursion-owned chain (its integration goal is formulated at
        /// once). A compound relation with one takes the marker route only as
        /// an induction demand (D-413): in a recursion-owned chain, positive,
        /// and not a two-input relation without output (preorder,
        /// strictOrder), whose demands would re-open the per-witness pump
        /// (D-257). A positive element that would take the marker route but
        /// has exactly two unbound arguments is two-unbound: the weak
        /// hypothesis's two-marker start reads it
        /// (D-586). Mints nothing.
        ///
        /// @param element         The renamed chain element (span over a stable
        ///                        buffer).
        /// @param remainingArgs   The chain's sorted-unique remaining arguments.
        /// @param remN            Their number.
        /// @param partOfRecursion Whether the chain is recursion-owned.
        /// @param unbound         Caller buffer of `ExecutionParameters::MAX_ARITY`
        ///                        spans receiving the distinct unbound arguments,
        ///                        sorted-unique (spans into @p element).
        /// @param unboundN        Receives their number.
        /// @return The element's demand kind.
        /// @invariant Pure: no interner, map or arena is touched; 0% heap.
        /// @see updateAdmissionMap.
        ChainElementDemand classifyChainElement(StrSpan element,
            const StrSpan* remainingArgs, int32_t remN, bool partOfRecursion,
            StrSpan* unbound, int32_t& unboundN) const {
            unboundN = 0;
            const StrSpan coreSp = extractExpressionSpan(element);
            const bool isOp = coreSp.len > 0
                && this->definitions.operatorNames().find(std::string_view(coreSp.ptr,
                    static_cast<std::size_t>(coreSp.len))) != this->definitions.operatorNames().end();
            bool isCompoundRelation = false;
            if (!isOp) {
                const LogicalEntity* ceEnt =
                    (coreSp.len > 0) ? this->definitions.compiledEntity(coreSp) : nullptr;
                if (ceEnt == nullptr
                    || equalSpans(StrSpan(ceEnt->category), StrSpan("atomic", 6))) {
                    return ChainElementDemand::none;
                }
                if (this->definitions.coreConfig(coreSp) == nullptr) {
                    return ChainElementDemand::none;
                }
                isCompoundRelation = true;
            }

            StrSpan argSpans[ExecutionParameters::MAX_ARITY];
            const int32_t argN = getArgsSpans(element, argSpans,
                                              ExecutionParameters::MAX_ARITY);
            for (int32_t a = 0; a < argN; ++a) {
                bool inRem = false;
                for (int32_t lo = 0, hi = remN; lo < hi; ) {
                    const int32_t mid = (lo + hi) / 2;
                    const int c = compareSpans(remainingArgs[mid], argSpans[a]);
                    if (c == 0) { inRem = true; break; }
                    if (c < 0) lo = mid + 1; else hi = mid;
                }
                if (inRem) continue;
                int32_t lo = 0, hi = unboundN; bool dup = false;
                while (lo < hi) {
                    const int32_t mid = (lo + hi) / 2;
                    const int c = compareSpans(unbound[mid], argSpans[a]);
                    if (c == 0) { dup = true; break; }
                    if (c < 0) lo = mid + 1; else hi = mid;
                }
                if (dup) continue;
                assert(unboundN < ExecutionParameters::MAX_ARITY
                    && "classifyChainElement: distinct unbound arg count exceeds MAX_ARITY");
                for (int32_t z = unboundN; z > lo; --z) unbound[z] = unbound[z - 1];
                unbound[lo] = argSpans[a]; ++unboundN;
            }

            if (isCompoundRelation && unboundN == 0) {
                return partOfRecursion ? ChainElementDemand::fullyBoundPremise
                                       : ChainElementDemand::none;
            }
            if (unboundN != 1 && unboundN != 2) return ChainElementDemand::none;
            const bool positive = element.len > 0 && element.ptr[0] != '!';
            if (isCompoundRelation) {
                const ce::CoreExpressionConfig* compoundCfg = this->definitions.coreConfig(coreSp);
                assert(compoundCfg && "classifyChainElement: compound relation must be config-backed");
                const bool twoInputRelation = compoundCfg->inputIndices.size() == 2
                    && compoundCfg->outputIndices.empty();
                if (!(partOfRecursion && positive && !twoInputRelation))
                    return ChainElementDemand::none;
            }
            if (unboundN == 1) return ChainElementDemand::singleMarker;
            return positive ? ChainElementDemand::twoUnbound : ChainElementDemand::none;
        }

public:
/// @brief Integration-side admission check — gates whether a
        /// compound integration body may fire from the given LB.
        ///
        /// @details
        /// Uses isIntegrationInstructionKeyEligible before probing or replaying.
        /// Prefixes non-marker arguments with `u_` and probes the
        /// `mb.overallHashMemory.admissionMapIntegration` keys visible at the
        /// scope: owned there or, for an OR case, at the scope its demands come
        /// from (`ancestorDemandOwner`, `matchingVisibleIntegrationDemands`,
        /// D-572).
        /// A pure probe reports key presence without mutation. A committed
        /// hit records the witness for each stored instruction and invokes
        /// `prepareIntegrationCore2` with its marker replaced by that witness;
        /// instructions already processed for the witness are skipped.
        ///
        /// @param mb           Owning LB.
        /// @param expr         Constituent expression (span source; read-only).
        /// @param var          Witness/replacement variable (span source).
        /// @param markedExpr   Marked template form (span source).
        /// @param validityName Validity scope (span source).
        /// @return True iff an eligible integration instruction key exists,
        ///         including when its instructions already processed the witness.
        /// @see `isAdmitted` — disintegration-side counterpart.
        /// @see [`HashMemory::admissionMapIntegration`](memory.hpp#hashmemory).
        /// @param commit Apply successful instruction updates; false is a pure probe.
        /// @param selectedKey One already-matched key to process, or 0.
        /// @param linked      The route's linked arguments for the witness-link
        ///                    gate (D-529): in an
        ///                    induction step block an instruction whose path
        ///                    list names one of them refuses the route — the
        ///                    probe ignores it and the commit leaves it
        ///                    untouched. Null when @p linkedCount is 0.
        /// @param linkedCount Number of @p linked.
        /// @param history     The route's witness-history probe
        ///                    (D-522), or null: an
        ///                    instruction the derivation tree links to its path
        ///                    list refuses the route as well. The gate is off
        ///                    when neither @p linkedCount nor @p history is set.
        /// @param intScopes   For a commit: receives the int scopes the commit
        ///                    opened, one per first fill of a premise-bound copy
        ///                    outside every int scope, whose continuation runs
        ///                    there (D-578), and whether an
        ///                    instruction that opens none accepted the witness.
        ///                    Null for a probe.
inline bool isAdmittedIntegration(Memory& mb,
            StrSpan expr,
            StrSpan var,
            StrSpan markedExpr,
            StrSpan validityName,
            bool commit = true, int64_t selectedKey = 0,
            const StrSpan* linked = nullptr, int32_t linkedCount = 0,
            const WitnessHistoryProbe* history = nullptr,
            InductionScopeSink* intScopes = nullptr);

/// @brief Open, or find, the int scope a first fill of a premise-bound
        ///        integration copy starts, and record it.
        ///
        /// @details
        /// Mints `<parent>_boundary_int_scope_<premise>_<witness>`
        /// (`ScopeSyntax::intScopePayloadScratch`): the premise is the key of
        /// the ancestor block at the copy's premise level
        /// (`ancestorPremiseKeyView`), the witness the name that filled the
        /// marker (D-578). The same premise and
        /// witness reached through another key of the copy find the same
        /// scope. The scope's record in `LbMemory::intScopes` holds its goal
        /// group (the copy's integration goal with unlabelled markers, minted
        /// into the NameMap; a copy bound to the second twin premise reads
        /// `rec` for `rec2`, so it joins the first premise's group,
        /// D-589) and its premise
        /// level; the group's record in
        /// `LbMemory::intScopeGroups` holds the group's premise levels. A
        /// scope a wipe has closed never reopens, and a promoted group opens
        /// no scope (D-582): the result is then 0.
        ///
        /// @param mb       Owning block (single-threaded seam).
        /// @param parentId The scope the first fill is decided at; neither an
        ///                 int scope nor below one.
        /// @param wi       The premise-bound instruction being filled.
        /// @param witness  The filling name.
        /// @return The int scope, or 0 when it was closed or its group was
        ///         promoted.
        /// @invariant Mints into the NameMap and writes the two records only.
        /// @see ScopeSyntax::isIntScope, StatementMemory::addStatement.
inline NameId openIntScope(Memory& mb, NameId parentId, const WorkInstruction& wi, StrSpan witness);

public:
/// @brief Whether the step-key rule binds the route-3 keys filed at a scope
        ///        of a block.
        ///
        /// @details
        /// The rule binds where an induction hypothesis has an instance. A
        /// weak step block installs its hypothesis at `rec` on `main`, so it
        /// binds at every scope of a weak step block
        /// (D-583). A strong step block
        /// instantiates its hypothesis only at a value a strong demand
        /// catches, inside the induction scope the catch opens; `main` is the
        /// step's own goal context. So in a strong step block the rule binds
        /// at an induction scope and below
        /// (`ScopeSyntax::atOrBelowInductionScope`,
        /// D-584). It binds nowhere in any
        /// other block.
        ///
        /// @param mb           The block (single-threaded seam).
        /// @param validityName The filing scope (registered, asserted in a
        ///                     strong step block).
        /// @return `true` iff a key filed at @p validityName must be connected
        ///         (@ref stepKeyConnected).
        /// @invariant Mints nothing.
        /// @see stepKeyConnected, IntegrationProcessor::prepareIntegrationCore2.
inline bool stepKeyRuleBinds(Memory& mb, StrSpan validityName);

/// @brief Whether an integration key filed in an induction step block is
        ///        connected to the induction.
        ///
        /// @details
        /// Where the step-key rule binds (@ref stepKeyRuleBinds: every scope
        /// of a weak step block, an induction scope and below in a strong
        /// one) a route-3 key is filed only when one of its arguments (a `u_`
        /// prefix stripped) is connected
        /// (D-583,
        /// D-584):
        /// - `rec` or the induction variable n, the second argument of the
        ///   block's hypothesis `(in2[rec,n,s])`
        ///   (`ScopeSyntax::keyHoldsInductionVariable`);
        /// - a size marker (`LbMemory::sizeMarkers`, `recordSizeMarker`): a
        ///   value an earlier fill admitted through a connected key, and in a
        ///   strong step block the two names a strong demand caught; unless
        ///   the filing scope knows it equal to the induction's zero
        ///   (`Memory::inductionZeroId`, up to the scope's classes): a size of
        ///   zero bounds the key to one-point objects;
        /// - the predecessor x of `rec`, when `(in2[x,rec,s])` is known at the
        ///   filing scope or above, or the successor y of n, when
        ///   `(in2[n,y,s])` is known there, both up to the scope's classes
        ///   (`EquivalenceQueries::elementKnownAtOrAbove`).
        /// The key's `marker` and the wildcard `pi_` are open and never
        /// connect. The decision is made at filing: a key refused before its
        /// `in2` link is known is not filed later. For a copy bound to the
        /// second twin premise `rec2`, the name of the twin partner's
        /// predecessor, stands in `rec`'s place and the partner l in n's:
        /// `rec2`, l, a predecessor of `rec2` and a successor of l connect
        /// (`inductionRolesOf`, D-589).
        ///
        /// @param mb           The step block (single-threaded seam).
        /// @param key          The key text: `u_`-prefixed arguments and a
        ///                     marker, on caller-owned storage.
        /// @param validityName The filing scope (registered).
        /// @param premiseLevel The premise level of the copy filed under the
        ///                     key, 0 when unbound.
        /// @return `true` iff an argument of @p key is connected.
        /// @invariant Mints nothing; synchronizes only the arena-backed
        ///            comparison indexes the zero test and the two `in2`
        ///            probes read.
        /// @see recordSizeMarker, IntegrationProcessor::prepareIntegrationCore2.
inline bool stepKeyConnected(Memory& mb, StrSpan key, StrSpan validityName, int32_t premiseLevel = 0);

/// @brief Whether the key a route-3 commit or a statement fill admits a
        ///        value through is connected, so the value is a size marker.
        ///
        /// @details
        /// A value is a size marker when a connected key admitted it: such a
        /// key ties it to `rec`, n or a caught value
        /// (D-583). In a weak step block every
        /// key was filed under the rule, so every admitting key is connected.
        /// In a strong step block a key filed at `main` or another scope the
        /// rule does not bind may hold nothing connected, so the admitting
        /// key's text is checked at its own scope with @ref stepKeyConnected
        /// (D-584).
        ///
        /// @param mb The step block (single-threaded seam; asserted).
        /// @param pk The admitting key: an `admissionMapIntegration` key, its
        ///           template and validity minted.
        /// @return `true` iff the key's text is connected at its scope.
        /// @invariant Mints nothing.
        /// @see stepKeyConnected, recordSizeMarker.
inline bool admittingKeyConnected(Memory& mb, int64_t pk);

/// @brief Whether a route-3 key may admit a value at a scope of the block.
        ///
        /// @details
        /// A strong step block instantiates its hypothesis only inside the
        /// induction scopes its catches open (D-584). Its `main` files keys
        /// freely, and those keys serve the induction scopes and the OR cases
        /// below them (D-572). Outside the induction scopes a key that holds
        /// nothing connected to the induction (@ref admittingKeyConnected)
        /// admits nothing: the route-3 probe, the route-3 commit and the
        /// statement fill refuse it there, so the step's own goal builds no
        /// object without a size (D-602). At an
        /// induction scope and below every key admits as before; in every
        /// other block every key admits.
        ///
        /// @param mb      The block (single-threaded seam).
        /// @param pk      The admitting key: an `admissionMapIntegration` key,
        ///                its template and validity minted.
        /// @param scopeId The scope the value would be admitted at; registered
        ///                (asserted in a strong step block).
        /// @return `false` iff @p mb is a strong step block, @p scopeId is
        ///         neither an induction scope nor below one, and @p pk is not
        ///         connected at its own scope.
        /// @invariant Mints nothing.
        /// @see admittingKeyConnected, stepKeyRuleBinds, isAdmittedIntegration,
        ///      updateAdmissionMapIntegration.
inline bool strongStepKeyAdmits(Memory& mb, int64_t pk, NameId scopeId);

/// @brief Whether a route-3 key outside the step blocks is connected to the
        ///        goal of an instruction filed under it.
        ///
        /// @details
        /// The step-key game played on the goal's own variables
        /// (D-593): a step block connects its keys
        /// through `rec`, the induction variable and the size markers (D-583),
        /// any other block through the head variables of the goal. Arguments
        /// lose their `u_` prefix, and the labels `marker`, `marker1`,
        /// `marker2`, every `pi_` name and the anchor's arguments
        /// (`ScopeSyntax::anchorArgumentSpans`, the constants every theorem of
        /// the batch shares) take no part. The goal's remaining arguments are
        /// its head variables, each read as its representative in the classes
        /// of @p scope (`CanonicalTemplateMatcher::canonicalArgumentAtScope`),
        /// as a key filed there names it; a head variable whose representative
        /// is an anchor argument is a constant there and drops out. A goal
        /// without head variables is exempt: every key of it is connected.
        /// Otherwise one of the key's remaining arguments must be a head
        /// variable or a connected value of the block: a value an earlier
        /// admission wrote into an instruction through a connected key
        /// (`LbMemory::sizeMarkers`). Names compare as written, as in D-583.
        /// `(ordered[u_1,u_4,marker])` for the goal
        /// `(hasPrimeFactorization[1,3,4,5,2,6,7,9])` under
        /// `(AnchorFTA[1,2,3,4,5,6,7,8])` names only anchor arguments and is
        /// not connected; a key naming `u_9` is. A key also connects when an
        /// anchor argument that the goal holds k > 1 times appears in the key
        /// exactly k times: the key repeats the goal's own constant shape, as
        /// `(interval[u_1,u_4,u_2,u_2,marker])` does for
        /// `(sequence[1,4,2,2,10])`, whose witness is the interval [0,0].
        ///
        /// @param mb    The block (its anchor chain, classes and connected
        ///              values).
        /// @param key   The key text: `u_`-prefixed arguments and a marker, in
        ///              the canonical form of @p scope, on caller-owned
        ///              storage.
        /// @param goal  The instruction's marked goal, on caller-owned storage.
        /// @param scope The key's scope (registered, asserted).
        /// @return `true` iff @p goal has no head variable, an argument of
        ///         @p key is a head variable of @p goal or a connected value,
        ///         or @p key repeats an anchor argument exactly as often as
        ///         @p goal does, more than once.
        /// @invariant Allocates and mints nothing.
        /// @see admittingKeyConnectedToGoal, recordSizeMarker,
        ///      IntegrationProcessor::prepareIntegrationCore2.
inline bool goalKeyConnected(Memory& mb, StrSpan key, StrSpan goal, NameId scope);

/// @brief Whether the key a route-3 commit or a statement fill admits a
        ///        value through, outside the step blocks, is connected to the
        ///        accepting instruction's goal.
        ///
        /// @details
        /// Such a value becomes a connected value of the block, so the
        /// instruction's continuation keys that hold it are filed
        /// (D-593). The key's text is read from its
        /// template and checked with @ref goalKeyConnected at the key's own
        /// scope; a key that equivalence classes re-keyed was not checked at
        /// its filing and is checked here as written.
        ///
        /// @param mb   The block, neither a weak nor a strong step block
        ///             (asserted; single-threaded seam).
        /// @param pk   The admitting key: an `admissionMapIntegration` key, its
        ///             template minted.
        /// @param goal The accepting instruction's marked goal, on caller-owned
        ///             storage.
        /// @return `true` iff the key's text is connected to @p goal.
        /// @invariant Mints nothing.
        /// @see goalKeyConnected, recordSizeMarker, isAdmittedIntegration,
        ///      updateAdmissionMapIntegration.
inline bool admittingKeyConnectedToGoal(Memory& mb, int64_t pk, StrSpan goal);

/// @brief Record a value as a size marker of an induction step block, or as
        ///        a connected value of any other block.
        ///
        /// @details
        /// A value filled into an integration instruction of a weak or strong
        /// step block, at the route-3 commit or at the statement fill,
        /// through a connected key (@ref admittingKeyConnected) carries a
        /// size bound by `rec`, n or a caught smaller value; every later key
        /// holding it is connected too (D-583).
        /// In a strong step block the two names a strong demand catches, the
        /// smaller value m and its distance x (`m + x = n`, or the cofactor
        /// of `x * m = n`), are recorded when the catch opens its induction
        /// scope: they are the strong step's sizes, as `rec` is the weak
        /// step's (D-584). In any other block a
        /// value admitted through a key connected to the accepting
        /// instruction's goal (@ref admittingKeyConnectedToGoal) is a
        /// connected value, and the instruction's continuation keys that hold
        /// it are connected (D-593). The record holds
        /// for the whole block: the value is interned in the block's
        /// `valueInterner` and its id kept in `LbMemory::sizeMarkers`; a
        /// witness name is never reused within a block, so no scope wipe
        /// removes it.
        ///
        /// @param mb    The block (single-threaded seam).
        /// @param value The bare value (no `u_` prefix, not a marker).
        /// @invariant Mints the value into `valueInterner` when absent and its
        ///            id into `sizeMarkers`; idempotent.
        /// @see stepKeyConnected, goalKeyConnected.
inline void recordSizeMarker(Memory& mb, StrSpan value);

/// @brief File the second twin premise's integration goal at the twin
        ///        partner's predecessor.
        ///
        /// @details
        /// In a weak step block whose hypothesis install filed the first twin
        /// premise's goal (`Memory::twinFirstGoalId`), the registration of
        /// `(in2[q,l,s])` at `main` or in the released successor case of the
        /// partner's zero-or-successor OR (`Memory::twinCaseScopeId`), l the
        /// twin partner and s the anchor's successor, names a predecessor q
        /// of the partner. The scope of that registration receives the copy
        /// `(=[q,rec2])` on internal mail (inside an absorb on its output
        /// channel, `Memory::replayMailDuringAbsorb`, since the drained input
        /// is cleared), origin `variable copy`: `rec2`
        /// names the partner's predecessor as `rec` names the induction
        /// variable's (D-581). The second twin
        /// premise's goal is the first goal with `rec2` in place of `rec`:
        /// exchanging the twins maps the one premise onto the other, and the
        /// partner stands where the first premise holds `rec`. It is prepared
        /// at that scope with both markers changeable and every other
        /// argument fixed, as the copy bound to the second premise
        /// (`twinPremiseForbiddenSets`: its premise level, the first premise's
        /// own variables and level forbidden); its int scopes join the first
        /// goal's group, so the two integrations are promoted together, each
        /// into its own parent scope (D-589). `rec2` is
        /// recorded (`Memory::twinPredecessorId`) and takes `rec`'s role for
        /// that copy (`inductionRolesOf`); the goal is filed once.
        ///
        /// @param mb          The weak step block (single-threaded seam).
        /// @param predecessor The NameMap id of q.
        /// @param scopeId     The registration's scope: `main` or the twin
        ///                    successor case (asserted).
        /// @return Void.
        /// @invariant Mints the copy's staging and the goal's preparation
        ///            only.
        /// @see updateAdmissionMap, inductionRolesOf, openIntScope,
        ///      StatementMemory::addStatement.
inline void fileTwinSecondGoal(Memory& mb, NameId predecessor, NameId scopeId);

/// @brief Record a witness that its route-3 admission wrote into an
        ///        integration instruction as a one-way witness.
        ///
        /// @details
        /// A witness admitted through an `admissionMapIntegration` key updates
        /// the instructions waiting under that key once, at its admission; its
        /// role there is fixed (D-585). The route-3 commit
        /// (`isAdmittedIntegration`) calls this for every witness it writes into
        /// an instruction, and from then on the statement fill
        /// (`updateAdmissionMapIntegration`) never tries the witness, or a name
        /// of its class at the deposit scope, as a marker again
        /// (`oneWayWitnessAtScope`). The witness is interned in the block's
        /// `NameMap` and its id kept in `LbMemory::oneWayWitnesses`; a witness
        /// name is never reused within a block, so no scope wipe removes it.
        ///
        /// @param mb      The owning block (single-threaded seam).
        /// @param witness The bare admitted witness (no `u_` prefix, not a marker).
        /// @invariant Mints the witness into the NameMap when absent and its id
        ///            into `oneWayWitnesses`; idempotent.
        /// @see oneWayWitnessAtScope, isAdmittedIntegration.
inline void recordOneWayWitness(Memory& mb, StrSpan witness);

/// @brief Whether a statement-fill argument is a one-way witness or a name
        ///        of a one-way witness's class at the deposit scope.
        ///
        /// @details
        /// The decision of the one-way rule (D-585): the
        /// statement fill skips such an argument and tries no key with it as the
        /// marker. The argument itself is looked up first; otherwise its
        /// representative in the scope's complete class view is compared with
        /// each recorded witness's representative there
        /// (`CanonicalTemplateMatcher::canonicalMemberIdAtScope`), so a class-equal
        /// spelling of a recorded witness is skipped as well. A name the block's
        /// `NameMap` never interned is neither recorded nor in a class.
        ///
        /// @param mb    The owning block, with its classes at @p scope current.
        /// @param value The bare argument (any lifetime; not copied).
        /// @param scope The deposit scope (registered, nonzero).
        /// @return `true` iff @p value or a name of its class at @p scope is
        ///         recorded in `LbMemory::oneWayWitnesses`.
        /// @invariant Mints nothing; `false` at once while nothing is recorded.
        /// @see recordOneWayWitness, updateAdmissionMapIntegration.
inline bool oneWayWitnessAtScope(Memory& mb, StrSpan value, NameId scope);

public:
/// @brief The scope whose route-3 integration demands also serve a scope.
        ///
        /// @details
        /// An integration demand serves its own scope and, beyond it, only an
        /// induction scope (`_is_`, `ScopeSyntax::isInductionScope`) or an OR
        /// case (`ScopeSyntax::classifyOrScopeView` kind `Disintegration`)
        /// whose parent chain up to the demand's scope runs through induction
        /// scopes only: the owner is `ScopeSyntax::inductionRootOf` of the
        /// scope's parent (D-572, maintainer
        /// design). Every other scope reads its own demands only; no owner is
        /// a defined result, returned as 0.
        ///
        /// @param mb      Owning block.
        /// @param scopeId A minted validity id.
        /// @return The id of the serving owner, a strict ancestor of
        ///         @p scopeId, for an induction scope or an OR case; 0 for any
        ///         other scope.
        /// @invariant Pure; allocates and mints nothing.
        /// @see integrationDemandVisible, matchingVisibleIntegrationDemands,
        ///      hypothesisShotOnPath,
        ///      RejectedSourceReplay::wakeDescendantIntegrationParks.
inline NameId ancestorDemandOwner(const Memory& mb, NameId scopeId);

public:
/// @brief Whether the induction hypothesis shot on the path from a scope up to
        ///        the owner of the demand that serves it.
        ///
        /// @details
        /// A commit served by an ancestor's integration demand always admits
        /// the witness; it records the witness in the ancestor's instruction and
        /// continues that instruction at @p scopeId only after the hypothesis
        /// shot at @p scopeId or at a scope between it and @p owner
        /// (`LbMemory::hypothesisShots`, minted by the statement door for a
        /// `recursionHead` statement; D-572,
        /// maintainer design).
        ///
        /// @param mb      Owning block.
        /// @param scopeId The served scope, a minted validity id.
        /// @param owner   The serving owner, a strict ancestor of @p scopeId.
        /// @return True iff a scope from @p scopeId up to, not including,
        ///         @p owner is in `LbMemory::hypothesisShots`.
        /// @invariant Pure; allocates and mints nothing.
        /// @see ancestorDemandOwner, isAdmittedIntegration.
inline bool hypothesisShotOnPath(const Memory& mb, NameId scopeId, NameId owner);

public:
/// @brief Whether a route-3 integration demand for a marked template is
        ///        visible at a scope.
        ///
        /// @details
        /// A demand of `admissionMapIntegration` is visible at its own scope
        /// and at every OR case it serves (`ancestorDemandOwner`), so the case
        /// that can build a witness serves the integration prepared where its
        /// case split began (D-572). The
        /// probe tests the scope itself first, then the serving owner; every
        /// comparison is canonical at @p scopeId
        /// (`CanonicalTemplateMatcher::canonicalTemplatePresent` with the
        /// owning scope as source and @p scopeId as target).
        /// A wildcard key (`pi_` at one non-marker position,
        /// D-586) matches when a
        /// single-position variant of the query (`makeWildcardQueryScratch`)
        /// equals it; the variants are probed after the query itself and only
        /// in a block that filed such a key (`integrationWildcardFiled`).
        ///
        /// @param mb           Owning block.
        /// @param renamedMarked The marked template in key spelling (`u_`
        ///                     prefixes on every non-marker argument).
        /// @param validityName The probe scope, registered.
        /// @param scopeId      The id of @p validityName.
        /// @return True iff a key owned at @p validityName or at its serving
        ///         owner matches canonically at @p validityName.
        /// @invariant Adds no admission record; may extend the derived
        ///            class-aware template index only.
        /// @see matchingVisibleIntegrationDemands, isAdmittedIntegration,
        ///      RejectedSourceReplay::hasPotentialParkedDemand.
inline bool integrationDemandVisible(Memory& mb, StrSpan renamedMarked, StrSpan validityName,
            NameId scopeId);

public:
/// @brief Collect the route-3 integration demand keys visible at a scope
        ///        that canonically match a marked template.
        ///
        /// @details
        /// The matching keys owned at @p validityName come first, in the
        /// order `matchingCanonicalTemplates` returns them, then, for an OR
        /// case, those owned at the scope its demands come from
        /// (`ancestorDemandOwner`, D-572). The wildcard keys follow
        /// (D-586): for each
        /// non-marker position in order, the keys equal to the query's variant
        /// with `pi_` there (`makeWildcardQueryScratch`), own scope before
        /// owner, probed only in a block that filed such a key
        /// (`integrationWildcardFiled`).
        /// Every comparison is canonical at @p scopeId. The keys are the packed
        /// (template, owning scope) keys of `admissionMapIntegration`, so a
        /// caller reads and commits the instructions of an ancestor's key
        /// exactly as those of its own.
        ///
        /// @param mb            Owning block.
        /// @param renamedMarked The marked template in key spelling (`u_`
        ///                      prefixes on every non-marker argument).
        /// @param validityName  The probe scope, registered.
        /// @param scopeId       The id of @p validityName.
        /// @param out           Empty caller-owned vector receiving the keys.
        /// @param arena         The caller's page-tier scratch arena.
        /// @return Nothing.
        /// @invariant Adds no admission record; may extend the derived
        ///            class-aware template index only.
        /// @see integrationDemandVisible, isAdmittedIntegration,
        ///      integrationTrialGates, integrationEnclosingFill,
        ///      integrationDemandOpen, markWitnessLinkRefusals.
inline void matchingVisibleIntegrationDemands(Memory& mb, StrSpan renamedMarked, StrSpan validityName,
            NameId scopeId, PagedVector<int64_t>& out, ScratchArena& arena);

public:
/// @brief Whether an integration instruction's path list refuses an
        ///        admission route (D-529).
        ///
        /// @details
        /// The witness-link gate. A route admits one or more witnesses; the
        /// other part of each admitted witness's existence names some
        /// arguments of the witness's own definition set — @p linked
        /// (`witnessLinkArguments`). The instruction refuses the route when
        /// one of them is a member of @p pathList: an original variable of the
        /// instruction's integration goal or a value admitted earlier on the
        /// same path. Both sides are compared in canonical form at @p scope
        /// (the complete class view, `canonicalAdmissionText`), so a linked
        /// argument equal by class to a list member refuses as well. Only
        /// induction step blocks (`Memory::recursionAuxyImplId != 0`) gate; in
        /// any other block, with an empty list or no linked argument, nothing
        /// refuses.
        ///
        /// The history extension (D-522): when the
        /// direct check accepts and @p history carries labels, the derivation
        /// tree of the existence being disintegrated is checked as well
        /// (`witnessHistoryRefuses`).
        ///
        /// @param mb          Owning block.
        /// @param pathList    The instruction's path list `[name,...]`, or empty.
        /// @param linked      The route's linked arguments (null when
        ///                    @p linkedCount is 0).
        /// @param linkedCount Number of @p linked.
        /// @param scope       The admission scope (registered).
        /// @param history     The route's witness-history probe, or null.
        /// @return True iff the instruction refuses the route.
        /// @invariant Mints nothing; reads the canonical view and the history
        ///            only.
        /// @see witnessLinkArguments, witnessHistoryRefuses, isAdmittedIntegration.
inline bool witnessLinkRefuses(Memory& mb, StrSpan pathList,
            const StrSpan* linked, int32_t linkedCount, NameId scope,
            const WitnessHistoryProbe* history = nullptr);

public:
/// @brief Whether the derivation tree of the existence being disintegrated
        ///        links an admitted witness to an instruction's path list
        ///        (D-522).
        ///
        /// @details
        /// The history extension of the witness-link gate. Walks the
        /// derivation tree of @p history's start (`DerivationTree::walk`: one
        /// derivation per statement as buildStack takes it, premises only)
        /// and refuses when an ancestor has a leaf argument whose canonical
        /// form at @p scope is one of @p members and whose definition set,
        /// typed by position in that leaf (`ScopeTypeResolver::definitionSetOf`),
        /// is the definition set of one of the route's checked witnesses
        /// (operator outputs carry no label). The walk stops at the first hit.
        /// A connection that comes from a forced disintegration does not
        /// count: the walk ends, without checking it, at every forced product
        /// of the block (`ForcedProductMatch`).
        ///
        /// @param mb      Owning block (an induction step block).
        /// @param members The instruction's path list in canonical form at
        ///                @p scope, as spans.
        /// @param memberN Number of @p members.
        /// @param scope   The admission scope (registered).
        /// @param history The route's probe (active).
        /// @return True iff an ancestor links a checked witness to a member.
        /// @invariant Mints nothing; reads the history, the canonical view and
        ///            the definitions only; asserts that history is tracked.
        /// @see witnessLinkRefuses, DerivationTree::walk, WitnessHistoryProbe,
        ///      ForcedProductMatch.
inline bool witnessHistoryRefuses(Memory& mb, const StrSpan* members, int32_t memberN,
            NameId scope, const WitnessHistoryProbe& history);

public:
/// @brief The definition set of a witness in one element of its existence
        ///        (D-522).
        ///
        /// @details Types @p part by position (`ScopeTypeResolver::definitionSetOf`)
        /// and returns the witness's definition set: the label the
        /// witness-history check compares an ancestor's argument against.
        ///
        /// @param part    One element of the witness's existence, in its
        ///                spelling.
        /// @param witness The witness's spelling in @p part.
        /// @return The definition-set label, or empty when @p part does not
        ///         name @p witness.
        /// @invariant Allocates and mints nothing.
        /// @see witnessHistoryRefuses, WitnessHistoryProbe::addLabel.
inline StrSpan witnessDefinitionSet(StrSpan part, StrSpan witness);

public:
/// @brief Append the arguments one existence part links its witness to
        ///        (D-529).
        ///
        /// @details
        /// The arguments of @p part whose definition set equals the witness's
        /// (`ScopeTypeResolver::sameDefinitionSetArguments`), the witness
        /// itself and the batch's anchor arguments (the constants every
        /// theorem shares, `ScopeSyntax::anchorArgumentSpans`) excluded, are
        /// appended to `out[count..)` unless already present.
        ///
        /// @param mb      Owning block (its anchor chain).
        /// @param part    The other part of the witness's existence (flat MPL).
        /// @param witness The witness's spelling in @p part.
        /// @param out     Caller array receiving spans into @p part.
        /// @param count   Spans already in @p out.
        /// @param cap     Capacity of @p out.
        /// @return The new span count.
        /// @invariant Allocates and mints nothing.
        /// @see witnessLinkRefuses, ScopeTypeResolver::sameDefinitionSetArguments.
inline int32_t witnessLinkArguments(Memory& mb, StrSpan part, StrSpan witness,
            StrSpan* out, int32_t count, int32_t cap);

public:
/// @brief The path list of a new integration tree: the non-anchor fixed
        ///        arguments of its goal (D-529).
        ///
        /// @details
        /// Strips a `u_` prefix from each of @p fixedArgs, drops the batch's
        /// anchor arguments and builds the list text (`buildPathListScratch`).
        /// The caller passes the arguments the integration keeps fixed.
        ///
        /// @param mb        Owning block (its anchor chain).
        /// @param fixedArgs The goal's fixed arguments (null when @p n is 0).
        /// @param n         Number of @p fixedArgs.
        /// @param arena     String-tier scratch arena receiving the text.
        /// @return The list text, or empty when nothing remains.
        /// @invariant Mints nothing.
        /// @see IntegrationProcessor::prepareIntegration, buildPathListScratch.
inline ScratchString rootPathListScratch(Memory& mb, const StrSpan* fixedArgs, int32_t n,
            ScratchArena& arena);

public:
/// @brief A path list in canonical form at a scope
        ///        (D-529).
        ///
        /// @details
        /// Replaces every name by its class representative in the complete
        /// view at @p scope (`canonicalAdmissionText`) and rebuilds the sorted,
        /// duplicate-free text — the form every stored list keeps, like the
        /// rest of `admissionMapIntegration`.
        ///
        /// @param mb    Owning block.
        /// @param list  The list text, or empty.
        /// @param scope The scope (registered).
        /// @param arena String-tier scratch arena receiving the text.
        /// @return The canonical list text, or empty for the empty list.
        /// @invariant Mints nothing into the block's interners.
        /// @see buildPathListScratch, IntegrationProcessor::prepareIntegrationCore2.
inline ScratchString canonicalPathListScratch(Memory& mb, StrSpan list, NameId scope,
            ScratchArena& arena);

public:
/// @brief Mark, under one marked key, the instructions whose path list
        ///        refused a parked source (D-529).
        ///
        /// @details
        /// Called when an expansion parks a witness under @p markedExpr in an
        /// induction step block. Every instruction stored under the keys that
        /// canonically match @p markedExpr at @p validityName (owned there or at a strict ancestor) and refuses the
        /// route (`witnessLinkRefuses` with @p linked) records the refusal mark
        /// of @p source (`refusalMarkScratch`) in its payload. A replay probe
        /// (`integrationDemandOpen`) then no longer counts that instruction as
        /// a demand for the source, so the park-time rendezvous cannot replay
        /// a source the same instructions refuse again; a new or unmarked
        /// instruction still wakes it.
        ///
        /// @param mb           Owning block (single-threaded seam).
        /// @param markedExpr   The park key, bare form.
        /// @param validityName The park scope (registered).
        /// @param linked       The route's linked arguments (null when
        ///                     @p linkedCount is 0).
        /// @param linkedCount  Number of @p linked.
        /// @param source       The parked source existence.
        /// @param history      The route's witness-history probe, or null
        ///                     (D-522); @p linkedCount
        ///                     is positive or @p history is active.
        /// @return Void.
        /// @invariant Writes only payload values of refusing instructions.
        /// @see integrationDemandOpen, witnessLinkRefuses.
inline void markWitnessLinkRefusals(Memory& mb, StrSpan markedExpr, StrSpan validityName,
            const StrSpan* linked, int32_t linkedCount, StrSpan source,
            const WitnessHistoryProbe* history = nullptr);

public:
/// @brief Whether an integration instruction under a marked key still
        ///        demands a parked source (D-529).
        ///
        /// @details
        /// The replay twin of the route-3 probe in an induction step block: a
        /// key counts as a demand for @p source only if one instruction under
        /// the keys that canonically match @p markedExpr at @p validityName (owned there or at a strict ancestor)
        /// carries no refusal mark of @p source. Outside step blocks it is the
        /// plain probe.
        ///
        /// @param mb           Owning block.
        /// @param markedExpr   The park key, bare form.
        /// @param validityName The demanding scope.
        /// @param source       The parked source existence.
        /// @return True iff an unmarked instruction is stored under the key.
        /// @invariant Mints nothing into the block's interners.
        /// @see markWitnessLinkRefusals, RejectedSourceReplay::queueParkedExistence.
inline bool integrationDemandOpen(Memory& mb, StrSpan markedExpr, StrSpan validityName,
            StrSpan source);

public:
/// @brief The payload value recording that an instruction refused a source
        ///        (D-529).
        ///
        /// @details `refused:` followed by the source in canonical form at
        /// @p scope, so a later class merge leaves the mark comparable.
        ///
        /// @param mb     Owning block.
        /// @param source The source existence.
        /// @param scope  The scope (registered).
        /// @param arena  String-tier scratch arena receiving the text.
        /// @return The mark text.
        /// @invariant Mints nothing into the block's interners.
        /// @see markWitnessLinkRefusals, integrationDemandOpen.
inline ScratchString refusalMarkScratch(Memory& mb, StrSpan source, NameId scope,
            ScratchArena& arena);

public:
/// @brief Whether one element of an and gates an integration witness's
        ///        admission (D-481).
        ///
        /// @details
        /// The integration twin of the implication-subproof gate element
        /// (`ImplicationSubproofs::isSubproofGateElement`): a non-implication
        /// element — atomic, negated or compound — whose every argument is
        /// fixed (`u_`-prefixed) or is @p variable, the argument the candidate
        /// witness fills: the `marker` on the witness's own route, the bound
        /// variable of the existence it fills on a coadmission route. Once the
        /// witness stands in for that argument the element is a ground
        /// statement that can become known. An implication element — a
        /// numbered `(implication<N>[…])` or a configured definition whose
        /// whole body is one implication (D-467) — is integrated through its
        /// own subproof and never gates; an element with any other argument
        /// (another bound variable, a `pi_` pattern, the marker on a
        /// coadmission route) is not a statement about the witness and never
        /// gates.
        ///
        /// @param element     One element of an and entity of a marked
        ///                    integration instruction (flat MPL, at least one
        ///                    argument).
        /// @param variable    The argument the candidate witness fills.
        /// @param definitions The compiled registry naming the configured
        ///                    implication definitions.
        /// @return True iff @p element gates the witness's admission.
        /// @invariant Pure; allocates nothing.
        /// @see integrationTrialGates, ImplicationSubproofs::isSubproofGateElement,
        ///      CompiledDefinitionRegistry::isConfiguredImplication.
static bool isIntegrationTrialGateElement(StrSpan element, StrSpan variable,
            const CompiledDefinitionRegistry& definitions);

public:
/// @brief Collect the gates of every new name one integration admission
        ///        creates (D-481,
        ///        D-482).
        ///
        /// @details
        /// Reads every instruction stored under the `admissionMapIntegration`
        /// keys that canonically match @p markedExpr at @p validityName (owned there or at a strict ancestor) — the
        /// keys the route-3 probe of `isAdmittedIntegration` finds — and takes
        /// in each the existence entities that list the matched key, directly
        /// or through an and on one of their sides.
        ///
        /// The matched name @p matched stands for the marker. Its gates are the
        /// gate elements (`isIntegrationTrialGateElement` for the `marker`) of
        /// the ands on the sides of those existences, nested ands included. The
        /// matched key is the admission premise, never a gate: an element equal
        /// to it is skipped and an and entity equal to it is not walked.
        ///
        /// Every other new name of the admission (@p others — the names on the
        /// admitted other sides, enclosing or nested) takes on the gates of its
        /// place in the instruction. A text fills a variable when its one
        /// argument that is neither fixed nor the marker is that variable and
        /// the text, with the marker read as @p matched and the variable as the
        /// name, is one of @p facts. Beside: an existence on a side of the
        /// key's existence, directly or inside an and below a side, one of
        /// whose sides fills its bound variable; the gates are the gate
        /// elements for that variable of the existence's other and sides.
        /// Above: the key's existence or an entity enclosing it, one of whose
        /// texts (signature, side, element of an and below a side; the key
        /// excluded) fills a variable; the gates are the gate elements for that
        /// variable of that entity's ands. The filling text and the matched key
        /// are never gates. A name without a place has no gates.
        ///
        /// Gate elements become ground statements by replacing the filled
        /// argument with the name and stripping the `u_` prefixes. One naming a
        /// new name is unknown by construction, and a fixed one counts only
        /// while it is not known at the scope (`elementKnownAtOrAbove`). An
        /// instruction with no unknown gate among all the names is ungated.
        ///
        /// With @p record null the call is a pure probe and mints nothing. With
        /// a record it appends one group per gated instruction — the gates of
        /// all the names encoded into the NameMap, in instruction order; a
        /// group equal in gates to an existing one (the same instruction filed
        /// under another key) is not appended again.
        ///
        /// @param mb           Owning block.
        /// @param markedExpr   Marked constituent of @p matched, bare form (the
        ///                     marked text the route-3 probe receives).
        /// @param validityName Registered scope of the admission.
        /// @param matched      Name of the witness the key matched; it stands
        ///                     for the marker. Materialized when @p record is
        ///                     set.
        /// @param others       The admission's other new names (null when
        ///                     @p otherCount is zero); materialized when
        ///                     @p record is set.
        /// @param otherCount   Number of @p others.
        /// @param facts        The constituents of all the admission's new
        ///                     names, in their spelling (null when @p factCount
        ///                     is zero).
        /// @param factCount    Number of @p facts.
        /// @param record       Optional out: groups and gates appended in place.
        /// @param linked       The route's linked arguments: an instruction
        ///                     the witness-link gate refuses is skipped
        ///                     (D-529). Null when
        ///                     @p linkedCount is 0.
        /// @param linkedCount  Number of @p linked.
        /// @param history      The route's witness-history probe, or null: an
        ///                     instruction its derivation tree refuses is
        ///                     skipped as well (D-522).
        /// @return True iff at least one instruction matched and every matched
        ///         instruction is gated.
        /// @invariant Group and gate capacities are asserted, never truncated.
        /// @see isAdmittedIntegration, integrationEnclosingFill,
        ///      DisintegrationProcessor::disintegrateExpr2,
        ///      EquivalenceQueries::elementKnownAtOrAbove.
inline bool integrationTrialGates(Memory& mb,
            StrSpan markedExpr,
            StrSpan validityName,
            StrSpan matched,
            const StrSpan* others,
            int32_t otherCount,
            const StrSpan* facts,
            int32_t factCount,
            IntegrationTrialRec* record,
            const StrSpan* linked = nullptr, int32_t linkedCount = 0,
            const WitnessHistoryProbe* history = nullptr);

/// The enclosing witness fills no level of the matched instruction.
static constexpr int32_t kEnclosingFillNone = 0;
/// The enclosing witness fills a variable at or above the matched key's
/// existence: it leads up to the nested witness's admission.
static constexpr int32_t kEnclosingFillAbove = 1;
/// The enclosing witness fills the bound variable of an existence on a side
/// of the matched key's existence: it results from the nested admission.
static constexpr int32_t kEnclosingFillBeside = 2;

public:
/// @brief Which level of the matched integration instruction an enclosing
        ///        witness fills when a witness nested in it is admitted under
        ///        an integration key (D-482).
        ///
        /// @details
        /// A nested witness @p descendant was matched by a key of
        /// `admissionMapIntegration` (route 3) through the marked constituent
        /// @p markedExpr. Its admission may carry an enclosing witness of the
        /// same source along only when that witness has a place in the matched
        /// instruction. Reads every instruction stored under the keys that
        /// canonically match @p markedExpr at @p validityName (owned there or at a strict ancestor) and takes in each
        /// the existence entities that list the key, directly or through an
        /// and on one of their sides.
        ///
        /// A text of the instruction fills a variable when its one argument
        /// that is neither fixed (`u_`) nor the marker is that variable and the
        /// text, with the marker read as @p descendant and the variable as
        /// @p enclosing and the `u_` prefixes stripped, is one of @p facts.
        ///
        /// Above (`kEnclosingFillAbove`): a text of the key's existence or of an
        /// entity enclosing it — its signature, a side, or an element of an and
        /// below a side, other than the key — fills a variable. Such a variable
        /// is bound before the key's existence (a goal's output, an enclosing
        /// existence's witness), so the enclosing witness leads up to the
        /// nested admission.
        ///
        /// Beside (`kEnclosingFillBeside`): a side of an existence entity that
        /// stands on a side of the key's existence, directly or inside an and
        /// below a side, fills that existence's bound variable. The enclosing witness is then constructed after the key's
        /// witness (the gates it takes on are `integrationTrialGates` on a
        /// coadmission route).
        ///
        /// Above wins over beside in any matched instruction; with no fill the
        /// result is `kEnclosingFillNone`.
        ///
        /// @param mb           Owning block.
        /// @param markedExpr   Marked constituent of @p descendant, bare form
        ///                     (the marked text the route-3 probe receives).
        /// @param validityName Registered scope of the admission.
        /// @param enclosing    Name of the enclosing witness in @p facts.
        /// @param descendant   Name of the nested witness in @p facts; it stands
        ///                     for the marker.
        /// @param facts        The constituents of the witnesses from
        ///                     @p descendant up to @p enclosing, in their
        ///                     spelling.
        /// @param factCount    Number of @p facts (positive).
        /// @param linked       The route's linked arguments: an instruction
        ///                     the witness-link gate refuses is skipped
        ///                     (D-529). Null when
        ///                     @p linkedCount is 0.
        /// @param linkedCount  Number of @p linked.
        /// @param history      The route's witness-history probe, or null: an
        ///                     instruction its derivation tree refuses is
        ///                     skipped as well (D-522).
        /// @return `kEnclosingFillNone`, `kEnclosingFillAbove` or
        ///         `kEnclosingFillBeside`.
        /// @invariant Pure probe: mints nothing and writes no record.
        /// @see integrationTrialGates, isAdmittedIntegration,
        ///      DisintegrationProcessor::disintegrateExpr2, I-252.
inline int32_t integrationEnclosingFill(Memory& mb,
            StrSpan markedExpr,
            StrSpan validityName,
            StrSpan enclosing,
            StrSpan descendant,
            const StrSpan* facts,
            int32_t factCount,
            const StrSpan* linked = nullptr, int32_t linkedCount = 0,
            const WitnessHistoryProbe* history = nullptr);

public:
/// @brief Whether a route-5 admission of a nested witness reaches each of
        ///        its unadmitted enclosing witnesses through the facts that
        ///        link them (D-501).
        ///
        /// @details
        /// A double-marker key carries no instruction, so an enclosing witness
        /// has no place in one (`integrationEnclosingFill` is route 3's
        /// test). It comes along with the nested witness @p descendant when
        /// @p facts link it to that witness: starting from @p descendant, a
        /// fact one of whose arguments is a reached name reaches every
        /// enclosing name among its arguments, until a pass reaches nothing
        /// new. In G3's step block D1's innermost constituent `in3[r,p,b,+]`
        /// reaches `r`, and `t`'s constituent `and30[q,b,t,*,r,a,+,N]` then
        /// reaches `t` and `q`. Only the witness names link: a fixed argument
        /// shared by two facts is never one of the names.
        ///
        /// @param descendant     Name of the nested witness in @p facts.
        /// @param enclosing      Names of the unadmitted enclosing witnesses,
        ///                       from the nested witness's parent outwards.
        /// @param enclosingCount Number of @p enclosing (1 to
        ///                       `MAX_INSTRUCTION_ELEMENTS`, asserted).
        /// @param facts          The constituents of @p descendant and of the
        ///                       enclosing witnesses, in their spelling.
        /// @param factCount      Number of @p facts (positive).
        /// @return True iff every enclosing witness is reached.
        /// @invariant Pure: reads the spans only and allocates nothing.
        /// @see integrationEnclosingFill, isAdmittedIntegrationDouble,
        ///      DisintegrationProcessor::disintegrateExpr2.
static bool doubleMarkerEnclosingLinked(StrSpan descendant, const StrSpan* enclosing,
            int32_t enclosingCount, const StrSpan* facts, int32_t factCount);

private:
/// @brief Disintegration-side admission check — gates whether a
        /// disintegration head may fire from the given LB.
        ///
        /// @details
        /// Walks `mb.overallHashMemory.admissionMap` for the
        /// `(expression, validity)` pair, looks up the matching
        /// `AdmissionMapValue`s, and checks each against the current
        /// depth / secondary-iteration budget. Returns true iff at
        /// least one admission slot is alive and all sibling premises
        /// of that slot are present in `mb.intLocalEncodedStatementsSet`.
        ///
        /// Companion to `isAdmittedIntegration` for the disintegration
        /// path. The two share the `admissionStatusMap` cache; revival
        /// from the rejection map (per
        /// [I-22](../../docs/agentic_swdd/30_invariants.md#i-22)) does not clean
        /// the admission entry.
        ///
        /// @param mb           Owning LB.
        /// @param expr         Constituent expression (span source; read-only).
        /// @param var          Witness/replacement variable (span source).
        /// @param markedExpr   Marked template form (span source).
        /// @param validityName Validity scope (span source).
        /// @return True iff the disintegration head fires on this LB.
        /// @see `isAdmittedIntegration` — integration-side counterpart.
        /// @see [I-22](../../docs/agentic_swdd/30_invariants.md#i-22).
        /// @param metrics Temporary-witness budgets, or null for permanent text.
        /// @param commit Apply successful demand updates; false reads scalar
        /// record fields directly and creates no update payload or string copies.
        /// @param inductionAllowed Whether a recursion-owned value (flag set)
        /// may admit: false when the original expression being disintegrated
        /// has an argument that is not a number (`hasOnlyNumberArguments`); an
        /// ordinary value admits either way
        /// (I-285).
        /// @param inductionScopes Receives, on a commit, the induction scopes
        /// the induction hypothesis's values opened (flag
        /// `kAdmissionInductionHypothesisFlag`; a recursion-owned value without
        /// it opens none): each element of such a chain the constituent fills
        /// is one eat of the hypothesis, opens
        /// `<comparison scope>_boundary_is_<chain with marker>_<var>` and files
        /// the advanced chain there; required on a commit that eats. An eat of
        /// the hypothesis whose marked chain already names an induction scope
        /// on the comparison scope's path is refused, probe and commit alike
        /// (D-519).
inline bool isAdmitted(Memory& mb,
            StrSpan expr,
            StrSpan var,
            StrSpan markedExpr,
            StrSpan validityName,
            const WitnessAdmissionMetrics* metrics = nullptr,
            bool commit = true,
            int64_t selectedKey = 0, NameId comparisonScope = 0,
            bool inductionAllowed = true,
            InductionScopeSink* inductionScopes = nullptr);

public:
/// @brief Ancestor-inclusive twin of @ref isAdmitted — probe the
        ///        exact deposit validity first, then every strict ancestor
        ///        scope (D-288).
        ///
        /// @details
        /// A demand key at an ancestor scope demands the product for every
        /// descendant, and a fired (instantiated) key lands at
        /// `deeperOf(subkey constituents)` — for the C8 shape, `main`,
        /// while the witness-minting disintegration deposits inside the
        /// released `_ordis_` branch. The exact-validity probe alone can
        /// therefore never connect the two; this twin closes the gap. The
        /// exact probe runs first (unchanged semantics when it hits), then
        /// the strict ancestors in `strictAncestorSpans`' deterministic
        /// `compareSpans` order, first hit winning. Each hit runs the full
        /// @ref isAdmitted interior against the ANCESTOR validity — the
        /// replKey propagation addresses the key that actually admitted. The
        /// ancestor names are `decodeView` spans copied onto the per-slot
        /// string scratch BEFORE any probe (I-3: an admitting probe's
        /// interior mints).
        ///
        /// @param mb           Owning LB.
        /// @param expr         Constituent expression (span source).
        /// @param var          Witness/replacement variable (span source).
        /// @param markedExpr   Marked template form (span source).
        /// @param validityName Deposit validity scope (span source).
        /// @return True iff the disintegration head is admitted at the
        ///         deposit validity or any strict ancestor of it.
        /// @invariant Misses mint nothing (the walk is non-minting probe
        ///            chains); at most one validity's key admits and is
        ///            consumed.
        /// @see isAdmitted — the exact-validity interior;
        ///      `NameMap::strictAncestorSpans` — the ancestor enumeration.
        /// @param metrics Temporary-witness budgets, or null for permanent text.
        /// @param commit Apply successful demand updates; false is a pure probe.
        /// @param inductionAllowed Whether recursion-owned values may admit;
        /// forwarded to every @ref isAdmitted probe
        /// (I-285).
        /// @param inductionScopes Receives the induction scopes a commit
        /// opened; forwarded to every @ref isAdmitted probe, compared at
        /// @p validityName (D-519).
inline bool isAdmittedIncludingAncestors(Memory& mb,
            StrSpan expr,
            StrSpan var,
            StrSpan markedExpr,
            StrSpan validityName,
            const WitnessAdmissionMetrics* metrics = nullptr,
            bool commit = true,
            bool inductionAllowed = true,
            InductionScopeSink* inductionScopes = nullptr);

public:

/// @brief Fill the integration demands a statement deposited at `main`, at
///        an int scope or in a twin successor case satisfies: each distinct
///        argument in turn becomes the marker, and every instruction under a
///        matching key owned at the deposit scope records the value and files
///        its continuation.
///
/// @details
/// The statement fill of route 3. An instruction copy bound to one premise
/// refuses the fill when the expression whose deposit produced the statement
/// names a forbidden variable or a forced witness of a forbidden level
/// (@p arrivingVariables / @p arrivingLevels, the masks
/// `signatureTokenSets` computes; D-579).
/// At `main` such a copy is a root copy: it also refuses unless every free
/// argument of that expression is a number or `rec`, and an accepted fill
/// opens the copy's int scope (`openIntScope`), where the copy continues. At
/// an int scope the scope's own keys fill in place
/// (D-578). In the weak step block's twin successor
/// case (`Memory::twinCaseScopeId`) only the copies bound to the second twin
/// premise fill, as root copies with `rec2` for `rec`
/// (D-589).
///
/// @param expression          The deposited statement (any lifetime; copied).
/// @param mb                  The owning LB.
/// @param validityName        The deposit scope; only `main`, int scopes and
///                            the twin successor case fill.
/// @param arrivingVariables   Decimal variables the arriving expression's
///                            signature names, as a mask (0 = none).
/// @param arrivingLevels      Forced-witness levels it names, as a mask (0 = none).
/// @param arrivingNumbersOnly Whether every free argument of the arriving
///                            expression is a number or `rec`
///                            (`hasOnlyNumberFreeArguments`).
/// @param arrivingTwinNumbersOnly Whether every free argument of the arriving
///                            expression is a number or `rec2`, the name of
///                            the twin partner's predecessor: the gate of a
///                            copy bound to the second twin premise
///                            (D-589); unread in a block
///                            without one.
/// @return Void.
/// @see isAdmittedIntegration, signatureTokenSets, premiseCopyRefuses.
void updateAdmissionMapIntegration(StrSpan expression,
            Memory& mb,
            StrSpan validityName,
            uint64_t arrivingVariables,
            uint64_t arrivingLevels,
            bool arrivingNumbersOnly,
            bool arrivingTwinNumbersOnly = true);

public:
/// @brief Replays the per-burst staged algebra-`admissionMap` writes
    ///        recorded on `Memory::admissionKeysAlgebra`, once, after the
    ///        hashburst fixpoint loop.
    ///
    /// @details
    /// Companion drain to the record append in
    /// `checkLocalEncodedMemoryStatic`'s marker branch. For each
    /// `AdmissionKeyAlgebraRecord` on `memoryBlock.admissionKeysAlgebra`, in
    /// firing order, it replays the four writes the marker branch used to
    /// perform inline mid-burst:
    ///   1. insert `record.value` into `overallHashMemory.admissionMap` under
    ///      `record.key` (fresh per-record lookup; create the set if absent);
    ///   2. set `admissionStatusMap[record.key] = false`;
    ///   3. add every non-`marker` argument of `record.key.original` to
    ///      `varsInAdmissionMapKeys`;
    ///   4. fire `revisitRejected2`, depositing any revival cohort on
    ///      `sameIterationInternalMail`.
    /// Called once from `performElemPhase2`, immediately before phase 3's
    /// post-burst `standardProcessing`, so the revival cohorts are present for
    /// that absorb and `admissionMap` is current before its `isAdmitted` reads
    /// it.
    ///
    /// @param memoryBlock The LB whose `admissionKeysAlgebra` buffer is
    ///        replayed and whose `overallHashMemory` admission containers
    ///        receive the writes. The buffer is left intact; the burst-start
    ///        clear in `performElem2` empties it.
    /// @return (void)
    /// @invariant Drains in append (firing) order.
    /// @invariant FROZEN interner-touch sequence per record: (1)
    ///            `templateInterner.encode(key.original)` inside
    ///            `mintTemplateKey`, (2) `nameMap.encode(key.validityName)`
    ///            inside `mintTemplateKey`, (3) `valueInterner.encode` per
    ///            staged key element in RULE order inside `stagedToArenaBlob`,
    ///            (4) `valueInterner.encode` per staged remainingArg in
    ///            presorted LEX order inside `stagedToArenaBlob` (the same
    ///            positions the retained `stagedToIdValue` oracle touches),
    ///            (5)
    ///            `templateInterner.encode` per non-`"marker"` argument of
    ///            `key.original` in `getArgs` order, (6) `revisitRejected2`'s
    ///            interior mints. Same bytes at the same positions →
    ///            identical ids → identical deload streams.
    /// @see `memory.cpp::checkLocalEncodedMemoryStatic` — the producer.
    /// @see `AdmissionKeyAlgebraRecord` — the staged record type.
void drainAdmissionKeysAlgebra(Memory& memoryBlock);

public:
/// @brief Replays the per-burst staged integration-template registrations
    ///        recorded on `Memory::deferredIntegrationPreps`, once, after the
    ///        hashburst fixpoint loop.
    ///
    /// @details
    /// Integration-side companion of `drainAdmissionKeysAlgebra`. For each
    /// `DeferredIntegrationPrep` on `memoryBlock.deferredIntegrationPreps`, in
    /// firing order, it replays the `prepareIntegration` call that
    /// `checkLocalEncodedMemoryStatic`'s marker branch deferred — registering the
    /// integration admission template (`admissionMapIntegration` plus the
    /// `varsInAdmissionMapIntegrationKeys` cache, via `prepareIntegrationCore2`).
    /// A record whose missing premise has a uniquely definable open argument
    /// already named by a statement known at the firing scope or an ancestor
    /// is not replayed (`IntegrationProcessor::knownVersionSettlesMarkedPremise`,
    /// D-568).
    /// The sealed views feed the `prepareIntegration` span core directly —
    /// no set rebuild and no string materialization at this boundary; the
    /// core's two compiled-definition edge strings are built inside it, past
    /// its early-return gates. Called once from `performElemPhase2`,
    /// immediately before phase 3's post-burst `standardProcessing`, so the
    /// templates are registered before that absorb's
    /// `updateAdmissionMapIntegration` cascade consumes them.
    ///
    /// @param memoryBlock The LB whose `deferredIntegrationPreps` buffer is
    ///        replayed; its `overallHashMemory` integration-admission containers
    ///        receive the writes. The buffer is left intact; the burst-start
    ///        clear in `performElem2` empties it.
    /// @return (void)
    /// @invariant Drains in append (firing) order.
    /// @see `memory.cpp::checkLocalEncodedMemoryStatic` — the producer.
    /// @see `DeferredIntegrationPrep` — the staged record type.
void drainDeferredIntegrationPreps(Memory& memoryBlock);

public:
/// @brief Drain the per-burst ordis2 demand records into
        ///        `admissionMapOrdis2` and wake the or heads filed in
        ///        `rejectedMapOrdis2` under the same keys
        ///        (D-267).
        ///
        /// @details
        /// Third sibling of `drainAdmissionKeysAlgebra` /
        /// `drainDeferredIntegrationPreps` at the same post-fixpoint seam
        /// (`performElemPhase2`, fixed call order after the integration
        /// drain). Per record, a FROZEN interner-touch sequence:
        /// (1) `mintTemplateKey(expression, validity)` — the packed demand
        /// key; (2) `insertAdmissionOrdis2IdsBlob` — the canonical RMW into
        /// `admissionMapOrdis2` (re-demands dedup); (3)
        /// `revisitRejectedOrdis2(expression, ...)` — the pair's key
        /// rendezvous: `rejectedMapOrdis2` is keyed by disjunct clean
        /// GROUND texts, the demand's own key language, so an or filed
        /// under a byte-equal text is un-known and re-deposited on
        /// `sameIterationInternalMail`; the post-burst absorb re-runs the
        /// opener, whose route (c) reads the fresh demand entry and opens
        /// (consuming it). Never a mid-burst map write — the records were
        /// staged in canonical firing order by `applyFiringRecords`
        /// (D-117). The OLD `rejectedMapOrdis` is never probed here
        /// (different key language — the 3X mis-design this pair fixes).
        /// A record of a goal near-miss (`Ordis2DemandRecord::goalNearMiss`)
        /// is no OR demand: it goes, in the same order, to
        /// `IntegrationProcessor::serveGoalNearMiss`, the goal-driven rule
        /// search, and touches neither map
        /// (D-574).
        ///
        /// @param memoryBlock The LB whose `admissionKeysOrdis2` buffer is
        ///        replayed; its `overallHashMemory.admissionMapOrdis2` and
        ///        `rejectedMapOrdis2` receive the writes. The caller
        ///        (`performElemPhase2`) clears the buffer post-drain.
        /// @return (void)
        /// @invariant Drains in append (canonical firing) order;
        ///            single-threaded seam only (I-83).
        /// @see `Ordis2DemandRecord` — the staged record type.
        /// @see `consumeOrLeavesCohort` — route (c), the opener-side reader.
        /// @see `IntegrationProcessor::serveGoalNearMiss` — the near-miss
        ///      reader.
void drainAdmissionKeysOrdis2(Memory& memoryBlock);

/// @brief File one missing expression of the induction hypothesis as a
        ///        ground OR demand at its scope.
        ///
        /// @details
        /// GL sees a missing expression in the induction hypothesis: at `main`
        /// a fully specified premise of the installed hypothesis, in an
        /// induction scope a premise the eats up to that scope left fully
        /// specified. When the premise is not known at the scope or an
        /// ancestor, it is a reason to open an OR one of whose disjuncts it is
        /// (D-575). The reason is written in the
        /// ordis2 demand language: the one record a firing's demand would
        /// stage, replayed like `drainAdmissionKeysOrdis2` replays it —
        /// `mintTemplateKey`, the canonical RMW insert into
        /// `admissionMapOrdis2` with the hypothesis rule as source and no
        /// levels, then the `revisitRejectedOrdis2` wake of an OR already
        /// parked under the key. A later OR arrival meets the demand through
        /// the route-(c) probe of `consumeOrLeavesCohort`, which opens the
        /// cohort with this disjunct as its first case. A premise outside the
        /// park-key language (`ordis2KeyEligible`: compiled, non-atomic, at
        /// least four arguments) can be no disjunct the pair rendezvous meets
        /// and files nothing; a known premise is no missing expression and
        /// files nothing (the D-358 rule of the firing producers).
        ///
        /// @param mb           The step block.
        /// @param element      The fully specified premise, ground text.
        /// @param validityName The scope the hypothesis misses it at (interned).
        /// @param sourceImplId The hypothesis rule (`isInductionHypothesisRule`).
        /// @return (void)
        /// @invariant Single-threaded per block (a worker's claim or the grid
        ///            build); writes `admissionMapOrdis2` and, through the
        ///            wake, the internal mail, nothing else.
        /// @see drainAdmissionKeysOrdis2, CaseProcessor::consumeOrLeavesCohort,
        ///      RejectedSourceReplay::revisitRejectedOrdis2.
void fileHypothesisOrDemand(Memory& mb, StrSpan element, StrSpan validityName, int32_t sourceImplId);

/// @brief File the missing expressions of a freshly installed induction
        ///        hypothesis at `main`.
        ///
        /// @details
        /// At `main` no eat has happened: the hypothesis's missing expressions
        /// are its premises with only fixed arguments, the `u_` names of its
        /// instruction form (D-575). Each one is
        /// written as a ground statement (`groundHypothesisPremisesScratch`)
        /// and offered to @ref fileHypothesisOrDemand at `main`. The step
        /// block is new, so no OR is parked there yet.
        ///
        /// @param mb           The step block, just after the install.
        /// @param premises     The hypothesis premises, instruction form.
        /// @param premiseN     Number of premises.
        /// @param sourceImplId The installed hypothesis rule.
        /// @return (void)
        /// @invariant Called at the grid build, once per installed rule.
        /// @see fileHypothesisOrDemand, ScopeSyntax::groundHypothesisPremisesScratch.
void fileHypothesisDemandsAtMain(Memory& mb, const StrSpan* premises, int32_t premiseN, int32_t sourceImplId);

/// @brief Demand the successor of a forced witness of a strong step block at
        ///        the scope that minted it.
        ///
        /// @details
        /// A forced witness of the strong step block (the induction head's or
        /// a seed's forced disintegration) whose definition set is `(1)` is a
        /// number the induction apparatus works with, such as the length of
        /// the hypothesis's factorization; extending what the hypothesis gave
        /// needs its successor. The successor function is the third argument
        /// of the block's recursion hypothesis `(in2[rec,n,s])`. The demand
        /// `(in2[w,marker,s])` is written through `updateAdmissionMap` at
        /// @p validityName, the scope the witness was minted at, so only
        /// disintegrations at that scope or below it read it
        /// (`isAdmittedIncludingAncestors`). The chain handed over is
        /// `(in2[w,f,s])` with fixed arguments w and s and a free number f:
        /// `updateAdmissionMap` puts the marker in place of the one argument
        /// outside the fixed run, as for every regular chain, and the
        /// canonical renaming at a scope with equivalence classes accepts
        /// only fixed and numbered arguments. The `revisitRejected2` wake then
        /// replays a successor existence already parked under the key
        /// (D-571).
        ///
        /// @param mb           The strong step block.
        /// @param witness      The forced witness's permanent name.
        /// @param validityName The scope the witness was minted at.
        /// @return (void)
        /// @invariant Single-threaded per block (a worker's claim). Asserts the
        ///            strong step block and its in2 recursion hypothesis.
        /// @see updateAdmissionMap, RejectedSourceReplay::revisitRejected2.
void fileForcedSuccessorDemand(Memory& mb, StrSpan witness, StrSpan validityName);

/// @brief Demand, in a base-case block's released successor case, the
        ///        partner's values of the functions the second twin premise
        ///        names.
        ///
        /// @details
        /// A base-case block of a twin conjecture refutes the successor case
        /// l = s(q) of its partner l (D-588).
        /// The refutation needs the value at l of a function the second twin
        /// premise is about, Q's last entry `(in2[l,x,Q])` for the full I5,
        /// whose existence is known at `main` but parked under
        /// `(in2[l,marker,Q])` while nothing demands it. Every premise of the
        /// block that holds l is read (`ancestorPremiseKeyView`); each of its
        /// arguments X that is neither l nor an anchor argument and whose
        /// definition set there is the one `in2` gives its function slot
        /// (`ScopeTypeResolver::definitionSetOf`) is demanded at
        /// @p scopeId: `updateAdmissionMap` with the chain `(in2[l,f,X])`,
        /// fixed arguments l and X and a free number f, as
        /// `fileForcedSuccessorDemand` writes its successor demand, then the
        /// `revisitRejected2` wake at that scope, which replays a source
        /// parked at an ancestor into it. The block records that it filed
        /// (`Memory::twinValueDemandsFiled`).
        ///
        /// @param mb      The base-case block (single-threaded seam).
        /// @param scopeId The released successor case (`Memory::twinCaseScopeId`,
        ///                asserted).
        /// @return (void)
        /// @invariant Filed once per block (asserted); a premise of the block
        ///            holds the partner (asserted).
        /// @see fileForcedSuccessorDemand, updateAdmissionMap,
        ///      RejectedSourceReplay::revisitRejected2,
        ///      StatementMemory::addStatement.
void fileTwinValueDemands(Memory& mb, NameId scopeId);

/// @brief Whether every argument of a premise is one of the fixed arguments.
        ///
        /// @details
        /// An admission chain stores its fixed arguments as a sorted,
        /// duplicate-free run (`remainingArgs`); a premise whose arguments all
        /// lie in it names no variable any more — the fully bound premise
        /// `updateAdmissionMap` prepares for integration and the hypothesis may
        /// miss (D-575).
        ///
        /// @param element     The premise, `(...)` or `!(...)`.
        /// @param fixedSorted The fixed arguments, `compareSpans`-ascending
        ///                    and unique.
        /// @param fixedN      Number of fixed arguments.
        /// @return `true` iff each argument of @p element is in the run.
        /// @invariant Pure; allocates nothing.
        /// @see updateAdmissionMap, fileHypothesisOrDemand.
static bool fullyBoundElement(StrSpan element, const StrSpan* fixedSorted, int32_t fixedN);

public:
/// @brief Seed the recursion admission budget for a freshly-registered
///        operator GOAL at the scope that owns the goal's numeral.
///
/// @details
/// Sole caller: `addExprToMemoryBlock`'s `status == 2` (toBeProved) branch,
/// when the goal's head operator has input args. The goal `(op[…,X])` is an
/// operator output to be PROVED; the admission budget seeded here is what lets
/// the hash engine unfold that operator backward while chasing the goal, capped
/// by `maxAdmissionDepth` / `maxSecondaryNumber`. The back-climb finds the first
/// ancestor LB whose `exprKey` shares a digit argument with the goal — the
/// recursion/fold block that introduced that numeral — and seeds the admission
/// there (delegating to `updateAdmissionMap`). A self-match seeds the calling
/// LB. This is goal-side backward-admission setup, NOT result aggregation.
///
/// I-28 detect-and-defer (trial): when the climb lands on a STRICT ANCESTOR and
/// the caller is a parallel phase-1/phase-3 worker (`g_inParallelWorkerPhase`),
/// writing the ancestor would be a cross-LB write that races sibling recursion
/// children sharing one fold ancestor. In that case the resolved
/// `updateAdmissionMap` call is sealed onto the shared `deferredAncestorPages`
/// `SealedPageSet` record chain and replayed single-threaded by
/// `drainDeferredAncestorAdmissions` post-join. A
/// self-write (always the worker's own LB) and any write outside the parallel
/// sweep (grid build, the drain itself) stay inline.
///
/// @param expr               Operator goal whose admission budget is seeded.
/// @param memoryBlock        LB the goal was registered in (climb start).
/// @param maxAdmissionDepth  Recursion admission depth cap.
/// @param maxSecondaryNumber Recursion secondary-iteration cap.
/// @param partOfRecursion    Marks the seeded entry as recursion-owned.
/// @invariant [I-28](../../docs/agentic_swdd/30_invariants.md#i-28) — a
///            cross-LB ancestor seed never lands inline in a parallel worker.
/// @see `drainDeferredAncestorAdmissions`, `g_inParallelWorkerPhase`,
///      `updateAdmissionMap`.
inline void updateAdmissionMap3(StrSpan expr,
            Memory& memoryBlock,
            int maxAdmissionDepth,
            int maxSecondaryNumber,
            bool partOfRecursion);

private:
/// @brief Element-wise byte-lexicographic compare of two
        ///        `SealedSpan<SealedString>` runs — the `std::vector`/`std::set`
        ///        `operator<` twin over sealed strings.
        ///
        /// @details
        /// Three-way (`<0` / `0` / `>0`) comparison identical to
        /// `std::lexicographical_compare` over the decoded `std::string`
        /// elements: element `k` compared by `compareSpans` (byte-lex, ==
        /// `std::string::compare`), shorter-run-is-prefix on a common prefix
        /// (`std::vector`/`std::set` `operator<` semantics). One helper serves
        /// both the `key` (vector) and `remainingArgs` (set) runs — both are
        /// element-wise lex.
        ///
        /// @param a Left run.
        /// @param b Right run.
        /// @return Negative when `a < b`, 0 when equal, positive when `a > b`.
        /// @see deferredAncestorAdmissionLess.
static int sealedStringRunLexCompare(const SealedSpan<SealedString>& a,
                                             const SealedSpan<SealedString>& b);

private:
/// @brief Deterministic replay order of two deferred ancestor admission
        ///        records — the byte-twin of the retired drain sort's 6-field
        ///        chain.
        ///
        /// @details
        /// Three-way compare returning at the first non-zero field, in order:
        ///  1. `ancestor->exprKeyView()` (byte-lex; `exprKeyView` bytes ==
        ///     `exprKey()`, skeletonInterner never deloads). exprKey is NOT
        ///     unique across recursion sub-LBs, so the chain continues.
        ///  2. `key` run (`sealedStringRunLexCompare`; the `std::vector<std::string>`
        ///     twin).
        ///  3. `remainingArgs` run (`sealedStringRunLexCompare`; the
        ///     `std::set<std::string>` twin).
        ///  4. `partOfRecursion` (false < true).
        ///  5. `maxAdmissionDepth`.
        ///  6. `maxSecondaryNumber`.
        /// Two fully-equal records tie (returns 0) → `std::sort` leaves them in
        /// input (append) order, exactly as the retired code did (identical
        /// records replay to the same admission state; not a regression).
        ///
        /// @param a Left record.
        /// @param b Right record.
        /// @return Negative when `a` sorts before `b`, 0 when equal, positive
        ///         otherwise.
        /// @invariant Byte-identical replay order to the retired heap sort
        ///            (I-107).
        /// @see drainDeferredAncestorAdmissions, sealedStringRunLexCompare,
        ///      Memory::exprKeyView.
int deferredAncestorAdmissionLess(const DeferredAncestorAdmission& a,
                                          const DeferredAncestorAdmission& b) const;

public:
/// @brief Replay every deferred cross-LB admission seed single-threaded, in
        ///        `proveKernel`'s post-join drain (I-28 detect-and-defer trial).
        ///
        /// @details
        /// `updateAdmissionMap3`, reached from a status-2 goal registration inside a
        /// parallel phase-1/phase-3 worker, may climb `parentMemory` to a strict
        /// ancestor (the fold/recursion block that owns the goal's numeral) and seed
        /// that ancestor's admission map. Writing the ancestor from the child's
        /// worker is a cross-LB write forbidden during the parallel phase
        /// ([I-28](../../docs/agentic_swdd/30_invariants.md#i-28)): sibling recursion
        /// children sharing one fold ancestor would mint its cold maps concurrently.
        /// Instead each worker seals the resolved write's strings and
        /// `appendRecord`s it on the shared `deferredAncestorPages`
        /// `SealedPageSet` (under `deferredAncestorAdmissionsMutex`); this drain,
        /// run once after `pool.join()` alongside `activateZeroCondition` /
        /// `updateGlobal`, replays them.
        ///
        /// Replay order is made deterministic by an index sort under
        /// `deferredAncestorAdmissionLess` (the retired 6-field chain: exprKey,
        /// key, remainingArgs, partOfRecursion, maxAdmissionDepth,
        /// maxSecondaryNumber) before applying — grant/timing order must not leak
        /// into proof output (I-107). A `dischargedForever` ancestor is skipped
        /// (a permanently-deactivated LB has no next step and must not be
        /// resurrected — I-102 / I-112); a still-live ancestor is `ensureLoaded` (the
        /// working-set pager may have evicted it during the parallel phase) and then
        /// receives the inline `updateAdmissionMap` call (this drain runs on the main
        /// thread, where `g_inParallelWorkerPhase` is false, so the recursive
        /// re-entry applies inline rather than re-deferring). The page set is
        /// sealed, `freePages`'d, and the `std::optional` reset after the drain.
        ///
        /// @return (void)
        /// @invariant [I-28](../../docs/agentic_swdd/30_invariants.md#i-28) — the
        ///            single-threaded post-join drain is the sole writer of a
        ///            deferred ancestor admission.
        /// @see `updateAdmissionMap3` — the producer; `g_inParallelWorkerPhase` — the
        ///      detect gate; `deferredAncestorAdmissionLess` — the replay order.
void drainDeferredAncestorAdmissions();

public:
/// @brief Propagate a newly added statement through the
        ///        recursion-owned admission chains that demand it.
        ///
        /// @details
        /// Called at the deposit door (`addExprToMemoryBlock`) for every added
        /// statement row (D-413).
        /// Participants: an operator (either polarity), which binds only its
        /// output, and a positive config-backed compound that is not a
        /// two-input relation, which binds any argument; relations, atomic
        /// cores and config-less compounds return at once. The door marks the
        /// operator's output, or each distinct argument of the compound, and
        /// matches the marked form against the `admissionMap` templates. Only
        /// a template whose `admissionStatusMap` status is 1 (recursion-owned)
        /// propagates: each chain element whose bindable free argument marks
        /// to the same form is re-keyed with the statement's argument. In the
        /// re-keyed chain, an operator with only its output free and a compound
        /// with exactly one free argument gain a derived admission entry (that
        /// argument marked) with the source status; for a positive element the
        /// marker form's integration is prepared as well
        /// (`prepareIntegration`), so the chain's missing argument is
        /// constructed, not only admitted. An operator's unmatched output
        /// probe leaves the default-false status entry of the historical probe
        /// (D-172); an unmatched compound probe leaves nothing.
        ///
        /// A recursion-owned template propagates only when @p inductionAllowed
        /// holds. The door clears it for a product of a disintegration whose
        /// original expression has an argument that is not a number
        /// (`hasOnlyNumberArguments`); the probes and the default-false status
        /// entries above are unaffected
        /// (I-285).
        ///
        /// An in-place feed of the induction hypothesis's own chain (a value with the
        /// induction-hypothesis flag) advances the chain in place at the
        /// statement's scope like any other chain. In an unsettled tree it is also
        /// an eat like a witness admission: under `isAdmitted`'s eat gates (the
        /// chain filed at the scope's eating position, the slot open at the scope,
        /// no backward readmission) it records the eaten slot at the statement's
        /// scope, opens the induction scope
        /// `<scope>_boundary_is_<chain with marker>_<name>` named by the existing
        /// name it fills, and files a copy of the advanced chain's demands there
        /// with their parked-source wakes. The integration preparations stay at
        /// the statement's scope, whose constructions the new scope sees. A
        /// refused eat opens nothing (D-520, D-546).
        ///
        /// @param expression   The added statement (span over a caller-stable
        ///                     buffer; copied on entry).
        /// @param mb           The owning LB.
        /// @param validityName The statement's scope (copied on entry).
        /// @param inductionAllowed Whether a recursion-owned template may
        ///                     propagate this statement; forwarded unchanged
        ///                     to the per-match call.
        /// @param selectedKey  0 on the door call; a matched template's packed
        ///                     key on the per-match recursive call.
        /// @param boundSlot    -1 on the door call; on the per-match call, the
        ///                     slot of the statement argument the matched
        ///                     template marks.
        /// @invariant Heap-free; mutates `admissionMap` inside its value loop,
        ///            so the loop walks a verbatim snapshot
        ///            (`snapshotAdmissionRun`).
        /// @see updateAdmissionMap (the install-time marker branch),
        ///      prepareIntegration, revisitRejected2, revisitRejectedOrdis.
void updateAdmissionMapRecursion(StrSpan expression,
            Memory& mb,
            StrSpan validityName, bool inductionAllowed,
            int64_t selectedKey = 0, int32_t boundSlot = -1);

public:
// `mb` threaded in to enable revisitRejectedIntegration2 to fire
        // after each admissionSetIntegration insert — rejectedMapIntegration
        // lives in mb.overallHashMemory regardless of which HashMemory the
        // caller is populating.
void makeAdmissionKeys(const std::vector<std::string>& implication,
            const std::string& expression,
            HashMemory& localMemory,
            const std::string& validityName,
            Memory& mb);

public:
/// @brief Span-run overload of @ref makeAdmissionKeys — the span-native
        ///        body; the `const std::vector<std::string>&` form forwards.
        ///
        /// @details
        /// Byte-identical to the vector form (same arg validation, same
        /// `mintTemplateKey` / `admissionSetIntegration` / `revisitRejected-
        /// Integration2` interner-touch sequence on the same bytes), but reads the
        /// implication chain as a caller-owned `StrSpan` run and takes the
        /// expression / validity as spans, so a span-native caller
        /// (`addToHashMemory`, `prepareIntegrationCore2`) reaches it 0% heap. The
        /// vector form builds a stack `StrSpan` run and delegates here.
        ///
        /// @param impl         The implication-chain elements (premise order,
        ///                     head last); cap `MAX_ADMISSION_KEY_ELEMENTS + 1`.
        /// @param implN        Number of chain elements in @p impl.
        /// @param expr         The candidate expression.
        /// @param localMemory  HashMemory to register the trigger / admission keys.
        /// @param validityName Validity scope.
        /// @param mb           Owning LB.
        /// @return (void)
        /// @see makeAdmissionKeys(const std::vector<std::string>&, …) — the twin
        ///      oracle; mintTemplateKey, revisitRejectedIntegration2.
void makeAdmissionKeys(const StrSpan* impl, int32_t implN,
            StrSpan expr,
            HashMemory& localMemory,
            StrSpan validityName,
            Memory& mb);

public:
/// @brief Build the double-marker demand template of one disjunct with two
        ///        open arguments (D-492).
        ///
        /// @details
        /// An open argument is one without the `u_` prefix. The disjunct
        /// qualifies when it has exactly two argument positions that are open
        /// and the two hold different names; the earlier open position becomes
        /// `marker1`, the later `marker2`, and every other argument keeps its
        /// `u_` spelling. Any other shape (no, one or three open positions, or
        /// one name at both open positions) is not a double-marker disjunct —
        /// a defined classification, not a failure.
        ///
        /// @param disjunct One OR disjunct in its `u_`-prefixed instruction form.
        /// @param arena    String-tier scratch arena receiving the template.
        /// @param out      Receives the template when the disjunct qualifies.
        /// @return True iff the disjunct has exactly two distinct open arguments.
        /// @invariant Pure: mints nothing; allocates only on @p arena.
        /// @see updateAdmissionMapIntegrationDouble, makeDoubleMarkerQueryScratch.
static bool makeDoubleMarkerKeyScratch(StrSpan disjunct, ScratchArena& arena, ScratchString& out);

public:
/// @brief Build one double-marker probe template from a single-marker
        ///        constituent (D-492).
        ///
        /// @details
        /// The constituent carries `marker` at @p markerAt (the witness being
        /// admitted) and a value at @p otherAt that stands for the second open
        /// argument. Of the two positions the earlier becomes `marker1` and the
        /// later `marker2`; every other argument is prefixed with `u_`, the
        /// spelling the stored keys carry. The text outside the argument list
        /// (sign, predicate, brackets) is copied unchanged.
        ///
        /// @param markedExpr Bare single-marker constituent (no `u_` arguments).
        /// @param markerAt   Position of `marker`.
        /// @param otherAt    Position standing for the second open argument;
        ///                   differs from @p markerAt.
        /// @param arena      String-tier scratch arena receiving the template.
        /// @return The probe template on @p arena.
        /// @invariant No argument of @p markedExpr carries `u_` (asserted);
        ///            both positions are inside the argument list (asserted).
        /// @see isAdmittedIntegrationDouble, makeDoubleMarkerKeyScratch.
static ScratchString makeDoubleMarkerQueryScratch(StrSpan markedExpr, int32_t markerAt,
            int32_t otherAt, ScratchArena& arena);

public:
/// @brief Whether a live double-marker key with the predicate of
        ///        @p expression is visible at @p scope
        ///        (D-492).
        ///
        /// @details
        /// Visible means filed at @p scope or at a strict ancestor of it. The
        /// predicate is the text of @p expression up to its argument list
        /// (sign included); stored keys compare without their arguments. This
        /// is the eligibility of a constituent for the double-marker check and
        /// the eligibility of an algebra-input source for the relaxed I-268
        /// gate.
        ///
        /// @param mb         Owning block (read only).
        /// @param expression Constituent or disjunct text.
        /// @param scope      Registered scope of the probe.
        /// @return True iff such a key exists.
        /// @invariant Pure: mints nothing.
        /// @see isAdmittedIntegrationDouble.
static bool doubleMarkerPredicateVisible(const Memory& mb, StrSpan expression, NameId scope);

public:
/// @brief Whether an integration key filed in a block carries the wildcard
        ///        `pi_` (D-586).
        ///
        /// @details
        /// The two-marker start keys a part that holds both open arguments of
        /// its goal once per path: the path's argument becomes `marker`, and
        /// the other open argument is spelled `pi_`, which matches any value.
        /// Every key filing mints the key's bare non-marker arguments into
        /// `HashMemory::varsInAdmissionMapIntegrationKeys`, so a wildcard key
        /// was filed in the block iff that set holds the token. The wildcard
        /// variants of a query are probed only then, so blocks without such a
        /// key pay nothing.
        ///
        /// @param mb Owning block (read only).
        /// @return True iff an integration key filed in @p mb carries `pi_`.
        /// @invariant Pure: mints nothing.
        /// @see makeWildcardQueryScratch,
        ///      IntegrationProcessor::prepareIntegrationCore2.
static bool integrationWildcardFiled(const Memory& mb);

public:
/// @brief Build the wildcard variant of a marked integration query at one
        ///        argument position (D-586).
        ///
        /// @details
        /// The argument at @p position becomes `pi_`, the spelling of a
        /// wildcard key's other open argument; every other byte is copied
        /// unchanged. A query matches a wildcard key exactly when one of its
        /// variants equals the key, so the callers probe the variant of each
        /// non-marker position in turn, after the query itself.
        ///
        /// @param query    Marked query, `marker` at the admitted value and `u_`
        ///                 on every other argument.
        /// @param position Argument position to replace; a non-marker argument
        ///                 (asserted).
        /// @param arena    String-tier scratch arena receiving the variant.
        /// @return The variant on @p arena.
        /// @invariant Pure: mints nothing; allocates only on @p arena.
        /// @see integrationWildcardFiled, integrationDemandVisible,
        ///      matchingVisibleIntegrationDemands, updateAdmissionMapIntegration.
static ScratchString makeWildcardQueryScratch(StrSpan query, int32_t position, ScratchArena& arena);

public:
/// @brief Route-5 admission probe: whether a single-marker constituent
        ///        matches a double-marker key at its scope or a strict ancestor
        ///        (D-492).
        ///
        /// @details
        /// The constituent must carry exactly one `marker` (the witness has
        /// one open argument) and a predicate with visible double-marker keys.
        /// Every other argument position in turn stands for the key's second
        /// open argument: the probe template
        /// (`makeDoubleMarkerQueryScratch`) is matched class-aware against
        /// `admissionMapIntegrationDouble` at @p validityName and at each strict
        /// ancestor, compared at @p validityName. A hit admits an ordinary
        /// `int_` witness; the map records nothing (the key's value carries
        /// no information, in particular no consumed witnesses).
        ///
        /// The block's induction demands (`Memory::inductionDoubleKey` and
        /// `Memory::inductionProductKey`, the strong step's keys) are gated:
        /// they count only when @p inductionAllowed, otherwise the probe looks
        /// for another matching key (D-502,
        /// D-566). Goal-owned keys count either
        /// way.
        ///
        /// @param mb               Owning block.
        /// @param markedExpr       Bare single-marker constituent.
        /// @param validityName     Scope of the admission (interned or not).
        /// @param inductionAllowed Whether the induction demand may admit:
        ///                         false when the original expression being
        ///                         disintegrated (or the parked source of a
        ///                         replay) has an argument that is not a
        ///                         number or is a refused lemma digit
        ///                         (`hasOnlyInductionDigits` over
        ///                         `Memory::inductionParameterDigits`).
        /// The strong demands also admit at most one catch per path: inside an
        /// induction scope that either chain (`Memory::inductionDoubleChainId`,
        /// `Memory::inductionProductChainId`) names, neither key counts
        /// (D-519,
        /// D-577).
        ///
        /// @param hit              Receives the accepted key and the two
        ///                         argument positions that stood for its
        ///                         markers; untouched on a miss.
        /// @return True iff some combination matches a visible key that may
        ///         admit.
        /// @invariant Adds no admission record; may extend the derived
        ///            class-aware template index only.
        /// @see makeDoubleMarkerQueryScratch, doubleMarkerPredicateVisible,
        ///      DisintegrationProcessor::disintegrateExpr2,
        ///      CanonicalTemplateMatcher::canonicalTemplatePresentIf.
bool isAdmittedIntegrationDouble(Memory& mb, StrSpan markedExpr, StrSpan validityName,
            bool inductionAllowed = true, DoubleMarkerHit* hit = nullptr);

public:
/// @brief File the double-marker demand of one OR disjunct of a flattened
        ///        root integration goal (D-492).
        ///
        /// @details
        /// A disjunct of the wrong shape (`makeDoubleMarkerKeyScratch` false),
        /// one whose predicate may not key an integration instruction
        /// (`isIntegrationInstructionKeyEligible`) or is not configured files
        /// nothing — the same eligibility the single-marker keys of
        /// `prepareIntegrationCore2` Case C use. A new key wakes the parked
        /// integration sources at its scope and below that it now admits; a
        /// key already filed changes nothing. The key's value carries no
        /// information (always 0).
        ///
        /// An induction demand (a bound premise of the strong step block's
        /// hypotheses) also records its new key: the sum demand as
        /// `Memory::inductionDoubleKey`, the product demand as
        /// `Memory::inductionProductKey`; route 5 through either serves only an
        /// original of lemma input values, sets and operators (D-502,
        /// D-566). The block files each once, so
        /// the key is new and none was recorded before (asserted).
        ///
        /// @param mb              Owning block (single-threaded seam).
        /// @param disjunct        The disjunct in `u_`-prefixed instruction form.
        /// @param validityName    The root goal's scope.
        /// @param inductionDemand Which induction demand of the strong step
        ///                        block the key is, or `none`.
        /// @return True iff a new key was filed.
        /// @see makeDoubleMarkerKeyScratch,
        ///      RejectedSourceReplay::revisitRejectedIntegrationDouble,
        ///      IntegrationProcessor::installRootIntegrationFamily,
        ///      isAdmittedIntegrationDouble.
bool updateAdmissionMapIntegrationDouble(Memory& mb, StrSpan disjunct, StrSpan validityName,
            InductionDemand inductionDemand = InductionDemand::none);
private:

AdmissionPolicy& admissionPolicy;
const ce::AnchorInfo& anchorInfo;
CanonicalTemplateMatcher& canonicalTemplates;
std::mutex& deferredAncestorAdmissionsMutex;
std::optional<SealedPageSet>& deferredAncestorPages;
const CompiledDefinitionRegistry& definitions;
const ProverParameters& parameters;
ScopeTypeResolver& scopeTypes;
std::unique_ptr<MemorySteward>& steward;
IntegrationProcessor& integrationProcessor;
RejectedSourceReplay& rejectedSourceReplay;
EquivalenceQueries& equivalenceQueries;
};
}
#include "../logical_block/integration_processor.hpp"
#include "../logical_block/rejected_source_replay.hpp"
#include "../logical_block/equivalence_queries.hpp"
namespace gl {

/// @brief Seed the LB's admission map from a renamed key chain —
        ///        span-run-native, zero-heap interior.
        ///
        /// @details
        /// Renames the key chain via `renamingChainScratch`, and for each
        /// operator element with exactly one distinct non-remaining argument —
        /// and, in a recursion-owned chain, each positive compound element so
        /// placed that is not a two-input relation
        /// (D-413) — integrates the
        /// marker-substituted form and records the admission entry. Every
        /// string arrives as a `StrSpan` run over stable caller
        /// input; `key`/`remainingArgs` are sorted-unique runs (the former
        /// `std::vector` / `std::set` iteration orders) and `remainingArgs`
        /// MUST be sorted-unique — the frozen sequence, the `withoutRemaining`
        /// set-difference membership, and `prepareIntegration`'s sorted
        /// `unchangeableArgsSorted` all depend on it.
        ///
        /// @param key               The admission key chain, as spans — the
        ///                          rule's PREMISES only, never its head: a
        ///                          premise-completion demand asks for a
        ///                          missing premise instance; the head is
        ///                          the product the rule derives and can
        ///                          never be demanded (a head marker such as
        ///                          `!(in3[q,d,t,marker])` on the `×` slot is
        ///                          the D-379 bug).
        /// @param keyN              Element count of `key`.
        /// @param remainingArgs     Sorted-unique remaining-argument run.
        /// @param remN              Element count of `remainingArgs`.
        /// @param maxAdmissionDepth Recursion admission depth cap.
        /// @param maxSecondaryNumber Recursion secondary-iteration cap.
        /// @param admissionFlags    The seeded entry's flag byte:
        ///                          `kAdmissionRecursionFlag` marks it
        ///                          recursion-owned, with
        ///                          `kAdmissionInductionHypothesisFlag` it is the
        ///                          induction hypothesis's chain (0, 1 or 3). A
        ///                          hypothesis chain filed in an induction scope
        ///                          wakes the parked sources its new demands can
        ///                          eat (`revisitRejected2` at that scope), so a
        ///                          source parked at an ancestor before the chain
        ///                          arrived reaches it
        ///                          (D-519).
        /// @param validityName      Validity scope name for the packed key.
        /// @param recursionStatus   Files each written key's status as
        ///                          recursion although the value stays
        ///                          ordinary (`admissionFlags == 0`): a witness
        ///                          admitted through the key becomes a product
        ///                          of recursion, pure for the marker gate,
        ///                          while the value keeps the ordinary
        ///                          admission gate, which admits for originals
        ///                          with witness arguments too (I-285 reads the
        ///                          value's flag only). The forced successor
        ///                          demand's setting
        ///                          (D-571).
        /// @param weakHypothesisInstall True only when a weak step block
        ///                          installs its induction hypothesis's chain:
        ///                          if no element of the chain has an ordinary
        ///                          demand (the hypothesis cannot start
        ///                          eating), every element with exactly two
        ///                          unbound arguments demands an integration
        ///                          with the two markers `marker1` and
        ///                          `marker2`
        ///                          (D-586).
        /// @invariant The interner-touch sequence per admitted element is
        ///            FROZEN, in this order: (non-atomic only)
        ///            `nameMap.encode(removed)`; `valueInterner.encode` per
        ///            remainingArg in SET order (remIds); `valueInterner.encode`
        ///            per renamedKey element in CHAIN order (keyIds);
        ///            `mintTemplateKey(removed, validityName)`;
        ///            `templateInterner.encode` per non-`"marker"` arg of
        ///            `removed` in getArgs order. The remainingArgs-before-key
        ///            order (remIds before keyIds) is load-bearing — pinned by
        ///            `admission_value_ctor_mint_order_frozen`. This span
        ///            conversion swaps argument TYPES at existing calls, never
        ///            the call order (`renamingChainScratch` mints nothing).
        /// @see renamingChainScratch, prepareIntegration, mintTemplateKey,
        ///      insertAdmissionIdsBlob, classifyChainElement.
inline void AdmissionMemory::updateAdmissionMap(Memory& mb,
            const StrSpan* key, int32_t keyN,
            const StrSpan* remainingArgs, int32_t remN,
            int maxAdmissionDepth,
            int maxSecondaryNumber,
            uint8_t admissionFlags,
            StrSpan validityName,
            bool recursionStatus,
            bool weakHypothesisInstall) {
            assert(isValidAdmissionFlagByte(admissionFlags)
                && "updateAdmissionMap: malformed flag byte");
            assert((!recursionStatus || admissionFlags == 0)
                && "updateAdmissionMap: a recursion status beside a recursion-owned value is its flag already");
            const bool partOfRecursion = admissionFlags != 0;

            // One ScratchScope for the whole call: the renamed chain and every
            // per-element `removed` byte-bump forward on `sArena` and reclaim
            // together at function exit (never rewound inside the loop, 09c
            // Section 4 point 4). renamingChainScratch mints nothing, so the
            // frozen interner-touch sequence below is untouched.
            const unsigned slot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : scratchArenas().slotCount() - 1;
            ScratchArena& sArena = scratchArenas().forSlot(slot);
            ScratchScope sc(sArena);
            const NameId canonicalScope = mb.nameMap.encode(validityName);
            const bool hypothesisInInductionScope =
                (admissionFlags & kAdmissionInductionHypothesisFlag) != 0
                && gl::ScopeSyntax::isInductionScope(mb.nameMap, canonicalScope);
            StrSpan renamedKey[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
            StrSpan canonicalRemaining[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
            bool excludedAnchor = false;
            remN = canonicalTemplates.canonicalAdmissionChain(mb, key, keyN, remainingArgs, remN,
                canonicalScope, sArena, renamedKey, canonicalRemaining, excludedAnchor);
            remainingArgs = canonicalRemaining;
            const int32_t renamedN = keyN;


            // The weak hypothesis's two-marker start
            // (D-586): when a weak step block
            // installs its hypothesis's chain and no element of it has an
            // ordinary demand (a fully bound premise or one unbound argument),
            // the hypothesis cannot start eating; only then does every element
            // with exactly two unbound arguments demand an integration with
            // both of them marked.
            bool twoMarkerStart = false;
            if (weakHypothesisInstall) {
                assert(admissionFlags == (kAdmissionRecursionFlag | kAdmissionInductionHypothesisFlag)
                    && "updateAdmissionMap: a weak hypothesis install carries the hypothesis flags");
                twoMarkerStart = true;
                for (int32_t i = 0; i < renamedN && twoMarkerStart; ++i) {
                    StrSpan probeUnbound[ExecutionParameters::MAX_ARITY];
                    int32_t probeUnboundN = 0;
                    const ChainElementDemand probe = classifyChainElement(renamedKey[i],
                        remainingArgs, remN, partOfRecursion, probeUnbound, probeUnboundN);
                    if (probe == ChainElementDemand::fullyBoundPremise
                        || probe == ChainElementDemand::singleMarker)
                        twoMarkerStart = false;
                }
            }

            // The twin partner's case experiment
            // (D-588): the hypothesis cannot
            // start eating, and one of two twin elements holds a free argument
            // where the other holds `rec`, so that premise's integration waits
            // for a predecessor of the twin partner. The block then releases
            // the successor case of the partner's zero-or-successor OR first,
            // where that predecessor arises, and the zero case after it.
            int32_t twinRecElement = -1;
            int32_t twinFreeElement = -1;
            if (twoMarkerStart && mb.twinPartnerId != 0) {
                const auto isFixed = [&](StrSpan argument) {
                    for (int32_t r = 0; r < remN; ++r)
                        if (equalSpans(remainingArgs[r], argument)) return true;
                    return false;
                };
                bool freeAgainstRec = false;
                for (int32_t i = 0; i < renamedN && !freeAgainstRec; ++i) {
                    StrSpan argsI[ExecutionParameters::MAX_ARITY];
                    const int32_t argIN = getArgsSpans(renamedKey[i], argsI, ExecutionParameters::MAX_ARITY);
                    for (int32_t j = 0; j < renamedN && !freeAgainstRec; ++j) {
                        if (j == i || !equalSpans(extractExpressionUniversalSpan(renamedKey[i]),
                                extractExpressionUniversalSpan(renamedKey[j]))) continue;
                        StrSpan argsJ[ExecutionParameters::MAX_ARITY];
                        const int32_t argJN = getArgsSpans(renamedKey[j], argsJ, ExecutionParameters::MAX_ARITY);
                        assert(argIN == argJN && "updateAdmissionMap: twin elements of one predicate differ in arity");
                        for (int32_t t = 0; t < argIN && !freeAgainstRec; ++t)
                            freeAgainstRec = equalSpans(argsI[t], StrSpan("rec", 3)) && !isFixed(argsJ[t]);
                        if (freeAgainstRec) {
                            twinRecElement = i;
                            twinFreeElement = j;
                        }
                    }
                }
                if (freeAgainstRec) mb.twinExperiment = TwinExperiment::successorThenZero;
            }

            // The integration goal a chain element demands: its two unbound
            // arguments as `marker1` and `marker2` in argument order (the
            // two-marker start), or its one unbound argument as `marker`.
            const auto twoMarkerGoal = [&](StrSpan element, const StrSpan* unbound) {
                StrSpan elemArgs[ExecutionParameters::MAX_ARITY];
                const int32_t elemArgN = getArgsSpans(element, elemArgs,
                                                      ExecutionParameters::MAX_ARITY);
                StrReplacement rm2[2];
                int32_t rm2N = 0;
                for (int32_t a = 0; a < elemArgN && rm2N < 2; ++a) {
                    const bool open = equalSpans(elemArgs[a], unbound[0])
                        || equalSpans(elemArgs[a], unbound[1]);
                    if (!open || (rm2N == 1 && equalSpans(elemArgs[a], rm2[0].key))) continue;
                    rm2[rm2N].key = elemArgs[a];
                    rm2[rm2N].value = rm2N == 0 ? StrSpan("marker1", 7) : StrSpan("marker2", 7);
                    ++rm2N;
                }
                assert(rm2N == 2 && "updateAdmissionMap: a two-unbound element names both unbound arguments");
                return replaceKeysScratch(sArena, element, rm2, 2);
            };
            const auto singleMarkerGoal = [&](StrSpan element, StrSpan onlyArg) {
                StrReplacement rm1[1];
                rm1[0].key = onlyArg;
                rm1[0].value = StrSpan("marker", 6);
                return replaceKeysScratch(sArena, element, rm1, 1);
            };

            // Identical integration goals of a weak hypothesis install
            // (D-579): elements whose
            // demanded goals have the same text prepare one integration, and
            // its instructions are filed once per element, each copy bound
            // to the original premise its element came from, the ancestor
            // logic block at level k + 1 for chain element k. installGoal[k]
            // is element k's demanded goal, empty when it demands none. The
            // other elements of a group still call `prepareIntegration`
            // unsplit and stop at its prepared-marker probe, as before.
            StrSpan installGoal[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
            if (weakHypothesisInstall) {
                assert(renamedN <= ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                    && "updateAdmissionMap: chain longer than MAX_ADMISSION_KEY_ELEMENTS");
                for (int32_t k = 0; k < renamedN; ++k) {
                    StrSpan unbound[ExecutionParameters::MAX_ARITY];
                    int32_t unboundN = 0;
                    const ChainElementDemand kind = classifyChainElement(renamedKey[k],
                        remainingArgs, remN, partOfRecursion, unbound, unboundN);
                    if (kind == ChainElementDemand::twoUnbound && twoMarkerStart)
                        installGoal[k] = StrSpan(twoMarkerGoal(renamedKey[k], unbound));
                    else if (kind == ChainElementDemand::singleMarker)
                        installGoal[k] = StrSpan(singleMarkerGoal(renamedKey[k], unbound[0]));
                }
            }
            // The forbidden sets of the group chain element k leads, one per
            // member in chain order, written to `sets`; the member count, or
            // 0 when k leads no group of two or more.
            const auto identicalGoalSplit = [&](int32_t k, IntegrationForbiddenSets* sets) -> int32_t {
                if (!weakHypothesisInstall || installGoal[k].len == 0) return 0;
                int32_t members[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                int32_t memberN = 0;
                for (int32_t j = 0; j < renamedN; ++j) {
                    if (!equalSpans(installGoal[j], installGoal[k])) continue;
                    if (j < k) return 0;
                    members[memberN++] = j;
                }
                if (memberN < 2) return 0;
                StrSpan premises[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                int32_t levels[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                for (int32_t m = 0; m < memberN; ++m) {
                    const int32_t level = members[m] + 1;
                    premises[m] = ancestorPremiseKeyView(mb, level);
                    levels[m] = level;
                    assert(equalSpans(extractExpressionUniversalSpan(premises[m]),
                               extractExpressionUniversalSpan(renamedKey[members[m]]))
                        && "updateAdmissionMap: an identical goal's original premise has another core expression");
                }
                // Each copy also carries its own premise's level: the premise
                // of the int scope its first fill opens
                // (D-578).
                for (int32_t m = 0; m < memberN; ++m) {
                    premiseForbiddenSets(premises, levels, memberN, m, sets[m].variables, sets[m].levels);
                    sets[m].premiseLevel = levels[m];
                }
                return memberN;
            };

            for (int32_t i = 0; i < renamedN; ++i) {
                const StrSpan element = renamedKey[i];
                // withoutRemaining = the DISTINCT arguments of `element` NOT
                // present in remainingArgs; the demand kind follows from them
                // and from the element's configuration (classifyChainElement:
                // operators, config-backed compound relations, D-257, D-413).
                StrSpan withoutRemaining[ExecutionParameters::MAX_ARITY];
                int32_t wrN = 0;
                const ChainElementDemand demand = classifyChainElement(element,
                    remainingArgs, remN, partOfRecursion, withoutRemaining, wrN);

                // Fully-bound compound-relation premise of a RECURSION-owned
                // rule (the induction-hypothesis chain): every arg is a
                // remaining arg, so there is nothing to marker-ize — the
                // demand is concrete at install time. Formulate the
                // integration goal immediately: prepareIntegration commits
                // the (possibly negated) template and Case A mints the
                // subproof scope now; the burst engine closes it when the
                // ambient facts arrive (I-164 inherited-fact scan).
                // partOfRecursion-scoped deliberately: general rule installs
                // (e.g. the per-witness or0-expansion rules) would mint one
                // subproof per witness variable per burst — unbounded churn.
                // A negated premise's reductio waits for the chain's positive
                // premises (prepareHypothesisPremise).
                if (demand == ChainElementDemand::fullyBoundPremise) {
                    RT_SCOPE_HERE("UAM_PREPARE_INTEGRATION");
                    integrationProcessor.prepareHypothesisPremise(element, renamedKey, renamedN,
                        remainingArgs, remN, mb, validityName);
                    continue;
                }

                // Two unbound arguments: only the weak hypothesis's two-marker
                // start demands such an element. Its two unbound arguments
                // become `marker1` and `marker2` in argument order, and the
                // integration is prepared with both changeable (the
                // one-changeable gate lifted for this demand only). No
                // algebra admission entry is filed: the route-1 eat reads
                // single-marker keys only.
                if (demand == ChainElementDemand::twoUnbound) {
                    if (!twoMarkerStart) continue;
                    const ScratchString twoMarkerS = twoMarkerGoal(element, withoutRemaining);
                    IntegrationForbiddenSets split[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                    int32_t splitN = identicalGoalSplit(i, split);
                    // The first twin premise's goal is filed as the copy bound
                    // to its premise, alone in its group until the second twin
                    // premise's goal follows at the partner's predecessor
                    // (AdmissionMemory::fileTwinSecondGoal,
                    // D-589). The installer calls this
                    // once per copy of the hypothesis (multiplyImplication),
                    // the original first; the goal is filed for the first.
                    if (i == twinRecElement && mb.twinFirstGoalId == 0) {
                        assert(splitN == 0
                            && "updateAdmissionMap: a twin premise's goal is in no identical-goal group");
                        mb.twinFirstLevel = twinRecElement + 1;
                        mb.twinSecondLevel = twinFreeElement + 1;
                        mb.twinFirstGoalId = mb.nameMap.encode(StrSpan(twoMarkerS));
                        twinPremiseForbiddenSets(mb, mb.twinFirstLevel, mb.twinSecondLevel, 0, split[0]);
                        splitN = 1;
                    }
                    if (splitN > 0) mb.premiseSplitFiled = true;
                    RT_SCOPE_HERE("UAM_PREPARE_INTEGRATION");
                    integrationProcessor.prepareIntegration(StrSpan(twoMarkerS), remainingArgs, remN,
                        mb, validityName, StrSpan(), false, true, splitN > 0 ? split : nullptr, splitN);
                    continue;
                }

                if (demand == ChainElementDemand::singleMarker) {
                    const StrSpan onlyArg = withoutRemaining[0];

                    const ScratchString removedS = singleMarkerGoal(element, onlyArg);
                    const StrSpan removed(removedS);

                    {
                        IntegrationForbiddenSets split[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                        const int32_t splitN = identicalGoalSplit(i, split);
                        if (splitN > 0) mb.premiseSplitFiled = true;
                        RT_SCOPE_HERE("UAM_PREPARE_INTEGRATION");
                        integrationProcessor.prepareIntegration(removed, remainingArgs, remN, mb, validityName, StrSpan(),
                            false, false, splitN > 0 ? split : nullptr, splitN);
                    }

                    {
                    RT_SCOPE_HERE("UAM_ADMISSION_BLOB");
                    const LogicalEntity* le = this->definitions.compiledEntity(extractExpressionSpan(removed));
                    assert(le && "updateAdmissionMap: removed core expression must be compiled");

                    // Mint the two value runs in MSVC's ctor-arg evaluation
                    // order (right-to-left): the former
                    // AdmissionMapValue(encodeValueVector(renamedKey),
                    // encodeValueSetSorted(remainingArgs), ...) evaluated the
                    // remainingArgs (arg2) encode BEFORE the renamedKey (arg1)
                    // encode. Frozen by the pinning test
                    // admission_value_ctor_mint_order_frozen. The mints fire even
                    // when the consumed-skip continue fires (as the ctor args did),
                    // only the arena runs are reclaimed.
                    const unsigned uamSlot = (gl::WorkerContext::coreId() >= 0)
                        ? static_cast<unsigned>(gl::WorkerContext::coreId())
                        : genScratchArenas().slotCount() - 1;
                    ScratchArena& uamArena = genScratchArenas().forSlot(uamSlot);
                    const ArenaOffset uamMark = uamArena.cursor();

                    const int32_t remCount = remN;
                    int32_t* remIds = (remCount > 0)
                        ? reinterpret_cast<int32_t*>(uamArena.resolve(
                              uamArena.alloc(remCount * sizeof(int32_t), alignof(int32_t))))
                        : nullptr;
                    for (int32_t r = 0; r < remCount; ++r)
                        remIds[r] = mb.valueInterner.encode(remainingArgs[r]);
                    const int32_t keyCount = renamedN;
                    int32_t* keyIds = (keyCount > 0)
                        ? reinterpret_cast<int32_t*>(uamArena.resolve(
                              uamArena.alloc(keyCount * sizeof(int32_t), alignof(int32_t))))
                        : nullptr;
                    for (int32_t ki = 0; ki < keyCount; ++ki)
                        keyIds[ki] = mb.valueInterner.encode(renamedKey[ki]);

                    // Write site — mint the packed key once, reuse below.
                    const int64_t removedPk = mintTemplateKey(mb.templateInterner,
                        mb.nameMap, removed, validityName);

                    // Cold RMW insert; the status merge mirrors the former
                    // find-then-insert: OR onto an existing status when the key
                    // already had one, else set partOfRecursion
                    // (D-172). The splice scratch rides this worker's
                    // gen-scratch slot (the helper's own mark/popTo frame).
                    const int32_t adId =
                        mb.overallHashMemory.admissionMap.lookup(removedPk);
                    const int32_t previousDemandCount = adId == 0 ? 0 : mb.overallHashMemory.admissionMap.runLen(adId);
                    insertAdmissionIdsBlob(mb.overallHashMemory.admissionMap,
                                           removedPk, maxAdmissionDepth, maxSecondaryNumber,
                                           admissionFlags, keyIds, keyCount, remIds, remCount,
                                           mb.valueInterner, uamArena);
                    if (mb.overallHashMemory.admissionMap.runLen(mb.overallHashMemory.admissionMap.lookup(removedPk)) != previousDemandCount)
                        mb.lbMemory.canonicalUses.algebraDemands.changed(removedPk);
                    uamArena.popTo(uamMark);
                    const uint8_t* stPtr = (adId != 0)
                        ? mb.overallHashMemory.admissionStatusMap.find(removedPk)
                        : nullptr;
                    const bool newStatus = partOfRecursion || recursionStatus
                        || (stPtr != nullptr && *stPtr != 0);
                    mb.overallHashMemory.admissionStatusMap.upsert(
                        removedPk, static_cast<uint8_t>(newStatus ? 1 : 0));
                    // Populate vars-in-keys cache for the algebra equi-class
                    // hook (applyEquivalenceClassToAdmissionMap short-circuits
                    // on zero overlap).
                    StrSpan remArgs[ExecutionParameters::MAX_ARITY];
                    const int32_t remArgN = getArgsSpans(removed, remArgs,
                                                         ExecutionParameters::MAX_ARITY);
                    for (int32_t a = 0; a < remArgN; ++a) {
                        if (!equalSpans(remArgs[a], StrSpan("marker", 6))) {
                            mb.overallHashMemory.varsInAdmissionMapKeys.mint(
                                mb.templateInterner.encode(remArgs[a]));
                        }
                    }
                    } // RT_SCOPE UAM_ADMISSION_BLOB
                    // Ordis revival at the Writer-B inline insert (the
                    // maintainer-named seam): a parked cohort whose product
                    // template just gained this key wakes by mail.
                    // revisitRejected2 stays absent for the general map — the
                    // D-243 Pass-B rendezvous covers it at the park's own
                    // scope. A hypothesis chain filed in an induction scope is
                    // the exception: the source of its next slot may be parked
                    // at an ancestor already, and only this wake releases it
                    // into the scope where the eating stands
                    // (D-519).
                    {
                        RT_SCOPE_HERE("UAM_REVISIT_ORDIS");
                        rejectedSourceReplay.revisitRejectedOrdis(removed, mb, validityName);
                    }
                    if (hypothesisInInductionScope) {
                        RT_SCOPE_HERE("UAM_REVISIT_INDUCTION");
                        rejectedSourceReplay.revisitRejected2(removed, mb, validityName);
                    }
                }
            }
        }

/// @brief Integration-side admission check — gates whether a
        /// compound integration body may fire from the given LB.
        ///
        /// @details
        /// Uses isIntegrationInstructionKeyEligible before probing or replaying.
        /// Prefixes non-marker arguments with `u_` and probes the
        /// `mb.overallHashMemory.admissionMapIntegration` keys visible at the
        /// scope: owned there or, for an OR case, at the scope its demands come
        /// from (`ancestorDemandOwner`, `matchingVisibleIntegrationDemands`,
        /// D-572).
        /// A pure probe reports key presence without mutation. A committed
        /// hit records the witness for each stored instruction and invokes
        /// `prepareIntegrationCore2` with its marker replaced by that witness;
        /// instructions already processed for the witness are skipped.
        ///
        /// @param mb           Owning LB.
        /// @param expr         Constituent expression (span source; read-only).
        /// @param var          Witness/replacement variable (span source).
        /// @param markedExpr   Marked template form (span source).
        /// @param validityName Validity scope (span source).
        /// @return True iff an eligible integration instruction key exists,
        ///         including when its instructions already processed the witness.
        /// @see `isAdmitted` — disintegration-side counterpart.
        /// @see [`HashMemory::admissionMapIntegration`](memory.hpp#hashmemory).
        /// @param commit Apply successful instruction updates; false is a pure probe.
        /// @param selectedKey One already-matched key to process, or 0.
        /// @param linked      The route's linked arguments for the witness-link
        ///                    gate (D-529): in an
        ///                    induction step block an instruction whose path
        ///                    list names one of them refuses the route — the
        ///                    probe ignores it and the commit leaves it
        ///                    untouched. Null when @p linkedCount is 0.
        /// @param linkedCount Number of @p linked; 0 disables the gate.
inline bool AdmissionMemory::isAdmittedIntegration(Memory& mb,
            StrSpan expr,
            StrSpan var,
            StrSpan markedExpr,
            StrSpan validityName,
            bool commit, int64_t selectedKey,
            const StrSpan* linked, int32_t linkedCount,
            const WitnessHistoryProbe* history,
            InductionScopeSink* intScopes) {

            assert(this->definitions.compiledEntity(extractExpressionUniversalSpan(StrSpan(markedExpr))) != nullptr
                && "isAdmittedIntegration: marked core expression must be compiled");
            assert((intScopes == nullptr || commit)
                && "isAdmittedIntegration: only a commit opens int scopes");
            if (!admissionPolicy.isIntegrationInstructionKeyEligible(expr)) {
                return false;
            }

            // 1-3. u_-collision pre-scan + rename every non-marker arg to u_+arg
            // (prefixNonMarkerArgumentsWithUScratch, the byte twin of the former
            // inline replacementMap + ce::replaceKeysInString). renamedMarkedExpr
            // is held named on the string arena; it survives the later interner
            // mints + the re-entrant prepareIntegrationCore2 because nested
            // ScratchScopes rewind only their own allocations (09b pitfall 6).
            const unsigned renSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : scratchArenas().slotCount() - 1;
            ScratchArena& renArena = scratchArenas().forSlot(renSlot);
            ScratchScope renameScope(renArena);
            {
                StrSpan preArgs[ExecutionParameters::MAX_ARITY];
                const int32_t preArgN = getArgsSpans(StrSpan(markedExpr), preArgs,
                                                     ExecutionParameters::MAX_ARITY);
                for (int32_t a = 0; a < preArgN; ++a) {
                    if (!equalSpans(preArgs[a], StrSpan("marker", 6))
                        && preArgs[a].len >= 2 && preArgs[a].ptr[0] == 'u'
                        && preArgs[a].ptr[1] == '_') {
                        std::cout << "Assertion violation in isAdmittedIntegration. Arg: "
                                  << std::string(preArgs[a].ptr,
                                       static_cast<std::size_t>(preArgs[a].len)) << std::endl;
                        assert(false && "Argument in isAdmittedIntegration already starts with u_");
                    }
                }
            }
            const ScratchString renamedMarkedExprS =
                gl::DefinitionExpansion::prefixNonMarkerArgumentsWithUScratch(renArena, StrSpan(markedExpr));
            const StrSpan renamedMarkedExpr(renamedMarkedExprS);


            // 4. Check if the marked expression exists in the integration map
            // Non-minting probe: a template never interned was never
            // registered, a definitive miss.
            const NameId scopeId = mb.nameMap.lookup(validityName);
            if (scopeId == 0) return false;
            // The witness-link gate (D-529): only in an
            // induction step block and only for a route with linked arguments
            // or a checked witness history (D-522). In a block whose weak
            // hypothesis install split identical goals by premise, every
            // route with an arriving expression reaches the per-instruction
            // decision, which first refuses a cross token
            // (D-579) and, for a root copy
            // outside every int scope, an arriving expression with an
            // argument that is not a number (D-578).
            const bool split = mb.premiseSplitFiled && history != nullptr;
            const bool gate = split || (mb.recursionAuxyImplId != 0
                && (linkedCount > 0 || (history != nullptr && history->active())));
            // Whether a premise-bound copy refuses the arriving expression.
            const auto crossRefuses = [&](const IntegrationForbiddenSets& forbidden) {
                return split && premiseCopyRefuses(mb, scopeId, forbidden, history->start);
            };
            // Whether one instruction stored under key `pk` accepts the route.
            const auto keyAccepts = [&](int64_t pk) {
                const unsigned acceptSlot = gl::WorkerContext::coreId() >= 0
                    ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
                ScratchArena& acceptArena = genScratchArenas().forSlot(acceptSlot);
                ScratchScope acceptFrame(acceptArena);
                ArenaIntegrationMap entries(&acceptArena, &mb.valueInterner);
                entries.buildFromCold(mb.overallHashMemory.admissionMapIntegration, pk);
                for (int32_t i = 0; i < entries.entries.size(); ++i) {
                    if (crossRefuses(entries.entries[i].forbidden)) continue;
                    const int32_t listId = entries.entries[i].pathVarsId;
                    const StrSpan list = listId == 0 ? StrSpan() : mb.valueInterner.decodeView(listId);
                    if (!witnessLinkRefuses(mb, list, linked, linkedCount, scopeId, history)) return true;
                }
                return false;
            };
            if (selectedKey == 0) {
                // A strong step block checks its keys one by one: outside the
                // induction scopes a key that holds nothing connected admits
                // nothing (D-602).
                if (!commit && !gate && !mb.strongStepBlock)
                    return integrationDemandVisible(mb, renamedMarkedExpr, validityName, scopeId);
                const unsigned slot = gl::WorkerContext::coreId() >= 0 ? static_cast<unsigned>(gl::WorkerContext::coreId())
                    : genScratchArenas().slotCount() - 1;
                ScratchArena& arena = genScratchArenas().forSlot(slot);
                ScratchScope frame(arena);
                DirtyState dirty = DirtyState::Clean;
                PagedVector<int64_t> matches(&arena, &dirty);
                const ScratchString witness = ScratchString::copyFrom(renArena, var.ptr, var.len);
                const ScratchString scope = ScratchString::copyFrom(renArena, validityName.ptr, validityName.len);
                const ScratchString canonical = canonicalTemplates.canonicalAdmissionText(mb, expr, scopeId, renArena, false, StrSpan(witness));
                const ScratchString query = canonicalTemplates.canonicalAdmissionText(mb, markedExpr, scopeId, renArena);
                matchingVisibleIntegrationDemands(mb, renamedMarkedExpr, StrSpan(scope), scopeId, matches, arena);
                if (!commit) {
                    for (int32_t i = 0; i < matches.size(); ++i)
                        if (strongStepKeyAdmits(mb, matches[i], scopeId) && (!gate || keyAccepts(matches[i])))
                            return true;
                    return false;
                }
                bool admitted = false;
                for (int32_t i = 0; i < matches.size(); ++i)
                    admitted |= isAdmittedIntegration(mb, StrSpan(canonical), StrSpan(witness), StrSpan(query),
                        StrSpan(scope), commit, matches[i], linked, linkedCount, history, intScopes);
                return admitted;
            }
            const int64_t renamedPk = selectedKey;
            // Non-minting presence probe — 0 means the key was never minted.
            if (mb.overallHashMemory.admissionMapIntegration.lookup(renamedPk) == 0) {
                return false;
            }
            // Outside the induction scopes of a strong step block a key that
            // holds nothing connected admits nothing
            // (D-602).
            if (!strongStepKeyAdmits(mb, renamedPk, scopeId)) return false;

            if (!commit) return !gate || keyAccepts(renamedPk);
            if (gate && !keyAccepts(renamedPk)) return false;
            // The key admits the witness; in a weak or strong step block a
            // connected key makes it a size marker
            // (D-583,
            // D-584).
            if ((mb.weakStepBlock || mb.strongStepBlock) && admittingKeyConnected(mb, renamedPk))
                recordSizeMarker(mb, var);
            // An ancestor's demand admits the witness here; its instruction is
            // updated and continued here only after the induction hypothesis
            // shot on the path (D-572).
            {
                const NameId owner = Codec<StatementKey>::decode(renamedPk).validity;
                if (owner != scopeId && !hypothesisShotOnPath(mb, scopeId, owner)) return true;
            }

            // 5. Prepare the one-pair replacement run ("marker" -> "u_"+var) on
            // renArena, held live through the re-entrant Core2 call below.
            assert(!startsWithSpan(var, "u_", 2));
            char* uVarBuf = renArena.allocBytes(2 + var.len);
            uVarBuf[0] = 'u'; uVarBuf[1] = '_';
            if (!var.empty()) std::memcpy(uVarBuf + 2, var.ptr, static_cast<size_t>(var.len));
            const StrReplacement replacementMap2[1] = {
                { StrSpan("marker", 6), StrSpan(uVarBuf, 2 + var.len) } };

            // 6. Decode the key's nested instruction map (an owned snapshot of
            // the cold run).
            const unsigned aimSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            ArenaIntegrationMap entryMap(
                &genScratchArenas().forSlot(aimSlot), &mb.valueInterner);
            entryMap.buildFromCold(
                mb.overallHashMemory.admissionMapIntegration, renamedPk);

            // Snapshot the original iteration set. Recursive preparation may
            // extend the same key; each consumed-value write reloads current storage.
            int32_t snapshot[ExecutionParameters::MAX_INTEGRATION_ENTRIES];
            const int32_t snapN = entryMap.sortedIndices(
                snapshot, ExecutionParameters::MAX_INTEGRATION_ENTRIES);

            for (int32_t si = 0; si < snapN; ++si) {
                const int32_t snapEi = snapshot[si];
                WorkInstruction wiSrc(&genScratchArenas().forSlot(aimSlot));
                entryMap.loadInstructionInto(snapEi, wiSrc, mb.valueInterner);
                // An instruction whose path list the route links its witness
                // to refuses it and stays untouched
                // (D-529); so does a copy bound to one premise whose
                // forbidden sets the arriving expression hits
                // (D-579).
                if (gate && (crossRefuses(wiSrc.forbidden)
                        || witnessLinkRefuses(mb, wiSrc.pathVariables(), linked, linkedCount, scopeId, history))) continue;
                ArenaIntegrationMap current(&genScratchArenas().forSlot(aimSlot), &mb.valueInterner);
                current.buildFromCold(mb.overallHashMemory.admissionMapIntegration, renamedPk);
                const int32_t currentCount = current.entries.size();
                const int32_t currentEntry = current.findOrEmplaceWork(wiSrc, mb.valueInterner);
                assert(current.entries.size() == currentCount
                    && "integration replay lost its original instruction");
                if (canonicalTemplates.integrationValueSeenAtScope(mb, current, currentEntry, var, scopeId)) continue;

                // The first fill of a premise-bound copy outside every int
                // scope opens the copy's int scope, and the copy continues
                // there; a closed int scope never reopens. Any other accepting
                // instruction continues here, where the witness is then
                // published (D-578).
                NameId intScope = 0;
                if (wiSrc.forbidden.premiseLevel != 0 && !gl::ScopeSyntax::atOrBelowIntScope(mb.nameMap, scopeId)) {
                    assert(intScopes != nullptr && "isAdmittedIntegration: a commit that opens an int scope reports it");
                    intScope = openIntScope(mb, scopeId, wiSrc, var);
                    if (intScope == 0) continue;
                } else if (intScopes != nullptr) {
                    intScopes->publishedHere = true;
                }

                // Recursive preparation may append instructions at this same key.
                // Preserve those live additions when recording the next consumed value.
                current.insertValue(currentEntry, mb.valueInterner.encode(StrSpan(var)));
                current.writeToCold(mb.overallHashMemory.admissionMapIntegration, renamedPk);
                mb.lbMemory.canonicalUses.integrationDemands.changed(renamedPk);
                // The witness updated an instruction at its admission; the
                // statement fill never tries it as a marker again
                // (D-585).
                recordOneWayWitness(mb, var);
                // Outside the step blocks a key connected to the accepting
                // instruction's goal makes the witness a connected value
                // (D-593).
                if (!mb.weakStepBlock && !mb.strongStepBlock
                        && admittingKeyConnectedToGoal(mb, renamedPk, wiSrc.markedGoal()))
                    recordSizeMarker(mb, var);

                const unsigned wiSlot = aimSlot;
                WorkInstruction wiCopy(&genScratchArenas().forSlot(wiSlot));
                cleanInstructionWork(wiSrc, wiCopy, renamedMarkedExpr);
                // The continuation's path list gains the admitted witness.
                if (mb.recursionAuxyImplId != 0) {
                    ScratchScope listScope(renArena);
                    wiCopy.setPathVariables(StrSpan(extendPathListScratch(renArena, wiCopy.pathVariables(), var)));
                }
                // Owned copy of the int scope's name: the continuation mints
                // the NameMap (I-3).
                ScratchString intScopeName;
                StrSpan continuationScope = validityName;
                if (intScope != 0) {
                    const StrSpan intScopeView = mb.nameMap.decodeView(intScope);
                    intScopeName = ScratchString::copyFrom(renArena, intScopeView.ptr, intScopeView.len);
                    continuationScope = StrSpan(intScopeName);
                    intScopes->add(intScope);
                }
                integrationProcessor.prepareIntegrationCore2(wiCopy, replacementMap2, 1, mb, continuationScope, StrSpan("", 0),
                    /*continuation=*/true);
            }

            // 8. Return true indicating the key was found and processed
            return true;
        }

/// @brief Open, or find, the int scope a first fill of a premise-bound
        ///        integration copy starts, and record it.
        ///
        /// @details
        /// Mints `<parent>_boundary_int_scope_<premise>_<witness>`: the
        /// premise is the key of the ancestor block at the copy's premise
        /// level, the witness the name that filled the marker
        /// (D-578). The same premise and witness
        /// reached through another key of the copy find the same scope. The
        /// scope's record holds its goal group (the copy's integration goal
        /// with `marker1` and `marker2` written `marker`, minted into the
        /// NameMap) and its premise level; the group's record holds the bit
        /// set of the group's premise levels (the copy's own and its
        /// forbidden ones). A scope a wipe has closed
        /// (`intValidityNamesToFilter`) never reopens, and a group whose
        /// record carries the promoted mark opens no scope, its integration
        /// having left through the promotion
        /// (D-582): the result is then 0 and nothing
        /// is recorded; the promoted check probes without minting.
        ///
        /// @param mb       Owning block (single-threaded seam).
        /// @param parentId The scope the first fill is decided at; neither an
        ///                 int scope nor below one (asserted).
        /// @param wi       The premise-bound instruction being filled.
        /// @param witness  The filling name (stable through the NameMap mints).
        /// @return The int scope, or 0 when it was closed or its group was
        ///         promoted.
        /// @invariant Mints into the NameMap and writes the two records only;
        ///            a scope's record never changes (asserted).
        /// @see ScopeSyntax::intScopePayloadScratch, ancestorPremiseKeyView,
        ///      StatementMemory::addStatement.
inline NameId AdmissionMemory::openIntScope(Memory& mb, NameId parentId, const WorkInstruction& wi,
            StrSpan witness) {
            const IntegrationForbiddenSets& binding = wi.forbidden;
            assert(binding.premiseLevel > 0 && binding.premiseLevel < 62
                && "openIntScope: a premise-bound copy below the promoted mark");
            assert(!gl::ScopeSyntax::atOrBelowIntScope(mb.nameMap, parentId)
                && "openIntScope: no int scope opens at or below another");
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
            ScratchArena& strings = scratchArenas().forSlot(slot);
            ScratchScope frame(strings);
            // A copy bound to the second twin premise joins the first
            // premise's group: its goal holds `rec2` where the first goal
            // holds `rec` (D-589).
            const bool secondTwin = binding.premiseLevel == mb.twinSecondLevel;
            assert((!secondTwin || mb.twinPredecessorId != 0)
                && "openIntScope: a second-twin copy exists once its predecessor is recorded");
            const StrReplacement unlabel[3] = {
                { StrSpan("marker1", 7), StrSpan("marker", 6) },
                { StrSpan("marker2", 7), StrSpan("marker", 6) },
                { secondTwin ? mb.nameMap.decodeView(mb.twinPredecessorId) : StrSpan(), StrSpan("rec", 3) } };
            const ScratchString group = replaceKeysScratch(strings, wi.markedGoal(), unlabel, secondTwin ? 3 : 2);
            // A promoted group is closed; a group whose name was never minted
            // has no record and is open.
            const NameId knownGroup = mb.nameMap.lookup(StrSpan(group));
            if (knownGroup != 0) {
                const int64_t* knownLevels = mb.lbMemory.intScopeGroups.find(static_cast<int64_t>(knownGroup));
                if (knownLevels != nullptr && (*knownLevels & kIntScopeGroupPromoted) != 0)
                    return 0;
            }
            // The premise reads the skeleton interner, which the NameMap mints
            // below leave alone (I-97).
            const ScratchString payload = gl::ScopeSyntax::intScopePayloadScratch(strings,
                ancestorPremiseKeyView(mb, binding.premiseLevel), witness);
            const int32_t namesBefore = mb.nameMap.nameCount();
            const NameId scope = mb.nameMap.encodePush(parentId, StrSpan(payload));
            if (mb.nameMap.nameCount() == namesBefore && mb.lbMemory.intValidityNamesToFilter.lookup(scope) != 0)
                return 0;
            const NameId groupId = mb.nameMap.encode(StrSpan(group));
            const int64_t record = packInt32Pair(groupId, binding.premiseLevel);
            const int64_t* stored = mb.lbMemory.intScopes.find(static_cast<int64_t>(scope));
            if (stored == nullptr)
                mb.lbMemory.intScopes.insert(static_cast<int64_t>(scope), record);
            else
                assert(*stored == record && "openIntScope: one int scope belongs to one premise of one group");
            const int64_t levels = static_cast<int64_t>(binding.levels | forbiddenSetBit(binding.premiseLevel));
            assert((levels & kIntScopeGroupPromoted) == 0 && "openIntScope: a premise level at the promoted mark");
            const int64_t* groupLevels = mb.lbMemory.intScopeGroups.find(static_cast<int64_t>(groupId));
            if (groupLevels == nullptr)
                mb.lbMemory.intScopeGroups.insert(static_cast<int64_t>(groupId), levels);
            else
                assert((*groupLevels & ~kIntScopeGroupPromoted) == levels
                    && "openIntScope: every copy of a group names the group's premise levels");
            return scope;
        }

/// @brief Whether the step-key rule binds the route-3 keys filed at a scope
        ///        of a block.
        ///
        /// @details
        /// The rule binds where an induction hypothesis has an instance: at
        /// every scope of a weak step block, whose hypothesis is installed at
        /// `rec` on `main` (D-583); in a strong
        /// step block, whose hypothesis is instantiated only at a caught value
        /// inside the induction scope the catch opens, at an induction scope
        /// and below (`ScopeSyntax::atOrBelowInductionScope`,
        /// D-584). A strong step block's
        /// `main` is the step's own goal context and files freely, like every
        /// other block.
        ///
        /// @param mb           The block (single-threaded seam).
        /// @param validityName The filing scope (registered, asserted in a
        ///                     strong step block).
        /// @return `true` iff a key filed at @p validityName must be connected.
        /// @invariant Mints nothing.
        /// @see stepKeyConnected, IntegrationProcessor::prepareIntegrationCore2.
inline bool AdmissionMemory::stepKeyRuleBinds(Memory& mb, StrSpan validityName) {
            if (mb.weakStepBlock) return true;
            if (!mb.strongStepBlock) return false;
            const NameId scopeId = mb.nameMap.lookup(validityName);
            assert(scopeId != 0 && "stepKeyRuleBinds: the filing scope is registered");
            return gl::ScopeSyntax::atOrBelowInductionScope(mb.nameMap, scopeId);
        }

/// @brief Whether the key a route-3 commit or a statement fill admits a
        ///        value through is connected, so the value is a size marker.
        ///
        /// @details
        /// A connected key ties the value it admits to `rec`, n or a caught
        /// value (D-583). Every key of a weak
        /// step block was filed under the rule, so it is connected. A strong
        /// step block files freely outside its induction scopes, so the
        /// admitting key's text is checked at its own scope with
        /// @ref stepKeyConnected (D-584); both
        /// are copied first, as the check reads the tables they live in.
        ///
        /// @param mb The step block (single-threaded seam; asserted).
        /// @param pk The admitting key: an `admissionMapIntegration` key, its
        ///           template and validity minted.
        /// @return `true` iff the key's text is connected at its scope.
        /// @invariant Mints nothing.
        /// @see stepKeyConnected, recordSizeMarker.
inline bool AdmissionMemory::admittingKeyConnected(Memory& mb, int64_t pk) {
            assert((mb.weakStepBlock || mb.strongStepBlock)
                && "admittingKeyConnected: only a step block records size markers");
            if (mb.weakStepBlock) return true;
            const StatementKey key = Codec<StatementKey>::decode(pk);
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
            ScratchArena& strings = scratchArenas().forSlot(slot);
            ScratchScope frame(strings);
            const StrSpan keyView = mb.templateInterner.decodeView(key.orig);
            const ScratchString keyText = ScratchString::copyFrom(strings, keyView.ptr, keyView.len);
            const StrSpan scopeView = mb.nameMap.decodeView(key.validity);
            const ScratchString scopeText = ScratchString::copyFrom(strings, scopeView.ptr, scopeView.len);
            return stepKeyConnected(mb, StrSpan(keyText), StrSpan(scopeText));
        }

/// @brief Whether a route-3 key may admit a value at a scope of the block.
        ///
        /// @details
        /// A strong step block instantiates its hypothesis only inside the
        /// induction scopes its catches open (D-584). Its `main` files keys
        /// freely, and those keys serve the induction scopes and the OR cases
        /// below them (D-572). Outside the induction scopes a key that holds
        /// nothing connected to the induction (@ref admittingKeyConnected)
        /// admits nothing: the route-3 probe, the route-3 commit and the
        /// statement fill refuse it there, so the step's own goal builds no
        /// object without a size (D-602). At an
        /// induction scope and below every key admits as before; in every
        /// other block every key admits.
        ///
        /// @param mb      The block (single-threaded seam).
        /// @param pk      The admitting key: an `admissionMapIntegration` key,
        ///                its template and validity minted.
        /// @param scopeId The scope the value would be admitted at; registered
        ///                (asserted in a strong step block).
        /// @return `false` iff @p mb is a strong step block, @p scopeId is
        ///         neither an induction scope nor below one, and @p pk is not
        ///         connected at its own scope.
        /// @invariant Mints nothing.
        /// @see admittingKeyConnected, stepKeyRuleBinds, isAdmittedIntegration,
        ///      updateAdmissionMapIntegration.
inline bool AdmissionMemory::strongStepKeyAdmits(Memory& mb, int64_t pk, NameId scopeId) {
            if (!mb.strongStepBlock) return true;
            assert(scopeId != 0 && "strongStepKeyAdmits: the admission scope is registered");
            if (gl::ScopeSyntax::atOrBelowInductionScope(mb.nameMap, scopeId)) return true;
            return admittingKeyConnected(mb, pk);
        }

/// @brief Whether a route-3 key outside the step blocks is connected to the
        ///        goal of an instruction filed under it.
        ///
        /// @details
        /// The step-key game played on the goal's own variables
        /// (D-593): a step block connects its keys
        /// through `rec`, the induction variable and the size markers (D-583),
        /// any other block through the head variables of the goal. Arguments
        /// lose their `u_` prefix, and the labels `marker`, `marker1`,
        /// `marker2`, every `pi_` name and the anchor's arguments
        /// (`ScopeSyntax::anchorArgumentSpans`, the constants every theorem of
        /// the batch shares) take no part. The goal's remaining arguments are
        /// its head variables, each read as its representative in the classes
        /// of @p scope (`CanonicalTemplateMatcher::canonicalArgumentAtScope`),
        /// as a key filed there names it; a head variable whose representative
        /// is an anchor argument is a constant there and drops out. A goal
        /// without head variables is exempt: every key of it is connected.
        /// Otherwise one of the key's remaining arguments must be a head
        /// variable or a connected value of the block: a value an earlier
        /// admission wrote into an instruction through a connected key
        /// (`LbMemory::sizeMarkers`). Names compare as written, as in D-583.
        /// `(ordered[u_1,u_4,marker])` for the goal
        /// `(hasPrimeFactorization[1,3,4,5,2,6,7,9])` under
        /// `(AnchorFTA[1,2,3,4,5,6,7,8])` names only anchor arguments and is
        /// not connected; a key naming `u_9` is. A key also connects when an
        /// anchor argument that the goal holds k > 1 times appears in the key
        /// exactly k times: the key repeats the goal's own constant shape, as
        /// `(interval[u_1,u_4,u_2,u_2,marker])` does for
        /// `(sequence[1,4,2,2,10])`, whose witness is the interval [0,0].
        ///
        /// @param mb    The block (its anchor chain, classes and connected
        ///              values).
        /// @param key   The key text: `u_`-prefixed arguments and a marker, in
        ///              the canonical form of @p scope, on caller-owned
        ///              storage.
        /// @param goal  The instruction's marked goal, on caller-owned storage.
        /// @param scope The key's scope (registered, asserted).
        /// @return `true` iff @p goal has no head variable, an argument of
        ///         @p key is a head variable of @p goal or a connected value,
        ///         or @p key repeats an anchor argument exactly as often as
        ///         @p goal does, more than once.
        /// @invariant Allocates and mints nothing.
        /// @see admittingKeyConnectedToGoal, recordSizeMarker,
        ///      IntegrationProcessor::prepareIntegrationCore2.
inline bool AdmissionMemory::goalKeyConnected(Memory& mb, StrSpan key, StrSpan goal, NameId scope) {
            assert(scope != 0 && "goalKeyConnected: the key's scope is registered");
            StrSpan anchors[ExecutionParameters::MAX_ARITY];
            const int32_t anchorN = gl::ScopeSyntax::anchorArgumentSpans(mb, StrSpan(anchorInfo.name),
                anchors, ExecutionParameters::MAX_ARITY);
            const auto anchorArgument = [&](StrSpan name) {
                for (int32_t k = 0; k < anchorN; ++k)
                    if (equalSpans(anchors[k], name)) return true;
                return false;
            };
            // The argument without its `u_` prefix; empty for a label or an
            // anchor argument, which take no part.
            const auto bareVariable = [&](StrSpan argument) -> StrSpan {
                if (startsWithSpan(argument, "u_", 2)) argument = StrSpan(argument.ptr + 2, argument.len - 2);
                if (equalSpans(argument, StrSpan("marker", 6)) || equalSpans(argument, StrSpan("marker1", 7))
                    || equalSpans(argument, StrSpan("marker2", 7)) || startsWithSpan(argument, "pi_", 3))
                    return StrSpan();
                return anchorArgument(argument) ? StrSpan() : argument;
            };
            StrSpan goalArgs[ExecutionParameters::MAX_ARITY];
            const int32_t goalN = getArgsSpans(goal, goalArgs, ExecutionParameters::MAX_ARITY);
            StrSpan heads[ExecutionParameters::MAX_ARITY];
            int32_t headN = 0;
            for (int32_t g = 0; g < goalN; ++g) {
                const StrSpan head = bareVariable(goalArgs[g]);
                if (head.empty()) continue;
                // A key filed at the scope names the head variable's
                // representative there; one that is an anchor argument is a
                // constant at the scope.
                const StrSpan representative = canonicalTemplates.canonicalArgumentAtScope(mb, head, scope);
                if (!anchorArgument(representative)) heads[headN++] = representative;
            }
            if (headN == 0) return true;
            StrSpan keyArgs[ExecutionParameters::MAX_ARITY];
            const int32_t keyN = getArgsSpans(key, keyArgs, ExecutionParameters::MAX_ARITY);
            for (int32_t a = 0; a < keyN; ++a) {
                const StrSpan name = bareVariable(keyArgs[a]);
                if (name.empty()) continue;
                for (int32_t h = 0; h < headN; ++h)
                    if (equalSpans(heads[h], name)) return true;
                const int32_t valueId = mb.valueInterner.lookup(name);
                if (valueId != 0 && mb.lbMemory.sizeMarkers.lookup(valueId) != 0) return true;
            }
            // An anchor argument the goal repeats, held by the key exactly as
            // often: the key repeats the goal's own constant shape.
            const auto occurrences = [](const StrSpan* args, int32_t n, StrSpan name) {
                int32_t count = 0;
                for (int32_t i = 0; i < n; ++i) {
                    StrSpan argument = args[i];
                    if (startsWithSpan(argument, "u_", 2)) argument = StrSpan(argument.ptr + 2, argument.len - 2);
                    if (equalSpans(argument, name)) ++count;
                }
                return count;
            };
            for (int32_t k = 0; k < anchorN; ++k) {
                const int32_t inGoal = occurrences(goalArgs, goalN, anchors[k]);
                if (inGoal > 1 && occurrences(keyArgs, keyN, anchors[k]) == inGoal) return true;
            }
            return false;
        }

/// @brief Whether the key a route-3 commit or a statement fill admits a
        ///        value through, outside the step blocks, is connected to the
        ///        accepting instruction's goal.
        ///
        /// @details
        /// Such a value becomes a connected value of the block, so the
        /// instruction's continuation keys that hold it are filed
        /// (D-593). The key's text is read from its
        /// template and checked with @ref goalKeyConnected at the key's own
        /// scope; a key that equivalence classes re-keyed was not checked at
        /// its filing and is checked here as written.
        ///
        /// @param mb   The block, neither a weak nor a strong step block
        ///             (asserted; single-threaded seam).
        /// @param pk   The admitting key: an `admissionMapIntegration` key, its
        ///             template minted.
        /// @param goal The accepting instruction's marked goal, on caller-owned
        ///             storage.
        /// @return `true` iff the key's text is connected to @p goal.
        /// @invariant Mints nothing.
        /// @see goalKeyConnected, recordSizeMarker, isAdmittedIntegration,
        ///      updateAdmissionMapIntegration.
inline bool AdmissionMemory::admittingKeyConnectedToGoal(Memory& mb, int64_t pk, StrSpan goal) {
            assert(!mb.weakStepBlock && !mb.strongStepBlock
                && "admittingKeyConnectedToGoal: a step block connects its keys to the induction");
            // The template's view stays valid: the check mints nothing. The key
            // text is canonical at the key's own scope.
            const StatementKey key = Codec<StatementKey>::decode(pk);
            return goalKeyConnected(mb, mb.templateInterner.decodeView(key.orig), goal, key.validity);
        }

/// @brief Whether an integration key filed in an induction step block is
        ///        connected to the induction.
        ///
        /// @details
        /// Where the step-key rule binds (@ref stepKeyRuleBinds: every scope
        /// of a weak step block, an induction scope and below in a strong
        /// one) a route-3 key is filed only when one of its arguments (a `u_`
        /// prefix stripped) is connected
        /// (D-583,
        /// D-584): `rec` or the induction
        /// variable n (`ScopeSyntax::keyHoldsInductionVariable`); a size
        /// marker (`LbMemory::sizeMarkers`; in a strong step block the names
        /// its strong demands caught are size markers too) that the filing
        /// scope does not know equal to the induction's zero
        /// (`Memory::inductionZeroId`): a size of zero bounds the key to
        /// one-point objects, of which every natural number gives one; the
        /// predecessor x of `rec`, when `(in2[x,rec,s])` is known at the
        /// filing scope or above; the successor y of n, when `(in2[n,y,s])`
        /// is known there. The zero test and the two `in2` probes go through
        /// `EquivalenceQueries::elementKnownAtOrAbove`, up to the scope's
        /// classes, so the decision does not depend on which class member the
        /// key names. The key's `marker` and the wildcard `pi_` never connect.
        /// For a copy bound to the second twin premise `rec2`, the name of the
        /// twin partner's predecessor, stands in `rec`'s place and the partner
        /// l in n's (`inductionRolesOf`, D-589).
        ///
        /// @param mb           The step block (single-threaded seam).
        /// @param key          The key text: `u_`-prefixed arguments and a
        ///                     marker, on caller-owned storage.
        /// @param validityName The filing scope (registered, asserted).
        /// @param premiseLevel The premise level of the copy filed under the
        ///                     key, 0 when unbound.
        /// @return `true` iff an argument of @p key is connected.
        /// @invariant Mints nothing; the block has its recursion hypothesis
        ///            `(in2[rec,n,s])` and its zero (both asserted).
        /// @see recordSizeMarker, IntegrationProcessor::prepareIntegrationCore2.
inline bool AdmissionMemory::stepKeyConnected(Memory& mb, StrSpan key, StrSpan validityName,
            int32_t premiseLevel) {
            assert((mb.weakStepBlock || mb.strongStepBlock)
                && "stepKeyConnected: only a step block holds its keys to the induction");
            assert(mb.inductionZeroId != 0 && "stepKeyConnected: a step block knows its induction's zero");
            // The copy's roles: `rec` and n, or the twin partner's
            // predecessor and the partner (D-589).
            const InductionRoles roles = inductionRolesOf(mb, premiseLevel);
            if (gl::ScopeSyntax::keyHoldsInductionRole(key, roles.rec, roles.variable)) return true;
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t argN = getArgsSpans(key, args, ExecutionParameters::MAX_ARITY);
            for (int32_t a = 0; a < argN; ++a) {
                if (startsWithSpan(args[a], "u_", 2)) args[a] = StrSpan(args[a].ptr + 2, args[a].len - 2);
                if (equalSpans(args[a], StrSpan("marker", 6)) || equalSpans(args[a], StrSpan("pi_", 3)))
                    args[a] = StrSpan();
            }

            const NameId scopeId = mb.nameMap.lookup(validityName);
            assert(scopeId != 0 && "stepKeyConnected: the filing scope is registered");
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
            ScratchArena& strings = scratchArenas().forSlot(slot);
            ScratchScope frame(strings);
            // The roles and the induction's zero, copied: the probes read the
            // NameMap the views would alias (I-3).
            const ScratchString recRole = ScratchString::copyFrom(strings, roles.rec.ptr, roles.rec.len);
            const ScratchString variable = ScratchString::copyFrom(strings, roles.variable.ptr, roles.variable.len);
            const StrSpan zeroView = mb.nameMap.decodeView(mb.inductionZeroId);
            const ScratchString zeroCopy = ScratchString::copyFrom(strings, zeroView.ptr, zeroView.len);
            const StrSpan zero(zeroCopy);
            // A size marker connects unless the scope knows it equal to the
            // zero (D-584).
            for (int32_t a = 0; a < argN; ++a) {
                if (args[a].empty()) continue;
                const int32_t valueId = mb.valueInterner.lookup(args[a]);
                if (valueId == 0 || mb.lbMemory.sizeMarkers.lookup(valueId) == 0) continue;
                // `(=[<size marker>,<zero>])` on the string arena.
                const int32_t length = 3 + args[a].len + 1 + zero.len + 2;
                char* bytes = strings.allocBytes(length);
                std::memcpy(bytes, "(=[", 3);
                std::memcpy(bytes + 3, args[a].ptr, static_cast<std::size_t>(args[a].len));
                bytes[3 + args[a].len] = ',';
                std::memcpy(bytes + 4 + args[a].len, zero.ptr, static_cast<std::size_t>(zero.len));
                std::memcpy(bytes + 4 + args[a].len + zero.len, "])", 2);
                if (!equivalenceQueries.elementKnownAtOrAbove(StrSpan(bytes, length), scopeId, mb)) return true;
            }
            // The hypothesis's successor, copied: the probes below read the
            // NameMap the view would alias (I-3).
            StrSpan hypothesisArgs[ExecutionParameters::MAX_ARITY];
            const int32_t hypothesisN = getArgsSpans(mb.nameMap.decodeView(mb.recursionHypothesisId),
                hypothesisArgs, ExecutionParameters::MAX_ARITY);
            assert(hypothesisN == 3 && equalSpans(hypothesisArgs[0], StrSpan("rec", 3))
                && "stepKeyConnected: the recursion hypothesis is (in2[rec,<variable>,<successor>])");
            const ScratchString successor = ScratchString::copyFrom(strings, hypothesisArgs[2].ptr, hypothesisArgs[2].len);
            // `(in2[<first>,<second>,<successor>])` on the string arena.
            const auto successorFact = [&](StrSpan first, StrSpan second) {
                const StrSpan s(successor);
                const int32_t length = 5 + first.len + 1 + second.len + 1 + s.len + 2;
                char* bytes = strings.allocBytes(length);
                int32_t at = 0;
                const auto put = [&](const char* p, int32_t n) {
                    std::memcpy(bytes + at, p, static_cast<std::size_t>(n));
                    at += n;
                };
                put("(in2[", 5); put(first.ptr, first.len); put(",", 1);
                put(second.ptr, second.len); put(",", 1); put(s.ptr, s.len); put("])", 2);
                assert(at == length && "stepKeyConnected: a successor fact is filled exactly");
                return ScratchString::wrap(strings, bytes, length);
            };
            for (int32_t a = 0; a < argN; ++a) {
                if (args[a].empty()) continue;
                const ScratchString predecessorOfRec = successorFact(args[a], StrSpan(recRole));
                if (equivalenceQueries.elementKnownAtOrAbove(StrSpan(predecessorOfRec), scopeId, mb)) return true;
                const ScratchString successorOfVariable = successorFact(StrSpan(variable), args[a]);
                if (equivalenceQueries.elementKnownAtOrAbove(StrSpan(successorOfVariable), scopeId, mb)) return true;
            }
            return false;
        }

/// @brief File the second twin premise's integration goal at the twin
        ///        partner's predecessor.
        ///
        /// @details
        /// In a weak step block whose hypothesis install filed the first twin
        /// premise's goal (`Memory::twinFirstGoalId`), the registration of
        /// `(in2[q,l,s])` at `main` or in the released successor case of the
        /// partner's zero-or-successor OR (`Memory::twinCaseScopeId`), l the
        /// twin partner and s the anchor's successor, names a predecessor q
        /// of the partner. The scope of that registration receives the copy
        /// `(=[q,rec2])` on internal mail (inside an absorb on its output
        /// channel, `Memory::replayMailDuringAbsorb`, since the drained input
        /// is cleared), origin `variable copy`: `rec2`
        /// names the partner's predecessor as `rec` names the induction
        /// variable's (D-581). The second twin
        /// premise's goal is the first goal with `rec2` in place of `rec`:
        /// exchanging the twins maps the one premise onto the other, and the
        /// partner stands where the first premise holds `rec`. It is prepared
        /// at that scope with both markers changeable and every other
        /// argument fixed, as the copy bound to the second premise
        /// (`twinPremiseForbiddenSets`: its premise level, the first premise's
        /// own variables and level forbidden); its int scopes join the first
        /// goal's group, so the two integrations are promoted together, each
        /// into its own parent scope (D-589). `rec2` is
        /// recorded (`Memory::twinPredecessorId`) and takes `rec`'s role for
        /// that copy (`inductionRolesOf`); the goal is filed once.
        ///
        /// @param mb          The weak step block (single-threaded seam).
        /// @param predecessor The NameMap id of q.
        /// @param scopeId     The registration's scope: `main` or the twin
        ///                    successor case (asserted).
        /// @return Void.
        /// @invariant Mints the copy's staging and the goal's preparation
        ///            only.
        /// @see updateAdmissionMap, inductionRolesOf, openIntScope,
        ///      StatementMemory::addStatement.
inline void AdmissionMemory::fileTwinSecondGoal(Memory& mb, NameId predecessor, NameId scopeId) {
            assert(mb.weakStepBlock && mb.twinFirstGoalId != 0 && mb.twinPredecessorId == 0 && predecessor != 0
                && "fileTwinSecondGoal: a twin weak step block files its second goal once");
            assert((scopeId == NameMap::MAIN_ID || (mb.twinCaseScopeId != 0 && scopeId == mb.twinCaseScopeId))
                && "fileTwinSecondGoal: the second goal is filed at main or in the twin successor case");
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
            ScratchArena& strings = scratchArenas().forSlot(slot);
            ScratchScope frame(strings);
            // Owned copies: the staging and the preparation mint the NameMap
            // (I-3).
            const StrSpan scopeView = mb.nameMap.decodeView(scopeId);
            const ScratchString scope = ScratchString::copyFrom(strings, scopeView.ptr, scopeView.len);
            const StrSpan predecessorView = mb.nameMap.decodeView(predecessor);
            const StrSpan alias("rec2", 4);
            const int32_t copyLength = 3 + predecessorView.len + 1 + alias.len + 2;
            char* copyBytes = strings.allocBytes(copyLength);
            int32_t at = 0;
            std::memcpy(copyBytes + at, "(=[", 3); at += 3;
            std::memcpy(copyBytes + at, predecessorView.ptr, static_cast<std::size_t>(predecessorView.len));
            at += predecessorView.len;
            copyBytes[at++] = ',';
            std::memcpy(copyBytes + at, alias.ptr, static_cast<std::size_t>(alias.len)); at += alias.len;
            std::memcpy(copyBytes + at, "])", 2); at += 2;
            assert(at == copyLength);
            const ScratchString copy = ScratchString::wrap(strings, copyBytes, copyLength);
            const int copyLevels[] = { mb.level };
            // The trigger runs inside an absorb, whose drained input channel is
            // cleared afterwards: there the copy rides the absorb's output
            // channel, as a source replay does.
            ColdMail& copyMail = mb.replayMailDuringAbsorb != nullptr
                ? *mb.replayMailDuringAbsorb : mb.sameIterationInternalMail;
            insertInternalStatement(copyMail, mb.nameMap, StrSpan(copy), StrSpan(scope), copyLevels, 1);
            if (parameters.trackHistory) {
                const int cap = parameters.compressor_mode ? parameters.compressor_max_origins_per_expr
                                                           : parameters.max_origin_per_expr;
                addInternalMailOrigin(copyMail, mb.originInterner, StrSpan(copy), StrSpan(scope),
                    OriginTag::variableCopy, nullptr, 0, cap);
            }
            mb.mutatedThisBurst = true;
            mb.hasWork = true;
            mb.twinPredecessorId = mb.nameMap.encode(alias);
            const StrReplacement recToAlias{ StrSpan("rec", 3), alias };
            const ScratchString goal = replaceKeysScratch(strings, mb.nameMap.decodeView(mb.twinFirstGoalId),
                                                          &recToAlias, 1);
            // Every argument but the two markers is fixed, sorted unique.
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t argN = getArgsSpans(StrSpan(goal), args, ExecutionParameters::MAX_ARITY);
            StrSpan fixed[ExecutionParameters::MAX_ARITY];
            int32_t fixedN = 0;
            for (int32_t a = 0; a < argN; ++a) {
                if (equalSpans(args[a], StrSpan("marker1", 7)) || equalSpans(args[a], StrSpan("marker2", 7))) continue;
                int32_t at = fixedN;
                while (at > 0 && compareSpans(args[a], fixed[at - 1]) < 0) --at;
                if (at > 0 && equalSpans(args[a], fixed[at - 1])) continue;
                for (int32_t k = fixedN; k > at; --k) fixed[k] = fixed[k - 1];
                fixed[at] = args[a];
                ++fixedN;
            }
            assert(argN - fixedN >= 2 && "fileTwinSecondGoal: the first goal holds both markers");
            IntegrationForbiddenSets binding;
            twinPremiseForbiddenSets(mb, mb.twinFirstLevel, mb.twinSecondLevel, 1, binding);
            RT_SCOPE_HERE("UAM_PREPARE_INTEGRATION");
            integrationProcessor.prepareIntegration(StrSpan(goal), fixed, fixedN, mb, StrSpan(scope),
                StrSpan(), false, true, &binding, 1);
        }

/// @brief Record a value as a size marker of an induction step block, or as
        ///        a connected value of any other block.
        ///
        /// @details
        /// A value filled into an integration instruction of a weak or strong
        /// step block through a connected key (@ref admittingKeyConnected)
        /// carries a size bound by `rec`, n or a caught smaller value, and
        /// every later key holding it is connected too
        /// (D-583). In a strong step block the
        /// two names a strong demand catches are recorded when the catch opens
        /// its induction scope (D-584). In any
        /// other block a value admitted through a key connected to the
        /// accepting instruction's goal (@ref admittingKeyConnectedToGoal) is
        /// a connected value (D-593). The record holds
        /// for the whole block: the value's `valueInterner` id is kept in
        /// `LbMemory::sizeMarkers`, which no scope wipe touches, since a
        /// witness name is never reused within a block.
        ///
        /// @param mb    The block (single-threaded seam).
        /// @param value The bare value (no `u_` prefix, not a marker).
        /// @invariant Mints the value into `valueInterner` when absent and its
        ///            id into `sizeMarkers`; idempotent.
        /// @see stepKeyConnected, admittingKeyConnected, goalKeyConnected.
inline void AdmissionMemory::recordSizeMarker(Memory& mb, StrSpan value) {
            assert(!value.empty() && !startsWithSpan(value, "u_", 2)
                && !equalSpans(value, StrSpan("marker", 6)) && !equalSpans(value, StrSpan("pi_", 3))
                && "recordSizeMarker: a bare filled value");
            mb.lbMemory.sizeMarkers.mint(mb.valueInterner.encode(value));
        }

/// @brief Record a witness that its route-3 admission wrote into an
        ///        integration instruction as a one-way witness.
        ///
        /// @details
        /// A witness admitted through an `admissionMapIntegration` key updates
        /// the instructions waiting under that key once, at its admission; its
        /// role there is fixed (D-585). The route-3 commit
        /// (`isAdmittedIntegration`) calls this for every witness it writes into
        /// an instruction, and from then on the statement fill
        /// (`updateAdmissionMapIntegration`) never tries the witness, or a name
        /// of its class at the deposit scope, as a marker again
        /// (`oneWayWitnessAtScope`). The witness is interned in the block's
        /// `NameMap` and its id kept in `LbMemory::oneWayWitnesses`; a witness
        /// name is never reused within a block, so no scope wipe removes it.
        ///
        /// @param mb      The owning block (single-threaded seam).
        /// @param witness The bare admitted witness (no `u_` prefix, not a marker).
        /// @invariant Mints the witness into the NameMap when absent and its id
        ///            into `oneWayWitnesses`; idempotent.
        /// @see oneWayWitnessAtScope, isAdmittedIntegration.
inline void AdmissionMemory::recordOneWayWitness(Memory& mb, StrSpan witness) {
            assert(!witness.empty() && !startsWithSpan(witness, "u_", 2)
                && !equalSpans(witness, StrSpan("marker", 6))
                && "recordOneWayWitness: a bare admitted witness");
            mb.lbMemory.oneWayWitnesses.mint(mb.nameMap.encode(witness));
        }

/// @brief Whether a statement-fill argument is a one-way witness or a name
        ///        of a one-way witness's class at the deposit scope.
        ///
        /// @details
        /// The decision of the one-way rule (D-585): the
        /// statement fill skips such an argument and tries no key with it as the
        /// marker. The argument itself is looked up first; otherwise its
        /// representative in the scope's complete class view is compared with
        /// each recorded witness's representative there
        /// (`CanonicalTemplateMatcher::canonicalMemberIdAtScope`), so a class-equal
        /// spelling of a recorded witness is skipped as well. A name the block's
        /// `NameMap` never interned is neither recorded nor in a class.
        ///
        /// @param mb    The owning block, with its classes at @p scope current.
        /// @param value The bare argument (any lifetime; not copied).
        /// @param scope The deposit scope (registered, nonzero).
        /// @return `true` iff @p value or a name of its class at @p scope is
        ///         recorded in `LbMemory::oneWayWitnesses`.
        /// @invariant Mints nothing; `false` at once while nothing is recorded.
        /// @see recordOneWayWitness, updateAdmissionMapIntegration.
inline bool AdmissionMemory::oneWayWitnessAtScope(Memory& mb, StrSpan value, NameId scope) {
            assert(scope != 0 && "oneWayWitnessAtScope: a registered scope");
            const ColdHashSet<PodKeyStore<int32_t>>& witnesses = mb.lbMemory.oneWayWitnesses;
            if (witnesses.empty()) return false;
            const NameId member = mb.nameMap.lookup(value);
            if (member == 0) return false;
            if (witnesses.lookup(member) != 0) return true;
            const NameId representative = canonicalTemplates.canonicalMemberIdAtScope(mb, member, scope);
            for (int32_t id = 1; id <= witnesses.count(); ++id)
                if (canonicalTemplates.canonicalMemberIdAtScope(mb, witnesses.keyAt(id), scope) == representative)
                    return true;
            return false;
        }

/// @brief The scope whose route-3 integration demands also serve a scope.
        ///
        /// @details
        /// An integration demand serves its own scope and, beyond it, only an
        /// induction scope (`_is_`, `ScopeSyntax::isInductionScope`) or an OR
        /// case (`ScopeSyntax::classifyOrScopeView` kind `Disintegration`)
        /// whose parent chain up to the demand's scope runs through induction
        /// scopes only: the owner is `ScopeSyntax::inductionRootOf` of the
        /// scope's parent (D-572, maintainer
        /// design). Every other scope reads its own demands only; no owner is
        /// a defined result, returned as 0.
        ///
        /// @param mb      Owning block.
        /// @param scopeId A minted validity id.
        /// @return The id of the serving owner, a strict ancestor of
        ///         @p scopeId, for an induction scope or an OR case; 0 for any
        ///         other scope.
        /// @invariant Pure; allocates and mints nothing.
        /// @see integrationDemandVisible, matchingVisibleIntegrationDemands,
        ///      hypothesisShotOnPath,
        ///      RejectedSourceReplay::wakeDescendantIntegrationParks.
inline NameId AdmissionMemory::ancestorDemandOwner(const Memory& mb, NameId scopeId) {
            assert(scopeId > 0 && "ancestorDemandOwner: expected a minted validity id");
            StrSpan signature, disjunct;
            if (!gl::ScopeSyntax::isInductionScope(mb.nameMap, scopeId)
                    && gl::ScopeSyntax::classifyOrScopeView(mb.nameMap, scopeId, signature, disjunct)
                        != OrScopeKind::Disintegration)
                return 0;
            const NameId parent = mb.nameMap.parentOf(scopeId);
            assert(parent != 0 && "ancestorDemandOwner: an induction scope or an OR case has a parent");
            return gl::ScopeSyntax::inductionRootOf(mb.nameMap, parent);
        }

/// @brief Whether the induction hypothesis shot on the path from a scope up to
        ///        the owner of the demand that serves it.
        ///
        /// @details
        /// A commit served by an ancestor's integration demand always admits
        /// the witness; it records the witness in the ancestor's instruction and
        /// continues that instruction at @p scopeId only after the hypothesis
        /// shot at @p scopeId or at a scope between it and @p owner
        /// (`LbMemory::hypothesisShots`, minted by the statement door for a
        /// `recursionHead` statement; D-572,
        /// maintainer design).
        ///
        /// @param mb      Owning block.
        /// @param scopeId The served scope, a minted validity id.
        /// @param owner   The serving owner, a strict ancestor of @p scopeId.
        /// @return True iff a scope from @p scopeId up to, not including,
        ///         @p owner is in `LbMemory::hypothesisShots`.
        /// @invariant Pure; allocates and mints nothing.
        /// @see ancestorDemandOwner, isAdmittedIntegration.
inline bool AdmissionMemory::hypothesisShotOnPath(const Memory& mb, NameId scopeId, NameId owner) {
            assert(scopeId > 0 && owner > 0 && mb.nameMap.isStrictAncestor(owner, scopeId)
                && "hypothesisShotOnPath: the owner is a strict ancestor of the scope");
            for (NameId scope = scopeId; scope != owner; scope = mb.nameMap.parentOf(scope))
                if (mb.lbMemory.hypothesisShots.lookup(scope) != 0) return true;
            return false;
        }

/// @brief Whether a route-3 integration demand for a marked template is
        ///        visible at a scope.
        ///
        /// @details
        /// A demand of `admissionMapIntegration` is visible at its own scope
        /// and at every OR case it serves (`ancestorDemandOwner`), so the case
        /// that can build a witness serves the integration prepared where its
        /// case split began (D-572). The
        /// probe tests the scope itself first, then the serving owner; every
        /// comparison is canonical at @p scopeId
        /// (`CanonicalTemplateMatcher::canonicalTemplatePresent` with the
        /// owning scope as source and @p scopeId as target).
        /// A wildcard key (`pi_` at one non-marker position,
        /// D-586) matches when a
        /// single-position variant of the query (`makeWildcardQueryScratch`)
        /// equals it; the variants are probed after the query itself and only
        /// in a block that filed such a key (`integrationWildcardFiled`).
        ///
        /// @param mb           Owning block.
        /// @param renamedMarked The marked template in key spelling (`u_`
        ///                     prefixes on every non-marker argument).
        /// @param validityName The probe scope, registered.
        /// @param scopeId      The id of @p validityName.
        /// @return True iff a key owned at @p validityName or at its serving
        ///         owner matches canonically at @p validityName.
        /// @invariant Adds no admission record; may extend the derived
        ///            class-aware template index only.
        /// @see matchingVisibleIntegrationDemands, isAdmittedIntegration,
        ///      RejectedSourceReplay::hasPotentialParkedDemand.
inline bool AdmissionMemory::integrationDemandVisible(Memory& mb, StrSpan renamedMarked,
            StrSpan validityName, NameId scopeId) {
            assert(scopeId != 0 && mb.nameMap.lookup(validityName) == scopeId
                && "integrationDemandVisible: the probe scope is registered under its id");
            const auto& demands = mb.overallHashMemory.admissionMapIntegration;
            if (demands.empty()) return false;
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
            ScratchArena& strings = scratchArenas().forSlot(slot);
            ScratchScope frame(strings);
            // Owned copies before any probe: the class-aware match mints
            // templates.
            const ScratchString scope = ScratchString::copyFrom(strings, validityName.ptr, validityName.len);
            const NameId owner = ancestorDemandOwner(mb, scopeId);
            const StrSpan ownerView = owner == 0 ? StrSpan() : mb.nameMap.decodeView(owner);
            const ScratchString ownerName = ScratchString::copyFrom(strings, ownerView.ptr, ownerView.len);
            const auto present = [&](StrSpan query) {
                if (canonicalTemplates.canonicalTemplatePresent(mb, demands, mb.lbMemory.canonicalUses.integrationMatches,
                        query, StrSpan(scope), scopeId)) return true;
                return owner != 0 && canonicalTemplates.canonicalTemplatePresent(mb, demands,
                    mb.lbMemory.canonicalUses.integrationMatches, query, StrSpan(ownerName), scopeId);
            };
            if (present(renamedMarked)) return true;
            if (!integrationWildcardFiled(mb)) return false;
            // A wildcard key holds `pi_` at one non-marker position
            // (D-586).
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t n = getArgsSpans(renamedMarked, args, ExecutionParameters::MAX_ARITY);
            for (int32_t i = 0; i < n; ++i) {
                if (equalSpans(args[i], StrSpan("marker", 6))) continue;
                ScratchScope variantFrame(strings);
                if (present(StrSpan(makeWildcardQueryScratch(renamedMarked, i, strings)))) return true;
            }
            return false;
        }

/// @brief Collect the route-3 integration demand keys visible at a scope
        ///        that canonically match a marked template.
        ///
        /// @details
        /// The matching keys owned at @p validityName come first, in the
        /// order `matchingCanonicalTemplates` returns them, then, for an OR
        /// case, those owned at the scope its demands come from
        /// (`ancestorDemandOwner`, D-572). The wildcard keys follow
        /// (D-586): for each
        /// non-marker position in order, the keys equal to the query's variant
        /// with `pi_` there (`makeWildcardQueryScratch`), own scope before
        /// owner, probed only in a block that filed such a key
        /// (`integrationWildcardFiled`).
        /// Every comparison is canonical at @p scopeId. The keys are the packed
        /// (template, owning scope) keys of `admissionMapIntegration`, so a
        /// caller reads and commits the instructions of an ancestor's key
        /// exactly as those of its own.
        ///
        /// @param mb            Owning block.
        /// @param renamedMarked The marked template in key spelling (`u_`
        ///                      prefixes on every non-marker argument).
        /// @param validityName  The probe scope, registered.
        /// @param scopeId       The id of @p validityName.
        /// @param out           Empty caller-owned vector receiving the keys.
        /// @param arena         The caller's page-tier scratch arena.
        /// @return Nothing.
        /// @invariant Adds no admission record; may extend the derived
        ///            class-aware template index only.
        /// @see integrationDemandVisible, isAdmittedIntegration,
        ///      integrationTrialGates, integrationEnclosingFill,
        ///      integrationDemandOpen, markWitnessLinkRefusals.
inline void AdmissionMemory::matchingVisibleIntegrationDemands(Memory& mb, StrSpan renamedMarked,
            StrSpan validityName, NameId scopeId, PagedVector<int64_t>& out, ScratchArena& arena) {
            assert(scopeId != 0 && mb.nameMap.lookup(validityName) == scopeId
                && "matchingVisibleIntegrationDemands: the probe scope is registered under its id");
            assert(out.size() == 0 && "matchingVisibleIntegrationDemands: the output starts empty");
            const auto& demands = mb.overallHashMemory.admissionMapIntegration;
            const NameId owner = ancestorDemandOwner(mb, scopeId);
            const auto collect = [&](StrSpan query) {
                DirtyState dirty = DirtyState::Clean;
                PagedVector<int64_t> found(&arena, &dirty);
                canonicalTemplates.matchingCanonicalTemplates(mb, demands, mb.lbMemory.canonicalUses.integrationMatches,
                    query, scopeId, scopeId, found, arena);
                for (int32_t k = 0; k < found.size(); ++k) out.push_back(found[k]);
                if (owner == 0) return;
                PagedVector<int64_t> owned(&arena, &dirty);
                canonicalTemplates.matchingCanonicalTemplates(mb, demands, mb.lbMemory.canonicalUses.integrationMatches,
                    query, owner, scopeId, owned, arena);
                for (int32_t k = 0; k < owned.size(); ++k) out.push_back(owned[k]);
            };
            collect(renamedMarked);
            if (!integrationWildcardFiled(mb)) return;
            // The wildcard keys, one non-marker position at a time
            // (D-586).
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
            ScratchArena& strings = scratchArenas().forSlot(slot);
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t n = getArgsSpans(renamedMarked, args, ExecutionParameters::MAX_ARITY);
            for (int32_t i = 0; i < n; ++i) {
                if (equalSpans(args[i], StrSpan("marker", 6))) continue;
                ScratchScope variantFrame(strings);
                collect(StrSpan(makeWildcardQueryScratch(renamedMarked, i, strings)));
            }
        }

/// @brief Whether an integration instruction's path list refuses an
        ///        admission route (D-529).
        ///
        /// @details
        /// The witness-link gate. A route admits one or more witnesses; the
        /// other part of each admitted witness's existence names some
        /// arguments of the witness's own definition set — @p linked
        /// (`witnessLinkArguments`). The instruction refuses the route when
        /// one of them is a member of @p pathList: an original variable of the
        /// instruction's integration goal or a value admitted earlier on the
        /// same path. Both sides are compared in canonical form at @p scope
        /// (the complete class view, `canonicalAdmissionText`), so a linked
        /// argument equal by class to a list member refuses as well. Only
        /// induction step blocks (`Memory::recursionAuxyImplId != 0`) gate; in
        /// any other block, with an empty list or no linked argument, nothing
        /// refuses.
        ///
        /// The history extension (D-522): when the
        /// direct check accepts and @p history carries labels, the derivation
        /// tree of the existence being disintegrated is checked as well
        /// (`witnessHistoryRefuses`).
        ///
        /// @param mb          Owning block.
        /// @param pathList    The instruction's path list `[name,...]`, or empty.
        /// @param linked      The route's linked arguments (null when
        ///                    @p linkedCount is 0).
        /// @param linkedCount Number of @p linked.
        /// @param scope       The admission scope (registered).
        /// @param history     The route's witness-history probe, or null.
        /// @return True iff the instruction refuses the route.
        /// @invariant Mints nothing; reads the canonical view and the history
        ///            only.
        /// @see witnessLinkArguments, witnessHistoryRefuses, isAdmittedIntegration.
inline bool AdmissionMemory::witnessLinkRefuses(Memory& mb, StrSpan pathList,
            const StrSpan* linked, int32_t linkedCount, NameId scope,
            const WitnessHistoryProbe* history) {
            const bool tree = history != nullptr && history->active();
            if (mb.recursionAuxyImplId == 0 || (linkedCount == 0 && !tree) || pathList.empty()) return false;
            assert(scope != 0 && "witnessLinkRefuses: expected a registered scope");
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
            ScratchArena& strings = scratchArenas().forSlot(slot);
            ScratchScope frame(strings);
            const ScratchString canonicalList = canonicalPathListScratch(mb, pathList, scope, strings);
            StrSpan members[ExecutionParameters::MAX_PATH_VARIABLES];
            const int32_t memberN = getArgsSpans(StrSpan(canonicalList), members,
                ExecutionParameters::MAX_PATH_VARIABLES);
            for (int32_t i = 0; i < linkedCount; ++i) {
                const ScratchString canonical = canonicalTemplates.canonicalAdmissionText(mb, linked[i], scope, strings);
                for (int32_t m = 0; m < memberN; ++m)
                    if (equalSpans(members[m], StrSpan(canonical))) return true;
            }
            return tree && witnessHistoryRefuses(mb, members, memberN, scope, *history);
        }

/// @brief Whether the derivation tree of the existence being disintegrated
        ///        links an admitted witness to an instruction's path list
        ///        (D-522).
        ///
        /// @details
        /// The history extension of the witness-link gate. Walks the
        /// derivation tree of @p history's start (`DerivationTree::walk`: one
        /// derivation per statement as buildStack takes it, premises only)
        /// and refuses when an ancestor has a leaf argument whose canonical
        /// form at @p scope is one of @p members and whose definition set,
        /// typed by position in that leaf (`ScopeTypeResolver::definitionSetOf`),
        /// is the definition set of one of the route's checked witnesses
        /// (operator outputs carry no label). The walk stops at the first hit.
        /// A connection that comes from a forced disintegration does not
        /// count: the walk ends, without checking it, at every forced product
        /// of the block (`ForcedProductMatch`).
        ///
        /// @param mb      Owning block (an induction step block).
        /// @param members The instruction's path list in canonical form at
        ///                @p scope, as spans.
        /// @param memberN Number of @p members.
        /// @param scope   The admission scope (registered).
        /// @param history The route's probe (active).
        /// @return True iff an ancestor links a checked witness to a member.
        /// @invariant Mints nothing; reads the history, the canonical view and
        ///            the definitions only; asserts that history is tracked.
        /// @see witnessLinkRefuses, DerivationTree::walk, WitnessHistoryProbe,
        ///      ForcedProductMatch.
inline bool AdmissionMemory::witnessHistoryRefuses(Memory& mb, const StrSpan* members, int32_t memberN,
            NameId scope, const WitnessHistoryProbe& history) {
            assert(history.active() && scope != 0 && "witnessHistoryRefuses: an active probe at a registered scope");
            assert(parameters.trackHistory && "witnessHistoryRefuses: the gate reads the derivation history");
            if (memberN == 0) return false;
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
            ScratchArena& strings = scratchArenas().forSlot(slot);
            const unsigned pageSlot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
            ForcedProductMatch forced(mb, canonicalTemplates, genScratchArenas().forSlot(pageSlot), strings);
            return DerivationTree::walk(mb, history.start, history.startScope,
                [&](StrSpan ancestor, StrSpan) {
                    return forEachLeafSpan(ancestor, [&](StrSpan leaf) {
                        StrSpan args[ExecutionParameters::MAX_ARITY];
                        const int32_t argN = getArgsSpans(leaf, args, ExecutionParameters::MAX_ARITY);
                        for (int32_t a = 0; a < argN; ++a) {
                            ScratchScope argScope(strings);
                            const ScratchString canonical =
                                canonicalTemplates.canonicalAdmissionText(mb, args[a], scope, strings);
                            bool member = false;
                            for (int32_t m = 0; m < memberN && !member; ++m)
                                member = equalSpans(members[m], StrSpan(canonical));
                            if (!member) continue;
                            if (!history.hasLabel(scopeTypes.definitionSetOf(leaf, args[a]))) continue;
                            return true;
                        }
                        return false;
                    });
                },
                [&](StrSpan ancestor, StrSpan ancestorScope) { return forced.matches(ancestor, ancestorScope); });
        }

/// @brief The definition set of a witness in one element of its existence
        ///        (D-522).
        ///
        /// @details Types @p part by position (`ScopeTypeResolver::definitionSetOf`)
        /// and returns the witness's definition set: the label the
        /// witness-history check compares an ancestor's argument against.
        ///
        /// @param part    One element of the witness's existence, in its
        ///                spelling.
        /// @param witness The witness's spelling in @p part.
        /// @return The definition-set label, or empty when @p part does not
        ///         name @p witness.
        /// @invariant Allocates and mints nothing.
        /// @see witnessHistoryRefuses, WitnessHistoryProbe::addLabel.
inline StrSpan AdmissionMemory::witnessDefinitionSet(StrSpan part, StrSpan witness) {
            return scopeTypes.definitionSetOf(part, witness);
        }

/// @brief Append the arguments one existence part links its witness to
        ///        (D-529).
        ///
        /// @details
        /// The arguments of @p part whose definition set equals the witness's
        /// (`ScopeTypeResolver::sameDefinitionSetArguments`), the witness
        /// itself and the batch's anchor arguments (the constants every
        /// theorem shares, `ScopeSyntax::anchorArgumentSpans`) excluded, are
        /// appended to `out[count..)` unless already present.
        ///
        /// @param mb      Owning block (its anchor chain).
        /// @param part    The other part of the witness's existence (flat MPL).
        /// @param witness The witness's spelling in @p part.
        /// @param out     Caller array receiving spans into @p part.
        /// @param count   Spans already in @p out.
        /// @param cap     Capacity of @p out.
        /// @return The new span count.
        /// @invariant Allocates and mints nothing.
        /// @see witnessLinkRefuses, ScopeTypeResolver::sameDefinitionSetArguments.
inline int32_t AdmissionMemory::witnessLinkArguments(Memory& mb, StrSpan part, StrSpan witness,
            StrSpan* out, int32_t count, int32_t cap) {
            StrSpan same[ExecutionParameters::MAX_ARITY];
            const int32_t sameN = scopeTypes.sameDefinitionSetArguments(part, witness, same,
                ExecutionParameters::MAX_ARITY);
            if (sameN == 0) return count;
            StrSpan anchors[ExecutionParameters::MAX_ARITY];
            const int32_t anchorN = gl::ScopeSyntax::anchorArgumentSpans(mb, StrSpan(anchorInfo.name),
                anchors, ExecutionParameters::MAX_ARITY);
            for (int32_t i = 0; i < sameN; ++i) {
                bool skip = false;
                for (int32_t k = 0; k < anchorN && !skip; ++k) skip = equalSpans(anchors[k], same[i]);
                for (int32_t k = 0; k < count && !skip; ++k) skip = equalSpans(out[k], same[i]);
                if (skip) continue;
                assert(count < cap && "witnessLinkArguments: linked argument count exceeds capacity");
                out[count++] = same[i];
            }
            return count;
        }

/// @brief The path list of a new integration tree: the non-anchor fixed
        ///        arguments of its goal (D-529).
        ///
        /// @details
        /// Strips a `u_` prefix from each of @p fixedArgs, drops the batch's
        /// anchor arguments and builds the list text (`buildPathListScratch`).
        /// The caller passes the arguments the integration keeps fixed.
        ///
        /// @param mb        Owning block (its anchor chain).
        /// @param fixedArgs The goal's fixed arguments (null when @p n is 0).
        /// @param n         Number of @p fixedArgs.
        /// @param arena     String-tier scratch arena receiving the text.
        /// @return The list text, or empty when nothing remains.
        /// @invariant Mints nothing.
        /// @see IntegrationProcessor::prepareIntegration, buildPathListScratch.
inline ScratchString AdmissionMemory::rootPathListScratch(Memory& mb, const StrSpan* fixedArgs,
            int32_t n, ScratchArena& arena) {
            StrSpan anchors[ExecutionParameters::MAX_ARITY];
            const int32_t anchorN = gl::ScopeSyntax::anchorArgumentSpans(mb, StrSpan(anchorInfo.name),
                anchors, ExecutionParameters::MAX_ARITY);
            StrSpan names[ExecutionParameters::MAX_PATH_VARIABLES];
            int32_t nameN = 0;
            for (int32_t i = 0; i < n; ++i) {
                StrSpan name = fixedArgs[i];
                if (startsWithSpan(name, "u_", 2)) name = StrSpan(name.ptr + 2, name.len - 2);
                bool anchor = false;
                for (int32_t k = 0; k < anchorN && !anchor; ++k) anchor = equalSpans(anchors[k], name);
                if (anchor) continue;
                assert(nameN < ExecutionParameters::MAX_PATH_VARIABLES
                    && "rootPathListScratch: name count exceeds MAX_PATH_VARIABLES");
                names[nameN++] = name;
            }
            return buildPathListScratch(arena, names, nameN);
        }

/// @brief A path list in canonical form at a scope
        ///        (D-529).
        ///
        /// @details
        /// Replaces every name by its class representative in the complete
        /// view at @p scope (`canonicalAdmissionText`) and rebuilds the sorted,
        /// duplicate-free text — the form every stored list keeps, like the
        /// rest of `admissionMapIntegration`.
        ///
        /// @param mb    Owning block.
        /// @param list  The list text, or empty.
        /// @param scope The scope (registered).
        /// @param arena String-tier scratch arena receiving the text.
        /// @return The canonical list text, or empty for the empty list.
        /// @invariant Mints nothing into the block's interners.
        /// @see buildPathListScratch, IntegrationProcessor::prepareIntegrationCore2.
inline ScratchString AdmissionMemory::canonicalPathListScratch(Memory& mb, StrSpan list, NameId scope,
            ScratchArena& arena) {
            if (list.empty()) return ScratchString();
            // Name by name: a list may hold more names than one expression
            // has arguments (the canonical rewrite of a text is MAX_ARITY-bound).
            StrSpan names[ExecutionParameters::MAX_PATH_VARIABLES];
            const int32_t nameN = getArgsSpans(list, names, ExecutionParameters::MAX_PATH_VARIABLES);
            for (int32_t i = 0; i < nameN; ++i)
                names[i] = StrSpan(canonicalTemplates.canonicalAdmissionText(mb, names[i], scope, arena));
            return buildPathListScratch(arena, names, nameN);
        }

/// @brief The payload value recording that an instruction refused a source
        ///        (D-529).
        ///
        /// @details `refused:` followed by the source in canonical form at
        /// @p scope, so a later class merge leaves the mark comparable.
        ///
        /// @param mb     Owning block.
        /// @param source The source existence.
        /// @param scope  The scope (registered).
        /// @param arena  String-tier scratch arena receiving the text.
        /// @return The mark text.
        /// @invariant Mints nothing into the block's interners.
        /// @see markWitnessLinkRefusals, integrationDemandOpen.
inline ScratchString AdmissionMemory::refusalMarkScratch(Memory& mb, StrSpan source, NameId scope,
            ScratchArena& arena) {
            static constexpr char kPrefix[] = "refused:";
            constexpr int32_t prefixLen = static_cast<int32_t>(sizeof(kPrefix) - 1);
            const ScratchString canonical = canonicalTemplates.canonicalAdmissionText(mb, source, scope, arena);
            const StrSpan body(canonical);
            char* bytes = arena.allocBytes(prefixLen + body.len);
            std::memcpy(bytes, kPrefix, static_cast<std::size_t>(prefixLen));
            std::memcpy(bytes + prefixLen, body.ptr, static_cast<std::size_t>(body.len));
            return ScratchString::wrap(arena, bytes, prefixLen + body.len);
        }

/// @brief Mark, under one marked key, the instructions whose path list
        ///        refused a parked source (D-529).
        ///
        /// @details
        /// Called when an expansion parks a witness under @p markedExpr in an
        /// induction step block. Every instruction stored under the keys that
        /// canonically match @p markedExpr at @p validityName (owned there or at a strict ancestor) and refuses the
        /// route (`witnessLinkRefuses` with @p linked) records the refusal mark
        /// of @p source (`refusalMarkScratch`) in its payload. A replay probe
        /// (`integrationDemandOpen`) then no longer counts that instruction as
        /// a demand for the source, so the park-time rendezvous cannot replay
        /// a source the same instructions refuse again; a new or unmarked
        /// instruction still wakes it. A copy bound to one premise whose
        /// forbidden sets @p source hits refuses it too, and records the same
        /// mark: the replay decides on the source as its arriving expression
        /// (D-579); so does a root copy
        /// outside every int scope whose initiation gate @p source fails
        /// (`premiseCopyRefuses`, D-578).
        ///
        /// @param mb           Owning block (single-threaded seam).
        /// @param markedExpr   The park key, bare form.
        /// @param validityName The park scope (registered).
        /// @param linked       The route's linked arguments (null when
        ///                     @p linkedCount is 0).
        /// @param linkedCount  Number of @p linked.
        /// @param source       The parked source existence.
        /// @param history      The route's witness-history probe, or null
        ///                     (D-522); in an induction step block
        ///                     @p linkedCount is positive or @p history is
        ///                     active, unless the block split identical goals
        ///                     by premise.
        /// @return Void.
        /// @invariant Writes only payload values of refusing instructions.
        /// @see integrationDemandOpen, witnessLinkRefuses, argumentsHitForbiddenSets.
inline void AdmissionMemory::markWitnessLinkRefusals(Memory& mb, StrSpan markedExpr, StrSpan validityName,
            const StrSpan* linked, int32_t linkedCount, StrSpan source,
            const WitnessHistoryProbe* history) {
            assert((mb.premiseSplitFiled
                    || (mb.recursionAuxyImplId != 0
                        && (linkedCount > 0 || (history != nullptr && history->active()))))
                && "markWitnessLinkRefusals: only a gated park in an induction step block");
            if (!admissionPolicy.isIntegrationInstructionKeyEligible(markedExpr)) return;
            const NameId scopeId = mb.nameMap.lookup(validityName);
            assert(scopeId != 0 && "markWitnessLinkRefusals: expected a registered scope");
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
            ScratchArena& arena = genScratchArenas().forSlot(slot);
            ScratchArena& strings = scratchArenas().forSlot(slot);
            ScratchScope frame(arena);
            ScratchScope stringFrame(strings);
            const ScratchString renamed =
                gl::DefinitionExpansion::prefixNonMarkerArgumentsWithUScratch(strings, markedExpr);
            DirtyState dirty = DirtyState::Clean;
            PagedVector<int64_t> matches(&arena, &dirty);
            matchingVisibleIntegrationDemands(mb, StrSpan(renamed), validityName, scopeId, matches, arena);
            const ScratchString mark = refusalMarkScratch(mb, source, scopeId, strings);
            for (int32_t m = 0; m < matches.size(); ++m) {
                const int64_t pk = matches[m];
                ArenaIntegrationMap current(&arena, &mb.valueInterner);
                current.buildFromCold(mb.overallHashMemory.admissionMapIntegration, pk);
                bool changed = false;
                for (int32_t i = 0; i < current.entries.size(); ++i) {
                    const IntegrationForbiddenSets forbidden = current.entries[i].forbidden;
                    if (!premiseCopyRefuses(mb, scopeId, forbidden, source)) {
                        const int32_t listId = current.entries[i].pathVarsId;
                        if (listId == 0) continue;
                        if (!witnessLinkRefuses(mb, mb.valueInterner.decodeView(listId), linked, linkedCount,
                                scopeId, history))
                            continue;
                    }
                    const int32_t markId = mb.valueInterner.encode(StrSpan(mark));
                    if (current.countValue(i, markId)) continue;
                    current.insertValue(i, markId);
                    changed = true;
                }
                if (!changed) continue;
                current.writeToCold(mb.overallHashMemory.admissionMapIntegration, pk);
                mb.lbMemory.canonicalUses.integrationDemands.changed(pk);
            }
        }

/// @brief Whether an integration instruction under a marked key still
        ///        demands a parked source (D-529).
        ///
        /// @details
        /// The replay twin of the route-3 probe in an induction step block: a
        /// key counts as a demand for @p source only if one instruction under
        /// the keys that canonically match @p markedExpr at @p validityName (owned there or at a strict ancestor)
        /// carries no refusal mark of @p source. Outside step blocks it is the
        /// plain probe.
        ///
        /// @param mb           Owning block.
        /// @param markedExpr   The park key, bare form.
        /// @param validityName The demanding scope.
        /// @param source       The parked source existence.
        /// @return True iff an unmarked instruction is stored under the key.
        /// @invariant Mints nothing into the block's interners.
        /// @see markWitnessLinkRefusals, RejectedSourceReplay::queueParkedExistence.
inline bool AdmissionMemory::integrationDemandOpen(Memory& mb, StrSpan markedExpr, StrSpan validityName,
            StrSpan source) {
            const StrSpan marker("marker", 6);
            if (mb.recursionAuxyImplId == 0)
                return isAdmittedIntegration(mb, markedExpr, marker, markedExpr, validityName, false);
            if (!admissionPolicy.isIntegrationInstructionKeyEligible(markedExpr)) return false;
            const NameId scopeId = mb.nameMap.lookup(validityName);
            if (scopeId == 0) return false;
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
            ScratchArena& arena = genScratchArenas().forSlot(slot);
            ScratchArena& strings = scratchArenas().forSlot(slot);
            ScratchScope frame(arena);
            ScratchScope stringFrame(strings);
            const ScratchString renamed =
                gl::DefinitionExpansion::prefixNonMarkerArgumentsWithUScratch(strings, markedExpr);
            DirtyState dirty = DirtyState::Clean;
            PagedVector<int64_t> matches(&arena, &dirty);
            matchingVisibleIntegrationDemands(mb, StrSpan(renamed), validityName, scopeId, matches, arena);
            if (matches.size() == 0) return false;
            const ScratchString mark = refusalMarkScratch(mb, source, scopeId, strings);
            const int32_t markId = mb.valueInterner.lookup(StrSpan(mark));
            if (markId == 0) return true;
            for (int32_t m = 0; m < matches.size(); ++m) {
                ArenaIntegrationMap entries(&arena, &mb.valueInterner);
                entries.buildFromCold(mb.overallHashMemory.admissionMapIntegration, matches[m]);
                for (int32_t i = 0; i < entries.entries.size(); ++i)
                    if (!entries.countValue(i, markId)) return true;
            }
            return false;
        }

/// @brief Build the double-marker demand template of one disjunct with two
        ///        open arguments (D-492).
        ///
        /// @details
        /// An open argument is one without the `u_` prefix. The disjunct
        /// qualifies when it has exactly two argument positions that are open
        /// and the two hold different names; the earlier open position becomes
        /// `marker1`, the later `marker2`, and every other argument keeps its
        /// `u_` spelling. Any other shape (no, one or three open positions, or
        /// one name at both open positions) is not a double-marker disjunct —
        /// a defined classification, not a failure.
        ///
        /// @param disjunct One OR disjunct in its `u_`-prefixed instruction form.
        /// @param arena    String-tier scratch arena receiving the template.
        /// @param out      Receives the template when the disjunct qualifies.
        /// @return True iff the disjunct has exactly two distinct open arguments.
        /// @invariant Pure: mints nothing; allocates only on @p arena.
        /// @see updateAdmissionMapIntegrationDouble, makeDoubleMarkerQueryScratch.
inline bool AdmissionMemory::makeDoubleMarkerKeyScratch(StrSpan disjunct, ScratchArena& arena,
            ScratchString& out) {
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t n = getArgsSpans(disjunct, args, ExecutionParameters::MAX_ARITY);
            int32_t open[2] = { -1, -1 };
            int32_t openN = 0;
            for (int32_t i = 0; i < n; ++i) {
                if (startsWithSpan(args[i], "u_", 2)) continue;
                if (openN == 2) return false;
                open[openN++] = i;
            }
            if (openN != 2 || equalSpans(args[open[0]], args[open[1]])) return false;
            // Each open name occurs once among the arguments, so the token
            // replacement rewrites exactly the two open positions.
            const StrReplacement pairs[2] = {
                { args[open[0]], StrSpan("marker1", 7) },
                { args[open[1]], StrSpan("marker2", 7) } };
            out = replaceKeysScratch(arena, disjunct, pairs, 2);
            return true;
        }

/// @brief Build one double-marker probe template from a single-marker
        ///        constituent (D-492).
        ///
        /// @details
        /// The constituent carries `marker` at @p markerAt (the witness being
        /// admitted) and a value at @p otherAt that stands for the second open
        /// argument. Of the two positions the earlier becomes `marker1` and the
        /// later `marker2`; every other argument is prefixed with `u_`, the
        /// spelling the stored keys carry. The text outside the argument list
        /// (sign, predicate, brackets) is copied unchanged.
        ///
        /// @param markedExpr Bare single-marker constituent (no `u_` arguments).
        /// @param markerAt   Position of `marker`.
        /// @param otherAt    Position standing for the second open argument;
        ///                   differs from @p markerAt.
        /// @param arena      String-tier scratch arena receiving the template.
        /// @return The probe template on @p arena.
        /// @invariant No argument of @p markedExpr carries `u_` (asserted);
        ///            both positions are inside the argument list (asserted).
        /// @see isAdmittedIntegrationDouble, makeDoubleMarkerKeyScratch.
inline ScratchString AdmissionMemory::makeDoubleMarkerQueryScratch(StrSpan markedExpr,
            int32_t markerAt, int32_t otherAt, ScratchArena& arena) {
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t n = getArgsSpans(markedExpr, args, ExecutionParameters::MAX_ARITY);
            assert(n > 0 && markerAt >= 0 && markerAt < n && otherAt >= 0 && otherAt < n
                && markerAt != otherAt && "makeDoubleMarkerQueryScratch: positions outside the arguments");
            const int32_t first = markerAt < otherAt ? markerAt : otherAt;
            const int32_t second = markerAt < otherAt ? otherAt : markerAt;
            const int32_t head = static_cast<int32_t>(args[0].ptr - markedExpr.ptr);
            const int32_t tail = static_cast<int32_t>(args[n - 1].ptr - markedExpr.ptr) + args[n - 1].len;
            assert(head > 0 && markedExpr.ptr[head - 1] == '[' && tail < markedExpr.len
                && markedExpr.ptr[tail] == ']' && "makeDoubleMarkerQueryScratch: not a flat argument list");
            int32_t len = head + (markedExpr.len - tail) + (n - 1);
            for (int32_t i = 0; i < n; ++i) {
                assert(!startsWithSpan(args[i], "u_", 2)
                    && "makeDoubleMarkerQueryScratch: a bare constituent carries a u_ argument");
                len += (i == first || i == second) ? 7 : 2 + args[i].len;
            }
            char* buffer = arena.allocBytes(len);
            int32_t at = 0;
            std::memcpy(buffer, markedExpr.ptr, static_cast<std::size_t>(head));
            at += head;
            for (int32_t i = 0; i < n; ++i) {
                if (i > 0) buffer[at++] = ',';
                if (i == first || i == second) {
                    std::memcpy(buffer + at, i == first ? "marker1" : "marker2", 7);
                    at += 7;
                    continue;
                }
                buffer[at++] = 'u';
                buffer[at++] = '_';
                std::memcpy(buffer + at, args[i].ptr, static_cast<std::size_t>(args[i].len));
                at += args[i].len;
            }
            std::memcpy(buffer + at, markedExpr.ptr + tail, static_cast<std::size_t>(markedExpr.len - tail));
            at += markedExpr.len - tail;
            assert(at == len && "makeDoubleMarkerQueryScratch: length pass and fill pass disagree");
            return ScratchString::wrap(arena, buffer, len);
        }

/// @brief Whether a live double-marker key with the predicate of
        ///        @p expression is visible at @p scope
        ///        (D-492).
        ///
        /// @details
        /// Visible means filed at @p scope or at a strict ancestor of it. The
        /// predicate is the text of @p expression up to its argument list
        /// (sign included); stored keys compare without their arguments. This
        /// is the eligibility of a constituent for the double-marker check and
        /// the eligibility of an algebra-input source for the relaxed I-268
        /// gate.
        ///
        /// @param mb         Owning block (read only).
        /// @param expression Constituent or disjunct text.
        /// @param scope      Registered scope of the probe.
        /// @return True iff such a key exists.
        /// @invariant Pure: mints nothing.
        /// @see isAdmittedIntegrationDouble.
inline bool AdmissionMemory::doubleMarkerPredicateVisible(const Memory& mb, StrSpan expression,
            NameId scope) {
            const TypedColdMap<int64_t, int64_t>& keys = mb.overallHashMemory.admissionMapIntegrationDouble;
            if (keys.empty() || scope == 0) return false;
            int32_t predicate = 0;
            while (predicate < expression.len && expression.ptr[predicate] != '[') ++predicate;
            assert(predicate < expression.len && "doubleMarkerPredicateVisible: an expression without arguments");
            for (int32_t id = 1; id <= keys.count(); ++id) {
                const StatementKey key = Codec<StatementKey>::decode(keys.keyAt(id));
                if (key.validity != scope && !mb.nameMap.isStrictAncestor(key.validity, scope)) continue;
                const StrSpan text = mb.templateInterner.decodeView(key.orig);
                if (text.len > predicate && text.ptr[predicate] == '['
                    && std::memcmp(text.ptr, expression.ptr, static_cast<std::size_t>(predicate)) == 0)
                    return true;
            }
            return false;
        }

/// @brief Whether an integration key filed in a block carries the wildcard
        ///        `pi_` (D-586).
        ///
        /// @details
        /// The two-marker start keys a part that holds both open arguments of
        /// its goal once per path: the path's argument becomes `marker`, and
        /// the other open argument is spelled `pi_`, which matches any value.
        /// Every key filing mints the key's bare non-marker arguments into
        /// `HashMemory::varsInAdmissionMapIntegrationKeys`, so a wildcard key
        /// was filed in the block iff that set holds the token. The wildcard
        /// variants of a query are probed only then, so blocks without such a
        /// key pay nothing.
        ///
        /// @param mb Owning block (read only).
        /// @return True iff an integration key filed in @p mb carries `pi_`.
        /// @invariant Pure: mints nothing.
        /// @see makeWildcardQueryScratch,
        ///      IntegrationProcessor::prepareIntegrationCore2.
inline bool AdmissionMemory::integrationWildcardFiled(const Memory& mb) {
            const NameId token = mb.templateInterner.lookup(StrSpan("pi_", 3));
            return token != 0 && mb.overallHashMemory.varsInAdmissionMapIntegrationKeys.contains(token);
        }

/// @brief Build the wildcard variant of a marked integration query at one
        ///        argument position (D-586).
        ///
        /// @details
        /// The argument at @p position becomes `pi_`, the spelling of a
        /// wildcard key's other open argument; every other byte is copied
        /// unchanged. A query matches a wildcard key exactly when one of its
        /// variants equals the key, so the callers probe the variant of each
        /// non-marker position in turn, after the query itself.
        ///
        /// @param query    Marked query, `marker` at the admitted value and `u_`
        ///                 on every other argument.
        /// @param position Argument position to replace; a non-marker argument
        ///                 (asserted).
        /// @param arena    String-tier scratch arena receiving the variant.
        /// @return The variant on @p arena.
        /// @invariant Pure: mints nothing; allocates only on @p arena.
        /// @see integrationWildcardFiled, integrationDemandVisible,
        ///      matchingVisibleIntegrationDemands, updateAdmissionMapIntegration.
inline ScratchString AdmissionMemory::makeWildcardQueryScratch(StrSpan query, int32_t position,
            ScratchArena& arena) {
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t n = getArgsSpans(query, args, ExecutionParameters::MAX_ARITY);
            assert(position >= 0 && position < n && !equalSpans(args[position], StrSpan("marker", 6))
                && "makeWildcardQueryScratch: the position is a non-marker argument");
            const int32_t before = static_cast<int32_t>(args[position].ptr - query.ptr);
            const int32_t after = before + args[position].len;
            assert(before > 0 && after < query.len
                && "makeWildcardQueryScratch: the argument lies inside the query");
            const int32_t len = before + 3 + (query.len - after);
            char* buffer = arena.allocBytes(len);
            std::memcpy(buffer, query.ptr, static_cast<std::size_t>(before));
            std::memcpy(buffer + before, "pi_", 3);
            std::memcpy(buffer + before + 3, query.ptr + after, static_cast<std::size_t>(query.len - after));
            return ScratchString::wrap(arena, buffer, len);
        }

/// @brief Route-5 admission probe: whether a single-marker constituent
        ///        matches a double-marker key at its scope or a strict ancestor
        ///        (D-492).
        ///
        /// @details
        /// The constituent must carry exactly one `marker` (the witness has
        /// one open argument) and a predicate with visible double-marker keys.
        /// Every other argument position in turn stands for the key's second
        /// open argument: the probe template
        /// (`makeDoubleMarkerQueryScratch`) is matched class-aware against
        /// `admissionMapIntegrationDouble` at @p validityName and at each strict
        /// ancestor, compared at @p validityName. A hit admits an ordinary
        /// `int_` witness; the map records nothing (the key's value carries
        /// no information, in particular no consumed witnesses).
        ///
        /// The block's induction demands (`Memory::inductionDoubleKey` and
        /// `Memory::inductionProductKey`, the strong step's keys) are gated:
        /// they count only when @p inductionAllowed, otherwise the probe looks
        /// for another matching key (D-502,
        /// D-566). Goal-owned keys count either
        /// way. Inside an induction scope that either strong chain names,
        /// neither strong key counts: one strong catch per path
        /// (D-519,
        /// D-577).
        ///
        /// @param mb               Owning block.
        /// @param markedExpr       Bare single-marker constituent.
        /// @param validityName     Scope of the admission (interned or not).
        /// @param inductionAllowed Whether the induction demand may admit:
        ///                         false when the original expression being
        ///                         disintegrated (or the parked source of a
        ///                         replay) has an argument that is not a
        ///                         number or is a refused lemma digit
        ///                         (`hasOnlyInductionDigits` over
        ///                         `Memory::inductionParameterDigits`).
        /// @param hit              Receives the accepted key and the two
        ///                         argument positions that stood for its
        ///                         markers; untouched on a miss.
        /// @return True iff some combination matches a visible key that may
        ///         admit.
        /// @invariant Adds no admission record; may extend the derived
        ///            class-aware template index only.
        /// @see makeDoubleMarkerQueryScratch, doubleMarkerPredicateVisible,
        ///      DisintegrationProcessor::disintegrateExpr2,
        ///      CanonicalTemplateMatcher::canonicalTemplatePresentIf.
inline bool AdmissionMemory::isAdmittedIntegrationDouble(Memory& mb, StrSpan markedExpr,
            StrSpan validityName, bool inductionAllowed, DoubleMarkerHit* hit) {
            if (mb.overallHashMemory.admissionMapIntegrationDouble.empty()) return false;
            const NameId target = mb.nameMap.lookup(validityName);
            if (target == 0) return false;
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t n = getArgsSpans(markedExpr, args, ExecutionParameters::MAX_ARITY);
            int32_t markerAt = -1;
            int32_t markers = 0;
            for (int32_t i = 0; i < n; ++i) {
                if (!equalSpans(args[i], StrSpan("marker", 6))) continue;
                markerAt = i;
                ++markers;
            }
            if (markers != 1 || n < 2) return false;
            if (!doubleMarkerPredicateVisible(mb, markedExpr, target)) return false;

            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
            ScratchArena& strings = scratchArenas().forSlot(slot);
            ScratchScope frame(strings);
            // Owned copies: the class-aware match mints templates, never the
            // caller's arena, but the copies keep this probe self-contained.
            const ScratchString expression = ScratchString::copyFrom(strings, markedExpr.ptr, markedExpr.len);
            const ScratchString scope = ScratchString::copyFrom(strings, validityName.ptr, validityName.len);
            StrSpan ancestors[ExecutionParameters::MAX_SCOPE_DEPTH];
            const int32_t ancestorN = gl::ScopeSyntax::copyStrictAncestorSpans(mb, StrSpan(scope), ancestors, strings);
            // A packed key is never 0, so 0 rejects nothing. One strong catch
            // per path: inside an induction scope that either demand (sum or
            // product, D-566) opened, both demands
            // are refused like a dirty original, so a catch never repeats the
            // case analysis of another catch's tree with fresh witnesses
            // (D-577). And one order per path:
            // where the hypothesis already ate another slot, both demands are
            // refused there too (D-519), and so
            // they are in a settled tree (D-520).
            const bool strongRefused = !inductionAllowed
                || ((mb.inductionDoubleKey != 0 || mb.inductionProductKey != 0)
                    && (!mb.inductionSlotOpen(target, Memory::kStrongInductionSlot)
                        || mb.inductionTreeSettled(target)));
            const bool belowInduction = gl::ScopeSyntax::atOrBelowInductionScope(mb.nameMap, target);
            const auto chainOnPath = [&](int64_t key, NameId chainId) -> bool {
                if (key == 0) return false;
                assert(chainId != 0
                    && "isAdmittedIntegrationDouble: an induction demand inside the tree has its chain");
                const StrSpan chainView = mb.nameMap.decodeView(chainId);
                const StrSpan chain(ScratchString::copyFrom(strings, chainView.ptr, chainView.len));
                return gl::ScopeSyntax::inductionChainOnPath(mb.nameMap, target, &chain, 1);
            };
            const bool strongCatchOnPath = !strongRefused && belowInduction
                && (chainOnPath(mb.inductionDoubleKey, mb.inductionDoubleChainId)
                    || chainOnPath(mb.inductionProductKey, mb.inductionProductChainId));
            const auto refusedKey = [&](int64_t key) -> int64_t {
                if (key == 0) return 0;
                return strongRefused || strongCatchOnPath ? key : 0;
            };
            const int64_t rejectedSum = refusedKey(mb.inductionDoubleKey);
            const int64_t rejectedProduct = refusedKey(mb.inductionProductKey);
            int64_t accepted = 0;
            const auto accept = [rejectedSum, rejectedProduct, &accepted](int64_t key) {
                if (key == rejectedSum || key == rejectedProduct) return false;
                accepted = key;
                return true;
            };
            const auto recordHit = [&](int32_t other) {
                if (hit == nullptr) return;
                assert(accepted != 0 && "isAdmittedIntegrationDouble: a hit names its key");
                hit->key = accepted;
                hit->markerAt = markerAt;
                hit->otherAt = other;
            };
            for (int32_t other = 0; other < n; ++other) {
                if (other == markerAt) continue;
                ScratchScope comboFrame(strings);
                const ScratchString query = makeDoubleMarkerQueryScratch(StrSpan(expression), markerAt, other, strings);
                if (canonicalTemplates.canonicalTemplatePresentIf(mb, mb.overallHashMemory.admissionMapIntegrationDouble,
                        mb.lbMemory.canonicalUses.integrationDoubleMatches, StrSpan(query), StrSpan(scope), target, accept)) {
                    recordHit(other);
                    return true;
                }
                for (int32_t a = 0; a < ancestorN; ++a)
                    if (canonicalTemplates.canonicalTemplatePresentIf(mb, mb.overallHashMemory.admissionMapIntegrationDouble,
                            mb.lbMemory.canonicalUses.integrationDoubleMatches, StrSpan(query), ancestors[a], target, accept)) {
                        recordHit(other);
                        return true;
                    }
            }
            return false;
        }

/// @brief File the double-marker demand of one OR disjunct of a flattened
        ///        root integration goal (D-492).
        ///
        /// @details
        /// A disjunct of the wrong shape (`makeDoubleMarkerKeyScratch` false),
        /// one whose predicate may not key an integration instruction
        /// (`isIntegrationInstructionKeyEligible`) or is not configured files
        /// nothing — the same eligibility the single-marker keys of
        /// `prepareIntegrationCore2` Case C use. A new key wakes the parked
        /// integration sources at its scope and below that it now admits; a
        /// key already filed changes nothing. The key's value carries no
        /// information (always 0).
        ///
        /// An induction demand (a bound premise of the strong step block's
        /// hypotheses) also records its new key: the sum demand as
        /// `Memory::inductionDoubleKey`, the product demand as
        /// `Memory::inductionProductKey`; route 5 through either serves only an
        /// original of lemma input values, sets and operators (D-502,
        /// D-566). The block files each once, so
        /// the key is new and none was recorded before (asserted).
        ///
        /// @param mb              Owning block (single-threaded seam).
        /// @param disjunct        The disjunct in `u_`-prefixed instruction form.
        /// @param validityName    The root goal's scope.
        /// @param inductionDemand Which induction demand of the strong step
        ///                        block the key is, or `none`.
        /// @return True iff a new key was filed.
        /// @see makeDoubleMarkerKeyScratch,
        ///      RejectedSourceReplay::revisitRejectedIntegrationDouble,
        ///      IntegrationProcessor::installRootIntegrationFamily,
        ///      isAdmittedIntegrationDouble.
inline bool AdmissionMemory::updateAdmissionMapIntegrationDouble(Memory& mb, StrSpan disjunct,
            StrSpan validityName, InductionDemand inductionDemand) {
            if (!admissionPolicy.isIntegrationInstructionKeyEligible(disjunct)) return false;
            const StrSpan core = extractExpressionUniversalSpan(disjunct);
            if (this->definitions.configuredNames().count(
                    std::string_view(core.ptr, static_cast<std::size_t>(core.len))) == 0) return false;
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
            ScratchArena& strings = scratchArenas().forSlot(slot);
            ScratchScope frame(strings);
            ScratchString keyText;
            if (!makeDoubleMarkerKeyScratch(disjunct, strings, keyText)) return false;
            const ScratchString scope = ScratchString::copyFrom(strings, validityName.ptr, validityName.len);
            const int64_t key = mintTemplateKey(mb.templateInterner, mb.nameMap, StrSpan(keyText), StrSpan(scope));
            TypedColdMap<int64_t, int64_t>& keys = mb.overallHashMemory.admissionMapIntegrationDouble;
            if (keys.find(key) != nullptr) {
                assert(inductionDemand == InductionDemand::none
                    && "updateAdmissionMapIntegrationDouble: an induction demand files a new key");
                return false;
            }
            keys.insert(key, int64_t{0});
            // Recorded before the wake, so the revived sources meet the gate.
            if (inductionDemand == InductionDemand::sum) {
                assert(mb.inductionDoubleKey == 0
                    && "updateAdmissionMapIntegrationDouble: a block files one sum demand");
                mb.inductionDoubleKey = key;
            } else if (inductionDemand == InductionDemand::product) {
                assert(mb.inductionProductKey == 0
                    && "updateAdmissionMapIntegrationDouble: a block files one product demand");
                mb.inductionProductKey = key;
            }
            rejectedSourceReplay.revisitRejectedIntegrationDouble(mb, StrSpan(keyText), StrSpan(scope));
            return true;
        }

/// @brief Whether one element of an and gates an integration witness's
        ///        admission (D-481).
        ///
        /// @details
        /// The integration twin of the implication-subproof gate element
        /// (`ImplicationSubproofs::isSubproofGateElement`): a non-implication
        /// element — atomic, negated or compound — whose every argument is
        /// fixed (`u_`-prefixed) or is @p variable, the argument the candidate
        /// witness fills: the `marker` on the witness's own route, the bound
        /// variable of the existence it fills on a coadmission route. Once the
        /// witness stands in for that argument the element is a ground
        /// statement that can become known. An implication element — a
        /// numbered `(implication<N>[…])` or a configured definition whose
        /// whole body is one implication (D-467) — is integrated through its
        /// own subproof and never gates; an element with any other argument
        /// (another bound variable, a `pi_` pattern, the marker on a
        /// coadmission route) is not a statement about the witness and never
        /// gates.
        ///
        /// @param element     One element of an and entity of a marked
        ///                    integration instruction (flat MPL, at least one
        ///                    argument).
        /// @param variable    The argument the candidate witness fills.
        /// @param definitions The compiled registry naming the configured
        ///                    implication definitions.
        /// @return True iff @p element gates the witness's admission.
        /// @invariant Pure; allocates nothing.
        /// @see integrationTrialGates, ImplicationSubproofs::isSubproofGateElement,
        ///      CompiledDefinitionRegistry::isConfiguredImplication.
inline bool AdmissionMemory::isIntegrationTrialGateElement(StrSpan element, StrSpan variable,
            const CompiledDefinitionRegistry& definitions) {
            assert(variable.len > 0 && "isIntegrationTrialGateElement: the witness fills an argument");
            if (startsWithSpan(element, "(implication", 12)
                || definitions.isConfiguredImplication(element))
                return false;
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t argN = getArgsSpans(element, args, ExecutionParameters::MAX_ARITY);
            assert(argN > 0 && "isIntegrationTrialGateElement: an and element carries arguments");
            for (int32_t a = 0; a < argN; ++a)
                if (!startsWithSpan(args[a], "u_", 2) && !equalSpans(args[a], variable))
                    return false;
            return true;
        }

/// @brief Collect the gates of every new name one integration admission
        ///        creates (D-481,
        ///        D-482).
        ///
        /// @details
        /// Reads every instruction stored under the `admissionMapIntegration`
        /// keys that canonically match @p markedExpr at @p validityName (owned there or at a strict ancestor) — the
        /// keys the route-3 probe of `isAdmittedIntegration` finds — and takes
        /// in each the existence entities that list the matched key, directly
        /// or through an and on one of their sides.
        ///
        /// The matched name @p matched stands for the marker. Its gates are the
        /// gate elements (`isIntegrationTrialGateElement` for the `marker`) of
        /// the ands on the sides of those existences, nested ands included. The
        /// matched key is the admission premise, never a gate: an element equal
        /// to it is skipped and an and entity equal to it is not walked.
        ///
        /// Every other new name of the admission (@p others — the names on the
        /// admitted other sides, enclosing or nested) takes on the gates of its
        /// place in the instruction. A text fills a variable when its one
        /// argument that is neither fixed nor the marker is that variable and
        /// the text, with the marker read as @p matched and the variable as the
        /// name, is one of @p facts. Beside: an existence on a side of the
        /// key's existence, directly or inside an and below a side, one of
        /// whose sides fills its bound variable; the gates are the gate
        /// elements for that variable of the existence's other and sides.
        /// Above: the key's existence or an entity enclosing it, one of whose
        /// texts (signature, side, element of an and below a side; the key
        /// excluded) fills a variable; the gates are the gate elements for that
        /// variable of that entity's ands. The filling text and the matched key
        /// are never gates. A name without a place has no gates.
        ///
        /// Gate elements become ground statements by replacing the filled
        /// argument with the name and stripping the `u_` prefixes. One naming a
        /// new name is unknown by construction, and a fixed one counts only
        /// while it is not known at the scope (`elementKnownAtOrAbove`). An
        /// instruction with no unknown gate among all the names is ungated.
        ///
        /// With @p record null the call is a pure probe and mints nothing. With
        /// a record it appends one group per gated instruction — the gates of
        /// all the names encoded into the NameMap, in instruction order; a
        /// group equal in gates to an existing one (the same instruction filed
        /// under another key) is not appended again.
        ///
        /// @param mb           Owning block.
        /// @param markedExpr   Marked constituent of @p matched, bare form (the
        ///                     marked text the route-3 probe receives).
        /// @param validityName Registered scope of the admission.
        /// @param matched      Name of the witness the key matched; it stands
        ///                     for the marker. Materialized when @p record is
        ///                     set.
        /// @param others       The admission's other new names (null when
        ///                     @p otherCount is zero); materialized when
        ///                     @p record is set.
        /// @param otherCount   Number of @p others.
        /// @param facts        The constituents of all the admission's new
        ///                     names, in their spelling (null when @p factCount
        ///                     is zero).
        /// @param factCount    Number of @p facts.
        /// @param record       Optional out: groups and gates appended in place.
        /// @param linked       The route's linked arguments: an instruction
        ///                     the witness-link gate refuses is skipped
        ///                     (D-529). Null when
        ///                     @p linkedCount is 0.
        /// @param linkedCount  Number of @p linked.
        /// @return True iff at least one instruction matched and every matched
        ///         instruction is gated.
        /// @invariant Group and gate capacities are asserted, never truncated.
        /// @see isAdmittedIntegration, integrationEnclosingFill,
        ///      DisintegrationProcessor::disintegrateExpr2,
        ///      EquivalenceQueries::elementKnownAtOrAbove.
inline bool AdmissionMemory::integrationTrialGates(Memory& mb,
            StrSpan markedExpr,
            StrSpan validityName,
            StrSpan matched,
            const StrSpan* others,
            int32_t otherCount,
            const StrSpan* facts,
            int32_t factCount,
            IntegrationTrialRec* record,
            const StrSpan* linked, int32_t linkedCount,
            const WitnessHistoryProbe* history) {
            assert(matched.len > 0 && otherCount >= 0 && factCount >= 0
                && (otherCount == 0 || others != nullptr) && (factCount == 0 || facts != nullptr)
                && "integrationTrialGates: an admission names its matched witness");
            const NameId scopeId = mb.nameMap.lookup(validityName);
            assert(scopeId != 0 && "integrationTrialGates: expected a registered scope");
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
            ScratchArena& arena = genScratchArenas().forSlot(slot);
            ScratchArena& strings = scratchArenas().forSlot(slot);
            ScratchScope frame(arena);
            ScratchScope stringFrame(strings);
            const StrSpan marker("marker", 6);
            const ScratchString renamed =
                gl::DefinitionExpansion::prefixNonMarkerArgumentsWithUScratch(strings, markedExpr);
            DirtyState dirty = DirtyState::Clean;
            PagedVector<int64_t> matches(&arena, &dirty);
            matchingVisibleIntegrationDemands(mb, StrSpan(renamed), validityName, scopeId, matches, arena);
            const auto namesArgument = [&](StrSpan element, StrSpan argument) {
                StrSpan args[ExecutionParameters::MAX_ARITY];
                const int32_t argN = getArgsSpans(element, args, ExecutionParameters::MAX_ARITY);
                for (int32_t a = 0; a < argN; ++a)
                    if (equalSpans(args[a], argument)) return true;
                return false;
            };
            // The one argument of a text that is neither fixed nor the marker,
            // or empty.
            const auto openVariable = [&](StrSpan text) {
                StrSpan args[ExecutionParameters::MAX_ARITY];
                const int32_t argN = getArgsSpans(text, args, ExecutionParameters::MAX_ARITY);
                StrSpan variable;
                for (int32_t a = 0; a < argN; ++a) {
                    if (startsWithSpan(args[a], "u_", 2) || equalSpans(args[a], marker)) continue;
                    if (variable.len > 0 && !equalSpans(variable, args[a])) return StrSpan();
                    variable = args[a];
                }
                return variable;
            };
            // Whether the text fills its open variable with `name`: with the
            // marker read as the matched witness it is one of the facts.
            const auto fills = [&](StrSpan text, StrSpan name) {
                const StrSpan variable = openVariable(text);
                if (variable.len == 0) return false;
                ScratchScope probeFrame(strings);
                const StrReplacement pairs[2] = { { marker, matched }, { variable, name } };
                const ScratchString named = replaceKeysScratch(strings, text, pairs, 2);
                const ScratchString probe = gl::DefinitionExpansion::removeUPrefixScratch(strings, StrSpan(named));
                for (int32_t x = 0; x < factCount; ++x)
                    if (equalSpans(facts[x], StrSpan(probe))) return true;
                return false;
            };
            bool matchedAny = false;
            for (int32_t m = 0; m < matches.size(); ++m) {
                const int64_t pk = matches[m];
                const StrSpan keyView = mb.templateInterner.decodeView(
                    static_cast<int32_t>(static_cast<uint64_t>(pk) >> 32));
                const ScratchString key = ScratchString::copyFrom(strings, keyView.ptr, keyView.len);
                ArenaIntegrationMap entries(&arena, &mb.valueInterner);
                entries.buildFromCold(mb.overallHashMemory.admissionMapIntegration, pk);
                int32_t order[ExecutionParameters::MAX_INTEGRATION_ENTRIES];
                const int32_t instructionN = entries.sortedIndices(order,
                    ExecutionParameters::MAX_INTEGRATION_ENTRIES);
                for (int32_t k = 0; k < instructionN; ++k) {
                    WorkInstruction wi(&arena);
                    entries.loadInstructionInto(order[k], wi, mb.valueInterner);
                    // An instruction the witness-link gate refuses admits
                    // nothing (D-529), nor does a copy bound to one premise
                    // that refuses the arriving expression
                    // (D-579,
                    // D-578).
                    if (history != nullptr && premiseCopyRefuses(mb, scopeId, wi.forbidden,
                            history->start)) continue;
                    if (witnessLinkRefuses(mb, wi.pathVariables(), linked, linkedCount, scopeId, history)) continue;
                    matchedAny = true;
                    const int32_t entityN = wi.entityCount();
                    assert(entityN <= ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
                        && "integrationTrialGates: entity count exceeds the walk table");
                    // The entity of the category an element names, or -1.
                    const auto entityOf = [&](StrSpan category, StrSpan text) {
                        for (int32_t p = 0; p < entityN; ++p)
                            if (equalSpans(wi.category(p), category) && equalSpans(wi.signature(p), text))
                                return p;
                        return int32_t{-1};
                    };
                    const auto andEntity = [&](StrSpan text) { return entityOf(StrSpan("and", 3), text); };
                    // Whether entity e names the text, directly or through the
                    // ands below its sides.
                    const auto names = [&](int32_t e, StrSpan text, int32_t depth, const auto& self) -> bool {
                        assert(depth <= entityN && "integrationTrialGates: cyclic instruction");
                        for (int32_t j = 0; j < wi.elemCount(e); ++j) {
                            const StrSpan element = wi.elemAt(e, j);
                            if (equalSpans(element, text)) return true;
                            const int32_t child = andEntity(element);
                            if (child >= 0 && self(child, text, depth + 1, self)) return true;
                        }
                        return false;
                    };
                    NameId groupGates[IntegrationTrialRec::kMaxGates];
                    int32_t groupN = 0;
                    bool gated = false;
                    const auto addGate = [&](StrSpan clean) {
                        gated = true;
                        if (record == nullptr) return;
                        const NameId gate = mb.nameMap.encode(clean);
                        for (int32_t i = 0; i < groupN; ++i)
                            if (groupGates[i] == gate) return;
                        assert(groupN < IntegrationTrialRec::kMaxGates
                            && "integrationTrialGates: gate elements exceed IntegrationTrialRec::kMaxGates");
                        groupGates[groupN++] = gate;
                    };
                    // The gate elements for `variable`, read as `name`, of and
                    // entity a and of the ands below it. The premise and the key
                    // are never gates, and ands equal to them are not walked.
                    const auto collectGates = [&](int32_t a, int32_t depth, StrSpan premise, StrSpan variable,
                                                  StrSpan name, bool* walked, const auto& self) -> void {
                        assert(depth <= entityN && "integrationTrialGates: cyclic instruction");
                        walked[a] = true;
                        for (int32_t j = 0; j < wi.elemCount(a); ++j) {
                            const StrSpan element = wi.elemAt(a, j);
                            if (equalSpans(element, premise) || equalSpans(element, StrSpan(key))) continue;
                            const int32_t child = andEntity(element);
                            if (child >= 0) {
                                if (!walked[child]) self(child, depth + 1, premise, variable, name, walked, self);
                                continue;
                            }
                            if (!isIntegrationTrialGateElement(element, variable, this->definitions)) continue;
                            ScratchScope gateFrame(strings);
                            if (namesArgument(element, variable)) {
                                const StrReplacement filled{ variable, name };
                                const ScratchString bound = replaceKeysScratch(strings, element, &filled, 1);
                                addGate(StrSpan(gl::DefinitionExpansion::removeUPrefixScratch(strings, StrSpan(bound))));
                            } else {
                                const ScratchString clean = gl::DefinitionExpansion::removeUPrefixScratch(strings, element);
                                if (!equivalenceQueries.elementKnownAtOrAbove(StrSpan(clean), scopeId, mb))
                                    addGate(StrSpan(clean));
                            }
                        }
                    };
                    // The gates for `variable`, read as `name`, of entity p's
                    // ands: an and entity itself, or the ands on an existence's
                    // sides; the premise and the key are skipped.
                    const auto entityGates = [&](int32_t p, StrSpan premise, StrSpan variable, StrSpan name) {
                        bool walked[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS] = {};
                        if (equalSpans(wi.category(p), StrSpan("and", 3))) {
                            collectGates(p, 0, premise, variable, name, walked, collectGates);
                            return;
                        }
                        for (int32_t j = 0; j < wi.elemCount(p); ++j) {
                            const StrSpan side = wi.elemAt(p, j);
                            if (equalSpans(side, premise) || equalSpans(side, StrSpan(key))) continue;
                            const int32_t a = andEntity(side);
                            if (a >= 0 && !walked[a]) collectGates(a, 0, premise, variable, name, walked, collectGates);
                        }
                    };
                    // Visit the existences on the sides of entity e, directly
                    // or inside the ands below its sides (the key not walked).
                    const auto forSideExistences = [&](int32_t e, const auto& visit) {
                        bool walked[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS] = {};
                        const auto walkAnd = [&](int32_t a, int32_t depth, const auto& self) -> void {
                            assert(depth <= entityN && "integrationTrialGates: cyclic instruction");
                            walked[a] = true;
                            for (int32_t j = 0; j < wi.elemCount(a); ++j) {
                                const StrSpan element = wi.elemAt(a, j);
                                const int32_t f = entityOf(StrSpan("existence", 9), element);
                                if (f >= 0) { visit(f); continue; }
                                const int32_t child = andEntity(element);
                                if (child >= 0 && !walked[child]) self(child, depth + 1, self);
                            }
                        };
                        for (int32_t j = 0; j < wi.elemCount(e); ++j) {
                            const StrSpan side = wi.elemAt(e, j);
                            if (equalSpans(side, StrSpan(key))) continue;
                            const int32_t f = entityOf(StrSpan("existence", 9), side);
                            if (f >= 0) { visit(f); continue; }
                            const int32_t a = andEntity(side);
                            if (a >= 0 && !walked[a]) walkAnd(a, 0, walkAnd);
                        }
                    };
                    // Above: whether a text of entity p — its signature, a side,
                    // or an element of an and below a side, the key excluded —
                    // fills a variable with `name`; the gates of p's ands for
                    // that variable are collected.
                    const auto aboveGates = [&](int32_t p, StrSpan name) {
                        const auto visitText = [&](StrSpan text) {
                            if (equalSpans(text, StrSpan(key)) || !fills(text, name)) return;
                            entityGates(p, text, openVariable(text), name);
                        };
                        visitText(wi.signature(p));
                        bool walked[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS] = {};
                        const auto walkAnd = [&](int32_t a, int32_t depth, const auto& self) -> void {
                            assert(depth <= entityN && "integrationTrialGates: cyclic instruction");
                            walked[a] = true;
                            for (int32_t j = 0; j < wi.elemCount(a); ++j) {
                                const StrSpan element = wi.elemAt(a, j);
                                const int32_t child = andEntity(element);
                                if (child >= 0) {
                                    if (!walked[child] && !equalSpans(element, StrSpan(key))) self(child, depth + 1, self);
                                    continue;
                                }
                                visitText(element);
                            }
                        };
                        for (int32_t j = 0; j < wi.elemCount(p); ++j) {
                            const StrSpan side = wi.elemAt(p, j);
                            visitText(side);
                            const int32_t a = andEntity(side);
                            if (a >= 0 && !walked[a] && !equalSpans(side, StrSpan(key))) walkAnd(a, 0, walkAnd);
                        }
                    };
                    for (int32_t e = 0; e < entityN; ++e) {
                        if (!equalSpans(wi.category(e), StrSpan("existence", 9))) continue;
                        if (!names(e, StrSpan(key), 0, names)) continue;
                        // The matched witness: the ands beside the key.
                        entityGates(e, StrSpan(key), marker, matched);
                        for (int32_t o = 0; o < otherCount; ++o) {
                            const StrSpan name = others[o];
                            // Beside: the existences on the sides of e, filled
                            // through one of their sides.
                            forSideExistences(e, [&](int32_t f) {
                                for (int32_t i = 0; i < wi.elemCount(f); ++i) {
                                    const StrSpan filler = wi.elemAt(f, i);
                                    if (fills(filler, name)) entityGates(f, filler, openVariable(filler), name);
                                }
                            });
                            // Above: e and every entity enclosing it, each once.
                            bool seen[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS] = {};
                            int32_t frontier[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
                            int32_t frontierN = 0;
                            frontier[frontierN++] = e;
                            seen[e] = true;
                            while (frontierN > 0) {
                                const int32_t current = frontier[--frontierN];
                                aboveGates(current, name);
                                for (int32_t p = 0; p < entityN; ++p) {
                                    if (seen[p] || !names(p, wi.signature(current), 0, names)) continue;
                                    seen[p] = true;
                                    frontier[frontierN++] = p;
                                }
                            }
                        }
                    }
                    if (!gated) return false;
                    if (record == nullptr) continue;
                    bool duplicate = false;
                    for (int32_t g = 0; g < record->groupCount && !duplicate; ++g) {
                        const int32_t begin = g == 0 ? 0 : record->groupEnd[g - 1];
                        if (record->groupEnd[g] - begin != groupN) continue;
                        duplicate = true;
                        for (int32_t i = 0; i < groupN; ++i)
                            if (record->gates[begin + i] != groupGates[i]) duplicate = false;
                    }
                    if (duplicate) continue;
                    assert(record->groupCount < IntegrationTrialRec::kMaxGroups
                        && "integrationTrialGates: gated instructions exceed IntegrationTrialRec::kMaxGroups");
                    const int32_t begin = record->groupCount == 0 ? 0 : record->groupEnd[record->groupCount - 1];
                    assert(begin + groupN <= IntegrationTrialRec::kMaxGates
                        && "integrationTrialGates: gate elements exceed IntegrationTrialRec::kMaxGates");
                    for (int32_t i = 0; i < groupN; ++i) record->gates[begin + i] = groupGates[i];
                    record->groupEnd[record->groupCount] = begin + groupN;
                    ++record->groupCount;
                }
            }
            return matchedAny;
        }

/// @brief Whether a route-5 admission of a nested witness reaches each of
        ///        its unadmitted enclosing witnesses through the facts that
        ///        link them (D-501).
        ///
        /// @details
        /// A double-marker key carries no instruction, so an enclosing witness
        /// has no place in one (`integrationEnclosingFill` is route 3's
        /// test). It comes along with the nested witness @p descendant when
        /// @p facts link it to that witness: starting from @p descendant, a
        /// fact one of whose arguments is a reached name reaches every
        /// enclosing name among its arguments, until a pass reaches nothing
        /// new. In G3's step block D1's innermost constituent `in3[r,p,b,+]`
        /// reaches `r`, and `t`'s constituent `and30[q,b,t,*,r,a,+,N]` then
        /// reaches `t` and `q`. Only the witness names link: a fixed argument
        /// shared by two facts is never one of the names.
        ///
        /// @param descendant     Name of the nested witness in @p facts.
        /// @param enclosing      Names of the unadmitted enclosing witnesses,
        ///                       from the nested witness's parent outwards.
        /// @param enclosingCount Number of @p enclosing (1 to
        ///                       `MAX_INSTRUCTION_ELEMENTS`, asserted).
        /// @param facts          The constituents of @p descendant and of the
        ///                       enclosing witnesses, in their spelling.
        /// @param factCount      Number of @p facts (positive).
        /// @return True iff every enclosing witness is reached.
        /// @invariant Pure: reads the spans only and allocates nothing.
        /// @see integrationEnclosingFill, isAdmittedIntegrationDouble,
        ///      DisintegrationProcessor::disintegrateExpr2.
inline bool AdmissionMemory::doubleMarkerEnclosingLinked(StrSpan descendant, const StrSpan* enclosing,
            int32_t enclosingCount, const StrSpan* facts, int32_t factCount) {
            assert(descendant.len > 0 && enclosing != nullptr && facts != nullptr && factCount > 0
                && enclosingCount > 0 && enclosingCount <= ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
                && "doubleMarkerEnclosingLinked: a nested admission names its enclosing witnesses and their facts");
            bool reached[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS] = {};
            int32_t reachedCount = 0;
            bool grew = true;
            while (grew && reachedCount < enclosingCount) {
                grew = false;
                for (int32_t f = 0; f < factCount; ++f) {
                    StrSpan args[ExecutionParameters::MAX_ARITY];
                    const int32_t n = getArgsSpans(facts[f], args, ExecutionParameters::MAX_ARITY);
                    bool touches = false;
                    for (int32_t i = 0; i < n && !touches; ++i) {
                        touches = equalSpans(args[i], descendant);
                        for (int32_t e = 0; e < enclosingCount && !touches; ++e)
                            touches = reached[e] && equalSpans(args[i], enclosing[e]);
                    }
                    if (!touches) continue;
                    for (int32_t i = 0; i < n; ++i)
                        for (int32_t e = 0; e < enclosingCount; ++e)
                            if (!reached[e] && equalSpans(args[i], enclosing[e])) {
                                reached[e] = true;
                                ++reachedCount;
                                grew = true;
                            }
                }
            }
            return reachedCount == enclosingCount;
        }

/// @brief Which level of the matched integration instruction an enclosing
        ///        witness fills when a witness nested in it is admitted under
        ///        an integration key (D-482).
        ///
        /// @details
        /// A nested witness @p descendant was matched by a key of
        /// `admissionMapIntegration` (route 3) through the marked constituent
        /// @p markedExpr. Its admission may carry an enclosing witness of the
        /// same source along only when that witness has a place in the matched
        /// instruction. Reads every instruction stored under the keys that
        /// canonically match @p markedExpr at @p validityName (owned there or at a strict ancestor) and takes in each
        /// the existence entities that list the key, directly or through an
        /// and on one of their sides.
        ///
        /// A text of the instruction fills a variable when its one argument
        /// that is neither fixed (`u_`) nor the marker is that variable and the
        /// text, with the marker read as @p descendant and the variable as
        /// @p enclosing and the `u_` prefixes stripped, is one of @p facts.
        ///
        /// Above (`kEnclosingFillAbove`): a text of the key's existence or of an
        /// entity enclosing it — its signature, a side, or an element of an and
        /// below a side, other than the key — fills a variable. Such a variable
        /// is bound before the key's existence (a goal's output, an enclosing
        /// existence's witness), so the enclosing witness leads up to the
        /// nested admission.
        ///
        /// Beside (`kEnclosingFillBeside`): a side of an existence entity that
        /// stands on a side of the key's existence, directly or inside an and
        /// below a side, fills that existence's bound variable. The enclosing witness is then constructed after the key's
        /// witness (the gates it takes on are `integrationTrialGates` on a
        /// coadmission route).
        ///
        /// Above wins over beside in any matched instruction; with no fill the
        /// result is `kEnclosingFillNone`.
        ///
        /// @param mb           Owning block.
        /// @param markedExpr   Marked constituent of @p descendant, bare form
        ///                     (the marked text the route-3 probe receives).
        /// @param validityName Registered scope of the admission.
        /// @param enclosing    Name of the enclosing witness in @p facts.
        /// @param descendant   Name of the nested witness in @p facts; it stands
        ///                     for the marker.
        /// @param facts        The constituents of the witnesses from
        ///                     @p descendant up to @p enclosing, in their
        ///                     spelling.
        /// @param factCount    Number of @p facts (positive).
        /// @param linked       The route's linked arguments: an instruction
        ///                     the witness-link gate refuses is skipped
        ///                     (D-529). Null when
        ///                     @p linkedCount is 0.
        /// @param linkedCount  Number of @p linked.
        /// @return `kEnclosingFillNone`, `kEnclosingFillAbove` or
        ///         `kEnclosingFillBeside`.
        /// @invariant Pure probe: mints nothing and writes no record.
        /// @see integrationTrialGates, isAdmittedIntegration,
        ///      DisintegrationProcessor::disintegrateExpr2, I-252.
inline int32_t AdmissionMemory::integrationEnclosingFill(Memory& mb,
            StrSpan markedExpr,
            StrSpan validityName,
            StrSpan enclosing,
            StrSpan descendant,
            const StrSpan* facts,
            int32_t factCount,
            const StrSpan* linked, int32_t linkedCount,
            const WitnessHistoryProbe* history) {
            assert(enclosing.len > 0 && descendant.len > 0 && facts != nullptr && factCount > 0
                && "integrationEnclosingFill: a nested admission names both witnesses and their facts");
            const NameId scopeId = mb.nameMap.lookup(validityName);
            assert(scopeId != 0 && "integrationEnclosingFill: expected a registered scope");
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
            ScratchArena& arena = genScratchArenas().forSlot(slot);
            ScratchArena& strings = scratchArenas().forSlot(slot);
            ScratchScope frame(arena);
            ScratchScope stringFrame(strings);
            const StrSpan marker("marker", 6);
            const ScratchString renamed =
                gl::DefinitionExpansion::prefixNonMarkerArgumentsWithUScratch(strings, markedExpr);
            DirtyState dirty = DirtyState::Clean;
            PagedVector<int64_t> matches(&arena, &dirty);
            matchingVisibleIntegrationDemands(mb, StrSpan(renamed), validityName, scopeId, matches, arena);
            // The one argument of a text that is neither fixed nor the marker,
            // or empty.
            const auto openVariable = [&](StrSpan text) {
                StrSpan args[ExecutionParameters::MAX_ARITY];
                const int32_t argN = getArgsSpans(text, args, ExecutionParameters::MAX_ARITY);
                StrSpan variable;
                for (int32_t a = 0; a < argN; ++a) {
                    if (startsWithSpan(args[a], "u_", 2) || equalSpans(args[a], marker)) continue;
                    if (variable.len > 0 && !equalSpans(variable, args[a])) return StrSpan();
                    variable = args[a];
                }
                return variable;
            };
            // Whether the text fills its open variable with the enclosing
            // witness: with the marker read as the descendant it is a fact.
            const auto fills = [&](StrSpan text) {
                const StrSpan variable = openVariable(text);
                if (variable.len == 0) return false;
                ScratchScope probeFrame(strings);
                const StrReplacement pairs[2] = { { marker, descendant }, { variable, enclosing } };
                const ScratchString named = replaceKeysScratch(strings, text, pairs, 2);
                const ScratchString probe = gl::DefinitionExpansion::removeUPrefixScratch(strings, StrSpan(named));
                for (int32_t x = 0; x < factCount; ++x)
                    if (equalSpans(facts[x], StrSpan(probe))) return true;
                return false;
            };
            bool beside = false;
            for (int32_t m = 0; m < matches.size(); ++m) {
                const int64_t pk = matches[m];
                const StrSpan keyView = mb.templateInterner.decodeView(
                    static_cast<int32_t>(static_cast<uint64_t>(pk) >> 32));
                const ScratchString key = ScratchString::copyFrom(strings, keyView.ptr, keyView.len);
                ArenaIntegrationMap entries(&arena, &mb.valueInterner);
                entries.buildFromCold(mb.overallHashMemory.admissionMapIntegration, pk);
                int32_t order[ExecutionParameters::MAX_INTEGRATION_ENTRIES];
                const int32_t instructionN = entries.sortedIndices(order,
                    ExecutionParameters::MAX_INTEGRATION_ENTRIES);
                for (int32_t k = 0; k < instructionN; ++k) {
                    WorkInstruction wi(&arena);
                    entries.loadInstructionInto(order[k], wi, mb.valueInterner);
                    // An instruction the witness-link gate refuses carries
                    // nothing along (D-529), nor does a copy bound to one
                    // premise that refuses the arriving expression
                    // (D-579,
                    // D-578).
                    if (history != nullptr && premiseCopyRefuses(mb, scopeId, wi.forbidden,
                            history->start)) continue;
                    if (witnessLinkRefuses(mb, wi.pathVariables(), linked, linkedCount, scopeId, history)) continue;
                    const int32_t entityN = wi.entityCount();
                    assert(entityN <= ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
                        && "integrationEnclosingFill: entity count exceeds the walk table");
                    const auto entityOf = [&](StrSpan category, StrSpan text) {
                        for (int32_t p = 0; p < entityN; ++p)
                            if (equalSpans(wi.category(p), category) && equalSpans(wi.signature(p), text))
                                return p;
                        return int32_t{-1};
                    };
                    const auto andEntity = [&](StrSpan text) { return entityOf(StrSpan("and", 3), text); };
                    // Whether entity e names the text, directly or through the
                    // ands below its sides.
                    const auto names = [&](int32_t e, StrSpan text, int32_t depth, const auto& self) -> bool {
                        assert(depth <= entityN && "integrationEnclosingFill: cyclic instruction");
                        for (int32_t j = 0; j < wi.elemCount(e); ++j) {
                            const StrSpan element = wi.elemAt(e, j);
                            if (equalSpans(element, text)) return true;
                            const int32_t child = andEntity(element);
                            if (child >= 0 && self(child, text, depth + 1, self)) return true;
                        }
                        return false;
                    };
                    // Whether a text of entity e — a side, or an element of an
                    // and below a side, other than the key — fills a variable.
                    const auto elementFills = [&](int32_t e, int32_t depth, const auto& self) -> bool {
                        assert(depth <= entityN && "integrationEnclosingFill: cyclic instruction");
                        for (int32_t j = 0; j < wi.elemCount(e); ++j) {
                            const StrSpan element = wi.elemAt(e, j);
                            if (equalSpans(element, StrSpan(key))) continue;
                            const int32_t child = andEntity(element);
                            if (child >= 0) {
                                if (self(child, depth + 1, self)) return true;
                                continue;
                            }
                            if (fills(element)) return true;
                        }
                        return false;
                    };
                    for (int32_t e = 0; e < entityN; ++e) {
                        if (!equalSpans(wi.category(e), StrSpan("existence", 9))) continue;
                        if (!names(e, StrSpan(key), 0, names)) continue;
                        // Above: the key's existence and every entity enclosing
                        // it, walked upward once each.
                        bool seen[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS] = {};
                        int32_t frontier[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
                        int32_t frontierN = 0;
                        frontier[frontierN++] = e;
                        seen[e] = true;
                        while (frontierN > 0) {
                            const int32_t current = frontier[--frontierN];
                            if (fills(wi.signature(current)) || elementFills(current, 0, elementFills))
                                return kEnclosingFillAbove;
                            for (int32_t p = 0; p < entityN; ++p) {
                                if (seen[p] || !names(p, wi.signature(current), 0, names)) continue;
                                seen[p] = true;
                                frontier[frontierN++] = p;
                            }
                        }
                        // Beside: the existences on the sides of the key's
                        // existence, directly or inside the ands below its sides
                        // (the key not walked), filled through one of their sides.
                        bool walkedAnd[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS] = {};
                        const auto besideIn = [&](int32_t f) {
                            for (int32_t i = 0; i < wi.elemCount(f) && !beside; ++i)
                                beside = fills(wi.elemAt(f, i));
                        };
                        const auto walkAnd = [&](int32_t a, int32_t depth, const auto& self) -> void {
                            assert(depth <= entityN && "integrationEnclosingFill: cyclic instruction");
                            walkedAnd[a] = true;
                            for (int32_t j = 0; j < wi.elemCount(a) && !beside; ++j) {
                                const StrSpan element = wi.elemAt(a, j);
                                const int32_t f = entityOf(StrSpan("existence", 9), element);
                                if (f >= 0) { besideIn(f); continue; }
                                const int32_t child = andEntity(element);
                                if (child >= 0 && !walkedAnd[child]) self(child, depth + 1, self);
                            }
                        };
                        for (int32_t j = 0; j < wi.elemCount(e) && !beside; ++j) {
                            const StrSpan side = wi.elemAt(e, j);
                            if (equalSpans(side, StrSpan(key))) continue;
                            const int32_t f = entityOf(StrSpan("existence", 9), side);
                            if (f >= 0) { besideIn(f); continue; }
                            const int32_t a = andEntity(side);
                            if (a >= 0 && !walkedAnd[a]) walkAnd(a, 0, walkAnd);
                        }
                    }
                }
            }
            return beside ? kEnclosingFillBeside : kEnclosingFillNone;
        }

/// @brief Disintegration-side admission check — gates whether a
        /// disintegration head may fire from the given LB.
        ///
        /// @details
        /// Walks `mb.overallHashMemory.admissionMap` for the
        /// `(expression, validity)` pair, looks up the matching
        /// `AdmissionMapValue`s, and checks each against the current
        /// depth / secondary-iteration budget. Returns true iff at
        /// least one admission slot is alive and all sibling premises
        /// of that slot are present in `mb.intLocalEncodedStatementsSet`.
        ///
        /// Companion to `isAdmittedIntegration` for the disintegration
        /// path. The two share the `admissionStatusMap` cache; revival
        /// from the rejection map (per
        /// [I-22](../../docs/agentic_swdd/30_invariants.md#i-22)) does not clean
        /// the admission entry.
        ///
        /// @param mb           Owning LB.
        /// @param expr         Constituent expression (span source; read-only).
        /// @param var          Witness/replacement variable (span source).
        /// @param markedExpr   Marked template form (span source).
        /// @param validityName Validity scope (span source).
        /// @return True iff the disintegration head fires on this LB.
        /// @see `isAdmittedIntegration` — integration-side counterpart.
        /// @see [I-22](../../docs/agentic_swdd/30_invariants.md#i-22).
        /// @param metrics Temporary-witness budgets, or null for permanent text.
        /// @param commit Apply successful demand updates; false reads scalar
        /// record fields directly and creates no update payload or string copies.
        /// @param inductionAllowed Whether a recursion-owned value (flag set)
        /// may admit: false when the original expression being disintegrated
        /// has an argument that is not a number (`hasOnlyNumberArguments`); an
        /// ordinary value admits either way
        /// (I-285).
inline bool AdmissionMemory::isAdmitted(Memory& mb,
            StrSpan expr,
            StrSpan var,
            StrSpan markedExpr,
            StrSpan validityName,
            const WitnessAdmissionMetrics* metrics,
            bool commit,
            int64_t selectedKey, NameId comparisonScope,
            bool inductionAllowed, InductionScopeSink* inductionScopes) {
            const NameId ownerScope = mb.nameMap.lookup(validityName);
            if (ownerScope == 0) return false;
            if (comparisonScope == 0) comparisonScope = ownerScope;
            assert(ownerScope == comparisonScope || mb.nameMap.isStrictAncestor(ownerScope, comparisonScope));
            assert((commit || inductionScopes == nullptr)
                && "isAdmitted: only a commit opens induction scopes");
            if (selectedKey == 0) {
                const unsigned slot = gl::WorkerContext::coreId() >= 0 ? static_cast<unsigned>(gl::WorkerContext::coreId())
                    : genScratchArenas().slotCount() - 1;
                ScratchArena& arena = genScratchArenas().forSlot(slot);
                ScratchArena& strings = scratchArenas().forSlot(slot);
                ScratchScope frame(arena), textFrame(strings);
                DirtyState dirty = DirtyState::Clean;
                PagedVector<int64_t> matches(&arena, &dirty);
                const ScratchString witness = ScratchString::copyFrom(strings, var.ptr, var.len);
                const ScratchString scope = ScratchString::copyFrom(strings, validityName.ptr, validityName.len);
                const ScratchString canonical = canonicalTemplates.canonicalAdmissionText(mb, expr, comparisonScope, strings, false, StrSpan(witness));
                const ScratchString query = canonicalTemplates.canonicalAdmissionText(mb, markedExpr, comparisonScope, strings);
                if (!commit) {
                    const int32_t text = mb.templateInterner.lookup(StrSpan(query));
                    if (text != 0 && isAdmitted(mb, StrSpan(canonical), StrSpan(witness), StrSpan(query),
                            StrSpan(scope), metrics, false, packInt32Pair(text, ownerScope), comparisonScope,
                            inductionAllowed))
                        return true;
                }
                canonicalTemplates.matchingCanonicalTemplates(mb, mb.overallHashMemory.admissionMap,
                    mb.lbMemory.canonicalUses.algebraMatches, StrSpan(query), ownerScope, comparisonScope, matches, arena);
                bool admitted = false;
                for (int32_t i = 0; i < matches.size(); ++i)
                    admitted |= isAdmitted(mb, StrSpan(canonical), StrSpan(witness), StrSpan(query),
                        StrSpan(scope), metrics, commit, matches[i], comparisonScope, inductionAllowed,
                        inductionScopes);
                return admitted;
            }
            bool result = false;

            // parts = expr.split(var) with assert len == 2 (exactly one occurrence).
            const StrSpan exprS(expr);
            const StrSpan varS(var);
            const int32_t fpos = findSpanFrom(exprS, varS, 0);
            assert(fpos >= 0);
            const int32_t fpos2 = findSpanFrom(exprS, varS, fpos + varS.len);
            assert(fpos2 < 0);   // exactly one occurrence
            const StrSpan part0(exprS.ptr, fpos);
            const StrSpan part1(exprS.ptr + fpos + varS.len,
                                exprS.len - fpos - varS.len);

            // Non-minting probe: a template never interned was never
            // registered, a definitive miss.
            const int64_t markedPk = selectedKey;
            const int32_t admittedId = mb.overallHashMemory.admissionMap.lookup(markedPk);
            if (admittedId == 0) {
                return result;
            }

            // Pure probes only read scalar record fields through a live view.
            // COPY-BEFORE-MUTATE (I-99 canonical run) applies to the committing
            // path: updateAdmissionMap mutates admissionMap (insertAdmissionIdsBlob restructures
            // the shared blob pool), so a peekRecordBytes pointer would dangle
            // past the first call. Copy every blob VERBATIM onto the gen-scratch
            // arena BEFORE the loop and walk the copies. The run at markedPk is
            // stored canonical under DecodedAdmissionValueLess (the RMW contract),
            // so run order IS the former admissionRecordsAt snapshot order (the
            // set dedup is a no-op on a canonical run); admissionRecordsAt /
            // admissionBlobLess stay the retained oracles.
            const unsigned slot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            ScratchArena& gArena = genScratchArenas().forSlot(slot);
            const ArenaOffset gMark = gArena.cursor();
            struct AdmBlob { ArenaOffset off; int32_t len; };
            DirtyState blobsDirty = DirtyState::Clean;
            PagedVector<AdmBlob> blobs(&gArena, &blobsDirty);
            const int32_t runN = mb.overallHashMemory.admissionMap.runLen(admittedId);
            if (commit) {
                for (int32_t j = 0; j < runN; ++j) {
                    int32_t bl = 0;
                    const char* bp = mb.overallHashMemory.admissionMap.peekRecordBytes(
                        admittedId, j, bl, gArena);
                    const ArenaOffset off = gArena.alloc(bl, 1);
                    std::memcpy(gArena.resolve(off), bp, static_cast<std::size_t>(bl));
                    blobs.push_back(AdmBlob{ off, bl });
                }
            }
            // The per-tuple ScratchStrings ride the SAME slot arena
            // updateAdmissionMap uses, so its inner ScratchScope nests LIFO and
            // reclaims only its own allocations (keyVec/remVec/replKey survive it;
            // 09b pitfall 6).
            ScratchArena& sArena = scratchArenas().forSlot(slot);

            for (int32_t i = 0; i < runN; ++i) {
                int32_t blobLength = 0;
                const char* blobBytes;
                if (commit) {
                    blobLength = blobs[i].len;
                    blobBytes = gArena.resolve(blobs[i].off);
                } else {
                    blobBytes = mb.overallHashMemory.admissionMap.peekRecordBytes(
                        admittedId, i, blobLength, gArena);
                }
                const AdmissionValueBlobView v(blobBytes, blobLength);
                // Ordis-only values are cohort-opening demand evidence, never
                // general admission: Pass B is blind to them. The ordis probe
                // is the one reader of tagged values.
                if (v.ordisByte() != 0) {
                    continue;
                }
                const int32_t keyN = v.keyCount();
                assert(keyN <= ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS);
                int32_t remN = v.remCount();
                assert(remN <= ExecutionParameters::MAX_ADMISSION_REM_ARGS);
                const int maxAdmissionDepth = v.depth();
                const int maxSecondaryNumber = v.sec();
                const bool partOfRecursion = (v.flagByte() != 0);
                // A recursion-owned value serves only a disintegration whose
                // original has numbers as its only arguments; it neither
                // admits nor moves its chain otherwise
                // (I-285).
                if (partOfRecursion && !inductionAllowed) {
                    continue;
                }
                // The induction hypothesis's own chain
                // (D-519). It serves a scope only from
                // where the eating stands there — the deepest induction scope
                // on the path, else `main`; a chain filed above that position
                // is a stale state. And one order per path: once the
                // hypothesis ate a slot at the comparison scope, it eats only
                // that slot there; every other slot is eaten inside the scopes
                // the first eats opened. Probe and commit alike. A settled
                // tree admits nothing more through the hypothesis: its first
                // firing ended the hypothesis's branching there
                // (D-520).
                const bool hypothesisChain = v.inductionHypothesis();
                const int32_t eatSlot = Codec<StatementKey>::decode(markedPk).orig;
                if (hypothesisChain
                    && (gl::ScopeSyntax::inductionEatingPosition(mb.nameMap, comparisonScope) != ownerScope
                        || !mb.inductionSlotOpen(comparisonScope, eatSlot)
                        || mb.inductionTreeSettled(comparisonScope))) {
                    continue;
                }

                const int mn = metrics ? metrics->generation : admissionPolicy.extractMaxIterationNumber(varS);
                const int cnt = metrics ? metrics->secondaryCount
                    : admissionPolicy.countPatternOccurrences(exprS, mb.overallHashMemory, mb.nameMap);

                // [user-directed thresh-symmetry fix] isAdmitted scans the CONSTITUENT
                // form, where the it_ witness IS counted, so its cnt is one higher than
                // the marked form the other sites see. The per-slot maxSecondaryNumber
                // (=1) is the wrong cap here; gate on parameters.maxNumberSecondaryVariables
                // (=2, with <=) so a sum of one secondary it_ (p') plus the witness
                // (cnt=2) is admitted. This restores symmetry with the marked-form cnt
                // sites (isAllowedAsOperatorInput, revisitRejected2) which already gate on
                // maxNumberSecondaryVariables. Was the asymmetry blocking l+p' on rung1.
                if (mn <= maxAdmissionDepth && cnt <= parameters.maxNumberSecondaryVariables) {
                    const uint8_t* sit =
                        mb.overallHashMemory.admissionStatusMap.find(markedPk);
                    assert(sit != nullptr);
                    // The induction hypothesis's chain: each element the
                    // constituent fills is one eat, and each eat opens its own
                    // induction scope. Inside the induction tree an eat whose
                    // marked chain already names an induction scope on the
                    // path is a backward readmission and is refused, probe and
                    // commit alike (D-519).
                    const bool pathChecked = hypothesisChain
                        && gl::ScopeSyntax::atOrBelowInductionScope(mb.nameMap, comparisonScope);
                    if (!commit && !pathChecked) {
                        result = true;
                        continue;
                    }
                    ScratchScope tupleScope(sArena);
                    // Owned decoded copies — updateAdmissionMap below mints the value
                    // interner (I-3); the copies ride sArena, not valueInterner.
                    StrSpan keyVec[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                    for (int32_t e = 0; e < keyN; ++e) {
                        const StrSpan dv = mb.valueInterner.decodeView(v.keyId(e));
                        keyVec[e] = StrSpan(ScratchString::copyFrom(sArena, dv.ptr, dv.len));
                    }
                    StrSpan remVec[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
                    for (int32_t r = 0; r < remN; ++r) {
                        const StrSpan dv = mb.valueInterner.decodeView(v.remId(r));
                        remVec[r] = StrSpan(ScratchString::copyFrom(sArena, dv.ptr, dv.len));
                    }
                    bool excludedAnchor = false;
                    remN = canonicalTemplates.canonicalAdmissionChain(mb, keyVec, keyN, remVec, remN,
                        comparisonScope, sArena, keyVec, remVec, excludedAnchor);

                    // The chain with one filled variable written as the
                    // marker — the chain an induction scope names.
                    const auto markChain = [&](StrSpan variable, StrSpan* marked) {
                        const StrReplacement toMarker[1] = { { variable, StrSpan("marker", 6) } };
                        for (int32_t k = 0; k < keyN; ++k)
                            marked[k] = StrSpan(replaceKeysScratch(sArena, keyVec[k], toMarker, 1));
                    };
                    // The eats: the elements the constituent fills, in chain
                    // order, with the variable each one fills.
                    int32_t eatN = 0;
                    int32_t matchedN = 0;
                    StrSpan eatVars[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                    for (int32_t e = 0; e < keyN; ++e) {
                        const StrSpan element = keyVec[e];

                        // startswith/endswith checks with part0/part1 — span
                        // sub-span compares, byte-identical to the former
                        // .compare(off,len,str) calls.
                        bool startsOk = false;
                        if (element.len >= part0.len) {
                            startsOk = compareSpans(StrSpan(element.ptr, part0.len), part0) == 0;
                        }
                        bool endsOk = false;
                        if (element.len >= part1.len) {
                            const int32_t off = element.len - part1.len;
                            endsOk = compareSpans(StrSpan(element.ptr + off, part1.len), part1) == 0;
                        }
                        if (!(startsOk && endsOk)) continue;
                        assert(element.len >= part0.len + part1.len && "admission-key element overlap: part0/part1 must not overlap");
                        const StrSpan varForRepl(element.ptr + part0.len,
                            element.len - part0.len - part1.len);
                        ++matchedN;
                        if (pathChecked) {
                            StrSpan marked[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                            markChain(varForRepl, marked);
                            if (gl::ScopeSyntax::inductionChainOnPath(mb.nameMap, comparisonScope, marked, keyN))
                                continue;
                        }
                        eatVars[eatN++] = varForRepl;
                    }
                    // Every eat of this chain refused: it does not admit.
                    if (hypothesisChain && matchedN != 0 && eatN == 0) continue;
                    result = true;
                    if (!commit) continue;
                    if (*sit != 0) {
                        mb.overallHashMemory.productsOfRecursionIds.mint(mb.nameMap.encode(varS));
                    }
                    // The first eat at the comparison scope fixes its slot.
                    if (hypothesisChain && eatN != 0) {
                        mb.recordInductionEat(comparisonScope, eatSlot);
                    }

                    for (int32_t k = 0; k < eatN; ++k) {
                        const StrSpan varForRepl = eatVars[k];

                        StrReplacement rp[1];
                        rp[0].key = varForRepl;
                        rp[0].value = varS;

                        // replKey: rewrite every keyVec element (fresh
                        // ScratchStrings that survive updateAdmissionMap's
                        // nested scope; allocated above its mark).
                        StrSpan replKey[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                        for (int32_t r = 0; r < keyN; ++r)
                            replKey[r] = StrSpan(replaceKeysScratch(sArena, keyVec[r], rp, 1));

                        // newRemArgs = remVec + var, sorted-unique inserted
                        // (the former std::set insert dedup).
                        StrSpan newRem[ExecutionParameters::MAX_ADMISSION_REM_ARGS + 1];
                        int32_t newRemN = 0;
                        bool inserted = false;
                        for (int32_t r = 0; r < remN; ++r) {
                            if (!inserted) {
                                const int c = compareSpans(remVec[r], varS);
                                if (c == 0) inserted = true;
                                else if (c > 0) { newRem[newRemN++] = varS; inserted = true; }
                            }
                            newRem[newRemN++] = remVec[r];
                        }
                        if (!inserted) newRem[newRemN++] = varS;

                        // An eat of the hypothesis opens its induction scope
                        // below the comparison scope and the advanced chain
                        // lives there; an ordinary chain advances at its
                        // key's own scope.
                        StrSpan chainScope(validityName);
                        ScratchString inductionValidity;
                        if (hypothesisChain) {
                            assert(inductionScopes != nullptr
                                && "isAdmitted: an eat of the induction hypothesis needs a scope sink");
                            StrSpan marked[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                            markChain(varForRepl, marked);
                            const StrSpan names[1] = { varS };
                            const ScratchString payload = gl::ScopeSyntax::inductionScopePayloadScratch(
                                sArena, marked, keyN, names, 1);
                            const NameId inductionScope = mb.nameMap.encodePush(comparisonScope, StrSpan(payload));
                            inductionScopes->add(inductionScope);
                            const StrSpan view = mb.nameMap.decodeView(inductionScope);
                            inductionValidity = ScratchString::copyFrom(sArena, view.ptr, view.len);
                            chainScope = StrSpan(inductionValidity);
                            // After this eat the advanced chain's premises with
                            // only fixed arguments are the candidates for the
                            // hypothesis's missing expressions in the new scope;
                            // the statement door files them once this
                            // disintegration's products are in. Both strong
                            // forms share every premise but their atomic bound
                            // premise, so the block's first hypothesis rule is
                            // the source (D-575).
                            assert(inductionScopes->hypothesisDemands != nullptr
                                && "isAdmitted: an eat of the induction hypothesis needs a demand channel");
                            for (int32_t r = 0; r < keyN; ++r) {
                                if (!fullyBoundElement(replKey[r], newRem, newRemN)
                                    || !admissionPolicy.ordis2KeyEligible(replKey[r])) continue;
                                assert(mb.recursionAuxyImplId != 0
                                    && "isAdmitted: a hypothesis chain lives in a step block");
                                inductionScopes->hypothesisDemands->append(replKey[r], chainScope,
                                    mb.ruleInterner.decodeView(mb.recursionAuxyImplId));
                            }
                        }

                        // The whole flag byte carries over: the advanced copy
                        // of the hypothesis's chain stays the hypothesis's.
                        updateAdmissionMap(mb,
                            replKey, keyN,
                            newRem, newRemN,
                            maxAdmissionDepth,
                            maxSecondaryNumber,
                            v.flagByte(),
                            chainScope);
                    }
                }
            }
            gArena.popTo(gMark);

            return result;
        }

/// @brief Ancestor-inclusive twin of @ref isAdmitted — probe the
        ///        exact deposit validity first, then every strict ancestor
        ///        scope (D-288).
        ///
        /// @details
        /// A demand key at an ancestor scope demands the product for every
        /// descendant, and a fired (instantiated) key lands at
        /// `deeperOf(subkey constituents)` — for the C8 shape, `main`,
        /// while the witness-minting disintegration deposits inside the
        /// released `_ordis_` branch. The exact-validity probe alone can
        /// therefore never connect the two; this twin closes the gap. The
        /// exact probe runs first (unchanged semantics when it hits), then
        /// the strict ancestors in `strictAncestorSpans`' deterministic
        /// `compareSpans` order, first hit winning. Each hit runs the full
        /// @ref isAdmitted interior against the ANCESTOR validity — the
        /// replKey propagation addresses the key that actually admitted. The
        /// ancestor names are `decodeView` spans copied onto the per-slot
        /// string scratch BEFORE any probe (I-3: an admitting probe's
        /// interior mints).
        ///
        /// @param mb           Owning LB.
        /// @param expr         Constituent expression (span source).
        /// @param var          Witness/replacement variable (span source).
        /// @param markedExpr   Marked template form (span source).
        /// @param validityName Deposit validity scope (span source).
        /// @return True iff the disintegration head is admitted at the
        ///         deposit validity or any strict ancestor of it.
        /// @invariant Misses mint nothing (the walk is non-minting probe
        ///            chains); at most one validity's key admits and is
        ///            consumed.
        /// @see isAdmitted — the exact-validity interior;
        ///      `NameMap::strictAncestorSpans` — the ancestor enumeration.
        /// @param metrics Temporary-witness budgets, or null for permanent text.
        /// @param commit Apply successful demand updates; false is a pure probe.
        /// @param inductionAllowed Whether recursion-owned values may admit;
        /// forwarded to every @ref isAdmitted probe
        /// (I-285).
inline bool AdmissionMemory::isAdmittedIncludingAncestors(Memory& mb,
            StrSpan expr,
            StrSpan var,
            StrSpan markedExpr,
            StrSpan validityName,
            const WitnessAdmissionMetrics* metrics,
            bool commit,
            bool inductionAllowed,
            InductionScopeSink* inductionScopes) {
            if (isAdmitted(mb, expr, var, markedExpr, validityName, metrics, commit,
                    0, 0, inductionAllowed, inductionScopes)) return true;

            const unsigned slot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : scratchArenas().slotCount() - 1;
            ScratchArena& sArena = scratchArenas().forSlot(slot);
            ScratchScope sc(sArena);
            StrSpan owned[ExecutionParameters::MAX_SCOPE_DEPTH];
            const int32_t ancN = gl::ScopeSyntax::copyStrictAncestorSpans(mb, validityName,
                owned, sArena);
            if (ancN == 0) return false;
            for (int32_t i = 0; i < ancN; ++i) {
                if (isAdmitted(mb, expr, var, markedExpr, owned[i], metrics, commit, 0,
                        mb.nameMap.lookup(validityName), inductionAllowed, inductionScopes)) return true;
            }
            return false;
        }


/// @brief Fill the integration demands a statement deposited at `main`, at
///        an int scope or in a twin successor case satisfies: each distinct
///        argument in turn becomes the marker, and every instruction under a
///        matching key owned at the deposit scope records the value and files
///        its continuation.
///
/// @details
/// The statement fill of route 3. An instruction copy bound to one premise
/// refuses the fill when the expression whose deposit produced the statement
/// names a forbidden variable or a forced witness of a forbidden level
/// (@p arrivingVariables / @p arrivingLevels, the masks
/// `signatureTokenSets` computes; D-579).
/// At `main` such a copy is a root copy: it also refuses unless every free
/// argument of that expression is a number or `rec`, and an accepted fill
/// opens the copy's int scope (`openIntScope`), where the copy continues. At
/// an int scope the scope's own keys fill in place
/// (D-578). In the weak step block's twin successor
/// case (`Memory::twinCaseScopeId`) only the copies bound to the second twin
/// premise fill, as root copies with `rec2` for `rec`
/// (D-589).
///
/// @param expression          The deposited statement (any lifetime; copied).
/// @param mb                  The owning LB.
/// @param validityName        The deposit scope; only `main`, int scopes and
///                            the twin successor case fill.
/// @param arrivingVariables   Decimal variables the arriving expression's
///                            signature names, as a mask (0 = none).
/// @param arrivingLevels      Forced-witness levels it names, as a mask (0 = none).
/// @param arrivingNumbersOnly Whether every free argument of the arriving
///                            expression is a number or `rec`
///                            (`hasOnlyNumberFreeArguments`).
/// @param arrivingTwinNumbersOnly Whether every free argument of the arriving
///                            expression is a number or `rec2`, the name of
///                            the twin partner's predecessor: the gate of a
///                            copy bound to the second twin premise
///                            (D-589); unread in a block
///                            without one.
/// @return Void.
/// @see isAdmittedIntegration, signatureTokenSets, premiseCopyRefuses.
inline void AdmissionMemory::updateAdmissionMapIntegration(StrSpan expression,
            Memory& mb,
            StrSpan validityName,
            uint64_t arrivingVariables,
            uint64_t arrivingLevels,
            bool arrivingNumbersOnly,
            bool arrivingTwinNumbersOnly) {
            if (!admissionPolicy.isIntegrationInstructionKeyEligible(expression)) return;
            const NameId canonicalScope = mb.nameMap.lookup(validityName);
            assert(canonicalScope != 0);
            // `main` fills, and so does an int scope from its own keys
            // (D-578); a weak step block's twin
            // successor case fills the second twin premise's copies
            // (D-589).
            const bool atIntScope = gl::ScopeSyntax::isIntScope(mb.nameMap, canonicalScope);
            const bool atTwinCase = mb.twinFirstGoalId != 0 && mb.twinCaseScopeId != 0
                && canonicalScope == mb.twinCaseScopeId;
            if (!equalSpans(validityName, StrSpan("main", 4)) && !atIntScope && !atTwinCase) return;

            const unsigned canonicalSlot = gl::WorkerContext::coreId() >= 0 ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : scratchArenas().slotCount() - 1;
            ScratchArena& canonicalStrings = scratchArenas().forSlot(canonicalSlot);
            ScratchScope canonicalFrame(canonicalStrings);
            const ScratchString originalInput = ScratchString::copyFrom(canonicalStrings, expression.ptr, expression.len);
            expression = StrSpan(originalInput);
            // Arg list as zero-copy spans; uniqueArgs as a sorted-unique StrSpan
            // array reproducing the former std::set's ascending iteration order —
            // observable, it drives the per-arg key mint / admission touch
            // (I-138).
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t argN = getArgsSpans(
                expression, args, ExecutionParameters::MAX_ARITY);
            StrSpan uniqueArgs[ExecutionParameters::MAX_ARITY];
            int32_t uniqueN = 0;
            for (int32_t i = 0; i < argN; ++i) {
                int32_t pos = 0;
                while (pos < uniqueN
                    && compareSpans(uniqueArgs[pos], args[i]) < 0) {
                    ++pos;
                }
                if (pos < uniqueN && equalSpans(uniqueArgs[pos], args[i])) {
                    continue;
                }
                for (int32_t j = uniqueN; j > pos; --j) {
                    uniqueArgs[j] = uniqueArgs[j - 1];
                }
                uniqueArgs[pos] = args[i];
                ++uniqueN;
            }

            for (int32_t ui = 0; ui < uniqueN; ++ui) {
                // A witness its admission wrote into an instruction, or a name
                // of its class here, is never tried as the marker again
                // (D-585).
                if (oneWayWitnessAtScope(mb, uniqueArgs[ui], canonicalScope)) continue;
                StrReplacement markerPair[1];
                markerPair[0].key = uniqueArgs[ui];
                markerPair[0].value = StrSpan("marker", 6);

                // Marker-form template key on the per-slot BYTE-scratch arena.
                // replaceKeysScratch is the byte-exact twin of replaceKeysToString
                // and addMissingUScratch of addMissingU, so the bytes reproduce the
                // former heap std::string. The arena is disjoint from the value /
                // template / NameMap interners this loop mints into, so replacedCopy
                // stays valid to its last use (cleanInstructionWork). A tight
                // ScratchScope frees it each iteration.
                const unsigned strSlot = (gl::WorkerContext::coreId() >= 0)
                    ? static_cast<unsigned>(gl::WorkerContext::coreId())
                    : scratchArenas().slotCount() - 1;
                ScratchArena& strArena = scratchArenas().forSlot(strSlot);
                ScratchScope replScope(strArena);
                const ScratchString markedScratch =
                    replaceKeysScratch(strArena, expression, markerPair, 1);
                const ScratchString canonicalMarked = canonicalTemplates.canonicalAdmissionText(mb, StrSpan(markedScratch), canonicalScope, strArena, true);
                const ScratchString replacedCopy = gl::DefinitionExpansion::addMissingUScratch(strArena, StrSpan(canonicalMarked));
                const StrSpan appliedArgument = StrSpan(canonicalTemplates.canonicalAdmissionText(mb, uniqueArgs[ui], canonicalScope, strArena));

                const unsigned matchSlot = gl::WorkerContext::coreId() >= 0 ? static_cast<unsigned>(gl::WorkerContext::coreId())
                    : genScratchArenas().slotCount() - 1;
                ScratchArena& matchArena = genScratchArenas().forSlot(matchSlot);
                DirtyState matchDirty = DirtyState::Clean;
                PagedVector<int64_t> matchingKeys(&matchArena,&matchDirty);
                canonicalTemplates.matchingCanonicalTemplates(mb,mb.overallHashMemory.admissionMapIntegration,
                    mb.lbMemory.canonicalUses.integrationMatches,StrSpan(replacedCopy),canonicalScope,canonicalScope,
                    matchingKeys,matchArena);
                // The wildcard keys follow, one non-marker position at a time
                // (D-586). The
                // statement's own text stays the consumed form: a wildcard
                // part keeps its other open argument.
                if (integrationWildcardFiled(mb)) {
                    StrSpan copyArgs[ExecutionParameters::MAX_ARITY];
                    const int32_t copyArgN = getArgsSpans(StrSpan(replacedCopy), copyArgs, ExecutionParameters::MAX_ARITY);
                    for (int32_t position = 0; position < copyArgN; ++position) {
                        if (equalSpans(copyArgs[position], StrSpan("marker", 6))) continue;
                        ScratchScope variantFrame(strArena);
                        const ScratchString variant = makeWildcardQueryScratch(StrSpan(replacedCopy), position, strArena);
                        DirtyState variantDirty = DirtyState::Clean;
                        PagedVector<int64_t> variantKeys(&matchArena, &variantDirty);
                        canonicalTemplates.matchingCanonicalTemplates(mb, mb.overallHashMemory.admissionMapIntegration,
                            mb.lbMemory.canonicalUses.integrationMatches, StrSpan(variant), canonicalScope, canonicalScope,
                            variantKeys, matchArena);
                        for (int32_t k = 0; k < variantKeys.size(); ++k) matchingKeys.push_back(variantKeys[k]);
                    }
                }
                for (int32_t match=0;match<matchingKeys.size();++match) {
                    const int64_t replacedPk=matchingKeys[match];
                    // Outside the induction scopes of a strong step block a
                    // key that holds nothing connected fills nothing
                    // (D-602).
                    if (!strongStepKeyAdmits(mb, replacedPk, canonicalScope)) continue;
                    // One-pair replacement run ("marker" -> "u_"+arg) on strArena,
                    // held live through the re-entrant Core2 call below (its
                    // StrReplacement map door consumes spans).
                    char* uVarBuf = strArena.allocBytes(2 + appliedArgument.len);
                    uVarBuf[0] = 'u'; uVarBuf[1] = '_';
                    if (appliedArgument.len > 0)
                        std::memcpy(uVarBuf + 2, appliedArgument.ptr,
                                    static_cast<std::size_t>(appliedArgument.len));
                    const StrReplacement changeableMap[1] = {
                        { StrSpan("marker", 6), StrSpan(uVarBuf, 2 + appliedArgument.len) } };

                    // Decode the key's nested instruction map (an owned snapshot
                    // of the cold run).
                    const unsigned aimSlot = (gl::WorkerContext::coreId() >= 0)
                        ? static_cast<unsigned>(gl::WorkerContext::coreId())
                        : genScratchArenas().slotCount() - 1;
                    ArenaIntegrationMap entryMap(
                        &genScratchArenas().forSlot(aimSlot), &mb.valueInterner);
                    entryMap.buildFromCold(
                        mb.overallHashMemory.admissionMapIntegration, replacedPk);

                    // Iterate only the original instructions, while preserving
                    // any same-key additions made by recursive preparation.
                    int32_t snapshot[ExecutionParameters::MAX_INTEGRATION_ENTRIES];
                    const int32_t snapN = entryMap.sortedIndices(
                        snapshot, ExecutionParameters::MAX_INTEGRATION_ENTRIES);

                    for (int32_t si = 0; si < snapN; ++si) {
                        const int32_t snapEi = snapshot[si];
                        WorkInstruction wiSrc(&genScratchArenas().forSlot(aimSlot));
                        entryMap.loadInstructionInto(snapEi, wiSrc, mb.valueInterner);
                        // A copy bound to one premise refuses a fill whose
                        // arriving expression names a forbidden variable or a
                        // forced witness of a forbidden level, and stays
                        // untouched (D-579);
                        // at `main` it is a root copy and also refuses an
                        // arriving expression with an argument that is not a
                        // number (D-578); a copy bound to
                        // the second twin premise counts `rec2` as its `rec`,
                        // and only such a copy fills in the twin successor
                        // case (D-589).
                        const bool secondTwin = wiSrc.forbidden.premiseLevel != 0
                            && wiSrc.forbidden.premiseLevel == mb.twinSecondLevel;
                        if (atTwinCase && !secondTwin) continue;
                        if (premiseCopyRefuses(mb.nameMap, canonicalScope, wiSrc.forbidden,
                                arrivingVariables, arrivingLevels,
                                secondTwin ? arrivingTwinNumbersOnly : arrivingNumbersOnly)) continue;
                        ArenaIntegrationMap current(&genScratchArenas().forSlot(aimSlot), &mb.valueInterner);
                        current.buildFromCold(mb.overallHashMemory.admissionMapIntegration, replacedPk);
                        const int32_t currentCount = current.entries.size();
                        const int32_t currentEntry = current.findOrEmplaceWork(wiSrc, mb.valueInterner);
                        assert(current.entries.size() == currentCount
                            && "integration replay lost its original instruction");
                        if (canonicalTemplates.integrationValueSeenAtScope(mb, current, currentEntry,
                                                       appliedArgument, canonicalScope)) continue;
                        // The first fill of a premise-bound copy at `main`
                        // opens the copy's int scope, where it continues; a
                        // closed int scope never reopens
                        // (D-578).
                        NameId intScope = 0;
                        if (wiSrc.forbidden.premiseLevel != 0 && !atIntScope) {
                            intScope = openIntScope(mb, canonicalScope, wiSrc, appliedArgument);
                            if (intScope == 0) continue;
                        }

                        // Update the current stored map, including same-key entries
                        // registered by earlier recursive preparation in this loop.
                        current.insertValue(currentEntry, mb.valueInterner.encode(appliedArgument));
                        current.writeToCold(mb.overallHashMemory.admissionMapIntegration, replacedPk);
                        mb.lbMemory.canonicalUses.integrationDemands.changed(replacedPk);
                        // Filled through a key connected to the induction: a
                        // size marker (D-583); outside
                        // the step blocks, through a key connected to the
                        // instruction's goal: a connected value
                        // (D-593).
                        if ((mb.weakStepBlock || mb.strongStepBlock)
                                ? admittingKeyConnected(mb, replacedPk)
                                : admittingKeyConnectedToGoal(mb, replacedPk, wiSrc.markedGoal()))
                            recordSizeMarker(mb, appliedArgument);

                        const unsigned wiSlot = aimSlot;
                        WorkInstruction wiCopy(&genScratchArenas().forSlot(wiSlot));
                        cleanInstructionWork(wiSrc, wiCopy, StrSpan(replacedCopy));
                        // The continuation's path list gains the filling value
                        // unless it is an anchor constant
                        // (D-529).
                        if (mb.recursionAuxyImplId != 0) {
                            StrSpan anchors[ExecutionParameters::MAX_ARITY];
                            const int32_t anchorN = gl::ScopeSyntax::anchorArgumentSpans(mb,
                                StrSpan(anchorInfo.name), anchors, ExecutionParameters::MAX_ARITY);
                            bool anchor = false;
                            for (int32_t k = 0; k < anchorN && !anchor; ++k)
                                anchor = equalSpans(anchors[k], appliedArgument);
                            if (!anchor) {
                                ScratchScope listScope(strArena);
                                wiCopy.setPathVariables(StrSpan(extendPathListScratch(strArena,
                                    wiCopy.pathVariables(), appliedArgument)));
                            }
                        }
                        // Owned copy of the int scope's name: the continuation
                        // mints the NameMap (I-3).
                        ScratchString intScopeName;
                        StrSpan continuationScope = validityName;
                        if (intScope != 0) {
                            const StrSpan intScopeView = mb.nameMap.decodeView(intScope);
                            intScopeName = ScratchString::copyFrom(strArena, intScopeView.ptr, intScopeView.len);
                            continuationScope = StrSpan(intScopeName);
                        }
                        integrationProcessor.prepareIntegrationCore2(wiCopy, changeableMap, 1, mb,
                                                continuationScope, StrSpan("", 0), /*continuation=*/true);
                    }
                }
            }
        }


/// @brief Seed the recursion admission budget for a freshly-registered
///        operator GOAL at the scope that owns the goal's numeral.
///
/// @details
/// Sole caller: `addExprToMemoryBlock`'s `status == 2` (toBeProved) branch,
/// when the goal's head operator has input args. The goal `(op[…,X])` is an
/// operator output to be PROVED; the admission budget seeded here is what lets
/// the hash engine unfold that operator backward while chasing the goal, capped
/// by `maxAdmissionDepth` / `maxSecondaryNumber`. The back-climb finds the first
/// ancestor LB whose `exprKey` shares a digit argument with the goal — the
/// recursion/fold block that introduced that numeral — and seeds the admission
/// there (delegating to `updateAdmissionMap`). A self-match seeds the calling
/// LB. This is goal-side backward-admission setup, NOT result aggregation.
///
/// I-28 detect-and-defer (trial): when the climb lands on a STRICT ANCESTOR and
/// the caller is a parallel phase-1/phase-3 worker (`g_inParallelWorkerPhase`),
/// writing the ancestor would be a cross-LB write that races sibling recursion
/// children sharing one fold ancestor. In that case the resolved
/// `updateAdmissionMap` call is sealed onto the shared `deferredAncestorPages`
/// `SealedPageSet` record chain and replayed single-threaded by
/// `drainDeferredAncestorAdmissions` post-join. A
/// self-write (always the worker's own LB) and any write outside the parallel
/// sweep (grid build, the drain itself) stay inline.
///
/// @param expr               Operator goal whose admission budget is seeded.
/// @param memoryBlock        LB the goal was registered in (climb start).
/// @param maxAdmissionDepth  Recursion admission depth cap.
/// @param maxSecondaryNumber Recursion secondary-iteration cap.
/// @param partOfRecursion    Marks the seeded entry as recursion-owned.
/// @invariant [I-28](../../docs/agentic_swdd/30_invariants.md#i-28) — a
///            cross-LB ancestor seed never lands inline in a parallel worker.
/// @see `drainDeferredAncestorAdmissions`, `g_inParallelWorkerPhase`,
///      `updateAdmissionMap`.
inline void AdmissionMemory::updateAdmissionMap3(StrSpan expr,
    Memory& memoryBlock,
    int maxAdmissionDepth,
    int maxSecondaryNumber,
    bool partOfRecursion) {
    // 1) Compute digit arguments of the expression (uses analyzer's anchor).
    // findDigitArgs is now 0% heap (sorted-unique StrSpan run == the former set
    // order); materialize the std::set in this still-heap caller. The compareSpans
    // order is preserved, which the downstream remaining-args run relies on -- that
    // run is deload-observable through the deferred-admission replay.
    StrSpan digitBuf[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
    const int32_t digitN = scopeTypes.findDigitArgs(expr, this->anchorInfo, digitBuf,
                                         ExecutionParameters::MAX_ADMISSION_REM_ARGS);
    // digitBuf is already sorted-unique (== the former std::set<std::string>
    // order), so it IS the digit-arg set -- no std::set materialized.

    // 2) Walk up the Memory chain to find the first block whose exprKey shares a digit arg
    Memory* tempMb = &memoryBlock;
    while (true) {
        // Args of this block's exprKey (zero-copy slices; no std::set -- the
        // intersection below is a pure membership test, order not observable).
        StrSpan keyArgs[ExecutionParameters::MAX_ARITY];
        const int32_t keyArgsN = getArgsSpans(tempMb->exprKeyView(), keyArgs,
                                              ExecutionParameters::MAX_ARITY);

        // Check intersection with the digit args (nested equalSpans scan; the
        // bool result is independent of iteration order).
        bool intersects = false;
        for (int32_t i = 0; i < digitN && !intersects; ++i) {
            for (int32_t j = 0; j < keyArgsN; ++j) {
                if (equalSpans(digitBuf[i], keyArgs[j])) { intersects = true; break; }
            }
        }

        if (intersects) {
            // Remaining args = digitArgs ∪ { last argument of expr }
            StrSpan exprArgs[ExecutionParameters::MAX_ARITY];
            const int32_t exprArgsN = getArgsSpans(StrSpan(expr), exprArgs,
                                                   ExecutionParameters::MAX_ARITY);
            if (exprArgsN > 0) {
                // remainingArgs = digitArgs U { last arg }, sorted-unique. Seed
                // with digitBuf (already sorted-unique == the former std::set
                // order), then sorted-insert-with-dedup the last arg. This run is
                // DELOAD-OBSERVABLE via the deferred-admission replay, so it MUST
                // reproduce the former std::set ascending order byte-for-byte
                // (compareSpans == std::string::compare). Every span aliases
                // `expr` (caller-stable) -- no interner mint touches them (I-3).
                StrSpan remRun[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
                assert(digitN <= ExecutionParameters::MAX_ADMISSION_REM_ARGS
                    && "updateAdmissionMap3: digit-arg count exceeds cap");
                for (int32_t i = 0; i < digitN; ++i) remRun[i] = digitBuf[i];
                int32_t remRunN = digitN;
                {
                    const StrSpan last = exprArgs[exprArgsN - 1];
                    int32_t pos = 0;
                    bool dup = false;
                    while (pos < remRunN) {
                        const int c = compareSpans(remRun[pos], last);
                        if (c == 0) { dup = true; break; }
                        if (c > 0) break;
                        ++pos;
                    }
                    if (!dup) {
                        assert(remRunN < ExecutionParameters::MAX_ADMISSION_REM_ARGS
                            && "updateAdmissionMap3: remaining-arg count exceeds cap");
                        for (int32_t k = remRunN; k > pos; --k) remRun[k] = remRun[k - 1];
                        remRun[pos] = last;
                        ++remRunN;
                    }
                }

                // I-28 detect-and-defer: a STRICT-ANCESTOR seed (the climb left
                // the calling LB) reached from inside a parallel worker is a
                // cross-LB write — seal it onto the shared page set for the
                // single-threaded post-join drain. A self-seed (tempMb ==
                // &memoryBlock, the worker's own LB) and any seed outside the
                // parallel sweep (grid build, the drain's own re-entry) apply
                // inline. The mutex serializes ALL page-set writes (alloc /
                // appendRecord are not internally thread-safe).
                if (tempMb != &memoryBlock && gl::WorkerContext::inParallelWorkerPhase()) {
                    std::lock_guard<std::mutex> lock(this->deferredAncestorAdmissionsMutex);
                    assert(this->deferredAncestorPages
                        && "updateAdmissionMap3: deferredAncestorPages not emplaced by proveKernel");
                    SealedPageSet& ps = *this->deferredAncestorPages;
                    // key = [expr] : one SealedString + a 1-element SealedSpan.
                    SealedString keyElems[1] = {
                        SealedString::copyFrom(ps, expr.ptr, expr.len) };
                    SealedSpan<SealedString> keySpan =
                        SealedSpan<SealedString>::copyFrom(ps, keyElems, 1);
                    // remainingArgs : seal each remRun element, then the run
                    // (byte-identical to the former sorted-unique set order).
                    SealedString remElems[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
                    for (int32_t i = 0; i < remRunN; ++i)
                        remElems[i] = SealedString::copyFrom(ps, remRun[i].ptr, remRun[i].len);
                    SealedSpan<SealedString> remSpan =
                        SealedSpan<SealedString>::copyFrom(ps, remElems, remRunN);
                    ps.appendRecord(DeferredAncestorAdmission{ tempMb, keySpan,
                        remSpan, maxAdmissionDepth, maxSecondaryNumber,
                        partOfRecursion });
                }
                else {
                    // Inline: key = [expr], remainingArgs = the sorted-unique
                    // remRun. Both outlive the call (expr caller-stable; remRun a
                    // stack copy of spans over stable input).
                    StrSpan keyRun[1] = { StrSpan(expr) };
                    this->updateAdmissionMap(*tempMb,
                        keyRun, 1,
                        remRun, remRunN,
                        maxAdmissionDepth,
                        maxSecondaryNumber,
                        partOfRecursion,
                        StrSpan("main", 4));
                    // WAKE DOOR 3 (inline twin, D-194): an
                    // inline (single-threaded) admission write into tempMb — its
                    // own LB when self-seeding (harmless), or an ancestor on a
                    // non-parallel re-entry climb — is rule state that can fire
                    // next burst; keep the target swept.
                    tempMb->hasWork = true;
                }
            }
            break;
        }
        else {
            if (tempMb->parentMemory != NULL) {
                tempMb = tempMb->parentMemory;
            }
            else {
                break;
            }
        }
    }
}
}
