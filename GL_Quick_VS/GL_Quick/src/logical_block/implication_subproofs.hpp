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
#include "canonical_template_matcher.hpp"
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
class RejectedSourceReplay;
class StatementMemory;
namespace tests { struct ImplicationSubproofsFixture; }
/// @brief Implication Subproofs operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class ImplicationSubproofs {
friend struct tests::ImplicationSubproofsFixture;
public:
/// @brief Bind inputs for implication subproofs.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param canonicalTemplates Shared canonical matcher used by parked-subproof maintenance.
/// @param canonicalizer Borrowed Canonicalizer interface or state; must outlive this component.
/// @param definitions Read-only compiled implication templates and their formal inputs.
/// @param integrationBuilder Borrowed IntegrationInstructionBuilder interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param equivalenceQueries Borrowed EquivalenceQueries interface or state; must outlive this component.
/// @param statementMemory Borrowed StatementMemory interface or state; must outlive this component.
/// @param rejectedSourceReplay Borrowed RejectedSourceReplay interface (the replay of a promoted trial's source); must outlive this component.
/// @return None; construction binds the supplied references.
ImplicationSubproofs(CanonicalTemplateMatcher& canonicalTemplates, Canonicalizer& canonicalizer, const CompiledDefinitionRegistry& definitions, IntegrationInstructionBuilder& integrationBuilder, const ProverParameters& parameters, EquivalenceQueries& equivalenceQueries, StatementMemory& statementMemory, RejectedSourceReplay& rejectedSourceReplay);

private:
/// @brief Whether one element of an and gates the and's implication
        ///        subproofs (D-406).
        ///
        /// @details
        /// A gate element is a non-implication element of the and — atomic,
        /// negated or compound — whose every argument is fixed (`u_`-prefixed
        /// in the rewritten instruction), so that after the `u_` strip it is a
        /// ground statement that can become known. An implication element — a
        /// numbered `(implication<N>[…])` or a configured definition whose
        /// whole body is one implication (`subset`, `primeseq`; D-467) — is
        /// integrated through its own subproof and never gates. An element
        /// with a bound variable or a `pi_` / `marker` pattern argument is not
        /// a statement and never gates.
        ///
        /// @param element     One element of an and entity of a rewritten
        ///                    integration instruction (flat MPL, at least one
        ///                    argument).
        /// @param definitions The compiled registry naming the configured
        ///                    implication definitions.
        /// @return True iff @p element gates the and's implication subproofs.
        /// @invariant Pure; allocates nothing.
        /// @see openImplicationSubproof, parkImplicationSubproof,
        ///      CompiledDefinitionRegistry::isConfiguredImplication.
static bool isSubproofGateElement(StrSpan element, const CompiledDefinitionRegistry& definitions);

private:
/// @brief Whether entity @p p of @p wi is an and that lists
        ///        @p signature as one of its elements.
        ///
        /// @details
        /// The parent relation of an integration instruction: an entity
        /// carries no parent link, and an and's element text equals the
        /// signature of the child entity it names. The implication gate reads
        /// its parent ands through this test, in instruction order.
        ///
        /// @param wi        Integration instruction; read only.
        /// @param p         Entity index in @p wi.
        /// @param signature Signature text of the candidate child entity.
        /// @return True iff entity @p p has category `and` and an element
        ///         byte-equal to @p signature.
        /// @invariant Pure; allocates nothing.
        /// @see openImplicationSubproof, parkImplicationSubproof.
static bool isParentAndOf(const WorkInstruction& wi, int32_t p, StrSpan signature);

public:


private:
/// @brief Park one implication subproof in `rejectedMapSubproof`
        ///        under an unknown gate element of its and
        ///        (D-406).
        ///
        /// @details
        /// Builds the park record in the `admissionMapIntegration` value
        /// family: entity 0 is the implication entity of @p wi verbatim
        /// (category, signature, defined set, arity, elements with the head
        /// last); entities 1..k are the ands of @p wi that list the
        /// implication's signature, verbatim and in instruction order; the
        /// marked goal is @p rootGoal. The record keeps the instruction's own
        /// text — the text the and's integration rule was installed with —
        /// while the key is canonical: the clean gate text normalized through
        /// the complete classes of @p validityName, so class-equal gates share
        /// one key. The record is filed through the canonical
        /// read-modify-write (`ArenaIntegrationMap`), which deduplicates a
        /// repeated park.
        ///
        /// @param wi           Rewritten integration instruction; read only.
        /// @param entity       Index of the implication entity in @p wi.
        /// @param gate         Clean (u_-stripped) text of the unknown gate
        ///                     element.
        /// @param mb           Owning block; receives the park.
        /// @param validityName Registered preparing scope (the park scope).
        /// @param rootGoal     Goal embedded by a main-level opening, else
        ///                     empty.
        /// @invariant Mints the key into templateInterner / nameMap and the
        ///            record into valueInterner; opens no scope.
        /// @see openImplicationSubproof, revisitRejectedSubproof,
        ///      canonicalAdmissionText.
void parkImplicationSubproof(const WorkInstruction& wi, int32_t entity,
            StrSpan gate, Memory& mb, StrSpan validityName, StrSpan rootGoal);

public:
/// @brief Open the boundary subproof of one implication entity of an
        ///        integration instruction — Case A of `prepareIntegrationCore2`.
        ///
        /// @details
        /// The one opening path of an implication element. In order:
        /// 1. Known part: the clean (u_-stripped) signature that is already a
        ///    known statement at the preparing scope, up to the scope's
        ///    complete classes (I-254), needs no subproof under any goal.
        /// 2. Prepared gate: the scope payload — `<rootGoal>_subproof_<clean>`
        ///    on main when a root goal drives the integration, the bare clean
        ///    signature otherwise — that is already prepared at the scope
        ///    (canonically, through the prepared-template index) is not
        ///    opened again. For a @p continuation a preparation at a
        ///    strict ancestor counts too (`canonicalTemplatePresentAtOrAbove`,
        ///    D-572).
        /// Equal-input refusal applies only when the owning block has its
        /// induction-step hypothesis installed (recursionAuxyImplId != 0).
        /// A direct positive equality premise permits its exact input pair.
        /// Built-on-listed witness refusal (I-293) also precedes the gate.
        /// 3. Gate (D-406): for every and of @p wi
        ///    that lists the implication, each gate element
        ///    (`isSubproofGateElement`) must be known at the scope or an
        ///    ancestor (`elementKnownAtOrAbove`). The first unknown one, in
        ///    instruction order, parks the implication in
        ///    `rejectedMapSubproof` (`parkImplicationSubproof`); the drain of
        ///    known-row arrivals reopens it through this function.
        /// 4. Opening: mint the prepared key, push the scope
        ///    `<validityName>_boundary_<payload>`, record the
        ///    expansion-for-integration origin, then add every renamed premise
        ///    at the new scope as status 0 with its premise-element origin and
        ///    the renamed head as status 2.
        /// 5. Reductio: a bare numbered implication (no embedded goal)
        ///    outside a primed contradiction block also gets its reductio
        ///    beside the subproof, `contradiction_!(<compact>)` at the same
        ///    scope (`prepareNegatedCompoundContradiction`); its discharge
        ///    writes the negated compact there, and
        ///    `retireSettledImplicationSubproofs` retires the subproof
        ///    (D-532).
        ///
        /// A goal embedded in the payload lets a later disproof of that goal
        /// enumerate and wipe the scope (`splitSubproofPayload`); deeper
        /// scopes need no embedding because they ride their main-level
        /// ancestor's subtree wipe.
        ///
        /// @param wi            Rewritten integration instruction (replacement
        ///                      map and canonical admission text already
        ///                      applied); read only.
        /// @param entity        Index of the implication entity in @p wi:
        ///                      category `implication`, every signature
        ///                      argument `u_`-prefixed.
        /// @param mb            Owning block; receives the prepared key, the
        ///                      scope and its seeded statements.
        /// @param validityName  Registered preparing scope.
        /// @param rootGoal      The goal whose integration drives this call
        ///                      (verbatim), or empty for statement-driven
        ///                      preparation.
        /// @param continuation  Whether the call replays a stored instruction
        ///                      after a marker fill
        ///                      (`IntegrationProcessor::prepareIntegrationCore2`).
        /// @return `Known`, `Prepared`, `Refused`, `Parked` or `Opened` (see
        ///         `ImplicationSubproofOutcome`).
        /// @invariant Asserts the entity contract of @p entity; every string
        ///            it builds rides its own string-scratch scope, rewound on
        ///            return.
        /// @see prepareIntegrationCore2 and revisitRejectedSubproof (the
        ///      callers), compactKnownUpToEquality,
        ///      canonicalTemplatePresent, renamingChain2Scratch,
        ///      expandSignatureForIntegrationScratch,
        ///      prepareNegatedCompoundContradiction.
ImplicationSubproofOutcome openImplicationSubproof(const WorkInstruction& wi,
            int32_t entity, Memory& mb, StrSpan validityName, StrSpan rootGoal,
            bool continuation = false);

/// @brief Contradiction-based integration of a negated compound
        ///        compact — mint a `contradiction_`-marked child scope and seed
        ///        it with the positive compact.
        ///
        /// @details
        /// The GL-native proof of a negated compound is reductio: assume the
        /// positive compact `(X[args])` as a status-0 statement in a
        /// dedicated child scope whose payload is `contradiction_!(X[args])`,
        /// pushed on @p validityName. The standard consumption machinery
        /// decomposes the seed (existence-bodied → witness facts, AND-bodied
        /// → conjuncts, implication-bodied → installed rule) and derives
        /// forward; when a contradiction fires at the scope,
        /// `dischargeContradictionScopes` emits the negated compact at the
        /// scope's parent and wipes the scope. No template is instantiated
        /// for the negated form — the polarity travels with the expression.
        /// Or-bodied compacts are never passed (all-branch refutation
        /// semantics; see the intercept comment).
        ///
        /// Two callers: the negated-compound intercept of
        /// `IntegrationProcessor::prepareIntegration` (a negated compound
        /// to be proved) and `openImplicationSubproof`, which primes the
        /// reductio of every bare numbered implication subproof it opens
        /// outside a primed contradiction block: a counterexample at the
        /// parent then writes the negated compact there, and
        /// `retireSettledImplicationSubproofs` retires the subproof
        /// (D-532).
        ///
        /// Idempotence gates, in order: the product already `registered` at
        /// @p validityName or at `"main"` (nothing to prove); the scope
        /// payload already minted in `integrationPrepared` at
        /// @p validityName (the scope exists or was already discharged).
        ///
        /// The seed's history row is `task formulation` (empty antecedents)
        /// at the contradiction scope — the primed-`__contradiction__` seed
        /// contract. The mail-side origin mirror rides `addMailOutOrigin`
        /// (origins are all-scopes per I-26); the statement itself is NOT
        /// mailed (statements are main-only per I-26).
        ///
        /// @param expression   The negated compact `!(X[args])` (span over a
        ///                     caller-stable buffer).
        /// @param mb           The owning LB.
        /// @param validityName The parent scope for the contradiction scope.
        /// @invariant Heap-free; interner mints only behind the non-minting
        ///            probes; single-threaded call context (admission seeding
        ///            seams).
        /// @see IntegrationProcessor::prepareIntegration and
        ///      openImplicationSubproof — the callers;
        ///      dischargeContradictionScopes — the discharge half;
        ///      contradictionScopeProductView — the payload parser;
        ///      retireSubproofReductios — the reductio's end with its
        ///      subproof.
void prepareNegatedCompoundContradiction(StrSpan expression,
            Memory& mb,
            StrSpan validityName);

/// @brief Park the reductio of a negated hypothesis premise under its first
///        positive premise not yet known, or report that every positive
///        premise is known.
///
/// @details
/// The gates are the hypothesis's fully bound positive premises
/// (`IntegrationProcessor::prepareHypothesisPremise` selects them). The
/// first gate not known at @p validityName or an ancestor
/// (`elementKnownAtOrAbove`, up to the scope's classes) keys the park: the
/// key is that gate's text normalized through the scope's complete classes
/// (`canonicalAdmissionText`), as for an implication park, and the record
/// is one entity of category `reductio` whose signature is @p negated and
/// whose elements are all gates. The record is filed in
/// `rejectedMapSubproof` through the canonical read-modify-write, which
/// deduplicates a repeated park; the arrival drain revives it
/// (`revisitRejectedSubproof`), which re-runs this check and opens the
/// reductio once every gate is known
/// (I-326).
///
/// @param negated      The negated premise `!(X[args])` (span over a
///                     caller-stable buffer).
/// @param gates        The positive gates (spans over caller-stable
///                     buffers).
/// @param gateN        Their number; zero means nothing to wait for.
/// @param mb           Owning block.
/// @param validityName Registered preparing scope (the park scope).
/// @return True when the reductio parked; false when every gate is known.
/// @invariant Opens no scope; mints only the park key and its record.
/// @see prepareNegatedCompoundContradiction, revisitRejectedSubproof,
///      IntegrationProcessor::prepareHypothesisPremise.
bool deferPremiseReductio(StrSpan negated, const StrSpan* gates, int32_t gateN,
            Memory& mb, StrSpan validityName);

/// @brief Category of a parked negated-premise reductio record in
///        `rejectedMapSubproof`.
static constexpr const char* kReductioParkCategory = "reductio";

/// @brief Category of a parked contradiction-scope demand candidate in
///        `rejectedMapSubproof` (D-560).
static constexpr const char* kContradictionDemandParkCategory = "contradiction_demand";

public:
/// @brief The verdict on one contradiction-scope demand candidate
///        (D-560).
enum class ContradictionDemandVerdict {
    /// The candidate's one unknown compound is to be integrated at the scope.
    Ready,
    /// The candidate waits in `rejectedMapSubproof` for a statement.
    Parked,
    /// The candidate can never become the scope's contradiction, or its rule
    /// fires without help.
    Dropped
};

/// @brief Whether a premise is a positive registry compound of category
///        `and` — the only premise a contradiction-scope demand integrates
///        (D-560).
///
/// @details
/// A negated premise, an operator the registry does not compile and an
/// entity of any other category (atomic, existence, or, implication) is not
/// demandable: only an and carries the integration rules and gate elements
/// the demand relies on.
///
/// @param premise Premise text, clean or `u_`-prefixed; only the sign and the
///                operator name are read.
/// @return True iff @p premise is positive and its operator compiles to an
///         entity of category `and`.
/// @invariant Pure; allocates nothing.
/// @see deferContradictionDemand, compoundGateElements.
bool isDemandableCompound(StrSpan premise) const;

/// @brief The gate elements of an and compound instance, as the clean
///        statements they wait for (D-560).
///
/// @details
/// The compiled entity's elements name its signature tokens. Each token is
/// replaced by `u_` and the instance's argument at the same position; an
/// element that then passes `isSubproofGateElement` is a gate, and its clean
/// form (the `u_` prefixes removed) is the statement it waits for — the gate
/// elements the and's integration instruction carries (D-406).
///
/// @param compound Clean instance `(X[args])` of an and entity
///                 (`isDemandableCompound`, asserted).
/// @param strings  String-scratch arena receiving the texts; the caller's
///                 frame keeps them.
/// @param gates    Receives the gate statements.
/// @param cap      Capacity of @p gates; more gates assert.
/// @return The number of gate elements.
/// @invariant Zero heap; mints nothing.
/// @see isSubproofGateElement, deferContradictionDemand.
int32_t compoundGateElements(StrSpan compound, ScratchArena& strings, StrSpan* gates,
            int32_t cap) const;

/// @brief Decide one contradiction-scope demand candidate: demand its one
///        unknown compound, park it under what it waits for, or drop it
///        (D-560).
///
/// @details
/// The candidate is a rule whose head negates a fact visible at the
/// contradiction scope @p scopeId, every premise fully bound by that match
/// (@p premises, clean, in chain order). In order:
/// 1. Every premise known at the scope or an ancestor
///    (`elementKnownAtOrAbove`, up to the scope's classes): dropped, since the
///    rule fires without help.
/// 2. Exactly one premise unknown and it is a demandable compound
///    (`isDemandableCompound`): the gate. The first gate element
///    (`compoundGateElements`) not known at the scope or an ancestor parks
///    the candidate. With every gate element known, at least one must be
///    known at the scope itself (`compactKnownUpToEquality`, through the
///    scope's classes); otherwise, and for a compound without gate elements,
///    the candidate is dropped: a compound whose gate holds only above the
///    scope cannot become this scope's contradiction. Otherwise the candidate
///    is ready and @p readyIndex names the compound.
/// 3. Otherwise the candidate parks under its first unknown premise that is
///    not a demandable compound, the premise ordinary derivation must supply,
///    or under the first unknown premise when every unknown one is a
///    compound.
/// A park files one record of category `contradiction_demand` whose elements
/// are the premises, keyed by the awaited statement normalized through the
/// scope's complete classes (`canonicalAdmissionText`), through the canonical
/// read-modify-write that deduplicates a repeated park; the arrival drain's
/// revival (`revisitRejectedSubproof`) calls this function again.
///
/// @param premises   Clean fully bound premises (spans over caller-stable
///                   buffers).
/// @param premiseN   Their number, at least one.
/// @param mb         Owning block.
/// @param scopeId    The contradiction scope.
/// @param readyIndex Receives the compound's index on a ready verdict, else
///                   `-1`.
/// @return The verdict.
/// @invariant Opens no scope; a park mints only its key and its record.
/// @see isDemandableCompound, compoundGateElements, revisitRejectedSubproof,
///      ContradictionDemand::drainContradictionDemands.
ContradictionDemandVerdict deferContradictionDemand(const StrSpan* premises, int32_t premiseN,
            Memory& mb, NameId scopeId, int32_t& readyIndex);

/// @brief Stage a ready contradiction-scope demand for the demand drain
///        (D-560).
///
/// @details
/// The compound is interned in the block's template interner, the id space of
/// admission and park templates, so no statement id is minted for it; the
/// drain integrates it at the scope.
///
/// @param mb       Owning block.
/// @param compound Clean compound text.
/// @param scopeId  The contradiction scope.
/// @return Nothing.
/// @invariant Appends to `pendingContradictionDemands` only.
/// @see deferContradictionDemand, ContradictionDemand::drainContradictionDemands.
static void stageContradictionDemand(Memory& mb, StrSpan compound, NameId scopeId);

public:


private:
/// @brief Move one park to its canonical gate key
        ///        (D-406).
        ///
        /// @details
        /// The key's clean gate text is normalized through the complete
        /// classes of its park scope; an unchanged text keeps the key.
        /// Otherwise the key's records move, verbatim, to the canonical key:
        /// they are read in decoded order, the old key is erased, and each
        /// record is merged into the canonical key's run through the
        /// read-modify-write, which deduplicates records already parked
        /// there. Only the key moves; a record keeps the instruction's own
        /// text, which the and's integration rule expects.
        ///
        /// @param key Packed `(gate template, park scope)` key with records.
        /// @param mb  Owning block.
        /// @return The key that now holds the records (== @p key when it was
        ///         already canonical).
        /// @invariant Records are never rewritten.
        /// @see parkImplicationSubproof, drainSubproofArrivals.
int64_t rekeySubproofPark(int64_t key, Memory& mb);

public:
/// @brief Erase the parks a disproved main goal owns
        ///        (D-406).
        ///
        /// @details
        /// A main-level park records the goal whose integration parked it.
        /// When that goal is disproved its machinery is dead: every record at
        /// a main key whose marked goal is @p goal is removed, a key left
        /// without records is erased, and the other keys are rewritten with
        /// their surviving records. Keys are visited in decoded order.
        ///
        /// @param mb   Owning block.
        /// @param goal Verbatim text of the disproved main goal.
        /// @invariant Touches only main-scope keys; other goals' records stay.
        /// @see drainDisprovedGoals, revisitRejectedSubproof.
void eraseGoalSubproofParks(Memory& mb, StrSpan goal);

private:
/// @brief Reopen the implications parked under one gate key once the
        ///        gate element is known (D-406).
        ///
        /// @details
        /// The revival of `rejectedMapSubproof`. A key without records (erased
        /// earlier in the same drain, or wiped with its scope) and a gate that
        /// is still not known at the park scope or an ancestor
        /// (`elementKnownAtOrAbove`) are defined no-ops; the park stands.
        /// Otherwise the records are snapshotted in decoded order and the key
        /// is erased. A parked negated-premise reductio (category `reductio`)
        /// opens its `contradiction_` scope once every positive premise it
        /// waits for is known, else parks under the next unknown one
        /// (`deferPremiseReductio`). A parked contradiction-scope demand
        /// candidate (category `contradiction_demand`) is decided again
        /// (`deferContradictionDemand`), and a ready one is staged for the
        /// demand drain (`stageContradictionDemand`). Each other record is
        /// then dropped when its root goal is no
        /// longer a main-level goal or when one of its parent ands is itself
        /// known; every other record is handed to `openImplicationSubproof`
        /// exactly as stored, which opens it or parks it again under its next
        /// unknown gate element. The record keeps the instruction's own text
        /// (the text the and's integration rule was installed with), so the
        /// opened subproof proves the implication the and waits for; the
        /// checks inside the opening compare through the current classes.
        ///
        /// @param parkKey Packed `(gate template, park scope)` key.
        /// @param mb      Owning block.
        /// @invariant Never re-entered for the same key (asserted through
        ///            `subproofRevisitInProgress`); every walk follows the
        ///            decoded record order (I-84).
        /// @see drainSubproofArrivals, openImplicationSubproof,
        ///      parkImplicationSubproof, deferPremiseReductio.
void revisitRejectedSubproof(int64_t parkKey, Memory& mb);

public:
/// @brief Match the staged known-row arrivals and class changes against
        ///        the parked gate keys and reopen every park whose gate became
        ///        known (D-406).
        ///
        /// @details
        /// Runs first in step 4b of `standardProcessing`. First the class
        /// changes: every park at or below a scope staged by
        /// `stageSubproofRecheck` moves to its canonical key
        /// (`rekeySubproofPark`, decoded order) and is rechecked, since a
        /// merge can make a gate known without a new row. Then the arrivals:
        /// a row known at scope A is known at A and every descendant, so each
        /// arrival visits the park scopes of its shape (`scopesByShape` of
        /// the park index) that are A or below A and collects, through the
        /// canonical template index, the park keys that equal the row
        /// literally or up to that park scope's classes. Both queues are
        /// detached before the revisits, which run in decoded
        /// key-then-scope order; work caused by a reopened subproof waits for
        /// the next drain. Parks under a Phase-3-ignored scope are skipped.
        ///
        /// @param mb Owning block.
        /// @invariant Consumes `pendingSubproofArrivals` and
        ///            `pendingSubproofRechecks` completely; opens scopes only
        ///            through `revisitRejectedSubproof`.
        /// @see stageSubproofArrival, stageSubproofRecheck,
        ///      rekeySubproofPark, revisitRejectedSubproof,
        ///      matchingCanonicalTemplates.
void drainSubproofArrivals(Memory& mb);

public:
/// @brief Promote every open integration trial whose gate became known
        ///        (D-481).
        ///
        /// @details
        /// Runs right after `drainSubproofArrivals` in step 4b of
        /// `standardProcessing`. An open trial passes when every gate element
        /// of one of its groups is known at its trial scope
        /// (`elementKnownAtOrAbove`, through the trial scope's classes). The
        /// trials are checked in record order first and promoted afterwards:
        /// a passed record becomes the parent's receipt, its source is
        /// replayed at the parent (`RejectedSourceReplay::replayTrialSource`),
        /// where its recorded witnesses are admitted under the names the trial
        /// minted, with the recorded expressions, and the
        /// trial scope joins the pending wipe and the validity filter, so the
        /// wipe drain of the same call removes the candidate's trial facts.
        ///
        /// @param mb Owning block.
        /// @return Void.
        /// @invariant Mints no scope; passed records are kept, open ones are
        ///            only read.
        /// @see DisintegrationProcessor::disintegrateExpr2 (trial opening),
        ///      RejectedSourceReplay::replayTrialSource, Memory::wipeSubtrees.
void drainIntegrationTrials(Memory& mb);

public:
/// @brief End-of-burst retirement of settled implication subproofs —
///        a subproof whose implication is already known or refuted at
///        its parent is wiped, and of two subproofs whose implications
///        became class-equal the one with more statements survives.
///
/// @details
/// Candidates are the OPEN implication subproofs with a bare payload:
/// `prepareIntegrationCore2` Case A scopes opened by an arriving
/// statement (admission and marker replays) or nested below another
/// scope. The prepared key at the parent names the payload; the
/// scope is `<parent>_boundary_<payload>`, found by a non-minting
/// lookup. A scope in `intValidityNamesToFilter` (already wiped) or
/// queued on `pendingWipeScopes`, itself or through an ancestor, is
/// not open. Goal-driven subproofs (`<goal>_subproof_` payloads)
/// follow their goal's lifecycle and are not candidates: disproof
/// cleanup relies on each goal owning its own copy.
///
/// Three retirements, all queued on `pendingWipeScopes` so this
/// burst's wipe drain removes them:
/// 1. Known part: the payload is a known statement at the parent
///    scope, exactly or through the parent's complete classes
///    (`compactKnownUpToEquality`) — the subproof can add nothing.
/// 2. Refuted: outside a primed contradiction block, the negated
///    payload is known at the parent scope or an ancestor, literally
///    or through the parent's complete classes
///    (`elementKnownAtOrAbove`) — written there by the subproof's
///    reductio (`prepareNegatedCompoundContradiction`, primed at
///    opening) or derived otherwise. The implication is false at the
///    parent, so the subproof can never close. A refutation at a scope
///    retires the implication's subproofs at that scope and below it,
///    never above it (D-532). A primed
///    block keeps its subproofs: its parent is meant to be
///    inconsistent.
/// 3. Collapse: open candidates under one parent whose payloads have
///    the same canonical form at the parent (`canonicalFormAtScope`)
///    are one case; the KEEPER is the one with the most statements in
///    its scope and nested scopes (`scopeSubtreeStatementCount`),
///    ties to the byte-lex smaller payload — deterministic. Every
///    other one is wiped.
/// The prepared key of a wiped subproof stays, so admission never
/// reopens it, and the wipe adds the scope to
/// `intValidityNamesToFilter`, so it is never queued again. Wipes
/// never touch proof history (I-44). The remaining work runs under
/// the survivor, whose result every equal compact matches through the
/// classes (I-254).
///
/// Runs in `standardProcessing` after the OR freeze and release
/// drains, immediately before the `pendingWipeScopes` drain, so no
/// other sweep sees a queued subproof. Heap-free (Rule 28): the
/// candidate runs ride the gen-scratch tier and the payload, scope
/// name and canonical texts ride the string tier, reclaimed at exit.
/// Mints nothing into the NameMap or the template interner.
///
/// @param body The LB whose open subproofs are swept (containers
///             resident — the drain runs inside the LB's own burst).
/// @return (void)
/// @invariant Candidates process in `integrationPrepared` insertion
///            order; the keeper of a collapse group is a pure function
///            of container content.
/// @invariant A scope is queued at most once over the LB's lifetime.
/// @see retireDuplicateOrCohorts — the OR-cohort twin;
///      compactKnownUpToEquality; elementKnownAtOrAbove;
///      scopeSubtreeStatementCount; retireSubproofReductios — the
///      reductio scopes of the retired subproofs.
void retireSettledImplicationSubproofs(Memory& body);

public:
/// @brief Queue the reductio scope of every closed or queued bare implication
///        subproof for the wipe drain.
///
/// @details
/// A bare numbered implication subproof opened outside a primed contradiction
/// block has its reductio beside it: `contradiction_!(<payload>)` pushed on
/// the same parent (`openImplicationSubproof`, step 5). The reductio serves
/// only the subproof's retirement, so it ends with the subproof
/// (D-532): once the subproof scope is closed
/// (in `intValidityNamesToFilter` — proved, retired or wiped with an
/// ancestor) or queued on `pendingWipeScopes` in this burst, its reductio
/// scope, when it exists and is still open, is queued too. A reductio that
/// discharged its refutation already queued itself, and one below a closed or
/// queued ancestor goes with that ancestor.
///
/// Candidates come from `integrationPrepared` in insertion order: bare
/// payloads starting `(implication`, whose subproof scope
/// `<parent>_boundary_<payload>` exists. Both scope names are found by
/// non-minting lookups. A primed contradiction block is skipped: its subproofs
/// get no reductio.
///
/// Runs in `standardProcessing` after every other retirement of the burst and
/// immediately before the `pendingWipeScopes` drain, so it sees every scope
/// queued in this burst. Heap-free (Rule 28): the scope names ride the string
/// tier, reclaimed per key. Mints nothing into the NameMap or the template
/// interner; reads no proof history.
///
/// @param body The LB whose reductio scopes are swept (containers resident —
///             the drain runs inside the LB's own burst).
/// @return (void)
/// @invariant A reductio scope is queued at most once: the wipe adds it to
///            `intValidityNamesToFilter`.
/// @see openImplicationSubproof; prepareNegatedCompoundContradiction;
///      retireSettledImplicationSubproofs;
///      GoalProcessor::dischargeContradictionScopes.
void retireSubproofReductios(Memory& body);

private:
/// @brief Count runtime statements in one scope and every scope nested below
///        it.
///
/// @details The collapse census of `retireSettledImplicationSubproofs`: one
/// pass over `intEncodedStatements`, counting each row whose validity is the
/// scope itself or a strict descendant in the validity forest. Reads no proof
/// history and mints nothing.
///
/// @param body  Owning resident block.
/// @param scope Minted validity id of the subtree root.
/// @return Number of runtime statements in the subtree.
/// @see retireSettledImplicationSubproofs; orCohortStatementCount.
int32_t scopeSubtreeStatementCount(const Memory& body, NameId scope) const;
public:
/// @brief Test whether distinct implication inputs coincide without an equality premise.
///
/// @details The compiled signature defines the input positions independently of
/// their actual names. Only a direct positive equality premise between the same
/// two formal inputs permits their identification, in either orientation. The
/// conclusion, nested compounds and bound premise variables grant no permission.
/// Complete equality classes at the supplied scope include ancestors but never
/// sibling or descendant facts; weak representatives do not affect membership.
///
/// @param signature Clean numbered implication instance whose inputs are fixed.
/// @param scope Registered scope at which to compare the actual input objects.
/// @param body Resident logic block supplying the complete equality view.
/// @return True exactly when an unlicensed pair of distinct input positions is equal.
/// @invariant This search predicate derives no statement and reads no proof history.
/// @see retireEqualInputImplicationSubproofs; openImplicationSubproof.
bool hasForbiddenEqualInputs(StrSpan signature, NameId scope, Memory& body);

/// @brief Test whether an implication instance names a built-on-listed object.
///
/// @details An argument names a built-on-listed object (I-293) when it is in
/// `Memory::builtOnListedWitnesses` and every member of its complete equality
/// class at the scope is in the registry too. A class member outside the
/// registry names the same object without the listed construction, so the
/// verdict does not depend on which member the class-canonical choice
/// (`compareWitnessNames`, I-88) puts into the instance (I-217). An argument
/// in no class stands for itself. Complete equality classes at the supplied
/// scope include ancestors but never sibling or descendant facts.
///
/// @param signature Clean numbered implication instance whose inputs are fixed.
/// @param scope Registered scope at which to compare the actual input objects.
/// @param body Resident logic block supplying the registry and the complete equality view.
/// @return True exactly when some argument's whole class is built-on-listed.
/// @invariant This search predicate derives no statement and reads no proof history.
/// @see openImplicationSubproof; hasForbiddenEqualInputs.
bool hasBuiltOnListedArgument(StrSpan signature, NameId scope, Memory& body);

/// @brief Retire implication subproofs whose distinct inputs become equal without permission.
///
/// @details Only induction-step blocks (recursionAuxyImplId != 0) are checked;
/// ordinary, contradiction, typing and induction-zero blocks are unchanged.
/// Both bare and goal-qualified numbered implication subproofs are selected
/// through their prepared scope identities. The input predicate reads
/// each subproof's own complete equality view. A hit queues that scope and its
/// descendants for the ordinary deferred wipe; history and the prepared key
/// survive, so the same candidate cannot reopen. This abandons search and never
/// records a contradiction, a proved implication or a parent goal closure.
///
/// @param body Resident logic block whose open implication subproofs are checked.
/// @return Nothing; offending scope roots are added to pendingWipeScopes.
/// @invariant Closed scopes and descendants of an already queued root are skipped.
/// @see hasForbiddenEqualInputs; retireSettledImplicationSubproofs.
void retireEqualInputImplicationSubproofs(Memory& body);

/// @brief Retire the open induction scopes whose admitted names all equal
///        names their root holds.
///
/// @details An induction scope carries no assumption (D-519): its admission's
/// facts hold at its root, and a hypothesis instance on names the root
/// already holds is the root's in-place advance. A scope whose admitted
/// names, the one or two names its scope name carries after the chain
/// (`ScopeSyntax::inductionScopeNames`), each share a class of the scope's
/// complete equality view with a root name adds nothing the root does not
/// hold, so it joins the wipe drain (D-545). A
/// root name is an ordinary name (a number, `rec`, a copy) or a witness of an
/// ancestor block (its level below the block's). A scope of a settled tree,
/// and a scope that is wiped or queued itself or through an ancestor, is no
/// candidate, and so is an in-place scope, a one-name scope named by a root
/// name (D-546): it carries the path its aliases
/// duplicate. Nothing keeps a later admission from opening the scope again.
/// Only a block whose hypothesis ate (`LbMemory::inductionEats` non-empty)
/// scans its NameMap; candidates come in NameMap id order. Heap-free (Rule
/// 28): the payload names and the strong chain are copied onto the string
/// tier, the class views ride the gen-scratch tier.
///
/// @param body Resident logic block whose induction scopes are swept.
/// @return Nothing; retired scopes are added to pendingWipeScopes.
/// @invariant Deterministic: NameMap id order, one pass.
/// @see ScopeSyntax::inductionScopeNames; retireEqualInputImplicationSubproofs;
///      Memory::settleInductionTrees.
void retireAliasInductionScopes(Memory& body);

private:

CanonicalTemplateMatcher& canonicalTemplates;
Canonicalizer& canonicalizer;
const CompiledDefinitionRegistry& definitions;
IntegrationInstructionBuilder& integrationBuilder;
const ProverParameters& parameters;
EquivalenceQueries& equivalenceQueries;
StatementMemory& statementMemory;
RejectedSourceReplay& rejectedSourceReplay;
};
}
#include "../logical_block/equivalence_queries.hpp"
#include "../logical_block/statement_memory.hpp"
#include "../logical_block/rejected_source_replay.hpp"
namespace gl {

/// @brief Whether one element of an and gates the and's implication
        ///        subproofs (D-406).
        ///
        /// @details
        /// A gate element is a non-implication element of the and — atomic,
        /// negated or compound — whose every argument is fixed (`u_`-prefixed
        /// in the rewritten instruction), so that after the `u_` strip it is a
        /// ground statement that can become known. An implication element — a
        /// numbered `(implication<N>[…])` or a configured definition whose
        /// whole body is one implication (`subset`, `primeseq`; D-467) — is
        /// integrated through its own subproof and never gates. An element
        /// with a bound variable or a `pi_` / `marker` pattern argument is not
        /// a statement and never gates.
        ///
        /// @param element     One element of an and entity of a rewritten
        ///                    integration instruction (flat MPL, at least one
        ///                    argument).
        /// @param definitions The compiled registry naming the configured
        ///                    implication definitions.
        /// @return True iff @p element gates the and's implication subproofs.
        /// @invariant Pure; allocates nothing.
        /// @see openImplicationSubproof, parkImplicationSubproof,
        ///      CompiledDefinitionRegistry::isConfiguredImplication.
inline bool ImplicationSubproofs::isSubproofGateElement(StrSpan element,
            const CompiledDefinitionRegistry& definitions) {
            if (startsWithSpan(element, "(implication", 12)
                || definitions.isConfiguredImplication(element))
                return false;
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t argN = getArgsSpans(element, args, ExecutionParameters::MAX_ARITY);
            assert(argN > 0 && "isSubproofGateElement: an and element carries arguments");
            for (int32_t a = 0; a < argN; ++a)
                if (!(args[a].len >= 2 && args[a].ptr[0] == 'u' && args[a].ptr[1] == '_'))
                    return false;
            return true;
        }

/// @brief Whether entity @p p of @p wi is an and that lists
        ///        @p signature as one of its elements.
        ///
        /// @details
        /// The parent relation of an integration instruction: an entity
        /// carries no parent link, and an and's element text equals the
        /// signature of the child entity it names. The implication gate reads
        /// its parent ands through this test, in instruction order.
        ///
        /// @param wi        Integration instruction; read only.
        /// @param p         Entity index in @p wi.
        /// @param signature Signature text of the candidate child entity.
        /// @return True iff entity @p p has category `and` and an element
        ///         byte-equal to @p signature.
        /// @invariant Pure; allocates nothing.
        /// @see openImplicationSubproof, parkImplicationSubproof.
inline bool ImplicationSubproofs::isParentAndOf(const WorkInstruction& wi, int32_t p, StrSpan signature) {
            assert(p >= 0 && p < wi.entityCount() && "isParentAndOf: entity index out of range");
            if (!equalSpans(wi.category(p), StrSpan("and", 3))) return false;
            for (int32_t j = 0; j < wi.elemCount(p); ++j)
                if (equalSpans(wi.elemAt(p, j), signature)) return true;
            return false;
        }



/// @brief Park one implication subproof in `rejectedMapSubproof`
        ///        under an unknown gate element of its and
        ///        (D-406).
        ///
        /// @details
        /// Builds the park record in the `admissionMapIntegration` value
        /// family: entity 0 is the implication entity of @p wi verbatim
        /// (category, signature, defined set, arity, elements with the head
        /// last); entities 1..k are the ands of @p wi that list the
        /// implication's signature, verbatim and in instruction order; the
        /// marked goal is @p rootGoal. The record keeps the instruction's own
        /// text — the text the and's integration rule was installed with —
        /// while the key is canonical: the clean gate text normalized through
        /// the complete classes of @p validityName, so class-equal gates share
        /// one key. The record is filed through the canonical
        /// read-modify-write (`ArenaIntegrationMap`), which deduplicates a
        /// repeated park.
        ///
        /// @param wi           Rewritten integration instruction; read only.
        /// @param entity       Index of the implication entity in @p wi.
        /// @param gate         Clean (u_-stripped) text of the unknown gate
        ///                     element.
        /// @param mb           Owning block; receives the park.
        /// @param validityName Registered preparing scope (the park scope).
        /// @param rootGoal     Goal embedded by a main-level opening, else
        ///                     empty.
        /// @invariant Mints the key into templateInterner / nameMap and the
        ///            record into valueInterner; opens no scope.
        /// @see openImplicationSubproof, revisitRejectedSubproof,
        ///      canonicalAdmissionText.
inline void ImplicationSubproofs::parkImplicationSubproof(const WorkInstruction& wi, int32_t entity,
            StrSpan gate, Memory& mb, StrSpan validityName, StrSpan rootGoal) {
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
            ScratchArena& arena = genScratchArenas().forSlot(slot);
            ScratchArena& strings = scratchArenas().forSlot(slot);
            ScratchScope frame(arena), textFrame(strings);
            const NameId scopeId = mb.nameMap.lookup(validityName);
            assert(scopeId != 0 && "parkImplicationSubproof: expected a registered scope");
            const ScratchString canonicalGate =
                canonicalTemplates.canonicalAdmissionText(mb, gate, scopeId, strings, true);
            WorkInstruction record(&arena);
            const auto copyEntity = [&](int32_t e) {
                const int32_t mark = record.elemMark();
                for (int32_t j = 0; j < wi.elemCount(e); ++j) record.addElement(wi.elemAt(e, j));
                record.commitEntity(wi.category(e), wi.signature(e), wi.definedSet(e),
                                    wi.arity(e), mark);
            };
            copyEntity(entity);
            const StrSpan signature = wi.signature(entity);
            for (int32_t p = 0; p < wi.entityCount(); ++p)
                if (isParentAndOf(wi, p, signature)) copyEntity(p);
            assert(record.entityCount() > 1 && "parkImplicationSubproof: a park needs a parent and");
            record.setMarkedGoal(rootGoal);
            const int64_t pk = mintTemplateKey(mb.templateInterner, mb.nameMap,
                StrSpan(canonicalGate), validityName);
            ArenaIntegrationMap parks(&arena, &mb.valueInterner);
            parks.buildFromCold(mb.overallHashMemory.rejectedMapSubproof, pk);
            const int32_t before = parks.entries.size();
            parks.findOrEmplaceWork(record, mb.valueInterner);
            if (parks.entries.size() != before)
                parks.writeToCold(mb.overallHashMemory.rejectedMapSubproof, pk);
        }

/// @brief Open the boundary subproof of one implication entity of an
        ///        integration instruction — Case A of `prepareIntegrationCore2`.
        ///
        /// @details
        /// The one opening path of an implication element. In order:
        /// 1. Known part: the clean (u_-stripped) signature that is already a
        ///    known statement at the preparing scope, up to the scope's
        ///    complete classes (I-254), needs no subproof under any goal.
        /// 2. Prepared gate: the scope payload — `<rootGoal>_subproof_<clean>`
        ///    on main when a root goal drives the integration, the bare clean
        ///    signature otherwise — that is already prepared at the scope
        ///    (canonically, through the prepared-template index) is not
        ///    opened again. For a @p continuation a preparation at a
        ///    strict ancestor counts too (`canonicalTemplatePresentAtOrAbove`,
        ///    D-572).
        /// Equal-input refusal applies only when the owning block has its
        /// induction-step hypothesis installed (recursionAuxyImplId != 0).
        /// A direct positive equality premise permits its exact input pair.
        /// Built-on-listed witness refusal (I-293) also precedes the gate.
        /// 3. Gate (D-406): for every and of @p wi
        ///    that lists the implication, each gate element
        ///    (`isSubproofGateElement`) must be known at the scope or an
        ///    ancestor (`elementKnownAtOrAbove`). The first unknown one, in
        ///    instruction order, parks the implication in
        ///    `rejectedMapSubproof` (`parkImplicationSubproof`); the drain of
        ///    known-row arrivals reopens it through this function.
        /// 4. Opening: mint the prepared key, push the scope
        ///    `<validityName>_boundary_<payload>`, record the
        ///    expansion-for-integration origin, then add every renamed premise
        ///    at the new scope as status 0 with its premise-element origin and
        ///    the renamed head as status 2.
        /// 5. Reductio: a bare numbered implication (no embedded goal)
        ///    outside a primed contradiction block also gets its reductio
        ///    beside the subproof, `contradiction_!(<compact>)` at the same
        ///    scope (`prepareNegatedCompoundContradiction`); its discharge
        ///    writes the negated compact there, and
        ///    `retireSettledImplicationSubproofs` retires the subproof
        ///    (D-532).
        ///
        /// A goal embedded in the payload lets a later disproof of that goal
        /// enumerate and wipe the scope (`splitSubproofPayload`); deeper
        /// scopes need no embedding because they ride their main-level
        /// ancestor's subtree wipe.
        ///
        /// @param wi            Rewritten integration instruction (replacement
        ///                      map and canonical admission text already
        ///                      applied); read only.
        /// @param entity        Index of the implication entity in @p wi:
        ///                      category `implication`, every signature
        ///                      argument `u_`-prefixed.
        /// @param mb            Owning block; receives the prepared key, the
        ///                      scope and its seeded statements.
        /// @param validityName  Registered preparing scope.
        /// @param rootGoal      The goal whose integration drives this call
        ///                      (verbatim), or empty for statement-driven
        ///                      preparation.
        /// @param continuation  Whether the call replays a stored instruction
        ///                      after a marker fill
        ///                      (`IntegrationProcessor::prepareIntegrationCore2`).
        /// @return `Known`, `Prepared`, `Refused`, `Parked` or `Opened` (see
        ///         `ImplicationSubproofOutcome`).
        /// @invariant Asserts the entity contract of @p entity; every string
        ///            it builds rides its own string-scratch scope, rewound on
        ///            return.
        /// @see prepareIntegrationCore2 and revisitRejectedSubproof (the
        ///      callers), compactKnownUpToEquality,
        ///      canonicalTemplatePresent, renamingChain2Scratch,
        ///      expandSignatureForIntegrationScratch,
        ///      prepareNegatedCompoundContradiction.
inline ImplicationSubproofOutcome ImplicationSubproofs::openImplicationSubproof(const WorkInstruction& wi,
            int32_t entity, Memory& mb, StrSpan validityName, StrSpan rootGoal, bool continuation) {
            assert(entity >= 0 && entity < wi.entityCount()
                && "openImplicationSubproof: entity index out of range");
            assert(equalSpans(wi.category(entity), StrSpan("implication", 11))
                && "openImplicationSubproof: entity is not an implication");
            assert(!gl::ScopeSyntax::atOrBelowRestrictedScope(mb.nameMap, mb.nameMap.lookup(validityName))
                && "openImplicationSubproof: no subproof opens inside a restricted scope");
            const StrSpan leSignatureSpan = wi.signature(entity);
            {
                StrSpan sigArgs[ExecutionParameters::MAX_ARITY];
                const int32_t sigArgN = getArgsSpans(leSignatureSpan, sigArgs,
                                                     ExecutionParameters::MAX_ARITY);
                for (int32_t a = 0; a < sigArgN; ++a)
                    assert(sigArgs[a].len >= 2 && sigArgs[a].ptr[0] == 'u' && sigArgs[a].ptr[1] == '_'
                        && "openImplicationSubproof: every signature argument must be u_-prefixed");
            }
            StrSpan leElements[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
            int32_t leElemN = 0;
            {
                const int32_t ec = wi.elemCount(entity);
                assert(ec <= ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
                    && "openImplicationSubproof: element count exceeds MAX_INSTRUCTION_ELEMENTS");
                for (int32_t j = 0; j < ec; ++j)
                    leElements[leElemN++] = wi.elemAt(entity, j);
            }
            const bool embedGoal =
                equalSpans(validityName, StrSpan("main", 4)) && rootGoal.len > 0;
            const unsigned coreSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : scratchArenas().slotCount() - 1;
            ScratchArena& coreArena = scratchArenas().forSlot(coreSlot);
            ScratchScope openScope(coreArena);

            const ScratchString cleanSignatureS =
                gl::DefinitionExpansion::removeUPrefixScratch(coreArena, leSignatureSpan);
            // Goal-carrying payload on MAIN: the scope payload AND the
            // prep-gate template are `<rootGoal>_subproof_<compact>`, so
            // (a) a disproof of rootGoal can enumerate this scope by
            // payload (splitSubproofPayload) and (b) a second goal
            // needing the same compact still preps its own subproof
            // after this goal's wipe (per-goal duplication; gates
            // goal-qualified in step). The known-part check below stays
            // on the BARE compact — a compact already known as a
            // statement at this scope, up to the scope's complete classes
            // (I-254), needs no subproof under ANY goal.
            ScratchString scopePayloadS;
            if (embedGoal) {
                const StrSpan SUBPROOF("_subproof_", 10);
                const StrSpan clean(cleanSignatureS);
                const int32_t total = rootGoal.len + SUBPROOF.len + clean.len;
                char* b = coreArena.allocBytes(total);
                std::memcpy(b, rootGoal.ptr, static_cast<size_t>(rootGoal.len));
                std::memcpy(b + rootGoal.len, SUBPROOF.ptr, static_cast<size_t>(SUBPROOF.len));
                std::memcpy(b + rootGoal.len + SUBPROOF.len, clean.ptr,
                            static_cast<size_t>(clean.len));
                scopePayloadS = ScratchString::wrap(coreArena, b, total);
            }
            const StrSpan scopePayload = embedGoal
                ? StrSpan(scopePayloadS) : StrSpan(cleanSignatureS);

            if (equivalenceQueries.compactKnownUpToEquality(StrSpan(cleanSignatureS), StrSpan(validityName), mb))
                return ImplicationSubproofOutcome::Known;

            if (continuation
                    ? canonicalTemplates.canonicalTemplatePresentAtOrAbove(mb, mb.integrationPrepared,
                        mb.lbMemory.canonicalUses.preparedMatches, scopePayload, StrSpan(validityName))
                    : canonicalTemplates.canonicalTemplatePresent(mb, mb.integrationPrepared,
                        mb.lbMemory.canonicalUses.preparedMatches, scopePayload, StrSpan(validityName),
                        mb.nameMap.lookup(validityName))) {
                return ImplicationSubproofOutcome::Prepared;
            }

            // Distinct formal inputs may coincide only through a direct equality
            // premise of this implication (I-325).
            if (mb.recursionAuxyImplId != 0
                && startsWithSpan(StrSpan(cleanSignatureS), "(implication", 12)
                && hasForbiddenEqualInputs(StrSpan(cleanSignatureS), mb.nameMap.encode(validityName), mb)) {
                RT_SCOPE_HERE("REFUSE_EQUAL_INPUT_SUBPROOF");
                return ImplicationSubproofOutcome::Refused;
            }

            // Built-on-listed refusal (I-293): an
            // implication naming a construction this recursion block built
            // over a witness of the hypothesis' conclusion or over
            // constructed objects is never opened, prepared or parked. An
            // argument whose complete class holds an unlisted member names
            // that member's object and refuses nothing.
            if (hasBuiltOnListedArgument(StrSpan(cleanSignatureS), mb.nameMap.lookup(validityName), mb))
                return ImplicationSubproofOutcome::Refused;

            // Gate (D-406): an implication listed by
            // an and waits until every gate element of that and is known at
            // the scope or an ancestor. The first unknown one, in instruction
            // order, keys the park; nothing is prepared or opened.
            {
                const NameId scopeId = mb.nameMap.lookup(validityName);
                assert(scopeId != 0 && "openImplicationSubproof: expected a registered scope");
                for (int32_t p = 0; p < wi.entityCount(); ++p) {
                    if (!isParentAndOf(wi, p, leSignatureSpan)) continue;
                    for (int32_t j = 0; j < wi.elemCount(p); ++j) {
                        const StrSpan element = wi.elemAt(p, j);
                        if (!isSubproofGateElement(element, definitions)) continue;
                        const ScratchString gateS = gl::DefinitionExpansion::removeUPrefixScratch(coreArena, element);
                        if (equivalenceQueries.elementKnownAtOrAbove(StrSpan(gateS), scopeId, mb)) continue;
                        parkImplicationSubproof(wi, entity, StrSpan(gateS), mb, validityName,
                                                embedGoal ? rootGoal : StrSpan());
                        return ImplicationSubproofOutcome::Parked;
                    }
                }
            }

            const int64_t gateKey = mintTemplateKey(
                mb.templateInterner, mb.nameMap, scopePayload, StrSpan(validityName));
            mb.integrationPrepared.mint(gateKey);

            NameId parentValId = mb.nameMap.encode(validityName);
            NameId scopeValId  = mb.nameMap.encodePush(parentValId, scopePayload);
            canonicalizer.initializeCanonicalScope(mb, scopeValId);
            // copyFrom onto coreArena (a DIFFERENT arena than NameMap):
            // nested addExprToMemoryBlock calls below mint NameMap, so a raw
            // decodeView span would dangle (I-3); the copy survives every
            // mint and rides below every nested scope's mark (09c pitfall 4).
            const ScratchString scopeValidity = ScratchString::copyFrom(
                coreArena, mb.nameMap.decodeView(scopeValId).ptr,
                mb.nameMap.decodeView(scopeValId).len);

            StrSpan renamedChain[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
            const int32_t renamedChainN = integrationBuilder.renamingChain2Scratch(
                coreArena, leElements, leElemN, mb,
                renamedChain, ExecutionParameters::MAX_INSTRUCTION_ELEMENTS);

            const ScratchString expandedImplicationS = integrationBuilder.expandSignatureForIntegrationScratch(
                coreArena, StrSpan("implication", 11), renamedChain, renamedChainN,
                StrSpan(cleanSignatureS));
            const StrSpan expandedImplication(expandedImplicationS);
            if (parameters.trackHistory)
            {
                // L3 span-record doors. The "expansion for integration"
                // KEY (expandedImplication) and antecedent
                // (cleanSignature) both take the "_integration_goal"
                // suffix; build them on the per-slot string-scratch arena
                // (explicit length — never a bare-array StrSpan).
                // validityName is a stable local; the doors mint into
                // originInterner, never this arena (I-3).
                const unsigned gSlot = (gl::WorkerContext::coreId() >= 0)
                    ? static_cast<unsigned>(gl::WorkerContext::coreId())
                    : scratchArenas().slotCount() - 1;
                ScratchArena& gArena = scratchArenas().forSlot(gSlot);
                ScratchScope gScope(gArena);
                const StrSpan GOAL("_integration_goal",
                    static_cast<int32_t>(sizeof("_integration_goal") - 1));
                const auto goalOf = [&](const StrSpan base) -> StrSpan {
                    const int32_t len = base.len + GOAL.len;
                    char* buf = gArena.allocBytes(len);
                    std::memcpy(buf, base.ptr,
                                static_cast<std::size_t>(base.len));
                    std::memcpy(buf + base.len, GOAL.ptr,
                                static_cast<std::size_t>(GOAL.len));
                    return StrSpan(buf, len);
                };
                const StrSpan keySpan = goalOf(expandedImplication);
                const OriginDep ieDeps[1] = {
                    { goalOf(StrSpan(cleanSignatureS)), StrSpan(validityName) } };
                const int ieCap = (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr);
                // The span-record doors replace the if() check, handling
                // both the max origin limits and preventing duplicates.
                addOriginEncoded(mb.exprOriginMap, mb.originInterner, keySpan, StrSpan(validityName), OriginTag::expansionForIntegration, ieDeps, 1, ieCap);
                mb.addMailOutOrigin(keySpan, StrSpan(validityName), OriginTag::expansionForIntegration, ieDeps, 1, ieCap);
            }

            for (int32_t i = 0; i < renamedChainN - 1; ++i) {
                const StrSpan elem = renamedChain[i];

                const unsigned peSlot = (gl::WorkerContext::coreId() >= 0)
                    ? static_cast<unsigned>(gl::WorkerContext::coreId())
                    : scratchArenas().slotCount() - 1;
                ScratchArena& peArena = scratchArenas().forSlot(peSlot);
                ScratchScope peScope(peArena);

                OriginDep peDeps[1];
                TransientOrigin originPremiseElement{};
                if (parameters.trackHistory)
                {
                    // antecedent = expandedImplication + "_integration_goal",
                    // built on the per-slot scratch arena (explicit length).
                    // The scope spans the addExprToMemoryBlock pass-through
                    // below, whose span door mints the antecedent into
                    // originInterner, never this arena (I-3).
                    const StrSpan PE_GOAL("_integration_goal",
                        static_cast<int32_t>(sizeof("_integration_goal") - 1));
                    const int32_t antLen = expandedImplication.len + PE_GOAL.len;
                    char* antBuf = peArena.allocBytes(antLen);
                    std::memcpy(antBuf, expandedImplication.ptr,
                                static_cast<std::size_t>(expandedImplication.len));
                    std::memcpy(antBuf + expandedImplication.len,
                                PE_GOAL.ptr, static_cast<std::size_t>(PE_GOAL.len));
                    peDeps[0] = { StrSpan(antBuf, antLen), StrSpan(validityName) };
                    originPremiseElement =
                        TransientOrigin{ true, OriginTag::premiseElement, peDeps, 1 };

                    const int peCap = (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr);
                    addOriginEncoded(mb.exprOriginMap, mb.originInterner, elem, StrSpan(scopeValidity), OriginTag::premiseElement, peDeps, 1, peCap);
                    mb.addMailOutOrigin(elem, StrSpan(scopeValidity), OriginTag::premiseElement, peDeps, 1, peCap);
                }

                // An integration premise: an OR compact opens its case split,
                // a negated conjunction splits into its complemented
                // conjuncts (D-460).
                statementMemory.addExprToMemoryBlock(elem, mb, -1, 0, nullptr, 0, originPremiseElement, -1, -1, StrSpan(scopeValidity), false,
                    /*allowOrDisintegration=*/false, /*registeredOut=*/nullptr,
                    /*recursionHead=*/false, /*convergenceMailOut=*/nullptr,
                    /*replayKinds=*/0, /*replayIntegrationDepth=*/-1,
                    /*integrationPremise=*/true);
            }

            statementMemory.addExprToMemoryBlock(renamedChain[renamedChainN - 1], mb, -1, 2, nullptr, 0, TransientOrigin{}, -1, -1, StrSpan(scopeValidity), false);
            // The subproof's own goal, as inserted (a goal is never respelled
            // on insertion): a later class rewrite of its toBeProved entry
            // closes the subproof in another spelling, and the closure cites
            // this one (D-557).
            mb.subproofGoals.upsert(static_cast<int64_t>(scopeValId),
                static_cast<int64_t>(mb.nameMap.encode(renamedChain[renamedChainN - 1])));

            // Reductio beside a bare numbered subproof: the implication,
            // assumed in a contradiction scope, fires on the parent's facts;
            // a counterexample writes its negation at the parent, which
            // retires the subproof (D-532). A
            // primed contradiction block is meant to be inconsistent there.
            if (!embedGoal && !mb.primedForContradiction
                && startsWithSpan(StrSpan(cleanSignatureS), "(implication", 12)) {
                RT_SCOPE_HERE("PRIME_SUBPROOF_REDUCTIO");
                const StrSpan clean(cleanSignatureS);
                char* negated = coreArena.allocBytes(clean.len + 1);
                negated[0] = '!';
                std::memcpy(negated + 1, clean.ptr, static_cast<size_t>(clean.len));
                prepareNegatedCompoundContradiction(StrSpan(negated, clean.len + 1), mb, validityName);
            }
            return ImplicationSubproofOutcome::Opened;
        }

/// @brief Contradiction-based integration of a negated compound
        ///        compact — mint a `contradiction_`-marked child scope and seed
        ///        it with the positive compact.
        ///
        /// @details
        /// The GL-native proof of a negated compound is reductio: assume the
        /// positive compact `(X[args])` as a status-0 statement in a
        /// dedicated child scope whose payload is `contradiction_!(X[args])`,
        /// pushed on @p validityName. The standard consumption machinery
        /// decomposes the seed (existence-bodied → witness facts, AND-bodied
        /// → conjuncts, implication-bodied → installed rule) and derives
        /// forward; when a contradiction fires at the scope,
        /// `dischargeContradictionScopes` emits the negated compact at the
        /// scope's parent and wipes the scope. No template is instantiated
        /// for the negated form — the polarity travels with the expression.
        /// Or-bodied compacts are never passed (all-branch refutation
        /// semantics; see the intercept comment).
        ///
        /// Two callers: the negated-compound intercept of
        /// `IntegrationProcessor::prepareIntegration` (a negated compound
        /// to be proved) and `openImplicationSubproof`, which primes the
        /// reductio of every bare numbered implication subproof it opens
        /// outside a primed contradiction block: a counterexample at the
        /// parent then writes the negated compact there, and
        /// `retireSettledImplicationSubproofs` retires the subproof
        /// (D-532).
        ///
        /// Idempotence gates, in order: the product already `registered` at
        /// @p validityName or at `"main"` (nothing to prove); the scope
        /// payload already minted in `integrationPrepared` at
        /// @p validityName (the scope exists or was already discharged).
        ///
        /// The seed's history row is `task formulation` (empty antecedents)
        /// at the contradiction scope — the primed-`__contradiction__` seed
        /// contract. The mail-side origin mirror rides `addMailOutOrigin`
        /// (origins are all-scopes per I-26); the statement itself is NOT
        /// mailed (statements are main-only per I-26).
        ///
        /// @param expression   The negated compact `!(X[args])` (span over a
        ///                     caller-stable buffer).
        /// @param mb           The owning LB.
        /// @param validityName The parent scope for the contradiction scope.
        /// @invariant Heap-free; interner mints only behind the non-minting
        ///            probes; single-threaded call context (admission seeding
        ///            seams).
        /// @see IntegrationProcessor::prepareIntegration and
        ///      openImplicationSubproof — the callers;
        ///      dischargeContradictionScopes — the discharge half;
        ///      contradictionScopeProductView — the payload parser;
        ///      retireSubproofReductios — the reductio's end with its
        ///      subproof.
inline void ImplicationSubproofs::prepareNegatedCompoundContradiction(StrSpan expression,
            Memory& mb,
            StrSpan validityName) {
            assert(expression.len >= 2 && expression.ptr[0] == '!'
                && expression.ptr[1] == '('
                && "prepareNegatedCompoundContradiction: input must be a negated compact");

            // Product already proven — at this scope or at main.
            {
                const StatementFlags* row = lookupStatementFlags(
                    mb.intKnownStatements, mb.nameMap, expression, validityName);
                if (row != nullptr) return;
                const StatementFlags* rowMain = lookupStatementFlags(
                    mb.intKnownStatements, mb.nameMap, expression,
                    StrSpan("main", 4));
                if (rowMain != nullptr) return;
            }

            const unsigned pncSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : scratchArenas().slotCount() - 1;
            ScratchArena& sArena = scratchArenas().forSlot(pncSlot);
            ScratchScope pncScope(sArena);

            // payload = "contradiction_" + expression (fresh bytes).
            const int32_t pfxLen =
                static_cast<int32_t>(sizeof(gl::ScopeSyntax::kContradictionScopePrefix) - 1);
            char* pb = sArena.allocBytes(pfxLen + expression.len);
            std::memcpy(pb, gl::ScopeSyntax::kContradictionScopePrefix,
                        static_cast<size_t>(pfxLen));
            std::memcpy(pb + pfxLen, expression.ptr,
                        static_cast<size_t>(expression.len));
            const StrSpan payload(pb, pfxLen + expression.len);

            // Once per payload@scope.
            if (canonicalTemplates.canonicalTemplatePresent(mb, mb.integrationPrepared, mb.lbMemory.canonicalUses.preparedMatches,
                payload, validityName, mb.nameMap.lookup(validityName))) {
                return;
            }
            const int64_t gateKey = mintTemplateKey(
                mb.templateInterner, mb.nameMap, payload, validityName);
            mb.integrationPrepared.mint(gateKey);

            NameId parentValId = mb.nameMap.encode(validityName);
            NameId scopeValId = mb.nameMap.encodePush(parentValId, payload);
            canonicalizer.initializeCanonicalScope(mb, scopeValId);
            // The scope looks for its own contradiction: the demand drain
            // matches the facts visible here against the rule heads that
            // negate them (D-560).
            mb.lbMemory.contradictionScopes.insert(static_cast<int64_t>(scopeValId), 0);
            // copyFrom (I-3): the seed deposit below mints NameMap, so a raw
            // decodeView span would dangle.
            const ScratchString scopeValidity = ScratchString::copyFrom(
                sArena, mb.nameMap.decodeView(scopeValId).ptr,
                mb.nameMap.decodeView(scopeValId).len);

            // Seed the POSITIVE compact (status 0) — standard existence
            // consumption decomposes it into witness facts inside the scope.
            const StrSpan positive(expression.ptr + 1, expression.len - 1);
            if (parameters.trackHistory) {
                const int cap = (parameters.compressor_mode
                    ? parameters.compressor_max_origins_per_expr
                    : parameters.max_origin_per_expr);
                addOriginEncoded(mb.exprOriginMap, mb.originInterner,
                    positive, StrSpan(scopeValidity),
                    OriginTag::taskFormulation, nullptr, 0, cap);
                mb.addMailOutOrigin(positive, StrSpan(scopeValidity),
                    OriginTag::taskFormulation, nullptr, 0, cap);
            }
            const int seedLevels[1] = { mb.level };
            statementMemory.addExprToMemoryBlock(positive, mb, -1, 0, seedLevels, 1,
                TransientOrigin{ true, OriginTag::taskFormulation, nullptr, 0 },
                -1, -1, StrSpan(scopeValidity), false);
        }

/// @brief Park the reductio of a negated hypothesis premise under its first
///        positive premise not yet known, or report that every positive
///        premise is known.
///
/// @details
/// The gates are the hypothesis's fully bound positive premises
/// (`IntegrationProcessor::prepareHypothesisPremise` selects them). The
/// first gate not known at @p validityName or an ancestor
/// (`elementKnownAtOrAbove`, up to the scope's classes) keys the park: the
/// key is that gate's text normalized through the scope's complete classes
/// (`canonicalAdmissionText`), as for an implication park, and the record
/// is one entity of category `reductio` whose signature is @p negated and
/// whose elements are all gates. The record is filed in
/// `rejectedMapSubproof` through the canonical read-modify-write, which
/// deduplicates a repeated park; the arrival drain revives it
/// (`revisitRejectedSubproof`), which re-runs this check and opens the
/// reductio once every gate is known
/// (I-326).
///
/// @param negated      The negated premise `!(X[args])` (span over a
///                     caller-stable buffer).
/// @param gates        The positive gates (spans over caller-stable
///                     buffers).
/// @param gateN        Their number; zero means nothing to wait for.
/// @param mb           Owning block.
/// @param validityName Registered preparing scope (the park scope).
/// @return True when the reductio parked; false when every gate is known.
/// @invariant Opens no scope; mints only the park key and its record.
/// @see prepareNegatedCompoundContradiction, revisitRejectedSubproof,
///      IntegrationProcessor::prepareHypothesisPremise.
inline bool ImplicationSubproofs::deferPremiseReductio(StrSpan negated, const StrSpan* gates,
            int32_t gateN, Memory& mb, StrSpan validityName) {
            assert(negated.len >= 2 && negated.ptr[0] == '!' && negated.ptr[1] == '('
                && "deferPremiseReductio: input must be a negated compact");
            const NameId scopeId = mb.nameMap.lookup(validityName);
            assert(scopeId != 0 && "deferPremiseReductio: expected a registered scope");
            int32_t unknown = 0;
            while (unknown < gateN
                && equivalenceQueries.elementKnownAtOrAbove(gates[unknown], scopeId, mb))
                ++unknown;
            if (unknown == gateN) return false;

            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
            ScratchArena& arena = genScratchArenas().forSlot(slot);
            ScratchArena& strings = scratchArenas().forSlot(slot);
            ScratchScope frame(arena), textFrame(strings);
            const ScratchString canonicalGate =
                canonicalTemplates.canonicalAdmissionText(mb, gates[unknown], scopeId, strings, true);
            WorkInstruction record(&arena);
            const int32_t mark = record.elemMark();
            for (int32_t g = 0; g < gateN; ++g) record.addElement(gates[g]);
            record.commitEntity(StrSpan(kReductioParkCategory,
                                    static_cast<int32_t>(std::strlen(kReductioParkCategory))),
                                negated, StrSpan("", 0), 0, mark);
            record.setMarkedGoal(StrSpan("", 0));
            const int64_t pk = mintTemplateKey(mb.templateInterner, mb.nameMap,
                StrSpan(canonicalGate), validityName);
            ArenaIntegrationMap parks(&arena, &mb.valueInterner);
            parks.buildFromCold(mb.overallHashMemory.rejectedMapSubproof, pk);
            const int32_t before = parks.entries.size();
            parks.findOrEmplaceWork(record, mb.valueInterner);
            if (parks.entries.size() != before)
                parks.writeToCold(mb.overallHashMemory.rejectedMapSubproof, pk);
            return true;
        }

/// @brief Whether a premise is a positive registry compound of category
///        `and` — the only premise a contradiction-scope demand integrates
///        (D-560).
///
/// @details
/// A negated premise, an operator the registry does not compile and an
/// entity of any other category (atomic, existence, or, implication) is not
/// demandable: only an and carries the integration rules and gate elements
/// the demand relies on.
///
/// @param premise Premise text, clean or `u_`-prefixed; only the sign and the
///                operator name are read.
/// @return True iff @p premise is positive and its operator compiles to an
///         entity of category `and`.
/// @invariant Pure; allocates nothing.
/// @see deferContradictionDemand, compoundGateElements.
inline bool ImplicationSubproofs::isDemandableCompound(StrSpan premise) const {
            if (premise.len == 0 || premise.ptr[0] != '(') return false;
            const LogicalEntity* entity = definitions.compiledEntity(extractExpressionUniversalSpan(premise));
            return entity != nullptr && entity->category == "and";
        }

/// @brief The gate elements of an and compound instance, as the clean
///        statements they wait for (D-560).
///
/// @details
/// The compiled entity's elements name its signature tokens. Each token is
/// replaced by `u_` and the instance's argument at the same position; an
/// element that then passes `isSubproofGateElement` is a gate, and its clean
/// form (the `u_` prefixes removed) is the statement it waits for — the gate
/// elements the and's integration instruction carries (D-406).
///
/// @param compound Clean instance `(X[args])` of an and entity
///                 (`isDemandableCompound`, asserted).
/// @param strings  String-scratch arena receiving the texts; the caller's
///                 frame keeps them.
/// @param gates    Receives the gate statements.
/// @param cap      Capacity of @p gates; more gates assert.
/// @return The number of gate elements.
/// @invariant Zero heap; mints nothing.
/// @see isSubproofGateElement, deferContradictionDemand.
inline int32_t ImplicationSubproofs::compoundGateElements(StrSpan compound, ScratchArena& strings,
            StrSpan* gates, int32_t cap) const {
            assert(isDemandableCompound(compound)
                && "compoundGateElements: expected an instance of an and entity");
            const LogicalEntity* entity = definitions.compiledEntity(extractExpressionUniversalSpan(compound));
            StrSpan signatureArgs[ExecutionParameters::MAX_ARITY];
            const int32_t signatureN = getArgsSpans(StrSpan(entity->signature), signatureArgs,
                ExecutionParameters::MAX_ARITY);
            StrSpan instanceArgs[ExecutionParameters::MAX_ARITY];
            const int32_t instanceN = getArgsSpans(compound, instanceArgs, ExecutionParameters::MAX_ARITY);
            assert(signatureN == instanceN
                && "compoundGateElements: the instance's arity differs from its definition");
            // Each signature token becomes the instance's argument, `u_`-prefixed
            // as in the rewritten integration instruction.
            StrReplacement rename[ExecutionParameters::MAX_ARITY];
            for (int32_t a = 0; a < signatureN; ++a) {
                const int32_t len = 2 + instanceArgs[a].len;
                char* b = strings.allocBytes(len);
                b[0] = 'u';
                b[1] = '_';
                std::memcpy(b + 2, instanceArgs[a].ptr, static_cast<size_t>(instanceArgs[a].len));
                rename[a] = { signatureArgs[a], StrSpan(b, len) };
            }
            int32_t gateN = 0;
            for (const std::string& element : entity->elements) {
                const ScratchString renamed = replaceKeysScratch(strings, StrSpan(element), rename, signatureN);
                if (!isSubproofGateElement(StrSpan(renamed), definitions)) continue;
                assert(gateN < cap && "compoundGateElements: more gate elements than the caller's capacity");
                gates[gateN++] = StrSpan(gl::DefinitionExpansion::removeUPrefixScratch(strings, StrSpan(renamed)));
            }
            return gateN;
        }

/// @brief Decide one contradiction-scope demand candidate: demand its one
///        unknown compound, park it under what it waits for, or drop it
///        (D-560).
///
/// @details
/// The candidate is a rule whose head negates a fact visible at the
/// contradiction scope @p scopeId, every premise fully bound by that match
/// (@p premises, clean, in chain order). In order:
/// 1. Every premise known at the scope or an ancestor
///    (`elementKnownAtOrAbove`, up to the scope's classes): dropped, since the
///    rule fires without help.
/// 2. Exactly one premise unknown and it is a demandable compound
///    (`isDemandableCompound`): the gate. The first gate element
///    (`compoundGateElements`) not known at the scope or an ancestor parks
///    the candidate. With every gate element known, at least one must be
///    known at the scope itself (`compactKnownUpToEquality`, through the
///    scope's classes); otherwise, and for a compound without gate elements,
///    the candidate is dropped: a compound whose gate holds only above the
///    scope cannot become this scope's contradiction. Otherwise the candidate
///    is ready and @p readyIndex names the compound.
/// 3. Otherwise the candidate parks under its first unknown premise that is
///    not a demandable compound, the premise ordinary derivation must supply,
///    or under the first unknown premise when every unknown one is a
///    compound.
/// A park files one record of category `contradiction_demand` whose elements
/// are the premises, keyed by the awaited statement normalized through the
/// scope's complete classes (`canonicalAdmissionText`), through the canonical
/// read-modify-write that deduplicates a repeated park; the arrival drain's
/// revival (`revisitRejectedSubproof`) calls this function again.
///
/// @param premises   Clean fully bound premises (spans over caller-stable
///                   buffers).
/// @param premiseN   Their number, at least one.
/// @param mb         Owning block.
/// @param scopeId    The contradiction scope.
/// @param readyIndex Receives the compound's index on a ready verdict, else
///                   `-1`.
/// @return The verdict.
/// @invariant Opens no scope; a park mints only its key and its record.
/// @see isDemandableCompound, compoundGateElements, revisitRejectedSubproof,
///      ContradictionDemand::drainContradictionDemands.
inline ImplicationSubproofs::ContradictionDemandVerdict ImplicationSubproofs::deferContradictionDemand(
            const StrSpan* premises, int32_t premiseN, Memory& mb, NameId scopeId, int32_t& readyIndex) {
            assert(premiseN >= 1 && premiseN <= ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                && "deferContradictionDemand: a candidate has one to MAX_ADMISSION_KEY_ELEMENTS premises");
            assert(scopeId != 0 && scopeId != NameMap::MAIN_ID
                && "deferContradictionDemand: expected a contradiction scope");
            readyIndex = -1;
            int32_t unknown[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
            int32_t unknownN = 0;
            for (int32_t p = 0; p < premiseN; ++p)
                if (!equivalenceQueries.elementKnownAtOrAbove(premises[p], scopeId, mb)) unknown[unknownN++] = p;
            if (unknownN == 0) return ContradictionDemandVerdict::Dropped;

            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
            ScratchArena& arena = genScratchArenas().forSlot(slot);
            ScratchArena& strings = scratchArenas().forSlot(slot);
            ScratchScope frame(arena), textFrame(strings);
            // copyFrom (I-3): the park below mints, and the scope text is read
            // across it.
            const StrSpan scopeView = mb.nameMap.decodeView(scopeId);
            const ScratchString scope = ScratchString::copyFrom(strings, scopeView.ptr, scopeView.len);
            StrSpan awaited;
            if (unknownN == 1 && isDemandableCompound(premises[unknown[0]])) {
                StrSpan gates[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                const int32_t gateN = compoundGateElements(premises[unknown[0]], strings, gates,
                    ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS);
                if (gateN == 0) return ContradictionDemandVerdict::Dropped;
                int32_t missing = 0;
                while (missing < gateN
                    && equivalenceQueries.elementKnownAtOrAbove(gates[missing], scopeId, mb))
                    ++missing;
                if (missing == gateN) {
                    bool atScope = false;
                    for (int32_t g = 0; g < gateN && !atScope; ++g)
                        atScope = equivalenceQueries.compactKnownUpToEquality(gates[g], StrSpan(scope), mb);
                    if (!atScope) return ContradictionDemandVerdict::Dropped;
                    readyIndex = unknown[0];
                    return ContradictionDemandVerdict::Ready;
                }
                awaited = gates[missing];
            } else {
                int32_t wait = unknown[0];
                for (int32_t u = 0; u < unknownN; ++u) {
                    if (isDemandableCompound(premises[unknown[u]])) continue;
                    wait = unknown[u];
                    break;
                }
                awaited = premises[wait];
            }

            const ScratchString canonical =
                canonicalTemplates.canonicalAdmissionText(mb, awaited, scopeId, strings, true);
            WorkInstruction record(&arena);
            const int32_t mark = record.elemMark();
            for (int32_t p = 0; p < premiseN; ++p) record.addElement(premises[p]);
            record.commitEntity(StrSpan(kContradictionDemandParkCategory,
                                    static_cast<int32_t>(std::strlen(kContradictionDemandParkCategory))),
                                StrSpan("", 0), StrSpan("", 0), 0, mark);
            record.setMarkedGoal(StrSpan("", 0));
            const int64_t pk = mintTemplateKey(mb.templateInterner, mb.nameMap,
                StrSpan(canonical), StrSpan(scope));
            ArenaIntegrationMap parks(&arena, &mb.valueInterner);
            parks.buildFromCold(mb.overallHashMemory.rejectedMapSubproof, pk);
            const int32_t before = parks.entries.size();
            parks.findOrEmplaceWork(record, mb.valueInterner);
            if (parks.entries.size() != before)
                parks.writeToCold(mb.overallHashMemory.rejectedMapSubproof, pk);
            return ContradictionDemandVerdict::Parked;
        }

/// @brief Stage a ready contradiction-scope demand for the demand drain
///        (D-560).
///
/// @details
/// The compound is interned in the block's template interner, the id space of
/// admission and park templates, so no statement id is minted for it; the
/// drain integrates it at the scope.
///
/// @param mb       Owning block.
/// @param compound Clean compound text.
/// @param scopeId  The contradiction scope.
/// @return Nothing.
/// @invariant Appends to `pendingContradictionDemands` only.
/// @see deferContradictionDemand, ContradictionDemand::drainContradictionDemands.
inline void ImplicationSubproofs::stageContradictionDemand(Memory& mb, StrSpan compound, NameId scopeId) {
            assert(scopeId != 0 && scopeId != NameMap::MAIN_ID
                && "stageContradictionDemand: expected a contradiction scope");
            mb.pendingContradictionDemands.push_back(
                packStatementKey(mb.templateInterner.encode(compound), scopeId));
        }



/// @brief Move one park to its canonical gate key
        ///        (D-406).
        ///
        /// @details
        /// The key's clean gate text is normalized through the complete
        /// classes of its park scope; an unchanged text keeps the key.
        /// Otherwise the key's records move, verbatim, to the canonical key:
        /// they are read in decoded order, the old key is erased, and each
        /// record is merged into the canonical key's run through the
        /// read-modify-write, which deduplicates records already parked
        /// there. Only the key moves; a record keeps the instruction's own
        /// text, which the and's integration rule expects.
        ///
        /// @param key Packed `(gate template, park scope)` key with records.
        /// @param mb  Owning block.
        /// @return The key that now holds the records (== @p key when it was
        ///         already canonical).
        /// @invariant Records are never rewritten.
        /// @see parkImplicationSubproof, drainSubproofArrivals.
inline int64_t ImplicationSubproofs::rekeySubproofPark(int64_t key, Memory& mb) {
            auto& parks = mb.overallHashMemory.rejectedMapSubproof;
            assert(parks.lookup(key) != 0 && "rekeySubproofPark: key carries no park");
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
            ScratchArena& arena = genScratchArenas().forSlot(slot);
            ScratchArena& strings = scratchArenas().forSlot(slot);
            ScratchScope frame(arena), textFrame(strings);
            const StatementKey owner = Codec<StatementKey>::decode(key);
            const StrSpan gateView = mb.templateInterner.decodeView(owner.orig);
            const ScratchString gate = ScratchString::copyFrom(strings, gateView.ptr, gateView.len);
            const ScratchString canonical =
                canonicalTemplates.canonicalAdmissionText(mb, StrSpan(gate), owner.validity, strings, true);
            if (equalSpans(StrSpan(canonical), StrSpan(gate))) return key;
            const StrSpan scopeView = mb.nameMap.decodeView(owner.validity);
            const ScratchString scope = ScratchString::copyFrom(strings, scopeView.ptr, scopeView.len);
            ArenaIntegrationMap moved(&arena, &mb.valueInterner);
            moved.buildFromCold(parks, key);
            int32_t order[ExecutionParameters::MAX_INTEGRATION_ENTRIES];
            const int32_t n = moved.sortedIndices(order, ExecutionParameters::MAX_INTEGRATION_ENTRIES);
            parks.eraseBlobIf([key](int64_t k) { return k == key; });
            const int64_t target = mintTemplateKey(mb.templateInterner, mb.nameMap,
                StrSpan(canonical), StrSpan(scope));
            ArenaIntegrationMap merged(&arena, &mb.valueInterner);
            merged.buildFromCold(parks, target);
            for (int32_t i = 0; i < n; ++i) {
                WorkInstruction record(&arena);
                moved.loadInstructionInto(order[i], record, mb.valueInterner);
                // Two class-equal parks share one record.
                merged.findOrEmplaceWork(record, mb.valueInterner);
            }
            merged.writeToCold(parks, target);
            return target;
        }

/// @brief Erase the parks a disproved main goal owns
        ///        (D-406).
        ///
        /// @details
        /// A main-level park records the goal whose integration parked it.
        /// When that goal is disproved its machinery is dead: every record at
        /// a main key whose marked goal is @p goal is removed, a key left
        /// without records is erased, and the other keys are rewritten with
        /// their surviving records. Keys are visited in decoded order.
        ///
        /// @param mb   Owning block.
        /// @param goal Verbatim text of the disproved main goal.
        /// @invariant Touches only main-scope keys; other goals' records stay.
        /// @see drainDisprovedGoals, revisitRejectedSubproof.
inline void ImplicationSubproofs::eraseGoalSubproofParks(Memory& mb, StrSpan goal) {
            auto& parks = mb.overallHashMemory.rejectedMapSubproof;
            if (parks.empty()) return;
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
            ScratchArena& arena = genScratchArenas().forSlot(slot);
            ScratchScope frame(arena);
            DirtyState dirty = DirtyState::Clean;
            PagedVector<int64_t> keys(&arena, &dirty);
            for (int32_t id = 1; id <= parks.count(); ++id) {
                const int64_t key = parks.keyAt(id);
                if (parks.lookup(key) == 0) continue;
                if (Codec<StatementKey>::decode(key).validity == NameMap::MAIN_ID) keys.push_back(key);
            }
            const auto less = [&](int32_t a, int32_t b) {
                return compareSpans(mb.templateInterner.decodeView(Codec<StatementKey>::decode(keys[a]).orig),
                    mb.templateInterner.decodeView(Codec<StatementKey>::decode(keys[b]).orig)) < 0;
            };
            ChunkSortedOrdinals<decltype(less)> order(arena, keys.size(), 0, less);
            PagedVector<int64_t> ordered(&arena, &dirty);
            for (int32_t i = 0; i < keys.size(); ++i) ordered.push_back(keys[order.next()]);
            for (int32_t i = 0; i < ordered.size(); ++i) {
                const int64_t key = ordered[i];
                ArenaIntegrationMap current(&arena, &mb.valueInterner);
                current.buildFromCold(parks, key);
                ArenaIntegrationMap kept(&arena, &mb.valueInterner);
                bool dropped = false;
                for (int32_t e = 0; e < current.entries.size(); ++e) {
                    if (equalSpans(mb.valueInterner.decodeView(current.entries[e].markedGoalId), goal)) {
                        dropped = true;
                        continue;
                    }
                    WorkInstruction record(&arena);
                    current.loadInstructionInto(e, record, mb.valueInterner);
                    const int32_t before = kept.entries.size();
                    kept.findOrEmplaceWork(record, mb.valueInterner);
                    assert(kept.entries.size() != before
                        && "eraseGoalSubproofParks: a stored run holds a duplicate record");
                }
                if (!dropped) continue;
                if (kept.entries.size() == 0) parks.eraseBlobIf([key](int64_t k) { return k == key; });
                else kept.writeToCold(parks, key);
            }
        }

/// @brief Reopen the implications parked under one gate key once the
        ///        gate element is known (D-406).
        ///
        /// @details
        /// The revival of `rejectedMapSubproof`. A key without records (erased
        /// earlier in the same drain, or wiped with its scope) and a gate that
        /// is still not known at the park scope or an ancestor
        /// (`elementKnownAtOrAbove`) are defined no-ops; the park stands.
        /// Otherwise the records are snapshotted in decoded order and the key
        /// is erased. A parked negated-premise reductio (category `reductio`)
        /// opens its `contradiction_` scope once every positive premise it
        /// waits for is known, else parks under the next unknown one
        /// (`deferPremiseReductio`). A parked contradiction-scope demand
        /// candidate (category `contradiction_demand`) is decided again
        /// (`deferContradictionDemand`), and a ready one is staged for the
        /// demand drain (`stageContradictionDemand`). Each other record is
        /// then dropped when its root goal is no
        /// longer a main-level goal or when one of its parent ands is itself
        /// known; every other record is handed to `openImplicationSubproof`
        /// exactly as stored, which opens it or parks it again under its next
        /// unknown gate element. The record keeps the instruction's own text
        /// (the text the and's integration rule was installed with), so the
        /// opened subproof proves the implication the and waits for; the
        /// checks inside the opening compare through the current classes.
        ///
        /// @param parkKey Packed `(gate template, park scope)` key.
        /// @param mb      Owning block.
        /// @invariant Never re-entered for the same key (asserted through
        ///            `subproofRevisitInProgress`); every walk follows the
        ///            decoded record order (I-84).
        /// @see drainSubproofArrivals, openImplicationSubproof,
        ///      parkImplicationSubproof, deferPremiseReductio.
inline void ImplicationSubproofs::revisitRejectedSubproof(int64_t parkKey, Memory& mb) {
            auto& parks = mb.overallHashMemory.rejectedMapSubproof;
            if (parks.lookup(parkKey) == 0) return;
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
            ScratchArena& arena = genScratchArenas().forSlot(slot);
            ScratchArena& strings = scratchArenas().forSlot(slot);
            ScratchScope frame(arena), stringFrame(strings);
            const StatementKey key = Codec<StatementKey>::decode(parkKey);
            const StrSpan gateView = mb.templateInterner.decodeView(key.orig);
            const ScratchString gate = ScratchString::copyFrom(strings, gateView.ptr, gateView.len);
            if (!equivalenceQueries.elementKnownAtOrAbove(StrSpan(gate), key.validity, mb)) return;

            auto& guard = mb.overallHashMemory.subproofRevisitInProgress;
            assert(!guard.contains(parkKey) && "revisitRejectedSubproof: re-entered for one key");
            guard.mint(parkKey);
            const StrSpan scopeView = mb.nameMap.decodeView(key.validity);
            const ScratchString scope = ScratchString::copyFrom(strings, scopeView.ptr, scopeView.len);
            ArenaIntegrationMap snapshot(&arena, &mb.valueInterner);
            snapshot.buildFromCold(parks, parkKey);
            int32_t order[ExecutionParameters::MAX_INTEGRATION_ENTRIES];
            const int32_t n = snapshot.sortedIndices(order, ExecutionParameters::MAX_INTEGRATION_ENTRIES);
            parks.eraseBlobIf([parkKey](int64_t k) { return k == parkKey; });

            for (int32_t i = 0; i < n; ++i) {
                ScratchScope recordFrame(strings);
                WorkInstruction record(&arena);
                snapshot.loadInstructionInto(order[i], record, mb.valueInterner);
                if (equalSpans(record.category(0), StrSpan(kReductioParkCategory,
                        static_cast<int32_t>(std::strlen(kReductioParkCategory))))) {
                    // A parked negated-premise reductio: it opens once every
                    // positive premise is known, else parks under the next
                    // unknown one.
                    StrSpan gates[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                    const int32_t gateN = record.elemCount(0);
                    assert(gateN <= ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                        && "revisitRejectedSubproof: a reductio park with more gates than chain elements");
                    for (int32_t g = 0; g < gateN; ++g) gates[g] = record.elemAt(0, g);
                    if (!deferPremiseReductio(record.signature(0), gates, gateN, mb, StrSpan(scope)))
                        prepareNegatedCompoundContradiction(record.signature(0), mb, StrSpan(scope));
                    continue;
                }
                if (equalSpans(record.category(0), StrSpan(kContradictionDemandParkCategory,
                        static_cast<int32_t>(std::strlen(kContradictionDemandParkCategory))))) {
                    // A parked contradiction-scope demand candidate: decided
                    // again now that what it awaited is known; a ready one is
                    // staged for the demand drain
                    // (D-560).
                    StrSpan premises[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                    const int32_t premiseN = record.elemCount(0);
                    assert(premiseN <= ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                        && "revisitRejectedSubproof: a demand park with more premises than chain elements");
                    for (int32_t p = 0; p < premiseN; ++p) premises[p] = record.elemAt(0, p);
                    int32_t ready = -1;
                    if (deferContradictionDemand(premises, premiseN, mb, key.validity, ready)
                            == ContradictionDemandVerdict::Ready)
                        stageContradictionDemand(mb, premises[ready], key.validity);
                    continue;
                }
                const StrSpan rootGoal = record.markedGoal();
                if (rootGoal.len > 0) {
                    const NameId goalId = mb.nameMap.lookup(rootGoal);
                    if (goalId == 0 || mb.intToBeProved.lookup(
                            packStatementKey(goalId, NameMap::MAIN_ID)) == 0)
                        continue;
                }
                bool parentKnown = false;
                for (int32_t p = 1; p < record.entityCount() && !parentKnown; ++p) {
                    const ScratchString parent = gl::DefinitionExpansion::removeUPrefixScratch(strings, record.signature(p));
                    parentKnown = equivalenceQueries.elementKnownAtOrAbove(StrSpan(parent), key.validity, mb);
                }
                if (parentKnown) continue;
                (void)openImplicationSubproof(record, 0, mb, StrSpan(scope), rootGoal);
            }
            guard.erase(parkKey);
        }

/// @brief Match the staged known-row arrivals and class changes against
        ///        the parked gate keys and reopen every park whose gate became
        ///        known (D-406).
        ///
        /// @details
        /// Runs first in step 4b of `standardProcessing`. First the class
        /// changes: every park at or below a scope staged by
        /// `stageSubproofRecheck` moves to its canonical key
        /// (`rekeySubproofPark`, decoded order) and is rechecked, since a
        /// merge can make a gate known without a new row. Then the arrivals:
        /// a row known at scope A is known at A and every descendant, so each
        /// arrival visits the park scopes of its shape (`scopesByShape` of
        /// the park index) that are A or below A and collects, through the
        /// canonical template index, the park keys that equal the row
        /// literally or up to that park scope's classes. Both queues are
        /// detached before the revisits, which run in decoded
        /// key-then-scope order; work caused by a reopened subproof waits for
        /// the next drain. Parks under a Phase-3-ignored scope are skipped.
        ///
        /// @param mb Owning block.
        /// @invariant Consumes `pendingSubproofArrivals` and
        ///            `pendingSubproofRechecks` completely; opens scopes only
        ///            through `revisitRejectedSubproof`.
        /// @see stageSubproofArrival, stageSubproofRecheck,
        ///      rekeySubproofPark, revisitRejectedSubproof,
        ///      matchingCanonicalTemplates.
inline void ImplicationSubproofs::drainSubproofArrivals(Memory& mb) {
            auto& arrivals = mb.pendingSubproofArrivals;
            auto& rechecks = mb.pendingSubproofRechecks;
            if (arrivals.size() == 0 && rechecks.size() == 0) return;
            auto& parks = mb.overallHashMemory.rejectedMapSubproof;
            auto& index = mb.lbMemory.canonicalUses.subproofParkMatches;
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
            ScratchArena& arena = genScratchArenas().forSlot(slot);
            ScratchArena& strings = scratchArenas().forSlot(slot);
            ScratchScope frame(arena), stringFrame(strings);
            DirtyState dirty = DirtyState::Clean;
            TypedColdSet<int64_t> selected(&arena, &dirty);
            if (!parks.empty() && rechecks.size() > 0) {
                // Class changes: every park at or below a changed scope moves
                // to its canonical key and is rechecked.
                TypedColdSet<int64_t> changed(&arena, &dirty);
                for (int32_t i = 0; i < rechecks.size(); ++i) changed.mint(rechecks[i]);
                PagedVector<int64_t> stale(&arena, &dirty);
                for (int32_t id = 1; id <= parks.count(); ++id) {
                    const int64_t key = parks.keyAt(id);
                    if (parks.lookup(key) == 0) continue;
                    const NameId parkScope = Codec<StatementKey>::decode(key).validity;
                    if (mb.phase3IgnoresScope(parkScope)) continue;
                    bool hit = false;
                    for (NameId s = parkScope; s != 0 && !hit; s = mb.nameMap.parentOf(s))
                        hit = changed.contains(s);
                    if (hit) stale.push_back(key);
                }
                const auto staleLess = [&](int32_t a, int32_t b) {
                    const auto ka = Codec<StatementKey>::decode(stale[a]);
                    const auto kb = Codec<StatementKey>::decode(stale[b]);
                    const int c = compareSpans(mb.templateInterner.decodeView(ka.orig),
                        mb.templateInterner.decodeView(kb.orig));
                    return c != 0 ? c < 0 : compareSpans(mb.nameMap.decodeView(ka.validity),
                        mb.nameMap.decodeView(kb.validity)) < 0;
                };
                ChunkSortedOrdinals<decltype(staleLess)> staleOrder(arena, stale.size(), 0, staleLess);
                PagedVector<int64_t> staleOrdered(&arena, &dirty);
                for (int32_t i = 0; i < stale.size(); ++i) staleOrdered.push_back(stale[staleOrder.next()]);
                for (int32_t i = 0; i < staleOrdered.size(); ++i)
                    selected.mint(rekeySubproofPark(staleOrdered[i], mb));
            }
            rechecks.clear();
            if (!parks.empty()) {
                canonicalTemplates.ensureCanonicalTemplateIndex(mb, parks, index, arena);
                for (int32_t i = 0; i < arrivals.size(); ++i) {
                    ScratchScope textFrame(strings);
                    const StatementKey arrival = Codec<StatementKey>::decode(arrivals[i]);
                    const StrSpan view = mb.nameMap.decodeView(arrival.orig);
                    const ScratchString text = ScratchString::copyFrom(strings, view.ptr, view.len);
                    const StrSpan query(text);
                    int32_t length = 0;
                    while (length < query.len && query[length] != '[') ++length;
                    const int32_t label = index.labels.lookup(StrSpan(query.ptr, length));
                    if (label == 0) continue;
                    PagedVector<NameId> scopes(&arena, &dirty);
                    index.scopesByShape.forEach(label, [&](int64_t s) { scopes.push_back(static_cast<NameId>(s)); });
                    for (int32_t j = 0; j < scopes.size(); ++j) {
                        const NameId parkScope = scopes[j];
                        if (parkScope != arrival.validity
                            && !mb.nameMap.isStrictAncestor(arrival.validity, parkScope)) continue;
                        if (mb.phase3IgnoresScope(parkScope)) continue;
                        PagedVector<int64_t> found(&arena, &dirty);
                        canonicalTemplates.matchingCanonicalTemplates(mb, parks, index, query, parkScope, parkScope, found, arena);
                        for (int32_t k = 0; k < found.size(); ++k) selected.mint(found[k]);
                    }
                }
            }
            arrivals.clear();
            const auto less = [&](int32_t a, int32_t b) {
                const auto ka = Codec<StatementKey>::decode(selected.keyAt(a + 1));
                const auto kb = Codec<StatementKey>::decode(selected.keyAt(b + 1));
                const int c = compareSpans(mb.templateInterner.decodeView(ka.orig),
                    mb.templateInterner.decodeView(kb.orig));
                return c != 0 ? c < 0 : compareSpans(mb.nameMap.decodeView(ka.validity),
                    mb.nameMap.decodeView(kb.validity)) < 0;
            };
            ChunkSortedOrdinals<decltype(less)> order(arena, selected.count(), 0, less);
            PagedVector<int64_t> ordered(&arena, &dirty);
            for (int32_t i = 0; i < selected.count(); ++i) ordered.push_back(selected.keyAt(order.next() + 1));
            for (int32_t i = 0; i < ordered.size(); ++i) revisitRejectedSubproof(ordered[i], mb);
        }

/// @brief Promote every open integration trial whose gate became known
        ///        (D-481).
        ///
        /// @details
        /// Runs right after `drainSubproofArrivals` in step 4b of
        /// `standardProcessing`. An open trial passes when every gate element
        /// of one of its groups is known at its trial scope
        /// (`elementKnownAtOrAbove`, through the trial scope's classes). The
        /// trials are checked in record order first and promoted afterwards:
        /// a passed record becomes the parent's receipt, its source is
        /// replayed at the parent (`RejectedSourceReplay::replayTrialSource`),
        /// where its recorded witnesses are admitted under the names the trial
        /// minted, with the recorded expressions, and the
        /// trial scope joins the pending wipe and the validity filter, so the
        /// wipe drain of the same call removes the candidate's trial facts.
        ///
        /// @param mb Owning block.
        /// @return Void.
        /// @invariant Mints no scope; passed records are kept, open ones are
        ///            only read.
        /// @see DisintegrationProcessor::disintegrateExpr2 (trial opening),
        ///      RejectedSourceReplay::replayTrialSource, Memory::wipeSubtrees.
inline void ImplicationSubproofs::drainIntegrationTrials(Memory& mb) {
            auto& trials = mb.lbMemory.integrationTrials;
            if (trials.count() == 0) return;
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
            ScratchArena& arena = genScratchArenas().forSlot(slot);
            ScratchArena& strings = scratchArenas().forSlot(slot);
            ScratchScope frame(arena), stringFrame(strings);
            DirtyState dirty = DirtyState::Clean;
            PagedVector<int32_t> passed(&arena, &dirty);
            for (int32_t id = 1; id <= trials.count(); ++id) {
                const IntegrationTrialRec trial = trials.valueAt(id);
                if (trial.status != IntegrationTrialRec::kOpen) continue;
                bool known = false;
                for (int32_t g = 0; g < trial.groupCount && !known; ++g) {
                    known = true;
                    for (int32_t i = g == 0 ? 0 : trial.groupEnd[g - 1]; i < trial.groupEnd[g] && known; ++i) {
                        ScratchScope gateFrame(strings);
                        const StrSpan view = mb.nameMap.decodeView(trial.gates[i]);
                        const ScratchString gate = ScratchString::copyFrom(strings, view.ptr, view.len);
                        known = equivalenceQueries.elementKnownAtOrAbove(StrSpan(gate), trial.trialScope, mb);
                    }
                }
                if (known) passed.push_back(id);
            }
            for (int32_t i = 0; i < passed.size(); ++i) {
                IntegrationTrialRec trial = trials.valueAt(passed[i]);
                trial.status = IntegrationTrialRec::kPassed;
                trials.setValueAt(passed[i], trial);
                rejectedSourceReplay.replayTrialSource(mb, trial);
                mb.pendingWipeScopes.mint(trial.trialScope);
                mb.intValidityNamesToFilter.mint(trial.trialScope);
            }
        }
}
