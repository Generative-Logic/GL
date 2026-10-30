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
class AdmissionMemory;
class CaseLedger;
class CaseProcessor;
class RejectedSourceReplay;
class StatementMemory;
namespace tests { struct DisintegrationProcessorFixture; }
/// @brief Disintegration Processor operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class DisintegrationProcessor {
friend struct tests::DisintegrationProcessorFixture;
public:
/// @brief Bind inputs for disintegration processor.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param admissionPolicy Borrowed AdmissionPolicy interface or state; must outlive this component.
/// @param canonicalTemplates Borrowed CanonicalTemplateMatcher interface or state; must outlive this component.
/// @param canonicalizer Borrowed Canonicalizer interface or state; must outlive this component.
/// @param ceFilteringActive Borrowed bool interface or state; must outlive this component.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @param existenceRuleCompiler Borrowed ExistenceRuleCompiler interface or state; must outlive this component.
/// @param hashRuleClassifier Borrowed HashRuleClassifier interface or state; must outlive this component.
/// @param integrationBuilder Borrowed IntegrationInstructionBuilder interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param rejectedSources Borrowed RejectedSourceMemory interface or state; must outlive this component.
/// @param admissionMemory Borrowed AdmissionMemory interface or state; must outlive this component.
/// @param caseLedger Borrowed CaseLedger interface or state; must outlive this component.
/// @param caseProcessor Borrowed CaseProcessor interface or state; must outlive this component.
/// @param rejectedSourceReplay Borrowed RejectedSourceReplay interface or state; must outlive this component.
/// @param statementMemory Borrowed StatementMemory interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
DisintegrationProcessor(AdmissionPolicy& admissionPolicy, CanonicalTemplateMatcher& canonicalTemplates, Canonicalizer& canonicalizer, const bool& ceFilteringActive, const CompiledDefinitionRegistry& definitions, ExistenceRuleCompiler& existenceRuleCompiler, HashRuleClassifier& hashRuleClassifier, IntegrationInstructionBuilder& integrationBuilder, const ProverParameters& parameters, RejectedSourceMemory& rejectedSources, AdmissionMemory& admissionMemory, CaseLedger& caseLedger, CaseProcessor& caseProcessor, RejectedSourceReplay& rejectedSourceReplay, StatementMemory& statementMemory);

private:

inline std::string makeMarkedExpr(const std::string& expr, const std::string& var);

public:


private:
/// @brief Publish prepared auxiliary compacts through ordinary disintegration.
/// @details Only template substitution and checked provenance occur here. Each
/// contrapositive is a registered compact whose normal expansion installs its
/// own rule. Collected traversal markers break reciprocal activation cycles.
/// @param instance Fixed-argument source-family instance.
/// @param ownerKey Current collected statement key.
/// @param memoryBlock Resident target logic block.
/// @param collected Scratch products of this disintegration traversal.
/// @param validityName Exact source scope.
/// @param trackHistoryLocal Whether publication history is enabled.
/// @param iteration Source disintegration generation.
/// @return Void.
/// @invariant No registry mint, grouping or contraposition construction occurs here.
void collectContrapositionProducts(StrSpan instance, StrSpan ownerKey,
            Memory& memoryBlock, CollectedArena& collected, StrSpan validityName,
            bool trackHistoryLocal, int iteration = 0);

private:
/// @brief Publish prepared refutation compacts with exact structural history.
/// @details Every template is instantiated without changing the shared registry.
/// Structural expansion, compilation and ordinary rule expansion remain distinct.
/// @param mb Owning resident block.
/// @param conjunction Fully fixed conjunction instance.
/// @param scope Exact deepest supporting scope.
/// @param dependencies Sole encountered OR citation.
/// @param dependencyCount Exactly one supporting citation.
/// @param levels Sorted nonempty supporting levels.
/// @param levelCount Number of levels.
/// @param output Current internal-mail output channel.
/// @return Void.
void publishConjunctionRefutation(Memory& mb, StrSpan conjunction, NameId scope,
            const OriginDep* dependencies, int32_t dependencyCount,
            const int* levels, int32_t levelCount, ColdMail& output);

public:
/// @brief Publish fixed direct refutations for an admitted OR's AND alternatives.
/// @details The known OR fixes every argument locally. No reverse-rule search,
/// descendant replay or equality index is created. Ordinary compact lifecycle
/// handles later canonical changes and scope cleanup.
/// @param mb Owning block with the admitted OR and its supporting levels.
/// @param statement Exact positive OR source.
/// @param output Current processing phase's output mail.
/// @return Void.
void activateDirectOrRefutations(Memory& mb, const IntEncodedExpr& statement, ColdMail& output);

public:


public:


public:
// Turn optimizations OFF for just this section
//#pragma optimize("", off)




public:


public:
// In GL_Quick_VS/GL_Quick/src/analyze_expressions.cpp

/// @brief Flatten one compiled OR entity into its ordered atomic leaves in the
///        current working instruction.
///
/// @details
/// The instantiated `WorkInstruction` already contains the outer OR and every
/// recursively prepared child entity. This function walks that read-only graph
/// without re-parsing or re-substituting strings: a child whose exact signature
/// resolves to another `or` entity is expanded immediately in the same call;
/// any non-OR child is appended as one leaf. Element order is preserved across
/// both left- and right-nested shapes. Duplicate entity records must be
/// structurally identical, recursion-path cycles assert, and capacity overflow
/// asserts before a write. The local recursive lambda is stack-only and the
/// function mints and allocates nothing.
///
/// @param instructions The instantiated, read-only instruction graph.
/// @param rootEntity The entity index of the outer OR.
/// @param out Caller array receiving ordered leaf spans.
/// @param cap Capacity of @p out.
/// @return Number of leaves written; always at least two.
/// @invariant Every returned span aliases `instructions`; the outer entity and
///            every recursively entered entity are category `or` with at least
///            two elements.
/// @see DisintegrationProcessor::disintegrateExprCore2.
int32_t flattenOrLeaves(
            const WorkInstruction& instructions,
            int32_t rootEntity,
            StrSpan* out,
            int32_t cap) const;

private:

void disintegrateExprCore2(StrSpan expr,
            WorkInstruction& instructions,
            Memory& memoryBlock,
            int iteration,
            CollectedArena& collected,
            NewVarStore& newVarMap,
            StrSpan validityName,
            bool trackHistoryLocal = true,
            bool allowOrDisintegration = false,
            const int* orSeedLevels = nullptr,
            int32_t orSeedLevelCount = 0,
            bool allowOrProbe = false,
            StrSpan parentWitness = StrSpan(),
            WitnessMetaStore* witnessMeta = nullptr,
            bool integrationPremise = false,
            bool conjecturePremise = false);

private:
/// @brief Expand one expression through an instruction substitution view.
/// @details The traversal, witness order, history and OR decisions are unchanged.
/// Witness recursion creates a read-only substitution view; instruction extension
/// and concrete OR consumers materialize only their current view.
/// @param expr Expression at this recursion node.
/// @param instructions Current instruction view.
/// @param memoryBlock Owning logic block.
/// @param iteration Firing generation.
/// @param collected Expansion products and history carriers.
/// @param newVarMap Temporary witnesses and their defining elements.
/// @param validityName Current validity scope.
/// @param trackHistoryLocal Whether history may be emitted.
/// @param allowOrDisintegration Existing unconditional OR signal.
/// @param orSeedLevels Caller-owned source levels.
/// @param orSeedLevelCount Number of source levels.
/// @param allowOrProbe Whether demand-based OR processing is enabled.
/// @param parentWitness Enclosing witness token, or the root empty token.
/// @param witnessMeta Current discovery or publication metadata.
/// @param integrationPremise The expression is the deposited premise of an
///        implication being integrated; recursive calls keep the default false,
///        so only the root reacts (D-460).
/// @param conjecturePremise The expression is, or is reached by existence
///        unfolding from, a conjecture's non-anchor premise seeded into its
///        block; an OR reached this way opens with the rule-product permission
///        where the block has proof work (D-495) and is marked for the flag-6
///        unconditional-ordis mail (D-528).
///        Existence recursion keeps it; conjunction and implication recursion
///        drop it.
/// @return Void.
/// @invariant Child substitution views cannot mutate their parent instruction.
/// @see DisintInstructionView, disintegrateExprCore2.
void disintegrateExprCoreView(StrSpan expr,
            DisintInstructionView& instructions, Memory& memoryBlock, int iteration,
            CollectedArena& collected, NewVarStore& newVarMap, StrSpan validityName,
            bool trackHistoryLocal, bool allowOrDisintegration,
            const int* orSeedLevels, int32_t orSeedLevelCount, bool allowOrProbe,
            StrSpan parentWitness, WitnessMetaStore* witnessMeta,
            bool integrationPremise = false,
            bool conjecturePremise = false);

public:
// Turn optimizations OFF for just this section
//#pragma optimize("", off)

/// @brief Expand a source temporarily and publish only admitted constructions.
/// @details Admission preserves its ordinary routes and commits successful updates
/// after permanent substitution. Ordinary integration requires an eligible depth
/// and a source without algebra-witness inputs. Unresolved eligible sources park
/// for ordinary mail replay; no temporary name or speculative witness history
/// enters persistent state.
/// @param expr Concrete source expression.
/// @param memoryBlock Owning logic block.
/// @param iteration Firing generation; -1 retains the ordinary zero-generation normalization.
/// @param forceDeep Select integration constructions on integration-only enclosing paths.
/// @param validityName Target scope.
/// @param out Accepted statements, rules, OR sources and relay carriers.
/// @param trackHistoryLocal Whether this call materializes history.
/// @param allowOrDisintegration Original unconditional OR-opening signal.
/// @param orSeedLevels Caller-owned source level run.
/// @param orSeedLevelCount Number of source levels.
/// @param allowOrProbe Whether ordinary OR demand processing is enabled.
/// @param replayKinds Explicit queued construction bits and inherited-carrier bit.
/// @param replayIntegrationDepth Original assigned depth for replay, or -1 for a new expansion.
/// @param inheritedSource Preserve a relay carrier's no-cascade status in source parks.
/// @param recursionHead The source was fired by the installed induction implication
/// (I-229); requires @p forceDeep. Every published witness enters
/// `Memory::recursionHeadWitnesses` (I-286).
/// @param integrationPremise The source is the deposited premise of an
/// implication being integrated (D-460).
/// @param conjecturePremise The source is a conjecture's non-anchor premise
/// seeded into its block; the publication walk opens the ORs reached through
/// existence unfolding where the block has proof work (D-495) and reports
/// those published at the deposit scope in `out.unconditionalOrdis`
/// (D-528).
/// @return Whether every relevant existence source is covered.
/// @invariant An inner witness's admission carries an enclosing witness along only
/// when that witness fills a level of the matched integration instruction at or
/// above the key, or the bound variable of an existence beside it; otherwise the
/// inner witness waits for its enclosing witnesses (D-482).
/// @see TemporaryWitnessStore, queueParkedExistence, I-252.
bool
            disintegrateExpr2(StrSpan expr,
                Memory& memoryBlock,
                int iteration,
                bool forceDeep,
                StrSpan validityName,
                DisintProducts& out,
                bool trackHistoryLocal = true,
                bool allowOrDisintegration = false,
                const int* orSeedLevels = nullptr,
                int32_t orSeedLevelCount = 0,
                bool allowOrProbe = false,
                int32_t replayKinds = 0,
                bool inheritedSource = false, int32_t replayIntegrationDepth = -1,
                bool recursionHead = false,
                bool integrationPremise = false,
                bool conjecturePremise = false);

public:
/// @brief Probe what integrating an expression would disintegrate into,
        ///        inside its sentinel and hypothesis scopes.
        ///
        /// @details
        /// Disintegrates @p expr into a fresh sentinel scope
        /// `product_of_hypo_disintegration_of_integration_goal_<n>_<expr>`
        /// (`<n>` = the name map's name count at entry, so a goal prepared
        /// again at the same scope gets a new scope) without
        /// history, and when some argument has the `(1)` definition set, opens
        /// the hypothesis scope `_var<i>_<x>..._hypo_<expr>` for the typed
        /// variables. A marker expression is left alone. The sentinel serves
        /// only this dry run: its name is minted here (asserted) and it is
        /// queued for the step's scope wipe right after the dry run
        /// (D-496), so no later class change
        /// visits it.
        ///
        /// @param expr         Expression being prepared (caller-stable span).
        /// @param memoryBlock  Owning block.
        /// @param validityName Preparation scope.
        /// @return (void)
        /// @see prepareIntegration — the caller.
void disintegrateExprHypothetically(StrSpan expr, Memory& memoryBlock, StrSpan validityName);
private:

AdmissionPolicy& admissionPolicy;
CanonicalTemplateMatcher& canonicalTemplates;
Canonicalizer& canonicalizer;
const bool& ceFilteringActive;
const CompiledDefinitionRegistry& definitions;
ExistenceRuleCompiler& existenceRuleCompiler;
HashRuleClassifier& hashRuleClassifier;
IntegrationInstructionBuilder& integrationBuilder;
const ProverParameters& parameters;
RejectedSourceMemory& rejectedSources;
AdmissionMemory& admissionMemory;
CaseLedger& caseLedger;
CaseProcessor& caseProcessor;
RejectedSourceReplay& rejectedSourceReplay;
StatementMemory& statementMemory;
};
}
#include "../logical_block/admission_memory.hpp"
#include "../logical_block/case_ledger.hpp"
#include "../logical_block/case_processor.hpp"
#include "../logical_block/rejected_source_replay.hpp"
#include "../logical_block/statement_memory.hpp"
namespace gl {


inline std::string DisintegrationProcessor::makeMarkedExpr(const std::string& expr, const std::string& var) {
            // Single key var -> "marker"; the value is a static literal, so no arena.
            const StrReplacement pairs[1] = { { StrSpan(var), StrSpan("marker", 6) } };
            return replaceKeysToString(expr, pairs, 1);
        }


}
