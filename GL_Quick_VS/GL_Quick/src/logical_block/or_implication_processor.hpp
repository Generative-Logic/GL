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
class CaseLedger;
class CaseProcessor;
class DisintegrationProcessor;
class ProofHistory;
class WatchProcessor;
namespace tests { struct OrImplicationProcessorFixture; }
/// @brief Or Implication Processor operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class OrImplicationProcessor {
friend struct tests::OrImplicationProcessorFixture;
public:
/// @brief Bind inputs for or implication processor.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param blockIdentity Borrowed LogicBlockIdentity interface or state; must outlive this component.
/// @param canonicalizer Borrowed Canonicalizer interface or state; must outlive this component.
/// @param ceFilteringActive Borrowed bool interface or state; must outlive this component.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @param hashRuleClassifier Borrowed HashRuleClassifier interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param caseLedger Borrowed CaseLedger interface or state; must outlive this component.
/// @param caseProcessor Borrowed CaseProcessor interface or state; must outlive this component.
/// @param disintegrationProcessor Borrowed DisintegrationProcessor interface or state; must outlive this component.
/// @param proofHistory Borrowed ProofHistory interface or state; must outlive this component.
/// @param watchProcessor Borrowed WatchProcessor interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
OrImplicationProcessor(LogicBlockIdentity& blockIdentity, Canonicalizer& canonicalizer, const bool& ceFilteringActive, const CompiledDefinitionRegistry& definitions, HashRuleClassifier& hashRuleClassifier, const ProverParameters& parameters, CaseLedger& caseLedger, CaseProcessor& caseProcessor, DisintegrationProcessor& disintegrationProcessor, ProofHistory& proofHistory, WatchProcessor& watchProcessor);

private:
/// @brief Retire symmetric duplicate exclusion sources before matching.
///
/// @details Reuses source canonical ids and paged scratch sorting. Groups never
/// cross source scopes. The source of the largest open cohort wins, with the
/// existing signature tie-break; absent open cohorts, source insertion order wins.
/// @param body Owning resident block with current equality classes.
/// @return Void.
/// @invariant Original leaves, levels, locality and history are never transferred.
/// @see canonicalOrIdAtScope; retireOrImplicationSource.
void retireDuplicateOrSources(Memory& body);

public:
/// @brief Preserve a finalized OR and its ordered complementary leaves.
/// @details Only an admitted positive category-or row can be a source. Levels
/// and locality come from the paired known registries after canonical admission.
/// Existing sources keep their payload; a newly local owner queues a recheck.
/// @param mb Owning block and finalized known-statement registries.
/// @param statement Admitted positive compiled OR with actual argument ids.
/// @param publicationMail Current phase output, or same-iteration mail for ordinary callers.
/// @return Void.
void registerOrImplicationSource(Memory& mb, const IntEncodedExpr& statement,
            ColdMail* publicationMail = nullptr);

private:
/// @brief Build newly needed signed-core views and reconcile staged changes.
/// @details Source events are collected before one scan for all new cores.
/// Existing core arrivals avoid that scan. Canonical changes settle before
/// exact complementary-template buckets queue comparable source scopes.
/// @param mb Owning block and its canonical and derived OR state.
/// @return Void.
void prepareOrImplicationMatches(Memory& mb);

private:
/// @brief Unlink one active source and facts no longer used by any source.
/// @details Watch views and ordinary fact views sharing an owner are distinct.
/// Every posting loses reachability immediately; payload compaction is batched.
/// @param mb Owning block and canonical/derived OR state.
/// @param sourceKey Active original OR and source scope.
/// @return Void.
void retireOrImplicationSource(Memory& mb, int64_t sourceKey);

public:


public:
/// @brief Activate a residual OR only after its deposit is admitted.
/// @details An existing ancestor supplies the continuation at its own scope.
/// Canonical duplicate refusals reuse the deposit chooser; other refused
/// deposits remain absent. Case-disintegration flags are independent.
/// @param mb Owning block.
/// @param original Stable original residual text.
/// @param scope Offered scope.
/// @param registeredOriginal Canonical id returned by the deposit door, or zero.
/// @param publicationMail Current phase output, or same-iteration mail for ordinary callers.
/// @return True if an admitted form supplies the source.
bool activateOrImplicationContinuation(Memory& mb, StrSpan original,
            NameId scope, NameId registeredOriginal = 0, ColdMail* publicationMail = nullptr);

private:
/// @brief Select deterministic refuters and compatible scopes for one reduction.
/// @details Canonical views may be prepared, but no known facts, proof history,
/// mail or source retirement are produced. A negative match leaves no firing.
/// Candidates are the fact carriers of the matching entries; the locality of
/// a spelling's rewrite equalities, and of the source's own, is computed here.
/// @param mb Owning block with prepared canonical matching state.
/// @param sourceKey Original OR and source scope.
/// @param lookupScope Candidate effective scope.
/// @param selection Stack-owned result, consumed only when this returns true.
/// @return True when a scope-compatible reduction was selected.
bool selectOrImplication(Memory& mb, int64_t sourceKey, NameId lookupScope,
            OrImplicationSelection& selection);

private:
/// @brief Materialize a selection through the existing checked firing path.
/// @details Applies ordinary hypothetical-scope gates, creates only selected
/// equality/proof records and mail, and retains the original retirement order.
/// @param mb The unchanged block from the successful selection call.
/// @param selection Successful selection for immediate consumption.
/// @param output Live internal-mail output for this processing phase.
/// @return True when the head is known or accepted for delivery.
bool materializeOrImplication(Memory& mb,
            const OrImplicationSelection& selection, ColdMail& output);

private:
/// @brief Select and materialize one scoped OR reduction without repeat matching.
/// @details The stack record contains ids and bounded values only. Selection
/// and materialization run consecutively against the same block state.
/// @param mb Owning block with prepared matching views.
/// @param sourceKey Original OR and source scope.
/// @param lookupScope Candidate effective scope.
/// @param output Live internal-mail output for this processing phase.
/// @return True when the selected head is known or accepted for delivery.
bool evaluateOrImplication(Memory& mb, int64_t sourceKey,
            NameId lookupScope, ColdMail& output);

public:
/// @brief Reconcile and evaluate staged OR work until no fresh event remains.
/// @details Stable logical identities survive source retirement and index
/// recycling. Parent scopes are considered before their descendants; new
/// canonical twins are reconciled in the next pass through the same drain.
/// Each pass deduplicates its current events, but a later fact or class change
/// can require evaluating the same source and scope again.
/// @param mb Owning block and staged OR state.
/// @param output Caller-selected live internal-mail output.
/// @return Void.
void drainOrImplications(Memory& mb, ColdMail& output);

private:
/// @brief Seed class-scope token postings for canonical OR lookup.
/// @details The initial pass reads each serialized class once; further calls
/// are cache hits. Class additions are handled by stageOrClassMember.
/// @param mb Owning block and current equivalence classes.
/// @param index Derived matching state on the block arena.
/// @return Void.
void buildOrClassDirectory(Memory& mb, OrImplicationMatchIndex& index);

private:
/// @brief Publish a new canonical entry with its form's ordered token stream.
/// @details Tokens of the canonical form stream directly into the index
/// transaction, avoiding a temporary collection and preserving NameMap mint
/// order. The index owns all entry and reverse-token updates; this function
/// owns parsing. A spelling's own rewritten tokens are posted per view
/// (postOrViewTokens).
/// @param mb Owning block and expression interner.
/// @param index Derived matching index receiving the new entry.
/// @param key Canonical expression and effective scope.
/// @param value Settled canonical result from the existing chooser.
/// @return Void.
void publishOrCanonicalCache(Memory& mb, OrImplicationMatchIndex& index,
            int64_t key, OrCanonicalCache value);

private:
/// @brief Queue every user of an entry at relevant strict-descendant equality scopes.
/// @details Each user is projected by projectOrUser; queued identities are
/// stable even when view slots are later recycled.
/// @param mb Owning block and scope tree.
/// @param index Derived matching state.
/// @param cacheKey Live canonical expression and lookup scope.
/// @return Void.
void projectOrCacheViews(Memory& mb, OrImplicationMatchIndex& index,
            int64_t cacheKey);

private:
/// @brief Queue one view at the strict-descendant scopes whose classes reach it.
/// @details A class member reaches the view through a token of its entry's
/// canonical form or through a token its own spelling adds to that form.
/// Token and class-scope postings avoid a source-by-scope product; only
/// identities without a live view are queued.
/// @param mb Owning block and scope tree.
/// @param index Derived matching state.
/// @param viewId Live view slot.
/// @return Void.
void projectOrUser(Memory& mb, OrImplicationMatchIndex& index, int32_t viewId);

private:
/// @brief Bind an OR lookup view to the canonical entry of its spelling.
/// @details The entry is keyed by canonical expression and effective scope,
/// so spellings with one canonical form share it. An existing identity keeps
/// its view and entry, including a dirty entry awaiting the drain. A new view
/// posts its rewritten tokens and is projected to reachable descendants.
/// @param mb Owning block, equivalence classes and interners.
/// @param index Derived matching state.
/// @param identity Original owner, leaf role and effective scope.
/// @param originalId Fully instantiated complementary leaf or known fact.
/// @return The live view slot for the supplied identity.
int32_t bindOrCanonicalView(Memory& mb, OrImplicationMatchIndex& index,
            OrViewIdentity identity, NameId originalId);

private:
/// @brief Reuse or create the canonical entry of an expression at a scope.
/// @details One text-only canonicalization (canonicalOrFormId) finds the
/// entry; every spelling with that canonical form at the scope shares it and
/// its token postings, scalar facts and derived OR leaf views alike.
/// @param mb Owning block, classes and interners.
/// @param index Derived matching state.
/// @param originalId Original whole expression in NameMap space.
/// @param scopeId Effective lookup scope.
/// @return Packed canonical-expression/effective-scope entry key.
int64_t ensureOrCanonicalCache(Memory& mb, OrImplicationMatchIndex& index,
            NameId originalId, NameId scopeId);

private:
/// @brief Canonical NameMap id of an expression at a scope for OR lookup.
/// @details A text-only call of the shared chooser with the OR equality and
/// inequality opt-in: no justifiers, locality or levels are built. The text
/// equals the form canonicalWatchFormAt produces with evidence.
/// @param mb Owning block with initialized canonical classes.
/// @param originalId Minted original expression.
/// @param scopeId Effective lookup scope.
/// @return The original id when it is canonical, else the id of its canonical text.
NameId canonicalOrFormId(Memory& mb, NameId originalId, NameId scopeId);

private:
/// @brief Whether the equalities rewriting a spelling into its entry's form are local.
/// @details Computed when a firing considers the spelling, from the current
/// classes, with the lookup canonicalization of canonicalWatchFormAt. The
/// rewritten text must be the entry's canonical form.
/// @param mb Owning block with initialized canonical classes.
/// @param originalId A spelling that differs from the canonical form.
/// @param scopeId The entry's effective scope.
/// @param canonicalId The entry's canonical form.
/// @return True when an applied equality is this block's own.
/// @invariant Every spelling of an entry rewrites to the entry's canonical form.
bool rewriteLocalAt(Memory& mb, NameId originalId, NameId scopeId, NameId canonicalId);

private:
/// @brief Post the substitution tokens a view's spelling adds to its canonical form.
/// @details Both texts are copied before token ids are minted. A canonical
/// spelling adds no token. These tokens let a class change on a rewritten
/// member dirty the entry or project only this view.
/// @param mb Owning block and NameMap.
/// @param index Derived matching state.
/// @param viewId Live view slot of the spelling.
/// @param originalId The view's original expression.
/// @param canonicalId Canonical form of the view's entry.
/// @return Void.
void postOrViewTokens(Memory& mb, OrImplicationMatchIndex& index,
            int32_t viewId, NameId originalId, NameId canonicalId);

private:
/// @brief Derive ordered complementary views from one canonical whole OR.
/// @details Canonicalizing the parent rewrites all argument occurrences,
/// including equality leaves; refuter views use the same equality opt-in.
/// The compiled registry supplies the existing ordered flattening operation.
/// @param mb Owning block and admitted source store.
/// @param index Derived matching state.
/// @param sourceKey Active original OR and source validity.
/// @param scopeId Effective lookup scope at or below the source scope.
/// @return Void.
void bindOrSourceViews(Memory& mb, OrImplicationMatchIndex& index,
            int64_t sourceKey, NameId scopeId);

private:
/// @brief Stage the canonical work affected by one class member.
/// @details A class at or above an entry dirties that entry once. A class below
/// an entry projects all its users when the member is a token of the canonical
/// form, and only the users whose spelling adds the member otherwise; sibling
/// scopes never combine.
/// @param mb Owning block and validity tree.
/// @param index Derived matching state with its class directory built.
/// @param memberId Merged class member in NameMap space.
/// @param scopeId Validity of the merged class.
/// @param invalidate Whether to invalidate visible caches in addition to new scope projections.
/// @return Void.
void stageOrClassMember(Memory& mb, OrImplicationMatchIndex& index,
            NameId memberId, NameId scopeId, bool invalidate = true);

private:
/// @brief Settle dirty entries and affected descendant views before matching.
/// @details A dirty entry canonicalizes its own canonical text once. An
/// unchanged form settles in place; a moved form moves every user to the
/// entry of the new form, merging with a live one, and its sources rebind
/// their leaves. Decoded ordering makes canonical-name creation independent
/// of posting order. No stored known fact or GPU premise registry is changed.
/// @param mb Owning block, equivalence classes and interners.
/// @param index Derived matching state and its deduplicated pending work.
/// @return Number of dirty entries sent through the shared canonical chooser.
int32_t refreshOrCanonicalViews(Memory& mb,
            OrImplicationMatchIndex& index);
private:

LogicBlockIdentity& blockIdentity;
Canonicalizer& canonicalizer;
const bool& ceFilteringActive;
const CompiledDefinitionRegistry& definitions;
HashRuleClassifier& hashRuleClassifier;
const ProverParameters& parameters;
CaseLedger& caseLedger;
CaseProcessor& caseProcessor;
DisintegrationProcessor& disintegrationProcessor;
ProofHistory& proofHistory;
WatchProcessor& watchProcessor;
};
}
#include "../logical_block/case_ledger.hpp"
#include "../logical_block/case_processor.hpp"
#include "../logical_block/disintegration_processor.hpp"
#include "../logical_block/proof_history.hpp"
#include "../logical_block/watch_processor.hpp"
namespace gl {
}
