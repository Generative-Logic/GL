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
#include "../logical_block/admission_memory.hpp"
#include "../logical_block/integration_processor.hpp"
#include "../logical_block/implication_subproofs.hpp"
#include "../logical_block/rejected_source_replay.hpp"
#include "../inference/hash_rule_installer.hpp"
#include "../logical_block/compact_rule_processor.hpp"
#include "../logical_block/statement_memory.hpp"
#include "../logical_block/equivalence_processor.hpp"
#include "../logical_block/equality_propagation.hpp"
#include "../logical_block/equivalence_admission_rewriter.hpp"
#include "../logical_block/watch_processor.hpp"
#include "../logical_block/case_ledger.hpp"
#include "../logical_block/case_processor.hpp"
#include "../logical_block/disintegration_processor.hpp"
#include "../logical_block/or_implication_processor.hpp"
#include "../logical_block/proof_history.hpp"
#include "../logical_block/unary_integration_processor.hpp"
#include "../logical_block/goal_processor.hpp"
#include "../logical_block/scope_lifecycle.hpp"
#include "../logical_block/logic_block_operations.hpp"
#include "../logical_block/block_mail_processor.hpp"
#include "../logical_block/anchor_scope_queries.hpp"
#include "../logical_block/equivalence_queries.hpp"
#include "../logical_block/block_key_reader.hpp"
#include "../logical_block/phase13_timing_slot.hpp"
#include "../logical_block/watch_ledger.hpp"
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
#include "../proof/or_candidate.hpp"
#include "../infra/rt_tracker.hpp"
#include "../infra/hashburst_dump.hpp"
#include "../memory_infra/steward.hpp"
#include "../memory_infra/lb_store.hpp"
#include "../mail_log.hpp"
#include "../simple_map_store.hpp"
#include "../memory_infra/or_implication_match_index.hpp"
#include "../memory_infra/temporary_witness_store.hpp"
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
class OrCandidateProgress;
namespace tests { struct TheoremPublisherFixture; }
/// @brief Theorem Publisher operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class TheoremPublisher {
friend struct tests::TheoremPublisherFixture;
public:
/// @brief Bind the theorem publisher operation inputs.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param anchorScopeQueries Borrowed AnchorScopeQueries interface or state; must outlive this component.
/// @param body Borrowed Memory interface or state; must outlive this component.
/// @param definitionExpansion Borrowed DefinitionExpansion interface or state; must outlive this component.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @param dependenciesMutex Borrowed std::mutex interface or state; must outlive this component.
/// @param fullTheoremList Borrowed std::vector interface or state; must outlive this component.
/// @param globalDependencies Borrowed Dependencies interface or state; must outlive this component.
/// @param globalTheoremList Borrowed std::vector interface or state; must outlive this component.
/// @param globalTheoremProducers Borrowed std::vector interface or state; must outlive this component.
/// @param globalTheoremStrings Borrowed std::vector interface or state; must outlive this component.
/// @param orInRunScannedRows Borrowed std::vector interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param pendingCompactionMutex Borrowed std::mutex interface or state; must outlive this component.
/// @param pendingCompactionQueue Borrowed std::vector interface or state; must outlive this component.
/// @param proofHistory Borrowed ProofHistory interface or state; must outlive this component.
/// @param retractedVacuousTheorems Borrowed std::vector interface or state; must outlive this component.
/// @param scopeLifecycle Borrowed ScopeLifecycle interface or state; must outlive this component.
/// @param simpleMapStore Borrowed SimpleMapStore interface or state; must outlive this component.
/// @param steward Borrowed std::vector interface or state; must outlive this component.
/// @param structuralCompiler Borrowed StructuralCompiler interface or state; must outlive this component.
/// @param theoremListMutex Borrowed std::mutex interface or state; must outlive this component.
/// @param updateGlobalSingleChapterPages Borrowed std::vector interface or state; must outlive this component.
/// @param orCandidateProgress Borrowed OrCandidateProgress interface or state; must outlive this component.
/// @param orInRunCompanions Existing shared state or operation interface, borrowed for this component lifetime.
/// @return None; construction binds the supplied references.
TheoremPublisher(AnchorScopeQueries& anchorScopeQueries, Memory& body, DefinitionExpansion& definitionExpansion, const CompiledDefinitionRegistry& definitions, std::mutex& dependenciesMutex, std::vector< std::tuple<std::string, std::string, std::string, std::string >>& fullTheoremList, Dependencies& globalDependencies, std::vector< std::tuple<std::string, std::string, std::string, std::string >>& globalTheoremList, std::vector<const Memory*>& globalTheoremProducers, std::unordered_set<std::string>& globalTheoremStrings, std::unordered_set<std::string>& orInRunScannedRows, const ProverParameters& parameters, std::mutex& pendingCompactionMutex, std::vector<std::tuple<std::string, int, int>>& pendingCompactionQueue, ProofHistory& proofHistory, std::vector<std::string>& retractedVacuousTheorems, ScopeLifecycle& scopeLifecycle, SimpleMapStore& simpleMapStore, std::unique_ptr<MemorySteward>& steward, StructuralCompiler& structuralCompiler, std::mutex& theoremListMutex, std::optional<SealedPageSet>& updateGlobalSingleChapterPages, OrCandidateProgress& orCandidateProgress, std::unordered_set<std::string>& orInRunCompanions);

public:
/// @brief Retire the logic blocks of a head-switched companion that an
        /// in-run or theorem subsumes.
        ///
        /// @details
        /// The companion follows from the or theorem, and a later proof of
        /// it is withheld (`orInRunCompanions`), so its own conjecture grid
        /// has no work left. The retirement gives the companion the lifecycle
        /// of a proved conjecture whose proof closed elsewhere:
        ///
        /// - its goal block (the last block of its premise chain) receives the
        ///   seed a firing complement twin would deposit, `negate(head)`, in
        ///   the settled-goal inbox, and wakes. The block's own end-of-burst
        ///   `drainDisprovedGoals` then closes the companion's goal with
        ///   success semantics and an opposite-head goal as disproved (D-279);
        /// - `deactivateUnnecessary` retires both contradiction twins and the
        ///   induction blocks of the companion and of its opposite-head
        ///   conjecture, as `updateGlobalSingleChapter` does for a proved
        ///   theorem, and deactivates every chain block left without an active
        ///   child or a `main` goal.
        ///
        /// A companion that was never scheduled as a conjecture has no goal
        /// block: the retirement is then a defined no-op. An inactive goal
        /// block has no goal left to settle and receives no seed. The walks
        /// set `deactivationSweepOwed`; the calling seam settles it.
        ///
        /// @param companion The head-switched companion of a registered
        ///        in-run or theorem, as recorded in `orInRunCompanions`.
        /// @return Nothing.
        /// @invariant Single-threaded phase-4 barrier seam only: it writes
        ///            other blocks' `isActive`, inbox and `hasWork` (I-28).
        /// @see `constructOrTheoremsInRun` — the only production caller.
        /// @see `deactivateUnnecessary`, `drainDisprovedGoals`,
        ///      `Memory::pendingDisprovedGoals`.
void retireSubsumedCompanion(const std::string& companion);
private:
/// @brief Look up an LB by its global key (the deepest-first chain of
/// `exprKey` values from the LB up to root).
///
/// @details
/// Companion to `getGlobalKey`: round-trip
/// `accessMemory(getGlobalKey(mb), root)` returns a pointer equal to
/// `&mb` for any LB still alive. Walks the LB tree from `bodyOfProves1`
/// step by step, taking the `SimpleMapStore` edge that matches each chain
/// element. Returns `nullptr` if any step doesn't have a matching
/// child — typically because the target LB has been torn down between
/// iterations.
///
/// @param theoremKey     Global key of the target LB (deepest first).
/// @param bodyOfProves1  Root sentinel from which to walk. Typically
///                       `body` for the main LB tree or `ceBody` for
///                       the CE-filter tree.
/// @return Pointer to the matching LB, or `nullptr` on miss.
/// @see `getGlobalKey` — produces the matching key.
Memory* accessMemory(const std::vector<std::string>& theoremKey,
            Memory& bodyOfProves1);

private:
/// @brief Try to back-reformulate a theorem's head from its disintegrated
/// form into a higher-level operator head.
///
/// @details
/// One of the three back-paths gated by `!parameters.ban_disintegration`
/// per I-7. The other two are Pass B disintegration and
/// hypo-disintegration. Back-reformulation looks at the head of a theorem
/// and asks whether it could be re-expressed using a configured operator
/// from `coreExpressionMap` — converting, for instance, the disintegrated
/// `(p[a]) ∧ (q[a])` form back into a single-operator equivalent.
///
/// Returns `true` iff the back-reformulation succeeded (i.e. the theorem
/// has a viable higher-level head); `backReformulated` is populated with
/// the rewritten form. Returns `false` if no operator matches, in which
/// case `backReformulated` stays unchanged. The function additionally
/// gates on `parameters.incubator_mode` — non-incubator runs early-exit
/// false because the back-reformulation product is unsound for the
/// real-math output set; incubator artefacts are tagged separately as
/// `incubator back reformulation` and never promoted to the main
/// theorem list.
///
/// @param theorem           Source theorem text.
/// @param backReformulated  Reformulated form on success; untouched on
///                          failure.
/// @return True iff back-reformulation succeeded.
/// @invariant I-7 — guarded by `!parameters.ban_disintegration`.
bool tryBackReformulateOperatorHead(
            const std::string& theorem,
            std::string& backReformulated);

public:
// analyze_expressions.cpp
// Thread-safe: locks globalTheoremList only when mutating it.
/// @brief Append a theorem tuple to `globalTheoremList` unless the
///        theorem string is already present — the sink dedup, with one
///        sanctioned exception: a first-class arrival UPGRADES an
///        existing proved-not-broadcast row in place.
///
/// @details
/// The single door for every `globalTheoremList` insertion. Locks
/// `theoremListMutex`, probes the companion `globalTheoremStrings`
/// membership set, and appends the `(theorem, method, aux2, aux3)`
/// tuple on first sight of the theorem string. First emission
/// wins between first-class methods: a theorem proved through two
/// first-class paths (e.g. a contradiction-LB
/// discharge and the ordinary goal closure of the same conjecture)
/// keeps the first-drained row and its method label; the later
/// arrival is dropped, so the proved-theorem artifacts and the
/// chapter export see each theorem exactly once. The
/// proved-not-broadcast tier is subordinate on BOTH sides of the
/// dedup: an incoming tier row never lands on an existing row of any
/// method, and an incoming first-class row REPLACES an existing tier
/// row in place (method, aux columns, recorded producer; the tier's
/// `fullTheoremList` mirror row leaves — the upgraded row re-mirrors
/// through the first-class path). The tier records a closure that
/// never circulated, so a circulating derivation of the same theorem
/// supersedes it. All callers are
/// single-threaded seams (post-join drains, registration, the
/// orchestrator, the export rebuild); the lock keeps the historical
/// guard of the raw `emplace_back` sites.
///
/// @param theorem  The theorem string (the dedup key).
/// @param method   Emission-path label (`single chapter proof`, `induction`, ...).
/// @param aux2     Method-specific third column.
/// @param aux3     Method-specific fourth column.
/// @param producer The grid LB whose discharge/promotion emitted the
///                 row (recorded in `globalTheoremProducers` for the
///                 vacuous-reversion sweep), or `nullptr` for
///                 producer-less paths.
/// @return `true` when the tuple was appended or an existing tier row
///         was upgraded in place, `false` when the
///         theorem string was already present (duplicate dropped).
/// @see `globalTheoremStrings`, `revertVacuousGlobalTheorems`; the
///      export rebuild in `visualizer.cpp` clears all three
///      containers together.
bool appendGlobalTheorem(const std::string& theorem,
            const std::string& method, const std::string& aux2,
            const std::string& aux3, const Memory* producer = nullptr);

public:
/// @brief Retract every global-theorem row whose producer LB turned
///        out unintentionally contradictory.
///
/// @details
/// The emission gates (`hasContradictedAncestor` at
/// `drainUpdateGlobalSingleChapter` and the induction promotion) suppress a
/// vacuous theorem at submission time — but an LB can emit theorems
/// in earlier iterations and only later derive its main-scope
/// statement/negation pair (the vacuous-premise flag,
/// `Memory::mainContradiction`). Those already-submitted rows are
/// retracted here: every `globalTheoremList` row whose recorded
/// producer has a contradicted self-or-ancestor leaves the list, its
/// string leaves the `globalTheoremStrings` dedup set (a later
/// legitimate prover of the same statement may re-append), its
/// producer slot leaves `globalTheoremProducers`, and matching
/// `fullTheoremList` rows leave too. Runs at the post-join seam right
/// after `drainUpdateGlobalSingleChapter`, once per iteration — the flags
/// set by this iteration's worker phases are all visible, and the
/// list is only consumed post-prove. Primed `__contradiction__` LBs
/// never carry the flag (their contradiction is the intended
/// reductio), so disproof theorems are never retracted.
///
/// Known residual (flagged, not silent): the theorem's earlier
/// broadcast as a hash rule into other LBs is NOT undone here — the
/// retraction covers the proved-theorem record and everything derived
/// from it downstream of the run (artifacts, exports, next-batch
/// pools).
///
/// @return Nothing.
/// @invariant Deterministic: the sweep order is list order; the
///            verdict is a pure function of producer flags.
/// @see appendGlobalTheorem, hasContradictedAncestor,
///      `Memory::mainContradiction`.
void revertVacuousGlobalTheorems();

public:
/// @brief Post-prove vacuity certificate: same-premise-chain theorems
///        with contradictory heads certify the chain unsatisfiable —
///        every theorem on that chain is a vacuous truth and leaves
///        the pool.
///
/// @details
/// Two proved theorems whose premise chains are the same multiset and
/// whose heads are `X` and `negate(X)` jointly refute their shared
/// premise set: with unsatisfiable premises both implications — and
/// every other theorem on the same chain — hold vacuously (ex falso)
/// and carry no mathematical content. No single LB need ever hold the
/// contradiction internally (the two heads can close in separate
/// conjecture LBs), so the vacuous-premise flag cannot catch this
/// class; the pair itself is the certificate — refutation-based
/// inconsistency detection of the hypothesis set, the vacuity-
/// detection judgment of model checking applied to the proved pool.
///
/// Runs single-threaded in `fullRun` after the prove pass and BEFORE
/// compression and every proved-theorem save, so a vacuous theorem
/// never reaches `theorems.txt` / `compiled_theorems.txt`, never
/// feeds a later batch, and never receives a chapter. Classified
/// theorems leave `globalTheoremList` (+ the aligned producer vector,
/// the dedup string set, and `fullTheoremList`) and are returned for
/// the `vacuous_theorems.txt` artifact.
///
/// @return The classified theorem strings, in pool order.
/// @invariant Deterministic: grouping keys are sorted premise chains;
///            removal order is pool order.
/// @see revertVacuousGlobalTheorems — the producer-flag certificate;
///      the two seed sets feed the same artifact and the post-batch
///      taint-closure filter.
std::vector<std::string> classifyVacuousPremisePairs();

private:
/// @brief Complete a tier row's provenance: deposit the `theorem`
/// history line for every registered theorem its LB chain cites
/// without one.
///
/// @details
/// A proved-theorem citation terminates a chapter walk through a
/// `theorem`-tagged origin row, which normally arrives with the
/// theorem's broadcast two iterations after registration. A
/// proved-not-broadcast closure characteristically kills its LB in
/// the same window BOTH its goals close (the pre-split twin
/// pattern), so the sibling theorem's broadcast never lands and the
/// tier chapter's walk would find a dangling citation — the
/// `no origin found` assert. Per Rule 16, the missing history line
/// is exactly that: missing documentation of a state the run
/// reached — this seam completes it at tier registration.
///
/// Walks the producer's LB chain to root; in each LB, scans the
/// decoded origin map for dependency expressions at `main` that are
/// registered theorems (`globalTheoremStrings`) with no origin row
/// of their own in that LB, and deposits `("theorem", [])` for
/// each. Runs on the single-threaded phase-4 drain; each LB is made
/// resident through the claim door when the steward is live (the
/// unit-test harness runs steward-less with resident LBs — both
/// contexts are defined, not a fallback).
///
/// @param producer The tier row's producing (leaf) LB.
/// @return Nothing.
/// @invariant Deposits only the `theorem` tag with zero
///            dependencies, only at `main`, only for
///            already-registered theorem strings — pure provenance
///            completion, never a proof input (I-44).
/// @see `updateGlobalSingleChapter` — sole caller (the tier branch).
void repairTierCitationOrigins(Memory* producer);

private:
/// @brief Settle a directly-proved theorem: run the goal-lifecycle
        ///        walks (chain deactivation, twin and auxiliary retirement),
        ///        then — only when `registerGlobally` — register the theorem
        ///        globally (reformulation, `appendGlobalTheorem`, broadcast,
        ///        compaction staging).
        ///
        /// @details
        /// The lifecycle section is level-free by contract
        /// (D-278): a proved goal must close
        /// and release its LB chain regardless of premise-level coverage.
        /// `registerGlobally` carries the sealed `allLevelsInvolved` verdict
        /// from the discharge site; when false the function returns after the
        /// two `deactivateUnnecessary` walks and the theorem is deliberately
        /// lost — an emitted implication must have consumed every premise
        /// level.
        ///
        /// @param theorem          The proven theorem's full-bind text.
        /// @param coreId           Sealing core (drain replay determinism).
        /// @param producer         The sealing LB (vacuity gate upstream).
        /// @param registerGlobally Sealed level verdict; false = lifecycle
        ///                         only, no registration.
        /// @return Nothing.
        /// @see drainUpdateGlobalSingleChapter — the sole caller;
        ///      appendGlobalTheorem — the registration door.
void updateGlobalSingleChapter(const std::string& theorem, int coreId,
            const Memory* producer, bool registerGlobally);

public:
// analyze_expressions.cpp
// Thread-safe: locks globalDependencies briefly, and locks globalTheoremList only when pushing.
/// @brief Drain the per-thread `updateGlobalTuples` queue into the
/// global theorem list / dependency tracker.
///
/// @details
/// Post-`pool.join()` collector for the indirect (auxy-indexed) path.
/// During the parallel hash burst, workers cannot write to
/// `globalTheoremList` directly per
/// [I-28](../../docs/agentic_swdd/30_invariants.md#i-28); they enqueue tuples
/// `(auxyIndex, levelsMask, coreId)` instead. This function
/// is the single drain point that locks `theoremListMutex` /
/// `dependenciesMutex` once and folds every queued tuple into the
/// global side-tables.
///
/// The level verdict of an induction is a union: the triad promotes once
/// both its step and base closures are in and their levels, together with
/// the levels of the typing fact the induction consumes, cover every
/// premise level of the conjecture
/// (`ScopeLifecycle::inductionPremiseLevels`,
/// D-516). Only a triad that passes every gate
/// switches the conjecture's other blocks off (`deactivateUnnecessary`).
///
/// A winning triad stopped only by the missing typing fact is recorded on
/// its step win, and the barrier re-queues it once the fact arrives
/// (D-517).
///
/// @param auxyIndex   Auxiliary index identifying the theorem.
/// @param levelsMask  The levels the closure's proof used, one bit per
///                    level; 0 for a re-queued promotion.
/// @param coreId      Producing thread id (purely informational).
/// @invariant [I-28](../../docs/agentic_swdd/30_invariants.md#i-28) — drain runs
///            only after `pool.join()`.
/// @see `updateGlobalSingleChapter` — direct (non-auxy) counterpart.
void updateGlobal(int auxyIndex, uint64_t levelsMask, int coreId);

public:
/// @brief Deterministic total order over two producer LBs — byte
        ///        lexicographic over the leaf-to-root `exprKey` chain.
        ///
        /// @details
        /// Walks both `parentMemory` chains in lockstep comparing the interned
        /// `exprKeyView` bytes level by level; the first differing level
        /// decides, a chain that is a strict prefix of the other sorts first,
        /// and the walk short-circuits to equal when the chains converge on a
        /// shared node. The chain bytes are a pure function of proof state —
        /// unlike a worker slot or an allocation address — so any drain that
        /// orders records by producer stays byte-identical across runs. The
        /// spans alias the never-deloaded skeleton interner, so the compare is
        /// legal for cold and discharged LBs and allocates nothing.
        ///
        /// @param a First producer LB (may be `nullptr` for producer-less rows).
        /// @param b Second producer LB (may be `nullptr`).
        /// @return Negative when `a` sorts before `b`, 0 when equal, positive
        ///         otherwise.
        /// @invariant Two DISTINCT LBs never share a byte-equal full chain —
        ///            asserted, the full-chain identity contract.
        /// @see updateGlobalSingleChapterLess — the consumer;
        ///      ProofBarrier::settleStepWins — orders the closed step blocks;
        ///      `Memory::exprKeyView`.
static int compareProducerChains(const Memory* a, const Memory* b);

private:
/// @brief Total order over `UpdateGlobalSingleChapterRec`: theorem bytes,
        ///        then level verdict (true first), then producer chain.
        ///
        /// @details
        /// Every key is a pure function of proof state, so the drain replays
        /// `updateGlobalSingleChapter` in a byte-identical order across runs. The
        /// verdict ranks BEFORE any producer identity because the sink's
        /// string dedup makes the first-drained record's method the
        /// registration: when one theorem seals from both a level-complete
        /// and a level-poor route in the same iteration, the level-complete
        /// record must win, else the theorem lands in the
        /// proved-not-broadcast tier although a full-level proof closed. The
        /// dispatch `coreId` is banned from the order — it is a scheduling
        /// race outcome, and ordering same-theorem records by it made the
        /// registration method (and, for a level-poor pair, the tier
        /// chapter's producer) flip run to run. Static — reads only the two
        /// records.
        ///
        /// @param a Left record.
        /// @param b Right record.
        /// @return Negative when `a` sorts before `b`, 0 when equal, positive
        ///         otherwise.
        /// @see drainUpdateGlobalSingleChapter — the replay; updateGlobalSingleChapter — the
        ///      sink; compareProducerChains — the producer tiebreak.
static int updateGlobalSingleChapterLess(const UpdateGlobalSingleChapterRec& a,
                                          const UpdateGlobalSingleChapterRec& b);

public:
/// @brief Drain the staged proven single-chapter theorems into
        ///        `updateGlobalSingleChapter`, sorted, after `pool.join()`.
        ///
        /// @details
        /// The single post-join replay of `updateGlobalSingleChapterPages`: gather the
        /// record pointers, index-sort on `updateGlobalSingleChapterLess` (theorem
        /// bytes, level verdict true-first, producer chain — every key a pure
        /// function of proof state), materialize each
        /// theorem's `std::string` at the still-heap `updateGlobalSingleChapter` edge and
        /// replay, then `seal` / `freePages` / `reset` the optional. Runs
        /// single-threaded; `updateGlobalSingleChapterPages` must be `emplace`d by
        /// `proveKernel`.
        ///
        /// @see updateGlobalSingleChapterLess; drainDeferredAncestorAdmissions — the
        ///      sibling sealed-page drain.
void drainUpdateGlobalSingleChapter();

private:
/// @brief Produce reformulated variants of a theorem — alternative
/// equivalent expressions that may be easier for the prover to absorb.
///
/// @details
/// Walks the theorem looking for reformulation opportunities:
/// - swap commutative-operator argument order,
/// - apply known operator-equivalences from `coreExpressionMap`,
/// - extract sub-implications that are independently reformulable.
///
/// Each successful reformulation becomes a `(tag, expression)` pair
/// in `outTheorems`. The tag distinguishes the reformulation type
/// (`reformulated from`, `mirrored from`, `incubator back reformulation`,
/// etc.). The tag is load-bearing: `reformulated from` is the
/// real-math (sound) provenance, while `incubator back reformulation`
/// is an incubator-only artefact that is unsound and must NEVER be
/// folded into a `reformulated from`. Downstream consumers must
/// preserve this distinction.
///
/// @param theorem      Source theorem text.
/// @param outTheorems  Output vector. Each pair is
///                     `(reformulation tag, reformulated text)`.
/// @return True iff at least one reformulation was produced.
bool reformulateTheorem(const std::string& theorem, std::vector<std::pair<std::string, std::string>>& outTheorems);

public:
/// @brief Record a mail-broadcast implication for deferred, single-threaded
/// compaction (D-76, Option A).
///
/// @details
/// The eight `updateGlobalSingleChapter` / `updateGlobal` broadcast sites call this
/// instead of compiling inline. Compiling inline is unsound: the
/// contradiction path reaches `updateGlobalSingleChapter` via the parallel
/// `addExprToMemoryBlock(... coreId ...)` worker call, and
/// `compileImplicationToCompact` mutates process-global `implCounter` /
/// `compiledExpressions` / `compilationReuse.entries()` with no lock — a data
/// race that made the compaction non-injective and the run
/// non-deterministic (violates I-28). This call only appends to
/// `pendingCompactionQueue` under `pendingCompactionMutex` (cheap,
/// parallel-safe — no global compile state touched). A single-threaded pass
/// after `pool.join()` sorts and drains the queue, performing the actual
/// compile + mail deposit deterministically.
///
/// @param original The original expanded implication string as it enters the
///        mail channel at the call site.
/// @param kySize `ky.size()` at the call site (0 for premise-chain-less
///        callers); participates only in the drain's deterministic sort —
///        the deposited level set is always empty (see the drain's
///        level-set comment).
/// @param coreId The broadcasting core id (no longer routes anything; the
///        deferred pass merges the compact deposit into the root's mailOut).
/// @return void.
/// @invariant Touches only `pendingCompactionQueue` (under
///            `pendingCompactionMutex`); never the global name-allocation
///            state.
/// @see compileImplicationToCompact, excludeRepetitions
void recordPendingCompaction(const std::string& original, int kySize, int coreId);

private:
// Appends a newly proved theorem to the persistent file storage.
void saveProvedTheorems();

public:
// Saves only the essential (compressor-surviving) theorems to theorems.txt.
        // This is the sole cross-batch propagation path: only essential theorems
        // are inherited by subsequent batches.
void saveProvedTheoremsFiltered(const std::vector<std::string>& essentialTheorems,
                                       const std::unordered_set<std::string>& externalTheorems = {});

public:
/// @brief Merge an in-run seam's theorem-origin documentation mail into the
/// root's mailOut.
///
/// @details
/// The in-run construction seams broadcast through the deferred-compaction
/// drain, whose compilation/expansion records form a two-node cycle with no
/// leaf. Their `("theorem", [])` origins at `main` travel in one documentation
/// `Mail` merged here into the root's mailOut, shipped by the next commit
/// barrier together with the drain's compacts: the documentation every other
/// proved theorem's broadcast carries (`updateGlobalSingleChapter`).
///
/// The root's mailOut is deloadable: the claim handshake reloads it when the
/// steward is live, `ensureLoadedForRead` in the steward-less unit-test context
/// (the `repairTierCitationOrigins` precedent). A mail without origin lines
/// documents nothing this window and leaves the root untouched.
///
/// @param documentation Origin lines only; statements are asserted absent.
/// @return Nothing.
/// @invariant Single-threaded phase-4 barrier seam only.
/// @see `constructOrTheoremsInRun`, `constructOrEliminationInRun`.
void mergeTheoremOriginMailIntoRoot(const Mail& documentation);
private:
std::unordered_set<std::string>& orInRunCompanions;

AnchorScopeQueries& anchorScopeQueries;
Memory& body;
DefinitionExpansion& definitionExpansion;
const CompiledDefinitionRegistry& definitions;
std::mutex& dependenciesMutex;
std::vector< std::tuple<std::string, std::string, std::string, std::string >>& fullTheoremList;
Dependencies& globalDependencies;
std::vector< std::tuple<std::string, std::string, std::string, std::string >>& globalTheoremList;
std::vector<const Memory*>& globalTheoremProducers;
std::unordered_set<std::string>& globalTheoremStrings;
std::unordered_set<std::string>& orInRunScannedRows;
const ProverParameters& parameters;
std::mutex& pendingCompactionMutex;
std::vector<std::tuple<std::string, int, int>>& pendingCompactionQueue;
ProofHistory& proofHistory;
std::vector<std::string>& retractedVacuousTheorems;
ScopeLifecycle& scopeLifecycle;
SimpleMapStore& simpleMapStore;
std::unique_ptr<MemorySteward>& steward;
StructuralCompiler& structuralCompiler;
std::mutex& theoremListMutex;
std::optional<SealedPageSet>& updateGlobalSingleChapterPages;
OrCandidateProgress& orCandidateProgress;

public:
/// @brief Read the published theorem records.
/// @details Returns the existing collection without permitting external mutation.
/// @return Read-only records, valid for the lifetime of the publisher owner.
const auto& published() const { return globalTheoremList; }
/// @brief Read the retained complete theorem records.
/// @details Returns the existing collection without permitting external mutation.
/// @return Read-only records, valid for the lifetime of the publisher owner.
const auto& archived() const { return fullTheoremList; }
/// @brief Read theorem strings retracted for vacuous premises.
/// @details Returns the existing collection without permitting external mutation.
/// @return Read-only records, valid for the lifetime of the publisher owner.
const auto& retractions() const { return retractedVacuousTheorems; }
/// @brief Sort published theorem records for postprocessing.
/// @details Uses tuple comparison and the original post-proof sort boundary.
/// @return No value.
void sortPublished() { std::sort(globalTheoremList.begin(), globalTheoremList.end()); }
/// @brief Retain one theorem record for later proof export.
/// @details Appends in the caller-selected order to the existing complete result collection.
/// @param record The theorem record selected by postprocessing.
/// @return No value.
void archive(const std::tuple<std::string, std::string, std::string, std::string>& record) { fullTheoremList.push_back(record); }
/// @brief Select and sort the theorem records for proof-graph export.
/// @details Uses the complete archived list when populated and the published list otherwise, preserving the original selection and tuple sort at the export boundary.
/// @return Read-only view of the selected, sorted records.
const auto& sortedForProofGraph() {
    auto& listForGraph = fullTheoremList.empty() ? globalTheoremList : fullTheoremList;
    std::sort(listForGraph.begin(), listForGraph.end());
    return listForGraph;
}
/// @brief Retain the original compression survivors and graph-support rows.
/// @details Uses the existing predicate: keep proved-not-broadcast rows and merge variants, then retain ordinary rows exactly when they survived compression.
/// @param survivorSet Theorem strings retained by compression.
/// @param mergeVariantRows Graph-support variants excluded from the compression probe.
/// @return No value.
void retainCompressionSurvivors(const std::unordered_set<std::string>& survivorSet, const std::set<std::string>& mergeVariantRows) {
                auto& gtl = globalTheoremList;
                gtl.erase(std::remove_if(gtl.begin(), gtl.end(),
                    [&](const std::tuple<std::string, std::string, std::string, std::string>& t) {
                        // Tier rows and merge variants never entered the
                        // probe, so absence from the survivor set says
                        // nothing about them — keep (graph support).
                        if (std::get<1>(t) == "proved not broadcast") return false;
                        if (mergeVariantRows.count(std::get<0>(t))) return false;
                        return survivorSet.find(std::get<0>(t)) == survivorSet.end();
                    }), gtl.end());
}
};
}
#include "../proof/or_candidate_progress.hpp"
namespace gl {
}
