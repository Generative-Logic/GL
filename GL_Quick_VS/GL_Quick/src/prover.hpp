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
#include "proof/prover_mail_usage.hpp"
#include "proof/proof_run_reporter.hpp"
#include "proof/prover_initialization.hpp"
#include "proof/phase2_backend.hpp"
#include "proof/phase2_executor.hpp"
#include "proof/proof_barrier.hpp"

#include "proof/cached_proof_stack.hpp"
#include "proof/theorem_usage_graph.hpp"
#include "proof/proof_reconstructor.hpp"
#include "proof/proof_graph_writer.hpp"

#include "proof/theorem_publisher.hpp"
#include "proof/proof_grid_builder.hpp"
#include "proof/or_theorem_builder.hpp"
#include "proof/or_candidate_progress.hpp"
#include "proof/or_elimination_builder.hpp"
#include "proof/counterexample_filter.hpp"
#include "inference/encoded_expression_normalizer.hpp"

#include "logical_block/admission_memory.hpp"
#include "logical_block/integration_processor.hpp"
#include "logical_block/contradiction_demand.hpp"
#include "logical_block/implication_subproofs.hpp"
#include "logical_block/rejected_source_replay.hpp"
#include "inference/hash_rule_installer.hpp"
#include "logical_block/compact_rule_processor.hpp"
#include "logical_block/statement_memory.hpp"
#include "logical_block/equivalence_processor.hpp"
#include "logical_block/equality_propagation.hpp"
#include "logical_block/equivalence_admission_rewriter.hpp"
#include "logical_block/watch_processor.hpp"
#include "logical_block/case_ledger.hpp"
#include "logical_block/case_processor.hpp"
#include "logical_block/disintegration_processor.hpp"
#include "logical_block/or_implication_processor.hpp"
#include "logical_block/proof_history.hpp"
#include "logical_block/unary_integration_processor.hpp"
#include "logical_block/goal_processor.hpp"
#include "logical_block/scope_lifecycle.hpp"
#include "logical_block/logic_block_operations.hpp"
#include "logical_block/block_mail_processor.hpp"
#include "logical_block/anchor_scope_queries.hpp"
#include "logical_block/equivalence_queries.hpp"
#include "logical_block/block_key_reader.hpp"
#include "logical_block/phase13_timing_slot.hpp"
#include "logical_block/watch_ledger.hpp"


#include "project_paths.hpp"
#include "infra/hashburst_dump.hpp"
#include "memory.hpp"
#include "filter.hpp"
#include "compilation_state.hpp"
#include "compiler/expression_parser.hpp"
#include "compiler/compilation_normalizer.hpp"
#include "compiler/compilation_persistence.hpp"
#include "compiler/compiled_definition_registry.hpp"
#include "inference/hash_key_builder.hpp"
#include "inference/implication_multiplier.hpp"
#include "inference/binding_partitions.hpp"
#include "inference/hash_rule_index.hpp"
#include "inference/hash_rule_classifier.hpp"
#include "inference/hash_request_generator.hpp"
#include "inference/hash_rule_evaluator.hpp"
#include "inference/burst_sink.hpp"
#include "logical_block/logic_block_identity.hpp"
#include "logical_block/statement_knowledge.hpp"
#include "logical_block/deferred_ancestor_admission.hpp"
#include "logical_block/update_global_single_chapter_record.hpp"
#include "logical_block/or_implication_refuter.hpp"
#include "logical_block/or_implication_selection.hpp"
#include "logical_block/watch_bridge.hpp"
#include "logical_block/canonical_form_cache.hpp"
#include "logical_block/compact_canon_pair.hpp"
#include "logical_block/numeric_less.hpp"
#include "logical_block/hypo_scope_parse.hpp"
#include "logical_block/implication_subproof_outcome.hpp"
#include "logical_block/collected_record.hpp"
#include "logical_block/collected_arena.hpp"
#include "logical_block/new_variable_store.hpp"
#include "logical_block/witness_meta_store.hpp"
#include "logical_block/rejection_record.hpp"
#include "logical_block/rejection_store.hpp"
#include "logical_block/scanned_token_runs.hpp"
#include "logical_block/equivalence_span_pair.hpp"
#include "logical_block/equivalence_rewrite.hpp"

#include "logical_block/admission_policy.hpp"
#include "logical_block/scope_type_resolver.hpp"
#include "logical_block/canonicalizer.hpp"
#include "logical_block/canonical_template_matcher.hpp"
#include "logical_block/rejected_source_memory.hpp"
#include "logical_block/integration_instruction_builder.hpp"
#include "logical_block/unary_rule_memory.hpp"
#include "logical_block/admission_marker_builder.hpp"
#include "compiler/existence_rule_compiler.hpp"
#include "compiler/implication_builder.hpp"
#include "compiler/or_rule_compiler.hpp"
#include "compiler/structural_compiler.hpp"
#include "compiler/supporting_rule_compiler.hpp"
#include "compiler/integration_rule_compiler.hpp"
#include "compiler/compilation_reuse.hpp"
#include "compiler/definition_expansion.hpp"
#include "proof/worker_context.hpp"
#include "proof/or_candidate.hpp"
#include "proof/variable_copy_detector.hpp"
#include "infra/rt_tracker.hpp"
#include "infra/hashburst_dump.hpp"
#include "memory_infra/steward.hpp"
#include "memory_infra/lb_store.hpp"
#include "mail_log.hpp"
#include "simple_map_store.hpp"
#include "memory_infra/or_implication_match_index.hpp"
#include "memory_infra/temporary_witness_store.hpp"
#include <array>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <memory>
#include <mutex>
#include <atomic>
#include <optional>
#include <unordered_set>

namespace gl {




    /// @brief Coordinate proof phases and retain stable per-batch owners.
    /// @details Initialization, compilation, logical-block algorithms, phase-two execution,
    /// result drains and export have focused operation classes. Prover selects and joins
    /// phases through those interfaces; its stores and helpers are private. Stable root
    /// blocks, counters, scratch services and serialization ownership remain unchanged.
    /// @invariant Parallel phases do not mutate sibling or ancestor logical blocks;
    /// cross-block publication runs at the original joined barriers.
    /// @see ProverInitialization, LogicBlockOperations, Phase2Executor, ProofBarrier.
    namespace tests { struct ProverFixture; }
    class Prover {
        friend struct tests::ProverFixture;
    private:
        ProverParameters parameters;
        Phase2Backend phase2Backend;
        // Filter artifact directory; the batch driver redirects shortcut output.
        std::filesystem::path filteredConjecturesDirectory =
            gl::projectRoot() / "files" / "theorems";
        // Smaller profile for explicit test fixtures; production FTA replay uses full.
        bool phase2ShortcutProfile{ false };
        // Observation-only wall time over the complete Phase 2 scheduler window.
        // Each proveKernel call adds exactly one barrier-to-barrier measurement;
        // no proof branch reads either value.
        double phase2CumulativeSeconds{ 0.0 };
        uint64_t phase2MeasuredIterations{ 0 };
        // Burst stamp of the CUDA Phase 2 measurement each active LB records at
        // the end of its phase 1 (Memory::phase2MeasurementEpoch); advanced once
        // per proveKernel call, before phase 1.
        uint64_t phase2MeasurementEpoch{ 0 };



        // Observation-only GPU filter/sort capacity census. The Phase 2 sweep
        // resets these counters before dispatch; request generators and stump
        // producers add their actual examined and retained statement counts.
        // No proof branch reads them.






#if PHASE13_DEEP_TIMING




        int64_t* phase13TimingRows{ nullptr };
        unsigned phase13TimingWorkers{ 0 };
#endif



        // Root logical block for theorem proving.
        Memory body;
        // Root for counter-example filtering memory
        Memory ceBody;
        std::vector< std::tuple<int, uint64_t, int>> updateGlobalTuples;
        std::vector< std::tuple<std::string, std::string, std::string, std::string >> globalTheoremList;
        // Companion membership set for globalTheoremList — the sink dedup.
        // Mutated ONLY through appendGlobalTheorem (and the export rebuild's
        // paired clear); insert-if-absent on the theorem string, first
        // emission wins.
        std::unordered_set<std::string> globalTheoremStrings;
        // Producer attribution parallel to globalTheoremList (same mutex, same
        // append door, cleared with the export rebuild): the grid LB whose
        // discharge/promotion emitted the row, or nullptr for producer-less
        // paths (or-theorem construction, debug export). Grid LB objects are
        // address-stable for the whole batch (I-109; discharged LBs persist
        // for the chapter export), so the pointer stays readable. Read by
        // revertVacuousGlobalTheorems: a producer whose self-or-ancestor
        // carries mainContradiction retracts its rows.
        std::vector<const Memory*> globalTheoremProducers;
        // Theorems retracted by the vacuous-reversion sweep this batch, in
        // retraction order — the seed set the post-batch vacuity
        // classification writes to the `vacuous_theorems.txt` artifact and
        // the taint-closure filter reads.
        std::vector<std::string> retractedVacuousTheorems;
        // All theorems ever proved (unpruned) — used for proof graph generation
        // so proof stacks can reference non-essential theorems
        std::vector< std::tuple<std::string, std::string, std::string, std::string >> fullTheoremList;
        // Survivors of the most recent compression run inside analyzeExpressions.
        // Includes old proved theorems, new theorems, and external theorems that
        // remain essential. Used by run_modes.cpp to partition externals and to
        // rewrite theorems.txt correctly across multi-batch runs.
        std::vector<std::string> lastCompressionSurvivors;
        std::vector<Memory*> inductionMemoryBlocks;
        // Contradiction LBs staged for deactivation by the discharge sites
        // (worker phase, I-28: cross-LB isActive writes are deferred) and
        // drained post-join by drainPendingTwinDeactivations. Deactivation
        // is idempotent and commutative — no ordering obligations.
        std::vector<Memory*> pendingTwinDeactivations;
        // Premise LBs whose `main` turned contradictory (vacuous premises),
        // staged by dischargeContradiction's normal-LB branch (worker
        // phase, I-28) and drained post-join by drainSubtreeDeactivations:
        // every descendant of a staged root is deactivated — nothing sound
        // and non-vacuous can be proved below an inconsistent premise set.
        std::vector<Memory*> pendingSubtreeDeactivations;


        /// @brief Per-`proveKernel`-iteration staging page set for deferred
        ///        ancestor admission seeds; workers seal + `appendRecord` under
        ///        `deferredAncestorAdmissionsMutex`, the post-join drain sorts +
        ///        replays + `freePages` (I-28 detect-and-defer trial). `emplace`d
        ///        fresh (Filling) at each `proveKernel` top, `reset` at the drain.
        ///        See `drainDeferredAncestorAdmissions`.
        std::optional<SealedPageSet> deferredAncestorPages;



        /// @brief Per-`proveKernel`-iteration staging page set for proven single-chapter
        ///        theorems; `dischargeToBeProved` / `dischargeContradiction` seal
        ///        + `appendRecord` under `updateGlobalSingleChapterMutex`, the post-join
        ///        drain index-sorts by `(theorem bytes, level verdict,
        ///        producer chain)` + replays into
        ///        `updateGlobalSingleChapter` + `freePages`. `emplace`d fresh (Filling)
        ///        at each `proveKernel` top, `reset` at the drain. See
        ///        `drainUpdateGlobalSingleChapter`.
        std::optional<SealedPageSet> updateGlobalSingleChapterPages;







        // pendingAncestorOrigins (D-39 deferred ancestor-origin queue) — RETIRED
        // by D-51. The mechanism walked memoryBlock.parentMemory and
        // wrote contradiction records into every ancestor's exprOriginMap; D-51
        // localised the contradiction record to the __contradiction__ LB and the
        // queue + drain loop never fired again. Struct, field, mutex, drain loop,
        // and filter.cpp reset are all gone.

        // Keyed by the packed (i, j) pair — (int64_t(i) << 32) | uint32_t(j),
        // the house packed-key pattern (packStatementKey precedent). Both
        // halves are positive counts, so the packing is injective and
        // order-preserving; construction-built config state, never iterated
        // in-tree (point find only), so no observable order exists.
        std::map<int64_t, std::vector<std::vector<int> > > allMappingsAna;
        std::map<int, std::vector<std::vector<int> > > allPermutationsAna;
        Dependencies globalDependencies;
		std::vector<ContradictionItem> contradictionTable;

		/// @brief Symmetric mirror-partner index for the CE filter's
		///        mirror-refutation pass (D-229).
		///
		/// @details
		/// Filled single-threaded in `runCeFilterOnly` from
		/// `files/theorems/mirror_pairs.txt` via `loadMirrorPairs` (only when
		/// `parameters.mirror_refutation` is on), read-only afterwards by
		/// `applyMirrorRefutations` inside each `filterConjecturesWithCE`
		/// pass. Keys and values are verbatim `conjectures.txt` lines; both
		/// directions of every pair are present.
		MirrorPartnerMap mirrorPartnerMap;

		bool ceFilteringActive = false;




        CompiledDefinitionRegistry definitions;
        IntegrationInstructionBuilder integrationBuilder{definitions};
        UnaryRuleMemory unaryMemory{definitions};
        Canonicalizer canonicalizer{parameters};
        CanonicalTemplateMatcher canonicalTemplates{canonicalizer};
        RejectedSourceMemory rejectedSources{parameters, canonicalizer, canonicalTemplates};
        HashRuleClassifier hashRuleClassifier{definitions};
        ImplicationMultiplier implicationMultiplier{definitions, parameters, ceFilteringActive};
        AdmissionPolicy admissionPolicy{definitions, parameters};
        AdmissionMarkerBuilder admissionMarkers{definitions, admissionPolicy, parameters, allPermutationsAna};
        ce::AnchorInfo anchorInfo;
        LogicBlockIdentity blockIdentity{parameters, anchorInfo};
        HashRequestGenerator requestGenerator{parameters};
        HashRuleEvaluator ruleEvaluator{parameters, definitions, admissionPolicy, blockIdentity};
        std::string anchorID_;

        std::vector<Memory*> permanentBodies;
        std::vector<Memory*> permanentBodiesCE;
        unsigned logicalCores;

        // New mail system (D-137), STATIFIED onto the dedicated
        // mail pool (D-141): per-LB pull logs that replace
        // the boxes/sendMail/smashMail broadcast. Each LB's sent mail is stored
        // once in its own append-only log; descendants pull from their ancestors
        // with a per-(recipient, ancestor) cursor. mailArena is the "local memory
        // manager" — one LbArena drawn from the never-deloaded mail pool that hands
        // pages to the three cold MailLog containers; mailDirty satisfies their
        // constructors but is never read (no deload image). Declared before mailLog
        // so the containers destruct before the arena whose pages they hold.
        LbArena mailArena{ &mailMemory() };
        DirtyState mailDirty = DirtyState::Clean;
        MailLog mailLog{ &mailArena, &mailDirty };

        // Mail-history telemetry. The grid-build scan sets the deterministic
        // dormant count once all LBs exist. Rolling mode is fixed for the
        // execution batch and is true only when that count is zero; otherwise
        // full history remains available for dormant catch-up.
        int dormantLogicBlocksAtGridBuild = 0;
        bool rollingMailHistoryEnabled = false;

        // Mail attribution telemetry. mailIn is concurrent during phase 1, so
        // its simultaneous current/peak counters are atomic. Deloadable mailOut
        // contributes to the MAIN pool, not the mail pool; its always-resident
        // per-LB live-byte snapshots are summed after joins without cold loads.
        std::atomic<int64_t> routingMailInBlocksInFlight{ 0 };
        std::atomic<int64_t> peakRoutingMailInBlocks{ 0 };
        int64_t peakDeloadableMailOutBytes = 0;

        // Quiescent-burst skip (D-194). `warmUpPhase` is set
        // around the warm-up prove() call so proveKernel forces every active LB
        // into the sweep during warm-up (belt-and-suspenders — warm-up bursts are
        // productive anyway). `lastSweptCount` / `lastSkippedCount` are the
        // previous kernel's post-skip sweep size and the number of active LBs it
        // skipped — telemetry only (Rule 16 / I-44), printed on the burst dt line.
        bool warmUpPhase = false;
        // The number of the batch's current proof burst, 1-based and counted
        // across the warm-up and the main iterations; the counterexample
        // filter's bursts on its clones do not count. The induction
        // usefulness test closes an uncaught test after
        // `InductionUsefulness::kTestBursts` of them
        // (D-521).
        int proofBurstNumber = 0;
        int lastSweptCount = 0;
        int lastSkippedCount = 0;

        // LB object store (D-150): the Memory node shells live
        // in fixed-size slots carved from the never-deloaded LB-body pool instead
        // of the malloc heap. Bound here (geometry from sizeof(Memory), where the
        // type is complete); draws no block until the first LB is created, so the
        // bind precedes initLbMemory in the constructor harmlessly (lazy, the
        // mailArena precedent). Slots are address-stable for the LB's whole life
        // (Memory is non-relocatable and every Memory* relies on it).
        LbStore lbStore{ &lbMemory(), sizeof(Memory), alignof(Memory) };

        // LB-tree routing edges (D-152), STATIFIED onto
        // the never-deloaded LB-body pool: SimpleMapStore replaces the per-LB heap
        // member Memory::simpleMap. Two instances — the main proof tree (this->body)
        // and the CE-filter tree (ceBody) can coexist, so each rides its own store,
        // cleared at its own teardown (destroyGrid / filterConjecturesWithCE). Both
        // arenas draw from the never-deloaded lbMemory() pool (the skeleton-interner
        // precedent); the dirty flags satisfy the container constructors but are
        // never read (no deload image). Each arena is declared before its store so
        // the cold containers destruct before the arena whose pages they hold.
        LbArena simpleMapArena{ &lbMemory() };
        DirtyState simpleMapDirty = DirtyState::Clean;
        SimpleMapStore simpleMapStore{ &simpleMapArena, &simpleMapDirty };
        LbArena ceSimpleMapArena{ &lbMemory() };
        DirtyState ceSimpleMapDirty = DirtyState::Clean;
        SimpleMapStore ceSimpleMapStore{ &ceSimpleMapArena, &ceSimpleMapDirty };

        // Scratch arena for the discharge-path compaction (LbArena::compactPages):
        // drawn from the never-deloaded LB-body pool, NOT the deloadable pool it
        // compacts, and exclusive to the main prove thread (the steward owns its
        // own; the two reshuffle paths can run concurrently).
        LbArena dischargeScratch{ &lbMemory() };


        // Deferred implication-compaction queue (D-76,
        // Option A). Filled mutex-guarded during the parallel proof phase by
        // recordPendingCompaction; drained single-threaded after pool.join() so
        // compileImplicationToCompact's global implCounter / compiledExpressions /
        // compilationReuse.entries() mutation is deterministic and never races the
        // parallel addExprToMemoryBlock->updateGlobalSingleChapter(coreId) path (I-28).
        // Tuple: (original implication string, ky size, coreId).
        std::vector<std::tuple<std::string, int, int>> pendingCompactionQueue;

        // Statification: the deload registry (ordinal -> full LB chain)
        // lives on GlobalMemoryManager (the unit tests dump without an
        // Prover in scope), rewritten to .deload/registry.txt
        // at each kernel end and cleared at batch start. Reload uses
        // Memory::deloadFiles, never the registry.

        // Statification: LBs that left the active set, awaiting their
        // once-per-lifecycle discharge dump — appended at the kernel's
        // end-of-iteration sweep in deterministic `active` order, flagged
        // dischargedForever on entry, drained FIRST under pool pressure
        // (cheapest relief: frozen content, blocks never needed again).
        // A run that never crosses the pressure mark drains nothing and
        // writes zero deload files (D-156). Reset at
        // batch start with the manifest; entries dangle harmlessly across
        // destroyGrid because nothing reads the list outside proveKernel.
        std::vector<Memory*> pendingDischarge;

        // The background memory custodian (D-159).
        // Alive exactly for the span of one prove() call — created and
        // started before the first kernel iteration, quiesced + stopped
        // after the last (the StewardScope guard inside prove()). CUDA CE
        // batches use the same prove scope; destroyGrid never sees it non-null.
        // proveKernel's barrier asserts its presence.
        std::unique_ptr<MemorySteward> steward;

        // Active-LB eviction plan (D-175): victims
        // selected at the kernel barrier when quiesced consumption sits in
        // the steward band — biggest-first by blocks held, ties broken by
        // active-order index (exprKey is NOT unique, Rule 12), enough to
        // project consumption down to the stop watermark. Lifecycle:
        // written at the barrier of iteration N, claims set + handed to
        // the steward at the entry of kernel N+1, executed during N+1's
        // phase 1 (steward ahead of the workers, worker self-service on
        // a lost claim), folded + cleared at N+1's barrier. Cleared at
        // prove() scope start and batch start — plans never cross scopes.
        std::vector<Memory*> stewardEvictionPlan;
        mutable std::mutex pendingCompactionMutex;

        std::map<std::string, CachedProofStack> cachedProofStacks;

        /// @brief Cross-chapter theorem-usage graph of the running chapter
        ///        export — see `TheoremUsageGraph`.
        ///
        /// @details
        /// Cleared at `generateRawProofGraph` entry, then seeded with fixed
        /// synthetic-chapter citations. Chosen proof edges commit per written
        /// chapter; `buildStack` probes both sets in its admissibility check.
        /// Export-loop state like `cachedProofStacks` — single-threaded.
        TheoremUsageGraph exportTheoremUsage;

        /// @brief Theorem whose chapter the export loop is currently
        ///        building (global-theorem-list string form).
        ///
        /// @details
        /// Set at the top of each `generateRawProofGraph` loop iteration and
        /// cleared after the iteration's chapter writes. Empty string means
        /// the cross-chapter admissibility check is DISABLED — the defined
        /// state for `findEnds` (debug chapters) and direct `buildStack`
        /// unit-test callers, which keep the legacy chapter-local behavior.
        std::string exportCurrentChapterTheorem;

        /// The load-time compiled forms of the batch's external theorems
        /// (the exact strings `prove2` broadcast after
        /// `precompileStructuralOperators`). Exported post-run to
        /// `precompiled_external_theorems.txt` so the processed graph's
        /// external registry carries the compiled twins of base-form
        /// externals — chapter citations use the run's compiled names,
        /// which the verifier's origin check matches only textually.
        std::vector<std::string> precompiledExternalsExport;


        std::vector<OrCandidate> orCandidates;
        mutable std::mutex orCandidatesMutex;

        // OR pairs from head switch: (existence compiled, companion compiled)
        std::vector<std::pair<std::string, std::string>> orPairsFromHeadSwitch;



        /// @brief Theorem rows already scanned by the in-run OR seam, keyed
        /// by full theorem string.
        ///
        /// @details
        /// `constructOrTheoremsInRun` walks `globalTheoremList` once per
        /// iteration and must not rescan rows: membership here marks a row
        /// as consumed by the scan regardless of what the scan did with it.
        /// Keyed by string, not by list index — `revertVacuousGlobalTheorems`
        /// erases rows in place, and an index cursor would silently skip the
        /// rows that slid into the erased positions. Never cleared during a
        /// batch; touched only on the single-threaded phase-4 barrier seam.
        ///
        /// @see `constructOrTheoremsInRun` — sole reader/writer.
        std::unordered_set<std::string> orInRunScannedRows;

        /// @brief Canonical or-pair ledger shared by BOTH construction
        /// seams: sorted `(min, max)` implication pair → the constructed
        /// or-theorem string.
        ///
        /// @details
        /// Mirror pairs `(x, y)` and `(y, x)` describe one disjunction, but
        /// their element orders differ, so the or-operator registry treats
        /// them as distinct entries — constructing both double-mints
        /// `or<N>` (the mandatory-canonical-dedup constraint from the A16
        /// Phase 2 plan). Membership here is the single cross-seam gate:
        /// the in-run seam skips a pair whose mirror was already built, and
        /// the export seam recovers the stored string for its return
        /// vector + subsumption bookkeeping without reconstructing (a
        /// reconstruction from the OTHER side would mint the mirrored
        /// operator). Single-threaded access on the phase-4 barrier and
        /// post-run seams only.
        ///
        /// @see `constructOrTheoremsInRun` / `constructOrTheoremsFromPairs`.
        std::map<std::pair<std::string, std::string>, std::string> orBuiltByPair;

        /// @brief Head-switched companions of the or theorems registered by
        /// the in-run seam.
        ///
        /// @details
        /// An in-run or theorem rests on its source theorem alone; its row
        /// names the head-switched companion only as the second parent
        /// reference. A later proof of that companion is subsumed by the or
        /// theorem and may rest on it, so registering it would make the two
        /// chapters cite each other. `updateGlobalSingleChapter` and
        /// `updateGlobal` therefore neither register nor broadcast a theorem
        /// listed here, and the companion's own logic blocks are retired when
        /// it is listed (`retireSubsumedCompanion`). Written by
        /// `constructOrTheoremsInRun` on the
        /// single-threaded phase-4 barrier seam, read by the single-threaded
        /// registration drains.
        ///
        /// @see `constructOrTheoremsInRun`, `updateGlobalSingleChapter`,
        ///      `updateGlobal`.
        std::unordered_set<std::string> orInRunCompanions;





        /// @brief Theorem rows already scanned by the in-run pre-split
        /// merge seam, keyed by full theorem string.
        ///
        /// @details
        /// `constructOrEliminationInRun` walks `globalTheoremList` once per
        /// iteration; membership here marks a row as consumed by the pair
        /// scan regardless of the scan's verdict. String-keyed for the same
        /// reason as `orInRunScannedRows`: `revertVacuousGlobalTheorems`
        /// erases rows in place, so an index cursor would silently skip the
        /// rows that slid into the erased positions. Never cleared during a
        /// batch; touched only on the single-threaded phase-4 barrier seam.
        ///
        /// @see `constructOrEliminationInRun` — sole reader/writer.
        std::unordered_set<std::string> orElimInRunScannedRows;

        /// @brief Guard-variant candidate index: the common part of a
        /// variant row → every (row, guard premise) filed under it.
        ///
        /// @details
        /// The common part is the premise chain minus its innermost link,
        /// '\x01'-joined, then '\x02', then the head — a byte-level key,
        /// sound under the pre-split authoring convention (the guard
        /// premise is the INNERMOST premise and binder-free, so the two
        /// variants of one pre-split share everything else byte-for-byte).
        /// Two entries under one key with different guards form a merge
        /// candidate pair. Grows monotonically; entries are never removed
        /// (the pair ledger and the append door gate re-attempts).
        /// Single-threaded phase-4 seam access only.
        ///
        /// @see `constructOrEliminationInRun` — sole reader/writer.
        std::map<std::string, std::vector<std::pair<std::string, std::string>>> orElimGuardIndex;

        /// @brief Candidate pairs awaiting their or-theorem license:
        /// (variantA, guardA, variantB, guardB), insertion-ordered.
        ///
        /// @details
        /// A pair forms exactly once — when the second variant is scanned —
        /// but its licensing or theorem may mint on a LATER iteration's
        /// drain (in-run or construction runs in the same window, earlier).
        /// Unresolved pairs therefore park here and are re-probed at every
        /// seam invocation, in insertion order (deterministic). A pair
        /// leaves the vector only by merging or by the append door's
        /// string dedup; a licenseless pair parks for the whole batch
        /// (defined non-action, the approved silent-refusal contract).
        /// Single-threaded phase-4 seam access only.
        ///
        /// @see `constructOrEliminationInRun` — sole reader/writer.
        std::vector<std::array<std::string, 4>> orElimPendingPairs;

        /// @brief Merge-pair ledger: sorted (variant, variant) pair → the
        /// merged theorem string.
        ///
        /// @details
        /// Records every pair the seam has merged — or refused through the
        /// append door's string dedup — so a pair is attempted at most once
        /// per batch. Mirrors `orBuiltByPair`'s discipline. Single-threaded
        /// phase-4 seam access only.
        ///
        /// @see `constructOrEliminationInRun`.
        std::map<std::pair<std::string, std::string>, std::string> orElimBuiltByPair;

        /// @brief Or-theorem citation ledger: merged theorem → the or
        /// theorem that licensed its merge.
        ///
        /// @details
        /// `globalTheoremList` rows carry only two auxiliary slots — the
        /// merge row's aux2/aux3 cite the two variants — so the third
        /// citation an `or elimination` chapter row must emit lives here.
        /// Written by the seam when a merge lands; read by
        /// `generateRawProofGraph`'s or-elimination branch, which asserts
        /// on a miss (a registered merge without its recorded license is a
        /// seam bug, never a skippable state). Single-threaded access
        /// (phase-4 seam + the export seam).
        ///
        /// @see `constructOrEliminationInRun` / `generateRawProofGraph`.
        std::map<std::string, std::string> orElimCitedOrByMerged;



















        mutable std::mutex dependenciesMutex;  // protects globalDependencies
        mutable std::mutex theoremListMutex;   // protects globalTheoremList + globalTheoremStrings
        mutable std::mutex inductionMemoryBlocksMutex;   // protects inductionMemoryBlocks
        mutable std::mutex pendingTwinDeactivationsMutex;   // protects pendingTwinDeactivations
        mutable std::mutex pendingSubtreeDeactivationsMutex; // protects pendingSubtreeDeactivations
        mutable std::mutex updateGlobalMutex;
        mutable std::mutex updateGlobalSingleChapterMutex;
        mutable std::mutex deferredAncestorAdmissionsMutex;  // protects deferredAncestorAdmissions

public:


        /// @brief Construct the prover for a named anchor batch.
        /// @details Loads `Config<anchorID>.json` via
        /// `ce::modifyCoreExpressionMap`, builds `anchorInfo` via
        /// `ce::initAnchor`, seeds the operator/variable counters from
        /// `ProverParameters`, and pre-populates `coreExpressionMap`.
        /// Body in `prover.cpp`. Construction does not run the pipeline.
        /// @param anchorID Short anchor name (e.g. `"Peano"`, `"Gauss"`,
        ///                 `"IncubatorPeano"`).
        /// @param selectedPhase2Backend Optional command-line test override.
        ///        When absent, `ProverParameters::use_gpu` selects CUDA or
        ///        processor execution from the batch config.
        /// @param glBinaryFile Optional per-batch `GL_binary_<Tag>.json` whose
        ///        entries and counters seed the compiled registry before the
        ///        core map is compiled (cross-batch persistent naming; the
        ///        batch driver passes the file Python seeded from
        ///        `GL_binary_shared.json`). Absent — every unit test — the
        ///        constructor reads no file and compiles from the config
        ///        alone; a test that needs a binary writes its own under
        ///        `std::filesystem::temp_directory_path()` and passes it.
        /// @param precompiledDefinitions Optional recorded compile of the same
        ///        configuration (`PrecompiledDefinitions`), installed instead
        ///        of compiling the configured definitions again; exclusive
        ///        with @p glBinaryFile. The batch driver never passes one; the
        ///        unit-test fixture records the first compile of each
        ///        configuration and passes it to later fixtures.
        Prover(
            std::string anchorID,
            std::optional<Phase2Backend> selectedPhase2Backend = std::nullopt,
            std::optional<std::filesystem::path> glBinaryFile = std::nullopt,
            const PrecompiledDefinitions* precompiledDefinitions = nullptr);
private:



















        // Key: (elements vector / splitNK, category). The category is part of
        // the key so two structurally identical bodies in different categories
        // (e.g. an implication with elements [E1,E2] and an existence with the
        // same elements) allocate distinct names. Pre–approach-A (D-54) only
        // main batches contributed to GL_binary_shared.json, so a cross-category
        // elements collision between incubator and main could not enter the
        // shared registry. Once every batch (including incubator) merges into
        // shared, the category had to become part of this key — otherwise the
        // next batch's loadGlBinary would register an incubator-allocated
        // implication under elements E, and the next batch's compiler would
        // reuse that name for its own existence-shape expression with the same
        // elements, attaching an implication category to a body that should be
        // existential. See D-60 and
        // G-42.
        DefinitionExpansion definitionExpansion{this->definitions.configurations(), this->definitions.entries()};
        CompilationReuse compilationReuse{definitions, allPermutationsAna};
        StructuralCompiler structuralCompiler{definitions, compilationReuse, definitionExpansion,
            orRuleCompiler, existenceRuleCompiler, supportingRuleCompiler, integrationRuleCompiler};
        ScopeTypeResolver scopeTypes{definitions, structuralCompiler, canonicalizer, anchorInfo};
        OrRuleCompiler orRuleCompiler{definitions, compilationReuse, structuralCompiler,
            supportingRuleCompiler};
        ExistenceRuleCompiler existenceRuleCompiler{definitions, structuralCompiler};
        SupportingRuleCompiler supportingRuleCompiler{definitions, structuralCompiler};
        IntegrationRuleCompiler integrationRuleCompiler{definitions, structuralCompiler, integrationBuilder};
        CompilationPersistence compilationPersistence{definitions, compilationReuse, anchorID_};







        // New kernels — the three-phase elementary step driven directly by
        // proveKernel's barriered sweeps (phase 1 for all LBs -> phase 2 for all
        // -> phase 3 for all), plus the per-executor hashburst unit performElem2.

        /// @brief Phase 1 of the elementary step — pre-hashburst mail absorb.
        ///
        /// @details
        /// The first of the three phases the elementary step is split into for
        /// the LB split. Runs the ENTRY hashburst-dump trap
        /// (Rule 14 — relocated here unchanged, follows its bracketed code)
        /// and runs the pre-burst
        /// `standardProcessing` (`PRE_FIXPOINT_MAIL_ABSORB`): empties the
        /// per-burst `workingMemory` / `intExternalStatements` staging and
        /// refills them from this cycle's
        /// `mailIn` + `nextIterationInternalMail`. After this phase the LB's
        /// request-generation inputs are current; the phase may discharge the
        /// LB (flip `isActive`) via the absorb's `dischargeToBeProved`.
        ///
        /// `PHASE13_DEEP_TIMING` attributes claim/load, routing pull, external
        /// and internal origin/statement absorption, equivalence processing,
        /// discharge, and cleanup into per-worker rows. These are summed worker
        /// times and remain explicitly separate from the complete barrier wall.
        ///
        /// @param body   The LB being stepped.
        /// @param coreId The worker core id (routes `standardProcessing`).
        /// @return (void)
        /// @see `performElemPhase2`, `performElemPhase3` — the other phases.
        /// @see `standardProcessing` — the absorb pipeline this phase invokes.
        void performElemPhase1(Memory& body, unsigned coreId);





        /// @brief Hard ceiling on the number of phase-2 parts one LB may carry.
        ///
        /// @details
        /// The one split dimension is the expression/bucket split: a straggler
        /// runs at `logicalCores` buckets (the machine core count). This ceiling
        /// bounds that fan-out, and it is large enough that hitting it means a
        /// machine with more cores than the constant — deliberately raise it then,
        /// never cap the split silently.
        ///
        /// It bounds two things, both loudly (Rule 19): the entry check that
        /// `logicalCores` fits, and the per-LB check in the finalize that the
        /// pass's part count fits. The finalize's parts array itself is a reused
        /// per-worker heap buffer sized to the LB's actual part count, not to this
        /// constant — a stack array at this size would overflow the thread stack.
        ///
        /// @invariant `logicalCores <= kMaxSplitParts`, asserted at `proveKernel`
        ///            entry before any phase-2 pass.
        /// @invariant An LB's total part count in one pass never exceeds this,
        ///            asserted per LB in the finalize.
        /// @see `performElemPhase2` — the parts-array consumer.



        /// @brief Phase 3 of the elementary step — post-hashburst absorb
        ///        and shared scope cleanup.
        ///
        /// @details
        /// Runs the post-burst `standardProcessing` (`POST_FIXPOINT_MAIL_FLUSH`)
        /// absorbing the hashburst's `sameIterationInternalMail` deposits — this
        /// is where the authoritative deactivation / discharge fires (I-66) when
        /// a closing head was deposited. The LB's `mailOut` is left populated for
        /// the post-join `MailLog` commit barrier in `proveKernel` to drain.
        /// Standard processing owns hypothesis reaction and ordered scope
        /// cleanup. This wrapper owns the temporary exclusion, its completion
        /// assertion, quiescence bookkeeping and EXIT hashburst dump (Rule 14).
        /// `PHASE13_DEEP_TIMING` attributes the standard-processing substeps,
        /// reaction, each sanitize/drain seam, quiescence bookkeeping, and the
        /// working-set claim/release into contention-free per-worker rows.
        ///
        /// @param body   The LB being stepped.
        /// @param coreId Worker core id (routes `standardProcessing`).
        /// @return (void)
        /// @see `performElemPhase1`, `performElemPhase2` — the other phases.
        void performElemPhase3(Memory& body, unsigned coreId);



        // Parallel kernel
        void proveKernel(const std::vector<Memory*>& bodies);

public:

        // Keep ONLY this prove()
        void prove(int numberIterations,
            const std::vector<Memory*>& bodies);
private:




















        // Set by every deactivateUnnecessary walk that reaches its end;
        // settled by ONE deactivateRecursively sweep right after the
        // iteration's theorem drains in proveKernel (D-451). The in-run or
        // seam runs after that sweep and settles its own companion walks.
        bool deactivationSweepOwed = false;

public:


















        void analyzeExpressions(const std::vector<std::string>& theorems,
                                const std::vector<std::string>& provedTheorems,
                                const std::vector<std::string>& externalTheorems = {});

        // Wall seconds of the start-up stages of the last `analyzeExpressions`
        // call before the prover run: the deload purge and extent reopen, the
        // counterexample filter, and the batch preparation (`prepareBatch`).
        // `run_modes.cpp · fullRun` records them as frame stages beside
        // `native.prover` (D-533).
        struct StartupTimings {
            double extentPurgeSeconds = 0.0;
            double ceFilterSeconds = 0.0;
            double batchPreparationSeconds = 0.0;
        } startupTimings;
private:


public:

        // CE-filter-only entry point — runs the counterexample filter on the
        // supplied theorem list and writes `filtered_conjectures.txt`. Used by
        // `run_modes::ceOnlyRun` (`--ce-only <Tag>`) for the fast iteration
        // loop: conjecturer + CE filter, skipping the main prover.
        // Honours `parameters.skip_ce_filter` exactly as `analyzeExpressions`
        // does — a `skip_ce_filter=true` config produces no output and prints
        // a warning. When `parameters.mirror_refutation` is on, loads
        // `mirror_pairs.txt` into `mirrorPartnerMap` before the per-fact-file
        // loop so every pass can flip the mirrors of CE-refuted conjectures.
        // Idempotent; safe to invoke multiple times.
        std::vector<std::string> runCeFilterOnly(const std::vector<std::string>& theorems);
private:



















        // Like analyzeExpressions, but for CE filtering


        std::vector<std::string> filterConjecturesWithCE(
            const std::vector<std::string>& conjectures,
            const std::vector<std::string>& simpleFacts);













        inline bool startsWithStr(const std::string& s, const std::string& pre) {
            if (s.size() < pre.size()) return false;
            for (std::size_t i = 0; i < pre.size(); ++i) if (s[i] != pre[i]) return false;
            return true;
        }









































        // Upper bound on the raw (pre-dedup) extra-level count a single
        // enumerateEqClassRewrites mapping can accumulate — the sum over the
        // <= MAX_ARITY substituted positions of each pair's admission-level set
        // size. Admission-level sets are per-iteration and tiny, so this is a
        // generous ceiling; a firing assert flags a previously-unknown state
        // (Rule 19), not a silent cap.


































































        AdmissionMemory admissionMemory{admissionPolicy, anchorInfo, canonicalTemplates, deferredAncestorAdmissionsMutex, deferredAncestorPages, definitions, parameters, scopeTypes, steward, integrationProcessor, rejectedSourceReplay, equivalenceQueries};
        IntegrationProcessor integrationProcessor{admissionPolicy, canonicalTemplates, canonicalizer, definitions, integrationBuilder, parameters, scopeTypes, admissionMemory, disintegrationProcessor, hashRuleInstaller, implicationSubproofs, rejectedSourceReplay, statementMemory, unaryIntegrationProcessor, compactRuleProcessor};
        ImplicationSubproofs implicationSubproofs{canonicalTemplates, canonicalizer, definitions, integrationBuilder, parameters, equivalenceQueries, statementMemory, rejectedSourceReplay};
        ContradictionDemand contradictionDemand{definitions, parameters, integrationProcessor, implicationSubproofs};
        RejectedSourceReplay rejectedSourceReplay{admissionPolicy, anchorInfo, canonicalTemplates, canonicalizer, definitions, parameters, rejectedSources, admissionMemory, caseLedger, proofHistory, watchProcessor};
        HashRuleInstaller hashRuleInstaller{admissionMarkers, allPermutationsAna, hashRuleClassifier, implicationMultiplier, parameters, unaryMemory, admissionMemory};
        CompactRuleProcessor compactRuleProcessor{canonicalizer, parameters, disintegrationProcessor, hashRuleInstaller, statementMemory, integrationBuilder, rejectedSourceReplay};
        StatementMemory statementMemory{admissionPolicy, canonicalizer, definitions, parameters, admissionMemory, caseLedger, caseProcessor, compactRuleProcessor, disintegrationProcessor, equalityPropagation, equivalenceProcessor, equivalenceQueries, integrationProcessor, orImplicationProcessor, proofHistory, unaryIntegrationProcessor, watchProcessor};
        EquivalenceProcessor equivalenceProcessor{canonicalTemplates, canonicalizer, parameters, rejectedSources, caseLedger, caseProcessor, compactRuleProcessor, equalityPropagation, equivalenceAdmissionRewriter, rejectedSourceReplay, watchProcessor};
        EqualityPropagation equalityPropagation{admissionPolicy, allMappingsAna, canonicalizer, ceFilteringActive, parameters, caseProcessor, proofHistory, statementMemory, watchProcessor};
        EquivalenceAdmissionRewriter equivalenceAdmissionRewriter{canonicalTemplates, canonicalizer, parameters, rejectedSources, caseLedger, proofHistory, rejectedSourceReplay, watchProcessor};
        WatchProcessor watchProcessor{canonicalTemplates, canonicalizer, ceFilteringActive, parameters, caseLedger, caseProcessor, proofHistory};
        CaseLedger caseLedger{canonicalTemplates, canonicalizer, definitions};
        CaseProcessor caseProcessor{admissionPolicy, anchorInfo, canonicalTemplates, canonicalizer, definitions, parameters, caseLedger, equivalenceQueries, proofHistory};
        DisintegrationProcessor disintegrationProcessor{admissionPolicy, canonicalTemplates, canonicalizer, ceFilteringActive, definitions, existenceRuleCompiler, hashRuleClassifier, integrationBuilder, parameters, rejectedSources, admissionMemory, caseLedger, caseProcessor, rejectedSourceReplay, statementMemory};
        OrImplicationProcessor orImplicationProcessor{blockIdentity, canonicalizer, ceFilteringActive, definitions, hashRuleClassifier, parameters, caseLedger, caseProcessor, disintegrationProcessor, proofHistory, watchProcessor};
        ProofHistory proofHistory{parameters};
        UnaryIntegrationProcessor unaryIntegrationProcessor{canonicalTemplates, canonicalizer, parameters, unaryMemory, hashRuleInstaller, rejectedSourceReplay, integrationProcessor};
        GoalProcessor goalProcessor{canonicalizer, contradictionTable, definitions, inductionMemoryBlocks, inductionMemoryBlocksMutex, parameters, updateGlobalMutex, updateGlobalSingleChapterMutex, updateGlobalSingleChapterPages, updateGlobalTuples, blockKeyReader, caseProcessor, implicationSubproofs, proofHistory, scopeLifecycle, statementMemory};
        ScopeLifecycle scopeLifecycle{body, deactivationSweepOwed, definitions, globalDependencies, pendingSubtreeDeactivations, pendingSubtreeDeactivationsMutex, pendingTwinDeactivations, pendingTwinDeactivationsMutex, simpleMapStore, anchorScopeQueries};
        InductionUsefulness inductionUsefulness{canonicalizer, definitions, proofBurstNumber};
        LogicBlockOperations logicBlockOperations{blockIdentity, ceFilteringActive, contradictionTable, parameters, rejectedSources, blockMailProcessor, caseProcessor, equalityPropagation, equivalenceProcessor, goalProcessor, implicationSubproofs, inductionUsefulness, orImplicationProcessor, statementMemory, unaryIntegrationProcessor, watchProcessor, contradictionDemand, rejectedSourceReplay
#if PHASE13_DEEP_TIMING
, phase13TimingRows
#endif

#if PHASE13_DEEP_TIMING
, phase13TimingWorkers
#endif
};
        BlockMailProcessor blockMailProcessor{blockIdentity, body, definitions, mailLog, parameters, permanentBodies, simpleMapStore, structuralCompiler, proofHistory};
        AnchorScopeQueries anchorScopeQueries{definitionExpansion};
        EquivalenceQueries equivalenceQueries{canonicalTemplates, canonicalizer};
        BlockKeyReader blockKeyReader{};

        OrCandidateProgress orCandidateProgress{orCandidates, orCandidatesMutex};
        TheoremPublisher theoremPublisher{anchorScopeQueries, body, definitionExpansion, definitions, dependenciesMutex, fullTheoremList, globalDependencies, globalTheoremList, globalTheoremProducers, globalTheoremStrings, orInRunScannedRows, parameters, pendingCompactionMutex, pendingCompactionQueue, proofHistory, retractedVacuousTheorems, scopeLifecycle, simpleMapStore, steward, structuralCompiler, theoremListMutex, updateGlobalSingleChapterPages, orCandidateProgress, orInRunCompanions};
        ProofGridBuilder proofGridBuilder{admissionMemory, anchorInfo, anchorScopeQueries, blockIdentity, body, definitions, hashRuleInstaller, inductionUsefulness, integrationProcessor, lbStore, mailLog, parameters, permanentBodies, scopeTypes, simpleMapStore, statementMemory, theoremPublisher};
        OrTheoremBuilder orTheoremBuilder{definitions, fullTheoremList, globalTheoremList, orBuiltByPair, orCandidates, orCandidatesMutex, orInRunScannedRows, orPairsFromHeadSwitch, orRuleCompiler, parameters, proofHistory, theoremListMutex, theoremPublisher, orInRunCompanions, scopeLifecycle, deactivationSweepOwed};
        OrEliminationBuilder orEliminationBuilder{definitions, fullTheoremList, globalTheoremList, orElimBuiltByPair, orElimCitedOrByMerged, orElimGuardIndex, orElimInRunScannedRows, orElimPendingPairs, orRuleCompiler, parameters, proofHistory, theoremListMutex, theoremPublisher};
        CounterexampleFilter counterexampleFilter{anchorInfo, ceBody, ceSimpleMapStore, definitions, filteredConjecturesDirectory, hashRuleInstaller, inductionMemoryBlocks, lbStore, parameters, permanentBodiesCE, statementMemory, updateGlobalTuples};
        EncodedExpressionNormalizer encodedExpressionNormalizer{};

        ProofReconstructor proofReconstructor{simpleMapStore, exportTheoremUsage, exportCurrentChapterTheorem};
        ProofGraphWriter proofGraphWriter{body, globalTheoremList, globalTheoremStrings, globalTheoremProducers, definitions, simpleMapStore, cachedProofStacks, exportTheoremUsage, exportCurrentChapterTheorem, orElimCitedOrByMerged, compilationPersistence, theoremPublisher, proofReconstructor};

        Phase2Executor phase2Executor{parameters, phase2Backend, phase2CumulativeSeconds, phase2MeasuredIterations, phase2MeasurementEpoch, ceFilteringActive, requestGenerator, ruleEvaluator, anchorID_, logicalCores, steward, admissionMemory, anchorInfo, definitions};
        ProofBarrier proofBarrier{parameters, body, updateGlobalTuples, inductionMemoryBlocks, ceFilteringActive, definitions, mailLog, rollingMailHistoryEnabled, peakDeloadableMailOutBytes, simpleMapStore, dischargeScratch, pendingCompactionQueue, pendingDischarge, steward, structuralCompiler, orRuleCompiler, deactivationSweepOwed, admissionMemory, proofHistory, scopeLifecycle, blockMailProcessor, anchorScopeQueries, theoremPublisher, orTheoremBuilder, orEliminationBuilder};

        ProverInitialization proverInitialization{parameters, phase2Backend, body, allMappingsAna, allPermutationsAna, globalDependencies, definitions, anchorInfo, anchorID_, permanentBodies, logicalCores, structuralCompiler, supportingRuleCompiler, compilationPersistence, compilationReuse};

        ProofRunReporter proofRunReporter{permanentBodies, globalTheoremList};

public:
/// @brief Access the compilation persistence operation interface.
/// @details The returned class retains private state and exposes only its own operations.
/// @return Borrowed operation interface for this prover lifetime.
CompilationPersistence& compilationFiles() { return compilationPersistence; }
/// @brief Access the compilation reuse operation interface.
/// @details The returned class retains private state and exposes only its own operations.
/// @return Borrowed operation interface for this prover lifetime.
const CompilationReuse& compiledNames() const { return compilationReuse; }
/// @brief Access the definition expansion operation interface.
/// @details The returned class retains private state and exposes only its own operations.
/// @return Borrowed operation interface for this prover lifetime.
DefinitionExpansion& expressionExpansion() { return definitionExpansion; }
/// @brief Access the structural compiler operation interface.
/// @details The returned class retains private state and exposes only its own operations.
/// @return Borrowed operation interface for this prover lifetime.
StructuralCompiler& compiler() { return structuralCompiler; }
/// @brief Access the theorem publisher operation interface.
/// @details The returned class retains private state and exposes only its own operations.
/// @return Borrowed operation interface for this prover lifetime.
TheoremPublisher& results() { return theoremPublisher; }
/// @brief Access the or theorem builder operation interface.
/// @details The returned class retains private state and exposes only its own operations.
/// @return Borrowed operation interface for this prover lifetime.
OrTheoremBuilder& orConstruction() { return orTheoremBuilder; }
/// @brief Access the proof grid builder operation interface.
/// @details The returned class retains private state and exposes only its own operations.
/// @return Borrowed operation interface for this prover lifetime.
ProofGridBuilder& gridOperations() { return proofGridBuilder; }
/// @brief Access the proof graph writer operation interface.
/// @details The returned class retains private state and exposes only its own operations.
/// @return Borrowed operation interface for this prover lifetime.
ProofGraphWriter& proofWriter() { return proofGraphWriter; }
/// @brief Access the compiled definition registry operation interface.
/// @details The returned class retains private state and exposes only its own operations.
/// @return Borrowed operation interface for this prover lifetime.
const CompiledDefinitionRegistry& definitionRegistry() const { return definitions; }
/// @brief Access the scope type resolver operation interface.
/// @details The returned class retains private state and exposes only its own operations.
/// @return Borrowed operation interface for this prover lifetime.
ScopeTypeResolver& types() { return scopeTypes; }
/// @brief Access the hash rule installer operation interface.
/// @details The returned class retains private state and exposes only its own operations.
/// @return Borrowed operation interface for this prover lifetime.
HashRuleInstaller& hashRules() { return hashRuleInstaller; }
/// @brief Access the statement memory operation interface.
/// @details The returned class retains private state and exposes only its own operations.
/// @return Borrowed operation interface for this prover lifetime.
StatementMemory& statements() { return statementMemory; }
/// @brief Read the active proof configuration.
/// @details Exposes the existing parameters without permitting arbitrary external mutation.
/// @return Read-only configuration for this prover lifetime.
const ProverParameters& configuration() const { return parameters; }
/// @brief Set the compressor execution flags together.
/// @details Performs the original two assignments in order at the compressor entry or exit boundary.
/// @param enabled Whether compression is entering its hash-burst phase.
/// @return No value.
void setCompressionMode(bool enabled) { parameters.compressor_mode = enabled; parameters.ban_disintegration = enabled; }
/// @brief Select the existing phase-two shortcut reporting mode.
/// @details Applies the caller selection before proof execution.
/// @param enabled Requested reporting mode.
/// @return No value.
void setPhase2ShortcutProfile(bool enabled) { phase2ShortcutProfile = enabled; }
/// @brief Set the batch destination for filtered conjectures.
/// @details Uses path assignment before filtering starts.
/// @param directory Requested output directory.
/// @return No value.
void setFilteredConjecturesDirectory(const std::filesystem::path& directory) { filteredConjecturesDirectory = directory; }
/// @brief Record the last compression result.
/// @details Copies the existing survivor list at its original postprocessing boundary.
/// @param survivors Theorem strings retained by compression.
/// @return No value.
void setCompressionSurvivors(const std::vector<std::string>& survivors) { lastCompressionSurvivors = survivors; }
/// @brief Read the last compression result.
/// @details Returns the current batch result without exposing mutation.
/// @return Read-only surviving theorem strings.
const std::vector<std::string>& compressionSurvivors() const { return lastCompressionSurvivors; }
/// @brief Read compiled twins of the external input theorems.
/// @details Exposes the existing export registry populated during batch execution.
/// @return Read-only compiled external theorem strings.
const std::vector<std::string>& precompiledExternals() const { return precompiledExternalsExport; }
/// @brief Allocate an independent compression logical block.
/// @details Uses the existing address-stable logical-block store and its construction operation.
/// @return New block whose lifetime is released by destroyCompressionBlock.
Memory* createCompressionBlock() { return lbStore.create<Memory>(); }
/// @brief Release an independent compression logical block.
/// @details Uses the same logical-block store destruction operation after graph extraction.
/// @param block Live compression block allocated by this prover.
/// @return No value.
void destroyCompressionBlock(Memory* block) { lbStore.destroy(block); }
/// @brief Read mail-memory observations after proof execution.
/// @details Samples the existing counters and collections at the quiescent reporting boundary without exposing mutable stores.
/// @return Passive mail-usage values for post-run reporting.
ProverMailUsage mailUsage() const {
    return {mailArena.blocksHeld(),
        peakRoutingMailInBlocks.load(std::memory_order_relaxed),
        peakDeloadableMailOutBytes,
        routingMailInBlocksInFlight.load(std::memory_order_relaxed),
        dormantLogicBlocksAtGridBuild,
        rollingMailHistoryEnabled,
        mailLog.totalCommittedHistoryBytes,
        mailLog.peakRetainedHistoryBytes};
}
};








// Surface API mirroring Python (camelCase):


} // namespace gl
