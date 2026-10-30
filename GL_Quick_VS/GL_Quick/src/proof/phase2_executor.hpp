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
#include "phase2_backend.hpp"
#include "../infra/phase2_trace.hpp"
#include "../proof/cached_proof_stack.hpp"
#include "../proof/theorem_usage_graph.hpp"
#include "../proof/proof_reconstructor.hpp"
#include "../proof/proof_graph_writer.hpp"
#include "../proof/theorem_publisher.hpp"
#include "../proof/proof_grid_builder.hpp"
#include "../proof/or_theorem_builder.hpp"
#include "../proof/or_elimination_builder.hpp"
#include "../proof/counterexample_filter.hpp"
#include "../inference/encoded_expression_normalizer.hpp"
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
#include "../proof/variable_copy_detector.hpp"
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
namespace tests { struct Phase2ExecutorFixture; }
/// @brief Phase2 Executor operations for the proof algorithm.
/// @details Uses explicit peer interfaces while the supplied logical blocks retain storage and proof state.
class Phase2Executor {
public:
        /// @brief How many expression stumps a straggler's producer grows PER
        ///        bucket, so the round-robin deal balances the buckets.
        ///
        /// @details A straggler splits into `logicalCores` expression buckets; the
        /// round-1 producer (`produceExpressionStumps`) grows this many times
        /// `logicalCores` base-candidate stumps, then `proveKernel` deals them
        /// round-robin so each bucket holds ~this many stumps of varied grow-tree
        /// size. A larger factor balances the buckets better at the cost of a longer
        /// enumeration; `1` would give one stump per bucket (worst balance). Not a
        /// correctness knob — the union of buckets is the whole search regardless.




static constexpr int kStumpsPerBucketTarget = 4;
friend struct tests::Phase2ExecutorFixture;
public:
/// @brief Bind the phase2 executor operation inputs.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param phase2Backend Borrowed Phase2Backend interface or state; must outlive this component.
/// @param phase2CumulativeSeconds Borrowed double interface or state; must outlive this component.
/// @param phase2MeasuredIterations Borrowed uint64_t interface or state; must outlive this component.
/// @param phase2MeasurementEpoch Borrowed uint64_t interface or state; must outlive this component.
/// @param ceFilteringActive Borrowed bool interface or state; must outlive this component.
/// @param requestGenerator Borrowed HashRequestGenerator interface or state; must outlive this component.
/// @param ruleEvaluator Borrowed HashRuleEvaluator interface or state; must outlive this component.
/// @param anchorID_ Borrowed std::string interface or state; must outlive this component.
/// @param logicalCores Borrowed unsigned interface or state; must outlive this component.
/// @param steward Borrowed std::unique_ptr interface or state; must outlive this component.
/// @param admissionMemory Borrowed AdmissionMemory interface or state; must outlive this component.
/// @param anchorInfo Borrowed ce::AnchorInfo interface or state; must outlive this component.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
Phase2Executor(const ProverParameters& parameters, Phase2Backend& phase2Backend, double& phase2CumulativeSeconds, uint64_t& phase2MeasuredIterations, uint64_t& phase2MeasurementEpoch, bool& ceFilteringActive, HashRequestGenerator& requestGenerator, HashRuleEvaluator& ruleEvaluator, std::string& anchorID_, unsigned& logicalCores, std::unique_ptr<MemorySteward>& steward, AdmissionMemory& admissionMemory, const ce::AnchorInfo& anchorInfo, const CompiledDefinitionRegistry& definitions);

public:
/// @brief One burst part — request generation + the fixpoint
        ///        evaluation pass, writing deposits to a per-part buffer.
        ///
        /// @details
        /// The inner unit of phase 2 — one `(LB, part)` task in `proveKernel`'s
        /// flat executor pool. Generates the hash requests for its part (an
        /// unsplit LB, one expression bucket of a straggler, or the CE path),
        /// then runs the single-pass `FIXPOINT_LOOP`
        /// capturing each firing's deposit onto the task's sealed-page record
        /// chain (`sealedPages.appendRecord`; it does **not**
        /// apply them — the per-LB finalize `performElemPhase2` hands all of an
        /// LB's sealed part sets, in part order, to `applyFiringRecords` after
        /// the pool join). The
        /// per-step delta clears are likewise NOT done here — the finalize clears
        /// them once per LB before the merge (so every part still reads the deltas
        /// it generates from). Read-only on the LB ([D-116])
        /// so concurrent parts on one LB never race. The arena buffers
        /// are per-thread scratch (caller-owned, reset per task), and
        /// `sealedPages` — records and their strings alike — is the task's own
        /// output slot.
        ///
        /// @param body          The shared LB (read for request generation /
        ///                      firing; deposits go to the sealed-page record
        ///                      chain, not the LB).
        /// @param coreId        Worker core id (passed to
        ///                      `checkLocalEncodedMemoryStatic`).
        /// @param partCount     Total concurrent parts of this LB this burst
        ///                      (`>= 1`): a whole-LB expression split runs
        ///                      `partCount` bucket parts. The doom line's winner
        ///                      is one of these parts.
        /// @param splitStump    The LB split's expression dimension: this bucket
        ///                      part's stump, which every request it generates
        ///                      must contain, plus its place among its siblings
        ///                      (which the generator's seed phase reads).
        ///                      Default-constructed for an unsplit LB and for the
        ///                      counter-example filter. Handed to every
        ///                      request-generator call in the burst.
        /// @param sealedPages   This task's page set: record chain + the sealed
        ///                      strings/spans the records reference. Bound and
        ///                      `Filling` at entry; the caller seals it after
        ///                      this returns.
        /// @param doomLine      External per-LB packed (position, ordinal) doom
        ///                      line, shared by all the LB's parts (lives OUTSIDE
        ///                      the LB so the hashburst stays read-only on the
        ///                      LB). `kNoDoomLine` at burst start. Read by the
        ///                      streaming sink's `canAccept`; lowered by CAS-min
        ///                      when `burstDeactivates` fires.
        /// @return (void). The split-policy marker is this part's submatch count
        ///         (`gl::HashRequestGenerator::growthMatchCount()`, the subkey growth-probe tally),
        ///         read by the worker from the thread_local right after this call;
        ///         `proveKernel` SUMS the LB's parts into its total work, which the
        ///         end-of-iteration `isStraggler` pass consumes.
        /// @see `performElemPhase2` — the orchestrator that calls this and merges.
        /// @see `isStraggler` — consumes the LB's summed submatch work.
        /// @see `produceExpressionStumps` — where a stump comes from.
void performElem2(const Memory& body, unsigned coreId,
            int partCount,
            const SplitStumpRef& splitStump,
            SealedPageSet& sealedPages,
            std::atomic<int64_t>& doomLine);

public:
/// @brief Whether an LB is a straggler that should split into `cores`
        ///        expression buckets next iteration — the split TRIGGER.
        ///
        /// @details The idle-core fair-share test. `totalWork / cores` is the
        /// ideally-balanced per-core load for this iteration's active main-path LBs;
        /// an LB whose own work exceeds it is, alone, a makespan bottleneck that
        /// leaves cores idle, so parallelising it across `cores` is justified. The
        /// second clause is the setup break-even (`minSplitWork`): below it a
        /// bucket's share of the work is smaller than the fixed per-bucket setup, so
        /// splitting provably cannot help — this suppresses splitting when the whole
        /// iteration is trivially cheap. Fair-share is masking-resistant (one
        /// straggler lifts the bar by only `work / cores`) and self-limiting (at
        /// most `cores - 1` LBs can exceed it -> "a few"). Integer arithmetic only,
        /// so the verdict is a deterministic function of the (deterministic) work
        /// totals — the property that keeps two runs byte-identical. Pure function of
        /// its arguments; `static` so it is unit-testable without an analyzer.
        ///
        /// @param work         This LB's total work this iteration = sum of its
        ///                     parts' submatch counts (split-invariant, D-117; `>= 0`).
        /// @param totalWork    Sum of `work` over all active main-path LBs (`>= 0`).
        /// @param cores        `logicalCores`, the split fan-out (`>= 1`).
        /// @param minSplitWork `parameters.min_split_work`, the setup break-even
        ///                     floor (`>= 0`).
        /// @return `true` iff this LB should run as `cores` expression buckets next
        ///         iteration; `false` to run it unsplit.
        /// @see `proveKernel` — the end-of-iteration stats pass that calls this.
        /// @see `Memory::numberOfParts` — the per-LB state this drives (1 or `cores`).
static bool isStraggler(int64_t work, int64_t totalWork,
            int cores, int64_t minSplitWork);

public:
/// @brief Phase 2 finalize — merge one LB's flat executor parts and drain
        ///        the staged buffers.
        ///
        /// @details
        /// The executors already ran in `proveKernel`'s **flat** (LB, part)
        /// work-stealing pool, each read-only on the LB
        /// ([D-116]) and appending its records to its own sealed page set;
        /// `proveKernel` hands this LB's sealed part sets in part order and
        /// calls this once per LB after the pool join. Reads `isActive` (the
        /// phase-1 absorb may have discharged the LB, in which case it generated
        /// no executor tasks and `partCount` is 0): if active, clears the
        /// per-step delta containers once and merges via the canonical-sorted
        /// `applyFiringRecords` (partition- and thread-order-independent,
        /// [D-117]). Always drains the staged
        /// algebra-admission and integration-prep buffers
        /// (`drainAdmissionKeysAlgebra` / `drainDeferredIntegrationPreps`) before
        /// phase 3's post-burst absorb. Single-LB — `proveKernel` runs it as an
        /// across-LB sweep (I-28-safe), so it may freely mutate `body`. The split
        /// decision (which LBs are stragglers next iteration) lives in
        /// `proveKernel`'s end-of-iteration stats pass via `isStraggler`, NOT here;
        /// every burst runs to completion, so `performElemPhase2` always applies a
        /// COMPLETE burst (D-201).
        ///
        /// @param body      The LB being finalized.
        /// @param parts     This LB's sealed part sets in part order; the
        ///                  records ride each set's chain (consumed / index-
        ///                  sorted by `applyFiringRecords`).
        /// @param partCount Number of part sets (`0` for a phase-1-discharged
        ///                  LB — nothing to merge, drains still run).
        /// @param doomLine  The LB's final packed doom line (`kNoDoomLine` when
        ///                  no doom trigger fired this burst). When set, the
        ///                  merge takes ONLY the winning part's chain — the part
        ///                  named by `doomLineOrdinal` — and the other parts'
        ///                  records are discarded unread (sound: the trigger
        ///                  guarantees phase 3 discharges the LB, so sibling
        ///                  records are wasted-once-doomed).
        /// @param firingRecordsCanonical True only for the CUDA adapter's one
        ///                  already-sorted, already-doom-compacted chain. In
        ///                  that route `partCount` is one and `doomLine` is
        ///                  `kNoDoomLine`; processor executor chains pass false.
        /// @return (void)
        /// @see `performElem2` — the per-part executor that ran in the flat pool.
        /// @see `applyFiringRecords` — the canonical-order deposit merge.
        /// @see `isStraggler` — the split trigger (applied in `proveKernel`).
void performElemPhase2(Memory& body,
            SealedPageSet* const* parts, int32_t partCount,
            int64_t doomLine,
            bool firingRecordsCanonical = false);

public:
/// @brief Record the CUDA Phase 2 read-image measurement of one claimed,
///        resident LB at the end of its phase 1.
///
/// @param body The active LB whose phase 1 just finished.
void recordPhase2Measurement(Memory& body) const;
public:
/// @brief Execute and finalize the current burst's phase-two work.
/// @details Runs the existing flat worker rounds, CUDA projection, sealed-record merge and split measurements. The caller retains the trace lifetime and joins phase one before this call.
/// @param active Logical blocks selected at the burst boundary.
/// @param workers Existing worker count.
/// @param phase2TraceCall Existing trace sequence identifier.
/// @param phase2Started Start time captured after phase one.
/// @param phase2Trace Caller-owned trace finished at the boundary.
/// @param trapPh2Ns Per-block phase-two timings when runtime measurement is enabled.
/// @param trapFinNs Per-block finalization timings when runtime measurement is enabled.
/// @param trapWork Per-block work totals when runtime measurement is enabled.
/// @return Existing main-path split eligibility sampled at the original phase-two point.
/// @invariant All executor and finalizer workers join before returning.
bool execute(const std::vector<Memory*>& active, unsigned workers, uint64_t phase2TraceCall, std::chrono::steady_clock::time_point phase2Started, Phase2TraceScope& phase2Trace
#if RT_MEASUREMENT
, std::vector<int64_t>& trapPh2Ns, std::vector<int64_t>& trapFinNs, std::vector<int64_t>& trapWork
#endif
);

public:
#if RT_MEASUREMENT
/// @brief Clear the target block's observation counters at burst entry.
/// @details Performs the four relaxed atomic stores in their order.
/// @return No value.
static void resetTargetCounters();
/// @brief Read the target block accepted observation counter.
/// @details Uses the relaxed atomic load at the diagnostic output site.
/// @return Current observation count.
static int64_t targetAccepted();
/// @brief Read the target block acceptedunlinked observation counter.
/// @details Uses the relaxed atomic load at the diagnostic output site.
/// @return Current observation count.
static int64_t targetAcceptedUnlinked();
/// @brief Read the target block attempts observation counter.
/// @details Uses the relaxed atomic load at the diagnostic output site.
/// @return Current observation count.
static int64_t targetAttempts();
/// @brief Read the target block attemptsunlinked observation counter.
/// @details Uses the relaxed atomic load at the diagnostic output site.
/// @return Current observation count.
static int64_t targetAttemptsUnlinked();
#endif

private:
#if RT_MEASUREMENT
        // [RT phase measurement] C15 submatch attribution in the
        // hashburst-dump target LB only. Armed per executor task; counters are
        // process-wide atomics reset/printed at the target LB's phase-1/phase-3.
        static thread_local bool g_crtArmed;
        static thread_local NameId g_crtPreorderId;
        static thread_local NameId g_crtFiveId;
        static std::atomic<int64_t> g_crtAttempts;
        static std::atomic<int64_t> g_crtAttemptsUnlinked;
        static std::atomic<int64_t> g_crtAccepted;
        static std::atomic<int64_t> g_crtAcceptedUnlinked;

        /// [RT phase measurement] True when the premise multiset holds
        /// two positive divisibility statements (preorder over the `*` slot)
        /// sharing no argument - the C15-exclusive interior shape in this pool.
        static bool crtHasUnlinkedDivPair(const IntEncodedExpr* const* exprs,
                                          NameId count) {
            NameId da[8][2];
            int nd = 0;
            for (NameId i = 0; i < count && nd < 8; ++i) {
                const IntEncodedExpr& e = *exprs[i];
                if (e.nameId == g_crtPreorderId && e.negation == 0
                    && e.arity == 4 && e.argId[1] == g_crtFiveId) {
                    da[nd][0] = e.argId[2];
                    da[nd][1] = e.argId[3];
                    ++nd;
                }
            }
            for (int a = 0; a < nd; ++a)
                for (int b = a + 1; b < nd; ++b)
                    if (da[a][0] != da[b][0] && da[a][0] != da[b][1]
                        && da[a][1] != da[b][0] && da[a][1] != da[b][1])
                        return true;
            return false;
        }
#endif  // RT_MEASUREMENT

private:

const ProverParameters& parameters;
Phase2Backend& phase2Backend;
double& phase2CumulativeSeconds;
uint64_t& phase2MeasuredIterations;
uint64_t& phase2MeasurementEpoch;
bool& ceFilteringActive;
HashRequestGenerator& requestGenerator;
HashRuleEvaluator& ruleEvaluator;
std::string& anchorID_;
unsigned& logicalCores;
std::unique_ptr<MemorySteward>& steward;
AdmissionMemory& admissionMemory;
const ce::AnchorInfo& anchorInfo;
const CompiledDefinitionRegistry& definitions;
};
}
namespace gl {
}
