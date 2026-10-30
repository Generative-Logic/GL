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
#include "../memory.hpp"
#include "gpu_evaluation_usage.hpp"
#include "hash_key_builder.hpp"
#include "../logical_block/admission_policy.hpp"
#include "../logical_block/logic_block_identity.hpp"
#include "../logical_block/statement_knowledge.hpp"
#include "../compiler/definition_expansion.hpp"
#include "../infra/rt_tracker.hpp"
namespace gl {
/// @brief Evaluate hash rules and apply their ordered firing records.
/// @details Owns this algorithm boundary; logical blocks retain their physical stores.
class HashRuleEvaluator {
public:
/// @brief Bind HashRuleEvaluator inputs.
/// @details Binds the existing inputs without performing proof work or moving storage.
/// @param parameters Existing parameters interface or settings.
/// @param definitions Existing definitions interface or settings.
/// @param admissionPolicy Existing admissionPolicy interface or settings.
/// @param blockIdentity Existing blockIdentity interface or settings.
/// @return None; construction binds the supplied references.
HashRuleEvaluator(const ProverParameters& parameters, const CompiledDefinitionRegistry& definitions, AdmissionPolicy& admissionPolicy, const LogicBlockIdentity& blockIdentity);
/// @brief Read the current worker's rule-evaluation census.
/// @details The reference remains worker-local and is observed after evaluation.
/// @return Read-only evaluation counters.
static const GpuEvaluationUsage& evaluationUsage() { return g_gpuEvaluationUsage; }
/// @brief Clear the worker evaluation census at task entry.
/// @details Retains the original per-task reset point and zero initialization.
/// @return Nothing.
static void resetEvaluationUsage() { g_gpuEvaluationUsage = GpuEvaluationUsage{}; }
/// @brief Count an emitted request before its dependency check.
/// @details Records observation only; no proof branch reads this count.
/// @return Nothing.
static void noteRequest() { ++g_gpuEvaluationUsage.requests; }
/// @brief Count a request whose dependencies are known.
/// @details Records observation only; the caller then evaluates the request.
/// @return Nothing.
static void noteDependencyPassRequest() { ++g_gpuEvaluationUsage.dependencyPassRequests; }

public:
/// @brief Phase-2 early-exit predicate: does this freshly-fired head
        ///        prove the LB is doomed, so its hashburst can stop now?
        ///
        /// @details
        /// The LB-split phase-2 hashburst (`performElem2`) generates and checks
        /// requests one at a time. Contradiction / induction-goal LBs become
        /// determinably doomed the moment a single head fires; continuing to
        /// generate the rest of their requests is the incubator's dominant
        /// wasted runtime. Run on each non-marker head `FiringRecord` the check
        /// just produced, this predicate reports whether the LB is now doomed so
        /// the executor can lower the LB's external doom line and the burst
        /// stops.
        ///
        /// It is a pure detection mirror of the phase-3 discharge sites — it
        /// emits nothing and mutates nothing. The reasons match
        /// `dischargeContradiction` (contradiction, vacuous truth, CE-filter
        /// refutation) and `dischargeToBeProved` (toBeProved reached) exactly, so
        /// a `true` here
        /// is always confirmed by the authoritative phase-3 discharge: the
        /// triggering head — appended to the task's sealed-page record chain
        /// before this runs — is
        /// deposited by the still-`isActive` finalize and found by phase 3.
        /// CE-filter refutations are included (a non-negative `contradictionIndex`
        /// whose fired head negates a loaded fact); the post-burst
        /// `dischargeContradiction` CE branch then records the result.
        ///
        /// Read-only on `body` and on the shared `NameMap`: it uses
        /// `NameMap::lookup` (never the minting `encode`) and only reads
        /// `intKnownStatements` / `intToBeProved`, so it is safe to call from the
        /// parallel part-threads sharing one LB (D-116).
        /// `body` is non-const only to match the established parallel idiom
        /// (`checkLocalEncodedMemoryStatic`); no member is written.
        ///
        /// @param body The LB whose part is firing. Inspected read-only:
        ///        `primedForContradiction`, `isPartOfRecursion`,
        ///        `contradictionIndex`, `nameMap`, `intKnownStatements`,
        ///        `toBeProved`.
        /// @param fr The head firing record just produced. Must be a head
        ///        (`!fr.isMarker && !fr.isOrdis2Demand`); markers and
        ///        ordis2-demand records never deactivate (a demand's
        ///        `rplExpr2` is a demanded premise text, not a deposit).
        /// @return `true` iff one of the deactivation reasons holds for
        ///         `fr` (the LB is doomed); `false` otherwise.
        /// @invariant Mutates nothing; never mints in the `NameMap`.
        /// @see `dischargeContradiction` / `dischargeToBeProved` — the phase-3
        ///      sites this predicate mirrors and that perform the real discharge.
        /// @see `BurstSink` — the phase-2 consumer that, on a hit, lowers the
        ///      external per-LB doom line so the burst stops.
bool burstDeactivates(const Memory& body, const FiringRecord& fr) {
            assert(!fr.isMarker && !fr.isOrdis2Demand
                && "burstDeactivates inspects head records only");

            const bool atMain =
                equalSpans(StrSpan(fr.validityName), StrSpan("main", 4));

            // Reason "toBeProved reached": an LB — direct OR recursion — fired
            // the head closing its SOLE remaining main-scope goal
            // (toBeProved.size()==1). A single-chapter proof that reaches its goal is as
            // doomed as an induction one, so this is NOT gated on
            // isPartOfRecursion (gating it left single-chapter-proof LBs unable to
            // early-exit at all). The size()==1 guard keeps a multi-goal LB
            // alive: under LB-split its goals are owned by different parts and
            // the shared stop flag would otherwise cut a sibling before it fires
            // its own goal. Only the last goal truly dooms the LB.
            if (!parameters.compressor_mode && atMain) {
                // Non-minting packed probe — this predicate runs in the
                // phase-2 parallel read-only context, so it must not touch
                // the NameMap write side (same contract as the negation
                // scan below).
                // Sealed views materialize only on this rare branch (sole
                // remaining main-scope goal) — the string-keyed probe is
                // C++17-bound until the registries' cold migration.
                if (body.intToBeProved.count() == 1
                    && lookupToBeProved(body.intToBeProved, body.nameMap,
                                        fr.rplExpr2.toStdString(),
                                        fr.validityName.toStdString())
                       != 0) {
                    return true;
                }
            }

            // D-222: phase 2 must mirror the
            // authoritative discharge exactly. A branch-local head cannot doom
            // the LB even when its negation is inherited from `main`.
            if (!atMain) return false;

            // Reasons "contradiction" and "contradiction in vacuous truth" both
            // require neg(head) already known at `main`. Skip the probe unless
            // one of their LB-role preconditions can hold.
            const bool primed = body.primedForContradiction;
            const bool vacuousCandidate = body.isPartOfRecursion;
            // CE-filter LB: a non-negative contradictionIndex marks a conjecture
            // under test; it is refuted the instant a fired head's negation is a
            // loaded fact (CE LBs are never primed or recursion).
            const bool ceCandidate = body.contradictionIndex >= 0;
            if (!primed && !vacuousCandidate && !ceCandidate) return false;

            // Non-minting main-scope negation probe, mirroring
            // dischargeContradiction's detection. lookup returns 0 for an
            // un-interned negation, which is never a key in intKnownStatements.
            const NameId negId =
                body.nameMap.lookup(gl::DefinitionExpansion::negate(fr.rplExpr2.toStdString()));
            const StatementFlags* mainNegation =
                body.intKnownStatements.find(
                    StatementKey{ negId, NameMap::MAIN_ID });
            if (mainNegation == nullptr) return false;

            // Reason "contradiction" (incubator __contradiction__ LB),
            // "contradiction in vacuous truth" (induction step), or CE-filter
            // conjecture refuted (fired head's negation is a loaded fact).
            return primed || vacuousCandidate || ceCandidate;
        }

public:
// ------------------------------------------------------------------
    // Hash-engine lookup-and-fire. The largest single body in the cluster
    // (~300 LOC). Consumes one StaticRequest produced by the request generator
    // above and either fires the matched rule (path 1, head LMV) or installs
    // the marker admission (path 2, marker LMV).
    // ------------------------------------------------------------------

    /// @brief Static-pipeline rule lookup — given a `StaticRequest` keyed on
    /// an `IntNormalizedKey`, find matching LMVs in `localHashMemory` and
    /// dispatch the head-firing path.
    ///
    /// @details
    /// This is the consumer of the LMVs that `addToHashMemory` installed.
    /// The function executes int-only fast pre-hit checks before any string
    /// work, then on a hit decodes the relevant pieces and walks the
    /// admission / OR / equivalence-class machinery.
    ///
    /// Phase outline:
    /// 1. **Validity-id consensus.** Iterate `req.intExprs[i]->validityId`
    ///    for each request element; require pairwise `nm.comparable` so a
    ///    consensus deepest scope can be picked. A divergent set returns
    ///    early (the request cannot fire under any single scope).
    /// 2. **encodedMap lookup.** Probe `memoryBlock.localHashMemory.encodedMap[tple]`.
    ///    Each candidate `LocalMemoryValue` is then admitted against:
    ///    - the request's max-iteration (`req.maxIteration + 1` cap),
    ///    - level-set admissibility (`lmv.levels`),
    ///    - the `productOfDisintegration` D-32 marker (gates OR firing),
    ///    - and per-LMV equivalence-class state. The equivalence-class
    ///      gate routes through `applyEquivalenceClass` and emits
    ///      `equality1`/`equality2` per
    ///      [I-32](../../docs/agentic_swdd/30_invariants.md#i-32) /
    ///      [I-34](../../docs/agentic_swdd/30_invariants.md#i-34).
    /// 3. **Head emit.** Each surviving LMV produces a head-firing event:
    ///    `addExprToMemoryBlock` for the head, `mailOut` for any
    ///    cross-scope deposits per [I-26](../../docs/agentic_swdd/30_invariants.md#i-26),
    ///    origin record under the LMV's `justification` tag.
    /// 4. **Admission record-keeping.** Each negative outcome (rejected
    ///    head, exhausted depth) is recorded in `rejectedMap` so the same
    ///    rejection isn't re-emitted next iteration.
    ///
    /// `coreId` is the per-core mailbox identifier; combined with the
    /// receiver's `Memory*` it picks the per-core slot of `mailOut` so
    /// `proveKernel`'s parallel phase does not need cross-LB locks
    /// ([I-28](../../docs/agentic_swdd/30_invariants.md#i-28)).
    ///
    /// @param req          The static request to look up.
    /// @param memoryBlock  Owning LB.
    /// @param coreId       Logical core id of the calling thread.
    ///
    /// @pre  `req.normalizedKey.data` references a live gen scratch arena allocation on the
    ///       per-slot gen scratch arena.
    /// @post `memoryBlock.localHashMemory` may have new origin entries;
    ///       `memoryBlock.mailOut[coreId]` may have new pending deposits;
    ///       `memoryBlock.overallHashMemory.admissionStatusMap` /
    ///       `rejectedMap` may have new rejection records.
    /// @invariant [I-12](../../docs/agentic_swdd/30_invariants.md#i-12) — `addStatement`
    ///            applies equivalence classes to `!(=[a,b])` one-sidedly;
    ///            this function honours that on the head-emit path.
    /// @invariant [I-26](../../docs/agentic_swdd/30_invariants.md#i-26) — mail-out
    ///            implications/statements MAIN-ONLY; exprOriginMap
    ///            ALL-SCOPES.
    /// @invariant [I-28](../../docs/agentic_swdd/30_invariants.md#i-28) — cross-LB
    ///            writes during the parallel phase forbidden; the head
    ///            emit defers to post-`pool.join()` collectors via
    ///            `mailOut`.
    /// @see `inference/hash_rule_evaluator.hpp::addToHashMemory` — installer of the LMVs read here.
    /// @see `inference/hash_rule_evaluator.hpp::addExprToMemoryBlock` — head-emit consumer.
    /// @see `inference/hash_rule_evaluator.hpp::applyEquivalenceClass` — class-gated rewrite.
void checkLocalEncodedMemoryStatic(
            const StaticRequest& req,
            const Memory& memoryBlock,
            unsigned coreId,
            SealedPageSet& sealedPages);

public:
/// @brief Canonically order one hashburst's firing records when needed,
        ///        then apply their deposits to the LB.
        ///
        /// @details
        /// Consumer of the `FiringRecord`s produced by
        /// `checkLocalEncodedMemoryStatic` across one static request-evaluation
        /// pass — handed in as the LB's sealed part sets in part order, each
        /// carrying its records on its own record chain
        /// (`SealedPageSet::forEachRecord`, append order). Sorting before
        /// applying makes the cap-bounded `addOrigin`
        /// selection, the per-head `disintegrationSignals` last-write, and the
        /// `admissionKeysAlgebra` / `deferredIntegrationPreps` drain order a
        /// function of the firing SET rather than the request-generation ORDER —
        /// the determinism the LB split depends on (a partition of the rules
        /// reproduces the same merged deposits). The sort is a pointer INDEX
        /// permutation over the chains' stable payload addresses under the same
        /// strict total content order (I-77): a strict-total-order sort has a
        /// unique output sequence and content-identical duplicates are
        /// byte-indistinguishable in the deposit stream, so the index sort's
        /// deposit sequence is byte-identical to the former in-place record
        /// sort regardless of input permutation and sort algorithm. Head
        /// records deposit, when not already known,
        /// `sameIterationInternalMail.statements` + `.exprOriginMap` +
        /// `.disintegrationSignals`. Marker records feed `deferredIntegrationPreps`,
        /// `admissionKeysAlgebra`.
        ///
        /// @param memoryBlock LB whose containers receive the deposits.
        /// @param parts     The LB's sealed part sets in part order; each set's
        ///                  record chain is read in append order (never
        ///                  mutated).
        /// @param partCount Number of part sets (`>= 0`; 0 deposits nothing).
        /// @param firingRecordsCanonical Whether the concatenated record chains
        ///                  already carry the exact canonical content order. The
        ///                  processor producer passes `false`; the GPU sealing
        ///                  adapter passes `true` only after device ordering and
        ///                  doom-prefix compaction.
        /// @see `memory.cpp::checkLocalEncodedMemoryStatic` — the producer.
        /// @see `FiringRecord` — the record shape.
        /// @see `D-117`.
void applyFiringRecords(Memory& memoryBlock,
            SealedPageSet* const* parts, int32_t partCount,
            bool firingRecordsCanonical);
private:
const ProverParameters& parameters;
const CompiledDefinitionRegistry& definitions;
AdmissionPolicy& admissionPolicy;
const LogicBlockIdentity& blockIdentity;
static thread_local GpuEvaluationUsage g_gpuEvaluationUsage;
};
}
