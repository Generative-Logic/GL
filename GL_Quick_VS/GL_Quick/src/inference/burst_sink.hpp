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
#include "hash_rule_evaluator.hpp"
#include "burst_stop_line.hpp"
namespace gl {
namespace tests { struct BurstSinkFixture; }
    /// @brief Evaluate processor requests through the deterministic early-exit prefix.
    ///
    /// @details
    /// Dependency-passing requests append complete firing records. A deactivating
    /// head publishes the original packed growth-position/part cutoff and ends
    /// its producer stream. Other parts cover the same deterministic prefix.
    /// Actual goal closure and discharge remain Phase 3 operations.
    ///
    /// @see StaticRequestEmitter, HashRuleEvaluator::applyFiringRecords.
    class BurstSink {
friend struct tests::BurstSinkFixture;
private:
        HashRuleEvaluator* evaluator;
        const Memory* body;
        unsigned coreId;
        SealedPageSet* sealedPages;  // this task's record + string storage (D-164)
        std::atomic<int64_t>* doomLine;  // external per-LB packed (position, ordinal) stop line
        int32_t partOrdinal;   // this part's deterministic invocation ordinal (bucket index; 0 unsplit)
        SealedRecordCursor<FiringRecord> cursor;  // early-exit read-back over the record chain
        int produced = 0;   // emitted-request counter (diagnostic traps); int to avoid overflow
        bool selfStopped = false;  // this part fired its own doom trigger (consume returned false)

public:
/// @brief Bind one streaming request consumer.
/// @details Binds the existing inputs without performing proof work or moving storage.
/// @param evaluator Rule evaluation operation.
/// @param body Resident block read view.
/// @param coreId Worker scratch slot.
/// @param sealedPages Caller-owned output pages.
/// @param doomLine Shared deterministic stopping line.
/// @param partOrdinal Deterministic part ordinal.
/// @param cursor Read cursor over this part's output records.
/// @return None; construction binds the supplied references.
BurstSink(HashRuleEvaluator* evaluator, const Memory* body, unsigned coreId, SealedPageSet* sealedPages, std::atomic<int64_t>* doomLine, int32_t partOrdinal, SealedRecordCursor<FiringRecord> cursor);
/// @brief Read the number of requests consumed by this part.
/// @details The existing counter increments before dependency checks.
/// @return Requests seen by this sink.
int producedCount() const { return produced; }
/// @brief Check whether this deterministic request stream can continue.
/// @details Stops after its own trigger or strictly beyond the shared growth position.
/// @return True while this part may accept a request.
        bool canAccept() const {
            // No submatch cap: a burst without a doom trigger runs to COMPLETION.
            // A heavy LB is split preemptively next iteration (proveKernel's
            // stats-driven trigger), never truncated mid-burst. Only the doom
            // line halts a burst, and it does so DETERMINISTICALLY for split
            // LBs too: a part stops once its OWN submatch counter lies strictly
            // past the line's position, so every part covers the identical
            // deterministic prefix of its stream regardless of thread timing
            // (a stale line read only delays the stop, and the merge takes the
            // winning part's chain alone, so a late-stopping sibling's tail is
            // discarded — timing shapes wall-clock, never output). Strictly
            // past — a part AT the line's position may still publish its own
            // position-tie trigger, which an earlier invocation ordinal must
            // win. The CE filter runs single-part and relies on the line to
            // detect its refuting contradiction. selfStopped covers the
            // triggering part itself: its consume returned false at the
            // trigger, which skips its remaining request batches.
            if (selfStopped) return false;
            return HashRequestGenerator::growthMatchCount()
                <= doomLinePosition(
                       doomLine->load(std::memory_order_relaxed));
        }

/// @brief Evaluate one request and inspect newly emitted records.
/// @details Keeps the established dependency and deterministic early-exit decisions.
/// @param req Complete request whose premise views stay valid during evaluation.
/// @return True when the generator should continue.
        bool consume(const StaticRequest& req) {
            ++produced;
            HashRuleEvaluator::noteRequest();
            // Dependency skip (verbatim from the former FIXPOINT_LOOP): every
            // request element must already be a known statement at this LB.
            for (NameId s = 0; s < req.count; ++s) {
                const StatementFlags* kf = body->intKnownStatements.find(StatementKey{
                    req.intExprs[s]->originalId, req.intExprs[s]->validityId });
                if (kf == nullptr) {
                    return true;  // counted but not fired; keep generating
                }
            }
            HashRuleEvaluator::noteDependencyPassRequest();
            {
                // RT: request evaluation + firing, separated from the
                // generation scope that emitted the request (the former
                // FIXPOINT_LOOP work, folded into this sink). Scope per
                // FIRED request — the dependency skip above returns
                // before this point, so the open/close cost never rides
                // the billions-iteration pairing interior.
                RT_SCOPE_HERE("STATIC_REQGEN_FIRE_EVAL");
                evaluator->checkLocalEncodedMemoryStatic(req, *body, coreId,
                                                    *sealedPages);
            }
            // Early-exit: when a fired head proves the LB doomed (contradiction /
            // vacuous-truth / toBeProved reached), stop this part's burst and
            // publish the trigger's deterministic stream position. Every part —
            // split or not — runs this scan on its own freshly appended records;
            // a hit publishes (own submatch counter, own invocation ordinal)
            // onto the LB's shared doom line by CAS-min, so the final line is
            // the lexicographic minimum over every published trigger no matter
            // in which wall-clock order near-simultaneous triggers arrive. The
            // winner — the part named by the final line — is a pure function of
            // the parts' deterministic streams: it can never be cut before its
            // own trigger, because the line's position never drops below the
            // winning position and canAccept stops only strictly past it. The
            // merge then takes the winner's chain alone, so the losing parts'
            // timing-dependent tails never reach the LB. Sound because
            // burstDeactivates mirrors the phase-3 discharge exactly (it never
            // over-exits): the LB discharges regardless, so every culled
            // request is wasted-once-doomed. Markers never deactivate. The
            // cursor scan covers exactly this call's appended records: every
            // prior consume that appended records also drained the cursor to
            // the tail before returning true, and a consume that returned false
            // ended this part's burst (canAccept gates generation), so "not yet
            // visited" == "appended by THIS call".
            while (const FiringRecord* fr = cursor.next()) {
                // Demand records are their OWN kind: rplExpr2 is a
                // DEMANDED premise text, never a deposited statement —
                // inspecting it as a head would be a false-deactivation
                // path (D-267).
                if (fr->isMarker || fr->isOrdis2Demand) continue;
                if (evaluator->burstDeactivates(*body, *fr)) {
                    lowerDoomLine(*doomLine,
                        packDoomLine(
                            HashRequestGenerator::growthMatchCount(),
                            partOrdinal));
                    selfStopped = true;
                    return false;
                }
            }
            return true;
        }
    };
}

namespace gl {
/// @brief Bind one streaming request consumer.
/// @details Binds the existing inputs without performing proof work or moving storage.
/// @param evaluator Rule evaluation operation.
/// @param body Resident block read view.
/// @param coreId Worker scratch slot.
/// @param sealedPages Caller-owned output pages.
/// @param doomLine Shared deterministic stopping line.
/// @param partOrdinal Deterministic part ordinal.
/// @param cursor Read cursor over this part's output records.
/// @return None; construction binds the supplied references.
inline BurstSink::BurstSink(HashRuleEvaluator* evaluator, const Memory* body, unsigned coreId, SealedPageSet* sealedPages, std::atomic<int64_t>* doomLine, int32_t partOrdinal, SealedRecordCursor<FiringRecord> cursor) : evaluator(evaluator), body(body), coreId(coreId), sealedPages(sealedPages), doomLine(doomLine), partOrdinal(partOrdinal), cursor(cursor) {}
} // namespace gl
