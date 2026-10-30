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
#include "logic_block_operations.hpp"
namespace gl {
/// @brief Bind inputs for logic block operations.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param blockIdentity Borrowed LogicBlockIdentity interface or state; must outlive this component.
/// @param ceFilteringActive Borrowed bool interface or state; must outlive this component.
/// @param contradictionTable Borrowed std::vector interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param rejectedSources Borrowed RejectedSourceMemory interface or state; must outlive this component.
/// @param blockMailProcessor Borrowed BlockMailProcessor interface or state; must outlive this component.
/// @param caseProcessor Borrowed CaseProcessor interface or state; must outlive this component.
/// @param equalityPropagation Borrowed EqualityPropagation interface or state; must outlive this component.
/// @param equivalenceProcessor Borrowed EquivalenceProcessor interface or state; must outlive this component.
/// @param goalProcessor Borrowed GoalProcessor interface or state; must outlive this component.
/// @param implicationSubproofs Borrowed ImplicationSubproofs interface or state; must outlive this component.
/// @param orImplicationProcessor Borrowed OrImplicationProcessor interface or state; must outlive this component.
/// @param statementMemory Borrowed StatementMemory interface or state; must outlive this component.
/// @param unaryIntegrationProcessor Borrowed UnaryIntegrationProcessor interface or state; must outlive this component.
/// @param watchProcessor Borrowed WatchProcessor interface or state; must outlive this component.
/// @param rejectedSourceReplay Borrowed RejectedSourceReplay (the demand-built OR event drain); must outlive this component.
/// @param phase13TimingRows Borrowed per-worker timing storage, available when detailed phase timing is enabled.
/// @param phase13TimingWorkers Borrowed unsigned interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
LogicBlockOperations::LogicBlockOperations(LogicBlockIdentity& blockIdentity, const bool& ceFilteringActive, std::vector<ContradictionItem>& contradictionTable, const ProverParameters& parameters, RejectedSourceMemory& rejectedSources, BlockMailProcessor& blockMailProcessor, CaseProcessor& caseProcessor, EqualityPropagation& equalityPropagation, EquivalenceProcessor& equivalenceProcessor, GoalProcessor& goalProcessor, ImplicationSubproofs& implicationSubproofs, InductionUsefulness& inductionUsefulness, OrImplicationProcessor& orImplicationProcessor, StatementMemory& statementMemory, UnaryIntegrationProcessor& unaryIntegrationProcessor, WatchProcessor& watchProcessor, ContradictionDemand& contradictionDemand, RejectedSourceReplay& rejectedSourceReplay
#if PHASE13_DEEP_TIMING
, int64_t*& phase13TimingRows
#endif

#if PHASE13_DEEP_TIMING
, unsigned& phase13TimingWorkers
#endif
) : blockIdentity(blockIdentity), ceFilteringActive(ceFilteringActive), contradictionTable(contradictionTable), parameters(parameters), rejectedSources(rejectedSources), blockMailProcessor(blockMailProcessor), caseProcessor(caseProcessor), equalityPropagation(equalityPropagation), equivalenceProcessor(equivalenceProcessor), goalProcessor(goalProcessor), implicationSubproofs(implicationSubproofs), inductionUsefulness(inductionUsefulness), orImplicationProcessor(orImplicationProcessor), statementMemory(statementMemory), unaryIntegrationProcessor(unaryIntegrationProcessor), watchProcessor(watchProcessor), contradictionDemand(contradictionDemand), rejectedSourceReplay(rejectedSourceReplay)
#if PHASE13_DEEP_TIMING
, phase13TimingRows(phase13TimingRows)
#endif

#if PHASE13_DEEP_TIMING
, phase13TimingWorkers(phase13TimingWorkers)
#endif
 {}

// Turn optimizations OFF for just this section
//#pragma optimize("", off)

// The hash-engine bodies live in memory.cpp: addToHashMemory(),
// makeNormalizedKeysForAdmission(), lessByName(), lessByOriginal(),
// filterIntEncodedStatements(), generateEncodedRequestsStatic(), and
// checkLocalEncodedMemoryStatic().

// ========================================================================
// Static pipeline: IntEncodedExpr-based, zero-alloc request generation
// ========================================================================

//#pragma optimize("", off)
/// @brief Install the dead-end variable-copy equalities for every
/// hypo-scoped validity before standard class application.
///
/// @details
/// `disintegrateExprHypothetically` opens validity scopes named
/// `parent + "_boundary_" + "_var0_<x>[_var1_<y>...]" + "_hypo_" + expr`,
/// one per hypothetically disintegrated integration goal. Both ordinary
/// standard-processing calls react after input clearing and before class
/// application, with one worker owning the LB. This function walks every equivalence-class scope of the LB and, on each
/// hypo-scoped row, checks every class member that is one of the payload
/// variables: when ALL of its class-mates are forced or ordinary integration witness names and at
/// least one class-mate exists, the dead-end equality `(=[var,var_copy])`
/// is installed at the PARENT scope via `addExprToMemoryBlock` with the
/// `variableCopy` origin (the "variable copy" proof tag).
///
/// The interior is statified
/// (I-142; 09b/09c cookbooks):
/// - Row order: a validity-id array on the per-slot gen-scratch byte-bump
///   tier, sorted by `compareSpans(decodeView(a), decodeView(b))` — the
///   decoded-name lex order of the retired string-pair snapshot, never id
///   order (I-84). Tie-free (distinct vids decode to distinct names) and
///   mint-free for the whole sort window (I-3).
/// - Row names: the cheap `_hypo_` gate runs on the raw `decodeView`
///   span; a passing row is copied to a per-row `ScratchString` on the
///   string-scratch tier so the `parseHypoScopeVars` spans survive the
///   class loop's NameMap mints.
/// - Per-row class read: an arena MEMBER-RUN SNAPSHOT (`memberPool` /
///   `classStarts`, gen-scratch page tier, cleared per row) taken BEFORE
///   the class loop. The loop's own `addExprToMemoryBlock` can splice
///   this same map's shared blob pool through a DIFFERENT key (the
///   parent-scope equality), so live blob views would both see mid-row
///   mutations the retired heap decode never saw and dangle; the snapshot
///   reproduces the heap `decodeClassesById` isolation exactly, and is
///   content-complete because the loop reads only member ids. Bucket
///   content is read at row start — the decode-at-the-point-of-processing
///   semantics: an earlier row's mutations to a later row's bucket ARE
///   seen, exactly as before.
/// - Firing product: exact-length `ScratchString` build of
///   `(=[var,var_copy])` plus the byte-exact `removeUPrefixScratch` twin;
///   the add door receives byte-identical spans, levels, origin, and
///   flags in the identical call order.
///
/// The door levels ride a hoisted one-element stack run (`{ mb.level }`):
/// `addExprToMemoryBlock` takes an ascending-unique `(const int*, int32_t)`
/// run, caller-owned per the levels-run chain contract. `mb.level` is
/// written only at LB creation, so one run per call equals the retired
/// one set per firing.
///
/// @param mb The LB whose equivalence-class scopes are surveyed; the add
///           door mints into `mb.nameMap` under the owning worker claim.
/// @see parseHypoScopeVars, HypoScopeParse — the name parse;
///      `disintegrateExprHypothetically` — the hypo-scope producer;
///      `EqClassNameCaches::kindOf` — the class-mate tier probe.
void LogicBlockOperations::reactToHypo(Memory& mb) {
    const StrSpan kHypo("_hypo_", 6);

    // Per-slot scratch arenas (the pendingWipeScopes drain idiom below):
    // gen-scratch hosts the vid array + snapshot pools + blob straddle
    // peeks; the string arena hosts the per-row name copy + the firing
    // product builds. Never mixed (I-124).
    const unsigned slot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : genScratchArenas().slotCount() - 1;
    ScratchArena& gArena = genScratchArenas().forSlot(slot);
    ScratchArena& strArena = scratchArenas().forSlot(
        (gl::WorkerContext::coreId() >= 0) ? static_cast<unsigned>(gl::WorkerContext::coreId())
                               : scratchArenas().slotCount() - 1);
    const ArenaOffset gMark = gArena.cursor();

    // Iterate all equivalence-class scopes via a decoded-name lex-sorted
    // vid snapshot — identical iteration order to the former string-keyed
    // map, never id order (I-84). The sort window is mint-free, so the
    // comparator's decodeView spans stay valid throughout (I-3).
    const int32_t n = mb.equivalenceClassesMap.count();
    if (n == 0) return;
    NameId* vids = reinterpret_cast<NameId*>(gArena.resolve(
        gArena.alloc(n * static_cast<int32_t>(sizeof(NameId)),
                     alignof(NameId))));
    for (int32_t kid = 1; kid <= n; ++kid)
        vids[kid - 1] = mb.equivalenceClassesMap.keyAt(kid);
    std::sort(vids, vids + n, [&](NameId a, NameId b) {
        return compareSpans(mb.nameMap.decodeView(a),
                            mb.nameMap.decodeView(b)) < 0;
    });

    // Member-run snapshot pools — declared once, cleared per row (the
    // cleanUpExpressions idiom). Page tier; freed by the destructors.
    DirtyState poolDirty = DirtyState::Clean;
    PagedVector<NameId> memberPool(&gArena, &poolDirty);
    PagedVector<int32_t> classStarts(&gArena, &poolDirty);

    // Hoisted door levels run — one per call; mb.level is written only at
    // LB creation, so every retired per-firing set held this same value.
    const int lvRun[1] = { mb.level };

    for (int32_t r = 0; r < n; ++r) {
        const NameId classesValId = vids[r];
        if (mb.phase3IgnoresScope(classesValId)) continue;

        // Cheap pre-filter on the raw span (gate 1; pure read). The full
        // parse below re-runs it on the owned copy — byte-identical
        // content, so the double check is unobservable.
        const StrSpan rawName = mb.nameMap.decodeView(classesValId);
        if (rfindSpanBefore(rawName, kHypo, rawName.len) == -1) continue;

        // Own the row name on the string tier: the parse spans (parent +
        // vars) alias it and must survive the class loop's NameMap mints.
        ScratchScope rowScope(strArena);
        const ScratchString rowName =
            ScratchString::copyFrom(strArena, rawName.ptr, rawName.len);
        HypoScopeParse parse;
        if (!parseHypoScopeVars(StrSpan(rowName), parse)) continue;

        // Member-run snapshot for THIS row, read at row start (current
        // content — see the Doxygen). The class loop reads ONLY the pools;
        // its addExprToMemoryBlock may splice the class map's blob pool.
        memberPool.clear();
        classStarts.clear();
        const int32_t bucketId = mb.equivalenceClassesMap.lookup(classesValId);
        const int32_t classCount =
            (bucketId == 0) ? 0 : mb.equivalenceClassesMap.runLen(bucketId);
        for (int32_t j = 0; j < classCount; ++j) {
            int32_t blen = 0;
            const char* bp = mb.equivalenceClassesMap.peekRecordBytes(
                bucketId, j, blen, gArena);
            const EquivalenceClassView cls{ bp, blen };
            classStarts.push_back(static_cast<int32_t>(memberPool.size()));
            const int32_t memberCount = cls.memberCount();
            for (int32_t k = 0; k < memberCount; ++k)
                memberPool.push_back(cls.memberId(k));
        }

        // Check the equivalence classes (pool reads only).
        for (int32_t j = 0; j < classCount; ++j) {
            const int32_t mStart = classStarts[j];
            const int32_t mEnd = (j + 1 < classCount)
                ? classStarts[j + 1]
                : static_cast<int32_t>(memberPool.size());
            for (int32_t mi = mStart; mi < mEnd; ++mi) {
                const NameId varId = memberPool[mi];
                // decodeView is safe here: the span's last read (the
                // equality fill below) completes before the add door
                // mints, and kindOf writes only kindById_ — a different
                // container (I-3); the next iteration re-decodes fresh.
                const StrSpan var = mb.nameMap.decodeView(varId);

                // Membership in the payload variables — linear equalSpans
                // over the insertion-order parse array (set order was
                // unobservable: membership + empty only).
                bool isTarget = false;
                for (int32_t vi = 0; vi < parse.varCount; ++vi) {
                    if (equalSpans(parse.vars[vi], var)) {
                        isTarget = true;
                        break;
                    }
                }
                if (!isTarget) continue;

                // Check "other" variables constraint (id compares, as the
                // retired memberIds loop did).
                bool conditionMet = true;
                bool hasOthers = false;
                for (int32_t oi = mStart; oi < mEnd; ++oi) {
                    const NameId otherId = memberPool[oi];
                    if (otherId == varId) continue;

                    hasOthers = true;
                    // Every other member must be a disintegration witness.
                    if (!isDisintegrationWitnessShape(mb.nameMap.decodeView(otherId))) {
                        conditionMet = false;
                        break;
                    }
                }

                // If yes, add (=[var, var_copy])
                if (conditionMet && hasOthers) {
                    // Exact-length fill of "(=[" var "," var "_copy])" on
                    // the string tier (09c Section 4 doctrine).
                    const int32_t eqLen = 2 * var.len + 11;
                    char* buf = strArena.allocBytes(eqLen);
                    int32_t at = 0;
                    std::memcpy(buf + at, "(=[", 3); at += 3;
                    std::memcpy(buf + at, var.ptr,
                                static_cast<size_t>(var.len));
                    at += var.len;
                    buf[at++] = ',';
                    std::memcpy(buf + at, var.ptr,
                                static_cast<size_t>(var.len));
                    at += var.len;
                    std::memcpy(buf + at, "_copy", 5); at += 5;
                    buf[at++] = ']';
                    buf[at++] = ')';
                    assert(at == eqLen);
                    const ScratchString equalityExpr =
                        ScratchString::wrap(strArena, buf, eqLen);

                    const ScratchString finalExpr =
                        gl::DefinitionExpansion::removeUPrefixScratch(strArena, StrSpan(equalityExpr));

                    // Dead-end axiom — see "variable copy" tag
                    const TransientOrigin origin{
                        true, OriginTag::variableCopy, nullptr, 0 };

                    statementMemory.addExprToMemoryBlock(StrSpan(finalExpr),
                        mb,
                        -1, // iteration
                        0,  // status (Local statement)
                        lvRun, 1,
                        origin,
                        -1, // coreId
                        -1, // auxyIndex
                        parse.parentValidity, // the extracted parent scope
                        false);
                }
            }
        }
    }
    gArena.popTo(gMark);   // vid array + straddle peeks; pools are page-tier
}
}
