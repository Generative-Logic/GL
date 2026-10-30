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
#include "goal_processor.hpp"
namespace gl {
/// @brief Bind inputs for goal processor.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param canonicalizer Borrowed Canonicalizer interface or state; must outlive this component.
/// @param contradictionTable Borrowed std::vector interface or state; must outlive this component.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @param inductionMemoryBlocks Borrowed std::vector interface or state; must outlive this component.
/// @param inductionMemoryBlocksMutex Borrowed std::mutex interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param updateGlobalMutex Borrowed std::mutex interface or state; must outlive this component.
/// @param updateGlobalSingleChapterMutex Borrowed std::mutex interface or state; must outlive this component.
/// @param updateGlobalSingleChapterPages Borrowed std::vector interface or state; must outlive this component.
/// @param updateGlobalTuples Borrowed std::vector interface or state; must outlive this component.
/// @param blockKeyReader Borrowed BlockKeyReader interface or state; must outlive this component.
/// @param caseProcessor Borrowed CaseProcessor interface or state; must outlive this component.
/// @param implicationSubproofs Borrowed ImplicationSubproofs interface or state; must outlive this component.
/// @param proofHistory Borrowed ProofHistory interface or state; must outlive this component.
/// @param scopeLifecycle Borrowed ScopeLifecycle interface or state; must outlive this component.
/// @param statementMemory Borrowed StatementMemory interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
GoalProcessor::GoalProcessor(Canonicalizer& canonicalizer, std::vector<ContradictionItem>& contradictionTable, const CompiledDefinitionRegistry& definitions, std::vector<Memory*>& inductionMemoryBlocks, std::mutex& inductionMemoryBlocksMutex, const ProverParameters& parameters, std::mutex& updateGlobalMutex, std::mutex& updateGlobalSingleChapterMutex, std::optional<SealedPageSet>& updateGlobalSingleChapterPages, std::vector< std::tuple<int, uint64_t, int>>& updateGlobalTuples, BlockKeyReader& blockKeyReader, CaseProcessor& caseProcessor, ImplicationSubproofs& implicationSubproofs, ProofHistory& proofHistory, ScopeLifecycle& scopeLifecycle, StatementMemory& statementMemory) : canonicalizer(canonicalizer), contradictionTable(contradictionTable), definitions(definitions), inductionMemoryBlocks(inductionMemoryBlocks), inductionMemoryBlocksMutex(inductionMemoryBlocksMutex), parameters(parameters), updateGlobalMutex(updateGlobalMutex), updateGlobalSingleChapterMutex(updateGlobalSingleChapterMutex), updateGlobalSingleChapterPages(updateGlobalSingleChapterPages), updateGlobalTuples(updateGlobalTuples), blockKeyReader(blockKeyReader), caseProcessor(caseProcessor), implicationSubproofs(implicationSubproofs), proofHistory(proofHistory), scopeLifecycle(scopeLifecycle), statementMemory(statementMemory) {}

/// @brief End-of-burst drain of the settled-goal inbox — close the
///        MAIN goal a contradiction twin settled: a disproved goal
///        loses its whole integration machinery, a proved goal closes
///        with the ordinary success-path semantics.
///
/// @details
/// The post-join theorem drain deposits the contradiction seed of every
/// emitting primed `__contradiction__` LB onto its parent's
/// `Memory::pendingDisprovedGoals` (the theorem head is `negate(seed)`
/// by I-165). This drain — running single-threaded at the end of the
/// parent's own burst, immediately BEFORE the `pendingWipeScopes`
/// drain so the wipes land in the same burst — probes each seed
/// against the MAIN goals in `intToBeProved`, in both directions.
///
/// PROOF direction (D-279): when
/// `negate(seed)` is a registered MAIN goal, the emitted theorem's
/// head IS that goal — the goal succeeded. The drain erases the goal
/// row and queues the goal-embedding MAIN scope roots (the same three
/// shapes as below) on `pendingWipeScopes` +
/// `intValidityNamesToFilter`, and does nothing else: no origin
/// erase (Rule 16 / I-44 — a proved goal's history stays), no
/// gate/cohort erase, no closed-subproof exception. Emission and twin
/// retirement already happened at the deposit seam. Closing the row
/// lets the I-48 sweeps retire the goal's LB chain even when the
/// level-gated local discharge could never fire (a
/// contradiction-scope proof may legitimately carry a narrow level
/// run).
///
/// DISPROOF direction: when the seed itself is a registered MAIN
/// goal, the verbatim twin fired — the goal is disproved and the
/// cleanup below triggers. A double miss is a defined no-op (a goal
/// already closed):
///
/// 1. the goal row leaves `intToBeProved`;
/// 2. every MAIN-level scope whose payload embeds the goal — the
///    `<goal>_subproof_<bare>` subproof/orint scopes
///    (I-170) and the `_var…_hypo_<goal>` working
///    scope — is queued on `pendingWipeScopes` (+ the synchronous
///    `intValidityNamesToFilter` insert, D-73), so the existing
///    radical subtree wipe removes the whole nested machinery.
///    EXCEPTION: a subproof root whose product (the bare compact /
///    OR signature) is a REGISTERED statement at main is skipped
///    whole — that subproof CLOSED before the disproof, its scope was
///    already wiped by the success path, its product is a legitimate
///    proven fact, and its preserved history is what the chapter
///    walker reads;
/// 3. the goal-template gates leave `integrationPrepared` /
///    `integrationPreparedMarker` (MAIN rows matching the goal, its
///    marker forms, or a goal-prefixed template) and
///    `integrationStartIntMap`;
/// 4. `exprOriginMap` rows keyed in the wiped subtrees, plus MAIN rows
///    whose key CONTAINS the goal text — excepting the disproof
///    products `negate(goal)` and the anchor-rooted theorem rows —
///    are erased (the maintainer-approved deviation from the
///    wipeSubtree I-44 preserve rule — the disproof's own record
///    lives in the `__contradiction__` LB per D-51);
/// 5. internal-mail origin runs at wiped scopes are erased
///    (`ColdMail::filterOrigins`; the statements are filtered by the
///    subtree wipe itself) so no orphaned history rows outlive their
///    statements;
/// 6. OR cohort bookkeeping (`orDisjunctCount` / `orBookkeeping`)
///    whose cohort parent validity lies in a wiped subtree is erased —
///    a dead parent scope means the cohort can never act again.
///
/// Known limitation (flagged, not silent): the seed is matched
/// verbatim in both directions; a MAIN goal rewritten by equivalence
/// classes between spawn and settlement would miss the
/// `intToBeProved` probe. Incubator batches run with classes skipped,
/// so the contradiction pipeline never hits this; see the chapter
/// Weaknesses entry.
///
/// @param memoryBlock The LB whose inbox is drained (its containers
///                    are resident — the drain runs inside the LB's
///                    own burst).
/// @invariant Deterministic: seeds process in byte-lex order; every
///            erase predicate is a pure function of container content.
/// @see Memory::pendingDisprovedGoals, Memory::wipeSubtree,
///      splitSubproofPayload, templateMatchesGoalWithMarkers.
void GoalProcessor::drainDisprovedGoals(Memory& body) {
    const int32_t seedN = body.pendingDisprovedGoals.count();
    if (seedN == 0) return;
    // Quiescence (D-194): the cleanup eradicates state; flag the mutation
    // directly rather than relying on a count diff.
    body.mutatedThisBurst = true;

    const unsigned gSlot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : genScratchArenas().slotCount() - 1;
    ScratchArena& gArena = genScratchArenas().forSlot(gSlot);
    const ArenaOffset gMark = gArena.cursor();
    const unsigned sSlot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : scratchArenas().slotCount() - 1;
    ScratchArena& sArena = scratchArenas().forSlot(sSlot);
    ScratchScope sScope(sArena);

    // Snapshot the seed bytes onto the string tier (the inbox resets below so
    // wipe-time inserts elsewhere run against a fresh container), then sort
    // byte-lex for a deterministic processing order (I-84). The spans alias
    // sArena bytes under the function-level sScope — nothing rewinds sArena
    // below them (09c pitfall 4 discipline).
    StrSpan* seeds = reinterpret_cast<StrSpan*>(gArena.resolve(
        gArena.alloc(seedN * static_cast<int32_t>(sizeof(StrSpan)),
                     static_cast<int32_t>(alignof(StrSpan)))));
    for (int32_t i = 1; i <= seedN; ++i) {
        const StrSpan v = body.pendingDisprovedGoals.view(i);
        char* buf = sArena.allocBytes(v.len);
        std::memcpy(buf, v.ptr, static_cast<size_t>(v.len));
        seeds[i - 1] = StrSpan(buf, v.len);
    }
    body.pendingDisprovedGoals.resetToFresh();
    std::sort(seeds, seeds + seedN, [](const StrSpan& a, const StrSpan& b) {
        return compareSpans(a, b) < 0;
    });

    // Shared goal-scope enumeration for both settlement directions below:
    // visit every MAIN-parented scope whose payload embeds `goal` — the
    // `<goal>_subproof_<bare>` shape (split into gOut/bOut) or the
    // `_var…_hypo_<goal>` working scope — and hand each hit to
    // onMatch(id, isSubproof, bOut). The hypothetical-disintegration sentinel
    // is not visited: it was wiped at the end of the step that made it
    // (D-496). Non-minting throughout (I-3).
    const auto forEachGoalScopeRoot = [&](const StrSpan goal, auto&& onMatch) {
        static const char HYPO_TOK[] = "_hypo_";
        const int32_t hypoLen = static_cast<int32_t>(sizeof(HYPO_TOK) - 1);
        for (NameId id = 2; id <= body.nameMap.nameCount(); ++id) {
            if (body.nameMap.stackEmpty(id)) continue;
            if (body.nameMap.parentOf(id) != NameMap::MAIN_ID) continue;
            const StrSpan payload =
                body.nameMap.decodeSubView(body.nameMap.stackBack(id));
            StrSpan gOut, bOut;
            if (gl::ScopeSyntax::splitSubproofPayload(payload, gOut, bOut)) {
                if (equalSpans(gOut, goal))
                    onMatch(id, /*isSubproof=*/true, bOut);
            } else if (payload.len > hypoLen + goal.len
                       && payload.ptr[0] == '_'
                       && equalSpans(StrSpan(payload.ptr + payload.len
                                                 - goal.len, goal.len),
                                     goal)
                       && std::memcmp(payload.ptr + payload.len - goal.len
                                          - hypoLen,
                                      HYPO_TOK,
                                      static_cast<size_t>(hypoLen)) == 0) {
                onMatch(id, /*isSubproof=*/false, StrSpan());
            }
        }
    };

    for (int32_t si = 0; si < seedN; ++si) {
        const StrSpan seed = seeds[si];

        // The seed's negation (double-negation cancelling). Both directions
        // read it: for a PROOF settlement it IS the parent goal the twin
        // proved (I-165: theorem head == gl::DefinitionExpansion::negate(seed)); for a DISPROOF hit it
        // is the disproof PRODUCT, which the MAIN-keyed origin sweep below
        // must never erase.
        StrSpan negSeed;
        if (seed.len > 0 && seed.ptr[0] == '!') {
            negSeed = StrSpan(seed.ptr + 1, seed.len - 1);
        } else {
            char* nb = sArena.allocBytes(seed.len + 1);
            nb[0] = '!';
            std::memcpy(nb + 1, seed.ptr, static_cast<size_t>(seed.len));
            negSeed = StrSpan(nb, seed.len + 1);
        }

        // ---- PROOF direction (D-279) ----
        // The twin emitted the theorem headed gl::DefinitionExpansion::negate(seed); when that head is
        // a registered MAIN goal here, the goal SUCCEEDED — close it with the
        // ordinary success-path semantics: erase the row, stage the goal's
        // scope wipes (wipeSubtree preserve rules — exprOriginMap stays,
        // Rule 16 / I-44), and nothing else. No origin erase, no gate/cohort
        // erase, no closed-subproof exception: those are disproof-only
        // scrubbing of a dead goal's machinery; a proved goal's history and
        // products stay like any proved statement's. Twin retirement (and,
        // when the sealed level verdict allowed it, registration) already
        // happened at the deposit seam (drainUpdateGlobalSingleChapter ->
        // deactivateUnnecessary); a level-refused settlement still closes
        // here (D-278).
        {
            const NameId provedGoalId = body.nameMap.lookup(negSeed);
            if (provedGoalId != 0) {
                const int64_t provedPk =
                    packStatementKey(provedGoalId, NameMap::MAIN_ID);
                if (body.intToBeProved.lookup(provedPk) != 0) {
                    body.intToBeProved.eraseSet(provedPk);
                    forEachGoalScopeRoot(negSeed,
                        [&](NameId id, bool, StrSpan) {
                            body.pendingWipeScopes.mint(id);
                            body.intValidityNamesToFilter.mint(id);
                        });
                }
            }
        }

        // ---- DISPROOF direction ----
        const NameId goalId = body.nameMap.lookup(seed);
        if (goalId == 0) continue;   // seed never a goal here
        const int64_t goalPk = packStatementKey(goalId, NameMap::MAIN_ID);
        if (body.intToBeProved.lookup(goalPk) == 0) continue;  // already closed

        // ---- 1. the goal row ----
        body.intToBeProved.eraseSet(goalPk);

        // ---- 2. MAIN-level scope roots that embed the goal ----
        // Subtree membership for the erases below is the TEXT gate
        // (equal-or-boundary-prefix against the root names): the origin /
        // cohort key spaces are originInterner / lbStateInterner, never the
        // NameMap, so a NameMap-id bitmap cannot serve them (I-91, I-93).
        //
        // A subproof root whose PRODUCT already closed (the bare compact /
        // OR signature is a REGISTERED statement at main) is SKIPPED
        // entirely: its scope was wiped by the success path at closure, the
        // product is a legitimate proven fact other proofs may cite, and its
        // preserved history rows are what the chapter walker reads — only
        // never-closed machinery is dead.
        DirtyState rootsDirty = DirtyState::Clean;
        PagedVector<StrSpan> rootNames(&gArena, &rootsDirty);
        static const char ORINT_PFX[] = "orint_";
        const int32_t orintLen = static_cast<int32_t>(sizeof(ORINT_PFX) - 1);
        forEachGoalScopeRoot(seed,
            [&](NameId id, bool isSubproof, StrSpan bOut) {
                if (isSubproof) {
                    // Derive the subproof's product: the bare compact
                    // (Case A) or the OR signature (Case OR branch).
                    StrSpan product;
                    if (bOut.len > 0 && bOut.ptr[0] == '(') {
                        product = bOut;
                    } else if (bOut.len > orintLen
                               && std::memcmp(bOut.ptr, ORINT_PFX,
                                              static_cast<size_t>(orintLen))
                                      == 0) {
                        int32_t sep = -1;
                        for (int32_t p = orintLen; p + 1 < bOut.len; ++p) {
                            if (bOut.ptr[p] == '_'
                                && bOut.ptr[p + 1] == '(') { sep = p; break; }
                        }
                        if (sep > orintLen)
                            product = StrSpan(bOut.ptr + orintLen,
                                              sep - orintLen);
                    }
                    if (product.len > 0) {
                        const StatementFlags* row = lookupStatementFlags(
                            body.intKnownStatements, body.nameMap, product,
                            StrSpan("main", 4));
                        if (row != nullptr) return;  // closed — keep
                    }
                }
                body.pendingWipeScopes.mint(id);
                body.intValidityNamesToFilter.mint(id);
                rootNames.push_back(body.nameMap.decodeView(id));
            });

        // ---- 3. goal-template gates at MAIN ----
        const auto tmplMatches = [&](int32_t tid) -> bool {
            const StrSpan t = body.templateInterner.decodeView(tid);
            if (templateMatchesGoalWithMarkers(t, seed)) return true;
            StrSpan tg, tb;
            return gl::ScopeSyntax::splitSubproofPayload(t, tg, tb) && equalSpans(tg, seed);
        };
        body.integrationPrepared.eraseIf([&](int64_t k) {
            const StatementKey sk = Codec<StatementKey>::decode(k);
            return sk.validity == NameMap::MAIN_ID && tmplMatches(sk.orig);
        });
        body.integrationPreparedMarker.eraseIf([&](int64_t k) {
            const StatementKey sk = Codec<StatementKey>::decode(k);
            return sk.validity == NameMap::MAIN_ID && tmplMatches(sk.orig);
        });
        body.integrationStartIntMap.eraseIf([&](NameId tid) {
            return tmplMatches(tid);
        });

        // ---- 3b. the goal's parked implication subproofs ----
        // (D-406)
        implicationSubproofs.eraseGoalSubproofParks(body, seed);

        // Origin keys pack BOTH halves in the originInterner space — NEVER
        // the NameMap (I-91) — so subtree membership must gate on the DECODED
        // validity name against the wiped root names (the expandedImplications
        // text-gate idiom), not on the NameMap bitmap.
        const StrSpan boundaryTok(
            NameMap::BOUNDARY_STR,
            static_cast<int32_t>(NameMap::BOUNDARY_LEN));
        const auto underWipedRoot = [&](StrSpan vName) -> bool {
            for (int32_t ri = 0; ri < rootNames.size(); ++ri) {
                const StrSpan rn = rootNames[ri];
                if (equalSpans(vName, rn)) return true;
                if (vName.len > rn.len + boundaryTok.len
                    && equalSpans(StrSpan(vName.ptr, rn.len), rn)
                    && equalSpans(StrSpan(vName.ptr + rn.len,
                                          boundaryTok.len),
                                  boundaryTok))
                    return true;
            }
            return false;
        };

        // ---- 4. exprOriginMap: wiped-subtree rows + MAIN goal rows ----
        // (The maintainer-approved deviation from wipeSubtree's I-44
        // preserve rule — disproof cleanup only; the success path keeps its
        // history untouched.)
        // MAIN-keyed rows erase on GOAL-TEXT containment, with the products
        // excepted: the negated fact `gl::DefinitionExpansion::negate(goal)` and the anchor-rooted
        // theorem rows stay. Rows citing a dead subproof's bare compact but
        // NOT the goal stay too — the same compact text can belong to a
        // CLOSED subproof of another goal (per-goal duplication), whose main
        // statement and history are load-bearing for the chapter walker.
        body.exprOriginMap.eraseBlobIf([&](int64_t k) {
            ScratchScope originDecodeScope(sArena);
            StrSpan e, v;
            decodeOriginKeyCopy(k, body.originInterner, e, v, sArena);
            if (underWipedRoot(v)) return true;
            if (equalSpans(v, StrSpan("main", 4))
                && containsSpan(e, seed)
                && !equalSpans(e, negSeed)
                && !containsSpan(e, StrSpan("(Anchor", 7)))
                return true;
            return false;
        });

        // ---- 5. internal-mail origins at wiped scopes ----
        // (The statements are filtered by wipeSubtree itself right after this
        // drain; without the origins sweep the orphaned history runs would
        // outlive their statements in the discharge-surviving container.
        // Same originInterner key space as exprOriginMap — text gate.)
        body.sameIterationInternalMail.filterOrigins([&](int64_t k) {
            ScratchScope originDecodeScope(sArena);
            StrSpan e, v;
            decodeOriginKeyCopy(k, body.originInterner, e, v, sArena);
            return underWipedRoot(v);
        });

        // ---- 6. OR cohort bookkeeping under wiped scopes ----
        // Cohort identity is (parentValidity, orSignature) in the
        // lbStateInterner space (I-167) — the parent arrives as TEXT, so the
        // dead test is the equal-or-boundary-prefix gate against each wiped
        // root name (the wipeSubtree expandedImplications idiom).
        const auto cohortDead = [&](int32_t cohortId) -> bool {
            const LbStatePairKey pv =
                decodeOrCohortIds(body.lbStateInterner, cohortId);
            return underWipedRoot(body.lbStateInterner.decodeView(pv.high));
        };
        body.orDisjunctCount.eraseIf([&](int32_t cid) {
            return cohortDead(cid);
        });
        body.orBookkeeping.eraseSetIf([&](const LbStatePairKey& k) {
            return cohortDead(static_cast<int32_t>(k.low));
        });
        body.orPendingBranches.eraseSetIf([&](int32_t k) {
            return cohortDead(k);
        });
        body.orPendingLevels.eraseSetIf([&](int32_t k) {
            return cohortDead(k);
        });
        // Processed-or ledger rows at wiped scopes leave with the cohort
        // rows: a disproof-cleaned scope can be re-derived and must
        // re-process its ors from scratch (keys are NameMap vids —
        // decode, then the wiped-root text gate like the origin sweeps
        // above).
        body.processedOrLedger.eraseSetIf([&](NameId vid) {
            return underWipedRoot(body.nameMap.decodeView(vid));
        });
    }
    gArena.popTo(gMark);
}
}
