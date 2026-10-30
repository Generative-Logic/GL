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
#include "../compiler/definition_expansion.hpp"
#include "case_processor.hpp"
namespace gl {
/// @brief Bind inputs for case processor.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param admissionPolicy Borrowed AdmissionPolicy interface or state; must outlive this component.
/// @param anchorInfo Borrowed ce::AnchorInfo interface or state; must outlive this component.
/// @param canonicalTemplates Borrowed CanonicalTemplateMatcher interface or state; must outlive this component.
/// @param canonicalizer Borrowed Canonicalizer interface or state; must outlive this component.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param caseLedger Borrowed CaseLedger interface or state; must outlive this component.
/// @param equivalenceQueries Borrowed EquivalenceQueries interface or state; must outlive this component.
/// @param proofHistory Borrowed ProofHistory interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
CaseProcessor::CaseProcessor(AdmissionPolicy& admissionPolicy, const ce::AnchorInfo& anchorInfo, CanonicalTemplateMatcher& canonicalTemplates, Canonicalizer& canonicalizer, const CompiledDefinitionRegistry& definitions, const ProverParameters& parameters, CaseLedger& caseLedger, EquivalenceQueries& equivalenceQueries, ProofHistory& proofHistory) : admissionPolicy(admissionPolicy), anchorInfo(anchorInfo), canonicalTemplates(canonicalTemplates), canonicalizer(canonicalizer), definitions(definitions), parameters(parameters), caseLedger(caseLedger), equivalenceQueries(equivalenceQueries), proofHistory(proofHistory) {}

/// @brief Count statements beneath one exact parent/signature OR cohort or
///        OR-integration family.
///
/// @details Counts every statement whose scope is a branch of the cohort or a
/// case of the family, `<parent>_boundary_ordis_<signature>_(...)` or
/// `<parent>_boundary_orint_<signature>_(...)` by @p kind, or a scope nested
/// below one. Equal-looking signatures under other parents and goal-qualified
/// main families (`<goal>_subproof_orint_...`) never match the prefix. The
/// scan is read-only and uses no interner mint.
/// @param body Owning resident block.
/// @param parent Parent validity text.
/// @param signature Original OR signature.
/// @param kind `Disintegration` for an `_ordis_` cohort, `Integration` for an
///        `_orint_` family; `NotOrScope` asserts.
/// @return Statement count of the branches or cases and their nested scopes.
/// @see retireDuplicateOrCohorts; retireDuplicateOrIntegrationFamilies;
///      retireDuplicateOrSources.
int32_t CaseProcessor::orCohortStatementCount(const Memory& body,
    StrSpan parent, StrSpan signature, OrScopeKind kind) const {
    assert(kind != OrScopeKind::NotOrScope
           && "orCohortStatementCount: an OR scope kind is required");
    const StrSpan boundary = kind == OrScopeKind::Integration
        ? StrSpan("_boundary_orint_", 16) : StrSpan("_boundary_ordis_", 16);
    int32_t count = 0;
    for (int32_t i = 0; i < body.intEncodedStatements.size(); ++i) {
        const StrSpan scope = body.nameMap.decodeView(body.intEncodedStatements[i].validityId);
        const int32_t prefix = parent.len + boundary.len;
        if (scope.len <= prefix + signature.len + 1) continue;
        if (equalSpans(StrSpan(scope.ptr, parent.len), parent)
            && equalSpans(StrSpan(scope.ptr + parent.len, boundary.len), boundary)
            && equalSpans(StrSpan(scope.ptr + prefix, signature.len), signature)
            && scope.ptr[prefix + signature.len] == '_') ++count;
    }
    return count;
}

/// @brief End-of-burst drain of the dead OR-branch inbox — retire every
///        `_ordis_` branch whose asserted disjunct is refuted, shrink
///        its cohort's bookkeeping, and re-check convergence at the
///        reduced disjunct count.
///
/// @details
/// `ordisMerge`'s dead-branch probe stages a branch scope's NameMap vid
/// on `Memory::pendingDeadOrBranches` when the branch's asserted
/// disjunct has its negation `known` at the branch scope or one of its
/// ancestors — a refuted assumption makes every fact the branch
/// derives ex falso, so the branch only burns runtime and memory and
/// blocks its cohort's convergence forever. This drain — running
/// under the owning worker claim at standard-processing completion, after
/// `drainDisprovedGoals` (so cohorts retired wholesale by a goal
/// disproof are gone first) and immediately BEFORE the
/// `pendingWipeScopes` drain so the branch wipes land in the same
/// call — retires each staged branch:
///
/// 1. the branch scope is queued on `pendingWipeScopes` and inserted
///    into `intValidityNamesToFilter`, so the radical subtree wipe
///    removes its statements and no future deposit revives it (a
///    filtered scope also never re-stages — the `ordisMerge` probe and
///    `refutedOrBranchAtOrAbove` both skip it — and the drain asserts that
///    a staged branch is not one its cohort already retired, I-172);
/// 2. the cohort's `orDisjunctCount` drops by the number of its dead
///    branches; ZERO survivors retires the cohort wholesale — the
///    count row leaves the container and every bookkeeping run of the
///    cohort empties in step 3 (a fully-refuted disjunction implies a
///    joint contradiction at the disjunction's scope, which the
///    parent-scope K mutual-exclusion rules and the contradiction
///    machinery surface: primed reductio discharge; the
///    vacuous-premise flag plus the global-theorem reversion for a
///    normal LB at main; and for a cohort under a deeper
///    hypothetical scope of a normal LB, nothing at the LB level —
///    the refuted disjunction only proves that scope's assumption
///    unsatisfiable, the enclosing clause is vacuously true, and the
///    LB and its theorems stay valid);
/// 3. every `orBookkeeping` row of the cohort loses the dead branches'
///    disjunct entries (a run holding only dead entries leaves the
///    container) — required for soundness: a stale dead entry would
///    let the reduced count fire a convergence no surviving branch
///    ever derived;
/// 4. convergence is re-checked: a surviving row whose run already
///    reaches the reduced count re-fires through `ordisMerge` (the
///    canonical promotion path — parent-scope emission via
///    `sameIterationInternalMail`, `or convergence` history citing the
///    surviving branches, per-branch collapse of the converged copy).
///    A row whose survivor statement is gone from its branch scope is
///    a defined skip — that expression converged earlier and already
///    lives at the parent.
///
/// A staged branch whose cohort has no `orDisjunctCount` row is a
/// defined skip: the cohort was retired wholesale by this same seam's
/// `drainDisprovedGoals` (its parent scope lies in a disproof-wiped
/// subtree, which covers the branch scope too).
///
/// Branch-subtree `exprOriginMap` history is KEPT (I-44 preserve rule):
/// earlier full-cohort convergences cite dead-branch derivations as
/// `or convergence` ingredients, and the chapter walker must still
/// resolve them.
///
/// @param memoryBlock The LB whose inbox is drained (its containers
///                    are resident — the drain runs inside the LB's
///                    own burst).
/// @invariant Deterministic: staged vids process in decoded-name
///            byte-lex order; every erase predicate and the rebuilt
///            run contents are pure functions of container content.
/// @see Memory::pendingDeadOrBranches, ordisMerge, mintOrCohortId,
///      Memory::orBookkeeping, Memory::orDisjunctCount.
void CaseProcessor::drainDeadOrBranches(Memory& body) {
    const int32_t seedN = body.pendingDeadOrBranches.count();
    if (seedN == 0) return;
    // Quiescence (D-194): the retirement eradicates state; flag the mutation
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

    // Snapshot the staged vids, reset the inbox (a re-fired ordisMerge below
    // runs its probe against a fresh container), sort by decoded scope name
    // (the pendingWipeScopes drain idiom: the pod set dedups and the NameMap
    // is injective, so the sort is tie-free — I-84).
    NameId* vids = reinterpret_cast<NameId*>(gArena.resolve(
        gArena.alloc(seedN * static_cast<int32_t>(sizeof(NameId)),
                     alignof(NameId))));
    for (int32_t i = 1; i <= seedN; ++i)
        vids[i - 1] = body.pendingDeadOrBranches.decode(i);
    body.pendingDeadOrBranches.resetToFresh();
    std::sort(vids, vids + seedN, [&](NameId a, NameId b) {
        return compareSpans(body.nameMap.decodeView(a),
                            body.nameMap.decodeView(b)) < 0;
    });

    // ---- 1. per-seed: wipe queue + cohort/disjunct ledger ----
    DirtyState ledgerDirty = DirtyState::Clean;
    PagedVector<int32_t> deadCohortIds(&gArena, &ledgerDirty);   // per seed
    PagedVector<int32_t> deadDjIds(&gArena, &ledgerDirty);       // per seed
    for (int32_t si = 0; si < seedN; ++si) {
        const NameId vid = vids[si];
        StrSpan orSigV, branchBodyV;
        const OrScopeKind kind = gl::ScopeSyntax::classifyOrScopeView(body.nameMap, vid,
                                                     orSigV, branchBodyV);
        assert(kind == OrScopeKind::Disintegration
            && "pendingDeadOrBranches vid must be an _ordis_ branch scope");
        (void)kind;
        const StrSpan vName = body.nameMap.decodeView(vid);
        assert(!body.nameMap.stackEmpty(vid));
        const int32_t payloadLen = body.nameMap.decodeSubView(
            body.nameMap.stackBack(vid)).len;
        const StrSpan parentV(vName.ptr,
            vName.len - static_cast<int32_t>(NameMap::BOUNDARY_LEN)
                - payloadLen);
        const int32_t cohortId = mintOrCohortId(body.lbStateInterner,
            body.lbStateInterner.encode(parentV),
            body.lbStateInterner.encode(orSigV));
        if (body.orDisjunctCount.lookup(cohortId) == 0) continue;
        // Defined skip above: the cohort was retired wholesale by this same
        // seam's drainDisprovedGoals — its parent scope lies in a
        // disproof-wiped subtree, which covers this branch scope too.
        body.pendingWipeScopes.mint(vid);
        body.intValidityNamesToFilter.mint(vid);
        const int32_t deadDjId = body.lbStateInterner.encode(branchBodyV);
        // Each branch retires once (I-172): a staged branch is never one its
        // cohort already retired — a second retirement would drop the count
        // again and empty it while a branch still lives.
        {
            const int32_t retiredRow = body.orRetiredDisjuncts.lookup(cohortId);
            if (retiredRow != 0) {
                for (int32_t j = 0; j < body.orRetiredDisjuncts.runLen(retiredRow); ++j) {
                    assert(Codec<LbStatePairKey>::decode(
                               body.orRetiredDisjuncts.valueAt(retiredRow, j)).high != deadDjId
                        && "drainDeadOrBranches: an OR branch is retired at most once");
                }
            }
        }
        deadCohortIds.push_back(cohortId);
        deadDjIds.push_back(deadDjId);

        // Sequenced release: a retired live branch RESOLVES its cohort — if
        // branches are still pending, stage the cohort for the release drain
        // that runs right after this one (drainPendingOrReleases).
        if (body.orPendingBranches.lookup(cohortId) != 0)
            body.pendingOrReleases.mint(cohortId);

        // Record the retired disjunct with the SHALLOWEST scope where its
        // negation is `known` — the reductio ingredient a later
        // reduced-cohort `or convergence` row cites for this branch. The
        // walk re-runs the ordisMerge probe (the record is immortal, I-58,
        // so the staged branch is guaranteed a hit); preferring the
        // shallowest hit cites the fact where it genuinely lives (typically
        // main) rather than an inherited rewrite inside the dead branch.
        {
            const NameId negId = body.nameMap.lookup(
                negateOrCaseAssumption(branchBodyV, sArena));
            assert(negId != 0
                && "staged dead branch lost its refutation interning");
            NameId refutScope = 0;
            for (NameId a = vid; a != 0; a = body.nameMap.parentOf(a)) {
                const StatementFlags* row = body.intKnownStatements.find(
                    StatementKey{ negId, a });
                if (row != nullptr) refutScope = a;
                if (a == NameMap::MAIN_ID) break;
            }
            assert(refutScope != 0
                && "staged dead branch lost its refutation record");
            body.orRetiredDisjuncts.insertSorted(cohortId,
                packInt32Pair(deadDjId, refutScope));
            // Every later convergence of the cohort depends on this
            // refutation: its levels are saved under the retired case's own
            // scope, which outlives the case's wipe and leaves with the
            // cohort's parent (D-470).
            const int32_t refutLevels = body.intStatementLevelsMap.lookup(
                StatementKey{ negId, refutScope });
            assert(refutLevels != 0
                && "a known refutation carries derivation levels");
            int levels[256];
            const int32_t levelCount = coldIntRunAt(
                body.intStatementLevelsMap, refutLevels, levels, 256);
            body.lbMemory.closedOrGoalLevels.assignSet(
                StatementKey{ negId, vid }, levels, levelCount);
        }
    }
    const int32_t deadN = deadCohortIds.size();
    if (deadN == 0) { gArena.popTo(gMark); return; }

    const auto isDeadCohort = [&](int32_t cid) -> bool {
        for (int32_t i = 0; i < deadN; ++i)
            if (deadCohortIds[i] == cid) return true;
        return false;
    };
    const auto isDeadDj = [&](int32_t cid, int32_t dj) -> bool {
        for (int32_t i = 0; i < deadN; ++i)
            if (deadCohortIds[i] == cid && deadDjIds[i] == dj) return true;
        return false;
    };

    // ---- 2. shrink each cohort's structural count ----
    // Zero survivors retires the cohort wholesale (count row erased); the
    // joint contradiction the fully-refuted disjunction implies is owned by
    // the K rules + contradiction machinery, never by branch retirement.
    for (int32_t i = 0; i < deadN; ++i) {
        const int32_t cid = deadCohortIds[i];
        bool firstOfCohort = true;
        for (int32_t j = 0; j < i; ++j)
            if (deadCohortIds[j] == cid) { firstOfCohort = false; break; }
        if (!firstOfCohort) continue;
        int32_t cohortDead = 0;
        for (int32_t j = 0; j < deadN; ++j)
            if (deadCohortIds[j] == cid) ++cohortDead;
        const int32_t cntRow = body.orDisjunctCount.lookup(cid);
        assert(cntRow != 0);
        const int oldCount = body.orDisjunctCount.valueAt(cntRow);
        const int newCount = oldCount - cohortDead;
        assert(newCount >= 0
            && "dead-branch retirement decremented an _ordis_ cohort below zero");
        if (newCount == 0) {
            // Every disjunct refuted: the disjunction's scope holds a joint
            // contradiction. The cohort retires wholesale — the count row
            // leaves the container here, and every bookkeeping run of the
            // cohort empties in the surgery below (all entries are dead).
            // The contradiction itself surfaces through the pre-existing
            // machinery: the parent-scope K mutual-exclusion rules hold all
            // the refutations, fire the last disjunct as a fact, and the
            // resulting statement/negation pair drives the primed discharge
            // (a reductio LB) or the vacuous-premise flag + global-theorem
            // reversion (a normal LB at main).
            body.orDisjunctCount.erase(cid);
            // Wholesale retirement also drops the cohort's pending queue —
            // nothing may release into a fully-refuted cohort. (A sequenced
            // cohort reaches zero survivors only after every branch was
            // released, so this is normally a no-op; the guard covers the
            // disproof-drain overlap.)
            body.orPendingBranches.eraseSetIf(
                [&](int32_t k) { return k == cid; });
            body.orPendingLevels.eraseSetIf(
                [&](int32_t k) { return k == cid; });
        } else {
            body.orDisjunctCount.setValueAt(cntRow, newCount);
        }
    }

    // ---- 3. orBookkeeping surgery: drop dead disjunct entries ----
    // Snapshot every row of a dead cohort (key + surviving run), erase them
    // all, reinsert the rows that keep at least one survivor. Required for
    // soundness: a stale dead entry inflates the run length against the
    // reduced count, so a later test could fire a convergence no surviving
    // branch ever derived.
    PagedVector<LbStatePairKey> rowKeys(&gArena, &ledgerDirty);
    PagedVector<int32_t> rowStarts(&gArena, &ledgerDirty);
    PagedVector<int32_t> rowSurvivors(&gArena, &ledgerDirty);
    const int32_t obN = body.orBookkeeping.count();
    for (int32_t id = 1; id <= obN; ++id) {
        const LbStatePairKey key = Codec<LbStatePairKey>::decode(
            body.orBookkeeping.keyAt(id));
        const int32_t cid = static_cast<int32_t>(key.low);
        if (!isDeadCohort(cid)) continue;
        rowKeys.push_back(key);
        rowStarts.push_back(rowSurvivors.size());
        const int32_t rl = body.orBookkeeping.runLen(id);
        for (int32_t j = 0; j < rl; ++j) {
            const int32_t dj = body.orBookkeeping.valueAt(id, j);
            if (!isDeadDj(cid, dj)) rowSurvivors.push_back(dj);
        }
    }
    rowStarts.push_back(rowSurvivors.size());
    body.orBookkeeping.eraseSetIf([&](const LbStatePairKey& k) {
        return isDeadCohort(static_cast<int32_t>(k.low));
    });
    for (int32_t r = 0; r < rowKeys.size(); ++r) {
        for (int32_t j = rowStarts[r]; j < rowStarts[r + 1]; ++j) {
            body.orBookkeeping.insertSorted(rowKeys[r], rowSurvivors[j],
                DecodedIdLess{ &body.lbStateInterner });
        }
    }

    // ---- 4. convergence re-check at the reduced count ----
    // A surviving row whose run already reaches the new count re-fires
    // through ordisMerge — the canonical promotion path. The deposit spans
    // (expression + rebuilt branch validity) alias lbStateInterner cold bytes
    // and the string-scratch arena respectively, both stable across the
    // re-fire's mints. A successfully wiped branch retains only the derivation
    // levels of goals current at closure. Those levels may be used only while
    // the expression remains a goal of the innermost goal scope of this
    // cohort's parent chain. Other wiped expressions have no retained levels
    // and do not re-fire from this seam.
    const StrSpan BOUNDARY_ORDIS("_boundary_ordis_", 16);
    for (int32_t r = 0; r < rowKeys.size(); ++r) {
        const int32_t survN = rowStarts[r + 1] - rowStarts[r];
        if (survN == 0) continue;
        const int32_t cid = static_cast<int32_t>(rowKeys[r].low);
        const int* cnt = body.orDisjunctCount.find(cid);
        assert(cnt != nullptr);
        if (survN < *cnt) continue;
        const StrSpan exprV = body.lbStateInterner.decodeView(
            static_cast<int32_t>(rowKeys[r].high));
        const LbStatePairKey cohort =
            decodeOrCohortIds(body.lbStateInterner, cid);
        const StrSpan parentV = body.lbStateInterner.decodeView(cohort.high);
        const StrSpan sigV = body.lbStateInterner.decodeView(cohort.low);
        const StrSpan djV = body.lbStateInterner.decodeView(
            rowSurvivors[rowStarts[r]]);
        const int32_t bvLen =
            parentV.len + BOUNDARY_ORDIS.len + sigV.len + 1 + djV.len;
        char* bvBuf = sArena.allocBytes(bvLen);
        {
            int32_t at = 0;
            std::memcpy(bvBuf + at, parentV.ptr,
                static_cast<std::size_t>(parentV.len));
            at += parentV.len;
            std::memcpy(bvBuf + at, BOUNDARY_ORDIS.ptr,
                static_cast<std::size_t>(BOUNDARY_ORDIS.len));
            at += BOUNDARY_ORDIS.len;
            std::memcpy(bvBuf + at, sigV.ptr,
                static_cast<std::size_t>(sigV.len));
            at += sigV.len;
            bvBuf[at++] = '_';
            std::memcpy(bvBuf + at, djV.ptr,
                static_cast<std::size_t>(djV.len));
            at += djV.len;
            assert(at == bvLen);
        }
        const StrSpan branchV(bvBuf, bvLen);
        const int32_t lvlsId = lookupStatementLevels(
            body.intStatementLevelsMap, body.nameMap, exprV, branchV);
        const NameId exprId = body.nameMap.lookup(exprV);
        const NameId branchId = body.nameMap.lookup(branchV);
        const NameId parentId = body.nameMap.lookup(parentV);
        // Saved levels exist only for goals of the innermost goal scope, the
        // goals the freeze counted (D-549).
        const NameId goalScope = StatementKnowledge::innermostGoalScope(body, parentId);
        const bool isGoal = goalScope != 0
            && body.intToBeProved.lookup(packStatementKey(exprId, goalScope)) != 0;
        const int32_t savedLevelsId = lvlsId == 0 && isGoal
            ? body.lbMemory.closedOrGoalLevels.lookup(StatementKey{ exprId, branchId })
            : 0;
        if (lvlsId == 0 && savedLevelsId == 0) continue;
        int lvRun[256];
        const int32_t lvN = lvlsId != 0
            ? coldIntRunAt(body.intStatementLevelsMap, lvlsId, lvRun, 256)
            : coldIntRunAt(body.lbMemory.closedOrGoalLevels, savedLevelsId, lvRun, 256);
        ordisMerge(exprV, branchV, lvRun, lvN, body);
    }
    gArena.popTo(gMark);
}

/// @brief End-of-burst retirement of duplicate or cohorts — when a
///        class merge makes two OPEN cohorts at one parent equi
///        variants of each other, keep the most advanced and retire
///        the rest (the or-uniqueness gate's merge-seam companion).
///
/// @details
/// The deposit-door gate stops NEW duplicates; cohorts opened
/// BEFORE their signatures became class-equal both keep running.
/// This drain walks the `processedOrLedger` (a snapshot — the
/// clears below restructure it): per scope with two or more
/// recorded ors, each spelling is canonicalized under the scope's
/// complete class bucket (`canonicalFormAtScope`, the gate's own test)
/// and grouped; a group holding two or more OPEN cohorts (an
/// `orDisjunctCount` row via the non-minting `lbStateInterner`
/// recovery) is a duplicate case split. The KEEPER is the cohort
/// with the most statements already registered under its
/// `_ordis_` branch scopes (one pass over `intEncodedStatements`,
/// prefix match on `<parent>_boundary_ordis_<sig>_(`), ties to the
/// byte-lex-smaller signature — deterministic. Every loser is
/// retired: its LIVE branches (still in `orLiveBranches`) are
/// queued on `pendingWipeScopes` and leave the live registry (the
/// freeze sweep runs after this drain and never sees them), its
/// scheduling rows leave (`orDisjunctCount`, `orPendingBranches`,
/// `orPendingLevels`, `orStarterPick` — a stale `pendingOrReleases`
/// staging is the release drain's defined skip on the missing
/// count row), and its ledger row is cleared so a re-deposit of
/// that spelling suppresses against the keeper. FROZEN branches,
/// `orBookkeeping` runs, `orRetiredDisjuncts` and every history
/// line are KEPT — a frozen branch's derivations are load-bearing
/// (I-206), and with the count row gone the loser can never
/// converge. Retired work re-derives under the keeper when needed
/// (the maintainer-accepted trade-off). Parked-only duplicates
/// need no handling here: the ordis park hooks rewrite and re-mail
/// them, and the door's gate makes the re-file passive.
///
/// Runs in `standardProcessing` between `drainDeadOrBranches` and
/// `freezeResolvedOrBranches`; the queued wipes land in this same
/// burst's `pendingWipeScopes` drain. Heap-free (Rule 28): the
/// ledger snapshot and canonical texts ride the gen-scratch and
/// string tiers, reclaimed at exit. Mints nothing into any
/// interner (canonicalization, recovery probes and the wipe-queue
/// staging are all non-minting), so held `decodeView` spans stay
/// valid throughout (I-3).
///
/// @param body The LB whose ledger is swept (containers resident —
///             the drain runs inside the LB's own burst).
/// @return (void)
/// @invariant Scopes and group members process in ledger snapshot
///            order (deterministic single-threaded insert order);
///            the keeper is a pure function of container content.
/// @invariant Never retires a cohort that is the only open one in
///            its equi group.
/// @see orEquiRepresentativeRecorded — the deposit-side gate;
///      drainPendingOrReleases — the defined skips the retirement
///      relies on; Memory::processedOrLedger.
void CaseProcessor::retireDuplicateOrCohorts(Memory& body) {
    const int32_t scopeN = body.processedOrLedger.count();
    if (scopeN == 0) return;

    const unsigned gSlot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : genScratchArenas().slotCount() - 1;
    ScratchArena& gArena = genScratchArenas().forSlot(gSlot);
    const ArenaOffset gMark = gArena.cursor();
    const unsigned sSlot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : scratchArenas().slotCount() - 1;
    ScratchArena& sArena = scratchArenas().forSlot(sSlot);

    // Ledger snapshot (gen-scratch page tier): scopes with fewer than two
    // recorded ors can hold no duplicate and are skipped at the source.
    struct LedgerScopeRow { NameId vid; int32_t start; int32_t n; };
    DirtyState snapDirty = DirtyState::Clean;
    PagedVector<int64_t> entryPool(&gArena, &snapDirty);
    PagedVector<LedgerScopeRow> scopeRows(&gArena, &snapDirty);
    for (int32_t id = 1; id <= scopeN; ++id) {
        const int32_t n = body.processedOrLedger.runLen(id);
        if (n < 2) continue;
        const LedgerScopeRow row{ body.processedOrLedger.decodeKey(id),
                                  static_cast<int32_t>(entryPool.size()), n };
        for (int32_t j = 0; j < n; ++j)
            entryPool.push_back(body.processedOrLedger.valueAt(id, j));
        scopeRows.push_back(row);
    }

    for (int32_t si = 0; si < static_cast<int32_t>(scopeRows.size()); ++si) {
        const LedgerScopeRow& sr = scopeRows[static_cast<std::size_t>(si)];
        const StrSpan vName = body.nameMap.decodeView(sr.vid);
        // Non-minting cohort recovery: an uninterned parent means no
        // cohort ever opened at this scope — nothing to retire.
        const int32_t parentId = body.lbStateInterner.lookup(vName);
        if (parentId == 0) continue;

        // Per-scope working arrays on the gen-scratch byte tier, sized by
        // the actual recorded-or count — a scope's ledger run is unbounded
        // (an FTA shortcut main records hundreds), so no stack cap. The
        // nested clearProcessedOr allocations below are LIFO above this
        // mark; the per-scope popTo reclaims everything.
        ScratchScope scopeStrScope(sArena);
        const ArenaOffset scopeMark = gArena.cursor();
        const auto allocRun = [&](int32_t bytes, int32_t align) -> char* {
            return gArena.resolve(gArena.alloc(bytes, align));
        };
        StrSpan* text = reinterpret_cast<StrSpan*>(allocRun(
            sr.n * static_cast<int32_t>(sizeof(StrSpan)),
            static_cast<int32_t>(alignof(StrSpan))));
        NameId* canonId = reinterpret_cast<NameId*>(allocRun(
            sr.n * static_cast<int32_t>(sizeof(NameId)),
            static_cast<int32_t>(alignof(NameId))));
        NameId* entryId = reinterpret_cast<NameId*>(allocRun(
            sr.n * static_cast<int32_t>(sizeof(NameId)),
            static_cast<int32_t>(alignof(NameId))));
        int32_t* cohortIdOf = reinterpret_cast<int32_t*>(allocRun(
            sr.n * static_cast<int32_t>(sizeof(int32_t)),
            static_cast<int32_t>(alignof(int32_t))));
        int32_t* groupOf = reinterpret_cast<int32_t*>(allocRun(
            sr.n * static_cast<int32_t>(sizeof(int32_t)),
            static_cast<int32_t>(alignof(int32_t))));
        int32_t* score = reinterpret_cast<int32_t*>(allocRun(
            sr.n * static_cast<int32_t>(sizeof(int32_t)),
            static_cast<int32_t>(alignof(int32_t))));
        for (int32_t j = 0; j < sr.n; ++j) {
            const int64_t e = entryPool[static_cast<std::size_t>(sr.start + j)];
            // The ledger's canonical half is current (re-keyed at every
            // class change at this scope, I-219), so the grouping below is
            // an id comparison — no canonicalization at the drain.
            entryId[j] = gl::CaseLedger::ledgerOriginalId(e);
            canonId[j] = gl::CaseLedger::ledgerCanonicalId(e);
            text[j] = body.nameMap.decodeView(entryId[j]);
            cohortIdOf[j] = 0;
            const int32_t cid = lookupOrCohortId(body.lbStateInterner,
                parentId, body.lbStateInterner.lookup(text[j]));
            if (cid != 0 && body.orDisjunctCount.lookup(cid) != 0) {
                cohortIdOf[j] = cid;   // OPEN cohort
            }
            groupOf[j] = j;
            for (int32_t k = 0; k < j; ++k) {
                if (canonId[k] == canonId[j]) { groupOf[j] = groupOf[k]; break; }
            }
        }

        for (int32_t g = 0; g < sr.n; ++g) {
            if (groupOf[g] != g) continue;   // one drain per group leader
            int32_t openN = 0;
            for (int32_t j = 0; j < sr.n; ++j)
                if (groupOf[j] == g && cohortIdOf[j] != 0) ++openN;
            if (openN < 2) continue;         // no duplicated OPEN case split

            // Census: statements already registered under each open
            // member's `_ordis_` branch scopes (prefix
            // `<parent>_boundary_ordis_<sig>_(`, which every branch scope
            // and branch descendant extends).
            for (int32_t j = 0; j < sr.n; ++j)
                score[j] = groupOf[j] == g && cohortIdOf[j] != 0
                    ? orCohortStatementCount(body, vName, text[j],
                                             OrScopeKind::Disintegration) : 0;

            // Keeper: most branch statements; ties to the byte-lex
            // smaller signature (deterministic).
            int32_t keep = -1;
            for (int32_t j = 0; j < sr.n; ++j) {
                if (groupOf[j] != g || cohortIdOf[j] == 0) continue;
                if (keep < 0 || score[j] > score[keep]
                    || (score[j] == score[keep]
                        && compareSpans(text[j], text[keep]) < 0)) {
                    keep = j;
                }
            }
            assert(keep >= 0);

            for (int32_t j = 0; j < sr.n; ++j) {
                if (groupOf[j] != g || cohortIdOf[j] == 0 || j == keep)
                    continue;
                const int32_t loserCid = cohortIdOf[j];
                // Live branches of the loser leave the live registry NOW
                // (the freeze sweep runs after this drain) and their
                // subtrees ride this burst's wipe drain. Frozen branches,
                // orBookkeeping runs and every history line stay.
                const int32_t liveN = body.orLiveBranches.count();
                for (int32_t li = 1; li <= liveN; ++li) {
                    const NameId bvid = body.orLiveBranches.decode(li);
                    StrSpan orSigV, branchBodyV;
                    if (gl::ScopeSyntax::classifyOrScopeView(body.nameMap, bvid,
                                            orSigV, branchBodyV)
                            != OrScopeKind::Disintegration) continue;
                    if (body.nameMap.parentOf(bvid) != sr.vid) continue;
                    if (!equalSpans(orSigV, text[j])) continue;
                    body.pendingWipeScopes.mint(bvid);
                }
                body.orLiveBranches.eraseIf([&](const NameId& v) {
                    return body.pendingWipeScopes.lookup(v) != 0;
                });
                // Scheduling rows leave; a stale pendingOrReleases
                // staging is the release drain's defined skip on the
                // missing count row.
                body.orDisjunctCount.eraseIf(
                    [loserCid](int32_t cid) { return cid == loserCid; });
                body.orPendingBranches.eraseSet(loserCid);
                body.orPendingLevels.eraseSet(loserCid);
                body.orStarterPick.erase(loserCid);
                caseLedger.clearProcessedOr(body, entryId[j], sr.vid);
                body.mutatedThisBurst = true;
            }
        }
        gArena.popTo(scopeMark);
    }
    gArena.popTo(gMark);
}

/// @brief End-of-burst retirement of duplicate OR-integration families —
///        of two open families under one parent whose ORs became
///        class-equal after both opened, the one with the most statements
///        survives (D-550).
///
/// @details
/// To integrate an OR at a parent scope, Case OR of
/// `prepareIntegrationCore2` opens one case scope per disjunct,
/// `<parent>_boundary_orint_<signature>_(<disjunct>)`, which assumes the
/// other disjuncts false and holds its own disjunct as goal. The cases of
/// one parent and one signature form a family; the first case that reaches
/// its goal emits the OR at the parent and `cleanUpOrIntegrationBranches`
/// wipes the family. The opening's prepared-template gate is class-aware,
/// so two families of class-equal ORs coexist only when the equality
/// arrived after both opened; both would then search to the end.
///
/// Candidates are the case scopes that hold a goal row in `intToBeProved`
/// (an open case always does: its goal leaves only with the family), whose
/// payload is bare (`orint_` first; a goal-qualified main family
/// `<goal>_subproof_orint_...` follows its goal's lifecycle, as D-414 keeps
/// goal-driven subproofs), and which are neither wiped nor queued, itself
/// or through an ancestor. Families of one parent whose signatures have the
/// same canonical form there (`canonicalFormAtScope`, text only) form one
/// group. The KEEPER has the most statements in its cases and every scope
/// nested below them (`orCohortStatementCount`, `Integration`); ties go to
/// the byte-lex smaller signature. Every case scope of every other member
/// is queued on `pendingWipeScopes`, so this burst's wipe drain removes it
/// with its nested scopes. `cleanUpOrIntegrationBranches` is not used: on
/// main it would also take goal-qualified families of the same signature.
///
/// The parent's prepared key stays, so a retired family never reopens, and
/// the wipe adds its scopes to `intValidityNamesToFilter`, so it is never
/// queued again. Proof history is untouched (I-44). Whatever the retired
/// family had derived re-derives under the keeper, whose OR every
/// class-equal consumer matches through the classes (I-254).
///
/// Runs in `standardProcessing` immediately after
/// `retireSettledImplicationSubproofs`, before the wipe drain. Heap-free
/// (Rule 28): the case and family runs ride the gen-scratch page tier and
/// the canonical texts the string tier, reclaimed at exit. Mints nothing
/// into any interner, so the held `decodeView` signature spans stay valid
/// (I-3).
///
/// @param body The LB whose open OR-integration families are swept
///             (containers resident — the sweep runs inside the LB's own
///             burst).
/// @return (void)
/// @invariant The retired set is a pure function of container content
///            (I-84): grouping and keeper choice sort by parent id,
///            canonical text and signature bytes.
/// @invariant Never retires the only open family of its group.
/// @see orCohortStatementCount; cleanUpOrIntegrationBranches;
///      retireDuplicateOrCohorts — the `_ordis_` twin;
///      ImplicationSubproofs::retireSettledImplicationSubproofs — the
///      implication-subproof twin.
void CaseProcessor::retireDuplicateOrIntegrationFamilies(Memory& body) {
    const int32_t goalN = body.intToBeProved.count();
    if (goalN < 2) return;

    const unsigned gSlot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : genScratchArenas().slotCount() - 1;
    ScratchArena& gArena = genScratchArenas().forSlot(gSlot);
    const ArenaOffset gMark = gArena.cursor();
    const unsigned sSlot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : scratchArenas().slotCount() - 1;
    ScratchArena& sArena = scratchArenas().forSlot(sSlot);
    ScratchScope sweepStrings(sArena);

    // A scope is open unless it or an ancestor is wiped or already queued.
    const auto closedOrQueued = [&](NameId scope) {
        for (NameId s = scope; s != 0; s = body.nameMap.parentOf(s))
            if (body.intValidityNamesToFilter.lookup(s) != 0
                || body.pendingWipeScopes.lookup(s) != 0) return true;
        return false;
    };

    // Open bare OR-integration case scopes, one row per goal row. The
    // signature span aliases the NameMap sub-table; nothing below mints.
    struct CaseRow { NameId scope; NameId parent; StrSpan signature; };
    DirtyState dirty = DirtyState::Clean;
    PagedVector<CaseRow> cases(&gArena, &dirty);
    for (int32_t g = 1; g <= goalN; ++g) {
        const NameId scope = body.intToBeProved.decodeKey(g).validity;
        StrSpan signature, wrapped;
        if (ScopeSyntax::classifyOrScopeView(body.nameMap, scope, signature, wrapped)
                != OrScopeKind::Integration) continue;
        if (!startsWithSpan(body.nameMap.decodeSubView(body.nameMap.stackBack(scope)),
                            "orint_", 6)) continue;
        if (closedOrQueued(scope)) continue;
        cases.push_back(CaseRow{ scope, body.nameMap.parentOf(scope), signature });
    }
    const int32_t caseN = cases.size();
    if (caseN < 2) { gArena.popTo(gMark); return; }

    const auto allocInts = [&](int32_t n) {
        return reinterpret_cast<int32_t*>(gArena.resolve(gArena.alloc(
            n * static_cast<int32_t>(sizeof(int32_t)), static_cast<int32_t>(alignof(int32_t)))));
    };
    // Case rows sorted by (parent, signature, scope); equal scopes are the
    // several goal rows of one case and collapse to one.
    int32_t* caseOrder = allocInts(caseN);
    for (int32_t i = 0; i < caseN; ++i) caseOrder[i] = i;
    std::sort(caseOrder, caseOrder + caseN, [&](int32_t x, int32_t y) {
        const CaseRow& l = cases[x];
        const CaseRow& r = cases[y];
        if (l.parent != r.parent) return l.parent < r.parent;
        const int c = compareSpans(l.signature, r.signature);
        if (c != 0) return c < 0;
        return l.scope < r.scope;
    });

    // Families: runs of equal (parent, signature) in the case order.
    struct Family { NameId parent; StrSpan signature; StrSpan canonical; int32_t first; int32_t end; };
    PagedVector<Family> families(&gArena, &dirty);
    for (int32_t start = 0; start < caseN;) {
        int32_t end = start + 1;
        while (end < caseN && cases[caseOrder[end]].parent == cases[caseOrder[start]].parent
               && equalSpans(cases[caseOrder[end]].signature, cases[caseOrder[start]].signature))
            ++end;
        const CaseRow& lead = cases[caseOrder[start]];
        families.push_back(Family{ lead.parent, lead.signature, StrSpan(), start, end });
        start = end;
    }
    const int32_t familyN = families.size();
    bool sharedParent = false;
    for (int32_t f = 1; f < familyN && !sharedParent; ++f)
        sharedParent = families[f].parent == families[f - 1].parent;
    if (!sharedParent) { gArena.popTo(gMark); return; }

    for (int32_t f = 0; f < familyN; ++f) {
        Family family = families[f];
        int32_t noLevels = 0;
        family.canonical = canonicalizer.canonicalFormAtScope(family.signature,
            body.nameMap.decodeView(family.parent), body, sArena, nullptr, noLevels, 0,
            /*rewriteEqualities=*/false, /*collectEvidence=*/false).text;
        families.setAt(f, family);
    }
    // Families sorted by (parent, canonical, signature): a group is a run of
    // equal (parent, canonical), its first member the byte-lex smallest.
    int32_t* familyOrder = allocInts(familyN);
    for (int32_t f = 0; f < familyN; ++f) familyOrder[f] = f;
    std::sort(familyOrder, familyOrder + familyN, [&](int32_t x, int32_t y) {
        const Family& l = families[x];
        const Family& r = families[y];
        if (l.parent != r.parent) return l.parent < r.parent;
        const int c = compareSpans(l.canonical, r.canonical);
        if (c != 0) return c < 0;
        return compareSpans(l.signature, r.signature) < 0;
    });
    for (int32_t start = 0; start < familyN;) {
        int32_t end = start + 1;
        while (end < familyN && families[familyOrder[end]].parent == families[familyOrder[start]].parent
               && equalSpans(families[familyOrder[end]].canonical, families[familyOrder[start]].canonical))
            ++end;
        if (end - start >= 2) {
            const StrSpan parentName = body.nameMap.decodeView(families[familyOrder[start]].parent);
            int32_t keep = start;
            int32_t keepScore = orCohortStatementCount(body, parentName,
                families[familyOrder[start]].signature, OrScopeKind::Integration);
            for (int32_t k = start + 1; k < end; ++k) {
                const int32_t score = orCohortStatementCount(body, parentName,
                    families[familyOrder[k]].signature, OrScopeKind::Integration);
                if (score > keepScore) { keep = k; keepScore = score; }
            }
            for (int32_t k = start; k < end; ++k) {
                if (k == keep) continue;
                RT_SCOPE_HERE("RETIRE_DUPLICATE_OR_INTEGRATION_FAMILY_HIT");
                const Family& loser = families[familyOrder[k]];
                // A case holding several goal rows repeats; mint is find-or-mint.
                for (int32_t c = loser.first; c < loser.end; ++c)
                    body.pendingWipeScopes.mint(cases[caseOrder[c]].scope);
            }
        }
        start = end;
    }
    gArena.popTo(gMark);
}

/// @brief Retire every open OR cohort one of whose still-open
///        disjuncts has come to hold at the cohort's parent scope
///        (D-438, the retire-later half).
///
/// @details
/// The deposit-side gate in `consumeOrLeavesCohort` opens no case
/// split for an OR whose disjunct already holds; this sweep covers a
/// disjunct that comes to hold after its cohort opened. The
/// still-open disjuncts of a cohort are the disjunct of its live
/// branch (at most one per cohort, I-174) and its queued disjuncts
/// (`orPendingBranches`); each is tested with `orDisjunctTrueAt` at
/// the cohort's parent scope. The branch of a disjunct that holds is
/// the parent scope itself, so the cohort can no longer conclude
/// anything the parent does not prove by itself.
///
/// A satisfied cohort is retired exactly like a duplicate loser in
/// `retireDuplicateOrCohorts`: its live branch is queued on
/// `pendingWipeScopes` and leaves `orLiveBranches`, and its
/// scheduling rows leave (`orDisjunctCount`, `orPendingBranches`,
/// `orPendingLevels`, `orStarterPick`; a stale `pendingOrReleases`
/// staging is the release drain's defined skip on the missing count
/// row). Frozen branches, `orBookkeeping` runs, `orRetiredDisjuncts`
/// and every history line are kept. The processed-or ledger row
/// stays, so a re-deposit of the OR at that scope stays suppressed.
///
/// Runs in `standardProcessing` after `retireDuplicateOrCohorts` and
/// before `freezeResolvedOrBranches` and `drainPendingOrReleases`, so
/// a retired cohort is neither evaluated nor released and its wipes
/// land in this same call's `pendingWipeScopes` drain. Heap-free
/// (Rule 28): the satisfied-cohort set rides the gen-scratch page
/// tier, reclaimed at exit. Mints nothing into the NameMap or the
/// `lbStateInterner`, so held `decodeView` spans stay valid (I-3).
///
/// @param body The LB whose open cohorts are swept (containers
///             resident — the sweep runs inside the LB's own burst).
/// @return (void)
/// @invariant Only cohorts with a still-open disjunct that holds at
///            the parent scope are retired; the decision is a pure
///            function of container content (I-84).
/// @see orDisjunctTrueAt, consumeOrLeavesCohort,
///      retireDuplicateOrCohorts, drainPendingOrReleases.
void CaseProcessor::retireSatisfiedOrCohorts(Memory& body) {
    const int32_t liveN = body.orLiveBranches.count();
    const int32_t pendingRowN = body.orPendingBranches.count();
    if (liveN == 0 && pendingRowN == 0) return;

    const unsigned gSlot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : genScratchArenas().slotCount() - 1;
    ScratchArena& gArena = genScratchArenas().forSlot(gSlot);
    const ArenaOffset gMark = gArena.cursor();
    DirtyState dirty = DirtyState::Clean;
    ColdHashSet<PodKeyStore<int32_t>> satisfied(&gArena, &dirty);
    satisfied.resetToFresh();

    // The cohort of one live branch; the branch's own disjunct comes back
    // through the out-span (a wrapped payload body, unwrapped here).
    const auto liveCohortOf = [&](NameId vid, StrSpan& disjunctOut) -> int32_t {
        StrSpan orSigV, branchBodyV;
        const OrScopeKind kind = gl::ScopeSyntax::classifyOrScopeView(body.nameMap, vid,
                                                     orSigV, branchBodyV);
        assert(kind == OrScopeKind::Disintegration
            && "orLiveBranches vid must be an _ordis_ branch scope");
        (void)kind;
        assert(branchBodyV.len >= 2 && branchBodyV.ptr[0] == '('
            && branchBodyV.ptr[branchBodyV.len - 1] == ')'
            && "ordis branch payload must be a wrapped (disjunct)");
        disjunctOut = StrSpan(branchBodyV.ptr + 1, branchBodyV.len - 2);
        const NameId parentId = body.nameMap.parentOf(vid);
        assert(parentId != 0 && "an _ordis_ branch scope has a cohort parent");
        return lookupOrCohortId(body.lbStateInterner,
            body.lbStateInterner.lookup(body.nameMap.decodeView(parentId)),
            body.lbStateInterner.lookup(orSigV));
    };
    // Whether a queued disjunct of the cohort holds at its parent scope.
    const auto pendingHolds = [&](int32_t cohortId, NameId parentId) -> bool {
        const int32_t pRow = body.orPendingBranches.lookup(cohortId);
        if (pRow == 0) return false;
        const int32_t n = body.orPendingBranches.runLen(pRow);
        for (int32_t j = 0; j < n; ++j) {
            const StrSpan w = body.lbStateInterner.decodeView(
                body.orPendingBranches.valueAt(pRow, j));
            assert(w.len >= 2 && w.ptr[0] == '(' && w.ptr[w.len - 1] == ')'
                && "pending ordis disjunct must be a wrapped (disjunct)");
            if (orDisjunctTrueAt(StrSpan(w.ptr + 1, w.len - 2), parentId, body))
                return true;
        }
        return false;
    };

    // ---- 1. live branches: the live disjunct and the queued ones ----
    for (int32_t li = 1; li <= liveN; ++li) {
        const NameId vid = body.orLiveBranches.decode(li);
        // A retired or wipe-closed branch is the freeze sweep's defined drop.
        if (body.intValidityNamesToFilter.lookup(vid) != 0) continue;
        StrSpan liveDisjunct;
        const int32_t cohortId = liveCohortOf(vid, liveDisjunct);
        assert(cohortId != 0 && "a live ordis branch has an interned cohort");
        // Defined skip: the cohort was retired wholesale by this seam's
        // drainDisprovedGoals; its parent lies in a disproof-wiped subtree,
        // which covers this branch too.
        if (body.orDisjunctCount.lookup(cohortId) == 0) continue;
        if (satisfied.lookup(cohortId) != 0) continue;
        const NameId parentId = body.nameMap.parentOf(vid);
        if (orDisjunctTrueAt(liveDisjunct, parentId, body)
            || pendingHolds(cohortId, parentId)) {
            satisfied.mint(cohortId);
        }
    }

    // ---- 2. queued cohorts without a live branch ----
    for (int32_t id = 1; id <= pendingRowN; ++id) {
        const int32_t cohortId = body.orPendingBranches.decodeKey(id);
        if (satisfied.lookup(cohortId) != 0) continue;
        // Defined skip: a cohort retired wholesale (count row gone).
        if (body.orDisjunctCount.lookup(cohortId) == 0) continue;
        const LbStatePairKey cohort =
            decodeOrCohortIds(body.lbStateInterner, cohortId);
        const NameId parentId = body.nameMap.lookup(
            body.lbStateInterner.decodeView(static_cast<int32_t>(cohort.high)));
        assert(parentId != 0 && "ordis cohort parent scope must be interned");
        // Defined skip: a filtered parent belongs to a retired region.
        if (body.intValidityNamesToFilter.lookup(parentId) != 0) continue;
        if (pendingHolds(cohortId, parentId)) satisfied.mint(cohortId);
    }

    // ---- 3. retire (the retireDuplicateOrCohorts loser steps) ----
    if (satisfied.count() > 0) {
        for (int32_t li = 1; li <= liveN; ++li) {
            const NameId vid = body.orLiveBranches.decode(li);
            if (body.intValidityNamesToFilter.lookup(vid) != 0) continue;
            StrSpan liveDisjunct;
            if (satisfied.lookup(liveCohortOf(vid, liveDisjunct)) != 0)
                body.pendingWipeScopes.mint(vid);
        }
        body.orLiveBranches.eraseIf([&](const NameId& v) {
            return body.pendingWipeScopes.lookup(v) != 0;
        });
        body.orDisjunctCount.eraseIf([&](int32_t cid) {
            return satisfied.lookup(cid) != 0;
        });
        body.orPendingBranches.eraseSetIf([&](int32_t cid) {
            return satisfied.lookup(cid) != 0;
        });
        body.orPendingLevels.eraseSetIf([&](int32_t cid) {
            return satisfied.lookup(cid) != 0;
        });
        for (int32_t k = 1; k <= satisfied.count(); ++k)
            body.orStarterPick.erase(satisfied.decode(k));
        body.mutatedThisBurst = true;
    }
    satisfied.release();
    gArena.popTo(gMark);
}

/// @brief Retire every implication subproof one of whose live OR cases
///        holds the negation of a goal of the subproof
///        (D-549).
///
/// @details
/// A live case serves the goals of its innermost goal scope
/// (`StatementKnowledge::innermostGoalScope`). When that scope is a
/// numbered implication subproof (its payload, bare or behind a
/// `<goal>_subproof_` prefix, is `(implication...)`) and the exact
/// negation of one of its goals is known at the case or above, the case
/// can reach that goal only by becoming contradictory: its cohort
/// cannot converge, so the subproof cannot close. The subproof scope is
/// queued on `pendingWipeScopes`, as `retireEqualInputImplicationSubproofs`
/// does, and its live cases leave `orLiveBranches`, as
/// `retireDuplicateOrCohorts` drops a loser's live branch, so the freeze
/// sweep never evaluates them. The prepared key survives, so the subproof
/// does not reopen. Nothing is proved or refuted: the wipe abandons search
/// only, and history is kept. Main's goals, integration subproofs and
/// OR-integration scopes are never retired here. A block primed for
/// contradiction is skipped: there the parent is meant to be inconsistent.
///
/// Runs in `standardProcessing` after `retireSatisfiedOrCohorts` and
/// before `freezeResolvedOrBranches`; the wipes land in this same call's
/// `pendingWipeScopes` drain. Heap-free (Rule 28): the retired-scope run
/// rides the gen-scratch page tier and the negated goal texts the string
/// tier, both reclaimed at exit. Mints nothing.
///
/// @param body The LB whose live cases are swept (containers resident —
///             the sweep runs inside the LB's own burst).
/// @return (void)
/// @invariant The decision is a pure function of container content
///            (I-84); a scope already closed or queued is skipped.
/// @see StatementKnowledge::innermostGoalScope,
///      ImplicationSubproofs::retireEqualInputImplicationSubproofs,
///      retireDuplicateOrCohorts, freezeResolvedOrBranches.
void CaseProcessor::retireRefutedGoalSubproofs(Memory& body) {
    if (body.primedForContradiction) return;
    const int32_t liveN = body.orLiveBranches.count();
    if (liveN == 0 || body.intToBeProved.empty()) return;

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

    // A scope is open unless it or an ancestor is wiped or already queued.
    const auto closedOrQueued = [&](NameId scope) {
        for (NameId s = scope; s != 0; s = body.nameMap.parentOf(s))
            if (body.intValidityNamesToFilter.lookup(s) != 0
                || body.pendingWipeScopes.lookup(s) != 0) return true;
        return false;
    };

    DirtyState dirty = DirtyState::Clean;
    PagedVector<NameId> retired(&gArena, &dirty);
    for (int32_t li = 1; li <= liveN; ++li) {
        const NameId vid = body.orLiveBranches.decode(li);
        if (body.frozenOrBranches.lookup(vid) != 0
            || body.pendingDeadOrBranches.lookup(vid) != 0
            || closedOrQueued(vid)) continue;
        const NameId goalScope = StatementKnowledge::innermostGoalScope(body, vid);
        if (goalScope == 0 || body.nameMap.stackEmpty(goalScope)) continue;
        const StrSpan bare = ScopeSyntax::stripSubproofPrefixView(
            body.nameMap.decodeSubView(body.nameMap.stackBack(goalScope)));
        if (!startsWithSpan(bare, "(implication", 12)) continue;
        bool refuted = false;
        for (int32_t g = 1; g <= body.intToBeProved.count() && !refuted; ++g) {
            const StatementKey goal = body.intToBeProved.decodeKey(g);
            if (goal.validity != goalScope) continue;
            ScratchScope negationScope(sArena);
            const StrSpan negation = gl::DefinitionExpansion::negateScratch(sArena,
                body.nameMap.decodeView(goal.orig));
            const NameId negationId = body.nameMap.lookup(negation);
            refuted = negationId != 0
                && StatementKnowledge::ancestorKnown(body, negationId, vid, /*includeSelf=*/true);
        }
        if (!refuted) continue;
        RT_SCOPE_HERE("RETIRE_REFUTED_GOAL_SUBPROOF_HIT");
        body.pendingWipeScopes.mint(goalScope);
        retired.push_back(goalScope);
    }
    if (!retired.empty()) {
        body.orLiveBranches.eraseIf([&](const NameId& v) {
            for (int32_t r = 0; r < retired.size(); ++r)
                if (body.nameMap.isStrictAncestor(retired[r], v)) return true;
            return false;
        });
    }
    gArena.popTo(gMark);
}

/// @brief Test whether this block has work that can require OR cases.
///
/// @details
/// Explicit goals retain their usual eligibility. An active primed
/// contradiction block also has proof work even without explicit goals;
/// its obligation is discharged by a registered main-scope contradiction.
/// A conjecture's goal block has proof work while its last premise is
/// seeded, before its goal is registered (`Memory::goalPending`,
/// D-495).
///
/// @param body The resident logical block whose proof state is inspected.
/// @return True for explicit goals, a pending conjecture goal, or an active
///         primed contradiction block.
/// @invariant Does not allocate or change proof, history or scheduling state.
/// @see consumeOrLeavesCohort, drainPendingOrReleases.
bool CaseProcessor::hasOrProofWork(const Memory& body) const {
    return !body.intToBeProved.empty()
        || body.goalPending
        || (body.isActive && body.primedForContradiction);
}

/// @brief Test whether an OR branch covers every goal of its innermost goal
///        scope.
///
/// @details
/// Relevant goals are those of the deepest goal-holding scope on the branch's
/// chain (`StatementKnowledge::innermostGoalScope`, D-549);
/// goals further up are shadowed, and goals in sibling and descendant scopes
/// do not participate. Each relevant goal must be known at the branch or
/// above, or supported up to equality at the branch. A chain without any goal
/// remains unresolved in an active primed contradiction block, whose main owns
/// the contradiction as its implicit goal; otherwise it qualifies.
///
/// @param body The resident logical block whose registries are read.
/// @param branchId A minted OR-disintegration branch root.
/// @return True when no relevant goal remains unknown.
/// @invariant The predicate neither mints names nor reads proof history.
/// @see ancestorKnown, innermostGoalScope, collectPhase3IgnoredOrRoots,
///      freezeResolvedOrBranches.
bool CaseProcessor::orBranchGoalsResolved(const Memory& body,
                                              NameId branchId) {
    StrSpan signature, disjunct;
    assert(gl::ScopeSyntax::classifyOrScopeView(body.nameMap, branchId, signature, disjunct)
            == OrScopeKind::Disintegration
        && "branch resolution requires an OR-disintegration root");
    const NameId goalScope = gl::StatementKnowledge::innermostGoalScope(body, branchId);
    if (goalScope == 0) return !body.primedForContradiction || !body.isActive;
    for (int32_t goalIndex = 1; goalIndex <= body.intToBeProved.count();
         ++goalIndex) {
        const StatementKey goal = body.intToBeProved.decodeKey(goalIndex);
        if (goal.validity != goalScope) continue;
        if (!gl::StatementKnowledge::ancestorKnown(body, goal.orig, branchId, /*includeSelf=*/true)
            && body.lbMemory.orSupportFacts.lookup(StatementKey{ goal.orig, branchId }) == 0)
            return false;
    }
    return true;
}

/// @brief Snapshot the OR roots eligible for Phase 3 inference exclusion.
///
/// @details
/// Walk existing scope identifiers, including roots whose seeds are still
/// queued. The result is fixed before any Phase 3 inference runs; descendants
/// are covered by the later ancestry test without extending this root set.
///
/// @param body The resident logical block, read without mutation.
/// @param roots An initially empty arena-backed output set.
/// @return Nothing.
/// @invariant No goal, branch marker, name or wipe queue is changed.
/// @see orBranchGoalsResolved, Memory::phase3IgnoresScope.
void CaseProcessor::collectPhase3IgnoredOrRoots(const Memory& body,
    ColdHashSet<PodKeyStore<NameId>>& roots) {
    assert(roots.empty() && "Phase 3 root snapshot must start empty");
    const int32_t nameCount = body.nameMap.nameCount();
    for (NameId scope = 1; scope <= nameCount; ++scope) {
        if (body.nameMap.parentOf(scope) == 0) continue;
        if (body.intValidityNamesToFilter.lookup(scope) != 0) continue;
        StrSpan signature, disjunct;
        if (gl::ScopeSyntax::classifyOrScopeView(body.nameMap, scope, signature, disjunct)
                != OrScopeKind::Disintegration) continue;
        if (orBranchGoalsResolved(body, scope)) roots.mint(scope);
    }
}

/// @brief Queue resolved OR-disintegration branches for subtree wipeout.
///
/// @details
/// Evaluate every live branch against the goals of its innermost goal scope,
/// the deepest goal-holding scope on its chain (D-549). A
/// goal must be known at the branch or above; shadowed and unrelated goals are
/// ignored and a chain without goals qualifies outside an active primed
/// contradiction block. The resolved marker and pending wipe queue are updated
/// after the read-only predicate completes.
///
/// The ordered end-of-burst wipe removes runtime state throughout each queued
/// subtree, preserving convergence bookkeeping, cohort counts and history.
/// Only booked current-goal derivation levels are saved separately for later
/// reduced-count convergence. Parent-chain goals close through the ordinary
/// convergence deposit path.
///
/// @param body The resident logical block whose live branches are evaluated.
/// @return Nothing.
/// @invariant The predicate reads statement membership, never origin history.
/// @invariant No runtime registry is erased while the goal snapshot is scanned.
/// @see Memory::wipeSubtree, Memory::pendingWipeScopes, ancestorKnown.
void CaseProcessor::freezeResolvedOrBranches(Memory& body) {
    const int32_t liveN = body.orLiveBranches.count();
    if (liveN == 0) return;
    const int32_t goalN = body.intToBeProved.count();
    for (int32_t li = 1; li <= liveN; ++li) {
        const NameId vid = body.orLiveBranches.decode(li);
        // A retired or wipe-closed branch never receives a deposit again:
        // dropped by the compaction below, never evaluated.
        if (body.intValidityNamesToFilter.lookup(vid) != 0) continue;
        assert(body.frozenOrBranches.lookup(vid) == 0
            && "a frozen branch must not be live");
        StrSpan orSigV, branchBodyV;
        const OrScopeKind kind = gl::ScopeSyntax::classifyOrScopeView(body.nameMap, vid,
                                                     orSigV, branchBodyV);
        assert(kind == OrScopeKind::Disintegration
            && "orLiveBranches vid must be an _ordis_ branch scope");
        (void)kind;
        const NameId chainStart = body.nameMap.parentOf(vid);
        assert(chainStart != 0
            && "an _ordis_ branch scope has a cohort parent");
        if (orBranchGoalsResolved(body, vid)) {
            // Retain only booked current-goal derivation levels of the
            // innermost goal scope, the goals resolution counted.
            // Ancestor-known goals with no contribution booked at this branch
            // need no record.
            const int32_t parentStateId = body.lbStateInterner.lookup(
                body.nameMap.decodeView(chainStart));
            const int32_t signatureId = body.lbStateInterner.lookup(orSigV);
            const int32_t disjunctId = body.lbStateInterner.lookup(branchBodyV);
            const int32_t cohortId = lookupOrCohortId(
                body.lbStateInterner, parentStateId, signatureId);
            const NameId goalScope = gl::StatementKnowledge::innermostGoalScope(body, vid);
            for (int32_t gi = 1; gi <= goalN; ++gi) {
                const StatementKey goal = body.intToBeProved.decodeKey(gi);
                if (goal.validity != goalScope) continue;
                const int32_t expressionId = body.lbStateInterner.lookup(
                    body.nameMap.decodeView(goal.orig));
                const int32_t booked = body.orBookkeeping.lookup(
                    LbStatePairKey{ expressionId, cohortId });
                if (booked == 0) continue;
                bool contributed = false;
                for (int32_t j = 0; j < body.orBookkeeping.runLen(booked); ++j) {
                    if (body.orBookkeeping.valueAt(booked, j) == disjunctId)
                        contributed = true;
                }
                if (!contributed) continue;
                const StatementKey key{ goal.orig, vid };
                if (body.lbMemory.orSupportFacts.lookup(key) != 0) {
                    assert(body.lbMemory.closedOrGoalLevels.lookup(key) != 0);
                    continue;
                }
                const int32_t levelRow = body.intStatementLevelsMap.lookup(key);
                assert(levelRow != 0
                    && "booked current goal must retain derivation levels until its branch wipe");
                int levels[256];
                const int32_t count = coldIntRunAt(
                    body.intStatementLevelsMap, levelRow, levels, 256);
                body.lbMemory.closedOrGoalLevels.assignSet(key, levels, count);
            }
            body.frozenOrBranches.mint(vid);
            body.pendingWipeScopes.mint(vid);
        }
    }
    body.orLiveBranches.eraseIf([&](const NameId& v) {
        return body.frozenOrBranches.lookup(v) != 0
            || body.intValidityNamesToFilter.lookup(v) != 0;
    });
}


/// @brief Test whether an explicit induction ancestor still awaits its hypothesis.
///
/// @details A settled tree keeps its winning scopes, so scope syntax alone is
/// insufficient. Unopened OR queues wait until the existing settlement drain
/// records the firing; ordinary scopes and an induction block's main stay open.
/// @param body Resident owning block.
/// @param parent Minted parent of the proposed case.
/// @return True while any enclosing induction tree is unsettled.
/// @invariant Reads runtime settlement state, never proof history.
/// @see Memory::settleInductionTrees, drainPendingOrReleases.
bool CaseProcessor::orSplitBlockedByInduction(const Memory& body, NameId parent) const {
    assert(parent != 0);
    for (NameId scope = parent; scope != 0; scope = body.nameMap.parentOf(scope)) {
        if (ScopeSyntax::isInductionScope(body.nameMap, scope)
            && !body.inductionTreeSettled(scope)) return true;
    }
    return false;
}

/// @brief Whether one of an OR leaf's matched demands is a missing expression
///        of the block's induction hypothesis.
///
/// @details
/// The route-(c) probe of `consumeOrLeavesCohort` matches a leaf against the
/// ordis2 demand keys of the OR's scope. A key holds one record per demanding
/// rule; a record whose source is an induction hypothesis rule of the block
/// (`Memory::isInductionHypothesisRule`) is the hypothesis's missing
/// expression (D-575). A cohort opened by such a demand is
/// released below an unsettled induction scope, the I-328 exemption that lets
/// the hypothesis fire in its case. The records are read in place, heap-free.
///
/// @param mb    The block owning the demand map.
/// @param keys  The matched demand keys, each present in `admissionMapOrdis2`
///              (asserted).
/// @param arena The per-slot page-tier scratch arena for record reads.
/// @return `true` iff some record of some key has a hypothesis rule as source.
/// @invariant Pure read; mints nothing.
/// @see consumeOrLeavesCohort, drainPendingOrReleases,
///      AdmissionMemory::fileHypothesisOrDemand.
bool CaseProcessor::inductionHypothesisDemanded(const Memory& mb, const PagedVector<int64_t>& keys,
    ScratchArena& arena) {
    const auto& demands = mb.overallHashMemory.admissionMapOrdis2;
    for (int32_t k = 0; k < keys.size(); ++k) {
        const int32_t id = demands.lookup(keys[k]);
        assert(id != 0 && "inductionHypothesisDemanded: a matched demand key is in the demand map");
        for (int32_t r = 0; r < demands.runLen(id); ++r) {
            ScratchScope frame(arena);
            int32_t length = 0;
            const char* bytes = demands.peekRecordBytes(id, r, length, arena);
            if (mb.isInductionHypothesisRule(AdmissionOrdis2ValueBlobView(bytes, length).sourceImplId()))
                return true;
        }
    }
    return false;
}

/// @brief Find an existing execution of a case at exactly the requested scope.
///
/// @details Compare assumptions using only the parent's visible equalities.
/// Live executions include queued seeds. Completed executions qualify only
/// when saved results cover the still-unknown goals of the innermost goal
/// scope on the common parent chain (D-549).
/// Ties use decoded scope order. Wiped parents and retiring live cases cannot
/// supply work; a completed case may survive its own runtime wipe.
/// @param body Resident owning block.
/// @param parent Minted common parent scope.
/// @param assumption Unwrapped case expression.
/// @param completed Select completed cases instead of live executions.
/// @return Matching case scope, or zero for a defined cache miss.
/// @invariant No sibling or child equality can identify two cases.
/// @see freezeResolvedOrBranches, reuseOrCase.
NameId CaseProcessor::commonOrCaseAt(Memory& body, NameId parent,
    StrSpan assumption, bool completed) {
    const unsigned slot = WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
    ScratchArena& arena = scratchArenas().forSlot(slot);
    ScratchScope scratch(arena);
    assert(parent != 0);
    const StrSpan parentText = body.nameMap.decodeView(parent);
    int32_t noLevels = 0;
    const CanonicalForm wanted = canonicalizer.canonicalFormAtScope(
        assumption, parentText, body, arena, nullptr, noLevels, 0, true, false);
    const auto& cases = completed ? body.frozenOrBranches : body.orLiveBranches;
    NameId selected = 0;
    for (int32_t i = 1; i <= cases.count(); ++i) {
        const NameId candidate = cases.decode(i);
        if (body.nameMap.parentOf(candidate) != parent) continue;
        bool closingParent = false;
        for (NameId p = parent; p != 0; p = body.nameMap.parentOf(p))
            closingParent = closingParent || body.pendingWipeScopes.lookup(p) != 0
                || body.intValidityNamesToFilter.lookup(p) != 0;
        if (closingParent) continue;
        if (!completed && (body.frozenOrBranches.lookup(candidate) != 0
            || body.intValidityNamesToFilter.lookup(candidate) != 0
            || body.pendingDeadOrBranches.lookup(candidate) != 0
            || body.pendingWipeScopes.lookup(candidate) != 0)) continue;
        StrSpan signature, wrapped;
        assert(ScopeSyntax::classifyOrScopeView(body.nameMap, candidate, signature, wrapped)
            == OrScopeKind::Disintegration);
        assert(wrapped.len >= 2);
        ScratchScope candidateScratch(arena);
        const CanonicalForm existing = canonicalizer.canonicalFormAtScope(
            StrSpan(wrapped.ptr + 1, wrapped.len - 2), parentText, body,
            arena, nullptr, noLevels, 0, true, false);
        if (!equalSpans(wanted.text, existing.text)) continue;
        bool covered = true;
        if (completed) {
            // The goals a completed case resolved on: the innermost goal
            // scope of the common parent's chain (D-549).
            const NameId goalScope = StatementKnowledge::innermostGoalScope(body, parent);
            for (int32_t g = 1; g <= body.intToBeProved.count(); ++g) {
                const StatementKey goal = body.intToBeProved.decodeKey(g);
                if (goal.validity != goalScope) continue;
                if (StatementKnowledge::ancestorKnown(body, goal.orig, parent, true)) continue;
                if (body.lbMemory.closedOrGoalLevels.lookup(StatementKey{ goal.orig, candidate }) == 0)
                    covered = false;
            }
        }
        if (covered && (selected == 0 || compareSpans(body.nameMap.decodeView(candidate),
            body.nameMap.decodeView(selected)) < 0)) selected = candidate;
    }
    return selected;
}


/// @brief Collect exactly the parent equalities that identify two case assumptions.
///
/// @details A differing argument contributes the directed equality from the
/// destination argument to the source argument, together with that pair's
/// recorded derivation levels. Equal argument positions contribute nothing.
/// Repeated pairs are deduplicated. All evidence comes from the common parent's
/// complete class view; no history is read and no interner is minted.
/// @param body Resident owning block.
/// @param source Source case scope.
/// @param target Destination case at the same parent, with an equal assumption.
/// @param arena String arena retaining output equality text for the caller.
/// @param equalities Caller output buffer for directed equality expressions.
/// @param capacity Number of entries available in the output buffer.
/// @param levels Caller level buffer, ascending and unique, with capacity 256.
/// @param levelCount Current level count, extended by only the used equalities.
/// @return Number of directed equalities; zero for identical assumptions.
/// @invariant Unchanged arguments never add premise credit.
/// @see commonOrCaseAt, copySharedOrCaseHistory, reuseOrCase.
int32_t CaseProcessor::collectSharedOrCaseEqualities(Memory& body, NameId source,
    NameId target, ScratchArena& arena, StrSpan* equalities, int32_t capacity,
    int* levels, int32_t& levelCount) {
    const NameId parent = body.nameMap.parentOf(source);
    assert(parent != 0 && parent == body.nameMap.parentOf(target));
    StrSpan signature, sourceWrapped, targetWrapped;
    assert(ScopeSyntax::classifyOrScopeView(body.nameMap, source, signature, sourceWrapped)
        == OrScopeKind::Disintegration);
    assert(ScopeSyntax::classifyOrScopeView(body.nameMap, target, signature, targetWrapped)
        == OrScopeKind::Disintegration);
    const StrSpan sourceAssumption(sourceWrapped.ptr + 1, sourceWrapped.len - 2);
    const StrSpan targetAssumption(targetWrapped.ptr + 1, targetWrapped.len - 2);
    if (equalSpans(sourceAssumption, targetAssumption)) return 0;
    assert(equalSpans(extractExpressionSpan(sourceAssumption), extractExpressionSpan(targetAssumption)));
    assert((sourceAssumption.ptr[0] == '!') == (targetAssumption.ptr[0] == '!'));
    StrSpan sourceArgs[ExecutionParameters::MAX_ARITY], targetArgs[ExecutionParameters::MAX_ARITY];
    const int32_t count = getArgsSpans(sourceAssumption, sourceArgs, ExecutionParameters::MAX_ARITY);
    assert(count == getArgsSpans(targetAssumption, targetArgs, ExecutionParameters::MAX_ARITY));
    const int32_t bucket = body.canonicalEquivalenceClassesMap.lookup(parent);
    assert(bucket != 0 && "different shared assumptions require parent equality evidence");
    const unsigned slot = WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
    ScratchArena& general = genScratchArenas().forSlot(slot);
    ScratchScope scratch(general);
    NameId fromIds[ExecutionParameters::MAX_ARITY], toIds[ExecutionParameters::MAX_ARITY];
    int32_t used = 0;
    for (int32_t j = 0; j < count; ++j) {
        if (equalSpans(sourceArgs[j], targetArgs[j])) continue;
        const NameId from = body.nameMap.lookup(targetArgs[j]);
        const NameId to = body.nameMap.lookup(sourceArgs[j]);
        assert(from != 0 && to != 0);
        bool repeated = false;
        for (int32_t k = 0; k < used; ++k)
            repeated = repeated || (fromIds[k] == from && toIds[k] == to);
        if (repeated) continue;
        bool found = false;
        for (int32_t k = 0; k < body.canonicalEquivalenceClassesMap.runLen(bucket); ++k) {
            int32_t length = 0;
            const char* bytes = body.canonicalEquivalenceClassesMap.peekRecordBytes(bucket, k, length, general);
            const EquivalenceClassView cls{bytes, length};
            bool hasFrom = false, hasTo = false;
            for (int32_t m = 0; m < cls.memberCount(); ++m) {
                hasFrom = hasFrom || cls.memberId(m) == from;
                hasTo = hasTo || cls.memberId(m) == to;
            }
            if (!hasFrom || !hasTo) continue;
            cls.forEachLevel(packEqPairKey(from, to), [&](int level) {
                if (level >= 0) levelCount = insertLevelSorted(levels, levelCount, level, 256);
            });
            found = true;
            break;
        }
        assert(found && "a shared assumption pair must belong to one parent class");
        assert(used < capacity && used < ExecutionParameters::MAX_ARITY);
        const int32_t length = targetArgs[j].len + sourceArgs[j].len + 6;
        char* bytes = arena.allocBytes(length);
        int32_t at = 0;
        std::memcpy(bytes + at, "(=[", 3); at += 3;
        std::memcpy(bytes + at, targetArgs[j].ptr, targetArgs[j].len); at += targetArgs[j].len;
        bytes[at++] = ',';
        std::memcpy(bytes + at, sourceArgs[j].ptr, sourceArgs[j].len); at += sourceArgs[j].len;
        std::memcpy(bytes + at, "])", 2); at += 2;
        assert(at == length);
        equalities[used] = StrSpan(bytes, length);
        fromIds[used] = from; toIds[used] = to; ++used;
    }
    assert(used > 0);
    return used;
}

/// @brief Write a cohort's own proof history for a reused case derivation.
///
/// @details Copy the completed source's scoped history with one structural
/// prefix substitution. Parent evidence remains at its original scope. Replace
/// the source case assumption by the destination cohort's own OR source and,
/// for parent-equal assumptions, an explicit equality bridge. This function
/// writes documentation only; no inference or scheduling reads its results.
/// @param body Resident block containing both named cases.
/// @param source Completed case scope.
/// @param target Destination cohort's case scope, at the same parent.
/// @return Nothing.
/// @invariant Every exported case retains the existing verifier contract.
/// @see reuseOrCase, ProofHistory::findEqualityCiteScope.
void CaseProcessor::copySharedOrCaseHistory(Memory& body, NameId source, NameId target) {
    assert(source != target && body.nameMap.parentOf(source) == body.nameMap.parentOf(target));
    const unsigned sSlot = WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
    const unsigned gSlot = WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
    ScratchArena& arena = scratchArenas().forSlot(sSlot);
    ScratchArena& general = genScratchArenas().forSlot(gSlot);
    ScratchScope scratch(arena);
    ScratchScope generalScratch(general);
    const auto hold = [&](StrSpan text) {
        return StrSpan(ScratchString::copyFrom(arena, text.ptr, text.len));
    };
    const StrSpan sourceText = hold(body.nameMap.decodeView(source));
    const StrSpan targetText = hold(body.nameMap.decodeView(target));
    const StrSpan parent = hold(body.nameMap.decodeView(body.nameMap.parentOf(source)));
    StrSpan sourceSignature, sourceWrapped, targetSignature, targetWrapped;
    assert(ScopeSyntax::classifyOrScopeView(body.nameMap, source, sourceSignature, sourceWrapped)
        == OrScopeKind::Disintegration);
    assert(ScopeSyntax::classifyOrScopeView(body.nameMap, target, targetSignature, targetWrapped)
        == OrScopeKind::Disintegration);
    sourceSignature = hold(sourceSignature); sourceWrapped = hold(sourceWrapped);
    targetSignature = hold(targetSignature); targetWrapped = hold(targetWrapped);
    const StrSpan sourceAssumption(sourceWrapped.ptr + 1, sourceWrapped.len - 2);
    const StrSpan targetAssumption(targetWrapped.ptr + 1, targetWrapped.len - 2);
    const auto underSource = [&](StrSpan scope) {
        return equalSpans(scope, sourceText) || (scope.len > sourceText.len
            && std::memcmp(scope.ptr, sourceText.ptr, sourceText.len) == 0
            && startsWithSpan(StrSpan(scope.ptr + sourceText.len, scope.len - sourceText.len),
                "_boundary_", 10));
    };
    const auto remap = [&](StrSpan scope) {
        if (!underSource(scope)) return scope;
        const int32_t length = targetText.len + scope.len - sourceText.len;
        char* bytes = arena.allocBytes(length);
        std::memcpy(bytes, targetText.ptr, targetText.len);
        std::memcpy(bytes + targetText.len, scope.ptr + sourceText.len, scope.len - sourceText.len);
        return StrSpan(bytes, length);
    };
    const int cap = parameters.compressor_mode ? parameters.compressor_max_origins_per_expr
        : parameters.max_origin_per_expr;
    const int32_t originalCount = body.exprOriginMap.count();
    for (int32_t row = 1; row <= originalCount; ++row) {
        ScratchScope rowScratch(arena);
        StrSpan expression, validity;
        decodeOriginKeyCopy(body.exprOriginMap.decodeKey(row), body.originInterner,
            expression, validity, arena);
        if (!underSource(validity)) continue;
        const StrSpan destination = remap(validity);
        const int32_t lineCount = body.exprOriginMap.runLen(row);
        for (int32_t line = 0; line < lineCount; ++line) {
            ScratchScope lineScratch(arena);
            int32_t length = 0;
            const char* bytes = body.exprOriginMap.peekRecordBytes(row, line, length, general);
            char saved[ExecutionParameters::kMaxOriginBlobBytes];
            assert(length <= static_cast<int32_t>(sizeof(saved)));
            std::memcpy(saved, bytes, length);
            const IdOriginBlobView origin = viewIdOriginBlob(saved, length);
            if (equalSpans(validity, sourceText)
                && static_cast<OriginTag>(origin.tag) == OriginTag::orDisintegration) {
                assert(equalSpans(expression, sourceAssumption));
                continue;
            }
            OriginDep dependencies[ExecutionParameters::kMaxOriginDeps];
            assert(origin.depN <= ExecutionParameters::kMaxOriginDeps);
            for (int32_t j = 0; j < origin.depN; ++j) {
                StrSpan text, scope;
                decodeOriginKeyCopy(idOriginDepAt(origin, j), body.originInterner,
                    text, scope, arena);
                dependencies[j] = { text, remap(scope) };
            }
            addOriginEncoded(body.exprOriginMap, body.originInterner, expression,
                destination, static_cast<OriginTag>(origin.tag), dependencies, origin.depN, cap);
        }
    }
    const OriginDep seed{ targetSignature, parent };
    addOriginEncoded(body.exprOriginMap, body.originInterner, targetAssumption,
        targetText, OriginTag::orDisintegration, &seed, 1, cap);
    if (!equalSpans(sourceAssumption, targetAssumption)) {
        int levels[256]; int32_t count = 0;
        StrSpan equalities[ExecutionParameters::kMaxOriginDeps - 1];
        const int32_t equalityCount = collectSharedOrCaseEqualities(body, source, target,
            arena, equalities, ExecutionParameters::kMaxOriginDeps - 1, levels, count);
        OriginDep dependencies[ExecutionParameters::kMaxOriginDeps];
        dependencies[0] = { targetAssumption, targetText };
        for (int32_t j = 0; j < equalityCount; ++j) {
            const StrSpan cite = proofHistory.findEqualityCiteScope(body, equalities[j], parent);
            dependencies[j + 1] = { equalities[j], hold(cite) };
        }
        addOriginEncoded(body.exprOriginMap, body.originInterner, sourceAssumption,
            targetText, OriginTag::equality1, dependencies, equalityCount + 1, cap);
    }
}

/// @brief Reuse a completed case without creating a second runtime subtree.
///
/// @details Save the destination's goal contributions and derivation levels,
/// including the parent equalities identifying the case. The destination
/// cohort adds its own OR-source levels only at convergence.
/// Equality-supported goals retain their actual fact and equality evidence.
/// Each contribution enters ordinary cohort convergence; the destination is
/// frozen before booking so it can never become a live inference branch.
/// @param body Resident owning block.
/// @param source Completed execution at the common parent.
/// @param target Newly named destination case at that same parent.
/// @return Nothing.
/// @invariant Reuse never imports a sibling assumption or reads history to prove a goal.
/// @see copySharedOrCaseHistory, commonOrCaseAt, ordisMerge.
void CaseProcessor::reuseOrCase(Memory& body, NameId source, NameId target) {
    assert(source != target && body.frozenOrBranches.lookup(source) != 0);
    const NameId parent = body.nameMap.parentOf(source);
    assert(parent == body.nameMap.parentOf(target));
    const unsigned slot = WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
    ScratchArena& arena = scratchArenas().forSlot(slot);
    ScratchScope scratch(arena);
    const auto hold = [&](StrSpan text) {
        return StrSpan(ScratchString::copyFrom(arena, text.ptr, text.len));
    };
    const StrSpan sourceText = hold(body.nameMap.decodeView(source));
    const StrSpan targetText = hold(body.nameMap.decodeView(target));
    const StrSpan parentText = hold(body.nameMap.decodeView(parent));
    const auto remapScope = [&](NameId scope) {
        if (!body.nameMap.ancContains(scope, source)) return scope;
        const StrSpan text = body.nameMap.decodeView(scope);
        const int32_t length = targetText.len + text.len - sourceText.len;
        char* buffer = arena.allocBytes(length);
        std::memcpy(buffer, targetText.ptr, targetText.len);
        std::memcpy(buffer + targetText.len, text.ptr + sourceText.len, text.len - sourceText.len);
        return body.nameMap.encode(StrSpan(buffer, length));
    };
    int sharedLevels[256]; int32_t sharedCount = 0;
    StrSpan equalities[ExecutionParameters::kMaxOriginDeps - 1];
    collectSharedOrCaseEqualities(body, source, target, arena, equalities,
        ExecutionParameters::kMaxOriginDeps - 1, sharedLevels, sharedCount);
    if (parameters.trackHistory) copySharedOrCaseHistory(body, source, target);
    body.frozenOrBranches.mint(target);
    body.pendingWipeScopes.mint(target);
    const int32_t goalCount = body.intToBeProved.count();
    // The goals the completed source resolved on (D-549).
    const NameId goalScope = StatementKnowledge::innermostGoalScope(body, parent);
    for (int32_t g = 1; g <= goalCount; ++g) {
        const StatementKey goal = body.intToBeProved.decodeKey(g);
        if (goal.validity != goalScope) continue;
        if (StatementKnowledge::ancestorKnown(body, goal.orig, parent, true)) continue;
        const StatementKey from{ goal.orig, source }, to{ goal.orig, target };
        const int32_t saved = body.lbMemory.closedOrGoalLevels.lookup(from);
        assert(saved != 0 && "completed shared case must cover every outstanding chain goal");
        int levels[256]; int32_t count = 0;
        for (int32_t j = 0; j < sharedCount; ++j)
            if (sharedLevels[j] >= 0) count = insertLevelSorted(levels, count, sharedLevels[j], 256);
        for (int32_t j = 0; j < body.lbMemory.closedOrGoalLevels.runLen(saved); ++j) {
            const int level = body.lbMemory.closedOrGoalLevels.valueAt(saved, j);
            if (level >= 0) count = insertLevelSorted(levels, count, level, 256);
        }
        if (count == 0) levels[count++] = -1;
        body.lbMemory.closedOrGoalLevels.assignSet(to, levels, count);
        const int64_t* support = body.lbMemory.orSupportFacts.find(from);
        if (support != nullptr) {
            const StatementKey fact = Codec<StatementKey>::decode(*support);
            const NameId scope = remapScope(fact.validity);
            body.lbMemory.orSupportFacts.upsert(to, packStatementKey(fact.orig, scope));
            const int32_t equalities = body.lbMemory.orSupportEqualities.lookup(from);
            assert(equalities != 0);
            const int32_t n = body.lbMemory.orSupportEqualities.runLen(equalities);
            for (int32_t j = 0; j < n; ++j) {
                const StatementKey eq = Codec<StatementKey>::decode(
                    body.lbMemory.orSupportEqualities.valueAt(equalities, j));
                const NameId at = remapScope(eq.validity);
                body.lbMemory.orSupportEqualities.insertSorted(to, packStatementKey(eq.orig, at),
                    [&](int64_t a, int64_t b) {
                        const StatementKey x = Codec<StatementKey>::decode(a), y = Codec<StatementKey>::decode(b);
                        const int order = compareSpans(body.nameMap.decodeView(x.orig), body.nameMap.decodeView(y.orig));
                        return order != 0 ? order < 0 : compareSpans(body.nameMap.decodeView(x.validity), body.nameMap.decodeView(y.validity)) < 0;
                    });
            }
        }
        const StrSpan expression = hold(body.nameMap.decodeView(goal.orig));
        ordisMerge(expression, targetText, levels, count, body);
    }
}

/// @brief End-of-burst release drain of the sequenced
        ///        or-disintegration cohorts — mint the next pending branch of
        ///        every cohort whose live branch resolved this burst.
        ///
        /// @details
        /// Consumes `Memory::pendingOrReleases` (staged by the cohort
        /// bootstrap in `disintegrateExprCore2` — the INITIAL release, since
        /// the bootstrap itself may not touch the mail channel being
        /// drained around it; by `ordisMerge`'s goal probe when a
        /// branch-scope deposit matches a `toBeProved` goal on the branch's
        /// parent chain; and by `drainDeadOrBranches` when a live branch
        /// retires refuted). Per staged cohort, in decoded cohort-identity
        /// order (parent bytes, then signature bytes — I-84):
        ///
        /// 1. **Defined skips.** No `orDisjunctCount` row (the cohort retired
        ///    wholesale — a stale staging), no `orPendingBranches` row
        ///    (nothing left to release), or the cohort's parent scope sits in
        ///    `intValidityNamesToFilter` (a retired region — the queue is
        ///    left untouched).
        /// 2. **Wait or reuse.** An unfired induction ancestor parks the
        ///    release, except for a cohort in `LbMemory::hypothesisOrCohorts`
        ///    (opened by the induction hypothesis's missing expression,
        ///    D-575), whose cases release beneath it.
        ///    Any live same-parent case contained in the remaining
        ///    cohort blocks the whole cohort, irrespective of rank. Completed
        ///    shared cases contribute their saved results and own scoped proof.
        /// 3. **Rank.** The pending disjuncts are re-ranked by
        ///    `pickTopOrDisjunct` over their clean forms plus the LB's anchor
        ///    arguments (`collectAnchorArgs`) — the release order is a pure
        ///    function of the remaining set, never stored.
        /// 4. **Release.** The top-ranked disjunct's branch scope name
        ///    `<parent>_boundary_ordis_<orSig>_(<disjunct>)` is built on the
        ///    string tier; the seed statement plus its `or disintegration`
        ///    origin ride `sameIterationInternalMail` (the `ordisMerge`
        ///    promotion channel) as a level-free hypothetical assumption;
        ///    `orPendingLevels` supplies this cohort's source levels at convergence, so the next absorb runs the seed through
        ///    the full kernel pipeline — a compound disjunct decomposes there
        ///    like any deposited statement.
        /// 5. **Queue update.** The released disjunct leaves
        ///    `orPendingBranches` (erase + reinsert through the canonical
        ///    doors); `orPendingLevels` stays until the cohort retires.
        ///    One release or reuse per staged cohort per burst.
        ///
        /// Runs in `standardProcessing` between `drainDeadOrBranches` (so a
        /// refuted live branch releases its successor in the same burst) and
        /// the `pendingWipeScopes` drain. Heap-free (Rule 28): the cohort
        /// snapshot rides the gen-scratch tier, the branch-validity strings
        /// the string tier, both reclaimed at exit.
        ///
        /// @param memoryBlock The LB whose release inbox is drained (its
        ///                    containers are resident — the drain runs inside
        ///                    the LB's own burst).
        /// @invariant A block without explicit goals or active contradiction
        ///            work releases nothing; its staging drops.
        /// @invariant At most one branch is released per cohort per burst;
        ///            the staged-cohort processing order and the released
        ///            disjunct are pure functions of container content.
        /// @see Memory::pendingOrReleases, Memory::orPendingBranches,
        ///      Memory::orPendingLevels, pickTopOrDisjunct,
        ///      collectAnchorArgs, ordisMerge, drainDeadOrBranches.
void CaseProcessor::drainPendingOrReleases(Memory& body) {
    const int32_t stagedN = body.pendingOrReleases.count();
    if (stagedN == 0) return;
    // Active contradiction blocks can advance cases without explicit goals.
    // Other blocks without proof work discard the release staging.
    if (!hasOrProofWork(body)) {
        body.pendingOrReleases.resetToFresh();
        return;
    }

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

    // Snapshot + reset (the pendingDeadOrBranches idiom), then sort by the
    // decoded cohort identity — parent bytes, then signature bytes. The pod
    // set dedups and the cohort id is injective over the (parent, orSig)
    // pair, so the sort is tie-free (I-84).
    int32_t* cids = reinterpret_cast<int32_t*>(gArena.resolve(
        gArena.alloc(stagedN * static_cast<int32_t>(sizeof(int32_t)),
                     alignof(int32_t))));
    for (int32_t i = 1; i <= stagedN; ++i)
        cids[i - 1] = body.pendingOrReleases.decode(i);
    body.pendingOrReleases.resetToFresh();
    std::sort(cids, cids + stagedN, [&](int32_t a, int32_t b) {
        const LbStatePairKey ca = decodeOrCohortIds(body.lbStateInterner, a);
        const LbStatePairKey cb = decodeOrCohortIds(body.lbStateInterner, b);
        const int c = compareSpans(
            body.lbStateInterner.decodeView(static_cast<int32_t>(ca.high)),
            body.lbStateInterner.decodeView(static_cast<int32_t>(cb.high)));
        if (c != 0) return c < 0;
        return compareSpans(
            body.lbStateInterner.decodeView(static_cast<int32_t>(ca.low)),
            body.lbStateInterner.decodeView(static_cast<int32_t>(cb.low)))
            < 0;
    });

    // The queue rewrites of the released cohorts are recorded here and applied
    // after the loop: ONE compaction of orPendingBranches per drain
    // instead of one per release — the whole-map
    // eraseSetIf (a full survivor slide plus an index rebuild) was the drain's
    // cost. The result is the sequential one byte for byte: every released key
    // is erased, then re-minted at the end of the key column in release order
    // with the same sorted inserts, and no cohort of this drain reads another
    // cohort's row in between.
    // Page-tier records (a drain can stage thousands of cohorts — the D1
    // negated-head block — so no single byte-bump run may size by the count).
    struct ReleasedCohort { int32_t cid; int32_t survivorStart; int32_t survivorCount; };
    DirtyState drainDirty = DirtyState::Clean;
    PagedVector<ReleasedCohort> released(&gArena, &drainDirty);
    PagedVector<int32_t> survivors(&gArena, &drainDirty);

    for (int32_t ci = 0; ci < stagedN; ++ci) {
        const int32_t cid = cids[ci];
        // Defined skips: the cohort retired wholesale (count row gone — a
        // stale staging), or nothing is pending.
        if (body.orDisjunctCount.lookup(cid) == 0) continue;
        const int32_t pRow = body.orPendingBranches.lookup(cid);
        if (pRow == 0) continue;
        const LbStatePairKey cohort =
            decodeOrCohortIds(body.lbStateInterner, cid);
        const StrSpan parentV = body.lbStateInterner.decodeView(
            static_cast<int32_t>(cohort.high));
        const StrSpan sigV = body.lbStateInterner.decodeView(
            static_cast<int32_t>(cohort.low));
        // A filtered parent scope belongs to a retired region — the seed
        // deposit would be refused there; leave the queue untouched.
        const NameId parentVid = body.nameMap.lookup(parentV);
        assert(parentVid != 0
            && "ordis cohort parent scope must be interned");
        if (body.intValidityNamesToFilter.lookup(parentVid) != 0) continue;
        bool parentClosing = false;
        for (NameId p = parentVid; p != 0; p = body.nameMap.parentOf(p))
            parentClosing = parentClosing || body.pendingWipeScopes.lookup(p) != 0;
        if (parentClosing) continue;
        // A cohort opened by the induction hypothesis's missing expression is
        // exempt: its cases carry what the hypothesis needs to fire, the one
        // event that settles the tree (D-575).
        if (body.lbMemory.hypothesisOrCohorts.lookup(cid) == 0
            && orSplitBlockedByInduction(body, parentVid)) {
            body.pendingOrReleases.mint(cid);
            continue;
        }

        // Rank the pending disjuncts over their clean forms and pick the
        // release. The wrapped payload bodies come off the queue run in
        // decoded-lex storage order; ids are captured before the rebuild.
        const int32_t pendN = body.orPendingBranches.runLen(pRow);
        assert(pendN >= 1
            && pendN <= ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
            && "ordis pending run exceeds the flattened-leaf cap");
        int32_t djIds[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
        StrSpan cleanSpans[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
        for (int32_t j = 0; j < pendN; ++j) {
            djIds[j] = body.orPendingBranches.valueAt(pRow, j);
            const StrSpan w = body.lbStateInterner.decodeView(djIds[j]);
            assert(w.len >= 2 && w.ptr[0] == '('
                && w.ptr[w.len - 1] == ')'
                && "pending ordis disjunct must be a wrapped (disjunct)");
            cleanSpans[j] = StrSpan(w.ptr + 1, w.len - 2);
        }
        // A zero block's case-experiment cohort releases its one case and no
        // other: once that case is refuted, the OR's exclusion derives the
        // other disjunct at the parent (D-588).
        const bool experiment = body.lbMemory.experimentOrCohorts.lookup(cid) != 0;
        int32_t experimentId = 0;
        if (experiment) {
            const int32_t* starter = body.orStarterPick.find(cid);
            if (starter == nullptr) continue;
            experimentId = *starter;
            bool stillPending = false;
            for (int32_t j = 0; j < pendN; ++j)
                stillPending = stillPending || djIds[j] == experimentId;
            if (!stillPending) {
                body.orStarterPick.erase(cid);
                continue;
            }
        }
        // A shared live case blocks the entire cohort before ranking or
        // consuming its starter. Completed shared cases are reused first.
        bool sharedLive = false;
        NameId sharedSource = 0;
        int32_t sharedIndex = -1;
        for (int32_t j = 0; j < pendN; ++j) {
            if (experiment && djIds[j] != experimentId) continue;
            if (commonOrCaseAt(body, parentVid, cleanSpans[j], false) != 0) {
                sharedLive = true;
                break;
            }
            const NameId finished = commonOrCaseAt(body, parentVid, cleanSpans[j], true);
            if (finished != 0 && sharedSource == 0) {
                sharedSource = finished;
                sharedIndex = j;
            }
        }
        if (sharedLive) {
            body.pendingOrReleases.mint(cid);
            continue;
        }
        StrSpan anchorArgs[ExecutionParameters::MAX_ARITY];
        const int32_t anchorArgN = collectAnchorArgs(
            body, anchorArgs, ExecutionParameters::MAX_ARITY);
        int32_t top =
            sharedSource != 0 ? sharedIndex
                : pickTopOrDisjunct(cleanSpans, pendN, anchorArgs, anchorArgN);
        // Route-(a) starter override, ONE-SHOT: a cohort opened by the
        // demand probe releases its ADMITTED disjunct first — the demand
        // names which branch carries the relevance. The row is erased on
        // consumption (later releases rank as usual); a recorded starter no
        // longer pending (already released or retired) falls back to the
        // ranking — a defined skip, not a failure.
        bool starterRelease = false;
        {
            const int32_t* starterPayload = body.orStarterPick.find(cid);
            if (starterPayload != nullptr && sharedSource == 0) {
                const int32_t starterId = *starterPayload;
                body.orStarterPick.erase(cid);
                for (int32_t j = 0; j < pendN; ++j) {
                    if (djIds[j] == starterId) {
                        top = j;
                        starterRelease = true;
                        break;
                    }
                }
            }
        }
        const StrSpan wrappedTop = body.lbStateInterner.decodeView(djIds[top]);

        // No OR case opens inside an integration trial scope or the
        // usefulness scope (D-481, D-521).
        assert(!gl::ScopeSyntax::atOrBelowRestrictedScope(body.nameMap, body.nameMap.lookup(parentV))
            && "drainPendingOrReleases: no OR case opens inside a restricted scope");
        // Branch validity "<parent>_boundary_ordis_<orSig>_(<disjunct>)" on
        // the string tier, explicit-length.
        ScratchScope relScope(sArena);
        const StrSpan BOUNDARY_ORDIS("_boundary_ordis_", 16);
        const int32_t bvLen = parentV.len + BOUNDARY_ORDIS.len + sigV.len
            + 1 + wrappedTop.len;
        char* bvBuf = sArena.allocBytes(bvLen);
        {
            int32_t at = 0;
            std::memcpy(bvBuf + at, parentV.ptr,
                static_cast<std::size_t>(parentV.len));
            at += parentV.len;
            std::memcpy(bvBuf + at, BOUNDARY_ORDIS.ptr,
                static_cast<std::size_t>(BOUNDARY_ORDIS.len));
            at += BOUNDARY_ORDIS.len;
            std::memcpy(bvBuf + at, sigV.ptr,
                static_cast<std::size_t>(sigV.len));
            at += sigV.len;
            bvBuf[at++] = '_';
            std::memcpy(bvBuf + at, wrappedTop.ptr,
                static_cast<std::size_t>(wrappedTop.len));
            at += wrappedTop.len;
            assert(at == bvLen);
        }
        const StrSpan branchV(bvBuf, bvLen);

        // A case is a hypothetical assumption. Its OR source contributes
        // levels only when the cohort converges, so the conditional proof
        // can be shared without inheriting a different cohort's premises.
        const int assumptionLevels[] = { -1 };

        // Quiescence (D-194): the release creates work for the next absorb.
        body.mutatedThisBurst = true;
        const NameId branchId = body.nameMap.encode(branchV);
        const StrSpan negation = negateOrCaseAssumption(wrappedTop, sArena);
        const NameId negationId = body.nameMap.lookup(negation);
        const bool refuted = negationId != 0
            && StatementKnowledge::ancestorKnown(body, negationId, parentVid, true);
        if (refuted) {
            // The existing retirement drain records the parent refutation
            // and reduces the cohort without executing the refuted case.
            body.pendingDeadOrBranches.mint(branchId);
        } else if (sharedSource != 0) {
            reuseOrCase(body, sharedSource, branchId);
            body.pendingOrReleases.mint(cid);
        } else {
            insertInternalStatement(body.sameIterationInternalMail, body.nameMap,
                cleanSpans[top], branchV, assumptionLevels, 1);
            // The case of a cohort opened from a conjecture premise's OR is an
            // assumption like a premise: the absorb disintegrates it with force_
            // witnesses (D-527).
            if (body.lbMemory.forcedOrCohorts.lookup(cid) != 0)
                setInternalDisintegrationSignal(body.sameIterationInternalMail, body.nameMap,
                    cleanSpans[top], branchV, /*dnd=*/false, /*aod=*/false, /*iteration=*/-1,
                    /*rh=*/false, /*orSource=*/false, /*forced=*/true);
            if (parameters.trackHistory) {
                const OriginDep relDeps[1] = { { sigV, parentV } };
                addInternalMailOrigin(body.sameIterationInternalMail,
                    body.originInterner, cleanSpans[top], branchV,
                    OriginTag::orDisintegration, relDeps, 1,
                    (parameters.compressor_mode
                        ? parameters.compressor_max_origins_per_expr
                        : parameters.max_origin_per_expr));
            }
            // Reserve the execution before the seed is absorbed, so later
            // cohorts in this same drain see the live common case.
            body.orLiveBranches.mint(branchId);
            // The weak step block's twin cohort releases its successor case
            // first, a base-case block's experiment cohort that case only;
            // the second twin goal or the partner's value demands are filed
            // in that case's scope (D-589,
            // D-588).
            assert((!experiment || body.twinExperiment == TwinExperiment::successorCase)
                && "drainPendingOrReleases: an experiment cohort belongs to a base-case block");
            if (starterRelease && (cid == body.twinCohortId || experiment)) {
                assert(body.twinCaseScopeId == 0
                    && "drainPendingOrReleases: the twin cohort releases its successor case once");
                body.twinCaseScopeId = branchId;
            }
        }

        // The released disjunct leaves the queue: record the survivors; the
        // run is rebuilt without it after the loop (one compaction per drain).
        // The OR-source levels survive until this cohort retires.
        released.push_back(ReleasedCohort{ cid, survivors.size(), pendN - 1 });
        for (int32_t j = 0; j < pendN; ++j) {
            if (j == top) continue;
            survivors.push_back(djIds[j]);
        }
    }

    const int32_t releasedN = released.size();
    if (releasedN > 0) {
        RT_SCOPE_HERE("DRAIN_OR_QUEUE_REWRITE");
        // Membership set for the queue compaction (the release
        // order stays in `released`).
        ColdHashSet<PodKeyStore<int32_t>> releasedSet(&gArena, &drainDirty);
        releasedSet.resetToFresh();
        for (int32_t r = 0; r < releasedN; ++r) {
            releasedSet.mint(released[r].cid);
        }
        body.orPendingBranches.eraseSetIf([&](int32_t k) {
            return releasedSet.lookup(k) != 0;
        });
        for (int32_t r = 0; r < releasedN; ++r) {
            const ReleasedCohort rc = released[r];
            for (int32_t j = 0; j < rc.survivorCount; ++j) {
                body.orPendingBranches.insertSorted(rc.cid,
                    survivors[rc.survivorStart + j],
                    DecodedIdLess{ &body.lbStateInterner });
            }
        }
        releasedSet.release();
    }
    released.clear();
    survivors.clear();
    gArena.popTo(gMark);
}

// Turn optimizations OFF for just this section
//#pragma optimize("", off)


/// @brief Queue every sibling branch of one proved OR-integration cohort for
///        radical subtree cleanup.
///
/// @details
/// Selects only validity scopes that classify as OR integration, carry the
/// exact outer signature @p orExprSig, and have @p parentValidityId as their
/// direct NameMap parent. The victims are synchronously filtered for the rest
/// of the current burst and queued for the end-of-burst `Memory::wipeSubtree`
/// drain. Because parent validity participates in selection, equal OR
/// signatures at another stack position remain live.
///
/// The scan stores ids on the request-generation scratch page tier and mints
/// no NameMap entries, preserving the caller-owned signature span and every
/// classify-view slice throughout both loops.
///
/// @param orExprSig Outer compiled OR signature shared by the cohort.
/// @param parentValidityId Exact direct parent of the proving goal scope.
/// @param mb Owning logic-block memory.
/// @return Nothing.
/// @invariant Every queued victim is a direct child of
///            @p parentValidityId and belongs to @p orExprSig.
/// @see Memory::wipeSubtree, ScopeSyntax::classifyOrScopeView.
void CaseProcessor::cleanUpOrIntegrationBranches(
    StrSpan orExprSig,
    NameId parentValidityId,
    Memory& mb) {
    const NameMap& nm = mb.nameMap;
    assert(parentValidityId != 0
           && parentValidityId <= nm.nameCount()
           && "cleanUpOrIntegrationBranches: invalid parent validity id");

    // The victim validity ids ride a PagedVector on the per-slot gen-scratch
    // PAGE tier (I-138), freed by ~PagedVector
    // at return.
    const unsigned slot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : genScratchArenas().slotCount() - 1;
    ScratchArena& gArena = genScratchArenas().forSlot(slot);
    DirtyState victimDirty = DirtyState::Clean;
    PagedVector<NameId> victimIds(&gArena, &victimDirty);

    StrSpan orExprOutV;
    StrSpan bodyOutV;

    // Scan every validity id (skip slot 0 = invalid, slot 1 = MAIN). Store
    // only the id; the victim's decoded scope name is re-derived at the use
    // site below via nm.decodeView + a non-minting lookup that round-trips to
    // the same id. No NameMap mint occurs anywhere in this function —
    // classifyOrScopeView takes a const NameMap& and the loop-2 mints target
    // pendingWipeScopes / intValidityNamesToFilter, not the NameMap — which
    // is doubly load-bearing now: it keeps the parameter span AND the
    // classify twin's sub-table out-spans valid across both loops (I-3;
    // orExprOutV is consumed by equalSpans in the SAME iteration, and
    // bodyOutV — like the discarded bodyOut before it — is never read; on
    // a non-Integration return the outs stay untouched, so a stale span
    // from a previous iteration exists but is never read, identical to the
    // string form's stale content).
    const int32_t nameHighWater = nm.nameCount();
    for (NameId id = 2; id <= nameHighWater; ++id) {
        if (gl::ScopeSyntax::classifyOrScopeView(nm, id, orExprOutV, bodyOutV)
                == OrScopeKind::Integration
            && nm.parentOf(id) == parentValidityId
            && equalSpans(orExprOutV, orExprSig)) {
            victimIds.push_back(id);
        }
    }

    for (int32_t i = 0; i < victimIds.size(); ++i) {
        const NameId id = victimIds[i];

        // D-72: defer the radical sweep to
        // end-of-burst drainage by performElemPhase3 —
        // immediate wipe would invalidate the kernel's in-flight
        // sortedNew loop iteration (the convergence path runs inside
        // that loop).
        {
            // Re-derive the decoded scope name from the stored id; no
            // NameMap mint occurs before this lookup, so the span is valid.
            const StrSpan vname = nm.decodeView(id);
            const NameId wipeVid = mb.nameMap.lookup(vname);
            assert(wipeVid != 0);
            mb.pendingWipeScopes.mint(wipeVid);
        }

        // D-73 (sibling-site parity):
        // parity with the same closure-time logic at the NotOrScope
        // branch of addExprToMemoryBlock.
        // Without these synchronous inserts the sortedNew loop
        // continues to admit rule-fires whose consensus scope is a
        // descendant of the victim _orint_ branch for the remainder of
        // the current burst: the ancestor-walk gate at memory.cpp
        // (checkLocalEncodedMemoryStatic) sees an empty
        // intValidityNamesToFilter for every ancestor of the victim id
        // until end-of-burst wipeSubtree's belt-and-suspenders step 11
        // populates the filters. The intervening rule-fires burn
        // sortedNew budget, mint counters, and routing-side hash work
        // inside the about-to-be-wiped subtree, and prevent the
        // post-convergence closure of the enclosing implication.
        // Filter from the moment of OR convergence; the wipe still
        // drains at end-of-burst for radical descendant cleanup.
        mb.intValidityNamesToFilter.mint(id);
    }
}

/// @brief Open or park one case-split leaf cohort.
/// @details Ordered flattened leaves share one parent-scope identity.
/// Proof work, depth, demand, and disintegration permission control cases;
/// an OR with a disjunct that already holds at the scope (`orDisjunctTrueAt`)
/// neither opens a case split nor parks (D-438). The depth is the number of
/// `_ordis_` case scopes on the scope's ancestry; `_orint_` branches do not
/// count (D-458). A source is a positive category-or
/// entity, or a negated conjunction deposited as an integration premise,
/// whose leaves are its complemented conjuncts and which always carries
/// route-(b) permission (D-460). A cohort opened through a demand whose
/// source is an induction hypothesis rule (`inductionHypothesisDemanded`) is
/// recorded in `LbMemory::hypothesisOrCohorts`: its cases are exempt from the
/// I-328 release block below an unsettled induction scope
/// (D-575).
/// Exclusion is handled separately by finalized source registration and
/// the on-demand evaluator; this consumer installs no exclusion rules.
/// @param expr Deposited OR expression for history.
/// @param entSignature Instantiated source signature with u-prefixed
///        arguments: an OR compact, or the negated conjunction.
/// @param orLeaves Ordered disjunct spans at their true polarity.
/// @param orLeafN Number of flattened disjuncts.
/// @param currentStatement Collected OR key.
/// @param memoryBlock Owning block.
/// @param collected Disintegration product sink.
/// @param validityName Parent scope.
/// @param trackHistoryLocal History permission.
/// @param allowOrDisintegration Unconditional case-opening signal.
/// @param orSeedLevels Ordered seed levels.
/// @param orSeedLevelCount Number of seed levels.
/// @param allowOrProbe Real-deposit demand-probe permission.
/// @param negatedConjunction The source is a negated conjunction integration
///        premise; false for a category-or entity.
/// @param forcedAssumptions The OR came from a conjecture premise (in place or
///        on a flag-6 arrival): a freshly opened cohort is recorded in
///        `LbMemory::forcedOrCohorts`, so each released case is disintegrated
///        with force_ witnesses (D-527).
/// @param sArena Caller string scratch.
/// @param instructions Caller instruction context.
/// @param iteration Caller witness iteration.
/// @param newVarMap Caller witness map.
/// @param parentWitness Caller parent witness.
/// @param witnessMeta Caller witness metadata.
/// @return Void.
/// @invariant Only category-or entities and route-(b) negated conjunction
///            premises enter this consumer; all writes are block-local.
/// @see registerOrImplicationSource, evaluateOrImplication, flattenOrLeaves.
void CaseProcessor::consumeOrLeavesCohort(StrSpan expr,
    StrSpan entSignature,
    const StrSpan* orLeaves, int32_t orLeafN,
    StrSpan currentStatement,
    Memory& memoryBlock,
    CollectedArena& collected,
    StrSpan validityName,
    bool trackHistoryLocal,
    bool allowOrDisintegration,
    const int* orSeedLevels, int32_t orSeedLevelCount,
    bool allowOrProbe,
    bool negatedConjunction,
    bool forcedAssumptions,
    ScratchArena& sArena,
    WorkInstruction& instructions,
    int iteration,
    NewVarStore& newVarMap,
    StrSpan parentWitness,
    WitnessMetaStore* witnessMeta,
    int32_t experimentLeaf) {
    assert((experimentLeaf < 0 || (experimentLeaf < orLeafN && allowOrDisintegration))
        && "consumeOrLeavesCohort: a case experiment names one of the leaves and opens like route (b)");
    // OR nesting depth: the `_ordis_` case scopes on the scope's NameMap
    // ancestry, the scope itself included. `_orint_` integration branches and
    // every other scope kind do not count, so `max_or_depth` limits case-split
    // nesting only (D-458).
    int currentOrDepth = 0;
    {
        const NameId scopeId = memoryBlock.nameMap.lookup(validityName);
        assert(scopeId != 0 && "consumeOrLeavesCohort: the OR scope is interned");
        for (NameId v = scopeId; v != 0 && v != NameMap::MAIN_ID;
             v = memoryBlock.nameMap.parentOf(v)) {
            StrSpan orExprView, branchBodyView;
            if (gl::ScopeSyntax::classifyOrScopeView(memoryBlock.nameMap, v, orExprView, branchBodyView)
                    == OrScopeKind::Disintegration) {
                ++currentOrDepth;
            }
        }
    }

    // Case history preserves the expanded OR independently of the
    // on-demand exclusion evaluator. Only case scopes use the depth gate.
    const StrSpan entCategory("or", 2);
    const ScratchString expandedOrSignature = gl::DefinitionExpansion::expandSignature(
        entCategory, entSignature, orLeaves, orLeafN, sArena);
    if (parameters.trackHistory && trackHistoryLocal) {
        const int expCap = (parameters.compressor_mode
            ? parameters.compressor_max_origins_per_expr
            : parameters.max_origin_per_expr);
        const ScratchString exprClean = gl::DefinitionExpansion::removeUPrefixScratch(sArena, StrSpan(expr));
        const OriginDep expDeps[1] = { { StrSpan(exprClean), StrSpan(validityName) } };
        addOriginEncoded(memoryBlock.exprOriginMap, memoryBlock.originInterner,
            StrSpan(expandedOrSignature), StrSpan(validityName),
            OriginTag::expansion, expDeps, 1, expCap);
        memoryBlock.addMailOutOrigin(StrSpan(expandedOrSignature),
            StrSpan(validityName), OriginTag::expansion, expDeps, 1, expCap);
    }

    // Exclusion processing is registered from the finalized OR statement.
    // This consumer only controls case opening and parking. A negated
    // conjunction enters only as a forced integration premise.
    assert((!negatedConjunction || allowOrDisintegration)
        && "consumeOrLeavesCohort: a negated conjunction source is a route-(b) integration premise");
    if (currentOrDepth < parameters.max_or_depth) {
        // 2. Cohort opening. Route (b): allowOrDisintegration — the firing
        //    rule is a product of disintegration (the threaded D-32
        //    signal), or the deposit is an integration premise
        //    (D-460); such sources open
        //    unconditionally. Route (c): a clean leaf is a whole-compound
        //    admissionMapOrdis2 demand. The route-(a) admissionMap
        //    product-template probe still runs but no longer authorizes
        //    opening. Neither route: the cohort PARKS in rejectedMapOrdis
        //    under its operator-based product templates (dual-filed in
        //    rejectedMapOrdis2) and revives by mail. Probe and park run
        //    only for real deposits (allowOrProbe); the hypothetical path
        //    neither opens nor parks. The explosion protection is BOTH the
        //    demand filter and the sequenced one-branch-at-a-time release
        //    below.

        // Clean leaf forms, computed once for probe / park / bootstrap
        // (byte-identical to the former per-leaf strip in the mint
        // block; multiple spans under the enclosing scope).
        ScratchString cleanHold[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
        StrSpan cleanLeaves[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
        for (int32_t i = 0; i < orLeafN; ++i) {
            cleanHold[i] = gl::DefinitionExpansion::removeUPrefixScratch(sArena, orLeaves[i]);
            cleanLeaves[i] = StrSpan(cleanHold[i]);
        }

        // Product templates of leaf `li`, fed to `sink(StrSpan)` inside
        // one scratch window: an existence disjunct yields its compiled
        // definition body's witness facts with the witness slot in
        // marker form (body instantiated in u_-form so the witness is
        // the only non-u_ token — the listLastRemovedArgsLE
        // discriminator, mirroring the real existence consumption); an
        // operator-application disjunct yields itself; equalities (no
        // writer can produce a matching demand), negated disjuncts
        // (deferred by design), and non-operator cores yield nothing.
        const auto forEachProductTemplate = [&](int32_t li, auto&& sink) {
            const StrSpan cleanLeaf = cleanLeaves[li];
            if (cleanLeaf.len > 0 && cleanLeaf.ptr[0] == '!') return;
            if (canonicalizer.isEquality(cleanLeaf)) return;
            const StrSpan core = extractExpressionSpan(cleanLeaf);
            const LogicalEntity* le = this->definitions.compiledEntity(core);
            if (le != nullptr && le->category == "existence") {
                ScratchScope tmplScope(sArena);
                StrSpan sigArgs[ExecutionParameters::MAX_ARITY];
                const int32_t sigN = getArgsSpans(StrSpan(le->signature),
                    sigArgs, ExecutionParameters::MAX_ARITY);
                StrSpan instArgs[ExecutionParameters::MAX_ARITY];
                const int32_t instN = getArgsSpans(orLeaves[li], instArgs,
                    ExecutionParameters::MAX_ARITY);
                assert(sigN == instN
                    && "ordis product template: existence instance arity "
                       "differs from the compiled definition");
                StrReplacement pairs[ExecutionParameters::MAX_ARITY];
                for (int32_t a = 0; a < sigN; ++a) {
                    pairs[a].key = sigArgs[a];
                    pairs[a].value = instArgs[a];
                }
                ScratchString bodyHold[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
                StrSpan bodyElems[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
                int32_t bodyN = 0;
                for (const std::string& rawElem : le->elements) {
                    assert(bodyN < ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
                        && "ordis product template: body element count "
                           "exceeds cap");
                    bodyHold[bodyN] = replaceKeysScratch(sArena,
                        StrSpan(rawElem), pairs, sigN);
                    bodyElems[bodyN] = StrSpan(bodyHold[bodyN]);
                    ++bodyN;
                }
                StrSpan witnesses[ExecutionParameters::MAX_ARITY];
                const int32_t wN = gl::DefinitionExpansion::listLastRemovedArgsLE(
                    StrSpan("existence", 9), bodyElems, bodyN,
                    witnesses, ExecutionParameters::MAX_ARITY);
                assert(wN == 1
                    && "ordis product template: existence definition must "
                       "bind exactly one witness");
                for (int32_t b = 0; b < bodyN; ++b) {
                    const StrSpan bCore = extractExpressionSpan(bodyElems[b]);
                    if (this->definitions.operatorNames().find(std::string_view(bCore.ptr,
                            static_cast<std::size_t>(bCore.len)))
                        == this->definitions.operatorNames().end()) {
                        continue;
                    }
                    StrSpan bArgs[ExecutionParameters::MAX_ARITY];
                    const int32_t bArgsN = getArgsSpans(bodyElems[b],
                        bArgs, ExecutionParameters::MAX_ARITY);
                    bool hasWitness = false;
                    for (int32_t a = 0; a < bArgsN; ++a) {
                        if (equalSpans(bArgs[a], witnesses[0])) {
                            hasWitness = true;
                            break;
                        }
                    }
                    if (!hasWitness) continue;
                    const ScratchString markedU = gl::DefinitionExpansion::makeMarkedExprScratch(
                        sArena, bodyElems[b], witnesses[0]);
                    const ScratchString tmpl =
                        gl::DefinitionExpansion::removeUPrefixScratch(sArena, StrSpan(markedU));
                    sink(StrSpan(tmpl));
                }
            } else if (this->definitions.operatorNames().find(std::string_view(core.ptr,
                           static_cast<std::size_t>(core.len)))
                       != this->definitions.operatorNames().end()) {
                sink(cleanLeaf);
            }
        };

        const bool routeB = allowOrDisintegration;
        bool leafAdmitted[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS] = {};
        if (!routeB && allowOrProbe) {
            for (int32_t i = 0; i < orLeafN; ++i) {
                forEachProductTemplate(i, [&](StrSpan tmpl) {
                    if (leafAdmitted[i]) return;
                    // isAdmitted's prologue, read-only: non-minting
                    // template probe + key presence. No updateAdmissionMap,
                    // no status write.
                    if (canonicalTemplates.canonicalTemplatePresent(memoryBlock, memoryBlock.overallHashMemory.admissionMap,
                            memoryBlock.lbMemory.canonicalUses.algebraMatches, tmpl, validityName)) {
                        leafAdmitted[i] = true;
                    }
                });
            }
        }
        // Route (c) — ordis2 demand (D-267):
        // per-leaf GROUND probe of admissionMapOrdis2 ONLY (never the
        // algebra map — the compact-only branch-seed property depends on
        // this path writing/reading no admission entry). A demand key's
        // template half is a ground premise text, so the clean leaf is the
        // exact probe (ground-to-ground byte equality through the template
        // interner). Non-minting. Placed BEFORE the park in this chain, so
        // the demand-first arrival order opens without ever parking.
        bool demandHit = false;
        bool hypothesisDemandHit = false;
        bool leafDemand[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS] = {};
        const unsigned demandSlot = gl::WorkerContext::coreId() >= 0 ? static_cast<unsigned>(gl::WorkerContext::coreId())
            : genScratchArenas().slotCount() - 1;
        ScratchArena& demandArena = genScratchArenas().forSlot(demandSlot);
        DirtyState demandDirty = DirtyState::Clean;
        if (!routeB && allowOrProbe
            && !memoryBlock.overallHashMemory.admissionMapOrdis2.empty()) {
            for (int32_t i = 0; i < orLeafN; ++i) {
                PagedVector<int64_t> matchedDemands(&demandArena, &demandDirty);
                const NameId target = memoryBlock.nameMap.lookup(validityName);
                canonicalTemplates.matchingCanonicalTemplates(memoryBlock, memoryBlock.overallHashMemory.admissionMapOrdis2,
                    memoryBlock.lbMemory.canonicalUses.ordis2Matches, cleanLeaves[i], target, target,
                    matchedDemands, demandArena);
                if (!matchedDemands.empty()) {
                    // "Really missing" gate at branch creation
                    // (D-358): a demanded
                    // leaf already known at this scope or an ancestor is a
                    // satisfied demand — the fact the rule waited for
                    // exists, the full rule fires on it — so it opens
                    // nothing; its stale entry is consumed here. Shared
                    // ancestorKnown predicate, non-minting lookups (an
                    // un-interned leaf text cannot be known).
                    const NameId leafOrigId =
                        memoryBlock.nameMap.lookup(cleanLeaves[i]);
                    const NameId leafValId =
                        memoryBlock.nameMap.lookup(validityName);
                    if (leafOrigId != 0 && leafValId != 0
                        && gl::StatementKnowledge::ancestorKnown(memoryBlock, leafOrigId, leafValId,
                                         /*includeSelf=*/true)) {
                        memoryBlock.overallHashMemory.admissionMapOrdis2
                            .eraseBlobIf([&](int64_t key) {
                                for (int32_t j = 0; j < matchedDemands.size(); ++j)
                                    if (key == matchedDemands[j]) return true;
                                return false;
                            });
                        continue;
                    }
                    leafDemand[i] = true;
                    demandHit = true;
                    // The induction hypothesis's missing expression marks the
                    // cohort for release below an unsettled induction scope
                    // (I-328 exemption, D-575).
                    hypothesisDemandHit = hypothesisDemandHit
                        || inductionHypothesisDemanded(memoryBlock, matchedDemands, demandArena);
                }
            }
        }
        // Explicit goals and active primed contradiction blocks can require
        // cases. Flat rules remain unconditional when no proof work remains.
        const bool hasProofWork = hasOrProofWork(memoryBlock);
        // Satisfied-OR gate (D-438): a disjunct that
        // already holds at this scope makes the case split useless — its
        // branch equals this scope, so convergence could only conclude what
        // the scope already proves. Such an OR opens no case split by any
        // route and is not parked either: a disjunct that holds keeps holding
        // (a scope's facts only accumulate and its classes only merge), so a
        // parked satisfied OR could never open. The probes above keep their
        // own consumption behavior.
        bool orSatisfied = false;
        if (hasProofWork && (routeB || allowOrProbe)) {
            const NameId orScopeId = memoryBlock.nameMap.lookup(validityName);
            assert(orScopeId != 0
                && "satisfied-OR gate: the OR's scope must be interned");
            for (int32_t i = 0; i < orLeafN && !orSatisfied; ++i)
                orSatisfied = orDisjunctTrueAt(cleanLeaves[i], orScopeId, memoryBlock);
        }
        // Algebra product-template matches do not authorize case creation.
        const bool orAdmitted = hasProofWork && !orSatisfied && (routeB || demandHit);
        if (orAdmitted) {
            // 3. Cohort identity + bootstrap guard. Equal signatures
            //    under different parents are independent case splits
            //    (I-167). An existing count row means this exact cohort
            //    is already scheduled (its branches live, pending,
            //    retired, or converged) — a re-deposit of the same or
            //    statement must NOT re-mint branch scopes or reset the
            //    release sequence; the K mutual-exclusion implications
            //    above were still (re-)emitted.
            const ScratchString orSignature = gl::DefinitionExpansion::removeUPrefixScratch(sArena, entSignature);  // e.g. "(or3[1,2,3])"
            const int32_t orParentId =
                memoryBlock.lbStateInterner.encode(validityName);
            const int32_t orSigId =
                memoryBlock.lbStateInterner.encode(StrSpan(orSignature));
            const int32_t orCohortId = mintOrCohortId(
                memoryBlock.lbStateInterner, orParentId, orSigId);
            if (memoryBlock.orDisjunctCount.lookup(orCohortId) == 0) {
                // Fresh cohort: register the FULL structural leaf count
                // (convergence semantics unchanged — dead-branch
                // retirement shrinks it in place, D-242), queue EVERY
                // leaf, and stage the cohort for release. The bootstrap
                // mints NO branch and sends NO mail: this code runs
                // inside standardProcessing's internal-mail drain, and
                // the drain's step-3 clear would wipe a seed inserted
                // into the channel being drained. The end-of-burst
                // drainPendingOrReleases — which runs AFTER that clear —
                // performs the first release exactly like every later
                // one (top-ranked disjunct, seed + origin on
                // sameIterationInternalMail, absorbed next step).
                memoryBlock.orDisjunctCount.insert(orCohortId, orLeafN);
                // A cohort of a conjecture premise's OR releases its cases as
                // assumptions disintegrated like a premise, with force_
                // witnesses (D-527).
                if (forcedAssumptions)
                    memoryBlock.lbMemory.forcedOrCohorts.mint(orCohortId);
                // A cohort opened by the induction hypothesis's missing
                // expression releases every case below an unsettled induction
                // scope: the hypothesis can fire only in its case, and only a
                // firing settles the tree (I-328 exemption,
                // D-575).
                if (hypothesisDemandHit)
                    memoryBlock.lbMemory.hypothesisOrCohorts.mint(orCohortId);

                // Route (a) / route (c) starter: the demand names WHICH
                // branch carries the relevance — the matched disjunct
                // opens first (ties among several matched fall to the
                // standing ranking). Demand WINS over admission when both
                // hit (maintainer decision): the demand-matched leaf is
                // the one a starved rule is waiting to consume. Recorded
                // as a one-shot side row consumed by
                // drainPendingOrReleases; route (b) writes no row and
                // starts at the top of the ranking.
                int32_t starterLeafIx = -1;
                if (!routeB) {
                    const bool* pick = demandHit ? leafDemand : leafAdmitted;
                    StrSpan admitted[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
                    int32_t admittedLeaf[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
                    int32_t admittedN = 0;
                    for (int32_t i = 0; i < orLeafN; ++i) {
                        if (pick[i]) {
                            admitted[admittedN] = cleanLeaves[i];
                            admittedLeaf[admittedN] = i;
                            ++admittedN;
                        }
                    }
                    assert(admittedN > 0
                        && "route (a)/(c) cohort opened without a matched "
                           "disjunct");
                    StrSpan anchorArgs[ExecutionParameters::MAX_ARITY];
                    const int32_t anchorArgN = collectAnchorArgs(
                        memoryBlock, anchorArgs,
                        ExecutionParameters::MAX_ARITY);
                    starterLeafIx = admittedLeaf[pickTopOrDisjunct(
                        admitted, admittedN, anchorArgs, anchorArgN)];
                }
                // A twin block's case experiment releases the successor case
                // first (D-588). A zero
                // block's cohort releases no other case. The weak step block's
                // cohort follows the ordinary order, so its zero case is
                // released once the successor case is resolved or retired;
                // the block records the cohort to find the successor case's
                // scope at its release.
                if (experimentLeaf >= 0) {
                    starterLeafIx = experimentLeaf;
                    if (memoryBlock.twinExperiment == TwinExperiment::successorCase) {
                        memoryBlock.lbMemory.experimentOrCohorts.mint(orCohortId);
                    } else {
                        assert(memoryBlock.twinExperiment == TwinExperiment::successorThenZero
                            && (memoryBlock.twinCohortId == 0 || memoryBlock.twinCohortId == orCohortId)
                            && "consumeOrLeavesCohort: a weak step block opens one twin cohort");
                        memoryBlock.twinCohortId = orCohortId;
                    }
                }

                // Opening a case split does not prove its demanded leaf.
                // Keep the shared demand for other eligible OR sources;
                // the probe removes it only when the fact is already known.
                // The cohort guard above prevents duplicate scheduling.

                // Queue each leaf in the wrapped payload-body form (the
                // orBookkeeping id convention). The clean strings ride
                // the enclosing scope's byte-bump tier (fresh-string
                // recipe: multiple spans under one scope).
                for (int32_t i = 0; i < orLeafN; ++i) {
                    const StrSpan cleanSpan = cleanLeaves[i];
                    const int32_t wLen = cleanSpan.len + 2;
                    char* wBuf = sArena.allocBytes(wLen);
                    wBuf[0] = '(';
                    if (cleanSpan.len > 0)
                        std::memcpy(wBuf + 1, cleanSpan.ptr,
                            static_cast<size_t>(cleanSpan.len));
                    wBuf[wLen - 1] = ')';
                    const int32_t wrappedId = memoryBlock.lbStateInterner
                        .encode(StrSpan(wBuf, wLen));
                    memoryBlock.orPendingBranches.insertSorted(orCohortId,
                        wrappedId,
                        DecodedIdLess{ &memoryBlock.lbStateInterner });
                    if (i == starterLeafIx) {
                        memoryBlock.orStarterPick.upsert(orCohortId,
                                                         wrappedId);
                    }
                }
                for (int32_t l = 0; l < orSeedLevelCount; ++l) {
                    memoryBlock.orPendingLevels.insertSorted(orCohortId,
                        orSeedLevels[l],
                        [](int32_t a, int32_t b) { return a < b; });
                }
                memoryBlock.pendingOrReleases.mint(orCohortId);
            }
        } else if (hasProofWork && allowOrProbe && !orSatisfied) {
            // 4. PARK: no route admitted — file the cohort under each
            //    operator-based disjunct product template at this
            //    validity. Value = the clean or statement + the seed
            //    level run (the complete reopening context; the key
            //    carries the validity). The RMW dedups re-parks. No
            //    park-time rendezvous is needed: the probe above just
            //    missed synchronously, and every later admission key
            //    gain passes a revisitRejectedOrdis seam.
            const ScratchString orSignature =
                gl::DefinitionExpansion::removeUPrefixScratch(sArena, entSignature);
            const int32_t parkStmtId =
                memoryBlock.valueInterner.encode(StrSpan(orSignature));
            const unsigned parkSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            ScratchArena& parkArena = genScratchArenas().forSlot(parkSlot);
            for (int32_t i = 0; i < orLeafN; ++i) {
                forEachProductTemplate(i, [&](StrSpan tmpl) {
                    const int64_t parkPk = mintTemplateKey(
                        memoryBlock.templateInterner, memoryBlock.nameMap,
                        tmpl, validityName);
                    const int32_t previousParkId = memoryBlock.overallHashMemory.rejectedMapOrdis.lookup(parkPk);
                    const int32_t previousParkCount = previousParkId == 0 ? 0 : memoryBlock.overallHashMemory.rejectedMapOrdis.runLen(previousParkId);
                    insertRejectedOrdisIdsBlob(
                        memoryBlock.overallHashMemory.rejectedMapOrdis,
                        parkPk, parkStmtId,
                        orSeedLevels, orSeedLevelCount,
                        memoryBlock.valueInterner, parkArena);
                    if (memoryBlock.overallHashMemory.rejectedMapOrdis.runLen(memoryBlock.overallHashMemory.rejectedMapOrdis.lookup(parkPk)) != previousParkCount)
                        memoryBlock.lbMemory.canonicalUses.ordisSources.changed(parkPk);
                });
            }
            // DUAL FILING (D-267): the same
            // cohort additionally files in rejectedMapOrdis2 under each
            // ELIGIBLE disjunct's clean GROUND text (polarity verbatim,
            // I-175 — a negated compound files WITH its '!', because
            // A15-family demands are minted negated) at this validity —
            // the demand map's key language, so the drain's wake is a
            // plain key rendezvous. The old filing above is untouched
            // (I-178); only drainAdmissionKeysOrdis2 wakes this index.
            for (int32_t i = 0; i < orLeafN; ++i) {
                if (!admissionPolicy.ordis2KeyEligible(cleanLeaves[i])) continue;
                const int64_t park2Pk = mintTemplateKey(
                    memoryBlock.templateInterner, memoryBlock.nameMap,
                    cleanLeaves[i], validityName);
                const int32_t previousParkId = memoryBlock.overallHashMemory.rejectedMapOrdis2.lookup(park2Pk);
                    const int32_t previousParkCount = previousParkId == 0 ? 0 : memoryBlock.overallHashMemory.rejectedMapOrdis2.runLen(previousParkId);
                    insertRejectedOrdisIdsBlob(
                    memoryBlock.overallHashMemory.rejectedMapOrdis2,
                    park2Pk, parkStmtId,
                    orSeedLevels, orSeedLevelCount,
                    memoryBlock.valueInterner, parkArena);
                    if (memoryBlock.overallHashMemory.rejectedMapOrdis2.runLen(memoryBlock.overallHashMemory.rejectedMapOrdis2.lookup(park2Pk)) != previousParkCount)
                        memoryBlock.lbMemory.canonicalUses.ordis2Sources.changed(park2Pk);
            }
        }
        // else: hypothetical-disintegration path — neither open nor park.
    }
    // else: max OR depth reached — K mutual-exclusion implications emitted above, no branch opening
}

/// @brief The leaf a twin block's case experiment releases first from an
///        OR, or -1 (D-588).
///
/// @details
/// A block armed for a case experiment (`Memory::twinExperiment`) opens the
/// zero-or-successor OR of its twin partner l at `main`: an OR of exactly two
/// leaves, the positive equality of l and the anchor's zero
/// (`Memory::twinZeroId`, either order), and a positive existence whose
/// compiled body holds `in2[w, l, s]` for its witness w and the anchor's
/// successor s (`Memory::twinSuccessorId`); FTA's `or1`, `inSequence[N,l,s]`
/// or `l = 0`. Both armed kinds release the existence leaf first: the weak
/// step block follows it with the zero leaf in the ordinary order, a zero
/// block releases nothing else. Any other OR, scope or block gives -1.
///
/// @param mb           The block whose OR is deposited.
/// @param orLeaves     The OR's flattened leaves, true polarity, `u_` form.
/// @param orLeafN      Number of leaves.
/// @param validityName The OR's scope.
/// @param sArena       String scratch for the cleaned and instantiated texts.
/// @return The index of the leaf to release, or -1.
/// @invariant Mints nothing; reads the compiled configuration and the block's
///            NameMap only.
/// @see consumeOrLeavesCohort, drainPendingOrReleases,
///      AdmissionMemory::updateAdmissionMap (arms the step block).
int32_t CaseProcessor::twinExperimentLeaf(const Memory& mb, const StrSpan* orLeaves, int32_t orLeafN,
    StrSpan validityName, ScratchArena& sArena) const {
    if (mb.twinExperiment == TwinExperiment::none) return -1;
    if (orLeafN != 2 || !equalSpans(validityName, StrSpan("main", 4))) return -1;
    assert(mb.twinPartnerId != 0 && mb.twinZeroId != 0 && mb.twinSuccessorId != 0
        && "twinExperimentLeaf: an armed block records its partner, zero and successor");
    const StrSpan partner = mb.nameMap.decodeView(mb.twinPartnerId);
    const StrSpan zero = mb.nameMap.decodeView(mb.twinZeroId);
    const StrSpan successor = mb.nameMap.decodeView(mb.twinSuccessorId);

    ScratchScope frame(sArena);
    int32_t zeroLeaf = -1;
    int32_t successorLeaf = -1;
    for (int32_t i = 0; i < orLeafN; ++i) {
        const ScratchString clean = gl::DefinitionExpansion::removeUPrefixScratch(sArena, orLeaves[i]);
        const StrSpan leaf(clean);
        if (leaf.len == 0 || leaf.ptr[0] == '!') continue;
        StrSpan args[ExecutionParameters::MAX_ARITY];
        const int32_t argN = getArgsSpans(leaf, args, ExecutionParameters::MAX_ARITY);
        if (canonicalizer.isEquality(leaf)) {
            assert(argN == 2 && "twinExperimentLeaf: an equality has two arguments");
            if ((equalSpans(args[0], partner) && equalSpans(args[1], zero))
                || (equalSpans(args[0], zero) && equalSpans(args[1], partner)))
                zeroLeaf = i;
            continue;
        }
        // The existence leaf: its compiled body instantiated in u_ form, so
        // the witness is the only token without the prefix.
        const LogicalEntity* le = this->definitions.compiledEntity(extractExpressionSpan(leaf));
        if (le == nullptr || le->category != "existence") continue;
        StrSpan sigArgs[ExecutionParameters::MAX_ARITY];
        const int32_t sigN = getArgsSpans(StrSpan(le->signature), sigArgs, ExecutionParameters::MAX_ARITY);
        StrSpan instArgs[ExecutionParameters::MAX_ARITY];
        const int32_t instN = getArgsSpans(orLeaves[i], instArgs, ExecutionParameters::MAX_ARITY);
        assert(sigN == instN && "twinExperimentLeaf: an existence instance has its definition's arity");
        StrReplacement pairs[ExecutionParameters::MAX_ARITY];
        for (int32_t a = 0; a < sigN; ++a) {
            pairs[a].key = sigArgs[a];
            pairs[a].value = instArgs[a];
        }
        ScratchString bodyHold[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
        StrSpan body[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
        int32_t bodyN = 0;
        for (const std::string& element : le->elements) {
            assert(bodyN < ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
                && "twinExperimentLeaf: an existence body exceeds the element cap");
            bodyHold[bodyN] = replaceKeysScratch(sArena, StrSpan(element), pairs, sigN);
            body[bodyN] = StrSpan(bodyHold[bodyN]);
            ++bodyN;
        }
        StrSpan witnesses[ExecutionParameters::MAX_ARITY];
        const int32_t witnessN = gl::DefinitionExpansion::listLastRemovedArgsLE(
            StrSpan("existence", 9), body, bodyN, witnesses, ExecutionParameters::MAX_ARITY);
        if (witnessN != 1) continue;
        for (int32_t b = 0; b < bodyN; ++b) {
            if (!equalSpans(extractExpressionSpan(body[b]), StrSpan("in2", 3))) continue;
            StrSpan in2Args[ExecutionParameters::MAX_ARITY];
            const int32_t in2N = getArgsSpans(body[b], in2Args, ExecutionParameters::MAX_ARITY);
            if (in2N != 3 || !equalSpans(in2Args[0], witnesses[0])) continue;
            const StrSpan output = startsWithSpan(in2Args[1], "u_", 2)
                ? StrSpan(in2Args[1].ptr + 2, in2Args[1].len - 2) : in2Args[1];
            const StrSpan function = startsWithSpan(in2Args[2], "u_", 2)
                ? StrSpan(in2Args[2].ptr + 2, in2Args[2].len - 2) : in2Args[2];
            if (equalSpans(output, partner) && equalSpans(function, successor)) successorLeaf = i;
        }
    }
    if (zeroLeaf < 0 || successorLeaf < 0) return -1;
    return successorLeaf;
}
}
