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
#include "../proof/or_candidate_progress.hpp"
#include "theorem_publisher.hpp"
namespace gl {
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
TheoremPublisher::TheoremPublisher(AnchorScopeQueries& anchorScopeQueries, Memory& body, DefinitionExpansion& definitionExpansion, const CompiledDefinitionRegistry& definitions, std::mutex& dependenciesMutex, std::vector< std::tuple<std::string, std::string, std::string, std::string >>& fullTheoremList, Dependencies& globalDependencies, std::vector< std::tuple<std::string, std::string, std::string, std::string >>& globalTheoremList, std::vector<const Memory*>& globalTheoremProducers, std::unordered_set<std::string>& globalTheoremStrings, std::unordered_set<std::string>& orInRunScannedRows, const ProverParameters& parameters, std::mutex& pendingCompactionMutex, std::vector<std::tuple<std::string, int, int>>& pendingCompactionQueue, ProofHistory& proofHistory, std::vector<std::string>& retractedVacuousTheorems, ScopeLifecycle& scopeLifecycle, SimpleMapStore& simpleMapStore, std::unique_ptr<MemorySteward>& steward, StructuralCompiler& structuralCompiler, std::mutex& theoremListMutex, std::optional<SealedPageSet>& updateGlobalSingleChapterPages, OrCandidateProgress& orCandidateProgress, std::unordered_set<std::string>& orInRunCompanions) : anchorScopeQueries(anchorScopeQueries), body(body), definitionExpansion(definitionExpansion), definitions(definitions), dependenciesMutex(dependenciesMutex), fullTheoremList(fullTheoremList), globalDependencies(globalDependencies), globalTheoremList(globalTheoremList), globalTheoremProducers(globalTheoremProducers), globalTheoremStrings(globalTheoremStrings), orInRunScannedRows(orInRunScannedRows), parameters(parameters), pendingCompactionMutex(pendingCompactionMutex), pendingCompactionQueue(pendingCompactionQueue), proofHistory(proofHistory), retractedVacuousTheorems(retractedVacuousTheorems), scopeLifecycle(scopeLifecycle), simpleMapStore(simpleMapStore), steward(steward), structuralCompiler(structuralCompiler), theoremListMutex(theoremListMutex), updateGlobalSingleChapterPages(updateGlobalSingleChapterPages), orCandidateProgress(orCandidateProgress) , orInRunCompanions(orInRunCompanions) {}

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
Memory* TheoremPublisher::accessMemory(const std::vector<std::string>& theoremKey,
    Memory& bodyOfProves1) {
    Memory* memoryBlock = &bodyOfProves1;
    for (std::size_t i = 0; i < theoremKey.size(); ++i) {
        const std::string& key = theoremKey[i];
        Memory* child = simpleMapStore.findChild(memoryBlock, key);
        if (child == NULL) {
            return NULL;
        }
        memoryBlock = child; // follow the pointer
    }
    return memoryBlock;
}

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
bool TheoremPublisher::tryBackReformulateOperatorHead(
    const std::string& theorem,
    std::string& backReformulated) {
    if (!parameters.incubator_mode) return false;

    std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> tempChain;
    std::string head = ce::disintegrateImplication(theorem, tempChain, this->definitions.configurations());

    // Must have exactly 2 chain elements (anchor + operator)
    if (tempChain.size() != 2) return false;

    // First chain element must be an Anchor
    {
        std::string firstName = ce::extractExpression(std::get<0>(tempChain[0]));
        if (firstName.find("Anchor") == std::string::npos) return false;
    }

    // Head must be equality or negated equality
    bool negated = false;
    std::string eqExpr = head;
    if (!head.empty() && head[0] == '!') {
        negated = true;
        eqExpr = head.substr(1);
    }

    std::string headExpr = ce::extractExpression(eqExpr);
    if (headExpr != "=") return false;

    std::vector<std::string> eqArgs = ce::getArgs(eqExpr);
    if (eqArgs.size() != 2) return false;

    // Last chain element must be an operator
    const std::string& lastPremise = std::get<0>(tempChain.back());
    std::string opName = ce::extractExpression(lastPremise);
    const ce::CoreExpressionConfig* cfg = this->definitions.coreConfig(StrSpan(opName));
    if (cfg == nullptr) return false;
    if (cfg->outputIndices.empty()) return false;  // not an operator

    int outIdx = cfg->outputIndices[0];  // 0-based
    std::vector<std::string> opArgs = ce::getArgs(lastPremise);
    if (outIdx >= static_cast<int>(opArgs.size())) return false;

    std::string x = opArgs[outIdx];  // the fresh variable in operator

    // Verify: one of the equality args must be x
    std::string a_out;
    if (eqArgs[0] == x) {
        a_out = eqArgs[1];
    } else if (eqArgs[1] == x) {
        a_out = eqArgs[0];
    } else {
        return false;  // x doesn't appear in equality — not our pattern
    }

    // Back-reformulate: replace x with a_out in the operator, drop the equality + inner >
    opArgs[outIdx] = a_out;
    std::string cleanHead = "(" + opName + "[";
    for (size_t i = 0; i < opArgs.size(); ++i) {
        if (i > 0) cleanHead += ",";
        cleanHead += opArgs[i];
    }
    cleanHead += "])";

    // Negate the operator head if the equality was negated
    if (negated) {
        cleanHead = "!" + cleanHead;
    }

    // Rebuild chain without the last element (operator was a premise, now it's the head)
    std::vector<std::string> outerChain;
    for (size_t i = 0; i < tempChain.size() - 1; ++i) {
        outerChain.push_back(std::get<0>(tempChain[i]));
    }

    backReformulated = gl::ImplicationBuilder::reconstructImplication(outerChain, cleanHead);
    return true;
}

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
bool TheoremPublisher::appendGlobalTheorem(const std::string& theorem,
    const std::string& method, const std::string& aux2,
    const std::string& aux3, const Memory* producer) {
    std::lock_guard<std::mutex> lock(this->theoremListMutex);
    if (!this->globalTheoremStrings.insert(theorem).second) {
        // Method-aware upgrade: a first-class registration REPLACES an
        // existing proved-not-broadcast row in place. The tier records a
        // closure that never circulated, so a level-complete (or
        // constructed) derivation of the same theorem supersedes it; every
        // other duplicate is dropped (first emission wins), and a tier
        // arrival never demotes an existing row of any method.
        if (method == "proved not broadcast") return false;
        for (std::size_t i = 0; i < this->globalTheoremList.size(); ++i) {
            auto& tpl = this->globalTheoremList[i];
            if (std::get<0>(tpl) != theorem) continue;
            if (std::get<1>(tpl) != "proved not broadcast") return false;
            std::get<1>(tpl) = method;
            std::get<2>(tpl) = aux2;
            std::get<3>(tpl) = aux3;
            this->globalTheoremProducers[i] = producer;
            // The in-run or-construction scan marks a tier row scanned when
            // it skips it; the upgraded row is first-class and must be
            // scannable as new, so the settled mark leaves with the tier
            // method.
            this->orInRunScannedRows.erase(theorem);
            // The tier registration mirrored its chapter row into
            // fullTheoremList; the upgraded row re-mirrors through the
            // first-class path, so the tier mirror leaves.
            for (std::size_t f = this->fullTheoremList.size(); f-- > 0; ) {
                if (std::get<0>(this->fullTheoremList[f]) == theorem
                    && std::get<1>(this->fullTheoremList[f])
                           == "proved not broadcast") {
                    this->fullTheoremList.erase(this->fullTheoremList.begin()
                        + static_cast<std::ptrdiff_t>(f));
                }
            }
            return true;
        }
        assert(false
            && "appendGlobalTheorem: globalTheoremStrings out of step with "
               "globalTheoremList");
        return false;
    }
    // A registered theorem is closed: every variable has a binder (I-4).
    {
        ce::TreeNode1* root = ce::parseExpr(theorem);
        assert(root != nullptr && root->arguments.empty()
            && "appendGlobalTheorem: a registered theorem has an unbound variable");
        ce::deleteTree(root);
    }
    this->globalTheoremList.emplace_back(theorem, method, aux2, aux3);
    this->globalTheoremProducers.push_back(producer);
    return true;
}

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
void TheoremPublisher::revertVacuousGlobalTheorems() {
    std::lock_guard<std::mutex> lock(this->theoremListMutex);
    assert(this->globalTheoremProducers.size()
               == this->globalTheoremList.size()
        && "producer attribution out of step with globalTheoremList");
    for (std::size_t i = this->globalTheoremList.size(); i-- > 0; ) {
        const Memory* producer = this->globalTheoremProducers[i];
        if (producer == nullptr || !gl::StatementKnowledge::hasContradictedAncestor(producer)) continue;
        const std::string theorem = std::get<0>(this->globalTheoremList[i]);
        std::cout << "[VACUOUS-REVERT] " << theorem << std::endl;
        this->retractedVacuousTheorems.push_back(theorem);
        this->globalTheoremStrings.erase(theorem);
        this->globalTheoremList.erase(this->globalTheoremList.begin()
            + static_cast<std::ptrdiff_t>(i));
        this->globalTheoremProducers.erase(this->globalTheoremProducers.begin()
            + static_cast<std::ptrdiff_t>(i));
        for (std::size_t j = this->fullTheoremList.size(); j-- > 0; ) {
            if (std::get<0>(this->fullTheoremList[j]) == theorem) {
                this->fullTheoremList.erase(this->fullTheoremList.begin()
                    + static_cast<std::ptrdiff_t>(j));
            }
        }
    }
}

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
std::vector<std::string> TheoremPublisher::classifyVacuousPremisePairs() {
    std::lock_guard<std::mutex> lock(this->theoremListMutex);
    assert(this->globalTheoremProducers.size()
               == this->globalTheoremList.size()
        && "producer attribution out of step with globalTheoremList");
    struct ChainGroup {
        std::unordered_set<std::string> heads;
        std::vector<std::size_t> indices;
        std::vector<std::string> headByIndex;
    };
    std::map<std::string, ChainGroup> byChain;
    for (std::size_t i = 0; i < this->globalTheoremList.size(); ++i) {
        const std::string& thm = std::get<0>(this->globalTheoremList[i]);
        std::vector<std::tuple<std::string, std::vector<std::string>,
                               std::set<std::string>>> tempChain;
        const std::string head = ce::disintegrateImplication(
            thm, tempChain, this->definitions.configurations());
        std::vector<std::string> ky;
        ky.reserve(tempChain.size());
        for (const auto& el : tempChain) ky.push_back(std::get<0>(el));
        std::sort(ky.begin(), ky.end());
        std::string key;
        for (const auto& el : ky) { key += el; key += '\x1f'; }
        ChainGroup& g = byChain[key];
        g.heads.insert(head);
        g.indices.push_back(i);
        g.headByIndex.push_back(head);
    }
    std::vector<std::size_t> vacIdx;
    for (const auto& kv : byChain) {
        const ChainGroup& g = kv.second;
        if (g.indices.size() < 2) continue;
        bool contradictory = false;
        for (const std::string& h : g.headByIndex) {
            if (g.heads.count(gl::DefinitionExpansion::negate(h)) != 0) { contradictory = true; break; }
        }
        if (!contradictory) continue;
        vacIdx.insert(vacIdx.end(), g.indices.begin(), g.indices.end());
    }
    std::sort(vacIdx.begin(), vacIdx.end());
    std::vector<std::string> classified;
    classified.reserve(vacIdx.size());
    for (const std::size_t i : vacIdx)
        classified.push_back(std::get<0>(this->globalTheoremList[i]));
    for (std::size_t j = vacIdx.size(); j-- > 0; ) {
        const std::size_t i = vacIdx[j];
        const std::string& theorem = classified[j];
        std::cout << "[VACUOUS-PREMISE-PAIR] " << theorem << std::endl;
        this->globalTheoremStrings.erase(theorem);
        this->globalTheoremList.erase(this->globalTheoremList.begin()
            + static_cast<std::ptrdiff_t>(i));
        this->globalTheoremProducers.erase(this->globalTheoremProducers.begin()
            + static_cast<std::ptrdiff_t>(i));
        for (std::size_t f = this->fullTheoremList.size(); f-- > 0; ) {
            if (std::get<0>(this->fullTheoremList[f]) == theorem) {
                this->fullTheoremList.erase(this->fullTheoremList.begin()
                    + static_cast<std::ptrdiff_t>(f));
            }
        }
    }
    return classified;
}

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
///            completion on the write side (I-44).
/// @see `updateGlobalSingleChapter` — sole caller (the tier branch).
void TheoremPublisher::repairTierCitationOrigins(Memory* producer) {
    assert(producer && "repairTierCitationOrigins: tier registration without a producer LB");
    const int maxOrigins = parameters.compressor_mode
        ? parameters.compressor_max_origins_per_expr
        : parameters.max_origin_per_expr;
    for (Memory* mb = producer; mb != nullptr; mb = mb->parentMemory) {
        if (steward) {
            steward->claimAndLoadForWork(*mb, /*phase=*/4,
                                         lbdeload::kDeloadDirectory);
        } else {
            mb->ensureLoadedForRead(lbdeload::kDeloadDirectory);
        }

        std::set<std::string> pendingDeposits;
        {
            const auto rows = decodeOriginMapSorted(mb->exprOriginMap,
                                                    mb->originInterner);
            for (const auto& row : rows) {
                for (const auto& line : row.second) {
                    for (const ExpressionWithValidity& dep : line.second) {
                        if (dep.validityName != "main") continue;
                        if (!gl::startsWith(dep.original, "(>[", 3)) continue;
                        if (this->globalTheoremStrings.count(dep.original) == 0) continue;
                        int64_t pk = 0;
                        if (lookupOriginKey(mb->originInterner, dep.original,
                                            "main", pk)) {
                            const int32_t oid = mb->exprOriginMap.lookup(pk);
                            if (oid != 0 && mb->exprOriginMap.runLen(oid) > 0) {
                                continue;   // already documented here
                            }
                        }
                        pendingDeposits.insert(dep.original);
                    }
                }
            }
        }
        for (const std::string& thm : pendingDeposits) {
            addOriginEncoded(mb->exprOriginMap, mb->originInterner,
                ExpressionWithValidity(thm, "main"),
                std::make_pair(std::string("theorem"),
                               std::vector<ExpressionWithValidity>()),
                maxOrigins);
        }

        if (steward) {
            mb->stewardClaim.store(
                static_cast<uint8_t>(Memory::StewardClaim::Idle),
                std::memory_order_release);
        }
    }
}

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
void TheoremPublisher::updateGlobalSingleChapter(const std::string& theorem, int coreId,
    const Memory* producer, bool registerGlobally) {
    // 1) Disintegrate to chain + head
    std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> tempChain;
    Mail mailOut;

    scopeLifecycle.deactivateUnnecessary(theorem, this->body);  // deactivates all blocks tied to this theorem.

    std::string value = ce::disintegrateImplication(theorem, tempChain, definitions.configurations());

    std::vector<std::string> ky;
    ky.reserve(tempChain.size());
    for (std::size_t i = 0; i < tempChain.size(); ++i) ky.push_back(std::get<0>(tempChain[i]));

    // The opposite-polarity conjecture (same chain, negated head) is settled
    // by this theorem too: a verbatim contradiction LB's disproof settles the
    // ORIGINAL conjecture, and a proved head makes the opposite-head
    // conjecture false — either way its recursion auxiliaries are pointless.
    // Retire them through the same matching walk; a twin form that matches
    // no originalAuxyMap entry is a defined no-op.
    scopeLifecycle.deactivateUnnecessary(gl::ImplicationBuilder::reconstructImplication(ky, gl::DefinitionExpansion::negate(value)), this->body);

    // The head-switched companion of an or theorem built in-run is subsumed
    // by it, and its proof may rest on it: the or theorem's row names the
    // companion as its second parent, so registering the companion would
    // make the two chapters cite each other. The goal lifecycle above has
    // run; nothing is registered, broadcast or recorded in either tier.
    if (orInRunCompanions.count(theorem) != 0) {
        std::cout << "Withheld, companion of an in-run or theorem: " << theorem << std::endl;
        return;
    }

    // Level-refused registration (D-278) — proved-but-not-broadcast tier:
    // the goal lifecycle above ran level-free; every CIRCULATION surface
    // below — reformulation, broadcast, compaction staging, mail merge —
    // stays refused when the sealed verdict says the proof did not consume
    // every premise level (the theorem's stronger form is the fact that
    // should circulate). But the closure IS a sound proof, so the theorem
    // is RECORDED: a global-list row with method "proved not broadcast"
    // and its own chapter. It never travels as a rule and never feeds the
    // or construction; its one downstream consumer is the pre-split merge,
    // which may cite it as a guard variant. Excluded from theorems.txt.
    if (!registerGlobally) {
        // Compressor-mode re-derivation audits FIRST-CLASS registration
        // only — its theorem lists must stay byte-identical to the run it
        // audits — so the tier registration is a defined no-op there,
        // exactly like both or-construction seams.
        if (!parameters.compressor_mode) {
            if (appendGlobalTheorem(theorem, "proved not broadcast", "-1", "-1", producer)) {
                this->fullTheoremList.emplace_back(theorem, "proved not broadcast", "-1", "-1");
                std::cout << "Proved, not broadcast (level-refused registration): "
                          << theorem << std::endl;
                // The tier chapter walks this LB chain's real derivation;
                // complete any theorem citation whose broadcast history
                // line will never land (the LB dies in this window).
                repairTierCitationOrigins(const_cast<Memory*>(producer));
            }
        }
        return;
    }

    // Save original decomposition before sections 3/4 overwrite ky/value
    const std::vector<std::string> origKy = ky;
    const std::string origValue = value;

    // 2) Original implication broadcast deferred until after back-reformulation check (section 5)
    //    If back-reformulation succeeds, only the back-reformulated version is broadcast.
    bool willBackReformulate = false;

    std::vector<std::pair<std::string, std::string>> outTheorems;
    if (reformulateTheorem(theorem, outTheorems)) {
        for (const auto& pair : outTheorems) {
            const std::string& refTheoremRaw = pair.first;
            const std::string& refTheoremCompiled = pair.second;

            // One run-log line per registered theorem: a second producer of
            // the same text (a twin block, a later burst) registers nothing
            // and prints nothing.
            if (appendGlobalTheorem(refTheoremCompiled, "reformulated statement", theorem, "-1", producer)) {
                std::cout << refTheoremCompiled << std::endl;
            }

            if (parameters.trackHistory) {
                ExpressionWithValidity ev(refTheoremCompiled, "main");
                proofHistory.addOrigin(mailOut.exprOriginMap, ev, std::make_pair("theorem", std::vector<ExpressionWithValidity>()), (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            }

            tempChain.clear();
            std::string valueR = ce::disintegrateImplication(refTheoremCompiled, tempChain, definitions.configurations());

            ky.clear();
            ky.reserve(tempChain.size());
            for (std::size_t i = 0; i < tempChain.size(); ++i) ky.push_back(std::get<0>(tempChain[i]));

            // No settled-head deposit at the reformulated chain LB: the
            // compact broadcast below reaches it through root mail, the rule
            // installs, and the head re-derives from the LB's own seed
            // premises with residence-honest levels {0..chain-1}. A direct
            // deposit would have to fabricate a level run — the retired
            // {0..ky.size()} stamp poisoned the receiving LB's level
            // arithmetic (one past its level universe) and broke the
            // allLevelsInvolved verdict of every statement derived from it
            // (D-324).

            // ASIC 0.1 reshuffle: the legacy Mail::implications channel is
            // removed; this implication now travels SOLELY as the D-76
            // compact (implication<N>[...]) statement (deferred compile +
            // deposit, drained single-threaded post-pool.join — parallel
            // worker path, so never inline; I-28). Linked to the original
            // by a `compilation` origin (also the receiver's mandatory
            // paired-origin; I-44: exprOriginMap is documentation, not a
            // proof input).
            recordPendingCompaction(refTheoremCompiled, static_cast<int>(ky.size()), coreId);

        }

    }

    // 5) Incubator: back-reformulate operator-equality pattern
    // Must be before the mail merge so the back-reformulated implication ships.
    std::string backReformulated;
    willBackReformulate = tryBackReformulateOperatorHead(theorem, backReformulated);
    if (!willBackReformulate) {
        // No back-reformulation: record original theorem. ASIC 0.1
        // reshuffle: it travels SOLELY as the D-76 compact statement
        // (legacy Mail::implications removed); deferred compile+deposit.
        recordPendingCompaction(theorem, static_cast<int>(origKy.size()), coreId);
        if (parameters.trackHistory) {
            ExpressionWithValidity ev(theorem, "main");
            proofHistory.addOrigin(mailOut.exprOriginMap, ev, std::make_pair("theorem", std::vector<ExpressionWithValidity>()), (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
        }
        if (appendGlobalTheorem(theorem, "single chapter proof", "-1", "-1", producer)) {
            std::cout << theorem << std::endl;
        }
        orCandidateProgress.checkOrCompletion(theorem, coreId);
    }
    if (willBackReformulate) {
        // Register the original for one complete proof chapter.
        // ASIC 0.1 reshuffle: it travels SOLELY as the D-76 compact
        // statement (legacy Mail::implications removed); deferred deposit.
        recordPendingCompaction(theorem, static_cast<int>(origKy.size()), coreId);
        if (parameters.trackHistory) {
            ExpressionWithValidity ev(theorem, "main");
            proofHistory.addOrigin(mailOut.exprOriginMap, ev, std::make_pair("theorem", std::vector<ExpressionWithValidity>()), (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
        }
        if (appendGlobalTheorem(theorem, "single chapter proof", "-1", "-1", producer)) {
            std::cout << theorem << std::endl;
        }
        if (appendGlobalTheorem(
                backReformulated, "incubator back reformulation", theorem, "-1",
                producer)) {
            std::cout << backReformulated << std::endl;
        }
        orCandidateProgress.checkOrCompletion(theorem, coreId);

        if (parameters.trackHistory) {
            ExpressionWithValidity ev(backReformulated, "main");
            proofHistory.addOrigin(mailOut.exprOriginMap, ev,
                std::make_pair("theorem", std::vector<ExpressionWithValidity>()),
                (parameters.compressor_mode
                    ? parameters.compressor_max_origins_per_expr
                    : parameters.max_origin_per_expr));
        }

        std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> brChain;
        std::string brHead = ce::disintegrateImplication(backReformulated, brChain, definitions.configurations());

        std::vector<std::string> brKey;
        brKey.reserve(brChain.size());
        for (std::size_t i = 0; i < brChain.size(); ++i)
            brKey.push_back(std::get<0>(brChain[i]));

        // ASIC 0.1 reshuffle: back-reformulated implication travels SOLELY
        // as the D-76 compact statement (legacy Mail::implications
        // removed); deferred compile+deposit, single-threaded post-join.
        recordPendingCompaction(backReformulated, static_cast<int>(brKey.size()), coreId);
    }

    // 6) Flush mail. New mail system (D-137): this local mailOut
    // carries only "theorem" / "reformulated statement" origin lines
    // (documentation, I-44 -- the proof-affecting statements ride
    // nextIterationInternalMail). Merge them into the root's mailOut; the NEXT
    // commit barrier ships them into the root's log for descendants to pull,
    // preserving the old post-smashMail one-iteration-later delivery. Runs
    // single-threaded post-join. The root's mailOut is deloadable, so the seam
    // claims and reloads it across the deposit.
    assert(coreId >= 0);
    steward->claimAndLoadForWork(this->body, /*phase=*/4,
                                 lbdeload::kDeloadDirectory);
    mergeBatchIntoMailOut(mailOut, this->body);
    this->body.stewardClaim.store(
        static_cast<uint8_t>(Memory::StewardClaim::Idle),
        std::memory_order_release);
}

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
void TheoremPublisher::updateGlobal(int auxyIndex, uint64_t levelsMask, int coreId) {
    // ---- dependency bookkeeping (short critical section) ----
    int originalIndex = -1;
    std::string expr;
    std::string indVar;
    std::string recCounter;
    bool strongInduction = false;
    bool triadClosed = false;
    uint64_t levelsUsed = 0;

    {
        std::lock_guard<std::mutex> lock(this->dependenciesMutex);

        // Resolve original and update its bookkeeping
        originalIndex = this->globalDependencies.auxyOriginalMap[auxyIndex];

        auto& origItem = this->globalDependencies.originalAuxyMap[originalIndex];
        origItem.auxies.erase(auxyIndex);
        origItem.levelsUsed |= levelsMask;
        triadClosed = origItem.auxies.empty();
        levelsUsed = origItem.levelsUsed;
        expr = origItem.expr;

        // Every auxiliary index belongs to an induction triad.
        const auto itIV = this->globalDependencies.originalInductionVariableMap.find(originalIndex);
        assert(itIV != this->globalDependencies.originalInductionVariableMap.end()
            && "updateGlobal: an auxiliary index outside every induction triad");
        indVar = std::get<0>(itIV->second);
        recCounter = std::get<1>(itIV->second);
        strongInduction =
            this->globalDependencies.strongInductionOriginals.count(originalIndex) != 0;
    }

    if (!triadClosed) return;

    // Disintegrate expr -> (chain, head)
    std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> tempChain;
    std::string value = ce::disintegrateImplication(expr, tempChain, definitions.configurations());

    std::vector<std::string> ky;
    ky.reserve(tempChain.size());
    for (std::size_t i = 0; i < tempChain.size(); ++i) ky.push_back(std::get<0>(tempChain[i]));

    // ---- heavy work without holding dependency lock ----
    Mail mailOut;

    // A winning triad whose promotion meets a goal block without the typing
    // fact waits for it (D-517).
    const auto itStepWin = this->globalDependencies.stepWins.find(expr);
    const bool winningTriad =
        itStepWin != this->globalDependencies.stepWins.end()
        && itStepWin->second.originalIndex == originalIndex;

    // Walk to the memory block for this key
    Memory* memoryBlock = accessMemory(ky, this->body);
    if (memoryBlock != NULL) {
        // Vacuous-premise emission gate (the induction twin of the
        // drainUpdateGlobalSingleChapter gate): an induction theorem whose chain
        // LB sits at or below a contradicted premise set is vacuous — do
        // not promote it. Runs at the single-threaded post-join drain of
        // updateGlobalTuples, so the parent walk is race-free.
        if (gl::StatementKnowledge::hasContradictedAncestor(memoryBlock)) {
            return;
        }
        // The goal-block reads below (typing fact, its levels, the goal
        // probe) and the goal erase run inside the barrier seam window, where
        // the steward may compact an Idle block's arena at any moment
        // (LbMemory::reshuffle moves its pages one at a time). Hold the block
        // through the steward door for them, as settleStepClosure and
        // retryWaitingPromotions do (I-122): an unclaimed read overlapping a
        // compaction can miss the typing fact and defer the promotion by one
        // barrier, which renumbers the compacted implications.
        steward->claimAndLoadForWork(*memoryBlock, /*phase=*/4,
                                     lbdeload::kDeloadDirectory);
        const auto releaseGoalBlock = [memoryBlock] {
            memoryBlock->stewardClaim.store(
                static_cast<uint8_t>(Memory::StewardClaim::Idle),
                std::memory_order_release);
        };
        // --- Induction typing gate ---
        // Before promoting an induction theorem, verify the induction variable is
        // constrained to N (anchor_args[0]). A sound induction requires
        // `(in[indVar,N])` derivable from the current chain; without it, the
        // theorem universally quantifies over everything rather than over N, and
        // the implicit-typing inference of the original induction-scheduling is
        // unsound. See docs/agentic_swdd/induction_typing_plan.md.
        // Typing not established — reject induction promotion silently. A
        // winning triad waits for the fact: the barrier re-queues this
        // promotion once the goal block knows it
        // (D-517).
        if (!scopeLifecycle.inductionTypingKnown(*memoryBlock, indVar)) {
            if (winningTriad) {
                itStepWin->second.promotionAwaitsTyping = true;
                itStepWin->second.awaitingAuxyIndex = auxyIndex;
            }
            releaseGoalBlock();
            return;
        }

        // Level verdict: the step proof, the base proof and the typing fact
        // the induction consumes must together use every premise level of
        // the conjecture; otherwise the triad registers nothing and switches
        // nothing off (D-516).
        const uint64_t typingLevels = scopeLifecycle.inductionTypingLevels(*memoryBlock, indVar);
        const uint64_t premiseLevels = ScopeLifecycle::inductionPremiseLevels(ky);
        if (((levelsUsed | typingLevels) & premiseLevels) != premiseLevels) {
            releaseGoalBlock();
            return;
        }

        scopeLifecycle.deactivateUnnecessary(expr, this->body);  // deactivates all blocks tied to this theorem.

        // Non-minting packed probe: a goal whose original was never
        // interned cannot be registered, so a lookup miss is a definitive
        // probe miss (the same negative branch the string find produced).
        const int32_t tbpValId =
            lookupToBeProved(memoryBlock->intToBeProved, memoryBlock->nameMap,
                             value, "main");
        if (tbpValId == 0) releaseGoalBlock();
        if (tbpValId != 0) {
            // Python asserts this is an empty set
            assert(memoryBlock->intToBeProved.runLen(tbpValId) == 0);
            const NameId tbpOrigId = memoryBlock->nameMap.lookup(value);
            assert(tbpOrigId != 0);
            memoryBlock->intToBeProved.eraseSet(
                packStatementKey(tbpOrigId, NameMap::MAIN_ID));
            // The rest reads only never-deloaded state of the goal block.
            releaseGoalBlock();

            // No settled-head deposit here: the compact broadcast below
            // reaches this LB through root mail, the rule installs, and the
            // head re-derives from the LB's own seed premises with
            // residence-honest levels {0..chain-1}. A direct deposit would
            // have to fabricate a level run — the retired {0..ky.size()}
            // stamp poisoned this LB's level arithmetic (one past its level
            // universe) and broke the allLevelsInvolved verdict of every
            // statement derived from it
            // (D-324).
            // WAKE DOOR 2 (D-194): a main goal was erased above — state the
            // deactivation survey must see next step; keep the LB swept (a
            // dischargedForever LB has no next step, I-102). hasWork is
            // never-deloaded logical state (I-153), so no claim handshake
            // is needed for this flag-only write.
            if (!memoryBlock->dischargedForever) {
                memoryBlock->hasWork = true;
            }

            // FullBind-rebuild expr for every OUTWARD channel (D-256). A form
            // whose outer binder omits an anchor slot must never circulate:
            // compacting it lifts the unbound anchor slots into template
            // parameters, producing the forbidden (Anchor...[...,u_N]) rule
            // elements (an anchor expression never carries u_ arguments)
            // whose canonical `compilation` citation has no origin row for
            // the chapter walker. Compaction, the `theorem` origin row, and
            // the registry all take the FullBind form; `expr` survives only
            // for or_pairs matching and as reformulateTheorem input (its
            // outputs rebind themselves).
            const std::string exprFullBind =
                gl::ImplicationBuilder::reconstructImplicationFullBind(ky, value);

            // A head-switched companion of an in-run or theorem is subsumed
            // by it (see updateGlobalSingleChapter): neither registered nor
            // broadcast. The goal erase and wake above have run; the local
            // mailOut is still empty, so nothing is left to flush.
            if (orInRunCompanions.count(exprFullBind) != 0) {
                std::cout << "Withheld, companion of an in-run or theorem: "
                          << exprFullBind << std::endl;
                return;
            }

            // ASIC 0.1 reshuffle: the legacy Mail::implications channel is
            // removed; this implication travels SOLELY as the D-76 compact
            // (implication<N>[...]) statement, linked to the original by a
            // `compilation` origin (also the receiver's mandatory
            // paired-origin; I-44: exprOriginMap is documentation, not a
            // proof input). Deferred compile+deposit (parallel worker path
            // -> I-28); drained single-threaded post-pool.join.
            recordPendingCompaction(exprFullBind, static_cast<int>(ky.size()), coreId);

            if (parameters.trackHistory) {
                ExpressionWithValidity ev(exprFullBind, "main");
                proofHistory.addOrigin(mailOut.exprOriginMap, ev, std::make_pair("theorem", std::vector<ExpressionWithValidity>()), (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            }

            // ---- record in globalTheoremList (short critical section) ----
            // The method names the hypothesis kind of the triad that closed;
            // the verifier and the Lean export rebuild that hypothesis
            // (D-500).
            if (appendGlobalTheorem(exprFullBind,
                    strongInduction ? "strong induction" : "induction", indVar,
                    recCounter, memoryBlock)) {
                std::cout << exprFullBind << std::endl;
            }
            // checkOrCompletion below intentionally keeps `expr` (the
            // as-scheduled form) because or_pairs.txt is loaded verbatim from
            // the conjecturer's output and is matched by exact spelling.
            orCandidateProgress.checkOrCompletion(expr, coreId);

            std::vector<std::pair<std::string, std::string>> outTheorems;
            if (reformulateTheorem(expr, outTheorems)) {
                for (const auto& pair : outTheorems) {
                    const std::string& refTheoremRaw = pair.first;
                    const std::string& refTheoremCompiled = pair.second;

                    if (appendGlobalTheorem(refTheoremCompiled,
                            "reformulated statement", expr, "-1", memoryBlock)) {
                        std::cout << refTheoremCompiled << std::endl;
                    }

                    if (parameters.trackHistory) {
                        ExpressionWithValidity ev(refTheoremCompiled, "main");
                        proofHistory.addOrigin(mailOut.exprOriginMap, ev, std::make_pair("theorem", std::vector<ExpressionWithValidity>()), (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
                    }

                    tempChain.clear();
                    std::string valueR = ce::disintegrateImplication(refTheoremCompiled, tempChain, definitions.configurations());

                    ky.clear();
                    ky.reserve(tempChain.size());
                    for (std::size_t i = 0; i < tempChain.size(); ++i) ky.push_back(std::get<0>(tempChain[i]));

                    // No settled-head deposit at the reformulated chain LB:
                    // the compact broadcast below reaches it through root
                    // mail, the rule installs, and the head re-derives from
                    // the LB's own seed premises with residence-honest levels
                    // {0..chain-1}. A direct deposit would have to fabricate
                    // a level run — the retired {0..ky.size()} stamp poisoned
                    // the receiving LB's level arithmetic (one past its level
                    // universe) and broke the allLevelsInvolved verdict of
                    // every statement derived from it
                    // (D-324).

                    // ASIC 0.1 reshuffle: reformulated implication travels
                    // SOLELY as the D-76 compact statement (legacy
                    // Mail::implications removed); deferred compile+deposit.
                    recordPendingCompaction(refTheoremCompiled, static_cast<int>(ky.size()), coreId);
                }
            }


            // Flush mail. New mail system (D-137): the local
            // mailOut carries only origin lines (documentation, I-44); merge them
            // into the root's mailOut, shipped by the NEXT commit barrier
            // (one-iteration delay, matching the old post-smashMail timing).
            // Single-threaded post-join; claim/reload the root because mailOut
            // rides its deloadable arena.
            assert(coreId >= 0);
            steward->claimAndLoadForWork(this->body, /*phase=*/4,
                                         lbdeload::kDeloadDirectory);
            mergeBatchIntoMailOut(mailOut, this->body);
            this->body.stewardClaim.store(
                static_cast<uint8_t>(Memory::StewardClaim::Idle),
                std::memory_order_release);
        }
    }
}

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
        /// @see updateGlobalSingleChapterLess — the consumer; `Memory::exprKeyView`.
int TheoremPublisher::compareProducerChains(const Memory* a, const Memory* b) {
    if (a == b) return 0;
    while (true) {
        if (a == nullptr) return (b == nullptr) ? 0 : -1;
        if (b == nullptr) return 1;
        if (a == b) {
            // Chains converged on a shared ancestor while every deeper key
            // compared equal — two distinct LBs with a byte-equal full chain
            // would break the full-chain identity every chain-matched
            // consumer (deload naming, debug traps) relies on.
            assert(false && "compareProducerChains: distinct LBs share a "
                            "byte-equal full parentMemory chain");
            return 0;
        }
        const int c = compareSpans(a->exprKeyView(), b->exprKeyView());
        if (c != 0) return c;
        a = a->parentMemory;
        b = b->parentMemory;
    }
}

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
int TheoremPublisher::updateGlobalSingleChapterLess(const UpdateGlobalSingleChapterRec& a,
                                               const UpdateGlobalSingleChapterRec& b) {
    const int c = compareSpans(StrSpan(a.theorem), StrSpan(b.theorem));
    if (c != 0) return c;
    // Level-verdict tiebreak, TRUE FIRST: when one theorem seals from both a
    // level-complete and a level-poor route in the same iteration, the sink's
    // string dedup makes the FIRST-drained record's method the registration —
    // so the level-complete record must drain first, else the theorem lands
    // in the proved-not-broadcast tier although a full-level proof closed.
    if (a.allLevelsInvolved != b.allLevelsInvolved)
        return a.allLevelsInvolved ? -1 : 1;
    // Producer-chain tiebreak: the deterministic LB identity (leaf-to-root
    // exprKey bytes). coreId must not participate anywhere in this order —
    // it is the phase-dispatch worker slot, a scheduling race outcome, and
    // for a level-poor pair the first record's producer chain becomes the
    // tier chapter's derivation.
    return compareProducerChains(a.producer, b.producer);
}

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
void TheoremPublisher::drainUpdateGlobalSingleChapter() {
    // proveKernel emplaces the page set before phase-1 and this is its sole,
    // unconditionally-reached post-join call site — always engaged here.
    assert(updateGlobalSingleChapterPages
        && "drainUpdateGlobalSingleChapter: proveKernel must emplace the page set");
    SealedPageSet& ps = *updateGlobalSingleChapterPages;
    if (ps.recordCount() == 0) { updateGlobalSingleChapterPages.reset(); return; }

    // Single-threaded post-join (gl::WorkerContext::coreId() == -1 -> reserved last gen
    // slot). Gather record pointers, index-sort on the (theorem bytes, level
    // verdict true-first, producer chain) total order — every key a pure
    // function of proof state, never the dispatch coreId. The refs / idx ride
    // the gen-scratch tiers; the sealed strings ride updateGlobalSingleChapterPages
    // (never crossed, I-124).
    const unsigned slot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : genScratchArenas().slotCount() - 1;
    ScratchArena& gArena = genScratchArenas().forSlot(slot);
    const ArenaOffset mark = gArena.cursor();
    DirtyState d = DirtyState::Clean;
    PagedVector<const UpdateGlobalSingleChapterRec*> refs(&gArena, &d);
    ps.forEachRecord<UpdateGlobalSingleChapterRec>(
        [&](const UpdateGlobalSingleChapterRec& r) { refs.push_back(&r); });

    const int32_t n = static_cast<int32_t>(refs.size());
    int32_t* idx = reinterpret_cast<int32_t*>(
        gArena.resolve(gArena.alloc(n * sizeof(int32_t), alignof(int32_t))));
    for (int32_t i = 0; i < n; ++i) idx[i] = i;
    std::sort(idx, idx + n, [&](int32_t a, int32_t b) {
        return updateGlobalSingleChapterLess(*refs[static_cast<size_t>(a)],
                                      *refs[static_cast<size_t>(b)]) < 0;
    });

    for (int32_t oi = 0; oi < n; ++oi) {
        const UpdateGlobalSingleChapterRec& r = *refs[static_cast<size_t>(idx[oi])];
        // Vacuous-premise emission gate: a theorem sealed by an LB whose
        // premise set turned out inconsistent (self-or-ancestor
        // mainContradiction) is vacuous — suppress it. The flags were set
        // in this iteration's worker phases, so the single-threaded drain
        // sees them; the parent walk is race-free here (the sanctioned
        // I-28 seam).
        assert(r.producer
            && "drainUpdateGlobalSingleChapter: seal sites must carry the producer LB");
        if (gl::StatementKnowledge::hasContradictedAncestor(r.producer)) continue;
        // Materialize the std::string at the still-heap updateGlobalSingleChapter edge
        // (the sink is not a statification target; the sealed bytes crossing the
        // join is the win). The sealed strings are live until freePages() below.
        // The sealed level verdict rides along: a false verdict runs only the
        // lifecycle section inside plus the proved-not-broadcast tier
        // registration (D-278).
        updateGlobalSingleChapter(StrSpan(r.theorem).toStdString(), r.coreId,
                           r.producer, r.allLevelsInvolved);

        // Settled-goal deposit: a primed __contradiction__ discharge emits a
        // theorem whose head is gl::DefinitionExpansion::negate(seed) (I-165), so hand the seed to the
        // parent's settled-goal inbox. The parent's own end-of-burst
        // drainDisprovedGoals probes BOTH directions against the parent's
        // MAIN goals — seed == goal is a disproof (the verbatim twin fired;
        // goal + machinery wiped), gl::DefinitionExpansion::negate(seed) == goal is a proof (the
        // complement twin fired; goal closed with success semantics,
        // D-279); a double miss is an
        // already-closed goal — a defined outcome, so the deposit is
        // unconditional for primed producers. It runs EVEN when the record's
        // level verdict refused registration above: goal settlement is
        // level-free by contract (D-278). Single-threaded post-join seam;
        // the inbox rides the never-deloaded persistent pool (the I-108
        // pattern), so no claim is needed while the parent is cold, and the
        // hasWork wake satisfies I-153.
        if (r.producer->primedForContradiction) {
            Memory* parent = r.producer->parentMemory;
            assert(parent
                && "primed contradiction LB must have a parent LB");
            const StrSpan key = r.producer->exprKeyView();
            static const char CONTRA_PFX[] = "__contradiction__";
            const int32_t pfxLen =
                static_cast<int32_t>(sizeof(CONTRA_PFX) - 1);
            assert(key.len > pfxLen
                && std::memcmp(key.ptr, CONTRA_PFX,
                               static_cast<size_t>(pfxLen)) == 0
                && "primed LB exprKey must carry the __contradiction__ prefix");
            parent->pendingDisprovedGoals.mint(
                StrSpan(key.ptr + pfxLen, key.len - pfxLen));
            parent->hasWork = true;
        }
    }
    gArena.popTo(mark);
    ps.seal();
    ps.freePages();
    updateGlobalSingleChapterPages.reset();
}

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
bool TheoremPublisher::reformulateTheorem(const std::string& theorem, std::vector<std::pair<std::string, std::string>>& outTheorems) {
    if (parameters.ban_disintegration) return false;

    bool foundAny = false;

    // 1. Disintegrate the original theorem
    std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> originalChain;
    std::string head = ce::disintegrateImplication(theorem, originalChain, definitions.configurations());

    // Disintegration safety check
    if (originalChain.empty()) {
        return false;
    }

    // 2. Identify indices of all Definitions in the chain
    std::vector<size_t> defIndices;
    for (size_t i = 0; i < originalChain.size(); ++i) {
        const LogicalEntity* le = definitions.compiledEntity(
            extractExpressionUniversalSpan(StrSpan(std::get<0>(originalChain[i]))));
        if (le && !le->definedSet.empty()) {
            defIndices.push_back(i);
        }
    }

    if (defIndices.empty()) return false;

    // 3. Iterate through each definition to see if it qualifies for reformulation
    for (size_t targetDefIdx : defIndices) {

        // A. Construct Reshuffled Key: [Others..., TargetDef]
        // We preserve the relative order of the non-target premises
        std::vector<std::string> reshuffledKey;
        reshuffledKey.reserve(originalChain.size());

        for (size_t i = 0; i < originalChain.size(); ++i) {
            if (i != targetDefIdx) {
                reshuffledKey.push_back(std::get<0>(originalChain[i]));
            }
        }
        // Add the target definition at the VERY END
        reshuffledKey.push_back(std::get<0>(originalChain[targetDefIdx]));

        // B. Reconstruct the Full Theorem with the ORIGINAL Head
        // This is crucial: we need the system to calculate the binding scope
        // for the new order of premises.
        std::string tempTheorem = gl::ImplicationBuilder::reconstructImplication(reshuffledKey, head);

        // C. Disintegrate again to inspect the binding of the last link (our TargetDef)
        std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> tempChain;
        std::string tempHead = ce::disintegrateImplication(tempTheorem, tempChain, definitions.configurations());

        // Safety checks
        if (tempChain.empty() || tempHead != head) continue;

        // The last link in tempChain corresponds to reshuffledKey.back() (our TargetDef)
        const auto& lastLink = tempChain.back();
        const std::string& lastDefExpr = std::get<0>(lastLink);

        // D. Verify Logic: Bound Variable == Set Argument
        const LogicalEntity* le = definitions.compiledEntity(
            extractExpressionUniversalSpan(StrSpan(lastDefExpr)));
        if (le == nullptr) continue;

        // Find the expected argument position from the signature
        std::vector<std::string> sigArgs = ce::getArgs(le->signature);
        auto itSig = std::find(sigArgs.begin(), sigArgs.end(), le->definedSet);
        if (itSig == sigArgs.end()) continue;

        size_t argIndex = std::distance(sigArgs.begin(), itSig);
        std::vector<std::string> actualArgs = ce::getArgs(lastDefExpr);
        if (argIndex >= actualArgs.size()) continue;

        std::string expectedArg = actualArgs[argIndex];

        // CHECK: Is this definition acting as the peeling layer?
        //
        // Pre-unification this was `boundVars.size() == 1 && boundVars[0] ==
        // expectedArg`, reading the reconstructed last-link >[...]. Under the
        // unified binder rule (D-75) every
        // non-u_ variable is bound, so that cardinality no longer encodes the
        // peeling-layer condition. Re-express the SAME predicate directly from
        // originalChain — exactly the set the old sparse reconstructImplication
        // placed at the targetDef (last) link: a non-u_ arg of the targetDef
        // that occurs >=2x across (all premises + head) and appears in NO
        // other premise (so its first occurrence in the [others..., targetDef,
        // head] order was the targetDef link). This keeps the reformulation
        // set byte-identical; only the emitted theorem's outer >[...] widen
        // (DoD-covered). The inner existence binder stays exactly the single
        // peeled set-argument (see below).
        std::map<std::string, int> reformOcc;
        auto reformCount = [&](const std::string& e) {
            for (const std::string& a : ce::getArgs(e)) {
                if (a.size() >= 2 && a[0] == 'u' && a[1] == '_') continue;
                reformOcc[a]++;
            }
        };
        for (size_t ci = 0; ci < originalChain.size(); ++ci)
            reformCount(std::get<0>(originalChain[ci]));
        reformCount(head);

        std::set<std::string> reformOtherArgs;
        for (size_t ci = 0; ci < originalChain.size(); ++ci) {
            if (ci == targetDefIdx) continue;
            for (const std::string& a : ce::getArgs(std::get<0>(originalChain[ci])))
                reformOtherArgs.insert(a);
        }

        std::set<std::string> lastLinkBound;
        for (const std::string& a : actualArgs) {
            if (a.size() >= 2 && a[0] == 'u' && a[1] == '_') continue;
            if (reformOcc[a] >= 2 && reformOtherArgs.find(a) == reformOtherArgs.end())
                lastLinkBound.insert(a);
        }

        // The emitted theorem binds only the defined set inside the
        // existence and rebinds the remaining premises' own variables, so
        // every other variable that is first bound at the definition link
        // (one occurring in no remaining premise, e.g. a value used only by
        // the definition) would be left without a binder. Such a theorem
        // is not formed (I-4).
        const std::vector<std::string>& lastLinkVars = std::get<1>(lastLink);
        const bool onlyDefinedSetBoundAtDefinition =
            lastLinkVars.size() == 1 && lastLinkVars[0] == expectedArg;

        if (onlyDefinedSetBoundAtDefinition
            && lastLinkBound.size() == 1 && *lastLinkBound.begin() == expectedArg) {

            // FOUND VALID REFORMULATION!

            // E. Apply Transformation
            // Structure: (Premises... -> !( > [vars] LastDef !Head ))

            // Prepare Negated Head
            std::string negatedHead;
            if (head.size() > 1 && head[0] == '!') {
                negatedHead = head.substr(1);
            }
            else {
                negatedHead = "!" + head;
            }

            // Build the Special Innermost Tail.
            // The gate guarantees the peeling set is exactly {expectedArg}, so
            // the negated-existence binds precisely the Definition's
            // set-argument — byte-identical to the pre-unification
            // `boundVars[0]`. This inner existential binder is NOT subject to
            // the unified "bind all non-u_" rule (that rule governs the outer
            // universal-implication wrappers, which DO widen via tempChain);
            // the existential's variable is fixed by the reformulation itself.
            std::string varsStr = expectedArg;

            // 1. Raw Version (Original Code)
            std::string rawInnerTail = "!(>[" + varsStr + "]" + lastDefExpr + negatedHead + ")";

            // 2. Compiled Version (New Logic)
            std::string compiledInnerTail = structuralCompiler.compileExpression(rawInnerTail);

            // Wrap the remaining premises (Standard Implication)
            std::string currentTailRaw = rawInnerTail;
            std::string currentTailCompiled = compiledInnerTail;

            for (int i = static_cast<int>(tempChain.size()) - 2; i >= 0; --i) {
                const auto& link = tempChain[i];
                const std::string& premise = std::get<0>(link);
                const std::vector<std::string>& vars = std::get<1>(link);

                std::string vStr;
                for (size_t v = 0; v < vars.size(); ++v) {
                    if (v > 0) vStr += ",";
                    vStr += vars[v];
                }

                std::string wrapper = "(>[" + vStr + "]" + premise;
                currentTailRaw = wrapper + currentTailRaw + ")";
                currentTailCompiled = wrapper + currentTailCompiled + ")";
            }

            outTheorems.push_back(std::make_pair(currentTailRaw, currentTailCompiled));
            foundAny = true;
        }
    }

    return foundAny;
}

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
void TheoremPublisher::recordPendingCompaction(const std::string& original, int kySize, int coreId) {
    std::lock_guard<std::mutex> lock(this->pendingCompactionMutex);
    this->pendingCompactionQueue.push_back(std::make_tuple(original, kySize, coreId));
}

// readSimpleFacts() and saveFilteredConjectures() — moved to filter.cpp.
void TheoremPublisher::saveProvedTheorems() {
    namespace fs = std::filesystem;

    // Resolve path: <repo>/files/theorems/theorems.txt
    const auto theoremsDir =
        gl::projectRoot()
        / "files" / "theorems";

    const auto filePath = theoremsDir / "theorems.txt";

    // Create directory if it doesn't exist
    std::error_code ec;
    fs::create_directories(theoremsDir, ec);

    // Open in append mode
    std::ofstream ofs(filePath, std::ios::app);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open " << filePath << " for appending." << std::endl;
        return;
    }

    int count = 0; // Move counter outside to use it after unlocking

    // START CHANGE: Explicit scope block
    {
        std::lock_guard<std::mutex> lock(this->theoremListMutex);

        for (const auto& entry : this->globalTheoremList) {
            // entry is tuple<theorem, method, var, recCounter>
            const std::string& theorem = std::get<0>(entry);
            const std::string& method = std::get<1>(entry);

            // Filter out debug/internal entries if necessary, or save all.
            // "debug" method entries are usually from findEnds and shouldn't be saved as proved theorems.
            if (method != "debug") {
                ofs << theorem << "\n";
                count++;
            }
        }
    }
    // END CHANGE: lock is released here, before the function exits or prints to console

    std::cout << "Saved " << count << " new proved theorems to file." << std::endl;
}


void TheoremPublisher::saveProvedTheoremsFiltered(const std::vector<std::string>& essentialTheorems,
                                                    const std::unordered_set<std::string>& externalTheorems) {
    namespace fs = std::filesystem;

    const auto theoremsDir =
        gl::projectRoot()
        / "files" / "theorems";

    const auto filePath = theoremsDir / "theorems.txt";

    std::error_code ec;
    fs::create_directories(theoremsDir, ec);

    // Build compact→expanded lookup from globalTheoremList.
    // globalTheoremList stores expanded forms; the compressor may have
    // reverse-compiled existence heads to compact form.
    // We compile each globalTheoremList entry the same way the compressor
    // constructor does, then map back.
    std::unordered_map<std::string, std::string> compactToExpanded;
    for (const auto& tpl : globalTheoremList) {
        const std::string& expanded = std::get<0>(tpl);
        compactToExpanded[expanded] = expanded;  // identity: already expanded

        // Also try reverse-compiling to build compact key
        std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> tempChain;
        std::string head = ce::disintegrateImplication(expanded, tempChain, this->definitions.configurations());

        if (gl::startsWith(head, "!(>", 3)) {
            std::string compiledHead = structuralCompiler.compileExpression(
                head);
            std::vector<std::string> chain;
            for (const auto& t : tempChain) chain.push_back(std::get<0>(t));
            std::string compact = gl::ImplicationBuilder::reconstructImplication(chain, compiledHead);
            compactToExpanded[compact] = expanded;
        }
    }

    std::ofstream ofs(filePath, std::ios::trunc);
    if (!ofs.is_open()) {
        std::cerr << "Error: Could not open " << filePath << " for writing." << std::endl;
        return;
    }

    // Second file: compiled forms (for proof graph pruning — keeps or0, existence2, etc.)
    const auto compiledPath = theoremsDir / "compiled_theorems.txt";
    std::ofstream ofsCompiled(compiledPath, std::ios::trunc);

    int written = 0;
    for (const std::string& theorem : essentialTheorems) {
        // Skip surviving external theorems — they are imported, not GL-proved here
        if (externalTheorems.count(theorem)) continue;

        auto it = compactToExpanded.find(theorem);
        std::string toWrite = (it != compactToExpanded.end()) ? it->second : theorem;

        // compiled_theorems.txt: write as-is (keeps compiled heads)
        if (ofsCompiled.is_open()) ofsCompiled << toWrite << "\n";

        // theorems.txt: expand runtime expressions (or0, etc.) to base form
        // so subsequent batches can parse them.
        toWrite = definitionExpansion.expandToBaseForm(toWrite);
        ofs << toWrite << "\n";
        ++written;
    }

    std::cout << "Rewrote theorems.txt with " << written
        << " globally essential theorems (excluded "
        << externalTheorems.size()
        << " external)." << std::endl;
}

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
void TheoremPublisher::mergeTheoremOriginMailIntoRoot(const Mail& documentation) {
    assert(documentation.statements.empty()
        && "mergeTheoremOriginMailIntoRoot: documentation mail carries origin lines only");
    if (documentation.exprOriginMap.empty()) return;
    if (steward) {
        steward->claimAndLoadForWork(this->body, /*phase=*/4,
                                     lbdeload::kDeloadDirectory);
        mergeBatchIntoMailOut(documentation, this->body);
        this->body.stewardClaim.store(
            static_cast<uint8_t>(Memory::StewardClaim::Idle),
            std::memory_order_release);
    } else {
        this->body.ensureLoadedForRead(lbdeload::kDeloadDirectory);
        mergeBatchIntoMailOut(documentation, this->body);
    }
}
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
void TheoremPublisher::retireSubsumedCompanion(const std::string& companion) {
    std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> chain;
    const std::string head =
        ce::disintegrateImplication(companion, chain, definitions.configurations());
    std::vector<std::string> ky;
    ky.reserve(chain.size());
    for (const auto& link : chain) ky.push_back(std::get<0>(link));

    // A companion that was never scheduled as a conjecture has no logic
    // blocks to retire.
    Memory* goalBlock = accessMemory(ky, this->body);
    if (goalBlock == nullptr) return;

    // Settle the goal exactly as a firing complement twin would: its seed is
    // negate(head), the theorem it emits is headed head (I-165). The goal
    // block's own end-of-burst drain closes the goal with success semantics
    // and closes the opposite-head goal as disproved (D-279). An inactive
    // goal block has no goal left to settle.
    if (goalBlock->isActive) {
        goalBlock->pendingDisprovedGoals.mint(StrSpan(gl::DefinitionExpansion::negate(head)));
        goalBlock->hasWork = true;
    }

    // The proved-theorem lifecycle of updateGlobalSingleChapter: both
    // contradiction twins and the induction blocks of the companion and of
    // its opposite-head conjecture retire.
    scopeLifecycle.deactivateUnnecessary(companion, this->body);
    scopeLifecycle.deactivateUnnecessary(gl::ImplicationBuilder::reconstructImplication(ky, gl::DefinitionExpansion::negate(head)), this->body);
}

}
