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
#include "or_theorem_builder.hpp"
namespace gl {
/// @brief Bind the or theorem builder operation inputs.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @param fullTheoremList Borrowed std::vector interface or state; must outlive this component.
/// @param globalTheoremList Borrowed std::vector interface or state; must outlive this component.
/// @param orBuiltByPair Borrowed std::vector interface or state; must outlive this component.
/// @param orCandidates Borrowed std::vector interface or state; must outlive this component.
/// @param orCandidatesMutex Borrowed std::mutex interface or state; must outlive this component.
/// @param orInRunScannedRows Borrowed std::vector interface or state; must outlive this component.
/// @param orPairsFromHeadSwitch Borrowed std::vector interface or state; must outlive this component.
/// @param orRuleCompiler Borrowed OrRuleCompiler interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param proofHistory Borrowed ProofHistory interface or state; must outlive this component.
/// @param theoremListMutex Borrowed std::mutex interface or state; must outlive this component.
/// @param theoremPublisher Borrowed TheoremPublisher interface or state; must outlive this component.
/// @param orInRunCompanions Existing shared state or operation interface, borrowed for this component lifetime.
/// @param scopeLifecycle Existing shared state or operation interface, borrowed for this component lifetime.
/// @param deactivationSweepOwed Existing shared state or operation interface, borrowed for this component lifetime.
/// @return None; construction binds the supplied references.
OrTheoremBuilder::OrTheoremBuilder(const CompiledDefinitionRegistry& definitions, std::vector< std::tuple<std::string, std::string, std::string, std::string >>& fullTheoremList, std::vector< std::tuple<std::string, std::string, std::string, std::string >>& globalTheoremList, std::map<std::pair<std::string, std::string>, std::string>& orBuiltByPair, std::vector<OrCandidate>& orCandidates, std::mutex& orCandidatesMutex, std::unordered_set<std::string>& orInRunScannedRows, std::vector<std::pair<std::string, std::string>>& orPairsFromHeadSwitch, OrRuleCompiler& orRuleCompiler, const ProverParameters& parameters, ProofHistory& proofHistory, std::mutex& theoremListMutex, TheoremPublisher& theoremPublisher, std::unordered_set<std::string>& orInRunCompanions, ScopeLifecycle& scopeLifecycle, bool& deactivationSweepOwed) : definitions(definitions), fullTheoremList(fullTheoremList), globalTheoremList(globalTheoremList), orBuiltByPair(orBuiltByPair), orCandidates(orCandidates), orCandidatesMutex(orCandidatesMutex), orInRunScannedRows(orInRunScannedRows), orPairsFromHeadSwitch(orPairsFromHeadSwitch), orRuleCompiler(orRuleCompiler), parameters(parameters), proofHistory(proofHistory), theoremListMutex(theoremListMutex), theoremPublisher(theoremPublisher) , orInRunCompanions(orInRunCompanions), scopeLifecycle(scopeLifecycle), deactivationSweepOwed(deactivationSweepOwed) {}


std::string OrTheoremBuilder::headSwitchOne(const std::string& theorem) const {
    // Stateless head-switch (contrapositive) construction.
    // Returns the rebuilt implication string if the chain has a negated,
    // non-quantified premise; returns an empty string otherwise. The byte-
    // level construction is identical to the one inlined inside the
    // big-iteration loop in analyzeExpressions: same disintegrateImplication
    // call, same rightmost-negated-premise scan, same manual `(>[vars]…)`
    // concatenation.
    std::vector<std::tuple<std::string, std::vector<std::string>,
                           std::set<std::string>>> chain;
    std::string head = ce::disintegrateImplication(theorem, chain, this->definitions.configurations());

    int negIdx = -1;
    for (int i = (int)chain.size() - 1; i >= 0; --i) {
        const std::string& prem = std::get<0>(chain[i]);
        if (!prem.empty() && prem[0] == '!') {
            if (std::get<1>(chain[i]).empty()) negIdx = i;
            break;
        }
    }
    if (negIdx < 0) return std::string();

    // The negated head takes the swapped premise's place, unless a later
    // premise binds a variable of the head: then it goes right after the
    // last such premise, so every variable it uses is bound before it. The
    // swapped premise precedes those binders and cannot mention them, so the
    // move keeps the theorem's meaning (D-564).
    const auto headMentions = [&head](const std::string& variable) {
        for (std::size_t open = head.find('['); open != std::string::npos;
             open = head.find('[', open + 1)) {
            const std::size_t close = head.find(']', open);
            assert(close != std::string::npos && "headSwitchOne: unclosed argument list");
            for (std::size_t start = open + 1; start <= close;) {
                std::size_t comma = head.find(',', start);
                if (comma == std::string::npos || comma > close) comma = close;
                if (head.compare(start, comma - start, variable) == 0) return true;
                start = comma + 1;
            }
        }
        return false;
    };
    int newPremAfter = -1;
    for (int i = negIdx + 1; i < (int)chain.size(); ++i) {
        for (const std::string& variable : std::get<1>(chain[i])) {
            if (headMentions(variable)) {
                newPremAfter = i;
                break;
            }
        }
    }

    const std::string& negPrem = std::get<0>(chain[negIdx]);
    std::string newHead = negPrem.substr(1);
    // Negate the old head; an already-negated head loses its `!` instead
    // (double-negation cancellation — GL has no universal `!!` normalizer,
    // and a raw `!!(...)` premise breaks every downstream compiled-core
    // consumer).
    std::string newPrem = (!head.empty() && head[0] == '!')
        ? head.substr(1)
        : "!" + head;

    std::string rebuilt = newHead;
    for (int i = (int)chain.size() - 1; i >= 0; --i) {
        if (i == newPremAfter) rebuilt = "(>[]" + newPrem + rebuilt + ")";
        if (i == negIdx && newPremAfter >= 0) continue;
        const auto& link = chain[i];
        const std::string& premise = (i == negIdx) ? newPrem : std::get<0>(link);
        const std::vector<std::string>& vars = std::get<1>(link);

        std::string vStr;
        for (size_t v = 0; v < vars.size(); ++v) {
            if (v > 0) vStr += ",";
            vStr += vars[v];
        }
        rebuilt = "(>[" + vStr + "]" + premise + rebuilt + ")";
    }
    return rebuilt;
}

/// @brief Fold a proved hypothesis-chain theorem into a compiled
        /// n-ary OR theorem and register its `or<N>` compact.
        ///
        /// @details
        /// Disintegrates @p existenceThm and folds EVERY binder-free
        /// negated premise into the OR head: classically
        /// `!d_1 → (!d_2 → h)` is equivalent to `d_1 ∨ d_2 ∨ h`, so a
        /// chain with K binder-free negated premises yields K+1 disjuncts
        /// — each folded premise un-negated, the head last, all in chain
        /// order and TRUE polarity (a negated head stays a negated
        /// disjunct). A negated premise that binds variables stays a
        /// shared premise: folding it would orphan its binder. Non-negated
        /// premises survive as the reassembled theorem's premise chain.
        ///
        /// The u_-canonical ordered element list is the or-operator's
        /// registry identity: an existing `or`-category entry with the
        /// identical element vector is reused (I-23); otherwise a fresh
        /// `or<orCounter>` is minted into `compiledExpressions` +
        /// `coreExpressionMap`.
        ///
        /// @param existenceThm  The proved direction; supplies premises,
        ///        folded disjuncts, and the head disjunct.
        /// @param companionThm  The head-switched companion; recorded by
        ///        the caller as parent reference only — not consulted for
        ///        construction.
        /// @return The reassembled OR theorem with compiled `(or<N>[…])`
        ///         head, or the empty string when the chain carries no
        ///         binder-free negated premise (caller asserts against
        ///         this for licensed pairs).
        /// @invariant Single-threaded seam — runs post-`prove()` only.
        /// @see `constructOrTheoremsFromPairs` — the walking caller.
        /// @see `headSwitchOne` — pair producer.
std::string OrTheoremBuilder::constructOrTheorem(
    const std::string& existenceThm, const std::string& companionThm) {

    // Disintegrate both theorems
    std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> chain1, chain2;
    std::string head1 = ce::disintegrateImplication(existenceThm, chain1, this->definitions.configurations());
    std::string head2 = ce::disintegrateImplication(companionThm, chain2, this->definitions.configurations());

    // Existence theorem: chain has shared premises + one or more binder-free
    // negated premises (!d_1, !d_2, ...), head = the last disjunct.

    // Every binder-free negated premise is a disjunct of the OR:
    // classically  !d_1 → (!d_2 → h)  ⟺  d_1 ∨ d_2 ∨ h, so the whole
    // hypothesis chain folds into one n-ary OR head. A negated premise
    // that binds variables cannot fold — its binder would orphan (the
    // or head's arguments must be bound by the surviving premise links)
    // — so it stays a shared premise.
    std::vector<int> foldedIdxs;
    for (std::size_t i = 1; i < chain1.size(); ++i) {
        const std::string& prem = std::get<0>(chain1[i]);
        if (!prem.empty() && prem[0] == '!' && std::get<1>(chain1[i]).empty()) {
            foldedIdxs.push_back((int)i);
        }
    }
    if (foldedIdxs.empty()) return "";

    // Build disjuncts in their TRUE polarity, chain order, head LAST.
    // A premise disjunct is the un-negated form of its negated premise;
    // the head enters VERBATIM — a negated head stays a negated disjunct.
    // Registered or-elements carry each disjunct's real sign; every
    // expansion negates elements with double-negation cancellation.
    // Storing only the positive core would flip a negated disjunct's
    // sign on round-trip and register a FALSE or theorem
    // ((a=b) OR (c=d) instead of (a=b) OR !(c=d)).
    std::vector<std::string> disjuncts;
    disjuncts.reserve(foldedIdxs.size() + 1);
    for (int idx : foldedIdxs) {
        disjuncts.push_back(std::get<0>(chain1[idx]).substr(1));  // strip !
    }
    disjuncts.push_back(head1);

    // Collect unique args across all disjuncts (ordered by first appearance)
    std::vector<std::string> orArgs;
    {
        std::set<std::string> seen;
        for (const std::string& d : disjuncts) {
            for (const auto& a : ce::getArgs(d)) {
                if (seen.insert(a).second) orArgs.push_back(a);
            }
        }
    }

    // Build the u_-canonical elements FIRST — the canonical body of the
    // element list is the or-operator's registry identity (I-323).
    std::map<std::string, std::string> argToU;
    for (std::size_t i = 0; i < orArgs.size(); ++i) {
        argToU[orArgs[i]] = "u_" + std::to_string(i + 1);
    }
    std::vector<std::string> elems;
    elems.reserve(disjuncts.size());
    for (const std::string& d : disjuncts) {
        elems.push_back(ce::replaceKeysInString(d, argToU));
    }

    // Registry dedup-or-mint through the one shared site: the or of this
    // canonical body (I-323), whose stored disjunct order may differ, so
    // the instance args below follow its signature through the returned
    // positions.
    std::vector<int> orArgumentPositions;
    const std::string orCoreName =
        orRuleCompiler.findOrMintOrOperator(elems, static_cast<int>(orArgs.size()), nullptr, &orArgumentPositions);

    // Build compiled OR head: (or0[actual_args])
    std::string compiledOrHead = "(" + orCoreName + "[";
    for (std::size_t i = 0; i < orArgs.size(); ++i) {
        if (i > 0) compiledOrHead += ",";
        compiledOrHead += orArgs[static_cast<std::size_t>(orArgumentPositions[i]) - 1];
    }
    compiledOrHead += "])";

    {
        // Report the De Morgan base form `expandToBaseForm` writes: each
        // disjunct negated with double-negation cancellation, two children
        // per conjunction, nested to the left.
        const auto negated = [](const std::string& d) {
            return (!d.empty() && d[0] == '!') ? d.substr(1) : "!" + d;
        };
        std::string conjunction = negated(disjuncts[0]);
        for (std::size_t i = 1; i < disjuncts.size(); ++i) {
            conjunction = "(&" + conjunction + negated(disjuncts[i]) + ")";
        }
        std::cout << "OR compiled: !" << conjunction << " -> " << compiledOrHead << std::endl;
    }

    // Build the shared premises (everything except the folded negated premises)
    std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> sharedChain;
    for (std::size_t i = 0; i < chain1.size(); ++i) {
        if (std::binary_search(foldedIdxs.begin(), foldedIdxs.end(), (int)i)) continue;
        sharedChain.push_back(chain1[i]);
    }

    // Reassemble with compiled OR head
    std::string result = compiledOrHead;
    for (std::size_t i = sharedChain.size(); i-- > 0; ) {
        const auto& [expr, args, _] = sharedChain[i];
        std::string joined;
        for (std::size_t j = 0; j < args.size(); ++j) {
            if (j > 0) joined += ',';
            joined += args[j];
        }
        result = "(>[" + joined + "]" + expr + result + ")";
    }

    return result;
}

/// @brief Construct every OR theorem licensed by the head-switch pairs
/// against the CURRENT `globalTheoremList`, register each in the global
/// registries, and report the parents it subsumes.
///
/// @details
/// Walks `orPairsFromHeadSwitch` — one `(theorem, companion)` pair per
/// proved theorem whose straightened shape carries a binder-free negated
/// premise (`headSwitchOne`). A SINGLE proved direction licenses the
/// disjunction: classically `!d_1 → (!d_2 → h)` is equivalent to
/// `d_1 ∨ d_2 ∨ h`, and every other direction is derivable from it, so
/// the companion's own proof is never required — it is recorded only as
/// the or-theorem row's second parent reference.
///
/// A pair whose source theorem is no longer in `globalTheoremList` is
/// skipped — compression pruning and vacuity retraction remove rows
/// between the pair walk and this call, and both removals are defined
/// pipeline states, not failures. Mirror pairs `(x, y)` and `(y, x)`
/// describe the same disjunction; the canonical (min, max) disjunct-set
/// dedup constructs it once.
///
/// @param consumedParents Out-parameter accumulating the proved parent
///        theorems subsumed by a constructed OR; the caller drops them
///        from the saved theorem files.
/// @return The constructed OR theorems in compiled form, in pair order.
/// @invariant Single-threaded seam — must run after `prove()` has joined.
/// @see `constructOrTheorem` — per-pair OR builder.
/// @see `headSwitchOne` — pair producer.
std::vector<std::string> OrTheoremBuilder::constructOrTheoremsFromPairs(
    std::set<std::string>& consumedParents) {

    std::vector<std::string> orTheorems;

    std::unordered_set<std::string> provedSet;
    for (const auto& t : this->globalTheoremList)
        provedSet.insert(std::get<0>(t));

    std::set<std::pair<std::string, std::string>> walkEmitted;

    for (const auto& pr : this->orPairsFromHeadSwitch) {
        const std::string& exist = pr.first;
        const std::string& comp = pr.second;

        // Source theorem pruned by compression or retracted as vacuous —
        // a defined pipeline state; the pair no longer licenses an OR.
        if (!provedSet.count(exist)) continue;

        // (exist, comp) and (comp, exist) describe the same OR (mirror
        // reformulations of one another). Canonicalize by lexicographically
        // sorting the pair; a pair this walk already emitted is skipped
        // outright (its first sighting consumed both parents).
        const std::pair<std::string, std::string> pairKey(
            std::min(exist, comp), std::max(exist, comp));
        if (!walkEmitted.insert(pairKey).second) {
            std::cout << "OR variant skipped (duplicate disjunct-set already constructed)"
                      << std::endl;
            continue;
        }

        // Cross-seam idempotence: an or built in-run is recovered from the
        // shared ledger — reconstructing from THIS pair's side could mint
        // the mirrored operator. A ledger miss constructs and registers as
        // before. Either way the subsumption bookkeeping and the return
        // vector fire, so the export caller still writes the or row and
        // drops its parents from the saved files.
        std::string orThm;
        const auto built = orBuiltByPair.find(pairKey);
        if (built != orBuiltByPair.end()) {
            orThm = built->second;
            std::cout << "OR theorem already constructed in-run: " << orThm << std::endl;
        } else {
            orThm = constructOrTheorem(exist, comp);
            assert(!orThm.empty()
                && "constructOrTheoremsFromPairs: pair source must carry a negated premise");
            orBuiltByPair.emplace(pairKey, orThm);
            if (theoremPublisher.appendGlobalTheorem(orThm, "or theorem", exist, comp)) {
                this->fullTheoremList.emplace_back(orThm, "or theorem", exist, comp);
                std::cout << "OR theorem constructed: " << orThm << std::endl;
            }
        }
        orTheorems.push_back(orThm);

        consumedParents.insert(exist);
        std::cout << "  parent removed (subsumed by OR): " << exist << std::endl;
        if (provedSet.count(comp)) {
            consumedParents.insert(comp);
            std::cout << "  parent removed (subsumed by OR): " << comp << std::endl;
        }
    }

    return orTheorems;
}

/// @brief In-run OR-theorem construction and broadcast — the A16
/// Phase 2 seam.
///
/// @details
/// Runs on the phase-4 barrier of every `proveKernel` iteration
/// (single-threaded, post-`pool.join()`, after the vacuous-theorem
/// retraction sweep). Walks `globalTheoremList` for rows not yet in
/// `orInRunScannedRows`; per new row, `headSwitchOne` licenses the
/// or (a straightened shape with a binder-free negated premise),
/// `constructOrTheorem` folds and registers the `or<N>` compact
/// (I-23 registry dedup), and the or theorem is registered via
/// `appendGlobalTheorem(or, "or theorem", row, companion)` +
/// `fullTheoremList`.
///
/// The broadcast rides the existing D-76 deferred-compaction drain:
/// `recordPendingCompaction` queues the or theorem, the drain this
/// seam feeds compiles it to its implication compact and merges it
/// into the root's mailOut with an empty level set, and every LB
/// installs it through the status-3 rule door two iterations later.
/// Where the rule's premises hold it fires, the `(or<N>[…])` head
/// deposits locally, disintegrates, and the cohort machinery opens
/// under its normal demand gates. A premise-free or theorem has no
/// implication-compact shape: it registers but is not broadcast.
///
/// Every broadcast or theorem also gets the `("theorem", [])` origin at
/// `main` in the root's mailOut, the documentation every other proved
/// theorem's broadcast carries (`updateGlobalSingleChapter`). Receivers
/// thereby hold a proved-theorem leaf for it besides the compaction's
/// compilation/expansion pair, which alone is a two-node cycle the
/// chapter walker (`buildStack`) cannot ground.
///
/// Idempotence is the string dedup in `appendGlobalTheorem`: a
/// mirror row scanned later — or the post-run export seam
/// (`constructOrTheoremsFromPairs`) — reconstructs the SAME or
/// string through the registry reuse, lands `false`, and neither
/// re-registers nor re-broadcasts. In compressor mode the seam is
/// a defined no-op: the compressor's internal re-derivation is a
/// redundancy probe whose theorem list must stay byte-identical to
/// the run it audits; the export seam constructs the ors on the
/// survivor list.
///
/// Every or theorem the seam registers records its companion in
/// `orInRunCompanions` and retires the companion's logic blocks
/// (`retireSubsumedCompanion`). The seam runs after the theorem
/// drains' deactivation sweep, so it settles the prune those walks
/// owe itself and leaves `deactivationSweepOwed` false (D-451).
///
/// @return Nothing.
/// @invariant Single-threaded seam — phase-4 barrier only; registry
///            mints (`compiledExpressions` / `coreExpressionMap` /
///            `orCounter`) are legal here for the same reason the
///            deferred-compaction drain's are (D-76).
/// @see `constructOrTheorem` — per-pair builder (registry dedup).
/// @see `constructOrTheoremsFromPairs` — the post-run export seam.
/// @see `recordPendingCompaction` — the broadcast queue.
/// @see `mergeTheoremOriginMailIntoRoot` — the documentation door.
/// @see `retireSubsumedCompanion` — the companion's retirement.
void OrTheoremBuilder::constructOrTheoremsInRun() {
    // Compressor re-derivation is a redundancy probe whose theorem list
    // must stay byte-identical to the run it audits; its ors are built by
    // the export seam on the survivor list. Defined mode branch.
    if (parameters.compressor_mode) return;

    std::vector<std::string> newRows;
    {
        std::lock_guard<std::mutex> lock(theoremListMutex);
        for (const auto& tpl : globalTheoremList) {
            const std::string& thm = std::get<0>(tpl);
            // Proved-not-broadcast rows never feed the or construction —
            // the tier's contract is zero circulation; mark scanned so the
            // row is settled, then skip.
            if (std::get<1>(tpl) == "proved not broadcast") {
                orInRunScannedRows.insert(thm);
                continue;
            }
            if (orInRunScannedRows.insert(thm).second) newRows.push_back(thm);
        }
    }

    // Documentation mail for this seam's broadcasts: the ("theorem", []) origin
    // at main that every other proved theorem's broadcast carries
    // (updateGlobalSingleChapter). Without it a receiver documents the or
    // theorem only through the compaction's compilation/expansion pair, a
    // two-node cycle the chapter walker cannot ground.
    Mail theoremOriginMail;

    for (const std::string& thm : newRows) {
        const std::string companion = headSwitchOne(thm);
        if (companion.empty()) continue;

        // Canonical pair gate: a mirror row scanned later describes the
        // SAME disjunction with mirrored element order — constructing it
        // would double-mint or<N> (mirrored orders are distinct registry
        // identities). The shared ledger makes the first-scanned side the
        // one that builds.
        const std::pair<std::string, std::string> pairKey(
            std::min(thm, companion), std::max(thm, companion));
        if (orBuiltByPair.count(pairKey)) continue;

        const std::string orThm = constructOrTheorem(thm, companion);
        assert(!orThm.empty()
            && "constructOrTheoremsInRun: head-switched row must carry a negated premise");
        orBuiltByPair.emplace(pairKey, orThm);

        // String dedup: two DIFFERENT pairs can still fold to one or
        // theorem; the second registers nothing and broadcasts nothing.
        if (!theoremPublisher.appendGlobalTheorem(orThm, "or theorem", thm, companion)) continue;
        this->fullTheoremList.emplace_back(orThm, "or theorem", thm, companion);
        std::cout << "OR theorem constructed (in-run): " << orThm << std::endl;
        // The companion is subsumed by this or theorem from now on: a later
        // proof of it is neither registered nor broadcast, so its own logic
        // blocks have no work left and retire now.
        orInRunCompanions.insert(companion);
        theoremPublisher.retireSubsumedCompanion(companion);

        // Broadcast like any proved implication: the deferred-compaction
        // drain that follows this seam compiles the or theorem to its
        // implication compact and merges it into the root's mailOut (empty
        // level set; status-3 rule door at every receiver). A premise-free
        // or theorem has no implication-compact shape — registry+list only.
        if (startsWith(orThm, "(>[", 3)) {
            theoremPublisher.recordPendingCompaction(orThm, /*kySize=*/0, /*coreId=*/-1);
            if (parameters.trackHistory) {
                proofHistory.addOrigin(theoremOriginMail.exprOriginMap,
                    ExpressionWithValidity(orThm, "main"),
                    std::make_pair(std::string("theorem"),
                                   std::vector<ExpressionWithValidity>()),
                    parameters.max_origin_per_expr);
            }
        }
    }

    theoremPublisher.mergeTheoremOriginMailIntoRoot(theoremOriginMail);

    // A retired companion's walk owes the tree-wide prune. This seam runs
    // after the theorem drains' sweep, so it settles its own before the
    // barrier discharge reads isActive (D-451).
    if (deactivationSweepOwed) {
        deactivationSweepOwed = false;
        scopeLifecycle.deactivateRecursively();
    }

    // Close the reduced-or single-elimination closure over everything now
    // registered (including the ors this seam just minted) so the
    // single-exclusion emission's registry lookup can hard-assert on a
    // miss (D-268). Idempotent.
    orRuleCompiler.preMintReducedOrs();
}



}
