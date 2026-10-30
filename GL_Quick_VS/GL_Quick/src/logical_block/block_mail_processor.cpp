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
#include "block_mail_processor.hpp"
namespace gl {
/// @brief Bind inputs for block mail processor.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param blockIdentity Borrowed LogicBlockIdentity interface or state; must outlive this component.
/// @param body Borrowed Memory interface or state; must outlive this component.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @param mailLog Borrowed MailLog interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param permanentBodies Borrowed std::vector interface or state; must outlive this component.
/// @param simpleMapStore Borrowed SimpleMapStore interface or state; must outlive this component.
/// @param structuralCompiler Borrowed StructuralCompiler interface or state; must outlive this component.
/// @param proofHistory Borrowed ProofHistory interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
BlockMailProcessor::BlockMailProcessor(LogicBlockIdentity& blockIdentity, Memory& body, const CompiledDefinitionRegistry& definitions, MailLog& mailLog, const ProverParameters& parameters, std::vector<Memory*>& permanentBodies, SimpleMapStore& simpleMapStore, StructuralCompiler& structuralCompiler, ProofHistory& proofHistory) : blockIdentity(blockIdentity), body(body), definitions(definitions), mailLog(mailLog), parameters(parameters), permanentBodies(permanentBodies), simpleMapStore(simpleMapStore), structuralCompiler(structuralCompiler), proofHistory(proofHistory) {}

// Turn optimizations OFF for just this section
//#pragma optimize("", off)
/// @brief Test a circulating expression against the witness-class mail boundary.
///
/// @details Any ordinary integration witness refuses statement or compact-rule
/// mail, regardless of depth or owning block. Forced and algebraic names do not
/// refuse it; the caller still enforces validity and routing requirements.
/// Proof-history and internal replay channels do not use this restriction.
/// @param expression Complete expression text, including prefixed arguments.
/// @param body Owning block, retained in the shared call interface.
/// @return True exactly when no ordinary integration witness occurs.
/// @invariant Pure lexical policy; no permission store or origin-map read.
bool BlockMailProcessor::allowedForMail(StrSpan expression, const Memory& body) {
    (void)body;
    return !containsIntegrationWitness(expression);
}

/// @brief Compile a batch of established theorems to compact rule
///        carriers and store the batch ONCE in `store`'s mail log.
///
/// @details
/// Load-time (and between-warm-up-and-main) seeding of an LB grid with
/// theorems proved by earlier batches or supplied externally. Every
/// theorem is disintegrated, rebuilt in first-occurrence binder order
/// (I-4), compiled to its `(implication<N>[...])` compact, and deposited
/// as a main-scope mail statement with an empty level set (I-51) plus
/// its `originTag` / `compilation` origin rows. The batch is committed
/// into `store`'s `MailLog` chain — every LB that lists `store` among
/// its mail ancestors pulls it on its next phase-1 walk — and
/// self-injected into `store.mailIn` (an LB never pulls its own log),
/// with `store.hasWork` raised (wake door, I-153). Single-threaded,
/// before any pull.
///
/// Which LBs receive the batch is decided by `store` together with
/// the mail-edge registration (`collectMailAncestors`): the root
/// sentinel's log reaches every LB outside incubator mode and only the
/// anchor LB inside it (D-332);
/// the anchor LB's log reaches the anchor LB and every LB below it.
///
/// @param provedTheorems The theorems to seed (raw MPL, structural
///        operators already precompiled by the caller).
/// @param originTag Origin tag of the seed rows (`"broadcast"` for
///        theorems proved by earlier batches, `"externally provided
///        theorem"` for the hand-written pool).
/// @param store The LB whose log stores the batch and whose inbox is
///        self-injected: the root sentinel (`body`) or, for the
///        externals of an incubator batch, the anchor LB.
/// @param nestedStore Shape routing (incubator mode, the prior-theorems
///        batch): when non-null, a theorem that `anchorOnlyRule`
///        rejects — a nested rule whose premise LB lies below the
///        anchor — is stored in `nestedStore`'s log (the anchor LB's,
///        read by every LB below it) instead of `store`'s. Null = no
///        split (every theorem goes to `store`).
/// @invariant The batch's level sets are empty (I-51); only
///            `statements` + `exprOriginMap` are stored (D-137 rule 3).
///            Compacts are compiled in list order regardless of the
///            split (the `implication<N>` numbering is observable).
/// @see collectMailAncestors, incubatorAnchorLb, anchorOnlyRule,
///      MailLog::commit, mergeBatchIntoMailIn.
void BlockMailProcessor::broadcastTheorems(const std::vector<std::string>& provedTheorems,
                                           const std::string& originTag,
                                           Memory& store,
                                           Memory* nestedStore) {
    if (provedTheorems.empty()) return;

    // One batch per store; the compile order stays the list order (the
    // implication<N> numbering is observable), only the destination differs.
    Mail storeBatch;
    Mail nestedBatch;
    for (const std::string& thOriginal : provedTheorems) {
        const bool toNested = nestedStore != nullptr
            && !this->anchorOnlyRule(thOriginal);
        Mail& broadcastMail = toNested ? nestedBatch : storeBatch;

        // 1. Disintegrate the theorem to inspect its head
        std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> tempChain;
        std::string head = ce::disintegrateImplication(thOriginal, tempChain, this->definitions.configurations());

        std::string finalTheorem = thOriginal;
        std::string finalHead = head;
        std::vector<std::string> finalChain;
        finalChain.reserve(tempChain.size());
        for (const auto& t : tempChain) finalChain.push_back(std::get<0>(t));

        // 2. Rebuild the theorem via reconstructImplicationFullBind so its
        //    bound-variable lists are ALWAYS in first-occurrence order (the
        //    single binder rule, I-4). If the head is an uncompiled existence
        //    ("!(>"), compile it to existence<N> first; otherwise use the head
        //    as-is. Previously the reconstruction was nested inside the
        //    existence check, so a non-existence-head external was broadcast as
        //    the raw thOriginal with its binders untouched -- a non-occurrence-
        //    order binder then leaked into the `compilation` row's rest[0] and
        //    the verifier's check_compilation rejected it.
        std::string compiledHead = head;
        if (gl::startsWith(head, "!(>", 3)) {
            compiledHead = structuralCompiler.compileExpression(
                head);
        }

        finalTheorem = gl::ImplicationBuilder::reconstructImplication(finalChain, compiledHead);

        // Re-disintegrate so finalChain / finalHead reflect the rebuilt theorem.
        tempChain.clear();
        finalChain.clear();
        finalHead = ce::disintegrateImplication(finalTheorem, tempChain, this->definitions.configurations());
        for (const auto& t : tempChain) finalChain.push_back(std::get<0>(t));

        // 3. Queue the (potentially modified) theorem for broadcast.
        // ASIC 0.1 reshuffle: the legacy Mail::implications tuple is
        // removed; the external travels SOLELY as the inline compact
        // statement deposited just below (broadcastTheorems is
        // single-threaded at LOAD, so inline compile is race-free and
        // makes the external available from burst 1 — the deferred queue
        // would drain too late).

        if (parameters.trackHistory) {
            ExpressionWithValidity ev(finalTheorem, "main");
            auto originBroadcast = std::make_pair(originTag, std::vector<ExpressionWithValidity>());
            proofHistory.addOrigin(broadcastMail.exprOriginMap, ev, originBroadcast, (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
        }

        // ASIC 0.1 reshuffle: externals must ALSO travel as the D-76
        // compact (implication<N>[...]) statement, because the legacy
        // Mail::implications channel is being retired and the receiver
        // now recovers rules via status=3 disintegration of the compact
        // form. broadcastTheorems runs single-threaded at LOAD time
        // (pre-proveKernel), so the post-pool.join() deferred-compaction
        // drain (recordPendingCompaction) would deliver these a full
        // prove-entry too late — the externals must be installed by
        // burst 1. Compile INLINE here instead: single-threaded at this
        // site, so there is no I-28 race on the global implCounter /
        // compiledExpressions / compilationReuse.entries() state that forced
        // the parallel broadcast sites to defer. Shape mirrors the
        // deferred drain (compact statement at "main" + EMPTY compactLevels
        // + paired `compilation` origin).
        {
            const std::string compactImpl = structuralCompiler.compileImplicationToCompact(finalTheorem);
            // compactLevels MUST stay EMPTY here, as at the
            // deferred-compaction drain in proveKernel. The implication
            // rule always deposits std::set<int>(); via addToHashMemory the
            // deposited level run becomes the installed rule's levels, so a
            // non-empty set injects an extra level into every statement derived
            // from that rule, making the derived level-set size exceed
            // memoryBlock.level + 1. The allLevelsInvolved registration verdict
            // sealed in prover.hpp::dischargeToBeProved then turns false and the
            // drain refuses appendGlobalTheorem: the goal still closes (closure
            // is level-free), but the derived head is never recorded as proved
            // or broadcast.
            std::set<int> compactLevels;
            ExpressionWithValidity compactEv(compactImpl, "main");
            broadcastMail.statements.insert(std::make_pair(compactEv, compactLevels));
            if (parameters.trackHistory) {
                // I-52: cite the binary's CANONICAL reconstruction, not the
                // input finalTheorem. compileImplicationToCompact registers a
                // normalized premise order (and dedups alpha-equivalent bodies
                // to the first-seen entry), so finalTheorem's premise order
                // can differ from the registered elements even for a single
                // source; the verifier's check_compilation rebuilds from the
                // binary, so the citation must be the same reconstruction.
                // Mirror of the deferred-compaction drain in proveKernel.
                const std::string compactCore = extractExpressionUniversalSpan(StrSpan(compactImpl)).toStdString();
                auto cit = this->definitions.entries().find(compactCore);
                assert(cit != this->definitions.entries().end()
                    && "broadcastTheorems: compact form must resolve to a compiledExpressions entry");
                const std::vector<std::string>& elems = cit->second.elements;
                assert(!elems.empty()
                    && "broadcastTheorems: implication entry must have at least one element (head)");
                std::vector<std::string> canonicalKey(elems.begin(), elems.end() - 1);
                const std::string& canonicalHead = elems.back();
                const std::string canonicalOriginal = gl::ImplicationBuilder::reconstructImplicationFullBind(canonicalKey, canonicalHead);
                proofHistory.addOrigin(broadcastMail.exprOriginMap, compactEv,
                    std::make_pair("compilation", std::vector<ExpressionWithValidity>{ ExpressionWithValidity(canonicalOriginal, "main") }),
                    (parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr));
            }
        }
    }

    // New mail system (D-137): store the broadcast ONCE in `store`'s log --
    // every LB that lists `store` among its mail ancestors pulls it on the
    // normal ancestor walk -- and self-inject `store`, which never pulls its
    // own log. `store` is the root sentinel (reaching every LB, or only the
    // anchor LB in incubator mode -- collectMailAncestors) or, for an
    // incubator batch's externals, the anchor LB (reaching the anchor LB and
    // every LB below it). Single-threaded at LOAD, before any pull, so the
    // batch is in place for burst 1. broadcastMail carries only statements +
    // exprOriginMap, exactly what mergeBatchInto and commit keep.
    mergeBatchIntoMailIn(storeBatch, store.mailIn);
    this->mailLog.commit(&store, std::move(storeBatch));
    // WAKE DOOR 5 (D-194): the seed batch self-injects `store`'s mailIn
    // directly, so wake it here; every reader of its log is woken by mailPeek
    // on the next active-build (it sees the bumped commit count). A no-op on
    // burst 1 (LBs born dirty), load-bearing for the between-warm-up-and-main
    // broadcast where `store` may have gone quiescent.
    store.hasWork = true;
    // Shape-routed nested rules: the anchor LB's log (every LB below it) plus
    // the anchor LB's own inbox.
    if (nestedStore != nullptr && !nestedBatch.statements.empty()) {
        mergeBatchIntoMailIn(nestedBatch, nestedStore->mailIn);
        this->mailLog.commit(nestedStore, std::move(nestedBatch));
        nestedStore->hasWork = true;
    }

    std::cout << "Distributed knowledge to " << permanentBodies.size() << " memory blocks." << std::endl;
}

/// @brief Collect the mail ancestors an LB registers with `MailLog`
///        (the logs it pulls from), nearest first.
///
/// @details
/// Outside incubator mode this is the whole `parentMemory` chain up to
/// and including the root sentinel (I-57: an LB pulls from every direct
/// ancestor). In incubator mode the walk STOPS at the anchor LB: an LB
/// strictly below the anchor LB lists its ancestors up to and including
/// the anchor LB and never the root sentinel, so everything the root's
/// log carries — the in-run compact rule broadcasts, their
/// `"theorem"`-origin rows, the load-time batch of theorems proved by
/// earlier incubator batches — reaches only the anchor LB
/// (D-332). The anchor LB itself
/// keeps its root edge; the root sentinel has no ancestors.
///
/// @param lb  The LB being registered.
/// @param out Receives the ancestor list, nearest first (cleared first).
/// @invariant In incubator mode every non-root LB has the anchor LB on
///            its chain (every incubator conjecture starts with the
///            anchor premise); a chain without it asserts.
/// @see isAnchorLb, MailLog::registerLb, broadcastTheorems.
void BlockMailProcessor::collectMailAncestors(const Memory* lb,
                                              std::vector<const Memory*>& out) const {
    assert(lb != nullptr && "collectMailAncestors: null LB");
    out.clear();
    bool stoppedAtAnchor = false;
    for (const Memory* p = lb->parentMemory; p != nullptr; p = p->parentMemory) {
        out.push_back(p);
        if (parameters.incubator_mode && blockIdentity.isAnchorLb(*p)) {
            stoppedAtAnchor = true;
            break;
        }
    }
    // Incubator contract: every LB strictly below the anchor LB has the anchor
    // LB on its chain (every incubator conjecture starts with the anchor
    // premise). The root sentinel and the anchor LB itself walk to the root.
    assert((!parameters.incubator_mode || stoppedAtAnchor
            || lb->parentMemory == nullptr || blockIdentity.isAnchorLb(*lb))
        && "collectMailAncestors: incubator LB without an anchor LB on its chain");
}

/// @brief The anchor LB of an incubator grid — the root sentinel's one
///        child whose key carries the anchor prefix.
///
/// @details
/// Every incubator conjecture has the anchor as its first premise, so the
/// root sentinel has exactly one anchor child and every other LB hangs
/// below it. The walk over the root's children asserts exactly one
/// anchor LB (Rule 19). The result is the `store` of the externals seed
/// batch in incubator mode (`broadcastTheorems`).
///
/// @return The anchor LB (never null).
/// @invariant Callable only in incubator mode, after the grid is built.
/// @see isAnchorLb, broadcastTheorems, collectMailAncestors.
Memory* BlockMailProcessor::incubatorAnchorLb() const {
    assert(parameters.incubator_mode
        && "incubatorAnchorLb: only an incubator grid has a single anchor LB");
    Memory* anchorLb = nullptr;
    int anchorChildren = 0;
    this->simpleMapStore.forEachChild(&this->body,
        [&](const StrSpan& /*key*/, Memory* child) {
            if (blockIdentity.isAnchorLb(*child)) {
                anchorLb = child;
                ++anchorChildren;
            }
        });
    assert(anchorChildren == 1
        && "incubatorAnchorLb: the root sentinel must have exactly one anchor child");
    assert(anchorLb != nullptr);
    return anchorLb;
}

/// @brief True iff an incubator-derived theorem's rule can only fire
///        at the anchor LB, so its compact need not travel below it.
///
/// @details
/// The shape rule of D-332
/// (maintainer-directed). Anchor-only:
/// - a single-premise theorem `anchor → head` whose head is not a
///   positive anchor;
/// - a two-premise theorem whose second premise is an operator
///   application and whose head is an equality connecting one of that
///   operator's bound (non-anchor) arguments to an anchor digit — an
///   anchor slot value (the slots an equality can mention are the
///   numerals), e.g. `anchor → (in2[10,x,3] → (=[x,11]))`: the
///   operator premise is derivable at the anchor LB, so the rule
///   fires there.
/// Positive anchor-to-anchor bridges are NOT anchor-only: copied
/// source anchors exist in descendant LBs, so those LBs must receive
/// the bridge rule and derive the copied target by ordinary inference.
/// Every other nested shape (`anchor → (A → B)` with a premise that is
/// local only in a premise LB below the anchor, three or more premises)
/// is NOT anchor-only: its compact is stored in the anchor LB's log so
/// the premise LBs below the anchor receive it. Heap parse
/// (`ce::disintegrateImplication`), single-threaded seams only (load,
/// the post-join compaction drain).
///
/// @param theorem The theorem text (raw MPL, structural operators
///        precompiled).
/// @return True iff the rule is anchor-only.
/// @invariant The theorem's first premise is an anchor predicate —
///            this batch's, or an earlier incubator batch's for a
///            loaded theorem (bridged by the cross-anchor conjecture
///            `current anchor → previous anchor`); anything else
///            asserts.
/// @see broadcastTheorems, the deferred-compaction drain in
///      `proveKernel`.
bool BlockMailProcessor::anchorOnlyRule(const std::string& theorem) const {
    std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> chain;
    const std::string head = ce::disintegrateImplication(theorem, chain, this->definitions.configurations());
    assert(!chain.empty() && "anchorOnlyRule: a theorem without premises");
    // The first premise is an anchor: this batch's, or an earlier incubator
    // batch's for a loaded theorem (the cross-anchor bridge conjecture
    // `current anchor -> previous anchor` derives that premise at the anchor
    // LB). The earlier anchor need not be a core expression of this batch.
    const std::string& anchorPremise = std::get<0>(chain[0]);
    assert(anchorPremise.rfind("(Anchor", 0) == 0
        && "anchorOnlyRule: an incubator-derived theorem starts with an anchor premise");
    if (chain.size() == 1) return head.rfind("(Anchor", 0) != 0;
    if (chain.size() != 2) return false;

    const std::string& premise2 = std::get<0>(chain[1]);
    const bool operatorPremise = premise2.size() > 2 && premise2[0] == '('
        && premise2[1] != '=' && premise2[1] != '>';
    if (!operatorPremise) return false;
    if (head.rfind("(=[", 0) != 0) return false;
    const std::vector<std::string> headArgs = ce::getArgs(head);
    if (headArgs.size() != 2) return false;

    // The anchor digits are the anchor's slot values that an equality can
    // mention: every slot of an incubator anchor other than the carrier set
    // and the function symbols, which are never equated. Reading the slot
    // values off the premise (not the definition sets) keeps the rule valid
    // for an earlier batch's anchor that this batch's configuration does not
    // define.
    const std::vector<std::string> anchorArgs = ce::getArgs(anchorPremise);
    const std::set<std::string> anchorArgSet(anchorArgs.begin(), anchorArgs.end());
    const std::set<std::string>& digits = anchorArgSet;
    std::set<std::string> operatorOutputs;
    for (const std::string& a : ce::getArgs(premise2)) {
        if (anchorArgSet.count(a) == 0) operatorOutputs.insert(a);
    }
    const auto connects = [&](const std::string& x, const std::string& d) {
        return operatorOutputs.count(x) != 0 && digits.count(d) != 0;
    };
    return connects(headArgs[0], headArgs[1]) || connects(headArgs[1], headArgs[0]);
}
}
