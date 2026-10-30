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
#include "contradiction_demand.hpp"
#include "integration_processor.hpp"
#include "implication_subproofs.hpp"
#include "../infra/rt_tracker.hpp"
#include "../proof/worker_context.hpp"

#include <cstring>

namespace gl {

/// @brief Bind inputs for the contradiction-scope demand.
/// @details Construction stores explicit references; block storage and all
/// invocation lifetimes remain with their owners.
/// @param definitions Read-only compiled definitions (operator configs).
/// @param parameters Borrowed ProverParameters; must outlive this component.
/// @param integrationProcessor The shared head match and the integration
///        door; must outlive this component.
/// @param implicationSubproofs The candidate verdict and the park machinery;
///        must outlive this component.
/// @return None; construction binds the supplied references.
ContradictionDemand::ContradictionDemand(const CompiledDefinitionRegistry& definitions,
        const ProverParameters& parameters, IntegrationProcessor& integrationProcessor,
        ImplicationSubproofs& implicationSubproofs)
    : definitions(definitions), parameters(parameters),
      integrationProcessor(integrationProcessor), implicationSubproofs(implicationSubproofs) {}

/// @brief Whether a registered contradiction scope takes part in a drain.
///
/// @details
/// A scope under a Phase-3-ignored OR root, and a scope filtered for the
/// end-of-burst wipe or below a filtered scope (a discharged scope and its
/// subtree), are closing: they are neither served nor bootstrapped, and a
/// demand staged for them is not run.
///
/// @param mb      Owning block.
/// @param scopeId A registered contradiction scope.
/// @return True iff the scope is not Phase-3 ignored and neither it nor an
///         ancestor is filtered.
/// @invariant Pure.
/// @see drainContradictionDemands.
bool ContradictionDemand::participates(const Memory& mb, NameId scopeId) const {
    if (mb.phase3IgnoresScope(scopeId)) return false;
    for (NameId v = scopeId; v != 0; v = mb.nameMap.parentOf(v))
        if (mb.lbMemory.intValidityNamesToFilter.lookup(v) != 0) return false;
    return true;
}

/// @brief The head-match input of a fact: its negation with every argument
///        `u_`-prefixed, on the string-scratch arena.
///
/// @param fact    Clean fact text.
/// @param strings String-scratch arena; the caller's frame keeps the result.
/// @return The input text.
/// @invariant Zero heap; mints nothing.
StrSpan ContradictionDemand::negatedInput(StrSpan fact, ScratchArena& strings) {
    StrSpan negated;
    if (fact.len > 0 && fact.ptr[0] == '!') {
        negated = StrSpan(fact.ptr + 1, fact.len - 1);
    } else {
        char* b = strings.allocBytes(fact.len + 1);
        b[0] = '!';
        std::memcpy(b + 1, fact.ptr, static_cast<size_t>(fact.len));
        negated = StrSpan(b, fact.len + 1);
    }
    return StrSpan(gl::DefinitionExpansion::prefixArgumentsWithUScratch(strings, negated));
}

/// @brief File one known fact in the fact index under every key of
///        `forEachFactKey`; a fact already filed is a defined repeat.
///
/// @param mb   Owning block.
/// @param fact Packed `(original, scope)` key of a known statement.
/// @return Nothing.
/// @invariant Mints only into the derived index.
/// @see forEachFactKey, ContradictionFactIndex.
void ContradictionDemand::fileFact(Memory& mb, int64_t fact) {
    ContradictionFactIndex& index = mb.lbMemory.contradictionFacts;
    if (!index.addFact(fact)) return;
    const unsigned slot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
    ScratchArena& strings = scratchArenas().forSlot(slot);
    ScratchScope frame(strings);
    const StrSpan view = mb.nameMap.decodeView(Codec<StatementKey>::decode(fact).orig);
    const ScratchString text = ScratchString::copyFrom(strings, view.ptr, view.len);
    forEachFactKey(StrSpan(text), strings, [&](StrSpan key) { index.fileUnder(key, fact); });
}

/// @brief Match one fact against the rule chains of an id range for one
///        contradiction scope and decide every candidate.
///
/// @details
/// The fact's negation with every argument `u_`-prefixed is the head-match
/// input; the chains whose head matches it (`collectHeadMatchChains`,
/// ascending) are decided in turn (`decideCandidate`).
///
/// @param mb         Owning block; its head index is built.
/// @param scopeId    The contradiction scope; it sees the fact.
/// @param fact       Packed `(original, scope)` key of a known statement.
/// @param firstChain Smallest chain id to match; at least 1.
/// @param lastChain  Largest chain id to match; below @p firstChain the range
///                   is empty.
/// @return Nothing.
/// @invariant Zero heap.
/// @see IntegrationProcessor::collectHeadMatchChains, decideCandidate.
void ContradictionDemand::matchFact(Memory& mb, NameId scopeId, int64_t fact,
                                    int32_t firstChain, int32_t lastChain) {
    assert(firstChain >= 1 && "matchFact: chain ids start at 1");
    if (lastChain < firstChain) return;
    const unsigned slot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
    ScratchArena& strings = scratchArenas().forSlot(slot);
    ScratchScope frame(strings);
    // copyFrom (I-3): a candidate's park mints between the decisions.
    const StrSpan view = mb.nameMap.decodeView(Codec<StatementKey>::decode(fact).orig);
    const ScratchString text = ScratchString::copyFrom(strings, view.ptr, view.len);
    const StrSpan input = negatedInput(StrSpan(text), strings);
    StrSpan inputArgs[ExecutionParameters::MAX_ARITY];
    const int32_t inputArgN = getArgsSpans(input, inputArgs, ExecutionParameters::MAX_ARITY);
    if (!integrationProcessor.headMatchPossible(extractExpressionSpan(input), inputArgN)) return;
    const unsigned genSlot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
    ScratchArena& gen = genScratchArenas().forSlot(genSlot);
    const ArenaOffset mark = gen.cursor();
    DirtyState dirty = DirtyState::Clean;
    PagedVector<int32_t> chains(&gen, &dirty);
    const int32_t chainN = integrationProcessor.collectHeadMatchChains(input, mb, firstChain,
        lastChain, gen, chains);
    for (int32_t c = 0; c < chainN; ++c) decideCandidate(mb, scopeId, chains[c], input);
    chains.clear();
    gen.popTo(mark);
}

/// @brief Decide one (scope, chain, fact) candidate.
///
/// @details
/// A chain without a demandable compound premise can never demand and is
/// left at once. Otherwise each head constant is replaced by the input's
/// argument at its slot (`headMatchConstantSlots`); a premise that keeps a
/// bound argument, as CNFE's premise loop (D-534), or that reads
/// `!(=[Y,Y])` and can never hold (D-535), leaves the candidate out of
/// scope. The clean premises go to `deferContradictionDemand`, and a ready
/// compound is staged.
///
/// @param mb      Owning block.
/// @param scopeId The contradiction scope.
/// @param chainId A chain whose head matches @p input.
/// @param input   The fact's negation with every argument `u_`-prefixed.
/// @return Nothing.
/// @invariant Zero heap; a park mints its key and record.
/// @see ImplicationSubproofs::deferContradictionDemand.
void ContradictionDemand::decideCandidate(Memory& mb, NameId scopeId, int32_t chainId,
                                          StrSpan input) {
    // The chain's rule ids, zero-copy into a stack buffer.
    int32_t ids[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
    int32_t count = 0;
    {
        const StrSpan kb = mb.overallHashMemory.originals.inner().keyAt(chainId);
        const IdVecKeyView kv = viewIdVecKey(kb);
        count = kv.count();
        assert(count >= 1 && count <= ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
            && "decideCandidate: a matched chain has a head and at most 63 premises");
        for (int32_t i = 0; i < count; ++i) ids[i] = kv.idAt(i);
    }
    // Only a demandable compound premise can ever be demanded.
    bool demandable = false;
    for (int32_t k = 0; k + 1 < count && !demandable; ++k)
        demandable = implicationSubproofs.isDemandableCompound(mb.ruleInterner.decodeView(ids[k]));
    if (!demandable) return;

    const unsigned slot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
    ScratchArena& strings = scratchArenas().forSlot(slot);
    ScratchScope frame(strings);
    const StrSpan inputName = extractExpressionSpan(input);
    StrSpan inputArgs[ExecutionParameters::MAX_ARITY];
    const int32_t inputArgN = getArgsSpans(input, inputArgs, ExecutionParameters::MAX_ARITY);
    const bool negated = input.len > 0 && input.ptr[0] == '!';
    // The rule interner's views stay valid up to the verdict below: nothing
    // before it mints rules.
    const StrSpan headView = mb.ruleInterner.decodeView(ids[count - 1]);
    int32_t constantSlots[2] = { -1, -1 };
    const int32_t constantCount = integrationProcessor.headMatchConstantSlots(headView, inputName,
        negated, inputArgs, inputArgN, constantSlots);
    StrSpan headArgs[ExecutionParameters::MAX_ARITY];
    const int32_t headArgN = getArgsSpans(headView, headArgs, ExecutionParameters::MAX_ARITY);
    assert(headArgN == inputArgN && "decideCandidate: the matched head has another arity");
    (void)headArgN;
    StrReplacement binding[2] = {};
    for (int32_t t = 0; t < constantCount; ++t)
        binding[t] = { headArgs[constantSlots[t]], inputArgs[constantSlots[t]] };

    StrSpan premises[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
    int32_t premiseN = 0;
    for (int32_t k = 0; k + 1 < count; ++k) {
        const ScratchString instantiated = replaceKeysScratch(strings,
            mb.ruleInterner.decodeView(ids[k]), binding, constantCount);
        const StrSpan premise(instantiated);
        StrSpan args[ExecutionParameters::MAX_ARITY];
        const int32_t argN = getArgsSpans(premise, args, ExecutionParameters::MAX_ARITY);
        // A premise the head leaves unbound names an object the fact does not
        // determine (D-534).
        for (int32_t a = 0; a < argN; ++a)
            if (!startsWithSpan(args[a], "u_", 2)) return;
        // A premise reading `!(=[Y,Y])` can never hold (D-535).
        if (startsWithSpan(premise, "!(=[", 4) && argN == 2 && equalSpans(args[0], args[1])) return;
        premises[premiseN++] = StrSpan(gl::DefinitionExpansion::removeUPrefixScratch(strings, premise));
    }
    int32_t ready = -1;
    if (implicationSubproofs.deferContradictionDemand(premises, premiseN, mb, scopeId, ready)
            == ImplicationSubproofs::ContradictionDemandVerdict::Ready)
        ImplicationSubproofs::stageContradictionDemand(mb, premises[ready], scopeId);
}

/// @brief Serve every registered contradiction scope of a block: decide the
///        new (scope, chain, fact) triples and integrate the ready demands.
///
/// @details
/// Runs in step 4b of `standardProcessing`, after the implication-gate drain
/// (whose revival of parked candidates stages ready demands) and the trial
/// drain; single-threaded block work. With no registered scope it only
/// empties its arrival queue and frees the fact index; under
/// `ban_disintegration` (the compressor) it empties both queues and does
/// nothing, like CNFE. Otherwise, over the participating scopes (registered,
/// not Phase-3 ignored, not filtered for wiping) in decoded order:
/// 1. The fact index is rebuilt when unbuilt: every known statement some
///    participating scope sees is filed under its sign-flipped masked keys.
/// 2. Each newly registered scope matches every fact it sees against the
///    chains `1..count` and becomes bootstrapped.
/// 3. Each staged arrival still known is filed and matched against the
///    chains `1..mark` for every earlier bootstrapped scope that sees it.
/// 4. Each chain above the mark is matched, through its masked key, against
///    the filed facts every earlier bootstrapped scope sees.
/// 5. The mark becomes the chain count at entry, and the staged ready
///    demands are integrated in decoded (compound, scope) order, each once.
/// Work the integrations cause (facts, chains, scopes) waits for the next
/// drain.
///
/// @param mb Owning block.
/// @return Nothing.
/// @invariant Consumes `pendingContradictionArrivals` and
///            `pendingContradictionDemands`; opens scopes only through
///            `prepareIntegration`; zero heap.
/// @see ImplicationSubproofs::deferContradictionDemand,
///      ImplicationSubproofs::drainSubproofArrivals, ContradictionFactIndex.
void ContradictionDemand::drainContradictionDemands(Memory& mb) {
    auto& scopes = mb.lbMemory.contradictionScopes;
    auto& arrivals = mb.pendingContradictionArrivals;
    auto& demands = mb.pendingContradictionDemands;
    ContradictionFactIndex& index = mb.lbMemory.contradictionFacts;
    if (scopes.empty()) {
        // A demand is staged only for a parked candidate of a registered
        // scope, and a scope leaves the registry with its parks.
        assert(demands.size() == 0
            && "drainContradictionDemands: a staged demand without a registered scope");
        arrivals.clear();
        if (index.built()) index.clear();
        return;
    }
    if (parameters.ban_disintegration) {
        arrivals.clear();
        demands.clear();
        return;
    }
    RT_SCOPE_HERE("CONTRA_DEMAND");
    HashMemory& hm = mb.overallHashMemory;
    integrationProcessor.ensureOriginalsHeadIndexBuilt(mb);
    const int32_t chainsAtStart = hm.originals.count();
    const int32_t oldMark = hm.contradictionChainMark;
    assert(oldMark >= 0 && oldMark <= chainsAtStart
        && "drainContradictionDemands: chain floor above the chain count");

    const unsigned slot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
    ScratchArena& gen = genScratchArenas().forSlot(slot);
    ScratchArena& strings = scratchArenas().forSlot(slot);
    ScratchScope genFrame(gen), stringFrame(strings);
    DirtyState dirty = DirtyState::Clean;

    // The participating scopes in decoded order: the newly registered ones
    // (bootstrapped below) and the ones earlier drains served.
    PagedVector<NameId> fresh(&gen, &dirty);
    PagedVector<NameId> served(&gen, &dirty);
    {
        PagedVector<NameId> live(&gen, &dirty);
        for (int32_t id = 1; id <= scopes.count(); ++id) {
            const NameId s = static_cast<NameId>(scopes.decodeKey(id));
            if (participates(mb, s)) live.push_back(s);
        }
        const auto less = [&](int32_t a, int32_t b) {
            return compareSpans(mb.nameMap.decodeView(live[a]), mb.nameMap.decodeView(live[b])) < 0;
        };
        ChunkSortedOrdinals<decltype(less)> order(gen, live.size(), 0, less);
        for (int32_t i = 0; i < live.size(); ++i) {
            const NameId s = live[order.next()];
            const int32_t* state = scopes.find(static_cast<int64_t>(s));
            assert(state != nullptr && (*state == 0 || *state == 1)
                && "drainContradictionDemands: a registered scope with an unknown state");
            if (*state == 0) fresh.push_back(s);
            else served.push_back(s);
        }
    }
    const auto sees = [&](NameId scope, NameId factScope) {
        return factScope == scope || mb.nameMap.isStrictAncestor(factScope, scope);
    };
    const auto seenByAny = [&](NameId factScope) {
        for (int32_t i = 0; i < fresh.size(); ++i) if (sees(fresh[i], factScope)) return true;
        for (int32_t i = 0; i < served.size(); ++i) if (sees(served[i], factScope)) return true;
        return false;
    };

    // 1. The fact index.
    if (!index.built()) {
        index.clear();
        for (int32_t id = 1; id <= mb.intKnownStatements.count(); ++id) {
            const StatementKey k = mb.intKnownStatements.decodeKey(id);
            if (seenByAny(k.validity)) fileFact(mb, packStatementKey(k.orig, k.validity));
        }
        index.markBuilt();
    }

    // 2. Bootstrap: every fact a new scope sees against every chain.
    for (int32_t i = 0; i < fresh.size(); ++i) {
        const NameId s = fresh[i];
        PagedVector<int64_t> seen(&gen, &dirty);
        for (int32_t id = 1; id <= mb.intKnownStatements.count(); ++id) {
            const StatementKey k = mb.intKnownStatements.decodeKey(id);
            if (sees(s, k.validity)) seen.push_back(packStatementKey(k.orig, k.validity));
        }
        for (int32_t f = 0; f < seen.size(); ++f) {
            fileFact(mb, seen[f]);
            matchFact(mb, s, seen[f], 1, chainsAtStart);
        }
        scopes.upsert(static_cast<int64_t>(s), 1);
    }

    // 3. Arrivals against the chains the served scopes have seen.
    {
        TypedColdSet<int64_t> once(&gen, &dirty);
        for (int32_t i = 0; i < arrivals.size(); ++i) {
            const int64_t a = arrivals[i];
            if (once.contains(a)) continue;
            once.mint(a);
            const StatementKey k = Codec<StatementKey>::decode(a);
            // A row erased since it was staged is no fact.
            if (mb.intKnownStatements.find(k) == nullptr) continue;
            if (!seenByAny(k.validity)) continue;
            fileFact(mb, a);
            for (int32_t j = 0; j < served.size(); ++j)
                if (sees(served[j], k.validity)) matchFact(mb, served[j], a, 1, oldMark);
        }
        arrivals.clear();
    }

    // 4. New chains against the filed facts the served scopes see.
    if (served.size() > 0) {
        char keyBuf[kMaxMaskedHeadKeyBytes];
        for (int32_t c = oldMark + 1; c <= chainsAtStart; ++c) {
            const StrSpan key = integrationProcessor.maskedKeyOfChain(mb, c, keyBuf, kMaxMaskedHeadKeyBytes);
            if (key.len == 0) continue;
            PagedVector<int64_t> facts(&gen, &dirty);
            index.factsWithKey(key, [&](int64_t f) { facts.push_back(f); });
            for (int32_t f = 0; f < facts.size(); ++f) {
                const StatementKey k = Codec<StatementKey>::decode(facts[f]);
                // A filed fact erased since is no fact.
                if (mb.intKnownStatements.find(k) == nullptr) continue;
                for (int32_t j = 0; j < served.size(); ++j) {
                    if (!sees(served[j], k.validity)) continue;
                    ScratchScope candidateFrame(strings);
                    const StrSpan view = mb.nameMap.decodeView(k.orig);
                    const ScratchString text = ScratchString::copyFrom(strings, view.ptr, view.len);
                    decideCandidate(mb, served[j], c, negatedInput(StrSpan(text), strings));
                }
            }
        }
    }
    hm.contradictionChainMark = chainsAtStart;

    // 5. The ready demands, once each, in decoded (compound, scope) order.
    if (demands.size() == 0) return;
    PagedVector<int64_t> ready(&gen, &dirty);
    {
        TypedColdSet<int64_t> once(&gen, &dirty);
        for (int32_t i = 0; i < demands.size(); ++i) {
            if (once.contains(demands[i])) continue;
            once.mint(demands[i]);
            ready.push_back(demands[i]);
        }
        demands.clear();
    }
    const auto less = [&](int32_t a, int32_t b) {
        const StatementKey ka = Codec<StatementKey>::decode(ready[a]);
        const StatementKey kb = Codec<StatementKey>::decode(ready[b]);
        const int c = compareSpans(mb.templateInterner.decodeView(ka.orig),
            mb.templateInterner.decodeView(kb.orig));
        return c != 0 ? c < 0 : compareSpans(mb.nameMap.decodeView(ka.validity),
            mb.nameMap.decodeView(kb.validity)) < 0;
    };
    ChunkSortedOrdinals<decltype(less)> order(gen, ready.size(), 0, less);
    PagedVector<int64_t> ordered(&gen, &dirty);
    for (int32_t i = 0; i < ready.size(); ++i) ordered.push_back(ready[order.next()]);
    for (int32_t i = 0; i < ordered.size(); ++i) {
        const StatementKey k = Codec<StatementKey>::decode(ordered[i]);
        assert(scopes.contains(static_cast<int64_t>(k.validity))
            && "drainContradictionDemands: a demand at an unregistered scope");
        // A scope closing since its candidate was decided gets nothing.
        if (!participates(mb, k.validity)) continue;
        ScratchScope demandFrame(strings);
        // copyFrom (I-3): the integration mints templates and names.
        const StrSpan compoundView = mb.templateInterner.decodeView(k.orig);
        const ScratchString compound = ScratchString::copyFrom(strings, compoundView.ptr, compoundView.len);
        const StrSpan scopeView = mb.nameMap.decodeView(k.validity);
        const ScratchString scope = ScratchString::copyFrom(strings, scopeView.ptr, scopeView.len);
        // Every argument of the compound is fixed: the sorted-unique run is
        // prepareIntegration's unchangeable set, as for a CNFE premise.
        StrSpan args[ExecutionParameters::MAX_ARITY];
        const int32_t argN = getArgsSpans(StrSpan(compound), args, ExecutionParameters::MAX_ARITY);
        StrSpan argsSet[ExecutionParameters::MAX_ARITY];
        int32_t argsSetN = 0;
        for (int32_t a = 0; a < argN; ++a) {
            int32_t lo = 0, hi = argsSetN;
            bool duplicate = false;
            while (lo < hi) {
                const int32_t mid = (lo + hi) / 2;
                const int cmp = compareSpans(argsSet[mid], args[a]);
                if (cmp == 0) { duplicate = true; break; }
                if (cmp < 0) lo = mid + 1; else hi = mid;
            }
            if (duplicate) continue;
            for (int32_t z = argsSetN; z > lo; --z) argsSet[z] = argsSet[z - 1];
            argsSet[lo] = args[a];
            ++argsSetN;
        }
        RT_SCOPE_HERE("CONTRA_DEMAND_PREPARE");
        integrationProcessor.prepareIntegration(StrSpan(compound), argsSet, argsSetN, mb,
            StrSpan(scope), StrSpan("", 0));
    }
}

}
