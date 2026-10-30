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
#include "integration_processor.hpp"
namespace gl {
/// @brief Bind inputs for integration processor.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param admissionPolicy Borrowed AdmissionPolicy interface or state; must outlive this component.
/// @param canonicalTemplates Borrowed CanonicalTemplateMatcher interface or state; must outlive this component.
/// @param canonicalizer Borrowed Canonicalizer interface or state; must outlive this component.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @param integrationBuilder Borrowed IntegrationInstructionBuilder interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param scopeTypes Borrowed ScopeTypeResolver interface or state; must outlive this component.
/// @param admissionMemory Borrowed AdmissionMemory interface or state; must outlive this component.
/// @param disintegrationProcessor Borrowed DisintegrationProcessor interface or state; must outlive this component.
/// @param hashRuleInstaller Borrowed HashRuleInstaller interface or state; must outlive this component.
/// @param implicationSubproofs Borrowed ImplicationSubproofs interface or state; must outlive this component.
/// @param rejectedSourceReplay Borrowed RejectedSourceReplay interface or state; must outlive this component.
/// @param statementMemory Borrowed StatementMemory interface or state; must outlive this component.
/// @param unaryIntegrationProcessor Borrowed UnaryIntegrationProcessor interface or state; must outlive this component.
/// @param compactRuleProcessor Borrowed CompactRuleProcessor interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
IntegrationProcessor::IntegrationProcessor(AdmissionPolicy& admissionPolicy, CanonicalTemplateMatcher& canonicalTemplates, Canonicalizer& canonicalizer, const CompiledDefinitionRegistry& definitions, IntegrationInstructionBuilder& integrationBuilder, const ProverParameters& parameters, ScopeTypeResolver& scopeTypes, AdmissionMemory& admissionMemory, DisintegrationProcessor& disintegrationProcessor, HashRuleInstaller& hashRuleInstaller, ImplicationSubproofs& implicationSubproofs, RejectedSourceReplay& rejectedSourceReplay, StatementMemory& statementMemory, UnaryIntegrationProcessor& unaryIntegrationProcessor, CompactRuleProcessor& compactRuleProcessor) : admissionPolicy(admissionPolicy), canonicalTemplates(canonicalTemplates), canonicalizer(canonicalizer), definitions(definitions), integrationBuilder(integrationBuilder), parameters(parameters), scopeTypes(scopeTypes), admissionMemory(admissionMemory), disintegrationProcessor(disintegrationProcessor), hashRuleInstaller(hashRuleInstaller), implicationSubproofs(implicationSubproofs), rejectedSourceReplay(rejectedSourceReplay), statementMemory(statementMemory), unaryIntegrationProcessor(unaryIntegrationProcessor), compactRuleProcessor(compactRuleProcessor) {}

/// @brief Whether any rule head can match an input under CNFE's head match,
///        judged by the input's operator alone.
///
/// @details
/// CNFE accepts a chain only when its head names the input's operator, that
/// operator has a config, and each of the head's one or two constants sits at
/// a slot whose definition set is `(1)`. Both config facts depend on the
/// operator alone, so an input whose operator has no config or no `(1)` slot
/// cannot match any chain and its probe is skipped before any chain is
/// touched. Shared by CNFE and the contradiction-scope demand.
///
/// @param inputName The input's operator name (`extractExpressionSpan`).
/// @param inputArgN The input's argument count.
/// @return True iff the operator has a config with at least one `(1)` slot
///         among the first @p inputArgN slots.
/// @invariant Pure; allocates nothing.
/// @see checkNecessityForEquality, collectHeadMatchChains.
bool IntegrationProcessor::headMatchPossible(StrSpan inputName, int32_t inputArgN) const {
    const ce::CoreExpressionConfig* inputCfg = this->definitions.coreConfig(inputName);
    if (inputCfg == nullptr) return false;
    for (int32_t i = 0; i < inputArgN; ++i) {
        const std::pair<std::string, bool>* ds = gl::CompiledDefinitionRegistry::findDefinitionSet(inputCfg, i + 1);
        if (ds != nullptr && equalSpans(StrSpan(ds->first), StrSpan("(1)", 3))) return true;
    }
    return false;
}

/// @brief The masked head key under which CNFE's head match indexes one
///        rule chain, or an empty span when no input can ever match it.
///
/// @details
/// A chain is acceptable when its head has one constant (a non-`u_`
/// argument), or two constants beside at least one other argument, the
/// head's operator has a config, and each constant's slot has the definition
/// set `(1)`. Its key is `maskedHeadKey` of the head with the constant slots
/// masked; the key carries the head's sign. These are the per-chain checks
/// that depend on the chain alone; the input side is probed with the same key
/// shape (`collectHeadMatchChains`). The contradiction-scope demand reads the
/// key of a newly installed chain to find the facts it can match.
///
/// @param mb      The owning LB (its `originals` and `ruleInterner`).
/// @param chainId The chain id in `overallHashMemory.originals`; `>= 1`.
/// @param keyBuf  Caller buffer receiving the key bytes.
/// @param cap     Capacity of @p keyBuf; an overflow asserts.
/// @return The key as a span over @p keyBuf, or an empty span.
/// @invariant Reads only; mints nothing.
/// @see ensureOriginalsHeadIndexBuilt, maskedHeadKey, OriginalsHeadIndex.
StrSpan IntegrationProcessor::maskedKeyOfChain(Memory& mb, int32_t chainId, char* keyBuf,
                                               int32_t cap) const {
    assert(chainId >= 1 && chainId <= mb.overallHashMemory.originals.count()
        && "maskedKeyOfChain: chain id outside the rule-chain map");
    const StrSpan kb = mb.overallHashMemory.originals.inner().keyAt(chainId);
    const IdVecKeyView kv = viewIdVecKey(kb);
    const int32_t n = kv.count();
    if (n == 0) return StrSpan("", 0);   // an empty chain has no head; never matched
    const StrSpan headView = mb.ruleInterner.decodeView(kv.idAt(n - 1));
    const StrSpan headName = extractExpressionSpan(headView);
    StrSpan headArgs[ExecutionParameters::MAX_ARITY];
    const int32_t headArgN = getArgsSpans(headView, headArgs,
                                          ExecutionParameters::MAX_ARITY);
    int32_t constantSlots[2] = { -1, -1 };
    int32_t constantCount = 0;
    for (int32_t i = 0; i < headArgN; ++i) {
        const bool isVar = (headArgs[i].len >= 2 && headArgs[i].ptr[0] == 'u'
                            && headArgs[i].ptr[1] == '_');
        if (isVar) continue;
        if (constantCount < 2) constantSlots[constantCount] = i;
        ++constantCount;
    }
    if (constantCount != 1 && !(constantCount == 2 && headArgN > 2))
        return StrSpan("", 0);
    const ce::CoreExpressionConfig* cfg = this->definitions.coreConfig(headName);
    if (cfg == nullptr) return StrSpan("", 0);
    for (int32_t t = 0; t < constantCount; ++t) {
        const std::pair<std::string, bool>* ds =
            gl::CompiledDefinitionRegistry::findDefinitionSet(cfg, constantSlots[t] + 1);
        if (ds == nullptr || !equalSpans(StrSpan(ds->first), StrSpan("(1)", 3)))
            return StrSpan("", 0);
    }
    // The head's sign is part of the key: a head concluding the negation of
    // a deposit never matches it.
    const bool negatedHead = headView.len > 0 && headView.ptr[0] == '!';
    return maskedHeadKey(headName, negatedHead, headArgs, headArgN,
                         constantSlots[0], keyBuf, cap, constantSlots[1]);
}

/// @brief Bring the block's derived rule-head index up to date with its
///        rule-chain map.
///
/// @details
/// Extends `overallHashMemory.originalsHeadIndex` over the chains installed
/// since its last build: each chain under its head operator and, when
/// `maskedKeyOfChain` accepts it, under its masked head key. Every consumer of
/// the masked family builds it through this one function, so the family holds
/// exactly the keys CNFE's verdict accepts.
///
/// @param mb The owning LB.
/// @return Nothing.
/// @invariant Mints only into the index's own derived interners.
/// @see OriginalsHeadIndex::ensureBuilt, maskedKeyOfChain.
void IntegrationProcessor::ensureOriginalsHeadIndexBuilt(Memory& mb) const {
    OriginalsHeadIndex& headIndex = mb.overallHashMemory.originalsHeadIndex;
    // One key buffer for the builder: the index copies a key before the next
    // callback.
    char keyBuf[kMaxMaskedHeadKeyBytes];
    headIndex.ensureBuilt(mb.overallHashMemory.originals.count(),
        [&](int32_t id) -> StrSpan {
            const StrSpan kb = mb.overallHashMemory.originals.inner().keyAt(id);
            const IdVecKeyView kv = viewIdVecKey(kb);
            const int32_t n = kv.count();
            if (n == 0) return StrSpan("", 0);   // an empty chain has no head; never matched
            return extractExpressionSpan(mb.ruleInterner.decodeView(kv.idAt(n - 1)));
        },
        [&](int32_t id) -> StrSpan {
            return maskedKeyOfChain(mb, id, keyBuf, kMaxMaskedHeadKeyBytes);
        });
}

/// @brief Serve one near-miss of a goal's flat integration rule: install
///        the goal-driven marked-head implications for the missing
///        expression (D-574).
///
/// @details
/// Called by `AdmissionMemory::drainAdmissionKeysOrdis2` for a demand record
/// whose variant was a goal near-miss (`Ordis2DemandRecord::goalNearMiss`),
/// in canonical firing order at the post-fixpoint seam. The rule search
/// (`forEachGoalDemandVariant`) yields the marked-head implications of the
/// rules that can conclude @p expression; each is installed at @p validity
/// (`HashRuleInstaller::installGoalDemandVariant`) once:
/// `LbMemory::goalDemandInstalls` holds the rule owner (identity text,
/// scope) of every implication installed, since a marker LMV run appends
/// without dedup. Under `ban_disintegration` (the compressor) nothing is
/// served, as in CNFE: the witness a demand wakes needs the disintegration
/// the compressor bans.
///
/// @param mb         The owning LB (single-threaded seam, I-83).
/// @param expression The ground missing expression; span stable across
///                   the call.
/// @param validity   The scope the near-miss fired at, an existing scope
///                   name; span stable across the call.
/// @return Nothing.
/// @invariant Each (implication, scope) pair is installed at most once per
///            block; 0% heap.
/// @see forEachGoalDemandVariant, AdmissionMemory::drainAdmissionKeysOrdis2.
void IntegrationProcessor::serveGoalNearMiss(Memory& mb, StrSpan expression, StrSpan validity) {
    if (parameters.ban_disintegration) return;
    RT_SCOPE_HERE("GOAL_NEAR_MISS");
    ensureOriginalsHeadIndexBuilt(mb);
    const unsigned strSlot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : scratchArenas().slotCount() - 1;
    const unsigned genSlot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : genScratchArenas().slotCount() - 1;
    ScratchArena& strArena = scratchArenas().forSlot(strSlot);
    ScratchArena& genArena = genScratchArenas().forSlot(genSlot);
    ScratchScope strFrame(strArena);
    forEachGoalDemandVariant(mb, expression, validity, strArena, genArena,
        [&](const StrSpan* key, int32_t keyN, StrSpan markedArg, StrSpan identity) {
            const RuleOwner installed = packRuleOwner(mb.ruleInterner.encode(identity),
                mb.nameMap.encode(validity));
            if (mb.lbMemory.goalDemandInstalls.contains(installed)) return;
            mb.lbMemory.goalDemandInstalls.mint(installed);
            hashRuleInstaller.installGoalDemandVariant(mb, key, keyN, markedArg, identity, validity);
        });
}

/// @brief Collect the rule chains whose head matches an input under CNFE's
///        head match, in ascending chain-id order.
///
/// @details
/// One probe of the masked family per argument slot of the input, and one
/// per pair of slots when a third slot remains; each probe carries the
/// input's sign. The answers are disjoint (a key fixes its set of masked
/// slots); their union restricted to `[firstChainId, lastChainId]` is sorted
/// ascending and appended to @p chainIdx. The index must be built
/// (`ensureOriginalsHeadIndexBuilt`).
///
/// @param genericInput The input with every argument `u_`-prefixed (span over
///                     a caller-stable buffer).
/// @param mb           The owning LB.
/// @param firstChainId The smallest chain id to collect; `>= 1`.
/// @param lastChainId  The largest chain id to collect.
/// @param ocArena      The gen-scratch arena holding the candidate run; the
///                     byte-bump run is popped before the return.
/// @param chainIdx     Receives the chain ids (appended).
/// @return The number of chain ids appended.
/// @invariant Zero heap; reads the index only.
/// @see checkNecessityForEquality, maskedHeadKey.
int32_t IntegrationProcessor::collectHeadMatchChains(StrSpan genericInput, Memory& mb,
        int32_t firstChainId, int32_t lastChainId, ScratchArena& ocArena,
        PagedVector<int32_t>& chainIdx) const {
    assert(firstChainId >= 1 && "collectHeadMatchChains: chain ids start at 1");
    const OriginalsHeadIndex& headIndex = mb.overallHashMemory.originalsHeadIndex;
    assert(headIndex.built() && "collectHeadMatchChains: probe on an unbuilt head index");
    const StrSpan inputName = extractExpressionSpan(genericInput);
    StrSpan inputArgs[ExecutionParameters::MAX_ARITY];
    const int32_t inputArgN = getArgsSpans(genericInput, inputArgs,
                                           ExecutionParameters::MAX_ARITY);
    const bool negatedInput = genericInput.len > 0 && genericInput.ptr[0] == '!';
    char keyBuf[kMaxMaskedHeadKeyBytes];
    const ArenaOffset candMark = ocArena.cursor();
    int32_t candCap = 0;
    int32_t* cand = nullptr;
    int32_t candN = 0;
    const auto collect = [&](int32_t id) {
        if (id < firstChainId || id > lastChainId) return;
        if (candN == candCap) {
            // Grow the byte-bump candidate run (doubling; the old run stays
            // below the cursor until the pop).
            const int32_t newCap = candCap == 0 ? 64 : candCap * 2;
            int32_t* grown = reinterpret_cast<int32_t*>(ocArena.resolve(
                ocArena.alloc(newCap * static_cast<int32_t>(sizeof(int32_t)),
                              static_cast<int32_t>(alignof(int32_t)))));
            for (int32_t c = 0; c < candN; ++c) grown[c] = cand[c];
            cand = grown;
            candCap = newCap;
        }
        cand[candN++] = id;
    };
    for (int32_t slot = 0; slot < inputArgN; ++slot) {
        const StrSpan probe = maskedHeadKey(inputName, negatedInput, inputArgs, inputArgN,
                                            slot, keyBuf, kMaxMaskedHeadKeyBytes);
        headIndex.chainsWithMaskedKey(probe, collect);
    }
    if (inputArgN > 2) {
        for (int32_t slotA = 0; slotA < inputArgN; ++slotA) {
            for (int32_t slotB = slotA + 1; slotB < inputArgN; ++slotB) {
                const StrSpan probe = maskedHeadKey(inputName, negatedInput, inputArgs,
                    inputArgN, slotA, keyBuf, kMaxMaskedHeadKeyBytes, slotB);
                headIndex.chainsWithMaskedKey(probe, collect);
            }
        }
    }
    std::sort(cand, cand + candN);
    for (int32_t c = 0; c < candN; ++c) chainIdx.push_back(cand[c]);
    ocArena.popTo(candMark);
    return candN;
}

/// @brief Locate the constant slots of a head that the masked family
///        matched against an input, asserting the rest of the match.
///
/// @details
/// The index guarantees the verdict: the head names the input's operator with
/// the input's sign and arity, carries one constant (a non-`u_` argument), or
/// two beside at least one other argument, each at a `(1)` definition slot,
/// and equals the input at every other slot. This function finds the
/// constant slots and asserts each of those facts (Rule 19).
///
/// @param headView      The chain's head text.
/// @param inputName     The input's operator name.
/// @param negatedInput  Whether the input is negated.
/// @param inputArgs     The input's `u_`-prefixed argument spans.
/// @param inputArgN     Their count.
/// @param constantSlots Receives the 0-based constant slots (unused entries
///                      `-1`).
/// @return The number of constant slots, 1 or 2.
/// @invariant Pure; allocates nothing.
/// @see collectHeadMatchChains, checkNecessityForEquality.
int32_t IntegrationProcessor::headMatchConstantSlots(StrSpan headView, StrSpan inputName,
        bool negatedInput, const StrSpan* inputArgs, int32_t inputArgN,
        int32_t constantSlots[2]) const {
    const StrSpan headName = extractExpressionSpan(headView);
    // Structural check: the index guarantees the name and the sign.
    assert(equalSpans(headName, inputName)
        && "headMatchConstantSlots: the head index yielded a chain of another operator");
    assert((headView.len > 0 && headView.ptr[0] == '!') == negatedInput
        && "headMatchConstantSlots: the head index yielded a chain of the opposite sign");
    StrSpan headArgs[ExecutionParameters::MAX_ARITY];
    const int32_t headArgN = getArgsSpans(headView, headArgs,
                                          ExecutionParameters::MAX_ARITY);
    constantSlots[0] = -1;
    constantSlots[1] = -1;
    int32_t constantCount = 0;
    for (int32_t i = 0; i < headArgN; ++i) {
        const bool isVar = (headArgs[i].len >= 2 && headArgs[i].ptr[0] == 'u' && headArgs[i].ptr[1] == '_');
        if (isVar) continue;
        assert(constantCount < 2
            && "headMatchConstantSlots: the head index yielded a head with more than two constants");
        constantSlots[constantCount++] = i;
    }
    assert(headArgN == inputArgN
        && "headMatchConstantSlots: the head index yielded a head of another arity");
    assert((constantCount == 1 || (constantCount == 2 && headArgN > 2))
        && "headMatchConstantSlots: the head index yielded a head without one or two constants");
    const ce::CoreExpressionConfig* cfg = this->definitions.coreConfig(headName);
    assert(cfg != nullptr
        && "headMatchConstantSlots: the head index yielded an operator without a config");
    for (int32_t t = 0; t < constantCount; ++t) {
        // definitionSets keys are 1-based; findDefinitionSet builds the
        // decimal key on the stack and linear-scans (0% heap).
        const std::pair<std::string, bool>* ds =
            gl::CompiledDefinitionRegistry::findDefinitionSet(cfg, constantSlots[t] + 1);
        assert(ds != nullptr && equalSpans(StrSpan(ds->first), StrSpan("(1)", 3))
            && "headMatchConstantSlots: the head index yielded a constant outside a (1) slot");
        (void)ds;
    }
    (void)cfg;
    for (int32_t i = 0; i < headArgN; ++i) {
        if (i == constantSlots[0] || i == constantSlots[1]) continue;
        assert(equalSpans(headArgs[i], inputArgs[i])
            && "headMatchConstantSlots: the head index yielded a head differing outside its constants");
    }
    return constantCount;
}

//#pragma optimize("", off)

/// @brief Prepare the integration of every premise of every rule chain whose
///        head can establish a goal up to one or two `(1)`-typed constants
///        (CNFE), and add the demanded variable copies.
///
/// @details
/// Candidates are the chains of the masked-key family of `OriginalsHeadIndex`
/// whose key equals the goal's key masked at one slot, or at two slots when
/// at least one other slot remains; the key carries the sign, so a rule
/// concluding the goal's negation is never a candidate. Each constant is
/// replaced in every premise by the goal's argument at its slot, and every
/// premise whose arguments are then all fixed is prepared for integration. A
/// premise that still holds a bound argument the head does not fix is neither
/// prepared nor counted for a copy verdict: removing the `u_` prefixes would
/// read that variable as the block's same-spelled name
/// (D-534). A premise that reads `!(=[Y,Y])` after
/// the replacement is skipped the same way: it can never hold, so it is
/// neither prepared nor counted for a copy verdict
/// (D-535). A constant demands a variable copy
/// `(=[Y,Y_copy])` of its goal argument Y when the replacement makes Y occur
/// at least twice in a premise where it occurred once (the premise already
/// held Y as a fixed argument), unless Y is already in an equivalence class at
/// the goal's scope; with two constants each gets its own verdict and copy
/// (D-483). The class test reads Y by its block name and the copy is written
/// at the goal's scope (D-536); an anchor argument whose
/// anchor slot is not `(1)` is never a copy target (asserted,
/// I-327). Only chains with an id of at least
/// @p firstChainId are visited: the goal door passes `1`, the goal rescan
/// passes the first chain its previous rescans left unvisited, since a
/// repeated visit of a (goal, chain) pair changes nothing. The candidate list
/// is snapshotted at entry; chains minted by the preparations are left to the
/// next call.
/// A registered open goal can request the sole remaining bound argument of
/// a positive atomic single premise, when an installed owner is visible at
/// the goal scope. The argument is marked before removing u_ prefixes; every
/// other argument stays fixed. Existing preparation and admission own the
/// resulting demand and its witness replay (D-488).
///
/// @param inputExprStr The goal (span over a caller-stable buffer).
/// @param mb           The owning LB.
/// @param validityName The goal's scope (span over a caller-stable buffer).
/// @param firstChainId The smallest chain id to visit; `>= 1`.
/// @return Nothing.
/// @invariant Zero heap; visits candidates in ascending chain-id order.
/// @see OriginalsHeadIndex, maskedHeadKey, prepareIntegration.
void IntegrationProcessor::checkNecessityForEquality(StrSpan inputExprStr, Memory& mb, StrSpan validityName,
                                                   int32_t firstChainId) {
    assert(firstChainId >= 1 && "checkNecessityForEquality: chain ids start at 1");

    if (parameters.ban_disintegration) return;
    unaryIntegrationProcessor.transferUnaryIntegrationDemands(mb);

    // Atomic RT attribution: self-time = the whole-registry chain scan
    // (decode + head match); the mint-heavy premise handling below opens
    // its own nested rows.
    RT_SCOPE_HERE("CNFE");

    // Use special function to make all args "u_" (D7). ONE ScratchScope spans the
    // whole function; genericInput's name/args are read throughout — held named, no
    // rewind inside the chain loop (all fresh bytes byte-bump forward, reclaimed at
    // exit; prepareIntegration's own nested scopes rewind only their allocations).
    const unsigned neSlot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : scratchArenas().slotCount() - 1;
    ScratchArena& sArena = scratchArenas().forSlot(neSlot);
    ScratchScope sc(sArena);
    const ScratchString genericInput = gl::DefinitionExpansion::prefixArgumentsWithUScratch(sArena, StrSpan(inputExprStr));

    // Parse the input expression string (D8).
    const StrSpan inputName = extractExpressionSpan(StrSpan(genericInput));
    StrSpan inputArgs[ExecutionParameters::MAX_ARITY];
    const int32_t inputArgN = getArgsSpans(StrSpan(genericInput), inputArgs,
                                           ExecutionParameters::MAX_ARITY);

    // Head prefilter (exact): an input whose operator has no config or no
    // `(1)` slot cannot match any chain (headMatchPossible), so the walk is
    // skipped before any chain is touched, with the same (empty) effect.
    if (!headMatchPossible(inputName, inputArgN)) return;

    // Candidate chains (D9): the masked family of the derived head index over
    // the originals rule registry (`OriginalsHeadIndex`), probed per argument
    // slot and per pair of argument slots of the input
    // (collectHeadMatchChains), in the ascending id order of the registry, the
    // order the former full walk visited. The loop writes (prepareIntegration
    // per matched premise), so order is observable (I-84). The candidate list
    // is snapshotted before the loop: the loop mints `originals`, and only the
    // chains present at entry are visited. Each chain's rule ids are read
    // zero-copy (IdVecKeyView) into a stack buffer, then copyFrom'd onto
    // sArena (0% heap; the ids extracted BEFORE any originals mint).
    const unsigned ocSlot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : genScratchArenas().slotCount() - 1;
    ScratchArena& ocArena = genScratchArenas().forSlot(ocSlot);
    const ArenaOffset ocMark = ocArena.cursor();
    DirtyState chainIdxDirty = DirtyState::Clean;
    PagedVector<int32_t> chainIdx(&ocArena, &chainIdxDirty);
    int32_t chainN = 0;
    const bool negatedInput = genericInput.size() > 0 && StrSpan(genericInput).ptr[0] == '!';
    {
        RT_SCOPE_HERE("CNFE_HEAD_INDEX");
        ensureOriginalsHeadIndexBuilt(mb);
        chainN = collectHeadMatchChains(StrSpan(genericInput), mb, firstChainId,
            mb.overallHashMemory.originals.count(), ocArena, chainIdx);
    }
    RT_NOTE_ITERATIONS_HERE(chainN);
    for (int32_t cix = 0; cix < chainN; ++cix) {
        // Per-cix scope (mirrors the former chain.clear()): chainSpans + the
        // premise-loop temporaries are freed each iteration; genericInput /
        // inputArgs (below the mark) survive.
        ScratchScope cixScope(sArena);
        // Read the chain key's rule ids ZERO-COPY into a stack buffer BEFORE any
        // mint (prepareIntegration -> addToHashMemory mints `originals`, the SAME
        // set, so a held keyAt span would dangle -- I-3).
        int32_t chainIds[64];
        int32_t count;
        {
            const StrSpan kb = mb.overallHashMemory.originals.inner().keyAt(chainIdx[cix]);
            const IdVecKeyView kv = viewIdVecKey(kb);
            count = kv.count();
            assert(count <= 64 && "checkNecessityForEquality: chain exceeds 64");
            for (int32_t i = 0; i < count; ++i) chainIds[i] = kv.idAt(i);
        }
        assert(count > 0
            && "checkNecessityForEquality: the head index yielded an empty chain");

        // Head-first read. The checks below are pure reads (no mint), so the
        // head -- the last element of the implication chain (D10) -- is
        // inspected through a raw decodeView span and the chain is copied onto
        // sArena only after them.
        const StrSpan headView = mb.ruleInterner.decodeView(chainIds[count - 1]);
        assert(chainIdx[cix] >= firstChainId
            && "checkNecessityForEquality: a chain below the floor reached the walk");
        StrSpan headArgs[ExecutionParameters::MAX_ARITY];
        const int32_t headArgN = getArgsSpans(headView, headArgs,
                                              ExecutionParameters::MAX_ARITY);

        // The index guarantees the verdict: the head equals the input except
        // at its one constant (non-`u_`) argument, or at its two constants
        // beside at least one other argument, each constant at a `(1)`
        // definition slot (headMatchConstantSlots locates them and asserts
        // the rest).
        int32_t constantSlots[2] = { -1, -1 };
        const int32_t constantCount = headMatchConstantSlots(headView, inputName, negatedInput,
            inputArgs, inputArgN, constantSlots);

        // Matched: copy each chain element onto sArena BEFORE the premise loop
        // (which mints ruleInterner via prepareIntegration -> addToHashMemory,
        // so the raw decodeView spans above would dangle -- I-3). Held under
        // cixScope. The head's argument spans are re-derived from the stable
        // copy because the constants feed the replacement pairs used across
        // those mints.
        StrSpan chainSpans[64];
        for (int32_t i = 0; i < count; ++i) {
            const StrSpan dv = mb.ruleInterner.decodeView(chainIds[i]);
            chainSpans[i] = StrSpan(ScratchString::copyFrom(sArena, dv.ptr, dv.len));
        }
        const StrSpan headStr = chainSpans[count - 1];
        const int32_t headArgNCopy = getArgsSpans(headStr, headArgs,
                                                  ExecutionParameters::MAX_ARITY);
        assert(headArgNCopy == headArgN
               && "checkNecessityForEquality: head copy parses differently");
        (void)headArgNCopy;

        // Each constant is replaced by the input's argument at its slot; that
        // argument is the variable the constant's copy verdict tracks.
        StrReplacement rp[2] = {};
        StrSpan targetVars[2];
        for (int32_t t = 0; t < constantCount; ++t) {
            rp[t] = { headArgs[constantSlots[t]], inputArgs[constantSlots[t]] };
            targetVars[t] = inputArgs[constantSlots[t]];
        }

        bool conditionMet[2] = { false, false };

        // Iterate over premises (all elements except the last one)
        for (int32_t k = 0; k + 1 < count; ++k) {
            const StrSpan premise = chainSpans[k];   // stable copy on sArena (cixScope)

            // Apply replacement (each constant of the head becomes the input's
            // argument at its slot)
            const ScratchString instantiatedPremise =
                replaceKeysScratch(sArena, premise, rp, constantCount);

            // A premise that reads `!(=[Y,Y])` after the replacement can never
            // hold, with or without a copy of Y: it takes no part in the goal's
            // decisions, neither prepared nor counted for a copy verdict
            // (D-535).
            {
                const StrSpan instantiated(instantiatedPremise);
                if (startsWithSpan(instantiated, "!(=[", 4)) {
                    StrSpan pair[2];
                    const int32_t pairN = getArgsSpans(instantiated, pair, 2);
                    assert(pairN == 2 && "checkNecessityForEquality: an equality premise has two arguments");
                    (void)pairN;
                    if (equalSpans(pair[0], pair[1])) continue;
                }
            }

            // A one-premise projection can need a witness absent from its head.
            // Preserve its bound role before stripping fixed u_ arguments, whose
            // unprefixed spelling may equal the bound variable's numeric name.
            StrSpan preparationPremise(instantiatedPremise);
            bool projectionWitness = false;
            const NameId goalId = mb.nameMap.lookup(inputExprStr);
            const NameId goalScope = mb.nameMap.lookup(validityName);
            const LogicalEntity* premiseEntity = definitions.compiledEntity(
                extractExpressionUniversalSpan(premise));
            if (count == 2 && premise.ptr[0] == '(' && premiseEntity != nullptr
                && premiseEntity->category == "atomic" && goalId != 0 && goalScope != 0
                && !mb.phase3IgnoresScope(goalScope)
                && mb.intToBeProved.lookup(packStatementKey(goalId, goalScope)) != 0) {
                bool visibleOwner = false;
                const int32_t ownerCount = mb.overallHashMemory.originals.runLen(chainIdx[cix]);
                for (int32_t o = 0; o < ownerCount && !visibleOwner; ++o) {
                    const RuleOwner owner = mb.overallHashMemory.originals.recordAt(chainIdx[cix], o).owner;
                    const NameId ownerScope = static_cast<NameId>(static_cast<uint64_t>(owner) & 0xffffffffu);
                    visibleOwner = !mb.phase3IgnoresScope(ownerScope)
                        && (ownerScope == goalScope || mb.nameMap.isStrictAncestor(ownerScope, goalScope));
                }
                StrSpan premiseArgs[ExecutionParameters::MAX_ARITY];
                const int32_t premiseArgN = getArgsSpans(instantiatedPremise, premiseArgs,
                    ExecutionParameters::MAX_ARITY);
                StrSpan freeWitness;
                int32_t freeCount = 0;
                for (int32_t a = 0; a < premiseArgN; ++a) {
                    if (startsWithSpan(premiseArgs[a], "u_", 2)) continue;
                    if (freeCount == 0) {
                        freeWitness = premiseArgs[a];
                        freeCount = 1;
                    } else if (!equalSpans(freeWitness, premiseArgs[a])) {
                        freeCount = 2;
                        break;
                    }
                }
                if (visibleOwner && freeCount == 1) {
                    const StrReplacement markWitness[] = {{freeWitness, StrSpan("marker", 6)}};
                    preparationPremise = StrSpan(replaceKeysScratch(sArena, instantiatedPremise,
                        markWitness, 1));
                    projectionWitness = true;
                }
            }
            // Outside the projection, a premise that still holds a bound
            // argument after the replacement names an object the goal does not
            // determine. Removing the u_ prefixes would read that variable as
            // the block's same-spelled name, so the premise is neither prepared
            // nor counted for a copy verdict (D-534).
            if (!projectionWitness) {
                StrSpan instantiatedArgs[ExecutionParameters::MAX_ARITY];
                const int32_t instantiatedArgN = getArgsSpans(StrSpan(instantiatedPremise),
                    instantiatedArgs, ExecutionParameters::MAX_ARITY);
                bool unfixedBound = false;
                for (int32_t a = 0; a < instantiatedArgN && !unfixedBound; ++a)
                    unfixedBound = !startsWithSpan(instantiatedArgs[a], "u_", 2);
                if (unfixedBound) continue;
            }
            const ScratchString removed =
                gl::DefinitionExpansion::removeUPrefixScratch(sArena, preparationPremise);
            // argsSet: sorted-unique StrSpan run over `removed` (the prepareIntegration
            // span core's @pre); spans into `removed` (on sArena, held).
            StrSpan removedArgs[ExecutionParameters::MAX_ARITY];
            const int32_t removedArgN = getArgsSpans(StrSpan(removed), removedArgs,
                                                     ExecutionParameters::MAX_ARITY);
            StrSpan argsSet[ExecutionParameters::MAX_ARITY];
            int32_t argsSetN = 0;
            for (int32_t a = 0; a < removedArgN; ++a) {
                if (projectionWitness && equalSpans(removedArgs[a], StrSpan("marker", 6))) continue;
                int32_t lo = 0, hi = argsSetN; bool dup = false;
                while (lo < hi) {
                    const int32_t mid = (lo + hi) / 2;
                    const int c = compareSpans(argsSet[mid], removedArgs[a]);
                    if (c == 0) { dup = true; break; }
                    if (c < 0) lo = mid + 1; else hi = mid;
                }
                if (dup) continue;
                for (int32_t z = argsSetN; z > lo; --z) argsSet[z] = argsSet[z - 1];
                argsSet[lo] = removedArgs[a]; ++argsSetN;
            }

            if (projectionWitness) {
                // An atomic premise needs algebra admission, not an instruction
                // assembling that same atom from itself.
                const StrSpan bindingChain[] = {StrSpan(instantiatedPremise)};
                admissionMemory.updateAdmissionMap(mb, bindingChain, 1, argsSet, argsSetN,
                    parameters.maxIterationNumberVariable, parameters.standardMaxSecondaryNumber,
                    false, validityName);
                rejectedSourceReplay.revisitRejected2(StrSpan(removed), mb, validityName);
            } else {
                RT_SCOPE_HERE("CNFE_PREPARE_INTEGRATION");
                prepareIntegration(StrSpan(removed), argsSet, argsSetN, mb, StrSpan(validityName), inputExprStr);
            }

            // Copy verdict per constant: its target variable occurs at least
            // twice in the instantiated premise but exactly once in the
            // original one (the premise already held it as a fixed argument).
            // The loop does not stop at a met verdict — every determined
            // premise of a matched chain is prepared, and a projection is
            // admitted for its missing witness.
            StrSpan pArgs[ExecutionParameters::MAX_ARITY];
            const int32_t pArgN = getArgsSpans(StrSpan(instantiatedPremise), pArgs,
                                               ExecutionParameters::MAX_ARITY);
            StrSpan origArgs[ExecutionParameters::MAX_ARITY];
            const int32_t origArgN = getArgsSpans(premise, origArgs,
                                                  ExecutionParameters::MAX_ARITY);
            for (int32_t t = 0; t < constantCount; ++t) {
                int countAfter = 0;
                for (int32_t a = 0; a < pArgN; ++a) {
                    if (equalSpans(pArgs[a], targetVars[t])) countAfter++;
                }
                int countBefore = 0;
                for (int32_t a = 0; a < origArgN; ++a) {
                    if (equalSpans(origArgs[a], targetVars[t])) countBefore++;
                }
                if (countAfter >= 2 && countBefore == 1) conditionMet[t] = true;
            }
        }

        for (int32_t t = 0; t < constantCount; ++t) {
            if (!conditionMet[t]) continue;
            const StrSpan targetVar = targetVars[t];
            // The goal's arguments carry the u_ prefix here; the block knows
            // the copied object by its bare name.
            assert(targetVar.len > 2 && startsWithSpan(targetVar, "u_", 2)
                && "checkNecessityForEquality: a copy target without its u_ prefix");
            const StrSpan copiedName(targetVar.ptr + 2, targetVar.len - 2);
            // A copy target sits at a (1) slot of the goal, so it names a
            // natural number. An anchor argument outside a (1) slot (the set
            // N, the successor, the operations) reaches that slot only through
            // a captured variable (I-327).
            {
                const Memory* top = &mb;
                while (top->parentMemory != nullptr && top->parentMemory->parentMemory != nullptr)
                    top = top->parentMemory;
                const StrSpan topKey = top->exprKeyView();
                if (startsWithSpan(topKey, "(Anchor", 7)) {
                    const ce::CoreExpressionConfig* anchorCfg =
                        this->definitions.coreConfig(extractExpressionSpan(topKey));
                    assert(anchorCfg != nullptr
                        && "checkNecessityForEquality: an anchor without a config");
                    StrSpan anchorArgs[ExecutionParameters::MAX_ARITY];
                    const int32_t anchorArgN = getArgsSpans(topKey, anchorArgs,
                                                            ExecutionParameters::MAX_ARITY);
                    for (int32_t a = 0; a < anchorArgN; ++a) {
                        if (!equalSpans(anchorArgs[a], copiedName)) continue;
                        const std::pair<std::string, bool>* ds =
                            gl::CompiledDefinitionRegistry::findDefinitionSet(anchorCfg, a + 1);
                        assert(ds != nullptr && equalSpans(StrSpan(ds->first), StrSpan("(1)", 3))
                            && "checkNecessityForEquality: a variable copy of an anchor argument outside a (1) slot");
                        (void)ds;
                    }
                }
            }
            // Reuse an equality already visible at the goal's scope before
            // introducing another variable copy.
            bool hasExistingEquality = false;
            {
                const NameId targetVarId = mb.nameMap.lookup(copiedName);
                // Zero-copy view walk over the scope's class run — the former
                // decodeClassesAt heap snapshot. Pure read (lookup non-minting,
                // classHasMember / memberCount), the whole scan consumed before
                // any mint, so the peeked views stay valid. Bucket resolution
                // mirrors decodeClassesAt: name -> id -> bucket, {} at either miss.
                const NameId gateValId = mb.nameMap.lookup(validityName);
                if (gateValId != 0) canonicalizer.initializeCanonicalScope(mb,gateValId);
                const int32_t bucketId = (gateValId == 0)
                    ? 0 : mb.canonicalEquivalenceClassesMap.lookup(gateValId);
                if (bucketId != 0) {
                    const unsigned slot = (gl::WorkerContext::coreId() >= 0)
                        ? static_cast<unsigned>(gl::WorkerContext::coreId())
                        : genScratchArenas().slotCount() - 1;
                    ScratchArena& gArena = genScratchArenas().forSlot(slot);
                    const int32_t cc = mb.canonicalEquivalenceClassesMap.runLen(bucketId);
                    for (int32_t ci = 0; ci < cc; ++ci) {
                        ScratchScope viewScope(gArena);
                        int32_t blen = 0;
                        const char* bp = mb.canonicalEquivalenceClassesMap.peekRecordBytes(
                            bucketId, ci, blen, gArena);
                        const EquivalenceClassView view{ bp, blen };
                        if (targetVarId != 0 && classHasMember(view, targetVarId)) {
                            // User Request: assert that an eq class which contains the var has more than 1 element
                            assert(view.memberCount() > 1);
                            hasExistingEquality = true;
                            break;
                        }
                    }
                }
            }

            // If yes, do not add a new copy
            if (hasExistingEquality) {
                continue;
            }

            // Build "(=[Y,Y_copy])" for the target variable Y on sArena (D14,
            // the reactToHypo precedent): "(=[" + Y + "," + Y + "_copy])", and
            // deposit it through addExprToMemoryBlock.
            const StrSpan Y = targetVar;
            const int32_t elen = 11 + 2 * Y.len;  // 3 + Y + 1 + Y + 7 ("_copy])")
            char* eb = sArena.allocBytes(elen);
            int32_t eat = 0;
            eb[eat++] = '('; eb[eat++] = '='; eb[eat++] = '[';
            if (Y.len > 0) std::memcpy(eb + eat, Y.ptr, static_cast<size_t>(Y.len));
            eat += Y.len;
            eb[eat++] = ',';
            if (Y.len > 0) std::memcpy(eb + eat, Y.ptr, static_cast<size_t>(Y.len));
            eat += Y.len;
            const char eSfx[] = "_copy])";
            std::memcpy(eb + eat, eSfx, 7); eat += 7;
            assert(eat == elen);
            const ScratchString equalityExpr = ScratchString::wrap(sArena, eb, elen);

            // Create origin info (dead-end axiom — see "variable copy" tag)
            const TransientOrigin origin{ true, OriginTag::variableCopy, nullptr, 0 };

            const ScratchString finalExpr =
                gl::DefinitionExpansion::removeUPrefixScratch(sArena, StrSpan(equalityExpr));

            // Door levels: the LB's own level as a one-element stack run.
            const int lvRun[1] = { mb.level };

            // Add the new equality at the goal's scope, where the class test
            // above looked (D-536).
            statementMemory.addExprToMemoryBlock(StrSpan(finalExpr),
                mb,
                -1, // iteration
                0,  // status (Local statement)
                lvRun, 1,
                origin,
                -1, // coreId
                -1, // auxyIndex
                validityName,
                false);
        }
    }
    chainIdx.clear();
    ocArena.popTo(ocMark);
}
/// @brief Run CNFE for every open goal over the rule chains no earlier
        ///        rescan visited.
        ///
        /// @details
        /// Every goal has been visited against chains 1..goalScanChainMark,
        /// so each goal is visited from the first chain above the mark; the
        /// mark is raised to the chain count seen at entry when the walk ends.
        /// Without new chains no goal is visited, but the unary-demand
        /// transfer runs once when goals exist. Goals are walked in the sorted
        /// order of their decoded statement and scope (I-87). Called by the
        /// door after it installs implications and by the unary composition
        /// drain after it installs composed rules.
        ///
        /// @param memoryBlock The owning LB.
        /// @return Nothing.
        /// @invariant Zero heap; the chain map never shrinks during the walk
        ///            (I-60).
        /// @see checkNecessityForEquality, HashMemory::goalScanChainMark,
        ///      drainUnaryComposition.
void IntegrationProcessor::rescanGoalsAgainstNewChains(Memory& memoryBlock) {
    HashMemory& goalScanHm = memoryBlock.overallHashMemory;
    const int32_t firstNewChain = goalScanHm.goalScanChainMark + 1;
    const int32_t chainsAtStart = goalScanHm.originals.count();
    assert(firstNewChain <= chainsAtStart + 1
        && "rescanGoalsAgainstNewChains: goal-rescan floor above the chain count");
    if (firstNewChain > chainsAtStart) {
        if (memoryBlock.intToBeProved.count() > 0)
            unaryIntegrationProcessor.transferUnaryIntegrationDemands(memoryBlock);
    } else {
        const unsigned tbpSlot = (gl::WorkerContext::coreId() >= 0)
            ? static_cast<unsigned>(gl::WorkerContext::coreId())
            : genScratchArenas().slotCount() - 1;
        ScratchArena& tbpArena = genScratchArenas().forSlot(tbpSlot);
        ScratchScope tbpScope(tbpArena);
        int32_t tbpN = 0;
        int64_t* tbpKeys = sortToBeProvedKeys(memoryBlock.intToBeProved,
            memoryBlock.nameMap, tbpArena, tbpN);
        const unsigned goalStrSlot = (gl::WorkerContext::coreId() >= 0)
            ? static_cast<unsigned>(gl::WorkerContext::coreId())
            : scratchArenas().slotCount() - 1;
        ScratchArena& goalStrArena = scratchArenas().forSlot(goalStrSlot);
        for (int32_t i = 0; i < tbpN; ++i) {
            const int64_t k = tbpKeys[i];
            const NameId oid = Codec<StatementKey>::decode(k).orig;
            const NameId vid = Codec<StatementKey>::decode(k).validity;
            ScratchScope goalRowScope(goalStrArena);
            const StrSpan os = memoryBlock.nameMap.decodeView(oid);
            const ScratchString original =
                ScratchString::copyFrom(goalStrArena, os.ptr, os.len);
            const StrSpan vs = memoryBlock.nameMap.decodeView(vid);
            const ScratchString validity =
                ScratchString::copyFrom(goalStrArena, vs.ptr, vs.len);
            checkNecessityForEquality(StrSpan(original), memoryBlock, StrSpan(validity),
                                            firstNewChain);
        }
    }
    // Every goal of the snapshot has now visited chains up to chainsAtStart;
    // goals registered meanwhile visited all chains at their door. A nested
    // rescan may already have raised the floor further.
    assert(goalScanHm.originals.count() >= chainsAtStart
        && "rescanGoalsAgainstNewChains: the chain map shrank during the goal rescan");
    if (goalScanHm.goalScanChainMark < chainsAtStart)
        goalScanHm.goalScanChainMark = chainsAtStart;
}

}
