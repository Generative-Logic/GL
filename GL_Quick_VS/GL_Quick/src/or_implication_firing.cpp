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

#include "prover.hpp"
#include "memory_infra/paged_sorted_ordinals.hpp"

namespace gl {

/// @brief Activate a residual OR only after its deposit is admitted.
/// @details An existing ancestor supplies the continuation at its own scope.
/// Canonical duplicate refusals reuse the deposit chooser; other refused
/// deposits remain absent. Case-disintegration flags are independent.
/// @param mb Owning block.
/// @param original Stable original residual text.
/// @param scope Offered scope.
/// @param registeredOriginal Canonical id returned by the deposit door, or zero.
/// @param publicationMail Current phase output, or same-iteration mail for ordinary callers.
/// @return True if an admitted form supplies the source.
bool ExpressionAnalyzer::activateOrImplicationContinuation(Memory& mb,
    StrSpan original, NameId scope, NameId registeredOriginal, ColdMail* publicationMail)
{
    if (mb.phase3IgnoresScope(scope)) return false;
    const unsigned slot = g_currentCoreId >= 0
        ? static_cast<unsigned>(g_currentCoreId) : scratchArenas().slotCount() - 1;
    ScratchArena& strings = scratchArenas().forSlot(slot);
    ScratchScope stringScope(strings);
    NameId id = registeredOriginal != 0 ? registeredOriginal : mb.nameMap.lookup(original);
    if (id == 0 || !ancestorKnown(mb, id, scope, true)) {
        if (registeredOriginal != 0) return false;
        const StrSpan view = mb.nameMap.decodeView(scope);
        const ScratchString validity = ScratchString::copyFrom(strings, view.ptr, view.len);
        int levels[256];
        int32_t count = 0;
        const CanonicalForm canonical = canonicalFormAtScope(original, StrSpan(validity),
            mb, strings, levels, count, 256);
        id = mb.nameMap.lookup(canonical.text);
        if (id == 0 || !ancestorKnown(mb, id, scope, true)) return false;
    }
    NameId knownScope = scope;
    while (mb.intKnownStatements.find(StatementKey{id, knownScope}) == nullptr) {
        knownScope = mb.nameMap.parentOf(knownScope);
        assert(knownScope != 0);
    }
    const StrSpan textView = mb.nameMap.decodeView(id);
    const ScratchString text = ScratchString::copyFrom(strings, textView.ptr, textView.len);
    const StrSpan scopeView = mb.nameMap.decodeView(knownScope);
    const ScratchString validity = ScratchString::copyFrom(strings, scopeView.ptr, scopeView.len);
    const IntEncodedExpr statement = encodeExpression(StrSpan(text), StrSpan(validity), mb.nameMap);
    registerOrImplicationSource(mb, statement, publicationMail);
    return true;
}

/// @brief Select deterministic refuters and compatible scopes for one reduction.
/// @details Canonical views may be prepared, but no known facts, proof history,
/// mail or source retirement are produced. A negative match leaves no firing.
/// @param mb Owning block with prepared canonical matching state.
/// @param sourceKey Original OR and source scope.
/// @param lookupScope Candidate effective scope.
/// @param selection Stack-owned result, consumed only when this returns true.
/// @return True when a scope-compatible reduction was selected.
bool ExpressionAnalyzer::selectOrImplication(Memory& mb, int64_t sourceKey,
    NameId lookupScope, OrImplicationSelection& selection)
{
    const OrImplicationSource* found = mb.orImplications.find(sourceKey);
    if (found == nullptr || mb.phase3IgnoresScope(lookupScope)) return false;
    const OrImplicationSource source = *found;
    const NameId sourceOriginal = static_cast<NameId>(uint64_t(sourceKey) >> 32);
    const NameId sourceScope = static_cast<NameId>(uint64_t(sourceKey));
    assert(sourceScope == lookupScope || mb.nameMap.isStrictAncestor(sourceScope, lookupScope));
    if (mb.phase3IgnoresScope(sourceScope)) return false;
    auto& index = mb.orImplicationMatches;
    bindOrSourceViews(mb, index, sourceKey, lookupScope);
    const int64_t sourceCacheKey = packInt32Pair(sourceOriginal, lookupScope);
    const OrCanonicalCache* cachedSource = index.caches().find(sourceCacheKey);
    assert(cachedSource != nullptr && cachedSource->dirty == 0);
    const NameId canonicalSource = cachedSource->canonicalId;
    const bool localRule = (source.flags & OrImplications::Local) != 0
        || cachedSource->localEquality != 0;
    constexpr int cap = ExecutionParameters::kMaxReducedOrLeaves;
    const int32_t leafCount = source.leafCount;
    assert(leafCount >= 2 && leafCount <= cap);
    selection = {};
    auto& choices = selection.choices;
    auto& used = selection.used;
    int32_t refuted = 0;
    const auto originalOf = [](int64_t key) { return static_cast<NameId>(uint64_t(key) >> 32); };
    const auto scopeOf = [](int64_t key) { return static_cast<NameId>(uint64_t(key)); };
    const auto better = [&](const OrImplicationRefuter& a, const OrImplicationRefuter& b) {
        if (b.owner == 0) return true;
        if (a.local != b.local) return a.local;
        const int32_t ad = mb.nameMap.ancLen(a.scope), bd = mb.nameMap.ancLen(b.scope);
        if (ad != bd) return ad < bd;
        const int cmp = compareSpans(mb.nameMap.decodeView(originalOf(a.owner)),
            mb.nameMap.decodeView(originalOf(b.owner)));
        if (cmp != 0) return cmp < 0;
        const int scopes = compareSpans(mb.nameMap.decodeView(scopeOf(a.owner)),
            mb.nameMap.decodeView(scopeOf(b.owner)));
        if (scopes != 0) return scopes < 0;
        return compareSpans(mb.nameMap.decodeView(a.scope), mb.nameMap.decodeView(b.scope)) < 0;
    };
    for (int32_t leaf = 0; leaf < leafCount; ++leaf) {
        const int32_t* id = index.viewIds.find(OrViewIdentity::sourceLeaf(sourceKey, lookupScope, leaf));
        assert(id != nullptr && *id >= 0);
        const OrIndexedView watch = index.views[*id];
        index.postings().forEach(orPostingKey(OrPostingKind::FactScopes, watch.templateId),
            [&](int64_t rawScope) {
                const NameId scope = static_cast<NameId>(rawScope);
                if (scope != lookupScope && !mb.nameMap.isStrictAncestor(scope, lookupScope)) return;
                index.postings().forEach(orPostingKey(OrPostingKind::FactKey,
                    packInt32Pair(watch.templateId, scope)), [&](int64_t rawId) {
                        const OrIndexedView fact = index.views[static_cast<int32_t>(rawId)];
                        assert(fact.live && fact.identity.isFact());
                        const StatementFlags* flags = mb.intKnownStatements.find(
                            StatementKey{originalOf(fact.identity.owner), scopeOf(fact.identity.owner)});
                        assert(flags != nullptr);
                        if (mb.phase3IgnoresScope(scopeOf(fact.identity.owner))) return;
                        const OrCanonicalCache* cache = index.caches().find(fact.cacheKey);
                        assert(cache != nullptr && cache->dirty == 0);
                        assert(fact.canonicalId == watch.canonicalId);
                        const OrImplicationRefuter candidate{fact.identity.owner, fact.identity.scope,
                            fact.canonicalId, flags->local || cache->localEquality != 0};
                        if (better(candidate, choices[leaf])) choices[leaf] = candidate;
                    });
            });
        used[leaf] = choices[leaf].owner != 0;
        if (used[leaf]) ++refuted;
    }
    if (refuted == 0) return false;
    int32_t headLeaf = -1;
    if (refuted == leafCount) {
        // Prefer the last leaf whose other refuters satisfy firing ownership.
        for (int32_t h = leafCount; h-- > 0;) {
            bool localPremise = false;
            for (int32_t i = 0; i < leafCount; ++i)
                if (i != h && choices[i].local) localPremise = true;
            if (ceFilteringActive || localRule || localPremise) { headLeaf = h; break; }
        }
        if (headLeaf < 0) return false;
        used[headLeaf] = false;
        --refuted;
    }
    const int32_t survivors = leafCount - refuted;
    if (survivors == 1 && headLeaf < 0)
        for (int32_t i = 0; i < leafCount; ++i) if (!used[i]) headLeaf = i;
    bool localPremise = false;
    for (int32_t i = 0; i < leafCount; ++i)
        if (used[i] && choices[i].local) localPremise = true;
    if (!ceFilteringActive && !localRule && !localPremise) return false;

    NameId sourceProofScope = canonicalSource == sourceOriginal ? sourceScope : lookupScope;
    auto& factProofScopes = selection.factProofScopes;
    NameId premiseScope = NameMap::MAIN_ID;
    for (int32_t i = 0; i < leafCount; ++i) {
        if (choices[i].owner == 0) continue;
        factProofScopes[i] = choices[i].canonical == originalOf(choices[i].owner)
            ? scopeOf(choices[i].owner) : choices[i].scope;
        if (!used[i]) continue;
        if (!mb.nameMap.comparable(premiseScope, factProofScopes[i])) return false;
        premiseScope = mb.nameMap.deeperOf(premiseScope, factProofScopes[i]);
    }
    if (!mb.nameMap.comparable(sourceProofScope, premiseScope)) return false;
    const NameId resultScope = mb.nameMap.deeperOf(sourceProofScope, premiseScope);
    if (mb.phase3IgnoresScope(resultScope)) return false;
    selection.sourceKey = sourceKey;
    selection.lookupScope = lookupScope;
    selection.source = source;
    selection.canonicalSource = canonicalSource;
    selection.sourceProofScope = sourceProofScope;
    selection.premiseScope = premiseScope;
    selection.resultScope = resultScope;
    selection.refuted = refuted;
    selection.survivors = survivors;
    selection.headLeaf = headLeaf;
    selection.localPremise = localPremise;
    return true;
}

/// @brief Materialize a selection through the existing checked firing path.
/// @details Applies ordinary hypothetical-scope gates, creates only selected
/// equality/proof records and mail, and retains the original retirement order.
/// @param mb The unchanged block from the successful selection call.
/// @param selection Successful selection for immediate consumption.
/// @param output Live internal-mail output for this processing phase.
/// @return True when the head is known or accepted for delivery.
bool ExpressionAnalyzer::materializeOrImplication(Memory& mb,
    const OrImplicationSelection& selection, ColdMail& output)
{
    const int64_t sourceKey = selection.sourceKey;
    const NameId lookupScope = selection.lookupScope;
    const OrImplicationSource& source = selection.source;
    assert(mb.orImplications.find(sourceKey) != nullptr);
    const NameId sourceOriginal = static_cast<NameId>(uint64_t(sourceKey) >> 32);
    const NameId sourceScope = static_cast<NameId>(uint64_t(sourceKey));
    const NameId canonicalSource = selection.canonicalSource;
    const NameId sourceProofScope = selection.sourceProofScope;
    const NameId premiseScope = selection.premiseScope;
    const NameId resultScope = selection.resultScope;
    const auto& choices = selection.choices;
    const auto& used = selection.used;
    const auto& factProofScopes = selection.factProofScopes;
    constexpr int cap = ExecutionParameters::kMaxReducedOrLeaves;
    const int32_t leafCount = source.leafCount;
    const int32_t refuted = selection.refuted;
    const int32_t survivors = selection.survivors;
    const int32_t headLeaf = selection.headLeaf;
    const bool localPremise = selection.localPremise;
    assert(leafCount >= 2 && leafCount <= cap && refuted > 0 && survivors > 0);
    const auto originalOf = [](int64_t key) { return static_cast<NameId>(uint64_t(key) >> 32); };
    const auto scopeOf = [](int64_t key) { return static_cast<NameId>(uint64_t(key)); };
    const unsigned pageSlot = g_currentCoreId >= 0
        ? static_cast<unsigned>(g_currentCoreId) : genScratchArenas().slotCount() - 1;
    const unsigned stringSlot = g_currentCoreId >= 0
        ? static_cast<unsigned>(g_currentCoreId) : scratchArenas().slotCount() - 1;
    ScratchScope workScope(genScratchArenas().forSlot(pageSlot));
    ScratchArena& strings = scratchArenas().forSlot(stringSlot);
    ScratchScope stringScope(strings);
    const auto owned = [&](NameId id) {
        const StrSpan view = mb.nameMap.decodeView(id);
        return ScratchString::copyFrom(strings, view.ptr, view.len);
    };
    const ScratchString sourceText = owned(canonicalSource);
    const ScratchString sourceValidity = owned(sourceProofScope);
    const ScratchString resultValidity = owned(resultScope);
    const ScratchString premiseValidity = owned(premiseScope);
    StrSpan premiseTexts[cap];
    int32_t premiseCount = 0;
    int32_t iteration = 0;
    for (int32_t i = 0; i < leafCount; ++i) {
        if (!used[i]) continue;
        const ScratchString factText = owned(choices[i].canonical);
        premiseTexts[premiseCount++] = StrSpan(factText);
        const ScratchString factValidity = owned(factProofScopes[i]);
        const IntEncodedExpr encoded = encodeExpression(StrSpan(factText),
            StrSpan(factValidity), mb.nameMap);
        if (containsSpan(StrSpan(premiseValidity), StrSpan("_hypo_", 6))
            && factProofScopes[i] != premiseScope && !encoded.isAnchor) return false;
        iteration = std::max(iteration, encoded.maxIteration + 1);
    }

    const LogicalEntity* orEntity = compiledEntity(extractExpressionUniversalSpan(StrSpan(sourceText)));
    assert(orEntity != nullptr && orEntity->category == "or");
    assert(static_cast<int32_t>(orEntity->implications.size()) == expectedOrImplicationCount(leafCount));
    int32_t ruleIndex = headLeaf;
    if (survivors > 1) {
        int32_t choose[cap + 1][cap + 1]{};
        for (int32_t n = 0; n <= cap; ++n) {
            choose[n][0] = choose[n][n] = 1;
            for (int32_t k = 1; k < n; ++k)
                choose[n][k] = choose[n - 1][k - 1] + choose[n - 1][k];
        }
        ruleIndex = leafCount;
        for (int32_t k = 1; k < refuted; ++k) ruleIndex += choose[leafCount][k];
        int32_t previous = -1, position = 0;
        for (int32_t i = 0; i < leafCount; ++i) if (used[i]) {
            for (int32_t candidate = previous + 1; candidate < i; ++candidate)
                ruleIndex += choose[leafCount - candidate - 1][refuted - position - 1];
            previous = i;
            ++position;
        }
    }
    assert(ruleIndex >= 0 && ruleIndex < static_cast<int32_t>(orEntity->implications.size()));
    const ScratchString sourceU = prefixArgumentsWithUScratch(strings, StrSpan(sourceText));
    StrSpan signatureArgs[ExecutionParameters::MAX_ARITY], actualArgs[ExecutionParameters::MAX_ARITY];
    const int32_t arity = getArgsSpans(StrSpan(orEntity->signature), signatureArgs, ExecutionParameters::MAX_ARITY);
    assert(getArgsSpans(StrSpan(sourceU), actualArgs, ExecutionParameters::MAX_ARITY) == arity);
    StrReplacement replacements[ExecutionParameters::MAX_ARITY];
    for (int32_t i = 0; i < arity; ++i) replacements[i] = {signatureArgs[i], actualArgs[i]};
    const ScratchString compactU = replaceKeysScratch(strings,
        StrSpan(orEntity->implications[ruleIndex]), replacements, arity);
    const ScratchString compact = removeUPrefixScratch(strings, StrSpan(compactU));
    const LogicalEntity* implication = compiledEntity(extractExpressionUniversalSpan(StrSpan(compactU)));
    assert(implication != nullptr && implication->category == "implication");
    const int32_t implicationArity = getArgsSpans(StrSpan(implication->signature),
        signatureArgs, ExecutionParameters::MAX_ARITY);
    assert(getArgsSpans(StrSpan(compactU), actualArgs, ExecutionParameters::MAX_ARITY) == implicationArity);
    for (int32_t i = 0; i < implicationArity; ++i) replacements[i] = {signatureArgs[i], actualArgs[i]};
    StrSpan elements[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
    const int32_t elementCount = static_cast<int32_t>(implication->elements.size());
    assert(elementCount >= 2 && elementCount <= ExecutionParameters::MAX_INSTRUCTION_ELEMENTS);
    for (int32_t i = 0; i < elementCount; ++i)
        elements[i] = StrSpan(replaceKeysScratch(strings, StrSpan(implication->elements[i]),
            replacements, implicationArity));
    const ScratchString ruleU = reconstructImplicationFullBindScratch(strings,
        elements, elementCount - 1, elements[elementCount - 1]);
    const ScratchString rule = removeUPrefixScratch(strings, StrSpan(ruleU));
    const ScratchString head = removeUPrefixScratch(strings, elements[elementCount - 1]);

    int levels[256];
    int32_t levelCount = 0;
    for (int32_t i = 0; i < source.levelCount; ++i) {
        const int level = mb.orImplications.levels[source.levelStart + i];
        if (level >= 0) levelCount = insertLevelSorted(levels, levelCount, level, 256);
    }
    WatchBridge sourceBridge;
    if (canonicalSource != sourceOriginal) {
        canonicalWatchFormAt(mb, sourceOriginal, lookupScope, strings,
            sourceBridge, levels, levelCount, 256);
        assert(mb.nameMap.lookup(sourceBridge.text) == canonicalSource);
    }
    for (int32_t i = 0; i < leafCount; ++i) {
        if (choices[i].owner == 0) continue;
        int factLevels[256];
        const int32_t levelId = mb.intStatementLevelsMap.lookup(choices[i].owner);
        assert(levelId != 0);
        int32_t factLevelCount = coldIntRunAt(mb.intStatementLevelsMap, levelId, factLevels, 256);
        if (choices[i].canonical != originalOf(choices[i].owner)) {
            WatchBridge bridge;
            canonicalWatchFormAt(mb, originalOf(choices[i].owner), choices[i].scope,
                strings, bridge, factLevels, factLevelCount, 256, true);
            assert(mb.nameMap.lookup(bridge.text) == choices[i].canonical);
            writeWatchTwinCanonical(mb, originalOf(choices[i].owner),
                scopeOf(choices[i].owner), choices[i].scope, true);
        }
        // The omitted all-refuted leaf is materialized for contradiction,
        // but contributes no premise level to this selected implication.
        if (used[i]) for (int32_t j = 0; j < factLevelCount; ++j)
            if (factLevels[j] >= 0)
                levelCount = insertLevelSorted(levels, levelCount, factLevels[j], 256);
    }
    int32_t nonDerived = 0;
    while (nonDerived < levelCount && levels[nonDerived] < 0) ++nonDerived;
    if (nonDerived != 0) {
        for (int32_t i = nonDerived; i < levelCount; ++i)
            levels[i - nonDerived] = levels[i];
        levelCount -= nonDerived;
    }
    if (mb.phase3IgnoresScope(resultScope)) return false;
    const NameId knownHead = mb.nameMap.lookup(StrSpan(head));
    if (!parameters.compressor_mode && knownHead != 0
        && ancestorKnown(mb, knownHead, resultScope, true)) {
        if (survivors > 1)
            activateOrImplicationContinuation(mb, StrSpan(head), resultScope, knownHead, &output);
        if (resultScope == sourceScope) retireOrImplicationSource(mb, sourceKey);
        return true;
    }
    bool doNotDisintegrate = isOrIntroInstall(premiseTexts, premiseCount, StrSpan(head));
    if (parameters.incubator_mode && !parameters.ban_disintegration && !doNotDisintegrate)
        doNotDisintegrate = isAnchorLb(mb) || !localPremise;

    if (parameters.trackHistory && !ceFilteringActive) {
        const int historyCap = parameters.compressor_mode
            ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr;
        const auto makeVisible = [&](NameId original, NameId wanted) {
            NameId knownAt = wanted;
            while (mb.intKnownStatements.find(StatementKey{original, knownAt}) == nullptr) {
                assert(knownAt != NameMap::MAIN_ID);
                knownAt = mb.nameMap.parentOf(knownAt);
                assert(knownAt != 0);
            }
            const ScratchString text = owned(original);
            const ScratchString at = owned(knownAt);
            copyOriginRowsToMailOut(mb, StrSpan(text), StrSpan(at), historyCap);
            if (knownAt != wanted) {
                const ScratchString target = owned(wanted);
                const OriginDep dependency{StrSpan(text), StrSpan(at)};
                if (!originRowExists(mb, StrSpan(text), StrSpan(target)))
                    addOriginEncoded(mb.exprOriginMap, mb.originInterner, StrSpan(text),
                        StrSpan(target), OriginTag::zoomIn, &dependency, 1, historyCap);
                copyOriginRowsToMailOut(mb, StrSpan(text), StrSpan(target), historyCap);
            }
        };
        const ScratchString originalSourceText = owned(sourceOriginal);
        const ScratchString originalSourceValidity = owned(sourceScope);
        copyOriginRowsToMailOut(mb, StrSpan(originalSourceText),
            StrSpan(originalSourceValidity), historyCap);
        if (canonicalSource != sourceOriginal) {
            if (!originRowExists(mb, StrSpan(sourceText), StrSpan(sourceValidity))) {
                OriginDep dependencies[1 + WatchBridge::CAP];
                dependencies[0] = {StrSpan(originalSourceText), StrSpan(originalSourceValidity)};
                for (int32_t i = 0; i < sourceBridge.eqN; ++i)
                    dependencies[i + 1] = {sourceBridge.eqText[i], sourceBridge.eqScope[i]};
                addOriginEncoded(mb.exprOriginMap, mb.originInterner, StrSpan(sourceText),
                    StrSpan(sourceValidity), OriginTag::equality1, dependencies,
                    1 + sourceBridge.eqN, historyCap);
            }
            copyOriginRowsToMailOut(mb, StrSpan(sourceText), StrSpan(sourceValidity), historyCap);
        }
        StrSpan sourceLeaves[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
        const int32_t n = flattenRegistryOrLeavesScratch(*orEntity, StrSpan(sourceU),
            sourceLeaves, ExecutionParameters::MAX_INSTRUCTION_ELEMENTS, strings);
        assert(n == leafCount);
        const ScratchString expanded = expandSignature(StrSpan("or", 2), StrSpan(sourceU),
            sourceLeaves, n, strings);
        const auto history = [&](StrSpan text, OriginTag tag, StrSpan dependency) {
            const OriginDep dep{dependency, StrSpan(sourceValidity)};
            addOriginEncoded(mb.exprOriginMap, mb.originInterner, text,
                StrSpan(sourceValidity), tag, &dep, 1, historyCap);
            mb.addMailOutOrigin(text, StrSpan(sourceValidity), tag, &dep, 1, historyCap);
        };
        history(StrSpan(expanded), OriginTag::expansion, StrSpan(sourceText));
        history(StrSpan(compact), OriginTag::disintegration, StrSpan(expanded));
        history(StrSpan(rule), OriginTag::expansion, StrSpan(compact));
        OriginDep dependencies[cap + 1];
        int32_t dependencyCount = 1;
        dependencies[0] = {StrSpan(rule), StrSpan(sourceValidity)};
        for (int32_t i = 0; i < leafCount; ++i) if (used[i]) {
            makeVisible(choices[i].canonical, factProofScopes[i]);
            const ScratchString text = owned(choices[i].canonical);
            const ScratchString at = owned(factProofScopes[i]);
            dependencies[dependencyCount++] = {StrSpan(text), StrSpan(at)};
        }
        std::sort(dependencies + 1, dependencies + dependencyCount,
            [](const OriginDep& a, const OriginDep& b) {
                const int expression = compareSpans(a.original, b.original);
                return expression != 0 ? expression < 0
                    : compareSpans(a.validity, b.validity) < 0;
            });
        addInternalMailOrigin(output, mb.originInterner, StrSpan(head),
            StrSpan(resultValidity), OriginTag::implication, dependencies, dependencyCount, historyCap);
    }
    insertInternalStatement(output, mb.nameMap, StrSpan(head), StrSpan(resultValidity),
        levels, levelCount);
    setInternalDisintegrationSignal(output, mb.nameMap, StrSpan(head), StrSpan(resultValidity),
        doNotDisintegrate, survivors == 1, iteration, false, survivors > 1);
    mb.mutatedThisBurst = true;
    if (resultScope == sourceScope) retireOrImplicationSource(mb, sourceKey);
    return true;
}

/// @brief Select and materialize one scoped OR reduction without repeat matching.
/// @details The stack record contains ids and bounded values only. Selection
/// and materialization run consecutively against the same block state.
/// @param mb Owning block with prepared matching views.
/// @param sourceKey Original OR and source scope.
/// @param lookupScope Candidate effective scope.
/// @param output Live internal-mail output for this processing phase.
/// @return True when the selected head is known or accepted for delivery.
bool ExpressionAnalyzer::evaluateOrImplication(Memory& mb, int64_t sourceKey,
    NameId lookupScope, ColdMail& output)
{
    RT_SCOPE_HERE("OR_IMPLICATION_EVALUATE");
    OrImplicationSelection selection;
    if (!selectOrImplication(mb, sourceKey, lookupScope, selection)) return false;
    return materializeOrImplication(mb, selection, output);
}

/// @brief Reconcile and evaluate staged OR work until no fresh event remains.
/// @details Stable logical identities survive source retirement and index
/// recycling. Parent scopes are considered before their descendants; new
/// canonical twins are reconciled in the next pass through the same drain.
/// Each pass deduplicates its current events, but a later fact or class change
/// can require evaluating the same source and scope again.
/// @param mb Owning block and staged OR state.
/// @param output Caller-selected live internal-mail output.
/// @return Void.
void ExpressionAnalyzer::drainOrImplications(Memory& mb, ColdMail& output)
{
    RT_SCOPE_HERE("OR_IMPLICATION_DRAIN");
    const unsigned slot = g_currentCoreId >= 0
        ? static_cast<unsigned>(g_currentCoreId) : genScratchArenas().slotCount() - 1;
    ScratchArena& arena = genScratchArenas().forSlot(slot);
    ScratchScope workScope(arena);
    DirtyState dirty = DirtyState::Clean;
    PagedVector<OrEvaluation> work(&arena, &dirty);
    for (;;) {
        prepareOrImplicationMatches(mb);
        auto& index = mb.orImplicationMatches;
        if (index.evaluations.empty()) break;
        work.clear();
        for (int32_t i = 1; i <= index.evaluations.count(); ++i)
            work.push_back(index.evaluations.keyAt(i));
        index.evaluations.resetToFresh();
        const auto less = [&](int32_t a, int32_t b) {
            const OrEvaluation x = work[a], y = work[b];
            const NameId xo = static_cast<NameId>(uint64_t(x.owner) >> 32);
            const NameId yo = static_cast<NameId>(uint64_t(y.owner) >> 32);
            if (xo != yo)
                return compareSpans(mb.nameMap.decodeView(xo), mb.nameMap.decodeView(yo)) < 0;
            if (x.owner != y.owner)
                return compareSpans(mb.nameMap.decodeView(static_cast<NameId>(uint64_t(x.owner))),
                    mb.nameMap.decodeView(static_cast<NameId>(uint64_t(y.owner)))) < 0;
            const int xd = mb.nameMap.ancLen(x.scope), yd = mb.nameMap.ancLen(y.scope);
            return xd != yd ? xd < yd
                : compareSpans(mb.nameMap.decodeView(x.scope), mb.nameMap.decodeView(y.scope)) < 0;
        };
        PagedSortedOrdinals<decltype(less)> order(arena, work.size(), less, 1024);
        for (int32_t i = 0; i < work.size(); ++i) {
            const OrEvaluation item = work[order.next()];
            evaluateOrImplication(mb, item.owner, item.scope, output);
        }
    }
}

}
