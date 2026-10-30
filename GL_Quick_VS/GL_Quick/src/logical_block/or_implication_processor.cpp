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
#include "../compiler/compact_expression_syntax.hpp"
#include "or_implication_processor.hpp"
namespace gl {
/// @brief Bind inputs for or implication processor.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param blockIdentity Borrowed LogicBlockIdentity interface or state; must outlive this component.
/// @param canonicalizer Borrowed Canonicalizer interface or state; must outlive this component.
/// @param ceFilteringActive Borrowed bool interface or state; must outlive this component.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @param hashRuleClassifier Borrowed HashRuleClassifier interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param caseLedger Borrowed CaseLedger interface or state; must outlive this component.
/// @param caseProcessor Borrowed CaseProcessor interface or state; must outlive this component.
/// @param disintegrationProcessor Borrowed DisintegrationProcessor interface or state; must outlive this component.
/// @param proofHistory Borrowed ProofHistory interface or state; must outlive this component.
/// @param watchProcessor Borrowed WatchProcessor interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
OrImplicationProcessor::OrImplicationProcessor(LogicBlockIdentity& blockIdentity, Canonicalizer& canonicalizer, const bool& ceFilteringActive, const CompiledDefinitionRegistry& definitions, HashRuleClassifier& hashRuleClassifier, const ProverParameters& parameters, CaseLedger& caseLedger, CaseProcessor& caseProcessor, DisintegrationProcessor& disintegrationProcessor, ProofHistory& proofHistory, WatchProcessor& watchProcessor) : blockIdentity(blockIdentity), canonicalizer(canonicalizer), ceFilteringActive(ceFilteringActive), definitions(definitions), hashRuleClassifier(hashRuleClassifier), parameters(parameters), caseLedger(caseLedger), caseProcessor(caseProcessor), disintegrationProcessor(disintegrationProcessor), proofHistory(proofHistory), watchProcessor(watchProcessor) {}

/// @brief Retire symmetric duplicate exclusion sources before matching.
///
/// @details Reuses source canonical ids and paged scratch sorting. Groups never
/// cross source scopes. The source of the largest open cohort wins, with the
/// existing signature tie-break; absent open cohorts, source insertion order wins.
/// @param body Owning resident block with current equality classes.
/// @return Void.
/// @invariant Original leaves, levels, locality and history are never transferred.
/// @see canonicalOrIdAtScope; retireOrImplicationSource.
void OrImplicationProcessor::retireDuplicateOrSources(Memory& body) {
    auto& sources = body.orImplications;
    if (sources.sources.count() < 2) return;
    const unsigned slot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
    ScratchArena& arena = genScratchArenas().forSlot(slot);
    DirtyState dirty = DirtyState::Clean;
    struct Row { int64_t source; NameId scope; NameId canonical; int32_t score; };
    PagedVector<Row> rows(&arena, &dirty);
    for (int32_t i = 1; i <= sources.sources.count(); ++i) {
        const int64_t key = sources.sources.keyAt(i);
        const auto* stored = sources.find(key);
        if (stored == nullptr) continue;
        const NameId original = static_cast<NameId>(uint64_t(key) >> 32);
        const NameId scope = static_cast<NameId>(uint64_t(key));
        const StrSpan text = body.nameMap.decodeView(original);
        const auto* entity = definitions.compiledEntity(extractExpressionUniversalSpan(text));
        if (!gl::CompactExpressionSyntax::isOrCompactSpan(text) || entity == nullptr || entity->symmetricDigitSlots[0] == 0) continue;
        const NameId canonical = caseLedger.canonicalOrIdAtScope(body, original, body.nameMap.decodeView(scope));
        const StrSpan signature = body.nameMap.decodeView(original);
        const StrSpan parent = body.nameMap.decodeView(scope);
        const int32_t cohort = lookupOrCohortId(body.lbStateInterner,
            body.lbStateInterner.lookup(parent), body.lbStateInterner.lookup(signature));
        const int32_t score = cohort != 0 && body.orDisjunctCount.lookup(cohort) != 0
            ? caseProcessor.orCohortStatementCount(body, parent, signature,
                                                OrScopeKind::Disintegration) : -1;
        OrImplicationSource updated = *sources.find(key);
        if (updated.canonicalId != canonical) {
            updated.canonicalId = canonical;
            sources.sources.upsert(key, updated);
        }
        rows.push_back(Row{key, scope, canonical, score});
    }
    const auto less = [&](int32_t a, int32_t b) {
        const Row x = rows[a], y = rows[b];
        if (x.scope != y.scope)
            return compareSpans(body.nameMap.decodeView(x.scope), body.nameMap.decodeView(y.scope)) < 0;
        if (x.canonical != y.canonical)
            return compareSpans(body.nameMap.decodeView(x.canonical), body.nameMap.decodeView(y.canonical)) < 0;
        if (x.score != y.score) return x.score > y.score;
        if (x.score >= 0) return compareSpans(
            body.nameMap.decodeView(static_cast<NameId>(uint64_t(x.source) >> 32)),
            body.nameMap.decodeView(static_cast<NameId>(uint64_t(y.source) >> 32))) < 0;
        return a < b;
    };
    PagedSortedOrdinals<decltype(less)> order(arena, rows.size(), less, 1024);
    NameId previousScope = 0, previousCanonical = 0;
    for (int32_t i = 0; i < rows.size(); ++i) {
        const Row row = rows[order.next()];
        if (row.scope == previousScope && row.canonical == previousCanonical)
            retireOrImplicationSource(body, row.source);
        else {
            previousScope = row.scope;
            previousCanonical = row.canonical;
        }
    }
}

/// @brief Preserve a finalized OR and its ordered complementary leaves.
/// @details Only an admitted positive category-or row can be a source. Levels
/// and locality come from the paired known registries after canonical admission.
/// Existing sources keep their payload; a newly local owner queues a recheck.
/// @param mb Owning block and finalized known-statement registries.
/// @param statement Admitted positive compiled OR with actual argument ids.
/// @param publicationMail Current phase output, or same-iteration mail for ordinary callers.
/// @return Void.
void OrImplicationProcessor::registerOrImplicationSource(Memory& mb,
    const IntEncodedExpr& statement, ColdMail* publicationMail) {
    assert(!statement.negation && statement.originalId != 0);
    const int64_t key = packStatementKey(statement.originalId, statement.validityId);
    const StatementFlags* known = mb.intKnownStatements.find(
        StatementKey{statement.originalId, statement.validityId});
    assert(known != nullptr);
    const bool local = known->local;
    const int32_t levelsId = mb.intStatementLevelsMap.lookup(key);
    assert(levelsId != 0 && mb.intStatementLevelsMap.runLen(levelsId) > 0);
    const unsigned slot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
    ScratchArena& strings = scratchArenas().forSlot(slot);
    ScratchScope stringScope(strings);
    const StrSpan view = mb.nameMap.decodeView(statement.originalId);
    const ScratchString original = ScratchString::copyFrom(strings, view.ptr, view.len);
    const LogicalEntity* entity = definitions.compiledEntity(extractExpressionUniversalSpan(StrSpan(original)));
    assert(entity != nullptr && entity->category == "or"
        && "OR source admission requires a positive category-or logical entity");
    const OrImplicationSource* previous = mb.orImplications.sources.find(key);
    if (previous != nullptr) {
        if ((previous->flags & OrImplications::Active) != 0 && local
            && (previous->flags & OrImplications::Local) == 0) {
            OrImplicationSource updated = *previous;
            updated.flags |= OrImplications::Local;
            mb.orImplications.sources.upsert(key, updated);
            mb.orImplications.pendingSources.push_back(key);
        }
        return;
    }
    StrSpan flat[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
    const int32_t count = hashRuleClassifier.flattenRegistryOrLeavesScratch(*entity, StrSpan(original),
        flat, ExecutionParameters::MAX_INSTRUCTION_ELEMENTS, strings);
    OrImplicationLeaf leaves[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
    for (int32_t i = 0; i < count; ++i) {
        leaves[i].literalId = mb.nameMap.encode(flat[i]);
        leaves[i].complementId = mb.nameMap.encode(gl::DefinitionExpansion::negateScratch(strings, flat[i]));
    }
    int levels[256];
    const int32_t levelCount = mb.intStatementLevelsMap.runLen(levelsId);
    assert(levelCount <= 256);
    for (int32_t i = 0; i < levelCount; ++i)
        levels[i] = mb.intStatementLevelsMap.valueAt(levelsId, i);
    const bool inserted = mb.orImplications.insert(key, statement.originalId,
        leaves, count, levels, levelCount, local);
    assert(inserted);
    disintegrationProcessor.activateDirectOrRefutations(mb, statement,
        publicationMail ? *publicationMail : mb.sameIterationInternalMail);
}

/// @brief Build newly needed signed-core views and reconcile staged changes.
/// @details Source events are collected before one scan for all new cores.
/// Existing core arrivals avoid that scan. Canonical changes settle before
/// exact complementary-template buckets queue comparable source scopes.
/// @param mb Owning block and its canonical and derived OR state.
/// @return Void.
void OrImplicationProcessor::prepareOrImplicationMatches(Memory& mb) {
    RT_SCOPE_HERE("OR_IMPLICATION_MATCHES");
    auto& sources = mb.orImplications;
    auto& index = mb.orImplicationMatches;
    if (!index.built || !sources.pendingSources.empty() || !sources.pendingClasses.empty()) {
        retireDuplicateOrSources(mb);
    }
    const unsigned slot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
    ScratchArena& arena = genScratchArenas().forSlot(slot);
    ScratchArena& strings = scratchArenas().forSlot(slot);
    ScratchScope workScope(arena);
    ScratchScope stringScope(strings);
    DirtyState dirty = DirtyState::Clean;
    {
        RT_SCOPE_HERE("OR_MATCH_COMPACT_RETIRED");
        sources.compactRetired(arena);
    }
    if (sources.sources.empty()) {
        index.clear();
        sources.pendingSources.clear();
        sources.pendingFacts.clear();
        sources.pendingClasses.clear();
        return;
    }
    TypedColdSet<int64_t> sourceKeys(&arena, &dirty);
    TypedColdSet<int64_t> factKeys(&arena, &dirty);
    TypedColdSet<int64_t> newCores(&arena, &dirty);
    const bool rebuilding = !index.built;
    if (rebuilding) {
        index.clear();
        buildOrClassDirectory(mb, index);
        for (int32_t id = 1; id <= sources.sources.count(); ++id)
            sourceKeys.mint(sources.sources.keyAt(id));
    } else {
        for (int32_t i = 0; i < sources.pendingClasses.size(); ++i) {
            const int64_t key = sources.pendingClasses[i];
            stageOrClassMember(mb, index, static_cast<NameId>(uint64_t(key) >> 32),
                static_cast<NameId>(uint64_t(key)), false);
        }
    }
    sources.pendingClasses.clear();
    for (int32_t i = 0; i < sources.pendingSources.size(); ++i)
        sourceKeys.mint(sources.pendingSources[i]);
    sources.pendingSources.clear();
    for (int32_t i = 0; i < sources.pendingFacts.size(); ++i)
        factKeys.mint(sources.pendingFacts[i]);
    sources.pendingFacts.clear();

    const auto coreFor = [&](NameId original, bool mint) {
        const StrSpan text = mb.nameMap.decodeView(original);
        const bool negated = text.len > 0 && text.ptr[0] == '!';
        const StrSpan core = extractExpressionUniversalSpan(text);
        NameId coreId = mb.nameMap.lookup(core);
        if (coreId == 0 && mint) {
            const ScratchString owned = ScratchString::copyFrom(strings, core.ptr, core.len);
            coreId = mb.nameMap.encode(StrSpan(owned));
        }
        // A never-minted predicate cannot be one of the watched cores.
        return coreId == 0 ? int64_t{0} : packInt32Pair(coreId, negated);
    };
    const auto sortedKeys = [&](const auto& keys, auto&& consume) {
        const auto less = [&](int32_t a, int32_t b) {
            const int64_t x = keys.keyAt(a + 1), y = keys.keyAt(b + 1);
            const NameId xo = static_cast<NameId>(uint64_t(x) >> 32);
            const NameId yo = static_cast<NameId>(uint64_t(y) >> 32);
            if (xo != yo)
                return compareSpans(mb.nameMap.decodeView(xo), mb.nameMap.decodeView(yo)) < 0;
            return compareSpans(mb.nameMap.decodeView(static_cast<NameId>(uint64_t(x))),
                mb.nameMap.decodeView(static_cast<NameId>(uint64_t(y)))) < 0;
        };
        PagedSortedOrdinals<decltype(less)> order(arena, keys.count(), less, 1024);
        for (int32_t i = 0; i < keys.count(); ++i)
            consume(keys.keyAt(order.next() + 1));
    };
    sortedKeys(sourceKeys, [&](int64_t key) {
        RT_SCOPE_HERE("OR_MATCH_SOURCE_BIND");
        const OrImplicationSource* found = sources.find(key);
        if (found == nullptr) return;
        const OrImplicationSource source = *found;
        const NameId scope = static_cast<NameId>(uint64_t(key));
        for (int32_t leaf = 0; leaf < source.leafCount; ++leaf) {
            const NameId complement = sources.leaves[source.leafStart + leaf].complementId;
            const int64_t core = coreFor(complement, true);
            assert(core != 0);
            index.watchSourceCore(key, core);
            if (!index.initializedCores.contains(core)) newCores.mint(core);
        }
        bindOrSourceViews(mb, index, key, scope);
        for (int32_t leaf = 0; leaf < source.leafCount; ++leaf)
            index.changedViews.mint(OrViewIdentity::sourceLeaf(key, scope, leaf));
    });
    if (!newCores.empty()) {
        RT_SCOPE_HERE("OR_NEW_CORES_KNOWN_SCAN");
        for (int32_t id = 1; id <= mb.intKnownStatements.count(); ++id) {
            const StatementKey known = mb.intKnownStatements.decodeKey(id);
            const int64_t core = coreFor(known.orig, false);
            if (core != 0 && newCores.contains(core))
                factKeys.mint(packStatementKey(known.orig, known.validity));
        }
        for (int32_t id = 1; id <= newCores.count(); ++id)
            index.initializedCores.mint(newCores.keyAt(id));
    }
    sortedKeys(factKeys, [&](int64_t key) {
        RT_SCOPE_HERE("OR_MATCH_FACT_BIND");
        const NameId original = static_cast<NameId>(uint64_t(key) >> 32);
        const NameId scope = static_cast<NameId>(uint64_t(key));
        const int64_t core = coreFor(original, false);
        if (core == 0) return;
        const bool watched = index.postings().containsKey(
            orPostingKey(OrPostingKind::SourceCore, core));
        const bool known = mb.intKnownStatements.find(StatementKey{original, scope}) != nullptr;
        if (watched && known && !mb.phase3IgnoresScope(scope)) {
            assert(mb.intStatementLevelsMap.lookup(key) != 0);
            index.recordFactCore(key, core);
            const int32_t view = bindOrCanonicalView(mb, index,
                OrViewIdentity::knownFact(key, scope), original);
            // A repeated arrival can change the carrier's locality: one join
            // of its entry's fact role covers every carrier.
            index.changedFacts.mint(index.views[view].cacheKey);
        } else {
            index.eraseFact(key, core, arena);
        }
    });
    index.built = true;
    {
        RT_SCOPE_HERE("OR_MATCH_REFRESH_VIEWS");
        refreshOrCanonicalViews(mb, index);
    }
    for (int32_t i = 1; i <= index.changedViews.count(); ++i) {
        RT_SCOPE_HERE("OR_MATCH_JOIN_VIEW");
        const OrViewIdentity identity = index.changedViews.keyAt(i);
        assert(!identity.isFact() && "fact carriers join through their entry");
        const int32_t* slotId = index.viewIds.find(identity);
        if (slotId == nullptr || *slotId < 0) continue;
        const OrIndexedView view = index.views[*slotId];
        const OrCanonicalCache* cache = index.caches().find(view.cacheKey);
        assert(cache != nullptr && cache->dirty == 0 && cache->templateId != 0);
        const int32_t templateId = view.templateId;
        index.postings().forEach(orPostingKey(OrPostingKind::FactScopes, templateId),
            [&](int64_t rawScope) {
                const NameId scope = static_cast<NameId>(rawScope);
                if (!mb.nameMap.comparable(identity.scope, scope)) return;
                index.postings().forEach(orPostingKey(OrPostingKind::FactKey,
                    packInt32Pair(templateId, scope)), [&](int64_t entryKey) {
                        assert(index.caches().find(entryKey) != nullptr
                            && index.caches().find(entryKey)->factCarriers > 0);
                        if (sources.find(identity.owner) == nullptr) return;
                        index.evaluations.mint(OrEvaluation{identity.owner,
                            mb.nameMap.deeperOf(identity.scope, scope)});
                    });
            });
    }
    for (int32_t i = 1; i <= index.changedFacts.count(); ++i) {
        RT_SCOPE_HERE("OR_MATCH_JOIN_VIEW");
        const int64_t entryKey = index.changedFacts.keyAt(i);
        const OrCanonicalCache* cache = index.caches().find(entryKey);
        // An entry merged away, retired or left without carriers since it was queued.
        if (cache == nullptr || cache->templateId == 0 || cache->factCarriers == 0) continue;
        assert(cache->dirty == 0);
        const int32_t templateId = cache->templateId;
        const NameId factScope = static_cast<NameId>(uint64_t(entryKey));
        index.postings().forEach(orPostingKey(OrPostingKind::WatchScopes, templateId),
            [&](int64_t rawScope) {
                const NameId scope = static_cast<NameId>(rawScope);
                if (!mb.nameMap.comparable(factScope, scope)) return;
                index.postings().forEach(orPostingKey(OrPostingKind::WatchKey,
                    packInt32Pair(templateId, scope)), [&](int64_t rawId) {
                        const OrIndexedView other = index.views[static_cast<int32_t>(rawId)];
                        assert(other.live);
                        if (sources.find(other.identity.owner) == nullptr) return;
                        index.evaluations.mint(OrEvaluation{other.identity.owner,
                            mb.nameMap.deeperOf(factScope, scope)});
                    });
            });
    }
    index.changedViews.resetToFresh();
    index.changedFacts.resetToFresh();
    {
        RT_SCOPE_HERE("OR_MATCH_MAINTAIN");
        index.maintain();
    }
}

/// @brief Unlink one active source and facts no longer used by any source.
/// @details Watch views and ordinary fact views sharing an owner are distinct.
/// Every posting loses reachability immediately; payload compaction is batched.
/// @param mb Owning block and canonical/derived OR state.
/// @param sourceKey Active original OR and source scope.
/// @return Void.
void OrImplicationProcessor::retireOrImplicationSource(Memory& mb, int64_t sourceKey) {
    const bool retired = mb.orImplications.retire(sourceKey);
    assert(retired);
    auto& index = mb.orImplicationMatches;
    const unsigned slot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
    ScratchArena& arena = genScratchArenas().forSlot(slot);
    index.retireSource(sourceKey, arena);
}



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
bool OrImplicationProcessor::activateOrImplicationContinuation(Memory& mb,
    StrSpan original, NameId scope, NameId registeredOriginal, ColdMail* publicationMail) {
    if (mb.phase3IgnoresScope(scope)) return false;
    const unsigned slot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
    ScratchArena& strings = scratchArenas().forSlot(slot);
    ScratchScope stringScope(strings);
    NameId id = registeredOriginal != 0 ? registeredOriginal : mb.nameMap.lookup(original);
    if (id == 0 || !gl::StatementKnowledge::ancestorKnown(mb, id, scope, true)) {
        if (registeredOriginal != 0) return false;
        const StrSpan view = mb.nameMap.decodeView(scope);
        const ScratchString validity = ScratchString::copyFrom(strings, view.ptr, view.len);
        int levels[256];
        int32_t count = 0;
        const CanonicalForm canonical = canonicalizer.canonicalFormAtScope(original, StrSpan(validity),
            mb, strings, levels, count, 256);
        id = mb.nameMap.lookup(canonical.text);
        if (id == 0 || !gl::StatementKnowledge::ancestorKnown(mb, id, scope, true)) return false;
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
bool OrImplicationProcessor::selectOrImplication(Memory& mb, int64_t sourceKey,
    NameId lookupScope, OrImplicationSelection& selection) {
    const OrImplicationSource* found = mb.orImplications.find(sourceKey);
    if (found == nullptr || mb.phase3IgnoresScope(lookupScope)) return false;
    const OrImplicationSource source = *found;
    const NameId sourceOriginal = static_cast<NameId>(uint64_t(sourceKey) >> 32);
    const NameId sourceScope = static_cast<NameId>(uint64_t(sourceKey));
    assert(sourceScope == lookupScope || mb.nameMap.isStrictAncestor(sourceScope, lookupScope));
    if (mb.phase3IgnoresScope(sourceScope)) return false;
    auto& index = mb.orImplicationMatches;
    bindOrSourceViews(mb, index, sourceKey, lookupScope);
    const int32_t* sourceView = index.viewIds.find(OrViewIdentity::sourceLeaf(sourceKey, lookupScope, 0));
    assert(sourceView != nullptr && *sourceView >= 0);
    const OrCanonicalCache* cachedSource = index.caches().find(index.views[*sourceView].cacheKey);
    assert(cachedSource != nullptr && cachedSource->dirty == 0 && cachedSource->templateId != 0);
    const NameId canonicalSource = cachedSource->canonicalId;
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
    const unsigned pageSlot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
    ScratchArena& arena = genScratchArenas().forSlot(pageSlot);
    for (int32_t leaf = 0; leaf < leafCount; ++leaf) {
        const int32_t* id = index.viewIds.find(OrViewIdentity::sourceLeaf(sourceKey, lookupScope, leaf));
        assert(id != nullptr && *id >= 0);
        const OrIndexedView watch = index.views[*id];
        index.postings().forEach(orPostingKey(OrPostingKind::FactScopes, watch.templateId),
            [&](int64_t rawScope) {
                const NameId scope = static_cast<NameId>(rawScope);
                if (scope != lookupScope && !mb.nameMap.isStrictAncestor(scope, lookupScope)) return;
                // A deeper entry cannot beat a local choice.
                if (choices[leaf].owner != 0 && choices[leaf].local
                    && mb.nameMap.ancLen(choices[leaf].scope) < mb.nameMap.ancLen(scope)) return;
                index.postings().forEach(orPostingKey(OrPostingKind::FactKey,
                    packInt32Pair(watch.templateId, scope)), [&](int64_t entryKey) {
                        const OrCanonicalCache* entry = index.caches().find(entryKey);
                        assert(entry != nullptr && entry->dirty == 0 && entry->factCarriers > 0);
                        assert(entry->canonicalId == watch.canonicalId);
                        // The carriers in today's tie order: spelling, then owner scope.
                        ScratchScope carrierScope(arena);
                        DirtyState carrierDirty = DirtyState::Clean;
                        PagedVector<int64_t> carriers(&arena, &carrierDirty);
                        index.postings().forEach(orPostingKey(OrPostingKind::CacheUsers, entryKey),
                            [&](int64_t rawId) {
                                const OrIndexedView user = index.views[static_cast<int32_t>(rawId)];
                                assert(user.live);
                                if (user.identity.isFact()) carriers.push_back(user.identity.owner);
                            });
                        assert(carriers.size() == entry->factCarriers);
                        const auto carrierLess = [&](int32_t a, int32_t b) {
                            const int spelling = compareSpans(mb.nameMap.decodeView(originalOf(carriers[a])),
                                mb.nameMap.decodeView(originalOf(carriers[b])));
                            if (spelling != 0) return spelling < 0;
                            return compareSpans(mb.nameMap.decodeView(scopeOf(carriers[a])),
                                mb.nameMap.decodeView(scopeOf(carriers[b]))) < 0;
                        };
                        ChunkSortedOrdinals<decltype(carrierLess)> order(arena, carriers.size(), 0, carrierLess);
                        for (int32_t c = 0; c < carriers.size(); ++c) {
                            const int64_t owner = carriers[order.next()];
                            const StatementFlags* flags = mb.intKnownStatements.find(
                                StatementKey{originalOf(owner), scopeOf(owner)});
                            assert(flags != nullptr);
                            if (mb.phase3IgnoresScope(scopeOf(owner))) continue;
                            const bool local = flags->local || (originalOf(owner) != entry->canonicalId
                                && rewriteLocalAt(mb, originalOf(owner), scope, entry->canonicalId));
                            const OrImplicationRefuter candidate{owner, scope, entry->canonicalId, local};
                            if (better(candidate, choices[leaf])) choices[leaf] = candidate;
                            // Later carriers of this entry cannot beat its first local one.
                            if (local) break;
                        }
                    });
            });
        used[leaf] = choices[leaf].owner != 0;
        if (used[leaf]) ++refuted;
    }
    if (refuted == 0) return false;
    const bool localRule = (source.flags & OrImplications::Local) != 0
        || (canonicalSource != sourceOriginal
            && rewriteLocalAt(mb, sourceOriginal, lookupScope, canonicalSource));
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
bool OrImplicationProcessor::materializeOrImplication(Memory& mb,
    const OrImplicationSelection& selection, ColdMail& output) {
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
    const unsigned pageSlot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
    const unsigned stringSlot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
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

    const LogicalEntity* orEntity = definitions.compiledEntity(extractExpressionUniversalSpan(StrSpan(sourceText)));
    assert(orEntity != nullptr && orEntity->category == "or");
    assert(static_cast<int32_t>(orEntity->implications.size()) == gl::OrRuleCompiler::expectedOrImplicationCount(leafCount));
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
    const ScratchString sourceU = gl::DefinitionExpansion::prefixArgumentsWithUScratch(strings, StrSpan(sourceText));
    StrSpan signatureArgs[ExecutionParameters::MAX_ARITY], actualArgs[ExecutionParameters::MAX_ARITY];
    const int32_t arity = getArgsSpans(StrSpan(orEntity->signature), signatureArgs, ExecutionParameters::MAX_ARITY);
    assert(getArgsSpans(StrSpan(sourceU), actualArgs, ExecutionParameters::MAX_ARITY) == arity);
    StrReplacement replacements[ExecutionParameters::MAX_ARITY];
    for (int32_t i = 0; i < arity; ++i) replacements[i] = {signatureArgs[i], actualArgs[i]};
    const ScratchString compactU = replaceKeysScratch(strings,
        StrSpan(orEntity->implications[ruleIndex]), replacements, arity);
    const ScratchString compact = gl::DefinitionExpansion::removeUPrefixScratch(strings, StrSpan(compactU));
    const LogicalEntity* implication = definitions.compiledEntity(extractExpressionUniversalSpan(StrSpan(compactU)));
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
    const ScratchString ruleU = gl::ImplicationBuilder::reconstructImplicationFullBindScratch(strings,
        elements, elementCount - 1, elements[elementCount - 1]);
    const ScratchString rule = gl::DefinitionExpansion::removeUPrefixScratch(strings, StrSpan(ruleU));
    const ScratchString head = gl::DefinitionExpansion::removeUPrefixScratch(strings, elements[elementCount - 1]);

    int levels[256];
    int32_t levelCount = 0;
    for (int32_t i = 0; i < source.levelCount; ++i) {
        const int level = mb.orImplications.levels[source.levelStart + i];
        if (level >= 0) levelCount = insertLevelSorted(levels, levelCount, level, 256);
    }
    WatchBridge sourceBridge;
    if (canonicalSource != sourceOriginal) {
        watchProcessor.canonicalWatchFormAt(mb, sourceOriginal, lookupScope, strings,
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
            watchProcessor.canonicalWatchFormAt(mb, originalOf(choices[i].owner), choices[i].scope,
                strings, bridge, factLevels, factLevelCount, 256, true);
            assert(mb.nameMap.lookup(bridge.text) == choices[i].canonical);
            watchProcessor.writeWatchTwinCanonical(mb, originalOf(choices[i].owner),
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
        && gl::StatementKnowledge::ancestorKnown(mb, knownHead, resultScope, true)) {
        if (survivors > 1)
            activateOrImplicationContinuation(mb, StrSpan(head), resultScope, knownHead, &output);
        if (resultScope == sourceScope) retireOrImplicationSource(mb, sourceKey);
        return true;
    }
    bool doNotDisintegrate = hashRuleClassifier.isOrIntroInstall(premiseTexts, premiseCount, StrSpan(head));
    if (parameters.incubator_mode && !parameters.ban_disintegration && !doNotDisintegrate)
        doNotDisintegrate = blockIdentity.isAnchorLb(mb) || !localPremise;

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
            proofHistory.copyOriginRowsToMailOut(mb, StrSpan(text), StrSpan(at), historyCap);
            if (knownAt != wanted) {
                const ScratchString target = owned(wanted);
                const OriginDep dependency{StrSpan(text), StrSpan(at)};
                if (!proofHistory.originRowExists(mb, StrSpan(text), StrSpan(target)))
                    addOriginEncoded(mb.exprOriginMap, mb.originInterner, StrSpan(text),
                        StrSpan(target), OriginTag::zoomIn, &dependency, 1, historyCap);
                proofHistory.copyOriginRowsToMailOut(mb, StrSpan(text), StrSpan(target), historyCap);
            }
        };
        const ScratchString originalSourceText = owned(sourceOriginal);
        const ScratchString originalSourceValidity = owned(sourceScope);
        proofHistory.copyOriginRowsToMailOut(mb, StrSpan(originalSourceText),
            StrSpan(originalSourceValidity), historyCap);
        if (canonicalSource != sourceOriginal) {
            if (!proofHistory.originRowExists(mb, StrSpan(sourceText), StrSpan(sourceValidity))) {
                OriginDep dependencies[1 + WatchBridge::CAP];
                dependencies[0] = {StrSpan(originalSourceText), StrSpan(originalSourceValidity)};
                for (int32_t i = 0; i < sourceBridge.eqN; ++i)
                    dependencies[i + 1] = {sourceBridge.eqText[i], sourceBridge.eqScope[i]};
                addOriginEncoded(mb.exprOriginMap, mb.originInterner, StrSpan(sourceText),
                    StrSpan(sourceValidity), OriginTag::equality1, dependencies,
                    1 + sourceBridge.eqN, historyCap);
            }
            proofHistory.copyOriginRowsToMailOut(mb, StrSpan(sourceText), StrSpan(sourceValidity), historyCap);
        }
        StrSpan sourceLeaves[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
        const int32_t n = hashRuleClassifier.flattenRegistryOrLeavesScratch(*orEntity, StrSpan(sourceU),
            sourceLeaves, ExecutionParameters::MAX_INSTRUCTION_ELEMENTS, strings);
        assert(n == leafCount);
        const ScratchString expanded = gl::DefinitionExpansion::expandSignature(StrSpan("or", 2), StrSpan(sourceU),
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
bool OrImplicationProcessor::evaluateOrImplication(Memory& mb, int64_t sourceKey,
    NameId lookupScope, ColdMail& output) {
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
void OrImplicationProcessor::drainOrImplications(Memory& mb, ColdMail& output) {
    RT_SCOPE_HERE("OR_IMPLICATION_DRAIN");
    const unsigned slot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
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

/// @brief Seed class-scope token postings for canonical OR lookup.
/// @details The initial pass reads each serialized class once; further calls
/// are cache hits. Class additions are handled by stageOrClassMember.
/// @param mb Owning block and current equivalence classes.
/// @param index Derived matching state on the block arena.
/// @return Void.
void OrImplicationProcessor::buildOrClassDirectory(Memory& mb,
    OrImplicationMatchIndex& index) {
    if (index.classScopesBuilt) return;
    const unsigned slot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
    ScratchArena& arena = genScratchArenas().forSlot(slot);
    for (int32_t id = 1; id <= mb.equivalenceClassesMap.count(); ++id) {
        const NameId scope = mb.equivalenceClassesMap.keyAt(id);
        for (int32_t j = 0; j < mb.equivalenceClassesMap.runLen(id); ++j) {
            ScratchScope scopeBytes(arena);
            int32_t length = 0;
            const char* bytes = mb.equivalenceClassesMap.peekRecordBytes(
                id, j, length, arena);
            const EquivalenceClassView cls{bytes, length};
            for (int32_t k = 0; k < cls.memberCount(); ++k)
                index.recordClassScope(cls.memberId(k), scope);
        }
    }
    index.classScopesBuilt = true;
}

/// @brief Publish a new canonical entry with its form's ordered token stream.
/// @details Tokens of the canonical form stream directly into the index
/// transaction, avoiding a temporary collection and preserving NameMap mint
/// order. The index owns all entry and reverse-token updates; this function
/// owns parsing. A spelling's own rewritten tokens are posted per view
/// (postOrViewTokens).
/// @param mb Owning block and expression interner.
/// @param index Derived matching index receiving the new entry.
/// @param key Canonical expression and effective scope.
/// @param value Settled canonical result from the existing chooser.
/// @return Void.
void OrImplicationProcessor::publishOrCanonicalCache(Memory& mb,
    OrImplicationMatchIndex& index, int64_t key, OrCanonicalCache value) {
    const unsigned stringSlot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
    ScratchArena& strings = scratchArenas().forSlot(stringSlot);
    ScratchScope stringScope(strings);
    const StrSpan raw = mb.nameMap.decodeView(value.canonicalId);
    const ScratchString owned = ScratchString::copyFrom(strings, raw.ptr, raw.len);
    index.publishCache(key, value, [&](auto&& appendToken) {
        forEachSubstitutionToken(StrSpan(owned), [&](StrSpan token) {
            appendToken(mb.nameMap.encode(token));
        });
    });
}

/// @brief Queue every user of an entry at relevant strict-descendant equality scopes.
/// @details Each user is projected by projectOrUser; queued identities are
/// stable even when view slots are later recycled.
/// @param mb Owning block and scope tree.
/// @param index Derived matching state.
/// @param cacheKey Live canonical expression and lookup scope.
/// @return Void.
void OrImplicationProcessor::projectOrCacheViews(Memory& mb,
    OrImplicationMatchIndex& index, int64_t cacheKey) {
    index.postings().forEach(orPostingKey(OrPostingKind::CacheUsers, cacheKey),
        [&](int64_t rawId) { projectOrUser(mb, index, static_cast<int32_t>(rawId)); });
}

/// @brief Queue one view at the strict-descendant scopes whose classes reach it.
/// @details A class member reaches the view through a token of its entry's
/// canonical form or through a token its own spelling adds to that form.
/// Token and class-scope postings avoid a source-by-scope product; only
/// identities without a live view are queued.
/// @param mb Owning block and scope tree.
/// @param index Derived matching state.
/// @param viewId Live view slot.
/// @return Void.
void OrImplicationProcessor::projectOrUser(Memory& mb,
    OrImplicationMatchIndex& index, int32_t viewId) {
    RT_SCOPE_HERE("OR_CACHE_PROJECT");
    const OrIndexedView view = index.views[viewId];
    assert(view.live);
    const NameId viewScope = view.identity.scope;
    const auto project = [&](int64_t token) {
        index.postings().forEach(orPostingKey(OrPostingKind::ClassScopes, token),
            [&](int64_t rawScope) {
                const NameId scope = static_cast<NameId>(rawScope);
                if (!mb.nameMap.isStrictAncestor(viewScope, scope)
                    || mb.phase3IgnoresScope(scope)) return;
                const OrViewIdentity projected = view.identity.atScope(scope);
                const int32_t* existing = index.viewIds.find(projected);
                if (existing == nullptr || *existing < 0)
                    index.projections.mint(projected);
            });
    };
    index.postings().forEach(orPostingKey(OrPostingKind::CacheTokens, view.cacheKey), project);
    index.postings().forEach(orPostingKey(OrPostingKind::ViewTokens, viewId), project);
}

/// @brief Bind an OR lookup view to the canonical entry of its spelling.
/// @details The entry is keyed by canonical expression and effective scope,
/// so spellings with one canonical form share it. An existing identity keeps
/// its view and entry, including a dirty entry awaiting the drain. A new view
/// posts its rewritten tokens and is projected to reachable descendants.
/// @param mb Owning block, equivalence classes and interners.
/// @param index Derived matching state.
/// @param identity Original owner, leaf role and effective scope.
/// @param originalId Fully instantiated complementary leaf or known fact.
/// @return The live view slot for the supplied identity.
int32_t OrImplicationProcessor::bindOrCanonicalView(Memory& mb,
    OrImplicationMatchIndex& index, OrViewIdentity identity, NameId originalId) {
    assert(originalId != 0 && identity.scope != 0);
    const NameId ownerScope = static_cast<NameId>(uint64_t(identity.owner));
    assert(ownerScope == identity.scope
        || mb.nameMap.isStrictAncestor(ownerScope, identity.scope));
    const int32_t* existing = index.viewIds.find(identity);
    if (existing != nullptr && *existing >= 0) {
        assert(index.views[*existing].originalId == originalId);
        return *existing;
    }
    const int64_t cacheKey = ensureOrCanonicalCache(mb, index, originalId, identity.scope);
    const int32_t id = index.bindView(identity, cacheKey, originalId);
    postOrViewTokens(mb, index, id, originalId, static_cast<NameId>(uint64_t(cacheKey) >> 32));
    projectOrUser(mb, index, id);
    return id;
}

/// @brief Reuse or create the canonical entry of an expression at a scope.
/// @details One text-only canonicalization (canonicalOrFormId) finds the
/// entry; every spelling with that canonical form at the scope shares it and
/// its token postings, scalar facts and derived OR leaf views alike.
/// @param mb Owning block, classes and interners.
/// @param index Derived matching state.
/// @param originalId Original whole expression in NameMap space.
/// @param scopeId Effective lookup scope.
/// @return Packed canonical-expression/effective-scope entry key.
int64_t OrImplicationProcessor::ensureOrCanonicalCache(Memory& mb,
    OrImplicationMatchIndex& index, NameId originalId, NameId scopeId) {
    assert(originalId != 0 && scopeId != 0);
    buildOrClassDirectory(mb, index);
    const NameId canonicalId = canonicalOrFormId(mb, originalId, scopeId);
    const int64_t cacheKey = packInt32Pair(canonicalId, scopeId);
    if (!index.cacheLive(cacheKey)) {
        RT_SCOPE_HERE("OR_CACHE_FRESH");
        const int32_t templateId = mb.templateInterner.encode(
            mb.nameMap.decodeView(canonicalId));
        publishOrCanonicalCache(mb, index, cacheKey,
            OrCanonicalCache{canonicalId, templateId, 0, 0});
    }
    return cacheKey;
}

/// @brief Canonical NameMap id of an expression at a scope for OR lookup.
/// @details A text-only call of the shared chooser with the OR equality and
/// inequality opt-in: no justifiers, locality or levels are built. The text
/// equals the form canonicalWatchFormAt produces with evidence.
/// @param mb Owning block with initialized canonical classes.
/// @param originalId Minted original expression.
/// @param scopeId Effective lookup scope.
/// @return The original id when it is canonical, else the id of its canonical text.
NameId OrImplicationProcessor::canonicalOrFormId(Memory& mb, NameId originalId, NameId scopeId) {
    RT_SCOPE_HERE("OR_CACHE_FORM");
    assert(originalId != 0 && scopeId != 0);
    const unsigned slot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
    ScratchArena& strings = scratchArenas().forSlot(slot);
    ScratchScope canonicalScope(strings);
    const StrSpan textView = mb.nameMap.decodeView(originalId);
    const ScratchString text = ScratchString::copyFrom(strings, textView.ptr, textView.len);
    const StrSpan scopeView = mb.nameMap.decodeView(scopeId);
    const ScratchString validity = ScratchString::copyFrom(strings, scopeView.ptr, scopeView.len);
    int32_t levelCount = 0;
    const CanonicalForm canonical = canonicalizer.canonicalFormAtScope(StrSpan(text),
        StrSpan(validity), mb, strings, nullptr, levelCount, 0, true, false, false);
    return canonical.changed ? mb.nameMap.encode(canonical.text) : originalId;
}

/// @brief Whether the equalities rewriting a spelling into its entry's form are local.
/// @details Computed when a firing considers the spelling, from the current
/// classes, with the lookup canonicalization of canonicalWatchFormAt. The
/// rewritten text must be the entry's canonical form.
/// @param mb Owning block with initialized canonical classes.
/// @param originalId A spelling that differs from the canonical form.
/// @param scopeId The entry's effective scope.
/// @param canonicalId The entry's canonical form.
/// @return True when an applied equality is this block's own.
/// @invariant Every spelling of an entry rewrites to the entry's canonical form.
bool OrImplicationProcessor::rewriteLocalAt(Memory& mb, NameId originalId,
    NameId scopeId, NameId canonicalId) {
    RT_SCOPE_HERE("OR_REWRITE_LOCALITY");
    assert(originalId != 0 && scopeId != 0 && originalId != canonicalId);
    const unsigned slot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
    ScratchArena& strings = scratchArenas().forSlot(slot);
    ScratchScope canonicalScope(strings);
    WatchBridge bridge;
    int32_t levelCount = 0;
    watchProcessor.canonicalWatchFormAt(mb, originalId, scopeId, strings,
        bridge, nullptr, levelCount, 0, true, false);
    assert(bridge.changed && mb.nameMap.lookup(bridge.text) == canonicalId
        && "an OR entry's spelling rewrites to the entry's canonical form");
    return bridge.localEquality;
}

/// @brief Post the substitution tokens a view's spelling adds to its canonical form.
/// @details Both texts are copied before token ids are minted. A canonical
/// spelling adds no token. These tokens let a class change on a rewritten
/// member dirty the entry or project only this view.
/// @param mb Owning block and NameMap.
/// @param index Derived matching state.
/// @param viewId Live view slot of the spelling.
/// @param originalId The view's original expression.
/// @param canonicalId Canonical form of the view's entry.
/// @return Void.
void OrImplicationProcessor::postOrViewTokens(Memory& mb, OrImplicationMatchIndex& index,
    int32_t viewId, NameId originalId, NameId canonicalId) {
    assert(originalId != 0 && canonicalId != 0);
    if (originalId == canonicalId) return;   // a canonical spelling adds no token
    const unsigned slot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
    ScratchArena& strings = scratchArenas().forSlot(slot);
    ScratchScope tokenScope(strings);
    const StrSpan originalView = mb.nameMap.decodeView(originalId);
    const ScratchString original = ScratchString::copyFrom(strings, originalView.ptr, originalView.len);
    const StrSpan canonicalView = mb.nameMap.decodeView(canonicalId);
    const ScratchString canonical = ScratchString::copyFrom(strings, canonicalView.ptr, canonicalView.len);
    index.addViewTokens(viewId, [&](auto&& appendToken) {
        forEachSubstitutionToken(StrSpan(original), [&](StrSpan token) {
            bool shared = false;
            forEachSubstitutionToken(StrSpan(canonical), [&](StrSpan other) {
                if (equalSpans(token, other)) shared = true;
            });
            if (!shared) appendToken(mb.nameMap.encode(token));
        });
    });
}

/// @brief Derive ordered complementary views from one canonical whole OR.
/// @details Canonicalizing the parent rewrites all argument occurrences,
/// including equality leaves; refuter views use the same equality opt-in.
/// The compiled registry supplies the existing ordered flattening operation.
/// @param mb Owning block and admitted source store.
/// @param index Derived matching state.
/// @param sourceKey Active original OR and source validity.
/// @param scopeId Effective lookup scope at or below the source scope.
/// @return Void.
void OrImplicationProcessor::bindOrSourceViews(Memory& mb,
    OrImplicationMatchIndex& index, int64_t sourceKey, NameId scopeId) {
    RT_SCOPE_HERE("OR_SOURCE_VIEWS");
    const OrImplicationSource* found = mb.orImplications.find(sourceKey);
    assert(found != nullptr);
    const int32_t leafCount = found->leafCount;
    const NameId original = static_cast<NameId>(uint64_t(sourceKey) >> 32);
    const NameId sourceScope = static_cast<NameId>(uint64_t(sourceKey));
    assert(scopeId == sourceScope || mb.nameMap.isStrictAncestor(sourceScope, scopeId));
    // The leaf views of one source at one scope exist together and share an
    // entry; existing views keep theirs until the drain settles or moves it.
    const int32_t* firstLeaf = index.viewIds.find(OrViewIdentity::sourceLeaf(sourceKey, scopeId, 0));
    const bool fresh = firstLeaf == nullptr || *firstLeaf < 0;
    const int64_t cacheKey = fresh ? ensureOrCanonicalCache(mb, index, original, scopeId)
        : index.views[*firstLeaf].cacheKey;
    const OrCanonicalCache* cache = index.caches().find(cacheKey);
    assert(cache != nullptr && cache->templateId != 0);
    const NameId canonical = cache->canonicalId;
    const unsigned slot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
    ScratchArena& strings = scratchArenas().forSlot(slot);
    ScratchScope stringScope(strings);
    const StrSpan view = mb.nameMap.decodeView(canonical);
    const ScratchString text = ScratchString::copyFrom(strings, view.ptr, view.len);
    const LogicalEntity* entity = definitions.compiledEntity(extractExpressionUniversalSpan(StrSpan(text)));
    assert(entity != nullptr && entity->category == "or");
    StrSpan leaves[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
    const int32_t count = hashRuleClassifier.flattenRegistryOrLeavesScratch(*entity, StrSpan(text),
        leaves, ExecutionParameters::MAX_INSTRUCTION_ELEMENTS, strings);
    assert(count == leafCount);
    for (int32_t i = 0; i < count; ++i) {
        const OrViewIdentity identity = OrViewIdentity::sourceLeaf(sourceKey, scopeId, i);
        const int32_t* existing = index.viewIds.find(identity);
        assert(fresh == (existing == nullptr || *existing < 0)
            && "a source's leaf views at one scope exist together");
        const StrSpan complement = gl::DefinitionExpansion::negateScratch(strings, leaves[i]);
        const NameId canonicalLeaf = mb.nameMap.encode(complement);
        const int32_t templateId = mb.templateInterner.encode(complement);
        const int32_t id = index.bindView(identity, cacheKey, original, canonicalLeaf, templateId);
        if (!fresh) continue;
        postOrViewTokens(mb, index, id, original, canonical);
        projectOrUser(mb, index, id);
    }
}

/// @brief Stage the canonical work affected by one class member.
/// @details A class at or above an entry dirties that entry once. A class below
/// an entry projects all its users when the member is a token of the canonical
/// form, and only the users whose spelling adds the member otherwise; sibling
/// scopes never combine.
/// @param mb Owning block and validity tree.
/// @param index Derived matching state with its class directory built.
/// @param memberId Merged class member in NameMap space.
/// @param scopeId Validity of the merged class.
/// @param invalidate Whether to invalidate visible caches in addition to new scope projections.
/// @return Void.
void OrImplicationProcessor::stageOrClassMember(Memory& mb,
    OrImplicationMatchIndex& index, NameId memberId, NameId scopeId, bool invalidate) {
    assert(index.classScopesBuilt && memberId != 0 && scopeId != 0);
    const bool newScope = index.recordClassScope(memberId, scopeId);
    if (!newScope && !invalidate) return;
    // Dirties an entry the class can rewrite; reports whether it projects instead.
    const auto stage = [&](int64_t cacheKey) {
        const NameId cacheScope = static_cast<NameId>(uint64_t(cacheKey));
        if (cacheScope == scopeId || mb.nameMap.isStrictAncestor(scopeId, cacheScope)) {
            if (invalidate) index.markCacheDirty(cacheKey);
            return false;
        }
        return mb.nameMap.isStrictAncestor(cacheScope, scopeId) && !mb.phase3IgnoresScope(scopeId);
    };
    index.postings().forEach(orPostingKey(OrPostingKind::TokenCaches, memberId),
        [&](int64_t cacheKey) {
            if (!stage(cacheKey)) return;
            index.postings().forEach(orPostingKey(
                OrPostingKind::CacheUsers, cacheKey), [&](int64_t rawId) {
                    const OrIndexedView view = index.views[static_cast<int32_t>(rawId)];
                    assert(view.live);
                    index.projections.mint(view.identity.atScope(scopeId));
                });
        });
    index.postings().forEach(orPostingKey(OrPostingKind::TokenViews, memberId),
        [&](int64_t rawId) {
            const OrIndexedView view = index.views[static_cast<int32_t>(rawId)];
            assert(view.live);
            if (stage(view.cacheKey)) index.projections.mint(view.identity.atScope(scopeId));
        });
}

/// @brief Settle dirty entries and affected descendant views before matching.
/// @details A dirty entry canonicalizes its own canonical text once. An
/// unchanged form settles in place; a moved form moves every user to the
/// entry of the new form, merging with a live one, and its sources rebind
/// their leaves. Decoded ordering makes canonical-name creation independent
/// of posting order. No stored known fact or GPU premise registry is changed.
/// @param mb Owning block, equivalence classes and interners.
/// @param index Derived matching state and its deduplicated pending work.
/// @return Number of dirty entries sent through the shared canonical chooser.
int32_t OrImplicationProcessor::refreshOrCanonicalViews(Memory& mb,
    OrImplicationMatchIndex& index) {
    const unsigned slot = gl::WorkerContext::coreId() >= 0
        ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
    ScratchArena& arena = genScratchArenas().forSlot(slot);
    ScratchScope workScope(arena);
    DirtyState dirty = DirtyState::Clean;
    PagedVector<int64_t> cacheKeys(&arena, &dirty);
    PagedVector<OrViewIdentity> projections(&arena, &dirty);
    PagedVector<int32_t> users(&arena, &dirty);
    TypedColdSet<int64_t> sourceOwners(&arena, &dirty);
    int32_t canonicalized = 0;
    const auto pairLess = [&](int64_t a, int64_t b) {
        const NameId ao = static_cast<NameId>(uint64_t(a) >> 32);
        const NameId bo = static_cast<NameId>(uint64_t(b) >> 32);
        if (ao != bo) return compareSpans(mb.nameMap.decodeView(ao), mb.nameMap.decodeView(bo)) < 0;
        return compareSpans(mb.nameMap.decodeView(static_cast<NameId>(uint64_t(a))),
            mb.nameMap.decodeView(static_cast<NameId>(uint64_t(b)))) < 0;
    };
    while (!index.dirtyCaches.empty() || !index.projections.empty()) {
        cacheKeys.clear();
        for (int32_t i = 1; i <= index.dirtyCaches.count(); ++i)
            cacheKeys.push_back(index.dirtyCaches.keyAt(i));
        index.dirtyCaches.resetToFresh();
        {
            const auto less = [&](int32_t a, int32_t b) {
                return pairLess(cacheKeys[a], cacheKeys[b]);
            };
            PagedSortedOrdinals<decltype(less)> order(arena, cacheKeys.size(), less, 1024);
            for (int32_t i = 0; i < cacheKeys.size(); ++i) {
                const int64_t cacheKey = cacheKeys[order.next()];
                const OrCanonicalCache* found = index.caches().find(cacheKey);
                if (found == nullptr || found->templateId == 0) continue;
                const OrCanonicalCache previous = *found;
                // A retired cache can be recreated canonically before its
                // old queued update drains; that work is already complete.
                if (previous.dirty == 0) continue;
                assert(previous.dirty == 1);
                const NameId canonical = static_cast<NameId>(uint64_t(cacheKey) >> 32);
                const NameId scope = static_cast<NameId>(uint64_t(cacheKey));
                // Every spelling of the entry has this form, so the entry's own
                // text yields their common form under the current classes.
                const NameId moved = canonicalOrFormId(mb, canonical, scope);
                ++canonicalized;
                if (moved == canonical) {
                    index.settleCache(cacheKey);
                    projectOrCacheViews(mb, index, cacheKey);
                    continue;
                }
                const int64_t target = packInt32Pair(moved, scope);
                if (!index.cacheLive(target)) {
                    const int32_t templateId = mb.templateInterner.encode(
                        mb.nameMap.decodeView(moved));
                    publishOrCanonicalCache(mb, index, target,
                        OrCanonicalCache{moved, templateId, 0, 0});
                }
                users.clear();
                sourceOwners.resetToFresh();
                index.postings().forEach(orPostingKey(OrPostingKind::CacheUsers, cacheKey),
                    [&](int64_t id) {
                        users.push_back(static_cast<int32_t>(id));
                        const OrIndexedView view = index.views[static_cast<int32_t>(id)];
                        if ((view.live & OrImplicationMatchIndex::DerivedLeaf) != 0)
                            sourceOwners.mint(view.identity.owner);
                    });
                for (int32_t j = 0; j < users.size(); ++j) {
                    index.eraseViewTokens(users[j], arena);
                    index.moveView(users[j], target);
                    postOrViewTokens(mb, index, users[j], index.views[users[j]].originalId, moved);
                }
                index.retireCache(cacheKey, arena);
                for (int32_t j = 1; j <= sourceOwners.count(); ++j)
                    bindOrSourceViews(mb, index, sourceOwners.keyAt(j), scope);
                projectOrCacheViews(mb, index, target);
            }
        }
        projections.clear();
        for (int32_t i = 1; i <= index.projections.count(); ++i)
            projections.push_back(index.projections.keyAt(i));
        index.projections.resetToFresh();
        const auto less = [&](int32_t a, int32_t b) {
            const OrViewIdentity x = projections[a], y = projections[b];
            if (x.owner != y.owner) return pairLess(x.owner, y.owner);
            if (x.scope != y.scope)
                return compareSpans(mb.nameMap.decodeView(x.scope), mb.nameMap.decodeView(y.scope)) < 0;
            return x.leaf < y.leaf;
        };
        PagedSortedOrdinals<decltype(less)> order(arena, projections.size(), less, 1024);
        for (int32_t i = 0; i < projections.size(); ++i) {
            const OrViewIdentity identity = projections[order.next()];
            if (mb.phase3IgnoresScope(identity.scope)) continue;
            const int32_t* existing = index.viewIds.find(identity);
            if (existing != nullptr && *existing >= 0) continue;
            NameId original = 0;
            bool derivedLeaf = false;
            index.postings().forEach(orPostingKey(OrPostingKind::OwnerViews, identity.owner),
                [&](int64_t rawId) {
                    const OrIndexedView view = index.views[static_cast<int32_t>(rawId)];
                    if (view.identity.leaf != identity.leaf
                        || !mb.nameMap.isStrictAncestor(view.identity.scope, identity.scope)) return;
                    assert(original == 0 || original == view.originalId);
                    original = view.originalId;
                    derivedLeaf = (view.live & OrImplicationMatchIndex::DerivedLeaf) != 0;
                });
            // Queued work whose last owner was retired has no remaining view.
            if (original == 0) continue;
            if (derivedLeaf)
                bindOrSourceViews(mb, index, identity.owner, identity.scope);
            else
                bindOrCanonicalView(mb, index, identity, original);
        }
    }
    index.maintain();
    return canonicalized;
}
}
