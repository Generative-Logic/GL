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

/// @brief Count statements beneath one exact parent/signature OR cohort.
///
/// @details Includes branch descendants and excludes equal-looking signatures
/// under other parents. The scan is read-only and uses no interner mint.
/// @param body Owning resident block.
/// @param parent Parent validity text.
/// @param signature Original OR signature.
/// @return Active branch statement count.
/// @see retireDuplicateOrCohorts; retireDuplicateOrSources.
int32_t ExpressionAnalyzer::orCohortStatementCount(const Memory& body,
    StrSpan parent, StrSpan signature) const {
    const StrSpan boundary("_boundary_ordis_", 16);
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

/// @brief Retire symmetric duplicate exclusion sources before matching.
///
/// @details Reuses source canonical ids and paged scratch sorting. Groups never
/// cross source scopes. The source of the largest open cohort wins, with the
/// existing signature tie-break; absent open cohorts, source insertion order wins.
/// @param body Owning resident block with current equality classes.
/// @return Void.
/// @invariant Original leaves, levels, locality and history are never transferred.
/// @see canonicalOrIdAtScope; retireOrImplicationSource.
void ExpressionAnalyzer::retireDuplicateOrSources(Memory& body) {
    auto& sources = body.orImplications;
    if (sources.sources.count() < 2) return;
    const unsigned slot = g_currentCoreId >= 0
        ? static_cast<unsigned>(g_currentCoreId) : genScratchArenas().slotCount() - 1;
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
        const auto* entity = compiledEntity(extractExpressionUniversalSpan(text));
        if (!isOrCompactSpan(text) || entity == nullptr || entity->symmetricDigitSlots[0] == 0) continue;
        const NameId canonical = canonicalOrIdAtScope(body, original, body.nameMap.decodeView(scope));
        const StrSpan signature = body.nameMap.decodeView(original);
        const StrSpan parent = body.nameMap.decodeView(scope);
        const int32_t cohort = lookupOrCohortId(body.lbStateInterner,
            body.lbStateInterner.lookup(parent), body.lbStateInterner.lookup(signature));
        const int32_t score = cohort != 0 && body.orDisjunctCount.lookup(cohort) != 0
            ? orCohortStatementCount(body, parent, signature) : -1;
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
void ExpressionAnalyzer::registerOrImplicationSource(Memory& mb,
    const IntEncodedExpr& statement, ColdMail* publicationMail)
{
    assert(!statement.negation && statement.originalId != 0);
    const int64_t key = packStatementKey(statement.originalId, statement.validityId);
    const StatementFlags* known = mb.intKnownStatements.find(
        StatementKey{statement.originalId, statement.validityId});
    assert(known != nullptr);
    const bool local = known->local;
    const int32_t levelsId = mb.intStatementLevelsMap.lookup(key);
    assert(levelsId != 0 && mb.intStatementLevelsMap.runLen(levelsId) > 0);
    const unsigned slot = g_currentCoreId >= 0
        ? static_cast<unsigned>(g_currentCoreId) : scratchArenas().slotCount() - 1;
    ScratchArena& strings = scratchArenas().forSlot(slot);
    ScratchScope stringScope(strings);
    const StrSpan view = mb.nameMap.decodeView(statement.originalId);
    const ScratchString original = ScratchString::copyFrom(strings, view.ptr, view.len);
    const LogicalEntity* entity = compiledEntity(extractExpressionUniversalSpan(StrSpan(original)));
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
    const int32_t count = flattenRegistryOrLeavesScratch(*entity, StrSpan(original),
        flat, ExecutionParameters::MAX_INSTRUCTION_ELEMENTS, strings);
    OrImplicationLeaf leaves[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
    for (int32_t i = 0; i < count; ++i) {
        leaves[i].literalId = mb.nameMap.encode(flat[i]);
        leaves[i].complementId = mb.nameMap.encode(negateScratch(strings, flat[i]));
    }
    int levels[256];
    const int32_t levelCount = mb.intStatementLevelsMap.runLen(levelsId);
    assert(levelCount <= 256);
    for (int32_t i = 0; i < levelCount; ++i)
        levels[i] = mb.intStatementLevelsMap.valueAt(levelsId, i);
    const bool inserted = mb.orImplications.insert(key, statement.originalId,
        leaves, count, levels, levelCount, local);
    assert(inserted);
    activateDirectOrRefutations(mb, statement,
        publicationMail ? *publicationMail : mb.sameIterationInternalMail);
}

/// @brief Build newly needed signed-core views and reconcile staged changes.
/// @details Source events are collected before one scan for all new cores.
/// Existing core arrivals avoid that scan. Canonical changes settle before
/// exact complementary-template buckets queue comparable source scopes.
/// @param mb Owning block and its canonical and derived OR state.
/// @return Void.
void ExpressionAnalyzer::prepareOrImplicationMatches(Memory& mb)
{
    RT_SCOPE_HERE("OR_IMPLICATION_MATCHES");
    auto& sources = mb.orImplications;
    auto& index = mb.orImplicationMatches;
    if (!index.built || !sources.pendingSources.empty() || !sources.pendingClasses.empty()) {
        retireDuplicateOrSources(mb);
    }
    const unsigned slot = g_currentCoreId >= 0
        ? static_cast<unsigned>(g_currentCoreId) : genScratchArenas().slotCount() - 1;
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
            index.changedViews.mint(index.views[view].identity);
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
        const int32_t* slotId = index.viewIds.find(identity);
        if (slotId == nullptr || *slotId < 0) continue;
        const OrIndexedView view = index.views[*slotId];
        const OrCanonicalCache* cache = index.caches().find(view.cacheKey);
        assert(cache != nullptr && cache->dirty == 0 && cache->templateId != 0);
        const int32_t templateId = view.templateId;
        const bool fact = identity.isFact();
        index.postings().forEach(orPostingKey(fact ? OrPostingKind::WatchScopes
                                                : OrPostingKind::FactScopes, templateId),
            [&](int64_t rawScope) {
                const NameId scope = static_cast<NameId>(rawScope);
                if (!mb.nameMap.comparable(identity.scope, scope)) return;
                index.postings().forEach(orPostingKey(fact ? OrPostingKind::WatchKey
                                                        : OrPostingKind::FactKey,
                    packInt32Pair(templateId, scope)), [&](int64_t rawId) {
                        const OrIndexedView other = index.views[static_cast<int32_t>(rawId)];
                        assert(other.live);
                        const int64_t sourceKey = fact ? other.identity.owner : identity.owner;
                        if (sources.find(sourceKey) == nullptr) return;
                        index.evaluations.mint(OrEvaluation{sourceKey,
                            mb.nameMap.deeperOf(identity.scope, scope)});
                    });
            });
    }
    index.changedViews.resetToFresh();
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
void ExpressionAnalyzer::retireOrImplicationSource(Memory& mb, int64_t sourceKey)
{
    const bool retired = mb.orImplications.retire(sourceKey);
    assert(retired);
    auto& index = mb.orImplicationMatches;
    const unsigned slot = g_currentCoreId >= 0
        ? static_cast<unsigned>(g_currentCoreId) : genScratchArenas().slotCount() - 1;
    ScratchArena& arena = genScratchArenas().forSlot(slot);
    index.retireSource(sourceKey, arena);
}



/// @brief Queue current-state reconciliation for a relevant known-fact key.
/// @details An unbuilt index will read current known state during bootstrap.
/// A built index only watches signed cores that still have active OR sources.
/// @param mb Owning block and its source/index state.
/// @param key Original expression and validity whose known membership changed.
/// @return Void.
void ExpressionAnalyzer::stageOrFactChange(Memory& mb, int64_t key)
{
    if (!mb.orImplicationMatches.built) return;
    const NameId original = static_cast<NameId>(uint64_t(key) >> 32);
    const StrSpan text = mb.nameMap.decodeView(original);
    const bool negated = text.len > 0 && text.ptr[0] == '!';
    const NameId core = mb.nameMap.lookup(extractExpressionUniversalSpan(text));
    assert(core != 0);
    const int64_t signedCore = packInt32Pair(core, negated);
    if (!mb.orImplicationMatches.postings().containsKey(
            orPostingKey(OrPostingKind::SourceCore, signedCore))) return;
    mb.orImplications.pendingFacts.push_back(key);
}

}
