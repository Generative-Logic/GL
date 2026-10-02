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

/// @brief Seed class-scope token postings for canonical OR lookup.
/// @details The initial pass reads each serialized class once; further calls
/// are cache hits. Class additions are handled by stageOrClassMember.
/// @param mb Owning block and current equivalence classes.
/// @param index Derived matching state on the block arena.
/// @return Void.
void ExpressionAnalyzer::buildOrClassDirectory(Memory& mb,
    OrImplicationMatchIndex& index)
{
    if (index.classScopesBuilt) return;
    const unsigned slot = g_currentCoreId >= 0
        ? static_cast<unsigned>(g_currentCoreId) : genScratchArenas().slotCount() - 1;
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

/// @brief Publish the chooser result with its ordered dependency-token stream.
/// @details Tokens stream directly into the index transaction, avoiding a
/// temporary collection and preserving NameMap mint order. The index owns all
/// cache, reverse-token and scalar-view updates; this function owns parsing.
/// @param mb Owning block and expression interner.
/// @param index Derived matching index receiving the complete update.
/// @param key Original expression and effective scope.
/// @param value Settled canonical result from the existing chooser.
/// @param bridge Canonicalizer dependency expressions on stable caller storage.
/// @return Void.
void ExpressionAnalyzer::publishOrCanonicalCache(Memory& mb,
    OrImplicationMatchIndex& index, int64_t key, OrCanonicalCache value,
    const WatchBridge& bridge)
{
    const unsigned stringSlot = g_currentCoreId >= 0
        ? static_cast<unsigned>(g_currentCoreId) : scratchArenas().slotCount() - 1;
    const unsigned pageSlot = g_currentCoreId >= 0
        ? static_cast<unsigned>(g_currentCoreId) : genScratchArenas().slotCount() - 1;
    ScratchArena& strings = scratchArenas().forSlot(stringSlot);
    ScratchScope stringScope(strings);
    const NameId originalId = static_cast<NameId>(uint64_t(key) >> 32);
    index.publishCache(key, value, [&](auto&& appendToken) {
        const auto collect = [&](StrSpan text) {
            forEachSubstitutionToken(text, [&](StrSpan token) {
                appendToken(mb.nameMap.encode(token));
            });
        };
        const NameId expressions[2] = {originalId, value.canonicalId};
        for (int32_t i = 0; i < 2; ++i) {
            if (i == 1 && value.canonicalId == originalId) break;
            const StrSpan raw = mb.nameMap.decodeView(expressions[i]);
            const ScratchString owned = ScratchString::copyFrom(strings, raw.ptr, raw.len);
            collect(StrSpan(owned));
        }
        for (int32_t i = 0; i < bridge.eqN; ++i) collect(bridge.eqText[i]);
    }, genScratchArenas().forSlot(pageSlot));
}

/// @brief Queue existing owners at relevant strict-descendant equality scopes.
/// @details Token and class-scope postings avoid a source-by-scope product.
/// Queued identities are stable even when view slots are later recycled.
/// @param mb Owning block and scope tree.
/// @param index Derived matching state.
/// @param cacheKey Live original expression and lookup scope.
/// @return Void.
void ExpressionAnalyzer::projectOrCacheViews(Memory& mb,
    OrImplicationMatchIndex& index, int64_t cacheKey)
{
    RT_SCOPE_HERE("OR_CACHE_PROJECT");
    const NameId cacheScope = static_cast<NameId>(uint64_t(cacheKey));
    index.postings().forEach(orPostingKey(OrPostingKind::CacheTokens, cacheKey),
        [&](int64_t token) {
            index.postings().forEach(orPostingKey(OrPostingKind::ClassScopes, token),
                [&](int64_t rawScope) {
                    const NameId scope = static_cast<NameId>(rawScope);
                    if (!mb.nameMap.isStrictAncestor(cacheScope, scope)
                        || mb.phase3IgnoresScope(scope)) return;
                    index.postings().forEach(orPostingKey(
                        OrPostingKind::CacheUsers, cacheKey), [&](int64_t rawId) {
                            const OrIndexedView view = index.views[static_cast<int32_t>(rawId)];
                            assert(view.live);
                            const OrViewIdentity projected = view.identity.atScope(scope);
                            const int32_t* existing = index.viewIds.find(projected);
                            if (existing == nullptr || *existing < 0)
                                index.projections.mint(projected);
                        });
                });
        });
}

/// @brief Bind an OR lookup view to one cached shared-canonicalizer result.
/// @details The cache is keyed by original expression and effective scope.
/// Additional owners reuse it, including a dirty cache awaiting the drain.
/// @param mb Owning block, equivalence classes and interners.
/// @param index Derived matching state.
/// @param identity Original owner, leaf role and effective scope.
/// @param originalId Fully instantiated complementary leaf or known fact.
/// @return The live view slot for the supplied identity.
int32_t ExpressionAnalyzer::bindOrCanonicalView(Memory& mb,
    OrImplicationMatchIndex& index, OrViewIdentity identity, NameId originalId)
{
    assert(originalId != 0 && identity.scope != 0);
    const NameId ownerScope = static_cast<NameId>(uint64_t(identity.owner));
    assert(ownerScope == identity.scope
        || mb.nameMap.isStrictAncestor(ownerScope, identity.scope));
    const int64_t cacheKey = ensureOrCanonicalCache(mb, index, originalId, identity.scope);
    const int32_t id = index.bindView(identity, cacheKey);
    projectOrCacheViews(mb, index, cacheKey);
    return id;
}

/// @brief Reuse or create a canonical whole-expression/scope cache.
/// @details Canonicalization and dependency-token indexing happen once per
/// original/scope pair, shared by scalar facts and derived OR leaf views.
/// @param mb Owning block, classes and interners.
/// @param index Derived matching state.
/// @param originalId Original whole expression in NameMap space.
/// @param scopeId Effective lookup scope.
/// @return Packed original-expression/effective-scope cache key.
int64_t ExpressionAnalyzer::ensureOrCanonicalCache(Memory& mb,
    OrImplicationMatchIndex& index, NameId originalId, NameId scopeId)
{
    assert(originalId != 0 && scopeId != 0);
    buildOrClassDirectory(mb, index);
    const int64_t cacheKey = packInt32Pair(originalId, scopeId);
    const OrCanonicalCache* cache = index.caches().find(cacheKey);
    const bool fresh = cache == nullptr || cache->templateId == 0;
    if (fresh) {
        RT_SCOPE_HERE("OR_CACHE_FRESH");
        const unsigned slot = g_currentCoreId >= 0
            ? static_cast<unsigned>(g_currentCoreId) : scratchArenas().slotCount() - 1;
        ScratchArena& strings = scratchArenas().forSlot(slot);
        ScratchScope canonicalScope(strings);
        WatchBridge bridge;
        int levels[256];
        int32_t levelCount = 0;
        canonicalWatchFormAt(mb, originalId, scopeId, strings,
            bridge, levels, levelCount, 256, true);
        const NameId canonicalId = bridge.changed ? mb.nameMap.encode(bridge.text) : originalId;
        const int32_t templateId = mb.templateInterner.encode(
            mb.nameMap.decodeView(canonicalId));
        publishOrCanonicalCache(mb, index, cacheKey,
            OrCanonicalCache{canonicalId, templateId, 0, bridge.localEquality}, bridge);
    }
    return cacheKey;
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
void ExpressionAnalyzer::bindOrSourceViews(Memory& mb,
    OrImplicationMatchIndex& index, int64_t sourceKey, NameId scopeId)
{
    RT_SCOPE_HERE("OR_SOURCE_VIEWS");
    const OrImplicationSource* found = mb.orImplications.find(sourceKey);
    assert(found != nullptr);
    const int32_t leafCount = found->leafCount;
    const NameId original = static_cast<NameId>(uint64_t(sourceKey) >> 32);
    const NameId sourceScope = static_cast<NameId>(uint64_t(sourceKey));
    assert(scopeId == sourceScope || mb.nameMap.isStrictAncestor(sourceScope, scopeId));
    const int64_t cacheKey = ensureOrCanonicalCache(mb, index, original, scopeId);
    const OrCanonicalCache* cache = index.caches().find(cacheKey);
    assert(cache != nullptr && cache->templateId != 0);
    const NameId canonical = cache->canonicalId;
    const unsigned slot = g_currentCoreId >= 0
        ? static_cast<unsigned>(g_currentCoreId) : scratchArenas().slotCount() - 1;
    ScratchArena& strings = scratchArenas().forSlot(slot);
    ScratchScope stringScope(strings);
    const StrSpan view = mb.nameMap.decodeView(canonical);
    const ScratchString text = ScratchString::copyFrom(strings, view.ptr, view.len);
    const LogicalEntity* entity = compiledEntity(extractExpressionUniversalSpan(StrSpan(text)));
    assert(entity != nullptr && entity->category == "or");
    StrSpan leaves[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
    const int32_t count = flattenRegistryOrLeavesScratch(*entity, StrSpan(text),
        leaves, ExecutionParameters::MAX_INSTRUCTION_ELEMENTS, strings);
    assert(count == leafCount);
    for (int32_t i = 0; i < count; ++i) {
        const StrSpan complement = negateScratch(strings, leaves[i]);
        const NameId canonicalLeaf = mb.nameMap.encode(complement);
        const int32_t templateId = mb.templateInterner.encode(complement);
        index.bindView(OrViewIdentity::sourceLeaf(sourceKey, scopeId, i), cacheKey,
            canonicalLeaf, templateId);
    }
    projectOrCacheViews(mb, index, cacheKey);
}

/// @brief Stage the canonical work affected by one class member.
/// @details A class at or above a cache dirties that cache once. A class below
/// a cache projects only its live owners; sibling scopes never combine.
/// @param mb Owning block and validity tree.
/// @param index Derived matching state with its class directory built.
/// @param memberId Merged class member in NameMap space.
/// @param scopeId Validity of the merged class.
/// @param invalidate Whether to invalidate visible caches in addition to new scope projections.
/// @return Void.
void ExpressionAnalyzer::stageOrClassMember(Memory& mb,
    OrImplicationMatchIndex& index, NameId memberId, NameId scopeId, bool invalidate)
{
    assert(index.classScopesBuilt && memberId != 0 && scopeId != 0);
    const bool newScope = index.recordClassScope(memberId, scopeId);
    if (!newScope && !invalidate) return;
    index.postings().forEach(orPostingKey(OrPostingKind::TokenCaches, memberId),
        [&](int64_t cacheKey) {
            const NameId cacheScope = static_cast<NameId>(uint64_t(cacheKey));
            if (cacheScope == scopeId || mb.nameMap.isStrictAncestor(scopeId, cacheScope)) {
                if (invalidate) index.markCacheDirty(cacheKey);
            } else if (mb.nameMap.isStrictAncestor(cacheScope, scopeId)
                && !mb.phase3IgnoresScope(scopeId)) {
                index.postings().forEach(orPostingKey(
                    OrPostingKind::CacheUsers, cacheKey), [&](int64_t rawId) {
                        const OrIndexedView view = index.views[static_cast<int32_t>(rawId)];
                        assert(view.live);
                        index.projections.mint(
                            view.identity.atScope(scopeId));
                    });
            }
        });
}

/// @brief Settle dirty caches and affected descendant views before matching.
/// @details Every rekey unlinks all old buckets before linking any new bucket.
/// Decoded ordering makes canonical-name creation independent of posting order.
/// No stored known fact or GPU premise registry is changed.
/// @param mb Owning block, equivalence classes and interners.
/// @param index Derived matching state and its deduplicated pending work.
/// @return Number of dirty caches sent through the shared canonical chooser.
int32_t ExpressionAnalyzer::refreshOrCanonicalViews(Memory& mb,
    OrImplicationMatchIndex& index)
{
    const unsigned slot = g_currentCoreId >= 0
        ? static_cast<unsigned>(g_currentCoreId) : genScratchArenas().slotCount() - 1;
    ScratchArena& arena = genScratchArenas().forSlot(slot);
    ScratchScope workScope(arena);
    DirtyState dirty = DirtyState::Clean;
    PagedVector<int64_t> cacheKeys(&arena, &dirty);
    PagedVector<OrViewIdentity> projections(&arena, &dirty);
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
                const NameId original = static_cast<NameId>(uint64_t(cacheKey) >> 32);
                const NameId scope = static_cast<NameId>(uint64_t(cacheKey));
                ScratchArena& strings = scratchArenas().forSlot(slot);
                ScratchScope canonicalScope(strings);
                WatchBridge bridge;
                int levels[256];
                int32_t levelCount = 0;
                canonicalWatchFormAt(mb, original, scope, strings,
                    bridge, levels, levelCount, 256, true);
                const NameId canonical = bridge.changed ? mb.nameMap.encode(bridge.text) : original;
                ++canonicalized;
                const int32_t templateId = mb.templateInterner.encode(
                    mb.nameMap.decodeView(canonical));
                sourceOwners.resetToFresh();
                index.postings().forEach(orPostingKey(OrPostingKind::CacheUsers, cacheKey),
                    [&](int64_t id) {
                        const OrIndexedView view = index.views[static_cast<int32_t>(id)];
                        if ((view.live & OrImplicationMatchIndex::DerivedLeaf) != 0)
                            sourceOwners.mint(view.identity.owner);
                    });
                publishOrCanonicalCache(mb, index, cacheKey,
                    OrCanonicalCache{canonical, templateId, 0, bridge.localEquality}, bridge);
                for (int32_t j = 1; j <= sourceOwners.count(); ++j)
                    bindOrSourceViews(mb, index, sourceOwners.keyAt(j), scope);
                projectOrCacheViews(mb, index, cacheKey);
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
                    const NameId candidate = static_cast<NameId>(uint64_t(view.cacheKey) >> 32);
                    assert(original == 0 || original == candidate);
                    original = candidate;
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
