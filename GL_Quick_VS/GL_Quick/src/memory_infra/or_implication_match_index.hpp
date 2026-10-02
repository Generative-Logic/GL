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

#include "or_implication_postings.hpp"
#include "scratch_arena.hpp"

namespace gl {

    /// @brief Independent lookup domains sharing one pool-backed posting index.
    /// @details Domain separation preserves full-width expression/validity keys.
    enum class OrPostingKind : int64_t {
        CacheUsers, OwnerViews, ScopeViews, TokenCaches, ClassScopes,
        WatchKey, FactKey, WatchScopes, FactScopes, SourceCore, OwnerCores,
        FactCore, CacheScope, CacheTokens
    };

    /// @brief Exact domain and full-width key of one posting bucket.
    /// @details Both fields have unique object representations for cold hashing.
    struct OrPostingKey { int64_t domain; int64_t key; };
    template<> struct Codec<OrPostingKey> : IdentityKeyCodec<OrPostingKey> {};

    /// @brief Form a domain-separated OR posting key without narrowing ids.
    /// @details The payload may be a scalar id or a packed pair of NameIds.
    /// @param kind The independent lookup domain.
    /// @param key The domain's full-width logical key.
    /// @return A fixed, padding-free posting key.
    inline OrPostingKey orPostingKey(OrPostingKind kind, int64_t key) {
        return OrPostingKey{static_cast<int64_t>(kind), key};
    }

    /// @brief Stable identity of one source-leaf or known-fact lookup view.
    /// @details Owner is the original expression/validity pair. Leaf -1 names
    /// a known fact; nonnegative leaves name an OR source's ordered leaf.
    /// Evaluation requests use a separate OrEvaluation record.
    struct OrViewIdentity {
        static constexpr int32_t FactLeaf = -1;
        int64_t owner;
        NameId scope;
        int32_t leaf;

        /// @brief Construct the lookup identity of a known fact.
        /// @details The fact role is independent of any source sharing its owner.
        /// @param owner Original expression and source validity.
        /// @param scope Effective lookup validity.
        /// @return A fact identity with unchanged packed-key representation.
        static OrViewIdentity knownFact(int64_t owner, NameId scope) {
            assert(owner != 0 && scope != 0);
            return {owner, scope, FactLeaf};
        }

        /// @brief Construct the lookup identity of an ordered source leaf.
        /// @details Leaf ordinals retain the original OR's order at every scope.
        /// @param owner Original OR and source validity.
        /// @param scope Effective lookup validity.
        /// @param leaf Nonnegative ordered leaf ordinal.
        /// @return An OR-leaf lookup identity.
        static OrViewIdentity sourceLeaf(int64_t owner, NameId scope, int32_t leaf) {
            assert(owner != 0 && scope != 0 && leaf >= 0);
            return {owner, scope, leaf};
        }

        /// @brief Preserve a view's owner and role at another lookup scope.
        /// @details Scope ancestry is checked by the matching caller.
        /// @param target Effective scope of the projected view.
        /// @return The same logical view at the supplied scope.
        OrViewIdentity atScope(NameId target) const {
            assert(target != 0 && leaf >= FactLeaf);
            return {owner, target, leaf};
        }

        /// @brief Distinguish a known fact from an OR source leaf.
        /// @details Evaluation work cannot enter this identity type.
        /// @return True for a known-fact view, false for an ordered source leaf.
        bool isFact() const { assert(leaf >= FactLeaf); return leaf == FactLeaf; }
    };
    template<> struct Codec<OrViewIdentity> : IdentityKeyCodec<OrViewIdentity> {};

    /// @brief One OR source to evaluate at a candidate scope.
    /// @details The explicit zero field makes every byte of the cold key defined;
    /// this record has no leaf or fact role and never identifies a lookup view.
    struct OrEvaluation { int64_t owner; NameId scope; int32_t reserved = 0; };
    template<> struct Codec<OrEvaluation> : IdentityKeyCodec<OrEvaluation> {};

    /// @brief Shared canonical result for one original-expression/scope pair.
    /// @details Canonical ids belong to NameMap; template ids belong to the
    /// rejected-map template interner. Dirty entries require the shared chooser.
    struct OrCanonicalCache {
        NameId canonicalId;
        int32_t templateId;
        int32_t dirty;
        int32_t localEquality;
    };

    /// @brief Recyclable lookup view referencing its shared canonical cache.
    /// @details Every link and identity is integer-valued and relocatable.
    struct OrIndexedView {
        OrViewIdentity identity;
        int64_t cacheKey;
        int32_t nextFree;
        int32_t live;
        NameId canonicalId;
        int32_t templateId;
    };

    /// @brief Derived indexes and deduplicated work for on-demand OR matching.
    ///
    /// @details
    /// Source state lives in OrImplications. This index groups canonical views
    /// by exact template/validity, source owner, cache, token and scope. One
    /// domain-separated posting engine supplies all lookup directions without
    /// allocating a separate hash-table object for every direction on every LB.
    /// The prover supplies canonical results through its existing watch chooser.
    /// Canonical release clears this object; raw eviction preserves its pages.
    /// @invariant Lookup indexes never supply proof validity or provenance.
    /// @see OrImplications, OrPostingIndex
    struct OrImplicationMatchIndex {
        static constexpr int32_t DerivedLeaf = 2;
        DirtyState dirty = DirtyState::Clean;
    private:
        OrPostingIndex<OrPostingKey> postings_;
    public:
        TypedColdMap<OrViewIdentity, int32_t> viewIds;
        PagedVector<OrIndexedView> views;
    private:
        TypedColdMap<int64_t, OrCanonicalCache> caches_;
    public:
        TypedColdSet<int64_t> initializedCores;
        TypedColdSet<int64_t> dirtyCaches;
        TypedColdSet<OrViewIdentity> projections;
        TypedColdSet<OrViewIdentity> changedViews;
        TypedColdSet<OrEvaluation> evaluations;
        int32_t freeView = -1;
        int32_t liveViews = 0;
        int32_t deadIdentities = 0;
        int32_t deadCaches = 0;
        bool built = false;
        bool classScopesBuilt = false;

        /// @brief Bind empty derived matching state to one block arena.
        /// @details A private dirty flag keeps cache changes out of the logical
        /// source image; the arena must outlive all derived containers.
        /// @param arena The owning block's deloadable arena.
        /// @return A constructed empty matching index.
        explicit OrImplicationMatchIndex(LbArena* arena)
            : postings_(arena), viewIds(arena, &dirty), views(arena, &dirty),
              caches_(arena, &dirty), initializedCores(arena, &dirty),
              dirtyCaches(arena, &dirty), projections(arena, &dirty),
              changedViews(arena, &dirty), evaluations(arena, &dirty) {}

        /// @brief Expose lookup postings without permitting partial mutations.
        /// @details Relationship writes belong to this index's lifecycle methods.
        /// @return The read-only posting index.
        const OrPostingIndex<OrPostingKey>& postings() const { return postings_; }

        /// @brief Expose canonical cache records for matching decisions.
        /// @details Publication, dirtying and retirement use the index methods.
        /// @return The read-only canonical cache table.
        const TypedColdMap<int64_t, OrCanonicalCache>& caches() const { return caches_; }

        /// @brief Link a source and its watched signed predicate core together.
        /// @details Exact duplicates preserve both existing lookup directions.
        /// @param source Original OR and source scope.
        /// @param core Signed complementary predicate core.
        /// @return Void.
        void watchSourceCore(int64_t source, int64_t core) {
            assert(source != 0 && core != 0);
            postings_.insert(orPostingKey(OrPostingKind::SourceCore, core), source);
            postings_.insert(orPostingKey(OrPostingKind::OwnerCores, source), core);
        }

        /// @brief Associate a known fact with a watched signed predicate core.
        /// @details Its canonical view is bound after the existing chooser runs.
        /// @param owner Original fact and source scope.
        /// @param core Signed predicate core.
        /// @return Void.
        void recordFactCore(int64_t owner, int64_t core) {
            assert(owner != 0 && core != 0);
            postings_.insert(orPostingKey(OrPostingKind::FactCore, core), owner);
        }

        /// @brief Record a class member's scope for descendant projection.
        /// @details Duplicate class observations retain the existing posting.
        /// @param member Whole substitution-token id.
        /// @param scope Validity containing the equality class.
        /// @return True when this member/scope edge is new.
        bool recordClassScope(NameId member, NameId scope) {
            assert(member != 0 && scope != 0);
            return postings_.insert(orPostingKey(OrPostingKind::ClassScopes, member), scope);
        }

        /// @brief Dirty a live cache and queue its canonical reconciliation.
        /// @details Repeated invalidations deduplicate without replacing its
        /// current buckets before the canonicalizer supplies the new value.
        /// @param key Original expression and effective scope.
        /// @return Void.
        void markCacheDirty(int64_t key) {
            const OrCanonicalCache* found = caches_.find(key);
            assert(found != nullptr && found->templateId != 0);
            OrCanonicalCache changed = *found;
            changed.dirty = 1;
            caches_.upsert(key, changed);
            dirtyCaches.mint(key);
        }

        /// @brief Publish a canonical result and update all scalar relationships.
        /// @details Fresh publication creates the scope/token directory; rekeying
        /// unlinks old scalar buckets and token pairs before installing the new
        /// ones. Whole-OR leaf spelling is refreshed by bindOrSourceViews using
        /// this same cache. Input token order is preserved, including duplicates.
        /// @param key Original expression and effective scope.
        /// @param value Settled canonical result from the existing chooser.
        /// @tparam TokenVisitor Synchronous heap-free dependency-token visitor.
        /// @param visitTokens Emits original, canonical and dependency tokens in order.
        /// @param scratch Page-tier scratch distinct from the index's arena.
        /// @return Void.
        template<class TokenVisitor>
        void publishCache(int64_t key, OrCanonicalCache value,
            TokenVisitor&& visitTokens, ScratchArena& scratch) {
            assert(key != 0 && value.canonicalId != 0 && value.templateId != 0 && value.dirty == 0);
            const OrCanonicalCache* previous = caches_.find(key);
            const bool fresh = previous == nullptr || previous->templateId == 0;
            if (fresh) {
                if (previous != nullptr) { assert(deadCaches > 0); --deadCaches; }
            } else {
                assert(postings_.containsKey(orPostingKey(OrPostingKind::CacheUsers, key)));
                // Only other posting domains mutate; this live user chain stays intact.
                postings_.forEach(orPostingKey(OrPostingKind::CacheUsers, key),
                    [&](int64_t rawId) {
                        const int32_t id = static_cast<int32_t>(rawId);
                        if ((views[id].live & DerivedLeaf) == 0)
                            detachCanonical(id, views[id].templateId);
                    });
                eraseCacheTokens(key, scratch);
            }
            caches_.upsert(key, value);
            visitTokens([&](NameId token) {
                assert(token != 0);
                postings_.insert(orPostingKey(OrPostingKind::TokenCaches, token), key);
                postings_.insert(orPostingKey(OrPostingKind::CacheTokens, key), token);
            });
            if (fresh)
                postings_.insert(orPostingKey(OrPostingKind::CacheScope,
                    static_cast<NameId>(uint64_t(key))), key);
            if (!fresh) postings_.forEach(orPostingKey(OrPostingKind::CacheUsers, key),
                [&](int64_t rawId) {
                    const int32_t id = static_cast<int32_t>(rawId);
                    if ((views[id].live & DerivedLeaf) == 0)
                        attachCanonical(id, value.templateId);
                    changedViews.mint(views[id].identity);
                });
        }

        /// @brief Remove a fact's lookup state while retaining any OR-source role.
        /// @details A missing fact-core posting is the defined removal case for
        /// a queued owner that is no longer watched or known.
        /// @param owner Original fact and source scope.
        /// @param core Signed predicate core.
        /// @param scratch Page-tier temporary storage.
        /// @return Void.
        void eraseFact(int64_t owner, int64_t core, ScratchArena& scratch) {
            eraseOwnerViews(owner, true, scratch);
            postings_.erase(orPostingKey(OrPostingKind::FactCore, core), owner);
        }

        /// @brief Remove a retired source and facts whose last watcher leaves.
        /// @details All source-core and view/cache relationships are removed in
        /// the same operation. Independent fact views of the owner survive.
        /// @param source Original OR and source scope already logically retired.
        /// @param scratch Page-tier temporary storage.
        /// @return Void.
        void retireSource(int64_t source, ScratchArena& scratch) {
            ScratchScope work(scratch);
            DirtyState temporaryDirty = DirtyState::Clean;
            PagedVector<int64_t> cores(&scratch, &temporaryDirty);
            PagedVector<int64_t> facts(&scratch, &temporaryDirty);
            eraseOwnerViews(source, false, scratch);
            postings_.forEach(orPostingKey(OrPostingKind::OwnerCores, source),
                [&](int64_t core) { cores.push_back(core); });
            for (int32_t i = 0; i < cores.size(); ++i) {
                const int64_t core = cores[i];
                const bool ownerRemoved = postings_.erase(orPostingKey(OrPostingKind::OwnerCores, source), core);
                const bool sourceRemoved = postings_.erase(orPostingKey(OrPostingKind::SourceCore, core), source);
                assert(ownerRemoved && sourceRemoved);
                if (postings_.containsKey(orPostingKey(OrPostingKind::SourceCore, core))) continue;
                facts.clear();
                postings_.forEach(orPostingKey(OrPostingKind::FactCore, core),
                    [&](int64_t fact) { facts.push_back(fact); });
                for (int32_t j = 0; j < facts.size(); ++j) {
                    eraseOwnerViews(facts[j], true, scratch);
                    const bool removed = postings_.erase(orPostingKey(OrPostingKind::FactCore, core), facts[j]);
                    assert(removed);
                }
                const bool initialized = initializedCores.erase(core);
                assert(initialized);
            }
        }

    private:
        /// @brief Remove both token-to-cache and cache-to-token postings.
        /// @details Token ids are captured before unlinking recyclable nodes.
        /// @param key Live canonical cache identity.
        /// @param scratch Page-tier temporary storage.
        /// @return Void.
        void eraseCacheTokens(int64_t key, ScratchArena& scratch) {
            ScratchScope work(scratch);
            DirtyState temporaryDirty = DirtyState::Clean;
            PagedVector<NameId> tokens(&scratch, &temporaryDirty);
            postings_.forEach(orPostingKey(OrPostingKind::CacheTokens, key),
                [&](int64_t token) { tokens.push_back(static_cast<NameId>(token)); });
            for (int32_t i = 0; i < tokens.size(); ++i) {
                const bool reverse = postings_.erase(orPostingKey(OrPostingKind::TokenCaches, tokens[i]), key);
                const bool forward = postings_.erase(orPostingKey(OrPostingKind::CacheTokens, key), tokens[i]);
                assert(reverse && forward && "OR cache token postings disagree");
            }
        }

        /// @brief Unlink only the requested role among one owner's views.
        /// @details Removing a fact never removes the owner's source-leaf views,
        /// and retiring a source never removes independent known-fact views.
        /// @param owner Original expression and source scope.
        /// @param facts True for known-fact views, false for source-leaf views.
        /// @param scratch Page-tier temporary storage.
        /// @return Void.
        void eraseOwnerViews(int64_t owner, bool facts, ScratchArena& scratch) {
            ScratchScope work(scratch);
            DirtyState temporaryDirty = DirtyState::Clean;
            PagedVector<int32_t> ids(&scratch, &temporaryDirty);
            postings_.forEach(orPostingKey(OrPostingKind::OwnerViews, owner),
                [&](int64_t id) {
                    if (views[static_cast<int32_t>(id)].identity.isFact() == facts)
                        ids.push_back(static_cast<int32_t>(id));
                });
            for (int32_t i = 0; i < ids.size(); ++i) eraseView(ids[i], scratch);
        }

        /// @brief Add a view to its exact canonical bucket and scope directory.
        /// @details The caller has populated a canonical cache. A staged dirty
        /// cache keeps its previous buckets until the canonical drain rekeys it.
        /// @param id Live view slot.
        /// @param templateId Canonical complementary/fact template id.
        /// @param canonicalId Derived leaf spelling; zero selects the scalar cache spelling.
        /// @return Void.
        void attachCanonical(int32_t id, int32_t templateId, NameId canonicalId = 0) {
            assert(id >= 0 && id < views.size() && views[id].live);
            assert(templateId != 0);
            OrIndexedView view = views[id];
            if ((view.live & DerivedLeaf) != 0) {
                assert(canonicalId != 0);
                view.canonicalId = canonicalId;
            } else {
                const OrCanonicalCache* cache = caches_.find(view.cacheKey);
                assert(cache != nullptr);
                view.canonicalId = cache->canonicalId;
            }
            view.templateId = templateId;
            views.setAt(id, view);
            const bool fact = view.identity.isFact();
            postings_.insert(orPostingKey(fact ? OrPostingKind::FactKey
                                             : OrPostingKind::WatchKey,
                packInt32Pair(templateId, view.identity.scope)), id);
            postings_.insert(orPostingKey(fact ? OrPostingKind::FactScopes
                                             : OrPostingKind::WatchScopes,
                templateId), view.identity.scope);
        }

        /// @brief Remove a view from its old canonical lookup bucket.
        /// @details An emptied bucket also leaves the canonical scope directory.
        /// Other owners sharing either bucket remain indexed.
        /// @param id Live view slot.
        /// @param templateId The old canonical template id.
        /// @return Void.
        void detachCanonical(int32_t id, int32_t templateId) {
            assert(id >= 0 && id < views.size() && views[id].live);
            const OrIndexedView view = views[id];
            const bool fact = view.identity.isFact();
            const OrPostingKey bucket = orPostingKey(
                fact ? OrPostingKind::FactKey : OrPostingKind::WatchKey,
                packInt32Pair(templateId, view.identity.scope));
            const bool erased = postings_.erase(bucket, id);
            assert(erased && "OR view missing its canonical posting");
            if (!postings_.containsKey(bucket)) {
                const bool scopeErased = postings_.erase(orPostingKey(
                    fact ? OrPostingKind::FactScopes : OrPostingKind::WatchScopes,
                    templateId), view.identity.scope);
                assert(scopeErased && "OR bucket missing its scope directory");
            }
        }

    public:
        /// @brief Bind one unique owner/leaf/scope identity to a canonical cache.
        /// @details Existing identities keep their slot; a removed slot is
        /// reused. The new view is queued for exact-match checking.
        /// @param identity Original owner, effective scope, and leaf/fact role.
        /// @param cacheKey Packed original expression and effective scope.
        /// @param canonicalId Derived complementary leaf id, or zero for a scalar view.
        /// @param leafTemplateId Derived complementary template, paired with canonicalId.
        /// @return The live view slot for this identity.
        int32_t bindView(OrViewIdentity identity, int64_t cacheKey,
            NameId canonicalId = 0, int32_t leafTemplateId = 0) {
            assert(identity.owner != 0 && identity.scope != 0 && identity.leaf >= OrViewIdentity::FactLeaf);
            const int32_t* existing = viewIds.find(identity);
            if (existing != nullptr && *existing >= 0) {
                const int32_t id = *existing;
                const OrIndexedView view = views[id];
                assert(view.live && view.cacheKey == cacheKey);
                if (canonicalId != 0) {
                    assert((view.live & DerivedLeaf) != 0 && leafTemplateId != 0);
                    if (canonicalId != view.canonicalId || leafTemplateId != view.templateId) {
                        detachCanonical(id, view.templateId);
                        attachCanonical(id, leafTemplateId, canonicalId);
                        changedViews.mint(identity);
                    }
                } else {
                    assert((view.live & DerivedLeaf) == 0 && leafTemplateId == 0);
                }
                return id;
            }
            const bool reactivated = existing != nullptr;
            const OrCanonicalCache* cache = caches_.find(cacheKey);
            assert(cache != nullptr && cache->dirty >= 0 && cache->templateId != 0);
            assert((canonicalId == 0) == (leafTemplateId == 0));
            const int32_t templateId = canonicalId == 0 ? cache->templateId : leafTemplateId;
            const int32_t flags = canonicalId == 0 ? 1 : 1 | DerivedLeaf;
            int32_t id;
            if (freeView >= 0) {
                id = freeView;
                freeView = views[id].nextFree;
            } else {
                id = views.size();
                views.push_back(OrIndexedView{});
            }
            views.setAt(id, OrIndexedView{identity, cacheKey, -1, flags, 0, 0});
            viewIds.upsert(identity, id);
            if (reactivated) { assert(deadIdentities > 0); --deadIdentities; }
            ++liveViews;
            postings_.insert(orPostingKey(OrPostingKind::CacheUsers, cacheKey), id);
            postings_.insert(orPostingKey(OrPostingKind::OwnerViews, identity.owner), id);
            postings_.insert(orPostingKey(OrPostingKind::ScopeViews, identity.scope), id);
            attachCanonical(id, templateId, canonicalId);
            changedViews.mint(identity);
            return id;
        }

        /// @brief Unlink and recycle one active lookup view.
        /// @details Every canonical, ownership, scope and cache-user posting is
        /// removed immediately. The last user also retires the cache and its
        /// token/scope postings; canonical source state is untouched.
        /// @param id A live view slot whose owner is being removed or wiped.
        /// @param scratch Page-tier temporary storage for token removal.
        /// @return Void.
        void eraseView(int32_t id, ScratchArena& scratch) {
            assert(id >= 0 && id < views.size() && views[id].live);
            const OrIndexedView view = views[id];
            const OrCanonicalCache* cache = caches_.find(view.cacheKey);
            assert(cache != nullptr);
            detachCanonical(id, view.templateId);
            const bool cacheErased = postings_.erase(
                orPostingKey(OrPostingKind::CacheUsers, view.cacheKey), id);
            const bool ownerErased = postings_.erase(
                orPostingKey(OrPostingKind::OwnerViews, view.identity.owner), id);
            const bool scopeErased = postings_.erase(
                orPostingKey(OrPostingKind::ScopeViews, view.identity.scope), id);
            assert(cacheErased && ownerErased && scopeErased);
            viewIds.upsert(view.identity, -1);
            views.setAt(id, OrIndexedView{view.identity, view.cacheKey, freeView, 0});
            freeView = id;
            --liveViews;
            ++deadIdentities;
            if (postings_.containsKey(orPostingKey(OrPostingKind::CacheUsers, view.cacheKey))) return;
            eraseCacheTokens(view.cacheKey, scratch);
            const bool scopeRemoved = postings_.erase(orPostingKey(OrPostingKind::CacheScope,
                static_cast<NameId>(uint64_t(view.cacheKey))), view.cacheKey);
            assert(scopeRemoved);
            caches_.upsert(view.cacheKey, OrCanonicalCache{0, 0, 0});
            ++deadCaches;
        }

        /// @brief Amortize dead identity/key cleanup after a matching drain.
        /// @details Live slots remain stable while dead map keys are compacted.
        /// At zero views the view columns release all pages; cache cleanup and
        /// source-core membership remain the enclosing drain's responsibility.
        /// @return Void.
        void maintain() {
            postings_.maintain();
            if (deadCaches >= 64 && deadCaches * 2 >= caches_.count()) {
                caches_.eraseIf([&](int64_t key) {
                    const OrCanonicalCache* cache = caches_.find(key);
                    assert(cache != nullptr);
                    return cache->templateId == 0;
                });
                deadCaches = 0;
            }
            if (liveViews == 0) {
                views.clear();
                viewIds.resetToFresh();
                freeView = -1;
                deadIdentities = 0;
                return;
            }
            if (deadIdentities < 64 || deadIdentities < liveViews) return;
            viewIds.eraseIf([&](const OrViewIdentity& identity) {
                const int32_t* id = viewIds.find(identity);
                assert(id != nullptr);
                return *id < 0;
            });
            deadIdentities = 0;
        }

        /// @brief Release derived state before canonical release or teardown.
        /// @details No cached spelling, source view or pending derived work
        /// survives. The next write-side drain rebuilds from logical state.
        /// @return Void.
        void clear() {
            postings_.clear();
            viewIds.resetToFresh();
            views.clear();
            caches_.resetToFresh();
            initializedCores.resetToFresh();
            dirtyCaches.resetToFresh();
            projections.resetToFresh();
            changedViews.resetToFresh();
            evaluations.resetToFresh();
            freeView = -1;
            liveViews = 0;
            deadIdentities = 0;
            deadCaches = 0;
            built = false;
            classScopesBuilt = false;
        }

        /// @brief Account for every retained derived matching column.
        /// @details Includes cache and queue bytes as well as posting storage.
        /// @return Total pool-backed bytes retained by this matching index.
        int64_t liveBytes() const {
            return postings_.liveBytes() + viewIds.liveBytes() + views.liveBytes()
                + caches_.liveBytes() + initializedCores.liveBytes()
                + dirtyCaches.liveBytes() + projections.liveBytes()
                + changedViews.liveBytes() + evaluations.liveBytes();
        }
    };
}
