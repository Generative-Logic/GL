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
    /// CacheTokens/TokenCaches hold the tokens of an entry's canonical form;
    /// ViewTokens/TokenViews hold the tokens a view's own spelling adds to it.
    enum class OrPostingKind : int64_t {
        CacheUsers, OwnerViews, ScopeViews, TokenCaches, ClassScopes,
        WatchKey, FactKey, WatchScopes, FactScopes, SourceCore, OwnerCores,
        FactCore, CacheScope, CacheTokens, ViewTokens, TokenViews
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

    /// @brief Shared canonical entry for one canonical-expression/scope pair.
    /// @details The map key packs this canonical NameMap id with the effective
    /// scope, so every spelling with this form at that scope shares the entry.
    /// Template ids belong to the rejected-map template interner. Dirty entries
    /// require the shared chooser before matching. The fact-carrier count is the
    /// number of known-fact views on the entry; the entry's single fact lookup
    /// posting exists exactly while that count is positive. No rewrite locality
    /// is stored: a firing computes it for the spelling it considers.
    struct OrCanonicalCache {
        NameId canonicalId;
        int32_t templateId;
        int32_t dirty;
        int32_t factCarriers;
    };

    /// @brief Recyclable lookup view referencing its shared canonical entry.
    /// @details Every link and identity is integer-valued and relocatable. The
    /// original id is the spelling the view was bound with: a known fact, a
    /// whole source OR, or a scalar complementary leaf. A known-fact view is a
    /// carrier: it mirrors its entry's canonical and template ids and owns no
    /// lookup posting of its own.
    struct OrIndexedView {
        OrViewIdentity identity;
        int64_t cacheKey;
        int32_t nextFree;
        int32_t live;
        NameId canonicalId;
        int32_t templateId;
        NameId originalId;
    };

    /// @brief Derived indexes and deduplicated work for on-demand OR matching.
    ///
    /// @details
    /// Source state lives in OrImplications. This index groups canonical views
    /// by exact template/validity, source owner, canonical entry, token and
    /// scope. One entry exists per canonical expression and effective scope:
    /// known-fact spellings with that form are its carriers and share one fact
    /// lookup posting and one join, while source leaves keep one view each.
    /// The tokens of the canonical form are posted once per entry; a view posts
    /// only the tokens of its own spelling that the canonical form lacks. One
    /// domain-separated posting engine supplies all lookup directions without
    /// allocating a separate hash-table object for every direction on every LB.
    /// The prover supplies canonical results through its existing watch chooser.
    /// Canonical release clears this object; raw eviction preserves its pages.
    /// @invariant Lookup indexes never supply proof validity or provenance.
    /// @invariant A live entry's key carries its own canonical id; a changed
    ///            form moves the entry's views to another key, never in place.
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
        TypedColdSet<int64_t> changedFacts;
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
              changedViews(arena, &dirty), changedFacts(arena, &dirty),
              evaluations(arena, &dirty) {}

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

        /// @brief Create or revive one canonical entry and post its form's tokens.
        /// @details The key's canonical half must equal the value's canonical id
        /// and no live entry may hold the key: an entry's form never changes in
        /// place, a changed form moves the entry's views to another key
        /// (moveView). A dead record of the same key is revived. The visitor
        /// streams the substitution tokens of the canonical form in mint order.
        /// @tparam TokenVisitor Synchronous heap-free token visitor.
        /// @param key Canonical expression and effective scope.
        /// @param value Settled canonical result without fact carriers.
        /// @param visitTokens Emits the canonical form's substitution tokens.
        /// @return Void.
        template<class TokenVisitor>
        void publishCache(int64_t key, OrCanonicalCache value, TokenVisitor&& visitTokens) {
            assert(key != 0 && value.canonicalId != 0 && value.templateId != 0
                && value.dirty == 0 && value.factCarriers == 0);
            assert(static_cast<NameId>(uint64_t(key) >> 32) == value.canonicalId
                && "an OR entry is keyed by its own canonical form");
            const OrCanonicalCache* previous = caches_.find(key);
            assert(previous == nullptr || previous->templateId == 0);
            if (previous != nullptr) { assert(deadCaches > 0); --deadCaches; }
            caches_.upsert(key, value);
            visitTokens([&](NameId token) {
                assert(token != 0);
                postings_.insert(orPostingKey(OrPostingKind::TokenCaches, token), key);
                postings_.insert(orPostingKey(OrPostingKind::CacheTokens, key), token);
            });
            postings_.insert(orPostingKey(OrPostingKind::CacheScope,
                static_cast<NameId>(uint64_t(key))), key);
        }

        /// @brief Whether a live canonical entry holds this key.
        /// @details A retired entry keeps a dead record until maintenance, which
        /// is not live.
        /// @param key Canonical expression and effective scope.
        /// @return True when the entry exists with a template.
        bool cacheLive(int64_t key) const {
            const OrCanonicalCache* found = caches_.find(key);
            return found != nullptr && found->templateId != 0;
        }

        /// @brief Clear a dirty entry whose canonical form did not change.
        /// @details A class change can alter a spelling's rewrite locality
        /// without moving the form, so every user is queued for matching again:
        /// source and scalar views through changedViews, the fact carriers once
        /// through changedFacts.
        /// @param key Live dirty entry.
        /// @return Void.
        void settleCache(int64_t key) {
            const OrCanonicalCache* found = caches_.find(key);
            assert(found != nullptr && found->templateId != 0 && found->dirty == 1);
            OrCanonicalCache settled = *found;
            settled.dirty = 0;
            caches_.upsert(key, settled);
            postings_.forEach(orPostingKey(OrPostingKind::CacheUsers, key), [&](int64_t rawId) {
                const OrIndexedView view = views[static_cast<int32_t>(rawId)];
                assert(view.live);
                if (view.identity.isFact()) changedFacts.mint(key);
                else changedViews.mint(view.identity);
            });
        }

        /// @brief Move one live view to another live entry at its scope.
        /// @details Its entry's canonical form moved under a class change. The
        /// view leaves the old user list and joins the target's. A fact carrier
        /// moves its count and the entry-level fact posting and queues the
        /// target's fact role; a scalar view rekeys its own lookup bucket. A
        /// derived source leaf keeps its leaf bucket until its source rebinds
        /// it. Rewritten-token postings are the caller's (eraseViewTokens,
        /// addViewTokens).
        /// @param id Live view slot.
        /// @param target Live entry key at the view's effective scope.
        /// @return Void.
        void moveView(int32_t id, int64_t target) {
            assert(id >= 0 && id < views.size() && views[id].live);
            OrIndexedView view = views[id];
            assert(view.cacheKey != target && cacheLive(target));
            assert(static_cast<NameId>(uint64_t(target)) == view.identity.scope);
            const bool derived = (view.live & DerivedLeaf) != 0;
            if (!derived) detachCanonical(id, view.templateId);
            const bool erased = postings_.erase(orPostingKey(OrPostingKind::CacheUsers, view.cacheKey), id);
            assert(erased && "OR view missing from its entry's users");
            postings_.insert(orPostingKey(OrPostingKind::CacheUsers, target), id);
            view.cacheKey = target;
            views.setAt(id, view);
            if (derived) return;
            attachCanonical(id, caches_.find(target)->templateId);
            if (view.identity.isFact()) changedFacts.mint(target);
            else changedViews.mint(view.identity);
        }

        /// @brief Retire an entry that no view uses any more.
        /// @details Removes the entry's token and scope postings and leaves a
        /// dead record for amortized compaction. Its fact posting has already
        /// left with its last carrier.
        /// @param key Live entry without users.
        /// @param scratch Page-tier temporary storage.
        /// @return Void.
        void retireCache(int64_t key, ScratchArena& scratch) {
            const OrCanonicalCache* found = caches_.find(key);
            assert(found != nullptr && found->templateId != 0 && found->factCarriers == 0);
            assert(!postings_.containsKey(orPostingKey(OrPostingKind::CacheUsers, key)));
            eraseCacheTokens(key, scratch);
            const bool scopeRemoved = postings_.erase(orPostingKey(OrPostingKind::CacheScope,
                static_cast<NameId>(uint64_t(key))), key);
            assert(scopeRemoved);
            caches_.upsert(key, OrCanonicalCache{0, 0, 0, 0});
            ++deadCaches;
        }

        /// @brief Post the rewritten tokens of one view's own spelling.
        /// @details These are the substitution tokens of the view's original
        /// expression that its entry's canonical form lacks. A class change on
        /// one of them dirties the entry or projects this view alone.
        /// @tparam TokenVisitor Synchronous heap-free token visitor.
        /// @param id Live view slot.
        /// @param visitTokens Emits the view's rewritten tokens.
        /// @return Void.
        template<class TokenVisitor>
        void addViewTokens(int32_t id, TokenVisitor&& visitTokens) {
            assert(id >= 0 && id < views.size() && views[id].live);
            visitTokens([&](NameId token) {
                assert(token != 0);
                postings_.insert(orPostingKey(OrPostingKind::ViewTokens, id), token);
                postings_.insert(orPostingKey(OrPostingKind::TokenViews, token), id);
            });
        }

        /// @brief Remove all rewritten-token postings of one view slot.
        /// @details Both directions leave together; a slot without rewritten
        /// tokens (a canonical spelling) is the defined empty case.
        /// @param id View slot whose postings are removed.
        /// @param scratch Page-tier temporary storage.
        /// @return Void.
        void eraseViewTokens(int32_t id, ScratchArena& scratch) {
            ScratchScope work(scratch);
            DirtyState temporaryDirty = DirtyState::Clean;
            PagedVector<NameId> tokens(&scratch, &temporaryDirty);
            postings_.forEach(orPostingKey(OrPostingKind::ViewTokens, id),
                [&](int64_t token) { tokens.push_back(static_cast<NameId>(token)); });
            for (int32_t i = 0; i < tokens.size(); ++i) {
                const bool reverse = postings_.erase(orPostingKey(OrPostingKind::TokenViews, tokens[i]), id);
                const bool forward = postings_.erase(orPostingKey(OrPostingKind::ViewTokens, id), tokens[i]);
                assert(reverse && forward && "OR view token postings disagree");
            }
        }

        /// @brief Dirty every live entry at one scope that a class member reaches.
        /// @details The member reaches an entry as a token of its canonical form
        /// or as a rewritten token of one of its views. Repeated hits
        /// deduplicate.
        /// @param member Class member whose representative or rewrite locality changed.
        /// @param scope Exact scope of the refreshed class view.
        /// @return Void.
        void markTokenUsersDirty(NameId member, NameId scope) {
            assert(member != 0 && scope != 0);
            postings_.forEach(orPostingKey(OrPostingKind::TokenCaches, member), [&](int64_t key) {
                if (static_cast<NameId>(uint64_t(key)) == scope) markCacheDirty(key);
            });
            postings_.forEach(orPostingKey(OrPostingKind::TokenViews, member), [&](int64_t rawId) {
                const int64_t key = views[static_cast<int32_t>(rawId)].cacheKey;
                if (static_cast<NameId>(uint64_t(key)) == scope) markCacheDirty(key);
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

        /// @brief Add a view to its canonical lookup role.
        /// @details The caller has populated a canonical entry. A known-fact
        /// view joins its entry's fact role: the entry is posted under its
        /// template and scope when its first carrier arrives. A watch view is
        /// posted itself under its own template. A staged dirty entry keeps its
        /// postings until the canonical drain settles or moves it.
        /// @param id Live view slot.
        /// @param templateId Canonical complementary/fact template id.
        /// @param canonicalId Derived leaf spelling; zero selects the scalar entry spelling.
        /// @return Void.
        void attachCanonical(int32_t id, int32_t templateId, NameId canonicalId = 0) {
            assert(id >= 0 && id < views.size() && views[id].live);
            assert(templateId != 0);
            OrIndexedView view = views[id];
            const OrCanonicalCache* cache = caches_.find(view.cacheKey);
            assert(cache != nullptr && cache->templateId != 0);
            if ((view.live & DerivedLeaf) != 0) {
                assert(canonicalId != 0);
                view.canonicalId = canonicalId;
            } else {
                view.canonicalId = cache->canonicalId;
            }
            view.templateId = templateId;
            views.setAt(id, view);
            if (view.identity.isFact()) {
                assert((view.live & DerivedLeaf) == 0 && templateId == cache->templateId);
                OrCanonicalCache counted = *cache;
                if (++counted.factCarriers == 1) {
                    postings_.insert(orPostingKey(OrPostingKind::FactKey,
                        packInt32Pair(templateId, view.identity.scope)), view.cacheKey);
                    postings_.insert(orPostingKey(OrPostingKind::FactScopes, templateId),
                        view.identity.scope);
                }
                caches_.upsert(view.cacheKey, counted);
                return;
            }
            postings_.insert(orPostingKey(OrPostingKind::WatchKey,
                packInt32Pair(templateId, view.identity.scope)), id);
            postings_.insert(orPostingKey(OrPostingKind::WatchScopes, templateId),
                view.identity.scope);
        }

        /// @brief Remove a view from its canonical lookup role.
        /// @details A known-fact view leaves its entry's carrier count; the last
        /// carrier takes the entry's fact posting and its scope directory edge
        /// with it. A watch view leaves its own bucket, and an emptied bucket
        /// leaves the scope directory. Other owners sharing a bucket remain.
        /// @param id Live view slot.
        /// @param templateId The view's current canonical template id.
        /// @return Void.
        void detachCanonical(int32_t id, int32_t templateId) {
            assert(id >= 0 && id < views.size() && views[id].live);
            const OrIndexedView view = views[id];
            if (view.identity.isFact()) {
                const OrCanonicalCache* cache = caches_.find(view.cacheKey);
                assert(cache != nullptr && cache->factCarriers > 0 && cache->templateId == templateId);
                OrCanonicalCache counted = *cache;
                if (--counted.factCarriers == 0) {
                    const OrPostingKey bucket = orPostingKey(OrPostingKind::FactKey,
                        packInt32Pair(templateId, view.identity.scope));
                    const bool erased = postings_.erase(bucket, view.cacheKey);
                    assert(erased && "OR entry missing its fact posting");
                    assert(!postings_.containsKey(bucket) && "one OR entry per template and scope");
                    const bool scopeErased = postings_.erase(orPostingKey(
                        OrPostingKind::FactScopes, templateId), view.identity.scope);
                    assert(scopeErased && "OR fact bucket missing its scope directory");
                }
                caches_.upsert(view.cacheKey, counted);
                return;
            }
            const OrPostingKey bucket = orPostingKey(OrPostingKind::WatchKey,
                packInt32Pair(templateId, view.identity.scope));
            const bool erased = postings_.erase(bucket, id);
            assert(erased && "OR view missing its canonical posting");
            if (!postings_.containsKey(bucket)) {
                const bool scopeErased = postings_.erase(orPostingKey(
                    OrPostingKind::WatchScopes, templateId), view.identity.scope);
                assert(scopeErased && "OR bucket missing its scope directory");
            }
        }

    public:
        /// @brief Bind one unique owner/leaf/scope identity to a canonical entry.
        /// @details Existing identities keep their slot; a removed slot is
        /// reused. A new watch view is queued for exact-match checking, a new
        /// fact carrier queues its entry's fact role.
        /// @param identity Original owner, effective scope, and leaf/fact role.
        /// @param cacheKey Packed canonical expression and effective scope.
        /// @param originalId The spelling bound: known fact, whole source OR or scalar leaf.
        /// @param canonicalId Derived complementary leaf id, or zero for a scalar view.
        /// @param leafTemplateId Derived complementary template, paired with canonicalId.
        /// @return The live view slot for this identity.
        int32_t bindView(OrViewIdentity identity, int64_t cacheKey, NameId originalId,
            NameId canonicalId = 0, int32_t leafTemplateId = 0) {
            assert(identity.owner != 0 && identity.scope != 0 && identity.leaf >= OrViewIdentity::FactLeaf);
            assert(originalId != 0 && (!identity.isFact() || canonicalId == 0));
            assert(static_cast<NameId>(uint64_t(cacheKey)) == identity.scope);
            const int32_t* existing = viewIds.find(identity);
            if (existing != nullptr && *existing >= 0) {
                const int32_t id = *existing;
                const OrIndexedView view = views[id];
                assert(view.live && view.cacheKey == cacheKey && view.originalId == originalId);
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
            views.setAt(id, OrIndexedView{identity, cacheKey, -1, flags, 0, 0, originalId});
            viewIds.upsert(identity, id);
            if (reactivated) { assert(deadIdentities > 0); --deadIdentities; }
            ++liveViews;
            postings_.insert(orPostingKey(OrPostingKind::CacheUsers, cacheKey), id);
            postings_.insert(orPostingKey(OrPostingKind::OwnerViews, identity.owner), id);
            postings_.insert(orPostingKey(OrPostingKind::ScopeViews, identity.scope), id);
            attachCanonical(id, templateId, canonicalId);
            if (identity.isFact()) changedFacts.mint(cacheKey);
            else changedViews.mint(identity);
            return id;
        }

        /// @brief Unlink and recycle one active lookup view.
        /// @details Every canonical, rewritten-token, ownership, scope and
        /// entry-user posting is removed immediately. The last user also
        /// retires the entry and its token/scope postings; canonical source
        /// state is untouched.
        /// @param id A live view slot whose owner is being removed or wiped.
        /// @param scratch Page-tier temporary storage for token removal.
        /// @return Void.
        void eraseView(int32_t id, ScratchArena& scratch) {
            assert(id >= 0 && id < views.size() && views[id].live);
            const OrIndexedView view = views[id];
            assert(cacheLive(view.cacheKey));
            detachCanonical(id, view.templateId);
            eraseViewTokens(id, scratch);
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
            retireCache(view.cacheKey, scratch);
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
            changedFacts.resetToFresh();
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
            return peak_snapshot::measure("postings_", postings_, [&] { return postings_.liveBytes(); }) + peak_snapshot::measure("viewIds", viewIds, [&] { return viewIds.liveBytes(); }) + peak_snapshot::measure("views", views, [&] { return views.liveBytes(); })
                + peak_snapshot::measure("caches_", caches_, [&] { return caches_.liveBytes(); }) + peak_snapshot::measure("initializedCores", initializedCores, [&] { return initializedCores.liveBytes(); })
                + peak_snapshot::measure("dirtyCaches", dirtyCaches, [&] { return dirtyCaches.liveBytes(); }) + peak_snapshot::measure("projections", projections, [&] { return projections.liveBytes(); })
                + peak_snapshot::measure("changedViews", changedViews, [&] { return changedViews.liveBytes(); }) + peak_snapshot::measure("changedFacts", changedFacts, [&] { return changedFacts.liveBytes(); })
                + peak_snapshot::measure("evaluations", evaluations, [&] { return evaluations.liveBytes(); });
        }
    };
}
