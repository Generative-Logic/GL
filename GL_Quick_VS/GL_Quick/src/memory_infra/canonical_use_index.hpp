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

#include "canonical_dependency_edges.hpp"

namespace gl {

/// @brief Index the names on which a scope-owned canonical record depends.
/// @details Stable logical record keys own reversible token postings. A record
/// replacement touches only its own postings; key synchronization preserves
/// existing edges and decodes only newly encountered records. This is derived
/// state: canonical release clears it and authoritative records rebuild it.
class CanonicalUseIndex {
    DirtyState dirty_ = DirtyState::Clean;
    TypedColdMap<int64_t, int32_t> records_;
    ColdHashSet<BytesKeyStore> names_;
    CanonicalDependencyEdges edges_;
    TypedColdMap<int64_t, int32_t> revisions_;
    TypedColdMap<int64_t, int64_t> observed_;
    TypedColdSet<int64_t> modified_;
    int32_t revision_ = 0;
    int32_t sourceCount_ = -1;
    uint32_t sourceEpoch_ = 0;
public:
    /// @brief Bind empty dependency postings to a resident arena.
    /// @details No logical record or token is created by construction.
    /// @param arena Owning arena, which outlives this index.
    /// @return An empty dependency index.
    explicit CanonicalUseIndex(LbArena* arena)
        : records_(arena, &dirty_), names_(arena, &dirty_), edges_(arena),
          revisions_(arena, &dirty_), observed_(arena, &dirty_),
          modified_(arena, &dirty_) {}

    /// @brief Register one record's dependency on one visible name.
    /// @details Repeated edges are idempotent; records may have several names.
    /// @param record Stable logical key of the owning record.
    /// @param scope Exact scope where this dependency is interpreted.
    /// @param member Substitution token, interned privately without minting NameMap.
    /// @return Nothing.
    void add(int64_t record, NameId scope, StrSpan member) {
        assert(scope != 0 && !member.empty());
        records_.upsert(record, 1);
        const int64_t token = packInt32Pair(names_.mint(member), scope);
        if (edges_.insert(token, record, revision_))
            revisions_.upsert(token, revision_);
    }

    /// @brief Remember a record with no current equality-name dependency.
    /// @details This prevents repeated decoding of unchanged token-free keys.
    /// @param record Stable logical record key.
    /// @return Nothing.
    void remember(int64_t record) { records_.upsert(record, 1); }

    /// @brief Test whether a logical record is currently live in this index.
    ///
    /// @details Records with no token dependencies remain live after remember;
    /// erased records are absent even when their retired key is still allocated.
    /// This read neither creates records nor changes dependency revisions.
    /// @param record Stable logical record key, including its effective scope.
    /// @return True exactly when the record is registered and not retired.
    /// @invariant The owning consumer synchronizes its authoritative source first.
    bool contains(int64_t record) const {
        const int32_t* live = records_.find(record);
        return live != nullptr && *live != 0;
    }

    /// @brief Remove one record's old token edges before replacing its text.
    /// @details Snapshot the small reverse run before unlinking either index.
    /// Other records and their insertion order are untouched.
    /// @param record Stable key to remove; absence is a defined no-op.
    /// @param scratch Caller-owned page-tier scratch arena.
    /// @return Nothing.
    void erase(int64_t record, LbArena& scratch) {
        DirtyState dirty = DirtyState::Clean;
        PagedVector<int64_t> tokens(&scratch, &dirty);
        edges_.forEachWhile<false>(record, [&](int64_t token, int32_t) {
            tokens.push_back(token); return true;
        });
        for (int32_t i = 0; i < tokens.size(); ++i) {
            const bool removed = edges_.erase(tokens[i], record);
            assert(removed);
            assert(revision_ < INT32_MAX);
            revisions_.upsert(tokens[i], ++revision_);
        }
        edges_.maintain();
        const int32_t* live = records_.find(record);
        if (live != nullptr && *live != 0) records_.upsert(record, 0);
    }

    /// @brief Mark one authoritative value as requiring dependency refresh.
    /// @details Repeated writes coalesce at its stable logical key. The next
    /// synchronization decodes this record before any dependency query.
    /// @param record Written logical record key.
    /// @return Nothing.
    void changed(int64_t record) { modified_.mint(record); }

    /// @brief Bring the index's logical keys into agreement with its source map.
    /// @details Unchanged count/epoch is constant time. Appends decode only new
    /// records; erasures unlink only absent keys and retain every live posting.
    /// Value-only updates are maintained explicitly by their writing doors.
    /// @tparam Map Source map with stable logical keys and an insertion epoch.
    /// @tparam Visit Callback that indexes one newly observed logical key.
    /// @param map Authoritative source map.
    /// @param scratch Caller-owned page-tier scratch for retired keys.
    /// @param visit Synchronous callback accepting the new key.
    /// @return Nothing.
    template<class Map, class Visit>
    void synchronize(const Map& map, LbArena& scratch, Visit&& visit) {
        const int32_t count = map.count();
        const uint32_t epoch = [&]() {
            if constexpr (std::is_same_v<Map, ColdHashSet<PodKeyStore<int64_t>>>)
                return map.insertEpoch();
            else return map.inner().insertEpoch();
        }();
        if (sourceCount_ == count && sourceEpoch_ == epoch && modified_.empty()) return;
        for (int32_t i = 1; i <= modified_.count(); ++i) {
            const int64_t key = modified_.keyAt(i);
            erase(key, scratch);
            if (map.lookup(key) != 0) { records_.upsert(key, 1); visit(key); }
        }
        modified_.resetToFresh();
        const bool appended = sourceCount_ >= 0 && count >= sourceCount_
            && epoch - sourceEpoch_ == static_cast<uint32_t>(count - sourceCount_);
        if (!appended) {
            DirtyState dirty = DirtyState::Clean;
            PagedVector<int64_t> retired(&scratch, &dirty);
            for (int32_t i = 1; i <= records_.count(); ++i) {
                const int64_t key = records_.keyAt(i);
                if (*records_.find(key) != 0 && map.lookup(key) == 0) retired.push_back(key);
            }
            for (int32_t i = 0; i < retired.size(); ++i) erase(retired[i], scratch);
        }
        for (int32_t i = appended ? sourceCount_ + 1 : 1; i <= count; ++i) {
            const int64_t key = map.keyAt(i);
            const int32_t* live = records_.find(key);
            if (live != nullptr && *live != 0) continue;
            records_.upsert(key, 1);
            visit(key);
        }
        sourceCount_ = count;
        sourceEpoch_ = epoch;
    }

    /// @brief Visit records depending on an exact member at an exact scope.
    /// @details Posting order is internal; proof-visible consumers sort selected
    /// logical keys before applying changes. The callback must not mutate this index.
    /// @tparam Sink Synchronous callback accepting a stable record key.
    /// @param scope Effective scope of the canonical record.
    /// @param member Name whose identity or equality metadata changed.
    /// @param sink Consumer of each distinct affected record.
    /// @return Nothing.
    template<class Sink>
    void forEach(NameId scope, StrSpan member, Sink&& sink) const {
        const int32_t name = names_.lookup(member);
        if (name == 0) return;
        edges_.forEachWhile<true>(packInt32Pair(name, scope), [&](int64_t record, int32_t) {
            sink(record); return true;
        });
    }

    /// @brief Select a token's dependent records only after a relevant change.
    /// @details Equality versions and posting revisions are compared exactly.
    /// An unchanged component with unchanged records performs no record visits.
    /// If only postings changed, visit the newly inserted prefix, including
    /// reinserted edges of replaced records. A changed equality version visits
    /// every live dependent. Revisions use logical edge keys, never node slots.
    /// @tparam Sink Synchronous callback receiving selected logical record keys.
    /// @param scope Effective comparison scope.
    /// @param member Original or canonical dependency name.
    /// @param version Current equality identity/evidence version for this member.
    /// @param sink Consumer that collects keys without mutating this index.
    /// @return Nothing.
    template<class Sink>
    void select(NameId scope, StrSpan member, int32_t version, Sink&& sink) {
        const int32_t name = names_.lookup(member);
        if (name == 0) return;
        const int64_t token = packInt32Pair(name, scope);
        const int32_t* revision = revisions_.find(token);
        if (revision == nullptr) return;
        const int64_t state = packInt32Pair(version, *revision);
        const int64_t* previous = observed_.find(token);
        if (previous != nullptr && *previous == state) return;
        const bool sameVersion = previous != nullptr
            && static_cast<int32_t>(uint64_t(*previous) >> 32) == version;
        const int32_t previousRevision = previous == nullptr ? 0 : static_cast<int32_t>(*previous);
        observed_.upsert(token, state);
        if (!sameVersion) {
            edges_.forEachWhile<true>(token, [&](int64_t record, int32_t) {
                sink(record); return true;
            });
            return;
        }
        edges_.forEachWhile<true>(token, [&](int64_t record, int32_t inserted) {
            if (inserted <= previousRevision) return false;
            sink(record);
            return true;
        });
    }

    /// @brief Retire dependencies belonging to closed scopes.
    ///
    /// @details Every record and token carries its scope in the low half.
    /// Surviving edges retain their insertion and observation revisions. Source
    /// synchronization rechecks authoritative keys after the scope-map compaction.
    /// Private name identifiers remain stable until canonical release.
    /// @tparam Closed Predicate over already minted scope identifiers.
    /// @param closed True for each closing scope and its descendants.
    /// @return Nothing.
    template<class Closed>
    void eraseScopes(const Closed& closed) {
        const auto dead = [&](int64_t key) { return closed(static_cast<NameId>(uint64_t(key))); };
        records_.eraseIf(dead);
        modified_.eraseIf(dead);
        edges_.eraseIf([&](int64_t token, int64_t record) { return dead(token) || dead(record); });
        revisions_.eraseIf(dead);
        observed_.eraseIf(dead);
        sourceCount_ = -1;
        sourceEpoch_ = 0;
        dirty_ = DirtyState::Restructured;
    }

    /// @brief Release all derived edges at an owning lifecycle boundary.
    /// @details Authoritative records remain in their separate owning stores.
    /// @return Nothing.
    void clear() {
        records_.resetToFresh(); names_.resetToFresh(); edges_.clear();
        sourceCount_ = -1; sourceEpoch_ = 0;
        revisions_.resetToFresh(); observed_.resetToFresh();
        modified_.resetToFresh(); revision_ = 0;
    }

    /// @brief Report derived mutations since the last raw image.
    /// @details Logical canonical images exclude these caches; raw images contain their pages.
    /// @return True when a raw snapshot must include changed cache storage.
    bool hasRawChanges() const { return dirty_ != DirtyState::Clean; }

    /// @brief Acknowledge a completed raw snapshot of this index.
    /// @details Call only once the owning arena image has been written successfully.
    /// @return Nothing.
    void markRawClean() { dirty_ = DirtyState::Clean; }

    /// @brief Account for all live and retained derived arena storage.
    /// @details Derived bytes are excluded from canonical deload facets.
    /// @return Allocated live-byte accounting for this index.
    int64_t liveBytes() const {
        return records_.count() * (sizeof(int64_t) + sizeof(int32_t))
            + names_.liveBytes() + edges_.liveBytes()
            + revisions_.count() * 12 + observed_.count() * 16 + modified_.count() * 8;
    }
};

/// @brief Cache marked-template comparisons at their effective target scopes.
/// @details Source records retain their original template and owner. Projection
/// keys are logical (template, target scope) pairs; equality dependencies select
/// only affected projections. The owning map always decides whether a result
/// still exists. These derived pages follow canonical release and raw eviction.
struct CanonicalTemplateIndex {
    DirtyState dirty = DirtyState::Clean;
    int64_t rebuildCount = 0;
    CanonicalUseIndex sources, uses;
    ColdHashSet<BytesKeyStore> labels;
    OrPostingIndex<int64_t> keysByShape, targetsByShape, scopesByShape;
    TypedColdMap<int64_t, int32_t> forms;
    OrPostingIndex<int64_t> pending;
    OrPostingIndex<int64_t> matches;

    /// @brief Bind an empty comparison cache to its resident arena.
    /// @details No templates or proof records are minted by construction.
    /// @param arena Owning arena, which outlives this cache.
    /// @return An empty derived comparison index.
    explicit CanonicalTemplateIndex(LbArena* arena)
        : sources(arena), uses(arena), labels(arena, &dirty), keysByShape(arena), targetsByShape(arena), scopesByShape(arena),
          forms(arena, &dirty), pending(arena), matches(arena) {}

    /// @brief Queue one original-expression projection at its effective scope.
    ///
    /// @details Repeated invalidations coalesce under the packed logical identity.
    /// Scope postings let a query drain only its own pending projections without
    /// scanning or compacting unrelated scopes. The queue is derived arena state.
    /// @param identity Packed original-expression identifier and target scope.
    /// @return Nothing.
    /// @invariant Both packed identifiers are already minted and nonzero.
    void enqueueProjection(int64_t identity) {
        const NameId scope = static_cast<NameId>(static_cast<uint64_t>(identity));
        assert(scope > 0 && (static_cast<uint64_t>(identity) >> 32) != 0);
        if (pending.insert(scope, identity)) dirty = DirtyState::Restructured;
    }

    /// @brief Invalidate cached projections depending on one changed member.
    ///
    /// @details The complete-class writer calls this only when the normal
    /// representative changes or the member first acquires a representative.
    /// Original and canonical token dependencies select exactly this scope's
    /// potentially affected projections; a missing dependency is a defined miss.
    /// @param scope Effective comparison scope whose representative changed.
    /// @param member Concrete substitution token with the changed representative.
    /// @return Nothing.
    /// @see enqueueProjection
    void invalidateMember(NameId scope, StrSpan member) {
        if (forms.count() == 0) return;
        uses.forEach(scope, member, [&](int64_t identity) { enqueueProjection(identity); });
    }

    /// @brief Retire closed owners and target projections while retaining survivors.
    ///
    /// @details Source-shape postings contain owner scopes; comparison forms,
    /// dependencies, matches and pending entries contain target scopes. Closing
    /// an owner also closes every descendant target where it could be projected.
    /// Shared shape labels remain stable and authoritative maps still filter hits.
    /// @tparam Closed Predicate over already minted scope identifiers.
    /// @param closed True for each closing scope and its descendants.
    /// @return Nothing.
    template<class Closed>
    void eraseScopes(const Closed& closed) {
        const auto dead = [&](int64_t key) { return closed(static_cast<NameId>(uint64_t(key))); };
        sources.eraseScopes(closed);
        uses.eraseScopes(closed);
        forms.eraseIf(dead);
        keysByShape.eraseIf([&](int64_t shape, int64_t source) { return dead(shape) || dead(source); });
        targetsByShape.eraseIf([&](int64_t shape, int64_t target) { return dead(shape) || closed(static_cast<NameId>(target)); });
        scopesByShape.eraseIf([&](int64_t, int64_t source) { return closed(static_cast<NameId>(source)); });
        pending.eraseIf([&](int64_t target, int64_t identity) { return closed(static_cast<NameId>(target)) || dead(identity); });
        matches.eraseIf([&](int64_t identity, int64_t) { return dead(identity); });
        dirty = DirtyState::Restructured;
    }

    /// @brief Release projections at a canonical lifecycle boundary.
    /// @details The authoritative admission and rejected maps remain untouched.
    /// @return Nothing.
    void clear() {
        sources.clear(); uses.clear(); labels.resetToFresh(); keysByShape.clear(); targetsByShape.clear(); scopesByShape.clear(); forms.resetToFresh();
        pending.clear(); matches.clear();
        rebuildCount = 0;
    }

    /// @brief Report changes to comparison projections or their dependencies.
    /// @details A cache-only warmup must not reuse a stale raw arena image.
    /// @return True when this cache changed since its last raw snapshot.
    bool hasRawChanges() const { return dirty != DirtyState::Clean || sources.hasRawChanges() || uses.hasRawChanges(); }

    /// @brief Acknowledge a completed raw snapshot of the comparison cache.
    /// @details Dependency and projection flags share the same snapshot boundary.
    /// @return Nothing.
    void markRawClean() { dirty = DirtyState::Clean; sources.markRawClean(); uses.markRawClean(); }

    /// @brief Include all derived projections in arena storage accounting.
    /// @details Raw eviction retains this state with its arena descriptors.
    /// @return Live bytes in this comparison cache.
    int64_t liveBytes() const {
        return sources.liveBytes() + uses.liveBytes() + labels.liveBytes() + keysByShape.liveBytes() + targetsByShape.liveBytes() + scopesByShape.liveBytes()
            + forms.count() * 12 + pending.liveBytes() + matches.liveBytes();
    }
};

/// @brief Own canonical-record dependencies independently of application state.
/// @details Each store retains its own key domain. Pending replay postings map
/// one canonical source/scope to its actual raw queued carriers; the reverse map
/// supports exact replacement without scanning every receipt at the scope.
struct CanonicalUseIndexes {
    DirtyState dirty = DirtyState::Clean;
    CanonicalUseIndex statements, receipts, goals, algebraDemands, integrationDemands;
    CanonicalUseIndex algebraSources, integrationSources, ordisSources, ordis2Sources, ordis2Demands;
    CanonicalUseIndex processedOr, compacts, watches;
    OrPostingIndex<int64_t> pendingCarriers;
    TypedColdMap<int64_t, int64_t> pendingIdentity;
    TypedColdMap<int64_t, int32_t> memberVersions;
    int32_t version = 0;
    CanonicalTemplateIndex algebraMatches, integrationMatches, integrationSetMatches;
    CanonicalTemplateIndex algebraParkMatches, integrationParkMatches, ordisParkMatches, ordis2ParkMatches, ordis2Matches;
    CanonicalTemplateIndex preparedMatches, preparedMarkerMatches, statementMatches, subproofParkMatches;
    bool pendingBuilt = false, orLedgersBuilt = false, watchesBuilt = false;
    bool deferWakes = false;
    TypedColdMap<int64_t, int32_t> wakeKinds;
    ColdHashSet<BytesKeyStore> algebraWakeParks, integrationWakeParks;

    /// @brief Bind all derived indexes to one logic block's arena.
    /// @details Empty indexes allocate no logical proof state.
    /// @param arena Owning main arena, which outlives these objects.
    /// @return An empty set of canonical dependency indexes.
    explicit CanonicalUseIndexes(LbArena* arena)
        : statements(arena), receipts(arena), goals(arena), algebraDemands(arena), integrationDemands(arena),
          algebraSources(arena), integrationSources(arena), ordisSources(arena), ordis2Sources(arena), ordis2Demands(arena),
          processedOr(arena), compacts(arena), watches(arena), pendingCarriers(arena), pendingIdentity(arena, &dirty),
          memberVersions(arena, &dirty), algebraMatches(arena), integrationMatches(arena), integrationSetMatches(arena),
          algebraParkMatches(arena), integrationParkMatches(arena), ordisParkMatches(arena), ordis2ParkMatches(arena), ordis2Matches(arena), preparedMatches(arena), preparedMarkerMatches(arena), statementMatches(arena), subproofParkMatches(arena),
          wakeKinds(arena, &dirty), algebraWakeParks(arena, &dirty), integrationWakeParks(arena, &dirty) {}

    /// @brief Queue affected comparisons at a complete-class mutation boundary.
    ///
    /// @details Each comparison store retains its own expression identifier
    /// space and source ownership. The mutation only invalidates derived views;
    /// the next query rebuilds its scope in the existing decoded-expression order.
    /// No admission, statement, receipt or equality-evidence record is changed.
    /// @param scope Exact effective scope whose normal representative changed.
    /// @param member Concrete token with a new or changed normal representative.
    /// @return Nothing.
    /// @see CanonicalTemplateIndex::invalidateMember
    void invalidateComparisonMember(NameId scope, StrSpan member) {
        CanonicalTemplateIndex* indexes[] = {
            &algebraMatches, &integrationMatches, &integrationSetMatches,
            &algebraParkMatches, &integrationParkMatches, &ordisParkMatches,
            &ordis2ParkMatches, &ordis2Matches, &preparedMatches,
            &preparedMarkerMatches, &statementMatches, &subproofParkMatches
        };
        for (CanonicalTemplateIndex* index : indexes) index->invalidateMember(scope, member);
    }

    /// @brief Remove closed-scope canonical state without cooling surviving scopes.
    ///
    /// @details Scope closure cannot change an ancestor or sibling equality view.
    /// Retained dependency revisions, comparison forms and pending identities
    /// remain valid. Full release and canonical reload still use clear().
    /// @tparam Closed Predicate over already minted scope identifiers.
    /// @param closed True for each closing scope and its descendants.
    /// @return Nothing.
    /// @invariant Deferred canonical maintenance is settled before scope closure.
    template<class Closed>
    void eraseScopes(const Closed& closed) {
        assert(!deferWakes && wakeKinds.empty() && algebraWakeParks.empty() && integrationWakeParks.empty());
        CanonicalUseIndex* dependencies[] = {
            &statements, &receipts, &goals, &algebraDemands, &integrationDemands,
            &algebraSources, &integrationSources, &ordisSources, &ordis2Sources,
            &ordis2Demands, &processedOr, &compacts, &watches
        };
        for (CanonicalUseIndex* index : dependencies) index->eraseScopes(closed);
        CanonicalTemplateIndex* comparisons[] = {
            &algebraMatches, &integrationMatches, &integrationSetMatches,
            &algebraParkMatches, &integrationParkMatches, &ordisParkMatches,
            &ordis2ParkMatches, &ordis2Matches, &preparedMatches,
            &preparedMarkerMatches, &statementMatches, &subproofParkMatches
        };
        for (CanonicalTemplateIndex* index : comparisons) index->eraseScopes(closed);
        const auto dead = [&](int64_t key) { return closed(static_cast<NameId>(uint64_t(key))); };
        pendingIdentity.eraseIf(dead);
        pendingCarriers.eraseIf([&](int64_t identity, int64_t carrier) { return dead(identity) || dead(carrier); });
        memberVersions.eraseIf(dead);
        dirty = DirtyState::Restructured;
    }

    /// @brief Clear derived state before canonical release or scope replacement.
    /// @details Raw eviction retains these arena pages and descriptors.
    /// @return Nothing.
    void clear() {
        assert(!deferWakes && wakeKinds.empty() && algebraWakeParks.empty() && integrationWakeParks.empty());
        statements.clear(); receipts.clear(); goals.clear(); algebraDemands.clear(); integrationDemands.clear();
        algebraSources.clear(); integrationSources.clear(); ordisSources.clear(); ordis2Sources.clear(); ordis2Demands.clear();
        processedOr.clear(); compacts.clear(); watches.clear(); pendingCarriers.clear(); pendingIdentity.resetToFresh();
        pendingBuilt = false; orLedgersBuilt = false; watchesBuilt = false;
        memberVersions.resetToFresh(); version = 0;
        algebraMatches.clear(); integrationMatches.clear(); integrationSetMatches.clear();
        algebraParkMatches.clear(); integrationParkMatches.clear(); ordisParkMatches.clear(); ordis2ParkMatches.clear(); ordis2Matches.clear(); preparedMatches.clear(); preparedMarkerMatches.clear(); statementMatches.clear(); subproofParkMatches.clear();
    }

    /// @brief Report cache mutations that require a new raw arena image.
    /// @details Canonical source state has its independent logical dirty flag.
    /// @return True when any derived canonical record has changed.
    bool hasRawChanges() const {
        return dirty != DirtyState::Clean
            || statements.hasRawChanges()
            || receipts.hasRawChanges()
            || goals.hasRawChanges()
            || algebraDemands.hasRawChanges()
            || integrationDemands.hasRawChanges()
            || algebraSources.hasRawChanges()
            || integrationSources.hasRawChanges()
            || ordisSources.hasRawChanges()
            || ordis2Sources.hasRawChanges()
            || ordis2Demands.hasRawChanges()
            || processedOr.hasRawChanges()
            || compacts.hasRawChanges()
            || watches.hasRawChanges()
            || algebraMatches.hasRawChanges()
            || integrationMatches.hasRawChanges()
            || integrationSetMatches.hasRawChanges()
            || algebraParkMatches.hasRawChanges()
            || integrationParkMatches.hasRawChanges()
            || ordisParkMatches.hasRawChanges()
            || ordis2ParkMatches.hasRawChanges()
            || ordis2Matches.hasRawChanges()
            || preparedMatches.hasRawChanges()
            || preparedMarkerMatches.hasRawChanges()
            || statementMatches.hasRawChanges()
            || subproofParkMatches.hasRawChanges()
            ;
    }

    /// @brief Acknowledge all canonical caches after a successful raw snapshot.
    /// @details This leaves logical proof data and canonical deload eligibility unchanged.
    /// @return Nothing.
    void markRawClean() {
        dirty = DirtyState::Clean;
        statements.markRawClean();
        receipts.markRawClean();
        goals.markRawClean();
        algebraDemands.markRawClean();
        integrationDemands.markRawClean();
        algebraSources.markRawClean();
        integrationSources.markRawClean();
        ordisSources.markRawClean();
        ordis2Sources.markRawClean();
        ordis2Demands.markRawClean();
        processedOr.markRawClean();
        compacts.markRawClean();
        watches.markRawClean();
        algebraMatches.markRawClean();
        integrationMatches.markRawClean();
        integrationSetMatches.markRawClean();
        algebraParkMatches.markRawClean();
        integrationParkMatches.markRawClean();
        ordisParkMatches.markRawClean();
        ordis2ParkMatches.markRawClean();
        ordis2Matches.markRawClean();
        preparedMatches.markRawClean();
        preparedMarkerMatches.markRawClean();
        statementMatches.markRawClean();
        subproofParkMatches.markRawClean();
    }

    /// @brief Include every derived canonical index in arena compaction sizing.
    /// @details Canonical class and receipt storage is counted by its owning maps.
    /// @return Total live bytes in these derived indexes.
    int64_t liveBytes() const {
        return statements.liveBytes() + receipts.liveBytes() + goals.liveBytes()
            + algebraDemands.liveBytes() + integrationDemands.liveBytes()
            + algebraSources.liveBytes() + integrationSources.liveBytes()
            + ordisSources.liveBytes() + ordis2Sources.liveBytes() + ordis2Demands.liveBytes()
            + processedOr.liveBytes() + compacts.liveBytes() + watches.liveBytes() + pendingCarriers.liveBytes()
            + pendingIdentity.count() * (sizeof(int64_t) + sizeof(int64_t))
            + memberVersions.count() * 12 + wakeKinds.count() * 12
            + algebraWakeParks.liveBytes() + integrationWakeParks.liveBytes()
            + algebraMatches.liveBytes() + integrationMatches.liveBytes() + integrationSetMatches.liveBytes()
            + algebraParkMatches.liveBytes() + integrationParkMatches.liveBytes()
            + ordisParkMatches.liveBytes() + ordis2ParkMatches.liveBytes() + ordis2Matches.liveBytes()
            + preparedMatches.liveBytes() + preparedMarkerMatches.liveBytes() + statementMatches.liveBytes()
            + subproofParkMatches.liveBytes();
    }
};
}
