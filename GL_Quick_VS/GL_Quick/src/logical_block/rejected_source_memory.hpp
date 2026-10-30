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
#include "canonicalizer.hpp"
#include "canonical_template_matcher.hpp"
namespace gl {
namespace tests { struct RejectedSourceMemoryFixture; }
/// @brief Register parked sources and maintain their canonical completion records.
/// @details Operations preserve the owning block's storage, ordering and source identity.
class RejectedSourceMemory {
friend struct tests::RejectedSourceMemoryFixture;
public:
/// @brief Bind the readers needed by rejected-source storage operations.
/// @details The supplied block retains parked records, completion receipts and indices.
/// @param parameters Existing source-registration settings.
/// @param canonicalizer Canonical expression-view operations.
/// @param canonicalTemplates Canonical template and source-index operations.
/// @return Constructed rejected-source memory interface.
RejectedSourceMemory(const ProverParameters& parameters, Canonicalizer& canonicalizer, CanonicalTemplateMatcher& canonicalTemplates);

public:
/// @brief Park an existence under one concrete marked demand key.
        /// @details The rejected expansion is not retained. Both indexes store
        /// source context and optional equality transformations only.
        /// @param markedKey Concrete marked demand key.
        /// @param source Source existence with concrete enclosing arguments.
        /// @param iteration Original firing generation.
        /// @param flags Root construction and OR processing flags.
        /// @param levels Ascending unique source levels.
        /// @param levelCount Number of levels.
        /// @param hashMemory Owning demand index.
        /// @param mb Owning memory block.
        /// @param validityName Source scope.
        /// @param rewrites Deferred before/after/equality identifiers.
        /// @param rewriteCount Number of rewrite identifiers.
        /// @param integrationDepth Original expansion assignment retained for replay naming.
        /// @return Void.
inline void updateRejectedMap(StrSpan markedKey, StrSpan source,
            int iteration, int32_t flags, const int* levels, int32_t levelCount,
            HashMemory& hashMemory, Memory& mb, StrSpan validityName,
            const int32_t* rewrites = nullptr, int32_t rewriteCount = 0, int32_t integrationDepth = 0) {
            assert(!containsSpan(markedKey, StrSpan("__gl_temporary_witness_", 23)));
            assert(!containsSpan(source, StrSpan("__gl_temporary_witness_", 23)));
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
            ScratchArena& arena = genScratchArenas().forSlot(slot);
            const int32_t sourceId = mb.valueInterner.encode(source);
            const int64_t key = mintTemplateKey(mb.templateInterner, mb.nameMap,
                markedKey, validityName);
            const int32_t beforeId = hashMemory.rejectedMap.lookup(key);
            const int32_t beforeCount = beforeId == 0 ? 0 : hashMemory.rejectedMap.runLen(beforeId);
            insertRejectedIdsBlob(hashMemory.rejectedMap, key, sourceId,
                iteration, flags, rewrites, rewriteCount, levels, levelCount,
                mb.valueInterner, arena, integrationDepth);
            auto& lookup = mb.lbMemory.existenceLookups.algebraSources;
            if (&hashMemory == &mb.overallHashMemory && hashMemory.rejectedMap.runLen(hashMemory.rejectedMap.lookup(key)) != beforeCount) {
                const NameId scope = Codec<StatementKey>::decode(key).validity;
                auto& uses = mb.lbMemory.canonicalUses.algebraSources;
                canonicalTemplates.indexCanonicalText(uses,key,scope,mb.templateInterner.decodeView(Codec<StatementKey>::decode(key).orig));
                canonicalTemplates.indexCanonicalText(uses,key,scope,mb.valueInterner.decodeView(sourceId));
                for (int32_t i=0;i<rewriteCount;++i) canonicalTemplates.indexCanonicalText(uses,key,scope,mb.valueInterner.decodeView(rewrites[i]));
                uses.changed(key);
                if (lookup.built) lookup.add(packInt32Pair(canonicalExistenceId(mb,
                    mb.valueInterner.decodeView(sourceId),scope),scope),key);
            }

        }

public:
/// @brief Park an existence under one concrete marked demand key.
        /// @details The rejected expansion is not retained. Both indexes store
        /// source context and optional equality transformations only.
        /// @param markedKey Concrete marked demand key.
        /// @param source Source existence with concrete enclosing arguments.
        /// @param iteration Original firing generation.
        /// @param flags Root construction and OR processing flags.
        /// @param levels Ascending unique source levels.
        /// @param levelCount Number of levels.
        /// @param hashMemory Owning demand index.
        /// @param mb Owning memory block.
        /// @param validityName Source scope.
        /// @param rewrites Deferred before/after/equality identifiers.
        /// @param rewriteCount Number of rewrite identifiers.
        /// @param integrationDepth Original expansion assignment retained for replay naming.
        /// @return Void.
inline void updateRejectedMapIntegration(StrSpan markedKey, StrSpan source,
            int iteration, int32_t flags, const int* levels, int32_t levelCount,
            HashMemory& hashMemory, Memory& mb, StrSpan validityName,
            const int32_t* rewrites = nullptr, int32_t rewriteCount = 0, int32_t integrationDepth = 0) {
            assert(integrationDepth <= parameters.maxIntegrationDepth);
            assert(!containsSpan(markedKey, StrSpan("__gl_temporary_witness_", 23)));
            assert(!containsSpan(source, StrSpan("__gl_temporary_witness_", 23)));
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
            ScratchArena& arena = genScratchArenas().forSlot(slot);
            const int32_t sourceId = mb.valueInterner.encode(source);
            const int64_t key = mintTemplateKey(mb.templateInterner, mb.nameMap,
                markedKey, validityName);
            const int32_t beforeId = hashMemory.rejectedMapIntegration.lookup(key);
            const int32_t beforeCount = beforeId == 0 ? 0 : hashMemory.rejectedMapIntegration.runLen(beforeId);
            insertRejectedIntegrationIdsBlob(hashMemory.rejectedMapIntegration, key, sourceId,
                iteration, flags, rewrites, rewriteCount, levels, levelCount,
                mb.valueInterner, arena, integrationDepth);
            auto& lookup = mb.lbMemory.existenceLookups.integrationSources;
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t count = getArgsSpans(markedKey, args, ExecutionParameters::MAX_ARITY);
            for (int32_t i = 0; i < count; ++i)
                if (!equalSpans(args[i], StrSpan("marker", 6)))
                    hashMemory.varsInRejectedMapIntegrationKeys.mint(mb.templateInterner.encode(args[i]));
            if (&hashMemory == &mb.overallHashMemory && hashMemory.rejectedMapIntegration.runLen(hashMemory.rejectedMapIntegration.lookup(key)) != beforeCount) {
                const NameId scope = Codec<StatementKey>::decode(key).validity;
                auto& uses = mb.lbMemory.canonicalUses.integrationSources;
                canonicalTemplates.indexCanonicalText(uses,key,scope,mb.templateInterner.decodeView(Codec<StatementKey>::decode(key).orig));
                canonicalTemplates.indexCanonicalText(uses,key,scope,mb.valueInterner.decodeView(sourceId));
                for (int32_t i=0;i<rewriteCount;++i) canonicalTemplates.indexCanonicalText(uses,key,scope,mb.valueInterner.decodeView(rewrites[i]));
                uses.changed(key);
                if (lookup.built) lookup.add(packInt32Pair(canonicalExistenceId(mb,
                    mb.valueInterner.decodeView(sourceId),scope),scope),key);
            }

        }

private:
/// @brief Update the canonical group of one actual queued receipt.
        /// @details Old reverse membership is removed before current pending bits
        /// are attached to the target-scope identity. The raw carrier is unchanged.
        /// @param mb Owning block with initialized target-scope classes.
        /// @param key Actual source/scope receipt key.
        /// @param state Current completed, pending and carrier-mode bits.
        /// @return Nothing.
inline void indexPendingExistence(Memory& mb, int64_t key, int32_t state) {
            auto& uses = mb.lbMemory.canonicalUses;
            const int64_t* previous = uses.pendingIdentity.find(key);
            const int64_t old = previous == nullptr ? 0 : *previous;
            int64_t current = 0;
            if ((state & 12) != 0) {
                const StatementKey source = Codec<StatementKey>::decode(key);
                current = packOriginKey(canonicalExistenceId(mb,
                    mb.nameMap.decodeView(source.orig), source.validity), source.validity);
            }
            if (old == current) return;
            if (old != 0) uses.pendingCarriers.erase(old, key);
            if (current != 0) uses.pendingCarriers.insert(current, key);
            uses.pendingIdentity.upsert(key, current);
        }

public:
/// @brief Synchronize receipt dependencies and initialize pending groups.
        /// @details Value-only production writes use writeExistenceState, so an
        /// unchanged authoritative key set needs no scope-wide receipt scan.
        /// @param mb Owning block.
        /// @param arena Caller-owned page-tier scratch.
        /// @return Nothing.
inline void ensureCanonicalReceiptUses(Memory& mb, ScratchArena& arena) {
            auto& uses = mb.lbMemory.canonicalUses;
            uses.receipts.synchronize(mb.existenceReplays, arena, [&](int64_t key) {
                const StatementKey source = Codec<StatementKey>::decode(key);
                canonicalTemplates.indexCanonicalText(uses.receipts, key, source.validity,
                    mb.nameMap.decodeView(source.orig));
                if (uses.pendingBuilt) indexPendingExistence(mb, key, *mb.existenceReplays.find(key));
            });
            if (!uses.pendingBuilt) {
                for (int32_t i = 1; i <= mb.existenceReplays.count(); ++i) {
                    const int64_t key = mb.existenceReplays.keyAt(i);
                    indexPendingExistence(mb, key, *mb.existenceReplays.find(key));
                }
                uses.pendingBuilt = true;
            }
        }

public:
/// @brief Write construction state and maintain its derived dependencies.
        /// @details The authoritative raw receipt retains pending delivery bits;
        /// completed identity selection is the caller's target-scope contract.
        /// @param mb Owning block.
        /// @param key Authoritative source/scope identity.
        /// @param state Completed, pending and carrier-mode bits to retain.
        /// @return Nothing.
inline void writeExistenceState(Memory& mb, int64_t key, int32_t state) {
            mb.existenceReplays.upsert(key, state);
            const StatementKey source = Codec<StatementKey>::decode(key);
            canonicalTemplates.indexCanonicalText(mb.lbMemory.canonicalUses.receipts, key,
                source.validity, mb.nameMap.decodeView(source.orig));
            if (mb.lbMemory.canonicalUses.pendingBuilt) indexPendingExistence(mb, key, state);
        }

/// @brief Retire the pending receipt of a source-replay delivery that an
        ///        absorb drops unprocessed.
        ///
        /// @details
        /// A pending receipt (bits 2..3, relay bit 16) promises that an internal
        /// mail channel still holds the replay delivery carrying its integration
        /// assignment; readers of pending carriers ask that delivery for its
        /// depth. Phase 3 drops the internal mail of entry-resolved OR subtrees
        /// without absorbing it (I-232), so a replay queued there would leave a
        /// receipt that no delivery backs. The dropped delivery therefore
        /// consumes the receipt exactly as an absorbed one does: only the
        /// completed construction bits remain, and a later demand may queue the
        /// replay afresh — the state a facts clone also keeps.
        ///
        /// A delivery without an assignment (a regular arrival sharing the
        /// source key) does not consume the receipt, as in the absorb; a
        /// statement without a pending receipt is not a replay delivery and
        /// changes nothing.
        ///
        /// @param mb Owning block.
        /// @param key Packed (source, scope) NameMap identity of the dropped delivery.
        /// @param carriedDepth Integration depth the dropped delivery carries, -1 without one.
        /// @return Nothing.
        /// @invariant Every pending receipt is backed by an undelivered replay carrying its assignment.
        /// @see writeExistenceState, RejectedSourceReplay::queueSourceReplay, Memory::phase3IgnoresScope.
inline void retireDroppedReplay(Memory& mb, int64_t key, int32_t carriedDepth) {
            if (carriedDepth < 0) return;
            const int32_t* state = mb.existenceReplays.find(key);
            if (state == nullptr || (*state & 12) == 0) return;
            writeExistenceState(mb, key, *state & 3);
        }

private:
/// @brief Erase a retired completion identity without losing queued work.
        /// @details Callers preserve pending carriers before selecting this door.
        /// Dependency and pending-group edges leave with the authoritative key.
        /// @param mb Owning block.
        /// @param key Existing receipt with no pending carrier bits.
        /// @param arena Caller-owned page-tier scratch.
        /// @return Nothing.
inline void eraseExistenceState(Memory& mb, int64_t key, ScratchArena& arena) {
            const int32_t* state = mb.existenceReplays.find(key);
            assert(state != nullptr && (*state & 12) == 0);
            indexPendingExistence(mb, key, 0);
            mb.lbMemory.canonicalUses.receipts.erase(key, arena);
            mb.existenceReplays.erase(key);
        }

public:
/// @brief Resolve a construction source to its target-scope canonical identity.
        ///
        /// @details Reads the already maintained complete class set with the same
        /// chooser as expression entry. Source and validity spans are owned before
        /// the resulting NameMap identifier is minted. No classes are built here.
        /// @param mb Owning block with initialized canonical classes.
        /// @param source Concrete construction source spelling.
        /// @param scope Target validity identifier.
        /// @return Canonical source identifier in the owning NameMap.
        /// @invariant Completion and entry use the same target-scope representative.
inline NameId canonicalExistenceId(Memory& mb, StrSpan source, NameId scope) {
            assert(scope != 0);
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
            ScratchArena& strings = scratchArenas().forSlot(slot);
            ScratchScope frame(strings);
            const ScratchString text = ScratchString::copyFrom(strings, source.ptr, source.len);
            const StrSpan scopeView = mb.nameMap.decodeView(scope);
            const ScratchString validity = ScratchString::copyFrom(strings, scopeView.ptr, scopeView.len);
            int levels[256];
            int32_t count = 0;
            const CanonicalForm canonical = canonicalizer.canonicalFormAtScope(StrSpan(text), StrSpan(validity),
                mb, strings, levels, count, 256);
            const NameId result = mb.nameMap.encode(canonical.text);
            return result;
        }

public:
/// @brief Read completed and pending construction kinds at one scope.
        ///
        /// @details Completed bits live at the canonical identity. Pending bits
        /// are combined from equivalent raw carrier keys at exactly this scope;
        /// each carrier remains authoritative until its queued mail is consumed.
        /// @param mb Owning block and receipt index.
        /// @param canonical Canonical source identifier.
        /// @param scope Exact construction scope.
        /// @return Completed bits 0..1 and pending bits 2..3; no carrier-mode bit.
        /// @invariant Ancestor and sibling receipts never suppress this scope.
inline int32_t existenceStateAt(Memory& mb, NameId canonical, NameId scope) {
            const int32_t* completed = mb.existenceReplays.find(packOriginKey(canonical, scope));
            int32_t result = completed == nullptr ? 0 : *completed & 3;
            const unsigned slot = gl::WorkerContext::coreId() >= 0 ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            ScratchArena& arena = genScratchArenas().forSlot(slot);
            ensureCanonicalReceiptUses(mb, arena);
            mb.lbMemory.canonicalUses.pendingCarriers.forEach(packOriginKey(canonical, scope), [&](int64_t key) {
                const int32_t* state = mb.existenceReplays.find(key);
                assert(state != nullptr && (*state & 12) != 0);
                result |= *state & 12;
            });
            return result;
        }

public:
/// @brief Re-key completed constructions after the target classes settle.
        ///
        /// @details Moves only completed bits to the target-scope canonical key.
        /// Pending kinds and source-mode bits remain on the actual queued carrier.
        /// Snapshot iteration preserves colliding completions without duplicating mail.
        /// @param mb Owning block with consolidated canonical classes.
        /// @param scope Exact scope to rewrite.
        /// @return Number of completed receipt identities changed.
        /// @invariant No ancestor-class substitution rewrites a descendant receipt.
inline int32_t rewriteExistenceCompletions(Memory& mb, NameId scope,
                const PagedVector<NameId>* members = nullptr) {
            RT_SCOPE_HERE("CANONICAL_EXISTENCE_COMPLETIONS");
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
            ScratchArena& arena = genScratchArenas().forSlot(slot);
            ScratchScope frame(arena);
            DirtyState dirty = DirtyState::Clean;
            ensureCanonicalReceiptUses(mb, arena);
            PagedVector<int64_t> keys(&arena, &dirty);
            TypedColdSet<int64_t> selected(&arena, &dirty);
            if (members == nullptr) {
                auto& receipts = mb.lbMemory.existenceLookups.receipts;
                receipts.ensureCurrent(mb.existenceReplays, [](int64_t key) {
                    return static_cast<int64_t>(Codec<StatementKey>::decode(key).validity);
                });
                receipts.forEach(scope, [&](int64_t key) { selected.mint(key); });
            } else {
                for (int32_t i = 0; i < members->size(); ++i)
                    mb.lbMemory.canonicalUses.receipts.forEach(scope, mb.nameMap.decodeView((*members)[i]),
                        [&](int64_t key) { selected.mint(key); });
            }
            for (int32_t i = 1; i <= selected.count(); ++i) keys.push_back(selected.keyAt(i));
            const auto less = [&](int32_t a, int32_t b) {
                return compareSpans(mb.nameMap.decodeView(Codec<StatementKey>::decode(keys[a]).orig),
                    mb.nameMap.decodeView(Codec<StatementKey>::decode(keys[b]).orig)) < 0;
            };
            ChunkSortedOrdinals<decltype(less)> order(arena, keys.size(), 0, less);
            int32_t changed = 0;
            for (int32_t i = 0; i < keys.size(); ++i) {
                const int64_t key = keys[order.next()];
                const int32_t* found = mb.existenceReplays.find(key);
                assert(found != nullptr);
                const int32_t state = *found;
                indexPendingExistence(mb, key, state);
                if ((state & 3) == 0) continue;
                const NameId original = Codec<StatementKey>::decode(key).orig;
                const NameId canonical = canonicalExistenceId(mb, mb.nameMap.decodeView(original), scope);
                const int64_t target = packOriginKey(canonical, scope);
                if (target == key) continue;
                const int32_t* prior = mb.existenceReplays.find(target);
                const int32_t targetState = prior == nullptr ? 0 : *prior;
                writeExistenceState(mb, target, targetState | (state & 3));
                if ((state & 12) != 0) writeExistenceState(mb, key, state & 28);
                else eraseExistenceState(mb, key, arena);
                ++changed;
            }
            return changed;
        }

private:
/// @brief Replace dependency and canonical-source edges for one parked key.
        /// @details The authoritative source run is read without witness creation.
        /// Only this parked key's old edges are unlinked; other source indexes stay warm.
        /// @tparam Map Algebra or integration rejected-source map.
        /// @param mb Owning block with current complete scope classes.
        /// @param map Authoritative source records.
        /// @param uses Dependency index for these records.
        /// @param lookup Existing canonical-source lookup, possibly not yet built.
        /// @param key Marked expression/scope key to synchronize.
        /// @param arena Caller-owned page-tier scratch.
        /// @return Nothing.
template<class Map>
        inline void refreshCanonicalSourceKey(Memory& mb, const Map& map,
                CanonicalUseIndex& uses, ExistenceSourceLookup& lookup,
                int64_t key, ScratchArena& arena) {
            RT_SCOPE_HERE("CANONICAL_SOURCE_KEY_REFRESH");
            uses.erase(key, arena);
            if (lookup.built) lookup.eraseOwner(key, arena);
            const int32_t id = map.lookup(key);
            if (id == 0) return;
            const auto owner = Codec<StatementKey>::decode(key);
            canonicalTemplates.indexCanonicalText(uses, key, owner.validity, mb.templateInterner.decodeView(owner.orig));
            for (int32_t r = 0; r < map.runLen(id); ++r) {
                ScratchScope frame(arena);
                int32_t length = 0;
                const char* bytes = map.peekRecordBytes(id, r, length, arena);
                const RejectedValueBlobView source(bytes, length);
                canonicalTemplates.indexCanonicalText(uses, key, owner.validity,
                    mb.valueInterner.decodeView(source.sourceExpressionId()));
                for (int32_t j = 0; j < source.rewriteCount(); ++j)
                    canonicalTemplates.indexCanonicalText(uses, key, owner.validity, mb.valueInterner.decodeView(source.rewriteId(j)));
                if (lookup.built) lookup.add(packOriginKey(canonicalExistenceId(mb,
                    mb.valueInterner.decodeView(source.sourceExpressionId()), owner.validity), owner.validity), key);
            }
        }

public:
/// @brief Refresh only parked keys whose source dependencies changed.
        /// @details Source indexes remain built. Re-keying a complete class never
        /// clears unrelated source edges or reconstructs all source identities.
        /// @param mb Owning block.
        /// @param scope Exact scope of the changed complete class.
        /// @param members Names whose canonical text representative changed.
        /// An empty list still synchronizes authoritative source writes.
        /// @param arena Caller-owned page-tier scratch.
        /// @return Nothing.
inline void refreshCanonicalSourceUses(Memory& mb, NameId scope,
                const PagedVector<NameId>& members, ScratchArena& arena) {
            RT_SCOPE_HERE("CANONICAL_SOURCE_USES");
            DirtyState dirty = DirtyState::Clean;
            const auto refresh = [&](const auto& map, CanonicalUseIndex& uses, ExistenceSourceLookup& lookup) {
                uses.synchronize(map, arena, [&](int64_t key) {
                    refreshCanonicalSourceKey(mb, map, uses, lookup, key, arena);
                });
                TypedColdSet<int64_t> selected(&arena, &dirty);
                for (int32_t i = 0; i < members.size(); ++i)
                    uses.forEach(scope, mb.nameMap.decodeView(members[i]), [&](int64_t key) { selected.mint(key); });
                const auto less = [&](int32_t a, int32_t b) {
                    return compareSpans(mb.templateInterner.decodeView(Codec<StatementKey>::decode(selected.keyAt(a + 1)).orig),
                        mb.templateInterner.decodeView(Codec<StatementKey>::decode(selected.keyAt(b + 1)).orig)) < 0;
                };
                ChunkSortedOrdinals<decltype(less)> order(arena, selected.count(), 0, less);
                for (int32_t i = 0; i < selected.count(); ++i)
                    refreshCanonicalSourceKey(mb, map, uses, lookup, selected.keyAt(order.next() + 1), arena);
            };
            refresh(mb.overallHashMemory.rejectedMap, mb.lbMemory.canonicalUses.algebraSources,
                mb.lbMemory.existenceLookups.algebraSources);
            refresh(mb.overallHashMemory.rejectedMapIntegration, mb.lbMemory.canonicalUses.integrationSources,
                mb.lbMemory.existenceLookups.integrationSources);
        }

public:
/// @brief Reconstruct source/scope postings from an authoritative map.
        /// @details The first lookup scans all source records. Later production
        /// inserts, replacements and retirements maintain the distinct edges.
        /// No expanded products, witness names or history decisions enter this index.
        /// @tparam Map An algebra or integration rejected-source map.
        /// @param map Resident authoritative source map.
        /// @param lookup Derived source index belonging to this map.
        /// @param arena Temporary storage for record peeks.
        /// @param mb Owning block supplying canonical source identities.
        /// @return Void.
template<class Map>
        inline void ensureExistenceSourceLookup(const Map& map,
            ExistenceSourceLookup& lookup, ScratchArena& arena, Memory& mb) {
            if (lookup.built) return;
            mb.lbMemory.canonicalUses.dirty=DirtyState::Restructured;
            for (int32_t id = 1; id <= map.count(); ++id) {
                const int64_t key = map.keyAt(id);
                const NameId scope = Codec<StatementKey>::decode(key).validity;
                for (int32_t r = 0; r < map.runLen(id); ++r) {
                    ScratchScope frame(arena);
                    int32_t length = 0;
                    const char* bytes = map.peekRecordBytes(id, r, length, arena);
                    const RejectedValueBlobView source(bytes, length);
                    lookup.add(packInt32Pair(canonicalExistenceId(mb,
                        mb.valueInterner.decodeView(source.sourceExpressionId()), scope), scope), key);
                }
            }
            lookup.built = true;
        }

public:
/// @brief Remove completed source constructions from their own scope's parks.
        /// @details Receipt and source indexes select candidate keys. Candidates
        /// retain their original map order and use the unchanged receipt predicate.
        /// Ancestor parks and incomplete constructions survive in canonical order.
        /// @param mb Owning memory block.
        /// @param scopeId Scope whose successful materializations were recorded.
        /// @return Number of retired source records across both demand indexes.
inline int32_t retireCompletedExistences(Memory& mb, NameId scopeId) {
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
            ScratchArena& arena = genScratchArenas().forSlot(slot);
            ScratchScope frame(arena);
            DirtyState dirty = DirtyState::Clean;
            auto& indexes = mb.lbMemory.existenceLookups;
            indexes.receipts.ensureCurrent(mb.existenceReplays, [](int64_t key) {
                return static_cast<int64_t>(Codec<StatementKey>::decode(key).validity);
            });
            PagedVector<int32_t> completedSources(&arena, &dirty);
            indexes.receipts.forEach(scopeId, [&](int64_t key) {
                const int32_t* state = mb.existenceReplays.find(key);
                assert(state != nullptr);
                if ((*state & 3) == 0) return;
                completedSources.push_back(Codec<StatementKey>::decode(key).orig);
            });
            int32_t retired = 0;
            const auto prune = [&](auto& map, ExistenceSourceLookup& lookup) {
                ensureExistenceSourceLookup(map, lookup, arena, mb);
                struct Candidate { int64_t key; int32_t ordinal; };
                PagedVector<Candidate> keys(&arena, &dirty);
                ColdHashSet<PodKeyStore<int64_t>> seen(&arena, &dirty);
                for (int32_t i = 0; i < completedSources.size(); ++i) {
                    lookup.postings.forEach(packInt32Pair(completedSources[i], scopeId),
                        [&](int64_t key) {
                            const int32_t ordinal = map.lookup(key);
                            assert(ordinal != 0);
                            const int32_t before = seen.count();
                            seen.mint(key);
                            if (seen.count() != before) keys.push_back(Candidate{key, ordinal});
                        });
                }
                const auto less = [&](int32_t a, int32_t b) { return keys[a].ordinal < keys[b].ordinal; };
                ChunkSortedOrdinals<decltype(less)> order(arena, keys.size(), 0, less);
                for (int32_t i = 0; i < keys.size(); ++i) {
                    ScratchScope rowFrame(arena);
                    const int64_t key = keys[order.next()].key;
                    const auto snapshot = [&]() {
                        if constexpr (std::is_same_v<std::decay_t<decltype(map)>,
                            TypedColdBlobMap<int64_t, RejectedMapValue>>)
                            return snapshotRejectedRun(map, key, arena);
                        else return snapshotRejectedIntegrationRun(map, key, arena);
                    }();
                    PagedVector<int32_t> survivors(&arena, &dirty);
                    for (int32_t r = 0; r < snapshot.count; ++r) {
                        const auto& source = snapshot.views[r];
                        const NameId sourceId = canonicalExistenceId(mb,
                            mb.valueInterner.decodeView(source.sourceExpressionId()), scopeId);
                        const int32_t* receipt = mb.existenceReplays.find(packOriginKey(sourceId, scopeId));
                        const int32_t kinds = source.flags() & 3;
                        if (receipt && (*receipt & kinds) == kinds) ++retired;
                        else survivors.push_back(r);
                    }
                    if (survivors.size() == snapshot.count) continue;
                    for (int32_t r = 0; r < snapshot.count; ++r)
                        lookup.remove(packInt32Pair(canonicalExistenceId(mb,
                            mb.valueInterner.decodeView(snapshot.views[r].sourceExpressionId()), scopeId), scopeId), key);
                    if (survivors.empty()) {
                        map.eraseBlobIf([&](int64_t stored) { return stored == key; });
                    } else {
                        const int32_t id = map.inner().assignRun(key, nullptr, nullptr, 0);
                        for (int32_t r = 0; r < survivors.size(); ++r) {
                            const auto& view = snapshot.views[survivors[r]];
                            map.inner().appendBlobToRun(id, view.p, view.len);
                            lookup.add(packInt32Pair(canonicalExistenceId(mb,
                                mb.valueInterner.decodeView(view.sourceExpressionId()), scopeId), scopeId), key);
                        }
                    }
                }
                lookup.postings.maintain();
            };
            prune(mb.overallHashMemory.rejectedMap, indexes.algebraSources);
            prune(mb.overallHashMemory.rejectedMapIntegration, indexes.integrationSources);
            return retired;
        }
private:
const ProverParameters& parameters;
Canonicalizer& canonicalizer;
CanonicalTemplateMatcher& canonicalTemplates;
};
}

namespace gl {
/// @brief Bind the readers needed by rejected-source storage operations.
/// @details The supplied block retains parked records, completion receipts and indices.
/// @param parameters Existing source-registration settings.
/// @param canonicalizer Canonical expression-view operations.
/// @param canonicalTemplates Canonical template and source-index operations.
/// @return Constructed rejected-source memory interface.
inline RejectedSourceMemory::RejectedSourceMemory(const ProverParameters& parameters, Canonicalizer& canonicalizer, CanonicalTemplateMatcher& canonicalTemplates)
 : parameters(parameters), canonicalizer(canonicalizer), canonicalTemplates(canonicalTemplates) {}
} // namespace gl
