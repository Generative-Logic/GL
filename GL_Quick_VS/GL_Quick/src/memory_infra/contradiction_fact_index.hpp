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
// ContradictionFactIndex -- the derived fact side of the contradiction-scope
// demand: the block's known statements under the masked head keys of the rule
// heads that negate them.
//
// A rule chain enters the masked family of `OriginalsHeadIndex` under its
// head's masked key (`maskedHeadKey`: sign, operator, arguments with the one
// or two `(1)`-typed constants masked). A fact F is negated by such a head
// exactly when the head's key equals F's key with the sign flipped, masked at
// the same slots. This index files every fact under each of those
// sign-flipped keys, so a rule chain installed after a contradiction scope
// opened finds the facts its head negates with one probe instead of a walk
// over every fact.
//
// It is DERIVED (I-117 pattern): it lives on the owning block's deloadable
// `LbArena` (pool-backed, zero heap), is never enrolled in
// `LbMemory::visitContainers`, never in the canonical deload stream and never
// dumped; it is cleared at the canonical release seam, at the canonical
// reload, at discharge and at every teardown (the raw image carries its pages
// verbatim), and rebuilt lazily by its one consumer,
// `ContradictionDemand::drainContradictionDemands`.
#include "cold_hash_map.hpp"
#include "or_implication_postings.hpp"

#include <cassert>
#include <cstdint>

namespace gl {

    /// @brief Derived masked-key to fact index of the contradiction-scope
    ///        demand (D-560).
    ///
    /// @details
    /// Three pieces on the owning block's arena with a private dirty flag (a
    /// derived container never marks the block dirty):
    /// - `facts_` — the packed `(original, scope)` keys of the filed facts; a
    ///   fact is filed once (@ref addFact reports a repeat).
    /// - `keys_` — the distinct masked keys, interned to dense label ids.
    /// - `postings_` — label id (widened to the posting key) to the facts
    ///   filed under that key.
    ///
    /// The index is a superset of the block's current facts at the scopes a
    /// registered contradiction scope sees: a fact erased after it was filed
    /// stays filed, so the consumer confirms that a returned fact is still
    /// known. Iteration order under one key is internal; the consumer sorts
    /// before any observable work.
    ///
    /// @invariant Every fact @ref containsFact reports was filed under every
    ///            key its filer supplied, and under no other key.
    /// @invariant Derived (I-117): never deloaded, never enrolled in
    ///            `LbMemory::visitContainers`, never dumped.
    /// @see ContradictionDemand::drainContradictionDemands, OriginalsHeadIndex,
    ///      maskedHeadKey.
    class ContradictionFactIndex {
    public:
        /// @brief Bind an empty unbuilt index to the owning block's arena.
        ///
        /// @details
        /// Lazy like every arena container: construction touches neither the
        /// arena nor a pool block, so an unused index consumes nothing.
        ///
        /// @param arena The block's arena; must be non-null.
        explicit ContradictionFactIndex(LbArena* arena)
            : facts_(arena, &dirty_), keys_(arena, &dirty_), postings_(arena) {
            assert(arena != nullptr && "ContradictionFactIndex: null arena");
        }

        ContradictionFactIndex(const ContradictionFactIndex&) = delete;
        ContradictionFactIndex& operator=(const ContradictionFactIndex&) = delete;
        ContradictionFactIndex(ContradictionFactIndex&&) = delete;
        ContradictionFactIndex& operator=(ContradictionFactIndex&&) = delete;

        /// @brief Record a fact as filed.
        ///
        /// @details
        /// The filer calls this once before @ref fileUnder for each of the
        /// fact's keys; a fact already recorded is a defined repeat whose
        /// keys are not filed again.
        ///
        /// @param fact Packed `(original, scope)` key of a known statement.
        /// @return True when the fact is new and its keys are to be filed;
        ///         false when it was already filed.
        bool addFact(int64_t fact) {
            if (facts_.lookup(fact) != 0) return false;
            facts_.mint(fact);
            return true;
        }

        /// @brief File a recorded fact under one masked key.
        ///
        /// @details
        /// Interns the key's bytes (copied at once, so the caller may reuse
        /// its buffer) and adds the fact to the key's postings.
        ///
        /// @param key  Masked key bytes (`maskedHeadKey`, sign flipped);
        ///             non-empty.
        /// @param fact The fact, recorded by @ref addFact.
        void fileUnder(StrSpan key, int64_t fact) {
            assert(key.len > 0 && "ContradictionFactIndex::fileUnder: empty key");
            assert(facts_.lookup(fact) != 0
                && "ContradictionFactIndex::fileUnder: fact not recorded by addFact");
            const int32_t label = keys_.mint(key);
            assert(label >= 1 && "ContradictionFactIndex::fileUnder: mint returned a null id");
            postings_.insert(static_cast<int64_t>(label), fact);
        }

        /// @brief Visit the facts filed under one masked key.
        ///
        /// @details
        /// A key no fact carries yields nothing. The visitor must not mutate
        /// the index.
        ///
        /// @tparam Sink Callable `void(int64_t fact)`.
        /// @param key  Masked key bytes (a chain's `maskedHeadKey`).
        /// @param sink The visitor.
        template <typename Sink>
        void factsWithKey(StrSpan key, Sink&& sink) const {
            const int32_t label = keys_.lookup(key);
            if (label == 0) return;
            postings_.forEach(static_cast<int64_t>(label), sink);
        }

        /// @brief Whether a fact has been filed.
        /// @param fact Packed `(original, scope)` key.
        /// @return True iff @ref addFact recorded it since the last clear.
        bool containsFact(int64_t fact) const { return facts_.lookup(fact) != 0; }

        /// @brief Number of filed facts.
        /// @return The recorded-fact count.
        int32_t factCount() const { return facts_.count(); }

        /// @brief Whether the index covers the facts every participating
        ///        contradiction scope sees.
        /// @return True after @ref markBuilt, false after @ref clear. A raw
        ///         reload restores the pages and this flag together.
        bool built() const { return built_; }

        /// @brief Declare the rebuild complete.
        void markBuilt() { built_ = true; }

        /// @brief Free every page and forget the build.
        ///
        /// @details
        /// Called at the canonical release seam (before the arena's blocks are
        /// reclaimed), at the canonical reload, at discharge, at every
        /// teardown, and by the consumer when no contradiction scope remains.
        void clear() {
            postings_.clear();
            keys_.resetToFresh();
            facts_.resetToFresh();
            built_ = false;
            dirty_ = DirtyState::Clean;
        }

        /// @brief Retained bytes for the memory census.
        /// @return Bytes pinned by the three pieces.
        int64_t liveBytes() const {
            return peak_snapshot::measure("facts_", facts_, [&] { return facts_.liveBytes(); })
                + peak_snapshot::measure("keys_", keys_, [&] { return keys_.liveBytes(); })
                + peak_snapshot::measure("postings_", postings_, [&] { return postings_.liveBytes(); });
        }

    private:
        DirtyState dirty_ = DirtyState::Clean;
        ColdHashSet<PodKeyStore<int64_t>> facts_;
        ColdHashSet<BytesKeyStore> keys_;
        OrPostingIndex<int64_t> postings_;
        bool built_ = false;
    };

} // namespace gl
