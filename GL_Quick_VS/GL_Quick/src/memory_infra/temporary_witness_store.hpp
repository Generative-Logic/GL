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

#include "cold_hash_map.hpp"
#include "paged_vector.hpp"
#include "str_ops.hpp"
#include <cassert>
#include <climits>
#include <cstdio>

namespace gl {

/// @brief The algebraic, integration and forced witness constructions supported by disintegration.
/// @details Kind and generation are metadata, independent of token spelling.
enum class TemporaryWitnessKind : uint8_t { Algebra, Integration, Forced };

/// @brief Admission budgets evaluated from expansion-local witness metadata.
/// @details Secondary count is an occurrence count, excluding known recursion
/// products exactly as for permanent names. Generation is independent of text.
struct WitnessAdmissionMetrics {
    int generation;
    int secondaryCount;
};

/// @brief One expansion-local witness slot and its admission state.
/// @details The parent is an earlier slot, or zero for a top-level existence.
/// `parentElement` is the index of the parent existence's element whose
/// expansion contains this slot, or -1 when it is not recorded (a top-level
/// slot). Permanent text is retained only within this expansion's scratch frame.
struct TemporaryWitnessSlot {
    TemporaryWitnessKind kind;
    int generation;
    int32_t parent;
    int32_t structuralPosition;
    int32_t tokenId;
    int32_t permanentId;
    int32_t sourceId;
    int32_t subtreeEnd;
    int32_t integrationDepth;
    int32_t parentElement;
};

/// @brief Own reusable witness placeholders for one disintegration expansion.
///
/// @details Storage lives on the supplied container scratch arena. A distinct
/// context number separates simultaneously active expansions; callers may reuse
/// the context after its frame ends. No operation touches an LB interner or
/// history. Only materialize advances the caller's permanent witness counter.
///
/// @invariant Slot identity and kind come from records, never permanent-name parsing.
/// @see TemporaryWitnessSlot, TemporaryWitnessKind.
class TemporaryWitnessStore {
    DirtyState dirty_ = DirtyState::Clean;
    ColdHashSet<BytesKeyStore> strings_;
    ColdHashMap<PodKeyStore<int32_t>, int32_t> tokenSlots_;
    PagedVector<TemporaryWitnessSlot> slots_;
    int32_t context_;

public:
    /// @brief Bind an empty expansion to container scratch storage.
    ///
    /// @details The arena outlives this store and may also hold nested stores.
    /// The context is unique among live expansions whose tokens can meet.
    /// @param arena Container scratch arena, never the string scratch arena.
    /// @param context Nonnegative live expansion identifier.
    /// @return A store with no allocated slots.
    TemporaryWitnessStore(LbArena& arena, int32_t context)
        : strings_(&arena, &dirty_), tokenSlots_(&arena, &dirty_),
          slots_(&arena, &dirty_), context_(context) {
        assert(context >= 0);
    }
    TemporaryWitnessStore(const TemporaryWitnessStore&) = delete;
    TemporaryWitnessStore& operator=(const TemporaryWitnessStore&) = delete;

    /// @brief Append a temporary slot without allocating a permanent witness.
    ///
    /// @details Append order is structural discovery order. The generated token
    /// is an opaque handle; its generation and kind are read from slot metadata.
    /// @param kind Algebra, integration or forced construction.
    /// @param generation Generation carried by the source firing.
    /// @param parent Enclosing slot identifier, or zero at the top level.
    /// @param structuralPosition Stable source traversal position.
    /// @param source Instantiated source existence, when supplied by disintegration.
    /// @param integrationDepth Shared outer-expansion integration assignment.
    /// @param parentElement Index of the parent existence's element whose
    ///        expansion contains the slot, or -1 when not recorded.
    /// @return The new one-based slot identifier.
    int32_t append(TemporaryWitnessKind kind, int generation,
                   int32_t parent, int32_t structuralPosition,
                   StrSpan source = StrSpan(), int32_t integrationDepth = 0,
                   int32_t parentElement = -1) {
        assert(generation >= 0 && structuralPosition >= 0 && integrationDepth >= 0);
        assert(parent >= 0 && parent <= slots_.size());
        assert(parentElement >= -1 && (parent != 0 || parentElement == -1));
        assert(slots_.size() < INT_MAX);
        const int32_t id = slots_.size() + 1;
        char bytes[96];
        const int len = std::snprintf(bytes, sizeof(bytes),
            "__gl_temporary_witness_%d_%d", context_, id);
        assert(len > 0 && len < static_cast<int>(sizeof(bytes)));
        const int32_t tokenId = strings_.mint(StrSpan(bytes, len));
        assert(tokenSlots_.lookup(tokenId) == 0);
        tokenSlots_.insert(tokenId, id);
        slots_.push_back(TemporaryWitnessSlot{
            kind, generation, parent, structuralPosition, tokenId, 0,
            source.empty() ? 0 : strings_.mint(source), id + 1, integrationDepth, parentElement});
        return id;
    }

    /// @brief Count the slots in structural discovery order.
    /// @details The count changes only when discovery appends a slot.
    /// @return Number of current expansion-local slots.
    int32_t count() const { return slots_.size(); }

    /// @brief Finish recording a witness's recursively expanded subtree.
    /// @details Publication skips an unadmitted subtree in one step using this
    /// exclusive endpoint; no instruction tree persists after the expansion.
    /// @param id One-based slot whose descendants have all been discovered.
    /// @return Void.
    void closeSubtree(int32_t id) {
        TemporaryWitnessSlot value = slot(id);
        value.subtreeEnd = slots_.size() + 1;
        slots_.setAt(id - 1, value);
    }

    /// @brief Read a witness's source existence.
    /// @details The returned view lives until the next mutation of this store.
    /// @param id One-based slot with a recorded disintegration source.
    /// @return The source expression, potentially containing outer temporary slots.
    StrSpan source(int32_t id) const {
        const TemporaryWitnessSlot value = slot(id);
        assert(value.sourceId != 0);
        return strings_.keyAt(value.sourceId);
    }

    /// @brief Read an already materialized witness name.
    /// @details This read allocates nothing and never advances the witness counter.
    /// @param id One-based admitted slot.
    /// @return Its permanent name, valid until the next store mutation.
    StrSpan permanent(int32_t id) const {
        const TemporaryWitnessSlot value = slot(id);
        assert(value.permanentId != 0);
        return strings_.keyAt(value.permanentId);
    }

    /// @brief Read metadata for an expansion-local slot.
    ///
    /// @details Returns a value copy so nested arena growth cannot invalidate it.
    /// @param id One-based slot identifier belonging to this expansion.
    /// @return The slot's complete metadata and materialization state.
    TemporaryWitnessSlot slot(int32_t id) const {
        assert(id > 0 && id <= slots_.size());
        return slots_[id - 1];
    }

    /// @brief Resolve an opaque token in this expansion without minting.
    ///
    /// @details Tokens belonging to another live context and permanent names
    /// are ordinary misses. Only temporary tokens appear in this index.
    /// @param token Complete argument token to look up.
    /// @return Its one-based slot identifier, or zero for a nonlocal token.
    int32_t lookup(StrSpan token) const {
        const int32_t tokenId = strings_.lookup(token);
        if (tokenId == 0) return 0;
        const int32_t entry = tokenSlots_.lookup(tokenId);
        return entry == 0 ? 0 : tokenSlots_.valueAt(entry);
    }

    /// @brief Read the temporary text used during structural expansion.
    ///
    /// @details The view is valid until this store is mutated or destroyed.
    /// @param id One-based slot identifier.
    /// @return The expansion-local placeholder token.
    StrSpan token(int32_t id) const {
        return strings_.keyAt(slot(id).tokenId);
    }

    /// @brief Allocate the permanent name of an admitted slot exactly once.
    ///
    /// @details Call only after admission and dependency closure authorize the
    /// slot. Repeated calls reuse its name within this expansion. Other scopes
    /// use their own expansion and therefore allocate their own fresh names.
    /// @param id One-based admitted slot identifier.
    /// @param level Owning logic-block level.
    /// @param nextPermanent The LB's monotonic permanent witness counter.
    /// @return The permanent name, valid until this store is mutated or destroyed.
    StrSpan materialize(int32_t id, int level, int& nextPermanent) {
        TemporaryWitnessSlot value = slot(id);
        assert(level >= 0 && nextPermanent >= 0);
        assert(value.parent == 0 || slot(value.parent).permanentId != 0);
        if (value.permanentId == 0) {
            assert(nextPermanent < INT_MAX);
            char bytes[96];
            const int len = value.kind == TemporaryWitnessKind::Algebra
                ? std::snprintf(bytes, sizeof(bytes), "it_%d_lev_%d_%d",
                    value.generation, level, nextPermanent)
                : value.kind == TemporaryWitnessKind::Forced
                    ? std::snprintf(bytes, sizeof(bytes), "force_lev_%d_%d", level, nextPermanent)
                    : std::snprintf(bytes, sizeof(bytes), "int_%d_lev_%d_%d",
                        value.integrationDepth, level, nextPermanent);
            assert(len > 0 && len < static_cast<int>(sizeof(bytes)));
            value.permanentId = strings_.mint(StrSpan(bytes, len));
            slots_.setAt(id - 1, value);
            ++nextPermanent;
        }
        return strings_.keyAt(value.permanentId);
    }

    /// @brief Give a slot a permanent name minted earlier, without drawing a
    ///        new one (D-481).
    ///
    /// @details
    /// A passed integration trial admits at its parent scope the witnesses it
    /// minted when it opened, under the same names. The promotion expands the
    /// source again and hands each listed slot its recorded name here instead
    /// of `materialize`; the LB's permanent counter is untouched, because the
    /// name was counted when it was first minted.
    ///
    /// @param id   One-based slot identifier; its enclosing slot already has a
    ///             permanent name, it has none yet.
    /// @param name The recorded permanent name (non-empty).
    /// @return The permanent name, valid until this store is mutated or destroyed.
    /// @see materialize, DisintegrationProcessor::disintegrateExpr2.
    StrSpan materializeAs(int32_t id, StrSpan name) {
        TemporaryWitnessSlot value = slot(id);
        assert(name.len > 0 && "materializeAs: a recorded name is non-empty");
        assert(value.parent == 0 || slot(value.parent).permanentId != 0);
        assert(value.permanentId == 0 && "materializeAs: a slot is named once");
        value.permanentId = strings_.mint(name);
        slots_.setAt(id - 1, value);
        return strings_.keyAt(value.permanentId);
    }
};

} // namespace gl
