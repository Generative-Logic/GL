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
#include "statement_registry_operations.hpp"
#include "../infra/rt_tracker.hpp"

namespace gl {

/// @brief `StrSpan` door of @ref removeExpressionFromMemoryBlock — erase a
        ///        statement (`state==0`) or discharge a goal (`state==1`) addressed
        ///        by its `(original, validityName)` bytes, with no
        ///        `EncodedExpression` materialized.
        ///
        /// @details
        /// Byte-for-byte the same procedure as the `EncodedExpression` overload,
        /// which now delegates here. Both names are resolved through the
        /// non-minting `NameMap::lookup(StrSpan)` — identical verdict to
        /// `lookup(const std::string&)` (which itself forwards to the span form) —
        /// so a never-interned target finds no stored row and the sweep is the same
        /// no-op the string scan produced. `state == 0` erases every matching
        /// `(originalId, validityId)` row from `intLocalEncodedStatements` /
        /// `intLocalEncodedStatementsDelta` / `intEncodedStatements` (back to front,
        /// indices stable) and repairs the persistent per-class statement-index
        /// waterlines for the `intEncodedStatements` positions it erased (see
        /// `repairEqClassWaterlines` — a stale waterline silently under-applies
        /// classes); `state == 1` discharges the goal from `intToBeProved`
        /// (the caller guarantees both names interned, so a miss is a contract
        /// violation asserted, not a probe miss). Lets a span-holding caller —
        /// `ordisMerge`'s branch-cleanup loop, which builds each branch validity on
        /// the scratch arena — remove without a transient `EncodedExpression`.
        ///
        /// @param original     The statement's canonical expression bytes.
        /// @param validityName The statement's validity-scope bytes.
        /// @param mb           The owning memory block.
        /// @param state        `0` = erase encoded-statement rows; `1` = discharge
        ///                     the `intToBeProved` goal.
        /// @see removeExpressionFromMemoryBlock(const EncodedExpression&, Memory&, int)
        ///      — the delegating owning-string overload.
/// @return Nothing; updates the specified storage in place.
void StatementRegistryOperations::removeExpressionFromMemoryBlock(StrSpan original, StrSpan validityName, Memory& mb, int state) {
    if (state == 0)
    {
        // Match on the int rows' (originalId, validityId) — equivalent to the
        // full-struct compare because stored rows are canonical-pipeline
        // encodings whose every field derives from that pair. lookup is
        // non-minting: a never-interned target has no stored row, so the
        // sweep finds nothing — the same no-op the string scan produced.
        const NameId origId = mb.nameMap.lookup(original);
        const NameId valId = mb.nameMap.lookup(validityName);
        if (origId == 0 || valId == 0) return;
        // Erase back to front to keep indices valid.
        auto eraseIntRows = [origId, valId](SharedExpressionList& intVec) {
            for (int32_t i = intVec.size(); i-- > 0; ) {
                if (intVec[i].originalId == origId
                    && intVec[i].validityId == valId) {
                    intVec.erase(i);
                }
            }
        };
        {
            RT_SCOPE_HERE("RMEXPR_LOCAL_ERASE");
            eraseIntRows(mb.intLocalEncodedStatements);
            eraseIntRows(mb.intLocalEncodedStatementsDelta);
        }
        // intEncodedStatements erasures shift the positions the persistent
        // per-class statement-index waterlines (eqClassSttmntIndexMapMap)
        // point into — record each erased position and repair the
        // waterlines, or equivalence-class application silently skips the
        // rows that slid under a stale waterline. The exact-key admission
        // gates keep (originalId, validityId) unique in the list, so the
        // capture buffer is a loud Rule-19 tripwire, not a real capacity.
        int32_t erasedPos[8];
        int32_t erasedN = 0;
        RT_SCOPE_HERE("RMEXPR_ENCODED_ERASE");
        for (int32_t i = mb.intEncodedStatements.size(); i-- > 0; ) {
            if (mb.intEncodedStatements[i].originalId == origId
                && mb.intEncodedStatements[i].validityId == valId) {
                assert(erasedN < 8
                    && "removeExpressionFromMemoryBlock: duplicate (text, scope) rows exceed the erased-position capture");
                erasedPos[erasedN++] = i;
                mb.intEncodedStatements.erase(i);
            }
        }
        if (erasedN > 0) {
            RT_SCOPE_HERE("RMEXPR_WATERLINES");
            // Collected back to front; the repair takes an ascending run.
            std::sort(erasedPos, erasedPos + erasedN);
            repairEqClassWaterlines(mb, erasedPos, erasedN);
        }
    }
    else if (state == 1) {
        // Remove from the goal registry. The caller guarantees the goal is
        // registered, so both names were interned by its insert site — a
        // lookup miss here is a contract violation, not a probe miss.
        const NameId tbpOrigId = mb.nameMap.lookup(original);
        const NameId tbpValId = mb.nameMap.lookup(validityName);
        assert(tbpOrigId != 0 && tbpValId != 0
               && "removeExpressionFromMemoryBlock: goal names not interned");
        const int64_t pkTBP = packStatementKey(tbpOrigId, tbpValId);
        assert(mb.intToBeProved.lookup(pkTBP) != 0);
        mb.intToBeProved.eraseSet(pkTBP);
    }
}

/// @brief Remove encoded rows or a goal addressed by an owning expression.
/// @details Delegates to the span overload without copying either string.
/// @param encExpr Expression and exact scope to remove.
/// @param mb Owning resident block.
/// @param state Zero removes statement-list rows; one removes the existing goal.
/// @return Nothing; updates the specified storage in place.
void StatementRegistryOperations::removeExpressionFromMemoryBlock(const EncodedExpression& encExpr, Memory& mb, int state) {
    // Owning-string overload: forward the two names as spans. NameMap::lookup
    // interns nothing, so the id verdict — and the erase / discharge it drives —
    // is identical to reading encExpr.original / encExpr.validityName directly.
    removeExpressionFromMemoryBlock(StrSpan(encExpr.original), StrSpan(encExpr.validityName), mb, state);
}

/// @brief Bind an empty batch to a gen-scratch arena.
///
/// @details
/// Nothing is allocated until the first staged removal. The arena must be a
/// per-slot `genScratchArenas()` arena (page tier) that outlives the batch.
///
/// @param arena The gen-scratch arena holding the recorded pairs.
StatementRegistryOperations::RowRemovalBatch::RowRemovalBatch(ScratchArena& arena)
    : arena_(arena), keys_(&arena, &dirty_) {}

/// @brief Record the removal of every row of one statement.
///
/// @details
/// Resolves both names through the non-minting `NameMap::lookup`, exactly as
/// the immediate removal does. A name that was never interned has no stored
/// row, so nothing is recorded (the immediate removal's no-op). The same pair
/// may be recorded more than once; it is removed once.
///
/// @param original     The statement's canonical expression bytes.
/// @param validityName The statement's validity-scope bytes.
/// @param mb           The owning memory block.
/// @return Nothing.
void StatementRegistryOperations::RowRemovalBatch::stage(StrSpan original, StrSpan validityName, const Memory& mb) {
    const NameId origId = mb.nameMap.lookup(original);
    const NameId valId = mb.nameMap.lookup(validityName);
    if (origId == 0 || valId == 0) return;
    keys_.push_back(packStatementKey(origId, valId));
}

/// @brief Erase the rows of every recorded statement and repair the
///        waterlines, then empty the batch.
///
/// @details
/// Sorts the recorded pairs (duplicates dropped), compacts
/// `intLocalEncodedStatements` and `intLocalEncodedStatementsDelta`, then
/// `intEncodedStatements` while collecting its erased positions in ascending
/// order, and repairs `eqClassSttmntIndexMapMap` once over all of them. The
/// per-statement duplicate tripwire of the immediate removal (at most eight
/// rows of one pair) is kept. An empty batch changes nothing.
///
/// @param mb The owning memory block; the same block every pair was recorded
///           against.
/// @return Nothing.
/// @invariant The lists and waterlines equal those left by
///            `removeExpressionFromMemoryBlock(state 0)` called once per
///            recorded pair, in any order.
void StatementRegistryOperations::RowRemovalBatch::apply(Memory& mb) {
    const int32_t staged = keys_.size();
    if (staged == 0) return;
    ScratchScope applyScope(arena_);
    int64_t* sorted = reinterpret_cast<int64_t*>(arena_.resolve(arena_.alloc(
        staged * static_cast<int32_t>(sizeof(int64_t)),
        static_cast<int32_t>(alignof(int64_t)))));
    for (int32_t i = 0; i < staged; ++i) sorted[i] = keys_[i];
    std::sort(sorted, sorted + staged);
    const int32_t pairs = static_cast<int32_t>(std::unique(sorted, sorted + staged) - sorted);
    // Index of a row's pair among the recorded pairs, or -1.
    const auto pairIndex = [&](const IntEncodedExpr& row) -> int32_t {
        const int64_t key = packStatementKey(row.originalId, row.validityId);
        const int64_t* at = std::lower_bound(sorted, sorted + pairs, key);
        return (at != sorted + pairs && *at == key) ? static_cast<int32_t>(at - sorted) : -1;
    };
    {
        RT_SCOPE_HERE("RMEXPR_LOCAL_ERASE");
        mb.intLocalEncodedStatements.eraseIf(
            [&](const IntEncodedExpr& row) { return pairIndex(row) >= 0; });
        mb.intLocalEncodedStatementsDelta.eraseIf(
            [&](const IntEncodedExpr& row) { return pairIndex(row) >= 0; });
    }
    // intEncodedStatements erasures shift the positions the persistent
    // per-class statement-index waterlines point into; the erased positions
    // (pre-erase, ascending) feed one repair. The exact-key admission gates
    // keep each pair unique in the list, so the per-pair count is the same
    // loud Rule-19 tripwire the immediate removal carries.
    int32_t* perPair = reinterpret_cast<int32_t*>(arena_.resolve(arena_.alloc(
        pairs * static_cast<int32_t>(sizeof(int32_t)),
        static_cast<int32_t>(alignof(int32_t)))));
    for (int32_t k = 0; k < pairs; ++k) perPair[k] = 0;
    DirtyState erasedDirty = DirtyState::Clean;
    PagedVector<int32_t> erased(&arena_, &erasedDirty);
    {
        RT_SCOPE_HERE("RMEXPR_ENCODED_ERASE");
        mb.intEncodedStatements.eraseIf(
            [&](const IntEncodedExpr& row) {
                const int32_t k = pairIndex(row);
                if (k < 0) return false;
                ++perPair[k];
                assert(perPair[k] <= 8
                    && "RowRemovalBatch::apply: duplicate (text, scope) rows exceed the erased-position capture");
                return true;
            },
            [&](int32_t position) { erased.push_back(position); });
    }
    if (erased.size() > 0) {
        RT_SCOPE_HERE("RMEXPR_WATERLINES");
        repairEqClassWaterlines(mb, erased, erased.size());
    }
    keys_.clear();
}
}
