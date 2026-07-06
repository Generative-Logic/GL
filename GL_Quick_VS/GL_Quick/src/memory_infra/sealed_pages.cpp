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

#include "sealed_pages.hpp"

namespace gl {

    /// @brief Asserts the lifecycle completed: no blocks outstanding
    ///        (freed after consumption, or never used).
    SealedPageSet::~SealedPageSet() {
        assert(bump_.blocksHeld() == 0
            && "SealedPageSet destroyed with blocks outstanding — the "
               "consumer must freePages() after draining the records");
    }

    /// @brief Exact-length allocation at the bump head; `Filling` only.
    ///
    /// @details
    /// A within-page bump on the set's own cold per-task arena (`LbArena`):
    /// exact-length, never straddling a page, the first allocation acquiring
    /// the first page. The arena's own asserts cover a positive length that
    /// fits a page.
    ///
    /// @param bytes Exact length of the new allocation; > 0, at most one
    ///              page.
    /// @return Pointer to `bytes` writable bytes, stable until
    ///         `freePages`.
    char* SealedPageSet::alloc(int32_t bytes) {
        assert(global_ != nullptr && "SealedPageSet::alloc before bind");
        assert(state_ == State::Filling
            && "SealedPageSet::alloc outside the Filling window");
        char* out = bump_.allocBytes(bytes);
        bytesAllocated_ += bytes;
        return out;
    }

    /// @brief Consumption complete: poison every filled page, return every
    ///        block, `Sealed` → `Freed`.
    ///
    /// @details
    /// Called by the consumer (the drain side) immediately after the records
    /// referencing this set are minted into their persistent id form.
    /// `reset()` frees every filled page — poisoning each with the cold
    /// arena byte (0xCD) and bumping the arena generation — so a straggler
    /// view reads loud garbage even if the block is re-granted later (the
    /// `State` machine, not the generation, is `SealedString`'s liveness
    /// guard); `releaseAll` then returns every block through the cold path.
    /// The record chain dies with the pages — its nodes live in the same
    /// blocks — so the chain state is zeroed here too (a `Freed` set
    /// reports nothing; every chain reader asserts on the state flip).
    /// Records never read before the free are simply dropped with their
    /// pages — the redo-discard shape, a defined outcome needing no
    /// separate cleanup.
    void SealedPageSet::freePages() {
        assert(state_ == State::Sealed
            && "SealedPageSet::freePages on an unsealed set — seal at "
               "task end, free after consumption");
        bump_.reset();        // poison every filled page before release
        bump_.releaseAll();   // return every hot block to the pool
        recordHead_ = nullptr;
        recordTail_ = nullptr;
        recordCount_ = 0;
        recordStride_ = 0;
        state_ = State::Freed;
    }

}
