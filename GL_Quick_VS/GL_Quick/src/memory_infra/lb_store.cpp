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

#include "lb_store.hpp"

namespace gl {

    /// @brief Compute the slot stride and per-block slot count on first use.
    ///
    /// @details
    /// See the declaration. Reads the manager's block size (valid by now — the
    /// pool is initialized before any LB is created), rounds the slot size up to
    /// the alignment for the stride, and divides the block into whole slots.
    /// Asserts the stride leaves room for the intrusive free-list link and that
    /// at least one slot fits a block.
    void LbStore::ensureGeometry() {
        if (slotsPerBlock_ != 0) {
            return;
        }
        const std::size_t blockBytes =
            static_cast<std::size_t>(manager_->blockBytes());
        slotStride_ = (slotBytes_ + slotAlign_ - 1) / slotAlign_ * slotAlign_;
        assert(slotStride_ >= sizeof(FreeSlot));
        slotsPerBlock_ = blockBytes / slotStride_;
        assert(slotsPerBlock_ >= 1
            && "LB-body block too small for one Memory slot — raise "
               "static_lb_block_bytes in parameters.hpp");
    }

    /// @brief Hand out a raw, aligned, slot-sized chunk of pool memory.
    ///
    /// @details
    /// See the declaration. Free-list first, then bump-carve, then a fresh
    /// block. The block base is pool-aligned and the stride a multiple of
    /// `slotAlign_`, so every carved slot inherits the alignment — asserted.
    void* LbStore::allocateSlot() {
        std::lock_guard<std::mutex> lock(mutex_);
        ensureGeometry();

        void* slot = nullptr;
        if (freeHead_ != nullptr) {
            slot = freeHead_;
            freeHead_ = freeHead_->next;
        }
        else {
            if (currentBlock_ == nullptr || currentCursor_ >= slotsPerBlock_) {
                currentBlock_ = manager_->acquireBlock();
                currentCursor_ = 0;
                blocks_.push_back(currentBlock_);
            }
            slot = currentBlock_ + currentCursor_ * slotStride_;
            ++currentCursor_;
        }

        assert(reinterpret_cast<std::uintptr_t>(slot) % slotAlign_ == 0
            && "LbStore handed out a misaligned slot");
        ++slotsInUse_;
        if (slotsInUse_ > peakSlotsInUse_) {
            peakSlotsInUse_ = slotsInUse_;
        }
        return slot;
    }

    /// @brief Return a slot to the store for reuse.
    ///
    /// @details
    /// See the declaration. Parks the slot on the intrusive free-list; the block
    /// is never returned to the pool. The in-use underflow assert is the
    /// double-free tripwire.
    void LbStore::freeSlot(void* slot) {
        std::lock_guard<std::mutex> lock(mutex_);
        assert(slot != nullptr);
        assert(slotsInUse_ > 0 && "LbStore::freeSlot underflow (double free?)");
        FreeSlot* node = static_cast<FreeSlot*>(slot);
        node->next = freeHead_;
        freeHead_ = node;
        --slotsInUse_;
    }

    /// @brief Live slot count (creates minus destroys). Telemetry only.
    std::size_t LbStore::slotsInUse() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return slotsInUse_;
    }

    /// @brief High-water mark of simultaneously-live slots. Telemetry only.
    std::size_t LbStore::peakSlotsInUse() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return peakSlotsInUse_;
    }

    /// @brief Number of pool blocks the store has drawn. Telemetry only.
    std::size_t LbStore::blocksHeld() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return blocks_.size();
    }

    /// @brief Return every drawn block to the pool and reset to empty.
    ///
    /// @details
    /// See the declaration. Mirrors `LbArena::releaseAll`: returns each block via
    /// `releaseBlock`, clears the block + free lists, and zeroes the live count.
    /// Cached slot geometry is kept so a store may be reused after a release.
    void LbStore::releaseAll() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (char* block : blocks_) {
            manager_->releaseBlock(block);
        }
        blocks_.clear();
        freeHead_ = nullptr;
        currentBlock_ = nullptr;
        currentCursor_ = 0;
        slotsInUse_ = 0;
    }

    /// @brief Return every drawn block to the pool (mirrors `~LbArena`).
    LbStore::~LbStore() {
        releaseAll();
    }

}
