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

#include "lb_arena.hpp"

#include <cassert>
#include <cstdint>
#include <type_traits>

namespace gl {

    /// @brief A LIFO stack of trivially-copyable `T` on an `LbArena`'s byte-bump
    ///        tier, with `popTo` reclaim on every `pop` — the grow/decline
    ///        "stump" a depth-first frontier wants.
    ///
    /// @details
    /// A bump arena gives last-in-first-out reclaim through `popTo`, but a plain
    /// fixed-stride array over it cannot survive the no-straddle padding `alloc`
    /// inserts at block boundaries (consecutive frames are not at a constant
    /// stride). `ArenaStack` instead threads each element's frame to the previous
    /// top by virtual offset: a `push` bump-allocates a `{prevTop, value}` frame
    /// and links it; a `pop` reads the frame, restores `prevTop` as the new top,
    /// and `popTo`s the just-removed frame — which is always the arena's cursor
    /// tail, so the reclaim is exact and the next `push` reuses the bytes.
    ///
    /// The footprint therefore tracks the LIVE frontier, not the total number of
    /// pushes: a deep search that grows and backtracks repeatedly holds only its
    /// current depth's worth of frames, and every pool block the frames pinned is
    /// returned when the owning arena is `releaseAll`-ed per worker task. The
    /// stack owns no storage of its own — all frames live in the supplied arena.
    /// It is single-threaded like its arena (one worker slot at a time).
    ///
    /// @tparam T The element type; trivially copyable (frames are raw arena
    ///         bytes, never constructed or destructed).
    /// @invariant Frames form a `popTo`-clean LIFO chain: the top frame is always
    ///            the arena's byte-bump cursor tail, so `pop`'s `popTo` reclaims
    ///            exactly it. This holds only while nothing else byte-bumps the
    ///            arena for the stack's lifetime (the arena's page tier may be
    ///            used freely by other containers — it is independent state).
    /// @invariant No element escapes as a pointer into the arena — `back` returns
    ///            a copy (the I-84 copy-before-mutate discipline).
    /// @see `LbArena::alloc` / `popTo` (the byte-bump tier), `ScratchArena`.
    template <typename T>
    class ArenaStack {
        static_assert(std::is_trivially_copyable<T>::value,
            "ArenaStack<T> requires a trivially-copyable T (frames are raw arena "
            "bytes, never constructed or destructed)");

    public:
        /// @brief Bind an empty stack to `arena`'s byte-bump tier.
        ///
        /// @details
        /// Captures the arena by reference and allocates nothing until the first
        /// `push`. The arena must outlive the stack and must not be byte-bumped
        /// by anyone else for the stack's lifetime (the `popTo` LIFO is exact
        /// only while this stack owns the byte-bump tail; the arena's page tier
        /// is independent and may back other scratch containers).
        ///
        /// @param arena The owning arena; its byte-bump tier backs every frame.
        explicit ArenaStack(LbArena& arena)
            : arena_(&arena), top_(kNullOffset), size_(0) {}

        /// @brief Whether the stack holds no elements.
        ///
        /// @return `true` when `size() == 0`.
        bool empty() const { return size_ == 0; }

        /// @brief Number of elements currently on the stack.
        ///
        /// @return The live element count.
        int32_t size() const { return size_; }

        /// @brief Push a copy of `v` onto the top.
        ///
        /// @details
        /// Bump-allocates one `{prevTop, value}` frame, links it to the current
        /// top, and makes it the new top. Acquires a fresh pool block through the
        /// arena when the cursor crosses a block boundary.
        ///
        /// @param v The value copied onto the stack.
        void push(const T& v) {
            const ArenaOffset off = arena_->alloc(
                static_cast<int32_t>(sizeof(Frame)),
                static_cast<int32_t>(alignof(Frame)));
            Frame* f = reinterpret_cast<Frame*>(arena_->resolve(off));
            f->prevTop = top_;
            f->value = v;
            top_ = off;
            ++size_;
        }

        /// @brief Read a copy of the top element without removing it.
        ///
        /// @details
        /// Returns by value — the caller must not hold a pointer into the arena
        /// across any later `push` / `pop` (a `push` may relocate nothing but a
        /// later `pop` poisons the bytes). Asserts the stack is non-empty
        /// (Rule 19).
        ///
        /// @return A copy of the top element.
        T back() const {
            assert(size_ > 0 && "ArenaStack::back on an empty stack");
            const Frame* f =
                reinterpret_cast<const Frame*>(arena_->resolve(top_));
            return f->value;
        }

        /// @brief Remove the top element, reclaiming its frame.
        ///
        /// @details
        /// Restores the previous top from the frame's link, then `popTo`s the
        /// removed frame — the arena's cursor tail — so the byte-bump reclaim is
        /// exact and the next `push` reuses the bytes. Asserts the stack is
        /// non-empty (Rule 19).
        void pop() {
            assert(size_ > 0 && "ArenaStack::pop on an empty stack");
            const Frame* f =
                reinterpret_cast<const Frame*>(arena_->resolve(top_));
            const ArenaOffset toFree = top_;
            top_ = f->prevTop;
            --size_;
            arena_->popTo(toFree);
        }

    private:
        /// @brief One stacked element plus the virtual offset of the frame
        ///        beneath it (the LIFO link; `kNullOffset` under the bottom).
        struct Frame {
            ArenaOffset prevTop;
            T value;
        };

        LbArena* arena_;   // owning arena (byte-bump tier); not owned
        ArenaOffset top_;  // offset of the top frame, kNullOffset when empty
        int32_t size_;     // live element count
    };

}
