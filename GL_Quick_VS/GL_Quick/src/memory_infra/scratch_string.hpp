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

#include "scratch_arena.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>

namespace gl {

    /// @brief Length-carrying view of one transient calculation string
    ///        living in a `ScratchArena` — the ScratchString class of the string
    ///        statification model.
    ///
    /// @details
    /// A ScratchString is `char* + length` plus the liveness stamps that make
    /// accidental escape ASSERT instead of silently reading dead memory:
    /// the arena generation at birth (a wholesale `releaseAll()` / `reset()`
    /// bumps it — any later access through the view asserts) and the page-tier
    /// position at birth (a `rewind()` past the string's position drops the
    /// used count below it —
    /// any later access asserts). Both checks run on every accessor; with
    /// the poison fill behind them, an escape is caught at the access site,
    /// stack intact.
    ///
    /// Content is immutable after creation: strings are allocated at birth
    /// with their exact length and filled once (`copyFrom`, or a string
    /// operation that computes its output length first). There is no
    /// elongation and no mutation API.
    ///
    /// A ScratchString must never outlive its scratch scope. The ONLY
    /// sanctioned escapes are `toStdString()` (boundary sites: file IO, debug
    /// dumps, export) and the explicit copy into a ColdString table.
    /// Persistent containers must not accept ScratchString —
    /// there is deliberately no conversion to anything storable.
    ///
    /// Detection limit (deliberate — no per-allocation metadata): a view
    /// that survives a rewind below its birth asserts while its span is
    /// still poisoned, but once later allocations refill past its birth
    /// position the cursor check can no longer tell it from a live view.
    /// The CONTRACT is therefore stack discipline — a view never crosses a
    /// rewind mark below its birth; whatever must survive the window is
    /// copied out before the window closes. The checks are the tripwire,
    /// not the license.
    ///
    /// @invariant `data()` is readable for exactly `size()` bytes while the
    ///            view is live (same generation, cursor not rewound past
    ///            birth).
    /// @invariant An empty ScratchString (length 0) carries no pointer and no
    ///            liveness coupling — it is always valid.
    /// @see `ScratchArena`, `ScratchScope` (the RAII rewind guard),
    ///      `kArenaPoisonByte`.
    class ScratchString {
    public:
        /// @brief The empty string: no bytes, no arena coupling, always
        ///        live.
        ScratchString() = default;

        /// @brief Allocate `len` bytes in `arena` and fill them from
        ///        `src` — the primitive ScratchString constructor.
        ///
        /// @details
        /// Exact-length allocation at the bump head plus one `memcpy`.
        /// `len == 0` yields the empty string without touching the arena
        /// (an empty operand is a defined result, not a failure).
        ///
        /// @param arena Hot arena owning the new bytes.
        /// @param src   Source bytes; readable for `len` bytes; may be
        ///              cold, heap, or hot content.
        /// @param len   Exact byte count; >= 0.
        /// @return View of the freshly filled allocation.
        static ScratchString copyFrom(ScratchArena& arena, const char* src,
                                  int32_t len) {
            assert(len >= 0);
            if (len == 0) return ScratchString();
            assert(src != nullptr);
            ScratchString out;
            out.ptr_ = arena.allocBytes(len);
            std::memcpy(out.ptr_, src, static_cast<size_t>(len));
            out.len_ = len;
            out.arena_ = &arena;
            out.generation_ = arena.generation();
            out.birthEnd_ = arena.usedBytes();
            return out;
        }

        /// @brief Wrap bytes already allocated and filled in `arena` (the
        ///        fill-then-wrap pattern used by the string operations that
        ///        compute into a fresh allocation).
        ///
        /// @details
        /// The caller guarantees `ptr` is the start of a live allocation of
        /// exactly `len` bytes in `arena`'s current generation — typically
        /// the pointer returned by `arena.allocBytes(len)` moments earlier.
        ///
        /// @param arena Hot arena owning `ptr`.
        /// @param ptr   Start of the filled allocation; non-null.
        /// @param len   Exact byte count; > 0 (use the default constructor
        ///              for empties — a wrapped empty has no allocation to
        ///              wrap).
        /// @return View of the existing allocation.
        static ScratchString wrap(ScratchArena& arena, char* ptr, int32_t len) {
            assert(ptr != nullptr);
            assert(len > 0);
            ScratchString out;
            out.ptr_ = ptr;
            out.len_ = len;
            out.arena_ = &arena;
            out.generation_ = arena.generation();
            out.birthEnd_ = arena.usedBytes();
            return out;
        }

        /// @brief Read access to the bytes; asserts the view is live.
        ///
        /// @return Pointer to `size()` readable bytes. Asserts a non-empty,
        ///         live view (generation match, cursor not rewound past
        ///         birth).
        const char* data() const {
            assert(len_ > 0 && "ScratchString::data on the empty string");
            assertLive();
            return ptr_;
        }

        /// @brief Byte count fixed at birth.
        ///
        /// @return Length in bytes; 0 for the empty string. Asserts
        ///         liveness on non-empty views — a stale view's length is
        ///         as dead as its bytes.
        int32_t size() const {
            if (len_ > 0) assertLive();
            return len_;
        }

        /// @brief Whether this is the empty string.
        ///
        /// @return `true` when the length is 0.
        bool empty() const { return len_ == 0; }

        /// @brief Single-byte read with bounds check; asserts liveness.
        ///
        /// @param i Byte index, `0 <= i < size()`.
        /// @return The byte at `i`.
        char operator[](int32_t i) const {
            assert(i >= 0 && i < len_);
            assertLive();
            return ptr_[i];
        }

        /// @brief Sanctioned escape to the std::string boundary (file IO,
        ///        debug dumps, proof export, user-facing rendering).
        ///
        /// @return Owned heap copy of the bytes. Asserts liveness first —
        ///         escaping a dead view is the exact bug this type exists
        ///         to catch.
        std::string toStdString() const {
            if (len_ == 0) return std::string();
            assertLive();
            return std::string(ptr_, static_cast<size_t>(len_));
        }

    private:
        /// @brief The escape assert: the owning arena must be in the
        ///        view's birth generation and not rewound past its birth
        ///        position.
        void assertLive() const {
            assert(arena_ != nullptr);
            assert(arena_->generation() == generation_
                && "ScratchString outlived its scratch scope (arena was "
                   "released) — copy to cold or toStdString BEFORE it ends");
            assert(arena_->usedBytes() >= birthEnd_
                && "ScratchString outlived its rewind window (arena was rewound "
                   "past it) — copy out before the mark's rewind");
        }

        char* ptr_ = nullptr;
        int32_t len_ = 0;
        const ScratchArena* arena_ = nullptr;
        uint64_t generation_ = 0;
        int64_t birthEnd_ = 0;         // page-tier used-bytes right after birth
    };

    /// @brief RAII rewind guard: marks the arena at construction, rewinds
    ///        to that mark at destruction — the per-call scratch window.
    ///
    /// @details
    /// The tool for bounded scopes inside a long-running executor (the
    /// fixpoint loop runs the firing check per request; without a per-call
    /// window the arena would grow for the whole burst). Every ScratchString
    /// allocated inside the scope dies at the closing brace; the freed span
    /// is poisoned by `ScratchArena::rewind`.
    ///
    /// Deliberately not nestable across interleaved lifetimes — scopes
    /// close in strict reverse order of opening (stack discipline), which
    /// `ScratchArena::rewind`'s backwards-only assert enforces at run time.
    ///
    /// @invariant The arena's cursor at destruction time is never before
    ///            the construction-time mark (enforced by the rewind
    ///            assert).
    /// @see `ScratchArena::mark` / `ScratchArena::rewind`, `ScratchString`.
    class ScratchScope {
    public:
        /// @brief Open the window: capture the current arena position.
        ///
        /// @param arena Arena this scope guards.
        explicit ScratchScope(ScratchArena& arena)
            : arena_(arena), mark_(arena.mark()) {}

        /// @brief Close the window: rewind to the construction-time mark,
        ///        poisoning everything allocated inside.
        ~ScratchScope() { arena_.rewind(mark_); }

        ScratchScope(const ScratchScope&) = delete;
        ScratchScope& operator=(const ScratchScope&) = delete;
        ScratchScope(ScratchScope&&) = delete;
        ScratchScope& operator=(ScratchScope&&) = delete;

    private:
        ScratchArena& arena_;
        ScratchArena::Mark mark_;
    };

}
