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

#include "global_memory_manager.hpp"
#include "lb_arena.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>

namespace gl {

    /// @brief Intrusive link header of one record on a `SealedPageSet`'s
    ///        record chain — a single forward pointer, nothing else.
    ///
    /// @details
    /// Each `appendRecord` call allocates one node header plus the record
    /// payload in a single page-set allocation; the node links append order
    /// head → tail. The node is type-erased (the payload type is a template
    /// parameter of the accessors, not stored), so one chain member set on
    /// `SealedPageSet` serves every record type — the single-type-per-set
    /// contract is enforced by a `sizeof` stride assert at append time.
    ///
    /// @invariant A node's `next` is written exactly twice: `nullptr` at its
    ///            own append, then once more when the following record links
    ///            behind it. Existing nodes are never otherwise mutated, which
    ///            is what lets `SealedRecordCursor` resume across appends.
    /// @see `SealedPageSet::appendRecord`, `SealedRecordCursor`.
    struct SealedRecordNode {
        SealedRecordNode* next;
    };

    template <typename T>
    class SealedRecordCursor;

    /// @brief Write-once page set for staging-record strings — the
    ///        ownership-handoff piece of the string statification model.
    ///
    /// @details
    /// Firing records are the burst's OUTPUT: produced by request-pool
    /// tasks, merged and drained by a different pool after the producing
    /// threads are gone. Their strings can live in neither a worker slot's
    /// scratch arena (reset per executor; one slot serves many LBs per phase)
    /// nor a ColdString table (minting is single-threaded; the records are
    /// born in the parallel phase). A `SealedPageSet` solves the lifetime
    /// by moving the DELETION POINT instead of the bytes: the task bump-
    /// fills pages it owns exclusively, seals them at task end, the set
    /// travels with the task's records, and the consumer frees the pages
    /// right after the drain mints the cold ids — scratch pages die with
    /// the thread, sealed output pages die with consumption.
    ///
    /// Lifecycle is a three-state machine asserted on every operation:
    /// `Filling` (alloc allowed; the owning task's exclusive window) →
    /// `Sealed` (read-only; crosses the pool join with the records) →
    /// `Freed` (pages poisoned and returned; views assert). `seal()` on an
    /// empty set and `freePages()` on a never-used set are defined no-ops
    /// of the same transitions — an empty task output is a result, not a
    /// failure.
    ///
    /// Deliberately non-movable: `SealedString` views carry a pointer to
    /// their owning set, so sets live at stable addresses — the wiring
    /// pre-sizes one slot per task (a container that never reallocates
    /// while tasks run), giving lock-free exclusive ownership during the
    /// parallel phase.
    ///
    /// Blocks come from the cold grant path (`acquireBlock`) on the set's
    /// own per-task `LbArena`; the pool's exhaustion assert is the sizing
    /// backstop (staging volume is work-content-dependent; an artificial
    /// per-set cap would be tuning disguised as safety). The set is freed
    /// the instant the drain consumes it — before any deload seam — so it is
    /// never deload-registered.
    ///
    /// Beyond raw string/span storage the set carries a RECORD CHAIN: the
    /// staging records themselves (`FiringRecord` and kin) live on the same
    /// pages as the strings and spans they reference, linked by intrusive
    /// `SealedRecordNode` headers in append order (`appendRecord` /
    /// `forEachRecord` / `SealedRecordCursor`). Putting the spine on the
    /// payload pages means spine lifetime == payload lifetime with zero
    /// extra code: `freePages` kills nodes and payload in one sweep, which
    /// also covers the redo-discard path that drops an LB's records unread.
    /// Record allocations interleave freely with string/span allocations —
    /// the chain preserves append order regardless, and an order-sensitive
    /// consumer builds a sorted permutation over the stable payload
    /// addresses.
    ///
    /// @invariant Bytes are written only in `Filling`, read in `Filling` /
    ///            `Sealed`, and unreachable in `Freed` (views assert; the
    ///            span is poisoned before the blocks return).
    /// @invariant A destructed set holds no blocks — consumption freed
    ///            them (`Freed`) or the set was never used.
    /// @invariant Record-chain append order is the `forEachRecord` /
    ///            `SealedRecordCursor` iteration order; payload addresses
    ///            are stable until `freePages`; one record type per set.
    /// @see `SealedString`, `SealedSpan`, `SealedRecordCursor`, the
    ///      per-worker scratch arena (the scratch sibling),
    ///      `GlobalMemoryManager::acquireBlock`.
    class SealedPageSet {
    public:
        SealedPageSet() = default;

        /// @brief Asserts the lifecycle completed: no blocks outstanding
        ///        (freed after consumption, or never used).
        ~SealedPageSet();

        SealedPageSet(const SealedPageSet&) = delete;
        SealedPageSet& operator=(const SealedPageSet&) = delete;
        SealedPageSet(SealedPageSet&&) = delete;
        SealedPageSet& operator=(SealedPageSet&&) = delete;

        /// @brief Bind the set to a pool. Separate from construction so
        ///        pre-sized slot containers can default-construct.
        ///
        /// @details
        /// Idempotent for the same pool (the wiring may bind lazily at
        /// first use); asserts against rebinding to a different pool.
        ///
        /// @param global Pool to draw blocks from (hot grant path).
        void bind(GlobalMemoryManager* global) {
            assert(global != nullptr && global->initialized());
            assert((global_ == nullptr || global_ == global)
                && "SealedPageSet rebind to a different pool");
            if (global_ == nullptr)
                bump_.bind(global);  // cold per-task page arena
            global_ = global;
        }

        /// @brief Exact-length allocation at the bump head; `Filling`
        ///        only.
        ///
        /// @details
        /// Same exact-length, never-straddles-a-page discipline as the hot
        /// scratch arena's `allocBytes`. The first allocation acquires the
        /// first page.
        ///
        /// @param bytes Exact length of the new allocation; > 0, at most
        ///              one page.
        /// @return Pointer to `bytes` writable bytes, stable until
        ///         `freePages`.
        char* alloc(int32_t bytes);

        /// @brief End of the task's write window: `Filling` → `Sealed`.
        ///        From here the set is read-only and crosses the pool
        ///        join with the task's records.
        void seal() {
            assert(state_ == State::Filling
                && "SealedPageSet::seal outside the Filling window");
            state_ = State::Sealed;
        }

        /// @brief Consumption complete: poison every filled page, return
        ///        every block, `Sealed` → `Freed`.
        ///
        /// @details
        /// Called by the consumer (the drain side) immediately after the
        /// records referencing this set are minted into their persistent
        /// id form. Poisoning before release makes a straggler view read
        /// loud garbage even if the block is re-granted later.
        void freePages();

        /// @brief Lifecycle observers (telemetry and assert support).
        ///
        /// @return Whether the set currently accepts writes.
        bool filling() const { return state_ == State::Filling; }

        /// @brief Whether the set is sealed (read-only, pre-consumption).
        ///
        /// @return `true` between `seal()` and `freePages()`.
        bool sealed() const { return state_ == State::Sealed; }

        /// @brief Whether the set's pages were returned.
        ///
        /// @return `true` after `freePages()`.
        bool freed() const { return state_ == State::Freed; }

        /// @brief Blocks currently held (telemetry).
        ///
        /// @return Retained block count.
        int32_t blocksHeld() const {
            return static_cast<int32_t>(bump_.blocksHeld());
        }

        /// @brief Total bytes allocated into the set (telemetry; includes
        ///        no block-tail waste — exact sum of allocation lengths).
        ///
        /// @return Sum of all `alloc` lengths.
        int64_t bytesAllocated() const { return bytesAllocated_; }

        /// @brief Append one staging record to the set's record chain;
        ///        `Filling` only.
        ///
        /// @details
        /// One page-set allocation carries the intrusive `SealedRecordNode`
        /// header plus the memcpy-filled payload: the node is aligned up
        /// first, the payload to `alignof(T)` after it (the `SealedSpan`
        /// align-up idiom, applied twice), so the byte-granular page storage
        /// hands back a real aligned `T`. The chain is single-type per set —
        /// the type-erased nodes cannot check `T` at runtime, so a
        /// `sizeof(T)` stride recorded at the first append is asserted on
        /// every later one (the realistic-mistake tripwire). The whole
        /// node+payload allocation must fit one arena block; the explicit
        /// fits-assert mirrors `LbArena::alloc`'s own `bytes <= blockBytes`
        /// bound and names the sizing knob.
        ///
        /// @param rec Record to copy in; `T` must be trivially copyable
        ///            (memcpy-filled, never constructed).
        /// @return Reference to the stored payload copy.
        /// @invariant The returned reference (and its address) is stable
        ///            until `freePages` — `forEachRecord`, the cursor, and
        ///            any pointer-permutation sort over the records rely on
        ///            this address stability.
        /// @invariant Append order is the chain's frozen iteration order.
        /// @see `forEachRecord`, `SealedRecordCursor`, `SealedSpan::copyFrom`.
        template <typename T>
        const T& appendRecord(const T& rec) {
            static_assert(std::is_trivially_copyable<T>::value,
                "SealedPageSet::appendRecord requires a trivially copyable "
                "record type — the payload is memcpy-filled, never "
                "constructed");
            assert(state_ == State::Filling
                && "SealedPageSet::appendRecord outside the Filling window");
            if (recordCount_ == 0) {
                recordStride_ = static_cast<int32_t>(sizeof(T));
            } else {
                assert(recordStride_ == static_cast<int32_t>(sizeof(T))
                    && "SealedPageSet record chain is single-type — a "
                       "second record type was appended to the same set");
            }
            constexpr int32_t kNodeAlign =
                static_cast<int32_t>(alignof(SealedRecordNode));
            constexpr int32_t kPayloadAlign =
                static_cast<int32_t>(alignof(T));
            const int32_t bytes =
                (kNodeAlign - 1)
                + static_cast<int32_t>(sizeof(SealedRecordNode))
                + (kPayloadAlign - 1)
                + static_cast<int32_t>(sizeof(T));
            assert(global_ != nullptr
                && "SealedPageSet::appendRecord before bind");
            assert(bytes <= bump_.blockBytes()
                && "record + chain node exceed one arena block — raise "
                   "static_block_bytes");
            char* raw = alloc(bytes);
            const std::uintptr_t rawAddr =
                reinterpret_cast<std::uintptr_t>(raw);
            const std::uintptr_t nodeAddr =
                (rawAddr + static_cast<std::uintptr_t>(kNodeAlign - 1))
                & ~static_cast<std::uintptr_t>(kNodeAlign - 1);
            SealedRecordNode* node =
                reinterpret_cast<SealedRecordNode*>(nodeAddr);
            node->next = nullptr;
            T* payload = const_cast<T*>(recordPayload<T>(node));
            std::memcpy(payload, &rec, sizeof(T));
            if (recordTail_ == nullptr) {
                recordHead_ = node;
            } else {
                static_cast<SealedRecordNode*>(recordTail_)->next = node;
            }
            recordTail_ = node;
            ++recordCount_;
            return *payload;
        }

        /// @brief Number of records appended to the chain so far.
        ///
        /// @details
        /// Legal in `Filling` and `Sealed` (symmetric with the view asserts);
        /// asserts on a `Freed` set, whose chain is gone with the pages.
        ///
        /// @return Record count; 0 for a chain-free set.
        int32_t recordCount() const {
            assert(!freed()
                && "SealedPageSet::recordCount on a freed set — the chain "
                   "died with the pages");
            return recordCount_;
        }

        /// @brief Walk every record head → tail in append order, invoking
        ///        `fn(const T&)` on each payload.
        ///
        /// @details
        /// Append order is the frozen iteration contract — an
        /// order-sensitive consumer may build a sorted pointer permutation
        /// over the payloads, but the raw walk itself is always the append
        /// sequence. Legal in `Filling` and `Sealed`; asserts on `Freed`.
        /// The functor is a template parameter (never `std::function`) so
        /// the walk allocates nothing.
        ///
        /// @param fn Functor invoked as `fn(const T&)` per record.
        /// @invariant The stride assert catches a `T` mismatch against the
        ///            appended record type whenever the chain is non-empty.
        /// @see `appendRecord`, `SealedRecordCursor`.
        template <typename T, typename F>
        void forEachRecord(F&& fn) const {
            static_assert(std::is_trivially_copyable<T>::value,
                "SealedPageSet::forEachRecord requires the trivially "
                "copyable record type the chain was filled with");
            assert(state_ != State::Freed
                && "SealedPageSet::forEachRecord on a freed set — read "
                   "records BEFORE freePages()");
            assert((recordCount_ == 0
                    || recordStride_ == static_cast<int32_t>(sizeof(T)))
                && "SealedPageSet::forEachRecord type mismatch against the "
                   "appended record type");
            for (const SealedRecordNode* n =
                     static_cast<const SealedRecordNode*>(recordHead_);
                 n != nullptr; n = n->next) {
                fn(*recordPayload<T>(n));
            }
        }

    private:
        friend class SealedString;
        template <typename T>
        friend class SealedRecordCursor;

        enum class State : uint8_t { Filling, Sealed, Freed };

        /// @brief Address of the payload stored behind a chain node: the
        ///        first `alignof(T)`-aligned address after the node header.
        ///
        /// @details
        /// Pure address arithmetic shared by `appendRecord` (write side),
        /// `forEachRecord`, and `SealedRecordCursor` (read side) so the
        /// three can never disagree on the payload location.
        ///
        /// @param node Chain node the payload sits behind.
        /// @return Typed pointer to the payload.
        template <typename T>
        static const T* recordPayload(const SealedRecordNode* node) {
            const std::uintptr_t after =
                reinterpret_cast<std::uintptr_t>(node)
                + sizeof(SealedRecordNode);
            const std::uintptr_t aligned =
                (after + static_cast<std::uintptr_t>(alignof(T) - 1))
                & ~static_cast<std::uintptr_t>(alignof(T) - 1);
            return reinterpret_cast<const T*>(aligned);
        }

        GlobalMemoryManager* global_ = nullptr;
        State state_ = State::Filling;
        LbArena bump_;                 // cold per-task page arena — bump storage
        int64_t bytesAllocated_ = 0;
        void*   recordHead_ = nullptr;   // first SealedRecordNode, or null
        void*   recordTail_ = nullptr;   // last SealedRecordNode, or null
        int32_t recordCount_ = 0;
        int32_t recordStride_ = 0;       // sizeof(T) of the first append
    };

    /// @brief Incremental reader of a `SealedPageSet`'s record chain —
    ///        hands out each record exactly once, in append order, resuming
    ///        correctly across later appends.
    ///
    /// @details
    /// The mid-burst consumer shape: a producer that appends records and,
    /// between appends, reads back exactly the records it has not yet
    /// visited (the `BurstSink::consume` early-exit scan). A whole-chain
    /// walk per read-back would be O(n²) across a burst; the cursor is
    /// O(1) amortized — it remembers the last node handed out and resumes
    /// at its `next`. Resumption across appends is sound because appending
    /// never mutates existing nodes, only the old tail's `next`
    /// (see `SealedRecordNode`).
    ///
    /// Legal in `Filling` (producer and reader are the same task/thread)
    /// and `Sealed`; `next()` asserts on a `Freed` set.
    ///
    /// @invariant Each record is handed out at most once, in append order.
    /// @see `SealedPageSet::appendRecord`, `SealedPageSet::forEachRecord`.
    template <typename T>
    class SealedRecordCursor {
    public:
        static_assert(std::is_trivially_copyable<T>::value,
            "SealedRecordCursor<T> requires the trivially copyable record "
            "type the chain was filled with");

        /// @brief Bind the cursor to a set; starts at the chain head.
        ///
        /// @param set Page set whose record chain to drain; outlives the
        ///            cursor.
        explicit SealedRecordCursor(const SealedPageSet& set) : set_(&set) {}

        /// @brief The next unvisited record in append order, or `nullptr`
        ///        when caught up.
        ///
        /// @details
        /// The `nullptr` return is a DEFINED result — the caught-up shape,
        /// exactly like a cache miss — not a failure signal: the caller may
        /// append more records and call `next()` again, and the cursor
        /// resumes where it left off. Misuse (a freed set, a record-type
        /// mismatch) asserts instead.
        ///
        /// @return Pointer to the next payload (stable until `freePages`),
        ///         or `nullptr` when every appended record has been visited.
        const T* next() {
            assert(!set_->freed()
                && "SealedRecordCursor::next on a freed set — drain records "
                   "BEFORE freePages()");
            assert((set_->recordCount_ == 0
                    || set_->recordStride_
                           == static_cast<int32_t>(sizeof(T)))
                && "SealedRecordCursor type mismatch against the appended "
                   "record type");
            const SealedRecordNode* start = lastVisited_
                ? static_cast<const SealedRecordNode*>(lastVisited_)->next
                : static_cast<const SealedRecordNode*>(set_->recordHead_);
            if (start == nullptr) return nullptr;   // caught up — defined
            lastVisited_ = start;
            return SealedPageSet::recordPayload<T>(start);
        }

    private:
        const SealedPageSet* set_;
        const void* lastVisited_ = nullptr;  // last node handed out
    };

    /// @brief Length-carrying view of one string on a `SealedPageSet` —
    ///        the staging-record string representation.
    ///
    /// @details
    /// Like `ScratchString` a non-owning `char* + length` view, but with the
    /// handoff lifecycle instead of scope lifecycle: valid through
    /// `Filling` and `Sealed`, asserting after `freePages` (the owning
    /// set's address is stable by the non-movable + pre-sized-slot
    /// contract, so the owner pointer stays good for the view's whole
    /// life). The empty string carries no owner and is always valid.
    ///
    /// Records hold these views by value across the pool join; the drain
    /// reads them, mints cold ids, then frees the set — after which any
    /// straggler access asserts.
    ///
    /// @invariant `data()` is readable for exactly `size()` bytes until
    ///            the owning set is freed.
    /// @see `SealedPageSet`, `ScratchString` (the scratch-scope sibling).
    class SealedString {
    public:
        /// @brief The empty string: no bytes, no owner, always valid.
        SealedString() = default;

        /// @brief Allocate `len` bytes in `set` and fill them from `src`.
        ///
        /// @details
        /// The one constructor of non-empty sealed strings: exact-length
        /// allocation plus one `memcpy`, inside the task's `Filling`
        /// window. `len == 0` yields the empty string without touching
        /// the set.
        ///
        /// @param set Page set owned by the calling task; `Filling`.
        /// @param src Source bytes; readable for `len` bytes.
        /// @param len Exact byte count; >= 0.
        /// @return View of the freshly filled allocation.
        static SealedString copyFrom(SealedPageSet& set, const char* src,
                                     int32_t len) {
            assert(len >= 0);
            if (len == 0) return SealedString();
            assert(src != nullptr);
            SealedString out;
            char* dst = set.alloc(len);
            std::memcpy(dst, src, static_cast<size_t>(len));
            out.ptr_ = dst;
            out.len_ = len;
            out.owner_ = &set;
            return out;
        }

        /// @brief Read access to the bytes; asserts the owning set is not
        ///        freed.
        ///
        /// @return Pointer to `size()` readable bytes.
        const char* data() const {
            assert(len_ > 0 && "SealedString::data on the empty string");
            assertLive();
            return ptr_;
        }

        /// @brief Byte count fixed at birth.
        ///
        /// @return Length in bytes; 0 for the empty string. Asserts
        ///         liveness on non-empty views.
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

        /// @brief Sanctioned escape to the std::string boundary (debug
        ///        dumps and transitional call sites during the migration).
        ///
        /// @return Owned heap copy of the bytes.
        std::string toStdString() const {
            if (len_ == 0) return std::string();
            assertLive();
            return std::string(ptr_, static_cast<size_t>(len_));
        }

    private:
        /// @brief The escape assert: the owning set must not be freed.
        void assertLive() const {
            assert(owner_ != nullptr);
            assert(!owner_->freed()
                && "SealedString outlived its page set — the drain freed "
                   "the pages; consume records BEFORE freePages()");
        }

        const char* ptr_ = nullptr;
        int32_t len_ = 0;
        const SealedPageSet* owner_ = nullptr;
    };

    /// @brief Length-carrying view of a contiguous, immutable run of POD
    ///        elements on a `SealedPageSet` — the array analogue of
    ///        `SealedString`.
    ///
    /// @details
    /// The staging records (`FiringRecord` and the admission / integration
    /// staging structs) carry small element runs across the parallel →
    /// single-threaded pool join — a firing's level set, its origin
    /// dependencies, a marker's sorted argument list. Before the
    /// transient-statification campaign those runs were `std::set<int>` /
    /// `std::vector<...>` spines on the malloc heap, one allocation per firing.
    /// `SealedSpan<T>` moves them onto the same per-task page set that already
    /// holds the record's strings: the producing task seals the finished run
    /// with `copyFrom`, the record holds a non-owning view, and the run dies
    /// with the set at the post-drain free — exactly the `SealedString`
    /// lifecycle, one rank up from raw bytes to typed elements.
    ///
    /// `T` must be trivially copyable: a run is filled by one `memcpy` and
    /// never constructs or destructs an element. The page tier hands out
    /// byte-granular, unaligned storage, so `copyFrom` over-allocates by
    /// `alignof(T) - 1` and aligns the run base up — the view is then a real
    /// aligned `const T*`, indexable and iterable like any array. The whole
    /// run must fit one page (the `SealedString` one-page limit, asserted in
    /// `SealedPageSet::alloc`); the staging runs are single digits of elements,
    /// far inside a page.
    ///
    /// @invariant `data()` is readable for exactly `size()` elements until the
    ///            owning set is freed; `operator[]` / iteration assert the set
    ///            is not `Freed` (the escaped-view tripwire).
    /// @see `SealedString` (the byte sibling), `SealedPageSet`.
    template <typename T>
    class SealedSpan {
    public:
        static_assert(std::is_trivially_copyable<T>::value,
            "SealedSpan<T> requires a trivially copyable element type — the "
            "run is memcpy-filled and never constructs an element");

        /// @brief The empty span: no elements, no owner, always valid.
        SealedSpan() = default;

        /// @brief Allocate room for `count` elements in `set` and fill them
        ///        from `src`.
        ///
        /// @details
        /// One aligned allocation plus one `memcpy`, inside the task's
        /// `Filling` window. `count == 0` yields the empty span without
        /// touching the set. The allocation reserves `alignof(T) - 1` extra
        /// bytes so the run base can be aligned up to `alignof(T)` within the
        /// byte-granular page storage; the aligned run stays inside the one
        /// page `SealedPageSet::alloc` guarantees.
        ///
        /// @param set   Page set owned by the calling task; `Filling`.
        /// @param src   Source elements; readable for `count` elements.
        /// @param count Exact element count; >= 0.
        /// @return View of the freshly filled run.
        static SealedSpan copyFrom(SealedPageSet& set, const T* src,
                                   int32_t count) {
            assert(count >= 0);
            if (count == 0) return SealedSpan();
            assert(src != nullptr);
            constexpr int32_t kAlign = static_cast<int32_t>(alignof(T));
            const int32_t bytes =
                count * static_cast<int32_t>(sizeof(T)) + (kAlign - 1);
            char* raw = set.alloc(bytes);
            const std::uintptr_t rawAddr =
                reinterpret_cast<std::uintptr_t>(raw);
            const std::uintptr_t alignedAddr =
                (rawAddr + static_cast<std::uintptr_t>(kAlign - 1))
                & ~static_cast<std::uintptr_t>(kAlign - 1);
            T* dst = reinterpret_cast<T*>(alignedAddr);
            std::memcpy(dst, src, static_cast<size_t>(count) * sizeof(T));
            SealedSpan out;
            out.ptr_ = dst;
            out.len_ = count;
            out.owner_ = &set;
            return out;
        }

        /// @brief Read access to the run; asserts the owning set is not freed.
        /// @return Pointer to `size()` readable elements.
        const T* data() const {
            assert(len_ > 0 && "SealedSpan::data on the empty span");
            assertLive();
            return ptr_;
        }

        /// @brief Element count fixed at birth.
        /// @return Number of elements; 0 for the empty span. Asserts liveness
        ///         on non-empty views.
        int32_t size() const {
            if (len_ > 0) assertLive();
            return len_;
        }

        /// @brief Whether this is the empty span.
        /// @return `true` when the length is 0.
        bool empty() const { return len_ == 0; }

        /// @brief Indexed read with bounds check; asserts liveness.
        /// @param i Element index, `0 <= i < size()`.
        /// @return Const reference to element `i`.
        const T& operator[](int32_t i) const {
            assert(i >= 0 && i < len_);
            assertLive();
            return ptr_[i];
        }

        /// @brief Range-for begin; the empty span yields an empty range.
        /// @return Pointer to the first element, or nullptr when empty.
        const T* begin() const {
            if (len_ == 0) return nullptr;
            assertLive();
            return ptr_;
        }

        /// @brief Range-for end.
        /// @return One past the last element, or nullptr when empty.
        const T* end() const {
            if (len_ == 0) return nullptr;
            assertLive();
            return ptr_ + len_;
        }

    private:
        /// @brief The escape assert: the owning set must not be freed.
        void assertLive() const {
            assert(owner_ != nullptr);
            assert(!owner_->freed()
                && "SealedSpan outlived its page set — the drain freed the "
                   "pages; consume records BEFORE freePages()");
        }

        const T* ptr_ = nullptr;
        int32_t len_ = 0;
        const SealedPageSet* owner_ = nullptr;
    };

}
