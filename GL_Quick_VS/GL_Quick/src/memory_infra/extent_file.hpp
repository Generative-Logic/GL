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

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <vector>

namespace gl {

    /// @brief One preallocated data file, held open for a whole batch, that
    ///        serves concurrent POSITIONED reads and writes at disjoint byte
    ///        ranges with no per-operation open/close and no I/O lock.
    ///
    /// @details
    /// This is the substrate for the v4 raw-eviction EXTENT FILE
    /// (`D-195` datapath): the whole batch's LB
    /// eviction images live in ONE file, each LB's image placed at a stable
    /// offset (its slab), and a re-dump is a single positioned overwrite into
    /// the already-open handle — eliminating the per-eviction NTFS
    /// `CreateFile`/`truncate`/`close` (and the per-file Windows Defender
    /// scan-on-close) that dominated the measured ~50 ms-per-image fixed cost.
    ///
    /// **Concurrency model.** The dumpers/loaders are the I/O executor pool
    /// plus the worker valve; they operate on DISTINCT LBs (arbitrated by the
    /// per-LB `stewardClaim` word), hence write/read DISJOINT byte ranges, so
    /// the positioned I/O needs no coordination.
    /// - On Windows each calling thread transparently gets its OWN independent
    ///   `CreateFileW` handle (its own OS file object, no shared file pointer,
    ///   no file-object serialization lock); positioned I/O is `WriteFile` /
    ///   `ReadFile` with a per-call `OVERLAPPED` carrying the offset — the
    ///   pwrite/pread equivalent on a synchronous handle (blocks until the
    ///   transfer completes AT the given offset).
    /// - On POSIX one shared `fd` serves every thread through `pwrite` /
    ///   `pread`, which POSIX guarantees do not touch the shared file offset
    ///   and are safe from concurrent threads at distinct offsets.
    ///
    /// The ONLY shared mutable state is the primary handle plus the file-size
    /// high-water, touched under `mutex_` on `open` / `close` / `preallocate`
    /// / `grow` and on a thread's FIRST I/O (to register its per-thread
    /// handle). It is NEVER touched on the byte-copy path of a stable-size
    /// re-dump, which is the whole point.
    ///
    /// **Failure policy (Rule 19).** Every I/O failure, short transfer, or
    /// growth failure asserts loudly naming the actionable knob — a failed
    /// state is a bug we want to see NOW, never a silent fallback. There is no
    /// artificial size ceiling; growth's only bound is the disk.
    ///
    /// @invariant Between `open` and `close` the file's logical size equals
    ///            `fileSize()`; every offset a caller reads was previously
    ///            written within `[0, fileSize())`.
    /// @see `D-195` (the datapath), `lbdeload`.
    class PositionedFile {
    public:
        /// @brief Chunk cap for one native transfer — the Windows `WriteFile`
        ///        / `ReadFile` byte count is a `DWORD`, so a larger request is
        ///        split into `<= kMaxIoChunkBytes` chunks. 1 GiB fits a DWORD
        ///        with room to spare; slabs are far smaller, so the loop is a
        ///        single pass in practice.
        static constexpr int64_t kMaxIoChunkBytes = 1LL << 30;

        /// @brief Construct a closed file. `open` must run before any I/O.
        PositionedFile() = default;

        /// @brief Close the file if still open (a defined teardown).
        ~PositionedFile();

        PositionedFile(const PositionedFile&) = delete;
        PositionedFile& operator=(const PositionedFile&) = delete;

        /// @brief Open (creating if absent) the data file and hold it open.
        ///
        /// @details
        /// Opens the primary handle with shared read/write access
        /// (`OPEN_ALWAYS` on Windows, `O_RDWR | O_CREAT` on POSIX). Does NOT
        /// size the file — the caller runs `preallocate` next. Bumps the open
        /// generation so any per-thread handle cached from a prior `open` of a
        /// reused object is transparently reopened on its next use. Asserts
        /// the file is not already open and the OS open succeeded.
        ///
        /// @param path The data-file path (its parent directory must exist).
        void open(const std::filesystem::path& path);

        /// @brief Close the primary handle and every per-thread handle.
        ///
        /// @details
        /// Closes all handles opened by any thread (OS handles are
        /// process-wide, so one thread may close another's) and bumps the open
        /// generation so stale thread-local cache entries are ignored. A no-op
        /// on an already-closed file. The batch owner calls this AFTER chapter
        /// export, because post-prove raw reloads of still-live LBs read the
        /// extent file.
        void close();

        /// @brief Whether the file is currently open.
        /// @return True between `open` and `close`.
        bool isOpen() const;

        /// @brief Set the file's logical size to `bytes` (initial sizing).
        ///
        /// @details
        /// Preallocation gives the OS one contiguous run and one scan at
        /// creation instead of a fault per eviction. The bytes are demand-zero
        /// (not physically written until first touched) which is acceptable
        /// because a slab is always fully written before it is read. Asserts
        /// the file is open and the resize succeeded (a failure names the
        /// preallocation knob). Under `mutex_`.
        ///
        /// @param bytes Target logical size in bytes; >= 0.
        void preallocate(int64_t bytes);

        /// @brief Extend the file's logical size to `newSize`.
        ///
        /// @details
        /// Called when a freshly allocated slab would exceed the current file
        /// size. Asserts `newSize >= fileSize()` (growth never shrinks) and
        /// that the resize succeeded — a failure is disk-full and asserts
        /// naming the actionable knobs (Rule 19), never a silent unbounded
        /// retry. Under `mutex_`.
        ///
        /// @param newSize Target logical size in bytes; >= current `fileSize()`.
        void grow(int64_t newSize);

        /// @brief Positioned write: place `len` bytes from `buf` at `offset`.
        ///
        /// @details
        /// Uses this thread's own handle (Windows) or the shared `fd`
        /// (POSIX). Blocks until the whole span is on the handle. Splits into
        /// `<= kMaxIoChunkBytes` chunks for the DWORD-count native API; each
        /// chunk asserts a FULL transfer — a short write is a surprise to
        /// surface, not to paper over (Rule 19). Asserts `offset >= 0`,
        /// `len >= 0`, and the file open.
        ///
        /// @param offset Byte offset in the file; >= 0.
        /// @param buf    Source bytes; at least `len` readable.
        /// @param len    Byte count; >= 0.
        void writeAt(int64_t offset, const void* buf, int64_t len);

        /// @brief Positioned read: fill `len` bytes into `buf` from `offset`.
        ///
        /// @details
        /// Symmetric to `writeAt`. A short read (EOF inside the requested span)
        /// asserts — the caller only ever reads bytes it previously wrote, so
        /// a short read is a corrupted or foreign file (Rule 19). Asserts
        /// `offset >= 0`, `len >= 0`, and the file open.
        ///
        /// @param offset Byte offset in the file; >= 0.
        /// @param buf    Destination buffer; at least `len` writable.
        /// @param len    Byte count; >= 0.
        void readAt(int64_t offset, void* buf, int64_t len);

        /// @brief The file's current logical size.
        /// @return The size set by the last `preallocate` / `grow`; 0 before
        ///         any sizing.
        int64_t fileSize() const;

        /// @brief The open file's path.
        /// @return The path passed to `open`.
        const std::filesystem::path& path() const { return path_; }

    private:
        /// @brief This thread's native handle for positioned I/O.
        ///
        /// @details
        /// On Windows, returns (opening and registering on first use) a
        /// per-thread `CreateFileW` handle so concurrent threads never share a
        /// file object. Registration (into `allHandles_` so `close` can reap
        /// it) runs under `mutex_` exactly once per thread per open-generation;
        /// the thread-local cache serves every subsequent call lock-free. On
        /// POSIX, returns the single shared `fd` (pread/pwrite are
        /// concurrency-safe at distinct offsets). Asserts the file is open.
        ///
        /// @return An OS handle valid for positioned I/O on this file.
#ifdef _WIN32
        void* threadHandle();
#else
        int threadHandle();
#endif

        std::filesystem::path path_;
        mutable std::mutex mutex_;
        int64_t fileSize_ = 0;
        // PROCESS-GLOBAL-unique generation stamped at each `open`. Keying the
        // per-thread handle cache by (this, generation) with a merely
        // per-object counter would COLLIDE when a new object reuses a freed
        // object's address at the same open-count (a closed handle would be
        // served — ERROR_INVALID_HANDLE); a global monotone id makes every
        // open distinct across the process.
        uint64_t openGeneration_ = 0;
        bool open_ = false;
#ifdef _WIN32
        void* primaryHandle_ = nullptr;      // INVALID_HANDLE_VALUE sentinel
        std::vector<void*> allHandles_;      // every per-thread handle, for close
#else
        int fd_ = -1;
#endif
    };

    /// @brief One slab allocation: the byte offset of the slab in the extent
    ///        file and the slab's full capacity (its size class).
    struct SlabAllocation {
        int64_t offset;      ///< Byte offset of the slab; block-aligned.
        int64_t classBytes;  ///< Slab capacity (a power-of-two block multiple).
    };

    /// @brief Power-of-two slab allocator over the extent file's byte space —
    ///        the ONLY shared mutable state of the raw-eviction datapath.
    ///
    /// @details
    /// Each LB owns exactly ONE slab for its active life; a stable-size
    /// re-dump overwrites the slab in place (no allocator interaction), and
    /// only a growth past the slab's class reallocates. Slabs are the smallest
    /// power-of-two multiple of `blockBytes` (256 KiB in production) that holds
    /// the LB's image, so an LB's ~6x growth over a batch is ~3 class
    /// promotions — a handful of allocator touches per LB across the whole run.
    ///
    /// **Why slabs over a best-fit in-file heap or an append log.** A best-fit
    /// heap degenerates into external fragmentation + its own compaction pass;
    /// an append log rewrites a fresh tail every eviction and grows the file to
    /// Σ(all evictions × image), forcing multi-GiB compaction — the thick I/O
    /// the extent design bounds. O(1) power-of-two slabs with in-place re-dump
    /// give zero steady-state file growth and no external fragmentation, at a
    /// bounded <= 2x internal slack per slab.
    ///
    /// **Allocation.** A free-list hit reuses a returned slab's offset; a miss
    /// bump-allocates from the high-water. **Free** returns the slab's offset
    /// to its class free-list (at LB discharge, when the LB flips Raw->Canonical
    /// and never raw-reloads again). **reset** recycles all offsets from 0 (the
    /// batch-start purge). The file grow that a bump past the file's end
    /// implies is the WIRING's job (`PositionedFile::grow`); this allocator is
    /// pure offset bookkeeping so it is unit-testable in isolation.
    ///
    /// **Concurrency.** All mutating entry points take `mutex_`. In production
    /// the allocator is touched only on first-dump / class-promotion /
    /// discharge-free — off the positioned-write byte-copy path (a stable-size
    /// re-dump never calls it).
    ///
    /// @invariant `allocatedBytes()` == Σ class sizes of currently-live slabs;
    ///            `highWaterBytes()` is monotone within one epoch and returns
    ///            to 0 on `reset`.
    /// @see `PositionedFile`, `D-195`.
    class ExtentAllocator {
    public:
        /// @brief Construct an uninitialized allocator (`init` sets the block
        ///        granularity before any allocation).
        ExtentAllocator() = default;

        /// @brief Bind the slab granularity and clear all state.
        ///
        /// @details
        /// The base class size (the k=0 class) equals `blockBytes`; every
        /// class is `blockBytes << k`. Clears free-lists, high-water, and the
        /// live accounting. Asserts `blockBytes > 0`.
        ///
        /// @param blockBytes The pool block size (256 KiB in production).
        void init(int64_t blockBytes);

        /// @brief Allocate the smallest slab that holds `bytes`.
        ///
        /// @details
        /// Rounds `bytes` up to a whole number of blocks, then to the next
        /// power-of-two block multiple — that class. Reuses a free-list offset
        /// for the class if one exists, else bump-allocates from the
        /// high-water. Adds the class size to `allocatedBytes()`. `bytes == 0`
        /// still yields the minimum (k=0) slab. Asserts `bytes >= 0` and the
        /// allocator initialized. Under `mutex_`.
        ///
        /// @param bytes The image size to hold (header + payload); >= 0.
        /// @return The slab's offset and its class capacity.
        SlabAllocation allocSlab(int64_t bytes);

        /// @brief Return a slab to its class free-list.
        ///
        /// @details
        /// The slab's offset becomes reusable by a later `allocSlab` of the
        /// same class. Subtracts the class size from `allocatedBytes()`.
        /// Asserts `classBytes` is a power-of-two block multiple and the
        /// allocator initialized. Does NOT shrink the file or the high-water
        /// (slack is reclaimed only by `reset`). Under `mutex_`.
        ///
        /// @param offset     The slab offset returned by `allocSlab`.
        /// @param classBytes The slab's class capacity (from the allocation).
        void freeSlab(int64_t offset, int64_t classBytes);

        /// @brief Clear all free-lists and reset the high-water to 0.
        ///
        /// @details
        /// The batch-start purge: after `reset`, offsets recycle from 0, so a
        /// batch always starts with a deterministic empty extent. Keeps the
        /// block granularity. Under `mutex_`.
        void reset();

        /// @brief Σ class sizes of currently-live slabs (live + internal slack).
        /// @return `extentAllocatedBytes` — the allocated-side telemetry.
        int64_t allocatedBytes() const;

        /// @brief The bump high-water: the max file bytes the allocations imply.
        /// @return The offset one past the highest bump-allocated slab; feeds
        ///         the file-grow decision and `extentFileBytes` telemetry.
        int64_t highWaterBytes() const;

        /// @brief The slab class capacity that would hold `bytes` (no allocation).
        ///
        /// @details
        /// The pure class-sizing function `allocSlab` uses internally, exposed
        /// so the wiring can decide in-place-overwrite vs. promotion (a re-dump
        /// fits when `bytes <= current slab classBytes`). Asserts initialized.
        ///
        /// @param bytes The image size; >= 0.
        /// @return The class capacity (a power-of-two block multiple).
        int64_t classBytesFor(int64_t bytes) const;

    private:
        /// @brief Class index for `bytes` (smallest k with `blockBytes<<k >= bytes`).
        int classIndexFor(int64_t bytes) const;
        /// @brief Class index for an exact class size (asserts it is a valid class).
        int indexForClassBytes(int64_t classBytes) const;

        mutable std::mutex mutex_;
        int64_t blockBytes_ = 0;
        int64_t highWater_ = 0;
        int64_t allocatedBytes_ = 0;
        std::vector<std::vector<int64_t>> freeLists_;   // indexed by class index
    };

}
