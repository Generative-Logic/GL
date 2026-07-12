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

#include "extent_file.hpp"

#include <atomic>
#include <cassert>
#include <cstdint>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#endif

namespace gl {

    namespace {
        /// @brief Hand out a PROCESS-GLOBAL-unique open generation.
        ///
        /// @details
        /// Stamped into `PositionedFile::openGeneration_` at every `open` so
        /// the per-thread handle cache's (file, generation) key never collides
        /// across an object-address reuse. Starts at 1 (0 is the never-opened
        /// sentinel).
        ///
        /// @return A monotonically increasing id, unique for the process.
        uint64_t nextOpenGeneration() {
            static std::atomic<uint64_t> counter{ 0 };
            return counter.fetch_add(1, std::memory_order_relaxed) + 1;
        }
    }

#ifdef _WIN32

    namespace {
        /// @brief Per-thread cache of open handles, keyed by the file object
        ///        and its open generation so a reused / reopened `PositionedFile`
        ///        never serves a stale (closed) handle.
        ///
        /// @details
        /// One entry per (file, generation) a thread has done I/O on. In
        /// practice a single extent file lives at a time, so this vector holds
        /// one element and the lookup is O(1). The handle is also registered
        /// in the owning file's `allHandles_` so `close` reaps it — this cache
        /// is only the lock-free fast path, not the ownership record.
        struct TlsHandleEntry {
            const PositionedFile* file;
            uint64_t generation;
            void* handle;
        };
        thread_local std::vector<TlsHandleEntry> g_tlsHandles;
    }

    void* PositionedFile::threadHandle() {
        assert(open_ && "positioned I/O on a closed extent file");
        for (const TlsHandleEntry& e : g_tlsHandles) {
            if (e.file == this && e.generation == openGeneration_) {
                assert(e.handle != INVALID_HANDLE_VALUE);
                return e.handle;
            }
        }
        // First I/O from this thread for this open-generation: open an
        // INDEPENDENT handle (own OS file object -> no shared file pointer,
        // no file-object serialization lock) and register it for close().
        HANDLE h = CreateFileW(
            path_.c_str(), GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS,
            FILE_ATTRIBUTE_NORMAL, nullptr);
        assert(h != INVALID_HANDLE_VALUE
            && "extent per-thread CreateFileW failed");
        {
            std::lock_guard<std::mutex> guard(mutex_);
            allHandles_.push_back(h);
        }
        g_tlsHandles.push_back(TlsHandleEntry{ this, openGeneration_, h });
        return h;
    }

    PositionedFile::~PositionedFile() {
        close();
    }

    void PositionedFile::open(const std::filesystem::path& path) {
        assert(!open_ && "extent file already open");
        HANDLE h = CreateFileW(
            path.c_str(), GENERIC_READ | GENERIC_WRITE,
            FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS,
            FILE_ATTRIBUTE_NORMAL, nullptr);
        assert(h != INVALID_HANDLE_VALUE && "extent CreateFileW failed");
        std::lock_guard<std::mutex> guard(mutex_);
        primaryHandle_ = h;
        path_ = path;
        fileSize_ = 0;
        openGeneration_ = nextOpenGeneration();
        open_ = true;
    }

    void PositionedFile::close() {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!open_) return;
        for (void* h : allHandles_) {
            const BOOL ok = CloseHandle(static_cast<HANDLE>(h));
            assert(ok && "extent per-thread CloseHandle failed");
            (void)ok;
        }
        allHandles_.clear();
        const BOOL ok = CloseHandle(static_cast<HANDLE>(primaryHandle_));
        assert(ok && "extent primary CloseHandle failed");
        (void)ok;
        primaryHandle_ = nullptr;
        fileSize_ = 0;
        // Leave openGeneration_ as-is: the next open draws a fresh global id,
        // so no stale thread-local (this, generation) entry can match.
        open_ = false;
    }

    void PositionedFile::preallocate(int64_t bytes) {
        assert(bytes >= 0);
        std::lock_guard<std::mutex> guard(mutex_);
        assert(open_ && "preallocate on a closed extent file");
        LARGE_INTEGER li;
        li.QuadPart = bytes;
        const BOOL sk = SetFilePointerEx(
            static_cast<HANDLE>(primaryHandle_), li, nullptr, FILE_BEGIN);
        assert(sk && "extent SetFilePointerEx failed");
        (void)sk;
        const BOOL ek = SetEndOfFile(static_cast<HANDLE>(primaryHandle_));
        assert(ek && "extent preallocate SetEndOfFile failed — free disk or "
                     "lower kExtentInitialBytes");
        (void)ek;
        fileSize_ = bytes;
    }

    void PositionedFile::grow(int64_t newSize) {
        std::lock_guard<std::mutex> guard(mutex_);
        assert(open_ && "grow on a closed extent file");
        assert(newSize >= fileSize_ && "extent grow must not shrink");
        LARGE_INTEGER li;
        li.QuadPart = newSize;
        const BOOL sk = SetFilePointerEx(
            static_cast<HANDLE>(primaryHandle_), li, nullptr, FILE_BEGIN);
        assert(sk && "extent SetFilePointerEx failed");
        (void)sk;
        const BOOL ek = SetEndOfFile(static_cast<HANDLE>(primaryHandle_));
        assert(ek && "extent growth failed — free disk, add a .deload "
                     "Defender exclusion, or raise kExtentInitialBytes");
        (void)ek;
        fileSize_ = newSize;
    }

    void PositionedFile::writeAt(int64_t offset, const void* buf, int64_t len) {
        assert(offset >= 0);
        assert(len >= 0);
        HANDLE h = static_cast<HANDLE>(threadHandle());
        const char* p = static_cast<const char*>(buf);
        int64_t pos = offset;
        int64_t remaining = len;
        while (remaining > 0) {
            const DWORD chunk = static_cast<DWORD>(
                remaining < kMaxIoChunkBytes ? remaining : kMaxIoChunkBytes);
            OVERLAPPED ov{};
            ov.Offset = static_cast<DWORD>(static_cast<uint64_t>(pos)
                & 0xFFFFFFFFull);
            ov.OffsetHigh = static_cast<DWORD>(
                static_cast<uint64_t>(pos) >> 32);
            DWORD written = 0;
            const BOOL ok = WriteFile(h, p, chunk, &written, &ov);
            assert(ok && "extent WriteFile failed — free disk or check the "
                         ".deload path");
            assert(written == chunk && "extent WriteFile short write");
            (void)ok;
            pos += chunk;
            p += chunk;
            remaining -= chunk;
        }
    }

    void PositionedFile::readAt(int64_t offset, void* buf, int64_t len) {
        assert(offset >= 0);
        assert(len >= 0);
        HANDLE h = static_cast<HANDLE>(threadHandle());
        char* p = static_cast<char*>(buf);
        int64_t pos = offset;
        int64_t remaining = len;
        while (remaining > 0) {
            const DWORD chunk = static_cast<DWORD>(
                remaining < kMaxIoChunkBytes ? remaining : kMaxIoChunkBytes);
            OVERLAPPED ov{};
            ov.Offset = static_cast<DWORD>(static_cast<uint64_t>(pos)
                & 0xFFFFFFFFull);
            ov.OffsetHigh = static_cast<DWORD>(
                static_cast<uint64_t>(pos) >> 32);
            DWORD got = 0;
            const BOOL ok = ReadFile(h, p, chunk, &got, &ov);
            assert(ok && "extent ReadFile failed");
            assert(got == chunk && "extent ReadFile short read — reading "
                                   "unwritten or foreign bytes");
            (void)ok;
            pos += chunk;
            p += chunk;
            remaining -= chunk;
        }
    }

#else   // POSIX

    int PositionedFile::threadHandle() {
        assert(open_ && "positioned I/O on a closed extent file");
        return fd_;   // pread/pwrite are safe from many threads at distinct offsets
    }

    PositionedFile::~PositionedFile() {
        close();
    }

    void PositionedFile::open(const std::filesystem::path& path) {
        assert(!open_ && "extent file already open");
        const int fd = ::open(path.c_str(), O_RDWR | O_CREAT, 0644);
        assert(fd >= 0 && "extent open() failed");
        std::lock_guard<std::mutex> guard(mutex_);
        fd_ = fd;
        path_ = path;
        fileSize_ = 0;
        openGeneration_ = nextOpenGeneration();
        open_ = true;
    }

    void PositionedFile::close() {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!open_) return;
        const int r = ::close(fd_);
        assert(r == 0 && "extent close() failed");
        (void)r;
        fd_ = -1;
        fileSize_ = 0;
        // Next open draws a fresh global id; no stale TLS entry can match.
        open_ = false;
    }

    void PositionedFile::preallocate(int64_t bytes) {
        assert(bytes >= 0);
        std::lock_guard<std::mutex> guard(mutex_);
        assert(open_ && "preallocate on a closed extent file");
        const int r = ::ftruncate(fd_, static_cast<off_t>(bytes));
        assert(r == 0 && "extent preallocate ftruncate failed — free disk or "
                         "lower kExtentInitialBytes");
        (void)r;
        fileSize_ = bytes;
    }

    void PositionedFile::grow(int64_t newSize) {
        std::lock_guard<std::mutex> guard(mutex_);
        assert(open_ && "grow on a closed extent file");
        assert(newSize >= fileSize_ && "extent grow must not shrink");
        const int r = ::ftruncate(fd_, static_cast<off_t>(newSize));
        assert(r == 0 && "extent growth ftruncate failed — free disk, add a "
                         ".deload exclusion, or raise kExtentInitialBytes");
        (void)r;
        fileSize_ = newSize;
    }

    void PositionedFile::writeAt(int64_t offset, const void* buf, int64_t len) {
        assert(offset >= 0);
        assert(len >= 0);
        const int fd = threadHandle();
        const char* p = static_cast<const char*>(buf);
        int64_t pos = offset;
        int64_t remaining = len;
        while (remaining > 0) {
            const int64_t chunk =
                remaining < kMaxIoChunkBytes ? remaining : kMaxIoChunkBytes;
            const ssize_t n = ::pwrite(fd, p, static_cast<size_t>(chunk),
                                       static_cast<off_t>(pos));
            assert(n > 0 && "extent pwrite failed — free disk or check the "
                            ".deload path");
            pos += n;
            p += n;
            remaining -= n;
        }
    }

    void PositionedFile::readAt(int64_t offset, void* buf, int64_t len) {
        assert(offset >= 0);
        assert(len >= 0);
        const int fd = threadHandle();
        char* p = static_cast<char*>(buf);
        int64_t pos = offset;
        int64_t remaining = len;
        while (remaining > 0) {
            const int64_t chunk =
                remaining < kMaxIoChunkBytes ? remaining : kMaxIoChunkBytes;
            const ssize_t n = ::pread(fd, p, static_cast<size_t>(chunk),
                                      static_cast<off_t>(pos));
            assert(n > 0 && "extent pread short read — reading unwritten or "
                            "foreign bytes");
            pos += n;
            p += n;
            remaining -= n;
        }
    }

#endif

    bool PositionedFile::isOpen() const {
        std::lock_guard<std::mutex> guard(mutex_);
        return open_;
    }

    int64_t PositionedFile::fileSize() const {
        std::lock_guard<std::mutex> guard(mutex_);
        return fileSize_;
    }

    // ---- ExtentAllocator ------------------------------------------------

    int ExtentAllocator::classIndexFor(int64_t bytes) const {
        assert(blockBytes_ > 0 && "ExtentAllocator not initialized");
        assert(bytes >= 0);
        int64_t blocks = (bytes + blockBytes_ - 1) / blockBytes_;   // ceil
        if (blocks < 1) blocks = 1;
        int k = 0;
        int64_t cap = 1;
        while (cap < blocks) { cap <<= 1; ++k; }
        return k;
    }

    int ExtentAllocator::indexForClassBytes(int64_t classBytes) const {
        assert(blockBytes_ > 0 && "ExtentAllocator not initialized");
        assert(classBytes >= blockBytes_ && classBytes % blockBytes_ == 0
            && "extent classBytes not a block multiple");
        const int64_t mult = classBytes / blockBytes_;
        assert((mult & (mult - 1)) == 0
            && "extent classBytes not a power-of-two block multiple");
        int k = 0;
        while ((static_cast<int64_t>(1) << k) < mult) ++k;
        return k;
    }

    void ExtentAllocator::init(int64_t blockBytes) {
        assert(blockBytes > 0 && "ExtentAllocator block size must be positive");
        std::lock_guard<std::mutex> guard(mutex_);
        blockBytes_ = blockBytes;
        highWater_ = 0;
        allocatedBytes_ = 0;
        freeLists_.clear();
    }

    int64_t ExtentAllocator::classBytesFor(int64_t bytes) const {
        std::lock_guard<std::mutex> guard(mutex_);
        const int k = classIndexFor(bytes);
        return blockBytes_ << k;
    }

    SlabAllocation ExtentAllocator::allocSlab(int64_t bytes) {
        std::lock_guard<std::mutex> guard(mutex_);
        const int k = classIndexFor(bytes);
        const int64_t classBytes = blockBytes_ << k;
        if (static_cast<int>(freeLists_.size()) <= k)
            freeLists_.resize(static_cast<std::size_t>(k) + 1);
        int64_t offset;
        std::vector<int64_t>& list = freeLists_[static_cast<std::size_t>(k)];
        if (!list.empty()) {
            offset = list.back();
            list.pop_back();
        } else {
            offset = highWater_;
            highWater_ += classBytes;
        }
        allocatedBytes_ += classBytes;
        return SlabAllocation{ offset, classBytes };
    }

    void ExtentAllocator::freeSlab(int64_t offset, int64_t classBytes) {
        std::lock_guard<std::mutex> guard(mutex_);
        assert(offset >= 0 && offset < highWater_
            && "extent freeSlab offset out of range");
        const int k = indexForClassBytes(classBytes);
        if (static_cast<int>(freeLists_.size()) <= k)
            freeLists_.resize(static_cast<std::size_t>(k) + 1);
        freeLists_[static_cast<std::size_t>(k)].push_back(offset);
        assert(allocatedBytes_ >= classBytes
            && "extent freeSlab double-free / accounting underflow");
        allocatedBytes_ -= classBytes;
    }

    void ExtentAllocator::reset() {
        std::lock_guard<std::mutex> guard(mutex_);
        highWater_ = 0;
        allocatedBytes_ = 0;
        freeLists_.clear();
    }

    int64_t ExtentAllocator::allocatedBytes() const {
        std::lock_guard<std::mutex> guard(mutex_);
        return allocatedBytes_;
    }

    int64_t ExtentAllocator::highWaterBytes() const {
        std::lock_guard<std::mutex> guard(mutex_);
        return highWater_;
    }

}
