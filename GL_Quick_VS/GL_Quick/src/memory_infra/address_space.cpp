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

#include "address_space.hpp"

#include <cassert>
#include <cstddef>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace gl {

    /// @brief Reserve `bytes` of contiguous address space without committing
    ///        any of it.
    ///
    /// @details
    /// The range is inaccessible until `commitAddressSpace` commits a part of
    /// it; reserving costs no commit charge and no physical memory. On Windows
    /// this is `VirtualAlloc(MEM_RESERVE)` (64 KiB-aligned base); elsewhere an
    /// anonymous `PROT_NONE` mapping with `MAP_NORESERVE` (page-aligned base).
    /// Both bases are multiples of 4096, which the CUDA host registration of
    /// pool blocks requires. Failure is an assert: the static pools are sized
    /// at program start and have no fallback (Rule 19).
    ///
    /// @param bytes Size of the range; positive.
    /// @return Base of the reserved range, aligned to at least 4096 bytes.
    /// @see commitAddressSpace, releaseAddressSpace
    char* reserveAddressSpace(int64_t bytes) {
        assert(bytes > 0);
#ifdef _WIN32
        void* const base = VirtualAlloc(nullptr, static_cast<SIZE_T>(bytes),
                                        MEM_RESERVE, PAGE_READWRITE);
        assert(base != nullptr && "address-space reservation failed");
#else
        void* const base = mmap(nullptr, static_cast<std::size_t>(bytes), PROT_NONE,
                                MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        assert(base != MAP_FAILED && "address-space reservation failed");
#endif
        assert(reinterpret_cast<uintptr_t>(base) % 4096 == 0);
        return static_cast<char*>(base);
    }

    /// @brief Commit a sub-range of a reservation for reading and writing.
    ///
    /// @details
    /// The range is widened to whole operating-system pages; committing a page
    /// that is already committed leaves its content unchanged, so blocks
    /// smaller than a page may share one. Freshly committed pages read as zero
    /// and take physical memory only when first written. A commit the system
    /// cannot back (commit limit reached) is an assert.
    ///
    /// @param address First byte to commit; inside a live reservation.
    /// @param bytes   Number of bytes to commit; positive, inside the same
    ///                reservation.
    /// @see reserveAddressSpace
    void commitAddressSpace(char* address, int64_t bytes) {
        assert(address != nullptr && bytes > 0);
#ifdef _WIN32
        // VirtualAlloc widens a commit to every page the range touches.
        void* const committed = VirtualAlloc(address, static_cast<SIZE_T>(bytes),
                                             MEM_COMMIT, PAGE_READWRITE);
        assert(committed != nullptr
            && "committing pool memory failed — the system commit limit is reached");
#else
        const uintptr_t page = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
        const uintptr_t start = reinterpret_cast<uintptr_t>(address);
        const uintptr_t first = start / page * page;
        const uintptr_t end = (start + static_cast<uintptr_t>(bytes) + page - 1) / page * page;
        const int rc = mprotect(reinterpret_cast<void*>(first),
                                static_cast<std::size_t>(end - first),
                                PROT_READ | PROT_WRITE);
        assert(rc == 0 && "committing pool memory failed");
#endif
    }

    /// @brief Return a whole reservation, committed parts included, to the
    ///        operating system.
    ///
    /// @details
    /// `address` and `bytes` must be exactly a `reserveAddressSpace` result
    /// and its size. Failure is an assert.
    ///
    /// @param address Base returned by `reserveAddressSpace`.
    /// @param bytes   The size passed to `reserveAddressSpace`.
    /// @see reserveAddressSpace
    void releaseAddressSpace(char* address, int64_t bytes) {
        assert(address != nullptr && bytes > 0);
#ifdef _WIN32
        // MEM_RELEASE frees the whole reservation and requires size 0.
        const BOOL ok = VirtualFree(address, 0, MEM_RELEASE);
        assert(ok && "releasing an address-space reservation failed");
#else
        const int rc = munmap(address, static_cast<std::size_t>(bytes));
        assert(rc == 0 && "releasing an address-space reservation failed");
#endif
    }

} // namespace gl
