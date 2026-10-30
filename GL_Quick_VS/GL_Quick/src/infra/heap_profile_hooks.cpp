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

// Replacement of the global allocation operators for the heap census
// (`heap_profile`). Active only in measurement builds that link mimalloc:
// every block comes from mimalloc exactly as through the default operators,
// and its usable size is added to or subtracted from the live-heap counter,
// with an optional sampled call stack. Production builds keep the default
// operators.

#include "heap_profile.hpp"
#include "../parameters.hpp"

#include <cstddef>
#include <new>

#if HEAP_MEASUREMENT && defined(USE_MIMALLOC)

#include <mimalloc.h>

namespace gl {
namespace heap_profile {
namespace {

    /// @brief Count (and maybe sample) one fresh block.
    ///
    /// @param block A block just returned by mimalloc; not null.
    /// @return `block`.
    void* counted(void* block) {
        onAllocate(block, static_cast<int64_t>(mi_usable_size(block)));
        return block;
    }

    /// @brief Uncount one block, then free it; freeing null does nothing.
    ///
    /// @param block A block from the replaced operators, or null.
    void released(void* block) noexcept {
        if (block == nullptr) return;
        onFree(block, static_cast<int64_t>(mi_usable_size(block)));
        mi_free(block);
    }

    /// @brief Count a nothrow allocation, which may legitimately be null.
    ///
    /// @param block Result of a nothrow mimalloc allocation.
    /// @return `block`.
    void* countedNothrow(void* block) noexcept {
        return block == nullptr ? block : counted(block);
    }

} // namespace

/// @brief Whether this build replaces the global allocation operators.
///
/// @return `true`: this translation unit's replacements are active.
bool countsNewDelete() { return true; }

} // namespace heap_profile
} // namespace gl

using gl::heap_profile::counted;
using gl::heap_profile::countedNothrow;
using gl::heap_profile::released;

void* operator new(std::size_t size) { return counted(mi_new(size)); }
void* operator new[](std::size_t size) { return counted(mi_new(size)); }
void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
    return countedNothrow(mi_new_nothrow(size));
}
void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
    return countedNothrow(mi_new_nothrow(size));
}
void* operator new(std::size_t size, std::align_val_t alignment) {
    return counted(mi_new_aligned(size, static_cast<std::size_t>(alignment)));
}
void* operator new[](std::size_t size, std::align_val_t alignment) {
    return counted(mi_new_aligned(size, static_cast<std::size_t>(alignment)));
}
void* operator new(std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
    return countedNothrow(mi_new_aligned_nothrow(size, static_cast<std::size_t>(alignment)));
}
void* operator new[](std::size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
    return countedNothrow(mi_new_aligned_nothrow(size, static_cast<std::size_t>(alignment)));
}

void operator delete(void* block) noexcept { released(block); }
void operator delete[](void* block) noexcept { released(block); }
void operator delete(void* block, std::size_t) noexcept { released(block); }
void operator delete[](void* block, std::size_t) noexcept { released(block); }
void operator delete(void* block, const std::nothrow_t&) noexcept { released(block); }
void operator delete[](void* block, const std::nothrow_t&) noexcept { released(block); }
void operator delete(void* block, std::align_val_t) noexcept { released(block); }
void operator delete[](void* block, std::align_val_t) noexcept { released(block); }
void operator delete(void* block, std::size_t, std::align_val_t) noexcept { released(block); }
void operator delete[](void* block, std::size_t, std::align_val_t) noexcept { released(block); }
void operator delete(void* block, std::align_val_t, const std::nothrow_t&) noexcept { released(block); }
void operator delete[](void* block, std::align_val_t, const std::nothrow_t&) noexcept { released(block); }

#else

namespace gl {
    namespace heap_profile {
        /// @brief Whether this build replaces the global allocation operators.
        ///
        /// @return `false`: production or non-mimalloc build, default operators.
        bool countsNewDelete() { return false; }
    }
}

#endif
