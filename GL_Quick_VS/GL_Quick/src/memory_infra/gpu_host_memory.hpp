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
#include "../parameters.hpp"
#include <cassert>
#include <cstdlib>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

namespace gl {

    /// @brief Validate a positive MiB budget written as decimal digits.
    /// @details This predicate checks the complete text and multiplication range.
    /// It is shared by the asserting configuration door and rejection tests.
    /// @param text Non-null, null-terminated decimal text.
    /// @return True exactly for a positive representable whole-MiB budget.
    inline bool isValidGpuHostBudgetText(const char* text) {
        assert(text != nullptr);
        uint64_t value = 0;
        if (*text == '\0') return false;
        constexpr uint64_t limit = INT64_MAX / (1024ull * 1024);
        for (const char* p = text; *p != '\0'; ++p) {
            if (*p < '0' || *p > '9') return false;
            const uint64_t digit = static_cast<uint64_t>(*p - '0');
            if (value > (limit - digit) / 10) return false;
            value = value * 10 + digit;
        }
        return value > 0;
    }

    /// @brief Read the independent GPU-host limit once at initialization.
    /// @details An absent override uses the configured byte default. Malformed,
    /// zero, negative and overflowing overrides assert instead of being ignored.
    /// @param defaultBytes Positive whole-block default from ProverParameters.
    /// @return Positive byte budget, independent of the device-memory budget.
    inline int64_t configuredGpuHostPoolBytes(int64_t defaultBytes) {
        assert(defaultBytes > 0 && defaultBytes % 262144 == 0);
#ifdef _WIN32
        char text[64]{};
        std::size_t required = 0;
        const errno_t error = getenv_s(&required, text, sizeof(text), "GL_GPU_HOST_BUDGET_MIB");
        assert(error == 0 && required <= sizeof(text));
        if (required == 0) return defaultBytes;
        const char* valueText = text;
#else
        const char* valueText = std::getenv("GL_GPU_HOST_BUDGET_MIB");
        if (valueText == nullptr) return defaultBytes;
#endif
        assert(isValidGpuHostBudgetText(valueText) && "GL_GPU_HOST_BUDGET_MIB must be a positive whole MiB count");
        uint64_t value = 0;
        for (const char* p = valueText; *p != '\0'; ++p) value = value * 10 + (*p - '0');
        return static_cast<int64_t>(value * 1024 * 1024);
    }

    /// @brief Initialize the process GPU-host pool on its first consumer.
    /// @details Processor-only production execution does not call this door.
    /// The process default and environment override are sampled once.
    /// @return Independently budgeted, initialized GPU-host manager.
    inline GlobalMemoryManager& gpuHostStorage() {
        static const bool ready = [] {
            const ProverParameters defaults;
            initGpuHostMemory({configuredGpuHostPoolBytes(defaults.static_gpu_host_pool_bytes),
                262144, 2048, PoolKind::GpuHost});
            return true;
        }();
        (void)ready;
        return gpuHostMemory();
    }

    /// @brief Exact aligned byte census for one contiguous host arena.
    /// @details The planner and construction use the same append arithmetic;
    /// logical emptiness adds no storage and block rounding is charged once.
    struct GpuHostLayout {
        uint64_t bytes = 0;

        /// @brief Count one aligned array without allocating it.
        /// @details Checks multiplication, padding and addition before updating.
        /// @tparam T Element type whose size and alignment are counted.
        /// @param count Element count; zero is an empty column.
        /// @return Nothing.
        template<class T> void add(uint64_t count) {
            assert(count <= UINT64_MAX / sizeof(T));
            if (count == 0) return;
            constexpr uint64_t alignment = alignof(T);
            const uint64_t padding = (alignment - bytes % alignment) % alignment;
            assert(bytes <= UINT64_MAX - padding);
            const uint64_t aligned = bytes + padding;
            assert(count * sizeof(T) <= UINT64_MAX - aligned);
            bytes = aligned + count * sizeof(T);
        }

        /// @brief Count the complete block lease required by this layout.
        /// @details Empty layouts require no lease. Round only after all arrays.
        /// @param blockBytes Positive pool block size.
        /// @return Rounded reserved bytes including unused tail capacity.
        uint64_t roundedBytes(uint64_t blockBytes) const {
            assert(blockBytes > 0);
            if (bytes == 0) return 0;
            const uint64_t blocks = (bytes - 1) / blockBytes + 1;
            assert(blocks <= UINT64_MAX / blockBytes);
            return blocks * blockBytes;
        }
    };

    /// @brief Scoped contiguous storage for fixed-capacity GPU-host columns.
    /// @details One whole-block lease backs aligned suballocations. Ownership
    /// lasts until the arena's destruction, after all views and transfers end.
    /// No allocation relocates and no individual column uses the general heap.
    class GpuHostArena {
    public:
        /// @brief Construct an empty unbound arena.
        /// @details Reservation happens only through reserve, after sizing.
        /// @return Empty arena with no live pool lease.
        GpuHostArena() = default;
        GpuHostArena(const GpuHostArena&) = delete;
        GpuHostArena& operator=(const GpuHostArena&) = delete;

        /// @brief Return the arena's complete unregistered lease.
        /// @details Every column and asynchronous consumer must already be done.
        /// @return Nothing.
        ~GpuHostArena() {
            if (manager_ != nullptr) manager_->releaseGpuHostRange(data_, blocks_);
        }

        /// @brief Reserve a previously measured contiguous payload.
        /// @details Rounds to whole blocks; the tail remains counted capacity.
        /// @param bytes Positive requested payload size.
        /// @param manager Initialized independent GPU-host pool.
        /// @return Nothing.
        void reserve(uint64_t bytes, GlobalMemoryManager& manager) {
            assert(manager_ == nullptr && bytes > 0);
            const uint64_t block = static_cast<uint64_t>(manager.blockBytes());
            assert(bytes <= static_cast<uint64_t>(INT32_MAX) * block);
            blocks_ = static_cast<int32_t>((bytes - 1) / block + 1);
            data_ = manager.acquireGpuHostRange(blocks_);
            manager_ = &manager;
            capacity_ = bytes;
        }

        /// @brief Reserve measured payload from the process GPU-host pool.
        /// @details Initializes its reservation lazily at the first GPU consumer.
        /// @param bytes Positive requested payload size.
        /// @return Nothing.
        void reserve(uint64_t bytes) { reserve(bytes, gpuHostStorage()); }

        /// @brief Carve one stable aligned column from this arena.
        /// @details Checked arithmetic prevents either padding or payload overflow.
        /// @param bytes Positive payload size.
        /// @param alignment Power-of-two alignment no larger than a pool block.
        /// @return Stable aligned byte address.
        void* allocate(uint64_t bytes, uint64_t alignment) {
            assert(manager_ != nullptr && bytes > 0);
            assert(alignment > 0 && (alignment & (alignment - 1)) == 0);
            assert(alignment <= static_cast<uint64_t>(manager_->blockBytes()));
            const uint64_t padding = (alignment - used_ % alignment) % alignment;
            assert(padding <= capacity_ - used_);
            const uint64_t start = used_ + padding;
            assert(bytes <= capacity_ - start && "GPU-host arena sizing omitted a live column");
            used_ = start + bytes;
            return data_ + start;
        }

        /// @brief Expose the exclusive lease head for CUDA registration.
        /// @details Interior columns remain within this stable complete range.
        /// @return Lease base, or null for an empty arena.
        char* data() const { return data_; }

        /// @brief Read the number of whole blocks owned by this arena.
        /// @details Used with exact-range registration accounting.
        /// @return Block count, zero for an empty arena.
        int32_t blockCount() const { return blocks_; }

        /// @brief Read the measured payload ceiling.
        /// @details Excludes whole-block rounding, which reservedBytes reports.
        /// @return Declared arena payload capacity.
        uint64_t capacity() const { return capacity_; }

        /// @brief Read payload plus alignment consumed so far.
        /// @details Observability only; no allocation is performed.
        /// @return Used byte prefix.
        uint64_t usedBytes() const { return used_; }

        /// @brief Read the whole-block lease size charged to the pool.
        /// @details Empty arenas own no bytes.
        /// @return Reserved byte count including the rounded tail.
        uint64_t reservedBytes() const {
            return manager_ == nullptr ? 0 : static_cast<uint64_t>(blocks_) * manager_->blockBytes();
        }

    private:
        GlobalMemoryManager* manager_ = nullptr;
        char* data_ = nullptr;
        uint64_t capacity_ = 0;
        uint64_t used_ = 0;
        int32_t blocks_ = 0;
    };
    /// @brief Fixed-count object storage for GPU-owned control structures.
    /// @details Objects are placement-constructed inside one pool lease. Their
    /// destructors run before that lease returns, including nested arena owners.
    /// @tparam T Default-constructible object type.
    template<class T> class GpuHostObjectArray {
        GpuHostArena storage_;
        T* data_ = nullptr;
        std::size_t count_ = 0;
    public:
        /// @brief Construct exactly the measured number of objects.
        /// @details A zero-count array is empty and needs no pool lease.
        /// @param count Fixed object count; multiplication must be representable.
        /// @return An owning array with stable object addresses.
        explicit GpuHostObjectArray(std::size_t count) : count_(count) {
            if (count == 0) return;
            assert(count <= UINT64_MAX / sizeof(T));
            storage_.reserve(static_cast<uint64_t>(count) * sizeof(T));
            data_ = static_cast<T*>(storage_.allocate(count * sizeof(T), alignof(T)));
            for (std::size_t i = 0; i < count_; ++i) ::new (static_cast<void*>(data_ + i)) T();
        }
        GpuHostObjectArray(const GpuHostObjectArray&) = delete;
        GpuHostObjectArray& operator=(const GpuHostObjectArray&) = delete;

        /// @brief Destroy owned objects before returning their storage.
        /// @details Reverse order matches ordinary array destruction.
        /// @return Nothing.
        ~GpuHostObjectArray() {
            for (std::size_t i = count_; i > 0; --i) data_[i - 1].~T();
        }

        /// @brief Access an existing owned object.
        /// @details Object addresses remain stable through this array's lifetime.
        /// @param index Position strictly below the fixed count.
        /// @return Mutable object reference.
        T& operator[](std::size_t index) { assert(index < count_); return data_[index]; }

        /// @brief Read an existing owned object.
        /// @details Object addresses remain stable through this array's lifetime.
        /// @param index Position strictly below the fixed count.
        /// @return Constant object reference.
        const T& operator[](std::size_t index) const { assert(index < count_); return data_[index]; }

        /// @brief Read the fixed object count.
        /// @details The array never resizes or relocates.
        /// @return Construction count, including zero for an empty array.
        std::size_t size() const { return count_; }

        /// @brief Expose the first object to a borrowed consumer.
        /// @details The pointer is valid only through this array's lifetime.
        /// @return Data address, null only for an empty array.
        T* data() { return data_; }

        /// @brief Read the complete lease cost of the object array.
        /// @details Includes alignment and pool-block rounding.
        /// @return Bytes charged to the independent GPU-host pool.
        uint64_t reservedBytes() const { return storage_.reservedBytes(); }
    };

    /// @brief A non-owning contiguous view of fixed GPU-host metadata.
    /// @details The producer preserves the underlying storage through every use.
    /// @tparam T Element type, optionally const.
    template<class T> class GpuHostSpan {
        T* data_ = nullptr;
        std::size_t count_ = 0;
    public:
        /// @brief Construct a borrowed run.
        /// @details Empty runs may have a null pointer; nonempty runs may not.
        /// @param data First element.
        /// @param count Number of available elements.
        /// @return Non-owning view with no allocation.
        GpuHostSpan(T* data, std::size_t count) : data_(data), count_(count) {
            assert(data != nullptr || count == 0);
        }
        /// @brief Read the borrowed element count.
        /// @details Does not inspect or allocate payload.
        /// @return Run length.
        std::size_t size() const { return count_; }
        /// @brief Test whether the borrowed run has no elements.
        /// @details Empty is an ordinary view state.
        /// @return True exactly for zero length.
        bool empty() const { return count_ == 0; }
        /// @brief Borrow an element without transferring ownership.
        /// @details Storage must still be live.
        /// @param index Position strictly below the run length.
        /// @return Element reference preserving T's constness.
        T& operator[](std::size_t index) const { assert(index < count_); return data_[index]; }
        /// @brief Expose the beginning for ordered iteration.
        /// @details Iteration order is the stored run order.
        /// @return First element address, or null for a null empty view.
        T* begin() const { return data_; }
        /// @brief Expose the ordered iteration endpoint.
        /// @details No pointer arithmetic is performed on a null empty view.
        /// @return One-past-last address.
        T* end() const { return count_ == 0 ? data_ : data_ + count_; }
        /// @brief Borrow the first element of a nonempty run.
        /// @details The nonempty precondition is asserted.
        /// @return First element reference.
        T& front() const { assert(count_ > 0); return data_[0]; }
    };

}
