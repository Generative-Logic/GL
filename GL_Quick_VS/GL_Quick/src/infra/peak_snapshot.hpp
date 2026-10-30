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
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace gl { namespace peak_snapshot {
// Opt-in memory measurement: two-pass resident-byte ownership observation, never a proof input.
struct Position {
    uint64_t ordinal = 0;
    uint64_t maximumOrdinal = 0;
    int64_t maximumBytes = -1;
    int maximumPhase = -1;
    /// @brief Advance a deterministic census position and retain its largest reading.
    /// @details Equal readings preserve the first position. The sequence is per batch.
    /// @param bytes Current process working-set bytes, nonnegative.
    /// @param phase Current named process phase, nonnegative.
    /// @return Nothing.
    void observe(int64_t bytes, int phase) {
        assert(bytes >= 0 && phase >= 0);
        ++ordinal;
        if (bytes > maximumBytes) {
            maximumBytes = bytes; maximumOrdinal = ordinal; maximumPhase = phase;
        }
    }
};

struct Sink {
    void* context = nullptr;
    uint32_t (*enter)(void*, uint32_t, const char*) = nullptr;
    void (*span)(void*, uint32_t, const void*, std::size_t, uint32_t) = nullptr;
};
inline thread_local Sink sink{};
inline thread_local uint32_t currentOwner = 0;
inline thread_local void* rootContext = nullptr;
inline thread_local void (*prepareRoot)(void*) = nullptr;
inline thread_local void (*collectRoot)(void*) = nullptr;

/// @brief Test whether this thread is writing a frozen ownership census.
/// @details Ordinary operations leave the observer disabled.
/// @return True only during capture enumeration.
inline bool active() { return sink.span != nullptr; }

/// @brief Record an exact address interval under the current owner.
/// @details Zero-length spans and disabled observation intentionally do nothing.
/// Kinds 0..3 are named object/capacity/directory/content ranges. Kinds 4..11
/// identify pool, arena, slab and heap backing with lower ownership specificity.
/// @param address First byte of the interval.
/// @param bytes Number of bytes; the caller owns the entire interval.
/// @param kind Specificity and physical role of the recorded interval.
/// @return Nothing.
inline void note(const void* address, std::size_t bytes, uint32_t kind = 3) {
    if (active() && bytes != 0) sink.span(sink.context, currentOwner, address, bytes, kind);
}

/// @brief Observe a paged container's directories and data during capture.
///
/// @details
/// Traverses the same inline, single-, double- and triple-directory geometry as
/// the owning container. Directory and capacity spans are emitted explicitly,
/// so ordinary arena page resolution needs no observer checks or thread-local
/// accesses. The visitor can refine each data page with its live prefix.
///
/// @tparam Arena Resident arena exposing pageAt and pageBytes.
/// @tparam Visitor Callable accepting a logical page index and its address.
/// @param arena Arena owning every referenced virtual page.
/// @param root Root data page or page-directory identifier.
/// @param count Positive number of data pages in the container.
/// @param directoryShift Logarithm of the directory entries per page.
/// @param visit Called once per data page, in logical order.
/// @return Nothing.
/// @invariant The sink is armed and the directory geometry fits three levels.
/// @see PagedVector::liveBytes, PagedHashIndex::liveBytes.
template<class Arena, class Visitor>
void observePagedPages(const Arena& arena, int32_t root, int32_t count,
                       int32_t directoryShift, Visitor&& visit) {
    assert(active() && count > 0 && directoryShift > 0 && directoryShift < 31);
    const int64_t capacity = int64_t(1) << directoryShift;
    assert(arena.pageBytes() == capacity * static_cast<int64_t>(sizeof(int32_t)));
    assert((static_cast<int64_t>(count) - 1) / (capacity * capacity) < capacity);
    const auto directory = [&](int32_t vid) {
        const char* page = arena.pageAt(vid);
        note(page, static_cast<std::size_t>(arena.pageBytes()), 2);
        return reinterpret_cast<const int32_t*>(page);
    };
    for (int32_t index = 0; index < count; ++index) {
        int32_t vid = root;
        if (count > 1) {
            const int32_t* upper = directory(root);
            if (count <= capacity) {
                vid = upper[index];
            } else {
                if (count > capacity * capacity)
                    upper = directory(upper[static_cast<int64_t>(index) >> (2 * directoryShift)]);
                const int32_t lower = upper[(index >> directoryShift) & (capacity - 1)];
                vid = directory(lower)[index & (capacity - 1)];
            }
        }
        const char* page = arena.pageAt(vid);
        note(page, static_cast<std::size_t>(arena.pageBytes()), 1);
        visit(index, page);
    }
}

/// @brief Nest a named field in the measurement-only ownership hierarchy.
/// @details Names are interned by the armed sink; no heap allocation occurs here.
class Scope {
    uint32_t previous_;
public:
    /// @brief Enter one field name within the current owner.
    /// @details A disabled observer preserves ordinary execution behavior.
    /// @param name Stable field label.
    /// @return Constructed restoring scope.
    explicit Scope(const char* name) : previous_(currentOwner) {
        if (sink.enter) currentOwner = sink.enter(sink.context, currentOwner, name);
    }
    /// @brief Restore the preceding owner.
    /// @details Nested observation cannot leak labels into the caller.
    /// @return Nothing.
    ~Scope() { currentOwner = previous_; }
};

/// @brief Observe one actual C++ allocation, including allocator size rounding.
/// @details The pointer must be an allocation base from the active C++ allocator.
/// A zero capacity denotes an empty owner and makes no allocator query.
/// @param address Allocation base.
/// @param capacity Requested owned bytes.
/// @param used Live prefix bytes, at most capacity.
/// @return Nothing.
void allocation(const void* address,std::size_t capacity,std::size_t used);

/// @brief Observe a member's inline representation and execute its existing census.
/// @details The callable's scalar result is unchanged; nested spans receive the field name.
/// @tparam T Member type.
/// @tparam F Non-owning callable type.
/// @param name Member name.
/// @param object Member whose inline bytes are owned here.
/// @param function Existing read-only footprint computation.
/// @return The existing footprint result.
template<class T,class F>
auto measure(const char* name,const T& object,F&& function) -> decltype(function()) {
    if (!active()) return function();
    Scope scope(name);
    note(&object,sizeof(T),3);
    return function();
}

/// @brief Bind complete-root preparation and enumeration to a census seam.
/// @details The binding nests and restores; callbacks are diagnostic only.
class RootScope {
    void* context_; void (*prepare_)(void*); void (*collect_)(void*);
public:
    /// @brief Install one root collector.
    /// @details Preparation must quiesce background writers before capture.
    /// @param context Callback context with scope lifetime.
    /// @param prepare Quiescence callback.
    /// @param collect Complete named-allocation enumeration callback.
    /// @return A restoring scope.
    RootScope(void* context, void (*prepare)(void*), void (*collect)(void*))
        : context_(rootContext), prepare_(prepareRoot), collect_(collectRoot) {
        rootContext=context; prepareRoot=prepare; collectRoot=collect;
    }
    /// @brief Restore the previous root binding.
    /// @details No proof state is changed.
    /// @return Nothing.
    ~RootScope() { rootContext=context_; prepareRoot=prepare_; collectRoot=collect_; }
};


/// @brief A stack-bound transient owner participating in the selected capture.
/// @details Construction allocates nothing. Only the native seam thread mutates
/// this list; preparation callbacks stop asynchronous writers without releasing data.
struct ExtraRootScope {
    inline static thread_local ExtraRootScope* head=nullptr;
    ExtraRootScope* previous;
    const char* name;
    void* context;
    void (*prepare)(void*);
    void (*collect)(void*);
    /// @brief Register a transient root for its lexical lifetime.
    /// @details Nested scopes are visited while their objects are alive.
    /// @param label Root label.
    /// @param object Callback context.
    /// @param before Optional quiescence callback.
    /// @param visit Read-only allocation visitor.
    /// @return A registered scope.
    ExtraRootScope(const char* label,void* object,void (*before)(void*),void (*visit)(void*))
        :previous(head),name(label),context(object),prepare(before),collect(visit) {head=this;}
    /// @brief Unregister the most recent transient root.
    /// @details Lexical nesting is asserted.
    /// @return Nothing.
    ~ExtraRootScope(){assert(head==this);head=previous;}
};

/// @brief Observe a vector's owned element allocation and inline representation.
/// @details Capacity and live elements are distinct exact ranges; the return is unchanged.
/// @tparam T Vector type.
/// @param name Field name.
/// @param vector Vector to inspect without mutation.
/// @return Reserved element bytes.
template<class T>
uint64_t vectorBytes(const char* name,const T& vector) {
    Scope scope(name);
    note(&vector,sizeof(vector),3);
    allocation(vector.data(),vector.capacity()*sizeof(typename T::value_type),
        vector.size()*sizeof(typename T::value_type));
    return static_cast<uint64_t>(vector.capacity())*sizeof(typename T::value_type);
}

/// @brief Initialize the optional two-pass census for a named native batch.
/// @details GL_PEAK_MODE is absent, scan or capture; capture reads scan's position.
/// @param batch Native batch identifier.
/// @return Nothing.
void initialize(const char* batch);
/// @brief Record a position, or capture its exact replay point in the second run.
/// @details Called only at single-threaded census seams after proof workers join.
/// @param bytes Working-set reading at this position.
/// @param phase Pool-phase identifier.
/// @return Nothing.
void checkpoint(int64_t bytes, int phase);
/// @brief Save position and operating-system lifetime peak after the batch.
/// @details Scan and capture write distinct files and preserve their identities.
/// @return Nothing.
void finish();
} }
