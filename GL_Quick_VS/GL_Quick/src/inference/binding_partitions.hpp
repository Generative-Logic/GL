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
#include "../memory.hpp"
#include "../compiler/compiled_definition_registry.hpp"
#include "copy_ref.hpp"
namespace gl {
/// @brief Enumerate the permitted variable-binding partitions in deterministic order.
/// @details Preserves the caller-owned scratch storage and ordered output contract.
class BindingPartitions {
public:

/// @brief Enumerate all set partitions of @p elements into an arena CSR
        ///        of element INDICES — the heap-free twin of the former
        ///        `std::vector<std::vector<std::vector<std::string>>>` builder.
        ///
        /// @details
        /// Fills three caller-owned `PagedVector<int32_t>` columns on @p arena's
        /// page tier that encode the partition family as index runs (members are
        /// indices into @p elements, never string copies):
        /// - @p outMember — every class's members, concatenated.
        /// - @p outClassStart — per-class start into @p outMember, with a
        ///   trailing terminator; class `c` spans
        ///   `[outClassStart[c], outClassStart[c+1])`.
        /// - @p outPartStart — per-partition start into @p outClassStart, with a
        ///   trailing terminator; partition `p` spans classes
        ///   `[outPartStart[p], outPartStart[p+1])`.
        ///
        /// The enumeration order AND the within-class member order are OBSERVABLE
        /// downstream (the partition order feeds `multiplyImplication`'s
        /// `seen`/`result` dedup; the within-class order feeds representative
        /// selection), so both are reproduced byte-for-byte against the retired
        /// heap builder. Three regimes, exactly as the heap form:
        /// - `n == 0` → one empty partition.
        /// - `n > cap` → flat: all-singletons, then every pairwise equality, then
        ///   every triple equality.
        /// - `1 <= n <= cap` → the recursive Bell enumeration, ported to an
        ///   iterative ping-pong over two arena CSR buffers (peel-first-element
        ///   is applied outer-last, so expanding indices `n-1 .. 0` reproduces
        ///   the recursion's Option-1-then-Option-2 emission order). Within-class
        ///   member order is the `compareSpans(elements[idx])` byte-lex sort — the
        ///   `std::sort` on member strings the heap form ran.
        ///
        /// @param elements       The elements to partition (caller-stable; the
        ///                       CSR indexes into this list).
        /// @param cap            The recursion/flat cutoff (`max_partition_size`).
        /// @param arena          Per-slot gen-scratch arena for the CSR columns +
        ///                       the recursion's ping-pong buffers (page tier).
        /// @param outMember      Out: concatenated class members (indices).
        /// @param outClassStart  Out: per-class start index + terminator.
        /// @param outPartStart   Out: per-partition start index + terminator.
        /// @invariant Byte-identical partition/class/member sequence to the heap
        ///            oracle (twin test `generate_set_partitions_arena_matches_heap`).
        /// @note Forwards to the `StrSpan`-run overload (the real implementation)
        ///       after building a span run over @p elements on @p arena.
        /// @see multiplyImplication — the sole consumer (maps index -> string).
/// @return Nothing.
static void generateSetPartitions(const std::vector<std::string>& elements,
            int cap, ScratchArena& arena,
            PagedVector<int32_t>& outMember,
            PagedVector<int32_t>& outClassStart,
            PagedVector<int32_t>& outPartStart);

/// @brief Span-run overload of @ref generateSetPartitions — the real
        ///        implementation, reading elements as `StrSpan`s.
        ///
        /// @details
        /// Byte-for-byte identical partition/class/member CSR to the
        /// `std::vector<std::string>` overload (which forwards to this form after
        /// building a `StrSpan` run over its strings). The interior reads each
        /// element only through `compareSpans` (the within-class byte-lex sort) +
        /// the element count, so no `std::string` is touched — the reason the
        /// overload is a mechanical retype. Every regime (`n == 0` / `n > cap`
        /// flat / `1 <= n <= cap` recursive Bell), the partition enumeration
        /// order, and the within-class member order are exactly as documented on
        /// the `std::vector<std::string>` overload.
        ///
        /// @param elements Pointer to @p n element spans (caller-stable; the CSR
        ///                 indexes into this run).
        /// @param n        The element count (`>= 0`).
        /// @param cap      The recursion/flat cutoff (`max_partition_size`).
        /// @param arena    Per-slot gen-scratch arena for the CSR columns + the
        ///                 recursion's ping-pong buffers (page tier).
        /// @param outMember      Out: concatenated class members (indices).
        /// @param outClassStart  Out: per-class start index + terminator.
        /// @param outPartStart   Out: per-partition start index + terminator.
        /// @invariant Byte-identical to the `std::vector<std::string>` overload and
        ///            the heap oracle (twin test
        ///            `generate_set_partitions_arena_matches_heap`).
        /// @see generateSetPartitions(const std::vector<std::string>&, int,
        ///      ScratchArena&, …) — the forwarding overload; multiplyImplication.
/// @return Nothing.
static void generateSetPartitions(const StrSpan* elements, int32_t n,
            int cap, ScratchArena& arena,
            PagedVector<int32_t>& outMember,
            PagedVector<int32_t>& outClassStart,
            PagedVector<int32_t>& outPartStart);
};
}
