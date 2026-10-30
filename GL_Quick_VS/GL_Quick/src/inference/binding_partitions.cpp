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
#include "binding_partitions.hpp"
#include "binding_partitions.hpp"
#include "../proof/worker_context.hpp"
#include <cctype>
#include <unordered_set>
namespace gl {

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
void BindingPartitions::generateSetPartitions(
    const std::vector<std::string>& elements, int cap, ScratchArena& arena,
    PagedVector<int32_t>& outMember, PagedVector<int32_t>& outClassStart,
    PagedVector<int32_t>& outPartStart) {
    const int32_t n = static_cast<int32_t>(elements.size());
    const StrSpan* spans = nullptr;
    if (n > 0) {
        StrSpan* s = reinterpret_cast<StrSpan*>(arena.resolve(
            arena.alloc(n * static_cast<int32_t>(sizeof(StrSpan)),
                        static_cast<int32_t>(alignof(StrSpan)))));
        for (int32_t i = 0; i < n; ++i)
            s[i] = StrSpan(elements[static_cast<std::size_t>(i)]);
        spans = s;
    }
    generateSetPartitions(spans, n, cap, arena, outMember, outClassStart,
        outPartStart);
}

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
void BindingPartitions::generateSetPartitions(
    const StrSpan* elements, int32_t n, int cap, ScratchArena& arena,
    PagedVector<int32_t>& outMember, PagedVector<int32_t>& outClassStart,
    PagedVector<int32_t>& outPartStart) {
    // Max class size in the recursive branch is n (<= cap) + the peeled index;
    // a loud assert on the stack sort buffer is the Rule-19 tripwire.
    constexpr int32_t kMaxClassBuf = 64;

    // Within-class byte-lex order on element indices == the heap std::sort on
    // the member strings (elements are distinct, so no ties).
    const auto idxLess = [&elements](int32_t a, int32_t b) {
        return compareSpans(elements[a], elements[b]) < 0;
    };

    // n == 0: one empty partition.
    if (n == 0) {
        outClassStart.push_back(0);
        outPartStart.push_back(0);
        outPartStart.push_back(0);   // partition 0 spans classes [0, 0)
        return;
    }

    // n > cap: flat — all-singletons, then every pairwise, then every triple.
    if (n > cap) {
        outClassStart.push_back(0);
        outPartStart.push_back(0);
        const auto closeClass = [&]() {
            outClassStart.push_back(outMember.size());
        };
        const auto closePartition = [&]() {
            outPartStart.push_back(outClassStart.size() - 1);
        };
        for (int32_t k = 0; k < n; ++k) { outMember.push_back(k); closeClass(); }
        closePartition();
        for (int32_t i = 0; i < n; ++i)
            for (int32_t j = i + 1; j < n; ++j) {
                for (int32_t k = 0; k < n; ++k) {
                    if (k == j) continue;
                    if (k == i) {
                        int32_t buf[2] = { i, j };
                        std::sort(buf, buf + 2, idxLess);
                        outMember.push_back(buf[0]);
                        outMember.push_back(buf[1]);
                    } else {
                        outMember.push_back(k);
                    }
                    closeClass();
                }
                closePartition();
            }
        for (int32_t i = 0; i < n; ++i)
            for (int32_t j = i + 1; j < n; ++j)
                for (int32_t l = j + 1; l < n; ++l) {
                    for (int32_t k = 0; k < n; ++k) {
                        if (k == j || k == l) continue;
                        if (k == i) {
                            int32_t buf[3] = { i, j, l };
                            std::sort(buf, buf + 3, idxLess);
                            outMember.push_back(buf[0]);
                            outMember.push_back(buf[1]);
                            outMember.push_back(buf[2]);
                        } else {
                            outMember.push_back(k);
                        }
                        closeClass();
                    }
                    closePartition();
                }
        return;
    }

    // 1 <= n <= cap: iterative Bell enumeration on two ping-pong CSR buffers.
    // The heap recursion peels elements[0] and recurses on the rest, so the
    // outermost peel is applied LAST; expanding indices n-1 .. 0 reproduces the
    // recursion's Option-1-then-Option-2 emission order byte-for-byte.
    DirtyState dA = DirtyState::Clean, dB = DirtyState::Clean;
    PagedVector<int32_t> memA(&arena, &dA), csA(&arena, &dA), psA(&arena, &dA);
    PagedVector<int32_t> memB(&arena, &dB), csB(&arena, &dB), psB(&arena, &dB);
    PagedVector<int32_t>* curMem = &memA; PagedVector<int32_t>* curCs = &csA;
    PagedVector<int32_t>* curPs = &psA;
    PagedVector<int32_t>* nxtMem = &memB; PagedVector<int32_t>* nxtCs = &csB;
    PagedVector<int32_t>* nxtPs = &psB;

    // current = [ [] ] : one empty partition.
    curCs->push_back(0);
    curPs->push_back(0);
    curPs->push_back(0);

    for (int32_t idx = n - 1; idx >= 0; --idx) {
        nxtMem->clear(); nxtCs->clear(); nxtPs->clear();
        nxtCs->push_back(0);
        nxtPs->push_back(0);
        const int32_t P = curPs->size() - 1;

        const auto copyClass = [&](int32_t c) {
            const int32_t mBegin = (*curCs)[c];
            const int32_t mEnd = (*curCs)[c + 1];
            for (int32_t m = mBegin; m < mEnd; ++m)
                nxtMem->push_back((*curMem)[m]);
            nxtCs->push_back(nxtMem->size());
        };

        for (int32_t p = 0; p < P; ++p) {
            const int32_t cBegin = (*curPs)[p];
            const int32_t cEnd = (*curPs)[p + 1];

            // Option 1: copy sp's classes, then a new singleton {idx}.
            for (int32_t c = cBegin; c < cEnd; ++c) copyClass(c);
            nxtMem->push_back(idx);
            nxtCs->push_back(nxtMem->size());
            nxtPs->push_back(nxtCs->size() - 1);

            // Option 2: for each class iSel, copy sp with idx added to iSel,
            // that class re-sorted by string.
            for (int32_t iSel = cBegin; iSel < cEnd; ++iSel) {
                for (int32_t c = cBegin; c < cEnd; ++c) {
                    if (c == iSel) {
                        int32_t buf[kMaxClassBuf];
                        int32_t bn = 0;
                        const int32_t mBegin = (*curCs)[c];
                        const int32_t mEnd = (*curCs)[c + 1];
                        for (int32_t m = mBegin; m < mEnd; ++m) {
                            assert(bn < kMaxClassBuf
                                && "generateSetPartitions: class exceeds "
                                   "kMaxClassBuf (Rule-19 tripwire)");
                            buf[bn++] = (*curMem)[m];
                        }
                        assert(bn < kMaxClassBuf);
                        buf[bn++] = idx;
                        std::sort(buf, buf + bn, idxLess);
                        for (int32_t b = 0; b < bn; ++b)
                            nxtMem->push_back(buf[b]);
                        nxtCs->push_back(nxtMem->size());
                    } else {
                        copyClass(c);
                    }
                }
                nxtPs->push_back(nxtCs->size() - 1);
            }
        }
        std::swap(curMem, nxtMem);
        std::swap(curCs, nxtCs);
        std::swap(curPs, nxtPs);
    }

    // Copy the final current CSR into the caller's output columns.
    for (int32_t i = 0; i < curMem->size(); ++i)
        outMember.push_back((*curMem)[i]);
    for (int32_t i = 0; i < curCs->size(); ++i)
        outClassStart.push_back((*curCs)[i]);
    for (int32_t i = 0; i < curPs->size(); ++i)
        outPartStart.push_back((*curPs)[i]);
}
}
