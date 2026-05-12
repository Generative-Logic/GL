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

/// @file msvc_sort.hpp
/// @brief Local re-implementation of MSVC STL's `_Sort_unchecked` (the
///        engine behind `std::sort` on Win11). Provides `gl::msvc_sort`
///        as a comparator-driven byte-identical replacement so the
///        prover's hot-path sort sites produce the same output on
///        every host regardless of the underlying STL implementation.
///
/// @details
/// The standard library's `std::sort` is allowed to be unstable, and
/// MSVC STL (`_Sort_unchecked`) and libstdc++ (`__introsort_loop`)
/// resolve ties using different pivot strategies, cutoff thresholds,
/// and recursion-budget formulas. For inputs with many equivalent
/// elements the two implementations produce visibly different byte
/// output even given identical input. GL's prover-pipeline behavior
/// depends on the post-sort order of `IntEncodedExpr` pointers in
/// `generateEncodedRequests*` because the rule-firing order through
/// `checkLocalEncodedMemoryStatic` is sensitive to it; cross-host byte
/// determinism therefore requires a sort algorithm whose output is a
/// pure function of `(input, comparator)` — independent of which
/// standard library is in use.
///
/// `gl::msvc_sort` is that algorithm: a faithful re-implementation of
/// MSVC's `_Sort_unchecked` (introsort dispatcher) using helpers
/// translated line-by-line from `microsoft/STL/stl/inc/algorithm`:
/// `_Insertion_sort_unchecked`, `_Med3_unchecked`,
/// `_Guess_median_unchecked` (Tukey's ninther for ranges > 40),
/// `_Partition_by_pivot_unchecked` (3-way Hoare partition that returns
/// a `(lt, eq_end)` boundary pair), `_Partition_by_median_guess_unchecked`,
/// and the `_Sort_unchecked` driver itself.
///
/// **Stability.** Like the original, `gl::msvc_sort` is **unstable** —
/// equivalent elements may be reordered within a tie group. The trade
/// is intentional: it matches what Win11 `std::sort` did historically
/// on this codebase, so swapping `std::stable_sort` for `gl::msvc_sort`
/// preserves whatever Win11 main HEAD behavior the prover was built
/// around, instead of switching to emergence-order semantics.
///
/// **What we DO NOT replicate.** The MSVC implementation uses
/// `_DEBUG_LT_PRED` for iterator-debug-mode comparator-symmetry checks
/// and `_Move_backward_unchecked` for a slightly-different move-vs-copy
/// dispatch. Both are inert in release builds (assertion-only) and
/// have no effect on the sorted output, so we drop them.
///
/// **Heap-sort fallback.** When the recursion budget `_Ideal` is
/// exhausted MSVC switches to `_Make_heap_unchecked` +
/// `_Sort_heap_unchecked`. Those use MSVC-internal sift-up /
/// sift-down primitives. We re-implement them inline (`make_heap_impl`,
/// `sort_heap_impl`, `sift_down_hole`, `sift_up_hole`) to avoid leaking
/// libstdc++'s heap implementation back into the path. For
/// 8192-element inputs the budget is exhausted only on pathological
/// (already-bad-partitioned) data; for the prover's actual `filteredIdx`
/// inputs the path is dominated by partition + insertion-sort.
///
/// @see microsoft/STL/stl/inc/algorithm — the upstream implementation
///      these functions translate.

#include <cstddef>
#include <iterator>
#include <utility>

namespace gl {
namespace msvc_sort_impl {

/// @brief Insertion-sort threshold copied verbatim from MSVC's
///        `_ISORT_MAX`. Any range of this size or smaller bottoms out
///        in insertion sort instead of recursing further into
///        quicksort.
constexpr int ISORT_MAX = 32;

/// @brief In-place insertion sort over `[first, last)`.
///
/// @details Translation of MSVC's `_Insertion_sort_unchecked`. Walks
/// the range left-to-right, lifting each element to its final position
/// by moving larger predecessors one slot right. With a strict-weak-
/// ordering comparator that returns `false` on equivalent inputs this
/// is incidentally stable (ties are never swapped), which matches
/// MSVC's behavior for small subarrays — important because introsort
/// hands off to insertion sort once a subarray shrinks to `ISORT_MAX`,
/// so any small tie cluster that ended up in a small subarray
/// preserves its emergence order in both this re-implementation and
/// the upstream MSVC code.
template <class BidIt, class Pr>
inline void insertion_sort(BidIt first, BidIt last, Pr pred) {
    if (first == last) return;
    for (BidIt mid = first; ++mid != last;) {
        BidIt hole = mid;
        typename std::iterator_traits<BidIt>::value_type val = std::move(*mid);

        if (pred(val, *first)) {
            // Found a new earliest element — slide the entire prefix
            // one position right and drop val at the front.
            std::move_backward(first, mid, ++hole);
            *first = std::move(val);
        } else {
            // Walk backward, moving each strictly-greater predecessor
            // forward by one slot, until we reach a predecessor that
            // is not strictly greater than val.
            for (BidIt prev = hole; pred(val, *--prev); hole = prev) {
                *hole = std::move(*prev);
            }
            *hole = std::move(val);
        }
    }
}

/// @brief Median-of-three pivot helper. Sorts `*first`, `*mid`,
///        `*last` in place so that `*mid` ends up holding the median.
///
/// @details Translation of MSVC's `_Med3_unchecked`. Uses a
/// three-comparison branching ladder; the order of swaps is what
/// MSVC's std::sort would observe, so cross-host outputs match for
/// ranges that exercise this code path.
template <class RanIt, class Pr>
inline void med3(RanIt first, RanIt mid, RanIt last, Pr pred) {
    using std::iter_swap;
    if (pred(*mid, *first)) iter_swap(mid, first);
    if (pred(*last, *mid)) {
        iter_swap(last, mid);
        if (pred(*mid, *first)) iter_swap(mid, first);
    }
}

/// @brief Pivot selector. For ranges of size > 40 picks the median of
///        nine candidates ("Tukey's ninther"); otherwise the median of
///        three. The chosen median lands at `*mid`.
///
/// @details Translation of MSVC's `_Guess_median_unchecked`. The step
/// computation `(count + 1) >> 3` and the four nested `med3` calls are
/// byte-identical to the upstream version.
template <class RanIt, class Pr>
inline void guess_median(RanIt first, RanIt mid, RanIt last, Pr pred) {
    using Diff = typename std::iterator_traits<RanIt>::difference_type;
    const Diff count = last - first;
    if (40 < count) {
        const Diff step = (count + 1) >> 3;
        const Diff two_step = step << 1;
        med3(first,                first + step,       first + two_step, pred);
        med3(mid - step,           mid,                mid + step,        pred);
        med3(last - two_step,      last - step,        last,              pred);
        med3(first + step,         mid,                last - step,       pred);
    } else {
        med3(first, mid, last, pred);
    }
}

/// @brief Three-way Hoare partition. Given a pivot at `*pfirst`,
///        partitions `[first, last)` into `[first, lt)` (strictly less),
///        `[lt, eq_end)` (equivalent to pivot), `[eq_end, last)`
///        (strictly greater). Returns `(lt, eq_end)`.
///
/// @details Translation of MSVC's `_Partition_by_pivot_unchecked`. The
/// algorithm scans outward from the pivot, growing an equal-to-pivot
/// run while pushing strictly-less elements to the left and
/// strictly-greater elements to the right; the pivot itself rotates
/// along with the boundary. Loop structure (including the inner
/// branches' `continue` / `break` / `swap+increment` order) is
/// preserved verbatim so MSVC and this implementation make the same
/// sequence of swaps given identical input.
template <class RanIt, class Pr>
inline std::pair<RanIt, RanIt>
partition_by_pivot(RanIt first, RanIt pfirst, RanIt last, Pr pred) {
    using std::iter_swap;
    RanIt plast = pfirst;
    ++plast;

    while (first < pfirst && !pred(*(pfirst - 1), *pfirst) && !pred(*pfirst, *(pfirst - 1))) {
        --pfirst;
    }

    while (plast < last && !pred(*plast, *pfirst) && !pred(*pfirst, *plast)) {
        ++plast;
    }

    RanIt gfirst = plast;
    RanIt glast  = pfirst;

    for (;;) {
        for (; gfirst < last; ++gfirst) {
            if (pred(*pfirst, *gfirst)) {
                continue;
            } else if (pred(*gfirst, *pfirst)) {
                break;
            } else if (plast != gfirst) {
                iter_swap(plast, gfirst);
                ++plast;
            } else {
                ++plast;
            }
        }

        for (; first < glast; --glast) {
            const RanIt glast_prev = glast - 1;
            if (pred(*glast_prev, *pfirst)) {
                continue;
            } else if (pred(*pfirst, *glast_prev)) {
                break;
            } else if (--pfirst != glast_prev) {
                iter_swap(pfirst, glast_prev);
            }
        }

        if (glast == first && gfirst == last) {
            return std::pair<RanIt, RanIt>(pfirst, plast);
        }

        if (glast == first) {
            // No room at the bottom — rotate the pivot block upward.
            if (plast != gfirst) {
                iter_swap(pfirst, plast);
            }
            ++plast;
            iter_swap(pfirst, gfirst);
            ++pfirst;
            ++gfirst;
        } else if (gfirst == last) {
            // No room at the top — rotate the pivot block downward.
            if (--glast != --pfirst) {
                iter_swap(glast, pfirst);
            }
            iter_swap(pfirst, --plast);
        } else {
            iter_swap(gfirst, --glast);
            ++gfirst;
        }
    }
}

/// @brief Top-level partition entry — picks the pivot via
///        `guess_median` and delegates to `partition_by_pivot`.
///
/// @details Translation of MSVC's `_Partition_by_median_guess_unchecked`.
/// The pivot lands at `mid = first + ((last - first) >> 1)`; the
/// signed-shift form is preserved to match codegen between this
/// translation and the upstream.
template <class RanIt, class Pr>
inline std::pair<RanIt, RanIt>
partition_by_median_guess(RanIt first, RanIt last, Pr pred) {
    RanIt mid = first + ((last - first) >> 1);
    guess_median(first, mid, last - 1, pred);
    return partition_by_pivot(first, mid, last, pred);
}

// -------------------- heap-sort fallback --------------------

/// @brief Sift-down (heapify) starting at `hole`, with `count` total
///        elements and `val` carried by value.
///
/// @details Classic max-heap sift down: while `hole` has a larger
/// child, move the larger child up to `hole`, hole := that child.
/// The "carried" element `val` is dropped into the final hole at the
/// end. Matches MSVC's sift order: the right child is preferred when
/// both children are present and the right child is not less than the
/// left child (i.e. on ties the right child wins). The choice has no
/// effect on the sorted output produced by `sort_heap` (any valid heap
/// pop-order produces the same sorted sequence) but matters if heap
/// state were ever inspected mid-sort; we preserve MSVC's choice for
/// completeness.
template <class RanIt, class Diff, class Val, class Pr>
inline void sift_down_hole(RanIt first, Diff hole, Diff count, Val&& val, Pr pred) {
    Diff top = hole;
    Diff idx = 2 * hole + 2; // right child of hole
    while (idx < count) {
        if (pred(*(first + idx), *(first + idx - 1))) {
            --idx; // left child is the larger of the two
        }
        *(first + hole) = std::move(*(first + idx));
        hole = idx;
        idx = 2 * idx + 2;
    }
    if (idx == count) { // only-left-child case
        --idx;
        *(first + hole) = std::move(*(first + idx));
        hole = idx;
    }
    // Sift `val` upward from `hole` until heap-order holds.
    Diff parent = (hole - 1) / 2;
    while (hole > top && pred(*(first + parent), val)) {
        *(first + hole) = std::move(*(first + parent));
        hole = parent;
        parent = (hole - 1) / 2;
    }
    *(first + hole) = std::forward<Val>(val);
}

/// @brief Heapify the range `[first, first + count)` into a max-heap
///        rooted at `*first` under `pred` (a "less" comparator).
///
/// @details Standard bottom-up heapify: starts at the last internal
/// node and works toward the root, calling `sift_down_hole` at each
/// position. The result is identical to MSVC's `_Make_heap_unchecked`
/// for the same `(input, pred)` pair.
template <class RanIt, class Pr>
inline void make_heap_impl(RanIt first, RanIt last, Pr pred) {
    using Diff = typename std::iterator_traits<RanIt>::difference_type;
    const Diff count = last - first;
    if (count < 2) return;
    for (Diff hole = (count - 2) / 2; hole >= 0; --hole) {
        typename std::iterator_traits<RanIt>::value_type val = std::move(*(first + hole));
        sift_down_hole(first, hole, count, std::move(val), pred);
        if (hole == 0) break; // unsigned-safe terminator
    }
}

/// @brief Sort a max-heap into ascending order by repeated `pop_heap`.
///
/// @details Each iteration moves the root (largest element) to the
/// current `last - 1` position, then re-heapifies the remaining
/// `[first, last - 1)`. Matches MSVC's `_Sort_heap_unchecked` loop.
template <class RanIt, class Pr>
inline void sort_heap_impl(RanIt first, RanIt last, Pr pred) {
    using Diff = typename std::iterator_traits<RanIt>::difference_type;
    using std::iter_swap;
    for (; last - first >= 2; --last) {
        // pop_heap: swap root with last-1, then sift down on the
        // smaller range. Carry the swapped value so the sift_down's
        // final placement step has the right `val`.
        typename std::iterator_traits<RanIt>::value_type val = std::move(*(last - 1));
        *(last - 1) = std::move(*first);
        sift_down_hole(first, Diff{0}, Diff{(last - 1) - first}, std::move(val), pred);
    }
}

/// @brief The introsort driver. Identical control flow to MSVC's
///        `_Sort_unchecked`: insertion-sort cutoff, heap-sort fallback
///        when the recursion budget is exhausted, otherwise partition
///        + tail-call on the larger half.
///
/// @details `ideal` is the recursion budget. It decays by a factor of
/// `3/4` (`(ideal >> 1) + (ideal >> 2)`) at each level; when it
/// reaches zero we switch to heap sort to bound the worst-case depth
/// (the algorithm's defining property). The `ideal` starting value is
/// `last - first` (N) and the decay law matches MSVC byte-for-byte.
template <class RanIt, class Pr>
inline void sort_driver(RanIt first, RanIt last,
                         typename std::iterator_traits<RanIt>::difference_type ideal,
                         Pr pred) {
    for (;;) {
        if (last - first <= ISORT_MAX) {
            insertion_sort(first, last, pred);
            return;
        }
        if (ideal <= 0) {
            make_heap_impl(first, last, pred);
            sort_heap_impl(first, last, pred);
            return;
        }
        auto mid = partition_by_median_guess(first, last, pred);
        ideal = (ideal >> 1) + (ideal >> 2);
        if (mid.first - first < last - mid.second) {
            sort_driver(first, mid.first, ideal, pred);
            first = mid.second;
        } else {
            sort_driver(mid.second, last, ideal, pred);
            last = mid.first;
        }
    }
}

} // namespace msvc_sort_impl

/// @brief MSVC-compatible introsort over `[first, last)` with the
///        given strict-weak-ordering predicate `pred`.
///
/// @details Drop-in replacement for `std::sort(first, last, pred)`
/// whose output bytes match what MSVC STL's `std::sort` would produce
/// on the same input, on any platform whose toolchain has C++17 and a
/// random-access iterator interface. See the header banner above for
/// the rationale.
template <class RanIt, class Pr>
inline void msvc_sort(RanIt first, RanIt last, Pr pred) {
    msvc_sort_impl::sort_driver(first, last, last - first, pred);
}

} // namespace gl
