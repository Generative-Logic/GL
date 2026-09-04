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

#include "scratch_arena.hpp"
#include "scratch_string.hpp"
#include "sealed_pages.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <string>

namespace gl {

    /// @brief Non-owning length-carrying view — the common parameter type
    ///        of the fixed-length string operations.
    ///
    /// @details
    /// A `StrSpan` is the read-side denominator over every string
    /// representation the prover touches: hot strings, sealed staging
    /// strings, `std::string` at the boundaries, and (once the cold side
    /// lands) cold table views. It carries NO lifetime machinery of its
    /// own — constructing a span from a `ScratchString` / `SealedString` runs
    /// that type's liveness assert once, and the span then inherits the
    /// source's lifetime. Spans are stack-and-parameter objects; storing
    /// one anywhere persistent is the same bug as storing the underlying
    /// view (I-116).
    ///
    /// @invariant `ptr` is readable for exactly `len` bytes for the
    ///            lifetime of the SOURCE the span was built from.
    struct StrSpan {
        const char* ptr = nullptr;
        int32_t len = 0;

        /// @brief The empty span.
        StrSpan() = default;

        /// @brief View over raw bytes.
        ///
        /// @param p Byte pointer; may be null only when `n` is 0.
        /// @param n Byte count; >= 0.
        StrSpan(const char* p, int32_t n) : ptr(p), len(n) {
            assert(n >= 0);
            assert(n == 0 || p != nullptr);
        }

        /// @brief View over a `std::string` (boundary sites and tests).
        ///
        /// @param s Source string; the span dangles if `s` dies first.
        StrSpan(const std::string& s)
            : ptr(s.data()), len(static_cast<int32_t>(s.size())) {}

        /// @brief View over a `ScratchString`; runs its liveness assert once.
        ///
        /// @param h Live hot string.
        StrSpan(const ScratchString& h)
            : ptr(h.empty() ? nullptr : h.data()), len(h.size()) {}

        /// @brief View over a `SealedString`; runs its liveness assert
        ///        once.
        ///
        /// @param s Live sealed string.
        StrSpan(const SealedString& s)
            : ptr(s.empty() ? nullptr : s.data()), len(s.size()) {}

        /// @brief Whether the span is empty.
        ///
        /// @return `true` when the length is 0.
        bool empty() const { return len == 0; }

        /// @brief Single-byte read with bounds check.
        ///
        /// @param i Byte index, `0 <= i < len`.
        /// @return The byte at `i`.
        char operator[](int32_t i) const {
            assert(i >= 0 && i < len);
            return ptr[i];
        }

        /// @brief Owned heap copy — the std::string boundary.
        ///
        /// @return Heap copy of the bytes.
        std::string toStdString() const {
            return len == 0 ? std::string()
                            : std::string(ptr, static_cast<size_t>(len));
        }
    };

    /// @brief Byte equality of two spans.
    ///
    /// @param a Left span.
    /// @param b Right span.
    /// @return `true` when lengths and bytes match.
    inline bool equalSpans(const StrSpan& a, const StrSpan& b) {
        if (a.len != b.len) return false;
        if (a.len == 0) return true;
        return std::memcmp(a.ptr, b.ptr, static_cast<size_t>(a.len)) == 0;
    }

    /// @brief Byte-equality operator — the canonical `equalSpans`, so a span is
    ///        a first-class key in the generic cold-map erase predicate
    ///        (`kd == k`, shared with the POD-key path).
    ///
    /// @param a Left span.
    /// @param b Right span.
    /// @return `true` when lengths and bytes match.
    inline bool operator==(const StrSpan& a, const StrSpan& b) {
        return equalSpans(a, b);
    }

    /// @brief Lexicographic three-way comparison, byte order — identical
    ///        ordering to `std::string::compare` on the same bytes.
    ///
    /// @details
    /// The ordering primitive behind every decoded-lex walk and sort
    /// comparator the migration touches; matching `std::string` ordering
    /// exactly is what keeps dump sections and canonical selections
    /// byte-identical through the migration.
    ///
    /// @param a Left span.
    /// @param b Right span.
    /// @return Negative when `a < b`, 0 when equal, positive when
    ///         `a > b`.
    inline int compareSpans(const StrSpan& a, const StrSpan& b) {
        const int32_t common = a.len < b.len ? a.len : b.len;
        if (common > 0) {
            const int c = std::memcmp(a.ptr, b.ptr,
                                      static_cast<size_t>(common));
            if (c != 0) return c;
        }
        return a.len - b.len;
    }

    /// @brief FNV-1a 64-bit hash of a span — the in-house deterministic
    ///        hash (same algorithm family as the deload file naming;
    ///        NEVER `std::hash`, whose value is implementation-defined).
    ///
    /// @param s Span to hash.
    /// @return 64-bit FNV-1a digest; equal spans hash equal across runs,
    ///         hosts, and compilers.
    inline uint64_t hashSpan(const StrSpan& s) {
        uint64_t h = 14695981039346656037ULL;
        for (int32_t i = 0; i < s.len; ++i) {
            h ^= static_cast<unsigned char>(s.ptr[i]);
            h *= 1099511628211ULL;
        }
        return h;
    }

    /// @brief Substring containment over spans — the zero-allocation twin
    ///        of `std::string::find(...) != npos`.
    ///
    /// @param haystack Span searched in.
    /// @param needle   Span searched for; an empty needle is contained
    ///                 (matching `std::string::find("") == 0`).
    /// @return `true` iff `needle`'s bytes occur in `haystack`.
    inline bool containsSpan(const StrSpan& haystack, const StrSpan& needle) {
        if (needle.len == 0) return true;
        if (haystack.len < needle.len) return false;
        const int32_t last = haystack.len - needle.len;
        for (int32_t i = 0; i <= last; ++i) {
            if (haystack.ptr[i] == needle.ptr[0]
                && std::memcmp(haystack.ptr + i, needle.ptr,
                               static_cast<size_t>(needle.len)) == 0) {
                return true;
            }
        }
        return false;
    }

    /// @brief Prefix test over a span — the zero-allocation twin of the
    ///        `startsWith(const std::string&, const char*, n)` helper used
    ///        across the prover.
    ///
    /// @details
    /// Byte-exact prefix check: `true` when `s` is at least `n` bytes long
    /// and its first `n` bytes equal `pfx[0..n)`. This reproduces the
    /// heap-string `startsWith` decision (`s.size() >= n` then per-byte
    /// compare) without materializing a `std::string` — the caller passes a
    /// span aliasing stable bytes (an interner `decodeView`, an
    /// `exprKeyView`), so an `exprKey().rfind(pre, 0) == 0`-style probe never
    /// touches the heap. An empty prefix (`n == 0`) always matches, exactly
    /// as `std::string::compare(0, 0, "")` does.
    ///
    /// @param s   Span to test; may alias any stable byte source.
    /// @param pfx Prefix bytes; must point to at least `n` bytes.
    /// @param n   Prefix length in bytes; `>= 0`.
    /// @return `true` iff `s` begins with `pfx[0..n)`.
    /// @see containsSpan; findSpanFrom.
    inline bool startsWithSpan(const StrSpan& s, const char* pfx,
                               std::size_t n) {
        if (static_cast<std::size_t>(s.len) < n) return false;
        return n == 0 || std::memcmp(s.ptr, pfx, n) == 0;
    }

    /// @brief Position of the first occurrence of `needle` at or after
    ///        `fromPos` — the span twin of `std::string::find(needle, fromPos)`.
    ///
    /// @details
    /// Byte-exact position twin of the std form: the smallest `i >= fromPos`
    /// with `i + needle.len <= haystack.len` and the bytes equal, else the
    /// defined miss `-1` (the span world's `npos`, like `NameMap::lookup`'s
    /// 0). A `fromPos` at or past the last viable start — including
    /// `fromPos == haystack.len` and beyond — misses, exactly as the std
    /// form returns `npos` there. Plain forward scan with a first-byte
    /// pre-test + `memcmp` per candidate (the `containsSpan` shape); the
    /// searched strings are tiny scope names, so no fancier algorithm is
    /// warranted.
    ///
    /// DELIBERATE divergence from the std contract: an empty needle ASSERTS
    /// instead of returning `min(fromPos, haystack.len)`. Every caller
    /// passes a literal marker; an empty needle here is a caller bug the
    /// std form would mask as a hit at `fromPos` (Rule 19).
    ///
    /// @param haystack Span searched in.
    /// @param needle   Span searched for; must be non-empty (asserted).
    /// @param fromPos  First candidate start; `>= 0` (asserted). May exceed
    ///                 `haystack.len` (a defined miss).
    /// @return The smallest matching start `>= fromPos`, or `-1` when there
    ///         is none.
    /// @see std::string::find, rfindSpanBefore, containsSpan.
    inline int32_t findSpanFrom(const StrSpan& haystack, const StrSpan& needle,
                                int32_t fromPos) {
        assert(needle.len > 0
            && "findSpanFrom: empty needle is a caller bug (Rule 19)");
        assert(fromPos >= 0);
        const int32_t lastStart = haystack.len - needle.len;
        for (int32_t i = fromPos; i <= lastStart; ++i) {
            if (haystack.ptr[i] == needle.ptr[0]
                && std::memcmp(haystack.ptr + i, needle.ptr,
                               static_cast<size_t>(needle.len)) == 0) {
                return i;
            }
        }
        return -1;
    }

    /// @brief Position of the last occurrence of `needle` starting at or
    ///        before `pos` — the span twin of `std::string::rfind(needle, pos)`.
    ///
    /// @details
    /// Byte-exact position twin of the std form: the largest
    /// `i <= min(pos, haystack.len - needle.len)` with the bytes equal, else
    /// the defined miss `-1`. The clamp is exactly `std::string::rfind`'s —
    /// `pos` may be `haystack.len` (the "no pos" `rfind(needle)` form) or
    /// anything larger, and an occurrence starting exactly AT `pos` is
    /// accepted. A needle longer than the haystack misses. Plain backward
    /// scan with a first-byte pre-test + `memcmp` per candidate.
    ///
    /// DELIBERATE divergence from the std contract: an empty needle ASSERTS
    /// instead of returning `min(pos, haystack.len)` — same rationale as
    /// `findSpanFrom` (Rule 19).
    ///
    /// @param haystack Span searched in.
    /// @param needle   Span searched for; must be non-empty (asserted).
    /// @param pos      Last candidate start; `>= 0` (asserted). May exceed
    ///                 the last viable start (clamped, as in std).
    /// @return The largest matching start `<= pos` (post-clamp), or `-1`
    ///         when there is none.
    /// @see std::string::rfind, findSpanFrom, containsSpan.
    inline int32_t rfindSpanBefore(const StrSpan& haystack, const StrSpan& needle,
                                   int32_t pos) {
        assert(needle.len > 0
            && "rfindSpanBefore: empty needle is a caller bug (Rule 19)");
        assert(pos >= 0);
        const int32_t lastStart = haystack.len - needle.len;
        if (lastStart < 0) return -1;
        for (int32_t i = pos < lastStart ? pos : lastStart; i >= 0; --i) {
            if (haystack.ptr[i] == needle.ptr[0]
                && std::memcmp(haystack.ptr + i, needle.ptr,
                               static_cast<size_t>(needle.len)) == 0) {
                return i;
            }
        }
        return -1;
    }

    /// @brief Copy a span into the scratch arena.
    ///
    /// @param arena Scratch arena receiving the copy.
    /// @param s     Source span.
    /// @return Scratch view of the copy.
    inline ScratchString copyScratch(ScratchArena& arena, const StrSpan& s) {
        return ScratchString::copyFrom(arena, s.ptr, s.len);
    }

    /// @brief View twin of `ce::getArgs` — slice the `[...]` argument
    ///        list of a flat canonical MPL expression into spans, zero
    ///        copies.
    ///
    /// @details
    /// Byte-exact reproduction of the original's semantics, including its
    /// documented edges: NOT bracket-balanced (a nested `[`/`]`/`,`
    /// inside an argument mis-slices exactly like the original — callers
    /// pass flat expressions); a trailing comma yields a trailing empty
    /// span; an expression without a `[...]` block, or with an empty one,
    /// yields zero arguments. Output spans point INTO `expr` — they
    /// inherit its lifetime and cost nothing.
    ///
    /// @param expr    Flat canonical MPL expression text.
    /// @param out     Caller array receiving the argument spans.
    /// @param maxArgs Capacity of `out`; exceeding it asserts (silent
    ///                truncation would mis-parse, never tolerated).
    /// @return Number of arguments written.
    /// @pre  `expr` is flat MPL, as for `ce::getArgs`.
    inline int32_t getArgsSpans(const StrSpan& expr, StrSpan* out,
                                int32_t maxArgs) {
        assert(out != nullptr && maxArgs > 0);
        int32_t start = -1;
        for (int32_t i = 0; i < expr.len; ++i) {
            if (expr.ptr[i] == '[') { start = i; break; }
        }
        if (start < 0) return 0;
        int32_t end = -1;
        for (int32_t i = start; i < expr.len; ++i) {
            if (expr.ptr[i] == ']') { end = i; break; }
        }
        if (end < 0) return 0;
        const int32_t begin = start + 1;
        const int32_t subLen = end > begin ? end - begin : 0;
        if (subLen == 0) return 0;
        const char* sub = expr.ptr + begin;
        int32_t count = 0;
        int32_t pos = 0;
        while (pos <= subLen) {
            int32_t comma = -1;
            for (int32_t i = pos; i < subLen; ++i) {
                if (sub[i] == ',') { comma = i; break; }
            }
            assert(count < maxArgs
                && "getArgsSpans: argument count exceeds caller capacity");
            if (comma < 0) {
                out[count++] = StrSpan(sub + pos, subLen - pos);
                break;
            }
            out[count++] = StrSpan(sub + pos, comma - pos);
            pos = comma + 1;
        }
        return count;
    }

    /// @brief Whether a byte is a single ASCII decimal digit `0`–`9`.
    ///
    /// @param c The byte to test.
    /// @return `true` when `c` is in `'0'`..`'9'`.
    /// @see matchItLevId, matchIntLevId — the callers; this mirrors the
    ///      ECMAScript-grammar `\d` of the `std::regex` forms they replace.
    inline bool isDecimalDigitByte(char c) { return c >= '0' && c <= '9'; }

    /// @brief Parse a fixed-length run of ASCII decimal digits into a
    ///        non-negative `int`, refusing overflow.
    ///
    /// @details
    /// The byte-exact twin of `std::stoi` over a known digit run, but with no
    /// heap `std::string` and no thrown exception: every byte in `[p, p+n)`
    /// must be a decimal digit, and a value above `INT_MAX` returns `false`
    /// instead of throwing. The disintegration gates that call this wrapped the
    /// former `std::stoi` in `try { … } catch (...) {}` whose only effect on
    /// overflow was to leave the gate untriggered; a `false` return here
    /// reproduces that exactly (the caller skips the comparison), so the
    /// observable behaviour is byte-identical for every input — including the
    /// in-practice-unreachable over-long-digit case.
    ///
    /// @param p   Pointer to the first digit byte; null only when `n` is 0.
    /// @param n   Digit count; the run must be non-empty (`n >= 1`).
    /// @param out Receives the parsed value on success; untouched on failure.
    /// @return `true` and sets @p out when `[p, p+n)` is `n >= 1` digits whose
    ///         value fits `int`; `false` otherwise.
    /// @see matchItLevId, matchIntLevId.
    inline bool parseDigitRunToInt(const char* p, int32_t n, int& out) {
        if (n <= 0) return false;
        int v = 0;
        for (int32_t i = 0; i < n; ++i) {
            if (!isDecimalDigitByte(p[i])) return false;
            const int d = p[i] - '0';
            if (v > (2147483647 - d) / 10) return false;  // would overflow int
            v = v * 10 + d;
        }
        out = v;
        return true;
    }

    /// @brief Write the base-10 digits of a non-negative `int32_t` into a
    ///        caller buffer — the allocation-free twin of `std::to_string`.
    ///
    /// @details
    /// Emits exactly the bytes `std::to_string(v)` would produce for any
    /// `v >= 0` (the oracle its unit test compares against): no sign, no
    /// leading zeros (except the single `'0'` for `v == 0`), no terminator,
    /// no locale, no allocation. The caller sizes the buffer; the maximum
    /// for `int32_t` is 10 digits (`2147483647`). Negative input is a
    /// caller bug and ASSERTS (the marker-building callers count from 0).
    ///
    /// @param dst Destination buffer; writable for at least 10 bytes (or
    ///            the caller's known bound); non-null (asserted).
    /// @param v   The value to format; `>= 0` (asserted).
    /// @return The digit count written (1..10).
    /// @see std::to_string — the oracle; parseDigitRunToInt — the inverse
    ///      direction.
    inline int32_t writeDecimalDigits(char* dst, int32_t v) {
        assert(dst != nullptr);
        assert(v >= 0 && "writeDecimalDigits: negative input is a caller bug");
        if (v == 0) { dst[0] = '0'; return 1; }
        char tmp[10];
        int32_t n = 0;
        while (v > 0) {
            tmp[n++] = static_cast<char>('0' + v % 10);
            v /= 10;
        }
        for (int32_t i = 0; i < n; ++i) dst[i] = tmp[n - 1 - i];
        return n;
    }

    /// @brief Whole-string match of `it_<digits>_lev_<digits>_<digits>`,
    ///        returning the trailing two numbers.
    ///
    /// @details
    /// Heap-free, regex-free twin of the `std::regex` `^it_\d+_lev_(\d+)_(\d+)$`
    /// (with `std::smatch` captures) used by `disintegrateExpr2`'s forceDeep and
    /// Pass-A admission gates, and byte-identical to the `std::string` `rfind` /
    /// `find("_lev_")` / `substr` / `std::stoi` extraction Pass A performed after
    /// its membership match. The whole span must be exactly `it_`
    /// `<iter digits>` `_lev_` `<level digits>` `_` `<id digits>` with no
    /// trailing bytes; the leading iteration number is required but unused (it
    /// was an uncaptured `\d+`). @p level is the digit group immediately after
    /// `_lev_`; @p id is the final digit group. An over-`INT_MAX` digit run
    /// returns `false` (see `parseDigitRunToInt`), matching the original
    /// regex-matched-but-`stoi`-threw path that left the gate untriggered.
    ///
    /// @param s     Candidate argument span (a slice of a caller-stable buffer).
    /// @param level Receives the `_lev_` number on a match; untouched otherwise.
    /// @param id    Receives the final number on a match; untouched otherwise.
    /// @return `true` (and sets @p level / @p id) on a whole-string match;
    ///         `false` otherwise (including the disjoint `int_lev_*` shape).
    /// @see matchIntLevId — the sibling `int_lev_*` shape.
    /// @see parseDigitRunToInt — the overflow-safe digit parse.
    inline bool matchItLevId(const StrSpan& s, int& level, int& id) {
        const char* p = s.ptr;
        const int32_t n = s.len;
        if (n < 3 || p == nullptr) return false;
        if (p[0] != 'i' || p[1] != 't' || p[2] != '_') return false;
        int32_t i = 3;
        const int32_t iterStart = i;
        while (i < n && isDecimalDigitByte(p[i])) ++i;
        if (i == iterStart) return false;  // need >= 1 iteration digit
        if (i + 5 > n || std::memcmp(p + i, "_lev_", 5) != 0) return false;
        i += 5;
        const int32_t levStart = i;
        while (i < n && isDecimalDigitByte(p[i])) ++i;
        int levVal;
        if (!parseDigitRunToInt(p + levStart, i - levStart, levVal)) return false;
        if (i >= n || p[i] != '_') return false;
        ++i;
        const int32_t idStart = i;
        while (i < n && isDecimalDigitByte(p[i])) ++i;
        int idVal;
        if (!parseDigitRunToInt(p + idStart, i - idStart, idVal)) return false;
        if (i != n) return false;  // no trailing bytes (whole-string match)
        level = levVal;
        id = idVal;
        return true;
    }

    /// @brief Whole-string match of `int_lev_<digits>_<digits>`, returning the
    ///        two numbers.
    ///
    /// @details
    /// Heap-free, regex-free twin of the `std::regex` `^int_lev_\d+_\d+$` used by
    /// `disintegrateExpr2`'s Pass-A gate, and of the `substr`/`std::stoi`
    /// extraction that followed it. The whole span must be exactly `int_lev_`
    /// `<level digits>` `_` `<id digits>` with no trailing bytes. Disjoint from
    /// `matchItLevId` (the `it_` shape has `_` as its third byte, `int_lev_` has
    /// `t`). Overflow handling is identical to `matchItLevId`.
    ///
    /// @param s     Candidate argument span (a slice of a caller-stable buffer).
    /// @param level Receives the `_lev_` number on a match; untouched otherwise.
    /// @param id    Receives the final number on a match; untouched otherwise.
    /// @return `true` (and sets @p level / @p id) on a whole-string match;
    ///         `false` otherwise.
    /// @see matchItLevId — the sibling `it_*_lev_*` shape.
    inline bool matchIntLevId(const StrSpan& s, int& level, int& id) {
        const char* p = s.ptr;
        const int32_t n = s.len;
        if (n < 8 || p == nullptr) return false;
        if (std::memcmp(p, "int_lev_", 8) != 0) return false;
        int32_t i = 8;
        const int32_t levStart = i;
        while (i < n && isDecimalDigitByte(p[i])) ++i;
        int levVal;
        if (!parseDigitRunToInt(p + levStart, i - levStart, levVal)) return false;
        if (i >= n || p[i] != '_') return false;
        ++i;
        const int32_t idStart = i;
        while (i < n && isDecimalDigitByte(p[i])) ++i;
        int idVal;
        if (!parseDigitRunToInt(p + idStart, i - idStart, idVal)) return false;
        if (i != n) return false;  // no trailing bytes (whole-string match)
        level = levVal;
        id = idVal;
        return true;
    }

    /// @brief Occurrence match of `int_lev_<digits>_<digits>` starting AT
    ///        `pos` — the end offset of the match, or `-1`.
    ///
    /// @details
    /// The unanchored-occurrence core: at `pos`, require the literal
    /// `int_lev_`, then a maximal digit run (>= 1 digit,
    /// `isDecimalDigitByte`), then `'_'`, then a maximal digit run (>= 1);
    /// return the offset just past the second run. There is deliberately NO
    /// end-of-span requirement — anchoring is the delegating shape scanner's
    /// job (`isIntLevShape` requires the returned end to equal the span
    /// length). At a fixed start the regex `int_lev_\d+_\d+` has a UNIQUE
    /// viable extent: every literal segment (`int_lev_`, `_`) begins with a
    /// non-digit byte and every `\d+` is a maximal digit run bounded by
    /// non-digits, so shortening any `\d+` would place a digit where a
    /// non-digit literal must match — impossible — and the greedy parse
    /// either fails outright or ends exactly at the end of the LAST maximal
    /// digit run (nothing follows it in the pattern). Examples:
    /// `"int_lev_12_"` has NO match at 0 (second run empty; first-run
    /// backtrack cannot help); `"int_lev_1_2_3"` matches exactly
    /// `"int_lev_1_2"`. No value parse anywhere (arbitrarily long digit
    /// runs match), no allocation, no exception path.
    ///
    /// Why this exists beside the siblings: `matchIntLevId` PARSES the two
    /// values (with the documented over-`INT_MAX`-returns-`false`
    /// divergence from the regex verdict); `isIntLevShape` is the anchored
    /// whole-span verdict. This core is occurrence + extent — no values,
    /// no anchoring.
    ///
    /// @param s   Span scanned (a slice of a caller-stable buffer).
    /// @param pos Candidate start offset; `>= 0`.
    /// @return The end offset just past the match starting at `pos`, or
    ///         `-1` when no match starts there.
    /// @see scanIntLevOccurrences — the left-to-right driver;
    ///      matchItLevOccurrenceAt — the sibling `it_*_lev_*` core;
    ///      isIntLevShape, matchIntLevId, scanSpecialTokens (`memory.hpp`)
    ///      — the retained regex oracle.
    inline int32_t matchIntLevOccurrenceAt(const StrSpan& s, int32_t pos) {
        assert(pos >= 0);
        const char* p = s.ptr;
        const int32_t n = s.len;
        if (pos + 11 > n) return -1;   // "int_lev_D_D" minimum: 11 bytes
        if (std::memcmp(p + pos, "int_lev_", 8) != 0) return -1;
        int32_t i = pos + 8;
        const int32_t d1 = i;
        while (i < n && isDecimalDigitByte(p[i])) ++i;
        if (i == d1) return -1;                       // >= 1 digit in run 1
        if (i >= n || p[i] != '_') return -1;
        ++i;
        const int32_t d2 = i;
        while (i < n && isDecimalDigitByte(p[i])) ++i;
        if (i == d2) return -1;                       // >= 1 digit in run 2
        return i;
    }

    /// @brief Occurrence match of `it_<digits>_lev_<digits>_<digits>`
    ///        starting AT `pos` — the end offset of the match, or `-1`.
    ///
    /// @details
    /// The `it_*_lev_*` unanchored-occurrence core: at `pos`, require the
    /// literal `it_`, a maximal digit run (>= 1), the literal `_lev_`, a
    /// maximal digit run (>= 1), `'_'`, and a final maximal digit run
    /// (>= 1); return the offset just past the final run. Same
    /// unique-extent / no-viable-backtrack argument as
    /// `matchIntLevOccurrenceAt` (every literal segment — `it_`, `_lev_`,
    /// `_` — begins with a non-digit byte); no end-of-span requirement
    /// (`isItLevShape` anchors), no value parse, no allocation.
    ///
    /// @param s   Span scanned (a slice of a caller-stable buffer).
    /// @param pos Candidate start offset; `>= 0`.
    /// @return The end offset just past the match starting at `pos`, or
    ///         `-1` when no match starts there.
    /// @see scanItLevOccurrences — the left-to-right driver;
    ///      matchIntLevOccurrenceAt — the sibling `int_lev_*` core;
    ///      isItLevShape, matchItLevId, scanSpecialTokens (`memory.hpp`)
    ///      — the retained regex oracle.
    inline int32_t matchItLevOccurrenceAt(const StrSpan& s, int32_t pos) {
        assert(pos >= 0);
        const char* p = s.ptr;
        const int32_t n = s.len;
        if (pos + 12 > n) return -1;   // "it_D_lev_D_D" minimum: 12 bytes
        if (p[pos] != 'i' || p[pos + 1] != 't' || p[pos + 2] != '_') return -1;
        int32_t i = pos + 3;
        const int32_t d1 = i;
        while (i < n && isDecimalDigitByte(p[i])) ++i;
        if (i == d1) return -1;
        if (i + 5 > n || std::memcmp(p + i, "_lev_", 5) != 0) return -1;
        i += 5;
        const int32_t d2 = i;
        while (i < n && isDecimalDigitByte(p[i])) ++i;
        if (i == d2) return -1;
        if (i >= n || p[i] != '_') return -1;
        ++i;
        const int32_t d3 = i;
        while (i < n && isDecimalDigitByte(p[i])) ++i;
        if (i == d3) return -1;
        return i;
    }

    /// @brief Left-to-right, non-overlapping scan of every `int_lev_*`
    ///        occurrence — the heap-free twin of `std::sregex_iterator`
    ///        with the UNANCHORED pattern `int_lev_\d+_\d+`.
    ///
    /// @details
    /// Fires `sink(start, len)` per occurrence, in scan order, duplicates
    /// passed through. Byte-exact iterator-twin argument, four steps:
    /// 1. LEFTMOST — the regex iterator repeatedly finds the leftmost match
    ///    at-or-after its resume point; this driver tries every candidate
    ///    start ascending, so its first success IS the leftmost match.
    /// 2. UNIQUE EXTENT at a fixed start, no viable backtrack — see
    ///    `matchIntLevOccurrenceAt` (the core carries the argument).
    /// 3. NON-OVERLAPPING RESUME — the iterator resumes at the match END;
    ///    the driver sets `pos = e`, so a new match may start exactly at
    ///    `e` (back-to-back `"int_lev_1_2int_lev_3_4"` yields two tokens).
    /// 4. FAILED CANDIDATES ADVANCE BY ONE byte — a later match may start
    ///    inside a tested window (`"print_lev_3_4"` yields `"int_lev_3_4"`
    ///    at offset 2 — the substring fidelity `scanSpecialTokens`
    ///    documents).
    /// The loop guard (`pos + 11 <= s.len`) prunes starts that cannot fit
    /// the 11-byte minimum match — pure optimization, no verdict surface.
    ///
    /// @tparam Sink A callable `void(int32_t start, int32_t len)`.
    /// @param s    Span scanned (a slice of a caller-stable buffer).
    /// @param sink Per-occurrence callback, called in left-to-right order.
    /// @see matchIntLevOccurrenceAt — the per-start core;
    ///      scanItLevOccurrences — the sibling driver;
    ///      scanSpecialTokens (`memory.hpp`) — the retained regex oracle;
    ///      `EqClassNameCaches::tokensViewOf` (`memory.hpp`) — the consumer.
    template <typename Sink>
    inline void scanIntLevOccurrences(const StrSpan& s, Sink&& sink) {
        int32_t pos = 0;
        while (pos + 11 <= s.len) {
            const int32_t e = matchIntLevOccurrenceAt(s, pos);
            if (e < 0) { ++pos; continue; }
            sink(pos, e - pos);
            pos = e;
        }
    }

    /// @brief Left-to-right, non-overlapping scan of every `it_*_lev_*`
    ///        occurrence — the heap-free twin of `std::sregex_iterator`
    ///        with the UNANCHORED pattern `it_\d+_lev_\d+_\d+`.
    ///
    /// @details
    /// Fires `sink(start, len)` per occurrence, in scan order, duplicates
    /// passed through. Same four-step iterator-twin argument as
    /// `scanIntLevOccurrences` (leftmost / unique extent / resume-at-end /
    /// advance-by-one), with the 12-byte minimum (`it_D_lev_D_D`). The two
    /// scans are INDEPENDENT passes over the whole text, exactly as
    /// `scanSpecialTokens` runs its two `sregex_iterator` passes.
    ///
    /// @tparam Sink A callable `void(int32_t start, int32_t len)`.
    /// @param s    Span scanned (a slice of a caller-stable buffer).
    /// @param sink Per-occurrence callback, called in left-to-right order.
    /// @see matchItLevOccurrenceAt — the per-start core;
    ///      scanIntLevOccurrences — the sibling driver;
    ///      scanSpecialTokens (`memory.hpp`) — the retained regex oracle;
    ///      `EqClassNameCaches::tokensViewOf` (`memory.hpp`) — the consumer.
    template <typename Sink>
    inline void scanItLevOccurrences(const StrSpan& s, Sink&& sink) {
        int32_t pos = 0;
        while (pos + 12 <= s.len) {
            const int32_t e = matchItLevOccurrenceAt(s, pos);
            if (e < 0) { ++pos; continue; }
            sink(pos, e - pos);
            pos = e;
        }
    }

    /// @brief Whole-span SHAPE test of `int_lev_<digits>_<digits>` — the
    ///        heap-free, value-free lexical twin of the anchored regex
    ///        `int_lev_\d+_\d+`.
    ///
    /// @details
    /// Byte-identical verdict to `std::regex_match` with the anchored pattern
    /// `int_lev_\d+_\d+` for EVERY input: the whole span must be exactly
    /// `int_lev_` `<digits>` `_` `<digits>` with no trailing bytes
    /// (ECMAScript `\d` == `'0'..'9'` == `isDecimalDigitByte`). The pattern
    /// is BACKTRACK-FREE by construction — every literal segment
    /// (`int_lev_`, `_`) begins with a byte disjoint from `\d`, and every
    /// `\d+` is a maximal digit run bounded by non-digits on both sides — a
    /// greedy `\d+` can neither extend into a literal nor shrink to let a
    /// literal match inside a digit run, so this single left-to-right scan
    /// decides exactly what the regex engine decides, for every input,
    /// including digit runs exceeding any integer range: NO value parse
    /// anywhere, no allocation, no exception path.
    ///
    /// Why this exists beside `matchIntLevId`: that matcher PARSES the two
    /// values for callers that need them, with a documented
    /// over-`INT_MAX`-returns-`false` contract — a semantic divergence from
    /// the regex verdict, which classifies `int_lev_99999999999_1` as a
    /// match without parsing. Shape-only consumers (`classifyName`) must not
    /// inherit that divergence.
    ///
    /// The verdict is computed by delegation to the occurrence core: the
    /// anchored form is exactly "an occurrence starts at 0 AND ends at the
    /// span end" (`matchIntLevOccurrenceAt(s, 0) == s.len`) — the same
    /// left-to-right parse with the end-of-span requirement re-imposed
    /// here. An empty or too-short span falls out via the core's
    /// minimum-length guard (`-1` never equals a non-negative length).
    ///
    /// @param s Candidate name span (a slice of a caller-stable buffer).
    /// @return `true` iff the whole span has the `int_lev_<digits>_<digits>`
    ///         shape.
    /// @see matchIntLevOccurrenceAt — the occurrence core this delegates
    ///      to; matchIntLevId, matchItLevId, isDecimalDigitByte,
    ///      isItLevShape — the sibling `it_*_lev_*` shape;
    ///      `classifyName(const StrSpan&)` (`memory.hpp`) — the consumer.
    inline bool isIntLevShape(const StrSpan& s) {
        return matchIntLevOccurrenceAt(s, 0) == s.len;
    }

    /// @brief Whole-span SHAPE test of `it_<digits>_lev_<digits>_<digits>` —
    ///        the heap-free, value-free lexical twin of the anchored regex
    ///        `it_\d+_lev_\d+_\d+`.
    ///
    /// @details
    /// Byte-identical verdict to `std::regex_match` with the anchored pattern
    /// `it_\d+_lev_\d+_\d+` for EVERY input, by the same backtrack-free
    /// argument as `isIntLevShape`: every literal segment (`it_`, `_lev_`,
    /// `_`) begins with a byte disjoint from `\d`, and every `\d+` is a
    /// maximal digit run bounded by non-digits on both sides, so the single
    /// left-to-right scan decides exactly what the regex engine decides —
    /// including arbitrarily long digit runs (no value parse anywhere, no
    /// allocation, no exception path).
    ///
    /// Why this exists beside `matchItLevId`: that matcher parses the
    /// trailing two values for callers that need them, with the documented
    /// over-`INT_MAX`-returns-`false` divergence from the regex verdict;
    /// shape-only consumers (`classifyName`) must not inherit it.
    ///
    /// The verdict is computed by delegation to the occurrence core, as in
    /// `isIntLevShape`: anchored == "an occurrence starts at 0 AND ends at
    /// the span end" (`matchItLevOccurrenceAt(s, 0) == s.len`).
    ///
    /// @param s Candidate name span (a slice of a caller-stable buffer).
    /// @return `true` iff the whole span has the
    ///         `it_<digits>_lev_<digits>_<digits>` shape.
    /// @see matchItLevOccurrenceAt — the occurrence core this delegates
    ///      to; matchItLevId, matchIntLevId, isDecimalDigitByte,
    ///      isIntLevShape — the sibling `int_lev_*` shape;
    ///      `classifyName(const StrSpan&)` (`memory.hpp`) — the consumer.
    inline bool isItLevShape(const StrSpan& s) {
        return matchItLevOccurrenceAt(s, 0) == s.len;
    }

    /// @brief One `it_(\d+)_lev_\d+_` prefix-shape occurrence: the extent and
    ///        the FIRST digit run (the regex capture group 1).
    ///
    /// @details
    /// The result of `matchItLevPrefixOccurrenceAt`: `end` is the offset just
    /// past the match (inclusive of the trailing `'_'`), or `< 0` for no match;
    /// `d1Start` / `d1Len` bound the first digit run (the `(\d+)` capture the
    /// `extractMaxIterationNumber` regex reads via `atoi`).
    ///
    /// @see matchItLevPrefixOccurrenceAt, scanItLevPrefixOccurrences.
    struct ItLevPrefixMatch { int32_t end; int32_t d1Start; int32_t d1Len; };

    /// @brief The `it_(\d+)_lev_\d+_` unanchored-occurrence core — the
    ///        heap-free twin of a `std::sregex_iterator` step over that pattern.
    ///
    /// @details
    /// Distinct from `matchItLevOccurrenceAt` (`it_\d+_lev_\d+_\d+`): this
    /// pattern stops at the `'_'` AFTER the second digit run — there is NO
    /// trailing digit run — so its extent differs and the sibling scanner is
    /// NOT a drop-in. At `pos` it requires the literal `it_`, a maximal digit
    /// run (>= 1, the capture), the literal `_lev_`, a maximal digit run
    /// (>= 1), and a final `'_'`; it returns the offset just past that `'_'`
    /// plus the capture bounds.
    ///
    /// BACKTRACK-FREE by construction (same argument as
    /// `matchItLevOccurrenceAt`): every literal segment (`it_`, `_lev_`, the
    /// trailing `_`) begins with a byte disjoint from `\d`, and each `\d+` is a
    /// maximal digit run bounded by non-digits, so this single left-to-right
    /// scan decides exactly what the engine decides at this start, for every
    /// input — no value parse, no allocation. The `pos + 11 > n` guard is a
    /// pure minimum-length optimization (`it_1_lev_1_` = 11 bytes).
    ///
    /// @param s   Span scanned (a slice of a caller-stable buffer).
    /// @param pos Candidate start offset; `>= 0`.
    /// @return `{ end, d1Start, d1Len }`; `end < 0` when no match starts at
    ///         @p pos.
    /// @see scanItLevPrefixOccurrences — the left-to-right driver;
    ///      matchItLevOccurrenceAt — the sibling full `it_*_lev_*_*` core;
    ///      `scanSpecialTokens` (`memory.hpp`) — the retained regex oracle.
    inline ItLevPrefixMatch matchItLevPrefixOccurrenceAt(const StrSpan& s, int32_t pos) {
        assert(pos >= 0);
        const char* p = s.ptr;
        const int32_t n = s.len;
        if (pos + 11 > n) return { -1, 0, 0 };   // "it_D_lev_D_" minimum: 11 bytes
        if (p[pos] != 'i' || p[pos + 1] != 't' || p[pos + 2] != '_') return { -1, 0, 0 };
        int32_t i = pos + 3;
        const int32_t d1 = i;
        while (i < n && isDecimalDigitByte(p[i])) ++i;
        if (i == d1) return { -1, 0, 0 };
        const int32_t d1Len = i - d1;
        if (i + 5 > n || std::memcmp(p + i, "_lev_", 5) != 0) return { -1, 0, 0 };
        i += 5;
        const int32_t d2 = i;
        while (i < n && isDecimalDigitByte(p[i])) ++i;
        if (i == d2) return { -1, 0, 0 };
        if (i >= n || p[i] != '_') return { -1, 0, 0 };
        ++i;   // consume the trailing '_' — the match END (no trailing digit run)
        return { i, d1, d1Len };
    }

    /// @brief Left-to-right, non-overlapping scan of every `it_(\d+)_lev_\d+_`
    ///        prefix-shape occurrence — the heap-free twin of
    ///        `std::sregex_iterator` with that UNANCHORED pattern.
    ///
    /// @details
    /// Fires `sink(start, end, d1Start, d1Len)` per occurrence in scan order,
    /// duplicates passed through. Same four-step iterator-twin argument as
    /// `scanItLevOccurrences`:
    /// 1. LEFTMOST — the driver tries every candidate start ascending, so its
    ///    first success is the leftmost match.
    /// 2. UNIQUE EXTENT at a fixed start, no viable backtrack — see
    ///    `matchItLevPrefixOccurrenceAt`.
    /// 3. NON-OVERLAPPING RESUME at the match END (`pos = m.end`) — for the
    ///    full form `it_3_lev_4_5` the prefix match consumes `it_3_lev_4_` and
    ///    resume lands just before the trailing `5`.
    /// 4. FAILED CANDIDATES ADVANCE BY ONE byte.
    /// The `pos + 11 <= s.len` guard prunes starts too short to fit the 11-byte
    /// minimum — pure optimization, no verdict surface.
    ///
    /// @tparam Sink A callable
    ///         `void(int32_t start, int32_t end, int32_t d1Start, int32_t d1Len)`.
    /// @param s    Span scanned (a slice of a caller-stable buffer).
    /// @param sink Per-occurrence callback, called in left-to-right order.
    /// @see matchItLevPrefixOccurrenceAt — the per-start core;
    ///      `extractMaxIterationNumber` (`prover.hpp`) — the consumer;
    ///      `scanSpecialTokens` (`memory.hpp`) — the retained regex oracle.
    template <typename Sink>
    inline void scanItLevPrefixOccurrences(const StrSpan& s, Sink&& sink) {
        int32_t pos = 0;
        while (pos + 11 <= s.len) {
            const ItLevPrefixMatch m = matchItLevPrefixOccurrenceAt(s, pos);
            if (m.end < 0) { ++pos; continue; }
            sink(pos, m.end, m.d1Start, m.d1Len);
            pos = m.end;
        }
    }

    /// @brief Existence + single-distinct collection of `int_lev_\d+_\d+`
    ///        tokens — the heap-free twin of a `regex_search` gate followed by
    ///        a `std::set<std::string>` occurrence collection.
    ///
    /// @details
    /// Returns `0` when no occurrence exists; `1` when exactly ONE DISTINCT
    /// token occurs (its span written to @p outVar); `2` when two or more
    /// distinct tokens occur. Distinctness is byte equality (`equalSpans`) —
    /// the exact semantics of the former `std::set<std::string>` dedup.
    ///
    /// @p outVar is the FIRST occurrence's span, which equals `*set.begin()`
    /// (the lex-least) WHENEVER the verdict is `1`: if any later token differed
    /// the verdict would be `2` (set size != 1) and the caller never reads the
    /// value, so first-found == lex-least exactly when it is the sole distinct
    /// token. Duplicates of one token yield verdict `1`.
    ///
    /// The scan is TOTAL (no early exit): existence and distinct-count are
    /// monotone, so an early exit could not change the verdict, but running the
    /// full scan keeps the function trivially total (cheap on the short inputs
    /// this gates).
    ///
    /// @param s      Span scanned (a slice of a caller-stable buffer).
    /// @param outVar Receives the sole distinct token's span on verdict `1`;
    ///               untouched otherwise.
    /// @return `0` / `1` / `2` as above.
    /// @see scanIntLevOccurrences — the occurrence driver; `allowedForMail`
    ///      (`prover.cpp`) — the consumer; `scanSpecialTokens` (`memory.hpp`)
    ///      — the retained regex oracle.
    inline int scanSingleDistinctIntLev(const StrSpan& s, StrSpan& outVar) {
        StrSpan first;
        bool haveFirst = false;
        bool multi = false;
        scanIntLevOccurrences(s, [&](int32_t start, int32_t len) {
            const StrSpan tok(s.ptr + start, len);
            if (!haveFirst) { first = tok; haveFirst = true; return; }
            if (!equalSpans(tok, first)) multi = true;
        });
        if (!haveFirst) return 0;
        if (multi) return 2;
        outVar = first;
        return 1;
    }

    /// @brief Whether EVERY `int_lev_<level>_<id>` occurrence in the span
    ///        carries a level STRICTLY BELOW the given bound — the
    ///        parent-level mail-gate predicate.
    ///
    /// @details
    /// A witness name minted at logic-block level `k` embeds `k` as the first
    /// digit run of its `int_lev_k_id` token. Mail flows only ancestor to
    /// descendant, so a token whose level is strictly below an LB's own level
    /// can only have reached that LB by ancestor mail — and every ancestor
    /// deposit is pulled by ALL of the LB's descendants directly. Such names
    /// are therefore known below by construction, and an expression composed
    /// exclusively of them is safe to mail onward regardless of how many
    /// distinct tokens it carries (`allowedForMail`'s parent-level pass).
    ///
    /// The scan is TOTAL (no early exit), matching the sibling
    /// `scanSingleDistinctIntLev`'s convention. The level parse accumulates
    /// into 64 bits; a run whose value exceeds `INT32_MAX` fails the
    /// predicate conservatively (the caller falls through to the memo gates
    /// — a refusal is always sound on this path). Zero occurrences satisfy
    /// the predicate vacuously; the caller's no-token early exit fires first.
    ///
    /// @param s          Span scanned (a slice of a caller-stable buffer).
    /// @param levelBound Exclusive upper bound — the owning LB's `level`.
    /// @return `true` iff every `int_lev` occurrence's level is
    ///         `< levelBound`.
    /// @see scanIntLevOccurrences — the occurrence driver;
    ///      scanSingleDistinctIntLev — the verdict sibling;
    ///      `allowedForMail` (`prover.cpp`) — the consumer.
    inline bool allIntLevLevelsBelow(const StrSpan& s, int32_t levelBound) {
        bool allBelow = true;
        scanIntLevOccurrences(s, [&](int32_t start, int32_t len) {
            (void)len;
            int32_t i = start + 8;   // just past "int_lev_"; run 1 has >= 1 digit
            int64_t v = 0;
            while (i < s.len && isDecimalDigitByte(s.ptr[i])) {
                v = v * 10 + (s.ptr[i] - '0');
                if (v > INT32_MAX) { allBelow = false; return; }
                ++i;
            }
            if (v >= levelBound) allBelow = false;
        });
        return allBelow;
    }

    /// @brief Whether the span contains any `it_\d+_lev_\d+_` prefix-shape
    ///        occurrence — the heap-free twin of `std::regex_search` with that
    ///        pattern.
    ///
    /// @details
    /// Existence is order-free, so the scan returns `true` on the first hit
    /// (an early exit that cannot change the verdict). Byte-identical to the
    /// `regex_search` existence test for every input.
    ///
    /// @param s Span scanned (a slice of a caller-stable buffer).
    /// @return `true` iff some `it_\d+_lev_\d+_` occurrence exists.
    /// @see matchItLevPrefixOccurrenceAt; `isProved` (`prover.hpp`) — the
    ///      consumer; `scanSpecialTokens` (`memory.hpp`) — the retained oracle.
    inline bool containsItLevPrefixShape(const StrSpan& s) {
        int32_t pos = 0;
        while (pos + 11 <= s.len) {
            const ItLevPrefixMatch m = matchItLevPrefixOccurrenceAt(s, pos);
            if (m.end >= 0) return true;
            ++pos;
        }
        return false;
    }

    /// @brief Whether the span contains a `c` immediately followed by at least
    ///        one decimal digit — the heap-free twin of `std::regex_search`
    ///        with the pattern `c\d+`.
    ///
    /// @details
    /// `regex_search(c\d+)` succeeds iff some `'c'` is immediately followed by
    /// one or more digits, so the existence test is exactly `∃ i : s[i] == 'c'
    /// && isDecimalDigitByte(s[i+1])`. Byte-identical verdict for every input
    /// (e.g. `"c"` alone → false, `"ac9"` → true, `"c_1"` → false).
    ///
    /// @param s Span scanned (a slice of a caller-stable buffer).
    /// @return `true` iff a `c<digit>` pair exists.
    /// @see `isProved` (`prover.hpp`) — the consumer.
    inline bool containsCDigit(const StrSpan& s) {
        for (int32_t i = 0; i + 1 < s.len; ++i) {
            if (s.ptr[i] == 'c' && isDecimalDigitByte(s.ptr[i + 1])) return true;
        }
        return false;
    }

    /// @brief Collect every bracketed-argument token of an MPL expression — the
    ///        flat-scan twin of the `ce::parseExpr` + per-node `getArgs` token walk.
    ///
    /// @details
    /// Every variable token in a canonical MPL expression appears inside some
    /// `[...]` (a leaf's argument list, or a `>` / `!>` binder's bound vars), so
    /// scanning for every `[...]` and comma-splitting its content yields exactly the
    /// same token SET as walking the parse tree and `getArgs`-ing each node label —
    /// without building the tree. Empty args are skipped (mirroring the original's
    /// `find_first_not_of` whitespace guard on whitespace-free input). `sink(StrSpan)`
    /// fires per token; duplicates are passed through (the caller dedups or, like
    /// `replaceKeysScratch`/`replaceKeysToString`, tolerates same-key-same-value pairs).
    ///
    /// @tparam Sink A callable `void(const StrSpan&)`.
    /// @param expr The MPL expression span.
    /// @param sink Per-token callback.
    /// @see getArgsSpans
    template <typename Sink>
    inline void collectExprTokens(const StrSpan& expr, Sink sink) {
        int32_t i = 0;
        while (i < expr.len) {
            if (expr.ptr[i] != '[') { ++i; continue; }
            int32_t close = i + 1;
            while (close < expr.len && expr.ptr[close] != ']') ++close;
            int32_t start = i + 1;
            for (int32_t p = i + 1; p <= close; ++p) {
                if (p == close || expr.ptr[p] == ',') {
                    if (p > start) sink(StrSpan(expr.ptr + start, p - start));
                    start = p + 1;
                }
            }
            i = close + 1;
        }
    }

    /// @brief View twin of `ce::extractExpression` — the core expression
    ///        name of a canonical MPL form, as a slice of the input.
    ///
    /// @param s Canonical MPL expression text (`(name[...])`,
    ///          `!(name[...])`, or bare `name[...]`).
    /// @return Name span pointing into `s`; empty span when there is no
    ///         `[`.
    inline StrSpan extractExpressionSpan(const StrSpan& s) {
        int32_t index = -1;
        for (int32_t i = 0; i < s.len; ++i) {
            if (s.ptr[i] == '[') { index = i; break; }
        }
        if (index < 0) return StrSpan();
        if (s.len > 0 && s.ptr[0] == '(') {
            return StrSpan(s.ptr + 1, index - 1);
        }
        if (s.len >= 2 && s.ptr[0] == '!' && s.ptr[1] == '(') {
            return StrSpan(s.ptr + 2, index - 2);
        }
        return StrSpan(s.ptr, index);
    }

    /// @brief The core-name extractor for a possibly-negated canonical MPL
    ///        form — like `extractExpressionSpan` but with no bare-name
    ///        fallback. The sole production form; its heap counterpart lives
    ///        only as a test-local oracle (D-193).
    ///
    /// @param s Canonical MPL expression text, possibly negated.
    /// @return Name span pointing into `s`; empty span when the shape is
    ///         neither `(name[...])` nor `!(name[...])`.
    inline StrSpan extractExpressionUniversalSpan(const StrSpan& s) {
        int32_t index = -1;
        for (int32_t i = 0; i < s.len; ++i) {
            if (s.ptr[i] == '[') { index = i; break; }
        }
        if (index < 0) return StrSpan();
        if (s.len > 0 && s.ptr[0] == '(') {
            return StrSpan(s.ptr + 1, index - 1);
        }
        if (s.len >= 2 && s.ptr[0] == '!' && s.ptr[1] == '(') {
            return StrSpan(s.ptr + 2, index - 2);
        }
        return StrSpan();
    }

    /// @brief View twin of `ce::extractExpressionFromNegation` — the core
    ///        name of a `!(name[...])` form.
    ///
    /// @details
    /// Byte-exact including the original's lenient search: the `!(` is
    /// FOUND (not anchored at position 0) and must precede the first
    /// `[`.
    ///
    /// @param s Canonical negated MPL expression text.
    /// @return Name span pointing into `s`; empty span when the shape
    ///         does not match.
    inline StrSpan extractExpressionFromNegationSpan(const StrSpan& s) {
        int32_t startIndex = -1;
        for (int32_t i = 0; i + 1 < s.len; ++i) {
            if (s.ptr[i] == '!' && s.ptr[i + 1] == '(') {
                startIndex = i;
                break;
            }
        }
        int32_t endIndex = -1;
        for (int32_t i = 0; i < s.len; ++i) {
            if (s.ptr[i] == '[') { endIndex = i; break; }
        }
        if (startIndex < 0 || endIndex < 0 || startIndex >= endIndex) {
            return StrSpan();
        }
        return StrSpan(s.ptr + startIndex + 2,
                       endIndex - (startIndex + 2));
    }

    /// @brief One key → value substitution pair for `replaceKeysScratch`.
    struct StrReplacement {
        StrSpan key;
        StrSpan value;
    };

    /// @brief Longest key matching at `pos`, or -1 — the dispatch core of
    ///        `replaceKeysScratch`, mirroring the original `KeyTrie`'s greedy
    ///        longest-match (a too-long match is NOT retried shorter).
    ///
    /// @param source Source span.
    /// @param pos    Match position.
    /// @param pairs  Substitution pairs.
    /// @param count  Number of pairs.
    /// @return Index of the longest matching key, or -1.
    inline int32_t matchLongestKeyAt(const StrSpan& source, int32_t pos,
                                     const StrReplacement* pairs,
                                     int32_t count) {
        int32_t best = -1;
        int32_t bestLen = 0;
        for (int32_t k = 0; k < count; ++k) {
            const StrSpan& key = pairs[k].key;
            assert(key.len > 0 && "empty replacement key");
            if (key.len <= bestLen) continue;
            if (pos + key.len > source.len) continue;
            if (std::memcmp(source.ptr + pos, key.ptr,
                            static_cast<size_t>(key.len)) == 0) {
                best = k;
                bestLen = key.len;
            }
        }
        return best;
    }

    /// @brief View twin of `ce::replaceKeysInString` — token-boundary
    ///        multi-key substitution into the scratch arena.
    ///
    /// @details
    /// Byte-exact reproduction of the original's semantics: a key is
    /// replaced only when the previous character is `[` or `,` (never at
    /// position 0), the GREEDY LONGEST key match is taken (with no
    /// shorter-key retry when it fails the boundary test, exactly like
    /// the `KeyTrie`), and the character after the match must be `]` or
    /// `,` (a match reaching the end of the string does not qualify).
    /// Two passes over the source — the first computes the exact output
    /// length, the second fills the single allocation — per the
    /// exact-length doctrine; the scan logic is shared so the passes
    /// cannot diverge.
    ///
    /// @param arena  Scratch arena receiving the result.
    /// @param source Source span.
    /// @param pairs  Substitution pairs (keys non-empty; order
    ///               irrelevant — longest match wins).
    /// @param count  Number of pairs; 0 degrades to a plain copy.
    /// @return Hot view of the substituted text.
    inline ScratchString replaceKeysScratch(ScratchArena& arena, const StrSpan& source,
                                    const StrReplacement* pairs,
                                    int32_t count) {
        assert(count >= 0);
        if (count == 0) return copyScratch(arena, source);

        // Pass 1: exact output length.
        int32_t outLen = 0;
        for (int32_t i = 0; i < source.len; ) {
            const char prev = (i == 0) ? '\0' : source.ptr[i - 1];
            if (prev == '[' || prev == ',') {
                const int32_t k =
                    matchLongestKeyAt(source, i, pairs, count);
                if (k >= 0) {
                    const int32_t nextPos = i + pairs[k].key.len;
                    if (nextPos < source.len
                        && (source.ptr[nextPos] == ']'
                            || source.ptr[nextPos] == ',')) {
                        outLen += pairs[k].value.len;
                        i = nextPos;
                        continue;
                    }
                }
            }
            ++outLen;
            ++i;
        }
        if (outLen == 0) return ScratchString();

        // Pass 2: single allocation, fill.
        char* buf = arena.allocBytes(outLen);
        int32_t at = 0;
        for (int32_t i = 0; i < source.len; ) {
            const char prev = (i == 0) ? '\0' : source.ptr[i - 1];
            if (prev == '[' || prev == ',') {
                const int32_t k =
                    matchLongestKeyAt(source, i, pairs, count);
                if (k >= 0) {
                    const int32_t nextPos = i + pairs[k].key.len;
                    if (nextPos < source.len
                        && (source.ptr[nextPos] == ']'
                            || source.ptr[nextPos] == ',')) {
                        if (pairs[k].value.len > 0) {
                            std::memcpy(buf + at, pairs[k].value.ptr,
                                static_cast<size_t>(pairs[k].value.len));
                            at += pairs[k].value.len;
                        }
                        i = nextPos;
                        continue;
                    }
                }
            }
            buf[at++] = source.ptr[i];
            ++i;
        }
        assert(at == outLen);
        return ScratchString::wrap(arena, buf, outLen);
    }

    /// @brief `std::string`-returning twin of `replaceKeysScratch` — token-boundary
    ///        multi-key substitution with NO scratch arena.
    ///
    /// @details
    /// For absorb-door callers whose result IS a returned `std::string` (the
    /// deferred-to-T8 boundary) and whose replacement values are slices of the
    /// input (so no arena is needed to hold the values). The scan is byte-exact
    /// to `replaceKeysScratch` / `ce::replaceKeysInString`: a key is replaced only
    /// when the previous character is `[` or `,` (never at position 0), the GREEDY
    /// LONGEST key match is taken (no shorter-key retry), and the character after
    /// the match must be `]` or `,`. Two passes — exact output length, then fill —
    /// over one shared scan lambda so the passes cannot diverge.
    ///
    /// @param source Source span.
    /// @param pairs  Substitution pairs (keys non-empty; longest match wins).
    /// @param count  Number of pairs; 0 returns a plain copy of `source`.
    /// @return The substituted text as a fresh `std::string`.
    inline std::string replaceKeysToString(const StrSpan& source,
                                           const StrReplacement* pairs,
                                           int32_t count) {
        assert(count >= 0);
        if (count == 0) return source.toStdString();

        // Shared scan: returns output length; when `out` is non-null, fills it.
        const auto scan = [&](char* out) -> int32_t {
            int32_t at = 0;
            for (int32_t i = 0; i < source.len; ) {
                const char prev = (i == 0) ? '\0' : source.ptr[i - 1];
                if (prev == '[' || prev == ',') {
                    const int32_t k = matchLongestKeyAt(source, i, pairs, count);
                    if (k >= 0) {
                        const int32_t nextPos = i + pairs[k].key.len;
                        if (nextPos < source.len
                            && (source.ptr[nextPos] == ']'
                                || source.ptr[nextPos] == ',')) {
                            if (out != nullptr && pairs[k].value.len > 0) {
                                std::memcpy(out + at, pairs[k].value.ptr,
                                    static_cast<size_t>(pairs[k].value.len));
                            }
                            at += pairs[k].value.len;
                            i = nextPos;
                            continue;
                        }
                    }
                }
                if (out != nullptr) out[at] = source.ptr[i];
                ++at;
                ++i;
            }
            return at;
        };

        const int32_t outLen = scan(nullptr);
        std::string result;
        if (outLen > 0) {
            result.resize(static_cast<size_t>(outLen));
            scan(&result[0]);
        }
        return result;
    }

    /// @brief View twin of `ExpressionAnalyzer::replaceUSubstrings` —
    ///        strip `u_` prefixes that begin a token (after `[` or `,`),
    ///        into the scratch arena.
    ///
    /// @details
    /// Byte-exact reproduction including the original's erase-and-recheck
    /// loop (`[u_u_x` strips to `[x` — after one strip the delimiter
    /// still precedes the next `u_`) and its index window (a candidate is
    /// only considered when a character follows the `u_`, mirroring
    /// `i + 1 < out.size()`; strings shorter than 2 copy through). Two
    /// passes — exact length, then fill.
    ///
    /// @param arena  Scratch arena receiving the result.
    /// @param source Source span.
    /// @return Hot view of the stripped text.
    inline ScratchString replaceUSubstringsScratch(ScratchArena& arena,
                                           const StrSpan& source) {
        if (source.len < 2) return copyScratch(arena, source);

        // Virtual scan over the post-erasure string the original mutates.
        // With j the source index and `emitted` the output count, the
        // original's mutated-string guard `i + 1 < out.size()` is exactly
        // `j + 1 < source.len` (i = emitted, j = emitted + 2*erasures,
        // out.size() = source.len - 2*erasures); its look-back `out[i-1]`
        // is the previously emitted byte; its erase-and-recheck is the
        // skip-without-emit (`[u_u_x` strips to `[x`). The loop never
        // considers position 0 and never lets the final character start
        // an erasure — both mirrored.
        const auto scan = [&source](char* fill) -> int32_t {
            if (fill != nullptr) fill[0] = source.ptr[0];
            int32_t emitted = 1;
            char prevEmitted = source.ptr[0];
            int32_t j = 1;
            while (j + 1 < source.len) {
                if (source.ptr[j] == 'u' && source.ptr[j + 1] == '_'
                    && (prevEmitted == '[' || prevEmitted == ',')) {
                    j += 2;
                    continue;
                }
                if (fill != nullptr) fill[emitted] = source.ptr[j];
                prevEmitted = source.ptr[j];
                ++emitted;
                ++j;
            }
            if (j < source.len) {
                if (fill != nullptr) fill[emitted] = source.ptr[j];
                ++emitted;
            }
            return emitted;
        };

        const int32_t outLen = scan(nullptr);
        if (outLen == source.len) return copyScratch(arena, source);
        char* buf = arena.allocBytes(outLen);
        const int32_t filled = scan(buf);
        assert(filled == outLen);
        (void)filled;
        return ScratchString::wrap(arena, buf, outLen);
    }

    /// @brief Deduplicate the bound-variable lists of an implication, dropping
    ///        `u_`-prefixed vars — the heap-free span twin of the former
    ///        file-static `deduplicateBoundVars`.
    ///
    /// @details
    /// After variable replacement a `(>[...]` bound-variable list may contain
    /// duplicate or `u_`-prefixed vars. This walks @p impl once, copying every
    /// byte verbatim EXCEPT each `(>[` list's comma-separated var run, which it
    /// rebuilds: parse the tokens between `(>[` and the next `]`, skip empty and
    /// `u_`-prefixed tokens, keep the first occurrence of each remaining var
    /// (linear `equalSpans` dedup against a bounded stack `StrSpan seen[]`), and
    /// re-emit the distinct vars comma-joined. The `(>[` and the trailing `]`
    /// echo 1:1; a `(>[` with no matching `]` copies through unchanged. The
    /// output only COPIES or REMOVES bytes, so its length is `<= impl.len` — one
    /// `allocBytes(impl.len)` worst-case buffer, filled in a single pass, wrapped
    /// at the exact filled length (`assert(at <= impl.len)`; an empty input or an
    /// all-stripped output returns the default empty `ScratchString`, since
    /// `wrap` requires `len > 0`).
    ///
    /// Byte-identical to the heap oracle by construction: same `(>[` detection,
    /// same first-seen order, same `u_` strip, same comma join — a pure string
    /// transform, no mint, no interner. The result rides @p arena's string tier
    /// and is spanned only at the caller's synchronous edge (09c pitfall 4/5).
    /// The per-list dedup set is a stack `StrSpan[kBoundVarCap]` (cap 64, the
    /// `reconstructImplicationFullBind` bound-var-list ceiling); an overrun is a
    /// loud Rule-19 widen-on-STOP assert.
    ///
    /// @param impl  The implication text to clean (a slice of a stable buffer).
    /// @param arena Per-slot string-tier scratch arena receiving the result.
    /// @return The cleaned implication as a `ScratchString` on @p arena; empty
    ///         when @p impl is empty or the transform emits nothing.
    /// @see equalSpans; getArgsSpans (the sibling `[...]` slicer).
    inline ScratchString deduplicateBoundVarsScratch(StrSpan impl, ScratchArena& arena) {
        if (impl.len == 0) return ScratchString();
        constexpr int32_t kBoundVarCap = 64;
        char* buf = arena.allocBytes(impl.len);   // output never grows
        int32_t at = 0;
        int32_t i = 0;
        while (i < impl.len) {
            // Look for "(>["
            if (i + 2 < impl.len && impl.ptr[i] == '(' && impl.ptr[i + 1] == '>'
                && impl.ptr[i + 2] == '[') {
                buf[at++] = '('; buf[at++] = '>'; buf[at++] = '[';
                i += 3;
                int32_t closeBracket = -1;
                for (int32_t s = i; s < impl.len; ++s) {
                    if (impl.ptr[s] == ']') { closeBracket = s; break; }
                }
                if (closeBracket >= 0) {
                    StrSpan seen[kBoundVarCap];
                    int32_t seenN = 0;
                    int32_t emitted = 0;
                    const int32_t listEnd = closeBracket;
                    int32_t pos = i;                 // absolute index of the var run
                    while (pos <= listEnd) {
                        int32_t comma = -1;
                        for (int32_t s = pos; s < listEnd; ++s) {
                            if (impl.ptr[s] == ',') { comma = s; break; }
                        }
                        const int32_t tokStart = pos;
                        int32_t tokEnd;              // exclusive
                        if (comma < 0) { tokEnd = listEnd; pos = listEnd + 1; }
                        else           { tokEnd = comma;   pos = comma + 1;   }
                        const int32_t tokLen = tokEnd - tokStart;
                        if (tokLen > 0) {
                            const StrSpan v(impl.ptr + tokStart, tokLen);
                            const bool isU =
                                (tokLen >= 2 && v.ptr[0] == 'u' && v.ptr[1] == '_');
                            if (!isU) {
                                bool dup = false;
                                for (int32_t k = 0; k < seenN; ++k)
                                    if (equalSpans(v, seen[k])) { dup = true; break; }
                                if (!dup) {
                                    assert(seenN < kBoundVarCap
                                        && "deduplicateBoundVarsScratch: bound-var list exceeds cap 64");
                                    seen[seenN++] = v;
                                    if (emitted > 0) buf[at++] = ',';
                                    std::memcpy(buf + at, v.ptr,
                                                static_cast<std::size_t>(tokLen));
                                    at += tokLen;
                                    ++emitted;
                                }
                            }
                        }
                    }
                    i = closeBracket;   // the ']' is copied on the next iteration
                }
            } else {
                buf[at++] = impl.ptr[i];
                ++i;
            }
        }
        assert(at <= impl.len);
        if (at == 0) return ScratchString();
        return ScratchString::wrap(arena, buf, at);
    }

    /// @brief Split a span on a single-character delimiter into caller
    ///        storage, zero copies.
    ///
    /// @details
    /// Generic sibling of `getArgsSpans` for delimiter-joined names (the
    /// validity `_boundary_` splitting uses the multi-character split at
    /// its own migration step; this covers the single-character cases).
    /// Adjacent delimiters and a leading/trailing delimiter yield empty
    /// spans, like repeated `std::string::find` slicing.
    ///
    /// @param s        Source span.
    /// @param delim    Delimiter byte.
    /// @param out      Caller array receiving the piece spans.
    /// @param maxParts Capacity of `out`; exceeding it asserts.
    /// @return Number of pieces written (always >= 1).
    inline int32_t splitByCharSpans(const StrSpan& s, char delim,
                                    StrSpan* out, int32_t maxParts) {
        assert(out != nullptr && maxParts > 0);
        int32_t count = 0;
        int32_t pos = 0;
        while (true) {
            int32_t next = -1;
            for (int32_t i = pos; i < s.len; ++i) {
                if (s.ptr[i] == delim) { next = i; break; }
            }
            assert(count < maxParts
                && "splitByCharSpans: piece count exceeds caller capacity");
            if (next < 0) {
                out[count++] = StrSpan(s.ptr + pos, s.len - pos);
                break;
            }
            out[count++] = StrSpan(s.ptr + pos, next - pos);
            pos = next + 1;
        }
        return count;
    }

    /// @brief Word-boundary single-key replacement — the heap-free span twin
    ///        of the former file-static `replaceArgInString`.
    ///
    /// @details
    /// Replaces every word-boundary-delimited occurrence of @p oldArg in
    /// @p str with @p newArg, byte-faithful to the original `replaceArgInString`
    /// rule and DELIBERATELY UNLIKE `replaceKeysScratch`: a match at position
    /// `i` qualifies when the character before it is not a word character
    /// (`[A-Za-z0-9_]`) OR `i == 0`, AND the character after the match is not a
    /// word character OR the match reaches the end of the string. Any non-word
    /// byte (space, `(`, `)`, `[`, `,`, `]`) is a boundary, position 0 qualifies,
    /// and an end-of-string match qualifies — none of which the `[`/`,` … `]`/`,`
    /// token rule of `replaceKeysScratch` allows. The two are therefore NOT
    /// interchangeable; this must not be folded into `replaceKeysScratch`.
    ///
    /// Two passes over @p str off one shared word-boundary scan lambda: the
    /// first computes the exact output length, the second fills a single
    /// `allocBytes(outLen)` and `ScratchString::wrap`s it (`assert(filled ==
    /// outLen)`), per the exact-length doctrine so the passes cannot diverge.
    /// The greedy forward scan advances past each replaced occurrence exactly
    /// as the heap original did (no overlap re-scan). @p newArg / @p str /
    /// @p oldArg are spans over caller-stable input; no interner is touched and
    /// nothing mints. The result rides @p arena's string tier and is spanned
    /// only at the caller's synchronous edge (09c pitfall 4/5).
    ///
    /// @param str    Text to transform (a slice of a stable buffer).
    /// @param oldArg Argument token to replace (word-boundary matched; must be
    ///               non-empty — an empty search token asserts, the original's
    ///               unbounded infinite-loop precondition made explicit).
    /// @param newArg Replacement bytes (copied verbatim; may be empty).
    /// @param arena  Per-slot string-tier scratch arena receiving the result.
    /// @return The transformed text as a `ScratchString` on @p arena; empty when
    ///         the transform emits nothing.
    /// @see replaceKeysScratch — the token-boundary multi-key sibling with a
    ///      DIFFERENT (non-interchangeable) boundary rule.
    inline ScratchString replaceArgSpanScratch(StrSpan str, StrSpan oldArg,
                                               StrSpan newArg, ScratchArena& arena) {
        assert(oldArg.len > 0 && "replaceArgSpanScratch: empty search token");
        const auto isWordChar = [](char c) -> bool {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
                || (c >= '0' && c <= '9') || c == '_';
        };
        // Shared scan: returns output length; when `out` is non-null, fills it.
        const auto scan = [&](char* out) -> int32_t {
            int32_t at = 0;
            int32_t i = 0;
            while (i < str.len) {
                const bool atBoundary =
                    (i == 0) || !isWordChar(str.ptr[i - 1]);
                if (atBoundary && i + oldArg.len <= str.len
                    && std::memcmp(str.ptr + i, oldArg.ptr,
                                   static_cast<size_t>(oldArg.len)) == 0) {
                    const int32_t after = i + oldArg.len;
                    const bool endBoundary =
                        (after >= str.len) || !isWordChar(str.ptr[after]);
                    if (endBoundary) {
                        if (out != nullptr && newArg.len > 0)
                            std::memcpy(out + at, newArg.ptr,
                                        static_cast<size_t>(newArg.len));
                        at += newArg.len;
                        i = after;
                        continue;
                    }
                }
                if (out != nullptr) out[at] = str.ptr[i];
                ++at;
                ++i;
            }
            return at;
        };
        const int32_t outLen = scan(nullptr);
        if (outLen == 0) return ScratchString();
        char* buf = arena.allocBytes(outLen);
        const int32_t filled = scan(buf);
        assert(filled == outLen);
        (void)filled;
        return ScratchString::wrap(arena, buf, outLen);
    }

    /// @brief Numeric-aware span comparison — the heap-free twin of the former
    ///        file-static `numericLess`.
    ///
    /// @details
    /// Orders two variable-name spans exactly as `numericLess` ordered the
    /// `std::string` pair: when BOTH spans are non-empty and consist entirely of
    /// decimal digits, compare their parsed integer values (so `"2" < "10"`);
    /// otherwise fall back to byte-lexicographic `compareSpans` (== the former
    /// `std::string::operator<`). Digit-run parsing is via `parseDigitRunToInt`
    /// (overflow-safe, no `std::stoi` throw); an all-digit run that overflows
    /// `int` is treated as non-numeric and falls through to the lexicographic
    /// branch — the in-practice-unreachable edge the original `std::stoi` would
    /// have thrown on (variable indices are tiny). No allocation, no locale, no
    /// mint.
    ///
    /// @param a Left span.
    /// @param b Right span.
    /// @return `true` iff @p a orders strictly before @p b under the
    ///         numeric-aware rule.
    /// @see parseDigitRunToInt — the overflow-safe digit parse; compareSpans —
    ///      the lexicographic fallback.
    inline bool numericLessSpan(const StrSpan& a, const StrSpan& b) {
        const auto allDigits = [](const StrSpan& s) -> bool {
            if (s.len == 0) return false;
            for (int32_t i = 0; i < s.len; ++i)
                if (!isDecimalDigitByte(s.ptr[i])) return false;
            return true;
        };
        int av = 0;
        int bv = 0;
        if (allDigits(a) && allDigits(b)
            && parseDigitRunToInt(a.ptr, a.len, av)
            && parseDigitRunToInt(b.ptr, b.len, bv)) {
            return av < bv;
        }
        return compareSpans(a, b) < 0;
    }

}
