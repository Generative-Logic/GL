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

/// @file
/// @brief Twin tests for the view-based string operations (string
///        statification — the fixed-length reimplementations).
///
/// @details
/// Every operation in `memory_infra/str_ops.hpp` that twins an existing
/// std::string helper is proven BYTE-IDENTICAL against its original here,
/// across canonical MPL inputs and the originals' documented edge cases
/// (the trailing-comma and nested-bracket mis-parse of `ce::getArgs`, the
/// greedy-longest-no-retry dispatch of `ce::replaceKeysInString`, the
/// erase-and-recheck chain of `replaceUSubstrings`). The migration commits
/// rest on these pins: a divergence here is a future byte-level pipeline
/// regression caught at unit-test speed.

#include "test_harness.hpp"

#include "../compiler.hpp"
#include "../memory_infra/str_ops.hpp"
#include "../prover.hpp"

#include <algorithm>
#include <cctype>
#include <map>
#include <regex>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace {
    // 1 MiB pool / 256 KiB block; 1-block hot reservation.
    const gl::StaticMemoryConfig kStrOpsTestCfg{ 1 << 20, 1 << 18 };

    /// @brief Twin-run `getArgs`: original vector vs span slices.
    bool getArgsTwinMatches(const std::string& expr) {
        const std::vector<std::string> ref = ce::getArgs(expr);
        gl::StrSpan spans[16];
        const int32_t n =
            gl::getArgsSpans(gl::StrSpan(expr), spans, 16);
        if (static_cast<size_t>(n) != ref.size()) return false;
        for (int32_t i = 0; i < n; ++i) {
            if (spans[i].toStdString() != ref[static_cast<size_t>(i)])
                return false;
        }
        return true;
    }

    /// @brief Twin-run `replaceKeysInString`: original vs hot twin.
    bool replaceKeysTwinMatches(
        gl::ScratchArena& arena, const std::string& source,
        const std::map<std::string, std::string>& map) {
        const std::string ref = ce::replaceKeysInString(source, map);
        std::vector<gl::StrReplacement> pairs;
        pairs.reserve(map.size());
        for (const auto& kv : map) {
            pairs.push_back(gl::StrReplacement{
                gl::StrSpan(kv.first), gl::StrSpan(kv.second) });
        }
        const gl::ScratchString out = gl::replaceKeysScratch(
            arena, gl::StrSpan(source), pairs.data(),
            static_cast<int32_t>(pairs.size()));
        return out.toStdString() == ref;
    }

    /// @brief Twin-run: original `ce::replaceKeysInString` vs the no-arena
    ///        `replaceKeysToString`.
    bool replaceKeysToStringTwinMatches(
        const std::string& source,
        const std::map<std::string, std::string>& map) {
        const std::string ref = ce::replaceKeysInString(source, map);
        std::vector<gl::StrReplacement> pairs;
        pairs.reserve(map.size());
        for (const auto& kv : map) {
            pairs.push_back(gl::StrReplacement{
                gl::StrSpan(kv.first), gl::StrSpan(kv.second) });
        }
        return gl::replaceKeysToString(gl::StrSpan(source), pairs.data(),
            static_cast<int32_t>(pairs.size())) == ref;
    }

    /// @brief The statified substitution build every equivalence-class
    ///        rewrite uses: each member / canonical name is copied onto the
    ///        string scratch arena (the `ScratchString` local dies each
    ///        iteration but its span persists until the arena rewinds),
    ///        several members share one canonical value, and the
    ///        `StrReplacement` run rides a contiguous byte-bump arena
    ///        allocation — the result must be byte-identical to the former
    ///        `std::map` + `ce::replaceKeysInString`. Members carry distinct
    ///        keys (equivalence classes at one validity are disjoint, so no
    ///        duplicate-key last-write-wins case arises).
    bool scratchSubstTwinMatches(
        gl::ScratchArena& sArena, gl::ScratchArena& gArena,
        const std::string& source,
        const std::vector<std::pair<std::string, std::string>>& members) {
        std::map<std::string, std::string> ref;
        for (const auto& kv : members) ref[kv.first] = kv.second;
        const std::string refOut = ce::replaceKeysInString(source, ref);
        if (ref.empty()) return refOut == source;

        gl::ScratchScope sScope(sArena);
        gl::ScratchScope gScope(gArena);
        const int total = static_cast<int>(members.size());
        auto* pairs = reinterpret_cast<gl::StrReplacement*>(gArena.resolve(
            gArena.alloc(
                static_cast<int32_t>(static_cast<std::size_t>(total)
                                     * sizeof(gl::StrReplacement)),
                static_cast<int32_t>(alignof(gl::StrReplacement)))));
        int n = 0;
        for (const auto& kv : members) {
            const gl::ScratchString k = gl::ScratchString::copyFrom(
                sArena, kv.first.data(), static_cast<int32_t>(kv.first.size()));
            const gl::ScratchString v = gl::ScratchString::copyFrom(
                sArena, kv.second.data(), static_cast<int32_t>(kv.second.size()));
            pairs[n].key = gl::StrSpan(k);
            pairs[n].value = gl::StrSpan(v);
            ++n;
        }
        const gl::ScratchString out =
            gl::replaceKeysScratch(sArena, gl::StrSpan(source), pairs, total);
        return out.toStdString() == refOut;
    }
}

TEST(str_ops, scratch_subst_run_twin) {
    gl::GlobalMemoryManager m;
    m.init(kStrOpsTestCfg);
    gl::ScratchArena sA; sA.bind(&m);
    gl::ScratchArena gA; gA.bind(&m);

    // Empty class set: expression returned unchanged.
    ASSERT_TRUE(scratchSubstTwinMatches(sA, gA, "(=[a,b])", {}));
    // Single member -> canonical.
    ASSERT_TRUE(scratchSubstTwinMatches(
        sA, gA, "(in2[int_lev_0_X,v1,s])", { { "int_lev_0_X", "i0" } }));
    // Several members share ONE canonical value (the many->one shape a single
    // equivalence class produces).
    ASSERT_TRUE(scratchSubstTwinMatches(
        sA, gA, "(p[int_lev_0,int_lev_1,int_lev_2])",
        { { "int_lev_0", "a" }, { "int_lev_1", "a" }, { "int_lev_2", "a" } }));
    // Two classes' worth of members, distinct canonicals, greedy-longest
    // overlap (v1 vs v, w1 vs w).
    ASSERT_TRUE(scratchSubstTwinMatches(
        sA, gA, "(p[v1,v,w1,w])",
        { { "v1", "X" }, { "v", "Y" }, { "w1", "P" }, { "w", "Q" } }));
    // No key present in the source: unchanged.
    ASSERT_TRUE(scratchSubstTwinMatches(
        sA, gA, "(in[absent])", { { "int_lev_0", "z" } }));
}

TEST(str_ops, spans_compare_equal_hash) {
    const std::string a = "(in2[i0,v1,s])";
    const std::string b = "(in2[i0,v1,t])";
    ASSERT_TRUE(gl::equalSpans(gl::StrSpan(a), gl::StrSpan(a)));
    ASSERT_FALSE(gl::equalSpans(gl::StrSpan(a), gl::StrSpan(b)));
    // operator== is the canonical equalSpans (the generic cold-map erase key
    // predicate uses it for byte keys); equal content / differing content / empty.
    ASSERT_TRUE(gl::StrSpan(a) == gl::StrSpan(a));
    ASSERT_FALSE(gl::StrSpan(a) == gl::StrSpan(b));
    ASSERT_TRUE(gl::StrSpan() == gl::StrSpan());
    // Ordering identical to std::string::compare's sign.
    ASSERT_LT(gl::compareSpans(gl::StrSpan(a), gl::StrSpan(b)), 0);
    ASSERT_EQ(gl::compareSpans(gl::StrSpan(a), gl::StrSpan(a)), 0);
    // Prefix orders before its extension, as with std::string.
    const std::string p = "(in2";
    ASSERT_LT(gl::compareSpans(gl::StrSpan(p), gl::StrSpan(a)), 0);
    // Hash: equal content equal hash, differing content differing here.
    ASSERT_EQ(gl::hashSpan(gl::StrSpan(a)), gl::hashSpan(gl::StrSpan(a)));
    ASSERT_NE(gl::hashSpan(gl::StrSpan(a)), gl::hashSpan(gl::StrSpan(b)));
    // The empty span hashes to the FNV-1a offset basis (pinned: the cold
    // find-index will persist nothing, but determinism matters).
    ASSERT_EQ(gl::hashSpan(gl::StrSpan()), 14695981039346656037ULL);
}

TEST(str_ops, contains_span_twin) {
    // Twin of `haystack.find(needle) != std::string::npos` across hit
    // positions, misses, and the documented edge cases.
    const std::string hay = "main_boundary__hypo_x_boundary_y";
    const std::string needles[] = {
        "main", "_hypo_", "_boundary_y", "main_boundary__hypo_x_boundary_y",
        "absent", "_hypo!", "y_boundary", "", "main_boundary__hypo_x_boundary_yz"
    };
    for (const std::string& n : needles) {
        const bool expected = hay.find(n) != std::string::npos;
        ASSERT_EQ(gl::containsSpan(gl::StrSpan(hay), gl::StrSpan(n)), expected);
    }
    // Empty haystack: only the empty needle is contained, as with find.
    const std::string empty;
    ASSERT_TRUE(gl::containsSpan(gl::StrSpan(empty), gl::StrSpan(empty)));
    ASSERT_FALSE(gl::containsSpan(gl::StrSpan(empty), gl::StrSpan(hay)));
}

// startsWithSpan — prefix twin of the heap `startsWith(std::string, pfx, n)`
// helper: `s.size() >= n && s.compare(0, n, pfx) == 0`. Covers a hit, a
// miss on differing bytes, a prefix longer than the span (miss), the whole
// span as its own prefix (hit), the empty prefix (always hit), and the empty
// span with a non-empty prefix (miss).
TEST(str_ops, starts_with_span_matches_string) {
    const std::string s = "(in2[rec0,N])";
    const auto oracle = [](const std::string& str, const char* pfx,
                           std::size_t n) {
        return str.size() >= n && std::memcmp(str.data(), pfx, n) == 0;
    };
    const struct { const char* pfx; std::size_t n; } cases[] = {
        { "(in2[rec", 8 }, { "(in2[rez", 8 }, { "(in3[rec", 8 },
        { "(in2[rec0,N])", 13 }, { "(in2[rec0,N])X", 14 }, { "", 0 },
    };
    for (const auto& c : cases) {
        ASSERT_EQ(gl::startsWithSpan(gl::StrSpan(s), c.pfx, c.n),
                  oracle(s, c.pfx, c.n));
    }
    // Empty span: only the empty prefix matches.
    const std::string empty;
    ASSERT_TRUE(gl::startsWithSpan(gl::StrSpan(empty), "", 0));
    ASSERT_FALSE(gl::startsWithSpan(gl::StrSpan(empty), "(", 1));
}

// findSpanFrom / rfindSpanBefore — position twins of std::string::find(n, pos)
// / rfind(n, pos), with -1 standing for npos. Covers: needle at 0 / at end /
// absent, repeated needles (find picks first-from-pos, rfind last-at-or-before-
// pos), fromPos past the last occurrence, fromPos == len, fromPos > len, rfind
// pos mid-occurrence (a start AT pos is accepted), pos = 0, needle longer than
// haystack, needle == whole haystack, and a mid-buffer slice (no null-
// terminator dependence). Empty needles are asserted out (the documented
// divergence), so the oracle sweep uses non-empty needles only.
TEST(str_ops, find_rfind_span_match_std_string_oracle) {
    const auto checkBoth = [](const std::string& hay, const std::string& needle,
                              int32_t pos) {
        const size_t refF = hay.find(needle, static_cast<size_t>(pos));
        const int32_t myF = gl::findSpanFrom(gl::StrSpan(hay),
                                             gl::StrSpan(needle), pos);
        if (refF == std::string::npos) { ASSERT_EQ(myF, -1); }
        else { ASSERT_EQ(myF, static_cast<int32_t>(refF)); }

        const size_t refR = hay.rfind(needle, static_cast<size_t>(pos));
        const int32_t myR = gl::rfindSpanBefore(gl::StrSpan(hay),
                                                gl::StrSpan(needle), pos);
        if (refR == std::string::npos) { ASSERT_EQ(myR, -1); }
        else { ASSERT_EQ(myR, static_cast<int32_t>(refR)); }
    };

    // Repeated needles inside a hypo-shaped name; sweep every pos 0..len+2.
    const std::string hay = "_var0_x_var1_y_hypo__hypo_z";
    const std::string needles[] = {
        "_var0_", "_var1_", "_hypo_", "x", "z", "absent", "_var2_",
        "_var0_x_var1_y_hypo__hypo_z",              // whole haystack
        "_var0_x_var1_y_hypo__hypo_z!",             // longer than haystack
    };
    for (const std::string& n : needles) {
        for (int32_t pos = 0;
             pos <= static_cast<int32_t>(hay.size()) + 2; ++pos) {
            checkBoth(hay, n, pos);
        }
    }

    // Needle at position 0 and at the very end.
    checkBoth("abcabc", "abc", 0);
    checkBoth("abcabc", "abc", 3);
    checkBoth("abcabc", "bc", 5);   // fromPos past the last occurrence
    checkBoth("abcabc", "c", 6);    // fromPos == haystack.len
    checkBoth("abcabc", "c", 8);    // fromPos > haystack.len

    // rfind pos mid-occurrence: an occurrence starting AT pos is accepted;
    // one starting after is not.
    ASSERT_EQ(gl::rfindSpanBefore(gl::StrSpan(std::string("xxabcxxabc")),
                                  gl::StrSpan(std::string("abc")), 7), 7);
    ASSERT_EQ(gl::rfindSpanBefore(gl::StrSpan(std::string("xxabcxxabc")),
                                  gl::StrSpan(std::string("abc")), 6), 2);
    checkBoth("xxabcxxabc", "abc", 7);
    checkBoth("xxabcxxabc", "abc", 6);
    checkBoth("xxabcxxabc", "abc", 0);   // pos = 0, no occurrence at 0

    // Mid-buffer slice: the span is carved from a larger live buffer, so a
    // match may not rely on a null terminator; the oracle runs on the
    // equivalent std::string copy of the slice.
    const std::string buf = "AA_hypo_BB_hypo_CC";
    const gl::StrSpan mid(buf.data() + 2, 14);      // "_hypo_BB_hypo_"
    const std::string midCopy(buf.data() + 2, 14);
    const std::string probe = "_hypo_";
    for (int32_t pos = 0;
         pos <= static_cast<int32_t>(midCopy.size()) + 1; ++pos) {
        const size_t refF = midCopy.find(probe, static_cast<size_t>(pos));
        const int32_t myF = gl::findSpanFrom(mid, gl::StrSpan(probe), pos);
        if (refF == std::string::npos) { ASSERT_EQ(myF, -1); }
        else { ASSERT_EQ(myF, static_cast<int32_t>(refF)); }
        const size_t refR = midCopy.rfind(probe, static_cast<size_t>(pos));
        const int32_t myR = gl::rfindSpanBefore(mid, gl::StrSpan(probe), pos);
        if (refR == std::string::npos) { ASSERT_EQ(myR, -1); }
        else { ASSERT_EQ(myR, static_cast<int32_t>(refR)); }
    }
}

// writeDecimalDigits — byte-identical to std::to_string for non-negative
// int32 values across digit-count boundaries and INT32_MAX.
TEST(str_ops, write_decimal_digits_matches_to_string) {
    const int32_t cases[] = { 0, 1, 9, 10, 15, 99, 100, 12345, 2147483647 };
    for (const int32_t v : cases) {
        char buf[10];
        const int32_t n = gl::writeDecimalDigits(buf, v);
        ASSERT_EQ(std::string(buf, static_cast<size_t>(n)), std::to_string(v));
    }
}

TEST(str_ops, get_args_twin_canonical_and_edges) {
    ASSERT_TRUE(getArgsTwinMatches("(in2[i0,v1,s])"));
    ASSERT_TRUE(getArgsTwinMatches("(=[a,b])"));
    ASSERT_TRUE(getArgsTwinMatches("(AnchorPeano[N,i0,s,+,*,i1])"));
    ASSERT_TRUE(getArgsTwinMatches("(p[x])"));
    ASSERT_TRUE(getArgsTwinMatches("no_brackets_here"));
    ASSERT_TRUE(getArgsTwinMatches("(p[])"));
    ASSERT_TRUE(getArgsTwinMatches("(p[a,])"));     // trailing empty arg
    ASSERT_TRUE(getArgsTwinMatches("(p[,a])"));     // leading empty arg
    ASSERT_TRUE(getArgsTwinMatches("(p[,])"));
    // The documented nested mis-parse — twins must mis-parse identically.
    ASSERT_TRUE(getArgsTwinMatches("(p[(q[a,b]),c])"));
}

TEST(str_ops, extract_expression_twins) {
    const std::string shapes[] = {
        "(=[a,b])", "!(in[x,y])", "in3[x,y,z]", "(AnchorPeano[N,i0])",
        "no_brackets", "!(=[s(a),b])", "x!(p[a])", "(p)"
    };
    for (const std::string& s : shapes) {
        ASSERT_EQ(gl::extractExpressionSpan(gl::StrSpan(s)).toStdString(),
                  ce::extractExpression(s));
        ASSERT_EQ(gl::extractExpressionUniversalSpan(
                      gl::StrSpan(s)).toStdString(),
                  extractExpressionUniversalOracle(s));
        ASSERT_EQ(gl::extractExpressionFromNegationSpan(
                      gl::StrSpan(s)).toStdString(),
                  ce::extractExpressionFromNegation(s));
    }
}

TEST(str_ops, match_lev_id_twin) {
    // Byte-identical twins of disintegrateExpr2's forceDeep / Pass-A regex
    // gates: matchItLevId == ^it_\d+_lev_(\d+)_(\d+)$ (captures = trailing two
    // numbers), matchIntLevId == ^int_lev_\d+_\d+$. An over-INT_MAX digit run
    // returns false, matching the original regex-matched-but-stoi-threw path
    // (the catch left the gate untriggered), so the comparison is against
    // "regex matched AND its numbers fit int", not the bare regex match.
    static const std::regex reItCap(R"(^it_\d+_lev_(\d+)_(\d+)$)");
    static const std::regex reInt(R"(^int_lev_\d+_\d+$)");

    const std::string corpus[] = {
        "it_0_lev_5_123", "it_10_lev_0_0", "it_3_lev_12_7", "it_00_lev_01_02",
        "int_lev_2_9", "int_lev_0_0", "int_lev_12_345",
        "it_lev_5_3", "it_0_lev_5", "it_0_lev__3", "it_0_lev_5_3_",
        "xit_0_lev_5_3", "it_0_lev_5_3x", "u_it_0_lev_5_3",
        "int_lev_5", "intlev_5_3", "print_lev_3_4", "in3[int_lev_0_1,x]",
        "", "i", "it_", "int_lev_",
        "it_0_lev_5_99999999999", "int_lev_99999999999_1"
    };

    for (const std::string& s : corpus) {
        // --- it_ shape (forceDeep captures + Pass-A extraction) ---
        std::smatch m;
        int capLvl = 0, capId = 0;
        bool capFits = false;
        if (std::regex_match(s, m, reItCap)) {
            try {
                capLvl = std::stoi(m[1].str());
                capId = std::stoi(m[2].str());
                capFits = true;
            } catch (...) { capFits = false; }
        }
        int lvl = -777, id = -777;
        const bool myIt = gl::matchItLevId(gl::StrSpan(s), lvl, id);
        ASSERT_TRUE(myIt == capFits);
        if (myIt) { ASSERT_EQ(lvl, capLvl); ASSERT_EQ(id, capId); }

        // --- int_ shape ---
        int iLvl = 0, iId = 0;
        bool intFits = false;
        if (std::regex_match(s, reInt)) {
            const std::size_t lp = 8;  // strlen("int_lev_")
            const std::size_t us = s.rfind('_');
            try {
                iLvl = std::stoi(s.substr(lp, us - lp));
                iId = std::stoi(s.substr(us + 1));
                intFits = true;
            } catch (...) { intFits = false; }
        }
        int lvl2 = -777, id2 = -777;
        const bool myInt = gl::matchIntLevId(gl::StrSpan(s), lvl2, id2);
        ASSERT_TRUE(myInt == intFits);
        if (myInt) { ASSERT_EQ(lvl2, iLvl); ASSERT_EQ(id2, iId); }

        // the two shapes are disjoint — never both match
        ASSERT_FALSE(myIt && myInt);
    }
}

// isIntLevShape / isItLevShape — scanner verdict equals the anchored
// std::regex_match oracle (int_lev_\d+_\d+ / it_\d+_lev_\d+_\d+, the
// classifyName patterns) across the classifyName pin list plus shape edges:
// trailing underscore, missing / empty / triple digit runs, leading zeros, an
// over-INT_MAX run (regex matches — the scanners parse no values), a 40-digit
// run, and interleaved shapes. std::regex lives in the TEST only.
TEST(str_ops, int_it_lev_shape_scanners_match_regex_oracle) {
    static const std::regex kInt(R"(int_lev_\d+_\d+)");
    static const std::regex kIt(R"(it_\d+_lev_\d+_\d+)");
    const std::string longRun(40, '7');
    const std::string cases[] = {
        // classifyName pin list
        "int_lev_3_4", "int_lev_0_12", "it_0_lev_1_2", "it_12_lev_0_7",
        "x", "repl_7", "zero", "print_lev_3_4", "int_lev_3",
        "int_lev_3_4_5", "it_0_lev_1", "",
        // shape edges
        "int_lev_3_", "it_1_lev_2_",
        "int_lev__4", "it__lev_1_2", "it_1_lev__2",
        "int_lev_007_1",
        "int_lev_99999999999_1",
        "int_lev_" + longRun + "_1",
        "it_" + longRun + "_lev_1_2",
        "it_1_2_lev_3_4", "it_1_lev_2_3_4",
    };
    for (const std::string& s : cases) {
        ASSERT_EQ(gl::isIntLevShape(gl::StrSpan(s)),
                  std::regex_match(s, kInt));
        ASSERT_EQ(gl::isItLevShape(gl::StrSpan(s)),
                  std::regex_match(s, kIt));
    }
    // The over-INT_MAX run is the divergence pin against the value-parsing
    // matchers: regex (and scanner) match the SHAPE; matchIntLevId refuses
    // (documented overflow-to-false contract). The scanners must not inherit
    // that divergence.
    const std::string big = "int_lev_99999999999_1";
    ASSERT_TRUE(gl::isIntLevShape(gl::StrSpan(big)));
    int lev = -777, id = -777;
    ASSERT_FALSE(gl::matchIntLevId(gl::StrSpan(big), lev, id));
}

// scanIntLevOccurrences / scanItLevOccurrences — the full (start, text)
// occurrence SEQUENCE equals std::sregex_iterator with the two unanchored
// patterns (the scanSpecialTokens patterns), for both tiers on every input.
// std::regex lives in the TEST only. Covers: the scanSpecialTokens pin
// inputs (duplicates kept + order; print_lev fidelity; token-free; empty),
// partial extents, a no-match trailing underscore, back-to-back adjacency
// (resume-at-end), candidate advance-by-one, empty digit runs, mid-string
// tokens, a 40-digit run (no value parse), tokens at span start/end, and a
// mid-buffer slice with live digit bytes past the span end (no
// null-terminator dependence, no over-read).
TEST(str_ops, int_it_lev_occurrence_scanners_match_regex_iterator_oracle) {
    static const std::regex kInt(R"(int_lev_\d+_\d+)");
    static const std::regex kIt(R"(it_\d+_lev_\d+_\d+)");

    using Occ = std::pair<int32_t, std::string>;
    const auto oracleSeq = [](const std::string& s, const std::regex& re) {
        std::vector<Occ> out;
        for (std::sregex_iterator i(s.begin(), s.end(), re), e; i != e; ++i)
            out.emplace_back(static_cast<int32_t>(i->position()), i->str());
        return out;
    };
    const auto checkSpan = [&](const gl::StrSpan& span,
                               const std::string& equivalentCopy) {
        const std::vector<Occ> refInt = oracleSeq(equivalentCopy, kInt);
        const std::vector<Occ> refIt = oracleSeq(equivalentCopy, kIt);
        std::vector<Occ> myInt, myIt;
        gl::scanIntLevOccurrences(span, [&](int32_t st, int32_t ln) {
            myInt.emplace_back(
                st, std::string(span.ptr + st, static_cast<size_t>(ln)));
        });
        gl::scanItLevOccurrences(span, [&](int32_t st, int32_t ln) {
            myIt.emplace_back(
                st, std::string(span.ptr + st, static_cast<size_t>(ln)));
        });
        ASSERT_TRUE(myInt == refInt);
        ASSERT_TRUE(myIt == refIt);
    };

    const std::string longRun(40, '7');
    const std::string cases[] = {
        // scanSpecialTokens pin inputs
        "(in3[int_lev_0_1,it_2_lev_0_3,int_lev_0_1])",  // duplicates + order
        "(P[print_lev_3_4])",                           // substring fidelity
        "(=[x,zero])",                                  // token-free
        "",                                             // empty
        // extents / resume / advance
        "int_lev_1_2_3",                 // partial extent
        "int_lev_12_",                   // no match (second run empty)
        "int_lev_1_2int_lev_3_4",        // adjacency: resume at match end
        "iint_lev_1_2",                  // candidate advance-by-one
        "int_int_lev_1_2",               // literal restart mid-window
        "it_1_lev_2_3_4",                // it partial extent
        "it__lev_1_2",                   // empty first run -> no match
        "it_1_lev__2",                   // empty second run -> no match
        "xit_5_lev_1_2y",                // mid-string it token
        "int_lev_" + longRun + "_1",     // 40-digit run, no value parse
        "it_" + longRun + "_lev_1_2",
        "int_lev_3_4",                   // token == whole span
        "zzint_lev_3_4",                 // token at span end
        "int_lev_3_4zz",                 // token at span start
        "it_9_lev_8_7",                  // it token == whole span
    };
    for (const std::string& s : cases) {
        checkSpan(gl::StrSpan(s), s);
    }

    // Mid-buffer slice with live DIGIT bytes beyond the span end: the
    // scanner must stop its final digit run at span.len even though the
    // backing buffer continues with digits (the oracle judges the
    // equivalent copy of the slice only).
    const std::string buf = "XXint_lev_1_2it_3_lev_4_577";
    const gl::StrSpan mid(buf.data() + 2, 23);  // "int_lev_1_2it_3_lev_4_5"
    const std::string midCopy(buf.data() + 2, 23);
    checkSpan(mid, midCopy);
}

// scanItLevPrefixOccurrences — the (start, end, capture-1) occurrence SEQUENCE
// equals std::sregex_iterator with the UNANCHORED pattern it_(\d+)_lev_\d+_
// (extent stops at the '_' after the second digit run, NO trailing digit run).
// std::regex lives in the TEST only. Covers: none, one, back-to-back, the full
// it_D_lev_D_D shape (prefix consumes it_D_lev_D_, resume before the trailing
// digit), missing/empty runs, multi-digit, mid-buffer with a boundary-respect
// slice (the trailing '_' outside the span must not match).
TEST(str_ops, it_lev_prefix_occurrence_scan_matches_regex_oracle) {
    static const std::regex kPrefix(R"(it_(\d+)_lev_\d+_)");
    using Occ = std::tuple<int32_t, int32_t, std::string>;  // start, end, cap1
    const auto oracleSeq = [](const std::string& s) {
        std::vector<Occ> out;
        for (std::sregex_iterator i(s.begin(), s.end(), kPrefix), e; i != e; ++i) {
            const int32_t st = static_cast<int32_t>(i->position());
            const int32_t en = st + static_cast<int32_t>(i->length());
            out.emplace_back(st, en, i->str(1));
        }
        return out;
    };
    const auto checkSpan = [&](const gl::StrSpan& span, const std::string& copy) {
        const std::vector<Occ> ref = oracleSeq(copy);
        std::vector<Occ> mine;
        gl::scanItLevPrefixOccurrences(span, [&](int32_t st, int32_t en,
                                                 int32_t d1s, int32_t d1l) {
            mine.emplace_back(st, en,
                std::string(span.ptr + d1s, static_cast<size_t>(d1l)));
        });
        ASSERT_TRUE(mine == ref);
    };
    const std::string cases[] = {
        "",                          // none
        "(=[x])",                    // none, token-free
        "it_1_lev_2_",               // one
        "it_1_lev_2_it_3_lev_4_",    // back-to-back
        "it_3_lev_4_5",              // full shape: consume it_3_lev_4_, resume at 5
        "it_1_lev",                  // missing tail
        "it__lev_1_",                // empty first run -> no match
        "it_1_lev__",                // empty second run -> no match
        "xit_5_lev_10_y",            // mid-string, multi-digit second run
        "it_12_lev_34_it_1_lev_2_",  // multi-digit first + second occurrence
    };
    for (const std::string& s : cases) checkSpan(gl::StrSpan(s), s);

    // Mid-buffer slice with a matching span and live tail bytes.
    const std::string buf1 = "ZZit_7_lev_8_9QQ";
    checkSpan(gl::StrSpan(buf1.data() + 2, 11),          // "it_7_lev_8_"
              std::string(buf1.data() + 2, 11));
    // Boundary respect: the trailing '_' lies OUTSIDE the span, so no match,
    // even though the backing buffer would match if read past span.len.
    const std::string buf2 = "it_7_lev_88_QQ";
    checkSpan(gl::StrSpan(buf2.data(), 11),              // "it_7_lev_88" (no trailing _)
              std::string(buf2.data(), 11));
}

// scanSingleDistinctIntLev — verdict (0/1/2) and the sole-distinct token value
// match the verbatim former allowedForMail collection (regex_search gate +
// sregex_iterator std::set dedup). std::regex lives in the TEST only. Covers:
// none, one, duplicates-of-one-token -> 1, two distinct -> 2, and the
// second-distinct-lex-smaller case (verdict 2 either way; the value is unread
// when verdict != 1).
TEST(str_ops, single_distinct_int_lev_scan_matches_regex_oracle) {
    static const std::regex kInt(R"(int_lev_\d+_\d+)");
    const auto oracle = [&](const std::string& s, std::string& outVar) -> int {
        if (!std::regex_search(s, kInt)) return 0;
        std::set<std::string> intVars;
        for (std::sregex_iterator i(s.begin(), s.end(), kInt), e; i != e; ++i)
            intVars.insert(i->str());
        if (intVars.size() != 1) return 2;
        outVar = *intVars.begin();
        return 1;
    };
    const std::string cases[] = {
        "(=[x,zero])",                    // none -> 0
        "int_lev_0_1",                    // one -> 1
        "(f[int_lev_0_1,int_lev_0_1])",   // duplicates of one -> 1
        "(g[int_lev_0_1,int_lev_2_3])",   // two distinct -> 2
        "(g[int_lev_9_9,int_lev_0_1])",   // second lex-smaller distinct -> 2
        "",                               // empty -> 0
    };
    for (const std::string& s : cases) {
        std::string refVar;
        const int refV = oracle(s, refVar);
        gl::StrSpan myVar;
        const int myV = gl::scanSingleDistinctIntLev(gl::StrSpan(s), myVar);
        ASSERT_EQ(myV, refV);
        if (refV == 1) {
            ASSERT_TRUE(std::string(myVar.ptr, static_cast<size_t>(myVar.len)) == refVar);
        }
    }
    // Mid-buffer slice with live tail bytes.
    const std::string buf = "QQint_lev_5_6ZZ";
    gl::StrSpan myVar;
    const int myV = gl::scanSingleDistinctIntLev(gl::StrSpan(buf.data() + 2, 11), myVar);
    ASSERT_EQ(myV, 1);
    ASSERT_TRUE(std::string(myVar.ptr, static_cast<size_t>(myVar.len)) == "int_lev_5_6");
}

// allIntLevLevelsBelow — the parent-level mail-gate predicate matches a regex
// oracle that extracts every int_lev token's level and requires all of them
// strictly below the bound. std::regex lives in the TEST only. Covers: no
// tokens (vacuously true), single parent-level, own-level (equal -> false),
// deeper-than-bound, mixed parent+own, duplicates, multi-digit levels, bound
// 0, and a mid-buffer slice with live tail bytes.
TEST(str_ops, all_int_lev_levels_below_matches_regex_oracle) {
    static const std::regex kInt(R"(int_lev_(\d+)_\d+)");
    const auto oracle = [&](const std::string& s, int32_t bound) -> bool {
        for (std::sregex_iterator i(s.begin(), s.end(), kInt), e; i != e; ++i)
            if (std::stoll((*i)[1].str()) >= bound) return false;
        return true;
    };
    const std::pair<std::string, int32_t> cases[] = {
        { "(=[x,zero])", 3 },                       // none -> true
        { "", 1 },                                  // empty -> true
        { "(in[int_lev_1_1,1])", 2 },               // parent level -> true
        { "(in[int_lev_2_1,1])", 2 },               // own level -> false
        { "(in[int_lev_3_1,1])", 2 },               // deeper -> false
        { "(g[int_lev_1_1,int_lev_2_1])", 3 },      // both parent -> true
        { "(g[int_lev_1_1,int_lev_2_1])", 2 },      // mixed -> false
        { "(g[int_lev_1_1,int_lev_1_1])", 2 },      // duplicates parent -> true
        { "(g[int_lev_10_2,int_lev_9_1])", 11 },    // multi-digit levels -> true
        { "(g[int_lev_10_2,int_lev_9_1])", 10 },    // multi-digit boundary -> false
        { "(in[int_lev_0_1,1])", 0 },               // bound 0 -> false
    };
    for (const auto& c : cases) {
        ASSERT_EQ(gl::allIntLevLevelsBelow(gl::StrSpan(c.first), c.second),
                  oracle(c.first, c.second));
    }
    // Mid-buffer slice with live tail bytes: span covers "int_lev_5_6" only.
    const std::string buf = "QQint_lev_5_6ZZ";
    ASSERT_TRUE(gl::allIntLevLevelsBelow(gl::StrSpan(buf.data() + 2, 11), 6));
    ASSERT_TRUE(!gl::allIntLevLevelsBelow(gl::StrSpan(buf.data() + 2, 11), 5));
}

// containsItLevPrefixShape — existence verdict matches regex_search over
// it_\d+_lev_\d+_. std::regex lives in the TEST only.
TEST(str_ops, contains_it_lev_prefix_matches_regex_oracle) {
    static const std::regex kPrefix(R"(it_\d+_lev_\d+_)");
    const std::string cases[] = {
        "", "(=[x])", "it_1_lev_2_", "it_3_lev_4_5", "it_1_lev", "it__lev_1_",
        "xit_5_lev_10_y", "prefix_it_2_lev_3_suffix", "it_1_lev__",
    };
    for (const std::string& s : cases) {
        ASSERT_EQ(gl::containsItLevPrefixShape(gl::StrSpan(s)),
                  std::regex_search(s, kPrefix));
    }
}

// containsCDigit — existence verdict matches regex_search over c\d+. std::regex
// lives in the TEST only. Covers the recipe's "c" (false), "ac9" (true),
// "c_1" (false), plus mid-word and adjacency cases.
TEST(str_ops, contains_c_digit_matches_regex_oracle) {
    static const std::regex kC(R"(c\d+)");
    const std::string cases[] = {
        "", "c", "ac9", "c_1", "abc123", "c0", "xcx", "9c", "cc5", "c9c",
    };
    for (const std::string& s : cases) {
        ASSERT_EQ(gl::containsCDigit(gl::StrSpan(s)),
                  std::regex_search(s, kC));
    }
}

TEST(str_ops, collected_arena) {
    // disintegrateExpr2's `collected` accumulator off the heap: append-only
    // records on a scratch arena, retrieved by key. Inserts copy into the arena
    // (dup), so the source temporaries need not outlive the accumulator.
    gl::GlobalMemoryManager m;
    m.init(kStrOpsTestCfg);
    gl::ScratchArena a;
    a.bind(&m);
    gl::ExpressionAnalyzer::CollectedArena collected(&a);

    const std::string s1 = "S1", s2 = "S2", s3 = "S3";
    const std::string i1 = "(impl1)", i2 = "(impl2)", i3 = "(impl3)";
    const std::string vMain = "main", vV2 = "v2";
    const std::string c1 = "C1", c2 = "C2";

    collected.insertImpl(gl::StrSpan(s1), gl::StrSpan(i1), gl::StrSpan(vMain));
    collected.insertImpl(gl::StrSpan(s1), gl::StrSpan(i2), gl::StrSpan(vV2));
    collected.insertChild(gl::StrSpan(s1), gl::StrSpan(c1));
    collected.insertChild(gl::StrSpan(s1), gl::StrSpan(c2));
    collected.ensureKey(gl::StrSpan(s2));               // empty-key marker
    collected.insertImpl(gl::StrSpan(s3), gl::StrSpan(i3), gl::StrSpan(vMain));

    // Implications under S1, in insertion order, as "original|validity".
    std::vector<std::string> impls;
    collected.forImpls(gl::StrSpan(s1), [&](const gl::StrSpan& o, const gl::StrSpan& v) {
        impls.push_back(o.toStdString() + "|" + v.toStdString());
    });
    ASSERT_EQ(impls.size(), static_cast<size_t>(2));
    ASSERT_EQ(impls[0], "(impl1)|main");
    ASSERT_EQ(impls[1], "(impl2)|v2");

    // Children under S1.
    std::vector<std::string> kids;
    collected.forChildren(gl::StrSpan(s1), [&](const gl::StrSpan& c) {
        kids.push_back(c.toStdString());
    });
    ASSERT_EQ(kids.size(), static_cast<size_t>(2));
    ASSERT_EQ(kids[0], "C1");
    ASSERT_EQ(kids[1], "C2");

    // The empty-key S2 carries no impls/children but IS a distinct key.
    int s2impls = 0;
    collected.forImpls(gl::StrSpan(s2), [&](const gl::StrSpan&, const gl::StrSpan&) { ++s2impls; });
    ASSERT_EQ(s2impls, 0);

    // Distinct keys, first-seen order: S1, S2, S3.
    std::vector<std::string> keys;
    collected.forEachKey([&](const gl::StrSpan& k) { keys.push_back(k.toStdString()); });
    ASSERT_EQ(keys.size(), static_cast<size_t>(3));
    ASSERT_EQ(keys[0], "S1");
    ASSERT_EQ(keys[1], "S2");
    ASSERT_EQ(keys[2], "S3");
}

TEST(str_ops, disintegrate_scratch_twins) {
    // The three string twins disintegrateExpr2 / disintegrateExprCore2 use to
    // keep their leaf calculation strings off the heap: each must be
    // BYTE-IDENTICAL to its ExpressionAnalyzer member original.
    gl::ExpressionAnalyzer ea("Peano");  // member originals + static pool
    gl::GlobalMemoryManager m;
    m.init(kStrOpsTestCfg);
    gl::ScratchArena a;
    a.bind(&m);

    // removeUPrefixScratch == removeUPrefixFromArguments;
    // addMissingUScratch  == addMissingU. Corpus exercises u_-stripping,
    // u_-completion, the "marker" skip, an already-u_ "u_marker", a no-bracket
    // token, an empty arg list, and a nested form (top-level args only).
    const std::string corpus[] = {
        "(in2[i0,v1,s])", "(in3[u_1,u_2,s,+])", "(=[u_a,b])", "(p[x])",
        "(AnchorPeano[N,i0,s,+,*,i1])", "(q[marker,u_1,z])", "(r[u_marker])",
        "no_brackets", "(p[])", "(>[u_x](a[u_x,y])(b[u_x]))",
    };
    for (const std::string& s : corpus) {
        ASSERT_EQ(ea.removeUPrefixScratch(a, gl::StrSpan(s)).toStdString(),
                  ea.removeUPrefixFromArguments(s));
        ASSERT_EQ(ea.addMissingUScratch(a, gl::StrSpan(s)).toStdString(),
                  ea.addMissingU(s));
    }

    // reconstructImplicationFullBindScratch == reconstructImplicationFullBind:
    // empty key pass-through, u_ filtering, multi-premise nesting, negated
    // premises.
    struct Case { std::vector<std::string> key; std::string value; };
    const Case cases[] = {
        { {}, "(p[x])" },
        { { "(a[x,u_y])" }, "(b[x])" },
        { { "(a[x])", "(b[y])" }, "(c[x,y])" },
        { { "(in[x,u_1])", "(in2[x,u_1,s])" }, "(in3[x,u_1,s,+])" },
        { { "!(p[a])", "!(q[b])" }, "(r[a,b])" },
    };
    for (const Case& c : cases) {
        std::vector<gl::StrSpan> keySpans;
        keySpans.reserve(c.key.size());
        for (const std::string& k : c.key) keySpans.push_back(gl::StrSpan(k));
        const std::string ref = ea.reconstructImplicationFullBind(c.key, c.value);
        const gl::ScratchString out = ea.reconstructImplicationFullBindScratch(
            a, keySpans.data(), static_cast<int>(keySpans.size()), gl::StrSpan(c.value));
        ASSERT_EQ(out.toStdString(), ref);
    }

    // makeMarkedExprScratch == makeMarkedExpr (var -> "marker"; var-absent unchanged).
    struct Mk { std::string expr; std::string var; };
    const Mk marks[] = {
        { "(in[it_0_lev_0_0,u_1])", "it_0_lev_0_0" },
        { "(in2[int_lev_0_1,u_1,s])", "int_lev_0_1" },
        { "(p[x,y])", "z" },
    };
    for (const Mk& mk : marks) {
        ASSERT_EQ(ea.makeMarkedExprScratch(a, gl::StrSpan(mk.expr), gl::StrSpan(mk.var)).toStdString(),
                  ea.makeMarkedExpr(mk.expr, mk.var));
    }
}

TEST(prover, renaming_chain2_scratch_matches_vector) {
    // renamingChain2Scratch == renamingChain2: renamed elements element-for-element
    // AND the startIntRepl progression byte-for-byte. u_-args map to their
    // u_-stripped slice; every other distinct arg mints a fresh
    // repl_lev_<level>_<startIntRepl> in first-seen order. Both forms run from an
    // identical (level, startIntRepl) start so their counter walks compare.
    gl::ExpressionAnalyzer ea("Peano");
    gl::GlobalMemoryManager m;
    m.init(kStrOpsTestCfg);
    gl::ScratchArena a;
    a.bind(&m);

    struct Case { std::vector<std::string> chain; };
    const Case cases[] = {
        { {} },                                          // empty chain
        { { "(in[u_1,u_2])" } },                         // u_-strip only (no fresh)
        { { "(p[x])" } },                                // one fresh repl assignment
        { { "(p[x,y])", "(q[x,z])" } },                  // multi-element + cross-element dedup
        { { "(in[u_1,x])", "(in2[u_1,y,x])" } },         // mixed u_ + fresh, repeats
    };

    for (const Case& c : cases) {
        gl::Memory memHeap; memHeap.level = 3; memHeap.startIntRepl = 5;
        gl::Memory memScr;  memScr.level  = 3; memScr.startIntRepl  = 5;

        const std::vector<std::string> ref = ea.renamingChain2(c.chain, memHeap);

        std::vector<gl::StrSpan> spans;
        spans.reserve(c.chain.size());
        for (const std::string& s : c.chain) spans.push_back(gl::StrSpan(s));
        gl::StrSpan out[gl::ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
        const int32_t n = ea.renamingChain2Scratch(
            a, spans.data(), static_cast<int32_t>(spans.size()), memScr,
            out, gl::ExecutionParameters::MAX_INSTRUCTION_ELEMENTS);

        ASSERT_EQ(n, static_cast<int32_t>(ref.size()));
        for (int32_t i = 0; i < n; ++i)
            ASSERT_EQ(out[i].toStdString(), ref[static_cast<size_t>(i)]);
        ASSERT_EQ(memScr.startIntRepl, memHeap.startIntRepl);
    }
}

// S10 C3 builder twins: each == its retained heap oracle, byte-for-byte.
TEST(prover, integration_builder_scratch_twins) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::GlobalMemoryManager m;
    m.init(kStrOpsTestCfg);
    gl::ScratchArena a;
    a.bind(&m);

    const auto sp2s = [](const gl::StrSpan& s) {
        return std::string(s.ptr, static_cast<size_t>(s.len));
    };

    // (i) negateScratch == negate: bang-strip / bang-prefix / empty.
    const std::string negs[] = { "x", "!x", "", "!(=[a,b])", "(p[y])" };
    for (const std::string& s : negs)
        ASSERT_EQ(sp2s(ea.negateScratch(a, gl::StrSpan(s))), ea.negate(s));

    // (ii) stripUPrefixASTScratch == stripUPrefixAST: no-u_ / single / multiple /
    //      nested-compound / duplicate token.
    const std::string strips[] = {
        "(p[x,y])", "(in[u_1,y])", "(in3[u_1,u_2,u_3])",
        "(&(a[u_x,u_y])(>[u_z](b[u_z])(c[u_x,u_z])))", "(q[u_1,u_1,u_2])",
    };
    for (const std::string& s : strips)
        ASSERT_EQ(ea.stripUPrefixASTScratch(a, gl::StrSpan(s)).toStdString(),
                  ea.stripUPrefixAST(s));

    // (iii) reconstructImplicationForIntegrationScratch == oracle: pi_lev-bound
    //       (kept) / occupied-bound (collapsed) / non-(>[ shape.
    struct RCase { std::vector<std::string> key; std::string value; };
    const RCase rcases[] = {
        { { "(a[pi_lev_0_1])" }, "(b[pi_lev_0_1])" },     // bound var pi_lev -> kept
        { { "(a[z])" }, "(b[z])" },                        // occupied -> (>[]
        { {}, "(p[x])" },                                  // no premise -> plain value
    };
    for (const RCase& c : rcases) {
        std::vector<gl::StrSpan> ks;
        for (const std::string& k : c.key) ks.push_back(gl::StrSpan(k));
        ASSERT_EQ(ea.reconstructImplicationForIntegrationScratch(
                      a, ks.data(), static_cast<int32_t>(ks.size()),
                      gl::StrSpan(c.value)).toStdString(),
                  ea.reconstructImplicationForIntegration(c.key, c.value));
    }

    // (iv) expandSignatureForIntegrationScratch == expandSignatureForIntegration:
    //      one per category + a multi-element and-fold + the hasPiBoundVars out-param.
    struct ECase { std::string cat; std::vector<std::string> chain; std::string sig; };
    const ECase ecases[] = {
        { "and",         { "(p[a])", "(q[b])", "(r[c])" }, "(sig[a,b,c])" },   // multi-fold
        { "and",         { "(p[a])" },                     "(sig[a])" },       // single
        { "existence",   { "(body[a,pi_lev_0_1])", "(head[a])" }, "(sig[a])" },// allPi
        { "existence",   { "(body[a,z])", "(head[a])" },   "(sig[a])" },       // occupied
        { "implication", { "(prem[a])", "(concl[a])" },    "(sig[a])" },       // implication
    };
    for (const ECase& c : ecases) {
        bool hasPiRef = false, hasPiScr = false;
        const std::string ref =
            ea.expandSignatureForIntegration(c.cat, c.chain, c.sig, &hasPiRef);
        std::vector<gl::StrSpan> cs;
        for (const std::string& e : c.chain) cs.push_back(gl::StrSpan(e));
        const std::string got = ea.expandSignatureForIntegrationScratch(
            a, gl::StrSpan(c.cat), cs.data(), static_cast<int32_t>(cs.size()),
            gl::StrSpan(c.sig), &hasPiScr).toStdString();
        ASSERT_EQ(got, ref);
        ASSERT_EQ(hasPiScr, hasPiRef);
    }

    // (v) buildIntegrationInstructionScratch == buildIntegrationInstruction:
    //     BOTH fields (history, hash) byte-compared.
    struct BCase { std::vector<std::string> elems; std::string sig; };
    const BCase bcases[] = {
        { { "(a[pi_lev_0_1])" }, "(b[pi_lev_0_1])" },   // pi_lev kept in history
        { { "(a[z])" }, "(b[z])" },                      // occupied stripped in history
        { { "(a[u_1])" }, "(b[u_1])" },                  // non-(>[ shape
    };
    for (const BCase& c : bcases) {
        const std::pair<std::string, std::string> ref =
            ea.buildIntegrationInstruction(c.elems, c.sig);
        std::vector<gl::StrSpan> es;
        for (const std::string& e : c.elems) es.push_back(gl::StrSpan(e));
        gl::ScratchString hist, hash;
        ea.buildIntegrationInstructionScratch(a, es.data(),
            static_cast<int32_t>(es.size()), gl::StrSpan(c.sig), hist, hash);
        ASSERT_EQ(hist.toStdString(), ref.first);
        ASSERT_EQ(hash.toStdString(), ref.second);
    }
}

// Mis-frame regression guard for the integration builders. The (v) twins above
// consumed their results IMMEDIATELY, before any post-rewind fill, so a result
// born under a hidden internal ScratchScope that rewinds before return would
// still have to survive only the natural post-builder cursor. This test forces
// a caller alloc+rewind cycle on the SAME arena AFTER the builder returns, then
// re-reads the result: a correctly-framed builder (result born at the caller-
// visible cursor) reads byte-identical; a mis-framed one (result born under an
// internal scope, birthEnd above the returned cursor) fires ScratchString::
// assertLive on the read. Codifies the contract the production Case B bug broke
// — a result born under prepareIntegrationCore2's igScope, consumed after it
// closed — at the builder-twin level so the class stays unit-catchable.
TEST(prover, integration_builder_result_survives_post_return_rewind) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::GlobalMemoryManager m;
    m.init(kStrOpsTestCfg);
    gl::ScratchArena a;
    a.bind(&m);

    // Open a scope at the current (post-builder) cursor, allocate + fill, then
    // rewind. The result lives BELOW this scope's mark, so a well-framed builder
    // survives; the fill proves the result bytes are a separate region.
    const auto cycleRewind = [&]() {
        gl::ScratchScope guard(a);
        char* junk = a.allocBytes(256);
        for (int i = 0; i < 256; ++i) junk[i] = 'Z';
    };

    // (i) reconstructImplicationForIntegrationScratch — the exact builder whose
    //     result the production Case B mis-frame consumed after igScope closed.
    {
        const std::vector<std::string> key = { "(a[z])" };
        const std::string value = "(b[z])";
        std::vector<gl::StrSpan> ks;
        for (const std::string& k : key) ks.push_back(gl::StrSpan(k));
        const gl::ScratchString r = ea.reconstructImplicationForIntegrationScratch(
            a, ks.data(), static_cast<int32_t>(ks.size()), gl::StrSpan(value));
        cycleRewind();
        ASSERT_EQ(r.toStdString(),
                  ea.reconstructImplicationForIntegration(key, value));
    }

    // (ii) expandSignatureForIntegrationScratch.
    {
        const std::string cat = "existence";
        const std::vector<std::string> chain = { "(body[a,z])", "(head[a])" };
        const std::string sig = "(sig[a])";
        bool hasPiRef = false, hasPiScr = false;
        const std::string ref =
            ea.expandSignatureForIntegration(cat, chain, sig, &hasPiRef);
        std::vector<gl::StrSpan> cs;
        for (const std::string& e : chain) cs.push_back(gl::StrSpan(e));
        const gl::ScratchString r = ea.expandSignatureForIntegrationScratch(
            a, gl::StrSpan(cat), cs.data(), static_cast<int32_t>(cs.size()),
            gl::StrSpan(sig), &hasPiScr);
        cycleRewind();
        ASSERT_EQ(r.toStdString(), ref);
        ASSERT_EQ(hasPiScr, hasPiRef);
    }

    // (iii) buildIntegrationInstructionScratch — BOTH fields (the production
    //       mis-frame: outHash from reconstructImplicationForIntegrationScratch
    //       was consumed AFTER the caller scope closed).
    {
        const std::vector<std::string> elems = { "(a[z])" };
        const std::string sig = "(b[z])";
        const std::pair<std::string, std::string> ref =
            ea.buildIntegrationInstruction(elems, sig);
        std::vector<gl::StrSpan> es;
        for (const std::string& e : elems) es.push_back(gl::StrSpan(e));
        gl::ScratchString hist, hash;
        ea.buildIntegrationInstructionScratch(a, es.data(),
            static_cast<int32_t>(es.size()), gl::StrSpan(sig), hist, hash);
        cycleRewind();
        ASSERT_EQ(hist.toStdString(), ref.first);
        ASSERT_EQ(hash.toStdString(), ref.second);
    }
}

// sortOriginalChainIndex yields the ids in ID ORDER (1..count) — the
// registry's deterministic insertion order. The decoded-lex sort it once
// carried is retired (maintainer directive 2026-09-01): it existed only to
// reproduce the pre-statification snapshot's byte order and cost 35% of all
// phase-1/3 runtime.
TEST(prover, sort_original_chain_index_yields_id_order) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::Memory mem;

    // Seed ruleInterner + originals with several DISTINCT chains, out of lex order.
    const std::vector<std::vector<std::string>> chains = {
        { "(cZ)", "(aA)" },
        { "(aA)" },
        { "(aA)", "(bB)", "(cC)" },
        { "(aA)", "(bB)" },
        { "(bB)" },
    };
    for (const std::vector<std::string>& ch : chains) {
        gl::IdVecKey k;
        for (const std::string& e : ch) k.ids.push_back(mem.ruleInterner.encode(e));
        mem.overallHashMemory.originals.mint(k);
    }
    const int32_t count = mem.overallHashMemory.originals.count();
    ASSERT_EQ(count, static_cast<int32_t>(chains.size()));

    std::vector<int32_t> out(static_cast<size_t>(count));
    const int32_t n = ea.sortOriginalChainIndex(mem, out.data(), count);
    ASSERT_EQ(n, count);
    for (int32_t k = 0; k < n; ++k)
        ASSERT_EQ(out[static_cast<size_t>(k)], k + 1);
}

// prefixArgumentsWithUScratch == prefixArgumentsWithU: u_-prefix EVERY arg
// (no marker skip; an already-u_ arg is prefixed again — NOT special-cased).
TEST(prover, prefix_arguments_with_u_scratch_matches_string) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::GlobalMemoryManager m;
    m.init(kStrOpsTestCfg);
    gl::ScratchArena a;
    a.bind(&m);

    const std::string cases[] = {
        "no_brackets",          // no args -> plain pass-through
        "(p[])",                // empty arg list
        "(p[x])",               // single arg
        "(in2[a,7,3])",         // multi arg
        "(q[u_1,y,marker])",    // already-u_ prefixed again; marker also prefixed
    };
    for (const std::string& s : cases)
        ASSERT_EQ(ea.prefixArgumentsWithUScratch(a, gl::StrSpan(s)).toStdString(),
                  ea.prefixArgumentsWithU(s));
}

TEST(str_ops, addstatement_shape_helper_span_twins) {
    // The four addStatement shape helpers (isEquality / isNegatedEquality /
    // extractMaxIterationNumber / countPatternOccurrences) gain StrSpan
    // overloads so a span-holding caller needs no std::string. Each must be
    // BYTE-IDENTICAL to its std::string original. `ea("Peano")` initialises the
    // process static pool (mirrors the other ExpressionAnalyzer-member twins);
    // `gl::Memory m` then default-binds a usable nameMap + overallHashMemory.
    gl::ExpressionAnalyzer ea("Peano");
    gl::Memory m;

    // isEquality / isNegatedEquality — the `(=[` and `!(=[` prefixes plus the
    // sub-length edge cases the len/size guard must short-circuit before any
    // operator[].
    const std::string shapes[] = {
        "(=[a,b])", "(=[", "(=", "(", "",
        "!(=[a,b])", "!(=[", "!(=", "!(", "!",
        "(p[x])", "(in2[a,b,s])", "x", "((=[a,b]))",
        "(=[u_x,it_0_lev_0_0])",
    };
    for (const std::string& s : shapes) {
        ASSERT_EQ(ea.isEquality(gl::StrSpan(s)), ea.isEquality(s));
        ASSERT_EQ(ea.isNegatedEquality(gl::StrSpan(s)), ea.isNegatedEquality(s));
    }

    // extractMaxIterationNumber — none (-1 sentinel), single, and multi-match
    // maxima across interleaved lev indices.
    const std::string iters[] = {
        "(p[x])", "(in[it_0_lev_0_0,s])",
        "(in[it_3_lev_1_0,it_1_lev_0_2])",
        "(in[it_2_lev_5_0,it_7_lev_0_0,it_4_lev_2_1])",
        "it_10_lev_0_0", "no_iter_here", "",
    };
    for (const std::string& s : iters) {
        ASSERT_EQ(ea.extractMaxIterationNumber(gl::StrSpan(s)),
                  ea.extractMaxIterationNumber(s));
    }

    // countPatternOccurrences — a fresh Memory's productsOfRecursionIds is
    // empty and its nameMap knows no lexeme, so every it_<i>_lev_<l>_<n> match
    // counts; the span/string twin equality holds for ANY registry contents
    // since both forms run identical lookups on identical match bytes.
    const std::string counts[] = {
        "(p[x])",
        "(in[it_0_lev_0_0,s])",
        "(in[it_0_lev_0_0,it_1_lev_0_1,it_2_lev_0_2])",
        "it_0_lev_0_0 and it_5_lev_3_9",
        "",
    };
    for (const std::string& s : counts) {
        ASSERT_EQ(ea.countPatternOccurrences(gl::StrSpan(s),
                                             m.overallHashMemory, m.nameMap),
                  ea.countPatternOccurrences(s, m.overallHashMemory, m.nameMap));
    }
}

TEST(str_ops, new_var_store) {
    // disintegrateExpr2's newVarMap off the heap: var -> ordered defining elements,
    // O(1) membership, and decoded-lex key iteration (Pass B's admission order).
    gl::GlobalMemoryManager m;
    m.init(kStrOpsTestCfg);
    gl::ScratchArena a;
    a.bind(&m);
    gl::ExpressionAnalyzer::NewVarStore nv(&a);

    // Insert keys out of lex order to prove forEachVarSorted re-sorts.
    const std::string vC = "it_0_lev_0_2", vA = "int_lev_0_0", vB = "it_0_lev_0_1";
    const std::string e1 = "(in[c_x,u_1])", e2 = "(in2[c_x,u_1,s])", e3 = "(p[c_y])";
    nv.addElement(gl::StrSpan(vC), gl::StrSpan(e1));
    nv.addElement(gl::StrSpan(vC), gl::StrSpan(e2));
    nv.addElement(gl::StrSpan(vA), gl::StrSpan(e3));
    nv.addElement(gl::StrSpan(vB), gl::StrSpan(e1));

    ASSERT_FALSE(nv.empty());
    ASSERT_EQ(nv.varCount(), 3);
    ASSERT_TRUE(nv.hasVar(gl::StrSpan(vA)));
    ASSERT_FALSE(nv.hasVar(gl::StrSpan("it_9_lev_9_9", 12)));

    // vC's run preserves insertion order (e1, e2).
    const int32_t cId = nv.lookupVar(gl::StrSpan(vC));
    ASSERT_EQ(nv.elemCount(cId), 2);
    ASSERT_EQ(nv.elemAt(cId, 0).toStdString(), e1);
    ASSERT_EQ(nv.elemAt(cId, 1).toStdString(), e2);

    // forEachVarSorted visits keys in std::map lex order:
    // "int_lev_0_0" < "it_0_lev_0_1" < "it_0_lev_0_2".
    std::vector<std::string> order;
    nv.forEachVarSorted([&](int32_t id) { order.push_back(nv.varAt(id).toStdString()); });
    ASSERT_EQ(order.size(), static_cast<size_t>(3));
    ASSERT_EQ(order[0], vA);
    ASSERT_EQ(order[1], vB);
    ASSERT_EQ(order[2], vC);
}

TEST(str_ops, rejection_store) {
    // disintegrateExpr2's pendingRejections / pendingRejectionsIntegration off the
    // heap: an append-only flat list of interned-id records, drained in order.
    gl::GlobalMemoryManager m;
    m.init(kStrOpsTestCfg);
    gl::ScratchArena a;
    a.bind(&m);
    gl::ExpressionAnalyzer::RejectionStore rs(&a);

    // Algebra-style record: compactExpr + a sorted level set.
    const std::string vA = "it_0_lev_0_0", ru1 = "(in[c_x])", mk1 = "(in[marker])",
                      cx1 = "(&(in[c_x])(p[c_x]))", sa = "(p[a])", sb = "(q[b])";
    const gl::StrSpan sibsA[2] = { gl::StrSpan(sa), gl::StrSpan(sb) };
    const int32_t levsA[3] = { 1, 2, 5 };
    rs.addRejection(gl::StrSpan(vA), gl::StrSpan(ru1), gl::StrSpan(mk1),
                    gl::StrSpan(cx1), sibsA, 2, levsA, 3);
    // Integration-style record: no compactExpr, no levels.
    const std::string vB = "int_lev_0_1", ru2 = "(in2[c_y])", mk2 = "(in2[marker])";
    rs.addRejection(gl::StrSpan(vB), gl::StrSpan(ru2), gl::StrSpan(mk2),
                    gl::StrSpan(), nullptr, 0, nullptr, 0);

    ASSERT_EQ(rs.count(), 2);
    const gl::ExpressionAnalyzer::RejRec& r0 = rs.recAt(0);
    ASSERT_EQ(rs.decode(r0.varId).toStdString(), vA);
    ASSERT_EQ(rs.decode(r0.s1Id).toStdString(), ru1);
    ASSERT_EQ(rs.decode(r0.s2Id).toStdString(), mk1);
    ASSERT_EQ(rs.decode(r0.s3Id).toStdString(), cx1);
    ASSERT_EQ(r0.sibCount, 2);
    ASSERT_EQ(rs.sibAt(r0, 0).toStdString(), sa);
    ASSERT_EQ(rs.sibAt(r0, 1).toStdString(), sb);
    ASSERT_EQ(r0.levCount, 3);
    ASSERT_EQ(rs.levAt(r0, 0), 1);
    ASSERT_EQ(rs.levAt(r0, 2), 5);

    const gl::ExpressionAnalyzer::RejRec& r1 = rs.recAt(1);
    ASSERT_EQ(rs.decode(r1.varId).toStdString(), vB);
    ASSERT_EQ(r1.s3Id, 0);
    ASSERT_TRUE(rs.decode(r1.s3Id).empty());
    ASSERT_EQ(r1.sibCount, 0);
    ASSERT_EQ(r1.levCount, 0);
}

TEST(str_ops, collect_expr_tokens_twin) {
    // collectExprTokens (flat bracket scan) must yield the SAME token SET as the
    // ce::parseExpr tree walk + per-node getArgs that expandSignature formerly used.
    auto parseWalk = [](const std::string& e) {
        std::set<std::string> tk;
        ce::TreeNode1* root = ce::parseExpr(e);
        std::vector<ce::TreeNode1*> st;
        if (root) st.push_back(root);
        while (!st.empty()) {
            ce::TreeNode1* c = st.back(); st.pop_back();
            for (const std::string& t : ce::getArgs(c->value)) {
                size_t f = t.find_first_not_of(" \t\r\n");
                if (f != std::string::npos) {
                    size_t l = t.find_last_not_of(" \t\r\n");
                    tk.insert(t.substr(f, l - f + 1));
                }
            }
            if (c->left) st.push_back(c->left);
            if (c->right) st.push_back(c->right);
        }
        ce::deleteTree(root);
        return tk;
    };
    auto scan = [](const std::string& e) {
        std::set<std::string> s;
        gl::collectExprTokens(gl::StrSpan(e),
            [&](const gl::StrSpan& t) { s.insert(t.toStdString()); });
        return s;
    };
    const std::string corpus[] = {
        "(&(in[x,y])(in2[a,b,s]))",
        "!(>[c_z](in[c_z,u_1])!(in2[c_z,u_2,s]))",
        "!(&!(p[x])!(q[y]))",
        "(>[i,j](in[i,j])(out[i,j,k]))",
        "(in3[u_1,u_2,s,+])",
        "!(&!(&!(a[x])!(b[y]))!(c[z]))",
    };
    for (const std::string& e : corpus) {
        ASSERT_TRUE(parseWalk(e) == scan(e));
    }
}

TEST(str_ops, replace_keys_twin_canonical) {
    gl::GlobalMemoryManager m;
    m.init(kStrOpsTestCfg);
    gl::ScratchArena a;
    a.bind(&m);
    ASSERT_TRUE(replaceKeysTwinMatches(a, "(in2[i0,v1,s])",
                                       { { "v1", "a" } }));
    ASSERT_TRUE(replaceKeysTwinMatches(a, "(p[x,x,x])",
                                       { { "x", "yy" } }));
    // Value longer and shorter than key; empty value.
    ASSERT_TRUE(replaceKeysTwinMatches(
        a, "(in3[i0,i1,v1,+])",
        { { "i0", "zero" }, { "i1", "" }, { "v1", "w" } }));
    // Empty map degrades to identity.
    ASSERT_TRUE(replaceKeysTwinMatches(a, "(=[a,b])", {}));
}

TEST(str_ops, replace_keys_twin_boundary_edges) {
    gl::GlobalMemoryManager m;
    m.init(kStrOpsTestCfg);
    gl::ScratchArena a;
    a.bind(&m);
    // Key not at a token boundary: no replacement.
    ASSERT_TRUE(replaceKeysTwinMatches(a, "(pv1[v1x,av1])",
                                       { { "v1", "Z" } }));
    // Key at string start (no preceding delimiter): no replacement.
    ASSERT_TRUE(replaceKeysTwinMatches(a, "v1,[v1]",
                                       { { "v1", "Z" } }));
    // Greedy longest with no shorter-key retry: "ab" matches, fails the
    // next-char test on 'c', and "a" is NOT retried.
    ASSERT_TRUE(replaceKeysTwinMatches(a, "(p[abc])",
                                       { { "a", "A" }, { "ab", "B" } }));
    // Longest beats shorter when both qualify.
    ASSERT_TRUE(replaceKeysTwinMatches(a, "(p[v1,v])",
                                       { { "v", "A" }, { "v1", "B" } }));
    // Match reaching end-of-string does not qualify (no next delimiter).
    ASSERT_TRUE(replaceKeysTwinMatches(a, "(p[a,v1", { { "v1", "Z" } }));
    // Bracket-bearing keys (marker maps carry compound sub-expressions).
    ASSERT_TRUE(replaceKeysTwinMatches(a, "(in2[(s[a]),N])",
                                       { { "(s[a])", "marker" } }));
}

TEST(str_ops, replace_keys_to_string_twin) {
    // The no-arena std::string-returning twin; same canonical + boundary inputs
    // as the replaceKeysScratch twin tests, pinned byte-identical to the original.
    ASSERT_TRUE(replaceKeysToStringTwinMatches("(in2[i0,v1,s])", { { "v1", "a" } }));
    ASSERT_TRUE(replaceKeysToStringTwinMatches("(p[x,x,x])", { { "x", "yy" } }));
    ASSERT_TRUE(replaceKeysToStringTwinMatches(
        "(in3[i0,i1,v1,+])", { { "i0", "zero" }, { "i1", "" }, { "v1", "w" } }));
    ASSERT_TRUE(replaceKeysToStringTwinMatches("(=[a,b])", {}));
    ASSERT_TRUE(replaceKeysToStringTwinMatches("(pv1[v1x,av1])", { { "v1", "Z" } }));
    ASSERT_TRUE(replaceKeysToStringTwinMatches("v1,[v1]", { { "v1", "Z" } }));
    ASSERT_TRUE(replaceKeysToStringTwinMatches("(p[abc])", { { "a", "A" }, { "ab", "B" } }));
    ASSERT_TRUE(replaceKeysToStringTwinMatches("(p[v1,v])", { { "v", "A" }, { "v1", "B" } }));
    ASSERT_TRUE(replaceKeysToStringTwinMatches("(p[a,v1", { { "v1", "Z" } }));
    ASSERT_TRUE(replaceKeysToStringTwinMatches("(in2[(s[a]),N])", { { "(s[a])", "marker" } }));
}

// addSanitizeSubstPair / sortSanitizeSubstPairs (C2) — duplicate keys with the
// same value dedup like std::map operator[] overwrite, and the sorted final
// (key, value) sequence equals iterating a std::map built by the same writes.
TEST(memory, sanitize_subst_pairs_dedup_and_sort_match_map) {
    const std::pair<std::string, std::string> writes[] = {
        { "it_0_lev_1_2", "int_lev_0_1" },
        { "int_lev_5_5", "int_lev_0_2" },
        { "it_0_lev_1_2", "int_lev_0_1" },   // duplicate key, same value
        { "aaa", "bbb" },
        { "int_lev_5_5", "int_lev_0_2" },    // duplicate again
    };
    gl::StrReplacement pairs[8];
    int32_t pairN = 0;
    std::map<std::string, std::string> oracle;
    for (const auto& w : writes) {
        gl::addSanitizeSubstPair(pairs, pairN,
            gl::StrSpan(w.first), gl::StrSpan(w.second), 8);
        oracle[w.first] = w.second;
    }
    gl::sortSanitizeSubstPairs(pairs, pairN);
    ASSERT_EQ(static_cast<std::size_t>(pairN), oracle.size());
    int32_t i = 0;
    for (const auto& kv : oracle) {
        ASSERT_EQ(pairs[i].key.toStdString(), kv.first);
        ASSERT_EQ(pairs[i].value.toStdString(), kv.second);
        ++i;
    }
}

// The frozen order-free substitution contract (C2): replaceKeysToString over
// the staged pair array equals ce::replaceKeysInString over the equivalent
// std::map, on sanitize-shaped sources, for BOTH ascending and deliberately
// reversed pair order (the output is a pure function of the pair SET; greedy
// longest match at each token boundary is unique).
TEST(memory, replace_keys_pairs_match_map_oracle_sanitize_shapes) {
    const std::pair<std::string, std::string> kv[] = {
        { "it_0_lev_1_2", "int_lev_0_1" },
        { "it_0_lev_1_22", "int_lev_0_9" },   // proper-prefix sibling key
        { "x", "int_lev_3_3" },
    };
    const std::string sources[] = {
        "(in3[int_lev_0_1,x,+])",
        "(>[7,8](in3[7,8,x,+])(in3[8,7,it_0_lev_1_2,+]))",  // repeated args
        "(P[it_0_lev_1_2])",                    // key at last arg before ]
        "(P[it_0_lev_1_22,it_0_lev_1_2])",      // prefix pair, both positions
        "(P[xit_0_lev_1_2,prefix_x])",          // not at token boundary -> no hit
        "(=[x,x])",
    };
    std::map<std::string, std::string> oracleMap;
    for (const auto& p : kv) oracleMap[p.first] = p.second;

    gl::StrReplacement asc[4];
    int32_t ascN = 0;
    for (const auto& p : kv)
        gl::addSanitizeSubstPair(asc, ascN,
            gl::StrSpan(p.first), gl::StrSpan(p.second), 4);
    gl::sortSanitizeSubstPairs(asc, ascN);

    gl::StrReplacement rev[4];
    int32_t revN = 0;
    for (int i = 2; i >= 0; --i)
        gl::addSanitizeSubstPair(rev, revN,
            gl::StrSpan(kv[i].first), gl::StrSpan(kv[i].second), 4);
    // rev is deliberately NOT sorted — pins pair-order independence.

    for (const std::string& src : sources) {
        const std::string expect = ce::replaceKeysInString(src, oracleMap);
        ASSERT_EQ(gl::replaceKeysToString(gl::StrSpan(src), asc, ascN), expect);
        ASSERT_EQ(gl::replaceKeysToString(gl::StrSpan(src), rev, revN), expect);
    }
}

TEST(str_ops, replace_u_substrings_twin) {
    gl::GlobalMemoryManager m;
    m.init(kStrOpsTestCfg);
    gl::ScratchArena a;
    a.bind(&m);
    gl::ExpressionAnalyzer analyzer{ "Peano" };
    const std::string cases[] = {
        "(p[u_v1,a])",        // token-start strip
        "(p[a,u_v1])",
        "(p[u_u_x])",         // erase-and-recheck chain
        "[u_u_x",
        "[u_",                // chain ending at string end
        "[u_u_",
        "u_a",                // position 0 never stripped
        "(p[au_,b])",         // mid-token u_ untouched
        "(p[v1,a])",          // identity
        "x",                  // shorter than 2
        "",
        "(in2[u_i0,u_v1,s])"  // multiple tokens
    };
    for (const std::string& s : cases) {
        const gl::ScratchString out =
            gl::replaceUSubstringsScratch(a, gl::StrSpan(s));
        ASSERT_EQ(out.toStdString(), analyzer.replaceUSubstrings(s));
    }
}

TEST(str_ops, copy_and_split) {
    gl::GlobalMemoryManager m;
    m.init(kStrOpsTestCfg);
    gl::ScratchArena a;
    a.bind(&m);
    const std::string mid = "a,b";
    const gl::ScratchString copied = gl::copyScratch(a, gl::StrSpan(mid));
    ASSERT_EQ(copied.toStdString(), mid);
    // splitByCharSpans: pieces, empties at adjacent delimiters and edges.
    const std::string joined = "main_boundary_x__y_";
    gl::StrSpan parts[8];
    const int32_t n =
        gl::splitByCharSpans(gl::StrSpan(joined), '_', parts, 8);
    ASSERT_EQ(n, 6);
    ASSERT_EQ(parts[0].toStdString(), std::string("main"));
    ASSERT_EQ(parts[1].toStdString(), std::string("boundary"));
    ASSERT_EQ(parts[2].toStdString(), std::string("x"));
    ASSERT_EQ(parts[3].toStdString(), std::string());
    ASSERT_EQ(parts[4].toStdString(), std::string("y"));
    ASSERT_EQ(parts[5].toStdString(), std::string());
}

TEST(str_ops, remove_expression_span_door_twin) {
    // removeExpressionFromMemoryBlock(StrSpan,StrSpan,...) must erase the same
    // (originalId, validityId) rows from all three encoded-statement vectors as
    // the EncodedExpression overload (which now delegates to it). Build two
    // identical Memories, remove the same target through each overload, and
    // assert the surviving rows are identical (byte-twin). `ea("Peano")`
    // initialises the process static pool.
    gl::ExpressionAnalyzer ea("Peano");

    const std::string TARGET = "(=[a,b])";
    const std::string TVAL = "main";
    const std::string OTHER = "(=[c,d])";   // different expression, same scope
    const std::string TVAL2 = "s1";         // TARGET again under a different scope

    auto build = [&](gl::Memory& m) {
        auto push = [&](const std::string& e, const std::string& v) {
            const gl::EncodedExpression enc(e, v);
            const auto ie = gl::encodeExpression(enc, m.nameMap);
            m.intEncodedStatements.push_back(ie);
            m.intLocalEncodedStatements.push_back(ie);
            m.intLocalEncodedStatementsDelta.push_back(ie);
        };
        push(TARGET, TVAL);
        push(OTHER, TVAL);
        push(TARGET, TVAL2);
        push(TARGET, TVAL);   // duplicate target row — both (TARGET,main) must go
    };

    gl::Memory mSpan;  build(mSpan);
    gl::Memory mEnc;   build(mEnc);

    ea.removeExpressionFromMemoryBlock(gl::StrSpan(TARGET), gl::StrSpan(TVAL), mSpan, 0);
    ea.removeExpressionFromMemoryBlock(gl::EncodedExpression(TARGET, TVAL), mEnc, 0);

    // Positive: both (TARGET,main) rows gone from every vector; OTHER and the
    // TARGET-under-s1 row survive. Span door == EncodedExpression door.
    ASSERT_EQ(mSpan.intEncodedStatements.size(), mEnc.intEncodedStatements.size());
    ASSERT_EQ(mSpan.intEncodedStatements.size(), 2);
    ASSERT_EQ(mSpan.intLocalEncodedStatements.size(), 2);
    ASSERT_EQ(mSpan.intLocalEncodedStatementsDelta.size(), 2);
    for (int32_t i = 0; i < mSpan.intEncodedStatements.size(); ++i) {
        ASSERT_EQ(mSpan.intEncodedStatements[i].originalId,
                  mEnc.intEncodedStatements[i].originalId);
        ASSERT_EQ(mSpan.intEncodedStatements[i].validityId,
                  mEnc.intEncodedStatements[i].validityId);
    }

    // Negative: a never-interned target is a no-op (lookup miss), not a crash.
    const std::string NOPE = "(=[z,z])";
    const int32_t before = mSpan.intEncodedStatements.size();
    ea.removeExpressionFromMemoryBlock(gl::StrSpan(NOPE), gl::StrSpan(TVAL), mSpan, 0);
    ASSERT_EQ(mSpan.intEncodedStatements.size(), before);
}

TEST(str_ops, classify_or_scope_view_twin) {
    // classifyOrScopeView must return the SAME kind as classifyOrScope and, when
    // the scope IS an OR scope, the SAME orSig / branch-body bytes as spans that
    // the owning overload materialises as std::strings. Covers Disintegration,
    // Integration, the root (empty stack), a non-or boundary, and a malformed
    // (no `_(` separator) payload.
    gl::ExpressionAnalyzer ea("Peano");
    gl::Memory m;
    using OSK = gl::ExpressionAnalyzer::OrScopeKind;

    struct Case { std::string scope; OSK expect; };
    const Case cases[] = {
        { "main_boundary_ordis_(=[a,b])_((=[c,d]))", OSK::Disintegration },
        { "main_boundary_orint_(=[e,f])_((=[g,h]))", OSK::Integration },
        { "main",                                    OSK::NotOrScope },  // root: empty stack
        { "main_boundary_something_else",            OSK::NotOrScope },  // non-or boundary
        { "main_boundary_ordis_nosephere",           OSK::NotOrScope },  // malformed: no `_(`
        // Goal-carrying payloads classify by their bare half; the compact
        // subproof shape stays NotOrScope.
        { "main_boundary_(interval[1,4,2,7,10])_subproof_orint_(=[e,f])_((=[g,h]))",
          OSK::Integration },
        { "main_boundary_!(in[7,1])_subproof_ordis_(=[a,b])_((=[c,d]))",
          OSK::Disintegration },
        { "main_boundary_(interval[1,4,2,7,10])_subproof_(implication22[1,4,2,7,10])",
          OSK::NotOrScope },
    };
    for (const Case& c : cases) {
        const int16_t vid = m.nameMap.encode(c.scope);
        std::string sig, body;
        const OSK k1 = ea.classifyOrScope(m.nameMap, vid, sig, body);
        gl::StrSpan sigV, bodyV;
        const OSK k2 = ea.classifyOrScopeView(m.nameMap, vid, sigV, bodyV);
        ASSERT_TRUE(k1 == c.expect);
        ASSERT_TRUE(k2 == k1);
        if (k1 != OSK::NotOrScope) {
            ASSERT_EQ(sigV.toStdString(), sig);
            ASSERT_EQ(bodyV.toStdString(), body);
        }
    }
}

TEST(str_ops, split_subproof_payload_grammar) {
    // splitSubproofPayload must accept exactly the `<goal>_subproof_<bare>`
    // grammar — optionally-negated balanced-paren goal, literal separator,
    // non-empty bare half — and reject every legacy payload shape.
    // stripSubproofPrefixView returns the bare half on a match and the
    // payload unchanged otherwise.
    gl::ExpressionAnalyzer ea("Peano");

    struct Ok { const char* payload; const char* goal; const char* bare; };
    const Ok oks[] = {
        { "(interval[1,4,2,7,10])_subproof_(implication22[1,4,2,7,10])",
          "(interval[1,4,2,7,10])", "(implication22[1,4,2,7,10])" },
        { "!(in[7,1])_subproof_(implication5[a,b])",
          "!(in[7,1])", "(implication5[a,b])" },
        // Goal containing a nested compact — balanced-paren parse keeps the
        // whole goal as the prefix.
        { "(>[1](implication7[a,b])(in[1,2]))_subproof_orint_(or1[a,b])_((=[a,X]))",
          "(>[1](implication7[a,b])(in[1,2]))", "orint_(or1[a,b])_((=[a,X]))" },
    };
    for (const Ok& c : oks) {
        gl::StrSpan g, b;
        ASSERT_TRUE(ea.splitSubproofPayload(
            gl::StrSpan(c.payload, static_cast<int32_t>(std::strlen(c.payload))),
            g, b));
        ASSERT_EQ(g.toStdString(), std::string(c.goal));
        ASSERT_EQ(b.toStdString(), std::string(c.bare));
        ASSERT_EQ(ea.stripSubproofPrefixView(
                      gl::StrSpan(c.payload,
                                  static_cast<int32_t>(std::strlen(c.payload))))
                      .toStdString(),
                  std::string(c.bare));
    }

    const char* rejects[] = {
        "(implication22[1,4,2,7,10])",               // legacy compact: nothing after group
        "orint_(or1[a,b])_((=[a,X]))",               // legacy orint: no leading paren
        "_var0_2_var1_7_hypo_(interval[1,4,2,7,10])",// hypo payload
        "product_of_hypo_disintegration_of_integration_goal_(in[1,2])",
        "(in[7,1]_subproof_(x)",                     // unbalanced goal group
        "(in[7,1])_subproof_",                       // empty bare half
        "(in[7,1])_subproo_(x)",                     // wrong separator
        "",                                          // empty payload
    };
    for (const char* r : rejects) {
        gl::StrSpan g, b;
        ASSERT_FALSE(ea.splitSubproofPayload(
            gl::StrSpan(r, static_cast<int32_t>(std::strlen(r))), g, b));
        ASSERT_EQ(ea.stripSubproofPrefixView(
                      gl::StrSpan(r, static_cast<int32_t>(std::strlen(r))))
                      .toStdString(),
                  std::string(r));
    }
}

// deduplicateBoundVarsScratch — the span twin of the former file-static heap
// deduplicateBoundVars (deleted from prover.cpp). The oracle below is a
// byte-identical copy of that heap form, kept TEST-LOCAL per the ce-twin
// survivorship discipline: a prover-only heap helper with a static twin is
// deleted from production, its oracle lives in tests (never dead production
// heap). The twin must match it byte-for-byte across duplicate vars, u_ strip,
// multiple >[...] lists, no-list passthrough, empty list, and empty input.
static std::string deduplicateBoundVarsOracle(const std::string& impl) {
    std::string result;
    result.reserve(impl.size());
    std::size_t i = 0;
    while (i < impl.size()) {
        if (i + 2 < impl.size() && impl[i] == '(' && impl[i + 1] == '>' && impl[i + 2] == '[') {
            result += "(>[";
            i += 3;
            std::size_t closeBracket = impl.find(']', i);
            if (closeBracket != std::string::npos) {
                std::string varList = impl.substr(i, closeBracket - i);
                std::vector<std::string> vars;
                std::set<std::string> seen;
                std::size_t pos = 0;
                while (pos <= varList.size()) {
                    std::size_t comma = varList.find(',', pos);
                    std::string var;
                    if (comma == std::string::npos) {
                        var = varList.substr(pos);
                        pos = varList.size() + 1;
                    } else {
                        var = varList.substr(pos, comma - pos);
                        pos = comma + 1;
                    }
                    if (!var.empty() && seen.find(var) == seen.end()) {
                        if (var.size() >= 2 && var[0] == 'u' && var[1] == '_') continue;
                        seen.insert(var);
                        vars.push_back(var);
                    }
                }
                for (std::size_t j = 0; j < vars.size(); ++j) {
                    if (j > 0) result += ',';
                    result += vars[j];
                }
                i = closeBracket;
            }
        } else {
            result += impl[i];
            ++i;
        }
    }
    return result;
}

TEST(str_ops, deduplicate_bound_vars_scratch_matches_heap) {
    gl::GlobalMemoryManager m;
    m.init(kStrOpsTestCfg);
    gl::ScratchArena a; a.bind(&m);

    const std::vector<std::string> cases = {
        "(>[v0,v1,v0](p[v0])(q[v0,v1]))",   // duplicate vars
        "(>[u_a,v0,u_a,v1,v0](p[v0]))",     // u_ strip + duplicate
        "(>[a,a,b](>[c,c](r[c])))",         // multiple >[...] lists
        "(p[x])(q[y])",                     // no bound-var list -> passthrough
        "(>[u_a,u_b](p[u_a]))",             // all-u_ list -> empty var run
        "(>[](p[x]))",                      // empty list
        "(>[,a,,b,](r[a]))",                // empty tokens dropped
        "",                                 // empty input
        "x",                                // single char / no list
    };
    for (const std::string& impl : cases) {
        gl::ScratchScope scope(a);
        const std::string twin =
            gl::deduplicateBoundVarsScratch(gl::StrSpan(impl), a).toStdString();
        ASSERT_EQ(twin, deduplicateBoundVarsOracle(impl));
    }
}

// replaceArgSpanScratch — the span twin of the still-live file-static heap
// replaceArgInString (prover.cpp; deleted in a later commit once its sole
// caller multiplyImplication goes span-native). The oracle below is a
// byte-identical copy of that heap body, kept TEST-LOCAL per the ce-twin
// survivorship discipline (a prover-only heap helper is not linkable from the
// tests, so the oracle is a verbatim copy). Word-boundary rule, DELIBERATELY
// NOT the [/, … ]/, token rule of replaceKeysScratch.
static std::string replaceArgInStringOracle(const std::string& str,
                                            const std::string& oldArg,
                                            const std::string& newArg) {
    auto isWordChar = [](char c) -> bool {
        return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
    };
    std::string result;
    result.reserve(str.size());
    std::size_t i = 0;
    while (i < str.size()) {
        bool atBoundary = (i == 0) || !isWordChar(str[i - 1]);
        if (atBoundary && str.compare(i, oldArg.size(), oldArg) == 0) {
            std::size_t after = i + oldArg.size();
            bool endBoundary = (after >= str.size()) || !isWordChar(str[after]);
            if (endBoundary) {
                result += newArg;
                i = after;
                continue;
            }
        }
        result += str[i];
        ++i;
    }
    return result;
}

// numericLess — the file-static heap comparator (prover.cpp) whose only caller
// multiplyImplication goes span-native later; oracle kept TEST-LOCAL (verbatim).
static bool numericLessOracle(const std::string& a, const std::string& b) {
    bool aDigit = !a.empty() && std::all_of(a.begin(), a.end(),
                      [](char c) { return std::isdigit(static_cast<unsigned char>(c)); });
    bool bDigit = !b.empty() && std::all_of(b.begin(), b.end(),
                      [](char c) { return std::isdigit(static_cast<unsigned char>(c)); });
    if (aDigit && bDigit) return std::stoi(a) < std::stoi(b);
    return a < b;
}

TEST(str_ops, replace_arg_span_scratch_matches_heap) {
    gl::GlobalMemoryManager m;
    m.init(kStrOpsTestCfg);
    gl::ScratchArena a; a.bind(&m);

    struct Case { const char* str; const char* oldA; const char* newA; };
    const Case cases[] = {
        { "v1", "v1", "w" },                  // whole-string / pos-0 match
        { "(p[v1,v10])", "v1", "w" },         // no match inside v10 (word char after)
        { "u_7", "7", "9" },                  // no match: 7 inside u_7 (word char before)
        { "(>[v0,v1](p[v0,v1]))", "v0", "" }, // empty newArg (deletion)
        { "(p[x,x,x])", "x", "yy" },          // multi-occurrence
        { "(=[a,b])", "z", "Q" },             // no occurrence
        { "a,v1", "v1", "Z" },                // end-of-string match qualifies
        { "v1,v1", "v1", "AB" },              // pos-0 + repeat
    };
    for (const Case& c : cases) {
        const std::string str = c.str;
        const std::string oldA = c.oldA;
        const std::string newA = c.newA;
        gl::ScratchScope scope(a);
        const std::string twin = gl::replaceArgSpanScratch(
            gl::StrSpan(str), gl::StrSpan(oldA), gl::StrSpan(newA), a).toStdString();
        ASSERT_EQ(twin, replaceArgInStringOracle(str, oldA, newA));
    }
}

TEST(str_ops, numeric_less_span_matches_numeric_less) {
    struct Pair { const char* a; const char* b; };
    const Pair pairs[] = {
        { "2", "10" },    // numeric: 2 < 10 (lex would say 10 < 2)
        { "10", "9" },    // numeric-vs-lex disagreement: 9 < 10 but lex "10" < "9"
        { "10", "10" },   // equal digit runs
        { "3", "3" },
        { "v1", "v2" },   // non-digit lex
        { "v10", "v2" },  // mixed: not all-digit -> lex
        { "2", "v2" },    // one digit one not -> lex
        { "", "5" },      // empty -> lex
        { "abc", "abd" },
        { "100", "20" },
    };
    for (const Pair& p : pairs) {
        const std::string a = p.a;
        const std::string b = p.b;
        ASSERT_EQ(gl::numericLessSpan(gl::StrSpan(a), gl::StrSpan(b)),
                  numericLessOracle(a, b));
        ASSERT_EQ(gl::numericLessSpan(gl::StrSpan(b), gl::StrSpan(a)),
                  numericLessOracle(b, a));
    }
}
