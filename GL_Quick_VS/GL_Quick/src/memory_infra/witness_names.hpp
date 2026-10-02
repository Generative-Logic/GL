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

#include "../parameters.hpp"
#include "str_ops.hpp"
#include <cassert>
#include <climits>

namespace gl {

/// @brief Semantic class of a permanent variable name.
/// @details The order is the canonical preference. Integration depth breaks
/// ties within Integration before the decoded lexical name is compared.
enum class WitnessNameKind : uint8_t { Ordinary, Forced, Integration, Algebra };

/// @brief Decoded fields of one permanent witness name.
/// @details Depth belongs only to Integration; generation belongs only to
/// Algebra. Forced names carry location and identity without either budget.
struct WitnessName {
    WitnessNameKind kind = WitnessNameKind::Ordinary;
    int depth = 0;
    int generation = 0;
    int level = 0;
    int serial = 0;
};

/// @brief Recognize a permanent witness starting at an expression offset.
///
/// @details The three spellings are disjoint. A lexical mismatch is a defined
/// non-witness result. Numeric overflow in a recognized field asserts because
/// producer counters and budgets must fit their integer representation. Pure
/// classification skips identity decoding; the complete digit runs still match.
/// @param text Stable input text.
/// @param start Candidate start, between zero and the input length inclusive.
/// @param result Receives the class and depth, plus identity fields when requested.
/// @param decodeIdentity Whether to decode generation, level and serial as integers.
/// @return Exclusive end of a match, or -1 for a lexical non-match.
/// @invariant No allocation, interner mutation or proof-history access.
inline int32_t matchWitnessNameAt(StrSpan text, int32_t start, WitnessName& result, bool decodeIdentity = true) {
    assert(start >= 0 && start <= text.len);
    int32_t at = start;
    WitnessName value;
    const auto literal = [&](const char* bytes, int32_t count) {
        if (count > text.len - at || std::memcmp(text.ptr + at, bytes, count) != 0)
            return false;
        at += count;
        return true;
    };
    StrSpan depthDigits, generationDigits, levelDigits, serialDigits;
    const auto digits = [&](StrSpan& run) {
        const int32_t begin = at;
        while (at < text.len && isDecimalDigitByte(text.ptr[at])) ++at;
        run = StrSpan(text.ptr + begin, at - begin);
        return at != begin;
    };
    if (literal("force_lev_", 10)) {
        value.kind = WitnessNameKind::Forced;
    } else if (literal("int_", 4)) {
        value.kind = WitnessNameKind::Integration;
        if (!digits(depthDigits) || !literal("_lev_", 5)) return -1;
    } else if (literal("it_", 3)) {
        value.kind = WitnessNameKind::Algebra;
        if (!digits(generationDigits) || !literal("_lev_", 5)) return -1;
    } else return -1;
    if (!digits(levelDigits) || !literal("_", 1) || !digits(serialDigits)) return -1;
    const auto number = [&](StrSpan run, int& target) {
        const bool fits = parseDigitRunToInt(run.ptr, run.len, target);
        assert(fits && "witness numeric field exceeds int representation");
    };
    if (value.kind == WitnessNameKind::Integration) number(depthDigits, value.depth);
    if (decodeIdentity) {
        if (value.kind == WitnessNameKind::Algebra) number(generationDigits, value.generation);
        number(levelDigits, value.level);
        number(serialDigits, value.serial);
    }
    result = value;
    return at;
}

/// @brief Classify a complete bare name and expose its witness fields.
///
/// @details Ordinary names and malformed witness-like strings remain ordinary.
/// Expression occurrence scanning is a separate operation; this door requires
/// a whole-name match and never strips prefixes from the caller's name.
/// @param name Complete bare variable name.
/// @param decodeIdentity Whether to decode identity fields; false leaves them zero.
/// @return Its witness fields, with Ordinary for a whole-name non-match.
/// @see matchWitnessNameAt, forEachWitnessName.
inline WitnessName classifyWitnessName(StrSpan name, bool decodeIdentity = true) {
    WitnessName result;
    if (matchWitnessNameAt(name, 0, result, decodeIdentity) != name.len) return WitnessName{};
    return result;
}

/// @brief Visit witness-name occurrences in source text without allocating.
///
/// @details The scan is left-to-right and non-overlapping. Prefixes such as
/// u_ remain outside the returned span, matching the existing occurrence-scan
/// convention used for scoped expressions and substitution products.
/// @param text Source expression or scope payload.
/// @param sink Synchronous consumer of the matched span and decoded fields.
/// @param decodeIdentity Whether callbacks need integer identity fields; depth is always decoded.
/// @return Void.
/// @invariant No view escapes the caller-owned text lifetime.
template<class Sink>
inline void forEachWitnessName(StrSpan text, Sink&& sink, bool decodeIdentity = true) {
    for (int32_t at = 0; at < text.len;) {
        WitnessName value;
        const int32_t end = matchWitnessNameAt(text, at, value, decodeIdentity);
        if (end < 0) { ++at; continue; }
        sink(StrSpan(text.ptr + at, end - at), value);
        at = end;
    }
}

/// @brief Assign one integration depth to an outer disintegration expansion.
///
/// @details Integration names alone contribute: no such name assigns zero;
/// otherwise the result is one plus their maximum depth. Structural nesting
/// within the expansion does not increment this assignment.
/// @param source Original outer expression, including its concrete arguments.
/// @return Nonnegative depth to give every ordinary integration witness.
/// @invariant Overflow asserts before incrementing the maximum.
inline int assignedIntegrationDepth(StrSpan source) {
    int maximum = -1;
    forEachWitnessName(source, [&](StrSpan, const WitnessName& value) {
        if (value.kind == WitnessNameKind::Integration && value.depth > maximum)
            maximum = value.depth;
    }, false);
    assert(maximum < INT_MAX && "integration depth increment overflows");
    return maximum + 1;
}

/// @brief Test whether every argument of an original expression is a number.
///
/// @details A number is a nonempty decimal token or the induction variable
/// `rec`. Every other spelling is not a number: the three witness classes,
/// anchor-slot copies such as `x2`, replacement names and the like. The
/// arguments are the flat top-level list that `getArgsSpans` reads, so a
/// leading negation is transparent and an expression without arguments
/// passes. A preparation `u_` prefix is stripped before an argument is
/// classified, as in the algebra-input gate of `disintegrateExpr2`, so `u_9`
/// is a number and `u_int_0_lev_1_2` is not. An induction admission serves a disintegration only when the
/// expression being disintegrated passes this test
/// (I-285); it is the stricter sibling
/// of the integration depth read by `assignedIntegrationDepth`.
/// @param expression Complete flat expression text.
/// @return True when every argument is a number.
/// @invariant Pure lexical read; no allocation, interner or proof-state access.
/// @see assignedIntegrationDepth, getArgsSpans.
inline bool hasOnlyNumberArguments(StrSpan expression) {
    StrSpan arguments[ExecutionParameters::MAX_ARITY];
    const int32_t count = getArgsSpans(expression, arguments, ExecutionParameters::MAX_ARITY);
    for (int32_t i = 0; i < count; ++i) {
        StrSpan argument = arguments[i];
        if (startsWithSpan(argument, "u_", 2))
            argument = StrSpan(argument.ptr + 2, argument.len - 2);
        assert(argument.len > 0 && "hasOnlyNumberArguments: empty argument");
        if (equalSpans(argument, StrSpan("rec", 3))) continue;
        for (int32_t j = 0; j < argument.len; ++j)
            if (!isDecimalDigitByte(argument.ptr[j])) return false;
    }
    return true;
}

/// @brief Compare bare names by the prover's canonical preference.
///
/// @details Ordinary names precede forced witnesses, then integration witnesses
/// by numeric depth, then algebraic witnesses. Decoded lexical order resolves
/// every remaining tie, independently of interner identifiers.
/// @param left First complete bare name.
/// @param right Second complete bare name.
/// @return Negative, zero or positive according to canonical preference.
/// @see classifyWitnessName.
inline int compareWitnessNames(StrSpan left, StrSpan right) {
    const WitnessName a = classifyWitnessName(left, false);
    const WitnessName b = classifyWitnessName(right, false);
    if (a.kind != b.kind) return a.kind < b.kind ? -1 : 1;
    if (a.kind == WitnessNameKind::Integration && a.depth != b.depth)
        return a.depth < b.depth ? -1 : 1;
    return compareSpans(left, right);
}


/// @brief Test whether a bare name belongs to either disintegration witness class.
/// @details Forced and ordinary integration names share construction mechanics
/// while retaining separate mail and budget policies.
/// @param name Complete bare name.
/// @return True for Forced or Integration, false for the other classes.
inline bool isDisintegrationWitnessShape(StrSpan name) {
    const auto kind = classifyWitnessName(name, false).kind;
    return kind == WitnessNameKind::Forced || kind == WitnessNameKind::Integration;
}

/// @brief Decode location and identity of either disintegration witness class.
/// @details A whole-name match is required. Algebraic and ordinary names are
/// defined non-matches; recognized numeric overflow asserts in the parser.
/// @param name Complete bare name.
/// @param level Receives the owning block level on a match.
/// @param serial Receives the permanent identity on a match.
/// @return Whether the name is a forced or ordinary integration witness.
inline bool matchDisintegrationWitnessId(StrSpan name, int& level, int& serial) {
    const auto value = classifyWitnessName(name);
    if (value.kind != WitnessNameKind::Forced && value.kind != WitnessNameKind::Integration)
        return false;
    level = value.level; serial = value.serial;
    return true;
}

/// @brief Visit occurrences of both disintegration witness classes.
/// @details This supplies the shared special-token cache without collapsing
/// the separate name kinds. Algebraic names use the independent existing run.
/// @param text Source text whose lifetime covers the synchronous calls.
/// @param sink Consumer of each byte offset and token length.
/// @return Void.
template<class Sink>
inline void scanDisintegrationWitnessOccurrences(StrSpan text, Sink&& sink) {
    forEachWitnessName(text, [&](StrSpan token, const WitnessName& value) {
        if (value.kind == WitnessNameKind::Forced || value.kind == WitnessNameKind::Integration)
            sink(static_cast<int32_t>(token.ptr - text.ptr), token.len);
    }, false);
}

/// @brief Test whether expression text contains any ordinary integration witness.
/// @details Forced and algebraic names remain separate classes. This predicate
/// guards circulating fact/rule publication, including direct setup deposits.
/// @param expression Complete expression text, with any argument prefixes.
/// @return True when at least one ordinary integration witness occurs.
/// @invariant Pure lexical read; no permission or proof-history state.
inline bool containsIntegrationWitness(StrSpan expression) {
    bool found = false;
    forEachWitnessName(expression, [&](StrSpan, const WitnessName& value) {
        if (value.kind == WitnessNameKind::Integration) found = true;
    }, false);
    return found;
}

} // namespace gl
