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
/// ties within Integration before the decoded lexical name is compared. A
/// test variable (`test_level_<level>_<serial>`) names the open output of an
/// induction hypothesis head inside the usefulness scope `is_useful` only;
/// it ranks last, so a class never chooses it over any other name.
enum class WitnessNameKind : uint8_t { Ordinary, Forced, Integration, Algebra, Test };

/// @brief Decoded fields of one permanent witness name.
/// @details Depth belongs only to Integration; generation belongs only to
/// Algebra. Forced and Test names carry location and identity without either
/// budget.
struct WitnessName {
    WitnessNameKind kind = WitnessNameKind::Ordinary;
    int depth = 0;
    int generation = 0;
    int level = 0;
    int serial = 0;
};

/// @brief Recognize a permanent witness starting at an expression offset.
///
/// @details The four spellings are disjoint. A lexical mismatch is a defined
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
    } else if (literal("test_level_", 11)) {
        value.kind = WitnessNameKind::Test;
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

/// @brief Test whether one argument token is a number, with a given name in
///        `rec`'s role.
///
/// @details The decimals and @p recRole are numbers, nothing else: a weak step
/// block's own integrations pass `rec`; a copy bound to the second premise of
/// a twin conjecture integrates that premise at `rec2`, the name of the twin
/// partner's predecessor, which takes `rec`'s role there, so it passes `rec2`,
/// and `rec` is then not a number (D-589). A preparation
/// `u_` prefix is stripped before the token is classified.
/// @param argument One complete argument token, possibly `u_`-prefixed.
/// @param recRole  The name in `rec`'s role (nonempty, no `u_` prefix).
/// @return True when the token is a decimal or @p recRole.
/// @invariant Pure lexical read; no allocation, interner or proof-state access.
/// @see isNumberArgument, hasOnlyNumberFreeArguments.
inline bool isNumberArgument(StrSpan argument, StrSpan recRole) {
    if (startsWithSpan(argument, "u_", 2))
        argument = StrSpan(argument.ptr + 2, argument.len - 2);
    assert(argument.len > 0 && "isNumberArgument: empty argument");
    assert(recRole.len > 0 && !startsWithSpan(recRole, "u_", 2)
        && "isNumberArgument: a bare name in rec's role");
    if (equalSpans(argument, recRole)) return true;
    for (int32_t j = 0; j < argument.len; ++j)
        if (!isDecimalDigitByte(argument.ptr[j])) return false;
    return true;
}

/// @brief Test whether one argument token is a number.
///
/// @details A number is a nonempty decimal token, the induction variable
/// `rec` or `rec2`: an original variable of the statement being proved, an
/// anchor slot included. `rec2` names the twin partner's predecessor in a twin
/// weak step block as `rec` names the induction variable's, through the
/// variable copy `(=[q,rec2])`, and has `rec`'s standing here
/// (D-581). Every other spelling is not a number:
/// the three witness classes (`int_`, `it_`, `force_lev_`), anchor-slot
/// copies such as `x2`, variable copies such as `2_copy`, replacement names
/// and the like. A preparation `u_` prefix is stripped before the token is
/// classified, as in the algebra-input gate of `disintegrateExpr2`, so `u_9`
/// is a number and `u_int_0_lev_1_2` is not.
///
/// It is the per-argument test of @ref hasOnlyNumberArguments (I-285, the
/// hidden-argument witnesses of I-292 and the subproof gate of I-293) and the
/// operator-parameter test of the standalone operator-input admission
/// `AdmissionPolicy::isAllowedAsOperatorInput`
/// (D-471).
/// @param argument One complete argument token, possibly `u_`-prefixed.
/// @return True when the token is a number.
/// @invariant Pure lexical read; no allocation, interner or proof-state access.
/// @see hasOnlyNumberArguments, AdmissionPolicy::isAllowedAsOperatorInput.
inline bool isNumberArgument(StrSpan argument) {
    const StrSpan bare = startsWithSpan(argument, "u_", 2)
        ? StrSpan(argument.ptr + 2, argument.len - 2) : argument;
    return equalSpans(bare, StrSpan("rec2", 4)) || isNumberArgument(argument, StrSpan("rec", 3));
}

/// @brief Test whether every argument of an original expression is a number.
///
/// @details Each argument is classified by @ref isNumberArgument. The
/// arguments are the flat top-level list that `getArgsSpans` reads, so a
/// leading negation is transparent and an expression without arguments
/// passes. An induction admission serves a disintegration only when the
/// expression being disintegrated passes this test (I-285); it is the
/// stricter sibling of the integration depth read by
/// `assignedIntegrationDepth`.
/// @param expression Complete flat expression text.
/// @return True when every argument is a number.
/// @invariant Pure lexical read; no allocation, interner or proof-state access.
/// @see isNumberArgument, assignedIntegrationDepth, getArgsSpans.
inline bool hasOnlyNumberArguments(StrSpan expression) {
    StrSpan arguments[ExecutionParameters::MAX_ARITY];
    const int32_t count = getArgsSpans(expression, arguments, ExecutionParameters::MAX_ARITY);
    for (int32_t i = 0; i < count; ++i)
        if (!isNumberArgument(arguments[i])) return false;
    return true;
}

/// @brief Test whether every free argument of an expression is a number,
///        with a given name in `rec`'s role.
///
/// @details As @ref hasOnlyNumberFreeArguments, each free argument classified
/// by @ref isNumberArgument with @p recRole in `rec`'s role: the gate of a
/// copy bound to the second twin premise reads `rec2`, the name of the twin
/// partner's predecessor, as its `rec` (D-589).
/// @param expression Complete expression text.
/// @param recRole    The name in `rec`'s role (nonempty, no `u_` prefix).
/// @return True when every free argument is a decimal or @p recRole.
/// @invariant Pure lexical read; no allocation, interner or proof-state
///            access; at most 64 bound variables (asserted).
/// @see isNumberArgument, hasOnlyNumberFreeArguments.
inline bool hasOnlyNumberFreeArguments(StrSpan expression, StrSpan recRole) {
    constexpr int32_t kMaxBound = 64;
    StrSpan bound[kMaxBound];
    int32_t boundN = 0;
    // Calls sink(token) for every token of the argument list opening at
    // `open`, and returns the index of its closing bracket.
    const auto forEachToken = [&](int32_t open, auto sink) {
        int32_t close = open + 1;
        while (close < expression.len && expression.ptr[close] != ']') ++close;
        int32_t start = open + 1;
        for (int32_t p = open + 1; p <= close; ++p) {
            if (p != close && expression.ptr[p] != ',') continue;
            if (p > start) sink(StrSpan(expression.ptr + start, p - start));
            start = p + 1;
        }
        return close;
    };
    const auto isBinder = [&](int32_t open) { return open > 0 && expression.ptr[open - 1] == '>'; };
    for (int32_t i = 0; i < expression.len; ++i) {
        if (expression.ptr[i] != '[' || !isBinder(i)) continue;
        i = forEachToken(i, [&](StrSpan token) {
            assert(boundN < kMaxBound && "hasOnlyNumberFreeArguments: too many bound variables");
            bound[boundN++] = token;
        });
    }
    bool numbers = true;
    for (int32_t i = 0; i < expression.len && numbers; ++i) {
        if (expression.ptr[i] != '[' || isBinder(i)) continue;
        i = forEachToken(i, [&](StrSpan token) {
            for (int32_t b = 0; b < boundN; ++b)
                if (equalSpans(bound[b], token)) return;
            if (!isNumberArgument(token, recRole)) numbers = false;
        });
    }
    return numbers;
}

/// @brief Test whether every free argument of an expression is a number.
///
/// @details The free arguments are the tokens of every argument list
/// `[...]` of the expression except the binder lists (`>[...]`, the bound
/// variables of an implication or existence), with the tokens a binder of
/// the expression binds left out. Each is classified by the role form of
/// @ref isNumberArgument with `rec` in `rec`'s role (a decimal or `rec`, a
/// `u_` prefix stripped; `rec2` is no number for this gate, which belongs to
/// the copies integrating at `rec`). A flat expression `(name[args])` without
/// `rec2` has no binder, so the test equals @ref hasOnlyNumberArguments there;
/// an existence in binder form
/// `!(>[x](...)!(...))` is judged by the names its body takes from outside.
/// A premise-bound integration copy starts its int scope only from an
/// arriving expression that passes (D-578).
/// @param expression Complete expression text.
/// @return True when every free argument is a number.
/// @invariant Pure lexical read; no allocation, interner or proof-state
///            access; at most 64 bound variables (asserted).
/// @see isNumberArgument, hasOnlyNumberArguments.
inline bool hasOnlyNumberFreeArguments(StrSpan expression) {
    return hasOnlyNumberFreeArguments(expression, StrSpan("rec", 3));
}

/// @brief Test whether an original may serve the strong step block's
///        induction demand (D-502).
///
/// @details Every flat top-level argument, a preparation `u_` prefix
/// stripped, must be a decimal token whose value is not one of @p refused.
/// `rec`, the witness classes, anchor copies and every other spelling fail,
/// and so does a refused digit. A strong step block refuses the lemma's
/// element-valued numbers that fill no input slot of a non-anchor premise or
/// of the head (`ProofGridBuilder::inductionParameterDigits`), so only the
/// lemma's input values and its sets and operators pass. The stricter
/// sibling of @ref hasOnlyNumberArguments, which serves the recursion-owned
/// admission values (I-285).
/// @param expression Complete flat expression text.
/// @param refused Refused digit values (null when @p refusedCount is zero).
/// @param refusedCount Number of @p refused.
/// @return True when every argument passes.
/// @invariant Pure lexical read; no allocation, interner or proof-state access.
/// @see hasOnlyNumberArguments, parseDigitRunToInt.
inline bool hasOnlyInductionDigits(StrSpan expression, const int32_t* refused, int32_t refusedCount) {
    assert(refusedCount >= 0 && (refusedCount == 0 || refused != nullptr)
        && "hasOnlyInductionDigits: a refused list without storage");
    StrSpan arguments[ExecutionParameters::MAX_ARITY];
    const int32_t count = getArgsSpans(expression, arguments, ExecutionParameters::MAX_ARITY);
    for (int32_t i = 0; i < count; ++i) {
        StrSpan argument = arguments[i];
        if (startsWithSpan(argument, "u_", 2))
            argument = StrSpan(argument.ptr + 2, argument.len - 2);
        assert(argument.len > 0 && "hasOnlyInductionDigits: empty argument");
        int value = 0;
        if (!parseDigitRunToInt(argument.ptr, argument.len, value)) return false;
        for (int32_t r = 0; r < refusedCount; ++r)
            if (refused[r] == value) return false;
    }
    return true;
}

/// @brief The bit that stands for one member of a forbidden set.
///
/// @details A forbidden set of an integration instruction copy bound to one
/// premise holds decimal variables or logic-block levels
/// (D-579). It is a 64-bit mask in which
/// member v is bit v. The members are small: the variables of a conjecture
/// and the levels of its logic-block chain. A member outside [0, 64) cannot
/// be represented and asserts.
/// @param member The member.
/// @return The mask with only bit @p member set.
/// @invariant Pure; no allocation.
/// @see forbiddenSetHas, premiseForbiddenSets.
inline uint64_t forbiddenSetBit(int member) {
    assert(member >= 0 && member < 64 && "forbiddenSetBit: member outside [0, 64)");
    return uint64_t{1} << member;
}

/// @brief Test whether a value is a member of a forbidden set.
///
/// @details The set is a 64-bit mask built by @ref forbiddenSetBit, so only
/// values in [0, 64) can be members; any other value is not a member
/// (D-579).
/// @param set The mask.
/// @param value The value.
/// @return True when bit @p value of @p set is set.
/// @invariant Pure; no allocation.
/// @see forbiddenSetBit, argumentsHitForbiddenSets.
inline bool forbiddenSetHas(uint64_t set, int value) {
    return value >= 0 && value < 64 && ((set >> value) & 1u) != 0;
}

/// @brief Compute the two forbidden sets of the instruction copy bound to one
///        premise of an identical-goal group.
///
/// @details @p premises are the original premises whose hypothesis premises
/// prepare the group's identical integration goals, @p levels the levels of
/// their logic blocks. The copy bound to premise @p assigned forbids every
/// decimal variable of another premise that does not occur in premise
/// @p assigned, and the level of every other premise
/// (D-579). The premises are the keys of
/// ancestor logic blocks of an induction step block, so every top-level
/// argument is a decimal variable (asserted), and their levels are pairwise
/// distinct (asserted).
/// @param premises The group's original premises.
/// @param levels Their logic-block levels.
/// @param count Number of premises, at least two.
/// @param assigned Index of the premise the copy is bound to.
/// @param variablesOut Receives the forbidden variables as a mask.
/// @param levelsOut Receives the forbidden levels as a mask.
/// @return Void.
/// @invariant Pure; no allocation.
/// @see forbiddenSetBit, argumentsHitForbiddenSets.
inline void premiseForbiddenSets(const StrSpan* premises, const int32_t* levels, int32_t count,
                                 int32_t assigned, uint64_t& variablesOut, uint64_t& levelsOut) {
    assert(count >= 2 && assigned >= 0 && assigned < count
        && "premiseForbiddenSets: a group of at least two premises and one of them assigned");
    const auto variablesOf = [](StrSpan premise) {
        StrSpan arguments[ExecutionParameters::MAX_ARITY];
        const int32_t n = getArgsSpans(premise, arguments, ExecutionParameters::MAX_ARITY);
        uint64_t set = 0;
        for (int32_t a = 0; a < n; ++a) {
            int value = 0;
            const bool decimal = parseDigitRunToInt(arguments[a].ptr, arguments[a].len, value);
            assert(decimal && "premiseForbiddenSets: a premise argument is not a decimal variable");
            set |= forbiddenSetBit(value);
        }
        return set;
    };
    const uint64_t own = variablesOf(premises[assigned]);
    uint64_t others = 0;
    uint64_t otherLevels = 0;
    for (int32_t p = 0; p < count; ++p) {
        if (p == assigned) continue;
        assert(levels[p] != levels[assigned] && "premiseForbiddenSets: two premises share a block level");
        others |= variablesOf(premises[p]);
        otherLevels |= forbiddenSetBit(levels[p]);
    }
    variablesOut = others & ~own;
    levelsOut = otherLevels;
}

/// @brief The decimal variables and the forced-witness levels an
///        expression's signature names, as masks.
///
/// @details The arguments are the flat top-level list that `getArgsSpans`
/// reads (the expression's signature), a preparation `u_` prefix stripped as
/// in @ref hasOnlyInductionDigits. A decimal argument v sets bit v of
/// @p variablesOut; a `force_lev_<L>_<n>` witness sets bit L of
/// @p levelsOut; every other spelling (`rec`, integration and algebra
/// witnesses, anchor copies) sets nothing. A value of 64 or more sets
/// nothing either: no forbidden set can hold it (@ref forbiddenSetBit), so
/// the masks answer every forbidden-set intersection exactly
/// (D-579).
/// @param expression Complete flat expression text (an existence or a statement).
/// @param variablesOut Receives the decimal variables as a mask.
/// @param levelsOut Receives the forced-witness levels as a mask.
/// @return Void.
/// @invariant Pure lexical read; no allocation, interner or proof-state access.
/// @see argumentsHitForbiddenSets, classifyWitnessName.
inline void signatureTokenSets(StrSpan expression, uint64_t& variablesOut, uint64_t& levelsOut) {
    variablesOut = 0;
    levelsOut = 0;
    StrSpan arguments[ExecutionParameters::MAX_ARITY];
    const int32_t count = getArgsSpans(expression, arguments, ExecutionParameters::MAX_ARITY);
    for (int32_t i = 0; i < count; ++i) {
        StrSpan argument = arguments[i];
        if (startsWithSpan(argument, "u_", 2))
            argument = StrSpan(argument.ptr + 2, argument.len - 2);
        assert(argument.len > 0 && "signatureTokenSets: empty argument");
        int value = 0;
        if (parseDigitRunToInt(argument.ptr, argument.len, value)) {
            if (value < 64) variablesOut |= uint64_t{1} << value;
            continue;
        }
        const WitnessName name = classifyWitnessName(argument);
        if (name.kind == WitnessNameKind::Forced && name.level < 64) levelsOut |= uint64_t{1} << name.level;
    }
}

/// @brief Test whether an arriving expression names a forbidden variable or
///        a forced witness of a forbidden level.
///
/// @details The expression's signature tokens (@ref signatureTokenSets) are
/// intersected with the two sets: a decimal argument hits when its value is
/// in @p variables, a `force_lev_<L>_<n>` witness when L is in @p levels.
/// An integration instruction copy bound to one premise refuses an arriving
/// expression that hits its sets (D-579);
/// empty sets never refuse.
/// @param expression Complete flat expression text (an existence or a statement).
/// @param variables Forbidden decimal variables as a mask.
/// @param levels Forbidden logic-block levels of forced witnesses as a mask.
/// @return True when one argument hits a set.
/// @invariant Pure lexical read; no allocation, interner or proof-state access.
/// @see signatureTokenSets, premiseForbiddenSets.
inline bool argumentsHitForbiddenSets(StrSpan expression, uint64_t variables, uint64_t levels) {
    if (variables == 0 && levels == 0) return false;
    uint64_t named = 0;
    uint64_t namedLevels = 0;
    signatureTokenSets(expression, named, namedLevels);
    return (named & variables) != 0 || (namedLevels & levels) != 0;
}

/// @brief Test whether two arguments of an expression are spelled alike.
///
/// @details The arguments are the flat top-level list that `getArgsSpans`
/// reads, so a leading negation is transparent; a preparation `u_` prefix is
/// stripped before two arguments are compared, as in
/// @ref hasOnlyInductionDigits. An induction admission refuses an original
/// that repeats an argument, a set and an operator included: the original
/// names one object twice, as `(preorder[1,4,10,10])` does
/// (D-530).
/// @param expression Complete flat expression text.
/// @return True when two arguments are equal.
/// @invariant Pure lexical read; no allocation, interner or proof-state access.
/// @see hasOnlyNumberArguments, hasOnlyInductionDigits.
inline bool hasRepeatedArgument(StrSpan expression) {
    StrSpan arguments[ExecutionParameters::MAX_ARITY];
    const int32_t count = getArgsSpans(expression, arguments, ExecutionParameters::MAX_ARITY);
    for (int32_t i = 0; i < count; ++i) {
        if (startsWithSpan(arguments[i], "u_", 2))
            arguments[i] = StrSpan(arguments[i].ptr + 2, arguments[i].len - 2);
        for (int32_t j = 0; j < i; ++j)
            if (equalSpans(arguments[i], arguments[j])) return true;
    }
    return false;
}

/// @brief Compare bare names by the prover's canonical preference.
///
/// @details Ordinary names precede forced witnesses, then integration witnesses
/// by numeric depth, then algebraic witnesses, then test variables. Decoded
/// lexical order resolves every remaining tie, independently of interner
/// identifiers.
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
