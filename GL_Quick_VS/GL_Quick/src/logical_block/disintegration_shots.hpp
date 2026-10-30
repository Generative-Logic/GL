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

#include <cassert>
#include <cstdint>
#include <cstring>

namespace gl {

/// @brief Capacity of the premise chain of a universal fired by a
///        disintegration shot.
constexpr int32_t kMaxShotPremises = 8;

/// @brief Capacity of the bound variables of a universal fired by a
///        disintegration shot.
constexpr int32_t kMaxShotBoundVariables = 32;

/// @brief One firing of a universal on products: a disintegration shot
///        (I-333).
///
/// @details
/// `premiseProduct[i]` is the index, in the product array given to
/// `shootUniversal`, of the product premise `i` of the universal matched;
/// `head` is the universal's head with its bound variables replaced by the
/// matched product arguments, verbatim.
struct UniversalShot {
    /// Product index per premise of the universal.
    int32_t premiseProduct[kMaxShotPremises];
    /// Number of premises of the universal.
    int32_t premiseN = 0;
    /// The instantiated head, on the caller's arena.
    ScratchString head;
};

/// @brief Strip one leading `u_` from an argument span.
///
/// @details
/// A shot compares arguments regardless of the `u_` prefix: the compiler
/// spells a fixed argument `u_<k>` and a placeholder `u_pi_lev_0_<n>`, the
/// disintegrator spells an enclosing witness `u_<token>` and the slot's own
/// witness as the bare token. Stripping one prefix makes both spellings
/// compare equal.
///
/// @param argument An argument span.
/// @return The argument without its leading `u_`, or the argument itself.
inline StrSpan shotArgumentCore(StrSpan argument) {
    if (argument.len >= 2 && argument.ptr[0] == 'u' && argument.ptr[1] == '_')
        return StrSpan(argument.ptr + 2, argument.len - 2);
    return argument;
}

/// @brief Whether @p text contains @p token as a bracket argument.
///
/// @details
/// A token occurs where it is preceded by `[` or `,` and followed by `]` or
/// `,`, the token boundary of `replaceKeysScratch`. Used to find a bound
/// variable of a universal in its head.
///
/// @param text  An MPL text, flat or compound.
/// @param token A non-empty argument name.
/// @return True iff @p token occurs as an argument of @p text.
inline bool shotTextHasToken(StrSpan text, StrSpan token) {
    assert(token.len > 0 && "shotTextHasToken: empty token");
    for (int32_t i = 1; i + token.len < text.len; ++i) {
        const char before = text.ptr[i - 1];
        if (before != '[' && before != ',') continue;
        if (std::memcmp(text.ptr + i, token.ptr, static_cast<size_t>(token.len)) != 0) continue;
        const char after = text.ptr[i + token.len];
        if (after == ']' || after == ',') return true;
    }
    return false;
}

/// @brief Fire one universal once on a set of products: the shared shot
///        routine of the disintegration package (I-333).
///
/// @details
/// The compiler preshoots a compiled existence with this routine (products
/// and universals of its unfolding, the existence's bound variables kept as
/// placeholders), and the disintegrator repeats the same shot on the real
/// products when a path through its head is published; one routine, so the
/// compile-time template and the runtime firing cannot diverge.
///
/// The universal is a full-bind implication text: its bound variables are
/// the names of its binder lists `>[...]`, every other argument is fixed.
/// Every complete assignment of products to its premises, in product order
/// per premise (depth first, premise 0 outermost), is one shot when:
/// - each premise and its product have the same text around their one
///   bracket list (operator and polarity) and the same arity;
/// - at each position a fixed premise argument equals the product argument,
///   and a bound variable takes one product argument consistently across
///   all premises (arguments compared without one leading `u_`);
/// - every bound variable that occurs in the head was taken by a premise.
/// A premise or product with more than one bracket list is a compound and
/// never matches: shots fire on flat facts only. The head is the
/// universal's head with each bound variable replaced by its product
/// argument, verbatim. A shot fires once: the caller never offers a head
/// as a product.
///
/// 0% heap: stack arrays for the premise chain, the bound variables and the
/// backtracking state; heads on @p arena.
///
/// @param rule     The universal, a full-bind implication with at least one
///                 premise (asserted).
/// @param products The candidate products.
/// @param productN Number of products.
/// @param arena    String-tier scratch arena receiving the heads.
/// @param out      Receives the shots.
/// @param outCap   Capacity of @p out (asserted).
/// @return The number of shots written to @p out.
/// @invariant Mints nothing; the result is a pure function of the inputs and
///            their order.
/// @see UniversalShot, ce::disintegrateImplicationSpans.
inline int32_t shootUniversal(StrSpan rule, const StrSpan* products, int32_t productN,
    ScratchArena& arena, UniversalShot* out, int32_t outCap) {
    StrSpan premises[kMaxShotPremises];
    int32_t premiseN = 0;
    StrSpan bound[kMaxShotBoundVariables];
    int32_t boundN = 0;
    StrSpan head;
    ce::disintegrateImplicationSpans(rule, head,
        [&](StrSpan key, const StrSpan* boundVariables, int32_t boundVariableN) {
            assert(premiseN < kMaxShotPremises
                && "shootUniversal: premise chain exceeds kMaxShotPremises");
            premises[premiseN++] = key;
            for (int32_t b = 0; b < boundVariableN; ++b) {
                assert(boundN < kMaxShotBoundVariables
                    && "shootUniversal: bound variables exceed kMaxShotBoundVariables");
                bound[boundN++] = boundVariables[b];
            }
        });
    assert(premiseN > 0 && "shootUniversal: the universal has no premise");

    // The text around the one bracket list of a flat fact; a compound (more
    // than one bracket list) reports false.
    const auto flatShape = [](StrSpan fact, StrSpan& before, StrSpan& after) -> bool {
        int32_t open = -1, close = -1, opens = 0;
        for (int32_t i = 0; i < fact.len; ++i) {
            if (fact.ptr[i] == '[') { ++opens; if (open < 0) open = i; }
            else if (fact.ptr[i] == ']' && close < 0) close = i;
        }
        if (opens != 1 || close < open) return false;
        before = StrSpan(fact.ptr, open);
        after = StrSpan(fact.ptr + close + 1, fact.len - close - 1);
        return true;
    };
    const auto boundIndex = [&](StrSpan name) -> int32_t {
        for (int32_t b = 0; b < boundN; ++b)
            if (equalSpans(bound[b], name)) return b;
        return -1;
    };

    StrSpan value[kMaxShotBoundVariables];
    bool taken[kMaxShotBoundVariables] = {};
    int32_t chosen[kMaxShotPremises];
    int32_t outN = 0;

    // Unify premise p with product q; the bound variables taken here are
    // recorded in newly[] so the caller can release them.
    const auto unify = [&](int32_t p, int32_t q, int32_t* newly, int32_t& newlyN) -> bool {
        newlyN = 0;
        StrSpan premiseBefore, premiseAfter, productBefore, productAfter;
        if (!flatShape(premises[p], premiseBefore, premiseAfter)) return false;
        if (!flatShape(products[q], productBefore, productAfter)) return false;
        if (!equalSpans(premiseBefore, productBefore) || !equalSpans(premiseAfter, productAfter))
            return false;
        StrSpan premiseArgs[ExecutionParameters::MAX_ARITY];
        StrSpan productArgs[ExecutionParameters::MAX_ARITY];
        const int32_t n = getArgsSpans(premises[p], premiseArgs, ExecutionParameters::MAX_ARITY);
        if (getArgsSpans(products[q], productArgs, ExecutionParameters::MAX_ARITY) != n) return false;
        for (int32_t a = 0; a < n; ++a) {
            const int32_t b = boundIndex(premiseArgs[a]);
            if (b < 0) {
                if (!equalSpans(shotArgumentCore(premiseArgs[a]), shotArgumentCore(productArgs[a])))
                    return false;
            } else if (taken[b]) {
                if (!equalSpans(shotArgumentCore(value[b]), shotArgumentCore(productArgs[a])))
                    return false;
            } else {
                taken[b] = true;
                value[b] = productArgs[a];
                newly[newlyN++] = b;
            }
        }
        return true;
    };

    const auto emit = [&]() {
        StrReplacement pairs[kMaxShotBoundVariables];
        int32_t pairN = 0;
        for (int32_t b = 0; b < boundN; ++b) {
            if (taken[b]) {
                pairs[pairN].key = bound[b];
                pairs[pairN].value = value[b];
                ++pairN;
            } else if (shotTextHasToken(head, bound[b])) {
                return;
            }
        }
        assert(outN < outCap && "shootUniversal: shot count exceeds the caller's capacity");
        UniversalShot& shot = out[outN++];
        shot.premiseN = premiseN;
        for (int32_t p = 0; p < premiseN; ++p) shot.premiseProduct[p] = chosen[p];
        shot.head = replaceKeysScratch(arena, head, pairs, pairN);
    };

    const auto match = [&](int32_t p, const auto& self) -> void {
        if (p == premiseN) { emit(); return; }
        for (int32_t q = 0; q < productN; ++q) {
            int32_t newly[ExecutionParameters::MAX_ARITY];
            int32_t newlyN = 0;
            if (unify(p, q, newly, newlyN)) {
                chosen[p] = q;
                self(p + 1, self);
            }
            for (int32_t k = 0; k < newlyN; ++k) taken[newly[k]] = false;
        }
    };
    match(0, match);
    return outN;
}

/// @brief Capacity of the binders of one disintegration package.
constexpr int32_t kMaxShotBinders = 16;

/// @brief One disintegration package applied along the current witness path
///        of a disintegration walk (I-333).
///
/// @details
/// The disintegrator opens a frame where it visits an existence whose own
/// package has shots: the package's owner is that existence, binder 0 is the
/// witness the visit mints. Nested existences the walk reaches below that
/// witness fill the package's further binders (`matchShotBinder`). While a
/// binder is filled, `binderToken` holds its witness — the temporary token
/// during discovery, the permanent name during publication — and the shots
/// attached to it become constituents of that witness. Frames chain to the
/// enclosing frame; all of them live on the walk's stack.
struct ShotFrame {
    /// The existence entity whose package this frame applies.
    const LogicalEntity* owner = nullptr;
    /// The owner's signature arguments (keys) and the visited instance's
    /// arguments (values); the value spans live as long as the frame.
    StrReplacement args[ExecutionParameters::MAX_ARITY];
    /// Number of signature arguments.
    int32_t argN = 0;
    /// The witness filling each binder on the current path; empty while the
    /// binder is not filled.
    StrSpan binderToken[kMaxShotBinders];
    /// The enclosing frame, or null.
    ShotFrame* outer = nullptr;
};

/// @brief Open a frame for an existence's package at a visit of one of its
///        instances (I-333).
///
/// @details
/// Pairs each signature argument of @p owner with the argument at the same
/// position of @p instance and leaves every binder unfilled. The compiled
/// signature spells its arguments `u_<k>`, the disintegrator an instance's
/// arguments `u_<name>`; both are asserted. 0% heap.
///
/// @param frame    The frame to fill.
/// @param owner    The visited existence's compiled entity, with a prepared
///                 package (asserted).
/// @param instance The visited instance's signature in the walk's spelling;
///                 it must outlive the frame.
/// @param outer    The enclosing frame, or null.
/// @return Void.
/// @see ShotFrame, instantiateShotText.
inline void bindShotFrame(ShotFrame& frame, const LogicalEntity& owner, StrSpan instance, ShotFrame* outer) {
    assert(owner.shotsPrepared && "bindShotFrame: the existence's package is prepared");
    assert(owner.shotBinders.size() <= static_cast<std::size_t>(kMaxShotBinders)
        && owner.shotPlaceholders.size() == owner.shotBinders.size()
        && "bindShotFrame: one placeholder per binder, within kMaxShotBinders");
    StrSpan signatureArgs[ExecutionParameters::MAX_ARITY];
    const int32_t n = getArgsSpans(StrSpan(owner.signature), signatureArgs, ExecutionParameters::MAX_ARITY);
    StrSpan instanceArgs[ExecutionParameters::MAX_ARITY];
    const int32_t m = getArgsSpans(instance, instanceArgs, ExecutionParameters::MAX_ARITY);
    assert(n == m && "bindShotFrame: the instance has the owner's arity");
    for (int32_t i = 0; i < n; ++i) {
        assert(startsWithSpan(signatureArgs[i], "u_", 2) && startsWithSpan(instanceArgs[i], "u_", 2)
            && "bindShotFrame: signature and instance arguments are spelled u_");
        frame.args[i] = StrReplacement{ signatureArgs[i], instanceArgs[i] };
    }
    frame.owner = &owner;
    frame.argN = n;
    for (int32_t b = 0; b < kMaxShotBinders; ++b) frame.binderToken[b] = StrSpan();
    frame.outer = outer;
}

/// @brief Instantiate a text of a package on the current witness path
///        (I-333).
///
/// @details
/// Replaces, in one simultaneous token-boundary pass (`replaceKeysScratch`),
/// the owner's signature arguments by the instance's arguments and the
/// placeholder of every filled binder by its witness: the binder @p current
/// by the bare token, as a witness spells itself in its own constituents,
/// every other binder by `u_<token>`, as the walk spells an enclosing
/// witness. Unfilled binders keep their placeholders. 0% heap.
///
/// @param text    A binder template, shot head, premise or universal of
///                the frame's package.
/// @param frame   The frame.
/// @param current The binder spelled bare, or -1 for none.
/// @param arena   String-tier scratch arena receiving the result.
/// @return The instantiated text on @p arena.
/// @see bindShotFrame, matchShotBinder.
inline ScratchString instantiateShotText(StrSpan text, const ShotFrame& frame, int32_t current,
    ScratchArena& arena) {
    assert(frame.owner != nullptr && "instantiateShotText: the frame is open");
    StrReplacement pairs[ExecutionParameters::MAX_ARITY + kMaxShotBinders];
    int32_t n = 0;
    for (int32_t i = 0; i < frame.argN; ++i) pairs[n++] = frame.args[i];
    const int32_t binderN = static_cast<int32_t>(frame.owner->shotBinders.size());
    for (int32_t b = 0; b < binderN; ++b) {
        const StrSpan token = frame.binderToken[b];
        if (token.len == 0) continue;
        StrSpan value = token;
        if (b != current) {
            char* bytes = arena.allocBytes(token.len + 2);
            bytes[0] = 'u';
            bytes[1] = '_';
            std::memcpy(bytes + 2, token.ptr, static_cast<size_t>(token.len));
            value = StrSpan(bytes, token.len + 2);
        }
        pairs[n++] = StrReplacement{ StrSpan(frame.owner->shotPlaceholders[static_cast<std::size_t>(b)]), value };
    }
    return replaceKeysScratch(arena, text, pairs, n);
}

/// @brief The binder of a frame's package that a visited existence fills
///        (I-333).
///
/// @details
/// A nested existence of the owner's definition is a binder of its package
/// (binder 0 is the owner itself). The visit fills the first unfilled
/// binder whose template, instantiated on the current path (every filled
/// binder spelled `u_<token>`), equals @p instance: the template names the
/// enclosing witnesses, so it matches only below them. 0% heap; the
/// instantiations are released before returning.
///
/// @param frame    The frame of an enclosing existence.
/// @param instance The visited existence's signature in the walk's spelling.
/// @param arena    String-tier scratch arena for the instantiations.
/// @return The binder index (at least 1), or -1 when the visit fills none.
/// @see instantiateShotText, ShotFrame.
inline int32_t matchShotBinder(const ShotFrame& frame, StrSpan instance, ScratchArena& arena) {
    assert(frame.owner != nullptr && "matchShotBinder: the frame is open");
    const int32_t binderN = static_cast<int32_t>(frame.owner->shotBinders.size());
    for (int32_t b = 1; b < binderN; ++b) {
        if (frame.binderToken[b].len != 0) continue;
        ScratchScope scope(arena);
        const ScratchString text = instantiateShotText(
            StrSpan(frame.owner->shotBinders[static_cast<std::size_t>(b)]), frame, -1, arena);
        if (equalSpans(StrSpan(text), instance)) return b;
    }
    return -1;
}

} // namespace gl
