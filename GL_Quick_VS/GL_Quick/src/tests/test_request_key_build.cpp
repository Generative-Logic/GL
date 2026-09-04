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
/// @brief Unit tests for the resumable normalized-key build — the pair
///        `buildIntNormalizedKeyPrefix` + `appendExprToIntNormalizedKey` that
///        the request generator's grow search uses instead of rebuilding every
///        candidate's key from scratch.
///
/// @details
/// The contract under test is byte identity, not equivalence. The key feeds
/// both owner-set probes and the emitted request, and `addToHashMemory`
/// installs a rule's whole permutation family against that byte language, so a
/// single differing slot changes which rules fire.
///
/// The oracle is `makeIntNormalizedKeyFromEncoded` over the whole candidate.
/// Three tests:
///
/// 1. Every candidate of a deterministic corpus: a prefix of length n-1 plus
///    one appended premise must reproduce the full build byte for byte.
/// 2. The production pattern — one prefix built once, then many different last
///    premises folded in with a counter rewind between them. This is where a
///    missed rollback shows: the second position would inherit the slots the
///    first one minted.
/// 3. Edge shapes — zero arity, maximum arity, every argument the same, every
///    argument distinct, both negation values, and an appended premise that
///    introduces no new argument at all.
///
/// The corpus comes from a fixed-seed linear congruential sequence, so the
/// tests are byte-reproducible across hosts and runs.

#include "test_harness.hpp"

#include "../memory.hpp"
#include "../prover.hpp"

#include <cstdint>
#include <cstring>

namespace {

    constexpr gl::NameId kCap = gl::ExecutionParameters::MAX_KEY_SLOTS;

    // Deterministic 32-bit source. No <random>, no clock seed: the corpus must
    // be identical on every host. Returns the HIGH bits — a linear congruential
    // generator's low bits have a period of a few values, so drawing small
    // ranges from them collapses the corpus.
    uint32_t nextRandom(uint32_t& s) {
        s = s * 1664525u + 1013904223u;
        return s >> 16;
    }

    // Only nameId, negation, arity and argId reach the normalized key; the rest
    // of the record is zeroed so an accidental read of another field shows up
    // as a difference rather than as noise.
    gl::IntEncodedExpr keyExpr(gl::NameId nameId, int32_t negation,
                               const gl::NameId* args, int32_t arity) {
        gl::IntEncodedExpr e{};
        e.nameId = nameId;
        e.negation = negation;
        e.arity = arity;
        for (int32_t j = 0; j < arity; ++j) e.argId[j] = args[j];
        return e;
    }

    // Full build vs prefix-then-append over the same premise array.
    bool splitMatchesFull(gl::ExpressionAnalyzer& ea,
                          const gl::IntEncodedExpr* const* exprs,
                          gl::NameId count) {
        gl::NameId full[kCap];
        const gl::NameId fullLen =
            ea.makeIntNormalizedKeyFromEncoded(exprs, count, full, kCap);

        gl::NameId split[kCap];
        gl::ExpressionAnalyzer::NormKeyBuildState st;
        gl::ExpressionAnalyzer::buildIntNormalizedKeyPrefix(
            exprs, static_cast<gl::NameId>(count - 1), split, kCap, st);
        gl::ExpressionAnalyzer::appendExprToIntNormalizedKey(
            *exprs[count - 1], st, split, kCap);

        return st.pos == fullLen
            && std::memcmp(full, split,
                           static_cast<std::size_t>(fullLen)
                               * sizeof(gl::NameId)) == 0;
    }

}  // namespace

// Every candidate of a random corpus: splitting the build at the last premise
// reproduces the whole-candidate key exactly. Arities stay small enough that
// eight premises fit MAX_KEY_SLOTS.
TEST(request_key_build, split_at_last_premise_matches_full_build) {
    gl::ExpressionAnalyzer ea("Peano");
    uint32_t seed = 20260821u;
    gl::IntEncodedExpr pool[64];
    for (int p = 0; p < 64; ++p) {
        const int32_t arity = static_cast<int32_t>(nextRandom(seed) % 7u);
        gl::NameId args[8];
        for (int32_t j = 0; j < arity; ++j)
            args[j] = static_cast<gl::NameId>(100u + nextRandom(seed) % 5u);
        // Sequenced draws: argument evaluation order is unspecified in C++.
        const gl::NameId nameId =
            static_cast<gl::NameId>(10u + nextRandom(seed) % 6u);
        const int32_t negation = static_cast<int32_t>(nextRandom(seed) % 2u);
        pool[p] = keyExpr(nameId, negation, args, arity);
    }

    int checked = 0;
    for (int trial = 0; trial < 600; ++trial) {
        const gl::NameId count = static_cast<gl::NameId>(
            1u + nextRandom(seed)
                     % static_cast<uint32_t>(gl::ExecutionParameters::MAX_EXPRESSIONS));
        const gl::IntEncodedExpr* ptrs[gl::ExecutionParameters::MAX_EXPRESSIONS];
        for (gl::NameId k = 0; k < count; ++k)
            ptrs[k] = &pool[nextRandom(seed) % 64u];
        ASSERT_TRUE(splitMatchesFull(ea, ptrs, count));
        ++checked;
    }
    ASSERT_EQ(checked, 600);
}

// The production pattern: one prefix, many last premises, a counter rewind in
// between. A rollback that forgets nVars, nextNormId or pos shows here and
// nowhere else — the first position would poison every later one.
TEST(request_key_build, rewind_reuses_one_prefix_across_positions) {
    uint32_t seed = 7771u;
    gl::IntEncodedExpr pool[32];
    for (int p = 0; p < 32; ++p) {
        const int32_t arity = static_cast<int32_t>(1u + nextRandom(seed) % 5u);
        gl::NameId args[8];
        for (int32_t j = 0; j < arity; ++j)
            args[j] = static_cast<gl::NameId>(200u + nextRandom(seed) % 6u);
        const gl::NameId nameId =
            static_cast<gl::NameId>(20u + nextRandom(seed) % 4u);
        const int32_t negation = static_cast<int32_t>(nextRandom(seed) % 2u);
        pool[p] = keyExpr(nameId, negation, args, arity);
    }

    gl::ExpressionAnalyzer ea("Peano");

    for (gl::NameId prefixCount = 0; prefixCount <= 4; ++prefixCount) {
        const gl::IntEncodedExpr* ptrs[gl::ExecutionParameters::MAX_EXPRESSIONS];
        for (gl::NameId k = 0; k < prefixCount; ++k) ptrs[k] = &pool[k * 3 + 1];

        gl::NameId shared[kCap];
        gl::ExpressionAnalyzer::NormKeyBuildState st;
        gl::ExpressionAnalyzer::buildIntNormalizedKeyPrefix(
            ptrs, prefixCount, shared, kCap, st);
        const gl::NameId prefixVars = st.nVars;
        const gl::NameId prefixNext = st.nextNormId;
        const gl::NameId prefixLen = st.pos;

        for (int last = 0; last < 32; ++last) {
            st.nVars = prefixVars;
            st.nextNormId = prefixNext;
            st.pos = prefixLen;
            gl::ExpressionAnalyzer::appendExprToIntNormalizedKey(
                pool[last], st, shared, kCap);

            ptrs[prefixCount] = &pool[last];
            gl::NameId full[kCap];
            const gl::NameId fullLen = ea.makeIntNormalizedKeyFromEncoded(
                ptrs, static_cast<gl::NameId>(prefixCount + 1), full, kCap);

            ASSERT_EQ(st.pos, fullLen);
            ASSERT_EQ(std::memcmp(full, shared,
                                  static_cast<std::size_t>(fullLen)
                                      * sizeof(gl::NameId)), 0);
        }
    }
}

// Named edge shapes, so a regression names itself instead of surfacing as one
// failed trial out of six hundred.
TEST(request_key_build, edge_shapes_match_full_build) {
    gl::ExpressionAnalyzer ea("Peano");
    const gl::NameId same[gl::ExecutionParameters::MAX_ARITY] = {
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5 };
    const gl::NameId distinct[gl::ExecutionParameters::MAX_ARITY] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };

    // Zero arity on both sides — the key is name and negation only.
    const gl::IntEncodedExpr nullary0 = keyExpr(31, 0, same, 0);
    const gl::IntEncodedExpr nullary1 = keyExpr(31, 1, same, 0);
    const gl::IntEncodedExpr* nullaryPair[2] = { &nullary0, &nullary1 };
    ASSERT_TRUE(splitMatchesFull(ea, nullaryPair, 2));

    // Maximum arity, all arguments identical: one slot minted, reused fifteen
    // times, and the appended premise mints nothing at all.
    const gl::IntEncodedExpr wideSame =
        keyExpr(32, 0, same, gl::ExecutionParameters::MAX_ARITY);
    const gl::IntEncodedExpr* sharedVarPair[2] = { &wideSame, &wideSame };
    ASSERT_TRUE(splitMatchesFull(ea, sharedVarPair, 2));

    // Maximum arity, all arguments distinct: the prefix mints every slot and
    // the appended premise reuses all of them in the same order.
    const gl::IntEncodedExpr wideDistinct =
        keyExpr(33, 1, distinct, gl::ExecutionParameters::MAX_ARITY);
    const gl::IntEncodedExpr* distinctPair[2] = { &wideDistinct, &wideDistinct };
    ASSERT_TRUE(splitMatchesFull(ea, distinctPair, 2));

    // Appended premise introduces only new arguments — the numbering must
    // continue from the prefix, not restart.
    const gl::IntEncodedExpr freshArgs = keyExpr(34, 0, distinct, 4);
    const gl::NameId beyond[4] = { 90, 91, 92, 93 };
    const gl::IntEncodedExpr newArgs = keyExpr(35, 0, beyond, 4);
    const gl::IntEncodedExpr* growPair[2] = { &freshArgs, &newArgs };
    ASSERT_TRUE(splitMatchesFull(ea, growPair, 2));

    // A single premise: the prefix is empty and the append does all the work.
    const gl::IntEncodedExpr* lone[1] = { &freshArgs };
    ASSERT_TRUE(splitMatchesFull(ea, lone, 1));
}
