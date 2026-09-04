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
/// @brief Tests for the rule install's per-copy premise encode memo
///        (`KeyEncodeMemo`) and the one-pass prefix key build the subkey
///        loops run on it.
///
/// @details
/// 1. `KeyEncodeMemo::at` encodes a premise exactly once: the first read
///    equals a direct `encodeExpression` of the same text byte for byte, a
///    later read returns the same stored row and mints nothing, and the fill
///    order is the read order (the lazy fill is the first-touch order).
/// 2. The subkey loops build one permutation's normalized key incrementally
///    (`appendExprToIntNormalizedKey`) and take each prefix's subkey as the
///    slots written so far; over a deterministic random corpus every prefix
///    equals the standalone per-prefix build (`makeIntNormalizedKeyFromEncoded`
///    with `ignoreU = false`) the loops used to run per prefix.
///
/// The corpus comes from a fixed-seed linear congruential sequence, so the
/// test is byte-reproducible across hosts and runs.

#include "test_harness.hpp"

#include "../memory.hpp"
#include "../prover.hpp"

#include <cstdint>
#include <cstring>

namespace {

    /// Standalone-NameMap rig: a NameMap is a facade over two cold string
    /// tables; this binds one to private tables on the harness pool
    /// (declaration order: manager first, so the tables destruct before it).
    struct NameMapRig {
        gl::LbArena lb{ &gl::staticMemory() };
        gl::DirtyState dirty = gl::DirtyState::Clean;
        gl::ColdStringTable names{ &lb, &dirty };
        gl::ColdStringTable subs{ &lb, &dirty };
        gl::PagedVector<gl::ValidityNode> nodes{ &lb, &dirty };
        gl::NameMap nm;
        NameMapRig() { nm.bind(&names, &subs, &nodes); }
    };

    constexpr gl::NameId kCap = gl::ExecutionParameters::MAX_KEY_SLOTS;

    // Deterministic 32-bit source (high bits — an LCG's low bits cycle fast).
    uint32_t nextRandom(uint32_t& s) {
        s = s * 1664525u + 1013904223u;
        return s >> 16;
    }

    // Only nameId, negation, arity and argId reach the normalized key; the
    // rest is zeroed so an accidental read of another field shows up.
    gl::IntEncodedExpr keyExpr(gl::NameId nameId, int32_t negation,
                               const gl::NameId* args, int32_t arity) {
        gl::IntEncodedExpr e{};
        e.nameId = nameId;
        e.negation = negation;
        e.arity = arity;
        for (int32_t j = 0; j < arity; ++j) e.argId[j] = args[j];
        return e;
    }

}  // namespace

// The first read encodes (and mints the premise's names); a direct encode of
// the same text afterwards is byte-identical and mints nothing more; a second
// read returns the same stored row; premises fill in the order they are read.
TEST(key_encode_memo, at_encodes_once_and_matches_encodeExpression) {
    NameMapRig rig;
    gl::NameMap& nm = rig.nm;
    const char* texts[3] = {
        "(in3[a,b,c])", "(in2[u_a,it_0_lev_1_3])", "!(in3[c,a,a])" };
    gl::StrSpan key[3];
    for (int i = 0; i < 3; ++i)
        key[i] = gl::StrSpan(texts[i], static_cast<int32_t>(std::strlen(texts[i])));

    gl::KeyEncodeMemo memo;
    memo.bind(key, 3);
    ASSERT_FALSE(memo.filled[0]);
    ASSERT_FALSE(memo.filled[1]);
    ASSERT_FALSE(memo.filled[2]);

    const gl::IntEncodedExpr& e1 = memo.at(1, nm);
    ASSERT_TRUE(memo.filled[1]);
    ASSERT_FALSE(memo.filled[0]);
    ASSERT_FALSE(memo.filled[2]);
    const int32_t namesAfterFirst = nm.nameCount();
    ASSERT_TRUE(namesAfterFirst > 0);

    // A direct re-encode sees the same ids: identical bytes, no new mint.
    const gl::IntEncodedExpr direct =
        gl::encodeExpression(key[1], gl::StrSpan("main", 4), nm);
    ASSERT_EQ(nm.nameCount(), namesAfterFirst);
    ASSERT_EQ(std::memcmp(&e1, &direct, sizeof(gl::IntEncodedExpr)), 0);
    ASSERT_EQ(e1.arity, 2);
    ASSERT_EQ(e1.argUnchangeable[0], 1);
    ASSERT_EQ(e1.argIteration[1], 0);
    ASSERT_EQ(e1.argLevPlus1[1], 2);

    // A second memo read: the same row, nothing minted.
    const gl::IntEncodedExpr& e1Again = memo.at(1, nm);
    ASSERT_TRUE(&e1Again == &e1);
    ASSERT_EQ(nm.nameCount(), namesAfterFirst);

    // Premises 2 then 0 fill on demand in that order: premise 2's original
    // text is minted before premise 0's — the lazy fill IS the first-touch
    // order the install relies on.
    const gl::IntEncodedExpr& e2 = memo.at(2, nm);
    const gl::IntEncodedExpr& e0 = memo.at(0, nm);
    ASSERT_TRUE(memo.filled[0]);
    ASSERT_TRUE(memo.filled[2]);
    ASSERT_EQ(e2.negation, 1);
    ASSERT_EQ(e0.negation, 0);
    ASSERT_TRUE(e2.originalId < e0.originalId);
    ASSERT_EQ(e2.nameId, e0.nameId);   // both "in3"
    ASSERT_TRUE(nm.nameCount() > namesAfterFirst);

    // Re-binding forgets every row.
    memo.bind(key, 2);
    ASSERT_EQ(memo.keyN, 2);
    ASSERT_FALSE(memo.filled[0]);
    ASSERT_FALSE(memo.filled[1]);
}

// One incremental build per permutation: after folding premise L the slots
// written so far equal the standalone build of the L-premise prefix, for every
// prefix of every candidate of a random corpus (arities small enough that
// eight premises fit MAX_KEY_SLOTS).
TEST(key_encode_memo, incremental_prefix_keys_match_per_prefix_build) {
    gl::ExpressionAnalyzer ea("Peano");
    uint32_t seed = 20260903u;
    gl::IntEncodedExpr pool[64];
    for (int p = 0; p < 64; ++p) {
        const int32_t arity = static_cast<int32_t>(nextRandom(seed) % 7u);
        gl::NameId args[8];
        for (int32_t j = 0; j < arity; ++j)
            args[j] = static_cast<gl::NameId>(100u + nextRandom(seed) % 5u);
        const gl::NameId nameId =
            static_cast<gl::NameId>(10u + nextRandom(seed) % 6u);
        const int32_t negation = static_cast<int32_t>(nextRandom(seed) % 2u);
        pool[p] = keyExpr(nameId, negation, args, arity);
    }

    int checked = 0;
    for (int trial = 0; trial < 600; ++trial) {
        const gl::NameId count = static_cast<gl::NameId>(1u + nextRandom(seed) % 8u);
        gl::IntEncodedExpr run[8];
        for (gl::NameId k = 0; k < count; ++k)
            run[k] = pool[nextRandom(seed) % 64u];

        // The subkey loops' build: fold one premise at a time, read the
        // prefix key as the slots written so far.
        gl::NameId incremental[kCap];
        gl::ExpressionAnalyzer::NormKeyBuildState st;
        st.nVars = 0;
        st.nextNormId = 1;
        st.pos = 0;
        for (gl::NameId L = 1; L <= count; ++L) {
            gl::ExpressionAnalyzer::appendExprToIntNormalizedKey(
                run[L - 1], st, incremental, kCap);

            gl::NameId standalone[kCap];
            const gl::NameId standaloneLen = ea.makeIntNormalizedKeyFromEncoded(
                run, L, false, standalone, kCap);
            ASSERT_EQ(st.pos, standaloneLen);
            ASSERT_EQ(std::memcmp(incremental, standalone,
                                  static_cast<std::size_t>(standaloneLen)
                                      * sizeof(gl::NameId)), 0);
            ++checked;
        }
    }
    ASSERT_TRUE(checked > 600);
}
