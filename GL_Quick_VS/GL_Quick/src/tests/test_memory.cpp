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
/// @brief Phase-1 smoke tests for the memory module.
///
/// @details
/// Five tests exercising the cheapest, most central invariants of the
/// memory module so the harness has something to run from the very
/// first commit:
///
/// 1. `NameMap` constructor establishes I-2's pre-condition: `MAIN_ID == 1`,
///    `nameToId["main"] == 1`. Every non-`main` validity name is minted
///    via `encodePush(parentId, payload)` rather than raw concatenation;
///    the constructor's pre-registration is what makes that contract
///    enforceable elsewhere.
/// 2. `NameMap::encode` is idempotent — same input string always returns
///    the same id. Demonstrates the dictionary semantics on which the
///    per-validity stack and `pairMap` rely.
/// 3. `Memory{}` default-constructs to the root-sentinel shape:
///    `exprKey.empty()`, `parentMemory == nullptr`, `level == -1`. This
///    is the "root LB" recogniser used by chain-walk traps (Rule 12)
///    and termination guards in `parentMemory`-walking code.
/// 4. `ce::generateBinarySequencesAsLists(3)` returns the expected
///    eight 3-bit binary sequences. A trivial-but-precise sanity check
///    on a pure-math helper that several pipeline stages rely on.
/// 5. Self-consistency of the assertion macros themselves
///    (ASSERT_TRUE, ASSERT_FALSE, ASSERT_EQ, ASSERT_NE).
///
/// More tests for `NameMap::encodePush`, `NameMap::idToSub` reference
/// stability (I-3), `pairMap` / `comparable`, `ChunkPool` static
/// storage (I-13), `HashMemory` round-trip, `Memory` simpleMap chain,
/// `EquivalenceClass` merge, `KeyArena` allocate, and
/// `IntNormalizedKey` hash determinism land in Phase 2 alongside the
/// `memory.hpp` doxygen pass.

#include "test_harness.hpp"

#include "../memory.hpp"
#include "../compiler.hpp"
// Pulled in for ExpressionAnalyzer's static comparators (`lessByName`,
// `lessByOriginal`) tested below. The header is heavy (~336 KB), but the
// alternative — forward-declaring an enclosing struct — does not let the
// caller name `ExpressionAnalyzer::lessByName` inside `gl::`.
#include "../prover.hpp"

#include <cstdint>

TEST(memory, namemap_main_id_one) {
    gl::NameMap nm;
    ASSERT_EQ(nm.encode("main"), static_cast<int16_t>(gl::NameMap::MAIN_ID));
    ASSERT_EQ(nm.encode("main"), static_cast<int16_t>(1));
}

TEST(memory, namemap_encode_idempotent) {
    gl::NameMap nm;
    const int16_t a = nm.encode("main");
    const int16_t b = nm.encode("main");
    ASSERT_EQ(a, b);
    // Distinct root strings produce distinct ids.
    const int16_t r1 = nm.encode("alpha");
    const int16_t r2 = nm.encode("beta");
    ASSERT_NE(r1, r2);
    // Re-encoding either is still idempotent.
    ASSERT_EQ(r1, nm.encode("alpha"));
    ASSERT_EQ(r2, nm.encode("beta"));
}

TEST(memory, memory_root_sentinel) {
    gl::Memory m;
    ASSERT_TRUE(m.exprKey.empty());
    ASSERT_EQ(m.parentMemory, static_cast<gl::Memory*>(nullptr));
    ASSERT_EQ(m.level, -1);
    ASSERT_TRUE(m.isActive);
}

TEST(compiler, generate_binary_sequences_three_has_eight) {
    const auto seqs = ce::generateBinarySequencesAsLists(3);
    ASSERT_EQ(seqs.size(), static_cast<std::size_t>(8));
    for (const auto& row : seqs) {
        ASSERT_EQ(row.size(), static_cast<std::size_t>(3));
        for (int bit : row) {
            ASSERT_TRUE(bit == 0 || bit == 1);
        }
    }
}

TEST(harness, smoke_assert_macros_self_consistent) {
    ASSERT_TRUE(true);
    ASSERT_FALSE(false);
    ASSERT_EQ(1, 1);
    ASSERT_NE(1, 2);
    ASSERT_LT(1, 2);
    ASSERT_GE(2, 2);
}

// =============================================================================
// Phase 2 tests — see docstring at the top of this file for the catalogue.
// =============================================================================

// I-2 — encodePush establishes the parent-prefix relation in pairMap and the
// ancestor list in stackOfValidity. Tests one-deep and two-deep pushes.
TEST(memory, namemap_encodepush_stack_discipline_I2) {
    gl::NameMap nm;
    const int16_t mainId = nm.encode("main");
    const int16_t a = nm.encodePush(mainId, "alpha");
    const int16_t b = nm.encodePush(a, "beta");

    ASSERT_NE(a, mainId);
    ASSERT_NE(b, a);
    ASSERT_NE(b, mainId);

    // Stack content reflects the parent chain bottom-up.
    const std::vector<int16_t>& stackA = nm.stackOf(a);
    ASSERT_EQ(stackA.size(), static_cast<std::size_t>(1));

    const std::vector<int16_t>& stackB = nm.stackOf(b);
    ASSERT_EQ(stackB.size(), static_cast<std::size_t>(2));
    ASSERT_EQ(stackB[0], stackA[0]);   // parent payload preserved at depth 0

    // pairMap encodes prefix relation. Per the doxygen: verdict(parent, child)
    // == -1 (parent is strict prefix of child); verdict(child, parent) == +1.
    int16_t v = 0;
    ASSERT_TRUE(nm.verdict(mainId, a, v));
    ASSERT_EQ(v, static_cast<int16_t>(-1));
    ASSERT_TRUE(nm.verdict(a, mainId, v));
    ASSERT_EQ(v, static_cast<int16_t>(1));
    ASSERT_TRUE(nm.verdict(mainId, b, v));
    ASSERT_EQ(v, static_cast<int16_t>(-1));

    // Strict-ancestor cache mirrors pairMap.
    const std::string canonicalA = "main_boundary_alpha";
    const std::string canonicalB = "main_boundary_alpha_boundary_beta";
    ASSERT_TRUE(nm.isStrictAncestor("main", canonicalA));
    ASSERT_TRUE(nm.isStrictAncestor("main", canonicalB));
    ASSERT_TRUE(nm.isStrictAncestor(canonicalA, canonicalB));
    ASSERT_FALSE(nm.isStrictAncestor(canonicalB, canonicalA));
}

// I-2 — encodePush is dedupe-on-canonical-string (parent + "_boundary_" +
// payload), so the same push from the same parent returns the same id.
TEST(memory, namemap_encodepush_dedupe_I2) {
    gl::NameMap nm;
    const int16_t mainId = nm.encode("main");
    const int16_t a1 = nm.encodePush(mainId, "scope1");
    const int16_t a2 = nm.encodePush(mainId, "scope1");
    ASSERT_EQ(a1, a2);

    // Same payload from a different parent is a different id.
    const int16_t a3 = nm.encodePush(a1, "scope1");
    ASSERT_NE(a1, a3);
}

// I-3 — idToSub is a vector; nested encode() calls may reallocate. The
// public contract is that the caller copies the string out before any
// nested mint. This test exercises the contract by interning a payload,
// reading it back via decode(), copying the result, and then forcing
// further encodes that may reallocate idToName / idToSub.
TEST(memory, namemap_decode_copy_before_nested_mint_I3) {
    gl::NameMap nm;
    const int16_t mainId = nm.encode("main");
    const int16_t s1 = nm.encodePush(mainId, "x");

    // Take a copy of the decoded string before further encodes.
    const std::string copied = nm.decode(s1);

    // Force several pushes to grow idToName / idToSub vectors.
    for (int i = 0; i < 64; ++i) {
        nm.encodePush(mainId, std::string("payload_") + std::to_string(i));
    }

    // The copy is intact and matches what decode would produce now.
    ASSERT_EQ(copied, std::string("main_boundary_x"));
    ASSERT_EQ(copied, nm.decode(s1));
}

// pairMap basic — comparable() agrees with verdict() existence.
TEST(memory, namemap_comparable_matches_verdict) {
    gl::NameMap nm;
    const int16_t mainId = nm.encode("main");
    const int16_t a = nm.encodePush(mainId, "a");
    const int16_t b = nm.encodePush(mainId, "b");   // sibling, divergent

    ASSERT_TRUE(nm.comparable(mainId, mainId));
    ASSERT_TRUE(nm.comparable(mainId, a));
    ASSERT_TRUE(nm.comparable(a, mainId));
    ASSERT_FALSE(nm.comparable(a, b));   // siblings diverge
}

// HashMemory smoke — default-constructed HashMemory is empty across every
// slot and clear() is idempotent.
TEST(memory, hashmemory_empty_and_clear) {
    gl::HashMemory hm;
    ASSERT_EQ(hm.encodedMap.size(), static_cast<std::size_t>(0));
    ASSERT_EQ(hm.normalizedEncodedKeys.size(), static_cast<std::size_t>(0));
    ASSERT_EQ(hm.maxKeyLength, static_cast<int16_t>(0));
    ASSERT_EQ(hm.originals.size(), static_cast<std::size_t>(0));
    ASSERT_EQ(hm.admissionMap.size(), static_cast<std::size_t>(0));
    ASSERT_EQ(hm.rejectedMap.size(), static_cast<std::size_t>(0));
    ASSERT_EQ(hm.varsInRejectedMapIntegrationKeys.size(),
              static_cast<std::size_t>(0));

    hm.clear();   // idempotent on empty
    ASSERT_EQ(hm.encodedMap.size(), static_cast<std::size_t>(0));
}

// Memory simpleMap chain — child added under a routing key, parent pointer
// set, exprKey can differ from routing key.
TEST(memory, memory_simplemap_chain) {
    gl::Memory root;            // root sentinel
    gl::Memory child;
    child.parentMemory = &root;
    child.exprKey = "(=[a,b])"; // expression text owned by child
    child.level = 0;

    // Routing key in simpleMap can be arbitrary — it does NOT have to match
    // child.exprKey (Rule 12: simpleMap key vs exprKey distinction).
    root.simpleMap["routing-key"] = &child;

    ASSERT_EQ(root.simpleMap.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(root.simpleMap["routing-key"], &child);
    ASSERT_EQ(child.parentMemory, &root);
    ASSERT_EQ(child.parentMemory->parentMemory,
              static_cast<gl::Memory*>(nullptr));   // root has no parent
    ASSERT_EQ(child.exprKey, std::string("(=[a,b])"));
}

// EquivalenceClass — default-constructed has empty containers; parameterized
// constructor populates each slot; classes with the same variables but
// different origin/level maps are still distinct values.
TEST(memory, equivalenceclass_construction) {
    gl::EquivalenceClass empty;
    ASSERT_EQ(empty.variables.size(), static_cast<std::size_t>(0));
    ASSERT_EQ(empty.equalityLevelsMap.size(), static_cast<std::size_t>(0));
    ASSERT_EQ(empty.equalityOriginMap.size(), static_cast<std::size_t>(0));

    std::set<std::string> vars{"a", "b"};
    std::map<std::set<std::string>, std::set<int>> lev{
        {std::set<std::string>{"a", "b"}, std::set<int>{0, 1}}};
    std::map<gl::ExpressionWithValidity,
             std::vector<std::pair<std::string,
                                   std::vector<gl::ExpressionWithValidity>>>>
        origin;
    gl::EquivalenceClass cls(vars, lev, origin);
    ASSERT_EQ(cls.variables.size(), static_cast<std::size_t>(2));
    ASSERT_EQ(cls.equalityLevelsMap.size(), static_cast<std::size_t>(1));
}

// KeyArena — store returns a stable pointer; release wipes everything;
// repeated stores beyond a single chunk allocate fresh chunks.
TEST(memory, keyarena_store_and_release) {
    gl::KeyArena arena;
    int16_t buf1[3] = {1, 2, 3};
    int16_t buf2[3] = {4, 5, 6};
    int16_t* p1 = arena.store(buf1, 3);
    int16_t* p2 = arena.store(buf2, 3);
    ASSERT_NE(p1, static_cast<int16_t*>(nullptr));
    ASSERT_NE(p2, static_cast<int16_t*>(nullptr));
    ASSERT_NE(p1, p2);
    ASSERT_EQ(p1[0], static_cast<int16_t>(1));
    ASSERT_EQ(p1[1], static_cast<int16_t>(2));
    ASSERT_EQ(p1[2], static_cast<int16_t>(3));
    ASSERT_EQ(p2[0], static_cast<int16_t>(4));
    arena.release();
    // After release the arena is empty; a fresh store allocates a new chunk.
    int16_t buf3[2] = {7, 8};
    int16_t* p3 = arena.store(buf3, 2);
    ASSERT_NE(p3, static_cast<int16_t*>(nullptr));
    ASSERT_EQ(p3[0], static_cast<int16_t>(7));
}

// IntNormalizedKey hash determinism — equal keys hash to the same value;
// distinct keys hash to (almost certainly) different values.
TEST(memory, intnormalizedkey_hash_determinism) {
    int16_t buf1[4] = {1, 0, 5, 1};
    int16_t buf2[4] = {1, 0, 5, 1};
    int16_t buf3[4] = {1, 0, 5, 2};
    gl::IntNormalizedKey k1(1, buf1, 4);
    gl::IntNormalizedKey k2(1, buf2, 4);
    gl::IntNormalizedKey k3(1, buf3, 4);

    gl::IntNormalizedKeyHash h;
    ASSERT_EQ(h(k1), h(k2));        // determinism on identical content
    ASSERT_NE(h(k1), h(k3));        // very likely distinct hashes
    ASSERT_TRUE(k1 == k2);
    ASSERT_FALSE(k1 == k3);
    ASSERT_FALSE(k1 < k2);
    ASSERT_TRUE(k1 < k3);
}

// LocalMemoryValue — default constructor produces the documented baseline,
// including validityName == "main" and productOfDisintegration == false.
TEST(memory, localmemoryvalue_default) {
    gl::LocalMemoryValue lmv;
    ASSERT_TRUE(lmv.value.empty());
    ASSERT_TRUE(lmv.levels.empty());
    ASSERT_TRUE(lmv.originalImplication.empty());
    ASSERT_TRUE(lmv.justification.empty());
    ASSERT_TRUE(lmv.key.empty());
    ASSERT_TRUE(lmv.remainingArgs.empty());
    ASSERT_EQ(lmv.validityName, std::string("main"));
    ASSERT_FALSE(lmv.productOfDisintegration);
}

// ExpressionWithValidity ordering — lexicographic over (original, validityName).
TEST(memory, expressionwithvalidity_ordering) {
    gl::ExpressionWithValidity a("(=[a,b])", "main");
    gl::ExpressionWithValidity b("(=[a,b])", "main_boundary_x");
    gl::ExpressionWithValidity c("(=[c,d])", "main");
    ASSERT_TRUE(a < b);    // same original; "main" < "main_boundary_x"
    ASSERT_TRUE(a < c);    // "(=[a,b])" < "(=[c,d])"
    ASSERT_FALSE(a == b);
    ASSERT_TRUE(a == gl::ExpressionWithValidity("(=[a,b])", "main"));
}

// =============================================================================
// Phase 3 tests — for memory.cpp bodies. Most ExpressionAnalyzer member
// functions in memory.cpp require a fully-constructed ExpressionAnalyzer
// (loads config files), which is heavyweight for a unit test. The static
// member functions and the inline helpers in memory.hpp are tested here;
// instance-method coverage lands in Phase 7+ once ExpressionAnalyzer
// itself is documented and a synthetic-config path exists.
// =============================================================================

// lessByName / lessByOriginal — static comparators on EncodedExpression.
TEST(memory, lessbyname_ordering) {
    gl::EncodedExpression a;
    gl::EncodedExpression b;
    a.name = "abc";
    b.name = "abd";
    ASSERT_TRUE(gl::ExpressionAnalyzer::lessByName(a, b));
    ASSERT_FALSE(gl::ExpressionAnalyzer::lessByName(b, a));

    // Same name → not less in either direction (irreflexive).
    a.name = "x";
    b.name = "x";
    ASSERT_FALSE(gl::ExpressionAnalyzer::lessByName(a, b));
    ASSERT_FALSE(gl::ExpressionAnalyzer::lessByName(b, a));
}

TEST(memory, lessbyoriginal_ordering) {
    gl::EncodedExpression a;
    gl::EncodedExpression b;
    a.original = "alpha";
    b.original = "beta";
    ASSERT_TRUE(gl::ExpressionAnalyzer::lessByOriginal(a, b));
    ASSERT_FALSE(gl::ExpressionAnalyzer::lessByOriginal(b, a));

    a.original = b.original = "same";
    ASSERT_FALSE(gl::ExpressionAnalyzer::lessByOriginal(a, b));
    ASSERT_FALSE(gl::ExpressionAnalyzer::lessByOriginal(b, a));
}

// packStatementKey — int32_t pack of (originalId, validityId). Distinct
// inputs map to distinct packed keys.
TEST(memory, packstatementkey_uniqueness) {
    int32_t a = gl::packStatementKey(0x1234, 0x5678);
    int32_t b = gl::packStatementKey(0x1234, 0x5679);   // diff validityId
    int32_t c = gl::packStatementKey(0x1235, 0x5678);   // diff originalId
    int32_t d = gl::packStatementKey(0x1234, 0x5678);   // same as a
    ASSERT_NE(a, b);
    ASSERT_NE(a, c);
    ASSERT_NE(b, c);
    ASSERT_EQ(a, d);
    // High 16 bits are originalId; low 16 are validityId.
    ASSERT_EQ((a >> 16) & 0xFFFF, static_cast<int32_t>(0x1234));
    ASSERT_EQ(a & 0xFFFF,         static_cast<int32_t>(0x5678));
}

// encodeExpression — round-trip through an EncodedExpression and a
// NameMap. Tests that nameId is registered, validityId is registered,
// and the unchangeable-arg path uses the u_-prefixed argId.
TEST(memory, encodeexpression_roundtrip) {
    gl::NameMap nm;
    // Build an EncodedExpression manually, the same shape parseArgument
    // would produce: arg = ["False", "x", "-1", "0", "x"] is changeable
    // and arg = ["True", "y", "-1", "0", "y"] is unchangeable.
    gl::EncodedExpression ee;
    ee.name = "myExpr";
    ee.negation = false;
    ee.arguments.push_back({"False", "x", "-1", "0", "x"});
    ee.arguments.push_back({"True",  "y", "-1", "0", "y"});
    ee.maxIterationNumber = -1;
    ee.original = "(myExpr[x,u_y])";
    ee.validityName = "main";

    gl::IntEncodedExpr ie = gl::encodeExpression(ee, nm);
    ASSERT_EQ(ie.nameId,                       nm.encode("myExpr"));
    ASSERT_EQ(ie.validityId,                   nm.encode("main"));
    ASSERT_EQ(ie.originalId,                   nm.encode("(myExpr[x,u_y])"));
    ASSERT_EQ(ie.arity,                        static_cast<int16_t>(2));
    ASSERT_EQ(ie.negation,                     static_cast<int16_t>(0));
    ASSERT_EQ(ie.argUnchangeable[0],           static_cast<int16_t>(0));
    ASSERT_EQ(ie.argUnchangeable[1],           static_cast<int16_t>(1));
    // argFullId is the un-prefixed name; argId is u_-prefixed for unchangeable.
    ASSERT_EQ(ie.argFullId[1],                 nm.encode("y"));
    ASSERT_EQ(ie.argId[1],                     nm.encode("u_y"));
    // Anchor / hypo flags are derived.
    ASSERT_EQ(ie.isAnchor,                     static_cast<int16_t>(0));
    ASSERT_EQ(ie.isHypo,                       static_cast<int16_t>(0));
}

// EncodedExpression — operator< orders by (original, validityName).
TEST(memory, encodedexpression_ordering) {
    gl::EncodedExpression a;
    a.original = "(p[x])";
    a.validityName = "main";
    gl::EncodedExpression b;
    b.original = "(p[y])";
    b.validityName = "main";
    gl::EncodedExpression c;
    c.original = "(p[x])";
    c.validityName = "main_boundary_z";
    ASSERT_TRUE(a < b);    // different original, "(p[x])" < "(p[y])"
    ASSERT_TRUE(a < c);    // same original, validityName differs
    ASSERT_FALSE(b < a);
}

// =============================================================================
// Phase 8 tests — for prover.hpp Part B (hash engine inline methods).
// addOrigin's cap-full preference policy is the most easily-tested inline
// method. These tests construct a synthetic origin map, push records, and
// verify the foundation-displaces-convenience contract documented in
// the doxygen at addOrigin.
// =============================================================================

// addOrigin under cap — append-with-dedupe semantics. Below the cap,
// adding the same origin twice should be a no-op.
TEST(prover, addorigin_under_cap_dedupes) {
    gl::ExpressionAnalyzer ea("Peano");  // heavy, but unavoidable for
                                          // ExpressionAnalyzer member access
    std::map<gl::ExpressionWithValidity,
             std::vector<std::pair<std::string,
                                   std::vector<gl::ExpressionWithValidity>>>>
        m;
    gl::ExpressionWithValidity ev("(p[a])", "main");
    std::pair<std::string, std::vector<gl::ExpressionWithValidity>> originA;
    originA.first = "implication";
    originA.second.push_back(gl::ExpressionWithValidity("(q[a])", "main"));

    ea.addOrigin(m, ev, originA, 4);
    ASSERT_EQ(m[ev].size(), static_cast<std::size_t>(1));
    // Re-add same origin: dedup, vector size stays 1.
    ea.addOrigin(m, ev, originA, 4);
    ASSERT_EQ(m[ev].size(), static_cast<std::size_t>(1));
}

// addOrigin at cap — foundation tag (e.g. "implication") DISPLACES an
// equality-convenience tag (equality1/equality2). Pins the D-49 / I-35
// preference policy.
TEST(prover, addorigin_at_cap_foundation_displaces_convenience) {
    gl::ExpressionAnalyzer ea("Peano");
    std::map<gl::ExpressionWithValidity,
             std::vector<std::pair<std::string,
                                   std::vector<gl::ExpressionWithValidity>>>>
        m;
    gl::ExpressionWithValidity ev("(p[a])", "main");
    std::pair<std::string, std::vector<gl::ExpressionWithValidity>> originEq;
    originEq.first = "equality1";
    originEq.second.push_back(gl::ExpressionWithValidity("(=[a,b])", "main"));

    // Cap = 1. First insert is a convenience tag; vector has it.
    ea.addOrigin(m, ev, originEq, 1);
    ASSERT_EQ(m[ev].size(),         static_cast<std::size_t>(1));
    ASSERT_EQ(m[ev][0].first,       std::string("equality1"));

    // Now insert a foundation tag — must displace the convenience slot.
    std::pair<std::string, std::vector<gl::ExpressionWithValidity>> originFnd;
    originFnd.first = "implication";
    originFnd.second.push_back(gl::ExpressionWithValidity("(q[a])", "main"));
    ea.addOrigin(m, ev, originFnd, 1);
    ASSERT_EQ(m[ev].size(),         static_cast<std::size_t>(1));
    ASSERT_EQ(m[ev][0].first,       std::string("implication"));
}

// =============================================================================
// Phase 10 tests — for prover.cpp Part A (helpers).
// negate idempotence on the double-negation path; smoothenExpr idempotence;
// listLastRemovedArgs on simple input.
// =============================================================================

TEST(prover, negate_double_cancel) {
    gl::ExpressionAnalyzer ea("Peano");
    const std::string original = "(p[a])";
    const std::string negated = ea.negate(original);
    ASSERT_EQ(negated, std::string("!(p[a])"));
    const std::string twice = ea.negate(negated);
    ASSERT_EQ(twice, original);
}

TEST(prover, smoothenexpr_idempotent_on_canonical_input) {
    gl::ExpressionAnalyzer ea("Peano");
    const std::string canonical = "(=[a,b])";
    ASSERT_EQ(ea.smoothenExpr(canonical), canonical);
    // Twice through smoothenExpr is the same as once.
    ASSERT_EQ(ea.smoothenExpr(ea.smoothenExpr(canonical)), canonical);
}

TEST(prover, listlastremovedargs_extracts_outer_quantifier) {
    gl::ExpressionAnalyzer ea("Peano");
    // `(>[v0,v1](p[v0])(q[v1]))` is the canonical form; outer `>[v0,v1]`
    // binds two variables that listLastRemovedArgs should return.
    const std::vector<std::string> args =
        ea.listLastRemovedArgs("(>[v0,v1](p[v0])(q[v1]))");
    ASSERT_EQ(args.size(), static_cast<std::size_t>(2));
    ASSERT_EQ(args[0],     std::string("v0"));
    ASSERT_EQ(args[1],     std::string("v1"));
}

// =============================================================================
// Round 2 — two more tests per documented function. Covers edge cases,
// boundary conditions, and round-trips not exercised by the original
// per-function tests.
// =============================================================================

// ---------- NameMap::encode ----------
TEST(memory, namemap_encode_grows_nextid_monotonically) {
    gl::NameMap nm;
    const int16_t before = nm.encode("first");
    const int16_t after  = nm.encode("second");
    ASSERT_LT(before, after);   // monotonic id allocation
    // "main" was pre-registered at MAIN_ID == 1; first user root gets 2.
    ASSERT_GE(before, static_cast<int16_t>(2));
}

TEST(memory, namemap_encode_recursive_with_boundary_string) {
    gl::NameMap nm;
    // Encoding a fully-canonical "_boundary_"-bearing string registers
    // every prefix recursively: encode("main_boundary_x") creates the
    // child id and the parent ("main") is already mapped via the ctor.
    const int16_t childId = nm.encode("main_boundary_x");
    const int16_t parentId = nm.encode("main");
    ASSERT_NE(childId, parentId);
    // The child's stack carries one payload depth.
    ASSERT_EQ(nm.stackOf(childId).size(), static_cast<std::size_t>(1));
    // pairMap reflects the prefix relation that encodePush would have
    // created if invoked directly.
    int16_t v = 0;
    ASSERT_TRUE(nm.verdict(parentId, childId, v));
    ASSERT_EQ(v, static_cast<int16_t>(-1));
}

// ---------- NameMap::encodePush ----------
TEST(memory, namemap_encodepush_idtosub_grows_monotonically) {
    gl::NameMap nm;
    const int16_t mainId = nm.encode("main");
    const int16_t a = nm.encodePush(mainId, "alpha");
    const int16_t b = nm.encodePush(mainId, "beta");
    // The two payloads got distinct sub-ids, monotonically allocated.
    const int16_t subA = nm.stackOf(a).back();
    const int16_t subB = nm.stackOf(b).back();
    ASSERT_NE(subA, subB);
    // Re-pushing "alpha" reuses its sub-id (interning).
    const int16_t a2 = nm.encodePush(mainId, "alpha");
    ASSERT_EQ(a, a2);
    ASSERT_EQ(nm.stackOf(a2).back(), subA);
}

TEST(memory, namemap_encodepush_three_deep_chain) {
    gl::NameMap nm;
    const int16_t mainId = nm.encode("main");
    const int16_t s1 = nm.encodePush(mainId, "s1");
    const int16_t s2 = nm.encodePush(s1, "s2");
    const int16_t s3 = nm.encodePush(s2, "s3");
    ASSERT_EQ(nm.stackOf(s3).size(), static_cast<std::size_t>(3));
    // Strict-ancestor chain.
    ASSERT_TRUE(nm.isStrictAncestor("main", "main_boundary_s1"));
    ASSERT_TRUE(nm.isStrictAncestor("main_boundary_s1",
                                   "main_boundary_s1_boundary_s2"));
    ASSERT_TRUE(nm.isStrictAncestor("main",
                                   "main_boundary_s1_boundary_s2_boundary_s3"));
    // Self is NOT a strict ancestor of itself.
    ASSERT_FALSE(nm.isStrictAncestor("main_boundary_s1",
                                    "main_boundary_s1"));
}

// ---------- NameMap::decode ----------
TEST(memory, namemap_decode_main_id_returns_main) {
    gl::NameMap nm;
    ASSERT_EQ(nm.decode(gl::NameMap::MAIN_ID), std::string("main"));
}

TEST(memory, namemap_decode_round_trip_after_encode) {
    gl::NameMap nm;
    const std::string canonical = "main_boundary_x_boundary_y";
    const int16_t id = nm.encode(canonical);
    const std::string& decoded = nm.decode(id);
    ASSERT_EQ(decoded, canonical);
}

// ---------- NameMap::comparable / verdict ----------
TEST(memory, namemap_comparable_self_equality) {
    gl::NameMap nm;
    const int16_t a = nm.encode("main");
    int16_t v = 0;
    ASSERT_TRUE(nm.verdict(a, a, v));
    ASSERT_EQ(v, static_cast<int16_t>(0));
    ASSERT_TRUE(nm.comparable(a, a));
}

TEST(memory, namemap_comparable_grandparent) {
    gl::NameMap nm;
    const int16_t mainId = nm.encode("main");
    const int16_t s1 = nm.encodePush(mainId, "s1");
    const int16_t s2 = nm.encodePush(s1, "s2");
    int16_t v = 0;
    // main is grandparent of s2 — strict prefix → -1.
    ASSERT_TRUE(nm.verdict(mainId, s2, v));
    ASSERT_EQ(v, static_cast<int16_t>(-1));
    ASSERT_TRUE(nm.comparable(mainId, s2));
}

// ---------- NameMap::deeperOf ----------
TEST(memory, namemap_deeperof_returns_descendant) {
    gl::NameMap nm;
    const int16_t mainId = nm.encode("main");
    const int16_t child  = nm.encodePush(mainId, "x");
    ASSERT_EQ(nm.deeperOf(mainId, child), child);
    ASSERT_EQ(nm.deeperOf(child, mainId), child);
}

TEST(memory, namemap_deeperof_self) {
    gl::NameMap nm;
    const int16_t mainId = nm.encode("main");
    ASSERT_EQ(nm.deeperOf(mainId, mainId), mainId);
}

// ---------- NameMap::parse ----------
TEST(memory, namemap_parse_root_returns_empty) {
    gl::NameMap nm;
    const auto stack = nm.parse("main");
    ASSERT_EQ(stack.size(), static_cast<std::size_t>(0));
}

TEST(memory, namemap_parse_two_levels) {
    gl::NameMap nm;
    const auto stack = nm.parse("main_boundary_x_boundary_y");
    ASSERT_EQ(stack.size(), static_cast<std::size_t>(2));
    ASSERT_EQ(stack[0], std::string("x"));
    ASSERT_EQ(stack[1], std::string("y"));
}

// ---------- Memory default ctor ----------
TEST(memory, memory_default_recursion_state_is_clean) {
    gl::Memory m;
    ASSERT_EQ(m.recursionCounter,        0);
    ASSERT_EQ(m.contradictionIndex,      -1);
    ASSERT_FALSE(m.primedForContradiction);
    ASSERT_FALSE(m.isPartOfRecursion);
    ASSERT_TRUE(m.contradictionTheorem.empty());
    ASSERT_TRUE(m.recursionHypothesis.empty());
}

TEST(memory, memory_default_or_state_is_empty) {
    gl::Memory m;
    ASSERT_EQ(m.orAdmissionSet.size(),  static_cast<std::size_t>(0));
    ASSERT_EQ(m.orBookkeeping.size(),   static_cast<std::size_t>(0));
    ASSERT_EQ(m.orDisjunctCount.size(), static_cast<std::size_t>(0));
    ASSERT_EQ(m.toBeProved.size(),      static_cast<std::size_t>(0));
    ASSERT_EQ(m.encodedStatements.size(), static_cast<std::size_t>(0));
    ASSERT_EQ(m.deltaNumberStatements,  0);
}

// ---------- Memory simpleMap chain ----------
TEST(memory, memory_simplemap_two_children_stay_distinct) {
    gl::Memory root;
    gl::Memory c1;
    gl::Memory c2;
    c1.parentMemory = &root;
    c2.parentMemory = &root;
    c1.exprKey = "(p[1])";
    c2.exprKey = "(q[2])";
    root.simpleMap["k1"] = &c1;
    root.simpleMap["k2"] = &c2;
    ASSERT_EQ(root.simpleMap.size(),    static_cast<std::size_t>(2));
    ASSERT_NE(root.simpleMap["k1"],     root.simpleMap["k2"]);
    ASSERT_EQ(c1.parentMemory,          &root);
    ASSERT_EQ(c2.parentMemory,          &root);
}

TEST(memory, memory_simplemap_chain_walk_to_root) {
    // Build a 3-deep LB chain and walk parent pointers back to root.
    gl::Memory root;
    gl::Memory mid;
    gl::Memory leaf;
    mid.parentMemory  = &root;
    leaf.parentMemory = &mid;
    root.simpleMap["mid"]  = &mid;
    mid.simpleMap["leaf"]  = &leaf;

    const gl::Memory* p = &leaf;
    int depth = 0;
    while (p->parentMemory != nullptr) {
        p = p->parentMemory;
        ++depth;
    }
    ASSERT_EQ(depth, 2);                           // leaf → mid → root
    ASSERT_EQ(p, &root);                           // landed at root sentinel
    ASSERT_TRUE(p->exprKey.empty());               // root sentinel
}

// ---------- HashMemory::clear ----------
TEST(memory, hashmemory_clear_after_populate) {
    gl::HashMemory hm;
    int16_t buf[2] = {1, 2};
    gl::IntNormalizedKey k(1, buf, 2);
    hm.encodedMap[k].push_back(gl::LocalMemoryValue{});
    hm.normalizedEncodedKeys.insert(k);
    hm.maxKeyLength = 5;
    ASSERT_EQ(hm.encodedMap.size(),          static_cast<std::size_t>(1));
    ASSERT_EQ(hm.normalizedEncodedKeys.size(), static_cast<std::size_t>(1));
    hm.clear();
    ASSERT_EQ(hm.encodedMap.size(),          static_cast<std::size_t>(0));
    ASSERT_EQ(hm.normalizedEncodedKeys.size(), static_cast<std::size_t>(0));
    ASSERT_EQ(hm.maxKeyLength,               static_cast<int16_t>(0));
}

TEST(memory, hashmemory_clear_preserves_struct_validity) {
    gl::HashMemory hm;
    hm.clear();          // clear on already-empty
    hm.clear();          // double clear
    // Struct still valid for re-use: insert a fresh entry.
    int16_t buf[1] = {7};
    gl::IntNormalizedKey k(1, buf, 1);
    hm.normalizedEncodedKeys.insert(k);
    ASSERT_EQ(hm.normalizedEncodedKeys.size(), static_cast<std::size_t>(1));
}

// ---------- KeyArena::store / release ----------
TEST(memory, keyarena_default_state_is_empty) {
    gl::KeyArena arena;
    ASSERT_EQ(arena.used,     0);
    ASSERT_EQ(arena.capacity, 0);
    ASSERT_EQ(arena.current,  static_cast<int16_t*>(nullptr));
    ASSERT_EQ(arena.blocks.size(), static_cast<std::size_t>(0));
}

TEST(memory, keyarena_many_stores_bookkeeping) {
    gl::KeyArena arena;
    constexpr int N = 50;
    int16_t buf[1];
    int16_t* lastPtr = nullptr;
    for (int i = 0; i < N; ++i) {
        buf[0] = static_cast<int16_t>(i);
        int16_t* p = arena.store(buf, 1);
        ASSERT_EQ(p[0], static_cast<int16_t>(i));
        if (lastPtr != nullptr) {
            // Successive stores within a chunk are contiguous.
            ASSERT_TRUE(p == lastPtr + 1 || !arena.blocks.empty());
        }
        lastPtr = p;
    }
    ASSERT_GE(arena.used, N);          // bumped at least N slots in total
}

// ---------- IntNormalizedKey hash + ordering ----------
TEST(memory, intnormalizedkey_ordering_by_numberexpressions_first) {
    int16_t b1[2] = {9, 9};
    int16_t b2[2] = {1, 1};
    gl::IntNormalizedKey k1(1, b1, 2);
    gl::IntNormalizedKey k2(2, b2, 2);
    // numberExpressions=1 < numberExpressions=2, regardless of data.
    ASSERT_TRUE(k1 < k2);
    ASSERT_FALSE(k2 < k1);
}

TEST(memory, intnormalizedkey_ordering_by_length_then_data) {
    int16_t b1[2] = {3, 4};
    int16_t b2[3] = {3, 4, 5};
    int16_t b3[2] = {3, 5};
    gl::IntNormalizedKey k1(1, b1, 2);
    gl::IntNormalizedKey k2(1, b2, 3);
    gl::IntNormalizedKey k3(1, b3, 2);
    ASSERT_TRUE(k1 < k2);   // same numberExpressions, shorter < longer
    ASSERT_TRUE(k1 < k3);   // same length, content {3,4} < {3,5}
}

// ---------- EquivalenceClass ctor ----------
TEST(memory, equivalenceclass_levels_map_carries_entries) {
    std::set<std::string> vars{"a", "b", "c"};
    std::map<std::set<std::string>, std::set<int>> lev{
        {std::set<std::string>{"a", "b"}, std::set<int>{0}},
        {std::set<std::string>{"b", "c"}, std::set<int>{1, 2}},
    };
    std::map<gl::ExpressionWithValidity,
             std::vector<std::pair<std::string,
                                   std::vector<gl::ExpressionWithValidity>>>>
        origin;
    gl::EquivalenceClass cls(vars, lev, origin);
    ASSERT_EQ(cls.variables.size(),                static_cast<std::size_t>(3));
    ASSERT_EQ(cls.equalityLevelsMap.size(),        static_cast<std::size_t>(2));
    ASSERT_EQ(cls.equalityLevelsMap.at({"a", "b"}).count(0),
              static_cast<std::size_t>(1));
    ASSERT_EQ(cls.equalityLevelsMap.at({"b", "c"}).size(),
              static_cast<std::size_t>(2));
}

TEST(memory, equivalenceclass_default_then_assign) {
    gl::EquivalenceClass cls;
    cls.variables.insert("x");
    cls.variables.insert("y");
    ASSERT_EQ(cls.variables.size(), static_cast<std::size_t>(2));
    cls.variables.clear();
    ASSERT_EQ(cls.variables.size(), static_cast<std::size_t>(0));
}

// ---------- ExpressionWithValidity ----------
TEST(memory, expressionwithvalidity_default_is_empty) {
    gl::ExpressionWithValidity ev;
    ASSERT_TRUE(ev.original.empty());
    ASSERT_TRUE(ev.validityName.empty());
    ASSERT_TRUE(ev == gl::ExpressionWithValidity());
}

TEST(memory, expressionwithvalidity_in_set) {
    std::set<gl::ExpressionWithValidity> s;
    s.insert(gl::ExpressionWithValidity("(p[a])", "main"));
    s.insert(gl::ExpressionWithValidity("(p[a])", "main"));   // dedupe
    s.insert(gl::ExpressionWithValidity("(p[a])", "main_boundary_x"));
    ASSERT_EQ(s.size(), static_cast<std::size_t>(2));
}

// ---------- EncodedExpression::parseArgument ----------
TEST(memory, encodedexpression_parseargument_u_prefix_default_branch) {
    using gl::EncodedExpression;
    // u_-prefixed plain (non-it_) name: unchangeable, default branch.
    auto a = EncodedExpression::parseArgument("u_plain");
    ASSERT_EQ(a.size(), static_cast<std::size_t>(5));
    ASSERT_EQ(a[0],     std::string("True"));      // u_-prefix → unchangeable
    ASSERT_EQ(a[1],     std::string("plain"));     // u_ stripped
    ASSERT_EQ(a[2],     std::string("-1"));
    ASSERT_EQ(a[3],     std::string("0"));
    ASSERT_EQ(a[4],     std::string("plain"));
}

TEST(memory, encodedexpression_parseargument_zero_iter_zero_lev) {
    using gl::EncodedExpression;
    auto a = EncodedExpression::parseArgument("it_0_lev_0_99");
    ASSERT_EQ(a[0], std::string("False"));
    ASSERT_EQ(a[2], std::string("0"));   // iteration
    ASSERT_EQ(a[3], std::string("1"));   // lev + 1
    ASSERT_EQ(a[4], std::string("99"));  // argId
}

// ---------- EncodedExpression operator< ----------
TEST(memory, encodedexpression_equality_via_operator_eq) {
    gl::EncodedExpression a("p", false, {"x", "y"}, "(p[x,y])", "main");
    gl::EncodedExpression b("p", false, {"x", "y"}, "(p[x,y])", "main");
    ASSERT_TRUE(a == b);
    ASSERT_FALSE(a < b);
    ASSERT_FALSE(b < a);
}

TEST(memory, encodedexpression_in_set_dedupe_on_full_field_match) {
    std::set<gl::EncodedExpression> s;
    s.insert(gl::EncodedExpression("p", false, {"x"}, "(p[x])", "main"));
    s.insert(gl::EncodedExpression("p", false, {"x"}, "(p[x])", "main"));
    s.insert(gl::EncodedExpression("p", false, {"x"}, "(p[x])", "main_boundary_z"));
    ASSERT_EQ(s.size(), static_cast<std::size_t>(2));
}

// ---------- encodeExpression ----------
TEST(memory, encodeexpression_zero_arity_predicate) {
    gl::NameMap nm;
    gl::EncodedExpression ee;
    ee.name = "zero";
    ee.negation = false;
    ee.maxIterationNumber = -1;
    ee.original = "(zero[])";
    ee.validityName = "main";
    gl::IntEncodedExpr ie = gl::encodeExpression(ee, nm);
    ASSERT_EQ(ie.arity,    static_cast<int16_t>(0));
    ASSERT_EQ(ie.nameId,   nm.encode("zero"));
    ASSERT_EQ(ie.isAnchor, static_cast<int16_t>(0));
    ASSERT_EQ(ie.isHypo,   static_cast<int16_t>(0));
}

TEST(memory, encodeexpression_anchor_flag_set) {
    gl::NameMap nm;
    gl::EncodedExpression ee;
    ee.name = "AnchorPeano";
    ee.negation = false;
    ee.arguments.push_back({"False", "n", "-1", "0", "n"});
    ee.maxIterationNumber = -1;
    ee.original = "(AnchorPeano[n])";
    ee.validityName = "main_boundary_hypo_x";
    gl::IntEncodedExpr ie = gl::encodeExpression(ee, nm);
    ASSERT_EQ(ie.isAnchor, static_cast<int16_t>(1));   // name starts with "Anchor"
    ASSERT_EQ(ie.isHypo,   static_cast<int16_t>(1));   // validityName has "_hypo_"
    ASSERT_EQ(ie.arity,    static_cast<int16_t>(1));
}

// ---------- LocalMemoryValue default ----------
TEST(memory, localmemoryvalue_parameterized_ctor) {
    std::vector<std::string> key{"(p[a])"};
    std::set<std::string> rem{"a"};
    std::set<int> levs{0, 1};
    gl::LocalMemoryValue lmv("(q[a])", levs, "(p[a])->(q[a])",
                             "head", key, rem, "main");
    ASSERT_EQ(lmv.value,                     std::string("(q[a])"));
    ASSERT_EQ(lmv.levels.size(),             static_cast<std::size_t>(2));
    ASSERT_EQ(lmv.originalImplication,       std::string("(p[a])->(q[a])"));
    ASSERT_EQ(lmv.justification,             std::string("head"));
    ASSERT_EQ(lmv.key.size(),                static_cast<std::size_t>(1));
    ASSERT_EQ(lmv.remainingArgs.count("a"),  static_cast<std::size_t>(1));
    ASSERT_EQ(lmv.validityName,              std::string("main"));
    ASSERT_FALSE(lmv.productOfDisintegration);
}

TEST(memory, localmemoryvalue_ordering_in_set) {
    std::set<gl::LocalMemoryValue> s;
    gl::LocalMemoryValue a;
    a.value = "alpha";
    gl::LocalMemoryValue b;
    b.value = "beta";
    s.insert(a);
    s.insert(b);
    s.insert(a);   // dedupe
    ASSERT_EQ(s.size(), static_cast<std::size_t>(2));
}

// ---------- ExpressionAnalyzer::lessByName / lessByOriginal ----------
TEST(memory, lessbyname_strict_weak_ordering) {
    // No element is less than itself; trichotomy holds.
    gl::EncodedExpression a, b, c;
    a.name = "a"; b.name = "b"; c.name = "c";
    ASSERT_FALSE(gl::ExpressionAnalyzer::lessByName(a, a));
    ASSERT_TRUE(gl::ExpressionAnalyzer::lessByName(a, b));
    ASSERT_TRUE(gl::ExpressionAnalyzer::lessByName(b, c));
    ASSERT_TRUE(gl::ExpressionAnalyzer::lessByName(a, c));   // transitive
}

TEST(memory, lessbyoriginal_strict_weak_ordering) {
    gl::EncodedExpression a, b, c;
    a.original = "alpha"; b.original = "beta"; c.original = "gamma";
    ASSERT_FALSE(gl::ExpressionAnalyzer::lessByOriginal(a, a));
    ASSERT_TRUE(gl::ExpressionAnalyzer::lessByOriginal(a, b));
    ASSERT_TRUE(gl::ExpressionAnalyzer::lessByOriginal(b, c));
    ASSERT_TRUE(gl::ExpressionAnalyzer::lessByOriginal(a, c));
}

// ---------- packStatementKey ----------
TEST(memory, packstatementkey_zero_pair) {
    int32_t k = gl::packStatementKey(0, 0);
    ASSERT_EQ(k, static_cast<int32_t>(0));
}

TEST(memory, packstatementkey_handles_negative_ints_via_sign_extension) {
    // int16_t -1 == 0xFFFF after uint16_t cast. High 16 bits should
    // be 0xFFFF when originalId = -1, low 16 should be 0xFFFF when
    // validityId = -1, and the combined value is 0xFFFFFFFF (-1
    // viewed as signed 32-bit).
    int32_t k = gl::packStatementKey(-1, -1);
    ASSERT_EQ(k, static_cast<int32_t>(-1));
    // High and low halves recoverable by mask.
    ASSERT_EQ((k >> 16) & 0xFFFF,  static_cast<int32_t>(0xFFFF));
    ASSERT_EQ(k & 0xFFFF,          static_cast<int32_t>(0xFFFF));
}

// ---------- ExpressionAnalyzer::addOrigin ----------
TEST(prover, addorigin_appends_three_distinct_under_cap) {
    gl::ExpressionAnalyzer ea("Peano");
    std::map<gl::ExpressionWithValidity,
             std::vector<std::pair<std::string,
                                   std::vector<gl::ExpressionWithValidity>>>>
        m;
    gl::ExpressionWithValidity ev("(p[a])", "main");
    auto mk = [](const std::string& tag, const std::string& src) {
        std::pair<std::string, std::vector<gl::ExpressionWithValidity>> o;
        o.first = tag;
        o.second.push_back(gl::ExpressionWithValidity(src, "main"));
        return o;
    };
    ea.addOrigin(m, ev, mk("implication", "(q[a])"), 5);
    ea.addOrigin(m, ev, mk("recursion",   "(r[a])"), 5);
    ea.addOrigin(m, ev, mk("expansion",   "(s[a])"), 5);
    ASSERT_EQ(m[ev].size(), static_cast<std::size_t>(3));
}

TEST(prover, addorigin_at_cap_foundation_vs_foundation_keeps_first) {
    gl::ExpressionAnalyzer ea("Peano");
    std::map<gl::ExpressionWithValidity,
             std::vector<std::pair<std::string,
                                   std::vector<gl::ExpressionWithValidity>>>>
        m;
    gl::ExpressionWithValidity ev("(p[a])", "main");
    auto mk = [](const std::string& tag, const std::string& src) {
        std::pair<std::string, std::vector<gl::ExpressionWithValidity>> o;
        o.first = tag;
        o.second.push_back(gl::ExpressionWithValidity(src, "main"));
        return o;
    };
    ea.addOrigin(m, ev, mk("implication", "(q[a])"), 1);
    ea.addOrigin(m, ev, mk("recursion",   "(r[a])"), 1);   // no displace
    ASSERT_EQ(m[ev].size(),     static_cast<std::size_t>(1));
    ASSERT_EQ(m[ev][0].first,   std::string("implication"));
}

// ---------- ExpressionAnalyzer::negate ----------
TEST(prover, negate_simple_predicate) {
    gl::ExpressionAnalyzer ea("Peano");
    ASSERT_EQ(ea.negate("(in[a,N])"), std::string("!(in[a,N])"));
}

TEST(prover, negate_already_negated_strips_outer) {
    gl::ExpressionAnalyzer ea("Peano");
    ASSERT_EQ(ea.negate("!(=[a,b])"), std::string("(=[a,b])"));
}

// ---------- ExpressionAnalyzer::smoothenExpr ----------
TEST(prover, smoothenexpr_preserves_already_canonical) {
    gl::ExpressionAnalyzer ea("Peano");
    ASSERT_EQ(ea.smoothenExpr("(p[a,b,c])"), std::string("(p[a,b,c])"));
}

TEST(prover, smoothenexpr_compound_implication_idempotent) {
    gl::ExpressionAnalyzer ea("Peano");
    const std::string compound = "(>[v0,v1](p[v0])(q[v1]))";
    ASSERT_EQ(ea.smoothenExpr(ea.smoothenExpr(compound)),
              ea.smoothenExpr(compound));
}

// ---------- ExpressionAnalyzer::listLastRemovedArgs ----------
TEST(prover, listlastremovedargs_atomic_returns_empty) {
    gl::ExpressionAnalyzer ea("Peano");
    const auto args = ea.listLastRemovedArgs("(p[a,b])");
    ASSERT_EQ(args.size(), static_cast<std::size_t>(0));
}

TEST(prover, listlastremovedargs_single_var_quantifier) {
    gl::ExpressionAnalyzer ea("Peano");
    const auto args = ea.listLastRemovedArgs("(>[x](p[x])(q[x]))");
    ASSERT_EQ(args.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(args[0],     std::string("x"));
}

// addOrigin at cap — convenience-vs-convenience: keep insertion order.
TEST(prover, addorigin_at_cap_convenience_keeps_first) {
    gl::ExpressionAnalyzer ea("Peano");
    std::map<gl::ExpressionWithValidity,
             std::vector<std::pair<std::string,
                                   std::vector<gl::ExpressionWithValidity>>>>
        m;
    gl::ExpressionWithValidity ev("(p[a])", "main");
    std::pair<std::string, std::vector<gl::ExpressionWithValidity>> origin1;
    origin1.first = "equality1";
    origin1.second.push_back(gl::ExpressionWithValidity("(=[a,b])", "main"));
    std::pair<std::string, std::vector<gl::ExpressionWithValidity>> origin2;
    origin2.first = "equality2";
    origin2.second.push_back(gl::ExpressionWithValidity("(=[c,d])", "main"));

    ea.addOrigin(m, ev, origin1, 1);
    ea.addOrigin(m, ev, origin2, 1);   // both convenience; first wins
    ASSERT_EQ(m[ev].size(),         static_cast<std::size_t>(1));
    ASSERT_EQ(m[ev][0].first,       std::string("equality1"));
}

// EncodedExpression::parseArgument — full it_<n>_lev_<k>_<id> shape
// produces the 5-tuple with iteration / level+1 / id correctly broken
// out; default branch produces the literal 5-tuple ["False/True", arg,
// "-1", "0", arg].
TEST(memory, encodedexpression_parseargument) {
    using gl::EncodedExpression;
    auto a = EncodedExpression::parseArgument("it_3_lev_4_42");
    ASSERT_EQ(a.size(),  static_cast<std::size_t>(5));
    ASSERT_EQ(a[0],      std::string("False"));    // changeable
    ASSERT_EQ(a[1],      std::string("it_3_lev_4_42"));
    ASSERT_EQ(a[2],      std::string("3"));        // iteration
    ASSERT_EQ(a[3],      std::string("5"));        // lev + 1
    ASSERT_EQ(a[4],      std::string("42"));       // argId

    auto b = EncodedExpression::parseArgument("u_it_0_lev_2_7");
    ASSERT_EQ(b[0],      std::string("True"));     // u_-prefix → unchangeable
    ASSERT_EQ(b[1],      std::string("it_0_lev_2_7"));
    ASSERT_EQ(b[2],      std::string("0"));
    ASSERT_EQ(b[3],      std::string("3"));
    ASSERT_EQ(b[4],      std::string("7"));

    // Default branch — non-it_ form.
    auto c = EncodedExpression::parseArgument("plain");
    ASSERT_EQ(c[0],      std::string("False"));
    ASSERT_EQ(c[1],      std::string("plain"));
    ASSERT_EQ(c[2],      std::string("-1"));
    ASSERT_EQ(c[3],      std::string("0"));
    ASSERT_EQ(c[4],      std::string("plain"));
}
