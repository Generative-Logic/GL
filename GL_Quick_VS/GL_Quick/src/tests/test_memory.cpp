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
/// memory module:
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
/// storage (I-13), `HashMemory` round-trip, `Memory` parent chain,
/// `EquivalenceClass` merge, gen-arena key store, and
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
#include <cstring>
#include <regex>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

namespace {
    /// Standalone-NameMap rig (strings campaign): a NameMap is a facade
    /// over two cold string tables; this binds one to private tables on
    /// the harness pool (declaration order: manager first, so the tables
    /// destruct before it).
    struct NameMapRig {
        gl::LbArena lb{ &gl::staticMemory() };
        gl::DirtyState dirty = gl::DirtyState::Clean;
        gl::ColdStringTable names{ &lb, &dirty };
        gl::ColdStringTable subs{ &lb, &dirty };
        gl::PagedVector<gl::ValidityNode> nodes{ &lb, &dirty };
        gl::NameMap nm;
        NameMapRig() { nm.bind(&names, &subs, &nodes); }
    };

    /// @brief Test-local byte-identical oracle for the retired production
    ///        helper `encodeValueVector`.
    ///
    /// @details
    /// Production code now inlines the positional value-id mint loop at its
    /// two `makeNormalizedKeysForAdmission` call sites (still-heap install
    /// sites where the boundary heap legitimately stays), so the free
    /// function left the `standardProcessing` tree. Its Rule-18 differential
    /// reference — a verbatim copy of the former body — lives here for the
    /// eight test builders that exercised it, per the ce-twin survivorship
    /// convention (a prover-only heap helper is not kept in the production
    /// tree). Positional `vi.encode(strings[i])` preserves mint order (I-84).
    ///
    /// @param strings The strings, in their positional order.
    /// @param vi      The value interner (mint side).
    /// @return The id vector, same positions.
    inline std::vector<int32_t> encodeValueVectorOracle(
        const std::vector<std::string>& strings, gl::ValueInterner& vi)
    {
        std::vector<int32_t> out;
        out.reserve(strings.size());
        for (const std::string& s : strings) out.push_back(vi.encode(s));
        return out;
    }
} // namespace


TEST(memory, namemap_main_id_one) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    ASSERT_EQ(nm.encode("main"), static_cast<int16_t>(gl::NameMap::MAIN_ID));
    ASSERT_EQ(nm.encode("main"), static_cast<int16_t>(1));
}

// Direct int->string (as on main): "main" is lazily interned as cold-table
// id 1 on first touch, and a NameMap id equals its cold-table id (no offset).
TEST(memory, namemap_main_on_table_direct_ids) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    // Before any encode: the names table is empty, yet "main" still resolves
    // (eternal root) and decodes without a table touch.
    ASSERT_EQ(nm.nameCount(), 0);
    ASSERT_EQ(nm.lookup("main"), static_cast<int16_t>(gl::NameMap::MAIN_ID));
    ASSERT_EQ(nm.decode(gl::NameMap::MAIN_ID), std::string("main"));

    // First encode interns "main" as cold-table id 1.
    const int16_t mainId = nm.encode("main");
    ASSERT_EQ(mainId, static_cast<int16_t>(1));
    ASSERT_EQ(nm.nameCount(), 1);                 // "main", on the table
    ASSERT_EQ(nm.decode(mainId), std::string("main"));

    // First real name is id 2 (== its cold-table id, direct), round-tripping.
    const int16_t a = nm.encodePush(mainId, "alpha");
    ASSERT_EQ(a, static_cast<int16_t>(2));
    ASSERT_EQ(nm.decode(a), std::string("main_boundary_alpha"));
    ASSERT_EQ(nm.nameCount(), 2);                 // main + alpha
    ASSERT_EQ(nm.lookup("main_boundary_alpha"), a);
}

// The span overload of encodePush (the heap-free scope-mint primitive) builds
// its canonical name on the string-scratch arena; it must produce byte-identical
// ids and decoded names to the std::string overload (which now delegates to it).
TEST(memory, namemap_encodepush_span_matches_string) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    const int16_t mainId = nm.encode("main");

    const std::string alpha = "alpha", beta = "beta";
    const int16_t a = nm.encodePush(mainId, gl::StrSpan(alpha));
    const int16_t b = nm.encodePush(a, gl::StrSpan(beta));

    // Canonical bytes correct (nested parent + "_boundary_" + payload).
    ASSERT_EQ(nm.decode(a), std::string("main_boundary_alpha"));
    ASSERT_EQ(nm.decode(b), std::string("main_boundary_alpha_boundary_beta"));

    // Dedup on the canonical string: the same payload from the same parent
    // returns the same id via BOTH overloads (std::string delegates to span).
    ASSERT_EQ(nm.encodePush(mainId, gl::StrSpan(alpha)), a);
    ASSERT_EQ(nm.encodePush(mainId, std::string("alpha")), a);
}

// encode() of a full scoped name splits at the last "_boundary_" and recurses
// span-native; the id must match building the same scope via encodePush, and the
// std::string overload (a forwarder) and the span overload must agree.
TEST(memory, namemap_encode_scoped_span_native) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    const int16_t m = nm.encode("main");
    const int16_t a = nm.encodePush(m, "alpha");
    const int16_t viaPush = nm.encodePush(a, "beta");

    const std::string full = "main_boundary_alpha_boundary_beta";
    ASSERT_EQ(nm.encode(full), viaPush);              // std::string -> span forward
    ASSERT_EQ(nm.encode(gl::StrSpan(full)), viaPush); // span-native scoped recursion
}

// encodePush directly from the MAIN_ID constant on a fresh map triggers the
// lazy intern of "main" (so decode(parentId) works) — the NotOrScope path.
TEST(memory, namemap_direct_encodepush_from_main_interns_main) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    // No prior encode("main"); push straight from the MAIN_ID constant.
    const int16_t child = nm.encodePush(gl::NameMap::MAIN_ID, "child");
    ASSERT_EQ(nm.nameCount(), 2);                 // main interned + child
    ASSERT_EQ(nm.decode(gl::NameMap::MAIN_ID), std::string("main"));
    ASSERT_EQ(nm.decode(child), std::string("main_boundary_child"));
    ASSERT_TRUE(nm.comparable(gl::NameMap::MAIN_ID, child));
}

TEST(memory, namemap_encode_idempotent) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
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

// Span overload of encode: byte-identical id to the std::string form on every
// name shape, reads only [ptr, len) (sub-slice safe), and is order-independent.
TEST(memory, namemap_encode_strspan_byte_identical) {
    // Flat name: span hits the direct-mint path; id matches the string form.
    {
        NameMapRig nmRig;
        gl::NameMap& nm = nmRig.nm;
        const int16_t viaStr  = nm.encode(std::string("in3"));
        const int16_t viaSpan = nm.encode(gl::StrSpan(std::string("in3")));
        ASSERT_EQ(viaSpan, viaStr);                 // hit, not a second mint
        ASSERT_EQ(nm.encode(gl::StrSpan(std::string("in3"))), viaStr);
    }
    // "main" via span resolves to the eternal MAIN_ID.
    {
        NameMapRig nmRig;
        gl::NameMap& nm = nmRig.nm;
        ASSERT_EQ(nm.encode(gl::StrSpan(std::string("main"))),
                  static_cast<int16_t>(gl::NameMap::MAIN_ID));
    }
    // Scoped name (contains "_boundary_"): span delegates to the recursive
    // std::string path; id identical to encoding the whole string.
    {
        NameMapRig nmRig;
        gl::NameMap& nm = nmRig.nm;
        const std::string scoped = "main_boundary_hypo_x";
        const int16_t viaStr  = nm.encode(scoped);
        const int16_t viaSpan = nm.encode(gl::StrSpan(scoped));
        ASSERT_EQ(viaSpan, viaStr);
        ASSERT_EQ(nm.decode(viaSpan), scoped);
    }
    // Sub-slice safety: a span over the first 5 bytes of a longer buffer must
    // encode "alpha", never "alphaZZZ" (proves only [ptr, len) is read).
    {
        NameMapRig nmRig;
        gl::NameMap& nm = nmRig.nm;
        const std::string buf = "alphaZZZ";
        const int16_t sliced = nm.encode(gl::StrSpan(buf.data(), 5));
        ASSERT_EQ(sliced, nm.encode(std::string("alpha")));
        ASSERT_EQ(nm.decode(sliced), std::string("alpha"));
        ASSERT_NE(sliced, nm.encode(std::string("alphaZZZ")));
    }
    // Order independence: span-first vs string-first reach identical ids.
    {
        NameMapRig rigA, rigB;
        const int16_t spanFirst = rigA.nm.encode(gl::StrSpan(std::string("gamma")));
        const int16_t strFirst  = rigB.nm.encode(std::string("gamma"));
        ASSERT_EQ(spanFirst, strFirst);
    }
}

TEST(memory, memory_root_sentinel) {
    gl::Memory m;
    ASSERT_TRUE(m.exprKey().empty());
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
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    const int16_t mainId = nm.encode("main");
    const int16_t a = nm.encodePush(mainId, "alpha");
    const int16_t b = nm.encodePush(a, "beta");

    ASSERT_NE(a, mainId);
    ASSERT_NE(b, a);
    ASSERT_NE(b, mainId);

    // Stack content reflects the parent chain bottom-up.
    ASSERT_EQ(nm.stackLen(a), 1);

    ASSERT_EQ(nm.stackLen(b), 2);
    ASSERT_EQ(nm.stackAt(b, 0), nm.stackAt(a, 0));   // parent payload preserved at depth 0

    // verdict() encodes the prefix relation, derived from ancestorsOf:
    // verdict(parent, child) == -1 (parent strict prefix); verdict(child,
    // parent) == +1.
    int16_t v = 0;
    ASSERT_TRUE(nm.verdict(mainId, a, v));
    ASSERT_EQ(v, static_cast<int16_t>(-1));
    ASSERT_TRUE(nm.verdict(a, mainId, v));
    ASSERT_EQ(v, static_cast<int16_t>(1));
    ASSERT_TRUE(nm.verdict(mainId, b, v));
    ASSERT_EQ(v, static_cast<int16_t>(-1));

    // isStrictAncestor agrees with the verdict derivation.
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
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    const int16_t mainId = nm.encode("main");
    const int16_t a1 = nm.encodePush(mainId, "scope1");
    const int16_t a2 = nm.encodePush(mainId, "scope1");
    ASSERT_EQ(a1, a2);

    // Same payload from a different parent is a different id.
    const int16_t a3 = nm.encodePush(a1, "scope1");
    ASSERT_NE(a1, a3);
}

// I-3 — decoded strings must be materialized before any nested mint:
// a view into the cold-table bytes can be invalidated by later interns.
// This test exercises the contract by interning a payload, reading it
// back via decode() (an owned copy), and then forcing further encodes
// that grow the names/subs tables.
TEST(memory, namemap_decode_copy_before_nested_mint_I3) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    const int16_t mainId = nm.encode("main");
    const int16_t s1 = nm.encodePush(mainId, "x");

    // Take a copy of the decoded string before further encodes.
    const std::string copied = nm.decode(s1);

    // Force several pushes to grow the names / subs tables.
    for (int i = 0; i < 64; ++i) {
        nm.encodePush(mainId, std::string("payload_") + std::to_string(i));
    }

    // The copy is intact and matches what decode would produce now.
    ASSERT_EQ(copied, std::string("main_boundary_x"));
    ASSERT_EQ(copied, nm.decode(s1));
}

// comparable() agrees with verdict() existence (both derived from ancestorsOf).
TEST(memory, namemap_comparable_matches_verdict) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    const int16_t mainId = nm.encode("main");
    const int16_t a = nm.encodePush(mainId, "a");
    const int16_t b = nm.encodePush(mainId, "b");   // sibling, divergent

    ASSERT_TRUE(nm.comparable(mainId, mainId));
    ASSERT_TRUE(nm.comparable(mainId, a));
    ASSERT_TRUE(nm.comparable(a, mainId));
    ASSERT_FALSE(nm.comparable(a, b));   // siblings diverge
}

// ancContains is the membership primitive behind verdict/comparable: a strict
// ancestor of id is present in ancestorsOf[id]; id itself is present (self at
// back); a divergent id is absent.
TEST(memory, namemap_anccontains_membership) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    const int16_t mainId = nm.encode("main");
    const int16_t a = nm.encodePush(mainId, "a");
    const int16_t b = nm.encodePush(a, "b");        // main < a < b
    const int16_t c = nm.encodePush(mainId, "c");   // sibling of a; diverges

    ASSERT_TRUE(nm.ancContains(b, a));        // a is an ancestor of b
    ASSERT_TRUE(nm.ancContains(b, mainId));   // main is an ancestor of b
    ASSERT_TRUE(nm.ancContains(b, b));        // self present
    ASSERT_FALSE(nm.ancContains(b, c));       // c diverges from b
    ASSERT_FALSE(nm.ancContains(a, b));       // b is a descendant, not ancestor
    ASSERT_FALSE(nm.ancContains(a, c));       // siblings diverge
}

// pairCount derives the count the dropped pairMap cached: two ordered entries
// per strict-ancestor relationship, i.e. 2 * sum_id (|ancestorsOf[id]| - 1).
TEST(memory, namemap_paircount_derives_relationships) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    const int16_t mainId = nm.encode("main");
    ASSERT_EQ(nm.pairCount(), static_cast<std::size_t>(0));   // only main, no pairs

    const int16_t a = nm.encodePush(mainId, "a");            // {main,a}: 1 strict
    ASSERT_EQ(nm.pairCount(), static_cast<std::size_t>(2));

    const int16_t b = nm.encodePush(a, "b");                 // {main,a,b}: 2 strict
    ASSERT_EQ(nm.pairCount(), static_cast<std::size_t>(6));   // 2 + 4

    const int16_t c = nm.encodePush(mainId, "c");            // {main,c}: 1 strict
    (void)b;
    (void)c;
    ASSERT_EQ(nm.pairCount(), static_cast<std::size_t>(8));   // 6 + 2
}

// Lazy seeding: a fresh NameMap holds no metadata rows (zero pool blocks for a
// transient Memory); the first encode seeds slot 0 + "main" (id 1: empty stack,
// ancestors {main}); a real scope appends after.
TEST(memory, namemap_lazy_seed_on_first_encode) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    ASSERT_EQ(nm.stackSize(), 0);          // unseeded: no rows
    ASSERT_EQ(nm.ancSize(), 0);
    // Eternal-root fallback: MAIN_ID resolves before any encode materializes it.
    ASSERT_TRUE(nm.stackEmpty(gl::NameMap::MAIN_ID));
    ASSERT_EQ(nm.stackLen(gl::NameMap::MAIN_ID), 0);
    ASSERT_EQ(nm.ancLen(gl::NameMap::MAIN_ID), 1);
    ASSERT_EQ(nm.ancAt(gl::NameMap::MAIN_ID, 0),
              static_cast<int16_t>(gl::NameMap::MAIN_ID));
    ASSERT_TRUE(nm.ancContains(gl::NameMap::MAIN_ID, gl::NameMap::MAIN_ID));

    const int16_t mainId = nm.encode("main");
    ASSERT_EQ(mainId, static_cast<int16_t>(gl::NameMap::MAIN_ID));
    ASSERT_EQ(nm.stackSize(), 2);          // slot 0 + main
    ASSERT_EQ(nm.ancSize(), 2);
    ASSERT_TRUE(nm.stackEmpty(mainId));    // main's stack is empty
    ASSERT_EQ(nm.ancLen(mainId), 1);       // main's ancestors = {main}
    ASSERT_EQ(nm.ancAt(mainId, 0), mainId);

    const int16_t a = nm.encodePush(mainId, "a");
    ASSERT_EQ(nm.stackSize(), 3);          // + the new scope
    ASSERT_EQ(nm.stackLen(a), 1);          // one payload deep
    ASSERT_EQ(nm.ancLen(a), 2);            // {main, a}
    ASSERT_EQ(nm.ancAt(a, 1), a);          // self at back
}

// D-105 — the request-generation validity prune
// predicate. A main-scope request is always kept (cannot prune — main is the
// root ancestor); a non-main request is kept iff some owner scope is
// comparable, and pruned (false) when every owner diverges.
TEST(prover, ownerset_has_comparable_validity_prune) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    const int16_t mainId = gl::NameMap::MAIN_ID;
    const int16_t child = nm.encodePush(mainId, "hypo_child");
    const int16_t sibling = nm.encodePush(mainId, "hypo_sibling");   // diverges from child

    // main request: trivially kept even against an empty owner set (the skip).
    gl::OwnerSet empty;
    ASSERT_TRUE(gl::ExpressionAnalyzer::ownerSetHasComparable(empty, mainId, nm));

    // Owners are packed composite ids; the prune reads the LOW half (the
    // scope validity id). The high half (the implication id) is arbitrary
    // for this test.
    const int16_t ruleId = nm.encodePush(mainId, "rule_impl");

    // owner at main is an ancestor of the child request -> comparable -> kept.
    gl::OwnerSet atMain;
    atMain.partitionIds.insert(gl::makePartitionId(ruleId, mainId));
    ASSERT_TRUE(gl::ExpressionAnalyzer::ownerSetHasComparable(atMain, child, nm));

    // owner at the equal scope -> comparable -> kept.
    gl::OwnerSet atChild;
    atChild.partitionIds.insert(gl::makePartitionId(ruleId, child));
    ASSERT_TRUE(gl::ExpressionAnalyzer::ownerSetHasComparable(atChild, child, nm));

    // the only owner diverges from the child request -> not comparable -> pruned.
    gl::OwnerSet atSibling;
    atSibling.partitionIds.insert(gl::makePartitionId(ruleId, sibling));
    ASSERT_FALSE(gl::ExpressionAnalyzer::ownerSetHasComparable(atSibling, child, nm));

    // mixed owner set: one diverging + one comparable -> kept (short-circuit).
    gl::OwnerSet mixed;
    mixed.partitionIds.insert(gl::makePartitionId(ruleId, sibling));
    mixed.partitionIds.insert(gl::makePartitionId(ruleId, mainId));
    ASSERT_TRUE(gl::ExpressionAnalyzer::ownerSetHasComparable(mixed, child, nm));
}

// OwnerSetBlob reads hasLooseOwner / partitionCount / partitionId / uSigOffset
// straight off the Codec<OwnerSet> bytes -- the no-decode peek the cold owner-set
// maps' prune uses instead of materializing the OwnerSet. The view's offsets must
// match the codec's layout.
TEST(memory, ownerset_blob_view_field_readers) {
    gl::OwnerSet os;
    os.hasLooseOwner = false;
    os.partitionIds.insert(7);
    os.partitionIds.insert(-3);
    os.partitionIds.insert(100000);
    os.uSignatures.insert(
        std::vector<std::pair<int16_t, int16_t>>{ { 2, 5 }, { 0, 9 } });
    os.uSignatures.insert(std::vector<std::pair<int16_t, int16_t>>{ { 1, 4 } });

    std::vector<char> buf = gl::Codec<gl::OwnerSet>::serialize(os);
    const gl::OwnerSetBlob v{ buf.data(), static_cast<int32_t>(buf.size()) };

    ASSERT_TRUE(v.hasLooseOwner() == os.hasLooseOwner);
    ASSERT_EQ(v.partitionCount(), static_cast<int32_t>(os.partitionIds.size()));
    int32_t i = 0;                          // partitionIds is a set -> ascending
    for (int32_t id : os.partitionIds) ASSERT_EQ(v.partitionId(i++), id);
    // uSigOffset points at the uSignatures count int32.
    ASSERT_EQ(gl::OwnerSetBlob::rdI32(v.p + v.uSigOffset()),
              static_cast<int32_t>(os.uSignatures.size()));

    // The loose flag round-trips as true too.
    gl::OwnerSet loose;
    loose.hasLooseOwner = true;
    loose.partitionIds.insert(1);
    std::vector<char> lbuf = gl::Codec<gl::OwnerSet>::serialize(loose);
    const gl::OwnerSetBlob lv{ lbuf.data(), static_cast<int32_t>(lbuf.size()) };
    ASSERT_TRUE(lv.hasLooseOwner());
}

// partitionAccepts(OwnerSetBlob) (the no-decode twin) must return the SAME verdict
// as partitionAccepts(const std::set&) for every executor index, including the
// unsplit short-circuit.
TEST(memory, ownerset_blob_partition_accepts_matches_set) {
    const int savedCount = gl::g_splitCount;
    const int savedPid = gl::g_splitProcessID;

    gl::OwnerSet os;
    os.partitionIds.insert(gl::makePartitionId(10, 0));
    os.partitionIds.insert(gl::makePartitionId(11, 1));
    os.partitionIds.insert(gl::makePartitionId(12, 2));
    os.partitionIds.insert(gl::makePartitionId(13, 3));
    std::vector<char> buf = gl::Codec<gl::OwnerSet>::serialize(os);
    const gl::OwnerSetBlob v{ buf.data(), static_cast<int32_t>(buf.size()) };

    gl::g_splitCount = 1;                    // unsplit: both keep unconditionally
    gl::g_splitProcessID = 0;
    ASSERT_TRUE(gl::partitionAccepts(os.partitionIds));
    ASSERT_TRUE(gl::partitionAccepts(v));

    for (int N = 2; N <= 4; ++N) {
        gl::g_splitCount = N;
        for (int n = 0; n < N; ++n) {
            gl::g_splitProcessID = n;
            ASSERT_TRUE(gl::partitionAccepts(v) ==
                        gl::partitionAccepts(os.partitionIds));
        }
    }

    gl::g_splitCount = savedCount;           // restore (thread_local persists)
    gl::g_splitProcessID = savedPid;
}

// ownerSetHasComparable(OwnerSetBlob) must agree with the const-OwnerSet& overload
// across every (owner-set, request-scope) pair the value-form test covers.
TEST(prover, ownerset_blob_has_comparable_matches_value) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    const int16_t mainId = gl::NameMap::MAIN_ID;
    const int16_t child = nm.encodePush(mainId, "hypo_child");
    const int16_t sibling = nm.encodePush(mainId, "hypo_sibling");
    const int16_t ruleId = nm.encodePush(mainId, "rule_impl");

    std::vector<gl::OwnerSet> cases(4);
    cases[0].partitionIds.insert(gl::makePartitionId(ruleId, mainId));
    cases[1].partitionIds.insert(gl::makePartitionId(ruleId, child));
    cases[2].partitionIds.insert(gl::makePartitionId(ruleId, sibling));
    cases[3].partitionIds.insert(gl::makePartitionId(ruleId, sibling));
    cases[3].partitionIds.insert(gl::makePartitionId(ruleId, mainId));

    const int16_t reqVids[3] = { mainId, child, sibling };
    for (const gl::OwnerSet& os : cases) {
        std::vector<char> buf = gl::Codec<gl::OwnerSet>::serialize(os);
        const gl::OwnerSetBlob v{ buf.data(), static_cast<int32_t>(buf.size()) };
        for (int16_t reqVid : reqVids) {
            ASSERT_TRUE(
                gl::ExpressionAnalyzer::ownerSetHasComparable(v, reqVid, nm) ==
                gl::ExpressionAnalyzer::ownerSetHasComparable(os, reqVid, nm));
        }
    }
}

// HashMemory smoke — default-constructed HashMemory is empty across every
// slot and clear() is idempotent.
TEST(memory, hashmemory_empty_and_clear) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::HashMemory hm(&lb, &d);
    ASSERT_EQ(hm.encodedMap.count(), 0);
    ASSERT_EQ(hm.normalizedEncodedKeys.count(), 0);
    ASSERT_EQ(hm.maxKeyLength, static_cast<int16_t>(0));
    ASSERT_EQ(hm.originals.count(), 0);
    ASSERT_EQ(hm.admissionMap.count(), 0);
    ASSERT_EQ(hm.rejectedMap.count(), 0);
    ASSERT_EQ(hm.varsInRejectedMapIntegrationKeys.count(), 0);

    hm.clear();   // idempotent on empty
    ASSERT_EQ(hm.encodedMap.count(), 0);
}

// WorkInstruction — the arena-backed working twin of Instruction. Build via the
// append API (elemMark / addElement / commitEntity / setMarkedGoal) and read the
// fields back as spans; no malloc, no ValueInterner needed.
TEST(memory, work_instruction_build_and_read) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::WorkInstruction wi(&lb);
    ASSERT_EQ(wi.entityCount(), 0);

    // Entity 0: two elements, non-empty fields.
    const std::string cat0 = "existence", sig0 = "(s0[a,b])", ds0 = "N";
    const std::string e00 = "(in[a,N])", e01 = "(in[b,N])";
    const int32_t m0 = wi.elemMark();
    wi.addElement(gl::StrSpan(e00));
    wi.addElement(gl::StrSpan(e01));
    wi.commitEntity(gl::StrSpan(cat0), gl::StrSpan(sig0), gl::StrSpan(ds0), 2, m0);

    // Entity 1: no elements, empty definedSet (the "" interns to a real id).
    const std::string cat1 = "and", sig1 = "(s1)", ds1 = "";
    const int32_t m1 = wi.elemMark();
    wi.commitEntity(gl::StrSpan(cat1), gl::StrSpan(sig1), gl::StrSpan(ds1), 0, m1);

    const std::string goal = "(goal[x])";
    wi.setMarkedGoal(gl::StrSpan(goal));

    ASSERT_EQ(wi.entityCount(), 2);
    ASSERT_TRUE(gl::equalSpans(wi.category(0), gl::StrSpan(cat0)));
    ASSERT_TRUE(gl::equalSpans(wi.signature(0), gl::StrSpan(sig0)));
    ASSERT_TRUE(gl::equalSpans(wi.definedSet(0), gl::StrSpan(ds0)));
    ASSERT_EQ(wi.arity(0), 2);
    ASSERT_EQ(wi.elemCount(0), 2);
    ASSERT_TRUE(gl::equalSpans(wi.elemAt(0, 0), gl::StrSpan(e00)));
    ASSERT_TRUE(gl::equalSpans(wi.elemAt(0, 1), gl::StrSpan(e01)));
    ASSERT_EQ(wi.elemCount(1), 0);
    ASSERT_TRUE(gl::equalSpans(wi.definedSet(1), gl::StrSpan(ds1)));   // empty, distinct id
    ASSERT_TRUE(gl::equalSpans(wi.markedGoal(), gl::StrSpan(goal)));

    // entityAt materializes a heap LogicalEntity copy (the per-iteration shadow).
    const gl::LogicalEntity le0 = wi.entityAt(0);
    ASSERT_TRUE(le0.category == cat0);
    ASSERT_TRUE(le0.signature == sig0);
    ASSERT_TRUE(le0.definedSet == ds0);
    ASSERT_EQ(le0.arity, 2);
    ASSERT_EQ(le0.elements.size(), static_cast<std::size_t>(2));
    ASSERT_TRUE(le0.elements[1] == e01);
    const gl::LogicalEntity le1 = wi.entityAt(1);
    ASSERT_EQ(le1.elements.size(), static_cast<std::size_t>(0));
    ASSERT_TRUE(le1.definedSet.empty());
}

// WorkInstruction codecs are byte/id-identical to the heap encodeInstruction /
// decodeInstruction path: encode a heap Instruction, load the arena form from it,
// re-encode, and assert the two IntInstructions are field-identical; then decode
// the arena form via the heap codec and assert it reproduces the original.
TEST(memory, work_instruction_codec_round_trip) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::ColdStringTable valTable(&lb, &d);
    gl::ValueInterner vi;
    vi.bind(&valTable);

    gl::Instruction instr;
    instr.data.push_back(gl::LogicalEntity(
        "existence", { "(in[a,N])", "(in[b,N])" }, "(s0[a,b])", 2, "N"));
    instr.data.push_back(gl::LogicalEntity("and", {}, "(s1)", 0, ""));
    instr.markedGoal = "(goal[x])";

    const gl::IntInstruction ii0 = gl::encodeInstruction(instr, vi);

    gl::WorkInstruction wi(&lb);
    gl::loadFromIntInstruction(wi, ii0, vi);
    const gl::IntInstruction ii1 = gl::encodeWorkInstruction(wi, vi);

    // One interner -> stable ids, so the re-encoded form matches field for field.
    ASSERT_EQ(ii1.data.size(), ii0.data.size());
    ASSERT_EQ(ii1.markedGoal, ii0.markedGoal);
    for (std::size_t i = 0; i < ii0.data.size(); ++i) {
        const gl::IntLogicalEntity& a = ii0.data[i];
        const gl::IntLogicalEntity& b = ii1.data[i];
        ASSERT_EQ(b.category, a.category);
        ASSERT_EQ(b.signature, a.signature);
        ASSERT_EQ(b.definedSet, a.definedSet);
        ASSERT_EQ(b.arity, a.arity);
        ASSERT_EQ(b.elements.size(), a.elements.size());
        for (std::size_t j = 0; j < a.elements.size(); ++j)
            ASSERT_EQ(b.elements[j], a.elements[j]);
    }

    // Decoding the working form via the heap codec reproduces the original.
    const gl::Instruction back = gl::decodeInstruction(ii1, vi);
    ASSERT_TRUE(back == instr);
}

// ArenaIntegrationMap reproduces the heap IntegrationEntryMap (std::map<IntInstruction,
// ValueIdSet, DecodedInstructionLess>) byte-for-byte: build from a recordsAt run, then
// flatten / findOrEmplace / insertValue parity, including the sort after a mutation.
namespace {
    // Field-equality of two IntInstructions (id form): same data + markedGoal.
    inline bool sameIntInstruction(const gl::IntInstruction& a, const gl::IntInstruction& b) {
        if (a.markedGoal != b.markedGoal || a.data.size() != b.data.size()) return false;
        for (std::size_t i = 0; i < a.data.size(); ++i) {
            const gl::IntLogicalEntity& x = a.data[i];
            const gl::IntLogicalEntity& y = b.data[i];
            if (x.category != y.category || x.signature != y.signature
                || x.definedSet != y.definedSet || x.arity != y.arity
                || x.elements != y.elements) return false;
        }
        return true;
    }
    inline bool sameRun(const std::vector<gl::IntegrationEntry>& a,
                        const std::vector<gl::IntegrationEntry>& b) {
        if (a.size() != b.size()) return false;
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (!sameIntInstruction(a[i].instruction, b[i].instruction)) return false;
            if (a[i].valueIds != b[i].valueIds) return false;
        }
        return true;
    }
    // Build the heap IntegrationEntryMap from a run, as admissionIntegrationRecordsAt does.
    inline gl::IntegrationEntryMap heapMapFromRun(const std::vector<gl::IntegrationEntry>& run,
                                                  const gl::ValueInterner& vi) {
        gl::IntegrationEntryMap em(gl::DecodedInstructionLess{ &vi });
        for (const gl::IntegrationEntry& e : run) {
            gl::ValueIdSet payload(gl::DecodedIdLess{ &vi });
            for (const int32_t v : e.valueIds) payload.insert(v);
            em.emplace(e.instruction, std::move(payload));
        }
        return em;
    }
}

// DisintPairChannel reproduces a std::set<ExpressionWithValidity>: append a
// scrambled multiset (with duplicates AND tie cases where original matches but
// validity differs — so a first-seen id order would DISAGREE with the required
// (original,validity) lex order), and assert forEachSorted yields the identical
// sequence the set iterator yields, count() == set size. This is the Rule-18
// byte-twin for the DisintProducts return channel.
TEST(memory, disint_products_channel_matches_set_ewv) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DisintPairChannel ch(&lb);

    const std::vector<std::pair<std::string, std::string>> pairs = {
        { "(in2[b,7,3])", "main" },
        { "(in2[a,7,3])", "main" },
        { "(in2[a,7,3])", "main" },   // duplicate -> skipped
        { "(in2[a,7,3])", "aux" },    // tie on original, differs on validity
        { "(=[x,y])",     "main" },
        { "(in2[b,7,3])", "aux" },    // tie on original
        { "(in2[b,7,3])", "main" },   // duplicate -> skipped
        { "(=[x,y])",     "aux" },
    };

    std::set<gl::ExpressionWithValidity> oracle;
    for (const auto& p : pairs) {
        ch.append(gl::StrSpan(p.first), gl::StrSpan(p.second));
        oracle.insert(gl::ExpressionWithValidity(p.first, p.second));
    }
    ASSERT_EQ(ch.count(), static_cast<int32_t>(oracle.size()));

    std::vector<std::pair<std::string, std::string>> got;
    ch.forEachSorted([&](gl::StrSpan orig, gl::StrSpan val) {
        got.push_back({ gl::StrSpan(orig).toStdString(),
                        gl::StrSpan(val).toStdString() });
    });
    ASSERT_EQ(got.size(), oracle.size());
    std::set<gl::ExpressionWithValidity>::const_iterator it = oracle.begin();
    for (std::size_t i = 0; i < got.size(); ++i, ++it) {
        ASSERT_EQ(got[i].first, it->original);
        ASSERT_EQ(got[i].second, it->validityName);
    }
}

TEST(memory, arena_integration_map_matches_heap) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::ColdStringTable valTable(&lb, &d);
    gl::ValueInterner vi;
    vi.bind(&valTable);

    const auto mkInstr = [&](const std::string& cat, const std::vector<std::string>& elems,
                             const std::string& sig, int ar, const std::string& ds,
                             const std::string& goal) {
        gl::Instruction inst;
        inst.data.push_back(gl::LogicalEntity(cat, elems, sig, ar, ds));
        inst.markedGoal = goal;
        return gl::encodeInstruction(inst, vi);
    };
    const auto sortVals = [&](std::vector<int32_t> v) {
        std::sort(v.begin(), v.end(),
            [&](int32_t a, int32_t b) { return gl::valueIdLess(a, b, vi); });
        return v;
    };

    // Two distinct instructions; the cold-store run is DecodedInstructionLess-sorted with
    // each value run DecodedIdLess-sorted.
    gl::IntegrationEntry e0;
    e0.instruction = mkInstr("existence", { "(in[a,N])", "(in[b,N])" }, "(zeta[a,b])", 2, "N", "(g0)");
    e0.valueIds = sortVals({ vi.encode("vC"), vi.encode("vA"), vi.encode("vB") });
    gl::IntegrationEntry e1;
    e1.instruction = mkInstr("and", {}, "(alpha[c])", 0, "", "(g1)");
    e1.valueIds = sortVals({ vi.encode("vA") });
    std::vector<gl::IntegrationEntry> run = { e0, e1 };

    gl::IntegrationEntryMap em = heapMapFromRun(run, vi);
    gl::ArenaIntegrationMap am(&lb, &vi);
    am.build(run);
    ASSERT_EQ(am.size(), static_cast<int32_t>(run.size()));

    // (1) flatten parity (order + entries + value runs), against the sorted heap map.
    ASSERT_TRUE(sameRun(am.flatten(), gl::flattenIntegrationEntryMap(em)));

    // (2) findOrEmplace: a hit returns the existing entry without appending.
    const int32_t before = am.size();
    const int32_t hit = am.findOrEmplace(e0.instruction);
    ASSERT_EQ(am.size(), before);
    ASSERT_TRUE(am.entryEquals(hit, e0.instruction));
    // a miss appends a new entry.
    const gl::IntInstruction novel =
        mkInstr("existence", { "(in[x,N])" }, "(beta[x])", 1, "N", "(g2)");
    const int32_t miss = am.findOrEmplace(novel);
    ASSERT_EQ(am.size(), before + 1);
    ASSERT_TRUE(am.entryEquals(miss, novel));

    // (3) value insert parity to std::set<int32, DecodedIdLess>, incl. dedup; and the
    //     post-mutation flatten matches the heap map after the same operations.
    gl::IntegrationEntryMap em2 = heapMapFromRun(run, vi);
    em2.emplace(novel, gl::ValueIdSet(gl::DecodedIdLess{ &vi }));   // the findOrEmplace miss
    const int32_t vNew = vi.encode("vNew");
    em2.find(e0.instruction)->second.insert(vNew);
    em2.find(e0.instruction)->second.insert(e0.valueIds.front());  // dedup (already present)
    am.insertValue(hit, vNew);
    am.insertValue(hit, e0.valueIds.front());                       // dedup
    // valuesAt(hit) == the heap set for e0.instruction.
    const std::vector<int32_t> avHit = am.valuesAt(hit);
    const gl::ValueIdSet& refHit = em2.find(e0.instruction)->second;
    ASSERT_EQ(avHit.size(), refHit.size());
    {
        std::size_t k = 0;
        for (const int32_t rv : refHit) { ASSERT_EQ(avHit[k], rv); ++k; }
    }
    // whole-map flatten parity after the append + inserts.
    ASSERT_TRUE(sameRun(am.flatten(), gl::flattenIntegrationEntryMap(em2)));

    // (4) Direct cold round trip (B4b): writeToCold serializes the arena form
    //     straight to the blob bytes, byte-identical to the former
    //     assignRun(flatten()); buildFromCold deserializes the blob bytes straight
    //     back into the arena form. No intermediate heap IntegrationEntry crosses
    //     either boundary.
    gl::TypedColdBlobMap<int32_t, gl::IntegrationEntry> coldDirect(&lb, &d);
    gl::TypedColdBlobMap<int32_t, gl::IntegrationEntry> coldHeap(&lb, &d);
    const int32_t pkCold = 0x00070003;
    am.writeToCold(coldDirect, pkCold);          // direct serialize (no heap record)
    coldHeap.assignRun(pkCold, am.flatten());     // the former heap write path
    const int32_t idD = coldDirect.lookup(pkCold);
    const int32_t idH = coldHeap.lookup(pkCold);
    ASSERT_NE(idD, 0);
    ASSERT_NE(idH, 0);
    ASSERT_EQ(coldDirect.runLen(idD), coldHeap.runLen(idH));
    // byte-identity: every record's blob bytes match the heap path's bytes.
    {
        std::vector<char> sD, sH;
        for (int32_t j = 0; j < coldDirect.runLen(idD); ++j) {
            int32_t lD = 0, lH = 0;
            const char* pD = coldDirect.peekRecordBytes(idD, j, lD, sD);
            const char* pH = coldHeap.peekRecordBytes(idH, j, lH, sH);
            ASSERT_EQ(lD, lH);
            ASSERT_EQ(std::memcmp(pD, pH, static_cast<std::size_t>(lD)), 0);
        }
    }
    // direct read parity: buildFromCold reconstructs the same flatten run as the
    // heap-decoded build(recordsAt).
    gl::ArenaIntegrationMap amDirect(&lb, &vi);
    amDirect.buildFromCold(coldDirect, pkCold);
    ASSERT_EQ(amDirect.size(), am.size());
    ASSERT_TRUE(sameRun(amDirect.flatten(), am.flatten()));

    // cold miss: a never-written key builds an empty arena map.
    gl::ArenaIntegrationMap amMiss(&lb, &vi);
    amMiss.buildFromCold(coldDirect, 0x00ABCDEF);
    ASSERT_EQ(amMiss.size(), 0);
}

// Caller-fill twin of ArenaIntegrationMap::sortedIndices(): the (out,cap) form
// writes the identical DecodedInstructionLess entry-index permutation into a
// caller stack buffer, element-for-element the heap-vector oracle.
TEST(memory, arena_integration_sorted_indices_fill_matches_vector) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::ColdStringTable valTable(&lb, &d);
    gl::ValueInterner vi;
    vi.bind(&valTable);

    const auto mkInstr = [&](const std::string& cat, const std::vector<std::string>& elems,
                             const std::string& sig, int ar, const std::string& ds,
                             const std::string& goal) {
        gl::Instruction inst;
        inst.data.push_back(gl::LogicalEntity(cat, elems, sig, ar, ds));
        inst.markedGoal = goal;
        return gl::encodeInstruction(inst, vi);
    };

    // (1) Empty map: fill count 0, matching sortedIndices().size() == 0.
    {
        gl::ArenaIntegrationMap amEmpty(&lb, &vi);
        int32_t idx[gl::ExecutionParameters::MAX_INTEGRATION_ENTRIES];
        const int32_t n = amEmpty.sortedIndices(
            idx, gl::ExecutionParameters::MAX_INTEGRATION_ENTRIES);
        ASSERT_EQ(n, 0);
        ASSERT_EQ(static_cast<int32_t>(amEmpty.sortedIndices().size()), 0);
    }

    // (2) Multi-entry map built via findOrEmplace in a NON-decoded insertion
    //     order so sortedIndices() imposes a real permutation.
    gl::ArenaIntegrationMap am(&lb, &vi);
    am.findOrEmplace(mkInstr("existence", { "(in[m,N])" }, "(zeta[m])", 1, "N", "(g_mmm)"));
    am.findOrEmplace(mkInstr("and", {}, "(alpha[a])", 0, "", "(g_aaa)"));
    am.findOrEmplace(mkInstr("existence", { "(in[k,N])" }, "(beta[k])", 1, "N", "(g_kkk)"));
    am.findOrEmplace(mkInstr("and", {}, "(gamma[z])", 0, "", "(g_zzz)"));
    ASSERT_GE(am.size(), 4);

    const std::vector<int32_t> heapIdx = am.sortedIndices();
    int32_t idx[gl::ExecutionParameters::MAX_INTEGRATION_ENTRIES];
    const int32_t n = am.sortedIndices(
        idx, gl::ExecutionParameters::MAX_INTEGRATION_ENTRIES);
    ASSERT_EQ(n, static_cast<int32_t>(heapIdx.size()));
    for (int32_t i = 0; i < n; ++i)
        ASSERT_EQ(idx[i], heapIdx[static_cast<std::size_t>(i)]);
}

// Caller-fill twin of ArenaIntegrationMap::valuesAt(): the (ei,out,cap) form
// writes the identical DecodedIdLess value sequence into a caller stack buffer,
// element-for-element the heap-vector oracle (incl. empty + out-of-order-insert).
TEST(memory, arena_integration_values_at_fill_matches_vector) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::ColdStringTable valTable(&lb, &d);
    gl::ValueInterner vi;
    vi.bind(&valTable);

    const auto mkInstr = [&](const std::string& cat, const std::vector<std::string>& elems,
                             const std::string& sig, int ar, const std::string& ds,
                             const std::string& goal) {
        gl::Instruction inst;
        inst.data.push_back(gl::LogicalEntity(cat, elems, sig, ar, ds));
        inst.markedGoal = goal;
        return gl::encodeInstruction(inst, vi);
    };

    gl::ArenaIntegrationMap am(&lb, &vi);
    const int32_t eMulti =
        am.findOrEmplace(mkInstr("existence", { "(in[a,N])" }, "(zeta[a])", 1, "N", "(g0)"));
    const int32_t eEmpty =
        am.findOrEmplace(mkInstr("and", {}, "(alpha[c])", 0, "", "(g1)"));
    // eMulti gets several values inserted OUT of decoded order (+ a dedup); eEmpty
    // stays value-less so the fill count is 0.
    am.insertValue(eMulti, vi.encode("vC"));
    am.insertValue(eMulti, vi.encode("vA"));
    am.insertValue(eMulti, vi.encode("vB"));
    am.insertValue(eMulti, vi.encode("vA"));   // dedup

    const std::vector<int32_t> heapMulti = am.valuesAt(eMulti);
    int32_t vals[gl::ExecutionParameters::MAX_INTEGRATION_ENTRY_VALUES];
    const int32_t nMulti = am.valuesAt(
        eMulti, vals, gl::ExecutionParameters::MAX_INTEGRATION_ENTRY_VALUES);
    ASSERT_EQ(nMulti, static_cast<int32_t>(heapMulti.size()));
    for (int32_t i = 0; i < nMulti; ++i)
        ASSERT_EQ(vals[i], heapMulti[static_cast<std::size_t>(i)]);

    const int32_t nEmpty = am.valuesAt(
        eEmpty, vals, gl::ExecutionParameters::MAX_INTEGRATION_ENTRY_VALUES);
    ASSERT_EQ(nEmpty, 0);
    ASSERT_EQ(static_cast<int32_t>(am.valuesAt(eEmpty).size()), 0);
}

// Door 1 — loadInstructionInto(ei, wi, vi) is the heap-free twin of
// loadFromIntInstruction(wi, instructionAt(ei), vi): both fill a WorkInstruction
// from stored entry ei; re-encoding both through the SAME interner yields
// field-identical IntInstructions (multi-entity, multi-element AND empty-element).
TEST(memory, load_instruction_into_matches_load_from_int) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::ColdStringTable valTable(&lb, &d);
    gl::ValueInterner vi;
    vi.bind(&valTable);

    // Entry 0: two entities — the first with two elements, the second with EMPTY
    // elements (so the door covers both). Entry 1: a single-entity instruction.
    gl::Instruction i0;
    i0.data.push_back(gl::LogicalEntity(
        "existence", { "(in[a,N])", "(in[b,N])" }, "(zeta[a,b])", 2, "N"));
    i0.data.push_back(gl::LogicalEntity("and", {}, "(s1)", 0, ""));
    i0.markedGoal = "(g0)";
    gl::Instruction i1;
    i1.data.push_back(gl::LogicalEntity("atomic", { "(in[x,N])" }, "(beta[x])", 1, "N"));
    i1.markedGoal = "(g1)";

    gl::IntegrationEntry e0; e0.instruction = gl::encodeInstruction(i0, vi);
    gl::IntegrationEntry e1; e1.instruction = gl::encodeInstruction(i1, vi);
    std::vector<gl::IntegrationEntry> run = { e0, e1 };

    gl::ArenaIntegrationMap am(&lb, &vi);
    am.build(run);
    ASSERT_EQ(am.size(), 2);

    for (int32_t ei = 0; ei < am.size(); ++ei) {
        gl::WorkInstruction w0(&lb);
        gl::loadFromIntInstruction(w0, am.instructionAt(ei), vi);   // oracle
        gl::WorkInstruction w1(&lb);
        am.loadInstructionInto(ei, w1, vi);                          // new door
        // One interner -> identical ids => identical content.
        ASSERT_TRUE(sameIntInstruction(
            gl::encodeWorkInstruction(w0, vi), gl::encodeWorkInstruction(w1, vi)));
    }
}

// Door 2 — flattenWorkInstructionInto mints in encodeWorkInstruction's EXACT
// order. On TWO FRESH interners: the oracle encodes wi (viA) then hand-flattens
// the IntInstruction into the door's layout; the door flattens wi (viB). Assert
// (a) the runs are element-identical and (b) viA/viB decode every id 1..count to
// the same bytes with equal count — the strict mint-order proof. The signature
// "(zeta…)" would sort AFTER the "(in…)" element strings, so a
// signature-before-elements mint reorder would diverge the id table and fail.
TEST(memory, flatten_work_instruction_into_matches_encode) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState dA = gl::DirtyState::Clean, dB = gl::DirtyState::Clean;
    gl::ColdStringTable tA(&lb, &dA), tB(&lb, &dB);
    gl::ValueInterner viA, viB;
    viA.bind(&tA); viB.bind(&tB);

    // wi: entity 0 = two elements + a signature that lex-sorts after them; entity
    // 1 = empty elements + empty definedSet; a marked goal.
    gl::WorkInstruction wi(&lb);
    {
        const std::string e00 = "(in[a,N])", e01 = "(in[b,N])";
        const int32_t m = wi.elemMark();
        wi.addElement(gl::StrSpan(e00));
        wi.addElement(gl::StrSpan(e01));
        wi.commitEntity(gl::StrSpan("existence", 9), gl::StrSpan("(zeta[a,b])", 11),
                        gl::StrSpan("N", 1), 2, m);
    }
    {
        const int32_t m = wi.elemMark();
        wi.commitEntity(gl::StrSpan("and", 3), gl::StrSpan("(s1)", 4),
                        gl::StrSpan("", 0), 0, m);
    }
    wi.setMarkedGoal(gl::StrSpan("(g0)", 4));

    // Oracle: encode through viA, then hand-flatten in the door's layout.
    const gl::IntInstruction ii = gl::encodeWorkInstruction(wi, viA);
    std::vector<int32_t> want;
    want.push_back(static_cast<int32_t>(ii.data.size()));
    for (const gl::IntLogicalEntity& le : ii.data) {
        want.push_back(le.category);
        want.push_back(le.signature);
        want.push_back(le.arity);
        want.push_back(le.definedSet);
        want.push_back(static_cast<int32_t>(le.elements.size()));
        for (const int32_t e : le.elements) want.push_back(e);
    }
    want.push_back(ii.markedGoal);

    // New door: flatten through viB into a pool.
    gl::DirtyState dP = gl::DirtyState::Clean;
    gl::PagedVector<int32_t> got(&lb, &dP);
    gl::ArenaIntegrationMap am(&lb, &viB);
    const gl::ArenaIntegrationMap::FlatRange fr =
        am.flattenWorkInstructionInto(wi, viB, got);
    ASSERT_EQ(fr.start, 0);
    ASSERT_EQ(fr.len, static_cast<int32_t>(want.size()));
    ASSERT_EQ(got.size(), static_cast<int32_t>(want.size()));
    for (int32_t i = 0; i < static_cast<int32_t>(want.size()); ++i)
        ASSERT_EQ(got[i], want[static_cast<std::size_t>(i)]);
    // The strict mint-order proof: the two fresh interners are byte-identical
    // (same count, same decoded bytes at every id).
    ASSERT_EQ(viA.table->count(), viB.table->count());
    for (int32_t i = 1; i <= viA.table->count(); ++i)
        ASSERT_TRUE(gl::equalSpans(viA.decodeView(i), viB.decodeView(i)));
}

// Door 3 — findOrEmplaceFlattened parses a flattened run and matches the heap
// findOrEmplace on both a HIT (Q present) and a MISS (novel Q): same returned
// index, same flatten afterwards. It mints nothing (id compare + copy).
TEST(memory, find_or_emplace_flattened_matches_heap) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::ColdStringTable valTable(&lb, &d);
    gl::ValueInterner vi;
    vi.bind(&valTable);

    const auto mkInstr = [&](const std::string& cat, const std::vector<std::string>& elems,
                             const std::string& sig, int ar, const std::string& ds,
                             const std::string& goal) {
        gl::Instruction inst;
        inst.data.push_back(gl::LogicalEntity(cat, elems, sig, ar, ds));
        inst.markedGoal = goal;
        return gl::encodeInstruction(inst, vi);
    };
    // Hand-flatten an IntInstruction into the door's layout (markedGoal LAST).
    const auto runOf = [](const gl::IntInstruction& q, std::vector<int32_t>& out) {
        out.clear();
        out.push_back(static_cast<int32_t>(q.data.size()));
        for (const gl::IntLogicalEntity& le : q.data) {
            out.push_back(le.category);
            out.push_back(le.signature);
            out.push_back(le.arity);
            out.push_back(le.definedSet);
            out.push_back(static_cast<int32_t>(le.elements.size()));
            for (const int32_t e : le.elements) out.push_back(e);
        }
        out.push_back(q.markedGoal);
    };

    gl::IntegrationEntry e0;
    e0.instruction = mkInstr("existence", { "(in[a,N])", "(in[b,N])" }, "(zeta[a,b])", 2, "N", "(g0)");
    gl::IntegrationEntry e1;
    e1.instruction = mkInstr("and", {}, "(alpha[c])", 0, "", "(g1)");
    std::vector<gl::IntegrationEntry> run = { e0, e1 };

    gl::ArenaIntegrationMap amO(&lb, &vi);   // heap findOrEmplace path
    gl::ArenaIntegrationMap amN(&lb, &vi);   // flattened path
    amO.build(run);
    amN.build(run);

    // HIT: e0.instruction already present.
    {
        std::vector<int32_t> r;
        runOf(e0.instruction, r);
        const int32_t a = amO.findOrEmplace(e0.instruction);
        const int32_t b = amN.findOrEmplaceFlattened(r.data(), static_cast<int32_t>(r.size()));
        ASSERT_EQ(a, b);
        ASSERT_EQ(amO.size(), 2);
        ASSERT_EQ(amN.size(), 2);
        ASSERT_TRUE(sameRun(amO.flatten(), amN.flatten()));
    }
    // MISS: a novel instruction appends.
    {
        const gl::IntInstruction novel =
            mkInstr("existence", { "(in[y,N])" }, "(beta[y])", 1, "N", "(g2)");
        std::vector<int32_t> r;
        runOf(novel, r);
        const int32_t a = amO.findOrEmplace(novel);
        const int32_t b = amN.findOrEmplaceFlattened(r.data(), static_cast<int32_t>(r.size()));
        ASSERT_EQ(a, b);
        ASSERT_EQ(amO.size(), 3);
        ASSERT_EQ(amN.size(), 3);
        ASSERT_TRUE(sameRun(amO.flatten(), amN.flatten()));
    }
}

// Door 4 — findOrEmplaceWork(wi, vi) is the heap-free twin of
// findOrEmplace(encodeWorkInstruction(wi, vi)). On fresh interners viA/viB: same
// returned index, byte-identical flatten, and equal id tables — through BOTH the
// MISS (first insert) and the mint-on-HIT (the door re-mints ALL fields
// unconditionally before the id compare, matching encodeWorkInstruction).
TEST(memory, find_or_emplace_work_matches_encode_find) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState dA = gl::DirtyState::Clean, dB = gl::DirtyState::Clean;
    gl::ColdStringTable tA(&lb, &dA), tB(&lb, &dB);
    gl::ValueInterner viA, viB;
    viA.bind(&tA); viB.bind(&tB);

    // A two-entity working instruction (multi-element + empty-element) with a
    // marked goal — the signature "(zeta…)" lex-sorts after the "(in…)" elements
    // so a mint reorder would diverge the fresh id tables.
    gl::WorkInstruction wi(&lb);
    {
        const std::string e00 = "(in[a,N])", e01 = "(in[b,N])";
        const int32_t m = wi.elemMark();
        wi.addElement(gl::StrSpan(e00));
        wi.addElement(gl::StrSpan(e01));
        wi.commitEntity(gl::StrSpan("existence", 9), gl::StrSpan("(zeta[a,b])", 11),
                        gl::StrSpan("N", 1), 2, m);
    }
    {
        const int32_t m = wi.elemMark();
        wi.commitEntity(gl::StrSpan("and", 3), gl::StrSpan("(s1)", 4),
                        gl::StrSpan("", 0), 0, m);
    }
    wi.setMarkedGoal(gl::StrSpan("(g0)", 4));

    gl::ArenaIntegrationMap amO(&lb, &viA);
    gl::ArenaIntegrationMap amN(&lb, &viB);

    // MISS: both paths mint all fields and append the first entry.
    const int32_t a = amO.findOrEmplace(gl::encodeWorkInstruction(wi, viA));
    const int32_t b = amN.findOrEmplaceWork(wi, viB);
    ASSERT_EQ(a, b);
    ASSERT_EQ(amO.size(), 1);
    ASSERT_EQ(amN.size(), 1);
    ASSERT_TRUE(sameRun(amO.flatten(), amN.flatten()));
    ASSERT_EQ(viA.table->count(), viB.table->count());
    for (int32_t i = 1; i <= viA.table->count(); ++i)
        ASSERT_TRUE(gl::equalSpans(viA.decodeView(i), viB.decodeView(i)));
    const int32_t countAfterMiss = viB.table->count();

    // HIT (mint-on-HIT): the SAME wi hits the existing entry; both paths re-mint
    // ALL fields (already interned -> no new ids), so the tables stay equal and
    // nothing is appended.
    const int32_t a2 = amO.findOrEmplace(gl::encodeWorkInstruction(wi, viA));
    const int32_t b2 = amN.findOrEmplaceWork(wi, viB);
    ASSERT_EQ(a2, a);
    ASSERT_EQ(b2, b);
    ASSERT_EQ(amO.size(), 1);
    ASSERT_EQ(amN.size(), 1);
    ASSERT_EQ(viB.table->count(), countAfterMiss);   // hit minted no new ids
    ASSERT_TRUE(sameRun(amO.flatten(), amN.flatten()));
    ASSERT_EQ(viA.table->count(), viB.table->count());
    for (int32_t i = 1; i <= viA.table->count(); ++i)
        ASSERT_TRUE(gl::equalSpans(viA.decodeView(i), viB.decodeView(i)));
}

// cleanInstructionWork mirrors the heap cleanInstruction cascade-erase on the
// arena WorkInstruction: drop the signature-matching entity, then for a container
// drop each element no longer referenced by another surviving container.
TEST(memory, clean_instruction_work_matches_heap) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);

    // Shape: "and" EXPR -> [E1, E2]; atomic E1 (orphaned -> erased); atomic E2
    // (shared by OTHER -> kept); "and" OTHER -> [E2] (kept).
    const std::string AND = "and", ATOMIC = "atomic", EMPTY = "";
    const std::string EXPR = "EXPR", E1 = "E1", E2 = "E2", OTHER = "OTHER";
    gl::WorkInstruction src(&lb);
    {
        const int32_t m = src.elemMark();
        src.addElement(gl::StrSpan(E1));
        src.addElement(gl::StrSpan(E2));
        src.commitEntity(gl::StrSpan(AND), gl::StrSpan(EXPR), gl::StrSpan(EMPTY), 0, m);
    }
    { const int32_t m = src.elemMark();
      src.commitEntity(gl::StrSpan(ATOMIC), gl::StrSpan(E1), gl::StrSpan(EMPTY), 0, m); }
    { const int32_t m = src.elemMark();
      src.commitEntity(gl::StrSpan(ATOMIC), gl::StrSpan(E2), gl::StrSpan(EMPTY), 0, m); }
    { const int32_t m = src.elemMark();
      src.addElement(gl::StrSpan(E2));
      src.commitEntity(gl::StrSpan(AND), gl::StrSpan(OTHER), gl::StrSpan(EMPTY), 0, m); }
    const std::string GOAL = "GOAL";
    src.setMarkedGoal(gl::StrSpan(GOAL));

    gl::WorkInstruction dst(&lb);
    gl::cleanInstructionWork(src, dst, EXPR);

    // Survivors in order: atomic E2 (shared, kept), then "and" OTHER -> [E2].
    // Erased: EXPR (the match) and E1 (orphaned, not shared elsewhere).
    ASSERT_EQ(dst.entityCount(), 2);
    ASSERT_TRUE(gl::equalSpans(dst.signature(0), gl::StrSpan(E2)));
    ASSERT_TRUE(gl::equalSpans(dst.category(0), gl::StrSpan(ATOMIC)));
    ASSERT_EQ(dst.elemCount(0), 0);
    ASSERT_TRUE(gl::equalSpans(dst.signature(1), gl::StrSpan(OTHER)));
    ASSERT_TRUE(gl::equalSpans(dst.category(1), gl::StrSpan(AND)));
    ASSERT_EQ(dst.elemCount(1), 1);
    ASSERT_TRUE(gl::equalSpans(dst.elemAt(1, 0), gl::StrSpan(E2)));
    ASSERT_TRUE(gl::equalSpans(dst.markedGoal(), gl::StrSpan(GOAL)));
}

// The StrSpan overload of cleanInstructionWork is byte-identical to the
// std::string forwarder: run both on the SAME source instruction and assert the
// two destinations match entity-for-entity, category / signature / element run
// and marked goal (I-138).
TEST(memory, clean_instruction_work_span_matches_string) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);

    // Same shape as clean_instruction_work_matches_heap.
    const std::string AND = "and", ATOMIC = "atomic", EMPTY = "";
    const std::string EXPR = "EXPR", E1 = "E1", E2 = "E2", OTHER = "OTHER";
    gl::WorkInstruction src(&lb);
    {
        const int32_t m = src.elemMark();
        src.addElement(gl::StrSpan(E1));
        src.addElement(gl::StrSpan(E2));
        src.commitEntity(gl::StrSpan(AND), gl::StrSpan(EXPR), gl::StrSpan(EMPTY), 0, m);
    }
    { const int32_t m = src.elemMark();
      src.commitEntity(gl::StrSpan(ATOMIC), gl::StrSpan(E1), gl::StrSpan(EMPTY), 0, m); }
    { const int32_t m = src.elemMark();
      src.commitEntity(gl::StrSpan(ATOMIC), gl::StrSpan(E2), gl::StrSpan(EMPTY), 0, m); }
    { const int32_t m = src.elemMark();
      src.addElement(gl::StrSpan(E2));
      src.commitEntity(gl::StrSpan(AND), gl::StrSpan(OTHER), gl::StrSpan(EMPTY), 0, m); }
    const std::string GOAL = "GOAL";
    src.setMarkedGoal(gl::StrSpan(GOAL));

    // std::string forwarder into dstStr; StrSpan overload into dstSpan.
    gl::WorkInstruction dstStr(&lb);
    gl::cleanInstructionWork(src, dstStr, EXPR);
    gl::WorkInstruction dstSpan(&lb);
    gl::cleanInstructionWork(src, dstSpan, gl::StrSpan(EXPR));

    // Byte-identical: same entity count and each entity's category / signature /
    // element run + the marked goal.
    ASSERT_EQ(dstStr.entityCount(), dstSpan.entityCount());
    for (int32_t i = 0; i < dstStr.entityCount(); ++i) {
        ASSERT_TRUE(gl::equalSpans(dstStr.category(i), dstSpan.category(i)));
        ASSERT_TRUE(gl::equalSpans(dstStr.signature(i), dstSpan.signature(i)));
        ASSERT_EQ(dstStr.elemCount(i), dstSpan.elemCount(i));
        for (int32_t j = 0; j < dstStr.elemCount(i); ++j) {
            ASSERT_TRUE(gl::equalSpans(dstStr.elemAt(i, j), dstSpan.elemAt(i, j)));
        }
    }
    ASSERT_TRUE(gl::equalSpans(dstStr.markedGoal(), dstSpan.markedGoal()));
    // And the span overload still matches the known-good heap result (survivors
    // E2, OTHER; EXPR + orphaned E1 erased).
    ASSERT_EQ(dstSpan.entityCount(), 2);
}

// intToBeProved lives in the never-deloaded persistent pool (the determinism
// fix, I-108): inserting a goal pins a persistent block and
// the registry stays resident regardless of the MAIN arena's deload state;
// discharge reclaims the block (resetToFresh + releaseAll) and empties it.
TEST(memory, inttobeproved_persistent_pool_lifecycle) {
    gl::Memory m;
    // Compaction scratch on an independent pool (the discharge triggers a
    // reshuffle; production draws this from the never-deloaded LB-body pool).
    gl::GlobalMemoryManager gScr;
    gScr.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena scratch(&gScr);
    // Lazy: no persistent block until the first goal insert.
    ASSERT_EQ(m.persistentArena.blocksHeld(), static_cast<int64_t>(0));
    ASSERT_TRUE(m.persistentArena.resident());

    const int32_t k = gl::packStatementKey(1, gl::NameMap::MAIN_ID);
    m.intToBeProved.insertSorted(k, 0);
    ASSERT_TRUE(m.persistentArena.blocksHeld() > 0);   // a persistent block is pinned
    ASSERT_EQ(m.intToBeProved.count(), static_cast<int32_t>(1));
    // Low 16 bits of the key are the validity id — a main-scope goal (the
    // deactivation survey's exact probe).
    ASSERT_EQ(static_cast<int16_t>(m.intToBeProved.keyAt(1) & 0xFFFF),
              gl::NameMap::MAIN_ID);

    // Discharge reclaims the persistent block and empties the registry.
    m.dischargeStatementContent(scratch);
    ASSERT_TRUE(m.dischargedForever);
    ASSERT_EQ(m.intToBeProved.count(), static_cast<int32_t>(0));
    ASSERT_EQ(m.persistentArena.blocksHeld(), static_cast<int64_t>(0));
    ASSERT_TRUE(m.persistentArena.resident());         // released, not deloaded
}

// EquivalenceClass — default-constructed has empty containers;
// setMembersFromNames + a manual levels entry populate the id-form slots.
TEST(memory, equivalenceclass_construction) {
    gl::EquivalenceClass empty;
    ASSERT_EQ(empty.memberIds.size(), static_cast<std::size_t>(0));
    ASSERT_EQ(empty.intEqualityLevelsMap.size(), static_cast<std::size_t>(0));
    ASSERT_EQ(empty.equalityOriginMap.size(), static_cast<std::size_t>(0));

    gl::Memory m;
    gl::EquivalenceClass cls;
    cls.setMembersFromNames({ "a", "b" }, m.nameMap);
    cls.intEqualityLevelsMap[gl::packEqPairKey(
        m.nameMap.encode("a"), m.nameMap.encode("b"))] = { 0, 1 };
    ASSERT_EQ(cls.memberIds.size(), static_cast<std::size_t>(2));
    ASSERT_EQ(cls.intEqualityLevelsMap.size(), static_cast<std::size_t>(1));
}

// checkForEquivalence — public prover hook used by the disintegration gate.
TEST(prover, check_for_equivalence_signature_available) {
    using CheckForEquivalenceFn = bool (gl::ExpressionAnalyzer::*)(
        const std::string&,
        const std::string&,
        gl::Memory&);
    CheckForEquivalenceFn fn = &gl::ExpressionAnalyzer::checkForEquivalence;
    ASSERT_TRUE(fn != nullptr);
}

// checkForEquivalence — true when a FULLY-DISINTEGRATED equivalence-class
// rewrite of the candidate expression is already a registered statement.
TEST(prover, check_for_equivalence_hits_existing_variant) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::Memory m;

    gl::EquivalenceClass cls;
    cls.setMembersFromNames({ "a", "b" }, m.nameMap);
    m.assignClassesById(gl::NameMap::MAIN_ID, { cls });
    gl::upsertStatementKey(m.intKnownStatements,
        gl::packStatementKey(m.nameMap.encode("(in2[b,7,3])"), gl::NameMap::MAIN_ID),
        /*local=*/true, /*registered=*/true, /*known=*/true,
        /*fullyDisintegrated=*/true);

    ASSERT_TRUE(ea.checkForEquivalence("(in2[a,7,3])", "main", m));
}

// checkForEquivalence — false when the matching variant is present but NOT
// fully disintegrated (e.g. a local twin whose witness was rejected). Such an
// entry must not suppress disintegration: the Gauss-fold cFE regression guard.
TEST(prover, check_for_equivalence_ignores_not_fully_disintegrated_variant) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::Memory m;

    gl::EquivalenceClass cls;
    cls.setMembersFromNames({ "a", "b" }, m.nameMap);
    m.assignClassesById(gl::NameMap::MAIN_ID, { cls });
    // Variant registered, even local, but not fully disintegrated.
    gl::upsertStatementKey(m.intKnownStatements,
        gl::packStatementKey(m.nameMap.encode("(in2[b,7,3])"), gl::NameMap::MAIN_ID),
        /*local=*/true, /*registered=*/true, /*known=*/true,
        /*fullyDisintegrated=*/false);

    ASSERT_FALSE(ea.checkForEquivalence("(in2[a,7,3])", "main", m));
}

// checkForEquivalence — false when the equivalence-class rewrite family has no
// registered hit (even a fully-disintegrated non-matching variant misses).
TEST(prover, check_for_equivalence_misses_unknown_variants) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::Memory m;

    gl::EquivalenceClass cls;
    cls.setMembersFromNames({ "a", "b" }, m.nameMap);
    m.assignClassesById(gl::NameMap::MAIN_ID, { cls });
    gl::upsertStatementKey(m.intKnownStatements,
        gl::packStatementKey(m.nameMap.encode("(in2[c,7,3])"), gl::NameMap::MAIN_ID),
        /*local=*/true, /*registered=*/true, /*known=*/true,
        /*fullyDisintegrated=*/true);

    ASSERT_FALSE(ea.checkForEquivalence("(in2[a,7,3])", "main", m));
}

// checkForEquivalence — the new StrSpan overload is the real body; the
// std::string form delegates to it. Byte-twin (Rule 18): on both a HIT and a
// MISS the two spellings return the identical bool. checkForEquivalence is a
// read-only gate (no NameMap mint), so calling both spellings on the same
// Memory is idempotent.
TEST(prover_span_twins, check_for_equivalence_span_matches_string) {
    const std::string exprStr = "(in2[a,7,3])";
    const std::string valStr = "main";

    {   // HIT: fully-disintegrated variant registered.
        gl::ExpressionAnalyzer ea("Peano");
        gl::Memory m;
        gl::EquivalenceClass cls;
        cls.setMembersFromNames({ "a", "b" }, m.nameMap);
        m.assignClassesById(gl::NameMap::MAIN_ID, { cls });
        gl::upsertStatementKey(m.intKnownStatements,
            gl::packStatementKey(m.nameMap.encode("(in2[b,7,3])"), gl::NameMap::MAIN_ID),
            /*local=*/true, /*registered=*/true, /*known=*/true,
            /*fullyDisintegrated=*/true);
        const bool viaString = ea.checkForEquivalence(exprStr, valStr, m);
        const bool viaSpan = ea.checkForEquivalence(
            gl::StrSpan(exprStr), gl::StrSpan(valStr), m);
        ASSERT_TRUE(viaString);
        ASSERT_EQ(viaString, viaSpan);
    }
    {   // MISS: only a non-matching variant registered.
        gl::ExpressionAnalyzer ea("Peano");
        gl::Memory m;
        gl::EquivalenceClass cls;
        cls.setMembersFromNames({ "a", "b" }, m.nameMap);
        m.assignClassesById(gl::NameMap::MAIN_ID, { cls });
        gl::upsertStatementKey(m.intKnownStatements,
            gl::packStatementKey(m.nameMap.encode("(in2[c,7,3])"), gl::NameMap::MAIN_ID),
            /*local=*/true, /*registered=*/true, /*known=*/true,
            /*fullyDisintegrated=*/true);
        const bool viaString = ea.checkForEquivalence(exprStr, valStr, m);
        const bool viaSpan = ea.checkForEquivalence(
            gl::StrSpan(exprStr), gl::StrSpan(valStr), m);
        ASSERT_FALSE(viaString);
        ASSERT_EQ(viaString, viaSpan);
    }
}

// compiledEntity — the C1 compiled-definition reader is the byte-twin of
// compiledExpressions.find()/.at() with a zero-allocation span probe: pointer
// identity on a hit, nullptr on a defined miss (== find()==end()), an empty
// span maps to the defined miss, and a mid-buffer span probe (live non-NUL
// tail bytes) hits the same entry as the whole-string probe.
TEST(prover, compiled_entity_reader_matches_map_at) {
    gl::ExpressionAnalyzer ea("Peano");

    // Seed synthetic entries directly (the struct is public), so the test does
    // not depend on whether the ctor already populates compiledExpressions.
    ea.compiledExpressions["testOp"] =
        gl::LogicalEntity("atomic", { "e0", "e1" }, "(testOp[u_1,u_2])", 2, "(1)");
    ea.compiledExpressions["biggerOp"] =
        gl::LogicalEntity("implication", { "x" }, "(biggerOp[u_1])", 1, "");

    for (const char* key : { "testOp", "biggerOp" }) {
        const std::string k(key);
        const gl::LogicalEntity* e = ea.compiledEntity(gl::StrSpan(k));
        // Pointer identity — strictly stronger than fieldwise (a future
        // copy-instead-of-alias edit fails here).
        ASSERT_TRUE(e == &ea.compiledExpressions.at(k));
        const gl::LogicalEntity& ref = ea.compiledExpressions.at(k);
        ASSERT_EQ(e->category, ref.category);
        ASSERT_EQ(e->signature, ref.signature);
        ASSERT_EQ(e->elements, ref.elements);
        ASSERT_EQ(e->arity, ref.arity);
        ASSERT_EQ(e->definedSet, ref.definedSet);
    }

    // Miss vs find()==end().
    const std::string absent("no_such_compiled_xyz");
    ASSERT_TRUE(ea.compiledEntity(gl::StrSpan(absent)) == nullptr);
    ASSERT_TRUE(ea.compiledExpressions.find(absent) == ea.compiledExpressions.end());
    // Empty span → defined miss.
    ASSERT_TRUE(ea.compiledEntity(gl::StrSpan()) == nullptr);

    // Mid-buffer span probe (no null-terminator dependence): present key
    // "testOp" embedded in a larger buffer with live non-NUL tail bytes.
    const char buf[] = "testOpXXXX";
    const gl::LogicalEntity* mid = ea.compiledEntity(gl::StrSpan(buf, 6));
    ASSERT_TRUE(mid == &ea.compiledExpressions.at("testOp"));
}

// coreConfig — the C1 compiled-definition reader is the byte-twin of
// coreExpressionMap.find()/.at() with a zero-allocation span probe. The map is
// ctor-populated for "Peano"; the test picks a real present key rather than
// seeding one.
TEST(prover, core_config_reader_matches_map_at) {
    gl::ExpressionAnalyzer ea("Peano");
    // Peano's config populates coreExpressionMap (in3 etc.).
    ASSERT_TRUE(!ea.coreExpressionMap.empty());

    const std::string presentKey = ea.coreExpressionMap.begin()->first;
    const ce::CoreExpressionConfig* c = ea.coreConfig(gl::StrSpan(presentKey));
    // Pointer identity — strictly stronger than fieldwise.
    ASSERT_TRUE(c == &ea.coreExpressionMap.at(presentKey));
    const ce::CoreExpressionConfig& ref = ea.coreExpressionMap.at(presentKey);
    ASSERT_EQ(c->arity, ref.arity);
    ASSERT_EQ(c->signature, ref.signature);
    ASSERT_EQ(c->inputIndices, ref.inputIndices);
    ASSERT_EQ(c->outputIndices, ref.outputIndices);
    ASSERT_TRUE(c->definitionSets == ref.definitionSets);

    // Miss vs find()==end().
    const std::string absent("no_such_core_xyz");
    ASSERT_TRUE(ea.coreConfig(gl::StrSpan(absent)) == nullptr);
    ASSERT_TRUE(ea.coreExpressionMap.find(absent) == ea.coreExpressionMap.end());
    // Empty span → defined miss.
    ASSERT_TRUE(ea.coreConfig(gl::StrSpan()) == nullptr);

    // Mid-buffer span probe: the present key's bytes embedded in a larger
    // buffer with live tail bytes must hit the same entry.
    std::vector<char> buf(presentKey.begin(), presentKey.end());
    buf.push_back('Z');
    buf.push_back('Z');
    const ce::CoreExpressionConfig* mid =
        ea.coreConfig(gl::StrSpan(buf.data(), static_cast<int32_t>(presentKey.size())));
    ASSERT_TRUE(mid == &ea.coreExpressionMap.at(presentKey));
}

// countPatternOccurrences (S6c item 6a) — the regex-free span form counts the
// it_\d+_lev_\d+_\d+ lexemes skipping recursion products, byte-identical to the
// verbatim former std::sregex_iterator body (reimplemented in-test as the
// oracle; std::regex lives in the TEST only).
TEST(prover, count_pattern_occurrences_matches_regex_oracle) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::Memory m;
    // Two it_*_lev_*_* names; one is a recursion product (must be skipped).
    const int16_t idProd = m.nameMap.encode("it_1_lev_2_3");
    (void)m.nameMap.encode("it_4_lev_5_6");
    m.overallHashMemory.productsOfRecursionIds.mint(idProd);

    static const std::regex kPat(R"(it_\d+_lev_\d+_\d+)");
    const auto oracle = [&](const std::string& s) -> int {
        int counter = 0;
        for (std::sregex_iterator i(s.begin(), s.end(), kPat), e; i != e; ++i) {
            const std::string match = i->str();
            const int16_t matchId = m.nameMap.lookup(match);
            if (matchId == 0
                || !m.overallHashMemory.productsOfRecursionIds.contains(matchId)) {
                ++counter;
            }
        }
        return counter;
    };
    const std::string cases[] = {
        "(=[x,zero])",                     // no match -> 0
        "it_4_lev_5_6",                    // one counted (not a product)
        "it_1_lev_2_3",                    // one skipped (product)
        "(f[it_4_lev_5_6,it_4_lev_5_6])",  // duplicates counted per occurrence
        "it_4_lev_5_6it_7_lev_8_9",        // back-to-back (resume at match end)
        "zzit_4_lev_5_6",                  // mid-buffer token fidelity
    };
    for (const std::string& s : cases) {
        ASSERT_EQ(ea.countPatternOccurrences(gl::StrSpan(s),
                      m.overallHashMemory, m.nameMap),
                  oracle(s));
    }
    // Mid-buffer slice with live tail bytes (no null-terminator dependence).
    const std::string buf = "QQit_4_lev_5_6ZZ";
    ASSERT_EQ(ea.countPatternOccurrences(gl::StrSpan(buf.data() + 2, 12),
                  m.overallHashMemory, m.nameMap),
              oracle(std::string(buf.data() + 2, 12)));
}

// extractMaxIterationNumber (S6c item 6b) — the regex-free span form returns the
// max first-capture of it_(\d+)_lev_\d+_ (or -1), byte-identical to the verbatim
// former std::sregex_iterator + atoi body on sane inputs (the over-INT_MAX case
// is impossible by construction and asserts; the oracle uses small values only).
TEST(prover, extract_max_iteration_number_matches_regex_oracle) {
    gl::ExpressionAnalyzer ea("Peano");
    static const std::regex kPat(R"(it_(\d+)_lev_\d+_)");
    const auto oracle = [&](const std::string& s) -> int {
        int maxVal = -1;
        for (std::sregex_iterator i(s.begin(), s.end(), kPat), e; i != e; ++i) {
            const int v = std::atoi(i->str(1).c_str());
            if (v > maxVal) maxVal = v;
        }
        return maxVal;
    };
    const std::string cases[] = {
        "(=[x])",                     // no match -> -1
        "it_3_lev_4_5",               // 3
        "it_7_lev_1_it_2_lev_9_",     // max(7, 2) = 7
        "it_2_lev_1_it_10_lev_1_",    // max(2, 10) = 10
        "it_1_lev",                   // no match -> -1
        "zzit_5_lev_6_",              // mid-buffer -> 5
    };
    for (const std::string& s : cases) {
        ASSERT_EQ(ea.extractMaxIterationNumber(gl::StrSpan(s)), oracle(s));
    }
    const std::string buf = "QQit_8_lev_9_XX";
    ASSERT_EQ(ea.extractMaxIterationNumber(gl::StrSpan(buf.data() + 2, 11)),
              oracle(std::string(buf.data() + 2, 11)));
}

// isProved (S6c item 7) — the regex-free span form is proved iff NEITHER
// it_\d+_lev_\d+_ NOR c\d+ occurs, byte-identical to the two former
// std::regex_search tests (oracle in-test; std::regex lives in the TEST only).
TEST(prover, is_proved_matches_regex_oracle) {
    gl::ExpressionAnalyzer ea("Peano");
    static const std::regex kP1(R"(it_\d+_lev_\d+_)");
    static const std::regex kP2(R"(c\d+)");
    const auto oracle = [&](const std::string& s) -> bool {
        return !std::regex_search(s, kP1) && !std::regex_search(s, kP2);
    };
    const std::string cases[] = {
        "(=[a,b])",             // proved
        "it_1_lev_2_",          // has it_ -> not proved
        "c5",                   // has c-digit -> not proved
        "abc7",                 // has c7 -> not proved
        "c",                    // 'c' with no digit -> proved
        "(f[it_3_lev_4_5])",    // has it_ -> not proved
        "c_1",                  // c not followed by a digit -> proved
    };
    for (const std::string& s : cases) {
        ASSERT_EQ(ea.isProved(gl::StrSpan(s)), oracle(s));
    }
}

// extractRemainingArgs (S6c items 2+3) — the two INDEPENDENT lexical passes
// produce the same std::set as the verbatim former two-regex body (reimplemented
// in-test as the oracle; std::regex lives in the TEST only). Covers the
// "[a>[b]c]" divergence trap (the passes' groups differ on nested brackets),
// unterminated brackets, empty groups, a token equal to "u_" alone
// (substr(2) -> empty), and back-to-back groups.
TEST(prover, extract_remaining_args_matches_regex_oracle) {
    gl::ExpressionAnalyzer ea("Peano");
    static const std::regex re1(R"(>\[([^\]]*)\])");
    static const std::regex re2(R"(\[([^\]]*)\])");
    const auto oracle = [&](const std::string& str) -> std::set<std::string> {
        std::set<std::string> out;
        // pass 2 alone determines the set (pass 1 is assert-only; inputs are
        // assert-clean).
        for (std::sregex_iterator it(str.begin(), str.end(), re2), end; it != end; ++it) {
            const std::size_t lpos = static_cast<std::size_t>(it->position());
            if (lpos > 0 && str[lpos - 1] == '>') continue;
            const std::string inside = (*it)[1].str();
            std::size_t start = 0;
            while (start <= inside.size()) {
                std::size_t pos = inside.find(',', start);
                const std::string token = ce::trimCopy(inside.substr(
                    start, (pos == std::string::npos ? inside.size() : pos) - start));
                if (token.rfind("u_", 0) == 0) out.insert(token.substr(2));
                if (pos == std::string::npos) break;
                start = pos + 1;
            }
        }
        return out;
    };
    const std::string cases[] = {
        "(f[u_1,u_2])",         // {"1","2"}
        "(g[u_5])",             // {"5"}
        ">[a](f[u_1])",         // >[a] block clean; [u_1] not preceded by '>' -> {"1"}
        "[a>[b]c]",             // divergence trap (independent passes) -> {}
        "[abc",                 // unterminated [ -> {}
        ">[abc",                // unterminated >[ -> {}
        "[]",                   // empty group -> {}
        ">[]",                  // empty >[] group -> {}
        "(f[a,u_9])",           // u_ at last position -> {"9"}
        "(f[u_])",              // token == "u_" -> substr(2) empty string inserted
        "(f[u_1])(g[u_2])",     // back-to-back groups -> {"1","2"}
    };
    for (const std::string& s : cases) {
        gl::StrSpan out[gl::ExecutionParameters::MAX_KEY_SLOTS];
        const int32_t n = ea.extractRemainingArgs(gl::StrSpan(s), out,
            gl::ExecutionParameters::MAX_KEY_SLOTS);
        // 0% heap span output; the set is built here for the oracle comparison,
        // exactly as the production caller builds it for addToHashMemory.
        std::set<std::string> got;
        for (int32_t i = 0; i < n; ++i)
            got.emplace(out[i].ptr, static_cast<std::size_t>(out[i].len));
        ASSERT_TRUE(got == oracle(s));
    }
}

// findAllUArgs (statified span form) — collects the DISTINCT u_-stripped args
// across a signature + its element spans, deduped in place, byte-identical set
// to the former std::set<std::string>. Covers cross-expression dedup and the
// empty-elements (signature-only) case.
TEST(prover, find_all_u_args_span_distinct) {
    gl::ExpressionAnalyzer ea("Peano");
    const std::string sig = "(f[u_1,u_2,a])";
    const std::string el0 = "(g[u_2,u_3])";   // u_2 repeats sig <-> el0
    const std::string el1 = "(h[b,u_1])";     // u_1 repeats sig <-> el1
    gl::StrSpan elems[2] = { gl::StrSpan(el0), gl::StrSpan(el1) };
    gl::StrSpan out[gl::ExecutionParameters::MAX_KEY_SLOTS];
    const int32_t n = ea.findAllUArgs(gl::StrSpan(sig), elems, 2, out,
        gl::ExecutionParameters::MAX_KEY_SLOTS);
    std::set<std::string> got;
    for (int32_t i = 0; i < n; ++i)
        got.emplace(out[i].ptr, static_cast<std::size_t>(out[i].len));
    ASSERT_TRUE((got == std::set<std::string>{"1", "2", "3"}));
    ASSERT_EQ(n, 3);   // distinct: u_1, u_2, u_3 (duplicates removed in place)

    // Signature only (no elements) — nullptr run is never dereferenced.
    const std::string sig2 = "(f[u_9,c])";
    const int32_t n2 = ea.findAllUArgs(gl::StrSpan(sig2), nullptr, 0, out,
        gl::ExecutionParameters::MAX_KEY_SLOTS);
    ASSERT_EQ(n2, 1);
    ASSERT_TRUE(gl::equalSpans(out[0], gl::StrSpan("9", 1)));
}

// extractSubstringsForAuxy (S6c item 5) — the lexical twin produces the same
// std::vector (order-sensitive) as the verbatim former std::sregex_iterator body
// over \(([^>(\[]+\[[^\]]*\])\) (oracle in-test; std::regex lives in the TEST
// only). Covers the offset-1 nested match, the excluded '>' run, unterminated
// bracket, empty bracket, back-to-back, trailing paren, and ']'-not-followed-by-')'.
TEST(prover, extract_substrings_for_auxy_matches_regex_oracle) {
    gl::ExpressionAnalyzer ea("Peano");
    static const std::regex re(R"(\(([^>(\[]+\[[^\]]*\])\))");
    const auto oracle = [&](const std::string& e) -> std::vector<std::string> {
        std::vector<std::string> out;
        for (std::sregex_iterator it(e.begin(), e.end(), re), end; it != end; ++it)
            out.push_back(it->str(0));
        return out;
    };
    const std::string cases[] = {
        "(f[x])",          // ["(f[x])"]
        "((a[b]))",        // nested: match starts at 1 -> ["(a[b])"]
        "(>x[y])",         // run starts with '>' -> excluded -> []
        "(f[a)",           // unterminated bracket -> []
        "(f[])",           // empty bracket content -> ["(f[])"]
        "(f)",             // no bracket -> []
        "(f[x])(g[y])",    // back-to-back -> two
        "(f[x]))",         // trailing paren -> ["(f[x])"]
        "(f[x]y)",         // ']' followed by non-')' -> []
    };
    for (const std::string& s : cases) {
        ASSERT_TRUE(ea.extractSubstringsForAuxy(s) == oracle(s));
    }
}

// chooseCanonical — picks one representative for a whole equivalence class,
// used by the admission/rejection key hooks. Priority: a normal name (not
// int_/it_; repl_ counts as normal) > int_ > it_; lexicographically smallest
// within a tier.
TEST(prover, choose_canonical_priority) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::Memory m;

    // Normal name beats int_ and it_; repl_ is a normal name and, being the
    // lex-smallest normal here, is chosen.
    gl::EquivalenceClass c1;
    c1.setMembersFromNames({ "it_0_lev_1_2", "int_lev_3_4", "repl_7", "zebra" }, m.nameMap);
    ASSERT_EQ(ea.chooseCanonical(c1, m, "main"), std::string("repl_7"));

    // No normal name: int_ beats it_, lex-smallest int_ wins.
    gl::EquivalenceClass c2;
    c2.setMembersFromNames({ "it_0_lev_1_2", "int_lev_3_4", "int_lev_1_1" }, m.nameMap);
    ASSERT_EQ(ea.chooseCanonical(c2, m, "main"), std::string("int_lev_1_1"));

    // it_ only: lex-smallest it_ wins.
    gl::EquivalenceClass c3;
    c3.setMembersFromNames({ "it_5_lev_1_2", "it_0_lev_1_2" }, m.nameMap);
    ASSERT_EQ(ea.chooseCanonical(c3, m, "main"), std::string("it_0_lev_1_2"));
}

// chooseCanonical — the defined "nothing to canonicalise" result: an empty
// class or an all-weak class yields the empty string; weak members are
// excluded, so a lone strong member still wins.
TEST(prover, choose_canonical_empty_and_weak) {
    gl::ExpressionAnalyzer ea("Peano");

    // Empty class -> empty string.
    gl::Memory m;
    gl::EquivalenceClass empty;
    ASSERT_TRUE(ea.chooseCanonical(empty, m, "main").empty());

    // All members weak at "main" -> empty string.
    gl::Memory mWeak;
    mWeak.intWeakVariables.mint(gl::packStatementKey(
        mWeak.nameMap.encode("int_lev_3_4"), gl::NameMap::MAIN_ID));
    mWeak.intWeakVariables.mint(gl::packStatementKey(
        mWeak.nameMap.encode("it_0_lev_1_2"), gl::NameMap::MAIN_ID));
    gl::EquivalenceClass cWeak;
    cWeak.setMembersFromNames({ "int_lev_3_4", "it_0_lev_1_2" }, mWeak.nameMap);
    ASSERT_TRUE(ea.chooseCanonical(cWeak, mWeak, "main").empty());

    // Only the int_ member is weak -> the strong it_ member is the pick.
    gl::Memory mPartial;
    mPartial.intWeakVariables.mint(gl::packStatementKey(
        mPartial.nameMap.encode("int_lev_3_4"), gl::NameMap::MAIN_ID));
    gl::EquivalenceClass cPartial;
    cPartial.setMembersFromNames({ "int_lev_3_4", "it_0_lev_1_2" }, mPartial.nameMap);
    ASSERT_EQ(ea.chooseCanonical(cPartial, mPartial, "main"),
              std::string("it_0_lev_1_2"));
}

// applyEquivalenceClassToAdmissionMap — drops the changed key and inserts the
// single canonical key (re-key, not additive), moving the status onto it. The
// rejectedMap is empty, so revisitRejected2 is a no-op and K' survives.
TEST(prover, apply_equi_admission_drops_and_rekeys) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::Memory m;

    const int32_t oldPk = gl::mintTemplateKey(m.templateInterner, m.nameMap,
        "(in2[it_5_lev_1_2,marker,3])", "main");
    gl::AdmissionMapValue value;
    value.key = encodeValueVectorOracle(
        std::vector<std::string>{ "(in2[it_5_lev_1_2,7,3])" }, m.valueInterner);
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);
    gl::insertAdmissionValue(m.overallHashMemory.admissionMap, oldPk,
                             value, m.valueInterner, tArena);
    m.overallHashMemory.admissionStatusMap.upsert(oldPk,
                                                  static_cast<uint8_t>(1));
    m.overallHashMemory.varsInAdmissionMapKeys.mint(
        m.templateInterner.encode("it_5_lev_1_2"));
    m.overallHashMemory.varsInAdmissionMapKeys.mint(
        m.templateInterner.encode("3"));

    // Class {it_0_lev_1_2, it_5_lev_1_2}: canon = it_0_lev_1_2 (lex-smallest it_).
    gl::EquivalenceClass cls;
    cls.setMembersFromNames({ "it_0_lev_1_2", "it_5_lev_1_2" }, m.nameMap);

    ea.applyEquivalenceClassToAdmissionMap(cls, m, gl::StrSpan("main", 4));

    // Old non-canonical key dropped from both maps.
    ASSERT_TRUE(m.overallHashMemory.admissionMap.lookup(oldPk) == 0);
    ASSERT_TRUE(m.overallHashMemory.admissionStatusMap.find(oldPk) == nullptr);

    // Canonical key inserted; status moved (true); value element rewritten.
    int32_t newPk = 0;
    ASSERT_TRUE(gl::lookupTemplateKey(m.templateInterner, m.nameMap,
        "(in2[it_0_lev_1_2,marker,3])", "main", newPk));
    const int32_t newAdmId = m.overallHashMemory.admissionMap.lookup(newPk);
    ASSERT_TRUE(newAdmId != 0);
    const uint8_t* itStatus = m.overallHashMemory.admissionStatusMap.find(newPk);
    ASSERT_TRUE(itStatus != nullptr);
    ASSERT_TRUE(*itStatus != 0);
    bool foundRewritten = false;
    const gl::AdmissionValueSet newAdmVals = gl::admissionRecordsAt(
        m.overallHashMemory.admissionMap, newPk, m.valueInterner);
    for (const gl::AdmissionMapValue& v : newAdmVals) {
        for (const int32_t elId : v.key) {
            if (m.valueInterner.decode(elId) == "(in2[it_0_lev_1_2,7,3])") foundRewritten = true;
        }
    }
    ASSERT_TRUE(foundRewritten);
}

// applyEquivalenceClassToRejectedMap — drops the changed key and mails the
// rewritten compound onto sameIterationInternalMail (drop + mail, never a
// direct re-key into rejectedMap).
TEST(prover, apply_equi_rejected_drops_and_mails) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::Memory m;

    const int32_t oldPk = gl::mintTemplateKey(m.templateInterner, m.nameMap,
        "(in2[it_5_lev_1_2,marker,3])", "main");
    gl::RejectedMapValue rv;
    rv.expression = m.valueInterner.encode("(in3[it_5_lev_1_2,7,3,plus])");
    rv.renamedExpression = m.valueInterner.encode("(in3[it_5_lev_1_2,7,3,plus])");
    rv.concreteConstituent = m.valueInterner.encode("(in3[it_5_lev_1_2,7,3,plus])");
    rv.levels = std::set<int>{ 0 };
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);
    gl::insertRejectedValue(m.overallHashMemory.rejectedMap, oldPk,
                            rv, m.valueInterner, tArena);

    gl::EquivalenceClass cls;
    cls.setMembersFromNames({ "it_0_lev_1_2", "it_5_lev_1_2" }, m.nameMap);

    ea.applyEquivalenceClassToRejectedMap(cls, m, gl::StrSpan("main", 4));

    // Old key dropped (never re-keyed directly into rejectedMap).
    ASSERT_TRUE(m.overallHashMemory.rejectedMap.lookup(oldPk) == 0);

    // Rewritten compound mailed onto sameIterationInternalMail at "main".
    bool foundMail = false;
    const gl::Mail mailSnap = gl::makeHeapMail(m.sameIterationInternalMail,
                                               m.nameMap, m.originInterner);
    for (const auto& pr : mailSnap.statements) {
        if (pr.first.original == "(in3[it_0_lev_1_2,7,3,plus])"
            && pr.first.validityName == "main") {
            foundMail = true;
        }
    }
    ASSERT_TRUE(foundMail);
}

// addInternalMailOrigin's D-49 cap-full preference (the new id-form wrapper):
// below cap append-if-absent; at cap a foundation (non-convenience) record
// displaces the first equality-convenience slot; a duplicate or another
// convenience record cannot. Read back through makeHeapMail (id -> string).
TEST(memory, internal_mail_origin_cap_full_preference) {
    gl::Memory m;
    gl::ColdMail& cm = m.sameIterationInternalMail;
    const gl::ExpressionWithValidity k("(h[1])", "main");
    auto line = [](const std::string& tag, const std::string& o,
                   const std::string& v) {
        return gl::OriginLine{ tag, std::vector<gl::ExpressionWithValidity>{
            gl::ExpressionWithValidity(o, v) } };
    };
    // Real OriginTag strings only — addInternalMailOrigin validates the tag via
    // originTagFromString (the closed vocabulary). "disintegration" / "premise"
    // are foundation (non-convenience) tags; "equality1" / "equality2" convenience.
    const int cap = 2;
    gl::addInternalMailOrigin(cm, m.originInterner, k, line("disintegration", "p", "main"), cap);
    gl::addInternalMailOrigin(cm, m.originInterner, k, line("equality1", "q", "main"), cap);
    gl::addInternalMailOrigin(cm, m.originInterner, k, line("disintegration", "p", "main"), cap); // dup
    gl::addInternalMailOrigin(cm, m.originInterner, k, line("premise", "r", "main"), cap);
    gl::addInternalMailOrigin(cm, m.originInterner, k, line("equality2", "s", "main"), cap); // can't displace

    const gl::Mail hm = gl::makeHeapMail(cm, m.nameMap, m.originInterner);
    auto it = hm.exprOriginMap.find(k);
    ASSERT_TRUE(it != hm.exprOriginMap.end());
    ASSERT_EQ(static_cast<int>(it->second.size()), 2);
    ASSERT_TRUE(it->second[0].first == "disintegration");
    ASSERT_TRUE(it->second[1].first == "premise");   // equality1 displaced
}

// insertInternalStatement span door (L4 statement sink): the StrSpan overload
// deposits byte-identically to the ExpressionWithValidity overload — same
// NameMap mints, same ColdMail statement key + levels. Two independent Memories
// (parallel mint order); compared through makeHeapMail (id -> string).
TEST(memory, internal_mail_statement_span_door_twin) {
    gl::Memory mEwv;
    gl::Memory mSpan;
    const std::string expr = "(in3[it_0_lev_1_2,7,3,plus])";
    const std::string valid = "main_boundary_ordis_((=[2,repl_lev_1_0]))";
    const std::set<int> levels{ 0, 1, 2 };

    gl::insertInternalStatement(mEwv.sameIterationInternalMail, mEwv.nameMap,
        gl::ExpressionWithValidity(expr, valid), levels);
    gl::insertInternalStatement(mSpan.sameIterationInternalMail, mSpan.nameMap,
        gl::StrSpan(expr), gl::StrSpan(valid), levels);

    const gl::Mail hEwv = gl::makeHeapMail(mEwv.sameIterationInternalMail,
                                           mEwv.nameMap, mEwv.originInterner);
    const gl::Mail hSpan = gl::makeHeapMail(mSpan.sameIterationInternalMail,
                                            mSpan.nameMap, mSpan.originInterner);

    ASSERT_EQ(static_cast<int>(hEwv.statements.size()), 1);
    ASSERT_EQ(static_cast<int>(hSpan.statements.size()),
              static_cast<int>(hEwv.statements.size()));
    const std::pair<gl::ExpressionWithValidity, std::set<int>>& sE =
        *hEwv.statements.begin();
    const std::pair<gl::ExpressionWithValidity, std::set<int>>& sS =
        *hSpan.statements.begin();
    ASSERT_TRUE(sS.first.original == sE.first.original);
    ASSERT_TRUE(sS.first.validityName == sE.first.validityName);
    ASSERT_TRUE(sS.second == sE.second);
    // Concrete deposited content.
    ASSERT_TRUE(sS.first.original == expr);
    ASSERT_TRUE(sS.first.validityName == valid);
    ASSERT_TRUE((sS.second == std::set<int>{ 0, 1, 2 }));
}

// insertInternalStatement levels-run door (C1): the ascending-run overload
// deposits byte-identically to the std::set<int> span overload — same NameMap
// mints, same IntMailStatementKey bytes per id (id-order facet equality ==
// deload-byte equality). Cases: multi-level {1,3,7}, single {0}, EMPTY run,
// and two statements differing only in levels (multiplicity preserved).
TEST(memory, internal_mail_statement_levels_run_door_twin) {
    gl::Memory mSet;
    gl::Memory mRun;
    const std::string expr = "(in3[it_0_lev_1_2,7,3,plus])";
    const std::string expr2 = "(=[2,repl_lev_1_0])";
    const std::string valid = "main";

    const std::set<int> lv137{ 1, 3, 7 };
    const std::set<int> lv0{ 0 };
    const std::set<int> lvEmpty;
    const std::set<int> lv2{ 2 };

    // Set-overload deposits into mSet.
    gl::insertInternalStatement(mSet.sameIterationInternalMail, mSet.nameMap,
        gl::StrSpan(expr), gl::StrSpan(valid), lv137);
    gl::insertInternalStatement(mSet.sameIterationInternalMail, mSet.nameMap,
        gl::StrSpan(expr2), gl::StrSpan(valid), lv0);
    gl::insertInternalStatement(mSet.sameIterationInternalMail, mSet.nameMap,
        gl::StrSpan(expr2), gl::StrSpan(valid), lvEmpty);
    gl::insertInternalStatement(mSet.sameIterationInternalMail, mSet.nameMap,
        gl::StrSpan(expr2), gl::StrSpan(valid), lv2);  // differs only in levels

    // Run-overload deposits into mRun (each run = the set copied to int[],
    // ascending-unique by std::set iteration; the empty case passes nullptr).
    auto depositRun = [&mRun](const std::string& o, const std::string& v,
                              const std::set<int>& s) {
        int buf[8];
        int32_t n = 0;
        for (const int x : s) buf[n++] = x;
        gl::insertInternalStatement(mRun.sameIterationInternalMail,
            mRun.nameMap, gl::StrSpan(o), gl::StrSpan(v),
            n > 0 ? buf : nullptr, n);
    };
    depositRun(expr, valid, lv137);
    depositRun(expr2, valid, lv0);
    depositRun(expr2, valid, lvEmpty);
    depositRun(expr2, valid, lv2);

    // Facet equality: same count, same key bytes per id.
    auto& a = mSet.sameIterationInternalMail.statements_;
    auto& b = mRun.sameIterationInternalMail.statements_;
    ASSERT_EQ(a.count(), b.count());
    ASSERT_EQ(a.count(), 4);
    for (int32_t i = 1; i <= a.count(); ++i) {
        ASSERT_TRUE(gl::equalSpans(a.inner().decode(i), b.inner().decode(i)));
    }
}

// RejectedValueBlobView (S4 C1): every accessor reads the exact field the
// codec wrote — checked against Codec<RejectedMapValue>::deserialize AND the
// input value, across empty/multi siblings x empty/multi levels x iteration
// 0/>0; copyLevels reproduces the set's ascending contents.
TEST(memory, rejected_value_blob_view_matches_codec) {
    auto check = [](const gl::RejectedMapValue& v) {
        const std::vector<char> blob =
            gl::Codec<gl::RejectedMapValue>::serialize(v);
        const int32_t n = static_cast<int32_t>(blob.size());
        const gl::RejectedValueBlobView view(blob.data(), n);
        const gl::RejectedMapValue back =
            gl::Codec<gl::RejectedMapValue>::deserialize(blob.data(), n);

        ASSERT_EQ(view.renamedExpressionId(), back.renamedExpression);
        ASSERT_EQ(view.renamedExpressionId(), v.renamedExpression);
        ASSERT_EQ(view.expressionId(), back.expression);
        ASSERT_EQ(view.expressionId(), v.expression);
        ASSERT_EQ(view.iteration(), back.iteration);
        ASSERT_EQ(view.iteration(), v.iteration);
        ASSERT_EQ(view.concreteConstituentId(), back.concreteConstituent);
        ASSERT_EQ(view.concreteConstituentId(), v.concreteConstituent);

        ASSERT_EQ(view.siblingCount(),
                  static_cast<int32_t>(back.siblings.size()));
        ASSERT_EQ(view.siblingCount(), static_cast<int32_t>(v.siblings.size()));
        for (int32_t i = 0; i < view.siblingCount(); ++i) {
            ASSERT_EQ(view.siblingId(i),
                      back.siblings[static_cast<std::size_t>(i)]);
            ASSERT_EQ(view.siblingId(i),
                      v.siblings[static_cast<std::size_t>(i)]);
        }

        ASSERT_EQ(view.levelCount(), static_cast<int32_t>(back.levels.size()));
        ASSERT_EQ(view.levelCount(), static_cast<int32_t>(v.levels.size()));
        int lv[16];
        view.copyLevels(lv);
        int32_t k = 0;
        for (const int x : v.levels) {
            ASSERT_EQ(lv[k], x);
            ++k;
        }
    };

    // Empty siblings + empty levels, iteration 0.
    check(gl::RejectedMapValue(11, 22, 0, 33, {}, {}));
    // 3 siblings + levels {0,2,7}, iteration > 0.
    check(gl::RejectedMapValue(5, 6, 3, 7, { 101, 102, 103 },
                               std::set<int>{ 0, 2, 7 }));
    // Empty siblings + one level.
    check(gl::RejectedMapValue(1, 2, 9, 4, {}, std::set<int>{ 5 }));
    // One sibling + empty levels, iteration 0.
    check(gl::RejectedMapValue(8, 9, 0, 10, { 201 }, {}));
}

// RejectedMapValue run-ctor byte-twin: constructing from an ascending-unique
// (const int*, int32_t) run serializes to the SAME Codec bytes as the set
// ctor — the levels MEMBER stays std::set<int> (storage contract), only the
// construction boundary changes. Cases: empty (nullptr, 0), singleton, multi.
TEST(memory, rejected_value_run_ctor_matches_set_ctor) {
    auto twin = [](const std::set<int>& levels,
                   std::vector<int32_t> siblings) {
        int run[8];
        int32_t n = 0;
        for (const int x : levels) run[n++] = x;
        const int* rp = (n > 0) ? run : nullptr;
        const gl::RejectedMapValue viaSet(5, 6, 3, 7, siblings, levels);
        const gl::RejectedMapValue viaRun(5, 6, 3, 7, std::move(siblings),
                                          rp, n);
        const std::vector<char> a =
            gl::Codec<gl::RejectedMapValue>::serialize(viaSet);
        const std::vector<char> b =
            gl::Codec<gl::RejectedMapValue>::serialize(viaRun);
        ASSERT_TRUE(a == b);
    };
    twin({}, {});
    twin({ 5 }, { 101 });
    twin({ 1, 4, 9 }, { 101, 102, 103 });
}

// Canonical-run pin (S4 C1): the cold rejectedMap run is STORED canonical
// under DecodedRejectedValueLess (the insertRejectedValue RMW contract), so
// walking recordsAt in run order IS walking the former RejectedValueSet
// snapshot — and the set's dedup is a no-op on a canonical run. Values are
// inserted deliberately shuffled and crafted so decoded order differs from
// insert order (varied renamedExpression bytes; renamed tie split on
// iteration; full tie through siblings split on levels).
TEST(memory, rejected_run_order_matches_decoded_set_oracle) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::ColdStringTable valTable(&lb, &d);
    gl::ValueInterner vi;
    vi.bind(&valTable);
    gl::TypedColdBlobMap<int32_t, gl::RejectedMapValue> rm(&lb, &d);

    // Mint order deliberately anti-lex so id order != decoded-lex order.
    const int32_t renZZ = vi.encode("zz_ren");
    const int32_t renAA = vi.encode("aa_ren");
    const int32_t renMM = vi.encode("mm_ren");
    const int32_t expr1 = vi.encode("(in3[a,b,c,plus])");
    const int32_t conc1 = vi.encode("(in[b,1])");

    // Decoded order: B("aa",it0) < E("aa",it2,{0,3}) < D("aa",it2,{2})
    //                < C("mm") < A("zz").
    const gl::RejectedMapValue A(renZZ, expr1, 5, conc1, {},
                                 std::set<int>{ 1 });
    const gl::RejectedMapValue B(renAA, expr1, 0, conc1, {}, {});
    const gl::RejectedMapValue C(renMM, expr1, 1, conc1, { conc1 },
                                 std::set<int>{ 4 });
    const gl::RejectedMapValue D(renAA, expr1, 2, conc1, {},
                                 std::set<int>{ 2 });
    const gl::RejectedMapValue E(renAA, expr1, 2, conc1, {},
                                 std::set<int>{ 0, 3 });

    const int32_t pk = 42;
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);
    gl::insertRejectedValue(rm, pk, A, vi, tArena);
    gl::insertRejectedValue(rm, pk, D, vi, tArena);
    gl::insertRejectedValue(rm, pk, C, vi, tArena);
    gl::insertRejectedValue(rm, pk, E, vi, tArena);
    gl::insertRejectedValue(rm, pk, B, vi, tArena);

    const int32_t id = rm.lookup(pk);
    ASSERT_TRUE(id != 0);
    const std::vector<gl::RejectedMapValue> run = rm.recordsAt(id);
    const gl::RejectedValueSet oracle = gl::rejectedRecordsAt(rm, pk, vi);

    ASSERT_EQ(run.size(), oracle.size());
    ASSERT_EQ(static_cast<int>(run.size()), 5);
    std::size_t i = 0;
    for (const gl::RejectedMapValue& o : oracle) {
        const gl::RejectedMapValue& r = run[i];
        ASSERT_EQ(r.renamedExpression, o.renamedExpression);
        ASSERT_EQ(r.expression, o.expression);
        ASSERT_EQ(r.iteration, o.iteration);
        ASSERT_EQ(r.concreteConstituent, o.concreteConstituent);
        ASSERT_TRUE(r.siblings == o.siblings);
        ASSERT_TRUE(r.levels == o.levels);
        ++i;
    }
    // The crafted decoded order held (first = B, last = A).
    ASSERT_EQ(run[0].renamedExpression, renAA);
    ASSERT_EQ(run[0].iteration, 0);
    ASSERT_EQ(run[4].renamedExpression, renZZ);
}

// insertAdmissionIdsBlob (S6d C7): the id-run door deposits byte-identically to
// the value-form insertAdmissionValue. Same pre-minted ids, both doors → two
// standalone maps on ONE interner; shuffled multi-record key with decoded order
// != insert order, a duplicate insert, and empty key/rem runs. Asserted per-key
// runLen + per-record peekRecordBytes memcmp == 0.
TEST(memory, insert_admission_ids_blob_matches_value_form) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::ColdStringTable valTable(&lb, &d);
    gl::ValueInterner vi;
    vi.bind(&valTable);
    gl::TypedColdBlobMap<int32_t, gl::AdmissionMapValue> mapV(&lb, &d);
    gl::TypedColdBlobMap<int32_t, gl::AdmissionMapValue> mapB(&lb, &d);
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);

    // Anti-lex mint so id order != decoded-lex order.
    const int32_t k1 = vi.encode("zz_key");
    const int32_t k2 = vi.encode("aa_key");
    const int32_t r1 = vi.encode("mm_rem");
    const int32_t r2 = vi.encode("bb_rem");
    struct Rec { std::vector<int32_t> key; std::vector<int32_t> rem; int depth; int sec; bool flag; };
    const std::vector<Rec> recs = {
        { {k1, k2}, {r1, r2}, 3, 4, false },
        { {k2}, {}, 1, 2, true },              // empty rem
        { {}, {r2}, 5, 0, false },             // empty key
        { {k1}, {r1}, 7, 8, true },
        { {k1, k2}, {r1, r2}, 3, 4, false },   // duplicate of the first
    };
    const int32_t pk = 42;
    for (const Rec& rc : recs) {
        gl::AdmissionMapValue v(rc.key, rc.rem, rc.depth, rc.sec, rc.flag);
        gl::insertAdmissionValue(mapV, pk, v, vi, tArena);
        gl::insertAdmissionIdsBlob(mapB, pk, rc.depth, rc.sec, rc.flag,
            rc.key.empty() ? nullptr : rc.key.data(), static_cast<int32_t>(rc.key.size()),
            rc.rem.empty() ? nullptr : rc.rem.data(), static_cast<int32_t>(rc.rem.size()),
            vi, tArena);
    }
    const int32_t idV = mapV.lookup(pk);
    const int32_t idB = mapB.lookup(pk);
    ASSERT_TRUE(idV != 0 && idB != 0);
    ASSERT_EQ(mapV.runLen(idV), mapB.runLen(idB));
    for (int32_t j = 0; j < mapV.runLen(idV); ++j) {
        int32_t lV = 0, lB = 0;
        const char* pV = mapV.peekRecordBytes(idV, j, lV, tArena);
        const std::vector<char> bufV(pV, pV + lV);
        const char* pB = mapB.peekRecordBytes(idB, j, lB, tArena);
        ASSERT_EQ(lV, lB);
        ASSERT_TRUE(std::memcmp(bufV.data(), pB, static_cast<size_t>(lV)) == 0);
    }
}

// snapshotAdmissionRun (batch3 row 229): the blob-view snapshot reproduces the
// admissionRecordsAt heap-set snapshot field-for-field, in canonical (stored-run
// == comparator) order. Anti-lex mint so decoded order != insert order; shuffled
// multi-record, empty key/rem runs, a duplicate (deduped by the RMW). A
// never-minted key yields count == 0 (defined miss, not a failure).
TEST(memory, snapshot_admission_run_matches_admissionRecordsAt) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::ColdStringTable valTable(&lb, &d);
    gl::ValueInterner vi;
    vi.bind(&valTable);
    gl::TypedColdBlobMap<int32_t, gl::AdmissionMapValue> m(&lb, &d);
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);

    const int32_t k1 = vi.encode("zz_key");
    const int32_t k2 = vi.encode("aa_key");
    const int32_t r1 = vi.encode("mm_rem");
    const int32_t r2 = vi.encode("bb_rem");
    struct Rec { std::vector<int32_t> key; std::vector<int32_t> rem; int depth; int sec; bool flag; };
    const std::vector<Rec> recs = {
        { {k1, k2}, {r1, r2}, 3, 4, false },
        { {k2}, {}, 1, 2, true },              // empty rem
        { {}, {r2}, 5, 0, false },             // empty key
        { {k1}, {r1}, 7, 8, true },
        { {k1, k2}, {r1, r2}, 3, 4, false },   // duplicate (deduped by RMW)
    };
    const int32_t pk = 42;
    for (const Rec& rc : recs) {
        gl::AdmissionMapValue v(rc.key, rc.rem, rc.depth, rc.sec, rc.flag);
        gl::insertAdmissionValue(m, pk, v, vi, tArena);
    }

    // Oracle: the retained heap-set snapshot in comparator order.
    const gl::AdmissionValueSet oracle = gl::admissionRecordsAt(m, pk, vi);

    const gl::ArenaOffset snapMark = tArena.cursor();
    const gl::AdmissionRunSnapshot snap = gl::snapshotAdmissionRun(m, pk, tArena);
    ASSERT_EQ(snap.count, static_cast<int32_t>(oracle.size()));

    // The stored run is canonical (sorted+deduped under DecodedAdmissionValueLess),
    // so view position i == oracle iteration position i, field for field.
    int32_t i = 0;
    for (const gl::AdmissionMapValue& ov : oracle) {
        const gl::AdmissionValueBlobView& v = snap.views[i];
        ASSERT_EQ(v.depth(), ov.standardMaxAdmissionDepth);
        ASSERT_EQ(v.sec(), ov.standardMaxSecondaryNumber);
        ASSERT_TRUE((v.flagByte() != 0) == ov.flag);
        ASSERT_EQ(v.keyCount(), static_cast<int32_t>(ov.key.size()));
        for (int32_t e = 0; e < v.keyCount(); ++e) ASSERT_EQ(v.keyId(e), ov.key[static_cast<std::size_t>(e)]);
        ASSERT_EQ(v.remCount(), static_cast<int32_t>(ov.remainingArgs.size()));
        for (int32_t r = 0; r < v.remCount(); ++r) ASSERT_EQ(v.remId(r), ov.remainingArgs[static_cast<std::size_t>(r)]);
        ++i;
    }
    tArena.popTo(snapMark);

    // Never-minted key: a defined empty query result.
    const gl::AdmissionRunSnapshot miss = gl::snapshotAdmissionRun(m, 999, tArena);
    ASSERT_EQ(miss.count, 0);
}

// snapshotRejectedRun (row 310): the arena blob-copy snapshot reproduces the
// heap rejectedRecordsAt set field-for-field in canonical run order, and a
// never-minted key is a defined { nullptr, 0 } miss. The literal sibling of
// snapshot_admission_run_matches_admissionRecordsAt.
TEST(memory, snapshot_rejected_run_matches_set) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::ColdStringTable valTable(&lb, &d);
    gl::ValueInterner vi;
    vi.bind(&valTable);
    gl::TypedColdBlobMap<int32_t, gl::RejectedMapValue> m(&lb, &d);
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);

    const int32_t ren1 = vi.encode("zz_ren");
    const int32_t ren2 = vi.encode("aa_ren");
    const int32_t ex1 = vi.encode("(in3[a,b,c,plus])");
    const int32_t ex2 = vi.encode("(in[b,1])");
    const int32_t cc1 = vi.encode("cc_one");
    const int32_t cc2 = vi.encode("cc_two");
    const int32_t s1 = vi.encode("sib_m");
    const int32_t s2 = vi.encode("sib_b");
    struct Rec { int32_t ren; int32_t ex; int it; int32_t cc;
                 std::vector<int32_t> sib; std::set<int> lv; };
    const std::vector<Rec> recs = {
        { ren1, ex1, 3, cc1, { s1, s2 }, { 0, 2, 5 } },
        { ren2, ex2, 1, cc2, {}, {} },                  // empty siblings + levels
        { ren1, ex2, 7, cc1, { s2 }, { 1 } },
        { ren1, ex1, 3, cc1, { s1, s2 }, { 0, 2, 5 } }, // duplicate (RMW dedups)
    };
    const int32_t pk = 42;
    for (const Rec& rc : recs) {
        gl::RejectedMapValue v(rc.ren, rc.ex, rc.it, rc.cc, rc.sib, rc.lv);
        gl::insertRejectedValue(m, pk, v, vi, tArena);
    }

    // Oracle: the retained heap-set snapshot in comparator order.
    const gl::RejectedValueSet oracle = gl::rejectedRecordsAt(m, pk, vi);

    const gl::ArenaOffset snapMark = tArena.cursor();
    const gl::RejectedRunSnapshot snap = gl::snapshotRejectedRun(m, pk, tArena);
    ASSERT_EQ(snap.count, static_cast<int32_t>(oracle.size()));

    int32_t i = 0;
    for (const gl::RejectedMapValue& ov : oracle) {
        const gl::RejectedValueBlobView& v = snap.views[i];
        ASSERT_EQ(v.renamedExpressionId(), ov.renamedExpression);
        ASSERT_EQ(v.expressionId(), ov.expression);
        ASSERT_EQ(v.iteration(), ov.iteration);
        ASSERT_EQ(v.concreteConstituentId(), ov.concreteConstituent);
        ASSERT_EQ(v.siblingCount(), static_cast<int32_t>(ov.siblings.size()));
        for (int32_t e = 0; e < v.siblingCount(); ++e)
            ASSERT_EQ(v.siblingId(e), ov.siblings[static_cast<std::size_t>(e)]);
        ASSERT_EQ(v.levelCount(), static_cast<int32_t>(ov.levels.size()));
        int32_t li = 0;
        for (const int lv : ov.levels) { ASSERT_EQ(v.levelAt(li), lv); ++li; }
        ++i;
    }
    tArena.popTo(snapMark);

    const gl::RejectedRunSnapshot miss = gl::snapshotRejectedRun(m, 999, tArena);
    ASSERT_EQ(miss.count, 0);
}

// snapshotRejectedIntegrationRun (row 309): the integration-side sibling of the
// above — reproduces rejectedIntegrationRecordsAt field-for-field in canonical
// run order, never-minted key is a { nullptr, 0 } miss.
TEST(memory, snapshot_rejected_integration_run_matches_set) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::ColdStringTable valTable(&lb, &d);
    gl::ValueInterner vi;
    vi.bind(&valTable);
    gl::TypedColdBlobMap<int32_t, gl::RejectedMapIntegrationValue> m(&lb, &d);
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);

    const int32_t cc1 = vi.encode("zz_cc");
    const int32_t cc2 = vi.encode("aa_cc");
    const int32_t comp1 = vi.encode("(comp[a,b])");
    const int32_t comp2 = vi.encode("(comp[c,d])");
    const int32_t s1 = vi.encode("sib_m");
    const int32_t s2 = vi.encode("sib_b");
    struct Rec { int32_t cc; std::vector<int32_t> sib; int32_t comp; };
    const std::vector<Rec> recs = {
        { cc1, { s1, s2 }, comp1 },
        { cc2, {}, comp2 },                  // empty siblings
        { cc1, { s2 }, comp2 },
        { cc1, { s1, s2 }, comp1 },          // duplicate (RMW dedups)
    };
    const int32_t pk = 7;
    for (const Rec& rc : recs) {
        gl::RejectedMapIntegrationValue v(rc.cc, rc.sib, rc.comp);
        gl::insertRejectedIntegrationValue(m, pk, v, vi);
    }

    const gl::RejectedIntegrationValueSet oracle =
        gl::rejectedIntegrationRecordsAt(m, pk, vi);

    const gl::ArenaOffset snapMark = tArena.cursor();
    const gl::RejectedIntegrationRunSnapshot snap =
        gl::snapshotRejectedIntegrationRun(m, pk, tArena);
    ASSERT_EQ(snap.count, static_cast<int32_t>(oracle.size()));

    int32_t i = 0;
    for (const gl::RejectedMapIntegrationValue& ov : oracle) {
        const gl::RejectedIntegrationValueBlobView& v = snap.views[i];
        ASSERT_EQ(v.concreteConstituentId(), ov.concreteConstituent);
        ASSERT_EQ(v.compoundExpressionId(), ov.compoundExpression);
        ASSERT_EQ(v.siblingCount(), static_cast<int32_t>(ov.siblings.size()));
        for (int32_t e = 0; e < v.siblingCount(); ++e)
            ASSERT_EQ(v.siblingId(e), ov.siblings[static_cast<std::size_t>(e)]);
        ++i;
    }
    tArena.popTo(snapMark);

    const gl::RejectedIntegrationRunSnapshot miss =
        gl::snapshotRejectedIntegrationRun(m, 999, tArena);
    ASSERT_EQ(miss.count, 0);
}

// serializeMailOriginTo (batch3 row 225): the stack-buffer serializer emits the
// exact bytes Codec<IntMailOrigin>::serialize produces — the oracle — across
// empty, single, and multi-dependency records, including the D-49 policy's
// equality1/equality2 convenience tags.
TEST(memory, serialize_mail_origin_to_matches_codec) {
    const std::vector<gl::IntMailOrigin> cases = {
        { static_cast<uint8_t>(gl::OriginTag::disintegration), {} },
        { static_cast<uint8_t>(gl::OriginTag::equality1), { 42 } },
        { static_cast<uint8_t>(gl::OriginTag::equality2), { -7, 1LL << 40 } },
        { static_cast<uint8_t>(gl::OriginTag::theorem), { 1, 2, 3, 4, 5 } },
    };
    for (const gl::IntMailOrigin& r : cases) {
        const std::vector<char> oracle =
            gl::Codec<gl::IntMailOrigin>::serialize(r);
        char buf[gl::ExecutionParameters::kMaxOriginBlobBytes];
        const int32_t len = gl::serializeMailOriginTo(
            buf, gl::ExecutionParameters::kMaxOriginBlobBytes, r);
        ASSERT_EQ(len, static_cast<int32_t>(oracle.size()));
        ASSERT_TRUE(std::memcmp(buf, oracle.data(),
                                static_cast<std::size_t>(len)) == 0);
    }
}

// Test-local verbatim copy of the heap addMailOriginRecord body (the retired
// decode-into-std::vector RMW) — the Rule-18 oracle the stack-buffer RMW twin
// test compares against.
static void addMailOriginRecordHeapOracle(
    gl::TypedColdBlobMap<int64_t, gl::IntMailOrigin>& col,
    int64_t key, const gl::IntMailOrigin& record, int maxOrigins) {
    const int32_t id = col.lookup(key);
    const std::vector<gl::IntMailOrigin> vec =
        id ? col.recordsAt(id) : std::vector<gl::IntMailOrigin>{};
    if (vec.size() < static_cast<size_t>(maxOrigins)) {
        if (std::find(vec.begin(), vec.end(), record) != vec.end()) return;
        if (id) col.appendRecord(key, record);
        else col.assignRun(key, std::vector<gl::IntMailOrigin>{ record });
        return;
    }
    if (std::find(vec.begin(), vec.end(), record) != vec.end()) return;
    auto isEqualityConvenienceTag = [](uint8_t tag) {
        return tag == static_cast<uint8_t>(gl::OriginTag::equality1)
            || tag == static_cast<uint8_t>(gl::OriginTag::equality2);
    };
    if (!isEqualityConvenienceTag(record.tag)) {
        std::vector<gl::IntMailOrigin> updated = vec;
        for (gl::IntMailOrigin& slot : updated) {
            if (isEqualityConvenienceTag(slot.tag)) {
                slot = record;
                col.assignRun(key, updated);
                return;
            }
        }
    }
}

// addMailOriginRecord (batch3 row 225): the stack-buffer RMW writes runs
// byte-identical to the heap oracle across every D-49 policy branch:
// append-new-key, append-existing, dedup no-op, cap-full convenience-replace,
// cap-full existing-wins (convenience-tag new record), and cap-full with no
// convenience slot left.
TEST(memory, add_mail_origin_record_rmw_matches_heap) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::TypedColdBlobMap<int64_t, gl::IntMailOrigin> mapNew(&lb, &d);
    gl::TypedColdBlobMap<int64_t, gl::IntMailOrigin> mapOracle(&lb, &d);
    const int64_t key = 77;
    const int cap = 3;

    const auto sameRun = [&]() -> bool {
        const int32_t ni = mapNew.lookup(key);
        const int32_t oi = mapOracle.lookup(key);
        if ((ni != 0) != (oi != 0)) return false;
        if (ni == 0) return true;
        if (mapNew.runLen(ni) != mapOracle.runLen(oi)) return false;
        std::vector<char> sn, so;
        for (int32_t j = 0; j < mapNew.runLen(ni); ++j) {
            int32_t ln = 0, lo = 0;
            const char* pn = mapNew.peekRecordBytes(ni, j, ln, sn);
            const std::vector<char> bn(pn, pn + ln);
            const char* po = mapOracle.peekRecordBytes(oi, j, lo, so);
            if (ln != lo) return false;
            if (std::memcmp(bn.data(), po, static_cast<std::size_t>(ln)) != 0)
                return false;
        }
        return true;
    };
    const auto step = [&](const gl::IntMailOrigin& r) {
        gl::addMailOriginRecord(mapNew, key, r, cap);
        addMailOriginRecordHeapOracle(mapOracle, key, r, cap);
        ASSERT_TRUE(sameRun());
    };

    const gl::IntMailOrigin A{ static_cast<uint8_t>(gl::OriginTag::disintegration), { 1, 2 } };
    const gl::IntMailOrigin B{ static_cast<uint8_t>(gl::OriginTag::equality1), { 3 } };
    const gl::IntMailOrigin C{ static_cast<uint8_t>(gl::OriginTag::theorem), {} };
    const gl::IntMailOrigin D{ static_cast<uint8_t>(gl::OriginTag::premise), { 4, 5, 6 } };
    const gl::IntMailOrigin E{ static_cast<uint8_t>(gl::OriginTag::equality2), { 9 } };
    const gl::IntMailOrigin F{ static_cast<uint8_t>(gl::OriginTag::broadcast), { 7 } };

    step(A);   // append on a brand-new key
    step(B);   // append on an existing key (below cap)
    step(A);   // duplicate — the no-op insert
    step(C);   // append reaching the cap (empty-deps blob)
    step(D);   // cap-full: non-convenience displaces the equality1 slot (B)
    step(E);   // cap-full: convenience-tag new record — existing wins, no write
    step(D);   // cap-full duplicate — no-op
    step(F);   // cap-full: no convenience slot left — no write
}

// serializeOriginTo (batch3 row 227): the stack-buffer serializer emits the
// exact bytes Codec<IdOrigin>::serialize produces — the oracle — across empty,
// single, and multi-dependency lines, convenience + foundation tags.
TEST(memory, serialize_origin_to_matches_codec) {
    const std::vector<gl::IdOrigin> cases = {
        gl::IdOrigin(gl::OriginTag::disintegration, {}),
        gl::IdOrigin(gl::OriginTag::equality1, { 42 }),
        gl::IdOrigin(gl::OriginTag::equality2, { -7, 1LL << 40 }),
        gl::IdOrigin(gl::OriginTag::theorem, { 1, 2, 3, 4, 5 }),
    };
    for (const gl::IdOrigin& r : cases) {
        const std::vector<char> oracle = gl::Codec<gl::IdOrigin>::serialize(r);
        char buf[gl::ExecutionParameters::kMaxOriginBlobBytes];
        const int32_t len = gl::serializeOriginTo(
            buf, gl::ExecutionParameters::kMaxOriginBlobBytes, r);
        ASSERT_EQ(len, static_cast<int32_t>(oracle.size()));
        ASSERT_TRUE(std::memcmp(buf, oracle.data(),
                                static_cast<std::size_t>(len)) == 0);
    }
}

// Test-local verbatim copy of the heap cold addOriginId body (the retired
// decode-into-std::vector RMW) — the Rule-18 oracle the stack-buffer RMW twin
// test compares against.
static void addOriginIdColdHeapOracle(
    gl::TypedColdBlobMap<int64_t, gl::IdOrigin>& map, int64_t key,
    const gl::IdOrigin& origin, int maxOrigins) {
    const int32_t id = map.lookup(key);
    const std::vector<gl::IdOrigin> vec =
        id ? map.recordsAt(id) : std::vector<gl::IdOrigin>{};
    if (vec.size() < static_cast<size_t>(maxOrigins)) {
        if (std::find(vec.begin(), vec.end(), origin) != vec.end()) return;
        if (id) map.appendRecord(key, origin);
        else map.assignRun(key, std::vector<gl::IdOrigin>{ origin });
        return;
    }
    if (std::find(vec.begin(), vec.end(), origin) != vec.end()) return;
    auto isEqualityConvenienceTag = [](gl::OriginTag tag) {
        return tag == gl::OriginTag::equality1
            || tag == gl::OriginTag::equality2;
    };
    if (!isEqualityConvenienceTag(origin.first)) {
        std::vector<gl::IdOrigin> updated = vec;
        for (auto& slot : updated) {
            if (isEqualityConvenienceTag(slot.first)) {
                slot = origin;
                map.assignRun(key, updated);
                return;
            }
        }
    }
}

// addOriginId cold overload (batch3 row 227): the stack-buffer RMW writes runs
// byte-identical to the heap oracle across every D-49 policy branch (same
// battery as the addMailOriginRecord twin).
TEST(memory, add_origin_id_cold_rmw_matches_heap) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::TypedColdBlobMap<int64_t, gl::IdOrigin> mapNew(&lb, &d);
    gl::TypedColdBlobMap<int64_t, gl::IdOrigin> mapOracle(&lb, &d);
    const int64_t key = 88;
    const int cap = 3;

    const auto sameRun = [&]() -> bool {
        const int32_t ni = mapNew.lookup(key);
        const int32_t oi = mapOracle.lookup(key);
        if ((ni != 0) != (oi != 0)) return false;
        if (ni == 0) return true;
        if (mapNew.runLen(ni) != mapOracle.runLen(oi)) return false;
        std::vector<char> sn, so;
        for (int32_t j = 0; j < mapNew.runLen(ni); ++j) {
            int32_t ln = 0, lo = 0;
            const char* pn = mapNew.peekRecordBytes(ni, j, ln, sn);
            const std::vector<char> bn(pn, pn + ln);
            const char* po = mapOracle.peekRecordBytes(oi, j, lo, so);
            if (ln != lo) return false;
            if (std::memcmp(bn.data(), po, static_cast<std::size_t>(ln)) != 0)
                return false;
        }
        return true;
    };
    const auto step = [&](const gl::IdOrigin& r) {
        gl::addOriginId(mapNew, key, r, cap);
        addOriginIdColdHeapOracle(mapOracle, key, r, cap);
        ASSERT_TRUE(sameRun());
    };

    const gl::IdOrigin A(gl::OriginTag::disintegration, { 1, 2 });
    const gl::IdOrigin B(gl::OriginTag::equality1, { 3 });
    const gl::IdOrigin C(gl::OriginTag::theorem, {});
    const gl::IdOrigin D(gl::OriginTag::premise, { 4, 5, 6 });
    const gl::IdOrigin E(gl::OriginTag::equality2, { 9 });
    const gl::IdOrigin F(gl::OriginTag::broadcast, { 7 });

    step(A);   // append on a brand-new key
    step(B);   // append on an existing key (below cap)
    step(A);   // duplicate — the no-op insert
    step(C);   // append reaching the cap (empty-deps blob)
    step(D);   // cap-full: non-convenience displaces the equality1 slot (B)
    step(E);   // cap-full: convenience-tag new record — existing wins, no write
    step(D);   // cap-full duplicate — no-op
    step(F);   // cap-full: no convenience slot left — no write
}

// ---- Batch6 c5: origin POD dep-run substrate twins ----

// serializeOriginTo POD overload: raw (tag, deps) fields emit bytes identical to
// the IdOrigin overload across empty / single / multi-dep, convenience tags.
TEST(memory, serialize_origin_to_pod_matches_idorigin) {
    const std::vector<gl::IdOrigin> cases = {
        gl::IdOrigin(gl::OriginTag::disintegration, {}),
        gl::IdOrigin(gl::OriginTag::equality1, { 42 }),
        gl::IdOrigin(gl::OriginTag::equality2, { -7, 1LL << 40 }),
        gl::IdOrigin(gl::OriginTag::theorem, { 1, 2, 3, 4, 5 }),
    };
    for (const gl::IdOrigin& r : cases) {
        char bufA[gl::ExecutionParameters::kMaxOriginBlobBytes];
        char bufB[gl::ExecutionParameters::kMaxOriginBlobBytes];
        const int32_t lenA = gl::serializeOriginTo(
            bufA, gl::ExecutionParameters::kMaxOriginBlobBytes, r);
        const int32_t lenB = gl::serializeOriginTo(
            bufB, gl::ExecutionParameters::kMaxOriginBlobBytes,
            static_cast<uint8_t>(r.first),
            r.second.empty() ? nullptr : r.second.data(),
            static_cast<int32_t>(r.second.size()));
        ASSERT_EQ(lenA, lenB);
        ASSERT_TRUE(std::memcmp(bufA, bufB, static_cast<std::size_t>(lenA)) == 0);
    }
}

// serializeMailOriginTo POD overload: raw fields emit bytes identical to the
// IntMailOrigin overload.
TEST(memory, serialize_mail_origin_to_pod_matches_intmailorigin) {
    const std::vector<gl::IntMailOrigin> cases = {
        { static_cast<uint8_t>(gl::OriginTag::disintegration), {} },
        { static_cast<uint8_t>(gl::OriginTag::equality1), { 42 } },
        { static_cast<uint8_t>(gl::OriginTag::equality2), { -7, 1LL << 40 } },
        { static_cast<uint8_t>(gl::OriginTag::theorem), { 1, 2, 3, 4, 5 } },
    };
    for (const gl::IntMailOrigin& r : cases) {
        char bufA[gl::ExecutionParameters::kMaxOriginBlobBytes];
        char bufB[gl::ExecutionParameters::kMaxOriginBlobBytes];
        const int32_t lenA = gl::serializeMailOriginTo(
            bufA, gl::ExecutionParameters::kMaxOriginBlobBytes, r);
        const int32_t lenB = gl::serializeMailOriginTo(
            bufB, gl::ExecutionParameters::kMaxOriginBlobBytes, r.tag,
            r.deps.empty() ? nullptr : r.deps.data(),
            static_cast<int32_t>(r.deps.size()));
        ASSERT_EQ(lenA, lenB);
        ASSERT_TRUE(std::memcmp(bufA, bufB, static_cast<std::size_t>(lenA)) == 0);
    }
}

// add_origin_id cold POD overload: raw (tag, deps) fields write runs
// byte-identical to the IdOrigin cold overload across the full D-49 battery.
TEST(memory, add_origin_id_cold_pod_matches_idorigin) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::TypedColdBlobMap<int64_t, gl::IdOrigin> mapNew(&lb, &d);
    gl::TypedColdBlobMap<int64_t, gl::IdOrigin> mapOracle(&lb, &d);
    const int64_t key = 88;
    const int cap = 3;

    const auto sameRun = [&]() -> bool {
        const int32_t ni = mapNew.lookup(key);
        const int32_t oi = mapOracle.lookup(key);
        if ((ni != 0) != (oi != 0)) return false;
        if (ni == 0) return true;
        if (mapNew.runLen(ni) != mapOracle.runLen(oi)) return false;
        std::vector<char> sn, so;
        for (int32_t j = 0; j < mapNew.runLen(ni); ++j) {
            int32_t ln = 0, lo = 0;
            const char* pn = mapNew.peekRecordBytes(ni, j, ln, sn);
            const std::vector<char> bn(pn, pn + ln);
            const char* po = mapOracle.peekRecordBytes(oi, j, lo, so);
            if (ln != lo) return false;
            if (std::memcmp(bn.data(), po, static_cast<std::size_t>(ln)) != 0)
                return false;
        }
        return true;
    };
    const auto step = [&](const gl::IdOrigin& r) {
        gl::addOriginId(mapNew, key, static_cast<uint8_t>(r.first),
            r.second.empty() ? nullptr : r.second.data(),
            static_cast<int32_t>(r.second.size()), cap);
        gl::addOriginId(mapOracle, key, r, cap);
        ASSERT_TRUE(sameRun());
    };

    step(gl::IdOrigin(gl::OriginTag::disintegration, { 1, 2 }));
    step(gl::IdOrigin(gl::OriginTag::equality1, { 3 }));
    step(gl::IdOrigin(gl::OriginTag::disintegration, { 1, 2 }));  // dup
    step(gl::IdOrigin(gl::OriginTag::theorem, {}));               // reach cap
    step(gl::IdOrigin(gl::OriginTag::premise, { 4, 5, 6 }));      // displace eq1
    step(gl::IdOrigin(gl::OriginTag::equality2, { 9 }));          // conv -> existing wins
    step(gl::IdOrigin(gl::OriginTag::premise, { 4, 5, 6 }));      // cap-full dup
    step(gl::IdOrigin(gl::OriginTag::broadcast, { 7 }));          // no conv slot left
}

// add_mail_origin_record POD overload: raw fields write runs byte-identical to
// the IntMailOrigin overload across the full D-49 battery.
TEST(memory, add_mail_origin_record_pod_matches_intmailorigin) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::TypedColdBlobMap<int64_t, gl::IntMailOrigin> mapNew(&lb, &d);
    gl::TypedColdBlobMap<int64_t, gl::IntMailOrigin> mapOracle(&lb, &d);
    const int64_t key = 77;
    const int cap = 3;

    const auto sameRun = [&]() -> bool {
        const int32_t ni = mapNew.lookup(key);
        const int32_t oi = mapOracle.lookup(key);
        if ((ni != 0) != (oi != 0)) return false;
        if (ni == 0) return true;
        if (mapNew.runLen(ni) != mapOracle.runLen(oi)) return false;
        std::vector<char> sn, so;
        for (int32_t j = 0; j < mapNew.runLen(ni); ++j) {
            int32_t ln = 0, lo = 0;
            const char* pn = mapNew.peekRecordBytes(ni, j, ln, sn);
            const std::vector<char> bn(pn, pn + ln);
            const char* po = mapOracle.peekRecordBytes(oi, j, lo, so);
            if (ln != lo) return false;
            if (std::memcmp(bn.data(), po, static_cast<std::size_t>(ln)) != 0)
                return false;
        }
        return true;
    };
    const auto step = [&](const gl::IntMailOrigin& r) {
        gl::addMailOriginRecord(mapNew, key, r.tag,
            r.deps.empty() ? nullptr : r.deps.data(),
            static_cast<int32_t>(r.deps.size()), cap);
        gl::addMailOriginRecord(mapOracle, key, r, cap);
        ASSERT_TRUE(sameRun());
    };

    step({ static_cast<uint8_t>(gl::OriginTag::disintegration), { 1, 2 } });
    step({ static_cast<uint8_t>(gl::OriginTag::equality1), { 3 } });
    step({ static_cast<uint8_t>(gl::OriginTag::disintegration), { 1, 2 } }); // dup
    step({ static_cast<uint8_t>(gl::OriginTag::theorem), {} });              // reach cap
    step({ static_cast<uint8_t>(gl::OriginTag::premise), { 4, 5, 6 } });     // displace eq1
    step({ static_cast<uint8_t>(gl::OriginTag::equality2), { 9 } });         // conv -> existing
    step({ static_cast<uint8_t>(gl::OriginTag::premise), { 4, 5, 6 } });     // cap-full dup
    step({ static_cast<uint8_t>(gl::OriginTag::broadcast), { 7 } });         // no conv slot
}

// add_origin_id heap POD overload: raw fields grow the IdOriginMap identically
// to the const IdOrigin& overload across the full D-49 battery.
TEST(memory, add_origin_id_heap_pod_matches_idorigin) {
    gl::IdOriginMap mapNew;
    gl::IdOriginMap mapOracle;
    const int64_t key = 55;
    const int cap = 3;

    const auto sameKey = [&]() -> bool {
        auto itN = mapNew.find(key);
        auto itO = mapOracle.find(key);
        const bool hasN = itN != mapNew.end();
        const bool hasO = itO != mapOracle.end();
        if (hasN != hasO) return false;
        if (!hasN) return true;
        return itN->second == itO->second;
    };
    const auto step = [&](const gl::IdOrigin& r) {
        gl::addOriginId(mapNew, key, static_cast<uint8_t>(r.first),
            r.second.empty() ? nullptr : r.second.data(),
            static_cast<int32_t>(r.second.size()), cap);
        gl::addOriginId(mapOracle, key, r, cap);
        ASSERT_TRUE(sameKey());
    };

    step(gl::IdOrigin(gl::OriginTag::disintegration, { 1, 2 }));
    step(gl::IdOrigin(gl::OriginTag::equality1, { 3 }));
    step(gl::IdOrigin(gl::OriginTag::disintegration, { 1, 2 }));  // dup
    step(gl::IdOrigin(gl::OriginTag::theorem, {}));               // reach cap
    step(gl::IdOrigin(gl::OriginTag::premise, { 4, 5, 6 }));      // displace eq1
    step(gl::IdOrigin(gl::OriginTag::equality2, { 9 }));          // conv -> existing wins
    step(gl::IdOrigin(gl::OriginTag::premise, { 4, 5, 6 }));      // cap-full dup
    step(gl::IdOrigin(gl::OriginTag::broadcast, { 7 }));          // no conv slot left
}

// mintOriginDepsInto: mints the same packed-key run into a caller buffer as
// encodeOriginSpans' out.second (idempotent find-or-mint into the same interner).
TEST(memory, mint_origin_deps_into_matches_encode_origin_spans) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::ColdStringTable oiTable(&lb, &d);
    gl::ValueInterner oi;
    oi.bind(&oiTable);

    const std::string e0 = "(plus[a,b])", v0 = "main";
    const std::string e1 = "(times[c,d])", v1 = "main_boundary_x";
    gl::OriginDep deps[2] = {
        { gl::StrSpan(e0), gl::StrSpan(v0) },
        { gl::StrSpan(e1), gl::StrSpan(v1) },
    };
    // encodeOriginSpans mints first; mintOriginDepsInto then finds the SAME ids.
    const gl::IdOrigin ref =
        gl::encodeOriginSpans(gl::OriginTag::equality2, deps, 2, oi);
    int64_t out[gl::ExecutionParameters::kMaxOriginDeps];
    const int32_t n = gl::mintOriginDepsInto(
        deps, 2, oi, out, gl::ExecutionParameters::kMaxOriginDeps);
    ASSERT_EQ(n, static_cast<int32_t>(ref.second.size()));
    for (int32_t i = 0; i < n; ++i)
        ASSERT_EQ(out[i], ref.second[static_cast<std::size_t>(i)]);

    // Empty-dep case: no ids written.
    int64_t out0[gl::ExecutionParameters::kMaxOriginDeps];
    ASSERT_EQ(gl::mintOriginDepsInto(deps, 0, oi, out0,
        gl::ExecutionParameters::kMaxOriginDeps), 0);
}

// mintOriginDepsFromEWVInto: mints the same packed-key run into a caller buffer
// as encodeOrigin's out.second.
TEST(memory, mint_origin_deps_from_ewv_into_matches_encode_origin) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::ColdStringTable oiTable(&lb, &d);
    gl::ValueInterner oi;
    oi.bind(&oiTable);

    std::vector<gl::ExpressionWithValidity> deps = {
        gl::ExpressionWithValidity("(plus[a,b])", "main"),
        gl::ExpressionWithValidity("(times[c,d])", "main_boundary_x"),
    };
    std::pair<std::string, std::vector<gl::ExpressionWithValidity>> origin = {
        gl::originTagName(gl::OriginTag::equality2), deps };
    const gl::IdOrigin ref = gl::encodeOrigin(origin, oi);
    int64_t out[gl::ExecutionParameters::kMaxOriginDeps];
    const int32_t n = gl::mintOriginDepsFromEWVInto(
        deps, oi, out, gl::ExecutionParameters::kMaxOriginDeps);
    ASSERT_EQ(n, static_cast<int32_t>(ref.second.size()));
    for (int32_t i = 0; i < n; ++i)
        ASSERT_EQ(out[i], ref.second[static_cast<std::size_t>(i)]);
}

// exprOriginRunReplace: the 0% heap serialize+memcmp raw-door run-replace is
// byte-identical to the retired heap `newRun` + typed `assignRun` merge
// (updateEquivalenceClasses' origin sync). Covers class-lines-only, a
// class+body dedup hit, cap-gated body truncation, and a page-straddling body
// blob. exprOriginMap is verifier-walked process documentation (Rule 16), so
// this is byte-insurance on the batch's only novel composition.
TEST(memory, expr_origin_run_replace_matches_heap) {
    using gl::OriginTag;
    using gl::IdOrigin;
    // The retired heap path (the golden oracle).
    const auto runOracle = [](gl::TypedColdBlobMap<int64_t, IdOrigin>& map,
                              int64_t key,
                              const gl::MergeClassAccum::LineView& lines, int cap) {
        std::vector<IdOrigin> newRun;
        newRun.reserve(static_cast<std::size_t>(lines.size()));
        for (int32_t t = 0; t < lines.size(); ++t) {
            std::vector<int64_t> deps;
            deps.reserve(static_cast<std::size_t>(lines.depCount(t)));
            for (int32_t dd = 0; dd < lines.depCount(t); ++dd)
                deps.push_back(lines.dep(t, dd));
            newRun.emplace_back(lines.tag(t), std::move(deps));
        }
        const int32_t bid = map.lookup(key);
        if (bid != 0) {
            const std::vector<IdOrigin> body = map.recordsAt(bid);
            for (const IdOrigin& bl : body) {
                if (newRun.size() >= static_cast<std::size_t>(cap)) break;
                if (std::find(newRun.begin(), newRun.end(), bl) == newRun.end())
                    newRun.push_back(bl);
            }
        }
        map.assignRun(key, newRun);
    };
    // Byte-compare one key's cold run across two maps.
    const auto sameRun = [](gl::TypedColdBlobMap<int64_t, IdOrigin>& a,
                            gl::TypedColdBlobMap<int64_t, IdOrigin>& b,
                            int64_t key) -> bool {
        const int32_t ia = a.lookup(key), ib = b.lookup(key);
        if ((ia != 0) != (ib != 0)) return false;
        if (ia == 0) return true;
        if (a.runLen(ia) != b.runLen(ib)) return false;
        for (int32_t j = 0; j < a.runLen(ia); ++j) {
            std::vector<char> sa, sb;
            int32_t la = 0, lb2 = 0;
            const char* pa = a.peekRecordBytes(ia, j, la, sa);
            const std::vector<char> ba(pa, pa + la);
            const char* pb = b.peekRecordBytes(ib, j, lb2, sb);
            if (la != lb2) return false;
            if (std::memcmp(ba.data(), pb, static_cast<std::size_t>(la)) != 0)
                return false;
        }
        return true;
    };
    struct Line { OriginTag tag; std::vector<int64_t> deps; };
    const int64_t key = 424242;
    // One case: class lines (explicit key/tag/deps in the accumulator), body
    // records under the same key pre-seeded into both maps, optional pool
    // padding to force the body run to straddle a page.
    const auto runCase = [&](int cap, const std::vector<Line>& classLines,
                             const std::vector<IdOrigin>& body, int padRecords) {
        gl::GlobalMemoryManager g;
        g.init(gl::StaticMemoryConfig{ 1 << 21, 1 << 18 });
        gl::LbArena lbCold(&g);       // the cold origin maps
        gl::LbArena lbScratch(&g);    // the accumulator + helper staging (mcArena)
        gl::DirtyState d = gl::DirtyState::Clean;
        gl::MergeClassAccum accum(&lbScratch);
        for (const Line& l : classLines)
            accum.addOriginLine(key, static_cast<uint8_t>(l.tag),
                                l.deps.empty() ? nullptr : l.deps.data(),
                                static_cast<int32_t>(l.deps.size()));
        gl::TypedColdBlobMap<int64_t, IdOrigin> mapNew(&lbCold, &d);
        gl::TypedColdBlobMap<int64_t, IdOrigin> mapOracle(&lbCold, &d);
        if (padRecords > 0) {         // fill ~8144 B so the body crosses 8192
            std::vector<IdOrigin> pad;
            for (int i = 0; i < padRecords; ++i) {
                std::vector<int64_t> deps(63);
                for (int k = 0; k < 63; ++k) deps[k] = 1000 + i * 100 + k;
                pad.emplace_back(OriginTag::premise, std::move(deps));
            }
            mapNew.assignRun(int64_t(999999), pad);
            mapOracle.assignRun(int64_t(999999), pad);
        }
        if (!body.empty()) {
            mapNew.assignRun(key, body);
            mapOracle.assignRun(key, body);
        }
        accum.forEachOriginSorted([&](int64_t k, auto lines) {
            gl::exprOriginRunReplace(mapNew, k, lines, cap, lbScratch);
        });
        accum.forEachOriginSorted([&](int64_t k, auto lines) {
            runOracle(mapOracle, k, lines, cap);
        });
        ASSERT_TRUE(sameRun(mapNew, mapOracle, key));
    };

    // A — class-only, no body (bid == 0).
    runCase(30, { { OriginTag::equality1, {} },
                  { OriginTag::disintegration, { 1, 2 } } }, {}, 0);
    // B — class + body: two byte-equal dups (dropped) + one new (appended).
    runCase(30, { { OriginTag::equality1, {} },
                  { OriginTag::disintegration, { 1, 2 } } },
        { IdOrigin(OriginTag::equality1, {}),
          IdOrigin(OriginTag::disintegration, { 1, 2 }),
          IdOrigin(OriginTag::theorem, { 9 }) }, 0);
    // C — cap-gated: class fills cap=2, every body line skipped.
    runCase(2, { { OriginTag::equality1, {} },
                 { OriginTag::disintegration, { 1, 2 } } },
        { IdOrigin(OriginTag::theorem, { 9 }),
          IdOrigin(OriginTag::premise, { 4, 5 }) }, 0);
    // D — page-straddling body blob (pad ~8144 B so the body crosses 8192).
    {
        std::vector<int64_t> d0(63), d1(63);
        for (int k = 0; k < 63; ++k) { d0[k] = 7000 + k; d1[k] = 8000 + k; }
        runCase(30, { { OriginTag::equality1, {} } },
            { IdOrigin(OriginTag::theorem, d0),
              IdOrigin(OriginTag::premise, d1) }, 16);
    }
}

// ---- Batch6 c9: generateSetPartitions arena CSR twin ----

// Test-local verbatim heap oracle for the retired generateSetPartitions body
// (kept test-local per the ce-twin survivorship convention — prover-only, no
// out-of-tree reader). The production form is now the arena CSR index builder
// ExpressionAnalyzer::generateSetPartitions.
using PartitionOracle = std::vector<std::vector<std::string>>;
static std::vector<PartitionOracle> generateSetPartitionsOracle(
    const std::vector<std::string>& elements, int cap) {
    std::vector<PartitionOracle> result;
    if (elements.empty()) { result.push_back(PartitionOracle()); return result; }
    if (static_cast<int>(elements.size()) > cap) {
        PartitionOracle orig;
        for (auto& e : elements) orig.push_back({ e });
        result.push_back(orig);
        for (std::size_t i = 0; i < elements.size(); ++i)
            for (std::size_t j = i + 1; j < elements.size(); ++j) {
                PartitionOracle p;
                for (std::size_t k = 0; k < elements.size(); ++k) {
                    if (k == j) continue;
                    if (k == i) {
                        std::vector<std::string> merged = { elements[i], elements[j] };
                        std::sort(merged.begin(), merged.end());
                        p.push_back(merged);
                    } else p.push_back({ elements[k] });
                }
                result.push_back(p);
            }
        for (std::size_t i = 0; i < elements.size(); ++i)
            for (std::size_t j = i + 1; j < elements.size(); ++j)
                for (std::size_t l = j + 1; l < elements.size(); ++l) {
                    PartitionOracle p;
                    for (std::size_t k = 0; k < elements.size(); ++k) {
                        if (k == j || k == l) continue;
                        if (k == i) {
                            std::vector<std::string> merged = { elements[i], elements[j], elements[l] };
                            std::sort(merged.begin(), merged.end());
                            p.push_back(merged);
                        } else p.push_back({ elements[k] });
                    }
                    result.push_back(p);
                }
        return result;
    }
    std::string first = elements[0];
    std::vector<std::string> rest(elements.begin() + 1, elements.end());
    std::vector<PartitionOracle> subPartitions = generateSetPartitionsOracle(rest, cap);
    for (auto& sp : subPartitions) {
        { PartitionOracle p = sp; p.push_back({ first }); result.push_back(p); }
        for (std::size_t i = 0; i < sp.size(); ++i) {
            PartitionOracle p = sp;
            p[i].push_back(first);
            std::sort(p[i].begin(), p[i].end());
            result.push_back(p);
        }
    }
    return result;
}

TEST(memory, generate_set_partitions_arena_matches_heap) {
    const auto check = [](const std::vector<std::string>& elems, int cap) {
        const std::vector<PartitionOracle> oracle =
            generateSetPartitionsOracle(elems, cap);
        gl::ScratchArena& a = gl::genScratchArenas().forSlot(
            gl::genScratchArenas().slotCount() - 1);
        gl::DirtyState d = gl::DirtyState::Clean;
        gl::PagedVector<int32_t> mem(&a, &d), cs(&a, &d), ps(&a, &d);
        gl::ExpressionAnalyzer::generateSetPartitions(elems, cap, a, mem, cs, ps);
        const int32_t numParts = ps.size() - 1;
        ASSERT_EQ(numParts, static_cast<int32_t>(oracle.size()));
        for (int32_t p = 0; p < numParts; ++p) {
            const int32_t cB = ps[p], cE = ps[p + 1];
            ASSERT_EQ(cE - cB,
                static_cast<int32_t>(oracle[static_cast<std::size_t>(p)].size()));
            for (int32_t c = cB; c < cE; ++c) {
                const int32_t mB = cs[c], mE = cs[c + 1];
                const std::vector<std::string>& oClass =
                    oracle[static_cast<std::size_t>(p)]
                          [static_cast<std::size_t>(c - cB)];
                ASSERT_EQ(mE - mB, static_cast<int32_t>(oClass.size()));
                for (int32_t m = mB; m < mE; ++m) {
                    ASSERT_TRUE(elems[static_cast<std::size_t>(mem[m])]
                                == oClass[static_cast<std::size_t>(m - mB)]);
                }
            }
        }

        // StrSpan-run overload (the real implementation) — byte-identical CSR to
        // the vector<string> forwarder above (which forwards to it), so the whole
        // partition/class/member sequence matches column-for-column.
        std::vector<gl::StrSpan> spans;
        spans.reserve(elems.size());
        for (const std::string& e : elems) spans.push_back(gl::StrSpan(e));
        gl::DirtyState dS = gl::DirtyState::Clean;
        gl::PagedVector<int32_t> memS(&a, &dS), csS(&a, &dS), psS(&a, &dS);
        gl::ExpressionAnalyzer::generateSetPartitions(
            spans.empty() ? nullptr : spans.data(),
            static_cast<int32_t>(spans.size()), cap, a, memS, csS, psS);
        ASSERT_EQ(memS.size(), mem.size());
        ASSERT_EQ(csS.size(), cs.size());
        ASSERT_EQ(psS.size(), ps.size());
        for (int32_t i = 0; i < mem.size(); ++i) ASSERT_EQ(memS[i], mem[i]);
        for (int32_t i = 0; i < cs.size(); ++i) ASSERT_EQ(csS[i], cs[i]);
        for (int32_t i = 0; i < ps.size(); ++i) ASSERT_EQ(psS[i], ps[i]);
    };
    check({}, 5);
    check({ "a" }, 5);
    check({ "b", "a" }, 5);                          // within-class byte-lex sort
    check({ "a", "b", "c", "d", "e" }, 5);           // cap (recursive branch)
    check({ "a", "b", "c", "d", "e", "f", "g" }, 5); // cap+2 (flat branch)
    check({ "u_2", "x", "u_1" }, 5);                 // u_-member set (rep order)
}

// multiplyImplication span form (row 312) vs the retained heap oracle: the
// emitted copy SEQUENCE (bytes + order) must be byte-identical. Differential on a
// real Peano-configured ExpressionAnalyzer (both forms share the same config, so
// they agree whatever the (1)-typed detection yields); covers the early-out and
// the multiplication paths across single/multi bound-var implications and a
// trivial-equality head.
TEST(memory, multiply_implication_span_matches_heap) {
    gl::ExpressionAnalyzer ea("Peano");
    ea.parameters.allow_multiplication = true;   // exercise the multiply path

    gl::ScratchArena& strArena = gl::scratchArenas().forSlot(
        gl::scratchArenas().slotCount() - 1);
    gl::ScratchArena& genArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);

    const std::vector<std::string> cases = {
        "(>[v0,v1](in[v0])(in2[v0,v1]))",
        "(>[v0,v1,v2](in3[v0,v1,v2])(in[v0]))",
        "(>[u_1,v0,v1](in[u_1])(in2[v0,v1]))",
        "(>[v0](in[v0]))",                            // single bound var -> early out
        "(in[v0,v1])",                                // no implication chain
        "(>[v0,v1](in2[v0,v1])(=[v0,v1]))",           // trivial-equality head shape
    };
    for (const std::string& impl : cases) {
        const std::vector<std::string> oracle = ea.multiplyImplication(impl);

        gl::ScratchScope sScope(strArena);
        gl::DirtyState d = gl::DirtyState::Clean;
        gl::PagedVector<gl::CopyRef> copies(&genArena, &d);
        ea.multiplyImplication(gl::StrSpan(impl), strArena, copies);

        ASSERT_EQ(copies.size(), static_cast<int32_t>(oracle.size()));
        for (int32_t c = 0; c < copies.size(); ++c) {
            const gl::CopyRef cr = copies[c];
            const std::string got(
                reinterpret_cast<const char*>(strArena.resolve(cr.off)),
                static_cast<std::size_t>(cr.len));
            ASSERT_EQ(got, oracle[static_cast<std::size_t>(c)]);
        }
    }
}

// Row-300 write doors (batch 7 c10) — each byte-identical to its owning-form
// oracle. encodeNormKeyInto == Codec<NormKey>::encode.
TEST(memory, encode_norm_key_into_matches_codec) {
    struct C { int16_t ne; std::vector<int16_t> data; };
    const std::vector<C> cases = {
        { 3, {} }, { 1, { 5 } }, { 2, { -1, 0, 7, 32000 } },
        { 8, { 1, 2, 3, 4, 5 } },
    };
    for (const C& c : cases) {
        const gl::NormKey nk{ c.ne, c.data };
        const std::string ref = gl::Codec<gl::NormKey>::encode(nk);
        char buf[gl::ExecutionParameters::kMaxNormKeyBytes];
        const int32_t n = gl::encodeNormKeyInto(c.ne,
            c.data.empty() ? nullptr : c.data.data(),
            static_cast<int32_t>(c.data.size()), buf,
            gl::ExecutionParameters::kMaxNormKeyBytes);
        ASSERT_EQ(n, static_cast<int32_t>(ref.size()));
        ASSERT_TRUE(std::memcmp(buf, ref.data(), ref.size()) == 0);
    }
}

// mergeOwnerRecord raw-key overload == the owning-NormKey overload (empty encList
// so recordUSignature adds no signature — the partitionId merge is the observable).
TEST(memory, merge_owner_record_raw_matches_owning) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::TypedColdBlobMap<gl::NormKey, gl::OwnerSet> mapOwn(&lb, &d);
    gl::TypedColdBlobMap<gl::NormKey, gl::OwnerSet> mapRaw(&lb, &d);
    gl::NameMap nm;   // (void)nm inside — unused

    struct M { int16_t ne; std::vector<int16_t> data; int32_t pid; };
    const std::vector<M> ms = {
        { 3, { 1, 2 }, 100 }, { 3, { 1, 2 }, 200 }, { 2, { 5 }, 100 },
        { 3, { 1, 2 }, 100 },   // duplicate partitionId (set no-op)
    };
    std::vector<gl::NormKey> keys;
    for (const M& m : ms) {
        const gl::NormKey nk{ m.ne, m.data };
        gl::ExpressionAnalyzer::mergeOwnerRecord(mapOwn, nk, m.pid,
            static_cast<const gl::IntEncodedExpr*>(nullptr), 0, nm);
        gl::ExpressionAnalyzer::mergeOwnerRecord(mapRaw, m.ne,
            m.data.empty() ? nullptr : m.data.data(),
            static_cast<int32_t>(m.data.size()), m.pid,
            static_cast<const gl::IntEncodedExpr*>(nullptr), 0, nm);
        keys.push_back(nk);
    }
    for (const gl::NormKey& nk : keys) {
        const int32_t io = mapOwn.lookup(nk);
        const int32_t ir = mapRaw.lookup(nk);
        ASSERT_TRUE(io != 0 && ir != 0);
        ASSERT_TRUE(gl::Codec<gl::OwnerSet>::serialize(mapOwn.recordAt(io, 0))
                 == gl::Codec<gl::OwnerSet>::serialize(mapRaw.recordAt(ir, 0)));
    }
}

// buildUSignatureRunInto == the (slot, argFullId) run + hasUArg flag that the
// IntEncodedExpr recordUSignature builds before it range-constructs the
// signature vector. Covers a tight owner (two premises, cross-expression slot
// accumulation) and a loose owner (no u_ args).
TEST(memory, build_usignature_run_matches_record_usignature) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;

    gl::EncodedExpression e0("(in3[a,b,u_p])", "main");   // u_p at slot 2
    gl::EncodedExpression e1("(in3[c,u_q,f])", "main");   // u_q at slot 4
    gl::IntEncodedExpr enc[2] = { gl::encodeExpression(e0, nm),
                                  gl::encodeExpression(e1, nm) };

    gl::OwnerSet osInt;                              // oracle
    gl::recordUSignature(osInt, enc, static_cast<int16_t>(2));

    std::pair<int16_t, int16_t> run[2 * gl::ExecutionParameters::MAX_ARITY];
    bool hasUArg = false;
    const int32_t n = gl::buildUSignatureRunInto(enc, 2, run,
        2 * gl::ExecutionParameters::MAX_ARITY, hasUArg);

    ASSERT_EQ(hasUArg, !osInt.hasLooseOwner);
    ASSERT_TRUE(hasUArg);
    ASSERT_EQ((int)osInt.uSignatures.size(), 1);
    const std::vector<std::pair<int16_t, int16_t>>& sig = *osInt.uSignatures.begin();
    ASSERT_EQ((int)sig.size(), (int)n);
    for (int32_t i = 0; i < n; ++i) {
        ASSERT_EQ(run[i].first, sig[static_cast<size_t>(i)].first);
        ASSERT_EQ(run[i].second, sig[static_cast<size_t>(i)].second);
    }

    // Loose owner: no u_ args -> hasUArg false, zero pairs, loose flag set.
    gl::EncodedExpression e2("(in2[a,b])", "main");
    gl::IntEncodedExpr encLoose[1] = { gl::encodeExpression(e2, nm) };
    gl::OwnerSet osLoose;
    gl::recordUSignature(osLoose, encLoose, static_cast<int16_t>(1));
    bool hasU2 = true;
    const int32_t n2 = gl::buildUSignatureRunInto(encLoose, 1, run,
        gl::ExecutionParameters::MAX_ARITY, hasU2);
    ASSERT_EQ(n2, 0);
    ASSERT_FALSE(hasU2);
    ASSERT_TRUE(osLoose.hasLooseOwner);
}

// Blob-native mergeOwnerRecord raw overload == the owning IntEncodedExpr
// overload (recordUSignature + Codec<OwnerSet>::serialize) across a SEQUENCE of
// merges exercising: fresh key + u_ sig, a second partitionId (dup sig no-op),
// a second DISTINCT sig, a dup partitionId + dup sig, and a LOOSE owner. Both
// paths accumulate on the SAME key; serialize(recordAt) must match at each step.
TEST(memory, merge_owner_record_raw_blob_matches_owning) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;

    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::TypedColdBlobMap<gl::NormKey, gl::OwnerSet> mapOwn(&lb, &d);
    gl::TypedColdBlobMap<gl::NormKey, gl::OwnerSet> mapRaw(&lb, &d);

    gl::EncodedExpression eA("(in3[a,b,u_p])", "main");  // u_p @ slot 2
    gl::EncodedExpression eB("(in3[c,u_q,f])", "main");  // u_q @ slot 4
    gl::EncodedExpression eC("(in2[a,b])", "main");      // no u_ (loose)
    gl::IntEncodedExpr iA = gl::encodeExpression(eA, nm);
    gl::IntEncodedExpr iB = gl::encodeExpression(eB, nm);
    gl::IntEncodedExpr iC = gl::encodeExpression(eC, nm);

    const int16_t ne = 3;
    const std::vector<int16_t> data = { 1, 2 };
    const gl::NormKey nk{ ne, data };

    struct Step { int32_t pid; std::vector<gl::IntEncodedExpr> enc; };
    const std::vector<Step> steps = {
        { 100, { iA } },   // fresh key + u_ sig {(2,p)}
        { 200, { iA } },   // second partitionId, same sig (dup no-op)
        { 100, { iB } },   // dup partitionId, a SECOND distinct sig {(4,q)}
        { 300, { iA } },   // third partition, sig {(2,p)} (dup no-op)
        { 400, { iC } },   // a LOOSE owner -> hasLooseOwner set, no new sig
    };

    for (const Step& s : steps) {
        gl::ExpressionAnalyzer::mergeOwnerRecord(mapOwn, nk, s.pid,
            s.enc.data(), static_cast<int16_t>(s.enc.size()), nm);
        gl::ExpressionAnalyzer::mergeOwnerRecord(mapRaw, ne, data.data(),
            static_cast<int32_t>(data.size()), s.pid,
            s.enc.data(), static_cast<int16_t>(s.enc.size()), nm);
        const int32_t io = mapOwn.lookup(nk);
        const int32_t ir = mapRaw.lookup(nk);
        ASSERT_TRUE(io != 0 && ir != 0);
        ASSERT_TRUE(gl::Codec<gl::OwnerSet>::serialize(mapOwn.recordAt(io, 0))
                 == gl::Codec<gl::OwnerSet>::serialize(mapRaw.recordAt(ir, 0)));
    }
}

// appendLmvIdsRecord == encodedMap.appendRecord(NormKey, LocalMemoryValue) with
// the marker install's field set (levels empty, justification none, pod false).
TEST(memory, append_lmv_ids_record_matches_value) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::TypedColdBlobMap<gl::NormKey, gl::LocalMemoryValue> mapOwn(&lb, &d);
    gl::TypedColdBlobMap<gl::NormKey, gl::LocalMemoryValue> mapRaw(&lb, &d);
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);

    struct L { int16_t ne; std::vector<int16_t> data; int32_t valueId;
               bool isMarker; std::vector<int32_t> keyIds;
               std::vector<int32_t> remIds; int32_t origImpl; int16_t vid; };
    const std::vector<L> ls = {
        { 3, { 1, 2 }, 42, true, { 7, 8 }, { 9 }, 100, 5 },
        { 3, { 1, 2 }, 43, false, {}, {}, 100, 5 },        // same key, second record
        { 2, { 5 }, 44, true, { 1, 2, 3 }, { 4, 5 }, 200, 1 },
    };
    std::vector<gl::NormKey> keys;
    for (const L& l : ls) {
        gl::LocalMemoryValue lmv;
        lmv.valueId = l.valueId;
        lmv.isMarker = l.isMarker;
        lmv.keyIds = l.keyIds;
        lmv.remainingArgIds = l.remIds;
        lmv.originalImplicationId = l.origImpl;
        lmv.validityId = l.vid;
        const gl::NormKey nk{ l.ne, l.data };
        mapOwn.appendRecord(nk, lmv);
        gl::appendLmvIdsRecord(mapRaw, l.ne,
            l.data.empty() ? nullptr : l.data.data(),
            static_cast<int32_t>(l.data.size()), l.valueId, l.isMarker,
            l.keyIds.empty() ? nullptr : l.keyIds.data(),
            static_cast<int32_t>(l.keyIds.size()),
            l.remIds.empty() ? nullptr : l.remIds.data(),
            static_cast<int32_t>(l.remIds.size()), l.origImpl, l.vid, tArena);
        keys.push_back(nk);
    }
    for (const gl::NormKey& nk : keys) {
        const int32_t io = mapOwn.lookup(nk);
        const int32_t ir = mapRaw.lookup(nk);
        ASSERT_TRUE(io != 0 && ir != 0);
        ASSERT_EQ(mapOwn.runLen(io), mapRaw.runLen(ir));
        for (int32_t j = 0; j < mapOwn.runLen(io); ++j)
            ASSERT_TRUE(gl::Codec<gl::LocalMemoryValue>::serialize(mapOwn.recordAt(io, j))
                     == gl::Codec<gl::LocalMemoryValue>::serialize(mapRaw.recordAt(ir, j)));
    }
}

// appendLmvIdsRecord extended params (levels, levelN, justification, productOf)
// == the owning LocalMemoryValue with those fields set — the HEAD install path
// (addToHashMemory). The default-param marker path is covered above.
TEST(memory, append_lmv_ids_record_head_matches_value) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::TypedColdBlobMap<gl::NormKey, gl::LocalMemoryValue> mapOwn(&lb, &d);
    gl::TypedColdBlobMap<gl::NormKey, gl::LocalMemoryValue> mapRaw(&lb, &d);
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);

    struct H { int16_t ne; std::vector<int16_t> data; int32_t valueId;
               bool isMarker; std::vector<int32_t> keyIds;
               std::vector<int32_t> remIds; int32_t origImpl; int16_t vid;
               std::vector<int> levels; gl::RuleJustification just; bool pod; };
    const std::vector<H> hs = {
        { 3, { 1, 2 }, 42, false, { 7, 8 }, { 9 }, 100, 5,
          { 1, 3, 7 }, gl::RuleJustification::implication, true },
        { 3, { 1, 2 }, 43, false, {}, {}, 100, 5,
          {}, gl::RuleJustification::integration, false },   // same key, empty levels
        { 2, { 5 }, 44, true, { 1, 2, 3 }, { 4, 5 }, 200, 1,
          { 2 }, gl::RuleJustification::implication, false },
    };
    std::vector<gl::NormKey> keys;
    for (const H& h : hs) {
        gl::LocalMemoryValue lmv;
        lmv.valueId = h.valueId;
        lmv.isMarker = h.isMarker;
        lmv.keyIds = h.keyIds;
        lmv.remainingArgIds = h.remIds;
        lmv.originalImplicationId = h.origImpl;
        lmv.validityId = h.vid;
        lmv.levels = std::set<int>(h.levels.begin(), h.levels.end());
        lmv.justification = h.just;
        lmv.productOfDisintegration = h.pod;
        const gl::NormKey nk{ h.ne, h.data };
        mapOwn.appendRecord(nk, lmv);
        gl::appendLmvIdsRecord(mapRaw, h.ne,
            h.data.empty() ? nullptr : h.data.data(),
            static_cast<int32_t>(h.data.size()), h.valueId, h.isMarker,
            h.keyIds.empty() ? nullptr : h.keyIds.data(),
            static_cast<int32_t>(h.keyIds.size()),
            h.remIds.empty() ? nullptr : h.remIds.data(),
            static_cast<int32_t>(h.remIds.size()), h.origImpl, h.vid, tArena,
            h.levels.empty() ? nullptr : h.levels.data(),
            static_cast<int32_t>(h.levels.size()), h.just, h.pod);
        keys.push_back(nk);
    }
    for (const gl::NormKey& nk : keys) {
        const int32_t io = mapOwn.lookup(nk);
        const int32_t ir = mapRaw.lookup(nk);
        ASSERT_TRUE(io != 0 && ir != 0);
        ASSERT_EQ(mapOwn.runLen(io), mapRaw.runLen(ir));
        for (int32_t j = 0; j < mapOwn.runLen(io); ++j)
            ASSERT_TRUE(gl::Codec<gl::LocalMemoryValue>::serialize(mapOwn.recordAt(io, j))
                     == gl::Codec<gl::LocalMemoryValue>::serialize(mapRaw.recordAt(ir, j)));
    }
}

// encodeIdVecKeyInto == Codec<IdVecKey>::encode, byte-for-byte (empty + multi).
TEST(memory, encode_id_vec_key_into_matches_codec) {
    const std::vector<std::vector<int32_t>> cases = {
        {}, { 5 }, { -1, 0, 7, 123456 }, { 1, 2, 3, 4, 5 },
    };
    for (const std::vector<int32_t>& ids : cases) {
        const gl::IdVecKey k{ ids };
        const std::string ref = gl::Codec<gl::IdVecKey>::encode(k);
        char buf[512];
        const int32_t n = gl::encodeIdVecKeyInto(
            ids.empty() ? nullptr : ids.data(),
            static_cast<int32_t>(ids.size()), buf, sizeof(buf));
        ASSERT_EQ(n, static_cast<int32_t>(ref.size()));
        ASSERT_TRUE(std::memcmp(buf, ref.data(), ref.size()) == 0);
    }
}

// viewIdOriginBlob reads Codec<IdOrigin> bytes exactly (0-dep + many-dep).
TEST(memory, view_id_origin_blob_matches_codec) {
    const std::vector<std::vector<int64_t>> depSets = {
        {}, { 42 }, { 1, -5, 0x1122334455667788LL, 9 },
    };
    for (std::size_t t = 0; t < depSets.size(); ++t) {
        const gl::IdOrigin line(static_cast<gl::OriginTag>(t + 1), depSets[t]);
        const std::vector<char> blob = gl::Codec<gl::IdOrigin>::serialize(line);
        const gl::IdOriginBlobView v =
            gl::viewIdOriginBlob(blob.data(), static_cast<int32_t>(blob.size()));
        ASSERT_EQ(static_cast<int>(v.tag),
                  static_cast<int>(static_cast<uint8_t>(line.first)));
        ASSERT_EQ(v.depN, static_cast<int32_t>(line.second.size()));
        for (int32_t i = 0; i < v.depN; ++i)
            ASSERT_TRUE(gl::idOriginDepAt(v, i) == line.second[static_cast<std::size_t>(i)]);
    }
}

// ruleJustificationFromString(StrSpan) == the std::string form on every name.
TEST(memory, rule_justification_from_string_span_matches_string) {
    for (std::size_t i = 0;
         i < static_cast<std::size_t>(gl::RuleJustification::COUNT); ++i) {
        const char* name =
            gl::ruleJustificationName(static_cast<gl::RuleJustification>(i));
        const gl::RuleJustification viaString =
            gl::ruleJustificationFromString(std::string(name));
        const gl::RuleJustification viaSpan =
            gl::ruleJustificationFromString(
                gl::StrSpan(name, static_cast<int32_t>(std::strlen(name))));
        ASSERT_TRUE(viaString == viaSpan);
    }
    ASSERT_TRUE(gl::ruleJustificationFromString(gl::StrSpan("", 0))
                == gl::RuleJustification::none);
    ASSERT_TRUE(gl::ruleJustificationFromString(gl::StrSpan("implication", 11))
                == gl::RuleJustification::implication);
    ASSERT_TRUE(gl::ruleJustificationFromString(gl::StrSpan("integration", 11))
                == gl::RuleJustification::integration);
}

// IdVecKeyView reads the Codec<IdVecKey> bytes exactly, no owning decode
// (empty + multi id runs).
TEST(memory, id_vec_key_view_matches_codec_decode) {
    const std::vector<std::vector<int32_t>> cases = {
        {}, { 5 }, { -1, 0, 7, 123456 }, { 1, 2, 3, 4, 5 },
    };
    for (const std::vector<int32_t>& ids : cases) {
        const gl::IdVecKey k{ ids };
        const std::string bytes = gl::Codec<gl::IdVecKey>::encode(k);
        const gl::IdVecKeyView v = gl::viewIdVecKey(
            gl::StrSpan(bytes.data(), static_cast<int32_t>(bytes.size())));
        ASSERT_EQ(v.count(), static_cast<int32_t>(ids.size()));
        for (int32_t i = 0; i < v.count(); ++i)
            ASSERT_EQ(v.idAt(i), ids[static_cast<std::size_t>(i)]);
    }
}

// findDefinitionSet == std::to_string(idx) + definitionSets.find(): same entry
// pointer on a hit, nullptr on a defined miss.
TEST(prover, find_definition_set_matches_map) {
    ce::CoreExpressionConfig cfg;
    cfg.definitionSets["1"] = std::make_pair(std::string("(1)"), true);
    cfg.definitionSets["2"] = std::make_pair(std::string("(N)"), false);
    cfg.definitionSets["10"] = std::make_pair(std::string("(x)"), false);
    for (int idx : { 1, 2, 3, 10, 11 }) {
        const std::pair<std::string, bool>* got =
            gl::ExpressionAnalyzer::findDefinitionSet(&cfg, idx);
        const auto it = cfg.definitionSets.find(std::to_string(idx));
        if (it == cfg.definitionSets.end())
            ASSERT_TRUE(got == nullptr);
        else
            ASSERT_TRUE(got == &it->second);
    }
}

// insertRemainingArgsNormKey raw-run overload == the set/NormKey forwarder.
TEST(memory, insert_remaining_args_normkey_raw_matches_owning) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::TypedColdBlobMap<gl::Int16SetKey, gl::NormKey> mapOwn(&lb, &d);
    gl::TypedColdBlobMap<gl::Int16SetKey, gl::NormKey> mapRaw(&lb, &d);
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);

    struct I { std::vector<int16_t> arg; int16_t ne; std::vector<int16_t> data; };
    const std::vector<I> is = {
        { { 1, 2 }, 3, { 5, 6 } }, { { 1, 2 }, 3, { 4, 6 } },   // same key, sorts
        { { 7 }, 2, { 9 } }, { { 1, 2 }, 3, { 5, 6 } },         // duplicate (no-op)
    };
    std::vector<gl::Int16SetKey> keys;
    for (const I& it : is) {
        const std::set<int16_t> argSet(it.arg.begin(), it.arg.end());
        const gl::NormKey nk{ it.ne, it.data };
        gl::ExpressionAnalyzer::insertRemainingArgsNormKey(mapOwn, argSet, nk, tArena);
        gl::ExpressionAnalyzer::insertRemainingArgsNormKey(mapRaw,
            it.arg.data(), static_cast<int32_t>(it.arg.size()), it.ne,
            it.data.empty() ? nullptr : it.data.data(),
            static_cast<int32_t>(it.data.size()), tArena);
        keys.push_back(gl::Int16SetKey{ std::vector<int16_t>(argSet.begin(), argSet.end()) });
    }
    for (const gl::Int16SetKey& k : keys) {
        const int32_t io = mapOwn.lookup(k);
        const int32_t ir = mapRaw.lookup(k);
        ASSERT_TRUE(io != 0 && ir != 0);
        ASSERT_EQ(mapOwn.runLen(io), mapRaw.runLen(ir));
        for (int32_t j = 0; j < mapOwn.runLen(io); ++j)
            ASSERT_TRUE(gl::Codec<gl::NormKey>::serialize(mapOwn.recordAt(io, j))
                     == gl::Codec<gl::NormKey>::serialize(mapRaw.recordAt(ir, j)));
    }
}

// insertRejectedIdsBlob (S6d C7): the id-run door deposits byte-identically to
// insertRejectedValue. Ascending-unique levels, empty sib/level runs, duplicate.
TEST(memory, insert_rejected_ids_blob_matches_value_form) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::ColdStringTable valTable(&lb, &d);
    gl::ValueInterner vi;
    vi.bind(&valTable);
    gl::TypedColdBlobMap<int32_t, gl::RejectedMapValue> mapV(&lb, &d);
    gl::TypedColdBlobMap<int32_t, gl::RejectedMapValue> mapB(&lb, &d);
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);

    const int32_t renZZ = vi.encode("zz_ren");
    const int32_t renAA = vi.encode("aa_ren");
    const int32_t expr1 = vi.encode("(in3[a,b,c,plus])");
    const int32_t conc1 = vi.encode("(in[b,1])");
    struct Rec { int32_t ren; int32_t expr; int iter; int32_t conc; std::vector<int32_t> sibs; std::vector<int> levels; };
    const std::vector<Rec> recs = {
        { renZZ, expr1, 5, conc1, {conc1}, {1, 3} },
        { renAA, expr1, 0, conc1, {}, {} },          // empty sibs + levels
        { renAA, expr1, 2, conc1, {conc1}, {0, 2, 3} },
        { renZZ, expr1, 5, conc1, {conc1}, {1, 3} },  // duplicate of the first
    };
    const int32_t pk = 7;
    for (const Rec& rc : recs) {
        gl::RejectedMapValue v(rc.ren, rc.expr, rc.iter, rc.conc,
            std::vector<int32_t>(rc.sibs),
            std::set<int>(rc.levels.begin(), rc.levels.end()));
        gl::insertRejectedValue(mapV, pk, v, vi, tArena);
        gl::insertRejectedIdsBlob(mapB, pk, rc.ren, rc.expr, rc.iter, rc.conc,
            rc.sibs.empty() ? nullptr : rc.sibs.data(), static_cast<int32_t>(rc.sibs.size()),
            rc.levels.empty() ? nullptr : rc.levels.data(), static_cast<int32_t>(rc.levels.size()),
            vi, tArena);
    }
    const int32_t idV = mapV.lookup(pk);
    const int32_t idB = mapB.lookup(pk);
    ASSERT_TRUE(idV != 0 && idB != 0);
    ASSERT_EQ(mapV.runLen(idV), mapB.runLen(idB));
    for (int32_t j = 0; j < mapV.runLen(idV); ++j) {
        int32_t lV = 0, lB = 0;
        const char* pV = mapV.peekRecordBytes(idV, j, lV, tArena);
        const std::vector<char> bufV(pV, pV + lV);
        const char* pB = mapB.peekRecordBytes(idB, j, lB, tArena);
        ASSERT_EQ(lV, lB);
        ASSERT_TRUE(std::memcmp(bufV.data(), pB, static_cast<size_t>(lV)) == 0);
    }
}

// insertRejectedIntegrationIdsBlob (S6d C7, manager-added conversion): the id-run
// door deposits byte-identically to the value-form insertRejectedIntegrationValue
// (the retained heap-set-RMW oracle). Shuffled multi-record, empty siblings,
// duplicate.
TEST(memory, insert_rejected_integration_ids_blob_matches_value_form) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::ColdStringTable valTable(&lb, &d);
    gl::ValueInterner vi;
    vi.bind(&valTable);
    gl::TypedColdBlobMap<int32_t, gl::RejectedMapIntegrationValue> mapV(&lb, &d);
    gl::TypedColdBlobMap<int32_t, gl::RejectedMapIntegrationValue> mapB(&lb, &d);
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);

    const int32_t cZZ = vi.encode("zz_conc");
    const int32_t cAA = vi.encode("aa_conc");
    const int32_t comp1 = vi.encode("(compound[x])");
    const int32_t s1 = vi.encode("(in[y,2])");
    struct Rec { int32_t conc; std::vector<int32_t> sibs; int32_t comp; };
    const std::vector<Rec> recs = {
        { cZZ, {s1, comp1}, comp1 },
        { cAA, {}, comp1 },              // empty siblings
        { cAA, {s1}, comp1 },
        { cZZ, {s1, comp1}, comp1 },     // duplicate of the first
    };
    const int32_t pk = 9;
    for (const Rec& rc : recs) {
        gl::RejectedMapIntegrationValue v(rc.conc, std::vector<int32_t>(rc.sibs), rc.comp);
        gl::insertRejectedIntegrationValue(mapV, pk, v, vi);
        gl::insertRejectedIntegrationIdsBlob(mapB, pk, rc.conc, rc.comp,
            rc.sibs.empty() ? nullptr : rc.sibs.data(), static_cast<int32_t>(rc.sibs.size()),
            vi, tArena);
    }
    const int32_t idV = mapV.lookup(pk);
    const int32_t idB = mapB.lookup(pk);
    ASSERT_TRUE(idV != 0 && idB != 0);
    ASSERT_EQ(mapV.runLen(idV), mapB.runLen(idB));
    for (int32_t j = 0; j < mapV.runLen(idV); ++j) {
        int32_t lV = 0, lB = 0;
        const char* pV = mapV.peekRecordBytes(idV, j, lV, tArena);
        const std::vector<char> bufV(pV, pV + lV);
        const char* pB = mapB.peekRecordBytes(idB, j, lB, tArena);
        ASSERT_EQ(lV, lB);
        ASSERT_TRUE(std::memcmp(bufV.data(), pB, static_cast<size_t>(lV)) == 0);
    }
}

// renamingChainScratch (S9 C1) — the span-run twin reproduces renamingChain
// element-for-element, byte-identical. Cases: no-u_ pass-through; single u_ arg
// with no changeable overlap; changeable-vs-u_ OVERLAP (exercises the nextFree
// remap + decimal build); a multi-element chain. renamingChain is the retained
// oracle.
TEST(prover, renaming_chain_scratch_matches_vector) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::ScratchArena& sa =
        gl::scratchArenas().forSlot(gl::scratchArenas().slotCount() - 1);

    const auto runCase = [&](const std::vector<std::string>& chainVec,
                             const std::set<std::string>& remSet) {
        const std::vector<std::string> oracle = ea.renamingChain(chainVec, remSet);

        gl::ScratchScope scope(sa);
        std::vector<gl::StrSpan> chainSpans;
        for (const std::string& s : chainVec) chainSpans.push_back(gl::StrSpan(s));
        std::vector<gl::StrSpan> remSpans;
        for (const std::string& s : remSet) remSpans.push_back(gl::StrSpan(s));
        gl::StrSpan outChain[gl::ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
        const int32_t n = ea.renamingChainScratch(sa,
            chainSpans.data(), static_cast<int32_t>(chainSpans.size()),
            remSpans.empty() ? nullptr : remSpans.data(),
            static_cast<int32_t>(remSpans.size()),
            outChain, gl::ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS);
        ASSERT_EQ(static_cast<size_t>(n), oracle.size());
        for (int32_t i = 0; i < n; ++i) {
            ASSERT_EQ(std::string(outChain[i].ptr,
                          static_cast<size_t>(outChain[i].len)),
                      oracle[static_cast<size_t>(i)]);
        }
    };

    runCase({ "(f[1,2])" }, {});                       // (1) no-u_ pass-through
    runCase({ "(f[u_1,2])" }, { "1" });                // (2) single u_, no overlap
    runCase({ "(f[u_1,1])" }, { "1" });                // (3) changeable-vs-u_ overlap
    runCase({ "(f[u_1,2])", "(g[u_3])" }, { "1", "3" }); // (4) multi-element chain
}

// prefixNonMarkerArgumentsWithUScratch (S9 C1) — the scratch twin reproduces the
// inline {arg != "marker" -> u_+arg} + ce::replaceKeysInString rename that
// isAdmittedIntegration performs. Cases: no args; all non-marker; a "marker" arg
// present (must stay "marker", NOT "u_marker"); mixed. The inline form is the
// in-test oracle.
TEST(prover, prefix_non_marker_arguments_with_u_scratch_matches_inline) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::ScratchArena& sa =
        gl::scratchArenas().forSlot(gl::scratchArenas().slotCount() - 1);

    const auto oracle = [&](const std::string& expr) -> std::string {
        std::map<std::string, std::string> replacementMap;
        for (const std::string& arg : ce::getArgs(expr)) {
            if (arg != "marker") replacementMap[arg] = "u_" + arg;
        }
        return ce::replaceKeysInString(expr, replacementMap);
    };

    const std::string cases[] = {
        "(f[])",                 // no args -> pass-through
        "(f[a,b])",              // all non-marker
        "(f[a,marker,b])",       // marker stays verbatim (NOT u_marker)
        "(g[marker,x,marker])",  // mixed
    };
    for (const std::string& expr : cases) {
        gl::ScratchScope scope(sa);
        const gl::ScratchString got =
            ea.prefixNonMarkerArgumentsWithUScratch(sa, gl::StrSpan(expr));
        ASSERT_EQ(got.toStdString(), oracle(expr));
    }
}

// deferredAncestorAdmissionLess (S9 C3) — the sealed 6-field replay comparator
// agrees in SIGN with the retired heap sort's !=/< chain at every field's
// decision point, and returns 0 on a fully-equal pair. Pins the deload-observable
// replay order (I-107). Field 1 (exprKey) reads Memory::exprKeyView, so the
// records' ancestors are Memory shells with fixed setExprKeys.
TEST(prover, deferred_ancestor_admission_less_matches_heap) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::SealedPageSet ps;
    ps.bind(&gl::staticMemory());
    const auto seal = [&ps](const std::string& v) {
        return gl::SealedString::copyFrom(ps, v.data(),
                                          static_cast<int32_t>(v.size()));
    };

    gl::Memory shellAA; shellAA.setExprKey("aa");
    gl::Memory shellBB; shellBB.setExprKey("bb");
    gl::Memory shellKK; shellKK.setExprKey("kk");

    struct Oracle {
        std::string exprKey;
        std::vector<std::string> key;
        std::set<std::string> remainingArgs;
        int maxAdmissionDepth;
        int maxSecondaryNumber;
        bool partOfRecursion;
    };
    const auto oracleLess = [](const Oracle& a, const Oracle& b) -> bool {
        if (a.exprKey != b.exprKey) return a.exprKey < b.exprKey;
        if (a.key != b.key) return a.key < b.key;
        if (a.remainingArgs != b.remainingArgs) return a.remainingArgs < b.remainingArgs;
        if (a.partOfRecursion != b.partOfRecursion) return a.partOfRecursion < b.partOfRecursion;
        if (a.maxAdmissionDepth != b.maxAdmissionDepth) return a.maxAdmissionDepth < b.maxAdmissionDepth;
        return a.maxSecondaryNumber < b.maxSecondaryNumber;
    };
    const auto makeDAA = [&](gl::Memory* anc, const std::vector<std::string>& key,
                             const std::vector<std::string>& rem,
                             int depth, int sec, bool por) {
        std::vector<gl::SealedString> ke;
        for (const std::string& s : key) ke.push_back(seal(s));
        std::vector<gl::SealedString> re;
        for (const std::string& s : rem) re.push_back(seal(s));
        gl::ExpressionAnalyzer::DeferredAncestorAdmission d;
        d.ancestor = anc;
        d.key = gl::SealedSpan<gl::SealedString>::copyFrom(
            ps, ke.data(), static_cast<int32_t>(ke.size()));
        d.remainingArgs = gl::SealedSpan<gl::SealedString>::copyFrom(
            ps, re.data(), static_cast<int32_t>(re.size()));
        d.maxAdmissionDepth = depth;
        d.maxSecondaryNumber = sec;
        d.partOfRecursion = por;
        return d;
    };
    const auto checkPair = [&](const gl::ExpressionAnalyzer::DeferredAncestorAdmission& sa,
                               const gl::ExpressionAnalyzer::DeferredAncestorAdmission& sb,
                               const Oracle& oa, const Oracle& ob) {
        const int cmp = ea.deferredAncestorAdmissionLess(sa, sb);
        if (oracleLess(oa, ob)) { ASSERT_TRUE(cmp < 0); }
        else if (oracleLess(ob, oa)) { ASSERT_TRUE(cmp > 0); }
        else { ASSERT_TRUE(cmp == 0); }
    };

    // Field 1 — exprKey difference.
    {
        Oracle oa{ "aa", {"(x[1])"}, {"a"}, 1, 1, false };
        Oracle ob{ "bb", {"(x[1])"}, {"a"}, 1, 1, false };
        checkPair(makeDAA(&shellAA, oa.key, {"a"}, 1, 1, false),
                  makeDAA(&shellBB, ob.key, {"a"}, 1, 1, false), oa, ob);
    }
    // Field 2 — exprKey tie, key difference.
    {
        Oracle oa{ "kk", {"(x[1])"}, {"a"}, 1, 1, false };
        Oracle ob{ "kk", {"(x[2])"}, {"a"}, 1, 1, false };
        checkPair(makeDAA(&shellKK, oa.key, {"a"}, 1, 1, false),
                  makeDAA(&shellKK, ob.key, {"a"}, 1, 1, false), oa, ob);
    }
    // Field 3 — key tie, remainingArgs difference.
    {
        Oracle oa{ "kk", {"(x[1])"}, {"a"}, 1, 1, false };
        Oracle ob{ "kk", {"(x[1])"}, {"b"}, 1, 1, false };
        checkPair(makeDAA(&shellKK, oa.key, {"a"}, 1, 1, false),
                  makeDAA(&shellKK, ob.key, {"b"}, 1, 1, false), oa, ob);
    }
    // Field 4 — remainingArgs tie, partOfRecursion (false < true).
    {
        Oracle oa{ "kk", {"(x[1])"}, {"a"}, 1, 1, false };
        Oracle ob{ "kk", {"(x[1])"}, {"a"}, 1, 1, true };
        checkPair(makeDAA(&shellKK, oa.key, {"a"}, 1, 1, false),
                  makeDAA(&shellKK, ob.key, {"a"}, 1, 1, true), oa, ob);
    }
    // Field 5 — maxAdmissionDepth.
    {
        Oracle oa{ "kk", {"(x[1])"}, {"a"}, 1, 1, false };
        Oracle ob{ "kk", {"(x[1])"}, {"a"}, 2, 1, false };
        checkPair(makeDAA(&shellKK, oa.key, {"a"}, 1, 1, false),
                  makeDAA(&shellKK, ob.key, {"a"}, 2, 1, false), oa, ob);
    }
    // Field 6 — maxSecondaryNumber.
    {
        Oracle oa{ "kk", {"(x[1])"}, {"a"}, 1, 1, false };
        Oracle ob{ "kk", {"(x[1])"}, {"a"}, 1, 2, false };
        checkPair(makeDAA(&shellKK, oa.key, {"a"}, 1, 1, false),
                  makeDAA(&shellKK, ob.key, {"a"}, 1, 2, false), oa, ob);
    }
    // Fully-equal pair — comparator returns 0.
    {
        Oracle oa{ "kk", {"(x[1])"}, {"a"}, 1, 1, false };
        Oracle ob = oa;
        checkPair(makeDAA(&shellKK, oa.key, {"a"}, 1, 1, false),
                  makeDAA(&shellKK, ob.key, {"a"}, 1, 1, false), oa, ob);
    }

    ps.seal();
    ps.freePages();
}

// Pinning tests (S6d C7): the value runs the site sequences must mint into the
// value interner in EXACTLY the order the former ctor-argument expression did
// under MSVC. Fresh interner A runs the OLD ctor VERBATIM; fresh interner B runs
// the NEW sequenced build; the two id tables must be byte-equal (mint-SEQUENCE
// equality via decodeView(1..size())). A failure means the observed MSVC order
// differs from the sequencing — flip the sequence to match. A future WSL/gcc
// failure is the pre-existing cross-compiler .deload divergence surfacing, not a
// regression (report; the gate baseline is MSVC).
static void assertInternerTablesEqual(const gl::ValueInterner& a,
                                      const gl::ValueInterner& b) {
    ASSERT_EQ(a.table->count(), b.table->count());
    for (int32_t i = 1; i <= a.table->count(); ++i) {
        ASSERT_TRUE(gl::equalSpans(a.decodeView(i), b.decodeView(i)));
    }
}

TEST(memory, admission_value_ctor_mint_order_frozen) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState dA = gl::DirtyState::Clean, dB = gl::DirtyState::Clean;
    gl::ColdStringTable tA(&lb, &dA), tB(&lb, &dB);
    gl::ValueInterner viA, viB;
    viA.bind(&tA); viB.bind(&tB);

    const std::vector<std::string> renamedKey = { "(op[u_1,marker])", "kk", "aa" };
    const std::set<std::string> remainingArgs = { "zz", "bb", "mm" };

    // A: the OLD ctor expression VERBATIM (MSVC ctor-arg evaluation order).
    const gl::AdmissionMapValue vA(
        encodeValueVectorOracle(renamedKey, viA),
        gl::encodeValueSetSorted(remainingArgs, viA), 3, 4, false);
    (void)vA;

    // B: the NEW sequenced build — remainingArgs (arg2) minted BEFORE renamedKey
    // (arg1), matching MSVC's right-to-left order.
    for (const std::string& s : remainingArgs) (void)viB.encode(gl::StrSpan(s));
    for (const std::string& s : renamedKey) (void)viB.encode(gl::StrSpan(s));

    assertInternerTablesEqual(viA, viB);
}

TEST(memory, rejected_value_ctor_mint_order_frozen) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState dA = gl::DirtyState::Clean, dB = gl::DirtyState::Clean;
    gl::ColdStringTable tA(&lb, &dA), tB(&lb, &dB);
    gl::ValueInterner viA, viB;
    viA.bind(&tA); viB.bind(&tB);

    const std::string renamedExpr = "zz_ren";
    const std::string expr = "mm_expr";
    const std::string concrete = "bb_conc";
    const std::vector<std::string> siblings = { "s2", "s1", "s3" };
    const int levels[] = { 0, 2 };

    // A: the OLD ctor VERBATIM.
    const gl::RejectedMapValue vA(
        viA.encode(renamedExpr), viA.encode(expr), 5, viA.encode(concrete),
        encodeValueVectorOracle(siblings, viA), levels, 2);
    (void)vA;

    // B: the NEW sequenced build — MSVC right-to-left: siblings (arg5), then
    // concrete (arg4), expr (arg2), renamed (arg1).
    for (const std::string& s : siblings) (void)viB.encode(gl::StrSpan(s));
    (void)viB.encode(gl::StrSpan(concrete));
    (void)viB.encode(gl::StrSpan(expr));
    (void)viB.encode(gl::StrSpan(renamedExpr));

    assertInternerTablesEqual(viA, viB);
}

TEST(memory, rejected_integration_value_ctor_mint_order_frozen) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState dA = gl::DirtyState::Clean, dB = gl::DirtyState::Clean;
    gl::ColdStringTable tA(&lb, &dA), tB(&lb, &dB);
    gl::ValueInterner viA, viB;
    viA.bind(&tA); viB.bind(&tB);

    const std::string concrete = "zz_conc";
    const std::vector<std::string> siblings = { "s2", "s1" };
    const std::string compound = "aa_comp";

    // A: the OLD ctor VERBATIM.
    const gl::RejectedMapIntegrationValue vA(
        viA.encode(concrete), encodeValueVectorOracle(siblings, viA),
        viA.encode(compound));
    (void)vA;

    // B: the NEW sequenced build — MSVC right-to-left: compound (arg3), then
    // siblings (arg2), concrete (arg1).
    (void)viB.encode(gl::StrSpan(compound));
    for (const std::string& s : siblings) (void)viB.encode(gl::StrSpan(s));
    (void)viB.encode(gl::StrSpan(concrete));

    assertInternerTablesEqual(viA, viB);
}

// emitIntegrationRevivalToInternalMailIn span/run door (S4 C1): deposits
// byte-identically to the string overload — same statement keys, same origin
// records. Cases: 0 siblings / 2 siblings, levels {1,3} / empty, equalities
// empty / 2-element sorted, pre != post and pre == post. Compared via the
// statements_ facet per id plus makeHeapMail (statements + exprOriginMap).
TEST(prover, emit_integration_revival_span_door_twin) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::Memory mA;  // string overload
    gl::Memory mB;  // span/run overload

    // Case 1: pre != post, 0 siblings, levels {1,3}, no equalities.
    const std::string pre1 = "(in3[it_5_lev_1_2,7,3,plus])";
    const std::string post1 = "(in3[it_0_lev_1_2,7,3,plus])";
    const std::string valid1 = "main";
    ea.emitIntegrationRevivalToInternalMailIn(pre1, post1, {}, {},
        std::set<int>{ 1, 3 }, valid1, std::set<std::string>{}, mA);
    const int lv13[2] = { 1, 3 };
    ea.emitIntegrationRevivalToInternalMailIn(
        gl::StrSpan(pre1), gl::StrSpan(post1),
        nullptr, 0, nullptr, 0,
        lv13, 2, gl::StrSpan(valid1), nullptr, 0, mB);

    // Case 2: pre == post, 2 siblings, EMPTY levels, 2 sorted equalities.
    const std::string p2 = "(in[b,N])";
    const std::string sPre0 = "(in[a,N])", sPre1 = "(in[c,N])";
    const std::string sPost0 = "(in[a2,N])", sPost1 = "(in[c2,N])";
    const std::string valid2 = "main_boundary_ordis_((=[2,repl_lev_1_0]))";
    const std::string eq0 = "(=[a,a2])", eq1 = "(=[c,c2])";
    ea.emitIntegrationRevivalToInternalMailIn(p2, p2,
        std::vector<std::string>{ sPre0, sPre1 },
        std::vector<std::string>{ sPost0, sPost1 },
        std::set<int>{}, valid2,
        std::set<std::string>{ eq0, eq1 }, mA);
    const gl::StrSpan sibsPre[2] = { gl::StrSpan(sPre0), gl::StrSpan(sPre1) };
    const gl::StrSpan sibsPost[2] = { gl::StrSpan(sPost0),
                                      gl::StrSpan(sPost1) };
    const gl::StrSpan eqs[2] = { gl::StrSpan(eq0), gl::StrSpan(eq1) };
    ea.emitIntegrationRevivalToInternalMailIn(
        gl::StrSpan(p2), gl::StrSpan(p2),
        sibsPre, 2, sibsPost, 2,
        nullptr, 0, gl::StrSpan(valid2), eqs, 2, mB);

    // Statement facet: same count, same key bytes per id.
    auto& stA = mA.sameIterationInternalMail.statements_;
    auto& stB = mB.sameIterationInternalMail.statements_;
    ASSERT_EQ(stA.count(), stB.count());
    for (int32_t i = 1; i <= stA.count(); ++i) {
        ASSERT_TRUE(gl::equalSpans(stA.inner().decode(i),
                                   stB.inner().decode(i)));
    }

    // Decoded equality: statements + full origin maps.
    const gl::Mail hA = gl::makeHeapMail(mA.sameIterationInternalMail,
                                         mA.nameMap, mA.originInterner);
    const gl::Mail hB = gl::makeHeapMail(mB.sameIterationInternalMail,
                                         mB.nameMap, mB.originInterner);
    ASSERT_TRUE(hB.statements == hA.statements);
    ASSERT_TRUE(hB.exprOriginMap == hA.exprOriginMap);

    // Concrete spot checks: case-1 head origin is equality1 with pre first;
    // case-2 sibling origin carries pre + both equalities.
    const auto it1 = hB.exprOriginMap.find(
        gl::ExpressionWithValidity(post1, valid1));
    ASSERT_TRUE(it1 != hB.exprOriginMap.end());
    ASSERT_EQ(it1->second.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(it1->second[0].first, std::string("equality1"));
    ASSERT_EQ(it1->second[0].second.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(it1->second[0].second[0].original, pre1);
    const auto it2 = hB.exprOriginMap.find(
        gl::ExpressionWithValidity(sPost1, valid2));
    ASSERT_TRUE(it2 != hB.exprOriginMap.end());
    ASSERT_EQ(it2->second.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(it2->second[0].second.size(), static_cast<std::size_t>(3));
    ASSERT_EQ(it2->second[0].second[0].original, sPre1);
    ASSERT_EQ(it2->second[0].second[1].original, eq0);
    ASSERT_EQ(it2->second[0].second[2].original, eq1);
}

// stagedToIdValue span encodes (S4 C3): the span-encode interior produces a
// field-identical AdmissionMapValue AND an identical interner mint SEQUENCE
// (id table byte-equal, not just value-equal) vs the retained toStdString
// oracle replicated verbatim below against a second fresh interner.
TEST(memory, staged_to_id_value_span_matches_string_oracle) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::ColdStringTable tableA(&lb, &d);
    gl::ColdStringTable tableB(&lb, &d);
    gl::ValueInterner viA;
    viA.bind(&tableA);
    gl::ValueInterner viB;
    viB.bind(&tableB);

    gl::SealedPageSet pages;
    pages.bind(&gl::staticMemory());
    const auto s = [&pages](const std::string& v) {
        return gl::SealedString::copyFrom(pages, v.data(),
                                          static_cast<int32_t>(v.size()));
    };

    gl::StagedAdmissionValue sv;
    const gl::SealedString keyArr[2] = { s("(in[b,1])"), s("10") };
    sv.key = gl::SealedSpan<gl::SealedString>::copyFrom(pages, keyArr, 2);
    const gl::SealedString argArr[2] = { s("3"), s("b") };
    sv.remainingArgsSorted =
        gl::SealedSpan<gl::SealedString>::copyFrom(pages, argArr, 2);
    sv.standardMaxAdmissionDepth = 4;
    sv.standardMaxSecondaryNumber = 2;
    sv.flag = true;
    pages.seal();

    // ORACLE — verbatim replica of the retired toStdString interior,
    // against interner A.
    std::vector<int32_t> oracleKeyIds;
    for (const gl::SealedString& k : sv.key) {
        oracleKeyIds.push_back(viA.encode(k.toStdString()));
    }
    std::vector<int32_t> oracleRemainingIds;
    for (const gl::SealedString& a : sv.remainingArgsSorted) {
        oracleRemainingIds.push_back(viA.encode(a.toStdString()));
    }
    const gl::AdmissionMapValue oracle(
        oracleKeyIds, oracleRemainingIds,
        sv.standardMaxAdmissionDepth, sv.standardMaxSecondaryNumber, sv.flag);

    // Production span path against interner B.
    const gl::AdmissionMapValue prod = gl::stagedToIdValue(sv, viB);

    ASSERT_TRUE(prod.key == oracle.key);
    ASSERT_TRUE(prod.remainingArgs == oracle.remainingArgs);
    ASSERT_EQ(prod.standardMaxAdmissionDepth, oracle.standardMaxAdmissionDepth);
    ASSERT_EQ(prod.standardMaxSecondaryNumber,
              oracle.standardMaxSecondaryNumber);
    ASSERT_TRUE(prod.flag == oracle.flag);

    // Mint-SEQUENCE equality: the two interners' id tables are byte-equal
    // (same strings at the same ids), not merely value-equal.
    ASSERT_EQ(viA.internedCount(), viB.internedCount());
    for (int32_t i = 1; i <= viA.internedCount(); ++i) {
        ASSERT_TRUE(viA.decode(i) == viB.decode(i));
    }

    pages.freePages();
}

// spanSetContains (S4 C4): binary_search under compareSpans over a
// sorted-unique run — verdict identical to std::set<std::string> membership
// for members, non-members, boundary elements, and the empty run.
TEST(prover, span_set_contains_matches_std_set) {
    const std::set<std::string> oracle{ "a", "b", "plus", "u_x" };
    const std::string elems[4] = { "a", "b", "plus", "u_x" };
    gl::StrSpan run[4];
    int32_t n = 0;
    for (const std::string& e : elems) run[n++] = gl::StrSpan(e);

    const std::string probes[8] = { "a", "b", "plus", "u_x",
                                    "", "aa", "c", "zz" };
    for (const std::string& p : probes) {
        const bool expect = oracle.find(p) != oracle.end();
        ASSERT_TRUE(gl::ExpressionAnalyzer::spanSetContains(
                        run, n, gl::StrSpan(p)) == expect);
    }
    // Empty run: nothing is a member.
    ASSERT_FALSE(gl::ExpressionAnalyzer::spanSetContains(
        run, 0, gl::StrSpan(std::string("a"))));
}

// prepareIntegration span core: caller-side set-flatten matches a hand-built
// sorted run. The heap delegator (const std::string& / std::set) was deleted;
// its set -> sorted-unique-run flatten now lives in the callers
// (addExprToMemoryBlock / prefillIntegrationMapsRecursive). Rig A flattens the
// std::set into a run inline (the former delegator body) and calls the core;
// rig B calls the core with the equivalent hand-built sorted array; every
// observable integration container must end identical. Case 2 inserts three
// unchangeable args into the set OUT of lex order, pinning
// set-iteration-order == sorted-run.
TEST(prover, prepare_integration_set_flatten_matches_sorted_run) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::Memory mA;
    gl::Memory mB;

    // Case 1: one unchangeable arg; the compiled core must resolve.
    {
        const std::string expr = "(in[b,N])";
        ASSERT_TRUE(ea.compiledExpressions.find(
                        extractExpressionUniversalOracle(expr))
                    != ea.compiledExpressions.end());
        const std::set<std::string> unchg{ "N" };
        gl::StrSpan sortedA[gl::ExecutionParameters::MAX_ARITY];
        int32_t nA = 0;
        for (const std::string& a : unchg) sortedA[nA++] = gl::StrSpan(a);
        ea.prepareIntegration(gl::StrSpan(expr), sortedA, nA, mA,
                              gl::StrSpan(std::string("main")));

        const std::string nStr = "N";
        const gl::StrSpan run[1] = { gl::StrSpan(nStr) };
        ea.prepareIntegration(gl::StrSpan(expr), run, 1, mB,
                              gl::StrSpan(std::string("main")));
    }

    // Case 2: three unchangeable args inserted out of lex order (pins the
    // set-iteration == sorted-run mapping); one changeable arg.
    {
        const std::string expr = "(in3[a,b,c,plus])";
        ASSERT_TRUE(ea.compiledExpressions.find(
                        extractExpressionUniversalOracle(expr))
                    != ea.compiledExpressions.end());
        std::set<std::string> unchg;
        unchg.insert("plus");
        unchg.insert("a");
        unchg.insert("b");
        gl::StrSpan sortedA[gl::ExecutionParameters::MAX_ARITY];
        int32_t nA = 0;
        for (const std::string& a : unchg) sortedA[nA++] = gl::StrSpan(a);
        ea.prepareIntegration(gl::StrSpan(expr), sortedA, nA, mA,
                              gl::StrSpan(std::string("main")));

        const std::string aStr = "a", bStr = "b", plusStr = "plus";
        const gl::StrSpan run[3] = { gl::StrSpan(aStr), gl::StrSpan(bStr),
                                     gl::StrSpan(plusStr) };
        ea.prepareIntegration(gl::StrSpan(expr), run, 3, mB,
                              gl::StrSpan(std::string("main")));
    }

    // Observable-state equality across the two rigs.
    ASSERT_EQ(mA.startIntPi, mB.startIntPi);
    ASSERT_EQ(mA.startInt, mB.startInt);

    ASSERT_EQ(mA.integrationPreparedMarker.count(),
              mB.integrationPreparedMarker.count());
    for (int32_t i = 1; i <= mA.integrationPreparedMarker.count(); ++i) {
        ASSERT_EQ(mA.integrationPreparedMarker.decode(i),
                  mB.integrationPreparedMarker.decode(i));
    }

    ASSERT_EQ(mA.integrationStartIntMap.count(),
              mB.integrationStartIntMap.count());
    for (int32_t i = 1; i <= mA.integrationStartIntMap.count(); ++i) {
        ASSERT_EQ(mA.integrationStartIntMap.decode(i),
                  mB.integrationStartIntMap.decode(i));
        ASSERT_EQ(mA.integrationStartIntMap.valueAt(i),
                  mB.integrationStartIntMap.valueAt(i));
    }

    auto& amiA = mA.overallHashMemory.admissionMapIntegration;
    auto& amiB = mB.overallHashMemory.admissionMapIntegration;
    ASSERT_EQ(amiA.count(), amiB.count());
    for (int32_t i = 1; i <= amiA.count(); ++i) {
        ASSERT_EQ(amiA.decodeKey(i), amiB.decodeKey(i));
        ASSERT_EQ(amiA.runLen(i), amiB.runLen(i));
        ASSERT_TRUE(sameRun(amiA.recordsAt(i), amiB.recordsAt(i)));
    }

    auto& varsA = mA.overallHashMemory.varsInAdmissionMapIntegrationKeys;
    auto& varsB = mB.overallHashMemory.varsInAdmissionMapIntegrationKeys;
    ASSERT_EQ(varsA.count(), varsB.count());
    for (int32_t i = 1; i <= varsA.count(); ++i) {
        ASSERT_EQ(varsA.decodeKey(i), varsB.decodeKey(i));
    }

    // The two per-rig NameMap / interner tables must be byte-equal too —
    // mint-sequence equality across the whole call, not just end-state.
    ASSERT_EQ(mA.nameMap.nameCount(), mB.nameMap.nameCount());
    ASSERT_EQ(mA.templateInterner.internedCount(),
              mB.templateInterner.internedCount());
    for (int16_t i = 1; i <= mA.templateInterner.internedCount(); ++i) {
        ASSERT_TRUE(mA.templateInterner.decode(i)
                    == mB.templateInterner.decode(i));
    }
}

// originTagFromString span twin (S4 C5): for EVERY enumerator the span form
// resolves the same tag as the string form and the identity round-trips;
// plus a mid-buffer span probe (no null-terminator dependence). The
// unknown-tag assert path is not harness-testable (assert aborts) — the
// string oracle asserts identically.
TEST(memory, origin_tag_from_string_span_matches_string) {
    for (int i = 0; i < static_cast<int>(gl::OriginTag::COUNT); ++i) {
        const gl::OriginTag t = static_cast<gl::OriginTag>(i);
        const char* name = gl::originTagName(t);
        const int32_t len = static_cast<int32_t>(std::strlen(name));
        ASSERT_TRUE(gl::originTagFromString(gl::StrSpan(name, len)) == t);
        ASSERT_TRUE(gl::originTagFromString(std::string(name)) == t);
    }
    // Mid-buffer slice with live bytes past len: "Xequality1Y" sliced.
    const std::string buf = "Xequality1Y";
    ASSERT_TRUE(gl::originTagFromString(gl::StrSpan(buf.data() + 1, 9))
                == gl::OriginTag::equality1);
}

// setInternalDisintegrationSignal span door (S4 C5): the span overload
// writes the identical packed key + flag byte as the EWV overload — 2
// statements x both flag combinations, ONE shared NameMap (identical key
// packing), two mailboxes compared facet-by-facet.
TEST(memory, internal_disintegration_signal_span_door_twin) {
    gl::Memory mA;  // EWV overload writes here
    gl::Memory mB;  // span overload writes here, keyed via mA.nameMap
    const std::string e1 = "(in3[a,b,c,plus])", v1 = "main";
    const std::string e2 = "(=[2,repl_lev_1_0])", v2 = "main_boundary_x";

    gl::setInternalDisintegrationSignal(mA.sameIterationInternalMail,
        mA.nameMap, gl::ExpressionWithValidity(e1, v1), true, false);
    gl::setInternalDisintegrationSignal(mA.sameIterationInternalMail,
        mA.nameMap, gl::ExpressionWithValidity(e2, v2), false, true);

    gl::setInternalDisintegrationSignal(mB.sameIterationInternalMail,
        mA.nameMap, gl::StrSpan(e1), gl::StrSpan(v1), true, false);
    gl::setInternalDisintegrationSignal(mB.sameIterationInternalMail,
        mA.nameMap, gl::StrSpan(e2), gl::StrSpan(v2), false, true);

    auto& sigA = mA.sameIterationInternalMail.disintegrationSignals_;
    auto& sigB = mB.sameIterationInternalMail.disintegrationSignals_;
    ASSERT_EQ(sigA.count(), sigB.count());
    ASSERT_EQ(sigA.count(), 2);
    for (int32_t i = 1; i <= sigA.count(); ++i) {
        ASSERT_EQ(sigA.decodeKey(i), sigB.decodeKey(i));
        ASSERT_EQ(sigA.valueAt(i), sigB.valueAt(i));
    }
}

// LmvBlobView (S4 C6): every accessor reads the exact field the LMV codec
// wrote — checked against Codec<LocalMemoryValue>::deserialize AND the input,
// across empty/multi levels x empty/multi keyIds x empty/multi remainingArgIds
// x both bool combos; keyIdsBytes is byte-identical to re-serializing just
// the keyIds run.
namespace {
    inline gl::LocalMemoryValue mkLmv(int32_t valueId, int32_t implId,
        int16_t vid, bool marker, bool pod, std::set<int> levels,
        std::vector<int32_t> keyIds, std::vector<int32_t> remaining) {
        gl::LocalMemoryValue v;
        v.valueId = valueId;
        v.originalImplicationId = implId;
        v.justification = gl::RuleJustification::implication;
        v.validityId = vid;
        v.isMarker = marker;
        v.productOfDisintegration = pod;
        v.levels = std::move(levels);
        v.keyIds = std::move(keyIds);
        v.remainingArgIds = std::move(remaining);
        return v;
    }
}

TEST(memory, lmv_blob_view_matches_codec) {
    auto check = [](const gl::LocalMemoryValue& v) {
        const std::vector<char> blob =
            gl::Codec<gl::LocalMemoryValue>::serialize(v);
        const int32_t n = static_cast<int32_t>(blob.size());
        const gl::LmvBlobView view(blob.data(), n);
        const gl::LocalMemoryValue back =
            gl::Codec<gl::LocalMemoryValue>::deserialize(blob.data(), n);

        ASSERT_EQ(view.originalImplicationId(), back.originalImplicationId);
        ASSERT_EQ(view.originalImplicationId(), v.originalImplicationId);
        ASSERT_EQ(view.validityId(), back.validityId);
        ASSERT_EQ(view.validityId(), v.validityId);
        ASSERT_EQ(view.validityId(),
                  gl::lmvBlobValidityId(blob.data(), n));
        ASSERT_EQ(view.levelCount(), static_cast<int32_t>(v.levels.size()));
        ASSERT_EQ(view.keyIdCount(), static_cast<int32_t>(v.keyIds.size()));
        ASSERT_EQ(view.keyIdsByteLen(),
                  static_cast<int32_t>(4 * v.keyIds.size()));
        for (int32_t i = 0; i < view.keyIdCount(); ++i) {
            ASSERT_EQ(view.keyIdAt(i),
                      v.keyIds[static_cast<std::size_t>(i)]);
            ASSERT_EQ(view.keyIdAt(i),
                      back.keyIds[static_cast<std::size_t>(i)]);
        }
        // keyIdsBytes region == the raw int32 run of the keyIds vector.
        if (!v.keyIds.empty()) {
            ASSERT_TRUE(std::memcmp(view.keyIdsBytes(), v.keyIds.data(),
                            static_cast<std::size_t>(view.keyIdsByteLen()))
                        == 0);
        }

        // C2 (_firing_check) extension: the firing-check value loop's accessors.
        ASSERT_EQ(view.valueId(), v.valueId);
        ASSERT_EQ(view.valueId(), back.valueId);
        ASSERT_EQ(static_cast<int>(view.justification()),
                  static_cast<int>(v.justification));
        ASSERT_EQ(static_cast<int>(view.justification()),
                  static_cast<int>(back.justification));
        ASSERT_EQ(view.isMarker(), v.isMarker);
        ASSERT_EQ(view.productOfDisintegration(), v.productOfDisintegration);
        // levelAt(i) reproduces the ascending std::set<int> iteration order.
        {
            int32_t i = 0;
            for (const int lv : v.levels) { ASSERT_EQ(view.levelAt(i), lv); ++i; }
        }
        ASSERT_EQ(view.remainingCount(),
                  static_cast<int32_t>(v.remainingArgIds.size()));
        for (int32_t i = 0; i < view.remainingCount(); ++i) {
            ASSERT_EQ(view.remainingArgIdAt(i),
                      v.remainingArgIds[static_cast<std::size_t>(i)]);
            ASSERT_EQ(view.remainingArgIdAt(i),
                      back.remainingArgIds[static_cast<std::size_t>(i)]);
        }
    };

    check(mkLmv(1, 0, 5, false, false, {}, {}, {}));
    check(mkLmv(2, 77, 3, true, false, { 0, 2, 9 }, { 10, 11, 12 }, { 4 }));
    check(mkLmv(3, 77, 3, false, true, { 1 }, {}, { 8, 9 }));
    check(mkLmv(4, 5, 12, true, true, {}, { 42 }, {}));
    // A non-default justification (mkLmv hardcodes `implication`) so the
    // justification() accessor's tag is actually exercised.
    gl::LocalMemoryValue jv =
        mkLmv(9, 3, 7, false, true, { 2, 5, 11 }, { 1, 2 }, { 8, 9, 20 });
    jv.justification = gl::RuleJustification::integration;
    check(jv);
}

// eradicateEncodedMapForImpl (S4 C6): the two-pass verbatim splice + chain
// side channel is byte-identical to the retired heap oracle (decode ->
// erase matching LMVs -> resetToFresh -> typed assignRun, with a std::set
// chain collection) — same key count/bytes, same run lengths, same per-blob
// bytes, same chain SET. Cases: fully-dropped key, partially-filtered key,
// untouched key, keyIds-empty matching LMV (drops without a chain), a
// duplicate chain across keys (set dedup), and a matching-rule-id-but-
// different-scope LMV (survives).
TEST(memory, eradicate_encoded_map_splice_matches_heap_oracle) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 21, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState dA = gl::DirtyState::Clean;
    gl::DirtyState dB = gl::DirtyState::Clean;
    gl::TypedColdBlobMap<gl::NormKey, gl::LocalMemoryValue> A(&lb, &dA);
    gl::TypedColdBlobMap<gl::NormKey, gl::LocalMemoryValue> B(&lb, &dB);

    const int32_t implId = 77;
    const int16_t implVid = 3;

    const gl::NormKey k1{ 1, { 100, 101 } };        // fully dropped
    const gl::NormKey k2{ 2, { 200 } };             // partially filtered
    const gl::NormKey k3{ 1, { 300, 301, 302 } };   // untouched
    const gl::NormKey k4{ 3, { 400 } };             // keyIds-empty match + dup chain

    const std::vector<gl::LocalMemoryValue> run1{
        mkLmv(1, implId, implVid, false, false, { 1 }, { 10, 11 }, {}),
        mkLmv(2, implId, implVid, true, false, {}, { 12 }, { 5 }) };
    const std::vector<gl::LocalMemoryValue> run2{
        mkLmv(3, implId, implVid, false, true, { 0 }, { 10, 11 }, {}),  // dup chain
        mkLmv(4, 99, implVid, false, false, { 2 }, { 13 }, {}),         // other rule
        mkLmv(5, implId, 8, false, false, {}, { 14 }, {}) };            // other scope
    const std::vector<gl::LocalMemoryValue> run3{
        mkLmv(6, 99, 8, true, true, { 3, 4 }, { 15, 16 }, { 7 }) };
    const std::vector<gl::LocalMemoryValue> run4{
        mkLmv(7, implId, implVid, false, false, { 5 }, {}, { 9 }) };    // no chain

    for (auto* m : { &A, &B }) {
        m->assignRun(k1, run1);
        m->assignRun(k2, run2);
        m->assignRun(k3, run3);
        m->assignRun(k4, run4);
    }

    // ORACLE on A — verbatim replica of the retired eraseFromHashMem body.
    std::set<std::vector<int32_t>> oracleChains;
    {
        const int32_t encN = A.count();
        std::vector<std::pair<gl::NormKey, std::vector<gl::LocalMemoryValue>>>
            survivors;
        for (int32_t id = 1; id <= encN; ++id) {
            std::vector<gl::LocalMemoryValue> lmvs = A.recordsAt(id);
            for (auto it = lmvs.begin(); it != lmvs.end(); ) {
                if (it->originalImplicationId == implId &&
                    it->validityId == implVid) {
                    if (!it->keyIds.empty()) oracleChains.insert(it->keyIds);
                    it = lmvs.erase(it);
                } else {
                    ++it;
                }
            }
            if (!lmvs.empty())
                survivors.emplace_back(A.decodeKey(id), std::move(lmvs));
        }
        A.resetToFresh();
        for (auto& kv : survivors)
            A.assignRun(kv.first, kv.second);
    }

    // PRODUCTION on B.
    gl::ScratchArena gArena;
    gArena.bind(&gl::staticMemory());
    const gl::ArenaOffset mark = gArena.cursor();
    {
        gl::DirtyState chainsDirty = gl::DirtyState::Clean;
        gl::ColdHashSet<gl::BytesKeyStore> erasedChains(&gArena, &chainsDirty);
        gl::eradicateEncodedMapForImpl(B, implId, implVid, gArena,
                                       erasedChains);

        // Chain sets equal AS SETS: same size; every oracle chain's raw
        // bytes present in the cold set.
        ASSERT_EQ(static_cast<int32_t>(oracleChains.size()),
                  erasedChains.count());
        for (const std::vector<int32_t>& ch : oracleChains) {
            ASSERT_TRUE(erasedChains.lookup(gl::StrSpan(
                reinterpret_cast<const char*>(ch.data()),
                static_cast<int32_t>(4 * ch.size()))) != 0);
        }
    }
    gArena.popTo(mark);

    // Map facet equality: counts, key bytes, run lengths, per-blob bytes.
    ASSERT_EQ(A.count(), B.count());
    std::vector<char> scrA, scrB;
    for (int32_t id = 1; id <= A.count(); ++id) {
        ASSERT_TRUE(gl::equalSpans(A.inner().decode(id), B.inner().decode(id)));
        ASSERT_EQ(A.runLen(id), B.runLen(id));
        for (int32_t j = 0; j < A.runLen(id); ++j) {
            int32_t la = 0, lb2 = 0;
            const char* pa = A.peekRecordBytes(id, j, la, scrA);
            const char* pb = B.peekRecordBytes(id, j, lb2, scrB);
            ASSERT_EQ(la, lb2);
            ASSERT_TRUE(std::memcmp(pa, pb, static_cast<std::size_t>(la)) == 0);
        }
    }
    // Sanity on the expected shape: k1 (both LMVs match) and k4 (its single
    // chain-less LMV matches) drop fully -> k2 + k3 remain.
    ASSERT_EQ(A.count(), 2);
}

// bestSanitizePeer (C2 shared scan core) — span result byte-equal to the
// retained heap oracle (the sanitize twins' historical decodeClassesAt loop,
// copied verbatim below) across: int-beats-it, same-tier lex pick with
// multi-class running state, only-Normal peers (empty), ancestor-scope-only
// pickup, a Normal arg inside a class, and a never-interned arg (argId 0).
TEST(memory, best_sanitize_peer_matches_decoded_class_oracle) {
    gl::Memory m;
    gl::ScratchArena arena;
    arena.bind(&gl::staticMemory());

    const int16_t mainId = m.nameMap.encode("main");
    const int16_t childId = m.nameMap.encodePush(mainId, "(impl1[q])");
    const std::string mainN = m.nameMap.decode(mainId);
    const std::string childN = m.nameMap.decode(childId);

    gl::EquivalenceClass a;
    a.setMembersFromNames({ "it_0_lev_0_1", "int_lev_0_2" }, m.nameMap);
    gl::EquivalenceClass b;
    b.setMembersFromNames({ "int_lev_0_1", "int_lev_0_2" }, m.nameMap);
    gl::EquivalenceClass c;
    c.setMembersFromNames({ "it_0_lev_0_2", "repl_1", "zz_plain" }, m.nameMap);
    // All classes at MAIN; the child scope has none (the ancestor-pickup case).
    m.assignClassesById(mainId, { a, b, c });

    // ORACLE — the retained heap loop, verbatim from the sanitize twins.
    auto oracleBestPeer = [&m](const std::string& argName,
                               const std::vector<std::string>& scopesToCheck,
                               std::string& bestPeer, bool& bestPeerIsInt) {
        const int16_t argId = m.nameMap.lookup(argName);
        bestPeer.clear();
        bestPeerIsInt = false;
        for (const std::string& scope : scopesToCheck) {
            const std::vector<gl::EquivalenceClass> scopeClasses =
                m.decodeClassesAt(scope);
            for (const gl::EquivalenceClass& clss : scopeClasses) {
                if (argId == 0
                    || std::find(clss.memberIds.begin(), clss.memberIds.end(),
                                 argId) == clss.memberIds.end()) continue;
                for (const int16_t peerId : clss.memberIds) {
                    if (peerId == argId) continue;
                    const gl::NameKind peerKind =
                        m.eqClassNameCaches.kindOf(peerId, m.nameMap);
                    if (peerKind == gl::NameKind::Normal) continue;
                    const bool peerIsInt = (peerKind == gl::NameKind::IntLev);
                    const std::string peer = m.nameMap.decode(peerId);
                    bool peerWins = false;
                    if (bestPeer.empty()) peerWins = true;
                    else if (peerIsInt && !bestPeerIsInt) peerWins = true;
                    else if (peerIsInt == bestPeerIsInt && peer < bestPeer)
                        peerWins = true;
                    if (peerWins) { bestPeer = peer; bestPeerIsInt = peerIsInt; }
                }
            }
        }
    };

    auto check = [&](const std::string& argName,
                     const std::vector<std::string>& scopeNames,
                     const std::string& expectPeer, bool expectIsInt) {
        gl::StrSpan scopes[8];
        for (std::size_t i = 0; i < scopeNames.size(); ++i)
            scopes[i] = gl::StrSpan(scopeNames[i]);
        const int16_t argId = m.nameMap.lookup(argName);
        gl::StrSpan bestPeer;
        bool bestPeerIsInt = false;
        gl::bestSanitizePeer(m, argId, scopes,
            static_cast<int32_t>(scopeNames.size()), arena,
            bestPeer, bestPeerIsInt);
        std::string oraclePeer;
        bool oracleIsInt = false;
        oracleBestPeer(argName, scopeNames, oraclePeer, oracleIsInt);
        ASSERT_EQ(bestPeer.toStdString(), oraclePeer);
        ASSERT_TRUE(bestPeerIsInt == oracleIsInt);
        // Concrete pin on top of the oracle equality.
        ASSERT_EQ(bestPeer.toStdString(), expectPeer);
        ASSERT_TRUE(bestPeerIsInt == expectIsInt);
    };

    check("it_0_lev_0_1", { mainN }, "int_lev_0_2", true);   // int beats it
    check("int_lev_0_2", { mainN }, "int_lev_0_1", true);    // multi-class: a offers it_, b's int_ wins lex
    check("int_lev_0_1", { mainN }, "int_lev_0_2", true);    // same tier, only peer
    check("it_0_lev_0_2", { mainN }, "", false);             // only Normal peers -> empty
    check("int_lev_0_2", { childN, mainN }, "int_lev_0_1", true); // ancestor-only pickup
    check("zz_plain", { mainN }, "it_0_lev_0_2", false);     // Normal arg inside class c
    check("never_interned_arg", { mainN }, "", false);       // argId 0 -> empty
}

// sanitizeToBeProved snapshot order (C3): the packed-key idx sort — the
// production comparator copied verbatim (compareSpans on the decoded high
// half, tie on the decoded low half) — reproduces decodeToBeProvedSorted's
// (original, validityName) row order key-for-key, on goals whose NameMap mint
// order deliberately differs from lex order, with shared validities and a
// shared original across two validities (both tie-break directions).
TEST(memory, tbp_snapshot_idx_sort_matches_decode_to_be_proved_sorted) {
    gl::Memory m;
    const int16_t mainId = m.nameMap.encode("main");
    const int16_t subId = m.nameMap.encodePush(mainId, "(impl1[s])");
    const std::set<int> aux{ 1, 2 };
    auto add = [&m, &aux](const char* o, int16_t vid) {
        const std::string os(o);
        m.intToBeProved.assignSetRange(
            gl::packStatementKey(m.nameMap.encode(os), vid),
            aux.begin(), aux.end());
    };
    // Mint order deliberately differs from lex order ("(z9..." first).
    add("(z9[a])", mainId);
    add("(a1[b])", mainId);   // shares the main validity
    add("(m5[c])", subId);
    add("(a1[a])", subId);
    add("(z9[a])", subId);    // same original at a second validity
    add("(m5[c])", mainId);

    // The production comparator, copied verbatim from sanitizeToBeProved.
    std::vector<int32_t> keys;
    const int32_t n = m.intToBeProved.count();
    for (int32_t id = 1; id <= n; ++id)
        keys.push_back(m.intToBeProved.keyAt(id));
    std::vector<int32_t> idx(static_cast<std::size_t>(n));
    for (int32_t k = 0; k < n; ++k) idx[static_cast<std::size_t>(k)] = k;
    std::sort(idx.begin(), idx.end(), [&](int32_t a, int32_t b) {
        const int32_t ka = keys[static_cast<std::size_t>(a)];
        const int32_t kb = keys[static_cast<std::size_t>(b)];
        const int c = gl::compareSpans(
            m.nameMap.decodeView(static_cast<int16_t>(
                (static_cast<uint32_t>(ka) >> 16) & 0xFFFF)),
            m.nameMap.decodeView(static_cast<int16_t>(
                (static_cast<uint32_t>(kb) >> 16) & 0xFFFF)));
        if (c != 0) return c < 0;
        return gl::compareSpans(
            m.nameMap.decodeView(static_cast<int16_t>(
                static_cast<uint32_t>(ka) & 0xFFFF)),
            m.nameMap.decodeView(static_cast<int16_t>(
                static_cast<uint32_t>(kb) & 0xFFFF))) < 0;
    });

    // ORACLE — the retained production sort.
    const std::vector<gl::DecodedToBeProvedRow> rows =
        gl::decodeToBeProvedSorted(m.intToBeProved, m.nameMap);
    ASSERT_EQ(rows.size(), keys.size());
    for (std::size_t k = 0; k < rows.size(); ++k) {
        ASSERT_EQ(keys[static_cast<std::size_t>(idx[k])], rows[k].key);
    }
}

// sanitizeHashMemory snapshot order (C5): the packed-key idx sort — the
// production comparator copied verbatim (compareSpans on the decoded
// lbStateInterner hi half, tie on the decoded lo half) — reproduces the
// retired decoded-EWV std::sort (ExpressionWithValidity::operator<) row for
// row, on strings whose mint order deliberately differs from lex order, with
// a shared original across two validities and shared validities (both
// tie-break directions).
TEST(memory, expimpl_snapshot_idx_sort_matches_ewv_sort) {
    gl::Memory m;
    // Mint order deliberately differs from lex order ("(implication9..." first).
    const int32_t zImpl = m.lbStateInterner.encode("(implication9[z,q])");
    const int32_t aImpl = m.lbStateInterner.encode("(implication1[a,b])");
    const int32_t mImpl = m.lbStateInterner.encode("(implication5[m,n])");
    const int32_t vMain = m.lbStateInterner.encode("main");
    const int32_t vSub = m.lbStateInterner.encode("main_boundary_(impl1[s])");

    const std::vector<int64_t> keys = {
        gl::packLbStateKey(zImpl, vMain),
        gl::packLbStateKey(aImpl, vSub),
        gl::packLbStateKey(aImpl, vMain),   // shares the original with prev
        gl::packLbStateKey(mImpl, vMain),   // shares the main validity
        gl::packLbStateKey(zImpl, vSub),
        gl::packLbStateKey(mImpl, vSub),
    };

    // The production comparator, copied verbatim from sanitizeHashMemory
    // (today's exact unpack casts).
    std::vector<int32_t> idx(keys.size());
    for (std::size_t k = 0; k < keys.size(); ++k)
        idx[k] = static_cast<int32_t>(k);
    std::sort(idx.begin(), idx.end(), [&](int32_t a, int32_t b) {
        const int64_t ka = keys[static_cast<std::size_t>(a)];
        const int64_t kb = keys[static_cast<std::size_t>(b)];
        const int c = gl::compareSpans(
            m.lbStateInterner.decodeView(
                static_cast<int32_t>(static_cast<uint64_t>(ka) >> 32)),
            m.lbStateInterner.decodeView(
                static_cast<int32_t>(static_cast<uint64_t>(kb) >> 32)));
        if (c != 0) return c < 0;
        return gl::compareSpans(
            m.lbStateInterner.decodeView(
                static_cast<int32_t>(ka & 0xFFFFFFFFLL)),
            m.lbStateInterner.decodeView(
                static_cast<int32_t>(kb & 0xFFFFFFFFLL))) < 0;
    });

    // ORACLE — decode each key exactly as the retired snapshot did, then
    // std::sort under ExpressionWithValidity::operator<.
    std::vector<gl::ExpressionWithValidity> rows;
    for (const int64_t pk : keys) {
        rows.emplace_back(
            std::string(m.lbStateInterner.decode(
                static_cast<int32_t>(static_cast<uint64_t>(pk) >> 32))),
            std::string(m.lbStateInterner.decode(
                static_cast<int32_t>(pk & 0xFFFFFFFFLL))));
    }
    std::sort(rows.begin(), rows.end());

    ASSERT_EQ(rows.size(), keys.size());
    for (std::size_t k = 0; k < keys.size(); ++k) {
        const int64_t pk = keys[static_cast<std::size_t>(idx[k])];
        ASSERT_EQ(m.lbStateInterner.decode(
                      static_cast<int32_t>(static_cast<uint64_t>(pk) >> 32)),
                  rows[k].original);
        ASSERT_EQ(m.lbStateInterner.decode(
                      static_cast<int32_t>(pk & 0xFFFFFFFFLL)),
                  rows[k].validityName);
    }
}

// sanitizeHashMemory apply-time levels probe (C5): the runLen/valueAt run
// copy equals the retired coldIntSetAt set element-for-element across empty /
// single / multi-level runs, and the span lookupStatementLevels miss path
// (never-interned pair) returns 0 — the empty case on both forms.
TEST(memory, sanitize_levels_run_probe_matches_cold_int_set) {
    gl::Memory m;
    const std::string valid = "main";
    const char* exprs[] = { "(p[a])", "(p[b])", "(p[c])" };
    const std::set<int> sets[] = { {}, { 0 }, { 1, 3, 7 } };
    for (int i = 0; i < 3; ++i) {
        m.intStatementLevelsMap.assignSetRange(
            gl::packStatementKey(m.nameMap.encode(std::string(exprs[i])),
                                 m.nameMap.encode(valid)),
            sets[i].begin(), sets[i].end());
    }
    for (int i = 0; i < 3; ++i) {
        const std::string e(exprs[i]);
        const int32_t id = gl::lookupStatementLevels(
            m.intStatementLevelsMap, m.nameMap,
            gl::StrSpan(e), gl::StrSpan(valid));
        ASSERT_NE(id, 0);
        const std::set<int> oracle =
            gl::coldIntSetAt(m.intStatementLevelsMap, id);
        // NEW — the apply-time run copy loop.
        std::vector<int> run;
        const int32_t nLv = m.intStatementLevelsMap.runLen(id);
        for (int32_t j = 0; j < nLv; ++j)
            run.push_back(m.intStatementLevelsMap.valueAt(id, j));
        ASSERT_EQ(run.size(), oracle.size());
        std::size_t k = 0;
        for (const int x : oracle) {
            ASSERT_EQ(run[k], x);
            ++k;
        }
    }
    // The id-0 miss path: a never-interned pair probes to 0.
    const std::string ghost = "(never[q])";
    ASSERT_EQ(gl::lookupStatementLevels(m.intStatementLevelsMap, m.nameMap,
                                        gl::StrSpan(ghost), gl::StrSpan(valid)),
              0);
}

// coldIntRunAt byte-twin: the stack-run fill equals coldIntSetAt (the retained
// heap oracle) element-for-element for empty, singleton, multi, and long runs;
// the exact-capacity case (cap == n) passes the assert.
TEST(memory, cold_int_run_at_matches_cold_int_set) {
    gl::Memory m;
    const std::string valid = "main";
    const char* exprs[] = { "(q[a])", "(q[b])", "(q[c])", "(q[d])" };
    std::set<int> sets[4];
    sets[0] = {};
    sets[1] = { 5 };
    sets[2] = { 0, 3, 9 };
    for (int v = 0; v < 20; ++v) sets[3].insert(v * 2);   // 20-element run
    for (int i = 0; i < 4; ++i) {
        m.intStatementLevelsMap.assignSetRange(
            gl::packStatementKey(m.nameMap.encode(std::string(exprs[i])),
                                 m.nameMap.encode(valid)),
            sets[i].begin(), sets[i].end());
    }
    for (int i = 0; i < 4; ++i) {
        const std::string e(exprs[i]);
        const int32_t id = gl::lookupStatementLevels(
            m.intStatementLevelsMap, m.nameMap,
            gl::StrSpan(e), gl::StrSpan(valid));
        ASSERT_NE(id, 0);
        const std::set<int> oracle =
            gl::coldIntSetAt(m.intStatementLevelsMap, id);
        int run[32];
        const int32_t n =
            gl::coldIntRunAt(m.intStatementLevelsMap, id, run, 32);
        ASSERT_EQ(n, static_cast<int32_t>(oracle.size()));
        int32_t k = 0;
        for (const int x : oracle) {
            ASSERT_EQ(run[k], x);
            ++k;
        }
    }
    // Exact-cap case: cap == n passes (the assert is n <= cap).
    {
        const std::string e(exprs[2]);
        const int32_t id = gl::lookupStatementLevels(
            m.intStatementLevelsMap, m.nameMap,
            gl::StrSpan(e), gl::StrSpan(valid));
        int tight[3];
        ASSERT_EQ(gl::coldIntRunAt(m.intStatementLevelsMap, id, tight, 3), 3);
        ASSERT_EQ(tight[0], 0);
        ASSERT_EQ(tight[1], 3);
        ASSERT_EQ(tight[2], 9);
    }
}

// insertLevelSorted byte-twin: driving the run helper and a std::set<int>
// oracle through one insert sequence keeps count + element order equal after
// every step — empty-run seed, front/middle/back inserts, duplicates of
// front/middle/back.
TEST(memory, insert_level_sorted_matches_set_insert) {
    int run[16];
    int32_t n = 0;
    std::set<int> oracle;
    const int seq[] = { 5, 2, 9, 5, 7, 2, 9, 0, 12, 7 };
    for (const int v : seq) {
        n = gl::insertLevelSorted(run, n, v, 16);
        oracle.insert(v);
        ASSERT_EQ(n, static_cast<int32_t>(oracle.size()));
        int32_t k = 0;
        for (const int x : oracle) {
            ASSERT_EQ(run[k], x);
            ++k;
        }
    }
    // Exact-cap growth: a 16th distinct value with cap 16 passes.
    int full[3];
    int32_t fn = 0;
    fn = gl::insertLevelSorted(full, fn, 1, 3);
    fn = gl::insertLevelSorted(full, fn, 0, 3);
    fn = gl::insertLevelSorted(full, fn, 2, 3);
    ASSERT_EQ(fn, 3);
    ASSERT_EQ(full[0], 0);
    ASSERT_EQ(full[1], 1);
    ASSERT_EQ(full[2], 2);
    // Duplicate at exact capacity is a no-op (no assert).
    ASSERT_EQ(gl::insertLevelSorted(full, fn, 1, 3), 3);
}

// sanitizeToBeProved apply-time value run (C3): assignSetRange fed a
// contiguous ascending int[] copy of a set produces facet-identical state to
// assignSetRange fed the std::set iterators — empty, single, and multi-level
// runs (keyAt / runLen / valueAt equality per id).
TEST(memory, tbp_assign_set_range_run_matches_set) {
    gl::Memory mA;
    gl::Memory mB;
    const std::set<int> cases[] = { {}, { 0 }, { 0, 3, 7 } };
    int16_t orig = 5;
    for (const std::set<int>& s : cases) {
        const int32_t pk = gl::packStatementKey(orig++, gl::NameMap::MAIN_ID);
        mA.intToBeProved.assignSetRange(pk, s.begin(), s.end());
        int buf[8];
        int32_t bn = 0;
        for (const int x : s) buf[bn++] = x;
        const int* rp = (bn > 0) ? buf : nullptr;
        mB.intToBeProved.assignSetRange(pk, rp, rp + bn);
    }
    ASSERT_EQ(mA.intToBeProved.count(), mB.intToBeProved.count());
    ASSERT_EQ(mA.intToBeProved.count(), 3);
    for (int32_t id = 1; id <= mA.intToBeProved.count(); ++id) {
        ASSERT_EQ(mA.intToBeProved.keyAt(id), mB.intToBeProved.keyAt(id));
        ASSERT_EQ(mA.intToBeProved.runLen(id), mB.intToBeProved.runLen(id));
        for (int32_t j = 0; j < mA.intToBeProved.runLen(id); ++j) {
            ASSERT_EQ(mA.intToBeProved.valueAt(id, j),
                      mB.intToBeProved.valueAt(id, j));
        }
    }
}

// addInternalMailOrigin span door (L4 origin-key sink): the StrSpan overload
// keys the deposit byte-identically to the ExpressionWithValidity overload —
// same originInterner mints, same origins run. The origin RECORD stays string
// (the deferred antecedent boundary). Compared through makeHeapMail.
TEST(memory, internal_mail_origin_span_door_twin) {
    gl::Memory mEwv;
    gl::Memory mSpan;
    const std::string expr = "(=[2,repl_lev_1_0])";
    const std::string valid = "main_boundary_ordis_sig";
    const gl::OriginLine origin{ "equality1",
        std::vector<gl::ExpressionWithValidity>{
            gl::ExpressionWithValidity("(=[2,3])", valid),
            gl::ExpressionWithValidity("(=[3,repl_lev_1_0])", valid) } };
    const int cap = 8;

    gl::addInternalMailOrigin(mEwv.sameIterationInternalMail, mEwv.originInterner,
        gl::ExpressionWithValidity(expr, valid), origin, cap);
    gl::addInternalMailOrigin(mSpan.sameIterationInternalMail, mSpan.originInterner,
        gl::StrSpan(expr), gl::StrSpan(valid), origin, cap);

    const gl::Mail hEwv = gl::makeHeapMail(mEwv.sameIterationInternalMail,
                                           mEwv.nameMap, mEwv.originInterner);
    const gl::Mail hSpan = gl::makeHeapMail(mSpan.sameIterationInternalMail,
                                            mSpan.nameMap, mSpan.originInterner);

    const gl::ExpressionWithValidity key(expr, valid);
    const auto itE = hEwv.exprOriginMap.find(key);
    const auto itS = hSpan.exprOriginMap.find(key);
    ASSERT_TRUE(itE != hEwv.exprOriginMap.end());
    ASSERT_TRUE(itS != hSpan.exprOriginMap.end());
    ASSERT_EQ(static_cast<int>(itS->second.size()),
              static_cast<int>(itE->second.size()));
    ASSERT_EQ(static_cast<int>(itS->second.size()), 1);
    // Same tag + same decoded antecedents.
    ASSERT_TRUE(itS->second[0].first == itE->second[0].first);
    ASSERT_TRUE(itS->second[0].first == "equality1");
    ASSERT_EQ(static_cast<int>(itS->second[0].second.size()),
              static_cast<int>(itE->second[0].second.size()));
    ASSERT_EQ(static_cast<int>(itS->second[0].second.size()), 2);
    for (std::size_t i = 0; i < itS->second[0].second.size(); ++i) {
        ASSERT_TRUE(itS->second[0].second[i].original
                    == itE->second[0].second[i].original);
        ASSERT_TRUE(itS->second[0].second[i].validityName
                    == itE->second[0].second[i].validityName);
    }
}

// insertRejectedIntegrationValue + rejectedIntegrationRecordsAt — cold blob-map
// round trip for the integration rejection buffer: inserted values come back as
// the sorted-unique set (dedup honored); a never-written key reads as empty.
TEST(prover, integration_rejected_cold_roundtrip) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::Memory m;
    auto& rmi = m.overallHashMemory.rejectedMapIntegration;

    const int32_t pk = gl::mintTemplateKey(m.templateInterner, m.nameMap,
        "(in2[marker,7,3])", "main");
    gl::RejectedMapIntegrationValue a(
        m.valueInterner.encode("(in2[5,7,3])"),
        encodeValueVectorOracle(std::vector<std::string>{ "(plus[5,7])" }, m.valueInterner),
        m.valueInterner.encode("(compound_a)"));
    gl::RejectedMapIntegrationValue b(
        m.valueInterner.encode("(in2[6,7,3])"),
        encodeValueVectorOracle(std::vector<std::string>{ "(plus[6,7])" }, m.valueInterner),
        m.valueInterner.encode("(compound_b)"));

    gl::insertRejectedIntegrationValue(rmi, pk, a, m.valueInterner);
    gl::insertRejectedIntegrationValue(rmi, pk, b, m.valueInterner);
    gl::insertRejectedIntegrationValue(rmi, pk, a, m.valueInterner);  // dedup

    const gl::RejectedIntegrationValueSet got =
        gl::rejectedIntegrationRecordsAt(rmi, pk, m.valueInterner);
    ASSERT_EQ(got.size(), static_cast<std::size_t>(2));

    // Cold miss: a minted-but-never-written key reads as an empty set.
    const int32_t absentPk = gl::mintTemplateKey(m.templateInterner, m.nameMap,
        "(in2[marker,9,9])", "main");
    ASSERT_TRUE(gl::rejectedIntegrationRecordsAt(rmi, absentPk, m.valueInterner).empty());
}

// admissionIntegrationRecordsAt + flattenIntegrationEntryMap — cold blob-map
// round trip for the NESTED admission-integration value (instruction -> payload
// id set). Exercises the production write pattern: decode -> payloadAt -> flatten
// -> assignRun, including accumulation across two round trips and the cold miss.
TEST(prover, integration_admission_cold_roundtrip) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::Memory m;
    auto& ami = m.overallHashMemory.admissionMapIntegration;

    const int32_t pk = gl::mintTemplateKey(m.templateInterner, m.nameMap,
        "(in2[u_a,marker,3])", "main");

    // One id-form instruction key.
    gl::IntInstruction iinstr;
    gl::IntLogicalEntity ile;
    ile.category  = m.valueInterner.encode("in");
    ile.signature = m.valueInterner.encode("(in2[u_a,marker,3])");
    ile.elements  = { m.valueInterner.encode("u_a"),
                      m.valueInterner.encode("marker"),
                      m.valueInterner.encode("3") };
    ile.arity     = 3;
    ile.definedSet = m.valueInterner.encode("N");
    iinstr.data.push_back(ile);
    iinstr.markedGoal = m.valueInterner.encode("marker");

    // Round trip 1: register the instruction with a two-id payload.
    {
        gl::IntegrationEntryMap em =
            gl::admissionIntegrationRecordsAt(ami, pk, m.valueInterner);
        gl::ValueIdSet& payload = gl::payloadAt(em, iinstr, m.valueInterner);
        payload.insert(m.valueInterner.encode("5"));
        payload.insert(m.valueInterner.encode("6"));
        ami.assignRun(pk, gl::flattenIntegrationEntryMap(em));
    }
    const gl::IntegrationEntryMap got1 =
        gl::admissionIntegrationRecordsAt(ami, pk, m.valueInterner);
    ASSERT_EQ(got1.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(got1.begin()->second.size(), static_cast<std::size_t>(2));

    // Round trip 2: accumulate — a re-decode sees round trip 1's write-back.
    {
        gl::IntegrationEntryMap em =
            gl::admissionIntegrationRecordsAt(ami, pk, m.valueInterner);
        gl::payloadAt(em, iinstr, m.valueInterner)
            .insert(m.valueInterner.encode("7"));
        ami.assignRun(pk, gl::flattenIntegrationEntryMap(em));
    }
    const gl::IntegrationEntryMap got2 =
        gl::admissionIntegrationRecordsAt(ami, pk, m.valueInterner);
    ASSERT_EQ(got2.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(got2.begin()->second.size(), static_cast<std::size_t>(3));

    // Cold miss: a minted-but-never-written key reads as an empty nested map.
    const int32_t absentPk = gl::mintTemplateKey(m.templateInterner, m.nameMap,
        "(in2[u_z,marker,9])", "main");
    ASSERT_TRUE(gl::admissionIntegrationRecordsAt(ami, absentPk, m.valueInterner).empty());
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
    ASSERT_EQ(lmv.valueId, 0);
    ASSERT_TRUE(lmv.levels.empty());
    ASSERT_EQ(lmv.originalImplicationId, 0);
    ASSERT_TRUE(lmv.justification == gl::RuleJustification::none);
    ASSERT_TRUE(lmv.keyIds.empty());
    ASSERT_TRUE(lmv.remainingArgIds.empty());
    ASSERT_EQ(lmv.validityId, gl::NameMap::MAIN_ID);
    ASSERT_FALSE(lmv.isMarker);
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
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
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

// decodeExpression — exact inverse of encodeExpression. The int row interns
// the whole (original, validityName) pair; decoding rebuilds the full
// EncodedExpression through the canonical parsing constructor and must be
// equal field-by-field (operator==) to the struct the row was encoded from.
TEST(memory, decodeexpression_roundtrip_plain) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    gl::EncodedExpression ee("(in2[i0,v1,s])", "main");
    gl::IntEncodedExpr ie = gl::encodeExpression(ee, nm);
    gl::EncodedExpression back = gl::decodeExpression(ie, nm);
    ASSERT_TRUE(back == ee);
}

// Negation shape: the leading "!(" is part of the interned original and is
// re-peeled by the parsing constructor on decode.
TEST(memory, decodeexpression_roundtrip_negation) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    gl::EncodedExpression ee("!(=[i0,i1])", "main");
    gl::IntEncodedExpr ie = gl::encodeExpression(ee, nm);
    ASSERT_EQ(ie.negation, static_cast<int16_t>(1));
    gl::EncodedExpression back = gl::decodeExpression(ie, nm);
    ASSERT_TRUE(back == ee);
    ASSERT_TRUE(back.negation);
}

// Argument grammar shapes: a u_-prefixed (unchangeable) arg and a full
// it_<iter>_lev_<lev>_<argId> arg both survive the round trip — the decoded
// struct re-parses them into the same 5-tuples.
TEST(memory, decodeexpression_roundtrip_u_and_itlev_args) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    gl::EncodedExpression ee("(myExpr[u_y,it_2_lev_1_x])", "main");
    gl::IntEncodedExpr ie = gl::encodeExpression(ee, nm);
    ASSERT_EQ(ie.argUnchangeable[0], static_cast<int16_t>(1));
    ASSERT_EQ(ie.argIteration[1],    static_cast<int16_t>(2));
    ASSERT_EQ(ie.maxIteration,       static_cast<int16_t>(2));
    gl::EncodedExpression back = gl::decodeExpression(ie, nm);
    ASSERT_TRUE(back == ee);
}

// Non-main validity (encodePush path) round-trips, and decode is genuinely
// non-minting: after an unrelated mint grows the NameMap, a second decode of
// the same row still reproduces the identical struct.
TEST(memory, decodeexpression_roundtrip_nonmain_validity) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    gl::EncodedExpression ee("(in[v1,N])", "main_boundary_hypo1");
    gl::IntEncodedExpr ie = gl::encodeExpression(ee, nm);
    gl::EncodedExpression back = gl::decodeExpression(ie, nm);
    ASSERT_TRUE(back == ee);
    nm.encode("(unrelated[z])");
    gl::EncodedExpression back2 = gl::decodeExpression(ie, nm);
    ASSERT_TRUE(back2 == ee);
}

// encodeExpression — arity exactly at MAX_ARITY is legal (the entry assert
// guards only the impossible above-cap case; lossless encoding requires every
// argument slot to fit) and still round-trips through decodeExpression.
TEST(memory, encodeexpression_arity_at_cap) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    std::string original = "(wide[a0";
    for (int i = 1; i < gl::ExecutionParameters::MAX_ARITY; ++i) {
        original += ",a" + std::to_string(i);
    }
    original += "])";
    gl::EncodedExpression ee(original, "main");
    ASSERT_EQ(static_cast<int>(ee.arguments.size()),
              static_cast<int>(gl::ExecutionParameters::MAX_ARITY));
    gl::IntEncodedExpr ie = gl::encodeExpression(ee, nm);
    ASSERT_EQ(ie.arity, gl::ExecutionParameters::MAX_ARITY);
    gl::EncodedExpression back = gl::decodeExpression(ie, nm);
    ASSERT_TRUE(back == ee);
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
// compileImplicationToCompact — the mail-broadcast compaction wrapper.
// Positive coverage across the three shapes that reach the mail channel:
//   (1) fully-bound theorem  -> (implication<N>[])   [also exercises the
//                                                     excludeRepetitions
//                                                     zero-arg fix end-to-end]
//   (2) genuinely-free arg    -> (implication<N>[<one arg>])
//   (3) u_-prefixed free arg  -> stripped across EVERY nested [...], so the
//                                result is byte-identical to (2)'s bare form
// The failure mode (an implication that fails to compile to a simple
// (implication<N>[...])) is a hard assert by design (CLAUDE.md Rule 19);
// the harness aborts on assert rather than catching it, so the negative
// path is intentionally not a death-test here.
// =============================================================================

TEST(prover, compileimpltocompact_fully_bound_theorem_has_no_args) {
    gl::ExpressionAnalyzer ea("Peano");
    // a, b both bound by the outer >[a,b]; zero free variables. With zero
    // free args the compact form must be the well-formed (implication<N>[])
    // — the case the excludeRepetitions zero-arg signature fix repairs.
    const std::string compact =
        ea.compileImplicationToCompact("(>[a,b](in[a,b])(in[b,a]))");
    ASSERT_TRUE(ce::expressionIsSimple(compact));
    ASSERT_TRUE(compact.rfind("(implication", 0) == 0);
    ASSERT_TRUE(ce::getArgs(compact).empty());
}

TEST(prover, compileimpltocompact_free_arg_appears_in_arglist) {
    gl::ExpressionAnalyzer ea("Peano");
    // a is bound by >[a]; b is bound nowhere -> exactly one free arg.
    const std::string compact =
        ea.compileImplicationToCompact("(>[a](in[a,b])(in[b,a]))");
    ASSERT_TRUE(ce::expressionIsSimple(compact));
    ASSERT_TRUE(compact.rfind("(implication", 0) == 0);
    ASSERT_EQ(ce::getArgs(compact).size(), static_cast<std::size_t>(1));
}

TEST(prover, compileimpltocompact_strips_u_prefix_across_nested_brackets) {
    gl::ExpressionAnalyzer ea("Peano");
    // u_b sits in INNER brackets only; removeUPrefixFromArguments (which
    // sees the first [...] only) would miss it. The AST-walking strip must
    // reduce u_b -> b so the u_-form compiles to the SAME compact instance
    // as the already-bare form (a cache hit on the identical normalized
    // chain). Equality of the two results is the strip-worked signal.
    const std::string fromU =
        ea.compileImplicationToCompact("(>[a](in[a,u_b])(in[u_b,a]))");
    const std::string fromBare =
        ea.compileImplicationToCompact("(>[a](in[a,b])(in[b,a]))");
    ASSERT_TRUE(ce::expressionIsSimple(fromU));
    ASSERT_TRUE(fromU.rfind("(implication", 0) == 0);
    ASSERT_EQ(ce::getArgs(fromU).size(), static_cast<std::size_t>(1));
    ASSERT_EQ(fromU, fromBare);
}

TEST(prover, compileimpltocompact_negation_is_injective) {
    gl::ExpressionAnalyzer ea("Peano");
    // Regression guard for the confirmed non-injectivity
    // (D-76). makeNormalizedEncodedKey emits
    // a per-constituent negation bit, so an implication and its negated-head
    // variant have DIFFERENT dedup keys and MUST get different implication<N>
    // names — negation is not overlooked. A structurally-distinct third form
    // must also get its own name (general injectivity). Run single-threaded
    // here, which post-fix is the only path the compile ever runs on.
    const std::string plain    = ea.compileImplicationToCompact("(>[a](in[a,b])(in[b,a]))");
    const std::string negHead  = ea.compileImplicationToCompact("(>[a](in[a,b])!(in[b,a]))");
    const std::string distinct = ea.compileImplicationToCompact("(>[a,b](in[a,b])(in[b,a]))");
    ASSERT_TRUE(ce::expressionIsSimple(plain));
    ASSERT_TRUE(ce::expressionIsSimple(negHead));
    ASSERT_TRUE(ce::expressionIsSimple(distinct));
    ASSERT_TRUE(plain.rfind("(implication", 0)    == 0);
    ASSERT_TRUE(negHead.rfind("(implication", 0)  == 0);
    ASSERT_TRUE(distinct.rfind("(implication", 0) == 0);
    const std::string cPlain    = extractExpressionUniversalOracle(plain);
    const std::string cNegHead  = extractExpressionUniversalOracle(negHead);
    const std::string cDistinct = extractExpressionUniversalOracle(distinct);
    ASSERT_TRUE(cPlain   != cNegHead);   // negation must change the name
    ASSERT_TRUE(cPlain   != cDistinct);  // distinct structure -> distinct name
    ASSERT_TRUE(cNegHead != cDistinct);
}

TEST(prover, recordpendingcompaction_appends_under_mutex) {
    gl::ExpressionAnalyzer ea("Peano");
    // recordPendingCompaction only appends (original, kySize, coreId) to
    // pendingCompactionQueue under pendingCompactionMutex — no global compile
    // state touched. The post-pool.join() drain consumes the queue.
    ASSERT_TRUE(ea.pendingCompactionQueue.empty());
    ea.recordPendingCompaction("(>[a](in[a,b])(in[b,a]))", 1, 0);
    ea.recordPendingCompaction("(>[a,b](in[a,b])(in[b,a]))", 2, 3);
    ASSERT_EQ(ea.pendingCompactionQueue.size(), static_cast<std::size_t>(2));
    ASSERT_EQ(std::get<0>(ea.pendingCompactionQueue[0]),
              std::string("(>[a](in[a,b])(in[b,a]))"));
    ASSERT_EQ(std::get<1>(ea.pendingCompactionQueue[0]), 1);
    ASSERT_EQ(std::get<2>(ea.pendingCompactionQueue[0]), 0);
    ASSERT_EQ(std::get<1>(ea.pendingCompactionQueue[1]), 2);
    ASSERT_EQ(std::get<2>(ea.pendingCompactionQueue[1]), 3);
}

// =============================================================================
// addExprToMemoryBlock status=3 / disintegrateExpr2 existence-ban (ASIC 0.1
// reshuffle). status=3 (external-mail absorb) now ENTERS disintegrateExpr2
// with allowExistenceDisintegration=false: disintegrateExprCore2's
// `existence` branch returns before minting any it_/int_ witness, so no
// fresh existential variable is admitted; structural + implication
// decomposition still runs and recovered implications install through the
// persistent 3-way addToHashMemory path (overall/local/localDelta).
//
// No direct unit test: disintegrateExpr2 cannot be driven to its
// `existence` branch from a bare ExpressionAnalyzer("Peano") + default
// Memory (precompiled existence<N> forms + a populated integration state
// are required) — the same reason this file's docstring defers all
// addExprToMemoryBlock / disintegrate instance-method coverage to a future
// synthetic-config harness. Coverage instead comes from two directions:
// (1) the rest of this 507-test suite stays green, proving the new
//     allowExistenceDisintegration parameter (default true) leaves every
//     existing disintegration caller byte-identical; and
// (2) the commit-8 Definition-of-Done pipeline run, which exercises
//     status=3 external-mail disintegration with the existence ban
//     end-to-end and deterministically — the Gauss batch loads proved
//     Peano theorems as externals through exactly this path, so a wrong
//     ban surfaces as a Gauss/rung-1 proof loss or a determinism break.
// =============================================================================

// =============================================================================
// Round 2 — two more tests per documented function. Covers edge cases,
// boundary conditions, and round-trips not exercised by the original
// per-function tests.
// =============================================================================

// ---------- NameMap::encode ----------
TEST(memory, namemap_encode_grows_nextid_monotonically) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    const int16_t before = nm.encode("first");
    const int16_t after  = nm.encode("second");
    ASSERT_LT(before, after);   // monotonic id allocation
    // "main" was pre-registered at MAIN_ID == 1; first user root gets 2.
    ASSERT_GE(before, static_cast<int16_t>(2));
}

TEST(memory, namemap_encode_recursive_with_boundary_string) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    // Encoding a fully-canonical "_boundary_"-bearing string registers
    // every prefix recursively: encode("main_boundary_x") creates the
    // child id and the parent ("main") is already mapped via the ctor.
    const int16_t childId = nm.encode("main_boundary_x");
    const int16_t parentId = nm.encode("main");
    ASSERT_NE(childId, parentId);
    // The child's stack carries one payload depth.
    ASSERT_EQ(nm.stackLen(childId), 1);
    // verdict (derived from ancestorsOf) reflects the prefix relation that
    // encodePush would have created if invoked directly.
    int16_t v = 0;
    ASSERT_TRUE(nm.verdict(parentId, childId, v));
    ASSERT_EQ(v, static_cast<int16_t>(-1));
}

// ---------- NameMap::encodePush ----------
TEST(memory, namemap_encodepush_idtosub_grows_monotonically) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    const int16_t mainId = nm.encode("main");
    const int16_t a = nm.encodePush(mainId, "alpha");
    const int16_t b = nm.encodePush(mainId, "beta");
    // The two payloads got distinct sub-ids, monotonically allocated.
    const int16_t subA = nm.stackBack(a);
    const int16_t subB = nm.stackBack(b);
    ASSERT_NE(subA, subB);
    // Re-pushing "alpha" reuses its sub-id (interning).
    const int16_t a2 = nm.encodePush(mainId, "alpha");
    ASSERT_EQ(a, a2);
    ASSERT_EQ(nm.stackBack(a2), subA);
}

TEST(memory, namemap_encodepush_three_deep_chain) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    const int16_t mainId = nm.encode("main");
    const int16_t s1 = nm.encodePush(mainId, "s1");
    const int16_t s2 = nm.encodePush(s1, "s2");
    const int16_t s3 = nm.encodePush(s2, "s3");
    ASSERT_EQ(nm.stackLen(s3), 3);
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
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    ASSERT_EQ(nm.decode(gl::NameMap::MAIN_ID), std::string("main"));
}

TEST(memory, namemap_decode_round_trip_after_encode) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    const std::string canonical = "main_boundary_x_boundary_y";
    const int16_t id = nm.encode(canonical);
    const std::string& decoded = nm.decode(id);
    ASSERT_EQ(decoded, canonical);
}

// ---------- NameMap::comparable / verdict ----------
TEST(memory, namemap_comparable_self_equality) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    const int16_t a = nm.encode("main");
    int16_t v = 0;
    ASSERT_TRUE(nm.verdict(a, a, v));
    ASSERT_EQ(v, static_cast<int16_t>(0));
    ASSERT_TRUE(nm.comparable(a, a));
}

TEST(memory, namemap_comparable_grandparent) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
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
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    const int16_t mainId = nm.encode("main");
    const int16_t child  = nm.encodePush(mainId, "x");
    ASSERT_EQ(nm.deeperOf(mainId, child), child);
    ASSERT_EQ(nm.deeperOf(child, mainId), child);
}

TEST(memory, namemap_deeperof_self) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    const int16_t mainId = nm.encode("main");
    ASSERT_EQ(nm.deeperOf(mainId, mainId), mainId);
}

// ---------- Comparability span/id twins (I-138) ----------

// isStrictAncestor(int16_t,int16_t) — id form agrees with the string form on
// every ordered pair of a three-deep chain, self is not strict, and the
// slot-0 sentinel behaves like a lookup miss (ancestor of nothing).
TEST(memory, namemap_is_strict_ancestor_id_form_matches_string) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    const int16_t mainId = nm.encode("main");
    const int16_t s1 = nm.encodePush(mainId, "s1");
    const int16_t s2 = nm.encodePush(s1, "s2");
    const std::string mainN = nm.decode(mainId);
    const std::string s1N   = nm.decode(s1);
    const std::string s2N   = nm.decode(s2);

    // Id form == string form on each ordered pair.
    ASSERT_EQ(nm.isStrictAncestor(mainId, s1), nm.isStrictAncestor(mainN, s1N));
    ASSERT_EQ(nm.isStrictAncestor(mainId, s2), nm.isStrictAncestor(mainN, s2N));
    ASSERT_EQ(nm.isStrictAncestor(s1, s2),     nm.isStrictAncestor(s1N, s2N));
    ASSERT_EQ(nm.isStrictAncestor(s2, mainId), nm.isStrictAncestor(s2N, mainN));
    ASSERT_TRUE(nm.isStrictAncestor(mainId, s2));   // grandparent is strict
    ASSERT_FALSE(nm.isStrictAncestor(s2, mainId));  // descendant is not
    ASSERT_FALSE(nm.isStrictAncestor(s1, s1));      // self is not strict

    // Slot-0 sentinel: an unknown id is an ancestor of nothing and has none.
    ASSERT_FALSE(nm.isStrictAncestor(static_cast<int16_t>(0), s1));
    ASSERT_FALSE(nm.isStrictAncestor(s1, static_cast<int16_t>(0)));
}

// deeperOf(StrSpan,StrSpan) — returns whichever input span is the deeper
// validity, byte-equal to the std::string overload's result, both directions
// and the equal-span case.
TEST(memory, namemap_deeperof_span_matches_string) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    const int16_t mainId = nm.encode("main");
    const int16_t child  = nm.encodePush(mainId, "x");
    const std::string mainN  = nm.decode(mainId);
    const std::string childN = nm.decode(child);

    // Span result byte-equal to string result, either argument order.
    ASSERT_TRUE(gl::equalSpans(
        nm.deeperOf(gl::StrSpan(mainN), gl::StrSpan(childN)),
        gl::StrSpan(nm.deeperOf(mainN, childN))));
    ASSERT_TRUE(gl::equalSpans(
        nm.deeperOf(gl::StrSpan(childN), gl::StrSpan(mainN)),
        gl::StrSpan(nm.deeperOf(childN, mainN))));
    ASSERT_EQ(nm.deeperOf(gl::StrSpan(mainN), gl::StrSpan(childN)).toStdString(),
              childN);   // child is deeper

    // Equal spans return the (byte-equal) first input.
    ASSERT_TRUE(gl::equalSpans(
        nm.deeperOf(gl::StrSpan(childN), gl::StrSpan(childN)),
        gl::StrSpan(childN)));
}

// strictAncestorNames(StrSpan) — span overload returns the byte-identical
// sorted ancestor-name vector as the std::string overload across a three-deep
// chain (leaf has both strict ancestors, root has none) and an un-interned
// name (empty), reading the span only during the synchronous lookup.
TEST(memory, namemap_strict_ancestor_names_span_matches_string) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    const int16_t mainId = nm.encode("main");
    const int16_t s1 = nm.encodePush(mainId, "s1");
    const int16_t s2 = nm.encodePush(s1, "s2");
    const std::string mainN = nm.decode(mainId);
    const std::string s1N   = nm.decode(s1);
    const std::string s2N   = nm.decode(s2);

    // Span form == string form on each chain node.
    ASSERT_TRUE(nm.strictAncestorNames(gl::StrSpan(mainN))
                == nm.strictAncestorNames(mainN));
    ASSERT_TRUE(nm.strictAncestorNames(gl::StrSpan(s1N))
                == nm.strictAncestorNames(s1N));
    ASSERT_TRUE(nm.strictAncestorNames(gl::StrSpan(s2N))
                == nm.strictAncestorNames(s2N));

    // Leaf has both strict ancestors (sorted); root has none.
    ASSERT_EQ(nm.strictAncestorNames(gl::StrSpan(s2N)).size(),
              static_cast<std::size_t>(2));
    ASSERT_TRUE(nm.strictAncestorNames(gl::StrSpan(mainN)).empty());

    // Un-interned name -> empty, same as the string overload.
    const std::string unknown("never_interned_scope");
    ASSERT_TRUE(nm.strictAncestorNames(gl::StrSpan(unknown)).empty());
    ASSERT_TRUE(nm.strictAncestorNames(unknown).empty());
}

// strictAncestorSpans — span-array twin of strictAncestorNames: identical
// count and byte-identical sorted names across a three-deep encodePush chain,
// a flat-minted root, main, and an un-interned name. The spans are read
// before any further mint (I-3).
TEST(memory, strict_ancestor_spans_matches_string_overload) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    const int16_t mainId = nm.encode("main");
    const int16_t a = nm.encodePush(mainId, "(impl1[a])");
    const int16_t b = nm.encodePush(a, "(in[b,N])");
    const int16_t c = nm.encodePush(b, "(in[c,N])");
    const int16_t flat = nm.encode("(in[7,1])");

    const std::string names[] = {
        nm.decode(mainId), nm.decode(a), nm.decode(b), nm.decode(c),
        nm.decode(flat), std::string("never_interned_scope"),
    };
    for (const std::string& name : names) {
        const std::vector<std::string> oracle = nm.strictAncestorNames(name);
        gl::StrSpan out[gl::ExecutionParameters::MAX_SCOPE_DEPTH];
        const int32_t m = nm.strictAncestorSpans(
            gl::StrSpan(name), out,
            gl::ExecutionParameters::MAX_SCOPE_DEPTH - 1);
        ASSERT_EQ(static_cast<std::size_t>(m), oracle.size());
        for (int32_t i = 0; i < m; ++i) {
            ASSERT_TRUE(gl::equalSpans(out[i],
                gl::StrSpan(oracle[static_cast<std::size_t>(i)])));
        }
    }

    // Depth pins: main / flat root / unknown -> 0; the three-deep leaf -> 3
    // (main, A, B in lex order — verified byte-wise against the oracle above).
    gl::StrSpan out[gl::ExecutionParameters::MAX_SCOPE_DEPTH];
    ASSERT_EQ(nm.strictAncestorSpans(gl::StrSpan(names[0]), out,
        gl::ExecutionParameters::MAX_SCOPE_DEPTH - 1), 0);
    ASSERT_EQ(nm.strictAncestorSpans(gl::StrSpan(names[4]), out,
        gl::ExecutionParameters::MAX_SCOPE_DEPTH - 1), 0);
    ASSERT_EQ(nm.strictAncestorSpans(gl::StrSpan(names[5]), out,
        gl::ExecutionParameters::MAX_SCOPE_DEPTH - 1), 0);
    ASSERT_EQ(nm.strictAncestorSpans(gl::StrSpan(names[3]), out,
        gl::ExecutionParameters::MAX_SCOPE_DEPTH - 1), 3);
}

// ---------- NameMap::parse ----------
TEST(memory, namemap_parse_root_returns_empty) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    const auto stack = nm.parse("main");
    ASSERT_EQ(stack.size(), static_cast<std::size_t>(0));
}

TEST(memory, namemap_parse_two_levels) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
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
    ASSERT_EQ(m.contradictionTheoremId, 0);
    ASSERT_EQ(m.recursionHypothesisId,  0);
}

TEST(memory, memory_default_or_state_is_empty) {
    gl::Memory m;
    ASSERT_EQ(m.orBookkeeping.count(), 0);
    ASSERT_EQ(m.orDisjunctCount.count(), 0);
    ASSERT_EQ(m.intToBeProved.count(), 0);
    ASSERT_EQ(m.intEncodedStatements.size(), static_cast<std::size_t>(0));
    ASSERT_EQ(m.deltaNumberStatements,  0);
}

TEST(memory, memory_parent_chain_walk_to_root) {
    // Build a 3-deep LB chain and walk parent pointers back to root.
    gl::Memory root;
    gl::Memory mid;
    gl::Memory leaf;
    mid.parentMemory  = &root;
    leaf.parentMemory = &mid;

    const gl::Memory* p = &leaf;
    int depth = 0;
    while (p->parentMemory != nullptr) {
        p = p->parentMemory;
        ++depth;
    }
    ASSERT_EQ(depth, 2);                           // leaf → mid → root
    ASSERT_EQ(p, &root);                           // landed at root sentinel
    ASSERT_TRUE(p->exprKey().empty());               // root sentinel
}

// exprKey is statified: stored as a 4-byte id interned in the never-deloaded
// analyzer-wide skeletonInterner(), recovered byte-identically by exprKey().
TEST(memory, exprkey_interns_and_round_trips_through_skeleton) {
    gl::Memory m;
    ASSERT_TRUE(m.exprKey().empty());                       // id-0 sentinel
    m.setExprKey("(in2[i0,i1,s])");
    ASSERT_EQ(m.exprKey(), std::string("(in2[i0,i1,s])"));  // byte-identical
    // A second LB with the same key interns to the SAME id (dedup).
    gl::Memory m2;
    m2.setExprKey("(in2[i0,i1,s])");
    ASSERT_EQ(m2.exprKey(), std::string("(in2[i0,i1,s])"));
    // The empty string maps back to the id-0 sentinel.
    m.setExprKey("");
    ASSERT_TRUE(m.exprKey().empty());
    // The shared analyzer-wide skeleton table holds the interned key, and a
    // lookup of that key resolves to m2's stored id.
    ASSERT_TRUE(gl::skeletonInterner().count() >= 1);
    const std::string key = "(in2[i0,i1,s])";
    ASSERT_EQ(gl::skeletonInterner().lookup(key), m2.exprKeyId);
}

// exprKeyView is the zero-copy span twin of exprKey (C1 _firing_check): the
// interned bytes with no heap std::string, an EMPTY span for the id-0 sentinel.
TEST(memory, expr_key_view_matches_string) {
    gl::Memory m;
    // id-0 sentinel: empty span, matching exprKey()'s empty string.
    ASSERT_EQ(m.exprKeyView().len, 0);
    ASSERT_TRUE(m.exprKey().empty());
    // Several keys: the span bytes are byte-identical to exprKey().
    const char* keys[] = { "(in2[i0,i1,s])", "(anchor[x])", "(=[a,b])" };
    for (const char* k : keys) {
        m.setExprKey(k);
        const std::string s = m.exprKey();
        const gl::StrSpan v = m.exprKeyView();
        ASSERT_EQ(v.len, static_cast<int32_t>(s.size()));
        ASSERT_TRUE(gl::equalSpans(gl::StrSpan(s), v));
    }
    // Back to empty -> empty span again.
    m.setExprKey("");
    ASSERT_EQ(m.exprKeyView().len, 0);
}

// int16SetKey byte helpers (C1 _firing_check): count / id peeks off a
// serialized Int16SetKey, and int16SetKeyLexCompare == std::set<int16_t>::
// operator< (SIGNED element lex, shorter-is-prefix) — NOT the count-prefixed
// byte order, the R1 candidate-enumeration-order trap.
TEST(memory, int16_set_key_helpers_match_std_set) {
    auto enc = [](const std::set<int16_t>& s) {
        gl::Int16SetKey k;
        k.ids.assign(s.begin(), s.end());   // ascending == std::set order
        return gl::Codec<gl::Int16SetKey>::encode(k);
    };
    const std::vector<std::set<int16_t>> sets = {
        {}, {5}, {1, 2, 3}, {-3, -1, 7}, {-32768, 0, 32767}, {2, 4}, {2, 4, 6} };
    // count + id peeks reproduce the set's ascending iteration.
    for (const std::set<int16_t>& s : sets) {
        const std::string bytes = enc(s);
        const gl::StrSpan span(bytes);
        ASSERT_EQ(gl::int16SetKeyCount(span), static_cast<int16_t>(s.size()));
        int idx = 0;
        for (int16_t e : s) {
            ASSERT_EQ(gl::int16SetKeyIdAt(span, idx), e);
            ++idx;
        }
    }
    // lex-compare sign == std::set<int16_t>::operator< over every ordered pair.
    for (const std::set<int16_t>& a : sets) {
        const std::string ba = enc(a);
        for (const std::set<int16_t>& b : sets) {
            const std::string bb = enc(b);
            const int c =
                gl::int16SetKeyLexCompare(gl::StrSpan(ba), gl::StrSpan(bb));
            if (a < b)      { ASSERT_TRUE(c < 0); }
            else if (b < a) { ASSERT_TRUE(c > 0); }
            else            { ASSERT_EQ(c, 0); }
        }
    }
}

// mailInterner() is the process-wide, never-deloaded GLOBAL mail string interner
// (on mailMemory()): one int32 id per distinct string, stable for the run, the
// cross-LB carrier id-space. Mirrors the skeletonInterner contract but on the mail
// pool. See mailInterner() in memory.hpp.
TEST(memory, mail_interner_interns_dedups_and_round_trips) {
    gl::ColdStringTable& t = gl::mailInterner();
    const std::string a = "(in2[7,10,3])_mail_interner_test";
    const std::string b = "main_mail_interner_test_scope";
    const int32_t ia = t.intern(a);
    const int32_t ib = t.intern(b);
    ASSERT_TRUE(ia != 0 && ib != 0);          // 0 is the absent/empty sentinel
    ASSERT_NE(ia, ib);                         // distinct strings -> distinct ids
    ASSERT_EQ(t.intern(a), ia);               // re-intern dedups to the same id
    ASSERT_EQ(t.lookup(a), ia);               // non-minting probe finds it
    ASSERT_EQ(t.decodeString(ia), a);         // id -> bytes round-trips
    ASSERT_EQ(t.decodeString(ib), b);
    const std::string absent = "(absent from mail interner)_xyzzy";
    ASSERT_EQ(t.lookup(absent), 0);           // absent -> 0
    ASSERT_EQ(&gl::mailInterner(), &t);       // analyzer-wide singleton
}

// Codec<IntMailStatementKey> round-trips the id triple (originalId, validityId,
// ascending levels) through its canonical byte layout. See mail_types.hpp.
TEST(memory, int_mail_statement_key_codec_round_trips) {
    gl::IntMailStatementKey k;
    k.originalId = 12345;
    k.validityId = 1;                 // e.g. the "main" id
    k.levels = { 0, 2, 5, 9 };
    const std::string bytes = gl::Codec<gl::IntMailStatementKey>::encode(k);
    const gl::IntMailStatementKey back =
        gl::Codec<gl::IntMailStatementKey>::decode(gl::StrSpan(bytes));
    ASSERT_TRUE(back == k);
    // Empty level set is valid (the compact-implication deposit ships empty).
    gl::IntMailStatementKey k2;
    k2.originalId = 7; k2.validityId = 42; k2.levels = {};
    const std::string bytes2 = gl::Codec<gl::IntMailStatementKey>::encode(k2);
    ASSERT_TRUE(
        gl::Codec<gl::IntMailStatementKey>::decode(gl::StrSpan(bytes2)) == k2);
    // Distinct level lists -> distinct bytes (multiplicity preserved).
    gl::IntMailStatementKey k3 = k;
    k3.levels = { 0, 2, 5 };
    ASSERT_NE(gl::Codec<gl::IntMailStatementKey>::encode(k3), bytes);
}

// ---------- HashMemory::clear ----------
TEST(memory, hashmemory_clear_after_populate) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::HashMemory hm(&lb, &d);
    int16_t buf[2] = {1, 2};
    // encodedMap is now the cold blob map: NormKey -> run of LMVs.
    gl::NormKey nk{ 1, std::vector<int16_t>(buf, buf + 2) };
    hm.encodedMap.assignRun(nk,
        std::vector<gl::LocalMemoryValue>{ gl::LocalMemoryValue{} });
    // D-72: the owner maps are cold blob maps too; assignRun one key (one
    // OwnerSet blob, run-length-1) for the count checks.
    gl::NormKey nkOwner{ 1, std::vector<int16_t>(buf, buf + 2) };
    hm.normalizedEncodedKeys.assignRun(nkOwner,
        std::vector<gl::OwnerSet>{ gl::OwnerSet{} });
    hm.maxKeyLength = 5;
    ASSERT_EQ(hm.encodedMap.count(),            1);
    ASSERT_EQ(hm.normalizedEncodedKeys.count(), 1);
    hm.clear();
    ASSERT_EQ(hm.encodedMap.count(),            0);
    ASSERT_EQ(hm.normalizedEncodedKeys.count(), 0);
    ASSERT_EQ(hm.maxKeyLength,                 static_cast<int16_t>(0));
}

TEST(memory, hashmemory_clear_preserves_struct_validity) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::HashMemory hm(&lb, &d);
    hm.clear();          // clear on already-empty
    hm.clear();          // double clear
    // Struct still valid for re-use: insert a fresh entry.
    int16_t buf[1] = {7};
    gl::NormKey nk{ 1, std::vector<int16_t>(buf, buf + 1) };
    hm.normalizedEncodedKeys.assignRun(nk,
        std::vector<gl::OwnerSet>{ gl::OwnerSet{} });
    ASSERT_EQ(hm.normalizedEncodedKeys.count(), 1);
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

// ---------- EquivalenceClass levels map ----------
TEST(memory, equivalenceclass_levels_map_carries_entries) {
    gl::Memory m;
    gl::EquivalenceClass cls;
    cls.setMembersFromNames({ "a", "b", "c" }, m.nameMap);
    const int16_t idA = m.nameMap.encode("a");
    const int16_t idB = m.nameMap.encode("b");
    const int16_t idC = m.nameMap.encode("c");
    cls.intEqualityLevelsMap[gl::packEqPairKey(idA, idB)] = { 0 };
    cls.intEqualityLevelsMap[gl::packEqPairKey(idB, idC)] = { 1, 2 };
    ASSERT_EQ(cls.memberIds.size(),            static_cast<std::size_t>(3));
    ASSERT_EQ(cls.intEqualityLevelsMap.size(), static_cast<std::size_t>(2));
    ASSERT_EQ(cls.intEqualityLevelsMap.at(gl::packEqPairKey(idB, idA)).count(0),
              static_cast<std::size_t>(1));
    ASSERT_EQ(cls.intEqualityLevelsMap.at(gl::packEqPairKey(idC, idB)).size(),
              static_cast<std::size_t>(2));
}

TEST(memory, equivalenceclass_default_then_assign) {
    gl::Memory m;
    gl::EquivalenceClass cls;
    cls.setMembersFromNames({ "x", "y" }, m.nameMap);
    ASSERT_EQ(cls.memberIds.size(), static_cast<std::size_t>(2));
    cls.memberIds.clear();
    ASSERT_EQ(cls.memberIds.size(), static_cast<std::size_t>(0));
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
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
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
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
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

// Span encodeExpression twin: byte-identical IntEncodedExpr to the
// EncodedExpression path across name / negation / arity / iteration / u_ /
// anchor / hypo shapes. Shared NameMap (struct mints, span looks up) so equal
// raw ids prove the spans resolve to identical strings in identical slots.
TEST(memory, encodeexpression_span_twin_byte_identical) {
    struct Case { const char* expr; const char* validity; };
    const Case cases[] = {
        {"(zero[])", "main"},
        {"(in3[1,2,q])", "main"},
        {"(=[a,b])", "main"},
        {"(AnchorPeano[n])", "main_boundary_hypo_x"},
        {"!(=[a,b])", "main"},
        {"(in2[u_a,b])", "main"},
        {"(p[u_it_0_lev_1_x,y])", "main"},
        {"(q[it_3_lev_2_z])", "main"},
    };
    for (const Case& c : cases) {
        NameMapRig nmRig;
        gl::NameMap& nm = nmRig.nm;
        const std::string expr = c.expr;
        const std::string validity = c.validity;
        const gl::IntEncodedExpr a =
            gl::encodeExpression(gl::EncodedExpression(expr, validity), nm);
        const gl::IntEncodedExpr b =
            gl::encodeExpression(gl::StrSpan(expr), gl::StrSpan(validity), nm);
        ASSERT_EQ(a.nameId,      b.nameId);
        ASSERT_EQ(a.negation,    b.negation);
        ASSERT_EQ(a.arity,       b.arity);
        ASSERT_EQ(a.maxIteration, b.maxIteration);
        ASSERT_EQ(a.originalId,  b.originalId);
        ASSERT_EQ(a.validityId,  b.validityId);
        ASSERT_EQ(a.isHypo,      b.isHypo);
        ASSERT_EQ(a.isAnchor,    b.isAnchor);
        for (int16_t i = 0; i < a.arity; ++i) {
            ASSERT_EQ(a.argId[i],          b.argId[i]);
            ASSERT_EQ(a.argUnchangeable[i], b.argUnchangeable[i]);
            ASSERT_EQ(a.argIteration[i],   b.argIteration[i]);
            ASSERT_EQ(a.argLevPlus1[i],    b.argLevPlus1[i]);
            ASSERT_EQ(a.argFullId[i],      b.argFullId[i]);
        }
    }
}

// ---------- LocalMemoryValue id form ----------
// Install-shaped round trip: fields encode in the LB's rule interner and
// decode back; remainingArgIds storage is decoded-lex sorted.
TEST(memory, localmemoryvalue_id_round_trip) {
    gl::Memory m;
    gl::LocalMemoryValue lmv;
    lmv.valueId = m.ruleInterner.encode("(q[1])");
    lmv.isMarker = false;
    lmv.originalImplicationId = m.ruleInterner.encode(">[1,1](p[u_1])(q[u_1])");
    lmv.justification = gl::ruleJustificationFromString("implication");
    std::set<std::string> rem{"b", "a"};
    lmv.remainingArgIds = gl::encodeValueSetSorted(rem, m.ruleInterner);
    lmv.keyIds = encodeValueVectorOracle({"(p[1])"}, m.ruleInterner);
    lmv.validityId = gl::NameMap::MAIN_ID;

    ASSERT_EQ(m.ruleInterner.decode(lmv.valueId), std::string("(q[1])"));
    ASSERT_EQ(m.ruleInterner.decode(lmv.originalImplicationId),
              std::string(">[1,1](p[u_1])(q[u_1])"));
    ASSERT_TRUE(lmv.justification == gl::RuleJustification::implication);
    const std::set<std::string> remBack =
        gl::decodeValueVectorToSet(lmv.remainingArgIds, m.ruleInterner);
    ASSERT_TRUE(remBack == rem);
    // Decoded-lex storage order: "a" before "b" regardless of mint order.
    ASSERT_EQ(m.ruleInterner.decode(lmv.remainingArgIds[0]), std::string("a"));
    ASSERT_EQ(m.ruleInterner.decode(lmv.remainingArgIds[1]), std::string("b"));
    ASSERT_EQ(m.ruleInterner.decode(lmv.keyIds[0]), std::string("(p[1])"));
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

// ---------- recordUSignature / ownerSetUSatisfied (u_ literal prune) ----------
// D-120: the four normalizedEncoded* maps are ignoreU=false
// (u_ literals erased), so a structurally-matched request may be doomed by a u_
// literal it can never satisfy. recordUSignature caches each owner's u_ literal
// signature at insert; ownerSetUSatisfied checks it at request generation as a
// sound necessary condition for firing. recordUSignature reads literals via the
// non-minting NameMap::lookup, so a test must intern the literal first (the real
// flow interns it in the preceding ignoreU=true key build) — encodeExpression is
// used as that interning step here.

TEST(memory, record_usignature_loose_when_no_u_args) {
    // No u_ (unchangeable) args -> no signature, owner-set flagged loose
    // (the "0 u_, nothing to check" fast path).
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    gl::OwnerSet os;
    std::vector<gl::EncodedExpression> encList = {
        gl::EncodedExpression("(in[a,b])", "main")
    };
    gl::recordUSignature(os, encList, nm);
    ASSERT_TRUE(os.hasLooseOwner);
    ASSERT_TRUE(os.uSignatures.empty());
}

TEST(memory, record_usignature_unknown_literal_flags_loose) {
    // A u_ arg whose literal is NOT yet interned -> lookup returns 0 -> the key
    // is conservatively flagged loose (no signature recorded). Sound: the request
    // is kept, never wrongly pruned.
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    gl::OwnerSet os;
    std::vector<gl::EncodedExpression> encList = {
        gl::EncodedExpression("(in3[a,b,u_p])", "main")   // "p" never interned
    };
    gl::recordUSignature(os, encList, nm);
    ASSERT_TRUE(os.hasLooseOwner);
    ASSERT_TRUE(os.uSignatures.empty());
}

TEST(memory, record_usignature_tight_records_slot_and_literal) {
    // A u_ arg with an interned literal records (linear-arg-slot, lookup(arg[1]))
    // — the same id encodeExpression stores in argFullId, i.e. what the firing
    // gate matches.
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    gl::OwnerSet os;
    gl::EncodedExpression rule("(in3[a,b,u_p])", "main");
    gl::encodeExpression(rule, nm);                // intern arg[1]s (ignoreU=true analog)
    std::vector<gl::EncodedExpression> encList = { rule };
    gl::recordUSignature(os, encList, nm);
    ASSERT_FALSE(os.hasLooseOwner);
    ASSERT_EQ((int)os.uSignatures.size(), 1);
    const std::vector<std::pair<int16_t,int16_t>>& sig = *os.uSignatures.begin();
    ASSERT_EQ((int)sig.size(), 1);                 // only the u_p slot
    ASSERT_EQ((int)sig[0].first, 2);               // 3rd flattened arg slot
    ASSERT_EQ(sig[0].second, nm.lookup("p"));      // literal == argFullId
}

TEST(memory, record_usignature_dedup_identical_signatures) {
    // Two owners with the same u_ signature collapse to one entry (deduped set).
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    gl::OwnerSet os;
    gl::EncodedExpression a("(in3[a,b,u_p])", "main");
    gl::EncodedExpression b("(in3[c,d,u_p])", "main");
    gl::encodeExpression(a, nm);
    gl::encodeExpression(b, nm);
    std::vector<gl::EncodedExpression> la = { a };
    std::vector<gl::EncodedExpression> lb = { b };
    gl::recordUSignature(os, la, nm);
    gl::recordUSignature(os, lb, nm);              // same slot+literal -> dedup
    ASSERT_EQ((int)os.uSignatures.size(), 1);
    ASSERT_FALSE(os.hasLooseOwner);
}

TEST(memory, record_usignature_int_arena_matches_string_oracle) {
    // Byte-twin for the heap-free int-form recordUSignature: the gen-scratch
    // arena-run build + os.uSignatures.emplace(first,last) must yield the
    // identical OwnerSet as the retained string oracle for a shared premise
    // set. Args interned via encodeExpression so the string form's lookup(arg)
    // equals the int form's argFullId (no unknown-literal loose fallback).
    // Two premises exercise cross-expression slot accumulation.
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;

    gl::EncodedExpression e0("(in3[a,b,u_p])", "main");   // u_p at slot 2
    gl::EncodedExpression e1("(in3[c,u_q,f])", "main");   // u_q at slot 4
    gl::IntEncodedExpr i0 = gl::encodeExpression(e0, nm); // interns args
    gl::IntEncodedExpr i1 = gl::encodeExpression(e1, nm);

    // String oracle path (retained).
    gl::OwnerSet osStr;
    std::vector<gl::EncodedExpression> encListStr = { e0, e1 };
    gl::recordUSignature(osStr, encListStr, nm);

    // Int arena-run path (statified).
    gl::OwnerSet osInt;
    gl::IntEncodedExpr encListInt[2] = { i0, i1 };
    gl::recordUSignature(osInt, encListInt, static_cast<int16_t>(2));

    ASSERT_EQ(osStr.hasLooseOwner, osInt.hasLooseOwner);
    ASSERT_FALSE(osInt.hasLooseOwner);
    ASSERT_TRUE(osStr.uSignatures == osInt.uSignatures);
    ASSERT_EQ((int)osInt.uSignatures.size(), 1);
    const std::vector<std::pair<int16_t, int16_t>>& sig = *osInt.uSignatures.begin();
    ASSERT_EQ((int)sig.size(), 2);
    ASSERT_EQ((int)sig[0].first, 2);
    ASSERT_EQ(sig[0].second, nm.lookup("p"));
    ASSERT_EQ((int)sig[1].first, 4);
    ASSERT_EQ(sig[1].second, nm.lookup("q"));
}

TEST(memory, record_usignature_mixed_loose_and_tight_flags_loose) {
    // A loose owner (no u_ args) alongside a tight one flags the key loose, so
    // ownerSetUSatisfied skips it (a loose owner can always fire).
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    gl::OwnerSet os;
    gl::EncodedExpression tight("(in3[a,b,u_p])", "main");
    gl::encodeExpression(tight, nm);
    std::vector<gl::EncodedExpression> lt = { tight };
    std::vector<gl::EncodedExpression> ll = { gl::EncodedExpression("(in3[a,b,c])", "main") };
    gl::recordUSignature(os, lt, nm);
    gl::recordUSignature(os, ll, nm);
    ASSERT_TRUE(os.hasLooseOwner);
    ASSERT_EQ((int)os.uSignatures.size(), 1);
}

TEST(memory, owner_set_u_satisfied_loose_keeps_unconditionally) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    gl::OwnerSet os;
    os.hasLooseOwner = true;
    gl::IntEncodedExpr req = gl::encodeExpression(gl::EncodedExpression("(in3[1,2,q])", "main"), nm);
    const gl::IntEncodedExpr* exprs[1] = { &req };
    ASSERT_TRUE(gl::ownerSetUSatisfied(os, exprs, 1));   // skip -- could fire
}

TEST(memory, owner_set_u_satisfied_empty_keeps) {
    // No signatures and not loose -> no basis to prune -> keep (sound).
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    gl::OwnerSet os;
    gl::IntEncodedExpr req = gl::encodeExpression(gl::EncodedExpression("(in3[1,2,p])", "main"), nm);
    const gl::IntEncodedExpr* exprs[1] = { &req };
    ASSERT_TRUE(gl::ownerSetUSatisfied(os, exprs, 1));
}

TEST(memory, owner_set_u_satisfied_match_keeps_mismatch_prunes) {
    // One owner requires "p" at slot 2. A request carrying "p" is kept; one
    // carrying "q" is pruned (no owner can fire it).
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    gl::OwnerSet os;
    gl::EncodedExpression rule("(in3[a,b,u_p])", "main");
    gl::encodeExpression(rule, nm);
    std::vector<gl::EncodedExpression> lr = { rule };
    gl::recordUSignature(os, lr, nm);
    ASSERT_EQ((int)os.uSignatures.size(), 1);

    gl::IntEncodedExpr ok  = gl::encodeExpression(gl::EncodedExpression("(in3[1,2,p])", "main"), nm);
    gl::IntEncodedExpr bad = gl::encodeExpression(gl::EncodedExpression("(in3[1,2,q])", "main"), nm);
    const gl::IntEncodedExpr* okE[1]  = { &ok };
    const gl::IntEncodedExpr* badE[1] = { &bad };
    ASSERT_TRUE(gl::ownerSetUSatisfied(os, okE, 1));
    ASSERT_FALSE(gl::ownerSetUSatisfied(os, badE, 1));
}

// ownerSetUSatisfied(OwnerSetBlob) (the no-decode twin) must agree with the
// const-OwnerSet& overload on the loose, single-signature, and MULTI-signature
// cases -- the last verifies the byte cursor advances past every pair of a
// non-matching signature so a later matching signature is still found.
TEST(memory, ownerset_blob_u_satisfied_matches_value) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;

    // Single tight owner requiring "p" at slot 2.
    gl::OwnerSet os;
    gl::EncodedExpression rule("(in3[a,b,u_p])", "main");
    gl::encodeExpression(rule, nm);
    std::vector<gl::EncodedExpression> lr = { rule };
    gl::recordUSignature(os, lr, nm);
    ASSERT_FALSE(os.hasLooseOwner);
    std::vector<char> buf = gl::Codec<gl::OwnerSet>::serialize(os);
    const gl::OwnerSetBlob v{ buf.data(), static_cast<int32_t>(buf.size()) };

    gl::IntEncodedExpr okx = gl::encodeExpression(gl::EncodedExpression("(in3[1,2,p])", "main"), nm);
    gl::IntEncodedExpr badx = gl::encodeExpression(gl::EncodedExpression("(in3[1,2,q])", "main"), nm);
    const gl::IntEncodedExpr* okE[1]  = { &okx };
    const gl::IntEncodedExpr* badE[1] = { &badx };
    ASSERT_TRUE(gl::ownerSetUSatisfied(v, okE, 1) == gl::ownerSetUSatisfied(os, okE, 1));
    ASSERT_TRUE(gl::ownerSetUSatisfied(v, badE, 1) == gl::ownerSetUSatisfied(os, badE, 1));
    ASSERT_TRUE(gl::ownerSetUSatisfied(v, okE, 1));     // keep
    ASSERT_FALSE(gl::ownerSetUSatisfied(v, badE, 1));   // prune

    // Loose owner-set: both keep unconditionally (reads only the hasLooseOwner byte).
    gl::OwnerSet loose;
    loose.hasLooseOwner = true;
    loose.partitionIds.insert(1);
    std::vector<char> lbuf = gl::Codec<gl::OwnerSet>::serialize(loose);
    const gl::OwnerSetBlob lv{ lbuf.data(), static_cast<int32_t>(lbuf.size()) };
    ASSERT_TRUE(gl::ownerSetUSatisfied(lv, badE, 1));

    // Two tight signatures with different slots. Set order puts [(0,x)] before
    // [(2,p)], so a request matching the SECOND ("p" at slot 2) forces the byte
    // walk to advance past the first signature's pairs.
    gl::OwnerSet os2;
    gl::EncodedExpression r1("(in3[a,b,u_p])", "main");
    gl::EncodedExpression r2("(in3[u_x,b,c])", "main");
    gl::encodeExpression(r1, nm);
    gl::encodeExpression(r2, nm);
    std::vector<gl::EncodedExpression> l1 = { r1 };
    std::vector<gl::EncodedExpression> l2 = { r2 };
    gl::recordUSignature(os2, l1, nm);
    gl::recordUSignature(os2, l2, nm);
    ASSERT_FALSE(os2.hasLooseOwner);
    ASSERT_EQ(static_cast<int>(os2.uSignatures.size()), 2);
    std::vector<char> b2 = gl::Codec<gl::OwnerSet>::serialize(os2);
    const gl::OwnerSetBlob v2{ b2.data(), static_cast<int32_t>(b2.size()) };

    gl::IntEncodedExpr keep2  = gl::encodeExpression(gl::EncodedExpression("(in3[1,2,p])", "main"), nm);
    gl::IntEncodedExpr prune2 = gl::encodeExpression(gl::EncodedExpression("(in3[1,2,3])", "main"), nm);
    const gl::IntEncodedExpr* keep2E[1]  = { &keep2 };
    const gl::IntEncodedExpr* prune2E[1] = { &prune2 };
    ASSERT_TRUE(gl::ownerSetUSatisfied(v2, keep2E, 1) == gl::ownerSetUSatisfied(os2, keep2E, 1));
    ASSERT_TRUE(gl::ownerSetUSatisfied(v2, prune2E, 1) == gl::ownerSetUSatisfied(os2, prune2E, 1));
    ASSERT_TRUE(gl::ownerSetUSatisfied(v2, keep2E, 1));    // matches the 2nd sig
    ASSERT_FALSE(gl::ownerSetUSatisfied(v2, prune2E, 1));  // matches neither
}

// mergeOwnerRecord RMWs the whole OwnerSet into a cold owner-set map: two merges
// into the same key accumulate both partition ids under one key; the u_ signature
// is recorded (loose for a no-u_ premise).
TEST(memory, merge_owner_record_rmw) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::TypedColdBlobMap<gl::NormKey, gl::OwnerSet> map(&lb, &d);

    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    int16_t buf[2] = { 1, 2 };
    gl::NormKey key{ 1, std::vector<int16_t>(buf, buf + 2) };
    std::vector<gl::EncodedExpression> enc = {
        gl::EncodedExpression("(in[a,b])", "main") };

    gl::ExpressionAnalyzer::mergeOwnerRecord(map, key, 111, enc, nm);
    gl::ExpressionAnalyzer::mergeOwnerRecord(map, key, 222, enc, nm);

    const int32_t id = map.lookup(key);
    ASSERT_TRUE(id != 0);
    ASSERT_EQ(map.count(), 1);                        // still ONE key
    const gl::OwnerSet os = map.recordAt(id, 0);
    ASSERT_EQ(static_cast<int>(os.partitionIds.size()), 2);
    ASSERT_TRUE(os.partitionIds.count(111) == 1);
    ASSERT_TRUE(os.partitionIds.count(222) == 1);
    ASSERT_TRUE(os.hasLooseOwner);                    // (in[a,b]) has no u_ args
}

// insertRemainingArgsNormKey RMWs a NormKey into the cold secondary index: two
// distinct NormKeys under one arg-set accumulate in ONE sorted-unique run; a
// duplicate insert is a no-op; a different arg-set is a separate key.
TEST(memory, insert_remaining_args_norm_key_rmw) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::TypedColdBlobMap<gl::Int16SetKey, gl::NormKey> map(&lb, &d);
    gl::LbArena scratch(&g);   // separate RMW scratch arena (never the map's own)

    const std::set<int16_t> argSetA{ 1, 2 };
    const std::set<int16_t> argSetB{ 3 };
    int16_t a1[1] = { 10 };
    int16_t a2[1] = { 20 };
    const gl::NormKey nk1{ 1, std::vector<int16_t>(a1, a1 + 1) };
    const gl::NormKey nk2{ 1, std::vector<int16_t>(a2, a2 + 1) };

    // Insert out of order; the run must come back sorted by (numberExpressions, data).
    gl::ExpressionAnalyzer::insertRemainingArgsNormKey(map, argSetA, nk2, scratch);
    gl::ExpressionAnalyzer::insertRemainingArgsNormKey(map, argSetA, nk1, scratch);
    gl::ExpressionAnalyzer::insertRemainingArgsNormKey(map, argSetA, nk1, scratch);  // duplicate -> no-op
    gl::ExpressionAnalyzer::insertRemainingArgsNormKey(map, argSetB, nk1, scratch);

    ASSERT_EQ(map.count(), 2);                                       // two arg-set keys
    const int32_t idA = map.lookup(
        gl::Int16SetKey{ std::vector<int16_t>(argSetA.begin(), argSetA.end()) });
    ASSERT_TRUE(idA != 0);
    const std::vector<gl::NormKey> runA = map.recordsAt(idA);
    ASSERT_EQ(static_cast<int>(runA.size()), 2);                     // deduped to nk1+nk2
    ASSERT_EQ(runA[0].data[0], static_cast<int16_t>(10));            // sorted ascending by data
    ASSERT_EQ(runA[1].data[0], static_cast<int16_t>(20));
    const int32_t idB = map.lookup(
        gl::Int16SetKey{ std::vector<int16_t>(argSetB.begin(), argSetB.end()) });
    ASSERT_TRUE(idB != 0);
    ASSERT_EQ(static_cast<int>(map.recordsAt(idB).size()), 1);
}

// ownerKeyAccepts: a present key at a comparable (main) scope with a loose owner
// is kept; an absent key is a lookup miss (false) before any predicate runs.
TEST(memory, owner_key_accepts_lookup_and_prune) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::TypedColdBlobMap<gl::NormKey, gl::OwnerSet> map(&lb, &d);

    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    int16_t kb[2] = { 5, 6 };
    gl::NormKey key{ 1, std::vector<int16_t>(kb, kb + 2) };
    std::vector<gl::EncodedExpression> enc = {
        gl::EncodedExpression("(in[a,b])", "main") };
    gl::ExpressionAnalyzer::mergeOwnerRecord(
        map, key, gl::makePartitionId(3, gl::NameMap::MAIN_ID), enc, nm);

    gl::IntEncodedExpr req =
        gl::encodeExpression(gl::EncodedExpression("(in[a,b])", "main"), nm);
    const gl::IntEncodedExpr* exprs[1] = { &req };

    // Present key, main scope, loose owner -> kept.
    ASSERT_TRUE(gl::ExpressionAnalyzer::ownerKeyAccepts(map, kb, 2, nm, exprs, 1));
    // Absent key -> false (lookup miss, no predicate run).
    int16_t miss[2] = { 9, 9 };
    ASSERT_FALSE(
        gl::ExpressionAnalyzer::ownerKeyAccepts(map, miss, 2, nm, exprs, 1));
}

TEST(memory, owner_set_u_satisfied_any_owner_matches) {
    // Two distinct tight owners (require "p" vs "r"); a request matching EITHER
    // is kept; one matching NEITHER is pruned.
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    gl::OwnerSet os;
    gl::EncodedExpression rp("(in3[a,b,u_p])", "main");
    gl::EncodedExpression rr("(in3[a,b,u_r])", "main");
    gl::encodeExpression(rp, nm);
    gl::encodeExpression(rr, nm);
    std::vector<gl::EncodedExpression> lp = { rp };
    std::vector<gl::EncodedExpression> lr = { rr };
    gl::recordUSignature(os, lp, nm);
    gl::recordUSignature(os, lr, nm);
    ASSERT_EQ((int)os.uSignatures.size(), 2);

    gl::IntEncodedExpr p = gl::encodeExpression(gl::EncodedExpression("(in3[1,2,p])", "main"), nm);
    gl::IntEncodedExpr s = gl::encodeExpression(gl::EncodedExpression("(in3[1,2,s])", "main"), nm);
    const gl::IntEncodedExpr* pE[1] = { &p };
    const gl::IntEncodedExpr* sE[1] = { &s };
    ASSERT_TRUE(gl::ownerSetUSatisfied(os, pE, 1));     // matches the "p" owner
    ASSERT_FALSE(gl::ownerSetUSatisfied(os, sE, 1));    // neither owner
}

TEST(memory, owner_set_u_satisfied_multi_premise_slot_alignment) {
    // Two premises; the u_ literal sits in the 2nd premise. The linear-arg-slot
    // must span the flattened request -- pins the cross-premise offset.
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    gl::OwnerSet os;
    gl::EncodedExpression e0("(in[a,b])", "main");        // slots 0,1 (changeable)
    gl::EncodedExpression e1("(in3[c,d,u_p])", "main");   // slots 2,3,4 ; u_p at slot 4
    gl::encodeExpression(e0, nm);
    gl::encodeExpression(e1, nm);
    std::vector<gl::EncodedExpression> rule = { e0, e1 };
    gl::recordUSignature(os, rule, nm);
    ASSERT_EQ((int)os.uSignatures.size(), 1);
    const std::vector<std::pair<int16_t,int16_t>>& sig = *os.uSignatures.begin();
    ASSERT_EQ((int)sig.size(), 1);
    ASSERT_EQ((int)sig[0].first, 4);                      // 5th flattened slot

    gl::IntEncodedExpr p0   = gl::encodeExpression(gl::EncodedExpression("(in[1,2])", "main"), nm);
    gl::IntEncodedExpr okP1 = gl::encodeExpression(gl::EncodedExpression("(in3[3,4,p])", "main"), nm);
    gl::IntEncodedExpr bdP1 = gl::encodeExpression(gl::EncodedExpression("(in3[3,4,q])", "main"), nm);
    const gl::IntEncodedExpr* okE[2] = { &p0, &okP1 };
    const gl::IntEncodedExpr* bdE[2] = { &p0, &bdP1 };
    ASSERT_TRUE(gl::ownerSetUSatisfied(os, okE, 2));
    ASSERT_FALSE(gl::ownerSetUSatisfied(os, bdE, 2));
}

// cloneFactsTemplate — a facts-only template (no hash rules, as produced by the
// status-4 fact load) deep-clones into a fresh single-use CE LB: the fact
// containers and nameMap copy across, hash/derived state resets to default, and
// the clone outlives destruction of the template (a true deep copy — no shared
// heap buffers, no aliased gen-arena interior pointers).
TEST(memory, clone_facts_template_deep_copy_outlives_template) {
    gl::Memory* tmpl = new gl::Memory();

    // Mirror a single status-4 fact load by hand.
    const gl::EncodedExpression e("(in2[i0,i1,s])", "main");
    const int16_t oid = tmpl->nameMap.encode("(in2[i0,i1,s])");
    const int16_t vid = tmpl->nameMap.encode("main");
    tmpl->intEncodedStatements.push_back(gl::encodeExpression(e, tmpl->nameMap));
    tmpl->intStatementLevelsMap.insertSorted(gl::packStatementKey(oid, vid), 0);
    tmpl->intKnownStatements.insert(
        gl::StatementKey{ oid, vid }, gl::StatementFlags{ true, false, true, true });
    tmpl->level = 0;

    gl::LbStore cloneStore{ &gl::lbMemory(), sizeof(gl::Memory), alignof(gl::Memory) };
    gl::Memory* clone = tmpl->cloneFactsTemplate(cloneStore);

    // Fact state copied across; the registry row decodes back to the fact.
    ASSERT_EQ(clone->intEncodedStatements.size(), static_cast<std::size_t>(1));
    ASSERT_TRUE(gl::decodeExpression(clone->intEncodedStatements[0],
                                     clone->nameMap) == e);
    ASSERT_EQ(clone->nameMap.lookup("(in2[i0,i1,s])"), oid);
    ASSERT_TRUE(clone->intKnownStatements.find(gl::StatementKey{ oid, vid }) != nullptr);
    ASSERT_TRUE(clone->intStatementLevelsMap.lookup(
        gl::packStatementKey(oid, vid)) != 0);
    ASSERT_EQ(clone->level, 0);

    // Hash / derived state reset; fresh empty arena; single-use leaf shape.
    ASSERT_TRUE(clone->overallHashMemory.encodedMap.empty());
    ASSERT_TRUE(clone->isActive);
    ASSERT_EQ(clone->contradictionIndex, -1);

    // Destroy the template; the clone must remain fully valid (deep copy).
    delete tmpl;
    ASSERT_EQ(clone->intEncodedStatements.size(), static_cast<std::size_t>(1));
    ASSERT_TRUE(gl::decodeExpression(clone->intEncodedStatements[0],
                                     clone->nameMap) == e);
    ASSERT_EQ(clone->nameMap.lookup("(in2[i0,i1,s])"), oid);

    cloneStore.destroy(clone);
}

// wipeSubtree — the packed-key statement indexes drop exactly the closed
// scope's rows and keep every other scope's, in lockstep with the
// string-keyed twins (the closedIds predicate of the registry sweeps; low
// 16 key bits carry the validityId).
TEST(memory, wipe_subtree_drops_packed_statement_index_rows) {
    gl::Memory m;

    const std::string closedScope = "main_boundary_w1";
    const gl::EncodedExpression keep("(in[1,2])", "main");
    const gl::EncodedExpression drop("(in[3,4])", closedScope);

    // Mirror a production local-statement deposit by hand.
    auto insertLocal = [&](const gl::EncodedExpression& e) -> int32_t {
        const gl::IntEncodedExpr ie = gl::encodeExpression(e, m.nameMap);
        const int32_t pk = gl::packStatementKey(ie.originalId, ie.validityId);
        m.intEncodedStatements.push_back(ie);
        m.intLocalEncodedStatements.push_back(ie);
        m.intLocalEncodedStatementsSet.mint(pk);
        m.intStatementLevelsMap.insertSorted(pk, 0);
        return pk;
    };
    const int32_t pkKeep = insertLocal(keep);
    const int32_t pkDrop = insertLocal(drop);

    // The wipe takes the closed scope's NameMap validity id (minted above
    // by the drop row's encodeExpression; lookup is the non-minting probe).
    const int16_t closedVid = m.nameMap.lookup(closedScope);
    ASSERT_TRUE(closedVid != 0);
    m.wipeSubtree(closedVid);

    // Closed-scope row gone from both packed indexes.
    ASSERT_FALSE(m.intLocalEncodedStatementsSet.contains(pkDrop));
    ASSERT_EQ(m.intStatementLevelsMap.lookup(pkDrop), 0);

    // Surviving scope intact, levels payload preserved.
    ASSERT_TRUE(m.intLocalEncodedStatementsSet.contains(pkKeep));
    ASSERT_TRUE(m.intStatementLevelsMap.lookup(pkKeep) != 0);
    const std::set<int> keepLevels{ 0 };
    ASSERT_TRUE(gl::coldIntSetAt(m.intStatementLevelsMap,
                                m.intStatementLevelsMap.lookup(pkKeep)) == keepLevels);
}

// ---------------------------------------------------------------------------
// wipeSubtree step 11 — ascending mint-order regression pin
// (D-190).
//
// Step 11 mints every closed validity id into intValidityNamesToFilter, a
// deload-enrolled ColdHashSet whose KeysView facet streams keys in id
// (== mint) order — the mint order is deload-byte-observable. Production
// mints in ascending id order under a byte-identity waiver
// (the historical unordered_set iteration order was collision-history-
// dependent and host-dependent). This test pins the COMMITTED contract:
// minting a closed-id population in ascending order reproduces exactly the
// ascending sequence at decode(1..count) — id order == mint order ==
// KeysView facet order.
// ---------------------------------------------------------------------------
TEST(memory, wipe_step11_mint_order_facet_twin) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 16 << 20, 1 << 18 });
    const int32_t sizes[] = { 1, 2, 4, 7, 8, 9, 16, 33, 64, 100, 512, 1000, 4096 };
    for (const int32_t n : sizes) {
        for (int pattern = 0; pattern < 2; ++pattern) {
            // Representative closed-id population shapes of a subtree wipe:
            // dense = ids 1..n; sparse = every 7th id starting at 3
            // (n values; max 3 + 7*4095 = 28668, inside int16_t).
            std::vector<int16_t> ascending;
            ascending.reserve(static_cast<std::size_t>(n));
            for (int32_t k = 0; k < n; ++k)
                ascending.push_back(static_cast<int16_t>(
                    pattern == 0 ? k + 1 : 3 + 7 * k));
            const std::unordered_set<int16_t> closedIds(ascending.begin(),
                                                        ascending.end());

            // Fresh per-shape arena; the set is declared after it so it
            // destructs first, and the arena destructor returns the blocks.
            gl::LbArena lb(&g);
            gl::DirtyState dirty = gl::DirtyState::Clean;
            gl::ColdHashSet<gl::PodKeyStore<int16_t>> filterSet(&lb, &dirty);

            // The production step-11 shape: ascending id scan gated on
            // closed-set membership (the test's scan ceiling is the
            // population max; ids above it are not members, so the mint
            // sequence is identical to production's nameCount-bounded scan).
            const int16_t hi = ascending.back();
            for (int16_t id = 1; id <= hi; ++id)
                if (closedIds.count(id)) filterSet.mint(id);

            // The committed contract: decode(i) walks the id column in id
            // order, which KeysView streams verbatim into the deload facet
            // — so decode(1..count) == ascending IS the facet byte order.
            ASSERT_EQ(filterSet.count(), n);
            for (int32_t i = 1; i <= filterSet.count(); ++i)
                ASSERT_EQ(filterSet.decode(i),
                          ascending[static_cast<std::size_t>(i - 1)]);
        }
    }
}

// The phase-3 wipe drain sorts pending scope IDS with compareSpans over
// nameMap.decodeView; this pins that id sort against the retained oracle —
// decode every id to an owned std::string, std::sort (byte-lex), map back.
// The ColdHashSet dedups vids and the interner is injective, so the
// comparator is tie-free and the two orders must be element-wise equal.
// Mint order deliberately differs from lex order; nested encodePush
// children and long-shared-prefix names included.
TEST(memory, wipe_drain_id_sort_matches_string_sort) {
    gl::Memory m;
    std::vector<int16_t> ids;
    ids.push_back(m.nameMap.encode("main_boundary_(zz[1])"));
    ids.push_back(m.nameMap.encode("main_boundary_(aa[1])"));
    ids.push_back(m.nameMap.encodePush(ids[0], "(in[z,N])"));
    ids.push_back(m.nameMap.encodePush(ids[1], "(in[a,N])"));
    ids.push_back(m.nameMap.encode("main_boundary_(aa[1])x"));
    ids.push_back(m.nameMap.encode("main_boundary_(aa[2])"));

    // Production comparator: sort the ids under compareSpans(decodeView).
    std::vector<int16_t> byId = ids;
    std::sort(byId.begin(), byId.end(), [&](int16_t a, int16_t b) {
        return gl::compareSpans(m.nameMap.decodeView(a),
                                m.nameMap.decodeView(b)) < 0;
    });

    // Retained oracle (heap allowed in tests): owned-string decode +
    // byte-lex std::sort + map back to ids via the non-minting lookup.
    std::vector<std::string> names;
    names.reserve(ids.size());
    for (const int16_t id : ids) names.push_back(m.nameMap.decode(id));
    std::sort(names.begin(), names.end());
    std::vector<int16_t> byString;
    byString.reserve(names.size());
    for (const std::string& s : names) {
        const int16_t id = m.nameMap.lookup(s);
        ASSERT_TRUE(id != 0);
        byString.push_back(id);
    }

    ASSERT_EQ(byId.size(), byString.size());
    for (std::size_t i = 0; i < byId.size(); ++i)
        ASSERT_EQ(byId[i], byString[i]);
}

// Direct coverage of the wipeSubtree(int16_t) overload: a full sweep over
// statement registries + goal registry keyed by the closed scope's vid.
// Positive: every row/key at the closed scope A and its descendant B is
// gone. Negative (the guarded predicate): the literal-prefix sibling S —
// whose decoded name starts with A's text but NOT with A + "_boundary_" —
// survives, as do all "main" rows. expandedImplications (the one sweep in
// the lbStateInterner id space, span-twin text gate): the closed-scope and
// descendant-scope entries drop, the survivor-scope and
// literal-prefix-no-delimiter entries stay. Step 11: the filter set
// contains exactly {A, B}.
TEST(memory, wipe_subtree_vid_scope_sweep) {
    gl::Memory m;
    const int16_t A = m.nameMap.encodePush(gl::NameMap::MAIN_ID, "(impl1[a])");
    const int16_t B = m.nameMap.encodePush(A, "(in[b,N])");
    const int16_t S = m.nameMap.encodePush(gl::NameMap::MAIN_ID, "(impl1[a])x");

    // Populate the registries directly through the public members, one
    // statement per scope (main / A / B / S).
    auto addRows = [&](const std::string& expr, int16_t vid) -> int32_t {
        const gl::EncodedExpression e(expr, m.nameMap.decode(vid));
        const gl::IntEncodedExpr ie = gl::encodeExpression(e, m.nameMap);
        const int32_t pk = gl::packStatementKey(ie.originalId, ie.validityId);
        m.intEncodedStatements.push_back(ie);
        m.intLocalEncodedStatements.push_back(ie);
        m.intLocalEncodedStatementsSet.mint(pk);
        m.intStatementLevelsMap.insertSorted(pk, 0);
        m.intKnownStatements.insert(
            gl::StatementKey{ ie.originalId, ie.validityId },
            gl::StatementFlags{ true, false, true, true });
        m.intToBeProved.assignSet(pk, nullptr, 0);
        return pk;
    };
    const int32_t pkMain = addRows("(in[1,2])", gl::NameMap::MAIN_ID);
    const int32_t pkA    = addRows("(in[3,4])", A);
    const int32_t pkB    = addRows("(in[5,6])", B);
    const int32_t pkS    = addRows("(in[7,8])", S);

    // expandedImplications entries — scope half minted into the
    // lbStateInterner: equal-to-closed, descendant (prefix WITH
    // delimiter), survivor, and the literal-prefix-no-delimiter shape.
    const int32_t implId      = m.lbStateInterner.encode("(impl1[a])");
    const int32_t sigClosed   = m.lbStateInterner.encode(m.nameMap.decode(A));
    const int32_t sigDesc     = m.lbStateInterner.encode(m.nameMap.decode(B));
    const int32_t sigSurvivor = m.lbStateInterner.encode("main");
    const int32_t sigPrefix   =
        m.lbStateInterner.encode(m.nameMap.decode(A) + "x");
    m.expandedImplications.mint(gl::LbStatePairKey{ implId, sigClosed });
    m.expandedImplications.mint(gl::LbStatePairKey{ implId, sigDesc });
    m.expandedImplications.mint(gl::LbStatePairKey{ implId, sigSurvivor });
    m.expandedImplications.mint(gl::LbStatePairKey{ implId, sigPrefix });

    m.wipeSubtree(A);

    // Closed scope A and descendant B: every row/key gone.
    for (const int32_t pk : { pkA, pkB }) {
        ASSERT_FALSE(m.intLocalEncodedStatementsSet.contains(pk));
        ASSERT_EQ(m.intStatementLevelsMap.lookup(pk), 0);
        ASSERT_EQ(m.intToBeProved.lookup(pk), 0);
    }
    ASSERT_TRUE(m.intKnownStatements.find(
        gl::StatementKey{ static_cast<int16_t>(pkA >> 16), A }) == nullptr);
    ASSERT_TRUE(m.intKnownStatements.find(
        gl::StatementKey{ static_cast<int16_t>(pkB >> 16), B }) == nullptr);
    for (std::size_t i = 0; i < m.intEncodedStatements.size(); ++i) {
        ASSERT_NE(m.intEncodedStatements[i].validityId, A);
        ASSERT_NE(m.intEncodedStatements[i].validityId, B);
    }

    // Survivors: "main" and the literal-prefix sibling S (negative
    // coverage of the _boundary_-delimited prefix predicate).
    for (const int32_t pk : { pkMain, pkS }) {
        ASSERT_TRUE(m.intLocalEncodedStatementsSet.contains(pk));
        ASSERT_TRUE(m.intStatementLevelsMap.lookup(pk) != 0);
        ASSERT_TRUE(m.intToBeProved.lookup(pk) != 0);
    }

    // expandedImplications: closed + descendant dropped; survivor +
    // literal-prefix kept (the span twin's equal and prefix branches,
    // positive and negative).
    ASSERT_FALSE(m.expandedImplications.contains(
        gl::LbStatePairKey{ implId, sigClosed }));
    ASSERT_FALSE(m.expandedImplications.contains(
        gl::LbStatePairKey{ implId, sigDesc }));
    ASSERT_TRUE(m.expandedImplications.contains(
        gl::LbStatePairKey{ implId, sigSurvivor }));
    ASSERT_TRUE(m.expandedImplications.contains(
        gl::LbStatePairKey{ implId, sigPrefix }));

    // Step 11: closed ids (and only they) land in the filter set.
    ASSERT_TRUE(m.intValidityNamesToFilter.contains(A));
    ASSERT_TRUE(m.intValidityNamesToFilter.contains(B));
    ASSERT_FALSE(m.intValidityNamesToFilter.contains(S));
    ASSERT_FALSE(m.intValidityNamesToFilter.contains(gl::NameMap::MAIN_ID));
}

// collectClosedSubtreeIds — the forest-walk membership build vs the retained
// text-predicate oracle (the retired decode-every-id inClosed lambda,
// replicated verbatim). Coverage: a nested chain A > B > C (C's decoded
// text carries three delimiters), a sibling scope, the literal-prefix trap
// ((impl1[a])x — prefix without delimiter, must NOT match), a deep name
// minted WHOLE through encode's recursive last-delimiter split, and flat
// expression names (no delimiter, chain {self}). For closed in {A, B} the
// bitmap and the ascending id vector are the function's only two products;
// both must equal the oracle id-for-id.
TEST(memory, collect_closed_subtree_matches_string_predicate) {
    gl::Memory m;
    const int16_t A = m.nameMap.encodePush(gl::NameMap::MAIN_ID, "(impl1[a])");
    const int16_t B = m.nameMap.encodePush(A, "(in[b,N])");
    const int16_t C = m.nameMap.encodePush(B, "(in[c,N])");
    (void)C;
    m.nameMap.encodePush(gl::NameMap::MAIN_ID, "(impl2[a])");   // sibling
    m.nameMap.encodePush(gl::NameMap::MAIN_ID, "(impl1[a])x");  // prefix trap
    m.nameMap.encode(
        std::string("main_boundary_(impl9[q])_boundary_(=[q,r])"));
    m.nameMap.encode("(in[7,1])");                              // flat names
    m.nameMap.encode("(in[8,2])");

    const int32_t hi = m.nameMap.nameCount();
    for (const int16_t closed : { A, B }) {
        // ORACLE — the retired decode-all-ids string predicate, verbatim.
        const std::string closedScope = m.nameMap.decode(closed);
        const std::string boundary = closedScope + "_boundary_";
        auto inClosed = [&](const std::string& v) -> bool {
            if (v == closedScope) return true;
            if (v.size() <= boundary.size()) return false;
            return v.compare(0, boundary.size(), boundary) == 0;
        };
        std::vector<int16_t> oracleAsc;
        for (int16_t id = 1; id <= hi; ++id)
            if (inClosed(m.nameMap.decode(id))) oracleAsc.push_back(id);

        // Production: one forest walk.
        gl::LbArena lb(&gl::staticMemory());
        gl::DirtyState d = gl::DirtyState::Clean;
        gl::PagedVector<int16_t> asc(&lb, &d);
        uint64_t bits[512];
        const int32_t matches =
            m.nameMap.collectClosedSubtreeIds(closed, bits, 512, asc);

        // Per-id bit == oracle verdict, for EVERY minted id.
        for (int16_t id = 1; id <= hi; ++id) {
            const bool bit = ((bits[static_cast<uint16_t>(id) >> 6]
                               >> (static_cast<uint16_t>(id) & 63)) & 1ull) != 0;
            ASSERT_EQ(bit, inClosed(m.nameMap.decode(id)));
        }
        // Ascending vector == the oracle's ascending matching-id list.
        ASSERT_EQ(matches, static_cast<int32_t>(oracleAsc.size()));
        ASSERT_EQ(asc.size(), static_cast<int32_t>(oracleAsc.size()));
        for (int32_t i = 0; i < asc.size(); ++i)
            ASSERT_EQ(asc[i], oracleAsc[static_cast<std::size_t>(i)]);
    }
}

namespace {
    /// Assorted LocalMemoryValue shapes for the codec / peek / splice tests:
    /// empty and non-empty levels/keyIds/remainingArgIds, negative-ish ids,
    /// both bool combos, and a caller-chosen validityId.
    gl::LocalMemoryValue makeLmv(int16_t validityId, int shape) {
        gl::LocalMemoryValue v;
        v.validityId = validityId;
        switch (shape % 4) {
        case 0:   // minimal: everything empty, flags off
            v.valueId = 1; v.originalImplicationId = 2;
            v.justification = gl::RuleJustification::none;
            v.isMarker = false; v.productOfDisintegration = false;
            break;
        case 1:   // levels + keyIds, marker on
            v.valueId = 77; v.originalImplicationId = -3;
            v.justification = gl::RuleJustification::implication;
            v.isMarker = true; v.productOfDisintegration = false;
            v.levels = { 0, 2, 5 };
            v.keyIds = { 10, -11, 12 };
            break;
        case 2:   // remainingArgIds, both flags on
            v.valueId = -9; v.originalImplicationId = 40;
            v.justification = gl::RuleJustification::integration;
            v.isMarker = true; v.productOfDisintegration = true;
            v.remainingArgIds = { 100, 200 };
            break;
        default:  // everything populated, flags off
            v.valueId = 5; v.originalImplicationId = 6;
            v.justification = gl::RuleJustification::implication;
            v.isMarker = false; v.productOfDisintegration = true;
            v.levels = { 1 };
            v.keyIds = { 3 };
            v.remainingArgIds = { -7, 8, 9 };
            break;
        }
        return v;
    }
} // namespace

// lmvBlobValidityId — the fixed-offset validityId peek reads exactly what
// the codec wrote, for every value shape (empty / non-empty variable
// sections, negative-ish ids, both bool combos).
TEST(memory, lmv_blob_validity_id_peek_matches_codec) {
    const int16_t vids[] = { 1, 7, 42, 32000 };
    for (int shape = 0; shape < 4; ++shape) {
        for (const int16_t vid : vids) {
            const gl::LocalMemoryValue v = makeLmv(vid, shape);
            const std::vector<char> blob =
                gl::Codec<gl::LocalMemoryValue>::serialize(v);
            ASSERT_GE(static_cast<int32_t>(blob.size()), 16);
            const int16_t peeked = gl::lmvBlobValidityId(
                blob.data(), static_cast<int32_t>(blob.size()));
            ASSERT_EQ(peeked, v.validityId);
            ASSERT_EQ(peeked, gl::Codec<gl::LocalMemoryValue>::deserialize(
                blob.data(), static_cast<int32_t>(blob.size())).validityId);
        }
    }
}

// Codec<LocalMemoryValue> round trip — serialize -> deserialize ->
// serialize is byte-identical for every shape. This is byte-contract
// point 3 of the wipe's verbatim splice: a kept blob copied verbatim
// equals the old decode-then-re-serialize.
TEST(memory, codec_lmv_round_trip_byte_identity) {
    for (int shape = 0; shape < 4; ++shape) {
        const gl::LocalMemoryValue v = makeLmv(9, shape);
        const std::vector<char> b1 =
            gl::Codec<gl::LocalMemoryValue>::serialize(v);
        const gl::LocalMemoryValue d =
            gl::Codec<gl::LocalMemoryValue>::deserialize(
                b1.data(), static_cast<int32_t>(b1.size()));
        const std::vector<char> b2 =
            gl::Codec<gl::LocalMemoryValue>::serialize(d);
        ASSERT_EQ(b1.size(), b2.size());
        ASSERT_TRUE(b1 == b2);
    }
}

// wipeEncodedMapForClosed — the production two-pass verbatim blob splice
// against a test-local verbatim replica of the former heap rebuild
// (recordsAt -> vector erase -> resetToFresh -> typed assignRun). Two maps
// with identical content; oracle sweeps A, production sweeps B; key bytes,
// run lengths, and record bytes must match position by position. Cases: a
// key fully dropped, a key untouched, a key partially filtered, an
// empty-levels LMV.
TEST(memory, wipe_encoded_map_filtered_splice_matches_heap_oracle) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 16 << 20, 1 << 18 });
    gl::LbArena lbA(&g);
    gl::LbArena lbB(&g);
    gl::LbArena lbScratch(&g);
    gl::DirtyState dA = gl::DirtyState::Clean;
    gl::DirtyState dB = gl::DirtyState::Clean;
    gl::TypedColdBlobMap<gl::NormKey, gl::LocalMemoryValue> A(&lbA, &dA);
    gl::TypedColdBlobMap<gl::NormKey, gl::LocalMemoryValue> B(&lbB, &dB);

    // Closed scope ids {3, 7}; ceiling 10.
    const int32_t nameHighWater = 10;
    uint64_t closedBits[512] = {};
    for (const int16_t id : { int16_t{ 3 }, int16_t{ 7 } })
        closedBits[static_cast<uint16_t>(id) >> 6] |=
            (1ull << (static_cast<uint16_t>(id) & 63));
    const auto closedOracle = [](int16_t vid) {
        return vid == 3 || vid == 7;
    };

    // Identical content into both maps via the TYPED assignRun.
    const auto install = [&](const gl::NormKey& k,
                             const std::vector<gl::LocalMemoryValue>& run) {
        A.assignRun(k, run);
        B.assignRun(k, run);
    };
    // Key untouched (no closed records; shape 0 has empty levels).
    install(gl::NormKey{ 1, { 10, 20 } },
            { makeLmv(1, 0), makeLmv(2, 1) });
    // Key fully dropped (all records closed).
    install(gl::NormKey{ 2, { 30 } },
            { makeLmv(3, 2), makeLmv(7, 3) });
    // Key partially filtered (mixed scopes).
    install(gl::NormKey{ 1, { 40, 50, 60 } },
            { makeLmv(5, 1), makeLmv(3, 0), makeLmv(9, 2), makeLmv(7, 1) });
    // Another surviving key after the dropped one (order pin).
    install(gl::NormKey{ 3, { 70 } },
            { makeLmv(4, 3) });

    // ORACLE on A — verbatim replica of the former heap rebuild.
    {
        const int32_t encN = A.count();
        std::vector<std::pair<gl::NormKey,
                              std::vector<gl::LocalMemoryValue>>> survivors;
        for (int32_t id = 1; id <= encN; ++id) {
            std::vector<gl::LocalMemoryValue> vec = A.recordsAt(id);
            for (std::size_t i = vec.size(); i-- > 0; ) {
                if (closedOracle(vec[i].validityId)) {
                    vec.erase(vec.begin()
                        + static_cast<std::ptrdiff_t>(i));
                }
            }
            if (!vec.empty())
                survivors.emplace_back(A.decodeKey(id), std::move(vec));
        }
        A.resetToFresh();
        for (auto& kv : survivors) A.assignRun(kv.first, kv.second);
    }

    // Production splice on B.
    gl::wipeEncodedMapForClosed(B, closedBits, nameHighWater, lbScratch);

    // Facet equality: counts, key bytes, run lengths, record bytes.
    ASSERT_EQ(A.count(), B.count());
    for (int32_t id = 1; id <= A.count(); ++id) {
        const gl::StrSpan ka = A.inner().decode(id);
        const gl::StrSpan kb = B.inner().decode(id);
        ASSERT_TRUE(gl::equalSpans(ka, kb));
        ASSERT_EQ(A.runLen(id), B.runLen(id));
        for (int32_t j = 0; j < A.runLen(id); ++j) {
            std::vector<char> bufA, bufB;
            int32_t la = 0, lb = 0;
            const char* pa = A.peekRecordBytes(id, j, la, bufA);
            const char* pb = B.peekRecordBytes(id, j, lb, bufB);
            ASSERT_EQ(la, lb);
            ASSERT_TRUE(std::memcmp(pa, pb,
                                    static_cast<std::size_t>(la)) == 0);
        }
    }
}

// wipeOwnerSetMapForClosed — the zero-decode blob-byte filter against a
// test-local verbatim replica of the former decode-erase-reencode
// wipeOwnerMap lambda. Cases: all-survive (whole-blob verbatim), partial
// with non-contiguous survivors, all-dropped (key lands in droppedKeys),
// hasLooseOwner true and false, and non-empty uSignatures (multi-sig,
// multi-pair — the verbatim tail). droppedKeys is compared as a SET
// against the oracle's unordered_set<NormKey> via encoded-byte lookups.
TEST(memory, wipe_owner_set_map_filtered_matches_heap_oracle) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 16 << 20, 1 << 18 });
    gl::LbArena lbA(&g);
    gl::LbArena lbB(&g);
    gl::LbArena lbScratch(&g);
    gl::DirtyState dA = gl::DirtyState::Clean;
    gl::DirtyState dB = gl::DirtyState::Clean;
    gl::TypedColdBlobMap<gl::NormKey, gl::OwnerSet> A(&lbA, &dA);
    gl::TypedColdBlobMap<gl::NormKey, gl::OwnerSet> B(&lbB, &dB);

    // Closed scope vids {3, 7}; ceiling 10.
    const int32_t nameHighWater = 10;
    uint64_t closedBits[512] = {};
    for (const int16_t id : { int16_t{ 3 }, int16_t{ 7 } })
        closedBits[static_cast<uint16_t>(id) >> 6] |=
            (1ull << (static_cast<uint16_t>(id) & 63));
    const auto closedOracle = [](int16_t vid) {
        return vid == 3 || vid == 7;
    };

    const auto install = [&](const gl::NormKey& k, const gl::OwnerSet& os) {
        A.assignRun(k, std::vector<gl::OwnerSet>{ os });
        B.assignRun(k, std::vector<gl::OwnerSet>{ os });
    };
    // All survive; loose flag on; multi-sig multi-pair signatures.
    {
        gl::OwnerSet os;
        os.partitionIds = { gl::makePartitionId(10, 2),
                            gl::makePartitionId(11, 5) };
        os.hasLooseOwner = true;
        os.uSignatures = { { { 1, 2 }, { 3, 4 } }, { { 5, 6 } } };
        install(gl::NormKey{ 1, { 10 } }, os);
    }
    // Partial with NON-CONTIGUOUS survivors (closed at positions 1 and 3).
    {
        gl::OwnerSet os;
        os.partitionIds = { gl::makePartitionId(10, 2),
                            gl::makePartitionId(11, 3),
                            gl::makePartitionId(12, 5),
                            gl::makePartitionId(13, 7),
                            gl::makePartitionId(14, 9) };
        os.hasLooseOwner = false;
        os.uSignatures = { { { 7, 8 } } };
        install(gl::NormKey{ 2, { 20, 21 } }, os);
    }
    // All dropped -> the key must land in droppedKeys.
    {
        gl::OwnerSet os;
        os.partitionIds = { gl::makePartitionId(15, 3),
                            gl::makePartitionId(16, 7) };
        os.hasLooseOwner = false;
        install(gl::NormKey{ 1, { 30, 31, 32 } }, os);
    }
    // Survivor after the dropped key (key-order pin); loose off, no sigs.
    {
        gl::OwnerSet os;
        os.partitionIds = { gl::makePartitionId(17, 9) };
        install(gl::NormKey{ 3, { 40 } }, os);
    }

    // ORACLE on A — verbatim replica of the former wipeOwnerMap lambda
    // (collectDropped == true).
    std::unordered_set<gl::NormKey, gl::NormKeyHash> oracleDropped;
    {
        const int32_t n = A.count();
        std::vector<std::pair<gl::NormKey, gl::OwnerSet>> survivors;
        survivors.reserve(static_cast<std::size_t>(n));
        for (int32_t id = 1; id <= n; ++id) {
            gl::OwnerSet os = A.recordAt(id, 0);
            for (auto oit = os.partitionIds.begin();
                 oit != os.partitionIds.end(); ) {
                const int16_t scopeVid =
                    static_cast<int16_t>(*oit & 0xFFFF);
                if (closedOracle(scopeVid))
                    oit = os.partitionIds.erase(oit);
                else ++oit;
            }
            gl::NormKey key = A.decodeKey(id);
            if (os.partitionIds.empty()) {
                oracleDropped.insert(std::move(key));
            } else {
                survivors.emplace_back(std::move(key), std::move(os));
            }
        }
        A.resetToFresh();
        for (auto& kv : survivors)
            A.assignRun(kv.first, std::vector<gl::OwnerSet>{ kv.second });
    }

    // Production filter on B.
    gl::DirtyState dDropped = gl::DirtyState::Clean;
    gl::ColdHashSet<gl::BytesKeyStore> droppedKeys(&lbScratch, &dDropped);
    gl::wipeOwnerSetMapForClosed(B, closedBits, nameHighWater, lbScratch,
                                 &droppedKeys);

    // Facet equality: counts, key bytes, run lengths, blob bytes.
    ASSERT_EQ(A.count(), B.count());
    for (int32_t id = 1; id <= A.count(); ++id) {
        ASSERT_TRUE(gl::equalSpans(A.inner().decode(id),
                                   B.inner().decode(id)));
        ASSERT_EQ(A.runLen(id), B.runLen(id));
        for (int32_t j = 0; j < A.runLen(id); ++j) {
            std::vector<char> bufA, bufB;
            int32_t la = 0, lb = 0;
            const char* pa = A.peekRecordBytes(id, j, la, bufA);
            const char* pb = B.peekRecordBytes(id, j, lb, bufB);
            ASSERT_EQ(la, lb);
            ASSERT_TRUE(std::memcmp(pa, pb,
                                    static_cast<std::size_t>(la)) == 0);
        }
    }
    // droppedKeys == the oracle's dropped set (as SETS: size + per-key
    // encoded-byte lookup).
    ASSERT_EQ(droppedKeys.count(),
              static_cast<int32_t>(oracleDropped.size()));
    for (const gl::NormKey& nk : oracleDropped) {
        const std::string enc = gl::Codec<gl::NormKey>::encode(nk);
        ASSERT_TRUE(droppedKeys.lookup(gl::StrSpan(enc)) != 0);
    }
}

// wipeRemainingArgsForClosed — the byte-peek membership prune against a
// test-local verbatim replica of the former 10c block. Runs mix dropped
// and kept NormKeys; two NormKeys share identical data but different
// numberExpressions (the encoding includes both fields, so they must NOT
// collide). Also pins the droppedKeys-empty gate: the map's facets are
// untouched (the defined nothing-to-prune branch).
TEST(memory, wipe_remaining_args_pruned_matches_heap_oracle) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 16 << 20, 1 << 18 });
    gl::LbArena lbA(&g);
    gl::LbArena lbB(&g);
    gl::LbArena lbScratch(&g);
    gl::DirtyState dA = gl::DirtyState::Clean;
    gl::DirtyState dB = gl::DirtyState::Clean;
    gl::TypedColdBlobMap<gl::Int16SetKey, gl::NormKey> A(&lbA, &dA);
    gl::TypedColdBlobMap<gl::Int16SetKey, gl::NormKey> B(&lbB, &dB);

    // Dropped set: {1,{50}} and {2,{60,61}} — and the twin {1,{60,61}}
    // (same data as the second, different numberExpressions) is KEPT.
    const gl::NormKey dropped1{ 1, { 50 } };
    const gl::NormKey dropped2{ 2, { 60, 61 } };
    const gl::NormKey keptTwin{ 1, { 60, 61 } };
    const gl::NormKey kept1{ 1, { 70 } };

    const auto install = [&](const gl::Int16SetKey& k,
                             const std::vector<gl::NormKey>& run) {
        A.assignRun(k, run);
        B.assignRun(k, run);
    };
    install(gl::Int16SetKey{ { 1, 2 } },
            { kept1, dropped1, keptTwin });          // partial prune
    install(gl::Int16SetKey{ { 3 } },
            { dropped1, dropped2 });                 // run empties -> key drops
    install(gl::Int16SetKey{ { 4, 5, 6 } },
            { keptTwin, kept1 });                    // untouched

    // The dropped-key byte set (as 10b collects it).
    gl::DirtyState dDropped = gl::DirtyState::Clean;
    gl::ColdHashSet<gl::BytesKeyStore> droppedKeys(&lbScratch, &dDropped);
    for (const gl::NormKey* nk : { &dropped1, &dropped2 }) {
        const std::string enc = gl::Codec<gl::NormKey>::encode(*nk);
        droppedKeys.mint(gl::StrSpan(enc));
    }

    // ORACLE on A — verbatim replica of the former 10c block.
    {
        std::unordered_set<gl::NormKey, gl::NormKeyHash> oracleDropped{
            dropped1, dropped2 };
        const int32_t raN = A.count();
        std::vector<std::pair<gl::Int16SetKey,
                              std::vector<gl::NormKey>>> survivors;
        survivors.reserve(static_cast<std::size_t>(raN));
        for (int32_t id = 1; id <= raN; ++id) {
            std::vector<gl::NormKey> run = A.recordsAt(id);
            std::vector<gl::NormKey> kept;
            kept.reserve(run.size());
            for (gl::NormKey& nk : run)
                if (!oracleDropped.count(nk)) kept.push_back(std::move(nk));
            if (!kept.empty())
                survivors.emplace_back(A.decodeKey(id), std::move(kept));
        }
        A.resetToFresh();
        for (auto& kv : survivors)
            A.assignRun(kv.first, kv.second);
    }

    // Production prune on B.
    gl::wipeRemainingArgsForClosed(B, droppedKeys, lbScratch);

    // Facet equality.
    ASSERT_EQ(A.count(), B.count());
    for (int32_t id = 1; id <= A.count(); ++id) {
        ASSERT_TRUE(gl::equalSpans(A.inner().decode(id),
                                   B.inner().decode(id)));
        ASSERT_EQ(A.runLen(id), B.runLen(id));
        for (int32_t j = 0; j < A.runLen(id); ++j) {
            std::vector<char> bufA, bufB;
            int32_t la = 0, lb = 0;
            const char* pa = A.peekRecordBytes(id, j, la, bufA);
            const char* pb = B.peekRecordBytes(id, j, lb, bufB);
            ASSERT_EQ(la, lb);
            ASSERT_TRUE(std::memcmp(pa, pb,
                                    static_cast<std::size_t>(la)) == 0);
        }
    }

    // Empty-gate pin: an empty dropped set leaves B untouched.
    std::vector<std::string> keysBefore;
    std::vector<std::vector<char>> recsBefore;
    for (int32_t id = 1; id <= B.count(); ++id) {
        const gl::StrSpan ks = B.inner().decode(id);
        keysBefore.emplace_back(ks.ptr, static_cast<std::size_t>(ks.len));
        for (int32_t j = 0; j < B.runLen(id); ++j) {
            std::vector<char> buf;
            int32_t bl = 0;
            const char* bp = B.peekRecordBytes(id, j, bl, buf);
            recsBefore.emplace_back(bp, bp + bl);
        }
    }
    gl::DirtyState dEmpty = gl::DirtyState::Clean;
    gl::ColdHashSet<gl::BytesKeyStore> emptyDropped(&lbScratch, &dEmpty);
    gl::wipeRemainingArgsForClosed(B, emptyDropped, lbScratch);
    std::size_t cursor = 0;
    ASSERT_EQ(static_cast<std::size_t>(B.count()), keysBefore.size());
    for (int32_t id = 1; id <= B.count(); ++id) {
        const gl::StrSpan ks = B.inner().decode(id);
        ASSERT_TRUE(gl::equalSpans(
            ks, gl::StrSpan(keysBefore[static_cast<std::size_t>(id - 1)])));
        for (int32_t j = 0; j < B.runLen(id); ++j, ++cursor) {
            std::vector<char> buf;
            int32_t bl = 0;
            const char* bp = B.peekRecordBytes(id, j, bl, buf);
            ASSERT_EQ(static_cast<std::size_t>(bl), recsBefore[cursor].size());
            ASSERT_TRUE(std::memcmp(bp, recsBefore[cursor].data(),
                                    static_cast<std::size_t>(bl)) == 0);
        }
    }
    ASSERT_EQ(cursor, recsBefore.size());
}

// lookupStatementLevels / isLocalEncodedStatement — the non-minting packed-key
// probes over the statement indexes. Three contract points each: a stored
// entry hits with its payload intact, an interned-but-absent pair misses, and
// a never-interned name is a definitive miss that does not mint (I-84 read
// discipline — the LB's id-assignment order stays untouched).
TEST(memory, packed_statement_index_probes_are_non_minting) {
    gl::Memory m;
    const gl::EncodedExpression e("(in[1,2])", "main");
    const gl::IntEncodedExpr ie = gl::encodeExpression(e, m.nameMap);
    const int32_t pk = gl::packStatementKey(ie.originalId, ie.validityId);
    m.intLocalEncodedStatementsSet.mint(pk);
    m.intStatementLevelsMap.insertSorted(pk, 0);
    m.intStatementLevelsMap.insertSorted(pk, 2);

    // Positive: both probes hit; the levels payload comes back intact.
    const int32_t lvId = gl::lookupStatementLevels(
        m.intStatementLevelsMap, m.nameMap, "(in[1,2])", "main");
    ASSERT_TRUE(lvId != 0);
    const std::set<int> expected{ 0, 2 };
    ASSERT_TRUE(gl::coldIntSetAt(m.intStatementLevelsMap, lvId) == expected);
    ASSERT_TRUE(gl::isLocalEncodedStatement(
        m.intLocalEncodedStatementsSet, m.nameMap, "(in[1,2])", "main"));

    // Negative: interned pair with no index entry.
    m.nameMap.encode("(in[3,4])");
    ASSERT_EQ(gl::lookupStatementLevels(
        m.intStatementLevelsMap, m.nameMap, "(in[3,4])", "main"), 0);
    ASSERT_FALSE(gl::isLocalEncodedStatement(
        m.intLocalEncodedStatementsSet, m.nameMap, "(in[3,4])", "main"));

    // Negative: never-interned name — definitive miss, nothing minted.
    const int32_t mintedBefore = m.nameMap.nameCount();
    ASSERT_EQ(gl::lookupStatementLevels(
        m.intStatementLevelsMap, m.nameMap, "(in[9,9])", "main"), 0);
    ASSERT_FALSE(gl::isLocalEncodedStatement(
        m.intLocalEncodedStatementsSet, m.nameMap, "(in[9,9])", "main"));
    ASSERT_EQ(m.nameMap.nameCount(), mintedBefore);
    ASSERT_EQ(m.nameMap.lookup("(in[9,9])"), static_cast<int16_t>(0));
}

// countPatternOccurrences — the secondary-variable cap counter tests
// productsOfRecursionIds membership through the non-minting NameMap::lookup
// (I-84 read discipline). Three contract points in one expression: a
// recursion product is skipped, an interned non-member counts, and a
// never-interned lexeme counts via the definitive lookup miss — all without
// minting into the NameMap.
TEST(prover, count_pattern_occurrences_skips_recursion_products) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::Memory mb;

    // Intern one recursion product the way the lockstep insert sites do.
    mb.overallHashMemory.productsOfRecursionIds.mint(
        mb.nameMap.encode("it_0_lev_1_2"));
    // An interned lexeme that is NOT a recursion product.
    mb.nameMap.encode("it_3_lev_4_5");
    const int32_t mintedBefore = mb.nameMap.nameCount();

    const std::string expr = "(+[it_0_lev_1_2,it_3_lev_4_5,it_7_lev_8_9])";
    ASSERT_EQ(ea.countPatternOccurrences(expr, mb.overallHashMemory, mb.nameMap), 2);

    // Non-minting read path: the never-interned lexeme stayed un-interned.
    ASSERT_EQ(mb.nameMap.nameCount(), mintedBefore);
    ASSERT_EQ(mb.nameMap.lookup("it_7_lev_8_9"), static_cast<int16_t>(0));
}

// The three residual HashMemory sets are statified cold
// (D-173): mint / contains / count / eraseIf round-trip
// through the cold-set surface, and resetToFresh empties every one.
TEST(prover, hashmemory_residual_sets_cold_roundtrip) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::Memory mb;
    auto& hm = mb.overallHashMemory;

    // Packed int32 admission/trigger sets.
    hm.admissionSetIntegration.mint(101);
    hm.admissionSetIntegration.mint(102);
    hm.triggersForAdmissionSetIntegration.mint(201);
    // int16 NameMap-id membership cache.
    hm.productsOfRecursionIds.mint(static_cast<int16_t>(7));

    ASSERT_TRUE(hm.admissionSetIntegration.contains(101));
    ASSERT_TRUE(hm.admissionSetIntegration.contains(102));
    ASSERT_FALSE(hm.admissionSetIntegration.contains(999));
    ASSERT_EQ(hm.admissionSetIntegration.count(), 2);
    ASSERT_TRUE(hm.triggersForAdmissionSetIntegration.contains(201));
    ASSERT_TRUE(hm.productsOfRecursionIds.contains(static_cast<int16_t>(7)));

    // Predicate erase drops only the matching key (the radical-wipe path).
    hm.admissionSetIntegration.eraseIf([](int32_t k) { return k == 101; });
    ASSERT_FALSE(hm.admissionSetIntegration.contains(101));
    ASSERT_TRUE(hm.admissionSetIntegration.contains(102));
    ASSERT_EQ(hm.admissionSetIntegration.count(), 1);

    // Wholesale reset empties every set.
    hm.resetToFresh();
    ASSERT_TRUE(hm.admissionSetIntegration.empty());
    ASSERT_TRUE(hm.triggersForAdmissionSetIntegration.empty());
    ASSERT_TRUE(hm.productsOfRecursionIds.empty());
}

// upsertStatementKey — OR-only membership-bit door for the packed statement
// registry. A fresh key stores all four flags as given; an existing key ORs
// registered / known / fullyDisintegrated in (never clears) and keeps the
// first writer's `local`.
TEST(memory, upsert_statement_key_or_only_keep_first_local) {
    gl::Memory mb;
    auto& reg = mb.intKnownStatements;   // the cold registry
    const int32_t k = gl::packStatementKey(7, 1);

    gl::upsertStatementKey(reg, k, /*local=*/false, /*registered=*/true, /*known=*/false);
    ASSERT_TRUE(reg.inner().find(k)->registered);
    ASSERT_FALSE(reg.inner().find(k)->known);
    ASSERT_FALSE(reg.inner().find(k)->local);
    ASSERT_FALSE(reg.inner().find(k)->fullyDisintegrated);

    // Second writer ORs `known` in, cannot clear `registered`, and the
    // stored `local` keeps the first writer's value.
    gl::upsertStatementKey(reg, k, /*local=*/true, /*registered=*/false, /*known=*/true);
    ASSERT_TRUE(reg.inner().find(k)->registered);
    ASSERT_TRUE(reg.inner().find(k)->known);
    ASSERT_FALSE(reg.inner().find(k)->local);

    // fullyDisintegrated ORs in and survives a later writer passing false.
    gl::upsertStatementKey(reg, k, true, true, true, /*fullyDisintegrated=*/true);
    ASSERT_TRUE(reg.inner().find(k)->fullyDisintegrated);
    gl::upsertStatementKey(reg, k, true, true, true, /*fullyDisintegrated=*/false);
    ASSERT_TRUE(reg.inner().find(k)->fullyDisintegrated);
    ASSERT_EQ(reg.count(), 1);
}

// lookupStatementFlags — non-minting probe of the packed statement registry.
// A stored row comes back by (original, validityName); a never-interned probe
// is a definitive nullptr without minting; an interned pair with no row is
// also nullptr.
TEST(memory, lookup_statement_flags_non_minting_definitive_miss) {
    gl::Memory mb;
    const int16_t oid = mb.nameMap.encode("(p[a])");
    gl::upsertStatementKey(mb.intKnownStatements,
        gl::packStatementKey(oid, gl::NameMap::MAIN_ID),
        /*local=*/true, /*registered=*/true, /*known=*/false);

    const gl::StatementFlags* hit = gl::lookupStatementFlags(
        mb.intKnownStatements, mb.nameMap, "(p[a])", "main");
    ASSERT_TRUE(hit != nullptr);
    ASSERT_TRUE(hit->registered);
    ASSERT_FALSE(hit->known);

    const int32_t mintedBefore = mb.nameMap.nameCount();
    ASSERT_TRUE(gl::lookupStatementFlags(mb.intKnownStatements, mb.nameMap,
                                         "(q[never])", "main") == nullptr);
    ASSERT_EQ(mb.nameMap.nameCount(), mintedBefore);

    mb.nameMap.encode("(r[b])");
    ASSERT_TRUE(gl::lookupStatementFlags(mb.intKnownStatements, mb.nameMap,
                                         "(r[b])", "main") == nullptr);
}

// lookupToBeProved — non-minting probe of the packed goal registry. A stored
// goal comes back by (original, validityName) with its auxy set intact; an
// interned pair with no goal is nullptr; a never-interned probe is a
// definitive nullptr without minting.
TEST(memory, lookup_to_be_proved_non_minting_probe) {
    gl::Memory mb;
    const int16_t oid = mb.nameMap.encode("(in[2,1])");
    mb.intToBeProved.insertSorted(
        gl::packStatementKey(oid, gl::NameMap::MAIN_ID), 7);

    const int32_t hitId = gl::lookupToBeProved(mb.intToBeProved, mb.nameMap,
                                               "(in[2,1])", "main");
    ASSERT_TRUE(hitId != 0);
    const std::set<int> expectedAuxies{ 7 };
    ASSERT_TRUE(gl::coldIntSetAt(mb.intToBeProved, hitId) == expectedAuxies);

    // Interned pair, no goal stored.
    mb.nameMap.encode("(in[3,1])");
    ASSERT_EQ(gl::lookupToBeProved(mb.intToBeProved, mb.nameMap,
                                   "(in[3,1])", "main"), 0);

    // Never-interned probe — definitive miss, nothing minted.
    const int32_t mintedBefore = mb.nameMap.nameCount();
    ASSERT_EQ(gl::lookupToBeProved(mb.intToBeProved, mb.nameMap,
                                   "(in[9,9])", "main"), 0);
    ASSERT_EQ(mb.nameMap.nameCount(), mintedBefore);
    ASSERT_EQ(mb.nameMap.lookup("(in[9,9])"), static_cast<int16_t>(0));
}

// decodeToBeProvedSorted — the order-preserving walk vehicle. Rows come back
// lex-sorted by decoded (original, validityName) regardless of mint order
// (raw ids are NOT the sort key), strings are owned copies, and each row's
// packed key round-trips to its registry entry.
TEST(memory, decode_to_be_proved_sorted_lex_order_not_id_order) {
    gl::Memory mb;
    // Mint deliberately in REVERSE lex order so id order != lex order.
    const int16_t idC = mb.nameMap.encode("(in[3,1])");
    const int16_t idB = mb.nameMap.encode("(in[2,1])");
    const int16_t idA = mb.nameMap.encode("(in[1,1])");
    ASSERT_TRUE(idC < idB && idB < idA);  // mint order, reversed lex

    mb.intToBeProved.assignSet(
        gl::packStatementKey(idC, gl::NameMap::MAIN_ID), nullptr, 0);
    mb.intToBeProved.assignSet(
        gl::packStatementKey(idB, gl::NameMap::MAIN_ID), nullptr, 0);
    mb.intToBeProved.assignSet(
        gl::packStatementKey(idA, gl::NameMap::MAIN_ID), nullptr, 0);

    const std::vector<gl::DecodedToBeProvedRow> rows =
        gl::decodeToBeProvedSorted(mb.intToBeProved, mb.nameMap);
    ASSERT_EQ(rows.size(), static_cast<std::size_t>(3));
    ASSERT_EQ(rows[0].original, std::string("(in[1,1])"));
    ASSERT_EQ(rows[1].original, std::string("(in[2,1])"));
    ASSERT_EQ(rows[2].original, std::string("(in[3,1])"));
    for (const gl::DecodedToBeProvedRow& row : rows) {
        ASSERT_EQ(row.validityName, std::string("main"));
        ASSERT_TRUE(mb.intToBeProved.lookup(row.key) != 0);
    }
    ASSERT_EQ(rows[0].key, gl::packStatementKey(idA, gl::NameMap::MAIN_ID));
    ASSERT_EQ(rows[2].key, gl::packStatementKey(idC, gl::NameMap::MAIN_ID));
}

// sortToBeProvedKeys — the mutate-during-walk twin of decodeToBeProvedSorted.
// The packed-key snapshot, decoded per key, reproduces the owning oracle's
// (original, validityName) order byte-for-byte, including the validity
// tie-break on a repeated original; the empty registry returns nullptr/count 0.
TEST(memory, sort_to_be_proved_keys_matches_decode_to_be_proved_sorted) {
    gl::Memory mb;
    // Mint in reverse lex so id order != lex order; a second scope on the same
    // original exercises the validity tie-break.
    const int16_t vSub = mb.nameMap.encode("sub_1");
    const int16_t idC = mb.nameMap.encode("(in[3,1])");
    const int16_t idB = mb.nameMap.encode("(in[2,1])");
    const int16_t idA = mb.nameMap.encode("(in[1,1])");
    mb.intToBeProved.assignSet(gl::packStatementKey(idC, gl::NameMap::MAIN_ID), nullptr, 0);
    mb.intToBeProved.assignSet(gl::packStatementKey(idB, gl::NameMap::MAIN_ID), nullptr, 0);
    mb.intToBeProved.assignSet(gl::packStatementKey(idA, gl::NameMap::MAIN_ID), nullptr, 0);
    mb.intToBeProved.assignSet(gl::packStatementKey(idA, vSub), nullptr, 0);

    // Oracle: the owning-copy sorted rows.
    const std::vector<gl::DecodedToBeProvedRow> oracle =
        gl::decodeToBeProvedSorted(mb.intToBeProved, mb.nameMap);

    // Twin: the packed-key snapshot decoded per key.
    gl::ScratchArena sA; sA.bind(&gl::staticMemory());
    {
        gl::ScratchScope scope(sA);
        int32_t n = -1;
        int32_t* keys = gl::sortToBeProvedKeys(mb.intToBeProved, mb.nameMap, sA, n);
        ASSERT_EQ(static_cast<std::size_t>(n), oracle.size());
        ASSERT_TRUE(keys != nullptr);
        for (int32_t i = 0; i < n; ++i) {
            const int16_t oid = static_cast<int16_t>(
                (static_cast<uint32_t>(keys[i]) >> 16) & 0xFFFF);
            const int16_t vid = static_cast<int16_t>(
                static_cast<uint32_t>(keys[i]) & 0xFFFF);
            ASSERT_EQ(mb.nameMap.decode(oid),
                      oracle[static_cast<std::size_t>(i)].original);
            ASSERT_EQ(mb.nameMap.decode(vid),
                      oracle[static_cast<std::size_t>(i)].validityName);
            ASSERT_EQ(keys[i], oracle[static_cast<std::size_t>(i)].key);
        }
    }

    // Empty registry -> nullptr, count 0.
    gl::Memory empty;
    gl::ScratchArena sB; sB.bind(&gl::staticMemory());
    {
        gl::ScratchScope scope(sB);
        int32_t n = -1;
        int32_t* keys = gl::sortToBeProvedKeys(empty.intToBeProved, empty.nameMap, sB, n);
        ASSERT_EQ(n, 0);
        ASSERT_TRUE(keys == nullptr);
    }
}

// sortStatementRows — the transient-buffer index sort. Reproduces
// std::sort(std::vector<ExpressionWithValidity>) (lex over (original,
// validityName)) byte-for-byte from a PagedVector<IntEncodedExpr>, including a
// duplicate (original, validity) pair and a same-original/different-validity
// pair that exercises the validity tie-break. The oracle is the EWV std::sort;
// the twin is encode -> sortStatementRows; the decoded pair sequences must match
// element-for-element.
TEST(memory, sort_statement_rows_matches_std_sort_ewv) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;

    const std::vector<gl::ExpressionWithValidity> input = {
        gl::ExpressionWithValidity("(in2[i0,v1,s])", "main"),
        gl::ExpressionWithValidity("(in3[a,b,c,d])", "scopeB"),
        gl::ExpressionWithValidity("(=[x,y])",       "main"),
        gl::ExpressionWithValidity("(p[q])",         "scopeB"),
        gl::ExpressionWithValidity("(p[q])",         "main"),   // same original as [3], other validity
        gl::ExpressionWithValidity("(=[x,y])",       "main"),   // exact duplicate of [2]
    };

    // Oracle: std::sort of the EWV vector under operator<.
    std::vector<gl::ExpressionWithValidity> oracle = input;
    std::sort(oracle.begin(), oracle.end());

    // Twin: encode to IntEncodedExpr on a paged buffer, then sortStatementRows.
    gl::LbArena rowsArena{ &gl::staticMemory() };
    gl::DirtyState rowsDirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> rows{ &rowsArena, &rowsDirty };
    for (const gl::ExpressionWithValidity& e : input) {
        rows.push_back(gl::encodeExpression(gl::StrSpan(e.original),
                                            gl::StrSpan(e.validityName), nm));
    }

    gl::ScratchArena sA; sA.bind(&gl::staticMemory());
    {
        gl::ScratchScope scope(sA);
        const int32_t* order = gl::sortStatementRows(rows, nm, sA);
        ASSERT_TRUE(order != nullptr);
        ASSERT_EQ(static_cast<std::size_t>(rows.size()), oracle.size());
        for (int32_t i = 0; i < rows.size(); ++i) {
            const gl::IntEncodedExpr& row = rows[order[i]];
            ASSERT_EQ(nm.decode(row.originalId),
                      oracle[static_cast<std::size_t>(i)].original);
            ASSERT_EQ(nm.decode(row.validityId),
                      oracle[static_cast<std::size_t>(i)].validityName);
        }
    }

    // Empty buffer -> nullptr (the caller loop then never dereferences it).
    gl::PagedVector<gl::IntEncodedExpr> emptyRows{ &rowsArena, &rowsDirty };
    {
        gl::ScratchScope emptyScope(sA);
        ASSERT_TRUE(gl::sortStatementRows(emptyRows, nm, sA) == nullptr);
    }
}

// ---------- L8 mail-cold decoded-order comparators (absorb entrance) --------

// originLineLessId — the crux comparator. Reproduces OriginLine::operator<
// (std::pair<std::string, std::vector<EWV>>) byte-for-byte on id-form
// IntMailOrigin blobs. Oracle = std::sort of heap OriginLines; twin = an index
// sorted by originLineLessId over the serialized records. The source deliberately
// includes: (a) a tag where STRING order disagrees with the OriginTag enum order
// ("CE_building_block" is enum #2 but sorts BEFORE "anchor handling" enum #0 by
// byte-lex), proving the tag is compared as a string not an enum; (b) equal-tag /
// different-dep pairs (deps decide); (c) a shorter-vector prefix tie (shorter
// first); (d) an empty-dep line.
TEST(memory, origin_line_less_id_matches_std_sort_originline) {
    gl::Memory m;
    gl::ValueInterner& oi = m.originInterner;

    struct Src { std::string tag;
                 std::vector<std::pair<std::string, std::string>> deps; };
    const std::vector<Src> src = {
        { "equality1",         { { "p", "main" }, { "q", "main" } } },
        { "equality1",         { { "p", "main" }, { "r", "main" } } }, // dep[1]
        { "CE_building_block",  { { "z", "main" } } },  // enum#2, string < "anchor.."
        { "anchor handling",    { { "a", "main" } } },  // enum#0, string > "CE_.."
        { "equality1",          { { "p", "main" } } },  // shorter prefix of src[0/1]
        { "disintegration",     {} },                    // empty deps
        { "disintegration",     { { "d", "main" } } },  // [] < [d]
    };

    std::vector<gl::OriginLine> oracle;
    for (const Src& s : src) {
        gl::OriginLine ol;
        ol.first = s.tag;
        for (const auto& d : s.deps) ol.second.emplace_back(d.first, d.second);
        oracle.push_back(ol);
    }
    std::sort(oracle.begin(), oracle.end());

    std::vector<std::vector<char>> blobs;
    for (const Src& s : src) {
        gl::IntMailOrigin rec;
        rec.tag = static_cast<uint8_t>(gl::originTagFromString(s.tag));
        for (const auto& d : s.deps)
            rec.deps.push_back(gl::mintOriginKey(oi, d.first, d.second));
        blobs.push_back(gl::Codec<gl::IntMailOrigin>::serialize(rec));
    }
    std::vector<gl::MailOriginBlobView> views;
    for (const std::vector<char>& b : blobs)
        views.push_back(gl::viewMailOriginBlob(
            b.data(), static_cast<int32_t>(b.size())));

    std::vector<int32_t> order(src.size());
    for (int32_t i = 0; i < static_cast<int32_t>(src.size()); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        return gl::originLineLessId(views[a], views[b], oi);
    });

    ASSERT_EQ(order.size(), oracle.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        const gl::MailOriginBlobView& v = views[order[i]];
        ASSERT_EQ(std::string(gl::originTagName(
                      static_cast<gl::OriginTag>(v.tag))),
                  oracle[i].first);
        ASSERT_EQ(v.depN, static_cast<int32_t>(oracle[i].second.size()));
        for (int32_t k = 0; k < v.depN; ++k) {
            const std::pair<std::string, std::string> ev =
                gl::decodeOriginKey(gl::mailOriginDepAt(v, k), oi);
            ASSERT_EQ(ev.first, oracle[i].second[static_cast<std::size_t>(k)].original);
            ASSERT_EQ(ev.second,
                      oracle[i].second[static_cast<std::size_t>(k)].validityName);
        }
    }
}

// decodedOriginKeyLess — reproduces ExpressionWithValidity::operator< on the
// decoded (expression, validity) pair of two packed origin-map keys. Oracle =
// std::sort of the EWV keys; twin = the packed keys sorted by decodedOriginKeyLess
// (a ValueInterner id space, minted in reverse-lex so id order != decoded order).
TEST(memory, decoded_origin_key_less_matches_ewv_order) {
    gl::Memory m;
    gl::ValueInterner& oi = m.originInterner;

    const std::vector<std::pair<std::string, std::string>> src = {
        { "(=[b,c])", "main" },
        { "(=[a,b])", "scopeZ" },
        { "(=[a,b])", "main" },   // same expr as [1], validity decides
        { "(=[a,b,c])", "main" }, // "(=[a,b" prefix < "(=[a,b,c" — longer
    };

    std::vector<gl::ExpressionWithValidity> oracle;
    for (const auto& s : src) oracle.emplace_back(s.first, s.second);
    std::sort(oracle.begin(), oracle.end());

    // Mint in reverse source order so interner ids do NOT track lex order.
    std::vector<int64_t> keys(src.size());
    for (std::size_t i = src.size(); i-- > 0;)
        keys[i] = gl::mintOriginKey(oi, src[i].first, src[i].second);

    std::vector<int32_t> order(src.size());
    for (int32_t i = 0; i < static_cast<int32_t>(src.size()); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        return gl::decodedOriginKeyLess(keys[a], keys[b], oi);
    });

    ASSERT_EQ(order.size(), oracle.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        const std::pair<std::string, std::string> ev =
            gl::decodeOriginKey(keys[order[i]], oi);
        ASSERT_EQ(ev.first, oracle[i].original);
        ASSERT_EQ(ev.second, oracle[i].validityName);
    }
}

// decodedStatementLess — reproduces std::pair<EWV, std::set<int>>::operator< on
// two IntMailStatementKey byte keys: EWV first (original, validity), then the
// ascending levels lexicographically (shorter run first). Covered on BOTH id
// spaces: NameMap (internal channel) and ColdStringTable (routing channel).
TEST(memory, decoded_statement_less_matches_std_sort_pair) {
    struct Src { std::string original; std::string validity; std::set<int> levels; };
    const std::vector<Src> src = {
        { "(in3[a,b,c,d])", "scopeB", { 1 } },
        { "(=[x,y])",       "main",   { 1, 2 } },
        { "(=[x,y])",       "main",   { 1 } },     // same EWV, prefix -> shorter first
        { "(=[x,y])",       "main",   { 1, 3 } },  // same EWV, element decides vs {1,2}
        { "(p[q])",         "main",   {} },        // empty levels
        { "(p[q])",         "scopeB", { 5 } },     // same original as [4], validity
    };

    std::vector<std::pair<gl::ExpressionWithValidity, std::set<int>>> oracle;
    for (const Src& s : src)
        oracle.emplace_back(gl::ExpressionWithValidity(s.original, s.validity),
                            s.levels);
    std::sort(oracle.begin(), oracle.end());

    // Internal channel — NameMap id space.
    {
        NameMapRig nmRig;
        gl::NameMap& nm = nmRig.nm;
        std::vector<std::string> keyBytes;
        for (const Src& s : src) {
            gl::IntMailStatementKey k;
            k.originalId = nm.encode(s.original);
            k.validityId = nm.encode(s.validity);
            k.levels.assign(s.levels.begin(), s.levels.end());
            keyBytes.push_back(gl::Codec<gl::IntMailStatementKey>::encode(k));
        }
        std::vector<int32_t> order(src.size());
        for (int32_t i = 0; i < static_cast<int32_t>(src.size()); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
            return gl::decodedStatementLess(gl::StrSpan(keyBytes[a]),
                                            gl::StrSpan(keyBytes[b]), nm);
        });
        ASSERT_EQ(order.size(), oracle.size());
        for (std::size_t i = 0; i < order.size(); ++i) {
            const gl::IntMailStatementKey k =
                gl::Codec<gl::IntMailStatementKey>::decode(
                    gl::StrSpan(keyBytes[order[i]]));
            ASSERT_EQ(std::string(nm.decode(static_cast<int16_t>(k.originalId))),
                      oracle[i].first.original);
            ASSERT_EQ(std::string(nm.decode(static_cast<int16_t>(k.validityId))),
                      oracle[i].first.validityName);
            ASSERT_EQ(k.levels.size(), oracle[i].second.size());
        }
    }

    // Routing channel — global ColdStringTable id space. Fresh table, int32 ids.
    {
        gl::LbArena lb{ &gl::staticMemory() };
        gl::DirtyState d = gl::DirtyState::Clean;
        gl::ColdStringTable table{ &lb, &d };
        std::vector<std::string> keyBytes;
        for (const Src& s : src) {
            gl::IntMailStatementKey k;
            k.originalId = table.intern(gl::StrSpan(s.original));
            k.validityId = table.intern(gl::StrSpan(s.validity));
            k.levels.assign(s.levels.begin(), s.levels.end());
            keyBytes.push_back(gl::Codec<gl::IntMailStatementKey>::encode(k));
        }
        std::vector<int32_t> order(src.size());
        for (int32_t i = 0; i < static_cast<int32_t>(src.size()); ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
            return gl::decodedStatementLess(gl::StrSpan(keyBytes[a]),
                                            gl::StrSpan(keyBytes[b]), table);
        });
        ASSERT_EQ(order.size(), oracle.size());
        for (std::size_t i = 0; i < order.size(); ++i) {
            const gl::IntMailStatementKey k =
                gl::Codec<gl::IntMailStatementKey>::decode(
                    gl::StrSpan(keyBytes[order[i]]));
            ASSERT_EQ(table.decodeString(k.originalId), oracle[i].first.original);
            ASSERT_EQ(table.decodeString(k.validityId), oracle[i].first.validityName);
        }
    }
}

// ---------- classifyName / scanSpecialTokens / EqClassNameCaches ----------

// classifyName — whole-string match only; the three tiers are exactly the
// historical filterIterations / chooseCanonical bucketing.
TEST(memory, classify_name_tiers) {
    ASSERT_TRUE(gl::classifyName("int_lev_3_4") == gl::NameKind::IntLev);
    ASSERT_TRUE(gl::classifyName("int_lev_0_12") == gl::NameKind::IntLev);
    ASSERT_TRUE(gl::classifyName("it_0_lev_1_2") == gl::NameKind::ItLev);
    ASSERT_TRUE(gl::classifyName("it_12_lev_0_7") == gl::NameKind::ItLev);
    // Normal tier: user names, repl_*, concrete values.
    ASSERT_TRUE(gl::classifyName("x") == gl::NameKind::Normal);
    ASSERT_TRUE(gl::classifyName("repl_7") == gl::NameKind::Normal);
    ASSERT_TRUE(gl::classifyName("zero") == gl::NameKind::Normal);
    // Whole-match negatives: embedded or truncated patterns do not qualify.
    ASSERT_TRUE(gl::classifyName("print_lev_3_4") == gl::NameKind::Normal);
    ASSERT_TRUE(gl::classifyName("int_lev_3") == gl::NameKind::Normal);
    ASSERT_TRUE(gl::classifyName("int_lev_3_4_5") == gl::NameKind::Normal);
    ASSERT_TRUE(gl::classifyName("it_0_lev_1") == gl::NameKind::Normal);
    ASSERT_TRUE(gl::classifyName("") == gl::NameKind::Normal);
}

// classifyName span overload — verdict equals the string (regex) overload
// across the tier pin list plus the over-INT_MAX shape (the scanners parse no
// values), and a mid-buffer span probe with live bytes beyond len (no
// null-terminator dependence; the string overload judges the equivalent copy).
TEST(memory, classify_name_span_matches_string) {
    const std::string cases[] = {
        "int_lev_3_4", "int_lev_0_12", "it_0_lev_1_2", "it_12_lev_0_7",
        "x", "repl_7", "zero", "print_lev_3_4", "int_lev_3",
        "int_lev_3_4_5", "it_0_lev_1", "",
        "int_lev_99999999999_1",
    };
    for (const std::string& s : cases) {
        ASSERT_TRUE(gl::classifyName(gl::StrSpan(s)) == gl::classifyName(s));
    }
    const std::string big = "int_lev_99999999999_1";
    ASSERT_TRUE(gl::classifyName(gl::StrSpan(big)) == gl::NameKind::IntLev);

    // Mid-buffer slice: bytes [4, 4+11) of the buffer are "int_lev_3_4" with
    // live bytes on both sides of the slice.
    const std::string buf = "zzz(int_lev_3_4)zzz";
    const gl::StrSpan mid(buf.data() + 4, 11);
    ASSERT_TRUE(gl::classifyName(mid) == gl::NameKind::IntLev);
    ASSERT_TRUE(gl::classifyName(mid)
                == gl::classifyName(std::string(buf.data() + 4, 11)));
}

// scanSpecialTokens — substring semantics (sregex_iterator), scan order,
// duplicates kept; int and it lists separate.
TEST(memory, scan_special_tokens) {
    const gl::SpecialTokenScan s1 =
        gl::scanSpecialTokens("(in3[int_lev_0_1,it_2_lev_0_3,int_lev_0_1])");
    ASSERT_EQ(s1.intTokens.size(), static_cast<std::size_t>(2)); // duplicate kept
    ASSERT_EQ(s1.intTokens[0], std::string("int_lev_0_1"));
    ASSERT_EQ(s1.intTokens[1], std::string("int_lev_0_1"));
    ASSERT_EQ(s1.itTokens.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(s1.itTokens[0], std::string("it_2_lev_0_3"));

    // Substring catch — deliberate fidelity to the historical direct scan:
    // "print_lev_3_4" contains "int_lev_3_4".
    const gl::SpecialTokenScan s2 = gl::scanSpecialTokens("(P[print_lev_3_4])");
    ASSERT_EQ(s2.intTokens.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(s2.intTokens[0], std::string("int_lev_3_4"));
    ASSERT_TRUE(s2.itTokens.empty());

    const gl::SpecialTokenScan s3 = gl::scanSpecialTokens("(=[x,zero])");
    ASSERT_TRUE(s3.intTokens.empty());
    ASSERT_TRUE(s3.itTokens.empty());

    const gl::SpecialTokenScan s4 = gl::scanSpecialTokens("");
    ASSERT_TRUE(s4.intTokens.empty());
    ASSERT_TRUE(s4.itTokens.empty());
}

// ---------- Parser + eq-class helper span/id twins (I-138) ----------

// chooseCanonicalId — decode(chooseCanonicalId) == chooseCanonical across all
// three tiers, and id 0 mirrors chooseCanonical's empty string for the empty
// and all-weak class (the byte-identity invariant the string form relies on).
TEST(prover, choose_canonical_id_matches_choose_canonical) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::Memory m;

    gl::EquivalenceClass c1;   // normal name wins
    c1.setMembersFromNames({ "it_0_lev_1_2", "int_lev_3_4", "repl_7", "zebra" }, m.nameMap);
    const int16_t id1 = ea.chooseCanonicalId(c1, m, gl::StrSpan("main", 4));
    ASSERT_NE(id1, static_cast<int16_t>(0));
    ASSERT_EQ(m.nameMap.decode(id1), ea.chooseCanonical(c1, m, "main"));

    gl::EquivalenceClass c2;   // int_ tier
    c2.setMembersFromNames({ "it_0_lev_1_2", "int_lev_3_4", "int_lev_1_1" }, m.nameMap);
    ASSERT_EQ(m.nameMap.decode(ea.chooseCanonicalId(c2, m, gl::StrSpan("main", 4))),
              ea.chooseCanonical(c2, m, "main"));

    gl::EquivalenceClass c3;   // it_ tier
    c3.setMembersFromNames({ "it_5_lev_1_2", "it_0_lev_1_2" }, m.nameMap);
    ASSERT_EQ(m.nameMap.decode(ea.chooseCanonicalId(c3, m, gl::StrSpan("main", 4))),
              ea.chooseCanonical(c3, m, "main"));

    // Empty class: id 0, string overload empty.
    gl::EquivalenceClass empty;
    ASSERT_EQ(ea.chooseCanonicalId(empty, m, gl::StrSpan("main", 4)), static_cast<int16_t>(0));
    ASSERT_TRUE(ea.chooseCanonical(empty, m, "main").empty());

    // All-weak class: id 0, string overload empty.
    gl::Memory mWeak;
    mWeak.intWeakVariables.mint(gl::packStatementKey(
        mWeak.nameMap.encode("int_lev_3_4"), gl::NameMap::MAIN_ID));
    mWeak.intWeakVariables.mint(gl::packStatementKey(
        mWeak.nameMap.encode("it_0_lev_1_2"), gl::NameMap::MAIN_ID));
    gl::EquivalenceClass cWeak;
    cWeak.setMembersFromNames({ "int_lev_3_4", "it_0_lev_1_2" }, mWeak.nameMap);
    ASSERT_EQ(ea.chooseCanonicalId(cWeak, mWeak, gl::StrSpan("main", 4)), static_cast<int16_t>(0));
    ASSERT_TRUE(ea.chooseCanonical(cWeak, mWeak, "main").empty());
}

// EqClassNameCaches::kindOf — lazy fill, memoized, classification follows the
// decoded string regardless of mint order.
TEST(memory, eqclass_name_caches_kind_of) {
    gl::Memory mb;
    // Mint in an order where id order disagrees with any tier order.
    const int16_t idIt = mb.nameMap.encode("it_0_lev_1_2");
    const int16_t idNormal = mb.nameMap.encode("repl_3");
    const int16_t idInt = mb.nameMap.encode("int_lev_0_9");

    ASSERT_TRUE(mb.eqClassNameCaches.kindOf(idInt, mb.nameMap) == gl::NameKind::IntLev);
    ASSERT_TRUE(mb.eqClassNameCaches.kindOf(idIt, mb.nameMap) == gl::NameKind::ItLev);
    ASSERT_TRUE(mb.eqClassNameCaches.kindOf(idNormal, mb.nameMap) == gl::NameKind::Normal);
    // Repeated probe serves the memoized slot and stays stable.
    ASSERT_TRUE(mb.eqClassNameCaches.kindOf(idInt, mb.nameMap) == gl::NameKind::IntLev);
    // Slots are recorded compactly per id.
    ASSERT_TRUE(mb.eqClassNameCaches.kindById_.size() >
                static_cast<std::size_t>(idInt > idIt ? (idInt > idNormal ? idInt : idNormal)
                                                      : (idIt > idNormal ? idIt : idNormal)) - 1);
}

// SpecialTokenScanView — the zero-copy view decodes exactly what the
// deserialize oracle decodes from the same canonical blob: counts, token
// bytes, walk order; each cursor exhausts exactly at its section end.
TEST(memory, special_token_scan_view_matches_deserialize_oracle) {
    gl::SpecialTokenScan scans[5];
    // scans[0]: empty/empty.
    scans[1].intTokens = { "int_lev_0_1", "int_lev_2_3" };          // int-only
    scans[2].itTokens = { "it_1_lev_2_3" };                         // it-only
    scans[3].intTokens = { "int_lev_0_1" };                         // both
    scans[3].itTokens = { "it_1_lev_2_3", "it_4_lev_5_6" };
    scans[4].intTokens = { "int_lev_0_1", "int_lev_0_1" };          // duplicates
    scans[4].itTokens = { "it_2_lev_0_3", "it_2_lev_0_3" };

    for (const gl::SpecialTokenScan& s : scans) {
        const std::vector<char> blob = gl::serializeSpecialTokenScan(s);
        const gl::SpecialTokenScanView v{
            blob.data(), static_cast<int32_t>(blob.size()) };
        const gl::SpecialTokenScan ref = gl::deserializeSpecialTokenScan(
            blob.data(), static_cast<int32_t>(blob.size()));

        ASSERT_EQ(v.intCount(), static_cast<int32_t>(ref.intTokens.size()));
        ASSERT_EQ(v.itCount(), static_cast<int32_t>(ref.itTokens.size()));

        gl::SpecialTokenScanView::TokenCursor ic = v.intTokens();
        gl::StrSpan tok;
        std::size_t k = 0;
        while (ic.next(tok)) {
            ASSERT_EQ(tok.toStdString(), ref.intTokens[k]);
            ++k;
        }
        ASSERT_EQ(k, ref.intTokens.size());
        // The int cursor exhausts exactly where the it section begins.
        ASSERT_TRUE(ic.cur == blob.data() + v.itOffset());

        gl::SpecialTokenScanView::TokenCursor tc = v.itTokens();
        k = 0;
        while (tc.next(tok)) {
            ASSERT_EQ(tok.toStdString(), ref.itTokens[k]);
            ++k;
        }
        ASSERT_EQ(k, ref.itTokens.size());
        // The it cursor exhausts exactly at the blob end.
        ASSERT_TRUE(tc.cur == blob.data() + blob.size());
    }
}

// filterIterationsCore view overload — verdicts equal the heap overload's
// across every (scan, class) pair: no-special-member class (canonical 0 ->
// always true), forbidden int / it member tokens, canonical-only tokens,
// interned non-member token, never-interned token, token-free statement.
TEST(memory, filter_iterations_core_view_matches_heap_scan) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::Memory m;

    gl::EquivalenceClass cInt;   // int-canonical class
    cInt.setMembersFromNames({ "int_lev_0_1", "int_lev_0_2", "x" }, m.nameMap);
    gl::EquivalenceClass cIt;    // it-only class
    cIt.setMembersFromNames({ "it_0_lev_1_2", "it_5_lev_1_2" }, m.nameMap);
    gl::EquivalenceClass cNone;  // no special member -> canonical 0
    cNone.setMembersFromNames({ "a", "b" }, m.nameMap);

    // Interned but in no class ("foreign" special token).
    m.nameMap.encode("int_lev_4_4");

    const std::string texts[] = {
        "(in2[int_lev_0_1,7])",   // canonical member of cInt
        "(in2[int_lev_0_2,7])",   // forbidden (non-canonical) int member
        "(P[it_0_lev_1_2])",      // canonical member of cIt
        "(P[it_5_lev_1_2])",      // forbidden it member
        "(in2[int_lev_4_4,7])",   // interned non-member token
        "(q[int_lev_9_9])",       // NEVER-interned token (lookup miss)
        "(=[x,zero])",            // token-free
    };
    const gl::EquivalenceClass* classes[] = { &cInt, &cIt, &cNone };

    for (const std::string& text : texts) {
        // Heap oracle scan + the view over its serialized canonical blob.
        const gl::SpecialTokenScan scan = gl::scanSpecialTokens(text);
        const std::vector<char> blob = gl::serializeSpecialTokenScan(scan);
        const gl::SpecialTokenScanView view{
            blob.data(), static_cast<int32_t>(blob.size()) };
        for (const gl::EquivalenceClass* cls : classes) {
            ASSERT_EQ(ea.filterIterationsCore(view, *cls, m),
                      ea.filterIterationsCore(scan, *cls, m));
        }
    }
}

// tokensViewOf — the memo's deload-facet bytes equal the retired heap
// path's: map A is filled by the ORACLE (scanSpecialTokens + typed
// assignRun, the retired tokensOf miss path replicated verbatim), map B by
// production tokensViewOf; same names minted in the same order in both
// Memories. Asserts key mint order, single-record runs, and per-record blob
// byte equality (the facet 705+1..4 byte proof). Then the hit-path pin
// (replacing the retired tokensOf cache test): a second tokensViewOf per id
// serves the memo — bytes unchanged, no key growth.
TEST(memory, tokens_view_of_facets_match_typed_assign_oracle) {
    gl::Memory mA;   // oracle side
    gl::Memory mB;   // production side
    gl::ScratchArena peekArena{ &gl::staticMemory() };

    const std::string names[] = {
        "(in3[int_lev_0_1,it_2_lev_0_3,int_lev_0_1])",  // duplicates + mixed
        "(P[print_lev_3_4])",                           // substring fidelity
        "(f[it_1_lev_2_3])",                            // it-only
        "(=[x,zero])",                                  // token-free
        "()",                                           // near-empty
    };
    for (const std::string& name : names) {
        const int16_t idA = mA.nameMap.encode(name);
        const int16_t idB = mB.nameMap.encode(name);
        ASSERT_EQ(idA, idB);   // same mint order -> same ids

        // ORACLE: the retired miss path, verbatim.
        gl::SpecialTokenScan scan =
            gl::scanSpecialTokens(mA.nameMap.decode(idA));
        mA.eqClassNameCaches.tokensByExprId_.assignRun(
            idA, std::vector<gl::SpecialTokenScan>{ scan });

        // PRODUCTION: the direct blob build + raw door.
        const gl::SpecialTokenScanView v =
            mB.eqClassNameCaches.tokensViewOf(idB, mB.nameMap, peekArena);
        ASSERT_TRUE(v.len >= 8);   // two counts minimum
    }

    // Facet-byte equality: count, key mint order, run lengths, blob bytes.
    const int32_t cnt = mA.eqClassNameCaches.tokensByExprId_.count();
    ASSERT_EQ(cnt, mB.eqClassNameCaches.tokensByExprId_.count());
    ASSERT_EQ(cnt, static_cast<int32_t>(sizeof(names) / sizeof(names[0])));
    std::vector<char> scratchA, scratchB;
    for (int32_t id = 1; id <= cnt; ++id) {
        ASSERT_EQ(mA.eqClassNameCaches.tokensByExprId_.keyAt(id),
                  mB.eqClassNameCaches.tokensByExprId_.keyAt(id));
        ASSERT_EQ(mA.eqClassNameCaches.tokensByExprId_.runLen(id), 1);
        ASSERT_EQ(mB.eqClassNameCaches.tokensByExprId_.runLen(id), 1);
        int32_t lenA = 0, lenB = 0;
        const char* pa = mA.eqClassNameCaches.tokensByExprId_.peekRecordBytes(
            id, 0, lenA, scratchA);
        const char* pb = mB.eqClassNameCaches.tokensByExprId_.peekRecordBytes(
            id, 0, lenB, scratchB);
        ASSERT_EQ(lenA, lenB);
        ASSERT_TRUE(std::memcmp(pa, pb, static_cast<size_t>(lenA)) == 0);
    }

    // Hit-path pin: repeated probes serve the memoized record.
    for (const std::string& name : names) {
        const int16_t idB = mB.nameMap.lookup(name);
        ASSERT_NE(idB, static_cast<int16_t>(0));
        const gl::SpecialTokenScanView v =
            mB.eqClassNameCaches.tokensViewOf(idB, mB.nameMap, peekArena);
        const int32_t cidA =
            mA.eqClassNameCaches.tokensByExprId_.lookup(idB);
        ASSERT_NE(cidA, 0);
        int32_t lenA = 0;
        const char* pa = mA.eqClassNameCaches.tokensByExprId_.peekRecordBytes(
            cidA, 0, lenA, scratchA);
        ASSERT_EQ(v.len, lenA);
        ASSERT_TRUE(std::memcmp(v.p, pa, static_cast<size_t>(lenA)) == 0);
    }
    ASSERT_EQ(mB.eqClassNameCaches.tokensByExprId_.count(), cnt);
}

// ---------- packEqPairKey / setMembersFromNames / unionMemberIdsByName ----------

// packEqPairKey — unordered-pair identity: {a,b} and {b,a} pack identically;
// distinct pairs pack distinctly; a==b packs naturally.
TEST(memory, pack_eq_pair_key_unordered) {
    const int16_t a = 7, b = 12, c = 300;
    ASSERT_EQ(gl::packEqPairKey(a, b), gl::packEqPairKey(b, a));
    ASSERT_EQ(gl::packEqPairKey(b, c), gl::packEqPairKey(c, b));
    ASSERT_NE(gl::packEqPairKey(a, b), gl::packEqPairKey(a, c));
    ASSERT_NE(gl::packEqPairKey(a, b), gl::packEqPairKey(b, c));
    ASSERT_EQ(gl::packEqPairKey(a, a),
              (static_cast<uint32_t>(7) << 16) | static_cast<uint32_t>(7));
}

// setMembersFromNames — memberIds comes out sorted by DECODED name even
// when mint order disagrees with lex order; repeated calls are idempotent.
TEST(memory, eqclass_set_members_from_names) {
    gl::Memory mb;
    // Mint in reverse lex order so id order != decoded-lex order.
    const int16_t idC = mb.nameMap.encode("it_2_lev_0_3");
    const int16_t idB = mb.nameMap.encode("int_lev_0_9");
    const int16_t idA = mb.nameMap.encode("a_normal");
    ASSERT_TRUE(idC < idB && idB < idA); // mint order, reversed lex

    gl::EquivalenceClass cls;
    cls.setMembersFromNames({ "a_normal", "int_lev_0_9", "it_2_lev_0_3" }, mb.nameMap);

    ASSERT_EQ(cls.memberIds.size(), static_cast<std::size_t>(3));
    // Decoded-lex order, NOT id order (id order would be idC, idB, idA).
    ASSERT_EQ(cls.memberIds[0], idA);
    ASSERT_EQ(cls.memberIds[1], idB);
    ASSERT_EQ(cls.memberIds[2], idC);

    // Idempotent — a second call reproduces the same list.
    cls.setMembersFromNames({ "a_normal", "int_lev_0_9", "it_2_lev_0_3" }, mb.nameMap);
    ASSERT_EQ(cls.memberIds.size(), static_cast<std::size_t>(3));
    ASSERT_EQ(cls.memberIds[0], idA);
}

// unionMemberIdsByName — decoded-lex merge of two sorted member lists;
// duplicates collapse; pure read (no minting).
TEST(memory, union_member_ids_by_name) {
    gl::Memory mb;
    gl::EquivalenceClass left;
    left.setMembersFromNames({ "alpha", "gamma", "zeta" }, mb.nameMap);
    gl::EquivalenceClass right;
    right.setMembersFromNames({ "beta", "gamma" }, mb.nameMap);

    const std::vector<int16_t> u =
        gl::unionMemberIdsByName(left.memberIds, right.memberIds, mb.nameMap);
    ASSERT_EQ(u.size(), static_cast<std::size_t>(4));
    ASSERT_EQ(u[0], mb.nameMap.encode("alpha"));
    ASSERT_EQ(u[1], mb.nameMap.encode("beta"));
    ASSERT_EQ(u[2], mb.nameMap.encode("gamma")); // duplicate collapsed
    ASSERT_EQ(u[3], mb.nameMap.encode("zeta"));

    // Symmetric content; empty operands are identities.
    const std::vector<int16_t> u2 =
        gl::unionMemberIdsByName(right.memberIds, left.memberIds, mb.nameMap);
    ASSERT_EQ(u2.size(), static_cast<std::size_t>(4));
    ASSERT_EQ(u2[0], u[0]);
    const std::vector<int16_t> none;
    const std::vector<int16_t> u3 =
        gl::unionMemberIdsByName(none, left.memberIds, mb.nameMap);
    ASSERT_EQ(u3.size(), left.memberIds.size());
}

// ---------- firstSpecialMemberId / filterIterations / reduceEqClassIds ----------

// firstSpecialMemberId — int tier beats it tier; within a tier the first
// member in decoded-lex storage order (= lex-min) wins; no special -> 0.
TEST(prover, first_special_member_id) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::Memory m;

    gl::EquivalenceClass cBoth;
    cBoth.setMembersFromNames({ "it_0_lev_1_2", "int_lev_9_9", "int_lev_3_4", "zebra" }, m.nameMap);
    ASSERT_EQ(ea.firstSpecialMemberId(cBoth, m),
              m.nameMap.encode("int_lev_3_4"));

    gl::EquivalenceClass cItOnly;
    cItOnly.setMembersFromNames({ "it_5_lev_1_2", "it_0_lev_1_2", "x" }, m.nameMap);
    ASSERT_EQ(ea.firstSpecialMemberId(cItOnly, m),
              m.nameMap.encode("it_0_lev_1_2"));

    gl::EquivalenceClass cNormal;
    cNormal.setMembersFromNames({ "a", "b" }, m.nameMap);
    ASSERT_EQ(ea.firstSpecialMemberId(cNormal, m),
              static_cast<int16_t>(0));
}

// filterIterations — a statement carrying a non-canonical special member of
// the class is rejected; the canonical member itself passes; non-member
// special tokens pass; id and string overloads agree.
TEST(prover, filter_iterations_id_and_string_overloads) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::Memory m;

    gl::EquivalenceClass cls;
    cls.setMembersFromNames({ "int_lev_0_1", "int_lev_0_2", "x" }, m.nameMap);

    const int16_t okId = m.nameMap.encode("(in2[int_lev_0_1,7])");
    const int16_t badId = m.nameMap.encode("(in2[int_lev_0_2,7])");
    const int16_t foreignId = m.nameMap.encode("(in2[int_lev_4_4,7])");

    // Canonical member passes; non-canonical member rejects.
    ASSERT_TRUE(ea.filterIterations(okId, cls, m));
    ASSERT_FALSE(ea.filterIterations(badId, cls, m));
    // A special token that is NOT a class member never rejects.
    ASSERT_TRUE(ea.filterIterations(foreignId, cls, m));

    // String overload returns the same verdicts.
    ASSERT_TRUE(ea.filterIterations(std::string("(in2[int_lev_0_1,7])"), cls, m));
    ASSERT_FALSE(ea.filterIterations(std::string("(in2[int_lev_0_2,7])"), cls, m));
    ASSERT_TRUE(ea.filterIterations(std::string("(in2[int_lev_4_4,7])"), cls, m));

    // it_-canonical class: int canonical absent, it member tested; an it
    // token equal to the canonical passes, a non-canonical it member fails.
    gl::EquivalenceClass cIt;
    cIt.setMembersFromNames({ "it_0_lev_1_2", "it_5_lev_1_2" }, m.nameMap);
    ASSERT_TRUE(ea.filterIterations(std::string("(P[it_0_lev_1_2])"), cIt, m));
    ASSERT_FALSE(ea.filterIterations(std::string("(P[it_5_lev_1_2])"), cIt, m));

    // No special members -> no filtering, everything passes.
    gl::EquivalenceClass cNone;
    cNone.setMembersFromNames({ "a", "b" }, m.nameMap);
    ASSERT_TRUE(ea.filterIterations(badId, cNone, m));
}

// reduceEqClassIds — weak members dropped via the packed twin; decoded-lex
// input order preserved; unknown validity keeps every member. Out-param run
// form (C4 _firing_check): the run matches an inline heap oracle replicating
// the weak filter, across mixed / no-weak / all-weak / exact-cap.
TEST(prover, reduce_eq_class_ids_run_matches_vector) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::Memory m;

    gl::EquivalenceClass cls;
    cls.setMembersFromNames({ "int_lev_0_1", "int_lev_0_2", "zebra" }, m.nameMap);

    m.intWeakVariables.mint(gl::packStatementKey(
        m.nameMap.encode("int_lev_0_1"), gl::NameMap::MAIN_ID));

    // Retained heap oracle — the retired vector-returning form's logic, inline.
    auto oracle = [&](gl::StrSpan validity) {
        std::vector<int16_t> out;
        const int16_t valId = m.nameMap.lookup(validity);
        for (int32_t i = 0; i < cls.memberCount(); ++i) {
            const int16_t id = cls.memberId(i);
            if (valId != 0
                && m.intWeakVariables.contains(gl::packStatementKey(id, valId)))
                continue;
            out.push_back(id);
        }
        return out;
    };

    // Run vs oracle, element-for-element.
    auto check = [&](gl::StrSpan validity) -> int32_t {
        const std::vector<int16_t> expected = oracle(validity);
        int16_t run[16];
        const int32_t w = ea.reduceEqClassIds(cls, m, validity, run, 16);
        ASSERT_EQ(w, static_cast<int32_t>(expected.size()));
        for (int32_t i = 0; i < w; ++i)
            ASSERT_EQ(run[i], expected[static_cast<std::size_t>(i)]);
        return w;
    };

    // Mixed: int_lev_0_1 is weak at main -> 2 strong (int_lev_0_2, zebra).
    int16_t run[16];
    const int32_t wMixed = ea.reduceEqClassIds(cls, m, gl::StrSpan("main", 4), run, 16);
    ASSERT_EQ(wMixed, 2);
    ASSERT_EQ(run[0], m.nameMap.encode("int_lev_0_2"));
    ASSERT_EQ(run[1], m.nameMap.encode("zebra"));
    ASSERT_EQ(check(gl::StrSpan("main", 4)), 2);

    // No-weak: a never-interned validity keeps every member (3).
    ASSERT_EQ(check(gl::StrSpan("never_interned_scope", 20)), 3);

    // Exact-cap: cap == strong count (3) fits exactly (no overflow assert).
    int16_t run3[3];
    const int32_t w3 = ea.reduceEqClassIds(
        cls, m, gl::StrSpan("never_interned_scope", 20), run3, 3);
    ASSERT_EQ(w3, 3);

    // All-weak: mark every member weak at a fresh scope -> w == 0.
    const int16_t sc = m.nameMap.encode("all_weak_scope");
    for (int32_t i = 0; i < cls.memberCount(); ++i)
        m.intWeakVariables.mint(gl::packStatementKey(cls.memberId(i), sc));
    ASSERT_EQ(check(gl::StrSpan("all_weak_scope", 14)), 0);
}


// decodeClassesAt — non-minting probe: an empty list both for a never-interned
// validity and for an interned validity without classes; the decoded list
// otherwise (the cold-blob-store successor of the former classesAt pointer).
TEST(memory, classes_at_probe) {
    gl::Memory m;
    ASSERT_TRUE(m.decodeClassesAt("never_interned_scope").empty());

    m.nameMap.encode("interned_scope");
    ASSERT_TRUE(m.decodeClassesAt("interned_scope").empty());

    gl::EquivalenceClass cls;
    cls.setMembersFromNames({ "a", "b" }, m.nameMap);
    m.assignClassesById(gl::NameMap::MAIN_ID, { cls });

    const std::vector<gl::EquivalenceClass> got = m.decodeClassesAt("main");
    ASSERT_EQ(got.size(), static_cast<std::size_t>(1));
    ASSERT_TRUE(got[0].memberIds == cls.memberIds);
    // Probing never minted: the unrelated scope is still class-free.
    ASSERT_TRUE(m.decodeClassesAt("interned_scope").empty());
}

// ---------- TemplateInterner ----------

// TemplateInterner — id 0 reserved invalid; encode is find-or-mint with
// stable ids; lookup is non-minting with the 0 miss sentinel; decode
// round-trips. The interner is a façade over its LB's cold string table
// (bound by Memory's constructor).
TEST(memory, template_interner_encode_lookup_decode) {
    gl::Memory m;
    gl::TemplateInterner& ti = m.templateInterner;
    ASSERT_EQ(ti.internedCount(), 0);

    // Non-minting miss before any encode.
    ASSERT_EQ(ti.lookup("(in2[x,marker,3])"), static_cast<int16_t>(0));
    ASSERT_EQ(ti.internedCount(), 0); // no mint

    const int16_t idA = ti.encode("(in2[x,marker,3])");
    const int16_t idB = ti.encode("(in2[u_x,int_0_1,u_z])");
    ASSERT_TRUE(idA > 0 && idB > 0);
    ASSERT_NE(idA, idB);

    // Idempotent encode; lookup agrees.
    ASSERT_EQ(ti.encode("(in2[x,marker,3])"), idA);
    ASSERT_EQ(ti.lookup("(in2[x,marker,3])"), idA);
    ASSERT_EQ(ti.lookup("(in2[u_x,int_0_1,u_z])"), idB);

    // Decode round-trips.
    ASSERT_EQ(ti.decode(idA), std::string("(in2[x,marker,3])"));
    ASSERT_EQ(ti.decode(idB), std::string("(in2[u_x,int_0_1,u_z])"));
}

// TemplateInterner — the id space is independent of the NameMap: interning
// templates never mints NameMap ids, and equal strings get unrelated ids in
// the two spaces.
TEST(memory, template_interner_space_is_disjoint_from_namemap) {
    gl::Memory m;
    const int32_t nameMapSizeBefore = m.nameMap.nameCount();

    const int16_t tid = m.templateInterner.encode("(p[a,marker])");
    ASSERT_TRUE(tid > 0);
    // NameMap untouched — the whole point of the dedicated space.
    ASSERT_EQ(m.nameMap.nameCount(), nameMapSizeBefore);
    ASSERT_EQ(m.nameMap.lookup("(p[a,marker])"), static_cast<int16_t>(0));

    // Packed admission key: template id high half, validity id low half.
    const int32_t pk = gl::packStatementKey(tid, gl::NameMap::MAIN_ID);
    ASSERT_EQ(static_cast<int16_t>((static_cast<uint32_t>(pk) >> 16) & 0xFFFF), tid);
    ASSERT_EQ(static_cast<int16_t>(static_cast<uint32_t>(pk) & 0xFFFF),
              static_cast<int16_t>(gl::NameMap::MAIN_ID));
}


// mintTemplateKey / lookupTemplateKey / decodeTemplateKey — write half mints
// both spaces; read half is non-minting with definitive misses; decode
// round-trips the pair.
TEST(memory, template_key_helpers) {
    gl::Memory m;

    // Read-half misses before any mint: unknown template, then unknown
    // validity.
    int32_t pk = 0;
    ASSERT_FALSE(gl::lookupTemplateKey(m.templateInterner, m.nameMap,
                                       "(p[a,marker])", "main", pk));
    const int32_t internerCountBefore = m.templateInterner.internedCount();

    const int32_t minted = gl::mintTemplateKey(m.templateInterner, m.nameMap,
                                               "(p[a,marker])", "main");
    ASSERT_TRUE(gl::lookupTemplateKey(m.templateInterner, m.nameMap,
                                      "(p[a,marker])", "main", pk));
    ASSERT_EQ(pk, minted);
    ASSERT_TRUE(m.templateInterner.internedCount() == internerCountBefore + 1);

    // Unknown validity is a definitive miss even with the template interned.
    ASSERT_FALSE(gl::lookupTemplateKey(m.templateInterner, m.nameMap,
                                       "(p[a,marker])", "never_interned_scope", pk));

    const auto tv = gl::decodeTemplateKey(minted, m.templateInterner, m.nameMap);
    ASSERT_EQ(tv.first, std::string("(p[a,marker])"));
    ASSERT_EQ(tv.second, std::string("main"));
}

// ---------- Index/key probe span twins (I-138) ----------

// lookupStatementFlags / lookupStatementLevels / isLocalEncodedStatement span
// overloads — each verdict/pointer/id identical to the std::string form on a
// stored entry, and a never-interned span is a definitive miss that mints
// nothing (I-84).
TEST(memory, statement_index_span_probes_match_string) {
    gl::Memory m;
    const gl::EncodedExpression e("(in[1,2])", "main");
    const gl::IntEncodedExpr ie = gl::encodeExpression(e, m.nameMap);
    const int32_t pk = gl::packStatementKey(ie.originalId, ie.validityId);
    m.intLocalEncodedStatementsSet.mint(pk);
    m.intStatementLevelsMap.insertSorted(pk, 0);
    m.intStatementLevelsMap.insertSorted(pk, 2);
    gl::upsertStatementKey(m.intKnownStatements, pk,
        /*local=*/true, /*registered=*/true, /*known=*/true);

    const std::string origS = "(in[1,2])";
    const std::string valS  = "main";
    const gl::StrSpan orig(origS);
    const gl::StrSpan val(valS);

    // Flags: span form pointer-identical to the string form (same cold row).
    ASSERT_TRUE(gl::lookupStatementFlags(m.intKnownStatements, m.nameMap, orig, val)
             == gl::lookupStatementFlags(m.intKnownStatements, m.nameMap, origS, valS));
    ASSERT_TRUE(gl::lookupStatementFlags(m.intKnownStatements, m.nameMap, orig, val)
             != nullptr);

    // Levels: same cold key id; local-origin membership: same verdict.
    ASSERT_EQ(gl::lookupStatementLevels(m.intStatementLevelsMap, m.nameMap, orig, val),
              gl::lookupStatementLevels(m.intStatementLevelsMap, m.nameMap, origS, valS));
    ASSERT_EQ(gl::isLocalEncodedStatement(m.intLocalEncodedStatementsSet, m.nameMap, orig, val),
              gl::isLocalEncodedStatement(m.intLocalEncodedStatementsSet, m.nameMap, origS, valS));
    ASSERT_TRUE(gl::isLocalEncodedStatement(m.intLocalEncodedStatementsSet, m.nameMap, orig, val));

    // Never-interned span: definitive miss on all three, and nothing minted.
    const int32_t before = m.nameMap.nameCount();
    const std::string missS = "(in[9,9])";
    const gl::StrSpan miss(missS);
    ASSERT_TRUE(gl::lookupStatementFlags(m.intKnownStatements, m.nameMap, miss, val) == nullptr);
    ASSERT_EQ(gl::lookupStatementLevels(m.intStatementLevelsMap, m.nameMap, miss, val), 0);
    ASSERT_FALSE(gl::isLocalEncodedStatement(m.intLocalEncodedStatementsSet, m.nameMap, miss, val));
    ASSERT_EQ(m.nameMap.nameCount(), before);
}

// mintTemplateKey / lookupTemplateKey span overloads + decodeTemplateKeyView —
// span mint yields the same packed key as string mint; span lookup agrees on
// hit and misses cleanly (outKey untouched); the view pair is byte-equal to the
// decodeTemplateKey owning pair.
TEST(memory, template_key_span_overloads_match_string) {
    gl::Memory m;
    const std::string templS = "(in2[u_a,marker,3])";
    const std::string valS   = "main";

    const int32_t viaStr  = gl::mintTemplateKey(m.templateInterner, m.nameMap,
                                                templS, valS);
    const int32_t viaSpan = gl::mintTemplateKey(m.templateInterner, m.nameMap,
        gl::StrSpan(templS), gl::StrSpan(valS));
    ASSERT_EQ(viaSpan, viaStr);   // idempotent interners -> same packed key

    int32_t keySpan = 0, keyStr = 0;
    const bool okSpan = gl::lookupTemplateKey(m.templateInterner, m.nameMap,
        gl::StrSpan(templS), gl::StrSpan(valS), keySpan);
    const bool okStr = gl::lookupTemplateKey(m.templateInterner, m.nameMap,
        templS, valS, keyStr);
    ASSERT_TRUE(okSpan && okStr);
    ASSERT_EQ(keySpan, keyStr);
    ASSERT_EQ(keySpan, viaStr);

    // Span lookup miss: unknown validity -> false, outKey left untouched.
    int32_t sink = -12345;
    ASSERT_FALSE(gl::lookupTemplateKey(m.templateInterner, m.nameMap,
        gl::StrSpan(templS), gl::StrSpan(std::string("never_interned_scope")), sink));
    ASSERT_EQ(sink, -12345);

    // decodeTemplateKeyView spans byte-equal to decodeTemplateKey strings.
    gl::StrSpan svKey, svVal;
    gl::decodeTemplateKeyView(viaStr, m.templateInterner, m.nameMap, svKey, svVal);
    const auto ss = gl::decodeTemplateKey(viaStr, m.templateInterner, m.nameMap);
    ASSERT_TRUE(gl::equalSpans(svKey, gl::StrSpan(ss.first)));
    ASSERT_TRUE(gl::equalSpans(svVal, gl::StrSpan(ss.second)));
    ASSERT_EQ(svKey.toStdString(), templS);
    ASSERT_EQ(svVal.toStdString(), valS);
}

// encodeEqClassKeyFromViewScratch — the arena-built packed eq-class key is
// byte-identical to the heap encodeEqClassKeyFromView, multi-member and empty
// class; length is 2*(memberCount+1) bytes.
TEST(memory, encode_eq_class_key_from_view_scratch_matches_heap) {
    gl::Memory m;                       // real NameMap on the global pool
    gl::GlobalMemoryManager g;          // separate pool for the scratch arena
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::ScratchArena arena; arena.bind(&g);

    gl::EquivalenceClass cls;
    cls.setMembersFromNames({ "int_lev_0_1", "int_lev_0_2", "x" }, m.nameMap);
    const std::vector<char> blob = gl::serializeEquivalenceClass(cls);
    const gl::EquivalenceClassView view{ blob.data(),
                                         static_cast<int32_t>(blob.size()) };
    const int16_t vid = gl::NameMap::MAIN_ID;

    const std::string heapKey = gl::encodeEqClassKeyFromView(vid, view);
    const gl::ScratchString scratchKey =
        gl::encodeEqClassKeyFromViewScratch(arena, vid, view);
    ASSERT_TRUE(gl::equalSpans(gl::StrSpan(scratchKey), gl::StrSpan(heapKey)));
    ASSERT_EQ(static_cast<int32_t>(scratchKey.size()),
              static_cast<int32_t>(2 * (view.memberCount() + 1)));

    // Empty class: just the 2-byte validity id, still byte-identical.
    gl::EquivalenceClass empty;
    const std::vector<char> emptyBlob = gl::serializeEquivalenceClass(empty);
    const gl::EquivalenceClassView emptyView{ emptyBlob.data(),
        static_cast<int32_t>(emptyBlob.size()) };
    ASSERT_TRUE(gl::equalSpans(
        gl::StrSpan(gl::encodeEqClassKeyFromViewScratch(arena, vid, emptyView)),
        gl::StrSpan(gl::encodeEqClassKeyFromView(vid, emptyView))));
}

// encodeEqClassKeyFromViewInto — the stack-buffer serializer is byte-identical
// to the heap encodeEqClassKeyFromView (validity LE, then each member LE),
// multi-member and empty; the returned length is 2*(memberCount+1).
TEST(memory, encode_eqclass_key_from_view_into_matches_heap) {
    gl::Memory m;
    gl::EquivalenceClass cls;
    cls.setMembersFromNames({ "int_lev_0_1", "int_lev_0_2", "x" }, m.nameMap);
    const std::vector<char> blob = gl::serializeEquivalenceClass(cls);
    const gl::EquivalenceClassView view{ blob.data(),
                                         static_cast<int32_t>(blob.size()) };
    const int16_t vid = gl::NameMap::MAIN_ID;

    char buf[gl::ExecutionParameters::kMaxEqClassKeyBytes];
    const int32_t n = gl::encodeEqClassKeyFromViewInto(
        buf, gl::ExecutionParameters::kMaxEqClassKeyBytes, vid, view);
    ASSERT_EQ(n, static_cast<int32_t>(2 * (view.memberCount() + 1)));
    ASSERT_TRUE(gl::equalSpans(gl::StrSpan(buf, n),
        gl::StrSpan(gl::encodeEqClassKeyFromView(vid, view))));

    // Empty class: just the 2-byte validity id.
    gl::EquivalenceClass empty;
    const std::vector<char> eb = gl::serializeEquivalenceClass(empty);
    const gl::EquivalenceClassView ev{ eb.data(),
                                       static_cast<int32_t>(eb.size()) };
    const int32_t en = gl::encodeEqClassKeyFromViewInto(
        buf, gl::ExecutionParameters::kMaxEqClassKeyBytes, vid, ev);
    ASSERT_TRUE(gl::equalSpans(gl::StrSpan(buf, en),
        gl::StrSpan(gl::encodeEqClassKeyFromView(vid, ev))));
}

// encodeEqClassKeyFromAccumInto — the stack-buffer serializer is byte-identical
// to the heap encodeEqClassKeyFromAccum, multi-member and empty.
TEST(memory, encode_eqclass_key_from_accum_into_matches_heap) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::MergeClassAccum accum(&lb);
    accum.addMember(3); accum.addMember(7); accum.addMember(12);
    const int16_t vid = static_cast<int16_t>(5);

    char buf[gl::ExecutionParameters::kMaxEqClassKeyBytes];
    const int32_t n = gl::encodeEqClassKeyFromAccumInto(
        buf, gl::ExecutionParameters::kMaxEqClassKeyBytes, vid, accum);
    ASSERT_EQ(n, static_cast<int32_t>(2 * (accum.members.size() + 1)));
    ASSERT_TRUE(gl::equalSpans(gl::StrSpan(buf, n),
        gl::StrSpan(gl::encodeEqClassKeyFromAccum(vid, accum))));

    // Empty accumulator: just the 2-byte validity id.
    gl::MergeClassAccum empty(&lb);
    const int32_t en = gl::encodeEqClassKeyFromAccumInto(
        buf, gl::ExecutionParameters::kMaxEqClassKeyBytes, vid, empty);
    ASSERT_TRUE(gl::equalSpans(gl::StrSpan(buf, en),
        gl::StrSpan(gl::encodeEqClassKeyFromAccum(vid, empty))));
}

// ---------- ValueInterner / decoded value compares ----------

// ValueInterner — int32 ids, slot 0 reserved; encode find-or-mint, lookup
// non-minting, decode round-trips; space independent of NameMap and
// TemplateInterner.
TEST(memory, value_interner_basics) {
    gl::Memory m;
    ASSERT_EQ(m.valueInterner.lookup("(in3[a,b,c,plus])"), 0);
    const int32_t nameMapBefore = m.nameMap.nameCount();
    const int32_t templBefore = m.templateInterner.internedCount();

    const int32_t idA = m.valueInterner.encode("(in3[a,b,c,plus])");
    const int32_t idB = m.valueInterner.encode("(in[b,1])");
    ASSERT_TRUE(idA > 0 && idB > 0);
    ASSERT_NE(idA, idB);
    ASSERT_EQ(m.valueInterner.encode("(in3[a,b,c,plus])"), idA);
    ASSERT_EQ(m.valueInterner.lookup("(in[b,1])"), idB);
    ASSERT_EQ(m.valueInterner.decode(idA), std::string("(in3[a,b,c,plus])"));

    // The other id spaces are untouched.
    ASSERT_EQ(m.nameMap.nameCount(), nameMapBefore);
    ASSERT_EQ(m.templateInterner.internedCount(), templBefore);
}

// ---------- Interner span-overload byte twins (I-138) ----------

// NameMap::decodeSubView — zero-copy payload span byte-identical to the
// decodeSub owning copy, on every payload shape and a sub-slice source.
TEST(memory, namemap_decode_sub_view_matches_decode_sub) {
    NameMapRig nmRig;
    gl::NameMap& nm = nmRig.nm;
    const int16_t sid1 = nm.encodeSub("child");
    const int16_t sid2 = nm.encodeSub("x_lev_3");
    // Idempotent: re-encoding a payload returns the same sub-id.
    ASSERT_EQ(nm.encodeSub("child"), sid1);
    // Span equals the owning copy byte-for-byte, and toStdString round-trips.
    ASSERT_TRUE(gl::equalSpans(nm.decodeSubView(sid1),
                               gl::StrSpan(nm.decodeSub(sid1))));
    ASSERT_EQ(nm.decodeSubView(sid1).toStdString(), nm.decodeSub(sid1));
    ASSERT_TRUE(gl::equalSpans(nm.decodeSubView(sid2),
                               gl::StrSpan(nm.decodeSub(sid2))));
    ASSERT_EQ(nm.decodeSubView(sid2).toStdString(), std::string("x_lev_3"));
}

// TemplateInterner::encode/lookup span overloads — byte-identical id to the
// std::string forms on hit, miss, sub-slice, and order-independence.
TEST(memory, template_interner_span_overloads_byte_identical) {
    gl::Memory m;
    gl::TemplateInterner& ti = m.templateInterner;

    // Span encode mints the same id as string encode (hit, not a second mint).
    const int16_t viaStr = ti.encode(std::string("(in2[x,marker,3])"));
    const int16_t viaSpan = ti.encode(gl::StrSpan(std::string("(in2[x,marker,3])")));
    ASSERT_EQ(viaSpan, viaStr);
    ASSERT_EQ(ti.internedCount(), 1);

    // Span lookup agrees with string lookup; a never-interned span is 0.
    ASSERT_EQ(ti.lookup(gl::StrSpan(std::string("(in2[x,marker,3])"))), viaStr);
    ASSERT_EQ(ti.lookup(gl::StrSpan(std::string("(never[y])"))),
              static_cast<int16_t>(0));
    ASSERT_EQ(ti.internedCount(), 1); // lookup never mints

    // Sub-slice safety: a span over the first 5 bytes interns "alpha", never
    // "alphaZZZ" (only [ptr, len) is read).
    const std::string buf = "alphaZZZ";
    const int16_t sliced = ti.encode(gl::StrSpan(buf.data(), 5));
    ASSERT_EQ(sliced, ti.encode(std::string("alpha")));
    ASSERT_EQ(ti.lookup(gl::StrSpan(buf.data(), 5)),
              ti.lookup(std::string("alpha")));
    ASSERT_NE(sliced, ti.encode(std::string("alphaZZZ")));
}

// ValueInterner::lookup span overload — byte-identical id to lookup(string) on
// hit/miss, agrees with the existing span-form encode, sub-slice safe.
TEST(memory, value_interner_span_lookup_byte_identical) {
    gl::Memory m;
    gl::ValueInterner& vi = m.valueInterner;

    const int32_t id = vi.encode(std::string("(in3[a,b,c,plus])"));
    ASSERT_EQ(vi.lookup(gl::StrSpan(std::string("(in3[a,b,c,plus])"))), id);
    ASSERT_EQ(vi.lookup(gl::StrSpan(std::string("(never)"))), 0);

    // Span lookup agrees with the pre-existing span-form encode.
    const int32_t id2 = vi.encode(gl::StrSpan(std::string("(in[b,1])")));
    ASSERT_EQ(vi.lookup(gl::StrSpan(std::string("(in[b,1])"))), id2);

    // Sub-slice: "value" absent until interned, then span-slice == string.
    const std::string buf = "valueZZZ";
    ASSERT_EQ(vi.lookup(gl::StrSpan(buf.data(), 5)), 0);
    vi.encode(std::string("value"));
    ASSERT_EQ(vi.lookup(gl::StrSpan(buf.data(), 5)),
              vi.lookup(std::string("value")));
}

// encodeValueSpanSetSorted — span twin of encodeValueSetSorted. On a FRESH
// value interner the returned id vector is byte-identical (same id VALUES at
// the same positions) to the std::set<std::string>+encodeValueSetSorted oracle:
// both mint the distinct members in decoded-lex order, so the mint ordinals
// (== ids) coincide. Covers reverse-lex input with a duplicate, the length
// tiebreak, a single element, and empty.
TEST(absorb_door_span_overloads, encode_value_span_set_sorted_byte_identical) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 22, 1 << 18 });

    // Two independent value interners (oracle vs twin) on one arena; a separate
    // scratch arena feeds the twin's transient dedup set + sort index.
    gl::LbArena lb(&g);
    gl::LbArena scratch(&g);
    gl::DirtyState dO = gl::DirtyState::Clean;
    gl::DirtyState dT = gl::DirtyState::Clean;
    gl::ColdStringTable oracleTable(&lb, &dO);
    gl::ColdStringTable twinTable(&lb, &dT);
    gl::ValueInterner oracleVi;
    gl::ValueInterner twinVi;
    oracleVi.bind(&oracleTable);
    twinVi.bind(&twinTable);

    auto runCase = [&](const std::vector<std::string>& raw) {
        // Fresh interners each case so ids restart at 1 (mint order is the id).
        oracleTable.resetToFresh();
        twinTable.resetToFresh();

        // Oracle: std::set dedups + lex-sorts, minted in iteration order.
        const std::set<std::string> oset(raw.begin(), raw.end());
        const std::vector<int32_t> oracleIds =
            gl::encodeValueSetSorted(oset, oracleVi);

        // Twin: same members as spans over the stable `raw` strings; the
        // out-param form fills a caller run and returns the distinct count.
        std::vector<gl::StrSpan> spans;
        spans.reserve(raw.size());
        for (const std::string& s : raw) spans.push_back(gl::StrSpan(s));
        int32_t twinBuf[gl::ExecutionParameters::MAX_ADMISSION_REM_ARGS];
        const int32_t twinN = gl::encodeValueSpanSetSorted(
            spans.data(), static_cast<int>(spans.size()), twinVi, scratch,
            twinBuf, gl::ExecutionParameters::MAX_ADMISSION_REM_ARGS);
        const std::vector<int32_t> twinIds(twinBuf, twinBuf + twinN);

        // Same distinct count, same id values in the same (lex) positions, and
        // each id decodes to the same member on its own fresh interner.
        ASSERT_EQ(twinIds.size(), oracleIds.size());
        ASSERT_TRUE(twinIds == oracleIds);
        for (std::size_t i = 0; i < twinIds.size(); ++i)
            ASSERT_EQ(twinVi.decode(twinIds[i]), oracleVi.decode(oracleIds[i]));
    };

    runCase({ "(in[c])", "(in[b])", "(in[a])", "(in[b])" }); // reverse-lex + dup
    runCase({ "(=[x,y])", "(=[x])", "(=[x,y,z])" });         // length tiebreak
    runCase({ "(zero[])" });                                 // single element
    runCase({});                                             // empty
}

// valueIdLess / valueIdVectorLess — decoded-string order, independent of
// mint order; vector compare reproduces vector<string> lexicographic rule.
TEST(memory, value_id_decoded_compares) {
    gl::Memory m;
    // Mint in reverse lex order so id order != decoded order.
    const int32_t idC = m.valueInterner.encode("ccc");
    const int32_t idB = m.valueInterner.encode("bbb");
    const int32_t idA = m.valueInterner.encode("aaa");
    ASSERT_TRUE(idC < idB && idB < idA); // mint order, reversed lex

    ASSERT_TRUE(gl::valueIdLess(idA, idB, m.valueInterner));
    ASSERT_TRUE(gl::valueIdLess(idB, idC, m.valueInterner));
    ASSERT_FALSE(gl::valueIdLess(idC, idA, m.valueInterner));
    ASSERT_FALSE(gl::valueIdLess(idA, idA, m.valueInterner));

    // Vector rule: element-wise decoded compare, then size tiebreak.
    const std::vector<int32_t> va{ idA, idB };
    const std::vector<int32_t> vb{ idA, idC };
    const std::vector<int32_t> vshort{ idA };
    ASSERT_TRUE(gl::valueIdVectorLess(va, vb, m.valueInterner));   // bbb < ccc
    ASSERT_FALSE(gl::valueIdVectorLess(vb, va, m.valueInterner));
    ASSERT_TRUE(gl::valueIdVectorLess(vshort, va, m.valueInterner)); // prefix shorter
    ASSERT_FALSE(gl::valueIdVectorLess(va, va, m.valueInterner));
}

// ---------- Origin-map id infrastructure ----------

// OriginTag tables — every enum value round-trips through its historical
// string; the vocabulary is closed and index-aligned.
TEST(memory, origin_tag_round_trip) {
    for (std::size_t i = 0; i < static_cast<std::size_t>(gl::OriginTag::COUNT); ++i) {
        const gl::OriginTag t = static_cast<gl::OriginTag>(i);
        const char* name = gl::originTagName(t);
        ASSERT_TRUE(name != nullptr && name[0] != '\0');
        ASSERT_TRUE(gl::originTagFromString(std::string(name)) == t);
    }
    // Spot-check the exact historical literals the policy code names.
    ASSERT_TRUE(gl::originTagFromString("equality1") == gl::OriginTag::equality1);
    ASSERT_TRUE(gl::originTagFromString("equality2") == gl::OriginTag::equality2);
    ASSERT_TRUE(gl::originTagFromString("implication") == gl::OriginTag::implication);
    ASSERT_EQ(std::string(gl::originTagName(gl::OriginTag::reformulationForIntegrationBound)),
              std::string("reformulation for integration >[bound]"));
    ASSERT_EQ(std::string(gl::originTagName(gl::OriginTag::ceBuildingBlock)),
              std::string("CE_building_block"));
}

// packOriginKey / mintOriginKey / lookupOriginKey / decodeOriginKey —
// bijective packing in the dedicated origin space; non-minting probe is a
// definitive miss; NameMap untouched.
TEST(memory, origin_key_helpers) {
    gl::Memory m;
    const int32_t nameMapBefore = m.nameMap.nameCount();

    ASSERT_EQ(m.originInterner.lookup("(=[a,b])"), 0);
    int64_t pk = 0;
    ASSERT_FALSE(gl::lookupOriginKey(m.originInterner, "(=[a,b])", "main", pk));

    const int64_t minted = gl::mintOriginKey(m.originInterner, "(=[a,b])", "main");
    ASSERT_TRUE(gl::lookupOriginKey(m.originInterner, "(=[a,b])", "main", pk));
    ASSERT_EQ(pk, minted);

    // Same expression under a never-interned validity is a definitive miss.
    ASSERT_FALSE(gl::lookupOriginKey(m.originInterner, "(=[a,b])", "never_scope", pk));

    // Distinct pairs pack to distinct keys; re-mint is stable.
    const int64_t minted2 = gl::mintOriginKey(m.originInterner, "(=[b,a])", "main");
    ASSERT_NE(minted2, minted);
    ASSERT_EQ(gl::mintOriginKey(m.originInterner, "(=[a,b])", "main"), minted);

    const auto kv = gl::decodeOriginKey(minted, m.originInterner);
    ASSERT_EQ(kv.first, std::string("(=[a,b])"));
    ASSERT_EQ(kv.second, std::string("main"));

    // The origin space is disjoint from the NameMap.
    ASSERT_EQ(m.nameMap.nameCount(), nameMapBefore);
}

// decodeOriginKeyView — zero-copy twin of decodeOriginKey. For a minted key the
// out-param spans are byte-equal to the owning decodeOriginKey pair (both the
// expression and validity halves), across two distinct (expr, validity) pairs.
TEST(memory, decode_origin_key_view_matches_decode_origin_key) {
    gl::Memory m;
    const int64_t k1 = gl::mintOriginKey(m.originInterner, "(=[a,b])", "main");
    const int64_t k2 = gl::mintOriginKey(m.originInterner, "(=[b,a])", "sub_1");

    for (const int64_t pk : { k1, k2 }) {
        const auto owned = gl::decodeOriginKey(pk, m.originInterner);
        gl::StrSpan eSpan, vSpan;
        gl::decodeOriginKeyView(pk, m.originInterner, eSpan, vSpan);
        ASSERT_TRUE(gl::equalSpans(eSpan, gl::StrSpan(owned.first)));
        ASSERT_TRUE(gl::equalSpans(vSpan, gl::StrSpan(owned.second)));
        ASSERT_EQ(eSpan.toStdString(), owned.first);
        ASSERT_EQ(vSpan.toStdString(), owned.second);
    }
}

// lookupOriginKey(StrSpan,StrSpan) — span overload returns the identical
// verdict and packed key as the std::string overload: a miss before mint, the
// exact minted key after, and a definitive miss under an un-interned validity.
TEST(memory, origin_key_lookup_span_matches_string) {
    gl::Memory m;
    const std::string expr("(=[a,b])");
    const std::string vld("main");
    const std::string neverVld("never_scope");

    int64_t pkStr = 0;
    int64_t pkSpan = 0;

    // Before mint: both overloads are a definitive miss.
    ASSERT_FALSE(gl::lookupOriginKey(m.originInterner, expr, vld, pkStr));
    ASSERT_FALSE(gl::lookupOriginKey(m.originInterner,
        gl::StrSpan(expr), gl::StrSpan(vld), pkSpan));

    const int64_t minted = gl::mintOriginKey(m.originInterner, expr, vld);

    // After mint: span verdict + key byte-identical to the string form.
    ASSERT_TRUE(gl::lookupOriginKey(m.originInterner, expr, vld, pkStr));
    ASSERT_TRUE(gl::lookupOriginKey(m.originInterner,
        gl::StrSpan(expr), gl::StrSpan(vld), pkSpan));
    ASSERT_EQ(pkSpan, minted);
    ASSERT_EQ(pkSpan, pkStr);

    // Un-interned validity is a definitive miss under both overloads.
    ASSERT_FALSE(gl::lookupOriginKey(m.originInterner, expr, neverVld, pkStr));
    ASSERT_FALSE(gl::lookupOriginKey(m.originInterner,
        gl::StrSpan(expr), gl::StrSpan(neverVld), pkSpan));
}

// encodeOrigin / decodeOrigin — tag and dependency round trip with
// positional order preserved.
TEST(memory, origin_record_round_trip) {
    gl::Memory m;
    std::pair<std::string, std::vector<gl::ExpressionWithValidity>> origin;
    origin.first = "implication";
    origin.second.emplace_back("(in3[a,b,c,plus])", "main");
    origin.second.emplace_back("(in[b,1])", "main_boundary_s1");

    const gl::IdOrigin enc = gl::encodeOrigin(origin, m.originInterner);
    ASSERT_TRUE(enc.first == gl::OriginTag::implication);
    ASSERT_EQ(enc.second.size(), static_cast<std::size_t>(2));

    const auto dec = gl::decodeOrigin(enc, m.originInterner);
    ASSERT_EQ(dec.first, origin.first);
    ASSERT_EQ(dec.second.size(), origin.second.size());
    ASSERT_TRUE(dec.second[0] == origin.second[0]);
    ASSERT_TRUE(dec.second[1] == origin.second[1]);
}

// addOriginId — below-cap dedup append; at-cap D-49 preference: a NEW
// non-equality record replaces the first equality-convenience slot;
// equality-vs-equality and foundational-vs-foundational keep insertion
// order.
TEST(memory, add_origin_id_cap_policy) {
    gl::Memory m;
    gl::IdOriginMap map;
    const int64_t key = gl::mintOriginKey(m.originInterner, "(=[a,b])", "main");

    gl::IdOrigin eq1(gl::OriginTag::equality1,
        std::vector<int64_t>{ gl::mintOriginKey(m.originInterner, "(=[b,a])", "main") });
    gl::IdOrigin impl(gl::OriginTag::implication,
        std::vector<int64_t>{ gl::mintOriginKey(m.originInterner, "(in[b,1])", "main") });
    gl::IdOrigin rec(gl::OriginTag::recursion, std::vector<int64_t>{});

    // Below cap: append + dedup.
    gl::addOriginId(map, key, eq1, 2);
    gl::addOriginId(map, key, eq1, 2);
    ASSERT_EQ(map.at(key).size(), static_cast<std::size_t>(1));
    gl::addOriginId(map, key, impl, 2);
    ASSERT_EQ(map.at(key).size(), static_cast<std::size_t>(2));

    // At cap, already present: kept, no growth.
    gl::addOriginId(map, key, impl, 2);
    ASSERT_EQ(map.at(key).size(), static_cast<std::size_t>(2));

    // At cap, NEW foundational record: replaces the equality1 slot.
    gl::addOriginId(map, key, rec, 2);
    ASSERT_EQ(map.at(key).size(), static_cast<std::size_t>(2));
    ASSERT_TRUE(map.at(key)[0] == rec);   // slot 0 was eq1
    ASSERT_TRUE(map.at(key)[1] == impl);

    // At cap, NEW equality record: dropped (insertion order wins).
    gl::IdOrigin eq2(gl::OriginTag::equality2, std::vector<int64_t>{});
    gl::addOriginId(map, key, eq2, 2);
    ASSERT_TRUE(map.at(key)[0] == rec);
    ASSERT_TRUE(map.at(key)[1] == impl);

    // At cap, NEW foundational, no equality slot left: dropped.
    gl::IdOrigin theo(gl::OriginTag::theorem, std::vector<int64_t>{});
    gl::addOriginId(map, key, theo, 2);
    ASSERT_TRUE(map.at(key)[0] == rec);
    ASSERT_TRUE(map.at(key)[1] == impl);
}

// overwriteOriginsId — per-key append in source vector order with dedup,
// capped; keys independent.
TEST(memory, overwrite_origins_id) {
    gl::Memory m;
    gl::IdOriginMap left, right;
    const int64_t k1 = gl::mintOriginKey(m.originInterner, "(p[a])", "main");
    const int64_t k2 = gl::mintOriginKey(m.originInterner, "(p[b])", "main");

    gl::IdOrigin o1(gl::OriginTag::implication, std::vector<int64_t>{});
    gl::IdOrigin o2(gl::OriginTag::expansion, std::vector<int64_t>{});
    gl::IdOrigin o3(gl::OriginTag::theorem, std::vector<int64_t>{});

    left[k1].push_back(o1);
    right[k1].push_back(o1);  // duplicate — must not re-append
    right[k1].push_back(o2);
    right[k1].push_back(o3);  // beyond cap 2 — must be dropped
    right[k2].push_back(o3);

    gl::overwriteOriginsId(left, right, 2);
    ASSERT_EQ(left.at(k1).size(), static_cast<std::size_t>(2));
    ASSERT_TRUE(left.at(k1)[0] == o1);
    ASSERT_TRUE(left.at(k1)[1] == o2);
    ASSERT_EQ(left.at(k2).size(), static_cast<std::size_t>(1));
    ASSERT_TRUE(left.at(k2)[0] == o3);
}

// decodeOriginMapSorted — rows sorted by decoded (expression, validity)
// regardless of mint/insertion order; per-row lines keep insertion order.
TEST(memory, decode_origin_map_sorted) {
    gl::Memory m;
    gl::IdOriginMap map;
    // Mint in reverse lex order so id order != decoded order.
    const int64_t kC = gl::mintOriginKey(m.originInterner, "(p[c])", "main");
    const int64_t kB = gl::mintOriginKey(m.originInterner, "(p[b])", "zz_scope");
    const int64_t kB2 = gl::mintOriginKey(m.originInterner, "(p[b])", "main");

    gl::IdOrigin o1(gl::OriginTag::implication, std::vector<int64_t>{});
    gl::IdOrigin o2(gl::OriginTag::equality1,
        std::vector<int64_t>{ gl::mintOriginKey(m.originInterner, "(p[d])", "main") });
    map[kC].push_back(o1);
    map[kB].push_back(o1);
    map[kB2].push_back(o2);
    map[kB2].push_back(o1);  // insertion order: o2 then o1

    const auto rows = gl::decodeOriginMapSorted(map, m.originInterner);
    ASSERT_EQ(rows.size(), static_cast<std::size_t>(3));
    // (p[b], main) < (p[b], zz_scope) < (p[c], main) — EWV operator< order.
    ASSERT_EQ(rows[0].first.first, std::string("(p[b])"));
    ASSERT_EQ(rows[0].first.second, std::string("main"));
    ASSERT_EQ(rows[1].first.first, std::string("(p[b])"));
    ASSERT_EQ(rows[1].first.second, std::string("zz_scope"));
    ASSERT_EQ(rows[2].first.first, std::string("(p[c])"));
    // Per-row line order is insertion order.
    ASSERT_EQ(rows[0].second.size(), static_cast<std::size_t>(2));
    ASSERT_EQ(rows[0].second[0].first, std::string("equality1"));
    ASSERT_EQ(rows[0].second[1].first, std::string("implication"));
    ASSERT_EQ(rows[0].second[0].second.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(rows[0].second[0].second[0].original, std::string("(p[d])"));
}

// addOriginEncoded — string-boundary wrapper: mints the packed key,
// encodes the record, and applies addOriginId's dedup/cap policy.
TEST(memory, add_origin_encoded) {
    gl::Memory m;
    gl::IdOriginMap map;
    const gl::ExpressionWithValidity ev("(=[a,b])", "main");
    std::pair<std::string, std::vector<gl::ExpressionWithValidity>> origin;
    origin.first = "implication";
    origin.second.emplace_back("(in[b,1])", "main");

    gl::addOriginEncoded(map, m.originInterner, ev, origin, 2);
    gl::addOriginEncoded(map, m.originInterner, ev, origin, 2);  // dedup

    int64_t pk = 0;
    ASSERT_TRUE(gl::lookupOriginKey(m.originInterner, ev.original,
                                    ev.validityName, pk));
    ASSERT_EQ(map.at(pk).size(), static_cast<std::size_t>(1));
    const auto dec = gl::decodeOrigin(map.at(pk)[0], m.originInterner);
    ASSERT_EQ(dec.first, std::string("implication"));
    ASSERT_EQ(dec.second.size(), static_cast<std::size_t>(1));
    ASSERT_TRUE(dec.second[0] == origin.second[0]);
}

// ---------- L3 origin-record span-antecedent doors ----------

// encodeOriginSpans — the L3 span-antecedent encoder: byte-identical IdOrigin
// (tag + packed deps in positional order) to encodeOrigin on the equivalent
// OriginLine, for the depN=0, 1, and 3 cases. Same interner => same ids, so the
// encoded records compare equal field-for-field.
TEST(memory, encode_origin_spans_twin) {
    gl::Memory m;

    // depN = 0 (empty antecedents).
    {
        std::pair<std::string, std::vector<gl::ExpressionWithValidity>> ol;
        ol.first = "recursion";
        const gl::IdOrigin a = gl::encodeOrigin(ol, m.originInterner);
        const gl::IdOrigin b = gl::encodeOriginSpans(gl::OriginTag::recursion,
                                                     nullptr, 0, m.originInterner);
        ASSERT_TRUE(a.first == b.first);
        ASSERT_TRUE(a.second == b.second);
        ASSERT_TRUE(b.second.empty());
    }
    // depN = 1.
    {
        const std::string e0 = "(=[b,a])";
        const std::string v0 = "main";
        std::pair<std::string, std::vector<gl::ExpressionWithValidity>> ol;
        ol.first = "symmetry of equality";
        ol.second.emplace_back(e0, v0);
        const gl::IdOrigin a = gl::encodeOrigin(ol, m.originInterner);
        const gl::OriginDep deps[1] = { { gl::StrSpan(e0), gl::StrSpan(v0) } };
        const gl::IdOrigin b = gl::encodeOriginSpans(
            gl::OriginTag::symmetryOfEquality, deps, 1, m.originInterner);
        ASSERT_TRUE(a.first == b.first);
        ASSERT_TRUE(a.second == b.second);
        ASSERT_EQ(b.second.size(), static_cast<std::size_t>(1));
    }
    // depN = 3 (positional order preserved).
    {
        const std::string e0 = "(in3[a,b,c,plus])";
        const std::string v0 = "main";
        const std::string e1 = "(in[b,1])";
        const std::string v1 = "main_boundary_s1";
        const std::string e2 = "(=[c,d])";
        const std::string v2 = "zz_scope";
        std::pair<std::string, std::vector<gl::ExpressionWithValidity>> ol;
        ol.first = "implication";
        ol.second.emplace_back(e0, v0);
        ol.second.emplace_back(e1, v1);
        ol.second.emplace_back(e2, v2);
        const gl::IdOrigin a = gl::encodeOrigin(ol, m.originInterner);
        const gl::OriginDep deps[3] = {
            { gl::StrSpan(e0), gl::StrSpan(v0) },
            { gl::StrSpan(e1), gl::StrSpan(v1) },
            { gl::StrSpan(e2), gl::StrSpan(v2) } };
        const gl::IdOrigin b = gl::encodeOriginSpans(gl::OriginTag::implication,
                                                     deps, 3, m.originInterner);
        ASSERT_TRUE(a.first == b.first);
        ASSERT_TRUE(a.second == b.second);
        ASSERT_EQ(b.second.size(), static_cast<std::size_t>(3));
    }
}

// addOriginEncoded (heap IdOriginMap) span-antecedent door: byte-identical
// deposit to the EWV/OriginLine overload, INCLUDING the D-49 cap-full preference
// replacement (a foundational record displaces the first equality-convenience
// slot). Two independent Memories; compared via the decoded key-sorted snapshot.
TEST(memory, add_origin_encoded_spans_twin) {
    gl::Memory mE, mS;
    gl::IdOriginMap mapE, mapS;
    const int cap = 2;
    const std::string kO = "(=[a,b])";
    const std::string kV = "main";
    const std::string a1O = "(=[b,a])";
    const std::string a1V = "main";
    const std::string a2O = "(in[b,1])";
    const std::string a2V = "main";

    const auto ewv = [](const std::string& o, const std::string& v) {
        return gl::ExpressionWithValidity(o, v);
    };
    const auto ol = [](const char* tag,
                       std::vector<gl::ExpressionWithValidity> deps) {
        return std::pair<std::string, std::vector<gl::ExpressionWithValidity>>(
            std::string(tag), std::move(deps));
    };

    // EWV/OriginLine path: equality1, implication (fills cap), then a
    // foundational recursion (cap-full: displaces the equality1 slot).
    gl::addOriginEncoded(mapE, mE.originInterner, ewv(kO, kV),
        ol("equality1", { ewv(a1O, a1V) }), cap);
    gl::addOriginEncoded(mapE, mE.originInterner, ewv(kO, kV),
        ol("implication", { ewv(a2O, a2V) }), cap);
    gl::addOriginEncoded(mapE, mE.originInterner, ewv(kO, kV),
        ol("recursion", {}), cap);

    // Span path (same sequence).
    const gl::OriginDep d1[1] = { { gl::StrSpan(a1O), gl::StrSpan(a1V) } };
    const gl::OriginDep d2[1] = { { gl::StrSpan(a2O), gl::StrSpan(a2V) } };
    gl::addOriginEncoded(mapS, mS.originInterner, gl::StrSpan(kO), gl::StrSpan(kV),
        gl::OriginTag::equality1, d1, 1, cap);
    gl::addOriginEncoded(mapS, mS.originInterner, gl::StrSpan(kO), gl::StrSpan(kV),
        gl::OriginTag::implication, d2, 1, cap);
    gl::addOriginEncoded(mapS, mS.originInterner, gl::StrSpan(kO), gl::StrSpan(kV),
        gl::OriginTag::recursion, nullptr, 0, cap);

    const auto rowsE = gl::decodeOriginMapSorted(mapE, mE.originInterner);
    const auto rowsS = gl::decodeOriginMapSorted(mapS, mS.originInterner);
    ASSERT_TRUE(rowsE == rowsS);
    // The cap-full replacement fired: recursion took equality1's slot.
    ASSERT_EQ(rowsS.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(rowsS[0].second.size(), static_cast<std::size_t>(2));
    ASSERT_EQ(rowsS[0].second[0].first, std::string("recursion"));
    ASSERT_EQ(rowsS[0].second[1].first, std::string("implication"));
}

// addOriginEncoded (cold exprOriginMap) span-antecedent door: byte-identical
// deposit to the span-key/OriginLine overload on the cold blob map, over the
// depN=3, dedup, and depN=0 cases. Two independent Memories; compared via the
// cold decodeOriginMapSorted snapshot.
TEST(memory, add_origin_encoded_cold_spans_twin) {
    gl::Memory mE, mS;
    const int cap = 8;
    const std::string kO = "(=[x,y])";
    const std::string kV = "main";
    const std::string k2O = "(p[z])";
    const std::string k2V = "zz_scope";
    const std::string e0 = "(=[x,w])", v0 = "main";
    const std::string e1 = "(=[w,y])", v1 = "main_boundary_s1";
    const std::string e2 = "(in[w,1])", v2 = "main";

    {
        std::pair<std::string, std::vector<gl::ExpressionWithValidity>> ol;
        ol.first = "equality2";
        ol.second.emplace_back(e0, v0);
        ol.second.emplace_back(e1, v1);
        ol.second.emplace_back(e2, v2);
        gl::addOriginEncoded(mE.exprOriginMap, mE.originInterner,
            gl::StrSpan(kO), gl::StrSpan(kV), ol, cap);
        gl::addOriginEncoded(mE.exprOriginMap, mE.originInterner,
            gl::StrSpan(kO), gl::StrSpan(kV), ol, cap);  // dedup
        std::pair<std::string, std::vector<gl::ExpressionWithValidity>> ol2;
        ol2.first = "goal";
        gl::addOriginEncoded(mE.exprOriginMap, mE.originInterner,
            gl::StrSpan(k2O), gl::StrSpan(k2V), ol2, cap);
    }
    {
        const gl::OriginDep deps[3] = {
            { gl::StrSpan(e0), gl::StrSpan(v0) },
            { gl::StrSpan(e1), gl::StrSpan(v1) },
            { gl::StrSpan(e2), gl::StrSpan(v2) } };
        gl::addOriginEncoded(mS.exprOriginMap, mS.originInterner,
            gl::StrSpan(kO), gl::StrSpan(kV),
            gl::OriginTag::equality2, deps, 3, cap);
        gl::addOriginEncoded(mS.exprOriginMap, mS.originInterner,
            gl::StrSpan(kO), gl::StrSpan(kV),
            gl::OriginTag::equality2, deps, 3, cap);  // dedup
        gl::addOriginEncoded(mS.exprOriginMap, mS.originInterner,
            gl::StrSpan(k2O), gl::StrSpan(k2V),
            gl::OriginTag::goal, nullptr, 0, cap);
    }
    const auto rowsE =
        gl::decodeOriginMapSorted(mE.exprOriginMap, mE.originInterner);
    const auto rowsS =
        gl::decodeOriginMapSorted(mS.exprOriginMap, mS.originInterner);
    ASSERT_TRUE(rowsE == rowsS);
    ASSERT_EQ(rowsS.size(), static_cast<std::size_t>(2));
    // (=[x,y],main) sorts before (p[z],zz_scope).
    ASSERT_EQ(rowsS[0].second.size(), static_cast<std::size_t>(1));  // dedup held
    ASSERT_EQ(rowsS[0].second[0].first, std::string("equality2"));
    ASSERT_EQ(rowsS[0].second[0].second.size(), static_cast<std::size_t>(3));
    ASSERT_EQ(rowsS[1].second[0].first, std::string("goal"));
    ASSERT_EQ(rowsS[1].second[0].second.size(), static_cast<std::size_t>(0));
}

// MergeClassAccum::addOriginEnc span-antecedent twin: the accumulator's
// serialize() is byte-identical whether the origin line is deposited via the
// EWV/OriginLine overload or the span/OriginTag overload.
TEST(memory, merge_accum_add_origin_enc_spans_twin) {
    gl::Memory mE, mS;
    gl::LbArena lbE{ &gl::staticMemory() };
    gl::LbArena lbS{ &gl::staticMemory() };
    const int cap = 8;
    const std::string kO = "(=[a,b])", kV = "main";
    const std::string e0 = "(=[a,c])", v0 = "main";
    const std::string e1 = "(=[c,b])", v1 = "main";

    std::pair<std::string, std::vector<gl::ExpressionWithValidity>> ol;
    ol.first = "equality2";
    ol.second.emplace_back(e0, v0);
    ol.second.emplace_back(e1, v1);

    gl::MergeClassAccum accE(&lbE);
    accE.addOriginEnc(mE.originInterner, gl::ExpressionWithValidity(kO, kV),
        ol, cap);

    gl::MergeClassAccum accS(&lbS);
    const gl::OriginDep deps[2] = {
        { gl::StrSpan(e0), gl::StrSpan(v0) },
        { gl::StrSpan(e1), gl::StrSpan(v1) } };
    accS.addOriginEnc(mS.originInterner, gl::StrSpan(kO), gl::StrSpan(kV),
        gl::OriginTag::equality2, deps, 2, cap);

    ASSERT_TRUE(accE.serialize() == accS.serialize());
}

// addInternalMailOrigin span-antecedent (L3) door: keys AND records the deposit
// byte-identically to the OriginLine overload — same originInterner mints, same
// origins run. Compared through makeHeapMail.
TEST(memory, internal_mail_origin_record_span_door_twin) {
    gl::Memory mE, mS;
    const std::string kO = "(=[2,repl_lev_1_0])";
    const std::string kV = "main_boundary_ordis_sig";
    const std::string a0O = "(=[2,3])", a0V = kV;
    const std::string a1O = "(=[3,repl_lev_1_0])", a1V = kV;
    const int cap = 8;

    const gl::OriginLine origin{ "equality1",
        std::vector<gl::ExpressionWithValidity>{
            gl::ExpressionWithValidity(a0O, a0V),
            gl::ExpressionWithValidity(a1O, a1V) } };
    gl::addInternalMailOrigin(mE.sameIterationInternalMail, mE.originInterner,
        gl::StrSpan(kO), gl::StrSpan(kV), origin, cap);

    const gl::OriginDep deps[2] = {
        { gl::StrSpan(a0O), gl::StrSpan(a0V) },
        { gl::StrSpan(a1O), gl::StrSpan(a1V) } };
    gl::addInternalMailOrigin(mS.sameIterationInternalMail, mS.originInterner,
        gl::StrSpan(kO), gl::StrSpan(kV), gl::OriginTag::equality1, deps, 2, cap);

    const gl::Mail hE = gl::makeHeapMail(mE.sameIterationInternalMail,
                                         mE.nameMap, mE.originInterner);
    const gl::Mail hS = gl::makeHeapMail(mS.sameIterationInternalMail,
                                         mS.nameMap, mS.originInterner);
    const gl::ExpressionWithValidity key(kO, kV);
    const auto itE = hE.exprOriginMap.find(key);
    const auto itS = hS.exprOriginMap.find(key);
    ASSERT_TRUE(itE != hE.exprOriginMap.end());
    ASSERT_TRUE(itS != hS.exprOriginMap.end());
    ASSERT_TRUE(itS->second == itE->second);
    ASSERT_EQ(itS->second.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(itS->second[0].first, std::string("equality1"));
    ASSERT_EQ(itS->second[0].second.size(), static_cast<std::size_t>(2));
    ASSERT_EQ(itS->second[0].second[0].original, std::string("(=[2,3])"));
    ASSERT_EQ(itS->second[0].second[1].original,
              std::string("(=[3,repl_lev_1_0])"));
}

// addRoutingMailOrigin span-antecedent (L3) door: keys AND records the mailOut
// deposit byte-identically to the EWV/OriginLine overload. Compared through
// decodeMailOutOrigins.
TEST(memory, routing_mail_origin_record_span_door_twin) {
    gl::Memory mE, mS;
    const std::string kO = "(in3[a,b,c,plus])", kV = "main";
    const std::string a0O = "(in[b,1])", a0V = "main";
    const int cap = 8;

    const gl::OriginLine origin{ "implication",
        std::vector<gl::ExpressionWithValidity>{
            gl::ExpressionWithValidity(a0O, a0V) } };
    gl::addRoutingMailOrigin(mE.mailOut, mE.originInterner,
        gl::ExpressionWithValidity(kO, kV), origin, cap);

    const gl::OriginDep deps[1] = { { gl::StrSpan(a0O), gl::StrSpan(a0V) } };
    gl::addRoutingMailOrigin(mS.mailOut, mS.originInterner,
        gl::StrSpan(kO), gl::StrSpan(kV), gl::OriginTag::implication, deps, 1,
        cap);

    const auto rowsE = gl::decodeMailOutOrigins(mE.mailOut, mE.originInterner);
    const auto rowsS = gl::decodeMailOutOrigins(mS.mailOut, mS.originInterner);
    ASSERT_TRUE(rowsE == rowsS);
    ASSERT_EQ(rowsS.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(rowsS[0].first.original, std::string("(in3[a,b,c,plus])"));
    ASSERT_EQ(rowsS[0].second.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(rowsS[0].second[0].first, std::string("implication"));
    ASSERT_EQ(rowsS[0].second[0].second.size(), static_cast<std::size_t>(1));
    ASSERT_EQ(rowsS[0].second[0].second[0].original, std::string("(in[b,1])"));
}

// ---------- Rule-registry id infrastructure ----------

// RuleJustification tables — round trip over the whole closed vocabulary.
// `none` maps to the empty string (the historical marker-LMV justification).
TEST(memory, rule_justification_round_trip) {
    for (std::size_t i = 0;
         i < static_cast<std::size_t>(gl::RuleJustification::COUNT); ++i) {
        const gl::RuleJustification j = static_cast<gl::RuleJustification>(i);
        const char* name = gl::ruleJustificationName(j);
        ASSERT_TRUE(name != nullptr);
        ASSERT_TRUE(gl::ruleJustificationFromString(std::string(name)) == j);
    }
    ASSERT_TRUE(gl::ruleJustificationFromString("")
                == gl::RuleJustification::none);
    ASSERT_TRUE(gl::ruleJustificationFromString("implication")
                == gl::RuleJustification::implication);
    ASSERT_TRUE(gl::ruleJustificationFromString("integration")
                == gl::RuleJustification::integration);
}

// Memory::ruleInterner — dedicated space, disjoint from the NameMap and
// the other interners; encode/lookup/decode round trip.
TEST(memory, rule_interner_basics) {
    gl::Memory m;
    const int32_t nameMapBefore = m.nameMap.nameCount();
    const int32_t valueBefore = m.valueInterner.internedCount();
    const int32_t originBefore = m.originInterner.internedCount();

    ASSERT_EQ(m.ruleInterner.lookup("(in3[u_1,u_2,u_3,plus])"), 0);
    const int32_t idA = m.ruleInterner.encode("(in3[u_1,u_2,u_3,plus])");
    const int32_t idB = m.ruleInterner.encode("(in[u_2,1])");
    ASSERT_TRUE(idA > 0 && idB > 0);
    ASSERT_NE(idA, idB);
    ASSERT_EQ(m.ruleInterner.encode("(in3[u_1,u_2,u_3,plus])"), idA);
    ASSERT_EQ(m.ruleInterner.lookup("(in[u_2,1])"), idB);
    ASSERT_EQ(m.ruleInterner.decode(idA), std::string("(in3[u_1,u_2,u_3,plus])"));

    // The other id spaces are untouched.
    ASSERT_EQ(m.nameMap.nameCount(), nameMapBefore);
    ASSERT_EQ(m.valueInterner.internedCount(), valueBefore);
    ASSERT_EQ(m.originInterner.internedCount(), originBefore);
}

// ---------- LB-state id infrastructure ----------

// Memory::lbStateInterner — dedicated space, disjoint from the NameMap
// and the other interners; encode/lookup/decode round trip.
TEST(memory, lb_state_interner_basics) {
    gl::Memory m;
    const int32_t nameMapBefore = m.nameMap.nameCount();
    const int32_t ruleBefore = m.ruleInterner.internedCount();

    ASSERT_EQ(m.lbStateInterner.lookup("(or3[1,2,3])"), 0);
    const int32_t idA = m.lbStateInterner.encode("(or3[1,2,3])");
    const int32_t idB = m.lbStateInterner.encode("(in[7,1])");
    ASSERT_TRUE(idA > 0 && idB > 0);
    ASSERT_NE(idA, idB);
    ASSERT_EQ(m.lbStateInterner.encode("(or3[1,2,3])"), idA);
    ASSERT_EQ(m.lbStateInterner.lookup("(in[7,1])"), idB);
    ASSERT_EQ(m.lbStateInterner.decode(idA), std::string("(or3[1,2,3])"));

    // The other id spaces are untouched.
    ASSERT_EQ(m.nameMap.nameCount(), nameMapBefore);
    ASSERT_EQ(m.ruleInterner.internedCount(), ruleBefore);
}

// orBookkeeping (ColdSetMap, Batch 2) — insertSorted with the stateful
// DecodedIdLess keeps the disjunct run in decoded-lex order regardless of mint
// order and dedupes; runLen / valueAt read it back in that order (never
// coldIntSetAt, which would re-sort by raw int).
TEST(memory, or_bookkeeping_decoded_order) {
    gl::Memory m;
    const int64_t pk = gl::packLbStateKey(
        m.lbStateInterner.encode("(in2[x,y,z])"),
        m.lbStateInterner.encode("(or3[a,b,c])"));
    const gl::DecodedIdLess cmp{ &m.lbStateInterner };

    // Mint in reverse lex order so id order != decoded order.
    const int32_t dEq = m.lbStateInterner.encode("((=[7,2]))");
    const int32_t dIn = m.lbStateInterner.encode("((in[7,1]))");
    const int32_t id = m.orBookkeeping.insertSorted(pk, dEq, cmp);
    m.orBookkeeping.insertSorted(pk, dIn, cmp);
    m.orBookkeeping.insertSorted(pk, dEq, cmp);   // dedupe

    ASSERT_EQ(m.orBookkeeping.lookup(pk), id);
    ASSERT_EQ(m.orBookkeeping.runLen(id), 2);
    // Decoded-lex order: "((=[7,2]))" < "((in[7,1]))".
    ASSERT_EQ(m.lbStateInterner.decode(m.orBookkeeping.valueAt(id, 0)),
              std::string("((=[7,2]))"));
    ASSERT_EQ(m.lbStateInterner.decode(m.orBookkeeping.valueAt(id, 1)),
              std::string("((in[7,1]))"));
}

// eqClassSttmntIndexMapMap (flattened to a byte-key map, Batch 2) — the
// encode/lookup/upsert/erase helpers round-trip a (validity, memberIds) ->
// waterline cache: an absent read is 0, upsert sets-or-inserts, erase removes,
// and the byte key is bijective over (validity, members).
TEST(memory, eq_class_index_flatten_round_trip) {
    gl::Memory m;
    const int16_t vA = m.nameMap.encode("main");
    const int16_t vB = m.nameMap.encode("main_boundary_x");
    const std::vector<int16_t> mem1{ 3, 7, 9 };
    const std::vector<int16_t> mem2{ 3, 7 };

    // Absent -> 0 (the heap operator[] default).
    ASSERT_EQ(gl::lookupEqClassIndex(m.eqClassSttmntIndexMapMap, vA, mem1), 0);

    // Insert distinct (validity, members) entries; read each back.
    gl::upsertEqClassIndex(m.eqClassSttmntIndexMapMap, vA, mem1, 5);
    gl::upsertEqClassIndex(m.eqClassSttmntIndexMapMap, vA, mem2, 8);
    gl::upsertEqClassIndex(m.eqClassSttmntIndexMapMap, vB, mem1, 11);
    ASSERT_EQ(gl::lookupEqClassIndex(m.eqClassSttmntIndexMapMap, vA, mem1), 5);
    ASSERT_EQ(gl::lookupEqClassIndex(m.eqClassSttmntIndexMapMap, vA, mem2), 8);
    ASSERT_EQ(gl::lookupEqClassIndex(m.eqClassSttmntIndexMapMap, vB, mem1), 11);

    // Update in place (set-or-insert on an existing key).
    gl::upsertEqClassIndex(m.eqClassSttmntIndexMapMap, vA, mem1, 42);
    ASSERT_EQ(gl::lookupEqClassIndex(m.eqClassSttmntIndexMapMap, vA, mem1), 42);

    // Erase one entry; the others survive; the erased re-reads as 0.
    gl::eraseEqClassIndex(m.eqClassSttmntIndexMapMap, vA, mem1);
    ASSERT_EQ(gl::lookupEqClassIndex(m.eqClassSttmntIndexMapMap, vA, mem1), 0);
    ASSERT_EQ(gl::lookupEqClassIndex(m.eqClassSttmntIndexMapMap, vA, mem2), 8);
    ASSERT_EQ(gl::lookupEqClassIndex(m.eqClassSttmntIndexMapMap, vB, mem1), 11);

    // The byte key is bijective: differing members or validity differ.
    ASSERT_FALSE(gl::encodeEqClassKey(vA, mem1) == gl::encodeEqClassKey(vA, mem2));
    ASSERT_FALSE(gl::encodeEqClassKey(vA, mem1) == gl::encodeEqClassKey(vB, mem1));
}

// ---- EquivalenceClass codec (the ColdBlobMap record serializer) --------

namespace {
    // A class with all three fields populated; `reverseOriginInsert` builds the
    // equalityOriginMap (an unordered_map) in the opposite key-insertion order so
    // the canonical-bytes property can be checked against insertion order.
    gl::EquivalenceClass makeSampleClass(bool reverseOriginInsert) {
        gl::EquivalenceClass cls;
        cls.memberIds = { -5, 3, 7, 12, 100 };
        cls.intEqualityLevelsMap[gl::packEqPairKey(3, 7)] = { 0, 2, 5 };
        cls.intEqualityLevelsMap[gl::packEqPairKey(7, 12)] = { 1 };
        const gl::IdOrigin l1{ gl::OriginTag::anchorHandling, { 11, 22, 33 } };
        const gl::IdOrigin l2{ gl::OriginTag::broadcast, {} };
        const gl::IdOrigin l3{ gl::OriginTag::broadcast, { 44 } };
        const int64_t k1 = 1000, k2 = 999999;
        if (!reverseOriginInsert) {
            cls.equalityOriginMap[k1] = { l1, l2 };
            cls.equalityOriginMap[k2] = { l3 };
        } else {
            cls.equalityOriginMap[k2] = { l3 };
            cls.equalityOriginMap[k1] = { l1, l2 };
        }
        return cls;
    }
}

// Round-trip: deserialize(serialize(cls)) reproduces every field, and
// re-serializing the decoded class reproduces the blob (a fixed point).
TEST(memory, eqclass_codec_round_trip_lossless) {
    const gl::EquivalenceClass a = makeSampleClass(false);
    const std::vector<char> blob = gl::serializeEquivalenceClass(a);
    const gl::EquivalenceClass b = gl::deserializeEquivalenceClass(
        blob.data(), static_cast<int32_t>(blob.size()));
    ASSERT_TRUE(b.memberIds == a.memberIds);
    ASSERT_TRUE(b.intEqualityLevelsMap == a.intEqualityLevelsMap);
    ASSERT_TRUE(b.equalityOriginMap == a.equalityOriginMap);
    const std::vector<char> blob2 = gl::serializeEquivalenceClass(b);
    ASSERT_TRUE(blob2 == blob);
    // The record codec the typed blob map calls forwards to the same functions.
    ASSERT_TRUE(gl::Codec<gl::EquivalenceClass>::serialize(a) == blob);
    const gl::EquivalenceClass c = gl::Codec<gl::EquivalenceClass>::deserialize(
        blob.data(), static_cast<int32_t>(blob.size()));
    ASSERT_TRUE(c.memberIds == a.memberIds);
}

// Determinism: two logically-equal classes whose equalityOriginMap was built in
// opposite key-insertion order serialize to byte-identical blobs (the codec
// sorts the unordered_map's keys before emit).
TEST(memory, eqclass_codec_canonical_under_permuted_insert) {
    const gl::EquivalenceClass a = makeSampleClass(false);
    const gl::EquivalenceClass b = makeSampleClass(true);
    ASSERT_TRUE(a.equalityOriginMap == b.equalityOriginMap);   // same content
    const std::vector<char> ba = gl::serializeEquivalenceClass(a);
    const std::vector<char> bb = gl::serializeEquivalenceClass(b);
    ASSERT_TRUE(ba == bb);
}

// The empty class round-trips and serializes to exactly three int32 zero counts.
TEST(memory, eqclass_codec_empty_round_trip) {
    const gl::EquivalenceClass empty;
    const std::vector<char> blob = gl::serializeEquivalenceClass(empty);
    ASSERT_EQ(blob.size(), static_cast<std::size_t>(12));
    const gl::EquivalenceClass b = gl::deserializeEquivalenceClass(
        blob.data(), static_cast<int32_t>(blob.size()));
    ASSERT_EQ(b.memberIds.size(), static_cast<std::size_t>(0));
    ASSERT_EQ(b.intEqualityLevelsMap.size(), static_cast<std::size_t>(0));
    ASSERT_EQ(b.equalityOriginMap.size(), static_cast<std::size_t>(0));
}

// serializeEquivalenceClassInto — the 0%-heap arena-fill twin of the heap codec
// serializeEquivalenceClass (the retained Rule-18 oracle). Byte-identical for
// every class shape: empty, members-only, members+levels, and full origin
// history with origin keys built OUT of ascending order (exercises the arena
// `ok` sort) plus multi-line / multi-dep records. The I-103 deload byte layout
// must not move by one byte.
TEST(memory, serialize_equivalence_class_into_matches_heap) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::ScratchArena a; a.bind(&g);

    auto check = [&](const gl::EquivalenceClass& cls) {
        gl::ScratchScope scope(a);
        const std::vector<char> ref = gl::serializeEquivalenceClass(cls);
        const gl::StrSpan got = gl::serializeEquivalenceClassInto(a, cls);
        ASSERT_EQ(got.len, static_cast<int32_t>(ref.size()));
        ASSERT_TRUE(std::memcmp(got.ptr, ref.data(), ref.size()) == 0);
    };

    check(gl::EquivalenceClass{});                 // empty (three int32 zero counts)
    {
        gl::EquivalenceClass m;
        m.memberIds = { -5, 3, 7, 12, 100 };
        check(m);                                  // members only
    }
    {
        gl::EquivalenceClass ml;
        ml.memberIds = { 1, 2 };
        ml.intEqualityLevelsMap[gl::packEqPairKey(3, 7)] = { 0, 2, 5 };
        ml.intEqualityLevelsMap[gl::packEqPairKey(7, 12)] = { 1 };
        check(ml);                                 // members + levels, no origins
    }
    check(makeSampleClass(false));                 // full, ascending-insert origins
    check(makeSampleClass(true));                  // full, reverse-insert -> ok sort
}

// EquivalenceClassView — the zero-copy read view reproduces every field of the
// class its blob was serialized from, so it is a faithful heap-free replacement
// for `deserializeEquivalenceClass` at the equi-class transient read sites.
TEST(memory, eqclass_view_reads_every_field) {
    const gl::EquivalenceClass a = makeSampleClass(false);
    const std::vector<char> blob = gl::serializeEquivalenceClass(a);
    const gl::EquivalenceClassView v{ blob.data(),
                                      static_cast<int32_t>(blob.size()) };

    // Members.
    ASSERT_EQ(v.memberCount(), static_cast<int32_t>(a.memberIds.size()));
    for (int32_t i = 0; i < v.memberCount(); ++i)
        ASSERT_EQ(v.memberId(i), a.memberIds[static_cast<std::size_t>(i)]);

    // Per-pair levels — rebuild through the view and compare to the source map.
    std::map<uint32_t, std::set<int> > levels;
    v.forEachLevelKey([&](uint32_t key, const char* lp, int32_t m) {
        std::set<int>& s = levels[key];
        for (int32_t j = 0; j < m; ++j)
            s.insert(static_cast<int>(
                gl::EquivalenceClassView::rdI32(lp + 4 * j)));
    });
    ASSERT_TRUE(levels == a.intEqualityLevelsMap);

    // Origins — rebuild through the view and compare to the source IdOriginMap.
    gl::IdOriginMap origins;
    v.forEachOriginLine([&](int64_t key, gl::OriginTag tag,
                            const char* dp, int32_t d) {
        std::vector<int64_t> deps;
        for (int32_t j = 0; j < d; ++j)
            deps.push_back(gl::EquivalenceClassView::rdI64(dp + 8 * j));
        origins[key].emplace_back(tag, std::move(deps));
    });
    ASSERT_TRUE(origins == a.equalityOriginMap);
}

// The empty class's view reports zero members / levels / origins and lands the
// origins section at the fixed 8-byte offset (two int32 zero counts).
TEST(memory, eqclass_view_empty) {
    const gl::EquivalenceClass empty;
    const std::vector<char> blob = gl::serializeEquivalenceClass(empty);
    const gl::EquivalenceClassView v{ blob.data(),
                                      static_cast<int32_t>(blob.size()) };
    ASSERT_EQ(v.memberCount(), 0);
    ASSERT_EQ(v.levelsOffset(), 4);
    ASSERT_EQ(v.originsOffset(), 8);
    int32_t lines = 0;
    v.forEachOriginLine([&](int64_t, gl::OriginTag, const char*, int32_t) {
        ++lines;
    });
    ASSERT_EQ(lines, 0);
}

// MergeClassAccum — the arena accumulator's serialize() is byte-identical to
// serializeEquivalenceClass of the equivalent heap class (the linchpin of the
// merge-statification seam: producer & consumer decouple across these bytes).
TEST(memory, merge_accum_serialize_matches_heap) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::MergeClassAccum accum(&lb);
    // Same content as makeSampleClass: members {-5,3,7,12,100}, two level keys,
    // origins {1000:[l1,l2], 999999:[l3]}. Insertion order is deliberately NOT
    // the canonical order — serialize() imposes ascending keys.
    accum.addMember(-5); accum.addMember(3); accum.addMember(7);
    accum.addMember(12); accum.addMember(100);
    const int32_t lv37[] = { 0, 2, 5 };
    const int32_t lv712[] = { 1 };
    accum.setLevel(gl::packEqPairKey(7, 12), lv712, 1);   // out of key order
    accum.setLevel(gl::packEqPairKey(3, 7), lv37, 3);
    const int64_t d1[] = { 11, 22, 33 };
    const int64_t d3[] = { 44 };
    accum.addOriginLine(999999, static_cast<uint8_t>(gl::OriginTag::broadcast), d3, 1);
    accum.addOriginLine(1000, static_cast<uint8_t>(gl::OriginTag::anchorHandling), d1, 3);
    accum.addOriginLine(1000, static_cast<uint8_t>(gl::OriginTag::broadcast), nullptr, 0);

    const std::vector<char> got = accum.serialize();
    const std::vector<char> want =
        gl::serializeEquivalenceClass(makeSampleClass(false));
    ASSERT_TRUE(got == want);
}

// addOriginId cap-preference (D-49): at cap 1, a non-equality tag replaces the
// existing equality-convenience line — byte-matches the heap addOriginId twin.
TEST(memory, merge_accum_add_origin_id_cap_preference) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::MergeClassAccum accum(&lb);
    const int64_t d[] = { 7 };
    accum.addOriginId(500, static_cast<uint8_t>(gl::OriginTag::equality1), d, 1, 1);
    accum.addOriginId(500, static_cast<uint8_t>(gl::OriginTag::anchorHandling), d, 1, 1);

    gl::EquivalenceClass ref;
    gl::addOriginId(ref.equalityOriginMap, 500,
        gl::IdOrigin{ gl::OriginTag::equality1, std::vector<int64_t>{ 7 } }, 1);
    gl::addOriginId(ref.equalityOriginMap, 500,
        gl::IdOrigin{ gl::OriginTag::anchorHandling, std::vector<int64_t>{ 7 } }, 1);
    ASSERT_TRUE(accum.serialize() == gl::serializeEquivalenceClass(ref));
}

// unionMembersByName keeps decoded-lex order — the in-place twin of
// unionMemberIdsByName.
TEST(memory, merge_accum_union_members_decoded_lex) {
    gl::Memory mb;                                   // pool-backed NameMap
    gl::LbArena lb{ &gl::staticMemory() };
    const int16_t a = mb.nameMap.encode("a");
    const int16_t b = mb.nameMap.encode("b");
    const int16_t c = mb.nameMap.encode("c");
    const int16_t d = mb.nameMap.encode("d");
    gl::MergeClassAccum accum(&lb);
    accum.addMember(b); accum.addMember(d);          // seed {b, d} (decoded-lex)
    const int16_t other[] = { a, c, d };             // {a, c, d} (decoded-lex)
    accum.unionMembersByName(other, 3, mb.nameMap);
    ASSERT_EQ(accum.members.size(), 4);
    ASSERT_EQ(accum.members[0], a);
    ASSERT_EQ(accum.members[1], b);
    ASSERT_EQ(accum.members[2], c);
    ASSERT_EQ(accum.members[3], d);
}

// B-base origin merge — mergeOriginsBBase reproduces the heap
// `tmp = classB.origins; overwriteOriginsId(tmp, classA, cap)` byte-for-byte:
// B's lines first (verbatim), then A's appended per key below cap and deduped
// (no D-49). Covers a shared key (dedup), an A-only key, and a B-only key.
TEST(memory, merge_accum_origins_bbase_matches_heap) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);

    const int cap = 8;
    gl::EquivalenceClass A;
    A.equalityOriginMap[100] = {
        gl::IdOrigin{ gl::OriginTag::equality1, { 1, 2 } },
        gl::IdOrigin{ gl::OriginTag::broadcast,  { 5 } } };   // dup of a B line at 100
    A.equalityOriginMap[200] = {
        gl::IdOrigin{ gl::OriginTag::anchorHandling, { 4 } } };
    gl::EquivalenceClass B;
    B.equalityOriginMap[100] = {
        gl::IdOrigin{ gl::OriginTag::broadcast,   { 5 } },
        gl::IdOrigin{ gl::OriginTag::implication, { 6 } } };
    B.equalityOriginMap[300] = {
        gl::IdOrigin{ gl::OriginTag::equality2, { 7, 8 } } };

    // Heap reference: tmp = B; overwriteOriginsId(tmp, A, cap).
    gl::IdOriginMap heapRes = B.equalityOriginMap;
    gl::overwriteOriginsId(heapRes, A.equalityOriginMap, cap);
    gl::EquivalenceClass heapClass;
    heapClass.equalityOriginMap = heapRes;
    const std::vector<char> want = gl::serializeEquivalenceClass(heapClass);

    // Accum: seed A's origins, then merge B-base via the classB view.
    gl::MergeClassAccum accum(&lb);
    for (const auto& kv : A.equalityOriginMap)
        for (const gl::IdOrigin& ln : kv.second)
            accum.addOriginLine(kv.first, static_cast<uint8_t>(ln.first),
                                ln.second.data(),
                                static_cast<int32_t>(ln.second.size()));
    const std::vector<char> bBlob = gl::serializeEquivalenceClass(B);
    const gl::EquivalenceClassView bView{ bBlob.data(),
                                          static_cast<int32_t>(bBlob.size()) };
    accum.mergeOriginsBBase(bView, cap);

    ASSERT_TRUE(accum.serialize() == want);
}

// Level merge B-base (mergeLevelsBBase) reproduces `newLevels = B;
// overwriteLevels(newLevels, A)` byte-for-byte (A-wins REPLACE): a shared key
// (A wins), an A-only key (kept), a B-only key (added).
TEST(memory, merge_accum_levels_bbase_matches_heap) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    const uint32_t kShared = gl::packEqPairKey(3, 7);
    const uint32_t kAonly  = gl::packEqPairKey(3, 9);
    const uint32_t kBonly  = gl::packEqPairKey(5, 8);
    gl::EquivalenceClass A, B;
    A.intEqualityLevelsMap[kShared] = { 1, 4 };   // A wins the shared key
    A.intEqualityLevelsMap[kAonly]  = { 2 };
    B.intEqualityLevelsMap[kShared] = { 9 };      // loses to A
    B.intEqualityLevelsMap[kBonly]  = { 0, 6 };

    // Heap reference: newLevels = B; overwriteLevels(newLevels, A) (A replaces).
    std::map<uint32_t, std::set<int> > heapLv = B.intEqualityLevelsMap;
    for (const auto& kv : A.intEqualityLevelsMap) heapLv[kv.first] = kv.second;
    gl::EquivalenceClass heapClass; heapClass.intEqualityLevelsMap = heapLv;
    const std::vector<char> want = gl::serializeEquivalenceClass(heapClass);

    gl::MergeClassAccum accum(&lb);
    for (const auto& kv : A.intEqualityLevelsMap) {
        const std::vector<int32_t> asc(kv.second.begin(), kv.second.end());
        accum.setLevel(kv.first, asc.data(), static_cast<int32_t>(asc.size()));
    }
    const std::vector<char> bBlob = gl::serializeEquivalenceClass(B);
    const gl::EquivalenceClassView bView{ bBlob.data(),
                                          static_cast<int32_t>(bBlob.size()) };
    accum.mergeLevelsBBase(bView);
    ASSERT_TRUE(accum.serialize() == want);
}

// Level merge from another accumulator (mergeLevelsFromAccum) reproduces
// `tmp = other; overwriteLevels(tmp, A)` (A-wins REPLACE) — the cross-product
// `mergedMap | A` step.
TEST(memory, merge_accum_levels_from_accum_matches_heap) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    const uint32_t kShared = gl::packEqPairKey(3, 7);
    const uint32_t kAonly  = gl::packEqPairKey(3, 9);
    const uint32_t kOonly  = gl::packEqPairKey(5, 8);
    gl::EquivalenceClass A, O;
    A.intEqualityLevelsMap[kShared] = { 1, 4 };
    A.intEqualityLevelsMap[kAonly]  = { 2 };
    O.intEqualityLevelsMap[kShared] = { 9 };
    O.intEqualityLevelsMap[kOonly]  = { 0, 6 };

    std::map<uint32_t, std::set<int> > heapLv = O.intEqualityLevelsMap;
    for (const auto& kv : A.intEqualityLevelsMap) heapLv[kv.first] = kv.second;
    gl::EquivalenceClass heapClass; heapClass.intEqualityLevelsMap = heapLv;
    const std::vector<char> want = gl::serializeEquivalenceClass(heapClass);

    gl::MergeClassAccum accum(&lb), other(&lb);
    for (const auto& kv : A.intEqualityLevelsMap) {
        const std::vector<int32_t> asc(kv.second.begin(), kv.second.end());
        accum.setLevel(kv.first, asc.data(), static_cast<int32_t>(asc.size()));
    }
    for (const auto& kv : O.intEqualityLevelsMap) {
        const std::vector<int32_t> asc(kv.second.begin(), kv.second.end());
        other.setLevel(kv.first, asc.data(), static_cast<int32_t>(asc.size()));
    }
    accum.mergeLevelsFromAccum(other);
    ASSERT_TRUE(accum.serialize() == want);
}

// Origin merge from another accumulator (mergeOriginsFromAccum) reproduces
// `tmp = other; overwriteOriginsId(tmp, A, cap)` — the cross-product
// `mergedOriginMap | A` step (other's lines first, then A below cap, deduped).
TEST(memory, merge_accum_origins_from_accum_matches_heap) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    const int cap = 8;
    gl::EquivalenceClass A, O;
    A.equalityOriginMap[100] = {
        gl::IdOrigin{ gl::OriginTag::equality1, { 1, 2 } },
        gl::IdOrigin{ gl::OriginTag::broadcast,  { 5 } } };   // dup of an O line
    A.equalityOriginMap[200] = {
        gl::IdOrigin{ gl::OriginTag::anchorHandling, { 4 } } };
    O.equalityOriginMap[100] = {
        gl::IdOrigin{ gl::OriginTag::broadcast,   { 5 } },
        gl::IdOrigin{ gl::OriginTag::implication, { 6 } } };
    O.equalityOriginMap[300] = {
        gl::IdOrigin{ gl::OriginTag::equality2, { 7, 8 } } };

    gl::IdOriginMap heapRes = O.equalityOriginMap;
    gl::overwriteOriginsId(heapRes, A.equalityOriginMap, cap);
    gl::EquivalenceClass heapClass; heapClass.equalityOriginMap = heapRes;
    const std::vector<char> want = gl::serializeEquivalenceClass(heapClass);

    gl::MergeClassAccum accum(&lb), other(&lb);
    for (const auto& kv : A.equalityOriginMap)
        for (const gl::IdOrigin& ln : kv.second)
            accum.addOriginLine(kv.first, static_cast<uint8_t>(ln.first),
                                ln.second.data(),
                                static_cast<int32_t>(ln.second.size()));
    for (const auto& kv : O.equalityOriginMap)
        for (const gl::IdOrigin& ln : kv.second)
            other.addOriginLine(kv.first, static_cast<uint8_t>(ln.first),
                                ln.second.data(),
                                static_cast<int32_t>(ln.second.size()));
    accum.mergeOriginsFromAccum(other, cap);
    ASSERT_TRUE(accum.serialize() == want);
}

// unionMembersFromView — the classB-view twin of unionMembersByName keeps
// decoded-lex order.
TEST(memory, merge_accum_union_members_from_view) {
    gl::Memory mb;
    gl::LbArena lb{ &gl::staticMemory() };
    const int16_t a = mb.nameMap.encode("a");
    const int16_t b = mb.nameMap.encode("b");
    const int16_t c = mb.nameMap.encode("c");
    const int16_t d = mb.nameMap.encode("d");
    gl::EquivalenceClass B;
    B.memberIds = { a, c, d };                       // {a, c, d} decoded-lex
    const std::vector<char> bBlob = gl::serializeEquivalenceClass(B);
    const gl::EquivalenceClassView bView{ bBlob.data(),
                                          static_cast<int32_t>(bBlob.size()) };
    gl::MergeClassAccum accum(&lb);
    accum.addMember(b); accum.addMember(d);          // seed {b, d}
    accum.unionMembersFromView(bView, mb.nameMap);
    ASSERT_EQ(accum.members.size(), 4);
    ASSERT_EQ(accum.members[0], a);
    ASSERT_EQ(accum.members[1], b);
    ASSERT_EQ(accum.members[2], c);
    ASSERT_EQ(accum.members[3], d);
}

// forEachOriginSorted iterates distinct keys ascending, per-key lines in
// insertion order — matches the heap `equalityOriginMap` std::map walk.
TEST(memory, merge_accum_for_each_origin_sorted) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::MergeClassAccum accum(&lb);
    const int64_t d12[] = { 1, 2 };
    const int64_t d3[]  = { 3 };
    accum.addOriginLine(300, static_cast<uint8_t>(gl::OriginTag::equality2), d3, 1);
    accum.addOriginLine(100, static_cast<uint8_t>(gl::OriginTag::equality1), d12, 2);
    accum.addOriginLine(100, static_cast<uint8_t>(gl::OriginTag::broadcast), nullptr, 0);
    accum.addOriginLine(200, static_cast<uint8_t>(gl::OriginTag::anchorHandling), d3, 1);

    gl::EquivalenceClass ref;
    ref.equalityOriginMap[300] = { gl::IdOrigin{ gl::OriginTag::equality2, { 3 } } };
    ref.equalityOriginMap[100] = {
        gl::IdOrigin{ gl::OriginTag::equality1, { 1, 2 } },
        gl::IdOrigin{ gl::OriginTag::broadcast, {} } };
    ref.equalityOriginMap[200] = { gl::IdOrigin{ gl::OriginTag::anchorHandling, { 3 } } };

    std::vector<std::pair<int64_t, std::vector<gl::IdOrigin> > > got;
    accum.forEachOriginSorted([&](int64_t k, auto lines) {
        std::vector<gl::IdOrigin> v;
        for (int32_t t = 0; t < lines.size(); ++t) {
            std::vector<int64_t> deps;
            for (int32_t d = 0; d < lines.depCount(t); ++d)
                deps.push_back(lines.dep(t, d));
            v.emplace_back(lines.tag(t), std::move(deps));
        }
        got.emplace_back(k, std::move(v));
    });
    // IdOriginMap is an unordered_map, so compare `got` (ascending key) against
    // ref's SORTED keys, per-key lines equal.
    std::vector<int64_t> refKeys;
    for (const auto& kv : ref.equalityOriginMap) refKeys.push_back(kv.first);
    std::sort(refKeys.begin(), refKeys.end());
    ASSERT_EQ(got.size(), refKeys.size());
    for (std::size_t j = 0; j < refKeys.size(); ++j) {
        ASSERT_TRUE(got[j].first == refKeys[j]);
        ASSERT_TRUE(got[j].second == ref.equalityOriginMap.at(refKeys[j]));
    }
}

// ChangedClassesBuffer push-from-bytes stores byte-identical content to the
// EquivalenceClass overload (both are the canonical serialize stream).
TEST(memory, changed_classes_buffer_push_from_bytes) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d1 = gl::DirtyState::Clean, d2 = gl::DirtyState::Clean;
    gl::ChangedClassesBuffer buf1(&lb, &d1), buf2(&lb, &d2);
    const gl::EquivalenceClass cls = makeSampleClass(false);
    buf1.push(static_cast<int16_t>(7), cls);
    const std::vector<char> blob = gl::serializeEquivalenceClass(cls);
    buf2.push(static_cast<int16_t>(7), blob.data(),
              static_cast<int32_t>(blob.size()));
    ASSERT_EQ(buf1.size(), 1);
    ASSERT_EQ(buf2.size(), 1);
    ASSERT_EQ(buf1.validityAt(0), static_cast<int16_t>(7));
    ASSERT_EQ(buf2.validityAt(0), static_cast<int16_t>(7));
    const gl::EquivalenceClass c1 = buf1.classAt(0);
    const gl::EquivalenceClass c2 = buf2.classAt(0);
    ASSERT_TRUE(c1.memberIds == c2.memberIds);
    ASSERT_TRUE(c1.intEqualityLevelsMap == c2.intEqualityLevelsMap);
    ASSERT_TRUE(c1.equalityOriginMap == c2.equalityOriginMap);
}

// upsertEqClassIndex from a MergeClassAccum builds the same byte key as the
// std::vector overload — the accum's members ARE the key.
TEST(memory, upsert_eq_class_index_from_accum) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState dd = gl::DirtyState::Clean;
    gl::TypedColdMap<gl::EqClassKey, int> idx(&lb, &dd);
    gl::MergeClassAccum accum(&lb);
    accum.addMember(3); accum.addMember(7); accum.addMember(12);
    gl::upsertEqClassIndex(idx, static_cast<int16_t>(5), accum, 42);
    ASSERT_EQ(gl::lookupEqClassIndex(idx, static_cast<int16_t>(5),
                                     std::vector<int16_t>{ 3, 7, 12 }), 42);
    gl::upsertEqClassIndex(idx, static_cast<int16_t>(5), accum, 99);   // in place
    ASSERT_EQ(gl::lookupEqClassIndex(idx, static_cast<int16_t>(5),
                                     std::vector<int16_t>{ 3, 7, 12 }), 99);
}

// upsert/lookupEqClassIndex from an EquivalenceClassView build the same byte key
// as the std::vector overload (the view's members ARE the key) — the equi-7f
// Pass-2 index path is byte-identical to the former heap-class-decode path.
TEST(memory, eq_class_index_from_view) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState dd = gl::DirtyState::Clean;
    gl::TypedColdMap<gl::EqClassKey, int> idx(&lb, &dd);

    gl::EquivalenceClass cls;
    cls.memberIds = { 3, 7, 12 };
    const std::vector<char> blob = gl::serializeEquivalenceClass(cls);
    const gl::EquivalenceClassView view{ blob.data(),
                                         static_cast<int32_t>(blob.size()) };

    gl::upsertEqClassIndex(idx, static_cast<int16_t>(5), view, 42);
    ASSERT_EQ(gl::lookupEqClassIndex(idx, static_cast<int16_t>(5), view), 42);
    // Same key the std::vector overload builds.
    ASSERT_EQ(gl::lookupEqClassIndex(idx, static_cast<int16_t>(5),
                                     std::vector<int16_t>{ 3, 7, 12 }), 42);
    gl::upsertEqClassIndex(idx, static_cast<int16_t>(5), view, 99);   // in place
    ASSERT_EQ(gl::lookupEqClassIndex(idx, static_cast<int16_t>(5), view), 99);

    // A miss (different members) reads 0, like the base overload's findOr default.
    gl::EquivalenceClass other;
    other.memberIds = { 1, 2 };
    const std::vector<char> ob = gl::serializeEquivalenceClass(other);
    const gl::EquivalenceClassView ov{ ob.data(),
                                       static_cast<int32_t>(ob.size()) };
    ASSERT_EQ(gl::lookupEqClassIndex(idx, static_cast<int16_t>(5), ov), 0);
}

// addOriginEnc — the accumulator twin of the free addOriginEncoded(IdOriginMap):
// mint key + encode record + addOriginId, byte-identical.
TEST(memory, merge_accum_add_origin_enc_matches_heap) {
    gl::Memory m;
    gl::LbArena lb{ &gl::staticMemory() };
    const gl::ExpressionWithValidity ev("(=[a,b])", "main");
    std::pair<std::string, std::vector<gl::ExpressionWithValidity> > origin;
    origin.first = "equality2";
    origin.second.push_back(gl::ExpressionWithValidity("(=[a,c])", "main"));
    origin.second.push_back(gl::ExpressionWithValidity("(=[c,b])", "main"));

    gl::EquivalenceClass ref;
    gl::addOriginEncoded(ref.equalityOriginMap, m.originInterner, ev, origin, 8);

    gl::MergeClassAccum accum(&lb);
    accum.addOriginEnc(m.originInterner, ev, origin, 8);
    ASSERT_TRUE(accum.serialize() == gl::serializeEquivalenceClass(ref));
}

// merge_full — the rewired mergeTwoEquivalenceClasses (accumulator + classB view)
// produces a class byte-identical to the heap merge algebra, on the A-subset-B
// branch AND the cross-product branch (trackHistory off isolates the members /
// levels / origins folds; the equality2 emission + its exprOriginMap side effects
// are validated at the full pipeline gate).
TEST(memory, merge_full_serialize_matches_heap) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::Memory m;
    ea.parameters.trackHistory = false;
    const int cap = ea.parameters.compressor_mode
        ? ea.parameters.compressor_max_origins_per_expr
        : ea.parameters.max_origin_per_expr;
    gl::LbArena lb{ &gl::staticMemory() };
    const int16_t a = m.nameMap.encode("a"), b = m.nameMap.encode("b"),
                  c = m.nameMap.encode("c"), d = m.nameMap.encode("d");

    // Seed an accumulator from a heap class A's content.
    const auto seedAccum = [&](gl::MergeClassAccum& acc, const gl::EquivalenceClass& A) {
        for (const int16_t id : A.memberIds) acc.addMember(id);
        for (const auto& kv : A.intEqualityLevelsMap) {
            const std::vector<int32_t> lv(kv.second.begin(), kv.second.end());
            acc.setLevel(kv.first, lv.data(), static_cast<int32_t>(lv.size()));
        }
        for (const auto& kv : A.equalityOriginMap)
            for (const gl::IdOrigin& ln : kv.second)
                acc.addOriginLine(kv.first, static_cast<uint8_t>(ln.first),
                                  ln.second.data(),
                                  static_cast<int32_t>(ln.second.size()));
    };
    // overwriteLevels twin (A-wins REPLACE).
    const auto overLevels = [](std::map<uint32_t, std::set<int> >& dst,
                               const std::map<uint32_t, std::set<int> >& src) {
        for (const auto& kv : src) dst[kv.first] = kv.second;
    };

    // ---------- A subset of B ----------
    {
        gl::EquivalenceClass A, B;
        A.memberIds = { a, b };
        A.intEqualityLevelsMap[gl::packEqPairKey(a, b)] = { 0, 3 };
        A.equalityOriginMap[100] = { gl::IdOrigin{ gl::OriginTag::equality1, { 1 } } };
        B.memberIds = { a, b, c, d };
        B.intEqualityLevelsMap[gl::packEqPairKey(a, b)] = { 9 };        // loses to A
        B.intEqualityLevelsMap[gl::packEqPairKey(c, d)] = { 2 };
        B.equalityOriginMap[100] = { gl::IdOrigin{ gl::OriginTag::broadcast, { 5 } } };
        B.equalityOriginMap[200] = { gl::IdOrigin{ gl::OriginTag::implication, { 6 } } };

        gl::EquivalenceClass exp;
        exp.memberIds = gl::unionMemberIdsByName(B.memberIds, A.memberIds, m.nameMap);
        exp.intEqualityLevelsMap = B.intEqualityLevelsMap;
        overLevels(exp.intEqualityLevelsMap, A.intEqualityLevelsMap);
        exp.equalityOriginMap = B.equalityOriginMap;
        gl::overwriteOriginsId(exp.equalityOriginMap, A.equalityOriginMap, cap);

        gl::MergeClassAccum accA(&lb);
        seedAccum(accA, A);
        const std::vector<char> bBlob = gl::serializeEquivalenceClass(B);
        const gl::EquivalenceClassView bView{ bBlob.data(),
                                              static_cast<int32_t>(bBlob.size()) };
        const std::vector<int16_t> eqArgsV{ a, b };
        ea.mergeTwoEquivalenceClasses(accA, bView, eqArgsV.data(),
                                      static_cast<int32_t>(eqArgsV.size()),
                                      nullptr, 0, m, gl::StrSpan("main", 4), gl::StrSpan("main", 4));
        ASSERT_TRUE(accA.serialize() == gl::serializeEquivalenceClass(exp));
    }

    // ---------- cross-product (new equality {a,b}, classB {b,c}, bridge b) ----------
    {
        const std::set<int> lv0 = { 7 };                    // the new equality's levels
        gl::EquivalenceClass A, B;
        A.memberIds = { a, b };                             // the eqArgs class
        A.intEqualityLevelsMap[gl::packEqPairKey(a, b)] = { 1 };
        A.equalityOriginMap[100] = { gl::IdOrigin{ gl::OriginTag::equality1, { 1 } } };
        B.memberIds = { b, c };                             // overlaps via b (not a)
        B.intEqualityLevelsMap[gl::packEqPairKey(b, c)] = { 4 };
        B.equalityOriginMap[300] = { gl::IdOrigin{ gl::OriginTag::broadcast, { 8 } } };

        // Heap oracle (trackHistory off -> no equality2 / mergedOriginMap): the
        // single pair (idA=a, idB=c), bridge b:
        // mergedMap[{a,c}] = lv0 ∪ A{b,a} ∪ B{b,c}.
        std::map<uint32_t, std::set<int> > mergedMap;
        {
            std::set<int> nl = lv0;
            nl.insert(A.intEqualityLevelsMap.at(gl::packEqPairKey(a, b)).begin(),
                      A.intEqualityLevelsMap.at(gl::packEqPairKey(a, b)).end());
            nl.insert(B.intEqualityLevelsMap.at(gl::packEqPairKey(b, c)).begin(),
                      B.intEqualityLevelsMap.at(gl::packEqPairKey(b, c)).end());
            mergedMap[gl::packEqPairKey(a, c)] = nl;
        }
        gl::EquivalenceClass exp;
        exp.memberIds = gl::unionMemberIdsByName(B.memberIds, A.memberIds, m.nameMap);
        exp.intEqualityLevelsMap = B.intEqualityLevelsMap;      // (B | A)
        overLevels(exp.intEqualityLevelsMap, A.intEqualityLevelsMap);
        {
            std::map<uint32_t, std::set<int> > tmp = mergedMap;  // mergedMap | (B|A)
            overLevels(tmp, exp.intEqualityLevelsMap);
            exp.intEqualityLevelsMap = tmp;
        }
        exp.equalityOriginMap = B.equalityOriginMap;             // B-base + A
        gl::overwriteOriginsId(exp.equalityOriginMap, A.equalityOriginMap, cap);

        gl::MergeClassAccum accA(&lb);
        seedAccum(accA, A);
        const std::vector<char> bBlob = gl::serializeEquivalenceClass(B);
        const gl::EquivalenceClassView bView{ bBlob.data(),
                                              static_cast<int32_t>(bBlob.size()) };
        const std::vector<int16_t> eqArgsV{ a, b };
        const int lv0Run[1] = { 7 };
        ea.mergeTwoEquivalenceClasses(accA, bView, eqArgsV.data(),
                                      static_cast<int32_t>(eqArgsV.size()),
                                      lv0Run, 1, m, gl::StrSpan("main", 4), gl::StrSpan("main", 4));
        ASSERT_TRUE(accA.serialize() == gl::serializeEquivalenceClass(exp));
    }
}

// Raw-door run splice (the equi-6b same-scope write-back primitive):
// concatenating the serialized blobs of a class list and writing the run
// through the inner `assignRun` byte door reconstructs the exact list, in
// order, field-for-field — byte-identical to `assignClassesById`. The kept-blob
// splice relies on this: a kept class's stored blob equals re-serializing its
// decoded form (a lossless, deterministic codec), so copying it verbatim ahead
// of the merged blob yields the same run the high-level path would have written.
TEST(memory, eqclass_raw_door_run_splice) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::Memory m;

    gl::EquivalenceClass a;
    a.setMembersFromNames({ "a", "b" }, m.nameMap);
    a.intEqualityLevelsMap[gl::packEqPairKey(
        m.nameMap.encode("a"), m.nameMap.encode("b"))] = { 0, 3 };
    gl::EquivalenceClass c;
    c.setMembersFromNames({ "x", "y", "z" }, m.nameMap);

    const std::vector<char> ba = gl::serializeEquivalenceClass(a);
    const std::vector<char> bc = gl::serializeEquivalenceClass(c);
    std::vector<char> concat(ba);
    concat.insert(concat.end(), bc.begin(), bc.end());
    const int32_t lens[2] = { static_cast<int32_t>(ba.size()),
                              static_cast<int32_t>(bc.size()) };
    const int16_t vId = m.nameMap.encode("spliceScope");
    m.equivalenceClassesMap.inner().assignRun(vId, concat.data(), lens, 2);

    const std::vector<gl::EquivalenceClass> got = m.decodeClassesById(vId);
    ASSERT_EQ(got.size(), static_cast<std::size_t>(2));
    ASSERT_TRUE(got[0].memberIds == a.memberIds);
    ASSERT_TRUE(got[0].intEqualityLevelsMap == a.intEqualityLevelsMap);
    ASSERT_TRUE(got[0].equalityOriginMap == a.equalityOriginMap);
    ASSERT_TRUE(got[1].memberIds == c.memberIds);
    ASSERT_TRUE(got[1].intEqualityLevelsMap == c.intEqualityLevelsMap);
}

// IdOrigin codec (the standalone cold exprOriginMap blob value) — one history
// line round-trips, and its bytes equal the per-line layout
// serializeEquivalenceClass emits, so equi-class and body origin blobs share one
// canonical line layout.
TEST(memory, codec_idorigin_round_trip_and_byte_identity) {
    const gl::IdOrigin line{ gl::OriginTag::implication, { 11, -22, 0x1234567890LL } };
    const std::vector<char> blob = gl::Codec<gl::IdOrigin>::serialize(line);

    // Hand-built canonical layout: [uint8 tag][int32 depCount][depCount x int64].
    std::vector<char> expected;
    const auto put = [&expected](const auto& x) {
        const char* p = reinterpret_cast<const char*>(&x);
        expected.insert(expected.end(), p, p + sizeof(x));
    };
    put(static_cast<uint8_t>(gl::OriginTag::implication));
    put(static_cast<int32_t>(3));
    put(static_cast<int64_t>(11));
    put(static_cast<int64_t>(-22));
    put(static_cast<int64_t>(0x1234567890LL));
    ASSERT_TRUE(blob == expected);

    const gl::IdOrigin back = gl::Codec<gl::IdOrigin>::deserialize(
        blob.data(), static_cast<int32_t>(blob.size()));
    ASSERT_TRUE(back == line);

    // An empty-dependency line round-trips and is exactly 5 bytes (tag + count).
    const gl::IdOrigin empty{ gl::OriginTag::theorem, {} };
    const std::vector<char> eb = gl::Codec<gl::IdOrigin>::serialize(empty);
    ASSERT_EQ(eb.size(), static_cast<std::size_t>(5));
    const gl::IdOrigin eback = gl::Codec<gl::IdOrigin>::deserialize(
        eb.data(), static_cast<int32_t>(eb.size()));
    ASSERT_TRUE(eback == empty);
}

// Part C record value codecs — each is a lossless, fixed-point round trip
// (deserialize(serialize(v)) reproduces every field; re-serializing reproduces
// the blob), the property the typed cold blob maps rely on.

TEST(memory, codec_local_memory_value_round_trip) {
    gl::LocalMemoryValue v;
    v.valueId = 42;
    v.originalImplicationId = 7;
    v.justification = gl::RuleJustification::none;
    v.validityId = 3;
    v.isMarker = true;
    v.productOfDisintegration = false;
    v.levels = { 1, 5, 9 };
    v.keyIds = { 10, 20, 30 };
    v.remainingArgIds = { -1, 100 };
    const std::vector<char> b = gl::Codec<gl::LocalMemoryValue>::serialize(v);
    const gl::LocalMemoryValue back =
        gl::Codec<gl::LocalMemoryValue>::deserialize(
            b.data(), static_cast<int32_t>(b.size()));
    ASSERT_EQ(back.valueId, 42);
    ASSERT_EQ(back.validityId, static_cast<int16_t>(3));
    ASSERT_TRUE(back.isMarker);
    ASSERT_FALSE(back.productOfDisintegration);
    ASSERT_TRUE(back.levels == v.levels);
    ASSERT_TRUE(back.keyIds == v.keyIds);
    ASSERT_TRUE(back.remainingArgIds == v.remainingArgIds);
    const std::vector<char> b2 = gl::Codec<gl::LocalMemoryValue>::serialize(back);
    ASSERT_TRUE(b == b2);
}

TEST(memory, codec_owner_set_round_trip) {
    gl::OwnerSet v;
    v.hasLooseOwner = true;
    v.partitionIds = { -3, 0, 7, 1000 };
    v.uSignatures.insert({ { int16_t(1), int16_t(2) },
                           { int16_t(3), int16_t(4) } });
    v.uSignatures.insert({ { int16_t(-5), int16_t(9) } });
    const std::vector<char> b = gl::Codec<gl::OwnerSet>::serialize(v);
    const gl::OwnerSet back = gl::Codec<gl::OwnerSet>::deserialize(
        b.data(), static_cast<int32_t>(b.size()));
    ASSERT_TRUE(back.hasLooseOwner);
    ASSERT_TRUE(back.partitionIds == v.partitionIds);
    ASSERT_TRUE(back.uSignatures == v.uSignatures);
    const std::vector<char> b2 = gl::Codec<gl::OwnerSet>::serialize(back);
    ASSERT_TRUE(b == b2);
}

TEST(memory, codec_admission_map_value_round_trip) {
    const gl::AdmissionMapValue v({ 11, 22, 33 }, { 4, 5 }, 7, 9, true);
    const std::vector<char> b = gl::Codec<gl::AdmissionMapValue>::serialize(v);
    const gl::AdmissionMapValue back =
        gl::Codec<gl::AdmissionMapValue>::deserialize(
            b.data(), static_cast<int32_t>(b.size()));
    ASSERT_TRUE(back.key == v.key);
    ASSERT_TRUE(back.remainingArgs == v.remainingArgs);
    ASSERT_EQ(back.standardMaxAdmissionDepth, 7);
    ASSERT_EQ(back.standardMaxSecondaryNumber, 9);
    ASSERT_TRUE(back.flag);
    const std::vector<char> b2 =
        gl::Codec<gl::AdmissionMapValue>::serialize(back);
    ASSERT_TRUE(b == b2);
}

TEST(memory, codec_rejected_map_value_round_trip) {
    const gl::RejectedMapValue v(1, 2, 3, 4, { 5, 6, 7 },
                                 std::set<int>{ 8, 9 });
    const std::vector<char> b = gl::Codec<gl::RejectedMapValue>::serialize(v);
    const gl::RejectedMapValue back =
        gl::Codec<gl::RejectedMapValue>::deserialize(
            b.data(), static_cast<int32_t>(b.size()));
    ASSERT_EQ(back.renamedExpression, 1);
    ASSERT_EQ(back.expression, 2);
    ASSERT_EQ(back.iteration, 3);
    ASSERT_EQ(back.concreteConstituent, 4);
    ASSERT_TRUE(back.siblings == v.siblings);
    ASSERT_TRUE(back.levels == v.levels);
    const std::vector<char> b2 = gl::Codec<gl::RejectedMapValue>::serialize(back);
    ASSERT_TRUE(b == b2);
}

TEST(memory, codec_rejected_map_integration_value_round_trip) {
    const gl::RejectedMapIntegrationValue v(7, { 1, 2, 3 }, 99);
    const std::vector<char> b =
        gl::Codec<gl::RejectedMapIntegrationValue>::serialize(v);
    const gl::RejectedMapIntegrationValue back =
        gl::Codec<gl::RejectedMapIntegrationValue>::deserialize(
            b.data(), static_cast<int32_t>(b.size()));
    ASSERT_EQ(back.concreteConstituent, 7);
    ASSERT_EQ(back.compoundExpression, 99);
    ASSERT_TRUE(back.siblings == v.siblings);
    const std::vector<char> b2 =
        gl::Codec<gl::RejectedMapIntegrationValue>::serialize(back);
    ASSERT_TRUE(b == b2);
}

TEST(memory, codec_integration_entry_round_trip) {
    gl::IntegrationEntry v;
    v.instruction.markedGoal = 5;
    gl::IntLogicalEntity e1;
    e1.category = 1; e1.signature = 2; e1.arity = 3; e1.definedSet = 4;
    e1.elements = { 10, 11, 12 };
    gl::IntLogicalEntity e2;
    e2.category = -1; e2.signature = 0; e2.arity = 1; e2.definedSet = 7;
    e2.elements = {};
    v.instruction.data = { e1, e2 };
    v.valueIds = { 100, 200, 300 };
    const std::vector<char> b = gl::Codec<gl::IntegrationEntry>::serialize(v);
    const gl::IntegrationEntry back =
        gl::Codec<gl::IntegrationEntry>::deserialize(
            b.data(), static_cast<int32_t>(b.size()));
    ASSERT_EQ(back.instruction.markedGoal, 5);
    ASSERT_EQ(back.instruction.data.size(), static_cast<std::size_t>(2));
    ASSERT_TRUE(back.instruction.data[0].elements == e1.elements);
    ASSERT_EQ(back.instruction.data[1].arity, 1);
    ASSERT_TRUE(back.valueIds == v.valueIds);
    const std::vector<char> b2 = gl::Codec<gl::IntegrationEntry>::serialize(back);
    ASSERT_TRUE(b == b2);
}

// ChangedClassesBuffer — the per-step delta on the LB's cold deloadable arena:
// push / size / validityAt / classAt round trip, duplicate validities, clear,
// and reuse after release.
TEST(memory, changed_classes_buffer_round_trip) {
    gl::GlobalMemoryManager g; g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g); gl::DirtyState d = gl::DirtyState::Clean;
    gl::ChangedClassesBuffer buf(&lb, &d);
    ASSERT_EQ(buf.size(), 0);

    const gl::EquivalenceClass a = makeSampleClass(false);
    gl::EquivalenceClass small;
    small.memberIds = { 1, 2 };

    buf.push(10, a);
    buf.push(20, small);
    buf.push(10, a);                              // duplicate validity is fine
    ASSERT_EQ(buf.size(), 3);
    ASSERT_EQ(buf.validityAt(0), 10);
    ASSERT_EQ(buf.validityAt(1), 20);
    ASSERT_EQ(buf.validityAt(2), 10);
    ASSERT_TRUE(buf.classAt(0).memberIds == a.memberIds);
    ASSERT_TRUE(buf.classAt(0).intEqualityLevelsMap == a.intEqualityLevelsMap);
    ASSERT_TRUE(buf.classAt(0).equalityOriginMap == a.equalityOriginMap);
    ASSERT_TRUE(buf.classAt(1).memberIds == small.memberIds);
    ASSERT_TRUE(buf.classAt(2).equalityOriginMap == a.equalityOriginMap);

    buf.clear();
    ASSERT_EQ(buf.size(), 0);

    // Reuse after clear, then after a full arena release.
    buf.push(7, small);
    ASSERT_EQ(buf.size(), 1);
    ASSERT_EQ(buf.validityAt(0), 7);
    buf.release();
    ASSERT_EQ(buf.size(), 0);
    buf.push(9, a);
    ASSERT_EQ(buf.size(), 1);
    ASSERT_TRUE(buf.classAt(0).memberIds == a.memberIds);
}

// classViewAt — the zero-copy view reads the same members + levels as the heap
// classAt decode, and the shared EquivalenceClass/EquivalenceClassView accessors
// (memberCount/memberId/forEachLevel/classHasMember) agree (equi-7 read path).
TEST(memory, changed_classes_view_matches_class) {
    gl::GlobalMemoryManager g; g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
    gl::LbArena lb(&g); gl::DirtyState d = gl::DirtyState::Clean;
    gl::ChangedClassesBuffer buf(&lb, &d);

    const gl::EquivalenceClass a = makeSampleClass(false);   // members {-5,3,7,12,100}
    buf.push(10, a);

    std::vector<char> scratch;
    const gl::EquivalenceClassView v = buf.classViewAt(0, scratch);

    // Members: the view, its own accessors, and the heap class all agree.
    ASSERT_EQ(v.memberCount(), a.memberCount());
    ASSERT_EQ(a.memberCount(), static_cast<int32_t>(a.memberIds.size()));
    for (int32_t i = 0; i < v.memberCount(); ++i) {
        ASSERT_EQ(v.memberId(i), a.memberId(i));
        ASSERT_EQ(a.memberId(i), a.memberIds[static_cast<std::size_t>(i)]);
    }

    // classHasMember is uniform over the view and the heap class.
    ASSERT_TRUE(gl::classHasMember(v, static_cast<int16_t>(7)));
    ASSERT_TRUE(gl::classHasMember(a, static_cast<int16_t>(7)));
    ASSERT_FALSE(gl::classHasMember(v, static_cast<int16_t>(999)));

    // forEachLevel: present keys yield identical ascending levels; an absent key
    // yields nothing on both.
    const uint32_t keys[] = { gl::packEqPairKey(3, 7), gl::packEqPairKey(7, 12),
                              gl::packEqPairKey(1, 2) };
    for (const uint32_t key : keys) {
        std::vector<int> fromView, fromHeap;
        v.forEachLevel(key, [&](int lv) { fromView.push_back(lv); });
        a.forEachLevel(key, [&](int lv) { fromHeap.push_back(lv); });
        ASSERT_TRUE(fromView == fromHeap);
    }
}

// classViewAt ScratchArena& overload — the arena-backed twin returns a
// BYTE-IDENTICAL EquivalenceClassView to the std::vector<char> overload (the
// ORACLE) on BOTH the single-page contiguous path AND a forced page straddle.
// This is the substrate twin applyEquiClasses' clsScratch statification
// (I-138) rests on: the equi-class apply peeks
// its class views onto a per-slot scratch arena instead of an owned
// std::vector<char>.
TEST(memory, changed_classes_view_arena_matches_vector) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });   // blobPool_ page = 8192 B
    gl::LbArena lb(&g); gl::DirtyState d = gl::DirtyState::Clean;
    gl::ChangedClassesBuffer buf(&lb, &d);

    // Entry 0: a small class whose blob fits on one page (contiguous).
    const gl::EquivalenceClass small = makeSampleClass(false);
    buf.push(10, small);

    // Entry 1: a class with 5000 members -> the member section alone is
    // 4 + 2*5000 = 10004 B, so its blob straddles the 8192-B blobPool_ page.
    // 5000 fits int16_t and is far under the 256-KiB byte-bump block, so the
    // arena assembles the straddle in ONE alloc (no block-ceiling assert).
    gl::EquivalenceClass big;
    big.memberIds.reserve(5000);
    for (int i = 1; i <= 5000; ++i)
        big.memberIds.push_back(static_cast<int16_t>(i));
    buf.push(20, big);

    // ScratchArena === LbArena; bind a distinct byte-bump arena to the pool.
    gl::ScratchArena scr; scr.bind(&g);

    const auto checkEntry = [&](int32_t i, bool expectStraddle) {
        // Oracle: the std::vector<char> overload.
        std::vector<char> vs;
        const gl::EquivalenceClassView vv = buf.classViewAt(i, vs);
        // Arena overload under test.
        const gl::EquivalenceClassView av = buf.classViewAt(i, scr);

        // Byte-identical: same length, same bytes.
        ASSERT_EQ(av.len, vv.len);
        ASSERT_TRUE(std::memcmp(av.p, vv.p,
                                static_cast<std::size_t>(vv.len)) == 0);
        // Members agree through the view accessors too.
        ASSERT_EQ(av.memberCount(), vv.memberCount());
        for (int32_t m = 0; m < av.memberCount(); ++m)
            ASSERT_EQ(av.memberId(m), vv.memberId(m));

        if (expectStraddle) {
            // Straddle: the arena assembled a FRESH buffer, distinct from the
            // oracle's vector copy.
            ASSERT_TRUE(av.p != vv.p);
        } else {
            // Contiguous: both overloads return the SAME in-place pool pointer
            // and never touch their scratch.
            ASSERT_TRUE(av.p == vv.p);
        }
    };
    checkEntry(0, false);
    checkEntry(1, true);
}

// SpecialTokenScan codec — lossless round trip including the empty scan.
TEST(memory, special_token_scan_codec_round_trip) {
    gl::SpecialTokenScan a;
    a.intTokens = { "int_lev_0_1", "int_lev_2_3" };
    a.itTokens = { "it_2_lev_0_3" };
    const std::vector<char> blob = gl::serializeSpecialTokenScan(a);
    const gl::SpecialTokenScan b = gl::deserializeSpecialTokenScan(
        blob.data(), static_cast<int32_t>(blob.size()));
    ASSERT_TRUE(b.intTokens == a.intTokens);
    ASSERT_TRUE(b.itTokens == a.itTokens);

    const gl::SpecialTokenScan empty;
    const std::vector<char> eblob = gl::serializeSpecialTokenScan(empty);
    ASSERT_EQ(eblob.size(), static_cast<std::size_t>(8));   // two int32 zero counts
    const gl::SpecialTokenScan eb = gl::deserializeSpecialTokenScan(
        eblob.data(), static_cast<int32_t>(eblob.size()));
    ASSERT_TRUE(eb.intTokens.empty());
    ASSERT_TRUE(eb.itTokens.empty());
}

// ~Memory empties the arena-backed encodedMaps before the LB's arena
// destructs. All four HashMemory instances are declared before lbMemory and all
// ride lbMemory.manager, so without the destructor body their encodedMap's
// ~PagedVector would freePage on an already-destroyed arena. Populate all four
// encodedMaps, then let the Memory leave scope: a clean exit proves ~Memory
// emptied them in time — a regression would fire the freePage out-of-range
// assert during destruction.
TEST(memory, dtor_empties_arena_backed_encodedmap_before_arena) {
    int16_t buf[2] = { 7, 9 };
    const gl::NormKey nk{ 1, std::vector<int16_t>(buf, buf + 2) };
    {
        gl::Memory m;
        m.overallHashMemory.encodedMap.assignRun(
            nk, std::vector<gl::LocalMemoryValue>{ gl::LocalMemoryValue{} });
        m.localHashMemory.encodedMap.assignRun(
            nk, std::vector<gl::LocalMemoryValue>{ gl::LocalMemoryValue{} });
        m.localHashMemoryDelta.encodedMap.assignRun(
            nk, std::vector<gl::LocalMemoryValue>{ gl::LocalMemoryValue{} });
        m.workingMemory.encodedMap.assignRun(
            nk, std::vector<gl::LocalMemoryValue>{ gl::LocalMemoryValue{} });
        ASSERT_EQ(m.overallHashMemory.encodedMap.count(), 1);
        ASSERT_EQ(m.localHashMemory.encodedMap.count(), 1);
        ASSERT_EQ(m.localHashMemoryDelta.encodedMap.count(), 1);
        ASSERT_EQ(m.workingMemory.encodedMap.count(), 1);
    }  // ~Memory here must empty all four cold encodedMaps before lbMemory destructs.
    ASSERT_TRUE(true);  // reached => no freePage-on-dead-arena assert.
}

// destroyGrid deletes every child Memory (each child's ~Memory empties its
// arena-backed encodedMaps first) and clears the grid links — no freePage on a
// dead arena even when a child's encodedMap is populated (the CE-clone shape).
TEST(prover, destroygrid_deletes_children_and_clears_grid) {
    gl::ExpressionAnalyzer ea("Peano");
    gl::Memory* child = ea.lbStore.create<gl::Memory>();
    int16_t buf[2] = { 3, 5 };
    const gl::NormKey nk{ 1, std::vector<int16_t>(buf, buf + 2) };
    child->overallHashMemory.encodedMap.assignRun(
        nk, std::vector<gl::LocalMemoryValue>{ gl::LocalMemoryValue{} });
    ea.simpleMapStore.linkChild(&ea.body, "destroygrid-test-child", child);
    ea.destroyGrid();   // deletes child (~Memory empties it) + clears the grid
    ASSERT_TRUE(ea.simpleMapStore.empty());
}

// parseHypoScopeVars — byte-exact extraction twin of reactToHypo's retired
// inline std::string block. ORACLE = that block replicated verbatim below
// (rfind / find / substr / std::set<std::string>); for every input the gate
// verdict, the parent slice, and the deduplicated variable SET must match
// (the twin's insertion-order array is compared as a set — membership-only
// consumption downstream). Adversarial inputs where the historical parse is
// arguably "wrong" (a variable whose text contains "_var1_", a broken marker
// sequence) must reproduce the oracle exactly, right or wrong.
TEST(prover, parse_hypo_scope_vars_matches_string_oracle) {
    // The retired block, verbatim (gates 1-4 + the marker walk).
    struct OracleResult {
        bool ok = false;
        std::string parent;
        std::set<std::string> vars;
    };
    const auto oracle = [](const std::string& validityName) {
        OracleResult r;
        const std::string hypoMarker = "_hypo_";
        size_t posHypo = validityName.rfind(hypoMarker);
        if (posHypo == std::string::npos) return r;
        if (validityName.find("__or__") != std::string::npos) return r;
        std::string boundaryMarker = "_boundary__var0_";
        size_t posBoundary = validityName.rfind(boundaryMarker, posHypo);
        if (posBoundary == std::string::npos) return r;
        r.parent = validityName.substr(0, posBoundary);
        size_t posVar = posBoundary + 10;
        int idx = 0;
        size_t currentPos = posVar;
        size_t suffixPos = posHypo;
        while (true) {
            std::string currentMarker = "_var" + std::to_string(idx) + "_";
            if (currentPos + currentMarker.size() > validityName.size()
                || validityName.compare(currentPos, currentMarker.size(),
                                        currentMarker) != 0) {
                break;
            }
            size_t startContent = currentPos + currentMarker.size();
            std::string nextMarker = "_var" + std::to_string(idx + 1) + "_";
            size_t nextPos = validityName.find(nextMarker, startContent);
            size_t endContent;
            bool isLast = false;
            if (nextPos == std::string::npos || nextPos >= suffixPos) {
                endContent = suffixPos;
                isLast = true;
            } else {
                endContent = nextPos;
            }
            r.vars.insert(
                validityName.substr(startContent, endContent - startContent));
            if (isLast) break;
            currentPos = nextPos;
            idx++;
        }
        r.ok = !r.vars.empty();
        return r;
    };

    gl::ExpressionAnalyzer ea("Peano");
    const std::string cases[] = {
        // Canonical producer shapes: 1, 2, 3 variables.
        "main_boundary__var0_x_hypo_(in[x,1])",
        "main_boundary__var0_x_var1_y_hypo_(in2[x,y])",
        "main_boundary__var0_x_var1_y_var2_z_hypo_(in3[x,y,z])",
        // Gate rejections: no _hypo_; __or__ tagged; no _boundary__var0_.
        "main_boundary__var0_x",
        "main__or__1_boundary__var0_x_hypo_(p[x])",
        "main_x_hypo_(p[x])",
        // TWO _hypo_ occurrences: rfind takes the LAST; the first rides
        // into the variable content.
        "main_boundary__var0_a_hypo_b_hypo_c",
        // Two _boundary__var0_ occurrences: the bounded rfind takes the
        // last at-or-before posHypo; the first stays in the parent slice.
        "main_boundary__var0_p_boundary__var0_q_hypo_(r[q])",
        // Duplicate variables collapse to one entry.
        "main_boundary__var0_x_var1_x_hypo_(p[x])",
        // Adversarial: intended var0 "a_var1_x" — the parse splits at the
        // embedded "_var1_" (twin must reproduce, right or wrong).
        "main_boundary__var0_a_var1_x_hypo_(p[y])",
        // Broken marker sequence (_var0_.._var2_ skipping 1): the walk
        // breaks after var0, swallowing the rest into var0's content.
        "main_boundary__var0_a_var2_b_hypo_(p[a])",
        // Next marker exists only PAST the hypo marker: nextPos >= suffixPos
        // turns it into "last variable" (the whole-span find subtlety).
        "main_boundary__var0_a_hypo_x_var1_y",
        // Empty payload variable: zero-length var still counts (gate 4
        // passes with the empty string in the set).
        "main_boundary__var0__hypo_(p[x])",
    };
    for (const std::string& name : cases) {
        const OracleResult ref = oracle(name);
        gl::ExpressionAnalyzer::HypoScopeParse parse;
        const bool ok = ea.parseHypoScopeVars(gl::StrSpan(name), parse);
        ASSERT_EQ(ok, ref.ok);
        if (!ok) continue;
        // Parent bytes equal.
        ASSERT_EQ(parse.parentValidity.toStdString(), ref.parent);
        // Variable SET equal: same size after dedup, every oracle element
        // present in the twin array, and the twin array duplicate-free.
        ASSERT_EQ(static_cast<size_t>(parse.varCount), ref.vars.size());
        for (const std::string& v : ref.vars) {
            bool found = false;
            for (int32_t i = 0; i < parse.varCount; ++i) {
                if (gl::equalSpans(parse.vars[i], gl::StrSpan(v))) {
                    found = true;
                    break;
                }
            }
            ASSERT_TRUE(found);
        }
        for (int32_t i = 0; i < parse.varCount; ++i) {
            for (int32_t j = i + 1; j < parse.varCount; ++j) {
                ASSERT_FALSE(gl::equalSpans(parse.vars[i], parse.vars[j]));
            }
        }
    }
}

// reactToHypo's vid snapshot + decoded-lex sort — the production comparator
// (compareSpans over decodeView, copied verbatim) must reproduce the retired
// std::pair<std::string,int16_t> snapshot sorted by a.first < b.first (the
// oracle, replicated below). Mint order deliberately disagrees with lex
// order so an id-order regression fails.
TEST(memory, react_to_hypo_vid_sort_matches_string_pair_sort) {
    gl::Memory m;
    const char* names[] = {
        "zeta_scope", "alpha", "mid_scope", "beta_x", "aa", "omega_9",
    };
    for (const char* nm : names) {
        const int16_t vid = m.nameMap.encode(nm);
        m.assignClassesById(
            vid, std::vector<gl::EquivalenceClass>{ gl::EquivalenceClass{} });
    }

    // Production shape: vid array over keyAt(1..count), sorted by the
    // decoded-name comparator (the reactToHypo copy).
    const int32_t n = m.equivalenceClassesMap.count();
    ASSERT_EQ(n, 6);
    std::vector<int16_t> vids(static_cast<size_t>(n));
    for (int32_t kid = 1; kid <= n; ++kid)
        vids[static_cast<size_t>(kid - 1)] = m.equivalenceClassesMap.keyAt(kid);
    std::sort(vids.begin(), vids.end(), [&](int16_t a, int16_t b) {
        return gl::compareSpans(m.nameMap.decodeView(a),
                                m.nameMap.decodeView(b)) < 0;
    });

    // Oracle: the retired snapshot code.
    std::vector<std::pair<std::string, int16_t>> classRows;
    for (int32_t kid = 1; kid <= n; ++kid) {
        const int16_t vId = m.equivalenceClassesMap.keyAt(kid);
        classRows.emplace_back(std::string(m.nameMap.decode(vId)), vId);
    }
    std::sort(classRows.begin(), classRows.end(),
              [](const std::pair<std::string, int16_t>& a,
                 const std::pair<std::string, int16_t>& b) {
                  return a.first < b.first;
              });

    for (int32_t i = 0; i < n; ++i) {
        ASSERT_EQ(vids[static_cast<size_t>(i)],
                  classRows[static_cast<size_t>(i)].second);
    }
}

// reactToHypo's per-row member-run snapshot — the production peek idiom
// (lookup + runLen + peekRecordBytes -> EquivalenceClassView -> memberPool /
// classStarts, copied verbatim) must carry exactly the member runs the heap
// decodeClassesById oracle decodes: same class count, same per-class member
// count, same ids in the same storage order. Covers multi-member,
// single-member, and the empty-class edge.
TEST(memory, hypo_member_snapshot_matches_decode_classes_by_id) {
    gl::Memory m;
    const int16_t vid = m.nameMap.encode("snapshot_scope");

    gl::EquivalenceClass c1;
    c1.setMembersFromNames({ "int_lev_0_1", "x", "it_2_lev_0_3" }, m.nameMap);
    gl::EquivalenceClass c2;
    c2.setMembersFromNames({ "y" }, m.nameMap);
    gl::EquivalenceClass c3;   // empty-class edge
    m.assignClassesById(vid, std::vector<gl::EquivalenceClass>{ c1, c2, c3 });

    // Production snapshot idiom (reactToHypo copy) on a test-local arena.
    gl::ScratchArena snapArena{ &gl::staticMemory() };
    gl::DirtyState poolDirty = gl::DirtyState::Clean;
    gl::PagedVector<int16_t> memberPool(&snapArena, &poolDirty);
    gl::PagedVector<int32_t> classStarts(&snapArena, &poolDirty);

    const int32_t bucketId = m.equivalenceClassesMap.lookup(vid);
    ASSERT_NE(bucketId, 0);
    const int32_t classCount = m.equivalenceClassesMap.runLen(bucketId);
    for (int32_t j = 0; j < classCount; ++j) {
        int32_t blen = 0;
        const char* bp = m.equivalenceClassesMap.peekRecordBytes(
            bucketId, j, blen, snapArena);
        const gl::EquivalenceClassView cls{ bp, blen };
        classStarts.push_back(static_cast<int32_t>(memberPool.size()));
        const int32_t memberCount = cls.memberCount();
        for (int32_t k = 0; k < memberCount; ++k)
            memberPool.push_back(cls.memberId(k));
    }

    // Oracle: the heap decode.
    const std::vector<gl::EquivalenceClass> ref = m.decodeClassesById(vid);
    ASSERT_EQ(classCount, static_cast<int32_t>(ref.size()));
    ASSERT_EQ(classCount, 3);
    for (int32_t j = 0; j < classCount; ++j) {
        const int32_t mStart = classStarts[j];
        const int32_t mEnd = (j + 1 < classCount)
            ? classStarts[j + 1]
            : static_cast<int32_t>(memberPool.size());
        ASSERT_EQ(mEnd - mStart,
                  static_cast<int32_t>(ref[static_cast<size_t>(j)]
                                           .memberIds.size()));
        for (int32_t k = 0; k < mEnd - mStart; ++k) {
            ASSERT_EQ(memberPool[mStart + k],
                      ref[static_cast<size_t>(j)]
                          .memberIds[static_cast<size_t>(k)]);
        }
    }
}

// ---------------------------------------------------------------------------
// S5 C4 — the D-172 RMW helpers' arena interiors: blob view, comparator
// twins, arena serializers, the blob-level splice, and the drain fast path.
// Each candidate is pinned against its retained heap oracle byte-for-byte.
// ---------------------------------------------------------------------------

namespace {
    /// Standalone admission/rejected-map rig: a local pool hosts the map +
    /// value-interner tables; the splice scratch rides the harness pool's
    /// reserved gen-scratch slot (the single-threaded production shape).
    struct ValueMapRig {
        gl::GlobalMemoryManager g;
        // Declaration-order init shim: g.init must run before lb binds to it.
        struct Init {
            Init(gl::GlobalMemoryManager& m) {
                m.init(gl::StaticMemoryConfig{ 1 << 20, 1 << 18 });
            }
        } init{ g };
        gl::DirtyState d = gl::DirtyState::Clean;
        gl::LbArena lb{ &g };
        gl::ColdStringTable valTable{ &lb, &d };
        gl::ValueInterner vi;
        ValueMapRig() { vi.bind(&valTable); }
    };
} // namespace

// AdmissionValueBlobView: per-field peeks equal the codec's deserialize on
// several shapes (empty key, empty rems, both populated). The framing assert
// (len == 17 + 4*(keyCount+remCount)) is exercised implicitly by every
// construction; the mismatch path aborts and is not harness-testable, per
// the house convention.
TEST(memory, admission_value_blob_view_matches_deserialize) {
    const auto check = [](const gl::AdmissionMapValue& v) {
        const std::vector<char> blob = gl::Codec<gl::AdmissionMapValue>::serialize(v);
        const gl::AdmissionMapValue back =
            gl::Codec<gl::AdmissionMapValue>::deserialize(
                blob.data(), static_cast<int32_t>(blob.size()));
        const gl::AdmissionValueBlobView view(
            blob.data(), static_cast<int32_t>(blob.size()));
        ASSERT_EQ(view.depth(), back.standardMaxAdmissionDepth);
        ASSERT_EQ(view.sec(), back.standardMaxSecondaryNumber);
        ASSERT_EQ(view.flagByte(), static_cast<uint8_t>(back.flag ? 1 : 0));
        ASSERT_EQ(view.keyCount(), static_cast<int32_t>(back.key.size()));
        for (int32_t i = 0; i < view.keyCount(); ++i)
            ASSERT_EQ(view.keyId(i), back.key[static_cast<std::size_t>(i)]);
        ASSERT_EQ(view.remCount(),
                  static_cast<int32_t>(back.remainingArgs.size()));
        for (int32_t i = 0; i < view.remCount(); ++i)
            ASSERT_EQ(view.remId(i),
                      back.remainingArgs[static_cast<std::size_t>(i)]);
    };
    check(gl::AdmissionMapValue({ 7, 3, 11 }, { 5, 2 }, 4, 9, true));
    check(gl::AdmissionMapValue({}, { 8 }, 0, 0, false));
    check(gl::AdmissionMapValue({ 42 }, {}, -1, 2, false));
    check(gl::AdmissionMapValue({}, {}, 1, 1, true));
}

// admissionBlobLess == DecodedAdmissionValueLess on blobs, both argument
// orders, with mint order deliberately anti-lex (the I-84 trap: id order
// must not leak into the verdict).
TEST(memory, admission_blob_less_matches_decoded_set_comparator) {
    ValueMapRig rig;
    // Anti-lex mints: ids ascend while decoded order descends.
    const int32_t zz = rig.vi.encode("zz");
    const int32_t mm = rig.vi.encode("mm");
    const int32_t aa = rig.vi.encode("aa");
    const int32_t bb = rig.vi.encode("bb");

    std::vector<gl::AdmissionMapValue> vals;
    vals.push_back(gl::AdmissionMapValue({ aa, zz }, { mm }, 1, 2, false)); // shared prefix
    vals.push_back(gl::AdmissionMapValue({ aa, zz, bb }, { mm }, 1, 2, false)); // longer key
    vals.push_back(gl::AdmissionMapValue({ aa, zz }, { bb }, 1, 2, false)); // equal key, rem differs
    vals.push_back(gl::AdmissionMapValue({ aa, zz }, { mm }, 1, 2, true));  // flag only
    vals.push_back(gl::AdmissionMapValue({ aa, zz }, { mm }, 1, 2, false)); // duplicate of [0]
    vals.push_back(gl::AdmissionMapValue({ zz }, {}, 0, 0, false));         // key differs at [0]
    vals.push_back(gl::AdmissionMapValue({ aa, zz }, { mm }, 3, 2, false)); // depth only
    vals.push_back(gl::AdmissionMapValue({ aa, zz }, { mm }, 1, 7, false)); // sec only

    const gl::DecodedAdmissionValueLess oracle{ &rig.vi };
    std::vector<std::vector<char>> blobs;
    for (const gl::AdmissionMapValue& v : vals)
        blobs.push_back(gl::Codec<gl::AdmissionMapValue>::serialize(v));
    for (std::size_t i = 0; i < vals.size(); ++i) {
        for (std::size_t j = 0; j < vals.size(); ++j) {
            const gl::AdmissionValueBlobView a(
                blobs[i].data(), static_cast<int32_t>(blobs[i].size()));
            const gl::AdmissionValueBlobView b(
                blobs[j].data(), static_cast<int32_t>(blobs[j].size()));
            ASSERT_EQ(gl::admissionBlobLess(a, b, rig.vi),
                      oracle(vals[i], vals[j]));
        }
    }
}

// rejectedBlobLess == DecodedRejectedValueLess on blobs, both argument
// orders — multi-sibling and multi-level values, levels crafted so the
// std::set<int> lexicographic order (NOT size-first) decides.
TEST(memory, rejected_blob_less_matches_decoded_set_comparator) {
    ValueMapRig rig;
    const int32_t zz = rig.vi.encode("zz");
    const int32_t mm = rig.vi.encode("mm");
    const int32_t aa = rig.vi.encode("aa");

    std::vector<gl::RejectedMapValue> vals;
    vals.push_back(gl::RejectedMapValue(aa, mm, 1, zz, { mm, zz },
                                        std::set<int>{ 1, 2 }));
    vals.push_back(gl::RejectedMapValue(zz, mm, 1, zz, { mm, zz },
                                        std::set<int>{ 1, 2 }));   // renamed differs
    vals.push_back(gl::RejectedMapValue(aa, zz, 1, zz, { mm, zz },
                                        std::set<int>{ 1, 2 }));   // expression differs
    vals.push_back(gl::RejectedMapValue(aa, mm, 4, zz, { mm, zz },
                                        std::set<int>{ 1, 2 }));   // iteration differs
    vals.push_back(gl::RejectedMapValue(aa, mm, 1, mm, { mm, zz },
                                        std::set<int>{ 1, 2 }));   // constituent differs
    vals.push_back(gl::RejectedMapValue(aa, mm, 1, zz, { mm },
                                        std::set<int>{ 1, 2 }));   // sibling prefix
    vals.push_back(gl::RejectedMapValue(aa, mm, 1, zz, { zz, mm },
                                        std::set<int>{ 1, 2 }));   // sibling order
    vals.push_back(gl::RejectedMapValue(aa, mm, 1, zz, { mm, zz },
                                        std::set<int>{ 0, 3, 9 })); // levels lex beats size
    vals.push_back(gl::RejectedMapValue(aa, mm, 1, zz, { mm, zz },
                                        std::set<int>{ 1 }));       // levels prefix
    vals.push_back(gl::RejectedMapValue(aa, mm, 1, zz, { mm, zz },
                                        std::set<int>{ 1, 2 }));    // duplicate of [0]

    const gl::DecodedRejectedValueLess oracle{ &rig.vi };
    std::vector<std::vector<char>> blobs;
    for (const gl::RejectedMapValue& v : vals)
        blobs.push_back(gl::Codec<gl::RejectedMapValue>::serialize(v));
    for (std::size_t i = 0; i < vals.size(); ++i) {
        for (std::size_t j = 0; j < vals.size(); ++j) {
            const gl::RejectedValueBlobView a(
                blobs[i].data(), static_cast<int32_t>(blobs[i].size()));
            const gl::RejectedValueBlobView b(
                blobs[j].data(), static_cast<int32_t>(blobs[j].size()));
            ASSERT_EQ(gl::rejectedBlobLess(a, b, rig.vi),
                      oracle(vals[i], vals[j]));
        }
    }
}

// The arena serializers write byte-identical blobs to the codec's heap
// serialize, for several shapes of each value type.
TEST(memory, arena_value_serializers_match_codec) {
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);
    const gl::ArenaOffset mark = tArena.cursor();

    const auto checkAdm = [&tArena](const gl::AdmissionMapValue& v) {
        const std::vector<char> oracle =
            gl::Codec<gl::AdmissionMapValue>::serialize(v);
        const gl::StrSpan got = gl::serializeAdmissionValueToArena(tArena, v);
        ASSERT_EQ(got.len, static_cast<int32_t>(oracle.size()));
        ASSERT_EQ(std::memcmp(got.ptr, oracle.data(), oracle.size()), 0);
    };
    checkAdm(gl::AdmissionMapValue({ 7, 3 }, { 5 }, 4, 9, true));
    checkAdm(gl::AdmissionMapValue({}, {}, 0, 0, false));
    checkAdm(gl::AdmissionMapValue({ 1, 2, 3, 4 }, { 9, 8 }, -2, 100, false));

    const auto checkRej = [&tArena](const gl::RejectedMapValue& v) {
        const std::vector<char> oracle =
            gl::Codec<gl::RejectedMapValue>::serialize(v);
        const gl::StrSpan got = gl::serializeRejectedValueToArena(tArena, v);
        ASSERT_EQ(got.len, static_cast<int32_t>(oracle.size()));
        ASSERT_EQ(std::memcmp(got.ptr, oracle.data(), oracle.size()), 0);
    };
    checkRej(gl::RejectedMapValue(1, 2, 3, 4, { 5, 6 }, std::set<int>{ 0, 7 }));
    checkRej(gl::RejectedMapValue(8, 9, 0, 10, {}, {}));
    checkRej(gl::RejectedMapValue(1, 2, 9, 4, {}, std::set<int>{ 5 }));

    tArena.popTo(mark);
}

// The load-bearing C4 pin: a map driven by a test-local VERBATIM replica of
// the retired decode-into-set RMW helper (the retained oracle) is facet-
// identical to a map driven by the blob-level splice — same keys, same
// runLen per key, same record bytes per (id, j). Covers front/middle/back
// insert positions, duplicate inserts, first-touch mints, and id-vs-lex
// divergent interner content.
TEST(memory, insert_admission_blob_sorted_matches_set_rmw_oracle) {
    ValueMapRig rig;
    gl::DirtyState dA = gl::DirtyState::Clean;
    gl::DirtyState dB = gl::DirtyState::Clean;
    gl::TypedColdBlobMap<int32_t, gl::AdmissionMapValue> mapA(&rig.lb, &dA);
    gl::TypedColdBlobMap<int32_t, gl::AdmissionMapValue> mapB(&rig.lb, &dB);
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);

    // Test-local VERBATIM replica of the retired set-RMW helper body.
    const auto oracleInsert =
        [](gl::TypedColdBlobMap<int32_t, gl::AdmissionMapValue>& m,
           int32_t pk, const gl::AdmissionMapValue& value,
           const gl::ValueInterner& vi) {
            gl::AdmissionValueSet s(gl::DecodedAdmissionValueLess{ &vi });
            const int32_t id = m.lookup(pk);
            if (id != 0) {
                const std::vector<gl::AdmissionMapValue> run = m.recordsAt(id);
                for (const gl::AdmissionMapValue& r : run) s.insert(r);
            }
            s.insert(value);
            m.assignRun(pk,
                std::vector<gl::AdmissionMapValue>(s.begin(), s.end()));
        };

    // Anti-lex mints (id order != decoded order).
    const int32_t zz = rig.vi.encode("zz");
    const int32_t mm = rig.vi.encode("mm");
    const int32_t aa = rig.vi.encode("aa");

    // Insert sequence: first-touch, front, middle, duplicate, back — on key
    // 7; plus a second key 9 minted later.
    const std::vector<std::pair<int32_t, gl::AdmissionMapValue>> seq = {
        { 7, gl::AdmissionMapValue({ mm }, { aa }, 1, 1, false) },
        { 7, gl::AdmissionMapValue({ aa }, {}, 0, 0, false) },        // front
        { 7, gl::AdmissionMapValue({ aa, zz }, { mm }, 2, 2, true) }, // middle
        { 7, gl::AdmissionMapValue({ mm }, { aa }, 1, 1, false) },    // duplicate
        { 7, gl::AdmissionMapValue({ zz, zz }, { zz }, 9, 9, true) }, // back
        { 9, gl::AdmissionMapValue({ zz }, { aa, mm }, 3, 4, false) }, // fresh key
        { 9, gl::AdmissionMapValue({ aa }, { aa }, 3, 4, false) },    // front on 9
    };
    for (const auto& step : seq) {
        oracleInsert(mapA, step.first, step.second, rig.vi);
        const gl::ArenaOffset mark = tArena.cursor();
        const gl::StrSpan blob =
            gl::serializeAdmissionValueToArena(tArena, step.second);
        gl::insertAdmissionBlobSorted(mapB, step.first, blob.ptr, blob.len,
                                      rig.vi, tArena);
        tArena.popTo(mark);
    }

    // Facet equality: same key population, same runs, same record bytes.
    ASSERT_EQ(mapA.count(), mapB.count());
    std::vector<char> scrA, scrB;
    for (int32_t id = 1; id <= mapA.count(); ++id) {
        ASSERT_EQ(mapA.decodeKey(id), mapB.decodeKey(id));
        ASSERT_EQ(mapA.runLen(id), mapB.runLen(id));
        for (int32_t j = 0; j < mapA.runLen(id); ++j) {
            int32_t lenA = 0, lenB = 0;
            const char* pa = mapA.peekRecordBytes(id, j, lenA, scrA);
            const char* pb = mapB.peekRecordBytes(id, j, lenB, scrB);
            ASSERT_EQ(lenA, lenB);
            ASSERT_EQ(std::memcmp(pa, pb, static_cast<std::size_t>(lenA)), 0);
        }
    }
}

// Widening door: a key's run driven past one arena block
// (kMaxAdmissionRunBytes) by insertAdmissionBlobSorted stays byte-identical to
// the paged value-form set-RMW oracle. The concat path handles the first values;
// once the run would straddle a block the widening (empty-run assignRun +
// per-blob appendBlobToRun) takes over, and a trailing duplicate exercises the
// widening's dup rewrite. A big local pool holds both multi-block runs; the peek
// scratch rides the process-global genScratchArenas.
TEST(memory, insert_admission_blob_sorted_widens_past_one_block) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 64 << 20, 1 << 18 });
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::LbArena lb{ &g };
    gl::ColdStringTable valTable{ &lb, &d };
    gl::ValueInterner vi;
    vi.bind(&valTable);

    gl::DirtyState dA = gl::DirtyState::Clean;
    gl::DirtyState dB = gl::DirtyState::Clean;
    gl::TypedColdBlobMap<int32_t, gl::AdmissionMapValue> mapA(&lb, &dA);
    gl::TypedColdBlobMap<int32_t, gl::AdmissionMapValue> mapB(&lb, &dB);
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);

    const auto oracleInsert =
        [](gl::TypedColdBlobMap<int32_t, gl::AdmissionMapValue>& m,
           int32_t pk, const gl::AdmissionMapValue& value,
           const gl::ValueInterner& vi) {
            gl::AdmissionValueSet s(gl::DecodedAdmissionValueLess{ &vi });
            const int32_t id = m.lookup(pk);
            if (id != 0) {
                const std::vector<gl::AdmissionMapValue> run = m.recordsAt(id);
                for (const gl::AdmissionMapValue& r : run) s.insert(r);
            }
            s.insert(value);
            m.assignRun(pk,
                std::vector<gl::AdmissionMapValue>(s.begin(), s.end()));
        };

    // ~200 shared padding ids fatten each value to ~0.8 KB, so a few hundred
    // distinct values on one key push the run past one arena block.
    std::vector<int32_t> pad;
    for (int i = 0; i < 200; ++i)
        pad.push_back(vi.encode("pad_" + std::to_string(i)));
    const auto makeValue = [&](int i) {
        std::vector<int32_t> keyIds;
        keyIds.push_back(vi.encode("v_" + std::to_string(i)));  // distinct front
        keyIds.insert(keyIds.end(), pad.begin(), pad.end());
        return gl::AdmissionMapValue(keyIds, {}, i, i, (i & 1) != 0);
    };

    const int32_t key = 42;
    const int N = 400;   // ~328 KB run at ~0.82 KB/value — well past one block
    for (int i = 0; i < N; ++i) {
        const gl::AdmissionMapValue v = makeValue(i);
        oracleInsert(mapA, key, v, vi);
        const gl::ArenaOffset mark = tArena.cursor();
        const gl::StrSpan blob = gl::serializeAdmissionValueToArena(tArena, v);
        gl::insertAdmissionBlobSorted(mapB, key, blob.ptr, blob.len, vi, tArena);
        tArena.popTo(mark);
    }
    // Trailing duplicate on the already-multi-block run: the widening's dup path
    // (rewrite the same run, no new blob).
    {
        const gl::AdmissionMapValue v = makeValue(0);
        oracleInsert(mapA, key, v, vi);
        const gl::ArenaOffset mark = tArena.cursor();
        const gl::StrSpan blob = gl::serializeAdmissionValueToArena(tArena, v);
        gl::insertAdmissionBlobSorted(mapB, key, blob.ptr, blob.len, vi, tArena);
        tArena.popTo(mark);
    }

    // The widening must actually have fired: the run exceeds one arena block.
    int64_t runBytes = 0;
    {
        std::vector<char> scr;
        const int32_t idB = mapB.lookup(key);
        for (int32_t j = 0; j < mapB.runLen(idB); ++j) {
            int32_t l = 0;
            mapB.peekRecordBytes(idB, j, l, scr);
            runBytes += l;
        }
    }
    ASSERT_TRUE(runBytes >
        static_cast<int64_t>(gl::ExecutionParameters::kMaxAdmissionRunBytes));

    // Byte-identity vs the paged value-form oracle across the whole run.
    ASSERT_EQ(mapA.count(), mapB.count());
    std::vector<char> scrA, scrB;
    for (int32_t id = 1; id <= mapA.count(); ++id) {
        ASSERT_EQ(mapA.decodeKey(id), mapB.decodeKey(id));
        ASSERT_EQ(mapA.runLen(id), mapB.runLen(id));
        for (int32_t j = 0; j < mapA.runLen(id); ++j) {
            int32_t lenA = 0, lenB = 0;
            const char* pa = mapA.peekRecordBytes(id, j, lenA, scrA);
            const char* pb = mapB.peekRecordBytes(id, j, lenB, scrB);
            ASSERT_EQ(lenA, lenB);
            ASSERT_EQ(std::memcmp(pa, pb, static_cast<std::size_t>(lenA)), 0);
        }
    }
}

// Mirror pin for the rejected splice — multi-sibling, multi-level values.
TEST(memory, insert_rejected_blob_sorted_matches_set_rmw_oracle) {
    ValueMapRig rig;
    gl::DirtyState dA = gl::DirtyState::Clean;
    gl::DirtyState dB = gl::DirtyState::Clean;
    gl::TypedColdBlobMap<int32_t, gl::RejectedMapValue> mapA(&rig.lb, &dA);
    gl::TypedColdBlobMap<int32_t, gl::RejectedMapValue> mapB(&rig.lb, &dB);
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);

    const auto oracleInsert =
        [](gl::TypedColdBlobMap<int32_t, gl::RejectedMapValue>& m,
           int32_t pk, const gl::RejectedMapValue& value,
           const gl::ValueInterner& vi) {
            gl::RejectedValueSet s(gl::DecodedRejectedValueLess{ &vi });
            const int32_t id = m.lookup(pk);
            if (id != 0) {
                const std::vector<gl::RejectedMapValue> run = m.recordsAt(id);
                for (const gl::RejectedMapValue& r : run) s.insert(r);
            }
            s.insert(value);
            m.assignRun(pk,
                std::vector<gl::RejectedMapValue>(s.begin(), s.end()));
        };

    const int32_t zz = rig.vi.encode("zz_ren");
    const int32_t aa = rig.vi.encode("aa_ren");
    const int32_t ex = rig.vi.encode("(in3[a,b,c,plus])");
    const int32_t cc = rig.vi.encode("(in[b,1])");

    const std::vector<std::pair<int32_t, gl::RejectedMapValue>> seq = {
        { 3, gl::RejectedMapValue(zz, ex, 5, cc, { cc, zz },
                                  std::set<int>{ 1, 4 }) },
        { 3, gl::RejectedMapValue(aa, ex, 0, cc, {}, {}) },          // front
        { 3, gl::RejectedMapValue(aa, ex, 2, cc, { cc },
                                  std::set<int>{ 2 }) },             // middle
        { 3, gl::RejectedMapValue(zz, ex, 5, cc, { cc, zz },
                                  std::set<int>{ 1, 4 }) },          // duplicate
        { 3, gl::RejectedMapValue(aa, ex, 2, cc, { cc },
                                  std::set<int>{ 0, 9 }) },          // level order
        { 11, gl::RejectedMapValue(aa, ex, 1, cc, { zz, cc, aa },
                                   std::set<int>{ 0, 1, 2 }) },      // fresh key
    };
    for (const auto& step : seq) {
        oracleInsert(mapA, step.first, step.second, rig.vi);
        const gl::ArenaOffset mark = tArena.cursor();
        const gl::StrSpan blob =
            gl::serializeRejectedValueToArena(tArena, step.second);
        gl::insertRejectedBlobSorted(mapB, step.first, blob.ptr, blob.len,
                                     rig.vi, tArena);
        tArena.popTo(mark);
    }

    ASSERT_EQ(mapA.count(), mapB.count());
    std::vector<char> scrA, scrB;
    for (int32_t id = 1; id <= mapA.count(); ++id) {
        ASSERT_EQ(mapA.decodeKey(id), mapB.decodeKey(id));
        ASSERT_EQ(mapA.runLen(id), mapB.runLen(id));
        for (int32_t j = 0; j < mapA.runLen(id); ++j) {
            int32_t lenA = 0, lenB = 0;
            const char* pa = mapA.peekRecordBytes(id, j, lenA, scrA);
            const char* pb = mapB.peekRecordBytes(id, j, lenB, scrB);
            ASSERT_EQ(lenA, lenB);
            ASSERT_EQ(std::memcmp(pa, pb, static_cast<std::size_t>(lenA)), 0);
        }
    }
}

// The drain fast path: stagedToArenaBlob's bytes equal
// Codec::serialize(stagedToIdValue(sv)) on TWIN interners fed the same mint
// sequence (same ids), pinning both the byte layout and the frozen
// interner-touch order (keys in rule order, then args in presorted lex
// order).
TEST(memory, staged_to_arena_blob_matches_id_value_oracle) {
    ValueMapRig rigA;   // production path interner
    ValueMapRig rigB;   // oracle path interner (twin mint sequence)
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);

    gl::SealedPageSet pages;
    pages.bind(&gl::staticMemory());
    const auto sealStr = [&pages](const char* s) {
        return gl::SealedString::copyFrom(
            pages, s, static_cast<int32_t>(std::strlen(s)));
    };

    const auto check = [&](std::vector<const char*> keyElems,
                           std::vector<const char*> remElems,
                           int depth, int sec, bool flag) {
        gl::StagedAdmissionValue sv;
        std::vector<gl::SealedString> keyBuf;
        for (const char* k : keyElems) keyBuf.push_back(sealStr(k));
        std::vector<gl::SealedString> remBuf;
        for (const char* a : remElems) remBuf.push_back(sealStr(a));
        sv.key = gl::SealedSpan<gl::SealedString>::copyFrom(
            pages, keyBuf.data(), static_cast<int32_t>(keyBuf.size()));
        sv.remainingArgsSorted = gl::SealedSpan<gl::SealedString>::copyFrom(
            pages, remBuf.data(), static_cast<int32_t>(remBuf.size()));
        sv.standardMaxAdmissionDepth = depth;
        sv.standardMaxSecondaryNumber = sec;
        sv.flag = flag;

        const gl::ArenaOffset mark = tArena.cursor();
        const gl::StrSpan got = gl::stagedToArenaBlob(sv, rigA.vi, tArena);
        const std::vector<char> oracle =
            gl::Codec<gl::AdmissionMapValue>::serialize(
                gl::stagedToIdValue(sv, rigB.vi));
        ASSERT_EQ(got.len, static_cast<int32_t>(oracle.size()));
        ASSERT_EQ(std::memcmp(got.ptr, oracle.data(), oracle.size()), 0);
        tArena.popTo(mark);
    };

    check({ "(in2[a,7,3])", "(in[a,1])" }, { "a", "b" }, 2, 3, true);
    check({}, { "x" }, 0, 0, false);
    check({ "(in[zz,1])" }, {}, 5, 1, false);
    // Shared strings across calls: dedup mints (same ids on both paths).
    check({ "(in2[a,7,3])" }, { "a" }, 1, 1, true);

    pages.seal();
    pages.freePages();
}

// cleanAdmissionMap span door (S5 C5): the function is fed as spans; gate
// verdicts and the consume path are unchanged. skip_eq_classes = true — the
// I-41 closure block is exercised by the full-pipeline batch gate, not this
// unit (it is byte-verbatim string machinery fed by a single
// materialization). Cases: (a) non-operator core -> zero state change;
// (b) operator with marker NOT at an output index -> zero consume;
// (c) operator + marker at the output index -> consumedAdmissionKeys gains
// the key and the admissionMap / admissionStatusMap entries are erased
// (the admissionMapIntegration erase is the documented key-form no-op
// vestige — integration holds u_-form templates, never this bare-marker
// form). Driven both string-fed (the three production callers' shape,
// implicit StrSpan conversion) and mid-buffer span-fed; end states equal.
TEST(prover, clean_admission_map_span_door_consume_path) {
    gl::ExpressionAnalyzer ea("Peano");
    ea.parameters.skip_eq_classes = true;
    gl::ScratchArena& tArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);

    const auto seedEntry = [&tArena](gl::Memory& m, const std::string& key) {
        const int32_t pk = gl::mintTemplateKey(m.templateInterner, m.nameMap,
                                               key, "main");
        gl::AdmissionMapValue v;
        v.key = encodeValueVectorOracle(
            std::vector<std::string>{ "(in2[5,7,3])" }, m.valueInterner);
        gl::insertAdmissionValue(m.overallHashMemory.admissionMap, pk, v,
                                 m.valueInterner, tArena);
        m.overallHashMemory.admissionStatusMap.upsert(
            pk, static_cast<uint8_t>(0));
        return pk;
    };

    // (a) Non-operator core: zero state change (early return).
    {
        gl::Memory m;
        const int32_t pk = seedEntry(m, "(zzz[5,marker])");
        ea.cleanAdmissionMap(std::string("(zzz[5,marker])"),
                             std::string("main"), m);
        ASSERT_TRUE(m.overallHashMemory.admissionMap.lookup(pk) != 0);
        ASSERT_FALSE(m.overallHashMemory.consumedAdmissionKeys.contains(pk));
    }

    // (b) Operator, marker NOT at an output index (in2's output is slot 1;
    //     marker sits at input slot 0): zero consume.
    {
        gl::Memory m;
        const int32_t pk = seedEntry(m, "(in2[marker,7,3])");
        ea.cleanAdmissionMap(std::string("(in2[marker,7,3])"),
                             std::string("main"), m);
        ASSERT_TRUE(m.overallHashMemory.admissionMap.lookup(pk) != 0);
        ASSERT_FALSE(m.overallHashMemory.consumedAdmissionKeys.contains(pk));
    }

    // (c) Operator + marker at the output slot: consumed + erased —
    //     string-fed (mA) and mid-buffer span-fed (mB) end states equal.
    {
        gl::Memory mA;
        const int32_t pkA = seedEntry(mA, "(in2[5,marker,3])");
        ea.cleanAdmissionMap(std::string("(in2[5,marker,3])"),
                             std::string("main"), mA);
        ASSERT_TRUE(mA.overallHashMemory.consumedAdmissionKeys.contains(pkA));
        ASSERT_TRUE(mA.overallHashMemory.admissionMap.lookup(pkA) == 0);
        ASSERT_TRUE(mA.overallHashMemory.admissionStatusMap.find(pkA)
                    == nullptr);

        gl::Memory mB;
        const int32_t pkB = seedEntry(mB, "(in2[5,marker,3])");
        const char buf[] = "xx(in2[5,marker,3])yy";
        const char vbuf[] = "zzmainww";
        ea.cleanAdmissionMap(gl::StrSpan(buf + 2, 17),
                             gl::StrSpan(vbuf + 2, 4), mB);
        ASSERT_TRUE(mB.overallHashMemory.consumedAdmissionKeys.contains(pkB));
        ASSERT_TRUE(mB.overallHashMemory.admissionMap.lookup(pkB) == 0);
        ASSERT_TRUE(mB.overallHashMemory.admissionStatusMap.find(pkB)
                    == nullptr);
    }
}
