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
// Unit tests for OriginalsHeadIndex — the derived head-operator -> chain-id
// index over `HashMemory::originals` that `checkNecessityForEquality` walks.
//
// The index takes the chain count and a head-name decoder, so the tests drive
// it with a synthetic chain table (chain id -> head operator name) and compare
// every answer against the brute-force ascending scan the index replaces.
#include "test_harness.hpp"
#include "../memory_infra/originals_head_index.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace {

    const gl::StaticMemoryConfig kHeadCfg{ 1 << 20, 1 << 18 };

    /// Brute-force twin of `chainsWithHead`: every chain id whose head name
    /// equals `op`, ascending — the retired full walk's candidate set.
    inline std::vector<int32_t> bruteForce(const std::vector<std::string>& heads,
                                           const std::string& op) {
        std::vector<int32_t> out;
        for (std::size_t i = 0; i < heads.size(); ++i)
            if (heads[i] == op) out.push_back(static_cast<int32_t>(i) + 1);
        return out;
    }

    inline std::vector<int32_t> indexed(const gl::OriginalsHeadIndex& idx,
                                        const std::string& op) {
        std::vector<int32_t> out;
        idx.chainsWithHead(gl::StrSpan(op), [&](int32_t id) { out.push_back(id); });
        return out;
    }

    /// Build through the three-argument door with NO masked keys (every chain
    /// left out of the masked family) — the head-family tests.
    inline void build(gl::OriginalsHeadIndex& idx, const std::vector<std::string>& heads) {
        idx.ensureBuilt(static_cast<int32_t>(heads.size()),
            [&](int32_t id) {
                return gl::StrSpan(heads[static_cast<std::size_t>(id) - 1]);
            },
            [](int32_t) { return gl::StrSpan("", 0); });
    }

    /// Build with a per-chain masked key table (an empty entry = the chain is
    /// not acceptable and stays out of the masked family).
    inline void buildMasked(gl::OriginalsHeadIndex& idx,
                            const std::vector<std::string>& heads,
                            const std::vector<std::string>& masked) {
        idx.ensureBuilt(static_cast<int32_t>(heads.size()),
            [&](int32_t id) {
                return gl::StrSpan(heads[static_cast<std::size_t>(id) - 1]);
            },
            [&](int32_t id) {
                return gl::StrSpan(masked[static_cast<std::size_t>(id) - 1]);
            });
    }

    /// Brute-force twin of `chainsWithMaskedKey`: every chain whose (non-empty)
    /// masked key equals `key`, ascending.
    inline std::vector<int32_t> bruteForceMasked(const std::vector<std::string>& masked,
                                                 const std::string& key) {
        std::vector<int32_t> out;
        for (std::size_t i = 0; i < masked.size(); ++i)
            if (!masked[i].empty() && masked[i] == key)
                out.push_back(static_cast<int32_t>(i) + 1);
        return out;
    }

    inline std::vector<int32_t> indexedMasked(const gl::OriginalsHeadIndex& idx,
                                              const std::string& key) {
        std::vector<int32_t> out;
        idx.chainsWithMaskedKey(gl::StrSpan(key), [&](int32_t id) { out.push_back(id); });
        return out;
    }

    /// `maskedHeadKey` over a name, a sign and a std::string argument list.
    inline std::string maskedKeyOf(const std::string& name,
                                   const std::vector<std::string>& args, int32_t slot,
                                   bool negated = false) {
        gl::StrSpan spans[16];
        for (std::size_t i = 0; i < args.size(); ++i) spans[i] = gl::StrSpan(args[i]);
        char buf[gl::kMaxMaskedHeadKeyBytes];
        const gl::StrSpan key = gl::maskedHeadKey(gl::StrSpan(name), negated, spans,
            static_cast<int32_t>(args.size()), slot, buf, gl::kMaxMaskedHeadKeyBytes);
        return std::string(key.ptr, static_cast<std::size_t>(key.len));
    }

    /// `maskedHeadKey` with two masked slots over a name and a std::string
    /// argument list.
    inline std::string maskedKeyOf2(const std::string& name,
                                    const std::vector<std::string>& args,
                                    int32_t slotA, int32_t slotB) {
        gl::StrSpan spans[16];
        for (std::size_t i = 0; i < args.size(); ++i) spans[i] = gl::StrSpan(args[i]);
        char buf[gl::kMaxMaskedHeadKeyBytes];
        const gl::StrSpan key = gl::maskedHeadKey(gl::StrSpan(name), false, spans,
            static_cast<int32_t>(args.size()), slotA, buf, gl::kMaxMaskedHeadKeyBytes, slotB);
        return std::string(key.ptr, static_cast<std::size_t>(key.len));
    }

    inline std::vector<std::string> sampleHeads() {
        return { "in3", "preorder", "in3", "=", "in2", "in3", "preorder",
                 "existence16", "in3", "or11", "in2", "in3" };
    }

} // namespace

// maskedHeadKey: the operator name, then every argument behind a comma with
// the masked slot replaced by the mask byte — equal keys iff equal name, equal
// argument count and equal arguments outside the slot.
TEST(originals_head_index, masked_head_key_masks_exactly_one_slot) {
    std::string expect1 = "in3,u_1,";
    expect1 += gl::kMaskedHeadSlotByte;
    expect1 += ",u_3";
    ASSERT_TRUE(maskedKeyOf("in3", { "u_1", "5", "u_3" }, 1) == expect1);
    std::string expect0 = "in3,";
    expect0 += gl::kMaskedHeadSlotByte;
    expect0 += ",5,u_3";
    ASSERT_TRUE(maskedKeyOf("in3", { "u_1", "5", "u_3" }, 0) == expect0);
    // the deposit side: the same slot masked on the u_-prefixed input equals
    // the chain side iff the other arguments agree
    ASSERT_TRUE(maskedKeyOf("in3", { "u_1", "u_9", "u_3" }, 1) == expect1);
    ASSERT_TRUE(maskedKeyOf("in3", { "u_1", "u_9", "u_4" }, 1) != expect1);
    ASSERT_TRUE(maskedKeyOf("in3", { "u_1", "u_9" }, 1) != expect1);       // arity
    ASSERT_TRUE(maskedKeyOf("in2", { "u_1", "u_9", "u_3" }, 1) != expect1); // name
    ASSERT_TRUE(maskedKeyOf("in3", { "u_1", "u_9", "u_3" }, 0)
                != maskedKeyOf("in3", { "u_1", "u_9", "u_3" }, 1));         // slot
    // a single-argument expression masks to name + mask
    std::string one = "in,";
    one += gl::kMaskedHeadSlotByte;
    ASSERT_TRUE(maskedKeyOf("in", { "u_1" }, 0) == one);
}

// maskedHeadKey with a second slot masks both slots, independent of their
// order; equal keys iff equal name, arity and arguments outside the two
// slots; a two-slot key never equals a one-slot key.
TEST(originals_head_index, masked_head_key_masks_two_slots) {
    std::string expect = "in2,";
    expect += gl::kMaskedHeadSlotByte;
    expect += ',';
    expect += gl::kMaskedHeadSlotByte;
    expect += ",u_6";
    ASSERT_TRUE(maskedKeyOf2("in2", { "1", "2", "u_6" }, 0, 1) == expect);
    // the deposit side: both slots masked on the u_-prefixed input
    ASSERT_TRUE(maskedKeyOf2("in2", { "u_9", "u_10", "u_6" }, 0, 1) == expect);
    ASSERT_TRUE(maskedKeyOf2("in2", { "u_9", "u_10", "u_6" }, 1, 0) == expect);  // order
    ASSERT_TRUE(maskedKeyOf2("in2", { "u_9", "u_10", "u_7" }, 0, 1) != expect);  // fixed arg
    ASSERT_TRUE(maskedKeyOf2("in2", { "1", "2", "u_6" }, 0, 2) != expect);       // slot set
    ASSERT_TRUE(maskedKeyOf("in2", { "1", "u_10", "u_6" }, 0) != expect);        // one slot
    ASSERT_TRUE(maskedKeyOf("in2", { "1", "u_10", "u_6" }, 0)
                != maskedKeyOf2("in2", { "1", "u_10", "u_6" }, 0, 1));
}

// One masked family holds one-slot and two-slot keys side by side: each probe
// answers exactly the chains of its key, and the answers are disjoint.
TEST(originals_head_index, masked_family_holds_one_and_two_slot_keys) {
    gl::GlobalMemoryManager g;
    g.init(kHeadCfg);
    gl::LbArena lb(&g);
    gl::OriginalsHeadIndex idx(&lb);

    const std::vector<std::string> heads = { "in2", "in2", "in2", "in2" };
    const std::string one = maskedKeyOf("in2", { "1", "u_10", "u_6" }, 0);
    const std::string two = maskedKeyOf2("in2", { "1", "2", "u_6" }, 0, 1);
    const std::vector<std::string> masked = { one, two, "", two };
    buildMasked(idx, heads, masked);
    ASSERT_EQ(idx.maskedKeyCount(), int32_t(2));
    ASSERT_TRUE(indexedMasked(idx, maskedKeyOf("in2", { "u_9", "u_10", "u_6" }, 0))
                == std::vector<int32_t>({ 1 }));
    ASSERT_TRUE(indexedMasked(idx, maskedKeyOf2("in2", { "u_9", "u_10", "u_6" }, 0, 1))
                == std::vector<int32_t>({ 2, 4 }));
    ASSERT_TRUE(indexedMasked(idx, maskedKeyOf2("in2", { "u_9", "u_10", "u_6" }, 0, 2)).empty());
}

// maskedHeadKey carries the sign: a negated expression's key is its positive
// key behind a `!`, so the same name, arguments and slot under opposite signs
// never share a key, while equal signs still do.
TEST(originals_head_index, masked_head_key_carries_the_sign) {
    std::string negatedKey = "!in,";
    negatedKey += gl::kMaskedHeadSlotByte;
    negatedKey += ",u_5";
    ASSERT_TRUE(maskedKeyOf("in", { "1", "u_5" }, 0, true) == negatedKey);
    ASSERT_TRUE(maskedKeyOf("in", { "1", "u_5" }, 0, true)
                == "!" + maskedKeyOf("in", { "1", "u_5" }, 0));
    ASSERT_TRUE(maskedKeyOf("in", { "u_x", "u_5" }, 0, false)
                != maskedKeyOf("in", { "1", "u_5" }, 0, true));          // sign
    ASSERT_TRUE(maskedKeyOf("in", { "u_x", "u_5" }, 0, true)
                == maskedKeyOf("in", { "1", "u_5" }, 0, true));          // both negated
    ASSERT_TRUE(maskedKeyOf("in", { "u_x", "u_5" }, 0, false)
                == maskedKeyOf("in", { "1", "u_5" }, 0, false));         // both positive
}

// The masked family answers a probe only with chains of the probe's sign: a
// positive and a negated head over the same arguments land under two keys.
TEST(originals_head_index, masked_family_separates_the_signs) {
    gl::GlobalMemoryManager g;
    g.init(kHeadCfg);
    gl::LbArena lb(&g);
    gl::OriginalsHeadIndex idx(&lb);

    const std::vector<std::string> heads = { "in", "in", "in", "in" };
    const std::string pos = maskedKeyOf("in", { "1", "u_5" }, 0, false);
    const std::string neg = maskedKeyOf("in", { "1", "u_5" }, 0, true);
    const std::vector<std::string> masked = { pos, neg, neg, pos };
    buildMasked(idx, heads, masked);
    ASSERT_EQ(idx.maskedKeyCount(), int32_t(2));
    ASSERT_TRUE(indexedMasked(idx, maskedKeyOf("in", { "u_x", "u_5" }, 0, false))
                == std::vector<int32_t>({ 1, 4 }));
    ASSERT_TRUE(indexedMasked(idx, maskedKeyOf("in", { "u_x", "u_5" }, 0, true))
                == std::vector<int32_t>({ 2, 3 }));
    // the head family stays sign-free: every chain carries the name
    ASSERT_TRUE(indexed(idx, "in") == std::vector<int32_t>({ 1, 2, 3, 4 }));
}

// The masked family answers exactly the chains carrying a key, ascending;
// chains without a key are absent; the append-only extension and clear cover
// the family like the head family.
TEST(originals_head_index, masked_family_matches_brute_force_and_extends) {
    gl::GlobalMemoryManager g;
    g.init(kHeadCfg);
    gl::LbArena lb(&g);
    gl::OriginalsHeadIndex idx(&lb);

    std::vector<std::string> heads = sampleHeads();
    const std::string kA = maskedKeyOf("in3", { "u_1", "5", "u_3" }, 1);
    const std::string kB = maskedKeyOf("in3", { "u_1", "u_2", "7" }, 2);
    const std::string kC = maskedKeyOf("in2", { "3", "u_2" }, 0);
    std::vector<std::string> masked = { kA, "", kB, "", kC, kA, "", "", kA, "", "", kB };
    buildMasked(idx, heads, masked);
    ASSERT_TRUE(idx.built());
    ASSERT_EQ(idx.maskedKeyCount(), int32_t(3));
    for (const std::string& k : { kA, kB, kC })
        ASSERT_TRUE(indexedMasked(idx, k) == bruteForceMasked(masked, k));
    ASSERT_TRUE(indexedMasked(idx, kA) == std::vector<int32_t>({ 1, 6, 9 }));
    // a miss: a key no chain carries; the empty key is never a hit
    ASSERT_TRUE(indexedMasked(idx, maskedKeyOf("in3", { "u_1", "5", "u_4" }, 1)).empty());
    ASSERT_TRUE(indexedMasked(idx, "").empty());
    // the head family is untouched by the masked keys
    ASSERT_TRUE(indexed(idx, "in3") == bruteForce(heads, "in3"));

    // append-only extension: new chains land behind the old ones per key
    heads.push_back("in3");  masked.push_back(kA);
    heads.push_back("in3");  masked.push_back("");
    heads.push_back("in2");  masked.push_back(kC);
    buildMasked(idx, heads, masked);
    ASSERT_EQ(idx.builtCount(), static_cast<int32_t>(heads.size()));
    ASSERT_TRUE(indexedMasked(idx, kA) == std::vector<int32_t>({ 1, 6, 9, 13 }));
    ASSERT_TRUE(indexedMasked(idx, kC) == std::vector<int32_t>({ 5, 15 }));

    // clear forgets the masked family too; a rebuild indexes it again
    idx.clear();
    ASSERT_EQ(idx.maskedKeyCount(), int32_t(0));
    buildMasked(idx, heads, masked);
    ASSERT_TRUE(indexedMasked(idx, kB) == bruteForceMasked(masked, kB));
}

TEST(originals_head_index, membership_matches_brute_force_in_ascending_order) {
    gl::GlobalMemoryManager g;
    g.init(kHeadCfg);
    gl::LbArena lb(&g);
    gl::OriginalsHeadIndex idx(&lb);
    ASSERT_FALSE(idx.built());
    ASSERT_EQ(idx.builtCount(), int32_t(0));

    const std::vector<std::string> heads = sampleHeads();
    build(idx, heads);
    ASSERT_TRUE(idx.built());
    ASSERT_EQ(idx.builtCount(), static_cast<int32_t>(heads.size()));
    ASSERT_EQ(idx.operatorCount(), int32_t(6));

    for (const std::string& op : { "in3", "preorder", "=", "in2", "existence16", "or11" }) {
        ASSERT_TRUE(indexed(idx, op) == bruteForce(heads, op));
    }
    // A miss: an operator that never headed a chain yields nothing.
    ASSERT_TRUE(indexed(idx, "strictOrder").empty());
    // A hash-neighbourhood miss: a prefix of a stored name is not a hit.
    ASSERT_TRUE(indexed(idx, "in").empty());
}

TEST(originals_head_index, append_only_extension_indexes_only_the_new_chains) {
    gl::GlobalMemoryManager g;
    g.init(kHeadCfg);
    gl::LbArena lb(&g);
    gl::OriginalsHeadIndex idx(&lb);

    std::vector<std::string> heads = sampleHeads();
    build(idx, heads);
    const int32_t first = idx.builtCount();

    // Grow the chain table (the staging flush appends keys after the old
    // count) and extend: the appended ids land behind the old ones, per
    // operator, in ascending order; a new operator gets its own entry.
    heads.push_back("in3");
    heads.push_back("strictOrder");
    heads.push_back("preorder");
    heads.push_back("strictOrder");
    build(idx, heads);
    ASSERT_EQ(idx.builtCount(), static_cast<int32_t>(heads.size()));
    ASSERT_EQ(idx.builtCount(), first + 4);
    ASSERT_EQ(idx.operatorCount(), int32_t(7));
    for (const std::string& op : { "in3", "preorder", "strictOrder", "=", "in2", "existence16", "or11" }) {
        ASSERT_TRUE(indexed(idx, op) == bruteForce(heads, op));
    }

    // A same-count call is a no-op.
    build(idx, heads);
    ASSERT_EQ(idx.builtCount(), static_cast<int32_t>(heads.size()));
    ASSERT_TRUE(indexed(idx, "in3") == bruteForce(heads, "in3"));
}

TEST(originals_head_index, mark_unbuilt_rebuilds_from_the_first_chain) {
    gl::GlobalMemoryManager g;
    g.init(kHeadCfg);
    gl::LbArena lb(&g);
    gl::OriginalsHeadIndex idx(&lb);

    std::vector<std::string> heads = sampleHeads();
    build(idx, heads);

    // The reload door: the pages may or may not survive, the next ensureBuilt
    // rebuilds from chain 1 against the CURRENT table.
    idx.markUnbuilt();
    ASSERT_FALSE(idx.built());
    ASSERT_EQ(idx.builtCount(), int32_t(0));

    heads[0] = "preorder";          // the table a canonical reload would present
    heads.push_back("in3");
    build(idx, heads);
    ASSERT_TRUE(idx.built());
    ASSERT_EQ(idx.builtCount(), static_cast<int32_t>(heads.size()));
    for (const std::string& op : { "in3", "preorder", "=", "in2", "existence16", "or11" }) {
        ASSERT_TRUE(indexed(idx, op) == bruteForce(heads, op));
    }
    // No stale node from the first build survives: "in3" lost chain 1.
    ASSERT_EQ(indexed(idx, "in3").front(), int32_t(3));
}

TEST(originals_head_index, clear_frees_and_forgets_the_build) {
    gl::GlobalMemoryManager g;
    g.init(kHeadCfg);
    gl::LbArena lb(&g);
    gl::OriginalsHeadIndex idx(&lb);

    const std::vector<std::string> heads = sampleHeads();
    build(idx, heads);
    ASSERT_TRUE(idx.built());

    // The teardown seam (HashMemory release / clear, canonical release).
    idx.clear();
    ASSERT_FALSE(idx.built());
    ASSERT_EQ(idx.builtCount(), int32_t(0));
    ASSERT_EQ(idx.operatorCount(), int32_t(0));

    // An empty table builds to an empty, probe-able index.
    const std::vector<std::string> none;
    build(idx, none);
    ASSERT_TRUE(idx.built());
    ASSERT_TRUE(indexed(idx, "in3").empty());

    // And a fresh table after that indexes normally.
    build(idx, heads);
    ASSERT_TRUE(indexed(idx, "in3") == bruteForce(heads, "in3"));
}
