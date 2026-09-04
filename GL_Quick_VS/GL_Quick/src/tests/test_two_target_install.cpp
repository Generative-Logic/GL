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

/// @file test_two_target_install.cpp
/// @brief Differential test of the two-target rule install: one
///        `addToHashMemory` build writing `overallHashMemory` and
///        `workingMemory` (the incubator's status-3 form) produces the same
///        bytes in every rule-index map as two sequential single-target
///        installs (D-333, idea D).

#include "../prover.hpp"
#include "test_harness.hpp"

#include <cstring>
#include <string>
#include <vector>

namespace {

    // A key's bytes: a byte-string key by length + bytes, a scalar key (the
    // admission map's int64) by its raw bytes.
    void putKey(std::vector<char>& out, const gl::StrSpan& k) {
        const int32_t len = k.len;
        const char* lp = reinterpret_cast<const char*>(&len);
        out.insert(out.end(), lp, lp + 4);
        out.insert(out.end(), k.ptr, k.ptr + k.len);
    }
    template <class Scalar>
    void putKey(std::vector<char>& out, const Scalar& k) {
        const char* p = reinterpret_cast<const char*>(&k);
        out.insert(out.end(), p, p + sizeof(Scalar));
    }

    // Byte image of a typed cold blob map's inner map: the key count, every
    // key in id order, every run's blobs. Equal images = identical.
    template <class Typed>
    std::vector<char> imageOf(const Typed& typed) {
        const auto& m = typed.inner();
        std::vector<char> out;
        const auto put32 = [&](int32_t v) {
            const char* p = reinterpret_cast<const char*>(&v);
            out.insert(out.end(), p, p + 4);
        };
        put32(m.count());
        for (int32_t id = 1; id <= m.count(); ++id) {
            putKey(out, m.keyAt(id));
            const int32_t rl = m.runLen(id);
            put32(rl);
            for (int32_t j = 0; j < rl; ++j) {
                std::vector<char> b;
                m.blobAt(id, j, b);
                put32(static_cast<int32_t>(b.size()));
                out.insert(out.end(), b.begin(), b.end());
            }
        }
        return out;
    }

    // Every rule-index map of one instance, in a fixed order.
    std::vector<std::vector<char>> imagesOf(const gl::HashMemory& hm) {
        std::vector<std::vector<char>> v;
        v.push_back(imageOf(hm.encodedMap));
        v.push_back(imageOf(hm.normalizedEncodedKeys));
        v.push_back(imageOf(hm.normalizedEncodedSubkeys));
        v.push_back(imageOf(hm.remainingArgsNormalizedEncodedMap));
        v.push_back(imageOf(hm.remainingArgsOwners));
        v.push_back(imageOf(hm.originals));
        v.push_back(imageOf(hm.copyOwners));
        v.push_back(imageOf(hm.admissionMap));
        return v;
    }

    // A two-premise rule with a u_ argument: whole keys, two subkey prefixes,
    // remaining args, admission variants and copy owners all get records.
    struct Rule {
        std::string chain0 = "(in3[1,2,3,u_4])";
        std::string chain1 = "(=[2,3])";
        std::string head = "(=[1,3])";
        std::string impl = "(>[1,2,3](in3[1,2,3,u_4])(=[2,3])(=[1,3]))";
    };

    void install(gl::ExpressionAnalyzer& ea, const Rule& r,
                 gl::HashMemory& target, gl::HashMemory* second) {
        const int lv[1] = { 0 };
        const gl::StrSpan chainRun[2] = { gl::StrSpan(r.chain0), gl::StrSpan(r.chain1) };
        ea.addToHashMemory(chainRun, 2, gl::StrSpan(r.head), nullptr, 0,
            ea.body, target, lv, 1, gl::StrSpan(r.impl),
            ea.parameters.maxIterationNumberVariable,
            ea.parameters.standardMaxSecondaryNumber, false,
            ea.parameters.minNumOperatorsKey, gl::StrSpan("implication", 11),
            true, gl::StrSpan(r.impl), gl::StrSpan("main", 4), gl::RuleIndexOp{}, second);
    }

} // namespace

TEST(two_target_install, one_build_matches_two_sequential_installs) {
    const Rule r;

    gl::ExpressionAnalyzer one(std::string("IncubatorGauss3"));
    install(one, r, one.body.overallHashMemory, &one.body.workingMemory);

    gl::ExpressionAnalyzer two(std::string("IncubatorGauss3"));
    install(two, r, two.body.overallHashMemory, nullptr);
    install(two, r, two.body.workingMemory, nullptr);

    // Something was installed on both sides.
    ASSERT_TRUE(one.body.overallHashMemory.encodedMap.count() > 0);
    ASSERT_TRUE(one.body.workingMemory.encodedMap.count() > 0);

    const auto oneOverall = imagesOf(one.body.overallHashMemory);
    const auto twoOverall = imagesOf(two.body.overallHashMemory);
    const auto oneWorking = imagesOf(one.body.workingMemory);
    const auto twoWorking = imagesOf(two.body.workingMemory);
    ASSERT_EQ(static_cast<int>(oneOverall.size()), 8);
    for (std::size_t i = 0; i < oneOverall.size(); ++i) {
        ASSERT_TRUE(oneOverall[i] == twoOverall[i]);
        ASSERT_TRUE(oneWorking[i] == twoWorking[i]);
    }
    // The second target received the same records as the first.
    for (std::size_t i = 0; i < 7; ++i) ASSERT_TRUE(oneOverall[i] == oneWorking[i]);
    ASSERT_EQ(one.body.workingMemory.maxKeyLength, one.body.overallHashMemory.maxKeyLength);

    // Both installs closed their windows: nothing staged on the seam.
    ASSERT_TRUE(gl::ruleStagings().slotIsEmpty(gl::ExpressionAnalyzer::currentGenSlot()));
}
