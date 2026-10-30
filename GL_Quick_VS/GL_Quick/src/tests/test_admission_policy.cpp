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
#include "test_harness.hpp"
#include "../logical_block/admission_policy.hpp"

TEST(admission_policy, reads_classic_thresholds_without_prover_state) {
    gl::CompiledDefinitionRegistry definitions;
    gl::ProverParameters parameters;
    parameters.minLenLongKey = 2;
    gl::AdmissionPolicy policy(definitions, parameters);
    const std::vector<std::string> fixed{"(p[u_1])", "(q[a])"};
    const std::vector<std::string> ordinary{"(p[a])", "(q[b])"};
    const std::vector<std::string> anchored{"(AnchorFixture[a])", "(p[u_1])"};
    const std::vector<std::string> shortKey{"(p[u_1])"};
    ASSERT_TRUE(policy.baselineClassicQualifies(fixed, "(r[a])", 99));
    ASSERT_FALSE(policy.baselineClassicQualifies(ordinary, "(r[a])", 99));
    ASSERT_FALSE(policy.baselineClassicQualifies(anchored, "(r[a])", 99));
    ASSERT_FALSE(policy.baselineClassicQualifies(shortKey, "(r[a])", 99));
    const gl::StrSpan fixedSpans[]{gl::StrSpan(fixed[0]), gl::StrSpan(fixed[1])};
    const gl::StrSpan ordinarySpans[]{gl::StrSpan(ordinary[0]), gl::StrSpan(ordinary[1])};
    ASSERT_TRUE(policy.baselineClassicQualifies(fixedSpans, 2, gl::StrSpan("(r[a])", 6), 99));
    ASSERT_FALSE(policy.baselineClassicQualifies(ordinarySpans, 2, gl::StrSpan("(r[a])", 6), 99));
    ASSERT_TRUE(gl::AdmissionPolicy::hasUPrefixInKey(fixedSpans, 2));
    ASSERT_FALSE(gl::AdmissionPolicy::hasUPrefixInKey(ordinarySpans, 2));
    ASSERT_TRUE(definitions.entries().empty());
    ASSERT_TRUE(definitions.configurations().empty());
}

TEST(admission_policy, resolves_configured_output_slots_through_read_only_registry) {
    gl::CompiledDefinitionRegistry definitions;
    ce::CoreExpressionMap configs;
    ce::CoreExpressionConfig map(3, std::string(), "(map[1,2,3])");
    map.inputArgs = {"1", "2"};
    map.outputArgs = {"3"};
    map.inputIndices = {0, 1};
    map.outputIndices = {2};
    configs["map"] = map;
    definitions.configure(configs);
    gl::ProverParameters parameters;
    gl::AdmissionPolicy policy(definitions, parameters);
    ASSERT_TRUE(policy.outputMatchesHeadOutputSlot(std::string("c"), std::string("(map[a,b,c])")));
    ASSERT_FALSE(policy.outputMatchesHeadOutputSlot(std::string("a"), std::string("(map[a,b,c])")));
    ASSERT_FALSE(policy.outputMatchesHeadOutputSlot(gl::StrSpan("c", 1), gl::StrSpan("(absent[a,b,c])", 15)));
    ASSERT_EQ(definitions.configurations().size(), std::size_t(1));
    ASSERT_TRUE(definitions.entries().empty());
}
