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
#include "../compiler/compact_expression_syntax.hpp"
#include "../proof/or_candidate_progress.hpp"
#include "test_harness.hpp"
#include <thread>

TEST(easy_cycles, compact_syntax_preserves_numbered_positive_prefixes) {
    using gl::CompactExpressionSyntax;
    const std::string implication = "(implication12[a])";
    const std::string positiveOr = "(or0[a,b])";
    ASSERT_TRUE(CompactExpressionSyntax::isCompactImplicationSpan(gl::StrSpan(implication)));
    ASSERT_TRUE(CompactExpressionSyntax::isOrCompactSpan(gl::StrSpan(positiveOr)));
    for (const std::string text : {"", "(implication", "(implication[a])",
            "!(implication12[a])", "(primeseq[a])", "(or0[a,b])"}) {
        ASSERT_FALSE(CompactExpressionSyntax::isCompactImplicationSpan(gl::StrSpan(text)));
    }
    for (const std::string text : {"", "(or", "(or[a,b])", "!(or0[a,b])",
            "(order[a,b])", "(implication12[a])"}) {
        ASSERT_FALSE(CompactExpressionSyntax::isOrCompactSpan(gl::StrSpan(text)));
    }
}

TEST(easy_cycles, candidate_completion_matches_both_roles_without_publication) {
    std::vector<gl::OrCandidate> candidates{
        {"existence", "companion"}, {"shared", "shared"}, {"other", "companion"}};
    std::mutex mutex;
    gl::OrCandidateProgress progress(candidates, mutex);
    progress.checkOrCompletion("missing", -1);
    for (const auto& candidate : candidates) {
        ASSERT_FALSE(candidate.existenceProved);
        ASSERT_FALSE(candidate.companionProved);
    }
    progress.checkOrCompletion("existence", 0);
    ASSERT_TRUE(candidates[0].existenceProved);
    ASSERT_FALSE(candidates[0].companionProved);
    progress.checkOrCompletion("companion", 1);
    ASSERT_TRUE(candidates[0].companionProved);
    ASSERT_TRUE(candidates[2].companionProved);
    ASSERT_FALSE(candidates[2].existenceProved);
    progress.checkOrCompletion("shared", 2);
    progress.checkOrCompletion("shared", 2);
    ASSERT_TRUE(candidates[1].existenceProved);
    ASSERT_TRUE(candidates[1].companionProved);
    ASSERT_EQ(candidates.size(), size_t(3));
}

TEST(easy_cycles, candidate_completion_serializes_concurrent_halves) {
    std::vector<gl::OrCandidate> candidates{{"left", "right"}};
    std::mutex mutex;
    gl::OrCandidateProgress progress(candidates, mutex);
    std::thread left([&] { progress.checkOrCompletion("left", 0); });
    std::thread right([&] { progress.checkOrCompletion("right", 1); });
    left.join();
    right.join();
    ASSERT_TRUE(candidates[0].existenceProved);
    ASSERT_TRUE(candidates[0].companionProved);
}
