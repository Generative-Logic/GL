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
/// @brief Unit tests for compressor graph values and reachability.
///
/// @details
/// `Compressor::run` requires a fully-constructed `Prover`
/// (loads config files) and runs the prover end-to-end on a set of
/// theorems, which belongs to the full-pipeline gate. These tests cover graph
/// values and compare the indexed reachability oracle against an independent
/// string-set fixed point, including cycles, scope distinctions, repeated
/// dependencies, unconditional alternatives, and removed theorem seeds.

#include "test_harness.hpp"

#include "../compressor.hpp"

#include <algorithm>

TEST(compressor, indexed_reachability_preserves_scope_and_duplicate_dependencies) {
    const gl::ExpressionWithValidity a("A", "main");
    const gl::ExpressionWithValidity b("B", "main");
    const gl::ExpressionWithValidity scopedA("A", "main_boundary_child");
    gl::CompressorNode node;
    node.head = b;
    node.graph[b] = {{a, a}, {scopedA}};
    node.graph[a] = {{b}};
    const gl::CompressorReachability indexed(node, {"A", "unused"});
    ASSERT_TRUE(indexed.isDerivable({}));
    ASSERT_FALSE(indexed.isDerivable({"A"}));
    ASSERT_TRUE(indexed.isDerivable({"unused"}));
    ASSERT_FALSE(indexed.isDerivable({"A", "unused"}));

    node.premises.insert(scopedA);
    const gl::CompressorReachability scoped(node, {"A"});
    ASSERT_TRUE(scoped.isDerivable({"A"}));

    node.premises.clear();
    node.graph[a].push_back({});
    const gl::CompressorReachability unconditional(node, {"A"});
    ASSERT_TRUE(unconditional.isDerivable({"A"}));
}

TEST(compressor, indexed_reachability_handles_heads_outside_graph_and_removed_seeds) {
    gl::CompressorNode node;
    node.head = gl::ExpressionWithValidity("A", "main");
    const gl::CompressorReachability theoremHead(node, {"A", "A", "irrelevant"});
    ASSERT_TRUE(theoremHead.isDerivable({}));
    ASSERT_FALSE(theoremHead.isDerivable({"A"}));

    node.premises.insert(node.head);
    const gl::CompressorReachability premiseHead(node, {});
    ASSERT_TRUE(premiseHead.isDerivable({"A"}));

    node.premises.clear();
    node.head.validityName = "main_boundary_child";
    const gl::CompressorReachability scopedHead(node, {"A"});
    ASSERT_FALSE(scopedHead.isDerivable({}));

    const gl::ExpressionWithValidity theorem("B", "main");
    node.graph[node.head] = {{theorem}};
    node.graph[theorem] = {{gl::ExpressionWithValidity("C", "main")}};
    const gl::CompressorReachability derivedRemoved(node, {"B", "C"});
    ASSERT_TRUE(derivedRemoved.isDerivable({"B"}));
    ASSERT_FALSE(derivedRemoved.isDerivable({"B", "C"}));
}

TEST(compressor, indexed_reachability_matches_full_scan_on_cyclic_graphs) {
    // Independent oracle: the original string-set least-fixed-point algorithm.
    const auto reference = [](const gl::CompressorNode& node,
                              const std::vector<std::string>& theorems,
                              const std::set<std::string>& removed) {
        std::set<gl::ExpressionWithValidity> alive = node.premises;
        for (const auto& theorem : theorems) {
            if (removed.count(theorem) == 0)
                alive.emplace(theorem, "main");
        }
        bool changed = true;
        while (changed) {
            changed = false;
            for (const auto& row : node.graph) {
                for (const auto& dependencies : row.second) {
                    bool reached = true;
                    for (const auto& dependency : dependencies)
                        reached = reached && alive.count(dependency) != 0;
                    if (reached && alive.insert(row.first).second) changed = true;
                }
            }
        }
        return alive.count(node.head) != 0;
    };
    const std::vector<std::string> theorems{"A", "B", "C", "unused"};
    std::vector<gl::ExpressionWithValidity> terms;
    for (const auto& name : {"A", "B", "C", "D"}) {
        terms.emplace_back(name, "main");
        terms.emplace_back(name, "main_boundary_child");
    }
    uint32_t random = 47291;
    const auto next = [&]() { random = random * 1664525u + 1013904223u; return random; };
    for (int sample = 0; sample < 256; ++sample) {
        gl::CompressorNode node;
        node.head = terms[(next() >> 16) % terms.size()];
        for (const auto& term : terms) {
            if ((next() >> 16) % 5 == 0) node.premises.insert(term);
            const uint32_t alternatives = (next() >> 16) % 4;
            for (uint32_t alternative = 0; alternative < alternatives; ++alternative) {
                std::vector<gl::ExpressionWithValidity> dependencies;
                const uint32_t count = (next() >> 16) % 4;
                for (uint32_t dependency = 0; dependency < count; ++dependency)
                    dependencies.push_back(terms[(next() >> 16) % terms.size()]);
                node.graph[term].push_back(dependencies);
            }
        }
        const gl::CompressorReachability indexed(node, theorems);
        for (unsigned mask = 0; mask < 16; ++mask) {
            std::set<std::string> removed;
            for (std::size_t i = 0; i < theorems.size(); ++i)
                if ((mask & (1u << i)) != 0) removed.insert(theorems[i]);
            ASSERT_EQ(indexed.isDerivable(removed), reference(node, theorems, removed));
        }
    }
}

TEST(compressor, compressornode_default_is_empty) {
    gl::CompressorNode n;
    ASSERT_EQ(n.graph.size(),    static_cast<std::size_t>(0));
    ASSERT_EQ(n.premises.size(), static_cast<std::size_t>(0));
    ASSERT_TRUE(n.head.original.empty());
    ASSERT_TRUE(n.head.validityName.empty());
    ASSERT_TRUE(n.originalTheorem.empty());
}

TEST(compressor, compressornode_populated) {
    gl::CompressorNode n;
    n.head = gl::ExpressionWithValidity("(p[a])", "main");
    n.premises.insert(gl::ExpressionWithValidity("(q[a])", "main"));
    n.premises.insert(gl::ExpressionWithValidity("(r[a])", "main"));
    // One alternative dep list for head.
    n.graph[n.head] = {{
        gl::ExpressionWithValidity("(q[a])", "main"),
        gl::ExpressionWithValidity("(r[a])", "main"),
    }};
    n.originalTheorem = "(>[a](q[a])(r[a])(p[a]))";

    ASSERT_EQ(n.premises.size(),         static_cast<std::size_t>(2));
    ASSERT_EQ(n.graph.size(),            static_cast<std::size_t>(1));
    ASSERT_EQ(n.graph.at(n.head).size(), static_cast<std::size_t>(1));
    ASSERT_EQ(n.graph.at(n.head)[0].size(),
              static_cast<std::size_t>(2));
    ASSERT_EQ(n.head.original,           std::string("(p[a])"));
}

// std::stable_sort determinism — OPEN-13 in docs/agentic_swdd/SwDD.md says the
// compressor relies on stable sort for reproducible kill order. Pin
// the stability property on a synthetic equal-key input.
TEST(compressor, stable_sort_preserves_input_order_for_equal_keys) {
    // Pairs of (key, original_index). After stable_sort on key, equal
    // keys must keep their original_index order.
    std::vector<std::pair<int, int>> v;
    v.emplace_back(2, 0);
    v.emplace_back(1, 1);
    v.emplace_back(2, 2);
    v.emplace_back(1, 3);
    v.emplace_back(2, 4);

    std::stable_sort(v.begin(), v.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });

    // Two key=1 entries (original indices 1, 3) come first in original order.
    ASSERT_EQ(v[0].first, 1);  ASSERT_EQ(v[0].second, 1);
    ASSERT_EQ(v[1].first, 1);  ASSERT_EQ(v[1].second, 3);
    // Three key=2 entries (original indices 0, 2, 4) keep original order.
    ASSERT_EQ(v[2].first, 2);  ASSERT_EQ(v[2].second, 0);
    ASSERT_EQ(v[3].first, 2);  ASSERT_EQ(v[3].second, 2);
    ASSERT_EQ(v[4].first, 2);  ASSERT_EQ(v[4].second, 4);
}

// =============================================================================
// Round 2 — two more tests per documented function.
// =============================================================================

// ---------- CompressorNode default ----------
TEST(compressor, compressornode_head_default_is_empty_evw) {
    gl::CompressorNode n;
    // The default-constructed head is an empty ExpressionWithValidity.
    ASSERT_TRUE(n.head.original.empty());
    ASSERT_TRUE(n.head.validityName.empty());
    ASSERT_TRUE(n.head == gl::ExpressionWithValidity());
}

TEST(compressor, compressornode_default_in_vector) {
    // Vector of default nodes — must be safe to resize and iterate.
    std::vector<gl::CompressorNode> v;
    v.resize(3);
    for (const auto& n : v) {
        ASSERT_EQ(n.graph.size(),    static_cast<std::size_t>(0));
        ASSERT_EQ(n.premises.size(), static_cast<std::size_t>(0));
    }
}

// ---------- CompressorNode populated ----------
TEST(compressor, compressornode_multiple_alternatives_per_head) {
    gl::CompressorNode n;
    n.head = gl::ExpressionWithValidity("(p[a])", "main");
    // Two alternative dependency lists for the same head.
    n.graph[n.head] = {
        { gl::ExpressionWithValidity("(q[a])", "main") },
        { gl::ExpressionWithValidity("(r[a])", "main"),
          gl::ExpressionWithValidity("(s[a])", "main") },
    };
    ASSERT_EQ(n.graph[n.head].size(),       static_cast<std::size_t>(2));
    ASSERT_EQ(n.graph[n.head][0].size(),    static_cast<std::size_t>(1));
    ASSERT_EQ(n.graph[n.head][1].size(),    static_cast<std::size_t>(2));
}

TEST(compressor, compressornode_premises_carry_validity) {
    gl::CompressorNode n;
    n.premises.insert(gl::ExpressionWithValidity("(p[a])", "main"));
    n.premises.insert(gl::ExpressionWithValidity("(p[a])",
                                                 "main_boundary_x"));
    // Same expression in two different scopes is two distinct premises.
    ASSERT_EQ(n.premises.size(), static_cast<std::size_t>(2));
}

// ---------- std::stable_sort property ----------
TEST(compressor, stable_sort_empty_range_is_no_op) {
    std::vector<int> v;
    std::stable_sort(v.begin(), v.end());
    ASSERT_EQ(v.size(), static_cast<std::size_t>(0));
}

TEST(compressor, stable_sort_single_element_unchanged) {
    std::vector<int> v{42};
    std::stable_sort(v.begin(), v.end());
    ASSERT_EQ(v.size(),  static_cast<std::size_t>(1));
    ASSERT_EQ(v[0],      42);
}
