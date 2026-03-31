/* Generative Logic : A deterministic reasoning and knowledge generation engine.
 Copyright(C) 2025 Generative Logic UG(haftungsbeschraenkt)

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
#include "prover.hpp"
#include <string>
#include <vector>
#include <set>
#include <map>

namespace gl {

    // ---------------------------------------------------------------
    // Per-LB proof graph extracted after hash bursts.
    //
    //   graph    — for each expression, all alternative derivation paths
    //              (each path = list of dependency expressions).
    //   premises — expressions loaded as fuel (always alive).
    //   head     — the proof goal of this LB.
    // ---------------------------------------------------------------
    struct CompressorNode {
        // expression -> list of alternative dep lists
        std::map<ExpressionWithValidity,
                 std::vector<std::vector<ExpressionWithValidity>>> graph;
        std::set<ExpressionWithValidity> premises;
        ExpressionWithValidity head;
        std::string originalTheorem;
    };

    // ---------------------------------------------------------------
    // Compressor — same public interface as before.
    // ---------------------------------------------------------------
    class Compressor {
    public:
        Compressor(ExpressionAnalyzer& analyzer,
               const std::vector<std::string>& all_theorems);

        // Run the full compressor pipeline (Phase 1 + Phase 2).
        // Returns the essential (surviving) theorem strings.
        std::vector<std::string> run();

    private:
        ExpressionAnalyzer&            analyzer;
        std::vector<std::string>       all_theorems;
        std::vector<CompressorNode>    extracted_graphs;  // N graphs

        // Maps compact form back to original expanded form from globalTheoremList
        std::unordered_map<std::string, std::string> compactToExpanded;

        // Phase 1 — build per-LB proof graphs via prover hash bursts.
        void runPhase1();

        // Phase 2 — greedy elimination with multi-pass.
        std::vector<std::string> runPhase2();

        // Forward-reachability derivability check.
        // Returns true iff the head of `node` is reachable from
        // its premises + all theorems NOT in `dead_theorems`.
        bool isDerivable(const CompressorNode& node,
                         const std::set<std::string>& dead_theorems) const;
    };

} // namespace gl
