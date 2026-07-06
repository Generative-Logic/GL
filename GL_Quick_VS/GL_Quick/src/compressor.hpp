/* Generative Logic : A deterministic reasoning and knowledge generation engine.
 Copyright(C) 2025-2026 Generative Logic UG(haftungsbeschraenkt)

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

    /// @brief Per-LB proof graph extracted from the prover's hash bursts —
    /// the input to the compressor's redundancy elimination pass.
    ///
    /// @details
    /// One `CompressorNode` per LB the compressor wants to consider. After
    /// the prover finishes, each LB's `exprOriginMap` is walked into a
    /// `(expression → list of alternative dependency-lists)` shape; that
    /// shape lives in `graph`. Premises are the expressions loaded as
    /// fuel (always alive), `head` is the proof goal of this LB, and
    /// `originalTheorem` carries the raw MPL text of the theorem the LB
    /// proves.
    ///
    /// The compressor's redundancy pass uses `isDerivable` to test, for
    /// each candidate "dead" theorem set, whether `head` is still
    /// reachable from the surviving theorems + premises through the graph.
    /// Theorems that can be killed without breaking any head's
    /// derivability are dropped; survivors form the output set.
    ///
    /// @see [`Compressor`](#compressor) — owns a `std::vector<CompressorNode>`.
    /// @see `compressor.cpp::isDerivable` — derivability oracle.
    struct CompressorNode {
        // expression -> list of alternative dep lists
        std::map<ExpressionWithValidity,
                 std::vector<std::vector<ExpressionWithValidity>>> graph;
        std::set<ExpressionWithValidity> premises;
        ExpressionWithValidity head;
        std::string originalTheorem;
    };

    /// @brief Post-proof redundancy eliminator — drops theorems that are
    /// derivable from the rest.
    ///
    /// @details
    /// The compressor runs after the main prover stage on the full set of
    /// proved theorems. For each theorem it runs the prover one more time
    /// (Phase 1) to extract a `CompressorNode` whose graph captures every
    /// alternative derivation path of the theorem's head. Phase 2 then
    /// performs greedy multi-pass redundancy elimination: a theorem is
    /// "dead" if it can be removed without breaking any other theorem's
    /// derivability. Survivors form the output set, which becomes the
    /// authoritative `files/theorems/theorems.txt` content — the
    /// regression-claim source of truth for the run.
    ///
    /// Determinism is non-negotiable: `std::stable_sort` orders the
    /// theorem list before the elimination pass so the kill order is
    /// reproducible across runs (see OPEN-13 in the SwDD).
    ///
    /// @see [`CompressorNode`](#compressornode) — per-LB graph type.
    /// @see `docs/agentic_swdd/10_pipeline/05_compressor.md` — full pipeline-stage
    ///      chapter.
    class Compressor {
    public:
        /// @brief Construct a compressor over an existing
        /// `ExpressionAnalyzer` and the full theorem list.
        /// @param analyzer    Prover instance that provides the hash-engine
        ///                    state for Phase 1.
        /// @param all_theorems Full set of proved theorems (raw MPL).
        Compressor(ExpressionAnalyzer& analyzer,
               const std::vector<std::string>& all_theorems);

        /// @brief Run Phase 1 (graph extraction) + Phase 2 (greedy
        /// redundancy elimination).
        /// @return The essential (surviving) theorem strings, in
        ///         deterministic stable order.
        std::vector<std::string> run();

    private:
        ExpressionAnalyzer&            analyzer;
        std::vector<std::string>       all_theorems;
        std::vector<CompressorNode>    extracted_graphs;  // N graphs

        // Maps compact form back to original expanded form from globalTheoremList
        std::unordered_map<std::string, std::string> compactToExpanded;

        /// @brief Phase 1 — extract per-LB proof graphs by running the
        /// prover one more time and walking each LB's exprOriginMap.
        /// @details Populates `extracted_graphs` with one
        ///          [`CompressorNode`](#compressornode) per theorem.
        void runPhase1();

        /// @brief Phase 2 — greedy multi-pass redundancy elimination.
        /// @details Each pass walks the theorem list in stable order,
        ///          tests each theorem for being killable (i.e. all
        ///          heads still derivable without it), and kills
        ///          survivors that pass. Repeats until a fixed point.
        /// @return Surviving theorem texts.
        std::vector<std::string> runPhase2();

        /// @brief Forward-reachability check — true iff `node.head` is
        /// reachable from its premises + theorems NOT in `dead_theorems`
        /// through `node.graph`.
        /// @details Walks alternative dependency lists; a head is
        ///          reachable if any alternative's deps are all
        ///          reachable. Used by Phase 2 to verify that a
        ///          candidate kill set leaves every theorem provable.
        bool isDerivable(const CompressorNode& node,
                         const std::set<std::string>& dead_theorems) const;
    };

} // namespace gl
