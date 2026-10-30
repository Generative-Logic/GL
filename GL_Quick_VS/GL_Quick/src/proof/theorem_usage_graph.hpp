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
#pragma once
#include <map>
#include <set>
#include <string>
#include <vector>
namespace gl {
namespace tests { struct TheoremUsageGraphFixture; }











    /// @brief Directed usage graph over proven theorems, maintained by the
    ///        chapter export to keep inter-chapter theorem citations acyclic.
    ///
    /// @details
    /// One instance lives on `Prover` (`exportTheoremUsage`) for
    /// the duration of one `generateRawProofGraph` run. Nodes are theorem
    /// strings exactly as they appear in the global theorem list; the edge
    /// T -> U records that a written chapter or a fixed synthetic chapter of
    /// theorem T cites theorem U — via a `theorem` row or a synthetic-chapter source citation
    /// (`reformulated from`, `incubator back reformulation`, `or theorem`).
    ///
    /// `buildStack` consults the graph through `reaches` before accepting a
    /// `theorem`-tagged origin candidate: citing U while building the chapter
    /// of T is admissible only when U does not already reach T in the
    /// committed graph (U == T included). An inadmissible candidate would
    /// close a cross-chapter cycle that every per-chapter check misses —
    /// within each chapter a cited theorem is a foundation leaf, so two
    /// chapters can otherwise ground themselves in each other (the
    /// `theorem usage termination` verifier check flags exactly this; see
    /// docs/agentic_swdd/10_pipeline/08_verifier.md).
    ///
    /// Fixed synthetic citations are seeded before any proof walk. Chosen
    /// proof edges commit after their chapter is written. Every edge starts
    /// at its own theorem, so a later fixed reformulation cannot silently
    /// close a cycle behind the admissibility probe. Backtracking needs no
    /// edge rollback.
    ///
    /// Post-prove export code on the sequential chapter-export loop:
    /// single-threaded by contract, heap containers are the norm here (the
    /// statified-kernel rules do not bind the export path).
    ///
    /// @invariant The committed graph is acyclic at every point in time.
    /// @invariant `cachedProofStacks` currently has no write site; if it is
    ///            ever repopulated, cached stacks bypass the buildStack
    ///            admissibility probe — the cache fill must then re-run the
    ///            probe over the cached rows (edge commitment itself stays
    ///            correct: it operates on emitted rows, however built).
    class TheoremUsageGraph {
friend struct tests::TheoremUsageGraphFixture;
private:
        /// Adjacency: theorem -> set of theorems its chapter cites.
        std::map<std::string, std::set<std::string>> edges;

        /// @brief Drop every edge — fresh graph for a new export run.
        ///
        /// @details
        /// Called once at `generateRawProofGraph` entry (next to
        /// `clearBuildStackPath`) so consecutive exports (main batch after
        /// incubator batch in one process) never see stale cross-run edges.
    public:
        void clear();

        /// @brief Commit one written chapter's citation edges into the graph.
        ///
        /// @details
        /// Scans the emitted rows of the chapter of `theorem` and records
        /// `theorem -> cited` for every citation channel: a `theorem` row
        /// cites its own row expression; `reformulated from` and
        /// `incubator back reformulation` rows cite their source at
        /// `row[3]`; an `or theorem` row cites its two source implications
        /// at `row[3]` and `row[5]`. Row arity is asserted per channel —
        /// a malformed row is a producer bug to stop on, never to skip.
        /// Induction triads call this once per chapter file; the three
        /// files' edges union under the one theorem node.
        ///
        /// @param theorem    The theorem whose chapter was just written
        ///                   (global-theorem-list string form).
        /// @param stackRows  The chapter's emitted rows (cells per row:
        ///                   expression, namespace, tag, then
        ///                   (dependency, namespace) pairs).
        void addEdgesFromStack(const std::string& theorem,
                               const std::vector<std::vector<std::string>>& stackRows);

        /// @brief Reachability probe: does `from` reach `to` over committed edges?
        ///
        /// @details
        /// Iterative depth-first search over `edges`; `from == to` returns
        /// true without a walk (a self-citation is a length-1 cycle).
        /// Theorems without committed out-edges are leaves. Used by
        /// `buildStack` as the cross-chapter admissibility probe: a
        /// `theorem` origin citing U is rejected while building the chapter
        /// of T iff `reaches(U, T)`.
        ///
        /// @param from  Candidate cited theorem (start node).
        /// @param to    Theorem whose chapter is currently being built.
        /// @return True iff `to` is reachable from `from` (including
        ///         `from == to`).
        bool reaches(const std::string& from, const std::string& to) const;
    };
}
