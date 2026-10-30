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
#include "theorem_usage_graph.hpp"
#include <cassert>
namespace gl {

/// @brief Drop every edge — fresh graph for a new export run.
///
/// @details
/// Called once at `generateRawProofGraph` entry (next to
/// `clearBuildStackPath`) so consecutive exports (main batch after
/// incubator batch in one process) never see stale cross-run edges.
void TheoremUsageGraph::clear() { edges.clear(); }

/// @brief Commit one written chapter's citation edges into the graph.
///
/// @details
/// Scans the emitted rows of the chapter of `theorem` and records
/// `theorem -> cited` for every citation channel: a `theorem` row cites its
/// own row expression; `reformulated from` and `incubator back
/// reformulation` rows cite their source at `row[3]`; an `or theorem` row
/// cites its two source implications at `row[3]` and `row[5]`; an
/// `or elimination` row cites its two guard variants at `row[3]`/`row[5]`
/// and its licensing or theorem at `row[7]`. Row arity is
/// asserted per channel — a malformed row is a producer bug to stop on,
/// never to skip. Induction triads call this once per chapter file; the
/// three files' edges union under the one theorem node.
///
/// @param theorem    The theorem whose chapter was just written
///                   (global-theorem-list string form).
/// @param stackRows  The chapter's emitted rows (cells per row: expression,
///                   namespace, tag, then (dependency, namespace) pairs).
void TheoremUsageGraph::addEdgesFromStack(
    const std::string& theorem,
    const std::vector<std::vector<std::string>>& stackRows)
{
    std::set<std::string>& out = edges[theorem];
    for (const std::vector<std::string>& row : stackRows) {
        assert(row.size() >= 3
            && "TheoremUsageGraph: chapter row must carry expression, namespace, tag");
        const std::string& tag = row[2];
        if (tag == "theorem") {
            out.insert(row[0]);
        } else if (tag == "reformulated from"
                || tag == "incubator back reformulation") {
            assert(row.size() >= 5
                && "TheoremUsageGraph: source citation at row[3] required");
            out.insert(row[3]);
        } else if (tag == "or theorem") {
            assert(row.size() >= 7
                && "TheoremUsageGraph: or-theorem sources at row[3]/row[5] required");
            out.insert(row[3]);
            out.insert(row[5]);
        } else if (tag == "or elimination") {
            assert(row.size() >= 9
                && "TheoremUsageGraph: or-elimination sources at row[3]/row[5]/row[7] required");
            out.insert(row[3]);
            out.insert(row[5]);
            out.insert(row[7]);
        }
    }
}

/// @brief Reachability probe: does `from` reach `to` over committed edges?
///
/// @details
/// Iterative depth-first search over `edges`; `from == to` returns true
/// without a walk (a self-citation is a length-1 cycle). Theorems without
/// committed out-edges are leaves. Used by `buildStack` as the
/// cross-chapter admissibility probe: a `theorem` origin citing U is
/// rejected while building the chapter of T iff `reaches(U, T)`.
///
/// @param from  Candidate cited theorem (start node).
/// @param to    Theorem whose chapter is currently being built.
/// @return True iff `to` is reachable from `from` (including `from == to`).
bool TheoremUsageGraph::reaches(const std::string& from,
                                const std::string& to) const
{
    if (from == to) return true;
    std::set<std::string> visited;
    std::vector<const std::string*> frontier;
    visited.insert(from);
    frontier.push_back(&from);
    while (!frontier.empty()) {
        const std::string* cur = frontier.back();
        frontier.pop_back();
        const auto it = edges.find(*cur);
        if (it == edges.end()) continue;
        for (const std::string& next : it->second) {
            if (next == to) return true;
            if (visited.insert(next).second) frontier.push_back(&next);
        }
    }
    return false;
}

}
