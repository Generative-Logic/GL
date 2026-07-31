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

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

/// @file
/// @brief Counterexample (CE) filter — declarations shared with the prover
/// translation unit.
///
/// @details
/// Method bodies for the CE filter live in `filter.cpp`; they are member
/// functions of `ExpressionAnalyzer` (declared in `prover.hpp`). This header
/// is the home for CE-only types and free helpers that should not pollute
/// `prover.hpp` / `memory.hpp`.
///
/// The CE filter is the *peek-and-prune* stage of the pipeline: every
/// candidate conjecture is loaded into a synthetic LB seeded with the simple
/// facts (`files/simple_facts/simple_facts_<anchor>_<n>.txt`), and the prover
/// is run for a small budget (`numberIterationsConjectureFiltering`) to see
/// whether the conjecture contradicts the fact base. A confirmed
/// contradiction marks the conjecture `successful` (i.e. *refuted*) in
/// `contradictionTable[i]`; the filter then DROPS it — only rows with
/// `successful == false` survive into `filtered_conjectures.txt` for the
/// main prover stage.
///
/// @see [`docs/agentic_swdd/10_pipeline/03_ce_filter.md`](../../docs/agentic_swdd/10_pipeline/03_ce_filter.md)
///      for the full pipeline-stage chapter.

namespace gl {

    /// @brief One row of the CE-filter contradiction table.
    ///
    /// @details
    /// `ExpressionAnalyzer` holds a `std::vector<ContradictionItem>`
    /// (`contradictionTable`) keyed by conjecture index. Per-row `successful`
    /// is set `true` when the prover detects that the conjecture contradicts
    /// the simple-fact base — `successful == true` means REFUTED; the filter
    /// keeps only `!successful` rows. CE-only — referenced from `filter.cpp`
    /// and from the contradiction-handler touch-point in `prover.cpp`.
    ///
    /// @see `prover.hpp::contradictionTable`.
    struct ContradictionItem {
        std::string expr;
        bool successful;
        ContradictionItem()
            : expr(),
            successful(false){
        }
        ContradictionItem(const std::string& expr_,
            int successful_)
            : expr(expr_),
            successful(successful_) {
        }
    };

    /// @brief Symmetric mirror-partner index: pool conjecture string → its
    ///        mirror partner strings, both directions inserted.
    using MirrorPartnerMap =
        std::unordered_map<std::string, std::vector<std::string>>;

    /// @brief Load `mirror_pairs.txt` into a symmetric partner map.
    ///
    /// @details
    /// Reads the conjecturer-written pairs artifact (one
    /// `source<TAB>mirror` row per operator-only conjecture whose mirror
    /// entered the pool; both columns byte-identical to `conjectures.txt`
    /// lines) and inserts BOTH directions — `source → mirror` and
    /// `mirror → source` — with per-key duplicate suppression, so the CE
    /// filter's flip pass can follow the relation from whichever member a
    /// counterexample refutes. Rows are read in file order (source-sorted by
    /// the writer), keeping each key's partner vector deterministic.
    /// Trailing `\r` is stripped; empty lines are skipped (an empty file is
    /// a defined result: no pairs, no flips).
    ///
    /// The file's existence is part of the pipeline contract: the
    /// conjecturer writes it unconditionally whenever `conjectures.txt` is
    /// written, so absence means the pipeline was invoked out of order —
    /// asserted, not tolerated.
    ///
    /// @param path Full path to `files/theorems/mirror_pairs.txt`.
    /// @return Symmetric partner map; empty when the file has no rows.
    /// @see [D-229](../../docs/agentic_swdd/40_decisions.md#d-229)
    MirrorPartnerMap loadMirrorPairs(const std::filesystem::path& path);

    /// @brief Flip the mirror partners of CE-refuted conjectures to refuted.
    ///
    /// @details
    /// The mirror-refutation heuristic's flip pass. Snapshots which slots the
    /// CE bursts refuted (`table[i].successful` seeds), then sweeps them in
    /// ascending index order: for every seeded conjecture, each partner from
    /// `partners` that is present in this batch has its slot's `successful`
    /// set `true`. Only CE-confirmed refutations seed — a flipped slot never
    /// seeds further flips (no cascading) — so the outcome is a pure function
    /// of the CE verdicts plus the pairs file, independent of sweep order. A
    /// partner absent from `conjectures` is a defined case (refuted or
    /// flipped in an earlier fact-file pass, or deduplicated at generation),
    /// not a failure.
    ///
    /// Refutation-side only: this marks conjectures as dropped pre-prover; no
    /// mirror ever re-enters as a proof step
    /// ([I-81](../../docs/agentic_swdd/30_invariants.md#i-81) / D-112).
    ///
    /// @param table       The CE contradiction table, one slot per
    ///                    conjecture; flipped in place.
    /// @param conjectures The batch's conjecture strings, index-aligned with
    ///                    `table`.
    /// @param partners    Symmetric partner map from `loadMirrorPairs`.
    /// @return Number of slots flipped by this pass.
    /// @see loadMirrorPairs — builds `partners`.
    int applyMirrorRefutations(std::vector<ContradictionItem>& table,
        const std::vector<std::string>& conjectures,
        const MirrorPartnerMap& partners);

} // namespace gl
