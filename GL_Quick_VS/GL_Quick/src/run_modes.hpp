/* Generative Logic : A deterministic reasoning and knowledge generation engine.
 Copyright(C) 2025-2026 Generative Logic UG(haftungsbeschr�nkt)

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
#include <vector>
#include <string>

namespace run_modes {

    // Full batch entry point (conjecture load -> CE filter -> prover ->
    // compression -> raw proof graph). `Config<anchor_id>.json` drives the
    // folder layout via `theorems_folder` / `raw_proof_graph_folder`.
    //
    // The two optional overrides serve the shortcut mode
    // (`gl_quick.exe <Tag> --conjectures-file <path> --externals-file <path>`):
    // when non-empty they replace, respectively, the conjecture-list file
    // (default `<theorems_folder>/conjectures.txt`) and the file external
    // theorems are loaded from (default
    // `<theorems_folder>/compressed_external_theorems.txt`). Relative paths
    // resolve against the project root. Empty string = no override.
    /**
     * @brief Run one configured proof batch through its complete native stages.
     *
     * @details The configured theorem pool provides prior results. The
     * development shortcut uses its independent pool and raw graph directories;
     * explicit input files are resolved after the directory selection.
     * @param anchor_id Configuration suffix selecting the batch.
     * @param conjecturesFileOverride Optional explicit conjecture input file.
     * @param externalsFileOverride Optional explicit external theorem input file.
     * @param phase2Backend Explicit processor or CUDA selection, or configured default.
     * @param shortcutMode Select the independent FTA development output tree.
     * @return None.
     */
    void fullRun(const std::string& anchor_id,
                 const std::string& conjecturesFileOverride = std::string(),
                 const std::string& externalsFileOverride = std::string(),
                 const std::string& phase2Backend = std::string(),
                 bool shortcutMode = false);

    // CE-filter-only entry point. Performs the same config + theorem-loading
    // setup as `fullRun` but runs only the counterexample filter and exits
    // after writing `filtered_conjectures.txt`. Compression, proof-graph
    // emission, and the main prover are all skipped.
    //
    // Used by the `--ce-only <Tag>` CLI flag for the fast-iteration
    // filter-design workflow (peano_filter_design_decisions.md D-rtf-1).
    void ceOnlyRun(const std::string& anchor_id);
}
