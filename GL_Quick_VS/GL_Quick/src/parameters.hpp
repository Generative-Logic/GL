/* Generative Logic : A deterministic reasoning and knowledge generation engine.
 Copyright(C) 2025 Generative Logic UG(haftungsbeschr�nkt)

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
// Subset of parameters needed for quick mode scaffold.
// Values mirror GL/parameters.py (quick defaults).
namespace gl {

    struct ProverParameters {
        // Defaults from parameters.py / old parameters.hpp
        int sizeAllBinariesAna = 10;
        int maxIterationNumberProof = 30;
        int numberIterationsConjectureFiltering = 1;
        int maxSizeDefSetMapping = 5;
        int maxSizeTargetSetMapping = 12;
        int maxNumberSecondaryVariables = 2;
        int sizeAllPermutationsAna = 7;
        int minNumOperatorsKey = 2;
        int minNumOperatorsKeyCE = 4;
        int maxIterationNumberVariable = 1;
        int standardMaxSecondaryNumber = 1;
        bool trackHistory = true;
        int standardMaxAdmissionDepth = 0;
        int inductionMaxAdmissionDepth = 1;
        int inductionMaxSecondaryNumber = 2;
        int counterExampleBoundary = 6;
        int minLenLongKey = 5;
        int maxLenHypoKey = 2;
        bool debug = false;

        // --- Compressor Parameters ---
        bool compressor_mode = false;
        bool ban_disintegration = false;
        int max_origin_per_expr = 1;                 // cap used during normal prover runs
        int compressor_max_origins_per_expr = 30;    // cap used during compressor Phase 1 hash bursts
        int compressor_hash_bursts = 15;

        // --- Incubator Parameters ---
        bool try_contradiction = false;
        bool skip_ce_filter = false;
        bool skip_eq_classes = false;
        bool incubator_mode = false;

        // --- multiplyImplication ---
        int max_partition_size = 5;
    };

}