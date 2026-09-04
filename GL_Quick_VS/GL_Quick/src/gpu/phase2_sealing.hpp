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

#include "phase2_cuda.hpp"
#include "phase2_projection.hpp"
#include "../memory.hpp"

#include <cstdint>
#include <vector>

namespace gl::gpu {

    /// @brief Fixed processor staging for device-decided Phase 2 output.
    ///
    /// @details
    /// Construction allocates every download and temporary sealing array to the
    /// evaluator's immutable capacity. `downloadAndSeal` reuses those addresses,
    /// decodes projected identifiers, and copies already-decided record content
    /// into ordinary `FiringRecord` page chains. It performs no rule evaluation,
    /// semantic gate, canonical sort, or proof-state mutation.
    class Phase2SealingArena {
    public:
        /// @brief Allocate all processor download and sealing scratch once.
        ///
        /// @details
        /// Sizes one vector for every evaluator output column, one canonical
        /// index vector, and reusable string/dependency scratch large enough for
        /// any single retained record. No later method changes these sizes or
        /// capacities.
        ///
        /// @param fixedCapacity Immutable evaluator element and byte ceilings.
        /// @return An empty reusable sealing arena.
        /// @invariant Every owned address remains stable for the object lifetime.
        explicit Phase2SealingArena(Phase2EvaluationCapacity fixedCapacity);

        Phase2SealingArena(const Phase2SealingArena&) = delete;
        Phase2SealingArena& operator=(const Phase2SealingArena&) = delete;
        Phase2SealingArena(Phase2SealingArena&&) = delete;
        Phase2SealingArena& operator=(Phase2SealingArena&&) = delete;

        /// @brief Download canonical retained GPU records and seal them by block.
        ///
        /// @details
        /// Downloads the exact used evaluator prefixes plus the doom-compacted
        /// canonical index permutation. Each retained header is translated to an
        /// ordinary `FiringRecord` by byte-copying generated slices, decoding
        /// local NameMap and rule-interner identifiers from the same immutable
        /// projection, and preserving all device verdict flags. Output page sets
        /// receive records in device canonical order and are sealed before return.
        /// No request, substitution, scope, admission, mail, doom, or provenance
        /// decision is recomputed on the processor.
        ///
        /// @param evaluation Device evaluator containing the completed sweep.
        /// @param projection Host twin of the projection uploaded for that sweep.
        /// @param used Exact materialization counts returned by the evaluator.
        /// @param retainedFiringCount Exact count returned by doom selection.
        /// @param outputs One bound, filling `SealedPageSet` per logical block.
        /// @param outputCount Number of output sets; exactly the projected block
        ///                    count.
        /// @return Number of `FiringRecord`s appended across all output sets.
        /// @invariant The evaluator and projection describe the same sweep and
        ///            retained indices are grouped by logical-block index.
        uint32_t downloadAndSeal(
            const CudaPhase2EvaluationBuffer& evaluation,
            const Phase2ProjectionArena& projection,
            const Phase2FiringExpressionResult& used,
            uint32_t retainedFiringCount,
            SealedPageSet* const* outputs,
            uint32_t outputCount);

    private:
        Phase2EvaluationCapacity capacity_{};
        std::vector<DevicePhase2FiringRecord> firingRecords_;
        std::vector<char> generatedBytes_;
        std::vector<int32_t> levelValues_;
        std::vector<DeviceEvaluationDependency> originDependencies_;
        std::vector<DeviceEvaluationByteSlice> markerKeys_;
        std::vector<int32_t> markerRemainingArgs_;
        std::vector<DeviceEvaluationByteSlice> markerArgs_;
        std::vector<uint32_t> firingOrder_;
        std::vector<SealedString> sealedStringsScratch_;
        std::vector<SealedExpressionWithValidity> sealedDependenciesScratch_;
    };

} // namespace gl::gpu
