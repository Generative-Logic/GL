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

#include <cstdint>

namespace gl::gpu {

    // The Phase 2 usage records: plain counts, free of every other GPU type, so
    // a logical block's shell (`Memory`) can hold the measurement the CUDA route
    // records at the end of the block's phase 1.

    /// @brief Safe element and byte bounds for one complete Phase 2 read image.
    ///
    /// @details
    /// Counts the pointer-free columns required by request generation and request
    /// evaluation without allocating or serializing them. Byte-map figures cover
    /// all four normalized whole-key/subkey registries plus the overall encoded
    /// and remaining-argument maps. Pod-map figures cover statement membership,
    /// level runs, validity filters, recursion products, mail filters, and goals.
    /// Hash-slot counts use the projection's fixed maximum load of one half. The
    /// remaining-argument reverse map reports a no-dedup upper bound derived from
    /// its forward blobs; the builder may use a shorter prefix after coalescing
    /// duplicate normalized keys.
    struct Phase2ProjectionUsage {
        uint64_t logicalBlocks{ 0 };
        uint64_t statements{ 0 };
        uint64_t nameRecords{ 0 };
        uint64_t nameBytes{ 0 };
        uint64_t nameSlots{ 0 };
        uint64_t ruleStringRecords{ 0 };
        uint64_t ruleStringBytes{ 0 };
        uint64_t byteMapViews{ 0 };
        uint64_t byteMapEntries{ 0 };
        uint64_t byteMapSlots{ 0 };
        uint64_t byteKeyBytes{ 0 };
        uint64_t blobRecords{ 0 };
        uint64_t blobBytes{ 0 };
        uint64_t reverseMapViews{ 0 };
        uint64_t reverseMapEntries{ 0 };
        uint64_t reverseMapSlots{ 0 };
        uint64_t reverseKeyBytes{ 0 };
        uint64_t reverseOwners{ 0 };
        uint64_t podMapViews{ 0 };
        uint64_t podMapEntries{ 0 };
        uint64_t podMapSlots{ 0 };
        uint64_t podRunValues{ 0 };
        uint64_t mandatoryStatementKeys{ 0 };
        uint64_t metadataBytes{ 0 };
    };

    /// @brief Exact fixed-column usage of one projected executor part.
    struct Phase2TaskProjectionUsage {
        uint64_t tasks{ 0 };
        uint64_t batches{ 0 };
        uint64_t terms{ 0 };
        uint64_t stumps{ 0 };
    };

} // namespace gl::gpu
