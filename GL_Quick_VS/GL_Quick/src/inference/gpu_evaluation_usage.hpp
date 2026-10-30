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
#include "../memory.hpp"
namespace gl {

        // Observation-only GPU request-evaluation capacity census. Each
        // executor task resets its thread-local record, BurstSink and
        // checkLocalEncodedMemoryStatic add the work and output shape they
        // actually encounter, and proveKernel publishes totals and maxima after
        // the task returns. No proof branch reads these fields.
        struct GpuEvaluationUsage {
            uint64_t requests{ 0 };
            uint64_t dependencyPassRequests{ 0 };
            uint64_t reverseOwners{ 0 };
            uint64_t candidateOwners{ 0 };
            uint64_t encodedHits{ 0 };
            uint64_t localValues{ 0 };
            uint64_t headRecords{ 0 };
            uint64_t markerRecords{ 0 };
            uint64_t demandRecords{ 0 };
            uint64_t generatedBytes{ 0 };
            uint64_t levelValues{ 0 };
            uint64_t originDependencies{ 0 };
            uint64_t markerKeys{ 0 };
            uint64_t markerRemainingArgs{ 0 };
            uint64_t markerArgs{ 0 };
            uint64_t maximumReverseOwnersPerRequest{ 0 };
            uint64_t maximumCandidateOwnersPerRequest{ 0 };
            uint64_t maximumLocalValuesPerHit{ 0 };
        };
}
