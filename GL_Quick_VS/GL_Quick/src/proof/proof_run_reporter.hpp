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
#include <chrono>
namespace gl {
class ProofRunReporter {
public:
/// @brief Bind the proof-run reporting inputs.
/// @details Borrows read-only views of the original block list and theorem records.
/// @param permanentBodies Address-stable blocks of this batch.
/// @param globalTheoremList Published theorem records.
/// @return Constructed reporting interface.
ProofRunReporter(const std::vector<Memory*>& permanentBodies, const std::vector<std::tuple<std::string, std::string, std::string, std::string>>& globalTheoremList);
/// @brief Print the existing completion and memory observations.
/// @details Samples completion time, checks block/witness accounting, writes the established output lines and resets the deload telemetry at the original run-end boundary.
/// @param initialBodyCount Block count recorded before proof execution.
/// @param initialWitnessCounterSum Witness counter sum recorded before execution.
/// @param t0 Existing proof-run start timestamp.
/// @return No value.
void reportCompletion(size_t initialBodyCount, int64_t initialWitnessCounterSum, std::chrono::high_resolution_clock::time_point t0);
private:
const std::vector<Memory*>& permanentBodies;
const std::vector<std::tuple<std::string, std::string, std::string, std::string>>& globalTheoremList;
}; }
