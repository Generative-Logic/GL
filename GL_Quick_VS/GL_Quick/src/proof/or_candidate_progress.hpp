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
#include "or_candidate.hpp"
#include <vector>
#include <mutex>
#include <string>

namespace gl {

/// @brief Marks existing OR candidates as their required theorems become known.
/// @details Storage, allocation and invocation order remain with callers.
class OrCandidateProgress {
public:

/// @brief Bind the candidate collection and its synchronization boundary.
/// @details The collection and mutex remain owned by the proof coordinator and
/// must outlive this view. Construction does not inspect or change candidates.
/// @param candidates Existing candidate collection.
/// @param mutex Mutex protecting candidate completion updates.
/// @return Constructed view over the supplied collection.
OrCandidateProgress(std::vector<OrCandidate>& candidates, std::mutex& mutex)
    : orCandidates(candidates), orCandidatesMutex(mutex) {}

/// @brief Mark every candidate role matching a newly proved theorem.
/// @details Holds the existing candidate mutex for the ordered scan. A theorem
/// can complete both roles of a candidate; unrelated candidates remain unchanged.
/// No theorem is emitted and no candidate is removed here.
/// @param provedTheorem Exact theorem spelling to match.
/// @param coreId Caller worker identity; completion marking does not use it.
/// @return Nothing; updates matching completion flags in place.
void checkOrCompletion(const std::string& provedTheorem, int coreId);
private:
std::vector<OrCandidate>& orCandidates;
std::mutex& orCandidatesMutex;

};


}
