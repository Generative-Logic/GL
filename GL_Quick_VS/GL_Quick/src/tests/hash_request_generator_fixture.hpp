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
#include "../inference/hash_request_generator.hpp"
#include <utility>
namespace gl::tests {
struct HashRequestGeneratorFixture {
static void seedDepth(int depth, uint64_t value) {
 HashRequestGenerator::g_gpuGrowAttemptsByDepth[depth] = value;
 HashRequestGenerator::g_gpuGrowFrontierByDepth[depth] = value;
 HashRequestGenerator::g_gpuGrowSubkeysByDepth[depth] = value;
 HashRequestGenerator::g_gpuGrowRequestsByDepth[depth] = value;
 HashRequestGenerator::g_gpuProducerAttemptsByDepth[depth] = value;
 HashRequestGenerator::g_gpuProducerSurvivorsByDepth[depth] = value;
}
static void seedFilters(HashRequestGenerator& target, uint64_t value) {
 target.gpuRequestFilterCalls.store(value);
 target.gpuProducerFilterCalls.store(value);
 target.gpuFilterInputStatements.store(value);
 target.gpuFilterOutputStatements.store(value);
 target.gpuFilterMaximumInputStatements.store(value);
 target.gpuFilterMaximumOutputStatements.store(value);
}
static void setGrowthMatchCount(int64_t value) { HashRequestGenerator::g_growthMatchCount = value; }
template<class... Args> static decltype(auto) filterIntEncodedStatements(HashRequestGenerator& target, Args&&... args) { return target.filterIntEncodedStatements(std::forward<Args>(args)...); }
template<class... Args> static decltype(auto) foldExprIntoRequestGates(Args&&... args) { return HashRequestGenerator::foldExprIntoRequestGates(std::forward<Args>(args)...); }
template<class... Args> static decltype(auto) foldPrefixIntoRequestGates(Args&&... args) { return HashRequestGenerator::foldPrefixIntoRequestGates(std::forward<Args>(args)...); }
template<class... Args> static decltype(auto) requestGateStateAccepts(HashRequestGenerator& target, Args&&... args) { return target.requestGateStateAccepts(std::forward<Args>(args)...); }
template<class... Args> static decltype(auto) requestGatesPass(HashRequestGenerator& target, Args&&... args) { return target.requestGatesPass(std::forward<Args>(args)...); }
template<class... Args> static decltype(auto) resetRequestGateState(Args&&... args) { return HashRequestGenerator::resetRequestGateState(std::forward<Args>(args)...); }
template<class... Args> static decltype(auto) subkeyUSatisfied(Args&&... args) { return HashRequestGenerator::subkeyUSatisfied(std::forward<Args>(args)...); }
};
}
