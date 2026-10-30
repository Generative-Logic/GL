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
#include "../proof/proof_reconstructor.hpp"
#include <utility>
namespace gl::tests {
struct ProofReconstructorFixture {
static bool pathContains(const ExpressionWithValidity& goal) { return ProofReconstructor::g_buildStackPath.count(goal) != 0; }
static void seedTraversalState(const ExpressionWithValidity& goal) { ProofReconstructor::g_coveredJournal.push_back(goal); ProofReconstructor::g_noGood[{nullptr,goal}] = {}; }
static bool traversalStateEmpty() { return ProofReconstructor::g_coveredJournal.empty() && ProofReconstructor::g_noGood.empty(); }

template<class Owner, class... Args> static decltype(auto) contradictionLbHoldsRecord(Owner& target, Args&&... args) { return target.contradictionLbHoldsRecord(std::forward<Args>(args)...); }
static std::vector<ExpressionWithValidity> originScopeAlternatives(const Memory& mb, const ExpressionWithValidity& v) { return ProofReconstructor::originScopeAlternatives(mb, v); }
static ExpressionWithValidity liftToShallowestOriginAncestor(const Memory& mb, const ExpressionWithValidity& v) { return ProofReconstructor::liftToShallowestOriginAncestor(mb, v); }
};
}
