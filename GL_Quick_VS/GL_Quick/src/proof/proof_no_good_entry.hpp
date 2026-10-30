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
/// @brief One cached failure: why it happened and what it depended on.
///
/// @details
/// `conflict` holds the on-path nodes that blocked the frame. `consulted`
/// holds every dependency the failed exploration tested against `covered`,
/// including those its children tested; `coveredAtFailure` is the subset of
/// `consulted` that was already covered when the frame gave up. Both
/// conditions below are one-directional, which is what makes the entry
/// reusable rather than exact:
///
///   * extra path members only remove candidates (a non-cyclic candidate can
///     become cyclic, never the reverse), so failure survives a longer path
///     as long as every recorded blocker is still on it;
///   * extra covered members only remove recursions (a failing child becomes
///     a skip), which could turn failure into success - so no consulted node
///     may be covered now that was not covered then.
///
/// With both satisfied the exploration replays identically, because every
/// `covered.insert` verdict and every cyclicity verdict it read is unchanged.
struct NoGoodEntry {
    std::set<ExpressionWithValidity> conflict;
    std::set<ExpressionWithValidity> consulted;
    std::set<ExpressionWithValidity> coveredAtFailure;
};
}
