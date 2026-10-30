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

// No-good memo for the walker's backtracking search.
//
// A frame fails for exactly three reasons: every candidate was cyclic (a
// dependency sits on `g_buildStackPath`), inadmissible (the cross-chapter
// usage graph), or had a failing subtree. Only the cyclic reason is a pure
// function of the path, so only a failure reached without touching the other
// two is reproducible from the path alone. Two things therefore make a frame
// CONTEXT-DEPENDENT and bar it from the memo: a candidate skipped as
// inadmissible (the usage graph gains edges after every chapter, and
// `exportCurrentChapterTheorem` moves with the chapter). Dependence on
// `covered` is not a bar but a recorded condition - see NoGoodEntry. Every
// other failing frame caches its conflict set: the on-path nodes that actually
// blocked it, unioned with the conflict sets of the children that failed under
// it. Re-entry with all of those still on the path, and with no consulted
// dependency newly covered, returns false immediately instead of re-walking
// the subtree.
//
// Without this the walker re-derives every failing subtree under each of the
// parent candidates that reach it, which is exponential in the number of
// dead ends.
struct WalkNode {
    const Memory* lb;
    ExpressionWithValidity ev;
    bool operator<(const WalkNode& o) const {
        if (lb != o.lb) return lb < o.lb;
        return ev < o.ev;
    }
};
}
