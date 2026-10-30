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
#include "../memory_infra/temporary_witness_store.hpp"
#include "equivalence_span_pair.hpp"
namespace gl {
namespace tests { struct RuntimeSupportFixture; }

        // Per-mapping result emitted by enumerateEqClassRewrites, consumed by
        // the closure's staging sink:
        //   - rewrittenExpr: wrapLeft + baseExpr + "[" + ",".join(tempList) + "]" + wrapRight
        //   - setEqualities: justifying (from,to) span pairs (only under
        //     trackHistory), ascending by (from,to), duplicates collapsed
        //   - extraLevels  : union of clss per-pair levels across substituted
        //     positions, ascending-unique
        // setEqualities / extraLevels point at loop-local stack buffers inside
        // enumerateEqClassRewrites, live for the duration of the sink call — no
        // heap set. rewrittenExpr is a fresh string built on the per-slot
        // string-tier scratch arena (scratchArenas), spanned only at the
        // synchronous sink(r) call — never crosses the per-mapping ScratchScope
        // rewind (09c Pitfall 4); the sink interns it into a DIFFERENT interner.
        struct EqClassRewrite {
            StrSpan rewrittenExpr;
            const EqClassSpanPair* setEqualities = nullptr;
            int32_t setEqualityCount = 0;
            const int* extraLevels = nullptr;
            int32_t extraLevelCount = 0;
            NameId argIds[ExecutionParameters::MAX_ARITY] = {};   // the product's per-position ids
            int32_t argCount = 0;
        };
}
