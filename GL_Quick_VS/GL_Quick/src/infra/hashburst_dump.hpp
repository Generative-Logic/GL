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

#include "memory.hpp"

#include <map>
#include <string>

namespace gl {

    /// @brief Hashburst trace dump infrastructure.
    ///
    /// The dump is targeted at one specific LB chain — currently
    /// `__contradiction__(=[7,10])` under `(AnchorIncubator[1..14])` —
    /// and writes a structured snapshot of
    /// every per-LB container on every hashburst ENTRY / EXIT trap
    /// fire. The dump additionally emits the
    /// reshuffle-introduced containers (`workingMemory`,
    /// `externalStatements`, `intExternalStatements`) via
    /// `writeReshuffleContainers`. The output file path is fixed at
    /// `.debug/hashburst_trace.txt`.
    ///
    /// @note **Rule 14 (the perform-elem hashburst dump is
    ///       sacred).** Every part of this infrastructure — the section
    ///       names, the call sites, the file path, the target LB chain —
    ///       requires explicit user approval to change. The dedicated
    ///       translation unit hosting this dump IS such an
    ///       approved arrangement.
    namespace hashburst_dump {

        /// @brief Predicate matching the target LB chain.
        ///
        /// Returns `true` iff `body.exprKey == "__contradiction__(=[7,10])"`,
        /// its parent is
        /// `(AnchorIncubator[1,2,3,4,5,6,7,8,9,10,11,12,13,14])`, and the
        /// grandparent is the root sentinel (empty `exprKey`,
        /// `parentMemory == nullptr`). Per Rule 12 the full chain to
        /// root is matched, not just `body.exprKey`.
        bool isTargetLB(const Memory& body);

        /// @brief Dump at hashburst ENTRY.
        ///
        /// On the first ENTRY for the targeted LB, the trace file is
        /// truncated; on every subsequent ENTRY (and at EXIT) it is
        /// appended. On the first ENTRY the
        /// `compiledExpressions` map is dumped once as a one-shot
        /// reference; later ENTRIES skip that section.
        ///
        /// @param body                 The targeted LB.
        /// @param compiledExpressions  Caller's `ExpressionAnalyzer::compiledExpressions`,
        ///                             passed in because the dump lives in
        ///                             a separate translation unit and
        ///                             cannot reach the analyzer member
        ///                             directly.
        void dumpEntry(const Memory& body,
                       const CompiledExpressionMap& compiledExpressions);

        /// @brief Dump at the final EXIT.
        void dumpExit(const Memory& body);

    } // namespace hashburst_dump

} // namespace gl
