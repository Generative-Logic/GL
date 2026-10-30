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

#include <cstddef>
#include <cstdint>
#include <iosfwd>
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
        /// Returns `true` iff `body.exprKey()` and every `parentMemory`
        /// `exprKey()` up to the root sentinel (empty `exprKey`,
        /// `parentMemory == nullptr`) equal the chain written in the
        /// definition. Per Rule 12 the full chain to root is matched, not
        /// just `body.exprKey`; the chain is retargeted per investigation
        /// (Rule 14), and the impossible innermost key `__no_dump_target__`
        /// switches the dump off.
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
        /// @param compiledExpressions  Caller's `Prover::compiledExpressions`,
        ///                             passed in because the dump lives in
        ///                             a separate translation unit and
        ///                             cannot reach the analyzer member
        ///                             directly.
        void dumpEntry(const Memory& body,
                       const CompiledExpressionMap& compiledExpressions);

        /// @brief Dump at the final EXIT.
        void dumpExit(const Memory& body);

        /// @brief Record the statement rows one Phase-2 projection sends to
        ///        the GPU for the target LB, and the rows the premise
        ///        patterns held back.
        ///
        /// @details
        /// Called by `Phase2ProjectionArena::appendLogicalBlock` for the
        /// target LB only (the caller gates on `isTargetLB`), once the
        /// block's upload slice is final. A row is SENT when its source index
        /// is in @p sentIndices; a row whose signed core is a premise core
        /// (@p coreUsedMask bit `1 << negation` at its core name id) but that
        /// was not sent was WITHHELD by the premise-pattern condition
        /// (`premisePatternAdmits`); every other row has no premise core and
        /// is only counted. The rows are decoded now (text and scope),
        /// because the statement list can be compacted before the EXIT dump,
        /// and buffered until the next `dumpExit` writes them through
        /// `writePhase2Uploads`. Thread-safe (projection shards run in
        /// parallel); its heap use is the Rule-14 dump island.
        ///
        /// @param body         The target LB.
        /// @param sentIndices  The uploaded rows' source indices, strictly
        ///                     ascending.
        /// @param sentCount    Their number.
        /// @param coreUsedMask Per core name id, bit `1 << negation` set when
        ///                     the signed core is a premise core.
        /// @param coreSlots    The mask's length (name count + 1).
        /// @return Nothing.
        /// @see `writePhase2Uploads`, `Phase2ProjectionArena::appendLogicalBlock`.
        void recordPhase2Upload(const Memory& body, const NameId* sentIndices,
                                uint32_t sentCount, const uint8_t* coreUsedMask,
                                std::size_t coreSlots);

        /// @brief Write the buffered Phase-2 uploads as the
        ///        `phase2.statementUpload` section and empty the buffer.
        ///
        /// @details
        /// The last section of every EXIT frame: a header line with the
        /// number of uploads since the previous EXIT and their sent and
        /// withheld rows, then per upload one summary line followed by its
        /// `sent` / `withheld` rows in source order (`[index] text | v=scope`,
        /// the `encodedStatements` row format). An EXIT without an upload (a
        /// processor-route burst) writes the header with zero counts.
        ///
        /// @param f Trace output stream.
        /// @return Nothing.
        /// @see `recordPhase2Upload`, `dumpExit`.
        void writePhase2Uploads(std::ofstream& f);

    } // namespace hashburst_dump

} // namespace gl
