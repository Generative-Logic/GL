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

#include "infra/hashburst_dump.hpp"
#include "infra/hashburst_dump_reader.hpp"

#include <algorithm>
#include <cassert>
#include <fstream>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <string>
#include <vector>

namespace gl {
namespace hashburst_dump {

    // File-scope counters + mutexes — one per call site (ENTRY / EXIT),
    // mirroring the inline `static` declarations that once lived inside the
    // dump lambdas. Per Rule 14 they may not be reset / repurposed autonomously.
    namespace {
        std::mutex entryMtx;
        int entryCount = 0;

        std::mutex exitMtx;
        int exitCount = 0;

        constexpr const char* TRACE_PATH = ".debug/hashburst_trace.txt";

        // The target LB's Phase-2 uploads since its last EXIT dump
        // (recordPhase2Upload), written and emptied by the next EXIT
        // (writePhase2Uploads).
        std::mutex uploadMtx;
        std::vector<std::string> uploadLines;
        int uploadCount = 0;
        std::size_t uploadSent = 0;
        std::size_t uploadWithheld = 0;

    } // anonymous namespace

    bool isTargetLB(const Memory& body) {
        // No dump target: a measured full run (Rule 14). The chain shape is
        // the goal block of the FTA lemmas "{(0,v)} is a sequence over
        // [0,0]" and "its fold is v": (EnumerationRelationSet1[2,9,10]) under
        // (in[9,1]), the anchor and the root (Rule 12); an investigation sets
        // the innermost key to the block under scrutiny.
        const Memory* p1 = body.parentMemory;
        const Memory* p2 = p1 ? p1->parentMemory : nullptr;
        const Memory* p3 = p2 ? p2->parentMemory : nullptr;
        return equalSpans(body.exprKeyView(), StrSpan("__no_dump_target__", sizeof("__no_dump_target__") - 1))
            && p1 && equalSpans(p1->exprKeyView(), StrSpan("(in[9,1])", sizeof("(in[9,1])") - 1))
            && p2 && equalSpans(p2->exprKeyView(), StrSpan("(AnchorFTA[1,2,3,4,5,6,7,8])", sizeof("(AnchorFTA[1,2,3,4,5,6,7,8])") - 1))
            && p3 && p3->exprKeyView().len == 0 && p3->parentMemory == nullptr;
    }

    void dumpEntry(const Memory& body,
                   const CompiledExpressionMap& compiledExpressions) {
        std::lock_guard<std::mutex> lock(entryMtx);
        ++entryCount;
        std::ofstream f(TRACE_PATH,
                        entryCount == 1 ? std::ios::trunc : std::ios::app);
        HashburstDumpReader::writeHeader(f, body, "ENTRY", entryCount);
        HashburstDumpReader::writeLBChain(f, body);
        HashburstDumpReader::writeEncodedStatements(f, body);
        HashburstDumpReader::writeToBeProved(f, body);
        HashburstDumpReader::writeExprOriginMap(f, body);
        HashburstDumpReader::writeHashOriginals(f, body);
        HashburstDumpReader::writeAdmissionMap(f, body);
        HashburstDumpReader::writeEncodedMapMarkers(f, body);
        HashburstDumpReader::writeMailIn(f, body);
        if (entryCount == 1) HashburstDumpReader::writeCompiledExpressions(f, compiledExpressions);
        HashburstDumpReader::writeExtendedSections(f, body);
        f.flush();
    }

    void dumpExit(const Memory& body) {
        std::lock_guard<std::mutex> lock(exitMtx);
        ++exitCount;
        std::ofstream f(TRACE_PATH, std::ios::app);
        HashburstDumpReader::writeHeader(f, body, "EXIT", exitCount);
        HashburstDumpReader::writeLBChain(f, body);
        HashburstDumpReader::writeEncodedStatements(f, body);
        HashburstDumpReader::writeToBeProved(f, body);
        HashburstDumpReader::writeExprOriginMap(f, body);
        HashburstDumpReader::writeHashOriginals(f, body);
        HashburstDumpReader::writeAdmissionMap(f, body);
        HashburstDumpReader::writeEncodedMapMarkers(f, body);
        HashburstDumpReader::writeMailIn(f, body);
        HashburstDumpReader::writeExtendedSections(f, body);
        writePhase2Uploads(f);
        f.flush();
    }

    /// @brief Record the statement rows one Phase-2 projection sends to the
    ///        GPU for the target LB, and the rows the premise patterns held
    ///        back.
    ///
    /// @details
    /// Walks the block's rows in source order: a row in @p sentIndices is
    /// sent (its signed core must be a premise core, asserted); any other
    /// row with a premise core was withheld by the premise-pattern
    /// condition; rows without a premise core are only counted. Every sent
    /// index must be a row of the block (asserted). The decoded rows are
    /// appended to the buffer `writePhase2Uploads` empties at the next EXIT.
    ///
    /// @param body         The target LB.
    /// @param sentIndices  The uploaded rows' source indices, strictly
    ///                     ascending.
    /// @param sentCount    Their number.
    /// @param coreUsedMask Per core name id, bit `1 << negation` set when the
    ///                     signed core is a premise core.
    /// @param coreSlots    The mask's length (name count + 1).
    /// @return Nothing.
    /// @see `writePhase2Uploads`, `Phase2ProjectionArena::appendLogicalBlock`.
    void recordPhase2Upload(const Memory& body, const NameId* sentIndices,
                            uint32_t sentCount, const uint8_t* coreUsedMask,
                            std::size_t coreSlots) {
        std::lock_guard<std::mutex> lock(uploadMtx);
        ++uploadCount;
        const std::size_t rows = body.intEncodedStatements.size();
        std::vector<std::string> lines;
        std::size_t sent = 0;
        std::size_t withheld = 0;
        std::size_t withoutCore = 0;
        uint32_t next = 0;
        for (std::size_t i = 0; i < rows; ++i) {
            const IntEncodedExpr& ie = body.intEncodedStatements[i];
            assert(ie.nameId > 0 && static_cast<std::size_t>(ie.nameId) < coreSlots
                && "recordPhase2Upload: a row core outside the premise-core mask");
            assert((ie.negation == 0 || ie.negation == 1)
                && "recordPhase2Upload: a row polarity other than 0 or 1");
            const bool isSent = next < sentCount
                && sentIndices[next] == static_cast<NameId>(i);
            if (isSent) ++next;
            const bool coreUsed = (coreUsedMask[ie.nameId] & (1u << ie.negation)) != 0;
            assert((!isSent || coreUsed)
                && "recordPhase2Upload: a sent row without a premise core");
            if (!coreUsed) {
                ++withoutCore;
                continue;
            }
            if (isSent) ++sent;
            else ++withheld;
            lines.push_back(std::string(isSent ? "  sent [" : "  withheld [")
                + std::to_string(i) + "] " + body.nameMap.decode(ie.originalId)
                + " | v=" + body.nameMap.decode(ie.validityId));
        }
        assert(next == sentCount
            && "recordPhase2Upload: a sent index is not a row of the block");
        uploadLines.push_back("  upload #" + std::to_string(uploadCount) + ": "
            + std::to_string(rows) + " rows, " + std::to_string(sent) + " sent, "
            + std::to_string(withheld) + " withheld, " + std::to_string(withoutCore)
            + " without a premise core");
        uploadLines.insert(uploadLines.end(), lines.begin(), lines.end());
        uploadSent += sent;
        uploadWithheld += withheld;
    }

    /// @brief Write the buffered Phase-2 uploads as the
    ///        `phase2.statementUpload` section and empty the buffer.
    ///
    /// @details
    /// One header line with the uploads since the previous EXIT and their
    /// sent and withheld rows, then the buffered summary and row lines in
    /// recording order; the counters restart for the next burst.
    ///
    /// @param f Trace output stream.
    /// @return Nothing.
    /// @see `recordPhase2Upload`, `dumpExit`.
    void writePhase2Uploads(std::ofstream& f) {
        std::lock_guard<std::mutex> lock(uploadMtx);
        f << "-- phase2.statementUpload (" << uploadCount << " uploads, "
          << uploadSent << " sent, " << uploadWithheld
          << " withheld by the premise patterns):\n";
        for (const std::string& line : uploadLines) f << line << "\n";
        uploadLines.clear();
        uploadCount = 0;
        uploadSent = 0;
        uploadWithheld = 0;
    }

} // namespace hashburst_dump
} // namespace gl
