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
#include "hashburst_dump.hpp"
#include <fstream>

namespace gl {
namespace hashburst_dump {
/// @brief Serialize logical-block and hash-memory state for the trace.
/// @details Only the dump entry points may invoke its section writers.
/// This diagnostic reader participates in no inference or admission decision.
class HashburstDumpReader {
    friend void dumpEntry(const Memory& body, const CompiledExpressionMap& compiledExpressions);
    friend void dumpExit(const Memory& body);
private:
/// @brief Write the header trace section.
///
/// @details Preserves the diagnostic field spelling, iteration order and output bytes.
/// @param f Trace output stream.
/// @param body Read-only logical-block state.
/// @param label ENTRY or EXIT label.
/// @param count Existing entry or exit sequence number.
/// @return None.
static void writeHeader(std::ofstream& f, const Memory& body,
                         const std::string& label, int count);
/// @brief Write the logical-block chain trace section.
///
/// @details Preserves the diagnostic field spelling, iteration order and output bytes.
/// @param f Trace output stream.
/// @param body Read-only logical-block state.
/// @return None.
static void writeLBChain(std::ofstream& f, const Memory& body);
/// @brief Write the encoded statements trace section.
///
/// @details Preserves the diagnostic field spelling, iteration order and output bytes.
/// @param f Trace output stream.
/// @param body Read-only logical-block state.
/// @return None.
static void writeEncodedStatements(std::ofstream& f, const Memory& body);
/// @brief Write the to be proved trace section.
///
/// @details Preserves the diagnostic field spelling, iteration order and output bytes.
/// @param f Trace output stream.
/// @param body Read-only logical-block state.
/// @return None.
static void writeToBeProved(std::ofstream& f, const Memory& body);
/// @brief Write the expression-origin trace section.
///
/// @details Preserves the diagnostic field spelling, iteration order and output bytes.
/// @param f Trace output stream.
/// @param body Read-only logical-block state.
/// @return None.
static void writeExprOriginMap(std::ofstream& f, const Memory& body);
/// @brief Write the hash originals trace section.
///
/// @details Preserves the diagnostic field spelling, iteration order and output bytes.
/// @param f Trace output stream.
/// @param body Read-only logical-block state.
/// @return None.
static void writeHashOriginals(std::ofstream& f, const Memory& body);
/// @brief Write the admission map trace section.
///
/// @details Preserves the diagnostic field spelling, iteration order and output bytes.
/// @param f Trace output stream.
/// @param body Read-only logical-block state.
/// @return None.
static void writeAdmissionMap(std::ofstream& f, const Memory& body);
/// @brief Write the encoded map markers trace section.
///
/// @details Preserves the diagnostic field spelling, iteration order and output bytes.
/// @param f Trace output stream.
/// @param body Read-only logical-block state.
/// @return None.
static void writeEncodedMapMarkers(std::ofstream& f, const Memory& body);
/// @brief Write the mail in trace section.
///
/// @details Preserves the diagnostic field spelling, iteration order and output bytes.
/// @param f Trace output stream.
/// @param body Read-only logical-block state.
/// @return None.
static void writeMailIn(std::ofstream& f, const Memory& body);
/// @brief Write the compiled expressions trace section.
///
/// @details Preserves the diagnostic field spelling, iteration order and output bytes.
/// @param f Trace output stream.
/// @param compiledExpressions Read-only compiled definition registry.
/// @return None.
static void writeCompiledExpressions(std::ofstream& f,
            const CompiledExpressionMap& compiledExpressions);
/// @brief Write the memory counters trace section.
///
/// @details Preserves the diagnostic field spelling, iteration order and output bytes.
/// @param f Trace output stream.
/// @param body Read-only logical-block state.
/// @return None.
static void writeMemoryCounters(std::ofstream& f, const Memory& body);
/// @brief Check whether a name identifier has a printable row.
///
/// @details Preserves the diagnostic field spelling, iteration order and output bytes.
/// @param nm Name table used to decode identifiers.
/// @param id Name identifier to inspect.
/// @return Whether the identifier lies in the printable range.
static bool nameInRange(const NameMap& nm, NameId id);
/// @brief Read the decoded name for a printable identifier.
///
/// @details Preserves the diagnostic field spelling, iteration order and output bytes.
/// @param nm Name table used to decode identifiers.
/// @param id Name identifier to inspect.
/// @return The decoded name string.
static std::string nameAt(const NameMap& nm, NameId id);
/// @brief Write the name map trace section.
///
/// @details Preserves the diagnostic field spelling, iteration order and output bytes.
/// @param f Trace output stream.
/// @param body Read-only logical-block state.
/// @return None.
static void writeNameMap(std::ofstream& f, const Memory& body);
/// @brief Write the statement levels map trace section.
///
/// @details Preserves the diagnostic field spelling, iteration order and output bytes.
/// @param f Trace output stream.
/// @param body Read-only logical-block state.
/// @return None.
static void writeStatementLevelsMap(std::ofstream& f, const Memory& body);
/// @brief Write the local encoded statements trace section.
///
/// @details Preserves the diagnostic field spelling, iteration order and output bytes.
/// @param f Trace output stream.
/// @param body Read-only logical-block state.
/// @return None.
static void writeLocalEncodedStatements(std::ofstream& f, const Memory& body);
/// @brief Write the integer known-statement trace section.
///
/// @details Preserves the diagnostic field spelling, iteration order and output bytes.
/// @param f Trace output stream.
/// @param body Read-only logical-block state.
/// @return None.
static void writeIntKnownStatements(std::ofstream& f, const Memory& body);
/// @brief Write the equivalence classes map trace section.
///
/// @details Preserves the diagnostic field spelling, iteration order and output bytes.
/// @param f Trace output stream.
/// @param body Read-only logical-block state.
/// @return None.
static void writeEquivalenceClassesMap(std::ofstream& f, const Memory& body);
/// @brief Write the integration prepared trace section.
///
/// @details Preserves the diagnostic field spelling, iteration order and output bytes.
/// @param f Trace output stream.
/// @param body Read-only logical-block state.
/// @return None.
static void writeIntegrationPrepared(std::ofstream& f, const Memory& body);
/// @brief Write the expanded implications trace section.
///
/// @details Preserves the diagnostic field spelling, iteration order and output bytes.
/// @param f Trace output stream.
/// @param body Read-only logical-block state.
/// @return None.
static void writeExpandedImplications(std::ofstream& f, const Memory& body);
/// @brief Write the weak variables trace section.
///
/// @details Preserves the diagnostic field spelling, iteration order and output bytes.
/// @param f Trace output stream.
/// @param body Read-only logical-block state.
/// @return None.
static void writeWeakVariables(std::ofstream& f, const Memory& body);
/// @brief Write the OR-state trace section.
///
/// @details Preserves the diagnostic field spelling, iteration order and output bytes.
/// @param f Trace output stream.
/// @param body Read-only logical-block state.
/// @return None.
static void writeOrState(std::ofstream& f, const Memory& body);
/// @brief Write the validity filters trace section.
///
/// @details Preserves the diagnostic field spelling, iteration order and output bytes.
/// @param f Trace output stream.
/// @param body Read-only logical-block state.
/// @return None.
static void writeValidityFilters(std::ofstream& f, const Memory& body);
/// @brief Write the mail out trace section.
///
/// @details Preserves the diagnostic field spelling, iteration order and output bytes.
/// @param f Trace output stream.
/// @param body Read-only logical-block state.
/// @return None.
static void writeMailOut(std::ofstream& f, const Memory& body);
/// @brief Write the internal mail in trace section.
///
/// @details Preserves the diagnostic field spelling, iteration order and output bytes.
/// @param f Trace output stream.
/// @param body Read-only logical-block state.
/// @return None.
static void writeInternalMailIn(std::ofstream& f, const Memory& body);
/// @brief Write the hash memory full trace section.
///
/// @details Preserves the diagnostic field spelling, iteration order and output bytes.
/// @param f Trace output stream.
/// @param hm Read-only hash-memory state.
/// @param tag Existing hash-memory section prefix.
/// @param nm Name table used to decode identifiers.
/// @param ti Template interner for decoded instruction names.
/// @param valIn Value interner for decoded payloads.
/// @param ruleIn Rule interner for decoded source chains.
/// @return None.
static void writeHashMemoryFull(std::ofstream& f, const HashMemory& hm,
                                 const std::string& tag, const NameMap& nm,
                                 const TemplateInterner& ti,
                                 const ValueInterner& valIn,
                                 const ValueInterner& ruleIn);
/// @brief Write the all hash memories trace section.
///
/// @details Preserves the diagnostic field spelling, iteration order and output bytes.
/// @param f Trace output stream.
/// @param body Read-only logical-block state.
/// @return None.
static void writeAllHashMemories(std::ofstream& f, const Memory& body);
/// @brief Write the reshuffle containers trace section.
///
/// @details Preserves the diagnostic field spelling, iteration order and output bytes.
/// @param f Trace output stream.
/// @param body Read-only logical-block state.
/// @return None.
static void writeReshuffleContainers(std::ofstream& f, const Memory& body);
/// @brief Write the extended sections trace section.
///
/// @details Preserves the diagnostic field spelling, iteration order and output bytes.
/// @param f Trace output stream.
/// @param body Read-only logical-block state.
/// @return None.
static void writeExtendedSections(std::ofstream& f, const Memory& body);
};
} // namespace hashburst_dump
} // namespace gl
