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

#include <algorithm>
#include <fstream>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <set>
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

        // ------------------------------------------------------------------
        // Section writers — every per-LB container. The output format for
        // pre-existing sections (header, LB chain, encodedStatements,
        // toBeProved, exprOriginMap, hashOriginals, admissionMap,
        // encodedMap markers, mailIn, compiledExpressions) is preserved
        // byte-for-byte from the prover.cpp inline-lambda version so that
        // diff / grep recipes the user has built up across investigations
        // keep working. Per Rule 14 the format is a contract — no silent
        // reshapes.
        // ------------------------------------------------------------------
        void writeHeader(std::ofstream& f, const Memory& body,
                         const std::string& label, int count) {
            f << "\n=== HASHBURST " << label << " #" << count
              << " | stmts=" << body.intEncodedStatements.size()
              << " | origins=" << body.exprOriginMap.count()
              << " | hashMem=" << body.overallHashMemory.originals.count()
              << " | toBeProved=" << body.intToBeProved.count()
              << " ===\n";
        }

        void writeLBChain(std::ofstream& f, const Memory& body) {
            f << "-- LB chain (innermost -> root):\n";
            const Memory* cur = &body;
            int depth = 0;
            while (cur != nullptr) {
                f << "  [" << depth << "] " << cur->exprKey() << "\n";
                cur = cur->parentMemory;
                ++depth;
            }
        }

        void writeEncodedStatements(std::ofstream& f, const Memory& body) {
            // Rows decoded from the int registry; the section title and the
            // per-row format are the Rule-14 output contract and stay
            // byte-identical (decode returns the exact interned strings).
            f << "-- encodedStatements:\n";
            for (std::size_t i = 0; i < body.intEncodedStatements.size(); ++i) {
                const IntEncodedExpr& ie = body.intEncodedStatements[i];
                f << "  [" << i << "] " << body.nameMap.decode(ie.originalId)
                  << " | v=" << body.nameMap.decode(ie.validityId) << "\n";
            }
        }

        void writeToBeProved(std::ofstream& f, const Memory& body) {
            // Rows decoded from the packed goal registry and lex-sorted on
            // (original, validityName) — byte-identical to the former
            // std::map<EncodedExpression, …> iteration (Rule-14 contract;
            // data source retargeted only).
            const std::vector<DecodedToBeProvedRow> rows =
                decodeToBeProvedSorted(body.intToBeProved, body.nameMap);
            f << "-- toBeProved (" << body.intToBeProved.count() << "):\n";
            for (const DecodedToBeProvedRow& row : rows) {
                const int32_t auxId = body.intToBeProved.lookup(row.key);
                assert(auxId != 0);
                const std::set<int> auxies = coldIntSetAt(body.intToBeProved, auxId);
                f << "  " << row.original << " | v=" << row.validityName;
                f << " | auxies={";
                bool firstA = true;
                for (int a : auxies) { if (!firstA) f << ","; f << a; firstA = false; }
                // The reserved tags member was dropped as dead (never
                // written, always {}); the literal keeps the Rule-14 row
                // format byte-identical.
                f << "} | tags={}\n";
            }
        }

        void writeExprOriginMap(std::ofstream& f, const Memory& body) {
            f << "-- exprOriginMap:\n";
            // Derived view (Rule 14): decoded + key-sorted — identical
            // bytes to the former std::map<ExpressionWithValidity, ...>
            // iteration; per-key line order stays insertion order.
            const auto rows =
                decodeOriginMapSorted(body.exprOriginMap, body.originInterner);
            for (const auto& row : rows) {
                f << "  " << row.first.first << " | v=" << row.first.second << "\n";
                for (const auto& [tag, deps] : row.second) {
                    f << "    <- " << tag;
                    for (const auto& d : deps)
                        f << " | " << d.original << " (v=" << d.validityName << ")";
                    f << "\n";
                }
            }
        }

        void writeHashOriginals(std::ofstream& f, const Memory& body) {
            f << "-- overallHashMemory.originals ("
              << body.overallHashMemory.originals.count() << "):\n";
            // Derived view (Rule 14): decoded chains, lex-sorted — identical
            // bytes to the former set<vector<string>> iteration order.
            {
                std::vector<std::vector<std::string>> chains;
                chains.reserve(body.overallHashMemory.originals.count());
                for (int32_t oi = 1; oi <= body.overallHashMemory.originals.count(); ++oi) {
                    chains.push_back(decodeValueVector(
                        body.overallHashMemory.originals.decodeKey(oi).ids, body.ruleInterner));
                }
                std::sort(chains.begin(), chains.end());
                for (const auto& chain : chains) {
                    f << " ";
                    for (const auto& s : chain) f << " " << s;
                    f << "\n";
                }
            }
        }

        void writeAdmissionMap(std::ofstream& f, const Memory& body) {
            const auto& am = body.overallHashMemory.admissionMap;
            f << "-- overallHashMemory.admissionMap ("
              << am.count() << "):\n";
            // Derived view: decoded (template, validity) lex-sorted —
            // byte-identical to the former EWV-keyed iteration (Rule 14). The
            // cold run is decoded into the historical sorted set form.
            std::vector<std::pair<std::pair<std::string, std::string>,
                                  AdmissionValueSet>> rows;
            rows.reserve(am.count());
            for (int32_t id = 1; id <= am.count(); ++id) {
                const int32_t pk = am.keyAt(id);
                rows.emplace_back(decodeTemplateKey(pk, body.templateInterner, body.nameMap),
                                  admissionRecordsAt(am, pk, body.valueInterner));
            }
            std::sort(rows.begin(), rows.end(),
                      [](const std::pair<std::pair<std::string, std::string>,
                                         AdmissionValueSet>& a,
                         const std::pair<std::pair<std::string, std::string>,
                                         AdmissionValueSet>& b) {
                          return a.first < b.first;
                      });
            for (const auto& [tv, values] : rows) {
                f << "  markerExpr=" << tv.first << " | v=" << tv.second
                  << " | entries=" << values.size() << "\n";
                int ei = 0;
                for (const auto& v : values) {
                    f << "    [" << ei++ << "] key={";
                    for (const int32_t sId : v.key) f << body.valueInterner.decode(sId) << " ";
                    f << "} remainingArgs={";
                    for (const int32_t rId : v.remainingArgs) f << body.valueInterner.decode(rId) << " ";
                    f << "} maxDepth=" << v.standardMaxAdmissionDepth
                      << " maxSec=" << v.standardMaxSecondaryNumber
                      << " flag=" << v.flag << "\n";
                }
            }
        }

        void writeEncodedMapMarkers(std::ofstream& f, const Memory& body) {
            std::vector<const LocalMemoryValue*> markerLmvs;
            // Decode every key's run first (kept alive in allRuns), then collect
            // marker pointers into them -- the blob map hands back decoded
            // copies, so the runs must outlive the pointers. The subsequent
            // decoded-string sort is unchanged (Rule 14: same derived view).
            const int32_t encN = body.overallHashMemory.encodedMap.count();
            std::vector<std::vector<LocalMemoryValue>> allRuns;
            allRuns.reserve(static_cast<std::size_t>(encN));
            for (int32_t id = 1; id <= encN; ++id)
                allRuns.push_back(
                    body.overallHashMemory.encodedMap.recordsAt(id));
            for (const auto& lmvList : allRuns) {
                for (const auto& lmv : lmvList) {
                    if (lmv.isMarker) {
                        markerLmvs.push_back(&lmv);
                    }
                }
            }
            // Derived view (Rule 14): decoded-string sort — identical bytes
            // to the former (value, key, remainingArgs) string compares.
            // remainingArgIds is decoded-lex sorted storage, so the vector
            // compare reproduces the former set<string> compare exactly.
            const ValueInterner& ruleIn = body.ruleInterner;
            std::sort(markerLmvs.begin(), markerLmvs.end(),
                [&ruleIn](const LocalMemoryValue* a, const LocalMemoryValue* b) {
                    if (a->valueId != b->valueId) {
                        return valueIdLess(a->valueId, b->valueId, ruleIn);
                    }
                    if (valueIdVectorLess(a->keyIds, b->keyIds, ruleIn)) return true;
                    if (valueIdVectorLess(b->keyIds, a->keyIds, ruleIn)) return false;
                    return valueIdVectorLess(a->remainingArgIds, b->remainingArgIds, ruleIn);
                });
            f << "-- overallHashMemory.encodedMap marker entries ("
              << markerLmvs.size() << "):\n";
            for (const auto* lmv : markerLmvs) {
                f << "  markerExpr=" << ruleIn.decode(lmv->valueId) << "\n";
                f << "    key={";
                for (const int32_t sId : lmv->keyIds) f << ruleIn.decode(sId) << " ";
                f << "}\n";
                f << "    remainingArgs={";
                for (const int32_t rId : lmv->remainingArgIds) f << ruleIn.decode(rId) << " ";
                f << "}\n";
            }
        }

        void writeMailIn(std::ofstream& f, const Memory& body) {
            // mailIn rides the never-deloaded mail pool (I-101) and is id-form
            // (GLOBAL mailInterner ids); materialize the canonical heap snapshot
            // via routingMailInToHeap (decodes through the global interner) so the
            // byte-identical format below is unchanged (user-approved Rule-14
            // touch, byte-neutral).
            const Mail mailIn = routingMailInToHeap(body.mailIn);
            f << "-- mailIn.statements (" << mailIn.statements.size() << "):\n";
            for (const auto& st : mailIn.statements) {
                f << "  " << st.first.original
                  << " (vName=" << st.first.validityName << ")\n";
            }
            // (mailIn.implications sub-section removed — ASIC 0.1
            //  reshuffle deleted Mail::implications; user-approved dump
            //  edit per Rule 14.)
            f << "-- mailIn.exprOriginMap (" << mailIn.exprOriginMap.size() << "):\n";
            for (const auto& [key, origins] : mailIn.exprOriginMap) {
                f << "  " << key.original << " | v=" << key.validityName << "\n";
                for (const auto& [tag, deps] : origins) {
                    f << "    <- " << tag;
                    for (const auto& d : deps)
                        f << " | " << d.original << " (v=" << d.validityName << ")";
                    f << "\n";
                }
            }
        }

        void writeCompiledExpressions(std::ofstream& f,
            const CompiledExpressionMap& compiledExpressions) {
            f << "-- compiledExpressions (" << compiledExpressions.size() << "):\n";
            for (const auto& [name, le] : compiledExpressions) {
                f << "  " << name << " | sig=" << le.signature << "\n";
            }
        }

        // ------------------------------------------------------------------
        // Extended sections — every per-LB container that wasn't already
        // dumped above. The user explicitly authorised this expansion
        // (Generative Logic, 2026-05-14) so that future diagnostics can be
        // done by reading the trace alone, no extra traps.
        // ------------------------------------------------------------------
        void writeMemoryCounters(std::ofstream& f, const Memory& body) {
            f << "-- counters: startInt=" << body.startInt
              << " startIntRepl=" << body.startIntRepl
              << " startIntPi=" << body.startIntPi
              << " level=" << body.level
              << " isActive=" << (body.isActive ? 1 : 0)
              << " isPartOfRecursion=" << (body.isPartOfRecursion ? 1 : 0)
              << " deltaNumberStatements=" << body.deltaNumberStatements
              << " recursionCounter=" << body.recursionCounter
              << " contradictionIndex=" << body.contradictionIndex
              << " primedForContradiction=" << (body.primedForContradiction ? 1 : 0)
              << "\n";
            if (body.recursionHypothesisId != 0)
                f << "-- recursionHypothesis: "
                  << body.nameMap.decode(body.recursionHypothesisId) << "\n";
            if (body.contradictionTheoremId != 0)
                f << "-- contradictionTheorem: "
                  << body.nameMap.decode(body.contradictionTheoremId) << "\n";
        }

        /// @brief Whether @p id has a printable row in the dump's name
        ///        section — slot 0 (the reserved empty string) through
        ///        `nameCount()`.
        ///
        /// @param nm The LB's NameMap.
        /// @param id Candidate id.
        /// @return Whether @p id is a printable row.
        static bool nameInRange(const NameMap& nm, int16_t id) {
            return id >= 0 && static_cast<int32_t>(id) <= nm.nameCount();
        }

        /// @brief The name printed for @p id (slot 0 is the empty
        ///        string).
        ///
        /// @param nm The LB's NameMap.
        /// @param id An id satisfying `nameInRange`.
        /// @return The name (owned copy).
        static std::string nameAt(const NameMap& nm, int16_t id) {
            return id == 0 ? std::string() : nm.decode(id);
        }

        void writeNameMap(std::ofstream& f, const Memory& body) {
            const NameMap& nm = body.nameMap;
            // Section labels and row layout are a fixed dump-format
            // contract (Rule 14): row 0 is the reserved empty slot, ids
            // 1..nameCount() decode from the table, so the printed size
            // is nameCount()+1 and nextId is nameCount()+1.
            const std::size_t nameRows =
                static_cast<std::size_t>(nm.nameCount()) + 1;
            f << "-- nameMap.idToName (" << nameRows
              << " entries; nextId=" << (nm.nameCount() + 1) << "):\n";
            for (std::size_t id = 0; id < nameRows; ++id) {
                f << "  [" << id << "] "
                  << (id == 0 ? std::string()
                              : nm.decode(static_cast<int16_t>(id)))
                  << "\n";
            }
            const std::size_t subRows =
                static_cast<std::size_t>(nm.subCount()) + 1;
            f << "-- nameMap.idToSub (" << subRows
              << " entries; nextSubId=" << (nm.subCount() + 1) << "):\n";
            for (std::size_t id = 0; id < subRows; ++id) {
                f << "  [" << id << "] "
                  << (id == 0 ? std::string()
                              : nm.decodeSub(static_cast<int16_t>(id)))
                  << "\n";
            }
            f << "-- nameMap.stackOfValidity (" << nm.stackSize() << "):\n";
            for (int32_t id = 0; id < nm.stackSize(); ++id) {
                f << "  [" << id << "] {";
                const int32_t stLen = nm.stackLen(static_cast<int16_t>(id));
                for (int32_t i = 0; i < stLen; ++i) {
                    if (i) f << ",";
                    f << nm.stackAt(static_cast<int16_t>(id), i);
                }
                f << "}\n";
            }
            f << "-- nameMap.ancestorsOf (" << nm.ancSize() << "):\n";
            for (int32_t id = 0; id < nm.ancSize(); ++id) {
                f << "  [" << id << "] {";
                const int32_t anLen = nm.ancLen(static_cast<int16_t>(id));
                for (int32_t i = 0; i < anLen; ++i) {
                    if (i) f << ",";
                    f << nm.ancAt(static_cast<int16_t>(id), i);
                }
                f << "}\n";
            }
            f << "-- nameMap.pairMap (" << nm.pairCount() << " entries; size only)\n";
        }

        void writeStatementLevelsMap(std::ofstream& f, const Memory& body) {
            // Section contract: the packed-key level index, decoded and
            // lex-sorted by (original, validityName) — the former
            // std::map<EncodedExpression, std::set<int>> iteration order,
            // byte-identical (Rule 14). Keys are unique pairs, so the sort
            // has no ties; the level set is an ordered std::set<int> and
            // prints ascending as before.
            std::vector<std::pair<std::pair<std::string, std::string>,
                                  std::set<int>>> rows;
            const int32_t levN = body.intStatementLevelsMap.count();
            rows.reserve(static_cast<std::size_t>(levN));
            for (int32_t id = 1; id <= levN; ++id) {
                const int32_t key = body.intStatementLevelsMap.keyAt(id);
                const int16_t origId = static_cast<int16_t>(
                    (static_cast<uint32_t>(key) >> 16) & 0xFFFF);
                const int16_t valId = static_cast<int16_t>(
                    static_cast<uint32_t>(key) & 0xFFFF);
                rows.emplace_back(
                    std::make_pair(body.nameMap.decode(origId),
                                   body.nameMap.decode(valId)),
                    coldIntSetAt(body.intStatementLevelsMap, id));
            }
            std::sort(rows.begin(), rows.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
            f << "-- statementLevelsMap (" << rows.size() << "):\n";
            for (const auto& row : rows) {
                f << "  " << row.first.first << " | v=" << row.first.second
                  << " | levels={";
                bool first = true;
                for (int lv : row.second) { if (!first) f << ","; f << lv; first = false; }
                f << "}\n";
            }
        }

        void writeWholeExpressions(std::ofstream& f, const Memory& body) {
            // Section contract: the `registered`-membership rows, decoded
            // and lex-sorted by (original, validityName) — the former
            // std::map<EncodedExpression, ...> iteration order,
            // byte-identical (Rule 14).
            std::vector<std::pair<std::string, std::string>> rows;
            rows.reserve(body.intKnownStatements.count());
            for (int32_t i = 1; i <= body.intKnownStatements.count(); ++i) {
                if (!body.intKnownStatements.valueAt(i).registered) continue;
                const int32_t key = body.intKnownStatements.keyAt(i);
                const int16_t origId = static_cast<int16_t>(
                    (static_cast<uint32_t>(key) >> 16) & 0xFFFF);
                const int16_t valId = static_cast<int16_t>(
                    static_cast<uint32_t>(key) & 0xFFFF);
                rows.emplace_back(body.nameMap.decode(origId),
                                  body.nameMap.decode(valId));
            }
            std::sort(rows.begin(), rows.end());
            f << "-- wholeExpressions (" << rows.size() << "):\n";
            for (const auto& row : rows) {
                f << "  " << row.first << " | v=" << row.second << "\n";
            }
        }

        void writeLocalEncodedStatements(std::ofstream& f, const Memory& body) {
            // Rows decoded from the int registries; titles + per-row format
            // are the Rule-14 output contract, byte-identical.
            f << "-- localEncodedStatements (" << body.intLocalEncodedStatements.size() << "):\n";
            for (std::size_t i = 0; i < body.intLocalEncodedStatements.size(); ++i) {
                const IntEncodedExpr& ie = body.intLocalEncodedStatements[i];
                f << "  [" << i << "] " << body.nameMap.decode(ie.originalId)
                  << " | v=" << body.nameMap.decode(ie.validityId) << "\n";
            }
            f << "-- localEncodedStatementsDelta (" << body.intLocalEncodedStatementsDelta.size() << "):\n";
            for (std::size_t i = 0; i < body.intLocalEncodedStatementsDelta.size(); ++i) {
                const IntEncodedExpr& ie = body.intLocalEncodedStatementsDelta[i];
                f << "  [" << i << "] " << body.nameMap.decode(ie.originalId)
                  << " | v=" << body.nameMap.decode(ie.validityId) << "\n";
            }
        }

        void writeIntKnownStatements(std::ofstream& f, const Memory& body) {
            // Section contract: the `known`-membership rows (the Site F
            // dedup record). The packed map also carries registered-only
            // rows, which belong to the wholeExpressions section.
            std::vector<int32_t> keys;
            keys.reserve(body.intKnownStatements.count());
            for (int32_t i = 1; i <= body.intKnownStatements.count(); ++i)
                if (body.intKnownStatements.valueAt(i).known)
                    keys.push_back(body.intKnownStatements.keyAt(i));
            std::sort(keys.begin(), keys.end());
            f << "-- intKnownStatements (" << keys.size()
              << " packed (origId,validityId) keys):\n";
            for (int32_t k : keys) {
                const int16_t origId  = static_cast<int16_t>((static_cast<uint32_t>(k) >> 16) & 0xFFFF);
                const int16_t valId   = static_cast<int16_t>(static_cast<uint32_t>(k) & 0xFFFF);
                f << "  origId=" << origId << " valId=" << valId;
                if (nameInRange(body.nameMap, origId))
                    f << " orig=" << nameAt(body.nameMap, origId);
                if (nameInRange(body.nameMap, valId))
                    f << " v=" << nameAt(body.nameMap, valId);
                f << "\n";
            }
        }

        void writeEquivalenceClassesMap(std::ofstream& f, const Memory& body) {
            f << "-- equivalenceClassesMap (" << body.equivalenceClassesMap.count() << "):\n";
            // Derived view: decoded validity names lex-sorted, members in
            // memberIds storage order (decoded-lex) — byte-identical to the
            // former string-keyed iteration (Rule 14). Read-side adaptation of
            // the cold blob store (Batch 3); each bucket decoded by id.
            std::vector<std::pair<std::string, std::vector<EquivalenceClass>>> rows;
            const int32_t eqKeyCount = body.equivalenceClassesMap.count();
            rows.reserve(static_cast<std::size_t>(eqKeyCount));
            for (int32_t kid = 1; kid <= eqKeyCount; ++kid) {
                const int16_t vId = body.equivalenceClassesMap.keyAt(kid);
                rows.emplace_back(std::string(body.nameMap.decode(vId)),
                                  body.decodeClassesById(vId));
            }
            std::sort(rows.begin(), rows.end(),
                      [](const std::pair<std::string, std::vector<EquivalenceClass>>& a,
                         const std::pair<std::string, std::vector<EquivalenceClass>>& b) {
                          return a.first < b.first;
                      });
            for (const auto& [validity, classes] : rows) {
                f << "  v=" << validity << " | classes=" << classes.size() << "\n";
                for (std::size_t ci = 0; ci < classes.size(); ++ci) {
                    const auto& ec = classes[ci];
                    f << "    [" << ci << "] vars={";
                    bool first = true;
                    for (const int16_t mid : ec.memberIds) {
                        if (!first) f << ",";
                        f << body.nameMap.decode(mid);
                        first = false;
                    }
                    f << "}\n";
                }
            }
        }

        void writeIntegrationPrepared(std::ofstream& f, const Memory& body) {
            // Derived views (Rule 14): decoded + lex-sorted — identical
            // bytes to the former EWV-set / string-map iterations.
            auto writePackedTemplateSection = [&](const char* name,
                                                  const ColdHashSet<PodKeyStore<int32_t>>& c) {
                f << "-- " << name << " (" << c.count() << "):\n";
                std::vector<std::pair<std::string, std::string>> rows;
                rows.reserve(c.count());
                for (int32_t i = 1; i <= c.count(); ++i) {
                    rows.push_back(decodeTemplateKey(c.decode(i), body.templateInterner,
                                                     body.nameMap));
                }
                std::sort(rows.begin(), rows.end());
                for (const auto& [orig, val] : rows) {
                    f << "  " << orig << " | v=" << val << "\n";
                }
            };
            writePackedTemplateSection("integrationPrepared", body.integrationPrepared);
            writePackedTemplateSection("integrationPreparedMarker",
                                       body.integrationPreparedMarker);
            f << "-- integrationStartIntMap (" << body.integrationStartIntMap.count() << "):\n";
            {
                std::vector<std::pair<std::string, int>> stiRows;
                stiRows.reserve(body.integrationStartIntMap.count());
                for (int32_t i = 1; i <= body.integrationStartIntMap.count(); ++i) {
                    stiRows.emplace_back(
                        std::string(body.templateInterner.decode(
                            body.integrationStartIntMap.keyAt(i))),
                        body.integrationStartIntMap.valueAt(i));
                }
                std::sort(stiRows.begin(), stiRows.end());
                for (const auto& [expr, sti] : stiRows) {
                    f << "  " << expr << " | startInt=" << sti << "\n";
                }
            }
        }

        void writeExpandedImplications(std::ofstream& f, const Memory& body) {
            f << "-- expandedImplications (" << body.expandedImplications.count() << "):\n";
            // Derived view (Rule 14): decoded + lex-sorted — identical
            // bytes to the former EWV-set iteration.
            std::vector<std::pair<std::string, std::string>> rows;
            rows.reserve(body.expandedImplications.count());
            for (int32_t i = 1; i <= body.expandedImplications.count(); ++i) {
                const int64_t pk = body.expandedImplications.decode(i);
                rows.emplace_back(
                    std::string(body.lbStateInterner.decode(
                        static_cast<int32_t>(static_cast<uint64_t>(pk) >> 32))),
                    std::string(body.lbStateInterner.decode(
                        static_cast<int32_t>(pk & 0xFFFFFFFFLL))));
            }
            std::sort(rows.begin(), rows.end());
            for (const auto& [orig, val] : rows) {
                f << "  " << orig << " | v=" << val << "\n";
            }
        }

        void writeWeakVariables(std::ofstream& f, const Memory& body) {
            f << "-- weakVariables (" << body.intWeakVariables.count() << "):\n";
            // Derived view: decode the packed keys, lex-sort on
            // (variable, validity) — byte-identical to the former
            // ExpressionWithValidity-set iteration (Rule 14).
            std::vector<std::pair<std::string, std::string>> rows;
            rows.reserve(body.intWeakVariables.count());
            for (int32_t i = 1; i <= body.intWeakVariables.count(); ++i) {
                const int32_t pk = body.intWeakVariables.decode(i);
                rows.emplace_back(
                    std::string(body.nameMap.decode(static_cast<int16_t>((pk >> 16) & 0xFFFF))),
                    std::string(body.nameMap.decode(static_cast<int16_t>(pk & 0xFFFF))));
            }
            std::sort(rows.begin(), rows.end());
            for (const auto& [orig, val] : rows) {
                f << "  " << orig << " | v=" << val << "\n";
            }
        }

        void writeOrState(std::ofstream& f, const Memory& body) {
            // The orAdmissionSet container was dropped (no insert site
            // existed — D-31 / D-135); the literal
            // empty section keeps the Rule-14 row format byte-identical.
            f << "-- orAdmissionSet (0):\n";
            f << "-- orBookkeeping (" << body.orBookkeeping.count() << "):\n";
            // Derived view (Rule 14): decoded + (expr, orSig) lex-sorted —
            // identical bytes to the former std::map iteration; the cold run
            // is decoded-lex ordered storage already (insertSorted +
            // DecodedIdLess), so the per-disjunct run order is iterated as-is
            // (never coldIntSetAt, which would re-sort by raw int).
            {
                std::vector<std::pair<std::pair<std::string, std::string>,
                                      std::vector<int32_t>>> obRows;
                const int32_t obN = body.orBookkeeping.count();
                obRows.reserve(static_cast<std::size_t>(obN));
                for (int32_t id = 1; id <= obN; ++id) {
                    const int64_t pk = body.orBookkeeping.keyAt(id);
                    std::vector<int32_t> djs;
                    const int32_t rl = body.orBookkeeping.runLen(id);
                    for (int32_t j = 0; j < rl; ++j)
                        djs.push_back(body.orBookkeeping.valueAt(id, j));
                    obRows.emplace_back(std::make_pair(
                        std::string(body.lbStateInterner.decode(
                            static_cast<int32_t>(static_cast<uint64_t>(pk) >> 32))),
                        std::string(body.lbStateInterner.decode(
                            static_cast<int32_t>(pk & 0xFFFFFFFFLL)))),
                        std::move(djs));
                }
                std::sort(obRows.begin(), obRows.end(),
                    [](const auto& a, const auto& b) { return a.first < b.first; });
                for (const auto& row : obRows) {
                    f << "  expr=" << row.first.first << " | orSig=" << row.first.second
                      << " | disjuncts={";
                    bool first = true;
                    for (const int32_t d : row.second) {
                        if (!first) f << ",";
                        f << body.lbStateInterner.decode(d);
                        first = false;
                    }
                    f << "}\n";
                }
            }
            f << "-- orDisjunctCount (" << body.orDisjunctCount.count() << "):\n";
            {
                std::vector<std::pair<std::string, int>> dcRows;
                dcRows.reserve(body.orDisjunctCount.count());
                for (int32_t i = 1; i <= body.orDisjunctCount.count(); ++i) {
                    dcRows.emplace_back(
                        std::string(body.lbStateInterner.decode(
                            body.orDisjunctCount.keyAt(i))),
                        body.orDisjunctCount.valueAt(i));
                }
                std::sort(dcRows.begin(), dcRows.end());
                for (const auto& [sig, cnt] : dcRows) {
                    f << "  " << sig << " | count=" << cnt << "\n";
                }
            }
        }

        void writeValidityFilters(std::ofstream& f, const Memory& body) {
            // The former string-set section is gone (user-approved): its
            // content was a strict subset of the id set (the wipe path
            // recorded only the closing scope name), so the id section
            // below is the single complete record.
            f << "-- intValidityNamesToFilter (" << body.intValidityNamesToFilter.count()
              << "):\n";
            {
                std::vector<int16_t> ids;
                ids.reserve(body.intValidityNamesToFilter.count());
                for (int32_t i = 1; i <= body.intValidityNamesToFilter.count(); ++i)
                    ids.push_back(body.intValidityNamesToFilter.decode(i));
                std::sort(ids.begin(), ids.end());
                for (int16_t id : ids) {
                    f << "  id=" << id;
                    if (nameInRange(body.nameMap, id))
                        f << " v=" << nameAt(body.nameMap, id);
                    f << "\n";
                }
            }
            f << "-- pendingWipeScopes (" << body.pendingWipeScopes.count() << "):\n";
            {
                std::vector<int16_t> pw;
                pw.reserve(body.pendingWipeScopes.count());
                for (int32_t i = 1; i <= body.pendingWipeScopes.count(); ++i)
                    pw.push_back(body.pendingWipeScopes.decode(i));
                std::sort(pw.begin(), pw.end());
                for (const int16_t v : pw) f << "  " << v << "\n";
            }
            // Section contract: decoded names in lexicographic order
            // (Rule-14 byte format); sourced from the id sets.
            {
                std::vector<std::string> sent;
                sent.reserve(body.canBeSentIds.count());
                for (int32_t i = 1; i <= body.canBeSentIds.count(); ++i)
                    sent.push_back(body.nameMap.decode(body.canBeSentIds.decode(i)));
                std::sort(sent.begin(), sent.end());
                f << "-- canBeSentSet (" << sent.size() << "):\n";
                for (const auto& s : sent) f << "  " << s << "\n";

                std::vector<std::string> marker;
                marker.reserve(body.canBeSentMarkerIds.count());
                for (int32_t i = 1; i <= body.canBeSentMarkerIds.count(); ++i)
                    marker.push_back(body.nameMap.decode(
                        body.canBeSentMarkerIds.decode(i)));
                std::sort(marker.begin(), marker.end());
                f << "-- canBeSentMarkerSet (" << marker.size() << "):\n";
                for (const auto& s : marker) f << "  " << s << "\n";
            }
            // Section contract: decoded names in lexicographic order
            // (Rule-14 byte format); sourced from the id set, which is
            // content-identical.
            {
                std::vector<std::string> axedNames;
                axedNames.reserve(body.intAxedVariables.count());
                for (int32_t i = 1; i <= body.intAxedVariables.count(); ++i)
                    axedNames.push_back(body.nameMap.decode(
                        body.intAxedVariables.decode(i)));
                std::sort(axedNames.begin(), axedNames.end());
                f << "-- axedVariables (" << axedNames.size() << "):\n";
                for (const auto& s : axedNames) f << "  " << s << "\n";
            }
            f << "-- intAxedVariables (" << body.intAxedVariables.count() << "):\n";
            {
                std::vector<int16_t> ids;
                ids.reserve(body.intAxedVariables.count());
                for (int32_t i = 1; i <= body.intAxedVariables.count(); ++i)
                    ids.push_back(body.intAxedVariables.decode(i));
                std::sort(ids.begin(), ids.end());
                for (int16_t id : ids) {
                    f << "  id=" << id;
                    if (nameInRange(body.nameMap, id))
                        f << " name=" << nameAt(body.nameMap, id);
                    f << "\n";
                }
            }
        }

        void writeMailOut(std::ofstream& f, const Memory& body) {
            // mailOut is deloadable LB state: materialize the
            // canonical heap snapshot so the byte-identical format below is
            // unchanged (user-approved Rule-14 touch, byte-neutral). mailOut is
            // id-form in its own per-LB interner; routingMailOutToHeap decodes
            // through that interner.
            const Mail mailOut = routingMailOutToHeap(body.mailOut,
                                                      body.mailOutInterner);
            f << "-- mailOut.statements (" << mailOut.statements.size() << "):\n";
            for (const auto& st : mailOut.statements) {
                f << "  " << st.first.original
                  << " (vName=" << st.first.validityName << ")\n";
            }
            // (mailOut.implications sub-section removed — ASIC 0.1
            //  reshuffle deleted Mail::implications; user-approved dump
            //  edit per Rule 14.)
            f << "-- mailOut.exprOriginMap (" << mailOut.exprOriginMap.size() << "):\n";
            for (const auto& [key, origins] : mailOut.exprOriginMap) {
                f << "  " << key.original << " | v=" << key.validityName << "\n";
                for (const auto& [tag, deps] : origins) {
                    f << "    <- " << tag;
                    for (const auto& d : deps)
                        f << " | " << d.original << " (v=" << d.validityName << ")";
                    f << "\n";
                }
            }
            // (mailOut.expandedImplications sub-section removed — the dead mail
            //  expandedImplications column was deleted; user-approved dump edit
            //  per Rule 14.)
        }

        void writeInternalMailIn(std::ofstream& f, const Memory& body) {
            // sameIterationInternalMail is a ColdMail (cold deloadable storage,
            // now id-form); read it through a heap-Mail snapshot. makeHeapMail
            // decodes the stored ids back through the LB's NameMap / originInterner
            // and re-imposes the canonical std::set / std::map order, so the dump
            // bytes are identical to before (user-approved Rule-14 edit,
            // byte-neutral — the same pattern the routing mailIn dump uses).
            const Mail im = makeHeapMail(body.sameIterationInternalMail,
                                         body.nameMap, body.originInterner);
            f << "-- sameIterationInternalMail.statements ("
              << im.statements.size() << "):\n";
            for (const auto& st : im.statements) {
                f << "  " << st.first.original
                  << " (vName=" << st.first.validityName << ")\n";
            }
            // (sameIterationInternalMail.implications sub-section removed — ASIC 0.1
            //  reshuffle deleted Mail::implications; user-approved dump
            //  edit per Rule 14. The sameIterationInternalMail channel itself stays;
            //  only its now-nonexistent implications bag is dropped.)
            f << "-- sameIterationInternalMail.exprOriginMap ("
              << im.exprOriginMap.size() << "):\n";
            for (const auto& [key, origins] : im.exprOriginMap) {
                f << "  " << key.original << " | v=" << key.validityName << "\n";
                for (const auto& [tag, deps] : origins) {
                    f << "    <- " << tag;
                    for (const auto& d : deps)
                        f << " | " << d.original << " (v=" << d.validityName << ")";
                    f << "\n";
                }
            }
        }

        // Per-HashMemory full dump. Called three times (overall / local /
        // delta) from `writeAllHashMemories`. The owner-set value type is
        // `std::set<ExpressionWithValidity>` per `D-72`.
        void writeHashMemoryFull(std::ofstream& f, const HashMemory& hm,
                                 const std::string& tag, const NameMap& nm,
                                 const TemplateInterner& ti,
                                 const ValueInterner& valIn,
                                 const ValueInterner& ruleIn) {
            f << "-- " << tag << ".encodedMap (" << hm.encodedMap.count()
              << " keys):\n";
            const int32_t encCount = hm.encodedMap.count();
            for (int32_t encId = 1; encId <= encCount; ++encId) {
                const std::vector<LocalMemoryValue> lmvList =
                    hm.encodedMap.recordsAt(encId);
                f << "  key (n=" << lmvList.size() << " LMV):\n";
                for (std::size_t i = 0; i < lmvList.size(); ++i) {
                    const auto& lmv = lmvList[i];
                    // Derived view (Rule 14): decode in place — the map and
                    // its per-key vectors are untouched, so iteration order
                    // and bytes match the former string fields exactly.
                    f << "    [" << i << "] value=" << ruleIn.decode(lmv.valueId)
                      << " | origImpl=" << ruleIn.decode(lmv.originalImplicationId)
                      << " | v=" << nm.decode(lmv.validityId)
                      << " | levels={";
                    bool first = true;
                    for (int lv : lmv.levels) { if (!first) f << ","; f << lv; first = false; }
                    f << "} | just=" << ruleJustificationName(lmv.justification)
                      << " | productOfDis=" << (lmv.productOfDisintegration ? 1 : 0)
                      << "\n      key={";
                    for (const int32_t sId : lmv.keyIds) f << ruleIn.decode(sId) << " ";
                    f << "} remainingArgs={";
                    for (const int32_t rId : lmv.remainingArgIds) f << ruleIn.decode(rId) << " ";
                    f << "}\n";
                }
            }

            auto writeIntKey = [&](const IntNormalizedKey& k) {
                f << "{nExpr=" << k.numberExpressions << " len=" << k.length
                  << " data=[";
                for (int16_t i = 0; i < k.length; ++i) {
                    if (i) f << ",";
                    f << k.data[i];
                }
                f << "]}";
            };
            auto dumpOwnerSetMap = [&](const char* label,
                const TypedColdBlobMap<NormKey, OwnerSet>& m) {
                const int32_t n = m.count();
                f << "-- " << tag << "." << label << " (" << n
                  << " keys, owner-set):\n";
                // Derived view (Rule 14): the cold blob map stores keys in id
                // (insertion) order; decode every (key, OwnerSet) and lex-sort by
                // key so the section is deterministic and storage-independent
                // (matches the writeEncodedMapMarkers decode+sort pattern). The
                // per-key format below is byte-identical to the former
                // unordered_map iteration.
                std::vector<std::pair<NormKey, OwnerSet>> rows;
                rows.reserve(static_cast<std::size_t>(n));
                for (int32_t id = 1; id <= n; ++id)
                    rows.emplace_back(m.decodeKey(id), m.recordAt(id, 0));
                std::sort(rows.begin(), rows.end(),
                    [](const std::pair<NormKey, OwnerSet>& a,
                       const std::pair<NormKey, OwnerSet>& b) {
                        if (a.first.numberExpressions != b.first.numberExpressions)
                            return a.first.numberExpressions
                                 < b.first.numberExpressions;
                        return a.first.data < b.first.data;
                    });
                for (const auto& [k, ownerSet] : rows) {
                    f << "  key=";
                    const IntNormalizedKey ik(k.numberExpressions, k.data.data(),
                        static_cast<int16_t>(k.data.size()));
                    writeIntKey(ik);
                    f << " | owners=" << ownerSet.partitionIds.size() << " | {";
                    // Derived view (Rule 14): both halves of each packed
                    // owner id are NameMap ids — decode and lex-sort the
                    // (implication, scope) pairs, identical bytes to the
                    // former owners-map iteration order.
                    std::vector<std::pair<std::string, std::string>> ownerRows;
                    ownerRows.reserve(ownerSet.partitionIds.size());
                    for (const int32_t ownerId : ownerSet.partitionIds) {
                        ownerRows.emplace_back(
                            std::string(nm.decode(static_cast<int16_t>(
                                static_cast<uint32_t>(ownerId) >> 16))),
                            std::string(nm.decode(
                                static_cast<int16_t>(ownerId & 0xFFFF))));
                    }
                    std::sort(ownerRows.begin(), ownerRows.end());
                    bool first = true;
                    for (const auto& orow : ownerRows) {
                        if (!first) f << ", ";
                        f << orow.first << "(v=" << orow.second << ")";
                        first = false;
                    }
                    f << "}\n";
                }
            };
            dumpOwnerSetMap("normalizedEncodedKeys",            hm.normalizedEncodedKeys);
            dumpOwnerSetMap("normalizedEncodedSubkeys",         hm.normalizedEncodedSubkeys);
            dumpOwnerSetMap("normalizedEncodedSubkeysMinusOne", hm.normalizedEncodedSubkeysMinusOne);
            dumpOwnerSetMap("normalizedEncodedSubkeysMinusTwo", hm.normalizedEncodedSubkeysMinusTwo);

            f << "-- " << tag << ".remainingArgsNormalizedEncodedMap ("
              << hm.remainingArgsNormalizedEncodedMap.count() << " key-sets):\n";
            // Derived view (Rule 14): the outer arg-sets are lex-sorted (the former
            // unordered_map hash order is gone; sorted is the deterministic
            // canonical form), the inner NormKeys lex-sorted by (numberExpressions,
            // data). Per-key format (writeIntKey) unchanged.
            {
                std::vector<std::pair<std::vector<int16_t>, std::vector<NormKey>>> raRows;
                raRows.reserve(static_cast<std::size_t>(
                    hm.remainingArgsNormalizedEncodedMap.count()));
                for (int32_t id = 1;
                     id <= hm.remainingArgsNormalizedEncodedMap.count(); ++id) {
                    raRows.emplace_back(
                        hm.remainingArgsNormalizedEncodedMap.decodeKey(id).ids,
                        hm.remainingArgsNormalizedEncodedMap.recordsAt(id));
                }
                std::sort(raRows.begin(), raRows.end(),
                    [](const std::pair<std::vector<int16_t>, std::vector<NormKey>>& a,
                       const std::pair<std::vector<int16_t>, std::vector<NormKey>>& b) {
                        return a.first < b.first;
                    });
                for (auto& [argSet, keys] : raRows) {
                    f << "  remainingArgs={";
                    bool first = true;
                    for (int16_t a : argSet) { if (!first) f << ","; f << a; first = false; }
                    f << "} | keys=" << keys.size() << "\n";
                    std::sort(keys.begin(), keys.end(),
                        [](const NormKey& a, const NormKey& b) {
                            if (a.numberExpressions != b.numberExpressions)
                                return a.numberExpressions < b.numberExpressions;
                            return a.data < b.data;
                        });
                    for (const NormKey& k : keys) {
                        f << "    ";
                        const IntNormalizedKey ik(k.numberExpressions,
                            k.data.data(), static_cast<int16_t>(k.data.size()));
                        writeIntKey(ik);
                        f << "\n";
                    }
                }
            }

            f << "-- " << tag << ".admissionMap (" << hm.admissionMap.count() << "):\n";
            // Derived view (Rule 14): decoded lex-sorted, as above. The cold run
            // is decoded into the historical sorted set form.
            std::vector<std::pair<std::pair<std::string, std::string>,
                                  AdmissionValueSet>> amRows;
            amRows.reserve(hm.admissionMap.count());
            for (int32_t id = 1; id <= hm.admissionMap.count(); ++id) {
                const int32_t pk = hm.admissionMap.keyAt(id);
                amRows.emplace_back(decodeTemplateKey(pk, ti, nm),
                                    admissionRecordsAt(hm.admissionMap, pk, valIn));
            }
            std::sort(amRows.begin(), amRows.end(),
                      [](const std::pair<std::pair<std::string, std::string>,
                                         AdmissionValueSet>& a,
                         const std::pair<std::pair<std::string, std::string>,
                                         AdmissionValueSet>& b) {
                          return a.first < b.first;
                      });
            for (const auto& [tv, values] : amRows) {
                f << "  markerExpr=" << tv.first << " | v=" << tv.second
                  << " | entries=" << values.size() << "\n";
                int vi = 0;
                for (const auto& v : values) {
                    f << "    [" << vi++ << "] key={";
                    for (const int32_t sId : v.key) f << valIn.decode(sId) << " ";
                    f << "} remainingArgs={";
                    for (const int32_t rId : v.remainingArgs) f << valIn.decode(rId) << " ";
                    f << "} maxDepth=" << v.standardMaxAdmissionDepth
                      << " maxSec=" << v.standardMaxSecondaryNumber
                      << " flag=" << (v.flag ? 1 : 0) << "\n";
                }
            }

            auto writeLogicalEntity = [&](const LogicalEntity& le, const char* indent) {
                f << indent << "cat=" << le.category
                  << " arity=" << le.arity
                  << " definedSet=" << le.definedSet
                  << " sig=" << le.signature
                  << " elements=" << le.elements.size() << "\n";
                for (std::size_t i = 0; i < le.elements.size(); ++i) {
                    f << indent << "  [" << i << "] " << le.elements[i] << "\n";
                }
            };

            f << "-- " << tag << ".admissionMapIntegration ("
              << hm.admissionMapIntegration.count() << "):\n";
            // Derived view (Rule 14): decoded lex-sorted. The cold run is decoded
            // into the historical nested instruction map form.
            std::vector<std::pair<std::pair<std::string, std::string>,
                                  IntegrationEntryMap>> amiRows;
            amiRows.reserve(hm.admissionMapIntegration.count());
            for (int32_t id = 1; id <= hm.admissionMapIntegration.count(); ++id) {
                const int32_t pk = hm.admissionMapIntegration.keyAt(id);
                amiRows.emplace_back(decodeTemplateKey(pk, ti, nm),
                                     admissionIntegrationRecordsAt(
                                         hm.admissionMapIntegration, pk, valIn));
            }
            std::sort(amiRows.begin(), amiRows.end(),
                      [](const std::pair<std::pair<std::string, std::string>,
                                         IntegrationEntryMap>& a,
                         const std::pair<std::pair<std::string, std::string>,
                                         IntegrationEntryMap>& b) {
                          return a.first < b.first;
                      });
            for (const auto& [tv, instrMap] : amiRows) {
                f << "  markerExpr=" << tv.first << " | v=" << tv.second
                  << " | distinct-instructions=" << instrMap.size() << "\n";
                int ii = 0;
                for (const auto& [instr, payload] : instrMap) {
                    f << "    [" << ii++ << "] markedGoal=" << valIn.decode(instr.markedGoal)
                      << " | data.size=" << instr.data.size()
                      << " | payload=" << payload.size() << "\n";
                    for (std::size_t di = 0; di < instr.data.size(); ++di) {
                        f << "      data[" << di << "]\n";
                        writeLogicalEntity(decodeLogicalEntity(instr.data[di], valIn), "        ");
                    }
                    for (const int32_t sId : payload) f << "      payload: " << valIn.decode(sId) << "\n";
                }
            }

            {
                // Derived views (Rule 14): decoded lex-sorted — identical
                // bytes to the former EWV std::set iteration.
                auto writePackedSection = [&](const char* name,
                                              const TypedColdSet<int32_t>& c) {
                    f << "-- " << tag << "." << name << " (" << c.count() << "):\n";
                    std::vector<std::pair<std::string, std::string>> setRows;
                    setRows.reserve(c.count());
                    for (int32_t i = 1; i <= c.count(); ++i) {
                        setRows.push_back(decodeTemplateKey(c.keyAt(i), ti, nm));
                    }
                    std::sort(setRows.begin(), setRows.end());
                    for (const auto& [orig, val] : setRows) {
                        f << "  " << orig << " | v=" << val << "\n";
                    }
                };
                writePackedSection("admissionSetIntegration", hm.admissionSetIntegration);
                writePackedSection("triggersForAdmissionSetIntegration",
                                   hm.triggersForAdmissionSetIntegration);
            }

            f << "-- " << tag << ".rejectedMap (" << hm.rejectedMap.count() << "):\n";
            // Derived view (Rule 14): decoded lex-sorted. The cold run is decoded
            // into the historical sorted set form.
            std::vector<std::pair<std::pair<std::string, std::string>,
                                  RejectedValueSet>> rmRows;
            rmRows.reserve(hm.rejectedMap.count());
            for (int32_t id = 1; id <= hm.rejectedMap.count(); ++id) {
                const int32_t pk = hm.rejectedMap.keyAt(id);
                rmRows.emplace_back(decodeTemplateKey(pk, ti, nm),
                                    rejectedRecordsAt(hm.rejectedMap, pk, valIn));
            }
            std::sort(rmRows.begin(), rmRows.end(),
                      [](const std::pair<std::pair<std::string, std::string>,
                                         RejectedValueSet>& a,
                         const std::pair<std::pair<std::string, std::string>,
                                         RejectedValueSet>& b) {
                          return a.first < b.first;
                      });
            for (const auto& [tv, values] : rmRows) {
                f << "  key=" << tv.first << " | v=" << tv.second
                  << " | entries=" << values.size() << "\n";
                int vi = 0;
                for (const auto& v : values) {
                    f << "    [" << vi++ << "] renamed=" << valIn.decode(v.renamedExpression)
                      << " | expr=" << valIn.decode(v.expression)
                      << " | iter=" << v.iteration
                      << " | concrete=" << valIn.decode(v.concreteConstituent)
                      << " | levels={";
                    bool first = true;
                    for (int lv : v.levels) { if (!first) f << ","; f << lv; first = false; }
                    f << "} | siblings=" << v.siblings.size() << "\n";
                    for (std::size_t si = 0; si < v.siblings.size(); ++si) {
                        f << "        sib[" << si << "] " << valIn.decode(v.siblings[si]) << "\n";
                    }
                }
            }
            f << "-- " << tag << ".rejectedMapIntegration ("
              << hm.rejectedMapIntegration.count() << "):\n";
            // Derived view (Rule 14): decoded lex-sorted. The cold run is decoded
            // into the historical sorted set form.
            std::vector<std::pair<std::pair<std::string, std::string>,
                                  RejectedIntegrationValueSet>> rmiRows;
            rmiRows.reserve(hm.rejectedMapIntegration.count());
            for (int32_t id = 1; id <= hm.rejectedMapIntegration.count(); ++id) {
                const int32_t pk = hm.rejectedMapIntegration.keyAt(id);
                rmiRows.emplace_back(decodeTemplateKey(pk, ti, nm),
                                     rejectedIntegrationRecordsAt(
                                         hm.rejectedMapIntegration, pk, valIn));
            }
            std::sort(rmiRows.begin(), rmiRows.end(),
                      [](const std::pair<std::pair<std::string, std::string>,
                                         RejectedIntegrationValueSet>& a,
                         const std::pair<std::pair<std::string, std::string>,
                                         RejectedIntegrationValueSet>& b) {
                          return a.first < b.first;
                      });
            for (const auto& [tv, values] : rmiRows) {
                f << "  key=" << tv.first << " | v=" << tv.second
                  << " | entries=" << values.size() << "\n";
                int vi = 0;
                for (const auto& v : values) {
                    f << "    [" << vi++ << "] concrete=" << valIn.decode(v.concreteConstituent)
                      << " | compound=" << valIn.decode(v.compoundExpression)
                      << " | siblings=" << v.siblings.size() << "\n";
                    for (std::size_t si = 0; si < v.siblings.size(); ++si) {
                        f << "        sib[" << si << "] " << valIn.decode(v.siblings[si]) << "\n";
                    }
                }
            }

            // Derived views: decoded names lex-sorted. DELIBERATE order
            // change vs the historical raw hash-iteration print of the
            // string sets — user-approved 2026-06-11 with the cache
            // migration (content unchanged; the one sanctioned byte
            // difference of D-132).
            // Decoded names, lex-sorted, read off the cold key column
            // (D-172).
            auto writeColdTemplateIdSet = [&](const char* name,
                                              const TypedColdSet<int16_t>& c) {
                f << "-- " << tag << "." << name << " (" << c.count() << "):\n";
                std::vector<std::string> names;
                names.reserve(c.count());
                for (int32_t i = 1; i <= c.count(); ++i)
                    names.push_back(ti.decode(c.keyAt(i)));
                std::sort(names.begin(), names.end());
                for (const auto& n : names) f << "  " << n << "\n";
            };
            writeColdTemplateIdSet("varsInRejectedMapIntegrationKeys", hm.varsInRejectedMapIntegrationKeys);
            writeColdTemplateIdSet("varsInAdmissionMapKeys", hm.varsInAdmissionMapKeys);
            writeColdTemplateIdSet("varsInAdmissionMapIntegrationKeys", hm.varsInAdmissionMapIntegrationKeys);

            f << "-- " << tag << ".admissionStatusMap ("
              << hm.admissionStatusMap.count() << "):\n";
            {
                // Derived view (Rule 14): decoded lex-sorted.
                std::vector<std::pair<std::pair<std::string, std::string>, bool>> stRows;
                stRows.reserve(hm.admissionStatusMap.count());
                for (int32_t id = 1; id <= hm.admissionStatusMap.count(); ++id) {
                    const int32_t pk = hm.admissionStatusMap.keyAt(id);
                    stRows.emplace_back(decodeTemplateKey(pk, ti, nm),
                                        hm.admissionStatusMap.valueAt(id) != 0);
                }
                std::sort(stRows.begin(), stRows.end(),
                          [](const std::pair<std::pair<std::string, std::string>, bool>& a,
                             const std::pair<std::pair<std::string, std::string>, bool>& b) {
                              return a.first < b.first;
                          });
                for (const auto& [tv, status] : stRows) {
                    f << "  " << tv.first << " | v=" << tv.second
                      << " | status=" << (status ? 1 : 0) << "\n";
                }
            }
            // Section contract: decoded names in lexicographic order
            // (Rule-14 byte format); sourced from the id set, which carries
            // the full membership.
            {
                std::vector<std::string> prodRecNames;
                prodRecNames.reserve(hm.productsOfRecursionIds.count());
                for (int32_t i = 1; i <= hm.productsOfRecursionIds.count(); ++i)
                    prodRecNames.push_back(nm.decode(hm.productsOfRecursionIds.keyAt(i)));
                std::sort(prodRecNames.begin(), prodRecNames.end());
                f << "-- " << tag << ".productsOfRecursion ("
                  << prodRecNames.size() << "):\n";
                for (const auto& s : prodRecNames) f << "  " << s << "\n";
            }
            f << "-- " << tag << ".productsOfRecursionIds ("
              << hm.productsOfRecursionIds.count() << "):\n";
            // Derived view (Rule 14): cold keys sorted ascending. The former
            // unordered_set raw-iteration order is gone; sorted is the
            // deterministic canonical form (D-173).
            {
                std::vector<int16_t> prodRecIdsSorted;
                prodRecIdsSorted.reserve(hm.productsOfRecursionIds.count());
                for (int32_t i = 1; i <= hm.productsOfRecursionIds.count(); ++i)
                    prodRecIdsSorted.push_back(hm.productsOfRecursionIds.keyAt(i));
                std::sort(prodRecIdsSorted.begin(), prodRecIdsSorted.end());
                for (int16_t id : prodRecIdsSorted) f << "  " << id << "\n";
            }
            // Cold-set variant (D-172): the derived
            // view (decoded (template, validity), lex-sorted) off the cold keys.
            auto writeColdPackedTemplateSet = [&](const char* name,
                                                  const TypedColdSet<int32_t>& c) {
                f << "-- " << tag << "." << name << " (" << c.count() << "):\n";
                std::vector<std::pair<std::string, std::string>> setRows;
                setRows.reserve(c.count());
                for (int32_t i = 1; i <= c.count(); ++i) {
                    setRows.push_back(decodeTemplateKey(c.keyAt(i), ti, nm));
                }
                std::sort(setRows.begin(), setRows.end());
                for (const auto& [orig, val] : setRows) {
                    f << "  " << orig << " | v=" << val << "\n";
                }
            };
            writeColdPackedTemplateSet("consumedAdmissionKeys", hm.consumedAdmissionKeys);
            writeColdPackedTemplateSet("revisitInProgress", hm.revisitInProgress);
            f << "-- " << tag << ".maxKeyLength=" << hm.maxKeyLength << "\n";
        }

        void writeAllHashMemories(std::ofstream& f, const Memory& body) {
            writeHashMemoryFull(f, body.overallHashMemory,     "overallHashMemory",    body.nameMap, body.templateInterner, body.valueInterner, body.ruleInterner);
            writeHashMemoryFull(f, body.localHashMemory,       "localHashMemory",      body.nameMap, body.templateInterner, body.valueInterner, body.ruleInterner);
            writeHashMemoryFull(f, body.localHashMemoryDelta,  "localHashMemoryDelta", body.nameMap, body.templateInterner, body.valueInterner, body.ruleInterner);
        }

        // Reshuffle-introduced per-LB containers (workingMemory /
        // externalStatements / intExternalStatements). Absent on main HEAD;
        // dumped here so the AnchorIncubator container-by-container diff
        // covers the new one-cycle-pipeline state. User-directed addition
        // under Rule 14.
        void writeReshuffleContainers(std::ofstream& f, const Memory& body) {
            // Rows decoded from the int registry; both count lines source the
            // same vector (they were lockstep-equal by construction), so the
            // emitted bytes are unchanged. Titles + format: Rule-14 contract.
            writeHashMemoryFull(f, body.workingMemory, "workingMemory", body.nameMap, body.templateInterner, body.valueInterner, body.ruleInterner);
            f << "-- externalStatements (" << body.intExternalStatements.size() << "):\n";
            for (std::size_t i = 0; i < body.intExternalStatements.size(); ++i) {
                const IntEncodedExpr& ie = body.intExternalStatements[i];
                f << "  [" << i << "] " << body.nameMap.decode(ie.originalId)
                  << " | v=" << body.nameMap.decode(ie.validityId) << "\n";
            }
            f << "-- intExternalStatements (" << body.intExternalStatements.size()
              << " — lockstep int mirror of externalStatements)\n";
        }

        // Common-suffix block — every section after the legacy lambda
        // block. Shared by ENTRY, EXIT so the call sites stay in lock-step.
        void writeExtendedSections(std::ofstream& f, const Memory& body) {
            writeMemoryCounters(f, body);
            writeNameMap(f, body);
            writeStatementLevelsMap(f, body);
            writeWholeExpressions(f, body);
            writeLocalEncodedStatements(f, body);
            writeIntKnownStatements(f, body);
            writeEquivalenceClassesMap(f, body);
            writeIntegrationPrepared(f, body);
            writeExpandedImplications(f, body);
            writeWeakVariables(f, body);
            writeOrState(f, body);
            writeValidityFilters(f, body);
            writeMailOut(f, body);
            writeInternalMailIn(f, body);
            writeAllHashMemories(f, body);
            writeReshuffleContainers(f, body);
        }

    } // anonymous namespace

    bool isTargetLB(const Memory& body) {
        // User-directed target (Rule 14): the Gauss summation theorem's
        // induction LB — recursion block #1 (exprKey (in2[rec0,9,3]),
        // digitArg 9, successor id 3) under the compiled theorem
        // (AnchorGauss)(in3[9,10,11,5])(fold[1,3,4,8,2,9,12])(in2[9,10,3])
        // -> head (in3[7,12,11,5]). Full parent chain to the root sentinel
        // per Rule 12 — the twin theorem copy spawns a recursion LB with
        // the same exprKey under a different chain and must NOT match.
        if (body.exprKey() != "(in2[rec0,9,3])") return false;
        const Memory* p1 = body.parentMemory;
        if (!p1 || p1->exprKey() != "(in2[9,10,3])") return false;
        const Memory* p2 = p1->parentMemory;
        if (!p2 || p2->exprKey() != "(fold[1,3,4,8,2,9,12])") return false;
        const Memory* p3 = p2->parentMemory;
        if (!p3 || p3->exprKey() != "(in3[9,10,11,5])") return false;
        const Memory* p4 = p3->parentMemory;
        if (!p4 || p4->exprKey() != "(AnchorGauss[1,2,3,4,5,6,7,8])") return false;
        const Memory* p5 = p4->parentMemory;
        return p5 && p5->exprKey().empty() && p5->parentMemory == nullptr;
    }

    void dumpEntry(const Memory& body,
                   const CompiledExpressionMap& compiledExpressions) {
        std::lock_guard<std::mutex> lock(entryMtx);
        ++entryCount;
        std::ofstream f(TRACE_PATH,
                        entryCount == 1 ? std::ios::trunc : std::ios::app);
        writeHeader(f, body, "ENTRY", entryCount);
        writeLBChain(f, body);
        writeEncodedStatements(f, body);
        writeToBeProved(f, body);
        writeExprOriginMap(f, body);
        writeHashOriginals(f, body);
        writeAdmissionMap(f, body);
        writeEncodedMapMarkers(f, body);
        writeMailIn(f, body);
        if (entryCount == 1) writeCompiledExpressions(f, compiledExpressions);
        writeExtendedSections(f, body);
        f.flush();
    }

    void dumpExit(const Memory& body) {
        std::lock_guard<std::mutex> lock(exitMtx);
        ++exitCount;
        std::ofstream f(TRACE_PATH, std::ios::app);
        writeHeader(f, body, "EXIT", exitCount);
        writeLBChain(f, body);
        writeEncodedStatements(f, body);
        writeToBeProved(f, body);
        writeExprOriginMap(f, body);
        writeHashOriginals(f, body);
        writeAdmissionMap(f, body);
        writeEncodedMapMarkers(f, body);
        writeMailIn(f, body);
        writeExtendedSections(f, body);
        f.flush();
    }

} // namespace hashburst_dump
} // namespace gl
