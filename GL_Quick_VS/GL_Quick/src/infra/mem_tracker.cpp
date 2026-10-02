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

#include "infra/mem_tracker.hpp"

#include "memory_infra/global_memory_manager.hpp"
#include "memory_infra/lb_memory.hpp"
#include "memory_infra/scratch_arena.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <type_traits>
#include <vector>

namespace gl {
    namespace mem_tracker {

        namespace {

            constexpr int kTagSpace = MemMeasurementParameters::MEM_TAG_SPACE;
            constexpr int kSlots = MemMeasurementParameters::MEM_MAX_SLOTS;
            constexpr int kGlobals = static_cast<int>(GlobalSlot::Count);

            /// One-past-the-last usable tag. Slack has no `ContainerTag` and is
            /// held in its own counter rather than a synthetic row, so this is
            /// purely the assert ceiling for a real tag.
            constexpr int kTagSlack = kTagSpace - 1;
            constexpr int kTagSharedBytes = kTagSpace - 2;
            constexpr int kTagSharedExpressions = kTagSpace - 3;

            // Per-worker accumulator rows. A worker writes only its own row, so
            // concurrent samples never share a cache line's owner and no atomic
            // is needed. Static storage — no heap on the sampling path.
            int64_t g_slotBytes[kSlots][kTagSpace];
            int64_t g_slotIndex[kSlots][kTagSpace];
            int64_t g_slotSlack[kSlots];
            int64_t g_folded[kTagSpace];
            int64_t g_foldedIndex[kTagSpace];
            int64_t g_foldedSlack = 0;
            int64_t g_peak[kTagSpace];
            int64_t g_peakIndex[kTagSpace];
            int64_t g_peakSlack = 0;
            int64_t g_globals[kGlobals];
            int64_t g_peakGlobals[kGlobals];
            int64_t g_peakTotal = 0;
            int64_t g_samples = 0;
            int64_t g_peakSample = -1;

            // Physical view (per worker slot, folded, and at the peak): the
            // pinned blocks split into live pages, free pages parked in held
            // blocks and the remaining block bytes, plus each container's held
            // pages as `LbMemory::visitLiveBytes` reports them. Container names
            // are string literals recorded per slot (a worker writes only its
            // own row), so the name capture needs no synchronization.
            constexpr int kPhysicalRows = 96;
            enum PhysicalSplit { kLivePages = 0, kFreePages, kOtherBlocks, kLbs, kSplitCount };
            int64_t g_slotPhysical[kSlots][kPhysicalRows];
            const char* g_slotPhysicalNames[kSlots][kPhysicalRows];
            int g_slotPhysicalCount[kSlots];
            int64_t g_slotSplit[kSlots][kSplitCount];
            int64_t g_foldedPhysical[kPhysicalRows];
            int64_t g_foldedSplit[kSplitCount];
            int64_t g_peakPhysical[kPhysicalRows];
            int64_t g_peakSplit[kSplitCount];
            const char* g_physicalNames[kPhysicalRows];
            int g_physicalCount = 0;

            /// Detection idiom for the `indexBytes()` accessor the FIRST facet
            /// of every cold container carries (C++17 — no concepts). A facet
            /// without it contributes no index bytes, which is what keeps a
            /// container's derived index counted exactly once.
            template <typename T, typename = void>
            struct HasIndexBytes : std::false_type {};
            template <typename T>
            struct HasIndexBytes<
                T, std::void_t<decltype(std::declval<const T&>().indexBytes())>>
                : std::true_type {};

            template <typename T, typename = void>
            struct HasResidentContentBytes : std::false_type {};
            template <typename T>
            struct HasResidentContentBytes<T, std::void_t<decltype(
                std::declval<const T&>().residentContentBytes())>> : std::true_type {};

            /// One row of the tag naming tables: the container the tag belongs
            /// to and the column inside it.
            struct FacetName {
                uint32_t offset;
                const char* container;
                const char* facet;
            };

            // The four HashMemory instances at 100-tag bases, both ColdMail
            // instances at 50-tag bases, and the three single-instance bands.
            // Offsets are the ones each family's `visitContainersImpl` emits;
            // holes in a band are unused tags that never reach the visitor.
            const FacetName kHashMemoryFacets[] = {
                {0, "encodedMap", "lengths"},
                {1, "encodedMap", "bytes"},
                {2, "encodedMap", "runStarts"},
                {3, "encodedMap", "blobStarts"},
                {4, "encodedMap", "blobPool"},
                {5, "normalizedEncodedKeys", "lengths"},
                {6, "normalizedEncodedKeys", "bytes"},
                {10, "normalizedEncodedSubkeys", "lengths"},
                {11, "normalizedEncodedSubkeys", "bytes"},
                {12, "normalizedEncodedSubkeys", "runStarts"},
                {13, "normalizedEncodedSubkeys", "blobStarts"},
                {14, "normalizedEncodedSubkeys", "blobPool"},
                {25, "admissionMap", "keys"},
                {26, "admissionMap", "runStarts"},
                {27, "admissionMap", "blobStarts"},
                {28, "admissionMap", "blobPool"},
                {29, "admissionStatusMap", "keys"},
                {30, "admissionStatusMap", "values"},
                {32, "varsInAdmissionMapKeys", "keys"},
                {33, "rejectedMap", "keys"},
                {34, "rejectedMap", "runStarts"},
                {35, "rejectedMap", "blobStarts"},
                {36, "rejectedMap", "blobPool"},
                {37, "revisitInProgress", "keys"},
                {38, "admissionMapIntegration", "keys"},
                {39, "admissionMapIntegration", "runStarts"},
                {40, "admissionMapIntegration", "blobStarts"},
                {41, "admissionMapIntegration", "blobPool"},
                {42, "rejectedMapIntegration", "keys"},
                {43, "rejectedMapIntegration", "runStarts"},
                {44, "rejectedMapIntegration", "blobStarts"},
                {45, "rejectedMapIntegration", "blobPool"},
                {46, "varsInAdmissionMapIntegrationKeys", "keys"},
                {47, "varsInRejectedMapIntegrationKeys", "keys"},
                {48, "admissionSetIntegration", "keys"},
                {49, "triggersForAdmissionSetIntegration", "keys"},
                {50, "productsOfRecursionIds", "keys"},
                {51, "originals", "lengths"},
                {52, "originals", "bytes"},
                {53, "remainingArgsNormalizedEncodedMap", "lengths"},
                {54, "remainingArgsNormalizedEncodedMap", "bytes"},
                {55, "remainingArgsNormalizedEncodedMap", "runStarts"},
                {56, "remainingArgsNormalizedEncodedMap", "blobStarts"},
                {57, "remainingArgsNormalizedEncodedMap", "blobPool"},
                {58, "rejectedMapOrdis", "keys"},
                {59, "rejectedMapOrdis", "runStarts"},
                {60, "rejectedMapOrdis", "blobStarts"},
                {61, "rejectedMapOrdis", "blobPool"},
                {62, "ordisRevisitInProgress", "keys"},
                {63, "admissionMapOrdis2", "keys"},
                {64, "admissionMapOrdis2", "runStarts"},
                {65, "admissionMapOrdis2", "blobStarts"},
                {66, "admissionMapOrdis2", "blobPool"},
                {67, "rejectedMapOrdis2", "keys"},
                {68, "rejectedMapOrdis2", "runStarts"},
                {69, "rejectedMapOrdis2", "blobStarts"},
                {70, "rejectedMapOrdis2", "blobPool"},
                {71, "ordis2RevisitInProgress", "keys"},
                {72, "normalizedEncodedKeys", "runStarts"},
                {73, "normalizedEncodedKeys", "blobStarts"},
                {74, "normalizedEncodedKeys", "blobPool"},
                {75, "originals", "runStarts"},
                {76, "originals", "blobStarts"},
                {77, "originals", "blobPool"},
                {78, "remainingArgsOwners", "lengths"},
                {79, "remainingArgsOwners", "bytes"},
                {80, "remainingArgsOwners", "runStarts"},
                {81, "remainingArgsOwners", "blobStarts"},
                {82, "remainingArgsOwners", "blobPool"},
                {83, "copyOwners", "lengths"},
                {84, "copyOwners", "bytes"},
                {85, "copyOwners", "runStarts"},
                {86, "copyOwners", "blobStarts"},
                {87, "copyOwners", "blobPool"},
                {88, "rejectedMapSubproof", "keys"},
                {89, "rejectedMapSubproof", "runStarts"},
                {90, "rejectedMapSubproof", "blobStarts"},
                {91, "rejectedMapSubproof", "blobPool"},
                {92, "subproofRevisitInProgress", "keys"},
            };

            const FacetName kColdMailFacets[] = {
                {0, "statements", "lengths"},
                {1, "statements", "bytes"},
                {2, "origins", "keys"},
                {3, "origins", "runStarts"},
                {4, "origins", "blobStarts"},
                {5, "origins", "blobPool"},
                {6, "disintegrationSignals", "keys"},
                {7, "disintegrationSignals", "values"},
            };

            const FacetName kChangedClassesFacets[] = {
                {0, "changedClassesThisStep", "validityIds"},
                {1, "changedClassesThisStep", "blobStarts"},
                {2, "changedClassesThisStep", "blobPool"},
            };

            const FacetName kNameCacheFacets[] = {
                {0, "kindById", "values"},
                {1, "tokens", "keys"},
                {2, "tokens", "runStarts"},
                {3, "tokens", "blobStarts"},
                {4, "tokens", "blobPool"},
            };

            const FacetName kMailOutFacets[] = {
                {0, "strings", "lengths"},
                {1, "strings", "bytes"},
                {2, "statements", "lengths"},
                {3, "statements", "bytes"},
                {4, "origins", "keys"},
                {5, "origins", "runStarts"},
                {6, "origins", "blobStarts"},
                {7, "origins", "blobPool"},
                {8, "statementFlags", "keys"},
                {9, "statementFlags", "values"},
            };

            // The direct LbMemory tags, in enum order. A tag appended to
            // `ContainerTag` without a row here renders as `tag<N>` rather than
            // silently vanishing, and the SwDD chapter says to add the row.
            const FacetName kLbMemoryTags[] = {
                {0, "intEncodedStatements", ""},
                {1, "intLocalEncodedStatements", ""},
                {2, "intLocalEncodedStatementsDelta", ""},
                {3, "intExternalStatements", ""},
                {4, "templateStrings", "lengths"},
                {5, "templateStrings", "bytes"},
                {6, "valueStrings", "lengths"},
                {7, "valueStrings", "bytes"},
                {8, "originStrings", "lengths"},
                {9, "originStrings", "bytes"},
                {10, "ruleStrings", "lengths"},
                {11, "ruleStrings", "bytes"},
                {12, "lbStateStrings", "lengths"},
                {13, "lbStateStrings", "bytes"},
                {14, "nameStrings", "lengths"},
                {15, "nameStrings", "bytes"},
                {16, "subStrings", "lengths"},
                {17, "subStrings", "bytes"},
                {18, "validityNodes", ""},
                {19, "intValidityNamesToFilter", "keys"},
                {20, "intAxedVariables", "keys"},
                {23, "pendingWipeScopes", "keys"},
                {24, "orDisjunctCount", "keys"},
                {25, "orDisjunctCount", "values"},
                {26, "integrationStartIntMap", "keys"},
                {27, "integrationStartIntMap", "values"},
                {28, "intLocalEncodedStatementsSet", "keys"},
                {29, "intWeakVariables", "keys"},
                {30, "integrationPrepared", "keys"},
                {31, "integrationPreparedMarker", "keys"},
                {32, "expandedImplications", "keys"},
                {33, "intKnownStatements", "keys"},
                {34, "intKnownStatements", "values"},
                {38, "intStatementLevelsMap", "keys"},
                {39, "intStatementLevelsMap", "runStarts"},
                {40, "intStatementLevelsMap", "values"},
                {41, "orBookkeeping", "keys"},
                {42, "orBookkeeping", "runStarts"},
                {43, "orBookkeeping", "values"},
                {44, "eqClassSttmntIndexMapMap", "lengths"},
                {45, "eqClassSttmntIndexMapMap", "bytes"},
                {46, "eqClassSttmntIndexMapMap", "values"},
                {47, "equivalenceClassesMap", "keys"},
                {48, "equivalenceClassesMap", "runStarts"},
                {49, "equivalenceClassesMap", "blobStarts"},
                {50, "equivalenceClassesMap", "blobPool"},
                {451, "exprOriginMap", "keys"},
                {452, "exprOriginMap", "runStarts"},
                {453, "exprOriginMap", "blobStarts"},
                {454, "exprOriginMap", "blobPool"},
                {805, "orPendingBranches", "keys"},
                {806, "orPendingBranches", "runStarts"},
                {807, "orPendingBranches", "values"},
                {808, "orPendingLevels", "keys"},
                {809, "orPendingLevels", "runStarts"},
                {810, "orPendingLevels", "values"},
                {811, "pendingRelayIter", "keys"},
                {812, "pendingRelayIter", "values"},
                {813, "pendingRelayLevels", "keys"},
                {814, "pendingRelayLevels", "runStarts"},
                {815, "pendingRelayLevels", "values"},
                {816, "compactExpansions", "keys"},
                {817, "compactExpansions", "runStarts"},
                {818, "compactExpansions", "blobStarts"},
                {819, "compactExpansions", "blobPool"},
                {820, "expansionCarrierCount", "keys"},
                {821, "expansionCarrierCount", "values"},
                {822, "usedAsPremiseCorePairs", "keys"},
                {824, "appliedPremiseCorePairs", "keys"},
                {826, "closedOrGoalLevels", "keys"},
                {827, "closedOrGoalLevels", "runStarts"},
                {828, "closedOrGoalLevels", "values"},
                {829, "orImplications", "facet0"},
                {830, "orImplications", "facet1"},
                {831, "orImplications", "facet2"},
                {832, "orImplications", "facet3"},
                {833, "orImplications", "facet4"},
                {834, "orImplications", "facet5"},
                {835, "orImplications", "facet6"},
                {836, "orImplications", "facet7"},
                {837, "existenceReplays", "keys"},
                {838, "existenceReplays", "values"},
                {839, "canonicalEquivalenceClassesMap", "keys"},
                {840, "canonicalEquivalenceClassesMap", "runStarts"},
                {841, "canonicalEquivalenceClassesMap", "blobStarts"},
                {842, "canonicalEquivalenceClassesMap", "blobPool"},
                {843, "canonicalMemberRepresentatives", "keys"},
                {844, "canonicalMemberRepresentatives", "values"},
                {845, "canonicalClassChanges", "keys"},
                {846, "canonicalClassChanges", "values"},
                {872, "orSupportFacts", "keys"},
                {873, "orSupportFacts", "values"},
                {874, "orSupportEqualities", "keys"},
                {875, "orSupportEqualities", "runStarts"},
                {876, "orSupportEqualities", "values"},
                {kTagSharedBytes, "sharedNativeKeys", "payloadAndLocations"},
                {kTagSharedExpressions, "sharedExpressions", "recordsAndMetadata"},
            };

            /// One band of tags owned by a repeated family.
            struct Band {
                uint32_t base;
                uint32_t span;
                const char* instance;
                const FacetName* facets;
                int facetCount;
            };

            const Band kBands[] = {
                {51u, 100u, "overallHashMemory", kHashMemoryFacets,
                 static_cast<int>(std::size(kHashMemoryFacets))},
                {151u, 100u, "localHashMemory", kHashMemoryFacets,
                 static_cast<int>(std::size(kHashMemoryFacets))},
                {251u, 100u, "localHashMemoryDelta", kHashMemoryFacets,
                 static_cast<int>(std::size(kHashMemoryFacets))},
                {351u, 100u, "workingMemory", kHashMemoryFacets,
                 static_cast<int>(std::size(kHashMemoryFacets))},
                {455u, 50u, "sameInternalMail", kColdMailFacets,
                 static_cast<int>(std::size(kColdMailFacets))},
                {505u, 50u, "nextInternalMail", kColdMailFacets,
                 static_cast<int>(std::size(kColdMailFacets))},
                {655u, 50u, "changedClasses", kChangedClassesFacets,
                 static_cast<int>(std::size(kChangedClassesFacets))},
                {705u, 50u, "eqClassNameCaches", kNameCacheFacets,
                 static_cast<int>(std::size(kNameCacheFacets))},
                {755u, 50u, "mailOut", kMailOutFacets,
                 static_cast<int>(std::size(kMailOutFacets))},
            };

            const char* const kGlobalNames[kGlobals] = {
                "pool: staticMemory (Main, deloadable per-LB store)",
                "pool: persistentMemory (intToBeProved)",
                "pool: mailMemory (MailLog + routing mailIn)",
                "pool: lbMemory (LbStore shells + skeletonInterner)",
                "scratch: scratchArenas (string tier, peak over slots)",
                "scratch: genScratchArenas (container tier, peak over slots)",
            };

            /// @brief Render a tag as `instance.container.facet` into a buffer.
            ///
            /// @details
            /// Resolution order is band first, then the direct-tag table, then
            /// the synthetic overhead tag. An unrecognised tag renders as
            /// `tag<N>` — a visible prompt to extend `kLbMemoryTags` rather than
            /// a silent omission.
            ///
            /// @param tag The `LbMemory::ContainerTag` value, or the synthetic
            ///            arena-overhead tag.
            /// @param out Destination buffer; always NUL-terminated.
            /// @param cap Capacity of `out` in bytes; must exceed 1.
            void renderTagName(int tag, char* out, int cap) {
                assert(out != nullptr && cap > 1 && "renderTagName: bad buffer");
                const auto write = [out, cap](const char* a, const char* b,
                                              const char* c) {
                    std::string s = a;
                    if (b != nullptr && b[0] != '\0') { s += '.'; s += b; }
                    if (c != nullptr && c[0] != '\0') { s += '.'; s += c; }
                    const int n = std::min(static_cast<int>(s.size()), cap - 1);
                    std::memcpy(out, s.data(), static_cast<size_t>(n));
                    out[n] = '\0';
                };
                const uint32_t t = static_cast<uint32_t>(tag);
                for (const Band& b : kBands) {
                    if (t < b.base || t >= b.base + b.span) continue;
                    const uint32_t off = t - b.base;
                    for (int i = 0; i < b.facetCount; ++i) {
                        if (b.facets[i].offset != off) continue;
                        write(b.instance, b.facets[i].container,
                              b.facets[i].facet);
                        return;
                    }
                    std::string s = std::string(b.instance) + ".offset"
                                  + std::to_string(off);
                    write(s.c_str(), "", "");
                    return;
                }
                for (const FacetName& f : kLbMemoryTags) {
                    if (f.offset != t) continue;
                    write(f.container, f.facet, "");
                    return;
                }
                const std::string s = "tag" + std::to_string(tag);
                write(s.c_str(), "", "");
            }

            /// @brief Strip the trailing `.facet` component of a rendered name.
            ///
            /// @details
            /// The report groups facets under their owning container, so
            /// `overallHashMemory.encodedMap.blobPool` collapses to
            /// `overallHashMemory.encodedMap`. A name with fewer than two dots
            /// is returned unchanged.
            ///
            /// @param full A name produced by `renderTagName`.
            /// @return The owning structure's name.
            std::string structOf(const std::string& full) {
                const size_t first = full.find('.');
                if (first == std::string::npos) return full;
                const size_t last = full.rfind('.');
                if (last == first) return full;
                return full.substr(0, last);
            }

        } // namespace

        void resetIteration() {
            std::memset(g_slotBytes, 0, sizeof(g_slotBytes));
            std::memset(g_slotIndex, 0, sizeof(g_slotIndex));
            std::memset(g_slotSlack, 0, sizeof(g_slotSlack));
            std::memset(g_slotPhysical, 0, sizeof(g_slotPhysical));
            std::memset(g_slotSplit, 0, sizeof(g_slotSplit));
        }

        void addLbSample(const LbMemory& lb, unsigned slot) {
            assert(slot < static_cast<unsigned>(kSlots)
                   && "mem_tracker: worker slot exceeds MEM_MAX_SLOTS");
            int64_t* row = g_slotBytes[slot];
            int64_t* irow = g_slotIndex[slot];
            int64_t logical = 0;
            int64_t indexes = 0;
            lb.visitContainers(
                [row, irow, &logical, &indexes](LbMemory::ContainerTag tag,
                                                const auto& c) {
                    using C = std::decay_t<decltype(c)>;
                    using T = typename C::value_type;
                    const int idx = static_cast<int>(tag);
                    assert(idx >= 0 && idx < kTagSlack
                           && "mem_tracker: ContainerTag exceeds MEM_TAG_SPACE");
                    int64_t bytes = static_cast<int64_t>(sizeof(T))
                        * static_cast<int64_t>(c.size());
                    if constexpr (HasResidentContentBytes<C>::value)
                        bytes = c.residentContentBytes();
                    row[idx] += bytes;
                    logical += bytes;
                    // Only a container's FIRST facet carries indexBytes(), so a
                    // derived index is attributed once, to the tag that names
                    // the container it belongs to.
                    if constexpr (HasIndexBytes<C>::value) {
                        const int64_t ib = c.indexBytes();
                        irow[idx] += ib;
                        indexes += ib;
                    }
                });
            const int64_t sharedBytes = lb.sharedNativeKeys.logicalByteCount()
                + static_cast<int64_t>(lb.sharedNativeKeys.count()) * sizeof(ColdStringLocation);
            const int64_t sharedIndex = lb.sharedNativeKeys.indexBytes();
            row[kTagSharedBytes] += sharedBytes;
            irow[kTagSharedBytes] += sharedIndex;
            logical += sharedBytes;
            indexes += sharedIndex;
            const int64_t expressionBytes = lb.sharedExpressions.residentContentBytes();
            const int64_t expressionIndex = lb.sharedExpressions.indexBytes();
            row[kTagSharedExpressions] += expressionBytes;
            irow[kTagSharedExpressions] += expressionIndex;
            logical += expressionBytes;
            indexes += expressionIndex;
            // The residual includes padding, parked pages, bookkeeping and
            // useful derived storage absent from this content enumeration.
            const int64_t physical = lb.manager.blocksHeld()
                                   * static_cast<int64_t>(lb.manager.blockBytes());
            const int64_t accounted = logical + indexes;
            if (physical > accounted) g_slotSlack[slot] += physical - accounted;

            // Physical view: live pages, free pages parked inside held blocks,
            // and whatever else the held blocks carry (byte-bump blocks and
            // spilled block directories), then each container's held pages.
            const int64_t pageBytes = lb.manager.pageBytes();
            const int64_t livePageBytes =
                static_cast<int64_t>(lb.manager.livePages()) * pageBytes;
            const int64_t freePageBytes =
                static_cast<int64_t>(lb.manager.freePageCount()) * pageBytes;
            int64_t* split = g_slotSplit[slot];
            split[kLivePages] += livePageBytes;
            split[kFreePages] += freePageBytes;
            split[kOtherBlocks] += physical - livePageBytes - freePageBytes;
            split[kLbs] += 1;
            int64_t* prow = g_slotPhysical[slot];
            const char** names = g_slotPhysicalNames[slot];
            int count = 0;
            lb.visitLiveBytes([prow, names, &count](const char* name, int64_t bytes) {
                assert(count < kPhysicalRows
                    && "mem_tracker: more visitLiveBytes terms than kPhysicalRows");
                assert((names[count] == nullptr || std::strcmp(names[count], name) == 0)
                    && "mem_tracker: visitLiveBytes order changed between samples");
                names[count] = name;
                prow[count] += bytes;
                ++count;
            });
            g_slotPhysicalCount[slot] = count;
        }

        void commitIterationSample() {
            std::memset(g_folded, 0, sizeof(g_folded));
            std::memset(g_foldedIndex, 0, sizeof(g_foldedIndex));
            std::memset(g_foldedPhysical, 0, sizeof(g_foldedPhysical));
            std::memset(g_foldedSplit, 0, sizeof(g_foldedSplit));
            g_foldedSlack = 0;
            for (int s = 0; s < kSlots; ++s) {
                for (int t = 0; t < kTagSpace; ++t) {
                    g_folded[t] += g_slotBytes[s][t];
                    g_foldedIndex[t] += g_slotIndex[s][t];
                }
                g_foldedSlack += g_slotSlack[s];
                for (int r = 0; r < kSplitCount; ++r)
                    g_foldedSplit[r] += g_slotSplit[s][r];
                for (int r = 0; r < g_slotPhysicalCount[s]; ++r) {
                    g_foldedPhysical[r] += g_slotPhysical[s][r];
                    if (g_physicalNames[r] == nullptr)
                        g_physicalNames[r] = g_slotPhysicalNames[s][r];
                }
                g_physicalCount = std::max(g_physicalCount, g_slotPhysicalCount[s]);
            }

            const auto poolBytes = [](GlobalMemoryManager& m) -> int64_t {
                if (!m.initialized()) return 0;
                return m.blocksInUse() * static_cast<int64_t>(m.blockBytes());
            };
            g_globals[static_cast<int>(GlobalSlot::PoolMain)] =
                poolBytes(staticMemory());
            g_globals[static_cast<int>(GlobalSlot::PoolPersistent)] =
                poolBytes(persistentMemory());
            g_globals[static_cast<int>(GlobalSlot::PoolMail)] =
                poolBytes(mailMemory());
            g_globals[static_cast<int>(GlobalSlot::PoolLb)] =
                poolBytes(lbMemory());

            const auto registryPeak = [](ScratchArenaRegistry& r) -> int64_t {
                int64_t sum = 0;
                for (unsigned i = 0; i < r.slotCount(); ++i)
                    sum += r.forSlot(i).peakUsedBytes();
                return sum;
            };
            g_globals[static_cast<int>(GlobalSlot::ScratchString)] =
                registryPeak(scratchArenas());
            g_globals[static_cast<int>(GlobalSlot::ScratchGen)] =
                registryPeak(genScratchArenas());

            int64_t total = g_foldedSlack;
            for (int t = 0; t < kTagSpace; ++t)
                total += g_folded[t] + g_foldedIndex[t];
            ++g_samples;
            if (total > g_peakTotal) {
                g_peakTotal = total;
                g_peakSample = g_samples;
                g_peakSlack = g_foldedSlack;
                std::memcpy(g_peak, g_folded, sizeof(g_peak));
                std::memcpy(g_peakIndex, g_foldedIndex, sizeof(g_peakIndex));
                std::memcpy(g_peakGlobals, g_globals, sizeof(g_peakGlobals));
                std::memcpy(g_peakPhysical, g_foldedPhysical, sizeof(g_peakPhysical));
                std::memcpy(g_peakSplit, g_foldedSplit, sizeof(g_peakSplit));
            }
        }

        int64_t sampleCount() { return g_samples; }

        int64_t peakTotalBytes() { return g_peakTotal; }

        void resetAllForTest() {
            std::memset(g_slotBytes, 0, sizeof(g_slotBytes));
            std::memset(g_slotIndex, 0, sizeof(g_slotIndex));
            std::memset(g_slotSlack, 0, sizeof(g_slotSlack));
            std::memset(g_folded, 0, sizeof(g_folded));
            std::memset(g_foldedIndex, 0, sizeof(g_foldedIndex));
            std::memset(g_peak, 0, sizeof(g_peak));
            std::memset(g_peakIndex, 0, sizeof(g_peakIndex));
            std::memset(g_globals, 0, sizeof(g_globals));
            std::memset(g_peakGlobals, 0, sizeof(g_peakGlobals));
            g_foldedSlack = 0;
            g_peakSlack = 0;
            g_peakTotal = 0;
            g_samples = 0;
            g_peakSample = -1;
            std::memset(g_slotPhysical, 0, sizeof(g_slotPhysical));
            std::memset(g_slotPhysicalNames, 0, sizeof(g_slotPhysicalNames));
            std::memset(g_slotPhysicalCount, 0, sizeof(g_slotPhysicalCount));
            std::memset(g_slotSplit, 0, sizeof(g_slotSplit));
            std::memset(g_foldedPhysical, 0, sizeof(g_foldedPhysical));
            std::memset(g_foldedSplit, 0, sizeof(g_foldedSplit));
            std::memset(g_peakPhysical, 0, sizeof(g_peakPhysical));
            std::memset(g_peakSplit, 0, sizeof(g_peakSplit));
            std::memset(g_physicalNames, 0, sizeof(g_physicalNames));
            g_physicalCount = 0;
        }

        /// @brief Read the physical split of the peak sample.
        ///
        /// @return The peak's physical split; zero before the first commit.
        PhysicalPeak physicalPeak() {
            PhysicalPeak p;
            p.livePageBytes = g_peakSplit[kLivePages];
            p.freePageBytes = g_peakSplit[kFreePages];
            p.otherBlockBytes = g_peakSplit[kOtherBlocks];
            p.logicBlocks = g_peakSplit[kLbs];
            int64_t content = 0, index = 0;
            for (int t = 0; t < kTagSpace; ++t) { content += g_peak[t]; index += g_peakIndex[t]; }
            p.contentBytes = content;
            p.indexBytes = index;
            p.containerCount = g_physicalCount;
            return p;
        }

        /// @brief Held-page bytes of one container at the peak sample.
        ///
        /// @param name A `LbMemory::visitLiveBytes` container name.
        /// @return The container's held-page bytes at the peak; asserts when
        ///         no term carries that name.
        int64_t physicalPeakBytesOf(const char* name) {
            for (int r = 0; r < g_physicalCount; ++r)
                if (g_physicalNames[r] != nullptr && std::strcmp(g_physicalNames[r], name) == 0)
                    return g_peakPhysical[r];
            assert(false && "mem_tracker: no visitLiveBytes term with that name");
            return -1;
        }

        namespace {

            /// One node of the rendered structure hierarchy: an instance, a
            /// container inside it, or a facet inside that.
            struct Node {
                std::string name;
                int64_t content = 0;
                int64_t index = 0;
                std::vector<Node> kids;
                int64_t total() const { return content + index; }
            };

            /// @brief Find or append a child node by name.
            ///
            /// @param parent Node to search.
            /// @param name   Child's display name.
            /// @return Reference to the child, created empty if absent.
            Node& childNamed(Node& parent, const std::string& name) {
                for (Node& k : parent.kids)
                    if (k.name == name) return k;
                parent.kids.push_back(Node{name, 0, 0, {}});
                return parent.kids.back();
            }

            /// @brief Sort a subtree by descending total and roll child sums up.
            ///
            /// @details
            /// A container node carries its own index bytes (booked at its first
            /// facet) plus the content of every facet below it, so the roll-up
            /// runs bottom-up before the sort.
            ///
            /// @param n Subtree root, modified in place.
            void rollUpAndSort(Node& n) {
                for (Node& k : n.kids) {
                    rollUpAndSort(k);
                    n.content += k.content;
                    n.index += k.index;
                }
                std::stable_sort(n.kids.begin(), n.kids.end(),
                                 [](const Node& a, const Node& b) {
                                     return a.total() > b.total();
                                 });
            }

            /// @brief Build the whole structure tree from the peak snapshot.
            ///
            /// @details
            /// A three-part rendered tag (`instance.container.facet`) nests
            /// three deep; a two-part or one-part tag is a direct `LbMemory`
            /// member and hangs under a synthetic instance node so every leaf
            /// sits at the same depth.
            ///
            /// @return Root node whose children are the instances.
            Node buildTree() {
                Node root{"peak", 0, 0, {}};
                for (int t = 0; t < kTagSpace; ++t) {
                    if (g_peak[t] == 0 && g_peakIndex[t] == 0) continue;
                    char buf[192];
                    renderTagName(t, buf, static_cast<int>(sizeof(buf)));
                    const std::string full(buf);
                    std::string a, b, c;
                    const size_t d1 = full.find('.');
                    const size_t d2 = (d1 == std::string::npos)
                                          ? std::string::npos
                                          : full.find('.', d1 + 1);
                    if (d2 != std::string::npos) {
                        a = full.substr(0, d1);
                        b = full.substr(d1 + 1, d2 - d1 - 1);
                        c = full.substr(d2 + 1);
                    } else if (d1 != std::string::npos) {
                        a = "LbMemory (direct members)";
                        b = full.substr(0, d1);
                        c = full.substr(d1 + 1);
                    } else {
                        a = "LbMemory (direct members)";
                        b = full;
                        c = "(single column)";
                    }
                    Node& inst = childNamed(root, a);
                    Node& cont = childNamed(inst, b);
                    Node& fac = childNamed(cont, c);
                    fac.content += g_peak[t];
                    // The derived index belongs to the CONTAINER, not to the
                    // facet that happened to report it, so it is booked one
                    // level up and never shown as a facet's own bytes.
                    cont.index += g_peakIndex[t];
                }
                rollUpAndSort(root);
                return root;
            }

            /// @brief Sum of every container's content bytes at the peak.
            ///
            /// @return Logical content bytes.
            int64_t peakContentBytes() {
                int64_t v = 0;
                for (int t = 0; t < kTagSpace; ++t) v += g_peak[t];
                return v;
            }

            /// @brief Sum of every derived index at the peak.
            ///
            /// @return Derived-index bytes.
            int64_t peakIndexBytes() {
                int64_t v = 0;
                for (int t = 0; t < kTagSpace; ++t) v += g_peakIndex[t];
                return v;
            }

            /// @brief Format a byte count as MiB with two decimals.
            ///
            /// @param b Bytes.
            /// @return Rendered MiB string.
            std::string mib(int64_t b) {
                std::ostringstream o;
                o << std::fixed << std::setprecision(2)
                  << static_cast<double>(b) / (1024.0 * 1024.0);
                return o.str();
            }

            /// @brief Format a share as a percentage with two decimals.
            ///
            /// @param part  Numerator bytes.
            /// @param whole Denominator bytes; zero yields "0.00".
            /// @return Rendered percentage string.
            std::string pct(int64_t part, int64_t whole) {
                std::ostringstream o;
                o << std::fixed << std::setprecision(2)
                  << (whole > 0 ? 100.0 * static_cast<double>(part)
                                      / static_cast<double>(whole)
                                : 0.0);
                return o.str();
            }

            /// @brief Open the destination, creating the parent directory.
            ///
            /// @details
            /// Asserts on any failure rather than returning a sentinel
            /// (Rule 19).
            ///
            /// @param path Destination file path.
            /// @param out  Stream to open.
            void openForWrite(const std::string& path, std::ofstream& out) {
                std::error_code ec;
                const std::filesystem::path p(path);
                if (p.has_parent_path()) {
                    std::filesystem::create_directories(p.parent_path(), ec);
                    assert(!ec && "mem_tracker: cannot create output directory");
                }
                out.open(p, std::ios::trunc);
                assert(out.is_open() && "mem_tracker: cannot open output file");
            }

            /// @brief Emit one subtree as nested HTML disclosure rows.
            ///
            /// @param o     Sink.
            /// @param n     Subtree root.
            /// @param depth Nesting depth, 0 for an instance.
            /// @param peak  Peak grand total, the `%peak` denominator.
            /// @param cont  Peak content total, the `%content` denominator.
            void emitHtmlNode(std::ostringstream& o, const Node& n, int depth,
                              int64_t peak, int64_t cont) {
                const std::string cls = "d" + std::to_string(depth);
                const bool leaf = n.kids.empty();
                const std::string bar = pct(n.total(), peak);
                if (!leaf) o << "<details" << (depth == 0 ? " open" : "") << ">";
                o << (leaf ? "<div class='row leaf " : "<summary class='row ")
                  << cls << "'>"
                  << "<span class='nm'>" << n.name << "</span>"
                  << "<span class='v'>" << mib(n.content) << "</span>"
                  << "<span class='v'>"
                  << (n.index > 0 ? mib(n.index) : std::string("-")) << "</span>"
                  << "<span class='v tot'>" << mib(n.total()) << "</span>"
                  << "<span class='v'>" << pct(n.total(), peak) << "</span>"
                  << "<span class='v'>" << pct(n.content, cont) << "</span>"
                  << "<span class='bar'><i style='width:" << bar
                  << "%'></i></span>"
                  << (leaf ? "</div>" : "</summary>");
                for (const Node& k : n.kids)
                    emitHtmlNode(o, k, depth + 1, peak, cont);
                if (!leaf) o << "</details>";
            }

            /// @brief Emit one subtree as indented plain-text rows.
            ///
            /// @param o     Sink.
            /// @param n     Subtree root.
            /// @param depth Nesting depth, 0 for an instance.
            /// @param peak  Peak grand total.
            /// @param cont  Peak content total.
            void emitTextNode(std::ostringstream& o, const Node& n, int depth,
                              int64_t peak, int64_t cont) {
                std::string pad(static_cast<size_t>(depth) * 2, ' ');
                std::string label = pad + n.name;
                if (label.size() > 58) label = label.substr(0, 58);
                o << std::left << std::setw(58) << label << " | " << std::right
                  << std::setw(9) << mib(n.total()) << " | " << std::setw(8)
                  << pct(n.total(), cont) << " | " << std::setw(6)
                  << pct(n.total(), peak) << "\n";
                for (const Node& k : n.kids)
                    emitTextNode(o, k, depth + 1, peak, cont);
            }

        } // namespace

        void dumpMemAggregate(const std::string& path) {
            const Node root = buildTree();
            const int64_t content = peakContentBytes();
            const int64_t index = peakIndexBytes();

            std::ostringstream o;
            o << "Per-structure static memory at the peak end-of-burst sample\n\n"
              << "Iteration samples committed : " << g_samples << "\n"
              << "Peak sample                 : #" << g_peakSample << "\n"
              << "Peak attributed bytes       : " << g_peakTotal << "  ("
              << mib(g_peakTotal) << " MiB)\n"
              << "  container content         : " << content << "  ("
              << mib(content) << " MiB, " << pct(content, g_peakTotal) << " %)\n"
              << "  derived hash indexes      : " << index << "  ("
              << mib(index) << " MiB, " << pct(index, g_peakTotal) << " %)\n"
              << "  block + page slack        : " << g_peakSlack << "  ("
              << mib(g_peakSlack) << " MiB, " << pct(g_peakSlack, g_peakTotal)
              << " %)\n\n"
              << std::left << std::setw(58) << "Structure"
              << " |       MiB | %content | %peak\n"
              << std::string(58, '-') << "-+-----------+----------+------\n";
            for (const Node& k : root.kids)
                emitTextNode(o, k, 0, g_peakTotal, content);
            {
                const PhysicalPeak pp = physicalPeak();
                const int64_t pinned = pp.livePageBytes + pp.freePageBytes + pp.otherBlockBytes;
                const int64_t padding = pp.livePageBytes - pp.contentBytes - pp.indexBytes;
                o << "\nPhysical split of the peak sample's pinned blocks ("
                  << pp.logicBlocks << " logic blocks sampled)\n\n"
                  << "  pinned blocks               : " << mib(pinned) << " MiB\n"
                  << "    live pages                : " << mib(pp.livePageBytes) << " MiB  ("
                  << pct(pp.livePageBytes, pinned) << " %)\n"
                  << "      content                 : " << mib(pp.contentBytes) << " MiB\n"
                  << "      derived indexes         : " << mib(pp.indexBytes) << " MiB\n"
                  << "      page padding            : " << mib(padding) << " MiB  ("
                  << pct(padding, pp.livePageBytes) << " % of live pages)\n"
                  << "    free pages in held blocks : " << mib(pp.freePageBytes) << " MiB  ("
                  << pct(pp.freePageBytes, pinned) << " %)\n"
                  << "    other block bytes         : " << mib(pp.otherBlockBytes) << " MiB  ("
                  << pct(pp.otherBlockBytes, pinned) << " %)\n\n"
                  << "Held pages per container (visitLiveBytes; pages incl. directories and\n"
                     "hash-index pages), share of live pages\n\n";
                int order[kPhysicalRows];
                for (int r = 0; r < g_physicalCount; ++r) order[r] = r;
                std::stable_sort(order, order + g_physicalCount, [](int a, int b) {
                    return g_peakPhysical[a] > g_peakPhysical[b];
                });
                for (int i = 0; i < g_physicalCount; ++i) {
                    const int r = order[i];
                    if (g_peakPhysical[r] == 0) continue;
                    o << "  " << std::left << std::setw(40)
                      << (g_physicalNames[r] != nullptr ? g_physicalNames[r] : "?")
                      << std::right << std::setw(10) << mib(g_peakPhysical[r]) << " MiB  "
                      << std::setw(6) << pct(g_peakPhysical[r], pp.livePageBytes) << " %\n";
                }
            }
            o << "\nProcess-wide footprint at the same sample (PHYSICAL block\n"
                 "bytes — a different, strictly larger denominator; NOT folded\n"
                 "into the percentages).\n\n";
            for (int i = 0; i < kGlobals; ++i)
                o << "  " << std::left << std::setw(56) << kGlobalNames[i] << " "
                  << std::right << std::setw(12) << g_peakGlobals[i] << "  ("
                  << mib(g_peakGlobals[i]) << " MiB)\n";

            std::ofstream f;
            openForWrite(path, f);
            f << o.str();
            f.flush();
            assert(f.good() && "mem_tracker: write failed");
        }

        void dumpMemHtml(const std::string& path) {
            const Node root = buildTree();
            const int64_t content = peakContentBytes();
            const int64_t index = peakIndexBytes();

            std::ostringstream o;
            o << "<!doctype html>\n<html><head><meta charset='utf-8'>\n"
              << "<title>GL static memory - peak " << mib(g_peakTotal)
              << " MiB</title>\n<style>\n"
                 "body{font:13px/1.45 ui-monospace,Consolas,monospace;margin:24px;"
                 "background:#0f1115;color:#d7dae0}"
                 "h1{font-size:17px;margin:0 0 4px}h2{font-size:14px;margin:26px 0 8px;"
                 "color:#9aa4b2}"
                 ".sub{color:#8b93a1;margin-bottom:18px}"
                 "table.k{border-collapse:collapse;margin:8px 0 4px}"
                 "table.k td{padding:2px 14px 2px 0}"
                 "table.k td.n{text-align:right;color:#e6c07b}"
                 ".head,.row{display:flex;align-items:center;gap:6px}"
                 ".head{color:#8b93a1;border-bottom:1px solid #2a2f3a;padding-bottom:4px;"
                 "margin-bottom:2px}"
                 ".nm{flex:0 0 460px;overflow:hidden;text-overflow:ellipsis;"
                 "white-space:nowrap}"
                 ".v{flex:0 0 84px;text-align:right}"
                 ".tot{color:#e6c07b}"
                 ".bar{flex:1 1 120px;height:9px;background:#1b1f28;border-radius:2px;"
                 "overflow:hidden}"
                 ".bar i{display:block;height:100%;background:#4c8eda}"
                 "summary{cursor:pointer;list-style:none}"
                 "summary::-webkit-details-marker{display:none}"
                 "summary:hover{background:#161a22}"
                 "details{margin:0}"
                 "details>details,details>.row{margin-left:14px;"
                 "border-left:1px solid #232833;padding-left:8px}"
                 ".d0>.nm{color:#98c379}.d1>.nm{color:#d7dae0}.d2>.nm{color:#8b93a1}"
                 ".leaf{padding:1px 0}"
                 "</style></head><body>\n";

            o << "<h1>GL static memory &mdash; per-structure, peak end-of-burst "
                 "sample</h1>\n<div class='sub'>Every logic block active in one "
                 "proveKernel iteration, sampled at the end of its burst; the "
                 "fullest of "
              << g_samples << " samples (#" << g_peakSample << ").</div>\n";

            o << "<table class='k'>"
              << "<tr><td>peak attributed</td><td class='n'>" << mib(g_peakTotal)
              << " MiB</td><td class='n'>100.00 %</td>"
                 "<td>physical blocks pinned by the active LBs</td></tr>"
              << "<tr><td>container content</td><td class='n'>" << mib(content)
              << " MiB</td><td class='n'>" << pct(content, g_peakTotal)
              << " %</td><td>element bytes, what the deload image stores</td></tr>"
              << "<tr><td>derived hash indexes</td><td class='n'>" << mib(index)
              << " MiB</td><td class='n'>" << pct(index, g_peakTotal)
              << " %</td><td>PagedHashIndex buckets, rebuilt on reload, never "
                 "deloaded</td></tr>"
              << "<tr><td>block + page slack</td><td class='n'>"
              << mib(g_peakSlack) << " MiB</td><td class='n'>"
              << pct(g_peakSlack, g_peakTotal)
              << " %</td><td>unused tail of every pinned block and touched "
                 "page</td></tr>"
              << "</table>\n";

            o << "<h2>Structures</h2>\n"
              << "<div class='head row'><span class='nm'>structure</span>"
                 "<span class='v'>MiB</span><span class='v'>%content</span>"
                 "<span class='v'>%peak</span><span class='bar'></span></div>\n";
            for (const Node& k : root.kids)
                emitHtmlNode(o, k, 0, g_peakTotal, content);

            o << "<h2>Process-wide pools and scratch</h2>\n<div class='sub'>"
                 "PHYSICAL block bytes &mdash; a different, strictly larger "
                 "denominator than the structure table; deliberately not folded "
                 "into its percentages.</div>\n<table class='k'>";
            for (int i = 0; i < kGlobals; ++i)
                o << "<tr><td>" << kGlobalNames[i] << "</td><td class='n'>"
                  << mib(g_peakGlobals[i]) << " MiB</td></tr>";
            o << "</table>\n</body></html>\n";

            std::ofstream f;
            openForWrite(path, f);
            f << o.str();
            f.flush();
            assert(f.good() && "mem_tracker: write failed");
        }
    } // namespace mem_tracker
} // namespace gl
