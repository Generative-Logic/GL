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

#include "lb_deload.hpp"

#include "deload_stats.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <type_traits>

namespace gl {
    namespace lbdeload {

        namespace {

            /// @brief One row of the in-header container directory:
            ///        which container (tag), element width, element count.
            struct DirectoryEntry {
                uint32_t tag;
                uint32_t elemSize;
                uint64_t elemCount;
            };

            /// @brief Append a fixed-width little-endian value to a byte
            ///        buffer.
            ///
            /// @details
            /// All GL targets (x64 Windows / Linux) are little-endian, so
            /// a raw `memcpy` of the in-memory representation IS the
            /// little-endian encoding; the helper exists to name the
            /// contract at every write site.
            ///
            /// @param buffer Destination byte buffer.
            /// @param value  Trivially-copyable value to append.
            template <typename V>
            void appendRaw(std::vector<char>& buffer, const V& value) {
                static_assert(std::is_trivially_copyable<V>::value,
                              "raw-append needs a trivially copyable value");
                const size_t base = buffer.size();
                buffer.resize(base + sizeof(V));
                std::memcpy(buffer.data() + base, &value, sizeof(V));
            }

            /// @brief Read a fixed-width little-endian value from a byte
            ///        buffer, advancing the offset.
            ///
            /// @param buffer Source bytes.
            /// @param offset In/out read position; asserted in range.
            /// @return The decoded value.
            template <typename V>
            V readRaw(const std::vector<char>& buffer, size_t& offset) {
                static_assert(std::is_trivially_copyable<V>::value,
                              "raw-read needs a trivially copyable value");
                assert(offset + sizeof(V) <= buffer.size());
                V value;
                std::memcpy(&value, buffer.data() + offset, sizeof(V));
                offset += sizeof(V);
                return value;
            }

            /// @brief Build the per-file header bytes (everything before
            ///        the payload chunk).
            ///
            /// @param chain     Full LB chain string (stored verbatim).
            /// @param kind      0 = base image, 1 = tail delta.
            /// @param ordinal   The LB's deload ordinal (header identity
            ///                  field; >= 0).
            /// @param part      1-based part index.
            /// @param partCount Total parts.
            /// @param blockBytes Block size recorded for sanity.
            /// @param entries   The container directory, tag order.
            /// @return The serialized header.
            std::vector<char> buildHeader(
                const std::string& chain, uint32_t kind, int64_t ordinal,
                uint32_t part, uint32_t partCount, int32_t blockBytes,
                const std::vector<DirectoryEntry>& entries) {
                std::vector<char> header;
                header.insert(header.end(), kMagic, kMagic + 4);
                appendRaw(header, kVersion);
                appendRaw(header, kind);
                appendRaw(header, ordinal);
                appendRaw(header, part);
                appendRaw(header, partCount);
                appendRaw(header, static_cast<uint32_t>(blockBytes));
                appendRaw(header,
                          static_cast<uint32_t>(entries.size()));
                for (const DirectoryEntry& e : entries) {
                    appendRaw(header, e.tag);
                    appendRaw(header, e.elemSize);
                    appendRaw(header, e.elemCount);
                }
                appendRaw(header, static_cast<uint32_t>(chain.size()));
                header.insert(header.end(), chain.begin(), chain.end());
                return header;
            }

            /// @brief Split a payload stream into self-describing files
            ///        of at most `maxPayloadBytes` and write them — the
            ///        shared back half of the base and tail dumps.
            ///
            /// @param payload         The canonical element stream.
            /// @param entries         Per-tag directory for this set.
            /// @param chain           Full LB chain string (header verbatim).
            /// @param ordinal         The LB's deload ordinal (file name +
            ///                        header identity).
            /// @param kind            0 = base image, 1 = tail delta.
            /// @param tailIndex       0 for base; the tail set index else.
            /// @param blockBytes      Block size for the header.
            /// @param directory       Target directory (exists).
            /// @param maxPayloadBytes Chunk bound; > 0.
            /// @return The written file names, part order 1..N.
            std::vector<std::string> writeFileSet(
                const std::vector<char>& payload,
                const std::vector<DirectoryEntry>& entries,
                const std::string& chain, int64_t ordinal, uint32_t kind,
                uint32_t tailIndex, int32_t blockBytes,
                const std::filesystem::path& directory,
                int32_t maxPayloadBytes) {
                const uint64_t total = payload.size();
                const uint32_t partCount = total == 0
                    ? 1u
                    : static_cast<uint32_t>(
                        (total + static_cast<uint64_t>(maxPayloadBytes) - 1)
                        / static_cast<uint64_t>(maxPayloadBytes));
                std::vector<std::string> fileNames;
                for (uint32_t part = 1; part <= partCount; ++part) {
                    const std::vector<char> header = buildHeader(
                        chain, kind, ordinal, part, partCount, blockBytes,
                        entries);
                    const uint64_t begin =
                        static_cast<uint64_t>(part - 1) * maxPayloadBytes;
                    const uint64_t end =
                        std::min<uint64_t>(total, begin + maxPayloadBytes);
                    const std::string name =
                        deloadFileName(ordinal, tailIndex, part, partCount);
                    std::ofstream out(directory / name,
                                      std::ios::binary | std::ios::trunc);
                    assert(out && "deload file not writable");
                    out.write(header.data(),
                              static_cast<std::streamsize>(header.size()));
                    if (end > begin) {
                        out.write(payload.data() + begin,
                                  static_cast<std::streamsize>(end - begin));
                    }
                    assert(out && "deload file write failed");
                    fileNames.push_back(name);
                }
                return fileNames;
            }

        }

        /// @brief Compose a deload file name:
        ///        `lb<ordinal>_[t<k>_]<part>_of_<partCount>.bin`.
        ///
        /// @details
        /// The `ordinal` is the file's injective identity (it replaced the
        /// FNV chain hash, so a name collision is impossible); the full
        /// chain lives in the header and `registry.txt`, not the name.
        ///
        /// @param ordinal   The LB's deload ordinal; >= 0.
        /// @param tailIndex 0 for the base image; k >= 1 for tail set k.
        /// @param part      1-based part index.
        /// @param partCount Total parts of this LB's dump.
        /// @return The file name (no directory).
        std::string deloadFileName(int64_t ordinal, uint32_t tailIndex,
                                   uint32_t part, uint32_t partCount) {
            assert(ordinal >= 0);
            assert(part >= 1 && part <= partCount);
            const std::string tail = tailIndex == 0
                ? std::string()
                : "t" + std::to_string(tailIndex) + "_";
            return "lb" + std::to_string(ordinal) + "_" + tail
                + std::to_string(part) + "_of_" + std::to_string(partCount)
                + ".bin";
        }

        /// @brief Serialize an `LbMemory` to its deload file set.
        ///
        /// @details
        /// The canonical-bytes operation
        /// (I-103): containers are streamed
        /// element-by-element in logical order, containers in tag order —
        /// the byte stream is a pure function of logical content,
        /// independent of page fragmentation and allocation history. This
        /// IS the approved "straightening": reload rebuilds a fresh
        /// consecutive virtual index from the stream.
        ///
        /// Every file of the set is self-describing: header (magic,
        /// version, ordinal, part n / N, page bytes, per-tag container
        /// directory, full chain string) followed by that part's chunk of
        /// the payload stream. The payload is split into chunks of at most
        /// `maxPayloadBytes` (production passes the block size — one file
        /// per repacked block, per the approved naming model). An empty
        /// aggregate still writes one header-only file.
        ///
        /// Does NOT release the LB's pages — the caller composes dump +
        /// release + markDeloaded (the `Memory`-level deload lands with
        /// the lifecycle wiring).
        ///
        /// @param lb              The aggregate to serialize (unchanged).
        /// @param chain           The full LB chain string (header verbatim).
        /// @param ordinal         The LB's deload ordinal (file name +
        ///                        header identity).
        /// @param directory       Target directory (created if absent).
        /// @param maxPayloadBytes Maximum payload bytes per file; > 0.
        /// @return The written file names, part order 1..N.
        std::vector<std::string> dumpLbMemory(
            const LbMemory& lb, const std::string& chain, int64_t ordinal,
            const std::filesystem::path& directory,
            int32_t maxPayloadBytes,
            const std::vector<const DeloadColumn*>& extraColumns) {
            assert(maxPayloadBytes > 0);
            const auto dumpStart = std::chrono::steady_clock::now();
            namespace fs = std::filesystem;
            std::error_code ec;
            fs::create_directories(directory, ec);
            assert(!ec);

            std::vector<DirectoryEntry> entries;
            lb.visitContainers(
                [&entries](LbMemory::ContainerTag tag, const auto& c) {
                    using T = typename std::decay_t<decltype(c)>::value_type;
                    entries.push_back(
                        { static_cast<uint32_t>(tag),
                          static_cast<uint32_t>(sizeof(T)),
                          static_cast<uint64_t>(c.size()) });
                });
            // Cold containers outside LbMemory (the cold HashMemory instances),
            // tag-ordered after LbMemory's (their tags are 51+).
            for (const DeloadColumn* col : extraColumns)
                entries.push_back(
                    { col->tag(), col->elemSize(),
                      static_cast<uint64_t>(col->size()) });

            std::vector<char> payload;
            lb.visitContainers(
                [&payload](LbMemory::ContainerTag, const auto& c) {
                    c.appendSpanBytes(payload, 0);
                });
            for (const DeloadColumn* col : extraColumns)
                col->appendSpanBytes(payload, 0);

            std::vector<std::string> files =
                writeFileSet(payload, entries, chain, ordinal,
                             /*kind=*/0u, /*tailIndex=*/0u,
                             lb.manager.blockBytes(), directory,
                             maxPayloadBytes);
            deloadStats().recordV3Dump(
                static_cast<int64_t>(payload.size()),
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - dumpStart).count());
            return files;
        }

        /// @brief Serialize only the rows appended since the last dump —
        ///        one tail-delta file set (the WAL/journal pattern).
        ///
        /// @details
        /// Legal only after an appended-only window
        /// (`DirtyState::AppendedOnly`): for each container, streams the
        /// rows from `startCounts[tag]` to its current size, in tag
        /// order, into a `kind = 1` file set named with `tailIndex`.
        /// Reload appends tail sets onto the base in recorded order, so
        /// base + tails reproduce the exact content a full dump would
        /// have written — the canonical-bytes property holds per
        /// compaction epoch (every full dump restores per-state
        /// canonical form). Asserts each container's size is >= its
        /// recorded start (an appended-only window cannot shrink).
        ///
        /// @param lb              The aggregate to serialize (unchanged).
        /// @param chain           The full LB chain string (header verbatim).
        /// @param ordinal         The LB's deload ordinal (file name +
        ///                        header identity).
        /// @param startCounts    Per-container element counts at the last
        ///                        deload, ascending tag order.
        /// @param tailIndex       1-based tail set index since the base.
        /// @param directory       Target directory (created if absent).
        /// @param maxPayloadBytes Maximum payload bytes per file; > 0.
        /// @return The written file names, part order 1..N.
        std::vector<std::string> dumpLbMemoryTail(
            const LbMemory& lb, const std::string& chain, int64_t ordinal,
            const std::vector<int32_t>& startCounts, uint32_t tailIndex,
            const std::filesystem::path& directory,
            int32_t maxPayloadBytes,
            const std::vector<const DeloadColumn*>& extraColumns) {
            assert(maxPayloadBytes > 0);
            assert(tailIndex >= 1);
            const auto dumpStart = std::chrono::steady_clock::now();
            namespace fs = std::filesystem;
            std::error_code ec;
            fs::create_directories(directory, ec);
            assert(!ec);

            std::vector<DirectoryEntry> entries;
            std::vector<char> payload;
            std::size_t idx = 0;
            lb.visitContainers(
                [&](LbMemory::ContainerTag tag, const auto& c) {
                    using T = typename std::decay_t<decltype(c)>::value_type;
                    assert(idx < startCounts.size()
                        && "tail dump: start count missing for a tag");
                    const int32_t start = startCounts[idx++];
                    assert(c.size() >= start
                        && "tail dump on a shrunk container - the window "
                           "was not appended-only");
                    entries.push_back(
                        { static_cast<uint32_t>(tag),
                          static_cast<uint32_t>(sizeof(T)),
                          static_cast<uint64_t>(c.size() - start) });
                    c.appendSpanBytes(payload, start);
                });
            // Extra cold HashMemory columns continue the startCounts walk,
            // tag-ordered after LbMemory's.
            for (const DeloadColumn* col : extraColumns) {
                assert(idx < startCounts.size()
                    && "tail dump: start count missing for an extra column");
                const int32_t start = startCounts[idx++];
                assert(col->size() >= start
                    && "tail dump on a shrunk extra column - the window was "
                       "not appended-only");
                entries.push_back(
                    { col->tag(), col->elemSize(),
                      static_cast<uint64_t>(col->size() - start) });
                col->appendSpanBytes(payload, start);
            }
            assert(idx == startCounts.size());

            std::vector<std::string> files =
                writeFileSet(payload, entries, chain, ordinal,
                             /*kind=*/1u, tailIndex,
                             lb.manager.blockBytes(), directory,
                             maxPayloadBytes);
            deloadStats().recordV3Dump(
                static_cast<int64_t>(payload.size()),
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - dumpStart).count());
            return files;
        }

        /// @brief Rebuild an `LbMemory` from its deload file set.
        ///
        /// @details
        /// Asserts every header field against expectations (magic,
        /// version, chain hash AND verbatim chain, part numbering, page
        /// bytes, directory consistency across parts, element sizes
        /// against the compiled-in types) — a mismatch is a corrupted or
        /// foreign file and must stop the run at its origin. Containers
        /// are cleared and refilled element-by-element, which lands them
        /// on a fresh consecutive virtual index.
        ///
        /// @param lb        The aggregate to rebuild into (resident, with
        ///                  its manager bound to an initialized global).
        /// @param chain     The expected full LB chain string.
        /// @param files     File names from `dumpLbMemory`, part order.
        /// @param directory The directory the files live in.
        void loadLbMemory(LbMemory& lb, const std::string& chain,
                          const std::vector<std::string>& files,
                          const std::filesystem::path& directory,
                          const std::vector<DeloadColumn*>& extraColumns) {
            assert(!files.empty());
            const auto loadStart = std::chrono::steady_clock::now();
            int64_t totalPayloadBytes = 0;

            std::size_t fi = 0;
            bool firstSet = true;
            while (fi < files.size()) {
                // ---- parse one FILE SET (base image or tail delta) ----
                std::vector<DirectoryEntry> entries;
                std::vector<char> payload;
                uint32_t setKind = 0;
                uint32_t setPartCount = 0;
                for (uint32_t part = 1; ; ++part) {
                    assert(fi < files.size()
                        && "deload file list ends mid file-set");
                    std::ifstream in(directory / files[fi],
                                     std::ios::binary);
                    assert(in && "deload file not readable");
                    // Size first, then ONE block read. The former
                    // istreambuf_iterator pair copied the image a byte at a
                    // time through the streambuf into an unreserved vector --
                    // a virtual call per byte plus geometric reallocation, so
                    // the cost scaled with image size and dominated every
                    // canonical reload (14 ms/LB on IncubatorPeano1, 80 ms on
                    // IncubatorGauss1, where the chapter export made it
                    // visible). The raw loader beside this one always read in
                    // blocks; this brings the canonical path in line.
                    in.seekg(0, std::ios::end);
                    const std::streamoff fileBytes = in.tellg();
                    assert(fileBytes >= 0
                        && "deload file size query failed");
                    in.seekg(0, std::ios::beg);
                    std::vector<char> bytes(
                        static_cast<std::size_t>(fileBytes));
                    if (fileBytes > 0) {
                        in.read(bytes.data(), fileBytes);
                        assert(in.gcount() == fileBytes
                            && "deload file short read");
                    }

                    size_t offset = 0;
                    assert(bytes.size() >= 4
                        && std::memcmp(bytes.data(), kMagic, 4) == 0
                        && "deload file magic mismatch");
                    offset = 4;
                    const uint32_t version =
                        readRaw<uint32_t>(bytes, offset);
                    assert(version == kVersion);
                    const uint32_t kind = readRaw<uint32_t>(bytes, offset);
                    const int64_t ordinal = readRaw<int64_t>(bytes, offset);
                    assert(ordinal >= 0
                        && "deload header carries a negative ordinal");
                    const uint32_t filePart =
                        readRaw<uint32_t>(bytes, offset);
                    assert(filePart == part);
                    const uint32_t partCount =
                        readRaw<uint32_t>(bytes, offset);
                    const uint32_t blockBytes =
                        readRaw<uint32_t>(bytes, offset);
                    assert(blockBytes
                        == static_cast<uint32_t>(lb.manager.blockBytes()));
                    const uint32_t containerCount =
                        readRaw<uint32_t>(bytes, offset);

                    std::vector<DirectoryEntry> fileEntries;
                    for (uint32_t e = 0; e < containerCount; ++e) {
                        DirectoryEntry entry;
                        entry.tag = readRaw<uint32_t>(bytes, offset);
                        entry.elemSize = readRaw<uint32_t>(bytes, offset);
                        entry.elemCount = readRaw<uint64_t>(bytes, offset);
                        fileEntries.push_back(entry);
                    }
                    const uint32_t chainLen =
                        readRaw<uint32_t>(bytes, offset);
                    assert(chainLen == static_cast<uint32_t>(chain.size()));
                    assert(offset + chainLen <= bytes.size());
                    assert(std::memcmp(bytes.data() + offset, chain.data(),
                                       chainLen) == 0
                        && "deload file chain mismatch");
                    offset += chainLen;

                    if (part == 1) {
                        setKind = kind;
                        setPartCount = partCount;
                        entries = fileEntries;
                        // The base image leads; every later set is a tail.
                        assert(firstSet ? setKind == 0u : setKind == 1u);
                    }
                    else {
                        assert(kind == setKind);
                        assert(partCount == setPartCount);
                        assert(fileEntries.size() == entries.size());
                        for (size_t e = 0; e < entries.size(); ++e) {
                            assert(fileEntries[e].tag == entries[e].tag);
                            assert(fileEntries[e].elemSize
                                == entries[e].elemSize);
                            assert(fileEntries[e].elemCount
                                == entries[e].elemCount);
                        }
                    }
                    payload.insert(payload.end(), bytes.begin() + offset,
                                   bytes.end());
                    ++fi;
                    if (part == setPartCount) break;
                }

                // ---- distribute the set: base clears, tails append ----
                size_t offset = 0;
                size_t index = 0;
                lb.visitContainers(
                    [&](LbMemory::ContainerTag tag, auto& c) {
                        assert(index < entries.size()
                            && "deload directory shorter than the "
                               "compiled-in container registry");
                        const DirectoryEntry& e = entries[index++];
                        assert(e.tag == static_cast<uint32_t>(tag));
                        using T =
                            typename std::decay_t<decltype(c)>::value_type;
                        assert(e.elemSize == sizeof(T));
                        if (setKind == 0u) c.clear();
                        const std::size_t setBytes =
                            static_cast<std::size_t>(e.elemCount)
                            * sizeof(T);
                        assert(offset + setBytes <= payload.size());
                        c.bulkAppendBytes(
                            payload.data() + offset,
                            static_cast<int64_t>(e.elemCount));
                        offset += setBytes;
                    });
                // Extra cold HashMemory columns, tag-ordered after LbMemory's.
                for (DeloadColumn* col : extraColumns) {
                    assert(index < entries.size()
                        && "deload directory shorter than the registry "
                           "(extra columns)");
                    const DirectoryEntry& e = entries[index++];
                    assert(e.tag == col->tag());
                    assert(e.elemSize == col->elemSize());
                    if (setKind == 0u) col->clear();
                    const std::size_t setBytes =
                        static_cast<std::size_t>(e.elemCount) * col->elemSize();
                    assert(offset + setBytes <= payload.size());
                    col->bulkAppendBytes(
                        payload.data() + offset,
                        static_cast<int64_t>(e.elemCount));
                    offset += setBytes;
                }
                assert(index == entries.size());
                assert(offset == payload.size()
                    && "deload payload longer than the directory describes");
                totalPayloadBytes += static_cast<int64_t>(payload.size());
                firstSet = false;
            }
            deloadStats().recordV3Load(
                totalPayloadBytes,
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - loadStart).count());
        }

        /// @brief Compose a v4 raw-image file name: `lb<ordinal>_raw.bin`.
        ///
        /// @param ordinal The LB's deload ordinal; >= 0.
        /// @return The file name (no directory).
        std::string rawFileName(int64_t ordinal) {
            assert(ordinal >= 0);
            return "lb" + std::to_string(ordinal) + "_raw.bin";
        }

        namespace {

            /// @brief The v4 dynamic-header byte layout for a given shape and
            ///        chain — the prefix + chain + bitmap span and its aligned
            ///        total.
            struct RawHeaderLayout {
                uint32_t chainLen;    ///< Chain byte length.
                int64_t bmBytes;      ///< Live-vid bitmap byte length.
                int64_t rawHeader;    ///< prefix + chain + bitmap (unpadded).
                int64_t headerBytes;  ///< rawHeader rounded up to alignment.
            };

            /// @brief Compute the dynamic-header layout (pure).
            ///
            /// @param shape The arena raw shape (vid count sizes the bitmap).
            /// @param chain The LB chain string (its length sizes the header).
            /// @return The layout with the aligned `headerBytes` total.
            RawHeaderLayout computeRawHeaderLayout(
                const LbArena::RawShape& shape, const std::string& chain) {
                RawHeaderLayout L;
                L.chainLen = static_cast<uint32_t>(chain.size());
                L.bmBytes = (static_cast<int64_t>(shape.vidCount) + 7) / 8;
                L.rawHeader = static_cast<int64_t>(kRawHeaderPrefixBytes)
                    + L.chainLen + L.bmBytes;
                L.headerBytes = (L.rawHeader + kRawHeaderAlignBytes - 1)
                    / kRawHeaderAlignBytes * kRawHeaderAlignBytes;
                return L;
            }

            /// @brief Stream one v4 raw image to a positioned byte sink — the
            ///        format core shared by the named-file and extent-file
            ///        dumps.
            ///
            /// @details
            /// Writes the dynamic header (prefix, chain, live bitmap, zero
            /// padding) then the raw page payload, each via `writeRel(rel,
            /// data, len)` at the image-RELATIVE offset `rel` — the named-file
            /// sink ignores `rel` (sequential `ofstream`), the extent sink adds
            /// the slab base. Heap-free: the prefix rides a stack buffer, the
            /// bitmap and padding stream through one bounded stack chunk (a
            /// shared buffer would need locking — the executor pool dumps
            /// concurrently). Byte-identical to the pre-refactor inline dump.
            ///
            /// @tparam WriteRel `void(int64_t rel, const void* data, int64_t len)`.
            /// @param lb       The aggregate to image (resident, unchanged).
            /// @param chain    The full LB chain string (header identity).
            /// @param ordinal  The LB's deload ordinal (header field).
            /// @param writeRel The positioned byte sink.
            /// @return The payload byte count written (page + byte-bump bytes).
            template <typename WriteRel>
            int64_t emitRawImageStream(const LbMemory& lb,
                                       const std::string& chain,
                                       int64_t ordinal, WriteRel&& writeRel) {
                const LbArena::RawShape shape = lb.manager.rawShape();
                const RawHeaderLayout L = computeRawHeaderLayout(shape, chain);

                char prefix[kRawHeaderPrefixBytes];
                std::size_t hn = 0;
                const auto put = [&](const void* p, std::size_t n) {
                    assert(hn + n
                            <= static_cast<std::size_t>(kRawHeaderPrefixBytes)
                        && "v4 raw prefix layout out of step with "
                           "kRawHeaderPrefixBytes");
                    std::memcpy(prefix + hn, p, n);
                    hn += n;
                };
                put(kMagic, 4);
                put(&kVersionRaw, sizeof(uint32_t));
                put(&kKindRaw, sizeof(uint32_t));
                put(&ordinal, sizeof(int64_t));
                const uint32_t headerBytes =
                    static_cast<uint32_t>(L.headerBytes);
                put(&headerBytes, sizeof(uint32_t));
                const uint32_t blockBytes =
                    static_cast<uint32_t>(shape.blockBytes);
                const uint32_t pageBytes =
                    static_cast<uint32_t>(shape.pageBytes);
                const uint32_t byteBumpCursor =
                    static_cast<uint32_t>(shape.byteBumpCursor);
                const uint32_t vidCount =
                    static_cast<uint32_t>(shape.vidCount);
                const uint32_t livePages =
                    static_cast<uint32_t>(shape.livePages);
                put(&blockBytes, sizeof(uint32_t));
                put(&pageBytes, sizeof(uint32_t));
                put(&byteBumpCursor, sizeof(uint32_t));
                put(&L.chainLen, sizeof(uint32_t));
                put(&vidCount, sizeof(uint32_t));
                put(&livePages, sizeof(uint32_t));
                assert(hn == static_cast<std::size_t>(kRawHeaderPrefixBytes)
                    && "v4 raw prefix layout out of step with "
                       "kRawHeaderPrefixBytes");

                int64_t rel = 0;
                writeRel(rel, prefix, kRawHeaderPrefixBytes);
                rel += kRawHeaderPrefixBytes;
                writeRel(rel, chain.data(),
                         static_cast<int64_t>(L.chainLen));
                rel += L.chainLen;
                {
                    unsigned char chunk[kRawHeaderAlignBytes];
                    const int32_t vidsPerChunk = kRawHeaderAlignBytes * 8;
                    for (int32_t startVid = 0; startVid < shape.vidCount;
                         startVid += vidsPerChunk) {
                        const int32_t vidSpan =
                            (shape.vidCount - startVid < vidsPerChunk)
                                ? shape.vidCount - startVid : vidsPerChunk;
                        lb.manager.fillLiveBitmapRange(chunk, startVid, vidSpan);
                        const int64_t nb = (vidSpan + 7) / 8;
                        writeRel(rel, chunk, nb);
                        rel += nb;
                    }
                }
                {
                    char zeros[kRawHeaderAlignBytes] = {};
                    int64_t pad = L.headerBytes - L.rawHeader;
                    while (pad > 0) {
                        const int64_t take = pad < kRawHeaderAlignBytes
                            ? pad : kRawHeaderAlignBytes;
                        writeRel(rel, zeros, take);
                        rel += take;
                        pad -= take;
                    }
                }
                assert(rel == L.headerBytes
                    && "v4 raw header stream length out of step with "
                       "headerBytes");
                int64_t payloadBytes = 0;
                lb.manager.emitRawImage(
                    [&](const char* data, int64_t len) {
                        writeRel(rel, data, len);
                        rel += len;
                        payloadBytes += len;
                    });
                return payloadBytes;
            }

            /// @brief Rebuild one `LbMemory` from a v4 raw image read through a
            ///        positioned byte source — the format core shared by the
            ///        named-file and extent-file loads.
            ///
            /// @details
            /// Reads and asserts the dynamic header (magic, version, kind, the
            /// TIGHTENED `ordinal == expectedOrdinal` slab-reuse tripwire,
            /// geometry, verbatim chain), stages the dense page-tier restore
            /// through the same bounded bitmap chunks the dump emitted, then
            /// reads the payload STRAIGHT into the restored pages — each read
            /// via `readRel(dst, rel, len)` at the image-RELATIVE offset (the
            /// named-file source seeks in the file; the extent source adds the
            /// slab base). Heap-free; byte-identical to the pre-refactor inline
            /// load.
            ///
            /// @tparam ReadRel `void(void* dst, int64_t rel, int64_t len)`.
            /// @param lb              The aggregate to rebuild (resident, empty).
            /// @param chain           The expected full LB chain string.
            /// @param expectedOrdinal The LB's deload ordinal — a header
            ///                        mismatch is a stale-occupant / torn image.
            /// @param readRel         The positioned byte source.
            /// @return The payload byte count read.
            template <typename ReadRel>
            int64_t consumeRawImageStream(LbMemory& lb,
                                          const std::string& chain,
                                          int64_t expectedOrdinal,
                                          ReadRel&& readRel) {
                assert(expectedOrdinal >= 0
                    && "raw deload expected a non-negative ordinal");
                char prefix[kRawHeaderPrefixBytes];
                readRel(prefix, 0, kRawHeaderPrefixBytes);
                std::size_t off = 0;
                const auto get = [&](void* p, std::size_t n) {
                    assert(off + n
                            <= static_cast<std::size_t>(kRawHeaderPrefixBytes));
                    std::memcpy(p, prefix + off, n);
                    off += n;
                };
                assert(std::memcmp(prefix, kMagic, 4) == 0
                    && "raw deload magic mismatch");
                off = 4;
                uint32_t version = 0;
                get(&version, sizeof(uint32_t));
                assert(version == kVersionRaw
                    && "raw deload version mismatch");
                uint32_t kind = 0;
                get(&kind, sizeof(uint32_t));
                assert(kind == kKindRaw && "raw deload kind mismatch");
                int64_t ordinal = 0;
                get(&ordinal, sizeof(int64_t));
                // TIGHTENED (was `ordinal >= 0`): now that all ordinals share
                // ONE extent file, a slab reused by a new LB after a torn write
                // could still hold a previous occupant's valid-looking header —
                // an exact ordinal match is the direct slab-reuse tripwire
                // (the chain check below is the second). See
                // D-195 §4.
                assert(ordinal == expectedOrdinal
                    && "raw deload slab holds a different LB's ordinal — stale "
                       "occupant / torn image");
                uint32_t headerBytes = 0;
                get(&headerBytes, sizeof(uint32_t));
                uint32_t blockBytes = 0;
                get(&blockBytes, sizeof(uint32_t));
                assert(blockBytes
                        == static_cast<uint32_t>(lb.manager.blockBytes())
                    && "raw deload block-bytes mismatch");
                uint32_t pageBytes = 0;
                get(&pageBytes, sizeof(uint32_t));
                assert(pageBytes
                        == static_cast<uint32_t>(lb.manager.pageBytes())
                    && "raw deload page-bytes mismatch");
                uint32_t byteBumpCursor = 0;
                get(&byteBumpCursor, sizeof(uint32_t));
                uint32_t chainLen = 0;
                get(&chainLen, sizeof(uint32_t));
                assert(chainLen == static_cast<uint32_t>(chain.size())
                    && "raw deload chain length mismatch");
                uint32_t vidCount = 0;
                get(&vidCount, sizeof(uint32_t));
                uint32_t livePages = 0;
                get(&livePages, sizeof(uint32_t));
                assert(off == static_cast<std::size_t>(kRawHeaderPrefixBytes)
                    && "v4 raw prefix layout out of step with "
                       "kRawHeaderPrefixBytes");
                const int64_t bmBytes64 =
                    (static_cast<int64_t>(vidCount) + 7) / 8;
                assert(headerBytes
                            % static_cast<uint32_t>(kRawHeaderAlignBytes) == 0
                    && "raw deload headerBytes not header-aligned");
                assert(static_cast<int64_t>(headerBytes)
                            >= kRawHeaderPrefixBytes + chainLen + bmBytes64
                    && "raw deload headerBytes too small for chain + bitmap");

                {
                    char chunk[kRawHeaderAlignBytes];
                    uint32_t done = 0;
                    while (done < chainLen) {
                        const uint32_t take =
                            (chainLen - done
                             < static_cast<uint32_t>(kRawHeaderAlignBytes))
                                ? chainLen - done
                                : static_cast<uint32_t>(kRawHeaderAlignBytes);
                        readRel(chunk,
                                static_cast<int64_t>(kRawHeaderPrefixBytes)
                                    + done,
                                take);
                        assert(std::memcmp(chunk, chain.data() + done, take)
                                == 0
                            && "raw deload chain mismatch");
                        done += take;
                    }
                }

                lb.manager.restoreForRawLoadBegin(
                    static_cast<int32_t>(vidCount),
                    static_cast<int32_t>(livePages),
                    static_cast<ArenaOffset>(byteBumpCursor));
                {
                    unsigned char chunk[kRawHeaderAlignBytes];
                    const int32_t vidsPerChunk = kRawHeaderAlignBytes * 8;
                    int32_t slot = 0;
                    const int64_t bmBase =
                        static_cast<int64_t>(kRawHeaderPrefixBytes) + chainLen;
                    for (int32_t startVid = 0;
                         startVid < static_cast<int32_t>(vidCount);
                         startVid += vidsPerChunk) {
                        const int32_t vidSpan =
                            (static_cast<int32_t>(vidCount) - startVid
                             < vidsPerChunk)
                                ? static_cast<int32_t>(vidCount) - startVid
                                : vidsPerChunk;
                        const int64_t takeBytes = (vidSpan + 7) / 8;
                        readRel(chunk,
                                bmBase + static_cast<int64_t>(startVid / 8),
                                takeBytes);
                        slot = lb.manager.restoreForRawLoadChunk(
                            chunk, startVid, vidSpan, slot);
                    }
                    lb.manager.restoreForRawLoadEnd(
                        static_cast<int32_t>(vidCount), slot);
                }
                int64_t cursor = static_cast<int64_t>(headerBytes);
                int64_t payloadBytes = 0;
                lb.manager.fillRawImage(
                    [&](char* dst, int64_t len) {
                        readRel(dst, cursor, len);
                        cursor += len;
                        payloadBytes += len;
                    });
                return payloadBytes;
            }

        }

        /// @brief Serialize an `LbMemory` to its v4 RAW arena image — the
        ///        near-memcpy eviction dump (see the header contract).
        ///
        /// @param lb        The aggregate to image (unchanged; resident).
        /// @param chain     The full LB chain string (header identity).
        /// @param ordinal   The LB's deload ordinal (file name + header).
        /// @param directory Target directory (created if absent).
        /// @return The payload byte count written.
        int64_t dumpLbMemoryRaw(const LbMemory& lb, const std::string& chain,
                                int64_t ordinal,
                                const std::filesystem::path& directory) {
            const auto dumpStart = std::chrono::steady_clock::now();
            namespace fs = std::filesystem;
            std::error_code ec;
            fs::create_directories(directory, ec);
            assert(!ec);

            // The named-file sink writes SEQUENTIALLY (one CreateFile/truncate/
            // close per image); the image-relative offset the core supplies is
            // unused here (the ofstream cursor already advances in step). The
            // extent variant `dumpLbMemoryRawAt` shares the SAME core but adds
            // the slab base — the whole point of the refactor.
            std::ofstream out(directory / rawFileName(ordinal),
                              std::ios::binary | std::ios::trunc);
            assert(out && "raw deload file not writable");
            const int64_t payloadBytes = emitRawImageStream(
                lb, chain, ordinal,
                [&out](int64_t, const void* data, int64_t len) {
                    out.write(static_cast<const char*>(data),
                              static_cast<std::streamsize>(len));
                });
            assert(out && "raw deload file write failed");
            deloadStats().recordRawDump(
                payloadBytes,
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - dumpStart).count());
            return payloadBytes;
        }

        /// @brief Rebuild an `LbMemory` from its v4 RAW arena image — the
        ///        near-memcpy reload (see the header contract).
        ///
        /// @param lb        The aggregate to rebuild into (resident, empty).
        /// @param chain     The expected full LB chain string.
        /// @param fileName  The single raw file name recorded at dump time.
        /// @param directory The directory the file lives in.
        void loadLbMemoryRaw(LbMemory& lb, const std::string& chain,
                             int64_t expectedOrdinal,
                             const std::string& fileName,
                             const std::filesystem::path& directory) {
            const auto loadStart = std::chrono::steady_clock::now();
            std::ifstream in(directory / fileName, std::ios::binary);
            assert(in && "raw deload file not readable");
            // The named-file source seeks to the image-relative offset; the
            // extent variant `loadLbMemoryRawAt` shares the SAME core but adds
            // the slab base.
            const int64_t payloadBytes = consumeRawImageStream(
                lb, chain, expectedOrdinal,
                [&in](void* dst, int64_t rel, int64_t len) {
                    in.seekg(static_cast<std::streamoff>(rel));
                    in.read(static_cast<char*>(dst),
                            static_cast<std::streamsize>(len));
                    assert(in.gcount() == static_cast<std::streamsize>(len)
                        && "raw deload short read");
                });
            deloadStats().recordRawLoad(
                payloadBytes,
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - loadStart).count());
        }

        /// @brief Total v4 raw-image bytes (dynamic header + page payload) for
        ///        an LB — the extent-slab sizing input (see the declaration).
        ///
        /// @param lb    The aggregate to image (resident, unchanged).
        /// @param chain The full LB chain string (its length sizes the header).
        /// @return The image byte count the extent slab must hold.
        int64_t rawImageBytesFor(const LbMemory& lb,
                                 const std::string& chain) {
            const LbArena::RawShape shape = lb.manager.rawShape();
            const RawHeaderLayout L = computeRawHeaderLayout(shape, chain);
            const int64_t payload =
                static_cast<int64_t>(shape.livePages)
                    * static_cast<int64_t>(shape.pageBytes)
                + static_cast<int64_t>(shape.byteBumpCursor);
            return L.headerBytes + payload;
        }

        /// @brief Serialize an `LbMemory` to its v4 raw image IN PLACE inside
        ///        the extent file at `offset` (see the declaration).
        ///
        /// @param lb        The aggregate to image (resident, unchanged).
        /// @param chain     The full LB chain string (header identity).
        /// @param ordinal   The LB's deload ordinal (header field).
        /// @param file      The open extent file (positioned I/O, no open/close).
        /// @param offset    The LB's slab offset in the extent file; >= 0.
        /// @param slabBytes The slab's class capacity (the image must fit it).
        /// @return The payload byte count written (page + byte-bump bytes).
        int64_t dumpLbMemoryRawAt(const LbMemory& lb, const std::string& chain,
                                  int64_t ordinal, PositionedFile& file,
                                  int64_t offset, int64_t slabBytes) {
            const auto dumpStart = std::chrono::steady_clock::now();
            assert(offset >= 0 && "raw extent dump at a negative offset");
            const int64_t total = rawImageBytesFor(lb, chain);
            assert(total <= slabBytes
                && "raw image exceeds its extent slab — the caller must promote "
                   "the slab class before dumping");
            const int64_t payloadBytes = emitRawImageStream(
                lb, chain, ordinal,
                [&file, offset](int64_t rel, const void* data, int64_t len) {
                    file.writeAt(offset + rel, data, len);
                });
            deloadStats().recordRawDump(
                payloadBytes,
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - dumpStart).count());
            return payloadBytes;
        }

        /// @brief Rebuild an `LbMemory` from its v4 raw image at `offset`
        ///        inside the extent file (see the declaration).
        ///
        /// @param lb              The aggregate to rebuild (resident, empty).
        /// @param chain           The expected full LB chain string.
        /// @param expectedOrdinal The LB's deload ordinal — the slab-reuse
        ///                        tripwire (`ordinal == expectedOrdinal`).
        /// @param file            The open extent file (positioned I/O).
        /// @param offset          The LB's slab offset in the extent file; >= 0.
        void loadLbMemoryRawAt(LbMemory& lb, const std::string& chain,
                               int64_t expectedOrdinal, PositionedFile& file,
                               int64_t offset) {
            const auto loadStart = std::chrono::steady_clock::now();
            assert(offset >= 0 && "raw extent load at a negative offset");
            const int64_t payloadBytes = consumeRawImageStream(
                lb, chain, expectedOrdinal,
                [&file, offset](void* dst, int64_t rel, int64_t len) {
                    file.readAt(offset + rel, dst, len);
                });
            deloadStats().recordRawLoad(
                payloadBytes,
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - loadStart).count());
        }

        /// @brief Empty (or create) a deload directory.
        ///
        /// @details
        /// Runs at batch start only — the end-of-run files stay on disk
        /// for post-run inspection. Asserts the directory name contains
        /// `"deload"`: this function recursively deletes, and it must be
        /// impossible to point it at anything but a deload-purposed
        /// directory.
        ///
        /// @param directory The deload directory.
        void purgeDeloadDirectory(const std::filesystem::path& directory) {
            namespace fs = std::filesystem;
            assert(directory.filename().string().find("deload")
                       != std::string::npos
                && "purge refused: directory name does not look like a "
                   "deload directory");
            std::error_code ec;
            fs::remove_all(directory, ec);
            assert(!ec);
            fs::create_directories(directory, ec);
            assert(!ec);
        }

        /// @brief Write `registry.txt`: one
        ///        `<ordinal>\t<extentOffset>\t<slabBytes>\t<full chain>`
        ///        line per LB, ascending by ordinal (see the declaration).
        ///
        /// @param ordinalToChain Deload ordinal → full chain string.
        /// @param slabByOrdinal  Deload ordinal → current extent slab.
        /// @param directory      The deload directory.
        void rewriteRegistry(
            const std::map<int64_t, std::string>& ordinalToChain,
            const std::map<int64_t, SlabAllocation>& slabByOrdinal,
            const std::filesystem::path& directory) {
            std::ofstream out(directory / "registry.txt",
                              std::ios::trunc);
            assert(out && "registry not writable");
            for (const auto& entry : ordinalToChain) {
                const auto slabIt = slabByOrdinal.find(entry.first);
                const int64_t offset = slabIt != slabByOrdinal.end()
                    ? slabIt->second.offset : -1;
                const int64_t slabBytes = slabIt != slabByOrdinal.end()
                    ? slabIt->second.classBytes : 0;
                out << entry.first << "\t" << offset << "\t" << slabBytes
                    << "\t" << entry.second << "\n";
            }
            assert(out && "registry write failed");
        }

    }
}
