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

#include <algorithm>
#include <cassert>
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

            return writeFileSet(payload, entries, chain, ordinal,
                                /*kind=*/0u, /*tailIndex=*/0u,
                                lb.manager.blockBytes(), directory,
                                maxPayloadBytes);
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

            return writeFileSet(payload, entries, chain, ordinal,
                                /*kind=*/1u, tailIndex,
                                lb.manager.blockBytes(), directory,
                                maxPayloadBytes);
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
                    std::vector<char> bytes(
                        (std::istreambuf_iterator<char>(in)),
                        std::istreambuf_iterator<char>());

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
                firstSet = false;
            }
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

        /// @brief Write `registry.txt`: one `<ordinal>\t<full chain>` line
        ///        per LB, ascending by ordinal.
        ///
        /// @details
        /// The registry resolves the numeric file names (`lb<ordinal>_...`)
        /// back to LB chains for humans / post-run inspection — reload uses
        /// the file lists recorded at dump time, never the registry.
        /// `std::map` ordering makes the output deterministic.
        ///
        /// @param ordinalToChain Deload ordinal → full chain string.
        /// @param directory      The deload directory.
        void rewriteRegistry(
            const std::map<int64_t, std::string>& ordinalToChain,
            const std::filesystem::path& directory) {
            std::ofstream out(directory / "registry.txt",
                              std::ios::trunc);
            assert(out && "registry not writable");
            for (const auto& entry : ordinalToChain)
                out << entry.first << "\t" << entry.second << "\n";
            assert(out && "registry write failed");
        }

    }
}
