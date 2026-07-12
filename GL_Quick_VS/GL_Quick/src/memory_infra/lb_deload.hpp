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

#include "extent_file.hpp"
#include "lb_memory.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace gl {
    namespace lbdeload {

        /// @brief Production deload directory, repo-root relative (the
        ///        working directory of every pipeline invocation).
        constexpr const char* kDeloadDirectory = ".deload";

        /// @brief File-format magic, the first four bytes of every deload
        ///        file ("GLDL").
        constexpr char kMagic[4] = { 'G', 'L', 'D', 'L' };

        /// @brief File-format version written into every header.
        ///        Version 2 added the `kind` field (base image vs tail
        ///        delta) after the version word; version 3 replaced the
        ///        FNV chain-hash field with the per-LB deload ordinal.
        constexpr uint32_t kVersion = 3;

        /// @brief File-format version of the v4 RAW arena image — the
        ///        near-memcpy eviction/reload datapath. Distinct from
        ///        `kVersion` so the load dispatch (`Memory::reloadFromImage`)
        ///        and the loader's header assert tell the two formats apart.
        constexpr uint32_t kVersionRaw = 4;

        /// @brief `kind` field of a v4 raw image (0 = v3 base, 1 = v3 tail).
        constexpr uint32_t kKindRaw = 2;

        /// @brief Size of the v4 header's FIXED PREFIX: magic (4), version
        ///        (4), kind (4), ordinal (8), headerBytes (4), block bytes
        ///        (4), page bytes (4), byte-bump cursor (4), chain length
        ///        (4), vid count (4), live pages (4).
        ///
        /// @details
        /// The v4 header is DYNAMIC: the prefix carries `headerBytes` — the
        /// total header size including the chain string, the live-vid
        /// bitmap, and the zero padding up to the next
        /// `kRawHeaderAlignBytes` multiple — so the header is always
        /// sufficient BY CONSTRUCTION, whatever the chain length or vid
        /// count; no capacity assert exists (the remaining asserts are the
        /// sanity / mismatch checks). The predecessor was a fixed 4 KiB
        /// header whose deliberate capacity tripwire fired on the first big
        /// Gauss LB at 4 GiB (a ~1000-block LB ≈ 32000 vids = a ~4000-byte
        /// bitmap that cannot share 4096 bytes with the chain string and
        /// the fixed fields).
        constexpr int32_t kRawHeaderPrefixBytes = 48;

        /// @brief v4 header alignment: the total header (prefix + chain +
        ///        bitmap + zero padding) is rounded up to a multiple of
        ///        this, so the page payload stays 4 KiB-aligned in the file
        ///        (enables future unbuffered I/O).
        constexpr int32_t kRawHeaderAlignBytes = 4096;

        /// @brief Compaction fraction denominator: accumulated tail rows
        ///        reaching `base rows / kTailCompactionDenominator`
        ///        trigger a full canonical rewrite at the next deload.
        constexpr int64_t kTailCompactionDenominator = 4;

        /// @brief Maximum tail file-sets per base image — bounds the
        ///        reload's file count regardless of row volume.
        constexpr int32_t kMaxTailSets = 16;

        /// @brief Pool-pressure high-water mark for the block-release
        ///        policy, as a fraction: release blocks at iteration end
        ///        only when `blocksInUse * kReleaseDen >
        ///        totalBlocks * kReleaseNum` (default 3/4 of the pool).
        ///        Below pressure, dumped-fresh LBs keep their blocks and
        ///        the next reload is a no-op — zero I/O for stable LBs.
        constexpr int64_t kReleaseHighWaterNum = 3;
        constexpr int64_t kReleaseHighWaterDen = 4;

        /// @brief Type-erased deload column for a container that lives OUTSIDE
        ///        `LbMemory` — the cold `HashMemory` instances during the
        ///        Part-C migration (`memory_infra` cannot name `HashMemory`,
        ///        which is defined far down in `memory.hpp`).
        ///
        /// @details
        /// Exposes exactly the operations the generic `visitContainers`
        /// dump/load lambdas use, behind a vtable. Used ONLY for the cold
        /// `HashMemory` facets, ONLY during deload (never the hot request
        /// path), so the virtual dispatch is off every measured path. The
        /// `dumpLbMemory` / `loadLbMemory` walks append these columns AFTER
        /// `LbMemory`'s own (their tags are 51+, past LbMemory's 0..50), so
        /// the per-tag directory and payload stay in ascending tag order.
        /// When `HashMemory` finally moves INTO `LbMemory` at the end of
        /// Part C, the extra-column walk goes empty and this interface is
        /// retired.
        ///
        /// @see `DeloadColumnAdapter`, `dumpLbMemory`, `loadLbMemory`.
        struct DeloadColumn {
            virtual ~DeloadColumn() = default;

            /// @brief The container's deload tag (51+ for HashMemory).
            /// @return The tag.
            virtual uint32_t tag() const = 0;

            /// @brief Element width — the directory `elemSize` field.
            /// @return `sizeof` the element type.
            virtual uint32_t elemSize() const = 0;

            /// @brief Current element count.
            /// @return The element count.
            virtual int32_t size() const = 0;

            /// @brief Append element bytes from `start` to the payload.
            /// @param out   Destination payload buffer.
            /// @param start First element index to stream.
            virtual void appendSpanBytes(std::vector<char>& out,
                                         int32_t start) const = 0;

            /// @brief Bulk-append `count` elements from `data` (reload).
            /// @param data  Source bytes (the payload slice).
            /// @param count Element count.
            virtual void bulkAppendBytes(const char* data, int64_t count) = 0;

            /// @brief Drop every element — a base-image reload clears first.
            virtual void clear() = 0;
        };

        /// @brief Adapts one deload facet (an `ArenaVector` / `PagedVector`
        ///        view) to the `DeloadColumn` interface, carrying its tag.
        ///
        /// @details
        /// Constructed by `Memory` for each cold `HashMemory` facet (where
        /// `HashMemory` is a complete type) and passed to the deload by base
        /// pointer. Forwards every call to the wrapped facet.
        ///
        /// @tparam Facet The facet view type (exposes `value_type` / `size` /
        ///               `appendSpanBytes` / `bulkAppendBytes` / `clear`).
        /// @see `DeloadColumn`.
        template <typename Facet>
        struct DeloadColumnAdapter final : DeloadColumn {
            /// @brief Bind the adapter to a facet and its tag.
            ///
            /// @param tag   The container's deload tag.
            /// @param facet The facet; must outlive the adapter.
            DeloadColumnAdapter(uint32_t tag, Facet* facet)
                : tag_(tag), facet_(facet) {}

            uint32_t tag() const override { return tag_; }
            uint32_t elemSize() const override {
                return static_cast<uint32_t>(
                    sizeof(typename Facet::value_type));
            }
            int32_t size() const override { return facet_->size(); }
            void appendSpanBytes(std::vector<char>& out,
                                 int32_t start) const override {
                facet_->appendSpanBytes(out, start);
            }
            void bulkAppendBytes(const char* data, int64_t count) override {
                facet_->bulkAppendBytes(data, count);
            }
            void clear() override { facet_->clear(); }

        private:
            uint32_t tag_;
            Facet* facet_;
        };

        /// @brief Compose a deload file name:
        ///        `lb<ordinal>_[t<k>_]<part>_of_<partCount>.bin`.
        ///
        /// @details
        /// The per-LB `ordinal` (from
        /// `GlobalMemoryManager::assignDeloadOrdinal`) is the file's
        /// injective identity — it replaced the FNV chain hash, so a name
        /// collision is impossible by construction. `tailIndex` 0 names a
        /// base-image part (`lb<ordinal>_<part>_of_<N>.bin`); a positive
        /// index names a tail-delta part
        /// (`lb<ordinal>_t<k>_<part>_of_<N>.bin`) so successive tails never
        /// collide with the base or each other. The full chain is recovered
        /// from `registry.txt` and the file header, not the name.
        ///
        /// @param ordinal   The LB's deload ordinal; >= 0.
        /// @param tailIndex 0 for the base image; k >= 1 for tail set k.
        /// @param part      1-based part index.
        /// @param partCount Total parts of this LB's dump.
        /// @return The file name (no directory).
        std::string deloadFileName(int64_t ordinal, uint32_t tailIndex,
                                   uint32_t part, uint32_t partCount);

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
        /// @param chain           The full LB chain string (header identity).
        /// @param ordinal         The LB's deload ordinal (file name + header).
        /// @param directory       Target directory (created if absent).
        /// @param maxPayloadBytes Maximum payload bytes per file; > 0.
        /// @param extraColumns   Cold containers OUTSIDE `LbMemory` (the cold
        ///                       `HashMemory` instances), streamed after
        ///                       LbMemory's own at their tags 51+; empty until
        ///                       the Part-C migration wires them.
        /// @return The written file names, part order 1..N.
        std::vector<std::string> dumpLbMemory(
            const LbMemory& lb, const std::string& chain, int64_t ordinal,
            const std::filesystem::path& directory,
            int32_t maxPayloadBytes,
            const std::vector<const DeloadColumn*>& extraColumns = {});

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
        /// @param chain           The full LB chain string (header identity).
        /// @param ordinal         The LB's deload ordinal (file name + header).
        /// @param startCounts    Per-container element counts at the last
        ///                        deload, ascending tag order.
        /// @param tailIndex       1-based tail set index since the base.
        /// @param directory       Target directory (created if absent).
        /// @param maxPayloadBytes Maximum payload bytes per file; > 0.
        /// @param extraColumns   Cold containers OUTSIDE `LbMemory` (the cold
        ///                       `HashMemory` instances), tag-appended after
        ///                       LbMemory's; their `startCounts` entries follow
        ///                       LbMemory's in the same vector. Empty until
        ///                       Part-C wiring.
        /// @return The written file names, part order 1..N.
        std::vector<std::string> dumpLbMemoryTail(
            const LbMemory& lb, const std::string& chain, int64_t ordinal,
            const std::vector<int32_t>& startCounts, uint32_t tailIndex,
            const std::filesystem::path& directory,
            int32_t maxPayloadBytes,
            const std::vector<const DeloadColumn*>& extraColumns = {});

        /// @brief Rebuild an `LbMemory` from its deload file set.
        ///
        /// @details
        /// Asserts every header field against expectations (magic,
        /// version, ordinal sign, verbatim chain — the real identity check,
        /// part numbering, page bytes, directory consistency across parts,
        /// element sizes against the compiled-in types) — a mismatch is a
        /// corrupted or foreign file and must stop the run at its origin.
        /// Containers are cleared and refilled element-by-element, which
        /// lands them on a fresh consecutive virtual index.
        ///
        /// The file list may carry several FILE SETS: one base image
        /// (`kind = 0`, first) followed by tail deltas (`kind = 1`) in
        /// dump order; a `part == 1` header starts a new set. Base sets
        /// clear-and-fill; tail sets append.
        ///
        /// @param lb        The aggregate to rebuild into (resident, with
        ///                  its manager bound to an initialized global).
        /// @param chain     The expected full LB chain string.
        /// @param files     File names recorded at dump time, set order.
        /// @param directory The directory the files live in.
        /// @param extraColumns Cold containers OUTSIDE `LbMemory` (the cold
        ///                     `HashMemory` instances), filled after LbMemory's
        ///                     at their tags 51+; empty until Part-C wiring.
        void loadLbMemory(LbMemory& lb, const std::string& chain,
                          const std::vector<std::string>& files,
                          const std::filesystem::path& directory,
                          const std::vector<DeloadColumn*>& extraColumns = {});

        /// @brief Compose a v4 raw-image file name: `lb<ordinal>_raw.bin`.
        ///
        /// @details
        /// A raw image is ONE file per LB (no parts, no tails — the whole live
        /// page tier streams into it), so the name needs no part / tail index.
        /// The `_raw` marker also lets a human tell the two formats apart in
        /// `.deload/`. The `ordinal` is the injective per-LB identity (as for
        /// v3), so a re-dump of the same LB truncates the same file.
        ///
        /// @param ordinal The LB's deload ordinal; >= 0.
        /// @return The file name (no directory).
        std::string rawFileName(int64_t ordinal);

        /// @brief Serialize an `LbMemory` to its v4 RAW arena image — the
        ///        near-memcpy eviction dump.
        ///
        /// @details
        /// Writes ONE file (`rawFileName(ordinal)`, truncated): a DYNAMIC
        /// header — the `kRawHeaderPrefixBytes` fixed prefix (magic,
        /// `kVersionRaw`, `kKindRaw`, ordinal, `headerBytes`, block/page
        /// bytes, byte-bump cursor, chain length, vid count, live page
        /// count), then the verbatim chain, then the live-vid bitmap, then
        /// zero padding up to `headerBytes` (the next `kRawHeaderAlignBytes`
        /// multiple — the payload stays 4 KiB-aligned) — followed by the raw
        /// page payload streamed straight from pool memory via
        /// `LbArena::emitRawImage`. No element walk, no heap anywhere on
        /// this path: the prefix rides a small stack buffer, the chain is
        /// written from the caller's string, and the bitmap and the padding
        /// stream through a bounded 4 KiB stack chunk buffer
        /// (`LbArena::fillLiveBitmapRange` fills bit-chunks at byte-aligned
        /// vid offsets) — chosen over a shared reusable buffer because the
        /// executor pool dumps concurrently. The header size is sufficient
        /// by construction; no capacity assert exists.
        ///
        /// The bytes are NONDETERMINISTIC (fragmentation + grant order leak in)
        /// — [I-103](30_invariants.md) is waived for eviction images
        /// (user-approved) — but the restored LOGICAL state is byte-identical.
        /// Does NOT release the LB's blocks — the caller composes dump +
        /// `Memory::releaseStaticBlocksRaw`.
        ///
        /// @param lb        The aggregate to image (unchanged; resident).
        /// @param chain     The full LB chain string (header identity).
        /// @param ordinal   The LB's deload ordinal (file name + header).
        /// @param directory Target directory (created if absent).
        /// @return The payload byte count written (page + byte-bump bytes) — the
        ///         per-LB `lastRawImageBytes` victim-ranking input.
        int64_t dumpLbMemoryRaw(const LbMemory& lb, const std::string& chain,
                                int64_t ordinal,
                                const std::filesystem::path& directory);

        /// @brief Rebuild an `LbMemory` from its v4 RAW arena image — the
        ///        near-memcpy reload.
        ///
        /// @details
        /// Reads the `kRawHeaderPrefixBytes` fixed prefix and asserts it
        /// (magic, `kVersionRaw`, `kKindRaw`, ordinal sign, `headerBytes`
        /// consistency — alignment + capacity for chain and bitmap — and
        /// block / page geometry against the compiled-in arena), verifies
        /// the chain in bounded 4 KiB stack chunks (the identity check),
        /// then streams the live-vid bitmap through the same bounded chunk
        /// loop into the STAGED arena restore
        /// (`restoreForRawLoadBegin` / `Chunk` / `End` — symmetric to the
        /// dump's chunked bitmap emit, heap-free) which
        /// rebuilds a fresh dense page tier that binds the same vids, and
        /// `fillRawImage` reads the file's page bytes STRAIGHT into those pages
        /// (one copy, no staging buffer, no per-key index rebuild). The
        /// container scalar bookkeeping in the `Memory` shell is untouched by
        /// the round trip, so every container resolves immediately.
        ///
        /// The `expectedOrdinal` header check is TIGHTENED from the old
        /// `ordinal >= 0`: with all ordinals sharing one extent file, an exact
        /// ordinal match is the direct slab-reuse / torn-image tripwire
        /// (`D-195` §4).
        ///
        /// @param lb              The aggregate to rebuild into (resident — the
        ///                        caller `markResident`s first — empty arena).
        /// @param chain           The expected full LB chain string.
        /// @param expectedOrdinal The LB's deload ordinal (header tripwire).
        /// @param fileName        The single raw file name recorded at dump time.
        /// @param directory       The directory the file lives in.
        void loadLbMemoryRaw(LbMemory& lb, const std::string& chain,
                             int64_t expectedOrdinal,
                             const std::string& fileName,
                             const std::filesystem::path& directory);

        /// @brief Total v4 raw-image bytes (dynamic header + page payload) for
        ///        an LB — the input that sizes its extent slab.
        ///
        /// @details
        /// A pure function of the arena raw shape and the chain length (no
        /// I/O, no allocation): `headerBytes(chain, vidCount) + livePages *
        /// pageBytes + byteBumpCursor`. The extent wiring calls this to pick
        /// the slab class (`ExtentAllocator::classBytesFor`) and to decide
        /// in-place overwrite vs. class promotion.
        ///
        /// @param lb    The aggregate to image (resident, unchanged).
        /// @param chain The full LB chain string (its length sizes the header).
        /// @return The image byte count the slab must hold.
        int64_t rawImageBytesFor(const LbMemory& lb, const std::string& chain);

        /// @brief Serialize an `LbMemory` to its v4 raw image IN PLACE inside
        ///        the extent file — the near-memcpy eviction dump with NO
        ///        per-operation file open/create/close.
        ///
        /// @details
        /// Shares the exact byte format of `dumpLbMemoryRaw` (same dynamic
        /// header + page payload) but writes through the already-open
        /// `PositionedFile` at the LB's slab `offset`, so a stable-size re-dump
        /// is a single positioned overwrite (the extent design's whole point).
        /// Asserts the total image fits `slabBytes` — the caller must have
        /// promoted the slab class if the image outgrew it. Does NOT release
        /// the LB's blocks (the caller composes dump + release).
        ///
        /// @param lb        The aggregate to image (resident, unchanged).
        /// @param chain     The full LB chain string (header identity).
        /// @param ordinal   The LB's deload ordinal (header field).
        /// @param file      The open extent file.
        /// @param offset    The LB's slab offset in the extent file; >= 0.
        /// @param slabBytes The slab's class capacity (the image must fit it).
        /// @return The payload byte count written (page + byte-bump bytes).
        int64_t dumpLbMemoryRawAt(const LbMemory& lb, const std::string& chain,
                                  int64_t ordinal, PositionedFile& file,
                                  int64_t offset, int64_t slabBytes);

        /// @brief Rebuild an `LbMemory` from its v4 raw image at `offset`
        ///        inside the extent file — the near-memcpy reload.
        ///
        /// @details
        /// Shares the exact format core of `loadLbMemoryRaw` (header asserts
        /// including the `expectedOrdinal` slab-reuse tripwire, staged dense
        /// page-tier restore, straight payload read) but reads through the
        /// already-open `PositionedFile` at the LB's slab `offset`.
        ///
        /// @param lb              The aggregate to rebuild (resident, empty).
        /// @param chain           The expected full LB chain string.
        /// @param expectedOrdinal The LB's deload ordinal (slab-reuse tripwire).
        /// @param file            The open extent file.
        /// @param offset          The LB's slab offset in the extent file; >= 0.
        void loadLbMemoryRawAt(LbMemory& lb, const std::string& chain,
                               int64_t expectedOrdinal, PositionedFile& file,
                               int64_t offset);

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
        void purgeDeloadDirectory(const std::filesystem::path& directory);

        /// @brief Write `registry.txt`: one
        ///        `<ordinal>\t<extentOffset>\t<slabBytes>\t<full chain>`
        ///        line per LB, ascending by ordinal.
        ///
        /// @details
        /// The registry resolves the numeric identities (`lb<ordinal>_...`
        /// file names / extent slab placements) back to LB chains for humans /
        /// post-run inspection — reload NEVER reads it (named-file reload uses
        /// the file lists recorded at dump time; extent reload uses the LB's
        /// stored `rawExtentOffset_`). The two extent columns come from
        /// `slabByOrdinal` (`GlobalMemoryManager::extentSlabRegistry`); an LB
        /// with no current slab (v3-only, discharged, or the extent path off)
        /// writes `-1\t0`. `std::map` ordering makes the output deterministic
        /// in structure (the offset VALUES are nondeterministic, like the raw
        /// bytes they place — I-103's extent exemption).
        ///
        /// @param ordinalToChain Deload ordinal → full chain string.
        /// @param slabByOrdinal  Deload ordinal → current extent slab; entries
        ///                       missing for slab-less LBs.
        /// @param directory      The deload directory.
        void rewriteRegistry(
            const std::map<int64_t, std::string>& ordinalToChain,
            const std::map<int64_t, SlabAllocation>& slabByOrdinal,
            const std::filesystem::path& directory);

    }
}
