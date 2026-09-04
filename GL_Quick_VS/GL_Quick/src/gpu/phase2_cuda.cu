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

#include "phase2_cuda.hpp"

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <thrust/iterator/counting_iterator.h>

#include <cassert>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>

// On failure, name the exact CUDA error and the device memory state before
// the retained assert aborts: out-of-memory, a sticky error from an earlier
// failed launch, and a genuine API failure are three different bugs, and the
// bare status equality cannot tell them apart after the process dies.
#define GL_CUDA_ASSERT(call)                                                 \
    do {                                                                     \
        const cudaError_t glCudaStatus = (call);                              \
        if (glCudaStatus != cudaSuccess) {                                    \
            std::size_t glCudaFreeBytes = 0;                                  \
            std::size_t glCudaTotalBytes = 0;                                 \
            (void)cudaMemGetInfo(&glCudaFreeBytes, &glCudaTotalBytes);        \
            std::fprintf(stderr,                                             \
                "[CUDA-ASSERT-FAILURE] error=%s (%d) free_bytes=%llu "       \
                "total_bytes=%llu at %s:%d expression=%s\n",                 \
                cudaGetErrorName(glCudaStatus),                              \
                static_cast<int>(glCudaStatus),                              \
                static_cast<unsigned long long>(glCudaFreeBytes),            \
                static_cast<unsigned long long>(glCudaTotalBytes),           \
                __FILE__, __LINE__, #call);                                   \
            std::fflush(stderr);                                             \
        }                                                                    \
        assert(glCudaStatus == cudaSuccess);                                  \
    } while (false)

namespace {

    constexpr uint32_t kProjectionColumnCount = 24;

    struct ProjectionChecksumColumns {
        const unsigned char* data[kProjectionColumnCount]{};
        uint64_t bytes[kProjectionColumnCount]{};
    };

    struct ProjectionSemanticColumns {
        const gl::gpu::DeviceLogicalBlockProjection* logicalBlocks{ nullptr };
        const gl::IntEncodedExpr* statements{ nullptr };
        const gl::gpu::DeviceNameRecord* nameRecords{ nullptr };
        const char* nameBytes{ nullptr };
        const int32_t* nameSlots{ nullptr };
        const gl::gpu::DeviceRuleStringRecord* ruleStringRecords{ nullptr };
        const char* ruleStringBytes{ nullptr };
        const gl::gpu::DeviceByteMapView* byteMapViews{ nullptr };
        const gl::gpu::DeviceByteMapEntry* byteMapEntries{ nullptr };
        const int32_t* byteMapSlots{ nullptr };
        const char* byteKeyBytes{ nullptr };
        const gl::gpu::DeviceBlobRecord* blobRecords{ nullptr };
        const char* blobBytes{ nullptr };
        const gl::gpu::DeviceReverseMapView* reverseMapViews{ nullptr };
        const gl::gpu::DeviceReverseMapEntry* reverseMapEntries{ nullptr };
        const int32_t* reverseMapSlots{ nullptr };
        const char* reverseKeyBytes{ nullptr };
        const int32_t* reverseOwners{ nullptr };
        const gl::gpu::DevicePodMapView* podMapViews{ nullptr };
        const gl::gpu::DevicePodMapEntry* podMapEntries{ nullptr };
        const int32_t* podMapSlots{ nullptr };
        const int32_t* podRunValues{ nullptr };
        const int64_t* mandatoryStatementKeys{ nullptr };
        const char* metadataBytes{ nullptr };
    };

    /// @brief Hash an arbitrary byte span exactly like the host projection builder.
    ///
    /// @details
    /// Applies 64-bit FNV-1a to unsigned bytes. Name, byte-map, and reverse-map
    /// open-addressing probes must use this exact hash or their fixed host-built
    /// slot arrays would be semantically unreadable on the device.
    ///
    /// @param bytes Device-visible start of the byte span.
    /// @param length Number of bytes to hash.
    /// @return Deterministic 64-bit FNV-1a value.
    /// @invariant `bytes` addresses at least `length` readable bytes.
    __device__ uint64_t projectionByteHash(
        const char* bytes, uint32_t length) {
        uint64_t value = 14695981039346656037ull;
        for (uint32_t i = 0; i < length; ++i) {
            value ^= static_cast<uint8_t>(bytes[i]);
            value *= 1099511628211ull;
        }
        return value;
    }

    /// @brief Hash one widened plain-data key like the host projection builder.
    ///
    /// @details
    /// Applies the same SplitMix64 finalizer used while constructing every
    /// `DevicePodMapView`, preserving slot choice for signed and unsigned keys.
    ///
    /// @param value Bit-preserving widened key value.
    /// @return Deterministic 64-bit slot hash.
    /// @invariant Host and device use identical unsigned wraparound operations.
    __device__ uint64_t projectionPodHash(uint64_t value) {
        value += 0x9e3779b97f4a7c15ull;
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
        return value ^ (value >> 31);
    }

    /// @brief Compare two device-visible byte spans without host library calls.
    ///
    /// @details
    /// Performs an exact unsigned-byte comparison over an already-equal length.
    /// It is intentionally allocation-free and usable from every lookup family.
    ///
    /// @param left First readable byte span.
    /// @param right Second readable byte span.
    /// @param length Common byte length.
    /// @return True exactly when all `length` bytes match.
    /// @invariant Both pointers address at least `length` readable bytes.
    __device__ bool projectionBytesEqual(
        const char* left, const char* right, uint32_t length) {
        for (uint32_t i = 0; i < length; ++i) {
            if (left[i] != right[i]) return false;
        }
        return true;
    }

    /// @brief Compare two projected byte spans in processor string order.
    ///
    /// @param left First readable byte span.
    /// @param leftLength First byte length.
    /// @param right Second readable byte span.
    /// @param rightLength Second byte length.
    /// @return Negative, zero, or positive for lexicographic less, equal, or greater.
    /// @invariant Both pointers address their declared readable byte ranges.
    __device__ int projectionBytesCompare(
        const char* left,
        uint32_t leftLength,
        const char* right,
        uint32_t rightLength) {
        const uint32_t common = leftLength < rightLength
            ? leftLength : rightLength;
        for (uint32_t index = 0; index < common; ++index) {
            const uint8_t a = static_cast<uint8_t>(left[index]);
            const uint8_t b = static_cast<uint8_t>(right[index]);
            if (a != b) return a < b ? -1 : 1;
        }
        if (leftLength == rightLength) return 0;
        return leftLength < rightLength ? -1 : 1;
    }

    /// @brief Probe one logical block's projected name interner by decoded bytes.
    ///
    /// @details
    /// Uses the host-built fixed-load slot table and linear probing. A missing
    /// decoded name is a defined result. A hit returns the arena-global name-record
    /// index so the caller can inspect parent and lexical-rank metadata directly.
    ///
    /// @param columns Typed device addresses for the resident projection.
    /// @param block Owning logical-block descriptor.
    /// @param key Decoded name bytes.
    /// @param length Decoded name byte length.
    /// @return Arena-global name-record index, or `-1` for a defined miss.
    /// @invariant The slot table has load at most one half and every nonnegative
    ///            slot selects a record inside `block`.
    __device__ int32_t findProjectedNameRecord(
        const ProjectionSemanticColumns& columns,
        const gl::gpu::DeviceLogicalBlockProjection& block,
        const char* key,
        uint32_t length) {
        if (block.nameSlotCount == 0) {
            assert(block.nameRecordCount == 1);
            return -1;
        }
        assert((block.nameSlotCount & (block.nameSlotCount - 1)) == 0);
        uint32_t slot = static_cast<uint32_t>(projectionByteHash(key, length))
            & (block.nameSlotCount - 1);
        while (true) {
            const int32_t recordIndex = columns.nameSlots[
                block.nameSlotOffset + slot];
            if (recordIndex == -1) return -1;
            assert(recordIndex >= 0);
            const uint32_t index = static_cast<uint32_t>(recordIndex);
            assert(index > block.nameRecordOffset);
            assert(index < block.nameRecordOffset + block.nameRecordCount);
            const gl::gpu::DeviceNameRecord& record = columns.nameRecords[index];
            if (record.byteLength == length
                && projectionBytesEqual(
                    columns.nameBytes + record.byteOffset, key, length)) {
                return recordIndex;
            }
            slot = (slot + 1) & (block.nameSlotCount - 1);
        }
    }

    /// @brief Probe one projected byte-key map by exact serialized key bytes.
    ///
    /// @details
    /// Selects the view by semantic enumeration value, then follows its fixed-load
    /// linear-probing slots. Empty views and absent keys are defined misses; hits
    /// preserve the host entry's blob-record run without decoding it.
    ///
    /// @param columns Typed device addresses for the resident projection.
    /// @param block Owning logical-block descriptor.
    /// @param kind Byte-map semantic identity.
    /// @param key Serialized key bytes.
    /// @param length Serialized key length.
    /// @return Arena-global byte-map entry index, or `-1` for a defined miss.
    /// @invariant `block` owns exactly ten views in enumeration order.
    __device__ int32_t findProjectedByteMapEntry(
        const ProjectionSemanticColumns& columns,
        const gl::gpu::DeviceLogicalBlockProjection& block,
        gl::gpu::DeviceByteMapKind kind,
        const char* key,
        uint32_t length) {
        const uint32_t ordinal = static_cast<uint32_t>(kind);
        assert(ordinal < block.byteMapViewCount);
        const gl::gpu::DeviceByteMapView& view =
            columns.byteMapViews[block.byteMapViewOffset + ordinal];
        assert(view.kind == kind);
        if (view.entryCount == 0) {
            assert(view.slotCount == 0);
            return -1;
        }
        assert(view.slotCount > view.entryCount);
        assert((view.slotCount & (view.slotCount - 1)) == 0);
        uint32_t slot = static_cast<uint32_t>(projectionByteHash(key, length))
            & (view.slotCount - 1);
        while (true) {
            const int32_t entryIndex = columns.byteMapSlots[
                view.slotOffset + slot];
            if (entryIndex == -1) return -1;
            assert(entryIndex >= 0);
            const uint32_t index = static_cast<uint32_t>(entryIndex);
            assert(index >= view.entryOffset);
            assert(index < view.entryOffset + view.entryCount);
            const gl::gpu::DeviceByteMapEntry& entry =
                columns.byteMapEntries[index];
            if (entry.keyLength == length
                && projectionBytesEqual(
                    columns.byteKeyBytes + entry.keyOffset, key, length)) {
                return entryIndex;
            }
            slot = (slot + 1) & (view.slotCount - 1);
        }
    }

    /// @brief Probe one projected byte-key map from a segmented normalized key.
    ///
    /// @details
    /// Hashes and compares the canonical two-`NameId` header, shared normalized
    /// prefix payload, and lane-local appended-expression payload as one exact byte
    /// sequence. This avoids copying the shared prefix into every cooperative lane
    /// while selecting exactly the slots and entries used by the contiguous probe.
    /// Empty views and absent keys remain defined misses.
    ///
    /// @param columns Typed device addresses for the resident projection.
    /// @param block Owning logical-block descriptor.
    /// @param kind Byte-map semantic identity.
    /// @param expressionCount Serialized normalized-key expression count.
    /// @param prefix Shared normalized prefix payload.
    /// @param prefixLength Prefix length in `NameId` values.
    /// @param suffix Lane-local appended-expression payload.
    /// @param suffixLength Suffix length in `NameId` values.
    /// @return Arena-global byte-map entry index, or `-1` for a defined miss.
    /// @invariant Prefix and suffix together fit `MAX_KEY_SLOTS` and their byte
    ///            concatenation equals the contiguous `NormKey` representation.
    /// @see findProjectedByteMapEntry
    __device__ int32_t findProjectedSegmentedByteMapEntry(
        const ProjectionSemanticColumns& columns,
        const gl::gpu::DeviceLogicalBlockProjection& block,
        gl::gpu::DeviceByteMapKind kind,
        gl::NameId expressionCount,
        const gl::NameId* prefix,
        gl::NameId prefixLength,
        const gl::NameId* suffix,
        gl::NameId suffixLength) {
        assert(expressionCount > 0);
        assert(prefixLength >= 0);
        assert(suffixLength > 0);
        assert(prefixLength + suffixLength
            <= gl::ExecutionParameters::MAX_KEY_SLOTS);
        const gl::NameId header[2]{
            expressionCount, prefixLength + suffixLength };
        const char* const headerBytes = reinterpret_cast<const char*>(header);
        const char* const prefixBytes = reinterpret_cast<const char*>(prefix);
        const char* const suffixBytes = reinterpret_cast<const char*>(suffix);
        constexpr uint32_t headerByteCount = 2 * sizeof(gl::NameId);
        const uint32_t prefixByteCount = static_cast<uint32_t>(
            prefixLength * sizeof(gl::NameId));
        const uint32_t suffixByteCount = static_cast<uint32_t>(
            suffixLength * sizeof(gl::NameId));
        const uint32_t totalByteCount = headerByteCount
            + prefixByteCount + suffixByteCount;

        const uint32_t ordinal = static_cast<uint32_t>(kind);
        assert(ordinal < block.byteMapViewCount);
        const gl::gpu::DeviceByteMapView& view =
            columns.byteMapViews[block.byteMapViewOffset + ordinal];
        assert(view.kind == kind);
        if (view.entryCount == 0) {
            assert(view.slotCount == 0);
            return -1;
        }
        assert(view.slotCount > view.entryCount);
        assert((view.slotCount & (view.slotCount - 1)) == 0);
        uint64_t hash = 14695981039346656037ull;
        for (uint32_t index = 0; index < headerByteCount; ++index) {
            hash ^= static_cast<uint8_t>(headerBytes[index]);
            hash *= 1099511628211ull;
        }
        for (uint32_t index = 0; index < prefixByteCount; ++index) {
            hash ^= static_cast<uint8_t>(prefixBytes[index]);
            hash *= 1099511628211ull;
        }
        for (uint32_t index = 0; index < suffixByteCount; ++index) {
            hash ^= static_cast<uint8_t>(suffixBytes[index]);
            hash *= 1099511628211ull;
        }
        uint32_t slot = static_cast<uint32_t>(hash)
            & (view.slotCount - 1);
        while (true) {
            const int32_t entryIndex = columns.byteMapSlots[
                view.slotOffset + slot];
            if (entryIndex == -1) return -1;
            assert(entryIndex >= 0);
            const uint32_t index = static_cast<uint32_t>(entryIndex);
            assert(index >= view.entryOffset);
            assert(index < view.entryOffset + view.entryCount);
            const gl::gpu::DeviceByteMapEntry& entry =
                columns.byteMapEntries[index];
            const char* const entryBytes =
                columns.byteKeyBytes + entry.keyOffset;
            if (entry.keyLength == totalByteCount
                && projectionBytesEqual(
                    entryBytes, headerBytes, headerByteCount)
                && projectionBytesEqual(
                    entryBytes + headerByteCount,
                    prefixBytes, prefixByteCount)
                && projectionBytesEqual(
                    entryBytes + headerByteCount + prefixByteCount,
                    suffixBytes, suffixByteCount)) {
                return entryIndex;
            }
            slot = (slot + 1) & (view.slotCount - 1);
        }
    }

    /// @brief Probe one projected remaining-argument reverse index.
    ///
    /// @details
    /// Uses the derived normalized-key table built from the forward remaining-
    /// argument map. The returned entry owns the exact run of forward key ids used
    /// by request evaluation; missing normalized keys are defined misses.
    ///
    /// @param columns Typed device addresses for the resident projection.
    /// @param block Owning logical-block descriptor.
    /// @param key Serialized normalized-key bytes.
    /// @param length Serialized normalized-key length.
    /// @return Arena-global reverse-map entry index, or `-1` for a defined miss.
    /// @invariant `block` owns exactly one reverse-map view.
    __device__ int32_t findProjectedReverseMapEntry(
        const ProjectionSemanticColumns& columns,
        const gl::gpu::DeviceLogicalBlockProjection& block,
        const char* key,
        uint32_t length) {
        assert(block.reverseMapViewCount == 1);
        const gl::gpu::DeviceReverseMapView& view =
            columns.reverseMapViews[block.reverseMapViewOffset];
        if (view.entryCount == 0) {
            assert(view.slotCount == 0);
            return -1;
        }
        assert(view.slotCount > view.entryCount);
        assert((view.slotCount & (view.slotCount - 1)) == 0);
        uint32_t slot = static_cast<uint32_t>(projectionByteHash(key, length))
            & (view.slotCount - 1);
        while (true) {
            const int32_t entryIndex = columns.reverseMapSlots[
                view.slotOffset + slot];
            if (entryIndex == -1) return -1;
            assert(entryIndex >= 0);
            const uint32_t index = static_cast<uint32_t>(entryIndex);
            assert(index >= view.entryOffset);
            assert(index < view.entryOffset + view.entryCount);
            const gl::gpu::DeviceReverseMapEntry& entry =
                columns.reverseMapEntries[index];
            if (entry.keyLength == length
                && projectionBytesEqual(
                    columns.reverseKeyBytes + entry.keyOffset, key, length)) {
                return entryIndex;
            }
            slot = (slot + 1) & (view.slotCount - 1);
        }
    }

    /// @brief Probe one projected plain-data map by its widened key.
    ///
    /// @details
    /// Selects the view by semantic enumeration value and uses its SplitMix64
    /// fixed-load slots. Sets, scalar maps, and run maps share the same entry shape;
    /// absence is a defined miss and never triggers processor fallback.
    ///
    /// @param columns Typed device addresses for the resident projection.
    /// @param block Owning logical-block descriptor.
    /// @param kind Plain-data map semantic identity.
    /// @param key Widened signed key.
    /// @return Arena-global plain-data entry index, or `-1` for a defined miss.
    /// @invariant `block` owns exactly nine views in enumeration order.
    __device__ int32_t findProjectedPodMapEntry(
        const ProjectionSemanticColumns& columns,
        const gl::gpu::DeviceLogicalBlockProjection& block,
        gl::gpu::DevicePodMapKind kind,
        int64_t key) {
        const uint32_t ordinal = static_cast<uint32_t>(kind);
        assert(ordinal < block.podMapViewCount);
        const gl::gpu::DevicePodMapView& view =
            columns.podMapViews[block.podMapViewOffset + ordinal];
        assert(view.kind == kind);
        if (view.entryCount == 0) {
            assert(view.slotCount == 0);
            return -1;
        }
        assert(view.slotCount > view.entryCount);
        assert((view.slotCount & (view.slotCount - 1)) == 0);
        uint32_t slot = static_cast<uint32_t>(projectionPodHash(
            static_cast<uint64_t>(key))) & (view.slotCount - 1);
        while (true) {
            const int32_t entryIndex = columns.podMapSlots[
                view.slotOffset + slot];
            if (entryIndex == -1) return -1;
            assert(entryIndex >= 0);
            const uint32_t index = static_cast<uint32_t>(entryIndex);
            assert(index >= view.entryOffset);
            assert(index < view.entryOffset + view.entryCount);
            if (columns.podMapEntries[index].key == key) return entryIndex;
            slot = (slot + 1) & (view.slotCount - 1);
        }
    }

    /// @brief Build the exact normalized-key payload for one projected candidate.
    ///
    /// @details
    /// Replays `appendExprToIntNormalizedKey` over logical-block-local statement
    /// indices. Argument identifiers receive sequential slots in first-appearance
    /// order; every emitted argument pair has changeable marker zero because an
    /// unchangeable argument is already distinguished by its u_-prefixed `argId`.
    ///
    /// @param columns Typed device addresses for resident statement records.
    /// @param block Owning logical-block descriptor.
    /// @param statementIndices Ordered logical-block-local candidate path.
    /// @param count Number of premises in the candidate.
    /// @param output Caller-owned `MAX_KEY_SLOTS` payload destination.
    /// @return Number of `NameId` values written to @p output.
    /// @invariant All indices select statements owned by @p block and the complete
    ///            payload fits `MAX_KEY_SLOTS`.
    __device__ gl::NameId buildProjectedNormalizedKey(
        const ProjectionSemanticColumns& columns,
        const gl::gpu::DeviceLogicalBlockProjection& block,
        const gl::NameId* statementIndices,
        gl::NameId count,
        gl::NameId* output) {
        gl::NameId variableIds[gl::ExecutionParameters::MAX_KEY_SLOTS];
        gl::NameId variableCount = 0;
        gl::NameId nextNormalizedId = 1;
        gl::NameId outputLength = 0;
        for (gl::NameId premise = 0; premise < count; ++premise) {
            assert(statementIndices[premise] >= 0);
            const uint32_t localIndex = static_cast<uint32_t>(
                statementIndices[premise]);
            assert(localIndex < block.statementCount);
            const gl::IntEncodedExpr& expression = columns.statements[
                block.statementOffset + localIndex];
            assert(expression.arity >= 0);
            assert(expression.arity <= gl::ExecutionParameters::MAX_ARITY);
            assert(outputLength + 2 <= gl::ExecutionParameters::MAX_KEY_SLOTS);
            output[outputLength++] = expression.nameId;
            output[outputLength++] = expression.negation;
            for (gl::NameId argument = 0;
                 argument < expression.arity; ++argument) {
                assert(outputLength + 2
                    <= gl::ExecutionParameters::MAX_KEY_SLOTS);
                const gl::NameId variableId = expression.argId[argument];
                gl::NameId normalizedId = 0;
                for (gl::NameId known = 0; known < variableCount; ++known) {
                    if (variableIds[known] == variableId) {
                        normalizedId = known + 1;
                        break;
                    }
                }
                if (normalizedId == 0) {
                    assert(variableCount
                        < gl::ExecutionParameters::MAX_KEY_SLOTS);
                    variableIds[variableCount++] = variableId;
                    normalizedId = nextNormalizedId++;
                }
                output[outputLength++] = normalizedId;
                output[outputLength++] = 0;
            }
        }
        assert(nextNormalizedId == variableCount + 1);
        return outputLength;
    }

    /// @brief Apply the projected owner-set u_ signature gate to one candidate.
    ///
    /// @details
    /// Reads the matched subkey entry's single canonical `OwnerSet` blob without
    /// decoding or allocation. A loose owner or empty signature set accepts. Each
    /// non-empty signature compares its flattened `(argument slot, argFullId)`
    /// requirements against the candidate in premise and argument order. The
    /// appended provenance-owner section does not affect this gate, but the reader
    /// consumes and validates it so the device layout stays coupled to the host
    /// codec.
    ///
    /// @param columns Typed device addresses for statements and owner blobs.
    /// @param block Owning logical-block descriptor.
    /// @param subkeyEntry Arena-global matched subkey-map entry.
    /// @param statementIndices Ordered logical-block-local candidate path.
    /// @param count Number of candidate premises, at least three.
    /// @return True exactly when some projected owner signature is satisfiable.
    /// @invariant The subkey entry has exactly one structurally valid owner blob.
    __device__ bool projectedOwnerSetUSatisfied(
        const ProjectionSemanticColumns& columns,
        const gl::gpu::DeviceLogicalBlockProjection& block,
        const gl::gpu::DeviceByteMapEntry& subkeyEntry,
        const gl::NameId* statementIndices,
        gl::NameId count) {
        assert(count >= 3);
        assert(subkeyEntry.blobRecordCount == 1);
        const gl::gpu::DeviceBlobRecord& blob = columns.blobRecords[
            subkeyEntry.blobRecordOffset];
        assert(blob.byteLength >= 9);
        const char* const begin = columns.blobBytes + blob.byteOffset;
        const char* cursor = begin;
        const char* const end = begin + blob.byteLength;
        const bool hasLooseOwner = static_cast<uint8_t>(*cursor++) != 0;
        const auto readInt32 = [&cursor, end]() {
            assert(cursor + sizeof(int32_t) <= end);
            uint32_t value = static_cast<uint8_t>(cursor[0])
                | (static_cast<uint32_t>(static_cast<uint8_t>(cursor[1])) << 8)
                | (static_cast<uint32_t>(static_cast<uint8_t>(cursor[2])) << 16)
                | (static_cast<uint32_t>(static_cast<uint8_t>(cursor[3])) << 24);
            cursor += sizeof(int32_t);
            return static_cast<int32_t>(value);
        };
        const int32_t signatureCount = readInt32();
        assert(signatureCount >= 0);
        gl::NameId requestArguments[gl::ExecutionParameters::MAX_KEY_SLOTS];
        int32_t requestArgumentCount = 0;
        if (!hasLooseOwner && signatureCount != 0) {
            for (gl::NameId premise = 0; premise < count; ++premise) {
                assert(statementIndices[premise] >= 0);
                const uint32_t localIndex = static_cast<uint32_t>(
                    statementIndices[premise]);
                assert(localIndex < block.statementCount);
                const gl::IntEncodedExpr& expression = columns.statements[
                    block.statementOffset + localIndex];
                for (int32_t argument = 0; argument < expression.arity;
                     ++argument) {
                    assert(requestArgumentCount
                        < gl::ExecutionParameters::MAX_KEY_SLOTS);
                    requestArguments[requestArgumentCount++] =
                        expression.argFullId[argument];
                }
            }
        }

        bool signatureAccepted = false;
        for (int32_t signature = 0;
             signature < signatureCount; ++signature) {
            const int32_t pairCount = readInt32();
            assert(pairCount >= 0);
            bool accepted = !hasLooseOwner;
            for (int32_t pair = 0; pair < pairCount; ++pair) {
                const int32_t slot = readInt32();
                const gl::NameId requiredId = readInt32();
                if (accepted && (slot < 0 || slot >= requestArgumentCount
                    || requestArguments[slot] != requiredId)) {
                    accepted = false;
                }
            }
            signatureAccepted = signatureAccepted || accepted;
        }

        const int32_t ownerCount = readInt32();
        assert(ownerCount >= 0);
        for (int32_t owner = 0; owner < ownerCount; ++owner) {
            (void)readInt32();
            (void)readInt32();
            const int32_t signatureIndex = readInt32();
            assert(signatureIndex >= -1);
            assert(signatureIndex < signatureCount);
        }
        assert(cursor == end);
        return hasLooseOwner || signatureCount == 0 || signatureAccepted;
    }

    /// @brief Build and probe one complete normalized-key candidate on the device.
    ///
    /// @details
    /// This direct semantic twin executes the same reusable key and owner helpers
    /// used by frontier growth. It serializes the fixed-width `NormKey` framing on
    /// its stack, probes the selected whole/subkey maps, and keeps subkey presence
    /// separate from owner-signature satisfaction.
    ///
    /// @param columns Typed device addresses for every resident semantic column.
    /// @param logicalBlockCount Uploaded logical-block descriptor count.
    /// @param probe Logical block, registry, and ordered statement-index path.
    /// @param output One fixed result record.
    /// @return Nothing; writes `output[0]`.
    /// @invariant The launch geometry is one block containing one thread.
    __global__ void growthCandidateProbeKernel(
        ProjectionSemanticColumns columns,
        uint32_t logicalBlockCount,
        gl::gpu::DeviceGrowthCandidateProbe probe,
        gl::gpu::DeviceGrowthCandidateProbeResult* output) {
        assert(blockIdx.x == 0 && blockDim.x == 1 && threadIdx.x == 0);
        assert(probe.logicalBlockIndex < logicalBlockCount);
        assert(probe.count >= 1);
        assert(probe.count <= gl::ExecutionParameters::MAX_EXPRESSIONS);
        const uint32_t memoryOrdinal = static_cast<uint32_t>(probe.memory);
        assert(memoryOrdinal <= static_cast<uint32_t>(
            gl::gpu::DeviceHashMemoryKind::working));
        const gl::gpu::DeviceLogicalBlockProjection& block =
            columns.logicalBlocks[probe.logicalBlockIndex];

        gl::gpu::DeviceGrowthCandidateProbeResult result{};
        result.normalizedKeyLength = buildProjectedNormalizedKey(
            columns, block, probe.statementIndices, probe.count,
            result.normalizedKey);
        gl::NameId serialized[gl::ExecutionParameters::MAX_KEY_SLOTS + 2];
        serialized[0] = probe.count;
        serialized[1] = result.normalizedKeyLength;
        for (gl::NameId index = 0;
             index < result.normalizedKeyLength; ++index) {
            serialized[index + 2] = result.normalizedKey[index];
        }
        const uint32_t serializedBytes = static_cast<uint32_t>(
            (result.normalizedKeyLength + 2) * sizeof(gl::NameId));
        const auto wholeKind = static_cast<gl::gpu::DeviceByteMapKind>(
            memoryOrdinal * 2);
        const auto subkeyKind = static_cast<gl::gpu::DeviceByteMapKind>(
            memoryOrdinal * 2 + 1);
        const int32_t subkeyIndex = findProjectedByteMapEntry(
            columns, block, subkeyKind,
            reinterpret_cast<const char*>(serialized), serializedBytes);
        const gl::NameId finalStatementIndex =
            probe.statementIndices[probe.count - 1];
        assert(finalStatementIndex >= 0);
        assert(static_cast<uint32_t>(finalStatementIndex)
            < block.statementCount);
        const gl::IntEncodedExpr& finalExpression = columns.statements[
            block.statementOffset
                + static_cast<uint32_t>(finalStatementIndex)];
        const gl::NameId suffixLength = 2 + finalExpression.arity * 2;
        assert(suffixLength <= result.normalizedKeyLength);
        const gl::NameId prefixLength =
            result.normalizedKeyLength - suffixLength;
        const int32_t segmentedSubkeyIndex =
            findProjectedSegmentedByteMapEntry(
                columns, block, subkeyKind, probe.count,
                result.normalizedKey, prefixLength,
                result.normalizedKey + prefixLength, suffixLength);
        assert(segmentedSubkeyIndex == subkeyIndex);
        if (subkeyIndex >= 0) {
            result.subkeyPresent = 1;
            if (probe.count < 3) {
                result.subkeySatisfied = 1;
            }
            else {
                result.subkeySatisfied = projectedOwnerSetUSatisfied(
                    columns, block,
                    columns.byteMapEntries[static_cast<uint32_t>(subkeyIndex)],
                    probe.statementIndices, probe.count) ? 1u : 0u;
            }
        }
        const int32_t wholeKeyIndex = findProjectedByteMapEntry(
            columns, block, wholeKind,
            reinterpret_cast<const char*>(serialized), serializedBytes);
        const int32_t segmentedWholeKeyIndex =
            findProjectedSegmentedByteMapEntry(
                columns, block, wholeKind, probe.count,
                result.normalizedKey, prefixLength,
                result.normalizedKey + prefixLength, suffixLength);
        assert(segmentedWholeKeyIndex == wholeKeyIndex);
        result.wholeKeyPresent = wholeKeyIndex >= 0 ? 1u : 0u;
        output[0] = result;
    }

    /// @brief Execute one direct semantic lookup against resident device columns.
    ///
    /// @details
    /// Exactly one thread dispatches to the four lookup families and copies the
    /// selected payload shape into a pointer-free result. This diagnostic kernel
    /// tests the same primitives later request kernels call in parallel.
    ///
    /// @param columns Typed device addresses for all resident semantic columns.
    /// @param logicalBlockCount Uploaded logical-block descriptor count.
    /// @param probe Lookup family, logical block, and key.
    /// @param output One-record fixed device result destination.
    /// @return Nothing; writes `output[0]`.
    /// @invariant The launch geometry is one block containing one thread.
    __global__ void projectionLookupProbeKernel(
        ProjectionSemanticColumns columns,
        uint32_t logicalBlockCount,
        gl::gpu::DeviceLookupProbe probe,
        gl::gpu::DeviceLookupProbeResult* output) {
        assert(blockIdx.x == 0 && blockDim.x == 1 && threadIdx.x == 0);
        assert(probe.logicalBlockIndex < logicalBlockCount);
        assert(probe.keyLength <= gl::gpu::DeviceLookupProbe::kMaximumKeyBytes);
        const gl::gpu::DeviceLogicalBlockProjection& block =
            columns.logicalBlocks[probe.logicalBlockIndex];
        gl::gpu::DeviceLookupProbeResult result{};

        switch (probe.kind) {
        case gl::gpu::DeviceLookupProbeKind::name: {
            const int32_t globalRecord = findProjectedNameRecord(
                columns, block, probe.key, probe.keyLength);
            if (globalRecord >= 0) {
                const uint32_t globalIndex = static_cast<uint32_t>(globalRecord);
                const gl::gpu::DeviceNameRecord& record =
                    columns.nameRecords[globalIndex];
                result.recordIndex = static_cast<int32_t>(
                    globalIndex - block.nameRecordOffset);
                result.payloadOffset = record.byteOffset;
                result.payloadCount = record.byteLength;
                result.firstIntValue = record.parentId;
                result.scalar = record.decodedLexRank;
            }
            break;
        }
        case gl::gpu::DeviceLookupProbeKind::byteMap: {
            assert(probe.mapKind <= static_cast<uint32_t>(
                gl::gpu::DeviceByteMapKind::overallRemainingArgs));
            const int32_t entryIndex = findProjectedByteMapEntry(
                columns, block,
                static_cast<gl::gpu::DeviceByteMapKind>(probe.mapKind),
                probe.key, probe.keyLength);
            result.recordIndex = entryIndex;
            if (entryIndex >= 0) {
                const gl::gpu::DeviceByteMapEntry& entry =
                    columns.byteMapEntries[static_cast<uint32_t>(entryIndex)];
                result.payloadOffset = entry.blobRecordOffset;
                result.payloadCount = entry.blobRecordCount;
                if (entry.blobRecordCount > 0) {
                    result.firstIntValue = static_cast<int32_t>(
                        columns.blobRecords[entry.blobRecordOffset].byteLength);
                }
            }
            break;
        }
        case gl::gpu::DeviceLookupProbeKind::reverseMap: {
            const int32_t entryIndex = findProjectedReverseMapEntry(
                columns, block, probe.key, probe.keyLength);
            result.recordIndex = entryIndex;
            if (entryIndex >= 0) {
                const gl::gpu::DeviceReverseMapEntry& entry =
                    columns.reverseMapEntries[static_cast<uint32_t>(entryIndex)];
                result.payloadOffset = entry.ownerOffset;
                result.payloadCount = entry.ownerCount;
                if (entry.ownerCount > 0)
                    result.firstIntValue = columns.reverseOwners[entry.ownerOffset];
            }
            break;
        }
        case gl::gpu::DeviceLookupProbeKind::podMap: {
            assert(probe.mapKind <= static_cast<uint32_t>(
                gl::gpu::DevicePodMapKind::mailEligibleMarkers));
            const int32_t entryIndex = findProjectedPodMapEntry(
                columns, block,
                static_cast<gl::gpu::DevicePodMapKind>(probe.mapKind),
                probe.podKey);
            result.recordIndex = entryIndex;
            if (entryIndex >= 0) {
                const gl::gpu::DevicePodMapEntry& entry =
                    columns.podMapEntries[static_cast<uint32_t>(entryIndex)];
                result.payloadOffset = entry.runOffset;
                result.payloadCount = entry.runCount;
                result.scalar = entry.scalar;
                if (entry.runCount > 0)
                    result.firstIntValue = columns.podRunValues[entry.runOffset];
            }
            break;
        }
        default:
            assert(false);
        }
        output[0] = result;
    }

    /// @brief Select the projected whole-key and subkey maps for one filter call.
    ///
    /// @details
    /// Converts the task-level hash-memory identity into the two byte-map kinds
    /// used by the processor statement filter. Every defined enumeration value
    /// maps to exactly one registry pair; an unknown value asserts.
    ///
    /// @param memory Task-level hash-memory identity.
    /// @param wholeKeys Output whole-key map kind.
    /// @param subkeys Output subkey map kind.
    /// @return Nothing.
    /// @invariant Both outputs belong to the same resident hash-memory registry.
    __device__ void filterMapKinds(
        gl::gpu::DeviceHashMemoryKind memory,
        gl::gpu::DeviceByteMapKind& wholeKeys,
        gl::gpu::DeviceByteMapKind& subkeys) {
        switch (memory) {
        case gl::gpu::DeviceHashMemoryKind::overall:
            wholeKeys = gl::gpu::DeviceByteMapKind::overallWholeKeys;
            subkeys = gl::gpu::DeviceByteMapKind::overallSubkeys;
            break;
        case gl::gpu::DeviceHashMemoryKind::local:
            wholeKeys = gl::gpu::DeviceByteMapKind::localWholeKeys;
            subkeys = gl::gpu::DeviceByteMapKind::localSubkeys;
            break;
        case gl::gpu::DeviceHashMemoryKind::localDelta:
            wholeKeys = gl::gpu::DeviceByteMapKind::deltaWholeKeys;
            subkeys = gl::gpu::DeviceByteMapKind::deltaSubkeys;
            break;
        case gl::gpu::DeviceHashMemoryKind::working:
            wholeKeys = gl::gpu::DeviceByteMapKind::workingWholeKeys;
            subkeys = gl::gpu::DeviceByteMapKind::workingSubkeys;
            break;
        default:
            assert(false);
        }
    }

    /// @brief Reproduce the processor's single-statement filter predicate.
    ///
    /// @details
    /// Excludes frozen validity subtrees, constructs the exact serialized
    /// one-expression normalized key, probes the selected subkey map and optional
    /// whole-key map, and applies the inclusive iteration ceiling. A one-premise
    /// subkey is presence-only on the processor, so no owner blob is read here.
    ///
    /// @param columns Resident device semantic columns.
    /// @param block Owning logical-block descriptor.
    /// @param call Filter-call semantic selectors.
    /// @param statementIndex Statement index relative to `block`.
    /// @return True exactly when the processor filter would retain the statement
    ///         before its first-8,192 accepted-row cap.
    /// @invariant The predicate reads only resident projection bytes and writes no
    ///            proof or scheduling state.
    __device__ bool phase2StatementPassesFilter(
        const ProjectionSemanticColumns& columns,
        const gl::gpu::DeviceLogicalBlockProjection& block,
        const gl::gpu::DevicePhase2FilterCall& call,
        uint32_t statementIndex) {
        assert(statementIndex < block.statementCount);
        const gl::IntEncodedExpr& statement = columns.statements[
            block.statementOffset + statementIndex];

        const uint32_t frozenOrdinal = static_cast<uint32_t>(
            gl::gpu::DevicePodMapKind::frozenOrBranches);
        assert(frozenOrdinal < block.podMapViewCount);
        const gl::gpu::DevicePodMapView& frozenView = columns.podMapViews[
            block.podMapViewOffset + frozenOrdinal];
        if (frozenView.entryCount > 0) {
            for (gl::NameId scope = statement.validityId;
                 scope != 0;) {
                if (findProjectedPodMapEntry(
                        columns, block,
                        gl::gpu::DevicePodMapKind::frozenOrBranches,
                        static_cast<int64_t>(scope)) >= 0) {
                    return false;
                }
                assert(scope >= 0);
                const uint32_t recordIndex = block.nameRecordOffset
                    + static_cast<uint32_t>(scope);
                assert(recordIndex < block.nameRecordOffset
                    + block.nameRecordCount);
                scope = columns.nameRecords[recordIndex].parentId;
            }
        }

        gl::NameId key[2 + gl::ExecutionParameters::MAX_KEY_SLOTS]{};
        gl::NameId payloadLength = 0;
        key[2 + payloadLength++] = statement.nameId;
        key[2 + payloadLength++] = statement.negation;
        gl::NameId rawIds[gl::ExecutionParameters::MAX_ARITY]{};
        gl::NameId normalizedIds[gl::ExecutionParameters::MAX_ARITY]{};
        gl::NameId normalizedCount = 0;
        gl::NameId nextNormalized = 1;
        for (gl::NameId argument = 0;
             argument < statement.arity; ++argument) {
            const gl::NameId raw = statement.argId[argument];
            gl::NameId normalized = 0;
            for (gl::NameId known = 0;
                 known < normalizedCount; ++known) {
                if (rawIds[known] == raw) {
                    normalized = normalizedIds[known];
                    break;
                }
            }
            if (normalized == 0) {
                assert(normalizedCount < gl::ExecutionParameters::MAX_ARITY);
                normalized = nextNormalized++;
                rawIds[normalizedCount] = raw;
                normalizedIds[normalizedCount] = normalized;
                ++normalizedCount;
            }
            key[2 + payloadLength++] = normalized;
            key[2 + payloadLength++] = 0;
        }
        assert(payloadLength <= gl::ExecutionParameters::MAX_KEY_SLOTS);
        key[0] = 1;
        key[1] = payloadLength;
        const uint32_t keyBytes = static_cast<uint32_t>(
            (2 + payloadLength) * sizeof(gl::NameId));

        gl::gpu::DeviceByteMapKind wholeKeys =
            gl::gpu::DeviceByteMapKind::overallWholeKeys;
        gl::gpu::DeviceByteMapKind subkeys =
            gl::gpu::DeviceByteMapKind::overallSubkeys;
        filterMapKinds(call.memory, wholeKeys, subkeys);
        const bool subkeyPresent = findProjectedByteMapEntry(
            columns, block, subkeys,
            reinterpret_cast<const char*>(key), keyBytes) >= 0;
        const bool wholeKeyPresent = call.alsoAcceptFullKeys != 0
            && findProjectedByteMapEntry(
                columns, block, wholeKeys,
                reinterpret_cast<const char*>(key), keyBytes) >= 0;
        if (!subkeyPresent && !wholeKeyPresent) return false;
        return statement.maxIteration
            <= call.maximumIterationNumberVariable;
    }

    constexpr uint32_t kFilterThreads = 256;
    constexpr uint32_t kProcessorMaximumFilteredRows = 8192;

    /// @brief Count each filter call's processor-retained first-8,192 rows.
    ///
    /// @details
    /// Launches one block per call and walks statement chunks in ascending index
    /// order. Block reductions count accepted rows until the processor envelope
    /// is full; all calls execute in one bulk launch.
    ///
    /// @param columns Resident device semantic columns.
    /// @param logicalBlockCount Uploaded logical-block descriptor count.
    /// @param calls Uploaded call descriptors.
    /// @param callCount Number of scheduled calls.
    /// @param counts Per-call retained-row output.
    /// @return Nothing.
    /// @invariant `gridDim.x == callCount` and `blockDim.x == kFilterThreads`.
    __global__ void phase2FilterCountKernel(
        ProjectionSemanticColumns columns,
        uint32_t logicalBlockCount,
        const gl::gpu::DevicePhase2FilterCall* calls,
        uint32_t callCount,
        uint32_t* counts) {
        const uint32_t callIndex = blockIdx.x;
        assert(callIndex < callCount);
        assert(blockDim.x == kFilterThreads);
        const gl::gpu::DevicePhase2FilterCall& call = calls[callIndex];
        assert(call.logicalBlockIndex < logicalBlockCount);
        assert(call.alsoAcceptFullKeys <= 1);
        const gl::gpu::DeviceLogicalBlockProjection& block =
            columns.logicalBlocks[call.logicalBlockIndex];
        assert(call.statementCount == block.statementCount);
        using BlockReduce = cub::BlockReduce<uint32_t, kFilterThreads>;
        __shared__ typename BlockReduce::TempStorage reduceStorage;
        __shared__ uint32_t retained;
        if (threadIdx.x == 0) retained = 0;
        __syncthreads();

        for (uint32_t base = 0; base < call.statementCount;
             base += kFilterThreads) {
            if (retained >= kProcessorMaximumFilteredRows) break;
            const uint32_t statementIndex = base + threadIdx.x;
            const uint32_t accepted = statementIndex < call.statementCount
                && phase2StatementPassesFilter(
                    columns, block, call, statementIndex) ? 1u : 0u;
            const uint32_t chunkAccepted =
                BlockReduce(reduceStorage).Sum(accepted);
            __syncthreads();
            if (threadIdx.x == 0) {
                const uint32_t next = retained + chunkAccepted;
                retained = next < kProcessorMaximumFilteredRows
                    ? next : kProcessorMaximumFilteredRows;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) counts[callIndex] = retained;
    }

    /// @brief Emit compact sortable keys for every retained filter row.
    ///
    /// @details
    /// Replays the pure filter predicate, uses a block scan to recover each
    /// accepted row's processor acceptance ordinal, and writes directly into the
    /// scanned compact call segment. The composite key orders by call ordinal,
    /// decoded-name rank, then original statement index, which is the stable
    /// name-only processor sort.
    ///
    /// @param columns Resident device semantic columns.
    /// @param logicalBlockCount Uploaded logical-block descriptor count.
    /// @param calls Uploaded call descriptors.
    /// @param callCount Number of scheduled calls.
    /// @param counts Per-call retained counts from `phase2FilterCountKernel`.
    /// @param offsets Exclusive compact output offsets.
    /// @param keys Compact unsorted composite-key output.
    /// @return Nothing.
    /// @invariant Each call writes exactly `counts[call]` disjoint keys.
    __global__ void phase2FilterEmitKernel(
        ProjectionSemanticColumns columns,
        uint32_t logicalBlockCount,
        const gl::gpu::DevicePhase2FilterCall* calls,
        uint32_t callCount,
        const uint32_t* counts,
        const uint32_t* offsets,
        uint64_t* keys) {
        const uint32_t callIndex = blockIdx.x;
        assert(callIndex < callCount);
        assert(blockDim.x == kFilterThreads);
        const gl::gpu::DevicePhase2FilterCall& call = calls[callIndex];
        assert(call.logicalBlockIndex < logicalBlockCount);
        const gl::gpu::DeviceLogicalBlockProjection& block =
            columns.logicalBlocks[call.logicalBlockIndex];
        assert(call.statementCount == block.statementCount);
        using BlockScan = cub::BlockScan<uint32_t, kFilterThreads>;
        __shared__ typename BlockScan::TempStorage scanStorage;
        __shared__ uint32_t retained;
        if (threadIdx.x == 0) retained = 0;
        __syncthreads();

        for (uint32_t base = 0; base < call.statementCount;
             base += kFilterThreads) {
            if (retained >= kProcessorMaximumFilteredRows) break;
            const uint32_t statementIndex = base + threadIdx.x;
            const uint32_t accepted = statementIndex < call.statementCount
                && phase2StatementPassesFilter(
                    columns, block, call, statementIndex) ? 1u : 0u;
            uint32_t prefix = 0;
            uint32_t chunkAccepted = 0;
            BlockScan(scanStorage).ExclusiveSum(
                accepted, prefix, chunkAccepted);
            const uint32_t ordinal = retained + prefix;
            if (accepted != 0 && ordinal < kProcessorMaximumFilteredRows) {
                assert(statementIndex <
                    (1u << gl::gpu::kDeviceFilterStatementIndexBits));
                const gl::NameId statementName = columns.statements[
                    block.statementOffset + statementIndex].nameId;
                assert(statementName >= 0);
                const uint32_t nameRecordIndex = block.nameRecordOffset
                    + static_cast<uint32_t>(statementName);
                assert(nameRecordIndex < block.nameRecordOffset
                    + block.nameRecordCount);
                const uint32_t rank =
                    columns.nameRecords[nameRecordIndex].decodedLexRank;
                assert(rank < (1u << gl::gpu::kDeviceFilterNameRankBits));
                keys[offsets[callIndex] + ordinal] =
                    (static_cast<uint64_t>(callIndex)
                        << gl::gpu::kDeviceFilterCallShift)
                    | (static_cast<uint64_t>(rank)
                        << gl::gpu::kDeviceFilterStatementIndexBits)
                    | statementIndex;
            }
            __syncthreads();
            if (threadIdx.x == 0) retained += chunkAccepted;
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const uint32_t capped = retained < kProcessorMaximumFilteredRows
                ? retained : kProcessorMaximumFilteredRows;
            assert(capped == counts[callIndex]);
        }
    }

    /// @brief Map original filter calls to their exact compact class spans.
    ///
    /// @details
    /// Assigns one thread to each processor-order call, reads its immutable class
    /// index, and copies the class retained count and offset into the original-call
    /// columns consumed by growth. Duplicate calls therefore share one stable
    /// sorted span while retaining independent processor ordinals elsewhere.
    ///
    /// @param callClassIndices Exact class index for every original call.
    /// @param callCount Number of original processor-order calls.
    /// @param classCounts Retained first-8,192 row count per exact class.
    /// @param classOffsets Exclusive compact span offset per exact class.
    /// @param classCount Number of exact class descriptors.
    /// @param callCounts Output retained count per original call.
    /// @param callOffsets Output compact span offset per original call.
    /// @return Nothing.
    /// @invariant Every original call names one class in `[0, classCount)`.
    __global__ void phase2FilterExpandClassSpansKernel(
        const uint32_t* callClassIndices,
        uint32_t callCount,
        const uint32_t* classCounts,
        const uint32_t* classOffsets,
        uint32_t classCount,
        uint32_t* callCounts,
        uint32_t* callOffsets) {
        const uint32_t callIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if (callIndex >= callCount) return;
        const uint32_t classIndex = callClassIndices[callIndex];
        assert(classIndex < classCount);
        callCounts[callIndex] = classCounts[classIndex];
        callOffsets[callIndex] = classOffsets[classIndex];
    }

    /// @brief Observation-only candidate-gate counters at one growth depth.
    ///
    /// @details
    /// Mirrors the host-visible census with CUDA-native unsigned-long-long fields
    /// so device atomics accumulate exact 64-bit counts. Semantic kernels write
    /// these fields only when the explicit census switch is active and never read
    /// them for control flow.
    struct DeviceGrowthGateCounters {
        unsigned long long candidateAttempts{ 0 };
        unsigned long long mandatoryReachable{ 0 };
        unsigned long long validityComparable{ 0 };
        unsigned long long hypothesisCompatible{ 0 };
        unsigned long long secondaryCompatible{ 0 };
        unsigned long long keyLengthAllowed{ 0 };
        unsigned long long subkeyPresent{ 0 };
        unsigned long long ownerSatisfied{ 0 };
        unsigned long long wholeKeyPresent{ 0 };
        unsigned long long termsSatisfied{ 0 };
        unsigned long long acceptedEvents{ 0 };
        unsigned long long children{ 0 };
    };

    static_assert(sizeof(DeviceGrowthGateCounters) == 96);

    /// @brief Mutable counters shared by one bounded request-growth launch.
    ///
    /// @details
    /// Frontier counts belong to the two ping-pong arrays. Accepted-event and
    /// raw-request counts persist through every expansion level. Every append
    /// uses an atomic reservation followed by an assert against its fixed arena.
    struct DeviceGrowthCounters {
        uint32_t frontierCounts[2]{};
        uint32_t acceptedEventCount{ 0 };
        uint32_t rawRequestCount{ 0 };
        uint32_t shortNodeCount{ 0 };
        uint32_t cooperativeNodeCount{ 0 };
        uint32_t prefixPayloadCount{ 0 };
        uint32_t prefixVariableCount{ 0 };
        uint32_t prefixSecondaryCount{ 0 };
        unsigned long long spanNodeCounts[
            gl::gpu::kDeviceGrowthSpanBucketCount]{};
        unsigned long long spanCandidateCounts[
            gl::gpu::kDeviceGrowthSpanBucketCount]{};
        DeviceGrowthGateCounters gateDepthCounts[
            gl::gpu::kDeviceGrowthCensusDepthCount]{};
    };

    static_assert(sizeof(DeviceGrowthCounters) == 1096);

    /// @brief Decode one statement index from the global sorted filter stream.
    ///
    /// @details
    /// The low composite-key field is the logical-block-local statement index;
    /// call and decoded-name rank fields are ordering metadata only.
    ///
    /// @param sortedKeys Global radix-sorted retained-key prefix.
    /// @param globalRow Arena-global retained-row position.
    /// @return Logical-block-local statement index.
    /// @invariant The encoded statement field fits `NameId` and the public mask.
    __device__ gl::NameId growthStatementIndex(
        const uint64_t* sortedKeys, uint32_t globalRow) {
        return static_cast<gl::NameId>(
            sortedKeys[globalRow]
                & gl::gpu::kDeviceFilterStatementIndexMask);
    }

    /// @brief Test whether one projected validity lies on another's ancestor path.
    ///
    /// @details
    /// Walks `DeviceNameRecord::parentId` from the descendant to the zero
    /// sentinel, reproducing `NameMap::ancContains` without a pointer or cache.
    ///
    /// @param columns Resident name-record column.
    /// @param block Owning logical-block name slice.
    /// @param ancestor Candidate ancestor identifier.
    /// @param descendant Candidate descendant identifier.
    /// @return True when the identifiers are equal or @p ancestor is strict above.
    /// @invariant Every nonzero parent remains inside the same block name slice.
    __device__ bool projectedValidityIsAncestor(
        const ProjectionSemanticColumns& columns,
        const gl::gpu::DeviceLogicalBlockProjection& block,
        gl::NameId ancestor,
        gl::NameId descendant) {
        assert(ancestor > 0);
        assert(descendant > 0);
        gl::NameId current = descendant;
        while (current != 0) {
            if (current == ancestor) return true;
            assert(current >= 0);
            const uint32_t recordIndex = block.nameRecordOffset
                + static_cast<uint32_t>(current);
            assert(recordIndex < block.nameRecordOffset
                + block.nameRecordCount);
            current = columns.nameRecords[recordIndex].parentId;
        }
        return false;
    }

    /// @brief Reproduce `NameMap::comparable` on projected validity records.
    ///
    /// @param columns Resident name-record column.
    /// @param block Owning logical-block name slice.
    /// @param left First validity identifier.
    /// @param right Second validity identifier.
    /// @return True exactly when one validity is an ancestor of the other.
    /// @invariant Both identifiers are nonzero members of @p block.
    __device__ bool projectedValiditiesComparable(
        const ProjectionSemanticColumns& columns,
        const gl::gpu::DeviceLogicalBlockProjection& block,
        gl::NameId left,
        gl::NameId right) {
        return projectedValidityIsAncestor(columns, block, left, right)
            || projectedValidityIsAncestor(columns, block, right, left);
    }

    /// @brief Reproduce `NameMap::deeperOf` on a comparable projected pair.
    ///
    /// @param columns Resident name-record column.
    /// @param block Owning logical-block name slice.
    /// @param left First comparable validity identifier.
    /// @param right Second comparable validity identifier.
    /// @return The descendant member of the pair.
    /// @invariant @p left and @p right are comparable.
    __device__ gl::NameId projectedDeeperValidity(
        const ProjectionSemanticColumns& columns,
        const gl::gpu::DeviceLogicalBlockProjection& block,
        gl::NameId left,
        gl::NameId right) {
        if (projectedValidityIsAncestor(columns, block, left, right))
            return right;
        assert(projectedValidityIsAncestor(columns, block, right, left));
        return left;
    }

    /// @brief Search one projected decoded name for a fixed byte literal.
    ///
    /// @details
    /// Performs the rare `_orint_` widening-path substring test directly over
    /// the packed name byte column, matching `containsSpan` without decoding.
    ///
    /// @param columns Resident name records and bytes.
    /// @param block Owning logical-block name slice.
    /// @param nameId Local name identifier to inspect.
    /// @param literal Fixed byte sequence.
    /// @param literalLength Number of bytes in @p literal.
    /// @return True when the literal occurs contiguously in the decoded name.
    /// @invariant @p nameId is nonzero and @p literal addresses readable bytes.
    __device__ bool projectedNameContains(
        const ProjectionSemanticColumns& columns,
        const gl::gpu::DeviceLogicalBlockProjection& block,
        gl::NameId nameId,
        const char* literal,
        uint32_t literalLength) {
        assert(nameId > 0);
        const uint32_t recordIndex = block.nameRecordOffset
            + static_cast<uint32_t>(nameId);
        assert(recordIndex < block.nameRecordOffset + block.nameRecordCount);
        const gl::gpu::DeviceNameRecord& record =
            columns.nameRecords[recordIndex];
        if (record.byteLength < literalLength) return false;
        for (uint32_t start = 0;
             start + literalLength <= record.byteLength; ++start) {
            bool equal = true;
            for (uint32_t index = 0; index < literalLength; ++index) {
                if (columns.nameBytes[record.byteOffset + start + index]
                    != literal[index]) {
                    equal = false;
                    break;
                }
            }
            if (equal) return true;
        }
        return false;
    }

    /// @brief Select the projected membership table for a mandatory view.
    ///
    /// @param view Task-level mandatory statement-view identity.
    /// @param output Matching resident plain-data map kind.
    /// @return Nothing.
    /// @invariant Every defined view has exactly one projected membership map.
    __device__ void mandatoryViewMapKind(
        gl::gpu::DeviceMandatoryViewKind view,
        gl::gpu::DevicePodMapKind& output) {
        switch (view) {
        case gl::gpu::DeviceMandatoryViewKind::local:
            output = gl::gpu::DevicePodMapKind::localStatements;
            break;
        case gl::gpu::DeviceMandatoryViewKind::localDelta:
            output = gl::gpu::DevicePodMapKind::localDeltaStatements;
            break;
        case gl::gpu::DeviceMandatoryViewKind::external:
            output = gl::gpu::DevicePodMapKind::externalStatements;
            break;
        default:
            assert(false);
        }
    }

    /// @brief Compute one filtered statement's mandatory-view membership bits.
    ///
    /// @details
    /// Packs `(originalId, validityId)` exactly like `packStatementKey`, probes
    /// every view of every batch term, and assigns bit `(term * 2 + view)` on a
    /// hit. At most four projected membership probes are performed.
    ///
    /// @param columns Resident statements and plain-data lookup columns.
    /// @param block Owning logical-block projection.
    /// @param batch Request batch naming zero to two mandatory terms.
    /// @param terms Uploaded global mandatory-term column.
    /// @param statementIndex Logical-block-local statement index.
    /// @return Union of this statement's term-view membership bits.
    /// @invariant Batch term and view slices are valid uploaded prefixes.
    __device__ uint8_t projectedStatementViewMask(
        const ProjectionSemanticColumns& columns,
        const gl::gpu::DeviceLogicalBlockProjection& block,
        const gl::gpu::DeviceRequestBatch& batch,
        const gl::gpu::DeviceMandatoryTerm* terms,
        gl::NameId statementIndex) {
        assert(statementIndex >= 0);
        assert(static_cast<uint32_t>(statementIndex) < block.statementCount);
        const gl::IntEncodedExpr& statement = columns.statements[
            block.statementOffset + static_cast<uint32_t>(statementIndex)];
        const int64_t packed = static_cast<int64_t>(
            (static_cast<uint64_t>(
                static_cast<uint32_t>(statement.originalId)) << 32)
            | static_cast<uint64_t>(
                static_cast<uint32_t>(statement.validityId)));
        uint8_t mask = 0;
        for (uint32_t termIndex = 0;
             termIndex < batch.termCount; ++termIndex) {
            const gl::gpu::DeviceMandatoryTerm& term =
                terms[batch.termOffset + termIndex];
            assert(term.viewCount >= 1 && term.viewCount <= 2);
            for (uint32_t viewIndex = 0;
                 viewIndex < term.viewCount; ++viewIndex) {
                gl::gpu::DevicePodMapKind mapKind =
                    gl::gpu::DevicePodMapKind::localStatements;
                mandatoryViewMapKind(term.views[viewIndex], mapKind);
                if (findProjectedPodMapEntry(
                        columns, block, mapKind, packed) >= 0) {
                    mask |= static_cast<uint8_t>(
                        1u << (termIndex * 2 + viewIndex));
                }
            }
        }
        return mask;
    }

    /// @brief Build the full-view requirement mask for each mandatory term.
    ///
    /// @param batch Request batch naming zero to two terms.
    /// @param terms Uploaded global mandatory-term column.
    /// @param termMasks Two-element output array.
    /// @return Nothing.
    /// @invariant Unused output slots are zero.
    __device__ void buildGrowthTermMasks(
        const gl::gpu::DeviceRequestBatch& batch,
        const gl::gpu::DeviceMandatoryTerm* terms,
        uint8_t* termMasks) {
        termMasks[0] = 0;
        termMasks[1] = 0;
        assert(batch.termCount <= 2);
        for (uint32_t termIndex = 0;
             termIndex < batch.termCount; ++termIndex) {
            const gl::gpu::DeviceMandatoryTerm& term =
                terms[batch.termOffset + termIndex];
            assert(term.viewCount >= 1 && term.viewCount <= 2);
            termMasks[termIndex] = static_cast<uint8_t>(
                (1u << (termIndex * 2))
                | (term.viewCount == 2
                    ? (1u << (termIndex * 2 + 1)) : 0u));
        }
    }

    /// @brief Test whether a candidate already completes some mandatory term.
    ///
    /// @param candidateMask Union of statement view bits in the candidate.
    /// @param termMasks Complete-view masks for the batch terms.
    /// @param termCount Number of active terms.
    /// @return True when at least one term has every named view represented.
    /// @invariant `termCount` is between one and two.
    __device__ bool growthTermsSatisfied(
        uint8_t candidateMask,
        const uint8_t* termMasks,
        uint32_t termCount) {
        assert(termCount >= 1 && termCount <= 2);
        for (uint32_t term = 0; term < termCount; ++term) {
            if ((candidateMask & termMasks[term]) == termMasks[term])
                return true;
        }
        return false;
    }

    /// @brief Test whether a mandatory term can still complete from a suffix.
    ///
    /// @param candidateMask Union of current candidate view bits.
    /// @param suffixMask Union of view bits in remaining sorted positions.
    /// @param candidateCount Current premise count.
    /// @param targetLength Selected registry maximum key length.
    /// @param termMasks Complete-view masks for the batch terms.
    /// @param termCount Number of active terms.
    /// @return True when one term is complete or can be completed deeper.
    /// @invariant `termCount` is between one and two.
    __device__ bool growthTermsReachable(
        uint8_t candidateMask,
        uint8_t suffixMask,
        gl::NameId candidateCount,
        gl::NameId targetLength,
        const uint8_t* termMasks,
        uint32_t termCount) {
        assert(termCount >= 1 && termCount <= 2);
        for (uint32_t term = 0; term < termCount; ++term) {
            const uint8_t missing = static_cast<uint8_t>(
                termMasks[term] & ~candidateMask);
            if (missing == 0) return true;
            if (candidateCount < targetLength
                && (missing & suffixMask) == missing) return true;
        }
        return false;
    }

    /// @brief Apply all map-independent processor request-shape gates.
    ///
    /// @details
    /// Reproduces hypothesis-scope consensus with main-anchor exemption,
    /// distinct secondary-variable counting with recursion-product exclusion,
    /// scoped `_orint_` widening, and the overall key-length ceiling. The device
    /// recomputes the bounded summary from the candidate path; no proof state is
    /// mutated and no fallback result exists.
    ///
    /// @param columns Resident statement, name, and recursion-product columns.
    /// @param block Owning logical-block projection.
    /// @param statementIndices Candidate statement path.
    /// @param filteredPositions Candidate positions in the call's sorted span;
    ///                          `-1` is legal only for a whole-only stump event.
    /// @param count Number of premises in the path.
    /// @param parameters Analyzer request-shape limits.
    /// @return True exactly when the processor would proceed to map probes.
    /// @invariant Candidate premises are pairwise validity-comparable.
    __device__ bool projectedRequestGatesPass(
        const ProjectionSemanticColumns& columns,
        const gl::gpu::DeviceLogicalBlockProjection& block,
        const gl::NameId* statementIndices,
        gl::NameId count,
        gl::gpu::DevicePhase2GrowthParameters parameters) {
        assert(count >= 1);
        assert(count <= gl::ExecutionParameters::MAX_EXPRESSIONS);
        constexpr int32_t maximumSeenSecondary =
            (gl::ExecutionParameters::MAX_EXPRESSIONS + 2)
                * gl::ExecutionParameters::MAX_ARITY;
        gl::NameId seenSecondary[maximumSeenSecondary]{};
        int32_t secondaryCount = 0;
        bool hypothesisFound = false;
        gl::NameId hypothesisValidity = -1;
        int32_t nonExemptScopes = 0;
        gl::NameId nonExemptValidity = -1;

        for (gl::NameId premise = 0; premise < count; ++premise) {
            const gl::NameId statementIndex = statementIndices[premise];
            assert(statementIndex >= 0);
            assert(static_cast<uint32_t>(statementIndex)
                < block.statementCount);
            const gl::IntEncodedExpr& expression = columns.statements[
                block.statementOffset
                    + static_cast<uint32_t>(statementIndex)];
            if (expression.isHypo) {
                if (hypothesisFound
                    && expression.validityId != hypothesisValidity) {
                    return false;
                }
                hypothesisFound = true;
                hypothesisValidity = expression.validityId;
            }
            if (!(expression.validityId == block.mainValidityId
                  && expression.isAnchor)) {
                if (nonExemptScopes == 0) {
                    nonExemptScopes = 1;
                    nonExemptValidity = expression.validityId;
                }
                else if (nonExemptScopes == 1
                         && nonExemptValidity != expression.validityId) {
                    nonExemptScopes = 2;
                }
            }
            for (gl::NameId argument = 0;
                 argument < expression.arity; ++argument) {
                if (expression.argIteration[argument] <= -1) continue;
                if (findProjectedPodMapEntry(
                        columns, block,
                        gl::gpu::DevicePodMapKind::recursionProducts,
                        static_cast<int64_t>(
                            expression.argFullId[argument])) >= 0) {
                    continue;
                }
                const gl::NameId id = expression.argFullId[argument];
                bool found = false;
                for (int32_t known = 0;
                     known < secondaryCount; ++known) {
                    if (seenSecondary[known] == id) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    assert(secondaryCount < maximumSeenSecondary);
                    seenSecondary[secondaryCount++] = id;
                }
            }
        }

        if (hypothesisFound) {
            if (count > parameters.maximumHypothesisKeyLength) return false;
            if (nonExemptScopes > 1) return false;
            if (nonExemptScopes == 1
                && nonExemptValidity != hypothesisValidity) return false;
        }
        if (secondaryCount > parameters.maximumSecondaryVariables) {
            if (secondaryCount
                > parameters.maximumSecondaryVariablesOrint) return false;
            const gl::NameId scope = columns.statements[
                block.statementOffset
                    + static_cast<uint32_t>(statementIndices[0])].validityId;
            for (gl::NameId premise = 1; premise < count; ++premise) {
                if (columns.statements[
                        block.statementOffset
                            + static_cast<uint32_t>(
                                statementIndices[premise])].validityId
                    != scope) return false;
            }
            constexpr char orintLiteral[] = "_orint_";
            if (!projectedNameContains(
                    columns, block, scope, orintLiteral, 7)) return false;
        }
        return count <= block.overallMaxKeyLength;
    }

    /// @brief Return the selected registry's request-growth target length.
    ///
    /// @param block Resident logical-block scalar metadata.
    /// @param memory Selected hash-memory identity.
    /// @param output Registry maximum normalized-key premise count.
    /// @return Nothing.
    /// @invariant Every defined memory identity maps to one scalar field. The
    ///            advertised maximum may exceed the fixed candidate arena; the
    ///            growth kernel asserts only if a live frontier would cross it.
    __device__ void selectedGrowthTargetLength(
        const gl::gpu::DeviceLogicalBlockProjection& block,
        gl::gpu::DeviceHashMemoryKind memory,
        gl::NameId& output) {
        switch (memory) {
        case gl::gpu::DeviceHashMemoryKind::overall:
            output = block.overallMaxKeyLength;
            break;
        case gl::gpu::DeviceHashMemoryKind::local:
            output = block.localMaxKeyLength;
            break;
        case gl::gpu::DeviceHashMemoryKind::localDelta:
            output = block.deltaMaxKeyLength;
            break;
        case gl::gpu::DeviceHashMemoryKind::working:
            output = block.workingMaxKeyLength;
            break;
        default:
            assert(false);
        }
    }

    /// @brief Probe one complete candidate against selected whole and subkey maps.
    ///
    /// @details
    /// Builds the canonical serialized normalized key once, performs presence-only
    /// whole lookup, and applies the owner u_ signature gate to subkeys of three
    /// or more premises. One- and two-premise subkeys remain presence-only.
    ///
    /// @param columns Resident semantic columns.
    /// @param block Owning logical-block projection.
    /// @param memory Selected hash-memory registry.
    /// @param statementIndices Candidate statement path.
    /// @param count Number of premises.
    /// @param subkeySatisfied Output owner-accepted growth verdict.
    /// @param wholeKeyPresent Output whole-key presence verdict.
    /// @return Nothing.
    /// @invariant The normalized key fits the fixed stack payload.
    __device__ void probeProjectedGrowthCandidate(
        const ProjectionSemanticColumns& columns,
        const gl::gpu::DeviceLogicalBlockProjection& block,
        gl::gpu::DeviceHashMemoryKind memory,
        const gl::NameId* statementIndices,
        gl::NameId count,
        bool& subkeySatisfied,
        bool& wholeKeyPresent) {
        gl::NameId payload[gl::ExecutionParameters::MAX_KEY_SLOTS]{};
        const gl::NameId payloadLength = buildProjectedNormalizedKey(
            columns, block, statementIndices, count, payload);
        gl::NameId serialized[gl::ExecutionParameters::MAX_KEY_SLOTS + 2]{};
        serialized[0] = count;
        serialized[1] = payloadLength;
        for (gl::NameId index = 0; index < payloadLength; ++index)
            serialized[index + 2] = payload[index];
        const uint32_t serializedBytes = static_cast<uint32_t>(
            (payloadLength + 2) * sizeof(gl::NameId));
        gl::gpu::DeviceByteMapKind wholeKind =
            gl::gpu::DeviceByteMapKind::overallWholeKeys;
        gl::gpu::DeviceByteMapKind subkeyKind =
            gl::gpu::DeviceByteMapKind::overallSubkeys;
        filterMapKinds(memory, wholeKind, subkeyKind);
        const int32_t subkeyEntry = findProjectedByteMapEntry(
            columns, block, subkeyKind,
            reinterpret_cast<const char*>(serialized), serializedBytes);
        subkeySatisfied = false;
        if (subkeyEntry >= 0) {
            subkeySatisfied = count < 3
                || projectedOwnerSetUSatisfied(
                    columns, block,
                    columns.byteMapEntries[
                        static_cast<uint32_t>(subkeyEntry)],
                    statementIndices, count);
        }
        wholeKeyPresent = findProjectedByteMapEntry(
            columns, block, wholeKind,
            reinterpret_cast<const char*>(serialized), serializedBytes) >= 0;
    }

    /// @brief Append one persistent semantic growth event and optional request.
    ///
    /// @details
    /// Reserves fixed-ledger slots atomically, copies the logical statement path,
    /// and stores independent subkey, whole-key, and containment flags. A raw
    /// request is appended only for a whole-key hit that satisfies containment.
    /// Its growth position remains zero until processor-order reconstruction.
    ///
    /// @param statementIndices Candidate statement path.
    /// @param count Number of premises.
    /// @param callIndex Growth-call identity.
    /// @param runOrdinal Stump-run identity within the call.
    /// @param subkeySatisfied Owner-accepted subkey verdict.
    /// @param wholeKeyPresent Whole-key presence verdict.
    /// @param termsSatisfied Mandatory-containment verdict.
    /// @param taskIndex Executor task owning this growth call.
    /// @param events Persistent event output array.
    /// @param requests Persistent raw-request output array.
    /// @param taskSubkeyCounts Exact processor split-work counts by task.
    /// @param counters Shared append counters.
    /// @param capacity Fixed event and request ceilings.
    /// @return Nothing.
    /// @invariant At least one of subkey satisfaction or recordability is true.
    __device__ void appendProjectedGrowthEvent(
        const gl::NameId* statementIndices,
        const gl::NameId* filteredPositions,
        gl::NameId count,
        uint32_t callIndex,
        uint32_t runOrdinal,
        bool subkeySatisfied,
        bool wholeKeyPresent,
        bool termsSatisfied,
        uint32_t taskIndex,
        gl::gpu::DeviceAcceptedGrowthEvent* events,
        gl::gpu::DeviceRawGrowthRequest* requests,
        uint32_t* taskSubkeyCounts,
        DeviceGrowthCounters* counters,
        gl::gpu::Phase2GrowthCapacity capacity) {
        const bool recordable = wholeKeyPresent && termsSatisfied;
        assert(subkeySatisfied || recordable);
        assert(taskIndex < capacity.calls);
        if (subkeySatisfied)
            atomicAdd(&taskSubkeyCounts[taskIndex], 1u);
        const uint32_t eventIndex = atomicAdd(
            &counters->acceptedEventCount, 1u);
        assert(eventIndex < capacity.acceptedEvents);
        gl::gpu::DeviceAcceptedGrowthEvent event{};
        for (gl::NameId index = 0; index < count; ++index) {
            event.statementIndices[index] = statementIndices[index];
            assert(filteredPositions[index] >= -1);
            assert(filteredPositions[index]
                < static_cast<gl::NameId>(
                    kProcessorMaximumFilteredRows));
            event.filteredPositionCodes[index] = static_cast<uint16_t>(
                filteredPositions[index] + 1);
        }
        event.callIndex = callIndex;
        event.runOrdinal = runOrdinal;
        event.count = count;
        if (subkeySatisfied)
            event.flags |= gl::gpu::kDeviceGrowthEventSubkeySatisfied;
        if (wholeKeyPresent)
            event.flags |= gl::gpu::kDeviceGrowthEventWholeKeyPresent;
        if (termsSatisfied)
            event.flags |= gl::gpu::kDeviceGrowthEventTermsSatisfied;
        events[eventIndex] = event;
        if (recordable) {
            const uint32_t requestIndex = atomicAdd(
                &counters->rawRequestCount, 1u);
            assert(requestIndex < capacity.rawRequests);
            gl::gpu::DeviceRawGrowthRequest request{};
            request.event = event;
            requests[requestIndex] = request;
        }
    }

    /// @brief Build per-row mandatory-view and suffix masks for every growth call.
    ///
    /// @details
    /// One block owns one disjoint filtered call span. Threads compute direct view
    /// masks; thread zero folds the span backward into the exact suffix union used
    /// by the processor's mandatory-reachability gate.
    ///
    /// @param columns Resident semantic columns.
    /// @param tasks Uploaded task descriptors.
    /// @param taskCount Uploaded task count.
    /// @param batches Uploaded batch descriptors.
    /// @param batchCount Uploaded batch count.
    /// @param terms Uploaded mandatory-term descriptors.
    /// @param growthCalls Uploaded request-growth schedule.
    /// @param growthCallCount Number of scheduled request calls.
    /// @param filterCalls Uploaded filter schedule.
    /// @param filterCallCount Number of completed filter calls.
    /// @param filterCounts Per-filter retained counts.
    /// @param filterOffsets Per-filter retained offsets.
    /// @param sortedKeys Global sorted retained keys.
    /// @param viewMasks Per-row direct membership output.
    /// @param suffixMasks Per-row backward-union output.
    /// @return Nothing.
    /// @invariant Growth calls name distinct filter spans.
    __global__ void phase2GrowthMandatoryMaskKernel(
        ProjectionSemanticColumns columns,
        const gl::gpu::DevicePhase2Task* tasks,
        uint32_t taskCount,
        const gl::gpu::DeviceRequestBatch* batches,
        uint32_t batchCount,
        const gl::gpu::DeviceMandatoryTerm* terms,
        const gl::gpu::DevicePhase2GrowthCall* growthCalls,
        uint32_t growthCallCount,
        const gl::gpu::DevicePhase2FilterCall* filterCalls,
        uint32_t filterCallCount,
        const uint32_t* filterCounts,
        const uint32_t* filterOffsets,
        const uint64_t* sortedKeys,
        uint8_t* viewMasks,
        uint8_t* suffixMasks) {
        const uint32_t growthCallIndex = blockIdx.x;
        assert(growthCallIndex < growthCallCount);
        const gl::gpu::DevicePhase2GrowthCall& growthCall =
            growthCalls[growthCallIndex];
        assert(growthCall.taskIndex < taskCount);
        assert(growthCall.batchIndex < batchCount);
        assert(growthCall.filterCallIndex < filterCallCount);
        const gl::gpu::DevicePhase2Task& task = tasks[growthCall.taskIndex];
        assert(growthCall.batchIndex >= task.batchOffset);
        assert(growthCall.batchIndex < task.batchOffset + task.batchCount);
        const gl::gpu::DeviceRequestBatch& batch =
            batches[growthCall.batchIndex];
        const gl::gpu::DevicePhase2FilterCall& filterCall =
            filterCalls[growthCall.filterCallIndex];
        assert(filterCall.logicalBlockIndex == task.logicalBlockIndex);
        assert(filterCall.memory == batch.memory);
        const gl::gpu::DeviceLogicalBlockProjection& block =
            columns.logicalBlocks[task.logicalBlockIndex];
        const uint32_t count = filterCounts[growthCall.filterCallIndex];
        const uint32_t offset = filterOffsets[growthCall.filterCallIndex];
        for (uint32_t row = threadIdx.x; row < count;
             row += blockDim.x) {
            const gl::NameId statementIndex = growthStatementIndex(
                sortedKeys, offset + row);
            viewMasks[offset + row] = projectedStatementViewMask(
                columns, block, batch, terms, statementIndex);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            uint8_t suffix = 0;
            for (uint32_t row = count; row > 0; --row) {
                suffix = static_cast<uint8_t>(
                    suffix | viewMasks[offset + row - 1]);
                suffixMasks[offset + row - 1] = suffix;
            }
        }
    }

    /// @brief Seed unsplit roots and exact stump nodes for all request calls.
    ///
    /// @details
    /// Each thread owns one growth call. It emits the stump-alone probe exactly,
    /// including whole-only requests, and appends only owner-accepted nonterminal
    /// stumps to the first frontier. Unsplit calls append one empty main-scope
    /// root. Missing filtered positions are legal only for a whole-only stump.
    ///
    /// @param columns Resident semantic columns.
    /// @param tasks Uploaded task descriptors.
    /// @param batches Uploaded batch descriptors.
    /// @param stumps Uploaded expression stumps.
    /// @param growthCalls Uploaded request-growth schedule.
    /// @param growthCallCount Number of scheduled calls.
    /// @param filterCounts Per-filter retained counts.
    /// @param filterOffsets Per-filter retained offsets.
    /// @param sortedKeys Global sorted retained keys.
    /// @param viewMasks Per-row mandatory-view bits.
    /// @param parameters Analyzer request-shape limits.
    /// @param seedDepth Absolute candidate depth owned by this launch.
    /// @param frontier First frontier output.
    /// @param events Persistent semantic-event output.
    /// @param requests Persistent raw-request output.
    /// @param taskSubkeyCounts Exact processor split-work counts by task.
    /// @param counters Shared append counters.
    /// @param capacity Fixed growth ceilings.
    /// @return Nothing.
    /// @invariant One thread owns every call and its stump run.
    __global__ void phase2GrowthSeedKernel(
        ProjectionSemanticColumns columns,
        const gl::gpu::DevicePhase2Task* tasks,
        const gl::gpu::DeviceRequestBatch* batches,
        const gl::gpu::DeviceMandatoryTerm* terms,
        const gl::gpu::DeviceExpressionStump* stumps,
        const gl::gpu::DevicePhase2GrowthCall* growthCalls,
        uint32_t growthCallCount,
        const uint32_t* filterCounts,
        const uint32_t* filterOffsets,
        const uint64_t* sortedKeys,
        const uint8_t* viewMasks,
        gl::gpu::DevicePhase2GrowthParameters parameters,
        uint32_t seedDepth,
        gl::gpu::DeviceGrowthNode* frontier,
        gl::gpu::DeviceAcceptedGrowthEvent* events,
        gl::gpu::DeviceRawGrowthRequest* requests,
        uint32_t* taskSubkeyCounts,
        DeviceGrowthCounters* counters,
        gl::gpu::Phase2GrowthCapacity capacity) {
        const uint32_t growthCallIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if (growthCallIndex >= growthCallCount) return;
        const gl::gpu::DevicePhase2GrowthCall& growthCall =
            growthCalls[growthCallIndex];
        const gl::gpu::DevicePhase2Task& task = tasks[growthCall.taskIndex];
        const gl::gpu::DeviceRequestBatch& batch =
            batches[growthCall.batchIndex];
        const gl::gpu::DeviceLogicalBlockProjection& block =
            columns.logicalBlocks[task.logicalBlockIndex];
        gl::NameId targetLength = 0;
        selectedGrowthTargetLength(block, batch.memory, targetLength);
        if (targetLength <= 0) return;
        const uint32_t filteredCount =
            filterCounts[growthCall.filterCallIndex];
        const uint32_t filteredOffset =
            filterOffsets[growthCall.filterCallIndex];

        if (task.stumpCount == 0) {
            if (seedDepth != 0) return;
            const uint32_t frontierIndex = atomicAdd(
                &counters->frontierCounts[0], 1u);
            assert(frontierIndex < capacity.frontierRecords);
            gl::gpu::DeviceGrowthNode root{};
            root.callIndex = growthCallIndex;
            root.validityId = block.mainValidityId;
            frontier[frontierIndex] = root;
            return;
        }
        if (seedDepth == 0) return;

        uint8_t termMasks[2]{};
        buildGrowthTermMasks(batch, terms, termMasks);
        for (uint32_t runOrdinal = 0;
             runOrdinal < task.stumpCount; ++runOrdinal) {
            const gl::gpu::DeviceExpressionStump& stump =
                stumps[task.stumpOffset + runOrdinal];
            assert(stump.count > 0);
            if (static_cast<uint32_t>(stump.count) != seedDepth) continue;
            if (stump.count > targetLength) continue;
            gl::NameId filteredPositions[
                gl::ExecutionParameters::MAX_EXPRESSIONS]{};
            bool everyPositionFound = true;
            uint8_t candidateMask = 0;
            gl::NameId validity = block.mainValidityId;
            for (gl::NameId premise = 0;
                 premise < stump.count; ++premise) {
                const gl::NameId statementIndex =
                    stump.statementIndices[premise];
                const gl::IntEncodedExpr& expression = columns.statements[
                    block.statementOffset
                        + static_cast<uint32_t>(statementIndex)];
                if (premise == 0) validity = expression.validityId;
                else {
                    assert(projectedValiditiesComparable(
                        columns, block, validity, expression.validityId));
                    validity = projectedDeeperValidity(
                        columns, block, validity, expression.validityId);
                }
                gl::NameId foundPosition = -1;
                for (uint32_t row = 0; row < filteredCount; ++row) {
                    if (growthStatementIndex(
                            sortedKeys, filteredOffset + row)
                        == statementIndex) {
                        foundPosition = static_cast<gl::NameId>(row);
                        break;
                    }
                }
                filteredPositions[premise] = foundPosition;
                if (foundPosition < 0) everyPositionFound = false;
                else candidateMask = static_cast<uint8_t>(
                    candidateMask
                    | viewMasks[filteredOffset
                        + static_cast<uint32_t>(foundPosition)]);
            }
            if (batch.termCount > 0 && !everyPositionFound) continue;
            if (!projectedRequestGatesPass(
                    columns, block, stump.statementIndices,
                    stump.count, parameters)) continue;
            bool subkeySatisfied = false;
            bool wholeKeyPresent = false;
            probeProjectedGrowthCandidate(
                columns, block, batch.memory, stump.statementIndices,
                stump.count, subkeySatisfied, wholeKeyPresent);
            const bool termsSatisfied = batch.termCount == 0
                || growthTermsSatisfied(
                    candidateMask, termMasks, batch.termCount);
            if (subkeySatisfied || (wholeKeyPresent && termsSatisfied)) {
                appendProjectedGrowthEvent(
                    stump.statementIndices, filteredPositions, stump.count,
                    growthCallIndex, runOrdinal,
                    subkeySatisfied, wholeKeyPresent, termsSatisfied,
                    growthCall.taskIndex, events, requests,
                    taskSubkeyCounts, counters, capacity);
            }
            if (!subkeySatisfied || stump.terminalOnly != 0
                || stump.count >= targetLength) continue;
            assert(everyPositionFound);
            for (gl::NameId premise = 1;
                 premise < stump.count; ++premise) {
                assert(filteredPositions[premise]
                    > filteredPositions[premise - 1]);
            }
            const uint32_t frontierIndex = atomicAdd(
                &counters->frontierCounts[0], 1u);
            assert(frontierIndex < capacity.frontierRecords);
            gl::gpu::DeviceGrowthNode node{};
            node.callIndex = growthCallIndex;
            node.runOrdinal = runOrdinal;
            node.count = stump.count;
            node.startPosition = filteredPositions[stump.count - 1] + 1;
            node.validityId = validity;
            node.termMask = candidateMask;
            for (gl::NameId premise = 0;
                 premise < stump.count; ++premise) {
                node.filteredPositions[premise] =
                    filteredPositions[premise];
            }
            frontier[frontierIndex] = node;
        }
    }

    /// @brief Count the remaining candidate-span shape of one live frontier.
    ///
    /// @details
    /// Assigns one thread to each immutable frontier node, derives the exact
    /// number of later filtered positions, and accumulates node and candidate
    /// totals into twelve power-of-two buckets. Each block first combines into
    /// shared counters so only twelve pairs of global atomics are issued. The
    /// counters are observation only and no semantic kernel reads them.
    ///
    /// @param growthCalls Uploaded growth schedule selecting filter calls.
    /// @param filterCounts Per-filter retained counts.
    /// @param current Current immutable frontier input.
    /// @param currentCount Current used node count.
    /// @param counters Shared observation-counter destination.
    /// @return Nothing.
    /// @invariant Every node start position lies inside its selected filter span.
    __global__ void phase2GrowthSpanCensusKernel(
        const gl::gpu::DevicePhase2GrowthCall* growthCalls,
        const uint32_t* filterCounts,
        const gl::gpu::DeviceGrowthNode* current,
        uint32_t currentCount,
        DeviceGrowthCounters* counters) {
        __shared__ unsigned long long blockNodeCounts[
            gl::gpu::kDeviceGrowthSpanBucketCount];
        __shared__ unsigned long long blockCandidateCounts[
            gl::gpu::kDeviceGrowthSpanBucketCount];
        if (threadIdx.x < gl::gpu::kDeviceGrowthSpanBucketCount) {
            blockNodeCounts[threadIdx.x] = 0;
            blockCandidateCounts[threadIdx.x] = 0;
        }
        __syncthreads();

        const uint32_t nodeIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if (nodeIndex < currentCount) {
            const gl::gpu::DeviceGrowthNode& node = current[nodeIndex];
            const gl::gpu::DevicePhase2GrowthCall& growthCall =
                growthCalls[node.callIndex];
            const uint32_t filteredCount =
                filterCounts[growthCall.filterCallIndex];
            assert(node.startPosition >= 0);
            assert(static_cast<uint32_t>(node.startPosition) <= filteredCount);
            const uint32_t span = filteredCount
                - static_cast<uint32_t>(node.startPosition);
            uint32_t bucket = 0;
            if (span > 0) {
                bucket = 1;
                uint32_t upper = 1;
                while (bucket + 1 < gl::gpu::kDeviceGrowthSpanBucketCount
                       && span > upper) {
                    ++bucket;
                    upper = (upper << 1) | 1u;
                }
            }
            atomicAdd(&blockNodeCounts[bucket], 1ull);
            atomicAdd(
                &blockCandidateCounts[bucket],
                static_cast<unsigned long long>(span));
        }
        __syncthreads();

        if (threadIdx.x < gl::gpu::kDeviceGrowthSpanBucketCount) {
            atomicAdd(
                &counters->spanNodeCounts[threadIdx.x],
                blockNodeCounts[threadIdx.x]);
            atomicAdd(
                &counters->spanCandidateCounts[threadIdx.x],
                blockCandidateCounts[threadIdx.x]);
        }
    }

    /// @brief Materialize one immutable pooled prefix for every live growth node.
    ///
    /// @details
    /// Assigns one thread to each node, classifies its remaining span into the
    /// fixed short or cooperative flag column, reconstructs the normalized prefix
    /// and order-free request summaries once, reserves exact slices in the three
    /// fixed per-wave pools, and publishes one header at the original node index.
    /// Stable device selection later compacts each flag column over ascending
    /// frontier indices. Pool allocation order is non-semantic because every
    /// header carries its exact slices and no later ordering key reads an offset.
    ///
    /// @param columns Resident semantic columns.
    /// @param batches Uploaded request batches.
    /// @param terms Uploaded mandatory terms.
    /// @param tasks Uploaded task descriptors.
    /// @param growthCalls Uploaded growth schedule.
    /// @param filterCounts Per-filter retained counts.
    /// @param filterOffsets Per-filter retained offsets.
    /// @param sortedKeys Global sorted retained keys.
    /// @param parameters Analyzer request-shape limits.
    /// @param current Current frontier input.
    /// @param currentCount Current used node count.
    /// @param prefixes Header output indexed exactly like @p current.
    /// @param prefixPayload Fixed normalized-payload pool.
    /// @param prefixVariables Fixed distinct normalization-variable pool.
    /// @param prefixSecondary Fixed distinct secondary-variable pool.
    /// @param shortNodeFlags One stable-selection flag per frontier node.
    /// @param cooperativeNodeFlags One stable-selection flag per frontier node.
    /// @param counters Shared prefix-pool reservation counters.
    /// @param capacity Fixed frontier and prefix-pool ceilings.
    /// @return Nothing.
    /// @invariant Every live node below the expression ceiling sets exactly one
    ///            flag and owns three in-capacity immutable slices.
    __global__ void phase2GrowthPreparePrefixesKernel(
        ProjectionSemanticColumns columns,
        const gl::gpu::DeviceRequestBatch* batches,
        const gl::gpu::DeviceMandatoryTerm* terms,
        const gl::gpu::DevicePhase2Task* tasks,
        const gl::gpu::DevicePhase2GrowthCall* growthCalls,
        const uint32_t* filterCounts,
        const uint32_t* filterOffsets,
        const uint64_t* sortedKeys,
        gl::gpu::DevicePhase2GrowthParameters parameters,
        const gl::gpu::DeviceGrowthNode* current,
        uint32_t currentCount,
        gl::gpu::DeviceGrowthPrefix* prefixes,
        gl::NameId* prefixPayload,
        gl::NameId* prefixVariables,
        gl::NameId* prefixSecondary,
        uint8_t* shortNodeFlags,
        uint8_t* cooperativeNodeFlags,
        DeviceGrowthCounters* counters,
        gl::gpu::Phase2GrowthCapacity capacity) {
        const uint32_t nodeIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if (nodeIndex >= currentCount) return;
        const gl::gpu::DeviceGrowthNode& node = current[nodeIndex];
        const gl::gpu::DevicePhase2GrowthCall& growthCall =
            growthCalls[node.callIndex];
        const gl::gpu::DevicePhase2Task& task = tasks[growthCall.taskIndex];
        const gl::gpu::DeviceRequestBatch& batch =
            batches[growthCall.batchIndex];
        const gl::gpu::DeviceLogicalBlockProjection& block =
            columns.logicalBlocks[task.logicalBlockIndex];
        const uint32_t filteredCount =
            filterCounts[growthCall.filterCallIndex];
        const uint32_t filteredOffset =
            filterOffsets[growthCall.filterCallIndex];
        assert(parameters.cooperativeSpanThreshold > 0);
        assert(node.startPosition >= 0);
        assert(static_cast<uint32_t>(node.startPosition) <= filteredCount);
        if (node.count == gl::ExecutionParameters::MAX_EXPRESSIONS) {
            shortNodeFlags[nodeIndex] = 0;
            cooperativeNodeFlags[nodeIndex] = 0;
            assert(node.startPosition >= static_cast<gl::NameId>(filteredCount)
                && "a live request frontier exceeds MAX_EXPRESSIONS");
            return;
        }
        assert(node.count < gl::ExecutionParameters::MAX_EXPRESSIONS);
        const uint32_t remainingSpan = filteredCount
            - static_cast<uint32_t>(node.startPosition);
        const bool cooperative =
            remainingSpan >= parameters.cooperativeSpanThreshold;
        shortNodeFlags[nodeIndex] = cooperative ? 0 : 1;
        cooperativeNodeFlags[nodeIndex] = cooperative ? 1 : 0;

        constexpr gl::NameId maximumPrefixArguments =
            (gl::ExecutionParameters::MAX_EXPRESSIONS - 1)
                * gl::ExecutionParameters::MAX_ARITY;
        gl::NameId localPayload[
            gl::ExecutionParameters::MAX_KEY_SLOTS]{};
        gl::NameId localVariables[maximumPrefixArguments]{};
        gl::NameId localSecondary[maximumPrefixArguments]{};
        gl::NameId payloadLength = 0;
        gl::NameId variableCount = 0;
        gl::NameId secondaryCount = 0;
        bool hypothesisFound = false;
        gl::NameId hypothesisValidity = -1;
        int32_t nonExemptScopes = 0;
        gl::NameId nonExemptValidity = -1;
        uint8_t termMasks[2]{};
        buildGrowthTermMasks(batch, terms, termMasks);
        for (gl::NameId premise = 0; premise < node.count; ++premise) {
            assert(node.filteredPositions[premise] >= 0);
            const gl::NameId statementIndex = growthStatementIndex(
                sortedKeys, filteredOffset + static_cast<uint32_t>(
                    node.filteredPositions[premise]));
            const gl::IntEncodedExpr& expression = columns.statements[
                block.statementOffset + static_cast<uint32_t>(statementIndex)];
            assert(payloadLength + 2
                <= gl::ExecutionParameters::MAX_KEY_SLOTS);
            localPayload[payloadLength++] = expression.nameId;
            localPayload[payloadLength++] = expression.negation;
            for (gl::NameId argument = 0;
                 argument < expression.arity; ++argument) {
                assert(payloadLength + 2
                    <= gl::ExecutionParameters::MAX_KEY_SLOTS);
                const gl::NameId variableId = expression.argId[argument];
                gl::NameId normalizedId = 0;
                for (gl::NameId known = 0; known < variableCount; ++known) {
                    if (localVariables[known] == variableId) {
                        normalizedId = known + 1;
                        break;
                    }
                }
                if (normalizedId == 0) {
                    assert(variableCount < maximumPrefixArguments);
                    localVariables[variableCount++] = variableId;
                    normalizedId = variableCount;
                }
                localPayload[payloadLength++] = normalizedId;
                localPayload[payloadLength++] = 0;
            }
            if (expression.isHypo) {
                assert(!hypothesisFound
                    || expression.validityId == hypothesisValidity);
                hypothesisFound = true;
                hypothesisValidity = expression.validityId;
            }
            if (!(expression.validityId == block.mainValidityId
                  && expression.isAnchor)) {
                if (nonExemptScopes == 0) {
                    nonExemptScopes = 1;
                    nonExemptValidity = expression.validityId;
                }
                else if (nonExemptScopes == 1
                         && nonExemptValidity != expression.validityId) {
                    nonExemptScopes = 2;
                }
            }
            for (gl::NameId argument = 0;
                 argument < expression.arity; ++argument) {
                if (expression.argIteration[argument] <= -1) continue;
                if (findProjectedPodMapEntry(
                        columns, block,
                        gl::gpu::DevicePodMapKind::recursionProducts,
                        static_cast<int64_t>(
                            expression.argFullId[argument])) >= 0) {
                    continue;
                }
                const gl::NameId id = expression.argFullId[argument];
                bool found = false;
                for (gl::NameId known = 0; known < secondaryCount; ++known) {
                    if (localSecondary[known] == id) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    assert(secondaryCount < maximumPrefixArguments);
                    localSecondary[secondaryCount++] = id;
                }
            }
        }
        assert(payloadLength <= 0xffff);
        assert(variableCount <= 0xffff);
        assert(secondaryCount <= 0xffff);
        const uint32_t payloadOffset = atomicAdd(
            &counters->prefixPayloadCount,
            static_cast<uint32_t>(payloadLength));
        const uint32_t variableOffset = atomicAdd(
            &counters->prefixVariableCount,
            static_cast<uint32_t>(variableCount));
        const uint32_t secondaryOffset = atomicAdd(
            &counters->prefixSecondaryCount,
            static_cast<uint32_t>(secondaryCount));
        assert(payloadOffset + static_cast<uint32_t>(payloadLength)
            <= capacity.prefixPayloadValues);
        assert(variableOffset + static_cast<uint32_t>(variableCount)
            <= capacity.prefixVariableValues);
        assert(secondaryOffset + static_cast<uint32_t>(secondaryCount)
            <= capacity.prefixSecondaryValues);
        for (gl::NameId index = 0; index < payloadLength; ++index)
            prefixPayload[payloadOffset + static_cast<uint32_t>(index)] =
                localPayload[index];
        for (gl::NameId index = 0; index < variableCount; ++index)
            prefixVariables[variableOffset + static_cast<uint32_t>(index)] =
                localVariables[index];
        for (gl::NameId index = 0; index < secondaryCount; ++index)
            prefixSecondary[secondaryOffset + static_cast<uint32_t>(index)] =
                localSecondary[index];
        gl::gpu::DeviceGrowthPrefix prefix{};
        prefix.payloadOffset = payloadOffset;
        prefix.variableOffset = variableOffset;
        prefix.secondaryOffset = secondaryOffset;
        prefix.payloadLength = static_cast<uint16_t>(payloadLength);
        prefix.variableCount = static_cast<uint16_t>(variableCount);
        prefix.secondaryCount = static_cast<uint16_t>(secondaryCount);
        prefix.hypothesisValidity = hypothesisValidity;
        prefix.nonExemptValidity = nonExemptValidity;
        prefix.packedSummary = (hypothesisFound ? 1u : 0u)
            | (static_cast<uint32_t>(nonExemptScopes) << 1)
            | (static_cast<uint32_t>(termMasks[0]) << 8)
            | (static_cast<uint32_t>(termMasks[1]) << 16);
        prefixes[nodeIndex] = prefix;
    }

    /// @brief Merge the stable short and cooperative node lists for semantic sorting.
    ///
    /// @details
    /// Copies both complementary compact lists into one frontier-sized column.
    /// Their concatenation order is deliberately non-semantic; the following
    /// stable radix passes establish exact `(call, run, path)` order.
    ///
    /// @param shortNodeIndices Stable short-node indices.
    /// @param shortNodeCount Number of short indices.
    /// @param cooperativeNodeIndices Stable long-node indices.
    /// @param cooperativeNodeCount Number of long indices.
    /// @param mergedNodeIndices Complete merged output.
    /// @param frontierCount Exact live frontier size.
    /// @return Nothing.
    /// @invariant The two inputs are complementary and their counts sum to the
    ///            live frontier size.
    __global__ void phase2GrowthMergeNodeIndicesKernel(
        const uint32_t* shortNodeIndices,
        uint32_t shortNodeCount,
        const uint32_t* cooperativeNodeIndices,
        uint32_t cooperativeNodeCount,
        uint32_t* mergedNodeIndices,
        uint32_t frontierCount) {
        assert(shortNodeCount + cooperativeNodeCount == frontierCount);
        const uint32_t outputIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if (outputIndex >= frontierCount) return;
        mergedNodeIndices[outputIndex] = outputIndex < shortNodeCount
            ? shortNodeIndices[outputIndex]
            : cooperativeNodeIndices[outputIndex - shortNodeCount];
    }

    /// @brief Build one stable-radix key column for canonical growth-node order.
    ///
    /// @details
    /// Reads node indices in the current stable order and emits either one path
    /// position, the stump-run ordinal, or the growth-call ordinal. Host code sorts
    /// path fields from last to first, then run, then call, so the final index
    /// stream is lexicographic `(call, run, path)` without moving frontier rows.
    ///
    /// @param frontier Immutable live frontier.
    /// @param nodeIndices Current stable node-index order.
    /// @param nodeCount Number of live nodes.
    /// @param field Path field in `[0,7]`, eight for run, or nine for call.
    /// @param keys Radix-key output aligned with @p nodeIndices.
    /// @return Nothing.
    /// @invariant Every requested path field exists in every depth-homogeneous
    ///            node and every emitted key is an unsigned semantic scalar.
    __global__ void phase2GrowthNodeSortKeyKernel(
        const gl::gpu::DeviceGrowthNode* frontier,
        const uint32_t* nodeIndices,
        uint32_t nodeCount,
        uint32_t field,
        uint64_t* keys) {
        const uint32_t orderIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if (orderIndex >= nodeCount) return;
        assert(field <= gl::ExecutionParameters::MAX_EXPRESSIONS + 1u);
        const gl::gpu::DeviceGrowthNode& node = frontier[
            nodeIndices[orderIndex]];
        uint32_t key = 0;
        if (field < gl::ExecutionParameters::MAX_EXPRESSIONS) {
            assert(field < static_cast<uint32_t>(node.count));
            assert(node.filteredPositions[field] >= 0);
            key = static_cast<uint32_t>(node.filteredPositions[field]);
        }
        else if (field == gl::ExecutionParameters::MAX_EXPRESSIONS) {
            key = node.runOrdinal;
        }
        else {
            key = node.callIndex;
        }
        keys[orderIndex] = key;
    }

    /// @brief Emit candidate-span lengths in canonical node order.
    ///
    /// @details
    /// Each value is the exact number of later filtered positions owned by one
    /// node. A fixed inclusive scan converts these lengths into canonical attempt
    /// end ordinals for deterministic window slicing.
    ///
    /// @param frontier Immutable live frontier.
    /// @param canonicalNodeIndices Lexicographic `(call, run, path)` node order.
    /// @param nodeCount Number of live nodes.
    /// @param growthCalls Uploaded growth schedule.
    /// @param filterCounts Per-filter retained counts.
    /// @param candidateCounts Exact 64-bit span-length output.
    /// @return Nothing.
    /// @invariant Every node cursor lies inside its retained filter span.
    __global__ void phase2GrowthCandidateCountsKernel(
        const gl::gpu::DeviceGrowthNode* frontier,
        const uint32_t* canonicalNodeIndices,
        uint32_t nodeCount,
        const gl::gpu::DevicePhase2GrowthCall* growthCalls,
        const uint32_t* filterCounts,
        uint64_t* candidateCounts) {
        const uint32_t orderIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if (orderIndex >= nodeCount) return;
        const gl::gpu::DeviceGrowthNode& node = frontier[
            canonicalNodeIndices[orderIndex]];
        const gl::gpu::DevicePhase2GrowthCall& growthCall = growthCalls[
            node.callIndex];
        const uint32_t filteredCount = filterCounts[
            growthCall.filterCallIndex];
        assert(node.startPosition >= 0);
        assert(static_cast<uint32_t>(node.startPosition) <= filteredCount);
        candidateCounts[orderIndex] = filteredCount
            - static_cast<uint32_t>(node.startPosition);
    }

    /// @brief Scatter canonical candidate start ordinals back to frontier indices.
    ///
    /// @details
    /// Converts the inclusive end scan into one 64-bit start ordinal per original
    /// node. Short and cooperative cheap-gate kernels can then write their disjoint
    /// attempts directly into a shared canonical window without binary searches or
    /// atomic placement.
    ///
    /// @param canonicalNodeIndices Lexicographic node order.
    /// @param candidateEnds Inclusive-scan end ordinal per canonical node.
    /// @param nodeCount Number of live nodes.
    /// @param candidateStartsByNode Start ordinal indexed by original node index.
    /// @return Nothing.
    /// @invariant Candidate ends are nondecreasing and the first start is zero.
    __global__ void phase2GrowthCandidateStartsKernel(
        const uint32_t* canonicalNodeIndices,
        const uint64_t* candidateEnds,
        uint32_t nodeCount,
        uint64_t* candidateStartsByNode) {
        const uint32_t orderIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if (orderIndex >= nodeCount) return;
        if (orderIndex > 0)
            assert(candidateEnds[orderIndex] >= candidateEnds[orderIndex - 1]);
        candidateStartsByNode[canonicalNodeIndices[orderIndex]] =
            orderIndex == 0 ? 0 : candidateEnds[orderIndex - 1];
    }

    /// @brief Reconstruct one statement index from a pooled node plus candidate.
    ///
    /// @details
    /// Reads a retained filtered position for an existing premise or the appended
    /// candidate position and decodes the statement-index field from the shared
    /// class span. This removes per-thread statement-path arrays from production
    /// window kernels.
    ///
    /// @param sortedKeys Global sorted retained keys.
    /// @param filteredOffset Owning class-span offset.
    /// @param node Immutable prefix node.
    /// @param candidatePosition Appended filtered position.
    /// @param premise Premise ordinal in the extended candidate.
    /// @return Logical-block-local statement index.
    /// @invariant `premise` is at most `node.count` and every selected position is
    ///            a nonnegative member of the class span.
    __device__ gl::NameId pooledGrowthStatementIndex(
        const uint64_t* sortedKeys,
        uint32_t filteredOffset,
        const gl::gpu::DeviceGrowthNode& node,
        uint32_t candidatePosition,
        gl::NameId premise) {
        assert(premise >= 0);
        assert(premise <= node.count);
        const uint32_t position = premise == node.count
            ? candidatePosition
            : static_cast<uint32_t>(node.filteredPositions[premise]);
        if (premise < node.count)
            assert(node.filteredPositions[premise] >= 0);
        return growthStatementIndex(sortedKeys, filteredOffset + position);
    }

    /// @brief Apply all pre-map gates to one pooled candidate attempt.
    ///
    /// @details
    /// Extends the immutable prefix summaries with one expression and evaluates
    /// mandatory reachability, validity ancestry, hypothesis/scope compatibility,
    /// secondary-variable limits, and maximum key length. It writes the compact
    /// candidate record only after every cheap gate passes; normalized-key and
    /// owner-map work is deliberately excluded for survivor compaction.
    ///
    /// @param columns Resident semantic columns.
    /// @param batches Uploaded request batches.
    /// @param tasks Uploaded task descriptors.
    /// @param growthCalls Uploaded growth schedule.
    /// @param filterCounts Per-filter retained counts.
    /// @param filterOffsets Per-filter retained offsets.
    /// @param sortedKeys Global sorted retained keys.
    /// @param viewMasks Per-row mandatory-view bits.
    /// @param suffixMasks Per-row suffix unions.
    /// @param parameters Analyzer request-shape limits.
    /// @param frontier Immutable current frontier.
    /// @param prefixes Immutable pooled-prefix headers.
    /// @param prefixSecondary Fixed distinct secondary-variable pool.
    /// @param nodeIndex Original frontier node index.
    /// @param positionValue Appended filtered position.
    /// @param output Compact survivor output.
    /// @return True exactly when every gate through maximum key length passes.
    /// @invariant The node and position identify one live in-span attempt.
    __device__ bool preparePooledGrowthCandidate(
        ProjectionSemanticColumns columns,
        const gl::gpu::DeviceRequestBatch* batches,
        const gl::gpu::DevicePhase2Task* tasks,
        const gl::gpu::DevicePhase2GrowthCall* growthCalls,
        const uint32_t* filterCounts,
        const uint32_t* filterOffsets,
        const uint64_t* sortedKeys,
        const uint8_t* viewMasks,
        const uint8_t* suffixMasks,
        gl::gpu::DevicePhase2GrowthParameters parameters,
        const gl::gpu::DeviceGrowthNode* frontier,
        const gl::gpu::DeviceGrowthPrefix* prefixes,
        const gl::NameId* prefixSecondary,
        uint32_t nodeIndex,
        uint32_t positionValue,
        gl::gpu::DeviceGrowthCandidateAttempt& output) {
        const gl::gpu::DeviceGrowthNode& node = frontier[nodeIndex];
        const gl::gpu::DeviceGrowthPrefix& prefix = prefixes[nodeIndex];
        const gl::gpu::DevicePhase2GrowthCall& growthCall = growthCalls[
            node.callIndex];
        const gl::gpu::DevicePhase2Task& task = tasks[growthCall.taskIndex];
        const gl::gpu::DeviceRequestBatch& batch = batches[
            growthCall.batchIndex];
        const gl::gpu::DeviceLogicalBlockProjection& block =
            columns.logicalBlocks[task.logicalBlockIndex];
        const uint32_t filteredCount = filterCounts[
            growthCall.filterCallIndex];
        const uint32_t filteredOffset = filterOffsets[
            growthCall.filterCallIndex];
        assert(node.startPosition >= 0);
        assert(positionValue >= static_cast<uint32_t>(node.startPosition));
        assert(positionValue < filteredCount);
        const gl::NameId newCount = node.count + 1;
        gl::NameId targetLength = 0;
        selectedGrowthTargetLength(block, batch.memory, targetLength);
        assert(newCount <= targetLength);
        const uint32_t globalRow = filteredOffset + positionValue;
        const gl::NameId statementIndex = growthStatementIndex(
            sortedKeys, globalRow);
        const gl::IntEncodedExpr& expression = columns.statements[
            block.statementOffset + static_cast<uint32_t>(statementIndex)];
        const uint8_t candidateMask = batch.termCount == 0
            ? 0 : static_cast<uint8_t>(node.termMask | viewMasks[globalRow]);
        if (batch.termCount > 0) {
            const uint8_t termMasks[2]{
                static_cast<uint8_t>((prefix.packedSummary >> 8) & 0xffu),
                static_cast<uint8_t>((prefix.packedSummary >> 16) & 0xffu) };
            const uint8_t suffix = positionValue + 1 < filteredCount
                ? suffixMasks[globalRow + 1] : 0;
            if (!growthTermsReachable(
                    candidateMask, suffix, newCount, targetLength,
                    termMasks, batch.termCount)) return false;
        }
        if (!projectedValiditiesComparable(
                columns, block, node.validityId, expression.validityId)) {
            return false;
        }
        const gl::NameId newValidity = projectedDeeperValidity(
            columns, block, node.validityId, expression.validityId);

        bool hypothesisFound = (prefix.packedSummary & 1u) != 0;
        gl::NameId hypothesisValidity = prefix.hypothesisValidity;
        int32_t nonExemptScopes = static_cast<int32_t>(
            (prefix.packedSummary >> 1) & 0x3u);
        gl::NameId nonExemptValidity = prefix.nonExemptValidity;
        if (expression.isHypo) {
            if (hypothesisFound
                && expression.validityId != hypothesisValidity) return false;
            hypothesisFound = true;
            hypothesisValidity = expression.validityId;
        }
        if (!(expression.validityId == block.mainValidityId
              && expression.isAnchor)) {
            if (nonExemptScopes == 0) {
                nonExemptScopes = 1;
                nonExemptValidity = expression.validityId;
            }
            else if (nonExemptScopes == 1
                     && nonExemptValidity != expression.validityId) {
                nonExemptScopes = 2;
            }
        }

        constexpr gl::NameId maximumCandidateSecondary =
            gl::ExecutionParameters::MAX_EXPRESSIONS
                * gl::ExecutionParameters::MAX_ARITY;
        int32_t secondaryCount = prefix.secondaryCount;
        for (gl::NameId argument = 0;
             argument < expression.arity; ++argument) {
            if (expression.argIteration[argument] <= -1) continue;
            if (findProjectedPodMapEntry(
                    columns, block,
                    gl::gpu::DevicePodMapKind::recursionProducts,
                    static_cast<int64_t>(expression.argFullId[argument])) >= 0) {
                continue;
            }
            const gl::NameId id = expression.argFullId[argument];
            bool found = false;
            for (int32_t known = 0;
                 known < prefix.secondaryCount; ++known) {
                if (prefixSecondary[
                        prefix.secondaryOffset
                            + static_cast<uint32_t>(known)] == id) {
                    found = true;
                    break;
                }
            }
            for (gl::NameId prior = 0;
                 !found && prior < argument; ++prior) {
                if (expression.argIteration[prior] > -1
                    && expression.argFullId[prior] == id) found = true;
            }
            if (!found) {
                ++secondaryCount;
                assert(secondaryCount <= maximumCandidateSecondary);
            }
        }
        if (hypothesisFound) {
            if (newCount > parameters.maximumHypothesisKeyLength) return false;
            if (nonExemptScopes > 1) return false;
            if (nonExemptScopes == 1
                && nonExemptValidity != hypothesisValidity) return false;
        }
        if (secondaryCount > parameters.maximumSecondaryVariables) {
            if (secondaryCount
                > parameters.maximumSecondaryVariablesOrint) return false;
            const gl::NameId sharedScope = node.count == 0
                ? expression.validityId
                : columns.statements[
                    block.statementOffset + static_cast<uint32_t>(
                        pooledGrowthStatementIndex(
                            sortedKeys, filteredOffset, node,
                            positionValue, 0))].validityId;
            if (expression.validityId != sharedScope) return false;
            for (gl::NameId premise = 1;
                 premise < node.count; ++premise) {
                const gl::NameId existingIndex = pooledGrowthStatementIndex(
                    sortedKeys, filteredOffset, node, positionValue, premise);
                if (columns.statements[
                        block.statementOffset
                            + static_cast<uint32_t>(existingIndex)].validityId
                    != sharedScope) return false;
            }
            constexpr char orintLiteral[] = "_orint_";
            if (!projectedNameContains(
                    columns, block, sharedScope, orintLiteral, 7)) return false;
        }
        if (newCount > block.overallMaxKeyLength) return false;
        output.nodeIndex = nodeIndex;
        output.position = positionValue;
        output.validityId = newValidity;
        output.termMask = candidateMask;
        return true;
    }

    /// @brief Apply cheap gates to short-node candidates overlapping one window.
    ///
    /// @details
    /// One thread owns one compact short node and scans at most the measured short
    /// cutoff. Canonical node start ordinals place every attempt directly into its
    /// deterministic window slot; no atomic output placement is used.
    ///
    /// @param columns Resident semantic columns.
    /// @param batches Uploaded request batches.
    /// @param tasks Uploaded task descriptors.
    /// @param growthCalls Uploaded growth schedule.
    /// @param filterCounts Per-filter retained counts.
    /// @param filterOffsets Per-filter retained offsets.
    /// @param sortedKeys Global sorted retained keys.
    /// @param viewMasks Per-row mandatory-view bits.
    /// @param suffixMasks Per-row suffix unions.
    /// @param parameters Analyzer request-shape limits.
    /// @param frontier Immutable current frontier.
    /// @param prefixes Immutable pooled-prefix headers.
    /// @param prefixSecondary Fixed distinct secondary-variable pool.
    /// @param shortNodeIndices Stable compact short-node indices.
    /// @param shortNodeCount Number of short nodes.
    /// @param candidateStartsByNode Canonical start ordinal by node index.
    /// @param windowStart Canonical first attempt ordinal.
    /// @param windowCount Number of used attempt slots.
    /// @param attempts Window-local candidate records.
    /// @param flags One stable-selection verdict per window slot.
    /// @return Nothing.
    /// @invariant Every window slot written by this kernel belongs to exactly one
    ///            compact short node.
    __global__ void phase2GrowthShortCheapGateKernel(
        ProjectionSemanticColumns columns,
        const gl::gpu::DeviceRequestBatch* batches,
        const gl::gpu::DevicePhase2Task* tasks,
        const gl::gpu::DevicePhase2GrowthCall* growthCalls,
        const uint32_t* filterCounts,
        const uint32_t* filterOffsets,
        const uint64_t* sortedKeys,
        const uint8_t* viewMasks,
        const uint8_t* suffixMasks,
        gl::gpu::DevicePhase2GrowthParameters parameters,
        const gl::gpu::DeviceGrowthNode* frontier,
        const gl::gpu::DeviceGrowthPrefix* prefixes,
        const gl::NameId* prefixSecondary,
        const uint32_t* shortNodeIndices,
        uint32_t shortNodeCount,
        const uint64_t* candidateStartsByNode,
        uint64_t windowStart,
        uint32_t windowCount,
        gl::gpu::DeviceGrowthCandidateAttempt* attempts,
        uint8_t* flags) {
        const uint32_t shortIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if (shortIndex >= shortNodeCount) return;
        const uint32_t nodeIndex = shortNodeIndices[shortIndex];
        const gl::gpu::DeviceGrowthNode& node = frontier[nodeIndex];
        const gl::gpu::DevicePhase2GrowthCall& growthCall = growthCalls[
            node.callIndex];
        const uint32_t filteredCount = filterCounts[
            growthCall.filterCallIndex];
        const uint64_t nodeStart = candidateStartsByNode[nodeIndex];
        const uint64_t nodeEnd = nodeStart + filteredCount
            - static_cast<uint32_t>(node.startPosition);
        const uint64_t windowEnd = windowStart + windowCount;
        const uint64_t overlapStart = nodeStart > windowStart
            ? nodeStart : windowStart;
        const uint64_t overlapEnd = nodeEnd < windowEnd ? nodeEnd : windowEnd;
        for (uint64_t ordinal = overlapStart;
             ordinal < overlapEnd; ++ordinal) {
            const uint32_t localIndex = static_cast<uint32_t>(
                ordinal - windowStart);
            const uint32_t position = static_cast<uint32_t>(node.startPosition)
                + static_cast<uint32_t>(ordinal - nodeStart);
            gl::gpu::DeviceGrowthCandidateAttempt attempt{};
            const bool accepted = preparePooledGrowthCandidate(
                columns, batches, tasks, growthCalls, filterCounts,
                filterOffsets, sortedKeys, viewMasks, suffixMasks, parameters,
                frontier, prefixes, prefixSecondary, nodeIndex, position,
                attempt);
            flags[localIndex] = accepted ? 1u : 0u;
            if (accepted) attempts[localIndex] = attempt;
        }
    }

    /// @brief Apply cheap gates to long-node candidates overlapping one window.
    ///
    /// @details
    /// One 64-lane block owns one compact cooperative node. Lanes cover disjoint
    /// ascending positions and write directly to canonical window slots using the
    /// node's scanned start ordinal.
    ///
    /// @param columns Resident semantic columns.
    /// @param batches Uploaded request batches.
    /// @param tasks Uploaded task descriptors.
    /// @param growthCalls Uploaded growth schedule.
    /// @param filterCounts Per-filter retained counts.
    /// @param filterOffsets Per-filter retained offsets.
    /// @param sortedKeys Global sorted retained keys.
    /// @param viewMasks Per-row mandatory-view bits.
    /// @param suffixMasks Per-row suffix unions.
    /// @param parameters Analyzer request-shape limits.
    /// @param frontier Immutable current frontier.
    /// @param prefixes Immutable pooled-prefix headers.
    /// @param prefixSecondary Fixed distinct secondary-variable pool.
    /// @param cooperativeNodeIndices Stable compact long-node indices.
    /// @param cooperativeNodeCount Number of long nodes.
    /// @param candidateStartsByNode Canonical start ordinal by node index.
    /// @param windowStart Canonical first attempt ordinal.
    /// @param windowCount Number of used attempt slots.
    /// @param attempts Window-local candidate records.
    /// @param flags One stable-selection verdict per window slot.
    /// @return Nothing.
    /// @invariant Every window slot written by this kernel belongs to exactly one
    ///            compact cooperative node.
    __global__ void phase2GrowthCooperativeCheapGateKernel(
        ProjectionSemanticColumns columns,
        const gl::gpu::DeviceRequestBatch* batches,
        const gl::gpu::DevicePhase2Task* tasks,
        const gl::gpu::DevicePhase2GrowthCall* growthCalls,
        const uint32_t* filterCounts,
        const uint32_t* filterOffsets,
        const uint64_t* sortedKeys,
        const uint8_t* viewMasks,
        const uint8_t* suffixMasks,
        gl::gpu::DevicePhase2GrowthParameters parameters,
        const gl::gpu::DeviceGrowthNode* frontier,
        const gl::gpu::DeviceGrowthPrefix* prefixes,
        const gl::NameId* prefixSecondary,
        const uint32_t* cooperativeNodeIndices,
        uint32_t cooperativeNodeCount,
        const uint64_t* candidateStartsByNode,
        uint64_t windowStart,
        uint32_t windowCount,
        gl::gpu::DeviceGrowthCandidateAttempt* attempts,
        uint8_t* flags) {
        assert(blockIdx.x < cooperativeNodeCount);
        const uint32_t nodeIndex = cooperativeNodeIndices[blockIdx.x];
        const gl::gpu::DeviceGrowthNode& node = frontier[nodeIndex];
        const gl::gpu::DevicePhase2GrowthCall& growthCall = growthCalls[
            node.callIndex];
        const uint32_t filteredCount = filterCounts[
            growthCall.filterCallIndex];
        const uint64_t nodeStart = candidateStartsByNode[nodeIndex];
        const uint64_t nodeEnd = nodeStart + filteredCount
            - static_cast<uint32_t>(node.startPosition);
        const uint64_t windowEnd = windowStart + windowCount;
        const uint64_t overlapStart = nodeStart > windowStart
            ? nodeStart : windowStart;
        const uint64_t overlapEnd = nodeEnd < windowEnd ? nodeEnd : windowEnd;
        for (uint64_t ordinal = overlapStart + threadIdx.x;
             ordinal < overlapEnd; ordinal += blockDim.x) {
            const uint32_t localIndex = static_cast<uint32_t>(
                ordinal - windowStart);
            const uint32_t position = static_cast<uint32_t>(node.startPosition)
                + static_cast<uint32_t>(ordinal - nodeStart);
            gl::gpu::DeviceGrowthCandidateAttempt attempt{};
            const bool accepted = preparePooledGrowthCandidate(
                columns, batches, tasks, growthCalls, filterCounts,
                filterOffsets, sortedKeys, viewMasks, suffixMasks, parameters,
                frontier, prefixes, prefixSecondary, nodeIndex, position,
                attempt);
            flags[localIndex] = accepted ? 1u : 0u;
            if (accepted) attempts[localIndex] = attempt;
        }
    }

    /// @brief Compare one owner-signature slot against a pooled candidate path.
    ///
    /// @details
    /// Walks existing node premises plus the appended position in semantic order
    /// and locates one flattened `argFullId` slot without constructing the
    /// processor's maximum-size request-argument array.
    ///
    /// @param columns Resident semantic columns.
    /// @param block Owning logical-block projection.
    /// @param sortedKeys Global sorted retained keys.
    /// @param filteredOffset Owning class-span offset.
    /// @param node Immutable prefix node.
    /// @param candidatePosition Appended filtered position.
    /// @param slot Flattened argument slot requested by an owner signature.
    /// @param requiredId Required full argument identifier.
    /// @return True exactly when the slot exists and equals @p requiredId.
    /// @invariant Every reconstructed statement index belongs to @p block.
    __device__ bool pooledGrowthArgumentMatches(
        const ProjectionSemanticColumns& columns,
        const gl::gpu::DeviceLogicalBlockProjection& block,
        const uint64_t* sortedKeys,
        uint32_t filteredOffset,
        const gl::gpu::DeviceGrowthNode& node,
        uint32_t candidatePosition,
        int32_t slot,
        gl::NameId requiredId) {
        if (slot < 0) return false;
        int32_t argumentBase = 0;
        const gl::NameId count = node.count + 1;
        for (gl::NameId premise = 0; premise < count; ++premise) {
            const gl::NameId statementIndex = pooledGrowthStatementIndex(
                sortedKeys, filteredOffset, node, candidatePosition, premise);
            assert(statementIndex >= 0);
            assert(static_cast<uint32_t>(statementIndex)
                < block.statementCount);
            const gl::IntEncodedExpr& expression = columns.statements[
                block.statementOffset + static_cast<uint32_t>(statementIndex)];
            if (slot < argumentBase + expression.arity) {
                return expression.argFullId[slot - argumentBase] == requiredId;
            }
            argumentBase += expression.arity;
        }
        return false;
    }

    /// @brief Apply a projected owner-set u_ gate to one pooled candidate.
    ///
    /// @details
    /// Parses the canonical `OwnerSet` blob exactly like
    /// `projectedOwnerSetUSatisfied`, but resolves each required flattened
    /// argument slot through the pooled node plus candidate position. This keeps
    /// the production survivor kernel free of a maximum-size argument array.
    ///
    /// @param columns Resident semantic columns and owner blobs.
    /// @param block Owning logical-block projection.
    /// @param subkeyEntry Matched subkey-map entry.
    /// @param sortedKeys Global sorted retained keys.
    /// @param filteredOffset Owning class-span offset.
    /// @param node Immutable prefix node.
    /// @param candidatePosition Appended filtered position.
    /// @return True exactly when some projected owner signature is satisfiable.
    /// @invariant The subkey entry has exactly one structurally valid owner blob.
    __device__ bool projectedPooledOwnerSetUSatisfied(
        const ProjectionSemanticColumns& columns,
        const gl::gpu::DeviceLogicalBlockProjection& block,
        const gl::gpu::DeviceByteMapEntry& subkeyEntry,
        const uint64_t* sortedKeys,
        uint32_t filteredOffset,
        const gl::gpu::DeviceGrowthNode& node,
        uint32_t candidatePosition) {
        assert(node.count + 1 >= 3);
        assert(subkeyEntry.blobRecordCount == 1);
        const gl::gpu::DeviceBlobRecord& blob = columns.blobRecords[
            subkeyEntry.blobRecordOffset];
        assert(blob.byteLength >= 9);
        const char* cursor = columns.blobBytes + blob.byteOffset;
        const char* const end = cursor + blob.byteLength;
        const bool hasLooseOwner = static_cast<uint8_t>(*cursor++) != 0;
        const auto readInt32 = [&cursor, end]() {
            assert(cursor + sizeof(int32_t) <= end);
            const uint32_t value = static_cast<uint8_t>(cursor[0])
                | (static_cast<uint32_t>(
                    static_cast<uint8_t>(cursor[1])) << 8)
                | (static_cast<uint32_t>(
                    static_cast<uint8_t>(cursor[2])) << 16)
                | (static_cast<uint32_t>(
                    static_cast<uint8_t>(cursor[3])) << 24);
            cursor += sizeof(int32_t);
            return static_cast<int32_t>(value);
        };
        const int32_t signatureCount = readInt32();
        assert(signatureCount >= 0);
        bool signatureAccepted = false;
        for (int32_t signature = 0;
             signature < signatureCount; ++signature) {
            const int32_t pairCount = readInt32();
            assert(pairCount >= 0);
            bool accepted = !hasLooseOwner;
            for (int32_t pair = 0; pair < pairCount; ++pair) {
                const int32_t slot = readInt32();
                const gl::NameId requiredId = readInt32();
                if (accepted && !pooledGrowthArgumentMatches(
                        columns, block, sortedKeys, filteredOffset,
                        node, candidatePosition, slot, requiredId)) {
                    accepted = false;
                }
            }
            signatureAccepted = signatureAccepted || accepted;
        }
        const int32_t ownerCount = readInt32();
        assert(ownerCount >= 0);
        for (int32_t owner = 0; owner < ownerCount; ++owner) {
            (void)readInt32();
            (void)readInt32();
            const int32_t signatureIndex = readInt32();
            assert(signatureIndex >= -1);
            assert(signatureIndex < signatureCount);
        }
        assert(cursor == end);
        return hasLooseOwner || signatureCount == 0 || signatureAccepted;
    }

    /// @brief Append one pooled candidate event without path scratch arrays.
    ///
    /// @details
    /// Reconstructs statement and compact filtered-position fields directly from
    /// the immutable node and appended position, then performs the same bounded
    /// event, request, and per-task counter appends as the observation path.
    ///
    /// @param sortedKeys Global sorted retained keys.
    /// @param filteredOffset Owning class-span offset.
    /// @param node Immutable prefix node.
    /// @param candidatePosition Appended filtered position.
    /// @param subkeySatisfied Owner-accepted subkey verdict.
    /// @param wholeKeyPresent Whole-key presence verdict.
    /// @param termsSatisfied Mandatory-containment verdict.
    /// @param taskIndex Executor task owning this growth call.
    /// @param events Persistent event output array.
    /// @param requests Persistent raw-request output array.
    /// @param taskSubkeyCounts Exact processor split-work counts by task.
    /// @param counters Shared append counters.
    /// @param capacity Fixed event and request ceilings.
    /// @return Nothing.
    /// @invariant At least one of subkey satisfaction or recordability is true.
    __device__ void appendPooledGrowthEvent(
        const uint64_t* sortedKeys,
        uint32_t filteredOffset,
        const gl::gpu::DeviceGrowthNode& node,
        uint32_t candidatePosition,
        bool subkeySatisfied,
        bool wholeKeyPresent,
        bool termsSatisfied,
        uint32_t taskIndex,
        gl::gpu::DeviceAcceptedGrowthEvent* events,
        gl::gpu::DeviceRawGrowthRequest* requests,
        uint32_t* taskSubkeyCounts,
        DeviceGrowthCounters* counters,
        gl::gpu::Phase2GrowthCapacity capacity) {
        const bool recordable = wholeKeyPresent && termsSatisfied;
        assert(subkeySatisfied || recordable);
        assert(taskIndex < capacity.calls);
        if (subkeySatisfied)
            atomicAdd(&taskSubkeyCounts[taskIndex], 1u);
        const uint32_t eventIndex = atomicAdd(
            &counters->acceptedEventCount, 1u);
        assert(eventIndex < capacity.acceptedEvents);
        gl::gpu::DeviceAcceptedGrowthEvent event{};
        event.count = node.count + 1;
        for (gl::NameId premise = 0; premise < event.count; ++premise) {
            const uint32_t position = premise == node.count
                ? candidatePosition
                : static_cast<uint32_t>(node.filteredPositions[premise]);
            if (premise < node.count)
                assert(node.filteredPositions[premise] >= 0);
            assert(position < kProcessorMaximumFilteredRows);
            event.statementIndices[premise] = growthStatementIndex(
                sortedKeys, filteredOffset + position);
            event.filteredPositionCodes[premise] = static_cast<uint16_t>(
                position + 1u);
        }
        event.callIndex = node.callIndex;
        event.runOrdinal = node.runOrdinal;
        if (subkeySatisfied)
            event.flags |= gl::gpu::kDeviceGrowthEventSubkeySatisfied;
        if (wholeKeyPresent)
            event.flags |= gl::gpu::kDeviceGrowthEventWholeKeyPresent;
        if (termsSatisfied)
            event.flags |= gl::gpu::kDeviceGrowthEventTermsSatisfied;
        events[eventIndex] = event;
        if (recordable) {
            const uint32_t requestIndex = atomicAdd(
                &counters->rawRequestCount, 1u);
            assert(requestIndex < capacity.rawRequests);
            gl::gpu::DeviceRawGrowthRequest request{};
            request.event = event;
            requests[requestIndex] = request;
        }
    }

    /// @brief Probe compact cheap-gate survivors and emit exact events and children.
    ///
    /// @details
    /// One thread owns one stable survivor ordinal. It folds only the appended
    /// expression suffix onto the pooled normalized prefix, probes subkey/owner
    /// and whole-key maps, applies mandatory containment, and appends bounded
    /// semantic events and next-frontier children. Window order is canonical;
    /// atomic append order remains non-semantic and is reconstructed later from
    /// the retained call, run, path, and candidate tokens.
    ///
    /// @param columns Resident semantic columns.
    /// @param batches Uploaded request batches.
    /// @param tasks Uploaded task descriptors.
    /// @param growthCalls Uploaded growth schedule.
    /// @param filterOffsets Per-filter retained offsets.
    /// @param sortedKeys Global sorted retained keys.
    /// @param frontier Immutable current frontier.
    /// @param prefixes Immutable pooled-prefix headers.
    /// @param prefixPayload Fixed normalized-payload pool.
    /// @param prefixVariables Fixed distinct normalization-variable pool.
    /// @param attempts Window-local attempt records.
    /// @param survivorIndices Stable compact window ordinals.
    /// @param survivorCount Number of cheap-gate survivors.
    /// @param next Next frontier output.
    /// @param nextFrontierOrdinal Counter slot belonging to @p next.
    /// @param events Persistent event output.
    /// @param requests Persistent raw-request output.
    /// @param taskSubkeyCounts Exact processor split-work counts by task.
    /// @param counters Shared append counters.
    /// @param capacity Fixed growth ceilings.
    /// @return Nothing.
    /// @invariant Every input record passed all gates through maximum key length.
    __global__ void phase2GrowthProbeSurvivorsKernel(
        ProjectionSemanticColumns columns,
        const gl::gpu::DeviceRequestBatch* batches,
        const gl::gpu::DevicePhase2Task* tasks,
        const gl::gpu::DevicePhase2GrowthCall* growthCalls,
        const uint32_t* filterOffsets,
        const uint64_t* sortedKeys,
        const gl::gpu::DeviceGrowthNode* frontier,
        const gl::gpu::DeviceGrowthPrefix* prefixes,
        const gl::NameId* prefixPayload,
        const gl::NameId* prefixVariables,
        const gl::gpu::DeviceGrowthCandidateAttempt* attempts,
        const uint32_t* survivorIndices,
        uint32_t survivorCount,
        gl::gpu::DeviceGrowthNode* next,
        uint32_t nextFrontierOrdinal,
        gl::gpu::DeviceAcceptedGrowthEvent* events,
        gl::gpu::DeviceRawGrowthRequest* requests,
        uint32_t* taskSubkeyCounts,
        DeviceGrowthCounters* counters,
        gl::gpu::Phase2GrowthCapacity capacity) {
        const uint32_t survivorIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if (survivorIndex >= survivorCount) return;
        const gl::gpu::DeviceGrowthCandidateAttempt& candidate = attempts[
            survivorIndices[survivorIndex]];
        assert(candidate.nodeIndex < capacity.frontierRecords);
        const gl::gpu::DeviceGrowthNode& node = frontier[candidate.nodeIndex];
        const gl::gpu::DeviceGrowthPrefix& prefix = prefixes[
            candidate.nodeIndex];
        const gl::gpu::DevicePhase2GrowthCall& growthCall = growthCalls[
            node.callIndex];
        const gl::gpu::DevicePhase2Task& task = tasks[growthCall.taskIndex];
        const gl::gpu::DeviceRequestBatch& batch = batches[
            growthCall.batchIndex];
        const gl::gpu::DeviceLogicalBlockProjection& block =
            columns.logicalBlocks[task.logicalBlockIndex];
        const uint32_t filteredOffset = filterOffsets[
            growthCall.filterCallIndex];
        const gl::NameId statementIndex = growthStatementIndex(
            sortedKeys, filteredOffset + candidate.position);
        const gl::IntEncodedExpr& expression = columns.statements[
            block.statementOffset + static_cast<uint32_t>(statementIndex)];
        const gl::NameId newCount = node.count + 1;

        gl::NameId normalizedSuffix[
            2 + gl::ExecutionParameters::MAX_ARITY * 2]{};
        gl::NameId suffixLength = 0;
        normalizedSuffix[suffixLength++] = expression.nameId;
        normalizedSuffix[suffixLength++] = expression.negation;
        gl::NameId appendedVariableCount = 0;
        for (gl::NameId argument = 0;
             argument < expression.arity; ++argument) {
            const gl::NameId variableId = expression.argId[argument];
            gl::NameId normalizedId = 0;
            for (gl::NameId known = 0;
                 known < prefix.variableCount; ++known) {
                if (prefixVariables[
                        prefix.variableOffset + static_cast<uint32_t>(known)]
                    == variableId) {
                    normalizedId = known + 1;
                    break;
                }
            }
            for (gl::NameId prior = 0;
                 normalizedId == 0 && prior < argument; ++prior) {
                if (expression.argId[prior] == variableId) {
                    normalizedId = normalizedSuffix[2 + prior * 2];
                    assert(normalizedId > prefix.variableCount);
                    break;
                }
            }
            if (normalizedId == 0) {
                assert(appendedVariableCount
                    < gl::ExecutionParameters::MAX_ARITY);
                normalizedId = prefix.variableCount
                    + appendedVariableCount + 1;
                ++appendedVariableCount;
            }
            normalizedSuffix[suffixLength++] = normalizedId;
            normalizedSuffix[suffixLength++] = 0;
        }
        assert(prefix.payloadLength + suffixLength
            <= gl::ExecutionParameters::MAX_KEY_SLOTS);
        gl::gpu::DeviceByteMapKind wholeKind =
            gl::gpu::DeviceByteMapKind::overallWholeKeys;
        gl::gpu::DeviceByteMapKind subkeyKind =
            gl::gpu::DeviceByteMapKind::overallSubkeys;
        filterMapKinds(batch.memory, wholeKind, subkeyKind);
        const int32_t subkeyEntry = findProjectedSegmentedByteMapEntry(
            columns, block, subkeyKind, newCount,
            prefixPayload + prefix.payloadOffset, prefix.payloadLength,
            normalizedSuffix, suffixLength);
        bool subkeySatisfied = false;
        if (subkeyEntry >= 0) {
            subkeySatisfied = newCount < 3
                || projectedPooledOwnerSetUSatisfied(
                    columns, block,
                    columns.byteMapEntries[static_cast<uint32_t>(subkeyEntry)],
                    sortedKeys, filteredOffset, node, candidate.position);
        }
        const bool wholeKeyPresent = findProjectedSegmentedByteMapEntry(
            columns, block, wholeKind, newCount,
            prefixPayload + prefix.payloadOffset, prefix.payloadLength,
            normalizedSuffix, suffixLength) >= 0;
        const uint8_t termMasks[2]{
            static_cast<uint8_t>((prefix.packedSummary >> 8) & 0xffu),
            static_cast<uint8_t>((prefix.packedSummary >> 16) & 0xffu) };
        const bool termsSatisfied = batch.termCount == 0
            || growthTermsSatisfied(
                candidate.termMask, termMasks, batch.termCount);
        if (subkeySatisfied || (wholeKeyPresent && termsSatisfied)) {
            appendPooledGrowthEvent(
                sortedKeys, filteredOffset, node, candidate.position,
                subkeySatisfied, wholeKeyPresent, termsSatisfied,
                growthCall.taskIndex, events, requests,
                taskSubkeyCounts, counters, capacity);
        }
        gl::NameId targetLength = 0;
        selectedGrowthTargetLength(block, batch.memory, targetLength);
        if (subkeySatisfied && newCount < targetLength) {
            const uint32_t nextIndex = atomicAdd(
                &counters->frontierCounts[nextFrontierOrdinal], 1u);
            assert(nextIndex < capacity.frontierRecords);
            gl::gpu::DeviceGrowthNode child{};
            child.callIndex = node.callIndex;
            child.runOrdinal = node.runOrdinal;
            child.count = newCount;
            child.startPosition = static_cast<gl::NameId>(
                candidate.position + 1u);
            child.validityId = candidate.validityId;
            child.termMask = candidate.termMask;
            for (gl::NameId premise = 0;
                 premise < node.count; ++premise) {
                child.filteredPositions[premise] =
                    node.filteredPositions[premise];
            }
            child.filteredPositions[node.count] = static_cast<gl::NameId>(
                candidate.position);
            next[nextIndex] = child;
        }
    }

    /// @brief Expand compacted short-span growth nodes by one premise.
    ///
    /// @details
    /// One device thread owns one preclassified short node and scans later sorted
    /// positions in ascending order. It reads the immutable pooled prefix built by
    /// `phase2GrowthPreparePrefixesKernel`, folds only the appended expression,
    /// and applies mandatory reachability, validity ancestry, request-shape gates,
    /// normalized whole/subkey probes, persistent event emission, and child append
    /// without materializing rejected attempts.
    ///
    /// @param columns Resident semantic columns.
    /// @param batches Uploaded request batches.
    /// @param tasks Uploaded task descriptors.
    /// @param growthCalls Uploaded growth schedule.
    /// @param filterCounts Per-filter retained counts.
    /// @param filterOffsets Per-filter retained offsets.
    /// @param sortedKeys Global sorted retained keys.
    /// @param viewMasks Per-row mandatory-view bits.
    /// @param suffixMasks Per-row suffix unions.
    /// @param parameters Analyzer request-shape limits.
    /// @param current Current frontier input.
    /// @param shortNodeIndices Compact short-span node indices.
    /// @param shortNodeCount Number of valid compact indices.
    /// @param prefixes Immutable header per original frontier node.
    /// @param prefixPayload Fixed normalized-payload pool.
    /// @param prefixVariables Fixed distinct normalization-variable pool.
    /// @param prefixSecondary Fixed distinct secondary-variable pool.
    /// @param next Next frontier output.
    /// @param nextFrontierOrdinal Counter slot belonging to @p next.
    /// @param events Persistent event output.
    /// @param requests Persistent raw-request output.
    /// @param taskSubkeyCounts Exact processor split-work counts by task.
    /// @param counters Shared append counters.
    /// @param capacity Fixed growth ceilings.
    /// @return Nothing.
    /// @invariant Every current node passed all gates when it was appended.
    template <bool collectGateCensus>
    __global__ void phase2GrowthShortExpandKernel(
        ProjectionSemanticColumns columns,
        const gl::gpu::DeviceRequestBatch* batches,
        const gl::gpu::DevicePhase2Task* tasks,
        const gl::gpu::DevicePhase2GrowthCall* growthCalls,
        const uint32_t* filterCounts,
        const uint32_t* filterOffsets,
        const uint64_t* sortedKeys,
        const uint8_t* viewMasks,
        const uint8_t* suffixMasks,
        gl::gpu::DevicePhase2GrowthParameters parameters,
        const gl::gpu::DeviceGrowthNode* current,
        const uint32_t* shortNodeIndices,
        uint32_t shortNodeCount,
        const gl::gpu::DeviceGrowthPrefix* prefixes,
        const gl::NameId* prefixPayload,
        const gl::NameId* prefixVariables,
        const gl::NameId* prefixSecondary,
        gl::gpu::DeviceGrowthNode* next,
        uint32_t nextFrontierOrdinal,
        gl::gpu::DeviceAcceptedGrowthEvent* events,
        gl::gpu::DeviceRawGrowthRequest* requests,
        uint32_t* taskSubkeyCounts,
        DeviceGrowthCounters* counters,
        gl::gpu::Phase2GrowthCapacity capacity) {
        const uint32_t shortIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if (shortIndex >= shortNodeCount) return;
        const uint32_t nodeIndex = shortNodeIndices[shortIndex];
        assert(nodeIndex < capacity.frontierRecords);
        const gl::gpu::DeviceGrowthNode& node = current[nodeIndex];
        const gl::gpu::DeviceGrowthPrefix& prefix = prefixes[nodeIndex];
        const gl::gpu::DevicePhase2GrowthCall& growthCall =
            growthCalls[node.callIndex];
        const gl::gpu::DevicePhase2Task& task = tasks[growthCall.taskIndex];
        const gl::gpu::DeviceRequestBatch& batch =
            batches[growthCall.batchIndex];
        const gl::gpu::DeviceLogicalBlockProjection& block =
            columns.logicalBlocks[task.logicalBlockIndex];
        gl::NameId targetLength = 0;
        selectedGrowthTargetLength(block, batch.memory, targetLength);
        assert(node.count < targetLength);
        const uint32_t filteredCount =
            filterCounts[growthCall.filterCallIndex];
        const uint32_t filteredOffset =
            filterOffsets[growthCall.filterCallIndex];
        assert(node.startPosition >= 0);
        assert(static_cast<uint32_t>(node.startPosition) <= filteredCount);
        assert(node.count < gl::ExecutionParameters::MAX_EXPRESSIONS);
        const uint8_t termMasks[2]{
            static_cast<uint8_t>((prefix.packedSummary >> 8) & 0xffu),
            static_cast<uint8_t>((prefix.packedSummary >> 16) & 0xffu) };

        gl::NameId statementIndices[
            gl::ExecutionParameters::MAX_EXPRESSIONS]{};
        gl::NameId candidatePositions[
            gl::ExecutionParameters::MAX_EXPRESSIONS]{};
        for (gl::NameId premise = 0; premise < node.count; ++premise) {
            assert(node.filteredPositions[premise] >= 0);
            candidatePositions[premise] =
                node.filteredPositions[premise];
            statementIndices[premise] = growthStatementIndex(
                sortedKeys, filteredOffset
                    + static_cast<uint32_t>(
                        node.filteredPositions[premise]));
        }

        constexpr gl::NameId maximumCandidateSecondary =
            gl::ExecutionParameters::MAX_EXPRESSIONS
                * gl::ExecutionParameters::MAX_ARITY;
        const gl::NameId prefixPayloadLength =
            static_cast<gl::NameId>(prefix.payloadLength);
        const gl::NameId prefixVariableCount =
            static_cast<gl::NameId>(prefix.variableCount);
        const gl::NameId prefixSecondaryCount =
            static_cast<gl::NameId>(prefix.secondaryCount);
        const bool prefixHypothesisFound =
            (prefix.packedSummary & 1u) != 0;
        const gl::NameId prefixHypothesisValidity =
            prefix.hypothesisValidity;
        const int32_t prefixNonExemptScopes = static_cast<int32_t>(
            (prefix.packedSummary >> 1) & 0x3u);
        const gl::NameId prefixNonExemptValidity =
            prefix.nonExemptValidity;
        DeviceGrowthGateCounters gateCensus{};
        const gl::NameId newCount = node.count + 1;
        assert(static_cast<uint32_t>(newCount)
            < gl::gpu::kDeviceGrowthCensusDepthCount);
        for (gl::NameId position = node.startPosition;
             position < static_cast<gl::NameId>(filteredCount); ++position) {
            if constexpr (collectGateCensus) {
                ++gateCensus.candidateAttempts;
            }
            const uint32_t globalRow = filteredOffset
                + static_cast<uint32_t>(position);
            const gl::NameId statementIndex = growthStatementIndex(
                sortedKeys, globalRow);
            const gl::IntEncodedExpr& expression = columns.statements[
                block.statementOffset
                    + static_cast<uint32_t>(statementIndex)];
            const uint8_t candidateMask = batch.termCount == 0
                ? 0 : static_cast<uint8_t>(
                    node.termMask | viewMasks[globalRow]);
            if (batch.termCount > 0) {
                const uint8_t suffix = position + 1
                        < static_cast<gl::NameId>(filteredCount)
                    ? suffixMasks[globalRow + 1] : 0;
                if (!growthTermsReachable(
                        candidateMask, suffix, newCount, targetLength,
                        termMasks, batch.termCount)) continue;
            }
            if constexpr (collectGateCensus) {
                ++gateCensus.mandatoryReachable;
            }
            if (!projectedValiditiesComparable(
                    columns, block, node.validityId,
                    expression.validityId)) continue;
            if constexpr (collectGateCensus) {
                ++gateCensus.validityComparable;
            }
            const gl::NameId newValidity = projectedDeeperValidity(
                columns, block, node.validityId, expression.validityId);
            statementIndices[node.count] = statementIndex;
            candidatePositions[node.count] = position;

            bool hypothesisFound = prefixHypothesisFound;
            gl::NameId hypothesisValidity = prefixHypothesisValidity;
            int32_t nonExemptScopes = prefixNonExemptScopes;
            gl::NameId nonExemptValidity = prefixNonExemptValidity;
            if (expression.isHypo) {
                if (hypothesisFound
                    && expression.validityId != hypothesisValidity) {
                    continue;
                }
                hypothesisFound = true;
                hypothesisValidity = expression.validityId;
            }
            if (!(expression.validityId == block.mainValidityId
                  && expression.isAnchor)) {
                if (nonExemptScopes == 0) {
                    nonExemptScopes = 1;
                    nonExemptValidity = expression.validityId;
                }
                else if (nonExemptScopes == 1
                         && nonExemptValidity != expression.validityId) {
                    nonExemptScopes = 2;
                }
            }
            int32_t secondaryCount = prefixSecondaryCount;
            for (gl::NameId argument = 0;
                 argument < expression.arity; ++argument) {
                if (expression.argIteration[argument] <= -1) continue;
                if (findProjectedPodMapEntry(
                        columns, block,
                        gl::gpu::DevicePodMapKind::recursionProducts,
                        static_cast<int64_t>(
                            expression.argFullId[argument])) >= 0) {
                    continue;
                }
                const gl::NameId id = expression.argFullId[argument];
                bool found = false;
                for (int32_t known = 0;
                     known < prefixSecondaryCount; ++known) {
                    if (prefixSecondary[
                            prefix.secondaryOffset
                                + static_cast<uint32_t>(known)] == id) {
                        found = true;
                        break;
                    }
                }
                for (gl::NameId prior = 0;
                     !found && prior < argument; ++prior) {
                    if (expression.argIteration[prior] > -1
                        && expression.argFullId[prior] == id) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    ++secondaryCount;
                    assert(secondaryCount <= maximumCandidateSecondary);
                }
            }
            if (hypothesisFound) {
                if (newCount > parameters.maximumHypothesisKeyLength)
                    continue;
                if (nonExemptScopes > 1) continue;
                if (nonExemptScopes == 1
                    && nonExemptValidity != hypothesisValidity) {
                    continue;
                }
            }
            if constexpr (collectGateCensus) {
                ++gateCensus.hypothesisCompatible;
            }
            if (secondaryCount > parameters.maximumSecondaryVariables) {
                if (secondaryCount
                    > parameters.maximumSecondaryVariablesOrint) {
                    continue;
                }
                const gl::NameId sharedScope = node.count == 0
                    ? expression.validityId
                    : columns.statements[
                        block.statementOffset
                            + static_cast<uint32_t>(
                                statementIndices[0])].validityId;
                bool oneScope = expression.validityId == sharedScope;
                for (gl::NameId premise = 1;
                     oneScope && premise < node.count; ++premise) {
                    oneScope = columns.statements[
                        block.statementOffset
                            + static_cast<uint32_t>(
                                statementIndices[premise])].validityId
                        == sharedScope;
                }
                if (!oneScope) continue;
                constexpr char orintLiteral[] = "_orint_";
                if (!projectedNameContains(
                        columns, block, sharedScope,
                        orintLiteral, 7)) continue;
            }
            if constexpr (collectGateCensus) {
                ++gateCensus.secondaryCompatible;
            }
            if (newCount > block.overallMaxKeyLength) continue;
            if constexpr (collectGateCensus) {
                ++gateCensus.keyLengthAllowed;
            }

            gl::NameId normalizedSuffix[
                2 + gl::ExecutionParameters::MAX_ARITY * 2]{};
            gl::NameId suffixLength = 0;
            normalizedSuffix[suffixLength++] = expression.nameId;
            normalizedSuffix[suffixLength++] = expression.negation;
            gl::NameId appendedVariableCount = 0;
            for (gl::NameId argument = 0;
                 argument < expression.arity; ++argument) {
                const gl::NameId variableId = expression.argId[argument];
                gl::NameId normalizedId = 0;
                for (gl::NameId known = 0;
                     known < prefixVariableCount; ++known) {
                    if (prefixVariables[
                            prefix.variableOffset
                                + static_cast<uint32_t>(known)] == variableId) {
                        normalizedId = known + 1;
                        break;
                    }
                }
                for (gl::NameId prior = 0;
                     normalizedId == 0 && prior < argument; ++prior) {
                    if (expression.argId[prior] == variableId) {
                        normalizedId = normalizedSuffix[2 + prior * 2];
                        assert(normalizedId > prefixVariableCount);
                        break;
                    }
                }
                if (normalizedId == 0) {
                    assert(appendedVariableCount
                        < gl::ExecutionParameters::MAX_ARITY);
                    normalizedId = prefixVariableCount
                        + appendedVariableCount + 1;
                    ++appendedVariableCount;
                }
                normalizedSuffix[suffixLength++] = normalizedId;
                normalizedSuffix[suffixLength++] = 0;
            }
            assert(prefixPayloadLength + suffixLength
                <= gl::ExecutionParameters::MAX_KEY_SLOTS);
            gl::gpu::DeviceByteMapKind wholeKind =
                gl::gpu::DeviceByteMapKind::overallWholeKeys;
            gl::gpu::DeviceByteMapKind subkeyKind =
                gl::gpu::DeviceByteMapKind::overallSubkeys;
            filterMapKinds(batch.memory, wholeKind, subkeyKind);
            const int32_t subkeyEntry =
                findProjectedSegmentedByteMapEntry(
                    columns, block, subkeyKind, newCount,
                    prefixPayload + prefix.payloadOffset,
                    prefixPayloadLength,
                    normalizedSuffix, suffixLength);
            bool subkeySatisfied = false;
            if (subkeyEntry >= 0) {
                if constexpr (collectGateCensus) {
                    ++gateCensus.subkeyPresent;
                }
                subkeySatisfied = newCount < 3
                    || projectedOwnerSetUSatisfied(
                        columns, block,
                        columns.byteMapEntries[
                            static_cast<uint32_t>(subkeyEntry)],
                        statementIndices, newCount);
            }
            if constexpr (collectGateCensus) {
                if (subkeySatisfied) ++gateCensus.ownerSatisfied;
            }
            const bool wholeKeyPresent =
                findProjectedSegmentedByteMapEntry(
                    columns, block, wholeKind, newCount,
                    prefixPayload + prefix.payloadOffset,
                    prefixPayloadLength,
                    normalizedSuffix, suffixLength) >= 0;
            const bool termsSatisfied = batch.termCount == 0
                || growthTermsSatisfied(
                    candidateMask, termMasks, batch.termCount);
            if constexpr (collectGateCensus) {
                if (wholeKeyPresent) ++gateCensus.wholeKeyPresent;
            }
            if constexpr (collectGateCensus) {
                if (termsSatisfied) ++gateCensus.termsSatisfied;
            }
            if (subkeySatisfied || (wholeKeyPresent && termsSatisfied)) {
                if constexpr (collectGateCensus) {
                    ++gateCensus.acceptedEvents;
                }
                appendProjectedGrowthEvent(
                    statementIndices, candidatePositions, newCount,
                    node.callIndex, node.runOrdinal,
                    subkeySatisfied, wholeKeyPresent, termsSatisfied,
                    growthCall.taskIndex, events, requests,
                    taskSubkeyCounts, counters, capacity);
            }
            if (subkeySatisfied && newCount < targetLength) {
                if constexpr (collectGateCensus) {
                    ++gateCensus.children;
                }
                const uint32_t nextIndex = atomicAdd(
                    &counters->frontierCounts[nextFrontierOrdinal], 1u);
                assert(nextIndex < capacity.frontierRecords);
                gl::gpu::DeviceGrowthNode child{};
                child.callIndex = node.callIndex;
                child.runOrdinal = node.runOrdinal;
                child.count = newCount;
                child.startPosition = position + 1;
                child.validityId = newValidity;
                child.termMask = candidateMask;
                for (gl::NameId premise = 0;
                     premise < node.count; ++premise) {
                    child.filteredPositions[premise] =
                        node.filteredPositions[premise];
                }
                child.filteredPositions[node.count] = position;
                next[nextIndex] = child;
            }
        }
        if constexpr (collectGateCensus) {
            DeviceGrowthGateCounters& destination =
                counters->gateDepthCounts[static_cast<uint32_t>(newCount)];
            if (gateCensus.candidateAttempts != 0) atomicAdd(
                &destination.candidateAttempts, gateCensus.candidateAttempts);
            if (gateCensus.mandatoryReachable != 0) atomicAdd(
                &destination.mandatoryReachable,
                gateCensus.mandatoryReachable);
            if (gateCensus.validityComparable != 0) atomicAdd(
                &destination.validityComparable,
                gateCensus.validityComparable);
            if (gateCensus.hypothesisCompatible != 0) atomicAdd(
                &destination.hypothesisCompatible,
                gateCensus.hypothesisCompatible);
            if (gateCensus.secondaryCompatible != 0) atomicAdd(
                &destination.secondaryCompatible,
                gateCensus.secondaryCompatible);
            if (gateCensus.keyLengthAllowed != 0) atomicAdd(
                &destination.keyLengthAllowed,
                gateCensus.keyLengthAllowed);
            if (gateCensus.subkeyPresent != 0) atomicAdd(
                &destination.subkeyPresent, gateCensus.subkeyPresent);
            if (gateCensus.ownerSatisfied != 0) atomicAdd(
                &destination.ownerSatisfied, gateCensus.ownerSatisfied);
            if (gateCensus.wholeKeyPresent != 0) atomicAdd(
                &destination.wholeKeyPresent, gateCensus.wholeKeyPresent);
            if (gateCensus.termsSatisfied != 0) atomicAdd(
                &destination.termsSatisfied, gateCensus.termsSatisfied);
            if (gateCensus.acceptedEvents != 0) atomicAdd(
                &destination.acceptedEvents, gateCensus.acceptedEvents);
            if (gateCensus.children != 0) atomicAdd(
                &destination.children, gateCensus.children);
        }
    }

    /// @brief Expand compacted long-span frontier nodes cooperatively.
    ///
    /// @details
    /// Assigns one block to each preclassified long node. Every lane reads the
    /// immutable pooled prefix built before classification completes, scans a
    /// disjoint ascending candidate subsequence, builds only the appended
    /// normalized-expression suffix, and probes the resident maps through the
    /// segmented prefix-plus-suffix reader. Accepted events and child nodes enter
    /// the same bounded ledgers as the short-node kernel, and their atomic append
    /// order remains non-semantic until canonical reconstruction.
    ///
    /// @param columns Resident semantic columns.
    /// @param batches Uploaded request batches.
    /// @param tasks Uploaded task descriptors.
    /// @param growthCalls Uploaded growth schedule.
    /// @param filterCounts Per-filter retained counts.
    /// @param filterOffsets Per-filter retained offsets.
    /// @param sortedKeys Global sorted retained keys.
    /// @param viewMasks Per-row mandatory-view bits.
    /// @param suffixMasks Per-row suffix unions.
    /// @param parameters Analyzer limits and measured cooperative cutoff.
    /// @param current Current immutable frontier.
    /// @param cooperativeNodeIndices Compact current-frontier node indices.
    /// @param cooperativeNodeCount Number of valid compact indices.
    /// @param prefixes Immutable header per original frontier node.
    /// @param prefixPayload Fixed normalized-payload pool.
    /// @param prefixVariables Fixed distinct normalization-variable pool.
    /// @param prefixSecondary Fixed distinct secondary-variable pool.
    /// @param next Next frontier output.
    /// @param nextFrontierOrdinal Counter slot belonging to @p next.
    /// @param events Persistent event output.
    /// @param requests Persistent raw-request output.
    /// @param taskSubkeyCounts Exact processor split-work counts by task.
    /// @param counters Shared append counters.
    /// @param capacity Fixed growth ceilings.
    /// @return Nothing.
    /// @invariant Every compacted node has at least the configured remaining span,
    ///            and one block owns exactly one immutable node.
    /// @see phase2GrowthPreparePrefixesKernel
    template <bool collectGateCensus>
    __global__ void phase2GrowthCooperativeExpandKernel(
        ProjectionSemanticColumns columns,
        const gl::gpu::DeviceRequestBatch* batches,
        const gl::gpu::DevicePhase2Task* tasks,
        const gl::gpu::DevicePhase2GrowthCall* growthCalls,
        const uint32_t* filterCounts,
        const uint32_t* filterOffsets,
        const uint64_t* sortedKeys,
        const uint8_t* viewMasks,
        const uint8_t* suffixMasks,
        gl::gpu::DevicePhase2GrowthParameters parameters,
        const gl::gpu::DeviceGrowthNode* current,
        const uint32_t* cooperativeNodeIndices,
        uint32_t cooperativeNodeCount,
        const gl::gpu::DeviceGrowthPrefix* prefixes,
        const gl::NameId* prefixPayload,
        const gl::NameId* prefixVariables,
        const gl::NameId* prefixSecondary,
        gl::gpu::DeviceGrowthNode* next,
        uint32_t nextFrontierOrdinal,
        gl::gpu::DeviceAcceptedGrowthEvent* events,
        gl::gpu::DeviceRawGrowthRequest* requests,
        uint32_t* taskSubkeyCounts,
        DeviceGrowthCounters* counters,
        gl::gpu::Phase2GrowthCapacity capacity) {
        assert(blockIdx.x < cooperativeNodeCount);
        const uint32_t nodeIndex = cooperativeNodeIndices[blockIdx.x];
        assert(nodeIndex < capacity.frontierRecords);
        const gl::gpu::DeviceGrowthNode& node = current[nodeIndex];
        const gl::gpu::DeviceGrowthPrefix& prefix = prefixes[nodeIndex];
        const gl::gpu::DevicePhase2GrowthCall& growthCall =
            growthCalls[node.callIndex];
        const gl::gpu::DevicePhase2Task& task = tasks[growthCall.taskIndex];
        const gl::gpu::DeviceRequestBatch& batch =
            batches[growthCall.batchIndex];
        const gl::gpu::DeviceLogicalBlockProjection& block =
            columns.logicalBlocks[task.logicalBlockIndex];
        gl::NameId targetLength = 0;
        selectedGrowthTargetLength(block, batch.memory, targetLength);
        assert(node.count < targetLength);
        assert(node.count < gl::ExecutionParameters::MAX_EXPRESSIONS);
        const uint32_t filteredCount =
            filterCounts[growthCall.filterCallIndex];
        const uint32_t filteredOffset =
            filterOffsets[growthCall.filterCallIndex];
        assert(parameters.cooperativeSpanThreshold > 0);
        assert(node.startPosition >= 0);
        assert(static_cast<uint32_t>(node.startPosition) <= filteredCount);
        assert(filteredCount - static_cast<uint32_t>(node.startPosition)
            >= parameters.cooperativeSpanThreshold);

        constexpr gl::NameId maximumCandidateSecondary =
            gl::ExecutionParameters::MAX_EXPRESSIONS
                * gl::ExecutionParameters::MAX_ARITY;
        __shared__ DeviceGrowthGateCounters blockGateCensus;
        if constexpr (collectGateCensus) {
            if (threadIdx.x == 0)
                blockGateCensus = DeviceGrowthGateCounters{};
            __syncthreads();
        }

        gl::NameId statementIndices[
            gl::ExecutionParameters::MAX_EXPRESSIONS]{};
        gl::NameId candidatePositions[
            gl::ExecutionParameters::MAX_EXPRESSIONS]{};
        for (gl::NameId premise = 0; premise < node.count; ++premise) {
            assert(node.filteredPositions[premise] >= 0);
            candidatePositions[premise] = node.filteredPositions[premise];
            statementIndices[premise] = growthStatementIndex(
                sortedKeys, filteredOffset + static_cast<uint32_t>(
                    node.filteredPositions[premise]));
        }
        const uint8_t termMasks[2]{
            static_cast<uint8_t>((prefix.packedSummary >> 8) & 0xffu),
            static_cast<uint8_t>((prefix.packedSummary >> 16) & 0xffu) };
        const gl::NameId newCount = node.count + 1;
        assert(static_cast<uint32_t>(newCount)
            < gl::gpu::kDeviceGrowthCensusDepthCount);
        DeviceGrowthGateCounters gateCensus{};
        for (uint32_t positionValue = static_cast<uint32_t>(node.startPosition)
                + threadIdx.x;
             positionValue < filteredCount;
             positionValue += blockDim.x) {
            if constexpr (collectGateCensus) {
                ++gateCensus.candidateAttempts;
            }
            const gl::NameId position = static_cast<gl::NameId>(positionValue);
            const uint32_t globalRow = filteredOffset + positionValue;
            const gl::NameId statementIndex = growthStatementIndex(
                sortedKeys, globalRow);
            const gl::IntEncodedExpr& expression = columns.statements[
                block.statementOffset + static_cast<uint32_t>(statementIndex)];
            const uint8_t candidateMask = batch.termCount == 0
                ? 0 : static_cast<uint8_t>(
                    node.termMask | viewMasks[globalRow]);
            if (batch.termCount > 0) {
                const uint8_t suffix = positionValue + 1 < filteredCount
                    ? suffixMasks[globalRow + 1] : 0;
                if (!growthTermsReachable(
                        candidateMask, suffix, newCount, targetLength,
                        termMasks, batch.termCount)) continue;
            }
            if constexpr (collectGateCensus) {
                ++gateCensus.mandatoryReachable;
            }
            if (!projectedValiditiesComparable(
                    columns, block, node.validityId,
                    expression.validityId)) continue;
            if constexpr (collectGateCensus) {
                ++gateCensus.validityComparable;
            }
            const gl::NameId newValidity = projectedDeeperValidity(
                columns, block, node.validityId, expression.validityId);
            statementIndices[node.count] = statementIndex;
            candidatePositions[node.count] = position;

            bool hypothesisFound = (prefix.packedSummary & 1u) != 0;
            gl::NameId hypothesisValidity = prefix.hypothesisValidity;
            int32_t nonExemptScopes = static_cast<int32_t>(
                (prefix.packedSummary >> 1) & 0x3u);
            gl::NameId nonExemptValidity = prefix.nonExemptValidity;
            if (expression.isHypo) {
                if (hypothesisFound
                    && expression.validityId != hypothesisValidity) {
                    continue;
                }
                hypothesisFound = true;
                hypothesisValidity = expression.validityId;
            }
            if (!(expression.validityId == block.mainValidityId
                  && expression.isAnchor)) {
                if (nonExemptScopes == 0) {
                    nonExemptScopes = 1;
                    nonExemptValidity = expression.validityId;
                }
                else if (nonExemptScopes == 1
                         && nonExemptValidity != expression.validityId) {
                    nonExemptScopes = 2;
                }
            }
            int32_t secondaryCount = prefix.secondaryCount;
            for (gl::NameId argument = 0;
                 argument < expression.arity; ++argument) {
                if (expression.argIteration[argument] <= -1) continue;
                if (findProjectedPodMapEntry(
                        columns, block,
                        gl::gpu::DevicePodMapKind::recursionProducts,
                        static_cast<int64_t>(
                            expression.argFullId[argument])) >= 0) {
                    continue;
                }
                const gl::NameId id = expression.argFullId[argument];
                bool found = false;
                for (int32_t known = 0;
                     known < prefix.secondaryCount; ++known) {
                    if (prefixSecondary[
                            prefix.secondaryOffset
                                + static_cast<uint32_t>(known)] == id) {
                        found = true;
                        break;
                    }
                }
                for (gl::NameId prior = 0;
                     !found && prior < argument; ++prior) {
                    if (expression.argIteration[prior] > -1
                        && expression.argFullId[prior] == id) {
                        found = true;
                        break;
                    }
                }
                if (!found) {
                    ++secondaryCount;
                    assert(secondaryCount <= maximumCandidateSecondary);
                }
            }
            if (hypothesisFound) {
                if (newCount > parameters.maximumHypothesisKeyLength)
                    continue;
                if (nonExemptScopes > 1) continue;
                if (nonExemptScopes == 1
                    && nonExemptValidity != hypothesisValidity) {
                    continue;
                }
            }
            if constexpr (collectGateCensus) {
                ++gateCensus.hypothesisCompatible;
            }
            if (secondaryCount > parameters.maximumSecondaryVariables) {
                if (secondaryCount
                    > parameters.maximumSecondaryVariablesOrint) {
                    continue;
                }
                const gl::NameId sharedScope = node.count == 0
                    ? expression.validityId
                    : columns.statements[
                        block.statementOffset + static_cast<uint32_t>(
                            statementIndices[0])].validityId;
                bool oneScope = expression.validityId == sharedScope;
                for (gl::NameId premise = 1;
                     oneScope && premise < node.count; ++premise) {
                    oneScope = columns.statements[
                        block.statementOffset + static_cast<uint32_t>(
                            statementIndices[premise])].validityId
                        == sharedScope;
                }
                if (!oneScope) continue;
                constexpr char orintLiteral[] = "_orint_";
                if (!projectedNameContains(
                        columns, block, sharedScope,
                        orintLiteral, 7)) continue;
            }
            if constexpr (collectGateCensus) {
                ++gateCensus.secondaryCompatible;
            }
            if (newCount > block.overallMaxKeyLength) continue;
            if constexpr (collectGateCensus) {
                ++gateCensus.keyLengthAllowed;
            }

            gl::NameId normalizedSuffix[
                2 + gl::ExecutionParameters::MAX_ARITY * 2]{};
            gl::NameId suffixLength = 0;
            normalizedSuffix[suffixLength++] = expression.nameId;
            normalizedSuffix[suffixLength++] = expression.negation;
            gl::NameId appendedVariableCount = 0;
            for (gl::NameId argument = 0;
                 argument < expression.arity; ++argument) {
                const gl::NameId variableId = expression.argId[argument];
                gl::NameId normalizedId = 0;
                for (gl::NameId known = 0;
                     known < prefix.variableCount; ++known) {
                    if (prefixVariables[
                            prefix.variableOffset
                                + static_cast<uint32_t>(known)] == variableId) {
                        normalizedId = known + 1;
                        break;
                    }
                }
                for (gl::NameId prior = 0;
                     normalizedId == 0 && prior < argument; ++prior) {
                    if (expression.argId[prior] == variableId) {
                        normalizedId = normalizedSuffix[2 + prior * 2];
                        assert(normalizedId > prefix.variableCount);
                        break;
                    }
                }
                if (normalizedId == 0) {
                    assert(appendedVariableCount
                        < gl::ExecutionParameters::MAX_ARITY);
                    normalizedId = prefix.variableCount
                        + appendedVariableCount + 1;
                    ++appendedVariableCount;
                }
                normalizedSuffix[suffixLength++] = normalizedId;
                normalizedSuffix[suffixLength++] = 0;
            }
            assert(prefix.payloadLength + suffixLength
                <= gl::ExecutionParameters::MAX_KEY_SLOTS);
            gl::gpu::DeviceByteMapKind wholeKind =
                gl::gpu::DeviceByteMapKind::overallWholeKeys;
            gl::gpu::DeviceByteMapKind subkeyKind =
                gl::gpu::DeviceByteMapKind::overallSubkeys;
            filterMapKinds(batch.memory, wholeKind, subkeyKind);
            const int32_t subkeyEntry =
                findProjectedSegmentedByteMapEntry(
                    columns, block, subkeyKind, newCount,
                    prefixPayload + prefix.payloadOffset,
                    prefix.payloadLength,
                    normalizedSuffix, suffixLength);
            bool subkeySatisfied = false;
            if (subkeyEntry >= 0) {
                if constexpr (collectGateCensus) {
                    ++gateCensus.subkeyPresent;
                }
                subkeySatisfied = newCount < 3
                    || projectedOwnerSetUSatisfied(
                        columns, block,
                        columns.byteMapEntries[
                            static_cast<uint32_t>(subkeyEntry)],
                        statementIndices, newCount);
            }
            if constexpr (collectGateCensus) {
                if (subkeySatisfied) ++gateCensus.ownerSatisfied;
            }
            const bool wholeKeyPresent =
                findProjectedSegmentedByteMapEntry(
                    columns, block, wholeKind, newCount,
                    prefixPayload + prefix.payloadOffset,
                    prefix.payloadLength,
                    normalizedSuffix, suffixLength) >= 0;
            const bool termsSatisfied = batch.termCount == 0
                || growthTermsSatisfied(
                    candidateMask, termMasks, batch.termCount);
            if constexpr (collectGateCensus) {
                if (wholeKeyPresent) ++gateCensus.wholeKeyPresent;
            }
            if constexpr (collectGateCensus) {
                if (termsSatisfied) ++gateCensus.termsSatisfied;
            }
            if (subkeySatisfied || (wholeKeyPresent && termsSatisfied)) {
                if constexpr (collectGateCensus) {
                    ++gateCensus.acceptedEvents;
                }
                appendProjectedGrowthEvent(
                    statementIndices, candidatePositions, newCount,
                    node.callIndex, node.runOrdinal,
                    subkeySatisfied, wholeKeyPresent, termsSatisfied,
                    growthCall.taskIndex, events, requests,
                    taskSubkeyCounts, counters, capacity);
            }
            if (subkeySatisfied && newCount < targetLength) {
                if constexpr (collectGateCensus) {
                    ++gateCensus.children;
                }
                const uint32_t nextIndex = atomicAdd(
                    &counters->frontierCounts[nextFrontierOrdinal], 1u);
                assert(nextIndex < capacity.frontierRecords);
                gl::gpu::DeviceGrowthNode child{};
                child.callIndex = node.callIndex;
                child.runOrdinal = node.runOrdinal;
                child.count = newCount;
                child.startPosition = position + 1;
                child.validityId = newValidity;
                child.termMask = candidateMask;
                for (gl::NameId premise = 0;
                     premise < node.count; ++premise) {
                    child.filteredPositions[premise] =
                        node.filteredPositions[premise];
                }
                child.filteredPositions[node.count] = position;
                next[nextIndex] = child;
            }
        }
        if constexpr (collectGateCensus) {
            if (gateCensus.candidateAttempts != 0) atomicAdd(
                &blockGateCensus.candidateAttempts,
                gateCensus.candidateAttempts);
            if (gateCensus.mandatoryReachable != 0) atomicAdd(
                &blockGateCensus.mandatoryReachable,
                gateCensus.mandatoryReachable);
            if (gateCensus.validityComparable != 0) atomicAdd(
                &blockGateCensus.validityComparable,
                gateCensus.validityComparable);
            if (gateCensus.hypothesisCompatible != 0) atomicAdd(
                &blockGateCensus.hypothesisCompatible,
                gateCensus.hypothesisCompatible);
            if (gateCensus.secondaryCompatible != 0) atomicAdd(
                &blockGateCensus.secondaryCompatible,
                gateCensus.secondaryCompatible);
            if (gateCensus.keyLengthAllowed != 0) atomicAdd(
                &blockGateCensus.keyLengthAllowed,
                gateCensus.keyLengthAllowed);
            if (gateCensus.subkeyPresent != 0) atomicAdd(
                &blockGateCensus.subkeyPresent, gateCensus.subkeyPresent);
            if (gateCensus.ownerSatisfied != 0) atomicAdd(
                &blockGateCensus.ownerSatisfied, gateCensus.ownerSatisfied);
            if (gateCensus.wholeKeyPresent != 0) atomicAdd(
                &blockGateCensus.wholeKeyPresent,
                gateCensus.wholeKeyPresent);
            if (gateCensus.termsSatisfied != 0) atomicAdd(
                &blockGateCensus.termsSatisfied, gateCensus.termsSatisfied);
            if (gateCensus.acceptedEvents != 0) atomicAdd(
                &blockGateCensus.acceptedEvents, gateCensus.acceptedEvents);
            if (gateCensus.children != 0) atomicAdd(
                &blockGateCensus.children, gateCensus.children);
            __syncthreads();
            if (threadIdx.x == 0) {
                DeviceGrowthGateCounters& destination =
                    counters->gateDepthCounts[
                        static_cast<uint32_t>(newCount)];
                if (blockGateCensus.candidateAttempts != 0) atomicAdd(
                    &destination.candidateAttempts,
                    blockGateCensus.candidateAttempts);
                if (blockGateCensus.mandatoryReachable != 0) atomicAdd(
                    &destination.mandatoryReachable,
                    blockGateCensus.mandatoryReachable);
                if (blockGateCensus.validityComparable != 0) atomicAdd(
                    &destination.validityComparable,
                    blockGateCensus.validityComparable);
                if (blockGateCensus.hypothesisCompatible != 0) atomicAdd(
                    &destination.hypothesisCompatible,
                    blockGateCensus.hypothesisCompatible);
                if (blockGateCensus.secondaryCompatible != 0) atomicAdd(
                    &destination.secondaryCompatible,
                    blockGateCensus.secondaryCompatible);
                if (blockGateCensus.keyLengthAllowed != 0) atomicAdd(
                    &destination.keyLengthAllowed,
                    blockGateCensus.keyLengthAllowed);
                if (blockGateCensus.subkeyPresent != 0) atomicAdd(
                    &destination.subkeyPresent,
                    blockGateCensus.subkeyPresent);
                if (blockGateCensus.ownerSatisfied != 0) atomicAdd(
                    &destination.ownerSatisfied,
                    blockGateCensus.ownerSatisfied);
                if (blockGateCensus.wholeKeyPresent != 0) atomicAdd(
                    &destination.wholeKeyPresent,
                    blockGateCensus.wholeKeyPresent);
                if (blockGateCensus.termsSatisfied != 0) atomicAdd(
                    &destination.termsSatisfied,
                    blockGateCensus.termsSatisfied);
                if (blockGateCensus.acceptedEvents != 0) atomicAdd(
                    &destination.acceptedEvents,
                    blockGateCensus.acceptedEvents);
                if (blockGateCensus.children != 0) atomicAdd(
                    &destination.children, blockGateCensus.children);
            }
        }
    }

    /// @brief Segmented inclusive-scan value for one ordered growth event.
    ///
    /// @details
    /// Events are already grouped by task. `growthPosition` is one for an
    /// owner-accepted subkey event and zero for a whole-only event before scan;
    /// after scan it is the task-local cumulative submatch tally.
    struct DeviceGrowthScanValue {
        uint32_t taskIndex{ 0 };
        uint32_t growthPosition{ 0 };
    };

    static_assert(sizeof(DeviceGrowthScanValue) == 8);

    /// @brief Associative segmented addition over task-grouped growth events.
    ///
    /// @param left Inclusive aggregate of the earlier contiguous segment.
    /// @param right Next aggregate or event value.
    /// @return Right task with summed positions when tasks match, otherwise right.
    /// @invariant Input order groups every task contiguously.
    struct DeviceGrowthSegmentedAdd {
        __host__ __device__ DeviceGrowthScanValue operator()(
            const DeviceGrowthScanValue& left,
            const DeviceGrowthScanValue& right) const {
            DeviceGrowthScanValue result = right;
            if (left.taskIndex == right.taskIndex)
                result.growthPosition += left.growthPosition;
            return result;
        }
    };

    /// @brief Compute one lexicographic component of processor event order.
    ///
    /// @details
    /// Components zero through seven encode one filtered path level. A direct
    /// child event uses its ascending position; a continuing subtree uses a
    /// disjoint larger range with inverted position, reproducing the processor's
    /// "emit children ascending, then pop growable children descending" stack
    /// walk. Components eight, nine, and ten are run, call, and task identity.
    ///
    /// @param event Persistent unordered growth event.
    /// @param calls Uploaded growth-call schedule.
    /// @param component Component ordinal in `[0,10]`.
    /// @return Unsigned radix key for this component.
    /// @invariant Stable least-significant-first passes over 7..0,8,9,10 produce
    ///            exact task/call/run/processor-walk order.
    __device__ uint32_t growthEventOrderComponent(
        const gl::gpu::DeviceAcceptedGrowthEvent& event,
        const gl::gpu::DevicePhase2GrowthCall* calls,
        uint32_t component) {
        if (component < gl::ExecutionParameters::MAX_EXPRESSIONS) {
            if (component >= static_cast<uint32_t>(event.count)) return 0;
            const uint16_t code = event.filteredPositionCodes[component];
            const uint32_t position = code == 0
                ? 0u : static_cast<uint32_t>(code - 1);
            assert(position < kProcessorMaximumFilteredRows);
            if (component + 1 == static_cast<uint32_t>(event.count))
                return position;
            return 65536u
                + (kProcessorMaximumFilteredRows - 1u - position);
        }
        if (component == 8) return event.runOrdinal;
        if (component == 9) return event.callIndex;
        assert(component == 10);
        return calls[event.callIndex].taskIndex;
    }

    /// @brief Initialize one identity permutation over unordered growth events.
    ///
    /// @param count Number of event indices.
    /// @param indices Output identity permutation.
    /// @return Nothing.
    /// @invariant Launch covers every index exactly once.
    __global__ void phase2OrderingInitializeIndicesKernel(
        uint32_t count,
        uint32_t* indices) {
        const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < count) indices[index] = index;
    }

    /// @brief Materialize one stable-radix component for the current permutation.
    ///
    /// @param events Persistent unordered event ledger.
    /// @param calls Uploaded growth-call schedule.
    /// @param indices Current event-index permutation.
    /// @param count Number of used events.
    /// @param component Processor-order component ordinal.
    /// @param keys Output unsigned radix keys.
    /// @return Nothing.
    /// @invariant Every index selects one used event.
    __global__ void phase2OrderingComponentKeysKernel(
        const gl::gpu::DeviceAcceptedGrowthEvent* events,
        const gl::gpu::DevicePhase2GrowthCall* calls,
        const uint32_t* indices,
        uint32_t count,
        uint32_t component,
        uint32_t* keys) {
        const uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
        if (position >= count) return;
        const uint32_t eventIndex = indices[position];
        assert(eventIndex < count);
        keys[position] = growthEventOrderComponent(
            events[eventIndex], calls, component);
    }

    /// @brief Build task-local growth-position scan inputs in exact event order.
    ///
    /// @param events Persistent unordered event ledger.
    /// @param calls Uploaded growth-call schedule.
    /// @param orderedIndices Exact ordered event permutation.
    /// @param count Number of ordered events.
    /// @param values Output segmented scan inputs.
    /// @return Nothing.
    /// @invariant Ordered indices group tasks and preserve processor event order.
    __global__ void phase2OrderingScanInputKernel(
        const gl::gpu::DeviceAcceptedGrowthEvent* events,
        const gl::gpu::DevicePhase2GrowthCall* calls,
        const uint32_t* orderedIndices,
        uint32_t count,
        DeviceGrowthScanValue* values) {
        const uint32_t position = blockIdx.x * blockDim.x + threadIdx.x;
        if (position >= count) return;
        const uint32_t eventIndex = orderedIndices[position];
        assert(eventIndex < count);
        const gl::gpu::DeviceAcceptedGrowthEvent& event = events[eventIndex];
        DeviceGrowthScanValue value{};
        value.taskIndex = calls[event.callIndex].taskIndex;
        value.growthPosition =
            (event.flags & gl::gpu::kDeviceGrowthEventSubkeySatisfied) != 0
            ? 1u : 0u;
        values[position] = value;
    }

    /// @brief Fold one 64-bit word into an FNV-1a request hash.
    ///
    /// @param hash In/out running 64-bit FNV-1a state.
    /// @param word Word whose little-endian bytes are folded.
    /// @return Nothing.
    /// @invariant Equality is still checked field-by-field; this hash selects slots.
    __device__ void foldRequestHashWord(uint64_t& hash, uint64_t word) {
        for (uint32_t byte = 0; byte < 8; ++byte) {
            hash ^= static_cast<uint8_t>(word >> (byte * 8));
            hash *= 1099511628211ull;
        }
    }

    /// @brief Hash one event's per-call semantic request key.
    ///
    /// @details
    /// Hashes call identity, premise count, and each statement's packed
    /// `(originalId, validityId)` pair. Statement-vector indices are deliberately
    /// excluded because duplicate rows with the same semantic key must deduplicate.
    ///
    /// @param columns Resident statement column.
    /// @param tasks Uploaded task descriptors.
    /// @param calls Uploaded growth calls.
    /// @param event Candidate event.
    /// @return 64-bit open-address slot hash.
    /// @invariant Event call/task/block links are valid.
    __device__ uint64_t semanticRequestHash(
        const ProjectionSemanticColumns& columns,
        const gl::gpu::DevicePhase2Task* tasks,
        const gl::gpu::DevicePhase2GrowthCall* calls,
        const gl::gpu::DeviceAcceptedGrowthEvent& event) {
        uint64_t hash = 14695981039346656037ull;
        foldRequestHashWord(hash, event.callIndex);
        foldRequestHashWord(hash, static_cast<uint32_t>(event.count));
        const gl::gpu::DevicePhase2Task& task =
            tasks[calls[event.callIndex].taskIndex];
        const gl::gpu::DeviceLogicalBlockProjection& block =
            columns.logicalBlocks[task.logicalBlockIndex];
        for (gl::NameId premise = 0; premise < event.count; ++premise) {
            const gl::NameId statementIndex = event.statementIndices[premise];
            assert(statementIndex >= 0);
            assert(static_cast<uint32_t>(statementIndex) < block.statementCount);
            const gl::IntEncodedExpr& expression = columns.statements[
                block.statementOffset
                    + static_cast<uint32_t>(statementIndex)];
            const uint64_t packed =
                (static_cast<uint64_t>(
                    static_cast<uint32_t>(expression.originalId)) << 32)
                | static_cast<uint32_t>(expression.validityId);
            foldRequestHashWord(hash, packed);
        }
        return hash;
    }

    /// @brief Compare two events by the processor emitter's semantic request key.
    ///
    /// @param columns Resident statement column.
    /// @param tasks Uploaded task descriptors.
    /// @param calls Uploaded growth calls.
    /// @param left First event.
    /// @param right Second event.
    /// @return True exactly for same call, count, and packed statement-key path.
    /// @invariant Both events belong to valid uploaded calls.
    __device__ bool semanticRequestKeysEqual(
        const ProjectionSemanticColumns& columns,
        const gl::gpu::DevicePhase2Task* tasks,
        const gl::gpu::DevicePhase2GrowthCall* calls,
        const gl::gpu::DeviceAcceptedGrowthEvent& left,
        const gl::gpu::DeviceAcceptedGrowthEvent& right) {
        if (left.callIndex != right.callIndex || left.count != right.count)
            return false;
        const gl::gpu::DevicePhase2Task& task =
            tasks[calls[left.callIndex].taskIndex];
        const gl::gpu::DeviceLogicalBlockProjection& block =
            columns.logicalBlocks[task.logicalBlockIndex];
        for (gl::NameId premise = 0; premise < left.count; ++premise) {
            const gl::IntEncodedExpr& leftExpression = columns.statements[
                block.statementOffset
                    + static_cast<uint32_t>(
                        left.statementIndices[premise])];
            const gl::IntEncodedExpr& rightExpression = columns.statements[
                block.statementOffset
                    + static_cast<uint32_t>(
                        right.statementIndices[premise])];
            if (leftExpression.originalId != rightExpression.originalId
                || leftExpression.validityId
                    != rightExpression.validityId) return false;
        }
        return true;
    }

    /// @brief Insert recordable events into the collision-exact per-call table.
    ///
    /// @details
    /// One thread handles one exact-order event. Open addressing uses the full
    /// semantic equality test after every occupied-slot hash collision. Duplicate
    /// requests atomically retain their minimum event order, independent of
    /// thread timing.
    ///
    /// @param columns Resident semantic columns.
    /// @param tasks Uploaded task descriptors.
    /// @param calls Uploaded growth calls.
    /// @param events Persistent unordered events.
    /// @param orderedIndices Exact event permutation.
    /// @param eventCount Number of events.
    /// @param owners Deduplication-slot representative event indices.
    /// @param minimumOrders Deduplication-slot minimum exact event orders.
    /// @param slotCount Power-of-two slot count.
    /// @return Nothing.
    /// @invariant Slot load never exceeds one half at retained capacities.
    __global__ void phase2DeduplicateRequestsKernel(
        ProjectionSemanticColumns columns,
        const gl::gpu::DevicePhase2Task* tasks,
        const gl::gpu::DevicePhase2GrowthCall* calls,
        const gl::gpu::DeviceAcceptedGrowthEvent* events,
        const uint32_t* orderedIndices,
        uint32_t eventCount,
        uint32_t* owners,
        uint32_t* minimumOrders,
        uint32_t slotCount) {
        const uint32_t order = blockIdx.x * blockDim.x + threadIdx.x;
        if (order >= eventCount) return;
        const uint32_t eventIndex = orderedIndices[order];
        const gl::gpu::DeviceAcceptedGrowthEvent& event = events[eventIndex];
        const uint32_t recordFlags =
            gl::gpu::kDeviceGrowthEventWholeKeyPresent
            | gl::gpu::kDeviceGrowthEventTermsSatisfied;
        if ((event.flags & recordFlags) != recordFlags) return;
        assert(slotCount > 0 && (slotCount & (slotCount - 1)) == 0);
        uint32_t slot = static_cast<uint32_t>(projectionPodHash(
            semanticRequestHash(columns, tasks, calls, event)))
            & (slotCount - 1);
        constexpr uint32_t emptySlot = 0xffffffffu;
        for (uint32_t probe = 0; probe < slotCount; ++probe) {
            const uint32_t observed = atomicCAS(
                &owners[slot], emptySlot, eventIndex);
            if (observed == emptySlot
                || semanticRequestKeysEqual(
                    columns, tasks, calls, events[observed], event)) {
                atomicMin(&minimumOrders[slot], order);
                return;
            }
            slot = (slot + 1) & (slotCount - 1);
        }
        assert(false);
    }

    /// @brief Emit one unique request token from every occupied deduplication slot.
    ///
    /// @param orderedIndices Exact event permutation.
    /// @param scanValues Inclusive task-local growth positions by event order.
    /// @param owners Deduplication-slot representative indices.
    /// @param minimumOrders Deduplication-slot retained minimum orders.
    /// @param slotCount Number of deduplication slots.
    /// @param tokens Arbitrary-order unique token output.
    /// @param tokenCapacity Fixed output capacity.
    /// @param tokenCount Shared output counter.
    /// @return Nothing.
    /// @invariant Every occupied slot has a finite minimum order.
    __global__ void phase2EmitUniqueRequestsKernel(
        const uint32_t* orderedIndices,
        const DeviceGrowthScanValue* scanValues,
        const uint32_t* owners,
        const uint32_t* minimumOrders,
        uint32_t slotCount,
        gl::gpu::DeviceOrderedRequestToken* tokens,
        uint32_t tokenCapacity,
        uint32_t* tokenCount) {
        const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
        if (slot >= slotCount) return;
        constexpr uint32_t emptySlot = 0xffffffffu;
        if (owners[slot] == emptySlot) return;
        const uint32_t order = minimumOrders[slot];
        assert(order != emptySlot);
        const uint32_t outputIndex = atomicAdd(tokenCount, 1u);
        assert(outputIndex < tokenCapacity);
        gl::gpu::DeviceOrderedRequestToken token{};
        token.growthPosition = scanValues[order].growthPosition;
        token.eventIndex = orderedIndices[order];
        token.eventOrder = order;
        tokens[outputIndex] = token;
    }

    /// @brief Materialize request event-order keys for the final radix sort.
    ///
    /// @param tokens Arbitrary-order unique request tokens.
    /// @param count Number of tokens.
    /// @param keys Output event-order keys.
    /// @return Nothing.
    /// @invariant Every event order is unique after semantic deduplication.
    __global__ void phase2RequestOrderKeysKernel(
        const gl::gpu::DeviceOrderedRequestToken* tokens,
        uint32_t count,
        uint32_t* keys) {
        const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < count) keys[index] = tokens[index].eventOrder;
    }

    struct DeviceEvaluationCounters {
        uint32_t dependencyPassCount{ 0 };
        uint32_t reverseOwnerCount{ 0 };
        uint32_t localValueCount{ 0 };
        uint32_t firingRecordCount{ 0 };
        uint32_t generatedByteCount{ 0 };
        uint32_t levelValueCount{ 0 };
        uint32_t originDependencyCount{ 0 };
        uint32_t markerKeyCount{ 0 };
        uint32_t markerRemainingArgCount{ 0 };
        uint32_t markerArgCount{ 0 };
    };

    static_assert(sizeof(DeviceEvaluationCounters) == 40);

    /// @brief Read one little-endian projected `NameId` from packed bytes.
    ///
    /// @param bytes Four readable bytes from a serialized key.
    /// @return Reconstructed signed identifier value.
    /// @invariant `NameId` is the project's fixed 32-bit identifier type.
    __device__ gl::NameId readProjectedNameId(const char* bytes) {
        static_assert(sizeof(gl::NameId) == 4);
        const uint32_t raw = static_cast<uint8_t>(bytes[0])
            | (static_cast<uint32_t>(static_cast<uint8_t>(bytes[1])) << 8)
            | (static_cast<uint32_t>(static_cast<uint8_t>(bytes[2])) << 16)
            | (static_cast<uint32_t>(static_cast<uint8_t>(bytes[3])) << 24);
        return static_cast<gl::NameId>(raw);
    }

    /// @brief Read one unaligned little-endian signed 32-bit projected field.
    ///
    /// @param bytes Start of the serialized record.
    /// @param length Total readable record length.
    /// @param offset Byte offset of the requested field.
    /// @return Reconstructed signed 32-bit value.
    /// @invariant The requested four-byte field lies wholly inside the record.
    __device__ int32_t readProjectedI32(
        const char* bytes, uint32_t length, uint32_t offset) {
        assert(offset <= length);
        assert(sizeof(int32_t) <= length - offset);
        return static_cast<int32_t>(readProjectedNameId(bytes + offset));
    }

    /// @brief Assert the exact serialized `LocalMemoryValue` frame.
    ///
    /// @param bytes Start of one projected LMV blob.
    /// @param length Blob byte length.
    /// @return Nothing.
    /// @invariant The frame is `34 + 4*(levels + keys + remaining)` bytes.
    __device__ void validateProjectedLmv(
        const char* bytes, uint32_t length) {
        assert(length >= 26);
        const int32_t levelCount = readProjectedI32(bytes, length, 22);
        assert(levelCount >= 0);
        const uint64_t keyCountOffset = 26ull
            + static_cast<uint64_t>(levelCount) * sizeof(int32_t);
        assert(keyCountOffset + sizeof(int32_t) <= length);
        const int32_t keyCount = readProjectedI32(
            bytes, length, static_cast<uint32_t>(keyCountOffset));
        assert(keyCount >= 0);
        const uint64_t remainingCountOffset = 30ull
            + static_cast<uint64_t>(levelCount + keyCount) * sizeof(int32_t);
        assert(remainingCountOffset + sizeof(int32_t) <= length);
        const int32_t remainingCount = readProjectedI32(
            bytes, length, static_cast<uint32_t>(remainingCountOffset));
        assert(remainingCount >= 0);
        const uint64_t expected = 34ull
            + static_cast<uint64_t>(levelCount + keyCount + remainingCount)
                * sizeof(int32_t);
        assert(expected == length);
    }

    /// @brief Sequential cursor over greedy-substituted rule bytes.
    ///
    /// @details
    /// Source positions retain processor token-boundary semantics. A replacement
    /// identifier addresses the logical block's projected NameMap and is emitted
    /// before the cursor resumes at the source delimiter after the numeric key.
    struct ProjectedSubstitutionCursor {
        const char* source{ nullptr };
        uint32_t sourceLength{ 0 };
        uint32_t sourcePosition{ 0 };
        const gl::NameId* reverseIds{ nullptr };
        gl::NameId reverseCount{ 0 };
        gl::NameId replacementId{ 0 };
        uint32_t replacementPosition{ 0 };
    };

    /// @brief Emit the next greedy-longest substituted rule byte.
    ///
    /// @details
    /// Numeric keys are exactly `1..reverseCount`. Parsing the complete numeric
    /// token is equivalent to the processor KeyTrie: a longer out-of-range token
    /// prevents boundary acceptance and is never retried as a shorter key.
    ///
    /// @param columns Resident name records and bytes.
    /// @param block Owning logical block.
    /// @param cursor Mutable substitution cursor.
    /// @param output Receives the next byte when one exists.
    /// @return True when a byte was emitted, false at source exhaustion.
    /// @invariant Every reverse identifier belongs to the block NameMap.
    __device__ bool nextProjectedSubstitutionByte(
        const ProjectionSemanticColumns& columns,
        const gl::gpu::DeviceLogicalBlockProjection& block,
        ProjectedSubstitutionCursor& cursor,
        char& output) {
        if (cursor.replacementId != 0) {
            assert(cursor.replacementId > 0);
            assert(static_cast<uint32_t>(cursor.replacementId)
                < block.nameRecordCount);
            const gl::gpu::DeviceNameRecord& record = columns.nameRecords[
                block.nameRecordOffset
                + static_cast<uint32_t>(cursor.replacementId)];
            assert(record.byteLength > 0);
            assert(cursor.replacementPosition < record.byteLength);
            output = columns.nameBytes[
                record.byteOffset + cursor.replacementPosition];
            ++cursor.replacementPosition;
            if (cursor.replacementPosition == record.byteLength) {
                cursor.replacementId = 0;
                cursor.replacementPosition = 0;
            }
            return true;
        }
        if (cursor.sourcePosition >= cursor.sourceLength) return false;

        const uint32_t position = cursor.sourcePosition;
        const char previous = position == 0
            ? '\0' : cursor.source[position - 1];
        if ((previous == '[' || previous == ',')
            && cursor.source[position] >= '0'
            && cursor.source[position] <= '9') {
            uint32_t end = position;
            uint64_t value = 0;
            bool exceedsRange = false;
            while (end < cursor.sourceLength
                && cursor.source[end] >= '0'
                && cursor.source[end] <= '9') {
                if (!exceedsRange) {
                    value = value * 10u
                        + static_cast<uint32_t>(cursor.source[end] - '0');
                    if (value > static_cast<uint64_t>(cursor.reverseCount))
                        exceedsRange = true;
                }
                ++end;
            }
            const bool canonical = cursor.source[position] != '0';
            const bool bounded = !exceedsRange && value >= 1
                && value <= static_cast<uint64_t>(cursor.reverseCount);
            const bool trailingBoundary = end < cursor.sourceLength
                && (cursor.source[end] == ']' || cursor.source[end] == ',');
            if (canonical && bounded && trailingBoundary) {
                cursor.sourcePosition = end;
                cursor.replacementId = cursor.reverseIds[
                    static_cast<uint32_t>(value - 1)];
                cursor.replacementPosition = 0;
                assert(cursor.replacementId > 0);
                return nextProjectedSubstitutionByte(
                    columns, block, cursor, output);
            }
        }
        output = cursor.source[cursor.sourcePosition++];
        return true;
    }

    /// @brief Cursor over substituted bytes after repeated token-leading `u_` removal.
    struct ProjectedFiringExpressionCursor {
        ProjectedSubstitutionCursor substitution{};
        uint32_t started{ 0 };
        uint32_t haveCurrent{ 0 };
        uint32_t haveFollowing{ 0 };
        char previousEmitted{ '\0' };
        char current{ '\0' };
        char following{ '\0' };
    };

    /// @brief Emit the next fully transformed firing-expression byte.
    ///
    /// @details
    /// Implements the processor erase-and-recheck rule exactly: after `[` or `,`,
    /// each adjacent `u_` pair is discarded without changing the previous emitted
    /// delimiter, so `[u_u_x]` becomes `[x]`.
    ///
    /// @param columns Resident name records and bytes.
    /// @param block Owning logical block.
    /// @param cursor Mutable final-expression cursor.
    /// @param output Receives the next final byte.
    /// @return True when a byte was emitted, false at transformed exhaustion.
    /// @invariant The substitution cursor and block belong to one projected LMV.
    __device__ bool nextProjectedFiringExpressionByte(
        const ProjectionSemanticColumns& columns,
        const gl::gpu::DeviceLogicalBlockProjection& block,
        ProjectedFiringExpressionCursor& cursor,
        char& output) {
        if (cursor.started == 0) {
            if (!nextProjectedSubstitutionByte(
                    columns, block, cursor.substitution, output)) return false;
            cursor.started = 1;
            cursor.previousEmitted = output;
            return true;
        }
        for (;;) {
            if (cursor.haveCurrent == 0) {
                if (!nextProjectedSubstitutionByte(
                        columns, block, cursor.substitution,
                        cursor.current)) return false;
                cursor.haveCurrent = 1;
            }
            if (cursor.haveFollowing == 0) {
                cursor.haveFollowing = nextProjectedSubstitutionByte(
                    columns, block, cursor.substitution,
                    cursor.following) ? 1u : 0u;
            }
            if (cursor.haveFollowing != 0
                && cursor.current == 'u' && cursor.following == '_'
                && (cursor.previousEmitted == '['
                    || cursor.previousEmitted == ',')) {
                cursor.haveCurrent = 0;
                cursor.haveFollowing = 0;
                continue;
            }
            output = cursor.current;
            cursor.previousEmitted = output;
            if (cursor.haveFollowing != 0) {
                cursor.current = cursor.following;
                cursor.haveCurrent = 1;
                cursor.haveFollowing = 0;
            }
            else {
                cursor.haveCurrent = 0;
            }
            return true;
        }
    }

    /// @brief Streaming state for `it_(digits)_lev_digits_` maximum extraction.
    struct ProjectedIterationScanner {
        uint32_t state{ 0 };
        uint64_t firstDigits{ 0 };
        uint32_t firstDigitsOverflow{ 0 };
        int32_t maximum{ -1 };
    };

    /// @brief Consume one final expression byte for iteration-prefix extraction.
    ///
    /// @param scanner Mutable deterministic lexical scanner.
    /// @param byte Next byte in transformed expression order.
    /// @return Nothing.
    /// @invariant A fully matched first digit run must fit signed 32-bit range.
    __device__ void consumeProjectedIterationByte(
        ProjectedIterationScanner& scanner, char byte) {
        const bool digit = byte >= '0' && byte <= '9';
        const auto restart = [&]() {
            scanner.state = byte == 'i' ? 1u : 0u;
            scanner.firstDigits = 0;
            scanner.firstDigitsOverflow = 0;
        };
        switch (scanner.state) {
        case 0: if (byte == 'i') scanner.state = 1; break;
        case 1: if (byte == 't') scanner.state = 2; else restart(); break;
        case 2: if (byte == '_') scanner.state = 3; else restart(); break;
        case 3:
            if (digit) {
                scanner.firstDigits = static_cast<uint32_t>(byte - '0');
                scanner.state = 4;
            }
            else restart();
            break;
        case 4:
            if (digit) {
                const uint32_t d = static_cast<uint32_t>(byte - '0');
                if (scanner.firstDigits
                        > (static_cast<uint64_t>(INT_MAX) - d) / 10u) {
                    scanner.firstDigitsOverflow = 1;
                }
                else if (scanner.firstDigitsOverflow == 0) {
                    scanner.firstDigits = scanner.firstDigits * 10u + d;
                }
            }
            else if (byte == '_') scanner.state = 5;
            else restart();
            break;
        case 5: if (byte == 'l') scanner.state = 6; else restart(); break;
        case 6: if (byte == 'e') scanner.state = 7; else restart(); break;
        case 7: if (byte == 'v') scanner.state = 8; else restart(); break;
        case 8: if (byte == '_') scanner.state = 9; else restart(); break;
        case 9: if (digit) scanner.state = 10; else restart(); break;
        case 10:
            if (!digit) {
                if (byte == '_') {
                    assert(scanner.firstDigitsOverflow == 0);
                    const int32_t value = static_cast<int32_t>(
                        scanner.firstDigits);
                    if (value > scanner.maximum) scanner.maximum = value;
                    scanner.state = 0;
                    scanner.firstDigits = 0;
                    scanner.firstDigitsOverflow = 0;
                }
                else restart();
            }
            break;
        default: assert(false); break;
        }
    }

    /// @brief Measure or write one byte-exact transformed firing expression.
    ///
    /// @param columns Resident name records and bytes.
    /// @param block Owning logical block.
    /// @param source Rule-head source bytes.
    /// @param sourceLength Rule-head source length.
    /// @param reverseIds Normalized-variable identifier reverse map.
    /// @param reverseCount Number of populated reverse identifiers.
    /// @param output Optional destination; null performs the measurement pass.
    /// @param maximumIteration Optional destination for maximum `it_` iteration.
    /// @return Exact transformed byte length.
    /// @invariant A non-null output addresses the returned number of writable bytes.
    __device__ uint32_t transformProjectedFiringExpression(
        const ProjectionSemanticColumns& columns,
        const gl::gpu::DeviceLogicalBlockProjection& block,
        const char* source,
        uint32_t sourceLength,
        const gl::NameId* reverseIds,
        gl::NameId reverseCount,
        char* output,
        int32_t* maximumIteration) {
        ProjectedFiringExpressionCursor cursor{};
        cursor.substitution.source = source;
        cursor.substitution.sourceLength = sourceLength;
        cursor.substitution.reverseIds = reverseIds;
        cursor.substitution.reverseCount = reverseCount;
        ProjectedIterationScanner scanner{};
        uint32_t length = 0;
        char byte = '\0';
        while (nextProjectedFiringExpressionByte(
                columns, block, cursor, byte)) {
            if (output != nullptr) output[length] = byte;
            if (maximumIteration != nullptr)
                consumeProjectedIterationByte(scanner, byte);
            ++length;
        }
        if (maximumIteration != nullptr) *maximumIteration = scanner.maximum;
        return length;
    }

    /// @brief Exact device summary of all `int_lev_<level>_<id>` occurrences.
    ///
    /// @details
    /// Reproduces the processor's non-overlapping regex-iterator twin: failed
    /// starts advance one byte and successful starts resume at the match end.
    /// It records the first token, whether another distinct token exists, and
    /// whether every parsed level is strictly below the logical-block level.
    ///
    /// @param source Canonical expression bytes.
    /// @param length Expression byte length.
    /// @param levelBound Exclusive logical-block level bound.
    /// @param firstStart Receives the first token start when the verdict is nonzero.
    /// @param firstLength Receives the first token length when nonzero.
    /// @param allBelow Receives one exactly when every occurrence level is below.
    /// @return Zero, one, or two for no token, one distinct token, or multiple.
    /// @invariant Every matched digit run may be arbitrarily long; overflow makes
    ///            `allBelow` false without changing occurrence recognition.
    __device__ uint32_t scanProjectedIntLevSummary(
        const char* source,
        uint32_t length,
        int32_t levelBound,
        uint32_t& firstStart,
        uint32_t& firstLength,
        uint32_t& allBelow) {
        uint32_t position = 0;
        uint32_t haveFirst = 0;
        uint32_t multiple = 0;
        allBelow = 1;
        while (position + 11 <= length) {
            if (!projectionBytesEqual(source + position, "int_lev_", 8)) {
                ++position;
                continue;
            }
            uint32_t cursor = position + 8;
            const uint32_t firstDigit = cursor;
            while (cursor < length
                && source[cursor] >= '0' && source[cursor] <= '9') ++cursor;
            if (cursor == firstDigit || cursor >= length
                || source[cursor] != '_') {
                ++position;
                continue;
            }
            const uint32_t firstDigitEnd = cursor;
            ++cursor;
            const uint32_t secondDigit = cursor;
            while (cursor < length
                && source[cursor] >= '0' && source[cursor] <= '9') ++cursor;
            if (cursor == secondDigit) {
                ++position;
                continue;
            }
            const uint32_t tokenLength = cursor - position;
            if (haveFirst == 0) {
                firstStart = position;
                firstLength = tokenLength;
                haveFirst = 1;
            }
            else if (firstLength != tokenLength
                || !projectionBytesEqual(
                    source + firstStart, source + position, tokenLength)) {
                multiple = 1;
            }
            uint64_t parsedLevel = 0;
            uint32_t overflow = 0;
            for (uint32_t digit = firstDigit;
                 digit < firstDigitEnd; ++digit) {
                const uint32_t value = static_cast<uint32_t>(
                    source[digit] - '0');
                if (parsedLevel
                    > (static_cast<uint64_t>(INT_MAX) - value) / 10u) {
                    overflow = 1;
                }
                else if (overflow == 0) {
                    parsedLevel = parsedLevel * 10u + value;
                }
            }
            if (overflow != 0
                || static_cast<int64_t>(parsedLevel) >= levelBound) {
                allBelow = 0;
            }
            position = cursor;
        }
        if (haveFirst == 0) return 0;
        return multiple != 0 ? 2u : 1u;
    }

    /// @brief Cursor over bytes after replacing every exact occurrence of one token.
    struct ProjectedSingleReplacementCursor {
        const char* source{ nullptr };
        uint32_t sourceLength{ 0 };
        uint32_t sourcePosition{ 0 };
        const char* token{ nullptr };
        uint32_t tokenLength{ 0 };
        const char* replacement{ nullptr };
        uint32_t replacementLength{ 0 };
        uint32_t replacementPosition{ 0 };
        uint32_t emittingReplacement{ 0 };
    };

    /// @brief Emit one byte of an exact all-occurrences single-token replacement.
    ///
    /// @param cursor Mutable replacement cursor.
    /// @param output Receives the next transformed byte.
    /// @return True when a byte was emitted, false at transformed exhaustion.
    /// @invariant The token and replacement are nonempty readable spans.
    __device__ bool nextProjectedSingleReplacementByte(
        ProjectedSingleReplacementCursor& cursor,
        char& output) {
        assert(cursor.tokenLength > 0);
        assert(cursor.replacementLength > 0);
        if (cursor.emittingReplacement != 0) {
            output = cursor.replacement[cursor.replacementPosition++];
            if (cursor.replacementPosition == cursor.replacementLength) {
                cursor.replacementPosition = 0;
                cursor.emittingReplacement = 0;
            }
            return true;
        }
        if (cursor.sourcePosition >= cursor.sourceLength) return false;
        if (cursor.tokenLength <= cursor.sourceLength - cursor.sourcePosition
            && projectionBytesEqual(
                cursor.source + cursor.sourcePosition,
                cursor.token, cursor.tokenLength)) {
            cursor.sourcePosition += cursor.tokenLength;
            output = cursor.replacement[0];
            cursor.replacementPosition = 1;
            cursor.emittingReplacement = cursor.replacementLength > 1 ? 1u : 0u;
            if (cursor.emittingReplacement == 0)
                cursor.replacementPosition = 0;
            return true;
        }
        output = cursor.source[cursor.sourcePosition++];
        return true;
    }

    /// @brief Probe NameMap after replacing one exact token by `marker` everywhere.
    ///
    /// @details
    /// Hashes and compares the transformed stream directly against projected name
    /// slots. No per-thread expression buffer is allocated; every collision probe
    /// restarts the deterministic cursor and compares the complete byte stream.
    ///
    /// @param columns Resident projected NameMap columns.
    /// @param block Owning logical block.
    /// @param source Original expression bytes.
    /// @param sourceLength Original expression byte length.
    /// @param token Sole distinct `int_lev` token to replace.
    /// @param tokenLength Token byte length.
    /// @return Arena-global name-record index, or `-1` for a defined miss.
    /// @invariant `block.nameSlots` is the fixed-load table for these name records.
    __device__ int32_t findProjectedMarkerReplacementName(
        const ProjectionSemanticColumns& columns,
        const gl::gpu::DeviceLogicalBlockProjection& block,
        const char* source,
        uint32_t sourceLength,
        const char* token,
        uint32_t tokenLength) {
        constexpr char replacement[] = "marker";
        const auto makeCursor = [&]() {
            ProjectedSingleReplacementCursor cursor{};
            cursor.source = source;
            cursor.sourceLength = sourceLength;
            cursor.token = token;
            cursor.tokenLength = tokenLength;
            cursor.replacement = replacement;
            cursor.replacementLength = 6;
            return cursor;
        };
        ProjectedSingleReplacementCursor measure = makeCursor();
        uint64_t hash = 14695981039346656037ull;
        uint32_t transformedLength = 0;
        char byte = '\0';
        while (nextProjectedSingleReplacementByte(measure, byte)) {
            hash ^= static_cast<uint8_t>(byte);
            hash *= 1099511628211ull;
            ++transformedLength;
        }
        if (block.nameSlotCount == 0) return -1;
        uint32_t slot = static_cast<uint32_t>(hash)
            & (block.nameSlotCount - 1);
        for (;;) {
            const int32_t recordIndex = columns.nameSlots[
                block.nameSlotOffset + slot];
            if (recordIndex == -1) return -1;
            assert(recordIndex >= 0);
            const gl::gpu::DeviceNameRecord& record = columns.nameRecords[
                static_cast<uint32_t>(recordIndex)];
            if (record.byteLength == transformedLength) {
                ProjectedSingleReplacementCursor compare = makeCursor();
                bool equal = true;
                for (uint32_t index = 0; index < transformedLength; ++index) {
                    char actual = '\0';
                    const bool emitted = nextProjectedSingleReplacementByte(
                        compare, actual);
                    assert(emitted);
                    if (actual != columns.nameBytes[record.byteOffset + index]) {
                        equal = false;
                        break;
                    }
                }
                if (equal) return recordIndex;
            }
            slot = (slot + 1) & (block.nameSlotCount - 1);
        }
    }

    /// @brief Exact GPU twin of `ExpressionAnalyzer::allowedForMail`.
    ///
    /// @param columns Resident name and plain-data columns.
    /// @param block Owning logical block and level.
    /// @param originalId Premise original-expression NameMap identifier.
    /// @return True exactly when the premise is eligible for statement mail.
    /// @invariant The original identifier belongs to the block NameMap and every
    ///            compiled-category assertion matches the processor read fence.
    __device__ bool projectedAllowedForMail(
        const ProjectionSemanticColumns& columns,
        const gl::gpu::DeviceLogicalBlockProjection& block,
        gl::NameId originalId) {
        assert(originalId > 0);
        const uint32_t recordIndex = block.nameRecordOffset
            + static_cast<uint32_t>(originalId);
        assert(recordIndex < block.nameRecordOffset + block.nameRecordCount);
        const gl::gpu::DeviceNameRecord& original =
            columns.nameRecords[recordIndex];
        const char* const expression = columns.nameBytes + original.byteOffset;
        uint32_t tokenStart = 0;
        uint32_t tokenLength = 0;
        uint32_t allBelow = 0;
        const uint32_t verdict = scanProjectedIntLevSummary(
            expression, original.byteLength, block.level,
            tokenStart, tokenLength, allBelow);
        if (verdict == 0) return true;
        const bool implication = original.byteLength >= 2
            && expression[0] == '(' && expression[1] == '>';
        if (!implication && allBelow != 0) return true;
        if (findProjectedPodMapEntry(
                columns, block,
                gl::gpu::DevicePodMapKind::mailEligibleStatements,
                originalId) >= 0) return true;
        if (verdict != 1 || implication) return false;
        const int32_t markerRecord = findProjectedMarkerReplacementName(
            columns, block, expression, original.byteLength,
            expression + tokenStart, tokenLength);
        if (markerRecord < 0) return false;
        assert(static_cast<uint32_t>(markerRecord) > block.nameRecordOffset);
        assert(static_cast<uint32_t>(markerRecord)
            < block.nameRecordOffset + block.nameRecordCount);
        const gl::NameId markerId = static_cast<gl::NameId>(
            static_cast<uint32_t>(markerRecord) - block.nameRecordOffset);
        if (findProjectedPodMapEntry(
                columns, block,
                gl::gpu::DevicePodMapKind::mailEligibleMarkers,
                markerId) < 0) return false;
        assert(original.compiledCategory
            != gl::gpu::DeviceCompiledCategory::absent);
        return original.compiledCategory
            != gl::gpu::DeviceCompiledCategory::atomic;
    }

    /// @brief Test the projected Site-F known-statement ancestor predicate.
    ///
    /// @param columns Resident NameMap and known-statement columns.
    /// @param block Owning logical block.
    /// @param expression Generated firing-head bytes.
    /// @param expressionLength Generated head byte length.
    /// @param validityId Deposit validity including self.
    /// @return True exactly when the head is known at self or an ancestor.
    /// @invariant `validityId` belongs to the block NameMap; a missing head name
    ///            is the defined not-known outcome and performs no mint.
    __device__ bool projectedAncestorKnown(
        const ProjectionSemanticColumns& columns,
        const gl::gpu::DeviceLogicalBlockProjection& block,
        const char* expression,
        uint32_t expressionLength,
        gl::NameId validityId) {
        const int32_t recordIndex = findProjectedNameRecord(
            columns, block, expression, expressionLength);
        if (recordIndex < 0) return false;
        const gl::NameId originalId = static_cast<gl::NameId>(
            static_cast<uint32_t>(recordIndex) - block.nameRecordOffset);
        assert(originalId > 0);
        for (gl::NameId validity = validityId; validity != 0;) {
            const int64_t packed = static_cast<int64_t>(
                (static_cast<uint64_t>(
                    static_cast<uint32_t>(originalId)) << 32)
                | static_cast<uint32_t>(validity));
            if (findProjectedPodMapEntry(
                    columns, block,
                    gl::gpu::DevicePodMapKind::knownStatements,
                    packed) >= 0) return true;
            const uint32_t validityRecord = block.nameRecordOffset
                + static_cast<uint32_t>(validity);
            assert(validityRecord
                < block.nameRecordOffset + block.nameRecordCount);
            validity = columns.nameRecords[validityRecord].parentId;
        }
        return false;
    }

    /// @brief Merge request and LMV levels into processor sorted-unique order.
    ///
    /// @details
    /// Premise level runs and the serialized LMV level run are already ascending.
    /// A bounded multiway merge advances one cursor per premise plus the rule run,
    /// discards the premise-only non-derived tier, and emits each non-negative
    /// value once. Passing null output performs the exact measurement pass.
    ///
    /// @param columns Resident plain-data runs and statements.
    /// @param block Owning logical block.
    /// @param event Ordered request path.
    /// @param lmv Serialized LMV bytes.
    /// @param lmvLength LMV byte length.
    /// @param assertRuleLevels Whether the head-path non-negative rule invariant
    ///                         is asserted.
    /// @param output Optional destination for the merged level run.
    /// @return Sorted-unique output count.
    /// @invariant `event.count` is bounded by `MAX_EXPRESSIONS` and every stored
    ///            run is ascending.
    __device__ uint32_t mergeProjectedFiringLevels(
        const ProjectionSemanticColumns& columns,
        const gl::gpu::DeviceLogicalBlockProjection& block,
        const gl::gpu::DeviceAcceptedGrowthEvent& event,
        const char* lmv,
        uint32_t lmvLength,
        bool assertRuleLevels,
        int32_t* output) {
        uint32_t premiseOffsets[gl::ExecutionParameters::MAX_EXPRESSIONS];
        uint32_t premiseCounts[gl::ExecutionParameters::MAX_EXPRESSIONS];
        uint32_t premiseCursors[gl::ExecutionParameters::MAX_EXPRESSIONS]{};
        assert(event.count <= gl::ExecutionParameters::MAX_EXPRESSIONS);
        for (gl::NameId premise = 0; premise < event.count; ++premise) {
            const gl::IntEncodedExpr& expression = columns.statements[
                block.statementOffset + static_cast<uint32_t>(
                    event.statementIndices[premise])];
            const int64_t packed = static_cast<int64_t>(
                (static_cast<uint64_t>(
                    static_cast<uint32_t>(expression.originalId)) << 32)
                | static_cast<uint32_t>(expression.validityId));
            const int32_t entryIndex = findProjectedPodMapEntry(
                columns, block,
                gl::gpu::DevicePodMapKind::statementLevels, packed);
            if (entryIndex < 0) {
                premiseOffsets[premise] = 0;
                premiseCounts[premise] = 0;
            }
            else {
                const gl::gpu::DevicePodMapEntry& entry =
                    columns.podMapEntries[static_cast<uint32_t>(entryIndex)];
                premiseOffsets[premise] = entry.runOffset;
                premiseCounts[premise] = entry.runCount;
            }
        }
        const int32_t ruleCountSigned = readProjectedI32(lmv, lmvLength, 22);
        assert(ruleCountSigned >= 0);
        const uint32_t ruleCount = static_cast<uint32_t>(ruleCountSigned);
        uint32_t ruleCursor = 0;
        uint32_t outputCount = 0;
        bool haveLast = false;
        int32_t last = 0;
        for (;;) {
            bool found = false;
            int32_t minimum = 0;
            for (gl::NameId premise = 0; premise < event.count; ++premise) {
                while (premiseCursors[premise] < premiseCounts[premise]
                    && columns.podRunValues[
                        premiseOffsets[premise] + premiseCursors[premise]] < 0) {
                    ++premiseCursors[premise];
                }
                if (premiseCursors[premise] < premiseCounts[premise]) {
                    const int32_t value = columns.podRunValues[
                        premiseOffsets[premise] + premiseCursors[premise]];
                    if (!found || value < minimum) {
                        minimum = value;
                        found = true;
                    }
                }
            }
            if (ruleCursor < ruleCount) {
                const int32_t value = readProjectedI32(
                    lmv, lmvLength, 26 + ruleCursor * sizeof(int32_t));
                if (assertRuleLevels) assert(value >= 0);
                if (!found || value < minimum) {
                    minimum = value;
                    found = true;
                }
            }
            if (!found) break;
            if (!haveLast || minimum != last) {
                if (output != nullptr) output[outputCount] = minimum;
                ++outputCount;
                last = minimum;
                haveLast = true;
            }
            for (gl::NameId premise = 0; premise < event.count; ++premise) {
                while (premiseCursors[premise] < premiseCounts[premise]
                    && columns.podRunValues[
                        premiseOffsets[premise] + premiseCursors[premise]]
                            == minimum) {
                    ++premiseCursors[premise];
                }
            }
            while (ruleCursor < ruleCount
                && readProjectedI32(
                    lmv, lmvLength, 26 + ruleCursor * sizeof(int32_t))
                    == minimum) ++ruleCursor;
        }
        return outputCount;
    }

    /// @brief Write one head's exact implication provenance dependency run.
    ///
    /// @details
    /// Entry zero is the source implication in rule-interner identifier space and
    /// its LMV validity. Request premises follow in decoded expression then
    /// decoded validity lexical order. Projection-time ranks reproduce both byte
    /// comparisons without device string sorting.
    ///
    /// @param columns Resident statement and decoded-name-rank columns.
    /// @param block Owning logical block.
    /// @param event Ordered request path.
    /// @param sourceImplicationId Rule-interner identifier of the source rule.
    /// @param sourceValidityId NameMap validity identifier stored by the LMV.
    /// @param output Destination for `1 + event.count` dependencies.
    /// @return Number of dependencies written.
    /// @invariant Every premise identifier belongs to the block NameMap.
    __device__ uint32_t writeProjectedHeadOriginDependencies(
        const ProjectionSemanticColumns& columns,
        const gl::gpu::DeviceLogicalBlockProjection& block,
        const gl::gpu::DeviceAcceptedGrowthEvent& event,
        int32_t sourceImplicationId,
        gl::NameId sourceValidityId,
        gl::gpu::DeviceEvaluationDependency* output) {
        assert(sourceImplicationId > 0);
        assert(event.count <= gl::ExecutionParameters::MAX_EXPRESSIONS);
        gl::NameId order[gl::ExecutionParameters::MAX_EXPRESSIONS];
        for (gl::NameId premise = 0; premise < event.count; ++premise)
            order[premise] = premise;
        const auto less = [&](gl::NameId left, gl::NameId right) {
            const gl::IntEncodedExpr& a = columns.statements[
                block.statementOffset + static_cast<uint32_t>(
                    event.statementIndices[left])];
            const gl::IntEncodedExpr& b = columns.statements[
                block.statementOffset + static_cast<uint32_t>(
                    event.statementIndices[right])];
            const uint32_t aOriginalRank = columns.nameRecords[
                block.nameRecordOffset + static_cast<uint32_t>(a.originalId)]
                    .decodedLexRank;
            const uint32_t bOriginalRank = columns.nameRecords[
                block.nameRecordOffset + static_cast<uint32_t>(b.originalId)]
                    .decodedLexRank;
            if (aOriginalRank != bOriginalRank)
                return aOriginalRank < bOriginalRank;
            const uint32_t aValidityRank = columns.nameRecords[
                block.nameRecordOffset + static_cast<uint32_t>(a.validityId)]
                    .decodedLexRank;
            const uint32_t bValidityRank = columns.nameRecords[
                block.nameRecordOffset + static_cast<uint32_t>(b.validityId)]
                    .decodedLexRank;
            return aValidityRank < bValidityRank;
        };
        for (gl::NameId index = 1; index < event.count; ++index) {
            const gl::NameId value = order[index];
            gl::NameId position = index;
            while (position > 0 && less(value, order[position - 1])) {
                order[position] = order[position - 1];
                --position;
            }
            order[position] = value;
        }
        output[0] = gl::gpu::DeviceEvaluationDependency{
            static_cast<gl::NameId>(sourceImplicationId), sourceValidityId };
        for (gl::NameId index = 0; index < event.count; ++index) {
            const gl::IntEncodedExpr& expression = columns.statements[
                block.statementOffset + static_cast<uint32_t>(
                    event.statementIndices[order[index]])];
            output[static_cast<uint32_t>(index) + 1] =
                gl::gpu::DeviceEvaluationDependency{
                    expression.originalId, expression.validityId };
        }
        return static_cast<uint32_t>(event.count) + 1;
    }

    /// @brief Test membership in one serialized projected `Int16SetKey`.
    ///
    /// @param columns Resident byte-key column.
    /// @param entry Overall remaining-argument map entry.
    /// @param identifier Raw argument identifier to find.
    /// @return True exactly when the entry's sorted set contains the identifier.
    /// @invariant The entry key is a canonical count-prefixed `NameId` set.
    __device__ bool projectedRemainingContains(
        const ProjectionSemanticColumns& columns,
        const gl::gpu::DeviceByteMapEntry& entry,
        gl::NameId identifier) {
        assert(entry.keyLength >= sizeof(gl::NameId));
        const char* const key = columns.byteKeyBytes + entry.keyOffset;
        const gl::NameId count = readProjectedNameId(key);
        assert(count >= 0);
        assert(entry.keyLength == static_cast<uint32_t>(count + 1)
            * sizeof(gl::NameId));
        for (gl::NameId index = 0; index < count; ++index) {
            if (readProjectedNameId(
                    key + static_cast<uint32_t>(index + 1)
                        * sizeof(gl::NameId)) == identifier) return true;
        }
        return false;
    }

    /// @brief Build the evaluator's ignore-u normalized key for one owner.
    ///
    /// @details
    /// Uses raw `argFullId` values throughout. Intrinsically unchangeable
    /// arguments and identifiers named by the remaining-argument owner pass
    /// through literally with marker one; every other argument is normalized in
    /// first-appearance order with marker zero.
    ///
    /// @param columns Resident statements and remaining-key bytes.
    /// @param block Owning logical block.
    /// @param event Request statement path.
    /// @param remainingEntry Selected remaining-argument owner.
    /// @param output Caller-owned normalized payload.
    /// @return Number of `NameId` payload values written.
    /// @invariant The request and output obey `MAX_EXPRESSIONS` and
    ///            `MAX_KEY_SLOTS`.
    __device__ gl::NameId buildProjectedEvaluationKey(
        const ProjectionSemanticColumns& columns,
        const gl::gpu::DeviceLogicalBlockProjection& block,
        const gl::gpu::DeviceAcceptedGrowthEvent& event,
        const gl::gpu::DeviceByteMapEntry& remainingEntry,
        gl::NameId* output) {
        gl::NameId variableIds[gl::ExecutionParameters::MAX_KEY_SLOTS];
        gl::NameId variableCount = 0;
        gl::NameId outputLength = 0;
        for (gl::NameId premise = 0; premise < event.count; ++premise) {
            const gl::NameId statementIndex = event.statementIndices[premise];
            assert(statementIndex >= 0);
            assert(static_cast<uint32_t>(statementIndex) < block.statementCount);
            const gl::IntEncodedExpr& expression = columns.statements[
                block.statementOffset + static_cast<uint32_t>(statementIndex)];
            assert(outputLength + 2
                <= gl::ExecutionParameters::MAX_KEY_SLOTS);
            output[outputLength++] = expression.nameId;
            output[outputLength++] = expression.negation;
            for (gl::NameId argument = 0;
                 argument < expression.arity; ++argument) {
                assert(outputLength + 2
                    <= gl::ExecutionParameters::MAX_KEY_SLOTS);
                const gl::NameId rawId = expression.argFullId[argument];
                const bool unchangeable = expression.argUnchangeable[argument]
                    != 0 || projectedRemainingContains(
                        columns, remainingEntry, rawId);
                if (unchangeable) {
                    output[outputLength++] = rawId;
                    output[outputLength++] = 1;
                    continue;
                }
                gl::NameId normalizedId = 0;
                for (gl::NameId known = 0;
                     known < variableCount; ++known) {
                    if (variableIds[known] == rawId) {
                        normalizedId = known + 1;
                        break;
                    }
                }
                if (normalizedId == 0) {
                    assert(variableCount
                        < gl::ExecutionParameters::MAX_KEY_SLOTS);
                    variableIds[variableCount++] = rawId;
                    normalizedId = variableCount;
                }
                output[outputLength++] = normalizedId;
                output[outputLength++] = 0;
            }
        }
        return outputLength;
    }

    /// @brief Reconstruct the evaluator reverse substitution for one request.
    ///
    /// @details
    /// Visits request arguments in processor order. Intrinsically unchangeable
    /// and remaining-owner identifiers are excluded; each first-seen variable is
    /// appended once, so output slot `i` is the raw NameId replacing numeric key
    /// `i + 1` in the rule template.
    ///
    /// @param columns Resident statements and remaining-key bytes.
    /// @param block Owning logical block.
    /// @param event Ordered request statement path.
    /// @param remainingEntry Selected remaining-argument owner.
    /// @param reverseIds Caller-owned reverse-map output.
    /// @return Number of populated reverse identifiers.
    /// @invariant The request is bounded by `MAX_KEY_SLOTS`.
    __device__ gl::NameId buildProjectedEvaluationReverseMap(
        const ProjectionSemanticColumns& columns,
        const gl::gpu::DeviceLogicalBlockProjection& block,
        const gl::gpu::DeviceAcceptedGrowthEvent& event,
        const gl::gpu::DeviceByteMapEntry& remainingEntry,
        gl::NameId* reverseIds) {
        gl::NameId count = 0;
        for (gl::NameId premise = 0; premise < event.count; ++premise) {
            const gl::NameId statementIndex = event.statementIndices[premise];
            assert(statementIndex >= 0);
            assert(static_cast<uint32_t>(statementIndex) < block.statementCount);
            const gl::IntEncodedExpr& expression = columns.statements[
                block.statementOffset + static_cast<uint32_t>(statementIndex)];
            for (gl::NameId argument = 0;
                 argument < expression.arity; ++argument) {
                const gl::NameId rawId = expression.argFullId[argument];
                if (expression.argUnchangeable[argument] != 0
                    || projectedRemainingContains(
                        columns, remainingEntry, rawId)) continue;
                bool seen = false;
                for (gl::NameId known = 0; known < count; ++known) {
                    if (reverseIds[known] == rawId) {
                        seen = true;
                        break;
                    }
                }
                if (!seen) {
                    assert(count < gl::ExecutionParameters::MAX_KEY_SLOTS);
                    reverseIds[count++] = rawId;
                }
            }
        }
        return count;
    }

    /// @brief Apply request dependency and validity gates and count reverse work.
    ///
    /// @param columns Resident semantic projection.
    /// @param tasks Uploaded task descriptors.
    /// @param calls Uploaded growth-call schedule.
    /// @param events Persistent growth events.
    /// @param tokens Exact ordered unique requests.
    /// @param requestCount Number of request tokens.
    /// @param states Per-request semantic state output.
    /// @param ownerCounts Per-request reverse-owner count output.
    /// @param counters Shared dependency/reverse totals.
    /// @return Nothing.
    /// @invariant Every token/event/call/task/block link belongs to one sweep.
    __global__ void phase2EvaluationRequestKernel(
        ProjectionSemanticColumns columns,
        const gl::gpu::DevicePhase2Task* tasks,
        const gl::gpu::DevicePhase2GrowthCall* calls,
        const gl::gpu::DeviceAcceptedGrowthEvent* events,
        const gl::gpu::DeviceOrderedRequestToken* tokens,
        uint32_t requestCount,
        gl::gpu::DeviceEvaluationRequestState* states,
        uint32_t* ownerCounts,
        DeviceEvaluationCounters* counters) {
        const uint32_t requestIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if (requestIndex >= requestCount) return;
        const gl::gpu::DeviceOrderedRequestToken& token = tokens[requestIndex];
        const gl::gpu::DeviceAcceptedGrowthEvent& event =
            events[token.eventIndex];
        const gl::gpu::DevicePhase2GrowthCall& call = calls[event.callIndex];
        const gl::gpu::DevicePhase2Task& task = tasks[call.taskIndex];
        const gl::gpu::DeviceLogicalBlockProjection& block =
            columns.logicalBlocks[task.logicalBlockIndex];
        gl::gpu::DeviceEvaluationRequestState state{};
        state.logicalBlockIndex = task.logicalBlockIndex;
        ownerCounts[requestIndex] = 0;

        for (gl::NameId premise = 0; premise < event.count; ++premise) {
            const gl::IntEncodedExpr& expression = columns.statements[
                block.statementOffset + static_cast<uint32_t>(
                    event.statementIndices[premise])];
            const int64_t packed = static_cast<int64_t>(
                (static_cast<uint64_t>(
                    static_cast<uint32_t>(expression.originalId)) << 32)
                | static_cast<uint32_t>(expression.validityId));
            if (findProjectedPodMapEntry(
                    columns, block,
                    gl::gpu::DevicePodMapKind::knownStatements,
                    packed) < 0) {
                states[requestIndex] = state;
                return;
            }
        }
        atomicAdd(&counters->dependencyPassCount, 1u);

        gl::NameId consensus = block.mainValidityId;
        int32_t maximumIteration = -1;
        bool pure = true;
        for (gl::NameId premise = 0; premise < event.count; ++premise) {
            const gl::IntEncodedExpr& expression = columns.statements[
                block.statementOffset + static_cast<uint32_t>(
                    event.statementIndices[premise])];
            if (!projectedValiditiesComparable(
                    columns, block, consensus, expression.validityId)) {
                states[requestIndex] = state;
                return;
            }
            consensus = projectedDeeperValidity(
                columns, block, consensus, expression.validityId);
            if (maximumIteration < expression.maxIteration)
                maximumIteration = expression.maxIteration;
            for (gl::NameId argument = 0;
                 argument < expression.arity; ++argument) {
                if (expression.argIteration[argument] > -1
                    && findProjectedPodMapEntry(
                        columns, block,
                        gl::gpu::DevicePodMapKind::recursionProducts,
                        expression.argFullId[argument]) < 0) pure = false;
            }
        }
        if (projectedNameContains(
                columns, block, consensus, "_hypo_", 6)) {
            for (gl::NameId premise = 0;
                 premise < event.count; ++premise) {
                const gl::IntEncodedExpr& expression = columns.statements[
                    block.statementOffset + static_cast<uint32_t>(
                        event.statementIndices[premise])];
                if (expression.validityId != consensus
                    && !expression.isAnchor) {
                    states[requestIndex] = state;
                    return;
                }
            }
        }
        for (gl::NameId validity = consensus;
             validity != 0;) {
            if (findProjectedPodMapEntry(
                    columns, block,
                    gl::gpu::DevicePodMapKind::validityFilter,
                    validity) >= 0) {
                states[requestIndex] = state;
                return;
            }
            const uint32_t nameIndex = block.nameRecordOffset
                + static_cast<uint32_t>(validity);
            assert(nameIndex < block.nameRecordOffset + block.nameRecordCount);
            validity = columns.nameRecords[nameIndex].parentId;
        }
        const gl::gpu::DeviceByteMapView& encodedView = columns.byteMapViews[
            block.byteMapViewOffset + static_cast<uint32_t>(
                gl::gpu::DeviceByteMapKind::overallEncoded)];
        if (encodedView.entryCount == 0) {
            states[requestIndex] = state;
            return;
        }

        gl::NameId normalized[gl::ExecutionParameters::MAX_KEY_SLOTS];
        const gl::NameId length = buildProjectedNormalizedKey(
            columns, block, event.statementIndices, event.count, normalized);
        gl::NameId serialized[gl::ExecutionParameters::MAX_KEY_SLOTS + 2];
        serialized[0] = event.count;
        serialized[1] = length;
        for (gl::NameId index = 0; index < length; ++index)
            serialized[index + 2] = normalized[index];
        const int32_t reverseEntry = findProjectedReverseMapEntry(
            columns, block, reinterpret_cast<const char*>(serialized),
            static_cast<uint32_t>(length + 2) * sizeof(gl::NameId));
        uint32_t ownerCount = 0;
        if (reverseEntry >= 0) {
            ownerCount = columns.reverseMapEntries[
                static_cast<uint32_t>(reverseEntry)].ownerCount;
        }
        state.reverseEntryIndex = reverseEntry;
        state.reverseOwnerCount = ownerCount;
        state.consensusValidityId = consensus;
        state.maximumPremiseIteration = maximumIteration;
        state.pure = pure ? 1u : 0u;
        state.active = 1;
        states[requestIndex] = state;
        ownerCounts[requestIndex] = ownerCount;
        atomicAdd(&counters->reverseOwnerCount, ownerCount);
    }

    /// @brief Expand request reverse-index runs into flat owner work.
    ///
    /// @param columns Resident reverse entries and owners.
    /// @param states Per-request reverse entry state.
    /// @param offsets Exclusive owner offsets by request.
    /// @param requestCount Number of request states.
    /// @param output Flat owner work output.
    /// @return Nothing.
    /// @invariant Offsets were scanned from each state's exact owner count.
    __global__ void phase2EvaluationEmitOwnersKernel(
        ProjectionSemanticColumns columns,
        const gl::gpu::DeviceEvaluationRequestState* states,
        const uint32_t* offsets,
        uint32_t requestCount,
        gl::gpu::DeviceEvaluationOwnerWork* output) {
        const uint32_t requestIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if (requestIndex >= requestCount) return;
        const gl::gpu::DeviceEvaluationRequestState& state =
            states[requestIndex];
        if (state.reverseOwnerCount == 0) return;
        assert(state.reverseEntryIndex >= 0);
        const gl::gpu::DeviceReverseMapEntry& reverseEntry =
            columns.reverseMapEntries[
                static_cast<uint32_t>(state.reverseEntryIndex)];
        assert(reverseEntry.ownerCount == state.reverseOwnerCount);
        const gl::gpu::DeviceLogicalBlockProjection& block =
            columns.logicalBlocks[state.logicalBlockIndex];
        const gl::gpu::DeviceByteMapView& remainingView = columns.byteMapViews[
            block.byteMapViewOffset + static_cast<uint32_t>(
                gl::gpu::DeviceByteMapKind::overallRemainingArgs)];
        for (uint32_t owner = 0; owner < reverseEntry.ownerCount; ++owner) {
            const int32_t localEntryId = columns.reverseOwners[
                reverseEntry.ownerOffset + owner];
            assert(localEntryId > 0);
            assert(static_cast<uint32_t>(localEntryId)
                <= remainingView.entryCount);
            output[offsets[requestIndex] + owner] =
                gl::gpu::DeviceEvaluationOwnerWork{
                    requestIndex,
                    remainingView.entryOffset
                        + static_cast<uint32_t>(localEntryId - 1) };
        }
    }

    /// @brief Apply remaining-argument subset and mapped encoded-key lookup.
    ///
    /// @param columns Resident statements and byte maps.
    /// @param tasks Uploaded task descriptors.
    /// @param calls Uploaded growth calls.
    /// @param events Persistent request events.
    /// @param tokens Exact ordered request tokens.
    /// @param owners Flat reverse-owner work.
    /// @param ownerCount Number of owner items.
    /// @param candidates Compact subset-surviving output.
    /// @param candidateCapacity Fixed output ceiling.
    /// @param candidateCount Shared compact output counter.
    /// @return Nothing.
    /// @invariant Every owner entry belongs to its request's logical block.
    __global__ void phase2EvaluationCandidateKernel(
        ProjectionSemanticColumns columns,
        const gl::gpu::DevicePhase2Task* tasks,
        const gl::gpu::DevicePhase2GrowthCall* calls,
        const gl::gpu::DeviceAcceptedGrowthEvent* events,
        const gl::gpu::DeviceOrderedRequestToken* tokens,
        const gl::gpu::DeviceEvaluationOwnerWork* owners,
        uint32_t ownerCount,
        gl::gpu::DeviceEvaluationCandidate* candidates,
        uint32_t candidateCapacity,
        uint32_t* candidateCount) {
        const uint32_t ownerIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if (ownerIndex >= ownerCount) return;
        const gl::gpu::DeviceEvaluationOwnerWork& work = owners[ownerIndex];
        const gl::gpu::DeviceAcceptedGrowthEvent& event =
            events[tokens[work.requestIndex].eventIndex];
        const gl::gpu::DevicePhase2GrowthCall& call = calls[event.callIndex];
        const gl::gpu::DevicePhase2Task& task = tasks[call.taskIndex];
        const gl::gpu::DeviceLogicalBlockProjection& block =
            columns.logicalBlocks[task.logicalBlockIndex];
        const gl::gpu::DeviceByteMapEntry& remainingEntry =
            columns.byteMapEntries[work.remainingEntryIndex];
        gl::gpu::DeviceEvaluationCandidate candidate{};
        candidate.requestIndex = work.requestIndex;
        candidate.remainingEntryIndex = work.remainingEntryIndex;
        bool subset = true;
        const char* const key = columns.byteKeyBytes + remainingEntry.keyOffset;
        const gl::NameId remainingCount = readProjectedNameId(key);
        assert(remainingCount >= 0);
        for (gl::NameId remaining = 0;
             remaining < remainingCount && subset; ++remaining) {
            const gl::NameId required = readProjectedNameId(
                key + static_cast<uint32_t>(remaining + 1)
                    * sizeof(gl::NameId));
            bool found = false;
            for (gl::NameId premise = 0;
                 premise < event.count && !found; ++premise) {
                const gl::IntEncodedExpr& expression = columns.statements[
                    block.statementOffset + static_cast<uint32_t>(
                        event.statementIndices[premise])];
                for (gl::NameId argument = 0;
                     argument < expression.arity; ++argument) {
                    if (expression.argFullId[argument] == required) {
                        found = true;
                        break;
                    }
                }
            }
            if (!found) subset = false;
        }
        if (subset) {
            gl::NameId normalized[gl::ExecutionParameters::MAX_KEY_SLOTS];
            const gl::NameId length = buildProjectedEvaluationKey(
                columns, block, event, remainingEntry, normalized);
            gl::NameId serialized[gl::ExecutionParameters::MAX_KEY_SLOTS + 2];
            serialized[0] = event.count;
            serialized[1] = length;
            for (gl::NameId index = 0; index < length; ++index)
                serialized[index + 2] = normalized[index];
            candidate.encodedEntryIndex = findProjectedByteMapEntry(
                columns, block,
                gl::gpu::DeviceByteMapKind::overallEncoded,
                reinterpret_cast<const char*>(serialized),
                static_cast<uint32_t>(length + 2) * sizeof(gl::NameId));
        }
        if (subset) {
            const uint32_t outputIndex = atomicAdd(candidateCount, 1u);
            assert(outputIndex < candidateCapacity);
            candidates[outputIndex] = candidate;
        }
    }

    /// @brief Compact subset candidates that hit the encoded map.
    ///
    /// @param candidates Subset-surviving candidates.
    /// @param count Number of candidates.
    /// @param hits Compact encoded-hit output.
    /// @param hitCapacity Fixed output ceiling.
    /// @param hitCount Shared compact output counter.
    /// @return Nothing.
    /// @invariant Every candidate already passed the subset gate.
    __global__ void phase2EvaluationCompactEncodedKernel(
        const gl::gpu::DeviceEvaluationCandidate* candidates,
        uint32_t count,
        gl::gpu::DeviceEvaluationCandidate* hits,
        uint32_t hitCapacity,
        uint32_t* hitCount) {
        const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= count || candidates[index].encodedEntryIndex < 0) return;
        const uint32_t outputIndex = atomicAdd(hitCount, 1u);
        assert(outputIndex < hitCapacity);
        hits[outputIndex] = candidates[index];
    }

    /// @brief Count LocalMemoryValue blobs for each encoded hit.
    ///
    /// @param columns Resident encoded-map entries.
    /// @param hits Compact encoded-hit candidates.
    /// @param count Number of encoded hits.
    /// @param valueCounts Per-hit LMV-run lengths.
    /// @param counters Shared local-value total.
    /// @return Nothing.
    /// @invariant Every hit names an encoded entry with a non-empty value run.
    __global__ void phase2EvaluationValueCountsKernel(
        ProjectionSemanticColumns columns,
        const gl::gpu::DeviceEvaluationCandidate* hits,
        uint32_t count,
        uint32_t* valueCounts,
        DeviceEvaluationCounters* counters) {
        const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= count) return;
        assert(hits[index].encodedEntryIndex >= 0);
        const uint32_t run = columns.byteMapEntries[
            static_cast<uint32_t>(hits[index].encodedEntryIndex)]
                .blobRecordCount;
        assert(run > 0);
        valueCounts[index] = run;
        atomicAdd(&counters->localValueCount, run);
    }

    /// @brief Expand encoded-hit runs into flat LocalMemoryValue work.
    ///
    /// @param columns Resident encoded-map entries.
    /// @param hits Compact encoded-hit candidates.
    /// @param offsets Exclusive value offsets by hit.
    /// @param count Number of encoded hits.
    /// @param output Flat request/blob work output.
    /// @return Nothing.
    /// @invariant Offsets were scanned from exact blob-run lengths.
    __global__ void phase2EvaluationEmitValuesKernel(
        ProjectionSemanticColumns columns,
        const gl::gpu::DeviceEvaluationCandidate* hits,
        const uint32_t* offsets,
        uint32_t count,
        gl::gpu::DeviceEvaluationValueWork* output) {
        const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= count) return;
        const gl::gpu::DeviceEvaluationCandidate& hit = hits[index];
        const gl::gpu::DeviceByteMapEntry& entry = columns.byteMapEntries[
            static_cast<uint32_t>(hit.encodedEntryIndex)];
        for (uint32_t value = 0; value < entry.blobRecordCount; ++value) {
            output[offsets[index] + value] =
                gl::gpu::DeviceEvaluationValueWork{
                    hit.requestIndex, entry.blobRecordOffset + value,
                    hit.remainingEntryIndex };
        }
    }

    /// @brief Materialize substituted expressions and source classification per LMV.
    ///
    /// @param columns Resident projection columns.
    /// @param tasks Uploaded task descriptors.
    /// @param calls Uploaded growth-call schedule.
    /// @param events Persistent request events.
    /// @param tokens Exact ordered request tokens.
    /// @param states Per-request validity and purity state.
    /// @param values Selected LMV work.
    /// @param valueCount Number of selected LMVs.
    /// @param records Fixed firing-header output.
    /// @param recordCapacity Firing-header ceiling.
    /// @param generatedBytes Fixed generated-byte output.
    /// @param generatedByteCapacity Generated-byte ceiling.
    /// @param levelValues Fixed sorted-level output.
    /// @param levelValueCapacity Level-value ceiling.
    /// @param originDependencies Fixed head-provenance output.
    /// @param originDependencyCapacity Provenance-dependency ceiling.
    /// @param markerKeys Fixed transformed marker-key slice output.
    /// @param markerKeyCapacity Marker-key ceiling.
    /// @param markerRemainingArgs Fixed sorted remaining-identifier output.
    /// @param markerRemainingArgCapacity Remaining-identifier ceiling.
    /// @param markerArgs Fixed sorted marker-head argument slice output.
    /// @param markerArgCapacity Marker-head argument ceiling.
    /// @param counters Shared firing and variable-output counters.
    /// @return Nothing.
    /// @invariant Every value was selected by the preceding work expansion for
    ///            this exact request sweep.
    __global__ void phase2FiringExpressionKernel(
        ProjectionSemanticColumns columns,
        const gl::gpu::DevicePhase2Task* tasks,
        const gl::gpu::DevicePhase2GrowthCall* calls,
        const gl::gpu::DeviceAcceptedGrowthEvent* events,
        const gl::gpu::DeviceOrderedRequestToken* tokens,
        const gl::gpu::DeviceEvaluationRequestState* states,
        const gl::gpu::DeviceEvaluationValueWork* values,
        uint32_t valueCount,
        gl::gpu::DevicePhase2FiringRecord* records,
        uint32_t recordCapacity,
        char* generatedBytes,
        uint32_t generatedByteCapacity,
        int32_t* levelValues,
        uint32_t levelValueCapacity,
        gl::gpu::DeviceEvaluationDependency* originDependencies,
        uint32_t originDependencyCapacity,
        gl::gpu::DeviceEvaluationByteSlice* markerKeys,
        uint32_t markerKeyCapacity,
        int32_t* markerRemainingArgs,
        uint32_t markerRemainingArgCapacity,
        gl::gpu::DeviceEvaluationByteSlice* markerArgs,
        uint32_t markerArgCapacity,
        DeviceEvaluationCounters* counters) {
        const uint32_t valueIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if (valueIndex >= valueCount) return;
        const gl::gpu::DeviceEvaluationValueWork& work = values[valueIndex];
        const gl::gpu::DeviceEvaluationRequestState& state =
            states[work.requestIndex];
        assert(state.active != 0);
        const gl::gpu::DeviceOrderedRequestToken& token =
            tokens[work.requestIndex];
        const gl::gpu::DeviceAcceptedGrowthEvent& event =
            events[token.eventIndex];
        const gl::gpu::DevicePhase2GrowthCall& call = calls[event.callIndex];
        const gl::gpu::DevicePhase2Task& task = tasks[call.taskIndex];
        assert(task.logicalBlockIndex == state.logicalBlockIndex);
        const gl::gpu::DeviceLogicalBlockProjection& block =
            columns.logicalBlocks[state.logicalBlockIndex];

        const gl::gpu::DeviceBlobRecord& blob =
            columns.blobRecords[work.blobRecordIndex];
        const char* const lmv = columns.blobBytes + blob.byteOffset;
        validateProjectedLmv(lmv, blob.byteLength);
        const gl::NameId lmvValidity = static_cast<gl::NameId>(
            readProjectedI32(lmv, blob.byteLength, 12));
        if (!projectedValiditiesComparable(
                columns, block, state.consensusValidityId, lmvValidity)) return;
        const gl::NameId hitValidity = projectedDeeperValidity(
            columns, block, state.consensusValidityId, lmvValidity);
        if (hitValidity != state.consensusValidityId) {
            for (gl::NameId validity = hitValidity; validity != 0;) {
                if (findProjectedPodMapEntry(
                        columns, block,
                        gl::gpu::DevicePodMapKind::validityFilter,
                        validity) >= 0) return;
                const uint32_t nameIndex = block.nameRecordOffset
                    + static_cast<uint32_t>(validity);
                assert(nameIndex < block.nameRecordOffset + block.nameRecordCount);
                validity = columns.nameRecords[nameIndex].parentId;
            }
        }

        uint32_t flags = 0;
        if (static_cast<uint8_t>(lmv[16]) != 0)
            flags |= gl::gpu::deviceFiringMarker;
        if (static_cast<uint8_t>(lmv[17]) != 0)
            flags |= gl::gpu::deviceFiringProductOfDisintegration;
        if (static_cast<uint8_t>(lmv[18]) != 0)
            flags |= gl::gpu::deviceFiringDisintegrationAllowed;
        if (static_cast<uint8_t>(lmv[19]) != 0)
            flags |= gl::gpu::deviceFiringOrdisOnly;
        if (static_cast<uint8_t>(lmv[20]) != 0)
            flags |= gl::gpu::deviceFiringSubsetExclusion;
        if (static_cast<uint8_t>(lmv[21]) != 0)
            flags |= gl::gpu::deviceFiringOrdis2Demand;
        const bool markerOrDemand = (flags
            & (gl::gpu::deviceFiringMarker
                | gl::gpu::deviceFiringOrdis2Demand)) != 0;
        const bool inOrBranch = projectedNameContains(
            columns, block, hitValidity, "_boundary_orint_", 16);
        if (markerOrDemand && state.pure == 0 && !inOrBranch) return;

        const gl::gpu::DeviceByteMapEntry& remainingEntry =
            columns.byteMapEntries[work.remainingEntryIndex];
        gl::NameId reverseIds[gl::ExecutionParameters::MAX_KEY_SLOTS];
        const gl::NameId reverseCount = buildProjectedEvaluationReverseMap(
            columns, block, event, remainingEntry, reverseIds);
        const int32_t valueId = readProjectedI32(lmv, blob.byteLength, 0);
        assert(valueId > 0);
        assert(static_cast<uint32_t>(valueId) < block.ruleStringRecordCount);
        const gl::gpu::DeviceRuleStringRecord& sourceRecord =
            columns.ruleStringRecords[
                block.ruleStringRecordOffset + static_cast<uint32_t>(valueId)];
        assert(sourceRecord.byteLength > 0);
        const char* const source =
            columns.ruleStringBytes + sourceRecord.byteOffset;
        int32_t maximumIteration = -1;
        const bool demand = (flags
            & gl::gpu::deviceFiringOrdis2Demand) != 0;
        const uint32_t expressionLength = transformProjectedFiringExpression(
            columns, block, source, sourceRecord.byteLength,
            reverseIds, reverseCount, nullptr,
            demand ? &maximumIteration : nullptr);
        assert(expressionLength > 0);
        if (demand
            && maximumIteration > task.maximumIterationNumberVariable) return;

        const bool marker = !demand
            && (flags & gl::gpu::deviceFiringMarker) != 0;
        const bool head = !demand && !marker;
        uint32_t levelsOffset = 0;
        uint32_t levelsCount = 0;
        if (!marker) {
            levelsCount = mergeProjectedFiringLevels(
                columns, block, event, lmv, blob.byteLength,
                head, nullptr);
            if (levelsCount > 0) {
                levelsOffset = atomicAdd(
                    &counters->levelValueCount, levelsCount);
                assert(levelsOffset <= levelValueCapacity);
                assert(levelsCount <= levelValueCapacity - levelsOffset);
                const uint32_t writtenLevels = mergeProjectedFiringLevels(
                    columns, block, event, lmv, blob.byteLength,
                    head, levelValues + levelsOffset);
                assert(writtenLevels == levelsCount);
            }
        }
        uint32_t originOffset = 0;
        uint32_t originCount = 0;
        if (head) {
            originCount = static_cast<uint32_t>(event.count) + 1;
            originOffset = atomicAdd(
                &counters->originDependencyCount, originCount);
            assert(originOffset <= originDependencyCapacity);
            assert(originCount <= originDependencyCapacity - originOffset);
            const uint32_t writtenOrigins =
                writeProjectedHeadOriginDependencies(
                    columns, block, event,
                    readProjectedI32(lmv, blob.byteLength, 4),
                    lmvValidity, originDependencies + originOffset);
            assert(writtenOrigins == originCount);
        }

        const uint32_t byteOffset = atomicAdd(
            &counters->generatedByteCount, expressionLength);
        assert(byteOffset <= generatedByteCapacity);
        assert(expressionLength <= generatedByteCapacity - byteOffset);
        const uint32_t written = transformProjectedFiringExpression(
            columns, block, source, sourceRecord.byteLength,
            reverseIds, reverseCount, generatedBytes + byteOffset, nullptr);
        assert(written == expressionLength);

        uint32_t markerKeyOffset = 0;
        uint32_t markerKeyCount = 0;
        uint32_t markerRemainingOffset = 0;
        uint32_t markerRemainingCount = 0;
        uint32_t markerArgOffset = 0;
        uint32_t markerArgCount = 0;
        if (head) {
            bool doNotDisintegrate = (flags
                & gl::gpu::deviceFiringDisintegrationAllowed) == 0;
            const bool incubatorMode = (block.evaluationFlags
                & gl::gpu::deviceEvaluationIncubatorMode) != 0;
            const bool banDisintegration = (block.evaluationFlags
                & gl::gpu::deviceEvaluationBanDisintegration) != 0;
            if (incubatorMode && !banDisintegration && !doNotDisintegrate) {
                assert(block.anchorNameCount > 0);
                const char* const blockKey =
                    columns.metadataBytes + block.metadataOffset;
                const char* const anchorName =
                    columns.metadataBytes + block.anchorNameOffset;
                const bool anchorBlock = block.metadataCount
                        > block.anchorNameCount
                    && blockKey[0] == '('
                    && projectionBytesEqual(
                        blockKey + 1, anchorName, block.anchorNameCount);
                if (anchorBlock) {
                    doNotDisintegrate = true;
                }
                else {
                    bool hasLocalPremise = false;
                    for (gl::NameId premise = 0;
                         premise < event.count; ++premise) {
                        const gl::IntEncodedExpr& expression = columns.statements[
                            block.statementOffset + static_cast<uint32_t>(
                                event.statementIndices[premise])];
                        const int64_t packed = static_cast<int64_t>(
                            (static_cast<uint64_t>(static_cast<uint32_t>(
                                expression.originalId)) << 32)
                            | static_cast<uint32_t>(expression.validityId));
                        if (findProjectedPodMapEntry(
                                columns, block,
                                gl::gpu::DevicePodMapKind::localStatements,
                                packed) >= 0) {
                            hasLocalPremise = true;
                            break;
                        }
                    }
                    if (!hasLocalPremise) doNotDisintegrate = true;
                }
            }
            if (doNotDisintegrate)
                flags |= gl::gpu::deviceFiringDoNotDisintegrate;
            if ((flags & gl::gpu::deviceFiringProductOfDisintegration) != 0
                && (flags & gl::gpu::deviceFiringSubsetExclusion) == 0) {
                flags |= gl::gpu::deviceFiringAllowOrDisintegration;
            }
            bool allGood = hitValidity == block.mainValidityId;
            for (gl::NameId premise = 0;
                 premise < event.count; ++premise) {
                const gl::IntEncodedExpr& expression = columns.statements[
                    block.statementOffset + static_cast<uint32_t>(
                        event.statementIndices[premise])];
                if (!projectedAllowedForMail(
                        columns, block, expression.originalId)) allGood = false;
            }
            if (allGood) flags |= gl::gpu::deviceFiringAllGood;
            const bool compressorMode = (block.evaluationFlags
                & gl::gpu::deviceEvaluationCompressorMode) != 0;
            if (!compressorMode && projectedAncestorKnown(
                    columns, block, generatedBytes + byteOffset,
                    expressionLength, hitValidity)) {
                flags |= gl::gpu::deviceFiringAlreadyKnown;
            }
        }
        else if (marker) {
            assert(sourceRecord.compiledCategory
                != gl::gpu::DeviceCompiledCategory::absent);
            if (sourceRecord.compiledCategory
                != gl::gpu::DeviceCompiledCategory::atomic) {
                flags |= gl::gpu::deviceFiringMarkerNotAtomic;
            }

            const int32_t levelCountSigned = readProjectedI32(
                lmv, blob.byteLength, 22);
            assert(levelCountSigned >= 0);
            const uint32_t keyCountOffset = 26u
                + static_cast<uint32_t>(levelCountSigned) * sizeof(int32_t);
            const int32_t keyCountSigned = readProjectedI32(
                lmv, blob.byteLength, keyCountOffset);
            assert(keyCountSigned >= 0);
            markerKeyCount = static_cast<uint32_t>(keyCountSigned);
            if (markerKeyCount > 0) {
                markerKeyOffset = atomicAdd(
                    &counters->markerKeyCount, markerKeyCount);
                assert(markerKeyOffset <= markerKeyCapacity);
                assert(markerKeyCount <= markerKeyCapacity - markerKeyOffset);
                for (uint32_t keyIndex = 0;
                     keyIndex < markerKeyCount; ++keyIndex) {
                    const int32_t keyId = readProjectedI32(
                        lmv, blob.byteLength,
                        keyCountOffset + sizeof(int32_t)
                            + keyIndex * sizeof(int32_t));
                    assert(keyId > 0);
                    assert(static_cast<uint32_t>(keyId)
                        < block.ruleStringRecordCount);
                    const gl::gpu::DeviceRuleStringRecord& keyRecord =
                        columns.ruleStringRecords[
                            block.ruleStringRecordOffset
                                + static_cast<uint32_t>(keyId)];
                    const char* const keySource =
                        columns.ruleStringBytes + keyRecord.byteOffset;
                    const uint32_t keyLength = transformProjectedFiringExpression(
                        columns, block, keySource, keyRecord.byteLength,
                        reverseIds, reverseCount, nullptr, nullptr);
                    const uint32_t keyByteOffset = atomicAdd(
                        &counters->generatedByteCount, keyLength);
                    assert(keyByteOffset <= generatedByteCapacity);
                    assert(keyLength
                        <= generatedByteCapacity - keyByteOffset);
                    const uint32_t keyWritten =
                        transformProjectedFiringExpression(
                            columns, block, keySource, keyRecord.byteLength,
                            reverseIds, reverseCount,
                            generatedBytes + keyByteOffset, nullptr);
                    assert(keyWritten == keyLength);
                    markerKeys[markerKeyOffset + keyIndex] =
                        gl::gpu::DeviceEvaluationByteSlice{
                            keyByteOffset, keyLength };
                }
            }

            const uint32_t remainingCountOffset = keyCountOffset
                + sizeof(int32_t)
                + markerKeyCount * sizeof(int32_t);
            const int32_t remainingRawCountSigned = readProjectedI32(
                lmv, blob.byteLength, remainingCountOffset);
            assert(remainingRawCountSigned >= 0);
            const uint32_t remainingRawCount =
                static_cast<uint32_t>(remainingRawCountSigned);
            const auto remainingIdAt = [&](uint32_t index) -> int32_t {
                assert(index < remainingRawCount);
                const int32_t id = readProjectedI32(
                    lmv, blob.byteLength,
                    remainingCountOffset + sizeof(int32_t)
                        + index * sizeof(int32_t));
                assert(id > 0);
                assert(static_cast<uint32_t>(id)
                    < block.ruleStringRecordCount);
                return id;
            };
            const auto compareRuleIds = [&](int32_t left, int32_t right) {
                const gl::gpu::DeviceRuleStringRecord& a =
                    columns.ruleStringRecords[
                        block.ruleStringRecordOffset
                            + static_cast<uint32_t>(left)];
                const gl::gpu::DeviceRuleStringRecord& b =
                    columns.ruleStringRecords[
                        block.ruleStringRecordOffset
                            + static_cast<uint32_t>(right)];
                return projectionBytesCompare(
                    columns.ruleStringBytes + a.byteOffset, a.byteLength,
                    columns.ruleStringBytes + b.byteOffset, b.byteLength);
            };
            for (uint32_t index = 0; index < remainingRawCount; ++index) {
                const int32_t id = remainingIdAt(index);
                bool duplicate = false;
                for (uint32_t prior = 0; prior < index; ++prior) {
                    if (compareRuleIds(id, remainingIdAt(prior)) == 0) {
                        duplicate = true;
                        break;
                    }
                }
                if (!duplicate) ++markerRemainingCount;
            }
            if (markerRemainingCount > 0) {
                markerRemainingOffset = atomicAdd(
                    &counters->markerRemainingArgCount,
                    markerRemainingCount);
                assert(markerRemainingOffset <= markerRemainingArgCapacity);
                assert(markerRemainingCount
                    <= markerRemainingArgCapacity - markerRemainingOffset);
                int32_t previous = 0;
                bool havePrevious = false;
                for (uint32_t outputIndex = 0;
                     outputIndex < markerRemainingCount; ++outputIndex) {
                    int32_t best = 0;
                    bool haveBest = false;
                    for (uint32_t inputIndex = 0;
                         inputIndex < remainingRawCount; ++inputIndex) {
                        const int32_t candidate = remainingIdAt(inputIndex);
                        if (havePrevious
                            && compareRuleIds(candidate, previous) <= 0) continue;
                        if (!haveBest
                            || compareRuleIds(candidate, best) < 0) {
                            best = candidate;
                            haveBest = true;
                        }
                    }
                    assert(haveBest);
                    markerRemainingArgs[
                        markerRemainingOffset + outputIndex] = best;
                    previous = best;
                    havePrevious = true;
                }
            }

            gl::gpu::DeviceEvaluationByteSlice bareArgs[
                gl::ExecutionParameters::MAX_ARITY];
            uint32_t bareCount = 0;
            bool sawMarker = false;
            uint32_t open = expressionLength;
            for (uint32_t index = 0; index < expressionLength; ++index) {
                if (generatedBytes[byteOffset + index] == '[') {
                    open = index;
                    break;
                }
            }
            uint32_t close = expressionLength;
            if (open < expressionLength) {
                for (uint32_t index = open; index < expressionLength; ++index) {
                    if (generatedBytes[byteOffset + index] == ']') {
                        close = index;
                        break;
                    }
                }
            }
            if (open < expressionLength && close < expressionLength
                && close > open + 1) {
                uint32_t argumentStart = open + 1;
                for (uint32_t index = argumentStart; index <= close; ++index) {
                    if (index != close
                        && generatedBytes[byteOffset + index] != ',') continue;
                    const uint32_t argumentLength = index - argumentStart;
                    const char* const argument = generatedBytes
                        + byteOffset + argumentStart;
                    if (argumentLength == 6
                        && projectionBytesEqual(argument, "marker", 6)) {
                        sawMarker = true;
                    }
                    else {
                        assert(bareCount
                            < gl::ExecutionParameters::MAX_ARITY);
                        bareArgs[bareCount++] =
                            gl::gpu::DeviceEvaluationByteSlice{
                                byteOffset + argumentStart, argumentLength };
                    }
                    argumentStart = index + 1;
                }
            }
            assert(sawMarker);
            for (uint32_t index = 1; index < bareCount; ++index) {
                const gl::gpu::DeviceEvaluationByteSlice value = bareArgs[index];
                uint32_t position = index;
                while (position > 0) {
                    const gl::gpu::DeviceEvaluationByteSlice& previousSlice =
                        bareArgs[position - 1];
                    if (projectionBytesCompare(
                            generatedBytes + value.offset, value.length,
                            generatedBytes + previousSlice.offset,
                            previousSlice.length) >= 0) break;
                    bareArgs[position] = previousSlice;
                    --position;
                }
                bareArgs[position] = value;
            }
            for (uint32_t index = 0; index < bareCount; ++index) {
                if (index == 0 || projectionBytesCompare(
                        generatedBytes + bareArgs[index].offset,
                        bareArgs[index].length,
                        generatedBytes + bareArgs[index - 1].offset,
                        bareArgs[index - 1].length) != 0) {
                    bareArgs[markerArgCount++] = bareArgs[index];
                }
            }
            if (markerArgCount > 0) {
                markerArgOffset = atomicAdd(
                    &counters->markerArgCount, markerArgCount);
                assert(markerArgOffset <= markerArgCapacity);
                assert(markerArgCount <= markerArgCapacity - markerArgOffset);
                for (uint32_t index = 0; index < markerArgCount; ++index)
                    markerArgs[markerArgOffset + index] = bareArgs[index];
            }
        }
        const uint32_t recordIndex = atomicAdd(
            &counters->firingRecordCount, 1u);
        assert(recordIndex < recordCapacity);
        gl::gpu::DevicePhase2FiringRecord record{};
        record.growthPosition = token.growthPosition;
        record.requestIndex = work.requestIndex;
        record.logicalBlockIndex = task.logicalBlockIndex;
        record.partOrdinal = static_cast<uint32_t>(task.stumpOrdinal);
        record.blobRecordIndex = work.blobRecordIndex;
        record.expression = gl::gpu::DeviceEvaluationByteSlice{
            byteOffset, expressionLength };
        record.levelsOffset = levelsOffset;
        record.levelsCount = levelsCount;
        record.originDependencyOffset = originOffset;
        record.originDependencyCount = originCount;
        record.markerKeyOffset = markerKeyOffset;
        record.markerKeyCount = markerKeyCount;
        record.markerRemainingArgOffset = markerRemainingOffset;
        record.markerRemainingArgCount = markerRemainingCount;
        record.markerArgOffset = markerArgOffset;
        record.markerArgCount = markerArgCount;
        record.validityId = hitValidity;
        record.iteration = markerOrDemand
            ? -1 : state.maximumPremiseIteration + 1;
        record.demandSourceImplId = demand
            ? readProjectedI32(lmv, blob.byteLength, 4) : 0;
        record.standardMaxAdmissionDepth = marker
            ? task.maximumIterationNumberVariable : 0;
        record.standardMaxSecondaryNumber = marker
            ? block.standardMaxSecondaryNumber : 0;
        record.flags = flags;
        records[recordIndex] = record;
    }

    /// @brief Compare two complete device firing records in canonical content order.
    ///
    /// @details
    /// Groups independent local identifier namespaces by logical-block index,
    /// then reproduces `applyFiringRecords`' head, marker, and ordis2-demand
    /// comparator over projected decoded bytes and fixed output runs. Fields the
    /// processor comparator deliberately omits remain omitted here as well.
    /// Content-identical records are finally ordered by source record index so
    /// the merge permutation is deterministic without changing deposit bytes.
    ///
    /// @param columns Resident name and rule-interner projection columns.
    /// @param records Complete firing headers.
    /// @param generatedBytes Generated expression, key, and argument bytes.
    /// @param levelValues Sorted-unique firing level arena.
    /// @param originDependencies Ordered head provenance arena.
    /// @param markerKeys Marker admission-key slice arena.
    /// @param markerRemainingArgs Sorted marker remaining-identifier arena.
    /// @param markerArgs Sorted marker bare-argument slice arena.
    /// @param leftIndex First firing-record index.
    /// @param rightIndex Second firing-record index.
    /// @return Negative, zero, or positive for canonical less, equal, or greater.
    /// @invariant Every slice and identifier belongs to its record's projected
    ///            logical block and addresses the last materialized used prefix.
    __device__ int compareDeviceFiringRecords(
        ProjectionSemanticColumns columns,
        const gl::gpu::DevicePhase2FiringRecord* records,
        const char* generatedBytes,
        const int32_t* levelValues,
        const gl::gpu::DeviceEvaluationDependency* originDependencies,
        const gl::gpu::DeviceEvaluationByteSlice* markerKeys,
        const int32_t* markerRemainingArgs,
        const gl::gpu::DeviceEvaluationByteSlice* markerArgs,
        uint32_t leftIndex,
        uint32_t rightIndex) {
        const gl::gpu::DevicePhase2FiringRecord& left = records[leftIndex];
        const gl::gpu::DevicePhase2FiringRecord& right = records[rightIndex];
        if (left.logicalBlockIndex != right.logicalBlockIndex) {
            return left.logicalBlockIndex < right.logicalBlockIndex ? -1 : 1;
        }
        const gl::gpu::DeviceLogicalBlockProjection& block =
            columns.logicalBlocks[left.logicalBlockIndex];
        const auto compareSlices = [&](const gl::gpu::DeviceEvaluationByteSlice& a,
                                       const gl::gpu::DeviceEvaluationByteSlice& b) {
            return projectionBytesCompare(
                generatedBytes + a.offset, a.length,
                generatedBytes + b.offset, b.length);
        };
        const auto compareNameIds = [&](gl::NameId a, gl::NameId b) {
            assert(a > 0 && b > 0);
            const uint32_t ai = block.nameRecordOffset + static_cast<uint32_t>(a);
            const uint32_t bi = block.nameRecordOffset + static_cast<uint32_t>(b);
            assert(ai < block.nameRecordOffset + block.nameRecordCount);
            assert(bi < block.nameRecordOffset + block.nameRecordCount);
            const gl::gpu::DeviceNameRecord& ar = columns.nameRecords[ai];
            const gl::gpu::DeviceNameRecord& br = columns.nameRecords[bi];
            return projectionBytesCompare(
                columns.nameBytes + ar.byteOffset, ar.byteLength,
                columns.nameBytes + br.byteOffset, br.byteLength);
        };
        const auto compareRuleIds = [&](int32_t a, int32_t b) {
            assert(a > 0 && b > 0);
            assert(static_cast<uint32_t>(a) < block.ruleStringRecordCount);
            assert(static_cast<uint32_t>(b) < block.ruleStringRecordCount);
            const gl::gpu::DeviceRuleStringRecord& ar =
                columns.ruleStringRecords[
                    block.ruleStringRecordOffset + static_cast<uint32_t>(a)];
            const gl::gpu::DeviceRuleStringRecord& br =
                columns.ruleStringRecords[
                    block.ruleStringRecordOffset + static_cast<uint32_t>(b)];
            return projectionBytesCompare(
                columns.ruleStringBytes + ar.byteOffset, ar.byteLength,
                columns.ruleStringBytes + br.byteOffset, br.byteLength);
        };
        const auto compareLevels = [&](uint32_t ao, uint32_t an,
                                       uint32_t bo, uint32_t bn) {
            const uint32_t common = an < bn ? an : bn;
            for (uint32_t index = 0; index < common; ++index) {
                const int32_t av = levelValues[ao + index];
                const int32_t bv = levelValues[bo + index];
                if (av != bv) return av < bv ? -1 : 1;
            }
            if (an != bn) return an < bn ? -1 : 1;
            return 0;
        };
        const auto compareSliceRuns = [&](const gl::gpu::DeviceEvaluationByteSlice* values,
                                          uint32_t ao, uint32_t an,
                                          uint32_t bo, uint32_t bn) {
            const uint32_t common = an < bn ? an : bn;
            for (uint32_t index = 0; index < common; ++index) {
                const int value = compareSlices(
                    values[ao + index], values[bo + index]);
                if (value != 0) return value;
            }
            if (an != bn) return an < bn ? -1 : 1;
            return 0;
        };

        int comparison = compareSlices(left.expression, right.expression);
        if (comparison != 0) return comparison;
        comparison = compareNameIds(left.validityId, right.validityId);
        if (comparison != 0) return comparison;
        const bool leftMarker = (left.flags & gl::gpu::deviceFiringMarker) != 0;
        const bool rightMarker = (right.flags & gl::gpu::deviceFiringMarker) != 0;
        if (leftMarker != rightMarker) return leftMarker ? 1 : -1;
        const bool leftDemand =
            (left.flags & gl::gpu::deviceFiringOrdis2Demand) != 0;
        const bool rightDemand =
            (right.flags & gl::gpu::deviceFiringOrdis2Demand) != 0;
        if (leftDemand != rightDemand) return leftDemand ? 1 : -1;
        if (leftDemand) {
            comparison = compareLevels(
                left.levelsOffset, left.levelsCount,
                right.levelsOffset, right.levelsCount);
            if (comparison != 0) return comparison;
            comparison = compareRuleIds(
                left.demandSourceImplId, right.demandSourceImplId);
            if (comparison != 0) return comparison;
        } else if (!leftMarker) {
            if (left.originDependencyCount != right.originDependencyCount) {
                return left.originDependencyCount < right.originDependencyCount
                    ? -1 : 1;
            }
            for (uint32_t index = 0;
                 index < left.originDependencyCount; ++index) {
                const gl::gpu::DeviceEvaluationDependency& a =
                    originDependencies[left.originDependencyOffset + index];
                const gl::gpu::DeviceEvaluationDependency& b =
                    originDependencies[right.originDependencyOffset + index];
                comparison = index == 0
                    ? compareRuleIds(a.originalId, b.originalId)
                    : compareNameIds(a.originalId, b.originalId);
                if (comparison != 0) return comparison;
                comparison = compareNameIds(a.validityId, b.validityId);
                if (comparison != 0) return comparison;
            }
            comparison = compareLevels(
                left.levelsOffset, left.levelsCount,
                right.levelsOffset, right.levelsCount);
            if (comparison != 0) return comparison;
            constexpr uint32_t headBits[] = {
                gl::gpu::deviceFiringDoNotDisintegrate,
                gl::gpu::deviceFiringAllowOrDisintegration,
                gl::gpu::deviceFiringAllGood,
                gl::gpu::deviceFiringAlreadyKnown
            };
            for (uint32_t bit : headBits) {
                const bool av = (left.flags & bit) != 0;
                const bool bv = (right.flags & bit) != 0;
                if (av != bv) return av ? 1 : -1;
            }
        } else {
            comparison = compareSliceRuns(
                markerKeys,
                left.markerKeyOffset, left.markerKeyCount,
                right.markerKeyOffset, right.markerKeyCount);
            if (comparison != 0) return comparison;
            comparison = compareSliceRuns(
                markerArgs,
                left.markerArgOffset, left.markerArgCount,
                right.markerArgOffset, right.markerArgCount);
            if (comparison != 0) return comparison;
            const uint32_t common = left.markerRemainingArgCount
                < right.markerRemainingArgCount
                ? left.markerRemainingArgCount : right.markerRemainingArgCount;
            for (uint32_t index = 0; index < common; ++index) {
                comparison = compareRuleIds(
                    markerRemainingArgs[left.markerRemainingArgOffset + index],
                    markerRemainingArgs[right.markerRemainingArgOffset + index]);
                if (comparison != 0) return comparison;
            }
            if (left.markerRemainingArgCount != right.markerRemainingArgCount) {
                return left.markerRemainingArgCount < right.markerRemainingArgCount
                    ? -1 : 1;
            }
            const bool av =
                (left.flags & gl::gpu::deviceFiringMarkerNotAtomic) != 0;
            const bool bv =
                (right.flags & gl::gpu::deviceFiringMarkerNotAtomic) != 0;
            if (av != bv) return av ? 1 : -1;
        }
        return leftIndex < rightIndex ? -1 : (leftIndex == rightIndex ? 0 : 1);
    }

    /// @brief Initialize one firing-order index per materialized record.
    ///
    /// @param output Fixed firing-order index array.
    /// @param count Materialized firing-record count.
    /// @return Nothing.
    /// @invariant `output` has capacity for `count` indices.
    __global__ void initializeFiringOrderKernel(
        uint32_t* output, uint32_t count) {
        const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < count) output[index] = index;
    }

    /// @brief Perform one parallel stable merge pass over firing-record indices.
    ///
    /// @details
    /// Each output rank finds its merge-path partition by binary search, then
    /// selects the lesser next record through the complete canonical comparator.
    /// The record-index tie break makes every comparison strict and deterministic.
    ///
    /// @param columns Resident name and rule-interner projection columns.
    /// @param records Complete firing headers.
    /// @param generatedBytes Generated expression, key, and argument bytes.
    /// @param levelValues Sorted firing levels.
    /// @param originDependencies Ordered head provenance.
    /// @param markerKeys Marker key slices.
    /// @param markerRemainingArgs Marker remaining identifiers.
    /// @param markerArgs Marker bare-argument slices.
    /// @param input Sorted runs from the preceding pass.
    /// @param output Merged runs for this pass.
    /// @param count Materialized firing-record count.
    /// @param width Length of each input run.
    /// @return Nothing.
    /// @invariant `width` is a power of two and both index arrays cover `count`.
    __global__ void mergeFiringOrderKernel(
        ProjectionSemanticColumns columns,
        const gl::gpu::DevicePhase2FiringRecord* records,
        const char* generatedBytes,
        const int32_t* levelValues,
        const gl::gpu::DeviceEvaluationDependency* originDependencies,
        const gl::gpu::DeviceEvaluationByteSlice* markerKeys,
        const int32_t* markerRemainingArgs,
        const gl::gpu::DeviceEvaluationByteSlice* markerArgs,
        const uint32_t* input,
        uint32_t* output,
        uint32_t count,
        uint32_t width) {
        const uint32_t outputIndex = blockIdx.x * blockDim.x + threadIdx.x;
        if (outputIndex >= count) return;
        const uint64_t pairWidth = static_cast<uint64_t>(width) * 2ull;
        const uint32_t pairStart = static_cast<uint32_t>(
            (static_cast<uint64_t>(outputIndex) / pairWidth) * pairWidth);
        const uint32_t aCount = pairStart < count
            ? ((count - pairStart) < width ? count - pairStart : width) : 0;
        const uint32_t bStart = pairStart + aCount;
        const uint32_t bCount = bStart < count
            ? ((count - bStart) < width ? count - bStart : width) : 0;
        const uint32_t rank = outputIndex - pairStart;
        uint32_t low = rank > bCount ? rank - bCount : 0;
        uint32_t high = rank < aCount ? rank : aCount;
        while (low < high) {
            const uint32_t aTaken = low + (high - low) / 2;
            const uint32_t bTaken = rank - aTaken;
            if (aTaken < aCount && bTaken > 0
                && compareDeviceFiringRecords(
                    columns, records, generatedBytes, levelValues,
                    originDependencies, markerKeys, markerRemainingArgs,
                    markerArgs, input[pairStart + aTaken],
                    input[bStart + bTaken - 1]) < 0) {
                low = aTaken + 1;
            } else {
                high = aTaken;
            }
        }
        const uint32_t aTaken = low;
        const uint32_t bTaken = rank - aTaken;
        if (aTaken < aCount && (bTaken >= bCount
            || compareDeviceFiringRecords(
                columns, records, generatedBytes, levelValues,
                originDependencies, markerKeys, markerRemainingArgs,
                markerArgs, input[pairStart + aTaken],
                input[bStart + bTaken]) < 0)) {
            output[outputIndex] = input[pairStart + aTaken];
        } else {
            assert(bTaken < bCount);
            output[outputIndex] = input[bStart + bTaken];
        }
    }

    /// @brief Probe the projected name interner for a firing head's negation.
    ///
    /// @details
    /// Reproduces `ExpressionAnalyzer::negate` as a virtual byte span: a leading
    /// exclamation mark is removed, otherwise one is prepended. The virtual bytes
    /// are hashed and compared directly against the fixed projected name table, so
    /// contradiction detection needs no temporary string or device allocation.
    ///
    /// @param columns Resident name projection columns.
    /// @param block Owning logical-block descriptor.
    /// @param expression Generated firing-head bytes.
    /// @param length Generated firing-head byte count.
    /// @return Arena-global name-record index, or `-1` for a defined miss.
    /// @invariant The projected name slot table uses the same FNV-1a hash and has
    ///            load at most one half.
    __device__ int32_t findProjectedNegatedNameRecord(
        const ProjectionSemanticColumns& columns,
        const gl::gpu::DeviceLogicalBlockProjection& block,
        const char* expression,
        uint32_t length) {
        const bool removeLeading = length > 0 && expression[0] == '!';
        const uint32_t negatedLength = removeLeading ? length - 1 : length + 1;
        if (block.nameSlotCount == 0) {
            assert(block.nameRecordCount == 1);
            return -1;
        }
        assert((block.nameSlotCount & (block.nameSlotCount - 1)) == 0);
        uint64_t hash = 14695981039346656037ull;
        for (uint32_t index = 0; index < negatedLength; ++index) {
            const char value = removeLeading
                ? expression[index + 1]
                : (index == 0 ? '!' : expression[index - 1]);
            hash ^= static_cast<uint8_t>(value);
            hash *= 1099511628211ull;
        }
        uint32_t slot = static_cast<uint32_t>(hash)
            & (block.nameSlotCount - 1);
        while (true) {
            const int32_t recordIndex = columns.nameSlots[
                block.nameSlotOffset + slot];
            if (recordIndex == -1) return -1;
            assert(recordIndex >= 0);
            const uint32_t globalIndex = static_cast<uint32_t>(recordIndex);
            assert(globalIndex > block.nameRecordOffset);
            assert(globalIndex < block.nameRecordOffset + block.nameRecordCount);
            const gl::gpu::DeviceNameRecord& record =
                columns.nameRecords[globalIndex];
            bool equal = record.byteLength == negatedLength;
            for (uint32_t index = 0; equal && index < negatedLength; ++index) {
                const char expected = removeLeading
                    ? expression[index + 1]
                    : (index == 0 ? '!' : expression[index - 1]);
                equal = columns.nameBytes[record.byteOffset + index] == expected;
            }
            if (equal) return recordIndex;
            slot = (slot + 1) & (block.nameSlotCount - 1);
        }
    }

    /// @brief Mirror the processor doom predicate for one device firing record.
    ///
    /// @details
    /// Markers and ordis2 demands never doom a block. A main-scope head dooms it
    /// when it closes the sole goal outside compressor mode, or when its negation
    /// is already known and the block is primed, recursive, or a counter-example
    /// filter. All probes use the immutable projected goal and known-statement
    /// tables and perform no minting.
    ///
    /// @param columns Resident semantic projection columns.
    /// @param block Owning logical-block descriptor.
    /// @param record Complete device firing header.
    /// @param generatedBytes Generated expression arena.
    /// @return True exactly when `burstDeactivates` would publish a doom line.
    /// @invariant Reads only the last uploaded projection and materialized record.
    __device__ bool deviceFiringDeactivates(
        const ProjectionSemanticColumns& columns,
        const gl::gpu::DeviceLogicalBlockProjection& block,
        const gl::gpu::DevicePhase2FiringRecord& record,
        const char* generatedBytes) {
        if ((record.flags & (gl::gpu::deviceFiringMarker
            | gl::gpu::deviceFiringOrdis2Demand)) != 0) return false;
        const bool atMain = record.validityId == block.mainValidityId;
        if (!atMain) return false;
        const char* const expression =
            generatedBytes + record.expression.offset;
        const uint32_t length = record.expression.length;

        const uint32_t goalsOrdinal = static_cast<uint32_t>(
            gl::gpu::DevicePodMapKind::goals);
        assert(goalsOrdinal < block.podMapViewCount);
        const gl::gpu::DevicePodMapView& goals = columns.podMapViews[
            block.podMapViewOffset + goalsOrdinal];
        const bool compressorMode = (block.evaluationFlags
            & gl::gpu::deviceEvaluationCompressorMode) != 0;
        if (!compressorMode && goals.entryCount == 1) {
            const int32_t globalHead = findProjectedNameRecord(
                columns, block, expression, length);
            if (globalHead >= 0) {
                const uint32_t globalIndex = static_cast<uint32_t>(globalHead);
                assert(globalIndex > block.nameRecordOffset);
                const gl::NameId headId = static_cast<gl::NameId>(
                    globalIndex - block.nameRecordOffset);
                const int64_t key = static_cast<int64_t>(
                    (static_cast<uint64_t>(static_cast<uint32_t>(headId)) << 32)
                    | static_cast<uint32_t>(record.validityId));
                if (findProjectedPodMapEntry(
                        columns, block, gl::gpu::DevicePodMapKind::goals,
                        key) >= 0) return true;
            }
        }

        const bool roleCandidate = block.primedForContradiction != 0
            || block.isPartOfRecursion != 0
            || block.contradictionIndex >= 0;
        if (!roleCandidate) return false;
        const int32_t globalNegation = findProjectedNegatedNameRecord(
            columns, block, expression, length);
        if (globalNegation < 0) return false;
        const uint32_t globalIndex = static_cast<uint32_t>(globalNegation);
        assert(globalIndex > block.nameRecordOffset);
        const gl::NameId negationId = static_cast<gl::NameId>(
            globalIndex - block.nameRecordOffset);
        const int64_t key = static_cast<int64_t>(
            (static_cast<uint64_t>(static_cast<uint32_t>(negationId)) << 32)
            | static_cast<uint32_t>(block.mainValidityId));
        return findProjectedPodMapEntry(
            columns, block, gl::gpu::DevicePodMapKind::knownStatements,
            key) >= 0;
    }

    /// @brief Initialize every logical block to the no-doom sentinel.
    ///
    /// @param doomLines Fixed per-block signed 64-bit output.
    /// @param doomRequestIndices Fixed per-block triggering-request output.
    /// @param count Uploaded logical-block count.
    /// @return Nothing.
    /// @invariant Both output arrays have capacity for `count` values.
    __global__ void initializeDoomLinesKernel(
        int64_t* doomLines,
        uint32_t* doomRequestIndices,
        uint32_t count) {
        const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < count) {
            doomLines[index] = static_cast<int64_t>(0x7fffffffffffffffll);
            doomRequestIndices[index] = 0xffffffffu;
        }
    }

    /// @brief Reduce exact device doom triggers to one packed line per block.
    ///
    /// @param columns Resident semantic projection columns.
    /// @param records Complete materialized firing headers.
    /// @param generatedBytes Generated expression arena.
    /// @param recordCount Materialized firing-record count.
    /// @param logicalBlockCount Uploaded logical-block count.
    /// @param doomLines Fixed per-block CAS-min destinations.
    /// @return Nothing.
    /// @invariant Growth positions fit the processor's 48-bit doom position and
    ///            part ordinals fit its 14-bit ordinal field.
    __global__ void detectDoomLinesKernel(
        ProjectionSemanticColumns columns,
        const gl::gpu::DevicePhase2FiringRecord* records,
        const char* generatedBytes,
        uint32_t recordCount,
        uint32_t logicalBlockCount,
        int64_t* doomLines) {
        const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= recordCount) return;
        const gl::gpu::DevicePhase2FiringRecord& record = records[index];
        assert(record.logicalBlockIndex < logicalBlockCount);
        const gl::gpu::DeviceLogicalBlockProjection& block =
            columns.logicalBlocks[record.logicalBlockIndex];
        if (!deviceFiringDeactivates(
                columns, block, record, generatedBytes)) return;
        assert(record.growthPosition < (uint64_t(1) << 48));
        assert(record.partOrdinal < (uint32_t(1) << 14));
        const uint64_t packed = (record.growthPosition << 14)
            | static_cast<uint64_t>(record.partOrdinal);
        atomicMin(
            reinterpret_cast<unsigned long long*>(
                doomLines + record.logicalBlockIndex),
            static_cast<unsigned long long>(packed));
    }

    /// @brief Reduce the first triggering request at each final packed doom line.
    ///
    /// @details
    /// Runs only after packed-line reduction in the same stream. Rechecking the
    /// pure deactivation predicate isolates records whose position and part equal
    /// the final winning line, then atomic-min selects their earliest exact
    /// ordered-request token. This distinguishes requests sharing one growth
    /// position without widening the processor-visible packed doom-line format.
    ///
    /// @param columns Resident semantic projection columns.
    /// @param records Complete materialized firing headers.
    /// @param generatedBytes Generated expression arena.
    /// @param recordCount Materialized firing-record count.
    /// @param logicalBlockCount Uploaded logical-block count.
    /// @param doomLines Final packed doom line per block.
    /// @param doomRequestIndices Fixed per-block request-index destinations.
    /// @return Nothing.
    /// @invariant Every non-sentinel doom line has at least one triggering head.
    __global__ void detectDoomRequestIndicesKernel(
        ProjectionSemanticColumns columns,
        const gl::gpu::DevicePhase2FiringRecord* records,
        const char* generatedBytes,
        uint32_t recordCount,
        uint32_t logicalBlockCount,
        const int64_t* doomLines,
        uint32_t* doomRequestIndices) {
        const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= recordCount) return;
        const gl::gpu::DevicePhase2FiringRecord& record = records[index];
        assert(record.logicalBlockIndex < logicalBlockCount);
        const uint64_t packed = (record.growthPosition << 14)
            | static_cast<uint64_t>(record.partOrdinal);
        if (packed != static_cast<uint64_t>(
                doomLines[record.logicalBlockIndex])) return;
        const gl::gpu::DeviceLogicalBlockProjection& block =
            columns.logicalBlocks[record.logicalBlockIndex];
        if (!deviceFiringDeactivates(
                columns, block, record, generatedBytes)) return;
        atomicMin(doomRequestIndices + record.logicalBlockIndex,
            record.requestIndex);
    }

    /// @brief Mark canonical firing indices retained by final doom-prefix policy.
    ///
    /// @param records Complete materialized firing headers.
    /// @param canonicalOrder Canonical firing-record permutation.
    /// @param doomLines Final packed doom line per logical block.
    /// @param doomRequestIndices First triggering request at each final line.
    /// @param count Canonical index count.
    /// @param keepFlags One integer selection flag per canonical index.
    /// @return Nothing.
    /// @invariant A doomed block retains only its winning part through the doom
    ///            growth position; the sentinel retains every part.
    __global__ void markDoomPrefixKernel(
        const gl::gpu::DevicePhase2FiringRecord* records,
        const uint32_t* canonicalOrder,
        const int64_t* doomLines,
        const uint32_t* doomRequestIndices,
        uint32_t count,
        uint32_t* keepFlags) {
        const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= count) return;
        const gl::gpu::DevicePhase2FiringRecord& record =
            records[canonicalOrder[index]];
        const int64_t line = doomLines[record.logicalBlockIndex];
        bool keep = line == static_cast<int64_t>(0x7fffffffffffffffll);
        if (!keep) {
            const uint64_t packed = static_cast<uint64_t>(line);
            const uint64_t position = packed >> 14;
            const uint32_t ordinal = static_cast<uint32_t>(
                packed & ((uint64_t(1) << 14) - 1));
            keep = record.partOrdinal == ordinal
                && (record.growthPosition < position
                    || (record.growthPosition == position
                        && record.requestIndex
                            <= doomRequestIndices[record.logicalBlockIndex]));
        }
        keepFlags[index] = keep ? 1u : 0u;
    }

    /// @brief Scatter retained canonical indices through an exclusive-scan prefix.
    ///
    /// @param input Canonical firing-record permutation.
    /// @param keepFlags One zero-or-one flag per input index.
    /// @param offsets Exclusive selection offsets.
    /// @param count Input index count.
    /// @param output Stable compacted firing-record permutation.
    /// @return Nothing.
    /// @invariant `offsets` is the exclusive sum of `keepFlags`.
    __global__ void scatterDoomPrefixKernel(
        const uint32_t* input,
        const uint32_t* keepFlags,
        const uint32_t* offsets,
        uint32_t count,
        uint32_t* output) {
        const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index < count && keepFlags[index] != 0)
            output[offsets[index]] = input[index];
    }

    /// @brief Finish a stable firing-index compaction count on the device.
    ///
    /// @param keepFlags One zero-or-one flag per input index.
    /// @param offsets Exclusive selection offsets.
    /// @param count Nonzero input index count.
    /// @param selectedCount One-word output.
    /// @return Nothing.
    /// @invariant The launch geometry is one block containing one thread.
    __global__ void finishDoomPrefixCountKernel(
        const uint32_t* keepFlags,
        const uint32_t* offsets,
        uint32_t count,
        uint32_t* selectedCount) {
        assert(blockIdx.x == 0 && blockDim.x == 1 && threadIdx.x == 0);
        assert(count > 0);
        *selectedCount = offsets[count - 1] + keepFlags[count - 1];
    }

    /// @brief Transform one deterministic word on the CUDA device.
    ///
    /// @details
    /// Exactly one thread writes the result. The deliberately tiny kernel is a
    /// startup contract, not a performance kernel: it proves that the linked
    /// fat binary contains launchable native code and that the runtime can write
    /// device memory before Phase 2 owns persistent arenas.
    ///
    /// @param input Input word supplied by the host contract probe.
    /// @param output One-word device destination allocated by the caller.
    /// @return Nothing; writes the transformed word to `output[0]`.
    /// @invariant The launch geometry is one block containing one thread.
    __global__ void cudaContractProbeKernel(uint32_t input, uint32_t* output) {
        assert(blockIdx.x == 0 && blockDim.x == 1 && threadIdx.x == 0);
        output[0] = input * 1664525u + 1013904223u;
    }

    /// @brief Compute one deterministic checksum over uploaded projection arrays.
    ///
    /// @details
    /// Exactly one thread consumes raw bytes in semantic array order. The kernel
    /// observes only the used prefixes supplied by the host, so untouched capacity
    /// suffixes cannot affect the result. FNV-1a provides a compact byte-exact twin
    /// for transfer validation; it is not a semantic hash-table primitive.
    ///
    /// @param columns Device addresses and byte lengths of all semantic columns.
    /// @param checksum One-word device destination for the FNV-1a result.
    /// @return Nothing; writes `checksum[0]`.
    /// @invariant The launch geometry is one block containing one thread.
    __global__ void projectionChecksumKernel(
        ProjectionChecksumColumns columns, uint64_t* checksum) {
        assert(blockIdx.x == 0 && blockDim.x == 1 && threadIdx.x == 0);
        uint64_t value = 14695981039346656037ull;
        constexpr uint64_t prime = 1099511628211ull;

        for (uint32_t column = 0; column < kProjectionColumnCount; ++column) {
            for (uint64_t index = 0; index < columns.bytes[column]; ++index) {
                value ^= static_cast<uint64_t>(columns.data[column][index]);
                value *= prime;
            }
        }
        checksum[0] = value;
    }

}  // namespace

namespace gl::gpu {

    /// @brief Query and validate the active CUDA device for the Phase 2 backend.
    ///
    /// @details
    /// Enumerates CUDA devices, selects the runtime's active device, reads its
    /// immutable properties, and asserts the minimum compute capability required
    /// by the native `sm_89` build. Missing devices, runtime failures, unsupported
    /// capability, or nonsensical sizing are contract violations and never fall
    /// back to the processor backend.
    ///
    /// @return Validated pointer-free properties for the active CUDA device.
    /// @invariant Successful return means at least compute capability 8.9 and
    ///            strictly positive memory, multiprocessor, and block limits.
    /// @see launchCudaContractProbe
    CudaDeviceContract queryCudaDeviceContract() {
        int deviceCount = 0;
        GL_CUDA_ASSERT(cudaGetDeviceCount(&deviceCount));
        assert(deviceCount > 0);

        int deviceOrdinal = -1;
        GL_CUDA_ASSERT(cudaGetDevice(&deviceOrdinal));
        assert(deviceOrdinal >= 0 && deviceOrdinal < deviceCount);

        cudaDeviceProp properties{};
        GL_CUDA_ASSERT(cudaGetDeviceProperties(&properties, deviceOrdinal));
        assert(properties.major > 8
            || (properties.major == 8 && properties.minor >= 9));
        assert(properties.totalGlobalMem > 0);
        assert(properties.multiProcessorCount > 0);
        assert(properties.maxThreadsPerBlock > 0);

        CudaDeviceContract contract{};
        contract.deviceOrdinal = deviceOrdinal;
        contract.computeMajor = properties.major;
        contract.computeMinor = properties.minor;
        contract.totalGlobalBytes =
            static_cast<uint64_t>(properties.totalGlobalMem);
        contract.multiprocessorCount = properties.multiProcessorCount;
        contract.maximumThreadsPerBlock = properties.maxThreadsPerBlock;
        return contract;
    }

    /// @brief Launch a real native CUDA kernel and return its deterministic word.
    ///
    /// @details
    /// Allocates one device word, launches the module's native contract kernel,
    /// synchronizes, copies the word back, and frees the device allocation. Every
    /// CUDA operation asserts success. This is the executable build/runtime seam
    /// used by the unit-test gate before larger Phase 2 arenas are introduced.
    ///
    /// @param input Deterministic input word transformed by the device kernel.
    /// @return `input * 1664525 + 1013904223` with unsigned 32-bit wraparound.
    /// @invariant A return proves that native device code launched, synchronized,
    ///            and transferred data successfully in the current process.
    /// @see queryCudaDeviceContract
    uint32_t launchCudaContractProbe(uint32_t input) {
        static_cast<void>(queryCudaDeviceContract());

        uint32_t* deviceOutput = nullptr;
        GL_CUDA_ASSERT(cudaMalloc(
            reinterpret_cast<void**>(&deviceOutput), sizeof(uint32_t)));
        assert(deviceOutput != nullptr);

        cudaContractProbeKernel<<<1, 1>>>(input, deviceOutput);
        GL_CUDA_ASSERT(cudaGetLastError());
        GL_CUDA_ASSERT(cudaDeviceSynchronize());

        uint32_t hostOutput = 0;
        GL_CUDA_ASSERT(cudaMemcpy(
            &hostOutput, deviceOutput, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        GL_CUDA_ASSERT(cudaFree(deviceOutput));
        return hostOutput;
    }

    /// @brief Create both reusable CUDA timing events.
    ///
    /// @details
    /// Allocates the runtime event pair once. The events retain timing support
    /// and are reused by every selected CUDA Phase 2 pass in the process.
    ///
    /// @invariant Successful construction owns two distinct valid CUDA events.
    CudaPhase2DeviceTimer::CudaPhase2DeviceTimer() {
        cudaEvent_t started = nullptr;
        cudaEvent_t finished = nullptr;
        GL_CUDA_ASSERT(cudaEventCreate(&started));
        GL_CUDA_ASSERT(cudaEventCreate(&finished));
        assert(started != nullptr && finished != nullptr);
        assert(started != finished);
        startedEvent_ = started;
        finishedEvent_ = finished;
    }

    /// @brief Destroy both reusable CUDA timing events.
    ///
    /// @details
    /// Releases the exact pair allocated by the constructor. A runtime failure
    /// asserts because silently leaking a device timing resource violates the
    /// process-owned lifecycle contract.
    ///
    /// @invariant No timing event remains owned after destruction completes.
    CudaPhase2DeviceTimer::~CudaPhase2DeviceTimer() {
        assert(!running_);
        assert(startedEvent_ != nullptr && finishedEvent_ != nullptr);
        GL_CUDA_ASSERT(cudaEventDestroy(
            reinterpret_cast<cudaEvent_t>(startedEvent_)));
        GL_CUDA_ASSERT(cudaEventDestroy(
            reinterpret_cast<cudaEvent_t>(finishedEvent_)));
        startedEvent_ = nullptr;
        finishedEvent_ = nullptr;
    }

    /// @brief Record the beginning of one exact device timeline interval.
    ///
    /// @details
    /// Enqueues the start event on CUDA's default stream. The caller then submits
    /// the uploads and kernels whose device elapsed time is required.
    ///
    /// @return Nothing.
    /// @invariant A matching `stopSeconds` follows before the next `start`.
    void CudaPhase2DeviceTimer::start() {
        assert(!running_);
        assert(startedEvent_ != nullptr && finishedEvent_ != nullptr);
        GL_CUDA_ASSERT(cudaEventRecord(
            reinterpret_cast<cudaEvent_t>(startedEvent_), nullptr));
        running_ = true;
    }

    /// @brief Finish and return the current device timeline interval.
    ///
    /// @details
    /// Records the finish event on the default stream, synchronizes that exact
    /// event, and returns CUDA's elapsed time between the retained pair. Host
    /// projection and sealing outside the two records are excluded.
    ///
    /// @return Non-negative CUDA event time in seconds.
    /// @invariant `start` has recorded the matching beginning event.
    double CudaPhase2DeviceTimer::stopSeconds() {
        assert(running_);
        assert(startedEvent_ != nullptr && finishedEvent_ != nullptr);
        const cudaEvent_t started =
            reinterpret_cast<cudaEvent_t>(startedEvent_);
        const cudaEvent_t finished =
            reinterpret_cast<cudaEvent_t>(finishedEvent_);
        GL_CUDA_ASSERT(cudaEventRecord(finished, nullptr));
        GL_CUDA_ASSERT(cudaEventSynchronize(finished));
        float milliseconds = 0.0F;
        GL_CUDA_ASSERT(cudaEventElapsedTime(
            &milliseconds, started, finished));
        assert(milliseconds >= 0.0F);
        running_ = false;
        return static_cast<double>(milliseconds) / 1000.0;
    }

    /// @brief Allocate all device projection storage once at engine startup.
    ///
    /// @details
    /// Validates the active CUDA device and allocates fixed arrays for logical
    /// blocks plus all 23 subordinate semantic columns and the checksum result.
    /// Every capacity is positive and every CUDA allocation asserts success.
    ///
    /// @param fixedCapacity Immutable device element and byte ceilings.
    /// @return An empty device buffer owning all declared allocations.
    /// @invariant No method changes an allocation address or capacity.
    CudaPhase2ProjectionBuffer::CudaPhase2ProjectionBuffer(
        Phase2ProjectionCapacity fixedCapacity)
        : capacity_(fixedCapacity) {
        static_cast<void>(queryCudaDeviceContract());
        assert(capacity_.logicalBlocks > 0);
        assert(capacity_.statements > 0);
        assert(capacity_.nameRecords > 0);
        assert(capacity_.nameBytes > 0);
        const uint64_t allocationBytes[kProjectionColumnCount] = {
            static_cast<uint64_t>(capacity_.logicalBlocks)
                * sizeof(DeviceLogicalBlockProjection),
            static_cast<uint64_t>(capacity_.statements) * sizeof(IntEncodedExpr),
            static_cast<uint64_t>(capacity_.nameRecords)
                * sizeof(DeviceNameRecord),
            capacity_.nameBytes,
            static_cast<uint64_t>(capacity_.nameSlots) * sizeof(int32_t),
            static_cast<uint64_t>(capacity_.ruleStringRecords)
                * sizeof(DeviceRuleStringRecord),
            capacity_.ruleStringBytes,
            static_cast<uint64_t>(capacity_.byteMapViews)
                * sizeof(DeviceByteMapView),
            static_cast<uint64_t>(capacity_.byteMapEntries)
                * sizeof(DeviceByteMapEntry),
            static_cast<uint64_t>(capacity_.byteMapSlots) * sizeof(int32_t),
            capacity_.byteKeyBytes,
            static_cast<uint64_t>(capacity_.blobRecords)
                * sizeof(DeviceBlobRecord),
            capacity_.blobBytes,
            static_cast<uint64_t>(capacity_.reverseMapViews)
                * sizeof(DeviceReverseMapView),
            static_cast<uint64_t>(capacity_.reverseMapEntries)
                * sizeof(DeviceReverseMapEntry),
            static_cast<uint64_t>(capacity_.reverseMapSlots) * sizeof(int32_t),
            capacity_.reverseKeyBytes,
            static_cast<uint64_t>(capacity_.reverseOwners) * sizeof(int32_t),
            static_cast<uint64_t>(capacity_.podMapViews)
                * sizeof(DevicePodMapView),
            static_cast<uint64_t>(capacity_.podMapEntries)
                * sizeof(DevicePodMapEntry),
            static_cast<uint64_t>(capacity_.podMapSlots) * sizeof(int32_t),
            static_cast<uint64_t>(capacity_.podRunValues) * sizeof(int32_t),
            static_cast<uint64_t>(capacity_.mandatoryStatementKeys)
                * sizeof(int64_t),
            capacity_.metadataBytes
        };
        fixedAllocationBytes_ = sizeof(DeviceGrowthCandidateProbeResult);
        for (uint32_t column = 0; column < kProjectionColumnCount; ++column) {
            assert(allocationBytes[column] > 0);
            assert(allocationBytes[column]
                <= std::numeric_limits<std::size_t>::max());
            GL_CUDA_ASSERT(cudaMalloc(
                &deviceColumns_[column],
                static_cast<std::size_t>(allocationBytes[column])));
            assert(deviceColumns_[column] != nullptr);
            fixedAllocationBytes_ += allocationBytes[column];
        }
        GL_CUDA_ASSERT(cudaMalloc(
            &deviceResultScratch_, sizeof(DeviceGrowthCandidateProbeResult)));
        assert(deviceResultScratch_ != nullptr);
        cudaStream_t deferredUploadStream = nullptr;
        GL_CUDA_ASSERT(cudaStreamCreateWithFlags(
            &deferredUploadStream, cudaStreamNonBlocking));
        assert(deferredUploadStream != nullptr);
        deferredUploadStream_ = reinterpret_cast<void*>(deferredUploadStream);
        for (uint32_t deferredColumn : kDeferredPhase2Columns) {
            assert(allocationBytes[deferredColumn]
                <= std::numeric_limits<std::size_t>::max());
            GL_CUDA_ASSERT(cudaHostAlloc(
                &deferredHostColumns_[deferredColumn],
                static_cast<std::size_t>(allocationBytes[deferredColumn]),
                cudaHostAllocDefault));
            assert(deferredHostColumns_[deferredColumn] != nullptr);
        }
    }

    /// @brief Release every device allocation owned by this projection buffer.
    ///
    /// @details
    /// Frees each startup allocation exactly once and asserts every CUDA free.
    /// A buffer is neither copyable nor movable, so ownership cannot alias.
    ///
    /// @return Nothing.
    CudaPhase2ProjectionBuffer::~CudaPhase2ProjectionBuffer() {
        assert(!phase2UploadPending_);
        for (uint32_t deferredColumn : kDeferredPhase2Columns) {
            assert(deferredHostColumns_[deferredColumn] != nullptr);
            GL_CUDA_ASSERT(cudaFreeHost(
                deferredHostColumns_[deferredColumn]));
        }
        assert(deferredUploadStream_ != nullptr);
        GL_CUDA_ASSERT(cudaStreamDestroy(
            reinterpret_cast<cudaStream_t>(deferredUploadStream_)));
        GL_CUDA_ASSERT(cudaFree(deviceResultScratch_));
        for (uint32_t column = kProjectionColumnCount; column > 0; --column)
            GL_CUDA_ASSERT(cudaFree(deviceColumns_[column - 1]));
    }

    /// @brief Copy one host projection's used prefixes into fixed device arrays.
    ///
    /// @details
    /// Asserts each host used length against the constructor ceiling, then
    /// copies all 24 semantic columns in canonical array order. Empty columns are
    /// defined and issue no zero-byte CUDA call. The method stores used counts
    /// for the next kernel launch.
    ///
    /// @param hostProjection Fixed-capacity host image to upload byte-for-byte.
    /// @return Nothing.
    /// @invariant Device used prefixes equal the supplied host prefixes after
    ///            return; unused suffixes are semantically inaccessible.
    void CudaPhase2ProjectionBuffer::upload(
        const Phase2ProjectionArena& hostProjection) {
        beginPhase2Upload(hostProjection);
        finishPhase2Upload();
    }

    /// @brief Start a dependency-staged Phase 2 projection upload.
    ///
    /// @details
    /// Copies filter, growth, and ordering columns synchronously, then queues
    /// evaluation-only rule-string, reverse-map, run-value, and metadata prefixes
    /// through process-owned page-locked staging arrays on a nonblocking CUDA
    /// stream. Kernels may consume only the early column set until
    /// `finishPhase2Upload` returns.
    ///
    /// @param hostProjection Fixed-capacity host image to upload byte-for-byte.
    /// @return Nothing.
    /// @invariant No earlier staged upload is pending and every queued prefix
    ///            remains immutable until `finishPhase2Upload`.
    void CudaPhase2ProjectionBuffer::beginPhase2Upload(
        const Phase2ProjectionArena& hostProjection) {
        assert(!phase2UploadPending_);
        const void* hostData[kProjectionColumnCount] = {
            hostProjection.logicalBlocks.data(),
            hostProjection.statements.data(),
            hostProjection.nameRecords.data(),
            hostProjection.nameBytes.data(),
            hostProjection.nameSlots.data(),
            hostProjection.ruleStringRecords.data(),
            hostProjection.ruleStringBytes.data(),
            hostProjection.byteMapViews.data(),
            hostProjection.byteMapEntries.data(),
            hostProjection.byteMapSlots.data(),
            hostProjection.byteKeyBytes.data(),
            hostProjection.blobRecords.data(),
            hostProjection.blobBytes.data(),
            hostProjection.reverseMapViews.data(),
            hostProjection.reverseMapEntries.data(),
            hostProjection.reverseMapSlots.data(),
            hostProjection.reverseKeyBytes.data(),
            hostProjection.reverseOwners.data(),
            hostProjection.podMapViews.data(),
            hostProjection.podMapEntries.data(),
            hostProjection.podMapSlots.data(),
            hostProjection.podRunValues.data(),
            hostProjection.mandatoryStatementKeys.data(),
            hostProjection.metadataBytes.data()
        };
        const uint64_t usedCounts[kProjectionColumnCount] = {
            hostProjection.logicalBlocks.size(),
            hostProjection.statements.size(),
            hostProjection.nameRecords.size(),
            hostProjection.nameBytes.size(),
            hostProjection.nameSlots.size(),
            hostProjection.ruleStringRecords.size(),
            hostProjection.ruleStringBytes.size(),
            hostProjection.byteMapViews.size(),
            hostProjection.byteMapEntries.size(),
            hostProjection.byteMapSlots.size(),
            hostProjection.byteKeyBytes.size(),
            hostProjection.blobRecords.size(),
            hostProjection.blobBytes.size(),
            hostProjection.reverseMapViews.size(),
            hostProjection.reverseMapEntries.size(),
            hostProjection.reverseMapSlots.size(),
            hostProjection.reverseKeyBytes.size(),
            hostProjection.reverseOwners.size(),
            hostProjection.podMapViews.size(),
            hostProjection.podMapEntries.size(),
            hostProjection.podMapSlots.size(),
            hostProjection.podRunValues.size(),
            hostProjection.mandatoryStatementKeys.size(),
            hostProjection.metadataBytes.size()
        };
        const uint32_t capacities[kProjectionColumnCount] = {
            capacity_.logicalBlocks, capacity_.statements,
            capacity_.nameRecords, capacity_.nameBytes, capacity_.nameSlots,
            capacity_.ruleStringRecords, capacity_.ruleStringBytes,
            capacity_.byteMapViews, capacity_.byteMapEntries,
            capacity_.byteMapSlots, capacity_.byteKeyBytes,
            capacity_.blobRecords, capacity_.blobBytes,
            capacity_.reverseMapViews, capacity_.reverseMapEntries,
            capacity_.reverseMapSlots, capacity_.reverseKeyBytes,
            capacity_.reverseOwners, capacity_.podMapViews,
            capacity_.podMapEntries, capacity_.podMapSlots,
            capacity_.podRunValues, capacity_.mandatoryStatementKeys,
            capacity_.metadataBytes
        };
        const uint32_t elementBytes[kProjectionColumnCount] = {
            sizeof(DeviceLogicalBlockProjection), sizeof(IntEncodedExpr),
            sizeof(DeviceNameRecord), 1, sizeof(int32_t),
            sizeof(DeviceRuleStringRecord), 1, sizeof(DeviceByteMapView),
            sizeof(DeviceByteMapEntry), sizeof(int32_t), 1,
            sizeof(DeviceBlobRecord), 1, sizeof(DeviceReverseMapView),
            sizeof(DeviceReverseMapEntry), sizeof(int32_t), 1,
            sizeof(int32_t), sizeof(DevicePodMapView),
            sizeof(DevicePodMapEntry), sizeof(int32_t), sizeof(int32_t),
            sizeof(int64_t), 1
        };
        const auto isDeferred = [](uint32_t column) {
            for (uint32_t deferredColumn : kDeferredPhase2Columns)
                if (column == deferredColumn) return true;
            return false;
        };
        for (uint32_t column = 0; column < kProjectionColumnCount; ++column) {
            assert(usedCounts[column] <= capacities[column]);
            assert(usedCounts[column] <= std::numeric_limits<uint32_t>::max());
            usedCounts_[column] = static_cast<uint32_t>(usedCounts[column]);
            if (isDeferred(column)) continue;
            if (usedCounts_[column] == 0) continue;
            const std::size_t bytes = static_cast<std::size_t>(
                usedCounts_[column]) * elementBytes[column];
            GL_CUDA_ASSERT(cudaMemcpy(
                deviceColumns_[column], hostData[column], bytes,
                cudaMemcpyHostToDevice));
        }
        const cudaStream_t deferredUploadStream =
            reinterpret_cast<cudaStream_t>(deferredUploadStream_);
        assert(deferredUploadStream != nullptr);
        for (uint32_t deferredColumn : kDeferredPhase2Columns) {
            if (usedCounts_[deferredColumn] == 0) continue;
            const std::size_t bytes = static_cast<std::size_t>(
                usedCounts_[deferredColumn]) * elementBytes[deferredColumn];
            assert(deferredHostColumns_[deferredColumn] != nullptr);
            std::memcpy(deferredHostColumns_[deferredColumn],
                hostData[deferredColumn], bytes);
            GL_CUDA_ASSERT(cudaMemcpyAsync(
                deviceColumns_[deferredColumn],
                deferredHostColumns_[deferredColumn], bytes,
                cudaMemcpyHostToDevice, deferredUploadStream));
        }
        phase2UploadPending_ = true;
    }

    /// @brief Join the evaluation-only projection upload stream.
    ///
    /// @details
    /// Synchronizes the process-owned nonblocking stream after independent host
    /// schedule construction has had an opportunity to overlap its copies. Every
    /// projection column is device-resident when the method returns, before the
    /// first semantic kernel starts.
    ///
    /// @return Nothing.
    /// @invariant Exactly one `beginPhase2Upload` is pending on entry.
    void CudaPhase2ProjectionBuffer::finishPhase2Upload() {
        assert(phase2UploadPending_);
        const cudaStream_t deferredUploadStream =
            reinterpret_cast<cudaStream_t>(deferredUploadStream_);
        assert(deferredUploadStream != nullptr);
        GL_CUDA_ASSERT(cudaStreamSynchronize(deferredUploadStream));
        phase2UploadPending_ = false;
    }

    /// @brief Hash the uploaded projection bytes on the CUDA device.
    ///
    /// @details
    /// Launches one deterministic verification thread over all 24 uploaded
    /// semantic columns in their fixed canonical order, synchronizes, and
    /// returns the copied 64-bit FNV-1a result. This is a projection transfer
    /// twin, not the eventual parallel request kernel.
    ///
    /// @return Device-computed checksum of all uploaded used prefixes.
    /// @invariant Requires at least one uploaded logical-block descriptor.
    uint64_t CudaPhase2ProjectionBuffer::launchChecksum() const {
        assert(!phase2UploadPending_);
        assert(usedCounts_[0] > 0);
        const uint32_t elementBytes[kProjectionColumnCount] = {
            sizeof(DeviceLogicalBlockProjection), sizeof(IntEncodedExpr),
            sizeof(DeviceNameRecord), 1, sizeof(int32_t),
            sizeof(DeviceRuleStringRecord), 1, sizeof(DeviceByteMapView),
            sizeof(DeviceByteMapEntry), sizeof(int32_t), 1,
            sizeof(DeviceBlobRecord), 1, sizeof(DeviceReverseMapView),
            sizeof(DeviceReverseMapEntry), sizeof(int32_t), 1,
            sizeof(int32_t), sizeof(DevicePodMapView),
            sizeof(DevicePodMapEntry), sizeof(int32_t), sizeof(int32_t),
            sizeof(int64_t), 1
        };
        ProjectionChecksumColumns columns{};
        for (uint32_t column = 0; column < kProjectionColumnCount; ++column) {
            columns.data[column] = static_cast<const unsigned char*>(
                deviceColumns_[column]);
            columns.bytes[column] = static_cast<uint64_t>(usedCounts_[column])
                * elementBytes[column];
        }
        projectionChecksumKernel<<<1, 1>>>(
            columns, static_cast<uint64_t*>(deviceResultScratch_));
        GL_CUDA_ASSERT(cudaGetLastError());
        GL_CUDA_ASSERT(cudaDeviceSynchronize());
        uint64_t hostChecksum = 0;
        GL_CUDA_ASSERT(cudaMemcpy(
            &hostChecksum, deviceResultScratch_, sizeof(hostChecksum),
            cudaMemcpyDeviceToHost));
        return hostChecksum;
    }

    /// @brief Execute one direct CUDA lookup against the uploaded projection.
    ///
    /// @details
    /// Launches a one-thread diagnostic kernel that exercises the exact device
    /// probes used by later parallel request kernels. The operation covers name
    /// interning, all byte-key views, the derived remaining-argument reverse map,
    /// and all plain-data views. A missing key is a defined result with
    /// `recordIndex == -1`; malformed kinds, indices, or lengths assert.
    ///
    /// @param probe Pointer-free lookup family, logical block, and key.
    /// @return Device-computed lookup result copied from fixed result scratch.
    /// @invariant `upload` has supplied the logical block and every selected view
    ///            before this method is called.
    DeviceLookupProbeResult CudaPhase2ProjectionBuffer::launchLookupProbe(
        const DeviceLookupProbe& probe) const {
        assert(usedCounts_[0] > 0);
        assert(probe.logicalBlockIndex < usedCounts_[0]);
        assert(probe.keyLength <= DeviceLookupProbe::kMaximumKeyBytes);

        ProjectionSemanticColumns columns{};
        columns.logicalBlocks = static_cast<
            const DeviceLogicalBlockProjection*>(deviceColumns_[0]);
        columns.statements = static_cast<const IntEncodedExpr*>(deviceColumns_[1]);
        columns.nameRecords = static_cast<const DeviceNameRecord*>(deviceColumns_[2]);
        columns.nameBytes = static_cast<const char*>(deviceColumns_[3]);
        columns.nameSlots = static_cast<const int32_t*>(deviceColumns_[4]);
        columns.ruleStringRecords = static_cast<
            const DeviceRuleStringRecord*>(deviceColumns_[5]);
        columns.ruleStringBytes = static_cast<const char*>(deviceColumns_[6]);
        columns.byteMapViews = static_cast<
            const DeviceByteMapView*>(deviceColumns_[7]);
        columns.byteMapEntries = static_cast<
            const DeviceByteMapEntry*>(deviceColumns_[8]);
        columns.byteMapSlots = static_cast<const int32_t*>(deviceColumns_[9]);
        columns.byteKeyBytes = static_cast<const char*>(deviceColumns_[10]);
        columns.blobRecords = static_cast<
            const DeviceBlobRecord*>(deviceColumns_[11]);
        columns.blobBytes = static_cast<const char*>(deviceColumns_[12]);
        columns.reverseMapViews = static_cast<
            const DeviceReverseMapView*>(deviceColumns_[13]);
        columns.reverseMapEntries = static_cast<
            const DeviceReverseMapEntry*>(deviceColumns_[14]);
        columns.reverseMapSlots = static_cast<const int32_t*>(deviceColumns_[15]);
        columns.reverseKeyBytes = static_cast<const char*>(deviceColumns_[16]);
        columns.reverseOwners = static_cast<const int32_t*>(deviceColumns_[17]);
        columns.podMapViews = static_cast<
            const DevicePodMapView*>(deviceColumns_[18]);
        columns.podMapEntries = static_cast<
            const DevicePodMapEntry*>(deviceColumns_[19]);
        columns.podMapSlots = static_cast<const int32_t*>(deviceColumns_[20]);
        columns.podRunValues = static_cast<const int32_t*>(deviceColumns_[21]);
        columns.mandatoryStatementKeys = static_cast<const int64_t*>(
            deviceColumns_[22]);
        columns.metadataBytes = static_cast<const char*>(deviceColumns_[23]);

        projectionLookupProbeKernel<<<1, 1>>>(
            columns, usedCounts_[0], probe,
            static_cast<DeviceLookupProbeResult*>(deviceResultScratch_));
        GL_CUDA_ASSERT(cudaGetLastError());
        GL_CUDA_ASSERT(cudaDeviceSynchronize());
        DeviceLookupProbeResult result{};
        GL_CUDA_ASSERT(cudaMemcpy(
            &result, deviceResultScratch_, sizeof(result),
            cudaMemcpyDeviceToHost));
        return result;
    }

    /// @brief Report every fixed projection allocation including probe scratch.
    ///
    /// @details
    /// Returns the constructor-captured sum of the 24 immutable projection-column
    /// allocation requests and the reusable growth-probe result allocation. The
    /// method performs host arithmetic only and never queries or mutates CUDA.
    ///
    /// @return Exact startup bytes owned by this projection buffer.
    /// @invariant The value is constant for the object's lifetime.
    uint64_t CudaPhase2ProjectionBuffer::fixedAllocationBytes() const {
        return fixedAllocationBytes_;
    }

    /// @brief Build and probe one complete candidate on the CUDA device.
    ///
    /// @details
    /// Launches the reusable normalized-key and owner-gate diagnostic kernel over
    /// the uploaded projection. The result is copied through the projection
    /// owner's fixed scratch; no device allocation or processor-side semantic
    /// calculation occurs in this method.
    ///
    /// @param probe Logical block, registry, and ordered statement-index path.
    /// @return Device-built key payload and independent lookup verdicts.
    /// @invariant The uploaded projection owns every statement selected by the
    ///            non-empty bounded probe path.
    DeviceGrowthCandidateProbeResult
    CudaPhase2ProjectionBuffer::launchGrowthCandidateProbe(
        const DeviceGrowthCandidateProbe& probe) const {
        assert(usedCounts_[0] > 0);
        assert(probe.logicalBlockIndex < usedCounts_[0]);
        assert(probe.count >= 1);
        assert(probe.count <= ExecutionParameters::MAX_EXPRESSIONS);

        ProjectionSemanticColumns columns{};
        columns.logicalBlocks = static_cast<
            const DeviceLogicalBlockProjection*>(deviceColumns_[0]);
        columns.statements = static_cast<const IntEncodedExpr*>(deviceColumns_[1]);
        columns.nameRecords = static_cast<const DeviceNameRecord*>(deviceColumns_[2]);
        columns.nameBytes = static_cast<const char*>(deviceColumns_[3]);
        columns.nameSlots = static_cast<const int32_t*>(deviceColumns_[4]);
        columns.ruleStringRecords = static_cast<
            const DeviceRuleStringRecord*>(deviceColumns_[5]);
        columns.ruleStringBytes = static_cast<const char*>(deviceColumns_[6]);
        columns.byteMapViews = static_cast<
            const DeviceByteMapView*>(deviceColumns_[7]);
        columns.byteMapEntries = static_cast<
            const DeviceByteMapEntry*>(deviceColumns_[8]);
        columns.byteMapSlots = static_cast<const int32_t*>(deviceColumns_[9]);
        columns.byteKeyBytes = static_cast<const char*>(deviceColumns_[10]);
        columns.blobRecords = static_cast<
            const DeviceBlobRecord*>(deviceColumns_[11]);
        columns.blobBytes = static_cast<const char*>(deviceColumns_[12]);
        columns.reverseMapViews = static_cast<
            const DeviceReverseMapView*>(deviceColumns_[13]);
        columns.reverseMapEntries = static_cast<
            const DeviceReverseMapEntry*>(deviceColumns_[14]);
        columns.reverseMapSlots = static_cast<const int32_t*>(deviceColumns_[15]);
        columns.reverseKeyBytes = static_cast<const char*>(deviceColumns_[16]);
        columns.reverseOwners = static_cast<const int32_t*>(deviceColumns_[17]);
        columns.podMapViews = static_cast<
            const DevicePodMapView*>(deviceColumns_[18]);
        columns.podMapEntries = static_cast<
            const DevicePodMapEntry*>(deviceColumns_[19]);
        columns.podMapSlots = static_cast<const int32_t*>(deviceColumns_[20]);
        columns.podRunValues = static_cast<const int32_t*>(deviceColumns_[21]);
        columns.mandatoryStatementKeys = static_cast<const int64_t*>(
            deviceColumns_[22]);
        columns.metadataBytes = static_cast<const char*>(deviceColumns_[23]);

        growthCandidateProbeKernel<<<1, 1>>>(
            columns, usedCounts_[0], probe,
            static_cast<DeviceGrowthCandidateProbeResult*>(deviceResultScratch_));
        GL_CUDA_ASSERT(cudaGetLastError());
        GL_CUDA_ASSERT(cudaDeviceSynchronize());
        DeviceGrowthCandidateProbeResult result{};
        GL_CUDA_ASSERT(cudaMemcpy(
            &result, deviceResultScratch_, sizeof(result),
            cudaMemcpyDeviceToHost));
        return result;
    }

    /// @brief Allocate all device task storage once at engine startup.
    ///
    /// @details
    /// Validates the active CUDA device, requires positive ceilings for every
    /// pointer-free task column, and allocates tasks, batches, mandatory terms,
    /// stumps, and one checksum word. No later method reallocates these arrays.
    ///
    /// @param fixedCapacity Immutable device task-column element ceilings.
    /// @return An empty device task buffer owning all declared allocations.
    /// @invariant No method changes an allocation address or capacity.
    CudaPhase2TaskBuffer::CudaPhase2TaskBuffer(
        Phase2TaskProjectionCapacity fixedCapacity)
        : capacity_(fixedCapacity) {
        static_cast<void>(queryCudaDeviceContract());
        assert(capacity_.tasks > 0);
        assert(capacity_.batches > 0);
        assert(capacity_.terms > 0);
        assert(capacity_.stumps > 0);
        const uint64_t allocationBytes[kTaskColumnCount] = {
            static_cast<uint64_t>(capacity_.tasks) * sizeof(DevicePhase2Task),
            static_cast<uint64_t>(capacity_.batches)
                * sizeof(DeviceRequestBatch),
            static_cast<uint64_t>(capacity_.terms)
                * sizeof(DeviceMandatoryTerm),
            static_cast<uint64_t>(capacity_.stumps)
                * sizeof(DeviceExpressionStump)
        };
        fixedAllocationBytes_ = sizeof(uint64_t);
        for (uint32_t column = 0; column < kTaskColumnCount; ++column) {
            assert(allocationBytes[column] > 0);
            assert(allocationBytes[column]
                <= std::numeric_limits<std::size_t>::max());
            GL_CUDA_ASSERT(cudaMalloc(
                &deviceColumns_[column],
                static_cast<std::size_t>(allocationBytes[column])));
            assert(deviceColumns_[column] != nullptr);
            fixedAllocationBytes_ += allocationBytes[column];
        }
        GL_CUDA_ASSERT(cudaMalloc(&deviceChecksum_, sizeof(uint64_t)));
        assert(deviceChecksum_ != nullptr);
    }

    /// @brief Release every device allocation owned by this task buffer.
    ///
    /// @details
    /// Frees the checksum word and each fixed task column exactly once. The
    /// disabled copy and move operations make the ownership relationship unique.
    ///
    /// @return Nothing.
    CudaPhase2TaskBuffer::~CudaPhase2TaskBuffer() {
        GL_CUDA_ASSERT(cudaFree(deviceChecksum_));
        for (uint32_t column = kTaskColumnCount; column > 0; --column)
            GL_CUDA_ASSERT(cudaFree(deviceColumns_[column - 1]));
    }

    /// @brief Copy one host task image's used prefixes into fixed device arrays.
    ///
    /// @details
    /// Asserts each host used length against its constructor ceiling, copies the
    /// four columns in canonical order, and records the uploaded used lengths for
    /// the next kernel. Empty subordinate columns issue no zero-byte CUDA call.
    ///
    /// @param hostTasks Fixed-capacity host task image to upload byte-for-byte.
    /// @return Nothing.
    /// @invariant Device used prefixes equal the supplied host prefixes after
    ///            return; unused suffixes are semantically inaccessible.
    void CudaPhase2TaskBuffer::upload(
        const Phase2TaskProjectionArena& hostTasks) {
        const void* hostData[kTaskColumnCount] = {
            hostTasks.tasks.data(), hostTasks.batches.data(),
            hostTasks.terms.data(), hostTasks.stumps.data()
        };
        const uint64_t usedCounts[kTaskColumnCount] = {
            hostTasks.tasks.size(), hostTasks.batches.size(),
            hostTasks.terms.size(), hostTasks.stumps.size()
        };
        const uint32_t capacities[kTaskColumnCount] = {
            capacity_.tasks, capacity_.batches,
            capacity_.terms, capacity_.stumps
        };
        const uint32_t elementBytes[kTaskColumnCount] = {
            sizeof(DevicePhase2Task), sizeof(DeviceRequestBatch),
            sizeof(DeviceMandatoryTerm), sizeof(DeviceExpressionStump)
        };
        for (uint32_t column = 0; column < kTaskColumnCount; ++column) {
            assert(usedCounts[column] <= capacities[column]);
            assert(usedCounts[column] <= std::numeric_limits<uint32_t>::max());
            usedCounts_[column] = static_cast<uint32_t>(usedCounts[column]);
            if (usedCounts_[column] == 0) continue;
            const std::size_t bytes = static_cast<std::size_t>(
                usedCounts_[column]) * elementBytes[column];
            GL_CUDA_ASSERT(cudaMemcpy(
                deviceColumns_[column], hostData[column], bytes,
                cudaMemcpyHostToDevice));
        }
    }

    /// @brief Hash the uploaded task bytes on the CUDA device.
    ///
    /// @details
    /// Launches one deterministic verification thread over the four uploaded
    /// columns in their canonical order. The remaining generic checksum columns
    /// are empty, so the result is exactly the FNV-1a hash of the task image.
    ///
    /// @return Device-computed checksum of every uploaded used prefix.
    /// @invariant Requires at least one uploaded task descriptor.
    uint64_t CudaPhase2TaskBuffer::launchChecksum() const {
        assert(usedCounts_[0] > 0);
        const uint32_t elementBytes[kTaskColumnCount] = {
            sizeof(DevicePhase2Task), sizeof(DeviceRequestBatch),
            sizeof(DeviceMandatoryTerm), sizeof(DeviceExpressionStump)
        };
        ProjectionChecksumColumns columns{};
        for (uint32_t column = 0; column < kTaskColumnCount; ++column) {
            columns.data[column] = static_cast<const unsigned char*>(
                deviceColumns_[column]);
            columns.bytes[column] = static_cast<uint64_t>(usedCounts_[column])
                * elementBytes[column];
        }
        projectionChecksumKernel<<<1, 1>>>(
            columns, static_cast<uint64_t*>(deviceChecksum_));
        GL_CUDA_ASSERT(cudaGetLastError());
        GL_CUDA_ASSERT(cudaDeviceSynchronize());
        uint64_t hostChecksum = 0;
        GL_CUDA_ASSERT(cudaMemcpy(
            &hostChecksum, deviceChecksum_, sizeof(hostChecksum),
            cudaMemcpyDeviceToHost));
        return hostChecksum;
    }

    /// @brief Report every fixed task-column allocation and checksum word.
    ///
    /// @details
    /// Returns the constructor-captured sum of the four immutable task-column
    /// allocation requests and the reusable checksum word. The method performs
    /// no CUDA operation and no ownership can change after construction.
    ///
    /// @return Exact startup bytes owned by this task buffer.
    /// @invariant The value is constant for the object's lifetime.
    uint64_t CudaPhase2TaskBuffer::fixedAllocationBytes() const {
        return fixedAllocationBytes_;
    }

    /// @brief Allocate all global filter/sort device storage once.
    ///
    /// @details
    /// Validates measured capacities, queries CUB exclusive-scan and radix-sort
    /// scratch sizes at their maximum item counts, and allocates original and
    /// class schedules, mappings, both count/offset pairs, two key arrays, and
    /// shared temporary storage.
    ///
    /// @param fixedCapacity Immutable filter/sort ceilings.
    /// @return An empty device filter/sort owner.
    /// @invariant No method changes an allocation address or capacity.
    CudaPhase2FilterSortBuffer::CudaPhase2FilterSortBuffer(
        Phase2FilterScheduleCapacity fixedCapacity)
        : capacity_(fixedCapacity) {
        static_cast<void>(queryCudaDeviceContract());
        assert(capacity_.calls > 0);
        assert(capacity_.examinedRows > 0);
        assert(capacity_.retainedRows > 0);
        assert(capacity_.maximumExaminedRowsPerCall > 0);
        assert(capacity_.calls
            <= static_cast<uint32_t>(std::numeric_limits<int>::max()));
        assert(capacity_.retainedRows
            <= static_cast<uint32_t>(std::numeric_limits<int>::max()));

        uint32_t* nullCounts = nullptr;
        uint32_t* nullOffsets = nullptr;
        uint64_t* nullKeysInput = nullptr;
        uint64_t* nullKeysOutput = nullptr;
        GL_CUDA_ASSERT(cub::DeviceScan::ExclusiveSum(
            nullptr, scanTemporaryBytes_, nullCounts, nullOffsets,
            static_cast<int>(capacity_.calls)));
        GL_CUDA_ASSERT(cub::DeviceRadixSort::SortKeys(
            nullptr, sortTemporaryBytes_, nullKeysInput, nullKeysOutput,
            static_cast<int>(capacity_.retainedRows)));
        assert(scanTemporaryBytes_ > 0);
        assert(sortTemporaryBytes_ > 0);

        GL_CUDA_ASSERT(cudaMalloc(
            &deviceCalls_, static_cast<std::size_t>(capacity_.calls)
                * sizeof(DevicePhase2FilterCall)));
        GL_CUDA_ASSERT(cudaMalloc(
            &deviceClasses_, static_cast<std::size_t>(capacity_.calls)
                * sizeof(DevicePhase2FilterCall)));
        GL_CUDA_ASSERT(cudaMalloc(
            &deviceCallClassIndices_,
            static_cast<std::size_t>(capacity_.calls) * sizeof(uint32_t)));
        GL_CUDA_ASSERT(cudaMalloc(
            &deviceCounts_, static_cast<std::size_t>(capacity_.calls)
                * sizeof(uint32_t)));
        GL_CUDA_ASSERT(cudaMalloc(
            &deviceOffsets_, static_cast<std::size_t>(capacity_.calls)
                * sizeof(uint32_t)));
        GL_CUDA_ASSERT(cudaMalloc(
            &deviceClassCounts_, static_cast<std::size_t>(capacity_.calls)
                * sizeof(uint32_t)));
        GL_CUDA_ASSERT(cudaMalloc(
            &deviceClassOffsets_, static_cast<std::size_t>(capacity_.calls)
                * sizeof(uint32_t)));
        GL_CUDA_ASSERT(cudaMalloc(
            &deviceKeysInput_, static_cast<std::size_t>(capacity_.retainedRows)
                * sizeof(uint64_t)));
        GL_CUDA_ASSERT(cudaMalloc(
            &deviceKeysOutput_, static_cast<std::size_t>(capacity_.retainedRows)
                * sizeof(uint64_t)));
        const std::size_t temporaryBytes = scanTemporaryBytes_
            > sortTemporaryBytes_ ? scanTemporaryBytes_ : sortTemporaryBytes_;
        fixedAllocationBytes_ =
            static_cast<uint64_t>(capacity_.calls)
                * sizeof(DevicePhase2FilterCall) * 2
            + static_cast<uint64_t>(capacity_.calls) * sizeof(uint32_t) * 5
            + static_cast<uint64_t>(capacity_.retainedRows)
                * sizeof(uint64_t) * 2
            + static_cast<uint64_t>(temporaryBytes);
        GL_CUDA_ASSERT(cudaMalloc(&deviceTemporary_, temporaryBytes));
        assert(deviceCalls_ != nullptr);
        assert(deviceClasses_ != nullptr);
        assert(deviceCallClassIndices_ != nullptr);
        assert(deviceCounts_ != nullptr);
        assert(deviceOffsets_ != nullptr);
        assert(deviceClassCounts_ != nullptr);
        assert(deviceClassOffsets_ != nullptr);
        assert(deviceKeysInput_ != nullptr);
        assert(deviceKeysOutput_ != nullptr);
        assert(deviceTemporary_ != nullptr);
    }

    /// @brief Release every fixed filter/sort device allocation.
    ///
    /// @details
    /// Frees the shared CUB scratch, both key arrays, both offset/count pairs,
    /// mapping, and class/original schedules exactly once. Copy and move
    /// operations are disabled.
    ///
    /// @return Nothing.
    CudaPhase2FilterSortBuffer::~CudaPhase2FilterSortBuffer() {
        GL_CUDA_ASSERT(cudaFree(deviceTemporary_));
        GL_CUDA_ASSERT(cudaFree(deviceKeysOutput_));
        GL_CUDA_ASSERT(cudaFree(deviceKeysInput_));
        GL_CUDA_ASSERT(cudaFree(deviceClassOffsets_));
        GL_CUDA_ASSERT(cudaFree(deviceClassCounts_));
        GL_CUDA_ASSERT(cudaFree(deviceOffsets_));
        GL_CUDA_ASSERT(cudaFree(deviceCounts_));
        GL_CUDA_ASSERT(cudaFree(deviceCallClassIndices_));
        GL_CUDA_ASSERT(cudaFree(deviceClasses_));
        GL_CUDA_ASSERT(cudaFree(deviceCalls_));
    }

    /// @brief Filter and sort every scheduled call on the CUDA device.
    ///
    /// @details
    /// Uploads original calls, exact classes, and class indices; counts and scans
    /// classes; maps original calls to shared spans; emits one class key stream;
    /// and radix-sorts that prefix. Empty schedules and retained prefixes are
    /// defined results with no invalid zero-grid or zero-item launch.
    ///
    /// @param projection Uploaded resident semantic projection.
    /// @param schedule Host schedule in processor call order.
    /// @return Total retained statement rows across all calls.
    /// @invariant Sorted keys encode class, decoded-name rank, and ascending
    ///            statement index; every original call maps to its exact span.
    uint32_t CudaPhase2FilterSortBuffer::filterAndSort(
        const CudaPhase2ProjectionBuffer& projection,
        const Phase2FilterScheduleArena& schedule) {
        assert(schedule.calls.size() <= capacity_.calls);
        assert(schedule.classes.size() <= capacity_.calls);
        assert(schedule.callClassIndices.size() == schedule.calls.size());
        assert(schedule.examinedRows <= capacity_.examinedRows);
        assert(schedule.classExaminedRows <= capacity_.examinedRows);
        assert(schedule.capacity.maximumExaminedRowsPerCall
            <= capacity_.maximumExaminedRowsPerCall);
        assert(schedule.capacity.retainedRows <= capacity_.retainedRows);
        assert(schedule.calls.size()
            <= static_cast<std::size_t>(std::numeric_limits<uint32_t>::max()));
        usedCalls_ = static_cast<uint32_t>(schedule.calls.size());
        usedClasses_ = static_cast<uint32_t>(schedule.classes.size());
        usedRetainedRows_ = 0;
        if (usedCalls_ == 0) return 0;
        assert(usedClasses_ > 0);
        assert(projection.usedCounts_[0] > 0);
        GL_CUDA_ASSERT(cudaMemcpy(
            deviceCalls_, schedule.calls.data(),
            static_cast<std::size_t>(usedCalls_)
                * sizeof(DevicePhase2FilterCall),
            cudaMemcpyHostToDevice));
        GL_CUDA_ASSERT(cudaMemcpy(
            deviceClasses_, schedule.classes.data(),
            static_cast<std::size_t>(usedClasses_)
                * sizeof(DevicePhase2FilterCall),
            cudaMemcpyHostToDevice));
        GL_CUDA_ASSERT(cudaMemcpy(
            deviceCallClassIndices_, schedule.callClassIndices.data(),
            static_cast<std::size_t>(usedCalls_) * sizeof(uint32_t),
            cudaMemcpyHostToDevice));

        ProjectionSemanticColumns columns{};
        columns.logicalBlocks = static_cast<
            const DeviceLogicalBlockProjection*>(projection.deviceColumns_[0]);
        columns.statements = static_cast<const IntEncodedExpr*>(
            projection.deviceColumns_[1]);
        columns.nameRecords = static_cast<const DeviceNameRecord*>(
            projection.deviceColumns_[2]);
        columns.nameBytes = static_cast<const char*>(projection.deviceColumns_[3]);
        columns.nameSlots = static_cast<const int32_t*>(projection.deviceColumns_[4]);
        columns.ruleStringRecords = static_cast<const DeviceRuleStringRecord*>(
            projection.deviceColumns_[5]);
        columns.ruleStringBytes = static_cast<const char*>(
            projection.deviceColumns_[6]);
        columns.byteMapViews = static_cast<const DeviceByteMapView*>(
            projection.deviceColumns_[7]);
        columns.byteMapEntries = static_cast<const DeviceByteMapEntry*>(
            projection.deviceColumns_[8]);
        columns.byteMapSlots = static_cast<const int32_t*>(
            projection.deviceColumns_[9]);
        columns.byteKeyBytes = static_cast<const char*>(
            projection.deviceColumns_[10]);
        columns.blobRecords = static_cast<const DeviceBlobRecord*>(
            projection.deviceColumns_[11]);
        columns.blobBytes = static_cast<const char*>(projection.deviceColumns_[12]);
        columns.reverseMapViews = static_cast<const DeviceReverseMapView*>(
            projection.deviceColumns_[13]);
        columns.reverseMapEntries = static_cast<const DeviceReverseMapEntry*>(
            projection.deviceColumns_[14]);
        columns.reverseMapSlots = static_cast<const int32_t*>(
            projection.deviceColumns_[15]);
        columns.reverseKeyBytes = static_cast<const char*>(
            projection.deviceColumns_[16]);
        columns.reverseOwners = static_cast<const int32_t*>(
            projection.deviceColumns_[17]);
        columns.podMapViews = static_cast<const DevicePodMapView*>(
            projection.deviceColumns_[18]);
        columns.podMapEntries = static_cast<const DevicePodMapEntry*>(
            projection.deviceColumns_[19]);
        columns.podMapSlots = static_cast<const int32_t*>(
            projection.deviceColumns_[20]);
        columns.podRunValues = static_cast<const int32_t*>(
            projection.deviceColumns_[21]);
        columns.mandatoryStatementKeys = static_cast<const int64_t*>(
            projection.deviceColumns_[22]);
        columns.metadataBytes = static_cast<const char*>(
            projection.deviceColumns_[23]);

        phase2FilterCountKernel<<<usedClasses_, kFilterThreads>>>(
            columns, projection.usedCounts_[0],
            static_cast<const DevicePhase2FilterCall*>(deviceClasses_),
            usedClasses_, static_cast<uint32_t*>(deviceClassCounts_));
        GL_CUDA_ASSERT(cudaGetLastError());
        GL_CUDA_ASSERT(cub::DeviceScan::ExclusiveSum(
            deviceTemporary_, scanTemporaryBytes_,
            static_cast<const uint32_t*>(deviceClassCounts_),
            static_cast<uint32_t*>(deviceClassOffsets_),
            static_cast<int>(usedClasses_)));

        uint32_t lastCount = 0;
        uint32_t lastOffset = 0;
        GL_CUDA_ASSERT(cudaMemcpy(
            &lastCount,
            static_cast<const uint32_t*>(deviceClassCounts_)
                + usedClasses_ - 1,
            sizeof(lastCount), cudaMemcpyDeviceToHost));
        GL_CUDA_ASSERT(cudaMemcpy(
            &lastOffset,
            static_cast<const uint32_t*>(deviceClassOffsets_)
                + usedClasses_ - 1,
            sizeof(lastOffset), cudaMemcpyDeviceToHost));
        assert(lastOffset <= capacity_.retainedRows);
        assert(lastCount <= capacity_.retainedRows - lastOffset);
        usedRetainedRows_ = lastOffset + lastCount;
        constexpr uint32_t expandThreads = 256;
        phase2FilterExpandClassSpansKernel<<<
            (usedCalls_ + expandThreads - 1) / expandThreads,
            expandThreads>>>(
            static_cast<const uint32_t*>(deviceCallClassIndices_),
            usedCalls_,
            static_cast<const uint32_t*>(deviceClassCounts_),
            static_cast<const uint32_t*>(deviceClassOffsets_),
            usedClasses_, static_cast<uint32_t*>(deviceCounts_),
            static_cast<uint32_t*>(deviceOffsets_));
        GL_CUDA_ASSERT(cudaGetLastError());
        if (usedRetainedRows_ == 0) return 0;

        phase2FilterEmitKernel<<<usedClasses_, kFilterThreads>>>(
            columns, projection.usedCounts_[0],
            static_cast<const DevicePhase2FilterCall*>(deviceClasses_),
            usedClasses_, static_cast<const uint32_t*>(deviceClassCounts_),
            static_cast<const uint32_t*>(deviceClassOffsets_),
            static_cast<uint64_t*>(deviceKeysInput_));
        GL_CUDA_ASSERT(cudaGetLastError());
        GL_CUDA_ASSERT(cub::DeviceRadixSort::SortKeys(
            deviceTemporary_, sortTemporaryBytes_,
            static_cast<const uint64_t*>(deviceKeysInput_),
            static_cast<uint64_t*>(deviceKeysOutput_),
            static_cast<int>(usedRetainedRows_)));
        GL_CUDA_ASSERT(cudaDeviceSynchronize());
        return usedRetainedRows_;
    }

    /// @brief Report every fixed filter allocation including CUB scratch.
    ///
    /// @details
    /// Returns the constructor-captured sum of the call, count, offset, double-key,
    /// and maximum reusable scan-or-sort scratch allocation requests. It performs
    /// no CUDA operation and is unchanged by later filtering calls.
    ///
    /// @return Exact startup bytes owned by this filter buffer.
    /// @invariant The value is constant for the object's lifetime.
    uint64_t CudaPhase2FilterSortBuffer::fixedAllocationBytes() const {
        return fixedAllocationBytes_;
    }

    /// @brief Download the most recent per-call retained counts.
    ///
    /// @details
    /// Copies exactly the last call prefix into caller-owned storage. A zero-call
    /// schedule returns zero without issuing a zero-byte CUDA copy.
    ///
    /// @param destination Caller-owned output array.
    /// @param capacity Destination element capacity.
    /// @return Number of counts copied.
    /// @invariant `capacity` covers every used call.
    uint32_t CudaPhase2FilterSortBuffer::downloadCallCounts(
        uint32_t* destination, uint32_t capacity) const {
        assert(usedCalls_ <= capacity);
        if (usedCalls_ == 0) return 0;
        assert(destination != nullptr);
        GL_CUDA_ASSERT(cudaMemcpy(
            destination, deviceCounts_,
            static_cast<std::size_t>(usedCalls_) * sizeof(uint32_t),
            cudaMemcpyDeviceToHost));
        return usedCalls_;
    }

    /// @brief Download the most recent globally sorted composite keys.
    ///
    /// @details
    /// Copies exactly the compact retained prefix. A zero-result sweep returns
    /// zero without issuing a zero-byte CUDA copy.
    ///
    /// @param destination Caller-owned output array.
    /// @param capacity Destination element capacity.
    /// @return Number of keys copied.
    /// @invariant `capacity` covers every retained key.
    uint32_t CudaPhase2FilterSortBuffer::downloadSortedKeys(
        uint64_t* destination, uint32_t capacity) const {
        assert(usedRetainedRows_ <= capacity);
        if (usedRetainedRows_ == 0) return 0;
        assert(destination != nullptr);
        GL_CUDA_ASSERT(cudaMemcpy(
            destination, deviceKeysOutput_,
            static_cast<std::size_t>(usedRetainedRows_) * sizeof(uint64_t),
            cudaMemcpyDeviceToHost));
        return usedRetainedRows_;
    }

    /// @brief Allocate every measured request-growth array once.
    ///
    /// @details
    /// Validates the active device and retained positive ceilings, then allocates
    /// the call schedule, direct/suffix masks, two equal frontiers, one frontier-
    /// sized prefix headers, the three measured prefix-value pools, stable short
    /// and cooperative selection flags/index lists, canonical node sort/scan
    /// columns, the bounded candidate-attempt/flag/survivor window, shared CUB
    /// scratch, persistent accepted-event and raw-request arrays, exact per-task
    /// subkey counters, and shared counters. Every byte count is computed in 64
    /// bits and asserted representable by the CUDA allocation API before ownership
    /// is established.
    ///
    /// @param fixedCapacity Immutable frontier, event, and request ceilings.
    /// @return An empty device growth owner.
    /// @invariant No later operation changes an allocation address or capacity.
    CudaPhase2GrowthBuffer::CudaPhase2GrowthBuffer(
        Phase2GrowthCapacity fixedCapacity)
        : capacity_(fixedCapacity) {
        static_cast<void>(queryCudaDeviceContract());
        assert(capacity_.calls > 0);
        assert(capacity_.retainedRows > 0);
        assert(capacity_.frontierRecords > 0);
        assert(capacity_.acceptedEvents > 0);
        assert(capacity_.rawRequests > 0);
        assert(capacity_.prefixPayloadValues > 0);
        assert(capacity_.prefixVariableValues > 0);
        assert(capacity_.prefixSecondaryValues > 0);
        assert(capacity_.candidateWindowRecords > 0);
        const uint64_t callBytes = static_cast<uint64_t>(capacity_.calls)
            * sizeof(DevicePhase2GrowthCall);
        const uint64_t maskBytes = capacity_.retainedRows;
        const uint64_t frontierBytes = static_cast<uint64_t>(
            capacity_.frontierRecords) * sizeof(DeviceGrowthNode);
        const uint64_t prefixBytes = static_cast<uint64_t>(
            capacity_.frontierRecords) * sizeof(DeviceGrowthPrefix);
        const uint64_t prefixPayloadBytes = static_cast<uint64_t>(
            capacity_.prefixPayloadValues) * sizeof(NameId);
        const uint64_t prefixVariableBytes = static_cast<uint64_t>(
            capacity_.prefixVariableValues) * sizeof(NameId);
        const uint64_t prefixSecondaryBytes = static_cast<uint64_t>(
            capacity_.prefixSecondaryValues) * sizeof(NameId);
        const uint64_t nodeIndexBytes = static_cast<uint64_t>(
            capacity_.frontierRecords) * sizeof(uint32_t);
        const uint64_t nodeSortKeyBytes = static_cast<uint64_t>(
            capacity_.frontierRecords) * sizeof(uint64_t);
        const uint64_t nodeAttemptOrdinalBytes = static_cast<uint64_t>(
            capacity_.frontierRecords) * sizeof(uint64_t);
        const uint64_t nodeFlagBytes = capacity_.frontierRecords;
        const uint64_t candidateAttemptBytes = static_cast<uint64_t>(
            capacity_.candidateWindowRecords)
            * sizeof(DeviceGrowthCandidateAttempt);
        const uint64_t candidateFlagBytes = capacity_.candidateWindowRecords;
        const uint64_t candidateSurvivorBytes = static_cast<uint64_t>(
            capacity_.candidateWindowRecords) * sizeof(uint32_t);
        const uint64_t acceptedEventBytes = static_cast<uint64_t>(
            capacity_.acceptedEvents) * sizeof(DeviceAcceptedGrowthEvent);
        const uint64_t rawRequestBytes = static_cast<uint64_t>(
            capacity_.rawRequests) * sizeof(DeviceRawGrowthRequest);
        const uint64_t taskSubkeyCountBytes = static_cast<uint64_t>(
            capacity_.calls) * sizeof(uint32_t);
        assert(callBytes <= std::numeric_limits<std::size_t>::max());
        assert(maskBytes <= std::numeric_limits<std::size_t>::max());
        assert(frontierBytes <= std::numeric_limits<std::size_t>::max());
        assert(prefixBytes <= std::numeric_limits<std::size_t>::max());
        assert(prefixPayloadBytes <= std::numeric_limits<std::size_t>::max());
        assert(prefixVariableBytes <= std::numeric_limits<std::size_t>::max());
        assert(prefixSecondaryBytes <= std::numeric_limits<std::size_t>::max());
        assert(nodeIndexBytes <= std::numeric_limits<std::size_t>::max());
        assert(nodeSortKeyBytes <= std::numeric_limits<std::size_t>::max());
        assert(nodeAttemptOrdinalBytes
            <= std::numeric_limits<std::size_t>::max());
        assert(nodeFlagBytes <= std::numeric_limits<std::size_t>::max());
        assert(candidateAttemptBytes
            <= std::numeric_limits<std::size_t>::max());
        assert(candidateFlagBytes <= std::numeric_limits<std::size_t>::max());
        assert(candidateSurvivorBytes
            <= std::numeric_limits<std::size_t>::max());
        assert(acceptedEventBytes <= std::numeric_limits<std::size_t>::max());
        assert(rawRequestBytes <= std::numeric_limits<std::size_t>::max());
        assert(taskSubkeyCountBytes
            <= std::numeric_limits<std::size_t>::max());
        thrust::counting_iterator<uint32_t> countingIndices(0);
        std::size_t requestedTemporaryBytes = 0;
        GL_CUDA_ASSERT(cub::DeviceSelect::Flagged(
            nullptr, requestedTemporaryBytes, countingIndices,
            static_cast<const uint8_t*>(nullptr),
            static_cast<uint32_t*>(nullptr),
            static_cast<uint32_t*>(nullptr),
            capacity_.frontierRecords));
        selectionTemporaryBytes_ = std::max(
            selectionTemporaryBytes_, requestedTemporaryBytes);
        requestedTemporaryBytes = 0;
        GL_CUDA_ASSERT(cub::DeviceRadixSort::SortPairs(
            nullptr, requestedTemporaryBytes,
            static_cast<const uint64_t*>(nullptr),
            static_cast<uint64_t*>(nullptr),
            static_cast<const uint32_t*>(nullptr),
            static_cast<uint32_t*>(nullptr),
            capacity_.frontierRecords, 0, 32));
        selectionTemporaryBytes_ = std::max(
            selectionTemporaryBytes_, requestedTemporaryBytes);
        requestedTemporaryBytes = 0;
        GL_CUDA_ASSERT(cub::DeviceScan::InclusiveSum(
            nullptr, requestedTemporaryBytes,
            static_cast<const uint64_t*>(nullptr),
            static_cast<uint64_t*>(nullptr),
            capacity_.frontierRecords));
        selectionTemporaryBytes_ = std::max(
            selectionTemporaryBytes_, requestedTemporaryBytes);
        requestedTemporaryBytes = 0;
        GL_CUDA_ASSERT(cub::DeviceSelect::Flagged(
            nullptr, requestedTemporaryBytes, countingIndices,
            static_cast<const uint8_t*>(nullptr),
            static_cast<uint32_t*>(nullptr),
            static_cast<uint32_t*>(nullptr),
            capacity_.candidateWindowRecords));
        selectionTemporaryBytes_ = std::max(
            selectionTemporaryBytes_, requestedTemporaryBytes);
        assert(selectionTemporaryBytes_ > 0);
        GL_CUDA_ASSERT(cudaMalloc(
            &deviceCalls_, static_cast<std::size_t>(callBytes)));
        GL_CUDA_ASSERT(cudaMalloc(
            &deviceViewMasks_, static_cast<std::size_t>(maskBytes)));
        GL_CUDA_ASSERT(cudaMalloc(
            &deviceSuffixMasks_, static_cast<std::size_t>(maskBytes)));
        for (uint32_t frontier = 0; frontier < 2; ++frontier) {
            GL_CUDA_ASSERT(cudaMalloc(
                &deviceFrontiers_[frontier],
                static_cast<std::size_t>(frontierBytes)));
            assert(deviceFrontiers_[frontier] != nullptr);
        }
        GL_CUDA_ASSERT(cudaMalloc(
            &devicePrefixes_, static_cast<std::size_t>(prefixBytes)));
        GL_CUDA_ASSERT(cudaMalloc(
            &devicePrefixPayload_,
            static_cast<std::size_t>(prefixPayloadBytes)));
        GL_CUDA_ASSERT(cudaMalloc(
            &devicePrefixVariables_,
            static_cast<std::size_t>(prefixVariableBytes)));
        GL_CUDA_ASSERT(cudaMalloc(
            &devicePrefixSecondary_,
            static_cast<std::size_t>(prefixSecondaryBytes)));
        GL_CUDA_ASSERT(cudaMalloc(
            &deviceShortNodeFlags_,
            static_cast<std::size_t>(nodeFlagBytes)));
        GL_CUDA_ASSERT(cudaMalloc(
            &deviceCooperativeNodeFlags_,
            static_cast<std::size_t>(nodeFlagBytes)));
        GL_CUDA_ASSERT(cudaMalloc(
            &deviceShortNodeIndices_,
            static_cast<std::size_t>(nodeIndexBytes)));
        GL_CUDA_ASSERT(cudaMalloc(
            &deviceCooperativeNodeIndices_,
            static_cast<std::size_t>(nodeIndexBytes)));
        for (uint32_t buffer = 0; buffer < 2; ++buffer) {
            GL_CUDA_ASSERT(cudaMalloc(
                &deviceCanonicalNodeIndices_[buffer],
                static_cast<std::size_t>(nodeIndexBytes)));
            GL_CUDA_ASSERT(cudaMalloc(
                &deviceNodeSortKeys_[buffer],
                static_cast<std::size_t>(nodeSortKeyBytes)));
        }
        GL_CUDA_ASSERT(cudaMalloc(
            &deviceNodeAttemptStarts_,
            static_cast<std::size_t>(nodeAttemptOrdinalBytes)));
        GL_CUDA_ASSERT(cudaMalloc(
            &deviceNodeAttemptEnds_,
            static_cast<std::size_t>(nodeAttemptOrdinalBytes)));
        GL_CUDA_ASSERT(cudaMalloc(
            &deviceCandidateAttempts_,
            static_cast<std::size_t>(candidateAttemptBytes)));
        GL_CUDA_ASSERT(cudaMalloc(
            &deviceCandidateFlags_,
            static_cast<std::size_t>(candidateFlagBytes)));
        GL_CUDA_ASSERT(cudaMalloc(
            &deviceCandidateSurvivorIndices_,
            static_cast<std::size_t>(candidateSurvivorBytes)));
        GL_CUDA_ASSERT(cudaMalloc(
            &deviceSelectionTemporary_, selectionTemporaryBytes_));
        GL_CUDA_ASSERT(cudaMalloc(
            &deviceAcceptedEvents_,
            static_cast<std::size_t>(acceptedEventBytes)));
        GL_CUDA_ASSERT(cudaMalloc(
            &deviceRawRequests_,
            static_cast<std::size_t>(rawRequestBytes)));
        GL_CUDA_ASSERT(cudaMalloc(
            &deviceTaskSubkeyCounts_,
            static_cast<std::size_t>(taskSubkeyCountBytes)));
        GL_CUDA_ASSERT(cudaMalloc(
            &deviceCounters_, sizeof(DeviceGrowthCounters)));
        assert(deviceCalls_ != nullptr);
        assert(deviceViewMasks_ != nullptr);
        assert(deviceSuffixMasks_ != nullptr);
        assert(devicePrefixes_ != nullptr);
        assert(devicePrefixPayload_ != nullptr);
        assert(devicePrefixVariables_ != nullptr);
        assert(devicePrefixSecondary_ != nullptr);
        assert(deviceShortNodeFlags_ != nullptr);
        assert(deviceCooperativeNodeFlags_ != nullptr);
        assert(deviceShortNodeIndices_ != nullptr);
        assert(deviceCooperativeNodeIndices_ != nullptr);
        for (uint32_t buffer = 0; buffer < 2; ++buffer) {
            assert(deviceCanonicalNodeIndices_[buffer] != nullptr);
            assert(deviceNodeSortKeys_[buffer] != nullptr);
        }
        assert(deviceNodeAttemptStarts_ != nullptr);
        assert(deviceNodeAttemptEnds_ != nullptr);
        assert(deviceCandidateAttempts_ != nullptr);
        assert(deviceCandidateFlags_ != nullptr);
        assert(deviceCandidateSurvivorIndices_ != nullptr);
        assert(deviceSelectionTemporary_ != nullptr);
        assert(deviceAcceptedEvents_ != nullptr);
        assert(deviceRawRequests_ != nullptr);
        assert(deviceTaskSubkeyCounts_ != nullptr);
        assert(deviceCounters_ != nullptr);
    }

    /// @brief Release every fixed request-growth device allocation.
    ///
    /// @details
    /// Frees shared counters, per-task subkey counts, raw request and accepted-
    /// event ledgers, selection scratch, candidate-window columns, semantic
    /// sort/scan columns, node-index and flag lists, the three prefix pools and
    /// headers, both node frontiers, mandatory masks, and the call schedule in
    /// reverse ownership order. Disabled copy and move operations keep every
    /// address uniquely owned.
    ///
    /// @return Nothing.
    CudaPhase2GrowthBuffer::~CudaPhase2GrowthBuffer() {
        GL_CUDA_ASSERT(cudaFree(deviceCounters_));
        GL_CUDA_ASSERT(cudaFree(deviceTaskSubkeyCounts_));
        GL_CUDA_ASSERT(cudaFree(deviceRawRequests_));
        GL_CUDA_ASSERT(cudaFree(deviceAcceptedEvents_));
        GL_CUDA_ASSERT(cudaFree(deviceSelectionTemporary_));
        GL_CUDA_ASSERT(cudaFree(deviceCandidateSurvivorIndices_));
        GL_CUDA_ASSERT(cudaFree(deviceCandidateFlags_));
        GL_CUDA_ASSERT(cudaFree(deviceCandidateAttempts_));
        GL_CUDA_ASSERT(cudaFree(deviceNodeAttemptEnds_));
        GL_CUDA_ASSERT(cudaFree(deviceNodeAttemptStarts_));
        for (uint32_t buffer = 2; buffer > 0; --buffer) {
            GL_CUDA_ASSERT(cudaFree(deviceNodeSortKeys_[buffer - 1]));
            GL_CUDA_ASSERT(cudaFree(deviceCanonicalNodeIndices_[buffer - 1]));
        }
        GL_CUDA_ASSERT(cudaFree(deviceCooperativeNodeIndices_));
        GL_CUDA_ASSERT(cudaFree(deviceShortNodeIndices_));
        GL_CUDA_ASSERT(cudaFree(deviceCooperativeNodeFlags_));
        GL_CUDA_ASSERT(cudaFree(deviceShortNodeFlags_));
        GL_CUDA_ASSERT(cudaFree(devicePrefixSecondary_));
        GL_CUDA_ASSERT(cudaFree(devicePrefixVariables_));
        GL_CUDA_ASSERT(cudaFree(devicePrefixPayload_));
        GL_CUDA_ASSERT(cudaFree(devicePrefixes_));
        for (uint32_t frontier = 2; frontier > 0; --frontier)
            GL_CUDA_ASSERT(cudaFree(deviceFrontiers_[frontier - 1]));
        GL_CUDA_ASSERT(cudaFree(deviceSuffixMasks_));
        GL_CUDA_ASSERT(cudaFree(deviceViewMasks_));
        GL_CUDA_ASSERT(cudaFree(deviceCalls_));
    }

    /// @brief Run bulk normalized-key request frontier growth.
    ///
    /// @details
    /// Uploads the pointer-free growth schedule, builds mandatory-view and suffix
    /// masks over completed filtered spans, seeds unsplit roots or exact stump
    /// nodes, then expands through fixed ping-pong frontiers. Production prepares
    /// one pooled prefix per node, compacts short and cooperative lists, establishes
    /// stable `(call, run, path)` order, and processes every candidate through
    /// bounded cheap-gate and compact survivor windows. Observation mode retains
    /// the separately compiled direct kernels for exact gate census. Stump seeds
    /// are grouped by absolute starting depth, so every live wave is depth-
    /// homogeneous and the measured per-depth frontier ceiling remains the exact
    /// allocation bound. Host reads only bounded primitive counts and the small
    /// frontier counter record between launches; all semantic rows remain resident
    /// and all append capacity failures assert on the device.
    ///
    /// @param projection Uploaded resident logical-block image.
    /// @param tasks Uploaded executor task, batch, term, and stump image.
    /// @param filter Completed global filter/sort result for these calls.
    /// @param schedule Host request-growth schedule in processor batch order.
    /// @param parameters Immutable analyzer request-shape limits.
    /// @return Used event/request counts and maximum live frontier size.
    /// @invariant Every schedule link names compatible uploaded task, batch,
    ///            filter, and logical-block records.
    Phase2GrowthResult CudaPhase2GrowthBuffer::runRequestGrowth(
        const CudaPhase2ProjectionBuffer& projection,
        const CudaPhase2TaskBuffer& tasks,
        const CudaPhase2FilterSortBuffer& filter,
        const Phase2GrowthScheduleArena& schedule,
        DevicePhase2GrowthParameters parameters) {
        assert(schedule.calls.size() <= capacity_.calls);
        assert(schedule.calls.size() <= filter.usedCalls_);
        assert(filter.usedRetainedRows_ <= capacity_.retainedRows);
        assert(parameters.maximumHypothesisKeyLength >= 0);
        assert(parameters.maximumSecondaryVariables >= 0);
        assert(parameters.maximumSecondaryVariablesOrint
            >= parameters.maximumSecondaryVariables);
        assert(parameters.cooperativeSpanThreshold > 0);
        assert(parameters.collectSpanCensus <= 2);
        assert(projection.usedCounts_[0] > 0);
        assert(tasks.usedCounts_[0] > 0);
        assert(tasks.usedCounts_[1] > 0);
        assert(schedule.calls.size()
            <= static_cast<std::size_t>(
                std::numeric_limits<uint32_t>::max()));
        const uint32_t callCount = static_cast<uint32_t>(
            schedule.calls.size());
        usedAcceptedEvents_ = 0;
        usedRawRequests_ = 0;
        usedTaskCount_ = tasks.usedCounts_[0];
        assert(usedTaskCount_ <= capacity_.calls);
        GL_CUDA_ASSERT(cudaMemset(
            deviceTaskSubkeyCounts_, 0,
            static_cast<std::size_t>(capacity_.calls) * sizeof(uint32_t)));
        GL_CUDA_ASSERT(cudaMemset(
            deviceCounters_, 0, sizeof(DeviceGrowthCounters)));
        if (callCount == 0) return Phase2GrowthResult{};
        GL_CUDA_ASSERT(cudaMemcpy(
            deviceCalls_, schedule.calls.data(),
            static_cast<std::size_t>(callCount)
                * sizeof(DevicePhase2GrowthCall),
            cudaMemcpyHostToDevice));

        ProjectionSemanticColumns columns{};
        columns.logicalBlocks = static_cast<
            const DeviceLogicalBlockProjection*>(projection.deviceColumns_[0]);
        columns.statements = static_cast<const IntEncodedExpr*>(
            projection.deviceColumns_[1]);
        columns.nameRecords = static_cast<const DeviceNameRecord*>(
            projection.deviceColumns_[2]);
        columns.nameBytes = static_cast<const char*>(
            projection.deviceColumns_[3]);
        columns.nameSlots = static_cast<const int32_t*>(
            projection.deviceColumns_[4]);
        columns.ruleStringRecords = static_cast<const DeviceRuleStringRecord*>(
            projection.deviceColumns_[5]);
        columns.ruleStringBytes = static_cast<const char*>(
            projection.deviceColumns_[6]);
        columns.byteMapViews = static_cast<const DeviceByteMapView*>(
            projection.deviceColumns_[7]);
        columns.byteMapEntries = static_cast<const DeviceByteMapEntry*>(
            projection.deviceColumns_[8]);
        columns.byteMapSlots = static_cast<const int32_t*>(
            projection.deviceColumns_[9]);
        columns.byteKeyBytes = static_cast<const char*>(
            projection.deviceColumns_[10]);
        columns.blobRecords = static_cast<const DeviceBlobRecord*>(
            projection.deviceColumns_[11]);
        columns.blobBytes = static_cast<const char*>(
            projection.deviceColumns_[12]);
        columns.reverseMapViews = static_cast<const DeviceReverseMapView*>(
            projection.deviceColumns_[13]);
        columns.reverseMapEntries = static_cast<const DeviceReverseMapEntry*>(
            projection.deviceColumns_[14]);
        columns.reverseMapSlots = static_cast<const int32_t*>(
            projection.deviceColumns_[15]);
        columns.reverseKeyBytes = static_cast<const char*>(
            projection.deviceColumns_[16]);
        columns.reverseOwners = static_cast<const int32_t*>(
            projection.deviceColumns_[17]);
        columns.podMapViews = static_cast<const DevicePodMapView*>(
            projection.deviceColumns_[18]);
        columns.podMapEntries = static_cast<const DevicePodMapEntry*>(
            projection.deviceColumns_[19]);
        columns.podMapSlots = static_cast<const int32_t*>(
            projection.deviceColumns_[20]);
        columns.podRunValues = static_cast<const int32_t*>(
            projection.deviceColumns_[21]);
        columns.mandatoryStatementKeys = static_cast<const int64_t*>(
            projection.deviceColumns_[22]);
        columns.metadataBytes = static_cast<const char*>(
            projection.deviceColumns_[23]);

        constexpr uint32_t maskThreads = 128;
        phase2GrowthMandatoryMaskKernel<<<callCount, maskThreads>>>(
            columns,
            static_cast<const DevicePhase2Task*>(tasks.deviceColumns_[0]),
            tasks.usedCounts_[0],
            static_cast<const DeviceRequestBatch*>(tasks.deviceColumns_[1]),
            tasks.usedCounts_[1],
            static_cast<const DeviceMandatoryTerm*>(tasks.deviceColumns_[2]),
            static_cast<const DevicePhase2GrowthCall*>(deviceCalls_), callCount,
            static_cast<const DevicePhase2FilterCall*>(filter.deviceCalls_),
            filter.usedCalls_,
            static_cast<const uint32_t*>(filter.deviceCounts_),
            static_cast<const uint32_t*>(filter.deviceOffsets_),
            static_cast<const uint64_t*>(filter.deviceKeysOutput_),
            static_cast<uint8_t*>(deviceViewMasks_),
            static_cast<uint8_t*>(deviceSuffixMasks_));
        GL_CUDA_ASSERT(cudaGetLastError());

        constexpr uint32_t growthThreads = 256;
        constexpr uint32_t cooperativeGrowthThreads = 64;
        DeviceGrowthCounters hostCounters{};
        uint32_t maximumFrontier = 0;
        uint32_t maximumCooperativeNodes = 0;
        uint64_t maximumPrefixPayloadValues = 0;
        uint64_t maximumPrefixVariableValues = 0;
        uint64_t maximumPrefixSecondaryValues = 0;
        for (uint32_t seedDepth = 0;
             seedDepth <= static_cast<uint32_t>(
                 ExecutionParameters::MAX_EXPRESSIONS);
             ++seedDepth) {
            GL_CUDA_ASSERT(cudaMemset(
                deviceCounters_, 0,
                sizeof(hostCounters.frontierCounts)));
            phase2GrowthSeedKernel<<<
                (callCount + growthThreads - 1) / growthThreads,
                growthThreads>>>(
                columns,
                static_cast<const DevicePhase2Task*>(tasks.deviceColumns_[0]),
                static_cast<const DeviceRequestBatch*>(tasks.deviceColumns_[1]),
                static_cast<const DeviceMandatoryTerm*>(tasks.deviceColumns_[2]),
                static_cast<const DeviceExpressionStump*>(tasks.deviceColumns_[3]),
                static_cast<const DevicePhase2GrowthCall*>(deviceCalls_),
                callCount,
                static_cast<const uint32_t*>(filter.deviceCounts_),
                static_cast<const uint32_t*>(filter.deviceOffsets_),
                static_cast<const uint64_t*>(filter.deviceKeysOutput_),
                static_cast<const uint8_t*>(deviceViewMasks_),
                parameters, seedDepth,
                static_cast<DeviceGrowthNode*>(deviceFrontiers_[0]),
                static_cast<DeviceAcceptedGrowthEvent*>(deviceAcceptedEvents_),
                static_cast<DeviceRawGrowthRequest*>(deviceRawRequests_),
                static_cast<uint32_t*>(deviceTaskSubkeyCounts_),
                static_cast<DeviceGrowthCounters*>(deviceCounters_), capacity_);
            GL_CUDA_ASSERT(cudaGetLastError());
            GL_CUDA_ASSERT(cudaDeviceSynchronize());
            GL_CUDA_ASSERT(cudaMemcpy(
                &hostCounters, deviceCounters_, sizeof(hostCounters),
                cudaMemcpyDeviceToHost));
            uint32_t currentFrontier = 0;
            uint32_t currentCount = hostCounters.frontierCounts[0];
            if (currentCount > maximumFrontier)
                maximumFrontier = currentCount;
            for (uint32_t level = seedDepth;
                 level < static_cast<uint32_t>(
                     ExecutionParameters::MAX_EXPRESSIONS)
                     && currentCount > 0; ++level) {
                const uint32_t nextFrontier = currentFrontier ^ 1u;
                const std::size_t counterOffset =
                    offsetof(DeviceGrowthCounters, frontierCounts)
                    + static_cast<std::size_t>(nextFrontier)
                        * sizeof(uint32_t);
                GL_CUDA_ASSERT(cudaMemset(
                    static_cast<char*>(deviceCounters_) + counterOffset,
                    0, sizeof(uint32_t)));
                const std::size_t preparationCounterOffset =
                    offsetof(DeviceGrowthCounters, shortNodeCount);
                constexpr std::size_t preparationCounterBytes =
                    5 * sizeof(uint32_t);
                GL_CUDA_ASSERT(cudaMemset(
                    static_cast<char*>(deviceCounters_)
                        + preparationCounterOffset,
                    0, preparationCounterBytes));
                if (parameters.collectSpanCensus != 0) {
                    phase2GrowthSpanCensusKernel<<<
                        (currentCount + growthThreads - 1) / growthThreads,
                        growthThreads>>>(
                        static_cast<const DevicePhase2GrowthCall*>(deviceCalls_),
                        static_cast<const uint32_t*>(filter.deviceCounts_),
                        static_cast<const DeviceGrowthNode*>(
                            deviceFrontiers_[currentFrontier]),
                        currentCount,
                        static_cast<DeviceGrowthCounters*>(deviceCounters_));
                    GL_CUDA_ASSERT(cudaGetLastError());
                }
                phase2GrowthPreparePrefixesKernel<<<
                    (currentCount + growthThreads - 1) / growthThreads,
                    growthThreads>>>(
                    columns,
                    static_cast<const DeviceRequestBatch*>(
                        tasks.deviceColumns_[1]),
                    static_cast<const DeviceMandatoryTerm*>(
                        tasks.deviceColumns_[2]),
                    static_cast<const DevicePhase2Task*>(
                        tasks.deviceColumns_[0]),
                    static_cast<const DevicePhase2GrowthCall*>(deviceCalls_),
                    static_cast<const uint32_t*>(filter.deviceCounts_),
                    static_cast<const uint32_t*>(filter.deviceOffsets_),
                    static_cast<const uint64_t*>(filter.deviceKeysOutput_),
                    parameters,
                    static_cast<const DeviceGrowthNode*>(
                        deviceFrontiers_[currentFrontier]), currentCount,
                    static_cast<DeviceGrowthPrefix*>(devicePrefixes_),
                    static_cast<NameId*>(devicePrefixPayload_),
                    static_cast<NameId*>(devicePrefixVariables_),
                    static_cast<NameId*>(devicePrefixSecondary_),
                    static_cast<uint8_t*>(deviceShortNodeFlags_),
                    static_cast<uint8_t*>(deviceCooperativeNodeFlags_),
                    static_cast<DeviceGrowthCounters*>(deviceCounters_),
                    capacity_);
                GL_CUDA_ASSERT(cudaGetLastError());
                thrust::counting_iterator<uint32_t> countingIndices(0);
                std::size_t selectionBytes = selectionTemporaryBytes_;
                GL_CUDA_ASSERT(cub::DeviceSelect::Flagged(
                    deviceSelectionTemporary_, selectionBytes,
                    countingIndices,
                    static_cast<const uint8_t*>(deviceShortNodeFlags_),
                    static_cast<uint32_t*>(deviceShortNodeIndices_),
                    reinterpret_cast<uint32_t*>(
                        static_cast<char*>(deviceCounters_)
                            + offsetof(DeviceGrowthCounters, shortNodeCount)),
                    currentCount));
                assert(selectionBytes <= selectionTemporaryBytes_);
                selectionBytes = selectionTemporaryBytes_;
                GL_CUDA_ASSERT(cub::DeviceSelect::Flagged(
                    deviceSelectionTemporary_, selectionBytes,
                    countingIndices,
                    static_cast<const uint8_t*>(
                        deviceCooperativeNodeFlags_),
                    static_cast<uint32_t*>(deviceCooperativeNodeIndices_),
                    reinterpret_cast<uint32_t*>(
                        static_cast<char*>(deviceCounters_)
                            + offsetof(
                                DeviceGrowthCounters,
                                cooperativeNodeCount)),
                    currentCount));
                assert(selectionBytes <= selectionTemporaryBytes_);
                GL_CUDA_ASSERT(cudaDeviceSynchronize());
                GL_CUDA_ASSERT(cudaMemcpy(
                    &hostCounters.shortNodeCount,
                    static_cast<const char*>(deviceCounters_)
                        + preparationCounterOffset,
                    preparationCounterBytes, cudaMemcpyDeviceToHost));
                assert(hostCounters.shortNodeCount
                        + hostCounters.cooperativeNodeCount
                    <= currentCount);
                maximumPrefixPayloadValues = std::max(
                    maximumPrefixPayloadValues,
                    static_cast<uint64_t>(hostCounters.prefixPayloadCount));
                maximumPrefixVariableValues = std::max(
                    maximumPrefixVariableValues,
                    static_cast<uint64_t>(hostCounters.prefixVariableCount));
                maximumPrefixSecondaryValues = std::max(
                    maximumPrefixSecondaryValues,
                    static_cast<uint64_t>(hostCounters.prefixSecondaryCount));
                if (hostCounters.cooperativeNodeCount
                    > maximumCooperativeNodes) {
                    maximumCooperativeNodes =
                        hostCounters.cooperativeNodeCount;
                }
                if (parameters.collectSpanCensus == 2) {
                    if (hostCounters.shortNodeCount > 0) {
                        phase2GrowthShortExpandKernel<true><<<
                            (hostCounters.shortNodeCount + growthThreads - 1)
                                / growthThreads,
                            growthThreads>>>(
                            columns,
                            static_cast<const DeviceRequestBatch*>(
                                tasks.deviceColumns_[1]),
                            static_cast<const DevicePhase2Task*>(
                                tasks.deviceColumns_[0]),
                            static_cast<const DevicePhase2GrowthCall*>(
                                deviceCalls_),
                            static_cast<const uint32_t*>(filter.deviceCounts_),
                            static_cast<const uint32_t*>(filter.deviceOffsets_),
                            static_cast<const uint64_t*>(
                                filter.deviceKeysOutput_),
                            static_cast<const uint8_t*>(deviceViewMasks_),
                            static_cast<const uint8_t*>(deviceSuffixMasks_),
                            parameters,
                            static_cast<const DeviceGrowthNode*>(
                                deviceFrontiers_[currentFrontier]),
                            static_cast<const uint32_t*>(
                                deviceShortNodeIndices_),
                            hostCounters.shortNodeCount,
                            static_cast<const DeviceGrowthPrefix*>(
                                devicePrefixes_),
                            static_cast<const NameId*>(devicePrefixPayload_),
                            static_cast<const NameId*>(devicePrefixVariables_),
                            static_cast<const NameId*>(devicePrefixSecondary_),
                            static_cast<DeviceGrowthNode*>(
                                deviceFrontiers_[nextFrontier]), nextFrontier,
                            static_cast<DeviceAcceptedGrowthEvent*>(
                                deviceAcceptedEvents_),
                            static_cast<DeviceRawGrowthRequest*>(
                                deviceRawRequests_),
                            static_cast<uint32_t*>(deviceTaskSubkeyCounts_),
                            static_cast<DeviceGrowthCounters*>(deviceCounters_),
                            capacity_);
                        GL_CUDA_ASSERT(cudaGetLastError());
                    }
                }
                else {
                    phase2GrowthMergeNodeIndicesKernel<<<
                        (currentCount + growthThreads - 1) / growthThreads,
                        growthThreads>>>(
                        static_cast<const uint32_t*>(deviceShortNodeIndices_),
                        hostCounters.shortNodeCount,
                        static_cast<const uint32_t*>(
                            deviceCooperativeNodeIndices_),
                        hostCounters.cooperativeNodeCount,
                        static_cast<uint32_t*>(deviceCanonicalNodeIndices_[0]),
                        currentCount);
                    GL_CUDA_ASSERT(cudaGetLastError());
                    uint32_t canonicalBuffer = 0;
                    const auto stableSortNodeField = [&](uint32_t field) {
                        phase2GrowthNodeSortKeyKernel<<<
                            (currentCount + growthThreads - 1) / growthThreads,
                            growthThreads>>>(
                            static_cast<const DeviceGrowthNode*>(
                                deviceFrontiers_[currentFrontier]),
                            static_cast<const uint32_t*>(
                                deviceCanonicalNodeIndices_[canonicalBuffer]),
                            currentCount, field,
                            static_cast<uint64_t*>(deviceNodeSortKeys_[0]));
                        GL_CUDA_ASSERT(cudaGetLastError());
                        std::size_t sortBytes = selectionTemporaryBytes_;
                        GL_CUDA_ASSERT(cub::DeviceRadixSort::SortPairs(
                            deviceSelectionTemporary_, sortBytes,
                            static_cast<const uint64_t*>(
                                deviceNodeSortKeys_[0]),
                            static_cast<uint64_t*>(deviceNodeSortKeys_[1]),
                            static_cast<const uint32_t*>(
                                deviceCanonicalNodeIndices_[canonicalBuffer]),
                            static_cast<uint32_t*>(
                                deviceCanonicalNodeIndices_[
                                    canonicalBuffer ^ 1u]),
                            currentCount, 0, 32));
                        assert(sortBytes <= selectionTemporaryBytes_);
                        canonicalBuffer ^= 1u;
                    };
                    for (uint32_t premise = level; premise > 0; --premise)
                        stableSortNodeField(premise - 1);
                    stableSortNodeField(
                        gl::ExecutionParameters::MAX_EXPRESSIONS);
                    stableSortNodeField(
                        gl::ExecutionParameters::MAX_EXPRESSIONS + 1u);
                    phase2GrowthCandidateCountsKernel<<<
                        (currentCount + growthThreads - 1) / growthThreads,
                        growthThreads>>>(
                        static_cast<const DeviceGrowthNode*>(
                            deviceFrontiers_[currentFrontier]),
                        static_cast<const uint32_t*>(
                            deviceCanonicalNodeIndices_[canonicalBuffer]),
                        currentCount,
                        static_cast<const DevicePhase2GrowthCall*>(
                            deviceCalls_),
                        static_cast<const uint32_t*>(filter.deviceCounts_),
                        static_cast<uint64_t*>(deviceNodeSortKeys_[0]));
                    GL_CUDA_ASSERT(cudaGetLastError());
                    std::size_t scanBytes = selectionTemporaryBytes_;
                    GL_CUDA_ASSERT(cub::DeviceScan::InclusiveSum(
                        deviceSelectionTemporary_, scanBytes,
                        static_cast<const uint64_t*>(deviceNodeSortKeys_[0]),
                        static_cast<uint64_t*>(deviceNodeAttemptEnds_),
                        currentCount));
                    assert(scanBytes <= selectionTemporaryBytes_);
                    uint64_t totalCandidateAttempts = 0;
                    GL_CUDA_ASSERT(cudaMemcpy(
                        &totalCandidateAttempts,
                        static_cast<const char*>(deviceNodeAttemptEnds_)
                            + static_cast<std::size_t>(currentCount - 1)
                                * sizeof(uint64_t),
                        sizeof(uint64_t), cudaMemcpyDeviceToHost));
                    phase2GrowthCandidateStartsKernel<<<
                        (currentCount + growthThreads - 1) / growthThreads,
                        growthThreads>>>(
                        static_cast<const uint32_t*>(
                            deviceCanonicalNodeIndices_[canonicalBuffer]),
                        static_cast<const uint64_t*>(deviceNodeAttemptEnds_),
                        currentCount,
                        static_cast<uint64_t*>(deviceNodeAttemptStarts_));
                    GL_CUDA_ASSERT(cudaGetLastError());
                    for (uint64_t windowStart = 0;
                         windowStart < totalCandidateAttempts;
                         windowStart += capacity_.candidateWindowRecords) {
                        const uint64_t remaining =
                            totalCandidateAttempts - windowStart;
                        const uint32_t windowCount = static_cast<uint32_t>(
                            std::min<uint64_t>(
                                remaining,
                                capacity_.candidateWindowRecords));
                        if (hostCounters.shortNodeCount > 0) {
                            phase2GrowthShortCheapGateKernel<<<
                                (hostCounters.shortNodeCount
                                    + growthThreads - 1) / growthThreads,
                                growthThreads>>>(
                                columns,
                                static_cast<const DeviceRequestBatch*>(
                                    tasks.deviceColumns_[1]),
                                static_cast<const DevicePhase2Task*>(
                                    tasks.deviceColumns_[0]),
                                static_cast<const DevicePhase2GrowthCall*>(
                                    deviceCalls_),
                                static_cast<const uint32_t*>(
                                    filter.deviceCounts_),
                                static_cast<const uint32_t*>(
                                    filter.deviceOffsets_),
                                static_cast<const uint64_t*>(
                                    filter.deviceKeysOutput_),
                                static_cast<const uint8_t*>(deviceViewMasks_),
                                static_cast<const uint8_t*>(deviceSuffixMasks_),
                                parameters,
                                static_cast<const DeviceGrowthNode*>(
                                    deviceFrontiers_[currentFrontier]),
                                static_cast<const DeviceGrowthPrefix*>(
                                    devicePrefixes_),
                                static_cast<const NameId*>(
                                    devicePrefixSecondary_),
                                static_cast<const uint32_t*>(
                                    deviceShortNodeIndices_),
                                hostCounters.shortNodeCount,
                                static_cast<const uint64_t*>(
                                    deviceNodeAttemptStarts_),
                                windowStart, windowCount,
                                static_cast<DeviceGrowthCandidateAttempt*>(
                                    deviceCandidateAttempts_),
                                static_cast<uint8_t*>(deviceCandidateFlags_));
                            GL_CUDA_ASSERT(cudaGetLastError());
                        }
                        if (hostCounters.cooperativeNodeCount > 0) {
                            phase2GrowthCooperativeCheapGateKernel<<<
                                hostCounters.cooperativeNodeCount,
                                cooperativeGrowthThreads>>>(
                                columns,
                                static_cast<const DeviceRequestBatch*>(
                                    tasks.deviceColumns_[1]),
                                static_cast<const DevicePhase2Task*>(
                                    tasks.deviceColumns_[0]),
                                static_cast<const DevicePhase2GrowthCall*>(
                                    deviceCalls_),
                                static_cast<const uint32_t*>(
                                    filter.deviceCounts_),
                                static_cast<const uint32_t*>(
                                    filter.deviceOffsets_),
                                static_cast<const uint64_t*>(
                                    filter.deviceKeysOutput_),
                                static_cast<const uint8_t*>(deviceViewMasks_),
                                static_cast<const uint8_t*>(deviceSuffixMasks_),
                                parameters,
                                static_cast<const DeviceGrowthNode*>(
                                    deviceFrontiers_[currentFrontier]),
                                static_cast<const DeviceGrowthPrefix*>(
                                    devicePrefixes_),
                                static_cast<const NameId*>(
                                    devicePrefixSecondary_),
                                static_cast<const uint32_t*>(
                                    deviceCooperativeNodeIndices_),
                                hostCounters.cooperativeNodeCount,
                                static_cast<const uint64_t*>(
                                    deviceNodeAttemptStarts_),
                                windowStart, windowCount,
                                static_cast<DeviceGrowthCandidateAttempt*>(
                                    deviceCandidateAttempts_),
                                static_cast<uint8_t*>(deviceCandidateFlags_));
                            GL_CUDA_ASSERT(cudaGetLastError());
                        }
                        std::size_t candidateSelectionBytes =
                            selectionTemporaryBytes_;
                        GL_CUDA_ASSERT(cub::DeviceSelect::Flagged(
                            deviceSelectionTemporary_, candidateSelectionBytes,
                            thrust::counting_iterator<uint32_t>(0),
                            static_cast<const uint8_t*>(deviceCandidateFlags_),
                            static_cast<uint32_t*>(
                                deviceCandidateSurvivorIndices_),
                            reinterpret_cast<uint32_t*>(
                                static_cast<char*>(deviceCounters_)
                                    + offsetof(
                                        DeviceGrowthCounters,
                                        shortNodeCount)),
                            windowCount));
                        assert(candidateSelectionBytes
                            <= selectionTemporaryBytes_);
                        uint32_t survivorCount = 0;
                        GL_CUDA_ASSERT(cudaMemcpy(
                            &survivorCount,
                            static_cast<const char*>(deviceCounters_)
                                + offsetof(
                                    DeviceGrowthCounters,
                                    shortNodeCount),
                            sizeof(uint32_t), cudaMemcpyDeviceToHost));
                        assert(survivorCount <= windowCount);
                        if (survivorCount > 0) {
                            phase2GrowthProbeSurvivorsKernel<<<
                                (survivorCount + growthThreads - 1)
                                    / growthThreads,
                                growthThreads>>>(
                                columns,
                                static_cast<const DeviceRequestBatch*>(
                                    tasks.deviceColumns_[1]),
                                static_cast<const DevicePhase2Task*>(
                                    tasks.deviceColumns_[0]),
                                static_cast<const DevicePhase2GrowthCall*>(
                                    deviceCalls_),
                                static_cast<const uint32_t*>(
                                    filter.deviceOffsets_),
                                static_cast<const uint64_t*>(
                                    filter.deviceKeysOutput_),
                                static_cast<const DeviceGrowthNode*>(
                                    deviceFrontiers_[currentFrontier]),
                                static_cast<const DeviceGrowthPrefix*>(
                                    devicePrefixes_),
                                static_cast<const NameId*>(
                                    devicePrefixPayload_),
                                static_cast<const NameId*>(
                                    devicePrefixVariables_),
                                static_cast<const DeviceGrowthCandidateAttempt*>(
                                    deviceCandidateAttempts_),
                                static_cast<const uint32_t*>(
                                    deviceCandidateSurvivorIndices_),
                                survivorCount,
                                static_cast<DeviceGrowthNode*>(
                                    deviceFrontiers_[nextFrontier]),
                                nextFrontier,
                                static_cast<DeviceAcceptedGrowthEvent*>(
                                    deviceAcceptedEvents_),
                                static_cast<DeviceRawGrowthRequest*>(
                                    deviceRawRequests_),
                                static_cast<uint32_t*>(
                                    deviceTaskSubkeyCounts_),
                                static_cast<DeviceGrowthCounters*>(
                                    deviceCounters_),
                                capacity_);
                            GL_CUDA_ASSERT(cudaGetLastError());
                        }
                    }
                }
                if (parameters.collectSpanCensus == 2
                    && hostCounters.cooperativeNodeCount > 0) {
                        phase2GrowthCooperativeExpandKernel<true><<<
                            hostCounters.cooperativeNodeCount,
                            cooperativeGrowthThreads>>>(
                            columns,
                            static_cast<const DeviceRequestBatch*>(
                                tasks.deviceColumns_[1]),
                            static_cast<const DevicePhase2Task*>(
                                tasks.deviceColumns_[0]),
                            static_cast<const DevicePhase2GrowthCall*>(
                                deviceCalls_),
                            static_cast<const uint32_t*>(filter.deviceCounts_),
                            static_cast<const uint32_t*>(filter.deviceOffsets_),
                            static_cast<const uint64_t*>(
                                filter.deviceKeysOutput_),
                            static_cast<const uint8_t*>(deviceViewMasks_),
                            static_cast<const uint8_t*>(deviceSuffixMasks_),
                            parameters,
                            static_cast<const DeviceGrowthNode*>(
                                deviceFrontiers_[currentFrontier]),
                            static_cast<const uint32_t*>(
                                deviceCooperativeNodeIndices_),
                            hostCounters.cooperativeNodeCount,
                            static_cast<const DeviceGrowthPrefix*>(
                                devicePrefixes_),
                            static_cast<const NameId*>(devicePrefixPayload_),
                            static_cast<const NameId*>(devicePrefixVariables_),
                            static_cast<const NameId*>(devicePrefixSecondary_),
                            static_cast<DeviceGrowthNode*>(
                                deviceFrontiers_[nextFrontier]), nextFrontier,
                            static_cast<DeviceAcceptedGrowthEvent*>(
                                deviceAcceptedEvents_),
                            static_cast<DeviceRawGrowthRequest*>(
                                deviceRawRequests_),
                            static_cast<uint32_t*>(deviceTaskSubkeyCounts_),
                            static_cast<DeviceGrowthCounters*>(deviceCounters_),
                            capacity_);
                    GL_CUDA_ASSERT(cudaGetLastError());
                }
                GL_CUDA_ASSERT(cudaDeviceSynchronize());
                GL_CUDA_ASSERT(cudaMemcpy(
                    &hostCounters, deviceCounters_, sizeof(hostCounters),
                    cudaMemcpyDeviceToHost));
                currentFrontier = nextFrontier;
                currentCount = hostCounters.frontierCounts[currentFrontier];
                if (currentCount > maximumFrontier)
                    maximumFrontier = currentCount;
            }
            assert(currentCount == 0);
        }
        assert(hostCounters.acceptedEventCount <= capacity_.acceptedEvents);
        assert(hostCounters.rawRequestCount <= capacity_.rawRequests);
        usedAcceptedEvents_ = hostCounters.acceptedEventCount;
        usedRawRequests_ = hostCounters.rawRequestCount;
        Phase2GrowthResult result{};
        result.acceptedEventCount = usedAcceptedEvents_;
        result.rawRequestCount = usedRawRequests_;
        result.maximumFrontierCount = maximumFrontier;
        result.maximumCooperativeNodeCount = maximumCooperativeNodes;
        result.maximumPrefixPayloadValues = maximumPrefixPayloadValues;
        result.maximumPrefixVariableValues = maximumPrefixVariableValues;
        result.maximumPrefixSecondaryValues = maximumPrefixSecondaryValues;
        for (uint32_t bucket = 0;
             bucket < kDeviceGrowthSpanBucketCount; ++bucket) {
            result.spanNodeCounts[bucket] =
                hostCounters.spanNodeCounts[bucket];
            result.spanCandidateCounts[bucket] =
                hostCounters.spanCandidateCounts[bucket];
        }
        for (uint32_t depth = 1;
             depth < kDeviceGrowthCensusDepthCount; ++depth) {
            const DeviceGrowthGateCounters& source =
                hostCounters.gateDepthCounts[depth];
            Phase2GrowthGateCensus& destination =
                result.gateDepthCounts[depth];
            destination.candidateAttempts = source.candidateAttempts;
            destination.mandatoryReachable = source.mandatoryReachable;
            destination.validityComparable = source.validityComparable;
            destination.hypothesisCompatible = source.hypothesisCompatible;
            destination.secondaryCompatible = source.secondaryCompatible;
            destination.keyLengthAllowed = source.keyLengthAllowed;
            destination.subkeyPresent = source.subkeyPresent;
            destination.ownerSatisfied = source.ownerSatisfied;
            destination.wholeKeyPresent = source.wholeKeyPresent;
            destination.termsSatisfied = source.termsSatisfied;
            destination.acceptedEvents = source.acceptedEvents;
            destination.children = source.children;
        }
        return result;
    }

    /// @brief Download the most recent accepted growth-event ledger.
    ///
    /// @details
    /// Copies the append prefix in arbitrary device execution order. Every record
    /// carries a complete statement path and call/run identity for host sorting.
    ///
    /// @param destination Caller-owned event output array.
    /// @param capacity Destination element capacity.
    /// @return Number of event records copied.
    /// @invariant `capacity` covers the last result's accepted-event count.
    uint32_t CudaPhase2GrowthBuffer::downloadAcceptedEvents(
        DeviceAcceptedGrowthEvent* destination,
        uint32_t capacity) const {
        assert(usedAcceptedEvents_ <= capacity);
        if (usedAcceptedEvents_ == 0) return 0;
        assert(destination != nullptr);
        GL_CUDA_ASSERT(cudaMemcpy(
            destination, deviceAcceptedEvents_,
            static_cast<std::size_t>(usedAcceptedEvents_)
                * sizeof(DeviceAcceptedGrowthEvent),
            cudaMemcpyDeviceToHost));
        return usedAcceptedEvents_;
    }

    /// @brief Download the most recent raw whole-key request ledger.
    ///
    /// @details
    /// Copies every pre-dedup whole-key hit. Positions remain zero pending the
    /// exact processor-order reconstruction stage.
    ///
    /// @param destination Caller-owned raw-request output array.
    /// @param capacity Destination element capacity.
    /// @return Number of request records copied.
    /// @invariant `capacity` covers the last result's raw-request count.
    uint32_t CudaPhase2GrowthBuffer::downloadRawRequests(
        DeviceRawGrowthRequest* destination,
        uint32_t capacity) const {
        assert(usedRawRequests_ <= capacity);
        if (usedRawRequests_ == 0) return 0;
        assert(destination != nullptr);
        GL_CUDA_ASSERT(cudaMemcpy(
            destination, deviceRawRequests_,
            static_cast<std::size_t>(usedRawRequests_)
                * sizeof(DeviceRawGrowthRequest),
            cudaMemcpyDeviceToHost));
        return usedRawRequests_;
    }

    /// @brief Download exact processor split-work tallies by task.
    ///
    /// @details
    /// Copies the fixed counter prefix belonging to the uploaded task image.
    /// Device growth increments a task counter only when the subkey owner probe
    /// accepts a node, excluding whole-key-only events exactly like
    /// `ExpressionAnalyzer::g_growthMatchCount`.
    ///
    /// @param destination Caller-owned task-count output array.
    /// @param capacity Destination element capacity.
    /// @return Number of task counts copied.
    /// @invariant `runRequestGrowth` completed and `capacity` covers the
    ///            uploaded task count.
    uint32_t CudaPhase2GrowthBuffer::downloadTaskSubkeyCounts(
        uint32_t* destination,
        uint32_t capacity) const {
        assert(usedTaskCount_ <= capacity);
        if (usedTaskCount_ == 0) return 0;
        assert(destination != nullptr);
        GL_CUDA_ASSERT(cudaMemcpy(
            destination, deviceTaskSubkeyCounts_,
            static_cast<std::size_t>(usedTaskCount_) * sizeof(uint32_t),
            cudaMemcpyDeviceToHost));
        return usedTaskCount_;
    }

    /// @brief Report the immutable bytes owned by the growth arrays.
    ///
    /// @details
    /// Recomputes the constructor's exact byte sum from fixed capacities and
    /// record sizes. The calculation is pure host arithmetic and excludes every
    /// allocation owned by other Phase 2 buffer classes.
    ///
    /// @return Exact startup allocation bytes owned by this growth buffer.
    /// @invariant The result is constant for the object's lifetime.
    uint64_t CudaPhase2GrowthBuffer::fixedAllocationBytes() const {
        return static_cast<uint64_t>(capacity_.calls)
                * sizeof(DevicePhase2GrowthCall)
            + static_cast<uint64_t>(capacity_.retainedRows) * 2
            + static_cast<uint64_t>(capacity_.frontierRecords)
                * sizeof(DeviceGrowthNode) * 2
            + static_cast<uint64_t>(capacity_.frontierRecords)
                * sizeof(DeviceGrowthPrefix)
            + static_cast<uint64_t>(capacity_.prefixPayloadValues)
                * sizeof(NameId)
            + static_cast<uint64_t>(capacity_.prefixVariableValues)
                * sizeof(NameId)
            + static_cast<uint64_t>(capacity_.prefixSecondaryValues)
                * sizeof(NameId)
            + static_cast<uint64_t>(capacity_.frontierRecords)
                * sizeof(uint32_t) * 4
            + static_cast<uint64_t>(capacity_.frontierRecords)
                * sizeof(uint64_t) * 4
            + static_cast<uint64_t>(capacity_.frontierRecords) * 2
            + static_cast<uint64_t>(capacity_.candidateWindowRecords)
                * sizeof(DeviceGrowthCandidateAttempt)
            + static_cast<uint64_t>(capacity_.candidateWindowRecords)
            + static_cast<uint64_t>(capacity_.candidateWindowRecords)
                * sizeof(uint32_t)
            + selectionTemporaryBytes_
            + static_cast<uint64_t>(capacity_.acceptedEvents)
                * sizeof(DeviceAcceptedGrowthEvent)
            + static_cast<uint64_t>(capacity_.rawRequests)
                * sizeof(DeviceRawGrowthRequest)
            + static_cast<uint64_t>(capacity_.calls) * sizeof(uint32_t)
            + sizeof(DeviceGrowthCounters);
    }

    /// @brief Allocate all exact-order and deduplication scratch once.
    ///
    /// @details
    /// Validates positive ceilings and a power-of-two deduplication table, queries
    /// event radix-sort, segmented-scan, and request radix-sort scratch at maximum
    /// occupancy, then allocates every reusable array plus one shared temporary
    /// region large enough for the largest primitive.
    ///
    /// @param fixedCapacity Immutable event, slot, and request ceilings.
    /// @return An empty reusable ordering owner.
    /// @invariant No later method changes an allocation address or capacity.
    CudaPhase2OrderingBuffer::CudaPhase2OrderingBuffer(
        Phase2OrderingCapacity fixedCapacity)
        : capacity_(fixedCapacity) {
        static_cast<void>(queryCudaDeviceContract());
        assert(capacity_.events > 0);
        assert(capacity_.deduplicationSlots > 0);
        assert((capacity_.deduplicationSlots
            & (capacity_.deduplicationSlots - 1)) == 0);
        assert(capacity_.requests > 0);
        assert(capacity_.events
            <= static_cast<uint32_t>(std::numeric_limits<int>::max()));
        assert(capacity_.requests
            <= static_cast<uint32_t>(std::numeric_limits<int>::max()));

        uint32_t* nullKeysInput = nullptr;
        uint32_t* nullKeysOutput = nullptr;
        uint32_t* nullIndicesInput = nullptr;
        uint32_t* nullIndicesOutput = nullptr;
        DeviceGrowthScanValue* nullScanInput = nullptr;
        DeviceGrowthScanValue* nullScanOutput = nullptr;
        DeviceOrderedRequestToken* nullTokenInput = nullptr;
        DeviceOrderedRequestToken* nullTokenOutput = nullptr;
        std::size_t eventSortBytes = 0;
        std::size_t scanBytes = 0;
        std::size_t requestSortBytes = 0;
        GL_CUDA_ASSERT(cub::DeviceRadixSort::SortPairs(
            nullptr, eventSortBytes,
            nullKeysInput, nullKeysOutput,
            nullIndicesInput, nullIndicesOutput,
            static_cast<int>(capacity_.events)));
        GL_CUDA_ASSERT(cub::DeviceScan::InclusiveScan(
            nullptr, scanBytes,
            nullScanInput, nullScanOutput,
            DeviceGrowthSegmentedAdd{},
            static_cast<int>(capacity_.events)));
        GL_CUDA_ASSERT(cub::DeviceRadixSort::SortPairs(
            nullptr, requestSortBytes,
            nullKeysInput, nullKeysOutput,
            nullTokenInput, nullTokenOutput,
            static_cast<int>(capacity_.requests)));
        temporaryBytes_ = eventSortBytes;
        if (scanBytes > temporaryBytes_) temporaryBytes_ = scanBytes;
        if (requestSortBytes > temporaryBytes_)
            temporaryBytes_ = requestSortBytes;
        assert(temporaryBytes_ > 0);

        const std::size_t eventWordBytes = static_cast<std::size_t>(
            capacity_.events) * sizeof(uint32_t);
        const std::size_t scanValueBytes = static_cast<std::size_t>(
            capacity_.events) * sizeof(DeviceGrowthScanValue);
        const std::size_t slotBytes = static_cast<std::size_t>(
            capacity_.deduplicationSlots) * sizeof(uint32_t);
        const std::size_t tokenBytes = static_cast<std::size_t>(
            capacity_.requests) * sizeof(DeviceOrderedRequestToken);
        for (uint32_t ordinal = 0; ordinal < 2; ++ordinal) {
            GL_CUDA_ASSERT(cudaMalloc(
                &deviceKeys_[ordinal], eventWordBytes));
            GL_CUDA_ASSERT(cudaMalloc(
                &deviceIndices_[ordinal], eventWordBytes));
            GL_CUDA_ASSERT(cudaMalloc(
                &deviceScanValues_[ordinal], scanValueBytes));
            GL_CUDA_ASSERT(cudaMalloc(
                &deviceRequestTokens_[ordinal], tokenBytes));
        }
        GL_CUDA_ASSERT(cudaMalloc(
            &deviceOrderedIndices_, eventWordBytes));
        GL_CUDA_ASSERT(cudaMalloc(
            &deviceDeduplicationOwners_, slotBytes));
        GL_CUDA_ASSERT(cudaMalloc(
            &deviceDeduplicationMinimumOrders_, slotBytes));
        GL_CUDA_ASSERT(cudaMalloc(
            &deviceUniqueRequestCount_, sizeof(uint32_t)));
        GL_CUDA_ASSERT(cudaMalloc(
            &deviceTemporary_, temporaryBytes_));
        for (uint32_t ordinal = 0; ordinal < 2; ++ordinal) {
            assert(deviceKeys_[ordinal] != nullptr);
            assert(deviceIndices_[ordinal] != nullptr);
            assert(deviceScanValues_[ordinal] != nullptr);
            assert(deviceRequestTokens_[ordinal] != nullptr);
        }
        assert(deviceOrderedIndices_ != nullptr);
        assert(deviceDeduplicationOwners_ != nullptr);
        assert(deviceDeduplicationMinimumOrders_ != nullptr);
        assert(deviceUniqueRequestCount_ != nullptr);
        assert(deviceTemporary_ != nullptr);
    }

    /// @brief Release every exact-order device allocation.
    ///
    /// @details
    /// Frees shared scratch, the unique counter, deduplication tables, preserved
    /// order, then both token, scan, index, and key arrays in reverse order.
    ///
    /// @return Nothing.
    CudaPhase2OrderingBuffer::~CudaPhase2OrderingBuffer() {
        GL_CUDA_ASSERT(cudaFree(deviceTemporary_));
        GL_CUDA_ASSERT(cudaFree(deviceUniqueRequestCount_));
        GL_CUDA_ASSERT(cudaFree(deviceDeduplicationMinimumOrders_));
        GL_CUDA_ASSERT(cudaFree(deviceDeduplicationOwners_));
        GL_CUDA_ASSERT(cudaFree(deviceOrderedIndices_));
        for (uint32_t ordinal = 2; ordinal > 0; --ordinal) {
            GL_CUDA_ASSERT(cudaFree(deviceRequestTokens_[ordinal - 1]));
            GL_CUDA_ASSERT(cudaFree(deviceScanValues_[ordinal - 1]));
            GL_CUDA_ASSERT(cudaFree(deviceIndices_[ordinal - 1]));
            GL_CUDA_ASSERT(cudaFree(deviceKeys_[ordinal - 1]));
        }
    }

    /// @brief Reconstruct processor event order and deduplicate requests.
    ///
    /// @details
    /// Runs eleven stable least-significant-first radix passes over path, run,
    /// call, and task components; preserves the final event permutation; assigns
    /// task-local growth positions by segmented inclusive scan; inserts every
    /// recordable event into a collision-exact per-call semantic hash table using
    /// atomic minimum order; emits one token per occupied slot; and radix-sorts
    /// those tokens by their retained event order.
    ///
    /// @param projection Uploaded resident logical-block image.
    /// @param tasks Uploaded task and batch image used by the growth calls.
    /// @param growth Completed unordered growth content.
    /// @return Ordered event and unique request counts.
    /// @invariant Growth content and its uploaded call schedule are unchanged.
    Phase2OrderingResult CudaPhase2OrderingBuffer::orderAndDeduplicate(
        const CudaPhase2ProjectionBuffer& projection,
        const CudaPhase2TaskBuffer& tasks,
        const CudaPhase2GrowthBuffer& growth) {
        assert(growth.usedAcceptedEvents_ <= capacity_.events);
        assert(growth.usedRawRequests_ <= capacity_.requests);
        assert(growth.usedRawRequests_
            <= capacity_.deduplicationSlots / 2);
        assert(projection.usedCounts_[0] > 0);
        assert(tasks.usedCounts_[0] > 0);
        usedEvents_ = growth.usedAcceptedEvents_;
        usedRequests_ = 0;
        GL_CUDA_ASSERT(cudaMemset(
            deviceUniqueRequestCount_, 0, sizeof(uint32_t)));
        if (usedEvents_ == 0) return Phase2OrderingResult{};
        assert(usedEvents_
            <= static_cast<uint32_t>(std::numeric_limits<int>::max()));

        ProjectionSemanticColumns columns{};
        columns.logicalBlocks = static_cast<
            const DeviceLogicalBlockProjection*>(projection.deviceColumns_[0]);
        columns.statements = static_cast<const IntEncodedExpr*>(
            projection.deviceColumns_[1]);
        constexpr uint32_t threads = 256;
        const uint32_t eventBlocks = (usedEvents_ + threads - 1) / threads;
        phase2OrderingInitializeIndicesKernel<<<eventBlocks, threads>>>(
            usedEvents_, static_cast<uint32_t*>(deviceIndices_[0]));
        GL_CUDA_ASSERT(cudaGetLastError());

        const uint32_t components[11] = {
            7, 6, 5, 4, 3, 2, 1, 0, 8, 9, 10
        };
        uint32_t currentIndices = 0;
        for (uint32_t pass = 0; pass < 11; ++pass) {
            const uint32_t nextIndices = currentIndices ^ 1u;
            phase2OrderingComponentKeysKernel<<<eventBlocks, threads>>>(
                static_cast<const DeviceAcceptedGrowthEvent*>(
                    growth.deviceAcceptedEvents_),
                static_cast<const DevicePhase2GrowthCall*>(
                    growth.deviceCalls_),
                static_cast<const uint32_t*>(
                    deviceIndices_[currentIndices]),
                usedEvents_, components[pass],
                static_cast<uint32_t*>(deviceKeys_[0]));
            GL_CUDA_ASSERT(cudaGetLastError());
            std::size_t sortBytes = temporaryBytes_;
            GL_CUDA_ASSERT(cub::DeviceRadixSort::SortPairs(
                deviceTemporary_, sortBytes,
                static_cast<const uint32_t*>(deviceKeys_[0]),
                static_cast<uint32_t*>(deviceKeys_[1]),
                static_cast<const uint32_t*>(
                    deviceIndices_[currentIndices]),
                static_cast<uint32_t*>(deviceIndices_[nextIndices]),
                static_cast<int>(usedEvents_)));
            currentIndices = nextIndices;
        }
        GL_CUDA_ASSERT(cudaMemcpy(
            deviceOrderedIndices_, deviceIndices_[currentIndices],
            static_cast<std::size_t>(usedEvents_) * sizeof(uint32_t),
            cudaMemcpyDeviceToDevice));

        phase2OrderingScanInputKernel<<<eventBlocks, threads>>>(
            static_cast<const DeviceAcceptedGrowthEvent*>(
                growth.deviceAcceptedEvents_),
            static_cast<const DevicePhase2GrowthCall*>(growth.deviceCalls_),
            static_cast<const uint32_t*>(deviceOrderedIndices_),
            usedEvents_,
            static_cast<DeviceGrowthScanValue*>(deviceScanValues_[0]));
        GL_CUDA_ASSERT(cudaGetLastError());
        std::size_t scanBytes = temporaryBytes_;
        GL_CUDA_ASSERT(cub::DeviceScan::InclusiveScan(
            deviceTemporary_, scanBytes,
            static_cast<const DeviceGrowthScanValue*>(deviceScanValues_[0]),
            static_cast<DeviceGrowthScanValue*>(deviceScanValues_[1]),
            DeviceGrowthSegmentedAdd{}, static_cast<int>(usedEvents_)));

        const std::size_t slotBytes = static_cast<std::size_t>(
            capacity_.deduplicationSlots) * sizeof(uint32_t);
        GL_CUDA_ASSERT(cudaMemset(
            deviceDeduplicationOwners_, 0xff, slotBytes));
        GL_CUDA_ASSERT(cudaMemset(
            deviceDeduplicationMinimumOrders_, 0xff, slotBytes));
        phase2DeduplicateRequestsKernel<<<eventBlocks, threads>>>(
            columns,
            static_cast<const DevicePhase2Task*>(tasks.deviceColumns_[0]),
            static_cast<const DevicePhase2GrowthCall*>(growth.deviceCalls_),
            static_cast<const DeviceAcceptedGrowthEvent*>(
                growth.deviceAcceptedEvents_),
            static_cast<const uint32_t*>(deviceOrderedIndices_),
            usedEvents_,
            static_cast<uint32_t*>(deviceDeduplicationOwners_),
            static_cast<uint32_t*>(deviceDeduplicationMinimumOrders_),
            capacity_.deduplicationSlots);
        GL_CUDA_ASSERT(cudaGetLastError());
        const uint32_t slotBlocks =
            (capacity_.deduplicationSlots + threads - 1) / threads;
        phase2EmitUniqueRequestsKernel<<<slotBlocks, threads>>>(
            static_cast<const uint32_t*>(deviceOrderedIndices_),
            static_cast<const DeviceGrowthScanValue*>(deviceScanValues_[1]),
            static_cast<const uint32_t*>(deviceDeduplicationOwners_),
            static_cast<const uint32_t*>(
                deviceDeduplicationMinimumOrders_),
            capacity_.deduplicationSlots,
            static_cast<DeviceOrderedRequestToken*>(deviceRequestTokens_[0]),
            capacity_.requests,
            static_cast<uint32_t*>(deviceUniqueRequestCount_));
        GL_CUDA_ASSERT(cudaGetLastError());
        GL_CUDA_ASSERT(cudaDeviceSynchronize());
        GL_CUDA_ASSERT(cudaMemcpy(
            &usedRequests_, deviceUniqueRequestCount_, sizeof(usedRequests_),
            cudaMemcpyDeviceToHost));
        assert(usedRequests_ <= growth.usedRawRequests_);
        if (usedRequests_ > 0) {
            const uint32_t requestBlocks =
                (usedRequests_ + threads - 1) / threads;
            phase2RequestOrderKeysKernel<<<requestBlocks, threads>>>(
                static_cast<const DeviceOrderedRequestToken*>(
                    deviceRequestTokens_[0]),
                usedRequests_, static_cast<uint32_t*>(deviceKeys_[0]));
            GL_CUDA_ASSERT(cudaGetLastError());
            std::size_t requestSortBytes = temporaryBytes_;
            GL_CUDA_ASSERT(cub::DeviceRadixSort::SortPairs(
                deviceTemporary_, requestSortBytes,
                static_cast<const uint32_t*>(deviceKeys_[0]),
                static_cast<uint32_t*>(deviceKeys_[1]),
                static_cast<const DeviceOrderedRequestToken*>(
                    deviceRequestTokens_[0]),
                static_cast<DeviceOrderedRequestToken*>(
                    deviceRequestTokens_[1]),
                static_cast<int>(usedRequests_)));
            GL_CUDA_ASSERT(cudaDeviceSynchronize());
        }
        return Phase2OrderingResult{ usedEvents_, usedRequests_ };
    }

    /// @brief Download the exact ordered event-index permutation.
    ///
    /// @param destination Caller-owned event-index output.
    /// @param capacity Destination element capacity.
    /// @return Number of ordered indices copied.
    /// @invariant `capacity` covers the last ordered event count.
    uint32_t CudaPhase2OrderingBuffer::downloadOrderedEventIndices(
        uint32_t* destination,
        uint32_t capacity) const {
        assert(usedEvents_ <= capacity);
        if (usedEvents_ == 0) return 0;
        assert(destination != nullptr);
        GL_CUDA_ASSERT(cudaMemcpy(
            destination, deviceOrderedIndices_,
            static_cast<std::size_t>(usedEvents_) * sizeof(uint32_t),
            cudaMemcpyDeviceToHost));
        return usedEvents_;
    }

    /// @brief Download unique request tokens in exact processor stream order.
    ///
    /// @param destination Caller-owned ordered-token output.
    /// @param capacity Destination element capacity.
    /// @return Number of unique request tokens copied.
    /// @invariant `capacity` covers the last unique request count.
    uint32_t CudaPhase2OrderingBuffer::downloadOrderedRequests(
        DeviceOrderedRequestToken* destination,
        uint32_t capacity) const {
        assert(usedRequests_ <= capacity);
        if (usedRequests_ == 0) return 0;
        assert(destination != nullptr);
        GL_CUDA_ASSERT(cudaMemcpy(
            destination, deviceRequestTokens_[1],
            static_cast<std::size_t>(usedRequests_)
                * sizeof(DeviceOrderedRequestToken),
            cudaMemcpyDeviceToHost));
        return usedRequests_;
    }

    /// @brief Report every fixed ordering allocation including CUB scratch.
    ///
    /// @return Exact startup bytes owned by this ordering buffer.
    /// @invariant The value is constant for the object's lifetime.
    uint64_t CudaPhase2OrderingBuffer::fixedAllocationBytes() const {
        return static_cast<uint64_t>(capacity_.events)
                * sizeof(uint32_t) * 5
            + static_cast<uint64_t>(capacity_.events)
                * sizeof(DeviceGrowthScanValue) * 2
            + static_cast<uint64_t>(capacity_.deduplicationSlots)
                * sizeof(uint32_t) * 2
            + static_cast<uint64_t>(capacity_.requests)
                * sizeof(DeviceOrderedRequestToken) * 2
            + sizeof(uint32_t)
            + static_cast<uint64_t>(temporaryBytes_);
    }

    /// @brief Allocate every retained evaluator work and output array once.
    ///
    /// @details
    /// Validates the retained next-power-of-two ceilings, queries all three
    /// exclusive-scan workloads at maximum occupancy, allocates request-to-value
    /// work, complete firing/provenance output arenas, canonical/doom arrays, two
    /// counters, and one shared scratch region sized for the largest scan.
    ///
    /// @param fixedCapacity Immutable evaluator work and output ceilings.
    /// @return An empty reusable evaluator owner.
    /// @invariant No later method changes an allocation address or capacity.
    CudaPhase2EvaluationBuffer::CudaPhase2EvaluationBuffer(
        Phase2EvaluationCapacity fixedCapacity)
        : capacity_(fixedCapacity) {
        static_cast<void>(queryCudaDeviceContract());
        const uint32_t capacities[13] = {
            capacity_.logicalBlocks, capacity_.requests, capacity_.reverseOwners,
            capacity_.candidateOwners, capacity_.encodedHits,
            capacity_.localValues, capacity_.firingRecords,
            capacity_.generatedBytes, capacity_.levelValues,
            capacity_.originDependencies, capacity_.markerKeys,
            capacity_.markerRemainingArgs, capacity_.markerArgs
        };
        for (uint32_t value : capacities) {
            assert(value > 0);
            assert((value & (value - 1)) == 0);
            assert(value <= static_cast<uint32_t>(
                std::numeric_limits<int>::max()));
        }
        assert(capacity_.candidateOwners <= capacity_.reverseOwners);
        assert(capacity_.encodedHits <= capacity_.candidateOwners);
        assert(capacity_.localValues >= capacity_.encodedHits);

        uint32_t* nullInput = nullptr;
        uint32_t* nullOutput = nullptr;
        std::size_t requestScanBytes = 0;
        std::size_t valueScanBytes = 0;
        std::size_t doomScanBytes = 0;
        GL_CUDA_ASSERT(cub::DeviceScan::ExclusiveSum(
            nullptr, requestScanBytes, nullInput, nullOutput,
            static_cast<int>(capacity_.requests)));
        GL_CUDA_ASSERT(cub::DeviceScan::ExclusiveSum(
            nullptr, valueScanBytes, nullInput, nullOutput,
            static_cast<int>(capacity_.encodedHits)));
        GL_CUDA_ASSERT(cub::DeviceScan::ExclusiveSum(
            nullptr, doomScanBytes, nullInput, nullOutput,
            static_cast<int>(capacity_.firingRecords)));
        temporaryBytes_ = requestScanBytes > valueScanBytes
            ? requestScanBytes : valueScanBytes;
        if (doomScanBytes > temporaryBytes_) temporaryBytes_ = doomScanBytes;
        assert(temporaryBytes_ > 0);

        GL_CUDA_ASSERT(cudaMalloc(&deviceRequestStates_,
            static_cast<std::size_t>(capacity_.requests)
                * sizeof(DeviceEvaluationRequestState)));
        GL_CUDA_ASSERT(cudaMalloc(&deviceRequestCounts_,
            static_cast<std::size_t>(capacity_.requests) * sizeof(uint32_t)));
        GL_CUDA_ASSERT(cudaMalloc(&deviceRequestOffsets_,
            static_cast<std::size_t>(capacity_.requests) * sizeof(uint32_t)));
        GL_CUDA_ASSERT(cudaMalloc(&deviceOwnerWork_,
            static_cast<std::size_t>(capacity_.reverseOwners)
                * sizeof(DeviceEvaluationOwnerWork)));
        GL_CUDA_ASSERT(cudaMalloc(&deviceCandidates_,
            static_cast<std::size_t>(capacity_.candidateOwners)
                * sizeof(DeviceEvaluationCandidate)));
        GL_CUDA_ASSERT(cudaMalloc(&deviceEncodedHits_,
            static_cast<std::size_t>(capacity_.encodedHits)
                * sizeof(DeviceEvaluationCandidate)));
        GL_CUDA_ASSERT(cudaMalloc(&deviceValueCounts_,
            static_cast<std::size_t>(capacity_.encodedHits)
                * sizeof(uint32_t)));
        GL_CUDA_ASSERT(cudaMalloc(&deviceValueOffsets_,
            static_cast<std::size_t>(capacity_.encodedHits)
                * sizeof(uint32_t)));
        GL_CUDA_ASSERT(cudaMalloc(&deviceValueWork_,
            static_cast<std::size_t>(capacity_.localValues)
                * sizeof(DeviceEvaluationValueWork)));
        GL_CUDA_ASSERT(cudaMalloc(&deviceFiringRecords_,
            static_cast<std::size_t>(capacity_.firingRecords)
                * sizeof(DevicePhase2FiringRecord)));
        GL_CUDA_ASSERT(cudaMalloc(&deviceGeneratedBytes_,
            capacity_.generatedBytes));
        GL_CUDA_ASSERT(cudaMalloc(&deviceLevelValues_,
            static_cast<std::size_t>(capacity_.levelValues) * sizeof(int32_t)));
        GL_CUDA_ASSERT(cudaMalloc(&deviceOriginDependencies_,
            static_cast<std::size_t>(capacity_.originDependencies)
                * sizeof(DeviceEvaluationDependency)));
        GL_CUDA_ASSERT(cudaMalloc(&deviceMarkerKeys_,
            static_cast<std::size_t>(capacity_.markerKeys)
                * sizeof(DeviceEvaluationByteSlice)));
        GL_CUDA_ASSERT(cudaMalloc(&deviceMarkerRemainingArgs_,
            static_cast<std::size_t>(capacity_.markerRemainingArgs)
                * sizeof(int32_t)));
        GL_CUDA_ASSERT(cudaMalloc(&deviceMarkerArgs_,
            static_cast<std::size_t>(capacity_.markerArgs)
                * sizeof(DeviceEvaluationByteSlice)));
        GL_CUDA_ASSERT(cudaMalloc(&deviceFiringOrderA_,
            static_cast<std::size_t>(capacity_.firingRecords)
                * sizeof(uint32_t)));
        GL_CUDA_ASSERT(cudaMalloc(&deviceFiringOrderB_,
            static_cast<std::size_t>(capacity_.firingRecords)
                * sizeof(uint32_t)));
        deviceFiringOrderResult_ = deviceFiringOrderA_;
        GL_CUDA_ASSERT(cudaMalloc(&deviceDoomLines_,
            static_cast<std::size_t>(capacity_.logicalBlocks)
                * sizeof(int64_t)));
        GL_CUDA_ASSERT(cudaMalloc(&deviceDoomRequestIndices_,
            static_cast<std::size_t>(capacity_.logicalBlocks)
                * sizeof(uint32_t)));
        GL_CUDA_ASSERT(cudaMalloc(
            &deviceSelectedCount_, sizeof(uint32_t)));
        GL_CUDA_ASSERT(cudaMalloc(
            &deviceCounters_, sizeof(DeviceEvaluationCounters)));
        GL_CUDA_ASSERT(cudaMalloc(&deviceTemporary_, temporaryBytes_));
        assert(deviceRequestStates_ != nullptr);
        assert(deviceRequestCounts_ != nullptr);
        assert(deviceRequestOffsets_ != nullptr);
        assert(deviceOwnerWork_ != nullptr);
        assert(deviceCandidates_ != nullptr);
        assert(deviceEncodedHits_ != nullptr);
        assert(deviceValueCounts_ != nullptr);
        assert(deviceValueOffsets_ != nullptr);
        assert(deviceValueWork_ != nullptr);
        assert(deviceFiringRecords_ != nullptr);
        assert(deviceGeneratedBytes_ != nullptr);
        assert(deviceLevelValues_ != nullptr);
        assert(deviceOriginDependencies_ != nullptr);
        assert(deviceMarkerKeys_ != nullptr);
        assert(deviceMarkerRemainingArgs_ != nullptr);
        assert(deviceMarkerArgs_ != nullptr);
        assert(deviceFiringOrderA_ != nullptr);
        assert(deviceFiringOrderB_ != nullptr);
        assert(deviceFiringOrderResult_ != nullptr);
        assert(deviceDoomLines_ != nullptr);
        assert(deviceDoomRequestIndices_ != nullptr);
        assert(deviceSelectedCount_ != nullptr);
        assert(deviceCounters_ != nullptr);
        assert(deviceTemporary_ != nullptr);
    }

    /// @brief Release every fixed evaluator allocation exactly once.
    ///
    /// @details
    /// Frees shared scratch, counters, variable output arenas, firing headers,
    /// value work, encoded/candidate work, owner work, and request columns in
    /// reverse ownership order.
    ///
    /// @return Nothing.
    CudaPhase2EvaluationBuffer::~CudaPhase2EvaluationBuffer() {
        GL_CUDA_ASSERT(cudaFree(deviceTemporary_));
        GL_CUDA_ASSERT(cudaFree(deviceCounters_));
        GL_CUDA_ASSERT(cudaFree(deviceSelectedCount_));
        GL_CUDA_ASSERT(cudaFree(deviceDoomRequestIndices_));
        GL_CUDA_ASSERT(cudaFree(deviceDoomLines_));
        GL_CUDA_ASSERT(cudaFree(deviceFiringOrderB_));
        GL_CUDA_ASSERT(cudaFree(deviceFiringOrderA_));
        GL_CUDA_ASSERT(cudaFree(deviceMarkerArgs_));
        GL_CUDA_ASSERT(cudaFree(deviceMarkerRemainingArgs_));
        GL_CUDA_ASSERT(cudaFree(deviceMarkerKeys_));
        GL_CUDA_ASSERT(cudaFree(deviceOriginDependencies_));
        GL_CUDA_ASSERT(cudaFree(deviceLevelValues_));
        GL_CUDA_ASSERT(cudaFree(deviceGeneratedBytes_));
        GL_CUDA_ASSERT(cudaFree(deviceFiringRecords_));
        GL_CUDA_ASSERT(cudaFree(deviceValueWork_));
        GL_CUDA_ASSERT(cudaFree(deviceValueOffsets_));
        GL_CUDA_ASSERT(cudaFree(deviceValueCounts_));
        GL_CUDA_ASSERT(cudaFree(deviceEncodedHits_));
        GL_CUDA_ASSERT(cudaFree(deviceCandidates_));
        GL_CUDA_ASSERT(cudaFree(deviceOwnerWork_));
        GL_CUDA_ASSERT(cudaFree(deviceRequestOffsets_));
        GL_CUDA_ASSERT(cudaFree(deviceRequestCounts_));
        GL_CUDA_ASSERT(cudaFree(deviceRequestStates_));
    }

    /// @brief Expand exact ordered requests through projected LMV work.
    ///
    /// @details
    /// Applies dependency and request-validity gates, scans reverse-owner runs,
    /// compacts remaining-argument subsets with capacity-checked atomics, builds
    /// ignore-u normalized keys, compacts encoded hits, scans LocalMemoryValue
    /// runs, and emits flat request/blob work. Every semantic decision runs on
    /// projected device state; the host observes only bounded used counts.
    ///
    /// @param projection Uploaded resident logical-block image.
    /// @param tasks Uploaded executor task image.
    /// @param growth Completed growth content and call schedule.
    /// @param ordering Exact ordered unique request tokens.
    /// @return Used counts at every work-expansion boundary.
    /// @invariant All inputs belong to the same uploaded Phase 2 sweep.
    Phase2EvaluationResult CudaPhase2EvaluationBuffer::expandEvaluationWork(
        const CudaPhase2ProjectionBuffer& projection,
        const CudaPhase2TaskBuffer& tasks,
        const CudaPhase2GrowthBuffer& growth,
        const CudaPhase2OrderingBuffer& ordering) {
        assert(ordering.usedRequests_ <= capacity_.requests);
        assert(projection.usedCounts_[0] > 0);
        assert(tasks.usedCounts_[0] > 0);
        assert(growth.usedAcceptedEvents_ > 0 || ordering.usedRequests_ == 0);
        usedCandidates_ = 0;
        usedEncodedHits_ = 0;
        usedValues_ = 0;
        usedFiringRecords_ = 0;
        usedGeneratedBytes_ = 0;
        usedLevelValues_ = 0;
        usedOriginDependencies_ = 0;
        usedMarkerKeys_ = 0;
        usedMarkerRemainingArgs_ = 0;
        usedMarkerArgs_ = 0;
        usedFiringOrder_ = 0;
        usedDoomLines_ = 0;
        deviceFiringOrderResult_ = deviceFiringOrderA_;
        GL_CUDA_ASSERT(cudaMemset(
            deviceCounters_, 0, sizeof(DeviceEvaluationCounters)));
        GL_CUDA_ASSERT(cudaMemset(
            deviceSelectedCount_, 0, sizeof(uint32_t)));
        if (ordering.usedRequests_ == 0) return Phase2EvaluationResult{};

        ProjectionSemanticColumns columns{};
        columns.logicalBlocks = static_cast<
            const DeviceLogicalBlockProjection*>(projection.deviceColumns_[0]);
        columns.statements = static_cast<const IntEncodedExpr*>(
            projection.deviceColumns_[1]);
        columns.nameRecords = static_cast<const DeviceNameRecord*>(
            projection.deviceColumns_[2]);
        columns.nameBytes = static_cast<const char*>(projection.deviceColumns_[3]);
        columns.nameSlots = static_cast<const int32_t*>(
            projection.deviceColumns_[4]);
        columns.ruleStringRecords = static_cast<const DeviceRuleStringRecord*>(
            projection.deviceColumns_[5]);
        columns.ruleStringBytes = static_cast<const char*>(
            projection.deviceColumns_[6]);
        columns.byteMapViews = static_cast<const DeviceByteMapView*>(
            projection.deviceColumns_[7]);
        columns.byteMapEntries = static_cast<const DeviceByteMapEntry*>(
            projection.deviceColumns_[8]);
        columns.byteMapSlots = static_cast<const int32_t*>(
            projection.deviceColumns_[9]);
        columns.byteKeyBytes = static_cast<const char*>(
            projection.deviceColumns_[10]);
        columns.blobRecords = static_cast<const DeviceBlobRecord*>(
            projection.deviceColumns_[11]);
        columns.blobBytes = static_cast<const char*>(
            projection.deviceColumns_[12]);
        columns.reverseMapViews = static_cast<const DeviceReverseMapView*>(
            projection.deviceColumns_[13]);
        columns.reverseMapEntries = static_cast<const DeviceReverseMapEntry*>(
            projection.deviceColumns_[14]);
        columns.reverseMapSlots = static_cast<const int32_t*>(
            projection.deviceColumns_[15]);
        columns.reverseKeyBytes = static_cast<const char*>(
            projection.deviceColumns_[16]);
        columns.reverseOwners = static_cast<const int32_t*>(
            projection.deviceColumns_[17]);
        columns.podMapViews = static_cast<const DevicePodMapView*>(
            projection.deviceColumns_[18]);
        columns.podMapEntries = static_cast<const DevicePodMapEntry*>(
            projection.deviceColumns_[19]);
        columns.podMapSlots = static_cast<const int32_t*>(
            projection.deviceColumns_[20]);
        columns.podRunValues = static_cast<const int32_t*>(
            projection.deviceColumns_[21]);
        columns.mandatoryStatementKeys = static_cast<const int64_t*>(
            projection.deviceColumns_[22]);
        columns.metadataBytes = static_cast<const char*>(
            projection.deviceColumns_[23]);

        constexpr uint32_t threads = 256;
        const uint32_t requestBlocks =
            (ordering.usedRequests_ + threads - 1) / threads;
        phase2EvaluationRequestKernel<<<requestBlocks, threads>>>(
            columns,
            static_cast<const DevicePhase2Task*>(tasks.deviceColumns_[0]),
            static_cast<const DevicePhase2GrowthCall*>(growth.deviceCalls_),
            static_cast<const DeviceAcceptedGrowthEvent*>(
                growth.deviceAcceptedEvents_),
            static_cast<const DeviceOrderedRequestToken*>(
                ordering.deviceRequestTokens_[1]),
            ordering.usedRequests_,
            static_cast<DeviceEvaluationRequestState*>(deviceRequestStates_),
            static_cast<uint32_t*>(deviceRequestCounts_),
            static_cast<DeviceEvaluationCounters*>(deviceCounters_));
        GL_CUDA_ASSERT(cudaGetLastError());
        std::size_t requestScanBytes = temporaryBytes_;
        GL_CUDA_ASSERT(cub::DeviceScan::ExclusiveSum(
            deviceTemporary_, requestScanBytes,
            static_cast<const uint32_t*>(deviceRequestCounts_),
            static_cast<uint32_t*>(deviceRequestOffsets_),
            static_cast<int>(ordering.usedRequests_)));
        GL_CUDA_ASSERT(cudaDeviceSynchronize());
        DeviceEvaluationCounters hostCounters{};
        GL_CUDA_ASSERT(cudaMemcpy(
            &hostCounters, deviceCounters_, sizeof(hostCounters),
            cudaMemcpyDeviceToHost));
        assert(hostCounters.reverseOwnerCount <= capacity_.reverseOwners);

        if (hostCounters.reverseOwnerCount > 0) {
            phase2EvaluationEmitOwnersKernel<<<requestBlocks, threads>>>(
                columns,
                static_cast<const DeviceEvaluationRequestState*>(
                    deviceRequestStates_),
                static_cast<const uint32_t*>(deviceRequestOffsets_),
                ordering.usedRequests_,
                static_cast<DeviceEvaluationOwnerWork*>(deviceOwnerWork_));
            GL_CUDA_ASSERT(cudaGetLastError());
            GL_CUDA_ASSERT(cudaMemset(
                deviceSelectedCount_, 0, sizeof(uint32_t)));
            const uint32_t ownerBlocks =
                (hostCounters.reverseOwnerCount + threads - 1) / threads;
            phase2EvaluationCandidateKernel<<<ownerBlocks, threads>>>(
                columns,
                static_cast<const DevicePhase2Task*>(tasks.deviceColumns_[0]),
                static_cast<const DevicePhase2GrowthCall*>(growth.deviceCalls_),
                static_cast<const DeviceAcceptedGrowthEvent*>(
                    growth.deviceAcceptedEvents_),
                static_cast<const DeviceOrderedRequestToken*>(
                    ordering.deviceRequestTokens_[1]),
                static_cast<const DeviceEvaluationOwnerWork*>(deviceOwnerWork_),
                hostCounters.reverseOwnerCount,
                static_cast<DeviceEvaluationCandidate*>(deviceCandidates_),
                capacity_.candidateOwners,
                static_cast<uint32_t*>(deviceSelectedCount_));
            GL_CUDA_ASSERT(cudaGetLastError());
            GL_CUDA_ASSERT(cudaDeviceSynchronize());
            GL_CUDA_ASSERT(cudaMemcpy(
                &usedCandidates_, deviceSelectedCount_, sizeof(uint32_t),
                cudaMemcpyDeviceToHost));
            assert(usedCandidates_ <= capacity_.candidateOwners);
        }

        if (usedCandidates_ > 0) {
            GL_CUDA_ASSERT(cudaMemset(
                deviceSelectedCount_, 0, sizeof(uint32_t)));
            const uint32_t candidateBlocks =
                (usedCandidates_ + threads - 1) / threads;
            phase2EvaluationCompactEncodedKernel<<<candidateBlocks, threads>>>(
                static_cast<const DeviceEvaluationCandidate*>(deviceCandidates_),
                usedCandidates_,
                static_cast<DeviceEvaluationCandidate*>(deviceEncodedHits_),
                capacity_.encodedHits,
                static_cast<uint32_t*>(deviceSelectedCount_));
            GL_CUDA_ASSERT(cudaGetLastError());
            GL_CUDA_ASSERT(cudaDeviceSynchronize());
            GL_CUDA_ASSERT(cudaMemcpy(
                &usedEncodedHits_, deviceSelectedCount_, sizeof(uint32_t),
                cudaMemcpyDeviceToHost));
            assert(usedEncodedHits_ <= capacity_.encodedHits);
        }

        if (usedEncodedHits_ > 0) {
            const uint32_t hitBlocks =
                (usedEncodedHits_ + threads - 1) / threads;
            phase2EvaluationValueCountsKernel<<<hitBlocks, threads>>>(
                columns,
                static_cast<const DeviceEvaluationCandidate*>(
                    deviceEncodedHits_),
                usedEncodedHits_,
                static_cast<uint32_t*>(deviceValueCounts_),
                static_cast<DeviceEvaluationCounters*>(deviceCounters_));
            GL_CUDA_ASSERT(cudaGetLastError());
            std::size_t valueScanBytes = temporaryBytes_;
            GL_CUDA_ASSERT(cub::DeviceScan::ExclusiveSum(
                deviceTemporary_, valueScanBytes,
                static_cast<const uint32_t*>(deviceValueCounts_),
                static_cast<uint32_t*>(deviceValueOffsets_),
                static_cast<int>(usedEncodedHits_)));
            GL_CUDA_ASSERT(cudaDeviceSynchronize());
            GL_CUDA_ASSERT(cudaMemcpy(
                &hostCounters, deviceCounters_, sizeof(hostCounters),
                cudaMemcpyDeviceToHost));
            assert(hostCounters.localValueCount <= capacity_.localValues);
            usedValues_ = hostCounters.localValueCount;
            phase2EvaluationEmitValuesKernel<<<hitBlocks, threads>>>(
                columns,
                static_cast<const DeviceEvaluationCandidate*>(
                    deviceEncodedHits_),
                static_cast<const uint32_t*>(deviceValueOffsets_),
                usedEncodedHits_,
                static_cast<DeviceEvaluationValueWork*>(deviceValueWork_));
            GL_CUDA_ASSERT(cudaGetLastError());
            GL_CUDA_ASSERT(cudaDeviceSynchronize());
        }
        return Phase2EvaluationResult{
            ordering.usedRequests_, hostCounters.dependencyPassCount,
            hostCounters.reverseOwnerCount, usedCandidates_,
            usedEncodedHits_, usedValues_ };
    }

    /// @brief Materialize byte-exact substituted firing expressions on the GPU.
    ///
    /// @details
    /// Consumes the selected LMV prefix from the preceding expansion, applies
    /// rule-scope and source-type gates, reconstructs normalized-variable reverse
    /// maps, and writes transformed expressions plus source classification into
    /// fixed firing arenas. It also merges sorted-unique premise/rule levels and
    /// writes source-first, decoded-premise-ordered head provenance. Only the four
    /// output counters are cleared; expansion counts remain available for
    /// diagnostics.
    ///
    /// @param projection Uploaded resident logical-block image.
    /// @param tasks Uploaded executor task image.
    /// @param growth Completed growth content and call schedule.
    /// @param ordering Exact ordered unique request tokens.
    /// @return Exact firing-record, generated-byte, level, and provenance counts.
    /// @invariant The four inputs are the same objects passed to the immediately
    ///            preceding `expandEvaluationWork` call.
    Phase2FiringExpressionResult
    CudaPhase2EvaluationBuffer::materializeFiringExpressions(
        const CudaPhase2ProjectionBuffer& projection,
        const CudaPhase2TaskBuffer& tasks,
        const CudaPhase2GrowthBuffer& growth,
        const CudaPhase2OrderingBuffer& ordering) {
        assert(usedValues_ <= capacity_.localValues);
        assert(ordering.usedRequests_ <= capacity_.requests);
        usedFiringRecords_ = 0;
        usedGeneratedBytes_ = 0;
        usedLevelValues_ = 0;
        usedOriginDependencies_ = 0;
        usedMarkerKeys_ = 0;
        usedMarkerRemainingArgs_ = 0;
        usedMarkerArgs_ = 0;
        GL_CUDA_ASSERT(cudaMemset(
            static_cast<char*>(deviceCounters_)
                + offsetof(DeviceEvaluationCounters, firingRecordCount),
            0, sizeof(uint32_t) * 7));
        if (usedValues_ == 0) return Phase2FiringExpressionResult{};

        ProjectionSemanticColumns columns{};
        columns.logicalBlocks = static_cast<
            const DeviceLogicalBlockProjection*>(projection.deviceColumns_[0]);
        columns.statements = static_cast<const IntEncodedExpr*>(
            projection.deviceColumns_[1]);
        columns.nameRecords = static_cast<const DeviceNameRecord*>(
            projection.deviceColumns_[2]);
        columns.nameBytes = static_cast<const char*>(projection.deviceColumns_[3]);
        columns.nameSlots = static_cast<const int32_t*>(
            projection.deviceColumns_[4]);
        columns.ruleStringRecords = static_cast<const DeviceRuleStringRecord*>(
            projection.deviceColumns_[5]);
        columns.ruleStringBytes = static_cast<const char*>(
            projection.deviceColumns_[6]);
        columns.byteMapViews = static_cast<const DeviceByteMapView*>(
            projection.deviceColumns_[7]);
        columns.byteMapEntries = static_cast<const DeviceByteMapEntry*>(
            projection.deviceColumns_[8]);
        columns.byteMapSlots = static_cast<const int32_t*>(
            projection.deviceColumns_[9]);
        columns.byteKeyBytes = static_cast<const char*>(
            projection.deviceColumns_[10]);
        columns.blobRecords = static_cast<const DeviceBlobRecord*>(
            projection.deviceColumns_[11]);
        columns.blobBytes = static_cast<const char*>(
            projection.deviceColumns_[12]);
        columns.reverseMapViews = static_cast<const DeviceReverseMapView*>(
            projection.deviceColumns_[13]);
        columns.reverseMapEntries = static_cast<const DeviceReverseMapEntry*>(
            projection.deviceColumns_[14]);
        columns.reverseMapSlots = static_cast<const int32_t*>(
            projection.deviceColumns_[15]);
        columns.reverseKeyBytes = static_cast<const char*>(
            projection.deviceColumns_[16]);
        columns.reverseOwners = static_cast<const int32_t*>(
            projection.deviceColumns_[17]);
        columns.podMapViews = static_cast<const DevicePodMapView*>(
            projection.deviceColumns_[18]);
        columns.podMapEntries = static_cast<const DevicePodMapEntry*>(
            projection.deviceColumns_[19]);
        columns.podMapSlots = static_cast<const int32_t*>(
            projection.deviceColumns_[20]);
        columns.podRunValues = static_cast<const int32_t*>(
            projection.deviceColumns_[21]);
        columns.mandatoryStatementKeys = static_cast<const int64_t*>(
            projection.deviceColumns_[22]);
        columns.metadataBytes = static_cast<const char*>(
            projection.deviceColumns_[23]);

        constexpr uint32_t threads = 256;
        const uint32_t blocks = (usedValues_ + threads - 1) / threads;
        phase2FiringExpressionKernel<<<blocks, threads>>>(
            columns,
            static_cast<const DevicePhase2Task*>(tasks.deviceColumns_[0]),
            static_cast<const DevicePhase2GrowthCall*>(growth.deviceCalls_),
            static_cast<const DeviceAcceptedGrowthEvent*>(
                growth.deviceAcceptedEvents_),
            static_cast<const DeviceOrderedRequestToken*>(
                ordering.deviceRequestTokens_[1]),
            static_cast<const DeviceEvaluationRequestState*>(
                deviceRequestStates_),
            static_cast<const DeviceEvaluationValueWork*>(deviceValueWork_),
            usedValues_,
            static_cast<DevicePhase2FiringRecord*>(deviceFiringRecords_),
            capacity_.firingRecords,
            static_cast<char*>(deviceGeneratedBytes_),
            capacity_.generatedBytes,
            static_cast<int32_t*>(deviceLevelValues_),
            capacity_.levelValues,
            static_cast<DeviceEvaluationDependency*>(
                deviceOriginDependencies_),
            capacity_.originDependencies,
            static_cast<DeviceEvaluationByteSlice*>(deviceMarkerKeys_),
            capacity_.markerKeys,
            static_cast<int32_t*>(deviceMarkerRemainingArgs_),
            capacity_.markerRemainingArgs,
            static_cast<DeviceEvaluationByteSlice*>(deviceMarkerArgs_),
            capacity_.markerArgs,
            static_cast<DeviceEvaluationCounters*>(deviceCounters_));
        GL_CUDA_ASSERT(cudaGetLastError());
        GL_CUDA_ASSERT(cudaDeviceSynchronize());
        DeviceEvaluationCounters hostCounters{};
        GL_CUDA_ASSERT(cudaMemcpy(
            &hostCounters, deviceCounters_, sizeof(hostCounters),
            cudaMemcpyDeviceToHost));
        assert(hostCounters.firingRecordCount <= capacity_.firingRecords);
        assert(hostCounters.generatedByteCount <= capacity_.generatedBytes);
        assert(hostCounters.levelValueCount <= capacity_.levelValues);
        assert(hostCounters.originDependencyCount
            <= capacity_.originDependencies);
        assert(hostCounters.markerKeyCount <= capacity_.markerKeys);
        assert(hostCounters.markerRemainingArgCount
            <= capacity_.markerRemainingArgs);
        assert(hostCounters.markerArgCount <= capacity_.markerArgs);
        usedFiringRecords_ = hostCounters.firingRecordCount;
        usedGeneratedBytes_ = hostCounters.generatedByteCount;
        usedLevelValues_ = hostCounters.levelValueCount;
        usedOriginDependencies_ = hostCounters.originDependencyCount;
        usedMarkerKeys_ = hostCounters.markerKeyCount;
        usedMarkerRemainingArgs_ = hostCounters.markerRemainingArgCount;
        usedMarkerArgs_ = hostCounters.markerArgCount;
        return Phase2FiringExpressionResult{
            usedFiringRecords_, usedGeneratedBytes_,
            usedLevelValues_, usedOriginDependencies_,
            usedMarkerKeys_, usedMarkerRemainingArgs_, usedMarkerArgs_ };
    }

    /// @brief Canonically order complete device firing records by content.
    ///
    /// @details
    /// Initializes one index per materialized record, then runs power-of-two
    /// parallel merge passes over two fixed index arrays. The device comparator
    /// groups logical blocks and reproduces `applyFiringRecords`' complete
    /// per-kind content order; variable payloads are compared in place and never
    /// moved or downloaded for sorting.
    ///
    /// @param projection Uploaded resident image that decodes local identifiers.
    /// @return Number of canonical firing indices.
    /// @invariant The last firing materialization used this same projection.
    uint32_t CudaPhase2EvaluationBuffer::orderFiringRecords(
        const CudaPhase2ProjectionBuffer& projection) {
        assert(usedFiringRecords_ <= capacity_.firingRecords);
        assert(projection.usedCounts_[0] > 0);
        usedFiringOrder_ = usedFiringRecords_;
        deviceFiringOrderResult_ = deviceFiringOrderA_;
        if (usedFiringRecords_ == 0) return 0;

        ProjectionSemanticColumns columns{};
        columns.logicalBlocks = static_cast<
            const DeviceLogicalBlockProjection*>(projection.deviceColumns_[0]);
        columns.statements = static_cast<const IntEncodedExpr*>(
            projection.deviceColumns_[1]);
        columns.nameRecords = static_cast<const DeviceNameRecord*>(
            projection.deviceColumns_[2]);
        columns.nameBytes = static_cast<const char*>(projection.deviceColumns_[3]);
        columns.nameSlots = static_cast<const int32_t*>(
            projection.deviceColumns_[4]);
        columns.ruleStringRecords = static_cast<const DeviceRuleStringRecord*>(
            projection.deviceColumns_[5]);
        columns.ruleStringBytes = static_cast<const char*>(
            projection.deviceColumns_[6]);
        columns.byteMapViews = static_cast<const DeviceByteMapView*>(
            projection.deviceColumns_[7]);
        columns.byteMapEntries = static_cast<const DeviceByteMapEntry*>(
            projection.deviceColumns_[8]);
        columns.byteMapSlots = static_cast<const int32_t*>(
            projection.deviceColumns_[9]);
        columns.byteKeyBytes = static_cast<const char*>(
            projection.deviceColumns_[10]);
        columns.blobRecords = static_cast<const DeviceBlobRecord*>(
            projection.deviceColumns_[11]);
        columns.blobBytes = static_cast<const char*>(
            projection.deviceColumns_[12]);
        columns.reverseMapViews = static_cast<const DeviceReverseMapView*>(
            projection.deviceColumns_[13]);
        columns.reverseMapEntries = static_cast<const DeviceReverseMapEntry*>(
            projection.deviceColumns_[14]);
        columns.reverseMapSlots = static_cast<const int32_t*>(
            projection.deviceColumns_[15]);
        columns.reverseKeyBytes = static_cast<const char*>(
            projection.deviceColumns_[16]);
        columns.reverseOwners = static_cast<const int32_t*>(
            projection.deviceColumns_[17]);
        columns.podMapViews = static_cast<const DevicePodMapView*>(
            projection.deviceColumns_[18]);
        columns.podMapEntries = static_cast<const DevicePodMapEntry*>(
            projection.deviceColumns_[19]);
        columns.podMapSlots = static_cast<const int32_t*>(
            projection.deviceColumns_[20]);
        columns.podRunValues = static_cast<const int32_t*>(
            projection.deviceColumns_[21]);
        columns.mandatoryStatementKeys = static_cast<const int64_t*>(
            projection.deviceColumns_[22]);
        columns.metadataBytes = static_cast<const char*>(
            projection.deviceColumns_[23]);

        constexpr uint32_t threads = 256;
        const uint32_t blocks = (usedFiringRecords_ + threads - 1) / threads;
        initializeFiringOrderKernel<<<blocks, threads>>>(
            static_cast<uint32_t*>(deviceFiringOrderA_), usedFiringRecords_);
        GL_CUDA_ASSERT(cudaGetLastError());
        uint32_t* input = static_cast<uint32_t*>(deviceFiringOrderA_);
        uint32_t* output = static_cast<uint32_t*>(deviceFiringOrderB_);
        for (uint32_t width = 1; width < usedFiringRecords_;) {
            mergeFiringOrderKernel<<<blocks, threads>>>(
                columns,
                static_cast<const DevicePhase2FiringRecord*>(
                    deviceFiringRecords_),
                static_cast<const char*>(deviceGeneratedBytes_),
                static_cast<const int32_t*>(deviceLevelValues_),
                static_cast<const DeviceEvaluationDependency*>(
                    deviceOriginDependencies_),
                static_cast<const DeviceEvaluationByteSlice*>(deviceMarkerKeys_),
                static_cast<const int32_t*>(deviceMarkerRemainingArgs_),
                static_cast<const DeviceEvaluationByteSlice*>(deviceMarkerArgs_),
                input, output, usedFiringRecords_, width);
            GL_CUDA_ASSERT(cudaGetLastError());
            uint32_t* const swap = input;
            input = output;
            output = swap;
            if (width > usedFiringRecords_ / 2) break;
            width *= 2;
        }
        GL_CUDA_ASSERT(cudaDeviceSynchronize());
        deviceFiringOrderResult_ = input;
        return usedFiringOrder_;
    }

    /// @brief Select exact doom winners and retain only their canonical prefixes.
    ///
    /// @details
    /// Runs the projected `burstDeactivates` mirror on every head, CAS-min
    /// reduces packed growth-position and part-ordinal lines per block, resolves
    /// the first triggering ordered request at that line, then exclusive-scans
    /// selection flags over the already canonical permutation.
    /// The scatter is stable, so removing losing parts and post-trigger tails
    /// cannot disturb canonical content order.
    ///
    /// @param projection Uploaded roles, goals, known statements, and names.
    /// @return Number of canonical firing indices retained for host sealing.
    /// @invariant `orderFiringRecords` completed against this projection and
    ///            all fixed work arrays cover their used prefixes.
    uint32_t CudaPhase2EvaluationBuffer::selectDoomPrefixes(
        const CudaPhase2ProjectionBuffer& projection) {
        assert(projection.usedCounts_[0] > 0);
        assert(projection.usedCounts_[0] <= capacity_.logicalBlocks);
        assert(usedFiringOrder_ == usedFiringRecords_);
        usedDoomLines_ = projection.usedCounts_[0];

        ProjectionSemanticColumns columns{};
        columns.logicalBlocks = static_cast<
            const DeviceLogicalBlockProjection*>(projection.deviceColumns_[0]);
        columns.statements = static_cast<const IntEncodedExpr*>(
            projection.deviceColumns_[1]);
        columns.nameRecords = static_cast<const DeviceNameRecord*>(
            projection.deviceColumns_[2]);
        columns.nameBytes = static_cast<const char*>(projection.deviceColumns_[3]);
        columns.nameSlots = static_cast<const int32_t*>(
            projection.deviceColumns_[4]);
        columns.ruleStringRecords = static_cast<const DeviceRuleStringRecord*>(
            projection.deviceColumns_[5]);
        columns.ruleStringBytes = static_cast<const char*>(
            projection.deviceColumns_[6]);
        columns.byteMapViews = static_cast<const DeviceByteMapView*>(
            projection.deviceColumns_[7]);
        columns.byteMapEntries = static_cast<const DeviceByteMapEntry*>(
            projection.deviceColumns_[8]);
        columns.byteMapSlots = static_cast<const int32_t*>(
            projection.deviceColumns_[9]);
        columns.byteKeyBytes = static_cast<const char*>(
            projection.deviceColumns_[10]);
        columns.blobRecords = static_cast<const DeviceBlobRecord*>(
            projection.deviceColumns_[11]);
        columns.blobBytes = static_cast<const char*>(
            projection.deviceColumns_[12]);
        columns.reverseMapViews = static_cast<const DeviceReverseMapView*>(
            projection.deviceColumns_[13]);
        columns.reverseMapEntries = static_cast<const DeviceReverseMapEntry*>(
            projection.deviceColumns_[14]);
        columns.reverseMapSlots = static_cast<const int32_t*>(
            projection.deviceColumns_[15]);
        columns.reverseKeyBytes = static_cast<const char*>(
            projection.deviceColumns_[16]);
        columns.reverseOwners = static_cast<const int32_t*>(
            projection.deviceColumns_[17]);
        columns.podMapViews = static_cast<const DevicePodMapView*>(
            projection.deviceColumns_[18]);
        columns.podMapEntries = static_cast<const DevicePodMapEntry*>(
            projection.deviceColumns_[19]);
        columns.podMapSlots = static_cast<const int32_t*>(
            projection.deviceColumns_[20]);
        columns.podRunValues = static_cast<const int32_t*>(
            projection.deviceColumns_[21]);
        columns.mandatoryStatementKeys = static_cast<const int64_t*>(
            projection.deviceColumns_[22]);
        columns.metadataBytes = static_cast<const char*>(
            projection.deviceColumns_[23]);

        constexpr uint32_t threads = 256;
        const uint32_t logicalBlockGrid =
            (usedDoomLines_ + threads - 1) / threads;
        initializeDoomLinesKernel<<<logicalBlockGrid, threads>>>(
            static_cast<int64_t*>(deviceDoomLines_),
            static_cast<uint32_t*>(deviceDoomRequestIndices_),
            usedDoomLines_);
        GL_CUDA_ASSERT(cudaGetLastError());
        if (usedFiringRecords_ == 0) {
            GL_CUDA_ASSERT(cudaDeviceSynchronize());
            return 0;
        }

        const uint32_t firingGrid =
            (usedFiringRecords_ + threads - 1) / threads;
        detectDoomLinesKernel<<<firingGrid, threads>>>(
            columns,
            static_cast<const DevicePhase2FiringRecord*>(deviceFiringRecords_),
            static_cast<const char*>(deviceGeneratedBytes_),
            usedFiringRecords_, usedDoomLines_,
            static_cast<int64_t*>(deviceDoomLines_));
        GL_CUDA_ASSERT(cudaGetLastError());
        detectDoomRequestIndicesKernel<<<firingGrid, threads>>>(
            columns,
            static_cast<const DevicePhase2FiringRecord*>(deviceFiringRecords_),
            static_cast<const char*>(deviceGeneratedBytes_),
            usedFiringRecords_, usedDoomLines_,
            static_cast<const int64_t*>(deviceDoomLines_),
            static_cast<uint32_t*>(deviceDoomRequestIndices_));
        GL_CUDA_ASSERT(cudaGetLastError());
        markDoomPrefixKernel<<<firingGrid, threads>>>(
            static_cast<const DevicePhase2FiringRecord*>(deviceFiringRecords_),
            static_cast<const uint32_t*>(deviceFiringOrderResult_),
            static_cast<const int64_t*>(deviceDoomLines_),
            static_cast<const uint32_t*>(deviceDoomRequestIndices_),
            usedFiringRecords_,
            static_cast<uint32_t*>(deviceRequestCounts_));
        GL_CUDA_ASSERT(cudaGetLastError());
        std::size_t scanBytes = temporaryBytes_;
        GL_CUDA_ASSERT(cub::DeviceScan::ExclusiveSum(
            deviceTemporary_, scanBytes,
            static_cast<const uint32_t*>(deviceRequestCounts_),
            static_cast<uint32_t*>(deviceRequestOffsets_),
            static_cast<int>(usedFiringRecords_)));
        uint32_t* const compacted =
            deviceFiringOrderResult_ == deviceFiringOrderA_
            ? static_cast<uint32_t*>(deviceFiringOrderB_)
            : static_cast<uint32_t*>(deviceFiringOrderA_);
        scatterDoomPrefixKernel<<<firingGrid, threads>>>(
            static_cast<const uint32_t*>(deviceFiringOrderResult_),
            static_cast<const uint32_t*>(deviceRequestCounts_),
            static_cast<const uint32_t*>(deviceRequestOffsets_),
            usedFiringRecords_, compacted);
        GL_CUDA_ASSERT(cudaGetLastError());
        finishDoomPrefixCountKernel<<<1, 1>>>(
            static_cast<const uint32_t*>(deviceRequestCounts_),
            static_cast<const uint32_t*>(deviceRequestOffsets_),
            usedFiringRecords_,
            static_cast<uint32_t*>(deviceSelectedCount_));
        GL_CUDA_ASSERT(cudaGetLastError());
        GL_CUDA_ASSERT(cudaDeviceSynchronize());
        GL_CUDA_ASSERT(cudaMemcpy(
            &usedFiringOrder_, deviceSelectedCount_, sizeof(uint32_t),
            cudaMemcpyDeviceToHost));
        assert(usedFiringOrder_ <= usedFiringRecords_);
        deviceFiringOrderResult_ = compacted;
        return usedFiringOrder_;
    }

    /// @brief Download compact subset-surviving evaluation candidates.
    ///
    /// @param destination Caller-owned candidate output.
    /// @param capacity Destination element capacity.
    /// @return Number of candidates copied.
    /// @invariant `capacity` covers the last candidate count.
    uint32_t CudaPhase2EvaluationBuffer::downloadCandidates(
        DeviceEvaluationCandidate* destination,
        uint32_t capacity) const {
        assert(usedCandidates_ <= capacity);
        if (usedCandidates_ == 0) return 0;
        assert(destination != nullptr);
        GL_CUDA_ASSERT(cudaMemcpy(
            destination, deviceCandidates_,
            static_cast<std::size_t>(usedCandidates_)
                * sizeof(DeviceEvaluationCandidate),
            cudaMemcpyDeviceToHost));
        return usedCandidates_;
    }

    /// @brief Download compact encoded-hit candidates.
    ///
    /// @param destination Caller-owned encoded-hit output.
    /// @param capacity Destination element capacity.
    /// @return Number of hits copied.
    /// @invariant `capacity` covers the last encoded-hit count.
    uint32_t CudaPhase2EvaluationBuffer::downloadEncodedHits(
        DeviceEvaluationCandidate* destination,
        uint32_t capacity) const {
        assert(usedEncodedHits_ <= capacity);
        if (usedEncodedHits_ == 0) return 0;
        assert(destination != nullptr);
        GL_CUDA_ASSERT(cudaMemcpy(
            destination, deviceEncodedHits_,
            static_cast<std::size_t>(usedEncodedHits_)
                * sizeof(DeviceEvaluationCandidate),
            cudaMemcpyDeviceToHost));
        return usedEncodedHits_;
    }

    /// @brief Download LocalMemoryValue work selected by the last expansion.
    ///
    /// @param destination Caller-owned value-work output.
    /// @param capacity Destination element capacity.
    /// @return Number of value references copied.
    /// @invariant `capacity` covers the last local-value count.
    uint32_t CudaPhase2EvaluationBuffer::downloadValueWork(
        DeviceEvaluationValueWork* destination,
        uint32_t capacity) const {
        assert(usedValues_ <= capacity);
        if (usedValues_ == 0) return 0;
        assert(destination != nullptr);
        GL_CUDA_ASSERT(cudaMemcpy(
            destination, deviceValueWork_,
            static_cast<std::size_t>(usedValues_)
                * sizeof(DeviceEvaluationValueWork),
            cudaMemcpyDeviceToHost));
        return usedValues_;
    }

    /// @brief Download firing headers from the last expression materialization.
    ///
    /// @param destination Caller-owned firing-header output.
    /// @param capacity Destination element capacity.
    /// @return Number of firing headers copied.
    /// @invariant `capacity` covers the last firing-record count.
    uint32_t CudaPhase2EvaluationBuffer::downloadFiringRecords(
        DevicePhase2FiringRecord* destination,
        uint32_t capacity) const {
        assert(usedFiringRecords_ <= capacity);
        if (usedFiringRecords_ == 0) return 0;
        assert(destination != nullptr);
        GL_CUDA_ASSERT(cudaMemcpy(
            destination, deviceFiringRecords_,
            static_cast<std::size_t>(usedFiringRecords_)
                * sizeof(DevicePhase2FiringRecord),
            cudaMemcpyDeviceToHost));
        return usedFiringRecords_;
    }

    /// @brief Download generated bytes from the last expression materialization.
    ///
    /// @param destination Caller-owned byte output.
    /// @param capacity Destination byte capacity.
    /// @return Number of generated bytes copied.
    /// @invariant `capacity` covers the last generated-byte count.
    uint32_t CudaPhase2EvaluationBuffer::downloadGeneratedBytes(
        char* destination,
        uint32_t capacity) const {
        assert(usedGeneratedBytes_ <= capacity);
        if (usedGeneratedBytes_ == 0) return 0;
        assert(destination != nullptr);
        GL_CUDA_ASSERT(cudaMemcpy(
            destination, deviceGeneratedBytes_, usedGeneratedBytes_,
            cudaMemcpyDeviceToHost));
        return usedGeneratedBytes_;
    }

    /// @brief Download sorted-unique level values from the last materialization.
    ///
    /// @param destination Caller-owned integer output.
    /// @param capacity Destination element capacity.
    /// @return Number of level values copied.
    /// @invariant `capacity` covers the last level-value count.
    uint32_t CudaPhase2EvaluationBuffer::downloadLevelValues(
        int32_t* destination,
        uint32_t capacity) const {
        assert(usedLevelValues_ <= capacity);
        if (usedLevelValues_ == 0) return 0;
        assert(destination != nullptr);
        GL_CUDA_ASSERT(cudaMemcpy(
            destination, deviceLevelValues_,
            static_cast<std::size_t>(usedLevelValues_) * sizeof(int32_t),
            cudaMemcpyDeviceToHost));
        return usedLevelValues_;
    }

    /// @brief Download head provenance dependencies from the last materialization.
    ///
    /// @param destination Caller-owned dependency output.
    /// @param capacity Destination element capacity.
    /// @return Number of dependencies copied.
    /// @invariant `capacity` covers the last origin-dependency count.
    uint32_t CudaPhase2EvaluationBuffer::downloadOriginDependencies(
        DeviceEvaluationDependency* destination,
        uint32_t capacity) const {
        assert(usedOriginDependencies_ <= capacity);
        if (usedOriginDependencies_ == 0) return 0;
        assert(destination != nullptr);
        GL_CUDA_ASSERT(cudaMemcpy(
            destination, deviceOriginDependencies_,
            static_cast<std::size_t>(usedOriginDependencies_)
                * sizeof(DeviceEvaluationDependency),
            cudaMemcpyDeviceToHost));
        return usedOriginDependencies_;
    }

    /// @brief Download transformed marker key slices in LMV order.
    ///
    /// @param destination Caller-owned byte-slice output.
    /// @param capacity Destination element capacity.
    /// @return Number of marker key slices copied.
    /// @invariant `capacity` covers the last marker-key count.
    uint32_t CudaPhase2EvaluationBuffer::downloadMarkerKeys(
        DeviceEvaluationByteSlice* destination,
        uint32_t capacity) const {
        assert(usedMarkerKeys_ <= capacity);
        if (usedMarkerKeys_ == 0) return 0;
        assert(destination != nullptr);
        GL_CUDA_ASSERT(cudaMemcpy(
            destination, deviceMarkerKeys_,
            static_cast<std::size_t>(usedMarkerKeys_)
                * sizeof(DeviceEvaluationByteSlice),
            cudaMemcpyDeviceToHost));
        return usedMarkerKeys_;
    }

    /// @brief Download sorted-unique marker remaining-argument identifiers.
    ///
    /// @param destination Caller-owned rule-interner identifier output.
    /// @param capacity Destination element capacity.
    /// @return Number of remaining-argument identifiers copied.
    /// @invariant `capacity` covers the last marker-remaining count.
    uint32_t CudaPhase2EvaluationBuffer::downloadMarkerRemainingArgs(
        int32_t* destination,
        uint32_t capacity) const {
        assert(usedMarkerRemainingArgs_ <= capacity);
        if (usedMarkerRemainingArgs_ == 0) return 0;
        assert(destination != nullptr);
        GL_CUDA_ASSERT(cudaMemcpy(
            destination, deviceMarkerRemainingArgs_,
            static_cast<std::size_t>(usedMarkerRemainingArgs_)
                * sizeof(int32_t),
            cudaMemcpyDeviceToHost));
        return usedMarkerRemainingArgs_;
    }

    /// @brief Download sorted-unique non-marker head-argument slices.
    ///
    /// @param destination Caller-owned byte-slice output.
    /// @param capacity Destination element capacity.
    /// @return Number of marker argument slices copied.
    /// @invariant `capacity` covers the last marker-argument count.
    uint32_t CudaPhase2EvaluationBuffer::downloadMarkerArgs(
        DeviceEvaluationByteSlice* destination,
        uint32_t capacity) const {
        assert(usedMarkerArgs_ <= capacity);
        if (usedMarkerArgs_ == 0) return 0;
        assert(destination != nullptr);
        GL_CUDA_ASSERT(cudaMemcpy(
            destination, deviceMarkerArgs_,
            static_cast<std::size_t>(usedMarkerArgs_)
                * sizeof(DeviceEvaluationByteSlice),
            cudaMemcpyDeviceToHost));
        return usedMarkerArgs_;
    }

    /// @brief Download the canonical firing-record index permutation.
    ///
    /// @param destination Caller-owned record-index output.
    /// @param capacity Destination element capacity.
    /// @return Number of canonical indices copied.
    /// @invariant `capacity` covers the last canonical order count.
    uint32_t CudaPhase2EvaluationBuffer::downloadFiringOrder(
        uint32_t* destination,
        uint32_t capacity) const {
        assert(usedFiringOrder_ <= capacity);
        if (usedFiringOrder_ == 0) return 0;
        assert(destination != nullptr);
        assert(deviceFiringOrderResult_ != nullptr);
        GL_CUDA_ASSERT(cudaMemcpy(
            destination, deviceFiringOrderResult_,
            static_cast<std::size_t>(usedFiringOrder_) * sizeof(uint32_t),
            cudaMemcpyDeviceToHost));
        return usedFiringOrder_;
    }

    /// @brief Download final packed doom lines for the uploaded logical blocks.
    ///
    /// @param destination Caller-owned signed 64-bit output.
    /// @param capacity Destination element capacity.
    /// @return Number of logical-block doom lines copied.
    /// @invariant `capacity` covers the last doom-selection block count.
    uint32_t CudaPhase2EvaluationBuffer::downloadDoomLines(
        int64_t* destination,
        uint32_t capacity) const {
        assert(usedDoomLines_ <= capacity);
        if (usedDoomLines_ == 0) return 0;
        assert(destination != nullptr);
        GL_CUDA_ASSERT(cudaMemcpy(
            destination, deviceDoomLines_,
            static_cast<std::size_t>(usedDoomLines_) * sizeof(int64_t),
            cudaMemcpyDeviceToHost));
        return usedDoomLines_;
    }

    /// @brief Download first triggering request indices for projected blocks.
    ///
    /// @param destination Caller-owned unsigned request-index output.
    /// @param capacity Destination element capacity.
    /// @return Number of logical-block trigger indices copied.
    /// @invariant `capacity` covers the last doom-selection block count.
    uint32_t CudaPhase2EvaluationBuffer::downloadDoomRequestIndices(
        uint32_t* destination,
        uint32_t capacity) const {
        assert(usedDoomLines_ <= capacity);
        if (usedDoomLines_ == 0) return 0;
        assert(destination != nullptr);
        GL_CUDA_ASSERT(cudaMemcpy(
            destination, deviceDoomRequestIndices_,
            static_cast<std::size_t>(usedDoomLines_) * sizeof(uint32_t),
            cudaMemcpyDeviceToHost));
        return usedDoomLines_;
    }

    /// @brief Report all fixed evaluator bytes including shared CUB scratch.
    ///
    /// @return Exact startup bytes owned by this evaluator buffer.
    /// @invariant The value is constant for the object's lifetime.
    uint64_t CudaPhase2EvaluationBuffer::fixedAllocationBytes() const {
        return static_cast<uint64_t>(capacity_.requests)
                * (sizeof(DeviceEvaluationRequestState)
                    + sizeof(uint32_t) * 2)
            + static_cast<uint64_t>(capacity_.reverseOwners)
                * sizeof(DeviceEvaluationOwnerWork)
            + static_cast<uint64_t>(capacity_.candidateOwners)
                * sizeof(DeviceEvaluationCandidate)
            + static_cast<uint64_t>(capacity_.encodedHits)
                * (sizeof(DeviceEvaluationCandidate) + sizeof(uint32_t) * 2)
            + static_cast<uint64_t>(capacity_.localValues)
                * sizeof(DeviceEvaluationValueWork)
            + static_cast<uint64_t>(capacity_.firingRecords)
                * sizeof(DevicePhase2FiringRecord)
            + capacity_.generatedBytes
            + static_cast<uint64_t>(capacity_.levelValues) * sizeof(int32_t)
            + static_cast<uint64_t>(capacity_.originDependencies)
                * sizeof(DeviceEvaluationDependency)
            + static_cast<uint64_t>(capacity_.markerKeys)
                * sizeof(DeviceEvaluationByteSlice)
            + static_cast<uint64_t>(capacity_.markerRemainingArgs)
                * sizeof(int32_t)
            + static_cast<uint64_t>(capacity_.markerArgs)
                * sizeof(DeviceEvaluationByteSlice)
            + static_cast<uint64_t>(capacity_.firingRecords)
                * sizeof(uint32_t) * 2
            + static_cast<uint64_t>(capacity_.logicalBlocks) * sizeof(int64_t)
            + static_cast<uint64_t>(capacity_.logicalBlocks) * sizeof(uint32_t)
            + sizeof(uint32_t) + sizeof(DeviceEvaluationCounters)
            + static_cast<uint64_t>(temporaryBytes_);
    }

}  // namespace gl::gpu

#undef GL_CUDA_ASSERT
