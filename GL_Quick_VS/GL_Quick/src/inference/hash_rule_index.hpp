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
#include "../memory.hpp"
#include "../proof/worker_context.hpp"
#include "../infra/rt_tracker.hpp"
#include "staged_sorted_unique.hpp"
#include "rem_args_batch_blob.hpp"
namespace gl {
namespace tests { struct HashRuleIndexFixture; }
/// @brief Maintain ordered rule owners and their normalized-key indices.
/// @details Mutations operate on the supplied hash memory. Staging preserves
/// first-touch key allocation, sorted owner runs and the caller's scratch window.
class HashRuleIndex {
private:
friend struct tests::HashRuleIndexFixture;
/// @brief The plan of one section flush: which map id each staged
        ///        ordinal lands on, including the ids the new keys will take.
        ///
        /// @details
        /// `(*ordOfId)[id - 1]` is the staged ordinal of map id `id` (`0` = the
        /// section does not touch that key); ids `1..oldCount` are the stored
        /// keys, ids `oldCount + 1 .. oldCount + newN` are the keys the map
        /// lacks, assigned in staging first-touch order and minted in that
        /// order by `mintStagedKeys` AFTER the run rebuild. The table is a
        /// page-tier `PagedVector` the caller owns (a map may hold far more
        /// keys than one block of `int32` slots).
        struct StagedFlushPlan {
            PagedVector<int32_t>* ordOfId;   ///< id-1 -> staged ordinal, `0` = untouched.
            int32_t oldCount;                ///< Keys stored before the flush.
            int32_t newN;                    ///< Keys the flush adds.
        };
public:
/// @brief Minimum subkey length (premises) at which the growth probe
        ///        reads the u_ signatures; shorter subkeys are presence-only.
        ///
        /// @details
        /// A one- or two-premise subkey is the pathological owner-set: one
        /// template premise shared by every ground instance of a rule family
        /// carries one signature per instance, scanned linearly per probe. The
        /// grow search reaches the same whole keys either way (the whole-key
        /// record probe keeps every prune); what the threshold trades is grow
        /// nodes visited below length three against the cost and size of the
        /// short subkeys' signature lists — measured at ~15 % of the Gauss grow
        /// search against a record that would hold every ground instance
        /// ([D-302](../40_decisions.md#d-302)).
        /// Both sides honour it: `subkeyUSatisfied` reads signatures only from
        /// this length on, and the install sites store an empty record below it.
        static constexpr NameId kSubkeyUCheckMinElements = 3;
static constexpr int32_t kMaxRemArgsEdgeKeyBytes =
            static_cast<int32_t>(sizeof(int32_t))
                * (2 + static_cast<int32_t>(ExecutionParameters::MAX_KEY_SLOTS))
            + ExecutionParameters::kMaxNormKeyBytes;

public:

//#pragma optimize("", off)

        // =================================================================
        // Static pipeline helpers (zero string operations in hot path)
        // =================================================================




        /// @brief Probe the whole-key set — the record (emission) probe of the
        ///        request generator.
        ///
        /// @details
        /// `normKeyLookup` alone: the verdict is `lookup != 0`. The whole-key
        /// index is a bare key set (D-303) — there is no record to read. A
        /// present whole key always emits its request; the firing site
        /// (`checkLocalEncodedMemoryStatic`) is the exact gate — its `encodedMap`
        /// lookup misses for a key whose rules are gone, so a stale key (the set
        /// is never scope-wiped) is a sound over-approximation
        /// ([I-79](../30_invariants.md#i-79)). Read-only, mints nothing, touches
        /// no scratch, burst-safe on the shared LB ([I-83](../30_invariants.md#i-83)).
        ///
        /// @param set       The whole-key set (`normalizedEncodedKeys`).
        /// @param keyBuf    The normalized-key `NameId` payload.
        /// @param keyLen    The payload length (`<= MAX_KEY_SLOTS`).
        /// @param exprCount Premise count == the key's `numberExpressions`.
        /// @return `true` iff the key is present in @p set.
        /// @see `normKeyLookup` — the shared lookup core; `subkeyUSatisfied` — the
        ///      growth probe; `ensureWholeKeyPresent` — the install door.
/// @param map Whole-key owner index to probe.
static bool wholeKeyPresent(
            const TypedColdBlobMap<NormKey, RuleOwnerRec>& map,
            const NameId* keyBuf, NameId keyLen, NameId exprCount) {
            const int32_t id = normKeyLookup(map, keyBuf, keyLen, exprCount);
            assert((id == 0 || map.runLen(id) > 0)
                && "wholeKeyPresent: an owner-less whole key survived the "
                   "end-of-apply erasure");
            return id != 0;
        }

/// @brief Add one owner to a key's owner run — the shared install door
        ///        of the owner-run maps (`normalizedEncodedKeys`, `originals`,
        ///        `remainingArgsOwners`).
        ///
        /// @details
        /// An absent key is created with the one-owner run; a present key's
        /// run is read (eight-byte `RuleOwnerRec` blobs, ascending), the owner
        /// inserted at its sorted position and the run written back whole
        /// (`assignRunAtId`) — an owner already in the run is a defined no-op
        /// (a rule with symmetric premises reaches the same key through two
        /// permutations). The run therefore stays sorted-unique, its bytes a
        /// pure function of the owner set (I-98). Zero heap: the merged run
        /// rides @p gArena under a mark/pop frame. Single-threaded install
        /// site only (I-83).
        ///
        /// @tparam OwnerMap A `TypedColdBlobMap<K, RuleOwnerRec>` over a byte key.
        /// @param map      The owner-run map.
        /// @param keyBytes The key's encoded bytes (`Codec<K>::encode` form).
        /// @param owner    The installing rule's packed owner.
        /// @param gArena   Per-slot gen-scratch arena for the merged run.
        /// @return `true` when the owner was added, `false` when it was already
        ///         in the run.
        /// @invariant Every present key has a non-empty owner run until the
        ///            end-of-apply `eraseEmptyRuns`.
        /// @see `RuleOwner`, `removeOwnerFromRun`, `Codec<RuleOwnerRec>`.
template <class OwnerMap>
        static bool addOwnerToRun(OwnerMap& map, StrSpan keyBytes,
                                  RuleOwner owner, ScratchArena& gArena) {
            constexpr int32_t kRec = static_cast<int32_t>(sizeof(RuleOwner));
            int32_t id = 0;
            {
                RT_SCOPE_HERE("OWNER_LOOKUP");
                id = map.inner().lookup(keyBytes);
            }
            if (id == 0) {
                RT_SCOPE_HERE("OWNER_WRITE");
                char blob[sizeof(RuleOwner)];
                std::memcpy(blob, &owner, sizeof(RuleOwner));
                const int32_t oneLen = kRec;
                map.inner().assignRun(keyBytes, blob, &oneLen, 1);
                return true;
            }
            const int32_t M = map.runLen(id);
            const ArenaOffset mark = gArena.cursor();
            RuleOwner* run = reinterpret_cast<RuleOwner*>(gArena.resolve(
                gArena.alloc((M + 1) * kRec, static_cast<int32_t>(alignof(RuleOwner)))));
            int32_t insertAt = M;
            {
            RT_SCOPE_HERE("OWNER_MERGE");
            RT_NOTE_ITERATIONS_HERE(M);   // owners peeked + copied
            for (int32_t j = 0; j < M; ++j) {
                int32_t bl = 0;
                const char* bp = map.peekRecordBytes(id, j, bl, gArena);
                assert(bl == kRec && "addOwnerToRun: owner blob length drift");
                RuleOwner o;
                std::memcpy(&o, bp, sizeof(RuleOwner));
                if (o == owner) {
                    gArena.popTo(mark);
                    return false;
                }
                run[j] = o;
                if (insertAt == M && o > owner) insertAt = j;
            }
            for (int32_t j = M; j > insertAt; --j) run[j] = run[j - 1];
            run[insertAt] = owner;
            } // RT_SCOPE OWNER_MERGE
            int32_t* lens = reinterpret_cast<int32_t*>(gArena.resolve(
                gArena.alloc((M + 1) * static_cast<int32_t>(sizeof(int32_t)),
                             static_cast<int32_t>(alignof(int32_t)))));
            for (int32_t j = 0; j <= M; ++j) lens[j] = kRec;
            {
                RT_SCOPE_HERE("OWNER_WRITE");
                map.inner().assignRunAtId(id, reinterpret_cast<const char*>(run),
                                          lens, M + 1);
            }
            gArena.popTo(mark);
            return true;
        }

/// @brief Delete one owner from a key's owner run — the removal twin
        ///        of `addOwnerToRun`.
        ///
        /// @details
        /// The key must be present and the owner must be in its run (both
        /// asserted — a removal re-enumerates the install's own keys, so a
        /// miss means the enumeration diverged from the install, Rule 19).
        /// Every writer keeps an owner run sorted ascending and unique
        /// (`addOwnerToRun`, `flushOwnerSection`, this removal), so a binary
        /// search over the stored records finds the owner, and
        /// `HashMap::eraseBlobFromRun` splices out that one record: the
        /// surviving records keep their stored bytes or shared-pool
        /// identifiers and are never re-interned (D-452).
        /// The run may become EMPTY: the key itself is erased later by
        /// `eraseOwnerlessEntries` (one compacting pass per container per
        /// apply — a per-key erase would restructure the index each time).
        /// Zero heap: a record spanning pages is read through @p gArena under a
        /// mark/pop frame. Single-threaded seam only (I-83).
        ///
        /// @tparam OwnerMap A `TypedColdBlobMap<K, RuleOwnerRec>` over a byte key.
        /// @param map      The owner-run map.
        /// @param keyBytes The key's encoded bytes.
        /// @param owner    The leaving rule's packed owner.
        /// @param gArena   Per-slot gen-scratch arena for record reads.
        /// @return The number of owners left on the key (0 = the key is now
        ///         owner-less and awaits erasure).
        /// @see `addOwnerToRun`, `eraseOwnerlessEntries`.
template <class OwnerMap>
        static int32_t removeOwnerFromRun(OwnerMap& map, StrSpan keyBytes,
                                          RuleOwner owner, ScratchArena& gArena) {
            constexpr int32_t kRec = static_cast<int32_t>(sizeof(RuleOwner));
            const int32_t id = map.inner().lookup(keyBytes);
            assert(id != 0 && "removeOwnerFromRun: the key is absent");
            const int32_t M = map.runLen(id);
            const ArenaOffset mark = gArena.cursor();
            const auto ownerAt = [&](int32_t j) {
                int32_t bl = 0;
                const char* bp = map.peekRecordBytes(id, j, bl, gArena);
                assert(bl == kRec && "removeOwnerFromRun: owner blob length drift");
                RuleOwner o;
                std::memcpy(&o, bp, sizeof(RuleOwner));
                return o;
            };
            // First record not below the owner; the run is sorted-unique.
            int32_t lo = 0, hi = M;
            while (lo < hi) {
                const int32_t mid = lo + (hi - lo) / 2;
                if (ownerAt(mid) < owner) lo = mid + 1;
                else hi = mid;
            }
            assert(lo < M && ownerAt(lo) == owner
                && "removeOwnerFromRun: the owner is not on the key");
            gArena.popTo(mark);
            map.inner().eraseBlobFromRun(id, lo);
            return M - 1;
        }

/// @brief Delete one owner from a whole key — the removal twin of
        ///        `addWholeKeyOwner`.
        ///
        /// @param map               The whole-key owner map.
        /// @param numberExpressions The key's leading count field.
        /// @param data              The key's NameId payload.
        /// @param len               The payload length.
        /// @param owner             The leaving rule's packed owner.
        /// @param gArena            Per-slot gen-scratch arena.
        /// @return The number of owners left on the key.
        /// @see `removeOwnerFromRun`, `addWholeKeyOwner`.
/// @details Maintains the documented rule-index ordering and scratch lifetime.
static int32_t removeWholeKeyOwner(TypedColdBlobMap<NormKey, RuleOwnerRec>& map,
                                           int32_t numberExpressions,
                                           const NameId* data, int32_t len,
                                           RuleOwner owner, ScratchArena& gArena) {
            char keyBuf[ExecutionParameters::kMaxNormKeyBytes];
            const int32_t kn = encodeNormKeyInto(numberExpressions, data, len,
                keyBuf, ExecutionParameters::kMaxNormKeyBytes);
            return removeOwnerFromRun(map, StrSpan(keyBuf, kn), owner, gArena);
        }

/// @brief Delete every (owner, signature) pair of one owner from a
        ///        subkey record — the removal twin of
        ///        `mergeSubkeySignatures`.
        ///
        /// @details
        /// The record is read zero-copy (`OwnerSetBlob`) and rewritten from
        /// the surviving pairs: a signature survives iff some surviving pair
        /// references it (surviving signatures keep their relative order and
        /// are renumbered), the loose byte is set iff some surviving pair is
        /// loose, the pairs keep their (owner, index) order under the
        /// monotone renumbering. The owner must hold at least one pair
        /// (asserted — a removal re-enumerates the install's own keys). A
        /// record left with no pair is written as the nine-byte tombstone
        /// (loose 0, no signature, no pair) that `eraseOwnerlessEntries`
        /// erases at the end of the apply; nothing reads the map between
        /// the two on the single-threaded seam. Zero heap.
        ///
        /// @param map               The subkey owner-set map.
        /// @param numberExpressions The key's leading count field.
        /// @param data              The key's NameId payload.
        /// @param len               The payload length.
        /// @param owner             The leaving rule's packed owner.
        /// @return The number of pairs left on the record.
        /// @see `mergeSubkeySignatures`, `ensureEmptySubkeyRecord`, `recordSubkeyOwner`.
static int32_t removeSubkeyOwner(TypedColdBlobMap<NormKey, OwnerSet>& map,
                                         int32_t numberExpressions,
                                         const NameId* data, int32_t len,
                                         RuleOwner owner) {
            assert(numberExpressions >= kSubkeyUCheckMinElements
                && "removeSubkeyOwner: short subkeys have no owners to remove");
            char keyBuf[ExecutionParameters::kMaxNormKeyBytes];
            const int32_t kn = encodeNormKeyInto(numberExpressions, data, len,
                keyBuf, ExecutionParameters::kMaxNormKeyBytes);
            const StrSpan keyView(keyBuf, kn);
            const unsigned slot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            ScratchArena& gArena = genScratchArenas().forSlot(slot);
            const ArenaOffset mark = gArena.cursor();

            const int32_t id = map.inner().lookup(keyView);
            assert(id != 0 && "removeSubkeyOwner: the subkey is absent");
            int32_t blen = 0;
            const char* bp = map.peekRecordBytes(id, 0, blen, gArena);
            const OwnerSetBlob ob{ bp, blen };
            const int32_t sigCount = ob.uSigCount();
            const int32_t ownersOff = ob.ownersOffset();
            const int32_t pairCount = ob.ownerCount(ownersOff);

            // 1. Which signatures survive (referenced by a surviving pair),
            //    and their new indexes.
            int32_t* sigKept = reinterpret_cast<int32_t*>(gArena.resolve(
                gArena.alloc((sigCount > 0 ? sigCount : 1) * static_cast<int32_t>(sizeof(int32_t)),
                             static_cast<int32_t>(alignof(int32_t)))));
            for (int32_t s = 0; s < sigCount; ++s) sigKept[s] = 0;
            int32_t survivingPairs = 0;
            bool found = false;
            bool looseOut = false;
            for (int32_t i = 0; i < pairCount; ++i) {
                if (ob.ownerAt(ownersOff, i) == owner) { found = true; continue; }
                ++survivingPairs;
                const int32_t idx = ob.ownerSigIndexAt(ownersOff, i);
                if (idx < 0) looseOut = true;
                else sigKept[idx] = 1;
            }
            assert(found && "removeSubkeyOwner: the owner holds no pair on the subkey");
            int32_t* sigNewIndex = reinterpret_cast<int32_t*>(gArena.resolve(
                gArena.alloc((sigCount > 0 ? sigCount : 1) * static_cast<int32_t>(sizeof(int32_t)),
                             static_cast<int32_t>(alignof(int32_t)))));
            int32_t sigCountOut = 0;
            int32_t sigBytesOut = 0;
            {
                int32_t off = ob.firstSigOffset();
                for (int32_t s = 0; s < sigCount; ++s) {
                    sigNewIndex[s] = sigKept[s] ? sigCountOut++ : -1;
                    if (sigKept[s]) sigBytesOut += ob.sigBytes(off);
                    off += ob.sigBytes(off);
                }
            }

            // 2. Rebuild: loose byte, surviving signatures, surviving pairs.
            const int32_t outLen = 1 + 4 + sigBytesOut + 4
                + OwnerSetBlob::kOwnerRecBytes * survivingPairs;
            char* out = reinterpret_cast<char*>(gArena.resolve(gArena.alloc(outLen, 1)));
            int32_t at = 0;
            const auto wr32 = [&](int32_t v) {
                std::memcpy(out + at, &v, sizeof(int32_t)); at += 4;
            };
            out[at++] = static_cast<char>(looseOut ? 1 : 0);
            wr32(sigCountOut);
            {
                int32_t off = ob.firstSigOffset();
                for (int32_t s = 0; s < sigCount; ++s) {
                    const int32_t sb = ob.sigBytes(off);
                    if (sigKept[s]) {
                        std::memcpy(out + at, ob.p + off, static_cast<std::size_t>(sb));
                        at += sb;
                    }
                    off += sb;
                }
            }
            wr32(survivingPairs);
            for (int32_t i = 0; i < pairCount; ++i) {
                const RuleOwner o = ob.ownerAt(ownersOff, i);
                if (o == owner) continue;
                const int32_t idx = ob.ownerSigIndexAt(ownersOff, i);
                std::memcpy(out + at, &o, sizeof(RuleOwner)); at += 8;
                wr32(idx < 0 ? -1 : sigNewIndex[idx]);
            }
            assert(at == outLen && "removeSubkeyOwner: rebuilt blob length drift");
            map.inner().assignRunAtId(id, out, &outLen, 1);
            gArena.popTo(mark);
            return survivingPairs;
        }

/// @brief Delete one owner from a remaining-args edge; when the edge
        ///        loses its last owner, drop the NormKey from the forward run
        ///        and unlink the reverse-index edge — the removal twin of
        ///        `insertRemainingArgsNormKeyBatch` for one record.
        ///
        /// @details
        /// The edge key (`remArgsEdgeKeyInto`) names `(argSet, NormKey)`; its
        /// owner run loses @p owner (`removeOwnerFromRun`, asserting
        /// presence). At zero the forward map's run under the arg-set key is
        /// rewritten without that NormKey blob (asserted present) and
        /// `ReverseArgsIndex::removeEdge` unlinks `(NormKey → forward key
        /// id)`. A forward key left with an empty run and the owner-less
        /// edge key are erased by `eraseOwnerlessEntries`. Zero heap.
        ///
        /// @param map         The forward map (`remainingArgsNormalizedEncodedMap`).
        /// @param rev         Its derived reverse index.
        /// @param owners      The edge owner map (`remainingArgsOwners`).
        /// @param argKeyIds   The arg-set key run: `[argN, args...]` (ascending ids).
        /// @param argKeyN     Its length in `NameId`s.
        /// @param normKeyBlob The NormKey blob bytes (`Codec<NormKey>` form).
        /// @param normKeyLen  The blob length.
        /// @param owner       The leaving rule's packed owner.
        /// @param gArena      Per-slot gen-scratch arena.
        /// @see `insertRemainingArgsNormKeyBatch`, `remArgsEdgeKeyInto`,
        ///      `ReverseArgsIndex::removeEdge`.
static void removeRemainingArgsOwner(
                TypedColdBlobMap<Int16SetKey, NormKey>& map,
                ReverseArgsIndex& rev,
                TypedColdBlobMap<IdVecKey, RuleOwnerRec>& owners,
                const NameId* argKeyIds, int32_t argKeyN,
                const char* normKeyBlob, int32_t normKeyLen,
                RuleOwner owner, ScratchArena& gArena) {
            const ArenaOffset mark = gArena.cursor();
            char edgeBuf[kMaxRemArgsEdgeKeyBytes];
            const int32_t edgeLen = remArgsEdgeKeyInto(
                argKeyIds, argKeyN, normKeyBlob, normKeyLen,
                edgeBuf, kMaxRemArgsEdgeKeyBytes);
            const StrSpan edgeKey(edgeBuf, edgeLen);
            const int32_t left = removeOwnerFromRun(owners, edgeKey, owner, gArena);
            if (left > 0) {
                gArena.popTo(mark);
                return;
            }
            // Last owner gone: the NormKey leaves the forward run and the
            // reverse index.
            const StrSpan keyBytes(reinterpret_cast<const char*>(argKeyIds),
                argKeyN * static_cast<int32_t>(sizeof(NameId)));
            const int32_t fid = map.inner().lookup(keyBytes);
            assert(fid != 0 && "removeRemainingArgsOwner: the forward key is absent");
            const int32_t M = map.runLen(fid);
            const ArenaOffset concatMark = gArena.cursor();
            int32_t* lens = reinterpret_cast<int32_t*>(gArena.resolve(
                gArena.alloc((M > 0 ? M : 1) * static_cast<int32_t>(sizeof(int32_t)),
                             static_cast<int32_t>(alignof(int32_t)))));
            int32_t total = 0;
            for (int32_t j = 0; j < M; ++j) {
                int32_t bl = 0;
                (void)map.peekRecordBytes(fid, j, bl, gArena);
                total += bl;
            }
            char* concat = reinterpret_cast<char*>(gArena.resolve(
                gArena.alloc(total > 0 ? total : 1, 1)));
            int32_t at = 0, kept = 0;
            bool found = false;
            for (int32_t j = 0; j < M; ++j) {
                int32_t bl = 0;
                const char* bp = map.peekRecordBytes(fid, j, bl, gArena);
                if (!found && bl == normKeyLen
                    && std::memcmp(bp, normKeyBlob, static_cast<std::size_t>(bl)) == 0) {
                    found = true;
                    continue;
                }
                std::memcpy(concat + at, bp, static_cast<std::size_t>(bl));
                lens[kept++] = bl;
                at += bl;
            }
            assert(found && "removeRemainingArgsOwner: the NormKey is not in the forward run");
            map.inner().assignRunAtId(fid, concat, lens, kept);
            rev.removeEdge(StrSpan(normKeyBlob, normKeyLen), fid);
            gArena.popTo(concatMark);
            gArena.popTo(mark);
        }

/// @brief Remove-policy twin of one permutation's whole-key + edge
        ///        install: delete the owner from the whole key and from the
        ///        `(argSet, NormKey)` edge, each on the removal's first visit.
        ///
        /// @details
        /// Shared by the head install loop (`addToHashMemory`) and the marker
        /// variant loop (`installAdmissionMarkerVariants`) under
        /// `RuleIndexOp::Remove`. The NormKey is the `ignoreU=false` form the
        /// install minted (`bufNotIgnored`); the edge key is rebuilt by
        /// `remArgsEdgeKeyInto`. A repeat visit (a symmetric rule's second
        /// permutation) is skipped through `RuleIndexOp::firstVisit`.
        ///
        /// @param hm               The hash-memory instance.
        /// @param op               The removal policy (carries the visited set).
        /// @param numberExpressions The key's premise count.
        /// @param nkData           The NormKey payload (`ignoreU=false`).
        /// @param nkLen            Its length.
        /// @param argSetSorted     The remaining-arg ids, ascending.
        /// @param argN             Their number.
        /// @param owner            The leaving rule's packed owner.
        /// @param gArena           Per-slot gen-scratch arena.
        /// @see `removeWholeKeyOwner`, `removeRemainingArgsOwner`, `RuleIndexOp`.
static void removeRuleKeyEntries(HashMemory& hm, const RuleIndexOp& op,
                                         int32_t numberExpressions,
                                         const NameId* nkData, int32_t nkLen,
                                         const NameId* argSetSorted, int32_t argN,
                                         RuleOwner owner, ScratchArena& gArena) {
            char keyBuf[ExecutionParameters::kMaxNormKeyBytes];
            const int32_t kn = encodeNormKeyInto(numberExpressions, nkData, nkLen,
                keyBuf, ExecutionParameters::kMaxNormKeyBytes);
            if (op.firstVisit(RuleIndexOp::WholeKey, StrSpan(keyBuf, kn), owner, gArena)) {
                removeOwnerFromRun(hm.normalizedEncodedKeys,
                    StrSpan(keyBuf, kn), owner, gArena);
            }
            // The edge: the arg-set run [argN, args...] ++ the NormKey blob
            // (which is exactly the whole-key bytes).
            NameId argKey[1 + ExecutionParameters::MAX_KEY_SLOTS];
            argKey[0] = static_cast<NameId>(argN);
            for (int32_t i = 0; i < argN; ++i) argKey[1 + i] = argSetSorted[i];
            char edgeBuf[kMaxRemArgsEdgeKeyBytes];
            const int32_t edgeLen = remArgsEdgeKeyInto(argKey, 1 + argN, keyBuf, kn,
                                                       edgeBuf, kMaxRemArgsEdgeKeyBytes);
            const StrSpan edgeKey(edgeBuf, edgeLen);
            const bool first = op.firstVisit(
                RuleIndexOp::Edge, edgeKey, owner, gArena);
            if (first) {
                removeRemainingArgsOwner(hm.remainingArgsNormalizedEncodedMap,
                    hm.remainingArgsReverseIndex, hm.remainingArgsOwners,
                    argKey, 1 + argN, keyBuf, kn, owner, gArena);
            }
        }

/// @brief Remove-policy twin of one subkey install: delete the owner's
        ///        pairs from the subkey record on the removal's first visit.
        ///
        /// @param hm               The hash-memory instance.
        /// @param op               The removal policy (carries the visited set).
        /// @param numberExpressions The subkey's premise count.
        /// @param nkData           The subkey payload (`ignoreU=false`).
        /// @param nkLen            Its length.
        /// @param owner            The leaving rule's packed owner.
        /// @param gArena           Per-slot gen-scratch arena (the visit key).
        /// @see `removeSubkeyOwner`, `RuleIndexOp`.
/// @details Maintains the documented rule-index ordering and scratch lifetime.
static void removeRuleSubkeyEntry(HashMemory& hm, const RuleIndexOp& op,
                                          int32_t numberExpressions,
                                          const NameId* nkData, int32_t nkLen,
                                          RuleOwner owner, ScratchArena& gArena) {
            assert(numberExpressions > 0);
            if (numberExpressions < kSubkeyUCheckMinElements) return;
            char keyBuf[ExecutionParameters::kMaxNormKeyBytes];
            const int32_t kn = encodeNormKeyInto(numberExpressions, nkData, nkLen,
                keyBuf, ExecutionParameters::kMaxNormKeyBytes);
            if (op.firstVisit(RuleIndexOp::Subkey, StrSpan(keyBuf, kn), owner, gArena)) {
                removeSubkeyOwner(hm.normalizedEncodedSubkeys, numberExpressions,
                                  nkData, nkLen, owner);
            }
        }

/// @brief Erase every owner-less entry of one `HashMemory`'s owner
        ///        maps — the end-of-apply compaction that follows the
        ///        removals.
        ///
        /// @details
        /// One compacting pass per container (`eraseBlobIf`, which rebuilds
        /// the key index once) instead of a per-key erase: the whole-key map,
        /// the originals map and the edge map drop the keys whose owner run is
        /// empty; the subkey map drops long records with no pair while
        /// retaining every empty short key; the
        /// remaining-args forward map drops the arg-set keys whose NormKey run
        /// emptied. The victim set of each pass is collected BEFORE the pass
        /// (a scratch key set), so the predicate never reads the container it
        /// is compacting. Runs on the single-threaded seam after every removal
        /// of an apply; a map with no removal since the last pass is skipped
        /// by the `ownerlessPending` flag the removal doors raise. Zero heap.
        ///
        /// @param hm     The hash-memory instance.
        /// @param gArena Per-slot gen-scratch arena for the victim sets.
        /// @see `removeOwnerFromRun`, `removeSubkeyOwner`, `removeRemainingArgsOwner`.
static void eraseOwnerlessEntries(HashMemory& hm, ScratchArena& gArena) {
            assert((ruleStagings().boundTo(currentGenSlot(), &hm) == nullptr
                    || ruleStagings().boundTo(currentGenSlot(), &hm)->empty())
                && "eraseOwnerlessEntries: staged rule-index writes reached the sweep");
            if (!hm.ownerlessPending) return;
            hm.ownerlessPending = false;
            const ArenaOffset mark = gArena.cursor();
            DirtyState victimDirty = DirtyState::Clean;
            const auto sweepEmptyRuns = [&](auto& map) {
                ColdHashSet<BytesKeyStore> victims(&gArena, &victimDirty);
                victims.resetToFresh();
                const int32_t n = map.count();
                for (int32_t id = 1; id <= n; ++id)
                    if (map.runLen(id) == 0) {
                        victims.mint(map.inner().keyAt(id));
                    }
                if (victims.count() > 0) {
                    map.inner().eraseBlobIf([&](auto k) {
                        return victims.lookup(StrSpan(k)) != 0;
                    });
                }
            };
            sweepEmptyRuns(hm.normalizedEncodedKeys);
            // The chain map's compaction renumbers its key ids, which the
            // derived head-operator index stores as its list entries: a sweep
            // that erased a chain marks the index stale, and its next consumer
            // rebuilds it lazily (I-242).
            const int32_t originalsBefore = hm.originals.count();
            // The goal-rescan floor follows the renumbering: the compaction
            // keeps the survivors' order, so the chains every goal has seen
            // are exactly the survivors among the old ids 1..mark.
            assert(hm.goalScanChainMark >= 0 && hm.goalScanChainMark <= originalsBefore
                && "eraseOwnerlessEntries: goal-rescan floor above the chain count");
            int32_t survivorsBelowMark = hm.goalScanChainMark;
            for (int32_t id = 1; id <= hm.goalScanChainMark; ++id)
                if (hm.originals.runLen(id) == 0) --survivorsBelowMark;
            // The contradiction-demand floor follows the same renumbering
            // (D-560).
            assert(hm.contradictionChainMark >= 0 && hm.contradictionChainMark <= originalsBefore
                && "eraseOwnerlessEntries: contradiction-demand floor above the chain count");
            int32_t survivorsBelowContradictionMark = hm.contradictionChainMark;
            for (int32_t id = 1; id <= hm.contradictionChainMark; ++id)
                if (hm.originals.runLen(id) == 0) --survivorsBelowContradictionMark;
            sweepEmptyRuns(hm.originals);
            hm.goalScanChainMark = survivorsBelowMark;
            hm.contradictionChainMark = survivorsBelowContradictionMark;
            if (hm.originals.count() != originalsBefore) {
                hm.originalsHeadIndex.markUnbuilt();
            }
            sweepEmptyRuns(hm.remainingArgsOwners);
            sweepEmptyRuns(hm.copyOwners);
            // A pattern key whose last owner left, and a reopened scope's
            // emptied closed record; shape keys keep their masks.
            sweepEmptyRuns(hm.premisePatterns);
            // The forward map's compaction renumbers its key ids, which the
            // derived reverse index stores as its edge targets (I-154): a
            // sweep that erased a key is followed by the wholesale rebuild
            // (the same rebuild a canonical reload runs).
            const int32_t forwardBefore = hm.remainingArgsNormalizedEncodedMap.count();
            sweepEmptyRuns(hm.remainingArgsNormalizedEncodedMap);
            if (hm.remainingArgsNormalizedEncodedMap.count() != forwardBefore) {
                hm.remainingArgsReverseIndex.rebuildReverseIndex(
                    hm.remainingArgsNormalizedEncodedMap, gArena);
            }
            {
                ColdHashSet<BytesKeyStore> victims(&gArena, &victimDirty);
                victims.resetToFresh();
                auto& map = hm.normalizedEncodedSubkeys;
                const int32_t n = map.count();
                for (int32_t id = 1; id <= n; ++id) {
                    const StrSpan key = map.inner().keyAt(id);
                    assert(key.len >= static_cast<int32_t>(2 * sizeof(NameId)));
                    NameId premiseCount = 0;
                    std::memcpy(&premiseCount, key.ptr, sizeof(premiseCount));
                    assert(premiseCount > 0);
                    if (premiseCount < kSubkeyUCheckMinElements) continue;
                    int32_t blen = 0;
                    const char* bp = map.peekRecordBytes(id, 0, blen, gArena);
                    const OwnerSetBlob ob{ bp, blen };
                    if (ob.ownerCount(ob.ownersOffset()) == 0)
                        victims.mint(map.inner().keyAt(id));
                }
                if (victims.count() > 0) {
                    map.inner().eraseBlobIf([&](auto k) {
                        return victims.lookup(StrSpan(k)) != 0;
                    });
                }
            }
            gArena.popTo(mark);
        }

// the permutation loop's NormKey blob ref (defined below)

        /// @brief Stage one owner under a raw key — the staging twin of
        ///        `addOwnerToRun`.
        ///
        /// @details
        /// The flush merges the key's staged owners into its run sorted-unique,
        /// which is the state the sequential `addOwnerToRun` calls leave (a
        /// repeated owner is a no-op there and a duplicate here).
        ///
        /// @param section  The owner map's staging section.
        /// @param keyBytes The key's encoded bytes.
        /// @param owner    The installing rule's packed owner.
        /// @return Nothing.
        /// @see `addOwnerToRun`, `flushOwnerSection`.
static void stageOwnerToRun(RuleIndexStagingSection& section,
                                    StrSpan keyBytes, RuleOwner owner) {
            section.stage(keyBytes, owner, nullptr, 0, 0);
        }

/// @brief Stage one rule premise into the premise-pattern map — the
        ///        shape's `u_` slot mask and the pattern's scope-major owner.
        ///
        /// @details
        /// The install half of the request statement filter's premise-pattern
        /// condition (`premisePatternAdmits`). The premise's shape key gains
        /// its mask (`premiseUnchangeableMask`), its pattern key under that
        /// mask gains the owner `(scope, copy id)`. The flush merges both
        /// runs sorted-unique (`flushOwnerSection`), so a premise repeated in
        /// one copy, a copy two rules share, and a reinstall add nothing.
        /// Single-threaded install seam (I-83).
        ///
        /// @param section    The premise-pattern staging section.
        /// @param premise    The encoded premise (its `u_` flags and literals).
        /// @param scope      The NameMap id of the install scope.
        /// @param copyImplId The `ruleInterner` id of the multiplication copy.
        /// @return Nothing.
        /// @see `removePremisePattern`, `premisePatternAdmitsIn`.
static void stagePremisePattern(RuleIndexStagingSection& section,
                                        const IntEncodedExpr& premise,
                                        NameId scope, int32_t copyImplId) {
            char key[kMaxPremisePatternKeyBytes];
            const int64_t mask = premiseUnchangeableMask(premise);
            const int32_t shapeLen =
                premiseShapeKeyInto(premise, key, kMaxPremisePatternKeyBytes);
            section.stage(StrSpan(key, shapeLen), mask, nullptr, 0, 0);
            const int32_t patternLen =
                premisePatternKeyInto(premise, mask, key, kMaxPremisePatternKeyBytes);
            section.stage(StrSpan(key, patternLen),
                          packPremisePatternOwner(scope, copyImplId), nullptr, 0, 0);
        }

/// @brief Delete one rule premise's owner from the premise-pattern map
        ///        — the removal twin of `stagePremisePattern`.
        ///
        /// @details
        /// The pattern key loses the owner on its first visit of this removal
        /// (a premise repeated in the copy reaches the same key twice); the
        /// key, emptied, leaves at the end-of-apply `eraseOwnerlessEntries`.
        /// The shape's mask stays (a stale mask costs one lookup). A removal
        /// re-enumerates the install's own premises, so the owner is on the
        /// key — asserted by `removeOwnerFromRun`; a subtree wipe never deletes
        /// an owner (it records the closed scope instead).
        ///
        /// @param hm         The hash memory.
        /// @param op         The removal policy (visited set).
        /// @param premise    The encoded premise.
        /// @param scope      The NameMap id of the install scope.
        /// @param copyImplId The `ruleInterner` id of the multiplication copy.
        /// @param gArena     Per-slot gen-scratch arena.
        /// @return Nothing.
        /// @see `stagePremisePattern`, `removeOwnerFromRun`.
static void removePremisePattern(HashMemory& hm, const RuleIndexOp& op,
                                         const IntEncodedExpr& premise,
                                         NameId scope, int32_t copyImplId,
                                         ScratchArena& gArena) {
            char key[kMaxPremisePatternKeyBytes];
            const int64_t mask = premiseUnchangeableMask(premise);
            const int32_t patternLen =
                premisePatternKeyInto(premise, mask, key, kMaxPremisePatternKeyBytes);
            const StrSpan keyView(key, patternLen);
            const RuleOwner owner = packPremisePatternOwner(scope, copyImplId);
            if (op.firstVisit(RuleIndexOp::PremisePattern, keyView, owner, gArena)) {
                removeOwnerFromRun(hm.premisePatterns, keyView, owner, gArena);
            }
        }

/// @brief Record that a subtree wipe closed a scope in one hash
        ///        memory's premise-pattern map.
        ///
        /// @details
        /// The wipe removed the scope's rules from `encodedMap`; their premise
        /// owners stay (a later removal still finds them), and the filter
        /// skips owners at a recorded closed scope. The scope's key gets a
        /// one-record run (minted on first close, refilled after a reopen);
        /// a scope already recorded closed is left as it is. Single-threaded
        /// wipe seam.
        ///
        /// @param hm     The hash memory whose `encodedMap` the wipe filtered.
        /// @param scope  The closed scope's NameMap id.
        /// @param gArena Per-slot gen-scratch arena.
        /// @return Nothing.
        /// @see `reopenPremisePatternScope`, `premisePatternScopeClosed`.
static void recordClosedPremiseScope(HashMemory& hm, NameId scope,
                                             ScratchArena& gArena) {
            assert(scope > 0 && scope != NameMap::MAIN_ID
                && "recordClosedPremiseScope: main or an unminted scope closed");
            char key[kMaxPremisePatternKeyBytes];
            const int32_t len =
                premiseClosedScopeKeyInto(scope, key, kMaxPremisePatternKeyBytes);
            const StrSpan keyView(key, len);
            const int32_t id = hm.premisePatterns.inner().lookup(keyView);
            if (id != 0 && hm.premisePatterns.runLen(id) > 0) return;
            addOwnerToRun(hm.premisePatterns, keyView,
                          static_cast<RuleOwner>(scope), gArena);
        }

/// @brief Reopen a scope an install writes rules into — the scope's
        ///        closed record in the premise-pattern map, if any, is emptied.
        ///
        /// @details
        /// An install at a scope a subtree wipe once closed makes the scope's
        /// premise owners live again for the filter (the new rule's owner, and
        /// any stale owner from the scope's earlier rules, which only widens
        /// the filter). No closed record is a defined no-op.
        ///
        /// @param hm    The hash memory the install targets.
        /// @param scope The install scope's NameMap id.
        /// @return Nothing.
        /// @see `recordClosedPremiseScope`.
static void reopenPremisePatternScope(HashMemory& hm, NameId scope) {
            char key[kMaxPremisePatternKeyBytes];
            const int32_t len =
                premiseClosedScopeKeyInto(scope, key, kMaxPremisePatternKeyBytes);
            const int32_t id = hm.premisePatterns.inner().lookup(StrSpan(key, len));
            if (id == 0 || hm.premisePatterns.runLen(id) == 0) return;
            assert(hm.premisePatterns.runLen(id) == 1
                && "reopenPremisePatternScope: a closed record with several values");
            hm.premisePatterns.inner().eraseBlobFromRun(id, 0);
        }

/// @brief Stage a whole key's owner — the staging twin of
        ///        `addWholeKeyOwner`.
        ///
        /// @param section           The whole-key staging section.
        /// @param numberExpressions The key's leading count field.
        /// @param data              The key's NameId payload.
        /// @param len               The payload length.
        /// @param owner             The installing rule's packed owner.
        /// @return Nothing.
        /// @see `addWholeKeyOwner`, `stageOwnerToRun`.
/// @details Maintains the documented rule-index ordering and scratch lifetime.
static void stageWholeKeyOwner(RuleIndexStagingSection& section,
                                       int32_t numberExpressions,
                                       const NameId* data, int32_t len,
                                       RuleOwner owner) {
            char keyBuf[ExecutionParameters::kMaxNormKeyBytes];
            const int32_t kn = encodeNormKeyInto(numberExpressions, data, len,
                keyBuf, ExecutionParameters::kMaxNormKeyBytes);
            section.stage(StrSpan(keyBuf, kn), owner, nullptr, 0, 0);
        }

/// @brief Stage a short subkey's presence with no owner or signature.
        ///
        /// @details
        /// Only the first install stages an empty item for the key. The flush
        /// writes the fixed empty record (I-228).
        ///
        /// @param section The subkey staging section.
        /// @param numberExpressions The premise count, one or two.
        /// @param data The normalized key's NameId payload.
        /// @param len The payload length.
        /// @return Nothing.
        /// @invariant No rule identity is retained.
        /// @see ensureEmptySubkeyRecord, flushSubkeySection.
static void stageEmptySubkeyRecord(RuleIndexStagingSection& section,
                                           int32_t numberExpressions,
                                           const NameId* data, int32_t len) {
            assert(numberExpressions > 0
                && numberExpressions < kSubkeyUCheckMinElements);
            char keyBuf[ExecutionParameters::kMaxNormKeyBytes];
            const int32_t kn = encodeNormKeyInto(numberExpressions, data, len,
                keyBuf, ExecutionParameters::kMaxNormKeyBytes);
            const StrSpan keyView(keyBuf, kn);
            if (section.lookup(keyView) == 0) {
                section.stage(keyView, 0, nullptr, 0, 0);
            }
        }

/// @brief Stage an owner's u_ signature for a subkey — the staging twin
        ///        of `mergeSubkeySignatures`.
        ///
        /// @details
        /// The signature is built by `buildUSignatureRunInto` exactly as the
        /// door builds it; a prefix without u_ arguments stages a loose
        /// contribution (the door's `!hasUArg` path: loose byte set, pair index
        /// -1, no signature), otherwise the pairs are staged as the record's
        /// own `(int32 slot, int32 id)` layout (a signature with zero pairs is
        /// a zero-length payload with the loose flag clear).
        ///
        /// @param section           The subkey staging section.
        /// @param numberExpressions The key's leading count field.
        /// @param data              The key's NameId payload.
        /// @param len               The payload length.
        /// @param encList           The subkey's pre-encoded premises.
        /// @param count             Their number (`>= kSubkeyUCheckMinElements`).
        /// @param owner             The installing rule's packed owner.
        /// @return Nothing.
        /// @see `mergeSubkeySignatures`, `buildUSignatureRunInto`, `flushSubkeySection`.
static void stageSubkeySignatures(RuleIndexStagingSection& section,
                                          int32_t numberExpressions,
                                          const NameId* data, int32_t len,
                                          const IntEncodedExpr* encList, NameId count,
                                          RuleOwner owner) {
            using Pair = std::pair<int32_t, NameId>;
            assert(count >= kSubkeyUCheckMinElements
                && "stageSubkeySignatures: a shorter subkey stages presence only");
            const unsigned slot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            ScratchArena& gArena = genScratchArenas().forSlot(slot);
            const ArenaOffset mark = gArena.cursor();

            char keyBuf[ExecutionParameters::kMaxNormKeyBytes];
            const int32_t kn = encodeNormKeyInto(numberExpressions, data, len,
                keyBuf, ExecutionParameters::kMaxNormKeyBytes);
            const StrSpan keyView(keyBuf, kn);

            const int32_t sigCap = static_cast<int32_t>(count)
                * static_cast<int32_t>(ExecutionParameters::MAX_ARITY);
            Pair* newSig = reinterpret_cast<Pair*>(gArena.resolve(gArena.alloc(
                sigCap * static_cast<int32_t>(sizeof(Pair)),
                static_cast<int32_t>(alignof(Pair)))));
            bool hasUArg = false;
            int32_t newSigN = 0;
            {
                RT_SCOPE_HERE("SKM_SIGNATURE");
                newSigN = buildUSignatureRunInto(encList, count, newSig, sigCap, hasUArg);
            }
            if (!hasUArg) {
                section.stage(keyView, owner, nullptr, 0, /*loose=*/1);
            } else {
                int32_t* sig = reinterpret_cast<int32_t*>(gArena.resolve(gArena.alloc(
                    (newSigN > 0 ? newSigN : 1) * 2 * static_cast<int32_t>(sizeof(int32_t)),
                    static_cast<int32_t>(alignof(int32_t)))));
                for (int32_t k = 0; k < newSigN; ++k) {
                    sig[2 * k] = newSig[k].first;
                    sig[2 * k + 1] = static_cast<int32_t>(newSig[k].second);
                }
                section.stage(keyView, owner, reinterpret_cast<const char*>(sig),
                              newSigN * 8, /*loose=*/0);
            }
            gArena.popTo(mark);
        }

/// @brief Stage a copy's remaining-args batch — the staging twin of
        ///        `insertRemainingArgsNormKeyBatch`.
        ///
        /// @details
        /// The batch is sorted and deduplicated as the door does; every
        /// distinct `NormKey` blob is staged under the arg-set key in the
        /// forward section, and the rule's owner is staged under the
        /// `(argSet, NormKey)` edge key in the owner section. The flush merges
        /// the forward run sorted-unique and appends one reverse-index edge
        /// per `NormKey` the run did not hold before.
        ///
        /// @param remArgs       The forward map's staging section.
        /// @param remArgsOwners The edge owner map's staging section.
        /// @param argSet        The arg-id set, ascending.
        /// @param argN          Its length.
        /// @param batch         The permutation loop's accumulated NormKey blobs.
        /// @param owner         The installing rule's packed owner.
        /// @param gArena        Per-slot gen-scratch arena for the sort index
        ///                      and the edge keys.
        /// @return Nothing.
        /// @see `insertRemainingArgsNormKeyBatch`, `flushRemArgsSection`.
static void stageRemainingArgsNormKeyBatch(
                RuleIndexStagingSection& remArgs,
                RuleIndexStagingSection& remArgsOwners,
                const NameId* argSet, int32_t argN,
                const PagedVector<RemArgsBatchBlob>& batch,
                RuleOwner owner,
                ScratchArena& gArena) {
            const int32_t batchN = batch.size();
            if (batchN == 0) return;   // no keys accumulated (all permutations
                                       // skipped) — a defined no-op, not a fallback
            const ArenaOffset mark = gArena.cursor();
            assert(argN >= 0 && argN <= ExecutionParameters::MAX_KEY_SLOTS
                && "stageRemainingArgsNormKeyBatch: argSet exceeds MAX_KEY_SLOTS");
            NameId keyBuf[1 + ExecutionParameters::MAX_KEY_SLOTS];
            int32_t kn = 0;
            keyBuf[kn++] = static_cast<NameId>(argN);
            for (int32_t i = 0; i < argN; ++i) keyBuf[kn++] = argSet[i];
            const StrSpan keyBytes(reinterpret_cast<const char*>(keyBuf),
                kn * static_cast<int32_t>(sizeof(NameId)));

            int32_t* bidx = reinterpret_cast<int32_t*>(gArena.resolve(
                gArena.alloc(batchN * static_cast<int32_t>(sizeof(int32_t)),
                             static_cast<int32_t>(alignof(int32_t)))));
            ArenaOffset* batchOffs = reinterpret_cast<ArenaOffset*>(
                gArena.resolve(gArena.alloc(
                    batchN * static_cast<int32_t>(sizeof(ArenaOffset)),
                    static_cast<int32_t>(alignof(ArenaOffset)))));
            int32_t* batchLens = reinterpret_cast<int32_t*>(gArena.resolve(
                gArena.alloc(batchN * static_cast<int32_t>(sizeof(int32_t)),
                             static_cast<int32_t>(alignof(int32_t)))));
            for (int32_t i = 0; i < batchN; ++i) {
                bidx[i] = i;
                batchOffs[i] = batch[i].off;
                batchLens[i] = batch[i].len;
            }
            const auto resolveBatch = [&](int32_t j) -> const char* {
                return reinterpret_cast<const char*>(gArena.resolve(batchOffs[j]));
            };
            int32_t dN = 0;
            {
            RT_SCOPE_HERE("RAB_SORT_DEDUP");
            std::sort(bidx, bidx + batchN, [&](int32_t a, int32_t b) {
                return remArgsNormKeyBlobCmp(resolveBatch(a), batchLens[a],
                                             resolveBatch(b), batchLens[b]) < 0;
            });
            for (int32_t k = 0; k < batchN; ++k) {
                const int32_t j = bidx[k];
                if (dN > 0) {
                    const int32_t prev = bidx[dN - 1];
                    if (remArgsNormKeyBlobCmp(resolveBatch(prev), batchLens[prev],
                                              resolveBatch(j), batchLens[j]) == 0) {
                        continue;   // dup within the batch — one record
                    }
                }
                bidx[dN++] = j;
            }
            } // RT_SCOPE RAB_SORT_DEDUP
            {
            RT_SCOPE_HERE("RAB_STAGE");
            for (int32_t k = 0; k < dN; ++k) {
                const int32_t bj = bidx[k];
                remArgs.stage(keyBytes, 0, resolveBatch(bj), batchLens[bj], 0);
                char edgeBuf[kMaxRemArgsEdgeKeyBytes];
                const int32_t edgeLen = remArgsEdgeKeyInto(
                    keyBuf, kn, resolveBatch(bj), batchLens[bj],
                    edgeBuf, kMaxRemArgsEdgeKeyBytes);
                remArgsOwners.stage(StrSpan(edgeBuf, edgeLen), owner, nullptr, 0, 0);
            }
            } // RT_SCOPE RAB_STAGE
            gArena.popTo(mark);
        }

/// @brief The caller's gen-scratch slot — the worker's core id, or the
        ///        registry's last slot at a single-threaded seam.
        /// @return The slot.
/// @details Maintains the documented rule-index ordering and scratch lifetime.
static unsigned currentGenSlot() {
            return (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
        }

/// @brief Flush a staging's rule-index writes into the `HashMemory`
        ///        instance it is bound to — the write half of a window close
        ///        and the removers' entry seam.
        ///
        /// @details
        /// One in-place rebuild per touched map (`flushOwnerSection` ×5 — the
        /// four owner runs and the premise patterns — `flushLmvSection`,
        /// `flushSubkeySection`, `flushRemArgsSection`),
        /// then the staging is emptied and unbound. A no-op on an empty
        /// staging (an O(1) probe) that KEEPS its binding: an empty bound
        /// staging belongs to an open frame that has not staged yet (a
        /// removal flushes mid-window), and only a window close
        /// (`closeRuleIndexStaging`) releases bindings. Runs on the
        /// single-threaded install seam (the LB's worker claim or a post-join
        /// drain) with the slot's gen-scratch arena.
        ///
        /// @param hm The instance; a non-empty @p st must be bound to it.
        /// @param st The staging.
        /// @return Nothing.
        /// @invariant After return `st.empty()` and every map's bytes equal
        ///            the sequential doors' result.
        /// @see `RuleIndexStaging`, `RuleStagingRegistry`, `Memory::ruleStagingArmDepth`.
static void flushRuleIndexStaging(HashMemory& hm, RuleIndexStaging& st) {
            if (st.empty()) return;
            assert(st.boundTo == &hm
                && "flushRuleIndexStaging: the staging is bound to another instance");
            RT_SCOPE_HERE("RULE_STAGING_FLUSH");
            ScratchArena& gArena = genScratchArenas().forSlot(currentGenSlot());
            const ArenaOffset mark = gArena.cursor();
            {
                RT_SCOPE_HERE("FLUSH_WHOLEKEY_OWNERS");
                RT_NOTE_ITERATIONS_HERE(st.wholeKeys.keyCount());
                flushOwnerSection(hm.normalizedEncodedKeys, st.wholeKeys, gArena);
            }
            {
                RT_SCOPE_HERE("FLUSH_ORIGINALS");
                RT_NOTE_ITERATIONS_HERE(st.originals.keyCount());
                flushOwnerSection(hm.originals, st.originals, gArena);
            }
            {
                RT_SCOPE_HERE("FLUSH_COPY_OWNERS");
                RT_NOTE_ITERATIONS_HERE(st.copyOwners.keyCount());
                flushOwnerSection(hm.copyOwners, st.copyOwners, gArena);
            }
            {
                RT_SCOPE_HERE("FLUSH_REMARGS_OWNERS");
                RT_NOTE_ITERATIONS_HERE(st.remArgsOwners.keyCount());
                flushOwnerSection(hm.remainingArgsOwners, st.remArgsOwners, gArena);
            }
            {
                RT_SCOPE_HERE("FLUSH_LMV");
                RT_NOTE_ITERATIONS_HERE(st.lmv.keyCount());
                flushLmvSection(hm.encodedMap, st.lmv, gArena);
            }
            {
                RT_SCOPE_HERE("FLUSH_SUBKEYS");
                RT_NOTE_ITERATIONS_HERE(st.subkeys.keyCount());
                flushSubkeySection(hm.normalizedEncodedSubkeys, st.subkeys, gArena);
            }
            {
                RT_SCOPE_HERE("FLUSH_REMARGS");
                RT_NOTE_ITERATIONS_HERE(st.remArgs.keyCount());
                flushRemArgsSection(hm.remainingArgsNormalizedEncodedMap,
                                    hm.remainingArgsReverseIndex,
                                    st.remArgs, gArena);
            }
            {
                RT_SCOPE_HERE("FLUSH_PREMISE_PATTERNS");
                RT_NOTE_ITERATIONS_HERE(st.premisePatterns.keyCount());
                flushOwnerSection(hm.premisePatterns, st.premisePatterns, gArena);
            }
            st.clear();
            st.boundTo = &hm;   // the binding survives a mid-window flush; only a window close releases it
            gArena.popTo(mark);
        }

/// @brief Flush the staging bound to one instance on this slot, if any
        ///        — the removers' entry seam (mid-window: the binding stays).
        ///
        /// @param hm The instance.
        /// @return Nothing.
        /// @see `flushRuleIndexStaging(HashMemory&, RuleIndexStaging&)`.
/// @details Maintains the documented rule-index ordering and scratch lifetime.
static void flushRuleIndexStagingFor(HashMemory& hm) {
            RuleIndexStaging* st = ruleStagings().boundTo(currentGenSlot(), &hm);
            if (st != nullptr) flushRuleIndexStaging(hm, *st);
        }

/// @brief Close the LB's staging window on this slot: flush the
        ///        stagings bound to its four hash memories and release their
        ///        bindings — the closer of the absorb / apply windows and of
        ///        the outermost install.
        ///
        /// @details
        /// All four instances, because a window's nested processing rounds
        /// stage into any of them and defer to this closer. Only this LB's
        /// instances: a staging bound to another LB belongs to that LB's
        /// still-open window further up the call stack (an install's nested
        /// deposit can run a whole processing round on another LB); the
        /// task-end seams assert that every window closed.
        ///
        /// @param mb The LB.
        /// @return Nothing.
        /// @see `flushRuleIndexStagingFor`, `RuleStagingRegistry::slotIsEmpty`.
static void closeRuleIndexStaging(Memory& mb) {
            const unsigned slot = currentGenSlot();
            HashMemory* const instances[4] = {
                &mb.overallHashMemory, &mb.localHashMemory,
                &mb.localHashMemoryDelta, &mb.workingMemory };
            for (HashMemory* hm : instances) {
                RuleIndexStaging* st = ruleStagings().boundTo(slot, hm);
                if (st == nullptr) continue;
                flushRuleIndexStaging(*hm, *st);
                st->boundTo = nullptr;
            }
            refreshUsedAsPremise(mb);
        }

/// @brief Look a normalized key up in a cold normalized-key container by
        ///        raw bytes — the shared lookup core of both request-generation
        ///        probes (the whole-key set and the subkey record map).
        ///
        /// @details
        /// Builds the `Codec<NormKey>` byte probe (`numberExpressions`, `length`,
        /// `data`) on the stack and `lookup`s it on the cold blob map. Reads no
        /// value; mints nothing; burst-safe on the shared LB
        /// ([I-83](../30_invariants.md#i-83)).
        ///
        /// @tparam ColdMap  `TypedColdSet<NormKey>` (the whole-key set) or
        ///                  `TypedColdBlobMap<NormKey, OwnerSet>` (the subkey map)
        ///                  — both expose the raw-bytes `lookup(StrSpan)`.
        /// @param map       The normalized-key container (`normalizedEncoded*`).
        /// @param keyBuf    The normalized-key `NameId` payload.
        /// @param keyLen    The payload length (`<= MAX_KEY_SLOTS`).
        /// @param exprCount Premise count == the key's `numberExpressions`.
        /// @return The key's id, `0` when absent (a defined miss).
        /// @see `wholeKeyPresent`, `subkeyUSatisfied`.
template <class ColdMap>
        static int32_t normKeyLookup(
            const ColdMap& map,
            const NameId* keyBuf, NameId keyLen, NameId exprCount) {
            NameId probe[2 + ExecutionParameters::MAX_KEY_SLOTS];
            probe[0] = exprCount;
            probe[1] = keyLen;
            std::memcpy(probe + 2, keyBuf,
                        static_cast<std::size_t>(keyLen) * sizeof(NameId));
            return map.lookup(StrSpan(
                reinterpret_cast<const char*>(probe),
                static_cast<std::size_t>(2 + keyLen) * sizeof(NameId)));
        }

/// @brief Record `nk` under the remaining-arg set `argSet` in the cold
        ///        secondary index, keeping the key's NormKey run sorted-unique —
        ///        0% heap.
        ///
        /// @details The heap-free cold read-modify-write replacement for
        ///          `remainingArgsNormalizedEncodedMap[argSet].insert(nk)`. The
        ///          `Int16SetKey` key bytes (`count ++ ascending ids`) are built
        ///          in a stack buffer, and `nk`'s record bytes
        ///          (`numberExpressions ++ length ++ data`) on `gArena` — both
        ///          byte-identical to their `Codec` `encode`/`serialize`. The
        ///          existing run is peeked zero-decode into the arena, `nk`'s
        ///          sorted position found by comparing the serialized blobs on
        ///          `(numberExpressions, data)` (the former `less`), and — unless
        ///          `nk` is already present (the set's no-op insert, asserted
        ///          byte-equal on a comparator tie) — the run is concatenated
        ///          with `nk` spliced in and written back via the raw
        ///          `inner().assignRun` door. A run whose concatenated bytes would
        ///          exceed one arena block (`kMaxAdmissionRunBytes`) takes the
        ///          widening door instead: `assignRun` opens the run empty, then
        ///          per-blob `appendBlobToRun` lands each blob at the run-end (the
        ///          map's blob pool straddles pages), byte-identical run content to
        ///          the one-block concat. The run stays sorted so the deload
        ///          bytes and the sacred dump stay canonical. Structurally the
        ///          sibling of `insertAdmissionBlobSorted`; byte-identical to the
        ///          former `std::vector<NormKey>` RMW.
        ///
        /// @param map    The secondary index (`Int16SetKey` -> a NormKey run).
        /// @param rev    The derived reverse membership index for @p map,
        ///               maintained in lockstep: one `appendEdge` when `nk` is
        ///               newly added to `argSet`'s run (never on the dup no-op).
        /// @param argSet The remaining-arg id set (the key, ascending).
        /// @param nk     The normalized key to record under `argSet`.
        /// @param gArena Per-slot gen-scratch arena for the RMW (cursor/popTo
        ///               framed; caller supplies it — never the map's own arena).
        /// @invariant The run is kept sorted-unique by `(numberExpressions,
        ///            data)`; a comparator-equal record is byte-equal (Codec
        ///            injectivity), asserted.
        /// @see insertAdmissionBlobSorted (the cold
        ///      RMW / raw-door siblings), Codec<Int16SetKey>, Codec<NormKey>.
static void insertRemainingArgsNormKey(
                TypedColdBlobMap<Int16SetKey, NormKey>& map,
                ReverseArgsIndex& rev,
                const std::set<NameId>& argSet, const NormKey& nk,
                ScratchArena& gArena) {
            assert(argSet.size()
                    <= static_cast<std::size_t>(ExecutionParameters::MAX_KEY_SLOTS)
                && "insertRemainingArgsNormKey: argSet exceeds MAX_KEY_SLOTS");
            NameId argBuf[ExecutionParameters::MAX_KEY_SLOTS];
            int32_t argN = 0;
            for (const NameId v : argSet) argBuf[argN++] = v;
            insertRemainingArgsNormKey(map, rev, argBuf, argN, nk.numberExpressions,
                nk.data.empty() ? nullptr : nk.data.data(),
                static_cast<int32_t>(nk.data.size()), gArena);
        }

/// @brief Raw-run overload of @ref insertRemainingArgsNormKey — the key
        ///        set + the normalized key both passed as int runs, no owning
        ///        `std::set<NameId>` / `NormKey`.
        ///
        /// @details
        /// Byte-identical to the `std::set<NameId>` / `NormKey` form (which
        /// forwards here): the `Int16SetKey` key bytes are built from
        /// `argSet[0..argN)` (ascending, the set's iteration order) and the
        /// `Codec<NormKey>` record bytes from `(numberExpressions, nkData, nkDataN)`,
        /// then the run is peeked, `nk` spliced at its sorted position (or the
        /// dup no-op), and written back through the raw `inner().assignRun` /
        /// `appendBlobToRun` doors — all exactly as before. Single-threaded
        /// install (I-83).
        ///
        /// @param map              The secondary index.
        /// @param rev              The derived reverse membership index for
        ///                         @p map; on the `!dup` insert this records one
        ///                         `appendEdge(nk bytes -> the key's id)`, so the
        ///                         reverse index stays exactly the inverse of the
        ///                         forward runs (I-154).
        /// @param argSet           The remaining-arg id run (ascending).
        /// @param argN             The run length.
        /// @param numberExpressions The normalized key's leading count field.
        /// @param nkData           The normalized key's int16 payload.
        /// @param nkDataN          The payload length.
        /// @param gArena           Per-slot gen-scratch arena for the RMW.
        /// @see insertRemainingArgsNormKey(…, const std::set<NameId>&, const
        ///      NormKey&, …) — the owning forwarder; encodeNormKeyInto.
static void insertRemainingArgsNormKey(
                TypedColdBlobMap<Int16SetKey, NormKey>& map,
                ReverseArgsIndex& rev,
                const NameId* argSet, int32_t argN,
                NameId numberExpressions, const NameId* nkData, int32_t nkDataN,
                ScratchArena& gArena) {
            const ArenaOffset mark = gArena.cursor();

            // 1. Int16SetKey bytes (count ++ ascending ids) in a stack buffer —
            //    byte-identical to Codec<Int16SetKey>::encode.
            assert(argN >= 0 && argN <= ExecutionParameters::MAX_KEY_SLOTS
                && "insertRemainingArgsNormKey: argSet exceeds MAX_KEY_SLOTS");
            NameId keyBuf[1 + ExecutionParameters::MAX_KEY_SLOTS];
            int32_t kn = 0;
            keyBuf[kn++] = static_cast<NameId>(argN);
            for (int32_t i = 0; i < argN; ++i) keyBuf[kn++] = argSet[i];
            const StrSpan keyBytes(reinterpret_cast<const char*>(keyBuf),
                kn * static_cast<int32_t>(sizeof(NameId)));

            // 2. nk record bytes (numberExpressions ++ length ++ data) on the
            //    arena — byte-identical to Codec<NormKey>::serialize.
            const int32_t nkLen = static_cast<int32_t>(
                (static_cast<std::size_t>(nkDataN) + 2) * sizeof(NameId));
            const ArenaOffset nkOff = gArena.alloc(nkLen,
                static_cast<int32_t>(alignof(NameId)));
            {
                NameId* p = reinterpret_cast<NameId*>(gArena.resolve(nkOff));
                p[0] = numberExpressions;
                p[1] = static_cast<NameId>(nkDataN);
                for (int32_t i = 0; i < nkDataN; ++i)
                    p[i + 2] = nkData[i];
            }
            const char* nkBytes =
                reinterpret_cast<const char*>(gArena.resolve(nkOff));

            // Compare two serialized NormKey blobs by (numberExpressions, then
            // data lexicographic) — the former `less` on the decoded form.
            const auto blobCmp = [](const char* a, int32_t aLen,
                                    const char* b, int32_t bLen) -> int {
                NameId neA = 0, neB = 0;
                std::memcpy(&neA, a, sizeof(NameId));
                std::memcpy(&neB, b, sizeof(NameId));
                if (neA != neB) return (neA < neB) ? -1 : 1;
                const int32_t nA =
                    aLen / static_cast<int32_t>(sizeof(NameId)) - 2;
                const int32_t nB =
                    bLen / static_cast<int32_t>(sizeof(NameId)) - 2;
                const int32_t nMin = (nA < nB) ? nA : nB;
                for (int32_t i = 0; i < nMin; ++i) {
                    NameId da = 0, db = 0;
                    std::memcpy(&da, a + sizeof(NameId) * (i + 2),
                                sizeof(NameId));
                    std::memcpy(&db, b + sizeof(NameId) * (i + 2),
                                sizeof(NameId));
                    if (da != db) return (da < db) ? -1 : 1;
                }
                if (nA != nB) return (nA < nB) ? -1 : 1;  // shorter is less
                return 0;
            };

            {
                struct BlobRec { ArenaOffset off; int32_t len; };
                DirtyState recsDirty = DirtyState::Clean;
                PagedVector<BlobRec> recs(&gArena, &recsDirty);

                // 3. Peek the existing run's records into the arena (zero-decode).
                const int32_t id = map.inner().lookup(keyBytes);
                if (id != 0) {
                    const int32_t rl = map.runLen(id);
                    for (int32_t j = 0; j < rl; ++j) {
                        int32_t bl = 0;
                        const char* bp = map.peekRecordBytes(id, j, bl, gArena);
                        const ArenaOffset off = gArena.alloc(bl, 1);
                        std::memcpy(gArena.resolve(off), bp,
                                    static_cast<std::size_t>(bl));
                        recs.push_back(BlobRec{ off, bl });
                    }
                }

                // 4. Find nk's sorted position; a comparator tie is the no-op.
                const int32_t M = recs.size();
                int32_t pos = 0;
                bool dup = false;
                for (; pos < M; ++pos) {
                    const BlobRec br = recs[pos];
                    const char* ev =
                        reinterpret_cast<const char*>(gArena.resolve(br.off));
                    const int c = blobCmp(ev, br.len, nkBytes, nkLen);
                    if (c < 0) continue;             // existing < new
                    if (c == 0) {
                        dup = (br.len == nkLen)
                            && std::memcmp(ev, nkBytes,
                                           static_cast<std::size_t>(nkLen)) == 0;
                        assert(dup && "comparator-equal NormKey blobs differ in "
                                      "bytes — Codec injectivity broken");
                    }
                    break;
                }

                // 5. Splice nk into the run at pos. The common case builds the
                //    whole run in one gArena block and writes it with one raw
                //    assignRun. A run whose bytes would straddle an arena block
                //    (an arena alloc may not exceed one block) widens to the
                //    chunked door: open k's run empty, then append each blob in
                //    sorted order through appendBlobToRun (the map's blob pool
                //    straddles pages, so the run there is unbounded). Either path
                //    lands the same sorted blobs in the same run-end order, so the
                //    deload run content is byte-identical. A dup is the former set
                //    no-op insert.
                if (!dup) {
                    // The forward-map key id, captured from whichever write door
                    // runs — the reverse edge below records (nk bytes -> this id).
                    int32_t writtenKeyId = 0;
                    const int32_t outCount = M + 1;
                    int64_t total = nkLen;
                    for (int32_t j = 0; j < M; ++j) total += recs[j].len;
                    if (total <= ExecutionParameters::kMaxAdmissionRunBytes) {
                        const ArenaOffset concatOff =
                            gArena.alloc(static_cast<int32_t>(total), 1);
                        const ArenaOffset lensOff = gArena.alloc(
                            outCount * static_cast<int32_t>(sizeof(int32_t)),
                            static_cast<int32_t>(alignof(int32_t)));
                        char* concat =
                            reinterpret_cast<char*>(gArena.resolve(concatOff));
                        int32_t* lens =
                            reinterpret_cast<int32_t*>(gArena.resolve(lensOff));
                        int32_t at = 0, oi = 0;
                        for (int32_t j = 0; j <= M; ++j) {
                            if (j == pos) {
                                std::memcpy(concat + at, nkBytes,
                                            static_cast<std::size_t>(nkLen));
                                lens[oi++] = nkLen; at += nkLen;
                            }
                            if (j == M) break;
                            const BlobRec br = recs[j];
                            std::memcpy(concat + at, gArena.resolve(br.off),
                                        static_cast<std::size_t>(br.len));
                            lens[oi++] = br.len; at += br.len;
                        }
                        assert(oi == outCount
                            && at == static_cast<int32_t>(total)
                            && "insertRemainingArgsNormKey: concat diverged from "
                               "the computed frame");
                        writtenKeyId =
                            map.inner().assignRun(keyBytes, concat, lens, outCount);
                    } else {
                        // Widening path: assignRun with M==0 opens (or clears) k's
                        // run empty and returns its id; per-blob appendBlobToRun
                        // then lands each blob at the run-end, byte-identical to the
                        // one-block concat's assignRun. gArena is not mutated by
                        // these map writes, so nkBytes and every recs[j] copy stay
                        // valid across the loop.
                        const int32_t rid =
                            map.inner().assignRun(keyBytes, nullptr, nullptr, 0);
                        for (int32_t j = 0; j <= M; ++j) {
                            if (j == pos)
                                map.inner().appendBlobToRun(rid, nkBytes, nkLen);
                            if (j == M) break;
                            const BlobRec br = recs[j];
                            map.inner().appendBlobToRun(
                                rid, gArena.resolve(br.off), br.len);
                        }
                        writtenKeyId = rid;
                    }
                    // Reverse edge: nk was newly added to this key's run, so
                    // record (nk bytes -> the key id). Never on the dup no-op
                    // (this is the !dup branch), so the reverse chain stays
                    // duplicate-free. nkBytes is still valid (gArena.popTo runs
                    // below), and appendEdge copies the bytes into the reverse
                    // interner's own byte pool.
                    assert(writtenKeyId >= 1
                        && "insertRemainingArgsNormKey: assignRun returned a "
                           "null key id");
                    rev.appendEdge(StrSpan(nkBytes, nkLen), writtenKeyId);
                }
            }
            gArena.popTo(mark);
        }

/// @brief Install a WHOLE BATCH of normalized keys into one
        ///        `Int16SetKey` key's run in ONE read-modify-write — the batched
        ///        peer of @ref insertRemainingArgsNormKey.
        ///
        /// @details
        /// The install-site permutation loops emit many normalized keys under the
        /// SAME (loop-invariant) remaining-arg key; feeding them one at a time
        /// pays a whole-run RMW (peek → sorted splice → `assignRun` → blob-pool
        /// tail memmove) PER record. This collapses them into ONE: peek the
        /// existing run ONCE, sorted-unique-merge it with the (sorted-unique)
        /// batch ONCE, and write it back with ONE `assignRun` (contiguous) or
        /// ONE `assignRunGenerated` (a segmented run wider than one arena block).
        ///
        /// Byte-identical to N sequential @ref insertRemainingArgsNormKey calls
        /// (the retained per-record path is the oracle): the final run is the
        /// sorted-unique union of the existing run and the batch — order-
        /// independent, so the stored bytes match regardless of how the union is
        /// assembled; the comparator is the same `(numberExpressions, data)` lex
        /// (a comparator tie is byte-equal by `Codec<NormKey>` injectivity,
        /// asserted). A reverse edge is appended for each record that is NEW
        /// (present in the batch, absent from the existing run) — exactly the
        /// records the sequential path's `!dup` branch would edge, so the reverse
        /// index answer is identical (its chain order is not observable). When no
        /// record is new (every batch key already present), the run is unchanged
        /// and NO `assignRun` runs — matching the sequential all-dup no-op.
        ///
        /// Single-threaded install only (I-83). Zero heap: the batch blobs, the
        /// peeked-run descriptors, the sort columns, and the merged concat all
        /// ride @p gArena.
        ///
        /// @param map    The secondary index (`Int16SetKey` -> a NormKey run).
        /// @param rev    The derived reverse membership index for @p map; one
        ///               `appendEdge` per NEW record.
        /// @param owners The edge owner map (`remainingArgsOwners`): every
        ///               distinct `(argSet, NormKey)` edge of the batch gains
        ///               @p owner, new or already present in the run.
        /// @param argSet The shared remaining-arg key id run (ascending).
        /// @param argN   The key run length.
        /// @param batch  The accumulated NormKey blobs (byte-bump `(off,len)` on
        ///               @p gArena), in permutation order; may contain duplicates.
        /// @param owner  The installing rule's packed owner.
        /// @param gArena Per-slot gen-scratch arena carrying the batch blobs and
        ///               the RMW scratch (`cursor`/`popTo` framed here).
        /// @invariant The run stays sorted-unique by `(numberExpressions, data)`;
        ///            a comparator-equal pair is byte-equal (asserted).
        /// @see @ref insertRemainingArgsNormKey (the per-record oracle),
        ///      @ref RemArgsBatchBlob, `ReverseArgsIndex`, `remArgsEdgeKeyInto`.
        /// @brief Build the owner-map key of one remaining-args EDGE — the
        ///        arg-set id run followed by the NormKey blob, as one
        ///        `IdVecKey`-shaped id run (`int32 count ++ ids`), on @p gArena.
        ///
        /// @details
        /// `remainingArgsOwners` keys every `(argSet, NormKey)` edge of the
        /// forward map `remainingArgsNormalizedEncodedMap` by these bytes:
        /// `Codec<IdVecKey>::encode` of the ids `[argN, args..., numberExpressions,
        /// dataLen, data...]` — the arg-set key's own `NameId` run followed by
        /// the NormKey blob's `NameId` run. Both halves are whole `NameId`
        /// words, so the concatenation is injective given `argN` up front.
        /// The bytes are written into the CALLER'S bounded stack buffer
        /// @p out (`kMaxRemArgsEdgeKeyBytes`), never onto an arena: an arena
        /// allocation's bytes live at the offset `alloc` RETURNS, and
        /// `LbArena::alloc` aligns the cursor and pads to the next block
        /// boundary when the request would straddle a block, so a key built
        /// at the cursor and read back at a pre-allocation cursor mark reads
        /// the poisoned tail of the previous block whenever the build wrapped
        /// — the G-72 abort (the edge key the removal looked up was
        /// `0xCDCDCDCD…`, the arena poison). Zero heap.
        ///
        /// @param argKeyIds   The arg-set key run: `[argN, args...]`.
        /// @param argKeyN     Its length in `NameId`s (`1 + argN`).
        /// @param normKeyBlob The NormKey blob bytes (`Codec<NormKey>` form).
        /// @param normKeyLen  The blob length in bytes (a multiple of `sizeof(NameId)`).
        /// @param out         The caller's key buffer (`kMaxRemArgsEdgeKeyBytes`).
        /// @param outCap      Its capacity in bytes (asserted sufficient, Rule 19).
        /// @return The key's byte length.
        /// @invariant The key bytes are complete in @p out on return; no arena
        ///            state changes.
        /// @see `insertRemainingArgsNormKeyBatch`, `Codec<IdVecKey>`, G-72.
static int32_t remArgsEdgeKeyInto(const NameId* argKeyIds, int32_t argKeyN,
                                          const char* normKeyBlob, int32_t normKeyLen,
                                          char* out, int32_t outCap) {
            assert(normKeyLen % static_cast<int32_t>(sizeof(NameId)) == 0
                && "remArgsEdgeKeyInto: NormKey blob is not a whole NameId run");
            assert(argKeyN >= 1 && argKeyN <= 1 + ExecutionParameters::MAX_KEY_SLOTS
                && "remArgsEdgeKeyInto: arg-set run exceeds MAX_KEY_SLOTS");
            const int32_t nkN = normKeyLen / static_cast<int32_t>(sizeof(NameId));
            const int32_t count = argKeyN + nkN;
            const int32_t bytes = (1 + count) * static_cast<int32_t>(sizeof(int32_t));
            assert(bytes <= outCap
                && "remArgsEdgeKeyInto: edge key exceeds the caller's buffer (kMaxRemArgsEdgeKeyBytes)");
            std::memcpy(out, &count, sizeof(int32_t));
            std::memcpy(out + sizeof(int32_t), argKeyIds,
                        static_cast<std::size_t>(argKeyN) * sizeof(NameId));
            std::memcpy(out + sizeof(int32_t) * (1 + argKeyN), normKeyBlob,
                        static_cast<std::size_t>(normKeyLen));
            return bytes;
        }


/// @details Maintains the documented rule-index ordering and scratch lifetime.
/// @param map Forward remaining-argument index.
/// @param rev Reverse remaining-argument index.
/// @param owners Owner records for each indexed edge.
/// @param argSet Sorted remaining-argument identifiers.
/// @param argN Number of remaining-argument identifiers.
/// @param batch Serialized normalized-key records on the scratch arena.
/// @param owner Rule and scope owning each inserted record.
/// @param gArena Caller-owned scratch arena retaining all batch records.
static void insertRemainingArgsNormKeyBatch(
                TypedColdBlobMap<Int16SetKey, NormKey>& map,
                ReverseArgsIndex& rev,
                TypedColdBlobMap<IdVecKey, RuleOwnerRec>& owners,
                const NameId* argSet, int32_t argN,
                const PagedVector<RemArgsBatchBlob>& batch,
                RuleOwner owner,
                ScratchArena& gArena) {
            const int32_t batchN = batch.size();
            if (batchN == 0) return;   // no keys accumulated (all permutations
                                       // skipped) — a defined no-op, not a fallback
            const ArenaOffset mark = gArena.cursor();

            // 1. Int16SetKey bytes (count ++ ascending ids) — byte-identical to
            //    Codec<Int16SetKey>::encode, exactly as the per-record path.
            assert(argN >= 0 && argN <= ExecutionParameters::MAX_KEY_SLOTS
                && "insertRemainingArgsNormKeyBatch: argSet exceeds MAX_KEY_SLOTS");
            NameId keyBuf[1 + ExecutionParameters::MAX_KEY_SLOTS];
            int32_t kn = 0;
            keyBuf[kn++] = static_cast<NameId>(argN);
            for (int32_t i = 0; i < argN; ++i) keyBuf[kn++] = argSet[i];
            const StrSpan keyBytes(reinterpret_cast<const char*>(keyBuf),
                kn * static_cast<int32_t>(sizeof(NameId)));

            // Compare two serialized NormKey blobs by (numberExpressions, then
            // data lexicographic) — the per-record path's `less` twin.
            const auto blobCmp = [](const char* a, int32_t aLen,
                                    const char* b, int32_t bLen) -> int {
                return remArgsNormKeyBlobCmp(a, aLen, b, bLen);
            };
            {
                // 2. Cache the paged batch descriptors into contiguous byte-bump
                //    columns, sort an index over those columns, then compact
                //    comparator-equal duplicates in that same index. std::sort's
                //    O(N log N) comparator must not repeatedly resolve the paged
                //    source. Each column has the same one-block bound as the
                //    former contiguous index.
                int32_t* bidx = reinterpret_cast<int32_t*>(gArena.resolve(
                    gArena.alloc(batchN * static_cast<int32_t>(sizeof(int32_t)),
                                 static_cast<int32_t>(alignof(int32_t)))));
                ArenaOffset* batchOffs = reinterpret_cast<ArenaOffset*>(
                    gArena.resolve(gArena.alloc(
                        batchN * static_cast<int32_t>(sizeof(ArenaOffset)),
                        static_cast<int32_t>(alignof(ArenaOffset)))));
                int32_t* batchLens = reinterpret_cast<int32_t*>(gArena.resolve(
                    gArena.alloc(batchN * static_cast<int32_t>(sizeof(int32_t)),
                                 static_cast<int32_t>(alignof(int32_t)))));
                for (int32_t i = 0; i < batchN; ++i) {
                    bidx[i] = i;
                    batchOffs[i] = batch[i].off;
                    batchLens[i] = batch[i].len;
                }
                const auto resolveBatch = [&](int32_t j) -> const char* {
                    return reinterpret_cast<const char*>(
                        gArena.resolve(batchOffs[j]));
                };
                int32_t dN = 0;
                {
                RT_SCOPE_HERE("RAB_SORT_DEDUP");
                std::sort(bidx, bidx + batchN, [&](int32_t a, int32_t b) {
                    return blobCmp(resolveBatch(a), batchLens[a],
                                   resolveBatch(b), batchLens[b]) < 0;
                });
                for (int32_t k = 0; k < batchN; ++k) {
                    const int32_t j = bidx[k];
                    if (dN > 0) {
                        const int32_t prev = bidx[dN - 1];
                        if (blobCmp(resolveBatch(prev), batchLens[prev],
                                    resolveBatch(j), batchLens[j]) == 0) {
                            assert(batchLens[prev] == batchLens[j]
                                && std::memcmp(resolveBatch(prev), resolveBatch(j),
                                    static_cast<std::size_t>(batchLens[j])) == 0
                                && "comparator-equal NormKey blobs differ in bytes "
                                   "— Codec injectivity broken");
                            continue;   // dup within batch — one edge, not two
                        }
                    }
                    bidx[dN++] = j;
                }
                } // RT_SCOPE RAB_SORT_DEDUP

                // 2b. Owner edges: every distinct (argSet, NormKey) edge of the
                //     batch records this rule as an owner, whether the NormKey
                //     is new to the run or already present (a second rule
                //     sharing the edge). The edge key is the arg-set run ++ the
                //     NormKey blob as one IdVecKey-shaped id run.
                {
                RT_SCOPE_HERE("RAB_OWNER_EDGES");
                RT_NOTE_ITERATIONS_HERE(dN);
                for (int32_t k = 0; k < dN; ++k) {
                    const int32_t bj = bidx[k];
                    char edgeBuf[kMaxRemArgsEdgeKeyBytes];
                    const int32_t edgeLen = remArgsEdgeKeyInto(
                        keyBuf, kn, resolveBatch(bj), batchLens[bj],
                        edgeBuf, kMaxRemArgsEdgeKeyBytes);
                    const StrSpan edgeKey(edgeBuf, edgeLen);
                    addOwnerToRun(owners, edgeKey, owner, gArena);
                }
                } // RT_SCOPE RAB_OWNER_EDGES

                // 3. Merge-count pass. Walk the existing run in place and mark
                //    which sorted batch records are genuinely new. No paged
                //    BlobRec/MergedRec scratch is materialized: this pass computes
                //    the exact output shape, and the write pass replays the same
                //    canonical two-pointer merge directly into its destination.
                char* batchNew = reinterpret_cast<char*>(gArena.resolve(
                    gArena.alloc(dN, 1)));
                std::memset(batchNew, 0, static_cast<std::size_t>(dN));
                int32_t existId = 0;
                int32_t M = 0;
                int32_t newCount = 0;
                int64_t total = 0;
                {
                RT_SCOPE_HERE("RAB_MERGE_COUNT");
                existId = map.inner().lookup(keyBytes);
                M = (existId != 0) ? map.runLen(existId) : 0;
                RT_NOTE_ITERATIONS_HERE(M);   // existing records walked
                int32_t k = 0;
                if (M > 0) {
                    map.inner().forEachBlobContiguous(
                        existId, gArena,
                        [&](const char* existingPtr, int32_t existingLen) {
                        while (k < dN) {
                            const int32_t bj = bidx[k];
                            const int cmp = blobCmp(
                                existingPtr, existingLen,
                                resolveBatch(bj), batchLens[bj]);
                            if (cmp < 0) break;
                            if (cmp == 0) {
                                assert(existingLen == batchLens[bj]
                                    && std::memcmp(
                                        existingPtr, resolveBatch(bj),
                                        static_cast<std::size_t>(batchLens[bj])) == 0
                                    && "comparator-equal NormKey blobs differ in "
                                       "bytes — Codec injectivity broken");
                                ++k;
                                break;
                            }
                            batchNew[k] = 1;
                            total += batchLens[bj];
                            ++newCount;
                            ++k;
                        }
                        total += existingLen;
                    });
                }
                while (k < dN) {
                    const int32_t bj = bidx[k];
                    batchNew[k] = 1;
                    total += batchLens[bj];
                    ++newCount;
                    ++k;
                }
                } // RT_SCOPE RAB_MERGE_COUNT

                // 4. Nothing new -> the run is unchanged; skip the write entirely
                //    (the sequential all-dup no-op). This also covers an existing
                //    key whose every batch record was already present.
                if (newCount == 0) {
                    gArena.popTo(mark);
                    return;
                }

                const int32_t outCount = M + newCount;
                assert(total <= 0x7fffffffLL
                    && "insertRemainingArgsNormKeyBatch: merged run exceeds the "
                       "int32 blob-pool address space");

                // Replay the canonical merge straight from the resident map into
                // a caller sink. The map remains unchanged throughout this replay,
                // so its direct pool views are valid until assignRun begins.
                const auto emitMergedFromMap = [&](const auto& sink) {
                    int32_t bk = 0, emitted = 0;
                    if (M > 0) {
                        map.inner().forEachBlobContiguous(
                            existId, gArena, [&](const char* ep, int32_t el) {
                            while (bk < dN) {
                                const int32_t bj = bidx[bk];
                                const int cmp = blobCmp(
                                    ep, el, resolveBatch(bj), batchLens[bj]);
                                if (cmp < 0) break;
                                if (cmp == 0) {
                                    assert(batchNew[bk] == 0);
                                    ++bk;
                                    break;
                                }
                                assert(batchNew[bk] != 0);
                                sink(resolveBatch(bj), batchLens[bj]);
                                ++bk; ++emitted;
                            }
                            sink(ep, el);
                            ++emitted;
                        });
                    }
                    while (bk < dN) {
                            assert(batchNew[bk] != 0);
                            const int32_t bj = bidx[bk++];
                            sink(resolveBatch(bj), batchLens[bj]);
                            ++emitted;
                    }
                    assert(emitted == outCount
                        && "insertRemainingArgsNormKeyBatch: replay count diverged");
                };

                // 5. Write the merged run back. A one-block result is emitted
                //    directly into concat+lens. An over-block result first
                //    preserves the existing source records in a linked scratch
                //    chain, then replays that chain through assignRunGenerated.
                int32_t writtenKeyId = 0;
                {
                RT_SCOPE_HERE("RAB_WRITE");
                if (total <= ExecutionParameters::kMaxAdmissionRunBytes) {
                    const ArenaOffset concatOff =
                        gArena.alloc(static_cast<int32_t>(total), 1);
                    const ArenaOffset lensOff = gArena.alloc(
                        outCount * static_cast<int32_t>(sizeof(int32_t)),
                        static_cast<int32_t>(alignof(int32_t)));
                    char* concat = reinterpret_cast<char*>(gArena.resolve(concatOff));
                    int32_t* lens = reinterpret_cast<int32_t*>(gArena.resolve(lensOff));
                    int32_t at = 0, record = 0;
                    emitMergedFromMap([&](const char* ptr, int32_t len) {
                        std::memcpy(concat + at, ptr,
                                    static_cast<std::size_t>(len));
                        lens[record++] = len;
                        at += len;
                    });
                    assert(at == static_cast<int32_t>(total) && record == outCount
                        && "insertRemainingArgsNormKeyBatch: concat diverged from "
                           "the computed frame");
                    writtenKeyId = (existId != 0)
                        ? map.inner().assignRunAtId(
                            existId, concat, lens, outCount)
                        : map.inner().assignRun(
                            keyBytes, concat, lens, outCount);
                } else {
                    struct PreservedRec {
                        ArenaOffset next;
                        int32_t len;
                    };
                    ArenaOffset firstOff = 0, previousOff = 0;
                    int32_t preserved = 0;
                    if (M > 0) map.inner().forEachBlobContiguous(
                        existId, gArena,
                        [&](const char* source, int32_t len) {
                        const ArenaOffset off = gArena.alloc(
                            static_cast<int32_t>(sizeof(PreservedRec)) + len,
                            static_cast<int32_t>(alignof(PreservedRec)));
                        PreservedRec* rec = reinterpret_cast<PreservedRec*>(
                            gArena.resolve(off));
                        rec->len = len;
                        std::memcpy(rec + 1, source,
                                    static_cast<std::size_t>(len));
                        if (preserved == 0) firstOff = off;
                        else {
                            PreservedRec* previous =
                                reinterpret_cast<PreservedRec*>(
                                    gArena.resolve(previousOff));
                            previous->next = off;
                        }
                        previousOff = off;
                        ++preserved;
                    });
                    assert(preserved == M
                        && "insertRemainingArgsNormKeyBatch: preserved scan "
                           "count diverged");
                    const auto emitMergedPreserved = [&](const auto& sink) {
                        int32_t ei = 0, bk = 0, emitted = 0;
                        ArenaOffset currentOff = firstOff;
                        const PreservedRec* rec = nullptr;
                        bool loaded = false;
                        while (ei < M || bk < dN) {
                            if (ei < M && !loaded) {
                                rec = reinterpret_cast<const PreservedRec*>(
                                    gArena.resolve(currentOff));
                                loaded = true;
                            }
                            const char* ep = loaded
                                ? reinterpret_cast<const char*>(rec + 1) : nullptr;
                            const int32_t el = loaded ? rec->len : 0;
                            int cmp;
                            if (ei >= M) cmp = 1;
                            else if (bk >= dN) cmp = -1;
                            else {
                                const int32_t bj = bidx[bk];
                                cmp = blobCmp(ep, el, resolveBatch(bj), batchLens[bj]);
                            }
                            if (cmp < 0 || cmp == 0) {
                                if (cmp == 0) {
                                    assert(batchNew[bk] == 0);
                                    ++bk;
                                }
                                sink(ep, el);
                                ++ei; ++emitted;
                                if (ei < M) currentOff = rec->next;
                                loaded = false;
                            } else {
                                assert(batchNew[bk] != 0);
                                const int32_t bj = bidx[bk++];
                                sink(resolveBatch(bj), batchLens[bj]);
                                ++emitted;
                            }
                        }
                        assert(emitted == outCount
                            && "insertRemainingArgsNormKeyBatch: preserved replay "
                               "count diverged");
                    };
                    writtenKeyId = (existId != 0)
                        ? map.inner().assignRunGeneratedAtId(
                            existId, outCount, static_cast<int32_t>(total),
                            emitMergedPreserved)
                        : map.inner().assignRunGenerated(
                            keyBytes, outCount, static_cast<int32_t>(total),
                            emitMergedPreserved);
                }
                } // RT_SCOPE RAB_WRITE

                // 6. Reverse edges: one per NEW record (batch-only), all pointing
                //    at this key's id — exactly the edges the sequential !dup path
                //    would append.
                assert(writtenKeyId >= 1
                    && "insertRemainingArgsNormKeyBatch: assignRun returned a "
                       "null key id");
                {
                RT_SCOPE_HERE("RAB_REVERSE_EDGES");
                for (int32_t j = 0; j < dN; ++j) {
                    if (batchNew[j] != 0) {
                        const int32_t bj = bidx[j];
                        rev.appendEdge(StrSpan(resolveBatch(bj), batchLens[bj]),
                                       writtenKeyId);
                    }
                }
                }
            }
            gArena.popTo(mark);
        }

private:

/// @brief Compare two serialized `NormKey` blobs in the remaining-args
        ///        run order: `numberExpressions`, then the data
        ///        lexicographically, a shorter key first.
        ///
        /// @details
        /// The one comparator of the remaining-args forward map
        /// (`insertRemainingArgsNormKeyBatch`'s per-record `less` and the
        /// flush's merge share it).
        ///
        /// @param a    First blob (`Codec<NormKey>` bytes).
        /// @param aLen Its length.
        /// @param b    Second blob.
        /// @param bLen Its length.
        /// @return `< 0`, `0`, `> 0` as `a` sorts before, equal to, after `b`.
static int remArgsNormKeyBlobCmp(const char* a, int32_t aLen,
                                         const char* b, int32_t bLen) {
            NameId neA = 0, neB = 0;
            std::memcpy(&neA, a, sizeof(NameId));
            std::memcpy(&neB, b, sizeof(NameId));
            if (neA != neB) return (neA < neB) ? -1 : 1;
            const int32_t nA = aLen / static_cast<int32_t>(sizeof(NameId)) - 2;
            const int32_t nB = bLen / static_cast<int32_t>(sizeof(NameId)) - 2;
            const int32_t nMin = (nA < nB) ? nA : nB;
            for (int32_t i = 0; i < nMin; ++i) {
                NameId da = 0, db = 0;
                std::memcpy(&da, a + sizeof(NameId) * (i + 2), sizeof(NameId));
                std::memcpy(&db, b + sizeof(NameId) * (i + 2), sizeof(NameId));
                if (da != db) return (da < db) ? -1 : 1;
            }
            if (nA != nB) return (nA < nB) ? -1 : 1;   // shorter is less
            return 0;
        }

/// @brief Plan a section flush: look every staged key up once and
        ///        assign the misses their future ids in first-touch order —
        ///        the shared prologue of the four section flushes.
        ///
        /// @details
        /// The misses are assigned `oldCount + 1, oldCount + 2, ...` in
        /// ordinal order, which is the order the sequential writes minted them
        /// (a key's first write is its first staging), so the key ids after
        /// the flush equal the sequential ids. Nothing is minted here: the
        /// rebuild reads the stored runs first (a mint before it would put the
        /// run-start column and the key store out of step), then
        /// `mintStagedKeys` appends the keys.
        ///
        /// @tparam TypedMap A `TypedColdBlobMap` over a byte key.
        /// @param map     The cold map.
        /// @param section Its staging section.
        /// @param ordOfId The caller's empty page-tier table; filled with
        ///                `oldCount + newN` entries.
        /// @return The plan.
        /// @see `mintStagedKeys`, `HashMap::assignAllRunsGenerated`.
template <class TypedMap>
        static StagedFlushPlan planStagedFlush(TypedMap& map,
                                               const RuleIndexStagingSection& section,
                                               PagedVector<int32_t>& ordOfId) {
            assert(ordOfId.empty() && "planStagedFlush: the plan table must start empty");
            StagedFlushPlan plan{ &ordOfId, map.count(), 0 };
            const int32_t kN = section.keyCount();
            for (int32_t i = 0; i < plan.oldCount; ++i) ordOfId.push_back(0);
            for (int32_t ord = 1; ord <= kN; ++ord) {
                const int32_t id = map.inner().lookup(section.keyAt(ord));
                if (id != 0) {
                    assert(ordOfId[id - 1] == 0
                        && "planStagedFlush: two staged keys resolve to one id");
                    ordOfId.setAt(id - 1, ord);
                } else {
                    ordOfId.push_back(ord);
                    ++plan.newN;
                }
            }
            return plan;
        }

/// @brief Mint the keys a section flush adds, in the plan's id order —
        ///        the epilogue of the four section flushes, after the rebuild.
        ///
        /// @tparam TypedMap A `TypedColdBlobMap` over a byte key.
        /// @param map     The cold map (its runs already rebuilt for
        ///                `plan.oldCount + plan.newN` keys).
        /// @param section The staging section.
        /// @param plan    The flush plan.
        /// @return Nothing.
        /// @invariant Every mint receives exactly the id the plan assigned.
        /// @see `planStagedFlush`.
/// @details Maintains the documented rule-index ordering and scratch lifetime.
template <class TypedMap>
        static void mintStagedKeys(TypedMap& map,
                                   const RuleIndexStagingSection& section,
                                   const StagedFlushPlan& plan) {
            for (int32_t i = 0; i < plan.newN; ++i) {
                const int32_t ord = (*plan.ordOfId)[plan.oldCount + i];
                const int32_t id = map.inner().mint(section.keyAt(ord));
                assert(id == plan.oldCount + i + 1
                    && "mintStagedKeys: a new key minted out of sequence");
            }
            assert(map.count() == plan.oldCount + plan.newN
                && "mintStagedKeys: key count diverged from the plan");
        }

/// @brief Flush an owner-run section into its map: every touched key's
        ///        run becomes the sorted-unique union of its run and the staged
        ///        owners; the others are carried over — one rebuild.
        ///
        /// @tparam TypedMap A `TypedColdBlobMap<K, RuleOwnerRec>` over a byte key.
        /// @param map     The owner map.
        /// @param section Its staging section (emptied by the caller after).
        /// @param gArena  Per-slot gen-scratch arena (page tier for the
        ///                tables, byte-bump for the sort chunks).
        /// @return Nothing.
        /// @invariant Byte-identical to the sequential `addOwnerToRun` calls.
        /// @see `stageOwnerToRun`, `HashMap::assignAllRunsGenerated`.
/// @details Maintains the documented rule-index ordering and scratch lifetime.
template <class TypedMap>
        static void flushOwnerSection(TypedMap& map,
                                      const RuleIndexStagingSection& section,
                                      ScratchArena& gArena) {
            if (section.empty()) return;
            constexpr int32_t kRec = static_cast<int32_t>(sizeof(RuleOwner));
            const ArenaOffset mark = gArena.cursor();
            DirtyState scratchDirty = DirtyState::Clean;
            PagedVector<int32_t> ordOfId(&gArena, &scratchDirty);
            PagedVector<RuleOwner> owners(&gArena, &scratchDirty);   // the current key's staged owners
            const StagedFlushPlan plan = planStagedFlush(map, section, ordOfId);
            const int32_t oldCount = plan.oldCount;
            map.inner().assignAllRunsGenerated(oldCount + plan.newN,
                                               [&](int32_t id, auto&& sink) {
                const int32_t ord = ordOfId[id - 1];
                if (ord == 0) {
                    assert(id <= oldCount);
                    sink.unchanged(id);
                    return;
                }
                const ArenaOffset keyMark = gArena.cursor();
                owners.clear();
                section.forEachItem(ord, [&](const StagedRuleItem& it) {
                    owners.push_back(it.owner);
                });
                StagedSortedUnique pull(gArena, owners.size(),
                    [&](int32_t a, int32_t b) { return owners[a] < owners[b]; },
                    [&](int32_t a, int32_t b) { return owners[a] == owners[b]; });
                int32_t cur = 0;
                bool has = pull.next(cur);
                const auto emitOwner = [&](RuleOwner o) {
                    char b[sizeof(RuleOwner)];
                    std::memcpy(b, &o, sizeof(RuleOwner));
                    sink.blob(b, kRec);
                };
                if (id <= oldCount) {
                    map.inner().forEachBlobContiguous(id, gArena,
                        [&](const char* bp, int32_t bl) {
                            assert(bl == kRec && "flushOwnerSection: owner blob length drift");
                            RuleOwner e;
                            std::memcpy(&e, bp, sizeof(RuleOwner));
                            while (has && owners[cur] < e) {
                                emitOwner(owners[cur]);
                                has = pull.next(cur);
                            }
                            if (has && owners[cur] == e) has = pull.next(cur);   // already on the key
                            sink.blob(bp, bl);
                        });
                }
                while (has) {
                    emitOwner(owners[cur]);
                    has = pull.next(cur);
                }
                gArena.popTo(keyMark);
            }, gArena);
            mintStagedKeys(map, section, plan);
            owners.clear();
            ordOfId.clear();
            gArena.popTo(mark);
        }

/// @brief Flush the LMV section into `encodedMap`: every touched key's
        ///        run becomes its run followed by the staged blobs in staging
        ///        order; the others are carried over — one rebuild.
        ///
        /// @param map     The LMV map.
        /// @param section Its staging section.
        /// @param gArena  Per-slot gen-scratch arena.
        /// @return Nothing.
        /// @invariant Byte-identical to the sequential `appendLmvIdsRecord` calls.
        /// @see `stageLmvIdsRecord`.
/// @details Maintains the documented rule-index ordering and scratch lifetime.
static void flushLmvSection(TypedColdBlobMap<NormKey, LocalMemoryValue>& map,
                                    const RuleIndexStagingSection& section,
                                    ScratchArena& gArena) {
            if (section.empty()) return;
            const ArenaOffset mark = gArena.cursor();
            DirtyState scratchDirty = DirtyState::Clean;
            PagedVector<int32_t> ordOfId(&gArena, &scratchDirty);
            const StagedFlushPlan plan = planStagedFlush(map, section, ordOfId);
            const int32_t oldCount = plan.oldCount;
            map.inner().assignAllRunsGenerated(oldCount + plan.newN,
                                               [&](int32_t id, auto&& sink) {
                const int32_t ord = ordOfId[id - 1];
                if (ord == 0) {
                    assert(id <= oldCount);
                    sink.unchanged(id);
                    return;
                }
                const ArenaOffset keyMark = gArena.cursor();
                if (id <= oldCount) {
                    map.inner().forEachBlobContiguous(id, gArena,
                        [&](const char* bp, int32_t bl) { sink.blob(bp, bl); });
                }
                section.forEachItem(ord, [&](const StagedRuleItem& it) {
                    sink.blob(section.payload(it), it.len);
                });
                gArena.popTo(keyMark);
            }, gArena);
            mintStagedKeys(map, section, plan);
            ordOfId.clear();
            gArena.popTo(mark);
        }

/// @brief Flush the subkey section into `normalizedEncodedSubkeys`:
        ///        every touched key's single `OwnerSet` record becomes the
        ///        merge of the record and the staged contributions; the others
        ///        are carried over — one rebuild.
        ///
        /// @details
        /// Short keys receive the fixed empty record without collecting any
        /// owner contributions. For a long key, the merged record is the one
        /// the sequential doors reach for the
        /// same contributions in any order: the loose byte is the OR of the
        /// record's and every loose contribution; the signature list is the
        /// sorted-unique union (the door's `cmpNewVsSig` order: pair by pair,
        /// a shorter run first); the owner pairs are the sorted-unique union
        /// of the record's pairs (their indexes renumbered into the merged
        /// list) and the staged pairs (owner, index of its signature, -1 for
        /// loose), ascending by owner then index. The per-key tables are page
        /// tier, the sorts block-bounded; the merged record itself rides the
        /// byte-bump tier like the door's.
        ///
        /// @param map     The subkey record map.
        /// @param section Its staging section.
        /// @param gArena  Per-slot gen-scratch arena.
        /// @return Nothing.
        /// @invariant Byte-identical to the sequential `mergeSubkeySignatures` /
        ///            `ensureEmptySubkeyRecord` calls.
        /// @see `stageSubkeySignatures`, `stageEmptySubkeyRecord`, `OwnerSetBlob`.
static void flushSubkeySection(TypedColdBlobMap<NormKey, OwnerSet>& map,
                                       const RuleIndexStagingSection& section,
                                       ScratchArena& gArena) {
            if (section.empty()) return;
            struct SigRef { const char* pairs; int32_t pairCount; };
            struct PairRec { RuleOwner owner; int32_t idx; };
            const auto rd32 = [](const char* q) -> int32_t {
                int32_t v; std::memcpy(&v, q, sizeof(int32_t)); return v;
            };
            const auto sigCmp = [&](const SigRef& x, const SigRef& y) -> int {
                const int32_t common = std::min(x.pairCount, y.pairCount);
                for (int32_t k = 0; k < common; ++k) {
                    const int32_t xf = rd32(x.pairs + 8 * k), yf = rd32(y.pairs + 8 * k);
                    if (xf != yf) return xf < yf ? -1 : 1;
                    const int32_t xs = rd32(x.pairs + 8 * k + 4), ys = rd32(y.pairs + 8 * k + 4);
                    if (xs != ys) return xs < ys ? -1 : 1;
                }
                if (x.pairCount != y.pairCount) return x.pairCount < y.pairCount ? -1 : 1;
                return 0;
            };
            const ArenaOffset mark = gArena.cursor();
            DirtyState scratchDirty = DirtyState::Clean;
            PagedVector<int32_t> ordOfId(&gArena, &scratchDirty);
            PagedVector<StagedRuleItem> items(&gArena, &scratchDirty);
            PagedVector<SigRef> sigs(&gArena, &scratchDirty);
            PagedVector<int32_t> stagedSig(&gArena, &scratchDirty);
            PagedVector<int32_t> mergedPos(&gArena, &scratchDirty);
            PagedVector<int32_t> mergedSig(&gArena, &scratchDirty);
            PagedVector<PairRec> pairs(&gArena, &scratchDirty);
            const StagedFlushPlan plan = planStagedFlush(map, section, ordOfId);
            const int32_t oldCount = plan.oldCount;
            map.inner().assignAllRunsGenerated(oldCount + plan.newN,
                                               [&](int32_t id, auto&& sink) {
                const int32_t ord = ordOfId[id - 1];
                if (ord == 0) {
                    assert(id <= oldCount);
                    sink.unchanged(id);
                    return;
                }
                const StrSpan key = section.keyAt(ord);
                assert(key.len >= static_cast<int32_t>(2 * sizeof(NameId)));
                NameId premiseCount = 0;
                std::memcpy(&premiseCount, key.ptr, sizeof(premiseCount));
                assert(premiseCount > 0);
                if (premiseCount < kSubkeyUCheckMinElements) {
                    const char emptyRecord[9]{};
                    sink.blob(emptyRecord, sizeof(emptyRecord));
                    return;
                }
                const ArenaOffset keyMark = gArena.cursor();
                items.clear();
                sigs.clear();
                stagedSig.clear();
                mergedPos.clear();
                mergedSig.clear();
                pairs.clear();

                // 1. The existing record, zero-copy (a straddle copy lands
                //    inside this frame).
                OwnerSetBlob ob{ nullptr, 0 };
                bool haveExisting = false;
                if (id <= oldCount) {
                    assert(map.runLen(id) == 1
                        && "flushSubkeySection: a subkey key holds one record");
                    int32_t bl = 0;
                    const char* bp = map.peekRecordBytes(id, 0, bl, gArena);
                    ob = OwnerSetBlob{ bp, bl };
                    haveExisting = true;
                }
                const int32_t existingSigCount = haveExisting ? ob.uSigCount() : 0;
                const int32_t existingOwnersOff = haveExisting ? ob.ownersOffset() : 0;
                const int32_t existingOwnerCount =
                    haveExisting ? ob.ownerCount(existingOwnersOff) : 0;
                bool loose = haveExisting && ob.hasLooseOwner();

                // 2. The staged contributions.
                section.forEachItem(ord, [&](const StagedRuleItem& it) { items.push_back(it); });
                const int32_t n = items.size();

                // 3. The signature table: the record's signatures, then every
                //    staged non-loose one; sorted-unique -> merged positions.
                {
                    int32_t off = haveExisting ? ob.firstSigOffset() : 0;
                    for (int32_t s = 0; s < existingSigCount; ++s) {
                        sigs.push_back(SigRef{ ob.p + off + 4, ob.sigPairCount(off) });
                        off += ob.sigBytes(off);
                    }
                }
                for (int32_t i = 0; i < n; ++i) {
                    const StagedRuleItem it = items[i];
                    if (it.flag != 0) {
                        stagedSig.push_back(-1);
                        loose = true;
                    } else {
                        stagedSig.push_back(sigs.size());
                        sigs.push_back(SigRef{ section.payload(it), it.len / 8 });
                    }
                }
                const int32_t sigN = sigs.size();
                for (int32_t i = 0; i < sigN; ++i) mergedPos.push_back(0);
                {
                    ChunkSortedOrdinals sorted(gArena, sigN, 0,
                        [&](int32_t a, int32_t b) { return sigCmp(sigs[a], sigs[b]) < 0; });
                    int32_t prevRep = -1;
                    for (int32_t t = 0; t < sigN; ++t) {
                        const int32_t o = sorted.next();
                        if (prevRep < 0 || sigCmp(sigs[prevRep], sigs[o]) != 0) {
                            mergedSig.push_back(o);
                            prevRep = o;
                        }
                        mergedPos.setAt(o, mergedSig.size() - 1);
                    }
                }
                const int32_t mergedN = mergedSig.size();

                // 4. The owner pairs: existing (renumbered) + staged; sorted-unique.
                for (int32_t i = 0; i < existingOwnerCount; ++i) {
                    const int32_t si = ob.ownerSigIndexAt(existingOwnersOff, i);
                    assert(si >= -1 && si < existingSigCount
                        && "flushSubkeySection: a record pair cites a signature outside its list");
                    pairs.push_back(PairRec{ ob.ownerAt(existingOwnersOff, i),
                                             si < 0 ? -1 : mergedPos[si] });
                }
                for (int32_t i = 0; i < n; ++i) {
                    const int32_t si = stagedSig[i];
                    pairs.push_back(PairRec{ items[i].owner, si < 0 ? -1 : mergedPos[si] });
                }
                StagedSortedUnique pairPull(gArena, pairs.size(),
                    [&](int32_t a, int32_t b) {
                        const PairRec x = pairs[a], y = pairs[b];
                        if (x.owner != y.owner) return x.owner < y.owner;
                        return x.idx < y.idx;
                    },
                    [&](int32_t a, int32_t b) {
                        const PairRec x = pairs[a], y = pairs[b];
                        return x.owner == y.owner && x.idx == y.idx;
                    });
                int32_t pM = 0;
                {
                    int32_t o = 0;
                    while (pairPull.next(o)) ++pM;
                }
                pairPull.reset();

                // 5. The merged record, in the Codec<OwnerSet> layout.
                int32_t total = 1 + 4;
                for (int32_t m = 0; m < mergedN; ++m) total += 4 + 8 * sigs[mergedSig[m]].pairCount;
                total += 4 + OwnerSetBlob::kOwnerRecBytes * pM;
                char* out = reinterpret_cast<char*>(gArena.resolve(gArena.alloc(total, 1)));
                int32_t at = 0;
                const auto wr32 = [&](int32_t v) {
                    std::memcpy(out + at, &v, sizeof(int32_t)); at += 4;
                };
                out[at++] = static_cast<char>(loose ? 1 : 0);
                wr32(mergedN);
                for (int32_t m = 0; m < mergedN; ++m) {
                    const SigRef sr = sigs[mergedSig[m]];
                    wr32(sr.pairCount);
                    if (sr.pairCount > 0) {
                        std::memcpy(out + at, sr.pairs, static_cast<std::size_t>(8 * sr.pairCount));
                        at += 8 * sr.pairCount;
                    }
                }
                wr32(pM);
                {
                    int32_t o = 0;
                    while (pairPull.next(o)) {
                        const PairRec pr = pairs[o];
                        std::memcpy(out + at, &pr.owner, sizeof(RuleOwner)); at += 8;
                        wr32(pr.idx);
                    }
                }
                assert(at == total && "flushSubkeySection: merged record length drift");
                sink.blob(out, total);
                gArena.popTo(keyMark);
            }, gArena);
            mintStagedKeys(map, section, plan);
            pairs.clear();
            mergedSig.clear();
            mergedPos.clear();
            stagedSig.clear();
            sigs.clear();
            items.clear();
            ordOfId.clear();
            gArena.popTo(mark);
        }

/// @brief Flush the remaining-args section into the forward map: every
        ///        touched arg-set key's run becomes the sorted-unique union of
        ///        its run and the staged `NormKey` blobs, and every blob the
        ///        run did not hold gets its reverse-index edge; the others are
        ///        carried over — one rebuild.
        ///
        /// @param map     The remaining-args forward map.
        /// @param rev     Its derived reverse index (I-154).
        /// @param section The forward map's staging section.
        /// @param gArena  Per-slot gen-scratch arena.
        /// @return Nothing.
        /// @invariant Byte-identical forward runs and the same reverse-index
        ///            answers as the sequential `insertRemainingArgsNormKeyBatch`
        ///            calls.
        /// @see `stageRemainingArgsNormKeyBatch`, `remArgsNormKeyBlobCmp`.
/// @details Maintains the documented rule-index ordering and scratch lifetime.
static void flushRemArgsSection(TypedColdBlobMap<Int16SetKey, NormKey>& map,
                                        ReverseArgsIndex& rev,
                                        const RuleIndexStagingSection& section,
                                        ScratchArena& gArena) {
            if (section.empty()) return;
            struct BlobRef { const char* p; int32_t len; };
            const ArenaOffset mark = gArena.cursor();
            DirtyState scratchDirty = DirtyState::Clean;
            PagedVector<int32_t> ordOfId(&gArena, &scratchDirty);
            PagedVector<BlobRef> refs(&gArena, &scratchDirty);
            const StagedFlushPlan plan = planStagedFlush(map, section, ordOfId);
            const int32_t oldCount = plan.oldCount;
            map.inner().assignAllRunsGenerated(oldCount + plan.newN,
                                               [&](int32_t id, auto&& sink) {
                const int32_t ord = ordOfId[id - 1];
                if (ord == 0) {
                    assert(id <= oldCount);
                    sink.unchanged(id);
                    return;
                }
                const ArenaOffset keyMark = gArena.cursor();
                refs.clear();
                section.forEachItem(ord, [&](const StagedRuleItem& it) {
                    refs.push_back(BlobRef{ section.payload(it), it.len });
                });
                StagedSortedUnique pull(gArena, refs.size(),
                    [&](int32_t a, int32_t b) {
                        const BlobRef x = refs[a], y = refs[b];
                        return remArgsNormKeyBlobCmp(x.p, x.len, y.p, y.len) < 0;
                    },
                    [&](int32_t a, int32_t b) {
                        const BlobRef x = refs[a], y = refs[b];
                        return remArgsNormKeyBlobCmp(x.p, x.len, y.p, y.len) == 0;
                    });
                int32_t cur = 0;
                bool has = pull.next(cur);
                const auto emitStaged = [&](int32_t o) {
                    const BlobRef r = refs[o];
                    sink.blob(r.p, r.len);
                    // New to the run: one reverse-index edge, in the write pass
                    // only (the in-place rebuild replays the emitter to count).
                    if (sink.isWritePass()) rev.appendEdge(StrSpan(r.p, r.len), id);
                };
                if (id <= oldCount) {
                    map.inner().forEachBlobContiguous(id, gArena,
                        [&](const char* bp, int32_t bl) {
                            while (has) {
                                const BlobRef r = refs[cur];
                                const int c = remArgsNormKeyBlobCmp(r.p, r.len, bp, bl);
                                if (c < 0) {
                                    emitStaged(cur);
                                    has = pull.next(cur);
                                    continue;
                                }
                                if (c == 0) has = pull.next(cur);   // already in the run: no edge
                                break;
                            }
                            sink.blob(bp, bl);
                        });
                }
                while (has) {
                    emitStaged(cur);
                    has = pull.next(cur);
                }
                gArena.popTo(keyMark);
            }, gArena);
            mintStagedKeys(map, section, plan);
            refs.clear();
            ordOfId.clear();
            gArena.popTo(mark);
        }
};
} // namespace gl
