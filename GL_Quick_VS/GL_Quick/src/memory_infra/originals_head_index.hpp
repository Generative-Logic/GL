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
// OriginalsHeadIndex -- the derived head-operator side-index for the per-LB
// rule-chain map `HashMemory::originals` (`TypedColdBlobMap<IdVecKey,
// RuleOwnerRec>`, one key per installed rule chain, the chain's LAST element
// being the rule head).
//
// `checkNecessityForEquality` needs, for a deposited expression, exactly the
// chains whose head carries the deposit's operator name, in ascending chain-id
// order. Before this index that answer cost a decode of EVERY chain head per
// deposit (the `CNFE` section: 151 K calls over ~100 K chains = 1.1 G head
// decodes on the FTA batch). This container answers it as one hash probe plus
// a walk of the operator's chain list.
//
// It is DERIVED (I-117 pattern, the cold-map family's key->id index): it lives
// on the LB's deloadable `LbArena` (pool-backed, zero heap) but is NEVER
// enrolled in `LbMemory::visitContainers`, NEVER in the canonical deload
// stream, and NEVER dumped -- it is rebuilt lazily from the chain map after a
// canonical reload, exactly as `ReverseArgsIndex` is rebuilt for the
// remaining-args map. The v4 RAW eviction image captures its arena pages
// verbatim, so a raw round-trip restores it without a rebuild.
#include "cold_hash_map.hpp"
#include "paged_vector.hpp"
#include "scratch_arena.hpp"

#include <cassert>
#include <cstdint>

namespace gl {

    /// @brief Derived head-operator -> chain-id index for `HashMemory::originals`.
    ///
    /// @details
    /// The chain map keys an id vector (the rule's premises followed by its
    /// head, `ruleInterner` ids) to a run of owner records. This side-index
    /// groups the chain ids by the operator name of the chain's head, so a
    /// consumer that only ever matches heads of one operator visits that
    /// operator's chains instead of decoding every chain.
    ///
    /// **Substrate (zero heap, derived, throw-away).** Four cooperating pieces,
    /// all on the owning LB's `LbArena` page/byte tier with a private dirty flag:
    /// - `heads_` — a `ColdHashSet<BytesKeyStore>` interning each DISTINCT head
    ///   operator name's bytes to a dense entry id (`1..count()`); membership is
    ///   by exact bytes, so the answer carries no false positives.
    /// - `headById_` / `tailById_` — `PagedVector<int32_t>` columns indexed by
    ///   `entryId - 1`, the first and last node of that operator's chain list
    ///   (`-1` = empty). The tail pointer makes @ref append an O(1) tail insert,
    ///   which keeps the list in ASCENDING chain-id order — the order the
    ///   retired full walk visited (`sortOriginalChainIndex`, id order).
    /// - `nodeChain_` / `nodeNext_` — parallel `PagedVector<int32_t>` columns,
    ///   one node per indexed chain (`nodeChain_[n]` = a chain id of the map;
    ///   `nodeNext_[n]` = the next node or `-1`).
    ///
    /// **Why append-only extension is exact.** Between two owner-less sweeps the
    /// chain ids of `originals` are position ids that never shift: keys enter
    /// the map only through the rule-index staging flush (`flushOwnerSection` →
    /// `mintStagedKeys`, new keys appended AFTER the old count, I-221) and a
    /// rule removal (`removeOwnerFromRun`) rewrites the key's owner run in place
    /// and keeps the key until the end-of-apply compaction
    /// (`eraseOwnerlessEntries`, `eraseBlobIf`) drops the owner-less keys and
    /// RENUMBERS the survivors — that sweep marks this index stale
    /// (@ref markUnbuilt) exactly as it rebuilds `ReverseArgsIndex`. So a built
    /// index stays exact while the map only grows, @ref ensureBuilt extends it
    /// from `builtCount() + 1` to the current count, and a rebuild from chain 1
    /// follows every sweep; the retired walk visited owner-less chains too (they
    /// exist until the sweep), so indexing every key (owners ignored)
    /// reproduces its candidate set exactly.
    ///
    /// **The masked-head family (the exact candidate set).** The consumer
    /// accepts a chain only when its head has the deposit's sign, one constant
    /// (non-`u_`) argument, or two constants beside at least one other
    /// argument, each constant at a `(1)`-definition slot, and every other
    /// argument equal to the deposit's `u_`-prefixed argument at the same
    /// position. That verdict is a key equality: the chain's MASKED HEAD KEY —
    /// the sign byte of a negated head, the head operator name, then its
    /// arguments with the constant slots replaced by the mask byte
    /// (`maskedHeadKey`) — equals the deposit's key masked at the same slots.
    /// The consumer's builder callback hands in that key for every chain the
    /// verdict can accept (an empty span for every other chain — not indexed
    /// here); the second interner `maskedKeys_` with its own list columns
    /// (`mHeadById_` / `mTailById_` / `mNodeChain_` / `mNodeNext_`) groups the
    /// chain ids by masked key in the same ascending order, and
    /// @ref chainsWithMaskedKey answers one probe per deposit slot and one per
    /// pair of deposit slots with exactly the chains whose constants sit at
    /// those slots and match — no per-candidate decode remains; the union over
    /// the probes is disjoint (a key fixes its set of masked slots) and the
    /// consumer merges it into ascending id order.
    ///
    /// **Lifecycle (mirrors `ReverseArgsIndex`).**
    /// - Built and extended lazily by @ref ensureBuilt at the consumer
    ///   (single-threaded LB work: phase 1 absorb and phase 3 processing hold
    ///   the LB's worker claim, I-83).
    /// - Cleared by @ref clear at the canonical release seam BEFORE the arena's
    ///   blocks are reclaimed, and at every `HashMemory` teardown
    ///   (`release` / `clear`), so no stale vid survives into a reload.
    /// - Marked unbuilt by @ref markUnbuilt at both reload doors (canonical and
    ///   read-only): the derived pages are not part of the canonical image, and
    ///   the next @ref ensureBuilt rebuilds from chain 1 — and at every
    ///   owner-less sweep that erased a chain (`eraseOwnerlessEntries`).
    ///
    /// @invariant For every operator name `op`, @ref chainsWithHead yields
    ///            exactly `{ id in 1..builtCount() : head operator of chain id
    ///            == op }` in ascending id order — no omission, no duplicate,
    ///            no false positive; for every masked key `k`,
    ///            @ref chainsWithMaskedKey yields exactly the chain ids whose
    ///            builder-supplied masked key equals `k`, ascending.
    /// @invariant Derived (I-117): never deloaded, never enrolled in
    ///            `LbMemory::visitContainers`, never dumped; rebuilt lazily after
    ///            a canonical reload, captured verbatim by the raw image.
    /// @invariant `builtCount() <= originals.count()` at all times between a
    ///            build and the next teardown or sweep — the chain map never
    ///            shrinks in place except through the sweep, which marks the
    ///            index stale first; @ref ensureBuilt asserts it.
    /// @see `ReverseArgsIndex`, `checkNecessityForEquality`, `flushOwnerSection`,
    ///      `removeOwnerFromRun`, I-154, I-221.
    class OriginalsHeadIndex {
    public:
        /// @brief Bind to the owning LB's arena (the page/byte source).
        ///
        /// @details
        /// Lazy like every arena container: construction touches neither the
        /// arena nor a pool block, so an unused index consumes nothing. All
        /// columns and the interner bind the SAME arena and the index's OWN
        /// private dirty flag (a derived container never marks the LB dirty).
        ///
        /// @param arena The LB's arena. Must be non-null.
        explicit OriginalsHeadIndex(LbArena* arena)
            : arena_(arena),
              heads_(arena, &derivedDirty_),
              headById_(arena, &derivedDirty_),
              tailById_(arena, &derivedDirty_),
              nodeChain_(arena, &derivedDirty_),
              nodeNext_(arena, &derivedDirty_),
              maskedKeys_(arena, &derivedDirty_),
              mHeadById_(arena, &derivedDirty_),
              mTailById_(arena, &derivedDirty_),
              mNodeChain_(arena, &derivedDirty_),
              mNodeNext_(arena, &derivedDirty_) {
            assert(arena != nullptr && "OriginalsHeadIndex: null arena");
        }

        OriginalsHeadIndex(const OriginalsHeadIndex&) = delete;
        OriginalsHeadIndex& operator=(const OriginalsHeadIndex&) = delete;
        OriginalsHeadIndex(OriginalsHeadIndex&&) = delete;
        OriginalsHeadIndex& operator=(OriginalsHeadIndex&&) = delete;

        /// @brief Bring the index up to date with the chain map's current count.
        ///
        /// @details
        /// Three cases, in order:
        /// 1. Unbuilt (fresh, or after @ref markUnbuilt): @ref clear, then index
        ///    every chain `1..chainCount`.
        /// 2. Built with `builtCount() < chainCount`: index only the appended
        ///    chains `builtCount() + 1 .. chainCount` (append-only extension,
        ///    exact by the id-stability argument in the class description).
        /// 3. Built with `builtCount() == chainCount`: nothing to do.
        /// A `chainCount < builtCount()` is a contract violation (the map shrank
        /// without a teardown) and asserts.
        ///
        /// @param chainCount  The chain map's current `count()`.
        /// @param headNameOf  Callable `StrSpan(int32_t chainId)` returning the
        ///                    operator-name bytes of that chain's head. The span
        ///                    must stay valid until the call returns for that
        ///                    id; the bytes are copied into the interner at once.
        /// @param maskedKeyOf Callable `StrSpan(int32_t chainId)` returning the
        ///                    chain's masked head key (`maskedHeadKey`) when the
        ///                    consumer's verdict can accept the chain, else an
        ///                    empty span (the chain is left out of the masked
        ///                    family). Same lifetime rule as @p headNameOf.
        /// @post `builtCount() == chainCount` and the index is built.
        template <typename HeadNameOf, typename MaskedKeyOf>
        void ensureBuilt(int32_t chainCount, HeadNameOf&& headNameOf,
                         MaskedKeyOf&& maskedKeyOf) {
            assert(chainCount >= 0 && "OriginalsHeadIndex::ensureBuilt: negative count");
            if (unbuilt_) {
                clear();
                unbuilt_ = false;
            }
            assert(builtCount_ <= chainCount
                && "OriginalsHeadIndex::ensureBuilt: the chain map shrank in place - "
                   "originals keys are append-only; a teardown must clear the index");
            for (int32_t id = builtCount_ + 1; id <= chainCount; ++id)
                append(headNameOf(id), maskedKeyOf(id), id);
            builtCount_ = chainCount;
        }

        /// @brief Visit the chain ids whose head carries @p headName, ascending.
        ///
        /// @details
        /// One hash probe into the interner, then the operator's list from head
        /// to tail. An operator that never headed a chain yields nothing.
        ///
        /// @param headName Operator-name bytes (the `extractExpressionSpan` of
        ///                 an expression).
        /// @param sink     Callable `void(int32_t chainId)`.
        /// @pre The index is built (@ref ensureBuilt ran since the last
        ///      @ref markUnbuilt); asserted.
        template <typename Sink>
        void chainsWithHead(StrSpan headName, Sink&& sink) const {
            assert(!unbuilt_
                && "OriginalsHeadIndex::chainsWithHead: probe on an unbuilt index");
            const int32_t entryId = heads_.lookup(headName);
            if (entryId == 0) return;
            assert(entryId >= 1 && entryId <= headById_.size()
                && "OriginalsHeadIndex::chainsWithHead: entry id out of range");
            for (int32_t node = headById_[entryId - 1]; node != -1;
                 node = nodeNext_[node]) {
                sink(nodeChain_[node]);
            }
        }

        /// @brief Visit the chain ids whose masked head key equals @p key,
        ///        ascending.
        ///
        /// @details
        /// One hash probe into the masked interner, then the key's list from
        /// head to tail. A key no chain carries yields nothing. The consumer
        /// probes once per argument slot of its deposit (the deposit's key
        /// masked at that slot, `maskedHeadKey`) and merges the disjoint
        /// per-slot answers into ascending id order.
        ///
        /// @param key  Masked head key bytes.
        /// @param sink Callable `void(int32_t chainId)`.
        /// @pre The index is built; asserted.
        template <typename Sink>
        void chainsWithMaskedKey(StrSpan key, Sink&& sink) const {
            assert(!unbuilt_
                && "OriginalsHeadIndex::chainsWithMaskedKey: probe on an unbuilt index");
            const int32_t entryId = maskedKeys_.lookup(key);
            if (entryId == 0) return;
            assert(entryId >= 1 && entryId <= mHeadById_.size()
                && "OriginalsHeadIndex::chainsWithMaskedKey: entry id out of range");
            for (int32_t node = mHeadById_[entryId - 1]; node != -1;
                 node = mNodeNext_[node]) {
                sink(mNodeChain_[node]);
            }
        }

        /// @brief Free every page and forget the build.
        ///
        /// @details
        /// Called at the canonical release seam (before the arena's blocks are
        /// reclaimed) and at every `HashMemory` teardown. The next
        /// @ref ensureBuilt rebuilds from chain 1.
        void clear() {
            heads_.release();
            headById_.clear();
            tailById_.clear();
            nodeChain_.clear();
            nodeNext_.clear();
            maskedKeys_.release();
            mHeadById_.clear();
            mTailById_.clear();
            mNodeChain_.clear();
            mNodeNext_.clear();
            builtCount_ = 0;
            unbuilt_ = true;
            derivedDirty_ = DirtyState::Clean;
        }

        /// @brief Declare the index stale without touching its pages.
        ///
        /// @details
        /// The reload doors call this: after a canonical reload the derived
        /// pages do not exist (they were cleared at release), and the next
        /// @ref ensureBuilt rebuilds; a raw reload restored them verbatim, and
        /// the rebuild merely recomputes the same content.
        void markUnbuilt() { unbuilt_ = true; }

        /// @brief Number of chains the index currently covers.
        /// @return `0` when unbuilt.
        int32_t builtCount() const { return unbuilt_ ? 0 : builtCount_; }

        /// @brief Whether a probe is currently legal.
        /// @return `true` after @ref ensureBuilt, `false` after a teardown or
        ///         @ref markUnbuilt.
        bool built() const { return !unbuilt_; }

        /// @brief Number of distinct head operators indexed.
        /// @return The interner's entry count.
        int32_t operatorCount() const { return headById_.size(); }

        /// @brief Number of distinct masked head keys indexed.
        /// @return The masked interner's entry count.
        int32_t maskedKeyCount() const { return mHeadById_.size(); }

    private:
        /// @brief Append one chain under its head operator and, when the
        ///        consumer supplied one, under its masked head key (tail
        ///        inserts).
        ///
        /// @details
        /// Mints the operator name (a hit returns the existing entry), grows the
        /// per-entry head/tail columns on a new entry, and links the node behind
        /// the entry's tail so the list stays in insertion — ascending chain-id —
        /// order; the masked family repeats the same three steps on its own
        /// columns for a non-empty @p maskedKey.
        ///
        /// @param headName  Operator-name bytes of the chain's head.
        /// @param maskedKey The chain's masked head key, or an empty span for a
        ///                  chain the consumer's verdict can never accept.
        /// @param chainId   The chain's id in the map (`>= 1`).
        void append(StrSpan headName, StrSpan maskedKey, int32_t chainId) {
            assert(chainId >= 1 && "OriginalsHeadIndex::append: chain id must be >= 1");
            const int32_t entryId = heads_.mint(headName);
            assert(entryId >= 1 && "OriginalsHeadIndex::append: mint returned a null id");
            while (headById_.size() < entryId) {
                headById_.push_back(-1);
                tailById_.push_back(-1);
            }
            const int32_t newNode = nodeChain_.size();
            nodeChain_.push_back(chainId);
            nodeNext_.push_back(-1);
            const int32_t oldTail = tailById_[entryId - 1];
            if (oldTail == -1) headById_.setAt(entryId - 1, newNode);
            else nodeNext_.setAt(oldTail, newNode);
            tailById_.setAt(entryId - 1, newNode);

            if (maskedKey.len == 0) return;
            const int32_t mEntry = maskedKeys_.mint(maskedKey);
            assert(mEntry >= 1 && "OriginalsHeadIndex::append: masked mint returned a null id");
            while (mHeadById_.size() < mEntry) {
                mHeadById_.push_back(-1);
                mTailById_.push_back(-1);
            }
            const int32_t mNode = mNodeChain_.size();
            mNodeChain_.push_back(chainId);
            mNodeNext_.push_back(-1);
            const int32_t mOldTail = mTailById_[mEntry - 1];
            if (mOldTail == -1) mHeadById_.setAt(mEntry - 1, mNode);
            else mNodeNext_.setAt(mOldTail, mNode);
            mTailById_.setAt(mEntry - 1, mNode);
        }

        LbArena* arena_;
        DirtyState derivedDirty_ = DirtyState::Clean;
        ColdHashSet<BytesKeyStore> heads_;
        PagedVector<int32_t> headById_;
        PagedVector<int32_t> tailById_;
        PagedVector<int32_t> nodeChain_;
        PagedVector<int32_t> nodeNext_;
        ColdHashSet<BytesKeyStore> maskedKeys_;
        PagedVector<int32_t> mHeadById_;
        PagedVector<int32_t> mTailById_;
        PagedVector<int32_t> mNodeChain_;
        PagedVector<int32_t> mNodeNext_;
        int32_t builtCount_ = 0;
        bool unbuilt_ = true;
    };

    /// @brief The mask byte of a masked head key — never a byte of MPL text.
    constexpr char kMaskedHeadSlotByte = '\x01';

    /// @brief Bound on a masked head key's length (sign byte, operator name
    ///        plus arguments); a longer key asserts.
    constexpr int32_t kMaxMaskedHeadKeyBytes = 1024;

    /// @brief Build the masked head key of an expression: a `!` for a negated
    ///        expression, its operator name, then `,`-separated arguments, the
    ///        argument at @p maskSlot (and at @p maskSlot2 when given) replaced
    ///        by the single mask byte.
    ///
    /// @details
    /// Two expressions have equal masked keys for the same masked slots iff
    /// they have the same sign, their operator names are equal, their argument
    /// counts are equal, and every argument outside the masked slots is equal
    /// byte for byte — exactly the verdict `checkNecessityForEquality` needs
    /// between a chain head (its one or two constant slots excepted) and the
    /// `u_`-prefixed deposit. A rule whose head has the opposite sign concludes
    /// the negation of the deposit and can never help establish it, so the sign
    /// is part of the key. The separator `,` never occurs inside an argument
    /// (arguments are the comma-split spans of the text), the mask byte never
    /// occurs in MPL text and no operator name starts with `!`, so the key is
    /// injective over (sign, name, argument list, set of masked slots); in
    /// particular a one-slot key never equals a two-slot key, so both live in
    /// one masked family.
    ///
    /// @param name      Operator-name bytes (`extractExpressionSpan`, which
    ///                  drops the sign).
    /// @param negated   Whether the expression is negated (its text starts
    ///                  with `!`).
    /// @param args      The argument spans (`getArgsSpans`).
    /// @param argN      Argument count.
    /// @param maskSlot  The 0-based slot to mask; `0 <= maskSlot < argN`.
    /// @param buf       Caller buffer receiving the key bytes.
    /// @param cap       Capacity of @p buf; an overflow asserts.
    /// @param maskSlot2 A second 0-based slot to mask, or `-1` for a one-slot
    ///                  key; when given, `0 <= maskSlot2 < argN` and
    ///                  `maskSlot2 != maskSlot` (asserted).
    /// @return The key as a span over @p buf.
    /// @invariant Never allocates; the result lives in @p buf.
    /// @see OriginalsHeadIndex::chainsWithMaskedKey, checkNecessityForEquality.
    inline StrSpan maskedHeadKey(StrSpan name, bool negated, const StrSpan* args,
                                 int32_t argN, int32_t maskSlot, char* buf,
                                 int32_t cap, int32_t maskSlot2 = -1) {
        assert(argN >= 0 && maskSlot >= 0 && maskSlot < argN
            && "maskedHeadKey: mask slot outside the argument list");
        assert((maskSlot2 == -1 || (maskSlot2 >= 0 && maskSlot2 < argN && maskSlot2 != maskSlot))
            && "maskedHeadKey: second mask slot outside the argument list or equal to the first");
        int32_t at = 0;
        assert((negated ? 1 : 0) + name.len <= cap && "maskedHeadKey: key exceeds the buffer");
        if (negated) buf[at++] = '!';
        for (int32_t b = 0; b < name.len; ++b) buf[at++] = name.ptr[b];
        for (int32_t a = 0; a < argN; ++a) {
            const bool masked = (a == maskSlot || a == maskSlot2);
            const int32_t need = masked ? 2 : 1 + args[a].len;
            assert(at + need <= cap && "maskedHeadKey: key exceeds the buffer");
            buf[at++] = ',';
            if (masked) {
                buf[at++] = kMaskedHeadSlotByte;
            } else {
                for (int32_t b = 0; b < args[a].len; ++b) buf[at++] = args[a].ptr[b];
            }
        }
        return StrSpan(buf, at);
    }

} // namespace gl
