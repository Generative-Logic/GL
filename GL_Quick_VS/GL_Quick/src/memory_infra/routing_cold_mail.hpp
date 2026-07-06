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

// RoutingColdMail -- the per-LB cross-LB routing mailbox (Memory::mailIn /
// mailOut), on the never-deloaded MAIL POOL. Each instance owns a per-LB cold
// LbArena drawn from the mail pool; its blocks are freed back to the pool the
// moment its content is read out (mailOut at the commit barrier, mailIn after
// the phase-1 absorb), so nothing is retained. The richer rationale is on the
// struct below.

#include "dirty_state.hpp"
#include "lb_arena.hpp"
#include "global_memory_manager.hpp"
#include "mail_types.hpp"
#include "typed_cold_map.hpp"

#include <algorithm>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace gl {

    struct Mail;  // heap mailbox, defined in memory.hpp; `toHeapMail` is declared
                  // here and defined out-of-line there once Mail is complete.

    /// @brief The per-LB cross-LB routing mailbox (`Memory::mailIn` /
    ///        `Memory::mailOut`) on the never-deloaded MAIL POOL, freed after
    ///        each read-out.
    ///
    /// @details
    /// `mailIn`/`mailOut` carry the cross-LB routing traffic — the per-LB
    /// endpoints of the pull-model mail system. The commit barrier reads every
    /// LB's `mailOut` (incl. inactive / evicted ones) to decide what to append to
    /// each `MailLog`; that all-bodies read must never hit a deloaded arena, so
    /// the storage belongs on the **never-deloaded MAIL POOL** (where the
    /// `MailLog` it feeds already lives), NOT the per-LB deloadable arena — the
    /// `MailLog` precedent, and still "0 hot" (the mail pool is a sanctioned cold
    /// reservation, not a hot arena). Each instance owns a per-LB cold `LbArena`
    /// drawn from the mail pool (`mailMemory()`), bound LAZILY (`ensureArena` on
    /// first write — the root LB predates the pool init), with a write-only
    /// `dirty_` sink (it never deloads). `clear()` returns the arena's blocks to
    /// the pool — called the moment the content is read out (`mailOut` at the
    /// commit barrier, `mailIn` after the phase-1 absorb), so nothing is retained;
    /// the next write re-acquires lazily.
    ///
    /// Two live columns: `statements_` keyed on the whole pair (id-form
    /// `IntMailStatementKey`); `origins_` as a per-EWV blob run of `OriginLine`.
    /// The heap-`Mail` fields `disintegrationSignals` and `expandedImplications`
    /// are DROPPED (the former is internal-channel-only; the latter was never
    /// serialized / delivered cross-LB — `Codec<Mail>` only carries statements +
    /// exprOriginMap — so it has been removed). The column types and codecs are
    /// reused from `mail_types.hpp`. Never deload-registered — no `ContainerTag`,
    /// absent from `LbMemory::visitContainers`.
    ///
    /// The cold maps mint in insertion order, so every observable boundary (the
    /// `memory.hpp` decode helpers) re-imposes the canonical
    /// `ExpressionWithValidity::operator<` order the heap containers had.
    ///
    /// @invariant Single-threaded access: a given LB is owned by one worker at a
    ///            time and phase-2 is read-only on the LB, so the per-LB arena is
    ///            never touched concurrently (no lock).
    /// @see ColdMail (the internal-channel sibling), MailLog, Mail, addMailOrigin.
    struct RoutingColdMail {
        /// @brief Dedicated per-LB cold arena on the mail pool (bound lazily).
        LbArena arena_;
        /// @brief Whether `arena_` has been bound (deferred-bind guard).
        bool arenaInited_ = false;
        /// @brief Write-only dirty sink — the mailbox never deloads, so its
        ///        mutations must not escalate any LB's deload-dirty flag.
        DirtyState dirty_ = DirtyState::Clean;
        /// @brief `set<pair<EWV, set<int>>>` as a set of id-form byte keys.
        ///        mailOut holds SENDER NameMap ids; mailIn holds GLOBAL mailInterner
        ///        ids (the commit seam translates sender -> global). The id-space is
        ///        chosen by the owning mailbox's role; decoded at the memory.hpp
        ///        boundary with the matching interner.
        TypedColdSet<IntMailStatementKey> statements_;
        /// @brief `map<EWV, vector<OriginLine>>` as an id-keyed blob run (key =
        ///        packed interner pair; value = `IntMailOrigin`). mailOut keys on
        ///        the sender `originInterner`; mailIn on the global `mailInterner`
        ///        (the commit seam translates sender -> global).
        TypedColdBlobMap<int64_t, IntMailOrigin> origins_;

        /// @brief Bind the three columns to the (unbound) dedicated arena.
        ///
        /// @details `arena_` is declared first so it destructs LAST — the
        /// columns destruct before it (reverse declaration order), so their page
        /// frees touch a live arena; the arena's destructor then returns the
        /// blocks to the mail pool.
        RoutingColdMail()
            : arena_(), statements_(&arena_, &dirty_),
              origins_(&arena_, &dirty_) {}
        RoutingColdMail(const RoutingColdMail&) = delete;
        RoutingColdMail& operator=(const RoutingColdMail&) = delete;

        /// @brief Bind the arena to the mail pool on first use (the pool is ready
        ///        by then — writes start only after `initMailMemory`).
        void ensureArena() {
            if (!arenaInited_) {
                arena_.bind(&mailMemory());
                arenaInited_ = true;
            }
        }

        /// @brief Insert one statement (id-form pair + levels) — set semantics.
        ///
        /// @details The caller supplies the interner ids in the owning mailbox's
        /// id-space (SENDER NameMap ids for mailOut via `fillMailOut`; GLOBAL
        /// mailInterner ids for mailIn via the pull / `deserializeInto`).
        ///
        /// @param originalId The expression's interner id.
        /// @param validityId The scope's interner id.
        /// @param levels     The statement's level set (ascending in the byte key).
        void insertStatement(int32_t originalId, int32_t validityId,
            const std::set<int>& levels) {
            ensureArena();
            IntMailStatementKey k;
            k.originalId = originalId;
            k.validityId = validityId;
            k.levels.assign(levels.begin(), levels.end());
            statements_.mint(k);
        }

        /// @brief Insert one statement from an ASCENDING-UNIQUE level RUN —
        ///        the contiguous-run twin of the `std::set<int>` overload
        ///        (mirror of `ColdMail::insertStatement`'s run door).
        ///
        /// @details
        /// Byte-identical deposit to the set overload when the precondition
        /// holds:
        /// 1. `[levels, levels + levelCount)` ascending strictly-unique is
        ///    exactly what `std::set<int>` iteration yields, so the key bytes
        ///    `k.levels` are IDENTICAL to the set overload's
        ///    `assign(levels.begin(), levels.end())`.
        /// 2. Identical key bytes mean identical `Codec<IntMailStatementKey>`
        ///    encoding, hence identical mint — dedup verdict, id, and blob
        ///    bytes at the commit seam.
        /// 3. `levelCount == 0` equals the empty-set case byte-for-byte
        ///    (empty `k.levels`).
        /// 4. `ensureArena()` runs before the first insert exactly like the
        ///    set overload — a run-door deposit may be the mailbox's very
        ///    first write, and an unbound arena is an access violation, not
        ///    an assert.
        /// Lets callers reading a level run straight off a sorted-unique CSR
        /// store (`intStatementLevelsMap` values) or a caller stack run
        /// deposit without building a transient `std::set<int>`.
        ///
        /// @pre `[levels, levels + levelCount)` is sorted ascending with no
        ///      duplicates (asserted for shape; the ORDER obligation is the
        ///      caller's source contract — a sorted-unique store run).
        /// @param originalId The expression's interner id (sender NameMap for
        ///                   mailOut; global mailInterner for mailIn).
        /// @param validityId The scope's interner id (same space as
        ///                   `originalId`).
        /// @param levels     Pointer to `levelCount` ascending-unique levels;
        ///                   may be null only when `levelCount` is 0.
        /// @param levelCount The level count (`>= 0`).
        /// @invariant [I-136](../../../../docs/agentic_swdd/30_invariants.md#i-136)
        ///            — the run is caller-owned stack/arena memory.
        /// @see insertStatement(int32_t, int32_t, const std::set<int>&) — the
        ///      byte-oracle overload.
        void insertStatement(int32_t originalId, int32_t validityId,
            const int* levels, int32_t levelCount) {
            assert(levelCount >= 0 && (levelCount == 0 || levels != nullptr));
            ensureArena();
            IntMailStatementKey k;
            k.originalId = originalId;
            k.validityId = validityId;
            k.levels.assign(levels, levels + levelCount);
            statements_.mint(k);
        }

        /// @brief Whether the statements column is empty.
        /// @return `true` if no statement has been inserted.
        bool statementsEmpty() const {
            return !arenaInited_ || statements_.count() == 0;
        }
        /// @brief Whether the origin column is empty.
        /// @return `true` if no origin line has been added.
        bool exprOriginMapEmpty() const {
            return !arenaInited_ || origins_.count() == 0;
        }
        /// @brief Whether both columns are empty.
        /// @return `true` if the mailbox carries nothing.
        bool empty() const {
            return statementsEmpty() && exprOriginMapEmpty();
        }

        // The string-decoding statements snapshot moved to memory.hpp
        // (decodeMailInStatements + routingMailOutToHeap / routingMailInToHeap),
        // which see the matching interner (sender NameMap for mailOut, global
        // mailInterner for mailIn) to turn the stored ids back into strings.
        // statements_ is exposed directly as the id-form column.

        // The string-decoding origins snapshot moved to memory.hpp
        // (decodeMailInOrigins / decodeMailOutOrigins + the routingMail*ToHeap
        // helpers), which see the matching interner (sender originInterner for
        // mailOut, global mailInterner for mailIn) to turn the stored ids back
        // into strings. origins_ is exposed directly as the id-form column.

        // toHeapMail() moved to memory.hpp as the role-specific free functions
        // routingMailOutToHeap(mo, NameMap&) / routingMailInToHeap(mi): the
        // statements column is now id-form, so materializing the heap Mail needs
        // the matching interner (sender NameMap for mailOut, global mailInterner
        // for mailIn) which the mail headers cannot see.

        /// @brief Empty the mailbox and RETURN its arena blocks to the mail pool
        ///        — the read-out free (mailOut at the commit barrier, mailIn after
        ///        the phase-1 absorb).
        ///
        /// @details Releases both columns' pages, then `releaseAll`s the
        /// arena (blocks back to the pool — no retention). The arena stays bound
        /// and resident, so the next write re-acquires lazily.
        void clear() {
            if (!arenaInited_) return;
            statements_.release();
            origins_.release();
            arena_.releaseAll();
        }

        /// @brief Erase the `statements` entries the predicate accepts — the
        ///        scope-targeted `wipeSubtree` sweep.
        ///
        /// @details Only `statements` is swept. A guaranteed no-op in practice
        /// (every routing item is "main", never closed — I-26), kept explicit
        /// (Rule 19).
        ///
        /// @tparam Pred A callable `bool(const IntMailStatementKey&)`.
        /// @param pred Accepts the keys to erase (typically a closed-scope id test).
        /// @return Number of statement keys removed.
        template <typename Pred>
        int32_t filterStatements(Pred pred) {
            if (!arenaInited_) return 0;
            return statements_.eraseIf(
                [&pred](const IntMailStatementKey& k) { return pred(k); });
        }
    };

    // addMailOrigin(RoutingColdMail&, ...) moved to memory.hpp as
    // addRoutingMailOrigin(RoutingColdMail&, ValueInterner&, EWV, OriginLine, int):
    // its D-49 cap-full preference compares OriginTag values, and OriginTag lives
    // in memory.hpp (invisible here). The helper encodes (ev, origin) into the
    // sender originInterner + packs the key, then does the RMW on hm.origins_ via
    // the shared addMailOriginRecord.

} // namespace gl
