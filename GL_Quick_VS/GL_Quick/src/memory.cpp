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

// Hash-engine rule installation and request operations.

#include "memory.hpp"
#include "memory_infra/paged_sorted_ordinals.hpp"
#include "logical_block/statement_registry_operations.hpp"
#include "logical_block/scope_syntax.hpp"

#include "prover.hpp"
#include "infra/hashburst_dump.hpp"
#include "parameters.hpp"
#include "memory_infra/lb_deload.hpp"
#include "memory_infra/deload_stats.hpp"
#include <cstdio>
#include <iostream>
#include "memory_infra/str_ops.hpp"
#include "memory_infra/arena_stack.hpp"
#include "memory_infra/scratch_arena.hpp"
#include <fstream>  // [ORDIS-TRAP] diagnostic trap output (sandbox/equi_reshuffle_rung1)
#include "infra/rt_tracker.hpp"

#include <algorithm>
#include <cstring>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

namespace gl {

    // Out-of-line IntEncodedExpr recordUSignature (declared in memory.hpp).
    // Heap-free: builds the (slot, argFullId) pairs on the per-slot gen-scratch
    // byte-bump tier, then range-constructs the signature vector INSIDE the set
    // node via emplace(first, last) — no function-local std::vector. Out-of-line
    // because it reaches genScratchArenas() + gl::WorkerContext::coreId(),
    // which an inline header body cannot see (the encodePush(StrSpan) precedent).
    void recordUSignature(OwnerSet& os,
                          const IntEncodedExpr* encList, NameId count) {
        using Pair = std::pair<int32_t, NameId>;
        // Single-threaded install (mergeOwnerRecord, I-83): workers publish
        // gl::WorkerContext::coreId(); -1 (setup contexts) maps to the reserved last slot.
        const unsigned uSlot = (gl::WorkerContext::coreId() >= 0)
            ? static_cast<unsigned>(gl::WorkerContext::coreId())
            : genScratchArenas().slotCount() - 1;
        ScratchArena& a = genScratchArenas().forSlot(uSlot);
        const ArenaOffset mark = a.cursor();

        // Worst-case run length = total arg slots = Sum arity <= count*MAX_ARITY;
        // count is not tightly bounded, so an arena run (cap-free) not a stack
        // array. The assert below is the Rule-19 tripwire on the derivation.
        const int32_t cap = static_cast<int32_t>(count)
            * static_cast<int32_t>(ExecutionParameters::MAX_ARITY);
        Pair* sig = (cap > 0)
            ? reinterpret_cast<Pair*>(a.resolve(a.alloc(
                  cap * static_cast<int32_t>(sizeof(Pair)),
                  static_cast<int32_t>(alignof(Pair)))))
            : nullptr;

        int32_t n = 0;
        int32_t slot = 0;
        bool hasUArg = false;
        for (NameId e = 0; e < count; ++e) {
            const IntEncodedExpr& expr = encList[e];
            const int32_t arity = std::min<int32_t>(expr.arity,
                static_cast<int32_t>(ExecutionParameters::MAX_ARITY));
            for (int32_t j = 0; j < arity; ++j) {
                if (expr.argUnchangeable[j] != 0) {
                    hasUArg = true;
                    assert(n < cap);
                    sig[n++] = Pair(slot, expr.argFullId[j]);
                }
                ++slot;
            }
        }
        if (!hasUArg) os.hasLooseOwner = true;
        else os.uSignatures.emplace(sig, sig + n);
        a.popTo(mark);
    }











    /// @brief Whether an application equivalence class at a scope or at one of
    ///        its strict ancestors holds a name.
    ///
    /// @details
    /// Walks the scope's NameMap parent chain from the scope itself to the root
    /// and tests every class stored in `equivalenceClassesMap` at each scope for
    /// the member. These are exactly the classes that `updateEquivalenceClasses`
    /// merges for an equality at the scope (the same-scope pass plus the
    /// strict-ancestor pass), so a `true` answer means an equality naming the
    /// member joins an existing class and a `false` answer means it would found
    /// a new one. A scope that was never interned has no classes and no
    /// interned ancestors, so the answer is `false`.
    ///
    /// Read-only and non-minting: class blobs are peeked onto the slot's
    /// gen-scratch arena under a per-class `ScratchScope`, so the walk allocates
    /// no heap memory.
    ///
    /// @param validity Scope of the equality; need not be interned.
    /// @param member   NameMap id of the name; must be interned (non-zero).
    /// @return `true` iff some application class at the scope or a strict
    ///         ancestor contains `member`.
    /// @invariant Changes no class, name or arena state visible after the call.
    /// @see EquivalenceProcessor::updateEquivalenceClasses — the caller that
    ///      keeps a reflexive equality from founding a one-member class.
    bool Memory::applicationClassHoldsName(StrSpan validity, NameId member) {
        assert(member != 0 && "applicationClassHoldsName: the name must be interned");
        const unsigned slot = gl::WorkerContext::coreId() >= 0
            ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
        ScratchArena& arena = genScratchArenas().forSlot(slot);
        for (NameId scope = nameMap.lookup(validity); scope != 0; scope = nameMap.parentOf(scope)) {
            const int32_t bucket = equivalenceClassesMap.lookup(scope);
            if (bucket == 0) continue;
            const int32_t count = equivalenceClassesMap.runLen(bucket);
            for (int32_t i = 0; i < count; ++i) {
                ScratchScope frame(arena);
                int32_t length = 0;
                const char* bytes = equivalenceClassesMap.peekRecordBytes(bucket, i, length, arena);
                if (classHasMember(EquivalenceClassView{bytes, length}, member)) return true;
            }
        }
        return false;
    }

    // ------------------------------------------------------------------
    // The single-root interface uses the same stable union cleanup as the
    // end-of-phase batch. NameMap parentage defines subtree membership;
    // expandedImplications retains its separate interned-text scope test.
    // Proof history, cohort bookkeeping and the owning parent goals survive.
    // ------------------------------------------------------------------
    /// @brief Record that the induction hypothesis fired at an induction
    ///        scope this burst.
    ///
    /// @details
    /// The installed induction implication's head was deposited at @p leaf
    /// (`StatementMemory::addExprToMemoryBlock` with `recursionHead`). The
    /// firing waits for this burst's `settleInductionTrees`, which settles the
    /// leaf's tree on its first firing (D-520).
    ///
    /// @param leaf The induction scope the hypothesis fired at.
    /// @see settleInductionTrees, LbMemory::inductionFirings.
    void Memory::recordInductionFiring(NameId leaf) {
        assert(leaf > 0 && ScopeSyntax::isInductionScope(nameMap, leaf)
            && "recordInductionFiring: the hypothesis fired at an induction scope");
        lbMemory.inductionFirings.mint(leaf);
    }

    /// @brief Whether the induction tree a scope belongs to has settled.
    ///
    /// @details
    /// A scope's tree is rooted at its nearest scope that is not an induction
    /// scope (`ScopeSyntax::inductionRootOf`). Once the hypothesis fired in one
    /// leaf of the tree, the rest of the tree was wiped and the hypothesis
    /// opens no further induction scope in it (D-520).
    ///
    /// @param scope A minted validity id.
    /// @return True iff the tree containing @p scope has settled.
    /// @see settleInductionTrees, LbMemory::inductionSettledRoots.
    bool Memory::inductionTreeSettled(NameId scope) const {
        assert(scope > 0 && "inductionTreeSettled: scope must be minted");
        return lbMemory.inductionSettledRoots.lookup(
            ScopeSyntax::inductionRootOf(nameMap, scope)) != 0;
    }

    /// @brief Settle the induction trees whose hypothesis fired this burst.
    ///
    /// @details
    /// Runs once per burst in the phase-3 scope cleanup, right before the
    /// wipe drain (D-520). The burst's
    /// firing leaves are taken in decoded-name order. A leaf of a tree that
    /// has not settled, and whose path to `main` is not being wiped this
    /// burst, settles the tree: its root joins
    /// `LbMemory::inductionSettledRoots`, and every induction scope whose
    /// parent lies on the path from the leaf to the root - the siblings along
    /// the path and the leaf's own children - joins `pendingWipeScopes`. The
    /// leaf and its ancestors stay, and work continues in the leaf. A later
    /// leaf of a settled tree changes nothing. The firing set is reset.
    ///
    /// @invariant Deterministic: leaves in decoded-name order, the scope scan
    ///            in NameMap id order; the wipe drain sorts by name.
    /// @see recordInductionFiring, inductionTreeSettled, wipeSubtrees.
    void Memory::settleInductionTrees() {
        const int32_t firingCount = lbMemory.inductionFirings.count();
        if (firingCount == 0) return;
        const unsigned slot = (gl::WorkerContext::coreId() >= 0)
            ? static_cast<unsigned>(gl::WorkerContext::coreId())
            : genScratchArenas().slotCount() - 1;
        ScratchArena& gArena = genScratchArenas().forSlot(slot);
        ScratchScope scratch(gArena);
        NameId* leaves = reinterpret_cast<NameId*>(gArena.resolve(
            gArena.alloc(firingCount * static_cast<int32_t>(sizeof(NameId)),
                         static_cast<int32_t>(alignof(NameId)))));
        for (int32_t i = 1; i <= firingCount; ++i)
            leaves[i - 1] = lbMemory.inductionFirings.decode(i);
        lbMemory.inductionFirings.resetToFresh();
        std::sort(leaves, leaves + firingCount, [&](NameId a, NameId b) {
            return compareSpans(nameMap.decodeView(a), nameMap.decodeView(b)) < 0;
        });
        const int32_t nameHighWater = nameMap.nameCount();
        for (int32_t leafIndex = 0; leafIndex < firingCount; ++leafIndex) {
            const NameId leaf = leaves[leafIndex];
            const NameId root = ScopeSyntax::inductionRootOf(nameMap, leaf);
            if (lbMemory.inductionSettledRoots.lookup(root) != 0) continue;
            NameId path[ExecutionParameters::MAX_SCOPE_DEPTH];
            int32_t pathCount = 0;
            for (NameId scope = leaf; scope != root; scope = nameMap.parentOf(scope)) {
                assert(pathCount + 1 < ExecutionParameters::MAX_SCOPE_DEPTH
                    && "settleInductionTrees: induction path exceeds the scope depth");
                path[pathCount++] = scope;
            }
            path[pathCount++] = root;
            bool pathWiped = false;
            for (NameId scope = leaf; scope != 0 && !pathWiped; scope = nameMap.parentOf(scope))
                pathWiped = pendingWipeScopes.lookup(scope) != 0;
            if (pathWiped) continue;
            lbMemory.inductionSettledRoots.mint(root);
            // The release drain already ran. A retained induction path may
            // now release its queued OR cases on the next burst.
            mutatedThisBurst = true;
            for (NameId id = 1; id <= nameHighWater; ++id) {
                const NameId parent = nameMap.parentOf(id);
                if (parent == 0 || !ScopeSyntax::isInductionScope(nameMap, id)) continue;
                bool parentOnPath = false;
                bool selfOnPath = false;
                for (int32_t k = 0; k < pathCount; ++k) {
                    parentOnPath = parentOnPath || path[k] == parent;
                    selfOnPath = selfOnPath || path[k] == id;
                }
                if (parentOnPath && !selfOnPath) pendingWipeScopes.mint(id);
            }
        }
    }

    void Memory::wipeSubtree(NameId closedVid) {
        wipeSubtrees(&closedVid, 1);
    }

    /// @brief Wipe an ordered union of closed scope subtrees in one pass.
    ///
    /// @details
    /// Build one membership bitmap through the NameMap parent forest, then
    /// stably filter each affected runtime container once. Assign each closed
    /// name to its first covering input root so final filter insertions match
    /// sequential single-root wipes, including overlapping and repeated roots.
    /// Proof history and booked convergence contributions remain unchanged.
    ///
    /// @param closedRoots Caller-owned ordered run of minted non-main roots.
    /// @param rootCount Number of roots; zero is an empty cleanup operation.
    /// @return Nothing.
    /// @invariant Survivor bytes and filter mint order equal sequential wipes.
    /// @invariant Saved goal levels leave only when their cohort parent closes.
    /// @see Memory::wipeSubtree, NameMap::parentOf, wipeEncodedMapForClosed.
    void Memory::wipeSubtrees(const NameId* closedRoots, int32_t rootCount) {
        assert(rootCount >= 0 && (rootCount == 0 || closedRoots != nullptr)
            && "wipeSubtrees: invalid root run");
        if (rootCount == 0) return;

        const unsigned slot = (gl::WorkerContext::coreId() >= 0)
            ? static_cast<unsigned>(gl::WorkerContext::coreId())
            : genScratchArenas().slotCount() - 1;
        ScratchArena& gArena = genScratchArenas().forSlot(slot);
        ScratchScope scratch(gArena);
        DirtyState scratchDirty = DirtyState::Clean;
        ColdHashMap<PodKeyStore<NameId>, int32_t> rootOrder(&gArena, &scratchDirty);
        struct ClosedScope { NameId id; int32_t firstRoot; };
        PagedVector<ClosedScope> closedScopes(&gArena, &scratchDirty);
        const int32_t nameHighWater = nameMap.nameCount();
        for (int32_t rootIndex = 0; rootIndex < rootCount; ++rootIndex) {
            const NameId root = closedRoots[rootIndex];
            assert(root > 0 && root <= nameHighWater
                && "wipeSubtrees: root is not a minted NameMap id");
            assert(root != NameMap::MAIN_ID
                && "wipeSubtrees: 'main' is never closed");
            if (rootOrder.lookup(root) == 0) rootOrder.insert(root, rootIndex);
        }
        const int32_t closedWords = nameHighWater / 64 + 1;
        uint64_t* closedBits = reinterpret_cast<uint64_t*>(gArena.resolve(
            gArena.alloc(closedWords * static_cast<int32_t>(sizeof(uint64_t)),
                         static_cast<int32_t>(alignof(uint64_t)))));
        for (int32_t word = 0; word < closedWords; ++word) closedBits[word] = 0;
        for (NameId id = 1; id <= nameHighWater; ++id) {
            int32_t firstRoot = rootCount;
            for (NameId ancestor = id; ancestor != 0; ancestor = nameMap.parentOf(ancestor)) {
                const int32_t* rank = rootOrder.find(ancestor);
                if (rank != nullptr && *rank < firstRoot) firstRoot = *rank;
            }
            if (firstRoot == rootCount) continue;
            closedBits[id >> 6] |= uint64_t{1} << (id & 63);
            closedScopes.push_back(ClosedScope{id, firstRoot});
        }
        const auto inClosedBit = [&](int32_t v) -> bool {
            // EXACT twin of the former closedIds.count(x): x can arrive
            // NEGATIVE (e.g. the low half of a GLOBAL mailIn id at the
            // routing-mail filter) — count(negative) was false; the
            // `v >= 1` guard short-circuits before the bitmap index, so
            // the index is always a non-negative NameId (no uint16 wrap —
            // a wrap would alias ids >= 65536 into the wrong bit once the
            // ceiling rises).
            return v >= 1 && v <= nameHighWater
                && ((closedBits[v >> 6] >> (v & 63)) & 1ull) != 0;
        };

        // ---- 1. statement registry ----
        // Keyed on the bitmap — a stored row's validity is interned, so
        // bit membership of its validityId is exactly the forest predicate.
        // Preserve survivor order with one forward scan. Each retained row
        // moves at most once; truncation releases the dead tail's pages after
        // every source row has been read.
        PagedVector<int32_t> erasedStatementPositions(&gArena, &scratchDirty);
        const auto compactStatements = [&](SharedExpressionList& rows, bool recordPositions) {
            const int32_t count = rows.size();
            int32_t kept = 0;
            for (int32_t read = 0; read < count; ++read) {
                const IntEncodedExpr& row = rows[read];
                if (!inClosedBit(row.validityId)) {
                    if (kept != read) rows.setAt(kept, row);
                    ++kept;
                } else if (recordPositions) {
                    erasedStatementPositions.push_back(read);
                }
            }
            rows.truncate(kept);
        };
        compactStatements(intEncodedStatements, true);
        // Surviving classes retain their processed prefix after registry compaction.
        // Only old positions before a class boundary shorten that prefix.
        if (erasedStatementPositions.size() > 0) {
            StatementRegistryOperations::repairEqClassWaterlines(*this,
                erasedStatementPositions, erasedStatementPositions.size());
        }

        // ---- 2. local statement registry + Set + Delta ----
        // The set mirrors the local-statement vector (I-86); drop its closed
        // keys in ONE eraseIf (O(n)) rather than a cold-index rebuild per row.
        intLocalEncodedStatementsSet.eraseIf([&](int64_t k) {
            return inClosedBit(Codec<StatementKey>::decode(k).validity);
        });
        compactStatements(intLocalEncodedStatements, false);
        compactStatements(intLocalEncodedStatementsDelta, false);

        // ---- 3. toBeProved ----
        // A stored goal's validity is interned, so bitmap membership of
        // the validity half of the packed key is exactly the forest predicate (same
        // argument as the intStatementLevelsMap sweep below).
        intToBeProved.eraseSetIf([&](const StatementKey& k) {
            return inClosedBit(k.validity);
        });
        // The watch ledgers mirror the registries wiped here: their keys
        // carry the scope id in the high half.
        negationLedger.eraseSetIf([&](int64_t k) {
            return inClosedBit(static_cast<NameId>(static_cast<uint64_t>(k) >> 32));
        });
        goalLedgerStale = true;   // derived from the goals wiped above

        // (Section 4 retired: the former wholeExpressions sweep — the
        // registered membership now lives in intKnownStatements, erased
        // in section 6 by the same bitmap predicate.)

        // ---- 5. intStatementLevelsMap (packed (origId, validityId)) ----
        // Safe to wipe because wipeSubtree now runs at end-of-burst
        // (drained by `performElemPhase3` immediately
        // after `sanitizeToBeProved`); the kernel's `sortedNew` loop
        // assertions all finished earlier in the burst. The validity half of the packed key
        // carry the validityId, and a stored row's validity is interned, so
        // bitmap membership is exactly the forest predicate
        // (same argument as the intKnownStatements sweep below).
        intStatementLevelsMap.eraseSetIf([&](const StatementKey& k) {
            return inClosedBit(k.validity);
        });
        // A successful branch's saved goal levels survive its own wipe, but
        // leave when the cohort's parent scope is also closed.
        lbMemory.closedOrGoalLevels.eraseSetIf([&](const StatementKey& k) {
            return inClosedBit(nameMap.parentOf(k.validity));
        });
        lbMemory.orSupportFacts.eraseIf([&](const StatementKey& k) {
            return inClosedBit(nameMap.parentOf(k.validity));
        });
        lbMemory.orSupportEqualities.eraseSetIf([&](const StatementKey& k) {
            return inClosedBit(nameMap.parentOf(k.validity));
        });

        // ---- 6. intKnownStatements (packed (origId, validityId) — low 32 bits) ----
        // Retire canonical views of closed scopes, including projected ancestor
        // sources, while retaining the unchanged ancestor and sibling views.
        orImplicationMatches.clear();
        lbMemory.existenceLookups.clear();
        lbMemory.canonicalUses.eraseScopes(inClosedBit);
        auto& compositions = lbMemory.unaryComposition;
        for (int32_t i = 0; i < compositions.sourceInfo.size(); ++i) {
            if (compositions.sourceActive[i] && inClosedBit(compositions.sourceInfo[i].scope)) {
                compositions.sourceActive.setAt(i, 0);
                if (compositions.rebuild.empty()) compositions.rebuild.push_back(1);
            }
        }

        for (int32_t id = 1; id <= orImplications.sources.count(); ++id) {
            const int64_t key = orImplications.sources.keyAt(id);
            if (inClosedBit(Codec<StatementKey>::decode(key).validity))
                orImplications.retire(key);
        }
        orImplications.compactRetired(gArena);
        orImplications.pendingSources.eraseIf([&](int64_t key) {
            return inClosedBit(Codec<StatementKey>::decode(key).validity);
        }, [](int64_t) {});
        orImplications.pendingFacts.eraseIf([&](int64_t key) {
            return inClosedBit(Codec<StatementKey>::decode(key).validity);
        }, [](int64_t) {});
        orImplications.pendingClasses.eraseIf([&](int64_t key) {
            return inClosedBit(Codec<StatementKey>::decode(key).validity);
        }, [](int64_t) {});
        intKnownStatements.eraseIf([&](const StatementKey& k) {
            return inClosedBit(k.validity);
        });
        existenceReplays.eraseIf([&](int64_t key) {
            return inClosedBit(Codec<StatementKey>::decode(key).validity);
        });
        // An induction eat leaves with its scope
        // (D-519).
        lbMemory.inductionEats.eraseIf([&](int64_t key) {
            return inClosedBit(static_cast<NameId>(key));
        });
        // A pending firing and a settled tree root leave with their scope
        // (D-520).
        lbMemory.inductionFirings.eraseIf([&](NameId id) { return inClosedBit(id); });
        lbMemory.inductionSettledRoots.eraseIf([&](NameId id) { return inClosedBit(id); });
        // A hypothesis shot leaves with its scope
        // (D-572).
        lbMemory.hypothesisShots.eraseIf([&](NameId id) { return inClosedBit(id); });
        // An int scope's record leaves with its scope; its group's levels and
        // promoted mark stay. A first finish leaves with its scope while its
        // group is not promoted: the premise has no finished scope left, and
        // the joint promotion needs an open one for every premise; a promoted
        // group's first finishes stay (D-578).
        lbMemory.intScopes.eraseIf([&](int64_t key) {
            return inClosedBit(static_cast<NameId>(key));
        });
        PagedVector<int64_t> staleFirstFinishes(&gArena, &scratchDirty);
        for (int32_t id = 1; id <= lbMemory.intScopeFirstFinish.count(); ++id) {
            if (!inClosedBit(static_cast<NameId>(lbMemory.intScopeFirstFinish.valueAt(id)))) continue;
            const int64_t finishKey = lbMemory.intScopeFirstFinish.decodeKey(id);
            const int64_t* groupLevels = lbMemory.intScopeGroups.find(
                static_cast<int64_t>(static_cast<int32_t>(static_cast<uint64_t>(finishKey) >> 32)));
            assert(groupLevels != nullptr && "wipeSubtrees: a first finish belongs to a recorded group");
            if ((*groupLevels & kIntScopeGroupPromoted) == 0) staleFirstFinishes.push_back(finishKey);
        }
        for (int32_t i = 0; i < staleFirstFinishes.size(); ++i)
            lbMemory.intScopeFirstFinish.erase(staleFirstFinishes[i]);
        // A forced product leaves with its scope
        // (D-522).
        lbMemory.forcedProducts.eraseIf([&](int64_t key) {
            return inClosedBit(Codec<StatementKey>::decode(key).validity);
        });
        // A contradiction scope leaves the demand's registry with its scope;
        // its parked candidates leave with the park wipe
        // (D-560).
        lbMemory.contradictionScopes.eraseIf([&](int64_t key) {
            return inClosedBit(static_cast<NameId>(key));
        });

        // ---- 7. equivalenceClassesMap (keyed by validity id) ----
        // Safe to wipe at end-of-burst (see step 5 comment). Run-aware
        // compacting erase on the cold blob map (Batch 3).
        equivalenceClassesMap.eraseBlobIf([&](NameId vId) {
            return inClosedBit(vId);
        });
        canonicalEquivalenceClassesMap.eraseBlobIf([&](NameId vId) {
            return inClosedBit(vId);
        });
        lbMemory.canonicalMemberRepresentatives.eraseIf([&](int64_t key) {
            return inClosedBit(Codec<StatementKey>::decode(key).validity);
        });
        lbMemory.canonicalClassChanges.eraseIf([&](int64_t key) {
            return inClosedBit(Codec<StatementKey>::decode(key).validity);
        });

        // ---- 8. Packed-keyed sets ----
        // expandedImplications: k.low is an lbStateInterner id — a DIFFERENT
        // id space than the NameMap bitmap — so this one sweep keeps a
        // string-shaped predicate, as a zero-copy span twin of the retired
        // `v == closedScope || prefix-with-delimiter` gate (strict `>`:
        // a name equal to closedScope + "_boundary_" with empty tail cannot
        // exist, but the twin keeps the exact gate). Lifetime (I-3): eraseIf
        // mutates only this TypedColdSet — nothing mints into nameMap or
        // lbStateInterner during the sweep, so both decodeView spans stay
        // valid across it (different-container rule).
        {
            const StrSpan boundaryTok(NameMap::BOUNDARY_STR,
                                      static_cast<int32_t>(NameMap::BOUNDARY_LEN));
            expandedImplications.eraseIf([&](const LbStatePairKey& k) {
                const StrSpan v = lbStateInterner.decodeView(k.low);
                for (int32_t rootIndex = 0; rootIndex < rootCount; ++rootIndex) {
                    const StrSpan closedName = nameMap.decodeView(closedRoots[rootIndex]);
                    if (equalSpans(v, closedName)) return true;
                    if (v.len > closedName.len + boundaryTok.len
                        && equalSpans(StrSpan(v.ptr, closedName.len), closedName)
                        && equalSpans(StrSpan(v.ptr + closedName.len, boundaryTok.len), boundaryTok))
                        return true;
                }
                return false;
            });
        }
        // The carrier index: a carrier registered at a closed scope leaves
        // with its scope (its rules leave through wipeHashMem below; the
        // owner maps keep their owners — I-49).
        compactExpansions.eraseBlobIf([&](int64_t k) {
            return inClosedBit(Codec<StatementKey>::decode(k).validity);
        });
        // Packed template-space twins — same low-16-bits validity
        // predicate as the other packed sweeps.
        auto filterPackedSet = [&](ColdHashSet<PodKeyStore<int64_t>>& s) {
            s.eraseIf([&](int64_t k) {
                return inClosedBit(Codec<StatementKey>::decode(k).validity);
            });
        };
        filterPackedSet(integrationPrepared);
        filterPackedSet(integrationPreparedMarker);

        // Integration trials (D-481):
        // every record leaves with its parent scope (the key's validity
        // half); an open record also leaves with its own trial scope. A
        // passed record outlives its wiped trial scope as the parent's
        // receipt for the promotion replay.
        if (lbMemory.integrationTrials.count() != 0) {
            ColdHashSet<PodKeyStore<int64_t>> closedTrials(&gArena, &scratchDirty);
            ColdHashSet<PodKeyStore<int32_t>> closedTrialScopes(&gArena, &scratchDirty);
            auto& trials = lbMemory.integrationTrials;
            for (int32_t id = 1; id <= trials.count(); ++id) {
                const int64_t key = trials.keyAt(id);
                const IntegrationTrialRec& trial = trials.valueAt(id);
                if (inClosedBit(Codec<StatementKey>::decode(key).validity)
                    || (trial.status == IntegrationTrialRec::kOpen && inClosedBit(trial.trialScope))) {
                    closedTrials.mint(key);
                    closedTrialScopes.mint(trial.trialScope);
                }
            }
            if (closedTrials.count() != 0) {
                trials.eraseIf([&](int64_t key) { return closedTrials.lookup(key) != 0; });
                // A record's listed witnesses and expressions leave with it.
                lbMemory.integrationTrialContents.eraseIf([&](int64_t key) {
                    return closedTrialScopes.lookup(
                        static_cast<int32_t>(static_cast<uint64_t>(key) >> 32)) != 0;
                });
            }
        }

        // Packed twin of weakVariables — same low-32-bits validity
        // predicate as the intKnownStatements sweep in step 6.
        intWeakVariables.eraseIf([&](int64_t k) {
            return inClosedBit(Codec<StatementKey>::decode(k).validity);
        });

        // ---- 9. Mail statements (in / out / internal) ----
        // Routing mail is main-only (I-26), so a subtree wipe removes nothing.
        // Walk each channel explicitly and assert the id-space-specific scope
        // contract rather than comparing unrelated interner ids.
        auto filterRoutingMailStatements = [&](RoutingColdMail& m) {
            m.filterStatements(
                [&](const IntMailStatementKey& k) {
                    assert(mailInterner().decodeString(k.validityId) == "main"
                        && "routing mailIn must be main-scope");
                    return false;
                });
        };
        filterRoutingMailStatements(mailIn);
        mailOut.filterStatements([&](const IntMailStatementKey& k) {
            assert(mailOutInterner.decode(k.validityId) == "main"
                && "routing mailOut must be main-scope");
            return false;
        });
        mailOutPending = !mailOut.empty();
        mailOutLiveBytes = mailOut.liveBytes();
        // sameIterationInternalMail is COLD (I-102): the
        // same scope-targeted erase via ColdMail::filterStatements. Only
        // statements is swept (origins / disintegrationSignals untouched),
        // matching the former heap filter. The id-form key carries the
        // statement's NameMap validityId, so the closed-scope test is the same
        // bitmap membership the intStatementLevelsMap / intKnownStatements
        // sweeps use (no decode).
        sameIterationInternalMail.filterStatements(
            [&](const IntMailStatementKey& k) {
                return inClosedBit(static_cast<NameId>(k.validityId));
            });

        // ---- 10. HashMemory wipe (overall / local / delta) ----
        auto wipeHashMem = [&](HashMemory& hm) {
            // 10a. encodedMap LMVs: filter each key's run by the LMV's
            //      scope (a fixed-offset byte peek, no record decode);
            //      drop the key when its run empties. Verbatim two-pass
            //      blob splice through the raw doors — survivor key order,
            //      per-key blob order, and blob bytes identical to the
            //      former decode-erase-reencode rebuild.
            wipeEncodedMapForClosed(hm.encodedMap, closedBits, nameHighWater,
                                    gArena);

            // 10b. Premise patterns: the owners stay (a later rule removal
            //      still finds its own); each closed scope is recorded below,
            //      in the scope-filter order, so the request statement filter
            //      skips the owners whose rules 10a just removed. The install
            //      staging is empty here: the windows closed inside
            //      standardProcessing, before its wipe.
            assert((ruleStagings().boundTo(slot, &hm) == nullptr
                    || ruleStagings().boundTo(slot, &hm)->empty())
                && "wipeSubtrees: staged rule-index writes reached the wipe");

            // The two normalized-key indexes and the remaining-args index are
            // NOT scope-wiped (D-303, I-49): they carry no owner record, so
            // there is nothing to filter, and a stale key is a sound
            // over-approximation — its request reaches the firing site, whose
            // encodedMap lookup (wiped in 10a) misses and skips it.

            // Cold admission/rejection containers
            // (D-172, D-173)
            // use the compacting / run-aware cold erase with the same low-32-bits
            // closed-scope predicate (the packed template key is int64).
            const auto coldScopeWipe = [&](int64_t k) {
                return inClosedBit(Codec<StatementKey>::decode(k).validity);
            };
            hm.admissionMap.eraseBlobIf(coldScopeWipe);
            hm.admissionStatusMap.eraseIf(coldScopeWipe);
            hm.rejectedMap.eraseBlobIf(coldScopeWipe);
            hm.admissionMapIntegration.eraseBlobIf(coldScopeWipe);
            hm.rejectedMapIntegration.eraseBlobIf(coldScopeWipe);
            // Parked or-cohorts of wiped subproof scopes must not leak: the
            // key's validity half carries the cohort parent, so the same
            // closed-bit predicate covers the ordis park.
            hm.rejectedMapOrdis.eraseBlobIf(coldScopeWipe);
            // Demand entries of wiped subproof scopes must not survive to
            // open cohorts in a dead scope — same closed-bit predicate on
            // the key's validity half (D-267).
            hm.admissionMapOrdis2.eraseBlobIf(coldScopeWipe);
            // The ordis2 park index files under the same cohort-parent
            // validity as rejectedMapOrdis — same closed-bit predicate
            // (D-267).
            hm.rejectedMapOrdis2.eraseBlobIf(coldScopeWipe);
            // Parked implication subproofs of a wiped scope must not open in
            // a dead scope — the key's validity half is the preparing scope
            // (D-406).
            hm.rejectedMapSubproof.eraseBlobIf(coldScopeWipe);
            hm.revisitInProgress.eraseIf(coldScopeWipe);
            hm.ordisRevisitInProgress.eraseIf(coldScopeWipe);
            hm.ordis2RevisitInProgress.eraseIf(coldScopeWipe);
            hm.subproofRevisitInProgress.eraseIf(coldScopeWipe);
            hm.admissionSetIntegration.eraseIf(coldScopeWipe);
            hm.triggersForAdmissionSetIntegration.eraseIf(coldScopeWipe);
            // Double-marker demands of a wiped scope must not admit a witness
            // in a dead scope (D-492).
            hm.admissionMapIntegrationDouble.eraseIf(coldScopeWipe);
        };
        wipeHashMem(overallHashMemory);
        wipeHashMem(localHashMemory);
        wipeHashMem(localHashMemoryDelta);

        // Scope filters follow the original root order, then ascending name
        // identifiers within each root. Overlaps belong to their first root;
        // existing entries retain their positions through ordinary deduplication.
        // The closed-scope records of the three wiped hash memories (step
        // 10b) follow the same order, so a bulk wipe leaves the bytes a
        // sequence of single-root wipes leaves.
        const auto insertionLess = [&](int32_t a, int32_t b) {
            const ClosedScope left = closedScopes[a];
            const ClosedScope right = closedScopes[b];
            return left.firstRoot != right.firstRoot
                ? left.firstRoot < right.firstRoot : left.id < right.id;
        };
        ChunkSortedOrdinals<decltype(insertionLess)> insertionOrder(
            gArena, closedScopes.size(), 0, insertionLess);
        for (int32_t index = 0; index < closedScopes.size(); ++index) {
            const NameId closed = closedScopes[insertionOrder.next()].id;
            intValidityNamesToFilter.mint(closed);
            gl::HashRuleIndex::recordClosedPremiseScope(overallHashMemory, closed, gArena);
            gl::HashRuleIndex::recordClosedPremiseScope(localHashMemory, closed, gArena);
            gl::HashRuleIndex::recordClosedPremiseScope(localHashMemoryDelta, closed, gArena);
        }
    }

    // ------------------------------------------------------------------
    // Memory::cloneFactsTemplate — a fresh single-use CE LB from a facts
    // template. See the Doxygen block above the declaration in memory.hpp
    // for the value-copy / reset rationale and the empty-template contract.
    // ------------------------------------------------------------------
    Memory* Memory::cloneFactsTemplate(LbStore& store) const {
        assert(overallHashMemory.encodedMap.empty()
            && "cloneFactsTemplate: template must carry no hash rules "
               "(facts load via the status-4 statement path)");
        // Parked implication subproofs are not cloned (like the ordis parks);
        // a facts template never prepares an integration, so it holds none.
        assert(overallHashMemory.rejectedMapSubproof.empty()
            && "cloneFactsTemplate: template must carry no parked implication subproofs");
        // Nor any integration trial: a facts template mints no witness.
        assert(lbMemory.integrationTrials.count() == 0
            && "cloneFactsTemplate: template must carry no integration trials");
        assert(lbMemory.integrationTrialContents.count() == 0
            && "cloneFactsTemplate: template must carry no integration trial contents");
        // Nor any induction eat: a facts template admits no witness.
        assert(lbMemory.inductionEats.count() == 0
            && "cloneFactsTemplate: template must carry no induction eats");
        // Nor any induction firing or settled tree.
        assert(lbMemory.inductionFirings.count() == 0
            && "cloneFactsTemplate: template must carry no induction firings");
        assert(lbMemory.inductionSettledRoots.count() == 0
            && "cloneFactsTemplate: template must carry no settled induction trees");
        // Nor any contradiction scope: a facts template opens no reductio.
        assert(lbMemory.contradictionScopes.count() == 0
            && "cloneFactsTemplate: template must carry no contradiction scopes");
        // Nor any int scope: a facts template files no premise-bound
        // integration.
        assert(lbMemory.intScopes.count() == 0 && lbMemory.intScopeGroups.count() == 0
            && lbMemory.intScopeFirstFinish.count() == 0
            && "cloneFactsTemplate: template must carry no int scopes");

        Memory* out = store.create<Memory>();

        // Fact-statement containers — the outputs of the status-4 load. All
        // pure value types (EncodedExpression / IntEncodedExpr / packed keys),
        // so copy-assignment is a safe deep copy.
        out->intEncodedStatements           = intEncodedStatements;
        out->intLocalEncodedStatementsSet.copyFrom(intLocalEncodedStatementsSet);
        out->intLocalEncodedStatements      = intLocalEncodedStatements;
        out->intLocalEncodedStatementsDelta = intLocalEncodedStatementsDelta;
        out->intStatementLevelsMap.copyFrom(intStatementLevelsMap);
        out->intKnownStatements.copyFrom(intKnownStatements);
        out->orImplications.copyFrom(orImplications);
        out->equivalenceClassesMap.copyFrom(equivalenceClassesMap);
        out->canonicalEquivalenceClassesMap.copyFrom(canonicalEquivalenceClassesMap);
        out->lbMemory.canonicalMemberRepresentatives.copyFrom(lbMemory.canonicalMemberRepresentatives);
        out->lbMemory.canonicalClassChanges.copyFrom(lbMemory.canonicalClassChanges);
        out->intWeakVariables.copyFrom(intWeakVariables);
        for (int32_t i = 1; i <= existenceReplays.count(); ++i) {
            const int64_t key = existenceReplays.keyAt(i);
            const int32_t completed = *existenceReplays.find(key) & 3;
            if (completed != 0) out->existenceReplays.insert(key, completed);
        }
        // A facts clone starts with empty mail and demand work. Preserve source
        // context and completed constructions, while pending replay may be
        // requested afresh by this clone's own demands.
        out->lbMemory.templateStrings.copyFrom(lbMemory.templateStrings);
        out->lbMemory.valueStrings.copyFrom(lbMemory.valueStrings);
        out->lbMemory.originStrings.copyFrom(lbMemory.originStrings);
        out->exprOriginMap.copyFrom(exprOriginMap);
        const auto copySources = [](HashMemory& target, const HashMemory& source) {
            target.rejectedMap.copyFrom(source.rejectedMap);
            target.rejectedMapIntegration.copyFrom(source.rejectedMapIntegration);
            target.varsInRejectedMapIntegrationKeys.copyFrom(source.varsInRejectedMapIntegrationKeys);
        };
        copySources(out->overallHashMemory, overallHashMemory);
        copySources(out->localHashMemory, localHashMemory);
        copySources(out->localHashMemoryDelta, localHashMemoryDelta);
        copySources(out->workingMemory, workingMemory);

        // Name dictionary — independent copy so the clone mints the conjecture's
        // names without touching the shared template. The façade copies via
        // copyFrom: the cold tables re-intern in id order onto the CLONE's
        // manager (every id matches the template's), the id-form metadata
        // copies as plain containers.
        out->nameMap.copyFrom(nameMap);
        // The negation ledger travels with the facts it mirrors (a clone's
        // contradiction gate must see the template's negated facts).
        out->negationLedger.copyFrom(negationLedger);
        // `goalLedger` is derived from the goal registry, which a clone does
        // not inherit; it stays stale and rebuilds on first probe.

        // Scalars the template carries from its load (everything else stays
        // Memory()-default).
        out->level        = level;
        out->startInt     = startInt;
        out->startIntRepl = startIntRepl;
        out->startIntPi   = startIntPi;

        // Hash rules, demands, mail, flags and goals
        // stay at their Memory() defaults:
        // empty containers, isActive=true,
        // contradictionIndex=-1, primedForContradiction=false. intToBeProved is
        // now a persistent-pool member (its own fresh persistentArena), so the
        // clone starts it empty automatically — no copyFrom, by design.
        return out;
    }

    /// @brief The process-wide, never-deloaded skeleton string interner. See
    ///        the declaration in `memory.hpp` for the full contract.
    ColdStringTable& skeletonInterner() {
        static LbArena arena{ &lbMemory() };
        static DirtyState dirty = DirtyState::Clean;
        static ColdStringTable table{ &arena, &dirty };
        return table;
    }

    /// @brief Return the process-wide, never-deloaded global mail string
    ///        interner.
    ///
    /// @details
    /// The table provides one shared id space for routing mail retained in the
    /// current delivery-history window. Its arena stays attached to
    /// `mailMemory`; rolling grids may clear the table at a quiescent
    /// post-phase-3 seam through `resetMailInterner`, while dormant grids retain
    /// it for the complete execution batch.
    ///
    /// @return The singleton global mail interner.
    /// @invariant Mutation is single-threaded; parallel proof phases only read
    ///            the table.
    /// @see resetMailInterner, MailLog::retireDeliveredBatches.
    ColdStringTable& mailInterner() {
        static LbArena arena{ &mailMemory() };
        static DirtyState dirty = DirtyState::Clean;
        static ColdStringTable table{ &arena, &dirty };
        return table;
    }

    /// @brief Clear the global mail interner after a delivered history window
    ///        has been retired.
    ///
    /// @details
    /// Invalidates every global mail id and returns the table's pages to its
    /// mail-pool arena for reuse. The arena keeps its physical blocks, bounding
    /// the interner by the largest single delivery window instead of cumulative
    /// windows. The caller must first prove that no retained `MailLog` blob,
    /// routing `mailIn`, or other live object can decode an old id.
    ///
    /// @return Nothing.
    /// @invariant Called single-threaded after phase 3 joins and after
    ///            `MailLog::retireDeliveredBatches`, before the next commit.
    /// @see mailInterner, Prover::proveKernel.
    void resetMailInterner() {
        mailInterner().resetToFresh();
    }

    /// @brief The process-wide batch byte dictionary. See the declaration in
    ///        `memory.hpp` for the full contract.
    ///
    /// @details Constructed and activated on the first call; thread-safe
    /// static initialization runs both exactly once.
    ///
    /// @return The active singleton dictionary.
    BatchByteDictionary& batchByteDictionary() {
        static BatchByteDictionary dictionary{ &lbMemory() };
        static const bool activated = (dictionary.activate(), true);
        assert(activated);
        return dictionary;
    }

    /// @brief The process-wide nomination registry. See the declaration.
    ///
    /// @details Constructed empty on the first call.
    ///
    /// @return The singleton registry.
    BatchByteCandidateRegistry& batchByteCandidates() {
        static BatchByteCandidateRegistry registry;
        return registry;
    }

    /// @brief Activate batch byte sharing for this process. See the declaration.
    ///
    /// @details The stripes draw their pages from the main pool like the
    /// other scratch registries.
    ///
    /// @return Nothing.
    void initBatchByteSharing() {
        batchByteCandidates().init(&staticMemory(), kBatchByteCandidateStripes);
        batchByteDictionary();
    }

    /// @brief The LB's expression key as a string. See the declaration.
    std::string Memory::exprKey() const {
        return exprKeyId == 0 ? std::string()
                              : skeletonInterner().decodeString(exprKeyId);
    }

    /// @brief The LB's expression key as a zero-copy span. See the declaration.
    StrSpan Memory::exprKeyView() const {
        return exprKeyId == 0 ? StrSpan(nullptr, 0)
                              : skeletonInterner().view(exprKeyId);
    }

    /// @brief Set the LB's expression key, interning it. See the declaration.
    void Memory::setExprKey(const std::string& s) {
        exprKeyId = s.empty() ? 0 : skeletonInterner().intern(s);
    }

    /// @brief Zero-copy view of the LB's variable definition-set record.
    ///        See the declaration.
    ///
    /// @details The `skeletonInterner()` bytes of `varDefSetsId`, or an empty
    /// span when the LB holds no record.
    ///
    /// @return A span over the record, or an empty span.
    /// @see setVarDefSets.
    StrSpan Memory::varDefSetsView() const {
        return varDefSetsId == 0 ? StrSpan(nullptr, 0)
                                 : skeletonInterner().view(varDefSetsId);
    }

    /// @brief Record the LB's variable definition sets. See the declaration.
    ///
    /// @details Interns the record once; an empty record stores 0.
    ///
    /// @param pairs The `name:label;name:label` record.
    /// @see varDefSetsView.
    void Memory::setVarDefSets(StrSpan pairs) {
        varDefSetsId = pairs.len == 0 ? 0 : skeletonInterner().intern(pairs);
    }

    /// @brief Build an LB's full chain string — every `exprKey` from the
    ///        LB itself up to the root sentinel, joined by `__`.
    ///
    /// @details
    /// The statification deload identity (file naming, header chain
    /// field): same walk as the hashburst dump's LB-chain section,
    /// flattened to one string. The root sentinel's empty `exprKey` is
    /// included as the final (empty) element, so a child of the root
    /// ends in a trailing `__` and the root alone yields the empty
    /// string. The full parent chain — never a bare `exprKey` — is the
    /// LB identity (Rule 12: many LBs share an `exprKey` at different
    /// tree positions).
    ///
    /// @param body The LB to identify.
    /// @return The chain string, innermost first.
    /// @see `lbdeload::deloadFileName`, `Memory::deloadStaticContainers`.
    std::string buildLbChainString(const Memory& body) {
        std::string chain;
        const Memory* cur = &body;
        bool first = true;
        while (cur != nullptr) {
            if (!first) chain += "__";
            chain += cur->exprKey();
            first = false;
            cur = cur->parentMemory;
        }
        return chain;
    }

    /// @brief Assign this LB's deload ordinal on first sight and record its
    ///        `ordinal -> chain` mapping in the global registry.
    ///
    /// @details
    /// See the declaration for the full contract. The `deloadOrdinal < 0`
    /// guard is the single assignment that makes the file-name identity
    /// lifetime-stable; the chain rebuild is a short parent-chain walk (the
    /// LB's stable identity, Rule 12). Re-registers every call so the
    /// per-batch-cleared registry always carries the current LB.
    void Memory::ensureDeloadOrdinal() {
        if (deloadOrdinal < 0)
            deloadOrdinal = staticMemory().assignDeloadOrdinal();
        staticMemory().registerDeloadOrdinal(deloadOrdinal,
                                              buildLbChainString(*this));
    }

    /// @brief Stream the statified containers to their `.deload/` file
    ///        set, return the blocks to the pool, and flag the LB
    ///        deloaded.
    ///
    /// @details
    /// Composes the standalone primitives in the contract order:
    /// `lbdeload::dumpLbMemory` (canonical element-order stream —
    /// I-103), then `release()` on every
    /// statified container (pages back to the LB queue), then
    /// `manager.releaseAll()` (blocks back to the global pool), then
    /// `markDeloaded()`. The file list + chain land in `deloadFiles` /
    /// `deloadChain` for `ensureLoaded`. Asserts the LB is resident — a
    /// double deload is a lifecycle bug, not a no-op.
    ///
    /// @param directory The deload directory (production: `.deload`).
    void Memory::dumpStaticContainers(const std::string& directory) {
        assert(lbMemory.manager.resident()
            && "dumpStaticContainers on a deloaded LB");
        deloadChain = buildLbChainString(*this);
        // Fallback single-assignment for direct (test / teardown) callers;
        // production pre-stamps at the barrier, making this idempotent.
        ensureDeloadOrdinal();
        // Three-way deload decision (skip / tail-delta / full rewrite —
        // the WAL/threshold pattern, all branches defined results):
        // - Clean: the on-disk file set already equals the in-memory
        //   content — release blocks without writing.
        // - AppendedOnly within thresholds: only push_backs happened
        //   since the last dump — write a small tail file set with the
        //   new rows; base files stay untouched.
        // - Restructured, first dump, or thresholds exceeded: full
        //   canonical rewrite (which is also where the straightening
        //   re-canonicalizes the whole image).
        const bool haveBase = !deloadFiles.empty();
        if (lbMemory.dirty == DirtyState::Clean && haveBase) {
            // Skip: nothing changed since RAM == disk was established.
            assert(deloadKind == DeloadKind::Canonical
                && "v3 dumpStaticContainers skip-clean over a raw base — the "
                   "v3 path only handles discharge, which is never Clean");
            deloadStats().evictSkippedClean.fetch_add(
                1, std::memory_order_relaxed);
        }
        else {
            // Every write below is the v3 canonical stream.
            deloadKind = DeloadKind::Canonical;
            // Current totals (tag order) and the appended-row volume relative to
            // the counts recorded at the LAST deload. The four HashMemory
            // instances ride lbMemory.visitContainers now
            // (D-147), so their tags 51..450 enter the counts
            // AND the dump through the one enumeration.
            std::vector<int32_t> currentCounts;
            lbMemory.visitContainers(
                [&currentCounts](LbMemory::ContainerTag,
                                 const auto& container) {
                    currentCounts.push_back(container.size());
                });
            int64_t newTailRows = 0;
            if (haveBase && deloadedCounts.size() == currentCounts.size()) {
                for (std::size_t i = 0; i < currentCounts.size(); ++i)
                    newTailRows += currentCounts[i] - deloadedCounts[i];
            }
            const bool tailEligible =
                lbMemory.dirty == DirtyState::AppendedOnly
                && haveBase
                && deloadedCounts.size() == currentCounts.size()
                && deloadTailCount < lbdeload::kMaxTailSets
                && (deloadTailRows + newTailRows)
                       * lbdeload::kTailCompactionDenominator
                   < deloadBaseRows;
            if (tailEligible) {
                const std::vector<std::string> tailFiles =
                    lbdeload::dumpLbMemoryTail(
                        lbMemory, deloadChain, deloadOrdinal, deloadedCounts,
                        static_cast<uint32_t>(deloadTailCount + 1),
                        directory, staticMemory().blockBytes());
                deloadFiles.insert(deloadFiles.end(), tailFiles.begin(),
                                   tailFiles.end());
                ++deloadTailCount;
                deloadTailRows += newTailRows;
                deloadStats().tailDumpCount.fetch_add(
                    1, std::memory_order_relaxed);
            }
            else {
                deloadFiles = lbdeload::dumpLbMemory(
                    lbMemory, deloadChain, deloadOrdinal, directory,
                    staticMemory().blockBytes());
                deloadStats().fullDumpCount.fetch_add(
                    1, std::memory_order_relaxed);
                deloadTailCount = 0;
                deloadTailRows = 0;
                deloadBaseRows = 0;
                for (const int32_t c : currentCounts)
                    deloadBaseRows += c;
            }
            // Record per-container totals (tag order) — the only legal
            // metadata about the containers while the LB is cold, and
            // the next tail's start positions.
            deloadedCounts = currentCounts;
        }
        // RAM == disk holds from here until the next mutation.
        lbMemory.dirty = DirtyState::Clean;
    }

    /// @brief Return the LB's blocks to the pool and flag it deloaded —
    ///        legal only when the on-disk image is fresh.
    ///
    /// @details
    /// The pressure path: asserts residency, a recorded file set, and
    /// `DirtyState::Clean` (i.e., `dumpStaticContainers` ran after the
    /// last mutation — releasing stale-imaged blocks would lose
    /// content). Releases every statified container's pages, returns the
    /// blocks, `markDeloaded()`.
    void Memory::releaseStaticBlocks() {
        assert(lbMemory.manager.resident()
            && "releaseStaticBlocks on an already deloaded LB");
        assert(!deloadFiles.empty()
            && "releaseStaticBlocks without an on-disk image");
        assert(lbMemory.dirty == DirtyState::Clean
            && "releaseStaticBlocks with a stale on-disk image - dump "
               "first");
        lbMemory.visitContainers(
            [](LbMemory::ContainerTag, auto& container) {
                container.release();
            });
        // The four HashMemory instances now ride lbMemory.visitContainers
        // (D-147), so the walk above releases their cold
        // containers too -- the former manual per-instance release (needed while
        // they lived outside LbMemory) is retired.
        // The derived reverse membership indexes are NOT enrolled in
        // visitContainers, so the walk above did not free their throw-away pages.
        // Free them now, while the arena is still resident, BEFORE releaseAll
        // returns the blocks -- else their stale vids would double-free at the
        // next canonical reload's rebuild (I-154). The raw
        // release path (releaseStaticBlocksRaw) deliberately skips this: the raw
        // image carries these pages verbatim.
        lbMemory.overallHashMemory.remainingArgsReverseIndex.clear();
        lbMemory.localHashMemory.remainingArgsReverseIndex.clear();
        lbMemory.localHashMemoryDelta.remainingArgsReverseIndex.clear();
        lbMemory.workingMemory.remainingArgsReverseIndex.clear();
        lbMemory.overallHashMemory.originalsHeadIndex.clear();
        lbMemory.localHashMemory.originalsHeadIndex.clear();
        lbMemory.localHashMemoryDelta.originalsHeadIndex.clear();
        lbMemory.workingMemory.originalsHeadIndex.clear();
        lbMemory.sharedNativeKeys.release();
        orImplicationMatches.clear();
        lbMemory.existenceLookups.clear();
        lbMemory.canonicalUses.clear();
        lbMemory.contradictionFacts.clear();
        // The release() calls escalate the dirty state (they are
        // mutators); the next reload re-establishes RAM == disk.
        lbMemory.dirty = DirtyState::Clean;
        lbMemory.manager.releaseAll();
        lbMemory.manager.markDeloaded();
    }

    /// @brief Dump + release in one call — the unconditional-deload
    ///        composition (tests, teardown paths).
    ///
    /// @param directory The deload directory (production: `.deload`).
    void Memory::deloadStaticContainers(const std::string& directory) {
        dumpStaticContainers(directory);
        releaseStaticBlocks();
    }

    /// @brief Write-through v4 RAW dump — the near-memcpy eviction dump (see
    ///        the declaration for the full contract).
    ///
    /// @details
    /// Captures `deloadedCounts` (the only legal cold metadata) BEFORE the dump
    /// — the raw image never releases the containers, so the counts are read
    /// straight off the live `size()`s. Skip-clean is a defined cache-hit
    /// (Rule 19), not a fallback. The raw path writes ONE file and no tails, so
    /// it clears the v3 tail bookkeeping.
    ///
    /// @param directory The deload directory (production: `.deload`).
    void Memory::dumpStaticContainersRaw(const std::string& directory) {
        assert(lbMemory.manager.resident()
            && "dumpStaticContainersRaw on a deloaded LB");
        deloadChain = buildLbChainString(*this);
        ensureDeloadOrdinal();
        const bool useExtent = staticMemory().useExtent();
        // A live slab exists (skip-clean / in-place eligible) when this LB has
        // a raw base. On the extent path the base is a slab in the current
        // epoch; on the named path it is a file in deloadFiles.
        const bool haveExtentSlab = useExtent
            && rawExtentOffset_ >= 0
            && rawExtentEpoch_ == staticMemory().extentEpoch()
            && deloadKind == DeloadKind::Raw;
        const bool haveRawBase = useExtent
            ? haveExtentSlab
            : (!deloadFiles.empty() && deloadKind == DeloadKind::Raw);
        if (lbMemory.dirty == DirtyState::Clean && !lbMemory.canonicalUses.hasRawChanges() && haveRawBase) {
            // Skip: the on-disk raw image already equals the in-memory content.
            deloadStats().evictSkippedClean.fetch_add(
                1, std::memory_order_relaxed);
        }
        else {
            // Cold metadata: per-container totals in tag order (the only legal
            // read while the LB is deloaded, I-111). Captured off the live
            // sizes — the raw dump releases nothing. Written DIRECTLY into
            // deloadedCounts (clear + push_back reuses its capacity after the
            // first dump): this is the eviction hot loop, where per-operation
            // fixed costs dominate on small LBs, so a per-dump heap temporary
            // is not acceptable. The v3 path keeps its temporary (discharge
            // only — off the hot loop, and its tail decision reads BOTH the
            // old and the new counts at once).
            deloadedCounts.clear();
            lbMemory.visitContainers(
                [this](LbMemory::ContainerTag, const auto& container) {
                    deloadedCounts.push_back(container.size());
                });
            if (useExtent) {
                // Extent path: place the image at the LB's slab. A stable /
                // shrinking size overwrites IN PLACE (no allocator touch — the
                // near-memcpy hot path); a first dump OR a growth past the slab
                // class allocates a fresh slab (freeing the old one first).
                const int64_t need =
                    lbdeload::rawImageBytesFor(lbMemory, deloadChain);
                const bool haveSlab = rawExtentOffset_ >= 0
                    && rawExtentEpoch_ == staticMemory().extentEpoch();
                if (!haveSlab || need > rawExtentClassBytes_) {
                    if (haveSlab)
                        staticMemory().freeExtentSlab(deloadOrdinal,
                                                      rawExtentOffset_,
                                                      rawExtentClassBytes_);
                    const SlabAllocation slab =
                        staticMemory().allocateExtentSlab(deloadOrdinal, need);
                    rawExtentOffset_ = slab.offset;
                    rawExtentClassBytes_ =
                        static_cast<int32_t>(slab.classBytes);
                    rawExtentEpoch_ = staticMemory().extentEpoch();
                }
                lastRawImageBytes_ = lbdeload::dumpLbMemoryRawAt(
                    lbMemory, deloadChain, deloadOrdinal,
                    staticMemory().extentFile(), rawExtentOffset_,
                    rawExtentClassBytes_);
                // extentLiveBytes delta: this LB's occupancy replaces its prior.
                const int64_t image =
                    lbdeload::rawImageBytesFor(lbMemory, deloadChain);
                staticMemory().addExtentLive(image - lastExtentLiveBytes_);
                lastExtentLiveBytes_ = image;
                // The extent path does not use deloadFiles (reload seeks to
                // rawExtentOffset_); keep it empty so a stale name is never read.
                deloadFiles.clear();
            } else {
                lastRawImageBytes_ = lbdeload::dumpLbMemoryRaw(
                    lbMemory, deloadChain, deloadOrdinal, directory);
                deloadFiles.clear();
                deloadFiles.push_back(lbdeload::rawFileName(deloadOrdinal));
            }
            deloadKind = DeloadKind::Raw;
            // The raw path writes no tails — reset the v3 tail bookkeeping so a
            // later v3 discharge dump of this LB starts a clean base.
            deloadTailCount = 0;
            deloadTailRows = 0;
            deloadBaseRows = 0;
            deloadStats().fullDumpCount.fetch_add(
                1, std::memory_order_relaxed);
        }
        // RAM == disk holds from here until the next mutation.
        lbMemory.dirty = DirtyState::Clean;
        lbMemory.canonicalUses.markRawClean();
    }

    /// @brief Return the LB's blocks WITHOUT the container-release walk — the
    ///        raw eviction's release step (see the declaration for the
    ///        contract).
    ///
    /// @details
    /// NO `container.release()` walk (the near-memcpy win): the raw image
    /// preserves the vids, so the container bookkeeping must SURVIVE for the
    /// raw reload to rebind it. Just return the blocks and flag deloaded; the
    /// dirty state is already `Clean` from the dump and stays `Clean` (no
    /// mutator ran). The teardown residency branch in `~PagedVector` /
    /// `~PagedHashIndex` makes a destruction on the now-deloaded arena a
    /// no-op; a LIVE `clear()` there still asserts.
    void Memory::releaseStaticBlocksRaw() {
        assert(lbMemory.manager.resident()
            && "releaseStaticBlocksRaw on an already deloaded LB");
        // The fresh-raw-image evidence is the slab on the extent path, the
        // file set on the named path.
        assert((staticMemory().useExtent()
                    ? rawExtentOffset_ >= 0
                    : !deloadFiles.empty())
                && deloadKind == DeloadKind::Raw
            && "releaseStaticBlocksRaw without a fresh raw image");
        assert(lbMemory.dirty == DirtyState::Clean
            && "releaseStaticBlocksRaw with a stale image - dump first");
        lbMemory.manager.releaseAllForRawImage();
        lbMemory.manager.markDeloaded();
    }

    /// @brief Release through the path matching the recorded on-disk format —
    ///        the release-side mirror of `reloadFromImage`'s dispatch (see the
    ///        declaration for the full contract).
    void Memory::releaseStaticBlocksDispatch() {
        if (deloadKind == DeloadKind::Raw)
            releaseStaticBlocksRaw();
        else
            releaseStaticBlocks();
    }

    /// @brief Raw dump + raw release in one call (see the declaration).
    ///
    /// @param directory The deload directory (production: `.deload`).
    void Memory::deloadStaticContainersRaw(const std::string& directory) {
        dumpStaticContainersRaw(directory);
        releaseStaticBlocksRaw();
    }

    /// @brief The discharge protocol's memory step: capture the exact
    ///        registry record, empty every dischargeable container, and
    ///        reshuffle — the copying compaction returns every block except
    ///        the cold-string survivors' to the pool with zero SSD I/O.
    ///
    /// @details
    /// See the declaration for the full contract. Order is load-bearing:
    /// the capture reads `intEncodedStatements` BEFORE the clear; the
    /// reshuffle after the clear reclaims the freed storage through the
    /// copying compaction (see `LbMemory::reshuffle`).
    /// `intLocalEncodedStatementsSet` — the local-origin gate's exact
    /// mirror — survives discharge as a cold tag (`LbMemory::survivesDischarge`
    /// keeps it), so the recursion-node gate still reads it post-prove.
    /// The persistent goal registry `intToBeProved` is reclaimed here too
    /// (resetToFresh then `persistentArena.releaseAll()`) — its block was
    /// attached only until discharge ([I-108]).
    void Memory::dischargeStatementContent(LbArena& scr) {
        assert(lbMemory.manager.resident()
            && "discharge on a deloaded LB - the protocol runs at the "
               "barrier of the LB's last active iteration");
        assert(!dischargedForever
            && "dischargeStatementContent ran twice on one LB");
        dischargedForever = true;
        // A raw-evicted LB being discharged flips Raw->Canonical (the next dump
        // is v3) and never raw-reloads again, so its extent slab is returned to
        // the allocator free-list — the one new line the extent path adds to
        // discharge (D-195 §4). Guarded on a live slab
        // in the current epoch (a purged prior-batch slab was already recycled).
        if (staticMemory().useExtent() && rawExtentOffset_ >= 0
            && rawExtentEpoch_ == staticMemory().extentEpoch()) {
            staticMemory().freeExtentSlab(deloadOrdinal, rawExtentOffset_,
                                          rawExtentClassBytes_);
            staticMemory().addExtentLive(-lastExtentLiveBytes_);
            lastExtentLiveBytes_ = 0;
            rawExtentOffset_ = -1;
            rawExtentClassBytes_ = 0;
            rawExtentEpoch_ = -1;
        }
        for (int32_t r = 0; r < intEncodedStatements.size(); ++r) {
            dischargedRegistryKeys.insert(packStatementKey(
                intEncodedStatements[r].originalId,
                intEncodedStatements[r].validityId));
        }
        // Reclaim the persistent goal registry: its block is attached to the LB
        // only until discharge (I-108). resetToFresh frees
        // its pages into persistentArena FIRST, then releaseAll returns the
        // arena's blocks to persistentMemory() (the reverse order would free
        // pages into a released arena and assert). A discharged LB never reads
        // intToBeProved again (it is !isActive; ensureLoaded asserts the flag).
        intToBeProved.resetToFresh();
        pendingDisprovedGoals.resetToFresh();
        pendingDeadOrBranches.resetToFresh();
        orRetiredDisjuncts.resetToFresh();
        pendingOrReleases.resetToFresh();
        orStarterPick.resetToFresh();
        processedOrLedger.resetToFresh();
        negationLedger.resetToFresh();
        goalLedger.resetToFresh();
        goalLedgerStale = true;
        pendingWatchArrivals.clear();
        pendingDeferredOrEvents.clear();
        deferredOrWaiting.resetToFresh();
        pendingWatchRechecks.clear();
        pendingSubproofArrivals.clear();
        pendingSubproofRechecks.clear();
        pendingContradictionArrivals.clear();
        pendingContradictionDemands.clear();
        orLiveBranches.resetToFresh();
        frozenOrBranches.resetToFresh();
        recursionHeadWitnesses.resetToFresh();
        hiddenArgumentWitnesses.resetToFresh();
        builtOnListedWitnesses.resetToFresh();
        subproofGoals.resetToFresh();
        persistentArena.releaseAll();
        lbMemory.clearDischargeableContainers();
        lbMemory.reshuffle(scr);
        // The cold string tags survive discharge (their ids stay referenced
        // by RAM-side containers); the emptied statement storage becomes
        // arena holes the copying compaction reclaims. The byte-exact
        // survivor residual check lands with that compaction.
    }

    /// @brief Bring a deloaded LB's statified containers back from the
    ///        file set recorded at deload; no-op when resident.
    ///
    /// @details
    /// The resident no-op is a defined contract (the touch points call
    /// unconditionally), not a fallback. On the reload path: asserts the
    /// chain still matches the LB's tree position, `markResident()`,
    /// then `lbdeload::loadLbMemory` rebuilds every container
    /// element-by-element onto a fresh consecutive virtual index (the
    /// approved "straightening").
    ///
    /// @param directory The deload directory the files live in.
    void Memory::ensureLoaded(const std::string& directory) {
        if (lbMemory.manager.resident()) return;
        assert(!dischargedForever
            && "ensureLoaded on a discharged LB - deactivation is "
               "permanent and the kernel-side readers probe RAM records "
               "(I-112); the post-prove export "
               "uses ensureLoadedForRead");
        reloadFromImage(directory);
    }

    /// @brief Sanctioned post-prove READ reload — like `ensureLoaded` but
    ///        legal on a discharged LB (D-158).
    ///
    /// @details
    /// The chapter export (`buildStack`) reads origin history in full; the
    /// strings campaign moved that history's strings into the deloadable
    /// cold image, so a discharged-and-drained LB must come back for its
    /// chapter walk. This does NOT reactivate anything: `isActive` and
    /// `dischargedForever` are untouched, no kernel runs after the export,
    /// and the reload is content-identical to the dumped image
    /// (I-112 keeps its meaning — an LB never
    /// returns to the ACTIVE set; readability is restored deliberately).
    /// Defined no-op when resident, like `ensureLoaded`.
    ///
    /// @param directory The deload directory the image lives in.
    void Memory::ensureLoadedForRead(const std::string& directory) {
        if (lbMemory.manager.resident()) return;
        // No request generation follows a read-only reload — every consumer of
        // this door reads origin history only (the chapter export's walk and
        // `repairTierCitationOrigins`). The reverse membership indexes exist
        // solely for the firing-check candidate probe, so rebuilding them here
        // is pure cost; on the contradiction-twin probe it dominated the whole
        // chapter export. They are marked unbuilt instead, which turns any
        // future reader on this path into an assert rather than a silent
        // empty-candidate answer.
        reloadFromImage(directory, /*rebuildDerivedIndexes=*/false);
    }

    // Export-phase reload sink (declared in memory.hpp). Null except during
    // the chapter export, which points it at a local vector so it can release
    // per-chapter the LBs the read-only walk reloads
    // (G-53).
    std::vector<Memory*>* g_exportReloadSink = nullptr;

    /// @brief Shared reload core of `ensureLoaded` / `ensureLoadedForRead`:
    ///        chain check, mark resident, element-by-element rebuild, dirty
    ///        reset. Caller established non-residency and its own policy
    ///        asserts.
    ///
    /// @param directory The deload directory the image lives in.
    /// @param rebuildDerivedIndexes When true (the default, and every
    ///        work-path caller), the canonical branch re-derives the four
    ///        `ReverseArgsIndex` membership indexes, which the canonical
    ///        stream deliberately omits (`I-117` / `I-154`). When false — the
    ///        read-only door — the rebuild is skipped and each index is marked
    ///        unbuilt, because no request generation follows. The RAW branch
    ///        ignores this: its whole-arena image restores the indexes
    ///        verbatim, so there is nothing to rebuild or to skip.
    void Memory::reloadFromImage(const std::string& directory,
                                 bool rebuildDerivedIndexes) {
        // The image reference is the extent slab on the raw-extent path, the
        // recorded file set otherwise.
        const bool extentRaw = deloadKind == DeloadKind::Raw
            && staticMemory().useExtent();
        assert((extentRaw ? rawExtentOffset_ >= 0 : !deloadFiles.empty())
            && "reload on a deloaded LB with no recorded image reference");
        assert(buildLbChainString(*this) == deloadChain
            && "LB moved in the tree between deload and reload");
        lbMemory.manager.markResident();
        // Dispatch on the recorded format of the LAST dump. Raw = the v4
        // near-memcpy arena image (one file, no rebuild); Canonical = the v3
        // element stream (discharge / export). Both loaders assert their own
        // header version, so a field/file mismatch is caught (Rule 19).
        if (deloadKind == DeloadKind::Raw) {
            if (extentRaw) {
                // Extent path: seek to the LB's slab in the one open file.
                lbdeload::loadLbMemoryRawAt(
                    lbMemory, deloadChain, deloadOrdinal,
                    staticMemory().extentFile(), rawExtentOffset_);
            } else {
                assert(deloadFiles.size() == 1
                    && "raw image is a single file");
                lbdeload::loadLbMemoryRaw(lbMemory, deloadChain, deloadOrdinal,
                                          deloadFiles.front(), directory);
            }
        }
        else {
            // The four HashMemory instances reload through
            // lbMemory.visitContainers (D-147), so loadLbMemory rebuilds their
            // tags 51..450 alongside LbMemory's own -- no extra-column list.
            lbdeload::loadLbMemory(lbMemory, deloadChain, deloadFiles,
                                   directory);
            // The derived reverse membership indexes are NOT in the canonical
            // deload stream (I-117 pattern), so a canonical reload restores an
            // empty index -- rebuild each from its now-resident forward map, the
            // twin of the cold-map family's KeysView -> rebuildIndex reload hook.
            // The RAW branch above needs no rebuild (the whole-arena image
            // restored these pages verbatim, I-154).
            //
            // A read-only reload skips this: it is proportional to hash-memory
            // size and only request generation ever reads the result.
            if (!rebuildDerivedIndexes) {
                lbMemory.overallHashMemory.remainingArgsReverseIndex.markUnbuilt();
                lbMemory.localHashMemory.remainingArgsReverseIndex.markUnbuilt();
                lbMemory.localHashMemoryDelta.remainingArgsReverseIndex.markUnbuilt();
                lbMemory.workingMemory.remainingArgsReverseIndex.markUnbuilt();
                lbMemory.overallHashMemory.originalsHeadIndex.markUnbuilt();
                lbMemory.localHashMemory.originalsHeadIndex.markUnbuilt();
                lbMemory.localHashMemoryDelta.originalsHeadIndex.markUnbuilt();
                lbMemory.workingMemory.originalsHeadIndex.markUnbuilt();
            } else {
            // The head index is lazy: the canonical image carries no derived
            // pages, so mark it stale and let the next consumer rebuild it.
            lbMemory.overallHashMemory.originalsHeadIndex.markUnbuilt();
            lbMemory.localHashMemory.originalsHeadIndex.markUnbuilt();
            lbMemory.localHashMemoryDelta.originalsHeadIndex.markUnbuilt();
            lbMemory.workingMemory.originalsHeadIndex.markUnbuilt();
            const unsigned revSlot =
                (gl::WorkerContext::coreId() >= 0)
                    ? static_cast<unsigned>(gl::WorkerContext::coreId())
                    : genScratchArenas().slotCount() - 1;
            ScratchArena& revArena = genScratchArenas().forSlot(revSlot);
            const ArenaOffset revMark = revArena.cursor();
            lbMemory.overallHashMemory.remainingArgsReverseIndex.rebuildReverseIndex(
                lbMemory.overallHashMemory.remainingArgsNormalizedEncodedMap,
                revArena);
            lbMemory.localHashMemory.remainingArgsReverseIndex.rebuildReverseIndex(
                lbMemory.localHashMemory.remainingArgsNormalizedEncodedMap,
                revArena);
            lbMemory.localHashMemoryDelta.remainingArgsReverseIndex.rebuildReverseIndex(
                lbMemory.localHashMemoryDelta.remainingArgsNormalizedEncodedMap,
                revArena);
            lbMemory.workingMemory.remainingArgsReverseIndex.rebuildReverseIndex(
                lbMemory.workingMemory.remainingArgsNormalizedEncodedMap,
                revArena);
            revArena.popTo(revMark);
            }
        }
        // The rebuild's clear()+push_back escalate the dirty state, but
        // reload by definition re-establishes RAM == disk: reset so an
        // unchanged LB skips its next dump.
        lbMemory.dirty = DirtyState::Clean;
        // Export-phase reload tracking (G-53): the
        // read-only chapter export reloads LBs on demand; record each so the
        // driver releases it per chapter, else the reloads accumulate to
        // static-pool exhaustion. Null (untracked) during the kernel's own
        // pressure reloads in proveKernel.
        if (g_exportReloadSink != nullptr) g_exportReloadSink->push_back(this);
    }

    // Explicit instantiation of the request-generator member template for the
    // streaming consumer (the only consumer; see BurstSink in prover.hpp).


    // ---- NameMap::encodePush (span overload) --------------------------------
    // Out-of-line: the heap-free canonical build reaches the string scratch
    // registry (scratchArenas()) and the worker-slot id
    // (gl::WorkerContext::coreId()), neither visible to an inline header
    // body. Byte-identical to the former inline heap-std::string body — same
    // canonical bytes (decodeView == decode), same dedup, same subs-then-names
    // mint order.
    NameId NameMap::encodePush(NameId parentId, const StrSpan& payload) {
        // Delimiter self-overlap contract (I-139): the payload may not contain
        // "_boundary_", nor end "_boundary" / start "boundary_".
        const std::string_view pv(payload.ptr,
                                  static_cast<std::size_t>(payload.len));
        assert(pv.find(BOUNDARY_STR, 0, BOUNDARY_LEN) == std::string_view::npos
               && "payload must not contain '_boundary_'");
        assert(!(payload.len >= 9
                 && equalSpans(StrSpan(payload.ptr + payload.len - 9, 9),
                               StrSpan("_boundary", 9)))
               && "payload must not end with '_boundary'");
        assert(!(payload.len >= 9
                 && equalSpans(StrSpan(payload.ptr, 9),
                               StrSpan("boundary_", 9)))
               && "payload must not start with 'boundary_'");
        seedIfEmpty();

        // canonical = decodeView(parentId) + "_boundary_" + payload, assembled
        // on the per-slot string-scratch arena. decodeView aliases the names
        // cold pages; it is copied into the scratch buffer here, before any mint
        // below, so the later mintName cannot dangle it (I-3).
        const StrSpan parentSpan = decodeView(parentId);
        const int32_t total = parentSpan.len
                            + static_cast<int32_t>(BOUNDARY_LEN)
                            + payload.len;
        const unsigned slot = (gl::WorkerContext::coreId() >= 0)
            ? static_cast<unsigned>(gl::WorkerContext::coreId())
            : scratchArenas().slotCount() - 1;
        ScratchArena& sa = scratchArenas().forSlot(slot);
        ScratchScope scope(sa);
        char* const buf = sa.allocBytes(total);
        std::memcpy(buf, parentSpan.ptr,
                    static_cast<std::size_t>(parentSpan.len));
        std::memcpy(buf + parentSpan.len, BOUNDARY_STR, BOUNDARY_LEN);
        std::memcpy(buf + parentSpan.len + BOUNDARY_LEN, payload.ptr,
                    static_cast<std::size_t>(payload.len));
        const StrSpan canonical(buf, total);

        const NameId existing = lookup(canonical);
        if (existing != 0) return existing;

        // Mint order (subs payload, then names canonical) preserved.
        const NameId subId = encodeSub(payload);
        const NameId newId = mintName(canonical);
        nodes->push_back(ValidityNode{parentId, subId});
        return newId;
    }

} // namespace gl
