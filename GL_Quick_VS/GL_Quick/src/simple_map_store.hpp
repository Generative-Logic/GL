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

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>

#include "memory.hpp"
#include "memory_infra/str_ops.hpp"

namespace gl {

    /// @brief One routing edge of the LB tree — a parent's child reached under a
    ///        routing-key id, plus the back-link forming the parent's chain.
    ///
    /// @details
    /// The statified replacement for one `std::map<std::string, Memory*>` entry.
    /// `child` is the address-stable `LbStore` slot the prover navigates to (the
    /// one value the prover DEREFERENCES — unlike the pure-identity `lbKey`); it
    /// stays valid for the child's whole life because the slab never relocates a
    /// slot. `keyId` is the routing key interned in the shared `skeletonInterner()`
    /// (the same table `exprKey` rides), NEVER the raw string. `prev` threads this
    /// parent's edges into a newest-first chain through the shared append-only
    /// `edges` pool, so a new edge never shifts another parent's bytes (the
    /// `mail_log.hpp` `BlobRef` pattern).
    ///
    /// A trivially-copyable POD so `PagedVector<EdgeNode>` holds every LB's edges
    /// in one shared column; the store is never deloaded, so the layout carries no
    /// canonical-bytes obligation. Field order packs to 16 bytes with no padding.
    ///
    /// @see SimpleMapStore, EdgeHead.
    struct EdgeNode {
        /// @brief The child LB reached under this edge (an `LbStore` slot).
        Memory* child;
        /// @brief Routing-key id in `skeletonInterner()` (>= 1; 0 never stored).
        std::int32_t keyId;
        /// @brief Index in `edges` of this parent's previous edge, or -1 for its first.
        std::int32_t prev;
    };

    /// @brief The head of one parent's edge chain — its newest edge and its count.
    ///
    /// @details
    /// The value type of `SimpleMapStore::heads`; a `linkChild` reads then
    /// overwrites it single-threaded. Trivially copyable (a single-value cold map
    /// column). `count` is the parent's child total — the `forEachChild` reserve
    /// hint and a chain-length cross-check.
    ///
    /// @see SimpleMapStore, EdgeNode.
    struct EdgeHead {
        /// @brief Index in `edges` of the parent's newest edge, or -1 if none.
        std::int32_t lastEdge;
        /// @brief Number of edges (children) this parent owns.
        std::int32_t count;
    };

    /// @brief The LB tree's routing edges (`parent -> child` by routing key),
    ///        STATIFIED off the heap onto a never-deloaded arena.
    ///
    /// @details
    /// `SimpleMapStore` replaces the per-LB heap member
    /// `std::map<std::string, Memory*> Memory::simpleMap` — the down-edge set of
    /// the LB tree (`parentMemory` is the inverse up-edge). It is the last large
    /// per-LB heap user on the otherwise-static `Memory` shell, and moving it off
    /// the heap is a step toward the campaign's zero-transitive-malloc-per-LB goal.
    ///
    /// The edges must be readable while an LB's main arena is COLD (the tree is
    /// navigated regardless of deload state), so they cannot ride the deloadable
    /// per-LB arena. The store therefore lives on a dedicated, NEVER-DELOADED
    /// arena (the `ExpressionAnalyzer`-owned `simpleMapArena{ &lbMemory() }`, the
    /// LB-body pool the skeleton interner already uses), exactly the `MailLog`
    /// model: its two cold containers are pure runtime containers — never
    /// enumerated by `LbMemory::visitContainers`, so never deloaded / dirty-tracked
    /// / reshuffled (the `intToBeProved` / `MailLog` precedent); their `DirtyState`
    /// is never read.
    ///
    /// **Per-LB independent chains (the heap-faithful structure).** Each parent's
    /// edges grow INDEPENDENTLY through one shared append-only pool, so an insert
    /// is O(1) and never touches another parent's bytes:
    /// - **`edges`** — `PagedVector<EdgeNode>`: every edge, APPEND-ONLY; each
    ///   parent's edges form a newest-first back-linked chain via `EdgeNode::prev`.
    /// - **`heads`** — `TypedColdMap<int64, EdgeHead>`: parent key -> its chain
    ///   head (newest edge + child count). Written single-threaded at `linkChild`,
    ///   read at every lookup / iteration.
    ///
    /// **Keys.** The routing-key STRING is interned in the process-wide
    /// `skeletonInterner()` (a 4-byte id, deduped across all LBs); the parent's
    /// IDENTITY is `lbKey(parent)` — the `uintptr_t` of its `const Memory*`, NEVER
    /// dereferenced (Rule 12; `exprKey` is not unique, so the pointer is the only
    /// stable LB identity). The store is never deloaded, so a pointer-valued key
    /// carries no canonical form. The routing key may DIFFER from the child's
    /// `exprKey` (recursion block #2), so it is a separate id slot — never derived
    /// from `exprKeyId`.
    ///
    /// **Determinism.** No prover observable depends on edge ENUMERATION order
    /// (every walk is a recursion or a boolean reduction). Even so, `forEachChild` yields
    /// children sorted by their DECODED routing-key string (the id is never
    /// observable, the `exprKey` discipline), reproducing the old
    /// `std::map<std::string, Memory*>` iteration order byte-for-byte — so the
    /// migration changes no artifact.
    ///
    /// **Threading.** `linkChild` runs single-threaded (LB birth — the same sites
    /// and threading as `setExprKey`); the parallel burst only READS via the
    /// non-minting `skeletonInterner().lookup` + frozen `edges` / `heads`, exactly
    /// like the `MailLog` pull.
    ///
    /// **Two instances.** The main tree (`this->body`) and the CE-filter tree
    /// (`ceBody`) can coexist, so each rides its OWN `SimpleMapStore`, cleared at
    /// its own teardown (`destroyGrid` for the main store, the CE teardown in
    /// `filterConjecturesWithCE` for the CE store). A site routes to the store of
    /// the tree it operates on.
    ///
    /// @invariant `linkChild` runs single-threaded; the parallel phase only reads
    ///            the interner / `edges` / `heads` (no insert, no mint) — race-free
    ///            without locks.
    /// @invariant A `(parent, routing-key)` pair is linked at most once (the prover
    ///            guards every insert with a prior `findChild` miss), so a parent's
    ///            chain holds no duplicate `keyId`.
    /// @see mail_log.hpp MailLog — the never-deloaded, pointer-keyed precedent.
    /// @see skeletonInterner (memory.hpp) — the shared routing-key interner.
    struct SimpleMapStore {
        /// @brief Bind the two cold containers to the store arena.
        ///
        /// @details
        /// Both draw their pages from `arena` (the `ExpressionAnalyzer`-owned
        /// `simpleMapArena` / `ceSimpleMapArena` on the never-deloaded LB-body
        /// pool). `dirty` is present only to satisfy the container constructors —
        /// it is NEVER read (these containers produce no deload image).
        ///
        /// @param arena The store arena (on `lbMemory()`).
        /// @param dirty The store dirty flag (never consulted).
        explicit SimpleMapStore(LbArena* arena, DirtyState* dirty)
            : edges(arena, dirty), heads(arena, dirty) {}

        /// Append-only edges; each parent's edges form a newest-first chain.
        PagedVector<EdgeNode> edges;

        /// Parent key -> its chain head (newest edge + child count).
        TypedColdMap<std::int64_t, EdgeHead> heads;

        /// @brief The runtime identity key of an LB — the `uintptr_t` of its
        ///        `const Memory*`, as an `int64_t` (the cold-map family's POD key).
        ///
        /// @details
        /// The pointer is pure identity, never dereferenced (Rule 12). The store is
        /// never deloaded, so a pointer-valued key needs no canonical form. (The
        /// `MailLog::lbKey` twin.)
        ///
        /// @param lb The LB (may be a tree root).
        /// @return Its identity key.
        static std::int64_t lbKey(const Memory* lb) {
            return static_cast<std::int64_t>(reinterpret_cast<std::uintptr_t>(lb));
        }

        /// @brief Install `child` as the LB reached from `parent` under `key`.
        ///
        /// @details
        /// The statified `parent->simpleMap[key] = child`. Called single-threaded
        /// at LB birth: interns `key` in the shared `skeletonInterner()` (the
        /// `setExprKey` threading), appends one `EdgeNode` linked to this parent's
        /// previous edge (O(1), no cross-parent shift), and bumps the parent's
        /// chain head (insert on its first child, in-place overwrite after). The
        /// prover guards every call with a prior `findChild` miss, so a parent's
        /// chain never gains a duplicate routing key.
        ///
        /// @param parent The parent LB (identity only — never dereferenced).
        /// @param key    The routing-key string (a non-empty MPL expression).
        /// @param child  The child LB installed under `key` (an `LbStore` slot).
        void linkChild(const Memory* parent, const std::string& key, Memory* child) {
            const std::int32_t keyId = skeletonInterner().intern(key);
            const std::int64_t p = lbKey(parent);
            const EdgeHead* h = heads.find(p);
            const std::int32_t prev = (h != nullptr) ? h->lastEdge : -1;
            const std::int32_t cnt = (h != nullptr) ? h->count : 0;
            const std::int32_t idx = edges.size();
            edges.push_back(EdgeNode{ child, keyId, prev });
            heads.upsert(p, EdgeHead{ idx, cnt + 1 });
        }

        /// @brief The child reached from `parent` under `key`, or `nullptr`.
        ///
        /// @details
        /// The statified `auto it = parent->simpleMap.find(key)`: a non-minting
        /// `skeletonInterner().lookup` (burst-safe) resolves the routing key to its
        /// id; an unminted key (id 0) is absent by construction (every stored key
        /// was interned), and a scan of the parent's chain returns the matching
        /// child. `nullptr` means "no such edge" — a defined query result, the
        /// bivalent twin of `find == end()`, NOT a defensive fallback (the stored
        /// child is never null, so `nullptr` is unambiguously "absent"). The
        /// per-parent fanout is single digits, so the linear scan beats the old
        /// `std::map` string comparisons.
        ///
        /// @param parent The parent LB (identity only — never dereferenced).
        /// @param key    The routing-key string to look up.
        /// @return The child LB, or `nullptr` if `parent` has no edge under `key`.
        Memory* findChild(const Memory* parent, const std::string& key) const {
            const std::int32_t keyId = skeletonInterner().lookup(StrSpan(key));
            if (keyId == 0) return nullptr;
            const EdgeHead* h = heads.find(lbKey(parent));
            if (h == nullptr) return nullptr;
            for (std::int32_t e = h->lastEdge; e >= 0; e = edges[e].prev) {
                if (edges[e].keyId == keyId) return edges[e].child;
            }
            return nullptr;
        }

        /// @brief Span twin of `findChild(parent, const std::string&)`.
        ///
        /// @details
        /// The identical non-minting lookup over a caller-owned span, for
        /// statified callers that must not materialize a `std::string`
        /// (the contradiction-twin lookups in the discharge functions).
        /// Same contract as the string overload: an unminted key is absent
        /// by construction, and `nullptr` means "no such edge" — a defined
        /// query result, not a defensive fallback.
        ///
        /// @param parent The parent LB (identity only — never dereferenced).
        /// @param key    The routing-key span to look up.
        /// @return The child LB, or `nullptr` if `parent` has no edge under `key`.
        Memory* findChild(const Memory* parent, StrSpan key) const {
            const std::int32_t keyId = skeletonInterner().lookup(key);
            if (keyId == 0) return nullptr;
            const EdgeHead* h = heads.find(lbKey(parent));
            if (h == nullptr) return nullptr;
            for (std::int32_t e = h->lastEdge; e >= 0; e = edges[e].prev) {
                if (edges[e].keyId == keyId) return edges[e].child;
            }
            return nullptr;
        }

        /// @brief Apply `fn` to each child of `parent`, in old-`std::map` key order.
        ///
        /// @details
        /// The statified `for (auto& kv : parent->simpleMap)`. Collects the
        /// parent's chain, sorts it by the DECODED routing-key string
        /// (`compareSpans` over `skeletonInterner().view` — allocation-free,
        /// std::string-identical ordering), then calls `fn(key, child)` in that
        /// order. The sort reproduces the old `std::map<std::string, Memory*>`
        /// iteration order exactly, so no enumeration-order-sensitive artifact
        /// changes. A parent with no edges is a defined no-op.
        ///
        /// @tparam Fn A callable invoked as `fn(const StrSpan& key, Memory* child)`.
        ///            `key` is a zero-copy view into the never-deloaded interner
        ///            (valid for the call); copy it out if it must outlive `fn`.
        /// @param parent The parent LB whose children are enumerated.
        /// @param fn     The per-child callback.
        template <class Fn>
        void forEachChild(const Memory* parent, Fn&& fn) const {
            const EdgeHead* h = heads.find(lbKey(parent));
            if (h == nullptr) return;
            std::vector<std::int32_t> nodes;
            nodes.reserve(static_cast<std::size_t>(h->count));
            for (std::int32_t e = h->lastEdge; e >= 0; e = edges[e].prev) {
                nodes.push_back(e);
            }
            std::sort(nodes.begin(), nodes.end(),
                [this](std::int32_t x, std::int32_t y) {
                    return compareSpans(skeletonInterner().view(edges[x].keyId),
                                        skeletonInterner().view(edges[y].keyId)) < 0;
                });
            for (std::int32_t e : nodes) {
                fn(skeletonInterner().view(edges[e].keyId), edges[e].child);
            }
        }

        /// @brief Drop every edge and chain head at tree teardown.
        ///
        /// @details
        /// Empties both cold containers and returns their pages to the store arena
        /// (the arena keeps its blocks for the next batch). The `Memory*` children
        /// are about to be freed, so this prevents a stale pointer-keyed entry from
        /// aliasing a reused address in the next batch's grid (the `MailLog::clear`
        /// contract). The shared `skeletonInterner()` is NOT touched — it is
        /// process-wide and outlives every grid.
        void clear() {
            edges.clear();
            heads.resetToFresh();
        }

        /// @brief Whether the store holds no edges.
        ///
        /// @return `true` when both cold containers are empty (the post-`clear` /
        ///         pre-first-`linkChild` state).
        bool empty() const {
            return edges.empty() && heads.empty();
        }
    };

}  // namespace gl
