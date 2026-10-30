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
#include "scope_syntax.hpp"
#include <cassert>
#include <cstring>

namespace gl {

/// @brief What one admission route hands the witness-history check
///        (D-522).
///
/// @details
/// The witness-history check is the derivation-history extension of the
/// witness-link gate: in an induction step block, an integration instruction
/// that would otherwise accept a route refuses it when an ancestor in the
/// derivation tree of the existence being disintegrated names a member of the
/// instruction's path list with the definition set of an admitted witness.
/// The probe carries what the check needs from the route: the existence and
/// its scope (the root of the walk) and the definition sets of the witnesses
/// the route admits, operator outputs excluded. A probe without labels checks
/// nothing.
///
/// The spans are owned by the caller (the existence being disintegrated and
/// its scope stay alive through the whole expansion; the labels point into
/// the configuration's definition-set strings, which never change during a
/// burst, I-137).
///
/// @invariant At most `kMaxLabels` distinct labels (asserted).
/// @see AdmissionMemory::witnessHistoryRefuses, DerivationTree::walk.
struct WitnessHistoryProbe {
    /// The label capacity: one per witness a route admits.
    static constexpr int32_t kMaxLabels = ExecutionParameters::MAX_INSTRUCTION_ELEMENTS;
    StrSpan start;              ///< The existence being disintegrated.
    StrSpan startScope;         ///< Its scope.
    StrSpan labels[kMaxLabels]; ///< Distinct definition sets of the checked witnesses.
    int32_t labelN = 0;         ///< Number of @ref labels.

    /// @brief Whether the probe checks anything.
    ///
    /// @details A route whose admitted witnesses are all operator outputs, or
    /// a block outside the gate, carries no label and refuses nothing.
    ///
    /// @return True iff at least one label is recorded.
    bool active() const { return labelN > 0; }

    /// @brief Whether a definition set is one of the checked witnesses'.
    ///
    /// @details Compared byte for byte; an empty span (an argument the typing
    /// did not reach) matches nothing.
    ///
    /// @param label A definition-set label.
    /// @return True iff @p label equals a recorded label.
    bool hasLabel(StrSpan label) const {
        if (label.len == 0) return false;
        for (int32_t i = 0; i < labelN; ++i)
            if (equalSpans(labels[i], label)) return true;
        return false;
    }

    /// @brief Record the definition set of one checked witness.
    ///
    /// @details A label already recorded is not added again; an empty label
    /// is a typing failure of the caller and asserts.
    ///
    /// @param label The witness's definition set (non-empty, stable span).
    /// @return Void.
    /// @invariant At most `kMaxLabels` labels.
    void addLabel(StrSpan label) {
        assert(label.len > 0 && "WitnessHistoryProbe::addLabel: a checked witness has a definition set");
        if (hasLabel(label)) return;
        assert(labelN < kMaxLabels && "WitnessHistoryProbe::addLabel: label count exceeds capacity");
        labels[labelN++] = label;
    }
};

/// @brief Walk the derivation tree of a statement through a block's history
///        (D-522).
///
/// @details
/// The derivation tree is the history buildStack reads: one derivation per
/// statement, followed through its premises to the end points. The walk reads
/// only the block's own origin map (which holds the history its ancestors
/// shipped by mail) and, for an equality whose history is still held by its
/// class, that class's equality history. It mints nothing and writes nothing.
class DerivationTree {
public:
    /// @brief One dependency of a chosen derivation, lifted to the scope that
    ///        holds its history.
    ///
    /// @details Either a run of the block's origin map (`bodyRun` non-zero) or
    /// one class of the scope's equivalence classes that holds the equality's
    /// history (`classBucket` non-zero). `key` is the packed origin key of the
    /// (expression, scope) pair in the block's origin interner.
    struct Node {
        int64_t key = 0;         ///< Packed (expression id, scope id) origin key.
        int32_t bodyRun = 0;     ///< Run id in `exprOriginMap`, or 0.
        int32_t classBucket = 0; ///< Run id in `equivalenceClassesMap`, or 0.
        int32_t classIndex = 0;  ///< Class index inside @ref classBucket.
    };

    /// @brief Whether a dependency names an integration goal.
    ///
    /// @details An integration goal key ends with `_integration_goal`. A goal
    /// is a demand, not a premise: it has no history by contract (I-44), and
    /// the walk neither checks nor follows it.
    ///
    /// @param expr The dependency's expression text.
    /// @return True iff @p expr ends with `_integration_goal`.
    static bool isIntegrationGoalKey(StrSpan expr) {
        static constexpr char kSuffix[] = "_integration_goal";
        constexpr int32_t n = static_cast<int32_t>(sizeof(kSuffix) - 1);
        return expr.len >= n && std::memcmp(expr.ptr + expr.len - n, kSuffix, n) == 0;
    }

    /// @brief Whether a dependency is a rule.
    ///
    /// @details A rule is an implication text `(>[...]...)`: the rule a firing
    /// applied, or the rules an implication composition joined. Only premises
    /// belong to a derivation tree, so the walk neither checks nor follows a
    /// rule.
    ///
    /// @param expr The dependency's expression text.
    /// @return True iff @p expr starts with `(>[`.
    static bool isRuleText(StrSpan expr) {
        return startsWithSpan(expr, "(>[", 3);
    }

    /// @brief The scopes a dependency may be read at, shallowest first, and the
    ///        shallowest one an OR branch permits.
    ///
    /// @details
    /// The byte twin of `ProofReconstructor::liftToShallowestOriginAncestor`'s
    /// candidate list: `main`, then for a scope of the form
    /// `main_boundary_...` each prefix that ends before a later `_boundary_`,
    /// then the scope itself. A candidate whose payload (with a goal prefix
    /// stripped, `ScopeSyntax::stripSubproofPrefixView`) starts with `orint_`
    /// or `ordis_` is an OR branch: a dependency inside it is never read above
    /// it, so the deepest such candidate is the lowest index the lift may use.
    ///
    /// @param scope   The cited scope.
    /// @param out     Caller array receiving spans into @p scope (and the
    ///                literal `main`).
    /// @param cap     Capacity of @p out.
    /// @param minLift Receives the index of the shallowest permitted candidate.
    /// @return The number of candidates written.
    /// @invariant Allocates nothing; asserts the capacity.
    static int32_t liftScopes(StrSpan scope, StrSpan* out, int32_t cap, int32_t& minLift) {
        static constexpr char kSep[] = "_boundary_";
        constexpr int32_t sepLen = static_cast<int32_t>(sizeof(kSep) - 1);
        assert(cap >= 1);
        int32_t n = 0;
        out[n++] = StrSpan("main", 4);
        if (scope.len > 4 && startsWithSpan(scope, "main", 4) && scope.len >= 4 + sepLen
            && std::memcmp(scope.ptr + 4, kSep, sepLen) == 0) {
            int32_t pos = 4 + sepLen;
            while (pos <= scope.len) {
                int32_t next = -1;
                for (int32_t i = pos; i + sepLen <= scope.len; ++i)
                    if (std::memcmp(scope.ptr + i, kSep, sepLen) == 0) { next = i; break; }
                assert(n < cap && "DerivationTree::liftScopes: scope depth exceeds capacity");
                if (next < 0) { out[n++] = scope; break; }
                out[n++] = StrSpan(scope.ptr, next);
                pos = next + sepLen;
            }
        }
        minLift = 0;
        for (int32_t i = 1; i < n; ++i) {
            const int32_t payloadStart = out[i - 1].len + sepLen;
            if (payloadStart >= out[i].len) continue;
            const StrSpan bare = ScopeSyntax::stripSubproofPrefixView(
                StrSpan(out[i].ptr + payloadStart, out[i].len - payloadStart));
            if (startsWithSpan(bare, "orint_", 6) || startsWithSpan(bare, "ordis_", 6)) minLift = i;
        }
        return n;
    }

    /// @brief Find where the history of one cited (expression, scope) pair is
    ///        held.
    ///
    /// @details
    /// Mirrors buildStack's lift. With @p exact (a dependency of a `zoom in`,
    /// `induction scope` or `int scope` row), or when the cited pair itself carries a
    /// `zoom in` row, the cited scope is read. Otherwise the shallowest
    /// permitted candidate of `liftScopes` whose key holds origin rows is
    /// chosen; when none does, the cited pair itself is read. An equality
    /// without origin rows at any of these scopes is looked up in the
    /// equivalence classes of the same scopes, shallowest first: the class at
    /// that scope that contains both sides and holds history lines under the
    /// equality's key.
    ///
    /// @param mb       Owning block (resident).
    /// @param expr     The cited expression (caller-stable bytes).
    /// @param scope    The cited scope (caller-stable bytes).
    /// @param exact    Whether the cited scope is kept.
    /// @param gArena   Container-tier scratch arena for record peeks.
    /// @param out      Receives the location when found.
    /// @return True iff a history location exists.
    /// @invariant Mints nothing.
    static bool locate(Memory& mb, StrSpan expr, StrSpan scope, bool exact,
                       ScratchArena& gArena, Node& out) {
        StrSpan candidates[ExecutionParameters::MAX_SCOPE_DEPTH + 2];
        int32_t minLift = 0;
        int32_t n = 0;
        const auto bodyAt = [&](StrSpan at) {
            int64_t key = 0;
            if (!lookupOriginKey(mb.originInterner, expr, at, key)) return false;
            const int32_t run = mb.exprOriginMap.lookup(key);
            if (run == 0 || mb.exprOriginMap.runLen(run) == 0) return false;
            out = Node{ key, run, 0, 0 };
            return true;
        };
        bool keepExact = exact;
        if (!keepExact && bodyAt(scope)) {
            for (int32_t j = 0; j < mb.exprOriginMap.runLen(out.bodyRun) && !keepExact; ++j) {
                int32_t len = 0;
                const char* bytes = mb.exprOriginMap.peekRecordBytes(out.bodyRun, j, len, gArena);
                keepExact = static_cast<OriginTag>(viewIdOriginBlob(bytes, len).tag) == OriginTag::zoomIn;
            }
        }
        if (keepExact) {
            candidates[0] = scope;
            n = 1;
        } else {
            n = liftScopes(scope, candidates, ExecutionParameters::MAX_SCOPE_DEPTH + 2, minLift);
        }
        for (int32_t i = minLift; i < n; ++i)
            if (bodyAt(candidates[i])) return true;
        if (!keepExact && bodyAt(scope)) return true;
        if (!startsWithSpan(expr, "(=[", 3)) return false;
        StrSpan sides[ExecutionParameters::MAX_ARITY];
        const int32_t sideN = getArgsSpans(expr, sides, ExecutionParameters::MAX_ARITY);
        assert(sideN == 2 && "DerivationTree::locate: an equality has two sides");
        const NameId left = mb.nameMap.lookup(sides[0]);
        const NameId right = mb.nameMap.lookup(sides[1]);
        if (left == 0 || right == 0) return false;
        for (int32_t i = minLift; i < n; ++i) {
            int64_t key = 0;
            if (!lookupOriginKey(mb.originInterner, expr, candidates[i], key)) continue;
            const NameId scopeId = mb.nameMap.lookup(candidates[i]);
            if (scopeId == 0) continue;
            const int32_t bucket = mb.equivalenceClassesMap.lookup(scopeId);
            if (bucket == 0) continue;
            for (int32_t c = 0; c < mb.equivalenceClassesMap.runLen(bucket); ++c) {
                int32_t len = 0;
                const char* bytes = mb.equivalenceClassesMap.peekRecordBytes(bucket, c, len, gArena);
                const EquivalenceClassView view{ bytes, len };
                bool hasLeft = false;
                bool hasRight = false;
                for (int32_t m = 0; m < view.memberCount(); ++m) {
                    hasLeft = hasLeft || view.memberId(m) == left;
                    hasRight = hasRight || view.memberId(m) == right;
                }
                if (!hasLeft || !hasRight || !view.hasOrigin(key)) continue;
                out = Node{ key, 0, bucket, c };
                return true;
            }
        }
        return false;
    }

    /// @brief Visit every ancestor in the derivation tree of a statement.
    ///
    /// @details
    /// Depth first from (@p expr, @p scope), which must hold origin rows. For
    /// each statement the walk takes one row the way buildStack takes its
    /// first candidate: rows that are not equality rows (`equality1`,
    /// `equality2`) before equality rows, each in insertion order, and a row
    /// is skipped when one of its followed dependencies lies on the current
    /// walk path (it would close a cycle). A `broadcast` or `externally
    /// provided theorem` row ends the walk at that statement. Of the chosen
    /// row the walk follows the premises: the first dependency of an
    /// `implication` row (the applied rule), every rule text and every
    /// integration goal are skipped. A `zoom in`, `induction scope` or `int scope` row
    /// keeps its dependencies' exact scopes; every other dependency is read
    /// where `locate` finds its history. A followed dependency without
    /// history violates the history contract (Rule 16) and asserts. Every
    /// statement is visited once, the root not at all.
    ///
    /// @p endPoint is asked first for each ancestor: a statement it accepts
    /// is an end point, neither visited nor followed (a forced product,
    /// `ForcedProductMatch`). @p visit then receives each other ancestor's
    /// expression and scope as spans that are valid only during the call,
    /// and returns true to stop the walk.
    ///
    /// @tparam Visit    Callable `bool(StrSpan expression, StrSpan scope)`.
    /// @tparam EndPoint Callable `bool(StrSpan expression, StrSpan scope)`.
    /// @param mb       Owning block (resident, single owner).
    /// @param expr     The root statement (caller-stable bytes).
    /// @param scope    The root's scope (caller-stable bytes).
    /// @param visit    The per-ancestor callback.
    /// @param endPoint The end-point predicate.
    /// @param visited  Optional out: the number of ancestors visited.
    /// @return True iff @p visit stopped the walk.
    /// @invariant Mints nothing and writes no block state; all scratch rides
    ///            the worker slot's arenas and is released on return.
    /// @see locate, AdmissionMemory::witnessHistoryRefuses,
    ///      ProofReconstructor::buildStack.
    template <typename Visit, typename EndPoint>
    static bool walk(Memory& mb, StrSpan expr, StrSpan scope, Visit&& visit,
                     EndPoint&& endPoint, int32_t* visited = nullptr) {
        const unsigned gSlot = WorkerContext::coreId() >= 0
            ? static_cast<unsigned>(WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
        const unsigned sSlot = WorkerContext::coreId() >= 0
            ? static_cast<unsigned>(WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
        ScratchArena& gArena = genScratchArenas().forSlot(gSlot);
        ScratchArena& strings = scratchArenas().forSlot(sSlot);
        ScratchScope gScope(gArena);
        ScratchScope sScope(strings);
        DirtyState dirty = DirtyState::Clean;
        ColdHashSet<PodKeyStore<int64_t>> seen(&gArena, &dirty);
        PagedVector<uint8_t> onPath(&gArena, &dirty);
        PagedVector<Node> deps(&gArena, &dirty);
        PagedVector<Frame> frames(&gArena, &dirty);
        int32_t visitedN = 0;

        Node root;
        const bool rooted = locate(mb, expr, scope, true, gArena, root);
        assert(rooted && root.bodyRun != 0
            && "DerivationTree::walk: the root statement holds origin rows");
        (void)rooted;
        const auto enter = [&](const Node& node) {
            const int32_t id = seen.mint(node.key);
            assert(id == onPath.size() + 1);
            onPath.push_back(1);
            const int32_t begin = deps.size();
            chooseRow(mb, node, seen, onPath, deps, gArena, strings);
            frames.push_back(Frame{ id, begin, deps.size(), begin });
        };
        enter(root);
        bool stopped = false;
        while (frames.size() > 0 && !stopped) {
            Frame top = frames[frames.size() - 1];
            if (top.next == top.end) {
                onPath.setAt(top.id - 1, 0);
                frames.truncate(frames.size() - 1);
                continue;
            }
            const Node dep = deps[top.next];
            ++top.next;
            frames.setAt(frames.size() - 1, top);
            if (seen.lookup(dep.key) != 0) continue;
            bool ends = false;
            {
                ScratchScope visitScope(strings);
                StrSpan depExpr, depScope;
                decodeOriginKeyCopy(dep.key, mb.originInterner, depExpr, depScope, strings);
                ends = endPoint(depExpr, depScope);
                if (!ends) {
                    ++visitedN;
                    stopped = visit(depExpr, depScope);
                }
            }
            if (ends) {
                // Seen but never entered: the seen id stays aligned with the
                // path marks, and no row of the end point is followed.
                seen.mint(dep.key);
                onPath.push_back(0);
                continue;
            }
            if (!stopped) enter(dep);
        }
        if (visited != nullptr) *visited = visitedN;
        return stopped;
    }

    /// @brief Visit every history row of one located statement in insertion
    ///        order.
    ///
    /// @details
    /// A body location (`bodyRun` non-zero) yields each row of the block's
    /// origin-map run; a class location yields each history line of that
    /// class held under the node's key. @p fn receives the row's tag, a
    /// reader of its packed dependency keys and the dependency count, and
    /// returns true to stop.
    ///
    /// @tparam Fn Callable `bool(uint8_t tag, DepAt depAt, int32_t depN)`,
    ///            where `depAt(int32_t)` returns a packed origin key.
    /// @param mb     Owning block (resident).
    /// @param node   The statement's history location (`locate`).
    /// @param gArena Container-tier scratch arena for record peeks.
    /// @param fn     The per-row callback.
    /// @return Nothing.
    /// @invariant Mints nothing and writes no block state.
    /// @see locate, chooseRow.
    template <typename Fn>
    static void forEachRow(Memory& mb, const Node& node, ScratchArena& gArena, Fn&& fn) {
        if (node.bodyRun != 0) {
            for (int32_t j = 0; j < mb.exprOriginMap.runLen(node.bodyRun); ++j) {
                int32_t len = 0;
                const char* bytes = mb.exprOriginMap.peekRecordBytes(node.bodyRun, j, len, gArena);
                const IdOriginBlobView view = viewIdOriginBlob(bytes, len);
                if (fn(view.tag, [&](int32_t d) { return idOriginDepAt(view, d); }, view.depN)) return;
            }
            return;
        }
        assert(node.classBucket != 0);
        int32_t len = 0;
        const char* bytes = mb.equivalenceClassesMap.peekRecordBytes(node.classBucket, node.classIndex,
            len, gArena);
        const EquivalenceClassView view{ bytes, len };
        bool stop = false;
        view.forEachOriginLine([&](int64_t key, OriginTag tag, const char* depBytes, int32_t depCount) {
            if (stop || key != node.key) return;
            stop = fn(static_cast<uint8_t>(tag), [&](int32_t d) {
                return EquivalenceClassView::rdI64(depBytes + static_cast<std::size_t>(8) * d);
            }, depCount);
        });
    }

private:
    /// @brief One statement on the walk stack: its seen id and the range of
    ///        its chosen row's followed dependencies in the dependency pool.
    struct Frame {
        int32_t id = 0;    ///< Seen id of the statement (1-based).
        int32_t begin = 0; ///< First dependency index.
        int32_t end = 0;   ///< One past the last dependency index.
        int32_t next = 0;  ///< Next dependency to follow.
    };

    /// @brief Choose the row of one statement and append its followed,
    ///        located dependencies to the pool.
    ///
    /// @details
    /// Candidate order and skips as documented at `walk`. A candidate whose
    /// followed dependency is the statement itself or lies on the walk path
    /// is skipped; when every candidate is skipped the statement is an end
    /// point and nothing is appended.
    ///
    /// @param mb      Owning block.
    /// @param node    The statement's history location.
    /// @param seen    The walk's seen set (ids index @p onPath).
    /// @param onPath  Per seen id, whether the statement is on the walk path.
    /// @param deps    The dependency pool receiving the chosen dependencies.
    /// @param gArena  Container-tier scratch arena (record peeks).
    /// @param strings String-tier scratch arena (decoded dependency texts).
    /// @return Void.
    /// @invariant Mints nothing; a followed dependency without history asserts.
    static void chooseRow(Memory& mb, const Node& node, const ColdHashSet<PodKeyStore<int64_t>>& seen,
                          const PagedVector<uint8_t>& onPath, PagedVector<Node>& deps,
                          ScratchArena& gArena, ScratchArena& strings) {
        const auto isEquality = [](uint8_t tag) {
            return static_cast<OriginTag>(tag) == OriginTag::equality1
                || static_cast<OriginTag>(tag) == OriginTag::equality2;
        };
        for (int pass = 0; pass < 2; ++pass) {
            bool chosen = false;
            forEachRow(mb, node, gArena, [&](uint8_t tag, const auto& depAt, int32_t depN) {
                if (isEquality(tag) != (pass == 1)) return false;
                const OriginTag t = static_cast<OriginTag>(tag);
                if (t == OriginTag::broadcast || t == OriginTag::externallyProvidedTheorem) {
                    chosen = true;
                    return true;
                }
                const bool exact = t == OriginTag::zoomIn || t == OriginTag::inductionScope
                    || t == OriginTag::intScope;
                assert(depN <= ExecutionParameters::kMaxOriginDeps);
                Node located[ExecutionParameters::kMaxOriginDeps];
                int32_t locatedN = 0;
                bool cycle = false;
                {
                    ScratchScope rowScope(strings);
                    for (int32_t d = (t == OriginTag::implication ? 1 : 0); d < depN && !cycle; ++d) {
                        StrSpan depExpr, depScope;
                        decodeOriginKeyCopy(depAt(d), mb.originInterner, depExpr, depScope, strings);
                        if (isRuleText(depExpr) || isIntegrationGoalKey(depExpr)) continue;
                        Node at;
                        const bool found = locate(mb, depExpr, depScope, exact, gArena, at);
                        assert(found && "DerivationTree: a followed dependency holds no history (Rule 16)");
                        (void)found;
                        const int32_t id = seen.lookup(at.key);
                        cycle = at.key == node.key || (id != 0 && onPath[id - 1] != 0);
                        located[locatedN++] = at;
                    }
                }
                if (cycle) return false;
                for (int32_t k = 0; k < locatedN; ++k) deps.push_back(located[k]);
                chosen = true;
                return true;
            });
            if (chosen) return;
        }
    }
};

} // namespace gl
