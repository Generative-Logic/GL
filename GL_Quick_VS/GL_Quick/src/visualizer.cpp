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

#include "prover.hpp"
#include "memory_infra/lb_deload.hpp"
#include <numeric>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <fstream>

namespace gl {

// =====================================================================
// D-51 (option 1 + backtracking) — path-stack cycle filter for buildStack
// origin choice.
//
// With max_origin_per_expr raised to 30 (matching compressor mode), each
// expression typically carries multiple candidate origins. buildStack:
//   1. Sorts origins per D-49-style preference: non-equality tags first
//      (preserving insertion order within group), then equality1/equality2
//      tags. Within each group, insertion order survives — matches the
//      legacy cap=1 + D-49 surviving-origin choice for the common case.
//   2. Walks candidates in that order; skips any whose deps include an
//      expression already on the current proof-tree path (would form a
//      cycle with the recursion stack — Knuth (1977) AND-OR graph
//      acyclic-derivation extraction).
//   3. Backtracks: if the recursive walk fails for a candidate's deps,
//      rolls back stack/covered/path snapshots and tries the next
//      candidate. Guarantees a valid acyclic tree if one exists.
//
// Returns true on success. False signals to the recursive caller that no
// acyclic origin exists for this node given the current path; caller
// retries with its own next candidate. Top-level callers (directStack,
// checkZeroStack, ...) ignore the return.
// =====================================================================
static thread_local std::set<ExpressionWithValidity> g_buildStackPath;

// Insertion journal for the walk's `covered` set — the exact-rollback
// replacement for the per-candidate set copy.
//
// A candidate that fails must leave `covered` exactly as it found it, because
// its emitted rows are dropped by `stack.resize` and a node left marked
// covered with no row would lose its row from the chapter. `covered` is
// insert-only inside the walk (no site erases from it), so the set at a
// candidate's entry is recovered by erasing exactly the keys inserted since
// that candidate's mark — a suffix truncation, which is always well defined
// because marks nest with the recursion.
//
// The copy this replaces was 38% of the whole chapter export: 4.5 M
// per-candidate deep copies of a set holding up to 311 string pairs, of which
// 97% were discarded unused because the candidate succeeded.
static thread_local std::vector<ExpressionWithValidity> g_coveredJournal;

// No-good memo for the walker's backtracking search.
//
// A frame fails for exactly three reasons: every candidate was cyclic (a
// dependency sits on `g_buildStackPath`), inadmissible (the cross-chapter
// usage graph), or had a failing subtree. Only the cyclic reason is a pure
// function of the path, so only a failure reached without touching the other
// two is reproducible from the path alone. Two things therefore make a frame
// CONTEXT-DEPENDENT and bar it from the memo: a candidate skipped as
// inadmissible (the usage graph gains edges after every chapter, and
// `exportCurrentChapterTheorem` moves with the chapter). Dependence on
// `covered` is not a bar but a recorded condition - see NoGoodEntry. Every
// other failing frame caches its conflict set: the on-path nodes that actually
// blocked it, unioned with the conflict sets of the children that failed under
// it. Re-entry with all of those still on the path, and with no consulted
// dependency newly covered, returns false immediately instead of re-walking
// the subtree.
//
// Without this the walker re-derives every failing subtree under each of the
// parent candidates that reach it, which is exponential in the number of
// dead ends.
struct WalkNode {
    const Memory* lb;
    ExpressionWithValidity ev;
    bool operator<(const WalkNode& o) const {
        if (lb != o.lb) return lb < o.lb;
        return ev < o.ev;
    }
};
/// @brief One cached failure: why it happened and what it depended on.
///
/// @details
/// `conflict` holds the on-path nodes that blocked the frame. `consulted`
/// holds every dependency the failed exploration tested against `covered`,
/// including those its children tested; `coveredAtFailure` is the subset of
/// `consulted` that was already covered when the frame gave up. Both
/// conditions below are one-directional, which is what makes the entry
/// reusable rather than exact:
///
///   * extra path members only remove candidates (a non-cyclic candidate can
///     become cyclic, never the reverse), so failure survives a longer path
///     as long as every recorded blocker is still on it;
///   * extra covered members only remove recursions (a failing child becomes
///     a skip), which could turn failure into success - so no consulted node
///     may be covered now that was not covered then.
///
/// With both satisfied the exploration replays identically, because every
/// `covered.insert` verdict and every cyclicity verdict it read is unchanged.
struct NoGoodEntry {
    std::set<ExpressionWithValidity> conflict;
    std::set<ExpressionWithValidity> consulted;
    std::set<ExpressionWithValidity> coveredAtFailure;
};
static thread_local std::map<WalkNode, NoGoodEntry> g_noGood;

// Conflict set, consulted set and context-dependence of the frame that most
// recently returned false - read by the parent immediately after the child
// returns.
static thread_local std::set<ExpressionWithValidity> g_failConflict;
static thread_local std::set<ExpressionWithValidity> g_failConsulted;
static thread_local bool g_failContextDependent = false;

// A frame whose consulted set outgrows this is not cached: the probe walks
// the set on every re-entry, so an unbounded footprint would cost more than
// the re-walk it saves.
static constexpr std::size_t kNoGoodConsultedCap = 2048;

void clearBuildStackPath() {
    g_buildStackPath.clear();
    g_coveredJournal.clear();
    g_noGood.clear();
    g_failConflict.clear();
    g_failConsulted.clear();
    g_failContextDependent = false;
}

/// @brief Drop every edge — fresh graph for a new export run.
///
/// @details
/// Called once at `generateRawProofGraph` entry (next to
/// `clearBuildStackPath`) so consecutive exports (main batch after
/// incubator batch in one process) never see stale cross-run edges.
void TheoremUsageGraph::clear() { edges.clear(); }

/// @brief Commit one written chapter's citation edges into the graph.
///
/// @details
/// Scans the emitted rows of the chapter of `theorem` and records
/// `theorem -> cited` for every citation channel: a `theorem` row cites its
/// own row expression; `reformulated from` and `incubator back
/// reformulation` rows cite their source at `row[3]`; an `or theorem` row
/// cites its two source implications at `row[3]` and `row[5]`; an
/// `or elimination` row cites its two guard variants at `row[3]`/`row[5]`
/// and its licensing or theorem at `row[7]`. Row arity is
/// asserted per channel — a malformed row is a producer bug to stop on,
/// never to skip. Induction triads call this once per chapter file; the
/// three files' edges union under the one theorem node.
///
/// @param theorem    The theorem whose chapter was just written
///                   (global-theorem-list string form).
/// @param stackRows  The chapter's emitted rows (cells per row: expression,
///                   namespace, tag, then (dependency, namespace) pairs).
void TheoremUsageGraph::addEdgesFromStack(
    const std::string& theorem,
    const std::vector<std::vector<std::string>>& stackRows)
{
    std::set<std::string>& out = edges[theorem];
    for (const std::vector<std::string>& row : stackRows) {
        assert(row.size() >= 3
            && "TheoremUsageGraph: chapter row must carry expression, namespace, tag");
        const std::string& tag = row[2];
        if (tag == "theorem") {
            out.insert(row[0]);
        } else if (tag == "reformulated from"
                || tag == "incubator back reformulation") {
            assert(row.size() >= 5
                && "TheoremUsageGraph: source citation at row[3] required");
            out.insert(row[3]);
        } else if (tag == "or theorem") {
            assert(row.size() >= 7
                && "TheoremUsageGraph: or-theorem sources at row[3]/row[5] required");
            out.insert(row[3]);
            out.insert(row[5]);
        } else if (tag == "or elimination") {
            assert(row.size() >= 9
                && "TheoremUsageGraph: or-elimination sources at row[3]/row[5]/row[7] required");
            out.insert(row[3]);
            out.insert(row[5]);
            out.insert(row[7]);
        }
    }
}

/// @brief Reachability probe: does `from` reach `to` over committed edges?
///
/// @details
/// Iterative depth-first search over `edges`; `from == to` returns true
/// without a walk (a self-citation is a length-1 cycle). Theorems without
/// committed out-edges are leaves. Used by `buildStack` as the
/// cross-chapter admissibility probe: a `theorem` origin citing U is
/// rejected while building the chapter of T iff `reaches(U, T)`.
///
/// @param from  Candidate cited theorem (start node).
/// @param to    Theorem whose chapter is currently being built.
/// @return True iff `to` is reachable from `from` (including `from == to`).
bool TheoremUsageGraph::reaches(const std::string& from,
                                const std::string& to) const
{
    if (from == to) return true;
    std::set<std::string> visited;
    std::vector<const std::string*> frontier;
    visited.insert(from);
    frontier.push_back(&from);
    while (!frontier.empty()) {
        const std::string* cur = frontier.back();
        frontier.pop_back();
        const auto it = edges.find(*cur);
        if (it == edges.end()) continue;
        for (const std::string& next : it->second) {
            if (next == to) return true;
            if (visited.insert(next).second) frontier.push_back(&next);
        }
    }
    return false;
}

// Order origins by (D-49-style preference, insertion index): non-equality
// tags first preserving insertion order, then equality1/equality2 tags
// preserving insertion order. This matches the cap=1 + D-49 surviving-
// origin choice for chapters whose proof structure is acyclic.
static std::vector<size_t> sortedOriginIndicesD49(
    const std::vector<std::pair<std::string, std::vector<ExpressionWithValidity>>>& origins)
{
    std::vector<size_t> result;
    result.reserve(origins.size());
    auto isEq = [&](size_t i) {
        return origins[i].first == "equality1" || origins[i].first == "equality2";
    };
    for (size_t i = 0; i < origins.size(); ++i) if (!isEq(i)) result.push_back(i);
    for (size_t i = 0; i < origins.size(); ++i) if (isEq(i)) result.push_back(i);
    return result;
}

// Lift (expr, validity) to the closest-to-"main" ancestor of validity whose
// (expr, ancestor) key has an origin in this LB's exprOriginMap. Per I-2 and
// NameMap::encodePush, every non-root validity is parent + "_boundary_" +
// payload, so the ancestor chain is recoverable by splitting on "_boundary_".
// Returns v unchanged if no ancestor (including v itself) has an origin —
// caller handles the miss (assertion / dump).
//
// OR-branch barrier: lifting may NOT cross "_boundary_orint_" or
// "_boundary_ordis_" delimiters. Those scopes are conditional on a disjunct
// hypothesis (or<N>-integration / -disintegration); chapter rows must preserve
// branch-distinct namespaces for the OR-family verifier checkers (or
// convergence, or branch proven, or branch assumption, or disintegration) to
// recognise the convergence/branch-proof pattern. Without the barrier, both
// branches' deps would be lifted to the parent boundary where the post-
// convergence origin lives, collapsing them to identical pairs and breaking
// `check_or_convergence`'s branch-distinctness expectation. The deepest
// orint_/ordis_ ancestor sets the shallowest allowed lift target.
static ExpressionWithValidity liftToShallowestOriginAncestor(
        const Memory& mb,
        const ExpressionWithValidity& v) {
    const std::string& s = v.validityName;
    static const std::string sep = "_boundary_";
    std::vector<std::string> ancestors;
    ancestors.reserve(8);
    ancestors.push_back("main");
    if (s.size() > 4
        && s.compare(0, 4, "main") == 0
        && s.size() >= 4 + sep.size()
        && s.compare(4, sep.size(), sep) == 0) {
        std::size_t pos = 4 + sep.size();
        while (pos <= s.size()) {
            std::size_t next = s.find(sep, pos);
            if (next == std::string::npos) {
                ancestors.push_back(s);
                break;
            }
            ancestors.push_back(s.substr(0, next));
            pos = next + sep.size();
        }
    }
    std::size_t minLift = 0;
    static const std::string kOrint = "orint_";
    static const std::string kOrdis = "ordis_";
    for (std::size_t i = 1; i < ancestors.size(); ++i) {
        const std::size_t payloadStart = ancestors[i - 1].size() + sep.size();
        if (payloadStart >= ancestors[i].size()) continue;
        const std::string& child = ancestors[i];
        const std::size_t payloadLen = child.size() - payloadStart;
        // A goal-carrying payload (`<goal>_subproof_orint_...`) is still an
        // OR-branch barrier — classify by the bare half via the single
        // payload-parsing choke point (I-170).
        const StrSpan bare = ExpressionAnalyzer::stripSubproofPrefixView(
            StrSpan(child.data() + payloadStart,
                    static_cast<int32_t>(payloadLen)));
        if ((bare.len >= static_cast<int32_t>(kOrint.size())
             && std::memcmp(bare.ptr, kOrint.data(), kOrint.size()) == 0)
            || (bare.len >= static_cast<int32_t>(kOrdis.size())
                && std::memcmp(bare.ptr, kOrdis.data(), kOrdis.size()) == 0)) {
            minLift = i;
        }
    }
    for (std::size_t i = minLift; i < ancestors.size(); ++i) {
        int64_t pk = 0;
        if (!lookupOriginKey(mb.originInterner, v.original, ancestors[i], pk)) {
            continue;
        }
        const int32_t oid = mb.exprOriginMap.lookup(pk);
        if (oid != 0 && mb.exprOriginMap.runLen(oid) > 0) {
            return ExpressionWithValidity(v.original, ancestors[i]);
        }
    }
    return v;
}

/// @brief Probe whether a `__contradiction__` twin LB can actually resolve
///        `proved` — i.e. holds at least one origin record for it — before
///        buildStack's D-51 fallback switches in.
///
/// @details
/// Mirrors buildStack's entry-side origin check exactly: read-only reload
/// (D-158), lift within the twin, non-minting key probe, non-empty run.
/// A twin that primed but never converged (no discharge record) fails the
/// probe; the fallback then does not apply and the walk unwinds as an
/// ordinary candidate failure so backtracking can reach acyclic records one
/// level up — instead of dying on the twin's entry no-origin assert.
///
/// @param contraLB The `__contradiction__` twin candidate.
/// @param proved   The expression/validity the walk must justify.
/// @return true if the twin holds at least one origin record for the lifted
///         form of `proved`; false otherwise.
/// @invariant Non-minting (I-91).
/// @see buildStack — both D-51 fallback sites are the only callers.
bool ExpressionAnalyzer::contradictionLbHoldsRecord(Memory& contraLB,
    const ExpressionWithValidity& proved) {
    contraLB.ensureLoadedForRead(lbdeload::kDeloadDirectory);
    const ExpressionWithValidity lifted =
        liftToShallowestOriginAncestor(contraLB, proved);
    int64_t pk = 0;
    if (!lookupOriginKey(contraLB.originInterner, lifted.original,
                         lifted.validityName, pk)) {
        return false;
    }
    const int32_t oid = contraLB.exprOriginMap.lookup(pk);
    return oid != 0 && contraLB.exprOriginMap.runLen(oid) > 0;
}

/// @brief Emit one chapter's derivation rows for `provedIn` — see the
///        declaration in `prover.hpp` for the full contract.
///
/// @details
/// Definition of the D-51 chapter walker. The `rowsSurviveOnFailure` parameter
/// is the only non-obvious one: it says whether rows this call pushes onto
/// `stack` can still reach the chapter file if the call returns false. A
/// recursion from a parent's candidate loop is the one case where they cannot —
/// that parent truncates the stack to its own mark the instant it receives
/// false — so the last-resort fallback block is skipped there rather than
/// emitting a row and walking a whole dependency subtree that the truncation is
/// guaranteed to discard.
///
/// @param memoryBlock          The LB whose origin records justify `provedIn`.
/// @param provedIn             The expression / validity to justify.
/// @param stack                Accumulating chapter rows.
/// @param covered              Nodes already given a row this chapter.
/// @param rowsSurviveOnFailure See the details above.
/// @return True when an acyclic derivation tree was emitted.
/// @see ExpressionAnalyzer::buildStack (declaration)
bool ExpressionAnalyzer::buildStack(Memory& memoryBlock,
    const ExpressionWithValidity& provedIn,
    std::vector<std::vector<std::string>>& stack,
    std::set<ExpressionWithValidity>& covered,
    bool rowsSurviveOnFailure) {
    // Hang tripwire: exponential candidate exploration can make a chapter
    // walk unboundedly long; cap calls at 5M so the process exits with a
    // clear signal instead of hanging indefinitely.
    static std::atomic<std::size_t> s_buildStackCalls{0};
    const std::size_t myCall = ++s_buildStackCalls;
    if (myCall > 5000000) {
        std::cerr << "[buildStack] call cap 5M reached — aborting" << std::endl;
        std::abort();
    }

    // Post-prove READ reload (D-158): the chapter walk
    // reads this LB's origin history through its cold string tables; a
    // pressure-drained LB — discharged ones included — comes back here.
    // Defined no-op when resident. This is the sanctioned export-side
    // entry of the reload touch-point list.
    memoryBlock.ensureLoadedForRead(lbdeload::kDeloadDirectory);

    // TRIPWIRE: sentinel validity used by disintegrateExprHypothetically.
    // Hypothetical disintegration products must never reach buildStack —
    // they are throw-away structural probes and the lambda no-track path
    // already suppresses their origin writes. Check the INCOMING expression
    // before any lifting; the sentinel is structural, not a validity-stack
    // ancestor of anything legitimate.
    static const std::string kHypoDisintMarker = "_boundary_hypothetical_disintegration";
    if (provedIn.validityName.find(kHypoDisintMarker) != std::string::npos) {
        std::cerr << "[buildStack] TRIPWIRE: hypothetical-disintegration sentinel "
                  << "reached buildStack: " << provedIn.original
                  << " (v=" << provedIn.validityName << ")\n";
        assert(false && "buildStack: hypothetical disintegration sentinel leaked into proof graph");
    }

    // Lift to the closest-to-"main" ancestor of provedIn.validityName whose
    // (expr, ancestor) key has an origin in this LB's exprOriginMap. From
    // here on, `proved` is the lifted form — chapter rows are emitted at
    // the lifted scope (truthful "this is where the derivation lives"), the
    // path-cycle filter uses the lifted form, and recursion lifts each dep
    // so duplicates collapse to a single (expr, lifted_v) row per chapter.
    const ExpressionWithValidity proved = liftToShallowestOriginAncestor(memoryBlock, provedIn);

    int64_t pkProved = 0;
    bool hasOrigins = false;
    if (lookupOriginKey(memoryBlock.originInterner, proved.original,
                        proved.validityName, pkProved)) {
        const int32_t oid = memoryBlock.exprOriginMap.lookup(pkProved);
        hasOrigins = oid != 0 && memoryBlock.exprOriginMap.runLen(oid) > 0;
    }
    if (!hasOrigins) {
        // _integration_goal expressions are synthetic markers — no origin expected.
        if (proved.original.find("_integration_goal") != std::string::npos) {
            return true;
        }
        // D-51: contradiction-LB fallback. When `proved` has no direct origin
        // in this LB (or any of its ancestors with an origin entry), search
        // the LB chain for the contradiction LB that ASSUMED its negation —
        // "__contradiction__" + negate(proved) — and resolve locally there.
        // Either polarity: a negated head proved by a verbatim positive-seed
        // LB, or a positive head proved by a complement LB
        // (I-165).
        if (!proved.original.empty()) {
            const std::string negated = (proved.original[0] == '!')
                ? proved.original.substr(1)
                : "!" + proved.original;
            std::string contraKey = "__contradiction__" + negated;
            Memory* contraLB = nullptr;
            for (Memory* anc = &memoryBlock; anc != nullptr; anc = anc->parentMemory) {
                Memory* sc = simpleMapStore.findChild(anc, contraKey);
                if (sc != nullptr) {
                    contraLB = sc;
                    break;
                }
            }
            // Resolution guard: the switch is only a resolution when the
            // twin actually holds a record for `proved` (a primed twin that
            // never converged holds none — switching in would end at the
            // twin's own no-origin assert, not at a derivation). Covers the
            // self case (the twin never records its own assumed head at the
            // probed scope) and the never-converged case alike; without a
            // record, fall through to the loud no-origin assert here.
            if (contraLB != nullptr && contraLB != &memoryBlock
                && contradictionLbHoldsRecord(*contraLB, proved)) {
                // The twin's rows share this frame's fate: its verdict becomes
                // this frame's verdict, so whoever would discard ours discards
                // its too.
                return buildStack(*contraLB, proved, stack, covered,
                                  rowsSurviveOnFailure);
            }
        }
        std::cerr << "[buildStack] no origin for: " << proved.original
                  << " | validity=" << proved.validityName
                  << " | exprKey=" << memoryBlock.exprKey() << "\n";
        std::cerr << "[buildStack] theorem: "
                  << exportCurrentChapterTheorem << "\n";
        std::cerr << "[buildStack] LB chain:";
        for (const Memory* p = &memoryBlock; p != nullptr;
             p = p->parentMemory) {
            std::cerr << " <- " << p->exprKey();
        }
        std::cerr << "\n";
        for (auto rowIt = stack.rbegin(); rowIt != stack.rend(); ++rowIt) {
            const std::vector<std::string>& row = *rowIt;
            bool ownsMissingDependency = false;
            for (std::size_t i = 3; i + 1 < row.size(); i += 2) {
                if (row[i] == proved.original
                    && row[i + 1] == proved.validityName) {
                    ownsMissingDependency = true;
                    break;
                }
            }
            if (!ownsMissingDependency) continue;
            std::cerr << "[buildStack] referring row:";
            for (const std::string& cell : row) {
                std::cerr << " | " << cell;
            }
            std::cerr << "\n";
            break;
        }
        assert(false && "buildStack: no origin found");
    }

    // Path-stack invariant fix: g_buildStackPath has no refcount.
    // If an OUTER scope already inserted `proved` (because this is a
    // recursive re-entry on a node further up the proof-tree path),
    // the inner scope MUST NOT erase on exit — that would wipe the
    // outer scope's entry too and break the cycle filter for the
    // outer's subsequent candidates. Track whether THIS invocation
    // is the inserter and only erase if so.
    const bool insertedHere = g_buildStackPath.insert(proved).second;

    // No-good probe. `proved` is already on the path, so a conflict set that
    // names it stays satisfied on re-entry - exactly the intended semantics.
    const WalkNode myNode{ &memoryBlock, proved };
    std::set<ExpressionWithValidity> myConflict;
    std::set<ExpressionWithValidity> myConsulted;
    bool myContextDependent = false;
    if (!rowsSurviveOnFailure) {
        const auto noGoodIt = g_noGood.find(myNode);
        if (noGoodIt != g_noGood.end()) {
            const NoGoodEntry& entry = noGoodIt->second;
            bool stillBlocked = true;
            for (const ExpressionWithValidity& blocker : entry.conflict) {
                if (g_buildStackPath.count(blocker) == 0) {
                    stillBlocked = false;
                    break;
                }
            }
            if (stillBlocked) {
                for (const ExpressionWithValidity& node : entry.consulted) {
                    if (covered.count(node) != 0
                        && entry.coveredAtFailure.count(node) == 0) {
                        stillBlocked = false;
                        break;
                    }
                }
            }
            if (stillBlocked) {
                g_failConflict = entry.conflict;
                g_failConsulted = entry.consulted;
                g_failContextDependent = false;
                if (insertedHere) g_buildStackPath.erase(proved);
                return false;
            }
        }
    }

    // Decoded owned copy — every downstream consumer (D-49 candidate
    // sort, cycle filter, emitRow, the last-resort fallback) keeps its
    // historical string form.
    std::vector<std::pair<std::string, std::vector<ExpressionWithValidity>>> origins;
    {
        const std::vector<IdOrigin> idOrigins =
            memoryBlock.exprOriginMap.recordsAt(
                memoryBlock.exprOriginMap.lookup(pkProved));
        origins.reserve(idOrigins.size());
        for (const IdOrigin& o : idOrigins) {
            origins.push_back(decodeOrigin(o, memoryBlock.originInterner));
        }
    }
    auto candidateOrder = sortedOriginIndicesD49(origins);

    auto emitRow = [&](const std::pair<std::string, std::vector<ExpressionWithValidity>>& origin,
                       const std::vector<ExpressionWithValidity>& liftedDeps) {
        // `hypothesis` provenance is internal-only: hypothesis-scope
        // constituents steer proof direction but are not part of any proof,
        // so their terminal origin line must never reach chapter emission
        // (and hence the verifier never sees the tag).
        assert(origin.first != "hypothesis"
            && "buildStack: internal-only hypothesis origin reached chapter emission");
        std::vector<std::string> row;
        row.reserve(3 + liftedDeps.size() * 2);
        row.push_back(proved.original);
        row.push_back(proved.validityName);
        row.push_back(origin.first);
        for (const auto& d : liftedDeps) {
            row.push_back(d.original);
            row.push_back(d.validityName);
        }
        stack.push_back(row);
    };

    // Candidate loop. Each candidate's deps are lifted before cycle-filter,
    // emission, and recursion so chapter cells, the path-cycle filter, and
    // the `covered` dedup all key on the same lifted form.
    for (size_t idx : candidateOrder) {
        const auto& origin = origins[idx];

        std::vector<ExpressionWithValidity> liftedDeps;
        liftedDeps.reserve(origin.second.size());
        for (const auto& d : origin.second) {
            liftedDeps.push_back(liftToShallowestOriginAncestor(memoryBlock, d));
        }

        bool cyclic = false;
        for (const auto& d : liftedDeps) {
            if (g_buildStackPath.count(d)) {
                cyclic = true;
                myConflict.insert(d);
                break;
            }
        }
        if (cyclic) continue;

        // Cross-chapter admissibility (D-276):
        // a `theorem` leaf cites `proved.original` itself as a proven
        // theorem U — a foundation inside THIS chapter, so the path filter
        // above cannot see a cycle routed through U's own chapter. The
        // candidate is admissible only if U does not already reach the
        // current chapter's theorem T in the committed usage graph
        // (U == T included). Inadmissible -> skip, exactly like a
        // path-cyclic candidate: next origin, then the contradiction-LB
        // fallback, then the degraded front() emit the verifier flags.
        // Empty exportCurrentChapterTheorem disables the check (findEnds /
        // direct unit-test callers keep the legacy chapter-local behavior).
        if (origin.first == "theorem"
            && !exportCurrentChapterTheorem.empty()
            && exportTheoremUsage.reaches(proved.original,
                                          exportCurrentChapterTheorem)) {
            // The usage graph and the current chapter both move between
            // chapters, so a verdict that depended on this skip is not
            // reproducible from the path alone.
            myContextDependent = true;
            continue;
        }

        // Early-return tags: theorem proved in previous batch / external — no row, success.
        if (origin.first == "broadcast" || origin.first == "externally provided theorem") {
            if (insertedHere) g_buildStackPath.erase(proved);
            return true;
        }

        const size_t stackSnap = stack.size();
        const std::size_t coveredMark = g_coveredJournal.size();

        emitRow(origin, liftedDeps);

        bool subtreeOk = true;
        for (const auto& ingredient : liftedDeps) {
            // Journal only a genuine insertion: a key that was already present
            // predates this candidate's mark and must survive its rollback.
            // Every dependency tested against `covered` joins the
            // footprint, whichever way the test went.
            myConsulted.insert(ingredient);
            if (covered.insert(ingredient).second) {
                g_coveredJournal.push_back(ingredient);
                // `false` for the child: the rollback below truncates `stack`
                // to `stackSnap`, taken before this candidate emitted, so every
                // row the child pushed goes with it.
                if (!buildStack(memoryBlock, ingredient, stack, covered,
                                /*rowsSurviveOnFailure=*/false)) {
                    myConflict.insert(g_failConflict.begin(), g_failConflict.end());
                    myConsulted.insert(g_failConsulted.begin(), g_failConsulted.end());
                    if (g_failContextDependent) myContextDependent = true;
                    subtreeOk = false; break;
                }
            }
        }

        if (subtreeOk) {
            if (insertedHere) g_buildStackPath.erase(proved);
            return true;
        }

        stack.resize(stackSnap);
        for (std::size_t k = g_coveredJournal.size(); k > coveredMark; --k) {
            covered.erase(g_coveredJournal[k - 1]);
        }
        g_coveredJournal.resize(coveredMark);
    }

    // D-51: no acyclic direct origin worked. Try the contradiction-LB
    // fallback before falling back to the degraded front()-emit. Walk the
    // LB chain for "__contradiction__" + negate(proved) — either head
    // polarity (I-165); if found,
    // switch in and resolve there.
    if (!proved.original.empty()) {
        const std::string negated = (proved.original[0] == '!')
            ? proved.original.substr(1)
            : "!" + proved.original;
        std::string contraKey = "__contradiction__" + negated;
        Memory* contraLB = nullptr;
        for (Memory* anc = &memoryBlock; anc != nullptr; anc = anc->parentMemory) {
            Memory* sc = simpleMapStore.findChild(anc, contraKey);
            if (sc != nullptr) {
                contraLB = sc;
                break;
            }
        }
        // Resolution guard (mirror of the entry-side site): switch only
        // when the twin actually holds a record for `proved`. A primed but
        // never-converged twin resolves nothing — without the guard the
        // switch ends at the twin's entry no-origin assert even while THIS
        // frame's caller still has acyclic candidates to backtrack to.
        if (contraLB != nullptr && contraLB != &memoryBlock
            && contradictionLbHoldsRecord(*contraLB, proved)) {
            if (insertedHere) g_buildStackPath.erase(proved);
            // Inherits this frame's fate, exactly as the entry-side site.
            const bool twinVerdict = buildStack(*contraLB, proved, stack, covered,
                                                rowsSurviveOnFailure);
            if (!twinVerdict) {
                // The twin conflict plus this frame own blockers; not cached,
                // because the verdict was produced on another LB.
                g_failConflict.insert(myConflict.begin(), myConflict.end());
                if (myContextDependent) g_failContextDependent = true;
            }
            return twinVerdict;
        }
    }

    // Last-resort fallback. A frame that returns false into a parent's
    // candidate loop has every row it emitted removed by that parent's
    // `stack.resize`, and every `covered` insertion undone by its journal
    // rollback — so emitting the front origin here and walking its whole
    // dependency subtree would produce nothing but work for the truncation to
    // throw away. Skip it: the verdict is false either way, and `covered` /
    // `g_buildStackPath` are left exactly as an executed-then-rolled-back
    // fallback would have left them.
    //
    // Rows DO survive for the top-level call (whose caller ignores the return),
    // for anything below it reached only by the fallback's own dependency walk,
    // and for a contradiction twin that inherited a surviving frame's fate —
    // which is precisely the degraded chapter shape this fallback exists to
    // produce.
    if (!rowsSurviveOnFailure) {
        if (!myContextDependent && myConsulted.size() <= kNoGoodConsultedCap) {
            // Keep the smallest conflict seen for this node - the smaller the
            // set, the more re-entries it blocks. `covered` is back to this
            // frame's entry state here (every candidate rolled back), which is
            // exactly the snapshot the replay argument needs.
            const auto noGoodIt = g_noGood.find(myNode);
            if (noGoodIt == g_noGood.end()
                || myConflict.size() < noGoodIt->second.conflict.size()) {
                NoGoodEntry entry;
                entry.conflict = myConflict;
                entry.consulted = myConsulted;
                for (const ExpressionWithValidity& node : myConsulted) {
                    if (covered.count(node) != 0) entry.coveredAtFailure.insert(node);
                }
                g_noGood[myNode] = entry;
            }
        }
        g_failConflict = myConflict;
        g_failConsulted = myConsulted;
        g_failContextDependent = myContextDependent;
        if (insertedHere) g_buildStackPath.erase(proved);
        return false;
    }

    // The surviving case: emit front() and recurse on its deps fully (matches
    // pre-D-51 chapter shape — chapter has rows even if cyclic, verifier
    // flags). Deps are lifted like the per-candidate loop.
    if (!origins.empty()) {
        const auto& fb = origins.front();
        if (fb.first == "broadcast" || fb.first == "externally provided theorem") {
            g_failConflict = myConflict;
            g_failConsulted = myConsulted;
            g_failContextDependent = true;
            if (insertedHere) g_buildStackPath.erase(proved);
            return false;
        }
        std::vector<ExpressionWithValidity> liftedDeps;
        liftedDeps.reserve(fb.second.size());
        for (const auto& d : fb.second) {
            liftedDeps.push_back(liftToShallowestOriginAncestor(memoryBlock, d));
        }
        emitRow(fb, liftedDeps);
        for (const auto& d : liftedDeps) {
            if (covered.insert(d).second) {
                g_coveredJournal.push_back(d);
                // Inherits: this frame ignores the verdict and never rolls
                // back, so the child's rows live exactly as long as ours.
                (void) buildStack(memoryBlock, d, stack, covered,
                                  rowsSurviveOnFailure);
            }
        }
    }
    g_failConflict = myConflict;
    g_failConsulted = myConsulted;
    g_failContextDependent = true;
    if (insertedHere) g_buildStackPath.erase(proved);
    return false;
}




std::vector<ExpressionWithValidity> ExpressionAnalyzer::sortByValuesDesc(const std::vector<ExpressionWithValidity>& expressions,
    const std::vector<int>& values) {

    if (expressions.size() != values.size()) {
        throw std::invalid_argument("expressions and values must have the same size");
    }

    // Create pairs of (Value, OriginalIndex)
    std::vector<std::pair<int, std::size_t>> pairs;
    pairs.reserve(expressions.size());
    for (std::size_t i = 0; i < expressions.size(); ++i) {
        pairs.push_back(std::make_pair(values[i], i));
    }

    // Sort descending by value
    std::sort(pairs.begin(), pairs.end(),
        [](const std::pair<int, std::size_t>& a, const std::pair<int, std::size_t>& b) {
            return a.first > b.first;
        });

    // Reconstruct the sorted vector
    std::vector<ExpressionWithValidity> result;
    result.reserve(expressions.size());
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        result.push_back(expressions[pairs[i].second]);
    }

    return result;
}


// In GL_Quick_VS/GL_Quick/src/analyze_expressions.cpp

void gl::ExpressionAnalyzer::findEnds(const std::vector<std::string>& path, const std::filesystem::path& outDirParam) {
    namespace fs = std::filesystem;

    // ---- prepare output dir: PROJECT_ROOT/files/raw_proof_graph ----
    std::filesystem::path outDir = outDirParam.empty() ? (fs::path("files") / "raw_proof_graph") : outDirParam;

    // MODIFIED: Do NOT remove existing files. Only ensure directory exists.
    std::error_code ec2;
    fs::create_directories(outDir, ec2);

    // ---- navigate to requested memory block ----
    Memory* memoryBlock = &this->body;
    for (std::size_t i = 0; i < path.size(); ++i) {
        const std::string& elt = path[i];
        Memory* child = simpleMapStore.findChild(memoryBlock, elt);
        if (child == NULL) {
            return; // path invalid
        }
        memoryBlock = child;
    }

    // ---- collect candidate ends (keys of exprOriginMap) ----
    std::set<ExpressionWithValidity> allExprs;
    for (int32_t oid = 1; oid <= memoryBlock->exprOriginMap.count(); ++oid) {
        auto [endExpr, endValidity] = decodeOriginKey(
            memoryBlock->exprOriginMap.decodeKey(oid),
            memoryBlock->originInterner);
        allExprs.insert(ExpressionWithValidity(std::move(endExpr),
                                               std::move(endValidity)));
    }

    // ---- compute stack sizes for sorting ----
    std::vector<ExpressionWithValidity> endsVec(allExprs.begin(), allExprs.end());
    std::vector<int> stackSizes;
    stackSizes.reserve(endsVec.size());
    for (std::size_t i = 0; i < endsVec.size(); ++i) {
        std::vector<std::vector<std::string>> stack;
        std::set<ExpressionWithValidity> covered;
        this->buildStack(*memoryBlock, endsVec[i], stack, covered);
        stackSizes.push_back(static_cast<int>(stack.size()));
    }

    // ---- order ends by size (desc) ----
    std::vector<ExpressionWithValidity> endsOrdered = this->sortByValuesDesc(endsVec, stackSizes);

    // ---- rebuild globalTheoremList (companion dedup set + producer
    // attribution cleared with it; the re-appends below record no producer) ----
    this->globalTheoremList.clear();
    this->globalTheoremStrings.clear();
    this->globalTheoremProducers.clear();
    std::string joinedPath;
    for (std::size_t i = 0; i < path.size(); ++i) {
        if (i > 0) joinedPath.push_back(';');
        joinedPath += path[i];
    }
    for (std::size_t i = 0; i < endsOrdered.size(); ++i) {
        std::string theoremStr;
        // Format: PATH + "+" + VALIDITY + "+" + ORIGINAL
        if (!joinedPath.empty()) {
            theoremStr = joinedPath + "+" + endsOrdered[i].validityName + "+" + endsOrdered[i].original;
        }
        else {
            theoremStr = "+" + endsOrdered[i].validityName + "+" + endsOrdered[i].original;
        }
        this->appendGlobalTheorem(theoremStr, "debug", "-1", "-1");
    }

    // MODIFIED: Scan for highest existing index to append
    int startIdx = 0;
    if (fs::exists(outDir)) {
        int max_found = -1;
        const std::regex reIdx(R"(^(\d+)_.*)");
        for (const auto& entry : fs::directory_iterator(outDir)) {
            if (entry.is_regular_file()) {
                std::string fname = entry.path().filename().string();
                std::smatch m;
                if (std::regex_search(fname, m, reIdx)) {
                    try {
                        int val = std::stoi(m[1].str());
                        if (val > max_found) max_found = val;
                    }
                    catch (...) {}
                }
            }
        }
        if (max_found >= 0) {
            startIdx = max_found + 1;
        }
    }

    // ---- write stacks as <index>_debug.txt and mapping file ----
    // MODIFIED: Open in Append Mode
    std::ofstream mapFile((outDir / "global_theorem_list.txt").string().c_str(), std::ios::out | std::ios::app);
    if (!mapFile.is_open()) {
        return;
    }

    for (std::size_t idx = 0; idx < this->globalTheoremList.size(); ++idx) {
        const std::tuple<std::string, std::string, std::string, std::string>& entry = this->globalTheoremList[idx];
        const std::string& theoremStr = std::get<0>(entry);
        const std::string& method = std::get<1>(entry);
        const std::string& var = std::get<2>(entry);

        // Parse theoremStr... (omitted logic remains same as original, just need to parse)
        std::string pathPart;
        std::string validityPart;
        std::string originalPart;
        std::size_t firstPlus = theoremStr.find('+');
        if (firstPlus == std::string::npos) {
            originalPart = theoremStr;
        }
        else {
            pathPart = theoremStr.substr(0, firstPlus);
            std::size_t secondPlus = theoremStr.find('+', firstPlus + 1);
            if (secondPlus == std::string::npos) {
                originalPart = theoremStr.substr(firstPlus + 1);
            }
            else {
                validityPart = theoremStr.substr(firstPlus + 1, secondPlus - (firstPlus + 1));
                originalPart = theoremStr.substr(secondPlus + 1);
            }
        }

        // navigate to target block
        Memory* mb = &this->body;
        if (!pathPart.empty()) {
            std::size_t start = 0;
            while (true) {
                std::size_t pos = pathPart.find(';', start);
                std::string node = (pos == std::string::npos)
                    ? pathPart.substr(start)
                    : pathPart.substr(start, pos - start);
                if (!node.empty()) {
                    Memory* child = simpleMapStore.findChild(mb, node);
                    if (child == NULL) {
                        mb = NULL; break;
                    }
                    mb = child;
                }
                if (pos == std::string::npos) break;
                start = pos + 1;
            }
        }
        if (mb == NULL) continue;

        // build stack
        ExpressionWithValidity targetKey(originalPart, validityPart);
        std::vector< std::vector<std::string> > stack;
        std::set<ExpressionWithValidity> covered;
        this->buildStack(*mb, targetKey, stack, covered);

        // MODIFIED: Use the calculated accumulated index
        int currentIdx = startIdx + static_cast<int>(idx);
        fs::path filePath = outDir / (std::to_string(static_cast<long long>(currentIdx)) + "_debug.txt");

        std::ofstream ofs(filePath.string().c_str(), std::ios::out | std::ios::trunc);
        if (ofs.is_open()) {
            for (std::size_t r = 0; r < stack.size(); ++r) {
                const std::vector<std::string>& row = stack[r];
                for (std::size_t c = 0; c < row.size(); ++c) {
                    if (c > 0) ofs << '\t';
                    ofs << row[c];
                }
                ofs << '\n';
            }
            ofs.close();
        }

        // record mapping line
        mapFile << theoremStr << '\t' << method << '\t' << var << '\n';
    }
    mapFile.close();
}

void ExpressionAnalyzer::loadGlBinary(const std::filesystem::path& jsonPath) {
    namespace fs = std::filesystem;
    if (!fs::exists(jsonPath)) {
        return;  // first batch on a clean run; nothing to load.
    }

    nlohmann::json root;
    {
        std::ifstream f(jsonPath.string());
        if (!f.is_open()) {
            return;
        }
        try {
            f >> root;
        } catch (const std::exception& e) {
            std::cerr << "[loadGlBinary] failed to parse " << jsonPath
                      << ": " << e.what() << std::endl;
            return;
        }
    }
    if (!root.is_object()) {
        return;
    }

    int loadedSpontaneous = 0;
    int loadedTotal = 0;
    int maxImpl = -1, maxExist = -1, maxAnd = -1, maxOr = -1;

    for (auto it = root.begin(); it != root.end(); ++it) {
        const std::string& name = it.key();
        const auto& entry = it.value();
        if (!entry.is_object()) continue;

        std::string category  = entry.value("category", std::string("atomic"));
        std::string signature = entry.value("signature", std::string());
        int arity             = entry.value("arity", 0);
        std::string definedSet = entry.value("definedSet", std::string());

        std::vector<std::string> elements;
        if (entry.contains("elements") && entry["elements"].is_array()) {
            for (const auto& e : entry["elements"]) {
                if (e.is_string()) elements.push_back(e.get<std::string>());
            }
        }

        // "implications" — an or's compact or-implication list
        // (D-309). The writer omits the key when the
        // list is empty, so absence is the defined state for every
        // non-or entry and for an or written before the field existed.
        std::vector<std::string> implications;
        if (entry.contains("implications")) {
            assert(entry["implications"].is_array()
                && "loadGlBinary: \"implications\" must be a JSON array");
            for (const auto& e : entry["implications"]) {
                assert(e.is_string()
                    && "loadGlBinary: \"implications\" entries must be strings");
                implications.push_back(e.get<std::string>());
            }
        }

        // Always populate compiledExpressions so any read-side path
        // (e.g. encoded-expression resolution) sees the entry.
        {
            LogicalEntity le(category, elements, signature, arity, definedSet);
            le.implications = implications;
            compiledExpressions[name] = le;
        }
        ++loadedTotal;

        // An or / existence lacking its list cannot be compiled here — the
        // constructor loads the binary BEFORE coreExpressionMap exists — so
        // it is recorded and compiled at the first preMintReducedOrs seam,
        // which precedes every burst.
        if (category == "or" && implications.empty()) {
            orsAwaitingImplications.insert(name);
        }
        if (category == "existence" && implications.empty()
            && ExpressionAnalyzer::expectedExistenceImplicationCount(
                   static_cast<int32_t>(elements.size())) > 0) {
            existencesAwaitingImplications.insert(name);
        }

        // Spontaneous categories also need a repetitionExclusionMap entry
        // and contribute to counter seeding. Anchor / atomic entries are
        // batch-local and stay out of these structures.
        const bool isSpontaneous =
            (category == "implication") || (category == "existence")
            || (category == "or") || (category == "and");
        if (!isSpontaneous) continue;

        // Identify the prefix that matches the category and parse the
        // trailing integer. Names that don't fit (e.g. an anchor
        // mis-categorised to "and") are skipped for counter-seeding but
        // still appear in compiledExpressions.
        const std::string& prefix = category;  // implication / existence / or / and
        if (name.size() <= prefix.size() ||
            name.compare(0, prefix.size(), prefix) != 0) {
            continue;
        }
        int n = -1;
        try {
            n = std::stoi(name.substr(prefix.size()));
        } catch (...) {
            continue;
        }
        if (n < 0) continue;

        if      (category == "implication" && n > maxImpl)  maxImpl  = n;
        else if (category == "existence"   && n > maxExist) maxExist = n;
        else if (category == "and"         && n > maxAnd)   maxAnd   = n;
        else if (category == "or"          && n > maxOr)    maxOr    = n;

        // Use the loaded elements vector directly as splitNK. excludeRepetitions
        // stores the splitNK in LogicalEntity::elements when allocating, so the
        // JSON's "elements" field for a spontaneous entry IS the splitNK that
        // was originally registered. We rebuild an identity reverseUnchMap over
        // the u_<i> slots; permutation variants are not pre-registered (only
        // the canonical chain order is), which is the conservative choice — the
        // worst case is that a future call computes a permuted splitNK and
        // misses the cache, falling through to the allocation path with a
        // counter that's already past the existing N, so identifiers stay
        // unique. The chapter-85 / Gauss-renames-Peano case (the bug this
        // change fixes) hits the canonical key so it is covered.
        std::map<std::string, std::string> reverseUnchMap;
        for (int i = 1; i <= arity; ++i) {
            const std::string uN = "u_" + std::to_string(i);
            reverseUnchMap[uN] = uN;
        }
        // Category included in the key so a cross-batch load can carry, e.g.,
        // an incubator-allocated implication with elements E and a main-batch
        // existence with the same elements E without the existence reusing the
        // implication's name. See repetitionExclusionMap declaration in
        // prover.hpp for the full rationale.
        repetitionExclusionMap[std::make_pair(elements, category)] = std::make_tuple(reverseUnchMap, name, elements);
        ++loadedSpontaneous;
    }

    if (maxImpl  >= 0) implCounter      = std::max(implCounter,      maxImpl  + 1);
    if (maxExist >= 0) existenceCounter = std::max(existenceCounter, maxExist + 1);
    if (maxAnd   >= 0) andCounter       = std::max(andCounter,       maxAnd   + 1);
    if (maxOr    >= 0) orCounter        = std::max(orCounter,        maxOr    + 1);

#if 0
    std::cout << "[loadGlBinary] loaded " << loadedTotal << " entries ("
              << loadedSpontaneous << " spontaneous) from " << jsonPath
              << " — counters seeded: implCounter=" << implCounter
              << " existenceCounter=" << existenceCounter
              << " andCounter=" << andCounter
              << " orCounter=" << orCounter << std::endl;
#endif
}


void ExpressionAnalyzer::exportCompiledExpressionsJSON(const std::filesystem::path& outDir) {
    namespace fs = std::filesystem;
    fs::create_directories(outDir);

    fs::path outPath = outDir / ("GL_binary_" + anchorID_ + ".json");
    nlohmann::json root = nlohmann::json::object();

    for (const auto& kv : this->compiledExpressions) {
        const std::string& coreName = kv.first;
        const auto& compExpr = kv.second;

        nlohmann::json entry;
        entry["category"] = compExpr.category;
        entry["signature"] = compExpr.signature;
        entry["arity"] = compExpr.arity;
        entry["definedSet"] = compExpr.definedSet;

        nlohmann::json elems = nlohmann::json::array();
        for (const auto& e : compExpr.elements) {
            elems.push_back(e);
        }
        entry["elements"] = elems;

        // "implications" only for an or carrying its or-implication
        // compacts (D-309); omitted when empty so
        // every other entry keeps the pre-field layout byte-for-byte.
        if (!compExpr.implications.empty()) {
            nlohmann::json impls = nlohmann::json::array();
            for (const auto& s : compExpr.implications) {
                impls.push_back(s);
            }
            entry["implications"] = impls;
        }

        root[coreName] = entry;
    }

    std::ofstream f(outPath.string().c_str());
    if (f.is_open()) {
        f << root.dump(2);
        f.close();
    }
}

void ExpressionAnalyzer::generateRawProofGraph(
    const std::vector<std::tuple<std::string, std::string, std::string, std::string>>& theoremList,
    const std::filesystem::path& outDirParam)
{
    namespace fs = std::filesystem;
    using FrameClock = std::chrono::steady_clock;

    const auto rawGenerationStarted = FrameClock::now();
    std::string frameBatch;
#ifdef _WIN32
    char* batchRaw = nullptr;
    size_t batchLength = 0;
    const errno_t batchEnvironmentResult =
        _dupenv_s(&batchRaw, &batchLength, "GL_FRAME_BATCH");
    assert(batchEnvironmentResult == 0);
    if (batchRaw != nullptr) {
        frameBatch.assign(batchRaw);
        std::free(batchRaw);
    }
#else
    const char* batchRaw = std::getenv("GL_FRAME_BATCH");
    if (batchRaw != nullptr) frameBatch.assign(batchRaw);
#endif
    const auto frameSecondsSince = [](FrameClock::time_point started) {
        return std::chrono::duration<double>(FrameClock::now() - started).count();
    };
    const auto recordFrameTiming = [&](const char* stage,
                                       double seconds,
                                       int64_t count = 1) {
        assert(stage != nullptr && stage[0] != '\0');
        assert(seconds >= 0.0);
        assert(count > 0);
        std::string timingPath;
#ifdef _WIN32
        char* timingPathRaw = nullptr;
        size_t timingPathLength = 0;
        const errno_t timingEnvironmentResult = _dupenv_s(
            &timingPathRaw, &timingPathLength, "GL_FRAME_TIMING_PATH");
        assert(timingEnvironmentResult == 0);
        if (timingPathRaw == nullptr) return;
        timingPath.assign(timingPathRaw);
        std::free(timingPathRaw);
#else
        const char* timingPathRaw = std::getenv("GL_FRAME_TIMING_PATH");
        if (timingPathRaw == nullptr) return;
        timingPath.assign(timingPathRaw);
#endif
        std::ofstream timing(timingPath, std::ios::app);
        assert(timing.is_open());
        timing << "{\"batch\":\"" << frameBatch
               << "\",\"count\":" << count
               << ",\"excluded\":false"
               << ",\"parent\":\"native.raw_proof\""
               << ",\"seconds\":" << std::setprecision(12) << seconds
               << ",\"stage\":\"" << stage << "\"}\n";
        timing.flush();
        assert(timing.good());
    };
    double buildStackSeconds = 0.0;
    double serializationSeconds = 0.0;
    double releaseSeconds = 0.0;
    int64_t buildStackCalls = 0;
    int64_t serializedChapters = 0;
    int64_t releasedTheorems = 0;

    // D-51 (option 1): reset thread_local path stack at run start. Each
    // chapter emission pushes/pops independently — the clear here is a
    // belt-and-suspenders against any leak across runs.
    clearBuildStackPath();

    // Fresh cross-chapter usage graph per export run; the current-chapter
    // theorem is set per loop iteration below and empty outside the loop
    // (empty = buildStack's cross-chapter admissibility check disabled).
    exportTheoremUsage.clear();
    exportCurrentChapterTheorem.clear();

    // Per-chapter reload release (G-53): the export
    // reloads LBs from their SSD images on demand; without releasing them the
    // resident block set grows monotonically across the walk to static-pool
    // exhaustion. Every reload appends to this sink (Memory::reloadFromImage);
    // each theorem's reloaded LBs are released after its chapters are written
    // (loop below). The on-disk image stays, so a later theorem that revisits
    // an LB reloads it again on demand.
    std::vector<Memory*> exportReloaded;
    g_exportReloadSink = &exportReloaded;

    std::cout << "Number proven theorems: "
        << theoremList.size() / 2 << "\n";

    // ---------- out dir: "files/raw_proof_graph" by default ----------
    fs::path outDir = outDirParam.empty() ? (fs::path("files") / "raw_proof_graph") : outDirParam;
    std::error_code ec;

    // No remove_all: earlier batches' chapters must survive — the export
    // appends, with the start index scanned from existing files below.
    fs::create_directories(outDir, ec);

    // Single canonical GL-binary folder regardless of per-batch outDir
    // (mirrors prover.cpp::compileCoreExpressionMap reader path). Without
    // this, configs that override raw_proof_graph_folder (incubator batches
    // set it to files/incubator/raw_proof_graph) would land the export in
    // files/incubator/GL_binaries/ — a directory the loader and the Python
    // _merge_into_shared step never read from. See D-54.
    fs::path glBinDir = fs::path(__FILE__).parent_path()
                        .parent_path().parent_path().parent_path()
                      / "files" / "GL_binaries";
    const auto binaryExportStarted = FrameClock::now();
    this->exportCompiledExpressionsJSON(glBinDir);
    const double binaryExportSeconds = frameSecondsSince(binaryExportStarted);

    // Scan for the start index from the highest existing numeric prefix.
    const auto indexStarted = FrameClock::now();
    int idx = 0;
    if (fs::exists(outDir)) {
        int max_found = -1;
        const std::regex reIdx(R"(^(\d+)_.*)");
        for (const auto& entry : fs::directory_iterator(outDir)) {
            if (entry.is_regular_file()) {
                std::string fname = entry.path().filename().string();
                std::smatch m;
                if (std::regex_search(fname, m, reIdx)) {
                    try {
                        int val = std::stoi(m[1].str());
                        if (val > max_found) max_found = val;
                    }
                    catch (...) {}
                }
            }
        }
        if (max_found >= 0) {
            idx = max_found + 1;
        }
    }
    const double indexSeconds = frameSecondsSince(indexStarted);

    const auto timedBuildStack = [&](Memory& memoryBlock,
                                     const ExpressionWithValidity& proved,
                                     std::vector<std::vector<std::string>>& stack,
                                     std::set<ExpressionWithValidity>& covered) {
        const auto started = FrameClock::now();
        // Each top-level walk starts with an empty `covered`, so its journal
        // starts empty too. Without this the outermost frame's surviving
        // insertions — which nothing ever rolls back — would accumulate across
        // the export.
        g_coveredJournal.clear();
        g_noGood.clear();
        const bool result = this->buildStack(memoryBlock, proved, stack, covered);
        buildStackSeconds += frameSecondsSince(started);
        ++buildStackCalls;
        return result;
    };

    auto toLower = [](std::string s) {
        for (std::size_t i = 0; i < s.size(); ++i) {
            s[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(s[i])));
        }
        return s;
        };

    auto startsWith = [](const std::string& s, const std::string& pref) -> bool {
        return s.size() >= pref.size() && std::equal(pref.begin(), pref.end(), s.begin());
        };
    auto endsWith = [](const std::string& s, const std::string& suf) -> bool {
        return s.size() >= suf.size() && std::equal(s.end() - suf.size(), s.end(), suf.begin());
        };

    // Generic over the statement-container type: the statified registry and
    // its sibling vectors (ArenaVector) all pass through; each exposes
    // size() + operator[].
    auto containsEncoded = [](const auto& vec,
                              const NameMap& nm, const std::string& expr) -> bool {
        // Stored rows are canonical-pipeline encodings, so full struct
        // equality collapses to the (originalId, validityId) pair. lookup is
        // non-minting: a never-interned needle cannot be a stored statement.
        const NameId origId = nm.lookup(expr);
        if (origId == 0) return false;
        for (int32_t i = 0; i < static_cast<int32_t>(vec.size()); ++i) {
            if (vec[i].originalId == origId
                && vec[i].validityId == NameMap::MAIN_ID) return true;
        }
        return false;
        };

    // The single choke point every chapter kind passes through (direct,
    // induction triad, debug, reformulated, incubator back-reformulation,
    // or-theorem): writes the chapter file, then commits the chapter's
    // theorem-citation edges into the export usage graph so later chapters'
    // buildStack admissibility probes see them
    // (D-276). Committing on emitted rows —
    // not on how they were built — also covers cache-served stacks.
    auto writeStackIndexed = [&](int idx, const std::string& part,
        const std::string& theoremName,
        const std::vector<std::vector<std::string>>& stackRows) {

            const auto started = FrameClock::now();

            fs::path fp = outDir / (std::to_string(idx) + "_" + part + ".txt");
            std::ofstream f(fp.c_str());

            for (std::size_t r = 0; r < stackRows.size(); ++r) {
                const std::vector<std::string>& row = stackRows[r];
                for (std::size_t c = 0; c < row.size(); ++c) {
                    if (c > 0) f << '\t';
                    f << row[c];
                }
                f << '\n';
            }
            f.flush();
            assert(f.good());
            exportTheoremUsage.addEdgesFromStack(theoremName, stackRows);
            serializationSeconds += frameSecondsSince(started);
            ++serializedChapters;
        };

    auto directStack = [&](const std::string& theorem) -> std::vector<std::vector<std::string> > {
        std::vector< std::tuple<
            std::string,                    // leftExpr
            std::vector<std::string>,       // args of the current implication node
            std::set<std::string>           // node->left.arguments
        > > tempChain;

        std::string head = ce::disintegrateImplication(theorem, tempChain, coreExpressionMap);

        std::vector<std::string> chain;
        chain.reserve(tempChain.size());
        for (std::size_t i = 0; i < tempChain.size(); ++i) {
            chain.push_back(std::get<0>(tempChain[i]));
        }

        Memory* mb = &body;
        for (std::size_t i = 0; i < chain.size(); ++i) {
            Memory* child = simpleMapStore.findChild(mb, chain[i]);
            if (child == NULL) return std::vector<std::vector<std::string> >();
            mb = child;
        }

        std::vector<std::vector<std::string> > stack;
        std::set<ExpressionWithValidity> covered;
        // D-51: insert the chapter goal (the wrapped theorem expression) into
        // the buildStack path. The existing path-cycle filter then rejects any
        // origin whose deps include the chapter goal — this is exactly the
        // self-reference shape (origin tag "implication" with the wrapped
        // theorem as a dep, which would emit a chapter row that fails the
        // verifier's self-reference check). Backtracking then falls through
        // to the contradiction-LB fallback (where the actual contradiction
        // recipe lives).
        ExpressionWithValidity chapterGoalEv(theorem, "main");
        g_buildStackPath.insert(chapterGoalEv);
        timedBuildStack(*mb, ExpressionWithValidity(head, "main"), stack, covered);
        g_buildStackPath.erase(chapterGoalEv);
        return stack;
        };

    auto checkZeroStack = [&](const std::string& theorem,
        const std::string& inductionVar,
        const std::string& recCounter) -> std::vector<std::vector<std::string> > {
            std::vector< std::tuple<std::string, std::vector<std::string>, std::set<std::string> > > tempChain;
            std::string head = ce::disintegrateImplication(theorem, tempChain, coreExpressionMap);

            std::vector<std::string> chain;
            chain.reserve(tempChain.size());
            for (std::size_t i = 0; i < tempChain.size(); ++i) {
                chain.push_back(std::get<0>(tempChain[i]));
            }

            Memory* mb = &body;
            for (std::size_t i = 0; i < chain.size(); ++i) {
                Memory* child = simpleMapStore.findChild(mb, chain[i]);
                if (child == NULL) {
                    return std::vector<std::vector<std::string> >();
                }
                mb = child;
            }

            std::vector<std::string> args0 = ce::getArgs(chain[0]);
            if (args0.size() < 2) {
                return std::vector<std::vector<std::string> >();
            }
            const std::string zeroName = args0[1];

            std::vector<std::pair<std::string, Memory*>> eqChildren;
            simpleMapStore.forEachChild(mb, [&](const gl::StrSpan& k, Memory* c) {
                eqChildren.emplace_back(std::string(k.ptr, static_cast<std::size_t>(k.len)), c);
            });
            for (const std::pair<std::string, Memory*>& kv : eqChildren) {
                const std::string& key = kv.first;
                if (!startsWith(key, std::string("(=[s(rec") + recCounter)
                    || !endsWith(key, std::string(",") + zeroName + "])")) {
                    continue;
                }

                Memory* eqNode = kv.second;
                if (eqNode == NULL) {
                    continue;
                }

                // Relaxed: head may live in the LB's full statement registry
                // (intEncodedStatements) without being in the local-origin
                // registry when it arrived via mailIn (status=3,
                // isLocal=false). A local-only gate misses this case after
                // the symmetry-disabling sequence.
                //
                // A DISCHARGED node emptied its registry at discharge —
                // the probe target is the exact captured pair set
                // (D-157). A live node (parked-
                // never-woken, or an active the final barrier's pressure
                // path dumped) reloads and walks the registry as before.
                bool inRegistry;
                if (eqNode->dischargedForever) {
                    // dischargedRegistryKeys (heap) is the probe target, but
                    // the nameMap lookup that forms its key needs the cold
                    // string tables resident; a discharged node may have been
                    // drained to SSD under pressure, so reload-for-read first
                    // (no-op if resident; sanctioned on discharged LBs,
                    // D-158). The registry stays empty.
                    eqNode->ensureLoadedForRead(lbdeload::kDeloadDirectory);
                    const NameId origId = eqNode->nameMap.lookup(head);
                    inRegistry = origId != 0
                        && eqNode->dischargedRegistryKeys.count(
                               packStatementKey(origId,
                                                NameMap::MAIN_ID)) > 0;
                }
                else {
                    eqNode->ensureLoaded(lbdeload::kDeloadDirectory);
                    inRegistry =
                        containsEncoded(eqNode->intEncodedStatements,
                                        eqNode->nameMap, head);
                }
                if (!inRegistry) continue;

                std::vector<std::string> ev = ce::getArgs(eqNode->exprKey());
                if (ev.empty() || ev[0] != inductionVar) {
                    continue;
                }

                std::vector<std::string> keyArgs = ce::getArgs(key);
                if (keyArgs.size() < 2) {
                    continue;
                }
                const std::string recName = keyArgs[0];

                const std::string tempExpr = std::string("(=[") + recName + "," + zeroName + "])";
                Memory* mbTarget = simpleMapStore.findChild(mb, tempExpr);
                if (mbTarget == NULL) {
                    continue;
                }
                std::vector<std::vector<std::string> > stack;
                std::set<ExpressionWithValidity> covered;
                timedBuildStack(
                    *mbTarget, ExpressionWithValidity(head, "main"), stack, covered);
                return stack;
            }
            return std::vector<std::vector<std::string> >();
        };

    auto checkInductionConditionStack = [&](const std::string& theorem,
        const std::string& inductionVar,
        const std::string& recCounter) -> std::vector<std::vector<std::string> > {
            std::vector< std::tuple<std::string, std::vector<std::string>, std::set<std::string> > > tempChain;
            std::string head = ce::disintegrateImplication(theorem, tempChain, coreExpressionMap);

            std::vector<std::string> chain;
            chain.reserve(tempChain.size());
            for (std::size_t i = 0; i < tempChain.size(); ++i) chain.push_back(std::get<0>(tempChain[i]));

            Memory* mb = &body;
            for (std::size_t i = 0; i < chain.size(); ++i) {
                Memory* child = simpleMapStore.findChild(mb, chain[i]);
                if (child == NULL) return std::vector<std::vector<std::string> >();
                mb = child;
            }

            std::vector<std::string> args0 = ce::getArgs(chain[0]);
            if (args0.size() < 4) return std::vector<std::vector<std::string> >();
            const std::string sName = args0[2];

            std::vector<std::pair<std::string, Memory*>> in2Children;
            simpleMapStore.forEachChild(mb, [&](const gl::StrSpan& k, Memory* c) {
                in2Children.emplace_back(std::string(k.ptr, static_cast<std::size_t>(k.len)), c);
            });
            for (const std::pair<std::string, Memory*>& kv : in2Children) {
                const std::string& key = kv.first;
                if (!startsWith(key, std::string("(in2[rec") + recCounter)) continue;
                if (!endsWith(key, std::string("") + inductionVar + "," + sName + "])")) continue;

                Memory* node = kv.second;
                if (node == NULL) continue;
                // The intLocalEncodedStatementsSet probe is a heap mirror
                // (maintained at every mutation site — I-86) valid on
                // deloaded/discharged nodes, but the nameMap lookup that
                // forms its key needs the cold string tables resident —
                // reload-for-read first (no-op if resident; sanctioned on
                // discharged LBs, D-158). buildStack below
                // reads only RAM state (exprOriginMap, Rule 16).
                node->ensureLoadedForRead(lbdeload::kDeloadDirectory);
                const NameId locOrigId = node->nameMap.lookup(head);
                if (locOrigId == 0) continue;
                if (!node->intLocalEncodedStatementsSet.contains(
                        packStatementKey(locOrigId, NameMap::MAIN_ID))) continue;

                std::vector<std::vector<std::string> > stack;
                std::set<ExpressionWithValidity> covered;
                timedBuildStack(
                    *node, ExpressionWithValidity(head, "main"), stack, covered);
                return stack;
            }
            return std::vector<std::vector<std::string> >();
        };

    // Induction typing chapter: walk back from (in[inductionVar, N]) in the
    // PARENT memoryBlock's exprOriginMap. fXY / fXYZ forward-chaining on a
    // positive (in2[inductionVar, …])/(in3[…, inductionVar, …, f]) premise
    // fires at the parent memoryBlock, so the derivation's origin chain lives
    // there (mail propagates statements to sub-blocks but the origin record
    // stays with the block where the rule fired). The chapter corresponds to
    // the typing sub-proof that gates induction promotion in updateGlobal.
    // See docs/agentic_swdd/induction_typing_plan.md.
    auto inductionTypingStack = [&](const std::string& theorem,
        const std::string& inductionVar,
        const std::string& /*recCounter*/) -> std::vector<std::vector<std::string> > {
            std::vector< std::tuple<std::string, std::vector<std::string>, std::set<std::string> > > tempChain;
            (void)ce::disintegrateImplication(theorem, tempChain, coreExpressionMap);

            std::vector<std::string> chain;
            chain.reserve(tempChain.size());
            for (std::size_t i = 0; i < tempChain.size(); ++i) chain.push_back(std::get<0>(tempChain[i]));

            Memory* mb = &body;
            for (std::size_t i = 0; i < chain.size(); ++i) {
                Memory* child = simpleMapStore.findChild(mb, chain[i]);
                if (child == NULL) return std::vector<std::vector<std::string> >();
                mb = child;
            }

            // Anchor N slot (args[0] of the anchor expression in chain[0])
            std::vector<std::string> args0 = ce::getArgs(chain[0]);
            if (args0.empty()) return std::vector<std::vector<std::string> >();
            const std::string nName = args0[0];

            const std::string typingGoal =
                std::string("(in[") + inductionVar + "," + nName + "])";

            std::vector<std::vector<std::string> > stack;
            std::set<ExpressionWithValidity> covered;
            timedBuildStack(
                *mb, ExpressionWithValidity(typingGoal, "main"), stack, covered);
            return stack;
        };

    auto debugStack = [&](const std::string& pathPlusEnd) -> std::vector<std::vector<std::string>> {
        std::string::size_type firstPlus = pathPlusEnd.find('+');
        if (firstPlus == std::string::npos) return std::vector<std::vector<std::string>>();

        std::string pathPart = pathPlusEnd.substr(0, firstPlus);
        std::string validityName;
        std::string endExpr;

        std::string::size_type secondPlus = pathPlusEnd.find('+', firstPlus + 1);
        if (secondPlus != std::string::npos) {
            validityName = pathPlusEnd.substr(firstPlus + 1, secondPlus - (firstPlus + 1));
            endExpr = pathPlusEnd.substr(secondPlus + 1);
        }
        else {
            endExpr = pathPlusEnd.substr(firstPlus + 1);
        }

        Memory* mb = &body;
        if (!pathPart.empty()) {
            std::string token;
            for (std::size_t i = 0; i <= pathPart.size(); ++i) {
                if (i == pathPart.size() || pathPart[i] == ';') {
                    if (!token.empty()) {
                        Memory* child = simpleMapStore.findChild(mb, token);
                        if (child == NULL)
                            return std::vector<std::vector<std::string>>();
                        mb = child;
                        token.clear();
                    }
                }
                else {
                    token.push_back(pathPart[i]);
                }
            }
        }

        ExpressionWithValidity target(endExpr, validityName);
        std::vector<std::vector<std::string>> stack;
        std::set<ExpressionWithValidity> covered;

        timedBuildStack(*mb, target, stack, covered);

        return stack;
        };


    // ---------- emit stacks + mapping file ----------
    // Append mode: later batches extend the mapping file; 'idx' already
    // holds the scanned start offset.
    std::ofstream mapping((outDir / "global_theorem_list.txt").c_str(), std::ios::out | std::ios::app);

    int lastDirectIdx = -1;
    for (std::size_t i = 0; i < theoremList.size(); ++i) {
        const std::string& name = std::get<0>(theoremList[i]);
        const std::string& methodOrig = std::get<1>(theoremList[i]);
        const std::string& var = std::get<2>(theoremList[i]);
        const std::string& recCtr = std::get<3>(theoremList[i]);

        const std::string method = toLower(methodOrig);

        // Arm buildStack's cross-chapter admissibility check for every
        // chapter of this theorem (induction triads included); inert for
        // the synthetic branches, which never call buildStack.
        exportCurrentChapterTheorem = name;

        if (method == "induction") {
            auto cacheIt = cachedProofStacks.find(name);

            // Chapter order: typing first, then check_zero, then
            // check_induction_condition. Typing is the gate — it establishes
            // `(in[var,N])` which both subsequent chapters rely on.
            // `mapping << name …` is written once against the TYPING chapter
            // (chapter index `idx`) so the verifier pairs the induction row
            // with index `idx` and its two siblings at `idx+1`, `idx+2`.
            std::vector<std::vector<std::string> > stT =
                (cacheIt != cachedProofStacks.end()) ? cacheIt->second.stack2 : inductionTypingStack(name, var, recCtr);
            writeStackIndexed(idx, "induction_typing", name, stT);
            mapping << name << '\t' << methodOrig << '\t' << var << '\n';
            ++idx;

            std::vector<std::vector<std::string> > st0 =
                (cacheIt != cachedProofStacks.end()) ? cacheIt->second.stack0 : checkZeroStack(name, var, recCtr);
            writeStackIndexed(idx, "check_zero", name, st0);
            ++idx;

            std::vector<std::vector<std::string> > st1 =
                (cacheIt != cachedProofStacks.end()) ? cacheIt->second.stack1 : checkInductionConditionStack(name, var, recCtr);
            writeStackIndexed(idx, "check_induction_condition", name, st1);
            ++idx;
        }
        else if (method == "direct") {
            auto cacheIt = cachedProofStacks.find(name);
            std::vector<std::vector<std::string> > st =
                (cacheIt != cachedProofStacks.end()) ? cacheIt->second.stack0 : directStack(name);
            writeStackIndexed(idx, "direct_proof", name, st);
            mapping << name << '\t' << methodOrig << '\t' << var << '\n';
            lastDirectIdx = idx;
            ++idx;
        }
        else if (method == "proved not broadcast") {
            // Level-refused closure recorded by the proved-not-broadcast
            // tier: a real derivation exists in the producing LB, so the
            // chapter is the ordinary direct-proof walk under its own
            // chapter kind.
            std::vector<std::vector<std::string> > st = directStack(name);
            writeStackIndexed(idx, "proved_not_broadcast", name, st);
            mapping << name << '\t' << methodOrig << '\t' << var << '\n';
            ++idx;
        }
        else if (method == "debug") {
            std::vector<std::vector<std::string> > st = debugStack(name);
            writeStackIndexed(idx, "debug", name, st);
            mapping << name << '\t' << methodOrig << '\t' << var << '\n';
            ++idx;
        }
        else if (method == "reformulated statement") {
            std::vector<std::vector<std::string> > st;
            st.push_back(std::vector<std::string>());
            st.back().push_back(name);
            st.back().push_back("main");
            st.back().push_back("reformulated from");
            st.back().push_back(var);
            st.back().push_back("main");
            writeStackIndexed(idx, "reformulated_statement", name, st);
            mapping << name << '\t' << methodOrig << '\t' << var << '\n';
            ++idx;
        }
        else if (method == "incubator back reformulation") {
            std::vector<std::vector<std::string>> st;
            st.push_back(std::vector<std::string>());
            st.back().push_back(name);
            st.back().push_back("main");
            st.back().push_back("incubator back reformulation");
            st.back().push_back(var);
            st.back().push_back("main");
            writeStackIndexed(idx, "back_reformulated_statement", name, st);
            mapping << name << '\t' << methodOrig << '\t' << var << '\n';
            ++idx;
        }
        else if (method == "or theorem") {
            std::vector<std::vector<std::string>> st;
            st.push_back(std::vector<std::string>());
            st.back().push_back(name);
            st.back().push_back("main");
            st.back().push_back("or theorem");
            st.back().push_back(var);       // existence theorem
            st.back().push_back("main");
            st.back().push_back(recCtr);    // companion theorem
            st.back().push_back("main");
            writeStackIndexed(idx, "or_theorem", name, st);
            mapping << name << '\t' << methodOrig << '\t' << var << '\n';
            ++idx;
        }
        else if (method == "or elimination") {
            // Pre-split merge chapter: one fabricated row citing the two
            // guard variants (the row's aux slots) plus the licensing or
            // theorem (the seam's citation ledger — a registered merge
            // without its license is a seam bug, never a skippable state).
            const auto cit = orElimCitedOrByMerged.find(name);
            assert(cit != orElimCitedOrByMerged.end()
                && "generateRawProofGraph: or-elimination row without a recorded or-theorem license");
            std::vector<std::vector<std::string>> st;
            st.push_back(std::vector<std::string>());
            st.back().push_back(name);
            st.back().push_back("main");
            st.back().push_back("or elimination");
            st.back().push_back(var);          // guard variant A
            st.back().push_back("main");
            st.back().push_back(recCtr);       // guard variant B
            st.back().push_back("main");
            st.back().push_back(cit->second);  // licensing or theorem
            st.back().push_back("main");
            writeStackIndexed(idx, "or_elimination", name, st);
            mapping << name << '\t' << methodOrig << '\t' << var << '\n';
            ++idx;
        }
        else {
            std::vector<std::vector<std::string> > empty;
            writeStackIndexed(idx, "unknown", name, empty);
            mapping << name << '\t' << methodOrig << '\t' << var << '\n';
            ++idx;
        }

        exportCurrentChapterTheorem.clear();

        // Release the LBs this theorem's chapters reloaded from SSD images.
        // The read-only export brought them back on demand; their on-disk
        // images stay, so a later theorem revisiting an LB reloads it again.
        // Without this, the reloads accumulate across theorems to static-pool
        // exhaustion (G-53). Release DISPATCHES on the recorded format exactly
        // as the reload did: a raw-format LB (a live equality node the export
        // reloaded from its eviction image) must keep its container bookkeeping
        // for the next raw rebind — and on the extent path it has no named file
        // set at all — so the v3 release walk would be wrong on both counts.
        const auto releaseStarted = FrameClock::now();
        for (Memory* lb : exportReloaded) lb->releaseStaticBlocksDispatch();
        exportReloaded.clear();
        releaseSeconds += frameSecondsSince(releaseStarted);
        ++releasedTheorems;
    }
    g_exportReloadSink = nullptr;
    mapping.close();

    const double totalSeconds = frameSecondsSince(rawGenerationStarted);
    const double knownSeconds = binaryExportSeconds + indexSeconds
        + buildStackSeconds + serializationSeconds + releaseSeconds;
    assert(totalSeconds >= knownSeconds);
    recordFrameTiming("native.raw.binary_export", binaryExportSeconds);
    recordFrameTiming("native.raw.index", indexSeconds);
    recordFrameTiming(
        "native.raw.build_stack", buildStackSeconds,
        buildStackCalls == 0 ? 1 : buildStackCalls);
    recordFrameTiming(
        "native.raw.serialization", serializationSeconds,
        serializedChapters == 0 ? 1 : serializedChapters);
    recordFrameTiming(
        "native.raw.release", releaseSeconds,
        releasedTheorems == 0 ? 1 : releasedTheorems);
    recordFrameTiming("native.raw.other", totalSeconds - knownSeconds);
}


} // namespace gl
