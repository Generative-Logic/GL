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
#include "proof_reconstructor.hpp"
#include "../memory_infra/lb_deload.hpp"
#include "../logical_block/scope_syntax.hpp"
namespace gl {
/// @brief Bind the proof reconstructor operation inputs.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param simpleMapStore Borrowed SimpleMapStore interface or state; must outlive this component.
/// @param exportTheoremUsage Borrowed TheoremUsageGraph interface or state; must outlive this component.
/// @param exportCurrentChapterTheorem Borrowed std::string interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
ProofReconstructor::ProofReconstructor(SimpleMapStore& simpleMapStore, TheoremUsageGraph& exportTheoremUsage, std::string& exportCurrentChapterTheorem) : simpleMapStore(simpleMapStore), exportTheoremUsage(exportTheoremUsage), exportCurrentChapterTheorem(exportCurrentChapterTheorem) {}

/// @brief Probe whether a `__contradiction__` twin LB can actually resolve
///        `proved` — i.e. holds at least one origin record for it — before
///        buildStack's D-51 fallback switches in.
///
/// @details
/// Mirrors the entry-side origin check of the walk the switch would start
/// exactly: read-only reload (D-158), the same scope choice (lifted within
/// the twin, or exact when that walk keeps its scope), non-minting key
/// probe, non-empty run. A twin that primed but never converged (no
/// discharge record) fails the probe, and so does a twin that holds the
/// statement only at a shallower scope than an exact walk asks for; the
/// fallback then does not apply and the walk unwinds as an ordinary
/// candidate failure so backtracking can reach acyclic records one level
/// up — instead of dying on the twin's entry no-origin assert.
///
/// @param contraLB      The `__contradiction__` twin candidate.
/// @param proved        The expression/validity the walk must justify.
/// @param preserveScope The switched walk's scope mode: true keeps
///                      `proved`'s exact scope (a `zoom in`,
///                      `induction scope` or `int scope` dependency), false
///                      lifts it.
/// @return true if the twin holds at least one origin record for `proved`
///         in the form the switched walk checks; false otherwise.
/// @invariant Non-minting (I-91).
/// @see buildStack — both D-51 fallback sites are the only callers.
bool ProofReconstructor::contradictionLbHoldsRecord(Memory& contraLB,
    const ExpressionWithValidity& proved, bool preserveScope) {
    contraLB.ensureLoadedForRead(lbdeload::kDeloadDirectory);
    const ExpressionWithValidity lifted = preserveScope ? proved
        : liftToShallowestOriginAncestor(contraLB, proved);
    int64_t pk = 0;
    if (!lookupOriginKey(contraLB.originInterner, lifted.original,
                         lifted.validityName, pk)) {
        return false;
    }
    const int32_t oid = contraLB.exprOriginMap.lookup(pk);
    return oid != 0 && contraLB.exprOriginMap.runLen(oid) > 0;
}

/// @brief Emit one chapter's derivation rows for `proved`, walking
        ///        this memoryBlock's origin records ([D-51](../../docs/agentic_swdd/40_decisions.md#d-51)).
        ///
        /// @details
        /// Recursive backtracking search for an acyclic derivation tree.
        /// Candidates are the node's origin records in D-49 preference order;
        /// a candidate whose lifted dependencies touch the current recursion
        /// path is skipped, and a candidate whose subtree fails is rolled back
        /// (rows truncated, `covered` insertions undone through the journal)
        /// before the next is tried.
        ///
        /// When every candidate fails, the contradiction-LB fallback is tried
        /// and then, as a last resort, the front origin is emitted anyway with
        /// its dependency subtree walked in full, so the chapter has content
        /// even though the verifier will flag the residual cycle.
        ///
        /// @param memoryBlock          The LB whose origin records justify
        ///                             `proved`.
        /// @param proved               The expression / validity to justify.
        ///                             Lifted to its shallowest origin-bearing
        ///                             ancestor on entry.
        /// @param stack                Accumulating chapter rows; a failing
        ///                             candidate truncates back to its mark.
        /// @param covered              Nodes already given a row this chapter,
        ///                             so each `(expression, lifted validity)`
        ///                             appears once. Membership only — never
        ///                             iterated.
        /// @param preserveScope Preserve the exact requested source scope of a zoom in dependency.
        /// @param rowsSurviveOnFailure Whether rows this call emits still
        ///                             reach the chapter if it returns false.
        ///                             False only for a recursion from a
        ///                             parent's candidate loop, which truncates
        ///                             the stack the moment it receives false —
        ///                             so the last-resort fallback is skipped
        ///                             there, being work whose only consumer is
        ///                             that truncation. Defaults to true, the
        ///                             conservative value every top-level
        ///                             caller wants.
        /// @return True when an acyclic derivation tree was emitted; false to
        ///         tell a recursive caller to backtrack onto another origin.
        ///
        /// @invariant A false return leaves `covered` and `g_buildStackPath`
        ///            exactly as the call found them.
        /// @see contradictionLbHoldsRecord
bool ProofReconstructor::buildStack(Memory& memoryBlock,
    const ExpressionWithValidity& provedIn,
    std::vector<std::vector<std::string>>& stack,
    std::set<ExpressionWithValidity>& covered,
    bool rowsSurviveOnFailure, bool preserveScope) {
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
    if (ScopeSyntax::namesHypotheticalSentinel(StrSpan(provedIn.validityName))) {
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
    const ExpressionWithValidity proved = preserveScope ? provedIn
        : liftToShallowestOriginAncestor(memoryBlock, provedIn);

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
        // "__contradiction__" + gl::DefinitionExpansion::negate(proved) — and resolve locally there.
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
                && contradictionLbHoldsRecord(*contraLB, proved, preserveScope)) {
                // The twin's rows share this frame's fate: its verdict becomes
                // this frame's verdict, so whoever would discard ours discards
                // its too.
                return buildStack(*contraLB, proved, stack, covered,
                                  rowsSurviveOnFailure, preserveScope);
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
    // the `covered` dedup all key on the same lifted form: pass 0 cites every
    // dependency at its shallowest origin-bearing scope. A deeper-scope walk
    // (the retry of `buildChapterStack`) adds pass 1 once no candidate
    // succeeds in pass 0: a dependency may then be cited at a deeper scope
    // that holds its own records, down to the scope its record names
    // (`originScopeAlternatives`), choices in lexicographic order, and a
    // deeper choice keeps its exact scope in the recursion.
    const int passes = g_deeperScopes ? 2 : 1;
    for (int pass = 0; pass < passes; ++pass)
    for (size_t idx : candidateOrder) {
        const auto& origin = origins[idx];
        // A zoom-in record keeps its dependencies' exact scopes, and so does
        // a goal deposited at an induction scope's root, whose dependency is
        // the same goal at the leaf (D-519), and a
        // statement promoted from an int scope to its parent, whose dependency
        // is the same statement at the scope (D-578):
        // lifting it would cite the row itself.
        const bool zoomIn = origin.first == "zoom in" || origin.first == "induction scope"
            || origin.first == "int scope";
        if (pass == 1 && zoomIn) continue;

        std::vector<ExpressionWithValidity> liftedDeps;
        liftedDeps.reserve(origin.second.size());
        std::vector<std::vector<ExpressionWithValidity>> scopeChoices;
        std::vector<std::size_t> choice;
        if (pass == 0) {
            for (const auto& d : origin.second) {
                liftedDeps.push_back(zoomIn ? d
                    : liftToShallowestOriginAncestor(memoryBlock, d));
            }
        }
        else {
            scopeChoices.reserve(origin.second.size());
            for (const auto& d : origin.second) {
                scopeChoices.push_back(originScopeAlternatives(memoryBlock, d));
            }
            choice.assign(origin.second.size(), 0);
        }
        const auto nextChoice = [&]() {
            for (std::size_t k = choice.size(); k > 0; --k) {
                if (++choice[k - 1] < scopeChoices[k - 1].size()) return true;
                choice[k - 1] = 0;
            }
            return false;
        };
        // Pass 0 makes one attempt; pass 1 tries every combination after the
        // all-shallowest one, which pass 0 already tried.
        bool more = pass == 0 || nextChoice();
        bool skipOrigin = false;
        for (; more && !skipOrigin; more = pass == 1 && nextChoice()) {
            if (pass == 1) {
                liftedDeps.clear();
                for (std::size_t k = 0; k < scopeChoices.size(); ++k) {
                    liftedDeps.push_back(scopeChoices[k][choice[k]]);
                }
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
                skipOrigin = true;
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
            for (std::size_t k = 0; k < liftedDeps.size(); ++k) {
                const ExpressionWithValidity& ingredient = liftedDeps[k];
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
                                    /*rowsSurviveOnFailure=*/false,
                                    /*preserveScope=*/zoomIn
                                        || (pass == 1 && choice[k] != 0))) {
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
    }

    // D-51: no acyclic direct origin worked. Try the contradiction-LB
    // fallback before falling back to the degraded front()-emit. Walk the
    // LB chain for "__contradiction__" + gl::DefinitionExpansion::negate(proved) — either head
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
            && contradictionLbHoldsRecord(*contraLB, proved, preserveScope)) {
            if (insertedHere) g_buildStackPath.erase(proved);
            // Inherits this frame's fate, exactly as the entry-side site.
            const bool twinVerdict = buildStack(*contraLB, proved, stack, covered,
                                                rowsSurviveOnFailure, preserveScope);
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
        const bool exactScopes = fb.first == "zoom in" || fb.first == "induction scope"
            || fb.first == "int scope";
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
            liftedDeps.push_back(exactScopes ? d
                : liftToShallowestOriginAncestor(memoryBlock, d));
        }
        emitRow(fb, liftedDeps);
        for (const auto& d : liftedDeps) {
            if (covered.insert(d).second) {
                g_coveredJournal.push_back(d);
                // Inherits: this frame ignores the verdict and never rolls
                // back, so the child's rows live exactly as long as ours.
                (void) buildStack(memoryBlock, d, stack, covered,
                                  rowsSurviveOnFailure, exactScopes);
            }
        }
    }
    g_failConflict = myConflict;
    g_failConsulted = myConsulted;
    g_failContextDependent = true;
    if (insertedHere) g_buildStackPath.erase(proved);
    return false;
}


/// @brief Order expressions by descending proof-stack size.
/// @details Sorts index/value pairs and projects the original expressions in that order; mismatched lengths retain the existing invalid-argument failure.
/// @param expressions Expressions to order.
/// @param values Corresponding proof-stack sizes.
/// @return Expressions in descending size order.
std::vector<ExpressionWithValidity> ProofReconstructor::sortByValuesDesc(const std::vector<ExpressionWithValidity>& expressions,
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
/// @brief Reset proof reconstruction state for a new export.
/// @details Clears the existing thread-local path, journal and failed-walk memo at the export boundary.
/// @return No value.
void ProofReconstructor::clearBuildStackPath() {
    g_buildStackPath.clear();
    g_coveredJournal.clear();
    g_noGood.clear();
    g_failConflict.clear();
    g_failConsulted.clear();
    g_failContextDependent = false;
}

/// @brief Order candidate origins for the reconstruction walk.
/// @details Preserves insertion order within non-equality and equality groups.
/// @param origins Candidate origin records.
/// @return Ordered candidate indices.
std::vector<size_t> ProofReconstructor::sortedOriginIndicesD49(
    const std::vector<std::pair<std::string, std::vector<ExpressionWithValidity>>>& origins) {
    std::vector<size_t> result;
    result.reserve(origins.size());
    auto isEq = [&](size_t i) {
        return origins[i].first == "equality1" || origins[i].first == "equality2";
    };
    for (size_t i = 0; i < origins.size(); ++i) if (!isEq(i)) result.push_back(i);
    for (size_t i = 0; i < origins.size(); ++i) if (isEq(i)) result.push_back(i);
    return result;
}

/// @brief Find the shallowest origin scope allowed by the lift barriers.
/// @details Uses the existing origin lookup. Two barriers bound the lift: an
/// OR-branch scope (`orint_` / `ordis_`), which keeps every statement inside
/// its conditional branch, and a contradiction scope whose assumption is the
/// expression itself, which keeps that assumption cited where it is assumed.
/// The deepest barrier on the path wins.
/// @param mb Logical block containing the origin records.
/// @param v Expression and initial validity.
/// @return The existing expression with its selected validity.
ExpressionWithValidity ProofReconstructor::liftToShallowestOriginAncestor(
        const Memory& mb,
        const ExpressionWithValidity& v) {
    int64_t exactKey = 0;
    if (lookupOriginKey(mb.originInterner, v.original, v.validityName, exactKey)) {
        const int32_t exactId = mb.exprOriginMap.lookup(exactKey);
        if (exactId != 0) {
            for (const IdOrigin& origin : mb.exprOriginMap.recordsAt(exactId)) {
                if (origin.first == OriginTag::zoomIn) return v;
            }
        }
    }
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
        const StrSpan bare = gl::ScopeSyntax::stripSubproofPrefixView(
            StrSpan(child.data() + payloadStart,
                    static_cast<int32_t>(payloadLen)));
        if ((bare.len >= static_cast<int32_t>(kOrint.size())
             && std::memcmp(bare.ptr, kOrint.data(), kOrint.size()) == 0)
            || (bare.len >= static_cast<int32_t>(kOrdis.size())
                && std::memcmp(bare.ptr, kOrdis.data(), kOrdis.size()) == 0)) {
            minLift = i;
        }
        // A contradiction scope is a barrier for its own assumption. The
        // scope `contradiction_<P>` assumes the negation of P (its `task
        // formulation`), and its contradiction must derive from that
        // assumption, never from a copy an ancestor proves: citing the
        // ancestor's copy would leave the reductio without its premise
        // (D-561).
        const int32_t contradictionPrefixLength = static_cast<int32_t>(
            sizeof(gl::ScopeSyntax::kContradictionScopePrefix) - 1);
        if (bare.len > contradictionPrefixLength
            && std::memcmp(bare.ptr, gl::ScopeSyntax::kContradictionScopePrefix,
                           static_cast<std::size_t>(contradictionPrefixLength)) == 0) {
            const char* product = bare.ptr + contradictionPrefixLength;
            const std::size_t productLength =
                static_cast<std::size_t>(bare.len - contradictionPrefixLength);
            const std::string& expression = v.original;
            const bool assumption = product[0] == '!'
                ? expression.size() == productLength - 1
                    && std::memcmp(expression.data(), product + 1, productLength - 1) == 0
                : expression.size() == productLength + 1
                    && expression[0] == '!'
                    && std::memcmp(expression.data() + 1, product, productLength) == 0;
            if (assumption) minLift = i;
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

/// @brief List every scope at which an expression carries origin records,
///        from the shallowest permitted ancestor down to its own scope.
///
/// @details
/// The first entry is the scope `liftToShallowestOriginAncestor` selects:
/// the shallowest of `main` and the `_boundary_` prefixes of
/// `v.validityName` whose `(expression, scope)` key holds an origin record,
/// at or below the lift barriers; `v` itself for an exact `zoom in` record or
/// an expression without records. The deeper prefixes, ending with the full
/// validity, follow in order whenever their own key holds a record. A first
/// entry equal to `v` leaves nothing deeper.
///
/// Only the deeper-scope walk of `buildChapterStack` reads the deeper
/// entries: a statement derived at main through the very implication a
/// subproof is proving can still be cited inside that subproof through its
/// own records.
///
/// @param mb Logical block holding the origin records.
/// @param v  Expression and the scope its citing record names.
/// @return The origin-bearing scopes, shallowest first; never empty.
/// @see liftToShallowestOriginAncestor, buildChapterStack.
std::vector<ExpressionWithValidity> ProofReconstructor::originScopeAlternatives(
        const Memory& mb,
        const ExpressionWithValidity& v) {
    std::vector<ExpressionWithValidity> scopes{ liftToShallowestOriginAncestor(mb, v) };
    // Continue the ancestor split below the shallowest scope. That scope is
    // at or below the lift barriers, so every deeper prefix is too.
    const std::string& s = v.validityName;
    static const std::string sep = "_boundary_";
    const std::size_t shallowest = scopes.front().validityName.size();
    if (shallowest >= s.size()
        || s.compare(0, 4, "main") != 0
        || s.compare(4, sep.size(), sep) != 0) {
        return scopes;
    }
    for (std::size_t pos = 4 + sep.size(); pos <= s.size();) {
        const std::size_t next = s.find(sep, pos);
        const std::size_t end = next == std::string::npos ? s.size() : next;
        if (end > shallowest) {
            const std::string scope = s.substr(0, end);
            int64_t pk = 0;
            if (lookupOriginKey(mb.originInterner, v.original, scope, pk)) {
                const int32_t oid = mb.exprOriginMap.lookup(pk);
                if (oid != 0 && mb.exprOriginMap.runLen(oid) > 0) {
                    scopes.emplace_back(v.original, scope);
                }
            }
        }
        if (next == std::string::npos) break;
        pos = next + sep.size();
    }
    return scopes;
}
thread_local std::set<ExpressionWithValidity> ProofReconstructor::g_buildStackPath;
thread_local std::vector<ExpressionWithValidity> ProofReconstructor::g_coveredJournal;
thread_local std::map<WalkNode, NoGoodEntry> ProofReconstructor::g_noGood;
thread_local std::set<ExpressionWithValidity> ProofReconstructor::g_failConflict;
thread_local std::set<ExpressionWithValidity> ProofReconstructor::g_failConsulted;
thread_local bool ProofReconstructor::g_failContextDependent = false;
thread_local bool ProofReconstructor::g_deeperScopes = false;
/// @brief Reset the journal and failed-walk memo for a root traversal.
/// @details The export driver invokes this immediately before each top-level buildStack call.
/// @return No value.
void ProofReconstructor::beginRootTraversal() { g_coveredJournal.clear(); g_noGood.clear(); }
/// @brief Mark a chapter goal on the reconstruction path.
/// @details The path filter must reject origins that depend on the chapter goal.
/// @param goal Chapter expression and validity to mark or remove.
/// @return No value.
void ProofReconstructor::enterChapterGoal(const ExpressionWithValidity& goal) { g_buildStackPath.insert(goal); }
/// @brief Remove the chapter goal after its reconstruction.
/// @details The export driver removes the same marker after the top-level walk.
/// @param goal Chapter expression and validity to mark or remove.
/// @return No value.
void ProofReconstructor::leaveChapterGoal(const ExpressionWithValidity& goal) { g_buildStackPath.erase(goal); }

/// @brief Emit one chapter's rows for `proved`: the shallowest-scope walk,
///        and a deeper-scope walk only when that one finds no acyclic tree.
///
/// @details
/// The first walk is `buildStack` exactly as every chapter has always been
/// built: each dependency cited at its shallowest origin-bearing scope
/// (I-39). When it returns true, its rows are the chapter.
///
/// When it returns false (its rows are the degraded chapter with a residual
/// cycle), the walk is repeated from a fresh traversal with
/// `g_deeperScopes` set: a node whose candidates all fail at the shallowest
/// scopes may then cite a dependency at a deeper scope that holds its own
/// records (`originScopeAlternatives`). A statement that also holds at main,
/// derived there only through the implication a subproof is proving, is
/// then cited inside the subproof, where its own record derives it. Every
/// cited row is an origin record, so nothing is invented. An acyclic retry
/// replaces the chapter; a failed retry leaves the first walk's rows, so a
/// chapter without any acyclic derivation is emitted exactly as before.
///
/// @param memoryBlock The LB whose origin records justify `proved`.
/// @param proved      The chapter's head expression and validity.
/// @param stack       Receives the chapter rows; empty on entry.
/// @param covered     Receives the covered nodes of the emitted walk; empty
///                    on entry.
/// @return True when the emitted chapter is an acyclic derivation tree.
///
/// @invariant A chapter the first walk resolves is identical to the
///            `buildStack` chapter; `g_deeperScopes` is false on entry and
///            on return.
/// @see buildStack, originScopeAlternatives, beginRootTraversal.
bool ProofReconstructor::buildChapterStack(Memory& memoryBlock,
    const ExpressionWithValidity& proved,
    std::vector<std::vector<std::string>>& stack,
    std::set<ExpressionWithValidity>& covered) {
    assert(stack.empty() && covered.empty() && !g_deeperScopes);
    beginRootTraversal();
    if (buildStack(memoryBlock, proved, stack, covered)) return true;
    std::vector<std::vector<std::string>> retryStack;
    std::set<ExpressionWithValidity> retryCovered;
    beginRootTraversal();
    g_deeperScopes = true;
    const bool retried = buildStack(memoryBlock, proved, retryStack, retryCovered);
    g_deeperScopes = false;
    if (!retried) return false;
    stack.swap(retryStack);
    covered.swap(retryCovered);
    return true;
}

}
