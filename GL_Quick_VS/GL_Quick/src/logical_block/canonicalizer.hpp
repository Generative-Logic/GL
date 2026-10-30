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
#include "../parameters.hpp"
#include "../proof/worker_context.hpp"
#include "../infra/rt_tracker.hpp"
#include "canonical_member_change.hpp"
#include "canonical_run_overlay.hpp"
#include "canonical_form.hpp"
#include "resolved_statement_tokens.hpp"
namespace gl {
namespace tests { struct CanonicalizerFixture; }
/// @brief Build complete canonical class views for a logical scope.
/// @details Preserves class ordering, representative selection, history and recheck requests.
class Canonicalizer {
friend struct tests::CanonicalizerFixture;
public:
/// @brief Bind the existing canonicalization and history settings.
/// @details Scope and class storage remain on the supplied logical block.
/// @param parameters Read-only equivalence and provenance parameters.
/// @return Constructed canonicalizer.
Canonicalizer(const ProverParameters& parameters);

/// @brief Stage one scope whose complete class view changed for the
        ///        implication-gate recheck (D-406).
        ///
        /// @details
        /// Called by `rebuildCanonicalScope` whenever a scope's complete
        /// classes or a representative change. A class merge can make a
        /// parked gate known without any new row (a known row's member joins
        /// the gate's class) and can leave a park key non-canonical, so the
        /// drain re-keys and rechecks every park at a staged scope or below
        /// it. While no implication is parked nothing is staged: a later park
        /// reads the classes current at its own time.
        ///
        /// @param mb    Owning block.
        /// @param scope Scope whose complete class view changed.
        /// @invariant Appends to `pendingSubproofRechecks` only.
        /// @see drainSubproofArrivals, rekeySubproofPark.
void stageSubproofRecheck(Memory& mb, NameId scope) {
            if (mb.overallHashMemory.rejectedMapSubproof.empty()) return;
            mb.pendingSubproofRechecks.push_back(scope);
        }

// Turn optimizations OFF for just this section
//#pragma optimize("", off)

        /// @brief Preferred special witness in a class.
        /// @details Forced names precede integration names by numeric depth,
        /// then algebraic names, with decoded lexical ties. Ordinary names
        /// are outside this special-only filtering and weakness contract.
        /// @param src Member source (heap `EquivalenceClass` or zero-copy
        ///            `EquivalenceClassView`), read via `memberCount()`/`memberId(i)`.
        /// @param mb        Owning LB — NameMap + name-kind cache.
        /// @return The canonical member id, or 0 when the class has no
        ///         special member. 0 is a defined "no filtering applies"
        ///         state the callers branch on, not a failure.
        /// @see filterIterationsCore, `logical_block/canonicalizer.hpp::updateWeakVariables` —
        ///      the two consumers of this rule.
template <class ClassT>
        inline NameId firstSpecialMemberId(const ClassT& src,
            Memory& mb) {
            NameId best = 0;
            for (int32_t i = 0; i < src.memberCount(); ++i) {
                const NameId id = src.memberId(i);
                if (mb.eqClassNameCaches.kindOf(id, mb.nameMap) == NameKind::Normal) continue;
                if (best == 0 || compareWitnessNames(mb.nameMap.decodeView(id), mb.nameMap.decodeView(best)) < 0)
                    best = id;
            }
            return best;
    }

/// @brief Choose the single canonical representative of an
        ///        equivalence class for admission-map / rejected-map KEY
        ///        rewriting.
        ///
        /// @details The admission and rejection key hooks collapse a whole
        /// equivalence class onto ONE representative name when they rewrite a
        /// key — unlike the expression-side machinery, which enumerates every
        /// equivalent variant. This helper picks that one name.
        ///
        /// Only "strong" members take part: weak / renamed members at @p
        /// validityName are skipped inline — the `reduceEqClassIds` weak-filter
        /// fused into the tier scan, so no intermediate `strong` vector is
        /// allocated. Among the strong members the priority is:
        /// ordinary names, forced witnesses, integration witnesses by numeric
        /// depth, then algebraic witnesses, with decoded lexical ties.
        /// Goal sanitization and the registry filter apply the same comparator
        /// within their special-name domain.
        ///
        /// @param clss          Equivalence class to reduce to one name.
        /// @param memoryBlock   Owning LB — weak-variable set, NameMap, and
        ///                      the lazy name-kind cache (hence non-const).
        /// @param validityName  Scope at which weakness is judged.
        /// @return The chosen representative (an owned copy, safe across
        ///         later mints); an empty string when the class has no
        ///         strong member (an empty or all-weak class). The empty
        ///         result is a defined "nothing to canonicalise" state the
        ///         caller skips on, not a failure.
        /// @see reduceEqClassIds, applyEquivalenceClassToAdmissionMap,
        ///      applyEquivalenceClassToRejectedMap.
template <class ClassT>
        inline std::string chooseCanonical(
            const ClassT& clss,
            Memory& memoryBlock,
            const std::string& validityName) {
            const NameId valId = memoryBlock.nameMap.lookup(validityName);
            NameId chosen = 0;
            for (int32_t i = 0; i < clss.memberCount(); ++i) {
                const NameId id = clss.memberId(i);
                if (valId != 0 && memoryBlock.intWeakVariables.contains(packStatementKey(id, valId))) continue;
                if (chosen == 0 || compareWitnessNames(memoryBlock.nameMap.decodeView(id),
                        memoryBlock.nameMap.decodeView(chosen)) < 0) chosen = id;
            }
            if (chosen == 0) return std::string();
            return std::string(memoryBlock.nameMap.decodeView(chosen).ptr,
                static_cast<size_t>(memoryBlock.nameMap.decodeView(chosen).len));
    }

/// @brief Choose the canonical representative's id — the id-returning
        ///        twin of `chooseCanonical`.
        ///
        /// @details
        /// Runs the identical weak-filtered tier scan `chooseCanonical`
        /// performs and returns the chosen member id directly, before the
        /// `nameMap.decode` copy. Scope-weak members are excluded. The common
        /// witness comparator orders class, numeric integration depth, then
        /// decoded lexical ties. An empty or all-weak class has no selection.
        ///
        /// @param clss          Equivalence class to reduce to one member.
        /// @param memoryBlock   Owning LB — weak-variable set, NameMap, and the
        ///                      lazy name-kind cache (hence non-const).
        /// @param validityName  Scope at which weakness is judged.
        /// @return The chosen representative's id, or `0` when there is no
        ///         strong member (an empty or all-weak class).
        /// @see chooseCanonical — the string-returning sibling;
        ///      reduceEqClassIds; firstSpecialMemberId.
template <class ClassT>
        inline NameId chooseCanonicalId(
            const ClassT& clss,
            Memory& memoryBlock,
            StrSpan validityName) {
            return chooseCanonicalId(clss, memoryBlock,
                memoryBlock.nameMap.lookup(validityName));
        }

/// @brief `chooseCanonicalId` for a scope given by its NameMap id.
        ///
        /// @details
        /// The same weak-filtered tier scan as the span overload, which
        /// forwards here with `nameMap.lookup(validityName)`. Callers that
        /// already hold the scope's id — the canonical door walks every class
        /// of one scope, the canonical rebuild every component of one scope —
        /// pass it and skip the per-class lookup of the scope name. `valId == 0`
        /// (a never-interned scope) disables the weak filter, as the span
        /// overload does for such a name.
        ///
        /// @param clss        Equivalence class to reduce to one member.
        /// @param memoryBlock Owning LB — weak-variable set, NameMap, and the
        ///                    lazy name-kind cache (hence non-const).
        /// @param valId       NameMap id of the scope at which weakness is
        ///                    judged; 0 when the scope name was never interned.
        /// @return The chosen representative's id, or `0` when there is no
        ///         strong member (an empty or all-weak class).
        /// @see chooseCanonicalId (span overload), chooseCanonical.
template <class ClassT>
        inline NameId chooseCanonicalId(
            const ClassT& clss,
            Memory& memoryBlock,
            NameId valId) {
            NameId chosen = 0;
            for (int32_t i = 0; i < clss.memberCount(); ++i) {
                const NameId id = clss.memberId(i);
                if (valId != 0 && memoryBlock.intWeakVariables.contains(packStatementKey(id, valId))) continue;
                if (chosen == 0 || compareWitnessNames(memoryBlock.nameMap.decodeView(id),
                        memoryBlock.nameMap.decodeView(chosen)) < 0) chosen = id;
            }
            return chosen;
    }

/// @brief Merge two equivalence classes that share at least one
        /// variable — fold all of `classB`'s state into `classA`.
        ///
        /// @details
        /// Triggered when a new equality reveals that two existing
        /// classes are actually the same equivalence relation. Under
        /// [I-33](../../docs/agentic_swdd/30_invariants.md#i-33):
        /// - The merge is *cross-vN preconditional*: when the two
        ///   classes live at different validity scopes, the merge
        ///   direction must be ancestor-only (the deeper-scoped class
        ///   inherits the ancestor-scoped class's content, not the
        ///   other way around).
        /// - The eqArgs-subset assert is *same-vN only*: the per-pair
        ///   eqArgs must form a subset of the variable set, but only
        ///   for classes at the same validity scope. Cross-scope
        ///   merges may legitimately have eqArgs that go outside
        ///   either class's variable set during the merge transient.
        ///
        /// Origin entries from `classB.equalityOriginMap` are appended
        /// into `classA.equalityOriginMap` via `addOrigin`, respecting
        /// the cap-full preference policy. Cross-pair `equality2` history
        /// cites each bridge equality at the scope where its body origin row
        /// lives; an ancestor class can contribute one side of a branch merge.
        ///
        /// @param classA  Target — the in-flight merged-class arena accumulator
        ///                (`MergeClassAccum`); absorbs `classB` in place.
        /// @param classB  Source — the absorbed class read through its cold blob
        ///                `EquivalenceClassView` (never mutated).
        /// @param mb      Owning LB.
        /// @param level   LB level at which the merge happens.
        /// @invariant [I-33](../../docs/agentic_swdd/30_invariants.md#i-33) —
        ///            cross-vN preconditions.
        /// @see [`docs/agentic_swdd/20_core_concepts/05_equivalence_classes.md`](../../docs/agentic_swdd/20_core_concepts/05_equivalence_classes.md).
inline void mergeTwoEquivalenceClasses(MergeClassAccum& classA,
            const EquivalenceClassView& classB,
            const NameId* eqArgIds,
            const int32_t eqArgN,
            const int* levels, int32_t levelCount,
            Memory& memoryBlock,
            // note: validityName / classBValidityName are BORROWED spans (a caller passing a temporary std::string would dangle).
            StrSpan validityName,
            StrSpan classBValidityName) {
            const int cap = (parameters.compressor_mode
                ? parameters.compressor_max_origins_per_expr
                : parameters.max_origin_per_expr);

            // Membership helpers — classA is the arena accumulator (members via
            // its run), classB the cold blob view (members via memberId).
            const auto viewHasId = [](const EquivalenceClassView& v, NameId id) {
                for (int32_t i = 0; i < v.memberCount(); ++i)
                    if (v.memberId(i) == id) return true;
                return false;
            };
            const auto accumHasId = [](const MergeClassAccum& a, NameId id) {
                for (int32_t i = 0; i < a.members.size(); ++i)
                    if (a.members[i] == id) return true;
                return false;
            };
            const auto bSubsetA = [&]() {              // classB subset of classA ?
                for (int32_t i = 0; i < classB.memberCount(); ++i)
                    if (!accumHasId(classA, classB.memberId(i))) return false;
                return true;
            };
            const auto aSubsetB = [&]() {              // classA subset of classB ?
                for (int32_t i = 0; i < classA.members.size(); ++i)
                    if (!viewHasId(classB, classA.members[i])) return false;
                return true;
            };
            const auto eqArgsSubsetB = [&]() {         // eqArgIds subset of classB ?
                for (int32_t i = 0; i < eqArgN; ++i)
                    if (!viewHasId(classB, eqArgIds[i])) return false;
                return true;
            };

            // Validate classB's vN relationship (see the invariant notes on the
            // former heap signature): sameVN, or classB at a STRICT ANCESTOR
            // (D-44). Descendant-direction merge is invalid (D-33 symmetry).
            const bool sameVN = equalSpans(classBValidityName, validityName);
            if (!sameVN) {
                assert(memoryBlock.nameMap.isStrictAncestor(
                           memoryBlock.nameMap.lookup(classBValidityName),
                           memoryBlock.nameMap.lookup(validityName))
                    && "cross-vN merge: classB's scope must be a strict "
                       "ancestor of the entry's scope");
                // Cross-vN early exit: ancestor class already a subset of the
                // descendant mergedClass — nothing to add (I-31: classB stays
                // at its scope; classA's origins unaffected).
                if (bSubsetA()) return;
            }

            // A subset of B: members B union A, levels (B | A), origins
            // (B-base + A). The accumulator holds A, so `mergeLevelsBBase` keeps
            // A's keys + adds B-only keys (overwriteLevels REPLACE), and
            // `mergeOriginsBBase` seeds B then appends A below cap.
            if (aSubsetB()) {
                classA.unionMembersFromView(classB, memoryBlock.nameMap);
                classA.mergeLevelsBBase(classB);
                classA.mergeOriginsBBase(classB, cap);
                return;
            }

            // Same-vN contract: eqArgs must NOT be subset of classB (otherwise
            // classB would already represent the equality and the subset path
            // above would have triggered, since classA's variables include
            // eqArgs by construction at first iteration; a later iteration
            // where classA has grown past being a subset of classB while
            // classB still has both eqArgs is forbidden in same-NS by the
            // overlap-iteration discipline). Cross-vN (ancestor) MAY have
            // eqArgs ⊆ classB — the equality is established at the ancestor
            // scope (e.g. via mail or prior derivation), and the descendant
            // is admitting it again. The merged-pair logic below handles the
            // multi-bridge case by picking commonArg deterministically.
            if (sameVN) {
                assert(!eqArgsSubsetB());
            }

            // common_arg := ((eq_args ∩ A) ∩ B), the lex-min bridge (eqArgIds is
            // decoded-lex, so the first qualifying id is lex-smallest). Same-vN
            // must have exactly one bridge; cross-vN one or two (D-44).
            NameId commonArgId = 0;
            {
                int bridgeCount = 0;
                for (int32_t i = 0; i < eqArgN; ++i) {
                    const NameId eqArgId = eqArgIds[i];
                    if (accumHasId(classA, eqArgId) && viewHasId(classB, eqArgId)) {
                        if (bridgeCount == 0) commonArgId = eqArgId;
                        ++bridgeCount;
                    }
                }
                if (bridgeCount == 0) {
                    // No equality-argument bridge: the two classes overlap
                    // through a shared MEMBER — the same-scope fixpoint pass of
                    // updateEquivalenceClasses
                    // (I-218).
                    // Bridge = the decoded-lex-min shared member
                    // (memberIds are decoded-lex, I-88), deterministic.
                    assert(sameVN
                        && "mergeTwoEquivalenceClasses: a member bridge is same-scope only");
                    for (int32_t ai = 0; ai < classA.members.size() && commonArgId == 0; ++ai)
                        if (viewHasId(classB, classA.members[ai])) commonArgId = classA.members[ai];
                    assert(commonArgId != 0
                        && "mergeTwoEquivalenceClasses: classes overlap neither through an equality argument nor through a member");
                } else if (sameVN) {
                    assert(bridgeCount == 1);
                }
            }
            // decodeView span — used only in the history spans below. No
            // NameMap mint fires anywhere in the pair loop (the only mints are
            // originInterner via the origin doors + mergedAccum's arena, a
            // DIFFERENT interner), so this span aliases NameMap's cold bytes
            // safely across the whole nested loop (I-3).
            const StrSpan commonSpan = memoryBlock.nameMap.decodeView(commonArgId);

            // Build the merged-pair levels + equality2 origin history in a
            // SECOND accumulator — the former heap `mergedMap` / `mergedOriginMap`
            // temporaries. The member loops run in memberIds storage order
            // (decoded-lex), the emission order the exprOriginMap depends on.
            const unsigned mergedSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            ScratchArena& mergedArena = genScratchArenas().forSlot(mergedSlot);
            MergeClassAccum mergedAccum(&mergedArena);

            for (int32_t ai = 0; ai < classA.members.size(); ++ai) {
                const NameId idA = classA.members[ai];
                if (idA == commonArgId) continue;
                const StrSpan varASpan = memoryBlock.nameMap.decodeView(idA); // no NameMap mint in loop (I-3)

                for (int32_t bi = 0; bi < classB.memberCount(); ++bi) {
                    const NameId idB = classB.memberId(bi);
                    if (idB == commonArgId) continue;
                    const StrSpan varBSpan = memoryBlock.nameMap.decodeView(idB); // no NameMap mint in loop (I-3)

                    // new_levels = levels ∪ classA{commonArg,idA} ∪ classB{commonArg,idB}
                    // — folded into one ascending-unique int32 run on the
                    // byte-bump tier of mergedAccum's arena, under a per-pair
                    // ScratchScope. setLevel copies the run into the page-tier
                    // levelPool, so the scope reclaims it before the next pair
                    // (byte-bump and page tiers coexist on one arena, pitfall 2).
                    // std::set<int> == sort + drop-adjacent-equal, so byte-exact.
                    {
                        const int32_t li = classA.levelIndex(
                            packEqPairKey(commonArgId, idA));
                        const int32_t aStart = (li >= 0) ? classA.levelStart[li] : 0;
                        const int32_t aLen = (li >= 0) ? classA.levelLen[li] : 0;
                        const char* lb = nullptr; int32_t lc = 0;
                        classB.findLevels(packEqPairKey(commonArgId, idB), lb, lc);
                        const int32_t total = levelCount + aLen + lc;
                        ScratchScope levScope(mergedArena);
                        int32_t* run = nullptr;
                        int32_t m = 0;
                        if (total > 0) {
                            run = reinterpret_cast<int32_t*>(mergedArena.resolve(
                                mergedArena.alloc(
                                    total * static_cast<int32_t>(sizeof(int32_t)),
                                    static_cast<int32_t>(alignof(int32_t)))));
                            int32_t w = 0;
                            for (int32_t k = 0; k < levelCount; ++k)
                                run[w++] = levels[k];
                            for (int32_t k = 0; k < aLen; ++k)
                                run[w++] = classA.levelPool[aStart + k];
                            for (int32_t k = 0; k < lc; ++k)
                                run[w++] = EquivalenceClassView::rdI32(lb + 4 * k);
                            assert(w == total
                                && "mergeTwoEquivalenceClasses newLevels count drift");
                            std::sort(run, run + total);
                            for (int32_t k = 0; k < total; ++k)
                                if (m == 0 || run[k] != run[m - 1]) run[m++] = run[k];
                        }
                        mergedAccum.setLevel(packEqPairKey(idA, idB), run, m);
                    }

                    // Cross-pair equality2 emission gate (I-32) — same strings +
                    // gate as the heap form; the `mergedOriginMap` / `classB`
                    // origin probes become the accumulator / view `hasOrigin`.
                    // The `exprOriginMap` side-effect writes are preserved.
                    if (parameters.trackHistory) {
                        // L3: the equality2 goal forms `(=[X,Y])` AND the RECORD
                        // antecedents are built fresh on the per-slot
                        // string-scratch arena via the eqSpan builder (explicit
                        // length — never a bare-array StrSpan), fed to the
                        // span-record origin doors. varASpan / varBSpan /
                        // commonSpan are NameMap decodeView spans (no NameMap
                        // mint in the loop) and validityName is a caller-stable
                        // StrSpan param; the doors mint into originInterner /
                        // mergedAccum's genScratchArenas, never this arena, so
                        // every span stays live across the four door calls (I-3).
                        const unsigned oSlot = (gl::WorkerContext::coreId() >= 0)
                            ? static_cast<unsigned>(gl::WorkerContext::coreId())
                            : scratchArenas().slotCount() - 1;
                        ScratchArena& oArena = scratchArenas().forSlot(oSlot);
                        ScratchScope oScope(oArena);
                        const auto eqSpan = [&](const StrSpan x,
                                                const StrSpan y) -> StrSpan {
                            const int32_t len = 3 + x.len + 1 + y.len + 2;
                            char* buf = oArena.allocBytes(len);
                            int32_t at = 0;
                            buf[at++] = '('; buf[at++] = '='; buf[at++] = '[';
                            std::memcpy(buf + at, x.ptr,
                                        static_cast<std::size_t>(x.len));
                            at += x.len;
                            buf[at++] = ',';
                            std::memcpy(buf + at, y.ptr,
                                        static_cast<std::size_t>(y.len));
                            at += y.len;
                            buf[at++] = ']'; buf[at++] = ')';
                            assert(at == len);
                            return StrSpan(buf, len);
                        };
                        const StrSpan vnSpan(validityName);
                        const StrSpan eq1s = eqSpan(varASpan, varBSpan);
                        const StrSpan eq2s = eqSpan(varBSpan, varASpan);
                        const StrSpan or1Left = eqSpan(varASpan, commonSpan);
                        const StrSpan or1Right = eqSpan(commonSpan, varBSpan);
                        const StrSpan or2Left = eqSpan(varBSpan, commonSpan);
                        const StrSpan or2Right = eqSpan(commonSpan, varASpan);
                        StrSpan classAncestors[ExecutionParameters::MAX_SCOPE_DEPTH];
                        const int32_t classAncestorCount =
                            memoryBlock.nameMap.strictAncestorSpans(
                                validityName, classAncestors,
                                ExecutionParameters::MAX_SCOPE_DEPTH);
                        const auto citationScopeFor = [&](StrSpan equality) {
                            const auto originRowAt = [&](StrSpan scope) {
                                int64_t pk = 0;
                                if (!lookupOriginKey(memoryBlock.originInterner,
                                                     equality, scope, pk)) {
                                    return false;
                                }
                                const int32_t oid =
                                    memoryBlock.exprOriginMap.lookup(pk);
                                return oid != 0
                                    && memoryBlock.exprOriginMap.runLen(oid) > 0;
                            };
                            if (originRowAt(validityName)) return vnSpan;
                            for (int32_t ai = classAncestorCount - 1;
                                 ai >= 0; --ai) {
                                if (originRowAt(classAncestors[ai])) {
                                    return classAncestors[ai];
                                }
                            }
                            assert(false
                                && "mergeTwoEquivalenceClasses: bridge equality has no body origin row at the merge scope or any ancestor");
                            return vnSpan;
                        };
                        const OriginDep or1Deps[2] = {
                            { or1Left, citationScopeFor(or1Left) },
                            { or1Right, citationScopeFor(or1Right) } };
                        const OriginDep or2Deps[2] = {
                            { or2Left, citationScopeFor(or2Left) },
                            { or2Right, citationScopeFor(or2Right) } };

                        auto alreadyKnown = [&](const StrSpan original) -> bool {
                            int64_t pk = 0;
                            if (!lookupOriginKey(memoryBlock.originInterner,
                                    original, validityName, pk)) {
                                return false;
                            }
                            if (mergedAccum.hasOrigin(pk)) return true;
                            if (classB.hasOrigin(pk)) return true;
                            const int32_t oid = memoryBlock.exprOriginMap.lookup(pk);
                            if (oid != 0
                                && memoryBlock.exprOriginMap.runLen(oid) > 0) return true;
                            return false;
                        };

                        if (!alreadyKnown(eq1s)) {
                            mergedAccum.addOriginEnc(memoryBlock.originInterner,
                                eq1s, vnSpan, OriginTag::equality2,
                                or1Deps, 2, cap);
                            addOriginEncoded(memoryBlock.exprOriginMap,
                                memoryBlock.originInterner, eq1s, vnSpan,
                                OriginTag::equality2, or1Deps, 2, cap);
                        }
                        if (!alreadyKnown(eq2s)) {
                            mergedAccum.addOriginEnc(memoryBlock.originInterner,
                                eq2s, vnSpan, OriginTag::equality2,
                                or2Deps, 2, cap);
                            addOriginEncoded(memoryBlock.exprOriginMap,
                                memoryBlock.originInterner, eq2s, vnSpan,
                                OriginTag::equality2, or2Deps, 2, cap);
                        }
                    }
                }
            }

            // classA := members (B ∪ A); levels (mergedMap | (B | A)); origins
            // (mergedOriginMap-base + (B-base + A)). Each fold keeps A's keys /
            // appends the source's — mergeLevelsBBase/FromAccum (REPLACE),
            // mergeOriginsBBase/FromAccum (MERGE below cap).
            classA.unionMembersFromView(classB, memoryBlock.nameMap);
            classA.mergeLevelsBBase(classB);
            classA.mergeLevelsFromAccum(mergedAccum);
            classA.mergeOriginsBBase(classB, cap);
            classA.mergeOriginsFromAccum(mergedAccum, cap);
        }

/// @brief Write every run a refresh held, in one pass over the class
        ///        store.
        ///
        /// @details One `assignAllRunsViaScratch` rebuild of
        /// `canonicalEquivalenceClassesMap`: held scopes receive their new
        /// runs, every other key keeps its run. The store is copied once
        /// instead of moving its pool tail once per rebuilt scope; the result
        /// is byte-identical to splicing the same runs one by one.
        /// @param mb Owning block.
        /// @param held The refresh's held runs; every scope already has a key.
        /// @param arena Scratch arena for the rebuild columns and record copies.
        /// @return (void)
        /// @see CanonicalRunOverlay, refreshCanonicalClasses.
inline void flushCanonicalRuns(Memory& mb, const CanonicalRunOverlay& held,
                                       ScratchArena& arena) {
            if (held.entryCount() == 0) return;
            RT_SCOPE_HERE("CANONICAL_RUNS_FLUSH");
            auto& store = mb.canonicalEquivalenceClassesMap;
            const int32_t keyCount = store.count();
            ScratchScope frame(arena);
            int32_t* entryOfKey = reinterpret_cast<int32_t*>(arena.resolve(arena.alloc(
                (keyCount + 1) * static_cast<int32_t>(sizeof(int32_t)), static_cast<int32_t>(alignof(int32_t)))));
            for (int32_t id = 0; id <= keyCount; ++id) entryOfKey[id] = -1;
            for (int32_t e = 0; e < held.entryCount(); ++e) {
                const int32_t id = store.lookup(held.scopeAt(e));
                assert(id != 0 && entryOfKey[id] < 0
                    && "flushCanonicalRuns: a held scope has no class-store key");
                entryOfKey[id] = e;
            }
            store.inner().assignAllRunsViaScratch(keyCount, [&](int32_t id, auto&& sink) {
                const int32_t e = entryOfKey[id];
                if (e < 0) { sink.unchanged(id); return; }
                for (int32_t r = 0; r < held.recordCount(e); ++r) {
                    ScratchScope recordFrame(arena);
                    const StrSpan record = held.copyRecord(e, r, arena);
                    sink.blob(record.ptr, static_cast<int32_t>(record.len));
                }
            }, arena);
        }

/// @brief Rebuild one scope's complete canonicalization classes.
        ///
        /// @details Combines its parent's maintained canonical classes with its
        /// own application classes. Overlaps close transitively; independent
        /// components remain separate. The write is one complete cold-map run.
        /// Only equality provenance is produced by the existing merge helper;
        /// application classes, their waterlines and their deltas are untouched.
        /// With @p held, a changed run is held there instead of written, and a
        /// parent held earlier in the same refresh is read from there. A
        /// changed view stages the scope for the implication-gate recheck
        /// (`stageSubproofRecheck`, D-406). Every member's record
        /// (`packCanonicalMemberRecord`) also keeps the locality of its rewrite
        /// equality; a changed locality is reported with change bit 16 and
        /// concerns only the OR caches
        /// (D-498).
        /// @param mb Owning block, with the parent canonical bucket initialized.
        /// @param scope Target validity identifier.
        /// @param fresh Optional output receiving only changes produced by this rebuild.
        /// @param held Optional runs of the current refresh, written later in
        ///        one pass by `flushCanonicalRuns`.
        /// @return Whether class contents or a selected representative changed.
        /// @invariant The result contains pairwise-disjoint components and keeps
        /// every equality's original citation scope and level information.
inline bool rebuildCanonicalScope(Memory& mb, NameId scope,
                PagedVector<CanonicalMemberChange>* fresh = nullptr,
                CanonicalRunOverlay* held = nullptr) {
            RT_SCOPE_HERE("CANONICAL_CLASS_VIEW");
            assert(scope != 0);
            const unsigned slot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : genScratchArenas().slotCount() - 1;
            ScratchArena& arena = genScratchArenas().forSlot(slot);
            ScratchScope frame(arena);
            const unsigned textSlot = gl::WorkerContext::coreId() >= 0
                ? static_cast<unsigned>(gl::WorkerContext::coreId()) : scratchArenas().slotCount() - 1;
            ScratchArena& strings = scratchArenas().forSlot(textSlot);
            ScratchScope textFrame(strings);
            const StrSpan scopeView = mb.nameMap.decodeView(scope);
            const ScratchString scopeText = ScratchString::copyFrom(strings, scopeView.ptr, scopeView.len);
            const NameId parent = mb.nameMap.parentOf(scope);
            const int32_t parentBucket = parent == 0 ? 0 : mb.canonicalEquivalenceClassesMap.lookup(parent);
            assert(parent == 0 || parentBucket != 0);
            const int32_t parentHeld = held == nullptr || parent == 0 ? -1 : held->find(parent);
            const int32_t ownBucket = mb.equivalenceClassesMap.lookup(scope);
            const int32_t parentCount = parentHeld >= 0 ? held->recordCount(parentHeld)
                : parentBucket == 0 ? 0 : mb.canonicalEquivalenceClassesMap.runLen(parentBucket);
            const int32_t ownCount = ownBucket == 0 ? 0 : mb.equivalenceClassesMap.runLen(ownBucket);
            const int32_t count = parentCount + ownCount;
            if (count == 0) {
                if (mb.canonicalEquivalenceClassesMap.lookup(scope) != 0) return false;
                mb.canonicalEquivalenceClassesMap.inner().assignRun(scope, nullptr, nullptr, 0);
                return true;
            }
            auto* inputs = reinterpret_cast<StrSpan*>(arena.resolve(arena.alloc(count * sizeof(StrSpan), alignof(StrSpan))));
            auto* outputs = reinterpret_cast<StrSpan*>(arena.resolve(arena.alloc(count * sizeof(StrSpan), alignof(StrSpan))));
            auto* used = reinterpret_cast<uint8_t*>(arena.resolve(arena.alloc(count, alignof(uint8_t))));
            std::memset(used, 0, count);
            int32_t inputCount = 0;
            const auto snapshot = [&](auto& map, int32_t bucket, int32_t n) {
                for (int32_t i = 0; i < n; ++i) {
                    int32_t length = 0;
                    const char* bytes = map.peekRecordBytes(bucket, i, length, arena);
                    char* copy = arena.resolve(arena.alloc(length, 1));
                    std::memcpy(copy, bytes, length);
                    inputs[inputCount++] = StrSpan(copy, length);
                }
            };
            snapshot(mb.equivalenceClassesMap, ownBucket, ownCount);
            if (parentHeld >= 0) {
                for (int32_t i = 0; i < parentCount; ++i)
                    inputs[inputCount++] = held->copyRecord(parentHeld, i, arena);
            } else {
                snapshot(mb.canonicalEquivalenceClassesMap, parentBucket, parentCount);
            }
            assert(inputCount == count);
            const int cap = parameters.compressor_mode
                ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr;
            int32_t outputCount = 0, byteCount = 0;
            for (int32_t seed = 0; seed < count; ++seed) {
                if (used[seed]) continue;
                MergeClassAccum merged(&arena);
                const EquivalenceClassView first{inputs[seed].ptr, inputs[seed].len};
                merged.unionMembersFromView(first, mb.nameMap);
                merged.mergeLevelsBBase(first);
                merged.mergeOriginsBBase(first, cap);
                used[seed] = 1;
                bool grew = true;
                while (grew) {
                    grew = false;
                    for (int32_t i = 0; i < count; ++i) {
                        if (used[i]) continue;
                        const EquivalenceClassView next{inputs[i].ptr, inputs[i].len};
                        NameId common = 0, extra = 0;
                        for (int32_t m = 0; m < merged.members.size(); ++m) {
                            const NameId member = merged.members[m];
                            if (classHasMember(next, member)) {
                                if (common == 0) common = member;
                            } else if (extra == 0) extra = member;
                        }
                        if (common == 0) continue;
                        NameId bridge[2] = {common, extra};
                        const int32_t bridgeCount = extra == 0 ? 1 : 2;
                        if (bridgeCount == 2 && compareSpans(mb.nameMap.decodeView(bridge[0]),
                            mb.nameMap.decodeView(bridge[1])) > 0) std::swap(bridge[0], bridge[1]);
                        mergeTwoEquivalenceClasses(merged, next, bridge, bridgeCount,
                            nullptr, 0, mb, StrSpan(scopeText), StrSpan(scopeText));
                        used[i] = 1;
                        grew = true;
                    }
                }
                // Previously documented equality paths stay available when a
                // complete component is rebuilt from its application sources.
                // History is carried only for writing; membership never reads it.
                const int32_t previousBucket = mb.canonicalEquivalenceClassesMap.lookup(scope);
                const int32_t previousCount = previousBucket == 0 ? 0 : mb.canonicalEquivalenceClassesMap.runLen(previousBucket);
                for (int32_t prior = 0; prior < previousCount; ++prior) {
                    int32_t priorLength = 0;
                    const char* priorBytes = mb.canonicalEquivalenceClassesMap.peekRecordBytes(previousBucket, prior, priorLength, arena);
                    const EquivalenceClassView old{priorBytes, priorLength};
                    bool included = true;
                    for (int32_t member = 0; member < old.memberCount() && included; ++member) {
                        bool found = false;
                        for (int32_t k = 0; k < merged.members.size() && !found; ++k)
                            found = merged.members[k] == old.memberId(member);
                        included = found;
                    }
                    if (included) merged.mergeOriginsBBase(old, cap);
                }
                assert(merged.members.size() >= 2
                    && "rebuildCanonicalScope: a stored equivalence class has two or more members");
                const int32_t length = merged.serializedSize();
                char* bytes = arena.resolve(arena.alloc(length, 1));
                assert(merged.serializeInto(bytes) == length);
                outputs[outputCount++] = StrSpan(bytes, length);
                byteCount += length;
                merged.forEachOriginSorted([&](int64_t key, auto lines) {
                    exprOriginRunReplace(mb.exprOriginMap, key, lines, cap, arena);
                });
            }
            std::sort(outputs, outputs + outputCount, [&](StrSpan a, StrSpan b) {
                const EquivalenceClassView left{a.ptr, a.len}, right{b.ptr, b.len};
                assert(left.memberCount() > 0 && right.memberCount() > 0);
                return compareSpans(mb.nameMap.decodeView(left.memberId(0)),
                    mb.nameMap.decodeView(right.memberId(0))) < 0;
            });
            const int32_t oldBucket = mb.canonicalEquivalenceClassesMap.lookup(scope);
            const int32_t oldCount = oldBucket == 0 ? 0 : mb.canonicalEquivalenceClassesMap.runLen(oldBucket);
            bool classChanged = oldBucket == 0 || oldCount != outputCount;
            bool representativeChanged = false;
            NameId chain[ExecutionParameters::MAX_SCOPE_DEPTH];
            const int32_t chainCount = equalityScopeChain(mb, scope, chain);
            for (int32_t i = 0; i < outputCount; ++i) {
                const EquivalenceClassView next{outputs[i].ptr, outputs[i].len};
                bool sameComponent = false;
                for (int32_t j = 0; j < oldCount && !sameComponent; ++j) {
                    int32_t length = 0;
                    const char* previous = mb.canonicalEquivalenceClassesMap.peekRecordBytes(oldBucket, j, length, arena);
                    const EquivalenceClassView old{previous, length};
                    if (old.memberCount() != next.memberCount()) continue;
                    bool sameMembers = true;
                    for (int32_t k = 0; k < next.memberCount(); ++k)
                        if (old.memberId(k) != next.memberId(k)) { sameMembers = false; break; }
                    if (!sameMembers) continue;
                    sameComponent = length == outputs[i].len && std::memcmp(previous, outputs[i].ptr, length) == 0;
                }
                if (!sameComponent) classChanged = true;
                const NameId normal = chooseCanonicalId(next, mb, scope);
                const NameId special = firstSpecialMemberId(next, mb);
                // Whether some member equality of this class is a local row,
                // computed once for the members without a direct row.
                int classLocal = -1;
                for (int32_t k = 0; k < next.memberCount(); ++k) {
                    const NameId member = next.memberId(k);
                    const int64_t key = packStatementKey(member, scope);
                    // The locality the rewrite of this member to its normal
                    // representative carries (appliedEqualityIsLocal); the
                    // statement rows it reads change without a class change.
                    bool local = false;
                    if (normal != 0 && member != normal) {
                        RT_SCOPE_HERE("CANONICAL_MEMBER_LOCALITY");
                        const int direct = directEqualityVerdict(mb, mb.nameMap.decodeView(member),
                            mb.nameMap.decodeView(normal), chain, chainCount, strings);
                        if (direct >= 0) {
                            local = direct == 1;
                        } else {
                            if (classLocal < 0)
                                classLocal = classHasLocalEqualityRow(mb, next, chain, chainCount, strings) ? 1 : 0;
                            local = classLocal == 1;
                        }
                    }
                    const int64_t selected = packCanonicalMemberRecord(normal, special, local);
                    const int64_t* prior = mb.lbMemory.canonicalMemberRepresentatives.find(key);
                    const int64_t previous = prior == nullptr ? 0 : *prior;
                    const bool normalMoved = prior == nullptr || canonicalRecordNormal(previous) != normal;
                    const bool specialMoved = prior == nullptr || canonicalRecordSpecial(previous) != special;
                    int32_t flags = !sameComponent ? 4 : 0;
                    if (prior == nullptr) flags |= 2;
                    if (normalMoved) flags |= 1;
                    if (specialMoved) flags |= 8;
                    if (prior != nullptr && canonicalRecordLocal(previous) != local) flags |= 16;
                    if (prior == nullptr || previous != selected) {
                        mb.lbMemory.canonicalMemberRepresentatives.upsert(key, selected);
                        if (normalMoved || specialMoved) representativeChanged = true;
                    }
                    // A locality-only change concerns the OR caches alone; the
                    // view's other consumers keep their change set.
                    const int32_t viewFlags = flags & ~16;
                    if (viewFlags != 0) {
                        const int32_t* previousFlags = mb.lbMemory.canonicalClassChanges.find(key);
                        const int32_t combined = viewFlags | (previousFlags == nullptr ? 0 : *previousFlags);
                        mb.lbMemory.canonicalClassChanges.upsert(key, combined);
                        auto& uses = mb.lbMemory.canonicalUses;
                        assert(uses.version < INT32_MAX);
                        uses.memberVersions.upsert(key, ++uses.version);
                        if ((viewFlags & 1) != 0) {
                            RT_SCOPE_HERE("CANONICAL_COMPARISON_INVALIDATION");
                            uses.invalidateComparisonMember(scope, mb.nameMap.decodeView(member));
                        }
                    }
                    if (flags != 0 && fresh != nullptr) fresh->push_back(CanonicalMemberChange{scope, member, flags});
                }
            }
            if (classChanged && held != nullptr) {
                held->add(scope, outputs, outputCount);
            } else if (classChanged) {
                mb.canonicalEquivalenceClassesMap.inner().assignRunGenerated(scope, outputCount, byteCount,
                    [&](auto&& sink) {
                        for (int32_t i = 0; i < outputCount; ++i) sink(outputs[i].ptr, outputs[i].len);
                    });
            }
            if (classChanged || representativeChanged) stageSubproofRecheck(mb, scope);
            return classChanged || representativeChanged;
        }

/// @brief Initialize a newly created scope's canonicalization classes.
        ///
        /// @details Initializes parent buckets first and then builds the target
        /// bucket. Existing buckets are maintained by equality updates, so this
        /// creation boundary never rebuilds them. Application buckets are untouched.
        /// @param mb Owning block.
        /// @param scope Scope already present in the validity forest.
        /// @return Nothing.
        /// @see rebuildCanonicalScope, updateEquivalenceClasses.
inline void initializeCanonicalScope(Memory& mb, NameId scope) {
            assert(scope != 0);
            if (mb.canonicalEquivalenceClassesMap.lookup(scope) != 0) return;
            const NameId parent = mb.nameMap.parentOf(scope);
            if (parent != 0) initializeCanonicalScope(mb, parent);
            rebuildCanonicalScope(mb, scope);
        }

private:
/// @brief Resolve a statement's member-walk tokens once.
        ///
        /// @details
        /// The token-to-id lookup does not depend on any class. This resolves
        /// the int section, the it section and the bracket tokens with one
        /// non-minting `NameMap::lookup` each, in walk order, for
        /// `canonicalFormAtScope`'s single token walk and for
        /// `forEachNonCanonicalMemberTokenCore` over any number of classes.
        /// Lookups have no side effect, so resolving them up front is
        /// unobservable.
        ///
        /// The special sections keep only scanned tokens that are whole
        /// bracket tokens of the statement. The scan is a substring search
        /// (`scanSpecialTokens`' contract, kept for the registry filter), so
        /// it also reports `it_0_lev_1_7` inside the longer name
        /// `it_0_lev_1_7_copy`. Such a fragment is no member occurrence: the
        /// door substitutes whole tokens only, and a pair collected for it
        /// would substitute nothing (the door's assert).
        ///
        /// @param scan      The statement's special-token scan.
        /// @param statement The statement bytes (bracket-token source).
        /// @param mb        Owning LB (NameMap).
        /// @param arena     Scratch arena receiving the three arrays; the caller's
        ///                  scope keeps them alive.
        /// @return The resolved runs.
        /// @see forEachNonCanonicalMemberTokenCore, ScannedTokenRuns.
inline ResolvedStatementTokens resolveStatementTokens(
            const SpecialTokenScanView& scan, StrSpan statement,
            Memory& mb, ScratchArena& arena) {
            ResolvedStatementTokens out;
            const auto allocate = [&](int32_t count) -> ResolvedToken* {
                if (count == 0) return nullptr;
                return reinterpret_cast<ResolvedToken*>(arena.resolve(arena.alloc(
                    count * static_cast<int32_t>(sizeof(ResolvedToken)),
                    static_cast<int32_t>(alignof(ResolvedToken)))));
            };
            const auto isBracketToken = [&](StrSpan token) -> bool {
                bool whole = false;
                collectExprTokens(statement, [&](StrSpan bracketToken) {
                    if (!whole && equalSpans(bracketToken, token)) whole = true;
                });
                return whole;
            };
            const auto fill = [&](SpecialTokenScanView::TokenCursor cursor,
                                  const ResolvedToken*& refs, int32_t& n) {
                StrSpan token;
                int32_t whole = 0;
                SpecialTokenScanView::TokenCursor counter = cursor;
                while (counter.next(token))
                    if (isBracketToken(token)) ++whole;
                ResolvedToken* slots = allocate(whole);
                while (cursor.next(token)) {
                    if (!isBracketToken(token)) continue;
                    assert(n < whole);
                    slots[n++] = ResolvedToken{ token, mb.nameMap.lookup(token) };
                }
                assert(n == whole);
                refs = slots;
            };
            fill(scan.intTokens(), out.intRefs, out.intCount);
            fill(scan.itTokens(), out.itRefs, out.itCount);
            int32_t normal = 0;
            collectExprTokens(statement, [&](StrSpan) { ++normal; });
            ResolvedToken* slots = allocate(normal);
            collectExprTokens(statement, [&](StrSpan token) {
                assert(out.normalCount < normal);
                slots[out.normalCount++] = ResolvedToken{ token, mb.nameMap.lookup(token) };
            });
            assert(out.normalCount == normal);
            out.normalRefs = slots;
            return out;
        }

private:
/// @brief Whether the equality a canonical rewrite applied is a LOCAL
        ///        statement of this LB — the locality the rewritten form
        ///        inherits.
        ///
        /// @details
        /// A rewrite `member -> canonical` is a derivation of this LB exactly
        /// when the equality that licenses it is this LB's own: the direct
        /// row `(=[member,canonical])` or its mirror `(=[canonical,member])`
        /// in `intKnownStatements` at the class scope or a strict ancestor,
        /// local. A pair joined only transitively (two members merged through
        /// a third) has no direct row; it is local iff any equality between
        /// two members of the class is a local row at those scopes — the
        /// class was formed here. All probes are non-minting; the transient
        /// equality texts ride @p arena under a scope popped on return.
        ///
        /// @param mb         Owning LB — statement registry + NameMap.
        /// @param from       The rewritten member's name.
        /// @param to         The canonical member's name.
        /// @param classScope The scope of the class the pair came from.
        /// @param cls        The class (member read interface).
        /// @param arena      String-tier scratch for the transient texts.
        /// @return `true` iff the applied equality is a local statement here.
        /// @see canonicalFormAtScope — fills `CanonicalForm::eqLocal`;
        ///      addExprToMemoryBlock — flips the deposit to local on it;
        ///      rebuildCanonicalScope — records it per member.
template <class ClassT>
        inline bool appliedEqualityIsLocal(Memory& mb, StrSpan from, StrSpan to,
                                           StrSpan classScope, const ClassT& cls,
                                           ScratchArena& arena) {
            const NameId scope = mb.nameMap.lookup(classScope);
            assert(scope != 0 && "appliedEqualityIsLocal: the class scope is interned");
            NameId chain[ExecutionParameters::MAX_SCOPE_DEPTH];
            const int32_t chainCount = equalityScopeChain(mb, scope, chain);
            const int direct = directEqualityVerdict(mb, from, to, chain, chainCount, arena);
            if (direct >= 0) return direct == 1;
            return classHasLocalEqualityRow(mb, cls, chain, chainCount, arena);
        }

/// @brief The scopes an equality's locality is read over — a scope and
        ///        every strict ancestor.
        ///
        /// @details
        /// A rewrite's equality row counts when it is known at the class scope
        /// or at any scope above it. The chain is collected by walking
        /// `NameMap::parentOf` to the root; the locality verdicts are
        /// order-free, so the walk order needs no sort. Reads only.
        ///
        /// @param mb    Owning LB — the validity forest.
        /// @param scope The class scope's NameMap id (nonzero).
        /// @param chain Caller stack array of `MAX_SCOPE_DEPTH` ids, filled
        ///              with @p scope first.
        /// @return The number of ids written.
        /// @invariant The chain fits `MAX_SCOPE_DEPTH` (asserted, the I-140
        ///            depth tripwire).
        /// @see appliedEqualityIsLocal, rebuildCanonicalScope.
inline int32_t equalityScopeChain(Memory& mb, NameId scope, NameId* chain) {
            assert(scope != 0);
            int32_t count = 0;
            for (NameId at = scope; at != 0; at = mb.nameMap.parentOf(at)) {
                assert(count < ExecutionParameters::MAX_SCOPE_DEPTH
                    && "equalityScopeChain: scope deeper than MAX_SCOPE_DEPTH");
                chain[count++] = at;
            }
            return count;
        }

/// @brief Locality verdict of one equality row over a scope chain.
        ///
        /// @details
        /// Builds `(=[from,to])` on @p arena and looks it up without minting;
        /// a text never interned has no row. Statement rows keep their first
        /// locality (`upsertStatementKey`), so the verdict changes only when a
        /// row is added or erased. The transient text is released on return.
        ///
        /// @param mb         Owning LB — statement registry + NameMap.
        /// @param from       Left equality argument.
        /// @param to         Right equality argument.
        /// @param chain      Scope ids from `equalityScopeChain`.
        /// @param chainCount Number of ids in @p chain.
        /// @param arena      String-tier scratch for the equality text.
        /// @return 1 when a local row exists at a chain scope, 0 when only
        ///         non-local rows exist, -1 when there is no row.
        /// @see directEqualityVerdict, classHasLocalEqualityRow.
inline int equalityRowVerdict(Memory& mb, StrSpan from, StrSpan to,
            const NameId* chain, int32_t chainCount, ScratchArena& arena) {
            ScratchScope sc(arena);
            const int32_t n = from.len + to.len + 6;
            char* buf = arena.allocBytes(n);
            int32_t at = 0;
            std::memcpy(buf + at, "(=[", 3); at += 3;
            std::memcpy(buf + at, from.ptr, static_cast<size_t>(from.len)); at += from.len;
            buf[at++] = ',';
            std::memcpy(buf + at, to.ptr, static_cast<size_t>(to.len)); at += to.len;
            std::memcpy(buf + at, "])", 2); at += 2;
            assert(at == n && "equalityRowVerdict: equality fill diverged from its length");
            const NameId equality = mb.nameMap.lookup(StrSpan(buf, n));
            if (equality == 0) return -1;
            int verdict = -1;
            for (int32_t s = 0; s < chainCount; ++s) {
                const StatementFlags* flags = mb.intKnownStatements.find(StatementKey{ equality, chain[s] });
                if (flags == nullptr) continue;
                if (flags->local) return 1;
                verdict = 0;
            }
            return verdict;
        }

/// @brief Locality verdict of a rewrite pair's direct equality.
        ///
        /// @details
        /// The direct row `(=[from,to])` decides; without it its mirror
        /// `(=[to,from])` does. A pair with neither row was joined
        /// transitively and its locality is the class's
        /// (`classHasLocalEqualityRow`).
        ///
        /// @param mb         Owning LB — statement registry + NameMap.
        /// @param from       The rewritten member's name.
        /// @param to         The canonical member's name.
        /// @param chain      Scope ids from `equalityScopeChain`.
        /// @param chainCount Number of ids in @p chain.
        /// @param arena      String-tier scratch for the equality texts.
        /// @return 1 local, 0 non-local, -1 no direct row in either orientation.
        /// @see equalityRowVerdict, appliedEqualityIsLocal.
inline int directEqualityVerdict(Memory& mb, StrSpan from, StrSpan to,
            const NameId* chain, int32_t chainCount, ScratchArena& arena) {
            const int verdict = equalityRowVerdict(mb, from, to, chain, chainCount, arena);
            return verdict >= 0 ? verdict : equalityRowVerdict(mb, to, from, chain, chainCount, arena);
        }

/// @brief Whether any equality between two members of a class is a local
        ///        row over a scope chain — the locality of a transitive pair.
        ///
        /// @details
        /// A pair joined only through other members has no direct row; it is
        /// local exactly when the class was formed here, i.e. some equality
        /// between two of its members, in either orientation, is a local row
        /// at a chain scope. The verdict belongs to the class, so a caller
        /// that needs it for several members computes it once. Reads only.
        ///
        /// @param mb         Owning LB — statement registry + NameMap.
        /// @param cls        The class (member read interface).
        /// @param chain      Scope ids from `equalityScopeChain`.
        /// @param chainCount Number of ids in @p chain.
        /// @param arena      String-tier scratch for the equality texts.
        /// @return `true` iff a local member equality exists on the chain.
        /// @see equalityRowVerdict, appliedEqualityIsLocal, rebuildCanonicalScope.
template <class ClassT>
        inline bool classHasLocalEqualityRow(Memory& mb, const ClassT& cls,
            const NameId* chain, int32_t chainCount, ScratchArena& arena) {
            const int32_t n = cls.memberCount();
            for (int32_t i = 0; i < n; ++i) {
                const StrSpan a = mb.nameMap.decodeView(cls.memberId(i));
                for (int32_t j = i + 1; j < n; ++j) {
                    const StrSpan b = mb.nameMap.decodeView(cls.memberId(j));
                    if (equalityRowVerdict(mb, a, b, chain, chainCount, arena) == 1) return true;
                    if (equalityRowVerdict(mb, b, a, chain, chainCount, arena) == 1) return true;
                }
            }
            return false;
        }

private:
/// @brief The class of a scope's complete canonical view that holds one
        ///        member — a zero-copy view over its blob.
        ///
        /// @details
        /// Walks the scope's `canonicalEquivalenceClassesMap` bucket in class
        /// order and returns the first class whose members include @p member.
        /// Classes of one view are disjoint ([I-218](../../docs/agentic_swdd/30_invariants.md#i-218)),
        /// so the first hit is the only one. The caller holds a recorded
        /// representative for the member at this scope
        /// (`canonicalMemberRepresentatives`), which the complete-view rebuild
        /// writes together with the bucket, so a miss is a broken contract and
        /// asserts (Rule 19). A page straddle is assembled on @p arena's
        /// byte-bump tier; the view lives until the caller rewinds it.
        ///
        /// @param mb       Owning LB — the canonical class store.
        /// @param bucketId The scope's bucket id in `canonicalEquivalenceClassesMap`.
        /// @param member   The member's NameMap id.
        /// @param arena    Gen-scratch arena receiving a straddle copy.
        /// @return The class view containing @p member.
        /// @invariant The bucket holds a class with @p member (asserted).
        /// @see liveCanonicalMemberAt, canonicalFormAtScope — the consumers.
inline EquivalenceClassView canonicalClassContaining(Memory& mb,
            int32_t bucketId, NameId member, ScratchArena& arena) {
            assert(bucketId != 0 && member != 0);
            const int32_t classCount = mb.canonicalEquivalenceClassesMap.runLen(bucketId);
            for (int32_t ci = 0; ci < classCount; ++ci) {
                int32_t length = 0;
                const char* bytes = mb.canonicalEquivalenceClassesMap.peekRecordBytes(
                    bucketId, ci, length, arena);
                const EquivalenceClassView view{ bytes, length };
                if (classHasMember(view, member)) return view;
            }
            assert(false && "canonicalClassContaining: recorded member has no class in the scope's bucket");
            return EquivalenceClassView{ nullptr, 0 };
        }

private:
/// @brief The member every occurrence of a token rewrites to under a
        ///        scope's complete canonical view — the class's live normal
        ///        representative, read from the recorded representatives.
        ///
        /// @details
        /// The complete-view rebuild (`rebuildCanonicalScope`) writes, for every
        /// member of every class of the scope's view, the class's
        /// `chooseCanonicalId` result at that scope into the high half of
        /// `canonicalMemberRepresentatives`, in the same pass that writes the
        /// bucket. Class membership changes only through that rebuild; weak
        /// variables only grow between rebuilds. The live chooser therefore
        /// still selects the recorded member unless that member turned weak at
        /// the scope since the rebuild (a weak update in a scope whose refresh
        /// was skipped); only then the token's class is located in the bucket
        /// and chosen live. A token in no class of the view, or whose class had
        /// no strong member, returns 0 — exactly the tokens the per-class walk
        /// of `forEachNonCanonicalMemberTokenCore` leaves unchanged. Reads
        /// only; no interner is minted.
        ///
        /// @param mb       Owning LB — recorded representatives, weak set, classes.
        /// @param member   The token's NameMap id (nonzero).
        /// @param scope    The scope's NameMap id (nonzero).
        /// @param bucketId The scope's bucket id in `canonicalEquivalenceClassesMap`.
        /// @param arena    Gen-scratch arena for the class peek of the live choice.
        /// @return The canonical member id (possibly @p member itself), or 0 when
        ///         the token is in no class of the view or its class has no strong member.
        /// @invariant Equals `chooseCanonicalId` over the bucket class holding
        ///            @p member at @p scope, or 0 when there is no such class.
        /// @see canonicalClassContaining, canonicalFormAtScope, chooseCanonicalId.
inline NameId liveCanonicalMemberAt(Memory& mb, NameId member, NameId scope,
            int32_t bucketId, ScratchArena& arena) {
            assert(member != 0 && scope != 0 && bucketId != 0);
            const int64_t* recorded = mb.lbMemory.canonicalMemberRepresentatives.find(
                packStatementKey(member, scope));
            if (recorded == nullptr) return 0;
            const NameId normal = canonicalRecordNormal(*recorded);
            if (normal == 0) return 0;
            if (!mb.intWeakVariables.contains(packStatementKey(normal, scope))) return normal;
            return chooseCanonicalId(canonicalClassContaining(mb, bucketId, member, arena), mb, scope);
        }

public:
/// @brief The canonical door's rewrite — one statement to its
        ///        class-canonical form under the equivalence classes at its
        ///        complete target-scope view, walking each class with the SAME
        ///        per-token member rule the registry's `filterIterationsCore`
        ///        judges with.
        ///
        /// @details
        /// Every member token of a class at @p validityName that is not
        /// that class's `chooseCanonicalId` representative (normal name >
        /// `force_lev_*` / `int_<depth>_lev_*` > `it_*_lev_*`, weak members at the scope skipped,
        /// lex-min within a tier) is rewritten to it — special tiers from
        /// the token scan (whole bracket tokens only, `resolveStatementTokens`),
        /// normal names from `collectExprTokens` over the statement's brackets. The maintained target-scope bucket already
        /// includes all visible ancestor classes, even before the first local
        /// equality. Scope creation and equality updates consolidate it before
        /// canonicalization; raw scope identities remain stored independently.
        ///
        /// Ordinary callers leave equalities and negated equalities unchanged
        /// (class-defining facts — the `addEquality` door and the
        /// I-12 one-sided expansion own them, goal closure is literal, and
        /// `applyEquiClasses` never multiplies either shape). OR matching opts
        /// into equality views so refuters and OR leaves share a normal form.
        /// The shape gate always preserves anchor
        /// predicates ([I-53](../../docs/agentic_swdd/30_invariants.md#i-53)
        /// — an anchor is a scope identity). Rule carriers, negated compact
        /// existences and or compacts canonicalize like any statement (the
        /// rule install's own gate then sees a canonical carrier; a cohort
        /// opens on the canonical signature).
        ///
        /// The token scan is a memo-independent `buildSpecialTokenScanView`
        /// over the raw bytes (a deposit may not be interned in the LB's
        /// NameMap). The scan's special tokens and the statement's bracket
        /// tokens are resolved to NameMap ids once per call
        /// (`resolveStatementTokens`) and walked once, in the order and with
        /// the per-token rule of `forEachNonCanonicalMemberTokenCore` (special
        /// runs with the tier assert, bracket tokens of normal kind). Each
        /// token's class representative comes from the recorded
        /// representatives (`liveCanonicalMemberAt`): one probe per token
        /// instead of a membership walk and a representative choice for every
        /// class of the view. A pair's class view is located in the bucket
        /// only when evidence is collected. For each applied pair the class's per-pair levels
        /// (`forEachLevel(packEqPairKey(member, canonical))`) are unioned
        /// into the caller's ascending-unique level run
        /// (`insertLevelSorted` — negatives skipped), the levels-gate
        /// eq-rewrite level-union discipline; the `{-1}` non-derived tier
        /// is transparent to the union — the leading marker is stripped
        /// before real levels merge in and the run re-enters as the
        /// `{-1}` singleton only when every input was non-derived (a
        /// level run is never mixed — the `addStatement` door's I-182
        /// assert). Interior is heap-free: the scan blob, class peeks, and
        /// pair staging ride a gen-scratch window reclaimed on return; the
        /// result text and justifiers ride @p outArena. No interner mint
        /// happens inside, so `decodeView` spans are I-3-safe throughout.
        ///
        /// @param statement    The statement's full expression bytes.
        /// @param validityName The deposit's scope name.
        /// @param mb           Owning LB — classes, NameMap, kind cache.
        /// @param outArena     String-tier arena receiving the canonical
        ///                     text + justifier builds (must be a different
        ///                     arena family than gen-scratch).
        /// @param lvRun        Caller's ascending-unique level run, unioned
        ///                     in place with the applied pairs' levels.
        /// @param levelCount   Run length of @p lvRun, updated in place.
        /// @param levelCap     Capacity of @p lvRun (Rule-19 tripwire).
        /// @param rewriteEqualities Opt into equality and inequality normal
        ///                     forms for OR views and their selected twins.
        /// @param collectEvidence Collect supporting levels, justifiers and locality.
        ///                     False requests text only and requires a null, empty
        ///                     level buffer; the chooser and substitution order agree.
        /// @param collectLevels Union proof-support levels when evidence is collected.
        ///                     False retains justifiers and locality but requires a
        ///                     null, empty level buffer for lookup-only consumers.
        /// @return See `CanonicalForm`; `changed == false` leaves
        ///         @p lvRun untouched. Text-only results contain no equality bridge.
        /// @invariant A collected member pair always substitutes (every
        ///            token occurrence in the gated shapes is
        ///            `[`/`,`-boundary-delimited), asserted — a silent
        ///            no-op substitution would admit the raw member.
        /// @see forEachNonCanonicalMemberTokenCore — the shared walk core;
        ///      resolveStatementTokens; filterIterationsCore — the refusing twin;
        ///      addExprToMemoryBlock — the one caller.
inline CanonicalForm canonicalFormAtScope(
            StrSpan statement, StrSpan validityName, Memory& mb,
            ScratchArena& outArena, int* lvRun, int32_t& levelCount,
            int32_t levelCap, bool rewriteEqualities = false,
            bool collectEvidence = true, bool collectLevels = true) {
            assert((collectEvidence && collectLevels)
                || (lvRun == nullptr && levelCount == 0 && levelCap == 0));
            CanonicalForm res{};
            res.text = statement;

            if (!rewriteEqualities
                && (isEquality(statement) || isNegatedEquality(statement))) return res;
            {
                const bool negated = statement.len > 0 && statement.ptr[0] == '!';
                StrSpan body = statement;
                if (negated) body = StrSpan(body.ptr + 1, body.len - 1);
                if (startsWithSpan(body, "(Anchor", 7)
                    || startsWithSpan(body, "Anchor", 6)) return res;
            }

            const NameId vid = mb.nameMap.lookup(validityName);
            if (vid == 0) return res;   // never interned -> no classes there
            const int32_t bucketId = mb.canonicalEquivalenceClassesMap.lookup(vid);
            if (bucketId == 0) return res;
            const int32_t cc = mb.canonicalEquivalenceClassesMap.runLen(bucketId);
            if (cc == 0) return res;

            const unsigned gSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            ScratchArena& gArena = genScratchArenas().forSlot(gSlot);
            ScratchScope gScope(gArena);

            const SpecialTokenScanView scan =
                buildSpecialTokenScanView(statement, gArena);
            const ResolvedStatementTokens tokens =
                resolveStatementTokens(scan, statement, mb, gArena);

            // One record per applied substitution pair.
            struct CanonPair {
                StrSpan from;
                StrSpan to;
                NameId fromId;
                NameId canonId;
            };
            CanonPair cp[ExecutionParameters::MAX_ARITY];
            int32_t pairN = 0;

            // The per-token rule of forEachNonCanonicalMemberTokenCore with each
            // token's class representative read from the recorded
            // representatives (liveCanonicalMemberAt) instead of walking every
            // class of the view: the same pairs, the same kind probes.
            const auto visit = [&](StrSpan token, NameId tokenId, bool special,
                                   NameKind tier) {
                if (tokenId == 0) return false;            // never interned -> not a member
                const NameId canonId = liveCanonicalMemberAt(mb, tokenId, vid, bucketId, gArena);
                if (canonId == 0) return false;            // no class, or no strong member
                if (special) {
                    assert(mb.eqClassNameCaches.kindOf(tokenId, mb.nameMap) == tier
                        || (tier == NameKind::IntLev && mb.eqClassNameCaches.kindOf(tokenId, mb.nameMap) == NameKind::Forced));
                } else if (mb.eqClassNameCaches.kindOf(tokenId, mb.nameMap) != NameKind::Normal) {
                    return false;                          // special tiers walked above
                }
                if (tokenId == canonId) return false;
                for (int32_t k = 0; k < pairN; ++k)
                    if (cp[k].fromId == tokenId) return false;
                assert(pairN < ExecutionParameters::MAX_ARITY
                    && "canonicalFormAtScope: substitution pairs exceed MAX_ARITY");
                cp[pairN++] = CanonPair{ token, mb.nameMap.decodeView(canonId), tokenId, canonId };
                return false;                              // collect all, never stop
            };
            tokens.forEachInt([&](StrSpan token, NameId tokenId) {
                return visit(token, tokenId, true, classifyName(token));
            });
            tokens.forEachIt([&](StrSpan token, NameId tokenId) {
                return visit(token, tokenId, true, NameKind::ItLev);
            });
            tokens.forEachNormal([&](StrSpan token, NameId tokenId) {
                return visit(token, tokenId, false, NameKind::Normal);
            });
            if (pairN == 0) return res;

            // Ascending member-name order — the house justifier order
            // (Pattern of the applyEquivalenceClass eqBuf sort); keys are
            // distinct so the order is total.
            std::sort(cp, cp + pairN,
                [](const CanonPair& a, const CanonPair& b) {
                    return compareSpans(a.from, b.from) < 0;
                });

            StrReplacement pairs[ExecutionParameters::MAX_ARITY];
            for (int32_t k = 0; k < pairN; ++k) {
                pairs[k].key = cp[k].from;
                pairs[k].value = cp[k].to;
            }
            const ScratchString canon =
                replaceKeysScratch(outArena, statement, pairs, pairN);
            assert(!equalSpans(StrSpan(canon), statement)
                && "canonicalFormAtScope: collected member pair produced no boundary substitution");

            res.text = StrSpan(canon);
            res.changed = true;
            if (!collectEvidence) return res;

            // The {-1} non-derived tier is TRANSPARENT to the union (the
            // enumerateEqClassRewrites sink's discipline): strip the
            // leading non-derived marker before real pair levels merge in
            // — a level run is either exactly {-1} or all-values-≥0,
            // never mixed (the addStatement door asserts it) — and
            // restore the singleton below if every input was non-derived.
            if (collectLevels) {
                int32_t lead = 0;
                while (lead < levelCount && lvRun[lead] < 0) ++lead;
                if (lead > 0) {
                    for (int32_t j = lead; j < levelCount; ++j)
                        lvRun[j - lead] = lvRun[j];
                    levelCount -= lead;
                }
            }
            for (int32_t k = 0; k < pairN; ++k) {
                const EquivalenceClassView cls =
                    canonicalClassContaining(mb, bucketId, cp[k].fromId, gArena);
                if (collectLevels) {
                    cls.forEachLevel(packEqPairKey(cp[k].fromId, cp[k].canonId),
                        [&](int lv) {
                            levelCount = insertLevelSorted(
                                lvRun, levelCount, lv, levelCap);
                        });
                }
                const int32_t n = cp[k].from.len + cp[k].to.len + 6;
                char* buf = outArena.allocBytes(n);
                int32_t at = 0;
                std::memcpy(buf + at, "(=[", 3); at += 3;
                std::memcpy(buf + at, cp[k].from.ptr,
                            static_cast<size_t>(cp[k].from.len));
                at += cp[k].from.len;
                buf[at++] = ',';
                std::memcpy(buf + at, cp[k].to.ptr,
                            static_cast<size_t>(cp[k].to.len));
                at += cp[k].to.len;
                std::memcpy(buf + at, "])", 2); at += 2;
                assert(at == n
                    && "canonicalFormAtScope: justifier fill diverged from its length");
                res.eqLocal[res.eqN] = appliedEqualityIsLocal(
                    mb, cp[k].from, cp[k].to, validityName, cls, outArena);
                res.eqJust[res.eqN++] = StrSpan(buf, n);
            }
            if (collectLevels && levelCount == 0) {
                // Every input was non-derived — the run re-enters the
                // {-1} singleton (ground material, the addStatement
                // door's own empty-run normalization).
                lvRun[0] = -1;
                levelCount = 1;
            }

            res.text = StrSpan(canon);
            res.changed = true;
            return res;
        }

public:
/// @brief Test the expression spelling for equality.
/// @details Uses prefix and delimiter checks.
/// @param cand Expression text to inspect.
/// @return Whether the expression has the requested form.
inline bool isEquality(const std::string& cand) {
            return cand.size() >= 3 &&
                cand[0] == '(' &&
                cand[1] == '=' &&
                cand[2] == '[';
        }

public:
/// @brief Span twin of `isEquality(const std::string&)` — reports
        ///        whether `cand` opens with the positive-equality prefix `(=[`.
        ///
        /// @details
        /// Byte-identical to the `std::string` overload: only the first three
        /// bytes are inspected, so a `StrSpan` view over the stable input lets a
        /// span-holding caller (the statified `addStatement` dispatch and the
        /// equivalence-class hooks) test the shape without materialising a
        /// `std::string`. The `cand.len >= 3` guard short-circuits before any
        /// `operator[]`, mirroring the original's `cand.size() >= 3` guard, so
        /// the two forms agree on every input including the sub-length-3 cases.
        ///
        /// @param cand Candidate expression view.
        /// @return `true` when `cand` begins with `(=[`.
        /// @see isEquality(const std::string&) — the byte-exact string twin,
        ///      kept as the differential-test oracle.
inline bool isEquality(StrSpan cand) {
            return cand.len >= 3 &&
                cand[0] == '(' &&
                cand[1] == '=' &&
                cand[2] == '[';
        }

public:
/// @brief Test the expression spelling for negated equality.
/// @details Uses prefix and delimiter checks.
/// @param cand Expression text to inspect.
/// @return Whether the expression has the requested form.
inline bool isNegatedEquality(const std::string& cand) {
            return cand.size() >= 4 &&
                cand[0] == '!' &&
                cand[1] == '(' &&
                cand[2] == '=' &&
                cand[3] == '[';
        }

public:
/// @brief Span twin of `isNegatedEquality(const std::string&)` —
        ///        reports whether `cand` opens with the negated-equality
        ///        prefix `!(=[`.
        ///
        /// @details
        /// Byte-identical to the `std::string` overload: only the first four
        /// bytes are inspected, so a `StrSpan` view over the stable input lets a
        /// span-holding caller test the shape with no `std::string`
        /// materialisation. The `cand.len >= 4` guard short-circuits before any
        /// `operator[]`, mirroring the original's `cand.size() >= 4` guard, so
        /// the two forms agree on every input including the sub-length-4 cases.
        ///
        /// @param cand Candidate expression view.
        /// @return `true` when `cand` begins with `!(=[`.
        /// @see isNegatedEquality(const std::string&) — the byte-exact string
        ///      twin, kept as the differential-test oracle.
inline bool isNegatedEquality(StrSpan cand) {
            return cand.len >= 4 &&
                cand[0] == '!' &&
                cand[1] == '(' &&
                cand[2] == '=' &&
                cand[3] == '[';
        }

private:
const ProverParameters& parameters;
};
}

namespace gl {
/// @brief Bind the existing canonicalization and history settings.
/// @details Scope and class storage remain on the supplied logical block.
/// @param parameters Read-only equivalence and provenance parameters.
/// @return Constructed canonicalizer.
inline Canonicalizer::Canonicalizer(const ProverParameters& parameters)
 : parameters(parameters) {}
} // namespace gl
