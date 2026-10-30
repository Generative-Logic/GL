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
#include "variable_copy_detector.hpp"
#include "../compiler/expression_parser.hpp"
namespace gl {
/// @brief Detect the antisymmetry conjecture shape and name the
    ///        variable to duplicate — the fourth variable-copy trigger.
    ///
    /// @details
    /// An antisymmetry-shaped conjecture carries two positive premises of
    /// the same operator whose argument lists are equal except exactly two
    /// positions holding the same two arguments swapped
    /// (`P[..,x,..,y,..]` / `P[..,y,..,x,..]`), with an equality head over
    /// exactly that argument pair. Proving such a head forces downstream
    /// rule instances to bind two template-distinct slots to one name —
    /// the collision pattern the exact normalized-key request match cannot
    /// assemble. The cure is a dead-end variable-copy axiom
    /// `(=[x,x_copy])` at the innermost premise LB: the equivalence class
    /// then generates all-distinct statement variants and the standard
    /// machinery (demand markers, witness revival, rule firings) closes
    /// the proof.
    ///
    /// This function is the pure detection half: it scans every ordered
    /// pair of chain elements for the shape and, on the first match,
    /// reports the swapped argument that appears first left-to-right in
    /// the earlier premise — the variable the caller duplicates. Both
    /// verdicts are defined results of the detection contract (most
    /// conjectures simply do not have the shape).
    ///
    /// @param chain      The disintegrated premise chain of the conjecture
    ///                   (raw element strings, e.g.
    ///                   `"(preorder[1,4,7,8])"`).
    /// @param head       The conjecture head (e.g. `"(=[7,8])"`).
    /// @param copyVarOut On detection, the variable to duplicate (the
    ///                   swapped argument at the earlier mismatch position
    ///                   of the earlier element); untouched otherwise.
    /// @return True iff the antisymmetry shape was detected.
    /// @see ProofGridBuilder::addTheoremToMemory — the deposit site;
    ///      IntegrationProcessor::checkNecessityForEquality and
    ///      LogicBlockOperations::reactToHypo — sibling variable-copy
    ///      emission sites sharing the `variableCopy` origin contract.
bool VariableCopyDetector::detectAntisymmetryCopyVar(const std::vector<std::string>& chain,
                               const std::string& head,
                               std::string& copyVarOut) {
    if (head.empty() || head[0] == '!') return false;
    if (ce::extractExpression(head) != "=") return false;
    const std::vector<std::string> headArgs = ce::getArgs(head);
    if (headArgs.size() != 2 || headArgs[0] == headArgs[1]) return false;

    for (std::size_t i = 0; i < chain.size(); ++i) {
        const std::string& p1 = chain[i];
        if (p1.empty() || p1[0] == '!') continue;
        for (std::size_t j = i + 1; j < chain.size(); ++j) {
            const std::string& p2 = chain[j];
            if (p2.empty() || p2[0] == '!') continue;
            if (ce::extractExpression(p1) != ce::extractExpression(p2)) continue;
            const std::vector<std::string> a1 = ce::getArgs(p1);
            const std::vector<std::string> a2 = ce::getArgs(p2);
            if (a1.size() != a2.size() || a1.empty()) continue;

            int p = -1, q = -1;
            bool tooMany = false;
            for (std::size_t k = 0; k < a1.size(); ++k) {
                if (a1[k] == a2[k]) continue;
                if (p < 0) p = static_cast<int>(k);
                else if (q < 0) q = static_cast<int>(k);
                else { tooMany = true; break; }
            }
            if (tooMany || q < 0) continue;

            const std::string& x = a1[static_cast<std::size_t>(p)];
            const std::string& y = a1[static_cast<std::size_t>(q)];
            if (a2[static_cast<std::size_t>(p)] != y
             || a2[static_cast<std::size_t>(q)] != x) continue;
            if (x.find('(') != std::string::npos
             || y.find('(') != std::string::npos) continue;

            const bool headMatches =
                (headArgs[0] == x && headArgs[1] == y) ||
                (headArgs[0] == y && headArgs[1] == x);
            if (!headMatches) continue;

            copyVarOut = x;
            return true;
        }
    }
    return false;
}
/// @brief Detect the output-collision conjecture shape and name the
    ///        variable to duplicate — the fifth variable-copy trigger.
    ///
    /// @details
    /// Two positive premises of the same operator carrying the same name
    /// in the operator's single output slot while their input slots
    /// differ (every other slot byte-equal) — the MPL encoding of an
    /// equality between two compound terms. The caller deposits the
    /// dead-end axiom `(=[r,r_copy])` at the SECOND colliding premise LB
    /// so the equivalence class can generate the all-distinct statement
    /// variants request matching needs while foreign conjectures sharing
    /// only the first colliding premise as a chain prefix never receive
    /// the copy. The gate is exactly-two premises by design; a third
    /// premise matching the same signature suppresses it, and an
    /// anchor-argument shared name never fires (theory constant — the
    /// I-24 analogue). Full contract at the definition.
    ///
    /// @param chain         Disintegrated premise chain of the conjecture.
    /// @param coreMap       Compiled operator configurations (slot
    ///                      classification source).
    /// @param copyVarOut    On detection, the shared output name;
    ///                      untouched otherwise.
    /// @param firstIndexOut On detection, the chain index of the second
    ///                      colliding premise (the deposit LB); untouched
    ///                      otherwise.
    /// @return True iff the output-collision shape was detected.
    /// @see gl::detectAntisymmetryCopyVar — sibling trigger sharing the
    ///      `variableCopy` deposit contract;
    ///      ProofGridBuilder::addTheoremToMemory — the deposit site.
bool VariableCopyDetector::detectOutputCollisionCopyVar(const std::vector<std::string>& chain,
                                  const ce::CoreExpressionMap& coreMap,
                                  std::string& copyVarOut,
                                  std::size_t& firstIndexOut) {
    for (std::size_t i = 0; i < chain.size(); ++i) {
        const std::string& p1 = chain[i];
        if (p1.empty() || p1[0] == '!') continue;
        const std::string op = ce::extractExpression(p1);
        const ce::CoreExpressionMap::const_iterator cfgIt = coreMap.find(op);
        if (cfgIt == coreMap.end()) continue;
        const ce::CoreExpressionConfig& cfg = cfgIt->second;
        if (cfg.outputIndices.size() != 1) continue;
        const int outIdx = cfg.outputIndices[0];
        const std::vector<std::string> a1 = ce::getArgs(p1);
        if (outIdx < 0 || outIdx >= static_cast<int>(a1.size())) continue;

        const auto isInputSlot = [&cfg](std::size_t k) -> bool {
            for (const int idx : cfg.inputIndices) {
                if (idx == static_cast<int>(k)) return true;
            }
            return false;
        };

        for (std::size_t j = i + 1; j < chain.size(); ++j) {
            const std::string& p2 = chain[j];
            if (p2.empty() || p2[0] == '!') continue;
            if (ce::extractExpression(p2) != op) continue;
            const std::vector<std::string> a2 = ce::getArgs(p2);
            if (a2.size() != a1.size()) continue;

            const std::string& r = a1[static_cast<std::size_t>(outIdx)];
            if (a2[static_cast<std::size_t>(outIdx)] != r) continue;
            if (r.find('(') != std::string::npos) continue;
            assert(r.rfind("u_", 0) != 0
                && "output-collision copy: theorem chains never carry u_ arguments");

            // ANCHOR GUARD (the I-24 analogue): never copy an anchor-slot
            // name. A shared output that is an anchor argument is a THEORY
            // CONSTANT (e.g. zero in `s(a)=0 ∧ s(b)=0 → a=b`), not a
            // theorem-bound result variable; duplicating a constant
            // variant-fans every statement carrying it across the grid and
            // breaches the one-changeable-arg integration contract.
            {
                bool rIsAnchorArg = false;
                for (std::size_t k2 = 0; k2 < chain.size() && !rIsAnchorArg;
                     ++k2) {
                    const std::string& el = chain[k2];
                    if (el.rfind("(Anchor", 0) != 0) continue;
                    const std::vector<std::string> aArgs = ce::getArgs(el);
                    for (const std::string& a : aArgs) {
                        if (a == r) { rIsAnchorArg = true; break; }
                    }
                }
                if (rIsAnchorArg) continue;
            }

            bool nonIoEqual = true;
            bool inputDiffers = false;
            for (std::size_t k = 0; k < a1.size(); ++k) {
                if (static_cast<int>(k) == outIdx) continue;
                if (isInputSlot(k)) {
                    if (a1[k] != a2[k]) inputDiffers = true;
                } else if (a1[k] != a2[k]) {
                    nonIoEqual = false;
                    break;
                }
            }
            if (!nonIoEqual || !inputDiffers) continue;

            // Exactly-two gate: a third positive premise matching the same
            // (operator, non-input/output args, output name) signature
            // suppresses the fire — narrow by design, extendable later.
            bool third = false;
            for (std::size_t k2 = 0; k2 < chain.size() && !third; ++k2) {
                if (k2 == i || k2 == j) continue;
                const std::string& p3 = chain[k2];
                if (p3.empty() || p3[0] == '!') continue;
                if (ce::extractExpression(p3) != op) continue;
                const std::vector<std::string> a3 = ce::getArgs(p3);
                if (a3.size() != a1.size()) continue;
                if (a3[static_cast<std::size_t>(outIdx)] != r) continue;
                bool nio = true;
                for (std::size_t k = 0; k < a1.size(); ++k) {
                    if (static_cast<int>(k) == outIdx || isInputSlot(k)) continue;
                    if (a1[k] != a3[k]) { nio = false; break; }
                }
                if (nio) third = true;
            }
            if (third) continue;

            copyVarOut = r;
            // Deposit at the SECOND colliding premise (maintainer decision
            // 2026-08-10): the first colliding premise is typically an
            // anchor-adjacent chain prefix SHARED by foreign conjectures,
            // and a deposit there mails the copy into their subtrees
            // (full-run Peano breach). Every consumer of the copy — the
            // deeper premise LBs, the contradiction twins under the
            // innermost premise, recursion auxiliaries — sits at or below
            // the second colliding premise, so mail still reaches them all.
            firstIndexOut = j;
            return true;
        }
    }
    return false;
}
/// @brief Detect the defined-set conjecture shape and name the number to
    ///        duplicate — the sixth variable-copy trigger
    ///        (D-599).
    ///
    /// @details
    /// The head is a property of one set D: D is the head's only argument
    /// besides the anchor's, at an input slot of the head's predicate. A
    /// positive premise defines D from one number n: D sits at a uniquely
    /// definable slot of that premise typed as a set of numbers (`P(1)`),
    /// and n is the premise's only other argument outside the anchor at an
    /// input slot typed as a number (`(1)`) — I1's `divisorSet[N,+,*,2,n,D]`
    /// with head `hasMinimum[N,+,D]`. A proof about D's members then
    /// applies rules at n with two of their variables bound to n (n ≤ n,
    /// n ∣ n for n ∈ D), which the exact normalized-key match cannot
    /// assemble while rule multiplication is off. The caller deposits the
    /// dead-end axiom `(=[n,n_copy])` at the innermost premise LB, and the
    /// equivalence class supplies the second name.
    ///
    /// Both verdicts are defined results of the detection contract; most
    /// conjectures do not have the shape.
    ///
    /// @param chain      Disintegrated premise chain of the conjecture, the
    ///                   anchor premise included.
    /// @param head       The conjecture head, possibly negated.
    /// @param coreMap    Compiled operator configurations (slot
    ///                   classification and definition sets).
    /// @param copyVarOut On detection, the number n to duplicate; untouched
    ///                   otherwise.
    /// @return True iff the defined-set shape was detected.
    /// @see gl::detectAntisymmetryCopyVar, gl::detectOutputCollisionCopyVar —
    ///      sibling triggers sharing the `variableCopy` deposit contract;
    ///      ProofGridBuilder::addTheoremToMemory — the deposit site.
bool VariableCopyDetector::detectDefinedSetCopyVar(const std::vector<std::string>& chain,
                                    const std::string& head,
                                    const ce::CoreExpressionMap& coreMap,
                                    std::string& copyVarOut) {
    // The anchor premise's arguments: the theory constants every lemma shares.
    std::vector<std::string> anchorArgs;
    for (const std::string& element : chain)
        if (element.rfind("(Anchor", 0) == 0) anchorArgs = ce::getArgs(element);
    const auto isAnchorArg = [&anchorArgs](const std::string& argument) {
        for (const std::string& a : anchorArgs)
            if (a == argument) return true;
        return false;
    };
    // The definition set a configuration gives the argument at position slot.
    const auto slotTyped = [](const ce::CoreExpressionConfig& cfg, int slot, const char* pattern) {
        const auto it = cfg.definitionSets.find(std::to_string(slot + 1));
        assert(it != cfg.definitionSets.end() && "defined-set copy: every configured slot has a definition set");
        return it->second.first == pattern;
    };
    const auto isIndexIn = [](const std::vector<int>& indices, int slot) {
        for (const int index : indices)
            if (index == slot) return true;
        return false;
    };

    // The head: a property of one set D, D at an input slot.
    const std::string headCore = (!head.empty() && head[0] == '!') ? head.substr(1) : head;
    const ce::CoreExpressionMap::const_iterator headCfg = coreMap.find(ce::extractExpression(headCore));
    if (headCfg == coreMap.end()) return false;
    const std::vector<std::string> headArgs = ce::getArgs(headCore);
    std::string set;
    int setSlot = -1;
    for (std::size_t k = 0; k < headArgs.size(); ++k) {
        if (isAnchorArg(headArgs[k])) continue;
        if (!set.empty() && headArgs[k] != set) return false;
        set = headArgs[k];
        setSlot = static_cast<int>(k);
    }
    if (set.empty() || !isIndexIn(headCfg->second.inputIndices, setSlot)) return false;
    assert(set.rfind("u_", 0) != 0 && "defined-set copy: theorem chains never carry u_ arguments");

    // A positive premise defining D from one number n.
    for (const std::string& premise : chain) {
        if (premise.empty() || premise[0] == '!' || premise.rfind("(Anchor", 0) == 0) continue;
        const ce::CoreExpressionMap::const_iterator cfgIt = coreMap.find(ce::extractExpression(premise));
        if (cfgIt == coreMap.end()) continue;
        const ce::CoreExpressionConfig& cfg = cfgIt->second;
        const std::vector<std::string> args = ce::getArgs(premise);
        bool defines = false;
        for (const int index : cfg.uniquelyDefinableIndices) {
            assert(index >= 0 && index < static_cast<int>(args.size())
                && "defined-set copy: a uniquely definable slot outside the arity");
            if (args[static_cast<std::size_t>(index)] == set && slotTyped(cfg, index, "P(1)")) defines = true;
        }
        if (!defines) continue;
        std::string number;
        bool oneNumber = true;
        for (const int index : cfg.inputIndices) {
            assert(index >= 0 && index < static_cast<int>(args.size())
                && "defined-set copy: an input slot outside the arity");
            const std::string& argument = args[static_cast<std::size_t>(index)];
            if (argument == set || isAnchorArg(argument) || !slotTyped(cfg, index, "(1)")) continue;
            if (!number.empty() && argument != number) oneNumber = false;
            number = argument;
        }
        if (number.empty() || !oneNumber) continue;
        copyVarOut = number;
        return true;
    }
    return false;
}
}
