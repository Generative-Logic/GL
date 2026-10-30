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
#include "or_elimination_builder.hpp"
namespace gl {
/// @brief Bind the or elimination builder operation inputs.
/// @details Construction stores explicit references; block storage and all invocation lifetimes remain with their owners.
/// @param definitions Borrowed CompiledDefinitionRegistry interface or state; must outlive this component.
/// @param fullTheoremList Borrowed std::vector interface or state; must outlive this component.
/// @param globalTheoremList Borrowed std::vector interface or state; must outlive this component.
/// @param orElimBuiltByPair Borrowed std::vector interface or state; must outlive this component.
/// @param orElimCitedOrByMerged Borrowed std::vector interface or state; must outlive this component.
/// @param orElimGuardIndex Borrowed std::vector interface or state; must outlive this component.
/// @param orElimInRunScannedRows Borrowed std::vector interface or state; must outlive this component.
/// @param orElimPendingPairs Borrowed std::vector interface or state; must outlive this component.
/// @param orRuleCompiler Borrowed OrRuleCompiler interface or state; must outlive this component.
/// @param parameters Borrowed ProverParameters interface or state; must outlive this component.
/// @param proofHistory Borrowed ProofHistory interface or state; must outlive this component.
/// @param theoremListMutex Borrowed std::mutex interface or state; must outlive this component.
/// @param theoremPublisher Borrowed TheoremPublisher interface or state; must outlive this component.
/// @return None; construction binds the supplied references.
OrEliminationBuilder::OrEliminationBuilder(const CompiledDefinitionRegistry& definitions, std::vector< std::tuple<std::string, std::string, std::string, std::string >>& fullTheoremList, std::vector< std::tuple<std::string, std::string, std::string, std::string >>& globalTheoremList, std::map<std::pair<std::string, std::string>, std::string>& orElimBuiltByPair, std::map<std::string, std::string>& orElimCitedOrByMerged, std::map<std::string, std::vector<std::pair<std::string, std::string>>>& orElimGuardIndex, std::unordered_set<std::string>& orElimInRunScannedRows, std::vector<std::array<std::string, 4>>& orElimPendingPairs, OrRuleCompiler& orRuleCompiler, const ProverParameters& parameters, ProofHistory& proofHistory, std::mutex& theoremListMutex, TheoremPublisher& theoremPublisher) : definitions(definitions), fullTheoremList(fullTheoremList), globalTheoremList(globalTheoremList), orElimBuiltByPair(orElimBuiltByPair), orElimCitedOrByMerged(orElimCitedOrByMerged), orElimGuardIndex(orElimGuardIndex), orElimInRunScannedRows(orElimInRunScannedRows), orElimPendingPairs(orElimPendingPairs), orRuleCompiler(orRuleCompiler), parameters(parameters), proofHistory(proofHistory), theoremListMutex(theoremListMutex), theoremPublisher(theoremPublisher) {}

/// @brief Token-aligned variable unification of two compiled
/// expressions — the or-elimination license probe's comparator.
///
/// @details
/// Walks @p a and @p b as alternating delimiter runs and token runs
/// (`[A-Za-z0-9_]+`). Delimiter runs must match byte-for-byte —
/// this makes polarity literal (a `!` present on one side only
/// fails the walk, I-175). Token pairs: a numeric token with value
/// >= 9 is a bound non-anchor variable and binds through the
/// @p forward / @p reverse maps (a consistent bijection across the
/// whole call sequence — both maps are caller-owned and shared
/// across the two disjunct alignments and the side-premise images);
/// every other token — operator names, `u_` tokens, anchor-slot
/// numerals 1..8 — must be byte-equal (anchor slots are pinned by
/// the shared `Anchor<...>[1..8]` authoring convention).
///
/// Both verdicts are contractual: `true` = the bijection extends,
/// `false` = the expressions do not align (a defined mismatch the
/// caller skips, not a failure state).
///
/// @param a       First expression (or-theorem side).
/// @param b       Second expression (variant-guard side).
/// @param forward Accumulated a→b variable bindings (extended).
/// @param reverse Accumulated b→a variable bindings (extended).
/// @return True iff the walk completes with a consistent bijection.
/// @see `findOrEliminationLicense` — sole caller.
bool OrEliminationBuilder::alignOrEliminationExprs(const std::string& a,
                                                 const std::string& b,
                                                 std::map<std::string, std::string>& forward,
                                                 std::map<std::string, std::string>& reverse) {
    const auto isTokenChar = [](char c) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')
            || (c >= '0' && c <= '9') || c == '_';
    };
    const auto isBindableVar = [](const std::string& tok) {
        for (char c : tok) {
            if (c < '0' || c > '9') return false;
        }
        return !tok.empty() && (tok.size() > 1 || tok[0] > '8');
    };

    std::size_t i = 0;
    std::size_t j = 0;
    while (i < a.size() && j < b.size()) {
        if (!isTokenChar(a[i]) || !isTokenChar(b[j])) {
            if (a[i] != b[j]) return false;
            ++i;
            ++j;
            continue;
        }
        std::size_t ie = i;
        std::size_t je = j;
        while (ie < a.size() && isTokenChar(a[ie])) ++ie;
        while (je < b.size() && isTokenChar(b[je])) ++je;
        const std::string ta = a.substr(i, ie - i);
        const std::string tb = b.substr(j, je - j);

        const bool bindA = isBindableVar(ta);
        const bool bindB = isBindableVar(tb);
        if (bindA != bindB) return false;
        if (bindA) {
            const auto fIt = forward.find(ta);
            if (fIt != forward.end()) {
                if (fIt->second != tb) return false;
            } else {
                if (reverse.count(tb)) return false;
                forward.emplace(ta, tb);
                reverse.emplace(tb, ta);
            }
        } else if (ta != tb) {
            return false;
        }
        i = ie;
        j = je;
    }
    return i == a.size() && j == b.size();
}

/// @brief Find a proved or theorem licensing the merge of two
/// guard premises; empty string = the defined no-license verdict.
///
/// @details
/// Scans a snapshot of `globalTheoremList` in list order
/// (deterministic — the first licensing row wins the citation).
/// A row licenses the pair iff: its head's core resolves through
/// `compiledEntity` to an or-category entity with exactly TWO
/// registry leaves (`flattenRegistryOrLeaves`, k=2 first build);
/// the leaves, instantiated with the head's actual args
/// (signature-args → head-args substitution), align with
/// (@p guardA, @p guardB) in either order under ONE variable
/// bijection (`alignOrEliminationExprs`); and every premise of the
/// or-theorem row is covered — its image under the bijection is
/// byte-present in @p commonChain, or the premise is a typing atom
/// `(in[v,...])` whose image variable occurs in @p commonChain
/// (the pragmatic prover-side gate; the external verifier redoes
/// this rigorously via the definition-set machinery).
///
/// The empty return is a contractual verdict, not a failure
/// sentinel (the `headSwitchOne` precedent): a licenseless pair
/// parks and is re-probed next iteration.
///
/// @param guardA      First guard premise (variant A's innermost).
/// @param guardB      Second guard premise (variant B's innermost).
/// @param commonChain The variants' shared premise chain
///                    (root-first, guard excluded).
/// @return The licensing or-theorem row, or "" if none exists yet.
/// @see `constructOrEliminationInRun` — sole caller.
/// @see `alignOrEliminationExprs` — the comparator.
std::string OrEliminationBuilder::findOrEliminationLicense(const std::string& guardA,
                                                         const std::string& guardB,
                                                         const std::vector<std::string>& commonChain) {
    std::vector<std::string> rows;
    {
        std::lock_guard<std::mutex> lock(theoremListMutex);
        rows.reserve(globalTheoremList.size());
        for (const auto& tpl : globalTheoremList) rows.push_back(std::get<0>(tpl));
    }

    const std::set<std::string> commonSet(commonChain.begin(), commonChain.end());

    const auto isTokenChar = [](char c) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')
            || (c >= '0' && c <= '9') || c == '_';
    };
    const auto occursAsToken = [&isTokenChar](const std::string& hay, const std::string& tok) {
        std::size_t pos = 0;
        while ((pos = hay.find(tok, pos)) != std::string::npos) {
            const std::size_t end = pos + tok.size();
            const bool leftOk = (pos == 0) || !isTokenChar(hay[pos - 1]);
            const bool rightOk = (end >= hay.size()) || !isTokenChar(hay[end]);
            if (leftOk && rightOk) return true;
            ++pos;
        }
        return false;
    };
    const auto eachVarMapped = [&isTokenChar](const std::string& expr,
                                              const std::map<std::string, std::string>& forward) {
        std::size_t i = 0;
        while (i < expr.size()) {
            if (!isTokenChar(expr[i])) { ++i; continue; }
            std::size_t ie = i;
            while (ie < expr.size() && isTokenChar(expr[ie])) ++ie;
            const std::string tok = expr.substr(i, ie - i);
            bool numeric = true;
            for (char c : tok) {
                if (c < '0' || c > '9') { numeric = false; break; }
            }
            if (numeric && (tok.size() > 1 || tok[0] > '8') && !forward.count(tok)) {
                return false;
            }
            i = ie;
        }
        return true;
    };

    for (const std::string& row : rows) {
        std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> chain;
        const std::string head = ce::disintegrateImplication(row, chain, this->definitions.configurations());

        const LogicalEntity* le = this->definitions.compiledEntity(extractExpressionUniversalSpan(StrSpan(head)));
        if (le == nullptr || le->category != "or") continue;

        std::vector<std::string> uLeaves;
        orRuleCompiler.flattenRegistryOrLeaves(*le, uLeaves);
        if (uLeaves.size() != 2) continue;   // k=2 in the first build

        // Instantiate the registry-canonical leaves with the head's args.
        const std::vector<std::string> sigArgs = ce::getArgs(le->signature);
        const std::vector<std::string> headArgs = ce::getArgs(head);
        if (sigArgs.size() != headArgs.size()) continue;
        std::map<std::string, std::string> subst;
        for (std::size_t k = 0; k < sigArgs.size(); ++k) subst[sigArgs[k]] = headArgs[k];
        const std::string leaf0 = ce::replaceKeysInString(uLeaves[0], subst);
        const std::string leaf1 = ce::replaceKeysInString(uLeaves[1], subst);

        for (int ordering = 0; ordering < 2; ++ordering) {
            const std::string& g0 = (ordering == 0) ? guardA : guardB;
            const std::string& g1 = (ordering == 0) ? guardB : guardA;
            std::map<std::string, std::string> forward;
            std::map<std::string, std::string> reverse;
            if (!alignOrEliminationExprs(leaf0, g0, forward, reverse)) continue;
            if (!alignOrEliminationExprs(leaf1, g1, forward, reverse)) continue;

            bool covered = true;
            for (const auto& link : chain) {
                const std::string& prem = std::get<0>(link);
                if (!eachVarMapped(prem, forward)) { covered = false; break; }
                const std::string img = ce::replaceKeysInString(prem, forward);
                if (commonSet.count(img)) continue;
                if (gl::startsWith(img, "(in[", 4)) {
                    const std::vector<std::string> inArgs = ce::getArgs(img);
                    if (!inArgs.empty()) {
                        bool occurs = false;
                        for (const std::string& c : commonChain) {
                            if (occursAsToken(c, inArgs[0])) { occurs = true; break; }
                        }
                        if (occurs) continue;
                    }
                }
                covered = false;
                break;
            }
            if (covered) return row;
        }
    }
    return std::string();
}

/// @brief In-run pre-split merge (or elimination): derive the
/// unguarded theorem from a proved guard-variant pair under a
/// proved or theorem.
///
/// @details
/// Runs on the phase-4 barrier of every `proveKernel` iteration,
/// single-threaded, immediately after `constructOrTheoremsInRun`
/// (so an or theorem minted this iteration can license a merge in
/// the same window) and before the deferred-compaction drain (so
/// the merged theorem's broadcast rides this iteration's drain).
///
/// Scan: `globalTheoremList` rows not yet in
/// `orElimInRunScannedRows` are disintegrated once; a row whose
/// innermost premise is binder-free files into `orElimGuardIndex`
/// under its common-part key. Two rows under one key with
/// different guards form a candidate pair, parked on
/// `orElimPendingPairs`.
///
/// Merge: for each pending pair, `findOrEliminationLicense` looks
/// for a proved or theorem whose two disjuncts align with the two
/// guards under one variable bijection (literal polarity, I-175)
/// and whose side premises are covered by the common chain. On a
/// license: the merged theorem is rebuilt canonically via
/// `reconstructImplicationFullBind(commonChain, head)` and lands as
/// a DERIVED row — `appendGlobalTheorem(merged, "or elimination",
/// variantA, variantB)`, one `fullTheoremList` mirror, the or
/// citation into `orElimCitedOrByMerged`, and the standard
/// deferred-compaction broadcast for implication-shaped results,
/// documented with the `("theorem", [])` origin at `main` through
/// `mergeTheoremOriginMailIntoRoot`. The merged theorem is never a
/// pool conjecture and no LB proof state is touched; the variants
/// stay first-class proved rows (never subsumed). In compressor mode
/// the seam is a defined no-op, exactly like the or seam.
///
/// @return Nothing.
/// @invariant Single-threaded seam — phase-4 barrier only. The
///            merge mints nothing into the operator registries; its
///            only writes are the theorem list, the seam ledgers,
///            the compaction queue, and the theorem-origin
///            documentation in the root's mailOut.
/// @see `findOrEliminationLicense` — the license probe.
/// @see `constructOrTheoremsInRun` — the sibling seam this one
///      mirrors (and must follow in the window).
void OrEliminationBuilder::constructOrEliminationInRun() {
    // Same defined no-op as the or seam: the compressor's re-derivation
    // must stay byte-identical to the run it audits.
    if (parameters.compressor_mode) return;

    std::vector<std::string> newRows;
    {
        std::lock_guard<std::mutex> lock(theoremListMutex);
        for (const auto& tpl : globalTheoremList) {
            const std::string& thm = std::get<0>(tpl);
            if (orElimInRunScannedRows.insert(thm).second) newRows.push_back(thm);
        }
    }

    // File new rows into the guard index; a second guard under one common
    // key forms a candidate pair, parked for the license probe below.
    for (const std::string& thm : newRows) {
        std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> chain;
        const std::string head = ce::disintegrateImplication(thm, chain, this->definitions.configurations());
        if (chain.size() < 2) continue;                    // a guard needs a shared premise before it
        if (!std::get<1>(chain.back()).empty()) continue;  // pre-split guards are binder-free

        const std::string& guard = std::get<0>(chain.back());
        std::string key;
        for (std::size_t k = 0; k + 1 < chain.size(); ++k) {
            key += std::get<0>(chain[k]);
            key += '\x01';
        }
        key += '\x02';
        key += head;

        std::vector<std::pair<std::string, std::string>>& entries = orElimGuardIndex[key];
        for (const std::pair<std::string, std::string>& other : entries) {
            if (other.second == guard) continue;
            orElimPendingPairs.push_back({other.first, other.second, thm, guard});
        }
        entries.emplace_back(thm, guard);
    }

    // License probe over the parked pairs, insertion order. A pair leaves
    // only by merging (or the append door's string dedup); a licenseless
    // pair parks for a later iteration's or mints.
    std::vector<std::array<std::string, 4>> stillPending;
    stillPending.reserve(orElimPendingPairs.size());
    // Documentation mail for this seam's broadcasts (the or seam's idiom):
    // the ("theorem", []) origin at main for every broadcast merged theorem.
    Mail theoremOriginMail;
    for (const std::array<std::string, 4>& pending : orElimPendingPairs) {
        const std::string& variantA = pending[0];
        const std::string& guardA = pending[1];
        const std::string& variantB = pending[2];
        const std::string& guardB = pending[3];

        const std::pair<std::string, std::string> pairKey(
            std::min(variantA, variantB), std::max(variantA, variantB));
        if (orElimBuiltByPair.count(pairKey)) continue;

        // Rebuild the shared context from variant A — byte-identical on
        // variant B by the guard-index key construction.
        std::vector<std::tuple<std::string, std::vector<std::string>, std::set<std::string>>> chain;
        const std::string head = ce::disintegrateImplication(variantA, chain, this->definitions.configurations());
        assert(chain.size() >= 2
            && "constructOrEliminationInRun: parked variant lost its guard chain");
        std::vector<std::string> commonChain;
        commonChain.reserve(chain.size() - 1);
        for (std::size_t k = 0; k + 1 < chain.size(); ++k) {
            commonChain.push_back(std::get<0>(chain[k]));
        }

        const std::string license = findOrEliminationLicense(guardA, guardB, commonChain);
        if (license.empty()) {
            stillPending.push_back(pending);
            continue;
        }

        const std::string merged = gl::ImplicationBuilder::reconstructImplicationFullBind(commonChain, head);
        orElimBuiltByPair.emplace(pairKey, merged);

        // String dedup door: two different pre-split pairs can fold to one
        // merged theorem; the second registers and broadcasts nothing.
        if (!theoremPublisher.appendGlobalTheorem(merged, "or elimination", variantA, variantB)) continue;
        this->fullTheoremList.emplace_back(merged, "or elimination", variantA, variantB);
        orElimCitedOrByMerged.emplace(merged, license);
        std::cout << "OR elimination merged (in-run): " << merged << std::endl;

        // Broadcast like any proved implication via the deferred-compaction
        // drain that follows this seam (empty level set at the drain; a
        // premise-free result registers without broadcast).
        if (gl::startsWith(merged, "(>[", 3)) {
            theoremPublisher.recordPendingCompaction(merged, /*kySize=*/0, /*coreId=*/-1);
            if (parameters.trackHistory) {
                proofHistory.addOrigin(theoremOriginMail.exprOriginMap,
                    ExpressionWithValidity(merged, "main"),
                    std::make_pair(std::string("theorem"),
                                   std::vector<ExpressionWithValidity>()),
                    parameters.max_origin_per_expr);
            }
        }
    }
    orElimPendingPairs.swap(stillPending);
    theoremPublisher.mergeTheoremOriginMailIntoRoot(theoremOriginMail);
}
}
