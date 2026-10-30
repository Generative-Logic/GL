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
#include "or_rule_compiler.hpp"
#include "structural_compiler.hpp"
#include "existence_rule_compiler.hpp"
#include "supporting_rule_compiler.hpp"
#include "implication_builder.hpp"
#include "compilation_reuse.hpp"
#include "definition_expansion.hpp"
#include "../compiler.hpp"
#include <numeric>
#include <fstream>

namespace gl {
/// @brief Bind the cooperating compilation components.
/// @details Construction stores references only; all bound objects outlive this component.
/// @param definitions Borrowed definitions component.
/// @param compilationReuse Borrowed compilationReuse component.
/// @param structuralCompiler Borrowed structuralCompiler component.
/// @param supportingRuleCompiler Borrowed supportingRuleCompiler component.
/// @return Constructed compilation component.
OrRuleCompiler::OrRuleCompiler(CompiledDefinitionRegistry& definitions, CompilationReuse& compilationReuse, StructuralCompiler& structuralCompiler, SupportingRuleCompiler& supportingRuleCompiler)
    : definitions(definitions), compilationReuse(compilationReuse), structuralCompiler(structuralCompiler), supportingRuleCompiler(supportingRuleCompiler) {}
/// @brief Find the registered `or`-category operator whose body is the
        /// canonical body of @p elems, or mint a fresh `or<orCounter>` entry
        /// in BOTH compile registries.
        ///
        /// @details
        /// The registry-dedup-then-mint block of every or-operator mint
        /// (D-268), one or per canonical body (I-323,
        /// D-556). The canonical key of
        /// @p elems (`CompilationReuse::canonicalReuseKey`: disjunct order
        /// and `u_` numbering do not matter) finds a public or first, else
        /// the numbered or registered under that body at its mint, its
        /// compilation or its binary load. The retained or may store its
        /// leaves in another order, so a reuse returns its signature's
        /// argument permutation and records the ordered list for
        /// `CompilationReuse::compiledOrByElements`. A miss mints
        /// `or<orCounter>` with the canonical `(or<N>[u_1,..,u_arity])`
        /// signature, registers `compiledExpressions` (entity resolution),
        /// `coreExpressionMap` (expansion / disintegration) and its canonical
        /// body in the repetition map, and bumps `orCounter`. Without the
        /// body key a construction in another disjunct order minted a second
        /// name for the same disjunction, and a later batch's compile of a
        /// reloaded theorem then named the theorem differently from its
        /// registration.
        ///
        /// @param elems     Ordered u_-canonical element list, each
        ///        disjunct in TRUE polarity (I-175).
        /// @param arity     Number of distinct `u_` arguments across
        ///        @p elems (the operator's signature arity).
        /// @param mintedOut Optional out-flag — set true iff a fresh entry
        ///        was minted (false on registry reuse).
        /// @param argumentPermutationOut Optional one-based input slots in
        ///        retained-signature order; required whenever the retained
        ///        or orders its signature differently (asserted).
        /// @return The operator name (existing or freshly minted).
        /// @invariant Single-threaded seam only (I-137) — no worker reads
        ///            the compile registries while a writer seam holds.
        ///            Every or is registered under its canonical body, so a
        ///            mint finds no or with the same elements (asserted).
        /// @see `constructOrTheorem`, `preMintReducedOrs` — the callers.
std::string OrRuleCompiler::findOrMintOrOperator(const std::vector<std::string>& elems,
                                         int arity,
                                         bool* mintedOut,
                                         std::vector<int>* argumentPermutationOut) {

    if (mintedOut != nullptr) *mintedOut = false;
    if (argumentPermutationOut != nullptr) {
        argumentPermutationOut->resize(static_cast<std::size_t>(arity));
        std::iota(argumentPermutationOut->begin(), argumentPermutationOut->end(), 1);
    }
    std::vector<EncodedExpression> encoded;
    for (const auto& element : elems) encoded.emplace_back(element, "main");
    const auto canonical = gl::CompilationReuse::canonicalReuseKey(encoded, {});
    const CompilationReuseKey key{canonical.first, "or"};
    // A public or has precedence; otherwise the numbered or of the body.
    const CompilationReuseMap& index = compilationReuse.publicEntries().count(key) != 0
        ? compilationReuse.publicEntries() : compilationReuse.entries();
    const auto retained = index.find(key);
    if (retained != index.end()) {
        const auto mapping = gl::CompilationNormalizer::combineUnchangeablePaths(canonical.second, std::get<0>(retained->second));
        const auto arguments = ce::getArgs(gl::CompilationNormalizer::createArgList(mapping));
        assert(arguments.size() == static_cast<std::size_t>(arity));
        std::vector<int> permutation;
        for (const auto& argument : arguments) permutation.push_back(std::stoi(argument));
        for (int i = 0; i < arity; ++i)
            assert(argumentPermutationOut != nullptr || permutation[static_cast<std::size_t>(i)] == i + 1);
        if (argumentPermutationOut != nullptr) *argumentPermutationOut = permutation;
        const auto& name = std::get<1>(retained->second);
        compilationReuse.rememberOrReuse(elems, name, permutation);
        return name;
    }
    assert(std::none_of(definitions.compiledExpressions.begin(), definitions.compiledExpressions.end(),
        [&elems](const auto& entry) { return entry.second.category == "or" && entry.second.elements == elems; })
        && "findOrMintOrOperator: an or with these elements is not registered under its canonical body (I-323)");

    const std::string orCoreName = "or" + std::to_string(definitions.orCounter);
    std::string sigArgs;
    for (int i = 0; i < arity; ++i) {
        if (i > 0) sigArgs += ",";
        sigArgs += "u_" + std::to_string(i + 1);
    }
    const std::string signature = "(" + orCoreName + "[" + sigArgs + "])";
    LogicalEntity orLe("or", elems, signature, arity);
    definitions.compiledExpressions.insert(std::make_pair(orCoreName, orLe));

    // Also register in coreExpressionMap for expandSignature/disintegration.
    ce::CoreExpressionConfig orCfg;
    orCfg.arity = arity;
    orCfg.signature = signature;
    definitions.coreExpressionMap.insert(std::make_pair(orCoreName, orCfg));
    definitions.orCounter++;
    // The body key: later mints and compiles of the same disjunction, in
    // any disjunct order, reuse this or (I-323).
    compilationReuse.registerStoredCompilation(orCoreName, false);

    // Or mint hook: the K-rule compacts ride the or's entry
    // (D-309); the subset-exclusion family is appended
    // at the preMintReducedOrs seam once the reduced closure exists.
    compileOrKRules(orCoreName);

    if (mintedOut != nullptr) *mintedOut = true;
    return orCoreName;
}

/// @brief Flatten a registered or-category entity into its ordered
        /// non-or leaves, expressed in the entity's OWN `u_` terms.
        ///
        /// @details
        /// The registry twin of `flattenOrLeaves` (which walks an
        /// instantiated `WorkInstruction`): walks the COMPILED entity
        /// graph instead, starting from the entity's own signature as the
        /// root instance (identity substitution), substituting each nested
        /// or-child's signature args by its instance args (the
        /// `isOrIntroInstall` substitution walk, heap form) and recursing
        /// into or-category children only (I-166). Non-or elements append
        /// in element order, so left- and right-nested shapes flatten to
        /// the same ordered leaf list the instance-side walk produces.
        ///
        /// Barrier-seam helper — heap use is sanctioned here (never on the
        /// burst path).
        ///
        /// @param orLe      The or-category compiled entity to flatten.
        /// @param outLeaves Receives the ordered non-or leaves (appended).
        /// @return (void)
        /// @invariant Recursion depth capped by MAX_INSTRUCTION_ELEMENTS
        ///            (assert); at least two leaves result; every leaf arg
        ///            is a `u_` token (registry canonical form).
        /// @see `flattenOrLeaves` — the instantiated-instruction twin.
        /// @see `preMintReducedOrs` — the pre-mint consumer.
void OrRuleCompiler::flattenRegistryOrLeaves(const LogicalEntity& orLe,
                                     std::vector<std::string>& outLeaves) const {

    assert(orLe.category == "or"
        && "flattenRegistryOrLeaves: entity must be or-category");

    int depth = 0;
    const auto walk = [&](const LogicalEntity& node, const std::string& instance,
                          const auto& self) -> void {
        assert(depth < ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
            && "flattenRegistryOrLeaves: OR depth exceeds cap");
        ++depth;
        const std::vector<std::string> sigArgs = ce::getArgs(node.signature);
        const std::vector<std::string> instArgs = ce::getArgs(instance);
        assert(sigArgs.size() == instArgs.size()
            && "flattenRegistryOrLeaves: instance arity differs from compiled or");
        std::map<std::string, std::string> subst;
        for (std::size_t a = 0; a < sigArgs.size(); ++a) {
            subst[sigArgs[a]] = instArgs[a];
        }
        for (const std::string& rawElem : node.elements) {
            const std::string elem = ce::replaceKeysInString(rawElem, subst);
            const LogicalEntity* childLe =
                (!elem.empty() && elem[0] == '(')
                ? definitions.compiledEntity(extractExpressionUniversalSpan(StrSpan(elem)))
                : nullptr;
            if (childLe != nullptr && childLe->category == "or") {
                self(*childLe, elem, self);
            } else {
                outLeaves.push_back(elem);
            }
        }
        --depth;
    };
    walk(orLe, orLe.signature, walk);
    assert(outLeaves.size() >= 2
        && "flattenRegistryOrLeaves: or entity flattens to fewer than two leaves");
}

/// @brief Pre-mint the reduced (k-1)-ary or-operators for every
        /// registered or, to fixpoint (the single-elimination closure).
        ///
        /// @details
        /// Part 1 of the B8 three-extension design
        /// (D-268): single-exclusion rules
        /// emitted at flat or consumption carry a genuine (k-1)-ary
        /// or-compact head, and registry writers are single-threaded seams
        /// (I-137) — so every reduced operator the emission could need is
        /// minted HERE, at the barrier / pre-burst seams, never mid-burst.
        /// The emission-side registry lookup then treats a miss as a hard
        /// assert.
        ///
        /// Walks a snapshot of the or-category entries of
        /// `compiledExpressions` in name order (deterministic), flattens
        /// each to its ordered non-or leaves (`flattenRegistryOrLeaves`),
        /// and for every entry with k >= 3 leaves mints the k
        /// leaves-minus-one lists (u_-renumbered, I-23 dedup) through
        /// `findOrMintOrOperator`. A fresh mint joins the worklist so ITS
        /// reductions exist too — the closure reaches single-elimination
        /// chains, bounded per entry by `kMaxReducedOrLeaves`.
        ///
        /// Idempotent: a re-run dedups to zero mints. Runs in compressor
        /// mode too — the mints are a pure function of the or-registry
        /// content, so the audited run and the compressor re-derivation
        /// stay registry-aligned.
        ///
        /// @return (void)
        /// @invariant Single-threaded seam only (I-137);
        ///            k <= kMaxReducedOrLeaves asserted per flattened
        ///            entry (Rule-19 tripwire naming the constant).
        /// @see `constructOrTheoremsInRun` — tail caller (in-run ors);
        ///      `proveKernel` — entry + post-compaction-drain callers.
void OrRuleCompiler::preMintReducedOrs() {
    std::vector<std::string> work;
    std::set<std::string> queued;
    for (const auto& [name, le] : definitions.compiledExpressions) {
        if (le.category == "or" && queued.insert(name).second) {
            work.push_back(name);
        }
    }
    for (std::size_t w = 0; w < work.size(); ++w) {
        const LogicalEntity* le = definitions.compiledEntity(StrSpan(work[w]));
        assert(le != nullptr && le->category == "or"
            && "preMintReducedOrs: queued name must resolve to an or entity");
        std::vector<std::string> leaves;
        flattenRegistryOrLeaves(*le, leaves);
        const int k = static_cast<int>(leaves.size());
        assert(k <= ExecutionParameters::kMaxReducedOrLeaves
            && "preMintReducedOrs: or leaf count exceeds kMaxReducedOrLeaves");
        if (k < 3) continue;
        for (int i = 0; i < k; ++i) {
            std::vector<std::string> reduced;
            reduced.reserve(static_cast<std::size_t>(k) - 1);
            for (int j = 0; j < k; ++j) {
                if (j != i) reduced.push_back(leaves[j]);
            }
            const int arity = gl::OrRuleCompiler::renumberULeaves(reduced);
            bool minted = false;
            std::vector<int> argumentPositions;
            const std::string rname =
                findOrMintOrOperator(reduced, arity, &minted, &argumentPositions);
            if (minted && queued.insert(rname).second) {
                work.push_back(rname);
            }
        }
    }

    // Subset-exclusion completion (D-309). Every or now has its reduced
    // closure, so the subset-exclusion compacts can resolve their heads.
    // Name-ordered snapshot (deterministic; the compiles insert implication
    // entries into the same map). Every or carries its K-rules from its mint
    // or from the binary it was loaded from.
    std::vector<std::string> orNames;
    for (const auto& [name, le] : definitions.compiledExpressions) {
        if (le.category == "or") orNames.push_back(name);
    }
    for (const std::string& name : orNames) {
        compileOrSymmetry(name);
        assert(!definitions.compiledEntity(StrSpan(name))->implications.empty()
            && "preMintReducedOrs: an or without its K-rule compacts (missed mint hook)");
        const LogicalEntity* le = definitions.compiledEntity(StrSpan(name));
        std::vector<std::string> leaves;
        flattenRegistryOrLeaves(*le, leaves);
        const int32_t k = static_cast<int32_t>(leaves.size());
        if (k >= 3 && static_cast<int32_t>(le->implications.size()) == k) {
            compileOrSubsetExclusions(name);
        }
        assert(static_cast<int32_t>(definitions.compiledEntity(StrSpan(name))->implications.size())
                == gl::OrRuleCompiler::expectedOrImplicationCount(k)
            && "preMintReducedOrs: or-implication list incomplete after the closure seam");
    }

    supportingRuleCompiler.compileContrapositionFamilies();
    supportingRuleCompiler.compileDisintegrationPackages();
}

/// @brief Compile an or's K-rule compacts — its "or implications"
        /// `!d_0 ∧ … ∧ !d_{i-1} ∧ !d_{i+1} ∧ … → d_i`, one per leaf, as
        /// `(implication<N>[u_…])` instances over the or's OWN `u_` tokens —
        /// into `LogicalEntity::implications` (D-309).
        ///
        /// @details
        /// Flattens the or to its registry-term leaves
        /// (`flattenRegistryOrLeaves`, the disintegrator's leaf order); for
        /// leaf `i` the premises are the other leaves negated with
        /// double-negation cancellation (`negate`, I-175) and the head is the
        /// leaf itself; the chain is assembled with
        /// `reconstructImplicationFullBind` (every argument is a `u_` token,
        /// so the rule is binder-free) and compiled through
        /// `compileImplicationToCompact` — the same dedup-or-mint the
        /// mail-broadcast compaction uses (`excludeRepetitions`, category
        /// `implication`), which strips the `u_` prefixes and returns the
        /// compact over the bare token names; `prefixArgumentsWithU`
        /// restores the or's token names, so the stored string's argument
        /// order IS the projection from the implication's signature positions
        /// onto the or's tokens. A dedup hit against an earlier-registered
        /// implication keeps THAT entity's first-seen premise order (I-52),
        /// exactly as the mail round-trip does.
        ///
        /// Idempotent: an or that already carries a list returns unchanged
        /// (a registry reuse or a reload re-registering the same or). Barrier
        /// seam only — heap use is sanctioned here, never on the burst path.
        ///
        /// @param orName Registry name of an or-category entity (asserted).
        /// @return (void)
        /// @invariant Single-threaded seam only (I-137). Afterwards the list
        ///            holds exactly `k` compacts, each with the or's full
        ///            arity (every leaf occurs in every K-rule, asserted).
        /// @see `compileOrSubsetExclusions` — appends the second family;
        ///      `findOrMintOrOperator`, `compileExpression` —
        ///      the mint-site callers.
void OrRuleCompiler::compileOrKRules(const std::string& orName) {
    compileOrSymmetry(orName);
    const auto it = definitions.compiledExpressions.find(orName);
    assert(it != definitions.compiledExpressions.end() && it->second.category == "or"
        && "compileOrKRules: name must resolve to an or entity");
    // Already carrying its list (registry reuse / reload re-registration):
    // a defined state, not a failure.
    if (!it->second.implications.empty()) return;

    std::vector<std::string> leaves;
    flattenRegistryOrLeaves(it->second, leaves);
    const std::size_t k = leaves.size();
    const std::size_t arity = ce::getArgs(it->second.signature).size();

    std::vector<std::string> compacts;
    compacts.reserve(k);
    for (std::size_t i = 0; i < k; ++i) {
        std::vector<std::string> premises;
        premises.reserve(k - 1);
        for (std::size_t j = 0; j < k; ++j) {
            // negate, not a blind "!" prefix: a disjunct carries its true
            // polarity, so a negated leaf's exclusion premise is its bare
            // positive core (I-175).
            if (j != i) premises.push_back(gl::DefinitionExpansion::negate(leaves[j]));
        }
        const std::string rule = gl::ImplicationBuilder::reconstructImplicationFullBind(premises, leaves[i]);
        const std::string compact = gl::DefinitionExpansion::prefixArgumentsWithU(structuralCompiler.compileImplicationToCompact(rule));
        // Every leaf occurs in every K-rule, so the compact spans the or's
        // full token set — its argument list is a permutation of the or's
        // signature tokens.
        assert(ce::getArgs(compact).size() == arity
            && "compileOrKRules: K-rule compact does not span the or's tokens");
        compacts.push_back(compact);
    }
    // Re-find: the compiles above insert implication entries into the same
    // map (node-stable, but the re-fetch is the honest form).
    definitions.compiledExpressions.find(orName)->second.implications = std::move(compacts);
}

/// @brief Mark an OR whose two digit parameters are interchangeable.
        ///
        /// @details Flattens positive OR nesting and compares signed leaf
        /// multisets after equality orientation, keeping free slots fixed.
        /// Exactly two definition-typed digit slots qualify. Other predicates
        /// retain ordered arguments; metadata is derived before proof work.
        /// @param orName Registered OR operator whose metadata is refreshed.
        /// @return Void.
        /// @invariant Does not change definitions, names or proof expressions.
        /// @see preMintReducedOrs; canonicalOrProcessingForm.
void OrRuleCompiler::compileOrSymmetry(const std::string& orName) {
    auto& entity = definitions.compiledExpressions.at(orName);
    assert(entity.category == "or");
    entity.symmetricDigitSlots = {0, 0};
    if (!isNumberedCompilationName(orName, "or")) return;
    const auto parameters = ce::getArgs(entity.signature);
    std::set<std::string> digits;
    const auto collectTypes = [&](const std::string& instance, int depth,
                                  const auto& self) -> void {
        assert(depth < ExecutionParameters::MAX_INSTRUCTION_ELEMENTS);
        const StrSpan core = extractExpressionUniversalSpan(StrSpan(instance));
        const auto arguments = ce::getArgs(instance);
        const auto* config = definitions.coreConfig(core);
        bool allTyped = config != nullptr && !config->definitionSets.empty();
        for (std::size_t i = 0; i < arguments.size(); ++i) {
            const auto* type = config == nullptr ? nullptr
                : gl::CompiledDefinitionRegistry::findDefinitionSet(config, static_cast<int>(i) + 1);
            allTyped = allTyped && type != nullptr;
            if (type != nullptr && type->first == "(1)"
                && std::find(parameters.begin(), parameters.end(), arguments[i]) != parameters.end())
                digits.insert(arguments[i]);
        }
        if (allTyped) return;
        const auto* child = definitions.compiledEntity(core);
        if (child == nullptr || child->category == "atomic") return;
        const auto formal = ce::getArgs(child->signature);
        assert(formal.size() == arguments.size());
        std::map<std::string, std::string> substitution;
        for (std::size_t i = 0; i < formal.size(); ++i)
            substitution[formal[i]] = arguments[i];
        for (const auto& element : child->elements)
            self(ce::replaceKeysInString(element, substitution), depth + 1, self);
    };
    collectTypes(entity.signature, 0, collectTypes);
    if (digits.size() != 2) return;
    int positions[2];
    int count = 0;
    for (std::size_t i = 0; i < parameters.size(); ++i)
        if (digits.count(parameters[i])) positions[count++] = static_cast<int>(i);
    assert(count == 2);
    std::vector<std::string> leaves;
    flattenRegistryOrLeaves(entity, leaves);
    const std::map<std::string, std::string> swap{
        {parameters[positions[0]], parameters[positions[1]]},
        {parameters[positions[1]], parameters[positions[0]]}};
    const auto normalized = [&](bool exchanged) {
        std::vector<std::string> result;
        for (const auto& leaf : leaves) {
            std::string value = exchanged ? ce::replaceKeysInString(leaf, swap) : leaf;
            if (equalSpans(extractExpressionUniversalSpan(StrSpan(value)), StrSpan("=", 1))) {
                auto arguments = ce::getArgs(value);
                assert(arguments.size() == 2);
                if (arguments[1] < arguments[0]) std::swap(arguments[0], arguments[1]);
                value = (value[0] == '!' ? "!(=[" : "(=[")
                    + arguments[0] + "," + arguments[1] + "])";
            }
            result.push_back(std::move(value));
        }
        std::sort(result.begin(), result.end());
        return result;
    };
    if (normalized(false) == normalized(true))
        entity.symmetricDigitSlots = {static_cast<int16_t>(positions[0] + 1),
                                      static_cast<int16_t>(positions[1] + 1)};
}

/// @brief Append an or's subset-exclusion compacts — the rules
        /// `!D_i1 ∧ … ∧ !D_ij → (reducedOr[survivors])` for every excluded
        /// index subset with 1 ≤ j ≤ k−2 (I-184, D-269) — to
        /// `LogicalEntity::implications`, after the K-rules.
        ///
        /// @details
        /// Enumerates the excluded subsets exactly as `consumeOrLeavesCohort`
        /// section 1b does (j ascending, size-j index subsets in lexicographic
        /// order), so list position `k + n` is the disintegrator's n-th
        /// emission. Per subset: the survivors in parent order are
        /// `u_`-renumbered (`renumberULeaves`, which also reports the parent
        /// tokens in first-appearance order) and resolved through
        /// `compiledOrByElements` — a miss asserts (I-185: the closure is
        /// registered before this runs); the head is
        /// `(reducedName[parent tokens in that order])`, the premises the
        /// excluded leaves negated in parent order; the chain is compiled like
        /// a K-rule (`reconstructImplicationFullBind` →
        /// `compileImplicationToCompact` → `prefixArgumentsWithU`).
        ///
        /// Called only from `preMintReducedOrs` after its closure walk, for an
        /// or whose list holds exactly its K-rules (precondition asserted).
        ///
        /// @param orName Registry name of an or-category entity with
        ///        3 ≤ k ≤ kMaxReducedOrLeaves leaves (asserted).
        /// @return (void)
        /// @invariant Single-threaded seam only (I-137). Afterwards the list
        ///            holds `expectedOrImplicationCount(k)` compacts.
        /// @see `compileOrKRules`, `preMintReducedOrs`,
        ///      `consumeOrLeavesCohort` (the consumer, same order).
void OrRuleCompiler::compileOrSubsetExclusions(const std::string& orName) {
    const auto it = definitions.compiledExpressions.find(orName);
    assert(it != definitions.compiledExpressions.end() && it->second.category == "or"
        && "compileOrSubsetExclusions: name must resolve to an or entity");
    std::vector<std::string> leaves;
    flattenRegistryOrLeaves(it->second, leaves);
    const int32_t k = static_cast<int32_t>(leaves.size());
    assert(k >= 3 && k <= ExecutionParameters::kMaxReducedOrLeaves
        && "compileOrSubsetExclusions: leaf count outside [3, kMaxReducedOrLeaves]");
    assert(static_cast<int32_t>(it->second.implications.size()) == k
        && "compileOrSubsetExclusions: precondition — the list holds exactly the K-rules");
    const std::size_t arity = ce::getArgs(it->second.signature).size();

    std::vector<std::string> compacts;
    // j = 1..k-2 excluded disjuncts; size-j index subsets in lexicographic
    // order — the disintegrator's enumeration (consumeOrLeavesCohort 1b).
    for (int32_t exclN = 1; exclN <= k - 2; ++exclN) {
        int32_t sel[ExecutionParameters::kMaxReducedOrLeaves];
        for (int32_t s = 0; s < exclN; ++s) sel[s] = s;
        for (;;) {
            bool exclMask[ExecutionParameters::kMaxReducedOrLeaves] = {};
            for (int32_t s = 0; s < exclN; ++s) exclMask[sel[s]] = true;

            // Survivors in parent order; u_-renumbered to the reduced or's
            // registry identity, keeping the parent tokens' first-appearance
            // order — the head's argument order (token u_<p> is parent
            // signature position p).
            std::vector<std::string> reduced;
            for (int32_t j = 0; j < k; ++j) {
                if (!exclMask[j]) reduced.push_back(leaves[j]);
            }
            std::vector<std::string> tokOrder;
            gl::OrRuleCompiler::renumberULeaves(reduced, &tokOrder);
            std::vector<StrSpan> canon;
            canon.reserve(reduced.size());
            for (const std::string& r : reduced) canon.emplace_back(r);
            std::vector<int> argumentPositions(tokOrder.size());
            const std::string* redName =
                compilationReuse.compiledOrByElements(canon.data(), static_cast<int32_t>(canon.size()),
                                    argumentPositions.data(), static_cast<int>(argumentPositions.size()));
            assert(redName != nullptr
                && "compileOrSubsetExclusions: reduced or missing (I-185 violated)");

            std::string head = "(" + *redName + "[";
            for (std::size_t s = 0; s < tokOrder.size(); ++s) {
                if (s > 0) head += ',';
                head += tokOrder[static_cast<std::size_t>(argumentPositions[s]) - 1];
            }
            head += "])";

            std::vector<std::string> premises;
            premises.reserve(static_cast<std::size_t>(exclN));
            for (int32_t s = 0; s < exclN; ++s) premises.push_back(gl::DefinitionExpansion::negate(leaves[sel[s]]));

            const std::string rule = gl::ImplicationBuilder::reconstructImplicationFullBind(premises, head);
            const std::string compact = gl::DefinitionExpansion::prefixArgumentsWithU(structuralCompiler.compileImplicationToCompact(rule));
            // Excluded leaves + surviving head together cover every leaf, so
            // the compact spans the or's full token set.
            assert(ce::getArgs(compact).size() == arity
                && "compileOrSubsetExclusions: compact does not span the or's tokens");
            compacts.push_back(compact);

            int32_t pos = exclN - 1;
            while (pos >= 0 && sel[pos] == k - exclN + pos) --pos;
            if (pos < 0) break;
            ++sel[pos];
            for (int32_t s = pos + 1; s < exclN; ++s) sel[s] = sel[s - 1] + 1;
        }
    }
    std::vector<std::string>& list = definitions.compiledExpressions.find(orName)->second.implications;
    list.insert(list.end(), compacts.begin(), compacts.end());
}

} // namespace gl
