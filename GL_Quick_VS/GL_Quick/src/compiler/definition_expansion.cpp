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
#include "definition_expansion.hpp"
#include "implication_builder.hpp"

namespace gl {

/// @brief Bind the definition tables used by expression expansion.
/// @details Both tables outlive this view; expansion preserves their contents.
/// @param configurations Configured source definitions.
/// @param definitions Registered compiled definitions.
/// @return Constructed definition-expansion view.
DefinitionExpansion::DefinitionExpansion(const ce::CoreExpressionMap& configurations,
    const CompiledExpressionMap& definitions)
    : coreExpressionMap(configurations), compiledExpressions(definitions) {}

/// @brief Find one configured expression without allocating a lookup key.
/// @details The returned row is borrowed from the stable configuration map.
/// @param core Expression name.
/// @return Configuration row, or null for the defined no-match case.
const ce::CoreExpressionConfig* DefinitionExpansion::coreConfig(const StrSpan& core) const {
    const std::string_view key = core.len > 0
        ? std::string_view(core.ptr, static_cast<std::size_t>(core.len)) : std::string_view();
    const auto found = coreExpressionMap.find(key);
    return found == coreExpressionMap.end() ? nullptr : &found->second;
}
/// @brief Expand a configured predicate without capturing its arguments.
///
/// @details Preserves expression spelling, binding and the existing transformation order.
/// @param expr Expression text.
/// @return Transformed expression text.
std::string DefinitionExpansion::expandExpr(const std::string& expr) {
    // 1. Check if expression is simple (e.g., "Macro[arg1, arg2]")
    if (!ce::expressionIsSimple(expr)) {
        return expr;
    }

    // 2. Extract the Head (Macro Name)
    const ce::CoreExpressionConfig* cfgPtr =
        coreConfig(extractExpressionUniversalSpan(StrSpan(expr)));
    if (cfgPtr == nullptr) {
        // Not a known macro, return original
        return expr;
    }
    const ce::CoreExpressionConfig& cfg = *cfgPtr;

    // 3. Retrieve the Definition Template
    // Assuming definition is stored as a string
    if (!std::holds_alternative<std::string>(cfg.definition)) {
        return expr;
    }
    std::string defString = std::get<std::string>(cfg.definition);

    // 4. Extract Arguments
    // replacingArgs: The values passed in the current expression
    std::vector<std::string> replacingArgs = ce::getArgs(expr);
    // argsToBeReplaced: The formal parameters defined in the macro signature
    std::vector<std::string> argsToBeReplaced = ce::getArgs(cfg.signature);

    // Basic arity check
    if (replacingArgs.size() != argsToBeReplaced.size()) {
        return expr;
    }

    // 5. Collision Avoidance (Disentanglement)
    // Goal: If an input argument (e.g., "x") is NOT a formal parameter but appears
    // inside the definition (as a bound variable), we must rename the bound variable
    // in the definition to avoid capturing the input "x".

    std::map<std::string, std::string> collisionRenamingMap;
    int counter = 1;

    // Check every input argument for potential collision
    for (const std::string& inputArg : replacingArgs) {

        // Check if this input arg is actually one of the formal parameters.
        // If it is, it will be replaced by the expansion map later, so it's safe.
        bool isFormalParam = false;
        for (const std::string& param : argsToBeReplaced) {
            if (param == inputArg) {
                isFormalParam = true;
                break;
            }
        }

        if (isFormalParam) {
            continue;
        }

        // If we are here, 'inputArg' is being passed in, but it is NOT a parameter slot.
        // We must ensure 'defString' does not contain 'inputArg' as a local variable.
        // We speculatively add it to the renaming map.
        // ce::replaceKeysInString will only perform the replacement if the key exists.

        // Generate a unique name: inputArg + Counter
        std::string newName;
        while (true) {
            newName = inputArg + std::to_string(counter);

            // Ensure the new name doesn't exist in the definition OR the input args
            // (Simple string find is used here for safety; ideally token-based)
            if (defString.find(newName) == std::string::npos) {
                // Also ensure we haven't generated a name that conflicts with another input
                bool nameClash = false;
                for (const auto& existingArg : replacingArgs) {
                    if (existingArg == newName) nameClash = true;
                }
                if (!nameClash) break;
            }
            counter++;
        }

        collisionRenamingMap[inputArg] = newName;
    }

    // Apply collision renaming to the definition template BEFORE standard expansion
    if (!collisionRenamingMap.empty()) {
        defString = ce::replaceKeysInString(defString, collisionRenamingMap);
    }

    // 6. Standard Expansion
    // Map formal parameters to input arguments
    std::map<std::string, std::string> expansionMap;
    for (std::size_t i = 0; i < argsToBeReplaced.size(); ++i) {
        expansionMap[argsToBeReplaced[i]] = replacingArgs[i];
    }

    // 7. Final Substitution
    std::string expandedExpr = ce::replaceKeysInString(defString, expansionMap);

    return expandedExpr;
}

/// @brief Expand compiled logical names to their underlying expression form.
///
/// @details Preserves expression spelling, binding and the existing transformation order.
/// @param expr Expression text.
/// @return Transformed expression text.
std::string DefinitionExpansion::expandToBaseForm(const std::string& expr) const {
    // Expand batch-specific compiled expressions (and0, or0, implication0, existence0, ...)
    // back to structural form so subsequent batches can parse them.
    // Config-level expressions (in, =, in2, in3, fold, etc.) stay untouched.

    auto isCompiled = [](const std::string& name) -> bool {
        static const std::vector<std::string> prefixes = {"and", "or", "implication", "existence"};
        for (const auto& p : prefixes) {
            if (name.size() > p.size() && name.compare(0, p.size(), p) == 0) {
                bool allDigits = true;
                for (size_t i = p.size(); i < name.size(); ++i) {
                    if (!std::isdigit(static_cast<unsigned char>(name[i]))) { allDigits = false; break; }
                }
                if (allDigits) return true;
            }
        }
        return false;
    };

    std::string result = expr;
    bool changed = true;
    int safety = 0;
    while (changed && safety++ < 20) {
        changed = false;

        for (const auto& [name, le] : compiledExpressions) {
            if (!isCompiled(name)) continue;  // config-level expression — don't expand

            // Find all occurrences of (name[ or !(name[
            std::string pat1 = "(" + name + "[";
            std::string pat2 = "!(" + name + "[";

            for (int pass = 0; pass < 2; ++pass) {
                // Check negated pattern FIRST so !(name[...]) is matched
                // before (name[...]) which would match inside at offset 1.
                const std::string& pat = (pass == 0) ? pat2 : pat1;
                bool negated = (pass == 0);

                std::size_t pos = 0;
                while ((pos = result.find(pat, pos)) != std::string::npos) {
                    // Find matching ]
                    std::size_t bracketStart = pos + pat.size() - 1; // position of [
                    std::size_t bracketEnd = result.find(']', bracketStart);
                    if (bracketEnd == std::string::npos) { pos++; continue; }

                    // Check for closing )
                    std::size_t closePos = bracketEnd + 1;
                    if (closePos >= result.size() || result[closePos] != ')') { pos++; continue; }

                    // Extract args
                    std::string argStr = result.substr(bracketStart + 1, bracketEnd - bracketStart - 1);
                    std::vector<std::string> actualArgs = ce::getArgs("(" + name + "[" + argStr + "])");

                    // Build substitution map: u_1 -> actualArgs[0], etc.
                    std::vector<std::string> sigArgs = ce::getArgs(le.signature);
                    if (sigArgs.size() != actualArgs.size()) { pos++; continue; }
                    std::map<std::string, std::string> subst;
                    for (std::size_t i = 0; i < sigArgs.size(); ++i) {
                        subst[sigArgs[i]] = actualArgs[i];
                    }

                    // Build expanded form based on category
                    std::string expanded;
                    if (le.category == "or" && le.elements.size() >= 2) {
                        // Elements carry each disjunct's true polarity;
                        // negate() cancels a double negation, so a negated
                        // disjunct contributes its bare positive core. A
                        // conjunction has exactly two children, so three or
                        // more conjuncts nest to the left,
                        // !(&(&!D_1!D_2)!D_3) — the shape of the MPL
                        // definitions and of expandSignature. The compiler
                        // collects the conjuncts of either nesting back into
                        // one or with all elements.
                        auto negate = [](const std::string& s) -> std::string {
                            if (!s.empty() && s[0] == '!') return s.substr(1);
                            return "!" + s;
                        };
                        std::string conjunction = negate(ce::replaceKeysInString(le.elements[0], subst));
                        for (std::size_t i = 1; i < le.elements.size(); ++i) {
                            conjunction = "(&" + conjunction
                                + negate(ce::replaceKeysInString(le.elements[i], subst)) + ")";
                        }
                        expanded = "!" + conjunction;
                    } else if (le.category == "and") {
                        // Two children per conjunction, nested to the left.
                        assert(le.elements.size() >= 2
                            && "expandToBaseForm: a compiled and has at least two elements");
                        expanded = ce::replaceKeysInString(le.elements[0], subst);
                        for (std::size_t i = 1; i < le.elements.size(); ++i) {
                            expanded = "(&" + expanded + ce::replaceKeysInString(le.elements[i], subst) + ")";
                        }
                    } else if (le.category == "existence" && le.elements.size() == 2) {
                        // Find the highest integer var used in the whole expression
                        // so the new bound var doesn't collide
                        int maxVar = 0;
                        {
                            // Scan all actual args in substitution values
                            for (const auto& kv : subst) {
                                try { int v = std::stoi(kv.second); if (v > maxVar) maxVar = v; }
                                catch (...) {}
                            }
                            // Also scan the current result string for [digits] patterns
                            for (std::size_t si = 0; si < result.size(); ++si) {
                                if (result[si] == '[' || result[si] == ',') {
                                    std::size_t start = si + 1;
                                    std::size_t end = start;
                                    while (end < result.size() && std::isdigit(static_cast<unsigned char>(result[end]))) end++;
                                    if (end > start && end < result.size() && (result[end] == ',' || result[end] == ']')) {
                                        try { int v = std::stoi(result.substr(start, end - start)); if (v > maxVar) maxVar = v; }
                                        catch (...) {}
                                    }
                                }
                            }
                        }
                        std::string freshVar = std::to_string(maxVar + 1);

                        // Find the raw bound var (non-u_ arg not in subst)
                        std::string rawBoundVar;
                        for (const auto& elem : le.elements) {
                            for (const auto& a : ce::getArgs(elem)) {
                                if (subst.find(a) == subst.end()) { rawBoundVar = a; break; }
                            }
                            if (!rawBoundVar.empty()) break;
                        }

                        // Replace raw bound var with fresh var in left/right
                        std::map<std::string, std::string> boundSubst = subst;
                        boundSubst[rawBoundVar] = freshVar;
                        std::string left = ce::replaceKeysInString(le.elements[0], boundSubst);
                        std::string right = ce::replaceKeysInString(le.elements[1], boundSubst);
                        // Negate the head element WITH double-negation
                        // cancellation — a negated element contributes its
                        // bare positive core; a blind '!' prefix would emit
                        // a malformed !!(...) row that crashes any later
                        // load of the base form.
                        const std::string negRight =
                            (!right.empty() && right[0] == '!')
                                ? right.substr(1) : "!" + right;
                        expanded = "!(>[" + freshVar + "]" + left + negRight + ")";
                    } else if (le.category == "implication") {
                        // (>[bound](premise)(head)) — elements has chain + head
                        // For now, reconstruct from elements
                        std::string body;
                        for (std::size_t i = le.elements.size(); i-- > 0; ) {
                            std::string elem = ce::replaceKeysInString(le.elements[i], subst);
                            if (body.empty()) {
                                body = elem;
                            } else {
                                body = "(>[]" + elem + body + ")";
                            }
                        }
                        expanded = body;
                    } else {
                        pos++;
                        continue;  // unknown category, skip
                    }

                    // Apply negation if we matched !(name[...])
                    if (negated) {
                        // !(or0[...]) with or0 expanding to !(&!a!b) -> !(!(&!a!b)) = (&!a!b)
                        // General: negate the expanded form
                        if (!expanded.empty() && expanded[0] == '!') {
                            expanded = expanded.substr(1);  // double negation cancels
                        } else {
                            expanded = "!" + expanded;
                        }
                    }

                    // Replace in result
                    std::size_t matchLen = closePos + 1 - pos;
                    result.replace(pos, matchLen, expanded);
                    changed = true;
                    // Don't advance pos — re-scan from same position for nested expansions
                }
            }
        }
    }
    return result;
}

/// @brief Mark the existing free argument names as unchangeable.
///
/// @details Preserves expression spelling, binding and the existing transformation order.
/// @param expr Expression text.
/// @return Transformed expression text.
std::string DefinitionExpansion::prefixArgumentsWithU(const std::string& expr) {
    StrSpan args[ExecutionParameters::MAX_ARITY];
    const int argN = getArgsSpans(expr, args, ExecutionParameters::MAX_ARITY);
    if (argN == 0) return expr;   // no args: plain pass-through (empty-map path)

    const unsigned slot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : scratchArenas().slotCount() - 1;   // reserved slot for single-threaded
    ScratchArena& arena = scratchArenas().forSlot(slot);
    ScratchScope scope(arena);

    StrReplacement pairs[ExecutionParameters::MAX_ARITY];
    for (int ai = 0; ai < argN; ++ai) {
        const StrSpan& arg = args[ai];
        // value = "u_" + arg : NEW bytes, bump-allocated on the scratch arena.
        char* v = arena.allocBytes(2 + arg.len);
        v[0] = 'u';
        v[1] = '_';
        if (arg.len > 0) std::memcpy(v + 2, arg.ptr, static_cast<size_t>(arg.len));
        pairs[ai].key = arg;
        pairs[ai].value = StrSpan(v, 2 + arg.len);
    }
    return replaceKeysToString(expr, pairs, argN);
}

/// @brief Remove unchangeable prefixes through token-aware substitution.
///
/// @details Preserves expression spelling, binding and the existing transformation order.
/// @param expr Expression text.
/// @return Transformed expression text.
std::string DefinitionExpansion::removeUPrefixFromArguments(const std::string& expr) {
    StrSpan args[ExecutionParameters::MAX_ARITY];
    const int argN = getArgsSpans(expr, args, ExecutionParameters::MAX_ARITY);
    StrReplacement pairs[ExecutionParameters::MAX_ARITY];
    int pairN = 0;
    for (int ai = 0; ai < argN; ++ai) {
        const StrSpan& arg = args[ai];
        // Only strip "u_" when the argument actually starts with it. The value is
        // the sub-span arg[2:], a slice of expr -- no new bytes, so no arena.
        if (arg.len >= 2 && arg.ptr[0] == 'u' && arg.ptr[1] == '_') {
            pairs[pairN].key = arg;
            pairs[pairN].value = StrSpan(arg.ptr + 2, arg.len - 2);
            ++pairN;
        }
    }
    return replaceKeysToString(expr, pairs, pairN);
}

/// @brief Add the existing unchangeable prefix to eligible expression arguments.
///
/// @details Preserves expression spelling, binding and the existing transformation order.
/// @param expr Expression text.
/// @return Transformed expression text.
std::string DefinitionExpansion::addMissingU(const std::string& expr) {
    std::vector<std::string> args = ce::getArgs(expr);
    std::map<std::string, std::string> replacementMap;

    for (const std::string& arg : args) {
        // Check if the argument does NOT start with "u_"
        if (!startsWithSpan(StrSpan(arg), "u_", 2) && arg != "marker") {
            replacementMap[arg] = "u_" + arg;
        }
    }

    // Replace the arguments in the expression string
    return ce::replaceKeysInString(expr, replacementMap);
}

/// @brief Replace unchangeable-name occurrences using the existing compiler spelling.
///
/// @details Preserves expression spelling, binding and the existing transformation order.
/// @param s Expression text.
/// @return Transformed expression text.
std::string DefinitionExpansion::replaceUSubstrings(const std::string& s) {
    std::string out = s;
    if (out.size() < 2) return out;

    std::size_t i = 1; // we look back at i-1
    while (i + 1 < out.size()) {
        if (out[i] == 'u' && out[i + 1] == '_' &&
            (out[i - 1] == '[' || out[i - 1] == ',')) {
            // erase "u_"
            out.erase(i, 2);
            // do not advance i; check again at the same spot
            continue;
        }
        ++i;
    }
    return out;
}

/// @brief Remove unchangeable prefixes from expression-tree argument tokens.
///
/// @details Preserves expression spelling, binding and the existing transformation order.
/// @param expr Expression text.
/// @return Transformed expression text.
std::string DefinitionExpansion::stripUPrefixAST(const std::string& expr) {
            std::set<std::string> tokens;
            ce::TreeNode1* root = ce::parseExpr(expr);

            std::vector<ce::TreeNode1*> stack;
            if (root) stack.push_back(root);

            while (!stack.empty()) {
                ce::TreeNode1* curr = stack.back();
                stack.pop_back();

                std::vector<std::string> args = ce::getArgs(curr->value);
                for (const std::string& a : args) {
                    if (!a.empty()) tokens.insert(a);
                }

                if (curr->left) stack.push_back(curr->left);
                if (curr->right) stack.push_back(curr->right);
            }

            ce::deleteTree(root);

            std::map<std::string, std::string> renameMap;
            for (const std::string& t : tokens) {
                if (t.size() >= 2 && t[0] == 'u' && t[1] == '_') {
                    renameMap[t] = t.substr(2);
                }
            }

            return ce::replaceKeysInString(expr, renameMap);
        }

/// @brief Read the two defining sides of a compound expression.
///
/// @details Preserves expression spelling, binding and the existing transformation order.
/// @param s Compound expression text.
/// @param coreExpressionMap Existing configuration view retained by this interface.
/// @return Left and right defining expressions.
std::pair<std::string, std::string> DefinitionExpansion::extractValues(const std::string& s, const ce::CoreExpressionMap& coreExpressionMap) {
            ce::TreeNode1* root = ce::parseExpr(s);


            std::string left = ce::treeToExpr(root ? root->left : NULL);
            std::string right = ce::treeToExpr(root ? root->right : NULL);

            ce::deleteTree(root);
            return std::make_pair(left, right);
        }

/// @brief Append the conjuncts of a two-child conjunction chain in leaf order.
///
/// @details
/// A conjunction has exactly two children, so a longer conjunction nests,
/// to the left `(&(&A B)C)` or to the right `(&A(&B C))`. A positive `&`
/// expression is split into its two children (`extractValues`) and each is
/// collected recursively; any other expression — an atom, an implication, a
/// negated conjunction `!(&..)` — is one conjunct. Both nestings of the same
/// conjuncts therefore yield the same ordered list (associativity of `&`).
///
/// @param expression        A conjunct or a positive conjunction; never empty.
/// @param coreExpressionMap Configuration view forwarded to `extractValues`.
/// @param conjuncts         Receives the conjuncts in left-to-right leaf order
///                          (appended).
/// @return Nothing.
/// @see extractValues; StructuralCompiler::compileExpression — the negated
///      conjunction (or) branch that collects its disjuncts through this.
void DefinitionExpansion::collectConjuncts(const std::string& expression,
    const ce::CoreExpressionMap& coreExpressionMap, std::vector<std::string>& conjuncts) {
    assert(!expression.empty() && "collectConjuncts: empty conjunct");
    if (expression.compare(0, 2, "(&") != 0) {
        conjuncts.push_back(expression);
        return;
    }
    const std::pair<std::string, std::string> children = extractValues(expression, coreExpressionMap);
    collectConjuncts(children.first, coreExpressionMap, conjuncts);
    collectConjuncts(children.second, coreExpressionMap, conjuncts);
}

/// @brief Toggle one leading expression negation.
///
/// @details Preserves expression spelling, binding and the existing transformation order.
/// @param expr Expression text.
/// @return Transformed expression text.
std::string DefinitionExpansion::negate(std::string expr) {
    if (!expr.empty() && expr.front() == '!') return expr.substr(1);
    return "!" + expr;
}




/// @brief Scratch twin of @ref prefixArgumentsWithU — `u_`-prefix EVERY argument
///        of @p expr onto the scratch tier, with no heap `std::string`.
///
/// @details
/// Prefixes ALL arguments (contrast @ref prefixNonMarkerArgumentsWithUScratch,
/// which SKIPS the literal `"marker"`). Each argument yields one substitution
/// pair whose value `"u_"+arg` is FRESH bytes bump-allocated on @p arena (the
/// `prefixArgumentsWithU` idiom); the substitution runs through
/// `replaceKeysScratch` (the byte-exact greedy-longest twin of
/// `ce::replaceKeysInString`). A no-argument input contributes zero pairs and
/// degrades to a plain pass-through copy — byte-identical to the oracle's `return
/// expr`. An already-`u_` argument is prefixed AGAIN (`u_u_arg`), exactly as the
/// string oracle does — NOT special-cased. The result rides @p arena (the caller
/// owns the `ScratchScope`) and survives a later interner mint.
///
/// @param arena Scratch (string-tier) arena receiving the fresh value bytes and
///              the substituted result.
/// @param expr  Expression whose arguments are prefixed, as a span over stable input.
/// @return The substituted text as a `ScratchString` on @p arena.
/// @invariant Byte-identical to `prefixArgumentsWithU`; mints into no interner.
/// @see prefixArgumentsWithU (the retained heap oracle),
///      prefixNonMarkerArgumentsWithUScratch (the marker-skipping sibling),
///      replaceKeysScratch, checkNecessityForEquality (the consumer).
ScratchString DefinitionExpansion::prefixArgumentsWithUScratch(ScratchArena& arena, StrSpan expr) {
    StrSpan args[ExecutionParameters::MAX_ARITY];
    const int argN = getArgsSpans(expr, args, ExecutionParameters::MAX_ARITY);

    StrReplacement pairs[ExecutionParameters::MAX_ARITY];
    for (int ai = 0; ai < argN; ++ai) {
        const StrSpan& arg = args[ai];
        // value = "u_" + arg : NEW bytes, bump-allocated on the scratch arena.
        char* v = arena.allocBytes(2 + arg.len);
        v[0] = 'u';
        v[1] = '_';
        if (arg.len > 0) std::memcpy(v + 2, arg.ptr, static_cast<size_t>(arg.len));
        pairs[ai].key = arg;
        pairs[ai].value = StrSpan(v, 2 + arg.len);
    }
    return replaceKeysScratch(arena, expr, pairs, argN);
}


/// @brief `u_`-prefix every NON-`"marker"` argument of `expr` onto the scratch
///        tier — the faithful twin of `isAdmittedIntegration`'s inline
///        `{arg != "marker" -> u_+arg}` rename.
///
/// @details
/// Unlike `prefixArgumentsWithU`, which prefixes ALL arguments, this helper
/// skips any argument equal to the literal `"marker"`, leaving it verbatim.
/// Each non-marker argument yields one substitution pair whose value
/// `"u_"+arg` is FRESH bytes bump-allocated on `arena`; the substitution runs
/// through `replaceKeysScratch` (the byte-exact greedy-longest twin of
/// `ce::replaceKeysInString`). A no-argument or all-`"marker"` input
/// contributes zero pairs and degrades to a plain pass-through copy. The
/// result rides `arena` (the caller owns the `ScratchScope`) and survives a
/// later interner mint (the interner writes its own cold pool, not this arena).
///
/// @param arena Scratch (string-tier) arena receiving the fresh value bytes
///              and the substituted result.
/// @param expr  Expression whose arguments are prefixed, as a span over stable
///              input.
/// @return The substituted text as a `ScratchString` on `arena`.
/// @see prefixArgumentsWithU (contrast — prefixes ALL args),
///      replaceKeysScratch, isAdmittedIntegration.
ScratchString DefinitionExpansion::prefixNonMarkerArgumentsWithUScratch(ScratchArena& arena, StrSpan expr) {
    StrSpan args[ExecutionParameters::MAX_ARITY];
    const int argN = getArgsSpans(expr, args, ExecutionParameters::MAX_ARITY);

    StrReplacement pairs[ExecutionParameters::MAX_ARITY];
    int pairN = 0;
    for (int ai = 0; ai < argN; ++ai) {
        const StrSpan& arg = args[ai];
        if (equalSpans(arg, StrSpan("marker", 6))) continue;   // marker stays verbatim
        // value = "u_" + arg : NEW bytes, bump-allocated on the scratch arena.
        char* v = arena.allocBytes(2 + arg.len);
        v[0] = 'u';
        v[1] = '_';
        if (arg.len > 0) std::memcpy(v + 2, arg.ptr, static_cast<size_t>(arg.len));
        pairs[pairN].key = arg;
        pairs[pairN].value = StrSpan(v, 2 + arg.len);
        ++pairN;
    }
    return replaceKeysScratch(arena, expr, pairs, pairN);
}

} // namespace gl

namespace gl {
/// @brief Expand a `LogicalEntity`'s signature into its full expression form
///        (the AND / existence / implication / OR De-Morgan build followed by
///        the `u_`/`it_`/`int_`/`c_` variable renaming) onto a caller arena — 0%
///        heap.
///
/// @details
/// The arena-builder twin of the former heap `expandSignature`. Every string is
/// built with an EXPLICIT length on @p out's byte-bump (string) tier
/// (`assert(at == n)` per the fresh-string doctrine), and the former
/// `std::set<std::string>` / `std::vector<std::string>` scratch became fixed
/// stack `StrSpan` arrays sorted by `compareSpans` (byte-identical to the
/// `std::set` iteration order). There is NO local `ScratchScope`: the returned
/// `ScratchString`, the intermediate build bytes, and the `c_` rename values all
/// ride @p out, held live until the CALLER's scope reclaims it (09c pitfall 4).
///
/// Per category: AND / OR build the left-nested `(& …)` / `!(& …)` forms
/// incrementally; EXISTENCE builds `!(>[boundVars]body negatedHead)` where the
/// bound vars are the body args NOT in the signature (sorted-unique); IMPLICATION
/// returns `reconstructImplicationFullBindScratch(out, …)` DIRECTLY (no renaming,
/// matching the heap's early return). The final renaming collects tokens via
/// `collectExprTokens` (every variable token is inside a `[...]`) and runs
/// `replaceKeysScratch` — the byte-exact arena twin of the former
/// `replaceKeysToString`.
///
/// @param le  The `LogicalEntity` to expand (category one of and/existence/
///            implication/or; signature args asserted `u_`-prefixed).
/// @param out Caller string-tier scratch arena receiving the result AND all
///            intermediate build bytes; must outlive the caller's use of the
///            returned span.
/// @return The expanded, renamed signature as a `ScratchString` on @p out.
/// @invariant Allocates on the heap NOTHING; builds only on @p out; mints no
///            interner (the callers' downstream doors mint, never @p out — I-3).
/// @see reconstructImplicationFullBindScratch, replaceKeysScratch, collectExprTokens.
ScratchString DefinitionExpansion::expandSignature(const LogicalEntity& le, ScratchArena& out) {
    // LE forwarder onto the span-run body (the additive twin) — build a stack
    // StrSpan run over le.elements (caller-stable), byte-identical.
    StrSpan elemRun[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS + 1];
    const int32_t elemFwdN = static_cast<int32_t>(le.elements.size());
    assert(elemFwdN <= ExecutionParameters::MAX_INSTRUCTION_ELEMENTS + 1
        && "expandSignature forwarder: element count exceeds cap");
    for (int32_t i = 0; i < elemFwdN; ++i) elemRun[i] = StrSpan(le.elements[i]);
    return expandSignature(StrSpan(le.category), StrSpan(le.signature),
                           elemRun, elemFwdN, out);
}

/// @brief Span-run overload of @ref expandSignature — the span-native
        ///        body; the `const LogicalEntity&` form forwards.
        ///
        /// @details
        /// Byte-identical to the `LogicalEntity` form (same AND / existence /
        /// implication / OR De-Morgan build and `u_`/`it_`/`int_`/`c_` renaming on
        /// the same bytes), but reads the category / signature / elements as spans
        /// / a caller-owned `StrSpan` run instead of a `LogicalEntity`, so a
        /// span-native caller (`disintegrateExprCore2`'s history tracker) reaches
        /// it 0% heap. The `LogicalEntity` form builds a stack `StrSpan` run over
        /// `le.elements` and delegates here.
        ///
        /// @param category The category (and/existence/implication/or), or an
        ///                 expansion-local counterexample whose last element
        ///                 is the negation of the original implication head.
        /// @param signature The signature span (args asserted `u_`-prefixed).
        /// @param elements  The entity elements as a `StrSpan` run; cap
        ///                  `MAX_INSTRUCTION_ELEMENTS + 1`.
        /// @param elemN     The element count.
        /// @param out       Caller string-tier scratch arena receiving the result
        ///                  AND all intermediate bytes; must outlive the use.
        /// @return The expanded, renamed signature as a `ScratchString` on @p out.
        /// @invariant Allocates nothing on the heap; builds only on @p out.
        /// @see expandSignature(const LogicalEntity&, ScratchArena&) — the twin
        ///      oracle; reconstructImplicationFullBindScratch, replaceKeysScratch.
ScratchString DefinitionExpansion::expandSignature(StrSpan category,
    StrSpan signature, const StrSpan* elements, int32_t elemN, ScratchArena& out) {
    // 1. Assert that all signature arguments start with 'u_'.
    StrSpan sigArgsVec[ExecutionParameters::MAX_ARITY];
    const int32_t sigArgsN = getArgsSpans(signature, sigArgsVec,
                                          ExecutionParameters::MAX_ARITY);
    for (int32_t i = 0; i < sigArgsN; ++i) {
        assert(sigArgsVec[i].len >= 2 && sigArgsVec[i].ptr[0] == 'u'
            && sigArgsVec[i].ptr[1] == '_'
            && "Signature arguments must start with 'u_' as per assumption.");
    }

    StrSpan result;

    // --- CASE 1: AND (&) ---
    if (equalSpans(category, StrSpan("and", 3))) {
        if (elemN == 0) result = signature;
        else if (elemN == 1) result = elements[0];
        else {
            // Build the left-nested "(& current elem)" hierarchy on `out`.
            StrSpan current = elements[0];
            for (int32_t i = 1; i < elemN; ++i) {
                const StrSpan elem = elements[i];
                const int32_t n = 2 + current.len + elem.len + 1;  // "(&" ... ")"
                char* buf = out.allocBytes(n);
                int32_t at = 0;
                buf[at++] = '('; buf[at++] = '&';
                std::memcpy(buf + at, current.ptr, static_cast<size_t>(current.len)); at += current.len;
                std::memcpy(buf + at, elem.ptr, static_cast<size_t>(elem.len)); at += elem.len;
                buf[at++] = ')';
                assert(at == n);
                current = StrSpan(buf, n);
            }
            result = current;
        }
    }

    // --- CASE 2: EXISTENCE ---
    else if (equalSpans(category, StrSpan("existence", 9))) {
        // Expecting at least 2 elements: [0]=body/definition, [1]=head/claim
        if (elemN < 2) result = signature;
        else {
            const StrSpan body = elements[0];
            const StrSpan head = elements[1];

            // Handle Head Negation: strip a leading '!' or prepend one.
            StrSpan negatedHead;
            if (head.len > 0 && head.ptr[0] == '!') {
                negatedHead = StrSpan(head.ptr + 1, head.len - 1);
            }
            else {
                const int32_t hn = 1 + head.len;
                char* hb = out.allocBytes(hn);
                hb[0] = '!';
                std::memcpy(hb + 1, head.ptr, static_cast<size_t>(head.len));
                negatedHead = StrSpan(hb, hn);
            }

            // Bound variables: body args NOT in the signature, sorted-unique
            // (byte-identical to the former std::set<std::string> order).
            StrSpan bodyArgs[ExecutionParameters::MAX_ARITY];
            const int32_t bodyArgsN = getArgsSpans(body, bodyArgs,
                                                   ExecutionParameters::MAX_ARITY);
            StrSpan bvBuf[ExecutionParameters::MAX_ARITY];
            int32_t bvN = 0;
            for (int32_t i = 0; i < bodyArgsN; ++i) {
                bool inSig = false;
                for (int32_t j = 0; j < sigArgsN; ++j)
                    if (equalSpans(bodyArgs[i], sigArgsVec[j])) { inSig = true; break; }
                if (!inSig) {
                    assert(bvN < ExecutionParameters::MAX_ARITY
                        && "expandSignature: bound-var count exceeds MAX_ARITY");
                    bvBuf[bvN++] = bodyArgs[i];
                }
            }
            std::sort(bvBuf, bvBuf + bvN, [](const StrSpan& a, const StrSpan& b) {
                return compareSpans(a, b) < 0;
            });
            StrSpan boundVars[ExecutionParameters::MAX_ARITY];
            int32_t boundN = 0;
            for (int32_t i = 0; i < bvN; ++i)
                if (i == 0 || !equalSpans(bvBuf[i], bvBuf[i - 1]))
                    boundVars[boundN++] = bvBuf[i];

            // result = "!(>[" + join(boundVars, ",") + "]" + body + negatedHead + ")"
            int32_t varsLen = 0;
            for (int32_t i = 0; i < boundN; ++i) varsLen += boundVars[i].len;
            if (boundN > 0) varsLen += boundN - 1;  // separating commas
            const int32_t n = 4 + varsLen + 1 + body.len + negatedHead.len + 1;
            char* buf = out.allocBytes(n);
            int32_t at = 0;
            buf[at++] = '!'; buf[at++] = '('; buf[at++] = '>'; buf[at++] = '[';
            for (int32_t i = 0; i < boundN; ++i) {
                if (i > 0) buf[at++] = ',';
                std::memcpy(buf + at, boundVars[i].ptr, static_cast<size_t>(boundVars[i].len));
                at += boundVars[i].len;
            }
            buf[at++] = ']';
            std::memcpy(buf + at, body.ptr, static_cast<size_t>(body.len)); at += body.len;
            std::memcpy(buf + at, negatedHead.ptr, static_cast<size_t>(negatedHead.len)); at += negatedHead.len;
            buf[at++] = ')';
            assert(at == n);
            result = StrSpan(buf, n);
        }
    }

    else if (equalSpans(category, StrSpan("counterexample", 14))) {
        assert(elemN >= 2);
        // Restore the original head before reconstructing its universal.
        // Only the admitted witness is non-u_; free instance parameters keep
        // their prefixes until the shared history renaming below.
        const StrSpan head = gl::DefinitionExpansion::negateScratch(out, elements[elemN - 1]);
        const ScratchString universal = gl::ImplicationBuilder::reconstructImplicationFullBindScratch(
            out, elements, elemN - 1, StrSpan(head));
        result = gl::DefinitionExpansion::negateScratch(out, StrSpan(universal));
    }

    // --- CASE 3: IMPLICATION (returns early, before the renaming) ---
    else if (equalSpans(category, StrSpan("implication", 11))) {
        if (elemN == 0) return copyScratch(out, signature);
        // chain = elements minus the last (head); no owning std::vector.
        const StrSpan head = elements[elemN - 1];
        const int32_t chainN = elemN - 1;
        assert(chainN <= 64 && "expandSignature: implication chain exceeds 64");
        StrSpan chainSpans[64];
        for (int32_t i = 0; i < chainN; ++i) chainSpans[i] = elements[i];
        return gl::ImplicationBuilder::reconstructImplicationFullBindScratch(out, chainSpans, chainN, head);
    }

    // --- CASE 4: OR ---
    else if (equalSpans(category, StrSpan("or", 2))) {
        if (elemN == 0) result = signature;
        else if (elemN == 1) result = elements[0];
        else {
            // Rebuild nested !(&!(...) !(...)) from the flat disjunct list on `out`.
            // Elements carry each disjunct's TRUE polarity, so a disjunct's
            // conjunct form is its negation WITH double-negation cancellation:
            // a negated element contributes its bare positive core, a positive
            // element gains the '!' prefix. Byte-identical to the former
            // blind-prefix builder for all-positive elements.
            const auto negLen = [](const StrSpan& e) -> int32_t {
                return (e.len > 0 && e.ptr[0] == '!') ? e.len - 1 : e.len + 1;
            };
            const auto putNeg = [](char* buf, int32_t at, const StrSpan& e) -> int32_t {
                if (e.len > 0 && e.ptr[0] == '!') {
                    std::memcpy(buf + at, e.ptr + 1, static_cast<size_t>(e.len - 1));
                    return at + e.len - 1;
                }
                buf[at++] = '!';
                std::memcpy(buf + at, e.ptr, static_cast<size_t>(e.len));
                return at + e.len;
            };
            const StrSpan e0 = elements[0];
            const StrSpan e1 = elements[1];
            StrSpan current;
            {
                const int32_t n = 3 + negLen(e0) + negLen(e1) + 1;  // "!(&" neg(e0) neg(e1) ")"
                char* buf = out.allocBytes(n);
                int32_t at = 0;
                buf[at++] = '!'; buf[at++] = '('; buf[at++] = '&';
                at = putNeg(buf, at, e0);
                at = putNeg(buf, at, e1);
                buf[at++] = ')';
                assert(at == n);
                current = StrSpan(buf, n);
            }
            for (int32_t i = 2; i < elemN; ++i) {
                // The or-so-far is a DISJUNCT of the next level, so it
                // enters the AND negated like any other disjunct — via
                // putNeg's double-negation cancellation its `!(&…)` form
                // contributes the bare positive `(&…)`:
                // `!(&(&!D_1!D_2)!D_3)` = (D_1 ∨ D_2) ∨ D_3. Inserting
                // `current` un-negated read as ¬(D_1 ∨ D_2) ∨ D_3 — the
                // mirrored polarity defect flagged in D-260, live once
                // ≥3-element ors are consumed in-run.
                const StrSpan elem = elements[i];
                const int32_t n = 3 + negLen(current) + negLen(elem) + 1;  // "!(&" neg(current) neg(elem) ")"
                char* buf = out.allocBytes(n);
                int32_t at = 0;
                buf[at++] = '!'; buf[at++] = '('; buf[at++] = '&';
                at = putNeg(buf, at, current);
                at = putNeg(buf, at, elem);
                buf[at++] = ')';
                assert(at == n);
                current = StrSpan(buf, n);
            }
            result = current;
        }
    }

    // --- ERROR ---
    else {
        assert(false && "Forbidden category encountered in expandSignature. Must be 'and', 'existence', 'implication', or 'or'.");
        return copyScratch(out, signature);
    }

    // --- FINAL RENAMING (Implication handled by its early return above) ---
    // Every variable token in `result` is inside some "[...]", so collectExprTokens
    // (a flat bracket scan) yields the same token set as the former ce::parseExpr
    // tree walk. Rules: a u_ arg loses the u_ (a slice); it_/int_ args are kept;
    // every other arg gains a c_ prefix (fresh bytes on `out`, held live -- NO local
    // ScratchScope, the caller owns `out`). replaceKeysScratch is the byte-exact
    // arena twin of the former replaceKeysToString.
    StrReplacement pairs[512];
    int32_t pairN = 0;
    int esLevel = 0, esId = 0;
    collectExprTokens(result, [&](const StrSpan& t) {
        if (t.len >= 2 && t.ptr[0] == 'u' && t.ptr[1] == '_') {
            assert(pairN < 512 && "expandSignature rename pairs exceed 512");
            pairs[pairN++] = StrReplacement{ t, StrSpan(t.ptr + 2, t.len - 2) };
        }
        else if (matchItLevId(t, esLevel, esId) || matchDisintegrationWitnessId(t, esLevel, esId)) {
            // it_ / int_ tokens are kept as-is (no replacement).
        }
        else {
            assert(pairN < 512 && "expandSignature rename pairs exceed 512");
            char* cval = out.allocBytes(2 + t.len);
            cval[0] = 'c'; cval[1] = '_';
            std::memcpy(cval + 2, t.ptr, static_cast<size_t>(t.len));
            pairs[pairN++] = StrReplacement{ t, StrSpan(cval, 2 + t.len) };
        }
    });

    return replaceKeysScratch(out, result, pairs, pairN);
}

/// @brief Collect the distinct non-`u_` argument names of an existence
///        `LogicalEntity` (the "last removed" bound-var candidates) into a
///        caller span buffer, sorted-unique — 0% heap.
///
/// @details
/// The span twin of the former `std::set<std::string>` collector: over every
/// `le.elements[i]`, `getArgsSpans` slices the flat `[...]` args and each arg
/// that does NOT begin `u_` is gathered as a `StrSpan` INTO `le.elements`
/// (input lifetime). The gather buffer is then `compareSpans`-sorted and
/// adjacent-deduplicated, reproducing the former `std::set<std::string>`
/// iteration order byte-for-byte (`compareSpans` == `std::string::compare`),
/// and the distinct run is copied into @p out. No arena is reached — every
/// output span aliases @p le, which must outlive the caller's use.
///
/// @param le  Existence `LogicalEntity` (asserted `category == "existence"`).
/// @param out Caller buffer receiving the distinct sorted argument spans.
/// @param cap Capacity of @p out (loud Rule-19 assert on overflow).
/// @return The number of distinct non-`u_` arguments written to @p out.
/// @invariant Allocates nothing; output spans alias @p le.elements.
/// @see getArgsSpans, compareSpans, equalSpans.
int32_t DefinitionExpansion::listLastRemovedArgsLE(const LogicalEntity& le,
                                                  StrSpan* out, int32_t cap) {
    // LE forwarder onto the span-run body (the additive twin) — build a stack
    // StrSpan run over le.elements (caller-stable), byte-identical.
    StrSpan elemRun[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS + 1];
    const int32_t elemFwdN = static_cast<int32_t>(le.elements.size());
    assert(elemFwdN <= ExecutionParameters::MAX_INSTRUCTION_ELEMENTS + 1
        && "listLastRemovedArgsLE forwarder: element count exceeds cap");
    for (int32_t i = 0; i < elemFwdN; ++i) elemRun[i] = StrSpan(le.elements[i]);
    return listLastRemovedArgsLE(StrSpan(le.category), elemRun, elemFwdN, out, cap);
}

/// @brief Span-run overload of @ref listLastRemovedArgsLE — the
        ///        span-native body; the `const LogicalEntity&` form forwards.
        ///
        /// @details
        /// Byte-identical to the `LogicalEntity` form (same non-`u_` gather over
        /// the elements, same `compareSpans` sort + adjacent-dedup into the
        /// `std::set<std::string>` order), but reads the category / elements as
        /// spans / a caller-owned `StrSpan` run, so a span-native caller
        /// (`disintegrateExprCore2`) reaches it 0% heap. Output spans alias the
        /// caller's @p elements (input lifetime). The `LogicalEntity` form builds
        /// a stack `StrSpan` run over `le.elements` and delegates here.
        ///
        /// @param category Witness category (`existence` or the expansion-local
        ///                 `counterexample` instruction).
        /// @param elements The entity elements as a `StrSpan` run; cap
        ///                 `MAX_INSTRUCTION_ELEMENTS + 1`.
        /// @param elemN    The element count.
        /// @param out      Caller buffer receiving the distinct sorted arg spans.
        /// @param cap      Capacity of @p out (loud Rule-19 assert on overflow).
        /// @return The number of distinct non-`u_` arguments written to @p out.
        /// @invariant Allocates nothing; output spans alias @p elements.
        /// @see listLastRemovedArgsLE(const LogicalEntity&, StrSpan*, int32_t) —
        ///      the twin oracle; getArgsSpans, compareSpans.
int32_t DefinitionExpansion::listLastRemovedArgsLE(StrSpan category,
    const StrSpan* elements, int32_t elemN, StrSpan* out, int32_t cap) {
    // Only witness-bearing instruction nodes have removed arguments.
    assert((equalSpans(category, StrSpan("existence", 9))
        || equalSpans(category, StrSpan("counterexample", 14)))
        && "Instruction must describe a witness construction");

    // Gather every non-'u_' arg span (slices of the elements, input lifetime).
    StrSpan buf[ExecutionParameters::MAX_KEY_SLOTS];
    int32_t bufN = 0;
    for (int32_t e = 0; e < elemN; ++e) {
        const StrSpan element = elements[e];
        StrSpan args[ExecutionParameters::MAX_ARITY];
        const int32_t argsN = getArgsSpans(element, args,
                                           ExecutionParameters::MAX_ARITY);
        for (int32_t a = 0; a < argsN; ++a) {
            // Check if arg does NOT start with "u_"
            if (!(args[a].len >= 2 && args[a].ptr[0] == 'u' && args[a].ptr[1] == '_')) {
                assert(bufN < ExecutionParameters::MAX_KEY_SLOTS
                    && "listLastRemovedArgsLE: gathered arg count exceeds cap");
                buf[bufN++] = args[a];
            }
        }
    }

    // Sort + adjacent-dedup -> the std::set<std::string> order, byte-identically.
    std::sort(buf, buf + bufN, [](const StrSpan& a, const StrSpan& b) {
        return compareSpans(a, b) < 0;
    });
    int32_t outN = 0;
    for (int32_t i = 0; i < bufN; ++i) {
        if (i == 0 || !equalSpans(buf[i], buf[i - 1])) {
            assert(outN < cap
                && "listLastRemovedArgsLE: distinct arg count exceeds out cap");
            out[outN++] = buf[i];
        }
    }
    return outN;
}
}
