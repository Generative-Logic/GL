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
#include "../compiler/compiled_definition_registry.hpp"
#include "../compiler/definition_expansion.hpp"
#include "../compiler/implication_builder.hpp"
#include "../proof/worker_context.hpp"
#include "../infra/rt_tracker.hpp"
#include <cstdio>
namespace gl {
namespace tests { struct IntegrationInstructionFixture; }
/// @brief Build integration signatures and instructions in their existing order.
/// @details Uses the supplied block and arenas without owning proof coordination state.
class IntegrationInstructionBuilder {
friend struct tests::IntegrationInstructionFixture;
public:
/// @brief Bind definitions used to prepare integration instructions.
/// @details Construction binds the reader only; instructions and fresh names use caller storage.
/// @param definitions Read-only compiled definitions and configured signatures.
/// @return Constructed integration instruction builder.
IntegrationInstructionBuilder(const CompiledDefinitionRegistry& definitions);

private:
/// @brief Build an implication around the supplied bound variables and expression.
/// @details Preserves binder treatment, expression spelling, output order and allocation ownership.
/// @param key Ordered premise expressions.
/// @param value Consequent expression.
/// @return The constructed result in the existing representation.
inline std::string reconstructImplicationForIntegration(const std::vector<std::string>& key,
            const std::string& value) {
            std::string result = gl::ImplicationBuilder::reconstructImplicationFullBind(key, value);

            // Find outermost >[...] content
            if (result.size() > 3 && result[0] == '(' && result[1] == '>' && result[2] == '[') {
                std::size_t bracketEnd = result.find(']', 3);
                if (bracketEnd != std::string::npos && bracketEnd > 3) {
                    std::string bv = result.substr(3, bracketEnd - 3);

                    // Check if bound var matches pi_lev_ prefix
                    if (bv.size() >= 7 && bv[0] == 'p' && bv[1] == 'i' && bv[2] == '_'
                        && bv[3] == 'l' && bv[4] == 'e' && bv[5] == 'v' && bv[6] == '_') {
                        return result; // unoccupied pi_lev_ bound var — return unchanged
                    }

                    // Occupied bound var — remove from >[...] → >[]
                    result = "(>[]" + result.substr(bracketEnd + 1);
                }
            }

            return result;
        }

public:
/**
         * @brief Constructs a sorted, formatted string list of arguments from a mapping, stripping "u_" prefixes.
         *
         * Iterates through the provided map \p mp to find entries where the *value* indicates an "unchangeable"
         * argument (starts with "u_" followed by an integer, e.g., "u_1"). It sorts these entries based on
         * the integer value of the target "u_N" string.
         *
         * The output is a bracketed list string (e.g., "[arg1,arg2]") formed from the *keys* corresponding
         * to these values. Before adding a key to the list, the function asserts that the key itself starts
         * with "u_" and strips this prefix (e.g., "u_x" becomes "x").
         *
         * @param mp A map where values are checked for the "u_N" pattern to determine inclusion and order.
         * @return A string representation of the list of keys (stripped of "u_") sorted by their mapped "u_N" values.
         *
         * @note Asserts that any value starting with "u_" is followed by a valid integer.
         * @note Asserts that any key added to the output list starts with "u_" and contains exactly one "u_" substring.
         */





        /**
         * @brief Creates a reverse mapping for "unchangeable" identifiers based on two association maps.
         *
         * Iterates through map \p b to find values conforming to the "unchangeable" pattern
         * (starting with "u_" followed by an integer). For each such value (e.g., "u_1"),
         * the function creates a mapping in the result where the key is "u_1" and the value
         * is taken from map \p a using the same key found in \p b.
         *
         * Effectively performs: result[b[k]] = a[k]
         *
         * @param a Map containing the target values to be mapped to the unchangeable keys.
         * @param b Map defining the association between original keys and "u_N" identifiers.
         * @return A new map { "u_N" : value_from_a }.
         *
         * @note Asserts that any value in \p b starting with "u_" is followed by a valid integer.
         * @note Asserts that for every key in \p b mapping to a "u_" value, a corresponding key exists in \p a.
         */




        /**
         * @brief Normalizes a logical construction to avoid duplicate definitions for structurally identical expressions.
         *
         * Checks if the combination of \p encodedTail and \p encodedHead results in a logical structure
         * (normalized key) that has already been defined. If an isomorphic structure exists (accounting for
         * variable permutations), it returns a call to the existing core expression with mapped arguments.
         * Otherwise, it registers a new logical entity.
         *
         * @param encodedTail The list of expressions forming the body or antecedent of the construction.
         * @param encodedHead The list of expressions forming the head or consequent of the construction.
         * @param newCoreExpressionName The proposed name for the new core expression if creation is necessary (e.g., "implication5").
         * @param category The logical category of the expression (e.g., "implication", "and", "existence").
         * @param force If \c true, bypasses the canonical lookup and forces the registration of \p defaultSignature.
         * @param defaultSignature The explicit signature to enforce when \p force is \c true.
         * @return A string representing the normalized expression call, typically "(CoreName[args...])".
         */











        // Turn optimizations OFF for just this section
//#pragma optimize("", off)


        /// @brief Instantiate a compiled integration template into @p instruction
        ///        — the span-native, arena-recursing integration core.
        ///
        /// @details
        /// Byte-for-byte the former heap body, plus the one-entity-per-signature
        /// probe: (1) the core is
        /// `extractExpressionUniversalSpan`; the `leEntry == nullptr` fast path
        /// returns on the `">"` universal, else asserts (a `std::cout`
        /// diagnostic materializes ONLY inside the abort branch, Rule 19). (2)
        /// The signature→expression argument map becomes a `StrReplacement` run
        /// over `getArgsSpans` spans (sig keys → expr-arg slices). (3) The
        /// instantiated signature is built from those pairs alone (template
        /// signatures carry only `u_` arguments and `replaceKeysScratch` is
        /// token-boundary, so the bound-variable pairs minted later can never
        /// rewrite it). Atomic input polarity is retained before probing the
        /// already-committed entities. On a hit the call RETURNS — the entity table is the unfolded definition
        /// graph, one node per distinct signature, and every consumer resolves
        /// children by first-match signature scan. Re-unfolding a repeated
        /// occurrence would mint divergent `pi_lev_` placeholders for the same
        /// node and break the duplicate-identity contract `flattenOrLeaves`
        /// asserts. (4) The
        /// `existence` branch mints a fresh `pi_lev_<level>_<startIntPi>` value
        /// per FIRST-SEEN bound (non-`u_`) element argument (the map's
        /// `find==end` dedup is a linear span-scan over the pairs built so far;
        /// `startIntPi++` order FROZEN; skipped entities burn no ids). (5) Each
        /// element is
        /// rewritten via `replaceKeysScratch` (the byte-exact greedy-longest twin
        /// of `ce::replaceKeysInString`) and committed with the step-(3)
        /// signature; the new elements are held
        /// on the arena for (6) the `existence`/`and`/`or` recursion, which
        /// re-enters this span body per element. All fresh bytes ride ONE
        /// `ScratchScope` on the string-tier arena; each nested recursion opens
        /// its own nested scope above this call's allocations, so the new-element
        /// spans survive the whole recursion loop (byte-bump LIFO).
        ///
        /// @param expression   The integration-template expression (span over
        ///                      stable input).
        /// @param instruction  The working instruction to append into.
        /// @param level        The level written into every fresh
        ///                      `pi_lev_<level>_<n>` name (a block's `level`;
        ///                      0 at compile time).
        /// @param nameCounter  The next fresh-name number `<n>`, advanced once
        ///                      per minted name (a block's `startIntPi`; a
        ///                      local counter at compile time).
        /// @param goal         The marked goal to set on @p instruction (span
        ///                      source; read-only, must outlive the call).
        /// @param disintegration Enable negative-OR conjunctions and one-witness
        ///                      negative implications, never integration demand.
        /// @param rootEntity   The entity to instantiate for @p expression
        ///                      instead of its registry entry, or `nullptr`.
        ///                      `prepareIntegration` passes the extensionality
        ///                      entity of a set- or relation-typed `=` goal
        ///                      (I-267); it applies to this
        ///                      call only, never to the recursion.
        /// @return Void.
        /// @invariant Mints into no interner (pure string/arena operations);
        ///            atomic leaves preserve input polarity (I-84 mint-neutral).
        /// @invariant At most ONE entity per distinct instantiated signature is
        ///            committed into @p instruction — the table is a function
        ///            signature → structure (the contract `flattenOrLeaves`'s
        ///            duplicate-identity asserts guard).
        /// @see replaceKeysScratch, extractExpressionUniversalSpan; callers pass
        ///      an explicit `StrSpan(expr)` over a stable `std::string` local.
void prepareIntegrationCore(StrSpan expression,
            WorkInstruction& instruction,
            int level, int& nameCounter,
            StrSpan goal, bool disintegration = false,
            const LogicalEntity* rootEntity = nullptr) {
            const unsigned slot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : scratchArenas().slotCount() - 1;
            ScratchArena& sArena = scratchArenas().forSlot(slot);
            ScratchScope scope(sArena);

            // 1. Extract core and find corresponding LogicalEntity (a typed `=`
            //    goal brings its extensionality entity instead).
            const StrSpan core = extractExpressionUniversalSpan(expression);
            assert((rootEntity == nullptr || !disintegration)
                && "prepareIntegrationCore: a typed root entity is integration-only");
            const LogicalEntity* leEntry = rootEntity != nullptr ? rootEntity : this->definitions.compiledEntity(core);
            instruction.setMarkedGoal(goal);

            // Safety check: if core not found, we cannot proceed with template instantiation
            if (leEntry == nullptr) {
                // ">" is the universal quantifier — a structural operator, never compiled.
                // Negated universals like !(>[...](...)(...)) reach here when OR conjectures
                // are disintegrated. They cannot be integrated; return with empty instruction.
                if (equalSpans(core, StrSpan(">", 1))) return;

                std::cout << "ASSERT: prepareIntegrationCore unknown core='"
                    << std::string(core.ptr, static_cast<size_t>(core.len))
                    << "' expression='"
                    << std::string(expression.ptr, static_cast<size_t>(expression.len))
                    << "'" << std::endl;
                assert(false);
            }

            // Use const reference to ensure we do not modify the original template
            const LogicalEntity& le = *leEntry;

            const bool negative = expression.len >= 2 && expression.ptr[0] == '!'
                && expression.ptr[1] == '(';
            bool counterexample = false;
            const bool negativeOr = disintegration && negative && le.category == "or";
            if (disintegration && negative && le.category == "implication") {
                // The construction boundary is exactly one quantified witness.
                // Count template locals before substituting free parameters.
                StrSpan bound;
                int32_t boundCount = 0;
                for (const std::string& element : le.elements) {
                    StrSpan args[ExecutionParameters::MAX_ARITY];
                    const int32_t n = getArgsSpans(StrSpan(element), args,
                        ExecutionParameters::MAX_ARITY);
                    for (int32_t a = 0; a < n; ++a) {
                        if (!startsWithSpan(args[a], "u_", 2)) {
                            if (boundCount == 0) { bound = args[a]; boundCount = 1; }
                            else if (!equalSpans(bound, args[a])) boundCount = 2;
                        }
                    }
                }
                counterexample = boundCount == 1;
                if (counterexample) assert(le.elements.size() >= 2);
            }

            // Polarity guard: a negated COMPOUND input never instantiates the
            // positive template — the polarity travels with the expression.
            // Integration never unfolds a negative compound. Disintegration
            // may request a one-witness counterexample or the conjunction of
            // negated OR leaves. Other negative compounds keep separate doors.
            if (expression.len >= 2 && expression.ptr[0] == '!'
                && expression.ptr[1] == '('
                && le.category != "atomic" && !counterexample && !negativeOr) {
                return;
            }

            // 2. Create Replacement Map (Signature -> Expression Arguments)
            StrSpan sigArgs[ExecutionParameters::MAX_ARITY];
            const int32_t sigN = getArgsSpans(StrSpan(le.signature), sigArgs,
                                              ExecutionParameters::MAX_ARITY);
            StrSpan exprArgs[ExecutionParameters::MAX_ARITY];
            const int32_t exprN = getArgsSpans(expression, exprArgs,
                                               ExecutionParameters::MAX_ARITY);
            StrReplacement pairs[2 * ExecutionParameters::MAX_ARITY];
            int32_t pairN = 0;

            // Map u_x in signature to actual arguments in expression
            const int32_t argLimit = sigN < exprN ? sigN : exprN;
            for (int32_t i = 0; i < argLimit; ++i) {
                pairs[pairN].key = sigArgs[i];
                pairs[pairN].value = exprArgs[i];
                ++pairN;
            }

            // One entity per DISTINCT instantiated signature. The signature
            // carries only `u_` arguments and replaceKeysScratch substitutes
            // whole argument tokens, so it is computable from the pairs built
            // so far — BEFORE the fresh-`pi_lev_` mint below. On a probe hit
            // this occurrence is already unfolded: return without minting,
            // committing, or recursing (every consumer resolves children by
            // first-match signature scan, so the first unfold IS this node; a
            // second unfold would mint divergent `pi_lev_` placeholders and
            // break the duplicate-identity contract flattenOrLeaves asserts).
            const ScratchString unsignedSignature =
                replaceKeysScratch(sArena, StrSpan(le.signature), pairs, pairN);
            // Registry signatures are positive; an atomic input's sign is part
            // of its instruction identity and subsequent admission demand.
            const bool negatedAtomic = le.category == "atomic"
                && expression.len >= 2 && expression.ptr[0] == '!'
                && expression.ptr[1] == '(';
            const StrSpan newSig = (negatedAtomic || counterexample || negativeOr)
                ? gl::DefinitionExpansion::negateScratch(sArena, StrSpan(unsignedSignature))
                : StrSpan(unsignedSignature);
            for (int32_t e = 0; e < instruction.entityCount(); ++e) {
                if (equalSpans(instruction.signature(e), StrSpan(newSig))) {
                    return;
                }
            }

            // Fresh-bytes builder for "pi_lev_<level>_<n>" on the string arena.
            const auto appendInt = [](char* buf, int32_t& at, int v) {
                char tmp[16]; int32_t t = 0;
                if (v == 0) tmp[t++] = '0';
                else { int vv = v; while (vv > 0) { tmp[t++] = static_cast<char>('0' + vv % 10); vv /= 10; } }
                for (int32_t k = t - 1; k >= 0; --k) buf[at++] = tmp[k];
            };
            const auto keyPresent = [&](const StrSpan& k) -> bool {
                for (int32_t p = 0; p < pairN; ++p)
                    if (equalSpans(pairs[p].key, k)) return true;
                return false;
            };

            // 3. Special Handling for "existence" category
            // We must identify bound variables (args without "u_") in the original elements
            // and map them to fresh variables.
            if (le.category == "existence" || counterexample) {
                for (const std::string& rawElement : le.elements) {
                    StrSpan elemArgs[ExecutionParameters::MAX_ARITY];
                    const int32_t elemArgN = getArgsSpans(StrSpan(rawElement),
                        elemArgs, ExecutionParameters::MAX_ARITY);
                    for (int32_t a = 0; a < elemArgN; ++a) {
                        const StrSpan& arg = elemArgs[a];
                        // Check if argument does NOT start with "u_"
                        const bool isU = (arg.len >= 2 && arg.ptr[0] == 'u' && arg.ptr[1] == '_');
                        if (!isU) {
                            // A bound variable — assign a fresh id iff unseen.
                            if (!keyPresent(arg)) {
                                char buf[64]; int32_t at = 0;
                                const char pref[] = "pi_lev_";
                                for (int32_t k = 0; k < 7; ++k) buf[at++] = pref[k];
                                appendInt(buf, at, level);
                                buf[at++] = '_';
                                appendInt(buf, at, nameCounter);
                                const ScratchString freshVar =
                                    ScratchString::copyFrom(sArena, buf, at);
                                assert(pairN < 2 * ExecutionParameters::MAX_ARITY
                                    && "prepareIntegrationCore: replacement pair count exceeds cap");
                                pairs[pairN].key = arg;
                                pairs[pairN].value = StrSpan(freshVar);
                                ++pairN;
                                nameCounter++;
                            }
                        }
                    }
                }
            }

            //if (le.category != "atomic")
            {
                // 4. Instantiate the template: rewrite the elements by the
                // replacement map and commit the entity (under the signature
                // built for the dedup probe above — byte-identical, the
                // bound-variable pairs never match inside a signature) into
                // the working instruction.
                StrSpan newElems[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
                int32_t newElemN = 0;
                assert(le.elements.size()
                        <= static_cast<size_t>(ExecutionParameters::MAX_INSTRUCTION_ELEMENTS)
                    && "prepareIntegrationCore: element count exceeds MAX_INSTRUCTION_ELEMENTS");
                const int32_t em = instruction.elemMark();
                for (const std::string& rawElement : le.elements) {
                    newElems[newElemN] =
                        StrSpan(replaceKeysScratch(sArena, StrSpan(rawElement), pairs, pairN));
                    if (negativeOr || (counterexample && newElemN + 1 == static_cast<int32_t>(le.elements.size())))
                        newElems[newElemN] = gl::DefinitionExpansion::negateScratch(sArena, StrSpan(newElems[newElemN]));
                    instruction.addElement(StrSpan(newElems[newElemN]));
                    ++newElemN;
                }
                instruction.commitEntity(counterexample ? StrSpan("counterexample", 14)
                    : negativeOr ? StrSpan("and", 3) : StrSpan(le.category), StrSpan(newSig),
                    StrSpan(le.definedSet), le.arity, em);

                // 5. Recursion
                // If category is "existence" or "and" or "or", recurse on the NEW elements
                if (le.category == "existence" || le.category == "and" || le.category == "or" || counterexample) {
                    for (int32_t k = 0; k < newElemN; ++k) {
                        prepareIntegrationCore(StrSpan(newElems[k]), instruction, level, nameCounter,
                            goal, disintegration);
                    }
                }
            }
        }

        /// @brief Instantiate a compiled integration template into
        ///        @p instruction for a logic block — the run-time door of the
        ///        block-free core.
        ///
        /// @details
        /// Forwards to the block-free overload with the block's `level` and
        /// its `startIntPi` counter, so fresh `pi_lev_<level>_<n>` names keep
        /// the block's numbering and the counter advances on the block.
        ///
        /// @param expression   The integration-template expression.
        /// @param instruction  The working instruction to append into.
        /// @param mb           The memory block (reads `level`, advances
        ///                      `startIntPi`).
        /// @param goal         The marked goal to set on @p instruction.
        /// @param disintegration Enable the disintegration-only shapes.
        /// @param rootEntity   The entity to instantiate instead of the
        ///                      registry entry, or `nullptr`.
        /// @return Void.
        /// @see the block-free overload (the body).
void prepareIntegrationCore(StrSpan expression,
            WorkInstruction& instruction,
            Memory& mb,
            StrSpan goal, bool disintegration = false,
            const LogicalEntity* rootEntity = nullptr) {
            prepareIntegrationCore(expression, instruction, mb.level, mb.startIntPi, goal,
                disintegration, rootEntity);
        }

public:
/// @brief Collect the DISTINCT `u_`-stripped argument names across a
        ///        signature + its elements, as zero-copy slices — 0% heap.
        ///
        /// @details Span-native: each expression is sliced by `getArgsSpans`,
        ///          and every `u_`-prefixed argument contributes its
        ///          `u_`-stripped tail. Duplicates are removed in place (the set
        ///          semantics the former `std::set<std::string>` gave), so the
        ///          output is the distinct set — but as `StrSpan` slices into the
        ///          caller's `signature` / `elements` bytes, owning no container.
        ///          Output order is first-seen, NOT sorted; a caller needing the
        ///          sorted set builds a `std::set<std::string>` from the slices
        ///          (which re-sorts). Byte-identical distinct set to the former
        ///          `ce::getArgs` + `substr(2)` collection.
        ///
        /// @param signature The entity signature (borrowed; slices point in).
        /// @param elements  Array of the entity's defining-element spans.
        /// @param elemN     Number of elements in `elements`.
        /// @param out       Caller array receiving the distinct stripped slices.
        /// @param cap       Capacity of `out`; overflow asserts (Rule 19).
        /// @return Number of distinct slices written to `out[0..return)`.
        /// @see prepareIntegrationCore2 (Case B — the sole caller); the still-
        ///      heap `addToHashMemory` sink there builds the `std::set`.
int32_t findAllUArgs(StrSpan signature, const StrSpan* elements,
            int32_t elemN, StrSpan* out, int32_t cap) {
            int32_t count = 0;

            // Slice each expression's args; emit each DISTINCT u_-stripped tail
            // as a slice (dedup in place — the former std::set's set semantics).
            const auto processExpression = [&](StrSpan expr) {
                StrSpan args[ExecutionParameters::MAX_ARITY];
                const int32_t argN = getArgsSpans(expr, args,
                                                  ExecutionParameters::MAX_ARITY);
                for (int32_t a = 0; a < argN; ++a) {
                    const StrSpan& arg = args[a];
                    if (arg.len >= 2 && arg.ptr[0] == 'u' && arg.ptr[1] == '_') {
                        const StrSpan tail(arg.ptr + 2, arg.len - 2);
                        bool seen = false;
                        for (int32_t j = 0; j < count; ++j)
                            if (equalSpans(out[j], tail)) { seen = true; break; }
                        if (!seen) {
                            assert(count < cap && "findAllUArgs: out capacity exceeded");
                            out[count++] = tail;
                        }
                    }
                }
            };

            // 1. Signature, then 2. all elements.
            processExpression(signature);
            for (int32_t e = 0; e < elemN; ++e) processExpression(elements[e]);

            return count;
        }

public:
/// @brief The sorted distinct `u_`-stripped arguments of a rule head and its
        ///        premises — the unchangeables an integration install carries.
        ///
        /// @details
        /// `findAllUArgs` collects the distinct names first-seen; this sorts
        /// them ascending under `compareSpans` (byte-lex, the former
        /// `std::set<std::string>` order), the run every integration install
        /// hands to `addToHashMemory`.
        ///
        /// @param head      The rule head.
        /// @param premises  The rule premises.
        /// @param premiseN  Number of @p premises.
        /// @param out       Caller array receiving the run (slices into the
        ///                  inputs).
        /// @param cap       Capacity of @p out; overflow asserts.
        /// @return Number of names written.
        /// @invariant Pure; allocates nothing.
        /// @see findAllUArgs, CompactRuleProcessor::installIntegrationCompact.
int32_t collectUnchangeables(StrSpan head, const StrSpan* premises,
            int32_t premiseN, StrSpan* out, int32_t cap) {
            StrSpan uArgSpans[ExecutionParameters::MAX_KEY_SLOTS];
            const int32_t uArgN = findAllUArgs(head, premises, premiseN,
                uArgSpans, ExecutionParameters::MAX_KEY_SLOTS);
            int32_t n = 0;
            for (int32_t k = 0; k < uArgN; ++k) {
                const StrSpan v = uArgSpans[k];
                int32_t lo = 0, hi = n;
                bool dup = false;
                while (lo < hi) {
                    const int32_t mid = (lo + hi) / 2;
                    const int c = compareSpans(out[mid], v);
                    if (c == 0) { dup = true; break; }
                    if (c < 0) lo = mid + 1; else hi = mid;
                }
                if (dup) continue;
                assert(n < cap && "collectUnchangeables: unchangeable run exceeds cap");
                for (int32_t x = n; x > lo; --x) out[x] = out[x - 1];
                out[lo] = v;
                ++n;
            }
            return n;
        }

public:
/// @brief Fill a compiled integration compact's owner tokens with an
        ///        instance's arguments (D-503).
        ///
        /// @details
        /// An integration set's compacts name only owner tokens `u_1..u_k`
        /// (`IntegrationRuleCompiler::compactOf` asserts it): token `u_i`
        /// stands for the owner's i-th argument, `u_<arity+1>` for an occupied
        /// existence's witness. One token-boundary pass replaces every token
        /// by @p owners[i-1], so an owner argument that itself reads `u_<j>`
        /// is never replaced twice. Every token is asserted to have an owner.
        ///
        /// @param arena   Scratch arena receiving the filled compact.
        /// @param compact The compiled compact `(implication<N>[u_…])`.
        /// @param owners  The owner's arguments (`u_`-prefixed run-time names).
        /// @param ownerN  Their number.
        /// @return The filled compact.
        /// @invariant 0% heap: the token names and the result ride @p arena.
        /// @see expandIntegrationCompactScratch.
ScratchString fillIntegrationCompactScratch(ScratchArena& arena, StrSpan compact,
            const StrSpan* owners, int32_t ownerN) const {
            assert(ownerN > 0 && ownerN <= ExecutionParameters::MAX_ARITY
                && "fillIntegrationCompactScratch: owner count out of range");
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t argN = getArgsSpans(compact, args, ExecutionParameters::MAX_ARITY);
            StrReplacement pairs[ExecutionParameters::MAX_ARITY];
            for (int32_t i = 0; i < ownerN; ++i) {
                char name[16];
                const int len = std::snprintf(name, sizeof(name), "u_%d", i + 1);
                assert(len > 0 && len < static_cast<int>(sizeof(name)));
                pairs[i] = StrReplacement{ StrSpan(ScratchString::copyFrom(arena, name, len)), owners[i] };
            }
            for (int32_t a = 0; a < argN; ++a) {
                bool owned = false;
                for (int32_t i = 0; i < ownerN && !owned; ++i) owned = equalSpans(args[a], pairs[i].key);
                assert(owned && "fillIntegrationCompactScratch: a compact token without an owner argument");
                (void)owned;
            }
            return replaceKeysScratch(arena, compact, pairs, ownerN);
        }

public:
/// @brief Rename the chain using the block's existing fresh-variable counters.
/// @details Preserves binder treatment, expression spelling, output order and allocation ownership.
/// @param chain Ordered expression chain.
/// @param mb Owning block supplying fresh-variable state.
/// @return The constructed result in the existing representation.
inline std::vector<std::string> renamingChain2(const std::vector<std::string>& chain,
            Memory& mb) {
            std::map<std::string, std::string> replacementMap;

            for (const std::string& expr : chain) {
                std::vector<std::string> args = ce::getArgs(expr);
                for (const std::string& arg : args) {
                    // Check for "u_" prefix
                    if (gl::startsWith(arg, "u_", 2)) {
                        // Remove "u_"
                        replacementMap[arg] = arg.substr(2);
                    }
                    else {
                        // If not already mapped, assign new "repl_..." name using mb.startInt
                        if (replacementMap.find(arg) == replacementMap.end()) {
                            // Pattern: repl_lev_<mb.level>_<mb.startInt>
                            std::string newName = "repl_lev_" + std::to_string(mb.level) + "_" + std::to_string(mb.startIntRepl);

                            // Update startInt immediately after use
                            mb.startIntRepl++;

                            replacementMap[arg] = newName;
                        }
                    }
                }
            }

            std::vector<std::string> renamedChain;
            renamedChain.reserve(chain.size());
            for (const std::string& expr : chain) {
                renamedChain.push_back(ce::replaceKeysInString(expr, replacementMap));
            }

            return renamedChain;
        }

public:
/// @brief Scratch twin of @ref renamingChain2 — rename a chain's arguments
        ///        into caller-owned spans over arena-built renamed elements, with
        ///        no heap `std::vector<std::string>` / `std::map`.
        ///
        /// @details
        /// Byte-for-byte the heap oracle: a first-seen walk over every chain
        /// element's arguments builds a `StrReplacement` run keyed by the distinct
        /// argument (the `std::map<std::string,std::string>` twin, deduped by a
        /// linear span-scan over the pairs built so far). A `u_`-prefixed argument
        /// maps to its `u_`-stripped INPUT SLICE (no fresh bytes); every other
        /// distinct argument mints a FRESH `repl_lev_<level>_<startIntRepl>` name
        /// on @p arena and bumps `startIntRepl` — the assignment ORDER (first-seen
        /// across the chain) is FROZEN, so the `startIntRepl` progression is
        /// byte-identical. Each element is then rewritten via `replaceKeysScratch`
        /// (the byte-exact greedy-longest twin of `ce::replaceKeysInString`) into
        /// `out`. Mints into no interner.
        ///
        /// The fresh `repl_lev_*` bytes and the renamed element bytes ride
        /// @p arena; the returned `out` spans stay live until the caller's
        /// `ScratchScope` rewinds it (Pitfall 4 — held as spans, never across a
        /// rewind).
        ///
        /// @param arena  Scratch (string-tier) arena receiving the fresh names +
        ///               the `replaceKeysScratch` results; the caller owns the scope.
        /// @param chain  The chain elements as spans over stable input.
        /// @param chainN Element count of @p chain.
        /// @param mb     The memory block (reads `level`, mutates `startIntRepl`).
        /// @param out    Caller buffer receiving the renamed spans; filled positionally.
        /// @param outCap Capacity of @p out.
        /// @return The renamed element count (always `== chainN`).
        /// @invariant Byte-identical to `renamingChain2` element-for-element AND in
        ///            the `startIntRepl` progression (I-84 mint-neutral).
        /// @see renamingChain2 (the retained byte-oracle), replaceKeysScratch.
int32_t renamingChain2Scratch(ScratchArena& arena,
            const StrSpan* chain, int32_t chainN, Memory& mb,
            StrSpan* out, int32_t outCap) {
            assert(chainN <= outCap
                && "renamingChain2Scratch: chain element count exceeds caller capacity");

            constexpr int32_t kPairsCap = ExecutionParameters::MAX_ADMISSION_REM_ARGS;
            StrReplacement pairs[kPairsCap]; int32_t pairN = 0;

            const auto appendInt = [](char* buf, int32_t& at, int v) {
                char tmp[16]; int32_t t = 0;
                if (v == 0) tmp[t++] = '0';
                else { int vv = v; while (vv > 0) { tmp[t++] = static_cast<char>('0' + vv % 10); vv /= 10; } }
                for (int32_t k = t - 1; k >= 0; --k) buf[at++] = tmp[k];
            };
            const auto keyPresent = [&](const StrSpan& k) -> bool {
                for (int32_t p = 0; p < pairN; ++p)
                    if (equalSpans(pairs[p].key, k)) return true;
                return false;
            };

            for (int32_t i = 0; i < chainN; ++i) {
                StrSpan args[ExecutionParameters::MAX_ARITY];
                const int32_t argN = getArgsSpans(chain[i], args,
                                                  ExecutionParameters::MAX_ARITY);
                for (int32_t a = 0; a < argN; ++a) {
                    const StrSpan& arg = args[a];
                    if (keyPresent(arg)) continue;
                    assert(pairN < kPairsCap
                        && "renamingChain2Scratch: replacement pair count exceeds cap");
                    if (arg.len >= 2 && arg.ptr[0] == 'u' && arg.ptr[1] == '_') {
                        pairs[pairN].key = arg;
                        pairs[pairN].value = StrSpan(arg.ptr + 2, arg.len - 2);
                        ++pairN;
                    } else {
                        char buf[64]; int32_t at = 0;
                        const char pref[] = "repl_lev_";
                        for (int32_t k = 0; k < 9; ++k) buf[at++] = pref[k];
                        appendInt(buf, at, mb.level);
                        buf[at++] = '_';
                        appendInt(buf, at, mb.startIntRepl);
                        const ScratchString name = ScratchString::copyFrom(arena, buf, at);
                        pairs[pairN].key = arg;
                        pairs[pairN].value = StrSpan(name);
                        ++pairN;
                        mb.startIntRepl++;
                    }
                }
            }

            for (int32_t i = 0; i < chainN; ++i)
                out[i] = StrSpan(replaceKeysScratch(arena, chain[i], pairs, pairN));

            return chainN;
        }

public:
/// @brief Expand a category signature using the existing binder rules.
/// @details Preserves binder treatment, expression spelling, output order and allocation ownership.
/// @param category Compiled logical category.
/// @param renamedChain Ordered renamed expression chain.
/// @param cleanSignature Signature after binder preparation.
/// @param hasPiBoundVars Receives whether the bound variables are unoccupied when provided.
/// @return The constructed result in the existing representation.
std::string expandSignatureForIntegration(const std::string& category, const std::vector<std::string>& renamedChain, const std::string& cleanSignature, bool* hasPiBoundVars = nullptr) {
            // 1. Extract arguments from cleanSignature and assert no 'u_' prefix
            std::vector<std::string> sigArgsVec = ce::getArgs(cleanSignature);
            std::set<std::string> sigArgs;
            std::map<std::string, std::string> toUReplacementMap;

            for (const std::string& arg : sigArgsVec) {
                assert((arg.size() < 2 || arg.substr(0, 2) != "u_") && "Signature argument already contains 'u_' prefix!");
                sigArgs.insert("u_" + arg);
                toUReplacementMap[arg] = "u_" + arg;
            }

            // 2. Apply "u_" prefix replacements to the entire renamedChain and the signature
            std::vector<std::string> modifiedChain;
            modifiedChain.reserve(renamedChain.size());
            for (const std::string& elem : renamedChain) {
                modifiedChain.push_back(ce::replaceKeysInString(elem, toUReplacementMap));
            }
            std::string modifiedSignature = ce::replaceKeysInString(cleanSignature, toUReplacementMap);

            std::string result;

            // 3. Build Result String based on Category Logic
            if (category == "and") {
                if (modifiedChain.empty()) result = modifiedSignature;
                else if (modifiedChain.size() == 1) result = modifiedChain[0];
                else {
                    std::string current = modifiedChain[0];
                    for (size_t i = 1; i < modifiedChain.size(); ++i) {
                        current = "(&" + current + modifiedChain[i] + ")";
                    }
                    result = current;
                }
            }
            else if (category == "existence") {
                if (modifiedChain.size() < 2) result = modifiedSignature;
                else {
                    std::string body = modifiedChain[0];
                    std::string head = modifiedChain[1];

                    std::string negatedHead;
                    if (!head.empty() && head[0] == '!') {
                        negatedHead = head.substr(1);
                    }
                    else {
                        negatedHead = "!" + head;
                    }

                    std::vector<std::string> bodyArgs = ce::getArgs(body);
                    std::set<std::string> boundVars;
                    for (const auto& arg : bodyArgs) {
                        if (sigArgs.find(arg) == sigArgs.end()) {
                            boundVars.insert(arg);
                        }
                    }

                    assert(!boundVars.empty() && "Existence must have at least one bound variable after expansion for integration");

                    // Check if bound vars are pi_lev_-prefixed (unoccupied) or not (occupied)
                    // Format: pi_lev_<level>_<counter>, e.g. pi_lev_0_1
                    bool allPi = true;
                    for (const auto& bv : boundVars) {
                        if (bv.size() < 7 || bv[0] != 'p' || bv[1] != 'i' || bv[2] != '_'
                            || bv[3] != 'l' || bv[4] != 'e' || bv[5] != 'v' || bv[6] != '_') {
                            allPi = false;
                            break;
                        }
                    }
                    if (hasPiBoundVars) {
                        *hasPiBoundVars = allPi;
                    }

                    if (allPi) {
                        // Unoccupied bound vars: preserve binding — !(>[bound](body)(head))
                        std::string varsStr;
                        for (const auto& v : boundVars) {
                            if (!varsStr.empty()) varsStr += ",";
                            varsStr += v;
                        }
                        result = "!(>[" + varsStr + "]" + body + negatedHead + ")";
                    } else {
                        // Occupied bound vars: empty binding — (>[](body)(head))
                        result = "(>[]" + body + head + ")";
                    }
                }
            }
            else if (category == "implication") {
                if (modifiedChain.empty()) {
                    result = modifiedSignature;
                }
                else {
                    std::vector<std::string> chain = modifiedChain;
                    std::string head = chain.back();
                    chain.pop_back();
                    result = gl::ImplicationBuilder::reconstructImplicationFullBind(chain, head);
                }
            }
            else {
                assert(false && "Forbidden category encountered in expandSignatureForIntegration.");
                return cleanSignature;
            }

            // 4. AST Token Parsing to collect all variables in the reconstructed result
            std::set<std::string> tokens;
            ce::TreeNode1* root = ce::parseExpr(result);

            std::vector<ce::TreeNode1*> stack;
            if (root) {
                stack.push_back(root);
            }

            while (!stack.empty()) {
                ce::TreeNode1* curr = stack.back();
                stack.pop_back();

                std::vector<std::string> args = ce::getArgs(curr->value);
                for (const std::string& t : args) {
                    tokens.insert(t);
                }

                if (curr->left) stack.push_back(curr->left);
                if (curr->right) stack.push_back(curr->right);
            }

            ce::deleteTree(root);

            // 5. Final Renaming: Strip "u_" prefix everywhere
            std::map<std::string, std::string> renameMap;

            for (const std::string& t : tokens) {
                if (t.size() >= 2 && t[0] == 'u' && t[1] == '_') {
                    renameMap[t] = t.substr(2);
                }
            }

            // 6. Apply Renaming to return the final clean signature
            return ce::replaceKeysInString(result, renameMap);
        }

public:
/// @brief Build the implication and its paired history expression.
/// @details Preserves binder treatment, expression spelling, output order and allocation ownership.
/// @param elements Ordered integration premise expressions.
/// @param signature Consequent signature to instantiate.
/// @return The constructed result in the existing representation.
std::pair<std::string, std::string> buildIntegrationInstruction(const std::vector<std::string>& elements, const std::string& signature) {
            // 1. Build the implication: only pi_-prefixed vars get bound (unoccupied existence bound vars)
            std::string implication = this->reconstructImplicationForIntegration(elements, signature);

            // 2. History version: strip u_, and strip non-pi_lev_ bound vars from outermost >[...]
            //    pi_lev_ stays in >[...], occupied (non-pi_lev_) vars get stripped to >[]
            std::string history = gl::DefinitionExpansion::stripUPrefixAST(implication);
            if (history.size() > 3 && history[0] == '(' && history[1] == '>' && history[2] == '[') {
                std::size_t bracketEnd = history.find(']', 3);
                if (bracketEnd != std::string::npos && bracketEnd > 3) {
                    std::string bv = history.substr(3, bracketEnd - 3);
                    bool isPiLev = (bv.size() >= 7 && bv[0] == 'p' && bv[1] == 'i' && bv[2] == '_'
                        && bv[3] == 'l' && bv[4] == 'e' && bv[5] == 'v' && bv[6] == '_');
                    if (!isPiLev) {
                        history = "(>[]" + history.substr(bracketEnd + 1);
                    }
                }
            }
            // 3. Return pair: .first = history (pi_lev_ kept, others stripped), .second = u_-preserved (for hash matching)
            return std::make_pair(history, implication);
        }

private:
/// @brief Scratch twin of @ref reconstructImplicationForIntegration —
        ///        build the fully-bound implication then apply the outermost
        ///        `(>[...]` pi_lev / occupied rewrite, on @p arena.
        ///
        /// @details
        /// Reuses the EXISTING T5 foundation `reconstructImplicationFullBindScratch`
        /// for the nested build, then reproduces the oracle's outermost-bracket
        /// check byte-for-byte: on a `(>[` head, the first `]` from index 3 slices
        /// the bound-var list; a `pi_lev_` prefix (the 7-char test) returns the
        /// build unchanged (unoccupied), otherwise the occupied form `"(>[]" +
        /// tail` is built on @p arena.
        ///
        /// @param arena Scratch arena receiving the result.
        /// @param key   Premise spans.
        /// @param keyN  Number of premises.
        /// @param value Consequent span.
        /// @return The reconstructed implication as a `ScratchString`.
        /// @invariant Byte-identical to `reconstructImplicationForIntegration`;
        ///            mints into no interner.
        /// @see reconstructImplicationForIntegration (the retained heap oracle),
        ///      reconstructImplicationFullBindScratch (the reused T5 foundation).
ScratchString reconstructImplicationForIntegrationScratch(ScratchArena& arena,
            const StrSpan* key, int32_t keyN, StrSpan value) {
            const ScratchString full =
                gl::ImplicationBuilder::reconstructImplicationFullBindScratch(arena, key, keyN, value);
            const StrSpan fs(full);
            if (fs.len > 3 && fs.ptr[0] == '(' && fs.ptr[1] == '>' && fs.ptr[2] == '[') {
                int32_t be = -1;
                for (int32_t p = 3; p < fs.len; ++p)
                    if (fs.ptr[p] == ']') { be = p; break; }
                if (be > 3) {
                    const StrSpan bv(fs.ptr + 3, be - 3);
                    const bool isPiLev = bv.len >= 7 && bv.ptr[0] == 'p' && bv.ptr[1] == 'i'
                        && bv.ptr[2] == '_' && bv.ptr[3] == 'l' && bv.ptr[4] == 'e'
                        && bv.ptr[5] == 'v' && bv.ptr[6] == '_';
                    if (!isPiLev) {
                        const int32_t tailLen = fs.len - (be + 1);
                        char* b = arena.allocBytes(4 + tailLen);
                        b[0] = '('; b[1] = '>'; b[2] = '['; b[3] = ']';
                        if (tailLen > 0)
                            std::memcpy(b + 4, fs.ptr + be + 1, static_cast<size_t>(tailLen));
                        return ScratchString::wrap(arena, b, 4 + tailLen);
                    }
                }
            }
            return full;
        }

public:
/// @brief Scratch twin of @ref expandSignatureForIntegration — expand a
        ///        category-specific signature template into its clean, `u_`-stripped
        ///        form on @p arena, with no heap `std::string` / `std::map` / `std::set`.
        ///
        /// @details
        /// Step-for-step the oracle: (1) the signature args build a `u_`-prefix
        /// `StrReplacement` run (keeping the per-arg no-`u_` assert) + a sorted
        /// `u_`-form membership run (`std::set` order for the boundVars test);
        /// (2) the chain + signature are `u_`-prefixed via `replaceKeysScratch`;
        /// (3) the category logic builds `result` on @p arena — the `and`
        /// left-nested `(&…)` fold, the `existence` `!(>[bound]…)` / `(>[]…)`
        /// form (its bound vars = body args not in the sorted `u_`-sig run,
        /// collected sorted-unique = the oracle's `std::set` order for the
        /// comma-join; `negateScratch` supplies the negated head; the `allPi`
        /// 7-char scan and `hasPiBoundVars` out-param preserved; the
        /// `!boundVars.empty()` assert kept), or the `implication`
        /// `reconstructImplicationFullBindScratch` (the reused foundation, chain
        /// split at the last element); the forbidden-category `assert(false)`
        /// kept; (4-6) the final token walk + `u_`-strip IS `stripUPrefixASTScratch`
        /// verbatim.
        ///
        /// @param arena          Scratch arena receiving all fresh bytes + the result.
        /// @param category       `"and"` / `"existence"` / `"implication"` span.
        /// @param renamedChain   The renamed chain element spans.
        /// @param chainN         Number of chain elements.
        /// @param cleanSignature The clean (no-`u_`) signature span.
        /// @param hasPiBoundVars Optional out-param: set to the existence-branch
        ///                       `allPi` verdict (unchanged from the oracle).
        /// @return The expanded, `u_`-stripped signature as a `ScratchString`.
        /// @invariant Byte-identical to `expandSignatureForIntegration`; mints into
        ///            no interner.
        /// @see expandSignatureForIntegration (the retained heap oracle),
        ///      negateScratch, stripUPrefixASTScratch,
        ///      reconstructImplicationFullBindScratch, replaceKeysScratch.
ScratchString expandSignatureForIntegrationScratch(ScratchArena& arena,
            StrSpan category, const StrSpan* renamedChain, int32_t chainN,
            StrSpan cleanSignature, bool* hasPiBoundVars = nullptr) {
            // 1. Signature args -> u_-prefix pairs + sorted u_-form membership run.
            StrSpan sigArgsVec[ExecutionParameters::MAX_ARITY];
            const int32_t sigVecN = getArgsSpans(cleanSignature, sigArgsVec,
                                                 ExecutionParameters::MAX_ARITY);
            StrReplacement toUPairs[ExecutionParameters::MAX_ARITY];
            int32_t toUN = 0;
            StrSpan uSigArgs[ExecutionParameters::MAX_ARITY];
            int32_t uSigN = 0;
            const auto insSorted = [](StrSpan* arr, int32_t& cnt, const StrSpan& v) {
                int32_t lo = 0, hi = cnt;
                while (lo < hi) {
                    const int32_t mid = (lo + hi) / 2;
                    const int c = compareSpans(arr[mid], v);
                    if (c == 0) return;
                    if (c < 0) lo = mid + 1; else hi = mid;
                }
                for (int32_t k = cnt; k > lo; --k) arr[k] = arr[k - 1];
                arr[lo] = v; ++cnt;
            };
            for (int32_t i = 0; i < sigVecN; ++i) {
                const StrSpan& arg = sigArgsVec[i];
                assert((arg.len < 2 || !(arg.ptr[0] == 'u' && arg.ptr[1] == '_'))
                    && "Signature argument already contains 'u_' prefix!");
                char* u = arena.allocBytes(2 + arg.len);
                u[0] = 'u'; u[1] = '_';
                if (arg.len > 0) std::memcpy(u + 2, arg.ptr, static_cast<size_t>(arg.len));
                const StrSpan uSpan(u, 2 + arg.len);
                toUPairs[toUN].key = arg;
                toUPairs[toUN].value = uSpan;
                ++toUN;
                insSorted(uSigArgs, uSigN, uSpan);
            }

            // 2. u_-prefix the chain + signature.
            ScratchString modifiedChain[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
            assert(chainN <= ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
                && "expandSignatureForIntegrationScratch: chain exceeds MAX_INSTRUCTION_ELEMENTS");
            for (int32_t i = 0; i < chainN; ++i)
                modifiedChain[i] = replaceKeysScratch(arena, renamedChain[i], toUPairs, toUN);
            const ScratchString modifiedSignature =
                replaceKeysScratch(arena, cleanSignature, toUPairs, toUN);

            // Arena concat helper.
            const auto concatN = [&](std::initializer_list<StrSpan> parts) -> ScratchString {
                int32_t total = 0;
                for (const StrSpan& p : parts) total += p.len;
                if (total == 0) return ScratchString();
                char* b = arena.allocBytes(total);
                int32_t at = 0;
                for (const StrSpan& p : parts) {
                    if (p.len > 0) { std::memcpy(b + at, p.ptr, static_cast<size_t>(p.len)); at += p.len; }
                }
                return ScratchString::wrap(arena, b, total);
            };

            // 3. Category logic -> result.
            ScratchString result;
            if (equalSpans(category, StrSpan("and", 3))) {
                if (chainN == 0) result = modifiedSignature;
                else if (chainN == 1) result = modifiedChain[0];
                else {
                    ScratchString current = modifiedChain[0];
                    for (int32_t i = 1; i < chainN; ++i)
                        current = concatN({ StrSpan("(&", 2), StrSpan(current),
                                            StrSpan(modifiedChain[i]), StrSpan(")", 1) });
                    result = current;
                }
            } else if (equalSpans(category, StrSpan("existence", 9))) {
                if (chainN < 2) result = modifiedSignature;
                else {
                    const StrSpan body(modifiedChain[0]);
                    const StrSpan head(modifiedChain[1]);
                    const StrSpan negatedHead = gl::DefinitionExpansion::negateScratch(arena, head);

                    StrSpan bodyArgs[ExecutionParameters::MAX_ARITY];
                    const int32_t bodyArgN = getArgsSpans(body, bodyArgs,
                                                          ExecutionParameters::MAX_ARITY);
                    StrSpan boundVars[ExecutionParameters::MAX_ARITY];
                    int32_t bvN = 0;
                    for (int32_t i = 0; i < bodyArgN; ++i) {
                        // membership in the sorted u_-sig run (binary_search).
                        const StrSpan& arg = bodyArgs[i];
                        int32_t lo = 0, hi = uSigN; bool found = false;
                        while (lo < hi) {
                            const int32_t mid = (lo + hi) / 2;
                            const int c = compareSpans(uSigArgs[mid], arg);
                            if (c == 0) { found = true; break; }
                            if (c < 0) lo = mid + 1; else hi = mid;
                        }
                        if (!found) insSorted(boundVars, bvN, arg);
                    }
                    assert(bvN > 0
                        && "Existence must have at least one bound variable after expansion for integration");

                    bool allPi = true;
                    for (int32_t i = 0; i < bvN; ++i) {
                        const StrSpan& bv = boundVars[i];
                        if (!(bv.len >= 7 && bv.ptr[0] == 'p' && bv.ptr[1] == 'i' && bv.ptr[2] == '_'
                            && bv.ptr[3] == 'l' && bv.ptr[4] == 'e' && bv.ptr[5] == 'v' && bv.ptr[6] == '_')) {
                            allPi = false;
                            break;
                        }
                    }
                    if (hasPiBoundVars) *hasPiBoundVars = allPi;

                    if (allPi) {
                        // varsStr = comma-joined boundVars (sorted order).
                        int32_t vlen = 0;
                        for (int32_t i = 0; i < bvN; ++i) { if (i > 0) ++vlen; vlen += boundVars[i].len; }
                        char* vb = arena.allocBytes(vlen);
                        int32_t at = 0;
                        for (int32_t i = 0; i < bvN; ++i) {
                            if (i > 0) vb[at++] = ',';
                            std::memcpy(vb + at, boundVars[i].ptr, static_cast<size_t>(boundVars[i].len));
                            at += boundVars[i].len;
                        }
                        const StrSpan varsStr(vb, vlen);
                        result = concatN({ StrSpan("!(>[", 4), varsStr, StrSpan("]", 1),
                                           body, negatedHead, StrSpan(")", 1) });
                    } else {
                        result = concatN({ StrSpan("(>[]", 4), body, head, StrSpan(")", 1) });
                    }
                }
            } else if (equalSpans(category, StrSpan("implication", 11))) {
                if (chainN == 0) result = modifiedSignature;
                else {
                    StrSpan chainSpans[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
                    for (int32_t i = 0; i < chainN - 1; ++i) chainSpans[i] = StrSpan(modifiedChain[i]);
                    result = gl::ImplicationBuilder::reconstructImplicationFullBindScratch(arena, chainSpans,
                        chainN - 1, StrSpan(modifiedChain[chainN - 1]));
                }
            } else {
                assert(false && "Forbidden category encountered in expandSignatureForIntegration.");
                return ScratchString::copyFrom(arena, cleanSignature.ptr, cleanSignature.len);
            }

            // 4-6. Token walk + final u_-strip == stripUPrefixASTScratch verbatim.
            return gl::DefinitionExpansion::stripUPrefixASTScratch(arena, StrSpan(result));
        }

public:
/// @brief Scratch twin of @ref buildIntegrationInstruction — build the
        ///        integration instruction's history + hash forms on @p arena, with
        ///        no heap `std::string` / `std::pair`.
        ///
        /// @details
        /// Byte-for-byte the oracle: the HASH form is
        /// `reconstructImplicationForIntegrationScratch` (@p outHash); the HISTORY
        /// form is `gl::DefinitionExpansion::stripUPrefixASTScratch(hash)` then, on a `(>[` head whose bound
        /// var is NOT `pi_lev_`, the outermost `(>[...]` collapses to `(>[]`
        /// (@p outHistory). Both fields byte-identical to the heap pair.
        ///
        /// @param arena      Scratch arena receiving both results.
        /// @param elements   The premise spans.
        /// @param elemN      Number of premises.
        /// @param signature  The consequent span.
        /// @param outHistory Filled with the history form (pi_lev_ kept, else stripped).
        /// @param outHash    Filled with the hash form (u_-preserved implication).
        /// @invariant Both outputs byte-identical to `buildIntegrationInstruction`;
        ///            mints into no interner.
        /// @see buildIntegrationInstruction (the retained heap oracle),
        ///      reconstructImplicationForIntegrationScratch, stripUPrefixASTScratch.
/// @return The constructed result through the documented output.
void buildIntegrationInstructionScratch(ScratchArena& arena,
            const StrSpan* elements, int32_t elemN, StrSpan signature,
            ScratchString& outHistory, ScratchString& outHash) {
            const ScratchString implication =
                reconstructImplicationForIntegrationScratch(arena, elements, elemN, signature);
            outHash = implication;
            ScratchString history = gl::DefinitionExpansion::stripUPrefixASTScratch(arena, StrSpan(implication));
            const StrSpan hs(history);
            if (hs.len > 3 && hs.ptr[0] == '(' && hs.ptr[1] == '>' && hs.ptr[2] == '[') {
                int32_t be = -1;
                for (int32_t p = 3; p < hs.len; ++p)
                    if (hs.ptr[p] == ']') { be = p; break; }
                if (be > 3) {
                    const StrSpan bv(hs.ptr + 3, be - 3);
                    const bool isPiLev = bv.len >= 7 && bv.ptr[0] == 'p' && bv.ptr[1] == 'i'
                        && bv.ptr[2] == '_' && bv.ptr[3] == 'l' && bv.ptr[4] == 'e'
                        && bv.ptr[5] == 'v' && bv.ptr[6] == '_';
                    if (!isPiLev) {
                        const int32_t tailLen = hs.len - (be + 1);
                        char* b = arena.allocBytes(4 + tailLen);
                        b[0] = '('; b[1] = '>'; b[2] = '['; b[3] = ']';
                        if (tailLen > 0)
                            std::memcpy(b + 4, hs.ptr + be + 1, static_cast<size_t>(tailLen));
                        history = ScratchString::wrap(arena, b, 4 + tailLen);
                    }
                }
            }
            outHistory = history;
        }

/// @brief Expand one filled integration compact into the premises and the
        ///        head of its rule, in `u_`-prefixed instruction form.
        ///
        /// @details
        /// The compact names an implication entity of the registry — an entry
        /// of some entity's integration set
        /// (D-503) whose owner tokens were filled
        /// with the instance's `u_`-prefixed arguments. The entity's elements
        /// are substituted in one token-boundary pass: every signature formal
        /// by the compact's argument at the same position, every bound
        /// placeholder (an element argument that is not a formal) by a fresh
        /// `pi_lev_<level>_<n>` name taken from the block's name counter, the
        /// names `prepareIntegrationCore` gives a definition's witnesses. A
        /// placeholder never meets an instance argument, so repeated and
        /// canonical arguments expand as they stand. The last element is the
        /// head; the others are the premises in stored order.
        ///
        /// @param arena      Scratch arena receiving the expanded elements.
        /// @param compact    The filled compact `(implication<N>[u_…])`.
        /// @param mb         The owning block: its level and name counter
        ///                   (advanced by one per placeholder).
        /// @param premises   Out: the premises (spans onto @p arena).
        /// @param premiseCap Capacity of @p premises; overflow asserts.
        /// @param head       Out: the head (span onto @p arena).
        /// @return The number of premises.
        /// @invariant 0% heap: stack pairs, fresh names and results on @p arena;
        ///            the registry is read only.
        /// @see IntegrationRuleCompiler::expandIntegrationCompact (the
        ///      compile-side twin), buildIntegrationInstructionScratch.
int32_t expandIntegrationCompactScratch(ScratchArena& arena, StrSpan compact, Memory& mb,
            StrSpan* premises, int32_t premiseCap, StrSpan& head) const {
            const LogicalEntity* le = this->definitions.compiledEntity(extractExpressionUniversalSpan(compact));
            assert(le != nullptr && le->category == "implication" && le->elements.size() >= 2
                && "expandIntegrationCompactScratch: a compact must name an implication entity");
            StrSpan formals[ExecutionParameters::MAX_ARITY];
            const int32_t formalN = getArgsSpans(StrSpan(le->signature), formals, ExecutionParameters::MAX_ARITY);
            StrSpan actuals[ExecutionParameters::MAX_ARITY];
            const int32_t actualN = getArgsSpans(compact, actuals, ExecutionParameters::MAX_ARITY);
            assert(formalN == actualN && "expandIntegrationCompactScratch: compact arity differs from its entity");
            StrReplacement pairs[ExecutionParameters::MAX_ARITY + ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
            int32_t pairN = 0;
            for (int32_t i = 0; i < formalN; ++i) pairs[pairN++] = StrReplacement{ formals[i], actuals[i] };
            for (const std::string& element : le->elements) {
                StrSpan args[ExecutionParameters::MAX_ARITY];
                const int32_t argN = getArgsSpans(StrSpan(element), args, ExecutionParameters::MAX_ARITY);
                for (int32_t a = 0; a < argN; ++a) {
                    bool known = false;
                    for (int32_t p = 0; p < pairN && !known; ++p) known = equalSpans(pairs[p].key, args[a]);
                    if (known) continue;
                    assert(pairN < ExecutionParameters::MAX_ARITY + ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
                        && "expandIntegrationCompactScratch: bound placeholders exceed the cap");
                    char name[64];
                    const int len = std::snprintf(name, sizeof(name), "pi_lev_%d_%d", mb.level, mb.startIntPi);
                    assert(len > 0 && len < static_cast<int>(sizeof(name)));
                    ++mb.startIntPi;
                    const ScratchString fresh = ScratchString::copyFrom(arena, name, len);
                    pairs[pairN++] = StrReplacement{ args[a], StrSpan(fresh) };
                }
            }
            const int32_t elementN = static_cast<int32_t>(le->elements.size());
            assert(elementN - 1 <= premiseCap && "expandIntegrationCompactScratch: premise count exceeds the cap");
            for (int32_t e = 0; e + 1 < elementN; ++e)
                premises[e] = StrSpan(replaceKeysScratch(arena, StrSpan(le->elements[static_cast<std::size_t>(e)]), pairs, pairN));
            head = StrSpan(replaceKeysScratch(arena, StrSpan(le->elements.back()), pairs, pairN));
            return elementN - 1;
        }
private:
const CompiledDefinitionRegistry& definitions;
};
}

namespace gl {
/// @brief Bind definitions used to prepare integration instructions.
/// @details Construction binds the reader only; instructions and fresh names use caller storage.
/// @param definitions Read-only compiled definitions and configured signatures.
/// @return Constructed integration instruction builder.
inline IntegrationInstructionBuilder::IntegrationInstructionBuilder(const CompiledDefinitionRegistry& definitions) : definitions(definitions) {}
} // namespace gl
