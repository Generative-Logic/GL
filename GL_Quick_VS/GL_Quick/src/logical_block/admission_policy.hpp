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
#include "../compiler/compiled_definition_registry.hpp"
#include "../parameters.hpp"
#include "../memory_infra/temporary_witness_store.hpp"
namespace gl {
/// @brief Classify the existing classic, local, OR and relation admission routes.
/// @details Decisions read definitions and policy parameters without mutating memory.
class AdmissionPolicy {
public:
/// @brief Count the `it_<i>_lev_<l>_<n>` secondary-variable lexemes in
        ///        an expression, skipping known recursion products.
        ///
        /// @details
        /// Scans @p inputString for the `it_\d+_lev_\d+_\d+` iteration-variable
        /// lexeme via the regex-free `str_ops::scanItLevOccurrences` lexical
        /// scanner (the proven left-to-right iterator twin of the former
        /// unanchored `it_\d+_lev_\d+_\d+` regex iterator) and counts every match
        /// that is NOT a known product of a recursion block. The count feeds the
        /// `maxNumberSecondaryVariables` admission caps at the statement,
        /// revival, and admission gates.
        ///
        /// Each match's bytes are looked up through the non-minting span
        /// `NameMap::lookup` (no per-match `std::string` copy) and tested against
        /// `HashMemory::productsOfRecursionIds` (read discipline of
        /// [I-84](../../docs/agentic_swdd/30_invariants.md#i-84)). Every
        /// recursion product is interned by its insert site
        /// (`updateAdmissionMapRecursion` / `isAdmitted`), so a lookup miss
        /// (id 0) is definitive: the lexeme cannot be a member.
        ///
        /// The former `const std::string&` overload is retired; its verbatim
        /// regex body is reimplemented in-test as the differential oracle
        /// (`test_memory.cpp::count_pattern_occurrences_matches_regex_oracle`).
        ///
        /// @param inputString Expression view to scan.
        /// @param localMemory Hash memory owning `productsOfRecursionIds`.
        /// @param nameMap     The owning LB's interning registry; read-only,
        ///                    never minted from here.
        /// @return Number of pattern matches that are not recursion products.
        /// @see `str_ops::scanItLevOccurrences` — the lexical scanner;
        ///      `updateAdmissionMapRecursion`, `isAdmitted` — the lockstep
        ///      insert sites for `productsOfRecursionIds`.
inline int countPatternOccurrences(StrSpan inputString,
            const HashMemory& localMemory, const NameMap& nameMap) {
            int counter = 0;
            scanItLevOccurrences(inputString, [&](int32_t start, int32_t len) {
                const StrSpan m(inputString.ptr + start, len);
                const NameId matchId = nameMap.lookup(m);
                if (matchId == 0
                    || !localMemory.productsOfRecursionIds.contains(matchId)) {
                    ++counter;
                }
            });
            return counter;
        }

// Standalone admission criterion: `stmt` is an operator invocation,
        // `var` sits at the operator's single declared input-arg index, AND
        // the same depth/secondary-count bounds used by the map-driven
        // `isAdmitted` gate hold (to prevent uncontrolled fan-out).
        // No admission map consulted. Operator output slots are deliberately
        // excluded — those have a working admission path via `isAdmitted` and
        // must not be double-admitted through this rule.
        /// @brief Pass-B single-input-operator gate.
        ///
        /// @details
        /// Returns true iff the candidate operator's `inputIndices.size() == 1`
        /// — the strict single-input-operator gate. This predicate is
        /// **load-bearing** for the Pass-B disintegration logic per
        /// [I-6](../../docs/agentic_swdd/30_invariants.md#i-6); widening it broke the
        /// Gauss batch on commit `463f402` and was reverted with the
        /// invariant added.
        ///
        /// @warning Per I-6, this gate must NOT be widened. Adding a
        ///          `inputIndices.size() == 2` branch silently admits
        ///          rules that the Gauss-batch-blow-up evidence proves
        ///          unsafe.
        /// @invariant [I-6](../../docs/agentic_swdd/30_invariants.md#i-6).
        /// @see [OPEN-3 in `docs/agentic_swdd/SwDD.md`](../../docs/agentic_swdd/SwDD.md#open-questions)
        ///      — the unresolved theoretical justification.
        /// @param mb Owning block, read-only.
        /// @param stmt Candidate defining expression.
        /// @param var Candidate witness token at the input slot.
        /// @param metrics Temporary-witness budgets, or null for permanent text.
        /// @return Whether the existing standalone admission rule qualifies.
inline bool isAllowedAsOperatorInput(const Memory& mb,
                                             StrSpan stmt,
                                             StrSpan var,
                                             const WitnessAdmissionMetrics* metrics = nullptr) {
            const StrSpan coreSpan = extractExpressionSpan(stmt);
            if (definitions.operatorNames().find(std::string_view(coreSpan.ptr,
                    static_cast<std::size_t>(coreSpan.len))) == definitions.operatorNames().end()) return false;
            const ce::CoreExpressionConfig* cfg = definitions.coreConfig(coreSpan);
            if (cfg == nullptr) return false;
            // Restrict to operators with exactly one input parameter — guardrail
            // against fan-out from multi-input operators (e.g. in3, fold).
            if (cfg->inputIndices.size() != 1 || cfg->outputIndices.empty()) return false;
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t argsN = getArgsSpans(stmt, args,
                                               ExecutionParameters::MAX_ARITY);
            const int idx = cfg->inputIndices[0];
            if (idx < 0 || idx >= argsN) return false;
            if (!equalSpans(args[idx], var)) return false;
            // Preimages only under functions the statement names as numbers
            // (D-471).
            for (int32_t i = 0; i < argsN; ++i) {
                if (i == idx) continue;
                bool output = false;
                for (const int out : cfg->outputIndices)
                    if (out == i) { output = true; break; }
                if (!output && !isNumberArgument(args[i])) return false;
            }

            const int generation = metrics ? metrics->generation : extractMaxIterationNumber(var);
            const int secondaryCount = metrics ? metrics->secondaryCount
                : countPatternOccurrences(stmt, mb.overallHashMemory, mb.nameMap);
            if (generation > parameters.maxIterationNumberVariable) return false;
            if (secondaryCount > parameters.maxNumberSecondaryVariables) return false;
            return true;
        }

/// @brief The input slots of an expression's core as a bit mask
        ///        (D-601).
        ///
        /// @details
        /// A core declared in the definition configuration
        /// (`CompiledDefinitionRegistry::configuredNames`) has the input
        /// slots its `inputIndices` name. Any other compiled entity (a
        /// numbered existence, and or implication) takes them from its
        /// elements: its argument at position i, the i-th parameter of its
        /// signature, is an input when some element holds that parameter at
        /// an input slot of the element's own core, read the same way. The
        /// declaration decides, not the presence of a configuration entry:
        /// `StructuralCompiler::extendCoreExpressionMap` gives every compiled
        /// entity it meets an entry with arity and definition sets but no
        /// input slots. The remainder existence `existence[N,+,*,a,d]`
        /// (`remainder[N,+,*,a,d,r]`) has the inputs a and d; the
        /// least-divisor existence `existence[N,+,*,c,n]`
        /// (`divisorSet[N,+,*,c,n,D]`, `hasMinimum[N,+,D]`) only n: the bound
        /// c is no input of `divisorSet`. A core that is neither declared nor
        /// compiled has no input slot (a defined result, not a miss).
        ///
        /// @param core  The core name of the expression.
        /// @param depth Nesting depth through compiled elements; definitions
        ///              are acyclic, asserted below 8.
        /// @return Bit i set iff argument i is an input.
        /// @invariant Reads the registry only; no allocation.
        /// @see inductionInputsAllowed — the strong catch's gate.
inline uint64_t inputSlotMask(StrSpan core, int depth = 0) const {
            assert(depth < 8 && "inputSlotMask: compiled definitions nest too deep");
            const std::string_view name = core.len > 0
                ? std::string_view(core.ptr, static_cast<std::size_t>(core.len)) : std::string_view();
            if (definitions.configuredNames().count(name) != 0) {
                const ce::CoreExpressionConfig* cfg = definitions.coreConfig(core);
                assert(cfg != nullptr && "inputSlotMask: a declared core has its configuration");
                uint64_t mask = 0;
                for (const int index : cfg->inputIndices) {
                    assert(index >= 0 && index < 64 && "inputSlotMask: an input slot outside the mask");
                    mask |= uint64_t(1) << index;
                }
                return mask;
            }
            const LogicalEntity* entity = definitions.compiledEntity(core);
            if (entity == nullptr) return 0;
            StrSpan signatureParameters[ExecutionParameters::MAX_ARITY];
            const int32_t parameterN = getArgsSpans(StrSpan(entity->signature), signatureParameters,
                ExecutionParameters::MAX_ARITY);
            assert(parameterN <= 64 && "inputSlotMask: a signature wider than the mask");
            uint64_t mask = 0;
            for (const std::string& element : entity->elements) {
                const StrSpan elementSpan(element);
                const uint64_t elementInputs = inputSlotMask(extractExpressionUniversalSpan(elementSpan), depth + 1);
                if (elementInputs == 0) continue;
                StrSpan args[ExecutionParameters::MAX_ARITY];
                const int32_t argsN = getArgsSpans(elementSpan, args, ExecutionParameters::MAX_ARITY);
                for (int32_t j = 0; j < argsN; ++j) {
                    if (((elementInputs >> j) & 1u) == 0) continue;
                    for (int32_t i = 0; i < parameterN; ++i)
                        if (equalSpans(args[j], signatureParameters[i])) mask |= uint64_t(1) << i;
                }
            }
            return mask;
        }

/// @brief Whether every input argument of an original is an induction
        ///        variable of the lemma — the strong catch's selection
        ///        (D-601).
        ///
        /// @details
        /// The strong step block's induction demands (the sum catch
        /// `m + x = n` and the product catch `x*m = n`) admit an original
        /// existence only when the smaller value it supplies is computed from
        /// the lemma's induction variables, as Euclid's remainder of a by n
        /// or a cofactor of n: every argument at an input slot of the
        /// original (`inputSlotMask`), a `u_` prefix stripped, must be a
        /// decimal token among @p variables. Arguments at other slots (sets,
        /// operators, a bound such as the 2 of the least-divisor existence)
        /// are not restricted here; the refused-digit gate
        /// (`hasOnlyInductionDigits`, D-502) still applies beside it. The
        /// remainder of 2 modulo n fails (2 is no induction variable); the
        /// remainder of a modulo b with both induction variables passes.
        ///
        /// @param original      The original expression the demand would
        ///                      admit a witness of.
        /// @param variables     The lemma's induction variables
        ///                      (`Memory::inductionVariableDigits`).
        /// @param variableCount Number of @p variables.
        /// @return True iff every argument at an input slot is one of the
        ///         induction variables.
        /// @invariant Pure read; no allocation.
        /// @see inputSlotMask; DisintegrationProcessor::disintegrateExpr2,
        ///      RejectedSourceReplay::queueParkedExistence — the callers.
inline bool inductionInputsAllowed(StrSpan original, const int32_t* variables, int32_t variableCount) const {
            assert(variableCount >= 0 && (variableCount == 0 || variables != nullptr)
                && "inductionInputsAllowed: variables without storage");
            const uint64_t inputs = inputSlotMask(extractExpressionUniversalSpan(original));
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t argsN = getArgsSpans(original, args, ExecutionParameters::MAX_ARITY);
            for (int32_t i = 0; i < argsN; ++i) {
                if (((inputs >> i) & 1u) == 0) continue;
                StrSpan argument = args[i];
                if (startsWithSpan(argument, "u_", 2))
                    argument = StrSpan(argument.ptr + 2, argument.len - 2);
                int value = 0;
                if (!parseDigitRunToInt(argument.ptr, argument.len, value)) return false;
                bool variable = false;
                for (int32_t k = 0; k < variableCount && !variable; ++k) variable = variables[k] == value;
                if (!variable) return false;
            }
            return true;
        }

/// @brief Whether a witness is the output of an operator element
        ///        (D-522).
        ///
        /// @details
        /// True iff @p element is a positive operator application (its core is
        /// an operator: a configured definition with input and output
        /// arguments) and @p var stands in one of the operator's output slots
        /// (`output_args`): `(in2[i,v,f])` makes `v` the value of `f` at `i`,
        /// `(in3[a,x,b,*])` makes `b` the product. A negated element, a
        /// relation, a compound element or an input slot is not an output. The
        /// witness-history check exempts an operator output: a function value
        /// is fixed by its inputs, so its history says nothing about circular
        /// construction.
        ///
        /// @param element One element of the witness's existence, in its
        ///                spelling.
        /// @param var     The witness's spelling in @p element.
        /// @return True iff @p var is an output of @p element's operator.
        /// @invariant Allocates and mints nothing.
        /// @see isAllowedAsOperatorInput, AdmissionMemory::witnessHistoryRefuses.
inline bool isOperatorOutput(StrSpan element, StrSpan var) const {
            if (element.len == 0 || element.ptr[0] != '(') return false;
            const StrSpan coreSpan = extractExpressionSpan(element);
            if (definitions.operatorNames().find(std::string_view(coreSpan.ptr,
                    static_cast<std::size_t>(coreSpan.len))) == definitions.operatorNames().end()) return false;
            const ce::CoreExpressionConfig* cfg = definitions.coreConfig(coreSpan);
            assert(cfg != nullptr && "isOperatorOutput: an operator has a configuration");
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t argsN = getArgsSpans(element, args, ExecutionParameters::MAX_ARITY);
            for (const int out : cfg->outputIndices) {
                assert(out >= 0 && out < argsN && "isOperatorOutput: output slot outside the arguments");
                if (equalSpans(args[out], var)) return true;
            }
            return false;
        }

/// @brief The largest iteration index `<i>` across the `it_<i>_lev_<l>_`
        ///        prefix-shape lexemes of @p s — the regex-free twin of the
        ///        former unanchored `it_(\d+)_lev_\d+_` regex iterator taking the
        ///        maximum of capture group 1.
        ///
        /// @details
        /// Scans via the `str_ops::scanItLevPrefixOccurrences` lexical scanner
        /// (the proven iterator twin of that UNANCHORED pattern — extent stops at
        /// the `'_'` after the second digit run, NO trailing digit run, so it is
        /// a DIFFERENT extent than the `it_\d+_lev_\d+_\d+` scanner) and takes
        /// the maximum of each match's first digit run, parsed with
        /// `str_ops::parseDigitRunToInt`. The -1 no-match sentinel is preserved.
        ///
        /// CONTRACT DIVERGENCE (documented, on impossible input): the former path
        /// parsed the capture with `std::atoi`, whose over-`INT_MAX` behaviour is
        /// platform-defined (MSVC saturates via strtol; 64-bit-long hosts
        /// truncate) — a latent cross-host divergence. This twin ASSERTS on an
        /// over-`INT_MAX` iteration index (Rule 19: an impossible state stops the
        /// world) rather than replicating platform-defined garbage. Iteration
        /// indices are small by construction, so this is byte-identical on every
        /// PRODUCIBLE input; the oracle test uses sane values only.
        ///
        /// The former `const std::string&` overload is retired; its verbatim
        /// regex body is reimplemented in-test as the differential oracle
        /// (`test_memory.cpp::extract_max_iteration_number_matches_regex_oracle`).
        ///
        /// @param s Expression view to scan.
        /// @return The maximum iteration index, or -1 when no lexeme matches.
        /// @see `str_ops::scanItLevPrefixOccurrences`, `str_ops::parseDigitRunToInt`.
inline int extractMaxIterationNumber(StrSpan s) {
            int maxVal = -1;
            scanItLevPrefixOccurrences(s, [&](int32_t start, int32_t end,
                                              int32_t d1Start, int32_t d1Len) {
                (void)start; (void)end;
                int v = 0;
                const bool ok = parseDigitRunToInt(s.ptr + d1Start, d1Len, v);
                assert(ok && "extractMaxIterationNumber: iteration index exceeds INT_MAX");
                (void)ok;
                if (v > maxVal) maxVal = v;
            });
            // Returns -1 if no matches (Python's default=None equivalent).
            return maxVal;
        }

/// @brief Decide whether a predicate may key an integration instruction.
        ///
        /// @details Predicates with exactly two inputs and no output cannot demand
        /// instruction replay, except positive equality. Typing and negative equality
        /// remain excluded. Other input counts and predicates with outputs retain
        /// their eligibility; excluded predicates remain instruction constituents.
        ///
        /// @param expression Signed constituent or marker expression.
        /// @return True when the configured predicate may provide a marker key.
        /// @invariant Does not mint names, mutate instructions or change facts.
        /// @see prepareIntegrationCore2, isAdmittedIntegration, updateAdmissionMapIntegration.
inline bool isIntegrationInstructionKeyEligible(StrSpan expression) const {
            const StrSpan core = extractExpressionUniversalSpan(expression);
            if (equalSpans(core, StrSpan("=", 1)))
                return expression.len > 0 && expression.ptr[0] == '(';
            if (equalSpans(core, StrSpan("in", 2)))
                return false;
            const ce::CoreExpressionConfig* cfg = this->definitions.coreConfig(core);
            return cfg != nullptr
                && !(cfg->inputIndices.size() == 2 && cfg->outputIndices.empty());
        }

/// @brief Bind read-only definitions and the existing admission parameters.
/// @details References outlive this policy; construction performs no classification.
/// @param definitions Compiled definition and configuration readers.
/// @param parameters Existing prover policy parameters.
/// @return Constructed admission policy.
AdmissionPolicy(const CompiledDefinitionRegistry& definitions, const ProverParameters& parameters);

/// @brief Test whether a premise key contains an unchangeable argument.
///
/// @details Uses admission conditions, definition reads and argument order.
/// @param key Ordered premise expressions.
/// @return Whether the documented condition holds.
//#pragma optimize("", off)

        // Detect u_-prefixed args in any key element (marker of a local
        // template rather than a broadcast/anchor theorem). Matches the same
        // convention used by extractRemainingArgs / replaceUSubstrings:
        // "u_" must appear directly after '[' or ',' — arg-position only.
inline static bool hasUPrefixInKey(const std::vector<std::string>& key) {
            for (const auto& el : key) {
                for (std::size_t i = 1; i + 1 < el.size(); ++i) {
                    if (el[i] == 'u' && el[i + 1] == '_' &&
                        (el[i - 1] == '[' || el[i - 1] == ',')) {
                        return true;
                    }
                }
            }
            return false;
        }

/// @brief Span-run overload of @ref hasUPrefixInKey — the span-native
        ///        twin; the `const std::vector<std::string>&` form is unchanged.
        ///
        /// @details Byte-identical scan: reads each element's bytes through the
        /// span (`el.ptr`/`el.len`) instead of a `std::string`, so a span-native
        /// caller (`makeNormalizedKeysForAdmission`'s span body) reaches it 0%
        /// heap. The `u_`-at-arg-position rule (`u_` directly after `[` or `,`)
        /// is unchanged.
        ///
        /// @param key  The key elements as a caller-owned `StrSpan` run.
        /// @param keyN The element count.
        /// @return `true` iff any element carries a `u_`-prefixed arg.
        /// @see hasUPrefixInKey(const std::vector<std::string>&).
inline static bool hasUPrefixInKey(const StrSpan* key, int32_t keyN) {
            for (int32_t k = 0; k < keyN; ++k) {
                const StrSpan el = key[k];
                for (int32_t i = 1; i + 1 < el.len; ++i) {
                    if (el.ptr[i] == 'u' && el.ptr[i + 1] == '_' &&
                        (el.ptr[i - 1] == '[' || el.ptr[i - 1] == ',')) {
                        return true;
                    }
                }
            }
            return false;
        }

/// @brief Check the local unchangeable-argument admission criterion.
///
/// @details Uses admission conditions, definition reads and argument order.
/// @param index Premise position to inspect.
/// @param key Ordered premise expressions.
/// @param value Head expression.
/// @return Whether the documented condition holds.
// User's "local u_-impl" admission criterion for
        // makeNormalizedKeysForAdmission. Returns true iff the element at
        // key[index] is an operator whose:
        //   - every input arg appears in some OTHER premise (paired),
        //   - output arg does NOT appear in any other premise,
        //   - value (head) is an operator that uses the output arg at one of
        //     its own input slots.
        // Designed to be additive to the theorem path's (validCount==n-1) /
        // (validCount==minLenLongKey-1) gates — qualifies local u_-impls that
        // otherwise fail implicationIsQualified / the per-element accept.
inline bool elementMatchesLocalUCriterion(
            std::size_t index,
            const std::vector<std::string>& key,
            const std::string& value) const {

            if (index >= key.size()) return false;
            const std::string& element = key[index];
            const ce::CoreExpressionConfig* cfg = this->definitions.coreConfig(extractExpressionSpan(StrSpan(element)));
            if (cfg == nullptr) return false;
            if (cfg->outputIndices.empty() || cfg->inputIndices.empty()) return false;

            // Row 242: getArgsSpans zero-copy [...] slicer into stack arrays;
            // spans alias element/key[i]/value (caller-owned, stable). Comparisons
            // via equalSpans == the former std::string ==.
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t argsN = getArgsSpans(StrSpan(element), args,
                                               ExecutionParameters::MAX_ARITY);
            int outIdx = cfg->outputIndices[0];
            if (outIdx < 0 || outIdx >= argsN) return false;
            const StrSpan outputArg = args[outIdx];

            // 1. Every input arg of element appears in some OTHER premise.
            for (int inIdx : cfg->inputIndices) {
                if (inIdx < 0 || inIdx >= argsN) continue;
                const StrSpan inputArg = args[inIdx];
                bool paired = false;
                for (std::size_t i = 0; i < key.size(); ++i) {
                    if (i == index) continue;
                    StrSpan iArgs[ExecutionParameters::MAX_ARITY];
                    const int32_t iArgsN = getArgsSpans(StrSpan(key[i]), iArgs,
                                                        ExecutionParameters::MAX_ARITY);
                    for (int32_t a = 0; a < iArgsN; ++a) {
                        if (equalSpans(iArgs[a], inputArg)) { paired = true; break; }
                    }
                    if (paired) break;
                }
                if (!paired) return false;
            }

            // 2. Output arg NOT in any other premise.
            for (std::size_t i = 0; i < key.size(); ++i) {
                if (i == index) continue;
                StrSpan iArgs[ExecutionParameters::MAX_ARITY];
                const int32_t iArgsN = getArgsSpans(StrSpan(key[i]), iArgs,
                                                    ExecutionParameters::MAX_ARITY);
                for (int32_t a = 0; a < iArgsN; ++a) {
                    if (equalSpans(iArgs[a], outputArg)) return false;
                }
            }

            // 3. Head (value) is an operator using output arg at one of its
            //    input slots.
            const ce::CoreExpressionConfig* hCfg = this->definitions.coreConfig(extractExpressionSpan(StrSpan(value)));
            if (hCfg == nullptr) return false;
            if (hCfg->outputIndices.empty() || hCfg->inputIndices.empty()) return false;
            StrSpan headArgs[ExecutionParameters::MAX_ARITY];
            const int32_t headArgsN = getArgsSpans(StrSpan(value), headArgs,
                                                   ExecutionParameters::MAX_ARITY);
            for (int inIdx : hCfg->inputIndices) {
                if (inIdx < 0 || inIdx >= headArgsN) continue;
                if (equalSpans(headArgs[inIdx], outputArg)) return true;
            }
            return false;
        }

/// @brief Span-run overload of @ref elementMatchesLocalUCriterion — the
        ///        span-native twin; the `const std::vector<std::string>&` form is
        ///        unchanged.
        ///
        /// @details Byte-identical verdict: the criterion (inputs paired in some
        /// other premise, output not in any other premise, head uses the output
        /// at an input slot) reads @p key / @p value through spans and
        /// `getArgsSpans` exactly as the owning form, so a span-native caller
        /// reaches it 0% heap.
        ///
        /// @param index The element index under test.
        /// @param key   The key elements as a caller-owned `StrSpan` run.
        /// @param keyN  The element count.
        /// @param value The head expression span.
        /// @return `true` iff `key[index]` matches the local-u criterion.
        /// @see elementMatchesLocalUCriterion(std::size_t, const
        ///      std::vector<std::string>&, const std::string&).
inline bool elementMatchesLocalUCriterion(
            std::size_t index,
            const StrSpan* key, int32_t keyN,
            StrSpan value) const {

            if (index >= static_cast<std::size_t>(keyN)) return false;
            const StrSpan element = key[index];
            const ce::CoreExpressionConfig* cfg = this->definitions.coreConfig(extractExpressionSpan(element));
            if (cfg == nullptr) return false;
            if (cfg->outputIndices.empty() || cfg->inputIndices.empty()) return false;

            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t argsN = getArgsSpans(element, args,
                                               ExecutionParameters::MAX_ARITY);
            int outIdx = cfg->outputIndices[0];
            if (outIdx < 0 || outIdx >= argsN) return false;
            const StrSpan outputArg = args[outIdx];

            // 1. Every input arg of element appears in some OTHER premise.
            for (int inIdx : cfg->inputIndices) {
                if (inIdx < 0 || inIdx >= argsN) continue;
                const StrSpan inputArg = args[inIdx];
                bool paired = false;
                for (int32_t i = 0; i < keyN; ++i) {
                    if (static_cast<std::size_t>(i) == index) continue;
                    StrSpan iArgs[ExecutionParameters::MAX_ARITY];
                    const int32_t iArgsN = getArgsSpans(key[i], iArgs,
                                                        ExecutionParameters::MAX_ARITY);
                    for (int32_t a = 0; a < iArgsN; ++a) {
                        if (equalSpans(iArgs[a], inputArg)) { paired = true; break; }
                    }
                    if (paired) break;
                }
                if (!paired) return false;
            }

            // 2. Output arg NOT in any other premise.
            for (int32_t i = 0; i < keyN; ++i) {
                if (static_cast<std::size_t>(i) == index) continue;
                StrSpan iArgs[ExecutionParameters::MAX_ARITY];
                const int32_t iArgsN = getArgsSpans(key[i], iArgs,
                                                    ExecutionParameters::MAX_ARITY);
                for (int32_t a = 0; a < iArgsN; ++a) {
                    if (equalSpans(iArgs[a], outputArg)) return false;
                }
            }

            // 3. Head (value) is an operator using output arg at one of its
            //    input slots.
            const ce::CoreExpressionConfig* hCfg = this->definitions.coreConfig(extractExpressionSpan(value));
            if (hCfg == nullptr) return false;
            if (hCfg->outputIndices.empty() || hCfg->inputIndices.empty()) return false;
            StrSpan headArgs[ExecutionParameters::MAX_ARITY];
            const int32_t headArgsN = getArgsSpans(value, headArgs,
                                                   ExecutionParameters::MAX_ARITY);
            for (int inIdx : hCfg->inputIndices) {
                if (inIdx < 0 || inIdx >= headArgsN) continue;
                if (equalSpans(headArgs[inIdx], outputArg)) return true;
            }
            return false;
        }

// Returns true iff the key qualifies via baseline's (A) or (B)
        // gates — i.e., would have been admitted independently of the
        // local u_-impl (C) extension. Used in makeNormalizedKeysForAdmission
        // to fence classic per-element accept to baseline-seen keys only.
        /// @brief Predicate: does the (key, value) pair pass the *baseline
        /// classic* qualification gate?
        ///
        /// @details
        /// Used by `makeNormalizedKeysForAdmission` to select classic
        /// per-element gating rather than the separate local-u criterion.
        /// The arithmetic route requires more than `minNumOperatorsKey`
        /// positive in3 premises and an in3 head. The long-key exception
        /// requires an unanchored product of disintegration: a premise
        /// argument starts with u_ and the premise count reaches
        /// `minLenLongKey`. An anchored theorem never gains this exception.
        ///
        /// @param key                Implication's key (string form).
        /// @param value              Head expression text.
        /// @param minNumOperatorsKey Threshold for the operator-count gate.
        /// @return True iff the pair matches the baseline classic shape.
        /// @see `implicationIsQualified` — superset gate that includes (C).
inline bool baselineClassicQualifies(const std::vector<std::string>& key,
                                             const std::string& value,
                                             int minNumOperatorsKey) const {
            int counterKey = 0;
            bool hasAnchor = false;
            for (std::size_t i = 0; i < key.size(); ++i) {
                const std::string& el = key[i];
                if (el.compare(0, 7, "(Anchor") == 0) hasAnchor = true;
                if (el.size() >= 4 && el[0] == '(' && el[1] == 'i' &&
                    el[2] == 'n' && el[3] == '3') {
                    counterKey += 1;
                }
            }
            const bool valueCond = (value.size() >= 5 &&
                value[0] == '(' && value[1] == 'i' && value[2] == 'n' &&
                value[3] == '3' && value[4] == '[');
            if (counterKey > minNumOperatorsKey && valueCond) return true;
            if (key.size() >= static_cast<std::size_t>(parameters.minLenLongKey)
                && !hasAnchor && hasUPrefixInKey(key)) return true;
            return false;
        }

/// @brief Span-run overload of @ref baselineClassicQualifies — the
        ///        span-native twin; the `const std::vector<std::string>&` form is
        ///        unchanged.
        ///
        /// @details Byte-identical verdict: counts `(in3` elements and tests the
        /// `(in3[` head shape / long-key length through spans, so a span-native
        /// caller reaches it 0% heap.
        ///
        /// @param key                The key elements as a `StrSpan` run.
        /// @param keyN               The element count.
        /// @param value              The head expression span.
        /// @param minNumOperatorsKey Threshold for the operator-count gate.
        /// @return True iff the pair matches the baseline classic shape.
        /// @see baselineClassicQualifies(const std::vector<std::string>&, const
        ///      std::string&, int).
inline bool baselineClassicQualifies(const StrSpan* key, int32_t keyN,
                                             StrSpan value,
                                             int minNumOperatorsKey) const {
            int counterKey = 0;
            bool hasAnchor = false;
            for (int32_t i = 0; i < keyN; ++i) {
                const StrSpan el = key[i];
                if (el.len >= 7 && equalSpans(StrSpan(el.ptr, 7), StrSpan("(Anchor", 7)))
                    hasAnchor = true;
                if (el.len >= 4 && el.ptr[0] == '(' && el.ptr[1] == 'i' &&
                    el.ptr[2] == 'n' && el.ptr[3] == '3') {
                    counterKey += 1;
                }
            }
            const bool valueCond = (value.len >= 5 &&
                value.ptr[0] == '(' && value.ptr[1] == 'i' && value.ptr[2] == 'n' &&
                value.ptr[3] == '3' && value.ptr[4] == '[');
            if (counterKey > minNumOperatorsKey && valueCond) return true;
            if (keyN >= parameters.minLenLongKey
                && !hasAnchor && hasUPrefixInKey(key, keyN)) return true;
            return false;
        }

/// @brief Predicate: does the rule qualify for regular admission
        /// marker installation?
        ///
        /// @details
        /// Wider gate than `baselineClassicQualifies`. Five branches
        /// in source comment shorthand:
        /// - **(A)** classic single-output path,
        /// - **(B)** long unanchored disintegration-product path,
        /// - **(C)** local-u path (admits keys with `u_`-prefixed args
        ///   that meet the local-u criterion),
        /// - **(D)** operator-relation path (some premise meets
        ///   `operatorRelationSlotQualifies`,
        ///   D-494),
        /// - **(E)** compound-operator path (some premise meets
        ///   `compoundOperatorSlotQualifies`,
        ///   D-569).
        ///
        /// Used by `makeNormalizedKeysForAdmission` to select regular
        /// marker variants. This verdict does not suppress ordinary
        /// head-rule installation or the independently qualified demand
        /// routes.
        ///
        /// @param key                Implication's key (string form).
        /// @param value              Head expression text.
        /// @param minNumOperatorsKey Threshold for the operator-count gate.
        /// @return True iff the pair qualifies for regular admission markers.
inline bool implicationIsQualified(const std::vector<std::string>& key, const std::string& value, int minNumOperatorsKey) const {
            bool result = baselineClassicQualifies(key, value, minNumOperatorsKey);

            // (C) Local u_-impl: qualify if any element matches the
            //     user's criterion (inputs paired, output only in head
            //     as input, head is operator). Strictly additive — never
            //     narrows the existing (A)/(B) gates above.
            if (!result && hasUPrefixInKey(key)) {
                for (std::size_t i = 0; i < key.size(); ++i) {
                    if (elementMatchesLocalUCriterion(i, key, value)) {
                        result = true;
                        break;
                    }
                }
            }

            // (D) Operator-relation: qualify if some premise is a missing
            //     in3 whose output alone is unspecified and sits in the head.
            if (!result) {
                constexpr std::size_t kKeyCap =
                    ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 1;
                assert(key.size() <= kKeyCap
                    && "implicationIsQualified: key size exceeds cap");
                StrSpan spans[kKeyCap];
                for (std::size_t i = 0; i < key.size(); ++i) spans[i] = StrSpan(key[i]);
                const int32_t keyN = static_cast<int32_t>(key.size());
                for (int32_t i = 0; i < keyN; ++i) {
                    if (operatorRelationSlotQualifies(spans, keyN, i,
                            StrSpan(value), minNumOperatorsKey)) {
                        result = true;
                        break;
                    }
                }
                // (E) Compound-operator: qualify if some premise is the one
                //     missing compound operator of a four-operator rule with
                //     a compound head.
                for (int32_t i = 0; i < keyN && !result; ++i) {
                    if (compoundOperatorSlotQualifies(spans, keyN, i, StrSpan(value)))
                        result = true;
                }
            }

            return result;
        }

/// @brief Span-run overload of @ref implicationIsQualified — the
        ///        span-native twin; the `const std::vector<std::string>&` form is
        ///        unchanged.
        ///
        /// @details Byte-identical verdict: runs the (A)/(B)/(C)/(D)/(E) gates over @p key
        /// / @p value spans, delegating to the span-run `hasUPrefixInKey` /
        /// `elementMatchesLocalUCriterion` overloads, so a span-native caller
        /// (`makeNormalizedKeysForAdmission`'s span body) reaches it 0% heap.
        ///
        /// @param key                The key elements as a `StrSpan` run.
        /// @param keyN               The element count.
        /// @param value              The head expression span.
        /// @param minNumOperatorsKey Threshold for the operator-count gate.
        /// @return True iff the pair qualifies for regular admission markers.
        /// @see implicationIsQualified(const std::vector<std::string>&, const
        ///      std::string&, int).
inline bool implicationIsQualified(const StrSpan* key, int32_t keyN,
            StrSpan value, int minNumOperatorsKey) const {
            bool result = baselineClassicQualifies(key, keyN, value, minNumOperatorsKey);

            if (!result && hasUPrefixInKey(key, keyN)) {
                for (int32_t i = 0; i < keyN; ++i) {
                    if (elementMatchesLocalUCriterion(
                            static_cast<std::size_t>(i), key, keyN, value)) {
                        result = true;
                        break;
                    }
                }
            }

            if (!result) {
                for (int32_t i = 0; i < keyN; ++i) {
                    if (operatorRelationSlotQualifies(key, keyN, i, value,
                            minNumOperatorsKey)) {
                        result = true;
                        break;
                    }
                }
            }

            if (!result) {
                for (int32_t i = 0; i < keyN && !result; ++i) {
                    if (compoundOperatorSlotQualifies(key, keyN, i, value))
                        result = true;
                }
            }

            return result;
        }

/// @brief Check whether an argument occupies a configured head output slot.
///
/// @details Uses admission conditions, definition reads and argument order.
/// @param outputArg Output argument to match.
/// @param value Head expression.
/// @return Whether the documented condition holds.
// Narrowing guard for local u_-impl marker registration: returns
        // true if the element's output arg also appears at the head's
        // OUTPUT slot. Such a template would just rewrite the head's own
        // output into a marker chain — a likely source of unbounded
        // propagation in updateAdmissionMapRecursion / consumer chains.
        // Only meaningful to consult when hasUPrefixInKey(key) is true.
inline bool outputMatchesHeadOutputSlot(
            const std::string& outputArg,
            const std::string& value) const {
            return outputMatchesHeadOutputSlot(StrSpan(outputArg), value);
        }

/// @brief `StrSpan` overload of @ref outputMatchesHeadOutputSlot — the
        ///        real implementation; the `const std::string&` form forwards.
        ///
        /// @details Byte-identical: reads @p outputArg only through `equalSpans`
        /// against the head's output-slot arg span, so no `std::string` is
        /// materialized when the caller already holds a span.
        ///
        /// @param outputArg The candidate output-argument text.
        /// @param value     The head expression span (a `const std::string&`
        ///                  caller converts implicitly).
        /// @return `true` iff the head's output slot equals @p outputArg.
        /// @see outputMatchesHeadOutputSlot(const std::string&, const std::string&).
inline bool outputMatchesHeadOutputSlot(
            StrSpan outputArg,
            StrSpan value) const {
            const ce::CoreExpressionConfig* hCfg = this->definitions.coreConfig(extractExpressionSpan(value));
            if (hCfg == nullptr) return false;
            if (hCfg->outputIndices.empty()) return false;
            StrSpan headArgs[ExecutionParameters::MAX_ARITY];
            const int32_t headArgsN = getArgsSpans(value, headArgs,
                                                   ExecutionParameters::MAX_ARITY);
            int headOut = hCfg->outputIndices[0];
            if (headOut < 0 || headOut >= headArgsN) return false;
            return equalSpans(headArgs[headOut], outputArg);
        }

/// @brief The ordis-only qualification gate — does a rule that FAILED
        ///        the regular (A) classic / (B) long-key / (C) local-u
        ///        admission gates still qualify for ordis-route marker keys?
        ///
        /// @details
        /// A rule qualifies iff every non-anchor element of its key AND its
        /// head are OPERATOR APPLICATIONS — the core is in the `operators`
        /// registry (a config declaring both input and output slots), which
        /// excludes equalities, typing atoms, and compacts by construction;
        /// a negated element disqualifies outright — and the rule is at
        /// least `kOrdisMinOperatorExpressions` (= 4) operator expressions
        /// long, the head included, anchor elements excluded from both the
        /// requirement and the count (corpus row 34: 3 premises + head = 4,
        /// passes at the boundary — the maintainer-confirmed reading).
        ///
        /// Keys fired through this route carry the `ordisOnly` value tag:
        /// they are or-cohort opening demand evidence only, invisible to
        /// general Pass-B admission.
        ///
        /// @param key   The key elements as a `StrSpan` run.
        /// @param keyN  The element count.
        /// @param value The head expression span.
        /// @return Whether the ordis candidate loop should run.
        /// @invariant Heap-free: span slicing + transparent `operators`
        ///            probes only (Rule 28). Read-only.
        /// @see makeNormalizedKeysForAdmission — the sole caller;
        ///      `AdmissionMapValue::ordisOnly` — the tag the route installs.
bool ordisRouteQualifies(const StrSpan* key, int32_t keyN,
                                 StrSpan value) const;

/// @brief The ordis2 pair's shared key-language predicate — may this
        ///        expression text be a demand key / park filing key
        ///        (D-267)?
        ///
        /// @details
        /// One predicate for BOTH halves of the pair so the two key
        /// populations can never diverge: the demand slot filter
        /// (`ordis2DemandSlotQualifies`, filters 2+3) and the park-side
        /// disjunct filing in `consumeOrLeavesCohort` both call it. A text
        /// is eligible iff (polarity-stripped — a negated form qualifies on
        /// its core, I-175, while demand and filing keep the `!` verbatim)
        /// its core has a compiled NON-atomic entity, and its argument
        /// count reaches `ExecutionParameters::kOrdis2DemandMinArity`. The
        /// `compiledEntity` fence alone is the operator test — it admits
        /// PREDICATES (preorder / strictOrder, empty output args), which
        /// the input-AND-output `operators` set excludes; consulting
        /// `operators` here was the 3X defect that silently disqualified
        /// the whole order family. Anchors and equality/membership shapes
        /// fail by construction (no compiled entity / arity).
        ///
        /// @param elem The clean expression text (no `u_` prefixes; polarity
        ///             verbatim).
        /// @return `true` iff the text is in the pair's key language.
        /// @invariant Read-only — consults `operators` and the I-137
        ///            `compiledEntity` fence, mints nothing.
        /// @see `consumeOrLeavesCohort` — the park filing;
        ///      `ExecutionParameters::kOrdis2DemandMinArity`.
bool ordis2KeyEligible(StrSpan elem) const;

/// @brief May premise slot @p index of an installing rule mint
        ///        ordis2 demand — the qualifying-slot analysis
        ///        (D-267)?
        ///
        /// @details
        /// Two layers. First the pair's shared key-language predicate
        /// @ref ordis2KeyEligible (compound with a compiled non-atomic
        /// entity, arity ≥ `kOrdis2DemandMinArity`, anchor refuse, polarity
        /// transparent). Then the SUBSET test: every argument of the slot
        /// must be bound by the rule's other NON-anchor premises, EXCEPT an
        /// anchor-slot argument, which is ground by construction and counts
        /// as bound (D-287 — the numeral 1 of `1 ≤ d` names an anchor slot
        /// and no substantive premise can ever bind it). A slot bound by
        /// the anchor alone still mints nothing (the A14 shape): at least
        /// one argument must come from a substantive premise. B5 (two
        /// product premises + the 1≤c guard bind all four slot args)
        /// qualifies — the load-bearing consumer; B9's `1 ≤ d` slot
        /// qualifies through the anchor-slot exception. The final
        /// minimum-premise gate is justification-split: general rules need
        /// `kOrdis2DemandMinPremises` NON-anchor premises, integration-
        /// justified rules (a goal's flat integration-rule variants and
        /// per-level conjunct rules) need
        /// `kOrdis2DemandMinPremisesIntegration` — the D1 closure rule
        /// (three typings + two matched conjuncts + the one missing ground
        /// compound premise), a floor above the general one.
        ///
        /// @param key   The rule's premise run (anchor included).
        /// @param keyN  The premise count.
        /// @param index The candidate slot.
        /// @param integrationJustified Whether the installing rule carries
        ///              the `"integration"` justification — selects the
        ///              integration floor in the minimum-premise gate.
        /// @return `true` iff the slot installs a demand variant.
        /// @invariant Read-only; mints nothing.
        /// @see makeNormalizedKeysForAdmission — the third qualification
        ///      pass; `ordis2KeyEligible` — the shared key-language core.
bool ordis2DemandSlotQualifies(const StrSpan* key,
            int32_t keyN, int32_t index,
            bool integrationJustified = false) const;

/// @brief Are all arguments of premise slot @p index fixed by the rule's
        ///        other premises — the subset test of the demand routes?
        ///
        /// @details
        /// An argument is fixed when another NON-anchor premise names it, or
        /// when it is an anchor-slot argument: ground by construction, since
        /// it names a fixed value of the anchor context (the numeral 1 in
        /// `1 <= d`, the carrier set, an operator) that no substantive premise
        /// can ever bind (D-287). A slot whose arguments are anchor slots
        /// alone does not qualify (the A14 shape): at least one argument must
        /// come from a substantive premise. Shared by the ordis2 demand route
        /// and the goal near-miss route.
        ///
        /// @param key   The rule's premise run (anchor included).
        /// @param keyN  The premise count.
        /// @param index The candidate slot.
        /// @return `true` iff every argument of `key[index]` is fixed and at
        ///         least one by a substantive premise.
        /// @invariant Read-only; mints nothing.
        /// @see ordis2DemandSlotQualifies, goalNearMissSlotQualifies.
bool slotArgumentsBoundByOthers(const StrSpan* key, int32_t keyN,
            int32_t index) const;

/// @brief May premise slot @p index of a goal's flat integration rule
        ///        install a near-miss demand variant
        ///        (D-574)?
        ///
        /// @details
        /// The slot is a positive compound premise (a compiled, non-atomic
        /// entity; never the anchor), every argument of it is fixed by the
        /// other premises (`slotArgumentsBoundByOthers`), and at least
        /// `ExecutionParameters::kGoalDemandMinPresentPremises` other
        /// NON-anchor premises sit beside it. When those others hold and the
        /// slot does not, the variant reports the slot's ground text: the
        /// one expression the goal's integration misses. No arity floor (the
        /// missing `ordered[N,+,G]` has three arguments) and no shared key
        /// language with the OR demands: only the goal-driven rule search
        /// reads the report.
        ///
        /// @param key   The flat rule's premise run (anchor included).
        /// @param keyN  The premise count.
        /// @param index The candidate slot.
        /// @return `true` iff the slot installs a near-miss demand variant.
        /// @invariant Read-only; mints nothing.
        /// @see AdmissionMarkerBuilder::makeNormalizedKeysForAdmission — the
        ///      near-miss pass; slotArgumentsBoundByOthers.
bool goalNearMissSlotQualifies(const StrSpan* key, int32_t keyN,
            int32_t index) const;

/// @brief May input slot @p inputPos of premise @p index mint an
        ///        UNTAGGED input-slot demand key — the witness-release
        ///        analysis (D-288)?
        ///
        /// @details
        /// The algebra-map sibling of the ordis input-slot candidate loop,
        /// sharing its three structural conditions on the candidate (the
        /// argument at the premise's config input slot @p inputPos):
        /// (1) the candidate also sits at a config input slot of the HEAD,
        /// (2) confinement — the candidate appears in no other key element
        /// (the head is allowed), (3) concreteness — every other argument
        /// of the marked premise appears in some other key element (anchor
        /// included), a repeated candidate failing. A negated or anchor
        /// premise, a non-config core (equality), an operator-less head,
        /// and an out-of-range slot are defined misses. Checked LAST, the
        /// fan-out gate: at least
        /// `ExecutionParameters::kInputSlotDemandMinPremises` NON-anchor
        /// premises (the D-267 RT lesson, mirrored). The qualifying shape
        /// is the C8 difference-transport rule: its `x + z = y` premise
        /// carries the witness `z` at an input slot of both the premise and
        /// the head while the three product/sum premises bind `x` and `y` —
        /// the fired (instantiated) key is byte-identical to the marker
        /// form under which the order-witness product parks in
        /// `rejectedMap`.
        ///
        /// @param key      The rule's premise run (anchor included).
        /// @param keyN     The premise count.
        /// @param index    The candidate premise slot.
        /// @param inputPos The position within the premise operator's
        ///                 `inputIndices` (NOT the raw argument index).
        /// @param value    The head expression text.
        /// @return `true` iff the (premise, input slot) pair installs an
        ///         untagged demand variant.
        /// @invariant Read-only; mints nothing.
        /// @see makeNormalizedKeysForAdmission — the fourth qualification
        ///      pass; `ordis2DemandSlotQualifies` — the compound-slot
        ///      sibling; `isAdmittedIncludingAncestors` — the consumer-side
        ///      probe the route's keys are read through.
bool inputSlotDemandSlotQualifies(const StrSpan* key,
            int32_t keyN, int32_t index, int32_t inputPos,
            StrSpan value) const;

/// @brief Decide whether one (premise, input slot) pair of a
        ///        RECEIVED THEOREM rule installs an untagged input-slot
        ///        demand variant through the theorem-relation route
        ///        (D-370) — the fifth,
        ///        independent qualification pass of
        ///        `makeNormalizedKeysForAdmission`.
        ///
        /// @details
        /// The route the D2 remainder-uniqueness stall needed: the
        /// difference witness of `q < q'` sits at an INPUT slot of the
        /// two-input `in3`, its only flat consumer (`a<b ∧ a+k=b → k≠0`)
        /// has a negated-equality head and two non-anchor premises, so the
        /// D-288 route (head-input condition, premise floor) never files
        /// the demand and the witness parks forever. This route replaces
        /// the head-input condition and the premise floor with a
        /// STRUCTURAL description of the rule: a received theorem whose
        /// other premises are all compound relations.
        ///
        /// A pair qualifies iff ALL hold: (1) the rule is a received theorem
        /// — its key carries an anchor element, no premise argument is
        /// `u_`-prefixed (so the install's D-32 `productOfDisintegration`
        /// is false) and the justification is not integration; (2) the
        /// marked premise is a positive config-operator application and
        /// `inputPos` names one of its config input slots; (3) every OTHER
        /// non-anchor premise is a positive config-backed compound relation
        /// (`compiledEntity` category non-atomic AND `coreConfig` present —
        /// `strictOrder`, `preorder`, `remainder`, …; atoms such as `in`,
        /// equalities and runtime compacts refuse); (4) confinement — the
        /// candidate appears in no other key element; (5) concreteness —
        /// every other argument of the marked premise appears in some other
        /// key element (anchor included), a repeated candidate failing; (6)
        /// at least `ExecutionParameters::kTheoremRelationDemandMinBoundArgs`
        /// of the marked premise's other arguments are bound by a RELATION
        /// premise (anchor slots do not count) — the fan-out gate that keeps
        /// the successor-witness shapes out. The head is unconstrained
        /// (negated heads allowed). Output slots are not this route's
        /// business: the regular pass already marks them.
        ///
        /// @param key      The rule's premise run (anchor included).
        /// @param keyN     The premise count.
        /// @param index    The candidate premise slot.
        /// @param inputPos The position within the premise operator's
        ///                 `inputIndices` (NOT the raw argument index).
        /// @param integrationJustified Whether the install carries the
        ///                 integration justification (refuses).
        /// @return `true` iff the (premise, input slot) pair installs an
        ///         untagged demand variant.
        /// @invariant Read-only; mints nothing.
        /// @see inputSlotDemandSlotQualifies — the D-288 sibling;
        ///      makeNormalizedKeysForAdmission — the fifth qualification
        ///      pass; `isAdmittedIncludingAncestors` — the consumer-side
        ///      probe the route's keys are read through.
bool theoremRelationDemandSlotQualifies(const StrSpan* key,
            int32_t keyN, int32_t index, int32_t inputPos,
            bool integrationJustified) const;

/// @brief Predicate: is @p elem a positive relation premise for the
        ///        operator-relation admission route?
        ///
        /// @details
        /// A relation premise is a positive application of a compiled
        /// compound definition (`compiledEntity` category non-atomic) whose
        /// configuration has no output slot: a predicate such as `bezEq`,
        /// `preorder` or `strictOrder`. Anchors, negated texts, atoms
        /// (`in`, `in3`, equality), runtime compacts without an entity and
        /// defined operators with an output slot are not relations.
        /// Read-only (D-494).
        ///
        /// @param elem One key element, polarity verbatim.
        /// @return `true` iff @p elem is a positive relation premise.
        /// @invariant Read-only; mints nothing.
        /// @see operatorRelationSlotQualifies — the sole production caller.
bool isRelationPremise(StrSpan elem) const;

/// @brief Decide whether premise slot @p index of a rule is the missing
        ///        premise of the operator-relation admission route.
        ///
        /// @details
        /// Extends the classic missing-premise admission, which needs more
        /// than `minNumOperatorsKey` in3 premises and an in3 head, to rules
        /// that pair the same in3 count with a relation and end in any head
        /// (D-494). The slot qualifies
        /// iff ALL hold: (1) the key carries more than `minNumOperatorsKey`
        /// positive in3 premises, the marked one included, and at least one
        /// relation premise (`isRelationPremise`); (2) the marked premise is
        /// a positive in3; (3) all of its arguments but one are specified:
        /// every argument other than the output appears in some other key
        /// element, anchor included, and a repeated output fails; (4) the
        /// one missing argument, the output, appears in no other key element
        /// and appears among the head's arguments. The regular pass then
        /// installs the slot's output-marker variant: when the other
        /// premises match, the demand asks for the operator's value on the
        /// specified inputs.
        ///
        /// @param key                The rule's premise run (anchor included).
        /// @param keyN               The premise count.
        /// @param index              The candidate premise slot, in range.
        /// @param value              The head expression span.
        /// @param minNumOperatorsKey The classic operator-count threshold.
        /// @return `true` iff the slot installs an output-marker variant.
        /// @invariant Read-only; mints nothing.
        /// @see implicationIsQualified — route (D);
        ///      makeNormalizedKeysForAdmission — the regular pass.
bool operatorRelationSlotQualifies(const StrSpan* key, int32_t keyN,
            int32_t index, StrSpan value, int minNumOperatorsKey) const;

/// @brief Decide whether premise slot @p index of a rule is the missing
        ///        compound premise of the compound-operator admission route.
        ///
        /// @details
        /// The route the I4 append step needs
        /// (D-569): FTA2 row 16 (k = s(m), Q on
        /// [0,m], P = Q with x appended at k, Π(Q) = p, p·x = w ⟹ Π(P) = w)
        /// has the four operator premises `in2`, `appendToSequence`, `fold`
        /// and `in3`, and its one missing premise is the compound append whose
        /// output P only the compound head names. The slot qualifies iff ALL
        /// hold: (1) the key carries exactly
        /// `ExecutionParameters::kCompoundOperatorRoutePremises` positive
        /// operator premises (a configuration with an output slot, the
        /// anchor excluded), exactly
        /// `ExecutionParameters::kCompoundOperatorRouteCompounds` of them
        /// compound (a compiled non-atomic entity); (2) the marked premise is
        /// one of those compound operators; (3) all of its arguments but its
        /// output are specified: each occurs in another key element, anchor
        /// included, and a repeated output fails; (4) the output occurs in no
        /// other key element and among the head's arguments; (5) the head is
        /// a positive compound expression; (6) the marked premise is the
        /// rule's only compound premise meeting (2) to (4). FTA pool row 87
        /// (k = s(m), Q on [0,k], x = Q(k), Π(Q,[0,m]) = p, Π(Q,[0,k]) = w ⟹
        /// p·x = w) refuses on both counts: its in3 head is atomic, and both
        /// fold values are missing. The regular pass then installs the slot's
        /// output-marker variant: when the other premises match, the demand
        /// asks for the operator's value on the specified inputs, which
        /// admits the witness algebraically.
        ///
        /// @param key   The rule's premise run (anchor included).
        /// @param keyN  The premise count.
        /// @param index The candidate premise slot, in range.
        /// @param value The head expression span.
        /// @return `true` iff the slot installs an output-marker variant.
        /// @invariant Read-only; mints nothing.
        /// @see implicationIsQualified — route (E);
        ///      operatorRelationSlotQualifies — the in3 sibling (D);
        ///      makeNormalizedKeysForAdmission — the regular pass.
bool compoundOperatorSlotQualifies(const StrSpan* key, int32_t keyN,
            int32_t index, StrSpan value) const;

private:
const CompiledDefinitionRegistry& definitions;
const ProverParameters& parameters;
};
} // namespace gl
