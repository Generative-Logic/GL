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
#include "../compiler.hpp"
#include "../compiler/compiled_definition_registry.hpp"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstring>
#include <mutex>

namespace gl {

/// @brief Capacity of a demand-built rule's text, witness template and carrier.
inline constexpr int32_t kDemandBuiltTextCap = 512;
/// @brief Capacity of the process-wide demand-built rule registry.
inline constexpr int32_t kDemandBuiltRuleCap = 16;
/// @brief Capacity of a demand-built rule's premises (the anchor premise
///        included).
inline constexpr int32_t kDemandBuiltPremiseCap = 8;
/// @brief Capacity of a demand-built rule's bound variables (anchor slots
///        and typed variables).
inline constexpr int32_t kDemandBuiltBoundCap = 16;
/// @brief Capacity of an OR-headed demand-built rule's trigger templates.
inline constexpr int32_t kDemandBuiltTriggerCap = 32;
/// @brief Capacity of the text of all trigger templates of one rule.
inline constexpr int32_t kDemandBuiltTriggerTextCap = 2048;

/// @brief The head shape of a demand-built rule.
enum class DemandBuiltKind : uint8_t {
    /// The totality of an anchor operator: an existence whose witness is
    /// the operator's output on the typed variables.
    OperatorWitness,
    /// A positive OR over the typed variables and anchor arguments.
    OrHead,
};

/// @brief The outcome of one demand-side build.
enum class DemandBuildResult : uint8_t {
    /// The instance entered the internal mail.
    Built,
    /// The block holds the rule nowhere on the demand scope's chain.
    NoRule,
    /// Two premises instantiate to the same statement.
    Diagonal,
    /// A premise is not known on the demand scope's chain.
    Untyped,
    /// A premise is beyond the generation cap.
    BeyondGeneration,
    /// The instance is already known on the firing scope's chain.
    Known,
};

/// @brief The event a trigger template of an OR-headed rule stands for.
enum class DemandBuiltTrigger : uint8_t {
    /// A known fact that refutes a disjunct: its complement, or the premise
    /// of a refutation compact the OR's registration publishes for an `and`
    /// disjunct.
    Refutation,
    /// A case-split demand for a disjunct (ground disjunct text).
    CaseDemand,
    /// A goal equal to the OR instance.
    Goal,
};

/// @brief A slice of a `DemandBuiltRule`'s own text buffer.
///
/// @details
/// Offsets, not pointers: a record is copied into the registry, and a slice
/// stays valid against whichever copy reads it.
struct DemandBuiltSlice {
    /// @brief Byte offset of the slice in `DemandBuiltRule::text`.
    int32_t offset = 0;
    /// @brief Byte length of the slice.
    int32_t length = 0;
};

/// @brief One demand-built rule, parsed once when a block installs it.
///
/// @details
/// Two head shapes qualify (`DemandBuiltKind`). The totality rule of an
/// anchor operator: every premise types one bound variable in an anchor set,
/// and the head is an existence whose witness is the operator's output on
/// exactly those variables
/// (`(>[1](in[1,u_1])(>[2](in[2,u_1])(existence1[u_1,1,2,u_4])))`); its
/// witness template is the operator application with the witness slot
/// spelled `marker` (`(in3[1,2,marker,u_4])`). A typed rule with an OR head:
/// an optional anchor premise binding the anchor slots, typing premises of at
/// least two variables in anchor sets, and a positive OR over those variables
/// and anchor arguments
/// (`(>[1,..,8](AnchorFTA[1,..,8])(>[9](in[9,1])(>[10](in[10,1])(or11[1,4,9,10]))))`);
/// its trigger templates spell the events that let the OR derive something
/// (`DemandBuiltTrigger`). The record holds the rule as installed, its
/// carrier, the premise and head slices, every bound variable (an anchor
/// slot carries its fixed anchor value), and the templates, each in the
/// rule's own spelling. Fixed-size, heap-free.
struct DemandBuiltRule {
    /// @brief The head shape.
    DemandBuiltKind kind = DemandBuiltKind::OperatorWitness;
    /// @brief The rule text as installed (full-bind implication).
    char text[kDemandBuiltTextCap];
    /// @brief Length of `text`.
    int32_t textLength = 0;
    /// @brief The witness template (operator totality only).
    char witness[kDemandBuiltTextCap];
    /// @brief Length of `witness`.
    int32_t witnessLength = 0;
    /// @brief The carrier statement whose expansion produced the rule.
    char carrier[kDemandBuiltTextCap];
    /// @brief Length of `carrier`.
    int32_t carrierLength = 0;
    /// @brief Number of premises, the anchor premise included.
    int32_t premiseCount = 0;
    /// @brief Each premise in chain order: the anchor premise, if any, and
    ///        the typing premises `(in[v,S])`.
    DemandBuiltSlice premises[kDemandBuiltPremiseCap];
    /// @brief Number of bound variables (anchor slots and typed variables).
    int32_t boundCount = 0;
    /// @brief Every bound variable. For the operator totality, the variable
    ///        each premise types, in premise order.
    DemandBuiltSlice bound[kDemandBuiltBoundCap];
    /// @brief The anchor value of each bound variable that is an anchor
    ///        slot, a slice of `anchorValues`; empty for a typed variable.
    DemandBuiltSlice fixedValue[kDemandBuiltBoundCap];
    /// @brief The anchor values the slots are bound to.
    char anchorValues[kDemandBuiltTextCap];
    /// @brief Length of `anchorValues`.
    int32_t anchorValuesLength = 0;
    /// @brief The head `(E[...])` or `(or<N>[...])`.
    DemandBuiltSlice head;
    /// @brief The trigger templates (OR heads only), back to back.
    char triggers[kDemandBuiltTriggerTextCap];
    /// @brief Length of `triggers`.
    int32_t triggersLength = 0;
    /// @brief Number of trigger templates.
    int32_t triggerCount = 0;
    /// @brief Each trigger template, a slice of `triggers`.
    DemandBuiltSlice triggerSlices[kDemandBuiltTriggerCap];
    /// @brief The event each trigger template stands for.
    DemandBuiltTrigger triggerKinds[kDemandBuiltTriggerCap];

    /// @brief The rule text.
    /// @return A span over `text`.
    StrSpan textSpan() const { return StrSpan(text, textLength); }
    /// @brief The witness template.
    /// @return A span over `witness`.
    StrSpan witnessSpan() const { return StrSpan(witness, witnessLength); }
    /// @brief The carrier statement.
    /// @return A span over `carrier`.
    StrSpan carrierSpan() const { return StrSpan(carrier, carrierLength); }
    /// @brief One slice of the rule text.
    /// @param slice The slice.
    /// @return A span over `text`.
    StrSpan sliceSpan(DemandBuiltSlice slice) const { return StrSpan(text + slice.offset, slice.length); }
    /// @brief The anchor value a bound variable is fixed to.
    /// @param index Position below `boundCount`.
    /// @return The value; empty for a typed variable.
    StrSpan fixedSpan(int32_t index) const {
        return StrSpan(anchorValues + fixedValue[index].offset, fixedValue[index].length);
    }
    /// @brief One trigger template.
    /// @param index Position below `triggerCount`.
    /// @return A span over `triggers`.
    StrSpan triggerSpan(int32_t index) const {
        return StrSpan(triggers + triggerSlices[index].offset, triggerSlices[index].length);
    }
};

/// @brief An argument without one leading `u_`.
///
/// @details
/// Rule texts spell fixed (anchor) arguments `u_<name>`; demand keys and
/// facts spell them bare. Comparisons and instantiations read the bare name.
///
/// @param argument One argument token.
/// @return The token without its `u_` prefix, or the token itself.
inline StrSpan demandBuiltArgumentCore(StrSpan argument) {
    if (argument.len >= 2 && argument.ptr[0] == 'u' && argument.ptr[1] == '_')
        return StrSpan(argument.ptr + 2, argument.len - 2);
    return argument;
}

/// @brief Write a flat statement with its arguments renamed.
///
/// @details
/// @p text is a flat `(core[args])`, optionally negated. Every argument is
/// replaced by the target of the equal source token; @p toggleNegation
/// writes the complement instead (a leading `!` dropped or added). An
/// argument with no source token, or one that is itself an expression, is a
/// token the renaming does not cover: the answer false is the predicate's,
/// and the caller's criterion does not hold.
///
/// @param text           The flat statement.
/// @param from           The source tokens.
/// @param to             Their targets, position for position.
/// @param n              Number of source tokens.
/// @param toggleNegation Write the complement.
/// @param out            Receives the text.
/// @param cap            Capacity of @p out (asserted).
/// @param length         Receives the text's length on true.
/// @return True when every argument has a source token.
/// @invariant Pure; mints and allocates nothing.
/// @see analyzeDemandBuiltRule.
inline bool demandBuiltRename(StrSpan text, const StrSpan* from, const StrSpan* to, int32_t n,
    bool toggleNegation, char* out, int32_t cap, int32_t& length) {
    const bool negated = text.len > 0 && text.ptr[0] == '!';
    const StrSpan body = negated ? StrSpan(text.ptr + 1, text.len - 1) : text;
    if (body.len < 3 || body.ptr[0] != '(') return false;
    StrSpan arguments[ExecutionParameters::MAX_ARITY];
    const int32_t argumentN = getArgsSpans(body, arguments, ExecutionParameters::MAX_ARITY);
    length = 0;
    const auto append = [&](StrSpan piece) {
        assert(length + piece.len < cap && "demandBuiltRename: text exceeds its capacity");
        std::memcpy(out + length, piece.ptr, static_cast<std::size_t>(piece.len));
        length += piece.len;
    };
    if (negated != toggleNegation) append(StrSpan("!", 1));
    append(StrSpan("(", 1));
    append(extractExpressionSpan(body));
    append(StrSpan("[", 1));
    for (int32_t j = 0; j < argumentN; ++j) {
        for (int32_t c = 0; c < arguments[j].len; ++c)
            if (arguments[j].ptr[c] == '(' || arguments[j].ptr[c] == '[') return false;
        int32_t i = 0;
        while (i < n && !equalSpans(from[i], arguments[j])) ++i;
        if (i == n) return false;
        if (j > 0) append(StrSpan(",", 1));
        append(to[i]);
    }
    append(StrSpan("])", 2));
    return true;
}

/// @brief Decide whether an installed rule is demand-built and parse it.
///
/// @details
/// The formal criterion (D-596,
/// D-597). The rule is a full-bind implication
/// chain. Its premises are at most one anchor premise, the block's anchor
/// predicate over distinct bound variables (the anchor slots, each fixed to
/// the anchor argument at its position), and typing premises `(in[v,S])`, one
/// per distinct bound variable that is no slot, at least two, every set `S`
/// an anchor argument (spelled `u_<name>`, or a slot); every bound variable
/// is a slot or typed. The head is one of two shapes:
/// - **Operator totality** (no anchor premise): a positive existence `E`
///   whose compiled definition has exactly one local placeholder, the
///   witness `w`, and consists of one application of a configured operator
///   (core with input and output indices) plus at most one typing
///   `(in[w,u_Z])` of the witness with `Z` an anchor argument. In the
///   operator application, an output index holds `w`, the input indices hold
///   the typed variables (each exactly once, all of them), and every other
///   index holds an anchor argument.
/// - **OR head**: a positive compiled OR whose arguments are typed
///   variables, slots and anchor arguments, every typed variable among them.
///   Each disjunct is a flat statement (no nested OR) over the OR's
///   parameters that names every typed variable; for an `and` disjunct, so
///   does the complement of each conjunct its prepared refutation compacts
///   cite. The trigger templates are: per disjunct its complement and the
///   complements of its `and` conjuncts (`Refutation`) and the disjunct
///   itself (`CaseDemand`); the head (`Goal`).
/// Anchor arguments are compared by name without the `u_` the rule spells
/// them with. Any other rule, or a block without anchor arguments, is not
/// demand-built: the answer false is the predicate's, not a failure.
///
/// On true, @p out holds a copy of the rule and its carrier, the premise,
/// bound-variable and head slices, the slots' anchor values, and the witness
/// or trigger templates.
///
/// @param rule        The installed rule's text.
/// @param carrier     The statement whose expansion produced it.
/// @param definitions The compiled definitions (existences, ORs, operator
///                    configs).
/// @param anchorName  The anchor predicate's name.
/// @param anchorArgs  The block's anchor arguments (bare names).
/// @param anchorN     Their number.
/// @param out         Receives the parsed rule on true.
/// @return True when the rule is demand-built.
/// @invariant Pure; mints and allocates nothing.
/// @see matchDemandBuiltWitness, matchDemandBuiltTemplate, DemandBuiltRuleRegistry.
inline bool analyzeDemandBuiltRule(StrSpan rule, StrSpan carrier,
    const CompiledDefinitionRegistry& definitions, StrSpan anchorName, const StrSpan* anchorArgs,
    int32_t anchorN, DemandBuiltRule& out) {
    if (anchorN == 0 || rule.len < 4 || std::memcmp(rule.ptr, "(>[", 3) != 0) return false;
    const auto isAnchor = [&](StrSpan argument) {
        if (argument.len < 3 || argument.ptr[0] != 'u' || argument.ptr[1] != '_') return false;
        const StrSpan bare = demandBuiltArgumentCore(argument);
        for (int32_t i = 0; i < anchorN; ++i)
            if (equalSpans(anchorArgs[i], bare)) return true;
        return false;
    };
    StrSpan premises[kDemandBuiltPremiseCap];
    StrSpan bound[kDemandBuiltBoundCap];
    int32_t premiseN = 0;
    int32_t boundN = 0;
    bool fits = true;
    StrSpan head;
    ce::disintegrateImplicationSpans(rule, head, [&](StrSpan key, const StrSpan* variables, int32_t variableN) {
        if (!fits || premiseN == kDemandBuiltPremiseCap || boundN + variableN > kDemandBuiltBoundCap) {
            fits = false;
            return;
        }
        premises[premiseN++] = key;
        for (int32_t v = 0; v < variableN; ++v) bound[boundN++] = variables[v];
    });
    if (!fits) return false;
    const auto boundIndex = [&](StrSpan token) -> int32_t {
        for (int32_t b = 0; b < boundN; ++b)
            if (equalSpans(bound[b], token)) return b;
        return -1;
    };

    // The anchor premise: at most one, the anchor predicate over distinct
    // bound variables, each fixed to the anchor argument at its position.
    int32_t anchorPremise = -1;
    bool isSlot[kDemandBuiltBoundCap] = {};
    StrSpan slotValue[kDemandBuiltBoundCap];
    for (int32_t p = 0; p < premiseN; ++p) {
        if (premises[p].len < 3 || premises[p].ptr[0] != '('
            || !equalSpans(extractExpressionSpan(premises[p]), anchorName)) continue;
        if (anchorPremise >= 0) return false;
        StrSpan arguments[ExecutionParameters::MAX_ARITY];
        if (getArgsSpans(premises[p], arguments, ExecutionParameters::MAX_ARITY) != anchorN) return false;
        for (int32_t i = 0; i < anchorN; ++i) {
            const int32_t b = boundIndex(arguments[i]);
            if (b < 0 || isSlot[b]) return false;
            isSlot[b] = true;
            slotValue[b] = anchorArgs[i];
        }
        anchorPremise = p;
    }

    // Typing premises: (in[v,S]), v a bound variable typed once and no slot,
    // S an anchor argument or a slot.
    StrSpan typed[kDemandBuiltPremiseCap];
    int32_t typedN = 0;
    for (int32_t p = 0; p < premiseN; ++p) {
        if (p == anchorPremise) continue;
        if (premises[p].len < 5 || std::memcmp(premises[p].ptr, "(in[", 4) != 0) return false;
        StrSpan arguments[ExecutionParameters::MAX_ARITY];
        if (getArgsSpans(premises[p], arguments, ExecutionParameters::MAX_ARITY) != 2) return false;
        const int32_t b = boundIndex(arguments[0]);
        if (b < 0 || isSlot[b]) return false;
        for (int32_t q = 0; q < typedN; ++q)
            if (equalSpans(typed[q], arguments[0])) return false;
        const int32_t set = boundIndex(arguments[1]);
        if (set >= 0 ? !isSlot[set] : !isAnchor(arguments[1])) return false;
        typed[typedN++] = arguments[0];
    }
    int32_t slotN = 0;
    for (int32_t b = 0; b < boundN; ++b) slotN += isSlot[b] ? 1 : 0;
    if (typedN < 2 || slotN + typedN != boundN) return false;

    // Head: a positive existence or OR of the definitions.
    if (head.len < 3 || head.ptr[0] != '(') return false;
    const LogicalEntity* entity = definitions.compiledEntity(extractExpressionSpan(head));
    if (entity == nullptr) return false;
    StrSpan parameters[ExecutionParameters::MAX_ARITY];
    StrSpan headArguments[ExecutionParameters::MAX_ARITY];
    const StrSpan signature(entity->signature.data(), static_cast<int32_t>(entity->signature.size()));
    const int32_t parameterN = getArgsSpans(signature, parameters, ExecutionParameters::MAX_ARITY);
    if (getArgsSpans(head, headArguments, ExecutionParameters::MAX_ARITY) != parameterN) return false;
    const auto sliceOf = [&](StrSpan part) {
        return DemandBuiltSlice{ static_cast<int32_t>(part.ptr - rule.ptr), part.len };
    };
    // The parts every head shape copies: rule, carrier, premises, bound
    // variables with the slots' anchor values, head.
    const auto copyCommon = [&](DemandBuiltKind kind) {
        assert(rule.len < kDemandBuiltTextCap && carrier.len < kDemandBuiltTextCap
            && "analyzeDemandBuiltRule: rule or carrier exceeds its capacity");
        out.kind = kind;
        std::memcpy(out.text, rule.ptr, static_cast<std::size_t>(rule.len));
        out.textLength = rule.len;
        std::memcpy(out.carrier, carrier.ptr, static_cast<std::size_t>(carrier.len));
        out.carrierLength = carrier.len;
        out.premiseCount = premiseN;
        for (int32_t p = 0; p < premiseN; ++p) out.premises[p] = sliceOf(premises[p]);
        out.boundCount = boundN;
        out.anchorValuesLength = 0;
        for (int32_t b = 0; b < boundN; ++b) {
            out.bound[b] = sliceOf(bound[b]);
            out.fixedValue[b] = DemandBuiltSlice{ out.anchorValuesLength, 0 };
            if (!isSlot[b]) continue;
            assert(out.anchorValuesLength + slotValue[b].len < kDemandBuiltTextCap
                && "analyzeDemandBuiltRule: anchor values exceed their capacity");
            std::memcpy(out.anchorValues + out.anchorValuesLength, slotValue[b].ptr,
                static_cast<std::size_t>(slotValue[b].len));
            out.fixedValue[b].length = slotValue[b].len;
            out.anchorValuesLength += slotValue[b].len;
        }
        out.head = sliceOf(head);
        out.witnessLength = 0;
        out.triggersLength = 0;
        out.triggerCount = 0;
    };

    if (entity->category == "or") {
        // Every head argument is a typed variable, a slot or an anchor
        // argument, and every typed variable occurs.
        for (int32_t j = 0; j < parameterN; ++j)
            if (boundIndex(headArguments[j]) < 0 && !isAnchor(headArguments[j])) return false;
        for (int32_t q = 0; q < typedN; ++q) {
            bool occurs = false;
            for (int32_t j = 0; j < parameterN; ++j) occurs = occurs || equalSpans(headArguments[j], typed[q]);
            if (!occurs) return false;
        }
        const auto namesEveryTyped = [&](StrSpan text) {
            const StrSpan body = text.len > 0 && text.ptr[0] == '!' ? StrSpan(text.ptr + 1, text.len - 1) : text;
            StrSpan arguments[ExecutionParameters::MAX_ARITY];
            const int32_t argumentN = getArgsSpans(body, arguments, ExecutionParameters::MAX_ARITY);
            for (int32_t q = 0; q < typedN; ++q) {
                bool named = false;
                for (int32_t j = 0; j < argumentN; ++j) named = named || equalSpans(arguments[j], typed[q]);
                if (!named) return false;
            }
            return true;
        };
        char templates[kDemandBuiltTriggerTextCap];
        int32_t templatesLength = 0;
        DemandBuiltSlice slices[kDemandBuiltTriggerCap];
        DemandBuiltTrigger kinds[kDemandBuiltTriggerCap];
        int32_t templateN = 0;
        const auto addTemplate = [&](StrSpan text, DemandBuiltTrigger kind) {
            for (int32_t t = 0; t < templateN; ++t)
                if (kinds[t] == kind
                    && equalSpans(StrSpan(templates + slices[t].offset, slices[t].length), text)) return;
            assert(templateN < kDemandBuiltTriggerCap && templatesLength + text.len < kDemandBuiltTriggerTextCap
                && "analyzeDemandBuiltRule: trigger templates exceed their capacity");
            std::memcpy(templates + templatesLength, text.ptr, static_cast<std::size_t>(text.len));
            slices[templateN] = DemandBuiltSlice{ templatesLength, text.len };
            kinds[templateN++] = kind;
            templatesLength += text.len;
        };
        for (const std::string& elementText : entity->elements) {
            const StrSpan element(elementText.data(), static_cast<int32_t>(elementText.size()));
            const bool negatedLeaf = element.len > 0 && element.ptr[0] == '!';
            const StrSpan leafBody = negatedLeaf ? StrSpan(element.ptr + 1, element.len - 1) : element;
            if (leafBody.len < 3 || leafBody.ptr[0] != '(') return false;
            const LogicalEntity* leafEntity = definitions.compiledEntity(extractExpressionSpan(leafBody));
            if (leafEntity != nullptr && leafEntity->category == "or") return false;
            char leaf[kDemandBuiltTextCap];
            int32_t leafLength = 0;
            if (!demandBuiltRename(element, parameters, headArguments, parameterN, false,
                                   leaf, kDemandBuiltTextCap, leafLength)
                || !namesEveryTyped(StrSpan(leaf, leafLength))) return false;
            char complement[kDemandBuiltTextCap];
            int32_t complementLength = 0;
            const bool renamed = demandBuiltRename(StrSpan(leaf, leafLength), headArguments, headArguments,
                parameterN, true, complement, kDemandBuiltTextCap, complementLength);
            assert(renamed && "analyzeDemandBuiltRule: a disjunct renames onto its own arguments");
            addTemplate(StrSpan(complement, complementLength), DemandBuiltTrigger::Refutation);
            addTemplate(StrSpan(leaf, leafLength), DemandBuiltTrigger::CaseDemand);
            if (negatedLeaf || leafEntity == nullptr || leafEntity->category != "and") continue;
            // The refutation compacts of an and disjunct refute it through
            // each conjunct's complement.
            if (!leafEntity->conjunctionRefutationsPrepared) return false;
            StrSpan andParameters[ExecutionParameters::MAX_ARITY];
            StrSpan leafArguments[ExecutionParameters::MAX_ARITY];
            const StrSpan andSignature(leafEntity->signature.data(),
                static_cast<int32_t>(leafEntity->signature.size()));
            const int32_t andN = getArgsSpans(andSignature, andParameters, ExecutionParameters::MAX_ARITY);
            if (getArgsSpans(StrSpan(leaf, leafLength), leafArguments, ExecutionParameters::MAX_ARITY) != andN)
                return false;
            for (const auto& row : leafEntity->conjunctionRefutations) {
                const StrSpan conjunct(row.first.data(), static_cast<int32_t>(row.first.size()));
                char refuter[kDemandBuiltTextCap];
                int32_t refuterLength = 0;
                if (!demandBuiltRename(conjunct, andParameters, leafArguments, andN, true,
                                       refuter, kDemandBuiltTextCap, refuterLength)
                    || !namesEveryTyped(StrSpan(refuter, refuterLength))) return false;
                addTemplate(StrSpan(refuter, refuterLength), DemandBuiltTrigger::Refutation);
            }
        }
        addTemplate(head, DemandBuiltTrigger::Goal);
        copyCommon(DemandBuiltKind::OrHead);
        std::memcpy(out.triggers, templates, static_cast<std::size_t>(templatesLength));
        out.triggersLength = templatesLength;
        out.triggerCount = templateN;
        for (int32_t t = 0; t < templateN; ++t) {
            out.triggerSlices[t] = slices[t];
            out.triggerKinds[t] = kinds[t];
        }
        return true;
    }

    // Operator totality: no anchor premise; a positive existence.
    if (entity->category != "existence" || anchorPremise >= 0) return false;
    // A definition token: a parameter maps to the head's argument at its
    // position; any other token is the definition's local witness.
    const auto mapped = [&](StrSpan token, bool& local) -> StrSpan {
        for (int32_t i = 0; i < parameterN; ++i)
            if (equalSpans(parameters[i], token)) {
                local = false;
                return headArguments[i];
            }
        local = true;
        return token;
    };

    StrSpan witnessName;
    int32_t operatorElement = -1;
    int32_t typingElements = 0;
    const int32_t elementN = static_cast<int32_t>(entity->elements.size());
    for (int32_t e = 0; e < elementN; ++e) {
        const std::string& elementText = entity->elements[static_cast<std::size_t>(e)];
        const StrSpan element(elementText.data(), static_cast<int32_t>(elementText.size()));
        if (element.len < 3 || element.ptr[0] != '(') return false;
        StrSpan arguments[ExecutionParameters::MAX_ARITY];
        const int32_t argumentN = getArgsSpans(element, arguments, ExecutionParameters::MAX_ARITY);
        for (int32_t a = 0; a < argumentN; ++a) {
            bool local = false;
            mapped(arguments[a], local);
            if (!local) continue;
            if (witnessName.len == 0) witnessName = arguments[a];
            else if (!equalSpans(witnessName, arguments[a])) return false;
        }
        const StrSpan core = extractExpressionSpan(element);
        bool local = false;
        if (equalSpans(core, StrSpan("in", 2)) && argumentN == 2
            && witnessName.len > 0 && equalSpans(arguments[0], witnessName)) {
            const StrSpan set = mapped(arguments[1], local);
            if (local || !isAnchor(set) || ++typingElements > 1) return false;
            continue;
        }
        if (definitions.operatorNames().find(std::string_view(core.ptr, static_cast<std::size_t>(core.len)))
                == definitions.operatorNames().end()
            || definitions.coreConfig(core) == nullptr || operatorElement >= 0)
            return false;
        operatorElement = e;
    }
    if (operatorElement < 0 || witnessName.len == 0) return false;

    // The operator application: witness at an output, the bound variables at
    // the inputs, anchor arguments everywhere else.
    const std::string& operatorText = entity->elements[static_cast<std::size_t>(operatorElement)];
    const StrSpan operation(operatorText.data(), static_cast<int32_t>(operatorText.size()));
    const StrSpan operatorCore = extractExpressionSpan(operation);
    const ce::CoreExpressionConfig* config = definitions.coreConfig(operatorCore);
    StrSpan operands[ExecutionParameters::MAX_ARITY];
    const int32_t operandN = getArgsSpans(operation, operands, ExecutionParameters::MAX_ARITY);
    if (static_cast<int32_t>(config->inputIndices.size()) != premiseN) return false;
    bool used[kDemandBuiltPremiseCap] = {};
    bool witnessAtOutput = false;
    char witness[kDemandBuiltTextCap];
    int32_t witnessLength = 0;
    const auto append = [&](StrSpan piece) {
        assert(witnessLength + piece.len < kDemandBuiltTextCap
            && "analyzeDemandBuiltRule: witness template exceeds its capacity");
        std::memcpy(witness + witnessLength, piece.ptr, static_cast<std::size_t>(piece.len));
        witnessLength += piece.len;
    };
    append(StrSpan("(", 1));
    append(operatorCore);
    append(StrSpan("[", 1));
    for (int32_t j = 0; j < operandN; ++j) {
        bool local = false;
        const StrSpan value = mapped(operands[j], local);
        const auto inList = [&](const std::vector<int>& indices) {
            for (const int index : indices) if (index == j) return true;
            return false;
        };
        if (j > 0) append(StrSpan(",", 1));
        if (inList(config->outputIndices)) {
            if (!local) return false;
            witnessAtOutput = true;
            append(StrSpan("marker", 6));
        } else if (inList(config->inputIndices)) {
            if (local) return false;
            int32_t b = -1;
            for (int32_t q = 0; q < premiseN; ++q) if (equalSpans(typed[q], value)) b = q;
            if (b < 0 || used[b]) return false;
            used[b] = true;
            append(value);
        } else {
            if (local || !isAnchor(value)) return false;
            append(value);
        }
    }
    append(StrSpan("])", 2));
    if (!witnessAtOutput) return false;

    copyCommon(DemandBuiltKind::OperatorWitness);
    std::memcpy(out.witness, witness, static_cast<std::size_t>(witnessLength));
    out.witnessLength = witnessLength;
    // The operator totality lists its bound variables in premise order.
    for (int32_t p = 0; p < premiseN; ++p) out.bound[p] = sliceOf(typed[p]);
    return true;
}

/// @brief Match a demand key against a demand-built rule's witness template.
///
/// @details
/// The key `(op[...])` matches when it is the same positive application with
/// the same arity, carries `marker` exactly where the template does, gives
/// every bound variable one concrete value (the same at every occurrence, not
/// a `u_` template argument), and equals the template's anchor arguments by
/// bare name. @p values receives each premise's bound value in premise order.
/// Only an operator-totality rule has a witness template.
///
/// @param rule   The demand-built rule.
/// @param key    The demand key in bare marker form.
/// @param values Receives `rule.boundCount` spans aliasing @p key.
/// @return True on a match.
/// @invariant Pure; mints and allocates nothing.
/// @see analyzeDemandBuiltRule, instantiateDemandBuiltPart.
inline bool matchDemandBuiltWitness(const DemandBuiltRule& rule, StrSpan key, StrSpan* values) {
    if (rule.kind != DemandBuiltKind::OperatorWitness) return false;
    const StrSpan witness = rule.witnessSpan();
    if (key.len < 3 || key.ptr[0] != '(') return false;
    if (!equalSpans(extractExpressionSpan(key), extractExpressionSpan(witness))) return false;
    StrSpan keyArguments[ExecutionParameters::MAX_ARITY];
    StrSpan templateArguments[ExecutionParameters::MAX_ARITY];
    const int32_t n = getArgsSpans(witness, templateArguments, ExecutionParameters::MAX_ARITY);
    if (getArgsSpans(key, keyArguments, ExecutionParameters::MAX_ARITY) != n) return false;
    bool found[kDemandBuiltBoundCap] = {};
    const StrSpan marker("marker", 6);
    for (int32_t j = 0; j < n; ++j) {
        const bool templateMarker = equalSpans(templateArguments[j], marker);
        if (templateMarker != equalSpans(keyArguments[j], marker)) return false;
        if (templateMarker) continue;
        int32_t b = -1;
        for (int32_t q = 0; q < rule.boundCount; ++q)
            if (equalSpans(rule.sliceSpan(rule.bound[q]), templateArguments[j])) b = q;
        if (b < 0) {
            if (!equalSpans(demandBuiltArgumentCore(templateArguments[j]),
                            demandBuiltArgumentCore(keyArguments[j]))) return false;
            continue;
        }
        if (keyArguments[j].len >= 2 && keyArguments[j].ptr[0] == 'u' && keyArguments[j].ptr[1] == '_')
            return false;
        if (found[b] && !equalSpans(values[b], keyArguments[j])) return false;
        values[b] = keyArguments[j];
        found[b] = true;
    }
    for (int32_t q = 0; q < rule.boundCount; ++q)
        assert(found[q] && "matchDemandBuiltWitness: the template binds every premise variable");
    return true;
}

/// @brief Match a statement against one trigger template of an OR-headed
///        demand-built rule.
///
/// @details
/// The statement matches when it has the template's polarity, core and
/// arity, holds each slot's anchor value and each anchor argument (by bare
/// name) where the template has them, and gives every typed variable one
/// concrete value (the same at every occurrence, not a `u_` template
/// argument). @p values receives the value of every bound variable: the
/// slots their anchor values, the typed variables the matched arguments.
///
/// @param rule      The OR-headed demand-built rule.
/// @param trigger   Position of the template below `rule.triggerCount`.
/// @param statement The statement text (a fact, a demand key or a goal).
/// @param values    Receives `rule.boundCount` spans aliasing @p statement
///                  or the rule.
/// @return True on a match.
/// @invariant Pure; mints and allocates nothing.
/// @see analyzeDemandBuiltRule, instantiateDemandBuiltPart.
inline bool matchDemandBuiltTemplate(const DemandBuiltRule& rule, int32_t trigger, StrSpan statement,
    StrSpan* values) {
    assert(rule.kind == DemandBuiltKind::OrHead && trigger >= 0 && trigger < rule.triggerCount
        && "matchDemandBuiltTemplate: a trigger of an OR-headed rule");
    StrSpan templateText = rule.triggerSpan(trigger);
    const bool templateNegated = templateText.ptr[0] == '!';
    const bool statementNegated = statement.len > 0 && statement.ptr[0] == '!';
    if (templateNegated != statementNegated) return false;
    if (templateNegated) templateText = StrSpan(templateText.ptr + 1, templateText.len - 1);
    if (statementNegated) statement = StrSpan(statement.ptr + 1, statement.len - 1);
    if (statement.len < 3 || statement.ptr[0] != '(') return false;
    if (!equalSpans(extractExpressionSpan(statement), extractExpressionSpan(templateText))) return false;
    StrSpan statementArguments[ExecutionParameters::MAX_ARITY];
    StrSpan templateArguments[ExecutionParameters::MAX_ARITY];
    const int32_t n = getArgsSpans(templateText, templateArguments, ExecutionParameters::MAX_ARITY);
    if (getArgsSpans(statement, statementArguments, ExecutionParameters::MAX_ARITY) != n) return false;
    bool found[kDemandBuiltBoundCap] = {};
    for (int32_t b = 0; b < rule.boundCount; ++b)
        if (rule.fixedValue[b].length > 0) {
            values[b] = rule.fixedSpan(b);
            found[b] = true;
        }
    for (int32_t j = 0; j < n; ++j) {
        int32_t b = -1;
        for (int32_t q = 0; q < rule.boundCount; ++q)
            if (equalSpans(rule.sliceSpan(rule.bound[q]), templateArguments[j])) b = q;
        if (b < 0) {
            if (!equalSpans(demandBuiltArgumentCore(templateArguments[j]),
                            demandBuiltArgumentCore(statementArguments[j]))) return false;
            continue;
        }
        if (rule.fixedValue[b].length > 0) {
            if (!equalSpans(rule.fixedSpan(b), demandBuiltArgumentCore(statementArguments[j]))) return false;
            continue;
        }
        if (statementArguments[j].len >= 2 && statementArguments[j].ptr[0] == 'u'
            && statementArguments[j].ptr[1] == '_') return false;
        if (found[b] && !equalSpans(values[b], statementArguments[j])) return false;
        values[b] = statementArguments[j];
        found[b] = true;
    }
    for (int32_t q = 0; q < rule.boundCount; ++q)
        assert(found[q] && "matchDemandBuiltTemplate: the template names every typed variable");
    return true;
}

/// @brief Instantiate one flat part of a demand-built rule.
///
/// @details
/// Writes @p part (a premise or the head) with every bound variable replaced
/// by its value and every other argument without its `u_`: the statement
/// text the rule's firing on those values would produce.
///
/// @param rule   The demand-built rule.
/// @param part   A slice of the rule (a premise or the head).
/// @param values Each bound variable's value, in bound-variable order.
/// @param out    Receives the text.
/// @param cap    Capacity of @p out (asserted).
/// @return The text's length.
/// @invariant Pure; mints and allocates nothing.
/// @see matchDemandBuiltWitness, matchDemandBuiltTemplate.
inline int32_t instantiateDemandBuiltPart(const DemandBuiltRule& rule, DemandBuiltSlice part,
    const StrSpan* values, char* out, int32_t cap) {
    const StrSpan text = rule.sliceSpan(part);
    const StrSpan core = extractExpressionSpan(text);
    StrSpan arguments[ExecutionParameters::MAX_ARITY];
    const int32_t n = getArgsSpans(text, arguments, ExecutionParameters::MAX_ARITY);
    int32_t length = 0;
    const auto append = [&](StrSpan piece) {
        assert(length + piece.len < cap && "instantiateDemandBuiltPart: text exceeds its capacity");
        std::memcpy(out + length, piece.ptr, static_cast<std::size_t>(piece.len));
        length += piece.len;
    };
    append(StrSpan("(", 1));
    append(core);
    append(StrSpan("[", 1));
    for (int32_t j = 0; j < n; ++j) {
        if (j > 0) append(StrSpan(",", 1));
        int32_t b = -1;
        for (int32_t q = 0; q < rule.boundCount; ++q)
            if (equalSpans(rule.sliceSpan(rule.bound[q]), arguments[j])) b = q;
        append(b >= 0 ? values[b] : demandBuiltArgumentCore(arguments[j]));
    }
    append(StrSpan("])", 2));
    return length;
}

/// @brief Whether two flat statements share polarity, core and arity.
///
/// @details
/// The cheap shape test a statement passes before a template match binds
/// its arguments.
///
/// @param left  One flat statement, optionally negated.
/// @param right The other.
/// @return True when polarity, core and argument count are equal.
/// @invariant Pure; mints and allocates nothing.
/// @see demandBuiltHasTrigger.
inline bool demandBuiltSameShape(StrSpan left, StrSpan right) {
    const bool leftNegated = left.len > 0 && left.ptr[0] == '!';
    const bool rightNegated = right.len > 0 && right.ptr[0] == '!';
    if (leftNegated != rightNegated) return false;
    if (leftNegated) {
        left = StrSpan(left.ptr + 1, left.len - 1);
        right = StrSpan(right.ptr + 1, right.len - 1);
    }
    if (left.len < 3 || right.len < 3 || left.ptr[0] != '(' || right.ptr[0] != '(') return false;
    if (!equalSpans(extractExpressionSpan(left), extractExpressionSpan(right))) return false;
    StrSpan leftArguments[ExecutionParameters::MAX_ARITY];
    StrSpan rightArguments[ExecutionParameters::MAX_ARITY];
    return getArgsSpans(left, leftArguments, ExecutionParameters::MAX_ARITY)
        == getArgsSpans(right, rightArguments, ExecutionParameters::MAX_ARITY);
}

/// @brief Whether a statement has the shape of one of an OR-headed rule's
///        trigger templates of one kind.
///
/// @details
/// A shape test only (`demandBuiltSameShape`): the statement may still fail
/// the template match on its anchor arguments. An operator-totality rule has
/// no trigger templates.
///
/// @param rule      The demand-built rule.
/// @param statement The statement text.
/// @param kind      The trigger kind.
/// @return True when some template of @p kind has the statement's shape.
/// @invariant Pure; mints and allocates nothing.
/// @see matchDemandBuiltTemplate, DemandBuiltRuleRegistry::hasOrTrigger.
inline bool demandBuiltHasTrigger(const DemandBuiltRule& rule, StrSpan statement, DemandBuiltTrigger kind) {
    if (rule.kind != DemandBuiltKind::OrHead) return false;
    for (int32_t t = 0; t < rule.triggerCount; ++t)
        if (rule.triggerKinds[t] == kind && demandBuiltSameShape(rule.triggerSpan(t), statement)) return true;
    return false;
}

namespace tests { struct DemandBuiltRuleRegistryFixture; }

/// @brief The process-wide registry of demand-built rules.
///
/// @details
/// Demand-built rules are anchor-level: every block of a batch installs the
/// same texts, and one batch runs per process. The registry holds each text
/// once (the first install's record, its carrier included); which block
/// holds a rule, and at which scope, stays in the block's installed-rule
/// index. Append-only: a record is complete before the count publishes it,
/// so readers on other worker threads read without a lock. Callers that
/// match several records order them by rule text themselves.
class DemandBuiltRuleRegistry {
friend struct tests::DemandBuiltRuleRegistryFixture;
public:
    /// @brief The registry of this process.
    /// @return The singleton.
    static DemandBuiltRuleRegistry& instance() {
        static DemandBuiltRuleRegistry registry;
        return registry;
    }

    /// @brief Register a parsed rule, once per rule text.
    /// @param rule The parsed rule (`analyzeDemandBuiltRule`).
    /// @return Nothing.
    /// @invariant Capacity `kDemandBuiltRuleCap` (asserted).
    void add(const DemandBuiltRule& rule) {
        std::lock_guard<std::mutex> lock(mutex_);
        const int32_t n = count_.load(std::memory_order_relaxed);
        for (int32_t i = 0; i < n; ++i)
            if (equalSpans(rules_[i].textSpan(), rule.textSpan())) return;
        assert(n < kDemandBuiltRuleCap && "DemandBuiltRuleRegistry: rule capacity exceeded");
        rules_[n] = rule;
        count_.store(n + 1, std::memory_order_release);
        if (rule.kind == DemandBuiltKind::OrHead) anyOr_.store(true, std::memory_order_release);
    }

    /// @brief Number of registered rules.
    /// @return The published count.
    int32_t count() const { return count_.load(std::memory_order_acquire); }

    /// @brief Whether an OR-headed rule is registered.
    /// @return True once the first OR-headed rule is published.
    bool hasOrRules() const { return anyOr_.load(std::memory_order_acquire); }

    /// @brief Whether a statement has the shape of a registered OR-headed
    ///        rule's trigger template of one kind (`demandBuiltHasTrigger`).
    /// @param statement The statement text.
    /// @param kind      The trigger kind.
    /// @return True when some registered OR head has such a template.
    /// @invariant Pure; reads the published rules only.
    bool hasOrTrigger(StrSpan statement, DemandBuiltTrigger kind) const {
        if (!hasOrRules()) return false;
        const int32_t n = count();
        for (int32_t i = 0; i < n; ++i)
            if (demandBuiltHasTrigger(rules_[i], statement, kind)) return true;
        return false;
    }

    /// @brief The registered OR-headed rules in rule-text order.
    /// @details The deterministic order every OR build visits them in.
    /// @param order Receives the rules' registry positions; capacity
    ///              `kDemandBuiltRuleCap`.
    /// @return The number of OR-headed rules written.
    /// @invariant Pure; reads the published rules only.
    int32_t orderedOrRules(int32_t* order) const {
        const int32_t n = count();
        int32_t orN = 0;
        for (int32_t i = 0; i < n; ++i)
            if (rules_[i].kind == DemandBuiltKind::OrHead) order[orN++] = i;
        std::sort(order, order + orN, [&](int32_t left, int32_t right) {
            return compareSpans(rules_[left].textSpan(), rules_[right].textSpan()) < 0;
        });
        return orN;
    }

    /// @brief The registered rule with a rule text.
    /// @param text The rule text as installed.
    /// @return The record, or nullptr when no registered rule has the text —
    ///         a defined lookup miss.
    /// @invariant Pure; reads the published rules only.
    const DemandBuiltRule* find(StrSpan text) const {
        const int32_t n = count();
        for (int32_t i = 0; i < n; ++i)
            if (equalSpans(rules_[i].textSpan(), text)) return &rules_[i];
        return nullptr;
    }

    /// @brief One registered rule.
    /// @param index Position below `count()` (asserted).
    /// @return The record.
    const DemandBuiltRule& at(int32_t index) const {
        assert(index >= 0 && index < count() && "DemandBuiltRuleRegistry: index out of range");
        return rules_[index];
    }

    /// @brief Whether a premise can match an instance of a registered rule's
    ///        head.
    ///
    /// @details
    /// A positive premise with a registered head's core and arity matches the
    /// head's instances when, at every anchor position of the head (an
    /// anchor argument or an anchor slot), it holds the same anchor argument
    /// by bare name or a variable of its own rule (an argument without `u_`:
    /// a rule text spells its fixed names `u_`). A typed position of the head
    /// takes any value.
    ///
    /// @param premise One premise of a rule being installed, in rule spelling.
    /// @return True when the premise could be served by a demand-built head.
    /// @invariant Pure; reads the published rules only.
    bool citesHead(StrSpan premise) const {
        if (premise.len < 3 || premise.ptr[0] != '(') return false;
        const int32_t n = count();
        for (int32_t i = 0; i < n; ++i) {
            const DemandBuiltRule& rule = rules_[i];
            const StrSpan head = rule.sliceSpan(rule.head);
            if (!equalSpans(extractExpressionSpan(head), extractExpressionSpan(premise))) continue;
            StrSpan headArguments[ExecutionParameters::MAX_ARITY];
            StrSpan premiseArguments[ExecutionParameters::MAX_ARITY];
            const int32_t argumentN = getArgsSpans(head, headArguments, ExecutionParameters::MAX_ARITY);
            if (getArgsSpans(premise, premiseArguments, ExecutionParameters::MAX_ARITY) != argumentN) continue;
            bool cites = true;
            for (int32_t j = 0; j < argumentN && cites; ++j) {
                int32_t b = -1;
                for (int32_t q = 0; q < rule.boundCount; ++q)
                    if (equalSpans(rule.sliceSpan(rule.bound[q]), headArguments[j])) b = q;
                const StrSpan argument = premiseArguments[j];
                const bool variable = argument.len < 2 || argument.ptr[0] != 'u' || argument.ptr[1] != '_';
                const StrSpan anchor = b >= 0 ? rule.fixedSpan(b) : demandBuiltArgumentCore(headArguments[j]);
                cites = (b >= 0 && rule.fixedValue[b].length == 0) || variable
                    || equalSpans(anchor, demandBuiltArgumentCore(argument));
            }
            if (cites) return true;
        }
        return false;
    }

private:
    std::mutex mutex_;
    std::atomic<int32_t> count_{0};
    std::atomic<bool> anyOr_{false};
    DemandBuiltRule rules_[kDemandBuiltRuleCap];
};

}
