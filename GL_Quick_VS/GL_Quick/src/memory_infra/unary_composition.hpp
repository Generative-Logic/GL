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

#include "scratch_string.hpp"
#include "str_ops.hpp"
#include "../parameters.hpp"
#include <cstdio>

namespace gl {

    /// @brief Store an ordered premise run as one interner-ready string.
    /// @details Newlines separate flat signed atoms and cannot occur in MPL.
    /// Unary runs retain their original atom spelling.
    /// @param arena String scratch owning the result.
    /// @param premises Ordered nonempty premise run.
    /// @param count Premise count within the existing chain capacity.
    /// @return Complete ordered premise-chain bytes.
    inline ScratchString joinCompositionPremises(ScratchArena& arena,
        const StrSpan* premises, int count) {
        assert(count > 0 && count < ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS);
        int length = count - 1;
        for (int i = 0; i < count; ++i) length += premises[i].len;
        char* buffer = arena.allocBytes(length);
        int position = 0;
        for (int i = 0; i < count; ++i) {
            if (i) buffer[position++] = '\n';
            std::memcpy(buffer + position, premises[i].ptr, premises[i].len);
            position += premises[i].len;
        }
        assert(position == length);
        return ScratchString::wrap(arena, buffer, length);
    }

    /// @brief Normalize every bound name in a complete producer chain.
    /// @details First occurrence follows ordered premises then head. Fixed
    /// parameters and repeated argument identities retain their exact meaning.
    /// @param arena String scratch owning the outputs.
    /// @param premises Complete ordered input premises.
    /// @param count Number of premises within the existing chain limit.
    /// @param sourceHead Input head.
    /// @param chain Receives newline-separated normalized premises.
    /// @param head Receives the normalized head.
    /// @return Nothing.
    inline void normalizeCompositionChain(ScratchArena& arena, const StrSpan* premises,
        int count, StrSpan sourceHead, ScratchString& chain, ScratchString& head) {
        constexpr int capacity = ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS * ExecutionParameters::MAX_ARITY;
        assert(count > 0 && count < ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS);
        StrReplacement mapping[capacity];
        char numbers[capacity][12];
        int size = 0;
        for (int atom = 0; atom <= count; ++atom) {
            StrSpan arguments[ExecutionParameters::MAX_ARITY];
            const int arity = getArgsSpans(atom == count ? sourceHead : premises[atom],
                arguments, ExecutionParameters::MAX_ARITY);
            for (int i = 0; i < arity; ++i) {
                if (startsWithSpan(arguments[i], "u_", 2)) continue;
                int found = 0;
                while (found < size && !equalSpans(mapping[found].key, arguments[i])) ++found;
                if (found < size) continue;
                assert(size < capacity);
                const int length = std::snprintf(numbers[size], sizeof(numbers[size]), "%d", size + 1);
                assert(length > 0 && length < static_cast<int>(sizeof(numbers[size])));
                mapping[size] = {arguments[i], StrSpan(numbers[size], length)};
                ++size;
            }
        }
        StrSpan normalized[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
        for (int i = 0; i < count; ++i)
            normalized[i] = replaceKeysScratch(arena, premises[i], mapping, size);
        chain = joinCompositionPremises(arena, normalized, count);
        head = replaceKeysScratch(arena, sourceHead, mapping, size);
    }

    /// @brief The two defined outcomes of a unary implication match.
    /// @details Incompatibility is an ordinary logical result, not an error.
    enum class UnaryCompositionMatch { incompatible, matched };

    /// @brief Compose a complete producer chain with one unary consumer.
    /// @details Intermediate ports must bijectively cover both complete source
    /// binder sets; fixed parameters cannot specialize a binder. Every producer
    /// premise survives in order and every binder occurs in the resulting chain.
    /// @param arena Caller-owned string scratch for successful outputs.
    /// @param premises Complete producer premise run.
    /// @param count Producer premise count.
    /// @param firstHead Producer terminal head.
    /// @param secondPremise Consumer's sole premise.
    /// @param secondHead Consumer terminal head.
    /// @param chain Receives the normalized result premise chain on a match.
    /// @param head Receives the normalized result head on a match.
    /// @return Matched or logically incompatible.
    inline UnaryCompositionMatch composeImplicationChain(ScratchArena& arena,
        const StrSpan* premises, int count, StrSpan firstHead, StrSpan secondPremise,
        StrSpan secondHead, ScratchString& chain, ScratchString& head) {
        assert(count > 0 && count < ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS);
        if ((firstHead.ptr[0] == '!') != (secondPremise.ptr[0] == '!')
            || !equalSpans(extractExpressionUniversalSpan(firstHead), extractExpressionUniversalSpan(secondPremise)))
            return UnaryCompositionMatch::incompatible;
        StrSpan left[ExecutionParameters::MAX_ARITY], right[ExecutionParameters::MAX_ARITY];
        const int arity = getArgsSpans(firstHead, left, ExecutionParameters::MAX_ARITY);
        if (arity != getArgsSpans(secondPremise, right, ExecutionParameters::MAX_ARITY))
            return UnaryCompositionMatch::incompatible;
        StrReplacement mapping[ExecutionParameters::MAX_ARITY];
        int mappingCount = 0;
        for (int i = 0; i < arity; ++i) {
            const bool leftFixed = startsWithSpan(left[i], "u_", 2);
            const bool rightFixed = startsWithSpan(right[i], "u_", 2);
            if (leftFixed || rightFixed) {
                if (!(leftFixed && rightFixed && equalSpans(left[i], right[i])))
                    return UnaryCompositionMatch::incompatible;
                continue;
            }
            bool seen = false;
            for (int j = 0; j < mappingCount; ++j) {
                const bool sameLeft = equalSpans(mapping[j].value, left[i]);
                const bool sameRight = equalSpans(mapping[j].key, right[i]);
                if (sameLeft != sameRight) return UnaryCompositionMatch::incompatible;
                seen |= sameLeft;
            }
            if (!seen) mapping[mappingCount++] = {right[i], left[i]};
        }
        bool survives[ExecutionParameters::MAX_ARITY]{};
        for (int atom = 0; atom <= count; ++atom) {
            StrSpan arguments[ExecutionParameters::MAX_ARITY];
            const int n = getArgsSpans(atom == count ? secondHead : premises[atom],
                arguments, ExecutionParameters::MAX_ARITY);
            for (int i = 0; i < n; ++i) {
                if (startsWithSpan(arguments[i], "u_", 2)) continue;
                int match = 0;
                while (match < mappingCount && !equalSpans(arguments[i],
                    atom == count ? mapping[match].key : mapping[match].value)) ++match;
                if (match == mappingCount) return UnaryCompositionMatch::incompatible;
                survives[match] = true;
            }
        }
        for (int i = 0; i < mappingCount; ++i)
            if (!survives[i]) return UnaryCompositionMatch::incompatible;
        const ScratchString translatedHead = replaceKeysScratch(arena, secondHead, mapping, mappingCount);
        normalizeCompositionChain(arena, premises, count, translatedHead, chain, head);
        return UnaryCompositionMatch::matched;
    }

    /// @brief Normalize one unary rule's bound names without composing it.
    /// @details Both endpoints share a first-occurrence renaming. Fixed u_
    /// parameters, repeated arguments and polarity remain unchanged. A variable
    /// may occur in only one endpoint; registration imposes no composition gate.
    /// @param arena Caller-owned scratch that owns both returned spans.
    /// @param sourcePremise Original registered premise.
    /// @param sourceHead Original registered head.
    /// @param premise Receives the normalized premise.
    /// @param head Receives the normalized head.
    /// @return Nothing.
    /// @invariant Normalization neither specializes nor identifies bound variables.
    inline void normalizeUnaryImplicationEndpoints(ScratchArena& arena,
        StrSpan sourcePremise, StrSpan sourceHead, ScratchString& premise,
        ScratchString& head) {
        constexpr int capacity = 2 * ExecutionParameters::MAX_ARITY;
        StrReplacement mapping[capacity];
        char numbers[capacity][12];
        int count = 0;
        for (StrSpan atom : {sourcePremise, sourceHead}) {
            StrSpan arguments[ExecutionParameters::MAX_ARITY];
            const int arity = getArgsSpans(atom, arguments, ExecutionParameters::MAX_ARITY);
            for (int i = 0; i < arity; ++i) {
                if (startsWithSpan(arguments[i], "u_", 2)) continue;
                int found = 0;
                while (found < count && !equalSpans(mapping[found].key, arguments[i])) ++found;
                if (found != count) continue;
                assert(count < capacity);
                const int length = std::snprintf(numbers[count], sizeof(numbers[count]), "%d", count + 1);
                assert(length > 0 && length < static_cast<int>(sizeof(numbers[count])));
                mapping[count] = {arguments[i], StrSpan(numbers[count], length)};
                ++count;
            }
        }
        premise = replaceKeysScratch(arena, sourcePremise, mapping, count);
        head = replaceKeysScratch(arena, sourceHead, mapping, count);
    }

    /// @brief Compose two single-premise, typed flat implications without heap allocation.
    ///
    /// @details Bound arguments belong to separate source namespaces. Matching
    /// is a bijective renaming of the complete source binder sets through the
    /// intermediate ports: bound-to-fixed specialization and binder collapse are
    /// incompatible. Every binder survives in the endpoints. Repeated arguments
    /// and fixed u_ parameters remain unchanged. Identical signed intermediate operators
    /// have identical port types; both source rules must already be well typed.
    /// The surviving endpoints use first-occurrence numeric bound names, so
    /// repeatedly composing a cycle cannot create fresh spelling variants.
    ///
    /// @param arena Caller-owned string scratch; owns successful output spans.
    /// @param firstPremise First implication's premise.
    /// @param firstHead First implication's head.
    /// @param secondPremise Second implication's premise.
    /// @param secondHead Second implication's head.
    /// @param premise Receives the composed premise only on a match.
    /// @param head Receives the composed head only on a match.
    /// @return Whether both rules have the same binders under the intermediate renaming.
    /// @invariant Both sources and the result retain one common binder set; source spans outlive the call.
    inline UnaryCompositionMatch composeUnaryImplications(
        ScratchArena& arena, StrSpan firstPremise, StrSpan firstHead,
        StrSpan secondPremise, StrSpan secondHead, ScratchString& premise,
        ScratchString& head) {
        constexpr int capacity = 4 * ExecutionParameters::MAX_ARITY;
        const StrSpan atoms[4] = {firstPremise, firstHead, secondPremise, secondHead};
        assert(firstHead.len > 0 && secondPremise.len > 0);
        if ((firstHead.ptr[0] == '!') != (secondPremise.ptr[0] == '!')
            || !equalSpans(extractExpressionUniversalSpan(firstHead),
                           extractExpressionUniversalSpan(secondPremise)))
            return UnaryCompositionMatch::incompatible;

        StrSpan arguments[4][ExecutionParameters::MAX_ARITY];
        int counts[4];
        for (int a = 0; a < 4; ++a)
            counts[a] = getArgsSpans(atoms[a], arguments[a], ExecutionParameters::MAX_ARITY);
        if (counts[1] != counts[2]) return UnaryCompositionMatch::incompatible;

        StrSpan names[capacity];
        int sides[capacity], parents[capacity], partners[capacity], argumentIds[4][ExecutionParameters::MAX_ARITY];
        StrSpan fixed[capacity];
        int size = 0;
        for (int a = 0; a < 4; ++a) {
            for (int i = 0; i < counts[a]; ++i) {
                const StrSpan name = arguments[a][i];
                const bool immutable = name.len >= 2 && name.ptr[0] == 'u' && name.ptr[1] == '_';
                const int side = immutable ? -1 : a / 2;
                int id = 0;
                while (id < size && !(side == sides[id] && equalSpans(name, names[id]))) ++id;
                if (id == size) {
                    assert(size < capacity);
                    names[id] = name;
                    sides[id] = side;
                    parents[id] = id;
                    partners[id] = -1;
                    fixed[id] = immutable ? name : StrSpan("", 0);
                    ++size;
                }
                argumentIds[a][i] = id;
            }
        }
        const auto root = [&](int id) {
            while (parents[id] != id) id = parents[id];
            return id;
        };
        for (int i = 0; i < counts[1]; ++i) {
            const int left = argumentIds[1][i];
            const int right = argumentIds[2][i];
            if (left == right) continue;
            if (sides[left] == -1 || sides[right] == -1)
                return UnaryCompositionMatch::incompatible;
            if ((partners[left] != -1 && partners[left] != right)
                || (partners[right] != -1 && partners[right] != left))
                return UnaryCompositionMatch::incompatible;
            partners[left] = right;
            partners[right] = left;
            parents[right] = left;
        }
        for (int i = 0; i < size; ++i)
            if (sides[i] != -1 && partners[i] == -1)
                return UnaryCompositionMatch::incompatible;

        char numbers[capacity][12];
        StrSpan replacements[capacity];
        bool assigned[capacity]{};
        int next = 0;
        const auto render = [&](int a) {
            StrReplacement mapping[ExecutionParameters::MAX_ARITY];
            int mappingCount = 0;
            for (int i = 0; i < counts[a]; ++i) {
                const int id = root(argumentIds[a][i]);
                if (!assigned[id]) {
                    assigned[id] = true;
                    if (fixed[id].len) replacements[id] = fixed[id];
                    else {
                        ++next;
                        const int n = std::snprintf(numbers[id], sizeof(numbers[id]), "%d", next);
                        assert(n > 0 && n < static_cast<int>(sizeof(numbers[id])));
                        replacements[id] = StrSpan(numbers[id], n);
                    }
                }
                bool repeated = false;
                for (int j = 0; j < mappingCount; ++j)
                    repeated |= equalSpans(mapping[j].key, arguments[a][i]);
                if (!repeated) mapping[mappingCount++] = {arguments[a][i], replacements[id]};
            }
            return replaceKeysScratch(arena, atoms[a], mapping, mappingCount);
        };
        const ScratchString normalizedPremise = render(0);
        const ScratchString normalizedHead = render(3);
        for (int i = 0; i < size; ++i)
            if (sides[i] != -1 && !assigned[root(i)])
                return UnaryCompositionMatch::incompatible;
        premise = normalizedPremise;
        head = normalizedHead;
        return UnaryCompositionMatch::matched;
    }
}
