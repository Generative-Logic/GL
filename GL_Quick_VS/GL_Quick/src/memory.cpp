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

// Hash engine — translation unit.
//
// Hosts the bodies of the hash-engine member functions of ExpressionAnalyzer
// that were extracted from prover.cpp so they live in the same TU as the data
// structures they operate on (HashMemory, LocalMemoryValue, EncodedExpression,
// IntNormalizedKey, NameMap — all already in memory.hpp). The class
// declaration itself remains in prover.hpp because C++ requires a single
// class definition; only the function bodies live here.
//
// Bodies are added across a chain of incremental commits (B–E in the
// hash-move plan). This commit (A) is scaffolding: the file compiles and
// links empty so the build manifest change is verified in isolation.

#include "memory.hpp"
#include "prover.hpp"
#include "parameters.hpp"
#include "msvc_sort.hpp"

#include <algorithm>
#include <cstring>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

namespace gl {


    // ------------------------------------------------------------------
    // Hash-engine rule installation (moved from prover.cpp in commit B).
    // Path 1 — head-implication LMV inserts via addToHashMemory.
    // Path 2 — marker LMV inserts via makeNormalizedKeysForAdmission.
    // ------------------------------------------------------------------

    /// @brief Install one or more `LocalMemoryValue` records into a `HashMemory`,
    /// keyed on the canonical normalization of `key`.
    ///
    /// @details
    /// Path 1 of the hash-engine rule installation: head-implication LMV inserts.
    /// Each call corresponds to one head-implication that the prover wants to
    /// register with the hash engine for later firing.
    ///
    /// The body unfolds in three logical phases:
    ///
    /// 1. **multiplyImplication.** The producer implication is expanded via
    ///    `multiplyImplication(originalImplication)` into one or more copies that
    ///    differ in how their bound variables and remaining-args are partitioned.
    ///    Copy 0 reuses the caller-supplied `key`/`value`; copies 1..N synthesize
    ///    their own from `multiplyImplication`'s output. Provenance for the
    ///    multiplied copies is recorded in `mb.exprOriginMap` (and the mail-out
    ///    mirror) under the `"multiplied from"` tag, capped per `max_origin_per_expr`
    ///    (`compressor_max_origins_per_expr` in compressor mode).
    /// 2. **Per-copy normalization + dedupe.** For each copy, build the
    ///    `IntNormalizedKey` from the int16_t-encoded key bytes; consult
    ///    `normalizedEncodedKeys` and `normalizedEncodedSubkeys*` for fast
    ///    rejection; if novel, allocate via `KeyArena::store` so the key
    ///    pointer stays stable for the rest of the LB's lifetime.
    /// 3. **LMV emit + admission.** Construct a `LocalMemoryValue` carrying
    ///    the value, levels, justification, originalImplication, key,
    ///    remainingArgs, validityName. Set `productOfDisintegration` per
    ///    [D-32](../../docs/40_decisions.md#d-32) when the producing
    ///    implication has at least one `u_`-prefixed premise arg (this gates
    ///    later OR-disintegration on the head-firing path). Push into
    ///    `targetIntMemory.encodedMap[normKey]`. When
    ///    `performAdmissionMapUpdate == true`, also call into the admission
    ///    side via `updateAdmissionMap3` so the `(expression, scope)` pair
    ///    is registered in `targetIntMemory.admissionMap`.
    ///
    /// @param key                     Pre-built normalization key (string form;
    ///                                int16_t mirror is computed inside).
    /// @param value                   Head expression text (what the rule produces).
    /// @param remainingArgs           Argument names not consumed by the key
    ///                                that must travel with the head.
    /// @param mb                      Owning `Memory` (used for `exprOriginMap`,
    ///                                `mailOut`, `keyArena`).
    /// @param targetIntMemory         The `HashMemory` to install into. Usually
    ///                                `mb.localHashMemory` for fresh installs;
    ///                                deltas go to `mb.localHashMemoryDelta`.
    /// @param levels                  LB levels at which the rule is admissible.
    /// @param originalImplication     Full producer implication text, kept for
    ///                                provenance and for `multiplyImplication`.
    /// @param maxAdmissionDepth       Standard depth budget passed to the
    ///                                admission map.
    /// @param maxSecondaryNumber      Standard secondary-iteration budget.
    /// @param partOfRecursion         True when the producer is part of a
    ///                                recursion block; routes admission-map
    ///                                updates differently.
    /// @param minNumOperatorsKey      Minimum-operator threshold used by some
    ///                                downstream filters; carried through.
    /// @param justification           Proof tag string emitted on hit
    ///                                (e.g. `"hash"`, `"head"`, `"equality1"`).
    /// @param performAdmissionMapUpdate If true, call `updateAdmissionMap3`
    ///                                in lockstep with the LMV install.
    /// @param originalImplicationClean De-renamed form used as the antecedent
    ///                                in `"multiplied from"` origin records.
    /// @param validityName            Scope at which the rule fires. `"main"`
    ///                                or a `_boundary_`-bearing form minted via
    ///                                `NameMap::encodePush` per
    ///                                [I-2](../../docs/30_invariants.md#i-2).
    ///
    /// @pre  `mb.keyArena` is alive for the LB's lifetime — pointers minted in
    ///       this call by `KeyArena::store` will be referenced from
    ///       `IntNormalizedKey::data` until the next LB-teardown.
    /// @post `targetIntMemory.encodedMap` and the related `normalizedEncoded*`
    ///       sets are extended with the new LMV(s). Origin entries are added
    ///       under the `"multiplied from"` tag for each non-zero copy.
    /// @invariant [I-2](../../docs/30_invariants.md#i-2) — non-`"main"`
    ///            validityName must be minted via `NameMap::encodePush`.
    /// @invariant [I-3](../../docs/30_invariants.md#i-3) — when this function
    ///            calls `nm.decode()` / reads `idToSub[]`, the result is
    ///            copied before any nested mint (re-entrance through
    ///            `multiplyImplication` may grow the underlying vectors).
    /// @invariant [I-25](../../docs/30_invariants.md#i-25) — cross-scope
    ///            deposits keep `validityName` end-to-end via
    ///            `ExpressionWithValidity` pairs.
    /// @see `prover.hpp::checkLocalEncodedMemoryStatic` — consumer of the
    ///      LMVs installed here.
    /// @see `prover.hpp::updateAdmissionMap3` — admission-side counterpart.
    /// @see `multiplyImplication` (`prover.cpp`) — generator of the per-copy
    ///      key/value pairs.
    void ExpressionAnalyzer::addToHashMemory(const std::vector<std::string>& key,
        const std::string& value,
        const std::set<std::string>& remainingArgs,
        Memory& mb,
        HashMemory& targetIntMemory,
        const std::set<int>& levels,
        const std::string& originalImplication,
        int maxAdmissionDepth,
        int maxSecondaryNumber,
        bool partOfRecursion,
        int minNumOperatorsKey,
        const std::string& justification,
        bool performAdmissionMapUpdate,
        const std::string& originalImplicationClean,
        const std::string& validityName) {

        // Own multiplyImplication loop — fully independent from addToHashMemory.
        std::vector<std::string> copies = multiplyImplication(originalImplication);

        for (std::size_t c = 0; c < copies.size(); ++c) {
            std::vector<std::string> curKey;
            std::string curValue;
            std::string curOrigImpl = copies[c];

            if (c == 0) {
                curKey = key;
                curValue = value;
            } else {
                // Track history for multiplied copies
                if (parameters.trackHistory) {
                    ExpressionWithValidity copyEv(copies[c], "main");
                    std::pair<std::string, std::vector<ExpressionWithValidity>> mulOrigin;
                    mulOrigin.first = "multiplied from";
                    mulOrigin.second.push_back(ExpressionWithValidity(originalImplicationClean, "main"));
                    addOrigin(mb.exprOriginMap, copyEv, mulOrigin,
                        (parameters.compressor_mode
                            ? parameters.compressor_max_origins_per_expr
                            : parameters.max_origin_per_expr));
                    addOrigin(mb.mailOut.exprOriginMap, copyEv, mulOrigin,
                        (parameters.compressor_mode
                            ? parameters.compressor_max_origins_per_expr
                            : parameters.max_origin_per_expr));
                }

                std::vector< std::tuple<
                    std::string,
                    std::vector<std::string>,
                    std::set<std::string>
                > > tc;
                curValue = ce::disintegrateImplication(copies[c], tc, coreExpressionMap);
                for (std::size_t ci = 0; ci < tc.size(); ++ci)
                    curKey.push_back(std::get<0>(tc[ci]));
            }

            // --- Shared members (path-independent, needed for integration) ---
            std::vector<std::string> implication = curKey;
            implication.push_back(curValue);
            targetIntMemory.originals.insert(implication);
            for (const auto& trigger : targetIntMemory.triggersForAdmissionSetIntegration)
            {
                makeAdmissionKeys(implication, trigger.original, targetIntMemory, trigger.validityName, mb);
            }

            this->makeNormalizedKeysForAdmission(curKey, targetIntMemory,
                mb.nameMap, mb.keyArena, curValue, minNumOperatorsKey, &mb);

            // Recompute remainingArgs for multiplied copies (u_ vars may have been equalized)
            const std::set<std::string>& curRemainingArgs = (c == 0) ? remainingArgs : getRemainingArgs(curKey);

            if (performAdmissionMapUpdate)
            {
                std::vector<std::string> keyPlusValue = curKey;
                keyPlusValue.push_back(curValue);
                updateAdmissionMap(mb,
                    keyPlusValue,
                    curRemainingArgs,
                    maxAdmissionDepth,
                    maxSecondaryNumber,
                    partOfRecursion,
                    "main");
            }

            // --- Inner: add this single key/value to int hash memory ---
            const std::size_t n = curKey.size();
            NameMap& nm = mb.nameMap;
            KeyArena& arena = mb.keyArena;

            std::set<int16_t> intRemainingArgs;
            for (const auto& ra : curRemainingArgs) {
                intRemainingArgs.insert(nm.encode(ra));
            }

            std::map<int, std::vector<std::vector<int>>>::const_iterator pit =
                this->allPermutationsAna.find(static_cast<int>(n));
            if (pit == this->allPermutationsAna.end()) {
                continue;
            }
            const std::vector<std::vector<int>>& permuts = pit->second;

            std::vector<std::string> ids;
            ids.reserve(n);
            for (std::size_t i = 0; i < n; ++i) {
                ids.push_back(ce::extractExpression(curKey[i]));
            }

            int16_t bufIgnored[ExecutionParameters::MAX_KEY_SLOTS];
            int16_t bufNotIgnored[ExecutionParameters::MAX_KEY_SLOTS];

            for (std::size_t p = 0; p < permuts.size(); ++p) {
                const std::vector<int>& permutation = permuts[p];

                bool toContinue = false;
                if (n >= 2) {
                    for (std::size_t k = 0; k + 1 < n; ++k) {
                        if (ids[permutation[k]] > ids[permutation[k + 1]]) {
                            toContinue = true;
                            break;
                        }
                    }
                }
                if (toContinue) continue;

                std::vector<EncodedExpression> tempEncoded;
                tempEncoded.reserve(n);
                for (std::size_t k = 0; k < permutation.size(); ++k) {
                    tempEncoded.push_back(EncodedExpression(curKey[permutation[k]], "main"));
                }

                int16_t reverseMap[ExecutionParameters::MAX_KEY_SLOTS];
                std::memset(reverseMap, 0, sizeof(reverseMap));
                int16_t numNormVars = 0;
                int16_t lenIgnored = makeIntNormalizedKeyWithMap(
                    tempEncoded, nm, true, bufIgnored,
                    ExecutionParameters::MAX_KEY_SLOTS, reverseMap, numNormVars);

                int16_t lenNotIgnored = makeIntNormalizedKey(
                    tempEncoded, nm, false, bufNotIgnored,
                    ExecutionParameters::MAX_KEY_SLOTS);

                const int16_t* pIgnored = arena.store(bufIgnored, lenIgnored);
                const int16_t* pNotIgnored = arena.store(bufNotIgnored, lenNotIgnored);

                IntNormalizedKey intIgnoredKey(static_cast<int16_t>(n), pIgnored, lenIgnored);
                IntNormalizedKey intNotIgnoredKey(static_cast<int16_t>(n), pNotIgnored, lenNotIgnored);

                std::string valueVariant = curValue;
                {
                    std::pair<NormalizedKey, std::map<std::string, std::string>> pairIgnored =
                        makeNormalizedEncodedKey(tempEncoded, true);
                    const std::map<std::string, std::string>& mp2 = pairIgnored.second;
                    valueVariant = ce::replaceKeysInString(curValue, mp2);
                }

                LocalMemoryValue lmv;
                lmv.value = valueVariant;
                lmv.levels = levels;
                lmv.originalImplication = curOrigImpl;
                lmv.remainingArgs = curRemainingArgs;
                lmv.justification = justification;
                lmv.validityName = validityName;

                // D-32: stamp product-of-disintegration flag — true iff at
                // least one premise (chain element) has an arg starting with
                // "u_". The "u_" prefix is reserved for bound-variable
                // placeholders introduced by prefixArgumentsWithU during
                // disintegration; an anchor-bound theorem chain (e.g. starts
                // with (AnchorPeano[1,2,3,...])) has only concrete integer
                // args. Consumed by checkLocalEncodedMemoryStatic to gate
                // OR-disintegration on the head-firing path.
                for (const auto& premise : curKey) {
                    bool found = false;
                    for (const auto& arg : ce::getArgs(premise)) {
                        if (arg.compare(0, 2, "u_") == 0) {
                            lmv.productOfDisintegration = true;
                            found = true;
                            break;
                        }
                    }
                    if (found) break;
                }

                targetIntMemory.encodedMap[intIgnoredKey].push_back(lmv);
                targetIntMemory.normalizedEncodedKeys.insert(intNotIgnoredKey);
                targetIntMemory.remainingArgsNormalizedEncodedMap[intRemainingArgs].insert(intNotIgnoredKey);
            }

            // --- makeNormalizedSubkeys equivalent ---
            for (std::size_t p = 0; p < permuts.size(); ++p) {
                const std::vector<int>& permut = permuts[p];

                std::vector<std::string> tempList;
                tempList.reserve(n);
                for (std::size_t k = 0; k < permut.size(); ++k) {
                    tempList.push_back(curKey[permut[k]]);
                }

                for (std::size_t index = 0; index < n; ++index) {
                    bool toBreak = false;
                    for (std::size_t index2 = 0; index2 < index; ++index2) {
                        if (ids[permut[index2]] > ids[permut[index2 + 1]]) {
                            toBreak = true;
                            break;
                        }
                    }
                    if (toBreak) break;

                    std::vector<EncodedExpression> subkeyEncoded;
                    subkeyEncoded.reserve(index + 1);
                    for (std::size_t t = 0; t <= index; ++t) {
                        subkeyEncoded.push_back(EncodedExpression(tempList[t], "main"));
                    }

                    int16_t subBuf[ExecutionParameters::MAX_KEY_SLOTS];
                    int16_t subLen = makeIntNormalizedKey(
                        subkeyEncoded, nm, false, subBuf,
                        ExecutionParameters::MAX_KEY_SLOTS);

                    const int16_t* pSub = arena.store(subBuf, subLen);
                    IntNormalizedKey intSubKey(static_cast<int16_t>(index + 1), pSub, subLen);

                    targetIntMemory.normalizedEncodedSubkeys.insert(intSubKey);

                    if (index + 1 == n - 1) {
                        targetIntMemory.normalizedEncodedSubkeysMinusOne.insert(intSubKey);
                    }
                    if (index + 1 == n - 2) {
                        targetIntMemory.normalizedEncodedSubkeysMinusTwo.insert(intSubKey);
                    }
                }
            }

            targetIntMemory.maxKeyLength = std::max(static_cast<int16_t>(n), targetIntMemory.maxKeyLength);
        }
    }

    /// @brief Path 2 of hash-engine rule installation — emit *marker* LMVs into
    /// `intHashMemory` for the per-element subkeys of an admitted implication.
    ///
    /// @details
    /// Where `addToHashMemory` installs the head-firing LMV, this function
    /// installs *marker* LMVs whose role is purely to gate the firing path —
    /// they don't carry a usable head, but their presence in the
    /// `normalizedEncodedSubkeys*` sets advertises that the rule is admissible
    /// for fast-rejection lookups.
    ///
    /// For each element index of the input `key`, the function:
    /// 1. Extracts the core expression and its declared output-index list via
    ///    `ce::getArgs` + the per-anchor `coreExpressionMap`.
    /// 2. Asserts there is at most one output argument
    ///    ([I-19](../../docs/30_invariants.md#i-19) — assert is first-class;
    ///    a firing assert here means the core map declared a multi-output
    ///    operator, which the rest of the engine does not handle).
    /// 3. Builds a binary mask + subkey by selecting other elements that do
    ///    NOT contain the output arg, then qualifies the result via two gates:
    ///    - the *classic* gate, baseline-only behavior preserved by the
    ///      static-pipeline migration; fires when the subkey size is exactly
    ///      `n - 1` and `baselineClassicQualifies` accepts.
    ///    - the *local-u* gate, used for `(C)` qualification (see
    ///      `implicationIsQualified` / `baselineClassicQualifies` in
    ///      prover.hpp); admits a wider class of subkeys but only when the
    ///      key has a `u_`-prefixed arg meeting the local-u criterion.
    /// 4. Emits the marker LMV via the same `KeyArena::store` storage
    ///    contract as `addToHashMemory` — the int16_t key bytes live in the
    ///    arena so the resulting `IntNormalizedKey::data` pointer stays
    ///    stable for the rest of the LB's lifetime.
    ///
    /// `mbTrap` is reserved for instrumented builds; the runtime path
    /// drops it on the floor (`(void)mbTrap;` at entry).
    ///
    /// @param key                 Producer's key (string form).
    /// @param intHashMemory       Target hash memory (typically the same as
    ///                            `targetIntMemory` in the matching
    ///                            `addToHashMemory` call).
    /// @param nameMap             `NameMap` for int16_t encoding.
    /// @param arena               `KeyArena` for stable storage of the
    ///                            normalized key bytes.
    /// @param value               Head expression text — used by the
    ///                            qualification gate but NOT stored in
    ///                            the resulting marker LMV.
    /// @param minNumOperatorsKey  Threshold for `implicationIsQualified`.
    /// @param mbTrap              Reserved (debug instrumentation hook).
    ///
    /// @pre  `coreExpressionMap` is populated for every core expression
    ///       referenced in `key` — populated at compiler-side init.
    /// @post `intHashMemory.normalizedEncodedSubkeys`,
    ///       `…SubkeysMinusOne`, and `…SubkeysMinusTwo` are extended with
    ///       fast-rejection entries for the qualifying subkeys.
    /// @invariant [I-19](../../docs/30_invariants.md#i-19) — the
    ///            single-output-arg assert here is intentional and must
    ///            not be weakened.
    /// @see [`HashMemory::normalizedEncodedSubkeys`](memory.hpp#hashmemory).
    /// @see `prover.hpp::implicationIsQualified`,
    ///      `prover.hpp::baselineClassicQualifies` — gate predicates.
    void ExpressionAnalyzer::makeNormalizedKeysForAdmission(
        const std::vector<std::string>& key,
        HashMemory& intHashMemory,
        NameMap& nameMap,
        KeyArena& arena,
        const std::string& value,
        int minNumOperatorsKey,
        const Memory* mbTrap) {

        (void)mbTrap;

        if (!implicationIsQualified(key, value, minNumOperatorsKey)) {
            return;
        }

        const std::size_t n = key.size();

        for (std::size_t index = 0; index < n; ++index) {
            const std::string coreExpr = ce::extractExpression(key[index]);
            const std::vector<std::string> args = ce::getArgs(key[index]);
            auto cit = this->coreExpressionMap.find(coreExpr);
            assert(cit != this->coreExpressionMap.end() && "Core expression not found");

            std::vector<std::string> outputArgsList;
            for (int idx : cit->second.outputIndices) {
                if (idx >= 0 && idx < static_cast<int>(args.size()))
                    outputArgsList.push_back(args[idx]);
            }
            assert(outputArgsList.size() <= 1 && "Expression must have 0 or 1 output argument");
            if (outputArgsList.empty()) {
                continue;
            }
            const std::string outputArg = outputArgsList[0];

            // Build binary mask and subkey (same logic as string path)
            std::vector<int> binary(n, 0);
            std::vector<std::string> subkey;
            subkey.reserve(n);
            int validCount = 0;
            for (std::size_t i = 0; i < n; ++i) {
                if (i == index) continue;
                const std::vector<std::string> iArgs = ce::getArgs(key[i]);
                bool containsOutput = false;
                for (const auto& arg : iArgs) {
                    if (arg == outputArg) { containsOutput = true; break; }
                }
                if (!containsOutput) {
                    binary[i] = 1;
                    subkey.push_back(key[i]);
                    validCount++;
                }
            }

            // Classic path is baseline behavior untouched — fires ONLY on
            // keys that would have qualified via (A) or (B) in baseline.
            // (C) qualification (hasUPrefix + localUCriterion) admits keys
            // into this loop but does NOT enable the classic per-element
            // gates. Those keys must go through the local-u path below.
            const bool baselineQualified = baselineClassicQualifies(key, value, minNumOperatorsKey);
            const bool acceptedByClassic = baselineQualified && (
                (validCount == static_cast<int>(n) - 1) ||
                (validCount == parameters.minLenLongKey - 1));
            const bool acceptedByLocalU = hasUPrefixInKey(key) &&
                this->elementMatchesLocalUCriterion(index, key, value);
            bool accept = acceptedByClassic || acceptedByLocalU;

            // Narrowing guard applies ONLY to the local u_-impl path — the
            // NEW case. Classic path behavior stays exactly as baseline.
            if (accept && !acceptedByClassic &&
                this->outputMatchesHeadOutputSlot(outputArg, value)) {
                accept = false;
            }

            if (!accept) continue;

            // Replace output arg with "marker"
            std::map<std::string, std::string> replacementMap;
            replacementMap.insert(std::make_pair(outputArg, "marker"));
            const std::string replaced = ce::replaceKeysInString(key[index], replacementMap);

            // --- Int subkeys (mirrors makeNormalizedSubkeys for HashMemory) ---
            {
                const std::size_t sn = subkey.size();
                auto pit = this->allPermutationsAna.find(static_cast<int>(sn));
                if (pit != this->allPermutationsAna.end()) {
                    const auto& permuts = pit->second;
                    std::vector<std::string> sids;
                    sids.reserve(sn);
                    for (std::size_t i = 0; i < sn; ++i)
                        sids.push_back(ce::extractExpression(subkey[i]));

                    for (std::size_t p = 0; p < permuts.size(); ++p) {
                        const auto& permut = permuts[p];
                        std::vector<std::string> tempList;
                        tempList.reserve(sn);
                        for (std::size_t k = 0; k < permut.size(); ++k)
                            tempList.push_back(subkey[permut[k]]);

                        for (std::size_t si = 0; si < sn; ++si) {
                            bool toBreak = false;
                            for (std::size_t si2 = 0; si2 < si; ++si2) {
                                if (sids[permut[si2]] > sids[permut[si2 + 1]]) {
                                    toBreak = true; break;
                                }
                            }
                            if (toBreak) break;

                            std::vector<EncodedExpression> subEnc;
                            subEnc.reserve(si + 1);
                            for (std::size_t t = 0; t <= si; ++t)
                                subEnc.push_back(EncodedExpression(tempList[t], "main"));

                            int16_t subBuf[ExecutionParameters::MAX_KEY_SLOTS];
                            int16_t subLen = makeIntNormalizedKey(
                                subEnc, nameMap, false, subBuf, ExecutionParameters::MAX_KEY_SLOTS);
                            const int16_t* pSub = arena.store(subBuf, subLen);
                            IntNormalizedKey intSubKey(static_cast<int16_t>(si + 1), pSub, subLen);

                            intHashMemory.normalizedEncodedSubkeys.insert(intSubKey);
                            if (si + 1 == sn - 1)
                                intHashMemory.normalizedEncodedSubkeysMinusOne.insert(intSubKey);
                            if (si + 1 == sn - 2)
                                intHashMemory.normalizedEncodedSubkeysMinusTwo.insert(intSubKey);
                        }
                    }
                }
            }

            // --- Main variants: create int keys for encodedMap marker entries ---
            {
                const std::size_t sn = subkey.size();
                auto pit = this->allPermutationsAna.find(static_cast<int>(sn));
                if (pit == this->allPermutationsAna.end()) continue;
                const auto& permuts = pit->second;

                std::vector<std::string> sids;
                sids.reserve(sn);
                for (std::size_t i = 0; i < sn; ++i)
                    sids.push_back(ce::extractExpression(subkey[i]));

                std::set<std::string> remainingArgs = getRemainingArgs(subkey);
                std::set<int16_t> intRemainingArgs;
                for (const auto& ra : remainingArgs)
                    intRemainingArgs.insert(nameMap.encode(ra));

                int16_t bufIgnored[ExecutionParameters::MAX_KEY_SLOTS];
                int16_t bufNotIgnored[ExecutionParameters::MAX_KEY_SLOTS];

                for (std::size_t p = 0; p < permuts.size(); ++p) {
                    const auto& permutation = permuts[p];

                    bool toContinue = false;
                    if (sn >= 2) {
                        for (std::size_t k = 0; k + 1 < sn; ++k) {
                            if (sids[permutation[k]] > sids[permutation[k + 1]]) {
                                toContinue = true; break;
                            }
                        }
                    }
                    if (toContinue) continue;

                    std::vector<EncodedExpression> tempEncoded;
                    tempEncoded.reserve(sn);
                    for (std::size_t k = 0; k < permutation.size(); ++k)
                        tempEncoded.push_back(EncodedExpression(subkey[permutation[k]], "main"));

                    // Int keys from subkey
                    int16_t lenIgnored = makeIntNormalizedKey(
                        tempEncoded, nameMap, true, bufIgnored, ExecutionParameters::MAX_KEY_SLOTS);
                    int16_t lenNotIgnored = makeIntNormalizedKey(
                        tempEncoded, nameMap, false, bufNotIgnored, ExecutionParameters::MAX_KEY_SLOTS);

                    const int16_t* pIgnored = arena.store(bufIgnored, lenIgnored);
                    const int16_t* pNotIgnored = arena.store(bufNotIgnored, lenNotIgnored);

                    IntNormalizedKey intIgnoredKey(static_cast<int16_t>(sn), pIgnored, lenIgnored);
                    IntNormalizedKey intNotIgnoredKey(static_cast<int16_t>(sn), pNotIgnored, lenNotIgnored);

                    // String normalization mapping for value variant and replKey
                    std::pair<NormalizedKey, std::map<std::string, std::string>> pairIgnored =
                        makeNormalizedEncodedKey(tempEncoded, true);
                    const std::map<std::string, std::string>& mp2 = pairIgnored.second;

                    std::string valueVariant = ce::replaceKeysInString(replaced, mp2);

                    // Build replKey (valid subkey elements with mapping applied).
                    // Earlier this also pushed key[index] — the marker-source
                    // element — which polluted lmv.key with the target expression
                    // itself. downstream consumers (isAdmitted, admissionMapPropagate)
                    // iterate lmv.key expecting real premises; the self-row
                    // self-matches and produces spurious replacements. Subkey only.
                    std::map<std::string, std::string> mpMap(mp2.begin(), mp2.end());
                    std::vector<std::string> replKey;
                    replKey.reserve(subkey.size());
                    for (std::size_t i = 0; i < key.size(); ++i) {
                        if (binary[i]) {
                            replKey.push_back(ce::replaceKeysInString(key[i], mpMap));
                        }
                    }

                    LocalMemoryValue lmv;
                    lmv.value = valueVariant;
                    lmv.key = replKey;
                    lmv.remainingArgs = remainingArgs;

                    intHashMemory.encodedMap[intIgnoredKey].push_back(lmv);
                    intHashMemory.normalizedEncodedKeys.insert(intNotIgnoredKey);
                    intHashMemory.remainingArgsNormalizedEncodedMap[intRemainingArgs].insert(intNotIgnoredKey);
                }
            }
        }
    }

    /// @brief Comparator on `EncodedExpression::name` — used when sorting
    /// statements by their core expression name.
    /// @details Stable lexicographic order over `.name`. Used when the static
    /// pipeline wants name-grouped iteration; pairs with `lessByOriginal` for
    /// the alternative ordering.
    bool ExpressionAnalyzer::lessByName(const EncodedExpression& a, const EncodedExpression& b) {
        return a.name < b.name;
    }

    /// @brief Comparator on `EncodedExpression::original` — used to sort by
    /// raw source text.
    /// @details Stable lexicographic order over `.original`, the un-rewritten
    /// expression string. Used by paths that need ordering by the
    /// user-visible form rather than by the internal name; pairs with
    /// `lessByName`.
    bool ExpressionAnalyzer::lessByOriginal(const EncodedExpression& a, const EncodedExpression& b) {
        return a.original < b.original;
    }

    // ------------------------------------------------------------------
    // Hash-engine mandatory-element filters (moved from prover.cpp in
    // commit C). Singles + Pairs versions used by the request-generator
    // pipeline.
    // ------------------------------------------------------------------

    /// @brief Singles variant of the mandatory-element filter — narrows a
    /// list of pre-encoded statements down to those that pass the
    /// `filterIntEncodedStatements` admission gate, then sorts by
    /// `originalId` for stable output.
    ///
    /// @details
    /// The static request pipeline operates on pre-encoded `IntEncodedExpr`
    /// arrays. Before generating requests it must filter the candidates so
    /// only statements whose normalized form is admissible against the
    /// rule's hash key remain. This singles variant handles the
    /// 1-statement case; `makeMandatoryEncodedStatementLists2Static` (in
    /// the same file) handles the pair case used during the
    /// `makeMandatory2` merge.
    ///
    /// @param mem        Reference `HashMemory` whose `normalizedEncoded*`
    ///                   sets the admission check consults.
    /// @param stmts      Pre-encoded statement array.
    /// @param stmtCount  Number of valid entries in `stmts`.
    /// @param outIndices Destination array for the surviving indices.
    /// @param maxOut     Capacity of `outIndices`.
    /// @return Number of indices written into `outIndices`. Capped at
    ///         `maxOut`; the local filter buffer is sized at 4096 which
    ///         comfortably exceeds any per-LB statement count seen so far.
    /// @pre  `stmts` lives at least until this function returns.
    /// @post `outIndices[0..return-1]` is the surviving subset, sorted
    ///       ascending by `originalId`.
    int16_t ExpressionAnalyzer::makeMandatoryEncodedStatementLists1Static(
        const HashMemory& mem,
        const IntEncodedExpr* stmts, int16_t stmtCount,
        int16_t* outIndices, int16_t maxOut)
    {
        // Filter, then sort by originalId (proxy for stable sort by original string)
        int16_t filtBuf[4096];
        int16_t nFilt = filterIntEncodedStatements(stmts, stmtCount, mem, filtBuf, 4096);

        // Insertion sort filtered indices by originalId (counts are small)
        for (int16_t i = 1; i < nFilt; ++i) {
            int16_t key = filtBuf[i];
            int16_t j = i - 1;
            while (j >= 0 && stmts[filtBuf[j]].originalId > stmts[key].originalId) {
                filtBuf[j + 1] = filtBuf[j];
                --j;
            }
            filtBuf[j + 1] = key;
        }

        int16_t nOut = std::min(nFilt, maxOut);
        std::memcpy(outIndices, filtBuf, nOut * sizeof(int16_t));
        return nOut;
    }

    // ------------------------------------------------------------------
    // Hash-engine request generators (moved from prover.cpp in commit D).
    // filterIntEncodedStatements was inline in prover.hpp's class body;
    // its declaration stays there, the body now lives here alongside its
    // single caller cluster (the singles/pairs generators).
    // ------------------------------------------------------------------

    /// @brief Filter pre-encoded statements down to those that pass the
    /// fast-rejection subkey-set check and the max-iteration cap.
    ///
    /// @details
    /// For each `IntEncodedExpr`, builds a single-expression
    /// `IntNormalizedKey` on a stack buffer (no heap allocation), normalizes
    /// the var ids sequentially by first appearance (`changeable=0`,
    /// `ignoreU=false`), and tests:
    /// 1. Membership in `mem.normalizedEncodedSubkeys` — the bloom-filter-
    ///    style fast-rejection set populated by `addToHashMemory` /
    ///    `makeNormalizedKeysForAdmission`. Misses are dropped.
    /// 2. `maxIteration <= parameters.maxIterationNumberVariable` — caps
    ///    blow-up on iteration counters.
    ///
    /// Output is the surviving indices into `stmts`, in original order.
    /// No heap allocation; the in-function `buf` and `varMap` are bounded
    /// by `ExecutionParameters::MAX_KEY_SLOTS`.
    ///
    /// @param stmts       Source array of pre-encoded statements.
    /// @param count       Number of valid entries in `stmts`.
    /// @param mem         Reference `HashMemory` for the subkey-set check.
    /// @param outIndices  Destination array for surviving indices.
    /// @param maxOut      Capacity of `outIndices`.
    /// @return Count of surviving indices written into `outIndices`.
    /// @pre  `stmts` and `outIndices` live for the duration of the call.
    int16_t ExpressionAnalyzer::filterIntEncodedStatements(
        const IntEncodedExpr* stmts, int16_t count,
        const HashMemory& mem,
        int16_t* outIndices, int16_t maxOut) {

        int16_t buf[ExecutionParameters::MAX_KEY_SLOTS];
        int16_t nOut = 0;

        for (int16_t i = 0; i < count && nOut < maxOut; ++i) {
            const IntEncodedExpr& s = stmts[i];

            // Build single-expr IntNormalizedKey on stack
            int16_t pos = 0;
            buf[pos++] = s.nameId;
            buf[pos++] = s.negation;
            for (int16_t j = 0; j < s.arity; ++j) {
                buf[pos++] = s.argId[j];
                buf[pos++] = 0; // changeable (ignoreU=false)
            }
            // Normalize: sequential IDs by first appearance
            {
                int16_t varMap[ExecutionParameters::MAX_KEY_SLOTS];
                int16_t nV = 0;
                int16_t nextN = 1;
                // Start after nameId+negation (pos 2), step by 2 (varId, changeable)
                for (int16_t p = 2; p < pos; p += 2) {
                    int16_t raw = buf[p];
                    int16_t norm = 0;
                    for (int16_t v = 0; v < nV; ++v) {
                        if (varMap[v * 2] == raw) { norm = varMap[v * 2 + 1]; break; }
                    }
                    if (norm == 0) {
                        norm = nextN++;
                        varMap[nV * 2] = raw;
                        varMap[nV * 2 + 1] = norm;
                        ++nV;
                    }
                    buf[p] = norm;
                }
            }

            IntNormalizedKey ik(1, buf, pos);
            if (mem.normalizedEncodedSubkeys.find(ik) == mem.normalizedEncodedSubkeys.end())
                continue;
            if (s.maxIteration > parameters.maxIterationNumberVariable)
                continue;

            outIndices[nOut++] = i;
        }
        return nOut;
    }

    /// @brief Shared grow loop for the *Singles* and *Pairs* variants of the
    /// mandatory-element filter — enumerates subsets of filtered statements
    /// up to `targetLen` collecting candidates whose subkey membership is
    /// in `targetSubkeys`.
    ///
    /// @details
    /// Implements the depth-first "grow" phase of `makeMandatory`. Starts with
    /// the empty set and grows by appending one filtered index at a time;
    /// at each depth checks whether the partial subkey is still a prefix of
    /// some target subkey (via `normalizedEncodedSubkeys` lookup) — pruning
    /// branches that cannot possibly converge. Each surviving candidate at
    /// `targetLen` is recorded as a `BaseCandidate` (its `allIdx[count]`
    /// statement indices plus the candidate's scope id).
    ///
    /// The function is hot — it runs inside `proveKernel`'s parallel hash
    /// burst — so it operates entirely on stack buffers and `int16_t` IDs.
    /// Per [I-28](../../docs/30_invariants.md#i-28), it does NOT write any
    /// shared cross-LB state during the parallel phase; results land only
    /// in the caller-supplied `baseCandidates` vector.
    ///
    /// @param body            Owning LB; supplies `nameMap` for any id work.
    /// @param intMemory       Reference hash memory for prefix-subkey lookup.
    /// @param allIntStmts     Pre-encoded statement array.
    /// @param filteredIdx     Indices into `allIntStmts` that passed the
    ///                        upstream filter pass.
    /// @param nFiltered       Length of `filteredIdx`.
    /// @param mainValidityId  Validity id used for candidates that match
    ///                        without forcing a deeper scope.
    /// @param targetLen       Subset size to grow to (typically 1 for
    ///                        Singles, 2 for Pairs).
    /// @param targetSubkeys   Set of subkeys the grow phase is steering
    ///                        toward.
    /// @param baseCandidates  Output vector. Each surviving subset is
    ///                        appended as a `BaseCandidate`.
    ///
    /// @pre  `allIntStmts` and `filteredIdx` live for the duration of the call.
    /// @post `baseCandidates` is appended (never cleared) with the
    ///       enumerated subsets.
    /// @invariant [I-28](../../docs/30_invariants.md#i-28) — no cross-LB
    ///            writes during the parallel phase.
    /// @see [`BaseCandidate`](memory.hpp#basecandidate).
    void ExpressionAnalyzer::growBaseCandidates(
        Memory& body,
        const HashMemory& intMemory,
        const IntEncodedExpr* allIntStmts,
        const int16_t* filteredIdx, int16_t nFiltered,
        int16_t mainValidityId,
        int targetLen,
        const std::unordered_set<IntNormalizedKey, IntNormalizedKeyHash>& targetSubkeys,
        std::vector<BaseCandidate>& baseCandidates)
    {
        NameMap& nm = body.nameMap;
        struct StackItem {
            int start;
            int16_t allIdx[ExecutionParameters::MAX_EXPRESSIONS];
            int16_t count;
            int16_t validityId;
        };

        std::vector<StackItem> stack;
        {
            StackItem init;
            init.start = 0;
            init.count = 0;
            init.validityId = mainValidityId;
            stack.push_back(init);
        }

        while (!stack.empty()) {
            StackItem top = stack.back();
            stack.pop_back();

            for (int i = top.start; i < nFiltered; ++i) {
                if (top.count + 1 > targetLen) break;

                const int16_t allIdx = filteredIdx[i];
                const IntEncodedExpr& ie = allIntStmts[allIdx];

                if (!nm.comparable(top.validityId, ie.validityId)) continue;
                int16_t newValidityId = nm.deeperOf(top.validityId, ie.validityId);

                const IntEncodedExpr* ptrs[ExecutionParameters::MAX_EXPRESSIONS];
                for (int16_t k = 0; k < top.count; ++k)
                    ptrs[k] = &allIntStmts[top.allIdx[k]];
                ptrs[top.count] = &ie;
                const int16_t newCount = static_cast<int16_t>(top.count + 1);

                std::pair<bool, IntNormalizedKey> pr =
                    preEvaluateFromEncoded(ptrs, newCount, body, mainValidityId,
                        intMemory.normalizedEncodedSubkeys);

                if (pr.first) {
                    if (targetSubkeys.find(pr.second) != targetSubkeys.end()) {
                        BaseCandidate bc;
                        std::memcpy(bc.allIdx, top.allIdx, top.count * sizeof(int16_t));
                        bc.allIdx[top.count] = allIdx;
                        bc.count = newCount;
                        bc.validityId = newValidityId;
                        baseCandidates.push_back(bc);
                    }
                    if (newCount < targetLen) {
                        StackItem next;
                        next.start = i + 1;
                        std::memcpy(next.allIdx, top.allIdx, top.count * sizeof(int16_t));
                        next.allIdx[top.count] = allIdx;
                        next.count = newCount;
                        next.validityId = newValidityId;
                        stack.push_back(next);
                    }
                }
            }
        }
    }

    /// @brief Singles variant of the static request generator — seed +
    /// grow + merge for one-mandatory-element kernel matches.
    ///
    /// @details
    /// Three-phase pipeline:
    /// - **Seed.** Each mandatory single is tested as a complete key
    ///   against `intMemory.normalizedEncodedKeys`. Hits emit a request via
    ///   `StaticRequestEmitter::emit`.
    /// - **Grow.** Statements are filtered through `filterIntEncodedStatements`
    ///   and sorted by core-expression name. `growBaseCandidates` enumerates
    ///   subsets of length `maxKeyLen - 1` whose subkey is in
    ///   `normalizedEncodedSubkeysMinusOne`.
    /// - **Merge.** Each `BaseCandidate` is paired with each mandatory
    ///   single; if `comparable(base.validityId, mandatory.validityId)`
    ///   passes and the merged set has no `originalId` duplicate, the
    ///   merged pointer array (in name-sorted order) is emitted as a
    ///   request.
    ///
    /// @param body                Owning LB; supplies `nameMap`,
    ///                            `keyArena`, and `intEncodedStatements`.
    /// @param intMemory           Reference hash memory (provides
    ///                            `maxKeyLength`, the `normalizedEncoded*`
    ///                            sets, and `comparable` for validity-id
    ///                            checks).
    /// @param mandatoryIndices    Indices into `mandatorySrcInt` of the
    ///                            mandatory singles.
    /// @param mandatoryCount      Length of `mandatoryIndices`.
    /// @param mandatorySrcInt     Source array for the mandatory entries.
    /// @param exprArena           Per-thread `TypedArena<IntEncodedExpr>`
    ///                            backing pointer-stable copies.
    /// @param outBuf              Destination buffer for emitted requests.
    /// @param maxOut              Capacity of `outBuf`.
    /// @return Number of requests emitted (`StaticRequestEmitter::outCount`).
    /// @pre  `mandatoryCount > 0` to do useful work; the early-return at
    ///       `mandatoryCount <= 0` short-circuits.
    /// @post Each emitted request's `IntNormalizedKey::data` references
    ///       memory owned by `body.keyArena`; the pointer is stable until
    ///       the LB tears down.
    /// @invariant [I-28](../../docs/30_invariants.md#i-28) — only
    ///            per-thread (arena, body) state is mutated.
    /// @see [`StaticRequestEmitter`](memory.hpp#staticrequestemitter).
    int16_t ExpressionAnalyzer::generateEncodedRequestsStatic(
        Memory& body,
        const HashMemory& intMemory,
        const int16_t* mandatoryIndices, int16_t mandatoryCount,
        const IntEncodedExpr* mandatorySrcInt,
        TypedArena<IntEncodedExpr>& exprArena,
        StaticRequest* outBuf, int16_t maxOut)
    {
        if (mandatoryCount <= 0) return 0;

        NameMap& nm = body.nameMap;
        int16_t buf[ExecutionParameters::MAX_KEY_SLOTS];
        const int16_t mainValidityId = NameMap::MAIN_ID;
        const int maxKeyLen = intMemory.maxKeyLength;
        const int targetLen = std::max(0, maxKeyLen - 1);

        StaticRequestEmitter emitter(exprArena, outBuf, maxOut);

        // --- Seed: each mandatory single as complete key ---
        for (int16_t i = 0; i < mandatoryCount; ++i) {
            const IntEncodedExpr& ie = mandatorySrcInt[mandatoryIndices[i]];
            const IntEncodedExpr* ptr = &ie;
            int16_t len = makeIntNormalizedKeyFromEncoded(&ptr, 1, buf,
                ExecutionParameters::MAX_KEY_SLOTS);
            IntNormalizedKey probe(1, buf, len);
            if (intMemory.normalizedEncodedKeys.find(probe) != intMemory.normalizedEncodedKeys.end()) {
                const int16_t* p = body.keyArena.store(buf, len);
                IntNormalizedKey nk(1, p, len);
                emitter.emit(&ptr, 1, nk);
            }
        }

        if (targetLen <= 0) return emitter.outCount;

        // --- Filter + sort ---
        const IntEncodedExpr* allIntStmts = body.intEncodedStatements.data();
        const int16_t allIntCount = static_cast<int16_t>(body.intEncodedStatements.size());
        int16_t filteredIdx[8192];
        int16_t nFiltered = filterIntEncodedStatements(allIntStmts, allIntCount,
            intMemory, filteredIdx, 8192);
        // gl::msvc_sort: MSVC introsort re-implementation, byte-identical
        // output on Win11 MSVC STL and Linux libstdc++ (sandbox/linux3,
        // 2026-05-12). Drops the host-specific tie-resolution that
        // produced different rule-firing orders between Win11 and WSL
        // and preserves Win11 main HEAD's historical sort output (which
        // the prover's behavior was built around). See msvc_sort.hpp.
        gl::msvc_sort(filteredIdx, filteredIdx + nFiltered, [&](int16_t a, int16_t b) {
            return nm.decode(allIntStmts[a].nameId) < nm.decode(allIntStmts[b].nameId);
        });

        // --- Grow ---
        std::vector<BaseCandidate> baseCandidates;
        growBaseCandidates(body, intMemory, allIntStmts, filteredIdx, nFiltered,
            mainValidityId, targetLen,
            intMemory.normalizedEncodedSubkeysMinusOne, baseCandidates);

        // --- Merge: baseCandidates × mandatory singles ---
        for (std::size_t bi = 0; bi < baseCandidates.size(); ++bi) {
            const BaseCandidate& base = baseCandidates[bi];

            for (int16_t j = 0; j < mandatoryCount; ++j) {
                const int16_t mIdx = mandatoryIndices[j];
                const IntEncodedExpr& mIe = mandatorySrcInt[mIdx];
                const int16_t mValidityId = mIe.validityId;

                if (!nm.comparable(base.validityId, mValidityId)) continue;

                bool dup = false;
                for (int16_t k = 0; k < base.count && !dup; ++k) {
                    const IntEncodedExpr& bIe = allIntStmts[base.allIdx[k]];
                    if (bIe.originalId == mIe.originalId && nm.comparable(bIe.validityId, mIe.validityId)) {
                        dup = true;
                    }
                }
                if (dup) continue;

                // Build merged pointer array in name-sorted order
                const IntEncodedExpr* merged[ExecutionParameters::MAX_EXPRESSIONS + 1];
                int16_t mc = 0;
                const std::string& mName = nm.decode(mIe.nameId);
                bool mInserted = false;
                for (int16_t k = 0; k < base.count; ++k) {
                    const std::string& bName = nm.decode(allIntStmts[base.allIdx[k]].nameId);
                    if (!mInserted && !(bName < mName)) {
                        merged[mc++] = &mIe;
                        mInserted = true;
                    }
                    merged[mc++] = &allIntStmts[base.allIdx[k]];
                }
                if (!mInserted) merged[mc++] = &mIe;

                std::pair<bool, IntNormalizedKey> pr2 =
                    preEvaluateFromEncoded(merged, mc, body, mainValidityId,
                        intMemory.normalizedEncodedKeys);

                if (pr2.first) {
                    emitter.emit(merged, mc, pr2.second);
                }
            }
        }

        return emitter.outCount;
    }

    /// @brief Pairs variant of the static request generator — seed +
    /// grow + merge for two-mandatory-element kernel matches.
    ///
    /// @details
    /// Mirror of `generateEncodedRequestsStatic` but consumes
    /// `MandatoryPair`s (each carrying an index into `firstSrcInt` and an
    /// index into `secondSrcInt`). Grow targets length `maxKeyLen - 2`
    /// using `normalizedEncodedSubkeysMinusTwo`. Merge produces
    /// `(base + first + second)` ordered by name; emits via
    /// `StaticRequestEmitter`.
    ///
    /// @param body          Owning LB.
    /// @param intMemory     Reference hash memory.
    /// @param pairs         Mandatory-pair array.
    /// @param pairCount     Length of `pairs`.
    /// @param firstSrcInt   Source array for the *first* element of each pair.
    /// @param secondSrcInt  Source array for the *second* element of each pair.
    /// @param exprArena     Per-thread arena for stable copies.
    /// @param outBuf        Destination buffer for emitted requests.
    /// @param maxOut        Capacity of `outBuf`.
    /// @return Number of requests emitted.
    /// @see `generateEncodedRequestsStatic` — singles counterpart.
    int16_t ExpressionAnalyzer::generateEncodedRequestsStaticPairs(
        Memory& body,
        const HashMemory& intMemory,
        const MandatoryPair* pairs, int16_t pairCount,
        const IntEncodedExpr* firstSrcInt,
        const IntEncodedExpr* secondSrcInt,
        TypedArena<IntEncodedExpr>& exprArena,
        StaticRequest* outBuf, int16_t maxOut)
    {
        if (pairCount <= 0) return 0;

        NameMap& nm = body.nameMap;
        int16_t buf[ExecutionParameters::MAX_KEY_SLOTS];
        const int16_t mainValidityId = NameMap::MAIN_ID;
        const int maxKeyLen = intMemory.maxKeyLength;
        const int targetLen = std::max(0, maxKeyLen - 2);

        StaticRequestEmitter emitter(exprArena, outBuf, maxOut);

        // Pre-sort each pair by name for consistent key generation
        struct SortedPair {
            IntEncodedExpr sorted[2];
        };
        std::vector<SortedPair> sortedPairs(pairCount);
        std::vector<bool> pairValid(pairCount, true);

        for (int16_t i = 0; i < pairCount; ++i) {
            const IntEncodedExpr& e1 = firstSrcInt[pairs[i].idx1];
            const IntEncodedExpr& e2 = secondSrcInt[pairs[i].idx2];
            const std::string& n1 = nm.decode(e1.nameId);
            const std::string& n2 = nm.decode(e2.nameId);
            if (n1 <= n2) { sortedPairs[i].sorted[0] = e1; sortedPairs[i].sorted[1] = e2; }
            else          { sortedPairs[i].sorted[0] = e2; sortedPairs[i].sorted[1] = e1; }

            if (!nm.comparable(e1.validityId, e2.validityId)) pairValid[i] = false;
        }

        // --- Seed: each pair as complete 2-element key ---
        for (int16_t i = 0; i < pairCount; ++i) {
            if (!pairValid[i]) continue;
            const IntEncodedExpr* ptrs[2] = { &sortedPairs[i].sorted[0], &sortedPairs[i].sorted[1] };
            int16_t len = makeIntNormalizedKeyFromEncoded(ptrs, 2, buf,
                ExecutionParameters::MAX_KEY_SLOTS);
            IntNormalizedKey probe(2, buf, len);
            if (intMemory.normalizedEncodedKeys.find(probe) != intMemory.normalizedEncodedKeys.end()) {
                const int16_t* p = body.keyArena.store(buf, len);
                IntNormalizedKey nk(2, p, len);
                emitter.emit(ptrs, 2, nk);
            }
        }

        if (targetLen <= 0) return emitter.outCount;

        // --- Filter + sort ---
        const IntEncodedExpr* allIntStmts = body.intEncodedStatements.data();
        const int16_t allIntCount = static_cast<int16_t>(body.intEncodedStatements.size());
        int16_t filteredIdx[8192];
        int16_t nFiltered = filterIntEncodedStatements(allIntStmts, allIntCount,
            intMemory, filteredIdx, 8192);
        // gl::msvc_sort — same rationale as the matching sort in
        // generateEncodedRequestsStatic above: MSVC-introsort output
        // byte-identical across hosts. See msvc_sort.hpp.
        gl::msvc_sort(filteredIdx, filteredIdx + nFiltered, [&](int16_t a, int16_t b) {
            return nm.decode(allIntStmts[a].nameId) < nm.decode(allIntStmts[b].nameId);
        });

        // --- Grow ---
        std::vector<BaseCandidate> baseCandidates;
        growBaseCandidates(body, intMemory, allIntStmts, filteredIdx, nFiltered,
            mainValidityId, targetLen,
            intMemory.normalizedEncodedSubkeysMinusTwo, baseCandidates);

        // --- Merge: baseCandidates × mandatory pairs ---
        for (std::size_t bi = 0; bi < baseCandidates.size(); ++bi) {
            const BaseCandidate& base = baseCandidates[bi];

            for (int16_t pi = 0; pi < pairCount; ++pi) {
                if (!pairValid[pi]) continue;
                const IntEncodedExpr& m0 = sortedPairs[pi].sorted[0];
                const IntEncodedExpr& m1 = sortedPairs[pi].sorted[1];

                // m0 and m1 are already known comparable (pair filter above).
                int16_t pairVid = nm.deeperOf(m0.validityId, m1.validityId);
                if (!nm.comparable(base.validityId, pairVid)) continue;

                bool dup = false;
                for (int16_t k = 0; k < base.count && !dup; ++k) {
                    const IntEncodedExpr& bIe = allIntStmts[base.allIdx[k]];
                    if ((bIe.originalId == m0.originalId && nm.comparable(bIe.validityId, m0.validityId)) ||
                        (bIe.originalId == m1.originalId && nm.comparable(bIe.validityId, m1.validityId))) {
                        dup = true;
                    }
                }
                if (dup) continue;

                const IntEncodedExpr* merged[ExecutionParameters::MAX_EXPRESSIONS + 2];
                int16_t mc = 0;
                for (int16_t k = 0; k < base.count; ++k)
                    merged[mc++] = &allIntStmts[base.allIdx[k]];
                merged[mc++] = &m0;
                merged[mc++] = &m1;
                // gl::msvc_sort — see the matching sort in
                // generateEncodedRequestsStatic for full rationale.
                gl::msvc_sort(merged, merged + mc, [&](const IntEncodedExpr* a, const IntEncodedExpr* b) {
                    return nm.decode(a->nameId) < nm.decode(b->nameId);
                });

                std::pair<bool, IntNormalizedKey> pr2 =
                    preEvaluateFromEncoded(merged, mc, body, mainValidityId,
                        intMemory.normalizedEncodedKeys);

                if (pr2.first) {
                    emitter.emit(merged, mc, pr2.second);
                }
            }
        }

        return emitter.outCount;
    }

    // ------------------------------------------------------------------
    // Hash-engine lookup-and-fire (moved from prover.cpp in commit E).
    // The largest single body in the cluster (~300 LOC). Consumes one
    // StaticRequest produced by the singles/pairs/CE generators above and
    // either fires the matched rule (path 1, head LMV) or installs the
    // marker admission (path 2, marker LMV).
    // ------------------------------------------------------------------

    /// @brief Static-pipeline rule lookup — given a `StaticRequest` keyed on
    /// an `IntNormalizedKey`, find matching LMVs in `localHashMemory` and
    /// dispatch the head-firing path.
    ///
    /// @details
    /// This is the consumer of the LMVs that `addToHashMemory` installed.
    /// The function executes int-only fast pre-hit checks before any string
    /// work, then on a hit decodes the relevant pieces and walks the
    /// admission / OR / equivalence-class machinery.
    ///
    /// Phase outline:
    /// 1. **Validity-id consensus.** Iterate `req.intExprs[i]->validityId`
    ///    for each request element; require pairwise `nm.comparable` so a
    ///    consensus deepest scope can be picked. A divergent set returns
    ///    early (the request cannot fire under any single scope).
    /// 2. **encodedMap lookup.** Probe `memoryBlock.localHashMemory.encodedMap[tple]`.
    ///    Each candidate `LocalMemoryValue` is then admitted against:
    ///    - the request's max-iteration (`req.maxIteration + 1` cap),
    ///    - level-set admissibility (`lmv.levels`),
    ///    - the `productOfDisintegration` D-32 marker (gates OR firing),
    ///    - the OR admission set (`Memory::orAdmissionSet`),
    ///    - and per-LMV equivalence-class state. The equivalence-class
    ///      gate routes through `applyEquivalenceClass` and emits
    ///      `equality1`/`equality2` per
    ///      [I-32](../../docs/30_invariants.md#i-32) /
    ///      [I-34](../../docs/30_invariants.md#i-34).
    /// 3. **Head emit.** Each surviving LMV produces a head-firing event:
    ///    `addExprToMemoryBlock` for the head, `mailOut` for any
    ///    cross-scope deposits per [I-26](../../docs/30_invariants.md#i-26),
    ///    origin record under the LMV's `justification` tag.
    /// 4. **Admission record-keeping.** Each negative outcome (rejected
    ///    head, exhausted depth) is recorded in `rejectedMap` so the same
    ///    rejection isn't re-emitted next iteration.
    ///
    /// `coreId` is the per-core mailbox identifier; combined with the
    /// receiver's `Memory*` it picks the per-core slot of `mailOut` so
    /// `proveKernel`'s parallel phase does not need cross-LB locks
    /// ([I-28](../../docs/30_invariants.md#i-28)).
    ///
    /// @param req          The static request to look up.
    /// @param memoryBlock  Owning LB.
    /// @param coreId       Logical core id of the calling thread.
    ///
    /// @pre  `req.normalizedKey.data` references a `KeyArena` chunk of
    ///       `memoryBlock.keyArena` that is alive.
    /// @post `memoryBlock.localHashMemory` may have new origin entries;
    ///       `memoryBlock.mailOut[coreId]` may have new pending deposits;
    ///       `memoryBlock.overallHashMemory.admissionStatusMap` /
    ///       `rejectedMap` may have new rejection records.
    /// @invariant [I-12](../../docs/30_invariants.md#i-12) — `addStatement`
    ///            applies equivalence classes to `!(=[a,b])` one-sidedly;
    ///            this function honours that on the head-emit path.
    /// @invariant [I-26](../../docs/30_invariants.md#i-26) — mail-out
    ///            implications/statements MAIN-ONLY; exprOriginMap
    ///            ALL-SCOPES.
    /// @invariant [I-28](../../docs/30_invariants.md#i-28) — cross-LB
    ///            writes during the parallel phase forbidden; the head
    ///            emit defers to post-`pool.join()` collectors via
    ///            `mailOut`.
    /// @see `prover.hpp::addToHashMemory` — installer of the LMVs read here.
    /// @see `prover.hpp::addExprToMemoryBlock` — head-emit consumer.
    /// @see `prover.hpp::applyEquivalenceClass` — class-gated rewrite.
    void ExpressionAnalyzer::checkLocalEncodedMemoryStatic(
        const StaticRequest& req,
        Memory& memoryBlock,
        unsigned coreId)
    {
        NameMap& nm = memoryBlock.nameMap;
        const int16_t reqCount = req.count;
        const int iteration = req.maxIteration + 1;
        const IntNormalizedKey& tple = req.normalizedKey;

        // 1. Validity check using int fields
        const int16_t mainValidityId = NameMap::MAIN_ID;
        int16_t consensusValidityId = mainValidityId;
        for (int16_t i = 0; i < reqCount; ++i) {
            const int16_t vid = req.intExprs[i]->validityId;
            if (!nm.comparable(consensusValidityId, vid)) {
                return;
            }
            consensusValidityId = nm.deeperOf(consensusValidityId, vid);
        }

        const std::string expressionListValidityName = nm.decode(consensusValidityId);

        if (expressionListValidityName.find("_hypo_") != std::string::npos) {
            for (int16_t i = 0; i < reqCount; ++i) {
                if (req.intExprs[i]->validityId != consensusValidityId) {
                    if (!req.intExprs[i]->isAnchor) {
                        return;
                    }
                }
            }
        }

        if (memoryBlock.validityNamesToFilter.count(expressionListValidityName)) {
            return;
        }

        // 2. productsOfRecursion check using int fields
        int algebraicCounter = 0;
        const auto& prodRecIds = memoryBlock.overallHashMemory.productsOfRecursionIds;
        for (int16_t i = 0; i < reqCount; ++i) {
            const IntEncodedExpr& ex = *req.intExprs[i];
            for (int16_t a = 0; a < ex.arity; ++a) {
                if (ex.argIteration[a] > -1) {
                    if (prodRecIds.find(ex.argFullId[a]) == prodRecIds.end()) {
                        algebraicCounter++;
                    }
                }
            }
        }
        const bool pure = (algebraicCounter == 0);

        // 3. combinedLevels via statementLevelsMap (decoded lookup, per-request cost)
        std::set<int> combinedLevels;
        for (int16_t i = 0; i < reqCount; ++i) {
            EncodedExpression lookup;
            lookup.original = nm.decode(req.intExprs[i]->originalId);
            lookup.validityName = nm.decode(req.intExprs[i]->validityId);
            auto itL = memoryBlock.statementLevelsMap.find(lookup);
            if (itL != memoryBlock.statementLevelsMap.end()) {
                for (const auto& lv : itL->second) combinedLevels.insert(lv);
            }
        }

        // 4. Build intAllArgs from int fields
        std::set<int16_t> intAllArgs;
        for (int16_t i = 0; i < reqCount; ++i) {
            const IntEncodedExpr& ex = *req.intExprs[i];
            for (int16_t a = 0; a < ex.arity; ++a) {
                intAllArgs.insert(ex.argFullId[a]);
            }
        }

        if (memoryBlock.overallHashMemory.encodedMap.empty()) {
            return;
        }

        // 5. Candidate loop: find remaining-arg sets that are subsets of intAllArgs (sorted for determinism)
        std::vector<std::set<int16_t>> candidates;
        for (const auto& kv : memoryBlock.overallHashMemory.remainingArgsNormalizedEncodedMap) {
            const std::set<int16_t>& st = kv.first;
            bool subsetOk = true;
            for (const auto& arg : st) {
                if (intAllArgs.find(arg) == intAllArgs.end()) {
                    subsetOk = false; break;
                }
            }
            if (subsetOk) candidates.push_back(st);
        }
        std::sort(candidates.begin(), candidates.end());

        for (const auto& intSt : candidates) {
            auto itRA = memoryBlock.overallHashMemory.remainingArgsNormalizedEncodedMap.find(intSt);
            if (itRA == memoryBlock.overallHashMemory.remainingArgsNormalizedEncodedMap.end()) continue;

            const std::unordered_set<IntNormalizedKey, IntNormalizedKeyHash>& nkSet = itRA->second;
            if (nkSet.find(tple) == nkSet.end()) {
                continue;
            }

            int16_t buf[ExecutionParameters::MAX_KEY_SLOTS];
            int16_t reverseMap[ExecutionParameters::MAX_KEY_SLOTS];
            std::memset(reverseMap, 0, sizeof(reverseMap));
            int16_t numNormVars = 0;

            // Build pointer array from request
            const IntEncodedExpr* reqExprs[ExecutionParameters::MAX_EXPRESSIONS];
            for (int16_t i = 0; i < reqCount; ++i) reqExprs[i] = req.intExprs[i];

            const int16_t len = makeIntNormalizedKeyFromEncodedWithMap(
                reqExprs, reqCount, intSt,
                buf, ExecutionParameters::MAX_KEY_SLOTS,
                reverseMap, numNormVars);

            IntNormalizedKey probe(reqCount, buf, len);
            auto itEnc = memoryBlock.overallHashMemory.encodedMap.find(probe);
            if (itEnc == memoryBlock.overallHashMemory.encodedMap.end()) {
                continue;
            }

            // HIT — build string back-replacement map from int reverseMap
            std::map<std::string, std::string> backReplacementMap;
            for (int16_t v = 1; v <= numNormVars; ++v) {
                backReplacementMap[std::to_string(static_cast<int>(v))] = nm.decode(reverseMap[v]);
            }

            const std::vector<LocalMemoryValue>& valueVec = itEnc->second;

            std::vector<LocalMemoryValue> ordered(valueVec.begin(), valueVec.end());
            struct ByValue {
                bool operator()(const LocalMemoryValue& a, const LocalMemoryValue& b) const {
                    return a.value < b.value;
                }
            };
            std::sort(ordered.begin(), ordered.end(), ByValue());

            // Build origin tail — preserve each premise's real validity.
            // Mirror of the fix in checkLocalEncodedMemory: decode both
            // originalId AND validityId per premise instead of flattening to
            // strings and restamping with expressionListValidityName.
            std::vector<ExpressionWithValidity> orderedPremises;
            orderedPremises.reserve(reqCount);
            for (int16_t i = 0; i < reqCount; ++i) {
                orderedPremises.emplace_back(
                    nm.decode(req.intExprs[i]->originalId),
                    nm.decode(req.intExprs[i]->validityId));
            }
            std::sort(orderedPremises.begin(), orderedPremises.end(),
                [](const ExpressionWithValidity& a, const ExpressionWithValidity& b) {
                    if (a.original != b.original) return a.original < b.original;
                    return a.validityName < b.validityName;
                });

            for (std::size_t vi = 0; vi < ordered.size(); ++vi) {
                const LocalMemoryValue& lmv = ordered[vi];

                const std::string rplExpr1 = ce::replaceKeysInString(lmv.value, backReplacementMap);
                const std::string rplExpr2 = ExpressionAnalyzer::replaceUSubstrings(rplExpr1);

                if (lmv.value.find("marker") == std::string::npos) {
                    // Cross-scope validity check (D-55).
                    // Pre-fix: strict equality `lmv.validityName != expressionListValidityName`
                    // rejected legitimate firings where the rule lives at an ancestor scope of
                    // the consensus (e.g. K mutual-exclusion implications registered at the OR's
                    // parent scope, fact at an OR-introduction subproof descendant). Pre-fix the
                    // chapter-6 mirror was a v=main universal so the strict check was masked by
                    // the "lmv.validityName != main" early return; with parents removed and the
                    // K-impl carrying a non-main scope, the mask disappears and the check rejects
                    // valid comparable-scope firings. Mirrors the request-generation-side rule
                    // documented in `02_hash_engine.md` §Locality semantics.
                    const int16_t lmvVid = nm.encode(lmv.validityName);
                    if (!nm.comparable(lmvVid, consensusValidityId)) {
                        continue;
                    }

                    std::set<int> tempLevels = combinedLevels;
                    for (const auto& lv : lmv.levels) tempLevels.insert(lv);

                    std::pair<std::string, std::vector<ExpressionWithValidity>> origin =
                        std::make_pair("implication", std::vector<ExpressionWithValidity>());
                    origin.second.push_back(ExpressionWithValidity(lmv.originalImplication, lmv.validityName));
                    for (const auto& p : orderedPremises) {
                        origin.second.push_back(p);
                    }

                    bool doNotDisintegrate = (lmv.justification == "integration");
                    // D-29 two-part disintegration gate (only active when
                    // incubator_mode && !ban_disintegration — the SE2 /
                    // FTA-rung-1 combination):
                    //   (1) anchor LB: ALWAYS block disintegration (anchor never
                    //       disintegrates in this mode — its job is broadcast +
                    //       handle external rules without local fan-out).
                    //   (2) non-anchor LB: block disintegration unless at least
                    //       one matched premise is in this LB's
                    //       localEncodedStatementsSet (i.e. the rule was triggered
                    //       at least partly by THIS LB's own derivation work, not
                    //       solely by anchor / broadcast inputs).
                    // Without (1) the prover crashed at burst 2 / ~4570 exprs.
                    // Without (2) the prover crashed at burst 3 / ~7686 exprs.
                    // Both clauses are load-bearing.
                    // Pre-D-29 the prover crashed with NameMap exhaustion (cap
                    // 16384) at prover.hpp:807 — fold/sequence/etc. external
                    // rules from the prior IncubatorGauss batch fanning out
                    // across local (in[X, N]) rows.
                    if (parameters.incubator_mode && !parameters.ban_disintegration && !doNotDisintegrate) {
                        const std::string anchorPrefix = "(" + this->anchorInfo.name;
                        const bool isAnchorLB =
                            (memoryBlock.exprKey.rfind(anchorPrefix, 0) == 0);
                        if (isAnchorLB) {
                            doNotDisintegrate = true;
                        } else {
                            bool hasLocalPremise = false;
                            for (const auto& p : orderedPremises) {
                                if (memoryBlock.localEncodedStatementsSet.count(
                                        EncodedExpression(p.original, p.validityName))) {
                                    hasLocalPremise = true;
                                    break;
                                }
                            }
                            if (!hasLocalPremise) doNotDisintegrate = true;
                        }
                    }

                    bool allGood = expressionListValidityName == "main";
                    for (int16_t i = 0; i < reqCount; ++i) {
                        if (!allowedForMail(nm.decode(req.intExprs[i]->originalId), memoryBlock)) {
                            allGood = false;
                        }
                    }
                    if (allGood) {
                        memoryBlock.canBeSentSet.insert(rplExpr2);
                    }

                    // D-32: forward the install-time disintegration-product
                    // flag so disintegrateExprCore2's OR case can fire the
                    // sharper bypass at the OR-disint admission gate.
                    this->addExprToMemoryBlock(rplExpr2, memoryBlock, iteration, 1,
                        tempLevels, origin, coreId, -1, expressionListValidityName, doNotDisintegrate,
                        lmv.productOfDisintegration);
                }
                else {
                    // Purity gate is relaxed ONLY inside OR-branch scopes
                    // (`_boundary_orint_` in the validity name). Case-split
                    // branches legitimately carry fresh it_/int_ witnesses
                    // that are iteration-bearing but not yet products of
                    // recursion — denying those admissions leaves §4.2-style
                    // ladder steps unreachable.
                    // Anywhere else (main / hypo / non-or boundary) the
                    // original purity gate stands, to prevent runaway fan-out.
                    const bool inOrBranch =
                        expressionListValidityName.find("_boundary_orint_")
                        != std::string::npos;
                    if (!pure && !inOrBranch) {
                        continue;
                    }

                    std::vector<std::string> replacedKey;
                    replacedKey.reserve(lmv.key.size());
                    for (std::size_t k = 0; k < lmv.key.size(); ++k) {
                        const std::string tmp = ce::replaceKeysInString(lmv.key[k], backReplacementMap);
                        replacedKey.push_back(ExpressionAnalyzer::replaceUSubstrings(tmp));
                    }

                    std::set<std::string> remainingArgs = lmv.remainingArgs;

                    auto itAdm = memoryBlock.overallHashMemory.admissionMap.find(
                        ExpressionWithValidity(rplExpr2, expressionListValidityName));
                    AdmissionMapValue admv;
                    admv.key = replacedKey;
                    admv.remainingArgs = remainingArgs;
                    admv.standardMaxAdmissionDepth = parameters.standardMaxAdmissionDepth;
                    admv.standardMaxSecondaryNumber = parameters.standardMaxSecondaryNumber;
                    admv.flag = false;

                    if (memoryBlock.overallHashMemory.consumedAdmissionKeys.find(
                        ExpressionWithValidity(rplExpr2, expressionListValidityName))
                        != memoryBlock.overallHashMemory.consumedAdmissionKeys.end()) {
                        continue;
                    }

                    std::vector<std::string> tempArgs = ce::getArgs(rplExpr2);
                    std::set<std::string> argSet(tempArgs.begin(), tempArgs.end());
                    assert(argSet.find("marker") != argSet.end());
                    argSet.erase("marker");

                    prepareIntegration(rplExpr2, argSet, memoryBlock, expressionListValidityName);

                    std::string core = ce::extractExpression(rplExpr2);
                    auto itComp = compiledExpressions.find(core);
                    assert(itComp != compiledExpressions.end());
                    if (itComp->second.category != "atomic") {
                        memoryBlock.canBeSentMarkerSet.insert(rplExpr2);
                    }

                    if (itAdm != memoryBlock.overallHashMemory.admissionMap.end()) {
                        itAdm->second.insert(admv);
                    } else {
                        std::set<AdmissionMapValue> newSet;
                        newSet.insert(admv);
                        memoryBlock.overallHashMemory.admissionMap.insert(
                            std::make_pair(ExpressionWithValidity(rplExpr2, expressionListValidityName), newSet));
                    }

                    memoryBlock.overallHashMemory.admissionStatusMap[
                        ExpressionWithValidity(rplExpr2, expressionListValidityName)] = false;

                    // Populate vars-in-keys cache for the algebra equi-class hook.
                    for (const std::string& a : ce::getArgs(rplExpr2)) {
                        if (a != "marker") {
                            memoryBlock.overallHashMemory.varsInAdmissionMapKeys.insert(a);
                        }
                    }

                    this->revisitRejected2(rplExpr2, memoryBlock, expressionListValidityName);
                }
            }
        }
    }

    /// @brief Pairs variant of the mandatory-element filter — emits index
    /// pairs whose merged subkey is mutually mandatory in `mem`.
    ///
    /// @details
    /// Two-layer counterpart to `makeMandatoryEncodedStatementLists1Static`.
    /// Both `first` and `second` are first filtered through
    /// `filterIntEncodedStatements`; the surviving indices are sorted by
    /// `originalId`; then each candidate pair `(i ∈ first, j ∈ second)` is
    /// tested for mutual mandatoryness against `mem`'s
    /// `normalizedEncodedSubkeysMinusOne` set. Surviving pairs are written
    /// as `MandatoryPair{ idx1: i, idx2: j }` into `outPairs`, capped at
    /// `maxOut`.
    ///
    /// O(n²) over the filtered counts; in practice both filters cut the
    /// statement count by an order of magnitude or two, so the inner loop
    /// stays bounded.
    ///
    /// @param body         Owning LB; supplies `nameMap`.
    /// @param mem          Reference hash memory.
    /// @param first        First-layer source array.
    /// @param firstCount   Length of `first`.
    /// @param second       Second-layer source array.
    /// @param secondCount  Length of `second`.
    /// @param outPairs     Destination buffer for emitted pairs.
    /// @param maxOut       Capacity of `outPairs`.
    /// @return Count of pairs emitted into `outPairs`.
    /// @pre  `first` and `second` live for the duration of the call.
    /// @post `outPairs[0..return-1]` contains the mutually-mandatory pairs.
    /// @see `generateEncodedRequestsStaticPairs` — primary consumer.
    int16_t ExpressionAnalyzer::makeMandatoryEncodedStatementLists2Static(
        Memory& body, const HashMemory& mem,
        const IntEncodedExpr* first, int16_t firstCount,
        const IntEncodedExpr* second, int16_t secondCount,
        MandatoryPair* outPairs, int16_t maxOut)
    {
        if (firstCount == 0 || secondCount == 0) return 0;

        // Filter both layers
        int16_t filt1Buf[4096], filt2Buf[4096];
        int16_t nF1 = filterIntEncodedStatements(first, firstCount, mem, filt1Buf, 4096);
        int16_t nF2 = filterIntEncodedStatements(second, secondCount, mem, filt2Buf, 4096);

        // Sort by originalId
        auto sortByOriginal = [](int16_t* arr, int16_t n, const IntEncodedExpr* stmts) {
            for (int16_t i = 1; i < n; ++i) {
                int16_t key = arr[i];
                int16_t j = i - 1;
                while (j >= 0 && stmts[arr[j]].originalId > stmts[key].originalId) {
                    arr[j + 1] = arr[j];
                    --j;
                }
                arr[j + 1] = key;
            }
        };
        sortByOriginal(filt1Buf, nF1, first);
        sortByOriginal(filt2Buf, nF2, second);

        // "main" ID is guaranteed to be NameMap::MAIN_ID (== 1).
        int16_t mainId = NameMap::MAIN_ID;

        int16_t nOut = 0;
        int16_t buf[ExecutionParameters::MAX_KEY_SLOTS];

        for (int16_t i = 0; i < nF1 && nOut < maxOut; ++i) {
            const IntEncodedExpr& e1 = first[filt1Buf[i]];
            for (int16_t j = 0; j < nF2 && nOut < maxOut; ++j) {
                const IntEncodedExpr& e2 = second[filt2Buf[j]];

                // Validity check
                if (e1.validityId != e2.validityId) {
                    if (e1.validityId != mainId && e2.validityId != mainId) continue;
                }

                // Skip identical expressions
                if (e1.originalId == e2.originalId && e1.validityId == e2.validityId)
                    continue;

                // Build sorted pair and preEvaluate — sort by name string (lexicographic),
                // NOT by nameId (insertion order), to match storage sort in addToHashMemory
                IntEncodedExpr sorted[2];
                const std::string& name1 = body.nameMap.decode(e1.nameId);
                const std::string& name2 = body.nameMap.decode(e2.nameId);
                if (name1 <= name2) { sorted[0] = e1; sorted[1] = e2; }
                else { sorted[0] = e2; sorted[1] = e1; }

                int16_t len = makeIntNormalizedKeyFromEncoded(sorted, 2, false, buf,
                    ExecutionParameters::MAX_KEY_SLOTS);
                IntNormalizedKey probe(2, buf, len);

                if (mem.normalizedEncodedSubkeys.find(probe) != mem.normalizedEncodedSubkeys.end()) {
                    outPairs[nOut].idx1 = filt1Buf[i];
                    outPairs[nOut].idx2 = filt2Buf[j];
                    ++nOut;
                }
            }
        }
        return nOut;
    }

} // namespace gl
