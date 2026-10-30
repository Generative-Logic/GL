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
#include "admission_marker_builder.hpp"
#include "../inference/hash_key_builder.hpp"
#include "../inference/hash_rule_index.hpp"
#include "../infra/hashburst_dump.hpp"
#include "../infra/rt_tracker.hpp"
#include "../proof/worker_context.hpp"
#include <fstream>
#include <optional>
#include <sstream>
namespace gl {
/// @brief Bind the inputs required to generate admission-marker rules.
/// @details All references outlive this algorithm; construction installs no rules.
/// @param definitions Read-only operator and configuration definitions.
/// @param admissionPolicy Admission classification operations.
/// @param parameters Existing length thresholds.
/// @param allPermutationsAna Precomputed premise permutations, read without mutation.
/// @return Constructed marker generator.
AdmissionMarkerBuilder::AdmissionMarkerBuilder(const CompiledDefinitionRegistry& definitions, const AdmissionPolicy& admissionPolicy, const ProverParameters& parameters, const std::map<int, std::vector<std::vector<int>>>& allPermutationsAna)
 : definitions(definitions), admissionPolicy(admissionPolicy), parameters(parameters), allPermutationsAna(allPermutationsAna) {}

/// @brief Install one admission-marker rule variant family — the
        ///        shared interior of @ref makeNormalizedKeysForAdmission's
        ///        regular accept path and the ordis candidate loop, and the
        ///        door of the goal-driven marked-head implication
        ///        (`HashRuleInstaller::installGoalDemandVariant`).
        ///
        /// @details
        /// For each accepted element, the variant installer
        /// builds the marker form of `key[index]` (@p markedArg replaced by
        /// the literal `marker`), installs the int subkey owner records
        /// (`normalizedEncodedSubkeys`),
        /// then the permutation main variants — normalized key buffers, the
        /// reverse-map decimal rename, the marker LMV via `appendLmvIdsRecord`
        /// (carrying @p ordisOnly), the `normalizedEncodedKeys` owner record,
        /// and the batched remaining-args index insert. The regular and OR routes share key construction; the OR route
        /// marks its records with the corresponding rule tag.
        ///
        /// @param key           The full key element run.
        /// @param n             The key element count.
        /// @param index         The marked premise's index in @p key.
        /// @param markedArg     The argument replaced by `marker` (the regular
        ///                      route's output arg / the ordis route's
        ///                      input-slot candidate); a slice of `key[index]`.
        /// @param binary        Per-element subkey membership mask (1 = in the
        ///                      subkey), as built by the caller.
        /// @param subkey        The subkey elements (spans into @p key).
        /// @param subCount      The subkey element count.
        /// @param intHashMemory Target hash memory.
        /// @param nameMap       `NameMap` for int16 encoding.
        /// @param ruleInterner  Rule-registry value interner.
        /// @param originalImpl  The admitting implication span (D-72 owner).
        /// @param ownerVid      The owner scope's NameMap id.
        /// @param ordisOnly     The LMV tag: true only for ordis-route
        ///                      installs.
        /// @param ordis2Demand  The LMV tag: true only for the qualifying-slot
        ///                      demand-variant install
        ///                      (D-267). The value
        ///                      stays VERBATIM (no marker substitution) but
        ///                      still decimal-renamed with the subkey's
        ///                      normalized variables: changeable args
        ///                      back-substitute at fire, unchangeable u_
        ///                      args match literally and the firing path's
        ///                      u_-strip grounds them. The LMV is an OWN
        ///                      kind — isMarker stays false (no marker
        ///                      anywhere in the ordis2 machinery); the
        ///                      firing dispatch checks ordis2Demand FIRST,
        ///                      so it never fires as a head.
        /// @param op            Install or the removal policy.
        /// @param memo          The per-copy premise encode memo bound to
        ///                      @p key / @p n; every subkey element is
        ///                      `key[i]` for a masked `i` (asserted), so the
        ///                      installer reads its encodes through the mask
        ///                      instead of re-encoding the text per
        ///                      permutation and prefix.
        /// @param staging       The rule-index staging serving
        ///                      @p intHashMemory on this slot.
        /// @param demandJustification The justification stamped on the
        ///                      LMV: `integration` for a demand extracted
        ///                      from an integration-justified rule,
        ///                      `goalDemand` for the goal near-miss demand
        ///                      and the goal-driven marked-head implication
        ///                      (D-574),
        ///                      `none` (the marker default) otherwise. Read
        ///                      by the firing site: its demand branch
        ///                      exempts `integration` and `goalDemand`
        ///                      demands from the purity gate (`D-357`), its
        ///                      marker branch exempts `goalDemand` markers.
        ///                      Default `none` — existing calls byte-stable.
        /// @return (void)
        /// @invariant Interner-touch sequence identical to the pre-extraction
        ///            inline code (the frozen mint order of the admission
        ///            build); scratch rides the per-slot string / gen arenas
        ///            under this function's own scopes (Rule 28).
        /// @invariant An install leaves `intHashMemory.maxKeyLength` at least
        ///            @p subCount, the premise count of the whole keys it
        ///            stages: request growth stops at that bound
        ///            (I-337).
        /// @see makeNormalizedKeysForAdmission — the six call sites of the
        ///      admission build; `HashRuleInstaller::installGoalDemandVariant`;
        ///      `appendLmvIdsRecord` — the LMV door; `KeyEncodeMemo`.
/// @param owner Rule and scope owning the installed variant.
void AdmissionMarkerBuilder::installAdmissionMarkerVariants(
        const StrSpan* key, std::size_t n,
        std::size_t index, StrSpan markedArg,
        const int* binary, const StrSpan* subkey, int32_t subCount,
        HashMemory& intHashMemory,
        NameMap& nameMap,
        ValueInterner& ruleInterner,
        StrSpan originalImpl,
        NameId ownerVid,
        RuleOwner owner,
        bool ordisOnly,
        bool ordis2Demand,
        const RuleIndexOp& op,
        KeyEncodeMemo& memo,
        RuleIndexStaging& staging,
        RuleJustification demandJustification) {
        {
            constexpr int32_t kKeyCap =
                ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 1;
            assert(n <= static_cast<std::size_t>(kKeyCap)
                && "installAdmissionMarkerVariants: key size exceeds cap");
            assert(memo.key == key && memo.keyN == static_cast<int32_t>(n)
                && "installAdmissionMarkerVariants: the memo is bound to another key run");

            // subkey[j] is key[subIdx[j]] — the j-th premise the caller's
            // mask selected, in key order (the span identity is asserted) —
            // so the memo is read through the mask.
            int32_t subIdx[kKeyCap];
            {
                int32_t j = 0;
                for (std::size_t i = 0; i < n; ++i) {
                    if (!binary[i]) continue;
                    assert(j < subCount
                        && "installAdmissionMarkerVariants: more masked premises than subkey elements");
                    assert(subkey[j].ptr == key[i].ptr && subkey[j].len == key[i].len
                        && "installAdmissionMarkerVariants: a subkey element is not the masked premise");
                    subIdx[j++] = static_cast<int32_t>(i);
                }
                assert(j == subCount
                    && "installAdmissionMarkerVariants: mask and subkey disagree");
            }

            // Replace the marked arg with "marker" -> a ScratchString on the
            // string tier (replaceKeysScratch, byte-exact to
            // ce::replaceKeysInString), held across both blocks by mnkStrScope.
            // The ordis2-demand route keeps the value VERBATIM instead — no
            // marker token; changeable args ride the decimal rename below
            // and back-substitute at fire, unchangeable u_ args match
            // literally and the firing path's u_-strip grounds them — the
            // fired demand is ground either way.
            // Per-registry slot derivation (the two registries differ by one
            // slot; each fallback comes from its OWN slotCount() — see the
            // addToHashMemory site note).
            const unsigned mnkStrSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : scratchArenas().slotCount() - 1;
            const unsigned mnkGenSlot = (gl::WorkerContext::coreId() >= 0)
                ? static_cast<unsigned>(gl::WorkerContext::coreId())
                : genScratchArenas().slotCount() - 1;
            ScratchArena& mnkStrArena = scratchArenas().forSlot(mnkStrSlot);
            ScratchScope mnkStrScope(mnkStrArena);
            StrReplacement markerPair[1];
            markerPair[0].key = ordis2Demand ? StrSpan() : markedArg;
            markerPair[0].value = StrSpan("marker", 6);
            const ScratchString replaced = ordis2Demand
                ? ScratchString::copyFrom(mnkStrArena, key[index].ptr,
                                          key[index].len)
                : replaceKeysScratch(mnkStrArena, key[index], markerPair, 1);

            // --- Int subkeys (mirrors makeNormalizedSubkeys for HashMemory) ---
            {
                const int32_t sn = subCount;
                auto pit = this->allPermutationsAna.find(sn);
                if (pit != this->allPermutationsAna.end()) {
                    RT_SCOPE_HERE("ADM_SUBKEYS");
                    StrSpan sids[kKeyCap];
                    for (int32_t i = 0; i < sn; ++i)
                        sids[i] = extractExpressionSpan(subkey[i]);

                    // Every operator-name-sorted prefix of the subkey's
                    // permutations, each built and staged once, in the order
                    // the permutation table first reaches it
                    // (HashKeyBuilder::forEachSortedPrefix). Premises from the
                    // per-copy memo (through the mask); a prefix's normalized
                    // key extends its parent's by one premise — after a prefix
                    // of length L the first `st.length()` slots are its subkey
                    // (the resumable builder's contract) — and leaving the
                    // prefix restores the parent's checkpoint. subEncoded[0..si]
                    // is the prefix in key order for the signature record. A
                    // prefix the table reaches again would stage the same
                    // contribution, which the flush's sorted-unique merge drops.
                    IntEncodedExpr subEncoded[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                    NameId subBuf[ExecutionParameters::MAX_KEY_SLOTS];
                    gl::HashKeyBuilder st;
                    std::optional<gl::HashKeyBuilder::Checkpoint> parentKey[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                    int64_t prefixesVisited = 0;
                    gl::HashKeyBuilder::forEachSortedPrefix(sids, sn,
                        [&](int32_t si, const int32_t* permut) {
                            ++prefixesVisited;
                            parentKey[si].emplace(st.checkpoint());

                            const NameId subEncCount = static_cast<NameId>(si + 1);
                            assert(subEncCount <= ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                                && "makeNormalizedKeysForAdmission subkey: element count exceeds cap");
                            {
                            RT_SCOPE_HERE("ADM_SK_ENCODE");
                            subEncoded[si] = memo.at(subIdx[permut[si]], nameMap);
                            }

                            NameId subLen = 0;
                            {
                            RT_SCOPE_HERE("ADM_SK_NORMKEY");
                            st.append(subEncoded[si], subBuf, ExecutionParameters::MAX_KEY_SLOTS);
                            subLen = st.length();
                            }
                            // Subkey map via the raw-key door (no owning NormKey):
                            // the owner on the signature-free record below
                            // kSubkeyUCheckMinElements, on the u_-signature
                            // record from there on; under the remove policy the
                            // owner's pairs leave the record.
                            if (op.isRemove()) {
                                gl::HashRuleIndex::removeRuleSubkeyEntry(intHashMemory, op,
                                    static_cast<NameId>(si + 1), subBuf, subLen,
                                    owner, genScratchArenas().forSlot(mnkGenSlot));
                            } else if (subEncCount < gl::HashRuleIndex::kSubkeyUCheckMinElements) {
                                RT_SCOPE_HERE("ADM_SK_WRITE_SHORT");
                                gl::HashRuleIndex::stageEmptySubkeyRecord(staging.subkeys,
                                    static_cast<NameId>(si + 1), subBuf, subLen);
                            } else {
                                RT_SCOPE_HERE("ADM_SK_WRITE_MERGE");
                                gl::HashRuleIndex::stageSubkeySignatures(staging.subkeys,
                                    static_cast<NameId>(si + 1), subBuf, subLen,
                                    subEncoded, subEncCount, owner);
                            }
                        },
                        [&](int32_t si) { st.restore(*parentKey[si]); });
                    RT_NOTE_ITERATIONS_HERE(prefixesVisited);   // sorted prefixes staged
                }
            }

            // --- Main variants: create int keys for encodedMap marker entries ---
            {
                const int32_t sn = subCount;
                auto pit = this->allPermutationsAna.find(sn);
                if (pit == this->allPermutationsAna.end()) return;
                RT_SCOPE_HERE("ADM_WHOLEKEY");

                StrSpan sids[kKeyCap];
                for (int32_t i = 0; i < sn; ++i)
                    sids[i] = extractExpressionSpan(subkey[i]);

                ScratchArena& mnkGenArena = genScratchArenas().forSlot(mnkGenSlot);

                // getRemainingArgs yields a compareSpans sorted-unique run == the
                // former std::set<std::string> order. MINT the NameMap ids in that
                // run order (I-84: mint order reaches the deload stream), then a
                // SORTED-ASCENDING copy is the Int16SetKey (the former
                // std::set<NameId> iteration order). The lmv.remainingArgIds mint
                // (ruleInterner) walks the run order = former set-lex order.
                StrSpan remScratch[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
                const int32_t remScratchN = gl::HashKeyBuilder::getRemainingArgs(
                    subkey, subCount, remScratch,
                    ExecutionParameters::MAX_ADMISSION_REM_ARGS);
                NameId intRemArgs[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
                for (int32_t i = 0; i < remScratchN; ++i)
                    intRemArgs[i] = nameMap.encode(remScratch[i]);   // mint in run order
                NameId intRemArgsSorted[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
                if (remScratchN > 0)
                    std::memcpy(intRemArgsSorted, intRemArgs,
                        static_cast<std::size_t>(remScratchN) * sizeof(NameId));
                std::sort(intRemArgsSorted, intRemArgsSorted + remScratchN);

                NameId bufIgnored[ExecutionParameters::MAX_KEY_SLOTS];
                NameId bufNotIgnored[ExecutionParameters::MAX_KEY_SLOTS];

                // Batch the remaining-args insert across the permutation loop:
                // the key (intRemArgsSorted/remScratchN) is loop-invariant, so
                // accumulate each permutation's NormKey and do ONE RMW after the
                // loop (D-199). The byte-bump blobs survive
                // the loop's self-framing appendLmvIdsRecord.
                const ArenaOffset remArgsBatchMark = mnkGenArena.cursor();
                DirtyState remArgsBatchDirty = DirtyState::Clean;
                PagedVector<gl::RemArgsBatchBlob> remArgsBatch(
                    &mnkGenArena, &remArgsBatchDirty);

                // The permutations of the table whose operator names never
                // decrease, in table order, without walking the other sn! ones.
                int64_t orderingsVisited = 0;
                gl::HashKeyBuilder::forEachSortedOrdering(sids, sn, [&](const int32_t* permutation) {
                    ++orderingsVisited;

                    // Per-permutation string window (valueVariant / replKey / decimals).
                    ScratchScope permScope(mnkStrArena);

                    // Span-native int encode (no heap EncodedExpression -> no
                    // parseArgument). The ignored buffer uses the WithMap variant
                    // (byte-identical outBuf) so its reverseMap can rebuild the
                    // value-variant mapping without makeNormalizedEncodedKey.
                    IntEncodedExpr intEncoded[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                    const NameId intEncCount = static_cast<NameId>(sn);
                    assert(intEncCount <= ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                        && "makeNormalizedKeysForAdmission: subkey element count exceeds cap");
                    {
                    RT_SCOPE_HERE("ADM_WK_ENCODE");
                    for (int32_t k = 0; k < sn; ++k)
                        intEncoded[k] = memo.at(subIdx[permutation[k]], nameMap);
                    }

                    NameId reverseMap[ExecutionParameters::MAX_KEY_SLOTS];
                    std::memset(reverseMap, 0, sizeof(reverseMap));
                    NameId numNormVars = 0;
                    NameId lenIgnored = 0;
                    NameId lenNotIgnored = 0;
                    {
                    RT_SCOPE_HERE("ADM_WK_NORMKEY");
                    lenIgnored = gl::HashKeyBuilder::makeIntNormalizedKeyFromEncodedWithMap(
                        intEncoded, intEncCount, true, bufIgnored,
                        ExecutionParameters::MAX_KEY_SLOTS, reverseMap, numNormVars);
                    lenNotIgnored = gl::HashKeyBuilder::makeIntNormalizedKeyFromEncoded(
                        intEncoded, intEncCount, false, bufNotIgnored, ExecutionParameters::MAX_KEY_SLOTS);
                    }

                    // Remove policy: this permutation's whole key and
                    // remaining-args edge lose the owner (first visit only);
                    // no marker LMV, no batch.
                    if (op.isRemove()) {
                        gl::HashRuleIndex::removeRuleKeyEntries(intHashMemory, op,
                            static_cast<NameId>(sn), bufNotIgnored, lenNotIgnored,
                            intRemArgsSorted, remScratchN, owner, mnkGenArena);
                        return;
                    }

                    // Rename run { normalized-var name -> decimal id } from
                    // reverseMap. mp2 order is NOT observable (replaceKeysScratch is
                    // greedy-longest, order-independent), so id order suffices —
                    // byte-identical to the former lex-ordered std::map. The key
                    // spans are NameMap decodeView: no NameMap mint falls between
                    // here and the last replaceKeysScratch use (the mints below are
                    // ruleInterner), so decodeView is I-3-safe (proved per-site).
                    StrReplacement mp2Pairs[ExecutionParameters::MAX_KEY_SLOTS];
                    int32_t mp2PairsN = 0;
                    int32_t valueId = 0;
                    bool isMarker = false;
                    {
                    RT_SCOPE_HERE("ADM_WK_VALUE_VARIANT");
                    for (NameId id = 1; id <= numNormVars; ++id) {
                        assert(mp2PairsN < ExecutionParameters::MAX_KEY_SLOTS
                            && "makeNormalizedKeysForAdmission valueVariant: rename pair count exceeds cap");
                        char decTmp[12];
                        const int32_t decN = writeDecimalDigits(decTmp, id);
                        char* decBuf = mnkStrArena.allocBytes(decN);
                        std::memcpy(decBuf, decTmp, static_cast<std::size_t>(decN));
                        mp2Pairs[mp2PairsN].key = nameMap.decodeView(reverseMap[id]);
                        mp2Pairs[mp2PairsN].value = StrSpan(decBuf, decN);
                        ++mp2PairsN;
                    }

                    const ScratchString valueVariant =
                        replaceKeysScratch(mnkStrArena, StrSpan(replaced), mp2Pairs, mp2PairsN);

                    // Marker value id + isMarker flag (id-run door, no owning
                    // LMV). An ordis2-demand value carries no marker token
                    // and is NOT flagged isMarker — the demand kind is its
                    // own (maintainer decision 2026-08-10); the firing
                    // dispatch checks ordis2Demand FIRST, so the head
                    // branch never fires it.
                    valueId = ruleInterner.encode(StrSpan(valueVariant));
                    isMarker =
                        containsSpan(StrSpan(valueVariant), StrSpan("marker", 6));
                    } // RT_SCOPE ADM_WK_VALUE_VARIANT

                    // replKey ids: each valid subkey element (binary[i], subkey only
                    // — never key[index]) with the rename applied, minted in binary
                    // order == the former replKey vector order.
                    int32_t keyIds[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
                    int32_t keyIdsN = 0;
                    {
                    RT_SCOPE_HERE("ADM_WK_KEYIDS");
                    for (std::size_t i = 0; i < n; ++i) {
                        if (binary[i]) {
                            assert(keyIdsN < ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                                && "makeNormalizedKeysForAdmission replKey: element count exceeds cap");
                            const ScratchString e = replaceKeysScratch(
                                mnkStrArena, key[i], mp2Pairs, mp2PairsN);
                            keyIds[keyIdsN++] = ruleInterner.encode(StrSpan(e));
                        }
                    }
                    } // RT_SCOPE ADM_WK_KEYIDS

                    // remainingArgIds: walk remScratch (run order == former set-lex).
                    int32_t remIds[ExecutionParameters::MAX_ADMISSION_REM_ARGS];
                    int32_t originalImplId = 0;
                    {
                    RT_SCOPE_HERE("ADM_WK_LMV_FIELDS");
                    for (int32_t i = 0; i < remScratchN; ++i)
                        remIds[i] = ruleInterner.encode(remScratch[i]);
                    originalImplId = ruleInterner.encode(originalImpl);
                    }

                    // encodedMap marker record + owner record + remaining-args index,
                    // all via the raw / id-run doors (no owning NormKey / LMV). D-72:
                    // the LMV carries the admitting implication + scope. The
                    // trailing defaults are spelled out to reach the ordisOnly
                    // slot; regular installs pass false — byte-identical.
                    {
                    RT_SCOPE_HERE("ADM_LMV_APPEND");
                    stageLmvIdsRecord(staging.lmv,
                        static_cast<NameId>(sn), bufIgnored, lenIgnored,
                        valueId, isMarker, keyIds, keyIdsN, remIds, remScratchN,
                        originalImplId, ownerVid, mnkGenArena,
                        nullptr, 0, demandJustification, false, true,
                        ordisOnly, /*subsetExclusion=*/false, ordis2Demand);
                    }
                    {
                    RT_SCOPE_HERE("ADM_WHOLEKEY_OWNER");
                    gl::HashRuleIndex::stageWholeKeyOwner(staging.wholeKeys,
                        static_cast<NameId>(sn), bufNotIgnored, lenNotIgnored, owner);
                    }
                    // Accumulate this permutation's NormKey (Codec<NormKey> bytes)
                    // for the one batched insert after the loop.
                    {
                        const int32_t nkBlobLen = (lenNotIgnored + 2)
                            * static_cast<int32_t>(sizeof(NameId));
                        const ArenaOffset nkBlobOff = mnkGenArena.alloc(
                            nkBlobLen, static_cast<int32_t>(alignof(NameId)));
                        NameId* pnk = reinterpret_cast<NameId*>(
                            mnkGenArena.resolve(nkBlobOff));
                        pnk[0] = static_cast<NameId>(sn);
                        pnk[1] = lenNotIgnored;
                        std::memcpy(pnk + 2, bufNotIgnored,
                            static_cast<std::size_t>(lenNotIgnored) * sizeof(NameId));
                        remArgsBatch.push_back(
                            gl::RemArgsBatchBlob{ nkBlobOff, nkBlobLen });
                    }
                });
                RT_NOTE_ITERATIONS_HERE(orderingsVisited);   // sorted orderings installed
                // ONE RMW for the whole permutation batch into the invariant key.
                {
                RT_SCOPE_HERE("ADM_REMARGS_BATCH");
                gl::HashRuleIndex::stageRemainingArgsNormKeyBatch(
                    staging.remArgs,
                    staging.remArgsOwners,
                    intRemArgsSorted, remScratchN, remArgsBatch, owner, mnkGenArena);
                }
                mnkGenArena.popTo(remArgsBatchMark);
                // Request growth stops at the instance's maxKeyLength, so the
                // bound must cover the sn-premise whole keys staged above, as
                // addToHashMemory's does for an ordinary rule
                // (I-337).
                if (!op.isRemove()) {
                    intHashMemory.maxKeyLength =
                        std::max(static_cast<NameId>(sn), intHashMemory.maxKeyLength);
                }
            }
        }
    }

/// @brief Span-run overload of @ref makeNormalizedKeysForAdmission — the
///        span-native body; the `const std::vector<std::string>&` form
///        forwards.
///
/// @details
/// Equivalent to the vector form: both retain an anchor premise in every
/// derivative key while leaving the regular, ordis-only, and ordis2-demand
/// non-anchor qualification gates unchanged. Both use the same
/// normalization / owner-map / marker LMV machinery on the same bytes and
/// the same interner-touch sequence. This overload reads the key as a
/// caller-owned `StrSpan` run and takes value /
/// originalImpl / validity as spans, so a span-native caller
/// (`addToHashMemory`) reaches it 0% heap. The vector form builds a stack
/// `StrSpan` run and delegates here.
///
/// @param key                The key elements as a `StrSpan` run; cap
///                           `MAX_ADMISSION_KEY_ELEMENTS + 1`.
/// @param keyN               The element count.
/// @param intHashMemory      Target hash memory.
/// @param nameMap            `NameMap` for int16 encoding.
/// @param ruleInterner       Rule-registry value interner.
/// @param value              Head expression span.
/// @param minNumOperatorsKey Threshold for `implicationIsQualified`.
/// @param originalImpl       The admitting implication span (D-72 owner).
/// @param validityName       Owner scope span.
/// @param mbTrap             Reserved (debug instrumentation hook).
/// @param op                 Install (default) or the removal policy.
/// @param memo               The caller's per-copy premise encode memo
///                           bound to exactly @p key / @p keyN
///                           (`addToHashMemory` shares its copy's
///                           memo); `nullptr` = a standalone call that
///                           owns a local memo. Either way every
///                           derivative key reads its premises through
///                           the memo, never re-encoding a text.
/// @param staging            The rule-index staging serving
///                           @p intHashMemory on this slot (the
///                           install's); `nullptr` = a standalone call,
///                           which acquires it and flushes at exit.
/// @param integrationJustified Whether the installing rule carries the
///                           `"integration"` justification — selects the
///                           integration floor in the ordis2 demand
///                           pass's minimum-premise gate.
/// @return (void)
/// @see makeNormalizedKeysForAdmission(const std::vector<std::string>&, …)
///      — the twin oracle.
void AdmissionMarkerBuilder::makeNormalizedKeysForAdmission(
        const std::vector<std::string>& key,
        HashMemory& intHashMemory,
        NameMap& nameMap,
        ValueInterner& ruleInterner,
        const std::string& value,
        int minNumOperatorsKey,
        const std::string& originalImpl,
        const std::string& validityName,
        const Memory* mbTrap,
        bool integrationJustified) {
        // Vector forwarder onto the span-run body (the additive twin) — build a
        // stack StrSpan run over the caller-stable vector strings, byte-identical.
        StrSpan keyRun[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 1];
        const int32_t keyFwdN = static_cast<int32_t>(key.size());
        assert(keyFwdN <= ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 1
            && "makeNormalizedKeysForAdmission forwarder: key size exceeds cap");
        for (int32_t i = 0; i < keyFwdN; ++i) keyRun[i] = StrSpan(key[i]);
        makeNormalizedKeysForAdmission(keyRun, keyFwdN, intHashMemory, nameMap,
            ruleInterner, StrSpan(value), minNumOperatorsKey,
            StrSpan(originalImpl), StrSpan(validityName), mbTrap,
            RuleIndexOp{}, nullptr, nullptr, integrationJustified);
    }

/// @brief Span-run overload of @ref makeNormalizedKeysForAdmission — the
        ///        span-native body; the `const std::vector<std::string>&` form
        ///        forwards.
        ///
        /// @details
        /// Equivalent to the vector form: both retain an anchor premise in every
        /// derivative key while leaving the regular, ordis-only, and ordis2-demand
        /// non-anchor qualification gates unchanged. Both use the same
        /// normalization / owner-map / marker LMV machinery on the same bytes and
        /// the same interner-touch sequence. This overload reads the key as a
        /// caller-owned `StrSpan` run and takes value /
        /// originalImpl / validity as spans, so a span-native caller
        /// (`addToHashMemory`) reaches it 0% heap. The vector form builds a stack
        /// `StrSpan` run and delegates here.
        ///
        /// @param key                The key elements as a `StrSpan` run; cap
        ///                           `MAX_ADMISSION_KEY_ELEMENTS + 1`.
        /// @param keyN               The element count.
        /// @param intHashMemory      Target hash memory.
        /// @param nameMap            `NameMap` for int16 encoding.
        /// @param ruleInterner       Rule-registry value interner.
        /// @param value              Head expression span.
        /// @param minNumOperatorsKey Threshold for `implicationIsQualified`.
        /// @param originalImpl       The admitting implication span (D-72 owner).
        /// @param validityName       Owner scope span.
        /// @param mbTrap             Reserved (debug instrumentation hook).
        /// @param op                 Install (default) or the removal policy.
        /// @param memo               The caller's per-copy premise encode memo
        ///                           bound to exactly @p key / @p keyN
        ///                           (`addToHashMemory` shares its copy's
        ///                           memo); `nullptr` = a standalone call that
        ///                           owns a local memo. Either way every
        ///                           derivative key reads its premises through
        ///                           the memo, never re-encoding a text.
        /// @param staging            The rule-index staging serving
        ///                           @p intHashMemory on this slot (the
        ///                           install's); `nullptr` = a standalone call,
        ///                           which acquires it and flushes at exit.
        /// @param integrationJustified Whether the installing rule carries the
        ///                           `"integration"` justification — selects the
        ///                           integration floor in the ordis2 demand
        ///                           pass's minimum-premise gate.
        /// @param goalNearMiss       Whether the installing rule is a root
        ///                           goal's flat integration rule — runs the
        ///                           near-miss pass, which installs a
        ///                           `goalDemand` demand variant for each
        ///                           qualifying premise slot
        ///                           (`goalNearMissSlotQualifies`;
        ///                           D-574).
        ///                           A removal replays it with the same value.
        /// @return (void)
        /// @see makeNormalizedKeysForAdmission(const std::vector<std::string>&, …)
        ///      — the twin oracle.
void AdmissionMarkerBuilder::makeNormalizedKeysForAdmission(
        const StrSpan* key, int32_t keyN,
        HashMemory& intHashMemory,
        NameMap& nameMap,
        ValueInterner& ruleInterner,
        StrSpan value,
        int minNumOperatorsKey,
        StrSpan originalImpl,
        StrSpan validityName,
        const Memory* mbTrap,
        const RuleIndexOp& op,
        KeyEncodeMemo* memo,
        RuleIndexStaging* staging,
        bool integrationJustified,
        bool goalNearMiss) {

        static const bool traceEnabled = [] {
#ifdef _WIN32
            size_t required = 0;
            const errno_t result = getenv_s(&required, nullptr, 0, "GL_FTA_BLOWUP_TRACE");
            assert(result == 0);
            return required != 0;
#else
            return std::getenv("GL_FTA_BLOWUP_TRACE") != nullptr;
#endif
        }();
        const bool traceTarget = traceEnabled && mbTrap && hashburst_dump::isTargetLB(*mbTrap);
        const auto traceQualification = [&](const char* stage, int32_t index, int32_t validCount,
                bool classic = false, bool local = false, bool headOverlap = false) {
            if (!traceTarget) return;
            std::ostringstream line;
            line << "[FTA-DEMAND-RULE] " << stage << " scope=" << validityName.toStdString()
                 << " remove=" << op.isRemove() << " index=" << index << " valid=" << validCount
                 << " minimum=" << minNumOperatorsKey << " classic=" << classic << " local=" << local
                 << " headOverlap=" << headOverlap << " key={";
            for (int32_t i = 0; i < keyN; ++i) line << (i ? ";" : "") << key[i].toStdString();
            line << "} head=" << value.toStdString() << " carrier=" << originalImpl.toStdString() << "\n";
            // permanent instrument: GL_FTA_BLOWUP_TRACE (AGENTS.md Rule 36), off unless the variable is set
            std::ofstream trace(".debug/traps.log", std::ios::app);
            assert(trace.is_open());
            trace << line.str();
        };
        traceQualification("enter", -1, -1);

        // The premise encode memo: the install's per-copy memo when the
        // caller shares one (bound to this very key run — asserted), else a
        // local one for a standalone call. Both are the same contract; every
        // derivative key below reads its premises through it.
        const bool standalone = (memo == nullptr);
        KeyEncodeMemo localMemo;
        if (standalone) {
            localMemo.bind(key, keyN);
            memo = &localMemo;
        }
        assert(memo->key == key && memo->keyN == keyN
            && "makeNormalizedKeysForAdmission: the memo is bound to another key run");
        KeyEncodeMemo& encMemo = *memo;
        // A standalone call (the twin-oracle use: no shared memo) owns its
        // staging window — it acquires the slot's staging for the instance
        // and its staged index writes flush at exit on every return path; a
        // call sharing the install's memo is inside addToHashMemory, whose
        // exit (or the armed window's closer) flushes.
        const unsigned stagingSlot = (gl::WorkerContext::coreId() >= 0)
            ? static_cast<unsigned>(gl::WorkerContext::coreId())
            : genScratchArenas().slotCount() - 1;
        if (staging == nullptr && !op.isRemove()) {
            staging = &ruleStagings().acquire(stagingSlot, &intHashMemory);
        }
        assert((op.isRemove() || staging->boundTo == &intHashMemory)
            && "makeNormalizedKeysForAdmission: the staging serves another instance");
        RuleIndexStaging& stg = (staging != nullptr) ? *staging
                                                      : ruleStagings().at(stagingSlot, 0);   // never read on the remove path
        struct StandaloneFlush {
            HashMemory* hm;
            RuleIndexStaging* st;
            ~StandaloneFlush() {
                if (hm == nullptr) return;
                gl::HashRuleIndex::flushRuleIndexStaging(*hm, *st);
                st->boundTo = nullptr;   // the standalone call is its own window
            }
        } standaloneFlush{ (standalone && !op.isRemove()) ? &intHashMemory : nullptr, &stg };
        const NameId ownerVid = nameMap.encode(validityName);
        // The admitting rule's owner identity for every derivative index entry
        // (same pair as the head install's: the implication text's ruleInterner
        // id + the install scope id).
        const RuleOwner owner =
            packRuleOwner(ruleInterner.encode(originalImpl), ownerVid);
        const auto isAnchor = [](StrSpan elem) -> bool {
            return elem.len >= 7
                && equalSpans(StrSpan(elem.ptr, 7), StrSpan("(Anchor", 7));
        };

        // THIRD, independent qualification pass
        // (D-267) — runs REGARDLESS of the two
        // mutually exclusive routes below (it neither enables nor
        // suppresses them): each qualifying premise slot installs a
        // marker-style variant whose key is every other premise, including
        // the anchor context, and whose VALUE is the slot verbatim (no marker
        // token, polarity kept), flagged isMarker + ordis2Demand. Retaining
        // the anchor preserves the normalized equality classes that bind an
        // actual rule's anchor slots to its non-anchor premises. When the key
        // matches, the firing site's demand branch back-substitutes the match
        // into the value — the GROUND missing premise — and stages a demand
        // record for the post-fixpoint drain.
        {
            RT_SCOPE_HERE("ADM_PASS_ORDIS2");
            constexpr int32_t kO2KeyCap =
                ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 1;
            assert(keyN <= kO2KeyCap
                && "ordis2-demand pass: key size exceeds cap");
            for (int32_t index = 0; index < keyN; ++index) {
                if (!admissionPolicy.ordis2DemandSlotQualifies(key, keyN, index,
                        integrationJustified)) continue;
                int binary[kO2KeyCap];
                for (int32_t i = 0; i < keyN; ++i) binary[i] = 0;
                StrSpan subkey[kO2KeyCap];
                int32_t subCount = 0;
                for (int32_t i = 0; i < keyN; ++i) {
                    if (i == index) continue;
                    binary[i] = 1;
                    subkey[subCount++] = key[i];
                }
                // The subset condition binds >= 4 args through non-anchor
                // others, so a qualifying slot always has a subkey.
                assert(subCount >= 1
                    && "ordis2-demand pass: qualifying slot with empty subkey");
                installAdmissionMarkerVariants(key,
                    static_cast<std::size_t>(keyN),
                    static_cast<std::size_t>(index), StrSpan(),
                    binary, subkey, subCount, intHashMemory, nameMap,
                    ruleInterner, originalImpl, ownerVid, owner,
                    /*ordisOnly=*/false, /*ordis2Demand=*/true, op, encMemo, stg,
                    integrationJustified ? RuleJustification::integration
                                         : RuleJustification::none);
            }
        }

        // The goal near-miss pass (D-574):
        // on a root goal's flat integration rule only, each qualifying
        // premise slot installs a demand-kind variant shaped like the
        // ordis2 pass's — key every other premise, anchor included, value
        // the slot verbatim — stamped `goalDemand`. Its firing reports the
        // ground missing premise to the goal-driven rule search, never to the
        // OR-demand map. Independent of every other pass.
        if (goalNearMiss) {
            RT_SCOPE_HERE("ADM_PASS_GOAL_NEAR_MISS");
            constexpr int32_t kNMKeyCap =
                ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 1;
            assert(keyN <= kNMKeyCap
                && "goal near-miss pass: key size exceeds cap");
            for (int32_t index = 0; index < keyN; ++index) {
                if (!admissionPolicy.goalNearMissSlotQualifies(key, keyN, index)) continue;
                int binary[kNMKeyCap];
                StrSpan subkey[kNMKeyCap];
                int32_t subCount = 0;
                for (int32_t i = 0; i < keyN; ++i) {
                    binary[i] = (i == index) ? 0 : 1;
                    if (i != index) subkey[subCount++] = key[i];
                }
                assert(subCount >= 1
                    && "goal near-miss pass: qualifying slot with empty subkey");
                installAdmissionMarkerVariants(key,
                    static_cast<std::size_t>(keyN),
                    static_cast<std::size_t>(index), StrSpan(),
                    binary, subkey, subCount, intHashMemory, nameMap,
                    ruleInterner, originalImpl, ownerVid, owner,
                    /*ordisOnly=*/false, /*ordis2Demand=*/true, op, encMemo, stg,
                    RuleJustification::goalDemand);
            }
        }

        // FOURTH, independent qualification pass
        // (D-288) — like the ordis2 pass above,
        // it neither enables nor suppresses the two mutually exclusive
        // routes below: each qualifying (premise, input slot) pair installs
        // an UNTAGGED marker variant whose marked argument is the premise's
        // input-slot candidate and whose subkey is every other premise,
        // anchor included. When the subkey matches live facts, the marker
        // branch stages the INSTANTIATED key — the exact marker form under
        // which a witness-minting disintegration probes (and parks in
        // rejectedMap): the demanded witness registers instead of parking,
        // and revisitRejected2 wakes an already-parked product. Untagged,
        // so isAdmitted reads it (unlike the ordisOnly route's or-cohort
        // evidence); consumers probe through the ancestor-inclusive twin
        // because a fired key lands at deeperOf(subkey constituents) while
        // the disintegration may deposit in a descendant (ordis-branch)
        // scope.
        {
            RT_SCOPE_HERE("ADM_PASS_INPUT_SLOT");
            constexpr int32_t kISKeyCap =
                ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 1;
            assert(keyN <= kISKeyCap
                && "input-slot demand pass: key size exceeds cap");
            for (int32_t index = 0; index < keyN; ++index) {
                if (isAnchor(key[index])) continue;
                if (key[index].len > 0 && key[index].ptr[0] == '!') continue;
                const ce::CoreExpressionConfig* elemCfg =
                    definitions.coreConfig(extractExpressionSpan(key[index]));
                if (elemCfg == nullptr) continue;
                StrSpan elemArgs[ExecutionParameters::MAX_ARITY];
                const int32_t elemArgsN = getArgsSpans(key[index], elemArgs,
                    ExecutionParameters::MAX_ARITY);
                const int32_t inputN =
                    static_cast<int32_t>(elemCfg->inputIndices.size());
                for (int32_t ip = 0; ip < inputN; ++ip) {
                    if (!admissionPolicy.inputSlotDemandSlotQualifies(key, keyN, index, ip,
                            value)) {
                        continue;
                    }
                    const int idx = elemCfg->inputIndices[ip];
                    assert(idx >= 0 && idx < elemArgsN
                        && "input-slot demand pass: qualified slot out of range");
                    const StrSpan cand = elemArgs[idx];
                    int binary[kISKeyCap];
                    for (int32_t i = 0; i < keyN; ++i) binary[i] = 0;
                    StrSpan subkey[kISKeyCap];
                    int32_t subCount = 0;
                    for (int32_t i = 0; i < keyN; ++i) {
                        if (i == index) continue;
                        binary[i] = 1;
                        subkey[subCount++] = key[i];
                    }
                    assert(subCount >= 1
                        && "input-slot demand pass: qualifying slot with empty subkey");
                    installAdmissionMarkerVariants(key,
                        static_cast<std::size_t>(keyN),
                        static_cast<std::size_t>(index), cand,
                        binary, subkey, subCount, intHashMemory, nameMap,
                        ruleInterner, originalImpl, ownerVid, owner,
                        /*ordisOnly=*/false, /*ordis2Demand=*/false, op, encMemo, stg);
                }
            }
        }

        // FIFTH, independent qualification pass
        // (D-370) — the theorem-relation route:
        // a RECEIVED THEOREM whose other premises are all compound relations
        // installs an UNTAGGED marker variant for each qualifying input
        // slot of its one operator premise, exactly like the fourth pass
        // (marked argument = the input-slot candidate, subkey = every other
        // premise, anchor included; instantiated by the marker branch when
        // the subkey matches live facts, drained into the algebra
        // admissionMap, read through the ancestor-inclusive probe). The
        // predicate replaces the D-288 head-input condition and premise
        // floor with the structural description of the rule plus the
        // bound-argument floor; a pair the fourth pass already installed
        // installs the same variant again, which the index dedups.
        {
            RT_SCOPE_HERE("ADM_PASS_THEOREM_RELATION");
            constexpr int32_t kTRKeyCap =
                ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 1;
            assert(keyN <= kTRKeyCap
                && "theorem-relation demand pass: key size exceeds cap");
            for (int32_t index = 0; index < keyN; ++index) {
                if (isAnchor(key[index])) continue;
                if (key[index].len > 0 && key[index].ptr[0] == '!') continue;
                const ce::CoreExpressionConfig* elemCfg =
                    definitions.coreConfig(extractExpressionSpan(key[index]));
                if (elemCfg == nullptr) continue;
                StrSpan elemArgs[ExecutionParameters::MAX_ARITY];
                const int32_t elemArgsN = getArgsSpans(key[index], elemArgs,
                    ExecutionParameters::MAX_ARITY);
                const int32_t inputN =
                    static_cast<int32_t>(elemCfg->inputIndices.size());
                for (int32_t ip = 0; ip < inputN; ++ip) {
                    // A pair the fourth pass installed is not installed
                    // twice.
                    if (admissionPolicy.inputSlotDemandSlotQualifies(key, keyN, index, ip,
                            value)) {
                        continue;
                    }
                    if (!admissionPolicy.theoremRelationDemandSlotQualifies(key, keyN, index,
                            ip, integrationJustified)) {
                        continue;
                    }
                    const int idx = elemCfg->inputIndices[ip];
                    assert(idx >= 0 && idx < elemArgsN
                        && "theorem-relation demand pass: qualified slot out of range");
                    const StrSpan cand = elemArgs[idx];
                    int binary[kTRKeyCap];
                    for (int32_t i = 0; i < keyN; ++i) binary[i] = 0;
                    StrSpan subkey[kTRKeyCap];
                    int32_t subCount = 0;
                    for (int32_t i = 0; i < keyN; ++i) {
                        if (i == index) continue;
                        binary[i] = 1;
                        subkey[subCount++] = key[i];
                    }
                    assert(subCount >= 1
                        && "theorem-relation demand pass: qualifying slot with empty subkey");
                    installAdmissionMarkerVariants(key,
                        static_cast<std::size_t>(keyN),
                        static_cast<std::size_t>(index), cand,
                        binary, subkey, subCount, intHashMemory, nameMap,
                        ruleInterner, originalImpl, ownerVid, owner,
                        /*ordisOnly=*/false, /*ordis2Demand=*/false, op, encMemo, stg);
                }
            }
        }

        // Two mutually exclusive qualification routes: the regular
        // (A)/(B)/(C) gates retain their existing eligibility criteria; a
        // rule failing all three may still install ordisOnly-tagged marker
        // keys through the ordis candidate loop below — or-cohort opening
        // demand evidence, invisible to general Pass-B admission.
        const bool regularQualified =
            admissionPolicy.implicationIsQualified(key, keyN, value, minNumOperatorsKey);
        traceQualification(regularQualified ? "regular" : "not-regular", -1, -1);
        if (!regularQualified && !admissionPolicy.ordisRouteQualifies(key, keyN, value)) {
            return;
        }

        const std::size_t n = static_cast<std::size_t>(keyN);

        if (!regularQualified) {
            RT_SCOPE_HERE("ADM_PASS_ORDIS");
            // Ordis-only route: candidates are variables at config INPUT
            // slots of a premise, gated by (1) presence at a config input
            // slot of the HEAD (the maintainer-adopted extra filter), (2)
            // confinement — the candidate appears in no other key element
            // (the head is allowed), (3) concreteness — every other argument
            // of the marked premise appears in some subkey element, so a
            // fired key is fully instantiated. The subkey is every other
            // premise, including the anchor context (confinement already
            // guarantees none of them mentions the candidate). Deterministic
            // order: premise index ascending, config input-slot order.
            constexpr int32_t kKeyCap =
                ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 1;
            assert(n <= static_cast<std::size_t>(kKeyCap)
                && "makeNormalizedKeysForAdmission ordis: key size exceeds cap");

            const ce::CoreExpressionConfig* headCfg =
                definitions.coreConfig(extractExpressionSpan(value));
            assert(headCfg
                && "ordis route: head operator core must be compiled");
            StrSpan headArgs[ExecutionParameters::MAX_ARITY];
            const int32_t headArgsN = getArgsSpans(value, headArgs,
                                                   ExecutionParameters::MAX_ARITY);

            for (std::size_t index = 0; index < n; ++index) {
                const StrSpan elem = key[index];
                if (isAnchor(elem)) continue;
                const ce::CoreExpressionConfig* cfg =
                    definitions.coreConfig(extractExpressionSpan(elem));
                assert(cfg && "ordis route: premise operator core must be compiled");
                StrSpan args[ExecutionParameters::MAX_ARITY];
                const int32_t argsN = getArgsSpans(elem, args,
                                                   ExecutionParameters::MAX_ARITY);

                for (const int inIdx : cfg->inputIndices) {
                    if (inIdx < 0 || inIdx >= argsN) continue;
                    const StrSpan cand = args[inIdx];

                    // (1) Candidate at a config input slot of the head.
                    bool atHeadInput = false;
                    for (const int hIdx : headCfg->inputIndices) {
                        if (hIdx >= 0 && hIdx < headArgsN
                            && equalSpans(headArgs[hIdx], cand)) {
                            atHeadInput = true;
                            break;
                        }
                    }
                    if (!atHeadInput) continue;

                    // (2) Confinement: cand in no other key element.
                    bool confined = true;
                    for (std::size_t i = 0; i < n && confined; ++i) {
                        if (i == index) continue;
                        StrSpan iArgs[ExecutionParameters::MAX_ARITY];
                        const int32_t iArgsN = getArgsSpans(key[i], iArgs,
                            ExecutionParameters::MAX_ARITY);
                        for (int32_t a = 0; a < iArgsN; ++a) {
                            if (equalSpans(iArgs[a], cand)) {
                                confined = false;
                                break;
                            }
                        }
                    }
                    if (!confined) continue;

                    // Subkey = every other premise, including the anchor.
                    int binary[kKeyCap];
                    for (int32_t i = 0; i < static_cast<int32_t>(n); ++i)
                        binary[i] = 0;
                    StrSpan subkey[kKeyCap];
                    int32_t subCount = 0;
                    for (std::size_t i = 0; i < n; ++i) {
                        if (i == index) continue;
                        binary[i] = 1;
                        subkey[subCount++] = key[i];
                    }

                    // (3) Concreteness: every other argument of the marked
                    // premise appears among some subkey element's args (a
                    // repeated candidate arg fails — it cannot be bound).
                    bool concrete = true;
                    for (int32_t a = 0; a < argsN && concrete; ++a) {
                        if (a == inIdx) continue;
                        const StrSpan other = args[a];
                        if (equalSpans(other, cand)) {
                            concrete = false;
                            break;
                        }
                        bool bound = false;
                        for (int32_t s = 0; s < subCount && !bound; ++s) {
                            StrSpan sArgs[ExecutionParameters::MAX_ARITY];
                            const int32_t sArgsN = getArgsSpans(subkey[s],
                                sArgs, ExecutionParameters::MAX_ARITY);
                            for (int32_t sa = 0; sa < sArgsN; ++sa) {
                                if (equalSpans(sArgs[sa], other)) {
                                    bound = true;
                                    break;
                                }
                            }
                        }
                        if (!bound) concrete = false;
                    }
                    if (!concrete) continue;

                    installAdmissionMarkerVariants(key, n, index, cand,
                        binary, subkey, subCount, intHashMemory, nameMap,
                        ruleInterner, originalImpl, ownerVid, owner,
                        /*ordisOnly=*/true, /*ordis2Demand=*/false, op, encMemo, stg);
                }
            }
            return;
        }

        RT_SCOPE_HERE("ADM_PASS_REGULAR");
        for (std::size_t index = 0; index < n; ++index) {
            // The anchor is permanent context for every derivative rule; it
            // is never itself the missing premise.
            if (isAnchor(key[index])) continue;
            const ce::CoreExpressionConfig* cfg =
                definitions.coreConfig(extractExpressionSpan(key[index]));
            assert(cfg && "Core expression not found");
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t argsN = getArgsSpans(key[index], args,
                                               ExecutionParameters::MAX_ARITY);

            // Single output arg (0 or 1) — a StrSpan slice of key[index].
            constexpr int32_t kKeyCap =
                ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS + 1;
            assert(n <= static_cast<std::size_t>(kKeyCap)
                && "makeNormalizedKeysForAdmission: key size exceeds cap");
            StrSpan outputArg;
            bool haveOutput = false;
            for (int idx : cfg->outputIndices) {
                if (idx >= 0 && idx < argsN) {
                    assert(!haveOutput
                        && "Expression must have 0 or 1 output argument");
                    outputArg = args[idx];
                    haveOutput = true;
                }
            }
            if (!haveOutput) {
                traceQualification("no-output", static_cast<int32_t>(index), -1);
                continue;
            }

            // Binary mask + subkey (spans into key[], caller-stable).
            int binary[kKeyCap];
            for (int32_t i = 0; i < static_cast<int32_t>(n); ++i) binary[i] = 0;
            StrSpan subkey[kKeyCap];
            int32_t subCount = 0;
            int validCount = 0;
            for (std::size_t i = 0; i < n; ++i) {
                if (i == index) continue;
                StrSpan iArgs[ExecutionParameters::MAX_ARITY];
                const int32_t iArgsN = getArgsSpans(key[i], iArgs,
                                                    ExecutionParameters::MAX_ARITY);
                bool containsOutput = false;
                for (int32_t a = 0; a < iArgsN; ++a) {
                    if (equalSpans(iArgs[a], outputArg)) { containsOutput = true; break; }
                }
                if (isAnchor(key[i])) {
                    // The anchor always remains in the derivative key, while
                    // the existing non-output qualification count is unchanged.
                    binary[i] = 1;
                    subkey[subCount++] = key[i];
                    if (!containsOutput) validCount++;
                    continue;
                }
                if (!containsOutput) {
                    binary[i] = 1;
                    subkey[subCount++] = key[i];
                    validCount++;
                }
            }

            // Classic eligibility remains based on the existing (A)/(B)
            // qualification and non-output premise count. Anchor retention
            // changes only the installed derivative key.
            // (C) qualification (hasUPrefix + localUCriterion) admits keys
            // into this loop but does NOT enable the classic per-element
            // gates. Those keys must go through the local-u path below.
            // (D) qualification accepts exactly the slots that meet the
            // operator-relation gate: a missing in3 whose output alone is
            // unspecified and appears in the head. (E) accepts exactly the
            // missing compound operator of a four-operator rule
            // (D-569).
            const bool baselineQualified = admissionPolicy.baselineClassicQualifies(key, keyN, value, minNumOperatorsKey);
            const bool acceptedByClassic = baselineQualified && (
                (validCount == static_cast<int>(n) - 1) ||
                (validCount == parameters.minLenLongKey - 1));
            const bool acceptedByLocalU = gl::AdmissionPolicy::hasUPrefixInKey(key, keyN) &&
                admissionPolicy.elementMatchesLocalUCriterion(index, key, keyN, value);
            const bool acceptedByOperatorRelation = admissionPolicy.operatorRelationSlotQualifies(
                key, keyN, static_cast<int32_t>(index), value, minNumOperatorsKey);
            const bool acceptedByCompoundOperator = admissionPolicy.compoundOperatorSlotQualifies(
                key, keyN, static_cast<int32_t>(index), value);
            bool accept = acceptedByClassic || acceptedByLocalU || acceptedByOperatorRelation
                || acceptedByCompoundOperator;

            // Narrowing guard applies ONLY to the local u_-impl path.
            // Classic, operator-relation and compound-operator behavior is
            // not narrowed.
            if (accept && !acceptedByClassic && !acceptedByOperatorRelation
                && !acceptedByCompoundOperator &&
                admissionPolicy.outputMatchesHeadOutputSlot(outputArg, value)) {
                accept = false;
            }

            traceQualification(accept ? "install-marker" : "reject-marker", static_cast<int32_t>(index),
                validCount, acceptedByClassic, acceptedByLocalU, admissionPolicy.outputMatchesHeadOutputSlot(outputArg, value));
            if (!accept) continue;

            installAdmissionMarkerVariants(key, n, index, outputArg,
                binary, subkey, subCount, intHashMemory, nameMap,
                ruleInterner, originalImpl, ownerVid, owner,
                /*ordisOnly=*/false, /*ordis2Demand=*/false, op, encMemo, stg);
        }
    }
} // namespace gl
