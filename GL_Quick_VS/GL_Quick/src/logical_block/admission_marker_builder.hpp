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
#include "admission_policy.hpp"
namespace gl {
/// @brief Generate and install the eligible admission-marker variants of a rule.
/// @details Reads policy and definition interfaces, then delegates ordered index
/// writes to HashRuleIndex using the supplied memory and scratch lifetimes.
class AdmissionMarkerBuilder {
public:
/// @brief Bind the inputs required to generate admission-marker rules.
/// @details All references outlive this algorithm; construction installs no rules.
/// @param definitions Read-only operator and configuration definitions.
/// @param admissionPolicy Admission classification operations.
/// @param parameters Existing length thresholds.
/// @param allPermutationsAna Precomputed premise permutations, read without mutation.
/// @return Constructed marker generator.
AdmissionMarkerBuilder(const CompiledDefinitionRegistry& definitions, const AdmissionPolicy& admissionPolicy, const ProverParameters& parameters, const std::map<int, std::vector<std::vector<int>>>& allPermutationsAna);

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
void installAdmissionMarkerVariants(
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
            RuleJustification demandJustification = RuleJustification::none);

public:
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
void makeNormalizedKeysForAdmission(
            const std::vector<std::string>& key,
            HashMemory& intHashMemory,
            NameMap& nameMap,
            ValueInterner& ruleInterner,
            const std::string& value,
            int minNumOperatorsKey,
            // D-72: owner provenance for the marker LMVs in encodedMap. Caller passes
            // the implication being admitted and its scope; the radical wipe
            // at impl close uses these to filter dead-scope entries out.
            const std::string& originalImpl,
            const std::string& validityName,
            const Memory* mbTrap = nullptr,
            bool integrationJustified = false);

public:
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
void makeNormalizedKeysForAdmission(
            const StrSpan* key, int32_t keyN,
            HashMemory& intHashMemory,
            NameMap& nameMap,
            ValueInterner& ruleInterner,
            StrSpan value,
            int minNumOperatorsKey,
            StrSpan originalImpl,
            StrSpan validityName,
            const Memory* mbTrap = nullptr,
            const RuleIndexOp& op = RuleIndexOp{},
            KeyEncodeMemo* memo = nullptr,
            RuleIndexStaging* staging = nullptr,
            bool integrationJustified = false,
            bool goalNearMiss = false);

private:
const CompiledDefinitionRegistry& definitions;
const AdmissionPolicy& admissionPolicy;
const ProverParameters& parameters;
const std::map<int, std::vector<std::vector<int>>>& allPermutationsAna;
};
} // namespace gl
