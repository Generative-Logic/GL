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
#include "admission_policy.hpp"
namespace gl {
/// @brief Bind read-only definitions and the existing admission parameters.
/// @details References outlive this policy; construction performs no classification.
/// @param definitions Compiled definition and configuration readers.
/// @param parameters Existing prover policy parameters.
/// @return Constructed admission policy.
AdmissionPolicy::AdmissionPolicy(const CompiledDefinitionRegistry& definitions, const ProverParameters& parameters)
    : definitions(definitions), parameters(parameters) {}

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
bool AdmissionPolicy::ordisRouteQualifies(const StrSpan* key, int32_t keyN,
                                 StrSpan value) const {
        const auto isOperatorApplication = [this](StrSpan elem) -> bool {
            if (elem.len > 0 && elem.ptr[0] == '!') return false;
            const StrSpan core = extractExpressionSpan(elem);
            return this->definitions.operatorNames().find(std::string_view(core.ptr,
                static_cast<std::size_t>(core.len))) != this->definitions.operatorNames().end();
        };
        int32_t opCount = 0;
        for (int32_t i = 0; i < keyN; ++i) {
            const StrSpan elem = key[i];
            if (elem.len >= 7
                && equalSpans(StrSpan(elem.ptr, 7), StrSpan("(Anchor", 7))) {
                continue;   // anchors: excluded from requirement and count
            }
            if (!isOperatorApplication(elem)) return false;
            ++opCount;
        }
        if (!isOperatorApplication(value)) return false;
        ++opCount;
        return opCount >= ExecutionParameters::kOrdisMinOperatorExpressions;
    }

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
bool AdmissionPolicy::ordis2KeyEligible(StrSpan elem) const {
        // An anchor can never be selected as the missing compound premise.
        if (elem.len >= 7
            && equalSpans(StrSpan(elem.ptr, 7), StrSpan("(Anchor", 7))) {
            return false;
        }

        // COMPOUND: a compact with a compiled non-atomic entity;
        // polarity-transparent (a negated form qualifies on its core,
        // I-175 — the demand and the filing both carry the '!' verbatim,
        // only the eligibility test looks through it). The compiledEntity
        // fence alone IS the operator test: compiledExpressions holds only
        // compiled operator DEFINITIONS, so any hit is an operator by
        // construction — and crucially it admits PREDICATES (preorder /
        // strictOrder, output_args empty), which `operators` (the
        // input-AND-output core-config set) excludes. Gating on
        // `operators` here was the 3X install-route defect that silently
        // disqualified the whole order family: no demand ever installed,
        // no disjunct ever filed (root-caused 2026-08-10, trap4 run).
        const StrSpan bare = (elem.len > 0 && elem.ptr[0] == '!')
            ? StrSpan(elem.ptr + 1, elem.len - 1) : elem;
        if (bare.len < 1 || bare.ptr[0] != '(') return false;
        const StrSpan core = extractExpressionSpan(bare);
        const LogicalEntity* le = definitions.compiledEntity(core);
        if (le == nullptr || le->category == "atomic") return false;

        // ARITY.
        StrSpan args[ExecutionParameters::MAX_ARITY];
        const int32_t argsN = getArgsSpans(elem, args,
                                           ExecutionParameters::MAX_ARITY);
        return argsN >= ExecutionParameters::kOrdis2DemandMinArity;
    }

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
bool AdmissionPolicy::ordis2DemandSlotQualifies(const StrSpan* key,
            int32_t keyN, int32_t index,
            bool integrationJustified) const {
        const StrSpan elem = key[index];

        // Key-language filters (compound, arity, anchor refuse) — the
        // pair's ONE shared predicate, so a minted demand key is always a
        // text the park filing could have filed.
        if (!ordis2KeyEligible(elem)) return false;

        // SUBSET over the other NON-ANCHOR premises (the sharpened form,
        // maintainer-kept 2026-08-10): every slot argument must be bound by
        // a substantive premise. Anchor-only-bound short rules (the
        // A14 shape) mint no demand — the strict route through the Part-1
        // reduced or closes without them, and the parked totality cohorts
        // stay parked (D-267, filter decision).
        if (!slotArgumentsBoundByOthers(key, keyN, index)) return false;
        const auto isAnchor = [](StrSpan e) -> bool {
            return e.len >= 7
                && equalSpans(StrSpan(e.ptr, 7), StrSpan("(Anchor", 7));
        };

        // MINIMUM PREMISE COUNT, checked LAST: only premise-rich rules
        // mint demand — without this gate every short rule with a
        // qualifying compound slot installs demand-marker families across
        // all LBs (the post-eligibility-fix RT explosion). The floor is
        // justification-split: general rules need
        // kOrdis2DemandMinPremises NON-anchor premises, the qualifying
        // slot included (the B5 shape: slot + guard + two products);
        // integration-justified rules — a goal's flat integration-rule
        // variants and its per-level conjunct rules — need
        // kOrdis2DemandMinPremisesIntegration (the D1 closure rule: three
        // typings + two matched conjuncts + the one missing ground
        // compound premise).
        {
            int32_t nonAnchorN = 0;
            for (int32_t i = 0; i < keyN; ++i) {
                if (isAnchor(key[i])) continue;
                ++nonAnchorN;
            }
            const int32_t minPremises = integrationJustified
                ? ExecutionParameters::kOrdis2DemandMinPremisesIntegration
                : ExecutionParameters::kOrdis2DemandMinPremises;
            if (nonAnchorN < minPremises) {
                return false;
            }
        }
        return true;
    }

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
bool AdmissionPolicy::slotArgumentsBoundByOthers(const StrSpan* key, int32_t keyN,
            int32_t index) const {
        assert(index >= 0 && index < keyN
            && "slotArgumentsBoundByOthers: slot outside the premise run");
        const auto isAnchor = [](StrSpan e) -> bool {
            return e.len >= 7
                && equalSpans(StrSpan(e.ptr, 7), StrSpan("(Anchor", 7));
        };
        StrSpan args[ExecutionParameters::MAX_ARITY];
        const int32_t argsN = getArgsSpans(key[index], args,
                                           ExecutionParameters::MAX_ARITY);
        bool anySubstantive = false;
        for (int32_t a = 0; a < argsN; ++a) {
            bool bound = false;
            for (int32_t i = 0; i < keyN && !bound; ++i) {
                if (i == index) continue;
                if (isAnchor(key[i])) continue;
                StrSpan iArgs[ExecutionParameters::MAX_ARITY];
                const int32_t iN = getArgsSpans(key[i], iArgs,
                    ExecutionParameters::MAX_ARITY);
                for (int32_t x = 0; x < iN; ++x) {
                    if (equalSpans(iArgs[x], args[a])) {
                        bound = true;
                        break;
                    }
                }
            }
            if (bound) {
                anySubstantive = true;
                continue;
            }
            bool anchorSlot = false;
            for (int32_t i = 0; i < keyN && !anchorSlot; ++i) {
                if (!isAnchor(key[i])) continue;
                StrSpan iArgs[ExecutionParameters::MAX_ARITY];
                const int32_t iN = getArgsSpans(key[i], iArgs,
                    ExecutionParameters::MAX_ARITY);
                for (int32_t x = 0; x < iN; ++x) {
                    if (equalSpans(iArgs[x], args[a])) {
                        anchorSlot = true;
                        break;
                    }
                }
            }
            if (!anchorSlot) return false;
        }
        return anySubstantive;
    }

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
bool AdmissionPolicy::goalNearMissSlotQualifies(const StrSpan* key, int32_t keyN,
            int32_t index) const {
        assert(index >= 0 && index < keyN
            && "goalNearMissSlotQualifies: slot outside the premise run");
        const auto isAnchor = [](StrSpan e) -> bool {
            return e.len >= 7
                && equalSpans(StrSpan(e.ptr, 7), StrSpan("(Anchor", 7));
        };
        const StrSpan elem = key[index];
        if (isAnchor(elem)) return false;
        if (elem.len < 1 || elem.ptr[0] != '(') return false;
        const LogicalEntity* le = definitions.compiledEntity(extractExpressionSpan(elem));
        if (le == nullptr || le->category == "atomic") return false;
        if (!slotArgumentsBoundByOthers(key, keyN, index)) return false;
        int32_t others = 0;
        for (int32_t i = 0; i < keyN; ++i) {
            if (i == index || isAnchor(key[i])) continue;
            ++others;
        }
        return others >= ExecutionParameters::kGoalDemandMinPresentPremises;
    }

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
bool AdmissionPolicy::inputSlotDemandSlotQualifies(const StrSpan* key,
            int32_t keyN, int32_t index, int32_t inputPos,
            StrSpan value) const {
        const auto isAnchor = [](StrSpan e) -> bool {
            return e.len >= 7
                && equalSpans(StrSpan(e.ptr, 7), StrSpan("(Anchor", 7));
        };

        // The route's domain is positive operator-application premises: an
        // anchor, a negated form, and a core without a config (equality)
        // are defined misses, not errors — regular-qualified rules carry
        // such elements legitimately.
        const StrSpan elem = key[index];
        if (isAnchor(elem)) return false;
        if (elem.len > 0 && elem.ptr[0] == '!') return false;
        const ce::CoreExpressionConfig* cfg =
            definitions.coreConfig(extractExpressionSpan(elem));
        if (cfg == nullptr) return false;
        if (inputPos < 0
            || inputPos >= static_cast<int32_t>(cfg->inputIndices.size())) {
            return false;
        }
        StrSpan args[ExecutionParameters::MAX_ARITY];
        const int32_t argsN = getArgsSpans(elem, args,
                                           ExecutionParameters::MAX_ARITY);
        const int idx = cfg->inputIndices[inputPos];
        if (idx < 0 || idx >= argsN) return false;
        const StrSpan cand = args[idx];

        // (1) Candidate at a config input slot of the HEAD — the demanded
        // witness must feed the head, or releasing it buys nothing.
        if (value.len > 0 && value.ptr[0] == '!') return false;
        const ce::CoreExpressionConfig* headCfg =
            definitions.coreConfig(extractExpressionSpan(value));
        if (headCfg == nullptr) return false;
        StrSpan headArgs[ExecutionParameters::MAX_ARITY];
        const int32_t headArgsN = getArgsSpans(value, headArgs,
                                               ExecutionParameters::MAX_ARITY);
        bool atHeadInput = false;
        for (const int hIdx : headCfg->inputIndices) {
            if (hIdx >= 0 && hIdx < headArgsN
                && equalSpans(headArgs[hIdx], cand)) {
                atHeadInput = true;
                break;
            }
        }
        if (!atHeadInput) return false;

        // (2) Confinement: the candidate appears in no other key element
        // (the head is allowed). An anchor-slot candidate fails here by
        // construction — it appears in the anchor element.
        for (int32_t i = 0; i < keyN; ++i) {
            if (i == index) continue;
            StrSpan iArgs[ExecutionParameters::MAX_ARITY];
            const int32_t iN = getArgsSpans(key[i], iArgs,
                                            ExecutionParameters::MAX_ARITY);
            for (int32_t a = 0; a < iN; ++a) {
                if (equalSpans(iArgs[a], cand)) return false;
            }
        }

        // (3) Concreteness: every other argument of the marked premise
        // appears in some other key element (anchor included), so a fired
        // key is fully instantiated; a repeated candidate fails — it
        // cannot be bound.
        for (int32_t a = 0; a < argsN; ++a) {
            if (a == idx) continue;
            const StrSpan other = args[a];
            if (equalSpans(other, cand)) return false;
            bool bound = false;
            for (int32_t i = 0; i < keyN && !bound; ++i) {
                if (i == index) continue;
                StrSpan iArgs[ExecutionParameters::MAX_ARITY];
                const int32_t iN = getArgsSpans(key[i], iArgs,
                                                ExecutionParameters::MAX_ARITY);
                for (int32_t x = 0; x < iN; ++x) {
                    if (equalSpans(iArgs[x], other)) {
                        bound = true;
                        break;
                    }
                }
            }
            if (!bound) return false;
        }

        // MINIMUM PREMISE COUNT, checked LAST — the fan-out gate mirroring
        // kOrdis2DemandMinPremises (the D-267 RT lesson): only premise-rich
        // rules mint input-slot demand.
        {
            int32_t nonAnchorN = 0;
            for (int32_t i = 0; i < keyN; ++i) {
                if (isAnchor(key[i])) continue;
                ++nonAnchorN;
            }
            if (nonAnchorN < ExecutionParameters::kInputSlotDemandMinPremises) {
                return false;
            }
        }
        return true;
    }

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
bool AdmissionPolicy::theoremRelationDemandSlotQualifies(const StrSpan* key,
            int32_t keyN, int32_t index, int32_t inputPos,
            bool integrationJustified) const {
        const auto isAnchor = [](StrSpan e) -> bool {
            return e.len >= 7
                && equalSpans(StrSpan(e.ptr, 7), StrSpan("(Anchor", 7));
        };
        const auto isUPrefixed = [](StrSpan a) -> bool {
            return a.len >= 2 && a.ptr[0] == 'u' && a.ptr[1] == '_';
        };

        // (1) A received theorem: anchored, no `u_` premise argument (the
        // D-32 productOfDisintegration would be true otherwise), and not an
        // integration install.
        if (integrationJustified) return false;
        if (index < 0 || index >= keyN) return false;
        bool anchored = false;
        for (int32_t i = 0; i < keyN; ++i) {
            if (isAnchor(key[i])) { anchored = true; continue; }
            StrSpan iArgs[ExecutionParameters::MAX_ARITY];
            const int32_t iN = getArgsSpans(key[i], iArgs,
                                            ExecutionParameters::MAX_ARITY);
            for (int32_t a = 0; a < iN; ++a) {
                if (isUPrefixed(iArgs[a])) return false;
            }
        }
        if (!anchored) return false;

        // (2) The marked premise: a positive config operator, slot in range.
        const StrSpan elem = key[index];
        if (isAnchor(elem)) return false;
        if (elem.len > 0 && elem.ptr[0] == '!') return false;
        const ce::CoreExpressionConfig* cfg =
            definitions.coreConfig(extractExpressionSpan(elem));
        if (cfg == nullptr) return false;
        if (inputPos < 0
            || inputPos >= static_cast<int32_t>(cfg->inputIndices.size())) {
            return false;
        }
        StrSpan args[ExecutionParameters::MAX_ARITY];
        const int32_t argsN = getArgsSpans(elem, args,
                                           ExecutionParameters::MAX_ARITY);
        const int idx = cfg->inputIndices[inputPos];
        if (idx < 0 || idx >= argsN) return false;
        const StrSpan cand = args[idx];

        // (3) Every other non-anchor premise is a positive config-backed
        // compound relation (the `updateAdmissionMap` notion: a compiled
        // NON-atomic core that has a config — preorder / strictOrder /
        // remainder …). At least one such premise must exist.
        int32_t relationN = 0;
        for (int32_t i = 0; i < keyN; ++i) {
            if (i == index || isAnchor(key[i])) continue;
            const StrSpan other = key[i];
            if (other.len > 0 && other.ptr[0] == '!') return false;
            const StrSpan core = extractExpressionSpan(other);
            const LogicalEntity* ent =
                (core.len > 0) ? definitions.compiledEntity(core) : nullptr;
            if (ent == nullptr
                || equalSpans(StrSpan(ent->category), StrSpan("atomic", 6))) {
                return false;
            }
            if (definitions.coreConfig(core) == nullptr) return false;
            ++relationN;
        }
        if (relationN == 0) return false;

        // (4) Confinement: the candidate appears in no other key element.
        for (int32_t i = 0; i < keyN; ++i) {
            if (i == index) continue;
            StrSpan iArgs[ExecutionParameters::MAX_ARITY];
            const int32_t iN = getArgsSpans(key[i], iArgs,
                                            ExecutionParameters::MAX_ARITY);
            for (int32_t a = 0; a < iN; ++a) {
                if (equalSpans(iArgs[a], cand)) return false;
            }
        }

        // (5) Concreteness + (6) the bound-argument floor: every other
        // argument of the marked premise appears in some other key element
        // (anchor included); a repeated candidate fails. An argument that
        // is an anchor slot name (appears in an anchor element — the
        // operator slot `4`, the numerals) is bound but NOT counted; only
        // the arguments bound by a RELATION premise count against
        // kTheoremRelationDemandMinBoundArgs.
        int32_t boundByRelation = 0;
        for (int32_t a = 0; a < argsN; ++a) {
            if (a == idx) continue;
            const StrSpan other = args[a];
            if (equalSpans(other, cand)) return false;
            bool inAnchor = false;
            bool byRelation = false;
            for (int32_t i = 0; i < keyN; ++i) {
                if (i == index) continue;
                StrSpan iArgs[ExecutionParameters::MAX_ARITY];
                const int32_t iN = getArgsSpans(key[i], iArgs,
                                                ExecutionParameters::MAX_ARITY);
                bool here = false;
                for (int32_t x = 0; x < iN; ++x) {
                    if (equalSpans(iArgs[x], other)) { here = true; break; }
                }
                if (!here) continue;
                if (isAnchor(key[i])) inAnchor = true; else byRelation = true;
            }
            if (!inAnchor && !byRelation) return false;
            if (!inAnchor && byRelation) ++boundByRelation;
        }
        if (boundByRelation
            < ExecutionParameters::kTheoremRelationDemandMinBoundArgs) {
            return false;
        }
        return true;
    }

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
bool AdmissionPolicy::isRelationPremise(StrSpan elem) const {
        if (elem.len < 1 || elem.ptr[0] != '(') return false;
        if (elem.len >= 7
            && equalSpans(StrSpan(elem.ptr, 7), StrSpan("(Anchor", 7))) {
            return false;
        }
        const StrSpan core = extractExpressionSpan(elem);
        if (core.len == 0) return false;
        const LogicalEntity* entity = definitions.compiledEntity(core);
        if (entity == nullptr || entity->category == "atomic") return false;
        const ce::CoreExpressionConfig* cfg = definitions.coreConfig(core);
        return cfg != nullptr && cfg->outputIndices.empty();
    }

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
bool AdmissionPolicy::operatorRelationSlotQualifies(const StrSpan* key,
            int32_t keyN, int32_t index, StrSpan value,
            int minNumOperatorsKey) const {
        assert(index >= 0 && index < keyN
            && "operatorRelationSlotQualifies: slot outside the key");
        const auto isIn3 = [](StrSpan e) -> bool {
            return e.len >= 5 && e.ptr[0] == '(' && e.ptr[1] == 'i'
                && e.ptr[2] == 'n' && e.ptr[3] == '3' && e.ptr[4] == '[';
        };
        const auto occursIn = [](StrSpan expression, StrSpan arg) -> bool {
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t argsN = getArgsSpans(expression, args,
                                               ExecutionParameters::MAX_ARITY);
            for (int32_t a = 0; a < argsN; ++a) {
                if (equalSpans(args[a], arg)) return true;
            }
            return false;
        };

        // (1) The rule: more than minNumOperatorsKey in3 premises beside at
        // least one relation premise.
        int32_t in3N = 0;
        bool relation = false;
        for (int32_t i = 0; i < keyN; ++i) {
            if (isIn3(key[i])) ++in3N;
            else if (isRelationPremise(key[i])) relation = true;
        }
        if (in3N <= minNumOperatorsKey || !relation) return false;

        // (2) The missing premise: a positive in3 with its one output slot.
        const StrSpan elem = key[index];
        if (!isIn3(elem)) return false;
        const ce::CoreExpressionConfig* cfg =
            definitions.coreConfig(extractExpressionSpan(elem));
        assert(cfg != nullptr && cfg->outputIndices.size() == 1
            && "operatorRelationSlotQualifies: in3 must have one output slot");
        StrSpan args[ExecutionParameters::MAX_ARITY];
        const int32_t argsN = getArgsSpans(elem, args,
                                           ExecutionParameters::MAX_ARITY);
        const int outIdx = cfg->outputIndices[0];
        assert(outIdx >= 0 && outIdx < argsN
            && "operatorRelationSlotQualifies: in3 output slot out of range");
        const StrSpan output = args[outIdx];

        // (4) The one missing argument: the output is specified by no other
        // premise and appears in the head.
        for (int32_t i = 0; i < keyN; ++i) {
            if (i != index && occursIn(key[i], output)) return false;
        }
        if (!occursIn(value, output)) return false;

        // (3) Every other argument is specified by some other key element,
        // anchor included; a repeated output cannot be specified.
        for (int32_t a = 0; a < argsN; ++a) {
            if (a == outIdx) continue;
            if (equalSpans(args[a], output)) return false;
            bool specified = false;
            for (int32_t i = 0; i < keyN && !specified; ++i) {
                if (i != index && occursIn(key[i], args[a])) specified = true;
            }
            if (!specified) return false;
        }
        return true;
    }

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
bool AdmissionPolicy::compoundOperatorSlotQualifies(const StrSpan* key,
            int32_t keyN, int32_t index, StrSpan value) const {
        assert(index >= 0 && index < keyN
            && "compoundOperatorSlotQualifies: slot outside the key");
        const auto isOperatorPremise = [this](StrSpan e) -> bool {
            if (e.len < 1 || e.ptr[0] != '(') return false;
            if (e.len >= 7 && equalSpans(StrSpan(e.ptr, 7), StrSpan("(Anchor", 7))) return false;
            const StrSpan core = extractExpressionSpan(e);
            return core.len > 0 && this->definitions.operatorNames().find(std::string_view(core.ptr,
                static_cast<std::size_t>(core.len))) != this->definitions.operatorNames().end();
        };
        const auto isCompound = [this](StrSpan e) -> bool {
            const LogicalEntity* entity = this->definitions.compiledEntity(extractExpressionSpan(e));
            return entity != nullptr && entity->category != "atomic";
        };
        const auto occursIn = [](StrSpan expression, StrSpan arg) -> bool {
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t argsN = getArgsSpans(expression, args,
                                               ExecutionParameters::MAX_ARITY);
            for (int32_t a = 0; a < argsN; ++a) {
                if (equalSpans(args[a], arg)) return true;
            }
            return false;
        };

        // (1) The rule: exactly the set number of operator premises, exactly
        // the set number of them compound.
        int32_t operatorN = 0;
        int32_t compoundN = 0;
        for (int32_t i = 0; i < keyN; ++i) {
            if (!isOperatorPremise(key[i])) continue;
            ++operatorN;
            if (isCompound(key[i])) ++compoundN;
        }
        if (operatorN != ExecutionParameters::kCompoundOperatorRoutePremises
            || compoundN != ExecutionParameters::kCompoundOperatorRouteCompounds) return false;

        // (5) The head is a positive compound expression.
        if (value.len < 1 || value.ptr[0] != '(' || !isCompound(value)) return false;

        // (2) to (4): slot @p slot is a missing compound premise.
        const auto isMissingCompound = [&](int32_t slot) -> bool {
            // (2) A positive compound operator with its one output slot.
            const StrSpan elem = key[slot];
            if (!isOperatorPremise(elem) || !isCompound(elem)) return false;
            const ce::CoreExpressionConfig* cfg =
                definitions.coreConfig(extractExpressionSpan(elem));
            assert(cfg != nullptr && cfg->outputIndices.size() == 1
                && "compoundOperatorSlotQualifies: an operator has one output slot");
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t argsN = getArgsSpans(elem, args,
                                               ExecutionParameters::MAX_ARITY);
            const int outIdx = cfg->outputIndices[0];
            assert(outIdx >= 0 && outIdx < argsN
                && "compoundOperatorSlotQualifies: output slot out of range");
            const StrSpan output = args[outIdx];

            // (4) The one missing argument: the output is specified by no
            // other premise and appears in the head.
            for (int32_t i = 0; i < keyN; ++i) {
                if (i != slot && occursIn(key[i], output)) return false;
            }
            if (!occursIn(value, output)) return false;

            // (3) Every other argument is specified by some other key
            // element, anchor included; a repeated output cannot be specified.
            for (int32_t a = 0; a < argsN; ++a) {
                if (a == outIdx) continue;
                if (equalSpans(args[a], output)) return false;
                bool specified = false;
                for (int32_t i = 0; i < keyN && !specified; ++i) {
                    if (i != slot && occursIn(key[i], args[a])) specified = true;
                }
                if (!specified) return false;
            }
            return true;
        };
        if (!isMissingCompound(index)) return false;

        // (6) The marked premise is the rule's only missing compound premise.
        for (int32_t j = 0; j < keyN; ++j) {
            if (j != index && isMissingCompound(j)) return false;
        }
        return true;
    }
} // namespace gl
