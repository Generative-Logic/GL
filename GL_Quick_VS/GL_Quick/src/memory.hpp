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

#include "compiler.hpp"
#include "memory_infra/lb_memory.hpp"
#include "memory_infra/routing_cold_mail.hpp"
#include "memory_infra/lb_store.hpp"
#include "memory_infra/hash_memory.hpp"
#include "memory_infra/str_ops.hpp"
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <atomic>
#include <limits>
#include <cstdlib>
#include <new>

namespace gl {



    /// @brief One element of a request *Instruction*: a typed (category, signature, arity)
    /// description of a single logical entity plus its bound elements.
    ///
    /// @details
    /// Used inside `Instruction::data` (a sequence of `LogicalEntity` values) to describe
    /// a logical pattern the prover wants to match or instantiate. The fields are deliberately
    /// stringly typed because the same struct is keyed on `std::set` / `std::map` for
    /// deduplication and admission-map indexing.
    ///
    /// `category` is one of the high-level conceptual labels (e.g. "expression",
    /// "anchor", "operator"); `signature` is the concrete operator name; `elements`
    /// holds the bound argument names; `arity` is the declared arity; `definedSet`
    /// names the type set the entity belongs to (added to track the typing-set
    /// per the Pass-B / induction-typing fix scaffolding — see I-18).
    ///
    /// `operator<` orders by (category, signature, elements, arity, definedSet) so the
    /// struct works as a stable key in `std::set<LogicalEntity>` and inside compound
    /// containers like `std::map<Instruction, ...>`.
    ///
    /// @see [`Instruction`](#instruction) — owns a `std::vector<LogicalEntity> data`.
    /// @see [`HashMemory::admissionMapIntegration`](#hashmemory) — keyed on `Instruction`,
    ///      so transitively keyed on this type.
    struct LogicalEntity {
        // Member Variables
        std::string category = "Uncategorized";
        std::vector<std::string> elements = {};
        std::string signature = "";
        int arity = 0;
        std::string definedSet = ""; // <--- NEW FIELD

        // 1. Default Constructor
        LogicalEntity() = default;

        // 2. Parameterized Constructor
        LogicalEntity(std::string cat, std::vector<std::string> elems, std::string sig, int ar, std::string ds = "")
            : category(std::move(cat)),
            elements(std::move(elems)),
            signature(std::move(sig)),
            arity(ar),
            definedSet(std::move(ds)) {
        }

        // 3. Helper to add items easily
        void addElement(const std::string& element) {
            elements.push_back(element);
        }

        // 4. Operator < for std::set usage
        bool operator<(const LogicalEntity& rhs) const {
            if (category != rhs.category) return category < rhs.category;
            if (signature != rhs.signature) return signature < rhs.signature;
            if (elements != rhs.elements) return elements < rhs.elements;
            if (arity != rhs.arity) return arity < rhs.arity;
            return definedSet < rhs.definedSet; // Include new field
        }

        bool operator==(const LogicalEntity& rhs) const {
            return category == rhs.category &&
                signature == rhs.signature &&
                elements == rhs.elements &&
                arity == rhs.arity &&
                definedSet == rhs.definedSet; // Include new field
        }
    };

    /// @brief The compiled-definition map: `LogicalEntity` keyed by core
    ///        expression name, with a transparent comparator.
    ///
    /// @details
    /// This is the storage type of `ExpressionAnalyzer::compiledExpressions`
    /// — the compiled-definition layer, an OPEN heap violation still to
    /// statify (NOT a sanctioned heap island, NOT a user-approved
    /// carve-out), tracked by `I-137`. The comparator is
    /// `std::less<>` (transparent) rather than the default
    /// `std::less<std::string>` so that a heterogeneous `find(std::string_view)`
    /// probe compiles without materializing a `std::string` key — the
    /// zero-allocation read the `ExpressionAnalyzer::compiledEntity` accessor
    /// relies on.
    ///
    /// **Byte-order contract.** `std::less<>` compares two stored
    /// `std::string` keys via `std::string::operator<` — byte-lexicographic,
    /// the SAME strict weak order the default `std::less<std::string>`
    /// produced. The tree shape, insertion results, and every whole-map
    /// iteration order are therefore byte-identical to the pre-switch map. A
    /// heterogeneous probe compares a stored `std::string` against a
    /// `std::string_view` via `std::string_view::operator<` — also
    /// byte-lexicographic — so a span probe returns exactly the entry a
    /// materialized-`std::string` probe returned. No mint, interner, or
    /// serialization touches this map (config-load populated, never
    /// deloaded), so the switch has zero observable surface beyond
    /// compile-time type identity.
    ///
    /// @invariant `I-137` — the zero-allocation read fence over the
    ///            still-heap compiled-definition layer (an OPEN violation
    ///            to statify, not a sanctioned carve-out).
    /// @see `ExpressionAnalyzer::compiledEntity` — the reader accessor.
    /// @see `ce::CoreExpressionMap` — the sibling transparent map.
    using CompiledExpressionMap = std::map<std::string, LogicalEntity, std::less<>>;

    /// @brief The closed justification vocabulary of rule-registry entries
    /// (D-133).
    ///
    /// @details
    /// Every installed `LocalMemoryValue` carries exactly one justification.
    /// The vocabulary is closed: `ruleJustificationFromString` asserts on a
    /// string outside it — a new install site must extend this enum AND the
    /// name table, never pass through silently (Rule 19). The underlying
    /// value is storage only; the only semantic comparison is equality
    /// (the `integration` gate in `checkLocalEncodedMemoryStatic`).
    enum class RuleJustification : uint8_t {
        none,           // ""  (marker LMVs — no head, nothing fires)
        implication,    // "implication"
        integration,    // "integration"
        COUNT           // sentinel — table size only
    };

    /// @brief Recover the exact historical justification string of a
    ///        `RuleJustification`.
    ///
    /// @param j A real justification (not the `COUNT` sentinel).
    /// @return The justification string literal.
    inline const char* ruleJustificationName(RuleJustification j) {
        static const char* const NAMES[] = {
            "",
            "implication",
            "integration",
        };
        static_assert(sizeof(NAMES) / sizeof(NAMES[0])
            == static_cast<std::size_t>(RuleJustification::COUNT),
            "ruleJustificationName table must stay index-aligned with RuleJustification");
        const std::size_t i = static_cast<std::size_t>(j);
        assert(i < static_cast<std::size_t>(RuleJustification::COUNT));
        return NAMES[i];
    }

    /// @brief Encode a justification string (closed-vocabulary contract).
    ///
    /// @details Asserts on a string outside the vocabulary — a firing
    /// assert is a report, never a fallback (Rule 19).
    ///
    /// @param s The justification string.
    /// @return The matching enum value.
    inline RuleJustification ruleJustificationFromString(const std::string& s) {
        for (std::size_t i = 0;
             i < static_cast<std::size_t>(RuleJustification::COUNT); ++i) {
            if (s == ruleJustificationName(static_cast<RuleJustification>(i))) {
                return static_cast<RuleJustification>(i);
            }
        }
        assert(false && "ruleJustificationFromString: unknown justification");
        return RuleJustification::implication;
    }

    /// @brief `StrSpan` overload of @ref ruleJustificationFromString — the
    ///        span-native twin; the `const std::string&` form is unchanged.
    ///
    /// @details
    /// Byte-identical verdict to the owning form: matches @p s against the
    /// closed justification vocabulary via `equalSpans` (== `std::string`
    /// equality), so a caller already holding a span (the statified
    /// `addToHashMemory` head install) reaches the enum without materializing a
    /// `std::string`. Asserts on a string outside the vocabulary — a firing
    /// assert is a report, never a fallback (Rule 19).
    ///
    /// @param s The justification span.
    /// @return The matching enum value.
    /// @see ruleJustificationFromString(const std::string&) — the twin oracle.
    inline RuleJustification ruleJustificationFromString(StrSpan s) {
        for (std::size_t i = 0;
             i < static_cast<std::size_t>(RuleJustification::COUNT); ++i) {
            const char* name =
                ruleJustificationName(static_cast<RuleJustification>(i));
            if (equalSpans(s,
                    StrSpan(name, static_cast<int32_t>(std::strlen(name))))) {
                return static_cast<RuleJustification>(i);
            }
        }
        assert(false && "ruleJustificationFromString: unknown justification");
        return RuleJustification::implication;
    }

    /// @brief Value record stored in `HashMemory::encodedMap` — the cached effect of
    /// firing a single hash-engine rule.
    ///
    /// @details
    /// One `LocalMemoryValue` (LMV) describes a single (premise-key → head) entry of
    /// the hash engine. Multiple LMVs may share a key when several distinct
    /// implications collapse onto the same normalized premise pattern (e.g. mirror
    /// statements after `addEquality` symmetrization). The fields:
    ///
    /// Id form per `D-133`: every string field is an
    /// id in the owning LB's `Memory::ruleInterner` (int32), minted at the
    /// single-threaded install sites; the parallel hashburst only decodes
    /// (array-index const refs, I-83-safe).
    ///
    /// - `valueId`          — the head expression template (what the rule produces).
    /// - `levels`           — the set of LB levels at which the rule is admissible.
    /// - `originalImplicationId` — the producer rule's full implication text, kept
    ///   for provenance and origin chaining.
    /// - `justification`    — the closed `RuleJustification` vocabulary
    ///   (`none` on marker LMVs, `implication` on installed rules).
    /// - `keyIds`           — the marker sub-key chain, positional. Empty on
    ///   head-implication LMVs (only marker LMVs carry it). The int16_t premise
    ///   mirror lives in `IntNormalizedKey` and is what the unordered_map
    ///   actually keys on.
    /// - `remainingArgIds`  — argument names not consumed by the key match,
    ///   decoded-lex sorted (set-iteration encode order).
    /// - `validityId`       — the NameMap id of the scope at which this rule
    ///   fires, minted at install (the same encode that feeds the owner
    ///   record), so the firing path's scope-comparability check reads it
    ///   without a per-firing lookup.
    /// - `isMarker`         — install-time classification: the head template
    ///   contains `"marker"`. Replaces the per-firing substring scan.
    /// - `productOfDisintegration` — D-32 marker. True when at least one premise of
    ///   the producing implication carries a `u_`-prefixed arg (i.e. the implication
    ///   itself is a product of an earlier disintegration step). Set in
    ///   `addToHashMemory`; consumed by `checkLocalEncodedMemoryStatic` to gate
    ///   OR-disintegration on the head-firing path.
    ///
    /// @see [`HashMemory`](#hashmemory) — owns the `encodedMap` keyed on
    ///      `IntNormalizedKey`, valued as `std::vector<LocalMemoryValue>`.
    /// @see `prover.hpp::addToHashMemory` — installs LMVs.
    /// @see `prover.hpp::checkLocalEncodedMemoryStatic` — reads LMVs on hash hit.
    /// @see [D-32 in 40_decisions.md](../../docs/agentic_swdd/40_decisions.md#d-32) for the
    ///      `productOfDisintegration` rationale.
    struct LocalMemoryValue {
        int32_t valueId;
        std::set<int> levels;
        int32_t originalImplicationId;
        RuleJustification justification;
        std::vector<int32_t> keyIds;
        std::vector<int32_t> remainingArgIds;
        // The NameMap scope id, minted at install. 1 == NameMap::MAIN_ID
        // ("main" pre-registers as id 1 — NameMap is declared later in
        // this header, hence the literal).
        int16_t validityId;
        // Install-time classification: head template contains "marker".
        bool isMarker;
        // D-32: marks LMVs whose implication is itself a "product of
        // disintegration" — at least one premise has an arg starting
        // with "u_". Set at install time in addToHashMemory; consumed by
        // checkLocalEncodedMemoryStatic to gate OR-disintegration on
        // the head-firing path. Default false (admission/marker LMVs
        // and non-disintegration-product implications).
        bool productOfDisintegration;

        LocalMemoryValue()
            : valueId(0),
            levels(),
            originalImplicationId(0),
            justification(RuleJustification::none),
            keyIds(),
            remainingArgIds(),
            validityId(1),
            isMarker(false),
            productOfDisintegration(false) {
        }
    };



    /// @brief One entry in the *admission map* — pre-computed parameters that bound how
    /// far a hash-engine rule may be applied at a given expression / validity pair.
    ///
    /// @details
    /// The admission map (`HashMemory::admissionMap`, keyed on `ExpressionWithValidity`
    /// → `std::set<AdmissionMapValue>`) caches the work of `updateAdmissionMap3` so
    /// repeated firings of the same rule under the same scope do not have to
    /// re-derive the depth budget. Each `AdmissionMapValue` carries:
    ///
    /// - `key` and `remainingArgs` — the same shapes used inside `LocalMemoryValue`,
    ///   so the admission record can be matched against the rule's own key during
    ///   `checkLocalEncodedMemoryStatic`.
    /// - `standardMaxAdmissionDepth` — the maximum recursive depth at which the rule
    ///   is allowed to fire from this anchor point. Caps blow-up.
    /// - `standardMaxSecondaryNumber` — the maximum secondary-iteration counter the
    ///   rule is allowed to consume.
    /// - `flag` — admission-side discriminator; `true` for marker LMVs (the entries
    ///   that exist only to gate other rules), `false` for ordinary head LMVs.
    ///
    /// The struct is value-equal under lexicographic ordering of all fields, which
    /// lets it live in the `std::set<AdmissionMapValue>` slot of `admissionMap`.
    ///
    /// @see [`HashMemory::admissionMap`](#hashmemory) — owns these values.
    /// @see `prover.hpp::updateAdmissionMap3` — populates new entries.
    /// @see [I-22](../../docs/agentic_swdd/30_invariants.md#i-22) — `rejectedMapIntegration`
    ///      revival does NOT clean the admission-map entry; the two side-tables
    ///      are intentionally not coupled.
    struct AdmissionMapValue {
        // Id-form fields (D-1322): every string lives
        // in Memory::valueInterner. `remainingArgs` is stored decoded-lex
        // sorted (the former std::set<std::string> iteration order), so a
        // plain id-vector compare via valueIdVectorLess reproduces the
        // historical set comparison and print order exactly.
        std::vector<int32_t> key;
        std::vector<int32_t> remainingArgs;
        int standardMaxAdmissionDepth;
        int standardMaxSecondaryNumber;
        bool flag;

        // Default constructor
        AdmissionMapValue()
            : key(),
            remainingArgs(),
            standardMaxAdmissionDepth(0),
            standardMaxSecondaryNumber(0),
            flag(false) {
        }

        // Full constructor (id-form inputs)
        AdmissionMapValue(std::vector<int32_t> key_,
            std::vector<int32_t> remainingArgs_,
            int standardMaxAdmissionDepth_,
            int standardMaxSecondaryNumber_,
            bool flag_)
            : key(std::move(key_)),
            remainingArgs(std::move(remainingArgs_)),
            standardMaxAdmissionDepth(standardMaxAdmissionDepth_),
            standardMaxSecondaryNumber(standardMaxSecondaryNumber_),
            flag(flag_) {
        }
    };

    /// @brief The STRING-form twin of [`AdmissionMapValue`](#admissionmapvalue),
    /// used only on the parallel staging path.
    ///
    /// @details The phase-2 burst executors build these inside
    /// `FiringRecord` / `AdmissionKeyAlgebraRecord` — they must not mint
    /// into any interner ([I-83](../../docs/agentic_swdd/30_invariants.md#i-83)/[I-68](../../docs/agentic_swdd/30_invariants.md#i-68)),
    /// so the strings travel verbatim and the single-threaded
    /// `drainAdmissionKeysAlgebra` converts via `stagedToArenaBlob` (the
    /// retained `stagedToIdValue` is its unit-test oracle). No ordering
    /// operator on the struct itself: the firing-record comparator reads the
    /// sealed `key` / `remainingArgsSorted` vectors by content as part of
    /// its strict total order (I-77), never through a member operator.
    struct StagedAdmissionValue {
        // Sealed spans into the producing task's page set
        // (D-164): the replaced key elements in
        // rule order, and the remaining args sorted-unique lex (the former
        // std::set<std::string> flattened; same order, same dedup). The
        // spines ride the page set too, so the whole staged value is POD.
        SealedSpan<SealedString> key;
        SealedSpan<SealedString> remainingArgsSorted;
        int standardMaxAdmissionDepth;
        int standardMaxSecondaryNumber;
        bool flag;

        StagedAdmissionValue()
            : key(),
            remainingArgsSorted(),
            standardMaxAdmissionDepth(0),
            standardMaxSecondaryNumber(0),
            flag(false) {
        }
    };




    /// @brief Legacy string-form of a hash-engine normalized key.
    ///
    /// @details
    /// Predecessor of `IntNormalizedKey`. Each `data` entry is an alternating
    /// expression-name / variable-name string sequence; `numberExpressions` is the
    /// count of expression segments. Read by older paths that have not yet been
    /// migrated to the int16-only static path; new code keys hash memory on
    /// `IntNormalizedKey` instead. The string-form key is kept primarily for
    /// debug printing and for cross-checks during the static-mirror build-up
    /// in `addToHashMemory`.
    ///
    /// `operator<` and `operator==` give lexicographic ordering over
    /// `(numberExpressions, data)` so the struct can be used as a `std::map` key
    /// or a `std::set` element.
    ///
    /// @see [`IntNormalizedKey`](#intnormalizedkey) — int16_t replacement on the
    ///      hot path.
    struct NormalizedKey {
        // Fields (camelCase)
        int numberExpressions;
        std::vector<std::string> data;

        // Constructors
        NormalizedKey()
            : numberExpressions(0), data() {
        }

        NormalizedKey(int numberExpressions_, const std::vector<std::string>& data_)
            : numberExpressions(numberExpressions_), data(data_) {
        }

        // Ordering so it can be used in std::set<> and as a std::map<> key.
        // Lexicographic over (numberExpressions, data).
        bool operator<(const NormalizedKey& rhs) const {
            if (numberExpressions != rhs.numberExpressions) {
                return numberExpressions < rhs.numberExpressions;
            }
            return data < rhs.data; // vector<string> has lexicographic operator<
        }

        // Optional equality (useful if you compare directly)
        bool operator==(const NormalizedKey& rhs) const {
            return numberExpressions == rhs.numberExpressions && data == rhs.data;
        }
    };


    /// @brief One rejection record for the disintegration-side admission map.
    ///
    /// @details
    /// Lives in `HashMemory::rejectedMap`, keyed on the rejected
    /// `ExpressionWithValidity`. Used to remember disintegration-rejected products
    /// across iterations so the same rejection isn't re-emitted on every fold of
    /// the same producer rule. The fields:
    ///
    /// - `renamedExpression` — the expression after the disintegration-rename pass
    ///   (variables substituted to fresh names so identity-collision is
    ///   distinguishable across LBs).
    /// - `expression`        — the original (pre-rename) compact-compound expression
    ///   text (the compound that was being disintegrated when this child was
    ///   rejected), retained for origin chaining, human-readable diagnostics, and
    ///   the `intStatementLevelsMap` levels lookup at revival time.
    /// - `iteration`         — the prover iteration at which the rejection was
    ///   recorded. Used to expire stale rejections when iteration advances.
    /// - `concreteConstituent` — the rejected child element with the `u_` prefix
    ///   stripped (matches the integration-side `RejectedMapIntegrationValue`
    ///   field). This is the post-disintegration form that revival mails to
    ///   `sameIterationInternalMail` as the primary constituent.
    /// - `siblings`            — the OTHER body elements of the parent compound
    ///   (same disintegration cohort, in concrete u_-stripped form). Carried so
    ///   the `sameIterationInternalMail` revival path can re-emit the full body together.
    /// - `levels`              — the deductive-depth level set inherited from the
    ///   parent compound. Snapshot at rejection time; used as the mail-deposit
    ///   levels at revival.
    ///
    /// `operator<` orders lexicographically over all fields so the struct fits
    /// directly into `std::set<RejectedMapValue>`.
    ///
    /// @see [`HashMemory::rejectedMap`](#hashmemory).
    /// @see [`RejectedMapIntegrationValue`](#rejectedmapintegrationvalue) — the
    ///      integration-side counterpart with no iteration semantics.
    struct RejectedMapValue {
        // Id-form fields (D-1322): the string fields
        // live in Memory::valueInterner; siblings keep their positional
        // order. Ordering lives in DecodedRejectedValueLess.
        int32_t renamedExpression;
        int32_t expression;
        int iteration;
        int32_t concreteConstituent;
        std::vector<int32_t> siblings;
        std::set<int> levels;

        // Constructors
        RejectedMapValue()
            : renamedExpression(0),
            expression(0),
            iteration(0),
            concreteConstituent(0),
            siblings(),
            levels() {
        }

        RejectedMapValue(int32_t renamedExpression_,
            int32_t expression_,
            int iteration_,
            int32_t concreteConstituent_,
            std::vector<int32_t> siblings_,
            const std::set<int>& levels_)
            : renamedExpression(renamedExpression_),
            expression(expression_),
            iteration(iteration_),
            concreteConstituent(concreteConstituent_),
            siblings(std::move(siblings_)),
            levels(levels_) {
        }

        /// @brief Construct from an ASCENDING-UNIQUE level RUN — the run twin
        ///        of the `std::set<int>` constructor.
        ///
        /// @details
        /// The `levels` MEMBER deliberately stays `std::set<int>`: it is a
        /// STORAGE field (`Codec<RejectedMapValue>` serializes it into the
        /// rejected-map blob run, and the blob bytes must not move). Only the
        /// construction boundary changes — an ascending-unique run inserted
        /// with end-hints produces the identical set content, hence identical
        /// codec bytes, while the caller drops its transient `std::set<int>`.
        ///
        /// @pre `[levels_, levels_ + levelCount_)` is sorted ascending with no
        ///      duplicates (shape asserted; the order obligation is the
        ///      caller's source contract — a sorted-unique store run).
        /// @param renamedExpression_   See the class field docs.
        /// @param expression_          See the class field docs.
        /// @param iteration_           See the class field docs.
        /// @param concreteConstituent_ See the class field docs.
        /// @param siblings_            See the class field docs (moved in).
        /// @param levels_    Pointer to `levelCount_` ascending-unique levels;
        ///                   may be null only when `levelCount_` is 0.
        /// @param levelCount_ The level count (`>= 0`).
        /// @invariant [I-136](../../docs/agentic_swdd/30_invariants.md#i-136)
        ///            — the run is caller-owned; the set member is the
        ///            storage boundary.
        /// @see RejectedMapValue(int32_t, int32_t, int, int32_t,
        ///      std::vector<int32_t>, const std::set<int>&) — the byte-oracle
        ///      constructor; `Codec<RejectedMapValue>`.
        RejectedMapValue(int32_t renamedExpression_, int32_t expression_,
            int iteration_, int32_t concreteConstituent_,
            std::vector<int32_t> siblings_,
            const int* levels_, int32_t levelCount_)
            : renamedExpression(renamedExpression_),
            expression(expression_),
            iteration(iteration_),
            concreteConstituent(concreteConstituent_),
            siblings(std::move(siblings_)),
            levels() {
            assert(levelCount_ >= 0 && (levelCount_ == 0 || levels_ != nullptr));
            for (int32_t i = 0; i < levelCount_; ++i)
                levels.insert(levels.end(), levels_[i]);
        }
    };

    /// @brief One rejection record for the integration-side admission map.
    ///
    /// @details
    /// Integration counterpart to `RejectedMapValue`. Stored in
    /// `HashMemory::rejectedMapIntegration`, keyed on a non-`(in[...])` constituent's
    /// marker form (the `int_` variable at the `marker` position). Integration has
    /// no iteration semantics (the `int_` mint pulls only `level + startInt` per
    /// the integration-side counter), so this struct has no iteration field.
    ///
    /// - `concreteConstituent` — the body element with the `u_` prefix stripped and
    ///   its `int_` marker intact; this is what the marker keys on.
    /// - `siblings`            — all OTHER body elements of the compound, including
    ///   the typing `(in[...])` element, in concrete form. Carried so the
    ///   `sameIterationInternalMail` revival path can re-emit the full body together.
    /// - `compoundExpression`  — the original compound, kept for origin chaining.
    ///
    /// @see [`HashMemory::rejectedMapIntegration`](#hashmemory).
    /// @see [I-22](../../docs/agentic_swdd/30_invariants.md#i-22) — revival does NOT clean the
    ///      admission-map entry; revival is additive.
    /// @see [I-30](../../docs/agentic_swdd/30_invariants.md#i-30) —
    ///      `applyEquivalenceClassToRejectedMapIntegration` is additive (existing
    ///      entries are never erased).
    struct RejectedMapIntegrationValue {
        // Id-form fields (D-1322): strings live in
        // Memory::valueInterner; siblings keep positional order. Ordering
        // lives in DecodedRejectedIntegrationValueLess.
        int32_t concreteConstituent;            // body element, u_ stripped, int_ intact
        std::vector<int32_t> siblings;          // other body elements (incl. in[]), concrete
        int32_t compoundExpression;             // original compound, for origin chaining

        RejectedMapIntegrationValue()
            : concreteConstituent(0), siblings(), compoundExpression(0) {}

        RejectedMapIntegrationValue(int32_t concreteConstituent_,
            std::vector<int32_t> siblings_,
            int32_t compoundExpression_)
            : concreteConstituent(concreteConstituent_),
              siblings(std::move(siblings_)),
              compoundExpression(compoundExpression_) {}
    };


    /// @brief Owner-set value for the `normalizedEncoded*` fast-rejection
    /// maps — each implication-with-scope that birthed a key, as one packed
    /// composite id per owner.
    ///
    /// @details `partitionIds` IS the owner record
    /// (`D-133` unification): one composite `int32`
    /// per owner, `makePartitionId(NameMap::encode(expandedOriginal),
    /// scopeVid)` — both halves NameMap-minted at install. It serves every
    /// former `owners`-map role at once:
    /// - **D-72 ownership** — a key is erased once the set empties; the
    ///   radical subtree wipe erases an owner by decoding the LOW half
    ///   (its scope validity id) for the closed-scope prefix test.
    /// - **D-105 comparability prune** — `ownerSetHasComparable` reads the
    ///   low half of each id; pure `int16_t` comparisons in the hot loop.
    /// - **D-119 LB-split partition cover** — a split executor `n` of `N`
    ///   accepts this (sub)key iff some id satisfies `id % N == n`; at
    ///   `N == 1` the test is a no-op (inert on the unsplit path).
    /// The former separate string-keyed `owners` map (full implication text
    /// per owner, per key, across the four maps) is gone; the I-80 lockstep
    /// is structural now — one container, nothing to drift. Membership
    /// probes still key on the outer map (`find`/`count`), never on this
    /// value.
    ///
    /// `hasLooseOwner` and `uSignatures` drive the request-generation u_ literal
    /// prune (D-120). The four `normalizedEncoded*` maps are
    /// built with `ignoreU=false`, which erases the literal values of a rule's
    /// unchangeable (`u_`) arguments — only the repetition pattern survives — so
    /// a structurally matched request may still be doomed by a u_ literal it can
    /// never satisfy. To catch that at request-generation time without the full
    /// firing lookup, every owner insert records its u_ signature here:
    /// - `uSignatures` — one signature per owner that *has* u_ args. A signature
    ///   is the list of `(linear-arg-slot, required id)` pairs for that owner's
    ///   unchangeable args, where the required id is the owner's `argFullId`
    ///   (== `NameMap::encode(arg[1])`), exactly what the firing gate compares
    ///   against the request's `argFullId`. Deduped across owners (a set).
    /// - `hasLooseOwner` — set when some owner has *zero* u_ args (a
    ///   fully-changeable prefix). Such an owner can always potentially fire, so
    ///   the check is skipped (`ownerSetUSatisfied` short-circuits to keep). This
    ///   is the "0 u_, nothing to check" fast path.
    /// Unlike `partitionIds`, these two are NOT maintained in lockstep
    /// on the radical subtree wipe: stale signatures only ever make the prune
    /// weaker (keep more), never unsound (I-79), so leaving
    /// them is a sound, zero-cost simplification (chosen for runtime).
    struct OwnerSet {
        std::set<int32_t> partitionIds;
        bool hasLooseOwner = false;
        std::set<std::vector<std::pair<int16_t, int16_t>>> uSignatures;
    };

    /// @brief Zero-copy read view over an `OwnerSet`'s canonical blob bytes — the
    ///        request-generation prune reads its fields straight off the cold
    ///        arena, never materializing the two `std::set`s.
    ///
    /// @details
    /// Mirrors the `Codec<OwnerSet>` byte layout one-for-one (keep the two in
    /// sync — a layout change there changes the offsets here):
    /// `hasLooseOwner` (uint8) - `partitionIds` (int32 count, then int32 each,
    /// ascending) - `uSignatures` (int32 count, then per signature an int32
    /// pair-count + each `(int16, int16)` pair). The four `normalizedEncoded*`
    /// maps store ONE blob per key (run-length-1, whole-value replace), so the
    /// view wraps that single blob's `(p, len)` obtained from
    /// `TypedCold::peekRecordBytes`.
    ///
    /// The hot prune touches only what its short-circuits need: `hasLooseOwner`
    /// reads byte 0; the partition walk reads the count + ids only when the LB is
    /// split (`partitionAccepts`) or the request is off-`main`
    /// (`ownerSetHasComparable`); the u_ walk runs only for a non-loose owner
    /// (`ownerSetUSatisfied`). No allocation, no decode. Read-only, burst-safe
    /// ([I-83](docs/agentic_swdd/30_invariants.md#i-83)).
    ///
    /// @invariant Layout-coupled to `Codec<OwnerSet>` — the two are edited
    ///            together.
    /// @see `OwnerSet`, `Codec<OwnerSet>`, `TypedCold::peekRecordBytes`.
    struct OwnerSetBlob {
        const char* p = nullptr;   ///< The single blob's bytes (codec layout).
        int32_t len = 0;           ///< The blob's byte length.

        /// @brief Read a little-endian `int32_t` at byte pointer `q`.
        /// @param q Source pointer (>= 4 readable bytes).
        /// @return The decoded value.
        static int32_t rdI32(const char* q) {
            int32_t v; std::memcpy(&v, q, sizeof(int32_t)); return v;
        }
        /// @brief Read a little-endian `int16_t` at byte pointer `q`.
        /// @param q Source pointer (>= 2 readable bytes).
        /// @return The decoded value.
        static int16_t rdI16(const char* q) {
            int16_t v; std::memcpy(&v, q, sizeof(int16_t)); return v;
        }

        /// @brief Whether some owner is u_-loose (the prune keep fast path).
        /// @return `hasLooseOwner` (byte 0).
        bool hasLooseOwner() const { return p[0] != 0; }

        /// @brief Number of partition (owner) ids.
        /// @return The `partitionIds` count.
        int32_t partitionCount() const { return rdI32(p + 1); }

        /// @brief Composite owner id at index `i`.
        /// @param i Index in `[0, partitionCount())`.
        /// @return The packed `(expandedOriginalId, scopeVid)` composite id.
        int32_t partitionId(int32_t i) const { return rdI32(p + 5 + 4 * i); }

        /// @brief Byte offset of the `uSignatures` section (just past the
        ///        partitionIds run).
        /// @return The offset into the blob.
        int32_t uSigOffset() const { return 5 + 4 * partitionCount(); }

        /// @brief Number of u_ signatures.
        /// @return The `uSignatures` count (int32 at `uSigOffset()`).
        int32_t uSigCount() const { return rdI32(p + uSigOffset()); }

        /// @brief Byte offset of the FIRST signature (just past the sig count).
        /// @return `uSigOffset() + 4`.
        int32_t firstSigOffset() const { return uSigOffset() + 4; }

        /// @brief Pair count of the signature whose record starts at byte @p off.
        /// @param off Byte offset of a signature record (its leading int32).
        /// @return The signature's `(int16,int16)` pair count.
        int32_t sigPairCount(int32_t off) const { return rdI32(p + off); }

        /// @brief Total byte length of the signature record at @p off
        ///        (the int32 pair-count header plus each 4-byte pair).
        /// @param off Byte offset of a signature record.
        /// @return `4 + 4 * sigPairCount(off)` — the offset delta to the next sig.
        int32_t sigBytes(int32_t off) const { return 4 + 4 * sigPairCount(off); }

        /// @brief First component of pair @p i of the signature at @p off.
        /// @param off Byte offset of a signature record.
        /// @param i   Pair index in `[0, sigPairCount(off))`.
        /// @return The pair's `first` (the argument slot).
        int16_t sigPairFirstAt(int32_t off, int32_t i) const {
            return rdI16(p + off + 4 + 4 * i);
        }

        /// @brief Second component of pair @p i of the signature at @p off.
        /// @param off Byte offset of a signature record.
        /// @param i   Pair index in `[0, sigPairCount(off))`.
        /// @return The pair's `second` (the raw name id).
        int16_t sigPairSecondAt(int32_t off, int32_t i) const {
            return rdI16(p + off + 4 + 4 * i + 2);
        }
    };

    /// @brief Pack an owner's (expanded-original id, scope validity id) into the
    ///        single composite `int32` used as its LB-split partition key.
    ///
    /// @details The LB-split hashburst (D-119) divides
    /// a logic block's rules across `N` parallel executors. Each rule is
    /// identified by its owner pair `(expandedOriginal, validity)`; both halves
    /// are already interned to `int16` ids by the per-LB `NameMap`
    /// (`origId = NameMap::encode(expandedOriginal)`,
    /// `scopeVid = NameMap::encode(validity)`). This function packs them —
    /// `origId` in the high 16 bits, `scopeVid` in the low 16 — into one
    /// `int32`. The packing is injective over the non-negative `int16` ids
    /// `NameMap` mints, so distinct owner pairs get distinct composite ids; a
    /// split executor `n` of `N` then accepts a (sub)key iff some composite id
    /// in its owner set satisfies `id % N == n`. The id is a pure function of
    /// the owner — no counter — computed identically at install and at the
    /// radical subtree wipe so `OwnerSet::partitionIds` stays consistent with
    /// `OwnerSet::owners`.
    ///
    /// @param origId   `NameMap::encode(expandedOriginal)` — the rule's id.
    /// @param scopeVid `NameMap::encode(validity)` — the scope's id.
    /// @return The composite `(origId << 16) | (uint16_t)scopeVid`.
    /// @see `OwnerSet::partitionIds` — where the result is stored.
    inline int32_t makePartitionId(int16_t origId, int16_t scopeVid) {
        return (static_cast<int32_t>(origId) << 16)
            | static_cast<int32_t>(static_cast<uint16_t>(scopeVid));
    }

    /// @brief Executor index (`n`) and executor count (`N`) of the currently
    ///        running LB-split hashburst, as thread-local context.
    ///
    /// @details The LB split runs one logic block's hashburst on `N` executor
    /// threads in parallel; each sets these at the top of its per-executor call
    /// (on its own thread, so the two values are per-executor with no
    /// shared-state race) and the request-generation acceptance sites read them
    /// via `partitionAccepts`. They default to the unsplit identity `(0, 1)`, so
    /// any path that never sets them — every path until the split lands, plus
    /// the across-LB phase-1 / phase-3 workers that do no request generation —
    /// sees `splitCount == 1` and the filter is a no-op. Thread-local, mirroring
    /// the existing per-thread context (`g_buildStackPath`, the RT-tracker
    /// hashburst index).
    extern thread_local int g_splitProcessID;
    extern thread_local int g_splitCount;

    /// @brief Whether the current LB runs as more than one concurrent part this
    ///        burst — the per-burst "multi-part" signal the burst early-exit gate
    ///        (I-76) reads instead of `g_splitCount`.
    ///
    /// @details `g_splitCount` is the RULE dimension alone; the expression/bucket
    /// split runs a whole LB as several parts at `g_splitCount == 1` (so
    /// `partitionAccepts` still accepts every rule). The early-exit must be off
    /// whenever an LB runs as > 1 part in EITHER dimension — a sibling bailing on
    /// another part's `stop` is a scheduling race (D-121) — so the gate keys on
    /// this flag, set in `performElem2` from its `partCount` argument
    /// (`= partCount > 1`). Default `false` (unsplit identity); thread-local,
    /// mirroring `g_splitProcessID` / `g_splitCount`.
    extern thread_local bool g_isMultiPart;

    /// @brief Whether the current LB-split executor handles a (sub)key carrying
    ///        the given owner partition ids.
    ///
    /// @details A request generated against a (sub)key is the work of executor
    /// `n = g_splitProcessID` of `N = g_splitCount` iff some composite owner id
    /// in `partitionIds` satisfies `id % N == n` — i.e. the key is owned by a
    /// rule assigned to this executor (see `makePartitionId`,
    /// D-119). With `N <= 1` (the unsplit path) the
    /// test short-circuits to `true` before touching the set, so it is provably
    /// inert: every request the single-executor burst would emit is still
    /// emitted, byte for byte. Composite ids are non-negative (a packing of two
    /// non-negative `int16` halves), so `id % N` lands in `[0, N)` and matches
    /// the executor-index range. The union over `n` of "some id `% N == n`" is
    /// "the set is non-empty" — and a matched key always has at least one owner
    /// ([I-49](../30_invariants.md#i-49)) hence one id — so the executors'
    /// accepted (sub)keys exhaustively cover every key the unsplit burst would
    /// touch (completeness); a key whose ids span residues is accepted by more
    /// than one executor (a harmless duplicate, collapsed by the
    /// order-independent deposit merge, [I-77]).
    ///
    /// @param partitionIds The owner partition ids of the matched (sub)key
    ///                     (`OwnerSet::partitionIds`).
    /// @return `true` if this executor handles the key, `false` to skip it.
    /// @see `makePartitionId`, `OwnerSet::partitionIds`.
    inline bool partitionAccepts(const std::set<int32_t>& partitionIds) {
        const int splitCount = g_splitCount;
        if (splitCount <= 1) return true;
        const int processID = g_splitProcessID;
        for (int32_t id : partitionIds) {
            if (id % splitCount == processID) return true;
        }
        return false;
    }

    /// @brief Owner-partition acceptance read from an `OwnerSet`'s blob bytes —
    ///        the no-decode twin of `partitionAccepts(const std::set&)`.
    ///
    /// @details
    /// Identical verdict to the set overload: at `g_splitCount <= 1` (the unsplit
    /// path) it returns `true` before touching the blob, so the cold owner-set
    /// maps cost nothing on every all-`main`/unsplit workload; otherwise it walks
    /// the partition ids straight off the bytes testing `id % N == n`. See the set
    /// overload for the completeness/soundness argument
    /// ([D-119](docs/agentic_swdd/40_decisions.md#d-119),
    /// [I-49](docs/agentic_swdd/30_invariants.md#i-49)).
    ///
    /// @param entry The matched (sub)key's owner-set blob view.
    /// @return `true` if this LB-split executor handles the key, `false` to skip.
    /// @see `partitionAccepts(const std::set<int32_t>&)`, `OwnerSetBlob`.
    inline bool partitionAccepts(const OwnerSetBlob& entry) {
        const int splitCount = g_splitCount;
        if (splitCount <= 1) return true;
        const int processID = g_splitProcessID;
        const int32_t pc = entry.partitionCount();
        const char* q = entry.p + 5;
        for (int32_t i = 0; i < pc; ++i, q += 4) {
            if (OwnerSetBlob::rdI32(q) % splitCount == processID) return true;
        }
        return false;
    }

    /// @brief One staged algebra-`admissionMap` write recorded during a
    ///        hashburst and replayed after the fixpoint loop by
    ///        `prover.hpp::drainAdmissionKeysAlgebra`.
    ///
    /// @details
    /// `checkLocalEncodedMemoryStatic`'s marker branch discovers, per marker
    /// disintegration, an `admissionMap` entry to insert together with its
    /// companion `admissionStatusMap` / `varsInAdmissionMapKeys` writes and a
    /// `revisitRejected2` revival. Rather than mutate those algebra containers
    /// inline mid-burst, each write is staged as one of these records on
    /// `Memory::admissionKeysAlgebra` and replayed once, after the loop, from a
    /// single drain point — keeping the fixpoint hot path free of algebra
    /// mutation. `key` is the `(markerExpression, validityName)` admission key;
    /// `value` is the `AdmissionMapValue` inserted under it.
    ///
    /// @invariant Records are appended in firing order and drained in that same
    ///            order — the canonical-closure scan in `cleanAdmissionMap`
    ///            (reached transitively via the drain's `revisitRejected2`) is
    ///            order-sensitive.
    /// @see `prover.hpp::drainAdmissionKeysAlgebra` — the sole consumer.
    /// @see `memory.cpp::checkLocalEncodedMemoryStatic` — the sole producer.
    /// @brief Sealed-page twin of `ExpressionWithValidity` — the
    ///        (expression, validity) pair as it rides inside the staging
    ///        records across the pool join.
    ///
    /// @details
    /// Both members are views into the producing task's `SealedPageSet`
    /// (D-164); the consumers materialize
    /// `ExpressionWithValidity` at the deposit boundaries (mail, origin
    /// map) and the drain mint points. No ordering operator — the
    /// firing-record sort compares members explicitly via `compareSpans`,
    /// reproducing the string pair's lexicographic order byte-exactly.
    struct SealedExpressionWithValidity {
        SealedString original;
        SealedString validityName;
    };

    struct AdmissionKeyAlgebraRecord {
        SealedExpressionWithValidity key;
        StagedAdmissionValue value;
    };

    /// @brief One deferred integration-template registration — the inputs to a
    ///        `prepareIntegration` call staged during a hashburst and replayed
    ///        after the fixpoint loop by `prover.hpp::drainDeferredIntegrationPreps`.
    ///
    /// @details
    /// Integration-side analog of `AdmissionKeyAlgebraRecord`. The only
    /// `admissionMapIntegration` write that runs *inside* the hashburst is the
    /// `prepareIntegration` seed registration in `checkLocalEncodedMemoryStatic`'s
    /// marker branch (which writes `admissionMapIntegration` and the
    /// `varsInAdmissionMapIntegrationKeys` cache via `prepareIntegrationCore2`).
    /// Rather than register inline mid-burst, the call's inputs are staged as one
    /// of these records on `Memory::deferredIntegrationPreps` and replayed once,
    /// after the loop, from a single drain point — keeping the fixpoint hot path
    /// free of integration-admission mutation. `expression` is the marker-form
    /// `rplExpr2`; `unchangeableArgs` is its bare argument set (marker removed);
    /// `validityName` is the scope. The owning `Memory` is supplied by the drain.
    ///
    /// @invariant Records are appended in firing order and drained in that same
    ///            order.
    /// @see `prover.hpp::drainDeferredIntegrationPreps` — the sole consumer.
    /// @see `memory.cpp::checkLocalEncodedMemoryStatic` — the sole producer.
    struct DeferredIntegrationPrep {
        // Sealed views (D-164); the drain
        // materializes at the prepareIntegration boundary. Args are the
        // former set as a sorted-unique sealed span on the page set.
        SealedString expression;
        SealedSpan<SealedString> unchangeableArgsSorted;
        SealedString validityName;
    };

    /// @brief One captured rule-firing from a hashburst — the deposit side of
    ///        `checkLocalEncodedMemoryStatic`, recorded instead of applied
    ///        inline so a whole burst's deposits can be sorted into a canonical
    ///        order before they touch the LB's containers.
    ///
    /// @details
    /// The static request-evaluation pass (the FIXPOINT_LOOP in
    /// `prover.cpp::performElem2`) historically mutated
    /// `sameIterationInternalMail` and the two staging vectors
    /// (`admissionKeysAlgebra`, `deferredIntegrationPreps`) directly, in
    /// request-generation order. That order is not partition-independent once an
    /// LB's rules are split across copies, so the cap-bounded `addOrigin`
    /// selection, the per-head `disintegrationSignals` last-write, and the
    /// firing-order drains would all differ per partition. Instead each firing
    /// now produces a `FiringRecord`; after the pass
    /// `ExpressionAnalyzer::applyFiringRecords` sorts the records by a canonical
    /// content key and applies their deposits, so the resulting container state
    /// is a function of the firing SET, not the firing ORDER — the property the
    /// LB split relies on.
    ///
    /// Deactivation is NOT recorded here: the phase-2 burst is read-only on the
    /// LB (I-66). A head firing that closes the LB is captured like any other
    /// firing; the actual deactivation + discharge run in phase 3's post-burst
    /// `standardProcessing` (`dischargeContradiction` / `dischargeToBeProved`).
    ///
    /// A record is either a head firing (`isMarker == false`) or a marker /
    /// admission firing (`isMarker == true`); the per-kind fields are only
    /// meaningful for the matching kind.
    ///
    /// @see `memory.cpp::checkLocalEncodedMemoryStatic` — the producer.
    /// @see `prover.hpp::applyFiringRecords` — the consumer (sort + apply).
    /// @see D-117.
    struct FiringRecord {
        bool isMarker = false;
        SealedString rplExpr2;       // head/marker expression deposited
        SealedString validityName;   // consensus validity scope of the firing

        // --- head firing (isMarker == false) ---
        // Sorted-unique ascending — the former std::set<int>, now a sealed
        // int span on the page set; the comparator and the mail deposit
        // preserve the set's ordering.
        SealedSpan<int> levels;
        SealedString originTag;
        SealedSpan<SealedExpressionWithValidity> originDeps;
        bool doNotDisintegrate = false;
        bool allowOrDisintegration = false;
        bool allGood = false;
        bool alreadyKnown = false;

        // --- marker firing (isMarker == true) ---
        // Sorted lex (decoded byte order), deduplicated — the former
        // std::set<std::string> as a flat sealed span; comparator and
        // consumers preserve the set's lexicographic semantics.
        SealedSpan<SealedString> markerArgsSorted;
        StagedAdmissionValue admv;
        bool markerNotAtomic = false;
    };

    /// @brief A typed sequence of `LogicalEntity` items plus a marked goal — one
    /// integration-side request shape.
    ///
    /// @details
    /// Used inside `HashMemory::admissionMapIntegration` (keyed on
    /// `ExpressionWithValidity`, valued as `map<Instruction, set<string>>`) to
    /// record, per scope, which structured request bodies have been admitted and
    /// which sibling expressions still gate them. The `data` vector holds the
    /// per-element `LogicalEntity` description; `markedGoal` is the consequent the
    /// instruction is steering toward. Together they form a stable, deduplicated
    /// description of an integration request.
    ///
    /// `operator<` orders lexicographically over `(data, markedGoal)`; both
    /// underlying types provide their own ordering, so the comparison is
    /// well-defined and stable.
    ///
    /// @see [`LogicalEntity`](#logicalentity) — the element type.
    /// @see [`HashMemory::admissionMapIntegration`](#hashmemory) — owns this map.
    struct Instruction {
        // Members
        std::vector<LogicalEntity> data;
        std::string markedGoal;

        // 1. Default Constructor
        Instruction()
            : data(), markedGoal() {
        }

        // 2. Parameterized Constructor
        Instruction(std::vector<LogicalEntity> insts, std::string goal)
            : data(std::move(insts)), markedGoal(std::move(goal)) {
        }

        // 3. Operator < (Required for std::map key)
        // Uses lexicographical comparison: first compares instructions, then markedGoal.
        bool operator<(const Instruction& rhs) const {
            if (data != rhs.data) {
                return data < rhs.data;
                // This works because std::vector and LogicalEntity already implement operator<
            }
            return markedGoal < rhs.markedGoal;
        }

        // 4. Equality Operator
        bool operator==(const Instruction& rhs) const {
            return data == rhs.data && markedGoal == rhs.markedGoal;
        }
    };

    // ========================================================================
    // Static hot path: int16_t-based hash memory structures
    // ========================================================================

    /// Bidirectional string <-> int16_t dictionary. Simple counter, no intelligence.
    ///
    /// A validityName is a LIFO stack of sub-names. Canonical grammar:
    ///
    ///   validityName := Root ( "_boundary_" Payload )*
    ///   Root         := arbitrary string; MUST NOT contain "_boundary_"
    ///   Payload      := arbitrary string; MUST NOT contain "_boundary_"
    ///
    /// The root is typically the literal "main" (the general validity scope),
    /// but any string free of the "_boundary_" delimiter is accepted as a
    /// root. Each deeper scope is formed by appending "_boundary_" + payload.
    /// Prefix relationship between two ids is answered by `verdict(a, b)`:
    /// -1 if a is a strict prefix of b, +1 if b is a strict prefix of a, 0 if
    /// a == b, diverge if neither. It is DERIVED from `ancestorsOf` (a is a
    /// strict ancestor of b iff a is in ancestorsOf[b]); no verdict cache.
    /// Storage note (strings campaign): the two string dictionaries are
    /// FAÇADES over cold string tables inside `LbMemory` (`nameStrings`,
    /// `subStrings` — D-153); the strings live on the
    /// LB's static pages and deload with it. The id-form bookkeeping is a flat
    /// per-id parent-pointer forest (`validityNodes`, one `ValidityNode` each)
    /// paged on the LB's arena and deloading with it too; ancestor and
    /// payload-stack queries walk the parent chain. "main" is the eternal root
    /// id 1, lazily interned as cold-table id 1 on the first encode, so a
    /// NameMap id equals its cold-table id directly (no offset) and an unused
    /// `Memory` consumes zero pool blocks (the lazy contract). The former
    /// `stringAncestorsOf` mirror is deleted — `isStrictAncestor` answers
    /// from `verdict` over `ancestorsOf`, `strictAncestorNames` decodes
    /// `ancestorsOf` lex-sorted (the set's old iteration order).
    struct NameMap {
        ColdStringTable* names = nullptr;  // canonical names (ids 1.., main on-table)
        ColdStringTable* subs = nullptr;   // payload strings (ids 1..)

        // Per-validity-id scope-forest node {parentId, ownSubId} — a flat
        // PagedVector owned by LbMemory, bound here as a façade pointer. Indexed
        // by validity id directly (id 0 = unused, id 1 = "main"); ancestor and
        // payload-stack queries walk the parentId chain. Lazily seeded on the
        // first encode so a transient Memory consumes no pool blocks.
        PagedVector<ValidityNode>* nodes = nullptr;

        static constexpr const char* BOUNDARY_STR = "_boundary_";
        static constexpr std::size_t BOUNDARY_LEN = 10;  // strlen("_boundary_")

        /// Canonical "main" (general validity scope) is always id 1.
        /// Guaranteed structurally: "main" is the special-cased tableless
        /// id, lazily seeded into the metadata on the first encode.
        static constexpr int16_t MAIN_ID = 1;

        NameMap() = default;  // unbound; Memory's constructor binds

        /// @brief The special-cased name of MAIN_ID (no table bytes).
        ///
        /// @return A span over the literal "main" (process-stable static
        ///         storage; no heap `std::string` is constructed).
        static StrSpan mainName() {
            static constexpr char kMain[] = "main";
            return StrSpan(kMain, static_cast<int32_t>(sizeof(kMain) - 1));
        }

        /// @brief Bind the façade to its cold tables and the paged validity-node
        ///        forest (idempotent for the same backings).
        ///
        /// @details
        /// The node container is NOT seeded here: `bind` runs in every `Memory`
        /// ctor and a transient `Memory` must consume zero pool blocks. The
        /// slot-0 + "main" nodes are appended lazily on the first
        /// `encode` / `encodePush` (`seedIfEmpty`), so only an LB that actually
        /// mints a scope touches the arena.
        ///
        /// @param n  The canonical-name table (`LbMemory::nameStrings`).
        /// @param s  The payload table (`LbMemory::subStrings`).
        /// @param nd The validity-node forest (`LbMemory::validityNodes`).
        void bind(ColdStringTable* n, ColdStringTable* s,
                  PagedVector<ValidityNode>* nd) {
            assert(n != nullptr && s != nullptr && nd != nullptr);
            assert((names == nullptr || names == n)
                && "NameMap rebind to a different table");
            names = n;
            subs = s;
            nodes = nd;
        }

        /// @brief Reset the paged node forest to empty — the `destroyGrid`
        ///        lockstep (slot-0 + "main" re-seed lazily on the next encode).
        void initMetadata() {
            assert(nodes != nullptr && "NameMap metadata used unbound");
            nodes->clear();
        }

        /// @brief Intern "main" as names-table id 1 — the lazy first touch that
        ///        makes a NameMap id equal its cold-table id (the direct
        ///        mapping, as on `main`).
        ///
        /// @details
        /// "main" is the eternal root scope: conceptually id 1 from the start,
        /// but kept off the table until the LB's first encode so a transient
        /// `Memory` allocates nothing. Interning it as the first names-table
        /// entry makes the table id equal the NameMap id with no offset; reads
        /// of `MAIN_ID` before this point resolve through the eternal-root
        /// branches in `lookup` / `decode` / `decodeView`.
        void internMain() {
            assert(names != nullptr && "NameMap used unbound");
            assert(names->count() == 0
                && "internMain after the names table already grew");
            const int32_t tid = names->intern(mainName());
            assert(tid == MAIN_ID && "main must intern as names table id 1");
            (void)tid;
        }

        /// @brief Intern "main" (id 1) and append the slot-0 sentinel + "main"
        ///        nodes on first use — the lazy seed that keeps a transient
        ///        `Memory` at zero pool blocks.
        ///
        /// @details
        /// Called at the top of `encode` / `encodePush`, so the first scope mint
        /// (or the first `encode("main")`) materializes both the names-table
        /// "main" row (`internMain`, cold-table id 1) and its node; an LB that
        /// never encodes stays empty. Slot 0 (id 0) is the unused sentinel;
        /// "main" (id 1) is a root — `parentId 0`, no payload. Idempotent — a
        /// non-empty container returns immediately.
        void seedIfEmpty() {
            if (!nodes->empty()) return;
            internMain();                            // "main" -> cold-table id 1
            nodes->push_back(ValidityNode{0, 0});    // slot 0 unused sentinel
            nodes->push_back(ValidityNode{0, 0});    // "main" (id 1): root, no payload
        }

        /// @brief Wholesale reset in lockstep with `destroyGrid` — every
        ///        id re-binds afterwards; "main" stays id 1.
        void resetToFresh() {
            assert(names != nullptr && "NameMap used unbound");
            names->resetToFresh();
            subs->resetToFresh();
            initMetadata();
        }

        /// @brief Deep copy for the CE clone path (`cloneFactsTemplate`) —
        ///        tables re-interned in id order onto THIS LB's manager,
        ///        metadata copied; every id matches the source.
        ///
        /// @param other The template's NameMap.
        void copyFrom(const NameMap& other) {
            assert(names != nullptr && other.names != nullptr);
            assert(nodes != nullptr && other.nodes != nullptr);
            names->resetToFresh();
            subs->resetToFresh();
            names->copyFrom(*other.names);
            subs->copyFrom(*other.subs);
            *nodes = *other.nodes;   // PagedVector deep copy (clears then copies)
        }

        /// Encode a validity name into a stable id, registering metadata on first sight.
        /// Any string without "_boundary_" inside it is registered as a root with
        /// an empty stack (canonical root = "main"). A string containing
        /// "_boundary_" is split at the last delimiter: everything before it is
        /// recursively encoded as the parent, and the tail is pushed as payload.
        int16_t encode(const std::string& s) {
            // Forward to the span overload — the span-native implementation
            // (flat-mint / scoped parent-then-payload recursion), no heap
            // std::string. Byte-identical: same bytes, same registration path.
            return encode(StrSpan(s));
        }

        /// @brief Span overload of @ref encode — register a validity name from
        ///        raw bytes without first materializing a heap `std::string`.
        ///
        /// @details
        /// The absorb door (the `encodeExpression` span twin and the U-prefix
        /// helpers) holds expression and validity bytes as `StrSpan` slices of
        /// stable caller buffers. This overload interns a name straight from
        /// such a slice. The common flat-name case (no `"_boundary_"`) probes
        /// through the existing `lookup(const StrSpan&)` and, on a miss, mints
        /// directly from the span — no `std::string` is built. The scoped case
        /// (a name containing `"_boundary_"`, i.e. a nested validity scope)
        /// slices the parent prefix and tail payload as spans and recurses
        /// span-native (register the parent, then `encodePush` the payload), so
        /// the whole overload is heap-free and the resulting id is byte-identical
        /// to encoding the same bytes as a `std::string`.
        ///
        /// @param s The name bytes to encode; a slice of a caller-stable buffer.
        /// @return The interned id (existing on a hit, freshly minted on a miss).
        /// @see `encode(const std::string&)` — the forwarding twin.
        /// @see `lookup(const StrSpan&)` — the non-minting probe it reuses.
        int16_t encode(const StrSpan& s) {
            seedIfEmpty();
            const int16_t found = lookup(s);
            if (found != 0) return found;

            const std::string_view sv(s.ptr, static_cast<std::size_t>(s.len));
            const std::size_t lastPos =
                sv.rfind(BOUNDARY_STR, std::string_view::npos, BOUNDARY_LEN);
            if (lastPos == std::string_view::npos) {
                // Flat name — mint directly from the span; no heap std::string.
                // Same delimiter self-overlap rejection as the std::string
                // overload's flat path (I-139
                // precondition).
                assert(!(s.len >= 9
                         && equalSpans(StrSpan(s.ptr + s.len - 9, 9),
                                       StrSpan("_boundary", 9)))
                       && "flat name must not end with '_boundary'");
                assert(!(s.len >= 9
                         && equalSpans(StrSpan(s.ptr, 9),
                                       StrSpan("boundary_", 9)))
                       && "flat name must not start with 'boundary_'");
                assert(names != nullptr && "NameMap used unbound");
                const int32_t id = names->intern(s);
                assert(id < ExecutionParameters::MAX_NAME_IDS);
                assert(id == nodes->size()
                    && "NameMap id/node misalignment (span mint outside encode)");
                nodes->push_back(ValidityNode{0, 0});
                return static_cast<int16_t>(id);
            }
            // Scoped name — slice the parent prefix and the tail payload as
            // spans and recurse span-native (no heap std::string): register the
            // parent, then push the payload. Byte-identical to encoding the
            // same bytes as a std::string.
            const StrSpan parent(s.ptr, static_cast<int32_t>(lastPos));
            const StrSpan payload(
                s.ptr + lastPos + BOUNDARY_LEN,
                s.len - static_cast<int32_t>(lastPos + BOUNDARY_LEN));
            const int16_t parentId = encode(parent);
            return encodePush(parentId, payload);
        }

        /// @brief Mint a brand-new canonical name into the cold table and
        ///        return its NameMap id — the cold-table id directly ("main" is
        ///        interned as id 1 by `internMain`, no offset). Caller pushes
        ///        the metadata.
        ///
        /// @param s The canonical name; not yet interned, never "main".
        /// @return The fresh id; asserts the int16 ceiling.
        int16_t mintName(const std::string& s) {
            assert(names != nullptr && "NameMap used unbound");
            const int32_t id = names->intern(StrSpan(s));
            assert(id < ExecutionParameters::MAX_NAME_IDS);
            assert(id == nodes->size()
                && "NameMap id/node misalignment — a name was minted "
                   "outside encode/encodePush");
            return static_cast<int16_t>(id);
        }

        /// @brief Span overload of @ref mintName — mint a canonical name from
        ///        raw bytes (a scratch / cold slice) with no heap `std::string`.
        /// @param s The canonical name bytes; not yet interned, never "main".
        /// @return The fresh id; asserts the int16 ceiling and node alignment.
        int16_t mintName(const StrSpan& s) {
            assert(names != nullptr && "NameMap used unbound");
            const int32_t id = names->intern(s);
            assert(id < ExecutionParameters::MAX_NAME_IDS);
            assert(id == nodes->size()
                && "NameMap id/node misalignment — a name was minted "
                   "outside encode/encodePush");
            return static_cast<int16_t>(id);
        }

        /// @brief Non-minting variant of `encode` — return the id of an
        ///        already-interned name, or the slot-0 sentinel if absent.
        ///
        /// @details
        /// `encode` is find-or-mint: an unknown name is inserted, mutating the
        /// `NameMap` (the cold names table and its find-index, `ancestorsOf`,
        /// …). `lookup` is the
        /// read-only half — it returns the existing id when present and `0`
        /// (slot 0 is the reserved invalid entry, never a
        /// real name) when absent, and never mutates any container, so it is
        /// `const` and safe to call concurrently from the LB-split executor
        /// threads on a shared LB. A name that has never been interned cannot
        /// appear in `intKnownStatements` (which keys on interned ids), so a `0`
        /// result drives the same not-known / no-contradiction decision a
        /// mint-then-miss would — without the write that would be a data race.
        ///
        /// @param s The name to look up.
        /// @return The interned id of `s`, or `0` if `s` was never interned.
        /// @see `encode` — the minting counterpart.
        int16_t lookup(const std::string& s) const {
            return lookup(StrSpan(s));
        }

        /// @brief Span overload of the non-minting probe — burst hot paths
        ///        probe sealed/hot views without materializing.
        ///
        /// @param s The name bytes to look up.
        /// @return The interned id, or `0` if absent.
        int16_t lookup(const StrSpan& s) const {
            assert(names != nullptr && "NameMap used unbound");
            const int32_t tid = names->lookup(s);
            if (tid != 0) return static_cast<int16_t>(tid);   // direct, no offset
            // "main" is the eternal root — id 1 even before it is lazily
            // interned into the names table (an idle Memory has none).
            if (equalSpans(s, mainName())) return MAIN_ID;
            return 0;
        }

        /// Intern a payload string, returning a stable sub-id.
        int16_t encodeSub(const std::string& payload) {
            assert(subs != nullptr && "NameMap used unbound");
            const int32_t id = subs->intern(StrSpan(payload));
            assert(id < ExecutionParameters::MAX_NAME_IDS);
            return static_cast<int16_t>(id);
        }

        /// @brief Span overload of @ref encodeSub — intern a payload from raw
        ///        bytes (a scratch / cold slice) with no heap `std::string`.
        /// @param payload The payload bytes; a slice of a caller-stable buffer.
        /// @return The stable sub-id; asserts the int16 ceiling.
        int16_t encodeSub(const StrSpan& payload) {
            assert(subs != nullptr && "NameMap used unbound");
            const int32_t id = subs->intern(payload);
            assert(id < ExecutionParameters::MAX_NAME_IDS);
            return static_cast<int16_t>(id);
        }

        /// Push `payload` onto `parentId`'s stack, returning id of the new scope.
        /// Deduplicates via canonical string (parent + "_boundary_" + payload).
        /// Asserts payload does not contain the "_boundary_" delimiter, does
        /// not end with "_boundary", and does not start with "boundary_" —
        /// the delimiter self-overlaps with period 9, so either overlap shape
        /// would create a spurious delimiter occurrence at a concatenation
        /// junction and break the delimiter-decomposition uniqueness the
        /// forest predicate rests on (I-139).
        int16_t encodePush(int16_t parentId, const std::string& payload) {
            return encodePush(parentId, StrSpan(payload));
        }

        /// @brief Span overload of @ref encodePush — the heap-free scope-mint
        ///        primitive (defined out-of-line in memory.cpp).
        ///
        /// @details
        /// Builds the canonical scope name `decode(parentId) + "_boundary_" +
        /// payload` on the per-slot string-scratch arena under a `ScratchScope`
        /// window (no heap `std::string`), then dedups via `lookup` and, on a
        /// miss, mints the payload sub-id and the canonical name id in that
        /// order — byte-identical ids and mint order to the former
        /// heap-`std::string` body. It is out-of-line because it reaches the
        /// scratch registry (`scratchArenas()`) and the worker-slot id
        /// (`ExpressionAnalyzer::g_currentCoreId`), neither of which an inline
        /// header body can see. The `decodeView(parentId)` span is copied into
        /// the scratch buffer before any mint, so the later `names` mint cannot
        /// dangle it (I-3).
        ///
        /// @param parentId The parent validity id.
        /// @param payload  The scope payload bytes — delimiter-free by the same
        ///                 self-overlap contract the `std::string` overload
        ///                 asserted; a slice of a caller-stable buffer.
        /// @return The interned scope id (existing on a canonical hit, freshly
        ///         minted on a miss).
        /// @see encodePush(int16_t, const std::string&) — the delegating twin.
        /// @see mintName(const StrSpan&), encodeSub(const StrSpan&) — the span
        ///      mint primitives it uses.
        int16_t encodePush(int16_t parentId, const StrSpan& payload);

        /// @brief Recover the canonical name of an id (owned copy — the
        ///        cold bytes live on pages). Hot readers that only
        ///        compare, slice, or hash use `decodeView`.
        ///
        /// @param id A real id (`>= 1`); "main" decodes without a table
        ///           touch.
        /// @return The canonical name (owned copy).
        std::string decode(int16_t id) const {
            assert(names != nullptr && "NameMap used unbound");
            assert(id >= MAIN_ID);
            // "main" is the eternal root: MAIN_ID resolves whether or not it has
            // been lazily interned. Once interned it reads from the table like
            // any name, so a real LB's dump shows the container as-is.
            if (id == MAIN_ID && names->count() == 0) {
                const StrSpan m = mainName();
                return std::string(m.ptr, static_cast<std::size_t>(m.len));
            }
            return names->decodeString(id);
        }

        /// @brief Zero-copy decode — a span over the cold bytes, stable
        ///        while the LB is resident; "main" spans a process-stable
        ///        literal.
        ///
        /// @param id A real id (`>= 1`).
        /// @return Span over the name's bytes.
        StrSpan decodeView(int16_t id) const {
            assert(names != nullptr && "NameMap used unbound");
            assert(id >= MAIN_ID);
            if (id == MAIN_ID && names->count() == 0) return mainName();
            return names->view(id);
        }

        /// @brief Number of interned names including "main" (ids run
        ///        1..nameCount(); `0` before the first encode interns "main").
        ///
        /// @return The id high-water mark — the cold-table count directly, now
        ///         that "main" is on the table (no off-table +1).
        int32_t nameCount() const {
            assert(names != nullptr && "NameMap used unbound");
            return names->count();
        }

        /// @brief Number of interned payloads (sub-ids run 1..subCount()).
        ///
        /// @return The sub-id high-water mark.
        int32_t subCount() const {
            assert(subs != nullptr && "NameMap used unbound");
            return subs->count();
        }

        /// @brief Recover a payload string (owned copy).
        ///
        /// @param id A real sub-id (`>= 1`).
        /// @return The payload (owned copy).
        std::string decodeSub(int16_t id) const {
            assert(subs != nullptr && "NameMap used unbound");
            return subs->decodeString(id);
        }

        /// @brief Zero-copy payload decode — a span over the sub-table's cold
        ///        bytes, the `decodeSub` twin.
        ///
        /// @details
        /// The validity-stack payload readers (`classifyOrScope`'s `_(...)`
        /// prefix probe, `dischargeToBeProved`'s `bareSig`) only compare and
        /// slice the payload, so they can read the interned bytes in place
        /// rather than take the `decodeSub` heap copy. Mirrors `decodeView`
        /// (the name-table twin) exactly: the span is stable while the LB is
        /// resident, and a nested mint into `subs` may reallocate the page it
        /// points at, so copy before minting (I-3) as for the name-table
        /// spans.
        ///
        /// @param id A real sub-id (`>= 1`).
        /// @return Span over the payload's bytes.
        /// @see decodeSub — the owning-copy sibling; decodeView — the
        ///      name-table span twin.
        StrSpan decodeSubView(int16_t id) const {
            assert(subs != nullptr && "NameMap used unbound");
            return subs->view(id);
        }

        /// @brief The immediate parent validity id of `id` — `0` for a root
        ///        (`"main"` and any flat `encode` root); the edge the scope
        ///        forest walks. On an unseeded map a root id resolves to `0`.
        ///
        /// @param id A validity id.
        /// @return The parent id, or `0` if `id` is a root.
        int16_t parentOf(int16_t id) const {
            if (nodes->empty()) {
                assert((id == 0 || id == MAIN_ID)
                    && "parentOf on an unseeded NameMap for a non-root id");
                return 0;
            }
            return (*nodes)[id].parentId;
        }

        /// @brief The own payload sub-id of validity `id` — `0` for a root (no
        ///        `_boundary_` payload).
        ///
        /// @param id A validity id.
        /// @return The scope's own payload sub-id, or `0` for a root.
        int16_t ownSubOf(int16_t id) const {
            if (nodes->empty()) {
                assert((id == 0 || id == MAIN_ID)
                    && "ownSubOf on an unseeded NameMap for a non-root id");
                return 0;
            }
            return (*nodes)[id].ownSubId;
        }

        /// @brief Number of validity rows — slot-0 sentinel + "main" + every
        ///        minted scope; `0` until the first encode seeds the metadata.
        ///
        /// @return The validity-id high-water (one past the last id).
        int32_t stackSize() const { return nodes->size(); }

        /// @brief Whether validity `id`'s payload stack is empty (a root such
        ///        as "main").
        ///
        /// @details
        /// On an unseeded map (no encode yet) `MAIN_ID` and the slot-0 sentinel
        /// resolve via the eternal-root fallback (both empty); a non-root id
        /// cannot exist there and asserts.
        ///
        /// @param id A validity id.
        /// @return `true` when the stack has no payload sub-ids.
        bool stackEmpty(int16_t id) const {
            return parentOf(id) == 0;   // a root (main / flat root) has no payloads
        }

        /// @brief Top (deepest) payload sub-id of validity `id`'s stack.
        ///
        /// @param id A seeded validity id with a non-empty stack.
        /// @return The deepest payload sub-id.
        int16_t stackBack(int16_t id) const {
            assert(!stackEmpty(id) && "stackBack on an empty (root) stack");
            return ownSubOf(id);   // the innermost payload IS this scope's own sub-id
        }

        /// @brief Depth of validity `id`'s payload stack.
        ///
        /// @details
        /// On an unseeded map `MAIN_ID` / slot 0 resolve to depth 0 via the
        /// eternal-root fallback; a non-root id there asserts.
        ///
        /// @param id A validity id.
        /// @return The number of payload sub-ids.
        int32_t stackLen(int16_t id) const {
            const int32_t a = ancLen(id);   // payloads = ancestors minus the root self
            return a > 0 ? a - 1 : 0;
        }

        /// @brief Payload sub-id at depth `k` of validity `id`'s stack.
        ///
        /// @param id A seeded validity id.
        /// @param k  Depth in `[0, stackLen(id))`.
        /// @return The payload sub-id.
        int16_t stackAt(int16_t id, int32_t k) const {
            // the k-th payload (root-to-leaf) is the own sub-id of the (k+1)-th
            // ancestor (index 0 is the payload-less root).
            return ownSubOf(ancAt(id, k + 1));
        }

        /// @brief Number of ancestor-list rows — equals `stackSize()` (both
        ///        indexed by validity id).
        ///
        /// @return The validity-id high-water for the ancestor store.
        int32_t ancSize() const { return nodes->size(); }

        /// @brief Length of validity `id`'s ancestor list (includes self).
        ///
        /// @details
        /// On an unseeded map `MAIN_ID` resolves to 1 (ancestors `{main}`) and
        /// slot 0 to 0 via the eternal-root fallback; a non-root id asserts.
        ///
        /// @param id A validity id.
        /// @return The ancestor count.
        int32_t ancLen(int16_t id) const {
            if (nodes->empty()) {
                assert((id == 0 || id == MAIN_ID)
                    && "ancLen on an unseeded NameMap for a non-root id");
                return id == MAIN_ID ? 1 : 0;
            }
            int32_t n = 0;
            for (int16_t cur = id; cur != 0; cur = (*nodes)[cur].parentId)
                ++n;
            return n;
        }

        /// @brief Ancestor at position `k` of validity `id`'s ancestor list.
        ///
        /// @details
        /// On an unseeded map only main's self-link (`MAIN_ID`, `k == 0`) is
        /// defined, resolving to `MAIN_ID` via the eternal-root fallback.
        ///
        /// @param id A validity id.
        /// @param k  Position in `[0, ancLen(id))`.
        /// @return The ancestor validity id (self at the back).
        int16_t ancAt(int16_t id, int32_t k) const {
            if (nodes->empty()) {
                assert(id == MAIN_ID && k == 0
                    && "ancAt on an unseeded NameMap outside main's self-link");
                return MAIN_ID;
            }
            const int32_t len = ancLen(id);
            assert(k >= 0 && k < len && "ancAt position out of range");
            // root-to-leaf index k is (len-1-k) parent-steps up from id (leaf).
            int16_t cur = id;
            for (int32_t steps = len - 1 - k; steps > 0; --steps)
                cur = (*nodes)[cur].parentId;
            return cur;
        }

        /// Split a validity name into its ordered payload list.
        /// The root prefix (everything up to the first "_boundary_") is
        /// discarded — the root has an empty stack. Returns payloads in
        /// bottom-to-top order. Returns {} for a pure root (e.g. "main").
        std::vector<std::string> parse(const std::string& s) const {
            std::vector<std::string> out;
            std::size_t pos = s.find(BOUNDARY_STR, 0, BOUNDARY_LEN);
            if (pos == std::string::npos) return out;   // pure root
            pos += BOUNDARY_LEN;
            while (true) {
                std::size_t next = s.find(BOUNDARY_STR, pos, BOUNDARY_LEN);
                if (next == std::string::npos) {
                    out.push_back(s.substr(pos));
                    break;
                }
                out.push_back(s.substr(pos, next - pos));
                pos = next + BOUNDARY_LEN;
            }
            return out;
        }

        /// @brief Whether `q` appears in `id`'s ancestor list — the membership
        ///        primitive the `verdict` / `comparable` derivation rests on.
        ///
        /// @details
        /// Walks `id`'s parent chain (id, its parent, …, up to a root); a hit on
        /// `q` with `q != id` means `q` is a STRICT ancestor of `id`. This is the
        /// single site that walks the forest for a verdict. Linear in the
        /// ancestor-chain length (validity nesting depth — small in practice);
        /// the callers are cold paths, never the parallel burst.
        ///
        /// @param id A minted validity id; `0 <= id < stackSize()`.
        /// @param q  The id to search for.
        /// @return `true` when some scope on `id`'s parent chain equals `q`.
        bool ancContains(int16_t id, int16_t q) const {
            if (nodes->empty()) {
                assert((id == 0 || id == MAIN_ID)
                    && "ancContains on an unseeded NameMap for a non-root id");
                return id == MAIN_ID && q == MAIN_ID;   // main: {main}; slot 0: {}
            }
            for (int16_t cur = id; cur != 0; cur = (*nodes)[cur].parentId)
                if (cur == q) return true;
            return false;
        }

        /// @brief Build the closed-subtree membership set for a subtree wipe —
        ///        one `validityNodes` forest walk instead of a per-id string
        ///        decode.
        ///
        /// @details
        /// Fills `bits` (a caller-owned bitmap, one bit per validity id) and
        /// `ascendingOut` (the matching ids in ascending order) with every
        /// minted id whose parent chain contains `closedVid`. The chain's
        /// first element is the id itself (`ancContains` includes self), so
        /// the closed scope's own bit IS set — matching the retired
        /// `v == closedScope` branch of the text predicate.
        ///
        /// Equivalence to the retired text predicate (`v == closedScope ||
        /// v starts with closedScope + "_boundary_"`) is a theorem, not a
        /// grammar hope:
        /// 1. Every mint path ASSERTS the overlap-free precondition set: a
        ///    payload / flat name contains no `"_boundary_"`, does not end
        ///    with `"_boundary"`, and does not start with `"boundary_"`
        ///    (the latter two close the delimiter's period-9 self-overlap)
        ///    — so no minted fragment can embed, extend into, or complete
        ///    a delimiter occurrence at a concatenation junction.
        /// 2. `encode` of any name CONTAINING the delimiter splits at the
        ///    LAST delimiter, recursively encodes the prefix as the parent,
        ///    and `encodePush`es the tail — so even a mail-absorbed deep
        ///    name receives the maximal-split parent chain (no
        ///    first-mint-wins parentage hazard).
        /// 3. Therefore every minted name's delimiter decomposition is
        ///    unique and equals its recorded parent chain:
        ///    text-prefix-with-delimiter membership == forest ancestry.
        ///    Flat names (`ValidityNode{0,0}`, e.g. statement originals
        ///    interned by `encode`) contain no delimiter, can never carry
        ///    the prefix, and their chain is {self} — both predicates say
        ///    "self only" there too.
        /// Complexity O(nameCount() x chain depth); replaces the former
        /// decode-every-id-to-an-owned-`std::string` scan.
        ///
        /// @param closedVid    The closed scope's minted validity id
        ///                     (`1 <= closedVid <= nameCount()`), never
        ///                     `MAIN_ID` (`"main"` is never closed) — both
        ///                     assert-enforced.
        /// @param bits         Caller-owned bitmap, zeroed here; bit `id`
        ///                     (`bits[id >> 6]`, bit `id & 63`) is set for
        ///                     every matching id.
        /// @param words        Bitmap capacity in 64-bit words. Asserts
        ///                     `nameCount() < words * 64` — the Rule-19
        ///                     tripwire against an id-ceiling raise outrunning
        ///                     the caller's stack buffer.
        /// @param ascendingOut Receives the matching ids in ascending order
        ///                     (the wipe's step-11 mint order).
        /// @return The number of matching ids.
        /// @invariant `I-139` — wipe-subtree membership
        ///            is the forest walk, provably equivalent to the retired
        ///            text-prefix predicate; [I-2](../../docs/agentic_swdd/30_invariants.md#i-2)
        ///            — every non-`"main"` scope is minted via `encodePush`.
        /// @see `ancContains`, `Memory::wipeSubtree`.
        int32_t collectClosedSubtreeIds(int16_t closedVid,
                                        uint64_t* bits, int32_t words,
                                        PagedVector<int16_t>& ascendingOut) const {
            assert(closedVid >= 1 && closedVid <= nameCount()
                && "collectClosedSubtreeIds: closedVid is not a minted id");
            assert(closedVid != MAIN_ID
                && "collectClosedSubtreeIds: 'main' is never closed");
            assert(nameCount() < words * 64
                && "collectClosedSubtreeIds: bitmap too small for nameCount");
            for (int32_t w = 0; w < words; ++w) bits[w] = 0;
            int32_t matches = 0;
            const int32_t hi = nameCount();
            for (int16_t id = 1; id <= hi; ++id) {
                if (ancContains(id, closedVid)) {
                    bits[static_cast<uint16_t>(id) >> 6] |=
                        (1ull << (static_cast<uint16_t>(id) & 63));
                    ascendingOut.push_back(id);
                    ++matches;
                }
            }
            return matches;
        }

        /// @brief Ancestor verdict between two validity ids, DERIVED from
        ///        `ancestorsOf` (no stored cache).
        ///
        /// @details
        /// Returns `true` with `out` set to `0` when `a == b`, `-1` when `a` is
        /// a strict ancestor of `b`, `+1` when `b` is a strict ancestor of `a`;
        /// returns `false` (leaving `out` untouched) when the pair is divergent
        /// — neither lies on the other's root-to-leaf path. Replaces the former
        /// `pairMap` hash probe with two `ancContains` scans.
        ///
        /// @param a   First validity id.
        /// @param b   Second validity id.
        /// @param out [out] The verdict value when the result is `true`.
        /// @return `true` when `a` and `b` are comparable (including equal).
        bool verdict(int16_t a, int16_t b, int16_t& out) const {
            if (a == b) { out = 0; return true; }
            if (ancContains(b, a)) { out = -1; return true; }
            if (ancContains(a, b)) { out = 1; return true; }
            return false;
        }

        /// @brief Whether `a` and `b` are comparable — one is an ancestor of
        ///        the other, or they are equal (`verdict` exists). Derived from
        ///        `ancestorsOf`.
        ///
        /// @param a First validity id.
        /// @param b Second validity id.
        /// @return `true` when the pair is comparable.
        bool comparable(int16_t a, int16_t b) const {
            if (a == b) return true;
            return ancContains(b, a) || ancContains(a, b);
        }

        /// @brief Number of ordered ancestor-pair verdicts the validity
        ///        hierarchy implies — the count the dropped `pairMap` cached.
        ///
        /// @details
        /// Each strict-ancestor relationship contributes two ordered entries
        /// (`(a,b) -> -1` and `(b,a) -> +1`), and id `id` has
        /// `|ancestorsOf[id]| - 1` strict ancestors, so the total is
        /// `2 * sum_id (|ancestorsOf[id]| - 1)`. Derived, not stored — the sole
        /// caller is the diagnostic hashburst dump, which keeps its byte-for-byte
        /// `pairMap (N entries; size only)` line across the pairMap removal
        /// (Rule 14).
        ///
        /// @return The ordered-pair verdict count.
        std::size_t pairCount() const {
            std::size_t total = 0;
            const int32_t rows = nodes->size();
            for (int32_t id = 0; id < rows; ++id) {
                const int32_t len = ancLen(static_cast<int16_t>(id));
                if (len > 1)
                    total += 2 * static_cast<std::size_t>(len - 1);
            }
            return total;
        }

        /// True iff `maybeAncestor` is a strict ancestor of `descendant` in
        /// the validity hierarchy — exactly the pairMap verdict == -1 when
        /// packed (maybeAncestor, descendant). The former eager
        /// `stringAncestorsOf` mirror is gone; two non-minting lookups plus
        /// one verdict probe answer the same question with the same
        /// outcomes (an unknown name or a root descendant is `false`, self
        /// is `false` — verdict 0 ≠ -1).
        inline bool isStrictAncestor(const std::string& maybeAncestor,
                                     const std::string& descendant) const {
            const int16_t anc = lookup(maybeAncestor);
            if (anc == 0) return false;
            const int16_t desc = lookup(descendant);
            if (desc == 0) return false;
            int16_t v;
            return verdict(anc, desc, v) && v == -1;
        }

        /// @brief Strict-ancestor test, id form — the ids-in-hand twin of
        ///        `isStrictAncestor(const std::string&, const std::string&)`.
        ///
        /// @details
        /// Byte-identical verdict to the string overload with both names
        /// pre-resolved: `true` iff @p maybeAncestor is a strict ancestor of
        /// @p descendant (`verdict == -1`). The slot-0 sentinel is treated as
        /// the string form treats a lookup miss — an unknown id is an ancestor
        /// of nothing and has no ancestor — so
        /// `isStrictAncestor(lookup(a), lookup(b))` equals
        /// `isStrictAncestor(a, b)` for every pair. Self is never a strict
        /// ancestor of itself (verdict `0 != -1`). Lets the equivalence-class
        /// apply tree's scope comparisons (`applyEquiClasses`
        /// stmtValidity/classValidity) run straight off the validity ids they
        /// already hold, skipping the two `nameMap.decode` copies.
        ///
        /// @param maybeAncestor Candidate ancestor's validity id.
        /// @param descendant    Candidate descendant's validity id.
        /// @return `true` iff @p maybeAncestor strictly precedes
        ///         @p descendant on the root-to-leaf scope path.
        /// @see isStrictAncestor(const std::string&, const std::string&) — the
        ///      string overload; deeperOf(int16_t, int16_t) — the id-form
        ///      comparability sibling.
        inline bool isStrictAncestor(int16_t maybeAncestor,
                                     int16_t descendant) const {
            if (maybeAncestor == 0 || descendant == 0) return false;
            int16_t v;
            return verdict(maybeAncestor, descendant, v) && v == -1;
        }

        /// @brief Strict-ancestor names of `v`, lexicographically sorted —
        ///        the drop-in replacement for iterating the former
        ///        `stringAncestorsOf` set (whose `std::set` order WAS lex).
        ///
        /// @param v The descendant's canonical validity name.
        /// @return Sorted ancestor names; empty when `v` is unknown or a
        ///         root.
        std::vector<std::string> strictAncestorNames(
            const std::string& v) const {
            std::vector<std::string> out;
            const int16_t id = lookup(v);
            if (id == 0) return out;
            const int32_t n = ancLen(id);
            out.reserve(static_cast<std::size_t>(n));
            for (int32_t i = 0; i + 1 < n; ++i) {
                out.push_back(decode(ancAt(id, i)));
            }
            std::sort(out.begin(), out.end());
            return out;
        }

        /// @brief Strict-ancestor names of @p v, lexicographically sorted —
        ///        the `StrSpan` twin of
        ///        `strictAncestorNames(const std::string&)`.
        ///
        /// @details
        /// Byte-identical result to the `std::string` overload: resolves @p v
        /// through the non-minting `lookup(const StrSpan&)`, then decodes each
        /// strict ancestor of the descendant's canonical validity into a fresh
        /// `std::vector<std::string>` in the same lexicographic order. Taking a
        /// span lets the equivalence-class apply tree
        /// (`applyEquivalenceClassToNegatedEquality`, `updateEquivalenceClasses`)
        /// enumerate ancestor scopes straight off the `StrSpan validityName`
        /// they already thread, with no throwaway `std::string` scope copy. The
        /// span is read only during the synchronous `lookup`, so a caller span
        /// over caller-stable bytes cannot dangle. An unknown or root @p v
        /// yields an empty vector, exactly as the string overload does.
        ///
        /// @param v The descendant's canonical validity name span.
        /// @return Sorted ancestor names; empty when @p v is unknown or a root.
        /// @see strictAncestorNames(const std::string&) — the canonical
        ///      overload; lookup(const StrSpan&) — the non-minting probe.
        std::vector<std::string> strictAncestorNames(
            const StrSpan& v) const {
            std::vector<std::string> out;
            const int16_t id = lookup(v);
            if (id == 0) return out;
            const int32_t n = ancLen(id);
            out.reserve(static_cast<std::size_t>(n));
            for (int32_t i = 0; i + 1 < n; ++i) {
                out.push_back(decode(ancAt(id, i)));
            }
            std::sort(out.begin(), out.end());
            return out;
        }

        /// @brief Strict-ancestor name SPANS of @p v, lexicographically
        ///        sorted into a caller stack array — the zero-copy,
        ///        zero-heap twin of `strictAncestorNames`.
        ///
        /// @details
        /// Replicates the string overload exactly, spans instead of owned
        /// strings. Byte-order contract:
        /// 1. The sort criterion is `compareSpans` — byte-lex, identical to
        ///    `std::string::operator<` — so the output permutation equals the
        ///    string overload's `std::sort` order.
        /// 2. The element SET is identical: `decodeView(ancAt(id, i))` spans
        ///    the same cold-table bytes `decode(ancAt(id, i))` copies.
        /// 3. Tie-freedom: the strict ancestors are pairwise-distinct NameMap
        ///    ids and the interner is injective (one id per byte string), so
        ///    all returned names are pairwise distinct — the order is strict
        ///    and the sorted permutation unique. (Each ancestor is in fact a
        ///    proper prefix of the next — `encodePush` canonical form is
        ///    parent + `"_boundary_"` + payload — so root-to-leaf is already
        ///    lex-ascending and the sort is provably the identity
        ///    permutation; the sort stays anyway: it IS the twin, costs
        ///    nothing at practical depths, and stays correct if a chain
        ///    shape ever changes.)
        /// 4. Lifetime: the returned spans alias this NameMap's cold byte
        ///    pool — valid until the next mint into THIS NameMap. Callers
        ///    must finish reading (or copy) before any such mint.
        ///
        /// The @p cap overflow assert is the Rule-19 depth tripwire: there is
        /// no structural bound on scope nesting (every `encodePush` deepens
        /// by one), but the practical depth is single-digit;
        /// `ExecutionParameters::MAX_SCOPE_DEPTH` (64) gives generous
        /// headroom while keeping the caller's stack array small, and an
        /// overflow stops at its origin instead of silently truncating a
        /// scope walk.
        ///
        /// @param v   The descendant's canonical validity name span.
        /// @param out Caller array receiving the sorted strict-ancestor
        ///            spans; capacity @p cap.
        /// @param cap Capacity of @p out; asserted against the ancestor
        ///            count BEFORE any write.
        /// @return The strict-ancestor count (0 for an unknown or root
        ///         @p v).
        /// @invariant Non-minting ([I-3](../../docs/agentic_swdd/30_invariants.md#i-3):
        ///            the spans alias cold bytes, valid until the next mint
        ///            into this NameMap); see
        ///            `I-140`.
        /// @see strictAncestorNames(const std::string&) — the canonical
        ///      overload (retained oracle); lookup(const StrSpan&) — the
        ///      non-minting probe; compareSpans.
        int32_t strictAncestorSpans(const StrSpan& v, StrSpan* out,
                                    int32_t cap) const {
            const int16_t id = lookup(v);
            if (id == 0) return 0;                 // unknown name -> none
            const int32_t n = ancLen(id);          // chain length incl. self
            const int32_t m = n - 1;               // strict ancestors
            if (m == 0) return 0;                  // root -> none
            assert(m <= cap
                && "strictAncestorSpans: validity nesting exceeds caller capacity (MAX_SCOPE_DEPTH)");
            for (int32_t i = 0; i < m; ++i) {
                out[i] = decodeView(ancAt(id, i));
            }
            std::sort(out, out + m, [](const StrSpan& a, const StrSpan& b) {
                return compareSpans(a, b) < 0;
            });
            return m;
        }

        /// Caller guarantees comparable(a, b).
        int16_t deeperOf(int16_t a, int16_t b) const {
            if (a == b) return a;
            int16_t v;
            bool ok = verdict(a, b, v);
            (void)ok;
            assert(ok && "deeperOf called on divergent pair");
            return (v >= 0) ? a : b;
        }

        /// String-level deeperOf — returns whichever of `a`, `b` is the
        /// deeper validity in the scope tree. Caller must have established
        /// that `a` and `b` are comparable (one is on the other's
        /// root-to-leaf path). Used by `applyEquivalenceClass` and
        /// `applyEquivalenceClassToRejectedMapIntegration` to route the
        /// rewritten expression to the deeper of (class scope, expr scope).
        inline std::string deeperOf(const std::string& a,
                                    const std::string& b) const {
            if (a == b) return a;
            if (isStrictAncestor(a, b)) return b;   // a shallower
            if (isStrictAncestor(b, a)) return a;   // b shallower
            assert(false && "deeperOf called on divergent string pair");
            return a;
        }

        /// @brief Span-level deeperOf — returns whichever of @p a, @p b spans
        ///        the deeper validity in the scope tree, with no heap copy.
        ///
        /// @details
        /// The `StrSpan` twin of the `std::string` overload: same contract
        /// (the caller has established that @p a and @p b are comparable —
        /// one is on the other's root-to-leaf path), same verdict, but it
        /// RETURNS one of its two input spans rather than a fresh
        /// `std::string`. That lets the equivalence-class apply tree's
        /// `depositValidity = deeperOf(validityName, tv.second)` sites mint
        /// the result span straight into their arena interner without the
        /// throwaway heap string. Resolves each name to its id once and defers
        /// to the id-form `isStrictAncestor`; the returned span is byte-equal
        /// to the string overload's result (it IS one of the inputs). Equal
        /// spans return @p a, matching the string form's `a == b -> a`.
        ///
        /// @param a One comparable validity name span.
        /// @param b The other comparable validity name span.
        /// @return The deeper of the two input spans (or @p a when equal).
        /// @see deeperOf(const std::string&, const std::string&) — the
        ///      canonical overload; isStrictAncestor(int16_t, int16_t).
        inline StrSpan deeperOf(const StrSpan& a, const StrSpan& b) const {
            if (equalSpans(a, b)) return a;
            const int16_t ida = lookup(a);
            const int16_t idb = lookup(b);
            if (isStrictAncestor(ida, idb)) return b;   // a shallower
            if (isStrictAncestor(idb, ida)) return a;   // b shallower
            assert(false && "deeperOf called on divergent span pair");
            return a;
        }
    };

    /// IntNormalizedKey — int16_t array encoding of a normalized key.
    ///
    /// Data layout (repeats for each expression, up to numberExpressions):
    ///
    ///   data[i+0] = expressionID    (NameMap ID of expression name)
    ///   data[i+1] = negation        (1 = negated, 0 = not negated)
    ///   then for each variable of the expression (arity times):
    ///     data[j+0] = varID
    ///     data[j+1] = changeableStatus  (1 = unchangeable, 0 = changeable)
    ///
    ///   If changeableStatus == 1 (unchangeable): varID is a NameMap ID of the
    ///     actual variable name — preserved literally, not normalized.
    ///   If changeableStatus == 0 (changeable): varID is a sequential normalization
    ///     ID (1, 2, 3, ...) assigned by first-appearance order within the key.
    ///
    /// Next expression follows immediately after the last variable pair.
    struct IntNormalizedKey {
        int16_t numberExpressions;
        int16_t length;
        const int16_t* data;  // non-owning pointer — references gen scratch arena byte-bump storage

        IntNormalizedKey() : numberExpressions(0), length(0), data(nullptr) {}

        IntNormalizedKey(int16_t numExpr, const int16_t* arr, int16_t len)
            : numberExpressions(numExpr), length(len), data(arr) {}

        bool operator==(const IntNormalizedKey& rhs) const {
            if (numberExpressions != rhs.numberExpressions) return false;
            if (length != rhs.length) return false;
            return std::memcmp(data, rhs.data, length * sizeof(int16_t)) == 0;
        }

        bool operator<(const IntNormalizedKey& rhs) const {
            if (numberExpressions != rhs.numberExpressions) return numberExpressions < rhs.numberExpressions;
            if (length != rhs.length) return length < rhs.length;
            for (int16_t i = 0; i < length; ++i) {
                if (data[i] != rhs.data[i]) return data[i] < rhs.data[i];
            }
            return false;
        }
    };

    /// @brief FNV-1a 64-bit hasher for `IntNormalizedKey`, suitable for
    /// `std::unordered_map` / `std::unordered_set`.
    ///
    /// @details
    /// FNV offset basis 14695981039346656037 / FNV prime 1099511628211, applied
    /// per-byte over `numberExpressions` first, then over each `int16_t` of `data`.
    /// Determinism is essential: the same key produced on two different LBs (or
    /// across two threads in the parallel hash burst) must hash identically so the
    /// dedupe in `addToHashMemory` and the lookup in `checkLocalEncodedMemoryStatic`
    /// agree.
    ///
    /// @see [`IntNormalizedKey`](#intnormalizedkey) — the type being hashed.
    struct IntNormalizedKeyHash {
        std::size_t operator()(const IntNormalizedKey& k) const {
            std::size_t h = 14695981039346656037ULL;
            h ^= static_cast<std::size_t>(k.numberExpressions);
            h *= 1099511628211ULL;
            for (int16_t i = 0; i < k.length; ++i) {
                h ^= static_cast<std::size_t>(static_cast<uint16_t>(k.data[i]));
                h *= 1099511628211ULL;
            }
            return h;
        }
    };

    // IntEncodedExpr lives in memory_infra/int_encoded_expr.hpp (pulled in
    // through memory_infra/lb_memory.hpp above) — it is the statified
    // containers' element type and belongs to the static-memory hierarchy.

    /// @brief Read-only indexed view over a statement array — the phase-2
    ///        request-generation pipeline's replacement for raw `data()` +
    ///        count parameter pairs.
    ///
    /// @details
    /// The statified containers cannot hand out a contiguous `data()`
    /// pointer (storage is paged), so every phase-2 consumer takes this
    /// view and indexes through `operator[]` — one shift + one mask + one
    /// page-table lookup per access.
    ///
    /// Element pointers (`&view[i]`) stay valid for the whole hashburst:
    /// phase 2 never mutates LB statement containers (I-66 discipline) and
    /// pages never move while the LB is resident — the stack-local
    /// pointer-array plumbing (`preEvaluateFromEncoded`, the merge loops)
    /// is untouched.
    ///
    /// Pass by value: two machine words.
    ///
    /// @invariant The viewed container must outlive the view, stay
    ///            unmodified, and stay resident while the view is in use.
    struct IntStmtView {
        const PagedVector<IntEncodedExpr>* vec;
        int32_t count;

        /// @brief Empty view (no statements).
        IntStmtView() : vec(nullptr), count(0) {}

        /// @brief View over a statified arena vector.
        ///
        /// @param source The arena vector to view; must outlive the view
        ///               and stay resident while the view is in use.
        explicit IntStmtView(const PagedVector<IntEncodedExpr>& source)
            : vec(&source), count(source.size()) {}

        /// @brief Number of statements in view.
        ///
        /// @return Element count.
        int32_t size() const { return count; }

        /// @brief Whether the view is empty.
        ///
        /// @return `true` when `size() == 0`.
        bool empty() const { return count == 0; }

        /// @brief Indexed element access.
        ///
        /// @param i Index in `[0, size())`.
        /// @return Const reference into the viewed storage.
        const IntEncodedExpr& operator[](int32_t i) const {
            assert(i >= 0 && i < count);
            return (*vec)[i];
        }
    };

    // LbMemory lives in memory_infra/lb_memory.hpp (included above) — the
    // per-LB statified aggregate belongs to the static-memory hierarchy
    // alongside its managers, containers, and the deload serializer.

    /// @brief Whether a structurally-matched (sub)key could still fire some
    ///        owner once its u_ (unchangeable) argument literals are checked
    ///        against the request — the per-site predicate of the
    ///        request-generation u_ literal prune (D-120).
    ///
    /// @details The four `normalizedEncoded*` fast-rejection maps are built with
    /// `ignoreU=false`, which normalizes every argument to a positional id and
    /// thereby erases the literal values of a rule's unchangeable args — only the
    /// repetition pattern survives. A growing request can therefore match a
    /// (sub)key structurally yet be doomed by a u_ literal it can never satisfy,
    /// a mismatch the firing gate (`checkLocalEncodedMemoryStatic`) would only
    /// catch later. This predicate brings that check forward using the per-owner
    /// `OwnerSet::uSignatures` recorded at insert by `recordUSignature`.
    ///
    /// Returns `true` (keep the request) when:
    /// - `entry.hasLooseOwner` — some owner has no u_ constraint and can always
    ///   potentially fire (the "0 u_, nothing to check" fast path), OR
    /// - `entry.uSignatures` is empty — no recorded u_ constraint, so there is no
    ///   basis to prune, OR
    /// - the request satisfies at least one signature: every `(slot, id)` pair of
    ///   that signature equals the request's flattened `argFullId` at `slot`.
    /// Returns `false` (prune) only when no owner's u_ literals can be satisfied.
    ///
    /// Soundness (I-79): request premises and stored
    /// (sub)keys are both name-sorted, so the linear arg slots align; a request
    /// that can fire some owner satisfies that owner's stored signature; the only
    /// outcome of a signature match is *keep*. Hence `false` is returned only for
    /// requests that could not have fired. The firing gate enforces the exact
    /// condition independently, so this is a runtime prune, never a soundness gate.
    ///
    /// The request's `argFullId` is read but never `encode`d, so this is safe on
    /// the read-only shared LB the LB-split executors run in parallel.
    ///
    /// @param entry The matched owner-set value (carries the cached signatures).
    /// @param exprs The request's premise pointers, in name-sorted order — the
    ///              same order the matched key was built from.
    /// @param count Length of `exprs`.
    /// @return `true` to keep the request, `false` to prune it.
    /// @see `recordUSignature` — the insert-side producer of the signatures.
    /// @see `ownerSetHasComparable` — the sibling scope-comparability prune.
    inline bool ownerSetUSatisfied(const OwnerSet& entry,
                                   const IntEncodedExpr* const* exprs,
                                   int16_t count) {
        if (entry.hasLooseOwner || entry.uSignatures.empty()) return true;

        int16_t reqArgs[ExecutionParameters::MAX_KEY_SLOTS];
        int16_t n = 0;
        for (int16_t i = 0; i < count; ++i) {
            const IntEncodedExpr& e = *exprs[i];
            for (int16_t j = 0; j < e.arity; ++j) {
                assert(n < ExecutionParameters::MAX_KEY_SLOTS);
                reqArgs[n++] = e.argFullId[j];
            }
        }

        for (const std::vector<std::pair<int16_t, int16_t>>& sig : entry.uSignatures) {
            bool ok = true;
            for (const std::pair<int16_t, int16_t>& sl : sig) {
                if (sl.first >= n || reqArgs[sl.first] != sl.second) { ok = false; break; }
            }
            if (ok) return true;
        }
        return false;
    }

    /// @brief u_-literal prune read from an `OwnerSet`'s blob bytes — the
    ///        no-decode twin of `ownerSetUSatisfied(const OwnerSet&)`.
    ///
    /// @details
    /// Identical verdict to the value overload
    /// ([D-120](docs/agentic_swdd/40_decisions.md#d-120),
    /// [I-79](docs/agentic_swdd/30_invariants.md#i-79)): keep when the owner set
    /// is loose (the `hasLooseOwner` byte, read first) or has no recorded
    /// signature; otherwise keep iff the request's flattened `argFullId` satisfies
    /// some signature. Walks the signatures straight off the blob — no allocation,
    /// no decode — advancing the cursor past every pair of each signature so the
    /// next signature is correctly positioned even after an early `ok = false`.
    ///
    /// @param entry The matched (sub)key's owner-set blob view.
    /// @param exprs The request's premise pointers, in name-sorted order.
    /// @param count Length of `exprs`.
    /// @return `true` to keep the request, `false` to prune it.
    /// @see `ownerSetUSatisfied(const OwnerSet&, ...)`, `OwnerSetBlob`.
    inline bool ownerSetUSatisfied(const OwnerSetBlob& entry,
                                   const IntEncodedExpr* const* exprs,
                                   int16_t count) {
        if (entry.hasLooseOwner()) return true;
        const char* cur = entry.p + entry.uSigOffset();
        const int32_t sc = OwnerSetBlob::rdI32(cur); cur += sizeof(int32_t);
        if (sc == 0) return true;

        int16_t reqArgs[ExecutionParameters::MAX_KEY_SLOTS];
        int16_t n = 0;
        for (int16_t i = 0; i < count; ++i) {
            const IntEncodedExpr& e = *exprs[i];
            for (int16_t j = 0; j < e.arity; ++j) {
                assert(n < ExecutionParameters::MAX_KEY_SLOTS);
                reqArgs[n++] = e.argFullId[j];
            }
        }

        for (int32_t s = 0; s < sc; ++s) {
            const int32_t pairCount = OwnerSetBlob::rdI32(cur);
            cur += sizeof(int32_t);
            bool ok = true;
            for (int32_t k = 0; k < pairCount; ++k) {
                const int16_t slot = OwnerSetBlob::rdI16(cur);
                cur += sizeof(int16_t);
                const int16_t id = OwnerSetBlob::rdI16(cur);
                cur += sizeof(int16_t);
                if (ok && (slot >= n || reqArgs[slot] != id)) ok = false;
            }
            if (ok) return true;
        }
        return false;
    }

    // encodeExpression is defined after EncodedExpression (forward reference).

    /// @brief Per-type bump allocator. Allocates a fixed pool at construction and
    /// hands out aligned slots; resets per LB without freeing the underlying
    /// buffer.
    ///
    /// @details
    /// Used to back the static-request pipeline's per-thread storage of
    /// `IntEncodedExpr`, `StaticRequest`, `Stump`, and `int16_t` index
    /// arrays. The contract is:
    ///
    /// - Construction with `cap` slots allocates `cap * sizeof(T)` bytes via
    ///   `::operator new`. `cap == 0` means no buffer (subsequent `alloc` is
    ///   undefined behaviour and trips the assert below).
    /// - `alloc(n)` advances the bump pointer by `n` slots and returns the head
    ///   pointer. Asserts that capacity is not exceeded — a firing assert here
    ///   means the per-thread arena was undersized for the LB.
    /// - `reset()` rewinds `used` to 0; the buffer stays alive. Use between LBs.
    /// - Destructor releases the buffer once.
    ///
    /// Move-only: copy construction and assignment are deleted because two
    /// arenas pointing at the same buffer would double-free on destruction.
    /// Move construction transfers ownership and zeroes the source.
    ///
    /// @tparam T element type. The arena makes no constructor calls, so `T` must
    ///         be trivially default-constructible or fully overwritten by callers
    ///         after `alloc`.
    /// @see [`ThreadArenas`](#threadarenas) — assembles four `TypedArena<T>` for
    ///      one thread of the static-request pipeline.
    /// @see [I-19](../../docs/agentic_swdd/30_invariants.md#i-19) — assert is first-class; the
    ///      capacity assert here is intentional and must not be weakened.
    template<typename T>
    struct TypedArena {
        T* buf;
        int32_t used;
        int32_t capacity;

        explicit TypedArena(int32_t cap = 0) : used(0), capacity(cap), buf(nullptr) {
            if (cap > 0) buf = static_cast<T*>(::operator new(static_cast<std::size_t>(cap) * sizeof(T)));
        }
        TypedArena(const TypedArena&) = delete;
        TypedArena& operator=(const TypedArena&) = delete;
        TypedArena(TypedArena&& o) noexcept : buf(o.buf), used(o.used), capacity(o.capacity) {
            o.buf = nullptr; o.used = 0; o.capacity = 0;
        }
        TypedArena& operator=(TypedArena&& o) noexcept {
            if (this != &o) {
                if (buf) ::operator delete(buf);
                buf = o.buf; used = o.used; capacity = o.capacity;
                o.buf = nullptr; o.used = 0; o.capacity = 0;
            }
            return *this;
        }
        void reset() { used = 0; }
        T* alloc(int32_t n = 1) {
            assert(used + n <= capacity);
            T* p = buf + used;
            used += n;
            return p;
        }
        ~TypedArena() { if (buf) ::operator delete(buf); }
    };

    /// @brief One obligatory stump — the already-known statements that every
    ///        request generated from a base candidate must contain.
    ///
    /// @details
    /// The request generator grows base candidates to `maxKeyLength - stumpLength`
    /// elements and then attaches a stump of `stumpLength` statements. A stump of
    /// one element carries an index in `idx0` only; a stump of two carries an index
    /// into each of the generator's two source views. A stump of zero elements has
    /// no instances at all — that is the counter-example filter, where no statement
    /// is obligatory and the base candidate is already the whole request.
    ///
    /// Plain index arithmetic; no ordering or hashing, because consumers walk the
    /// array sequentially.
    ///
    /// @see `memory.cpp::makeMandatoryEncodedStatementLists1Static` — emits one-element stumps.
    /// @see `memory.cpp::makeMandatoryEncodedStatementLists2Static` — emits two-element stumps.
    /// @see `memory.cpp::generateEncodedRequestsStatic` — the consumer.
    struct Stump {
        int16_t idx0;
        int16_t idx1;
    };

    /// @brief Fully-static request — no string data, no heap allocations.
    ///
    /// @details
    /// One element of the per-LB request stream consumed by the static pipeline.
    /// Each request bundles up to `ExecutionParameters::MAX_EXPRESSIONS` pointers
    /// into a `TypedArena<IntEncodedExpr>` (so the pointers stay stable through
    /// the arena's lifetime), the count of those pointers, the maximum iteration
    /// counter across the bundle, and the precomputed `IntNormalizedKey` used to
    /// look the request up in the hash memory.
    ///
    /// Pointer stability matters: requests are emplaced into a contiguous
    /// `StaticRequest*` output buffer; if the underlying arena were a
    /// `std::vector<IntEncodedExpr>`, every push could reallocate and invalidate
    /// the pointers. The arena is append-only with stable interior pointers, so
    /// pointer-based identity holds through emission.
    ///
    /// @see [`StaticRequestEmitter`](#staticrequestemitter) — produces these.
    struct StaticRequest {
        const IntEncodedExpr* intExprs[ExecutionParameters::MAX_EXPRESSIONS]; // ptrs into arena
        int16_t count;
        int16_t maxIteration;
        IntNormalizedKey normalizedKey;
    };

    /// @brief Pack `(originalId, validityId)` into a single `int32_t` for fast
    /// `std::unordered_set` / `std::map` lookup.
    ///
    /// @details
    /// Concatenates the two `int16_t` halves into a 32-bit value with
    /// `originalId` in the high 16 bits and `validityId` in the low 16. Cheaper
    /// than hashing a `std::pair<int16_t, int16_t>` and equivalent to
    /// `std::tuple<int16_t, int16_t>` ordering for keys with the same
    /// `originalId`. Used by `Memory::intKnownStatements` for the O(1)
    /// "is this statement already known here" check.
    ///
    /// @param originalId NameMap id of the original (un-normalized) expression
    ///                   string.
    /// @param validityId NameMap id of the scope's canonical validity name.
    /// @return Combined key value. Uniqueness is guaranteed because both inputs
    ///         are distinct ids minted by the same `NameMap`.
    inline int32_t packStatementKey(int16_t originalId, int16_t validityId) {
        return (static_cast<int32_t>(static_cast<uint16_t>(originalId)) << 16)
             | static_cast<int32_t>(static_cast<uint16_t>(validityId));
    }

    /// @brief Dedicated id space for admission/rejected TEMPLATE strings —
    /// the marker-form, u_-form, and bare-arg names that key the
    /// admission/rejected subsystem.
    ///
    /// @details
    /// Template strings (e.g. `(in2[x,marker,3])`, `(in2[u_x,int_0_1,u_z])`)
    /// are a population disjoint from the statement originals and validity
    /// names the per-LB `NameMap` interns. They get their OWN id space
    /// deliberately: interning them in the `NameMap` would mint mid-run,
    /// shifting every later statement id — which changes the hashburst
    /// dump's `nameMap` table and raw-id sections byte-for-byte and
    /// pressures the 32000-id ceiling
    /// ([D-132](../../docs/agentic_swdd/40_decisions.md#d-132)).
    ///
    /// The packed admission/rejected key is
    /// `packStatementKey(templateId, validityId)` — the HIGH half from THIS
    /// interner, the LOW half from the `NameMap`. The two spaces are never
    /// mixed inside one container, and a template id never meets a
    /// statement-original id in the same key position.
    ///
    /// `encode` is the find-or-mint write half; minting is confined to
    /// single-threaded sites (the post-fixpoint drains, disintegration,
    /// `standardProcessing` — [I-68](../../docs/agentic_swdd/30_invariants.md#i-68)/[I-69](../../docs/agentic_swdd/30_invariants.md#i-69)
    /// keep the parallel burst staging on strings). `lookup` is the
    /// read-only probe half: a template never interned was never registered
    /// in any admission/rejected container, so `0` is a definitive miss —
    /// a defined result, not a fallback.
    ///
    /// @invariant Id 0 is the reserved invalid entry, mirroring the
    ///            `NameMap` convention; real ids start at 1.
    /// @invariant Ids never re-bind while the interner lives; it resets
    ///            only in lockstep with the `destroyGrid` `nameMap` reset
    ///            (the validity halves of packed keys re-bind there).
    /// @see packStatementKey — the shared packing helper.
    ///
    /// Storage note (strings campaign): the interner is an int16 FAÇADE
    /// over the LB's cold string table (`LbMemory::templateStrings`,
    /// D-153) — the strings live on the LB's static
    /// pages and deload with it; the façade carries no state beyond the
    /// binding, so ids survive the deload round trip untouched. `Memory`'s
    /// constructor binds it; standalone uses (tests) bind explicitly.
    struct TemplateInterner {
        ColdStringTable* table = nullptr;

        /// @brief Bind the façade to its cold table (idempotent for the
        ///        same table; rebinding asserts).
        ///
        /// @param t The backing table (`LbMemory::templateStrings`).
        void bind(ColdStringTable* t) {
            assert(t != nullptr);
            assert((table == nullptr || table == t)
                && "TemplateInterner rebind to a different table");
            table = t;
        }

        /// @brief Find-or-mint the id of a template string (write half).
        ///
        /// @details Asserts the capacity ceiling (the same
        /// `ExecutionParameters::MAX_NAME_IDS` bound the `NameMap` uses) —
        /// exhaustion is a bug to see at its origin, never a degrade.
        ///
        /// @param s The template string to intern.
        /// @return The stable id of @p s; mints on first sight.
        int16_t encode(const std::string& s) {
            assert(table != nullptr && "TemplateInterner used unbound");
            const int32_t id = table->intern(StrSpan(s));
            assert(id < ExecutionParameters::MAX_NAME_IDS);
            return static_cast<int16_t>(id);
        }

        /// @brief Span overload of the find-or-mint write half — interns
        ///        template bytes without materializing a `std::string`.
        ///
        /// @details
        /// Byte-identical id to `encode(const std::string&)` — the backing
        /// cold table interns by bytes, so the id is a pure function of the
        /// content regardless of the source type. This unlocks the
        /// `mintTemplateKey` span overload and any admission/rejection key
        /// writer that already holds a `StrSpan` over a stable buffer (a
        /// `decodeView` of a sibling interner, an arena `ScratchString`),
        /// removing the throwaway heap string the write formerly required.
        /// Asserts the same `MAX_NAME_IDS` ceiling as the `std::string`
        /// overload — an exhausted id space is a bug to see at its origin,
        /// never a degrade (Rule 19).
        ///
        /// @param s The template bytes to intern; a slice of a stable buffer.
        /// @return The stable id of @p s; mints on first sight.
        /// @see encode(const std::string&) — the canonical overload;
        ///      lookup(const StrSpan&) — the non-minting probe.
        int16_t encode(const StrSpan& s) {
            assert(table != nullptr && "TemplateInterner used unbound");
            const int32_t id = table->intern(s);
            assert(id < ExecutionParameters::MAX_NAME_IDS);
            return static_cast<int16_t>(id);
        }

        /// @brief Non-minting probe (read half).
        ///
        /// @details Returns `0` when @p s was never interned — a definitive
        /// miss for the admission/rejected containers, which only ever hold
        /// keys whose template half was minted at a registration write.
        ///
        /// @param s The template string to look up.
        /// @return The interned id, or `0` if absent.
        int16_t lookup(const std::string& s) const {
            assert(table != nullptr && "TemplateInterner used unbound");
            return static_cast<int16_t>(table->lookup(StrSpan(s)));
        }

        /// @brief Span overload of the non-minting probe — probes template
        ///        bytes without materializing a `std::string`.
        ///
        /// @details
        /// Byte-identical verdict to `lookup(const std::string&)`: the
        /// interned id, or `0` when @p s was never interned (a definitive
        /// container miss, since a registered key's template half is always
        /// minted at its write site). The symmetric partner of the span-form
        /// `encode`, so the `lookupTemplateKey` span overload and the
        /// admission-map overlap probes can feed a `decodeView` span straight
        /// in.
        ///
        /// @param s The template bytes to look up; a slice of a stable buffer.
        /// @return The interned id, or `0` if absent.
        /// @see lookup(const std::string&) — the canonical overload;
        ///      encode(const StrSpan&) — the minting counterpart.
        int16_t lookup(const StrSpan& s) const {
            assert(table != nullptr && "TemplateInterner used unbound");
            return static_cast<int16_t>(table->lookup(s));
        }

        /// @brief Recover the template string of an interned id.
        ///
        /// @details Returns an OWNED copy (the cold bytes live on pages;
        /// there is no heap string to reference). The former
        /// copy-before-mint caveat is gone — the copy is the contract.
        /// Hot-path readers that only compare or slice use `decodeView`.
        ///
        /// @param id A real id minted by this interner (`> 0`).
        /// @return The template string (owned copy).
        std::string decode(int16_t id) const {
            assert(table != nullptr && "TemplateInterner used unbound");
            return table->decodeString(id);
        }

        /// @brief Zero-copy decode — a span over the cold bytes, stable
        ///        while the LB is resident (the residency discipline of
        ///        element references).
        ///
        /// @param id A real id minted by this interner (`> 0`).
        /// @return Span over the template's bytes.
        StrSpan decodeView(int16_t id) const {
            assert(table != nullptr && "TemplateInterner used unbound");
            return table->view(id);
        }

        /// @brief Number of interned templates (ids run 1..count).
        ///
        /// @return The backing table's string count.
        int32_t internedCount() const {
            assert(table != nullptr && "TemplateInterner used unbound");
            return table->count();
        }

        /// @brief Wholesale reset in lockstep with the `destroyGrid`
        ///        `nameMap` reset — every id re-binds afterwards.
        void resetToFresh() {
            assert(table != nullptr && "TemplateInterner used unbound");
            table->resetToFresh();
        }
    };

    /// @brief Dedicated int32 id space for admission/rejected VALUE-field
    /// strings (D-1322).
    ///
    /// @details
    /// The value structs (`AdmissionMapValue`, `RejectedMapValue`,
    /// `RejectedMapIntegrationValue`, `Instruction`/`LogicalEntity`) store
    /// their string fields as ids in THIS space. Separate from
    /// `TemplateInterner` deliberately: value strings never participate in
    /// packed `(id16, id16)` pair keys — they sit in id vectors — so they
    /// take int32 ids and the int16 capacity ceiling does not apply at
    /// Gauss-scale value populations. Separate from the `NameMap` for the
    /// same reason as the template space: minting there would shift
    /// statement ids and break the dump's nameMap-table / raw-id sections.
    ///
    /// `encode` mints (single-threaded sites only — the admission value
    /// writers all run in disintegration / post-fixpoint phases); `lookup`
    /// is the non-minting read half; `decode` recovers the string (copy
    /// before any nested mint — I-3).
    ///
    /// @invariant Slot 0 is the reserved invalid entry; real ids start at 1.
    /// @invariant Ids never re-bind while the interner lives; it resets only
    ///            in lockstep with the `destroyGrid` `nameMap` reset.
    /// @see TemplateInterner — the int16 KEY-template sibling space.
    /// Storage note (strings campaign): like `TemplateInterner`, each
    /// instance is a stateless int32 FAÇADE over its own cold string table
    /// inside `LbMemory` (D-153) — strings live on
    /// the LB's static pages and deload with it; `Memory`'s constructor
    /// binds the four instances to their tables; ids survive the deload
    /// round trip untouched.
    struct ValueInterner {
        ColdStringTable* table = nullptr;

        /// @brief Bind the façade to its cold table (idempotent for the
        ///        same table; rebinding asserts).
        ///
        /// @param t The backing table inside the owning `LbMemory`.
        void bind(ColdStringTable* t) {
            assert(t != nullptr);
            assert((table == nullptr || table == t)
                && "ValueInterner rebind to a different table");
            table = t;
        }

        /// @brief Find-or-mint the id of a value string (write half).
        ///
        /// @param s The value string to intern.
        /// @return The stable id of @p s; mints on first sight.
        int32_t encode(const std::string& s) {
            assert(table != nullptr && "ValueInterner used unbound");
            return table->intern(StrSpan(s));
        }

        /// @brief Span overload of the find-or-mint write half — interns value
        ///        bytes without materializing a `std::string`. Byte-identical id
        ///        to `encode(std::string)` (the table interns by bytes); lets the
        ///        absorb-door origin mints feed a `StrSpan` over the stable input.
        ///
        /// @param s The value bytes to intern.
        /// @return The stable id of @p s; mints on first sight.
        int32_t encode(const StrSpan& s) {
            assert(table != nullptr && "ValueInterner used unbound");
            return table->intern(s);
        }

        /// @brief Non-minting probe (read half).
        ///
        /// @param s The value string to look up.
        /// @return The interned id, or `0` if absent — a definitive miss
        ///         for any id-form value field.
        int32_t lookup(const std::string& s) const {
            assert(table != nullptr && "ValueInterner used unbound");
            return table->lookup(StrSpan(s));
        }

        /// @brief Span overload of the non-minting probe — probes value bytes
        ///        without materializing a `std::string`.
        ///
        /// @details
        /// Byte-identical verdict to `lookup(const std::string&)`: the id, or
        /// `0` for any never-interned value field. The symmetric partner of
        /// the existing span-form `encode`, so an integration probe holding a
        /// `StrSpan` over a stable buffer (a `uniqueArgs` slice) can test
        /// membership before deciding to mint — no heap string built.
        ///
        /// @param s The value bytes to look up; a slice of a stable buffer.
        /// @return The interned id, or `0` if absent.
        /// @see lookup(const std::string&) — the canonical overload;
        ///      encode(const StrSpan&) — the span-form write half.
        int32_t lookup(const StrSpan& s) const {
            assert(table != nullptr && "ValueInterner used unbound");
            return table->lookup(s);
        }

        /// @brief Recover the value string of an interned id.
        ///
        /// @details Returns an OWNED copy (the cold bytes live on pages;
        /// there is no heap string to reference). The former
        /// copy-before-mint caveat is gone — the copy is the contract.
        /// Ordering and other read-only hot paths use `decodeView`.
        ///
        /// @param id A real id minted by this interner (`> 0`).
        /// @return The value string (owned copy).
        std::string decode(int32_t id) const {
            assert(table != nullptr && "ValueInterner used unbound");
            return table->decodeString(id);
        }

        /// @brief Zero-copy decode — a span over the cold bytes, stable
        ///        while the LB is resident (the residency discipline of
        ///        element references).
        ///
        /// @param id A real id minted by this interner (`> 0`).
        /// @return Span over the value's bytes.
        StrSpan decodeView(int32_t id) const {
            assert(table != nullptr && "ValueInterner used unbound");
            return table->view(id);
        }

        /// @brief Number of interned values (ids run 1..count).
        ///
        /// @return The backing table's string count.
        int32_t internedCount() const {
            assert(table != nullptr && "ValueInterner used unbound");
            return table->count();
        }

        /// @brief Wholesale reset in lockstep with the `destroyGrid`
        ///        `nameMap` reset — every id re-binds afterwards. The
        ///        lb-state space deliberately never calls this
        ///        (`expandedImplications` survives `destroyGrid`).
        void resetToFresh() {
            assert(table != nullptr && "ValueInterner used unbound");
            table->resetToFresh();
        }
    };

    /// @brief Decoded-string compare of two value ids — the ordering
    /// primitive for the id-form value structs.
    ///
    /// @details Id equality short-circuits (the interner is bijective, so
    /// equal ids ARE the equal-string case); otherwise compares the decoded
    /// strings. Never compares raw id values
    /// ([I-84](../../docs/agentic_swdd/30_invariants.md#i-84): id order is
    /// not lex order).
    ///
    /// @param a  First value id.
    /// @param b  Second value id.
    /// @param vi The owning LB's value interner (read-only).
    /// @return `true` iff `decode(a) < decode(b)`.
    inline bool valueIdLess(int32_t a, int32_t b, const ValueInterner& vi) {
        if (a == b) return false;
        // Zero-copy span compare — identical ordering to the former
        // decoded-reference compare, with no allocation per comparison
        // (this primitive sits under every decoded-order container).
        return compareSpans(vi.decodeView(a), vi.decodeView(b)) < 0;
    }

    /// @brief Decoded lexicographic compare of two value-id vectors —
    /// reproduces `std::vector<std::string>::operator<` on the decoded
    /// contents exactly.
    ///
    /// @details Element-wise decoded compare, then size as the tiebreak —
    /// the same lexicographic rule the string vectors used. Also serves
    /// for the former `std::set<std::string>` fields when their id storage
    /// is kept decoded-lex sorted (a sorted set compares exactly like its
    /// sorted element sequence).
    ///
    /// @param a  First id vector.
    /// @param b  Second id vector.
    /// @param vi The owning LB's value interner (read-only).
    /// @return `true` iff decoded @p a precedes decoded @p b.
    inline bool valueIdVectorLess(const std::vector<int32_t>& a,
        const std::vector<int32_t>& b, const ValueInterner& vi)
    {
        const std::size_t n = a.size() < b.size() ? a.size() : b.size();
        for (std::size_t i = 0; i < n; ++i) {
            if (a[i] == b[i]) continue;
            return compareSpans(vi.decodeView(a[i]), vi.decodeView(b[i])) < 0;
        }
        return a.size() < b.size();
    }

    /// @brief Encode a string set into value ids, decoded-lex sorted.
    ///
    /// @details `std::set<std::string>` iterates in lex order, so pushing
    /// in iteration order yields an id vector sorted by decoded name with
    /// no sort call ([I-84](../../docs/agentic_swdd/30_invariants.md#i-84)).
    ///
    /// @param strings The set.
    /// @param vi      The owning LB's value interner (mint side).
    /// @return Decoded-lex sorted id vector.
    inline std::vector<int32_t> encodeValueSetSorted(const std::set<std::string>& strings,
        ValueInterner& vi)
    {
        std::vector<int32_t> out;
        out.reserve(strings.size());
        for (const std::string& s : strings) out.push_back(vi.encode(s));
        return out;
    }

    /// @brief Span twin of `encodeValueSetSorted` — encode value member
    ///        spans into value ids in decoded-lex order, with no
    ///        `std::set<std::string>` on the heap.
    ///
    /// @details
    /// `encodeValueSetSorted` leans on `std::set<std::string>` to present its
    /// members deduplicated AND lexicographically sorted, then mints each in
    /// that iteration order; because a `ValueInterner` id is its mint ordinal,
    /// the resulting id VALUES are a pure function of the decoded-lex member
    /// order ([I-84](../../docs/agentic_swdd/30_invariants.md#i-84)). This twin
    /// reproduces the exact same id run WITHOUT the heap set: the member
    /// spans are deduplicated by interning them into a scratch
    /// `ColdHashSet<BytesKeyStore>` on @p scratch, a byte-bump index over the
    /// distinct ids is sorted by `compareSpans` (byte-identical to
    /// `std::string::compare`, hence to `std::set<std::string>` iteration
    /// order), and the distinct members are minted into @p vi in that sorted
    /// order INTO the caller's @p out run. The out[0..count) run — same ids,
    /// same positions — is therefore byte-identical to `encodeValueSetSorted`
    /// over the equivalent `std::set<std::string>` (09b row 87: caller out-param
    /// run + returned count, so the function keeps 0% heap).
    ///
    /// The transient dedup set and sort index ride @p scratch and are fully
    /// reclaimed before return (`ColdHashSet::release` for the page-tier set,
    /// `popTo` for the byte-bump index), so the door is per-call footprint-
    /// neutral like the heap RAII form it replaces. The scratch set's
    /// `view(id)` spans alias @p scratch's byte pool; `vi.encode` mints into a
    /// DIFFERENT interner (its own cold table), so no mint into the scratch
    /// arena occurs across the read-mint loop and the spans stay valid (I-3).
    /// Interned members are distinct, so `compareSpans` never ties and the
    /// sort order is fully determined.
    ///
    /// @param members     The member spans (duplicates allowed; each a slice
    ///                    of a buffer alive for the call).
    /// @param memberCount The number of spans in @p members (`>= 0`).
    /// @param vi          The owning LB's value interner (mint side;
    ///                    single-threaded write sites only, I-83).
    /// @param scratch     A per-slot `genScratchArenas` arena for the
    ///                    transient dedup set + sort index.
    /// @param out         Caller-owned run receiving the decoded-lex value ids;
    ///                    `out[0..count)` is filled, positions preserved.
    /// @param outCap      Capacity of @p out; a `distinct > outCap` overrun is a
    ///                    loud Rule-19 assert (widen `MAX_ADMISSION_REM_ARGS`).
    /// @return The distinct member count written into `out[0..count)` —
    ///         byte-identical to `encodeValueSetSorted` over the same distinct
    ///         members.
    /// @see encodeValueSetSorted — the `std::set<std::string>` oracle this
    ///      twins; ValueInterner::encode(const StrSpan&) — the span mint half.
    inline int32_t encodeValueSpanSetSorted(
        const StrSpan* members, int memberCount,
        ValueInterner& vi, ScratchArena& scratch,
        int32_t* out, int32_t outCap)
    {
        assert(memberCount >= 0);
        assert((members != nullptr) || (memberCount == 0));

        // Dedup by interning into a scratch byte set (the std::set<std::string>
        // uniqueness half); ids run 1..distinct in first-seen (mint) order.
        DirtyState seenDirty = DirtyState::Clean;
        ColdHashSet<BytesKeyStore> seen(&scratch, &seenDirty);
        for (int i = 0; i < memberCount; ++i) seen.mint(members[i]);

        const int32_t distinct = seen.count();
        assert(distinct <= outCap
            && "encodeValueSpanSetSorted: distinct member count exceeds caller "
               "capacity (widen MAX_ADMISSION_REM_ARGS)");
        if (distinct == 0) { seen.release(); return 0; }

        // Byte-bump index over the distinct ids, sorted by decoded bytes so the
        // mint order into `vi` matches the std::set<std::string> iteration order
        // (compareSpans == std::string::compare). Distinct members never tie.
        // Byte-bump `alloc` coexists with the page-tier `seen` (I-124).
        const ArenaOffset mark = scratch.cursor();
        int32_t* order = reinterpret_cast<int32_t*>(scratch.resolve(scratch.alloc(
            distinct * static_cast<int32_t>(sizeof(int32_t)),
            static_cast<int32_t>(alignof(int32_t)))));
        for (int32_t i = 0; i < distinct; ++i) order[i] = i + 1;
        std::sort(order, order + distinct, [&seen](int32_t a, int32_t b) {
            return compareSpans(seen.view(a), seen.view(b)) < 0;
        });

        // Mint into `vi` in decoded-lex order — the caller out-param run.
        for (int32_t i = 0; i < distinct; ++i)
            out[i] = vi.encode(seen.view(order[i]));

        scratch.popTo(mark);   // reclaim the byte-bump index
        seen.release();        // reclaim the page-tier dedup set
        return distinct;
    }

    /// @brief Decode a value-id vector to owned strings, positions
    ///        preserved.
    ///
    /// @details Owned copies — callers hold the result across further
    /// mints (I-3).
    ///
    /// @param ids The id vector.
    /// @param vi  The owning LB's value interner (read-only).
    /// @return Decoded strings, same positions.
    inline std::vector<std::string> decodeValueVector(const std::vector<int32_t>& ids,
        const ValueInterner& vi)
    {
        std::vector<std::string> out;
        out.reserve(ids.size());
        for (const int32_t id : ids) out.push_back(vi.decodeView(id).toStdString());
        return out;
    }

    /// @brief Decode a value-id vector into an owned string set.
    ///
    /// @param ids The id vector (typically decoded-lex sorted storage).
    /// @param vi  The owning LB's value interner (read-only).
    /// @return The decoded set.
    inline std::set<std::string> decodeValueVectorToSet(const std::vector<int32_t>& ids,
        const ValueInterner& vi)
    {
        std::set<std::string> out;
        for (const int32_t id : ids) out.insert(vi.decodeView(id).toStdString());
        return out;
    }

    // ================= Origin-map id infrastructure =================
    // (D-131)

    /// @brief The closed origin-tag vocabulary of the history maps
    /// (`Memory::exprOriginMap`, `EquivalenceClass::equalityOriginMap`).
    ///
    /// @details
    /// Every history line names WHY an expression was deposited; the tag
    /// set is closed — every emission site passes one of these literals
    /// (or forwards one received via mail, which some other LB emitted
    /// from the same set). `originTagFromString` asserts on any string
    /// outside the vocabulary: an unknown tag is a new emission site that
    /// must extend this enum AND the name table, never a value to pass
    /// through silently.
    ///
    /// The underlying value is storage only. No observable ordering may
    /// ever sort by it ([I-84](../../docs/agentic_swdd/30_invariants.md#i-84)
    /// discipline applies to tags exactly as to interned ids); the only
    /// semantic comparisons are equality and the equality1/equality2
    /// convenience-tag test of the D-49 cap-full policy.
    ///
    /// Opaque-declared in `memory_infra/int_encoded_expr.hpp` (so `LbMemory` can
    /// name the `TypedColdBlobMap<int64_t, IdOrigin>` member); this is its full
    /// definition.
    enum class OriginTag : uint8_t {
        anchorHandling,                      // "anchor handling"
        broadcast,                           // "broadcast"
        ceBuildingBlock,                     // "CE_building_block"
        compilation,                         // "compilation"
        contradiction,                       // "contradiction"
        disintegration,                      // "disintegration"
        equality1,                           // "equality1"
        equality2,                           // "equality2"
        expansion,                           // "expansion"
        expansionForIntegration,             // "expansion for integration"
        externallyProvidedTheorem,           // "externally provided theorem"
        goal,                                // "goal"
        implication,                         // "implication"
        multipliedFrom,                      // "multiplied from"
        orBranchAssumption,                  // "or branch assumption"
        orBranchProven,                      // "or branch proven"
        orConvergence,                       // "or convergence"
        orDisintegration,                    // "or disintegration"
        premise,                             // "premise"
        premiseElement,                      // "premise element"
        recursion,                           // "recursion"
        reformulationForIntegrationBracket,  // "reformulation for integration >[]"
        reformulationForIntegrationBound,    // "reformulation for integration >[bound]"
        reformulationForIntegrationAnd,      // "reformulation for integration and"
        symmetryOfEquality,                  // "symmetry of equality"
        symmetryOfInequality,                // "symmetry of inequality"
        taskFormulation,                     // "task formulation"
        theorem,                             // "theorem"
        vacuousTruth,                        // "vacuous truth"
        validityName,                        // "validity name"
        variableCopy,                        // "variable copy"
        COUNT                                // sentinel — table size only
    };

    /// @brief Recover the exact historical tag string of an `OriginTag`.
    ///
    /// @details The table is index-aligned with the enum; the strings are
    /// byte-exact — the dump, the chapter export, and the mail copies all
    /// reproduce the historical output through this table.
    ///
    /// @param t A real tag (not the `COUNT` sentinel).
    /// @return The tag string literal.
    inline const char* originTagName(OriginTag t) {
        static const char* const NAMES[] = {
            "anchor handling",
            "broadcast",
            "CE_building_block",
            "compilation",
            "contradiction",
            "disintegration",
            "equality1",
            "equality2",
            "expansion",
            "expansion for integration",
            "externally provided theorem",
            "goal",
            "implication",
            "multiplied from",
            "or branch assumption",
            "or branch proven",
            "or convergence",
            "or disintegration",
            "premise",
            "premise element",
            "recursion",
            "reformulation for integration >[]",
            "reformulation for integration >[bound]",
            "reformulation for integration and",
            "symmetry of equality",
            "symmetry of inequality",
            "task formulation",
            "theorem",
            "vacuous truth",
            "validity name",
            "variable copy",
        };
        static_assert(sizeof(NAMES) / sizeof(NAMES[0])
            == static_cast<std::size_t>(OriginTag::COUNT),
            "originTagName table must stay index-aligned with OriginTag");
        const std::size_t i = static_cast<std::size_t>(t);
        assert(i < static_cast<std::size_t>(OriginTag::COUNT));
        return NAMES[i];
    }

    /// @brief Encode a tag string into its `OriginTag` (closed-vocabulary
    ///        contract).
    ///
    /// @details Asserts on a string outside the vocabulary — a firing
    /// assert means a new emission site appeared and the enum must be
    /// extended; it is a report, never a fallback (Rule 19).
    ///
    /// @param s The tag string (one of the `originTagName` literals).
    /// @return The matching tag.
    inline OriginTag originTagFromString(const std::string& s) {
        static const std::unordered_map<std::string, OriginTag> TABLE = [] {
            std::unordered_map<std::string, OriginTag> t;
            for (std::size_t i = 0;
                 i < static_cast<std::size_t>(OriginTag::COUNT); ++i) {
                t.emplace(originTagName(static_cast<OriginTag>(i)),
                          static_cast<OriginTag>(i));
            }
            return t;
        }();
        auto it = TABLE.find(s);
        assert(it != TABLE.end());
        return it->second;
    }

    /// @brief Span twin of @ref originTagFromString — resolve a tag from
    ///        bytes with no `std::string` materialized.
    ///
    /// @details
    /// Byte-equivalent to the string form: a linear scan over the CLOSED
    /// `originTagName` table (the same table the string form's static map is
    /// seeded from) resolves the enumerator whose name equals the input
    /// bytes; both forms stop the world on a miss — an unknown tag is a new
    /// emission site that must extend the enum, never a value to pass
    /// through silently (Rule 19, the closed-enum contract of I-91). The
    /// table is a dozen-ish entries, so the scan costs less than the heap
    /// tag string the map lookup used to force on the caller.
    ///
    /// @param s The tag bytes (one of the `originTagName` literals).
    /// @return The matching tag; a miss never returns (assert for message
    ///         quality, then `std::abort` — loud even where asserts are
    ///         compiled out).
    /// @see originTagFromString(const std::string&) — the retained string
    ///      oracle; originTagName — the index-aligned table.
    inline OriginTag originTagFromString(const StrSpan& s) {
        for (int i = 0; i < static_cast<int>(OriginTag::COUNT); ++i) {
            const char* n = originTagName(static_cast<OriginTag>(i));
            if (equalSpans(s, StrSpan(n, static_cast<int32_t>(std::strlen(n))))) {
                return static_cast<OriginTag>(i);
            }
        }
        assert(false && "originTagFromString: unknown origin tag");
        std::abort(); // Rule 19: loud even with asserts compiled out
    }

    /// @brief Pack an origin-interner (expressionId, validityId) pair into
    ///        one `int64_t` map key.
    ///
    /// @details Both halves are int32 ids of the SAME per-LB origin
    /// interner (`Memory::originInterner`) — deliberately not the NameMap:
    /// the trace dumps the full NameMap id table, and mail-carried
    /// dependencies can name child-LB scopes never interned locally, so a
    /// NameMap mint here would break A/B trace comparability. The packing
    /// is bijective; key equality IS (expression, validity) string-pair
    /// equality.
    ///
    /// @param exprId     Interned expression id (high 32 bits).
    /// @param validityId Interned validity id (low 32 bits).
    /// @return The packed key.
    inline int64_t packOriginKey(int32_t exprId, int32_t validityId) {
        return packInt32Pair(exprId, validityId);
    }

    /// @brief Find-or-mint the packed origin key of an
    ///        (expression, validity) string pair (write half).
    ///
    /// @details Mints into the per-LB origin interner — single-threaded
    /// write sites only (I-83 discipline).
    ///
    /// @param oi       The owning LB's origin interner (mint side).
    /// @param expr     The expression string.
    /// @param validity The validity (scope) string.
    /// @return The packed key.
    inline int64_t mintOriginKey(ValueInterner& oi, const std::string& expr,
        const std::string& validity)
    {
        const int32_t exprId = oi.encode(expr);
        return packOriginKey(exprId, oi.encode(validity));
    }

    /// @brief Span overload of `mintOriginKey` — packs the origin key from
    ///        `StrSpan`s over the stable input, no `std::string` materialized.
    ///
    /// @details Byte-identical key to the `std::string` overload: `encode` is
    /// the same find-or-mint over the same bytes, and the pack is unchanged.
    /// Lets the absorb-door origin deposits skip the transient
    /// `ExpressionWithValidity` whose strings were only interned and dropped.
    ///
    /// @param oi       The owning LB's origin interner (mint side).
    /// @param expr     Span over the expression bytes.
    /// @param validity Span over the validity (scope) bytes.
    /// @return The packed key.
    inline int64_t mintOriginKey(ValueInterner& oi, const StrSpan& expr,
        const StrSpan& validity)
    {
        const int32_t exprId = oi.encode(expr);
        return packOriginKey(exprId, oi.encode(validity));
    }

    /// @brief Non-minting probe for the packed origin key of an
    ///        (expression, validity) string pair (read half).
    ///
    /// @param oi       The owning LB's origin interner (read-only).
    /// @param expr     The expression string.
    /// @param validity The validity (scope) string.
    /// @param outKey   The packed key on success; untouched on miss.
    /// @return `true` iff both strings are interned — a `false` is a
    ///         definitive "no such history entry" for any id-form origin
    ///         map of this LB.
    inline bool lookupOriginKey(const ValueInterner& oi, const std::string& expr,
        const std::string& validity, int64_t& outKey)
    {
        const int32_t exprId = oi.lookup(expr);
        if (exprId == 0) return false;
        const int32_t validityId = oi.lookup(validity);
        if (validityId == 0) return false;
        outKey = packOriginKey(exprId, validityId);
        return true;
    }

    /// @brief Look up the packed origin key of an `(expression, validity)`
    ///        pair — the `StrSpan` twin of
    ///        `lookupOriginKey(const ValueInterner&, const std::string&,
    ///        const std::string&, int64_t&)`.
    ///
    /// @details
    /// Byte-identical verdict to the `std::string` overload: probes @p expr
    /// and @p validity through the non-minting `ValueInterner::lookup(const
    /// StrSpan&)` and packs the two ids on success via `packOriginKey`. Taking
    /// spans lets the equivalence-class merge path
    /// (`mergeTwoEquivalenceClasses`) probe an origin key off the
    /// `StrSpan validityName` it threads without materialising a `std::string`
    /// scope copy. A `false` return is a definitive "no such history entry"
    /// for this LB's id-form origin map, identical to the string overload —
    /// either string un-interned means no such key can exist. The spans are
    /// read only during the synchronous lookups, so caller spans over
    /// caller-stable bytes cannot dangle.
    ///
    /// @param oi       The owning LB's origin interner (read-only).
    /// @param expr     The expression text span.
    /// @param validity The validity/scope name span.
    /// @param outKey   The packed key on success; untouched on miss.
    /// @return `true` iff both spans are interned in @p oi.
    /// @see lookupOriginKey(const ValueInterner&, const std::string&,
    ///      const std::string&, int64_t&) — the canonical overload.
    inline bool lookupOriginKey(const ValueInterner& oi, StrSpan expr,
        StrSpan validity, int64_t& outKey)
    {
        const int32_t exprId = oi.lookup(expr);
        if (exprId == 0) return false;
        const int32_t validityId = oi.lookup(validity);
        if (validityId == 0) return false;
        outKey = packOriginKey(exprId, validityId);
        return true;
    }

    /// @brief Recover the (expression, validity) string pair of a packed
    ///        origin key.
    ///
    /// @details Returns OWNED copies — safe to hold across further mints
    /// (I-3).
    ///
    /// @param pk The packed key.
    /// @param oi The owning LB's origin interner (read-only).
    /// @return `(expression, validity)` strings.
    inline std::pair<std::string, std::string> decodeOriginKey(int64_t pk,
        const ValueInterner& oi)
    {
        const int32_t exprId =
            static_cast<int32_t>(static_cast<uint64_t>(pk) >> 32);
        const int32_t validityId = static_cast<int32_t>(pk & 0xFFFFFFFFLL);
        return std::make_pair(oi.decodeView(exprId).toStdString(),
                              oi.decodeView(validityId).toStdString());
    }

    /// @brief Zero-copy twin of `decodeOriginKey` — split a packed origin key
    ///        into (expression, validity) spans over the origin interner's
    ///        cold bytes.
    ///
    /// @details
    /// Same unpack as `decodeOriginKey` (expression id = high 32 bits; validity
    /// id = low 32 bits), but writes each half into a caller `StrSpan&`
    /// out-param via `decodeView` rather than returning an owning
    /// `std::pair<std::string,std::string>` (09b row 87: no by-value heap in the
    /// signature). The still-heap origin-snapshot callers materialize an owned
    /// `std::string` at their own edge in the SAME statement as the decode, so
    /// no mint into @p oi intervenes and no heap pair is built here. The spans
    /// are stable only while the LB is resident and @p oi is un-mutated (I-3) —
    /// copy before any mint into @p oi.
    ///
    /// @param pk        The packed key (`packOriginKey(exprId, validityId)`).
    /// @param oi        The owning LB's origin interner (read-only).
    /// @param exprOut   Receives the expression span over @p oi's cold bytes.
    /// @param validityOut Receives the validity span over @p oi's cold bytes.
    /// @see decodeOriginKey — the owning-copy sibling kept for the visualizer
    ///      and the twin-test oracle; packOriginKey; decodeTemplateKeyView.
    inline void decodeOriginKeyView(int64_t pk, const ValueInterner& oi,
        StrSpan& exprOut, StrSpan& validityOut)
    {
        const int32_t exprId =
            static_cast<int32_t>(static_cast<uint64_t>(pk) >> 32);
        const int32_t validityId = static_cast<int32_t>(pk & 0xFFFFFFFFLL);
        exprOut = oi.decodeView(exprId);
        validityOut = oi.decodeView(validityId);
    }

    // IdOrigin / IdOriginMap are defined in memory_infra/int_encoded_expr.hpp
    // (shared with LbMemory's cold exprOriginMap member); OriginTag's full
    // definition is above.

    /// @brief Encode one string history line into its id form (write
    ///        half).
    ///
    /// @details Tag via the closed-vocabulary table (asserts on unknown);
    /// dependencies minted in positional order. Single-threaded write
    /// sites only.
    ///
    /// @param origin The string `(tag, antecedents)` record.
    /// @param oi     The owning LB's origin interner (mint side).
    /// @return The id-form record.
    inline IdOrigin encodeOrigin(
        const std::pair<std::string, std::vector<ExpressionWithValidity>>& origin,
        ValueInterner& oi)
    {
        IdOrigin out;
        out.first = originTagFromString(origin.first);
        out.second.reserve(origin.second.size());
        for (const ExpressionWithValidity& dep : origin.second) {
            out.second.push_back(mintOriginKey(oi, dep.original, dep.validityName));
        }
        return out;
    }

    /// @brief One antecedent of a transient origin record, as spans over
    ///        caller-stable bytes — the id-form-door input that replaces a
    ///        heap `ExpressionWithValidity` antecedent.
    ///
    /// @details
    /// Stack-only, trivially copyable, owns nothing: the two spans view the
    /// caller's `std::string` / `ScratchString` (or a per-slot scratch-arena
    /// buffer for a freshly-built antecedent). The origin doors mint the spans
    /// into the LB's `originInterner` synchronously, so the viewed bytes need
    /// outlive only the single door call. Positional order in a `deps[]` array
    /// is observable — it is the history line's dependency order — so a builder
    /// must fill it in the exact push order the equivalent `OriginLine` used and
    /// never sort or dedup it.
    ///
    /// @invariant The spans must alias buffers OTHER than the door's target
    ///            `originInterner` byte store — `mintOriginKey` mints, so a span
    ///            into that interner would dangle across the mint
    ///            ([I-3](../../docs/agentic_swdd/30_invariants.md#i-3)); copy or
    ///            recover-from-id first.
    /// @see encodeOriginSpans, encodeOrigin — the heap antecedent this replaces.
    struct OriginDep {
        StrSpan original;
        StrSpan validity;
    };

    /// @brief The threaded-origin carrier for the
    ///        `addExprToMemoryBlock -> addStatement -> addEquality /
    ///        addNegatedEquality` pass-through — an `OriginTag` + a borrowed
    ///        `OriginDep` span, replacing the threaded heap `OriginLine`.
    ///
    /// @details
    /// The deepest emission chain builds its origin at the TOP caller and
    /// consumes it several frames down. This POD carries that value with no
    /// heap: the backing `OriginDep[]` + antecedent bytes live on the top
    /// caller's stack, valid for the whole synchronous descent. `present`
    /// encodes today's empty `std::pair<>()` sentinel (the status-2
    /// disintegration path threads an empty origin that emits no history line):
    /// `present == false` means "no history line", and the door is never called;
    /// only a `present == true` carrier reaches an origin door, and then `tag`
    /// is a real tag, never `COUNT` — never feed an empty tag to
    /// `encodeOriginSpans`.
    ///
    /// @invariant `present == false` iff the source was an empty `OriginLine`; a
    ///            `present == true` carrier always has `tag != OriginTag::COUNT`
    ///            and `deps` valid for `depN` entries.
    /// @see OriginDep, encodeOriginSpans.
    struct TransientOrigin {
        bool present = false;
        OriginTag tag = OriginTag::COUNT;
        const OriginDep* deps = nullptr;
        int depN = 0;
    };

    /// @brief Span-antecedent twin of @ref encodeOrigin — encode a history line
    ///        from an `OriginTag` + `OriginDep` span, no transient heap
    ///        `OriginLine` built.
    ///
    /// @details
    /// Byte-identical id-form record to `encodeOrigin` on the equivalent
    /// `OriginLine`: `encodeOrigin` sets `out.first =
    /// originTagFromString(origin.first)` — exactly the enumerator the emission
    /// site names — and this takes that same enumerator directly (the tag
    /// string is never re-materialized). The dependencies are minted by the
    /// same `mintOriginKey` span twin over the same bytes in the same positional
    /// order, so `out.second` is the same packed-key vector. The tag is NOT
    /// interned by either path (`originTagFromString` reads a separate static
    /// table), so the `originInterner` mint sequence is the dependency mints
    /// alone — identical to `encodeOrigin`.
    ///
    /// @param tag  The history-line tag (the enumerator the site would have
    ///             named as a string).
    /// @param deps Pointer to `depN` antecedent spans, in positional order.
    /// @param depN The antecedent count (0 for an empty-antecedent line).
    /// @param oi   The owning LB's origin interner (mint side).
    /// @return The id-form `(tag, packed deps)` record — byte-identical to
    ///         `encodeOrigin` on the equivalent `OriginLine`.
    /// @invariant Single-threaded write sites only
    ///            ([I-83](../../docs/agentic_swdd/30_invariants.md#i-83)), like
    ///            `encodeOrigin`.
    /// @see encodeOrigin — the `OriginLine` overload this reproduces
    ///      byte-for-byte; mintOriginKey (span twin); OriginDep.
    inline IdOrigin encodeOriginSpans(OriginTag tag, const OriginDep* deps,
        int depN, ValueInterner& oi)
    {
        IdOrigin out;
        out.first = tag;
        out.second.reserve(static_cast<size_t>(depN));
        for (int i = 0; i < depN; ++i) {
            out.second.push_back(mintOriginKey(oi, deps[i].original,
                                               deps[i].validity));
        }
        return out;
    }

    /// @brief Mint an `OriginDep` span run into a caller int64 buffer — the
    ///        dependency-mint half of @ref encodeOriginSpans, no `IdOrigin`
    ///        heap vector built.
    ///
    /// @details
    /// Mints `out[i] = mintOriginKey(oi, deps[i].original, deps[i].validity)` in
    /// positional order — byte-identical id run to `encodeOriginSpans`'
    /// `out.second` (the same `mintOriginKey` span twin over the same bytes in
    /// the same order), but written into a caller-owned stack buffer instead of
    /// a heap `std::vector<int64_t>`. The capacity assert is a Rule-19 tripwire
    /// naming `kMaxOriginDeps` (realistic dependency counts are single-digit).
    /// Single-threaded write sites only
    /// ([I-83](../../docs/agentic_swdd/30_invariants.md#i-83)).
    ///
    /// @param deps   Pointer to `depN` antecedent spans, positional order.
    /// @param depN   The antecedent count (0 for an empty-antecedent line).
    /// @param oi     The owning LB's origin interner (mint side).
    /// @param out    Caller-owned destination for the `depN` packed keys.
    /// @param outCap `out`'s capacity in int64 elements.
    /// @return `depN` (the number of ids written).
    /// @invariant The written ids equal `encodeOriginSpans(tag, deps, depN,
    ///            oi).second` exactly (twin test
    ///            `mint_origin_deps_into_matches_encode_origin_spans`).
    /// @see encodeOriginSpans — the `IdOrigin`-returning form; mintOriginKey.
    inline int32_t mintOriginDepsInto(const OriginDep* deps, int depN,
        ValueInterner& oi, int64_t* out, int32_t outCap)
    {
        assert(depN <= outCap
            && "mintOriginDepsInto: dependency count exceeds the caller buffer "
               "(Rule-19 tripwire — kMaxOriginDeps)");
        for (int i = 0; i < depN; ++i) {
            out[i] = mintOriginKey(oi, deps[i].original, deps[i].validity);
        }
        return depN;
    }

    /// @brief Mint an EWV antecedent vector into a caller int64 buffer — the
    ///        dependency-mint half of @ref encodeOrigin, no `IdOrigin` heap
    ///        vector built.
    ///
    /// @details
    /// Mints `out[i] = mintOriginKey(oi, deps[i].original, deps[i].validityName)`
    /// in EWV-vector order — byte-identical id run to `encodeOrigin`'s
    /// `out.second` (the same mints in the same positional order) into a
    /// caller-owned stack buffer. The capacity assert is a Rule-19 tripwire
    /// naming `kMaxOriginDeps`. Single-threaded write sites only
    /// ([I-83](../../docs/agentic_swdd/30_invariants.md#i-83)).
    ///
    /// @param deps   The antecedent EWV vector, positional order.
    /// @param oi     The owning LB's origin interner (mint side).
    /// @param out    Caller-owned destination for the `deps.size()` packed keys.
    /// @param outCap `out`'s capacity in int64 elements.
    /// @return `deps.size()` (the number of ids written).
    /// @invariant The written ids equal `encodeOrigin({tag, deps}, oi).second`
    ///            exactly (twin test
    ///            `mint_origin_deps_from_ewv_into_matches_encode_origin`).
    /// @see encodeOrigin — the `IdOrigin`-returning form; mintOriginKey.
    inline int32_t mintOriginDepsFromEWVInto(
        const std::vector<ExpressionWithValidity>& deps, ValueInterner& oi,
        int64_t* out, int32_t outCap)
    {
        const int32_t depN = static_cast<int32_t>(deps.size());
        assert(depN <= outCap
            && "mintOriginDepsFromEWVInto: dependency count exceeds the caller "
               "buffer (Rule-19 tripwire — kMaxOriginDeps)");
        for (int32_t i = 0; i < depN; ++i) {
            out[i] = mintOriginKey(oi, deps[static_cast<size_t>(i)].original,
                                   deps[static_cast<size_t>(i)].validityName);
        }
        return depN;
    }

    /// @brief Decode one id-form history line back to the string form
    ///        (read half).
    ///
    /// @details Owned copies throughout (I-3); the tag string is the exact
    /// historical literal.
    ///
    /// @param origin The id-form `(tag, packed deps)` record.
    /// @param oi     The owning LB's origin interner (read-only).
    /// @return The string `(tag, antecedents)` record.
    inline std::pair<std::string, std::vector<ExpressionWithValidity>> decodeOrigin(
        const IdOrigin& origin, const ValueInterner& oi)
    {
        std::pair<std::string, std::vector<ExpressionWithValidity>> out;
        out.first = originTagName(origin.first);
        out.second.reserve(origin.second.size());
        for (const int64_t pk : origin.second) {
            StrSpan eSpan, vSpan;
            decodeOriginKeyView(pk, oi, eSpan, vSpan);
            out.second.emplace_back(eSpan.toStdString(), vSpan.toStdString());
        }
        return out;
    }

    /// @brief Id-form twin of `ExpressionAnalyzer::addOrigin` — append a
    ///        history line with the D-49 cap-full preference policy.
    ///
    /// @details Replicates the string form exactly: below the cap, append
    /// if absent (set semantics in the vector); at the cap, an already
    /// present record keeps existing slots; a NEW non-equality record
    /// replaces the first equality-convenience slot (`equality1` /
    /// `equality2`) — foundation displaces convenience; otherwise the
    /// existing slot wins (insertion order). Record equality on ids IS
    /// string equality (the interner is bijective).
    ///
    /// @param map        The id-form origin map.
    /// @param key        Packed (expressionId, validityId) key.
    /// @param origin     The id-form record to insert.
    /// @param maxOrigins Cap; caller picks the compressor or standard
    ///                   parameter exactly as at the string sites.
    inline void addOriginId(IdOriginMap& map, int64_t key, const IdOrigin& origin,
        int maxOrigins)
    {
        auto& vec = map[key];
        if (vec.size() < static_cast<size_t>(maxOrigins)) {
            if (std::find(vec.begin(), vec.end(), origin) == vec.end()) {
                vec.push_back(origin);
            }
            return;
        }
        // Cap-full origin-preference replacement (D-49): foundation
        // displaces convenience — see the string twin for the full policy
        // rationale.
        if (std::find(vec.begin(), vec.end(), origin) != vec.end()) return;
        auto isEqualityConvenienceTag = [](OriginTag tag) {
            return tag == OriginTag::equality1 || tag == OriginTag::equality2;
        };
        if (!isEqualityConvenienceTag(origin.first)) {
            for (auto& slot : vec) {
                if (isEqualityConvenienceTag(slot.first)) {
                    slot = origin;
                    return;
                }
            }
        }
        // Otherwise: existing slot wins (insertion-order tiebreak).
    }

    /// @brief POD overload of the heap @ref addOriginId — append a `(tag, deps)`
    ///        history line with the D-49 cap-full preference, no transient
    ///        `IdOrigin` materialized.
    ///
    /// @details
    /// Reproduces the `const IdOrigin&` overload's policy byte-for-byte while
    /// growing ONLY the sink container's heap (the caller-owned `IdOriginMap`),
    /// never a function-local `IdOrigin`: dedup by a manual whole-record compare
    /// (`o.first == OriginTag(tag)` && `o.second` matches `deps[0..depN)` —
    /// exactly `IdOrigin::operator==`); below the cap, `emplace_back()` a fresh
    /// slot and fill its `first`/`second` in place; at the cap, the D-49
    /// convenience-replace overwrites the first `equality1`/`equality2` slot's
    /// fields in place (`slot.first = OriginTag(tag); slot.second.assign(deps,
    /// deps + depN)`). Record equality on ids IS string equality (the interner is
    /// bijective). Single-threaded write sites only
    /// ([I-83](../../docs/agentic_swdd/30_invariants.md#i-83)).
    ///
    /// @param map        The id-form origin map.
    /// @param key        Packed (expressionId, validityId) key.
    /// @param tag        The history-line tag byte.
    /// @param deps       Pointer to `depN` dependency packed keys, positional
    ///                   order.
    /// @param depN       The dependency count.
    /// @param maxOrigins Cap; caller picks the compressor or standard parameter.
    /// @invariant The resulting map state equals the `const IdOrigin&` overload
    ///            on the equivalent record (twin test
    ///            `add_origin_id_heap_pod_matches_idorigin`).
    /// @see addOriginId(IdOriginMap&, int64_t, const IdOrigin&, int) — the
    ///      `IdOrigin` overload this reproduces byte-for-byte.
    inline void addOriginId(IdOriginMap& map, int64_t key, uint8_t tag,
        const int64_t* deps, int32_t depN, int maxOrigins)
    {
        auto& vec = map[key];
        const OriginTag otag = static_cast<OriginTag>(tag);
        const auto matches = [&](const IdOrigin& o) {
            if (o.first != otag) return false;
            if (static_cast<int32_t>(o.second.size()) != depN) return false;
            for (int32_t i = 0; i < depN; ++i) {
                if (o.second[static_cast<size_t>(i)] != deps[i]) return false;
            }
            return true;
        };
        if (vec.size() < static_cast<size_t>(maxOrigins)) {
            for (const IdOrigin& o : vec) {
                if (matches(o)) return;
            }
            vec.emplace_back();
            IdOrigin& back = vec.back();
            back.first = otag;
            back.second.reserve(static_cast<size_t>(depN));
            for (int32_t i = 0; i < depN; ++i) back.second.push_back(deps[i]);
            return;
        }
        // Cap-full origin-preference replacement (D-49): foundation displaces
        // convenience.
        for (const IdOrigin& o : vec) {
            if (matches(o)) return;
        }
        auto isEqualityConvenienceTag = [](OriginTag t) {
            return t == OriginTag::equality1 || t == OriginTag::equality2;
        };
        if (!isEqualityConvenienceTag(otag)) {
            for (auto& slot : vec) {
                if (isEqualityConvenienceTag(slot.first)) {
                    slot.first = otag;
                    slot.second.assign(deps, deps + depN);
                    return;
                }
            }
        }
        // Otherwise: existing slot wins (insertion-order tiebreak).
    }

    /// @brief Id-form twin of `ExpressionAnalyzer::overwriteOrigins` —
    ///        merge `right`'s history lines into `left` up to the cap.
    ///
    /// @details Per key: append `right`'s records in their vector order,
    /// dedup by equality, stop appending at the cap (no D-49 replacement
    /// here — the string twin has none either). Keys are independent (the
    /// cap is per key), so `right`'s map iteration order cannot affect the
    /// merged content — an unordered source is safe.
    ///
    /// @param left       Destination map (typically the merged class map).
    /// @param right      Source map.
    /// @param maxOrigins Per-key cap.
    inline void overwriteOriginsId(IdOriginMap& left, const IdOriginMap& right,
        int maxOrigins)
    {
        for (const auto& kv : right) {
            auto& leftVec = left[kv.first];
            for (const IdOrigin& orig : kv.second) {
                if (leftVec.size() < static_cast<size_t>(maxOrigins)) {
                    if (std::find(leftVec.begin(), leftVec.end(), orig)
                        == leftVec.end()) {
                        leftVec.push_back(orig);
                    }
                }
            }
        }
    }

    /// @brief Decode an id-form origin map into key-sorted string rows —
    ///        the shared derivation for every order-sensitive walk.
    ///
    /// @details Produces `(key, history lines)` rows sorted by the decoded
    /// (expression, validity) pair — exactly the historical
    /// `std::map<ExpressionWithValidity, …>` iteration order
    /// (`ExpressionWithValidity::operator<` is `(original, validityName)`
    /// lex). Per-row history lines keep their insertion order. Owned
    /// strings throughout (I-3).
    ///
    /// @param map The id-form origin map.
    /// @param oi  The owning LB's origin interner (read-only).
    /// @return Rows of `((expression, validity), string history lines)`,
    ///         key-sorted.
    inline std::vector<std::pair<std::pair<std::string, std::string>,
        std::vector<std::pair<std::string, std::vector<ExpressionWithValidity>>>>>
    decodeOriginMapSorted(const IdOriginMap& map, const ValueInterner& oi)
    {
        std::vector<std::pair<std::pair<std::string, std::string>,
            std::vector<std::pair<std::string, std::vector<ExpressionWithValidity>>>>> rows;
        rows.reserve(map.size());
        for (const auto& kv : map) {
            std::vector<std::pair<std::string, std::vector<ExpressionWithValidity>>> lines;
            lines.reserve(kv.second.size());
            for (const IdOrigin& origin : kv.second) {
                lines.push_back(decodeOrigin(origin, oi));
            }
            StrSpan eSpan, vSpan;
            decodeOriginKeyView(kv.first, oi, eSpan, vSpan);
            rows.emplace_back(std::make_pair(eSpan.toStdString(), vSpan.toStdString()),
                std::move(lines));
        }
        std::sort(rows.begin(), rows.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
        return rows;
    }

    /// @brief Encode-and-append convenience for the body-map emission
    ///        sites: string `(expression, validity)` key + string history
    ///        record -> id-form `addOriginId`.
    ///
    /// @details Minting wrapper (single-threaded write sites only,
    /// [I-83](../../docs/agentic_swdd/30_invariants.md#i-83)). The paired
    /// mail writes at the same sites keep using the string `addOrigin` —
    /// mail origin maps stay string per the batch boundary.
    ///
    /// @param map        The id-form origin map.
    /// @param oi         The owning LB's origin interner (mint side).
    /// @param ev         `(expression, validity)` key.
    /// @param origin     `(tag, antecedents)` string record.
    /// @param maxOrigins Cap, exactly as at the string sites.
    inline void addOriginEncoded(IdOriginMap& map, ValueInterner& oi,
        const ExpressionWithValidity& ev,
        const std::pair<std::string, std::vector<ExpressionWithValidity>>& origin,
        int maxOrigins)
    {
        // Key-first (user byte-identity waiver): mint the origin KEY, resolve
        // the tag via the span twin (interns nothing), then mint the EWV
        // antecedents positionally — no IdOrigin materialized.
        const int64_t key = mintOriginKey(oi, ev.original, ev.validityName);
        const OriginTag tag = originTagFromString(StrSpan(origin.first));
        int64_t d[ExecutionParameters::kMaxOriginDeps];
        const int32_t n = mintOriginDepsFromEWVInto(origin.second, oi, d,
            ExecutionParameters::kMaxOriginDeps);
        addOriginId(map, key, static_cast<uint8_t>(tag), d, n, maxOrigins);
    }

    /// @brief Span-antecedent twin of the heap `addOriginEncoded` — key AND
    ///        record built from spans + an `OriginTag`, no transient
    ///        `ExpressionWithValidity` / `OriginLine` materialized.
    ///
    /// @details
    /// Same-content deposit as the `ExpressionWithValidity` / `OriginLine`
    /// overload: the same `addOriginId` sink over the same `mintOriginKey`
    /// (span twin) key and the same dependency mints in the same positional
    /// order — but through the POD dep-run door, so no transient `IdOrigin` is
    /// materialized. Mint order is KEY-FIRST (mint the origin key, then the
    /// dependencies via `mintOriginDepsInto`): this pins the previously
    /// compiler-chosen (unspecified argument-evaluation) order of the old
    /// `addOriginId(map, mintOriginKey(...), encodeOriginSpans(...), cap)` to a
    /// deterministic key-first sequence under the user's byte-identity waiver.
    /// The `files/` proof artifacts are id-value-independent (the chapter export
    /// decodes `originInterner` ids back to strings); only the gitignored
    /// `.deload/` id VALUES may reassign. The D-49 cap-full preference fires
    /// identically (it reads the tag + deps, which match byte-for-byte).
    ///
    /// @param map        The id-form origin map.
    /// @param oi         The owning LB's origin interner (mint side).
    /// @param evO        Span over the key expression bytes.
    /// @param evV        Span over the key validity (scope) bytes.
    /// @param tag        The history-line tag (the enumerator the site names).
    /// @param deps       Pointer to `depN` antecedent spans, positional order.
    /// @param depN       The antecedent count.
    /// @param maxOrigins Cap, exactly as at the string sites.
    /// @invariant The spans must alias buffers OTHER than @p oi's own byte store
    ///            ([I-3](../../docs/agentic_swdd/30_invariants.md#i-3)).
    /// @see addOriginEncoded(IdOriginMap&, ValueInterner&, const ExpressionWithValidity&, const OriginLine&, int)
    ///      — the owning-string overload this reproduces byte-for-byte;
    ///      encodeOriginSpans.
    inline void addOriginEncoded(IdOriginMap& map, ValueInterner& oi,
        const StrSpan& evO, const StrSpan& evV,
        OriginTag tag, const OriginDep* deps, int depN, int maxOrigins)
    {
        // Key-first (user byte-identity waiver): mint the origin KEY, then the
        // dependency ids positionally into a stack buffer — no IdOrigin built.
        const int64_t key = mintOriginKey(oi, evO, evV);
        int64_t d[ExecutionParameters::kMaxOriginDeps];
        const int32_t n = mintOriginDepsInto(deps, depN, oi, d,
            ExecutionParameters::kMaxOriginDeps);
        addOriginId(map, key, static_cast<uint8_t>(tag), d, n, maxOrigins);
    }

    // ============= Mail-cold decoded-order comparators =================
    // (L8 — the entrance decode-snapshot elimination in
    // standardProcessing::absorb; I-138)
    //
    // The absorb path consumed two heap std::vector snapshots decoded from the
    // already-id-form cold mail (statements + origins), each std::sorted into
    // the canonical string order before its consumers ran. To iterate the cold
    // ids directly and reproduce those std::sort orders byte-for-byte, the three
    // comparators below compare cold key/record forms via the SAME decoded-lex
    // rule the string std::sorts used — never a raw id / raw tag order
    // (I-84). Each is templated on the id-space interner (per-LB NameMap /
    // originInterner for the internal channel; the global mailInterner for the
    // routing channel), dispatched to a StrSpan by `mailIdView`.

    /// @brief Decode a mail id-space id to a byte span — the `NameMap`
    ///        overload of the mail-comparator id→span dispatcher.
    ///
    /// @details
    /// The mail-cold comparators (`originLineLessId`, `decodedOriginKeyLess`,
    /// `decodedStatementLess`) are templated on the interner that owns the id
    /// space of the column they order, and recover each id's bytes through this
    /// overload set so one comparator body serves every id space. The internal
    /// channel's statement ids are per-LB `NameMap` ids stored in an `int32_t`
    /// field; this overload narrows to the `int16_t` the `NameMap` decodes and
    /// returns the zero-copy span, identical to every other decoded-lex
    /// `NameMap` walk (`unionMemberIdsByName`, `sortStatementRows`).
    ///
    /// @param nm The owning LB's `NameMap` (read-only).
    /// @param id The `NameMap` id, widened into an `int32_t` field.
    /// @return Span over the id's bytes, stable while the LB is resident.
    /// @invariant Read-only — no mint; two spans from one interner stay valid
    ///            simultaneously (the two-`decodeView` compare the decoded-lex
    ///            primitives already rely on, I-3).
    /// @see mailIdView(const ValueInterner&, int32_t),
    ///      mailIdView(const ColdStringTable&, int32_t).
    inline StrSpan mailIdView(const NameMap& nm, int32_t id) {
        return nm.decodeView(static_cast<int16_t>(id));
    }

    /// @brief Decode a mail id-space id to a byte span — the `ValueInterner`
    ///        overload of the mail-comparator id→span dispatcher.
    ///
    /// @details The internal channel's origin ids are per-LB `originInterner`
    /// (`ValueInterner`) ids; this overload returns the interner's zero-copy
    /// decode span, byte-identical to the `decode(id)` string the heap snapshot
    /// held. Read-only; no mint.
    ///
    /// @param vi The owning LB's origin interner (read-only).
    /// @param id The interner id.
    /// @return Span over the id's bytes, stable while the LB is resident.
    /// @see mailIdView(const NameMap&, int32_t),
    ///      mailIdView(const ColdStringTable&, int32_t).
    inline StrSpan mailIdView(const ValueInterner& vi, int32_t id) {
        return vi.decodeView(id);
    }

    /// @brief Decode a mail id-space id to a byte span — the `ColdStringTable`
    ///        overload of the mail-comparator id→span dispatcher.
    ///
    /// @details The routing channel's statement AND origin ids are GLOBAL
    /// `mailInterner` (`ColdStringTable`) ids; this overload returns the table's
    /// zero-copy `view` span. Read-only; no mint (the routing mail ids are
    /// frozen during the parallel phase, I-127).
    ///
    /// @param t  The global mail interner (read-only).
    /// @param id The interner id.
    /// @return Span over the id's bytes, stable while the table is alive.
    /// @see mailIdView(const NameMap&, int32_t),
    ///      mailIdView(const ValueInterner&, int32_t).
    inline StrSpan mailIdView(const ColdStringTable& t, int32_t id) {
        return t.view(id);
    }

    /// @brief Zero-copy view over one `IntMailOrigin` blob record — tag byte +
    ///        the contiguous packed dependency-key run, parsed against the
    ///        `Codec<IntMailOrigin>` layout without materializing a record.
    ///
    /// @details
    /// The absorb-path origin sort + deposit read each cold origin record
    /// straight from the blob pool (via `TypedColdBlobMap::peekRecordBytes`),
    /// never a decoding `recordAt` per comparison. This view exposes the two
    /// fields the OriginLine order + deposit need — the `OriginTag` byte and the
    /// packed `(exprId, validityId)` dependency keys — as a pointer into the
    /// blob bytes. Dependency keys are read one at a time through
    /// `mailOriginDepAt` (an unaligned-safe `memcpy`), so `depsBytes` need not be
    /// 8-byte aligned. Layout mirrors `Codec<IntMailOrigin>::serialize` exactly:
    /// `uint8 tag`, `int32 depCount`, then `depCount` little-endian `int64`
    /// dependency keys; the parse asserts the length matches (Rule 19).
    ///
    /// @invariant `depsBytes` aliases the caller's peeked bytes and inherits
    ///            their lifetime — valid only while the source blob (pool page or
    ///            straddle-scratch) is alive; never stored past the call.
    /// @see mailOriginDepAt, Codec<IntMailOrigin>, originLineLessId.
    struct MailOriginBlobView {
        uint8_t tag = 0;          ///< The `OriginTag` enumerator, raw.
        const char* depsBytes = nullptr;  ///< First dependency key's bytes.
        int32_t depN = 0;         ///< Dependency-key count.
    };

    /// @brief Parse an `IntMailOrigin` blob's bytes into a zero-copy
    ///        @ref MailOriginBlobView.
    ///
    /// @details The exact inverse of `Codec<IntMailOrigin>::serialize`'s field
    /// layout — reads the tag byte and the `int32` dependency count, then points
    /// `depsBytes` at the dependency run. Asserts the byte length is exactly
    /// `5 + 8 * depN` (Rule 19): a mismatch is a codec-vs-view drift, never a
    /// tolerated shape.
    ///
    /// @param data The record blob bytes (from `peekRecordBytes`).
    /// @param len  The blob byte length.
    /// @return A view over `data`; `depsBytes` aliases `data`.
    /// @see MailOriginBlobView, mailOriginDepAt.
    inline MailOriginBlobView viewMailOriginBlob(const char* data, int32_t len) {
        assert(data != nullptr && len >= 5
            && "viewMailOriginBlob: truncated IntMailOrigin blob");
        MailOriginBlobView v;
        v.tag = static_cast<uint8_t>(data[0]);
        std::memcpy(&v.depN, data + 1, sizeof(int32_t));
        assert(v.depN >= 0 && len == 5 + 8 * v.depN
            && "viewMailOriginBlob: IntMailOrigin length mismatch");
        v.depsBytes = data + 5;
        return v;
    }

    /// @brief Read dependency key @p i of a @ref MailOriginBlobView
    ///        (unaligned-safe).
    ///
    /// @details A `memcpy` load of the `i`-th packed `(exprId, validityId)` key —
    /// the dependency run is not guaranteed 8-byte aligned in the blob, so a raw
    /// `int64_t` cast would be undefined. The returned key is unpacked with
    /// `packOriginKey`'s inverse shifts at the read site.
    ///
    /// @param v The blob view.
    /// @param i Dependency index in `[0, v.depN)`.
    /// @return The packed dependency key.
    /// @see MailOriginBlobView, packOriginKey.
    inline int64_t mailOriginDepAt(const MailOriginBlobView& v, int32_t i) {
        assert(i >= 0 && i < v.depN);
        int64_t d;
        std::memcpy(&d, v.depsBytes + static_cast<std::size_t>(8) * i,
                    sizeof(int64_t));
        return d;
    }

    /// @brief Zero-copy view over one `IdOrigin` blob record — tag byte + the
    ///        contiguous packed dependency-key run, parsed against the
    ///        `Codec<IdOrigin>` layout without materializing an `IdOrigin`.
    ///
    /// @details
    /// The internal-origin peer of @ref MailOriginBlobView: `exprOriginMap` and
    /// `EquivalenceClass::equalityOriginMap` store `IdOrigin` records
    /// (`std::pair<OriginTag, std::vector<int64_t>>`) whose canonical blob
    /// (`Codec<IdOrigin>::serialize`) is byte-for-byte identical to the mail
    /// codec's — `uint8 tag`, `int32 depCount`, then `depCount` little-endian
    /// `int64` packed `(exprId, validityId)` keys. The central-add front-origin
    /// read (`addExprToMemoryBlock`) peeks record 0 of a key's run straight from
    /// the blob pool via `peekRecordBytes` and walks this view instead of
    /// decoding a heap `IdOrigin`. Dependency keys are read one at a time through
    /// `idOriginDepAt` (an unaligned-safe `memcpy`), so `depsBytes` need not be
    /// 8-byte aligned.
    ///
    /// @invariant `depsBytes` aliases the caller's peeked bytes and inherits
    ///            their lifetime — valid only while the source blob (pool page or
    ///            straddle-scratch) is alive and unmutated; never stored past the
    ///            call (I-116).
    /// @see idOriginDepAt, Codec<IdOrigin>, MailOriginBlobView.
    struct IdOriginBlobView {
        uint8_t tag = 0;          ///< The `OriginTag` enumerator, raw.
        const char* depsBytes = nullptr;  ///< First dependency key's bytes.
        int32_t depN = 0;         ///< Dependency-key count.
    };

    /// @brief Parse an `IdOrigin` blob's bytes into a zero-copy
    ///        @ref IdOriginBlobView.
    ///
    /// @details The exact inverse of `Codec<IdOrigin>::serialize`'s field layout
    /// — reads the tag byte and the `int32` dependency count, then points
    /// `depsBytes` at the dependency run. Asserts the byte length is exactly
    /// `5 + 8 * depN` (Rule 19): a mismatch is a codec-vs-view drift, never a
    /// tolerated shape.
    ///
    /// @param data The record blob bytes (from `peekRecordBytes`).
    /// @param len  The blob byte length.
    /// @return A view over @p data; `depsBytes` aliases @p data.
    /// @see IdOriginBlobView, idOriginDepAt.
    inline IdOriginBlobView viewIdOriginBlob(const char* data, int32_t len) {
        assert(data != nullptr && len >= 5
            && "viewIdOriginBlob: truncated IdOrigin blob");
        IdOriginBlobView v;
        v.tag = static_cast<uint8_t>(data[0]);
        std::memcpy(&v.depN, data + 1, sizeof(int32_t));
        assert(v.depN >= 0 && len == 5 + 8 * v.depN
            && "viewIdOriginBlob: IdOrigin length mismatch");
        v.depsBytes = data + 5;
        return v;
    }

    /// @brief Read dependency key @p i of an @ref IdOriginBlobView
    ///        (unaligned-safe).
    ///
    /// @details A `memcpy` load of the `i`-th packed `(exprId, validityId)` key —
    /// the dependency run is not guaranteed 8-byte aligned in the blob, so a raw
    /// `int64_t` cast would be undefined. The returned key is unpacked with
    /// `packOriginKey`'s inverse shifts at the read site.
    ///
    /// @param v The blob view.
    /// @param i Dependency index in `[0, v.depN)`.
    /// @return The packed dependency key.
    /// @see IdOriginBlobView, packOriginKey.
    inline int64_t idOriginDepAt(const IdOriginBlobView& v, int32_t i) {
        assert(i >= 0 && i < v.depN);
        int64_t d;
        std::memcpy(&d, v.depsBytes + static_cast<std::size_t>(8) * i,
                    sizeof(int64_t));
        return d;
    }

    /// @brief Decoded-order comparator over two mail-origin records — the
    ///        byte-exact twin of `OriginLine::operator<`
    ///        (`std::pair<std::string, std::vector<ExpressionWithValidity>>`).
    ///
    /// @details
    /// The per-run origin `std::sort` the absorb path ran (over the decoded
    /// `std::vector<OriginLine>`) is reproduced on the id-form records by this
    /// comparator so the bulk `exprOriginMap` merge and the per-statement
    /// `.front()` origin pick still consume the runs in the exact same order.
    /// The order rule mirrors `std::pair::operator<` on the decoded string form
    /// EXACTLY: FIRST the justification tag as a STRING (`originTagName` compared
    /// byte-lex via `compareSpans` — NOT the raw `OriginTag` enum value, whose
    /// order differs, I-84); on a tag tie the antecedent list
    /// LEXICOGRAPHICALLY, element-wise under `ExpressionWithValidity::operator<`
    /// (decoded `original` then `validityName`), the shorter run ordering first
    /// on a prefix tie (`std::vector::operator<` semantics). Each dependency key
    /// packs `(originalId, validityId)`; equal packed keys short-circuit (the
    /// interner is bijective, so equal ids ARE equal strings — the
    /// `valueIdVectorLess` optimization). Distinct records never tie (they differ
    /// in tag or a dependency), so the induced `std::sort` order is fully
    /// determined and byte-identical to the heap path regardless of sort
    /// stability.
    ///
    /// @tparam Interner The id space of the records' dependency keys
    ///         (`ValueInterner` internal / `ColdStringTable` routing), via
    ///         `mailIdView`.
    /// @param a  First record view.
    /// @param b  Second record view.
    /// @param oi The dependency-key interner (read-only).
    /// @return `true` iff decoded @p a orders before decoded @p b.
    /// @invariant Read-only — no mint across the compare, so the two per-key
    ///            `mailIdView` spans stay valid simultaneously (I-3).
    /// @see OriginLine, mailOriginDepAt, compareSpans, decodedOriginKeyLess.
    template <typename Interner>
    inline bool originLineLessId(const MailOriginBlobView& a,
        const MailOriginBlobView& b, const Interner& oi)
    {
        const char* nameA = originTagName(static_cast<OriginTag>(a.tag));
        const char* nameB = originTagName(static_cast<OriginTag>(b.tag));
        const int tc = compareSpans(
            StrSpan(nameA, static_cast<int32_t>(std::strlen(nameA))),
            StrSpan(nameB, static_cast<int32_t>(std::strlen(nameB))));
        if (tc != 0) return tc < 0;
        const int32_t n = a.depN < b.depN ? a.depN : b.depN;
        for (int32_t i = 0; i < n; ++i) {
            const int64_t da = mailOriginDepAt(a, i);
            const int64_t db = mailOriginDepAt(b, i);
            if (da == db) continue;
            const int32_t exprA = static_cast<int32_t>(static_cast<uint64_t>(da) >> 32);
            const int32_t validA = static_cast<int32_t>(da & 0xFFFFFFFFLL);
            const int32_t exprB = static_cast<int32_t>(static_cast<uint64_t>(db) >> 32);
            const int32_t validB = static_cast<int32_t>(db & 0xFFFFFFFFLL);
            const int ec = compareSpans(mailIdView(oi, exprA), mailIdView(oi, exprB));
            if (ec != 0) return ec < 0;
            const int vc = compareSpans(mailIdView(oi, validA), mailIdView(oi, validB));
            if (vc != 0) return vc < 0;
        }
        return a.depN < b.depN;
    }

    /// @brief Decoded-order comparator over two packed origin-map keys — the
    ///        byte-exact twin of `ExpressionWithValidity::operator<` on the
    ///        keys' decoded `(expression, validity)` pairs.
    ///
    /// @details
    /// The absorb origin snapshot was EWV-key sorted (`decode*Origins` sorts the
    /// rows by `a.first < b.first`) before the bulk merge iterated it; that row
    /// order drives BOTH the `exprOriginMap` key mint order (a cold map — key
    /// ids assigned in first-insertion order) AND, on the routing channel, the
    /// `originInterner` re-mint order — so it is deload-observable and must be
    /// reproduced (plan risk 3). This comparator orders two `origins_` cold-map
    /// keys by the decoded `(expression, validity)` pair — `original` then
    /// `validityName` via `compareSpans` — exactly `ExpressionWithValidity::
    /// operator<`. Equal packed keys short-circuit (bijective interner). Keys are
    /// unique per row, so no non-trivial ties.
    ///
    /// @tparam Interner The keys' id space (`ValueInterner` internal /
    ///         `ColdStringTable` routing), via `mailIdView`.
    /// @param keyA First packed `(exprId, validityId)` key.
    /// @param keyB Second packed key.
    /// @param oi   The key interner (read-only).
    /// @return `true` iff decoded @p keyA orders before decoded @p keyB.
    /// @invariant Read-only — no mint; the two `mailIdView` spans coexist (I-3).
    /// @see packOriginKey, originLineLessId, compareSpans.
    template <typename Interner>
    inline bool decodedOriginKeyLess(int64_t keyA, int64_t keyB,
        const Interner& oi)
    {
        if (keyA == keyB) return false;
        const int32_t exprA = static_cast<int32_t>(static_cast<uint64_t>(keyA) >> 32);
        const int32_t validA = static_cast<int32_t>(keyA & 0xFFFFFFFFLL);
        const int32_t exprB = static_cast<int32_t>(static_cast<uint64_t>(keyB) >> 32);
        const int32_t validB = static_cast<int32_t>(keyB & 0xFFFFFFFFLL);
        const int ec = compareSpans(mailIdView(oi, exprA), mailIdView(oi, exprB));
        if (ec != 0) return ec < 0;
        return compareSpans(mailIdView(oi, validA), mailIdView(oi, validB)) < 0;
    }

    /// @brief Decoded-order comparator over two `IntMailStatementKey` byte keys —
    ///        the byte-exact twin of
    ///        `std::pair<ExpressionWithValidity, std::set<int>>::operator<`.
    ///
    /// @details
    /// The absorb statement snapshot was sorted by `std::pair<EWV,
    /// std::set<int>>::operator<` before the drain; that order is observable
    /// (`intExternalStatements` push order, rule-firing order, delta order). This
    /// comparator reproduces it on the id-form set keys: decode `originalId`
    /// then `validityId` through @p interner and `compareSpans` (the EWV half),
    /// then, on an EWV tie, compare the ascending `levels` runs lexicographically
    /// with the shorter run first — exactly `std::set<int>::operator<` (the
    /// stored levels are ascending + duplicate-free, so a raw element-wise
    /// compare reproduces the set order). The key bytes are parsed against the
    /// `Codec<IntMailStatementKey>` layout (`int32 originalId`, `int32
    /// validityId`, `int32 count`, `count × int32` levels). Two members sharing
    /// `(expression, scope)` but differing in levels stay distinct and ordered by
    /// levels — the multiplicity the drain's level gate relies on.
    ///
    /// @tparam Interner The keys' id space (`NameMap` internal / `ColdStringTable`
    ///         routing), via `mailIdView`.
    /// @param aKey     First key's raw bytes (`statements_.keyAt(id)`).
    /// @param bKey     Second key's raw bytes.
    /// @param interner The `(originalId, validityId)` id space (read-only).
    /// @return `true` iff decoded @p aKey orders before decoded @p bKey.
    /// @invariant Read-only; the id spans are extracted from the key bytes BEFORE
    ///            any `mailIdView` call, so the (distinct-store) interner decode
    ///            cannot clobber them (I-3).
    /// @see IntMailStatementKey, Codec<IntMailStatementKey>, decodedOriginKeyLess.
    template <typename Interner>
    inline bool decodedStatementLess(StrSpan aKey, StrSpan bKey,
        const Interner& interner)
    {
        assert(aKey.len >= 12 && bKey.len >= 12
            && "decodedStatementLess: truncated IntMailStatementKey");
        int32_t oA, vA, cA, oB, vB, cB;
        std::memcpy(&oA, aKey.ptr + 0, sizeof(int32_t));
        std::memcpy(&vA, aKey.ptr + 4, sizeof(int32_t));
        std::memcpy(&cA, aKey.ptr + 8, sizeof(int32_t));
        std::memcpy(&oB, bKey.ptr + 0, sizeof(int32_t));
        std::memcpy(&vB, bKey.ptr + 4, sizeof(int32_t));
        std::memcpy(&cB, bKey.ptr + 8, sizeof(int32_t));
        const int oc = compareSpans(mailIdView(interner, oA), mailIdView(interner, oB));
        if (oc != 0) return oc < 0;
        const int vc = compareSpans(mailIdView(interner, vA), mailIdView(interner, vB));
        if (vc != 0) return vc < 0;
        const int32_t n = cA < cB ? cA : cB;
        for (int32_t i = 0; i < n; ++i) {
            int32_t lA, lB;
            std::memcpy(&lA, aKey.ptr + 12 + 4 * i, sizeof(int32_t));
            std::memcpy(&lB, bKey.ptr + 12 + 4 * i, sizeof(int32_t));
            if (lA != lB) return lA < lB;
        }
        return cA < cB;
    }

    /// @brief Stateful decoded-order comparator for
    ///        [`AdmissionMapValue`](#admissionmapvalue) sets.
    ///
    /// @details Replicates the historical string `operator<` field order —
    /// `(key, remainingArgs, standardMaxAdmissionDepth,
    /// standardMaxSecondaryNumber, flag)` — on decoded strings via
    /// valueIdVectorLess; never raw id order
    /// ([I-84](../../docs/agentic_swdd/30_invariants.md#i-84)). The
    /// comparator holds the owning LB's value interner; sets are
    /// constructed through admissionValuesAt so the pointer is always
    /// supplied.
    struct DecodedAdmissionValueLess {
        const ValueInterner* vi;
        bool operator()(const AdmissionMapValue& a, const AdmissionMapValue& b) const {
            if (valueIdVectorLess(a.key, b.key, *vi)) return true;
            if (valueIdVectorLess(b.key, a.key, *vi)) return false;
            if (valueIdVectorLess(a.remainingArgs, b.remainingArgs, *vi)) return true;
            if (valueIdVectorLess(b.remainingArgs, a.remainingArgs, *vi)) return false;
            if (a.standardMaxAdmissionDepth != b.standardMaxAdmissionDepth) {
                return a.standardMaxAdmissionDepth < b.standardMaxAdmissionDepth;
            }
            if (a.standardMaxSecondaryNumber != b.standardMaxSecondaryNumber) {
                return a.standardMaxSecondaryNumber < b.standardMaxSecondaryNumber;
            }
            return a.flag < b.flag;
        }
    };

    /// Decoded-lex ordered admission-value set (see DecodedAdmissionValueLess).
    using AdmissionValueSet = std::set<AdmissionMapValue, DecodedAdmissionValueLess>;

    // admissionValuesAt retired (D-172): admissionMap
    // is a cold blob map; use admissionRecordsAt (read snapshot) and
    // insertAdmissionValue (RMW insert), defined after Codec<AdmissionMapValue>
    // (they instantiate the record codec, declared further down this header).

    /// @brief Convert a parallel-staged string-form admission value to the
    ///        id form — retained as the byte-contract ORACLE for
    ///        `stagedToArenaBlob` (which fused this conversion with the
    ///        codec serialize on the drain's production path).
    ///
    /// @param sv The staged value (strings).
    /// @param vi The owning LB's value interner (mint side —
    ///           single-threaded call sites only).
    /// @return The id-form value; `remainingArgs` decoded-lex sorted by
    ///         construction.
    /// @see `stagedToArenaBlob` — the production fast path this oracles.
    inline AdmissionMapValue stagedToIdValue(const StagedAdmissionValue& sv,
        ValueInterner& vi)
    {
        // Sealed views mint through the span encode — same bytes, same ids,
        // no per-element std::string materialized. The key encodes in rule
        // order (the former encodeValueVector) and the args in their
        // pre-sorted lex order (the former encodeValueSetSorted iterated the
        // set in the same order), so the interner-touch sequence is
        // unchanged.
        std::vector<int32_t> keyIds;
        keyIds.reserve(sv.key.size());
        for (const SealedString& k : sv.key) {
            keyIds.push_back(vi.encode(StrSpan(k)));
        }
        std::vector<int32_t> remainingIds;
        remainingIds.reserve(sv.remainingArgsSorted.size());
        for (const SealedString& a : sv.remainingArgsSorted) {
            remainingIds.push_back(vi.encode(StrSpan(a)));
        }
        return AdmissionMapValue(
            keyIds,
            remainingIds,
            sv.standardMaxAdmissionDepth,
            sv.standardMaxSecondaryNumber,
            sv.flag);
    }

    /// @brief Stateful decoded-order comparator for
    ///        [`RejectedMapValue`](#rejectedmapvalue) sets.
    ///
    /// @details Replicates the historical string `operator<` field order —
    /// `(renamedExpression, expression, iteration, concreteConstituent,
    /// siblings, levels)` — on decoded strings; never raw id order
    /// ([I-84](../../docs/agentic_swdd/30_invariants.md#i-84)).
    struct DecodedRejectedValueLess {
        const ValueInterner* vi;
        bool operator()(const RejectedMapValue& a, const RejectedMapValue& b) const {
            if (a.renamedExpression != b.renamedExpression) {
                return valueIdLess(a.renamedExpression, b.renamedExpression, *vi);
            }
            if (a.expression != b.expression) {
                return valueIdLess(a.expression, b.expression, *vi);
            }
            if (a.iteration != b.iteration) return a.iteration < b.iteration;
            if (a.concreteConstituent != b.concreteConstituent) {
                return valueIdLess(a.concreteConstituent, b.concreteConstituent, *vi);
            }
            if (valueIdVectorLess(a.siblings, b.siblings, *vi)) return true;
            if (valueIdVectorLess(b.siblings, a.siblings, *vi)) return false;
            return a.levels < b.levels;
        }
    };

    /// Decoded-lex ordered rejected-value set (see DecodedRejectedValueLess).
    using RejectedValueSet = std::set<RejectedMapValue, DecodedRejectedValueLess>;

    // rejectedValuesAt retired (D-172): rejectedMap
    // is a cold blob map; use rejectedRecordsAt (read snapshot) and
    // insertRejectedValue (RMW insert), defined after Codec<RejectedMapValue>
    // (they instantiate the record codec, declared further down this header).

    /// @brief Stateful decoded-order comparator for
    ///        [`RejectedMapIntegrationValue`](#rejectedmapintegrationvalue)
    ///        sets.
    ///
    /// @details Replicates the historical string `operator<` field order —
    /// `(concreteConstituent, siblings, compoundExpression)` — on decoded
    /// strings; never raw id order
    /// ([I-84](../../docs/agentic_swdd/30_invariants.md#i-84)).
    struct DecodedRejectedIntegrationValueLess {
        const ValueInterner* vi;
        bool operator()(const RejectedMapIntegrationValue& a,
                        const RejectedMapIntegrationValue& b) const {
            if (a.concreteConstituent != b.concreteConstituent) {
                return valueIdLess(a.concreteConstituent, b.concreteConstituent, *vi);
            }
            if (valueIdVectorLess(a.siblings, b.siblings, *vi)) return true;
            if (valueIdVectorLess(b.siblings, a.siblings, *vi)) return false;
            return valueIdLess(a.compoundExpression, b.compoundExpression, *vi);
        }
    };

    /// Decoded-lex ordered rejected-integration value set.
    using RejectedIntegrationValueSet =
        std::set<RejectedMapIntegrationValue, DecodedRejectedIntegrationValueLess>;

    // rejectedIntegrationValuesAt retired (D-172):
    // rejectedMapIntegration is a cold blob map; use rejectedIntegrationRecordsAt
    // (read snapshot) and insertRejectedIntegrationValue (RMW insert), defined
    // after Codec<IntegrationEntry> (they instantiate the record codec, declared
    // further down this header).

    /// @brief Stateful decoded-order comparator for plain value-id sets.
    ///
    /// @details Single-field analog of the value-struct comparators —
    /// decoded-string order via valueIdLess, never raw id order
    /// ([I-84](../../docs/agentic_swdd/30_invariants.md#i-84)).
    struct DecodedIdLess {
        const ValueInterner* vi;
        bool operator()(int32_t a, int32_t b) const {
            return valueIdLess(a, b, *vi);
        }
    };

    /// Decoded-lex ordered id set (former std::set<std::string> payloads).
    using ValueIdSet = std::set<int32_t, DecodedIdLess>;

    /// @brief Pack two int32 LB-state interner ids into one `int64_t` map
    ///        key (D-135).
    ///
    /// @details Same bijective shift packing as `packOriginKey`, in the
    /// `Memory::lbStateInterner` space — key equality IS string-pair
    /// equality.
    ///
    /// @param highId Interned id of the first pair element (high 32 bits).
    /// @param lowId  Interned id of the second pair element (low 32 bits).
    /// @return The packed key.
    inline int64_t packLbStateKey(int32_t highId, int32_t lowId) {
        return packInt32Pair(highId, lowId);
    }


    /// @brief Id-form twin of the GLOBAL [`LogicalEntity`](#logicalentity),
    /// used ONLY inside the stored integration-admission instructions.
    ///
    /// @details The global `LogicalEntity` (the `compiledExpressions`
    /// registry and the disintegration machinery) is still string-typed —
    /// an OPEN heap violation to statify, NOT a sanctioned or user-approved
    /// end-state. This twin exists so `admissionMapIntegration`'s stored
    /// keys shed their strings without touching that machinery; D-1322
    /// scoped that change to the admission map alone. All
    /// five string fields live in `Memory::valueInterner`.
    struct IntLogicalEntity {
        int32_t category;
        std::vector<int32_t> elements;
        int32_t signature;
        int arity;
        int32_t definedSet;

        IntLogicalEntity()
            : category(0), elements(), signature(0), arity(0), definedSet(0) {}
    };

    /// @brief Id-form twin of [`Instruction`](#instruction) — the STORED
    /// key form of `admissionMapIntegration`'s inner map.
    ///
    /// @details The string `Instruction` remains the working/processing
    /// form (`cleanInstruction`, the `prepareIntegration` flow); encode /
    /// decode conversions happen at exactly the map touchpoints.
    struct IntInstruction {
        std::vector<IntLogicalEntity> data;
        int32_t markedGoal;

        IntInstruction() : data(), markedGoal(0) {}
    };

    /// @brief Encode a working LogicalEntity into the stored id form.
    ///
    /// @param le The string-form entity.
    /// @param vi The owning LB's value interner (mint side).
    /// @return The id-form twin.
    inline IntLogicalEntity encodeLogicalEntity(const LogicalEntity& le, ValueInterner& vi) {
        IntLogicalEntity out;
        out.category = vi.encode(le.category);
        out.elements.reserve(le.elements.size());
        for (const std::string& e : le.elements) out.elements.push_back(vi.encode(e));
        out.signature = vi.encode(le.signature);
        out.arity = le.arity;
        out.definedSet = vi.encode(le.definedSet);
        return out;
    }

    /// @brief Decode a stored id-form entity back to the working form.
    ///
    /// @param ile The id-form entity.
    /// @param vi  The owning LB's value interner (read-only).
    /// @return An owned string-form LogicalEntity.
    inline LogicalEntity decodeLogicalEntity(const IntLogicalEntity& ile, const ValueInterner& vi) {
        LogicalEntity out;
        out.category = vi.decode(ile.category);
        out.elements.reserve(ile.elements.size());
        for (const int32_t e : ile.elements) out.elements.push_back(vi.decode(e));
        out.signature = vi.decode(ile.signature);
        out.arity = ile.arity;
        out.definedSet = vi.decode(ile.definedSet);
        return out;
    }

    /// @brief Encode a working Instruction into the stored id form.
    ///
    /// @param instr The string-form instruction.
    /// @param vi    The owning LB's value interner (mint side).
    /// @return The id-form twin.
    inline IntInstruction encodeInstruction(const Instruction& instr, ValueInterner& vi) {
        IntInstruction out;
        out.data.reserve(instr.data.size());
        for (const LogicalEntity& le : instr.data) out.data.push_back(encodeLogicalEntity(le, vi));
        out.markedGoal = vi.encode(instr.markedGoal);
        return out;
    }

    /// @brief Decode a stored id-form instruction back to the working form.
    ///
    /// @param ii The id-form instruction.
    /// @param vi The owning LB's value interner (read-only).
    /// @return An owned string-form Instruction.
    inline Instruction decodeInstruction(const IntInstruction& ii, const ValueInterner& vi) {
        Instruction out;
        out.data.reserve(ii.data.size());
        for (const IntLogicalEntity& ile : ii.data) out.data.push_back(decodeLogicalEntity(ile, vi));
        out.markedGoal = vi.decode(ii.markedGoal);
        return out;
    }

    /// @brief Arena-backed transient WORKING twin of one `LogicalEntity` — a flat,
    ///        trivially-copyable record over @ref WorkInstruction's string interner.
    ///
    /// @details
    /// The integration path builds, rewrites, and reads `LogicalEntity` /
    /// `Instruction` objects every burst. On the malloc heap each `LogicalEntity`
    /// costs three `std::string` plus a `std::vector<std::string>`; this record
    /// replaces all four with interned ids into the owning @ref WorkInstruction's
    /// `ColdHashSet<BytesKeyStore>`, so the working form rides a per-slot
    /// `genScratchArenas` arena and allocates no per-element heap — the
    /// disintegration accumulator idiom (`CollectedRec` / `RejRec`).
    ///
    /// The five `LogicalEntity` fields map directly: `catId` / `sigId` / `dsId`
    /// are interner ids of `category` / `signature` / `definedSet`; `arity` is the
    /// raw int; the `elements` list is the half-open run
    /// `[elemsStart, elemsStart + elemsCount)` into the owner's `elemCol` column
    /// (the `RejRec` sibling-run pattern), each entry an interner id.
    ///
    /// @see WorkInstruction — owns the records, the interner, and the element column.
    /// @see IntLogicalEntity — the STORED (cold-blob) id twin; this is the WORKING twin.
    /// @see LogicalEntity — the heap form this replaces on the integration path.
    struct WorkLogicalEntity {
        int32_t catId;
        int32_t sigId;
        int32_t dsId;
        int32_t arity;
        int32_t elemsStart;
        int32_t elemsCount;
    };

    /// @brief Arena-backed transient WORKING twin of one `Instruction` — the
    ///        integration path's per-burst working form, off the malloc heap.
    ///
    /// @details
    /// Replaces the heap `Instruction` (`std::vector<LogicalEntity> data` +
    /// `std::string markedGoal`) with three page-tier containers on ONE per-slot
    /// `genScratchArenas` arena: a `ColdHashSet<BytesKeyStore>` that interns every
    /// string field, a `PagedVector<WorkLogicalEntity>` of entity records, and a
    /// `PagedVector<int32_t>` element-id column the records run into. No
    /// `std::vector`/`std::string` node heap — the disintegration accumulator
    /// idiom (`CollectedArena` / `RejectionStore`).
    ///
    /// Build is append-only: capture the element-run start with @ref elemMark,
    /// push the entity's elements with @ref addElement, then close the entity with
    /// @ref commitEntity; @ref setMarkedGoal records the goal. The integration
    /// path's one rewrite (`prepareIntegrationCore2`'s count-preserving field
    /// substitution) rebuilds into a FRESH `WorkInstruction` rather than mutating
    /// in place — the records are read-only after commit. Reads are zero-copy spans
    /// into the interner, valid while the arena is resident (the per-task lifetime).
    ///
    /// @invariant Lives within a single integration call on its slot's arena;
    ///            every span it returns points into that arena and dies at the
    ///            per-task `releaseAll`.
    /// @see WorkLogicalEntity — the per-entity record.
    /// @see encodeWorkInstruction / loadFromIntInstruction — the IntInstruction boundary.
    struct WorkInstruction {
        LbArena* arena;
        DirtyState dirty = DirtyState::Clean;
        ColdHashSet<BytesKeyStore> strings;        ///< all string fields (page tier)
        PagedVector<WorkLogicalEntity> data;       ///< entity records (page tier)
        PagedVector<int32_t> elemCol;              ///< flattened element-id runs (page tier)
        int32_t markedGoalId = 0;                  ///< interner id of markedGoal (0 = unset)

        /// @brief Bind to a per-slot scratch arena; all three containers ride its
        ///        PAGE tier.
        /// @param a The arena; outlives this working instruction.
        explicit WorkInstruction(LbArena* a)
            : arena(a), strings(a, &dirty), data(a, &dirty), elemCol(a, &dirty) {}
        WorkInstruction(const WorkInstruction&) = delete;
        WorkInstruction& operator=(const WorkInstruction&) = delete;

        /// @brief Capture the start of a new entity's element run.
        /// @return The current element-column size (pass to @ref commitEntity).
        int32_t elemMark() const { return elemCol.size(); }

        /// @brief Append one element to the current entity's run (interned on insert).
        /// @param elem The element expression bytes.
        void addElement(const StrSpan& elem) { elemCol.push_back(strings.mint(elem)); }

        /// @brief Close an entity, recording its fields and element run.
        /// @param category   The entity category.
        /// @param signature  The entity signature.
        /// @param definedSet The defining-set label.
        /// @param arity      The entity arity.
        /// @param elemStart  The run start captured by @ref elemMark before the
        ///                   entity's @ref addElement calls.
        void commitEntity(const StrSpan& category, const StrSpan& signature,
                          const StrSpan& definedSet, int32_t arity, int32_t elemStart) {
            data.push_back(WorkLogicalEntity{
                strings.mint(category), strings.mint(signature), strings.mint(definedSet),
                arity, elemStart, elemCol.size() - elemStart });
        }

        /// @brief Record the instruction's marked goal.
        /// @param goal The goal expression bytes.
        void setMarkedGoal(const StrSpan& goal) { markedGoalId = strings.mint(goal); }

        /// @brief Number of entities. @return The entity count.
        int32_t entityCount() const { return data.size(); }
        /// @brief Decode entity @p i's category. @param i Entity index. @return Span.
        StrSpan category(int32_t i) const { return strings.keyAt(data[i].catId); }
        /// @brief Decode entity @p i's signature. @param i Entity index. @return Span.
        StrSpan signature(int32_t i) const { return strings.keyAt(data[i].sigId); }
        /// @brief Decode entity @p i's defining set. @param i Entity index. @return Span.
        StrSpan definedSet(int32_t i) const { return strings.keyAt(data[i].dsId); }
        /// @brief Entity @p i's arity. @param i Entity index. @return The arity.
        int32_t arity(int32_t i) const { return data[i].arity; }
        /// @brief Entity @p i's element count. @param i Entity index. @return The count.
        int32_t elemCount(int32_t i) const { return data[i].elemsCount; }
        /// @brief Decode the @p j-th element of entity @p i.
        /// @param i Entity index. @param j Element index within the entity.
        /// @return The element span.
        StrSpan elemAt(int32_t i, int32_t j) const {
            return strings.keyAt(elemCol[data[i].elemsStart + j]);
        }
        /// @brief Decode the marked goal (empty span when unset).
        /// @return The goal span; empty when @ref setMarkedGoal was never called.
        StrSpan markedGoal() const {
            return markedGoalId == 0 ? StrSpan() : strings.keyAt(markedGoalId);
        }

        /// @brief Materialize entity @p i as an owned heap `LogicalEntity` — the
        ///        decode boundary for code that still consumes a `LogicalEntity`.
        ///
        /// @details
        /// Copies the entity's five fields out of the arena into a `LogicalEntity`.
        /// This is the per-iteration boundary used while a consumer's body still
        /// reads `LogicalEntity`; the consumer materializes ONE entity at a time
        /// rather than holding the whole `std::vector<LogicalEntity>` resident, so
        /// the peak heap is one entity, not the count. As consumers move to read
        /// the span accessors directly this boundary retires.
        ///
        /// @param i Entity index (`0 <= i < entityCount()`).
        /// @return An owned `LogicalEntity` copy of record @p i.
        LogicalEntity entityAt(int32_t i) const {
            LogicalEntity le;
            const StrSpan c = category(i);
            le.category = std::string(c.ptr, static_cast<std::size_t>(c.len));
            const StrSpan s = signature(i);
            le.signature = std::string(s.ptr, static_cast<std::size_t>(s.len));
            const StrSpan d = definedSet(i);
            le.definedSet = std::string(d.ptr, static_cast<std::size_t>(d.len));
            le.arity = arity(i);
            const int32_t ec = elemCount(i);
            le.elements.reserve(static_cast<std::size_t>(ec));
            for (int32_t j = 0; j < ec; ++j) {
                const StrSpan e = elemAt(i, j);
                le.elements.push_back(std::string(e.ptr, static_cast<std::size_t>(e.len)));
            }
            return le;
        }
    };

    /// @brief Encode a working @ref WorkInstruction into the STORED `IntInstruction`
    ///        id form — the cold-boundary twin of `encodeInstruction`.
    ///
    /// @details
    /// Walks the entity records in order, re-interning every field span through the
    /// owning LB's @p vi (the SAME interner `encodeInstruction` mints into, so a
    /// `WorkInstruction` loaded from an `IntInstruction` re-encodes to byte-identical
    /// ids). Field order matches `encodeLogicalEntity` (`category`, `elements`,
    /// `signature`, `arity`, `definedSet`) so the result is indistinguishable from
    /// the heap path. The returned `IntInstruction` is a heap boundary handed to
    /// the cold admission store — an OPEN violation to statify, not a sanctioned
    /// heap boundary.
    ///
    /// @param wi The working instruction (resident; spans valid).
    /// @param vi The owning LB's value interner (mint side).
    /// @return The stored id-form instruction.
    /// @see loadFromIntInstruction — the inverse. @see encodeInstruction — the heap twin.
    inline IntInstruction encodeWorkInstruction(const WorkInstruction& wi, ValueInterner& vi) {
        IntInstruction out;
        const int32_t n = wi.entityCount();
        out.data.reserve(static_cast<std::size_t>(n));
        for (int32_t i = 0; i < n; ++i) {
            IntLogicalEntity ile;
            ile.category = vi.encode(wi.category(i));
            const int32_t ec = wi.elemCount(i);
            ile.elements.reserve(static_cast<std::size_t>(ec));
            for (int32_t j = 0; j < ec; ++j) ile.elements.push_back(vi.encode(wi.elemAt(i, j)));
            ile.signature = vi.encode(wi.signature(i));
            ile.arity = wi.arity(i);
            ile.definedSet = vi.encode(wi.definedSet(i));
            out.data.push_back(std::move(ile));
        }
        out.markedGoal = vi.encode(wi.markedGoal());
        return out;
    }

    /// @brief Rebuild a working @ref WorkInstruction from a STORED `IntInstruction`
    ///        — the heap-free twin of `decodeInstruction`.
    ///
    /// @details
    /// Reads @p ii (the boundary id form, typically a cold-store snapshot) and
    /// mints every field's bytes into @p wi's own arena interner via
    /// `vi.decodeView` (a zero-copy span over the resident cold bytes, copied on
    /// mint — no `std::string` materialized). The I-3 copy-before-nested-mint
    /// discipline is moot here: the decode reads @p vi while the mint writes the
    /// DISTINCT @p wi interner, so no span is invalidated by its own consumption.
    /// Element ids land in the entity's run via @ref WorkInstruction::addElement
    /// before @ref WorkInstruction::commitEntity closes it. @p wi must be freshly
    /// constructed (the load appends).
    ///
    /// @param wi The destination working instruction (empty; appended in place).
    /// @param ii The stored id-form instruction (the boundary input).
    /// @param vi The owning LB's value interner (read side).
    /// @see encodeWorkInstruction — the inverse. @see decodeInstruction — the heap twin.
    inline void loadFromIntInstruction(WorkInstruction& wi, const IntInstruction& ii,
                                       const ValueInterner& vi) {
        for (const IntLogicalEntity& ile : ii.data) {
            const int32_t start = wi.elemMark();
            for (const int32_t eid : ile.elements) wi.addElement(vi.decodeView(eid));
            wi.commitEntity(vi.decodeView(ile.category), vi.decodeView(ile.signature),
                            vi.decodeView(ile.definedSet), ile.arity, start);
        }
        wi.setMarkedGoal(vi.decodeView(ii.markedGoal));
    }

    /// @brief WorkInstruction twin of the heap `cleanInstruction` — the
    ///        cascade-erase that prunes an instruction to the entities NOT
    ///        reachable-and-orphaned from a target signature, rebuilt onto a
    ///        fresh arena.
    ///
    /// @details
    /// Mirrors the heap `cleanInstruction` exactly, on the append-only
    /// `WorkInstruction` (records are read-only after commit, so "erase" is a
    /// rebuild): find the first surviving entity whose signature equals the target
    /// and drop it; if it is a container (`"existence"` / `"and"`), drop each of its
    /// elements that is no longer referenced by another surviving container,
    /// recursing on that element's signature. The surviving entities are committed
    /// to @p dst in their original order and @p dst inherits @p src's marked goal.
    ///
    /// Heap-free: a stack `removed[]` flag array tracks the dropped entities and
    /// every comparison is a span compare (`equalSpans`) over @p src's arena bytes —
    /// no `LogicalEntity`, no `std::string`. The target signature arrives as a span
    /// too, so a caller holding an arena `ScratchString` or a `decodeView` slice
    /// feeds the bytes with no throwaway heap string. The byte-for-byte result is
    /// the same instruction the heap path produced (verified by
    /// `clean_instruction_work_matches_heap` and, against the `std::string`
    /// forwarder, `clean_instruction_work_span_matches_string`).
    ///
    /// @param src        The source working instruction (read-only).
    /// @param dst        The destination working instruction (empty; survivors are appended).
    /// @param expression The target signature bytes whose entity (and its
    ///                   now-orphaned sub-entities) are pruned.
    /// @see WorkInstruction, encodeWorkInstruction,
    ///      cleanInstructionWork(const WorkInstruction&, WorkInstruction&, const std::string&)
    inline void cleanInstructionWork(const WorkInstruction& src, WorkInstruction& dst,
        const StrSpan& expression) {
        const int32_t n = src.entityCount();
        constexpr int32_t kCleanCap = 1024;
        assert(n <= kCleanCap
            && "cleanInstructionWork: entity count exceeds the stack cap");
        bool removed[kCleanCap] = { false };

        const StrSpan existenceTag("existence", 9);
        const StrSpan andTag("and", 3);
        const auto isContainer = [&](const StrSpan& cat) {
            return equalSpans(cat, existenceTag) || equalSpans(cat, andTag);
        };

        // Drop the first surviving entity whose signature matches `target`; for a
        // container, drop each element no longer referenced by another surviving
        // container, recursing.
        const auto eraseMatching = [&](const StrSpan& target, const auto& self) -> void {
            int32_t hit = -1;
            for (int32_t i = 0; i < n; ++i) {
                if (!removed[i] && equalSpans(src.signature(i), target)) { hit = i; break; }
            }
            if (hit < 0) return;
            removed[hit] = true;
            if (!isContainer(src.category(hit))) return;
            const int32_t ec = src.elemCount(hit);
            for (int32_t j = 0; j < ec; ++j) {
                const StrSpan elem = src.elemAt(hit, j);
                bool usedElsewhere = false;
                for (int32_t k = 0; k < n && !usedElsewhere; ++k) {
                    if (removed[k] || !isContainer(src.category(k))) continue;
                    const int32_t kec = src.elemCount(k);
                    for (int32_t m = 0; m < kec; ++m) {
                        if (equalSpans(src.elemAt(k, m), elem)) { usedElsewhere = true; break; }
                    }
                }
                if (!usedElsewhere) self(elem, self);
            }
        };
        eraseMatching(expression, eraseMatching);

        dst.setMarkedGoal(src.markedGoal());
        for (int32_t i = 0; i < n; ++i) {
            if (removed[i]) continue;
            const int32_t em = dst.elemMark();
            const int32_t ec = src.elemCount(i);
            for (int32_t j = 0; j < ec; ++j) dst.addElement(src.elemAt(i, j));
            dst.commitEntity(src.category(i), src.signature(i), src.definedSet(i),
                src.arity(i), em);
        }
    }

    /// @brief `std::string` forwarder for @ref cleanInstructionWork(const
    ///        WorkInstruction&, WorkInstruction&, const StrSpan&).
    ///
    /// @details
    /// Preserves the original `const std::string&` call site while the span
    /// overload carries the logic; `StrSpan(expression)` is a zero-copy view of
    /// the string's bytes, so the forward is byte-identical to calling the span
    /// overload directly (both prune the same target signature). Kept so existing
    /// heap-string callers compile unchanged; new callers pass a span / arena
    /// `ScratchString` to the span overload with no throwaway heap string.
    ///
    /// @param src        The source working instruction (read-only).
    /// @param dst        The destination working instruction (empty; survivors are appended).
    /// @param expression The target signature string whose entity (and its
    ///                   now-orphaned sub-entities) are pruned.
    /// @see cleanInstructionWork(const WorkInstruction&, WorkInstruction&, const StrSpan&)
    inline void cleanInstructionWork(const WorkInstruction& src, WorkInstruction& dst,
        const std::string& expression) {
        cleanInstructionWork(src, dst, StrSpan(expression));
    }

    /// @brief Stateful decoded-order comparator for stored instructions.
    ///
    /// @details Replicates the historical orders exactly:
    /// `Instruction::(data, markedGoal)` with
    /// `LogicalEntity::(category, signature, elements, arity, definedSet)`
    /// per element and vector size as the lexicographic tiebreak — on
    /// decoded strings, never raw id order
    /// ([I-84](../../docs/agentic_swdd/30_invariants.md#i-84)).
    struct DecodedInstructionLess {
        const ValueInterner* vi;

        // -1 / 0 / +1 decoded comparison of one entity pair.
        int compareEntity(const IntLogicalEntity& a, const IntLogicalEntity& b) const {
            if (a.category != b.category) {
                return valueIdLess(a.category, b.category, *vi) ? -1 : 1;
            }
            if (a.signature != b.signature) {
                return valueIdLess(a.signature, b.signature, *vi) ? -1 : 1;
            }
            if (valueIdVectorLess(a.elements, b.elements, *vi)) return -1;
            if (valueIdVectorLess(b.elements, a.elements, *vi)) return 1;
            if (a.arity != b.arity) return a.arity < b.arity ? -1 : 1;
            if (a.definedSet != b.definedSet) {
                return valueIdLess(a.definedSet, b.definedSet, *vi) ? -1 : 1;
            }
            return 0;
        }

        bool operator()(const IntInstruction& a, const IntInstruction& b) const {
            const std::size_t n = a.data.size() < b.data.size() ? a.data.size() : b.data.size();
            for (std::size_t i = 0; i < n; ++i) {
                const int c = compareEntity(a.data[i], b.data[i]);
                if (c != 0) return c < 0;
            }
            if (a.data.size() != b.data.size()) return a.data.size() < b.data.size();
            return valueIdLess(a.markedGoal, b.markedGoal, *vi);
        }
    };

    /// Stored inner map of admissionMapIntegration (decoded-ordered).
    using IntegrationEntryMap = std::map<IntInstruction, ValueIdSet, DecodedInstructionLess>;

    /// @brief One flattened entry of `admissionMapIntegration`'s inner instruction
    ///        map — a `(IntInstruction, ValueIdSet)` pair as one blob record.
    ///
    /// @details
    /// `admissionMapIntegration` is `packedKey -> std::map<IntInstruction,
    /// ValueIdSet>`. On the cold blob map each key's run is the inner map's entries
    /// in their decoded order; one `IntegrationEntry` is one such entry. The
    /// decoded-order comparators (`DecodedInstructionLess` / `DecodedIdLess`) order
    /// the run at the read-modify-write boundary, never inside the codec. The
    /// struct lives here (before `HashMemory`) so the cold member's
    /// `TypedColdBlobMap<int32_t, IntegrationEntry>` sees a complete record type;
    /// the record `Codec<IntegrationEntry>` is defined further down with the other
    /// codecs.
    ///
    /// @see `IntInstruction`, `Codec<IntegrationEntry>`.
    struct IntegrationEntry {
        /// @brief The inner-map key (the stored id-form instruction).
        IntInstruction instruction;
        /// @brief The inner-map value set, flattened to a vector in decoded order.
        std::vector<int32_t> valueIds;
    };

    // integrationEntryAt retired (D-172):
    // admissionMapIntegration is a cold blob map; use admissionIntegrationRecordsAt
    // (read snapshot, returns an owned IntegrationEntryMap) and
    // insertAdmissionIntegrationValue (RMW insert), defined after
    // Codec<IntegrationEntry>. payloadAt below still serves the decoded
    // snapshot's inner-map create-if-absent.

    /// @brief Find-or-emplace one instruction's payload set inside an
    ///        instruction map — supplies the payload comparator (the
    ///        former map operator[] create-if-absent semantics).
    ///
    /// @param em    The instruction map.
    /// @param instr The stored instruction key.
    /// @param vi    The owning LB's value interner (comparator state).
    /// @return The instruction's payload id set.
    inline ValueIdSet& payloadAt(IntegrationEntryMap& em,
        const IntInstruction& instr, const ValueInterner& vi)
    {
        auto it = em.find(instr);
        if (it == em.end()) {
            it = em.emplace(instr, ValueIdSet(DecodedIdLess{ &vi })).first;
        }
        return it->second;
    }

    /// @brief Mint the packed admission/rejected key of a template at a
    ///        scope (write half).
    ///
    /// @details Encodes the template into @p ti (minting on first sight) and
    /// the validity into @p nm (idempotent — registration scopes are always
    /// already interned). Write sites only; probe paths use
    /// lookupTemplateKey.
    ///
    /// @param ti           The owning LB's template interner.
    /// @param nm           The owning LB's NameMap.
    /// @param templ        The template string (marker / u_ / bare form).
    /// @param validityName The scope.
    /// @return `packStatementKey(templateId, validityId)`.
    /// @see TemplateInterner, lookupTemplateKey, decodeTemplateKey.
    inline int32_t mintTemplateKey(TemplateInterner& ti, NameMap& nm,
        const std::string& templ, const std::string& validityName)
    {
        const int16_t tid = ti.encode(templ);
        const int16_t vid = nm.encode(validityName);
        return packStatementKey(tid, vid);
    }

    /// @brief Span overload of the packed admission/rejected key mint (write
    ///        half).
    ///
    /// @details
    /// Byte-identical key to `mintTemplateKey(..., const std::string&, const
    /// std::string&)` — both ids are a pure function of the interned bytes, so
    /// `packStatementKey(ti.encode(templ), nm.encode(validityName))` matches
    /// whether the arguments arrive as spans or strings. Interns the template
    /// into @p ti (span-form `encode`) and the validity into @p nm (span-form
    /// `encode`, idempotent — registration scopes are always already interned).
    /// Lets the integration/admission key writers mint from an arena
    /// `ScratchString` or a sibling `decodeView` span with no throwaway heap
    /// string. Write sites only; probe paths use `lookupTemplateKey`.
    ///
    /// @param ti           The owning LB's template interner.
    /// @param nm           The owning LB's NameMap.
    /// @param templ        The template bytes (marker / u_ / bare form).
    /// @param validityName The scope bytes.
    /// @return `packStatementKey(templateId, validityId)`.
    /// @see mintTemplateKey(..., const std::string&, const std::string&) — the
    ///      canonical overload; lookupTemplateKey; decodeTemplateKeyView.
    inline int32_t mintTemplateKey(TemplateInterner& ti, NameMap& nm,
        const StrSpan& templ, const StrSpan& validityName)
    {
        const int16_t tid = ti.encode(templ);
        const int16_t vid = nm.encode(validityName);
        return packStatementKey(tid, vid);
    }

    /// @brief Non-minting packed-key probe for the admission/rejected
    ///        containers (read half).
    ///
    /// @details A template or validity never interned cannot key any
    /// registered entry, so a lookup miss on either half is a definitive
    /// container miss — a defined result, not a fallback. Safe on read
    /// paths including the phase-2 parallel staging gate (both lookups are
    /// `const`).
    ///
    /// @param ti           The owning LB's template interner (read-only).
    /// @param nm           The owning LB's NameMap (read-only).
    /// @param templ        The template string to probe.
    /// @param validityName The scope.
    /// @param outKey       The packed key, valid only when `true` returned.
    /// @return `true` when both halves are interned (the key MAY be in a
    ///         container); `false` = definitive miss everywhere.
    inline bool lookupTemplateKey(const TemplateInterner& ti, const NameMap& nm,
        const std::string& templ, const std::string& validityName, int32_t& outKey)
    {
        const int16_t tid = ti.lookup(templ);
        if (tid == 0) return false;
        const int16_t vid = nm.lookup(validityName);
        if (vid == 0) return false;
        outKey = packStatementKey(tid, vid);
        return true;
    }

    /// @brief Span overload of the non-minting packed-key probe (read half).
    ///
    /// @details
    /// Byte-identical verdict + key to `lookupTemplateKey(..., const
    /// std::string&, const std::string&)`: a template or validity never
    /// interned is a definitive container miss, so a miss on either half
    /// returns `false` without touching @p outKey. Uses the span-form
    /// `TemplateInterner::lookup` / `NameMap::lookup`, both `const`, so it is
    /// safe on the read paths (including the phase-2 parallel staging gate)
    /// and lets a caller feed a `decodeView` span or arena `ScratchString`
    /// straight in.
    ///
    /// @param ti           The owning LB's template interner (read-only).
    /// @param nm           The owning LB's NameMap (read-only).
    /// @param templ        The template bytes to probe.
    /// @param validityName The scope bytes.
    /// @param outKey       The packed key, valid only when `true` returned.
    /// @return `true` when both halves are interned; `false` = definitive miss.
    /// @see lookupTemplateKey(..., const std::string&, const std::string&) —
    ///      the canonical overload; mintTemplateKey.
    inline bool lookupTemplateKey(const TemplateInterner& ti, const NameMap& nm,
        const StrSpan& templ, const StrSpan& validityName, int32_t& outKey)
    {
        const int16_t tid = ti.lookup(templ);
        if (tid == 0) return false;
        const int16_t vid = nm.lookup(validityName);
        if (vid == 0) return false;
        outKey = packStatementKey(tid, vid);
        return true;
    }

    /// @brief Decode a packed admission/rejected key back to its
    ///        (template, validity) string pair.
    ///
    /// @details Owned copies — dump derivations and order-sensitive walks
    /// sort and hold these across further work (I-3 discipline).
    ///
    /// @param packedKey Packed key minted by mintTemplateKey.
    /// @param ti        The owning LB's template interner.
    /// @param nm        The owning LB's NameMap.
    /// @return `(template, validityName)` as owned strings.
    inline std::pair<std::string, std::string> decodeTemplateKey(int32_t packedKey,
        const TemplateInterner& ti, const NameMap& nm)
    {
        const int16_t tid = static_cast<int16_t>((static_cast<uint32_t>(packedKey) >> 16) & 0xFFFF);
        const int16_t vid = static_cast<int16_t>(static_cast<uint32_t>(packedKey) & 0xFFFF);
        return std::make_pair(std::string(ti.decode(tid)), std::string(nm.decode(vid)));
    }

    /// @brief Zero-copy twin of `decodeTemplateKey` — split a packed key into
    ///        (template, validity) spans over the interners' cold bytes.
    ///
    /// @details
    /// Same unpack as `decodeTemplateKey` (template id = high half via the
    /// dedicated `templateInterner`; validity id = low half via the NameMap),
    /// but writes each half into a caller `StrSpan&` out-param via `decodeView`
    /// rather than returning an owning `decode` copy (09b row 87: no by-value
    /// heap in the signature). The equivalence-class apply tree's key-rebuild
    /// walks (`decodeTemplateKeyView(pk, …, tKey, tVal)` then
    /// `arena.mint(tKey)`) consume the spans by minting them straight into
    /// their per-slot arena interner — a DIFFERENT interner than @p ti / @p nm —
    /// so no source-interner mint intervenes and no heap pair is built. The
    /// spans are stable only while the LB is resident and both interners are
    /// un-mutated (I-3) — copy before any mint into @p ti / @p nm.
    ///
    /// @param packedKey   Packed key minted by `mintTemplateKey`.
    /// @param ti          The owning LB's template interner.
    /// @param nm          The owning LB's NameMap.
    /// @param templateOut Receives the template span over @p ti's cold bytes.
    /// @param validityOut Receives the validity-name span over @p nm's cold bytes.
    /// @see decodeTemplateKey — the owning-copy sibling; mintTemplateKey.
    inline void decodeTemplateKeyView(int32_t packedKey,
        const TemplateInterner& ti, const NameMap& nm,
        StrSpan& templateOut, StrSpan& validityOut)
    {
        const int16_t tid = static_cast<int16_t>((static_cast<uint32_t>(packedKey) >> 16) & 0xFFFF);
        const int16_t vid = static_cast<int16_t>(static_cast<uint32_t>(packedKey) & 0xFFFF);
        templateOut = ti.decodeView(tid);
        validityOut = nm.decodeView(vid);
    }

    /// @brief Dedupe + emit for all `generateEncodedRequests*` variants.
    ///
    /// @details
    /// The static request pipeline produces a stream of `StaticRequest` values
    /// destined for the hash memory lookup. Many of those requests turn out to
    /// duplicate previously emitted ones (same expressions in same order under
    /// the same scope). The emitter keeps a `seen` set of `RequestKey` (the
    /// packed `(originalId, validityId)` tuple per expression) and rejects
    /// duplicates before they ever reach the output buffer.
    ///
    /// `emit` flow:
    /// 1. Build a `RequestKey` from the input expressions' packed IDs.
    /// 2. Insert into `seen`. If the entry already existed, return early.
    /// 3. Otherwise copy each `IntEncodedExpr` into the per-thread arena and
    ///    record pointers into the output buffer slot.
    /// 4. Compute and store `maxIteration` across the bundle.
    /// 5. Increment `outCount`.
    ///
    /// The arena copy is what guarantees pointer stability across the rest of
    /// the request stream — without it, the input expression pointers might
    /// dangle when the caller's container reallocates.
    ///
    /// @see [`StaticRequest`](#staticrequest) — output element type.
    /// @see [`TypedArena`](#typedarena) — arena that backs `intExprs` storage.
    template <typename Consumer>
    struct StaticRequestEmitter {
        struct RequestKey {
            int32_t packed[ExecutionParameters::MAX_EXPRESSIONS];
            int16_t count;
            bool operator<(const RequestKey& rhs) const {
                if (count != rhs.count) return count < rhs.count;
                return std::memcmp(packed, rhs.packed, count * sizeof(int32_t)) < 0;
            }
        };
        // Per-batch dedup: the emitted-request key set on this slot's gen
        // scratch arena (the interner pattern -- ColdHashSet keyed by the packed
        // bytes), so no per-emit heap node. Owns its throwaway dirty flag,
        // declared before `seen` so the member init can take its address;
        // scratch never deloads, so the flag is never read.
        DirtyState seenDirty = DirtyState::Clean;
        ColdHashSet<BytesKeyStore> seen;
        // The IntEncodedExpr copies + the dedup set both ride this slot's gen
        // scratch arena (no TypedArena operator-new): the copies on the
        // byte-bump tier, persistent per task (a StaticRequest points at them
        // through consume), freed by the per-task releaseAll. They sit below
        // the per-request combinedLevels mark in checkLocalEncodedMemoryStatic,
        // so that function's popTo never touches them.
        ScratchArena& genArena;
        Consumer& consumer;

        StaticRequestEmitter(ScratchArena& gen, Consumer& c)
            : seen(&gen, &seenDirty), genArena(gen), consumer(c) {}

        // Dedupe, build one arena-stable StaticRequest, and hand it to the
        // consumer (which either stores it or checks it inline). Returns the
        // consumer's keep-going verdict so the generator loops can break on a
        // false: false = stop generating (cap reached / early-exit), true =
        // keep going. `consumer.canAccept()` is checked first (the pre-stream
        // `outCount >= maxOut` cap guard), then dedup, then build, then
        // `consumer.consume(req)`.
        bool emit(const IntEncodedExpr* const* exprs, int16_t count,
                  const IntNormalizedKey& nk) {
            if (!consumer.canAccept()) return false;
            RequestKey dk;
            dk.count = count;
            for (int16_t i = 0; i < count; ++i)
                dk.packed[i] = packStatementKey(exprs[i]->originalId, exprs[i]->validityId);
            // Dedup on the packed bytes: count int32s uniquely encode
            // (count, packed) -- distinct counts give distinct byte lengths.
            // intern copies the bytes into the cold key column; an unchanged
            // count() means the key was already present, so skip (the std::set
            // insert().second == false twin). No heap.
            const StrSpan dkSpan(reinterpret_cast<const char*>(dk.packed),
                                 static_cast<int32_t>(count * sizeof(int32_t)));
            const int32_t beforeCount = seen.count();
            seen.intern(dkSpan);
            if (seen.count() == beforeCount) return true;

            StaticRequest req{};
            req.count = count;
            req.maxIteration = -1;
            for (int16_t i = 0; i < count; ++i) {
                IntEncodedExpr* copy = reinterpret_cast<IntEncodedExpr*>(
                    genArena.resolve(genArena.alloc(
                        static_cast<int32_t>(sizeof(IntEncodedExpr)),
                        static_cast<int32_t>(alignof(IntEncodedExpr)))));
                *copy = *exprs[i];
                req.intExprs[i] = copy;
                if (copy->maxIteration > req.maxIteration)
                    req.maxIteration = copy->maxIteration;
            }
            req.normalizedKey = nk;
            return consumer.consume(req);
        }
    };


    /// @brief One candidate produced by the *grow* phase of the static
    /// `makeMandatory` merge pass.
    ///
    /// @details
    /// The grow phase iterates through pre-encoded statements building partial
    /// match-candidate sets. Each surviving partial candidate is recorded as a
    /// `BaseCandidate` (its statement indices in `allIdx[count]` plus the scope
    /// id). The merge phase then folds these candidates into final
    /// `StaticRequest`s. Plain index storage; no ordering or hashing.
    ///
    /// @see `memory.cpp::makeMandatoryEncodedStatementLists2Static` — grow phase.
    struct BaseCandidate {
        int16_t allIdx[ExecutionParameters::MAX_EXPRESSIONS];
        int16_t count;
        int16_t validityId;
    };

    /// @brief One stump of the LB split's second (expression) dimension.
    ///
    /// @details
    /// A straggler's producer returns a list of these for `proveKernel` to deal
    /// into expression buckets. Every request a bucket generates contains one
    /// stump: the request generator attaches the stump's expressions to each
    /// growing candidate for the owner-set probes and materialises the union
    /// into a `BaseCandidate` only where the record probe passes.
    ///
    /// A regular stump is both probed on its own and used as a depth-first-search
    /// seed. A terminal stump preserves a recordable candidate from a shallower
    /// producer level that was replaced by its children: it is probed on its own
    /// against the request batch's actual hash memory, then stops. Its children
    /// remain in the producer frontier and cover every larger base candidate.
    ///
    /// `allIdx[0..count)` are indices into `Memory::intEncodedStatements` — not
    /// into any filtered list, whose contents differ per request batch — held in
    /// the generator's own candidate order: ascending decoded name, ties broken
    /// by ascending statement index. The expressions carry their own scope, so
    /// no validity id is stored; the consumer folds them with `deeperOf`.
    ///
    /// Trivially copyable: it rides the producing task's `SealedPageSet` record
    /// chain and is read back after the pool join
    /// ([I-135](../../docs/agentic_swdd/30_invariants.md#i-135)).
    ///
    /// @see `ExpressionAnalyzer::produceExpressionStumps` — the producer.
    /// @see `ExpressionAnalyzer::generateEncodedRequestsStatic` — the consumer.
    struct ExpressionStump {
        int16_t allIdx[ExecutionParameters::MAX_EXPRESSIONS];
        int16_t count;
        uint8_t terminalOnly;
    };

    /// @brief What one phase-2 sub-part knows about the LB's stump split.
    ///
    /// @details
    /// `stumps[0..count)` is this sub-part's **bucket**: the search runs once per
    /// stump in it, and every request the sub-part emits contains one of them.
    ///
    /// A bucket, rather than a single stump, is what bounds the fan-out. A
    /// rule-part returns one stump per expression surviving its request filter —
    /// hundreds — and one sub-part per stump multiplied against the rule dimension
    /// into thousands of parts, each paying the same fixed setup: the statement
    /// filter and the obligatory-stump builders, rebuilt from scratch, five
    /// batches over. Dealing the stumps into `total` buckets pays that setup once
    /// per bucket. It also lets the emitter's dedup collapse a request that two
    /// stumps of the same bucket both reach, which two separate sub-parts would
    /// each have fired.
    ///
    /// `ordinal` and `total` place the sub-part among its siblings. Besides sizing
    /// the deal they do one job: the generator's seed phase emits requests that are
    /// nothing but the obligatory stump, which contain no split stump at all and
    /// would otherwise be emitted once per sub-part. They are dealt out by
    /// obligatory-stump index, so each is emitted exactly once. The stump dimension
    /// is orthogonal to the rule dimension, which the thread-locals
    /// `g_splitProcessID` / `g_splitCount` carry and `partitionAccepts` reads.
    ///
    /// A default-constructed value means *no stump split*: the generator is then
    /// line-for-line the unsplit one.
    ///
    /// @invariant `(count == 0) == (stumps == nullptr)`.
    /// @invariant `count > 0` implies `total >= 1` and `0 <= ordinal < total`.
    /// @see `ExpressionAnalyzer::produceExpressionStumps`,
    ///      `ExpressionAnalyzer::generateEncodedRequestsStatic`.
    struct SplitStumpRef {
        const ExpressionStump* stumps = nullptr;
        int16_t count = 0;
        int16_t ordinal = 0;
        int16_t total = 0;
    };

    /// @brief Per-thread bundle of typed arenas backing one thread's slice of
    /// the static request pipeline.
    ///
    /// @details
    /// Four sub-arenas, sized for the typical worst case observed on a Gauss
    /// batch:
    ///
    /// - `requests`     (capacity 4096) — `StaticRequest` storage.
    /// - `encodedExprs` (capacity 2048) — `IntEncodedExpr` copies.
    /// - `pairs`        (capacity 8192) — `Stump`s.
    /// - `indices`      (capacity 4096) — `int16_t` index arrays.
    ///
    /// `reset()` rewinds all four arenas to zero use; called once per LB-
    /// iteration boundary so storage is reused without going through the
    /// allocator. The destructor releases the underlying buffers.
    ///
    /// @see [`TypedArena`](#typedarena) — the per-arena type.
    /// @see [I-28](../../docs/agentic_swdd/30_invariants.md#i-28) — cross-LB writes during
    ///      `proveKernel`'s parallel phase are forbidden; per-thread arenas are
    ///      one of the mechanisms that keep that invariant true.
    struct ThreadArenas {
        TypedArena<StaticRequest>   requests;
        TypedArena<IntEncodedExpr>  encodedExprs;
        TypedArena<Stump>           pairs;
        TypedArena<int16_t>         indices;

        ThreadArenas()
            : requests(4096),
              encodedExprs(2048),
              pairs(8192),
              indices(4096)
        {}

        void reset() {
            requests.reset();
            encodedExprs.reset();
            pairs.reset();
            indices.reset();
        }
    };

    /// @brief FNV-1a 64-bit hasher for `std::set<int16_t>`, used as the key type
    /// in `HashMemory::remainingArgsNormalizedEncodedMap`.
    ///
    /// @details
    /// `std::set` orders its elements, so iteration is deterministic; combined
    /// with the same FNV constants used elsewhere in the file, the hash value
    /// is stable across LBs and across threads. That stability is what allows
    /// the `remainingArgs`-keyed dispatch in `checkLocalEncodedMemoryStatic` to
    /// agree with the inserts performed in `addToHashMemory`.
    ///
    /// @see [`HashMemory::remainingArgsNormalizedEncodedMap`](#hashmemory) — the
    ///      consumer.
    struct SetInt16Hash {
        std::size_t operator()(const std::set<int16_t>& s) const {
            std::size_t h = 14695981039346656037ULL;
            for (int16_t v : s) {
                h ^= static_cast<std::size_t>(static_cast<uint16_t>(v));
                h *= 1099511628211ULL;
            }
            return h;
        }
    };

    // The HashMemory deload-base constants live in memory_infra/hash_memory.hpp
    // now that LbMemory holds the four instances and enumerates them through its
    // own visitContainers (D-147).

    /// @brief Central rule store of one *logic block* (LB) — the hash engine's
    /// per-LB state.
    ///
    /// @details
    /// A `Memory` instance owns four `HashMemory` slots (`overallHashMemory`,
    /// `localHashMemory`, `localHashMemoryDelta`, `workingMemory`); together they
    /// index every
    /// rule the prover can fire from this LB. The members fall into three
    /// groups:
    ///
    /// **Hot-path indices (int16_t-keyed)**
    /// - `encodedMap` — primary index. Maps a normalized premise key to the
    ///   stack of `LocalMemoryValue`s that can fire from it.
    /// - `remainingArgsNormalizedEncodedMap` — secondary index by the set of
    ///   carried arg ids; lets `checkLocalEncodedMemoryStatic` skip rules
    ///   whose remaining-args set has no overlap with the candidate.
    /// - `normalizedEncodedKeys`, `normalizedEncodedSubkeys`,
    ///   `normalizedEncodedSubkeysMinusOne`, `normalizedEncodedSubkeysMinusTwo`
    ///   — bloom-filter-style fast-rejection sets keyed at full / 1-short /
    ///   2-short / 3-short subkey lengths.
    /// - `maxKeyLength` — saturating upper bound used to short-circuit lookup
    ///   loops when the candidate exceeds anything stored.
    ///
    /// **Path-independent shared state**
    /// - `originals` — the set of original expression strings ever installed,
    ///   used during dedupe.
    /// - `admissionMap`, `admissionMapIntegration` — depth and instruction
    ///   budgets per `(expression, validity)`. See [`AdmissionMapValue`](#admissionmapvalue).
    /// - `admissionSetIntegration`, `triggersForAdmissionSetIntegration` —
    ///   integration-side admission control.
    /// - `rejectedMap`, `rejectedMapIntegration` — disintegration- and
    ///   integration-side rejection caches respectively. The integration side
    ///   has an extra `varsInRejectedMapIntegrationKeys` cache: a
    ///   monotonically-growing set of non-marker args appearing anywhere in
    ///   `rejectedMapIntegration` keys, used by
    ///   `applyEquivalenceClassToRejectedMapIntegration` to short-circuit
    ///   when a class has no key overlap (was the hot spot on Gauss
    ///   batches with ~10⁵ rmi entries × ~10⁶ class calls).
    /// - `admissionStatusMap` — per-key admission verdict cache.
    /// - `productsOfRecursionIds` — int16_t ids of expressions known to be
    ///   products of a recursion block; consulted for OR-disintegration
    ///   gating.
    /// - `consumedAdmissionKeys`, `revisitInProgress` — re-entrant guards.
    ///
    /// `clear()` wipes every slot; called when a `HashMemory` is reset for
    /// reuse or torn down.
    ///
    /// @see [`Memory`](#memory) — owns three `HashMemory` slots.
    /// @see [`LocalMemoryValue`](#localmemoryvalue) — value type of `encodedMap`.
    /// @see [I-22](../../docs/agentic_swdd/30_invariants.md#i-22),
    ///      [I-30](../../docs/agentic_swdd/30_invariants.md#i-30) — invariants on the
    ///      `rejectedMapIntegration` lifecycle.
    /// @see [I-32](../../docs/agentic_swdd/30_invariants.md#i-32) — cross-pair `equality2`
    ///      emission gated on existing class / LB origin (consumes this state).
    // struct HashMemory is defined in memory_infra/hash_memory.hpp
    // (every member is a cold container; held directly by LbMemory).


    // ========================================================================

    /// @brief String-form encoded expression — the legacy expression record kept
    /// alongside `IntEncodedExpr` for paths that have not yet migrated to the
    /// int16-only static pipeline.
    ///
    /// @details
    /// Carries the parsed structure of one expression — name, negation flag,
    /// argument vector (5 fields per argument: `[isUnchangeable, arg, iteration,
    /// levPlus1, argId]`), the maximum iteration counter across all arguments,
    /// the original (raw) string, and the validity scope name.
    ///
    /// Three constructors:
    /// - default — empty record (used for `std::map`/`std::set` insertion).
    /// - parameterized — caller already has the parsed pieces; we normalize
    ///   each raw arg via `parseArgument` and compute `maxIterationNumber`.
    /// - explicit-from-`(original, validityName)` — peels off the leading `!`
    ///   if present, splits the expression name from the args via
    ///   `ce::extractExpression*`, parses each arg.
    ///
    /// `parseArgument` handles two grammar shapes:
    /// 1. The full `it_<iter>_lev_<lev>_<argId>` form (with optional `u_`
    ///    prefix) — produces a 5-tuple with the raw iteration / level + 1 /
    ///    argId broken out.
    /// 2. Anything else — passes through as `[u/False, arg, "-1", "0", arg]`.
    ///
    /// `operator<` orders by `(original, validityName)` so identical text in
    /// different scopes is two distinct entries; `operator==` is a full
    /// field-by-field compare.
    ///
    /// @warning The `arguments` vector's inner shape is positional. Callers
    ///          rely on indices 0..4 having the documented meaning. Adding a
    ///          new positional field requires a coordinated change wherever
    ///          this struct is read.
    /// @see [`IntEncodedExpr`](#intencodedexpr) — int16_t replacement on the
    ///      hot path.
    /// @see `encodeExpression` — converts this into an `IntEncodedExpr`.
    struct EncodedExpression {
        // Fields (CamelCase)
        std::string name;
        bool negation;
        std::vector<std::vector<std::string> > arguments; // each: [isUnchangeable, arg, iteration, levPlus1, argId]
        int maxIterationNumber;
        std::string original;
        std::string validityName;

        // --- Constructors ---
        EncodedExpression()
            : name(),
            negation(false),
            arguments(),
            maxIterationNumber(-1),
            original(),
            validityName() {
        }

        // Build from pre-parsed pieces (you provide name, negation, and raw args; we normalize args like Python)
        EncodedExpression(const std::string& name_,
            bool negation_,
            const std::vector<std::string>& rawArgs,
            const std::string& original_,
            const std::string& validityName_)
            : name(name_),
            negation(negation_),
            arguments(),
            maxIterationNumber(-1),
            original(original_),
            validityName(validityName_) {
            for (std::size_t i = 0; i < rawArgs.size(); ++i) {
                arguments.push_back(parseArgument(rawArgs[i]));
            }
            maxIterationNumber = computeMaxIterationNumber(arguments);
        }

        explicit EncodedExpression(const std::string& original_, const std::string& validityName_)
            : name(),
            negation(false),
            arguments(),
            maxIterationNumber(-1),
            original(original_),
            validityName(validityName_) {
            if (original.size() >= 2 && original[0] == '!' && original[1] == '(') {
                negation = true;
                name = ce::extractExpressionFromNegation(original);
            }
            else {
                name = ce::extractExpression(original);
            }
            const std::vector<std::string> rawArgs = ce::getArgs(original);
            for (std::size_t i = 0; i < rawArgs.size(); ++i) {
                arguments.push_back(parseArgument(rawArgs[i]));
            }
            maxIterationNumber = computeMaxIterationNumber(arguments);
        }

        bool operator<(const EncodedExpression& rhs) const {
            if (original != rhs.original) {
                return original < rhs.original;
            }
            return validityName < rhs.validityName;
        }

        bool operator==(const EncodedExpression& rhs) const {
            return name == rhs.name
                && negation == rhs.negation
                && arguments == rhs.arguments
                && maxIterationNumber == rhs.maxIterationNumber
                && original == rhs.original
                && validityName == rhs.validityName;
        }

        // --- Helpers mirroring Python private methods ---

        // Matches: it_<iter>_lev_<lev>_<argId>
        // Fast path avoids regex; preserves exact output strings used elsewhere.
        static std::vector<std::string> parseArgument(const std::string& argIn) {
            using std::string;
            using std::size_t;

            std::string_view v(argIn);
            bool isU = (v.size() >= 2 && v[0] == 'u' && v[1] == '_');
            if (isU) v.remove_prefix(2); // strip "u_"

            // Quick shape check: "it_" ... "_lev_" ... "_" ...
            if (v.size() >= 7 && v[0] == 'i' && v[1] == 't' && v[2] == '_') {
                size_t p = v.find("_lev_", 3);
                if (p != std::string_view::npos) {
                    size_t p2 = v.find('_', p + 5);
                    if (p2 != std::string_view::npos) {
                        // Pull slices
                        std::string iter(v.substr(3, p - 3));
                        std::string levs(v.substr(p + 5, p2 - (p + 5)));
                        std::string id(v.substr(p2 + 1));

                        // lev + 1
                        int lev = std::atoi(levs.c_str());
                        std::string levPlus1 = std::to_string(lev + 1);

                        std::vector<std::string> out;
                        out.reserve(5);
                        out.emplace_back(isU ? "True" : "False");
                        out.emplace_back(std::string(v)); // arg without "u_"
                        out.emplace_back(std::move(iter));
                        out.emplace_back(std::move(levPlus1));
                        out.emplace_back(std::move(id));
                        return out;
                    }
                }
            }

            // Default branch from Python: ['False/True', arg, '-1', '0', arg]
            std::vector<std::string> out;
            out.reserve(5);
            out.emplace_back(isU ? "True" : "False");
            out.emplace_back(std::string(v));
            out.emplace_back("-1");
            out.emplace_back("0");
            out.emplace_back(std::string(v));
            return out;
        }


        static int computeMaxIterationNumber(const std::vector<std::vector<std::string> >& args) {
            int maxIt = -1;
            for (std::size_t i = 0; i < args.size(); ++i) {
                if (args[i].size() >= 3) {
                    const int it = std::atoi(args[i][2].c_str());
                    if (it > maxIt) maxIt = it;
                }
            }
            return maxIt;
        }
    };

    // StatementFlags (the intKnownStatements value: the local /
    // fullyDisintegrated / registered / known bits) moved to
    // int_encoded_expr.hpp so LbMemory can hold a
    // ColdHashMap<PodKeyStore<int32_t>, StatementFlags>; the registry needs the
    // complete value type at its member declaration. Its full Doxygen lives
    // there, beside IntEncodedExpr.

    /// @brief OR-style upsert into the packed-key statement registry.
    ///
    /// @details
    /// The single write door for `Memory::intKnownStatements` membership
    /// bits. If `key` is absent, a fresh `StatementFlags` row is inserted
    /// with exactly the given values. If `key` is present, the `registered`
    /// / `known` / `fullyDisintegrated` bits are OR-ed into the stored row
    /// and `local` keeps its first-writer value — the keep-first semantics
    /// every `insert({key, flags})` site has, where a duplicate insert is a
    /// flag-preserving no-op.
    ///
    /// Bits are OR-only by contract: no caller clears a bit through this
    /// door. The only bit-clearing writers are whole-entry erases
    /// (`Memory::wipeSubtree`, `eradicateImplicationFromLB`,
    /// `resetResentExpressionRegistries`) and the CE teardown's
    /// `registered`-membership reset in `releaseCEBatchMemory`.
    ///
    /// @param registry           The packed-key registry to write
    ///                           (`Memory::intKnownStatements`).
    /// @param key                Packed `(originalId, validityId)` key from
    ///                           `packStatementKey`.
    /// @param local              Stored only when the row is fresh
    ///                           (keep-first; write-only provenance field).
    /// @param registered         OR-ed membership bit — statement passed an
    ///                           add-path registration door.
    /// @param known              OR-ed membership bit — statement entered
    ///                           the level registry (Site F dedup record).
    /// @param fullyDisintegrated OR-ed disintegration verdict
    ///                           (`checkForEquivalence` sets it; never
    ///                           cleared here).
    /// @see `packStatementKey`, `StatementFlags`.
    inline void upsertStatementKey(
        TypedColdMap<StatementKey, StatementFlags>& registry,
        int32_t key, bool local, bool registered, bool known,
        bool fullyDisintegrated = false) {
        // The packed (originalId, validityId) scalar IS the inner engine's key
        // view; this low-level OR-merge write door reaches it through inner().
        const int32_t id = registry.inner().lookup(key);
        if (id == 0) {
            registry.inner().insert(key, StatementFlags{ local, fullyDisintegrated,
                                                         registered, known });
            return;
        }
        // OR the membership bits into the existing row via the reviewed
        // in-place update door (the cold insert is set-once).
        StatementFlags f = registry.inner().valueAt(id);
        f.registered         = f.registered || registered;
        f.known              = f.known || known;
        f.fullyDisintegrated = f.fullyDisintegrated || fullyDisintegrated;
        registry.inner().setValueAt(id, f);
    }

    /// @brief Non-minting flags fetch from the packed-key statement registry.
    ///
    /// @details
    /// Resolves `(original, validityName)` through `NameMap::lookup` — the
    /// read-only half of the interning registry — and returns the stored
    /// `StatementFlags` row, or `nullptr` when no row exists. A lookup miss
    /// on either name is definitive absence, not a failure: every row in the
    /// registry is keyed with minted ids
    /// ([I-84](../../docs/agentic_swdd/30_invariants.md#i-84) read
    /// discipline), so an un-interned name cannot have a row. Nothing is
    /// minted on any path, keeping the LB's id-assignment order untouched.
    ///
    /// Callers test the membership bit their contract names — `registered`
    /// for the statement-registration gates, `known` for the Site F family —
    /// never bare row presence.
    ///
    /// @param registry     The packed-key registry (`Memory::intKnownStatements`).
    /// @param nm           The owning LB's interning registry; read-only.
    /// @param original     Statement text to probe.
    /// @param validityName Scope name to probe.
    /// @return Pointer to the stored flags row, or `nullptr` when absent
    ///         (a contracted result — the gate's negative branch).
    /// @see `upsertStatementKey` — the write door.
    inline const StatementFlags* lookupStatementFlags(
        const TypedColdMap<StatementKey, StatementFlags>& registry,
        const NameMap& nm,
        const std::string& original, const std::string& validityName) {
        const int16_t origId = nm.lookup(original);
        if (origId == 0) return nullptr;
        const int16_t valId = nm.lookup(validityName);
        if (valId == 0) return nullptr;
        // Cold find returns the value pointer directly (nullptr on miss).
        return registry.find(StatementKey{ origId, valId });
    }

    /// @brief Span overload of the non-minting statement-flags probe.
    ///
    /// @details
    /// Byte-identical result to `lookupStatementFlags(..., const std::string&,
    /// const std::string&)`: resolves `(original, validityName)` through the
    /// non-minting `NameMap::lookup(StrSpan)` and returns the stored flags row
    /// (or `nullptr` on a definitive miss — an un-interned name can key no row,
    /// I-84 read discipline). Lets a caller holding spans over stable buffers
    /// (the addEquality / addNegatedEquality mirror gate over its built
    /// `(=[b,a])` scratch span) probe without materializing a `std::string`.
    /// Nothing is minted on any path.
    ///
    /// @param registry     The packed-key registry (`Memory::intKnownStatements`).
    /// @param nm           The owning LB's interning registry; read-only.
    /// @param original     Statement bytes to probe.
    /// @param validityName Scope bytes to probe.
    /// @return Pointer to the stored flags row, or `nullptr` when absent.
    /// @see lookupStatementFlags(..., const std::string&, const std::string&) —
    ///      the canonical overload; upsertStatementKey — the write door.
    inline const StatementFlags* lookupStatementFlags(
        const TypedColdMap<StatementKey, StatementFlags>& registry,
        const NameMap& nm,
        const StrSpan& original, const StrSpan& validityName) {
        const int16_t origId = nm.lookup(original);
        if (origId == 0) return nullptr;
        const int16_t valId = nm.lookup(validityName);
        if (valId == 0) return nullptr;
        return registry.find(StatementKey{ origId, valId });
    }

    /// @brief Non-minting levels fetch from the packed-key level index.
    ///
    /// @details
    /// Resolves `(original, validityName)` through `NameMap::lookup` — the
    /// read-only half of the interning registry — and returns the stored
    /// per-statement LB-level set from
    /// `Memory::intStatementLevelsMap`, or `nullptr` when no entry exists.
    /// A lookup miss on either name is definitive absence, not a failure:
    /// every index entry is keyed with minted ids
    /// ([I-84](../../docs/agentic_swdd/30_invariants.md#i-84) read
    /// discipline — both strings were interned by the entry's insert site),
    /// so an un-interned name cannot have an entry. Nothing is minted on
    /// any path, keeping the LB's id-assignment order untouched.
    ///
    /// Callers that already hold the int16 ids (an `IntEncodedExpr` row in
    /// hand) should probe the index directly with `packStatementKey` and
    /// skip the string resolution entirely.
    ///
    /// @param index        The packed-key level index
    ///                     (`Memory::intStatementLevelsMap`).
    /// @param nm           The owning LB's interning registry; read-only.
    /// @param original     Statement text to probe.
    /// @param validityName Scope name to probe.
    /// @return The cold map's key id (>= 1) for the entry, or 0 when absent
    ///         (a contracted result — the probe's negative branch). Iterate
    ///         the level run with `index.runLen(id)` / `index.valueAt(id, j)`
    ///         (ascending), or reconstruct the whole set via `coldIntSetAt`.
    /// @see `lookupStatementFlags` — the sibling probe on
    ///      `intKnownStatements`; `coldIntSetAt`; `packStatementKey`.
    inline int32_t lookupStatementLevels(
        const TypedColdSetMap<StatementKey, int>& index,
        const NameMap& nm,
        const std::string& original, const std::string& validityName) {
        const int16_t origId = nm.lookup(original);
        if (origId == 0) return 0;
        const int16_t valId = nm.lookup(validityName);
        if (valId == 0) return 0;
        return index.lookup(packStatementKey(origId, valId));
    }

    /// @brief Span overload of the non-minting levels fetch.
    ///
    /// @details
    /// Byte-identical result to `lookupStatementLevels(..., const std::string&,
    /// const std::string&)`: resolves `(original, validityName)` via the
    /// non-minting `NameMap::lookup(StrSpan)` and returns the cold key id
    /// (`>= 1`) of the level-set entry, or `0` on a definitive miss (I-84).
    /// Lets the equivalence-class apply tree's post-loop level lookups probe
    /// from arena spans over rebuilt keys without a heap string. Iterate the
    /// run with `index.runLen(id)` / `index.valueAt(id, j)` or rebuild via
    /// `coldIntSetAt`. Nothing is minted on any path.
    ///
    /// @param index        The packed-key level index
    ///                     (`Memory::intStatementLevelsMap`).
    /// @param nm           The owning LB's interning registry; read-only.
    /// @param original     Statement bytes to probe.
    /// @param validityName Scope bytes to probe.
    /// @return The cold map's key id (`>= 1`), or `0` when absent.
    /// @see lookupStatementLevels(..., const std::string&, const std::string&) —
    ///      the canonical overload; coldIntSetAt; packStatementKey.
    inline int32_t lookupStatementLevels(
        const TypedColdSetMap<StatementKey, int>& index,
        const NameMap& nm,
        const StrSpan& original, const StrSpan& validityName) {
        const int16_t origId = nm.lookup(original);
        if (origId == 0) return 0;
        const int16_t valId = nm.lookup(validityName);
        if (valId == 0) return 0;
        return index.lookup(packStatementKey(origId, valId));
    }

    /// @brief Non-minting membership probe on the packed-key local-origin
    ///        statement index.
    ///
    /// @details
    /// Resolves `(original, validityName)` through `NameMap::lookup` and
    /// probes `Memory::intLocalEncodedStatementsSet`. A lookup miss on
    /// either name is definitive non-membership, not a failure: every set
    /// member was inserted with minted ids
    /// ([I-84](../../docs/agentic_swdd/30_invariants.md#i-84) read
    /// discipline), so an un-interned name cannot be a member. Nothing is
    /// minted on any path, keeping the LB's id-assignment order untouched.
    ///
    /// Callers that already hold the int16 ids should probe the set
    /// directly with `packStatementKey` and skip the string resolution.
    ///
    /// @param index        The packed-key local-origin index
    ///                     (`Memory::intLocalEncodedStatementsSet`).
    /// @param nm           The owning LB's interning registry; read-only.
    /// @param original     Statement text to probe.
    /// @param validityName Scope name to probe.
    /// @return `true` when the `(original, validityName)` statement is a
    ///         local-origin statement of the LB; `false` otherwise (a
    ///         contracted result — the gate's negative branch).
    /// @see `lookupStatementLevels`, `lookupStatementFlags` — sibling
    ///      non-minting probes; `packStatementKey`.
    inline bool isLocalEncodedStatement(
        const ColdHashSet<PodKeyStore<int32_t>>& index,
        const NameMap& nm,
        const std::string& original, const std::string& validityName) {
        const int16_t origId = nm.lookup(original);
        if (origId == 0) return false;
        const int16_t valId = nm.lookup(validityName);
        if (valId == 0) return false;
        return index.contains(packStatementKey(origId, valId));
    }

    /// @brief Span overload of the non-minting local-origin membership probe.
    ///
    /// @details
    /// Byte-identical verdict to `isLocalEncodedStatement(..., const
    /// std::string&, const std::string&)`: resolves `(original, validityName)`
    /// via the non-minting `NameMap::lookup(StrSpan)` and tests
    /// `Memory::intLocalEncodedStatementsSet`; a never-interned name is
    /// definitive non-membership (I-84). Lets the equivalence-class rejected-map
    /// post-loop `isLocalEncodedStatement` gate probe from an arena span over
    /// the rebuilt compound without a heap string. Nothing is minted.
    ///
    /// @param index        The packed-key local-origin index
    ///                     (`Memory::intLocalEncodedStatementsSet`).
    /// @param nm           The owning LB's interning registry; read-only.
    /// @param original     Statement bytes to probe.
    /// @param validityName Scope bytes to probe.
    /// @return `true` iff `(original, validityName)` is a local-origin
    ///         statement of the LB; `false` otherwise.
    /// @see isLocalEncodedStatement(..., const std::string&, const
    ///      std::string&) — the canonical overload; packStatementKey.
    inline bool isLocalEncodedStatement(
        const ColdHashSet<PodKeyStore<int32_t>>& index,
        const NameMap& nm,
        const StrSpan& original, const StrSpan& validityName) {
        const int16_t origId = nm.lookup(original);
        if (origId == 0) return false;
        const int16_t valId = nm.lookup(validityName);
        if (valId == 0) return false;
        return index.contains(packStatementKey(origId, valId));
    }

    /// @brief Non-minting value fetch from the packed-key goal registry.
    ///
    /// @details
    /// Resolves `(original, validityName)` through `NameMap::lookup` — the
    /// read-only half of the interning registry — and returns the stored
    /// goal's auxy set, or `nullptr` when no goal exists. A lookup
    /// miss on either name is definitive absence, not a failure: every goal
    /// in `Memory::intToBeProved` is keyed with ids minted at its insert
    /// site ([I-84](../../docs/agentic_swdd/30_invariants.md#i-84) read
    /// discipline), so an un-interned name cannot have a goal. Nothing is
    /// minted on any path, keeping the LB's id-assignment order untouched —
    /// safe for the phase-2 read-only predicates (`burstDeactivates`).
    ///
    /// Callers that already hold the int16 ids (an `IntEncodedExpr` row in
    /// hand, as in `dischargeToBeProved`'s delta walk) should probe the
    /// registry directly with `packStatementKey` and skip the string
    /// resolution entirely.
    ///
    /// @param goals        The packed-key goal registry
    ///                     (`Memory::intToBeProved`).
    /// @param nm           The owning LB's interning registry; read-only.
    /// @param original     Goal expression text to probe.
    /// @param validityName Scope name to probe.
    /// @return The cold registry's key id (>= 1) for the goal, or 0 when
    ///         absent (a contracted result — the probe's negative branch).
    ///         Iterate the auxy run with `goals.runLen(id)` /
    ///         `goals.valueAt(id, j)`, or reconstruct via `coldIntSetAt`.
    /// @see `lookupStatementFlags`, `lookupStatementLevels` — sibling
    ///      non-minting probes; `decodeToBeProvedSorted` — the
    ///      order-preserving walk companion; `coldIntSetAt`; `packStatementKey`.
    inline int32_t lookupToBeProved(
        const TypedColdSetMap<StatementKey, int>& goals,
        const NameMap& nm,
        const std::string& original, const std::string& validityName) {
        const int16_t origId = nm.lookup(original);
        if (origId == 0) return 0;
        const int16_t valId = nm.lookup(validityName);
        if (valId == 0) return 0;
        return goals.lookup(packStatementKey(origId, valId));
    }

    /// @brief Reconstruct a cold set-map key's run as an ordered `std::set<int>`
    ///        — the boundary for the few sites needing the whole level / auxy
    ///        set (the run is already ascending-unique, so the set mirrors it).
    ///
    /// The prover's phase call trees feed level runs through `coldIntRunAt`
    /// instead; this heap form remains as the run twin's TEST ORACLE, and its
    /// only production callers live in the Rule-14 hashburst dump (diagnostic
    /// infrastructure off the phase trees).
    ///
    /// @param index A resident cold set-map (level index / goal registry).
    /// @param id    A key id from `lookupStatementLevels` / `lookupToBeProved`
    ///              (>= 1).
    /// @return The key's value run as a `std::set<int>`.
    /// @see `coldIntRunAt` — the zero-heap stack-run twin;
    ///      `lookupStatementLevels`, `lookupToBeProved`.
    inline std::set<int> coldIntSetAt(
        const TypedColdSetMap<StatementKey, int>& index, int32_t id) {
        std::set<int> out;
        const int32_t n = index.runLen(id);
        for (int32_t j = 0; j < n; ++j) out.insert(index.valueAt(id, j));
        return out;
    }

    /// @brief Copy a cold set-map key's value run into a caller-owned stack
    ///        buffer — the zero-heap twin of `coldIntSetAt`.
    ///
    /// @details
    /// The CSR run behind a `TypedColdSetMap` key is stored ascending and
    /// duplicate-free (`SetValueStore` contract), so the filled prefix
    /// `out[0..return)` is element-for-element identical to iterating the
    /// `std::set<int>` that `coldIntSetAt` would build — same values, same
    /// order, no heap.
    ///
    /// The copy is the point, not an inefficiency: the level runs feed the
    /// `addExprToMemoryBlock` kernel chain, whose callees mutate
    /// `intStatementLevelsMap` itself (`assignSetRange` splices the CSR pool
    /// and shifts later runs, and the chain re-enters through
    /// `addStatement -> updateEquivalenceClasses ->
    /// applyEquivalenceClassToNegatedEquality -> addStatement`). A pointer
    /// into the cold value column would be invalidated mid-call; the stack
    /// copy gives the caller frame ownership for the run's whole lifetime —
    /// byte-identical to the caller-owned `std::set<int>` copies it replaces.
    /// (`PagedVector` runs may also straddle pages, so the column is not even
    /// guaranteed contiguous.)
    ///
    /// @param index A resident cold set-map (level index / goal registry).
    /// @param id    A key id from `lookupStatementLevels` /
    ///              `lookupToBeProved` (>= 1).
    /// @param out   Caller-owned buffer receiving the run front-to-back.
    /// @param cap   Capacity of `out`; a run longer than `cap` asserts
    ///              (Rule 19 — never truncate).
    /// @return The run length (number of ints written).
    /// @invariant [I-136](../../docs/agentic_swdd/30_invariants.md#i-136)
    ///            — runs on the kernel chain are caller-owned and never alias
    ///            a callee-mutable cold column.
    /// @see `coldIntSetAt` — the retained heap twin and test oracle;
    ///      `lookupStatementLevels`.
    inline int32_t coldIntRunAt(
        const TypedColdSetMap<StatementKey, int>& index, int32_t id,
        int* out, int32_t cap) {
        const int32_t n = index.runLen(id);
        assert(n <= cap && "coldIntRunAt: level run exceeds caller capacity");
        for (int32_t j = 0; j < n; ++j) out[j] = index.valueAt(id, j);
        return n;
    }

    /// @brief Insert a value into an ascending-unique stack run, keeping it
    ///        ascending-unique — the run twin of `std::set<int>::insert`.
    ///
    /// @details
    /// On an ascending duplicate-free run this reproduces
    /// `std::set<int>::insert` exactly: the value lands at its sorted
    /// position, a duplicate is a no-op, and the resulting element sequence
    /// equals the set's iteration order byte-for-byte. Later elements shift
    /// one slot right to make room (plain backward copy — the run lives in a
    /// single caller stack frame, so there is no aliasing to reason about).
    ///
    /// @param run Caller-owned ascending-unique run (stack memory).
    /// @param n   Current element count.
    /// @param v   Value to insert.
    /// @param cap Capacity of `run`; growth past `cap` asserts
    ///            (Rule 19 — never truncate).
    /// @return The new element count (`n` when `v` was already present).
    /// @invariant [I-136](../../docs/agentic_swdd/30_invariants.md#i-136)
    ///            — the union sites on the kernel chain build their runs
    ///            through this tested primitive, not ad-hoc memmoves.
    /// @see `coldIntRunAt` — the run-fill sibling.
    inline int32_t insertLevelSorted(int* run, int32_t n, int v, int32_t cap) {
        int32_t i = 0;
        while (i < n && run[i] < v) ++i;
        if (i < n && run[i] == v) return n;
        assert(n + 1 <= cap && "insertLevelSorted: level run exceeds caller capacity");
        for (int32_t j = n; j > i; --j) run[j] = run[j - 1];
        run[i] = v;
        return n + 1;
    }

    /// @brief Encode an equivalence-class index key — (validityId, memberIds)
    ///        flattened to one byte key for the cold byte-key map.
    ///
    /// @details
    /// `eqClassSttmntIndexMapMap` was a nested
    /// `std::map<int16_t, std::map<std::vector<int16_t>, int>>`; the cold form
    /// (Batch 2) flattens it to a `ColdHashMap<BytesKeyStore, int>` keyed by the
    /// validity id (2 bytes) followed by the class member ids (2 bytes each, in
    /// their stored sorted order). Distinct (validity, members) pairs yield
    /// distinct byte sequences, so the byte key is a bijective identity — point
    /// lookups only (the map is never iterated in key order nor dumped), so the
    /// little-endian byte layout carries no ordering meaning.
    ///
    /// @param validityId The class's validity-scope id.
    /// @param memberIds  The class's sorted member ids.
    /// @return The packed byte key.
    /// @see `lookupEqClassIndex`, `upsertEqClassIndex`, `eraseEqClassIndex`.
    inline std::string encodeEqClassKey(int16_t validityId,
                                        const std::vector<int16_t>& memberIds) {
        return Codec<EqClassKey>::encode(EqClassKey{ validityId, memberIds });
    }

    /// @brief Read the statement-index waterline of a (validityId, memberIds)
    ///        equivalence class — 0 when absent.
    ///
    /// @details
    /// 0 reproduces the heap nested-map's `operator[]` default: the old
    /// `eqClassSttmntIndexMapMap[v][m]` auto-created a 0 entry on a miss and the
    /// caller used it as a from-the-start waterline. The cold read returns 0
    /// without creating — the immediately-following write installs the entry.
    ///
    /// @param m          The cold eq-class index map.
    /// @param validityId The class's validity-scope id.
    /// @param memberIds  The class's sorted member ids.
    /// @return The stored waterline, or 0 when the class has no entry.
    /// @see `encodeEqClassKey`, `upsertEqClassIndex`.
    inline int lookupEqClassIndex(
        const TypedColdMap<EqClassKey, int>& m, int16_t validityId,
        const std::vector<int16_t>& memberIds) {
        return m.findOr(EqClassKey{ validityId, memberIds }, 0);
    }

    /// @brief Set the statement-index waterline of a (validityId, memberIds)
    ///        class — set-or-insert (the heap `[v][m] = x` write door).
    ///
    /// @details
    /// In-place `setValueAt` on a hit, set-once `insert` on a miss.
    /// Single-threaded write side only (I-83).
    ///
    /// @param m          The cold eq-class index map.
    /// @param validityId The class's validity-scope id.
    /// @param memberIds  The class's sorted member ids.
    /// @param value      The waterline to store.
    /// @see `lookupEqClassIndex`, `encodeEqClassKey`.
    inline void upsertEqClassIndex(
        TypedColdMap<EqClassKey, int>& m, int16_t validityId,
        const std::vector<int16_t>& memberIds, int value) {
        m.upsert(EqClassKey{ validityId, memberIds }, value);
    }

    /// @brief Remove a (validityId, memberIds) class's index entry — the
    ///        merge-away erase (the heap `[v].erase(m)`). A miss is a defined
    ///        no-op.
    ///
    /// @param m          The cold eq-class index map.
    /// @param validityId The class's validity-scope id.
    /// @param memberIds  The (merged-away) class's sorted member ids.
    /// @see `upsertEqClassIndex`.
    inline void eraseEqClassIndex(
        TypedColdMap<EqClassKey, int>& m, int16_t validityId,
        const std::vector<int16_t>& memberIds) {
        m.erase(EqClassKey{ validityId, memberIds });
    }

    /// @brief One decoded row of the packed-key goal registry — owned
    ///        string copies plus the packed key they decode from.
    ///
    /// @details
    /// Produced by `decodeToBeProvedSorted`. The strings are owned copies,
    /// not views into the NameMap's cold-table bytes
    /// ([I-3](../../docs/agentic_swdd/30_invariants.md#i-3)) — consumers
    /// (`sanitizeToBeProved`, the post-absorb `checkNecessityForEquality`
    /// sweep, the vacuous-truth goal walk) call into machinery that may
    /// mint and would dangle direct `decode()` references.
    struct DecodedToBeProvedRow {
        std::string original;
        std::string validityName;
        int32_t key;
    };

    /// @brief Decoded, lexicographically sorted snapshot of the packed-key
    ///        goal registry.
    ///
    /// @details
    /// Unpacks every key of `Memory::intToBeProved`, decodes both ids into
    /// owned string copies, and sorts the rows by `(original,
    /// validityName)` — exactly the iteration order the former
    /// `std::map<EncodedExpression, …>` container produced via
    /// `EncodedExpression::operator<`. Every order-sensitive walk of the
    /// goal registry (the `sanitizeToBeProved` rewrite staging, the
    /// post-absorb `checkNecessityForEquality` sweep, the vacuous-truth
    /// first-main-goal pick, the hashburst dump section) iterates this
    /// snapshot so processing order — and therefore every downstream
    /// trace byte — is invariant under the container re-keying. Raw ids
    /// are never used as a sort key
    /// ([I-84](../../docs/agentic_swdd/30_invariants.md#i-84): mint order
    /// is not lex order).
    ///
    /// The snapshot is also the safe-iteration vehicle: consumers may
    /// insert into or erase from the live registry while walking the
    /// returned rows.
    ///
    /// @param goals The packed-key goal registry (`Memory::intToBeProved`).
    /// @param nm    The owning LB's interning registry; read-only here.
    /// @return Rows sorted ascending by `(original, validityName)`; one
    ///         row per registry entry.
    /// @see `lookupToBeProved` — the point-probe companion;
    ///      `packStatementKey`.
    inline std::vector<DecodedToBeProvedRow> decodeToBeProvedSorted(
        const TypedColdSetMap<StatementKey, int>& goals,
        const NameMap& nm) {
        std::vector<DecodedToBeProvedRow> rows;
        const int32_t n = goals.count();
        rows.reserve(static_cast<size_t>(n));
        for (int32_t id = 1; id <= n; ++id) {
            const int32_t key = goals.keyAt(id);
            const int16_t origId = static_cast<int16_t>(
                (static_cast<uint32_t>(key) >> 16) & 0xFFFF);
            const int16_t valId = static_cast<int16_t>(
                static_cast<uint32_t>(key) & 0xFFFF);
            DecodedToBeProvedRow row;
            row.original = std::string(nm.decodeView(origId).ptr,
                static_cast<std::size_t>(nm.decodeView(origId).len));
            row.validityName = std::string(nm.decodeView(valId).ptr,
                static_cast<std::size_t>(nm.decodeView(valId).len));
            row.key = key;
            rows.push_back(std::move(row));
        }
        std::sort(rows.begin(), rows.end(),
            [](const DecodedToBeProvedRow& a, const DecodedToBeProvedRow& b) {
                if (a.original != b.original) return a.original < b.original;
                return a.validityName < b.validityName;
            });
        return rows;
    }

    /// @brief Heap-free twin of `decodeToBeProvedSorted` — a snapshot of the
    ///        goal registry's packed keys, sorted by decoded
    ///        `(original, validityName)`.
    ///
    /// @details
    /// The owning `decodeToBeProvedSorted` materializes one `DecodedToBeProvedRow`
    /// (two owned `std::string`s) per goal so a mutate-during-walk consumer can
    /// hold them across mints. This twin returns instead the PACKED KEYS
    /// themselves — stable `int32_t` (`packStatementKey(originalId, validityId)`)
    /// on the caller's arena byte-bump tier — sorted by the SAME decoded
    /// `(original, validityName)` comparator (`compareSpans` == `std::string::compare`,
    /// then a validity tie-break), so the key order is byte-identical to the
    /// former row order. The caller edge-materializes each row's two strings
    /// FRESH right before use (09c pitfall 5), after any prior mint, so no
    /// `decodeView` span crosses a mint.
    ///
    /// The snapshot survives everything the consumer does during the walk:
    /// `originalId` / `validityId` are NameMap ids and NameMap only APPENDS, so
    /// no key value shifts even as `checkNecessityForEquality` mints and the goal
    /// registry is inserted into / erased from mid-walk (exactly why the owning
    /// form snapshots too). No mint occurs during the sort — the comparator only
    /// reads cold-table spans — so every `decodeView` span stays valid across the
    /// whole `std::sort` ([I-3](../../docs/agentic_swdd/30_invariants.md#i-3)).
    /// The `int32_t[]` rides @p arena's byte-bump tier; the caller MUST hold a
    /// live `ScratchScope` across the call and every use, and the array dies at
    /// that scope's rewind. Because the array is allocated BEFORE any callee
    /// `ScratchScope` mark, every inner rewind pops back to above it.
    ///
    /// @param goals    The packed-key goal registry (`Memory::intToBeProved`).
    /// @param nm       The owning LB's NameMap; read-only — nothing is minted.
    /// @param arena    The per-slot scratch (bump) arena the key snapshot rides.
    /// @param outCount Receives the number of keys written (== `goals.count()`).
    /// @return A pointer to `outCount` `int32_t` packed keys in ascending
    ///         `(original, validityName)` order; `nullptr` when the registry is
    ///         empty.
    /// @see decodeToBeProvedSorted — the owning-copy sort twin kept for the
    ///      hashburst dump + the oracle; sortStatementRows; packStatementKey.
    inline int32_t* sortToBeProvedKeys(
        const TypedColdSetMap<StatementKey, int>& goals,
        const NameMap& nm, LbArena& arena, int32_t& outCount) {
        const int32_t n = goals.count();
        outCount = n;
        if (n == 0) return nullptr;
        int32_t* keys = reinterpret_cast<int32_t*>(arena.resolve(arena.alloc(
            n * static_cast<int32_t>(sizeof(int32_t)),
            static_cast<int32_t>(alignof(int32_t)))));
        for (int32_t i = 0; i < n; ++i) keys[i] = goals.keyAt(i + 1);
        std::sort(keys, keys + n, [&](int32_t ka, int32_t kb) {
            const int16_t origA = static_cast<int16_t>(
                (static_cast<uint32_t>(ka) >> 16) & 0xFFFF);
            const int16_t origB = static_cast<int16_t>(
                (static_cast<uint32_t>(kb) >> 16) & 0xFFFF);
            const int c = compareSpans(nm.decodeView(origA), nm.decodeView(origB));
            if (c != 0) return c < 0;
            const int16_t valA = static_cast<int16_t>(
                static_cast<uint32_t>(ka) & 0xFFFF);
            const int16_t valB = static_cast<int16_t>(
                static_cast<uint32_t>(kb) & 0xFFFF);
            return compareSpans(nm.decodeView(valA), nm.decodeView(valB)) < 0;
        });
        return keys;
    }

    /// @brief Byte-bump index that orders a `PagedVector<IntEncodedExpr>`
    ///        exactly as `std::sort` under `ExpressionWithValidity::operator<`
    ///        would — the decode-then-lex-sort for the transient statement
    ///        buffers.
    ///
    /// @details
    /// The transient statement buffers `newStatements` (returned by
    /// `addStatement`) and `products` (filled per statement in
    /// `applyEquiClasses`) were `std::vector<ExpressionWithValidity>` sorted with
    /// `std::sort` — lexicographic over `(original, validityName)` as raw
    /// `std::string` (`ExpressionWithValidity::operator<`) — before their
    /// consumers ran. Statified to `PagedVector<IntEncodedExpr>` (id form), the
    /// same observable consumption ORDER is reproduced here by an index sort:
    /// allocate an `int32_t[]` identity permutation on @p arena's byte-bump tier,
    /// then `std::sort` that permutation by comparing each row's decoded
    /// `originalId`, then (on a tie) its decoded `validityId`, via `compareSpans`
    /// (== `std::string::compare`, byte-lex). Because `IntEncodedExpr::originalId
    /// == nm.encode(original)` and `validityId == nm.encode(validityName)`,
    /// `nm.decodeView(originalId)` reconstructs the exact bytes the former
    /// `ExpressionWithValidity::original` held, so the produced order is
    /// byte-identical to the former `std::sort(std::vector<EWV>)`. Duplicate
    /// `(original, validityName)` rows tie under the total order exactly as they
    /// did for the EWV sort; being byte-identical, their relative order is
    /// immaterial to the idempotent consumers (`ordisMerge` / the admission
    /// consumers).
    ///
    /// No mint occurs during the sort — the comparator only reads cold-table
    /// spans — so every `decodeView` span stays valid across the whole
    /// `std::sort` ([I-3](../../docs/agentic_swdd/30_invariants.md#i-3)). The
    /// byte-bump `alloc` coexists with the page-tier `PagedVector` on the same
    /// arena (only a page-tier scratch fill via `allocBytes` must not share the
    /// arena with a page container — see
    /// [I-132](../../docs/agentic_swdd/30_invariants.md#i-132));
    /// this is the identical coexistence the `applyEquiClasses` `vkOrder` sort
    /// already relies on. The returned pointer is owned by @p arena: the caller
    /// MUST hold a live `ScratchScope` (or otherwise keep the arena position)
    /// across the call and every use, and the array dies at that scope's rewind.
    ///
    /// @param rows  The transient statement buffer to order; read-only.
    /// @param nm    The owning LB's NameMap; read-only — nothing is minted.
    /// @param arena The per-slot scratch (bump) arena the index rides.
    /// @return A pointer to `rows.size()` `int32_t`s — a permutation of
    ///         `[0, rows.size())` in ascending `(original, validityName)` order;
    ///         `nullptr` when @p rows is empty (nothing to order).
    /// @see `decodeToBeProvedSorted` — the owning-copy sort twin;
    ///      `ExpressionWithValidity::operator<`; `compareSpans`.
    inline int32_t* sortStatementRows(const PagedVector<IntEncodedExpr>& rows,
                                      const NameMap& nm, LbArena& arena) {
        const int32_t n = rows.size();
        if (n == 0) return nullptr;
        int32_t* order = reinterpret_cast<int32_t*>(arena.resolve(arena.alloc(
            n * static_cast<int32_t>(sizeof(int32_t)),
            static_cast<int32_t>(alignof(int32_t)))));
        for (int32_t i = 0; i < n; ++i) order[i] = i;
        std::sort(order, order + n, [&](int32_t a, int32_t b) {
            const int c = compareSpans(nm.decodeView(rows[a].originalId),
                                       nm.decodeView(rows[b].originalId));
            if (c != 0) return c < 0;
            return compareSpans(nm.decodeView(rows[a].validityId),
                                nm.decodeView(rows[b].validityId)) < 0;
        });
        return order;
    }

    /// @brief Convert a parsed `EncodedExpression` (string fields) into an
    /// `IntEncodedExpr` (int16_t fields) using a `NameMap` to mint stable ids.
    ///
    /// @details
    /// The string-side `EncodedExpression` is the language-level record; the
    /// int16-side `IntEncodedExpr` is the hot-path record. This function maps
    /// every string field through `nm.encode` (mints a fresh id on first
    /// sight, returns the existing id otherwise) and copies the per-arg flags
    /// directly. Three encoder details worth flagging:
    ///
    /// 1. `argId[i]` stores the *u_-prefixed* version when the argument is
    ///    unchangeable — matching what `makeIntNormalizedKey` does with
    ///    `ignoreU=false`. The non-prefixed full string is in `argFullId[i]`.
    /// 2. Arity above `ExecutionParameters::MAX_ARITY` asserts — the int row
    ///    must be a lossless re-encoding of `ee` (the registry's only stored
    ///    form; `decodeExpression` reconstructs the string side from it), so
    ///    an argument that would not fit is a contract violation, never a
    ///    truncation. In practice expressions stay far below the cap
    ///    (Gauss-tail Anchor at arity 6 is the observed maximum).
    /// 3. `isHypo` is derived by substring-search for `_hypo_` in the
    ///    validity name; `isAnchor` by prefix-match `Anchor` on the
    ///    expression name. Both are precomputed here so the hot path can
    ///    branch on a single `int16_t` flag instead of a string scan.
    ///
    /// @pre `ee` was constructed via the canonical pipeline (parser →
    ///      `EncodedExpression(...)` ctor); fields are not partially-populated.
    /// @post Returns an `IntEncodedExpr` whose every id field references `nm`.
    ///       The returned value is trivially copyable; consumers store it
    ///       into `TypedArena<IntEncodedExpr>` or
    ///       `Memory::intEncodedStatements`.
    /// @invariant All non-`"main"` validity names registered through `nm`
    ///            during this call are minted via `NameMap::encode` (which
    ///            either returns an existing id or recurses into `encodePush`
    ///            for `_boundary_`-bearing names). See
    ///            [I-2](../../docs/agentic_swdd/30_invariants.md#i-2).
    /// @see [`EncodedExpression`](#encodedexpression),
    ///      [`IntEncodedExpr`](#intencodedexpr).
    struct Memory;

    /// @brief Export-phase reload sink. When non-null, `Memory::reloadFromImage`
    ///        appends each LB it brings back, so the chapter-export driver can
    ///        release it per chapter.
    ///
    /// @details
    /// The read-only chapter export reloads LBs from their SSD images on demand
    /// (`ensureLoadedForRead`), but nothing released them afterward, so the
    /// resident block set grew monotonically across the walk to static-pool
    /// exhaustion (G-53). The driver sets this to a
    /// local vector for the export window and releases the recorded LBs per
    /// theorem; it is null everywhere else, so the kernel's own pressure reloads
    /// during `proveKernel` stay untracked. Single-threaded: the export runs
    /// after the parallel phases.
    extern std::vector<Memory*>* g_exportReloadSink;

    /// @brief Build an LB's full chain string — every `exprKey` from the
    ///        LB itself up to the root sentinel, joined by `__`.
    ///
    /// @details
    /// The statification deload identity (file naming, header chain
    /// field): same walk as the hashburst dump's LB-chain section,
    /// flattened to one string. The root sentinel's empty `exprKey` is
    /// included as the final (empty) element, so a child of the root
    /// ends in a trailing `__` and the root alone yields the empty
    /// string. The full parent chain — never a bare `exprKey` — is the
    /// LB identity (Rule 12: many LBs share an `exprKey` at different
    /// tree positions).
    ///
    /// @param body The LB to identify.
    /// @return The chain string, innermost first.
    /// @see `lbdeload::deloadFileName`, `Memory::deloadStaticContainers`.
    std::string buildLbChainString(const Memory& body);

    inline IntEncodedExpr encodeExpression(const EncodedExpression& ee, NameMap& nm) {
        IntEncodedExpr ie;
        ie.nameId = nm.encode(ee.name);
        ie.negation = ee.negation ? 1 : 0;
        assert(ee.arguments.size()
            <= static_cast<std::size_t>(ExecutionParameters::MAX_ARITY));
        ie.arity = static_cast<int16_t>(ee.arguments.size());
        ie.maxIteration = static_cast<int16_t>(ee.maxIterationNumber);
        ie.originalId = nm.encode(ee.original);
        ie.validityId = nm.encode(ee.validityName);
        ie.isHypo = (ee.validityName.find("_hypo_") != std::string::npos) ? 1 : 0;
        ie.isAnchor = (ee.name.rfind("Anchor", 0) == 0) ? 1 : 0;

        std::memset(ie.argId, 0, sizeof(ie.argId));
        std::memset(ie.argUnchangeable, 0, sizeof(ie.argUnchangeable));
        std::memset(ie.argIteration, 0, sizeof(ie.argIteration));
        std::memset(ie.argLevPlus1, 0, sizeof(ie.argLevPlus1));
        std::memset(ie.argFullId, 0, sizeof(ie.argFullId));

        for (int16_t i = 0; i < ie.arity; ++i) {
            const std::vector<std::string>& arg = ee.arguments[static_cast<std::size_t>(i)];
            if (arg.size() >= 2) {
                ie.argUnchangeable[i] = (arg[0] == "True") ? 1 : 0;
                ie.argFullId[i] = nm.encode(arg[1]);
                // argId stores u_-prefixed version for unchangeable args,
                // matching what makeIntNormalizedKey does with ignoreU=false
                ie.argId[i] = (arg[0] == "True")
                    ? nm.encode("u_" + arg[1])
                    : nm.encode(arg[1]);
            }
            if (arg.size() >= 3) {
                ie.argIteration[i] = static_cast<int16_t>(std::atoi(arg[2].c_str()));
            } else {
                ie.argIteration[i] = -1;
            }
            if (arg.size() >= 4) {
                ie.argLevPlus1[i] = static_cast<int16_t>(std::atoi(arg[3].c_str()));
            }
        }
        return ie;
    }

    /// @brief Span overload of @ref encodeExpression — parse a canonical MPL
    ///        expression and encode it to an `IntEncodedExpr` in one pass,
    ///        without building the intermediate `EncodedExpression`.
    ///
    /// @details
    /// `addExprToMemoryBlock` and `addStatement` previously built an
    /// `EncodedExpression` (a heap `std::string name` plus a nested
    /// `std::vector<std::vector<std::string>> arguments` from `ce::getArgs`)
    /// purely to feed `encodeExpression(ee, nm)`. This overload skips that
    /// scratch: it slices the name and arguments as `StrSpan`s of the stable
    /// `original` buffer (via `getArgsSpans`), interns them through the additive
    /// `NameMap::encode(const StrSpan&)` overload, and computes the integer
    /// fields by arithmetic — the one value the struct path materialized,
    /// `to_string(lev+1)`, was immediately `atoi`'d back, so it becomes
    /// `lev + 1`. Per argument: `argId` interns the raw arg span (which equals
    /// `"u_" + name` for an unchangeable arg and `name` otherwise), `argFullId`
    /// interns the `u_`-stripped span, `argUnchangeable` flags the `u_` prefix,
    /// and `argIteration` / `argLevPlus1` parse the `it_<i>_lev_<l>_<id>` shape.
    /// The result is byte-identical to
    /// `encodeExpression(EncodedExpression(original, validity), nm)`.
    ///
    /// @param original Canonical MPL expression text; a slice of a stable buffer.
    /// @param validity The scope (validity) name; a slice of a stable buffer.
    /// @param nm The owning LB's NameMap (mints/looks up ids).
    /// @return The encoded int row, identical to the `EncodedExpression` path.
    /// @see `encodeExpression(const EncodedExpression&, NameMap&)` — the twin.
    /// @see `getArgsSpans`, `NameMap::encode(const StrSpan&)`.
    inline IntEncodedExpr encodeExpression(const StrSpan& original,
                                           const StrSpan& validity,
                                           NameMap& nm) {
        IntEncodedExpr ie;

        const std::string_view orig(original.ptr,
                                    static_cast<std::size_t>(original.len));
        const bool negation =
            (original.len >= 2 && original.ptr[0] == '!' && original.ptr[1] == '(');
        const std::size_t br = orig.find('[');

        // Name span: between the opening "(" (or "!(") and the first "[".
        // No '[' -> empty name, byte-identical to ce::extractExpression.
        StrSpan nameSpan;
        if (br != std::string_view::npos) {
            std::size_t nameStart = 0;
            if (negation) nameStart = 2;                                  // "!(name["
            else if (original.len >= 1 && original.ptr[0] == '(') nameStart = 1; // "(name["
            nameSpan = StrSpan(original.ptr + nameStart,
                               static_cast<int32_t>(br - nameStart));
        }

        ie.nameId = nm.encode(nameSpan);
        ie.negation = negation ? 1 : 0;
        ie.originalId = nm.encode(original);
        ie.validityId = nm.encode(validity);
        ie.isHypo =
            (std::string_view(validity.ptr, static_cast<std::size_t>(validity.len))
                 .find("_hypo_") != std::string_view::npos) ? 1 : 0;
        ie.isAnchor =
            (nameSpan.len >= 6 && std::memcmp(nameSpan.ptr, "Anchor", 6) == 0) ? 1 : 0;

        std::memset(ie.argId, 0, sizeof(ie.argId));
        std::memset(ie.argUnchangeable, 0, sizeof(ie.argUnchangeable));
        std::memset(ie.argIteration, 0, sizeof(ie.argIteration));
        std::memset(ie.argLevPlus1, 0, sizeof(ie.argLevPlus1));
        std::memset(ie.argFullId, 0, sizeof(ie.argFullId));

        StrSpan argSpans[ExecutionParameters::MAX_ARITY];
        const int32_t argN = getArgsSpans(original, argSpans,
                                          ExecutionParameters::MAX_ARITY);
        assert(argN <= ExecutionParameters::MAX_ARITY);
        ie.arity = static_cast<int16_t>(argN);

        // atoi over a non-owning view (the substrings are short integer runs),
        // byte-identical to std::atoi on the struct path's std::string fields.
        auto atoiView = [](std::string_view sv) -> int {
            char buf[32];
            const std::size_t n = sv.size() < 31 ? sv.size() : 31;
            std::memcpy(buf, sv.data(), n);
            buf[n] = '\0';
            return std::atoi(buf);
        };

        int maxIt = -1;
        for (int32_t i = 0; i < argN; ++i) {
            const StrSpan& a = argSpans[i];
            const bool isU = (a.len >= 2 && a.ptr[0] == 'u' && a.ptr[1] == '_');
            const StrSpan vspan = isU ? StrSpan(a.ptr + 2, a.len - 2) : a;

            ie.argUnchangeable[i] = isU ? 1 : 0;
            ie.argFullId[i] = nm.encode(vspan);   // arg[1] — u_-stripped name
            ie.argId[i] = nm.encode(a);           // raw arg — "u_"+name or name

            int iterVal = -1;
            int levPlus1 = 0;
            const std::string_view v(vspan.ptr, static_cast<std::size_t>(vspan.len));
            if (v.size() >= 7 && v[0] == 'i' && v[1] == 't' && v[2] == '_') {
                const std::size_t p = v.find("_lev_", 3);
                if (p != std::string_view::npos) {
                    const std::size_t p2 = v.find('_', p + 5);
                    if (p2 != std::string_view::npos) {
                        iterVal = atoiView(v.substr(3, p - 3));
                        levPlus1 = atoiView(v.substr(p + 5, p2 - (p + 5))) + 1;
                    }
                }
            }
            ie.argIteration[i] = static_cast<int16_t>(iterVal);
            ie.argLevPlus1[i] = static_cast<int16_t>(levPlus1);
            if (iterVal > maxIt) maxIt = iterVal;
        }
        ie.maxIteration = static_cast<int16_t>(maxIt);
        return ie;
    }

    /// @brief Reconstruct the string-form `EncodedExpression` of an
    /// `IntEncodedExpr` row from the owning LB's `NameMap` — the exact inverse
    /// of `encodeExpression`.
    ///
    /// @details
    /// `encodeExpression` interns the WHOLE original expression text as
    /// `originalId` and the validity-scope name as `validityId`; every other
    /// `EncodedExpression` field (name, negation, the per-argument 5-tuples,
    /// `maxIterationNumber`) is a pure derivation of those two strings through
    /// the parsing constructor `EncodedExpression(original, validityName)`.
    /// Decoding therefore needs only the two ids: look both strings up, then
    /// let the canonical parsing constructor re-derive the rest. The result is
    /// field-for-field equal (full `operator==`) to the `EncodedExpression`
    /// the row was encoded from — which is what makes the int registry
    /// (`Memory::intEncodedStatements` and siblings) a lossless single source
    /// of truth, with string structs existing only transiently at boundaries
    /// (diagnostic dump, mail fill, visualizer, equivalence-class rewriting).
    ///
    /// The decodes are materialized (by-value `decode`, never `decodeView`):
    /// a view into the cold-table bytes would dangle if the caller mints
    /// afterwards
    /// ([I-3](../../docs/agentic_swdd/30_invariants.md#i-3)). The helper
    /// itself never mints — it takes `const NameMap&` to prove it, so calling
    /// it cannot disturb the LB's id-assignment order (byte-identical runs).
    ///
    /// The cross-field asserts after reconstruction (arity / negation /
    /// maxIteration agreement with the int row) are the cheap tripwire for
    /// encode/decode drift: if the forward encoder and this inverse ever
    /// disagree on a derived field, the run stops here with the divergence in
    /// hand instead of emitting subtly-wrong boundary output.
    ///
    /// @param ie The int row to reconstruct. Must carry real interned ids;
    ///           the slot-0 sentinel (never minted for a stored statement)
    ///           asserts.
    /// @param nm The owning LB's `NameMap` — the same map that encoded `ie`.
    ///           Read-only.
    /// @return The reconstructed `EncodedExpression`, equal under the full
    ///         field-by-field `operator==` to the struct `ie` was encoded
    ///         from.
    /// @invariant Non-minting: `nm` is `const`, id-assignment order is
    ///            untouched.
    /// @see `encodeExpression` — the forward direction.
    /// @see [I-3](../../docs/agentic_swdd/30_invariants.md#i-3) — decode
    ///      returns references; copy before any mint.
    inline EncodedExpression decodeExpression(const IntEncodedExpr& ie,
                                              const NameMap& nm) {
        assert(ie.originalId > 0
            && static_cast<int32_t>(ie.originalId) <= nm.nameCount());
        assert(ie.validityId > 0
            && static_cast<int32_t>(ie.validityId) <= nm.nameCount());
        const std::string original = nm.decode(ie.originalId);      // copy (I-3)
        const std::string validityName = nm.decode(ie.validityId);  // copy (I-3)
        EncodedExpression ee(original, validityName);
        assert(static_cast<int16_t>(ee.arguments.size()) == ie.arity);
        assert(static_cast<int16_t>(ee.negation ? 1 : 0) == ie.negation);
        assert(static_cast<int16_t>(ee.maxIterationNumber) == ie.maxIteration);
        return ee;
    }

    /// @brief Record one owner's u_ (unchangeable) argument signature into a
    ///        fast-rejection (sub)key's owner-set — the insert-side producer for
    ///        the request-generation u_ literal prune (D-120).
    ///
    /// @details Called at every owner insert into the four `normalizedEncoded*`
    /// maps, beside the `OwnerSet::partitionIds.insert`, with the
    /// `EncodedExpression` list the (sub)key was built from. It walks that list in
    /// arg order (mirroring how `encodeExpression` / `makeIntNormalizedKey` lay
    /// out arguments) and, for each unchangeable arg (`arg[0] == "True"`), appends
    /// `(linear-arg-slot, NameMap::encode(arg[1]))` to the owner's signature. The
    /// recorded id is the owner's `argFullId` (`encode(arg[1])` — see
    /// `encodeExpression`), exactly the value the firing gate matches against the
    /// request's `argFullId`, so `ownerSetUSatisfied` performs the firing gate's
    /// necessary condition early.
    ///
    /// An owner with no unchangeable args sets `OwnerSet::hasLooseOwner` — it can
    /// always potentially fire, so the check is skipped for the whole key.
    /// Non-empty signatures are inserted into the deduped `OwnerSet::uSignatures`.
    ///
    /// The literal id is read with the non-minting `NameMap::lookup`, never
    /// `encode`: minting a fresh id here would change the LB's id-assignment
    /// order and break byte-identicality. At the head-rule insert sites
    /// (`addToHashMemory`) every `arg[1]` is already interned by the preceding
    /// `ignoreU=true` key build, so `lookup` returns the real id and the full
    /// signature is recorded. At a marker-subkey insert
    /// (`makeNormalizedKeysForAdmission`) a `u_` literal may not be interned yet;
    /// `lookup` then returns the slot-0 sentinel `0`, and the key is flagged
    /// loose rather than recording an incomplete signature — a sound
    /// over-approximation (the request is kept, never wrongly pruned;
    /// I-79). The literal is interned later in the same
    /// install, so the same owner's *full-key* signature (recorded after that
    /// build) still carries the real ids.
    ///
    /// @param os      The (sub)key's owner-set being populated.
    /// @param encList The premise list the (sub)key was built from, in the same
    ///                permutation order as the key bytes.
    /// @param nm      The owning LB's NameMap, read via `lookup` only (no mint).
    /// @see `ownerSetUSatisfied` — the request-generation consumer.
    /// @see `OwnerSet::uSignatures`, `OwnerSet::hasLooseOwner`.
    inline void recordUSignature(OwnerSet& os,
                                 const std::vector<EncodedExpression>& encList,
                                 const NameMap& nm) {
        std::vector<std::pair<int16_t, int16_t>> sig;
        int16_t slot = 0;
        bool hasUArg = false;
        bool unknownLiteral = false;
        for (const EncodedExpression& expr : encList) {
            const std::vector<std::vector<std::string>>& args = expr.arguments;
            const int16_t arity = static_cast<int16_t>(std::min(args.size(),
                static_cast<std::size_t>(ExecutionParameters::MAX_ARITY)));
            for (int16_t j = 0; j < arity; ++j) {
                const std::vector<std::string>& arg = args[static_cast<std::size_t>(j)];
                if (arg.size() >= 2 && arg[0] == "True") {
                    hasUArg = true;
                    const int16_t id = nm.lookup(arg[1]);   // non-minting
                    if (id == 0) unknownLiteral = true;     // literal not yet interned
                    else sig.emplace_back(slot, id);
                }
                ++slot;
            }
        }
        // No u_ args (genuinely loose), or a u_ literal not yet interned (can't
        // soundly represent the owner's constraint) -> flag the key loose so
        // ownerSetUSatisfied skips it. Otherwise record the full signature.
        if (!hasUArg || unknownLiteral) os.hasLooseOwner = true;
        else os.uSignatures.insert(std::move(sig));
    }

    /// @brief IntEncodedExpr overload of @ref recordUSignature — the u_ signature
    ///        from pre-encoded premises (no heap EncodedExpression, so no
    ///        parseArgument). `argFullId` is the raw (non-u_) name id;
    ///        `argUnchangeable` the True/False flag.
    ///
    /// @details The raw name is always interned (encodeExpression minted
    /// `argFullId`), so the string overload's `unknownLiteral -> loose owner`
    /// fallback — a non-minting lookup that missed — cannot fire; in the
    /// admission-install paths the raw names are interned before this runs, so
    /// the two agree. (Empirically validated by the full-pipeline byte gate.)
    ///
    /// Heap-free: the `(slot, argFullId)` pairs are built on the per-slot
    /// `genScratchArenas` byte-bump tier and the signature vector is
    /// range-constructed INSIDE the set node via
    /// `os.uSignatures.emplace(first, last)` — the sink container's heap, owned
    /// by the still-heap caller's `OwnerSet`, is grown; no function-local
    /// `std::vector` is materialized. It is defined out-of-line (memory.cpp)
    /// because it reaches the scratch registry (`genScratchArenas()`) and the
    /// worker-slot id (`ExpressionAnalyzer::g_currentCoreId`), neither of which
    /// an inline header body can see — the `encodePush(StrSpan)` precedent.
    /// Single-threaded install site (mergeOwnerRecord, I-83), so the arena
    /// reach via `g_currentCoreId` (workers publish it; -1 -> reserved last
    /// slot) is valid.
    ///
    /// @param os      The owner set to record into.
    /// @param encList The pre-encoded premise expressions.
    /// @param count   Their number.
    /// @see recordUSignature(OwnerSet&, const std::vector<EncodedExpression>&,
    ///      const NameMap&) — the string twin (retained oracle).
    void recordUSignature(OwnerSet& os,
                          const IntEncodedExpr* encList, int16_t count);

    /// @brief Produce the u_ signature run of a set of pre-encoded premises into
    ///        a caller buffer — the sig-producer twin of the `IntEncodedExpr`
    ///        @ref recordUSignature, with no `OwnerSet` and no `std::set`.
    ///
    /// @details
    /// Byte-identical to the `(slot, argFullId)` pair sequence @ref recordUSignature
    /// builds before it range-constructs the signature vector: `slot` counts EVERY
    /// argument slot (matched or not) across all premises in order, and a pair is
    /// emitted only for a `u_`-marked slot (`argUnchangeable[j] != 0`), carrying the
    /// raw name id (`argFullId[j]`). The run is therefore slot-ascending, exactly
    /// the order the signature vector holds. @p hasUArg reports whether any pair was
    /// emitted — the same flag that drives `os.hasLooseOwner` in the oracle
    /// (`!hasUArg` -> loose). Pure fill into @p out (a caller-owned run on the
    /// arena / stack), so it reaches no interner and no arena itself: 0% heap.
    ///
    /// @param encList  The pre-encoded premise expressions.
    /// @param count    Their number.
    /// @param out      Caller-owned pair buffer of capacity @p cap; filled `[0, n)`.
    /// @param cap      Capacity of @p out (the caller's `count * MAX_ARITY`
    ///                 derivation); a loud Rule-19 assert fires on overflow.
    /// @param hasUArg  [out] `true` iff at least one pair was emitted.
    /// @return The number of pairs written (`n <= cap`).
    /// @see recordUSignature(OwnerSet&, const IntEncodedExpr*, int16_t) — the
    ///      retained oracle; mergeOwnerRecord (the raw-key overload) — the consumer.
    inline int32_t buildUSignatureRunInto(const IntEncodedExpr* encList,
                                          int16_t count,
                                          std::pair<int16_t, int16_t>* out,
                                          int32_t cap, bool& hasUArg) {
        int32_t n = 0;
        int16_t slot = 0;
        hasUArg = false;
        for (int16_t e = 0; e < count; ++e) {
            const IntEncodedExpr& expr = encList[e];
            const int16_t arity = std::min(expr.arity,
                static_cast<int16_t>(ExecutionParameters::MAX_ARITY));
            for (int16_t j = 0; j < arity; ++j) {
                if (expr.argUnchangeable[j] != 0) {
                    hasUArg = true;
                    assert(n < cap
                        && "buildUSignatureRunInto: signature run exceeds cap");
                    out[n].first = slot;
                    out[n].second = expr.argFullId[j];
                    ++n;
                }
                ++slot;
            }
        }
        return n;
    }

    /// @brief Inter-LB message bag — statements + implications + origin map
    /// queued for the receiver to absorb on its next hash burst.
    ///
    /// @details
    /// The mail subsystem is the *only* sanctioned cross-LB write channel
    /// during a parallel hash burst (per
    /// [I-28](../../docs/agentic_swdd/30_invariants.md#i-28) — "cross-LB writes during
    /// `proveKernel`'s parallel phase forbidden — defer to post-`pool.join()`
    /// collectors"). The flow direction is parent-to-children only; no
    /// upward writes (per [D-51](../../docs/agentic_swdd/40_decisions.md#d-51)).
    ///
    /// Three slots:
    /// - `statements`     — each element is `(ExpressionWithValidity, levels)`.
    ///   The EWV's `validityName` is `"main"` for routing-channel traffic
    ///   (`mailIn`/`mailOut`) — enforced at the receiver-side mail absorb's
    ///   per-item assert per [I-26](../../docs/agentic_swdd/30_invariants.md#i-26) —
    ///   and may be non-main for the per-LB integration-revival channel
    ///   `Memory::sameIterationInternalMail`. Migrated 2026-05-07 under
    ///   [D-53](../../docs/agentic_swdd/40_decisions.md#d-53); was previously
    ///   `pair<string, set<int>>` with validity hardcoded to `"main"` at
    ///   the receiver.
    /// - `exprOriginMap`  — origin-chain mirror, ALL-SCOPES per the same
    ///   invariant. Carries the provenance edges that the verifier needs
    ///   to reconstruct the proof tree on the receiver side.
    ///
    /// The legacy `implications` 5-tuple channel was removed in the ASIC
    /// 0.1 reshuffle: implications now travel only as the D-76 compact
    /// `(implication<N>[…])` statement (a `statements` element) and the
    /// receiver recovers the rule via `status=3` disintegration. There is
    /// no separate implications bag on `Mail` any more.
    ///
    /// @see [I-21](../../docs/agentic_swdd/30_invariants.md#i-21) — `sameIterationInternalMail`
    ///      is cleared immediately after its absorb in the relocated
    ///      post-fixpoint mail-absorption block.
    /// @see [I-26](../../docs/agentic_swdd/30_invariants.md#i-26) — channel-membership
    ///      rule for routed traffic.
    struct Mail {
        // statements element: pair<ExpressionWithValidity, levels>. The EWV's
        // validityName is "main" for routing-channel traffic (mailIn/mailOut)
        // — enforced at the receiver-side mail absorb's per-item assert (I-26) — and may
        // be non-main for the per-LB integration-revival channel
        // (Memory::sameIterationInternalMail). Migrated 2026-05-07 (D-53, renumbered from main's D-46 on merge into sandbox/incub_fix) — was previously
        // pair<string, set<int>>, with validity hardcoded to "main" at the
        // receiver.
        std::set< std::pair<ExpressionWithValidity, std::set<int> > > statements;
        // (Mail::implications removed — ASIC 0.1 reshuffle. Implications
        //  travel as the D-76 compact (implication<N>[…]) statement and
        //  are recovered receiver-side via status=3 disintegration.)

        // CHANGED: Tracking multiple origins (Vector of Pairs)
        std::map<ExpressionWithValidity, std::vector<std::pair<std::string, std::vector<ExpressionWithValidity>>>> exprOriginMap;

        // Per-statement disintegration signals for the internal-mail channel.
        // The hashburst rule-firing computes whether a produced head should be
        // (OR-)disintegrated, but the head travels via
        // sameIterationInternalMail.statements, which carries only levels. This
        // parallel map carries the two firing-time decisions so the post-burst
        // absorb feeds them into addExprToMemoryBlock instead of defaulting them
        // false (the default-false dropped allowOrDisintegration, leaving or0
        // facts un-disintegrated — no _ordis_ branches, no or convergence).
        // Keyed by the same ExpressionWithValidity as the statements entry;
        // absent => both false (restores prior behaviour for non-firing writers).
        struct DisintegrationFlags {
            bool doNotDisintegrate = false;
            bool allowOrDisintegration = false;
        };
        std::map<ExpressionWithValidity, DisintegrationFlags> disintegrationSignals;

        Mail() : statements(), exprOriginMap(), disintegrationSignals() {}
    };

    /// @brief Decode a `ColdMail`'s id-form statements into the string snapshot
    ///        the internal-mail absorb + `makeHeapMail` consume, canonical order.
    ///
    /// @details
    /// The internal channel stores `IntMailStatementKey` (NameMap ids + levels);
    /// this turns each back into `(ExpressionWithValidity, levels)` via the LB's
    /// `NameMap` and sorts by `std::pair<EWV, std::set<int>>::operator<` —
    /// byte-identical to the heap `std::set` iteration order the former string
    /// `ColdMail::sortedStatements` produced. The decode-to-string is a CONSUMER
    /// boundary (mail carries ids; only the reader materializes strings).
    ///
    /// @param cm The internal-mail mailbox.
    /// @param nm The owning LB's `NameMap` (the statements id-space).
    /// @return The statements `(expression+scope, levels)`, key-sorted.
    inline std::vector<std::pair<ExpressionWithValidity, std::set<int>>>
    decodeInternalMailStatements(const ColdMail& cm, const NameMap& nm) {
        std::vector<std::pair<ExpressionWithValidity, std::set<int>>> out;
        const int32_t n = cm.statements_.count();
        out.reserve(static_cast<size_t>(n));
        for (int32_t id = 1; id <= n; ++id) {
            const IntMailStatementKey k = cm.statements_.decodeKey(id);
            out.emplace_back(
                ExpressionWithValidity(
                    std::string(nm.decode(static_cast<int16_t>(k.originalId))),
                    std::string(nm.decode(static_cast<int16_t>(k.validityId)))),
                std::set<int>(k.levels.begin(), k.levels.end()));
        }
        std::sort(out.begin(), out.end());
        return out;
    }

    /// @brief Decode a `ColdMail`'s id-form origins into the string snapshot the
    ///        internal-mail absorb + `makeHeapMail` consume, EWV-key sorted.
    ///
    /// @details
    /// The internal channel stores `IntMailOrigin` runs keyed by a packed
    /// `originInterner` pair; this turns each key + record back into
    /// `(ExpressionWithValidity, vector<OriginLine>)` via the LB's `originInterner`
    /// (the `OriginTag` byte → its historical string through `decodeOrigin`, the
    /// packed dep keys → EWV pairs) and sorts the rows by
    /// `ExpressionWithValidity::operator<`. Per-key runs keep their stored
    /// (insertion) order.
    ///
    /// @param cm The internal-mail mailbox.
    /// @param oi The owning LB's `originInterner` (the origins id-space).
    /// @return The origin rows `(expression+scope, history lines)`, key-sorted.
    inline std::vector<std::pair<ExpressionWithValidity, std::vector<OriginLine>>>
    decodeInternalMailOrigins(const ColdMail& cm, const ValueInterner& oi) {
        std::vector<std::pair<ExpressionWithValidity,
            std::vector<OriginLine>>> out;
        const int32_t n = cm.origins_.count();
        out.reserve(static_cast<size_t>(n));
        for (int32_t id = 1; id <= n; ++id) {
            const int64_t key = cm.origins_.decodeKey(id);
            StrSpan eSpan, vSpan;
            decodeOriginKeyView(key, oi, eSpan, vSpan);
            std::pair<std::string, std::string> ev(eSpan.toStdString(), vSpan.toStdString());
            const std::vector<IntMailOrigin> recs = cm.origins_.recordsAt(id);
            std::vector<OriginLine> lines;
            lines.reserve(recs.size());
            for (const IntMailOrigin& r : recs) {
                lines.push_back(decodeOrigin(
                    IdOrigin{ static_cast<OriginTag>(r.tag), r.deps }, oi));
            }
            out.emplace_back(
                ExpressionWithValidity(std::move(ev.first), std::move(ev.second)),
                std::move(lines));
        }
        std::sort(out.begin(), out.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
        return out;
    }

    /// @brief Decode a `ColdMail`'s id-form disintegration signals into the string
    ///        snapshot `makeHeapMail` consumes, EWV-key sorted.
    ///
    /// @details
    /// The signals column keys on a packed `NameMap` pair; this unpacks each key,
    /// decodes the two ids via the LB's `NameMap`, and unpacks the stored byte
    /// into the two firing-time bools.
    ///
    /// @param cm The internal-mail mailbox.
    /// @param nm The owning LB's `NameMap` (the signals id-space).
    /// @return The rows `(expression+scope, {doNotDisintegrate,
    ///         allowOrDisintegration})`, key-sorted.
    inline std::vector<std::pair<ExpressionWithValidity, std::pair<bool, bool>>>
    decodeInternalMailDisintegrationSignals(const ColdMail& cm,
        const NameMap& nm) {
        std::vector<std::pair<ExpressionWithValidity,
            std::pair<bool, bool>>> out;
        const int32_t n = cm.disintegrationSignals_.count();
        out.reserve(static_cast<size_t>(n));
        for (int32_t id = 1; id <= n; ++id) {
            const int64_t key = cm.disintegrationSignals_.decodeKey(id);
            const int32_t origId =
                static_cast<int32_t>(static_cast<uint64_t>(key) >> 32);
            const int32_t validId =
                static_cast<int32_t>(key & 0xFFFFFFFFLL);
            const uint8_t packed = cm.disintegrationSignals_.valueAt(id);
            out.emplace_back(
                ExpressionWithValidity(
                    std::string(nm.decode(static_cast<int16_t>(origId))),
                    std::string(nm.decode(static_cast<int16_t>(validId)))),
                std::make_pair((packed & 1u) != 0u, (packed & 2u) != 0u));
        }
        std::sort(out.begin(), out.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
        return out;
    }

    /// @brief Materialize a heap `Mail` from an id-form `ColdMail` (canonical
    ///        order) — the internal-mail read boundary (sacred dump, tests).
    ///
    /// @details
    /// Rebuilds a heap `Mail` whose `std::set` / `std::map` re-impose the
    /// canonical `ExpressionWithValidity::operator<` order, from the three decode
    /// helpers above (which turn the stored ids back into strings via the LB's
    /// `NameMap` / `originInterner`): `statements` (with levels), `exprOriginMap`
    /// (origin runs), and `disintegrationSignals` (the two bools unpacked).
    /// The
    /// `ColdMail` twin of the routing `toHeap` helpers, except it carries
    /// `disintegrationSignals` (which `RoutingColdMail` drops). A free function
    /// (not a `ColdMail` member) because `ColdMail` lives in `memory_infra`, below
    /// `Mail` and the interners. Used at the off-hot-path boundaries (the sacred
    /// dump, tests); the production absorb reads the decode helpers directly.
    ///
    /// @param cm The internal-mail mailbox to snapshot.
    /// @param nm The owning LB's `NameMap` (statements / signals id-space).
    /// @param oi The owning LB's `originInterner` (origins id-space).
    /// @return The equivalent heap `Mail`.
    inline Mail makeHeapMail(const ColdMail& cm, const NameMap& nm,
        const ValueInterner& oi) {
        Mail m;
        for (const std::pair<ExpressionWithValidity, std::set<int>>& st
             : decodeInternalMailStatements(cm, nm))
            m.statements.insert(st);
        for (std::pair<ExpressionWithValidity, std::vector<OriginLine>>& og
             : decodeInternalMailOrigins(cm, oi))
            m.exprOriginMap.emplace(og.first, std::move(og.second));
        for (const std::pair<ExpressionWithValidity, std::pair<bool, bool>>& ds
             : decodeInternalMailDisintegrationSignals(cm, nm)) {
            Mail::DisintegrationFlags flags;
            flags.doNotDisintegrate = ds.second.first;
            flags.allowOrDisintegration = ds.second.second;
            m.disintegrationSignals.emplace(ds.first, flags);
        }
        return m;
    }

    /// @brief Insert a statement into an internal-mail `ColdMail`, interning the
    ///        EWV into the LB's `NameMap` — the id-form `insertStatement` wrapper.
    ///
    /// @details Encodes `(original, validityName)` into `nm` (single-threaded LB
    /// owner, mint legal) and forwards the ids to `ColdMail::insertStatement`. The
    /// transient EWV strings are NOT stored in mail — only their ids.
    ///
    /// @param cm     The internal-mail mailbox.
    /// @param nm     The owning LB's `NameMap`.
    /// @param ev     The statement's expression + scope.
    /// @param levels The statement's level set.
    inline void insertInternalStatement(ColdMail& cm, NameMap& nm,
        const ExpressionWithValidity& ev, const std::set<int>& levels) {
        cm.insertStatement(nm.encode(ev.original), nm.encode(ev.validityName),
                           levels);
    }

    /// @brief Span door of @ref insertInternalStatement — insert a statement
    ///        into an internal-mail `ColdMail` from `StrSpan`s over stable
    ///        input, with no transient `ExpressionWithValidity` materialized.
    ///
    /// @details
    /// Byte-identical deposit to the `ExpressionWithValidity` overload: the
    /// same two `NameMap::encode` mints — the span twin `encode(const StrSpan&)`
    /// interns the same bytes to the same id — in the same evaluation order,
    /// forwarded to the same `ColdMail::insertStatement`. Lets the internal-mail
    /// writers (`ordisMerge`, the parent-scope emissions, the equi-class rewrite
    /// hooks) stop building an `ExpressionWithValidity` whose two strings were
    /// only interned and dropped.
    ///
    /// @param cm           The internal-mail mailbox.
    /// @param nm           The owning LB's `NameMap`.
    /// @param original     Span over the expression bytes.
    /// @param validityName Span over the scope bytes.
    /// @param levels       The statement's level set.
    /// @invariant Span-lifetime rule (I-3), stated precisely so a caller can
    ///            decide: appends to the paged cold tables never relocate
    ///            existing bytes (pages are stable until deload / the
    ///            copying compaction), so a `decodeView` span SURVIVES
    ///            same-table appends; the real hazards are a deload (frees
    ///            pages outright) and aliasing a table a mint path does not
    ///            even write. Which table each mint path writes: a FLAT
    ///            `nm.encode` (delimiter-free name) writes only the names
    ///            table + `validityNodes`; only `encodePush` (and the
    ///            scoped-`encode` recursion that ends in it) writes the
    ///            sub-expression table. Hence the sanctioned S1c discharge
    ///            caller passes a sub-table span (`classifyOrScopeView` out)
    ///            across this door's two encodes — a lookup HIT plus a
    ///            provably FLAT mint, neither touching the sub-table. The
    ///            DEFAULT for an unanalyzed call site remains: span
    ///            caller-owned `std::string` / `ScratchString` buffers, or
    ///            copy / recover-from-id before the mint.
    /// @see insertInternalStatement(ColdMail&, NameMap&, const ExpressionWithValidity&, const std::set<int>&)
    ///      — the owning-string overload this reproduces byte-for-byte.
    inline void insertInternalStatement(ColdMail& cm, NameMap& nm,
        const StrSpan& original, const StrSpan& validityName,
        const std::set<int>& levels) {
        cm.insertStatement(nm.encode(original), nm.encode(validityName),
                           levels);
    }

    /// @brief Levels-RUN twin of the span @ref insertInternalStatement — the
    ///        statement key from spans, the level set from an
    ///        ascending-unique contiguous run.
    ///
    /// @details
    /// Byte-identical deposit to the `std::set<int>` overloads: the two
    /// `nm.encode` calls sit in the same argument positions (same mint
    /// sequence for never-before-seen names), and the run forwards to
    /// `ColdMail::insertStatement(const int*, int32_t)`, whose contract
    /// documents key-byte identity with the set form. Lets a caller holding
    /// a sorted-unique level run (an `intStatementLevelsMap` value run) feed
    /// internal mail with no transient `std::set<int>`.
    ///
    /// @pre `[levels, levels + levelCount)` is sorted ascending with no
    ///      duplicates (the CSR store contract of the source run).
    /// @param cm           The internal-mail mailbox.
    /// @param nm           The owning LB's `NameMap`.
    /// @param original     Span over the expression bytes.
    /// @param validityName Span over the scope bytes.
    /// @param levels       Pointer to `levelCount` ascending-unique levels;
    ///                     may be null only when `levelCount` is 0.
    /// @param levelCount   The level count (`>= 0`).
    /// @invariant Same span-lifetime rule as the span overload above: the
    ///            two encodes mint, so the spans must alias buffers the mint
    ///            paths do not relocate (caller-owned `std::string` /
    ///            `ScratchString`, or an analyzed-safe cold table).
    /// @see insertInternalStatement(ColdMail&, NameMap&, const StrSpan&, const StrSpan&, const std::set<int>&)
    ///      — the set-form span door; ColdMail::insertStatement — the run
    ///      door with the byte-order contract.
    inline void insertInternalStatement(ColdMail& cm, NameMap& nm,
        const StrSpan& original, const StrSpan& validityName,
        const int* levels, int32_t levelCount) {
        cm.insertStatement(nm.encode(original), nm.encode(validityName),
                           levels, levelCount);
    }

    /// @brief Serialize one `IntMailOrigin` history line into a caller buffer
    ///        — the stack-buffer twin of `Codec<IntMailOrigin>::serialize`.
    ///
    /// @details
    /// Emits the codec's exact canonical frame — `uint8 tag`, `int32 depCount`,
    /// `depCount x int64 dep` — byte-for-byte into `buf`, with the closed-form
    /// length `5 + 8 * depCount` asserted against both the dependency ceiling
    /// (`ExecutionParameters::kMaxOriginDeps`, a Rule-19 tripwire — realistic
    /// dependency counts are single-digit) and the buffer capacity, and an
    /// exact-fill assert after the write (the two-pass doctrine collapsed to a
    /// closed form). No heap, no arena.
    ///
    /// @param buf    Caller-owned destination buffer.
    /// @param cap    The buffer's capacity in bytes.
    /// @param record The `(tag, deps)` record to serialize.
    /// @return The blob's byte length (`5 + 8 * deps.size()`).
    /// @invariant The written bytes equal `Codec<IntMailOrigin>::serialize(record)`
    ///            exactly (twin test `serialize_mail_origin_to_matches_codec`).
    /// @see Codec<IntMailOrigin> — the heap oracle; addMailOriginRecord — the
    ///      consuming RMW.
    inline int32_t serializeMailOriginTo(char* buf, int32_t cap,
        const IntMailOrigin& record) {
        const int32_t depN = static_cast<int32_t>(record.deps.size());
        assert(depN <= ExecutionParameters::kMaxOriginDeps
            && "serializeMailOriginTo: dependency count exceeds kMaxOriginDeps "
               "(Rule-19 tripwire — widen the constant with evidence)");
        const int32_t len = 5 + 8 * depN;
        assert(len <= cap
            && "serializeMailOriginTo: blob exceeds the caller buffer capacity");
        char* at = buf;
        *at++ = static_cast<char>(record.tag);
        std::memcpy(at, &depN, sizeof(int32_t));
        at += sizeof(int32_t);
        for (const int64_t d : record.deps) {
            std::memcpy(at, &d, sizeof(int64_t));
            at += sizeof(int64_t);
        }
        assert(at == buf + len
            && "serializeMailOriginTo: fill diverged from the closed-form length");
        return len;
    }

    /// @brief POD overload of @ref serializeMailOriginTo — serialize a
    ///        `(tag, deps)` mail history line from raw fields, no `IntMailOrigin`
    ///        materialized.
    ///
    /// @details
    /// Emits the identical canonical frame as the `IntMailOrigin` overload —
    /// `uint8 tag`, `int32 depCount`, `depCount x int64 dep` — from a raw tag
    /// byte + int64 dependency run, so a caller that already holds the packed
    /// dependency keys never builds an `IntMailOrigin{tag, deps}` heap vector.
    /// Same closed-form length `5 + 8 * depN`, same Rule-19 `kMaxOriginDeps` and
    /// capacity asserts, same exact-fill assert.
    ///
    /// @param buf  Caller-owned destination buffer.
    /// @param cap  The buffer's capacity in bytes.
    /// @param tag  The history-line tag byte.
    /// @param deps Pointer to `depN` dependency packed keys, positional order.
    /// @param depN The dependency count.
    /// @return The blob's byte length (`5 + 8 * depN`).
    /// @invariant The written bytes equal `serializeMailOriginTo(buf, cap,
    ///            IntMailOrigin{tag, {deps..}})` exactly (twin test
    ///            `serialize_mail_origin_to_pod_matches_intmailorigin`).
    /// @see serializeMailOriginTo(char*, int32_t, const IntMailOrigin&) — the
    ///      `IntMailOrigin` overload this reproduces byte-for-byte.
    inline int32_t serializeMailOriginTo(char* buf, int32_t cap, uint8_t tag,
        const int64_t* deps, int32_t depN) {
        assert(depN <= ExecutionParameters::kMaxOriginDeps
            && "serializeMailOriginTo: dependency count exceeds kMaxOriginDeps "
               "(Rule-19 tripwire — widen the constant with evidence)");
        const int32_t len = 5 + 8 * depN;
        assert(len <= cap
            && "serializeMailOriginTo: blob exceeds the caller buffer capacity");
        char* at = buf;
        *at++ = static_cast<char>(tag);
        std::memcpy(at, &depN, sizeof(int32_t));
        at += sizeof(int32_t);
        for (int32_t i = 0; i < depN; ++i) {
            std::memcpy(at, &deps[i], sizeof(int64_t));
            at += sizeof(int64_t);
        }
        assert(at == buf + len
            && "serializeMailOriginTo: fill diverged from the closed-form length");
        return len;
    }

    /// @brief The shared D-49 cap-full preference RMW on an id-form mail origin
    ///        run — the engine behind `addInternalMailOrigin` /
    ///        `addDeloadableMailOutOrigin` / the `deserializeInto` origin fold.
    ///
    /// @details Below the cap, append `record` if absent; at the cap, a new
    /// non-`equality1`/`equality2` record displaces the first equality-convenience
    /// slot (foundation displaces convenience), else the existing slot wins. Record
    /// equality is `IntMailOrigin::operator==` (tag + packed deps); the interner is
    /// bijective so id-equality is string-equality, and the codec is injective so
    /// blob byte-equality IS record equality. The `OriginTag` comparison
    /// lives here (the enum is invisible to the mail headers). The caller supplies
    /// the key + record already in the column's id-space (sender `originInterner`
    /// for mailOut / the internal channel; global `mailInterner` for mailIn).
    ///
    /// The interior is heap-free on fixed-capacity STACK buffers (no arena, no
    /// per-slot scratch): the new record serializes once into a
    /// `kMaxOriginBlobBytes` frame; one scan peeks each existing blob through the
    /// caller-buffer `peekRecordBytes` door (dedup by whole-blob `memcmp` + the
    /// first convenience-slot index); the append paths take the raw
    /// `appendBlobToRun` / one-blob `assignRun` doors; the rare cap-full
    /// convenience-replace assembles survivors verbatim + the replacement into
    /// ONE `kMaxOriginRunBytes` stack frame and writes it back through the raw
    /// whole-run `assignRun` door — byte-identical to the retired
    /// decode-into-`std::vector`-reserialize cycle (twin test
    /// `add_mail_origin_record_rmw_matches_heap`). The rebuild's run-length
    /// assert against `kMaxOriginRunBlobs` is a Rule-19 tripwire (origin runs
    /// are config-capped at 30). Single-threaded write sites only (I-83).
    ///
    /// @param col        The id-form origin blob column.
    /// @param key        Packed `(origExprId, origValidId)` key in the column space.
    /// @param record     The id-form `(tag, deps)` record to insert.
    /// @param maxOrigins Cap.
    inline void addMailOriginRecord(TypedColdBlobMap<int64_t, IntMailOrigin>& col,
        int64_t key, const IntMailOrigin& record, int maxOrigins) {
        char newBlob[ExecutionParameters::kMaxOriginBlobBytes];
        const int32_t newLen = serializeMailOriginTo(
            newBlob, ExecutionParameters::kMaxOriginBlobBytes, record);

        const auto isEqualityConvenienceTag = [](uint8_t tag) {
            return tag == static_cast<uint8_t>(OriginTag::equality1)
                || tag == static_cast<uint8_t>(OriginTag::equality2);
        };

        // One scan over the existing run: duplicate detection (blob byte
        // equality == IntMailOrigin::operator==, the codec is injective) and
        // the first equality-convenience slot for the cap-full replace.
        const int32_t id = col.lookup(key);
        const int32_t rl = id ? col.runLen(id) : 0;
        int32_t convenienceIdx = -1;
        {
            char blobBuf[ExecutionParameters::kMaxOriginBlobBytes];
            for (int32_t j = 0; j < rl; ++j) {
                int32_t bl = 0;
                const char* bp = col.peekRecordBytes(id, j, bl, blobBuf,
                    ExecutionParameters::kMaxOriginBlobBytes);
                if (bl == newLen
                    && std::memcmp(bp, newBlob,
                                   static_cast<std::size_t>(bl)) == 0) {
                    return;  // duplicate — the no-op insert
                }
                if (convenienceIdx < 0
                    && isEqualityConvenienceTag(static_cast<uint8_t>(bp[0]))) {
                    convenienceIdx = j;
                }
            }
        }

        if (rl < maxOrigins) {
            // Below the cap: splice the one new blob at the run-end (existing
            // key) or open a one-blob run (new key) — the raw doors, byte-
            // identical to the typed appendRecord / one-record assignRun.
            if (id) col.inner().appendBlobToRun(id, newBlob, newLen);
            else col.inner().assignRun(key, newBlob, &newLen, 1);
            return;
        }

        // Cap-full origin-preference replacement (D-49): foundation displaces
        // convenience; a convenience-tag record or a run with no convenience
        // slot leaves the existing run untouched (existing wins).
        if (isEqualityConvenienceTag(record.tag)) return;
        if (convenienceIdx < 0) return;
        assert(rl <= ExecutionParameters::kMaxOriginRunBlobs
            && "addMailOriginRecord: cap-full run exceeds kMaxOriginRunBlobs "
               "(Rule-19 tripwire — widen the constant with evidence)");
        char runBuf[ExecutionParameters::kMaxOriginRunBytes];
        int32_t lens[ExecutionParameters::kMaxOriginRunBlobs];
        int32_t at = 0;
        for (int32_t j = 0; j < rl; ++j) {
            if (j == convenienceIdx) {
                std::memcpy(runBuf + at, newBlob,
                            static_cast<std::size_t>(newLen));
                lens[j] = newLen;
                at += newLen;
                continue;
            }
            int32_t bl = 0;
            const char* bp = col.peekRecordBytes(id, j, bl, runBuf + at,
                ExecutionParameters::kMaxOriginRunBytes - at);
            assert(at + bl <= ExecutionParameters::kMaxOriginRunBytes
                && "addMailOriginRecord: survivor run exceeds kMaxOriginRunBytes");
            if (bp != runBuf + at) {
                std::memcpy(runBuf + at, bp, static_cast<std::size_t>(bl));
            }
            lens[j] = bl;
            at += bl;
        }
        col.inner().assignRun(key, runBuf, lens, rl);
    }

    /// @brief POD overload of @ref addMailOriginRecord — append a `(tag, deps)`
    ///        mail history line with the D-49 cap-full RMW, no `IntMailOrigin`
    ///        materialized.
    ///
    /// @details
    /// Byte-identical to the `IntMailOrigin` overload: the new line serializes
    /// once via the POD `serializeMailOriginTo(buf, cap, tag, deps, depN)` (no
    /// `IntMailOrigin{tag, deps}` heap vector), then the SAME whole-blob-memcmp
    /// dedup scan, the SAME below-cap `appendBlobToRun` / one-blob `assignRun`
    /// splice, and the SAME D-49 cap-full convenience-replace assembling
    /// survivors verbatim + the replacement into one `kMaxOriginRunBytes` stack
    /// frame for a single raw whole-run `assignRun`. The convenience-tag guard
    /// reads @p tag directly. Single-threaded write sites only
    /// ([I-83](../../docs/agentic_swdd/30_invariants.md#i-83)).
    ///
    /// @param col        The cold `IntMailOrigin` blob map.
    /// @param key        Packed (expressionId, validityId) key.
    /// @param tag        The history-line tag byte.
    /// @param deps       Pointer to `depN` dependency packed keys, positional
    ///                   order.
    /// @param depN       The dependency count.
    /// @param maxOrigins Cap; caller picks the compressor or standard parameter.
    /// @invariant The resulting run equals the `IntMailOrigin` overload on the
    ///            equivalent record (twin test
    ///            `add_mail_origin_record_pod_matches_intmailorigin`).
    /// @see addMailOriginRecord(TypedColdBlobMap<int64_t, IntMailOrigin>&,
    ///      int64_t, const IntMailOrigin&, int) — the `IntMailOrigin` overload.
    inline void addMailOriginRecord(TypedColdBlobMap<int64_t, IntMailOrigin>& col,
        int64_t key, uint8_t tag, const int64_t* deps, int32_t depN,
        int maxOrigins) {
        char newBlob[ExecutionParameters::kMaxOriginBlobBytes];
        const int32_t newLen = serializeMailOriginTo(
            newBlob, ExecutionParameters::kMaxOriginBlobBytes, tag, deps, depN);

        const auto isEqualityConvenienceTag = [](uint8_t t) {
            return t == static_cast<uint8_t>(OriginTag::equality1)
                || t == static_cast<uint8_t>(OriginTag::equality2);
        };

        const int32_t id = col.lookup(key);
        const int32_t rl = id ? col.runLen(id) : 0;
        int32_t convenienceIdx = -1;
        {
            char blobBuf[ExecutionParameters::kMaxOriginBlobBytes];
            for (int32_t j = 0; j < rl; ++j) {
                int32_t bl = 0;
                const char* bp = col.peekRecordBytes(id, j, bl, blobBuf,
                    ExecutionParameters::kMaxOriginBlobBytes);
                if (bl == newLen
                    && std::memcmp(bp, newBlob,
                                   static_cast<std::size_t>(bl)) == 0) {
                    return;  // duplicate — the no-op insert
                }
                if (convenienceIdx < 0
                    && isEqualityConvenienceTag(static_cast<uint8_t>(bp[0]))) {
                    convenienceIdx = j;
                }
            }
        }

        if (rl < maxOrigins) {
            if (id) col.inner().appendBlobToRun(id, newBlob, newLen);
            else col.inner().assignRun(key, newBlob, &newLen, 1);
            return;
        }

        if (isEqualityConvenienceTag(tag)) return;
        if (convenienceIdx < 0) return;
        assert(rl <= ExecutionParameters::kMaxOriginRunBlobs
            && "addMailOriginRecord: cap-full run exceeds kMaxOriginRunBlobs "
               "(Rule-19 tripwire — widen the constant with evidence)");
        char runBuf[ExecutionParameters::kMaxOriginRunBytes];
        int32_t lens[ExecutionParameters::kMaxOriginRunBlobs];
        int32_t at = 0;
        for (int32_t j = 0; j < rl; ++j) {
            if (j == convenienceIdx) {
                std::memcpy(runBuf + at, newBlob,
                            static_cast<std::size_t>(newLen));
                lens[j] = newLen;
                at += newLen;
                continue;
            }
            int32_t bl = 0;
            const char* bp = col.peekRecordBytes(id, j, bl, runBuf + at,
                ExecutionParameters::kMaxOriginRunBytes - at);
            assert(at + bl <= ExecutionParameters::kMaxOriginRunBytes
                && "addMailOriginRecord: survivor run exceeds kMaxOriginRunBytes");
            if (bp != runBuf + at) {
                std::memcpy(runBuf + at, bp, static_cast<std::size_t>(bl));
            }
            lens[j] = bl;
            at += bl;
        }
        col.inner().assignRun(key, runBuf, lens, rl);
    }

    /// @brief Append a history line to an internal-mail `ColdMail`'s origin run —
    ///        encodes `(ev, origin)` into the LB's `originInterner`, then the
    ///        shared cap-full RMW.
    ///
    /// @param cm         The internal-mail mailbox.
    /// @param oi         The owning LB's `originInterner`.
    /// @param ev         `(expression, scope)` key.
    /// @param origin     `(tag, antecedents)` string record.
    /// @param maxOrigins Cap; caller picks the compressor or standard parameter.
    inline void addInternalMailOrigin(ColdMail& cm, ValueInterner& oi,
        const ExpressionWithValidity& ev, const OriginLine& origin,
        int maxOrigins) {
        // Already key-first: mint the KEY, resolve the tag via the span twin
        // (interns nothing), then mint the EWV antecedents positionally — no
        // IdOrigin / IntMailOrigin materialized.
        const int64_t key = mintOriginKey(oi, ev.original, ev.validityName);
        const OriginTag tag = originTagFromString(StrSpan(origin.first));
        int64_t d[ExecutionParameters::kMaxOriginDeps];
        const int32_t n = mintOriginDepsFromEWVInto(origin.second, oi, d,
            ExecutionParameters::kMaxOriginDeps);
        addMailOriginRecord(cm.origins_, key, static_cast<uint8_t>(tag), d, n,
            maxOrigins);
    }

    /// @brief Span door of @ref addInternalMailOrigin — key the history-line
    ///        deposit from `StrSpan`s over stable input, with no transient
    ///        `ExpressionWithValidity` built for the key.
    ///
    /// @details
    /// Byte-identical deposit to the `ExpressionWithValidity` overload: the same
    /// `mintOriginKey` span twin over the same key bytes, the tag resolved via
    /// the span `originTagFromString`, the EWV antecedents minted positionally
    /// via `mintOriginDepsFromEWVInto`, and the same POD cap-full RMW
    /// (`addMailOriginRecord`) — no `IdOrigin` / `IntMailOrigin` materialized.
    /// The caller's `origin` is still a heap `OriginLine` (borrowed by `const&`),
    /// read directly. The internal-mail writers whose key `ev` was built only to
    /// be interned-then-dropped use this.
    ///
    /// @param cm           The internal-mail mailbox.
    /// @param oi           The owning LB's `originInterner`.
    /// @param original     Span over the expression bytes (key).
    /// @param validityName Span over the scope bytes (key).
    /// @param origin       `(tag, antecedents)` string record.
    /// @param maxOrigins   Cap; caller picks the compressor or standard
    ///                     parameter.
    /// @invariant The spans must alias buffers OTHER than @p oi's own byte store
    ///            (the callers span caller-owned `std::string` / `ScratchString`);
    ///            `mintOriginKey` mints, so a span into `oi` would dangle across
    ///            the mint (I-3).
    /// @see addInternalMailOrigin(ColdMail&, ValueInterner&, const ExpressionWithValidity&, const OriginLine&, int)
    ///      — the owning-string overload this reproduces byte-for-byte.
    inline void addInternalMailOrigin(ColdMail& cm, ValueInterner& oi,
        const StrSpan& original, const StrSpan& validityName,
        const OriginLine& origin, int maxOrigins) {
        // Already key-first: mint the KEY, resolve the tag via the span twin
        // (interns nothing), then mint the EWV antecedents positionally — no
        // IdOrigin / IntMailOrigin materialized.
        const int64_t key = mintOriginKey(oi, original, validityName);
        const OriginTag tag = originTagFromString(StrSpan(origin.first));
        int64_t d[ExecutionParameters::kMaxOriginDeps];
        const int32_t n = mintOriginDepsFromEWVInto(origin.second, oi, d,
            ExecutionParameters::kMaxOriginDeps);
        addMailOriginRecord(cm.origins_, key, static_cast<uint8_t>(tag), d, n,
            maxOrigins);
    }

    /// @brief Span-antecedent twin of @ref addInternalMailOrigin — key AND
    ///        record built from spans + an `OriginTag`, no transient
    ///        `ExpressionWithValidity` / `OriginLine` materialized.
    ///
    /// @details
    /// Byte-identical deposit to the `OriginLine` overload: the same key mint
    /// (`mintOriginKey` span twin, FIRST — matching the twin's statement order,
    /// so the `originInterner` mint sequence is preserved), then the dependency
    /// ids are minted positionally via `mintOriginDepsInto` and fed to the POD
    /// cap-full `addMailOriginRecord` RMW — no transient `IdOrigin` /
    /// `IntMailOrigin` materialized. This is the L3 flip of the deferred
    /// origin-antecedent boundary the L4 span-key door left string.
    ///
    /// @param cm           The internal-mail mailbox.
    /// @param oi           The owning LB's `originInterner`.
    /// @param original     Span over the key expression bytes.
    /// @param validityName Span over the key scope bytes.
    /// @param tag          The history-line tag (the enumerator the site names).
    /// @param deps         Pointer to `depN` antecedent spans, positional order.
    /// @param depN         The antecedent count.
    /// @param maxOrigins   Cap; caller picks the compressor or standard
    ///                     parameter.
    /// @invariant The spans must alias buffers OTHER than @p oi's own byte store
    ///            ([I-3](../../docs/agentic_swdd/30_invariants.md#i-3)).
    /// @see addInternalMailOrigin(ColdMail&, ValueInterner&, const StrSpan&, const StrSpan&, const OriginLine&, int)
    ///      — the span-key/`OriginLine`-record overload this reproduces
    ///      byte-for-byte; encodeOriginSpans.
    inline void addInternalMailOrigin(ColdMail& cm, ValueInterner& oi,
        const StrSpan& original, const StrSpan& validityName,
        OriginTag tag, const OriginDep* deps, int depN, int maxOrigins) {
        // Key-first (already the twin's order): mint the KEY, then the deps
        // positionally — no IdOrigin / IntMailOrigin materialized.
        const int64_t key = mintOriginKey(oi, original, validityName);
        int64_t d[ExecutionParameters::kMaxOriginDeps];
        const int32_t n = mintOriginDepsInto(deps, depN, oi, d,
            ExecutionParameters::kMaxOriginDeps);
        addMailOriginRecord(cm.origins_, key, static_cast<uint8_t>(tag), d, n,
            maxOrigins);
    }

    /// @brief Append a history line to a deloadable `mailOut` origin run.
    ///
    /// @details Encodes the key and dependencies into the mailbox's dedicated
    /// per-LB interner, then applies the shared cap-full RMW. The commit seam
    /// later translates this private id space into global mail ids.
    ///
    /// @param mo         The deloadable routing outbox.
    /// @param oi         The outbox's dedicated `mailOutInterner`.
    /// @param ev         `(expression, scope)` key.
    /// @param origin     `(tag, antecedents)` string record.
    /// @param maxOrigins Cap.
    inline void addDeloadableMailOutOrigin(DeloadableMailOut& mo,
        ValueInterner& oi,
        const ExpressionWithValidity& ev, const OriginLine& origin,
        int maxOrigins) {
        const int64_t key = mintOriginKey(oi, ev.original, ev.validityName);
        const OriginTag tag = originTagFromString(StrSpan(origin.first));
        int64_t d[ExecutionParameters::kMaxOriginDeps];
        const int32_t n = mintOriginDepsFromEWVInto(origin.second, oi, d,
            ExecutionParameters::kMaxOriginDeps);
        addMailOriginRecord(mo.origins_, key, static_cast<uint8_t>(tag), d, n,
            maxOrigins);
    }

    /// @brief Span-antecedent twin of @ref addDeloadableMailOutOrigin — key and record
    ///        built from spans + an `OriginTag`, no transient
    ///        `ExpressionWithValidity` / `OriginLine` materialized.
    ///
    /// @details
    /// Byte-identical deposit to the `ExpressionWithValidity` / `OriginLine`
    /// overload: `mo.ensureArena()` FIRST (matching the twin's statement order),
    /// then the same key mint (`mintOriginKey` span twin, so the mint sequence
    /// is preserved), then the dependency ids minted positionally via
    /// `mintOriginDepsInto` and fed to the POD cap-full `addMailOriginRecord`
    /// RMW — no transient `IdOrigin` / `IntMailOrigin` materialized. Every id
    /// belongs to the mailbox's private deloadable interner.
    ///
    /// @param mo           The routing outbox.
    /// @param oi           The producing LB's `mailOutInterner`.
    /// @param original     Span over the key expression bytes.
    /// @param validityName Span over the key scope bytes.
    /// @param tag          The history-line tag (the enumerator the site names).
    /// @param deps         Pointer to `depN` antecedent spans, positional order.
    /// @param depN         The antecedent count.
    /// @param maxOrigins   Cap.
    /// @invariant The spans must alias buffers OTHER than @p oi's own byte store
    ///            ([I-3](../../docs/agentic_swdd/30_invariants.md#i-3)).
    /// @see addDeloadableMailOutOrigin(DeloadableMailOut&, ValueInterner&, const ExpressionWithValidity&, const OriginLine&, int)
    ///      — the owning-string overload this reproduces byte-for-byte;
    ///      encodeOriginSpans.
    inline void addDeloadableMailOutOrigin(DeloadableMailOut& mo,
        ValueInterner& oi,
        const StrSpan& original, const StrSpan& validityName,
        OriginTag tag, const OriginDep* deps, int depN, int maxOrigins) {
        const int64_t key = mintOriginKey(oi, original, validityName);
        int64_t d[ExecutionParameters::kMaxOriginDeps];
        const int32_t n = mintOriginDepsInto(deps, depN, oi, d,
            ExecutionParameters::kMaxOriginDeps);
        addMailOriginRecord(mo.origins_, key, static_cast<uint8_t>(tag), d, n,
            maxOrigins);
    }

    /// @brief Set an internal-mail `ColdMail` statement's disintegration signals,
    ///        packing the EWV's `NameMap` pair — the id-form wrapper.
    ///
    /// @param cm  The internal-mail mailbox.
    /// @param nm  The owning LB's `NameMap`.
    /// @param ev  The statement's expression + scope.
    /// @param dnd `doNotDisintegrate`.
    /// @param aod `allowOrDisintegration`.
    inline void setInternalDisintegrationSignal(ColdMail& cm, NameMap& nm,
        const ExpressionWithValidity& ev, bool dnd, bool aod) {
        cm.setDisintegrationSignal(
            packOriginKey(nm.encode(ev.original), nm.encode(ev.validityName)),
            dnd, aod);
    }

    /// @brief Span door of @ref setInternalDisintegrationSignal — set a
    ///        statement's disintegration signals from `StrSpan`s, no
    ///        transient `ExpressionWithValidity` built for the key.
    ///
    /// @details
    /// Byte-identical to the `ExpressionWithValidity` overload: the same two
    /// `nm.encode` calls in the same argument-evaluation order (at the
    /// callers both are idempotent re-encodes — the statement insert above
    /// already minted both names, exactly as the EWV path re-encoded them),
    /// the identical packed key, the identical signal write.
    ///
    /// @param cm           The internal-mail mailbox.
    /// @param nm           The owning LB's `NameMap`.
    /// @param original     Span over the expression bytes (key).
    /// @param validityName Span over the scope bytes (key).
    /// @param dnd          `doNotDisintegrate`.
    /// @param aod          `allowOrDisintegration`.
    /// @invariant Same span-lifetime rule as the sibling span doors
    ///            (`insertInternalStatement`): the two encodes mint, so the
    ///            spans must alias buffers the mint paths do not relocate
    ///            (caller-owned `std::string` / `ScratchString` / sealed
    ///            pages — I-3).
    /// @see setInternalDisintegrationSignal(ColdMail&, NameMap&, const ExpressionWithValidity&, bool, bool)
    ///      — the owning-string overload this reproduces byte-for-byte.
    inline void setInternalDisintegrationSignal(ColdMail& cm, NameMap& nm,
        const StrSpan& original, const StrSpan& validityName,
        bool dnd, bool aod) {
        cm.setDisintegrationSignal(
            packOriginKey(nm.encode(original), nm.encode(validityName)),
            dnd, aod);
    }

    /// @brief Syntactic tier of a bare variable name, as the equivalence-class
    /// machinery sees it.
    ///
    /// @details
    /// Canonical-member selection and iteration filtering partition names into
    /// three tiers, matching exactly the two precompiled patterns historically
    /// inlined in `prover.hpp::filterIterations`, `chooseCanonical` and
    /// `canonicalizeUnderClasses`:
    /// - `IntLev` — whole-string match of `int_lev_<digits>_<digits>`.
    /// - `ItLev`  — whole-string match of `it_<digits>_lev_<digits>_<digits>`.
    /// - `Normal` — everything else (concrete values, `repl_*`, user names).
    ///
    /// The underlying numeric values exist only for compact storage inside
    /// [`EqClassNameCaches`](#eqclassnamecaches); they are never persisted or
    /// compared across runs.
    ///
    /// @see classifyName — the single classification authority.
    enum class NameKind : uint8_t { Normal = 0, IntLev = 1, ItLev = 2 };

    /// @brief Classify one bare name into its [`NameKind`](#namekind) tier.
    ///
    /// @details
    /// Whole-string (`std::regex_match`) semantics: `"int_lev_3_4"` is
    /// `IntLev`, but `"print_lev_3_4"` and the truncated `"int_lev_3"` are
    /// `Normal` — only a name that is nothing but the pattern qualifies.
    /// This mirrors byte-for-byte the class-member bucketing the string-side
    /// helpers performed with their own static regexes; any change here
    /// changes canonical-member selection and is therefore a Rule-8
    /// architectural change.
    ///
    /// @param name The bare variable name to classify (never a whole
    ///             expression — expressions go through scanSpecialTokens).
    /// @return The matching tier; `NameKind::Normal` when neither pattern
    ///         matches the whole string.
    /// @see EqClassNameCaches::kindOf — the id-keyed memoization layer.
    inline NameKind classifyName(const std::string& name) {
        static const std::regex PATTERN_INT(R"(int_lev_\d+_\d+)");
        static const std::regex PATTERN_IT(R"(it_\d+_lev_\d+_\d+)");
        if (std::regex_match(name, PATTERN_INT)) return NameKind::IntLev;
        if (std::regex_match(name, PATTERN_IT))  return NameKind::ItLev;
        return NameKind::Normal;
    }

    /// @brief Span twin of `classifyName(const std::string&)` — classify one
    ///        bare name into its [`NameKind`](#namekind) tier, zero-heap.
    ///
    /// @details
    /// Byte-identical verdict to the string overload for every input: the two
    /// anchored regexes are replaced by their pure lexical shape scanners
    /// `isIntLevShape` / `isItLevShape` (`memory_infra/str_ops.hpp`), whose
    /// backtrack-free equivalence argument lives on those functions. The
    /// check order is preserved from the regex form (`int` first, then `it`
    /// — the two shapes are also mutually exclusive: third byte `t` vs `_`),
    /// and an empty span falls out `Normal` via the scanners' prefix length
    /// checks. Chosen over the value-parsing matchers (`matchIntLevId` /
    /// `matchItLevId`, whose over-`INT_MAX` `false` diverges from the regex
    /// verdict) and over `std::regex` on a span (per-match allocation);
    /// shape-only classification needs neither the values nor the heap.
    ///
    /// @param name The bare variable name span to classify (never a whole
    ///             expression — expressions go through `scanSpecialTokens`).
    /// @return The matching tier; `NameKind::Normal` when neither shape
    ///         matches the whole span.
    /// @see classifyName(const std::string&) — the canonical overload
    ///      (retained test oracle); isIntLevShape, isItLevShape;
    ///      EqClassNameCaches::kindOf — the id-keyed memoization layer.
    inline NameKind classifyName(const StrSpan& name) {
        if (isIntLevShape(name)) return NameKind::IntLev;
        if (isItLevShape(name))  return NameKind::ItLev;
        return NameKind::Normal;
    }

    /// @brief Result of one substring scan of an expression for the special
    /// `int_lev_*` / `it_*_lev_*` tokens.
    ///
    /// @details
    /// Two separate token lists because the consumer
    /// (`prover.hpp::filterIterations`) runs two independent passes — an
    /// `int_lev` pass and an `it_lev` pass — and the two must stay
    /// distinguishable. Tokens appear in left-to-right scan order and
    /// duplicates are kept; the membership checks downstream are boolean, so
    /// duplicates are harmless, and keeping them preserves the historical
    /// scan faithfully.
    ///
    /// This struct remains the CODEC / ORACLE interchange: the
    /// `Codec<SpecialTokenScan>` record type and the unit tests' decoded
    /// form. Production reads ride its zero-copy sibling
    /// `SpecialTokenScanView` over the same canonical blob.
    ///
    /// @see scanSpecialTokens — producer (retained test oracle).
    /// @see SpecialTokenScanView — the zero-copy read sibling.
    /// @see EqClassNameCaches::tokensViewOf — the per-expression-id cache.
    struct SpecialTokenScan {
        std::vector<std::string> intTokens;
        std::vector<std::string> itTokens;
    };

    /// @brief Scan an expression string for embedded `int_lev_*` /
    /// `it_*_lev_*` tokens.
    ///
    /// @details
    /// SUBSTRING search (`std::sregex_iterator`), not whole-string match —
    /// deliberately so. The historical `filterIterations` scanned the whole
    /// expression text with `sregex_iterator`, which finds `"int_lev_3_4"`
    /// inside `"print_lev_3_4"`; this function reproduces that behaviour
    /// exactly so the cached-token path is bit-compatible with the direct
    /// scan. The scan is a pure function of @p text and is therefore safe to
    /// memoize per NameMap id without any invalidation story.
    ///
    /// RETAINED as the unit-test regex ORACLE (the `classifyName(std::string)`
    /// precedent): production reads ride `EqClassNameCaches::tokensViewOf`,
    /// whose miss path produces the identical token sequences via the
    /// regex-free occurrence scanners (`scanIntLevOccurrences` /
    /// `scanItLevOccurrences`, `memory_infra/str_ops.hpp`) — this function
    /// keeps no production caller.
    ///
    /// @param text Full expression text (e.g. `"(in3[int_lev_0_1,x,v,+])"`).
    /// @return Both token lists in left-to-right scan order, duplicates kept.
    /// @see SpecialTokenScan, EqClassNameCaches::tokensViewOf,
    ///      scanIntLevOccurrences, scanItLevOccurrences.
    inline SpecialTokenScan scanSpecialTokens(const std::string& text) {
        static const std::regex PATTERN_INT(R"(int_lev_\d+_\d+)");
        static const std::regex PATTERN_IT(R"(it_\d+_lev_\d+_\d+)");
        SpecialTokenScan scan;
        for (std::sregex_iterator i(text.begin(), text.end(), PATTERN_INT), end; i != end; ++i) {
            scan.intTokens.push_back(i->str());
        }
        for (std::sregex_iterator i(text.begin(), text.end(), PATTERN_IT), end; i != end; ++i) {
            scan.itTokens.push_back(i->str());
        }
        return scan;
    }

    /// @brief Serialize a `SpecialTokenScan` into a canonical byte blob — the
    ///        codec the `eqClassNameCaches` token memo stores on its cold map.
    ///
    /// @details
    /// Fixed-width little-endian, length-prefixed strings: int32 intToken count
    /// then each `(int32 length, bytes)`, then the same for `itTokens`. A pure
    /// function of the scan (itself a pure function of the decoded text), so the
    /// cold form is deterministic. The memo is derived (reconstructible from
    /// `nameMap`) but IS deload-enrolled: `LbMemory::visitContainers` splices
    /// it in at the reserved base (`kEqClassNameCachesDeloadBase`, facets
    /// 705+0..4 — see `eq_class_name_caches.hpp`), so this canonical blob
    /// layout is byte-observable in the deload stream, not merely a
    /// convention (the `I-134` byte contract rests on
    /// exactly that observability).
    ///
    /// @param scan The token scan to serialize.
    /// @return The blob; `deserializeSpecialTokenScan` is its inverse.
    /// @see deserializeSpecialTokenScan, scanSpecialTokens.
    inline std::vector<char> serializeSpecialTokenScan(
        const SpecialTokenScan& scan)
    {
        std::vector<char> out;
        const auto putCount = [&out](int32_t n) {
            const char* p = reinterpret_cast<const char*>(&n);
            out.insert(out.end(), p, p + sizeof(int32_t));
        };
        const auto putStr = [&out, &putCount](const std::string& s) {
            putCount(static_cast<int32_t>(s.size()));
            out.insert(out.end(), s.begin(), s.end());
        };
        putCount(static_cast<int32_t>(scan.intTokens.size()));
        for (const std::string& s : scan.intTokens) putStr(s);
        putCount(static_cast<int32_t>(scan.itTokens.size()));
        for (const std::string& s : scan.itTokens) putStr(s);
        return out;
    }

    /// @brief Reconstruct a `SpecialTokenScan` from its canonical blob — the
    ///        exact inverse of `serializeSpecialTokenScan`.
    ///
    /// @details
    /// A short blob (a count or string overrunning the end) and a long blob
    /// (bytes left after the last token) are both hard asserts (Rule 19), never
    /// a truncated decode.
    ///
    /// @param data Blob bytes.
    /// @param n    Blob length.
    /// @return The reconstructed scan.
    /// @see serializeSpecialTokenScan.
    inline SpecialTokenScan deserializeSpecialTokenScan(const char* data,
                                                        int32_t n)
    {
        SpecialTokenScan scan;
        const char* cur = data;
        const char* const end = data + n;
        const auto getCount = [&cur, end]() -> int32_t {
            assert(cur + sizeof(int32_t) <= end
                && "deserializeSpecialTokenScan: blob shorter than the layout");
            int32_t v = 0;
            std::memcpy(&v, cur, sizeof(int32_t));
            cur += sizeof(int32_t);
            return v;
        };
        const auto getStr = [&cur, end, &getCount]() -> std::string {
            const int32_t len = getCount();
            assert(len >= 0 && cur + len <= end
                && "deserializeSpecialTokenScan: string overruns the blob");
            std::string s(cur, cur + len);
            cur += len;
            return s;
        };
        const int32_t intCount = getCount();
        scan.intTokens.reserve(static_cast<std::size_t>(intCount));
        for (int32_t i = 0; i < intCount; ++i) scan.intTokens.push_back(getStr());
        const int32_t itCount = getCount();
        scan.itTokens.reserve(static_cast<std::size_t>(itCount));
        for (int32_t i = 0; i < itCount; ++i) scan.itTokens.push_back(getStr());
        assert(cur == end
            && "deserializeSpecialTokenScan: blob longer than the layout");
        return scan;
    }

    /// @brief Blob codec for `SpecialTokenScan` — the `TypedColdBlobMap<int16_t,
    ///        SpecialTokenScan>` record serializer (`EqClassNameCaches`).
    ///
    /// @details Forwards to the free `serializeSpecialTokenScan` /
    /// `deserializeSpecialTokenScan` (the `Codec<EquivalenceClass>` pattern), so
    /// the typed blob map stores one canonical `SpecialTokenScan` blob per id.
    template <>
    struct Codec<SpecialTokenScan> {
        /// @brief Serialize a scan to its canonical blob.
        /// @param s The token scan.
        /// @return The canonical byte blob.
        static std::vector<char> serialize(const SpecialTokenScan& s) {
            return serializeSpecialTokenScan(s);
        }
        /// @brief Deserialize a scan from its blob.
        /// @param p Blob bytes.
        /// @param n Blob length.
        /// @return The reconstructed scan.
        static SpecialTokenScan deserialize(const char* p, int32_t n) {
            return deserializeSpecialTokenScan(p, n);
        }
    };

    /// @brief Zero-copy read view over one canonical `SpecialTokenScan` blob
    ///        — the no-allocation consumer boundary that replaces the heap
    ///        `deserializeSpecialTokenScan` decode on the token-memo read
    ///        path.
    ///
    /// @details
    /// Mirrors the `serializeSpecialTokenScan` byte layout one-for-one —
    /// `[i32 nInt][(i32 len, bytes) x nInt][i32 nIt][(i32 len, bytes) x nIt]`,
    /// little-endian `memcpy` reads — exactly as `EquivalenceClassView`
    /// mirrors `serializeEquivalenceClass` and `OwnerSetBlob` its codec.
    /// The token sections are walked by a SEQUENTIAL `TokenCursor` (not an
    /// indexed `tokenAt(i)`) because the consumer
    /// (`prover.hpp::filterIterationsCore`) visits tokens in order with an
    /// early exit — the cursor keeps the probe sequence structurally
    /// identical to the retired vector loop.
    ///
    /// Parameter/stack-only value ([I-116] discipline): the view aliases the
    /// cold blob pool (or a page-straddle copy on the caller's scratch
    /// arena) and is valid only until the next `tokensByExprId_` mutation or
    /// the caller's arena rewind, whichever comes first — never stored.
    ///
    /// @invariant Layout-coupled to `serializeSpecialTokenScan` — the two
    ///            are edited together.
    /// @see serializeSpecialTokenScan, deserializeSpecialTokenScan (the
    ///      codec + test oracle), EqClassNameCaches::tokensViewOf — the
    ///      producer, EquivalenceClassView — the pattern sibling.
    struct SpecialTokenScanView {
        const char* p = nullptr;   ///< One canonical scan blob (serialize layout).
        int32_t len = 0;           ///< The blob's byte length.

        /// @brief Read a little-endian `int32_t` at byte pointer `q`.
        /// @param q Source pointer (>= 4 readable bytes).
        /// @return The decoded value.
        static int32_t rdI32(const char* q) {
            int32_t v; std::memcpy(&v, q, sizeof(int32_t)); return v;
        }

        /// @brief Forward walk over one length-prefixed token section.
        ///
        /// @details Each `next` decodes one `(i32 len, bytes)` entry into a
        /// span aliasing the blob and advances past it; `false` when the
        /// section's `remaining` count is exhausted. Copyable value — a
        /// fresh cursor restarts the walk.
        struct TokenCursor {
            const char* cur = nullptr;   ///< Next entry's length prefix.
            int32_t remaining = 0;       ///< Entries left in the section.

            /// @brief Decode the next token; advance the cursor.
            /// @param out Receives a span over the token's blob bytes.
            /// @return `true` when a token was produced; `false` at
            ///         section end (`out` untouched).
            bool next(StrSpan& out) {
                if (remaining == 0) return false;
                const int32_t tokenLen = rdI32(cur);
                assert(tokenLen >= 0);
                out = StrSpan(cur + 4, tokenLen);
                cur += 4 + tokenLen;
                --remaining;
                return true;
            }
        };

        /// @brief Number of `int_lev_*` tokens.
        /// @return The int-section count (blob offset 0).
        int32_t intCount() const { return rdI32(p); }

        /// @brief Cursor over the `int_lev_*` token section.
        /// @return A cursor positioned at the first int token.
        TokenCursor intTokens() const { return TokenCursor{ p + 4, intCount() }; }

        /// @brief Byte offset of the it section's count — derived by
        ///        walking past the variable-length int section.
        /// @return The offset into the blob.
        int32_t itOffset() const {
            int32_t off = 4;
            const int32_t n = intCount();
            for (int32_t i = 0; i < n; ++i) off += 4 + rdI32(p + off);
            return off;
        }

        /// @brief Number of `it_*_lev_*` tokens.
        /// @return The it-section count.
        int32_t itCount() const { return rdI32(p + itOffset()); }

        /// @brief Cursor over the `it_*_lev_*` token section.
        /// @return A cursor positioned at the first it token.
        TokenCursor itTokens() const {
            const int32_t off = itOffset();
            return TokenCursor{ p + off + 4, rdI32(p + off) };
        }
    };

    /// @brief Out-of-line `EqClassNameCaches::kindOf` — declared in
    ///        `memory_infra/eq_class_name_caches.hpp`, defined here so `NameMap`
    ///        + `classifyName` are complete.
    /// @param id NameMap id of a bare variable name; must be a real id (`>0`).
    /// @param nm The owning LB's NameMap (read-only; no minting).
    /// @return The name's tier.
    inline NameKind EqClassNameCaches::kindOf(int16_t id, const NameMap& nm) {
        assert(id > 0);
        while (kindById_.size() <= id) kindById_.push_back(KIND_UNCOMPUTED);
        if (kindById_[id] == KIND_UNCOMPUTED)
            kindById_.setAt(id,
                static_cast<uint8_t>(classifyName(nm.decodeView(id))));
        return static_cast<NameKind>(kindById_[id]);
    }

    /// @brief Out-of-line `EqClassNameCaches::tokensViewOf` — declared in
    ///        `memory_infra/eq_class_name_caches.hpp`, defined here so
    ///        `NameMap`, the occurrence scanners, and `SpecialTokenScanView`
    ///        are complete.
    ///
    /// @details
    /// Memoized token scan of `nm.decode(exprId)`, zero-heap on hit AND miss.
    /// HIT: the single stored record is peeked in place
    /// (`peekRecordBytes` — zero-copy; a page straddle is assembled once
    /// onto @p peekArena's byte-bump tier) and wrapped as a view. MISS: the
    /// canonical blob is built DIRECTLY on @p peekArena via the regex-free
    /// occurrence scanners in two passes (count, then exact-length fill —
    /// asserted), byte-identical to
    /// `serializeSpecialTokenScan(scanSpecialTokens(text))` by the scanners'
    /// iterator-twin contract + the layout coupling; the blob is installed
    /// through the raw `inner().assignRun` byte door (identity `int16_t`
    /// key) and the INSTALLED record is re-peeked — one uniform
    /// view-construction path whose re-peek validates the install. Same
    /// misses in the same caller order mint the same keys, so the memo's
    /// deload facets (705+1..4) stay byte-identical to the retired heap
    /// path (I-134).
    ///
    /// The memo purity/reset discipline is unchanged (I-88 clause 3): a
    /// pure function of `nm.decode(exprId)`, never wiped on scope teardown,
    /// reset only with `destroyGrid`'s NameMap reset. Filling mutates the
    /// memo, so calls run only in per-LB-exclusive phases (one worker owns
    /// the LB) — never from phase-2 parallel split parts.
    ///
    /// NO internal mark/rewind: the straddle copy / miss build buffer must
    /// outlive the return, so the CALLER owns the arena window (marks
    /// before, pops after the consumer finishes).
    ///
    /// `decodeView` lifetime: the text span is consumed by the scanners +
    /// fill BEFORE `assignRun`, and `assignRun` mutates `tokensByExprId_`,
    /// not the NameMap ([I-3] different-container rule).
    ///
    /// @param exprId    NameMap id of a full expression string; must be `>0`.
    /// @param nm        The owning LB's NameMap (read-only; no minting).
    /// @param peekArena Byte-bump arena for straddle copies and the miss
    ///                  build; the caller pops it after consuming the view.
    /// @return View over the memoized canonical blob — valid until the next
    ///         `tokensByExprId_` mutation or the caller's arena pop,
    ///         whichever comes first.
    /// @invariant Blob bytes byte-identical to the retired
    ///            `scanSpecialTokens` + codec path
    ///            (I-134); single-record runs asserted.
    /// @see scanSpecialTokens — the retained regex oracle;
    ///      serializeSpecialTokenScan, SpecialTokenScanView,
    ///      `prover.hpp::filterIterations` — the consumer funnel.
    inline SpecialTokenScanView EqClassNameCaches::tokensViewOf(int16_t exprId,
        const NameMap& nm, ScratchArena& peekArena) {
        assert(exprId > 0);
        const int32_t cid = tokensByExprId_.lookup(exprId);
        if (cid != 0) {
            assert(tokensByExprId_.runLen(cid) == 1
                && "tokensViewOf: token-memo runs are single-record by construction");
            int32_t blen = 0;
            const char* bp =
                tokensByExprId_.peekRecordBytes(cid, 0, blen, peekArena);
            return SpecialTokenScanView{ bp, blen };
        }

        // Miss: direct canonical-blob build on the caller's byte-bump tier —
        // [i32 nInt][(i32 len, bytes)…][i32 nIt][(i32 len, bytes)…].
        const StrSpan text = nm.decodeView(exprId);

        // Pass 1 — count.
        int32_t nInt = 0, sumIntLen = 0;
        scanIntLevOccurrences(text, [&](int32_t, int32_t tokenLen) {
            ++nInt;
            sumIntLen += tokenLen;
        });
        int32_t nIt = 0, sumItLen = 0;
        scanItLevOccurrences(text, [&](int32_t, int32_t tokenLen) {
            ++nIt;
            sumItLen += tokenLen;
        });
        const int32_t blobLen = 8 + 4 * nInt + sumIntLen + 4 * nIt + sumItLen;
        char* blob = peekArena.allocBytes(blobLen);

        // Pass 2 — exact-length fill (same scans, same order).
        int32_t at = 0;
        const auto putI32 = [&blob, &at](int32_t v) {
            std::memcpy(blob + at, &v, sizeof(int32_t));
            at += static_cast<int32_t>(sizeof(int32_t));
        };
        putI32(nInt);
        scanIntLevOccurrences(text, [&](int32_t start, int32_t tokenLen) {
            putI32(tokenLen);
            std::memcpy(blob + at, text.ptr + start,
                        static_cast<size_t>(tokenLen));
            at += tokenLen;
        });
        putI32(nIt);
        scanItLevOccurrences(text, [&](int32_t start, int32_t tokenLen) {
            putI32(tokenLen);
            std::memcpy(blob + at, text.ptr + start,
                        static_cast<size_t>(tokenLen));
            at += tokenLen;
        });
        assert(at == blobLen
            && "tokensViewOf: exact-length blob fill diverged from the count pass");

        // Raw byte door: Codec<int16_t> is the identity key codec, so the
        // KeyView IS the id. Then re-peek the INSTALLED record — one uniform
        // view path, and the re-peek validates the install.
        const int32_t nid =
            tokensByExprId_.inner().assignRun(exprId, blob, &blobLen, 1);
        int32_t blen = 0;
        const char* bp = tokensByExprId_.peekRecordBytes(nid, 0, blen, peekArena);
        return SpecialTokenScanView{ bp, blen };
    }

    /// @brief Pack an UNORDERED pair of NameMap ids into a single `uint32_t`
    /// key for the equivalence-class per-pair levels map.
    ///
    /// @details
    /// Normalizes the pair by numeric id order (smaller id in the high 16
    /// bits) so `{a,b}` and `{b,a}` produce the same key — the identity the
    /// old `std::set<std::string>` map key encoded. The numeric normalization
    /// is identity-only: nothing ever iterates these keys in a way that
    /// reaches the proof, so id order is permitted here (the
    /// [I-84](../../docs/agentic_swdd/30_invariants.md#i-84) "never sort by
    /// id" rule constrains ORDER-sensitive walks, not key identity).
    /// `a == b` packs naturally (high == low).
    ///
    /// @param a NameMap id of one pair element.
    /// @param b NameMap id of the other pair element.
    /// @return Combined key; unique per unordered pair because both inputs
    ///         are ids minted by the same `NameMap`.
    /// @see EquivalenceClass::intEqualityLevelsMap — the keyed container.
    inline uint32_t packEqPairKey(int16_t a, int16_t b) {
        const uint16_t ua = static_cast<uint16_t>(a);
        const uint16_t ub = static_cast<uint16_t>(b);
        const uint16_t lo = ua < ub ? ua : ub;
        const uint16_t hi = ua < ub ? ub : ua;
        return (static_cast<uint32_t>(lo) << 16) | static_cast<uint32_t>(hi);
    }

    /// @brief One equivalence class — a set of variables known to be equal
    /// in some scope, together with provenance.
    ///
    /// @details
    /// Equivalence classes are the prover's internal representation of
    /// `(=[a,b])` chains. Whenever `addEquality` fires it either creates a
    /// fresh class with two variables or merges existing classes; whenever
    /// `applyEquivalenceClass` rewrites an expression it routes through this
    /// data structure to look up which variable to substitute.
    ///
    /// The slots
    /// ([D-134](../../docs/agentic_swdd/40_decisions.md#d-134)):
    /// - `memberIds`           — the class members as NameMap ids, sorted by
    ///   DECODED name (never by id): the first member of a tier is that
    ///   tier's lex-min — the canonical-selection order. Membership is
    ///   symmetric and transitive; insertion order is not preserved.
    /// - `intEqualityLevelsMap` — per pair of members (packEqPairKey
    ///   unordered id-pair keys), the set of LB levels at which the
    ///   equality is admissible. Used by `applyEquivalenceClass` to gate
    ///   substitution by level.
    /// - `equalityOriginMap`   — origin chains for each `(expression, scope)`
    ///   pair that this class produced. Carries multiple origins per pair
    ///   so the verifier can reconstruct the full provenance tree. Id form
    ///   per `D-131`: same `IdOriginMap` shape and
    ///   the same per-LB `Memory::originInterner` space as `exprOriginMap`,
    ///   so the class->body history copies are pure id operations.
    ///
    /// Class storage lives in `Memory::equivalenceClassesMap` (keyed by an
    /// internal class-id string). Mutations are subject to:
    /// - [I-30](../../docs/agentic_swdd/30_invariants.md#i-30) —
    ///   `applyEquivalenceClassToRejectedMapIntegration` is additive; the
    ///   original entries are never erased.
    /// - [I-31](../../docs/agentic_swdd/30_invariants.md#i-31) —
    ///   `updateEquivalenceClasses` ancestor-pass never modifies ancestor-scope
    ///   class state.
    /// - [I-32](../../docs/agentic_swdd/30_invariants.md#i-32) — cross-pair `equality2`
    ///   emission gated on existing class / LB origin.
    /// - [I-33](../../docs/agentic_swdd/30_invariants.md#i-33) —
    ///   `mergeTwoEquivalenceClasses` cross-vN preconditions.
    /// - [I-34](../../docs/agentic_swdd/30_invariants.md#i-34) — cross-substitution
    ///   `equality1` emission gated on existing target origin.
    ///
    /// @see [`Memory::equivalenceClassesMap`](#memory) — owns these.
    /// @see `prover.hpp::applyEquivalenceClass` — consumer.
    /// @see `prover.hpp::mergeTwoEquivalenceClasses` — merger.
    struct EquivalenceClass {
        // Class members as NameMap ids, sorted by DECODED name — never by
        // id (I-84). First member of a tier is that tier's lex-min: the
        // canonical-selection order (D-134).
        std::vector<int16_t> memberIds;

        // Per-pair admission levels, keyed by packEqPairKey unordered id
        // pairs.
        std::map<uint32_t, std::set<int> > intEqualityLevelsMap;

        // Tracking multiple origins. Id form in the owning LB's
        // originInterner space (D-131) — the same
        // space as Memory::exprOriginMap, so syncs/copies between the two
        // never re-encode.
        IdOriginMap equalityOriginMap;

        EquivalenceClass() : memberIds(), intEqualityLevelsMap(), equalityOriginMap() {}

        /// @brief Set the member list from a set of bare names.
        ///
        /// @details Write-site helper (`nm.encode` may mint). A
        /// `std::set<std::string>`'s iteration order IS decoded-lex order,
        /// so `memberIds` comes out sorted by decoded name by construction
        /// — no sort call, no id-order dependence
        /// ([I-84](../../docs/agentic_swdd/30_invariants.md#i-84)).
        ///
        /// @param names The member names (typically a new equality's args).
        /// @param nm    The owning LB's NameMap (mint side).
        /// @see unionMemberIdsByName — the merge-side counterpart.
        void setMembersFromNames(const std::set<std::string>& names, NameMap& nm) {
            memberIds.clear();
            memberIds.reserve(names.size());
            for (const std::string& n : names) {
                memberIds.push_back(nm.encode(n));
            }
        }

        /// @brief Member count — the `EquivalenceClassView::memberCount` twin, so
        ///        the equi-class helpers template uniformly over the heap class
        ///        and the zero-copy blob view (equi-7 read cascade).
        /// @return The number of members.
        int32_t memberCount() const {
            return static_cast<int32_t>(memberIds.size());
        }

        /// @brief Member id at index @p i (decoded-lex storage order) — the
        ///        `EquivalenceClassView::memberId` twin.
        /// @param i Index in `[0, memberCount())`.
        /// @return The member's NameMap id.
        int16_t memberId(int32_t i) const {
            return memberIds[static_cast<std::size_t>(i)];
        }

        /// @brief Visit the ascending level ints stored for one packed pair key —
        ///        the `EquivalenceClassView::forEachLevel` twin (the
        ///        `intEqualityLevelsMap.find` + `std::set<int>` iteration, which
        ///        is ascending).
        /// @param key Packed pair key (`packEqPairKey`).
        /// @param fn  Callable `(int level)`, invoked per level in ascending order.
        template <typename Fn>
        void forEachLevel(uint32_t key, Fn&& fn) const {
            const auto it = intEqualityLevelsMap.find(key);
            if (it == intEqualityLevelsMap.end()) return;
            for (const int lv : it->second) fn(lv);
        }
    };

    /// @brief Union of two member-id vectors, each sorted by decoded name;
    ///        the result keeps that order.
    ///
    /// @details Two-pointer merge comparing DECODED names — a string
    /// compare, never an id compare
    /// ([I-84](../../docs/agentic_swdd/30_invariants.md#i-84)). Ids are
    /// NameMap-bijective, so equal names imply equal ids; ties advance both
    /// inputs once and the assert pins the bijection. Pure read of @p nm —
    /// no minting.
    ///
    /// @param a  First member list, decoded-lex sorted.
    /// @param b  Second member list, decoded-lex sorted.
    /// @param nm NameMap for decoding (read-only).
    /// @return The union, decoded-lex sorted, duplicates collapsed.
    /// @see EquivalenceClass::setMembersFromNames — the creation-side
    ///      counterpart; `prover.hpp::mergeTwoEquivalenceClasses` —
    ///      consumer.
    inline std::vector<int16_t> unionMemberIdsByName(const std::vector<int16_t>& a,
        const std::vector<int16_t>& b,
        const NameMap& nm)
    {
        std::vector<int16_t> out;
        out.reserve(a.size() + b.size());
        std::size_t i = 0, j = 0;
        while (i < a.size() && j < b.size()) {
            if (a[i] == b[j]) { out.push_back(a[i]); ++i; ++j; continue; }
            const StrSpan nameA = nm.decodeView(a[i]);
            const StrSpan nameB = nm.decodeView(b[j]);
            assert(!equalSpans(nameA, nameB));
            if (compareSpans(nameA, nameB) < 0) { out.push_back(a[i]); ++i; }
            else               { out.push_back(b[j]); ++j; }
        }
        while (i < a.size()) out.push_back(a[i++]);
        while (j < b.size()) out.push_back(b[j++]);
        return out;
    }

    /// @brief Serialize one `EquivalenceClass` into its canonical byte blob —
    ///        the write boundary of the `ColdBlobMap`-backed
    ///        `equivalenceClassesMap`.
    ///
    /// @details
    /// The codec the record value store (`BlobCsrValueStore`) stores opaque
    /// bytes for. The byte stream is a PURE FUNCTION OF LOGICAL CONTENT
    /// (I-103): two logically-equal classes — same
    /// members, same per-pair levels, same origin history — serialize
    /// byte-identically regardless of how their hash containers were built. This
    /// codec is the SINGLE determinism point; every hash-backed field is emitted
    /// in a content-derived order. All integers are fixed-width little-endian
    /// (every GL target is x64 LE, the deload-format assumption).
    ///
    /// Layout (in order):
    /// - `memberIds`: int32 count, then count × int16 — verbatim (already
    ///   decoded-lex sorted by construction, I-84).
    /// - `intEqualityLevelsMap`: int32 K, then K × { uint32 packedKey, int32 m,
    ///   m × int32 level } — the `std::map` iterates ascending by key and each
    ///   `std::set<int>` ascending, so the canonical order is free.
    /// - `equalityOriginMap`: int32 G, then G × { int64 originKey, int32 L,
    ///   L × { uint8 OriginTag, int32 d, d × int64 depKey } } — the map is an
    ///   `unordered_map`, so its keys are SORTED ASCENDING before emit (the one
    ///   place bucket order would otherwise leak); the per-key history-line
    ///   vector and each line's dependency vector are insertion-ordered and
    ///   OBSERVABLE (dump / chapter export), so they are preserved verbatim.
    ///
    /// @param cls The class to serialize.
    /// @return The canonical blob; `deserializeEquivalenceClass` is its exact
    ///         inverse.
    /// @see deserializeEquivalenceClass, `BlobCsrValueStore`,
    ///      D-169.
    inline std::vector<char> serializeEquivalenceClass(
        const EquivalenceClass& cls)
    {
        std::vector<char> out;
        const auto put = [&out](const auto& v) {
            const char* p = reinterpret_cast<const char*>(&v);
            out.insert(out.end(), p, p + sizeof(v));
        };
        put(static_cast<int32_t>(cls.memberIds.size()));
        for (const int16_t id : cls.memberIds) put(id);

        put(static_cast<int32_t>(cls.intEqualityLevelsMap.size()));
        for (const auto& kv : cls.intEqualityLevelsMap) {
            put(static_cast<uint32_t>(kv.first));
            put(static_cast<int32_t>(kv.second.size()));
            for (const int level : kv.second) put(static_cast<int32_t>(level));
        }

        std::vector<int64_t> originKeys;
        originKeys.reserve(cls.equalityOriginMap.size());
        for (const auto& kv : cls.equalityOriginMap)
            originKeys.push_back(kv.first);
        std::sort(originKeys.begin(), originKeys.end());
        put(static_cast<int32_t>(originKeys.size()));
        for (const int64_t originKey : originKeys) {
            put(originKey);
            const std::vector<IdOrigin>& lines =
                cls.equalityOriginMap.at(originKey);
            put(static_cast<int32_t>(lines.size()));
            for (const IdOrigin& line : lines) {
                put(static_cast<uint8_t>(line.first));
                put(static_cast<int32_t>(line.second.size()));
                for (const int64_t dep : line.second) put(dep);
            }
        }
        return out;
    }

    /// @brief Serialize one `EquivalenceClass` into its canonical byte blob on a
    ///        caller `ScratchArena` — the 0%-heap arena-fill twin of
    ///        `serializeEquivalenceClass`.
    ///
    /// @details
    /// Byte-for-byte identical to `serializeEquivalenceClass` (the retained heap
    /// codec + twin oracle) with the ONLY interior change being the sink and the
    /// origin-key sort scratch: the `std::vector<char>` return becomes a
    /// caller-arena byte-bump buffer, and the `std::vector<int64_t>` origin-key
    /// sort scratch becomes an arena `int64_t*`. Neither touches a byte VALUE or
    /// an emit ORDER, so the I-103 canonical-bytes contract (the single
    /// determinism point that reaches the `.deload/` stream) is preserved to the
    /// byte. It reads the SAME `const EquivalenceClass&` fields in the SAME
    /// order: `memberIds` verbatim (decoded-lex by construction, I-84); the
    /// `intEqualityLevelsMap` `std::map` ascending with each `std::set<int>`
    /// ascending; the `equalityOriginMap` keys sorted ascending via
    /// `std::sort(ok, ok + G)` (identical to the heap `std::sort(originKeys)`, an
    /// `unordered_map` whose bucket order must not leak); per-key history lines
    /// and per-line deps verbatim (insertion order, observable).
    ///
    /// Two passes: a size pass (arithmetic mirror of the layout, no fill) then a
    /// fill pass writing the identical `put()` sequence into one arena
    /// allocation; `assert(at == total)` pins the two passes together (the
    /// exact-length doctrine). The origin-key sort scratch @c ok and the result
    /// @c buf are byte-bump allocations on the SAME @p arena; byte-bump blocks
    /// are stable (only page compaction relocates, I-107), so both stay valid to
    /// the caller's scope. NO nested `ScratchScope` — a rewind would free @c buf.
    /// The result rides @p arena until the caller's outer `releaseAll`; it is
    /// spanned into the mail-merge run splice synchronously before any rewind.
    /// No stack cap (a class blob is unbounded at FTA scale — an arena is the
    /// correct tier).
    ///
    /// @param arena Caller-owned scratch arena receiving the sort scratch and the
    ///              blob bytes (byte-bump tier).
    /// @param cls   The class to serialize.
    /// @return A `StrSpan` over the canonical blob on @p arena;
    ///         `deserializeEquivalenceClass` is its exact inverse.
    /// @invariant Layout-coupled to `serializeEquivalenceClass` — the two MUST be
    ///            edited together; any byte-layout change that skips this twin
    ///            fails `serialize_equivalence_class_into_matches_heap`.
    /// @see serializeEquivalenceClass — the retained heap codec + byte-twin
    ///      oracle; deserializeEquivalenceClass; I-103.
    inline StrSpan serializeEquivalenceClassInto(ScratchArena& arena,
                                                 const EquivalenceClass& cls)
    {
        // Sort the unordered_map origin keys ascending on the arena byte-bump
        // tier — byte-identical to the heap std::vector<int64_t> sort (int64
        // ascending is total-ordered).
        const int32_t G = static_cast<int32_t>(cls.equalityOriginMap.size());
        int64_t* ok = nullptr;
        if (G > 0) {
            ok = reinterpret_cast<int64_t*>(arena.resolve(
                arena.alloc(G * static_cast<int32_t>(sizeof(int64_t)),
                            static_cast<int32_t>(alignof(int64_t)))));
            int32_t oi = 0;
            for (const auto& kv : cls.equalityOriginMap) ok[oi++] = kv.first;
            std::sort(ok, ok + G);
        }

        // Size pass: arithmetic mirror of the put() sequence below (no fill).
        int64_t total = 0;
        total += 4;                                         // memberCount
        total += 2 * static_cast<int64_t>(cls.memberIds.size());
        total += 4;                                         // level-key count K
        for (const auto& kv : cls.intEqualityLevelsMap) {
            total += 4;                                     // packedKey uint32
            total += 4;                                     // m int32
            total += 4 * static_cast<int64_t>(kv.second.size());
        }
        total += 4;                                         // origin-key count G
        for (int32_t i = 0; i < G; ++i) {
            total += 8;                                     // originKey int64
            const std::vector<IdOrigin>& lines =
                cls.equalityOriginMap.at(ok[i]);
            total += 4;                                     // line count L
            for (const IdOrigin& line : lines) {
                total += 1;                                 // tag uint8
                total += 4;                                 // dep count d
                total += 8 * static_cast<int64_t>(line.second.size());
            }
        }

        // Fill pass: identical put() sequence into one arena allocation.
        char* buf = reinterpret_cast<char*>(arena.resolve(
            arena.alloc(static_cast<int32_t>(total), 1)));
        int32_t at = 0;
        const auto put = [&buf, &at](const auto& v) {
            std::memcpy(buf + at, &v, sizeof(v));
            at += static_cast<int32_t>(sizeof(v));
        };
        put(static_cast<int32_t>(cls.memberIds.size()));
        for (const int16_t id : cls.memberIds) put(id);

        put(static_cast<int32_t>(cls.intEqualityLevelsMap.size()));
        for (const auto& kv : cls.intEqualityLevelsMap) {
            put(static_cast<uint32_t>(kv.first));
            put(static_cast<int32_t>(kv.second.size()));
            for (const int level : kv.second) put(static_cast<int32_t>(level));
        }

        put(static_cast<int32_t>(G));
        for (int32_t i = 0; i < G; ++i) {
            const int64_t originKey = ok[i];
            put(originKey);
            const std::vector<IdOrigin>& lines =
                cls.equalityOriginMap.at(originKey);
            put(static_cast<int32_t>(lines.size()));
            for (const IdOrigin& line : lines) {
                put(static_cast<uint8_t>(line.first));
                put(static_cast<int32_t>(line.second.size()));
                for (const int64_t dep : line.second) put(dep);
            }
        }
        assert(at == static_cast<int32_t>(total));
        return StrSpan(buf, static_cast<int32_t>(total));
    }

    /// @brief Reconstruct an `EquivalenceClass` from its canonical byte blob —
    ///        the read boundary of the `ColdBlobMap`-backed
    ///        `equivalenceClassesMap`.
    ///
    /// @details
    /// The exact inverse of `serializeEquivalenceClass`: the working struct is a
    /// transient heap object decoded at the read boundary (the I-84 pattern —
    /// only the cold store holds bytes). Lossless — every field is fully
    /// reconstructed — so `serialize(deserialize(b)) == b` for any blob `b` this
    /// codec produced. A blob shorter than the layout demands or longer than it
    /// consumes is a hard `assert` (Rule 19), never a truncated decode.
    ///
    /// @param data The blob bytes.
    /// @param n    The blob length.
    /// @return The reconstructed class.
    /// @see serializeEquivalenceClass.
    inline EquivalenceClass deserializeEquivalenceClass(const char* data,
                                                        int32_t n)
    {
        EquivalenceClass cls;
        const char* cur = data;
        const char* const end = data + n;
        const auto get = [&cur, end](auto& v) {
            assert(cur + sizeof(v) <= end
                && "deserializeEquivalenceClass: blob shorter than the layout");
            std::memcpy(&v, cur, sizeof(v));
            cur += sizeof(v);
        };
        int32_t memberCount = 0;
        get(memberCount);
        cls.memberIds.reserve(static_cast<size_t>(memberCount));
        for (int32_t i = 0; i < memberCount; ++i) {
            int16_t id = 0; get(id); cls.memberIds.push_back(id);
        }

        int32_t levelKeyCount = 0;
        get(levelKeyCount);
        for (int32_t i = 0; i < levelKeyCount; ++i) {
            uint32_t packedKey = 0; get(packedKey);
            int32_t m = 0; get(m);
            std::set<int>& levels = cls.intEqualityLevelsMap[packedKey];
            for (int32_t j = 0; j < m; ++j) {
                int32_t level = 0; get(level);
                levels.insert(static_cast<int>(level));
            }
        }

        int32_t originKeyCount = 0;
        get(originKeyCount);
        for (int32_t i = 0; i < originKeyCount; ++i) {
            int64_t originKey = 0; get(originKey);
            int32_t lineCount = 0; get(lineCount);
            std::vector<IdOrigin>& lines = cls.equalityOriginMap[originKey];
            lines.reserve(static_cast<size_t>(lineCount));
            for (int32_t j = 0; j < lineCount; ++j) {
                uint8_t tag = 0; get(tag);
                int32_t depCount = 0; get(depCount);
                std::vector<int64_t> deps;
                deps.reserve(static_cast<size_t>(depCount));
                for (int32_t d = 0; d < depCount; ++d) {
                    int64_t dep = 0; get(dep); deps.push_back(dep);
                }
                lines.emplace_back(static_cast<OriginTag>(tag),
                                   std::move(deps));
            }
        }
        assert(cur == end
            && "deserializeEquivalenceClass: blob longer than the layout");
        return cls;
    }

    /// @brief Zero-copy read view over one `EquivalenceClass`'s canonical byte
    ///        blob — the no-allocation read boundary that replaces the heap
    ///        `deserializeEquivalenceClass` decode at the equi-class processing
    ///        transient sites.
    ///
    /// @details
    /// Mirrors the `serializeEquivalenceClass` byte layout one-for-one (keep the
    /// two in sync — a layout change there changes the offsets here), exactly as
    /// `OwnerSetBlob` mirrors `Codec<OwnerSet>`. The view wraps the `(p, len)` of
    /// a single class's blob — obtained from the `equivalenceClassesMap` blob map
    /// via `TypedCold::peekBlobContiguous` / `peekBlobAt` — and reads members /
    /// per-pair levels / origin history straight off the arena, never
    /// materializing the `std::vector` / `std::map` / `IdOriginMap` a full
    /// `EquivalenceClass` carries. Members and per-pair levels are the hot reads;
    /// origins are needed only by the merge path.
    ///
    /// Layout (from `serializeEquivalenceClass`):
    /// - `memberIds`: int32 count, then count × int16 (decoded-lex sorted, I-84).
    /// - `intEqualityLevelsMap`: int32 K, then K × { uint32 packedKey, int32 m,
    ///   m × int32 level } (ascending by key, each level set ascending).
    /// - `equalityOriginMap`: int32 G, then G × { int64 originKey, int32 L,
    ///   L × { uint8 OriginTag, int32 d, d × int64 depKey } } (keys ascending).
    ///
    /// Every read is a little-endian `memcpy` (every GL target is x64 LE, the
    /// deload-format assumption). Read-only and burst-safe
    /// ([I-83](docs/agentic_swdd/30_invariants.md#i-83)); the pointer stays valid
    /// only while the owning LB is resident and the blob run is not mutated (the
    /// same contract as `OwnerSetBlob`) — the view is a parameter-only value, never
    /// stored across a class-map mutation.
    ///
    /// @invariant Layout-coupled to `serializeEquivalenceClass` — edited together.
    /// @see serializeEquivalenceClass, OwnerSetBlob,
    ///      TypedCold::peekBlobContiguous, EquivalenceClass.
    struct EquivalenceClassView {
        const char* p = nullptr;   ///< The class blob's bytes (serialize layout).
        int32_t len = 0;           ///< The blob's byte length.

        /// @brief Read a little-endian `int32_t` at byte pointer `q`.
        /// @param q Source pointer (>= 4 readable bytes).
        /// @return The decoded value.
        static int32_t rdI32(const char* q) {
            int32_t v; std::memcpy(&v, q, sizeof(int32_t)); return v;
        }
        /// @brief Read a little-endian `int16_t` at byte pointer `q`.
        /// @param q Source pointer (>= 2 readable bytes).
        /// @return The decoded value.
        static int16_t rdI16(const char* q) {
            int16_t v; std::memcpy(&v, q, sizeof(int16_t)); return v;
        }
        /// @brief Read a little-endian `uint32_t` at byte pointer `q`.
        /// @param q Source pointer (>= 4 readable bytes).
        /// @return The decoded value.
        static uint32_t rdU32(const char* q) {
            uint32_t v; std::memcpy(&v, q, sizeof(uint32_t)); return v;
        }
        /// @brief Read a little-endian `int64_t` at byte pointer `q`.
        /// @param q Source pointer (>= 8 readable bytes).
        /// @return The decoded value.
        static int64_t rdI64(const char* q) {
            int64_t v; std::memcpy(&v, q, sizeof(int64_t)); return v;
        }

        /// @brief Number of class members.
        /// @return The `memberIds` count.
        int32_t memberCount() const { return rdI32(p); }

        /// @brief Member id at index `i`, in decoded-lex storage order.
        /// @param i Index in `[0, memberCount())`.
        /// @return The NameMap id of that member.
        int16_t memberId(int32_t i) const { return rdI16(p + 4 + 2 * i); }

        /// @brief Byte offset of the levels section (its int32 key count comes
        ///        first) — just past the member run.
        /// @return The offset into the blob.
        int32_t levelsOffset() const { return 4 + 2 * memberCount(); }

        /// @brief Visit each per-pair-levels entry in stored (ascending-key)
        ///        order.
        ///
        /// @param fn Callable `(uint32_t packedKey, const char* levelBytes,
        ///           int32_t levelCount)`; `levelBytes` points at `levelCount`
        ///           consecutive little-endian int32 levels (read via `rdI32`).
        template <typename Fn>
        void forEachLevelKey(Fn&& fn) const {
            int32_t off = levelsOffset();
            const int32_t k = rdI32(p + off); off += 4;
            for (int32_t i = 0; i < k; ++i) {
                const uint32_t key = rdU32(p + off); off += 4;
                const int32_t m = rdI32(p + off); off += 4;
                fn(key, p + off, m);
                off += 4 * m;
            }
        }

        /// @brief Byte offset of the origins section (its int32 key count comes
        ///        first) — derived by walking past the variable-length levels
        ///        section.
        /// @return The offset into the blob.
        int32_t originsOffset() const {
            int32_t off = levelsOffset();
            const int32_t k = rdI32(p + off); off += 4;
            for (int32_t i = 0; i < k; ++i) {
                off += 4;                          // packedKey
                const int32_t m = rdI32(p + off); off += 4;
                off += 4 * m;                      // m × int32 level
            }
            return off;
        }

        /// @brief Visit each origin history line (flattened over keys), in stored
        ///        order.
        ///
        /// @param fn Callable `(int64_t originKey, OriginTag tag,
        ///           const char* depBytes, int32_t depCount)`; `depBytes` points
        ///           at `depCount` consecutive little-endian int64 dependency keys
        ///           (read via `rdI64`).
        template <typename Fn>
        void forEachOriginLine(Fn&& fn) const {
            int32_t off = originsOffset();
            const int32_t g = rdI32(p + off); off += 4;
            for (int32_t i = 0; i < g; ++i) {
                const int64_t originKey = rdI64(p + off); off += 8;
                const int32_t lineCount = rdI32(p + off); off += 4;
                for (int32_t j = 0; j < lineCount; ++j) {
                    const OriginTag tag =
                        static_cast<OriginTag>(static_cast<uint8_t>(*(p + off)));
                    off += 1;
                    const int32_t d = rdI32(p + off); off += 4;
                    fn(originKey, tag, p + off, d);
                    off += 8 * d;
                }
            }
        }

        /// @brief Point lookup of one packed-pair-key's level run — the
        ///        `intEqualityLevelsMap.find(...)` twin for the merge path.
        ///
        /// @details Linear scan of the (small) levels section; a class carries
        /// few per-pair entries. Read-only.
        ///
        /// @param key      Packed pair key (`packEqPairKey`).
        /// @param outBytes [out] Pointer to `outCount` consecutive little-endian
        ///                 int32 levels when found; untouched otherwise.
        /// @param outCount [out] The level count when found; untouched otherwise.
        /// @return `true` iff `key` is present.
        bool findLevels(uint32_t key, const char*& outBytes,
                        int32_t& outCount) const {
            int32_t off = levelsOffset();
            const int32_t k = rdI32(p + off); off += 4;
            for (int32_t i = 0; i < k; ++i) {
                const uint32_t kk = rdU32(p + off); off += 4;
                const int32_t m = rdI32(p + off); off += 4;
                if (kk == key) { outBytes = p + off; outCount = m; return true; }
                off += 4 * m;
            }
            return false;
        }

        /// @brief Whether the class carries any origin history line under
        ///        `originKey` — the `equalityOriginMap.count(pk) > 0` twin.
        ///
        /// @param originKey Packed (expressionId, validityId) origin key.
        /// @return `true` iff at least one line is stored under `originKey`.
        bool hasOrigin(int64_t originKey) const {
            int32_t off = originsOffset();
            const int32_t g = rdI32(p + off); off += 4;
            for (int32_t i = 0; i < g; ++i) {
                const int64_t kk = rdI64(p + off); off += 8;
                const int32_t lineCount = rdI32(p + off); off += 4;
                if (kk == originKey) return lineCount > 0;
                for (int32_t j = 0; j < lineCount; ++j) {
                    off += 1;                              // tag
                    const int32_t d = rdI32(p + off); off += 4;
                    off += 8 * d;
                }
            }
            return false;
        }

        /// @brief Visit the ascending level ints stored for one packed pair key —
        ///        the `EquivalenceClass::forEachLevel` twin over the blob, so the
        ///        equi-class helpers read levels uniformly (equi-7 read cascade).
        /// @param key Packed pair key (`packEqPairKey`).
        /// @param fn  Callable `(int level)`, invoked per level in ascending order
        ///            (the blob stores each level run ascending).
        template <typename Fn>
        void forEachLevel(uint32_t key, Fn&& fn) const {
            const char* b = nullptr;
            int32_t c = 0;
            if (!findLevels(key, b, c)) return;
            for (int32_t j = 0; j < c; ++j)
                fn(static_cast<int>(rdI32(b + 4 * j)));
        }
    };

    /// @brief Whether class-source @p c contains member id @p id — the
    ///        `std::find(memberIds…) != end` twin, uniform over a heap
    ///        `EquivalenceClass` and a zero-copy `EquivalenceClassView` so the
    ///        equi-class helpers template over both (equi-7 read cascade).
    /// @param c  A member source exposing `memberCount()` / `memberId(i)`.
    /// @param id The member id to test.
    /// @return `true` iff some member equals @p id.
    template <typename ClassT>
    inline bool classHasMember(const ClassT& c, int16_t id) {
        const int32_t n = c.memberCount();
        for (int32_t i = 0; i < n; ++i)
            if (c.memberId(i) == id) return true;
        return false;
    }

    /// @brief One arena-backed sorted-unique `(original, validity)` channel —
    ///        the transient replacement for a `std::set<ExpressionWithValidity>`
    ///        returned by `disintegrateExpr2`.
    ///
    /// @details
    /// A flat POD accumulator on a per-slot `genScratchArenas` arena (the
    /// `CollectedArena` / `NewVarStore` idiom), holding distinct
    /// `(original, validity)` rows the way `std::set<ExpressionWithValidity>`
    /// did. Dedup is O(1): `append` builds a composite key `original + 0x01 +
    /// validity` on the arena's BYTE-BUMP tier (freed per call by a
    /// `ScratchScope`) and probes/mints it in `keys`; a repeat is skipped. Each
    /// NEW row mints `original` and `validity` into the SAME byte interner (a
    /// throwaway arena `ColdHashSet` — NOT NameMap/valueInterner/originInterner,
    /// so nothing observable mints, I-84) and records their ids in the parallel
    /// `origIds` / `valIds` PAGE-tier columns. `forEachSorted` reproduces
    /// `ExpressionWithValidity::operator<` = `(original, validityName)`
    /// lexicographic byte-for-byte via a decoded-lex index on the byte-bump tier
    /// (09c §5): the sort does pure `decode` reads (no mint), so the zero-copy
    /// spans stay valid across `std::sort` (I-3). The `0x01` separator never
    /// appears in MPL text, so a composite key can never collide with a
    /// component string or another `(original, validity)` pair.
    ///
    /// Dedup key equality (`original` byte-equal AND `validity` byte-equal) ==
    /// `ExpressionWithValidity::operator==`, so the channel holds exactly the
    /// distinct `EWV`s the `std::set` held; `forEachSorted`'s comparator IS
    /// `EWV::operator<`, so consumers fire in byte-identical order.
    ///
    /// @invariant Lives within a single `disintegrateExpr2` call; every span it
    ///            yields points into the `arena` it was constructed on.
    /// @see DisintProducts, ExpressionWithValidity, compareSpans, CollectedArena.
    struct DisintPairChannel {
        LbArena* arena;                        ///< per-slot scratch arena.
        DirtyState dirty = DirtyState::Clean;  ///< scratch (never deloaded).
        ColdHashSet<BytesKeyStore> keys;       ///< dedup + component byte interner (page tier).
        PagedVector<int32_t> origIds;          ///< per-row original id (page tier).
        PagedVector<int32_t> valIds;           ///< per-row validity id (page tier).

        /// @brief Bind the interner and the two id columns to the arena's PAGE
        ///        tier; the composite build and the sort index use its BYTE-BUMP
        ///        tier (independent substrate, no interference).
        /// @param a The per-slot scratch arena; outlives this channel.
        explicit DisintPairChannel(LbArena* a)
            : arena(a), keys(a, &dirty), origIds(a, &dirty), valIds(a, &dirty) {}
        DisintPairChannel(const DisintPairChannel&) = delete;
        DisintPairChannel& operator=(const DisintPairChannel&) = delete;

        /// @brief Append one distinct `(original, validity)` row (idempotent).
        ///
        /// @details Builds the composite dedup key on the byte-bump tier under a
        /// `ScratchScope`, probes `keys`; if already present, returns without
        /// change. Otherwise mints the composite, then mints @p original and
        /// @p validity and records their ids in the parallel columns.
        ///
        /// @param original The expression bytes (span source, read synchronously).
        /// @param validity The validity-scope bytes (span source).
        void append(StrSpan original, StrSpan validity) {
            ScratchScope sc(*arena);
            const int32_t n = original.len + 1 + validity.len;
            char* buf = reinterpret_cast<char*>(arena->resolve(arena->alloc(n, 1)));
            if (original.len > 0) std::memcpy(buf, original.ptr, original.len);
            buf[original.len] = '\x01';
            if (validity.len > 0)
                std::memcpy(buf + original.len + 1, validity.ptr, validity.len);
            const StrSpan composite(buf, n);
            if (keys.lookup(composite) != 0) return;  // already present
            keys.mint(composite);
            origIds.push_back(keys.mint(original));
            valIds.push_back(keys.mint(validity));
        }

        /// @brief Number of distinct rows. @return Row count (== `std::set` size).
        int32_t count() const { return origIds.size(); }

        /// @brief Invoke @p cb(original, validity) for each distinct row in
        ///        `ExpressionWithValidity::operator<` order.
        ///
        /// @details Allocates a decoded-lex index on the byte-bump tier, sorts
        /// it by `compareSpans(original)` then, on a tie, `compareSpans(validity)`
        /// — byte-identical to the former `std::set<EWV>` iterator order. The
        /// sort does pure `decode` reads (no mint), so the spans stay valid.
        ///
        /// @param cb Callback `(StrSpan original, StrSpan validity)`.
        template <typename Fn> void forEachSorted(Fn cb) const {
            const int32_t n = origIds.size();
            if (n == 0) return;
            ScratchScope sc(*arena);
            int32_t* idx = reinterpret_cast<int32_t*>(arena->resolve(
                arena->alloc(n * static_cast<int32_t>(sizeof(int32_t)),
                             alignof(int32_t))));
            for (int32_t k = 0; k < n; ++k) idx[k] = k;
            std::sort(idx, idx + n, [&](int32_t a, int32_t b) {
                const int c = compareSpans(keys.decode(origIds[a]),
                                           keys.decode(origIds[b]));
                if (c != 0) return c < 0;
                return compareSpans(keys.decode(valIds[a]),
                                    keys.decode(valIds[b])) < 0;
            });
            for (int32_t k = 0; k < n; ++k) {
                const int32_t r = idx[k];
                cb(keys.decode(origIds[r]), keys.decode(valIds[r]));
            }
        }
    };

    /// @brief The two-channel return holder for `disintegrateExpr2` — the
    ///        transient replacement for its `std::tuple<std::set<EWV>,
    ///        std::set<EWV>, int, bool>` return.
    ///
    /// @details A caller-owned pair of `DisintPairChannel`s: `implications`
    /// (former `get<0>`) and `statements` (former `get<1>`). The redundant
    /// `startInt` (former `get<2>`) is dropped (delivered by the live
    /// `memoryBlock.startInt` mutation) and the bool (former `get<3>`) moves to
    /// `disintegrateExpr2`'s return value. Both channels ride ONE per-slot
    /// scratch arena the caller supplies.
    ///
    /// @invariant Lives within a single `disintegrateExpr2` call; non-copyable.
    /// @see DisintPairChannel, disintegrateExpr2.
    struct DisintProducts {
        DisintPairChannel implications;  ///< former `get<0>` set.
        DisintPairChannel statements;    ///< former `get<1>` set.

        /// @brief Bind both channels to @p a.
        /// @param a The per-slot scratch arena; outlives this holder.
        explicit DisintProducts(LbArena* a) : implications(a), statements(a) {}
        DisintProducts(const DisintProducts&) = delete;
        DisintProducts& operator=(const DisintProducts&) = delete;
    };

    /// @brief Mutable arena-backed accumulator that BUILDS one merged
    ///        equivalence class off the malloc heap and emits it as the
    ///        canonical blob — the transient-statification replacement for the
    ///        heap `mergedClass` in `updateEquivalenceClasses` /
    ///        `mergeTwoEquivalenceClasses`.
    ///
    /// @details
    /// A flat POD structure on a per-slot `genScratchArenas` arena (the
    /// `WorkInstruction` / `CollectedArena` idiom), holding the three fields a
    /// class carries — members, per-pair levels, origin history — with only the
    /// mutating operations the merge performs (seed, decoded-lex member union,
    /// per-key level set/lookup, `addOriginId` with the D-49 cap-preference,
    /// per-key origin overwrite-merge). It replaces the heap
    /// `EquivalenceClass mergedClass`: the five consumers in
    /// `updateEquivalenceClasses` read it (serialize→blob, `changedClassesThisStep`
    /// snapshot bytes, `eqClassSttmntIndexMap` members, the origin-sync loop over
    /// its origins) so no heap `EquivalenceClass` is decoded or built.
    ///
    /// Storage:
    /// - members: a `PagedVector<int16_t>` in decoded-lex order (I-84), grown by
    ///   `unionMembersByName` — the byte-exact twin of `unionMemberIdsByName`.
    /// - levels: parallel `levelKeys` / `levelStart` / `levelLen` into a
    ///   `levelPool` of ascending-unique int runs; `setLevel` find-or-repoints
    ///   (an overwrite leaves dead pool bytes the serialize never reads).
    /// - origins: FLAT insertion-order lines (`lineKey` / `lineTag` /
    ///   `lineDepStart` / `lineDepLen` into `depPool`); `serialize` groups them by
    ///   ascending key, stable within a key, so the per-key line order stays
    ///   insertion order.
    ///
    /// `serialize()` emits the blob BYTE-IDENTICAL to
    /// `serializeEquivalenceClass` of the equivalent heap class — members int32
    /// count + int16s; levels ascending packedKey + ascending int sets; origins
    /// ascending int64 keys + per-key insertion-order lines/deps. Like
    /// `serializeEquivalenceClass` it returns a `std::vector<char>` (the same
    /// pre-existing cold-write boundary — not new per-call heap).
    ///
    /// @invariant Layout-coupled to `serializeEquivalenceClass` — edited
    ///            together with it and `EquivalenceClassView`.
    /// @see serializeEquivalenceClass, EquivalenceClassView,
    ///      unionMemberIdsByName, addOriginId, overwriteOriginsId.
    struct MergeClassAccum {
        LbArena* arena;                       ///< per-slot scratch arena (PAGE tier).
        DirtyState dirty = DirtyState::Clean; ///< scratch (never deloaded).
        PagedVector<int16_t> members;         ///< decoded-lex member ids.
        PagedVector<uint32_t> levelKeys;      ///< per-key packed pair key.
        PagedVector<int32_t>  levelStart;     ///< run start into levelPool.
        PagedVector<int32_t>  levelLen;       ///< run length.
        PagedVector<int32_t>  levelPool;      ///< ascending-unique int runs.
        PagedVector<int64_t>  lineKey;        ///< per-line origin key (flat).
        PagedVector<uint8_t>  lineTag;        ///< per-line OriginTag.
        PagedVector<int32_t>  lineDepStart;   ///< per-line dep run start into depPool.
        PagedVector<int32_t>  lineDepLen;     ///< per-line dep run length.
        PagedVector<int64_t>  depPool;        ///< dependency-key runs.

        /// @brief Bind every container to the arena's PAGE tier.
        /// @param a The per-slot scratch arena; outlives this accumulator.
        explicit MergeClassAccum(LbArena* a)
            : arena(a), members(a, &dirty),
              levelKeys(a, &dirty), levelStart(a, &dirty), levelLen(a, &dirty),
              levelPool(a, &dirty), lineKey(a, &dirty), lineTag(a, &dirty),
              lineDepStart(a, &dirty), lineDepLen(a, &dirty), depPool(a, &dirty) {}
        MergeClassAccum(const MergeClassAccum&) = delete;
        MergeClassAccum& operator=(const MergeClassAccum&) = delete;

        /// @brief Append one member id (decoded-lex order is the caller's
        ///        responsibility — used to seed from a decoded-lex id list).
        /// @param id The NameMap member id.
        void addMember(int16_t id) { members.push_back(id); }

        /// @brief Union the current members with another decoded-lex id list,
        ///        keeping the result decoded-lex — the in-place twin of
        ///        `unionMemberIdsByName`.
        ///
        /// @details Two-pointer merge comparing DECODED names (I-84): equal ids
        /// advance both, else the lex-smaller name is taken. Rebuilds `members`
        /// through an arena scratch snapshot (no heap).
        ///
        /// @param other The other member id list (decoded-lex).
        /// @param n     Its length.
        /// @param nm    NameMap for decoding (read-only).
        void unionMembersByName(const int16_t* other, int32_t n, const NameMap& nm) {
            PagedVector<int16_t> cur(arena, &dirty);
            for (int32_t i = 0; i < members.size(); ++i) cur.push_back(members[i]);
            members.clear();
            int32_t i = 0, j = 0;
            const int32_t ni = cur.size();
            while (i < ni && j < n) {
                const int16_t a = cur[i];
                const int16_t b = other[j];
                if (a == b) { members.push_back(a); ++i; ++j; continue; }
                const StrSpan na = nm.decodeView(a);
                const StrSpan nb = nm.decodeView(b);
                assert(!equalSpans(na, nb));
                if (compareSpans(na, nb) < 0) { members.push_back(a); ++i; }
                else                          { members.push_back(b); ++j; }
            }
            while (i < ni) members.push_back(cur[i++]);
            while (j < n)  members.push_back(other[j++]);
        }

        /// @brief Index of `key` in the levels section, or -1.
        /// @param key Packed pair key.
        /// @return The entry index, or -1 when absent.
        int32_t levelIndex(uint32_t key) const {
            for (int32_t i = 0; i < levelKeys.size(); ++i)
                if (levelKeys[i] == key) return i;
            return -1;
        }

        /// @brief Set `key`'s ascending-unique level run (replace-or-add) — the
        ///        `intEqualityLevelsMap[key] = set` twin.
        ///
        /// @details On an existing key the new run is appended to `levelPool` and
        /// the entry repointed (the old run becomes dead bytes the serialize never
        /// reads); a fresh key appends a new entry. The caller supplies the run
        /// ascending-unique (a `std::set<int>` iteration).
        ///
        /// @param key The packed pair key.
        /// @param asc Pointer to `n` ascending-unique int32 levels.
        /// @param n   The level count.
        void setLevel(uint32_t key, const int32_t* asc, int32_t n) {
            const int32_t start = levelPool.size();
            for (int32_t k = 0; k < n; ++k) levelPool.push_back(asc[k]);
            const int32_t idx = levelIndex(key);
            if (idx >= 0) { levelStart.setAt(idx, start); levelLen.setAt(idx, n); }
            else { levelKeys.push_back(key); levelStart.push_back(start); levelLen.push_back(n); }
        }

        /// @brief Whether any origin line is stored under `key`.
        /// @param key The origin key.
        /// @return `true` iff at least one line exists.
        bool hasOrigin(int64_t key) const {
            for (int32_t i = 0; i < lineKey.size(); ++i)
                if (lineKey[i] == key) return true;
            return false;
        }

        /// @brief Whether line index `i` equals `(tag, deps)` — the `IdOrigin`
        ///        `operator==` twin used for dedup.
        bool lineEquals(int32_t i, uint8_t tag,
                        const int64_t* deps, int32_t depN) const {
            if (lineTag[i] != tag || lineDepLen[i] != depN) return false;
            const int32_t s = lineDepStart[i];
            for (int32_t d = 0; d < depN; ++d)
                if (depPool[s + d] != deps[d]) return false;
            return true;
        }

        /// @brief Append one origin line under `key` (insertion order), no dedup
        ///        — the raw builder used when the caller already guarantees
        ///        uniqueness (e.g. seeding from a decoded view).
        /// @param key  The origin key.
        /// @param tag  The `OriginTag` byte.
        /// @param deps Pointer to `depN` int64 dependency keys.
        /// @param depN The dependency count.
        void addOriginLine(int64_t key, uint8_t tag,
                           const int64_t* deps, int32_t depN) {
            const int32_t s = depPool.size();
            for (int32_t d = 0; d < depN; ++d) depPool.push_back(deps[d]);
            lineKey.push_back(key);
            lineTag.push_back(tag);
            lineDepStart.push_back(s);
            lineDepLen.push_back(depN);
        }

        /// @brief `addOriginId` twin — dedup + cap + D-49 cap-full preference.
        ///
        /// @details Mirrors `gl::addOriginId` byte-for-byte on the flat lines of
        /// `key`: below cap, dedup-append; at cap, skip if present, else (a
        /// non-equality tag) replace the FIRST equality-convenience line under
        /// `key` (its tag + deps repointed), otherwise the existing slot wins.
        ///
        /// @param key  The origin key.
        /// @param tag  The `OriginTag` byte.
        /// @param deps Pointer to `depN` int64 dependency keys.
        /// @param depN The dependency count.
        /// @param cap  The per-key cap (`max_origin_per_expr` / compressor twin).
        void addOriginId(int64_t key, uint8_t tag,
                         const int64_t* deps, int32_t depN, int cap) {
            int32_t count = 0;
            int32_t firstEqLine = -1;
            bool present = false;
            for (int32_t i = 0; i < lineKey.size(); ++i) {
                if (lineKey[i] != key) continue;
                ++count;
                if (lineEquals(i, tag, deps, depN)) present = true;
                if (firstEqLine < 0
                    && (lineTag[i] == static_cast<uint8_t>(OriginTag::equality1)
                        || lineTag[i] == static_cast<uint8_t>(OriginTag::equality2)))
                    firstEqLine = i;
            }
            if (count < cap) {
                if (!present) addOriginLine(key, tag, deps, depN);
                return;
            }
            if (present) return;
            const bool isEqTag = (tag == static_cast<uint8_t>(OriginTag::equality1)
                               || tag == static_cast<uint8_t>(OriginTag::equality2));
            if (!isEqTag && firstEqLine >= 0) {
                const int32_t s = depPool.size();
                for (int32_t d = 0; d < depN; ++d) depPool.push_back(deps[d]);
                lineTag.setAt(firstEqLine, tag);
                lineDepStart.setAt(firstEqLine, s);
                lineDepLen.setAt(firstEqLine, depN);
            }
            // else existing slot wins (insertion-order tiebreak).
        }

        /// @brief Encode a string `(expression, validity)` key + `(tag,
        ///        antecedents)` record and append it — the accumulator twin of
        ///        the free `addOriginEncoded(IdOriginMap, oi, ev, origin, cap)`.
        ///
        /// @details Mints the origin key FIRST (already this overload's order),
        /// resolves the tag via the span `originTagFromString` (interns nothing),
        /// then mints the EWV antecedents positionally via
        /// `mintOriginDepsFromEWVInto` and routes to this accumulator's POD
        /// `addOriginId` (dedup + cap + D-49) — no transient `IdOrigin`
        /// materialized. Single-threaded write side only (I-83).
        ///
        /// @param oi     The owning LB's origin interner (mint side).
        /// @param ev     `(expression, validity)` key.
        /// @param origin `(tag, antecedents)` string record.
        /// @param cap    Per-key cap.
        /// @see addOriginEncoded, addOriginId.
        void addOriginEnc(ValueInterner& oi, const ExpressionWithValidity& ev,
            const std::pair<std::string, std::vector<ExpressionWithValidity> >& origin,
            int cap) {
            // Already key-first: mint the KEY, resolve the tag via the span twin
            // (interns nothing), then mint the EWV antecedents positionally into
            // a stack buffer — no IdOrigin materialized.
            const int64_t key = mintOriginKey(oi, ev.original, ev.validityName);
            const OriginTag tag = originTagFromString(StrSpan(origin.first));
            int64_t d[ExecutionParameters::kMaxOriginDeps];
            const int32_t n = mintOriginDepsFromEWVInto(origin.second, oi, d,
                ExecutionParameters::kMaxOriginDeps);
            addOriginId(key, static_cast<uint8_t>(tag), d, n, cap);
        }

        /// @brief Span-antecedent twin of @ref addOriginEnc — encode a string
        ///        key + an `OriginTag` / `OriginDep` record and append it, no
        ///        transient `ExpressionWithValidity` / `OriginLine` built.
        ///
        /// @details
        /// Byte-identical to the `ExpressionWithValidity` / `OriginLine`
        /// overload: the key is minted FIRST (matching that overload's statement
        /// order, so the `originInterner` mint sequence is preserved), then the
        /// dependency ids are minted positionally via `mintOriginDepsInto` into
        /// a stack buffer and fed to this accumulator's POD `addOriginId` sink
        /// (dedup + cap + D-49) — no transient `IdOrigin` materialized.
        /// Single-threaded write side only
        /// ([I-83](../../docs/agentic_swdd/30_invariants.md#i-83)).
        ///
        /// @param oi   The owning LB's origin interner (mint side).
        /// @param evO  Span over the key expression bytes.
        /// @param evV  Span over the key validity bytes.
        /// @param tag  The history-line tag (the enumerator the site names).
        /// @param deps Pointer to `depN` antecedent spans, positional order.
        /// @param depN The antecedent count.
        /// @param cap  Per-key cap.
        /// @see addOriginEnc, addOriginId, encodeOriginSpans.
        void addOriginEnc(ValueInterner& oi, const StrSpan& evO,
            const StrSpan& evV, OriginTag tag, const OriginDep* deps, int depN,
            int cap) {
            // Key-first (already the twin's order): mint the KEY, then the deps
            // positionally into a stack buffer — no IdOrigin materialized.
            const int64_t key = mintOriginKey(oi, evO, evV);
            int64_t d[ExecutionParameters::kMaxOriginDeps];
            const int32_t n = mintOriginDepsInto(deps, depN, oi, d,
                ExecutionParameters::kMaxOriginDeps);
            addOriginId(key, static_cast<uint8_t>(tag), d, n, cap);
        }

        /// @brief B-base origin merge — reproduce the heap
        ///        `tmp = classB.equalityOriginMap; overwriteOriginsId(tmp, A, cap)`
        ///        with this accumulator holding A.
        ///
        /// @details
        /// The merge's origin combine is B-BASE: classB's history lines come
        /// FIRST (a verbatim copy of `classB.equalityOriginMap`), then this
        /// accumulator's own lines (A) are appended per key below the cap,
        /// deduped — exactly `overwriteOriginsId`'s inner
        /// (`leftVec.size() < cap && find == end`). This is NOT `addOriginId`:
        /// there is NO D-49 cap-full preference here (that belongs to the
        /// equality2 emission path, `addOriginId`). Since the accumulator holds
        /// A but the result must lead with B, the op (1) snapshots A's lines,
        /// (2) clears the origin section, (3) re-seeds it verbatim from the
        /// classB view, then (4) merges the A snapshot back below cap. Per-key
        /// line order is observable (serialize emits it verbatim), so the
        /// B-first-then-A order is the byte-critical contract. All temporaries
        /// ride the same page-tier arena — no heap.
        ///
        /// @param classB The absorbed class, read through its cold blob
        ///               (origins via `forEachOriginLine`).
        /// @param cap    Per-key cap (`max_origin_per_expr` / compressor twin).
        /// @see overwriteOriginsId — the heap twin this reproduces byte-for-byte.
        void mergeOriginsBBase(const EquivalenceClassView& classB, int cap) {
            // 1. Snapshot A's current origin lines (this accum's) — deps copied
            //    by value, so the clear below cannot disturb them.
            PagedVector<int64_t> aKey(arena, &dirty);
            PagedVector<uint8_t> aTag(arena, &dirty);
            PagedVector<int32_t> aStart(arena, &dirty);
            PagedVector<int32_t> aLen(arena, &dirty);
            PagedVector<int64_t> aPool(arena, &dirty);
            for (int32_t i = 0; i < lineKey.size(); ++i) {
                aKey.push_back(lineKey[i]);
                aTag.push_back(lineTag[i]);
                const int32_t s = aPool.size();
                const int32_t src = lineDepStart[i];
                for (int32_t d = 0; d < lineDepLen[i]; ++d) aPool.push_back(depPool[src + d]);
                aStart.push_back(s);
                aLen.push_back(lineDepLen[i]);
            }
            // 2. Clear the origin section (levels + members untouched).
            lineKey.clear(); lineTag.clear();
            lineDepStart.clear(); lineDepLen.clear(); depPool.clear();
            // 3. Seed verbatim from classB (`tmp = classB.equalityOriginMap`):
            //    every B line, in the view's stored order, no dedup.
            classB.forEachOriginLine([&](int64_t key, OriginTag tag,
                                         const char* depBytes, int32_t depN) {
                const int32_t s = depPool.size();
                for (int32_t d = 0; d < depN; ++d)
                    depPool.push_back(EquivalenceClassView::rdI64(depBytes + 8 * d));
                lineKey.push_back(key);
                lineTag.push_back(static_cast<uint8_t>(tag));
                lineDepStart.push_back(s);
                lineDepLen.push_back(depN);
            });
            // 4. Merge the A snapshot below cap, deduped — `overwriteOriginsId`
            //    inner (the per-key count grows as lines are appended).
            for (int32_t i = 0; i < aKey.size(); ++i) {
                const int64_t key = aKey[i];
                const uint8_t tag = aTag[i];
                const int32_t alen = aLen[i];
                const int32_t asrc = aStart[i];
                int32_t cnt = 0;
                bool present = false;
                for (int32_t j = 0; j < lineKey.size(); ++j) {
                    if (lineKey[j] != key) continue;
                    ++cnt;
                    if (lineTag[j] == tag && lineDepLen[j] == alen) {
                        bool eq = true;
                        const int32_t bs = lineDepStart[j];
                        for (int32_t d = 0; d < alen; ++d)
                            if (depPool[bs + d] != aPool[asrc + d]) { eq = false; break; }
                        if (eq) present = true;
                    }
                }
                if (cnt < cap && !present) {
                    const int32_t s = depPool.size();
                    for (int32_t d = 0; d < alen; ++d) depPool.push_back(aPool[asrc + d]);
                    lineKey.push_back(key);
                    lineTag.push_back(tag);
                    lineDepStart.push_back(s);
                    lineDepLen.push_back(alen);
                }
            }
        }

        /// @brief Level-merge from a classB view — the `newLevels =
        ///        classB.intEqualityLevelsMap; overwriteLevels(newLevels, A)`
        ///        twin (A-wins REPLACE).
        ///
        /// @details `overwriteLevels` REPLACES per key (`left[k] = right[k]`),
        /// not unions: the result is `A[k]` when key `k` is in A, else `B[k]`.
        /// Since this accumulator already holds A, the op keeps every existing
        /// (A) key's run untouched and only ADDS the B keys the accumulator
        /// lacks. `serialize()` sorts keys ascending, so the append order is
        /// invisible.
        ///
        /// @param classB The absorbed class, levels read via `forEachLevelKey`.
        /// @see overwriteLevels — the heap twin.
        void mergeLevelsBBase(const EquivalenceClassView& classB) {
            classB.forEachLevelKey([&](uint32_t key, const char* lp, int32_t m) {
                if (levelIndex(key) >= 0) return;   // A wins — keep A's run
                const int32_t start = levelPool.size();
                for (int32_t k = 0; k < m; ++k)
                    levelPool.push_back(EquivalenceClassView::rdI32(lp + 4 * k));
                levelKeys.push_back(key);
                levelStart.push_back(start);
                levelLen.push_back(m);
            });
        }

        /// @brief Level-merge from another accumulator — the `tmp = other;
        ///        overwriteLevels(tmp, A)` twin (A-wins REPLACE). Keeps every
        ///        existing (A) key, adds `other`'s keys the accumulator lacks.
        ///
        /// @param other The source accumulator (e.g. the cross-product
        ///              `mergedMap` temp).
        /// @see mergeLevelsBBase — the view-source sibling.
        void mergeLevelsFromAccum(const MergeClassAccum& other) {
            for (int32_t i = 0; i < other.levelKeys.size(); ++i) {
                const uint32_t key = other.levelKeys[i];
                if (levelIndex(key) >= 0) continue;   // A wins — keep A's run
                const int32_t start = levelPool.size();
                const int32_t src = other.levelStart[i];
                for (int32_t k = 0; k < other.levelLen[i]; ++k)
                    levelPool.push_back(other.levelPool[src + k]);
                levelKeys.push_back(key);
                levelStart.push_back(start);
                levelLen.push_back(other.levelLen[i]);
            }
        }

        /// @brief Origin-merge from another accumulator — the `tmp = other;
        ///        overwriteOriginsId(tmp, A, cap)` twin (other's lines FIRST,
        ///        then A appended per key below cap, deduped, NO D-49).
        ///
        /// @details Identical shape to `mergeOriginsBBase` with an accumulator
        /// source instead of a classB view: snapshot A's lines, clear, re-seed
        /// verbatim from `other`, then merge A below cap. Per-key line order is
        /// serialized verbatim, so other-first-then-A is the byte-critical
        /// contract.
        ///
        /// @param other The source accumulator (e.g. the cross-product
        ///              `mergedOriginMap` temp).
        /// @param cap   Per-key cap (`max_origin_per_expr` / compressor twin).
        /// @see mergeOriginsBBase — the view-source sibling; overwriteOriginsId.
        void mergeOriginsFromAccum(const MergeClassAccum& other, int cap) {
            // 1. Snapshot A's current origin lines (deps copied by value).
            PagedVector<int64_t> aKey(arena, &dirty);
            PagedVector<uint8_t> aTag(arena, &dirty);
            PagedVector<int32_t> aStart(arena, &dirty);
            PagedVector<int32_t> aLen(arena, &dirty);
            PagedVector<int64_t> aPool(arena, &dirty);
            for (int32_t i = 0; i < lineKey.size(); ++i) {
                aKey.push_back(lineKey[i]);
                aTag.push_back(lineTag[i]);
                const int32_t s = aPool.size();
                const int32_t src = lineDepStart[i];
                for (int32_t d = 0; d < lineDepLen[i]; ++d) aPool.push_back(depPool[src + d]);
                aStart.push_back(s);
                aLen.push_back(lineDepLen[i]);
            }
            // 2. Clear the origin section.
            lineKey.clear(); lineTag.clear();
            lineDepStart.clear(); lineDepLen.clear(); depPool.clear();
            // 3. Seed verbatim from `other`.
            for (int32_t i = 0; i < other.lineKey.size(); ++i) {
                const int32_t s = depPool.size();
                const int32_t src = other.lineDepStart[i];
                for (int32_t d = 0; d < other.lineDepLen[i]; ++d)
                    depPool.push_back(other.depPool[src + d]);
                lineKey.push_back(other.lineKey[i]);
                lineTag.push_back(other.lineTag[i]);
                lineDepStart.push_back(s);
                lineDepLen.push_back(other.lineDepLen[i]);
            }
            // 4. Merge A below cap, deduped.
            for (int32_t i = 0; i < aKey.size(); ++i) {
                const int64_t key = aKey[i];
                const uint8_t tag = aTag[i];
                const int32_t alen = aLen[i];
                const int32_t asrc = aStart[i];
                int32_t cnt = 0;
                bool present = false;
                for (int32_t j = 0; j < lineKey.size(); ++j) {
                    if (lineKey[j] != key) continue;
                    ++cnt;
                    if (lineTag[j] == tag && lineDepLen[j] == alen) {
                        bool eq = true;
                        const int32_t bs = lineDepStart[j];
                        for (int32_t d = 0; d < alen; ++d)
                            if (depPool[bs + d] != aPool[asrc + d]) { eq = false; break; }
                        if (eq) present = true;
                    }
                }
                if (cnt < cap && !present) {
                    const int32_t s = depPool.size();
                    for (int32_t d = 0; d < alen; ++d) depPool.push_back(aPool[asrc + d]);
                    lineKey.push_back(key);
                    lineTag.push_back(tag);
                    lineDepStart.push_back(s);
                    lineDepLen.push_back(alen);
                }
            }
        }

        /// @brief Union the current members with a classB VIEW's members,
        ///        keeping the result decoded-lex — the twin of
        ///        `unionMembersByName` with a `EquivalenceClassView` source.
        ///
        /// @details Two-pointer merge comparing DECODED names (I-84), reading
        /// the other operand via `view.memberId` instead of a contiguous id
        /// array (the merge's `classB` is read through its cold blob, whose
        /// members are not a heap vector).
        ///
        /// @param other The absorbed class view (members decoded-lex).
        /// @param nm    NameMap for decoding (read-only).
        void unionMembersFromView(const EquivalenceClassView& other,
                                  const NameMap& nm) {
            PagedVector<int16_t> cur(arena, &dirty);
            for (int32_t i = 0; i < members.size(); ++i) cur.push_back(members[i]);
            members.clear();
            int32_t i = 0, j = 0;
            const int32_t ni = cur.size();
            const int32_t n = other.memberCount();
            while (i < ni && j < n) {
                const int16_t a = cur[i];
                const int16_t b = other.memberId(j);
                if (a == b) { members.push_back(a); ++i; ++j; continue; }
                const StrSpan na = nm.decodeView(a);
                const StrSpan nb = nm.decodeView(b);
                assert(!equalSpans(na, nb));
                if (compareSpans(na, nb) < 0) { members.push_back(a); ++i; }
                else                          { members.push_back(b); ++j; }
            }
            while (i < ni) members.push_back(cur[i++]);
            while (j < n)  members.push_back(other.memberId(j++));
        }

        /// @brief Zero-copy view of one origin key's history lines (insertion
        ///        order) handed to the `forEachOriginSorted` visitor — the
        ///        heap-free replacement for the former materialized
        ///        `std::vector<IdOrigin>`.
        ///
        /// @details Lines are addressed by their indices into the accumulator's
        /// flat line arrays; a dep is read element-wise because `depPool` may
        /// straddle pages. Valid only for the duration of the visitor call (the
        /// backing index run is reused per key).
        struct LineView {
            const MergeClassAccum* accum;  ///< owning accumulator (read-only).
            const int32_t* idx;            ///< this key's line indices.
            int32_t n;                     ///< line count for this key.
            /// @return Number of history lines under this key.
            int32_t size() const { return n; }
            /// @param t Line ordinal, `0 <= t < size()`.
            /// @return The line's OriginTag.
            OriginTag tag(int32_t t) const {
                return static_cast<OriginTag>(accum->lineTag[idx[t]]);
            }
            /// @param t Line ordinal.
            /// @return The line's dependency count.
            int32_t depCount(int32_t t) const { return accum->lineDepLen[idx[t]]; }
            /// @param t Line ordinal; @param d Dep ordinal (`< depCount(t)`).
            /// @return The dependency key.
            int64_t dep(int32_t t, int32_t d) const {
                return accum->depPool[accum->lineDepStart[idx[t]] + d];
            }
        };

        /// @brief Visit each origin key ascending, passing the key and a
        ///        zero-copy `LineView` of its history lines — the
        ///        `for (kv : equalityOriginMap)` walk in sorted-key order.
        ///
        /// @details Reproduces the heap `equalityOriginMap`'s sorted-key
        /// iteration: distinct keys ascending, per-key lines in insertion order.
        /// Both the distinct-key set and the per-key line-index run are built on
        /// the accumulator's own arena byte-bump tier (reclaimed by the closing
        /// `popTo`), so the walk is 0% heap; the caller materializes whatever it
        /// needs (the origin-sync loop builds its `newRun` for
        /// `exprOriginMap.assignRun`). The distinct-then-sort logic is byte-for-
        /// byte the former `std::find` + numeric `std::sort`.
        ///
        /// @tparam Fn Callable `(int64_t key, LineView lines)`.
        /// @param fn The per-key visitor.
        template <typename Fn>
        void forEachOriginSorted(Fn&& fn) const {
            const int32_t nLines = lineKey.size();
            if (nLines == 0) return;
            const ArenaOffset mark = arena->cursor();
            // Distinct keys on the byte-bump tier — same O(n^2) linear find +
            // numeric sort as the former heap `okeys`.
            int64_t* const okeys = reinterpret_cast<int64_t*>(arena->resolve(
                arena->alloc(nLines * static_cast<int32_t>(sizeof(int64_t)),
                             static_cast<int32_t>(alignof(int64_t)))));
            int32_t nk = 0;
            for (int32_t i = 0; i < nLines; ++i) {
                const int64_t k = lineKey[i];
                bool seen = false;
                for (int32_t j = 0; j < nk; ++j)
                    if (okeys[j] == k) { seen = true; break; }
                if (!seen) okeys[nk++] = k;
            }
            std::sort(okeys, okeys + nk);
            // Per-key line-index run (insertion order), reused each key.
            int32_t* const idx = reinterpret_cast<int32_t*>(arena->resolve(
                arena->alloc(nLines * static_cast<int32_t>(sizeof(int32_t)),
                             static_cast<int32_t>(alignof(int32_t)))));
            for (int32_t ki = 0; ki < nk; ++ki) {
                const int64_t key = okeys[ki];
                int32_t ln = 0;
                for (int32_t i = 0; i < nLines; ++i)
                    if (lineKey[i] == key) idx[ln++] = i;
                fn(key, LineView{this, idx, ln});
            }
            arena->popTo(mark);
        }

        /// @brief Byte length of the canonical blob `serializeInto` writes — the
        ///        heap-free size query the run-splice callers use to size their
        ///        output buffer before writing.
        ///
        /// @details Derived from the container element counts (the three count
        /// headers plus member / level / dep bytes) and the distinct origin-key
        /// count (computed on the arena byte-bump tier, reclaimed on return).
        /// Equals the exact byte count `serializeInto` emits — asserted where the
        /// two are used together (and in `serialize`).
        ///
        /// @return The blob length in bytes (>= 12, the three count headers).
        int32_t serializedSize() const {
            const int32_t nLines = lineKey.size();
            int32_t nk = 0;
            if (nLines > 0) {
                const ArenaOffset mark = arena->cursor();
                int64_t* const okeys = reinterpret_cast<int64_t*>(arena->resolve(
                    arena->alloc(nLines * static_cast<int32_t>(sizeof(int64_t)),
                                 static_cast<int32_t>(alignof(int64_t)))));
                for (int32_t i = 0; i < nLines; ++i) {
                    const int64_t k = lineKey[i];
                    bool seen = false;
                    for (int32_t j = 0; j < nk; ++j)
                        if (okeys[j] == k) { seen = true; break; }
                    if (!seen) okeys[nk++] = k;
                }
                arena->popTo(mark);
            }
            int64_t levelInts = 0;
            for (int32_t i = 0; i < levelKeys.size(); ++i) levelInts += levelLen[i];
            int64_t depInts = 0;
            for (int32_t i = 0; i < nLines; ++i) depInts += lineDepLen[i];
            const int64_t total =
                4 + static_cast<int64_t>(members.size()) * 2                       // members
              + 4 + static_cast<int64_t>(levelKeys.size()) * 8 + levelInts * 4     // levels
              + 4 + static_cast<int64_t>(nk) * 12
                  + static_cast<int64_t>(nLines) * 5 + depInts * 8;                // origins
            return static_cast<int32_t>(total);
        }

        /// @brief Write the canonical blob into a caller buffer of at least
        ///        `serializedSize()` bytes — the heap-free replacement for the
        ///        former `serialize()`-returns-`std::vector<char>` cold-write
        ///        boundary. BYTE-IDENTICAL to `serializeEquivalenceClass`.
        ///
        /// @details Members in stored order; levels ascending packedKey
        /// (index-sorted on the arena); origins ascending int64 key (distinct
        /// keys sorted on the arena) with per-key lines in insertion order and
        /// each line's deps verbatim. The two ordering scratch runs (`lidx`,
        /// `okeys`) ride the accumulator's arena byte-bump tier, reclaimed by the
        /// closing `popTo` — only the blob bytes reach `dst`, so it is 0% heap.
        ///
        /// @param dst Destination buffer; must hold `serializedSize()` bytes.
        /// @return The number of bytes written (== `serializedSize()`).
        int32_t serializeInto(char* dst) const {
            const ArenaOffset mark = arena->cursor();
            int32_t at = 0;
            const auto put = [&](const auto& v) {
                std::memcpy(dst + at, &v, sizeof(v));
                at += static_cast<int32_t>(sizeof(v));
            };
            // members
            put(static_cast<int32_t>(members.size()));
            for (int32_t i = 0; i < members.size(); ++i) put(members[i]);
            // levels — ascending packedKey (index-sorted on the arena)
            const int32_t nlev = levelKeys.size();
            int32_t* const lidx = nlev > 0 ? reinterpret_cast<int32_t*>(
                arena->resolve(arena->alloc(
                    nlev * static_cast<int32_t>(sizeof(int32_t)),
                    static_cast<int32_t>(alignof(int32_t))))) : nullptr;
            for (int32_t i = 0; i < nlev; ++i) lidx[i] = i;
            std::sort(lidx, lidx + nlev,
                [this](int32_t a, int32_t b) { return levelKeys[a] < levelKeys[b]; });
            put(static_cast<int32_t>(nlev));
            for (int32_t t = 0; t < nlev; ++t) {
                const int32_t i = lidx[t];
                put(static_cast<uint32_t>(levelKeys[i]));
                put(static_cast<int32_t>(levelLen[i]));
                const int32_t s = levelStart[i];
                for (int32_t k = 0; k < levelLen[i]; ++k)
                    put(static_cast<int32_t>(levelPool[s + k]));
            }
            // origins — distinct keys ascending (sorted on the arena); per-key
            // lines in insertion order
            const int32_t nLines = lineKey.size();
            int64_t* const okeys = nLines > 0 ? reinterpret_cast<int64_t*>(
                arena->resolve(arena->alloc(
                    nLines * static_cast<int32_t>(sizeof(int64_t)),
                    static_cast<int32_t>(alignof(int64_t))))) : nullptr;
            int32_t nk = 0;
            for (int32_t i = 0; i < nLines; ++i) {
                const int64_t k = lineKey[i];
                bool seen = false;
                for (int32_t j = 0; j < nk; ++j)
                    if (okeys[j] == k) { seen = true; break; }
                if (!seen) okeys[nk++] = k;
            }
            std::sort(okeys, okeys + nk);
            put(static_cast<int32_t>(nk));
            for (int32_t ki = 0; ki < nk; ++ki) {
                const int64_t key = okeys[ki];
                put(key);
                int32_t lineCount = 0;
                for (int32_t i = 0; i < nLines; ++i)
                    if (lineKey[i] == key) ++lineCount;
                put(static_cast<int32_t>(lineCount));
                for (int32_t i = 0; i < nLines; ++i) {
                    if (lineKey[i] != key) continue;
                    put(static_cast<uint8_t>(lineTag[i]));
                    put(static_cast<int32_t>(lineDepLen[i]));
                    const int32_t s = lineDepStart[i];
                    for (int32_t d = 0; d < lineDepLen[i]; ++d)
                        put(static_cast<int64_t>(depPool[s + d]));
                }
            }
            arena->popTo(mark);
            return at;
        }

        /// @brief Canonical blob as an owned `std::vector<char>` — RETAINED as
        ///        the differential-test oracle; production writes via
        ///        `serializeInto` (0% heap). Delegates so the two stay
        ///        byte-identical by construction.
        /// @return The canonical byte blob.
        std::vector<char> serialize() const {
            std::vector<char> out(static_cast<std::size_t>(serializedSize()));
            const int32_t w = serializeInto(out.data());
            assert(w == static_cast<int32_t>(out.size()));
            (void)w;
            return out;
        }
    };

    /// @brief Encode the `eqClassSttmntIndexMapMap` byte key directly from a
    ///        `MergeClassAccum`'s member run — the heap-free twin of
    ///        `encodeEqClassKey(validityId, mergedClass.memberIds)`.
    ///
    /// @details Same byte layout as `Codec<EqClassKey>::encode`: 2-byte
    /// little-endian `validityId`, then each member id (2 bytes) in the
    /// accumulator's stored (decoded-lex) order — bijective, point-lookup only.
    /// Builds the key straight from the arena member `PagedVector`, so no
    /// intermediate `std::vector<int16_t>` is materialized (only the byte key
    /// string the cold-map access needs regardless).
    ///
    /// @param validityId The class's validity-scope id.
    /// @param accum      The merged-class accumulator.
    /// @return The packed byte key.
    /// @see encodeEqClassKey, upsertEqClassIndex.
    inline std::string encodeEqClassKeyFromAccum(int16_t validityId,
                                                 const MergeClassAccum& accum) {
        std::string key(sizeof(int16_t)
            * (static_cast<std::size_t>(accum.members.size()) + 1), '\0');
        std::memcpy(&key[0], &validityId, sizeof(int16_t));
        for (int32_t i = 0; i < accum.members.size(); ++i) {
            const int16_t m = accum.members[i];
            std::memcpy(&key[sizeof(int16_t) * (static_cast<std::size_t>(i) + 1)],
                        &m, sizeof(int16_t));
        }
        return key;
    }

    /// @brief Serialize the `eqClassSttmntIndexMapMap` byte key from a
    ///        `MergeClassAccum`'s member run INTO a caller stack buffer — the
    ///        0% heap twin of `encodeEqClassKeyFromAccum`.
    ///
    /// @details
    /// Writes the identical byte layout `Codec<EqClassKey>::encode` /
    /// `encodeEqClassKeyFromAccum` produce — a 2-byte little-endian `validityId`,
    /// then each member id (2 bytes) in the accumulator's stored (decoded-lex)
    /// order — straight into @p buf, so no heap `std::string` key is
    /// materialized: the index helpers probe the cold map with
    /// `StrSpan(buf, n)` directly (`Codec<EqClassKey>::view(e)` is itself just
    /// `StrSpan(e)`, so the raw span carries the identical `BytesKeyStore`
    /// hash / equality). The key is fixed length (`2*(members+1)` bytes); the
    /// total is asserted against @p cap before any write — the Rule-19
    /// widen-on-STOP tripwire naming `kMaxEqClassKeyBytes`. The cap holds because
    /// a class's member count is bounded by `MAX_ARITY*MAX_KEY_SLOTS` (the
    /// `reduceEqClassIds` ceiling).
    ///
    /// @param buf        Caller stack buffer receiving the key bytes.
    /// @param cap        Capacity of @p buf (== `kMaxEqClassKeyBytes` at callers).
    /// @param validityId The class's validity-scope id.
    /// @param accum      The merged-class accumulator (its members are the key).
    /// @return The key length written (`== 2*(members.size()+1)`).
    /// @see encodeEqClassKeyFromAccum — the retained heap-string oracle;
    ///      Codec<EqClassKey>::encode; upsertEqClassIndex.
    inline int32_t encodeEqClassKeyFromAccumInto(char* buf, int32_t cap,
        int16_t validityId, const MergeClassAccum& accum) {
        const int32_t total = static_cast<int32_t>(sizeof(int16_t))
            * (accum.members.size() + 1);
        assert(total <= cap
            && "encodeEqClassKeyFromAccumInto: key exceeds kMaxEqClassKeyBytes");
        std::memcpy(buf, &validityId, sizeof(int16_t));
        for (int32_t i = 0; i < accum.members.size(); ++i) {
            const int16_t m = accum.members[i];
            std::memcpy(buf + sizeof(int16_t) * (static_cast<std::size_t>(i) + 1),
                        &m, sizeof(int16_t));
        }
        return total;
    }

    /// @brief Set an equivalence class's statement-index waterline keyed by a
    ///        `MergeClassAccum`'s members — the heap-free twin of
    ///        `upsertEqClassIndex(m, validityId, mergedClass.memberIds, value)`.
    ///
    /// @details Serializes the byte key into a stack buffer via
    /// `encodeEqClassKeyFromAccumInto` and drives the write through the cold
    /// map's byte-key door (`inner()`) with a raw `StrSpan(keyBuf, n)` probe,
    /// reproducing `TypedCold::upsert` (in-place `setValueAt` on a hit, set-once
    /// `insert` on a miss) without a heap key `std::string` or an `EqClassKey`
    /// `std::vector<int16_t>`. Single-threaded write side only (I-83).
    ///
    /// @param m          The cold eq-class index map.
    /// @param validityId The class's validity-scope id.
    /// @param accum      The merged-class accumulator (its members are the key).
    /// @param value      The waterline to store.
    /// @see upsertEqClassIndex, encodeEqClassKeyFromAccumInto.
    inline void upsertEqClassIndex(TypedColdMap<EqClassKey, int>& m,
                                   int16_t validityId,
                                   const MergeClassAccum& accum, int value) {
        char keyBuf[ExecutionParameters::kMaxEqClassKeyBytes];
        const int32_t n = encodeEqClassKeyFromAccumInto(keyBuf,
            ExecutionParameters::kMaxEqClassKeyBytes, validityId, accum);
        auto& inner = m.inner();
        const int32_t id = inner.lookup(StrSpan(keyBuf, n));
        if (id != 0) inner.setValueAt(id, value);
        else inner.insert(StrSpan(keyBuf, n), value);
    }

    /// @brief Encode the `eqClassSttmntIndexMapMap` byte key from a cold-blob
    ///        `EquivalenceClassView`'s member run — the heap-free twin of
    ///        `encodeEqClassKey(validityId, class.memberIds)` for a viewed class.
    ///
    /// @param validityId The class's validity-scope id.
    /// @param view       The class view (its members are the key).
    /// @return The packed byte key.
    /// @see encodeEqClassKey, encodeEqClassKeyFromAccum.
    inline std::string encodeEqClassKeyFromView(int16_t validityId,
                                                const EquivalenceClassView& view) {
        std::string key(sizeof(int16_t)
            * (static_cast<std::size_t>(view.memberCount()) + 1), '\0');
        std::memcpy(&key[0], &validityId, sizeof(int16_t));
        for (int32_t i = 0; i < view.memberCount(); ++i) {
            const int16_t m = view.memberId(i);
            std::memcpy(&key[sizeof(int16_t) * (static_cast<std::size_t>(i) + 1)],
                        &m, sizeof(int16_t));
        }
        return key;
    }

    /// @brief Serialize the `eqClassSttmntIndexMapMap` byte key from a cold-blob
    ///        `EquivalenceClassView`'s member run INTO a caller stack buffer —
    ///        the 0% heap twin of `encodeEqClassKeyFromView`.
    ///
    /// @details
    /// Writes the identical byte layout `Codec<EqClassKey>::encode` /
    /// `encodeEqClassKeyFromView` produce — a 2-byte little-endian `validityId`,
    /// then each member id (2 bytes) in the class's stored (decoded-lex) order —
    /// straight into @p buf, so no heap `std::string` key is materialized: the
    /// index helpers probe the cold map with `StrSpan(buf, n)` directly
    /// (`Codec<EqClassKey>::view(e)` is itself just `StrSpan(e)`, so the raw span
    /// carries the identical `BytesKeyStore` hash / equality). The key is fixed
    /// length (`2*(members+1)` bytes); the total is asserted against @p cap
    /// before any write — the Rule-19 widen-on-STOP tripwire naming
    /// `kMaxEqClassKeyBytes`. The cap holds because a class's member count is
    /// bounded by `MAX_ARITY*MAX_KEY_SLOTS` (the `reduceEqClassIds` ceiling).
    ///
    /// @param buf        Caller stack buffer receiving the key bytes.
    /// @param cap        Capacity of @p buf (== `kMaxEqClassKeyBytes` at callers).
    /// @param validityId The class's validity-scope id.
    /// @param view       The class view (its members are the key).
    /// @return The key length written (`== 2*(memberCount()+1)`).
    /// @see encodeEqClassKeyFromView — the retained heap-string oracle;
    ///      Codec<EqClassKey>::encode; lookupEqClassIndex.
    inline int32_t encodeEqClassKeyFromViewInto(char* buf, int32_t cap,
        int16_t validityId, const EquivalenceClassView& view) {
        const int32_t total = static_cast<int32_t>(sizeof(int16_t))
            * (view.memberCount() + 1);
        assert(total <= cap
            && "encodeEqClassKeyFromViewInto: key exceeds kMaxEqClassKeyBytes");
        std::memcpy(buf, &validityId, sizeof(int16_t));
        for (int32_t i = 0; i < view.memberCount(); ++i) {
            const int16_t m = view.memberId(i);
            std::memcpy(buf + sizeof(int16_t) * (static_cast<std::size_t>(i) + 1),
                        &m, sizeof(int16_t));
        }
        return total;
    }

    /// @brief Build a `Codec<NormKey>` byte key from raw int fields into a caller
    ///        stack buffer — the owning-`NormKey`-free key encoder.
    ///
    /// @details
    /// Byte-exact to `Codec<NormKey>::encode` (`int16 numberExpressions ++ int16
    /// length ++ length x int16 data`, fixed-width little-endian), but written
    /// straight into @p out from a `(numberExpressions, data, len)` triple, so no
    /// owning `NormKey` (`std::vector<int16_t> data`) is materialized. The write
    /// doors below probe / write the cold map with `StrSpan(out, n)` directly.
    /// The total (`2*(len+2)` bytes) is asserted against @p cap first — the
    /// Rule-19 widen-on-STOP tripwire naming `kMaxNormKeyBytes`.
    ///
    /// @param numberExpressions The key's leading expression-count field.
    /// @param data The key's `int16_t` payload.
    /// @param len  The payload length (`data` has @p len elements).
    /// @param out  Caller stack buffer receiving the key bytes.
    /// @param cap  Capacity of @p out (== `kMaxNormKeyBytes` at callers).
    /// @return The key length written (`== 2*(len+2)`).
    /// @see Codec<NormKey>::encode — the byte-layout owner + the twin oracle;
    ///      mergeOwnerRecord (the raw-key overload), appendLmvIdsRecord.
    inline int32_t encodeNormKeyInto(int16_t numberExpressions,
        const int16_t* data, int32_t len, char* out, int32_t cap) {
        const int32_t total = static_cast<int32_t>(sizeof(int16_t)) * (len + 2);
        assert(total <= cap
            && "encodeNormKeyInto: key exceeds kMaxNormKeyBytes");
        assert(len == 0 || data != nullptr);
        const int16_t length = static_cast<int16_t>(len);
        std::memcpy(out, &numberExpressions, sizeof(int16_t));
        std::memcpy(out + sizeof(int16_t), &length, sizeof(int16_t));
        for (int32_t i = 0; i < len; ++i)
            std::memcpy(out + sizeof(int16_t) * (static_cast<std::size_t>(i) + 2),
                        &data[i], sizeof(int16_t));
        return total;
    }

    /// @brief Build a `Codec<IdVecKey>` byte key from a raw id run into a caller
    ///        stack buffer — the owning-`IdVecKey`-free key encoder.
    ///
    /// @details
    /// Byte-exact to `Codec<IdVecKey>::encode` (`int32 count ++ count x int32`,
    /// fixed-width little-endian), but written straight into @p out from an
    /// `(ids, n)` run, so no owning `IdVecKey` (`std::vector<int32_t> ids`) is
    /// materialized. The caller mints the `originals` set with `StrSpan(out, r)`
    /// directly through `inner().mint`. The total (`4*(n+1)` bytes) is asserted
    /// against @p cap first — the Rule-19 widen-on-STOP tripwire. The ids keep
    /// their positional (rule) order — they are NOT re-sorted — mirroring
    /// `Codec<IdVecKey>::encode`.
    ///
    /// @param ids The id run (positional order).
    /// @param n   The run length (`ids` has @p n elements).
    /// @param out Caller stack buffer receiving the key bytes.
    /// @param cap Capacity of @p out.
    /// @return The key length written (`== 4*(n+1)`).
    /// @see Codec<IdVecKey>::encode — the byte-layout owner + the twin oracle;
    ///      encodeNormKeyInto.
    inline int32_t encodeIdVecKeyInto(const int32_t* ids, int32_t n,
        char* out, int32_t cap) {
        const int32_t total = static_cast<int32_t>(sizeof(int32_t)) * (n + 1);
        assert(total <= cap
            && "encodeIdVecKeyInto: key exceeds buffer capacity");
        assert(n == 0 || ids != nullptr);
        std::memcpy(out, &n, sizeof(int32_t));
        for (int32_t i = 0; i < n; ++i)
            std::memcpy(out + sizeof(int32_t) * (static_cast<std::size_t>(i) + 1),
                        &ids[i], sizeof(int32_t));
        return total;
    }

    /// @brief Append one `LocalMemoryValue` record to a NormKey-keyed encodedMap
    ///        run, serialized DIRECTLY from raw id fields — no owning
    ///        `LocalMemoryValue` and no owning `NormKey`.
    ///
    /// @details
    /// The id-run door for the `encodedMap` marker AND head inserts: it builds
    /// the `Codec<NormKey>` key bytes (`encodeNormKeyInto`) and the
    /// `Codec<LocalMemoryValue>` record blob straight from the caller's id runs
    /// into @p gArena, then appends at the key's run-end through the raw
    /// `inner().appendBlobToRun` / one-blob `inner().assignRun` doors —
    /// byte-identical to `encodedMap.appendRecord(NormKey{...}, LocalMemoryValue{...})`.
    /// The four trailing params default to the MARKER install's field set
    /// (`levels` empty, `justification` none, `productOfDisintegration` false —
    /// the `LocalMemoryValue` defaults), so the marker call site stays
    /// byte-identical; the `addToHashMemory` HEAD install passes the real
    /// involved-level run, justification, and disintegration-product flag. The
    /// blob layout mirrors `Codec<LocalMemoryValue>::serialize` field-for-field:
    /// `valueId`, `originalImplicationId`, `justification`(int32),
    /// `validityId`(int16), `isMarker`(uint8), `productOfDisintegration`(uint8),
    /// `levelN` levels (int32 count + int32 each, ascending), `keyN` + key ids,
    /// `remN` + rem ids.
    ///
    /// @param map              The NormKey-keyed encodedMap.
    /// @param numberExpressions The key's leading count field.
    /// @param nkData           The key's int16 payload.
    /// @param nkLen            The payload length.
    /// @param valueId          The LMV `valueId`.
    /// @param isMarker         The LMV `isMarker` flag.
    /// @param keyIds           The LMV `keyIds` run.
    /// @param keyN             The `keyIds` count.
    /// @param remIds           The LMV `remainingArgIds` run.
    /// @param remN             The `remainingArgIds` count.
    /// @param originalImplId   The LMV `originalImplicationId`.
    /// @param validityId       The LMV `validityId`.
    /// @param gArena           Per-slot gen-scratch arena for the key/blob bytes.
    /// @param levels           The LMV `levels` run (ascending-unique); default
    ///                         empty (marker install).
    /// @param levelN           The `levels` count; default 0.
    /// @param justification    The LMV `justification`; default `none` (marker).
    /// @param productOf        The LMV `productOfDisintegration`; default false.
    /// @see Codec<LocalMemoryValue>::serialize / Codec<NormKey>::encode — the
    ///      byte-layout owners + the twin oracle; encodeNormKeyInto.
    inline void appendLmvIdsRecord(
        TypedColdBlobMap<NormKey, LocalMemoryValue>& map,
        int16_t numberExpressions, const int16_t* nkData, int32_t nkLen,
        int32_t valueId, bool isMarker,
        const int32_t* keyIds, int32_t keyN,
        const int32_t* remIds, int32_t remN,
        int32_t originalImplId, int16_t validityId, ScratchArena& gArena,
        const int* levels = nullptr, int32_t levelN = 0,
        RuleJustification justification = RuleJustification::none,
        bool productOf = false) {
        const ArenaOffset mark = gArena.cursor();

        // 1. NormKey key bytes on the stack.
        char keyBuf[ExecutionParameters::kMaxNormKeyBytes];
        const int32_t kn = encodeNormKeyInto(numberExpressions, nkData, nkLen,
            keyBuf, ExecutionParameters::kMaxNormKeyBytes);
        const StrSpan keyView(keyBuf, kn);

        // 2. Codec<LocalMemoryValue> blob on the arena, field-for-field. The
        //    four trailing params default to the marker install (justification
        //    none, levels empty, productOf false); the head install passes real
        //    values.
        assert(keyN >= 0 && remN >= 0 && levelN >= 0);
        assert(levelN == 0 || levels != nullptr);
        const int32_t blobLen =
            4 + 4 + 4 + 2 + 1 + 1     // valueId,origImpl,just,validity,isMarker,pod
            + 4 + 4 * levelN          // levels count + levels
            + 4 + 4 * keyN            // keyIds count + ids
            + 4 + 4 * remN;           // remIds count + ids
        const ArenaOffset blobOff = gArena.alloc(blobLen, 1);
        char* b = reinterpret_cast<char*>(gArena.resolve(blobOff));
        int32_t at = 0;
        const auto put32 = [&](int32_t v) {
            std::memcpy(b + at, &v, sizeof(int32_t)); at += 4;
        };
        const auto put16 = [&](int16_t v) {
            std::memcpy(b + at, &v, sizeof(int16_t)); at += 2;
        };
        const auto put8 = [&](uint8_t v) { b[at++] = static_cast<char>(v); };
        put32(valueId);
        put32(originalImplId);
        put32(static_cast<int32_t>(justification));
        put16(validityId);
        put8(isMarker ? 1 : 0);
        put8(productOf ? 1 : 0);
        put32(levelN);
        for (int32_t i = 0; i < levelN; ++i) put32(levels[i]);
        put32(keyN);
        for (int32_t i = 0; i < keyN; ++i) put32(keyIds[i]);
        put32(remN);
        for (int32_t i = 0; i < remN; ++i) put32(remIds[i]);
        assert(at == blobLen
            && "appendLmvIdsRecord: blob fill diverged from the layout");

        // 3. Append at the key's run-end (the typed appendRecord idiom).
        const int32_t id = map.inner().lookup(keyView);
        if (id != 0) {
            map.inner().appendBlobToRun(id, b, blobLen);
        } else {
            const int32_t len = blobLen;
            map.inner().assignRun(keyView, b, &len, 1);
        }
        gArena.popTo(mark);
    }

    /// @brief Scratch-arena twin of `encodeEqClassKeyFromView` — build the
    ///        packed eq-class byte key on @p arena, no heap `std::string`.
    ///
    /// @details
    /// Byte-identical to `encodeEqClassKeyFromView` (the validity id followed
    /// by each member id, all little-endian `int16_t`), but the result rides a
    /// `ScratchString` on @p arena's byte-bump tier instead of the malloc heap.
    /// `applyEquiClasses`' per-delta-class key (`deltaKey =
    /// encodeEqClassKeyFromView(validityAt(d), deltaView)` fed to
    /// `deltaIds.mint(StrSpan(deltaKey))` / `.lookup(...)`) keeps the transient
    /// key off the heap. The key is fixed-length (`2 * (memberCount + 1)`
    /// bytes), so it is one `allocBytes` fill and a `ScratchString::wrap` — the
    /// same idiom as `replaceKeysScratch` / `removeUPrefixScratch`.
    ///
    /// @param arena      Per-slot scratch arena receiving the key bytes.
    /// @param validityId The class's validity-scope id.
    /// @param view       The class view (its members are the key).
    /// @return The packed byte key as a `ScratchString` on @p arena.
    /// @see encodeEqClassKeyFromView — the heap-string sibling; encodeEqClassKey.
    inline ScratchString encodeEqClassKeyFromViewScratch(ScratchArena& arena,
        int16_t validityId, const EquivalenceClassView& view) {
        const int32_t total = static_cast<int32_t>(sizeof(int16_t))
            * (view.memberCount() + 1);
        char* buf = arena.allocBytes(total);
        std::memcpy(buf, &validityId, sizeof(int16_t));
        for (int32_t i = 0; i < view.memberCount(); ++i) {
            const int16_t m = view.memberId(i);
            std::memcpy(buf + sizeof(int16_t) * (static_cast<std::size_t>(i) + 1),
                        &m, sizeof(int16_t));
        }
        return ScratchString::wrap(arena, buf, total);
    }

    /// @brief Remove an equivalence class's statement-index entry keyed by a
    ///        cold-blob `EquivalenceClassView`'s members — the heap-free twin of
    ///        `eraseEqClassIndex(m, validityId, class.memberIds)` (the
    ///        merge-away erase for an overlapping class read through its blob).
    ///        A miss is a defined no-op.
    ///
    /// @param m          The cold eq-class index map.
    /// @param validityId The class's validity-scope id.
    /// @param view       The (merged-away) class view.
    /// @see eraseEqClassIndex, encodeEqClassKeyFromViewInto.
    inline void eraseEqClassIndex(TypedColdMap<EqClassKey, int>& m,
                                  int16_t validityId,
                                  const EquivalenceClassView& view) {
        char keyBuf[ExecutionParameters::kMaxEqClassKeyBytes];
        const int32_t n = encodeEqClassKeyFromViewInto(keyBuf,
            ExecutionParameters::kMaxEqClassKeyBytes, validityId, view);
        m.inner().erase(StrSpan(keyBuf, n));
    }

    /// @brief Read the statement-index waterline of a class read through a
    ///        cold-blob `EquivalenceClassView` — the heap-free twin of
    ///        `lookupEqClassIndex(m, validityId, class.memberIds)`.
    ///
    /// @details Serializes the byte key into a stack buffer via
    /// `encodeEqClassKeyFromViewInto` and probes the cold map's byte-key door
    /// (`inner()`) with a raw `StrSpan(keyBuf, n)`, returning 0 on a miss exactly
    /// like the base overload's `findOr(..., 0)`. No heap key `std::string` or
    /// `EqClassKey` `std::vector<int16_t>` is materialized. Non-minting read;
    /// burst-safe.
    ///
    /// @param m          The cold eq-class index map.
    /// @param validityId The class's validity-scope id.
    /// @param view       The class view (its members are the key).
    /// @return The stored waterline, or 0 when the class has no entry.
    /// @see lookupEqClassIndex, encodeEqClassKeyFromViewInto, upsertEqClassIndex.
    inline int lookupEqClassIndex(
        const TypedColdMap<EqClassKey, int>& m, int16_t validityId,
        const EquivalenceClassView& view) {
        char keyBuf[ExecutionParameters::kMaxEqClassKeyBytes];
        const int32_t n = encodeEqClassKeyFromViewInto(keyBuf,
            ExecutionParameters::kMaxEqClassKeyBytes, validityId, view);
        const int32_t id = m.inner().lookup(StrSpan(keyBuf, n));
        return id != 0 ? m.inner().valueAt(id) : 0;
    }

    /// @brief Set the statement-index waterline of a class read through a
    ///        cold-blob `EquivalenceClassView` — the heap-free twin of
    ///        `upsertEqClassIndex(m, validityId, class.memberIds, value)`.
    ///
    /// @details Serializes the byte key into a stack buffer via
    /// `encodeEqClassKeyFromViewInto` and drives the write through the cold map's
    /// byte-key door (`inner()`) with a raw `StrSpan(keyBuf, n)` probe,
    /// reproducing `TypedCold::upsert` (in-place `setValueAt` on a hit, set-once
    /// `insert` on a miss) without a heap key `std::string` or an `EqClassKey`
    /// `std::vector<int16_t>`. Single-threaded write side only (I-83).
    ///
    /// @param m          The cold eq-class index map.
    /// @param validityId The class's validity-scope id.
    /// @param view       The class view (its members are the key).
    /// @param value      The waterline to store.
    /// @see upsertEqClassIndex, encodeEqClassKeyFromViewInto, lookupEqClassIndex.
    inline void upsertEqClassIndex(TypedColdMap<EqClassKey, int>& m,
                                   int16_t validityId,
                                   const EquivalenceClassView& view, int value) {
        char keyBuf[ExecutionParameters::kMaxEqClassKeyBytes];
        const int32_t n = encodeEqClassKeyFromViewInto(keyBuf,
            ExecutionParameters::kMaxEqClassKeyBytes, validityId, view);
        auto& inner = m.inner();
        const int32_t id = inner.lookup(StrSpan(keyBuf, n));
        if (id != 0) inner.setValueAt(id, value);
        else inner.insert(StrSpan(keyBuf, n), value);
    }

    /// @brief Record codec for `EquivalenceClass` — the `ColdBlobMap` value
    ///        serializer the typed blob map (`TypedColdBlobMap<int16_t,
    ///        EquivalenceClass>`) calls; forwards to the canonical
    ///        `serializeEquivalenceClass` / `deserializeEquivalenceClass`.
    ///
    /// @see serializeEquivalenceClass, deserializeEquivalenceClass,
    ///      D-171.
    template <>
    struct Codec<EquivalenceClass> {
        /// @brief Serialize a class to its canonical blob.
        ///
        /// @param c The class.
        /// @return The canonical byte blob.
        static std::vector<char> serialize(const EquivalenceClass& c) {
            return serializeEquivalenceClass(c);
        }

        /// @brief Deserialize a class from its blob.
        ///
        /// @param p Blob bytes.
        /// @param n Blob length.
        /// @return The decoded class.
        static EquivalenceClass deserialize(const char* p, int32_t n) {
            return deserializeEquivalenceClass(p, n);
        }
    };

    /// @brief Record codec for one `IdOrigin` history line — the `ColdBlobMap`
    ///        value serializer for the standalone cold `Memory::exprOriginMap`
    ///        (`TypedColdBlobMap<int64_t, IdOrigin>`).
    ///
    /// @details
    /// Where `Codec<EquivalenceClass>` folds an entire `IdOriginMap` into ONE blob
    /// (keys + counts inline) because that map is a sub-field of a class record, the
    /// standalone origin map lets the cold store own the key column and the per-key
    /// run structure directly, so each blob is ONE history line. The byte layout is
    /// therefore exactly the per-line stream `serializeEquivalenceClass` emits —
    /// `uint8 OriginTag`, `int32 depCount`, `depCount × int64 depKey` — so an
    /// equi-class origin blob and a body origin blob share one canonical line
    /// layout. All integers are fixed-width little-endian (the deload-format
    /// assumption, every GL target is x64 LE); the byte stream is a pure function of
    /// content (I-103). The dependency vector is
    /// positional and OBSERVABLE (dump / chapter export), so it is emitted verbatim.
    ///
    /// @see serializeEquivalenceClass, `IdOrigin`, `BlobCsrValueStore`,
    ///      D-169, I-121.
    template <>
    struct Codec<IdOrigin> {
        /// @brief Serialize one history line to its canonical blob.
        ///
        /// @param line The `(tag, packed dependency keys)` record.
        /// @return The canonical byte blob; `deserialize` is its exact inverse.
        static std::vector<char> serialize(const IdOrigin& line) {
            std::vector<char> out;
            const auto put = [&out](const auto& x) {
                const char* p = reinterpret_cast<const char*>(&x);
                out.insert(out.end(), p, p + sizeof(x));
            };
            put(static_cast<uint8_t>(line.first));
            put(static_cast<int32_t>(line.second.size()));
            for (const int64_t dep : line.second) put(dep);
            return out;
        }

        /// @brief Reconstruct one history line from its blob.
        ///
        /// @details The exact inverse of `serialize`: lossless, so
        /// `serialize(deserialize(b)) == b` for any blob `b` this codec produced.
        /// A blob shorter than the layout demands or longer than it consumes is a
        /// hard `assert` (Rule 19), never a truncated decode.
        ///
        /// @param data The blob bytes.
        /// @param n    The blob length.
        /// @return The decoded history line.
        static IdOrigin deserialize(const char* data, int32_t n) {
            const char* cur = data;
            const char* const end = data + n;
            const auto get = [&cur, end](auto& v) {
                assert(cur + sizeof(v) <= end
                    && "Codec<IdOrigin>::deserialize: blob shorter than the layout");
                std::memcpy(&v, cur, sizeof(v));
                cur += sizeof(v);
            };
            uint8_t tag = 0; get(tag);
            int32_t depCount = 0; get(depCount);
            std::vector<int64_t> deps;
            deps.reserve(static_cast<size_t>(depCount));
            for (int32_t d = 0; d < depCount; ++d) {
                int64_t dep = 0; get(dep); deps.push_back(dep);
            }
            assert(cur == end
                && "Codec<IdOrigin>::deserialize: blob longer than the layout");
            return IdOrigin(static_cast<OriginTag>(tag), std::move(deps));
        }
    };

    /// @brief Record codec for one committed mail batch — the value type of the
    ///        statified mail log's per-LB blob run (`mail_log.hpp`).
    ///
    /// @details
    /// Serializes ONLY `Mail::statements` and `Mail::exprOriginMap` — the exact
    /// two fields a committed batch carries (matching `MailLog::commit` /
    /// `mergeBatchInto`; `disintegrationSignals` never crosses the log, so it is
    /// dropped — a `deserialize`d batch leaves it empty). Statements travel as
    /// GLOBAL `mailInterner` ids (cross-batch interning, decoded at the receiver);
    /// origins travel the same way (EWV pairs as global ids, the tag as its
    /// `OriginTag` byte). The blob moves the bulk mail
    /// content onto the dedicated, never-deloaded mail pool; a phase-1 pull
    /// decodes it into a transient heap `Mail` and folds that into the receiver's
    /// heap `mailIn` via `mergeBatchInto`. The byte layout is fixed-width
    /// little-endian length-prefixed (one `int32` count per container, one
    /// `int32` length per string), deterministic (sets / maps iterate sorted,
    /// origin vectors round-trip in order) — though the mail pool is never
    /// deloaded, so the canonical-bytes obligation (I-103,
    /// which governs deload streams) does NOT bind it; determinism here is only
    /// for testability and a lossless round trip.
    ///
    /// @see Mail, MailLog (`mail_log.hpp`), mergeBatchInto, Codec<IdOrigin>.
    // mailInterner() is declared in full further below (beside skeletonInterner);
    // forward-declared here so Codec<Mail> + the routing toHeap helpers can name it
    // (they intern / decode mail statements through the global interner).
    ColdStringTable& mailInterner();

    // RoutingColdMail (Memory::mailIn) and DeloadableMailOut (Memory::mailOut)
    // are complete through lb_memory.hpp; their Codec doors are declared below.
    template <>
    struct Codec<Mail> {
        /// @brief Serialize a batch's statements + exprOriginMap to a blob.
        ///
        /// @param batch The committed batch (only the two logged fields are read).
        /// @return The byte blob; `deserialize` is its exact inverse.
        static std::vector<char> serialize(const Mail& batch) {
            std::vector<char> out;
            const auto putPod = [&out](const auto& x) {
                const char* p = reinterpret_cast<const char*>(&x);
                out.insert(out.end(), p, p + sizeof(x));
            };
            // Every mail EWV travels as two GLOBAL mailInterner ids (the cross-LB
            // carrier id-space). Single-threaded callers only (commit seam /
            // broadcast), so the mailInterner mint is race-free.
            const auto putGlobalEwv = [&putPod](const ExpressionWithValidity& e) {
                putPod(mailInterner().intern(e.original));
                putPod(mailInterner().intern(e.validityName));
            };
            // statements: set<pair<EWV, set<int>>>
            putPod(static_cast<int32_t>(batch.statements.size()));
            for (const std::pair<ExpressionWithValidity, std::set<int>>& st
                 : batch.statements) {
                putGlobalEwv(st.first);
                putPod(static_cast<int32_t>(st.second.size()));
                for (const int lv : st.second) putPod(static_cast<int32_t>(lv));
            }
            // exprOriginMap: each EWV key / dep as global ids, the tag as its
            // OriginTag byte (written as int32).
            putPod(static_cast<int32_t>(batch.exprOriginMap.size()));
            for (const std::pair<const ExpressionWithValidity,
                     std::vector<std::pair<std::string,
                         std::vector<ExpressionWithValidity>>>>& keyed
                 : batch.exprOriginMap) {
                putGlobalEwv(keyed.first);
                putPod(static_cast<int32_t>(keyed.second.size()));
                for (const std::pair<std::string,
                         std::vector<ExpressionWithValidity>>& origin
                     : keyed.second) {
                    putPod(static_cast<int32_t>(
                        originTagFromString(StrSpan(origin.first))));
                    putPod(static_cast<int32_t>(origin.second.size()));
                    for (const ExpressionWithValidity& dep : origin.second)
                        putGlobalEwv(dep);
                }
            }
            return out;
        }

        /// @brief Reconstruct a batch from its blob (the two dropped fields stay
        ///        empty).
        ///
        /// @details Exact inverse of `serialize`; a blob shorter or longer than
        /// the layout is a hard `assert` (Rule 19), never a truncated decode.
        ///
        /// @param data The blob bytes.
        /// @param n    The blob length.
        /// @return The decoded batch (statements + exprOriginMap only).
        static Mail deserialize(const char* data, int32_t n) {
            const char* cur = data;
            const char* const end = data + n;
            const auto getPod = [&cur, end](auto& v) {
                assert(cur + sizeof(v) <= end
                    && "Codec<Mail>::deserialize: blob shorter than the layout");
                std::memcpy(&v, cur, sizeof(v));
                cur += sizeof(v);
            };
            const auto getGlobalEwv = [&getPod]() {
                int32_t o = 0; getPod(o);
                int32_t v = 0; getPod(v);
                return ExpressionWithValidity(mailInterner().decodeString(o),
                                              mailInterner().decodeString(v));
            };
            Mail batch;
            int32_t stCount = 0; getPod(stCount);
            for (int32_t i = 0; i < stCount; ++i) {
                // statements travel as two GLOBAL mailInterner ids; decode here.
                ExpressionWithValidity ev = getGlobalEwv();
                int32_t lvCount = 0; getPod(lvCount);
                std::set<int> levels;
                for (int32_t j = 0; j < lvCount; ++j) {
                    int32_t lv = 0; getPod(lv); levels.insert(lv);
                }
                batch.statements.insert(
                    std::make_pair(std::move(ev), std::move(levels)));
            }
            int32_t keyCount = 0; getPod(keyCount);
            for (int32_t k = 0; k < keyCount; ++k) {
                ExpressionWithValidity key = getGlobalEwv();
                int32_t oCount = 0; getPod(oCount);
                std::vector<std::pair<std::string,
                    std::vector<ExpressionWithValidity>>> origins;
                origins.reserve(static_cast<size_t>(oCount));
                for (int32_t o = 0; o < oCount; ++o) {
                    int32_t tag = 0; getPod(tag);
                    std::string label =
                        originTagName(static_cast<OriginTag>(tag));
                    int32_t depCount = 0; getPod(depCount);
                    std::vector<ExpressionWithValidity> deps;
                    deps.reserve(static_cast<size_t>(depCount));
                    for (int32_t d = 0; d < depCount; ++d)
                        deps.push_back(getGlobalEwv());
                    origins.push_back(
                        std::make_pair(std::move(label), std::move(deps)));
                }
                batch.exprOriginMap.emplace(std::move(key), std::move(origins));
            }
            assert(cur == end
                && "Codec<Mail>::deserialize: blob longer than the layout");
            return batch;
        }

        /// @brief Serialize a `DeloadableMailOut` to a committed mail blob.
        ///        byte-identical to `serialize(hm.toHeapMail())`, no transient
        ///        heap `Mail`.
        ///
        /// @details Reads both id-form columns through the mailbox's dedicated
        /// per-LB interner, canonical-sorts their decoded rows, and emits the
        /// exact `serialize(const Mail&)` layout with global mail ids. Defined
        /// out-of-line after the mail decode helpers.
        ///
        /// @param hm The deloadable mailbox being committed.
        /// @param senderMi The mailbox's dedicated per-LB interner.
        /// @return The byte blob; `deserializeInto` is its inverse.
        static std::vector<char> serialize(const DeloadableMailOut& hm,
                                           const ValueInterner& senderMi);

        /// @brief Decode a blob straight into a `RoutingColdMail` inbox via its write
        ///        doors — the fused `deserialize` + `mergeBatchInto(const Mail&,
        ///        RoutingColdMail&)`, 0% heap: no transient `Mail` and no
        ///        `IntMailOrigin`.
        ///
        /// @details The retained char* oracle, now a thin wrapper over the shared
        /// parse body `deserializeMailBlobInto` (a `CharMailSource` over the
        /// contiguous `[data, data + n)` window): it walks the same wire layout as
        /// `deserialize`, routing each decoded statement through
        /// `RoutingColdMail::insertStatement` and each decoded origin through the POD
        /// `addMailOriginRecord(origins_, …, INT_MAX)` — so the result equals
        /// `mergeBatchInto(deserialize(data, n), inbox)` byte-for-byte, with no
        /// owning container built. Merge order is irrelevant (statements set-merge;
        /// origins fold uncapped and the absorb re-sorts under the D-49 cap).
        /// Over/under-run is a hard `assert` (Rule 19). Defined out-of-line below
        /// `RoutingColdMail`.
        ///
        /// @param data  The blob bytes.
        /// @param n     The blob length.
        /// @param inbox The receiver's `mailIn` (a `RoutingColdMail`), folded in place.
        /// @see deserializeMailBlobInto — the shared parse body; CharMailSource.
        static void deserializeInto(const char* data, int32_t n, RoutingColdMail& inbox);

        /// @brief Decode a blob straight from the mail blob pool into a
        ///        `RoutingColdMail` inbox — the production pool-native twin of
        ///        `deserializeInto(const char*, ...)`, 0% heap and buffer-free.
        ///
        /// @details The production sink `MailLog::readBlobInto` forwards to this:
        /// it wraps a `PoolMailSource` over `pool[start, start + len)` and drives
        /// the shared parse body `deserializeMailBlobInto`, reading each field
        /// straight off the (possibly page-straddling) pool with NO
        /// `std::vector<char>` reassembly buffer. Byte-identical result to
        /// `deserializeInto(<reassembled bytes>, len, inbox)` (twin
        /// `deserialize_into_pool_matches_char`, incl. a forced mid-field
        /// straddle). Over/under-run is a hard `assert` (Rule 19). Defined
        /// out-of-line below `RoutingColdMail` + `PoolMailSource`.
        ///
        /// @param pool  The mail blob pool (`MailLog::mailBlobPool`).
        /// @param start Byte offset of the blob in `pool`.
        /// @param len   Blob length in bytes.
        /// @param inbox The receiver's `mailIn` (a `RoutingColdMail`), folded in place.
        /// @see deserializeMailBlobInto — the shared parse body; PoolMailSource.
        static void deserializeInto(const PagedVector<char>& pool,
            std::uint32_t start, std::uint32_t len, RoutingColdMail& inbox);
    };

    // ====================================================================
    //  Cold-map twins of the id-form origin helpers. The standalone
    //  Memory::exprOriginMap is a cold TypedColdBlobMap<int64_t, IdOrigin>
    //  (one history line = one blob); these overloads carry the identical
    //  observable semantics of the IdOriginMap helpers above onto it via a
    //  decode-run / apply-policy / write-run RMW. EquivalenceClass::
    //  equalityOriginMap stays a heap IdOriginMap transient decode and keeps
    //  the IdOriginMap overloads. Overload resolution routes each call site by
    //  the map's type — a site passing mb.exprOriginMap (cold) binds here, a
    //  site passing a class's equalityOriginMap (heap) binds above; the only
    //  sites that must change are the ad-hoc ones the heap container's raw API
    //  (.find / .swap / range-for) served, which do not compile against the
    //  cold surface. I-121.
    // ====================================================================

    /// @brief Serialize one `IdOrigin` history line into a caller buffer — the
    ///        stack-buffer twin of `Codec<IdOrigin>::serialize`.
    ///
    /// @details
    /// Emits the codec's exact canonical frame — `uint8 tag`, `int32 depCount`,
    /// `depCount x int64 dep` — byte-for-byte into `buf`, with the closed-form
    /// length `5 + 8 * depCount` asserted against both the dependency ceiling
    /// (`ExecutionParameters::kMaxOriginDeps`, a Rule-19 tripwire — realistic
    /// dependency counts are single-digit) and the buffer capacity, and an
    /// exact-fill assert after the write. No heap, no arena. The
    /// `serializeMailOriginTo` sibling serves the `IntMailOrigin` blob owner —
    /// the two record types keep separate helpers on purpose (distinct codecs;
    /// no over-unification).
    ///
    /// @param buf  Caller-owned destination buffer.
    /// @param cap  The buffer's capacity in bytes.
    /// @param line The `(tag, deps)` history line to serialize.
    /// @return The blob's byte length (`5 + 8 * deps.size()`).
    /// @invariant The written bytes equal `Codec<IdOrigin>::serialize(line)`
    ///            exactly (twin test `serialize_origin_to_matches_codec`).
    /// @see Codec<IdOrigin> — the heap oracle; addOriginId — the consuming RMW;
    ///      serializeMailOriginTo — the `IntMailOrigin` sibling.
    inline int32_t serializeOriginTo(char* buf, int32_t cap,
        const IdOrigin& line) {
        const int32_t depN = static_cast<int32_t>(line.second.size());
        assert(depN <= ExecutionParameters::kMaxOriginDeps
            && "serializeOriginTo: dependency count exceeds kMaxOriginDeps "
               "(Rule-19 tripwire — widen the constant with evidence)");
        const int32_t len = 5 + 8 * depN;
        assert(len <= cap
            && "serializeOriginTo: blob exceeds the caller buffer capacity");
        char* at = buf;
        *at++ = static_cast<char>(static_cast<uint8_t>(line.first));
        std::memcpy(at, &depN, sizeof(int32_t));
        at += sizeof(int32_t);
        for (const int64_t d : line.second) {
            std::memcpy(at, &d, sizeof(int64_t));
            at += sizeof(int64_t);
        }
        assert(at == buf + len
            && "serializeOriginTo: fill diverged from the closed-form length");
        return len;
    }

    /// @brief POD overload of @ref serializeOriginTo — serialize a `(tag, deps)`
    ///        history line from raw fields, no `IdOrigin` materialized.
    ///
    /// @details
    /// Emits the identical canonical frame as the `IdOrigin` overload —
    /// `uint8 tag`, `int32 depCount`, `depCount x int64 dep` — from a raw tag
    /// byte + int64 dependency run, so a caller holding the packed dependency
    /// keys never builds an `IdOrigin{tag, deps}` heap vector. Same closed-form
    /// length `5 + 8 * depN`, same Rule-19 `kMaxOriginDeps` and capacity asserts,
    /// same exact-fill assert.
    ///
    /// @param buf  Caller-owned destination buffer.
    /// @param cap  The buffer's capacity in bytes.
    /// @param tag  The history-line tag byte.
    /// @param deps Pointer to `depN` dependency packed keys, positional order.
    /// @param depN The dependency count.
    /// @return The blob's byte length (`5 + 8 * depN`).
    /// @invariant The written bytes equal `serializeOriginTo(buf, cap,
    ///            IdOrigin{OriginTag(tag), {deps..}})` exactly (twin test
    ///            `serialize_origin_to_pod_matches_idorigin`).
    /// @see serializeOriginTo(char*, int32_t, const IdOrigin&) — the `IdOrigin`
    ///      overload this reproduces byte-for-byte.
    inline int32_t serializeOriginTo(char* buf, int32_t cap, uint8_t tag,
        const int64_t* deps, int32_t depN) {
        assert(depN <= ExecutionParameters::kMaxOriginDeps
            && "serializeOriginTo: dependency count exceeds kMaxOriginDeps "
               "(Rule-19 tripwire — widen the constant with evidence)");
        const int32_t len = 5 + 8 * depN;
        assert(len <= cap
            && "serializeOriginTo: blob exceeds the caller buffer capacity");
        char* at = buf;
        *at++ = static_cast<char>(tag);
        std::memcpy(at, &depN, sizeof(int32_t));
        at += sizeof(int32_t);
        for (int32_t i = 0; i < depN; ++i) {
            std::memcpy(at, &deps[i], sizeof(int64_t));
            at += sizeof(int64_t);
        }
        assert(at == buf + len
            && "serializeOriginTo: fill diverged from the closed-form length");
        return len;
    }

    /// @brief Replace `exprOriginMap[key]`'s cold run with the class-then-body
    ///        origin merge, 0% heap — the serialize+`memcmp` raw-door twin of the
    ///        former heap `newRun` + typed `assignRun` merge.
    ///
    /// @details
    /// Reproduces `updateEquivalenceClasses`'s origin sync byte-for-byte without a
    /// heap `std::vector<IdOrigin> newRun` / `recordsAt` body snapshot: the run is
    /// ALL class lines (in @ref MergeClassAccum::LineView order, no dedup — the
    /// accumulator already deduped/capped), THEN the body's existing lines appended
    /// in run order, each only if not already present and while `accepted < cap` —
    /// exactly the heap `merged := classOrigins; overwriteOriginsId(merged, body)`
    /// order.
    ///
    /// Staging rides @p arena's byte-bump tier under a function-scoped
    /// `ScratchScope`, nested ABOVE the caller's `forEachOriginSorted` reused
    /// `okeys`/`idx` allocations (they byte-bump the SAME accumulator arena before
    /// invoking the visitor, so this rewind cannot free them — LOAD-BEARING).
    /// Pass 1 serializes each class line (`serializeOriginTo`, byte-exact to
    /// `Codec<IdOrigin>::serialize`) into a `cap * kMaxOriginBlobBytes` buffer
    /// (`classLines.size() <= cap` by the accumulator's per-key cap, asserted);
    /// pass 2 peeks each body blob (`peekRecordBytes`, a page straddle spilling
    /// onto @p arena ABOVE the buffer), dedups by whole-blob `memcmp` (the codec is
    /// injective, so byte-equality IS `IdOrigin` equality — the twin of the heap
    /// `std::find` `IdOrigin::operator==`), and appends below the cap; one raw
    /// `inner().assignRun` COPIES the buffer into the map's own blob pool (a
    /// different arena) before the scope rewinds, so nothing dangles.
    ///
    /// The merge mints NOTHING — it operates on already-interned `int64`
    /// dependency keys shared by class and body — so the produced cold blobs (and
    /// the id-ordered deload stream) are byte-identical to `assignRun(key, newRun)`.
    /// `exprOriginMap` is verifier-walked process documentation (Rule 16); its
    /// content is unchanged.
    ///
    /// @param exprOriginMap The per-LB cold origin blob map (written in place).
    /// @param key           The packed `(exprId, validityId)` origin key.
    /// @param classLines    This key's class history lines (the accumulator view).
    /// @param cap           Per-key origin cap (`max_origin_per_expr` /
    ///                      `compressor_max_origins_per_expr`).
    /// @param arena         The accumulator's byte-bump scratch arena (staging +
    ///                      page-straddle peek spill).
    /// @invariant Emits the EXACT bytes the typed `assignRun(key, newRun)` would;
    ///            the twin `expr_origin_run_replace_matches_heap` enforces it.
    /// @see serializeOriginTo, MergeClassAccum::forEachOriginSorted,
    ///      MergeClassAccum::LineView, ExecutionParameters::kMaxOriginBlobBytes.
    inline void exprOriginRunReplace(
        TypedColdBlobMap<int64_t, IdOrigin>& exprOriginMap, int64_t key,
        const MergeClassAccum::LineView& classLines, int cap,
        ScratchArena& arena) {
        ScratchScope lineScope(arena);
        assert(classLines.size() <= cap
            && "exprOriginRunReplace: class origin lines exceed the per-key cap "
               "(accumulator cap contract broke)");
        const int32_t bufCap = cap * ExecutionParameters::kMaxOriginBlobBytes;
        char* buf = arena.resolve(arena.alloc(bufCap, 1));
        int32_t* lens = reinterpret_cast<int32_t*>(arena.resolve(
            arena.alloc(cap * static_cast<int32_t>(sizeof(int32_t)),
                        static_cast<int32_t>(alignof(int32_t)))));
        int32_t accepted = 0, off = 0;

        // Pass 1 — all class lines, in LineView order, no dedup (== the heap
        // `newRun := classLines` copy).
        for (int32_t t = 0; t < classLines.size(); ++t) {
            int64_t depBuf[ExecutionParameters::kMaxOriginDeps];
            const int32_t dn = classLines.depCount(t);
            assert(dn <= ExecutionParameters::kMaxOriginDeps
                && "exprOriginRunReplace: class line dep count exceeds "
                   "kMaxOriginDeps");
            for (int32_t d = 0; d < dn; ++d) depBuf[d] = classLines.dep(t, d);
            const int32_t w = serializeOriginTo(buf + off, bufCap - off,
                static_cast<uint8_t>(classLines.tag(t)), depBuf, dn);
            lens[accepted++] = w;
            off += w;
        }

        // Pass 2 — body lines: dedup-append against ALL accepted lines by whole-
        // blob memcmp (== the heap `std::find` `IdOrigin` equality), cap-gated (==
        // the heap `newRun.size() >= cap` break).
        const int32_t bid = exprOriginMap.lookup(key);
        if (bid != 0) {
            const int32_t rl = exprOriginMap.runLen(bid);
            for (int32_t j = 0; j < rl; ++j) {
                if (accepted >= cap) break;
                int32_t blen = 0;
                const char* bp = exprOriginMap.peekRecordBytes(bid, j, blen, arena);
                bool dup = false;
                for (int32_t k = 0, o2 = 0; k < accepted; o2 += lens[k], ++k)
                    if (lens[k] == blen
                        && std::memcmp(buf + o2, bp,
                                       static_cast<std::size_t>(blen)) == 0) {
                        dup = true;
                        break;
                    }
                if (!dup) {
                    assert(off + blen <= bufCap
                        && "exprOriginRunReplace: merged origin run exceeds buf");
                    std::memcpy(buf + off, bp, static_cast<std::size_t>(blen));
                    lens[accepted++] = blen;
                    off += blen;
                }
            }
        }
        assert(accepted <= cap);
        exprOriginMap.inner().assignRun(key, buf, lens, accepted);
    }

    /// @brief Cold twin of `addOriginId` — append a history line to one key's
    ///        run with the D-49 cap-full preference, on the cold exprOriginMap.
    ///
    /// @details
    /// Applies the EXACT policy of the `IdOriginMap` twin (below the cap:
    /// append if absent; at the cap: a new non-equality record overwrites the
    /// first `equality1`/`equality2` convenience slot, else the existing slot
    /// wins). Record equality on ids IS string equality (the interner is
    /// bijective), and the codec is injective so blob byte-equality IS record
    /// equality. Single-threaded write sites only (I-83).
    ///
    /// The interior is heap-free on fixed-capacity STACK buffers (no arena, no
    /// per-slot scratch), mirroring `addMailOriginRecord`: the new line
    /// serializes once into a `kMaxOriginBlobBytes` frame
    /// (`serializeOriginTo`); one scan peeks each existing blob through the
    /// caller-buffer `peekRecordBytes` door (dedup by whole-blob `memcmp` +
    /// the first convenience-slot index — the tag is the blob's first byte);
    /// the append paths take the raw `appendBlobToRun` / one-blob `assignRun`
    /// doors; the rare cap-full convenience-replace assembles survivors
    /// verbatim + the replacement into ONE `kMaxOriginRunBytes` stack frame
    /// and writes it back through the raw whole-run `assignRun` door —
    /// byte-identical to the retired decode-into-`std::vector`-reserialize
    /// cycle (twin test `add_origin_id_cold_rmw_matches_heap`). The rebuild's
    /// run-length assert against `kMaxOriginRunBlobs` is a Rule-19 tripwire
    /// (origin runs are config-capped at 30).
    ///
    /// @param map        The cold origin map.
    /// @param key        Packed (expressionId, validityId) key.
    /// @param origin     The id-form record to insert.
    /// @param maxOrigins Cap; caller picks the compressor or standard parameter.
    inline void addOriginId(TypedColdBlobMap<int64_t, IdOrigin>& map, int64_t key,
        const IdOrigin& origin, int maxOrigins)
    {
        char newBlob[ExecutionParameters::kMaxOriginBlobBytes];
        const int32_t newLen = serializeOriginTo(
            newBlob, ExecutionParameters::kMaxOriginBlobBytes, origin);

        const auto isEqualityConvenienceTag = [](uint8_t tag) {
            return tag == static_cast<uint8_t>(OriginTag::equality1)
                || tag == static_cast<uint8_t>(OriginTag::equality2);
        };

        // One scan over the existing run: duplicate detection (blob byte
        // equality == IdOrigin equality, the codec is injective) and the
        // first equality-convenience slot for the cap-full replace.
        const int32_t id = map.lookup(key);
        const int32_t rl = id ? map.runLen(id) : 0;
        int32_t convenienceIdx = -1;
        {
            char blobBuf[ExecutionParameters::kMaxOriginBlobBytes];
            for (int32_t j = 0; j < rl; ++j) {
                int32_t bl = 0;
                const char* bp = map.peekRecordBytes(id, j, bl, blobBuf,
                    ExecutionParameters::kMaxOriginBlobBytes);
                if (bl == newLen
                    && std::memcmp(bp, newBlob,
                                   static_cast<std::size_t>(bl)) == 0) {
                    return;  // duplicate — the no-op insert
                }
                if (convenienceIdx < 0
                    && isEqualityConvenienceTag(static_cast<uint8_t>(bp[0]))) {
                    convenienceIdx = j;
                }
            }
        }

        if (rl < maxOrigins) {
            // Below the cap: splice the one new blob at the run-end (existing
            // key) or open a one-blob run (new key) — the raw doors, byte-
            // identical to the typed appendRecord / one-record assignRun.
            if (id) map.inner().appendBlobToRun(id, newBlob, newLen);
            else map.inner().assignRun(key, newBlob, &newLen, 1);
            return;
        }

        // Cap-full origin-preference replacement (D-49): foundation displaces
        // convenience — see the string twin for the full policy rationale; a
        // convenience-tag record or a run with no convenience slot leaves the
        // existing run untouched (insertion-order tiebreak, existing wins).
        if (isEqualityConvenienceTag(static_cast<uint8_t>(origin.first))) return;
        if (convenienceIdx < 0) return;
        assert(rl <= ExecutionParameters::kMaxOriginRunBlobs
            && "addOriginId: cap-full run exceeds kMaxOriginRunBlobs "
               "(Rule-19 tripwire — widen the constant with evidence)");
        char runBuf[ExecutionParameters::kMaxOriginRunBytes];
        int32_t lens[ExecutionParameters::kMaxOriginRunBlobs];
        int32_t at = 0;
        for (int32_t j = 0; j < rl; ++j) {
            if (j == convenienceIdx) {
                std::memcpy(runBuf + at, newBlob,
                            static_cast<std::size_t>(newLen));
                lens[j] = newLen;
                at += newLen;
                continue;
            }
            int32_t bl = 0;
            const char* bp = map.peekRecordBytes(id, j, bl, runBuf + at,
                ExecutionParameters::kMaxOriginRunBytes - at);
            assert(at + bl <= ExecutionParameters::kMaxOriginRunBytes
                && "addOriginId: survivor run exceeds kMaxOriginRunBytes");
            if (bp != runBuf + at) {
                std::memcpy(runBuf + at, bp, static_cast<std::size_t>(bl));
            }
            lens[j] = bl;
            at += bl;
        }
        map.inner().assignRun(key, runBuf, lens, rl);
    }

    /// @brief POD overload of the cold @ref addOriginId — append a `(tag, deps)`
    ///        history line with the D-49 cap-full preference, no `IdOrigin`
    ///        materialized.
    ///
    /// @details
    /// Byte-identical to the `const IdOrigin&` cold overload: the new line
    /// serializes once via the POD `serializeOriginTo(buf, cap, tag, deps, depN)`
    /// (no `IdOrigin{tag, deps}` heap vector), then the SAME whole-blob-memcmp
    /// dedup scan, the SAME below-cap `appendBlobToRun` / one-blob `assignRun`
    /// splice, and the SAME D-49 cap-full convenience-replace assembling
    /// survivors verbatim + the replacement into one `kMaxOriginRunBytes` stack
    /// frame for a single raw whole-run `assignRun`. The convenience-tag guard
    /// reads @p tag directly. Single-threaded write sites only
    /// ([I-83](../../docs/agentic_swdd/30_invariants.md#i-83)).
    ///
    /// @param map        The cold origin map.
    /// @param key        Packed (expressionId, validityId) key.
    /// @param tag        The history-line tag byte.
    /// @param deps       Pointer to `depN` dependency packed keys, positional
    ///                   order.
    /// @param depN       The dependency count.
    /// @param maxOrigins Cap; caller picks the compressor or standard parameter.
    /// @invariant The resulting run equals the `const IdOrigin&` cold overload on
    ///            the equivalent record (twin test
    ///            `add_origin_id_cold_pod_matches_idorigin`).
    /// @see addOriginId(TypedColdBlobMap<int64_t, IdOrigin>&, int64_t, const
    ///      IdOrigin&, int) — the `IdOrigin` overload this reproduces.
    inline void addOriginId(TypedColdBlobMap<int64_t, IdOrigin>& map, int64_t key,
        uint8_t tag, const int64_t* deps, int32_t depN, int maxOrigins)
    {
        char newBlob[ExecutionParameters::kMaxOriginBlobBytes];
        const int32_t newLen = serializeOriginTo(
            newBlob, ExecutionParameters::kMaxOriginBlobBytes, tag, deps, depN);

        const auto isEqualityConvenienceTag = [](uint8_t t) {
            return t == static_cast<uint8_t>(OriginTag::equality1)
                || t == static_cast<uint8_t>(OriginTag::equality2);
        };

        const int32_t id = map.lookup(key);
        const int32_t rl = id ? map.runLen(id) : 0;
        int32_t convenienceIdx = -1;
        {
            char blobBuf[ExecutionParameters::kMaxOriginBlobBytes];
            for (int32_t j = 0; j < rl; ++j) {
                int32_t bl = 0;
                const char* bp = map.peekRecordBytes(id, j, bl, blobBuf,
                    ExecutionParameters::kMaxOriginBlobBytes);
                if (bl == newLen
                    && std::memcmp(bp, newBlob,
                                   static_cast<std::size_t>(bl)) == 0) {
                    return;  // duplicate — the no-op insert
                }
                if (convenienceIdx < 0
                    && isEqualityConvenienceTag(static_cast<uint8_t>(bp[0]))) {
                    convenienceIdx = j;
                }
            }
        }

        if (rl < maxOrigins) {
            if (id) map.inner().appendBlobToRun(id, newBlob, newLen);
            else map.inner().assignRun(key, newBlob, &newLen, 1);
            return;
        }

        if (isEqualityConvenienceTag(tag)) return;
        if (convenienceIdx < 0) return;
        assert(rl <= ExecutionParameters::kMaxOriginRunBlobs
            && "addOriginId: cap-full run exceeds kMaxOriginRunBlobs "
               "(Rule-19 tripwire — widen the constant with evidence)");
        char runBuf[ExecutionParameters::kMaxOriginRunBytes];
        int32_t lens[ExecutionParameters::kMaxOriginRunBlobs];
        int32_t at = 0;
        for (int32_t j = 0; j < rl; ++j) {
            if (j == convenienceIdx) {
                std::memcpy(runBuf + at, newBlob,
                            static_cast<std::size_t>(newLen));
                lens[j] = newLen;
                at += newLen;
                continue;
            }
            int32_t bl = 0;
            const char* bp = map.peekRecordBytes(id, j, bl, runBuf + at,
                ExecutionParameters::kMaxOriginRunBytes - at);
            assert(at + bl <= ExecutionParameters::kMaxOriginRunBytes
                && "addOriginId: survivor run exceeds kMaxOriginRunBytes");
            if (bp != runBuf + at) {
                std::memcpy(runBuf + at, bp, static_cast<std::size_t>(bl));
            }
            lens[j] = bl;
            at += bl;
        }
        map.inner().assignRun(key, runBuf, lens, rl);
    }

    /// @brief Cold twin of `addOriginEncoded` — encode a string history line and
    ///        append it to the cold exprOriginMap.
    ///
    /// @details
    /// Minting wrapper (single-threaded write sites only, I-83) = the cold
    /// `addOriginId` over `mintOriginKey` / `encodeOrigin`. The paired mail writes
    /// at the same sites keep using the string `addOrigin` — mail origin maps stay
    /// string per the batch boundary.
    ///
    /// @param map        The cold origin map.
    /// @param oi         The owning LB's origin interner (mint side).
    /// @param ev         `(expression, validity)` key.
    /// @param origin     `(tag, antecedents)` string record.
    /// @param maxOrigins Cap, exactly as at the string sites.
    inline void addOriginEncoded(TypedColdBlobMap<int64_t, IdOrigin>& map,
        ValueInterner& oi, const ExpressionWithValidity& ev,
        const std::pair<std::string, std::vector<ExpressionWithValidity>>& origin,
        int maxOrigins)
    {
        // Key-first (user byte-identity waiver): mint the origin KEY, resolve
        // the tag via the span twin (interns nothing), then mint the EWV
        // antecedents positionally — no IdOrigin materialized.
        const int64_t key = mintOriginKey(oi, ev.original, ev.validityName);
        const OriginTag tag = originTagFromString(StrSpan(origin.first));
        int64_t d[ExecutionParameters::kMaxOriginDeps];
        const int32_t n = mintOriginDepsFromEWVInto(origin.second, oi, d,
            ExecutionParameters::kMaxOriginDeps);
        addOriginId(map, key, static_cast<uint8_t>(tag), d, n, maxOrigins);
    }

    /// @brief Span overload of the cold `addOriginEncoded` — keys the deposit
    ///        from `StrSpan`s over the stable input, with no transient
    ///        `ExpressionWithValidity` materialized.
    ///
    /// @details Byte-identical deposit to the `ev` overload: the same
    /// `mintOriginKey` over the same bytes. Only the KEY side is spanned; the
    /// `origin` history record stays string (its EWV antecedents are the
    /// deferred mail/EWV boundary). The absorb-door deposits whose `ev` was
    /// built only to be interned-then-dropped use this.
    ///
    /// @param map          The cold origin map.
    /// @param oi           The owning LB's origin interner (mint side).
    /// @param original     Span over the expression bytes.
    /// @param validityName Span over the validity (scope) bytes.
    /// @param origin       `(tag, antecedents)` string record.
    /// @param maxOrigins   Cap, exactly as at the string sites.
    inline void addOriginEncoded(TypedColdBlobMap<int64_t, IdOrigin>& map,
        ValueInterner& oi, const StrSpan& original, const StrSpan& validityName,
        const std::pair<std::string, std::vector<ExpressionWithValidity>>& origin,
        int maxOrigins)
    {
        // Key-first (user byte-identity waiver): mint the origin KEY, resolve
        // the tag via the span twin (interns nothing), then mint the EWV
        // antecedents positionally — no IdOrigin materialized.
        const int64_t key = mintOriginKey(oi, original, validityName);
        const OriginTag tag = originTagFromString(StrSpan(origin.first));
        int64_t d[ExecutionParameters::kMaxOriginDeps];
        const int32_t n = mintOriginDepsFromEWVInto(origin.second, oi, d,
            ExecutionParameters::kMaxOriginDeps);
        addOriginId(map, key, static_cast<uint8_t>(tag), d, n, maxOrigins);
    }

    /// @brief Span-antecedent twin of the cold `addOriginEncoded` — key AND
    ///        record built from spans + an `OriginTag`, no transient
    ///        `ExpressionWithValidity` / `OriginLine` materialized.
    ///
    /// @details
    /// Same-content deposit as the span-key/`OriginLine`-record overload above:
    /// the same cold `addOriginId` sink over the same `mintOriginKey` (span
    /// twin) key and the same dependency mints in the same positional order —
    /// but through the POD dep-run door, so no transient `IdOrigin` is
    /// materialized. Mint order is KEY-FIRST (mint the origin key, then the
    /// dependencies via `mintOriginDepsInto`): this pins the previously
    /// compiler-chosen argument-evaluation order to a deterministic key-first
    /// sequence under the user's byte-identity waiver (`files/` artifacts are
    /// id-value-independent; only the gitignored `.deload/` id VALUES may
    /// reassign). The D-49 cap-full preference fires identically. This is the L3
    /// flip of the deferred origin-antecedent boundary the L1/L2 span-key door
    /// left string.
    ///
    /// @param map          The cold origin map.
    /// @param oi           The owning LB's origin interner (mint side).
    /// @param original     Span over the key expression bytes.
    /// @param validityName Span over the key validity (scope) bytes.
    /// @param tag          The history-line tag (the enumerator the site names).
    /// @param deps         Pointer to `depN` antecedent spans, positional order.
    /// @param depN         The antecedent count.
    /// @param maxOrigins   Cap, exactly as at the string sites.
    /// @invariant The spans must alias buffers OTHER than @p oi's own byte store
    ///            ([I-3](../../docs/agentic_swdd/30_invariants.md#i-3)).
    /// @see addOriginEncoded(TypedColdBlobMap<int64_t, IdOrigin>&, ValueInterner&, const StrSpan&, const StrSpan&, const OriginLine&, int)
    ///      — the span-key/`OriginLine`-record overload this reproduces
    ///      byte-for-byte; encodeOriginSpans.
    inline void addOriginEncoded(TypedColdBlobMap<int64_t, IdOrigin>& map,
        ValueInterner& oi, const StrSpan& original, const StrSpan& validityName,
        OriginTag tag, const OriginDep* deps, int depN, int maxOrigins)
    {
        // Key-first (user byte-identity waiver): mint the origin KEY, then the
        // dependency ids positionally into a stack buffer — no IdOrigin built.
        const int64_t key = mintOriginKey(oi, original, validityName);
        int64_t d[ExecutionParameters::kMaxOriginDeps];
        const int32_t n = mintOriginDepsInto(deps, depN, oi, d,
            ExecutionParameters::kMaxOriginDeps);
        addOriginId(map, key, static_cast<uint8_t>(tag), d, n, maxOrigins);
    }

    /// @brief Cold twin of `decodeOriginMapSorted` — decode the cold origin map
    ///        into key-sorted string rows.
    ///
    /// @details
    /// Walks every minted key (id `1..count()` — insertion order, deterministic),
    /// decodes each key + its run, then sorts the rows by the decoded
    /// `(expression, validity)` pair — byte-identical to the `IdOriginMap` twin
    /// (both produce the same sorted snapshot). Per-row history lines keep their
    /// stored (insertion) order. Owned strings throughout (I-3).
    ///
    /// @param map The cold origin map.
    /// @param oi  The owning LB's origin interner (read-only).
    /// @return Rows of `((expression, validity), string history lines)`,
    ///         key-sorted.
    inline std::vector<std::pair<std::pair<std::string, std::string>,
        std::vector<std::pair<std::string, std::vector<ExpressionWithValidity>>>>>
    decodeOriginMapSorted(const TypedColdBlobMap<int64_t, IdOrigin>& map,
        const ValueInterner& oi)
    {
        std::vector<std::pair<std::pair<std::string, std::string>,
            std::vector<std::pair<std::string, std::vector<ExpressionWithValidity>>>>> rows;
        rows.reserve(static_cast<size_t>(map.count()));
        for (int32_t id = 1; id <= map.count(); ++id) {
            const int64_t key = map.decodeKey(id);
            const std::vector<IdOrigin> recs = map.recordsAt(id);
            std::vector<std::pair<std::string, std::vector<ExpressionWithValidity>>> lines;
            lines.reserve(recs.size());
            for (const IdOrigin& origin : recs) {
                lines.push_back(decodeOrigin(origin, oi));
            }
            StrSpan eSpan, vSpan;
            decodeOriginKeyView(key, oi, eSpan, vSpan);
            rows.emplace_back(std::make_pair(eSpan.toStdString(), vSpan.toStdString()),
                std::move(lines));
        }
        std::sort(rows.begin(), rows.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
        return rows;
    }

    // ====================================================================
    //  Part C — record VALUE codecs for the HashMemory migration. Each
    //  serializes ONE record to fixed-width little-endian bytes (the
    //  put/get idiom of serializeEquivalenceClass): the byte stream is a
    //  pure function of content (std::set fields emit in their canonical
    //  iteration order). The decoded-order set comparators the prover's
    //  record-set values carry stay a per-call argument at the typed
    //  surface — never inside the codec. Pipeline-unused until the owning
    //  container migrates onto the typed cold facade (commits 1-6).
    // ====================================================================

    /// @brief Record codec for `LocalMemoryValue` — the `encodedMap` blob value.
    ///
    /// @details
    /// Layout: `valueId`, `originalImplicationId`, `justification` (int32 each),
    /// `validityId` (int16), `isMarker`, `productOfDisintegration` (uint8 each),
    /// then `levels` (int32 count + int32 each, ascending set order), `keyIds` and
    /// `remainingArgIds` (int32 count + int32 each, positional order).
    ///
    /// @see `LocalMemoryValue`, `Codec`.
    template <>
    struct Codec<LocalMemoryValue> {
        /// @brief Serialize a value to its canonical blob.
        ///
        /// @param v The value.
        /// @return The byte blob.
        static std::vector<char> serialize(const LocalMemoryValue& v) {
            std::vector<char> out;
            const auto put = [&out](const auto& x) {
                const char* p = reinterpret_cast<const char*>(&x);
                out.insert(out.end(), p, p + sizeof(x));
            };
            put(v.valueId);
            put(v.originalImplicationId);
            put(static_cast<int32_t>(v.justification));
            put(v.validityId);
            put(static_cast<uint8_t>(v.isMarker ? 1 : 0));
            put(static_cast<uint8_t>(v.productOfDisintegration ? 1 : 0));
            put(static_cast<int32_t>(v.levels.size()));
            for (const int lev : v.levels) put(static_cast<int32_t>(lev));
            put(static_cast<int32_t>(v.keyIds.size()));
            for (const int32_t id : v.keyIds) put(id);
            put(static_cast<int32_t>(v.remainingArgIds.size()));
            for (const int32_t id : v.remainingArgIds) put(id);
            return out;
        }

        /// @brief Deserialize a value from its blob.
        ///
        /// @param data Blob bytes.
        /// @param n    Blob length.
        /// @return The decoded value.
        static LocalMemoryValue deserialize(const char* data, int32_t n) {
            LocalMemoryValue v;
            const char* cur = data;
            const char* const end = data + n;
            const auto get = [&cur, end](auto& x) {
                assert(cur + sizeof(x) <= end
                    && "Codec<LocalMemoryValue>: blob shorter than the layout");
                std::memcpy(&x, cur, sizeof(x));
                cur += sizeof(x);
            };
            get(v.valueId);
            get(v.originalImplicationId);
            int32_t just = 0; get(just);
            v.justification = static_cast<RuleJustification>(just);
            get(v.validityId);
            uint8_t marker = 0; get(marker); v.isMarker = (marker != 0);
            uint8_t pod = 0; get(pod); v.productOfDisintegration = (pod != 0);
            int32_t lc = 0; get(lc);
            for (int32_t i = 0; i < lc; ++i) {
                int32_t lev = 0; get(lev); v.levels.insert(static_cast<int>(lev));
            }
            int32_t kc = 0; get(kc);
            v.keyIds.resize(static_cast<std::size_t>(kc));
            for (int32_t i = 0; i < kc; ++i)
                get(v.keyIds[static_cast<std::size_t>(i)]);
            int32_t rc = 0; get(rc);
            v.remainingArgIds.resize(static_cast<std::size_t>(rc));
            for (int32_t i = 0; i < rc; ++i)
                get(v.remainingArgIds[static_cast<std::size_t>(i)]);
            assert(cur == end
                && "Codec<LocalMemoryValue>: blob longer than the layout");
            return v;
        }
    };

    /// @brief Peek a serialized `LocalMemoryValue` blob's `validityId` — a
    ///        fixed-offset read, no record decode.
    ///
    /// @details
    /// `Codec<LocalMemoryValue>` writes a fixed 16-byte prefix — `valueId`
    /// (int32 @0), `originalImplicationId` (int32 @4), `justification`
    /// (int32 @8), `validityId` (int16 @12), `isMarker` (uint8 @14),
    /// `productOfDisintegration` (uint8 @15) — before the variable
    /// `levels` / `keyIds` / `remainingArgIds` sections, so the scope id is
    /// always the two bytes at offset 12. `Memory::wipeSubtree`'s
    /// `encodedMap` sweep reads it through this peek to test closed-scope
    /// membership without decoding (and re-encoding) any record.
    ///
    /// Layout-coupled to `Codec<LocalMemoryValue>`: the two are edited
    /// together — any prefix change there must update the offset (and the
    /// length assert) here, exactly as `OwnerSetBlob` mirrors
    /// `Codec<OwnerSet>`.
    ///
    /// @param p   The blob's bytes (a `peekRecordBytes` span).
    /// @param len The blob's byte length; asserted `>= 16` (the fixed
    ///            prefix — every serialized LMV carries it).
    /// @return The record's `validityId`.
    /// @invariant `I-139` — the wipe consumes this peek
    ///            through its bitmap membership lambda.
    /// @see `Codec<LocalMemoryValue>`, `wipeEncodedMapForClosed`,
    ///      `OwnerSetBlob`.
    inline int16_t lmvBlobValidityId(const char* p, int32_t len) {
        assert(len >= 16
            && "lmvBlobValidityId: blob shorter than the fixed LMV prefix");
        int16_t v;
        std::memcpy(&v, p + 12, sizeof(v));
        return v;
    }

    /// @brief Read the element count of a serialized `Int16SetKey` — the
    ///        `int16` at offset 0.
    ///
    /// @details
    /// `Codec<Int16SetKey>` writes `int16 count ++ count×int16 ids`, the ids
    /// ascending (the `std::set<int16_t>` iteration order). This zero-decode
    /// peek reads the count straight off `raMap.keyAt(id)`'s raw bytes, the
    /// firing-check candidate scan's lever (no per-candidate heap
    /// `Int16SetKey` decode). Layout-coupled to `Codec<Int16SetKey>`: the two
    /// are edited together, exactly as `lmvBlobValidityId` mirrors
    /// `Codec<LocalMemoryValue>`.
    ///
    /// @param k The raw key bytes (a `keyAt(id)` span).
    /// @return The number of ids in the set.
    /// @invariant Asserts `k.len >= 2 && k.len % 2 == 0` — a malformed key is a
    ///            bug surfaced at its origin (Rule 19).
    /// @see `Codec<Int16SetKey>`, `int16SetKeyIdAt`, `int16SetKeyLexCompare`.
    inline int16_t int16SetKeyCount(StrSpan k) {
        assert(k.len >= 2 && (k.len % 2) == 0
            && "int16SetKeyCount: malformed Int16SetKey blob");
        int16_t v;
        std::memcpy(&v, k.ptr, sizeof(v));
        return v;
    }

    /// @brief Read id @p i of a serialized `Int16SetKey` — the `int16` at
    ///        offset `2 + 2*i`.
    ///
    /// @details
    /// The ids follow the count in ascending positional order (the
    /// `std::set<int16_t>` iteration order); this reads the i-th one
    /// zero-decode. Layout-coupled to `Codec<Int16SetKey>`.
    ///
    /// @param k The raw key bytes (a `keyAt(id)` span).
    /// @param i The id index.
    /// @return The `int16` id at position @p i.
    /// @invariant Asserts `0 <= i < int16SetKeyCount(k)` (Rule 19).
    /// @see `int16SetKeyCount`, `int16SetKeyLexCompare`.
    inline int16_t int16SetKeyIdAt(StrSpan k, int32_t i) {
        assert(i >= 0 && i < int16SetKeyCount(k)
            && "int16SetKeyIdAt: index out of range");
        int16_t v;
        std::memcpy(&v, k.ptr + 2 + 2 * i, sizeof(v));
        return v;
    }

    /// @brief Lexicographic compare of two serialized `Int16SetKey`s — the
    ///        byte-for-byte twin of `std::set<int16_t>::operator<`.
    ///
    /// @details
    /// Compares element-by-element as SIGNED `int16_t` (matching the
    /// `std::set<int16_t>` default `std::less<int16_t>`), shorter-is-prefix-is-
    /// less. This is NOT the count-prefixed `Int16SetKey` byte order — a raw
    /// byte-lex compare would order on the `int16 count` field FIRST, a
    /// DIFFERENT order — so the firing-check candidate enumeration order (R1,
    /// deload-observable through the firing records) is preserved only by this
    /// signed-element lex, never by a byte compare.
    ///
    /// @param a First key's raw bytes.
    /// @param b Second key's raw bytes.
    /// @return `-1` when @p a `<` @p b, `+1` when `>`, `0` when equal.
    /// @see `int16SetKeyCount`, `int16SetKeyIdAt`, `Codec<Int16SetKey>`.
    inline int int16SetKeyLexCompare(StrSpan a, StrSpan b) {
        const int16_t na = int16SetKeyCount(a);
        const int16_t nb = int16SetKeyCount(b);
        const int16_t m = na < nb ? na : nb;
        for (int16_t k = 0; k < m; ++k) {
            const int16_t ia = int16SetKeyIdAt(a, k);
            const int16_t ib = int16SetKeyIdAt(b, k);
            if (ia != ib) return ia < ib ? -1 : 1;
        }
        return na < nb ? -1 : (na > nb ? 1 : 0);
    }

    /// @brief Zero-decode reader over one serialized `LocalMemoryValue` blob
    ///        — extends the `lmvBlobValidityId` fixed-prefix peek to the
    ///        variable sections, no record materialized.
    ///
    /// @details
    /// `Codec<LocalMemoryValue>` writes a fixed 16-byte prefix — `valueId`
    /// (int32 @0), `originalImplicationId` (int32 @4), `justification`
    /// (int32 @8), `validityId` (int16 @12), `isMarker` (uint8 @14),
    /// `productOfDisintegration` (uint8 @15) — then `levelCount` (int32) +
    /// levels (int32 each), `keyIdCount` (int32) + keyIds (int32 each,
    /// positional), `remainingCount` (int32) + remainingArgIds (int32 each).
    /// Every read goes through `std::memcpy` (unaligned pool bytes).
    ///
    /// `keyIdsBytes()` / `keyIdsByteLen()` is the load-bearing lever:
    /// `keyIds` serialize as a contiguous fixed-width int32 run with no
    /// padding, so RUN-BYTE equality == `std::vector<int32_t>` equality
    /// (the length is part of the comparison). `eradicateImplicationFromLB`
    /// keys its erased-chain set AND its originals-orphan scan on it.
    ///
    /// Layout-coupled to `Codec<LocalMemoryValue>`: the two are edited
    /// together — any layout change there must update the offsets (and the
    /// framing assert) here, exactly as `lmvBlobValidityId` and
    /// `OwnerSetBlob` mirror their codecs.
    ///
    /// @invariant Construction asserts the exact frame
    ///            `len == 28 + 4*(levelCount + keyIdCount + remainingCount)`
    ///            — the view twin of the codec's `cur == end` asserts
    ///            (Rule 19).
    /// @see `Codec<LocalMemoryValue>`, `lmvBlobValidityId`,
    ///      `wipeEncodedMapForClosed`, `eradicateImplicationFromLB`.
    struct LmvBlobView {
        const char* p;
        int32_t len;

        /// @brief Construct over one blob's bytes, asserting the exact frame.
        ///
        /// @details
        /// Framing walked once: the fixed prefix, then the level section,
        /// then the keyIds section, then the remainingArgIds section; the
        /// total length must equal the layout exactly.
        ///
        /// @param p_   The blob's bytes (a `peekRecordBytes` span or a
        ///             verbatim gen-arena copy).
        /// @param len_ The blob's byte length.
        LmvBlobView(const char* p_, int32_t len_) : p(p_), len(len_) {
            assert(len >= 20
                && "LmvBlobView: blob shorter than the fixed LMV prefix");
            const int32_t lc = levelCount();
            assert(lc >= 0 && len >= 24 + 4 * lc
                && "LmvBlobView: blob shorter than the levels section");
            const int32_t kc = keyIdCount();
            assert(kc >= 0 && len >= 28 + 4 * (lc + kc)
                && "LmvBlobView: blob shorter than the keyIds section");
            int32_t rc;
            std::memcpy(&rc, p + 24 + 4 * (lc + kc), sizeof(rc));
            assert(rc >= 0 && len == 28 + 4 * (lc + kc + rc)
                && "LmvBlobView: blob length does not match the layout");
        }

        /// @brief Read one int32 field at a byte offset (unaligned-safe).
        ///
        /// @param off The byte offset into the blob.
        /// @return The int32 at `off`.
        int32_t readI32(int32_t off) const {
            assert(off >= 0 && off + 4 <= len
                && "LmvBlobView: read past the blob");
            int32_t v;
            std::memcpy(&v, p + off, sizeof(v));
            return v;
        }

        /// @brief The record's `originalImplicationId` (rule-interner id).
        /// @return The int32 at offset 4.
        int32_t originalImplicationId() const { return readI32(4); }

        /// @brief The record's `validityId` (== `lmvBlobValidityId`).
        /// @return The int16 at offset 12.
        int16_t validityId() const {
            int16_t v;
            std::memcpy(&v, p + 12, sizeof(v));
            return v;
        }

        /// @brief The level count.
        /// @return The int32 at offset 16.
        int32_t levelCount() const { return readI32(16); }

        /// @brief The keyIds count.
        /// @return The int32 directly past the levels section.
        int32_t keyIdCount() const { return readI32(20 + 4 * levelCount()); }

        /// @brief The keyIds run's raw bytes (a contiguous int32 run).
        /// @return Pointer to `keyIdsByteLen()` bytes.
        const char* keyIdsBytes() const { return p + 24 + 4 * levelCount(); }

        /// @brief The keyIds run's byte length.
        /// @return `4 * keyIdCount()`.
        int32_t keyIdsByteLen() const { return 4 * keyIdCount(); }

        /// @brief KeyId `i` (positional order).
        ///
        /// @param i The index; asserted `0 <= i < keyIdCount()`.
        /// @return The int32 at `keyIdsBytes() + 4*i`.
        int32_t keyIdAt(int32_t i) const {
            assert(i >= 0 && i < keyIdCount()
                && "LmvBlobView::keyIdAt: index out of range");
            return readI32(24 + 4 * levelCount() + 4 * i);
        }

        /// @brief The record's `valueId` (rule-interner id, the rule head).
        /// @return The int32 at offset 0.
        int32_t valueId() const { return readI32(0); }

        /// @brief The record's `justification` tag.
        /// @return The `RuleJustification` at offset 8.
        RuleJustification justification() const {
            return static_cast<RuleJustification>(readI32(8));
        }

        /// @brief The record's `isMarker` flag.
        /// @return The uint8 at offset 14, as a bool.
        bool isMarker() const {
            uint8_t v;
            std::memcpy(&v, p + 14, 1);
            return v != 0;
        }

        /// @brief The record's `productOfDisintegration` flag.
        /// @return The uint8 at offset 15, as a bool.
        bool productOfDisintegration() const {
            uint8_t v;
            std::memcpy(&v, p + 15, 1);
            return v != 0;
        }

        /// @brief Level `i` (ascending set order — the `std::set<int>`
        ///        iteration order the codec serialized).
        /// @param i The index; asserted `0 <= i < levelCount()`.
        /// @return The int32 at offset `20 + 4*i`.
        int32_t levelAt(int32_t i) const {
            assert(i >= 0 && i < levelCount()
                && "LmvBlobView::levelAt: out of range");
            return readI32(20 + 4 * i);
        }

        /// @brief The remainingArgIds count.
        /// @return The int32 directly past the keyIds section.
        int32_t remainingCount() const {
            return readI32(24 + 4 * levelCount() + 4 * keyIdCount());
        }

        /// @brief RemainingArgId `i` (positional order).
        /// @param i The index; asserted `0 <= i < remainingCount()`.
        /// @return The int32 at offset
        ///         `28 + 4*levelCount() + 4*keyIdCount() + 4*i`.
        int32_t remainingArgIdAt(int32_t i) const {
            assert(i >= 0 && i < remainingCount()
                && "LmvBlobView::remainingArgIdAt: out of range");
            return readI32(28 + 4 * levelCount() + 4 * keyIdCount() + 4 * i);
        }
    };

    /// @brief Closed-scope sweep of a `HashMemory` `encodedMap` — the
    ///        verbatim two-pass blob splice behind `Memory::wipeSubtree`'s
    ///        step 10a.
    ///
    /// @details
    /// Pass 1 walks every key id in ascending order, byte-peeks each record
    /// (`peekRecordBytes` — zero decode) and keeps the blobs whose
    /// `lmvBlobValidityId` is NOT in the closed bitmap, copying the kept
    /// blob bytes AND the surviving keys' bytes VERBATIM onto `gArena`'s
    /// byte-bump tier. The copy-before-reset is load-bearing: the peeked
    /// spans alias the live cold pool, and `resetToFresh` frees those
    /// pages. Pass 2 resets the map and re-installs each surviving key in
    /// ascending old-id order through the raw engine doors — an `M == 0`
    /// `assignRun` (opens an empty run) followed by one `appendBlobToRun`
    /// per kept blob in original j order.
    ///
    /// Byte contract (vs the former decode-erase-reencode rebuild):
    /// survivor key order = ascending old id = the old survivors-vector
    /// order; per-key blob order = old j order minus dropped; blob bytes
    /// verbatim == the old re-serialize because `Codec<LocalMemoryValue>`
    /// round-trips canonical blobs losslessly; the `M == 0` open + per-blob
    /// append is documented byte-identical to the one-shot `assignRun`; the
    /// scope verdict equals the old decoded-name predicate
    /// (`I-139`).
    ///
    /// All scratch rides `gArena` under a `cursor()` / `popTo` frame (the
    /// metadata `PagedVector`s are destructor-freed page tier — I-124).
    ///
    /// @param closedBits    The wipe's closed-id bitmap (bit `id` set for
    ///                      closed validity ids).
    /// @param nameHighWater The NameMap id ceiling backing `closedBits`;
    ///                      the membership lambda ASSERTS every stored vid
    ///                      inside `[1, nameHighWater]` — a decode-heritage
    ///                      read where out-of-range means corruption, never
    ///                      legal input (the retired per-record `decode`
    ///                      range-asserted the same way).
    /// @param m             The `encodedMap` to sweep (any of the three
    ///                      `HashMemory` instances').
    /// @param gArena        The per-slot gen-scratch arena for the
    ///                      pass-1 copies and metadata columns.
    /// @invariant `I-139`; [I-103](../../docs/agentic_swdd/30_invariants.md#i-103)
    ///            — the rebuilt map's deload bytes stay a pure function of
    ///            surviving content.
    /// @see `lmvBlobValidityId`, `Memory::wipeSubtree`,
    ///      `HashMap::assignRun`, `HashMap::appendBlobToRun`.
    inline void wipeEncodedMapForClosed(
        TypedColdBlobMap<NormKey, LocalMemoryValue>& m,
        const uint64_t* closedBits, int32_t nameHighWater,
        ScratchArena& gArena) {
        struct KeyRec {
            ArenaOffset keyOff; int32_t keyLen;
            int32_t firstBlob; int32_t blobCount;
        };
        struct BlobRec { ArenaOffset off; int32_t len; };
        const auto closed = [&](int32_t v) -> bool {
            // Decode-heritage read: a stored LMV validityId is a minted
            // NameMap id — out-of-range is impossible by contract and must
            // abort HERE, at the bitmap edge (the retired decode() of the
            // vid range-asserted the same corruption; a tolerant guard
            // would classify it as "not closed" silently).
            assert(v >= 1 && v <= nameHighWater
                && "wipeEncodedMapForClosed: stored validityId outside the "
                   "minted NameMap range");
            return ((closedBits[static_cast<uint16_t>(v) >> 6]
                     >> (static_cast<uint16_t>(v) & 63)) & 1ull) != 0;
        };

        const ArenaOffset mark = gArena.cursor();
        {
            DirtyState keysDirty = DirtyState::Clean;
            DirtyState blobsDirty = DirtyState::Clean;
            PagedVector<KeyRec> keyRecs(&gArena, &keysDirty);
            PagedVector<BlobRec> blobRecs(&gArena, &blobsDirty);

            // Pass 1 — collect survivors (bytes copied out BEFORE the
            // reset frees the pages the peeks alias).
            const int32_t n = m.count();
            for (int32_t id = 1; id <= n; ++id) {
                const int32_t firstBlob = blobRecs.size();
                const int32_t rl = m.runLen(id);
                for (int32_t j = 0; j < rl; ++j) {
                    int32_t bl = 0;
                    const char* bp = m.peekRecordBytes(id, j, bl, gArena);
                    if (closed(lmvBlobValidityId(bp, bl))) continue;
                    const ArenaOffset off = gArena.alloc(bl, 1);
                    std::memcpy(gArena.resolve(off), bp,
                                static_cast<std::size_t>(bl));
                    blobRecs.push_back(BlobRec{ off, bl });
                }
                const int32_t kept = blobRecs.size() - firstBlob;
                if (kept >= 1) {
                    const StrSpan ks = m.inner().decode(id);
                    const ArenaOffset koff = gArena.alloc(ks.len, 1);
                    std::memcpy(gArena.resolve(koff), ks.ptr,
                                static_cast<std::size_t>(ks.len));
                    keyRecs.push_back(KeyRec{ koff, ks.len,
                                              firstBlob, kept });
                }
            }

            // Pass 2 — rebuild through the raw doors.
            m.resetToFresh();
            for (int32_t ki = 0; ki < keyRecs.size(); ++ki) {
                const KeyRec kr = keyRecs[ki];
                const int32_t nid = m.inner().assignRun(
                    StrSpan(gArena.resolve(kr.keyOff), kr.keyLen),
                    nullptr, nullptr, 0);
                for (int32_t j = 0; j < kr.blobCount; ++j) {
                    const BlobRec br = blobRecs[kr.firstBlob + j];
                    m.inner().appendBlobToRun(nid, gArena.resolve(br.off),
                                              br.len);
                }
            }
        }
        gArena.popTo(mark);
    }

    /// @brief Implication-eradication sweep of a `HashMemory` `encodedMap` —
    ///        the verbatim two-pass blob splice behind
    ///        `eradicateImplicationFromLB`'s step 3, with a chain-collection
    ///        side channel.
    ///
    /// @details
    /// The `wipeEncodedMapForClosed` pattern with a different predicate:
    /// pass 1 walks every key id ascending, byte-peeks each record
    /// (`peekRecordBytes` — zero decode) and DROPS the blobs whose
    /// `originalImplicationId == implRuleId && validityId == implVid`
    /// (the rules derived from the implication during disintegration),
    /// copying every kept blob's bytes AND the surviving keys' bytes
    /// VERBATIM onto `gArena`'s byte-bump tier. Each DROPPED blob with a
    /// non-empty keyIds run mints its raw keyIds bytes into
    /// @p erasedChains (mint copies the bytes into the set's own pool —
    /// taken in pass 1, while the source map is still resident; the
    /// copy-before-reset doctrine). Pass 2 resets the map and re-installs
    /// each surviving key in ascending old-id order through the raw engine
    /// doors — an `M == 0` `assignRun` followed by one `appendBlobToRun`
    /// per kept blob in original j order.
    ///
    /// Byte contract (the `wipeEncodedMapForClosed` points plus the
    /// chain-set argument):
    /// 1. survivor KEY order = ascending old id = the old `survivors`
    ///    vector order → identical key facet bytes;
    /// 2. survivor BLOB order per key = old j order minus dropped = the
    ///    old `lmvs.erase` result;
    /// 3. blob bytes VERBATIM == the old decode-then-re-serialize (the LMV
    ///    codec round-trips canonical blobs losslessly);
    /// 4. `M == 0` `assignRun` + per-blob `appendBlobToRun` == the one-shot
    ///    `assignRun`;
    /// 5. chain-set content == the old `std::set<std::vector<int32_t>>`
    ///    content: byte-key equality == vector equality (the `LmvBlobView`
    ///    keyIds lever — fixed-width run, length in the key), and `mint`
    ///    dedups exactly as `std::set::insert`.
    ///
    /// @param m            The `encodedMap` to sweep (any of the three
    ///                     `HashMemory` instances').
    /// @param implRuleId   The implication's `ruleInterner` id (nonzero —
    ///                     the caller pre-guards, preserving the historical
    ///                     whole-map skip on a never-installed implication).
    /// @param implVid      The implication's scope `NameMap` id (nonzero).
    /// @param gArena       The per-slot gen-scratch arena for the pass-1
    ///                     copies and metadata columns.
    /// @param erasedChains The chain-collection side channel (raw keyIds
    ///                     run bytes; page tier on the same arena).
    /// @invariant [I-103](../../docs/agentic_swdd/30_invariants.md#i-103) —
    ///            the rebuilt map's deload bytes stay a pure function of
    ///            surviving content.
    /// @see `wipeEncodedMapForClosed` — the pattern sibling; `LmvBlobView`;
    ///      `eradicateImplicationFromLB` — the sole caller.
    inline void eradicateEncodedMapForImpl(
        TypedColdBlobMap<NormKey, LocalMemoryValue>& m,
        int32_t implRuleId, int16_t implVid,
        ScratchArena& gArena,
        ColdHashSet<BytesKeyStore>& erasedChains) {
        struct KeyRec {
            ArenaOffset keyOff; int32_t keyLen;
            int32_t firstBlob; int32_t blobCount;
        };
        struct BlobRec { ArenaOffset off; int32_t len; };

        const ArenaOffset mark = gArena.cursor();
        {
            DirtyState keysDirty = DirtyState::Clean;
            DirtyState blobsDirty = DirtyState::Clean;
            PagedVector<KeyRec> keyRecs(&gArena, &keysDirty);
            PagedVector<BlobRec> blobRecs(&gArena, &blobsDirty);

            // Pass 1 — collect survivors + dropped chains (bytes copied /
            // minted out BEFORE the reset frees the pages the peeks alias).
            const int32_t n = m.count();
            for (int32_t id = 1; id <= n; ++id) {
                const int32_t firstBlob = blobRecs.size();
                const int32_t rl = m.runLen(id);
                for (int32_t j = 0; j < rl; ++j) {
                    int32_t bl = 0;
                    const char* bp = m.peekRecordBytes(id, j, bl, gArena);
                    const LmvBlobView view(bp, bl);
                    if (view.originalImplicationId() == implRuleId
                        && view.validityId() == implVid) {
                        if (view.keyIdCount() > 0) {
                            erasedChains.mint(StrSpan(view.keyIdsBytes(),
                                                      view.keyIdsByteLen()));
                        }
                        continue;
                    }
                    const ArenaOffset off = gArena.alloc(bl, 1);
                    std::memcpy(gArena.resolve(off), bp,
                                static_cast<std::size_t>(bl));
                    blobRecs.push_back(BlobRec{ off, bl });
                }
                const int32_t kept = blobRecs.size() - firstBlob;
                if (kept >= 1) {
                    const StrSpan ks = m.inner().decode(id);
                    const ArenaOffset koff = gArena.alloc(ks.len, 1);
                    std::memcpy(gArena.resolve(koff), ks.ptr,
                                static_cast<std::size_t>(ks.len));
                    keyRecs.push_back(KeyRec{ koff, ks.len,
                                              firstBlob, kept });
                }
            }

            // Pass 2 — rebuild through the raw doors.
            m.resetToFresh();
            for (int32_t ki = 0; ki < keyRecs.size(); ++ki) {
                const KeyRec kr = keyRecs[ki];
                const int32_t nid = m.inner().assignRun(
                    StrSpan(gArena.resolve(kr.keyOff), kr.keyLen),
                    nullptr, nullptr, 0);
                for (int32_t j = 0; j < kr.blobCount; ++j) {
                    const BlobRec br = blobRecs[kr.firstBlob + j];
                    m.inner().appendBlobToRun(nid, gArena.resolve(br.off),
                                              br.len);
                }
            }
        }
        gArena.popTo(mark);
    }

    /// @brief Record codec for `OwnerSet` — the value of the four owner-set maps.
    ///
    /// @details
    /// Layout: `hasLooseOwner` (uint8), `partitionIds` (int32 count + int32 each,
    /// ascending set order), `uSignatures` (int32 count, then per signature an
    /// int32 pair-count + each `(int16, int16)` pair — both the outer set and the
    /// inner vectors emit in their canonical order).
    ///
    /// @see `OwnerSet`, `Codec`.
    template <>
    struct Codec<OwnerSet> {
        /// @brief Serialize an owner set to its canonical blob.
        ///
        /// @param v The owner set.
        /// @return The byte blob.
        static std::vector<char> serialize(const OwnerSet& v) {
            std::vector<char> out;
            const auto put = [&out](const auto& x) {
                const char* p = reinterpret_cast<const char*>(&x);
                out.insert(out.end(), p, p + sizeof(x));
            };
            put(static_cast<uint8_t>(v.hasLooseOwner ? 1 : 0));
            put(static_cast<int32_t>(v.partitionIds.size()));
            for (const int32_t id : v.partitionIds) put(id);
            put(static_cast<int32_t>(v.uSignatures.size()));
            for (const std::vector<std::pair<int16_t, int16_t>>& sig
                     : v.uSignatures) {
                put(static_cast<int32_t>(sig.size()));
                for (const std::pair<int16_t, int16_t>& pr : sig) {
                    put(pr.first);
                    put(pr.second);
                }
            }
            return out;
        }

        /// @brief Deserialize an owner set from its blob.
        ///
        /// @param data Blob bytes.
        /// @param n    Blob length.
        /// @return The decoded owner set.
        static OwnerSet deserialize(const char* data, int32_t n) {
            OwnerSet v;
            const char* cur = data;
            const char* const end = data + n;
            const auto get = [&cur, end](auto& x) {
                assert(cur + sizeof(x) <= end
                    && "Codec<OwnerSet>: blob shorter than the layout");
                std::memcpy(&x, cur, sizeof(x));
                cur += sizeof(x);
            };
            uint8_t loose = 0; get(loose); v.hasLooseOwner = (loose != 0);
            int32_t pc = 0; get(pc);
            for (int32_t i = 0; i < pc; ++i) {
                int32_t id = 0; get(id); v.partitionIds.insert(id);
            }
            int32_t sc = 0; get(sc);
            for (int32_t i = 0; i < sc; ++i) {
                int32_t pairCount = 0; get(pairCount);
                std::vector<std::pair<int16_t, int16_t>> sig;
                sig.reserve(static_cast<std::size_t>(pairCount));
                for (int32_t j = 0; j < pairCount; ++j) {
                    int16_t a = 0, b = 0; get(a); get(b);
                    sig.emplace_back(a, b);
                }
                v.uSignatures.insert(std::move(sig));
            }
            assert(cur == end
                && "Codec<OwnerSet>: blob longer than the layout");
            return v;
        }
    };

    /// @brief Closed-scope sweep of one owner-set map (`normalizedEncoded*`)
    ///        — the zero-decode blob-byte filter behind `Memory::wipeSubtree`'s
    ///        step 10b.
    ///
    /// @details
    /// Pass 1 walks every key id ascending and reads the key's SINGLE blob
    /// (the owner maps are run-length-1 — `mergeOwnerRecord` installs one
    /// whole-value record per key; asserted). Through `OwnerSetBlob` it
    /// counts the partition ids whose LOW half (the interned scope vid,
    /// `makePartitionId`'s D-105 low 16 bits, cast exactly as the former
    /// decode loop cast it) sits in the closed bitmap:
    /// - all survive  → the WHOLE blob is byte-bump-copied verbatim;
    /// - none survive → the key is dropped; when `droppedKeys` is non-null
    ///   the key's bytes are minted into it (mint copies the bytes into the
    ///   set's own pool — taken in pass 1, while the source map is still
    ///   resident);
    /// - some survive → the blob is rewritten into a fresh
    ///   `bl - 4 * dropped` buffer: byte 0 (`hasLooseOwner`) verbatim, the
    ///   int32 survivor count, the surviving int32 ids in their original
    ///   (ascending `std::set`) order, then the whole `uSignatures` tail
    ///   memcpy'd verbatim from `uSigOffset()` — the wipe filters ONLY
    ///   `partitionIds`; `hasLooseOwner` + `uSignatures` are deliberately
    ///   left stale (sound, weaker-only — I-79).
    /// Pass 2 resets the map and re-installs each surviving key in
    /// ascending old-id order via the raw single-blob `assignRun` (M == 1).
    ///
    /// Byte contract: key order = ascending old id (the old survivors
    /// vector); a filtered blob equals `Codec<OwnerSet>::serialize` of the
    /// old decode-erase result (erasing from a `std::set<int32_t>` keeps
    /// the ascending survivor order, and re-serializing the untouched
    /// loose-flag / signature sets reproduces their bytes — a pure function
    /// of content); all-survive == whole-blob verbatim.
    ///
    /// @param m             One of the four owner-set maps.
    /// @param closedBits    The wipe's closed-id bitmap.
    /// @param nameHighWater The bitmap's id ceiling; the membership lambda
    ///                      ASSERTS every composite low half inside
    ///                      `[1, nameHighWater]` — a decode-heritage read
    ///                      where out-of-range means corruption, never legal
    ///                      input (the retired per-owner `decode`
    ///                      range-asserted the same way).
    /// @param gArena        The per-slot gen-scratch arena (byte-bump copies
    ///                      + page-tier metadata columns, `cursor`/`popTo`
    ///                      framed).
    /// @param droppedKeys   Non-null ONLY for `normalizedEncodedKeys` (the
    ///                      map whose dropped keys feed the 10c
    ///                      `remainingArgsNormalizedEncodedMap` prune);
    ///                      null for the three subkey maps.
    /// @invariant `I-139`; [I-49](../../docs/agentic_swdd/30_invariants.md#i-49)
    ///            — a key is dropped exactly when its owner set empties.
    /// @see `OwnerSetBlob`, `Codec<OwnerSet>`, `wipeRemainingArgsForClosed`,
    ///      `Memory::wipeSubtree`.
    inline void wipeOwnerSetMapForClosed(
        TypedColdBlobMap<NormKey, OwnerSet>& m,
        const uint64_t* closedBits, int32_t nameHighWater,
        ScratchArena& gArena,
        ColdHashSet<BytesKeyStore>* droppedKeys) {
        struct KeyRec { ArenaOffset keyOff; int32_t keyLen;
                        ArenaOffset blobOff; int32_t blobLen; };
        const auto closed = [&](int32_t v) -> bool {
            // Decode-heritage read: the composite's low half is a minted
            // NameMap scope vid (makePartitionId packs interned ids) —
            // out-of-range is impossible by contract and must abort HERE
            // (the retired decode() of the vid range-asserted the same
            // corruption; a tolerant guard would classify it as "not
            // closed" silently).
            assert(v >= 1 && v <= nameHighWater
                && "wipeOwnerSetMapForClosed: partition scope vid outside "
                   "the minted NameMap range");
            return ((closedBits[static_cast<uint16_t>(v) >> 6]
                     >> (static_cast<uint16_t>(v) & 63)) & 1ull) != 0;
        };

        const ArenaOffset mark = gArena.cursor();
        {
            DirtyState keysDirty = DirtyState::Clean;
            PagedVector<KeyRec> keyRecs(&gArena, &keysDirty);

            // Pass 1 — filter each key's single blob at the byte level.
            const int32_t n = m.count();
            for (int32_t id = 1; id <= n; ++id) {
                assert(m.runLen(id) == 1
                    && "wipeOwnerSetMapForClosed: owner maps are run-length-1");
                int32_t bl = 0;
                const char* bp = m.peekRecordBytes(id, 0, bl, gArena);
                const OwnerSetBlob ob{ bp, bl };
                const int32_t pc = ob.partitionCount();
                int32_t dropped = 0;
                for (int32_t i = 0; i < pc; ++i) {
                    if (closed(static_cast<int16_t>(
                            ob.partitionId(i) & 0xFFFF)))
                        ++dropped;
                }
                if (dropped == pc) {
                    // No partition id survives (an already-empty set counts
                    // — the former `partitionIds.empty()` gate) -> the key
                    // is dropped.
                    if (droppedKeys != nullptr)
                        droppedKeys->mint(m.inner().decode(id));
                    continue;
                }
                ArenaOffset blobOff; int32_t blobLen;
                if (dropped == 0) {
                    // all survive -> whole blob verbatim.
                    blobLen = bl;
                    blobOff = gArena.alloc(bl, 1);
                    std::memcpy(gArena.resolve(blobOff), bp,
                                static_cast<std::size_t>(bl));
                } else {
                    // partial -> rewrite: byte0 + count' + ascending
                    // survivors + verbatim uSig tail.
                    blobLen = bl - 4 * dropped;
                    blobOff = gArena.alloc(blobLen, 1);
                    char* w = gArena.resolve(blobOff);
                    w[0] = bp[0];
                    const int32_t keptCount = pc - dropped;
                    std::memcpy(w + 1, &keptCount, sizeof(int32_t));
                    int32_t at = 5;
                    for (int32_t i = 0; i < pc; ++i) {
                        const int32_t pid = ob.partitionId(i);
                        if (closed(static_cast<int16_t>(pid & 0xFFFF)))
                            continue;
                        std::memcpy(w + at, &pid, sizeof(int32_t));
                        at += 4;
                    }
                    const int32_t tailLen = bl - ob.uSigOffset();
                    std::memcpy(w + at, bp + ob.uSigOffset(),
                                static_cast<std::size_t>(tailLen));
                    assert(at + tailLen == blobLen
                        && "wipeOwnerSetMapForClosed: rewrite length drift");
                }
                const StrSpan ks = m.inner().decode(id);
                const ArenaOffset koff = gArena.alloc(ks.len, 1);
                std::memcpy(gArena.resolve(koff), ks.ptr,
                            static_cast<std::size_t>(ks.len));
                keyRecs.push_back(KeyRec{ koff, ks.len, blobOff, blobLen });
            }

            // Pass 2 — rebuild via the single-blob raw door.
            m.resetToFresh();
            for (int32_t ki = 0; ki < keyRecs.size(); ++ki) {
                const KeyRec kr = keyRecs[ki];
                m.inner().assignRun(
                    StrSpan(gArena.resolve(kr.keyOff), kr.keyLen),
                    gArena.resolve(kr.blobOff), &kr.blobLen, 1);
            }
        }
        gArena.popTo(mark);
    }

    /// @brief Prune `remainingArgsNormalizedEncodedMap` of the normalized
    ///        keys the 10b owner-set wipe dropped — `Memory::wipeSubtree`'s
    ///        step 10c, as a byte-peek membership filter.
    ///
    /// @details
    /// A record of this map IS an encoded `NormKey` — and
    /// `Codec<NormKey>::serialize` is documented byte-identical to the key
    /// `encode`, so a record's raw bytes are directly usable as the
    /// dropped-key lookup key: keep record `j` iff
    /// `droppedKeys.lookup(StrSpan(bytes, len)) == 0`. Zero decode
    /// anywhere. Pass 1 copies surviving keys + records verbatim onto the
    /// byte-bump tier; pass 2 resets and re-installs each surviving key in
    /// ascending old-id order via an `M == 0` `assignRun` + per-record
    /// `appendBlobToRun` (runs here are many-record). Keys whose run
    /// empties are dropped.
    ///
    /// The `droppedKeys.count() == 0` early return is the defined twin of
    /// the former `if (!droppedKeys.empty())` gate — a contracted bivalent
    /// branch (nothing to prune), not a failure fallback.
    ///
    /// Byte contract: the four points of the 10a splice — ascending
    /// survivor key order, original record order minus dropped, verbatim
    /// record bytes (trivially round-trip-stable — fixed int16 framing),
    /// and the append door's byte-identity to a one-shot `assignRun`.
    ///
    /// @param ra          The remaining-args secondary index.
    /// @param rev         The derived reverse membership index for @p ra;
    ///                    rebuilt wholesale from the pruned forward map after the
    ///                    survivor reinstall (the survivor key ids are reassigned
    ///                    by the resetToFresh+reinstall, so an incremental edit
    ///                    cannot track them — I-154).
    /// @param droppedKeys The 10b-collected dropped `NormKey` byte set.
    /// @param gArena      The per-slot gen-scratch arena (`cursor`/`popTo`
    ///                    framed).
    /// @invariant `I-139`; the serialize==encode
    ///            identity of `Codec<NormKey>` is the membership lever.
    /// @see `wipeOwnerSetMapForClosed`, `Codec<NormKey>`,
    ///      `Memory::wipeSubtree`, `ReverseArgsIndex`.
    inline void wipeRemainingArgsForClosed(
        TypedColdBlobMap<Int16SetKey, NormKey>& ra,
        ReverseArgsIndex& rev,
        const ColdHashSet<BytesKeyStore>& droppedKeys,
        ScratchArena& gArena) {
        if (droppedKeys.count() == 0) return;
        struct KeyRec { ArenaOffset keyOff; int32_t keyLen;
                        int32_t firstBlob; int32_t blobCount; };
        struct BlobRec { ArenaOffset off; int32_t len; };

        const ArenaOffset mark = gArena.cursor();
        {
            DirtyState keysDirty = DirtyState::Clean;
            DirtyState blobsDirty = DirtyState::Clean;
            PagedVector<KeyRec> keyRecs(&gArena, &keysDirty);
            PagedVector<BlobRec> blobRecs(&gArena, &blobsDirty);

            // Pass 1 — keep records whose bytes are NOT dropped keys.
            const int32_t n = ra.count();
            for (int32_t id = 1; id <= n; ++id) {
                const int32_t firstBlob = blobRecs.size();
                const int32_t rl = ra.runLen(id);
                for (int32_t j = 0; j < rl; ++j) {
                    int32_t bl = 0;
                    const char* bp = ra.peekRecordBytes(id, j, bl, gArena);
                    if (droppedKeys.lookup(StrSpan(bp, bl)) != 0) continue;
                    const ArenaOffset off = gArena.alloc(bl, 1);
                    std::memcpy(gArena.resolve(off), bp,
                                static_cast<std::size_t>(bl));
                    blobRecs.push_back(BlobRec{ off, bl });
                }
                const int32_t kept = blobRecs.size() - firstBlob;
                if (kept >= 1) {
                    const StrSpan ks = ra.inner().decode(id);
                    const ArenaOffset koff = gArena.alloc(ks.len, 1);
                    std::memcpy(gArena.resolve(koff), ks.ptr,
                                static_cast<std::size_t>(ks.len));
                    keyRecs.push_back(KeyRec{ koff, ks.len,
                                              firstBlob, kept });
                }
            }

            // Pass 2 — rebuild through the raw doors.
            ra.resetToFresh();
            for (int32_t ki = 0; ki < keyRecs.size(); ++ki) {
                const KeyRec kr = keyRecs[ki];
                const int32_t nid = ra.inner().assignRun(
                    StrSpan(gArena.resolve(kr.keyOff), kr.keyLen),
                    nullptr, nullptr, 0);
                for (int32_t j = 0; j < kr.blobCount; ++j) {
                    const BlobRec br = blobRecs[kr.firstBlob + j];
                    ra.inner().appendBlobToRun(nid, gArena.resolve(br.off),
                                               br.len);
                }
            }
        }
        // Re-derive the reverse index from the pruned forward map: the survivor
        // key ids were reassigned by the resetToFresh + reinstall above, so an
        // incremental edit cannot track them (I-154). The
        // early return above skips this when nothing was dropped (ra unchanged).
        rev.rebuildReverseIndex(ra, gArena);
        gArena.popTo(mark);
    }

    /// @brief Record codec for `AdmissionMapValue` — an `admissionMap` set member.
    ///
    /// @details
    /// Layout: `standardMaxAdmissionDepth`, `standardMaxSecondaryNumber` (int32
    /// each), `flag` (uint8), then `key` and `remainingArgs` (int32 count + int32
    /// each, positional / decoded-lex order preserved verbatim).
    ///
    /// @see `AdmissionMapValue`, `Codec`.
    template <>
    struct Codec<AdmissionMapValue> {
        /// @brief Serialize a value to its canonical blob.
        ///
        /// @param v The value.
        /// @return The byte blob.
        static std::vector<char> serialize(const AdmissionMapValue& v) {
            std::vector<char> out;
            const auto put = [&out](const auto& x) {
                const char* p = reinterpret_cast<const char*>(&x);
                out.insert(out.end(), p, p + sizeof(x));
            };
            put(static_cast<int32_t>(v.standardMaxAdmissionDepth));
            put(static_cast<int32_t>(v.standardMaxSecondaryNumber));
            put(static_cast<uint8_t>(v.flag ? 1 : 0));
            put(static_cast<int32_t>(v.key.size()));
            for (const int32_t id : v.key) put(id);
            put(static_cast<int32_t>(v.remainingArgs.size()));
            for (const int32_t id : v.remainingArgs) put(id);
            return out;
        }

        /// @brief Deserialize a value from its blob.
        ///
        /// @param data Blob bytes.
        /// @param n    Blob length.
        /// @return The decoded value.
        static AdmissionMapValue deserialize(const char* data, int32_t n) {
            AdmissionMapValue v;
            const char* cur = data;
            const char* const end = data + n;
            const auto get = [&cur, end](auto& x) {
                assert(cur + sizeof(x) <= end
                    && "Codec<AdmissionMapValue>: blob shorter than the layout");
                std::memcpy(&x, cur, sizeof(x));
                cur += sizeof(x);
            };
            int32_t depth = 0; get(depth); v.standardMaxAdmissionDepth = depth;
            int32_t sec = 0; get(sec); v.standardMaxSecondaryNumber = sec;
            uint8_t flag = 0; get(flag); v.flag = (flag != 0);
            int32_t kc = 0; get(kc);
            v.key.resize(static_cast<std::size_t>(kc));
            for (int32_t i = 0; i < kc; ++i)
                get(v.key[static_cast<std::size_t>(i)]);
            int32_t rc = 0; get(rc);
            v.remainingArgs.resize(static_cast<std::size_t>(rc));
            for (int32_t i = 0; i < rc; ++i)
                get(v.remainingArgs[static_cast<std::size_t>(i)]);
            assert(cur == end
                && "Codec<AdmissionMapValue>: blob longer than the layout");
            return v;
        }
    };

    /// @brief Zero-decode reader over one serialized `AdmissionMapValue` blob —
    ///        fixed-offset field peeks plus the variable key / remainingArgs id
    ///        sections, no record materialized.
    ///
    /// @details
    /// `Codec<AdmissionMapValue>` writes `standardMaxAdmissionDepth` (int32 @0),
    /// `standardMaxSecondaryNumber` (int32 @4), `flag` (uint8 @8), then
    /// `keyCount` (int32 @9) + key ids (int32 each, positional order) +
    /// `remCount` (int32) + remainingArgs ids (int32 each, decoded-lex order).
    /// Everything past the uint8 is UNALIGNED — every read goes through
    /// `std::memcpy`. The D-172 RMW splice (`insertAdmissionBlobSorted`)
    /// compares run blobs through this view instead of decoding a heap
    /// `AdmissionMapValue` per record.
    ///
    /// Layout-coupled to `Codec<AdmissionMapValue>`: the two are edited
    /// together — any layout change there must update the offsets (and the
    /// framing assert) here, exactly as `RejectedValueBlobView` mirrors
    /// `Codec<RejectedMapValue>`.
    ///
    /// @invariant Construction asserts the exact frame
    ///            `len == 17 + 4*(keyCount + remCount)` — the view twin of the
    ///            codec's `cur == end` asserts (Rule 19: a malformed blob is a
    ///            bug surfaced at its origin, never tolerated).
    /// @see `Codec<AdmissionMapValue>`, `RejectedValueBlobView`,
    ///      `admissionBlobLess`, `insertAdmissionBlobSorted`.
    struct AdmissionValueBlobView {
        const char* p;
        int32_t len;

        /// @brief Construct over one blob's bytes, asserting the exact frame.
        ///
        /// @details
        /// Framing is walked once here: the fixed prefix must be readable,
        /// then the key section, then the remainingArgs section, and the
        /// total length must equal the layout exactly.
        ///
        /// @param p_   The blob's bytes (a `peekRecordBytes` span or a
        ///             verbatim gen-arena copy).
        /// @param len_ The blob's byte length.
        AdmissionValueBlobView(const char* p_, int32_t len_) : p(p_), len(len_) {
            assert(len >= 13
                && "AdmissionValueBlobView: blob shorter than the fixed prefix");
            const int32_t kc = keyCount();
            assert(kc >= 0 && len >= 17 + 4 * kc
                && "AdmissionValueBlobView: blob shorter than the key section");
            const int32_t rc = remCount();
            assert(rc >= 0 && len == 17 + 4 * (kc + rc)
                && "AdmissionValueBlobView: blob length does not match the layout");
        }

        /// @brief Read one int32 field at a byte offset (unaligned-safe).
        ///
        /// @param off The byte offset into the blob.
        /// @return The int32 at `off`.
        int32_t readI32(int32_t off) const {
            assert(off >= 0 && off + 4 <= len
                && "AdmissionValueBlobView: read past the blob");
            int32_t v;
            std::memcpy(&v, p + off, sizeof(v));
            return v;
        }

        /// @brief The record's `standardMaxAdmissionDepth`.
        /// @return The int32 at offset 0.
        int32_t depth() const { return readI32(0); }

        /// @brief The record's `standardMaxSecondaryNumber`.
        /// @return The int32 at offset 4.
        int32_t sec() const { return readI32(4); }

        /// @brief The record's `flag` byte (0 or 1).
        /// @return The uint8 at offset 8.
        uint8_t flagByte() const {
            uint8_t v;
            std::memcpy(&v, p + 8, sizeof(v));
            return v;
        }

        /// @brief The key element count.
        /// @return The int32 at offset 9.
        int32_t keyCount() const { return readI32(9); }

        /// @brief Key element `i`'s value-interner id (positional order).
        ///
        /// @param i The key index; asserted `0 <= i < keyCount()`.
        /// @return The int32 at offset `13 + 4*i`.
        int32_t keyId(int32_t i) const {
            assert(i >= 0 && i < keyCount()
                && "AdmissionValueBlobView::keyId: index out of range");
            return readI32(13 + 4 * i);
        }

        /// @brief The remainingArgs element count.
        /// @return The int32 directly past the key section.
        int32_t remCount() const { return readI32(13 + 4 * keyCount()); }

        /// @brief remainingArgs element `i`'s value-interner id (decoded-lex
        ///        storage order).
        ///
        /// @param i The index; asserted `0 <= i < remCount()`.
        /// @return The int32 at offset `17 + 4*keyCount() + 4*i`.
        int32_t remId(int32_t i) const {
            assert(i >= 0 && i < remCount()
                && "AdmissionValueBlobView::remId: index out of range");
            return readI32(17 + 4 * keyCount() + 4 * i);
        }
    };

    /// @brief Decoded-order less-than over two `AdmissionMapValue` blobs — the
    ///        zero-decode twin of `DecodedAdmissionValueLess`.
    ///
    /// @details
    /// Field order exactly as `DecodedAdmissionValueLess::operator()`: `key`
    /// (decoded-lex id-vector compare), `remainingArgs`, then
    /// `standardMaxAdmissionDepth`, `standardMaxSecondaryNumber`, `flag`. Each
    /// id-vector section runs the `valueIdVectorLess` semantics as a three-way
    /// walk — id-equality short-circuits per element (the interner is
    /// bijective, so equal ids ARE the equal-string case), a differing pair
    /// compares its `decodeView` spans, and equal prefixes tie-break on
    /// length. The pair-call shape of the set comparator (`less(a,b)` /
    /// `less(b,a)`) collapses to `cmp != 0 → cmp < 0`, which is the identical
    /// verdict for a strict weak order. Distinct ids decoding to equal bytes
    /// are impossible (the interner dedups) and assert.
    ///
    /// @param a  First blob view.
    /// @param b  Second blob view.
    /// @param vi The owning LB's value interner (read-only).
    /// @return `true` iff decoded @p a orders before decoded @p b.
    /// @invariant Comparator-equal (`!less(a,b) && !less(b,a)`) implies
    ///            blob-byte-equal: every compared field is a POD int or an
    ///            interner id and the interner is injective, so the set's
    ///            dedup is exactly byte-dedup on the serialized blob.
    /// @see `DecodedAdmissionValueLess` — the heap-set oracle;
    ///      `valueIdVectorLess`, `insertAdmissionBlobSorted`.
    inline bool admissionBlobLess(const AdmissionValueBlobView& a,
        const AdmissionValueBlobView& b, const ValueInterner& vi)
    {
        const auto runCmp = [&vi](const AdmissionValueBlobView& x,
                                  const AdmissionValueBlobView& y,
                                  bool keySection) -> int {
            const int32_t nx = keySection ? x.keyCount() : x.remCount();
            const int32_t ny = keySection ? y.keyCount() : y.remCount();
            const int32_t n = nx < ny ? nx : ny;
            for (int32_t i = 0; i < n; ++i) {
                const int32_t xi = keySection ? x.keyId(i) : x.remId(i);
                const int32_t yi = keySection ? y.keyId(i) : y.remId(i);
                if (xi == yi) continue;
                const int c = compareSpans(vi.decodeView(xi), vi.decodeView(yi));
                assert(c != 0
                    && "distinct value ids decode to equal bytes — interner "
                       "injectivity broken");
                return c;
            }
            if (nx != ny) return nx < ny ? -1 : 1;
            return 0;
        };
        const int kc = runCmp(a, b, true);
        if (kc != 0) return kc < 0;
        const int rc = runCmp(a, b, false);
        if (rc != 0) return rc < 0;
        if (a.depth() != b.depth()) return a.depth() < b.depth();
        if (a.sec() != b.sec()) return a.sec() < b.sec();
        return a.flagByte() < b.flagByte();
    }

    /// @brief Serialize one `AdmissionMapValue` onto a gen-scratch byte-bump
    ///        frame — the arena twin of `Codec<AdmissionMapValue>::serialize`.
    ///
    /// @details
    /// Writes the identical byte layout (`depth`, `sec`, `flag`, `keyCount` +
    /// key ids, `remCount` + remainingArgs ids) with the identical `put`
    /// sequence into one `alloc(len, 1)` run; the closed-form length is
    /// computed first and the exact-fill asserted (the two-pass exact-length
    /// doctrine). No heap `std::vector<char>` is materialized.
    ///
    /// @param gArena The per-slot gen-scratch arena the blob bytes land on;
    ///               the caller owns the surrounding mark/popTo frame.
    /// @param v      The value to serialize.
    /// @return Span over the freshly written blob (valid until the caller's
    ///         `popTo` past it).
    /// @see `Codec<AdmissionMapValue>` — the byte-layout owner;
    ///      `insertAdmissionValue` — the RMW wrapper feeding the splice.
    inline StrSpan serializeAdmissionValueToArena(ScratchArena& gArena,
        const AdmissionMapValue& v)
    {
        const int32_t kc = static_cast<int32_t>(v.key.size());
        const int32_t rc = static_cast<int32_t>(v.remainingArgs.size());
        const int32_t len = 17 + 4 * (kc + rc);
        char* out = reinterpret_cast<char*>(gArena.resolve(gArena.alloc(len, 1)));
        char* at = out;
        const auto put = [&at](const auto& x) {
            std::memcpy(at, &x, sizeof(x));
            at += sizeof(x);
        };
        put(static_cast<int32_t>(v.standardMaxAdmissionDepth));
        put(static_cast<int32_t>(v.standardMaxSecondaryNumber));
        put(static_cast<uint8_t>(v.flag ? 1 : 0));
        put(kc);
        for (const int32_t id : v.key) put(id);
        put(rc);
        for (const int32_t id : v.remainingArgs) put(id);
        assert(at == out + len
            && "serializeAdmissionValueToArena: fill diverged from the "
               "closed-form length");
        return StrSpan(out, len);
    }

    /// @brief Splice one serialized `AdmissionMapValue` blob into an admission
    ///        key's canonical run at blob level — the arena interior of the
    ///        D-172 RMW, no heap set and no per-record decode.
    ///
    /// @details
    /// The run is stored canonical (sorted under `DecodedAdmissionValueLess`,
    /// I-99), so the insert is a positional splice: copy every existing blob
    /// VERBATIM onto the gen arena (copy-before-`assignRun` — the run rewrite
    /// restructures the pool under peeked pointers), find the insert position
    /// with the blob comparator (first index where the new blob orders before
    /// the existing one), and write the whole run back through the raw
    /// `inner().assignRun` byte door (`PodKeyStore<int32_t>` — the KeyView is
    /// the int32 itself); a run whose concatenated bytes would straddle an arena
    /// block instead widens — one empty-run `assignRun` then per-blob
    /// `appendBlobToRun`, byte-identical run content (the blob pool spans pages,
    /// so the run is unbounded). A comparator-equal position is a DUPLICATE — by
    /// interner injectivity comparator-equal is byte-equal (asserted) — and
    /// mirrors the heap set's no-op insert; the run is written back even
    /// then (the retired set-RMW helper called `assignRun` unconditionally
    /// after the set no-op — same writes, same `DirtyState` escalation; do
    /// not "optimize" the duplicate path into a skip). Survivor blobs are
    /// byte-copied, never decoded/re-encoded. Mints `pk` on first touch.
    /// Single-threaded write side only (I-83).
    ///
    /// @param m       The cold admission blob map.
    /// @param pk      The packed (templateId, validityId) key.
    /// @param newBlob The new value's serialized blob bytes.
    /// @param newLen  The new blob's byte length.
    /// @param vi      The owning LB's value interner (comparator state,
    ///                read-only).
    /// @param gArena  The per-slot gen-scratch arena for the verbatim copies +
    ///                the concatenation buffer; reclaimed by this function's
    ///                own mark/popTo (LIFO-self-contained).
    /// @invariant The written run is byte-identical to the retired
    ///            decode-into-`std::set`-insert-reserialize cycle: survivors
    ///            verbatim, the new blob at its canonical position, duplicate
    ///            rewrites the identical run.
    /// @see `insertAdmissionValue` — the value-form wrapper;
    ///      `admissionBlobLess`, `Codec<AdmissionMapValue>`, `D-172`, `I-99`.
    inline void insertAdmissionBlobSorted(
        TypedColdBlobMap<int32_t, AdmissionMapValue>& m,
        int32_t pk, const char* newBlob, int32_t newLen,
        const ValueInterner& vi, ScratchArena& gArena)
    {
        const ArenaOffset mark = gArena.cursor();
        struct BlobRec { ArenaOffset off; int32_t len; };
        DirtyState recsDirty = DirtyState::Clean;
        PagedVector<BlobRec> recs(&gArena, &recsDirty);
        const int32_t id = m.lookup(pk);
        if (id != 0) {
            const int32_t rl = m.runLen(id);
            for (int32_t j = 0; j < rl; ++j) {
                int32_t bl = 0;
                const char* bp = m.peekRecordBytes(id, j, bl, gArena);
                const ArenaOffset coff = gArena.alloc(bl, 1);
                std::memcpy(gArena.resolve(coff), bp,
                            static_cast<std::size_t>(bl));
                recs.push_back(BlobRec{ coff, bl });
            }
        }
        const AdmissionValueBlobView nv(newBlob, newLen);
        const int32_t M = recs.size();
        int32_t pos = 0;
        bool dup = false;
        for (; pos < M; ++pos) {
            const BlobRec br = recs[pos];
            const AdmissionValueBlobView ev(
                reinterpret_cast<const char*>(gArena.resolve(br.off)), br.len);
            if (admissionBlobLess(ev, nv, vi)) continue;  // existing < new
            if (!admissionBlobLess(nv, ev, vi)) {
                // Comparator-equal — the heap set's no-op insert. Interner
                // injectivity makes this byte-equality; assert it.
                dup = (br.len == newLen)
                    && std::memcmp(gArena.resolve(br.off), newBlob,
                                   static_cast<std::size_t>(newLen)) == 0;
                assert(dup
                    && "comparator-equal admission blobs differ in bytes — "
                       "interner injectivity broken");
            }
            break;
        }
        const int32_t outCount = dup ? M : M + 1;
        int64_t total = 0;
        for (int32_t j = 0; j < M; ++j) total += recs[j].len;
        if (!dup) total += newLen;
        // A run whose concatenated bytes would straddle an arena block (an arena
        // alloc may not exceed one block) takes the widening door: open the run
        // empty, then append each blob at the run-end in sorted order —
        // existing[0..pos), newBlob at pos when !dup, existing[pos..M). The map's
        // blob pool straddles pages, so the run there is unbounded, and
        // appendBlobToRun is byte-identical to the one-block concat's assignRun.
        // gArena is untouched by these map writes, so the recs / newBlob copies
        // stay valid across the loop.
        if (total > ExecutionParameters::kMaxAdmissionRunBytes) {
            const int32_t wid = m.inner().assignRun(pk, nullptr, nullptr, 0);
            for (int32_t j = 0; j <= M; ++j) {
                if (!dup && j == pos)
                    m.inner().appendBlobToRun(wid, newBlob, newLen);
                if (j == M) break;
                const BlobRec br = recs[j];
                m.inner().appendBlobToRun(wid, gArena.resolve(br.off), br.len);
            }
            gArena.popTo(mark);
            return;
        }
        // Common case: the whole run fits one arena block — one concat + assignRun.
        const ArenaOffset concatOff =
            gArena.alloc(static_cast<int32_t>(total), 1);
        const ArenaOffset lensOff = gArena.alloc(
            outCount * static_cast<int32_t>(sizeof(int32_t)),
            static_cast<int32_t>(alignof(int32_t)));
        char* concat = reinterpret_cast<char*>(gArena.resolve(concatOff));
        int32_t* lens =
            reinterpret_cast<int32_t*>(gArena.resolve(lensOff));
        int32_t at = 0;
        int32_t oi = 0;
        for (int32_t j = 0; j <= M; ++j) {
            if (!dup && j == pos) {
                std::memcpy(concat + at, newBlob,
                            static_cast<std::size_t>(newLen));
                lens[oi++] = newLen;
                at += newLen;
            }
            if (j == M) break;
            const BlobRec br = recs[j];
            std::memcpy(concat + at, gArena.resolve(br.off),
                        static_cast<std::size_t>(br.len));
            lens[oi++] = br.len;
            at += br.len;
        }
        assert(oi == outCount && at == static_cast<int32_t>(total)
            && "insertAdmissionBlobSorted: concatenation diverged from the "
               "computed frame");
        m.inner().assignRun(pk, concat, lens, outCount);
        gArena.popTo(mark);
    }

    /// @brief Encode a parallel-staged string-form admission value STRAIGHT to
    ///        its serialized blob on a gen-scratch frame — the drain's
    ///        zero-heap fast path.
    ///
    /// @details
    /// Fuses `stagedToIdValue` + `Codec<AdmissionMapValue>::serialize` without
    /// the intermediate heap `AdmissionMapValue`: the closed-form length is
    /// written first (`17 + 4*(keyCount + remCount)`), then the ids are minted
    /// through `vi.encode(StrSpan)` in EXACTLY `stagedToIdValue`'s order —
    /// every key element in rule order, then every remaining arg in the
    /// presorted lex order — so the interner-touch sequence (and therefore
    /// every id and every deload byte) is unchanged. `stagedToIdValue` is
    /// retained as the unit-test oracle for this byte contract.
    ///
    /// @param sv     The staged value (sealed string views).
    /// @param vi     The owning LB's value interner (mint side —
    ///               single-threaded drain only).
    /// @param gArena The per-slot gen-scratch arena the blob lands on; the
    ///               caller owns the surrounding mark/popTo frame.
    /// @return Span over the freshly written blob.
    /// @see `stagedToIdValue` — the retained id-form oracle;
    ///      `drainAdmissionKeysAlgebra` — the consumer;
    ///      `Codec<AdmissionMapValue>` — the byte-layout owner.
    inline StrSpan stagedToArenaBlob(const StagedAdmissionValue& sv,
        ValueInterner& vi, ScratchArena& gArena)
    {
        const int32_t kc = sv.key.size();
        const int32_t rc = sv.remainingArgsSorted.size();
        const int32_t len = 17 + 4 * (kc + rc);
        char* out = reinterpret_cast<char*>(gArena.resolve(gArena.alloc(len, 1)));
        char* at = out;
        const auto put = [&at](const auto& x) {
            std::memcpy(at, &x, sizeof(x));
            at += sizeof(x);
        };
        put(static_cast<int32_t>(sv.standardMaxAdmissionDepth));
        put(static_cast<int32_t>(sv.standardMaxSecondaryNumber));
        put(static_cast<uint8_t>(sv.flag ? 1 : 0));
        put(kc);
        for (const SealedString& k : sv.key) {
            put(vi.encode(StrSpan(k)));
        }
        put(rc);
        for (const SealedString& a : sv.remainingArgsSorted) {
            put(vi.encode(StrSpan(a)));
        }
        assert(at == out + len
            && "stagedToArenaBlob: fill diverged from the closed-form length");
        return StrSpan(out, len);
    }

    /// @brief Decode an admission key's whole value run into its sorted set form
    ///        — the cold blob map read snapshot.
    ///
    /// @details
    /// `admissionMap` stores each `AdmissionMapValue` as a record blob in a run
    /// kept sorted by `DecodedAdmissionValueLess`. This rebuilds that run as the
    /// historical `std::set` form (same comparator, same order, same dedup) so the
    /// snapshot-and-iterate callers (`isAdmitted`, the equi-class hook, the dump)
    /// read it exactly as they read the former heap set. A key never minted yields
    /// an empty set — a defined query result, not a failure
    /// (D-172).
    ///
    /// @param m  The cold admission blob map.
    /// @param pk The packed (templateId, validityId) key.
    /// @param vi The owning LB's value interner (the comparator's state).
    /// @return The key's value set; empty when `pk` was never minted.
    inline AdmissionValueSet admissionRecordsAt(
        const TypedColdBlobMap<int32_t, AdmissionMapValue>& m,
        int32_t pk, const ValueInterner& vi)
    {
        AdmissionValueSet s(DecodedAdmissionValueLess{ &vi });
        const int32_t id = m.lookup(pk);
        if (id != 0) {
            const std::vector<AdmissionMapValue> run = m.recordsAt(id);
            for (const AdmissionMapValue& r : run) s.insert(r);
        }
        return s;
    }

    /// @brief A read-only snapshot of one admission key's value run — an array
    ///        of zero-decode @ref AdmissionValueBlobView over verbatim blob
    ///        copies held on a gen-scratch arena.
    ///
    /// @details
    /// The heap-free replacement for `admissionRecordsAt(m, pk, vi)` on the
    /// read-and-mutate paths: instead of an owned `std::set<AdmissionMapValue>`,
    /// `snapshotAdmissionRun` copies each record blob verbatim onto the caller's
    /// gen-scratch frame and returns non-owning views over the copies. Because
    /// the copies live on the arena (not in `m`'s blob pool), the snapshot
    /// survives any later mutation of `m` (`insertAdmissionIdsBlob` / `eraseBlobIf`)
    /// during the walk — the copy-before-loop discipline of the `isAdmitted`
    /// S9 C4 consumer (09b pitfall 6, I-99). The `views` pointer and every
    /// view's `p` bytes stay valid until the caller rewinds the arena past the
    /// snapshot's mark.
    ///
    /// @invariant The struct owns nothing — `views` and the copied bytes ride
    ///            @p gArena; the caller must not let a view outlive that arena's
    ///            rewind. `count == 0` for a never-minted key is a defined query
    ///            result, not a failure (Rule 19).
    ///
    /// @see admissionRecordsAt — the retained heap-set oracle (identical order,
    ///      the run is stored canonical under `DecodedAdmissionValueLess`);
    ///      AdmissionValueBlobView, snapshotAdmissionRun.
    struct AdmissionRunSnapshot {
        const AdmissionValueBlobView* views;
        int32_t count;
    };

    /// @brief Snapshot an admission key's value run as verbatim blob copies +
    ///        zero-decode views on a gen-scratch arena.
    ///
    /// @details
    /// Looks up @p pk, peeks each of its record blobs (`peekRecordBytes`), copies
    /// the bytes verbatim onto @p gArena, and constructs one
    /// @ref AdmissionValueBlobView over each copy into an arena-held array. The
    /// stored run is canonical (sorted+deduped under `DecodedAdmissionValueLess`
    /// by the RMW contract), so iterating the views reproduces the former
    /// `admissionRecordsAt` set order byte-for-byte. Mints nothing. A never-minted
    /// key yields `{ nullptr, 0 }`.
    ///
    /// @param m      The cold admission blob map (read-only here).
    /// @param pk     The packed `(templateId, validityId)` key.
    /// @param gArena The caller's gen-scratch arena (byte-bump tier); the blob
    ///               copies + view array ride it, reclaimed by the caller's
    ///               `popTo` / task-exit `releaseAll`.
    /// @return `{ views, count }` — non-owning views over the arena copies;
    ///         `count == 0` when @p pk was never minted.
    /// @invariant Copies survive any later mutation of @p m (blob pool
    ///            restructure), unlike a live `peekRecordBytes` pointer (I-99).
    /// @see admissionRecordsAt — the retained heap oracle; AdmissionRunSnapshot,
    ///      AdmissionValueBlobView, TypedColdBlobMap::peekRecordBytes.
    inline AdmissionRunSnapshot snapshotAdmissionRun(
        const TypedColdBlobMap<int32_t, AdmissionMapValue>& m,
        int32_t pk, ScratchArena& gArena)
    {
        const int32_t id = m.lookup(pk);
        const int32_t runN = id ? m.runLen(id) : 0;
        if (runN == 0) return AdmissionRunSnapshot{ nullptr, 0 };
        AdmissionValueBlobView* views = reinterpret_cast<AdmissionValueBlobView*>(
            gArena.resolve(gArena.alloc(
                runN * static_cast<int32_t>(sizeof(AdmissionValueBlobView)),
                static_cast<int32_t>(alignof(AdmissionValueBlobView)))));
        for (int32_t j = 0; j < runN; ++j) {
            int32_t bl = 0;
            const char* bp = m.peekRecordBytes(id, j, bl, gArena);
            const ArenaOffset off = gArena.alloc(bl, 1);
            std::memcpy(gArena.resolve(off), bp, static_cast<std::size_t>(bl));
            new (&views[j]) AdmissionValueBlobView(gArena.resolve(off), bl);
        }
        return AdmissionRunSnapshot{ views, runN };
    }

    /// @brief Read-modify-write one `AdmissionMapValue` into an admission key's
    ///        run, preserving sorted-unique (`std::set`) semantics.
    ///
    /// @details
    /// The cold-map equivalent of the retired `admissionValuesAt(m, pk,
    /// vi).insert(value)`. The interior is heap-free: the value is serialized
    /// onto the caller's gen-scratch frame
    /// (`serializeAdmissionValueToArena`) and spliced into the canonical run
    /// at blob level (`insertAdmissionBlobSorted` — survivors byte-copied
    /// verbatim, position by the `admissionBlobLess` comparator twin,
    /// duplicate ⟺ byte-equal by interner injectivity, run written back
    /// unconditionally exactly like the retired set no-op path). The run
    /// stays canonical, so the deload
    /// bytes and the sacred dump are byte-identical to the former heap set. Mints
    /// `pk` on first touch. Single-threaded write side only (I-83); the parallel
    /// burst stages strings and the single-threaded drain calls this
    /// (D-172).
    ///
    /// @param m      The cold admission blob map.
    /// @param pk     The packed (templateId, validityId) key.
    /// @param value  The value to insert (sorted + deduped into the run).
    /// @param vi     The owning LB's value interner (the comparator's state).
    /// @param gArena The caller's per-slot gen-scratch arena (a memory.hpp
    ///               free function cannot reach the worker slot itself);
    ///               reclaimed by this function's own mark/popTo.
    inline void insertAdmissionValue(
        TypedColdBlobMap<int32_t, AdmissionMapValue>& m,
        int32_t pk, const AdmissionMapValue& value, const ValueInterner& vi,
        ScratchArena& gArena)
    {
        const ArenaOffset mark = gArena.cursor();
        const StrSpan blob = serializeAdmissionValueToArena(gArena, value);
        insertAdmissionBlobSorted(m, pk, blob.ptr, blob.len, vi, gArena);
        gArena.popTo(mark);
    }

    /// @brief Record codec for `RejectedMapValue` — a `rejectedMap` set member.
    ///
    /// @details
    /// Layout: `renamedExpression`, `expression`, `iteration`, `concreteConstituent`
    /// (int32 each), then `siblings` (int32 count + int32 each, positional) and
    /// `levels` (int32 count + int32 each, ascending set order).
    ///
    /// @see `RejectedMapValue`, `Codec`.
    template <>
    struct Codec<RejectedMapValue> {
        /// @brief Serialize a value to its canonical blob.
        ///
        /// @param v The value.
        /// @return The byte blob.
        static std::vector<char> serialize(const RejectedMapValue& v) {
            std::vector<char> out;
            const auto put = [&out](const auto& x) {
                const char* p = reinterpret_cast<const char*>(&x);
                out.insert(out.end(), p, p + sizeof(x));
            };
            put(v.renamedExpression);
            put(v.expression);
            put(static_cast<int32_t>(v.iteration));
            put(v.concreteConstituent);
            put(static_cast<int32_t>(v.siblings.size()));
            for (const int32_t id : v.siblings) put(id);
            put(static_cast<int32_t>(v.levels.size()));
            for (const int lev : v.levels) put(static_cast<int32_t>(lev));
            return out;
        }

        /// @brief Deserialize a value from its blob.
        ///
        /// @param data Blob bytes.
        /// @param n    Blob length.
        /// @return The decoded value.
        static RejectedMapValue deserialize(const char* data, int32_t n) {
            RejectedMapValue v;
            const char* cur = data;
            const char* const end = data + n;
            const auto get = [&cur, end](auto& x) {
                assert(cur + sizeof(x) <= end
                    && "Codec<RejectedMapValue>: blob shorter than the layout");
                std::memcpy(&x, cur, sizeof(x));
                cur += sizeof(x);
            };
            get(v.renamedExpression);
            get(v.expression);
            int32_t iter = 0; get(iter); v.iteration = iter;
            get(v.concreteConstituent);
            int32_t sc = 0; get(sc);
            v.siblings.resize(static_cast<std::size_t>(sc));
            for (int32_t i = 0; i < sc; ++i)
                get(v.siblings[static_cast<std::size_t>(i)]);
            int32_t lc = 0; get(lc);
            for (int32_t i = 0; i < lc; ++i) {
                int32_t lev = 0; get(lev); v.levels.insert(static_cast<int>(lev));
            }
            assert(cur == end
                && "Codec<RejectedMapValue>: blob longer than the layout");
            return v;
        }
    };

    /// @brief Zero-decode reader over one serialized `RejectedMapValue` blob —
    ///        fixed-offset field peeks plus the variable siblings / levels
    ///        sections, no record materialized.
    ///
    /// @details
    /// `Codec<RejectedMapValue>` writes `renamedExpression` (int32 @0),
    /// `expression` (int32 @4), `iteration` (int32 @8), `concreteConstituent`
    /// (int32 @12), then `siblingCount` (int32 @16) + siblings (int32 each,
    /// positional order) + `levelCount` (int32) + levels (int32 each, ascending
    /// set order). Every read goes through `std::memcpy` — blob bytes are
    /// unaligned pool bytes. `revisitRejected2`'s copy-before-erase walk parses
    /// its gen-arena verbatim blob copies through this view instead of decoding
    /// a heap `RejectedMapValue` per record.
    ///
    /// Layout-coupled to `Codec<RejectedMapValue>`: the two are edited
    /// together — any layout change there must update the offsets (and the
    /// framing assert) here, exactly as `lmvBlobValidityId` mirrors
    /// `Codec<LocalMemoryValue>` and `OwnerSetBlob` mirrors `Codec<OwnerSet>`.
    ///
    /// @invariant Construction asserts the exact frame
    ///            `len == 24 + 4*(siblingCount + levelCount)` — the view twin
    ///            of the codec's `cur == end` asserts (Rule 19: a malformed
    ///            blob is a bug surfaced at its origin, never tolerated).
    /// @see `Codec<RejectedMapValue>`, `lmvBlobValidityId`, `OwnerSetBlob`,
    ///      `revisitRejected2`.
    struct RejectedValueBlobView {
        const char* p;
        int32_t len;

        /// @brief Construct over one blob's bytes, asserting the exact frame.
        ///
        /// @details
        /// Framing is walked once here: the fixed prefix must be readable,
        /// then the sibling section, then the level section, and the total
        /// length must equal the layout exactly.
        ///
        /// @param p_   The blob's bytes (a `peekRecordBytes` span or a
        ///             verbatim gen-arena copy).
        /// @param len_ The blob's byte length.
        RejectedValueBlobView(const char* p_, int32_t len_) : p(p_), len(len_) {
            assert(len >= 20
                && "RejectedValueBlobView: blob shorter than the fixed prefix");
            const int32_t sc = siblingCount();
            assert(sc >= 0 && len >= 24 + 4 * sc
                && "RejectedValueBlobView: blob shorter than the siblings section");
            const int32_t lc = levelCount();
            assert(lc >= 0 && len == 24 + 4 * (sc + lc)
                && "RejectedValueBlobView: blob length does not match the layout");
        }

        /// @brief Read one int32 field at a byte offset (unaligned-safe).
        ///
        /// @param off The byte offset into the blob.
        /// @return The int32 at `off`.
        int32_t readI32(int32_t off) const {
            assert(off >= 0 && off + 4 <= len
                && "RejectedValueBlobView: read past the blob");
            int32_t v;
            std::memcpy(&v, p + off, sizeof(v));
            return v;
        }

        /// @brief The record's `renamedExpression` value-interner id.
        /// @return The int32 at offset 0.
        int32_t renamedExpressionId() const { return readI32(0); }

        /// @brief The record's `expression` value-interner id.
        /// @return The int32 at offset 4.
        int32_t expressionId() const { return readI32(4); }

        /// @brief The record's rejection iteration.
        /// @return The int32 at offset 8.
        int32_t iteration() const { return readI32(8); }

        /// @brief The record's `concreteConstituent` value-interner id.
        /// @return The int32 at offset 12.
        int32_t concreteConstituentId() const { return readI32(12); }

        /// @brief The sibling count.
        /// @return The int32 at offset 16.
        int32_t siblingCount() const { return readI32(16); }

        /// @brief Sibling `i`'s value-interner id (positional order).
        ///
        /// @param i The sibling index; asserted `0 <= i < siblingCount()`.
        /// @return The int32 at offset `20 + 4*i`.
        int32_t siblingId(int32_t i) const {
            assert(i >= 0 && i < siblingCount()
                && "RejectedValueBlobView::siblingId: index out of range");
            return readI32(20 + 4 * i);
        }

        /// @brief The level count.
        /// @return The int32 directly past the siblings section.
        int32_t levelCount() const { return readI32(20 + 4 * siblingCount()); }

        /// @brief Copy the level run (ascending set order) into `dst`.
        ///
        /// @param dst Caller buffer of at least `levelCount()` ints.
        void copyLevels(int* dst) const {
            static_assert(sizeof(int) == sizeof(int32_t),
                "RejectedValueBlobView::copyLevels: int must be 32-bit");
            const int32_t lc = levelCount();
            std::memcpy(dst, p + 24 + 4 * siblingCount(),
                static_cast<std::size_t>(lc) * sizeof(int32_t));
        }

        /// @brief Level `i` of the run (ascending set order).
        ///
        /// @param i The level index; asserted `0 <= i < levelCount()`.
        /// @return The int32 at offset `24 + 4*siblingCount() + 4*i`.
        int32_t levelAt(int32_t i) const {
            assert(i >= 0 && i < levelCount()
                && "RejectedValueBlobView::levelAt: index out of range");
            return readI32(24 + 4 * siblingCount() + 4 * i);
        }
    };

    /// @brief Decoded-order less-than over two `RejectedMapValue` blobs — the
    ///        zero-decode twin of `DecodedRejectedValueLess`.
    ///
    /// @details
    /// Field order exactly as `DecodedRejectedValueLess::operator()`:
    /// `renamedExpression`, `expression` (each an id-inequality gate followed
    /// by the `valueIdLess` decoded compare), `iteration` (plain int),
    /// `concreteConstituent`, `siblings` (`valueIdVectorLess` semantics as a
    /// three-way walk), then `levels` — `std::set<int>::operator<` is a
    /// lexicographic compare over the elements, and the blob stores the
    /// levels ascending (`Codec<RejectedMapValue>` layout), so an int-wise
    /// lex walk with a length tie-break is its exact twin. Distinct ids
    /// decoding to equal bytes are impossible (the interner dedups) and
    /// assert.
    ///
    /// @param a  First blob view.
    /// @param b  Second blob view.
    /// @param vi The owning LB's value interner (read-only).
    /// @return `true` iff decoded @p a orders before decoded @p b.
    /// @invariant Comparator-equal implies blob-byte-equal (every compared
    ///            field is a POD int or an injective interner id).
    /// @see `DecodedRejectedValueLess` — the heap-set oracle; `valueIdLess`,
    ///      `valueIdVectorLess`, `insertRejectedBlobSorted`.
    inline bool rejectedBlobLess(const RejectedValueBlobView& a,
        const RejectedValueBlobView& b, const ValueInterner& vi)
    {
        const auto idLess3 = [&vi](int32_t x, int32_t y) -> int {
            const int c = compareSpans(vi.decodeView(x), vi.decodeView(y));
            assert(c != 0
                && "distinct value ids decode to equal bytes — interner "
                   "injectivity broken");
            return c;
        };
        if (a.renamedExpressionId() != b.renamedExpressionId()) {
            return idLess3(a.renamedExpressionId(), b.renamedExpressionId()) < 0;
        }
        if (a.expressionId() != b.expressionId()) {
            return idLess3(a.expressionId(), b.expressionId()) < 0;
        }
        if (a.iteration() != b.iteration()) return a.iteration() < b.iteration();
        if (a.concreteConstituentId() != b.concreteConstituentId()) {
            return idLess3(a.concreteConstituentId(),
                           b.concreteConstituentId()) < 0;
        }
        {
            // siblings: valueIdVectorLess pair-call collapsed to three-way.
            const int32_t na = a.siblingCount();
            const int32_t nb = b.siblingCount();
            const int32_t n = na < nb ? na : nb;
            for (int32_t i = 0; i < n; ++i) {
                const int32_t xi = a.siblingId(i);
                const int32_t yi = b.siblingId(i);
                if (xi == yi) continue;
                return idLess3(xi, yi) < 0;
            }
            if (na != nb) return na < nb;
        }
        // levels: std::set<int> operator< == int-wise lex over the ascending
        // runs, shorter-prefix-first.
        const int32_t la = a.levelCount();
        const int32_t lb = b.levelCount();
        const int32_t n = la < lb ? la : lb;
        for (int32_t i = 0; i < n; ++i) {
            const int32_t xa = a.levelAt(i);
            const int32_t yb = b.levelAt(i);
            if (xa != yb) return xa < yb;
        }
        return la < lb;
    }

    /// @brief Serialize one `RejectedMapValue` onto a gen-scratch byte-bump
    ///        frame — the arena twin of `Codec<RejectedMapValue>::serialize`.
    ///
    /// @details
    /// Writes the identical byte layout (`renamedExpression`, `expression`,
    /// `iteration`, `concreteConstituent`, `siblingCount` + siblings,
    /// `levelCount` + ascending levels) with the identical `put` sequence
    /// into one `alloc(len, 1)` run; the closed-form length is computed first
    /// and the exact-fill asserted. No heap `std::vector<char>` is
    /// materialized.
    ///
    /// @param gArena The per-slot gen-scratch arena the blob bytes land on;
    ///               the caller owns the surrounding mark/popTo frame.
    /// @param v      The value to serialize.
    /// @return Span over the freshly written blob (valid until the caller's
    ///         `popTo` past it).
    /// @see `Codec<RejectedMapValue>` — the byte-layout owner;
    ///      `insertRejectedValue` — the RMW wrapper feeding the splice.
    inline StrSpan serializeRejectedValueToArena(ScratchArena& gArena,
        const RejectedMapValue& v)
    {
        const int32_t sc = static_cast<int32_t>(v.siblings.size());
        const int32_t lc = static_cast<int32_t>(v.levels.size());
        const int32_t len = 24 + 4 * (sc + lc);
        char* out = reinterpret_cast<char*>(gArena.resolve(gArena.alloc(len, 1)));
        char* at = out;
        const auto put = [&at](const auto& x) {
            std::memcpy(at, &x, sizeof(x));
            at += sizeof(x);
        };
        put(v.renamedExpression);
        put(v.expression);
        put(static_cast<int32_t>(v.iteration));
        put(v.concreteConstituent);
        put(sc);
        for (const int32_t id : v.siblings) put(id);
        put(lc);
        for (const int lev : v.levels) put(static_cast<int32_t>(lev));
        assert(at == out + len
            && "serializeRejectedValueToArena: fill diverged from the "
               "closed-form length");
        return StrSpan(out, len);
    }

    /// @brief Splice one serialized `RejectedMapValue` blob into a rejected
    ///        key's canonical run at blob level — the arena interior of the
    ///        D-172 RMW, no heap set and no per-record decode.
    ///
    /// @details
    /// Mirror of `insertAdmissionBlobSorted` for `rejectedMap`: verbatim
    /// gen-arena copies of the existing run (copy-before-`assignRun`),
    /// positional insert under the `rejectedBlobLess` comparator twin,
    /// duplicate ⟺ byte-equal by interner injectivity (asserted), the run
    /// written back unconditionally through the raw `inner().assignRun` byte
    /// door (a run that would straddle an arena block widens to an empty-run
    /// `assignRun` + per-blob `appendBlobToRun`, byte-identical, unbounded) —
    /// same writes and same `DirtyState` escalation as the retired
    /// set-RMW path, duplicate included. Mints `pk` on first touch.
    /// Single-threaded write side only (I-83).
    ///
    /// @param m       The cold rejected blob map.
    /// @param pk      The packed (templateId, validityId) key.
    /// @param newBlob The new value's serialized blob bytes.
    /// @param newLen  The new blob's byte length.
    /// @param vi      The owning LB's value interner (comparator state,
    ///                read-only).
    /// @param gArena  The per-slot gen-scratch arena; reclaimed by this
    ///                function's own mark/popTo (LIFO-self-contained).
    /// @invariant The written run is byte-identical to the retired
    ///            decode-into-`std::set`-insert-reserialize cycle.
    /// @see `insertRejectedValue` — the value-form wrapper;
    ///      `rejectedBlobLess`, `Codec<RejectedMapValue>`, `D-172`, `I-99`.
    inline void insertRejectedBlobSorted(
        TypedColdBlobMap<int32_t, RejectedMapValue>& m,
        int32_t pk, const char* newBlob, int32_t newLen,
        const ValueInterner& vi, ScratchArena& gArena)
    {
        const ArenaOffset mark = gArena.cursor();
        struct BlobRec { ArenaOffset off; int32_t len; };
        DirtyState recsDirty = DirtyState::Clean;
        PagedVector<BlobRec> recs(&gArena, &recsDirty);
        const int32_t id = m.lookup(pk);
        if (id != 0) {
            const int32_t rl = m.runLen(id);
            for (int32_t j = 0; j < rl; ++j) {
                int32_t bl = 0;
                const char* bp = m.peekRecordBytes(id, j, bl, gArena);
                const ArenaOffset coff = gArena.alloc(bl, 1);
                std::memcpy(gArena.resolve(coff), bp,
                            static_cast<std::size_t>(bl));
                recs.push_back(BlobRec{ coff, bl });
            }
        }
        const RejectedValueBlobView nv(newBlob, newLen);
        const int32_t M = recs.size();
        int32_t pos = 0;
        bool dup = false;
        for (; pos < M; ++pos) {
            const BlobRec br = recs[pos];
            const RejectedValueBlobView ev(
                reinterpret_cast<const char*>(gArena.resolve(br.off)), br.len);
            if (rejectedBlobLess(ev, nv, vi)) continue;  // existing < new
            if (!rejectedBlobLess(nv, ev, vi)) {
                dup = (br.len == newLen)
                    && std::memcmp(gArena.resolve(br.off), newBlob,
                                   static_cast<std::size_t>(newLen)) == 0;
                assert(dup
                    && "comparator-equal rejected blobs differ in bytes — "
                       "interner injectivity broken");
            }
            break;
        }
        const int32_t outCount = dup ? M : M + 1;
        int64_t total = 0;
        for (int32_t j = 0; j < M; ++j) total += recs[j].len;
        if (!dup) total += newLen;
        // A run whose concatenated bytes would straddle an arena block (an arena
        // alloc may not exceed one block) takes the widening door: open the run
        // empty, then append each blob at the run-end in sorted order —
        // existing[0..pos), newBlob at pos when !dup, existing[pos..M). The map's
        // blob pool straddles pages, so the run there is unbounded, and
        // appendBlobToRun is byte-identical to the one-block concat's assignRun.
        // gArena is untouched by these map writes, so the recs / newBlob copies
        // stay valid across the loop.
        if (total > ExecutionParameters::kMaxAdmissionRunBytes) {
            const int32_t wid = m.inner().assignRun(pk, nullptr, nullptr, 0);
            for (int32_t j = 0; j <= M; ++j) {
                if (!dup && j == pos)
                    m.inner().appendBlobToRun(wid, newBlob, newLen);
                if (j == M) break;
                const BlobRec br = recs[j];
                m.inner().appendBlobToRun(wid, gArena.resolve(br.off), br.len);
            }
            gArena.popTo(mark);
            return;
        }
        // Common case: the whole run fits one arena block — one concat + assignRun.
        const ArenaOffset concatOff =
            gArena.alloc(static_cast<int32_t>(total), 1);
        const ArenaOffset lensOff = gArena.alloc(
            outCount * static_cast<int32_t>(sizeof(int32_t)),
            static_cast<int32_t>(alignof(int32_t)));
        char* concat = reinterpret_cast<char*>(gArena.resolve(concatOff));
        int32_t* lens =
            reinterpret_cast<int32_t*>(gArena.resolve(lensOff));
        int32_t at = 0;
        int32_t oi = 0;
        for (int32_t j = 0; j <= M; ++j) {
            if (!dup && j == pos) {
                std::memcpy(concat + at, newBlob,
                            static_cast<std::size_t>(newLen));
                lens[oi++] = newLen;
                at += newLen;
            }
            if (j == M) break;
            const BlobRec br = recs[j];
            std::memcpy(concat + at, gArena.resolve(br.off),
                        static_cast<std::size_t>(br.len));
            lens[oi++] = br.len;
            at += br.len;
        }
        assert(oi == outCount && at == static_cast<int32_t>(total)
            && "insertRejectedBlobSorted: concatenation diverged from the "
               "computed frame");
        m.inner().assignRun(pk, concat, lens, outCount);
        gArena.popTo(mark);
    }

    /// @brief Decode a rejected key's whole value run into its sorted set form
    ///        — the cold blob map read snapshot.
    ///
    /// @details
    /// Mirror of `admissionRecordsAt` for `rejectedMap`: rebuild the run as the
    /// historical `std::set` (`RejectedValueSet`, same `DecodedRejectedValueLess`
    /// order + dedup) so `revisitRejected2`'s snapshot loop and the equi-class
    /// hook read it exactly as the former heap set. A key never minted yields an
    /// empty set — a defined query result (D-172).
    ///
    /// @param m  The cold rejected blob map.
    /// @param pk The packed (templateId, validityId) key.
    /// @param vi The owning LB's value interner (the comparator's state).
    /// @return The key's value set; empty when `pk` was never minted.
    inline RejectedValueSet rejectedRecordsAt(
        const TypedColdBlobMap<int32_t, RejectedMapValue>& m,
        int32_t pk, const ValueInterner& vi)
    {
        RejectedValueSet s(DecodedRejectedValueLess{ &vi });
        const int32_t id = m.lookup(pk);
        if (id != 0) {
            const std::vector<RejectedMapValue> run = m.recordsAt(id);
            for (const RejectedMapValue& r : run) s.insert(r);
        }
        return s;
    }

    /// @brief A read-only snapshot of one rejected key's value run — an array of
    ///        zero-decode @ref RejectedValueBlobView over verbatim blob copies on
    ///        a gen-scratch arena.
    ///
    /// @details
    /// The literal rejected-side sibling of @ref AdmissionRunSnapshot: the
    /// heap-free replacement for `rejectedRecordsAt(m, pk, vi)` on the
    /// read-and-mutate paths (`applyEquivalenceClassToRejectedMap`,
    /// `revisitRejected2`). The copies live on the arena (not in @p m's blob
    /// pool), so the snapshot survives any later mutation of @p m
    /// (`insertRejectedValue` / `eraseBlobIf`) during the walk — the
    /// copy-before-loop discipline (09b pitfall 6, I-99). The stored run is
    /// canonical under `DecodedRejectedValueLess` (the `insertRejectedValue` RMW
    /// contract), so iterating the views reproduces the former `rejectedRecordsAt`
    /// set order byte-for-byte.
    ///
    /// @invariant The struct owns nothing — `views` and the copied bytes ride the
    ///            snapshotting arena; the caller must not let a view outlive that
    ///            arena's rewind. `count == 0` for a never-minted key is a defined
    ///            query result, not a failure (Rule 19).
    /// @see rejectedRecordsAt — the retained heap-set oracle (identical order, the
    ///      run is stored canonical under `DecodedRejectedValueLess`);
    ///      RejectedValueBlobView, snapshotRejectedRun, AdmissionRunSnapshot.
    struct RejectedRunSnapshot {
        const RejectedValueBlobView* views;
        int32_t count;
    };

    /// @brief Snapshot a rejected key's value run as verbatim blob copies +
    ///        zero-decode views on a gen-scratch arena.
    ///
    /// @details
    /// Byte-for-byte the @ref snapshotAdmissionRun body for `rejectedMap`: looks
    /// up @p pk, peeks each record blob (`peekRecordBytes`), copies the bytes
    /// verbatim onto @p gArena, and constructs one @ref RejectedValueBlobView over
    /// each copy into an arena-held array. The stored run is canonical
    /// (`DecodedRejectedValueLess` sorted+deduped by the RMW contract), so
    /// iterating the views reproduces the former `rejectedRecordsAt` set order
    /// byte-for-byte. Mints nothing. A never-minted key yields `{ nullptr, 0 }`.
    ///
    /// @param m      The cold rejected blob map (read-only here).
    /// @param pk     The packed `(templateId, validityId)` key.
    /// @param gArena The caller's gen-scratch arena (byte-bump tier); the blob
    ///               copies + view array ride it, reclaimed by the caller's
    ///               `popTo` / task-exit `releaseAll`.
    /// @return `{ views, count }` — non-owning views over the arena copies;
    ///         `count == 0` when @p pk was never minted.
    /// @invariant Copies survive any later mutation of @p m (blob pool
    ///            restructure), unlike a live `peekRecordBytes` pointer (I-99).
    /// @see rejectedRecordsAt — the retained heap oracle; RejectedRunSnapshot,
    ///      RejectedValueBlobView, snapshotAdmissionRun.
    inline RejectedRunSnapshot snapshotRejectedRun(
        const TypedColdBlobMap<int32_t, RejectedMapValue>& m,
        int32_t pk, ScratchArena& gArena)
    {
        const int32_t id = m.lookup(pk);
        const int32_t runN = id ? m.runLen(id) : 0;
        if (runN == 0) return RejectedRunSnapshot{ nullptr, 0 };
        RejectedValueBlobView* views = reinterpret_cast<RejectedValueBlobView*>(
            gArena.resolve(gArena.alloc(
                runN * static_cast<int32_t>(sizeof(RejectedValueBlobView)),
                static_cast<int32_t>(alignof(RejectedValueBlobView)))));
        for (int32_t j = 0; j < runN; ++j) {
            int32_t bl = 0;
            const char* bp = m.peekRecordBytes(id, j, bl, gArena);
            const ArenaOffset off = gArena.alloc(bl, 1);
            std::memcpy(gArena.resolve(off), bp, static_cast<std::size_t>(bl));
            new (&views[j]) RejectedValueBlobView(gArena.resolve(off), bl);
        }
        return RejectedRunSnapshot{ views, runN };
    }

    /// @brief Read-modify-write one `RejectedMapValue` into a rejected key's run,
    ///        preserving sorted-unique (`std::set`) semantics.
    ///
    /// @details
    /// Mirror of `insertAdmissionValue` for `rejectedMap`. The interior is
    /// heap-free: the value is serialized onto the caller's gen-scratch frame
    /// (`serializeRejectedValueToArena`) and spliced into the canonical run at
    /// blob level (`insertRejectedBlobSorted` — survivors byte-copied
    /// verbatim, position by the `rejectedBlobLess` comparator twin, duplicate
    /// ⟺ byte-equal by interner injectivity, run written back unconditionally
    /// exactly like the retired set no-op path). The run stays canonical, so
    /// the deload bytes and the sacred
    /// dump are byte-identical to the former heap set. Mints `pk` on first touch.
    /// Single-threaded write side only (I-83). This and `revisitRejected2`'s
    /// consume-erase are the only `rejectedMap` writers — the equi-class hook
    /// never writes it directly (I-37) (D-172).
    ///
    /// @param m      The cold rejected blob map.
    /// @param pk     The packed (templateId, validityId) key.
    /// @param value  The value to insert (sorted + deduped into the run).
    /// @param vi     The owning LB's value interner (the comparator's state).
    /// @param gArena The caller's per-slot gen-scratch arena (a memory.hpp
    ///               free function cannot reach the worker slot itself);
    ///               reclaimed by this function's own mark/popTo.
    inline void insertRejectedValue(
        TypedColdBlobMap<int32_t, RejectedMapValue>& m,
        int32_t pk, const RejectedMapValue& value, const ValueInterner& vi,
        ScratchArena& gArena)
    {
        const ArenaOffset mark = gArena.cursor();
        const StrSpan blob = serializeRejectedValueToArena(gArena, value);
        insertRejectedBlobSorted(m, pk, blob.ptr, blob.len, vi, gArena);
        gArena.popTo(mark);
    }

    /// @brief Deposit one admission value from PRE-MINTED id runs — the
    ///        value-object-free door onto the D-172 blob splice.
    ///
    /// @details
    /// The id-run twin of `insertAdmissionValue`: instead of taking a heap
    /// `AdmissionMapValue` (two `std::vector<int32_t>` spines), it serializes
    /// the `Codec<AdmissionMapValue>` byte layout DIRECTLY from two
    /// caller-owned id runs (`17 + 4*(keyCount+remCount)` closed-form length,
    /// the identical `put` sequence — depth, sec, flag, keyCount + key ids,
    /// remCount + remainingArgs ids), then splices via
    /// `insertAdmissionBlobSorted`. No minting occurs inside the door — the ids
    /// arrive already interned, so the CALLER owns the mint order (frozen by
    /// its pinning test). Byte-identical blob bytes ⇒ identical
    /// `admissionBlobLess` splice position, dedup verdict, run bytes, and
    /// deload bytes to the value-form door.
    ///
    /// @param m        The cold admission blob map.
    /// @param pk       The packed (templateId, validityId) key.
    /// @param depth    `standardMaxAdmissionDepth`.
    /// @param sec      `standardMaxSecondaryNumber`.
    /// @param flag     The value flag.
    /// @param keyIds   The key value ids (positional order); null iff
    ///                 `keyCount == 0`.
    /// @param keyCount The key id count (`>= 0`).
    /// @param remIds   The remainingArgs value ids (decoded-lex sorted order);
    ///                 null iff `remCount == 0`.
    /// @param remCount The remainingArgs id count (`>= 0`).
    /// @param vi       The owning LB's value interner (comparator state,
    ///                 read-only — no mint here).
    /// @param gArena   The per-slot gen-scratch arena for the blob + the splice
    ///                 copies; reclaimed by this function's own mark/popTo.
    /// @invariant No mint inside the door — the ids come in; the caller's mint
    ///            order is the byte-transparent contract.
    /// @see `insertAdmissionValue` — the value-form sibling + test oracle;
    ///      `serializeAdmissionValueToArena`, `insertAdmissionBlobSorted`.
    inline void insertAdmissionIdsBlob(
        TypedColdBlobMap<int32_t, AdmissionMapValue>& m, int32_t pk,
        int32_t depth, int32_t sec, bool flag,
        const int32_t* keyIds, int32_t keyCount,
        const int32_t* remIds, int32_t remCount,
        const ValueInterner& vi, ScratchArena& gArena)
    {
        assert(keyCount >= 0 && (keyCount == 0 || keyIds != nullptr));
        assert(remCount >= 0 && (remCount == 0 || remIds != nullptr));
        const ArenaOffset mark = gArena.cursor();
        const int32_t len = 17 + 4 * (keyCount + remCount);
        char* out = reinterpret_cast<char*>(gArena.resolve(gArena.alloc(len, 1)));
        char* at = out;
        const auto put = [&at](const auto& x) {
            std::memcpy(at, &x, sizeof(x));
            at += sizeof(x);
        };
        put(depth);
        put(sec);
        put(static_cast<uint8_t>(flag ? 1 : 0));
        put(keyCount);
        for (int32_t i = 0; i < keyCount; ++i) put(keyIds[i]);
        put(remCount);
        for (int32_t i = 0; i < remCount; ++i) put(remIds[i]);
        assert(at == out + len
            && "insertAdmissionIdsBlob: fill diverged from the closed-form length");
        insertAdmissionBlobSorted(m, pk, out, len, vi, gArena);
        gArena.popTo(mark);
    }

    /// @brief Deposit one rejected value from PRE-MINTED id runs — the
    ///        value-object-free door onto the D-172 blob splice.
    ///
    /// @details
    /// The id-run twin of `insertRejectedValue`: serializes the
    /// `Codec<RejectedMapValue>` byte layout DIRECTLY from the caller-owned id
    /// runs (`24 + 4*(siblingCount+levelCount)` closed-form length, the
    /// identical `put` sequence — renamedExpression, expression, iteration,
    /// concreteConstituent, siblingCount + sibling ids, levelCount + levels),
    /// then splices via `insertRejectedBlobSorted`. No minting inside the door.
    /// Byte-identical blob bytes ⇒ identical `rejectedBlobLess` splice position,
    /// dedup verdict, run bytes, and deload bytes to the value-form door.
    ///
    /// @param m           The cold rejected blob map.
    /// @param pk          The packed (templateId, validityId) key.
    /// @param renamedId   `renamedExpression` value id.
    /// @param exprId      `expression` value id.
    /// @param iteration   The rejection iteration.
    /// @param concreteId  `concreteConstituent` value id.
    /// @param siblingIds  Sibling value ids (positional order); null iff
    ///                    `siblingCount == 0`.
    /// @param siblingCount The sibling count (`>= 0`).
    /// @param levels      The levels, ASCENDING strictly-unique (`std::set<int>`
    ///                    iteration order); null iff `levelCount == 0`.
    /// @param levelCount  The level count (`>= 0`).
    /// @param vi          The owning LB's value interner (comparator state,
    ///                    read-only — no mint here).
    /// @param gArena      The per-slot gen-scratch arena; reclaimed by this
    ///                    function's own mark/popTo.
    /// @invariant No mint inside the door; the ids come in. @p levels must be
    ///            ascending strictly-unique (the `std::set<int>` storage order).
    /// @see `insertRejectedValue` — the value-form sibling + test oracle;
    ///      `serializeRejectedValueToArena`, `insertRejectedBlobSorted`.
    inline void insertRejectedIdsBlob(
        TypedColdBlobMap<int32_t, RejectedMapValue>& m, int32_t pk,
        int32_t renamedId, int32_t exprId, int32_t iteration, int32_t concreteId,
        const int32_t* siblingIds, int32_t siblingCount,
        const int* levels, int32_t levelCount,
        const ValueInterner& vi, ScratchArena& gArena)
    {
        assert(siblingCount >= 0 && (siblingCount == 0 || siblingIds != nullptr));
        assert(levelCount >= 0 && (levelCount == 0 || levels != nullptr));
        const ArenaOffset mark = gArena.cursor();
        const int32_t len = 24 + 4 * (siblingCount + levelCount);
        char* out = reinterpret_cast<char*>(gArena.resolve(gArena.alloc(len, 1)));
        char* at = out;
        const auto put = [&at](const auto& x) {
            std::memcpy(at, &x, sizeof(x));
            at += sizeof(x);
        };
        put(renamedId);
        put(exprId);
        put(iteration);
        put(concreteId);
        put(siblingCount);
        for (int32_t i = 0; i < siblingCount; ++i) put(siblingIds[i]);
        put(levelCount);
        for (int32_t i = 0; i < levelCount; ++i) {
            assert((i == 0 || levels[i] > levels[i - 1])
                && "insertRejectedIdsBlob: levels must be ascending strictly-unique");
            put(static_cast<int32_t>(levels[i]));
        }
        assert(at == out + len
            && "insertRejectedIdsBlob: fill diverged from the closed-form length");
        insertRejectedBlobSorted(m, pk, out, len, vi, gArena);
        gArena.popTo(mark);
    }

    /// @brief Record codec for `RejectedMapIntegrationValue` — a
    ///        `rejectedMapIntegration` set member.
    ///
    /// @details
    /// Layout: `concreteConstituent`, `compoundExpression` (int32 each), then
    /// `siblings` (int32 count + int32 each, positional order).
    ///
    /// @see `RejectedMapIntegrationValue`, `Codec`.
    template <>
    struct Codec<RejectedMapIntegrationValue> {
        /// @brief Serialize a value to its canonical blob.
        ///
        /// @param v The value.
        /// @return The byte blob.
        static std::vector<char> serialize(const RejectedMapIntegrationValue& v) {
            std::vector<char> out;
            const auto put = [&out](const auto& x) {
                const char* p = reinterpret_cast<const char*>(&x);
                out.insert(out.end(), p, p + sizeof(x));
            };
            put(v.concreteConstituent);
            put(v.compoundExpression);
            put(static_cast<int32_t>(v.siblings.size()));
            for (const int32_t id : v.siblings) put(id);
            return out;
        }

        /// @brief Deserialize a value from its blob.
        ///
        /// @param data Blob bytes.
        /// @param n    Blob length.
        /// @return The decoded value.
        static RejectedMapIntegrationValue deserialize(const char* data,
                                                       int32_t n) {
            RejectedMapIntegrationValue v;
            const char* cur = data;
            const char* const end = data + n;
            const auto get = [&cur, end](auto& x) {
                assert(cur + sizeof(x) <= end
                    && "Codec<RejectedMapIntegrationValue>: blob too short");
                std::memcpy(&x, cur, sizeof(x));
                cur += sizeof(x);
            };
            get(v.concreteConstituent);
            get(v.compoundExpression);
            int32_t sc = 0; get(sc);
            v.siblings.resize(static_cast<std::size_t>(sc));
            for (int32_t i = 0; i < sc; ++i)
                get(v.siblings[static_cast<std::size_t>(i)]);
            assert(cur == end
                && "Codec<RejectedMapIntegrationValue>: blob too long");
            return v;
        }
    };

    /// @brief Record codec for `IntegrationEntry` — one flattened
    ///        `admissionMapIntegration` inner-map entry.
    ///
    /// @details
    /// Layout: `instruction.markedGoal` (int32), `instruction.data` count (int32),
    /// then per `IntLogicalEntity`: `category`, `signature`, `arity`, `definedSet`
    /// (int32 each) + `elements` (int32 count + int32 each); finally `valueIds`
    /// (int32 count + int32 each).
    ///
    /// @see `IntegrationEntry`, `IntLogicalEntity`, `Codec`.
    template <>
    struct Codec<IntegrationEntry> {
        /// @brief Serialize an entry to its canonical blob.
        ///
        /// @param v The entry.
        /// @return The byte blob.
        static std::vector<char> serialize(const IntegrationEntry& v) {
            std::vector<char> out;
            const auto put = [&out](const auto& x) {
                const char* p = reinterpret_cast<const char*>(&x);
                out.insert(out.end(), p, p + sizeof(x));
            };
            put(v.instruction.markedGoal);
            put(static_cast<int32_t>(v.instruction.data.size()));
            for (const IntLogicalEntity& e : v.instruction.data) {
                put(e.category);
                put(e.signature);
                put(static_cast<int32_t>(e.arity));
                put(e.definedSet);
                put(static_cast<int32_t>(e.elements.size()));
                for (const int32_t id : e.elements) put(id);
            }
            put(static_cast<int32_t>(v.valueIds.size()));
            for (const int32_t id : v.valueIds) put(id);
            return out;
        }

        /// @brief Deserialize an entry from its blob.
        ///
        /// @param data Blob bytes.
        /// @param n    Blob length.
        /// @return The decoded entry.
        static IntegrationEntry deserialize(const char* data, int32_t n) {
            IntegrationEntry v;
            const char* cur = data;
            const char* const end = data + n;
            const auto get = [&cur, end](auto& x) {
                assert(cur + sizeof(x) <= end
                    && "Codec<IntegrationEntry>: blob shorter than the layout");
                std::memcpy(&x, cur, sizeof(x));
                cur += sizeof(x);
            };
            get(v.instruction.markedGoal);
            int32_t dc = 0; get(dc);
            v.instruction.data.resize(static_cast<std::size_t>(dc));
            for (int32_t i = 0; i < dc; ++i) {
                IntLogicalEntity& e =
                    v.instruction.data[static_cast<std::size_t>(i)];
                get(e.category);
                get(e.signature);
                int32_t arity = 0; get(arity); e.arity = arity;
                get(e.definedSet);
                int32_t ec = 0; get(ec);
                e.elements.resize(static_cast<std::size_t>(ec));
                for (int32_t j = 0; j < ec; ++j)
                    get(e.elements[static_cast<std::size_t>(j)]);
            }
            int32_t vc = 0; get(vc);
            v.valueIds.resize(static_cast<std::size_t>(vc));
            for (int32_t i = 0; i < vc; ++i)
                get(v.valueIds[static_cast<std::size_t>(i)]);
            assert(cur == end
                && "Codec<IntegrationEntry>: blob longer than the layout");
            return v;
        }
    };

    /// @brief Decode a rejected-integration key's whole value run into its sorted
    ///        set form — the cold blob map read snapshot.
    ///
    /// @details
    /// `rejectedMapIntegration` stores each `RejectedMapIntegrationValue` as a
    /// record blob in a run kept sorted by `DecodedRejectedIntegrationValueLess`.
    /// This rebuilds that run as the historical `std::set` form (same comparator,
    /// order, dedup) so the snapshot-and-iterate callers
    /// (`revisitRejectedIntegration2`, the equi-class hook, the dump) read it
    /// exactly as the former heap set. A key never minted yields an empty set — a
    /// defined query result, not a failure (D-172).
    ///
    /// @param m  The cold rejected-integration blob map.
    /// @param pk The packed (templateId, validityId) key.
    /// @param vi The owning LB's value interner (the comparator's state).
    /// @return The key's value set; empty when `pk` was never minted.
    inline RejectedIntegrationValueSet rejectedIntegrationRecordsAt(
        const TypedColdBlobMap<int32_t, RejectedMapIntegrationValue>& m,
        int32_t pk, const ValueInterner& vi)
    {
        RejectedIntegrationValueSet s(DecodedRejectedIntegrationValueLess{ &vi });
        const int32_t id = m.lookup(pk);
        if (id != 0) {
            const std::vector<RejectedMapIntegrationValue> run = m.recordsAt(id);
            for (const RejectedMapIntegrationValue& r : run) s.insert(r);
        }
        return s;
    }

    /// @brief Read-modify-write one `RejectedMapIntegrationValue` into a
    ///        rejected-integration key's run, preserving sorted-unique semantics.
    ///
    /// @details
    /// The cold-map equivalent of the retired `rejectedIntegrationValuesAt(m, pk,
    /// vi).insert(value)`: decode the existing run into the sorted set, insert
    /// `value` (dedup + order via `DecodedRejectedIntegrationValueLess`), then
    /// write the whole run back with `assignRun`. The run stays canonical, so the
    /// deload bytes and the sacred dump are byte-identical to the former heap set.
    /// Mints `pk` on first touch. Single-threaded write side only (I-83)
    /// (D-172).
    ///
    /// @param m     The cold rejected-integration blob map.
    /// @param pk    The packed (templateId, validityId) key.
    /// @param value The value to insert (sorted + deduped into the run).
    /// @param vi    The owning LB's value interner (the comparator's state).
    inline void insertRejectedIntegrationValue(
        TypedColdBlobMap<int32_t, RejectedMapIntegrationValue>& m,
        int32_t pk, const RejectedMapIntegrationValue& value, const ValueInterner& vi)
    {
        RejectedIntegrationValueSet s(DecodedRejectedIntegrationValueLess{ &vi });
        const int32_t id = m.lookup(pk);
        if (id != 0) {
            const std::vector<RejectedMapIntegrationValue> run = m.recordsAt(id);
            for (const RejectedMapIntegrationValue& r : run) s.insert(r);
        }
        s.insert(value);
        m.assignRun(pk, std::vector<RejectedMapIntegrationValue>(s.begin(), s.end()));
    }

    /// @brief Zero-copy field view over one serialized
    ///        `RejectedMapIntegrationValue` blob — the read half of the D-172
    ///        integration blob splice.
    ///
    /// @details
    /// Framing exactly as `Codec<RejectedMapIntegrationValue>`:
    /// `concreteConstituent` (int32 @0), `compoundExpression` (int32 @4),
    /// `siblingCount` (int32 @8), sibling ids (int32 @ `12 + 4*i`). Total length
    /// `12 + 4*siblingCount`, walked once and asserted at construction.
    ///
    /// @see `Codec<RejectedMapIntegrationValue>` — the byte-layout owner;
    ///      `rejectedIntegrationBlobLess`, `RejectedValueBlobView` — the sibling.
    struct RejectedIntegrationValueBlobView {
        const char* p;
        int32_t len;

        /// @brief Construct over one blob's bytes, asserting the exact frame.
        /// @param p_   The blob's bytes (a `peekRecordBytes` span or a verbatim
        ///             gen-arena copy).
        /// @param len_ The blob's byte length.
        RejectedIntegrationValueBlobView(const char* p_, int32_t len_)
            : p(p_), len(len_) {
            assert(len >= 12
                && "RejectedIntegrationValueBlobView: blob shorter than the fixed prefix");
            const int32_t sc = siblingCount();
            assert(sc >= 0 && len == 12 + 4 * sc
                && "RejectedIntegrationValueBlobView: blob length does not match the layout");
        }

        /// @brief Read one int32 field at a byte offset (unaligned-safe).
        /// @param off The byte offset into the blob.
        /// @return The int32 at `off`.
        int32_t readI32(int32_t off) const {
            assert(off >= 0 && off + 4 <= len
                && "RejectedIntegrationValueBlobView: read past the blob");
            int32_t v;
            std::memcpy(&v, p + off, sizeof(v));
            return v;
        }

        /// @brief The record's `concreteConstituent` value-interner id.
        /// @return The int32 at offset 0.
        int32_t concreteConstituentId() const { return readI32(0); }

        /// @brief The record's `compoundExpression` value-interner id.
        /// @return The int32 at offset 4.
        int32_t compoundExpressionId() const { return readI32(4); }

        /// @brief The sibling count.
        /// @return The int32 at offset 8.
        int32_t siblingCount() const { return readI32(8); }

        /// @brief Sibling `i`'s value-interner id (positional order).
        /// @param i The sibling index; asserted `0 <= i < siblingCount()`.
        /// @return The int32 at offset `12 + 4*i`.
        int32_t siblingId(int32_t i) const {
            assert(i >= 0 && i < siblingCount()
                && "RejectedIntegrationValueBlobView::siblingId: index out of range");
            return readI32(12 + 4 * i);
        }
    };

    /// @brief A read-only snapshot of one rejected-integration key's value run —
    ///        an array of zero-decode @ref RejectedIntegrationValueBlobView over
    ///        verbatim blob copies on a gen-scratch arena.
    ///
    /// @details
    /// The integration-side sibling of @ref RejectedRunSnapshot: the heap-free
    /// replacement for `rejectedIntegrationRecordsAt(m, pk, vi)` on the
    /// snapshot-and-iterate paths (`revisitRejectedIntegration2`,
    /// `applyEquivalenceClassToRejectedMapIntegration`). The copies live on the
    /// arena, so the snapshot survives any later mutation of @p m
    /// (`insertRejectedIntegrationValue` / `eraseBlobIf`) during the walk (09b
    /// pitfall 6, I-99). The stored run is canonical under
    /// `DecodedRejectedIntegrationValueLess` (the RMW contract), so iterating the
    /// views reproduces the former `rejectedIntegrationRecordsAt` set order
    /// byte-for-byte.
    ///
    /// @invariant The struct owns nothing — `views` and the copied bytes ride the
    ///            snapshotting arena; the caller must not let a view outlive that
    ///            arena's rewind. `count == 0` for a never-minted key is a defined
    ///            query result (Rule 19).
    /// @see rejectedIntegrationRecordsAt — the retained heap-set oracle;
    ///      RejectedIntegrationValueBlobView, snapshotRejectedIntegrationRun,
    ///      RejectedRunSnapshot.
    struct RejectedIntegrationRunSnapshot {
        const RejectedIntegrationValueBlobView* views;
        int32_t count;
    };

    /// @brief Snapshot a rejected-integration key's value run as verbatim blob
    ///        copies + zero-decode views on a gen-scratch arena.
    ///
    /// @details
    /// Byte-for-byte the @ref snapshotRejectedRun body for
    /// `rejectedMapIntegration`: looks up @p pk, peeks each record blob
    /// (`peekRecordBytes`), copies the bytes verbatim onto @p gArena, and
    /// constructs one @ref RejectedIntegrationValueBlobView over each copy into an
    /// arena-held array. The stored run is canonical
    /// (`DecodedRejectedIntegrationValueLess` sorted+deduped by the RMW contract),
    /// so iterating the views reproduces the former
    /// `rejectedIntegrationRecordsAt` set order byte-for-byte. Mints nothing. A
    /// never-minted key yields `{ nullptr, 0 }`.
    ///
    /// @param m      The cold rejected-integration blob map (read-only here).
    /// @param pk     The packed `(templateId, validityId)` key.
    /// @param gArena The caller's gen-scratch arena (byte-bump tier); the blob
    ///               copies + view array ride it, reclaimed by the caller's
    ///               `popTo` / task-exit `releaseAll`.
    /// @return `{ views, count }` — non-owning views over the arena copies;
    ///         `count == 0` when @p pk was never minted.
    /// @invariant Copies survive any later mutation of @p m, unlike a live
    ///            `peekRecordBytes` pointer (I-99).
    /// @see rejectedIntegrationRecordsAt — the retained heap oracle;
    ///      RejectedIntegrationRunSnapshot, RejectedIntegrationValueBlobView,
    ///      snapshotRejectedRun.
    inline RejectedIntegrationRunSnapshot snapshotRejectedIntegrationRun(
        const TypedColdBlobMap<int32_t, RejectedMapIntegrationValue>& m,
        int32_t pk, ScratchArena& gArena)
    {
        const int32_t id = m.lookup(pk);
        const int32_t runN = id ? m.runLen(id) : 0;
        if (runN == 0) return RejectedIntegrationRunSnapshot{ nullptr, 0 };
        RejectedIntegrationValueBlobView* views =
            reinterpret_cast<RejectedIntegrationValueBlobView*>(
                gArena.resolve(gArena.alloc(
                    runN * static_cast<int32_t>(
                        sizeof(RejectedIntegrationValueBlobView)),
                    static_cast<int32_t>(
                        alignof(RejectedIntegrationValueBlobView)))));
        for (int32_t j = 0; j < runN; ++j) {
            int32_t bl = 0;
            const char* bp = m.peekRecordBytes(id, j, bl, gArena);
            const ArenaOffset off = gArena.alloc(bl, 1);
            std::memcpy(gArena.resolve(off), bp, static_cast<std::size_t>(bl));
            new (&views[j]) RejectedIntegrationValueBlobView(
                gArena.resolve(off), bl);
        }
        return RejectedIntegrationRunSnapshot{ views, runN };
    }

    /// @brief Decoded-order less-than over two `RejectedMapIntegrationValue`
    ///        blobs — the zero-decode twin of `DecodedRejectedIntegrationValueLess`.
    ///
    /// @details
    /// Field order exactly as `DecodedRejectedIntegrationValueLess::operator()`:
    /// `concreteConstituent` (an id-inequality gate then the `valueIdLess`
    /// decoded compare), then `siblings` (`valueIdVectorLess` semantics as a
    /// three-way walk, shorter-run-first on a shared prefix), then
    /// `compoundExpression`. Distinct ids decoding to equal bytes are impossible
    /// (interner injectivity) and assert.
    ///
    /// @param a  First blob view.
    /// @param b  Second blob view.
    /// @param vi The owning LB's value interner (read-only).
    /// @return `true` iff decoded @p a orders before decoded @p b.
    /// @invariant Comparator-equal implies blob-byte-equal (every compared field
    ///            is an injective interner id or a POD count).
    /// @see `DecodedRejectedIntegrationValueLess` — the heap-set oracle;
    ///      `rejectedBlobLess`, `insertRejectedIntegrationBlobSorted`.
    inline bool rejectedIntegrationBlobLess(
        const RejectedIntegrationValueBlobView& a,
        const RejectedIntegrationValueBlobView& b, const ValueInterner& vi)
    {
        const auto idLess3 = [&vi](int32_t x, int32_t y) -> int {
            const int c = compareSpans(vi.decodeView(x), vi.decodeView(y));
            assert(c != 0
                && "distinct value ids decode to equal bytes — interner "
                   "injectivity broken");
            return c;
        };
        if (a.concreteConstituentId() != b.concreteConstituentId()) {
            return idLess3(a.concreteConstituentId(), b.concreteConstituentId()) < 0;
        }
        {
            const int32_t na = a.siblingCount();
            const int32_t nb = b.siblingCount();
            const int32_t n = na < nb ? na : nb;
            for (int32_t i = 0; i < n; ++i) {
                const int32_t xi = a.siblingId(i);
                const int32_t yi = b.siblingId(i);
                if (xi == yi) continue;
                return idLess3(xi, yi) < 0;
            }
            if (na != nb) return na < nb;
        }
        if (a.compoundExpressionId() != b.compoundExpressionId()) {
            return idLess3(a.compoundExpressionId(), b.compoundExpressionId()) < 0;
        }
        return false;   // all fields equal
    }

    /// @brief Serialize one `RejectedMapIntegrationValue` onto a gen-scratch
    ///        byte-bump frame — the arena twin of
    ///        `Codec<RejectedMapIntegrationValue>::serialize`.
    ///
    /// @details
    /// Writes the identical byte layout (`concreteConstituent`,
    /// `compoundExpression`, `siblingCount` + sibling ids) with the identical
    /// `put` sequence into one `alloc(len, 1)` run; the closed-form length
    /// (`12 + 4*siblingCount`) is computed first and the exact-fill asserted.
    ///
    /// @param gArena The per-slot gen-scratch arena the blob bytes land on.
    /// @param v      The value to serialize.
    /// @return Span over the freshly written blob (valid until the caller's
    ///         `popTo` past it).
    /// @see `Codec<RejectedMapIntegrationValue>`, `insertRejectedIntegrationBlobSorted`.
    inline StrSpan serializeRejectedIntegrationValueToArena(ScratchArena& gArena,
        const RejectedMapIntegrationValue& v)
    {
        const int32_t sc = static_cast<int32_t>(v.siblings.size());
        const int32_t len = 12 + 4 * sc;
        char* out = reinterpret_cast<char*>(gArena.resolve(gArena.alloc(len, 1)));
        char* at = out;
        const auto put = [&at](const auto& x) {
            std::memcpy(at, &x, sizeof(x));
            at += sizeof(x);
        };
        put(v.concreteConstituent);
        put(v.compoundExpression);
        put(sc);
        for (const int32_t id : v.siblings) put(id);
        assert(at == out + len
            && "serializeRejectedIntegrationValueToArena: fill diverged from the "
               "closed-form length");
        return StrSpan(out, len);
    }

    /// @brief Splice one serialized `RejectedMapIntegrationValue` blob into a
    ///        rejected-integration key's canonical run at blob level — the arena
    ///        interior of the D-172 RMW, no heap set and no per-record decode.
    ///
    /// @details
    /// Mirror of `insertRejectedBlobSorted` for `rejectedMapIntegration`:
    /// verbatim gen-arena copies of the existing run (copy-before-`assignRun`),
    /// positional insert under the `rejectedIntegrationBlobLess` comparator twin,
    /// duplicate ⟺ byte-equal by interner injectivity (asserted), the run
    /// written back unconditionally through the raw `inner().assignRun` byte
    /// door (a run that would straddle an arena block widens to an empty-run
    /// `assignRun` + per-blob `appendBlobToRun`, byte-identical, unbounded) —
    /// same writes and same `DirtyState` escalation as the value-form
    /// `insertRejectedIntegrationValue` (the retained heap-set RMW oracle).
    /// Mints `pk` on first touch. Single-threaded write side only (I-83).
    ///
    /// @param m       The cold rejected-integration blob map.
    /// @param pk      The packed (templateId, validityId) key.
    /// @param newBlob The new value's serialized blob bytes.
    /// @param newLen  The new blob's byte length.
    /// @param vi      The owning LB's value interner (comparator state, read-only).
    /// @param gArena  The per-slot gen-scratch arena; reclaimed by this
    ///                function's own mark/popTo (LIFO-self-contained).
    /// @invariant The written run is byte-identical to the value-form
    ///            decode-into-`std::set`-insert-reserialize cycle.
    /// @see `insertRejectedIntegrationValue` — the value-form oracle;
    ///      `rejectedIntegrationBlobLess`, `Codec<RejectedMapIntegrationValue>`.
    inline void insertRejectedIntegrationBlobSorted(
        TypedColdBlobMap<int32_t, RejectedMapIntegrationValue>& m,
        int32_t pk, const char* newBlob, int32_t newLen,
        const ValueInterner& vi, ScratchArena& gArena)
    {
        const ArenaOffset mark = gArena.cursor();
        struct BlobRec { ArenaOffset off; int32_t len; };
        DirtyState recsDirty = DirtyState::Clean;
        PagedVector<BlobRec> recs(&gArena, &recsDirty);
        const int32_t id = m.lookup(pk);
        if (id != 0) {
            const int32_t rl = m.runLen(id);
            for (int32_t j = 0; j < rl; ++j) {
                int32_t bl = 0;
                const char* bp = m.peekRecordBytes(id, j, bl, gArena);
                const ArenaOffset coff = gArena.alloc(bl, 1);
                std::memcpy(gArena.resolve(coff), bp,
                            static_cast<std::size_t>(bl));
                recs.push_back(BlobRec{ coff, bl });
            }
        }
        const RejectedIntegrationValueBlobView nv(newBlob, newLen);
        const int32_t M = recs.size();
        int32_t pos = 0;
        bool dup = false;
        for (; pos < M; ++pos) {
            const BlobRec br = recs[pos];
            const RejectedIntegrationValueBlobView ev(
                reinterpret_cast<const char*>(gArena.resolve(br.off)), br.len);
            if (rejectedIntegrationBlobLess(ev, nv, vi)) continue;  // existing < new
            if (!rejectedIntegrationBlobLess(nv, ev, vi)) {
                dup = (br.len == newLen)
                    && std::memcmp(gArena.resolve(br.off), newBlob,
                                   static_cast<std::size_t>(newLen)) == 0;
                assert(dup
                    && "comparator-equal rejected-integration blobs differ in "
                       "bytes — interner injectivity broken");
            }
            break;
        }
        const int32_t outCount = dup ? M : M + 1;
        int64_t total = 0;
        for (int32_t j = 0; j < M; ++j) total += recs[j].len;
        if (!dup) total += newLen;
        // A run whose concatenated bytes would straddle an arena block (an arena
        // alloc may not exceed one block) takes the widening door: open the run
        // empty, then append each blob at the run-end in sorted order —
        // existing[0..pos), newBlob at pos when !dup, existing[pos..M). The map's
        // blob pool straddles pages, so the run there is unbounded, and
        // appendBlobToRun is byte-identical to the one-block concat's assignRun.
        // gArena is untouched by these map writes, so the recs / newBlob copies
        // stay valid across the loop.
        if (total > ExecutionParameters::kMaxAdmissionRunBytes) {
            const int32_t wid = m.inner().assignRun(pk, nullptr, nullptr, 0);
            for (int32_t j = 0; j <= M; ++j) {
                if (!dup && j == pos)
                    m.inner().appendBlobToRun(wid, newBlob, newLen);
                if (j == M) break;
                const BlobRec br = recs[j];
                m.inner().appendBlobToRun(wid, gArena.resolve(br.off), br.len);
            }
            gArena.popTo(mark);
            return;
        }
        // Common case: the whole run fits one arena block — one concat + assignRun.
        const ArenaOffset concatOff =
            gArena.alloc(static_cast<int32_t>(total), 1);
        const ArenaOffset lensOff = gArena.alloc(
            outCount * static_cast<int32_t>(sizeof(int32_t)),
            static_cast<int32_t>(alignof(int32_t)));
        char* concat = reinterpret_cast<char*>(gArena.resolve(concatOff));
        int32_t* lens =
            reinterpret_cast<int32_t*>(gArena.resolve(lensOff));
        int32_t at = 0;
        int32_t oi = 0;
        for (int32_t j = 0; j <= M; ++j) {
            if (!dup && j == pos) {
                std::memcpy(concat + at, newBlob,
                            static_cast<std::size_t>(newLen));
                lens[oi++] = newLen;
                at += newLen;
            }
            if (j == M) break;
            const BlobRec br = recs[j];
            std::memcpy(concat + at, gArena.resolve(br.off),
                        static_cast<std::size_t>(br.len));
            lens[oi++] = br.len;
            at += br.len;
        }
        assert(oi == outCount && at == static_cast<int32_t>(total)
            && "insertRejectedIntegrationBlobSorted: concatenation diverged from "
               "the computed frame");
        m.inner().assignRun(pk, concat, lens, outCount);
        gArena.popTo(mark);
    }

    /// @brief Deposit one rejected-integration value from PRE-MINTED id runs —
    ///        the value-object-free door onto the D-172 integration blob splice.
    ///
    /// @details
    /// The id-run twin of `insertRejectedIntegrationValue`: serializes the
    /// `Codec<RejectedMapIntegrationValue>` byte layout DIRECTLY from the
    /// caller-owned id run (`12 + 4*siblingCount`; `put` sequence
    /// concreteConstituent, compoundExpression, siblingCount + sibling ids),
    /// then splices via `insertRejectedIntegrationBlobSorted`. No minting inside
    /// the door — the ids arrive already interned, so the CALLER owns the mint
    /// order (frozen by its pinning test). Byte-identical blob bytes ⇒ identical
    /// splice position, dedup verdict, run bytes, and deload bytes to the
    /// value-form door.
    ///
    /// @param m            The cold rejected-integration blob map.
    /// @param pk           The packed (templateId, validityId) key.
    /// @param concreteId   `concreteConstituent` value id.
    /// @param compoundId   `compoundExpression` value id.
    /// @param siblingIds   Sibling value ids (positional order); null iff
    ///                     `siblingCount == 0`.
    /// @param siblingCount The sibling count (`>= 0`).
    /// @param vi           The owning LB's value interner (comparator state,
    ///                     read-only — no mint here).
    /// @param gArena       The per-slot gen-scratch arena; reclaimed by this
    ///                     function's own mark/popTo.
    /// @invariant No mint inside the door — the ids come in; the caller's mint
    ///            order is the byte-transparent contract.
    /// @see `insertRejectedIntegrationValue` — the value-form sibling + test
    ///      oracle; `serializeRejectedIntegrationValueToArena`,
    ///      `insertRejectedIntegrationBlobSorted`.
    inline void insertRejectedIntegrationIdsBlob(
        TypedColdBlobMap<int32_t, RejectedMapIntegrationValue>& m, int32_t pk,
        int32_t concreteId, int32_t compoundId,
        const int32_t* siblingIds, int32_t siblingCount,
        const ValueInterner& vi, ScratchArena& gArena)
    {
        assert(siblingCount >= 0 && (siblingCount == 0 || siblingIds != nullptr));
        const ArenaOffset mark = gArena.cursor();
        const int32_t len = 12 + 4 * siblingCount;
        char* out = reinterpret_cast<char*>(gArena.resolve(gArena.alloc(len, 1)));
        char* at = out;
        const auto put = [&at](const auto& x) {
            std::memcpy(at, &x, sizeof(x));
            at += sizeof(x);
        };
        put(concreteId);
        put(compoundId);
        put(siblingCount);
        for (int32_t i = 0; i < siblingCount; ++i) put(siblingIds[i]);
        assert(at == out + len
            && "insertRejectedIntegrationIdsBlob: fill diverged from the closed-form length");
        insertRejectedIntegrationBlobSorted(m, pk, out, len, vi, gArena);
        gArena.popTo(mark);
    }

    /// @brief Decode an admission-integration key's whole run back into its nested
    ///        `IntegrationEntryMap` form — the cold blob map read snapshot.
    ///
    /// @details
    /// `admissionMapIntegration` stores its inner `std::map<IntInstruction,
    /// ValueIdSet>` flattened to a run of `IntegrationEntry` record blobs, one per
    /// inner entry, held in the inner map's decoded order. This rebuilds that
    /// nested map (both stateful comparators — `DecodedInstructionLess` outer,
    /// `DecodedIdLess` inner — threaded from @p vi) so the snapshot-and-iterate
    /// callers (`isAdmittedIntegration`, the equi-class hook, the dump) read it
    /// exactly as the former heap inner map. A key never minted yields an empty
    /// map — a defined query result, not a failure
    /// (D-172).
    ///
    /// @param m  The cold admission-integration blob map.
    /// @param pk The packed (templateId, validityId) key.
    /// @param vi The owning LB's value interner (both comparators' state).
    /// @return The key's instruction map; empty when `pk` was never minted.
    inline IntegrationEntryMap admissionIntegrationRecordsAt(
        const TypedColdBlobMap<int32_t, IntegrationEntry>& m,
        int32_t pk, const ValueInterner& vi)
    {
        IntegrationEntryMap em(DecodedInstructionLess{ &vi });
        const int32_t id = m.lookup(pk);
        if (id != 0) {
            const std::vector<IntegrationEntry> run = m.recordsAt(id);
            for (const IntegrationEntry& e : run) {
                ValueIdSet payload(DecodedIdLess{ &vi });
                for (const int32_t v : e.valueIds) payload.insert(v);
                em.emplace(e.instruction, std::move(payload));
            }
        }
        return em;
    }

    /// @brief Flatten a nested `IntegrationEntryMap` back to a sorted
    ///        `IntegrationEntry` run — the write half of the cold round trip.
    ///
    /// @details
    /// Run order is the map's decoded order (`DecodedInstructionLess`); each
    /// entry's payload order is the `ValueIdSet`'s decoded order (`DecodedIdLess`).
    /// Producing the run from the comparator-bearing containers keeps the cold
    /// bytes canonical without the codec needing any comparator
    /// (D-172).
    ///
    /// @param em The instruction map to flatten.
    /// @return The record run, ready for `assignRun`.
    inline std::vector<IntegrationEntry> flattenIntegrationEntryMap(
        const IntegrationEntryMap& em)
    {
        std::vector<IntegrationEntry> run;
        run.reserve(em.size());
        for (const auto& kv : em) {
            IntegrationEntry e;
            e.instruction = kv.first;
            e.valueIds.assign(kv.second.begin(), kv.second.end());
            run.push_back(std::move(e));
        }
        return run;
    }

    /// @brief Per-entity id-form record inside @ref ArenaIntegrationMap — the
    ///        arena twin of one `IntLogicalEntity` (already int-id, so a flat POD).
    ///
    /// @details `catId` / `sigId` / `dsId` / `arity` are the `IntLogicalEntity`
    /// int-ids verbatim; the element ids are the half-open run
    /// `[elemsStart, elemsStart + elemsCount)` into the owning map's `elemCol`.
    /// @see ArenaIntegrationMap, IntLogicalEntity.
    struct ArenaIntLE {
        int32_t catId, sigId, dsId, arity, elemsStart, elemsCount;
    };

    /// @brief Per-entry record inside @ref ArenaIntegrationMap — one
    ///        `(IntInstruction, ValueIdSet)` map entry.
    ///
    /// @details The instruction key is the `[leStart, leStart + leCount)` run of
    /// @ref ArenaIntLE plus `markedGoalId`; the value set is the append-only chain
    /// headed at `valHead` (`-1` when empty), `valCount` members, walked through
    /// the owning map's `valNodes`.
    /// @see ArenaIntegrationMap, IntInstruction.
    struct ArenaIntEntry {
        int32_t leStart, leCount, markedGoalId, valHead, valCount;
    };

    /// @brief One node of an entry's value chain in @ref ArenaIntegrationMap.
    ///
    /// @details `prev` links toward the chain tail (`-1` ends it). The chain is
    /// append-only, so a value insert never shifts another entry's storage; the
    /// `DecodedIdLess` order is imposed at read (`valuesAt` / `flatten`).
    struct ArenaValNode { int32_t valueId, prev; };

    /// @brief Transient arena-backed twin of @ref IntegrationEntryMap — the
    ///        per-call decoded admission-integration snapshot, off the malloc heap.
    ///
    /// @details
    /// Replaces the heap `std::map<IntInstruction, ValueIdSet,
    /// DecodedInstructionLess>` (and its per-site `std::vector<std::pair<...>>`
    /// snapshots) with flat int-POD `PagedVector`s on a per-slot `genScratchArenas`
    /// arena: the entry records, the `IntLogicalEntity` records, the element-id
    /// column, and the append-only value chains. `IntInstruction` is ALREADY the
    /// int-id form, so no string interning is needed — the records are flat ints.
    /// The cold-blob boundary itself is DIRECT: `buildFromCold` deserializes each
    /// record's canonical bytes straight into these `PagedVector`s (via the
    /// substrate's no-allocation `peekRecordBytes` read door — no intermediate heap
    /// `IntegrationEntry`), and `writeToCold` serializes them straight back into the
    /// blob run (the engine's raw `assignRun` byte door — no `flatten()` heap
    /// vector). Only the flat `bytes` / `lens` byte buffers cross the boundary —
    /// the exact buffers the typed `assignRun` builds internally. The heap-form
    /// `build` / `flatten` / `flattenIntegrationEntryMap` survive solely as the test
    /// golden references the direct paths are checked byte-identical against.
    ///
    /// Reproduces the heap map's observable behaviour byte-for-byte:
    /// - **find-or-emplace** (@ref findOrEmplace) is ID-EQUALITY — one interned
    ///   `IntInstruction` per decoded form, so decode-equal == id-equal; create-if-
    ///   absent appends a new entry with an empty value chain (`payloadAt`).
    /// - **value insert** (@ref insertValue) is `std::set<int32, DecodedIdLess>`'s
    ///   insert: dedup (chain walk) then append; the `DecodedIdLess` order is
    ///   imposed at read, never stored.
    /// - **flatten** (@ref flatten) emits entries in `DecodedInstructionLess` order
    ///   (an index-sort reproducing the comparator exactly) with each value run in
    ///   `DecodedIdLess` order — byte-identical to `flattenIntegrationEntryMap`.
    /// - the **snapshot + mutate-while-iterate** sites take a sorted entry-index
    ///   snapshot (@ref sortedIndices) before the loop and iterate it while
    ///   `findOrEmplace` / `insertValue` grow EXISTING entries' value chains (no new
    ///   entry reaches this map during the loop), exactly as the heap
    ///   `vector<pair>(em.begin(), em.end())` copy does.
    ///
    /// @invariant Lives within one integration call on its slot's arena;
    ///            single-threaded (the marker site + the equi-class hooks run
    ///            single-threaded-in-burst). Every returned `IntInstruction` /
    ///            `std::vector` is an owned heap boundary copy.
    /// @see IntegrationEntryMap — the heap form this replaces.
    /// @see flattenIntegrationEntryMap, payloadAt, admissionIntegrationRecordsAt.
    struct ArenaIntegrationMap {
        LbArena* arena;
        DirtyState dirty = DirtyState::Clean;
        const ValueInterner* vi;
        PagedVector<ArenaIntEntry> entries;
        PagedVector<ArenaIntLE> les;
        PagedVector<int32_t> elemCol;
        PagedVector<ArenaValNode> valNodes;

        /// @brief Bind to a per-slot scratch arena + the comparator interner.
        /// @param a The arena (outlives this map).
        /// @param v The value interner (the `DecodedInstructionLess`/`DecodedIdLess`
        ///          decode source).
        ArenaIntegrationMap(LbArena* a, const ValueInterner* v)
            : arena(a), vi(v), entries(a, &dirty), les(a, &dirty),
              elemCol(a, &dirty), valNodes(a, &dirty) {}
        ArenaIntegrationMap(const ArenaIntegrationMap&) = delete;
        ArenaIntegrationMap& operator=(const ArenaIntegrationMap&) = delete;

        /// @brief Append a new entry for @p instr with an empty value chain.
        /// @param instr The instruction key. @return The new entry index.
        int32_t appendEntry(const IntInstruction& instr) {
            ArenaIntEntry e;
            e.leStart = les.size();
            e.leCount = static_cast<int32_t>(instr.data.size());
            for (const IntLogicalEntity& ile : instr.data) {
                ArenaIntLE le;
                le.catId = ile.category; le.sigId = ile.signature;
                le.dsId = ile.definedSet; le.arity = ile.arity;
                le.elemsStart = elemCol.size();
                le.elemsCount = static_cast<int32_t>(ile.elements.size());
                for (const int32_t eid : ile.elements) elemCol.push_back(eid);
                les.push_back(le);
            }
            e.markedGoalId = instr.markedGoal;
            e.valHead = -1;
            e.valCount = 0;
            entries.push_back(e);
            return entries.size() - 1;
        }

        /// @brief Append @p v to entry @p ei's value chain (no dedup — caller-guarded).
        /// @param ei Entry index. @param v Value id.
        void pushValue(int32_t ei, int32_t v) {
            ArenaIntEntry e = entries[ei];
            valNodes.push_back(ArenaValNode{ v, e.valHead });
            e.valHead = valNodes.size() - 1;
            e.valCount += 1;
            entries.setAt(ei, e);
        }

        /// @brief Build from a decoded cold-blob run snapshot — the heap-form
        ///        reference path the direct `buildFromCold` reproduces.
        ///
        /// @details Each entry's value ids are already `DecodedIdLess`-sorted +
        /// deduped (cold-store canonical), so they push straight into the chain.
        /// Retained as the unit test's golden reference (and the semantic definition
        /// `buildFromCold` is checked byte-identical against); production reads go
        /// through `buildFromCold`, which never materializes a `std::vector<
        /// IntegrationEntry>`.
        ///
        /// @param run The decoded record run (e.g. `recordsAt`'s vector).
        /// @see buildFromCold (the direct production read).
        void build(const std::vector<IntegrationEntry>& run) {
            for (const IntegrationEntry& e : run) {
                const int32_t ei = appendEntry(e.instruction);
                for (const int32_t v : e.valueIds) pushValue(ei, v);
            }
        }

        /// @brief Build directly from `m[pk]`'s cold run — deserialize each record's
        ///        canonical bytes straight into the arena form, NO intermediate heap
        ///        `IntegrationEntry`.
        ///
        /// @details
        /// Iterates the key's blob run and peeks each record's contiguous bytes
        /// through the substrate's no-allocation `peekRecordBytes` read door
        /// (zero-copy when the blob is single-page; a page straddle copies once onto
        /// the map's own arena via the `ScratchArena&` peek overload — 0% heap, no
        /// `std::vector<char>` spill), then parses the `Codec<IntegrationEntry>` field layout —
        /// `markedGoal`, entity count, then per entity `category` / `signature` /
        /// `arity` / `definedSet` + its element run, then the value run — field by
        /// field straight into `entries` / `les` / `elemCol` / `valNodes`. The value
        /// ids are stored `DecodedIdLess`-canonical and push straight into the chain
        /// in stored order, so the arena state is byte-identical to
        /// `build(m.recordsAt(pk))` without ever materializing a heap record. A key
        /// never minted yields an empty map (a defined query result, not a failure;
        /// D-172).
        ///
        /// @param m  The cold admission-integration blob map.
        /// @param pk The packed (templateId, validityId) key.
        /// @invariant Reads the EXACT field order `Codec<IntegrationEntry>::serialize`
        ///            writes; the unit test enforces byte-identity to the heap path.
        /// @see writeToCold (the write half), build, Codec<IntegrationEntry>,
        ///      TypedCold::peekRecordBytes.
        void buildFromCold(const TypedColdBlobMap<int32_t, IntegrationEntry>& m,
                           int32_t pk) {
            const int32_t id = m.lookup(pk);
            if (id == 0) return;
            const int32_t rl = m.runLen(id);
            for (int32_t j = 0; j < rl; ++j) {
                int32_t len = 0;
                // Page-straddle spill rides the map's own arena (byte-bump tier,
                // independent of its page-tier containers) — 0% heap.
                const char* const data = m.peekRecordBytes(id, j, len, *arena);
                const char* cur = data;
                const char* const end = data + len;
                const auto get = [&cur, end](int32_t& x) {
                    assert(cur + sizeof(int32_t) <= end
                        && "ArenaIntegrationMap::buildFromCold: blob shorter "
                           "than the layout");
                    std::memcpy(&x, cur, sizeof(int32_t));
                    cur += sizeof(int32_t);
                };
                ArenaIntEntry e;
                e.leStart = les.size();
                get(e.markedGoalId);
                int32_t dc = 0; get(dc);
                e.leCount = dc;
                for (int32_t i = 0; i < dc; ++i) {
                    ArenaIntLE le;
                    get(le.catId);
                    get(le.sigId);
                    get(le.arity);
                    get(le.dsId);
                    int32_t ec = 0; get(ec);
                    le.elemsStart = elemCol.size();
                    le.elemsCount = ec;
                    for (int32_t ej = 0; ej < ec; ++ej) {
                        int32_t eid = 0; get(eid);
                        elemCol.push_back(eid);
                    }
                    les.push_back(le);
                }
                e.valHead = -1;
                e.valCount = 0;
                const int32_t ei = entries.size();
                entries.push_back(e);
                int32_t vc = 0; get(vc);
                for (int32_t i = 0; i < vc; ++i) {
                    int32_t v = 0; get(v);
                    pushValue(ei, v);
                }
                assert(cur == end
                    && "ArenaIntegrationMap::buildFromCold: blob longer "
                       "than the layout");
            }
        }

        /// @brief Number of entries. @return The entry count.
        int32_t size() const { return entries.size(); }

        /// @brief Whether stored entry @p ei equals @p q (id-equality).
        /// @param ei Entry index. @param q Query instruction. @return Equal?
        bool entryEquals(int32_t ei, const IntInstruction& q) const {
            const ArenaIntEntry& e = entries[ei];
            if (e.leCount != static_cast<int32_t>(q.data.size())) return false;
            if (e.markedGoalId != q.markedGoal) return false;
            for (int32_t i = 0; i < e.leCount; ++i) {
                const ArenaIntLE& le = les[e.leStart + i];
                const IntLogicalEntity& qle = q.data[static_cast<std::size_t>(i)];
                if (le.catId != qle.category || le.sigId != qle.signature
                    || le.dsId != qle.definedSet || le.arity != qle.arity) return false;
                if (le.elemsCount != static_cast<int32_t>(qle.elements.size())) return false;
                for (int32_t j = 0; j < le.elemsCount; ++j)
                    if (elemCol[le.elemsStart + j]
                        != qle.elements[static_cast<std::size_t>(j)]) return false;
            }
            return true;
        }

        /// @brief Find @p instr or append it (empty value chain) — `payloadAt`.
        /// @param instr The instruction key. @return The entry index.
        int32_t findOrEmplace(const IntInstruction& instr) {
            const int32_t n = entries.size();
            for (int32_t i = 0; i < n; ++i) if (entryEquals(i, instr)) return i;
            return appendEntry(instr);
        }

        /// @brief A half-open `[start, start+len)` id run inside a flattened
        ///        instruction pool — the return of @ref flattenWorkInstructionInto.
        /// @see flattenWorkInstructionInto, findOrEmplaceFlattened.
        struct FlatRange { int32_t start; int32_t len; };

        /// @brief Mint a @ref WorkInstruction's fields into @p vi and append its
        ///        flattened id run to @p pool — the heap-free twin of
        ///        `encodeWorkInstruction` followed by a manual pool flatten.
        ///
        /// @details
        /// Replaces the `encodeWorkInstruction(wi, vi)` → throwaway heap
        /// `IntInstruction` → manual `pool.push_back` flatten pattern with a single
        /// pass that mints straight into @p vi and stores straight into @p pool —
        /// no heap `IntInstruction` / `IntLogicalEntity` / `std::vector<int32_t>`
        /// materialized.
        ///
        /// **The `valueInterner` mint sequence is FROZEN, byte-for-byte
        /// `encodeWorkInstruction`:** iterate entities in order; per entity mint
        /// `vi.encode(category)`, then `vi.encode(element)` for each element in
        /// order, then `vi.encode(signature)` (arity is a raw int, NOT minted),
        /// then `vi.encode(definedSet)`; AFTER all entities, mint
        /// `vi.encode(markedGoal)` LAST. First-seen mint order defines the id→string
        /// map that reaches the stored `Codec<IntegrationEntry>` blob and the
        /// `.deload/` id table ([I-84](#i-84)), so a reordered mint would diverge the
        /// deload bytes even when no theorem or verifier byte moves — the strict
        /// freeze the `flatten_work_instruction_into_matches_encode` fresh-interner
        /// twin asserts.
        ///
        /// The STORAGE layout is an INTERNAL free choice (the pool is transient
        /// `genScratchArenas`, never deloaded; only the `(start, len)` range is
        /// recorded downstream): `[entityCount][per entity: catId, sigId, arity,
        /// dsId, elemCount, elemIds…][markedGoalId]` — markedGoal LAST so the single
        /// mint pass stores in encode order with no placeholder patch. Element ids
        /// are minted before signature/definedSet but stored after them, so one
        /// entity's element ids are held in a stack `elemIds[]` (cap
        /// `MAX_INSTRUCTION_ELEMENTS`, a loud Rule-19 assert) between mint and store.
        /// @ref findOrEmplaceFlattened parses this exact layout.
        ///
        /// @param wi   The working instruction (resident; spans valid).
        /// @param vi   The owning LB's value interner (mint side).
        /// @param pool The flattened id pool to append into (page tier).
        /// @return The `{start, len}` range of the appended run inside @p pool.
        /// @invariant Mints into @p vi in `encodeWorkInstruction`'s exact order;
        ///            @p wi's own interner (the decode source) is never minted.
        /// @see encodeWorkInstruction — the heap twin. @see findOrEmplaceFlattened —
        ///      parses the layout this writes. @see ExecutionParameters::MAX_INSTRUCTION_ELEMENTS.
        FlatRange flattenWorkInstructionInto(const WorkInstruction& wi, ValueInterner& vi,
                                             PagedVector<int32_t>& pool) const {
            const int32_t start = pool.size();
            const int32_t n = wi.entityCount();
            pool.push_back(n);
            for (int32_t i = 0; i < n; ++i) {
                const int32_t catId = vi.encode(wi.category(i));
                const int32_t ec = wi.elemCount(i);
                assert(ec <= ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
                    && "ArenaIntegrationMap::flattenWorkInstructionInto: element count "
                       "exceeds MAX_INSTRUCTION_ELEMENTS");
                int32_t elemIds[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
                for (int32_t j = 0; j < ec; ++j) elemIds[j] = vi.encode(wi.elemAt(i, j));
                const int32_t sigId = vi.encode(wi.signature(i));
                const int32_t arity = wi.arity(i);
                const int32_t dsId = vi.encode(wi.definedSet(i));
                pool.push_back(catId);
                pool.push_back(sigId);
                pool.push_back(arity);
                pool.push_back(dsId);
                pool.push_back(ec);
                for (int32_t j = 0; j < ec; ++j) pool.push_back(elemIds[j]);
            }
            pool.push_back(vi.encode(wi.markedGoal()));
            return FlatRange{ start, pool.size() - start };
        }

        /// @brief Whether stored entry @p ei equals the flattened run
        ///        `run[0, len)` (id-equality) — the run-form twin of @ref entryEquals.
        ///
        /// @details Parses the canonical flattened layout
        /// (`[entityCount][per entity: catId, sigId, arity, dsId, elemCount,
        /// elemIds…][markedGoalId]` — the layout @ref flattenWorkInstructionInto
        /// writes) sequentially and compares each field to entry @p ei's `les` /
        /// `elemCol` records. Mints NOTHING; compares already-minted ids, so it is
        /// byte-identical to `entryEquals(ei, rebuilt IntInstruction)` by
        /// construction. Every read is bounds-asserted against @p len (a loud
        /// Rule-19 tripwire on a malformed run).
        ///
        /// @param ei  Entry index.
        /// @param run The flattened id run.
        /// @param len The run length (int count).
        /// @return Whether entry @p ei's instruction equals the run.
        /// @see entryEquals — the `IntInstruction` twin. @see findOrEmplaceFlattened.
        bool entryEqualsFlattened(int32_t ei, const int32_t* run, int32_t len) const {
            const ArenaIntEntry& e = entries[ei];
            int32_t pos = 0;
            const auto rd = [&pos, run, len]() -> int32_t {
                assert(pos < len
                    && "ArenaIntegrationMap::entryEqualsFlattened: run shorter "
                       "than the layout");
                return run[pos++];
            };
            const int32_t n = rd();
            if (e.leCount != n) return false;
            for (int32_t i = 0; i < n; ++i) {
                const int32_t catId = rd();
                const int32_t sigId = rd();
                const int32_t arity = rd();
                const int32_t dsId = rd();
                const int32_t ec = rd();
                const ArenaIntLE& le = les[e.leStart + i];
                bool eq = (le.catId == catId && le.sigId == sigId && le.arity == arity
                           && le.dsId == dsId && le.elemsCount == ec);
                for (int32_t j = 0; j < ec; ++j) {
                    const int32_t eid = rd();
                    if (eq && elemCol[le.elemsStart + j] != eid) eq = false;
                }
                if (!eq) return false;
            }
            const int32_t mgId = rd();
            return e.markedGoalId == mgId;
        }

        /// @brief Append a new entry parsed from the flattened run `run[0, len)` —
        ///        the run-form twin of @ref appendEntry.
        ///
        /// @details Parses the canonical flattened layout (the layout
        /// @ref flattenWorkInstructionInto writes) and copies its ids straight into
        /// `entries` / `les` / `elemCol` with an empty value chain — no heap
        /// `IntInstruction`. Every read is bounds-asserted against @p len; the exact
        /// consumption is asserted at the end (a Rule-19 tripwire on a run longer
        /// than the layout).
        ///
        /// @param run The flattened id run.
        /// @param len The run length (int count).
        /// @return The new entry index.
        /// @see appendEntry — the `IntInstruction` twin. @see findOrEmplaceFlattened.
        int32_t appendEntryFlattened(const int32_t* run, int32_t len) {
            int32_t pos = 0;
            const auto rd = [&pos, run, len]() -> int32_t {
                assert(pos < len
                    && "ArenaIntegrationMap::appendEntryFlattened: run shorter "
                       "than the layout");
                return run[pos++];
            };
            ArenaIntEntry e;
            e.leStart = les.size();
            const int32_t n = rd();
            e.leCount = n;
            for (int32_t i = 0; i < n; ++i) {
                ArenaIntLE le;
                le.catId = rd();
                le.sigId = rd();
                le.arity = rd();
                le.dsId = rd();
                const int32_t ec = rd();
                le.elemsStart = elemCol.size();
                le.elemsCount = ec;
                for (int32_t j = 0; j < ec; ++j) elemCol.push_back(rd());
                les.push_back(le);
            }
            e.markedGoalId = rd();
            e.valHead = -1;
            e.valCount = 0;
            entries.push_back(e);
            assert(pos == len
                && "ArenaIntegrationMap::appendEntryFlattened: run longer "
                   "than the layout");
            return entries.size() - 1;
        }

        /// @brief Find the entry equal to the flattened run `run[0, len)` or append
        ///        it (empty value chain) — the run-form twin of @ref findOrEmplace.
        ///
        /// @details Scans `entries` for id-equality (@ref entryEqualsFlattened) and
        /// returns the hit, or appends a new entry (@ref appendEntryFlattened). The
        /// door MINTS NOTHING — it compares and copies already-minted ids — so it is
        /// byte-identical to `findOrEmplace(rebuilt IntInstruction)` by construction
        /// (same id comparison, same append of the same ids). The contiguous
        /// `const int32_t*` mirrors `writeToCold`'s contiguous-on-`*arena` idiom.
        ///
        /// @param run The flattened id run (@ref flattenWorkInstructionInto layout).
        /// @param len The run length (int count).
        /// @return The entry index (hit or newly appended).
        /// @see findOrEmplace — the `IntInstruction` twin. @see findOrEmplaceWork.
        int32_t findOrEmplaceFlattened(const int32_t* run, int32_t len) {
            const int32_t n = entries.size();
            for (int32_t i = 0; i < n; ++i)
                if (entryEqualsFlattened(i, run, len)) return i;
            return appendEntryFlattened(run, len);
        }

        /// @brief Find the entry for @p wi or append it (empty value chain) — the
        ///        heap-free twin of `findOrEmplace(encodeWorkInstruction(wi, vi))`.
        ///
        /// @details
        /// Composes @ref flattenWorkInstructionInto (into a byte-bump run instead of
        /// a pool) with @ref findOrEmplaceFlattened: under a `ScratchScope` on the
        /// map's own `*arena`, compute the run length in one pass over @p wi
        /// (`2 + Σ(5 + elemCount)`), one byte-bump `alloc`, mint+fill the run in the
        /// FROZEN `encodeWorkInstruction` mint order, then id-compare / append via
        /// @ref findOrEmplaceFlattened. The append copies the ids into the page-tier
        /// records, so the scope may rewind on return with nothing dangling
        /// (byte-bump ↔ page-tier coexistence on one arena, as `writeToCold`
        /// already relies on).
        ///
        /// **CRITICAL — mint-on-HIT:** the flatten (mint) runs FIRST and
        /// UNCONDITIONALLY, then the id-compare. `encodeWorkInstruction` minted ALL
        /// fields before `findOrEmplace` ran, so the mint must happen even when the
        /// entry is a HIT — a probe-then-mint design would diverge the id table on
        /// hits and is banned.
        ///
        /// @param wi The working instruction (resident; spans valid).
        /// @param vi The owning LB's value interner (mint side).
        /// @return The entry index (hit or newly appended).
        /// @invariant Mints into @p vi in `encodeWorkInstruction`'s exact order,
        ///            unconditionally, before any id comparison.
        /// @see flattenWorkInstructionInto, findOrEmplaceFlattened,
        ///      ExecutionParameters::MAX_INSTRUCTION_ELEMENTS.
        int32_t findOrEmplaceWork(const WorkInstruction& wi, ValueInterner& vi) {
            const int32_t n = wi.entityCount();
            int32_t len = 2;   // entityCount header + markedGoal trailer
            for (int32_t i = 0; i < n; ++i) len += 5 + wi.elemCount(i);
            ScratchScope scope(*arena);
            int32_t* run = reinterpret_cast<int32_t*>(arena->resolve(
                arena->alloc(len * static_cast<int32_t>(sizeof(int32_t)),
                             static_cast<int32_t>(alignof(int32_t)))));
            int32_t pos = 0;
            run[pos++] = n;
            for (int32_t i = 0; i < n; ++i) {
                const int32_t catId = vi.encode(wi.category(i));
                const int32_t ec = wi.elemCount(i);
                assert(ec <= ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
                    && "ArenaIntegrationMap::findOrEmplaceWork: element count "
                       "exceeds MAX_INSTRUCTION_ELEMENTS");
                int32_t elemIds[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
                for (int32_t j = 0; j < ec; ++j) elemIds[j] = vi.encode(wi.elemAt(i, j));
                const int32_t sigId = vi.encode(wi.signature(i));
                const int32_t arity = wi.arity(i);
                const int32_t dsId = vi.encode(wi.definedSet(i));
                run[pos++] = catId;
                run[pos++] = sigId;
                run[pos++] = arity;
                run[pos++] = dsId;
                run[pos++] = ec;
                for (int32_t j = 0; j < ec; ++j) run[pos++] = elemIds[j];
            }
            run[pos++] = vi.encode(wi.markedGoal());
            assert(pos == len
                && "ArenaIntegrationMap::findOrEmplaceWork: fill length "
                   "!= computed length");
            return findOrEmplaceFlattened(run, len);
        }

        /// @brief Whether entry @p ei's value set contains @p v.
        /// @param ei Entry index. @param v Value id. @return Member?
        bool countValue(int32_t ei, int32_t v) const {
            int32_t node = entries[ei].valHead;
            while (node != -1) {
                if (valNodes[node].valueId == v) return true;
                node = valNodes[node].prev;
            }
            return false;
        }

        /// @brief Insert @p v into entry @p ei's value set (dedup) — set insert.
        /// @param ei Entry index. @param v Value id.
        void insertValue(int32_t ei, int32_t v) {
            if (countValue(ei, v)) return;
            pushValue(ei, v);
        }

        /// @brief Materialize entry @p ei's instruction (owned heap copy).
        /// @param ei Entry index. @return The `IntInstruction`.
        IntInstruction instructionAt(int32_t ei) const {
            const ArenaIntEntry& e = entries[ei];
            IntInstruction out;
            out.data.reserve(static_cast<std::size_t>(e.leCount));
            for (int32_t i = 0; i < e.leCount; ++i) {
                const ArenaIntLE& le = les[e.leStart + i];
                IntLogicalEntity ile;
                ile.category = le.catId; ile.signature = le.sigId;
                ile.definedSet = le.dsId; ile.arity = le.arity;
                ile.elements.reserve(static_cast<std::size_t>(le.elemsCount));
                for (int32_t j = 0; j < le.elemsCount; ++j)
                    ile.elements.push_back(elemCol[le.elemsStart + j]);
                out.data.push_back(std::move(ile));
            }
            out.markedGoal = e.markedGoalId;
            return out;
        }

        /// @brief Load stored entry @p ei straight into @p wi — the heap-free twin
        ///        of `loadFromIntInstruction(wi, instructionAt(ei), vi)`.
        ///
        /// @details
        /// Reads entry @p ei's arena records (`les[e.leStart + i]`, `elemCol`,
        /// `markedGoalId`) DIRECTLY and appends them to @p wi, with the intermediate
        /// heap `IntInstruction` `instructionAt` used to materialize inlined away:
        /// per entity, `wi.addElement(vi.decodeView(elemId))` for each element then
        /// `wi.commitEntity(vi.decodeView(catId), vi.decodeView(sigId),
        /// vi.decodeView(dsId), arity, mark)`, finally
        /// `wi.setMarkedGoal(vi.decodeView(markedGoalId))`.
        ///
        /// Mints ONLY into @p wi's own arena interner (`wi.strings`), never into
        /// @p vi (it READS @p vi via non-minting `decodeView`), so no observable id
        /// order is disturbed — `wi.strings` is a transient page-tier interner,
        /// never deloaded, so its mint order is unobservable, and the I-3
        /// copy-before-nested-mint discipline is moot (the decode source @p vi and
        /// the mint target `wi.strings` are DISTINCT interners). The decoded content
        /// is identical to the `loadFromIntInstruction` path because both read the
        /// same entry ids and `decodeView` the same bytes in the same structure.
        /// @p wi must be freshly constructed (the load appends).
        ///
        /// @param ei Entry index.
        /// @param wi The destination working instruction (empty; appended in place).
        /// @param vi The owning LB's value interner (read side).
        /// @invariant Mints only into @p wi's interner; @p vi is read-only.
        /// @see loadFromIntInstruction — the heap-`IntInstruction` twin.
        ///      @see instructionAt — the intermediate this inlines away.
        void loadInstructionInto(int32_t ei, WorkInstruction& wi,
                                 const ValueInterner& vi) const {
            const ArenaIntEntry& e = entries[ei];
            for (int32_t i = 0; i < e.leCount; ++i) {
                const ArenaIntLE& le = les[e.leStart + i];
                const int32_t start = wi.elemMark();
                for (int32_t j = 0; j < le.elemsCount; ++j)
                    wi.addElement(vi.decodeView(elemCol[le.elemsStart + j]));
                wi.commitEntity(vi.decodeView(le.catId), vi.decodeView(le.sigId),
                                vi.decodeView(le.dsId), le.arity, start);
            }
            wi.setMarkedGoal(vi.decodeView(e.markedGoalId));
        }

        /// @brief Entry @p ei's value set, `DecodedIdLess`-sorted (owned heap copy).
        /// @param ei Entry index. @return The sorted value ids.
        std::vector<int32_t> valuesAt(int32_t ei) const {
            std::vector<int32_t> vals;
            vals.reserve(static_cast<std::size_t>(entries[ei].valCount));
            int32_t node = entries[ei].valHead;
            while (node != -1) {
                vals.push_back(valNodes[node].valueId);
                node = valNodes[node].prev;
            }
            const ValueInterner* v = vi;
            std::sort(vals.begin(), vals.end(),
                [v](int32_t a, int32_t b) { return valueIdLess(a, b, *v); });
            return vals;
        }

        /// @brief Caller-fill twin of @ref valuesAt(int32_t) — writes entry
        ///        @p ei's `DecodedIdLess`-sorted value ids into @p out with no
        ///        heap `std::vector<int32_t>`.
        ///
        /// @details
        /// Byte-for-byte the same value sequence as the heap `valuesAt(ei)`: the
        /// value-chain walk collects the same ids, and the same `valueIdLess`
        /// `std::sort` imposes the identical `DecodedIdLess` order into the
        /// caller's stack buffer. The heap form survives as this twin's unit-test
        /// oracle (`arena_integration_values_at_fill_matches_vector`).
        ///
        /// @param ei  Entry index.
        /// @param out Caller-owned buffer receiving the sorted value ids.
        /// @param cap Capacity of @p out (`MAX_INTEGRATION_ENTRY_VALUES` at the
        ///            external call sites); a `count > cap` overrun is a loud
        ///            Rule-19 assert whose documented widening path is a page-tier
        ///            `PagedVector<int32_t>` run — never a silent clamp.
        /// @return The value count written into `out[0..count)`.
        /// @invariant `out[0..count)` equals `valuesAt(ei)` element-for-element.
        /// @see valuesAt(int32_t) (the retained heap oracle),
        ///      ExecutionParameters::MAX_INTEGRATION_ENTRY_VALUES.
        int32_t valuesAt(int32_t ei, int32_t* out, int32_t cap) const {
            int32_t count = 0;
            int32_t node = entries[ei].valHead;
            while (node != -1) {
                assert(count < cap
                    && "ArenaIntegrationMap::valuesAt: value count exceeds "
                       "MAX_INTEGRATION_ENTRY_VALUES");
                out[count++] = valNodes[node].valueId;
                node = valNodes[node].prev;
            }
            const ValueInterner* v = vi;
            std::sort(out, out + count,
                [v](int32_t a, int32_t b) { return valueIdLess(a, b, *v); });
            return count;
        }

        /// @brief `valueIdVectorLess` sign of two LE element runs (-1 / 0 / +1).
        /// @param a First LE. @param b Second LE. @return Sign of a.elements <=> b.elements.
        int compareElems(const ArenaIntLE& a, const ArenaIntLE& b) const {
            const int32_t n = a.elemsCount < b.elemsCount ? a.elemsCount : b.elemsCount;
            for (int32_t i = 0; i < n; ++i) {
                const int32_t ea = elemCol[a.elemsStart + i];
                const int32_t eb = elemCol[b.elemsStart + i];
                if (ea == eb) continue;
                return compareSpans(vi->decodeView(ea), vi->decodeView(eb)) < 0 ? -1 : 1;
            }
            if (a.elemsCount != b.elemsCount) return a.elemsCount < b.elemsCount ? -1 : 1;
            return 0;
        }

        /// @brief `DecodedInstructionLess` sign of entries @p ai, @p bi (-1 / 0 / +1).
        /// @param ai First entry. @param bi Second entry. @return Sign of ai <=> bi.
        int compareEntries(int32_t ai, int32_t bi) const {
            const ArenaIntEntry& a = entries[ai];
            const ArenaIntEntry& b = entries[bi];
            const int32_t n = a.leCount < b.leCount ? a.leCount : b.leCount;
            for (int32_t i = 0; i < n; ++i) {
                const ArenaIntLE& la = les[a.leStart + i];
                const ArenaIntLE& lb = les[b.leStart + i];
                if (la.catId != lb.catId)
                    return valueIdLess(la.catId, lb.catId, *vi) ? -1 : 1;
                if (la.sigId != lb.sigId)
                    return valueIdLess(la.sigId, lb.sigId, *vi) ? -1 : 1;
                const int ce = compareElems(la, lb);
                if (ce != 0) return ce;
                if (la.arity != lb.arity) return la.arity < lb.arity ? -1 : 1;
                if (la.dsId != lb.dsId)
                    return valueIdLess(la.dsId, lb.dsId, *vi) ? -1 : 1;
            }
            if (a.leCount != b.leCount) return a.leCount < b.leCount ? -1 : 1;
            if (a.markedGoalId == b.markedGoalId) return 0;
            return valueIdLess(a.markedGoalId, b.markedGoalId, *vi) ? -1 : 1;
        }

        /// @brief Entry indices in `DecodedInstructionLess` order — the snapshot /
        ///        flatten order (owned, tiny: one int per entry).
        /// @return The sorted entry-index list.
        std::vector<int32_t> sortedIndices() const {
            std::vector<int32_t> idx(static_cast<std::size_t>(entries.size()));
            for (int32_t i = 0; i < entries.size(); ++i)
                idx[static_cast<std::size_t>(i)] = i;
            std::sort(idx.begin(), idx.end(),
                [this](int32_t a, int32_t b) { return compareEntries(a, b) < 0; });
            return idx;
        }

        /// @brief Caller-fill twin of @ref sortedIndices() — writes the entry
        ///        indices in `DecodedInstructionLess` order into @p out with no
        ///        heap `std::vector<int32_t>`.
        ///
        /// @details
        /// Byte-for-byte the same permutation as the heap `sortedIndices()`:
        /// `out[k]=k` seed then the identical `compareEntries` `std::sort`. The
        /// heap form survives as this twin's unit-test oracle
        /// (`arena_integration_sorted_indices_fill_matches_vector`) and as
        /// `flatten()`'s internal reference path.
        ///
        /// @param out Caller-owned buffer receiving the sorted entry indices.
        /// @param cap Capacity of @p out (`MAX_INTEGRATION_ENTRIES` at the
        ///            external call sites); an `entries.size() > cap` overrun is a
        ///            loud Rule-19 assert whose documented widening path is a
        ///            page-tier `PagedVector<int32_t>` index — never a silent
        ///            clamp or truncated sort.
        /// @return The entry count written into `out[0..count)`.
        /// @invariant `out[0..count)` equals `sortedIndices()` element-for-element.
        /// @see sortedIndices() (the retained heap oracle),
        ///      ExecutionParameters::MAX_INTEGRATION_ENTRIES.
        int32_t sortedIndices(int32_t* out, int32_t cap) const {
            const int32_t n = entries.size();
            assert(n <= cap
                && "ArenaIntegrationMap::sortedIndices: entry count exceeds "
                   "MAX_INTEGRATION_ENTRIES");
            for (int32_t i = 0; i < n; ++i) out[i] = i;
            std::sort(out, out + n,
                [this](int32_t a, int32_t b) { return compareEntries(a, b) < 0; });
            return n;
        }

        /// @brief Flatten to a sorted `IntegrationEntry` run — the heap-form
        ///        reference path, byte-identical to `flattenIntegrationEntryMap`.
        ///
        /// @details Retained as the unit test's golden reference for the write
        /// direction (and the semantic definition `writeToCold` is checked
        /// byte-identical against); production writes go through `writeToCold`, which
        /// serializes straight to the blob bytes without this `std::vector<
        /// IntegrationEntry>`.
        ///
        /// @return The record run, in `DecodedInstructionLess` order with each value
        ///         run in `DecodedIdLess` order.
        /// @see writeToCold (the direct production write).
        std::vector<IntegrationEntry> flatten() const {
            const std::vector<int32_t> idx = sortedIndices();
            std::vector<IntegrationEntry> run;
            run.reserve(idx.size());
            for (const int32_t ei : idx) {
                IntegrationEntry e;
                e.instruction = instructionAt(ei);
                e.valueIds = valuesAt(ei);
                run.push_back(std::move(e));
            }
            return run;
        }

        /// @brief Serialize the arena form directly into `m[pk]`'s cold run — emit
        ///        each entry's canonical bytes straight to the blob store with NO
        ///        intermediate heap `IntegrationEntry` / `flatten()` vector.
        ///
        /// @details
        /// Walks the entries in `DecodedInstructionLess` order (`sortedIndices`) and,
        /// per entry, appends its `Codec<IntegrationEntry>`-layout bytes (`markedGoal`,
        /// entity count, then per entity `category` / `signature` / `arity` /
        /// `definedSet` + its element run, then the `DecodedIdLess`-sorted value run
        /// from `valuesAt`) into a flat byte buffer + per-record length buffer, then
        /// replaces the key's whole run in one engine `assignRun`. The produced bytes
        /// are byte-identical to `m.assignRun(pk, flatten())` — same record order,
        /// same per-record layout — but no `std::vector<IntegrationEntry>` is ever
        /// materialized.
        ///
        /// The flat `bytes` / `lens` buffers ARE the sanctioned cold boundary (the
        /// exact buffers the typed `assignRun` builds internally); only the
        /// structured `IntegrationEntry` heap is removed. The engine's raw
        /// `assignRun(KeyView, bytes, lens, M)` is reached through the typed
        /// wrapper's `inner()` escape hatch — `Codec<int32_t>::view(pk) == pk`, so
        /// the call is identical to the typed `assignRun`'s own forwarding.
        ///
        /// The `bytes` / `lens` staging is byte-bump on the map's OWN `*arena`
        /// (the exact arena `buildFromCold` already byte-bumps for peek-straddle
        /// spills, so byte-bump ↔ page-tier coexistence is proven for this map)
        /// under a function-scoped `ScratchScope` — no heap `std::vector<char>` /
        /// `std::vector<int32_t>` and no heap `valuesAt(ei)` (the `(out, cap)`
        /// twin fills a reused arena buffer). Pass 1 sums each record's exact
        /// length (byte-identical to the former `Codec` field walk) so the whole
        /// run is one contiguous `alloc`; the raw `assignRun` COPIES it into the
        /// map's own blob pool (a different arena) so the scope may rewind on
        /// return with nothing dangling.
        ///
        /// @param m  The cold admission-integration blob map (written in place).
        /// @param pk The packed (templateId, validityId) key.
        /// @invariant Writes the EXACT field order `Codec<IntegrationEntry>::serialize`
        ///            writes; the unit test enforces byte-identity to
        ///            `assignRun(flatten())`.
        /// @see buildFromCold (the read half), flatten, Codec<IntegrationEntry>,
        ///      ExecutionParameters::kMaxIntegrationRunBytes.
        void writeToCold(TypedColdBlobMap<int32_t, IntegrationEntry>& m,
                         int32_t pk) const {
            int32_t idx[ExecutionParameters::MAX_INTEGRATION_ENTRIES];
            const int32_t idxN =
                sortedIndices(idx, ExecutionParameters::MAX_INTEGRATION_ENTRIES);
            if (idxN == 0) {
                m.inner().assignRun(pk, nullptr, nullptr, 0);
                return;
            }
            ScratchScope scope(*arena);

            // Pass 1 — per-record length + total, on the arena byte-bump tier.
            // Each record's length is the exact byte count the field walk below
            // emits: 8 (markedGoalId + leCount) + per-entity (20 header ints +
            // 4*elemsCount) + 4 (valCount header) + 4*valCount. `valCount` is the
            // deduped value-chain length `valuesAt` walks (insertValue dedups
            // before pushValue), so it equals pass 2's emitted value count.
            int32_t* lens = reinterpret_cast<int32_t*>(arena->resolve(
                arena->alloc(idxN * static_cast<int32_t>(sizeof(int32_t)),
                             static_cast<int32_t>(alignof(int32_t)))));
            int64_t total = 0;
            for (int32_t ii = 0; ii < idxN; ++ii) {
                const ArenaIntEntry& e = entries[idx[ii]];
                int32_t rl = 8;
                for (int32_t i = 0; i < e.leCount; ++i)
                    rl += 20 + 4 * les[e.leStart + i].elemsCount;
                rl += 4 + 4 * e.valCount;
                lens[ii] = rl;
                total += rl;
            }
            assert(total <= ExecutionParameters::kMaxIntegrationRunBytes
                && "ArenaIntegrationMap::writeToCold: run exceeds one pool block "
                   "(kMaxIntegrationRunBytes) — widen via chunked appendBlobToRun");
            char* outBytes =
                arena->resolve(arena->alloc(static_cast<int32_t>(total), 1));
            int32_t* valsBuf = reinterpret_cast<int32_t*>(arena->resolve(
                arena->alloc(ExecutionParameters::MAX_INTEGRATION_ENTRY_VALUES
                             * static_cast<int32_t>(sizeof(int32_t)),
                             static_cast<int32_t>(alignof(int32_t)))));

            // Pass 2 — fill outBytes in the exact former put() field order.
            int32_t off = 0;
            const auto put = [&outBytes, &off](int32_t x) {
                std::memcpy(outBytes + off, &x, sizeof(int32_t));
                off += static_cast<int32_t>(sizeof(int32_t));
            };
            for (int32_t ii = 0; ii < idxN; ++ii) {
                const int32_t ei = idx[ii];
                const int32_t before = off;
                const ArenaIntEntry& e = entries[ei];
                put(e.markedGoalId);
                put(e.leCount);
                for (int32_t i = 0; i < e.leCount; ++i) {
                    const ArenaIntLE& le = les[e.leStart + i];
                    put(le.catId);
                    put(le.sigId);
                    put(le.arity);
                    put(le.dsId);
                    put(le.elemsCount);
                    for (int32_t ej = 0; ej < le.elemsCount; ++ej)
                        put(elemCol[le.elemsStart + ej]);
                }
                const int32_t vn = valuesAt(
                    ei, valsBuf, ExecutionParameters::MAX_INTEGRATION_ENTRY_VALUES);
                put(vn);
                for (int32_t k = 0; k < vn; ++k) put(valsBuf[k]);
                assert(off - before == lens[ii]
                    && "ArenaIntegrationMap::writeToCold: pass-2 length "
                       "!= pass-1 length");
            }
            assert(off == static_cast<int32_t>(total));
            m.inner().assignRun(pk, outBytes, lens, idxN);
        }
    };

    /// @brief Out-of-line `ChangedClassesBuffer::push` — declared in
    ///        `memory_infra/changed_classes_buffer.hpp`, defined here so
    ///        `EquivalenceClass` + `serializeEquivalenceClass` are complete.
    /// @param vid The class's validity id.
    /// @param cls The merged class to snapshot (serialized to a blob).
    inline void ChangedClassesBuffer::push(int16_t vid,
        const EquivalenceClass& cls) {
        const std::vector<char> blob = serializeEquivalenceClass(cls);
        validityIds_.push_back(vid);
        blobStarts_.push_back(blobPool_.size());
        if (!blob.empty())
            blobPool_.appendRun(blob.data(),
                                static_cast<int32_t>(blob.size()));
    }

    /// @brief Out-of-line `ChangedClassesBuffer::classAt` — declared in
    ///        `memory_infra/changed_classes_buffer.hpp`, defined here so
    ///        `EquivalenceClass` + `deserializeEquivalenceClass` are complete.
    /// @param i Index in `[0, size())`.
    /// @return The reconstructed class.
    inline EquivalenceClass ChangedClassesBuffer::classAt(int32_t i) const {
        const int32_t start = blobStarts_[i];
        const int32_t end = (i + 1 < blobStarts_.size())
            ? blobStarts_[i + 1] : blobPool_.size();
        std::vector<char> blob(static_cast<std::size_t>(end - start));
        int32_t pos = start, off = 0;
        while (pos < end) {
            int32_t run = 0;
            const char* p = blobPool_.contiguousRun(pos, run);
            if (run > end - pos) run = end - pos;
            std::memcpy(blob.data() + off, p, static_cast<std::size_t>(run));
            pos += run; off += run;
        }
        return deserializeEquivalenceClass(
            blob.data(), static_cast<int32_t>(blob.size()));
    }

    /// @brief Zero-copy `EquivalenceClassView` over changed-class entry @p i's
    ///        blob — the `classAt` twin that avoids the heap `EquivalenceClass`
    ///        decode on the equi-class apply read path (equi-7).
    ///
    /// @details A contiguous blob (the common case) yields a view pointing
    /// straight into the buffer's cold pool; a page-straddling blob is copied
    /// once into @p scratch (the `peekBlobContiguous` pattern). The returned view
    /// is valid while the buffer stays resident and, on a straddle, while
    /// @p scratch outlives the view. Read-only.
    ///
    /// @param i       Index in `[0, size())`.
    /// @param scratch Caller-owned reuse buffer, written only on a page straddle;
    ///                left untouched on the contiguous case.
    /// @return A view over the entry's canonical class blob.
    /// @see classAt — the heap-decoding counterpart.
    inline EquivalenceClassView ChangedClassesBuffer::classViewAt(
        int32_t i, std::vector<char>& scratch) const {
        const int32_t start = blobStarts_[i];
        const int32_t end = (i + 1 < blobStarts_.size())
            ? blobStarts_[i + 1] : blobPool_.size();
        const int32_t len = end - start;
        int32_t run = 0;
        const char* p = blobPool_.contiguousRun(start, run);
        if (run >= len) {
            return EquivalenceClassView{ p, len };
        }
        scratch.resize(static_cast<std::size_t>(len));
        int32_t pos = start, off = 0;
        while (pos < end) {
            int32_t r = 0;
            const char* q = blobPool_.contiguousRun(pos, r);
            if (r > end - pos) r = end - pos;
            std::memcpy(scratch.data() + off, q, static_cast<std::size_t>(r));
            pos += r; off += r;
        }
        return EquivalenceClassView{ scratch.data(), len };
    }

    /// @brief Assemble entry `i`'s class snapshot as a zero-copy view over the
    ///        paged blob pool, or (on a page straddle) onto @p scratch's
    ///        byte-bump tier — the arena-backed twin of the `std::vector<char>`
    ///        `classViewAt` overload.
    ///
    /// @details Byte-for-byte the vector overload's logic with the straddle sink
    /// swapped: the single-page contiguous run is returned in place (arena
    /// untouched); a straddle is copied once into a FRESH
    /// `scratch.resolve(scratch.alloc(len, 1))` run. NO rewind, so a re-entrant
    /// apply that peeks never clobbers an outer view — the equi-class apply's
    /// recursion-safe accumulation pattern, matching the `ScratchArena&` overload
    /// of `HashMap::peekBlobContiguous`. `len` is the exact blob byte length, so
    /// the assembled bytes equal the vector path's.
    ///
    /// @param i       Index in `[0, size())`.
    /// @param scratch Byte-bump arena the straddle copy is assembled onto; left
    ///                untouched on the contiguous case.
    /// @return A view over `len` contiguous canonical class blob bytes.
    /// @see classViewAt(int32_t, std::vector<char>&) const — the vector twin.
    inline EquivalenceClassView ChangedClassesBuffer::classViewAt(
        int32_t i, ScratchArena& scratch) const {
        const int32_t start = blobStarts_[i];
        const int32_t end = (i + 1 < blobStarts_.size())
            ? blobStarts_[i + 1] : blobPool_.size();
        const int32_t len = end - start;
        int32_t run = 0;
        const char* p = blobPool_.contiguousRun(start, run);
        if (run >= len) {
            return EquivalenceClassView{ p, len };
        }
        char* buf = scratch.resolve(scratch.alloc(len, 1));
        int32_t pos = start, off = 0;
        while (pos < end) {
            int32_t r = 0;
            const char* q = blobPool_.contiguousRun(pos, r);
            if (r > end - pos) r = end - pos;
            std::memcpy(buf + off, q, static_cast<std::size_t>(r));
            pos += r; off += r;
        }
        return EquivalenceClassView{ buf, len };
    }

    /// @brief Decode a routing mailIn's id-form statements (GLOBAL mailInterner
    ///        ids) into the canonical-sorted string snapshot.
    ///
    /// @details The absorb + `routingMailInToHeap` read boundary: mailIn holds
    /// global ids (the pull copies them from the blob); decode via the global
    /// `mailInterner` and sort by `pair<EWV, set<int>>::operator<`.
    ///
    /// @param mi The routing inbox (`mailIn`).
    /// @return The statements `(expression+scope, levels)`, key-sorted.
    inline std::vector<std::pair<ExpressionWithValidity, std::set<int>>>
    decodeMailInStatements(const RoutingColdMail& mi) {
        std::vector<std::pair<ExpressionWithValidity, std::set<int>>> out;
        if (!mi.arenaInited_) return out;
        out.reserve(static_cast<size_t>(mi.statements_.count()));
        for (int32_t id = 1; id <= mi.statements_.count(); ++id) {
            const IntMailStatementKey k = mi.statements_.decodeKey(id);
            out.emplace_back(
                ExpressionWithValidity(mailInterner().decodeString(k.originalId),
                                       mailInterner().decodeString(k.validityId)),
                std::set<int>(k.levels.begin(), k.levels.end()));
        }
        std::sort(out.begin(), out.end());
        return out;
    }

    /// @brief Decode a deloadable mailOut's private-id statements
    ///        into the canonical-sorted string snapshot.
    ///
    /// @param mo The routing outbox (`mailOut`).
    /// @param mi The producing LB's dedicated `mailOutInterner`.
    /// @return The statements `(expression+scope, levels)`, key-sorted.
    inline std::vector<std::pair<ExpressionWithValidity, std::set<int>>>
    decodeMailOutStatements(const DeloadableMailOut& mo,
        const ValueInterner& mi) {
        std::vector<std::pair<ExpressionWithValidity, std::set<int>>> out;
        out.reserve(static_cast<size_t>(mo.statements_.count()));
        for (int32_t id = 1; id <= mo.statements_.count(); ++id) {
            const IntMailStatementKey k = mo.statements_.decodeKey(id);
            out.emplace_back(
                ExpressionWithValidity(
                    mi.decode(k.originalId), mi.decode(k.validityId)),
                std::set<int>(k.levels.begin(), k.levels.end()));
        }
        std::sort(out.begin(), out.end());
        return out;
    }

    /// @brief Decode a deloadable mailOut's private-id origins into the
    ///        canonical-sorted string snapshot.
    ///
    /// @param mo The routing outbox.
    /// @param mi The producing LB's dedicated `mailOutInterner`.
    /// @return The origin rows `(expression+scope, history lines)`, key-sorted.
    inline std::vector<std::pair<ExpressionWithValidity, std::vector<OriginLine>>>
    decodeMailOutOrigins(const DeloadableMailOut& mo, const ValueInterner& mi) {
        std::vector<std::pair<ExpressionWithValidity,
            std::vector<OriginLine>>> out;
        out.reserve(static_cast<size_t>(mo.origins_.count()));
        for (int32_t id = 1; id <= mo.origins_.count(); ++id) {
            const int64_t key = mo.origins_.decodeKey(id);
            StrSpan eSpan, vSpan;
            decodeOriginKeyView(key, mi, eSpan, vSpan);
            std::pair<std::string, std::string> ev(eSpan.toStdString(), vSpan.toStdString());
            const std::vector<IntMailOrigin> recs = mo.origins_.recordsAt(id);
            std::vector<OriginLine> lines;
            lines.reserve(recs.size());
            for (const IntMailOrigin& r : recs)
                lines.push_back(decodeOrigin(
                    IdOrigin{ static_cast<OriginTag>(r.tag), r.deps }, mi));
            out.emplace_back(
                ExpressionWithValidity(std::move(ev.first), std::move(ev.second)),
                std::move(lines));
        }
        std::sort(out.begin(), out.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
        return out;
    }

    /// @brief Decode a routing mailIn's id-form origins (GLOBAL mailInterner ids)
    ///        into the canonical-sorted string snapshot.
    ///
    /// @param mi The routing inbox.
    /// @return The origin rows `(expression+scope, history lines)`, key-sorted.
    inline std::vector<std::pair<ExpressionWithValidity, std::vector<OriginLine>>>
    decodeMailInOrigins(const RoutingColdMail& mi) {
        std::vector<std::pair<ExpressionWithValidity,
            std::vector<OriginLine>>> out;
        if (!mi.arenaInited_) return out;
        out.reserve(static_cast<size_t>(mi.origins_.count()));
        const auto unpackEwv = [](int64_t pk) {
            const int32_t o =
                static_cast<int32_t>(static_cast<uint64_t>(pk) >> 32);
            const int32_t v = static_cast<int32_t>(pk & 0xFFFFFFFFLL);
            return ExpressionWithValidity(mailInterner().decodeString(o),
                                          mailInterner().decodeString(v));
        };
        for (int32_t id = 1; id <= mi.origins_.count(); ++id) {
            const int64_t key = mi.origins_.decodeKey(id);
            const std::vector<IntMailOrigin> recs = mi.origins_.recordsAt(id);
            std::vector<OriginLine> lines;
            lines.reserve(recs.size());
            for (const IntMailOrigin& r : recs) {
                OriginLine line;
                line.first = originTagName(static_cast<OriginTag>(r.tag));
                line.second.reserve(r.deps.size());
                for (const int64_t dpk : r.deps)
                    line.second.push_back(unpackEwv(dpk));
                lines.push_back(std::move(line));
            }
            out.emplace_back(unpackEwv(key), std::move(lines));
        }
        std::sort(out.begin(), out.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
        return out;
    }

    /// @brief Materialize one deloadable `mailOut` as a heap `Mail`.
    ///
    /// @param mo The routing outbox.
    /// @param mi The producing LB's dedicated `mailOutInterner`.
    /// @return The equivalent heap `Mail`.
    inline Mail routingMailOutToHeap(const DeloadableMailOut& mo,
        const ValueInterner& mi) {
        Mail m;
        for (const std::pair<ExpressionWithValidity, std::set<int>>& st
             : decodeMailOutStatements(mo, mi))
            m.statements.insert(st);
        for (std::pair<ExpressionWithValidity, std::vector<OriginLine>>& og
             : decodeMailOutOrigins(mo, mi))
            m.exprOriginMap.emplace(og.first, std::move(og.second));
        return m;
    }

    /// @brief Materialize the heap `Mail` of a routing `mailIn` (GLOBAL ids) —
    ///        the dump / test boundary (statements decoded via `mailInterner`;
    ///        origins still string).
    ///
    /// @param mi The routing inbox.
    /// @return The equivalent heap `Mail`.
    inline Mail routingMailInToHeap(const RoutingColdMail& mi) {
        Mail m;
        for (const std::pair<ExpressionWithValidity, std::set<int>>& st
             : decodeMailInStatements(mi))
            m.statements.insert(st);
        for (std::pair<ExpressionWithValidity, std::vector<OriginLine>>& og
             : decodeMailInOrigins(mi))
            m.exprOriginMap.emplace(og.first, std::move(og.second));
        return m;
    }

    /// @brief Out-of-line `Codec<Mail>::serialize(const DeloadableMailOut&)`.
    inline std::vector<char> Codec<Mail>::serialize(const DeloadableMailOut& hm,
        const ValueInterner& senderMi) {
        std::vector<char> out;
        const auto putPod = [&out](const auto& x) {
            const char* p = reinterpret_cast<const char*>(&x);
            out.insert(out.end(), p, p + sizeof(x));
        };
        // Every mail EWV travels as two GLOBAL mailInterner ids (the cross-LB
        // carrier space). Single-threaded commit seam, so the mint is race-free.
        const auto putGlobalEwv = [&putPod](const ExpressionWithValidity& e) {
            putPod(mailInterner().intern(e.original));
            putPod(mailInterner().intern(e.validityName));
        };
        // statements: decode private mailbox ids -> strings (canonical sorted),
        // then emit as global ids. Byte-identical to
        // serialize(routingMailOutToHeap(hm, senderMi)).
        const std::vector<std::pair<ExpressionWithValidity, std::set<int>>>
            stmts = decodeMailOutStatements(hm, senderMi);
        putPod(static_cast<int32_t>(stmts.size()));
        for (const std::pair<ExpressionWithValidity, std::set<int>>& st : stmts) {
            putGlobalEwv(st.first);
            putPod(static_cast<int32_t>(st.second.size()));
            for (const int lv : st.second) putPod(static_cast<int32_t>(lv));
        }
        // exprOriginMap: decode private mailbox ids -> strings, emit
        // each EWV / dep as global ids and the tag as its OriginTag byte.
        const std::vector<std::pair<ExpressionWithValidity,
            std::vector<OriginLine>>> origins = decodeMailOutOrigins(hm, senderMi);
        putPod(static_cast<int32_t>(origins.size()));
        for (const std::pair<ExpressionWithValidity,
                 std::vector<OriginLine>>& keyed : origins) {
            putGlobalEwv(keyed.first);
            putPod(static_cast<int32_t>(keyed.second.size()));
            for (const OriginLine& origin : keyed.second) {
                putPod(static_cast<int32_t>(
                    originTagFromString(StrSpan(origin.first))));
                putPod(static_cast<int32_t>(origin.second.size()));
                for (const ExpressionWithValidity& dep : origin.second)
                    putGlobalEwv(dep);
            }
        }
        return out;
    }

    /// @brief A contiguous sequential POD source over a raw `Codec<Mail>` blob —
    ///        the trivial in-memory twin of `PoolMailSource`.
    ///
    /// @details
    /// Wraps a `[cur, end)` byte window and reads fixed-width little-endian PODs
    /// forward with `memcpy` (never a `reinterpret_cast` of the cursor, so an
    /// unaligned field is safe). It is the source that
    /// `Codec<Mail>::deserializeInto(const char*, ...)` feeds to the shared parse
    /// body `deserializeMailBlobInto`; a blob shorter than the layout trips the
    /// loud bounds `assert` inside `getPod` (Rule 19), and `atEnd()` lets the parse
    /// body assert the blob was not longer than the layout.
    ///
    /// @invariant The window aliases caller memory that outlives the parse; this
    ///            source owns nothing.
    /// @see PoolMailSource — the page-straddling pool twin; deserializeMailBlobInto
    ///      — the shared parse body both sources drive.
    struct CharMailSource {
        /// @brief The read cursor (advances by `sizeof(T)` per `getPod`).
        const char* cur;
        /// @brief One past the last readable byte.
        const char* end;

        /// @brief Read one fixed-width POD forward, advancing the cursor.
        ///
        /// @details Copies `sizeof(T)` bytes with `memcpy` (unaligned-safe) after
        /// asserting the window still holds them (Rule 19 — a short blob is a bug,
        /// never a truncated decode).
        ///
        /// @tparam T A trivially-copyable POD (the blob's `int32_t` fields).
        /// @param v Destination, filled from the cursor.
        template <class T>
        void getPod(T& v) {
            assert(cur + sizeof(T) <= end
                && "CharMailSource::getPod: blob shorter than the layout");
            std::memcpy(&v, cur, sizeof(T));
            cur += sizeof(T);
        }

        /// @brief Whether the cursor has consumed the whole window.
        /// @return `true` when `cur == end` (the parse-complete check).
        bool atEnd() const { return cur == end; }
    };

    /// @brief A straddle-aware sequential POD source over a `Codec<Mail>` blob in
    ///        the mail blob pool — the production twin of `CharMailSource`.
    ///
    /// @details
    /// `MailLog::mailBlobPool` is a `PagedVector<char>` whose physical pages may
    /// split a blob, so this source reads each fixed-width field forward crossing
    /// pages transparently: it fetches the contiguous run at the cursor
    /// (`PagedVector::contiguousRun`) and, when that run holds the whole field,
    /// `memcpy`s it in place (no copy of the surrounding bytes); when a field
    /// straddles a page boundary it assembles the field byte-run-by-byte across
    /// consecutive runs into the destination — NEVER a `reinterpret_cast` of a
    /// straddling pool pointer, so an unaligned field is safe. It lets
    /// `Codec<Mail>::deserializeInto(const PagedVector<char>&, ...)` decode
    /// pool-native, with no `std::vector<char>` reassembly buffer.
    ///
    /// The pool is on the never-deloaded mail pool and is FROZEN during the
    /// parallel phase-1 pull ([I-94]), so the `contiguousRun` pointers stay valid
    /// for the whole read; a blob shorter than the layout trips the loud bounds
    /// `assert` in `getPod` (Rule 19), and `atEnd()` closes the parse.
    ///
    /// @invariant The pool outlives the parse and is not compacted/deloaded during
    ///            it ([I-94]); this source owns nothing.
    /// @see CharMailSource — the contiguous twin; deserializeMailBlobInto — the
    ///      shared parse body; PagedVector::contiguousRun.
    struct PoolMailSource {
        /// @brief The blob pool being read (aliased, not owned).
        const PagedVector<char>& pool;
        /// @brief The read cursor (a byte offset into `pool`).
        int32_t cur;
        /// @brief One past the blob's last byte (a byte offset into `pool`).
        int32_t end;

        /// @brief Read one fixed-width POD forward, crossing page boundaries.
        ///
        /// @details Asserts the window still holds `sizeof(T)` bytes (Rule 19),
        /// then: if the contiguous run at the cursor is long enough, one `memcpy`
        /// straight from the pool; else assemble the field byte-run-by-byte across
        /// consecutive runs (the straddle path — unaligned-safe, no pointer cast).
        ///
        /// @tparam T A trivially-copyable POD (the blob's `int32_t` fields).
        /// @param v Destination, filled from the pool at the cursor.
        template <class T>
        void getPod(T& v) {
            assert(cur + static_cast<int32_t>(sizeof(T)) <= end
                && "PoolMailSource::getPod: blob shorter than the layout");
            int32_t runLen = 0;
            const char* p = pool.contiguousRun(cur, runLen);
            if (runLen >= static_cast<int32_t>(sizeof(T))) {
                std::memcpy(&v, p, sizeof(T));
            } else {
                // The field straddles a physical page boundary: assemble it
                // byte-run-by-byte across consecutive contiguous runs (never a
                // cast of a straddling pool pointer — unaligned-safe).
                char* dst = reinterpret_cast<char*>(&v);
                int32_t filled = 0;
                int32_t at = cur;
                while (filled < static_cast<int32_t>(sizeof(T))) {
                    int32_t rl = 0;
                    const char* q = pool.contiguousRun(at, rl);
                    const int32_t need =
                        static_cast<int32_t>(sizeof(T)) - filled;
                    const int32_t take = (rl < need) ? rl : need;
                    std::memcpy(dst + filled, q,
                                static_cast<std::size_t>(take));
                    filled += take;
                    at += take;
                }
            }
            cur += static_cast<int32_t>(sizeof(T));
        }

        /// @brief Whether the cursor has consumed the whole blob window.
        /// @return `true` when `cur == end` (the parse-complete check).
        bool atEnd() const { return cur == end; }
    };

    /// @brief The one shared parse body of the frozen `Codec<Mail>` blob layout,
    ///        decoding straight into a `RoutingColdMail` inbox's write doors — 0%
    ///        heap, no transient `Mail` and no `IntMailOrigin`.
    ///
    /// @details
    /// Walks the frozen wire layout `Codec<Mail>::serialize` produces (statements
    /// loop, then origins loop, in that field order) reading every field through
    /// the sequential source @p src, and feeds the id-form `RoutingColdMail` doors:
    /// each statement via `RoutingColdMail::insertStatement(originalId, validityId,
    /// levels, levelCount)` from a stack level run, each origin via the POD
    /// `addMailOriginRecord(origins_, key, tag, deps, depCount, INT_MAX)` from a
    /// stack dependency run — so NO owning `IntMailOrigin` is built (byte-identical
    /// to the `IntMailOrigin` overload by the twin
    /// `add_mail_origin_record_pod_matches_intmailorigin`). Templating on the
    /// source lets the contiguous `CharMailSource` (the retained char* oracle) and
    /// the page-straddling `PoolMailSource` (the production sink) share one body —
    /// the campaign's single-source-of-truth discipline (Rule 18).
    ///
    /// `inbox.ensureArena()` is the FIRST statement, before EITHER loop: an
    /// origins-only blob (`stCount == 0`) runs no `insertStatement` to bind the
    /// arena, so the origins loop's `addMailOriginRecord` would otherwise write
    /// through an unbound arena — the G-55 access violation. The `depCount` bound
    /// is asserted BEFORE the stack fill (the door's own `kMaxOriginDeps` assert
    /// fires only after it receives the run, so this earlier tripwire is
    /// load-bearing), and `src.atEnd()` closes the parse (the blob-longer-than-the-
    /// layout tripwire). Statements + origins travel as GLOBAL `mailInterner` ids
    /// and are stored verbatim; the receiver's absorb decodes them — this body
    /// mints into NO string interner, so it is race-free under the parallel pull
    /// ([I-127]).
    ///
    /// @tparam Src A sequential POD source (`CharMailSource` / `PoolMailSource`)
    ///             exposing `getPod<T>(T&)` and `atEnd()`.
    /// @param src   The blob source, consumed forward.
    /// @param inbox The receiver's `mailIn` (a `RoutingColdMail`), folded in place.
    /// @invariant Mint-free into every string interner — see @details ([I-127]).
    /// @see Codec<Mail>::serialize — the frozen layout this reverses;
    ///      addMailOriginRecord — the POD origin door; CharMailSource;
    ///      PoolMailSource.
    template <class Src>
    inline void deserializeMailBlobInto(Src& src, RoutingColdMail& inbox) {
        // Bind the inbox's mail arena before ANY insert (G-55): an origins-only
        // blob runs no insertStatement, and the origins loop writes directly, so
        // an unbound arena would be an access violation, not an assert.
        inbox.ensureArena();
        int32_t stCount = 0; src.getPod(stCount);
        for (int32_t i = 0; i < stCount; ++i) {
            // statements travel as two GLOBAL mailInterner ids; mailIn stores them
            // verbatim (the absorb decodes via the global interner).
            int32_t origId = 0; src.getPod(origId);
            int32_t validId = 0; src.getPod(validId);
            int32_t lvCount = 0; src.getPod(lvCount);
            // Blob order is ascending-unique (the serializers walk a
            // std::set<int> or an already-ascending decoded snapshot), so the
            // stack run reproduces the set's iteration byte-for-byte.
            int lvRun[256];
            assert(lvCount <= 256
                && "mail blob level run exceeds the decode buffer");
            for (int32_t j = 0; j < lvCount; ++j) {
                int32_t lv = 0; src.getPod(lv); lvRun[j] = lv;
            }
            inbox.insertStatement(origId, validId, lvRun, lvCount);
        }
        int32_t keyCount = 0; src.getPod(keyCount);
        for (int32_t k = 0; k < keyCount; ++k) {
            // origins travel as GLOBAL ids; mailIn stores them verbatim (the absorb
            // decodes via the global interner). Key + each dep are two ids; the tag
            // is its OriginTag byte (written as int32).
            int32_t keyOrigId = 0; src.getPod(keyOrigId);
            int32_t keyValidId = 0; src.getPod(keyValidId);
            const int64_t key = packOriginKey(keyOrigId, keyValidId);
            int32_t oCount = 0; src.getPod(oCount);
            for (int32_t o = 0; o < oCount; ++o) {
                int32_t tag = 0; src.getPod(tag);
                int32_t depCount = 0; src.getPod(depCount);
                // Assert the dep count BEFORE the stack fill: the POD door's own
                // kMaxOriginDeps assert fires only after it receives the run, so
                // this earlier tripwire is load-bearing (Rule 19, widen-on-STOP).
                assert(depCount <= ExecutionParameters::kMaxOriginDeps
                    && "mail blob origin dep count exceeds kMaxOriginDeps "
                       "(Rule-19 tripwire — widen the constant with evidence)");
                int64_t d[ExecutionParameters::kMaxOriginDeps];
                for (int32_t dd = 0; dd < depCount; ++dd) {
                    int32_t depOrigId = 0; src.getPod(depOrigId);
                    int32_t depValidId = 0; src.getPod(depValidId);
                    d[dd] = packOriginKey(depOrigId, depValidId);
                }
                addMailOriginRecord(inbox.origins_, key,
                    static_cast<uint8_t>(tag), d, depCount,
                    (std::numeric_limits<int>::max)());
            }
        }
        assert(src.atEnd()
            && "deserializeMailBlobInto: blob longer than the layout");
    }

    /// @brief Out-of-line `Codec<Mail>::deserializeInto(const char*, ...)` — the
    ///        retained char* oracle, now a thin 0%-heap wrapper over the shared
    ///        parse body `deserializeMailBlobInto` via a `CharMailSource`. Declared
    ///        in `Codec<Mail>`, defined here so `RoutingColdMail` + the POD origin
    ///        door are complete.
    inline void Codec<Mail>::deserializeInto(const char* data, int32_t n,
        RoutingColdMail& inbox) {
        CharMailSource src{ data, data + n };
        deserializeMailBlobInto(src, inbox);
    }

    /// @brief Out-of-line `Codec<Mail>::deserializeInto(const PagedVector<char>&,
    ///        ...)` — the production pool-native sink, a thin 0%-heap wrapper over
    ///        the shared parse body via a `PoolMailSource`. Declared in
    ///        `Codec<Mail>`, defined here so `RoutingColdMail`, the POD origin
    ///        door, and `PoolMailSource` are complete.
    inline void Codec<Mail>::deserializeInto(const PagedVector<char>& pool,
        std::uint32_t start, std::uint32_t len, RoutingColdMail& inbox) {
        PoolMailSource src{ pool, static_cast<int32_t>(start),
                            static_cast<int32_t>(start + len) };
        deserializeMailBlobInto(src, inbox);
    }





    /// @brief One *logic block* (LB) — the prover's primary unit of state.
    ///
    /// @details
    /// A `Memory` instance is one node in the LB tree. The tree's root is the
    /// "main" sentinel (default-constructed, `exprKey.empty()`,
    /// `parentMemory == nullptr`); every non-root LB carries the expression
    /// it owns (`exprKey`) and a back-pointer to its parent (`parentMemory`).
    /// Children are reached via the analyzer's `SimpleMapStore`
    /// (`simpleMapStore.findChild(parent, K)`), where `K` is a routing-key
    /// string set at LB creation in `prover.cpp`; the down-edges no longer live
    /// on the `Memory` shell.
    ///
    /// The `Memory` struct is intentionally large — it bundles every kind of
    /// per-LB state the prover touches:
    ///
    /// **Tree structure**
    /// - down-edges (routing-key → child LB) live in the analyzer's
    ///   `SimpleMapStore`, not on the shell; the routing-key strings intern in
    ///   `skeletonInterner()` (D-152).
    /// - `parentMemory`     — parent pointer (`nullptr` at the sentinel root).
    /// - `exprKey`          — the expression text owned by this LB. Empty at
    ///   the root sentinel; otherwise carries the disintegrated form. NOTE:
    ///   the routing key may differ from `exprKey` (the recursion-block-#2 site
    ///   links the routing key `(=[s(recN),zero])` to the child while the
    ///   child's `exprKey` is the substituted induction-variable form
    ///   `(=[<digitArg>,<zero>])`). Always match a target LB by walking the
    ///   full parent chain to root, never by `exprKey` alone.
    /// - `level`            — depth in the LB tree (`-1` for the root sentinel).
    ///
    /// **Iteration / counter state**
    /// - `startInt`, `startIntRepl`, `startIntPi` — counters for the three
    ///   independent name-mint streams used during disintegration. See
    ///   [I-17](../../docs/agentic_swdd/30_invariants.md#i-17): the freshness check
    ///   relies on these being a single monotonic stream within their
    ///   respective category.
    /// - `recursionCounter`, `contradictionIndex`, `recursionHypothesisId`,
    ///   `isPartOfRecursion`, `primedForContradiction`, `contradictionTheoremId`
    ///   — induction- / recursion- / contradiction-LB bookkeeping (the two
    ///   ids decode through the NameMap; 0 = unset).
    ///
    /// **Statement state**
    /// - `intToBeProved`            — pending obligations on this LB.
    /// - `intEncodedStatements`     — THE statement registry: every statement
    ///   known here, as int16 rows; string form reconstructed on demand via
    ///   `decodeExpression`.
    /// - `intLocalEncodedStatements`,
    ///   `intLocalEncodedStatementsDelta` — local-origin statements (status
    ///   0/1 derivations) + their per-burst delta, int16 rows consumed by
    ///   the static request pipeline and the delta consumers
    ///   (`fillMailOut`, `dischargeToBeProved`).
    /// - `intLocalEncodedStatementsSet` — packed-key membership index of
    ///   the local-origin statements for the local-premise gate (see
    ///   [I-7](../../docs/agentic_swdd/30_invariants.md#i-7) /
    ///   [D-29](../../docs/agentic_swdd/40_decisions.md#d-29)); maintained
    ///   beside `intLocalEncodedStatements`, O(1) membership inside
    ///   `checkLocalEncodedMemoryStatic`.
    /// - `intExternalStatements`    — EXTERNAL-mail facts (int16 rows)
    ///   staged for the next burst; emptied and refilled every
    ///   burst (NOT persistent, unlike the local-premise containers).
    /// - `intKnownStatements`       — packed `(originalId, validityId)`
    ///   keys; the value's `known` bit answers "is this statement already
    ///   known here" (Site F), the `registered` bit answers "was this
    ///   statement registered through an add-path door".
    /// - `intStatementLevelsMap`    — packed `(originalId, validityId)`
    ///   keys → per-statement set of LB levels at which it was deposited.
    ///
    /// **Hash memory slots** (each is a [`HashMemory`](#hashmemory))
    /// - `overallHashMemory`        — full set of rules visible from this LB.
    /// - `localHashMemory`          — rules installed at this LB.
    /// - `localHashMemoryDelta`     — rules installed since the last burst,
    ///   awaiting absorption.
    /// - `workingMemory`            — rules recovered from EXTERNAL mail;
    ///   emptied and refilled every burst (NOT persistent).
    ///
    /// **Equivalence classes**
    /// - `equivalenceClassesMap`    — validity id → list of [`EquivalenceClass`](#equivalenceclass).
    /// - `eqClassSttmntIndexMapMap` — validity name → (class `memberIds` →
    ///   statement index) cache used by the cross-pair `equality2` gate
    ///   ([I-32](../../docs/agentic_swdd/30_invariants.md#i-32)).
    ///
    /// **Mail subsystem**
    /// - `mailIn`, `mailOut`        — routing-channel mail (parent → child;
    ///   [I-26](../../docs/agentic_swdd/30_invariants.md#i-26) /
    ///   [I-28](../../docs/agentic_swdd/30_invariants.md#i-28)).
    /// - `sameIterationInternalMail`           — per-LB integration-revival channel
    ///   (typed `Mail` since 2026-05-07; see
    ///   [D-53](../../docs/agentic_swdd/40_decisions.md#d-53)). Cleared at the top of
    ///   the hashburst after absorb per
    ///   [I-21](../../docs/agentic_swdd/30_invariants.md#i-21).
    ///
    /// **Origin / provenance**
    /// - `exprOriginMap`            — packed `(expressionId, validityId)`
    ///   origin-interner key → vector of `(OriginTag, packed antecedent
    ///   keys)` pairs (id form per `D-131`). The
    ///   ALL-SCOPES half of the mail-out contract per
    ///   [I-26](../../docs/agentic_swdd/30_invariants.md#i-26).
    /// - `integrationStartIntMap`   — per-int_-prefix counter snapshots used
    ///   to keep integration-side renames stable across iterations (keyed
    ///   by templateInterner id).
    /// - `integrationPrepared`,
    ///   `integrationPreparedMarker` — integration-side admission tracking
    ///   (packed (templateId, validityId) keys).
    /// - `intValidityNamesToFilter` — receiver-side mail filter set.
    /// - `canBeSentIds`,
    ///   `canBeSentMarkerIds`       — sender-side mail-eligibility memo
    ///   sets (NameMap ids; the plain set holds `int_lev_*` carriers only).
    /// - `weakVariables`,
    ///   `intAxedVariables`         — variable-eligibility classification
    ///   used by `multiplyImplication` and admission.
    ///
    /// **OR-branching state**
    /// - `orBookkeeping`            — convergence tracking: keyed by
    ///   packed `(exprId, sigId)` lbStateInterner pairs → decoded-lex
    ///   ordered branch-disjunct id sets (the legacy `orAdmissionSet`
    ///   gate container was dropped — no insert site existed, D-31).
    /// - `orDisjunctCount`          — disjunct count per OR-signature id
    ///   for the convergence check.
    ///
    /// **State flags**
    /// - `isActive`                 — set to `false` by `deactivate*` paths
    ///   when this LB has been retired.
    /// - `hasWork`                  — quiescence latch: `true` when this LB's
    ///   next burst may produce work. Cleared at `performElemPhase3` exit on a
    ///   provable no-op burst, set by every cross-LB wake door
    ///   (`D-194`); the burst-skip predicate reads it.
    /// - `deltaNumberStatements`    — VESTIGIAL (never incremented); the no-op
    ///   short-circuit it was reserved for is now carried by `hasWork`.
    ///
    /// **Static hot path**
    /// - `nameMap`                  — per-LB `NameMap` (see
    ///   [`NameMap`](#namemap)).
    ///
    /// The default constructor produces the root sentinel (empty `exprKey`,
    /// `parentMemory == nullptr`, `level == -1`, `isActive == true`,
    /// `recursionCounter == 0`, all containers empty). Tests can rely on
    /// this shape.
    ///
    /// @see [`NameMap`](#namemap), [`HashMemory`](#hashmemory),
    ///      [`Mail`](#mail), [`EquivalenceClass`](#equivalenceclass).
    /// @see The full-LB-chain trace requirement when matching a target
    ///      LB in any debug trap (full parent walk to the root sentinel,
    ///      never `exprKey` alone).
    /// @see The hashburst dump (Rule 14; ENTRY in `performElemPhase1`, EXIT in `performElemPhase3`) operating
    ///      on this state is treated as sacred infrastructure: format,
    ///      lambdas, call sites, and target-LB chain match all require
    ///      explicit approval to change.
    struct Memory {
        int startInt;
        int startIntRepl;
        int startIntPi;
        // Tracks every `(implicationN[...])` form whose rules were
        // installed into this LB's hash registry via `addToHashMemory`.
        // Used by `sanitizeHashMemory` (called once at end of every
        // `performElemPhase3`) to rewrite implications under
        // equi-class canonicalization: when an entry's `it_/int_` arg
        // is now downprioritized in its equi-class, the canonical-
        // priority form is mailed to `sameIterationInternalMail` and the old form
        // is eradicated from the LB so the gate at
        // `addExprToMemoryBlock` allows re-entry. Set semantics
        // because `addToHashMemory`'s three-way fan-out per install
        // would otherwise insert duplicates.
        // Packed lbStateInterner pairs packLbStateKey(implId, scopeId)
        // (D-135) — both halves own-space, NameMap-
        // decoupled, so ids never re-bind; the registry deliberately
        // survives destroyGrid (existing behavior), which is why the
        // lbStateInterner does NOT reset there.
        TypedColdSet<LbStatePairKey>& expandedImplications =
            lbMemory.expandedImplications;
        // The LB's expression key (its identity) — statified: interned in the
        // never-deloaded analyzer-wide skeletonInterner() and stored as a 4-byte
        // id (0 == empty, the root sentinel). The former `std::string exprKey`
        // member is now the exprKey() accessor, byte-identical by construction
        // (D-151).
        int32_t exprKeyId = 0;

        /// @brief The LB's expression key as a string (byte-identical to the
        ///        former `std::string exprKey` member).
        /// @return The interned key, or "" when `exprKeyId == 0`.
        std::string exprKey() const;

        /// @brief Zero-copy span twin of @ref exprKey() — the LB identity's
        ///        interned bytes without materializing a `std::string`.
        ///
        /// @details
        /// Returns a `StrSpan` over the `skeletonInterner()` cold bytes for
        /// `exprKeyId`, byte-identical to `exprKey()`'s content; an EMPTY span
        /// (`len == 0`) when `exprKeyId == 0` (the root sentinel), matching
        /// `exprKey()`'s empty string. The span is valid while the LB is
        /// resident — `skeletonInterner()` is the process-wide, never-deloaded
        /// LB-identity table, so its bytes never move under a resident LB
        /// (D-151 / D-152); no per-firing heap `std::string` on the phase-2
        /// firing-check hot path.
        /// @return A span over the interned key bytes, or an empty span.
        /// @invariant `I-97` — the LB identity is a 4-byte id into the shared
        ///            skeleton interner; the span decodes byte-identically.
        /// @see exprKey — the owning-string twin; `skeletonInterner`.
        StrSpan exprKeyView() const;

        /// @brief Set the LB's expression key, interning it (empty -> id 0).
        /// @param s The key string.
        void setExprKey(const std::string& s);
        // The concrete "recursion" hypothesis deposited in this LB
        // (tempExpr2 for check_induction_condition, tempExpr4 for
        // check_zero), as a NameMap id; 0 when the LB is not an
        // induction sub-block. Used by vacuous-truth emission as the 3rd
        // ingredient so the verifier's chapter-local trace can actually
        // reach it (the derivation chains inside the LB use this form,
        // not exprKey). Decoded at use; the string content lives on the
        // cold name table, deloadable with the LB.
        int16_t recursionHypothesisId = 0;
        Memory* parentMemory;
        int level;

        // Reference aliases into lbMemory's four folded-in HashMemory instances
        // (D-147), so the body.overallHashMemory... call sites
        // are unchanged. The instances live in lbMemory (declared below); binding
        // to their stable storage before lbMemory constructs is the same pattern
        // as the equivalenceClassesMap alias.
        HashMemory& overallHashMemory = lbMemory.overallHashMemory;
        HashMemory& localHashMemory = lbMemory.localHashMemory;
        HashMemory& localHashMemoryDelta = lbMemory.localHashMemoryDelta;
        /// @brief Rules recovered from EXTERNAL mail for the next burst.
        ///
        /// @details Holds the implications that `disintegrateExpr2`
        ///          recovers from this LB's external (`mailIn`) traffic
        ///          during the post-hashburst mail-absorption phase.
        ///          Unlike the three persistent slots above, this slot is
        ///          emptied and refilled from fresh external mail on
        ///          every burst — the burst's encoded-request blocks read
        ///          it in place of the former throwaway scratch memory.
        ///          Implications reach it only through
        ///          `addExprToMemoryBlock` (the single hash-memory door).
        HashMemory& workingMemory = lbMemory.workingMemory;

        // Keyed by validity id (D-134), migrated onto the cold BLOB map (Batch
        // 3): a reference alias into lbMemory; each class is one canonical byte
        // blob (serializeEquivalenceClass). Read paths decode through
        // decodeClassesAt / decodeClassesById; write paths assignClassesById.
        // Order-sensitive walks (reactToHypo, applyEquiClasses pass 2, the dump
        // section) iterate a decoded-name lex-sorted snapshot — never raw id
        // order (I-84).
        TypedColdBlobMap<int16_t, EquivalenceClass>& equivalenceClassesMap =
            lbMemory.equivalenceClassesMap;

        /// @brief Per-step delta set of equivalence classes whose membership
        ///        changed during the current elementary step's
        ///        mail-absorb pass. Cleared at the start of every elementary
        ///        step; populated by `updateEquivalenceClasses` immediately
        ///        after a merged class is committed to
        ///        `equivalenceClassesMap`; consumed by `applyEquiClasses`
        ///        which re-applies each delta class against every current
        ///        `encodedStatement`.
        ///
        /// @details Each entry carries `(validityId, class value copy)` —
        ///          a value copy rather than an index, because subsequent
        ///          merges at the same scope can shrink the underlying
        ///          class vector and invalidate any cached index. The
        ///          validity id is the class's scope (what
        ///          `updateEquivalenceClasses` was called with).
        ///
        ///          Introduced by the `sandbox/equi_reshuffle` refactor that
        ///          extracts equivalence-class application out of
        ///          `addExprToMemoryBlock` into a single explicit pass per
        ///          elementary step. The entries are serialized class snapshots
        ///          on the LB's cold deloadable arena (tag base 655), accessed via
        ///          push / size / validityAt / classAt. A reference alias into
        ///          lbMemory (the cold deloadable storage). See
        ///          `ChangedClassesBuffer`.
        ChangedClassesBuffer& changedClassesThisStep = lbMemory.changedClassesThisStep;

        /// @brief Per-hashburst staging buffer for algebra-`admissionMap`
        ///        writes, drained once after the fixpoint loop.
        ///
        /// @details
        /// Populated only inside `checkLocalEncodedMemoryStatic`'s marker
        /// branch: instead of mutating `overallHashMemory.admissionMap` /
        /// `admissionStatusMap` / `varsInAdmissionMapKeys` and firing
        /// `revisitRejected2` inline mid-burst, each such write is appended
        /// here as an `AdmissionKeyAlgebraRecord`. `drainAdmissionKeysAlgebra`
        /// replays the buffer in firing order immediately before the
        /// post-burst `standardProcessing` absorb, so the revival cohorts land
        /// on `sameIterationInternalMail` in time and `admissionMap` is current
        /// before that absorb's `isAdmitted` reads it. Cleared at burst start
        /// beside the other per-step delta clears; never read during the
        /// fixpoint loop.
        ///
        /// @see `AdmissionKeyAlgebraRecord` — the element type.
        /// @see `prover.hpp::drainAdmissionKeysAlgebra` — the drain.
        std::vector<AdmissionKeyAlgebraRecord> admissionKeysAlgebra;

        /// @brief Per-hashburst staging buffer for integration-template
        ///        registrations, drained once after the fixpoint loop.
        ///
        /// @details
        /// Integration-side companion of `admissionKeysAlgebra`. Populated only
        /// inside `checkLocalEncodedMemoryStatic`'s marker branch: instead of
        /// calling `prepareIntegration` inline mid-burst (which writes
        /// `admissionMapIntegration` / `varsInAdmissionMapIntegrationKeys` via
        /// `prepareIntegrationCore2`), the call's inputs are appended here as a
        /// `DeferredIntegrationPrep`. `drainDeferredIntegrationPreps` replays the
        /// buffer in firing order immediately before the post-burst
        /// `standardProcessing` absorb, so the registered templates are present
        /// before that absorb's `updateAdmissionMapIntegration` cascade reads
        /// them. Cleared at burst start beside the other per-step delta clears;
        /// never read during the fixpoint loop.
        ///
        /// @see `DeferredIntegrationPrep` — the element type.
        /// @see `prover.hpp::drainDeferredIntegrationPreps` — the drain.
        std::vector<DeferredIntegrationPrep> deferredIntegrationPreps;
        // Keyed by the templateInterner id of the integration template
        // (D-135; the former string key). No validity
        // dimension — the map never filters at wipeSubtree (counter
        // snapshots, no scope tag).
        ColdHashMap<PodKeyStore<int16_t>, int>& integrationStartIntMap =
            lbMemory.integrationStartIntMap;

        // Routing mailIn uses a dedicated transient mail-pool arena that is
        // filled, absorbed, and released within the phase-1 worker claim.
        // Outgoing mail lives inside LbMemory and is claim/reloaded only when
        // `mailOutPending` is set.
        RoutingColdMail mailIn;
        DeloadableMailOut& mailOut = lbMemory.mailOut;
        // Two-channel internal-mail split (sandbox/equi_reshuffle
        // follow-up consolidation, 2026-05-24). `sameIter` is the
        // ephemeral hashburst-output channel: filled by
        // `checkLocalEncodedMemoryStatic` during the burst and by any
        // discharge that happens during the pre-burst absorb (so the
        // emissions feed the same step's hashburst); drained by the
        // post-burst standardProcessing call. `nextIter` is the
        // cross-iteration channel: filled by every other current
        // writer (`dischargeToBeProved` parent-scope emissions when
        // routed via the post-burst standardProcessing call's
        // internalMailOut, the vacuous-truth path inside
        // `addExprToMemoryBlock`, the `updateGlobalDirect` siblings);
        // drained at the start of the next step alongside `mailIn`.
        // Reference aliases into the two LbMemory ColdMail members (the cold
        // deloadable storage); the spellings are kept so call sites are
        // unchanged. Like the other lbMemory.* aliases, declared before lbMemory
        // and bound by reference (the subobject address is valid; the references
        // are used only after lbMemory constructs).
        ColdMail& sameIterationInternalMail = lbMemory.sameInternalMail;  // hashburst output channel
        ColdMail& nextIterationInternalMail = lbMemory.nextInternalMail;  // cross-iteration deferral channel

        // Equivalence-class statement-index cache. Was a nested
        // std::map<validityId, std::map<memberIds, int>>; flattened (Batch 2)
        // onto the cold byte-key map keyed by bytes(validityId ++ memberIds) ->
        // the statement-registry waterline. A reference alias into lbMemory;
        // all access is point lookups through the encodeEqClassKey /
        // lookupEqClassIndex / upsertEqClassIndex / eraseEqClassIndex helpers
        // (never iterated in key order, never dumped). Dischargeable, reset at
        // destroyGrid.
        TypedColdMap<EqClassKey, int>& eqClassSttmntIndexMapMap =
            lbMemory.eqClassSttmntIndexMapMap;

        /// @brief Decode a validity id's equivalence-class list from the cold
        ///        blob store — the read boundary that replaced the former
        ///        `classesAt` pointer return.
        ///
        /// @details
        /// Each class is reconstructed from its canonical blob
        /// (`deserializeEquivalenceClass`) into a transient heap working struct
        /// (the I-84 pattern). An id with no bucket — and the empty run an
        /// existing id may hold — both yield an empty vector, the defined "no
        /// classes at this scope" result callers branch on exactly as they did
        /// on the historical `find() == end()`. Burst-safe (non-minting reads).
        ///
        /// @param validityId Scope id to decode.
        /// @return The scope's class list (empty when the id has no bucket).
        /// @see assignClassesById — the write-back counterpart;
        ///      decodeClassesAt — the by-name overload.
        std::vector<EquivalenceClass> decodeClassesById(int16_t validityId) const {
            const int32_t id = equivalenceClassesMap.lookup(validityId);
            if (id == 0) return {};
            return equivalenceClassesMap.recordsAt(id);
        }

        /// @brief Decode a validity scope's class list by NAME — the non-minting
        ///        by-name read boundary.
        ///
        /// @details
        /// A `NameMap::lookup` miss (never-interned validity) yields an empty
        /// vector, identical to the absent-bucket result, so read paths holding
        /// only the validity name keep their old branch. Never mints.
        ///
        /// @param validityName Scope to decode.
        /// @return The scope's class list (empty when the scope has none).
        /// @see decodeClassesById.
        std::vector<EquivalenceClass> decodeClassesAt(
            const std::string& validityName) const
        {
            const int16_t vId = nameMap.lookup(validityName);
            if (vId == 0) return {};
            return decodeClassesById(vId);
        }

        /// @brief Serialize and store a validity id's whole class list — the
        ///        bucket-replace write boundary (the cold-store form of the
        ///        former `equivalenceClassesMap[vid].swap(...)`).
        ///
        /// @details
        /// Each class is serialized to its canonical blob
        /// (`serializeEquivalenceClass`); the concatenated blobs + per-blob
        /// lengths drive one `ColdBlobMap::assignRun` (a brand-new key opens its
        /// run, an existing key's run is replaced in place). Single-threaded
        /// write side only (I-83).
        ///
        /// @param validityId The scope id.
        /// @param classes    The scope's new class list (may be empty — an
        ///                   empty run).
        /// @see decodeClassesById — the read counterpart.
        void assignClassesById(int16_t validityId,
                               const std::vector<EquivalenceClass>& classes)
        {
            equivalenceClassesMap.assignRun(validityId, classes);
        }
        bool isActive;

        /// @brief Quiescence latch — `true` when this LB's next burst MAY
        ///        produce new logical work, `false` when its previous burst
        ///        provably did nothing and no input has arrived since.
        ///
        /// @details
        /// Producer-side half of the quiescent-burst skip predicate
        /// (`D-194`). A plain scalar on the `Memory` shell,
        /// which lives in the never-deloaded LB-body slab (I-109), so
        /// `proveKernel`'s single-threaded active-build can read it while the
        /// LB's deloadable arena is cold. SLEEP: cleared at `performElemPhase3`
        /// exit iff the burst mutated nothing — no statement-count change vs
        /// `encodedCountAtBurstStart`, `mutatedThisBurst` false, and both
        /// INTERNAL mail channels (`sameIter` / `nextIter` — inputs this LB's
        /// own next burst absorbs) empty. `mailOut` is deliberately excluded
        /// from the SLEEP fold: it is OUTGOING-ONLY under the pull model
        /// (I-64 — fillMailOut and the deposit doors write it, the commit
        /// barrier is the sole consumer, no burst path derives local state
        /// from it), and the barrier's commit sweep iterates `bodies`
        /// regardless of sweep status, so pending mailOut content ships
        /// whether or not the LB sweeps and can never make its next burst
        /// productive. WAKE: set `true` by every cross-LB write door that
        /// deposits new work into this LB (`I-153`), and at
        /// birth (default `true`) so an LB's first burst always runs. The skip
        /// filter reads `hasWork || mailPeek` and NOTHING residency-derived
        /// (the I-106 / I-108 determinism doctrine). Cross-LB pending mail is the
        /// one wake polled fresh each iteration via `MailLog::mailPeek`, not
        /// flagged here (keeps the O(descendants) fan-out off the commit path).
        /// @invariant A missed wake is a missed theorem (unsound); every deposit
        ///            door must set it — validated by `QUIESCE_SHADOW_CHECK`.
        bool hasWork = true;

        /// @brief Always-resident summary bit for the deloadable outgoing
        ///        mailbox.
        ///
        /// @details Every outgoing-mail write door sets this bit. The serial
        /// commit sweep reads it from the LB shell before deciding whether the
        /// cold LB must be claimed and loaded; it therefore replaces the old
        /// never-deloaded mailbox emptiness probe. Clearing a committed mailbox
        /// clears the bit only after the columns and private interner reset.
        ///
        /// @invariant `false` implies `mailOut.empty()`; `true` requires a
        ///            claim/reload before inspecting the mailbox.
        ///            [I-163]
        bool mailOutPending = false;

        /// @brief Always-resident logical live-byte snapshot of `mailOut`.
        ///
        /// @details Updated by every outgoing-mail write door and reset with the
        /// mailbox. Telemetry can therefore sum all LBs after a phase join
        /// without loading cold mailboxes. This is a logical component size;
        /// physical main-pool blocks remain accounted by the whole-pool peak.
        int64_t mailOutLiveBytes = 0;

        /// @brief Per-burst dirty flag for NON-statement mutations (admission
        ///        churn, subtree wipe) the statement-count diff and the
        ///        mail-channel belt do not observe.
        ///
        /// @details
        /// `false`-initialised at `performElemPhase1` entry; set `true` by the
        /// burst's non-statement mutation choke points (the admission-drain site
        /// in `performElemPhase2` and the pending-wipe-scope drain in
        /// `performElemPhase3`). Folded into `hasWork` at phase-3 exit. Statement
        /// deposits / rewrites are caught separately by the
        /// `intEncodedStatementsCount()` vs `encodedCountAtBurstStart` diff;
        /// revival / discharge emission by the `sameIter` / `nextIter`
        /// emptiness belt (`mailOut` deliberately excluded — outgoing-only,
        /// see `hasWork`). Over-approximating on purpose (Rule 19: when in
        /// doubt, stay dirty).
        bool mutatedThisBurst = false;

        /// @brief `intEncodedStatementsCount()` snapshot taken at
        ///        `performElemPhase1` entry (before the pre-burst mail absorb).
        ///
        /// @details
        /// The SLEEP detector compares it against the count at
        /// `performElemPhase3` exit: any change — a deposit, an equi-class
        /// rewrite, or a subtree-wipe erase — keeps `hasWork` set. A cheap,
        /// deload-tolerant integer read (the count lives with the persistent
        /// registry, I-108), so it is legal while the main arena is cold.
        int encodedCountAtBurstStart = 0;

        /// @brief `QUIESCE_SHADOW_CHECK` scratch: whether the skip predicate
        ///        WOULD have skipped this LB's burst this iteration.
        ///
        /// @details
        /// Meaningful only under the `QUIESCE_SHADOW_CHECK` compile flag: the
        /// active-build records the predicate's verdict here but still sweeps the
        /// LB, and `performElemPhase3` asserts a would-skip LB's burst was a
        /// no-op. Inert (written but never read) in a normal build.
        bool shadowWouldSkip = false;

        /// LB-split part count = number of EXPRESSION BUCKETS this LB runs as. `1`
        /// is the unsplit identity (byte-identical to no split). Two-state per LB:
        /// `1` (not a straggler) or `logicalCores` (a straggler, split into that
        /// many expression buckets). `proveKernel`'s end-of-iteration stats pass
        /// sets it for the NEXT iteration from this iteration's total submatch work
        /// (`isStraggler`, the idle-core fair-share; D-201).
        /// A value `> 1` also marks the LB an ineligible steward eviction victim.
        /// Persists across iterations; never reset. Default `1`.
        int numberOfParts = 1;

        /// One-shot "just activated this iteration" flag. A born-parked induction
        /// zero block that `activateZeroCondition` flips active (the only
        /// inactive->active transition, I-112) has no prior burst, so the submatch
        /// straggler stat cannot have flagged it — yet its first active burst
        /// processes a full backlog. `proveKernel`'s task build splits such an LB
        /// PREEMPTIVELY this iteration (expression buckets) regardless of
        /// `numberOfParts`, then clears the flag (one-shot; later iterations fall
        /// back to the submatch stat). Set single-threaded in
        /// `activateZeroCondition`, cleared single-threaded at task build.
        bool justActivated = false;
        bool isPartOfRecursion;
        // VESTIGIAL (never incremented — no prover increment site exists). The
        // no-op-iteration short-circuit it was reserved for is now carried by
        // `hasWork` + the phase-3 SLEEP detector (D-194). Kept
        // only for the CE-swap / dump / test references that still name it; not a
        // live counter.
        int deltaNumberStatements;
        // Id-form history map (D-131) on the cold blob map
        // (I-121): packed (expressionId, validityId)
        // origin-interner keys -> the key's run of (OriginTag, packed antecedent
        // keys) lines. A reference alias into lbMemory (the equivalenceClassesMap
        // pattern) so the body.exprOriginMap call sites keep their spelling.
        // Order-sensitive walks derive decoded lex-sorted snapshots
        // (decodeOriginMapSorted). Mail origin maps stay string — encode at
        // absorb, decode at fillMailOut. Survives wipeSubtree (I-44) and survives
        // discharge (the chapter export reads it on discharged LBs); reset to
        // empty at CE-clone teardown (resetToFresh). The originInterner its keys
        // index resets in lockstep at destroyGrid.
        TypedColdBlobMap<int64_t, IdOrigin>& exprOriginMap = lbMemory.exprOriginMap;
        int recursionCounter;
        int contradictionIndex;
        // Packed (templateId, validityId) keys (D-135)
        // — the populations are clean-signature / marker-form template
        // strings, the same family as the admission EWV sets. Writers via
        // mintTemplateKey, gates via the non-minting lookupTemplateKey;
        // wipeSubtree filters by the low-16-bits validity predicate; the
        // dump sections derive decode + lex-sort.
        ColdHashSet<PodKeyStore<int32_t>>& integrationPrepared =
            lbMemory.integrationPrepared;
        ColdHashSet<PodKeyStore<int32_t>>& integrationPreparedMarker =
            lbMemory.integrationPreparedMarker;
        // D-72 (sandbox/implication_cleanup):
        // Pending scopes to wipe at the END of the hashburst (drained
        // by `performElemPhase3` immediately after
        // `sanitizeToBeProved` and before the EXIT trap dump). Impl
        // closure call sites and `cleanUpOrIntegrationBranches` insert
        // into this set instead of calling `wipeSubtree` immediately;
        // immediate wipe was unsafe because it ran mid-iteration in
        // `addExprToMemoryBlockKernel`'s `sortedNew` loop and
        // invalidated subsequent iterations' lookups in
        // `intStatementLevelsMap` / `equivalenceClassesMap`. Deferring to
        // burst boundary makes the full wipe (including those scope-
        // keyed maps) safe.
        // NameMap validity ids (D-135): every queued
        // scope was created via encodePush, so the insert is a non-minting
        // lookup + assert. The drain decodes + lex-sorts before the
        // wipeSubtree calls — the former string-set order exactly.
        ColdHashSet<PodKeyStore<int16_t>>& pendingWipeScopes =
            lbMemory.pendingWipeScopes;
        // Mail-eligibility memo sets as NameMap ids (the
        // pendingWipeScopes id-set pattern). `allowedForMail` is the only
        // reader and probes the plain set only for `int_lev_*`-carrying
        // expressions, so the insert sites gate on that substring — an
        // entry for any other expression could never be probed. The
        // marker set holds marker-substituted forms (no gate; its probe
        // key is the marker form, not the original). Inserts encode
        // (single-threaded sites only — I-83); probes are non-minting
        // lookups. The dump sections derive their byte-identical output
        // by decode + lex-sort.
        ColdHashSet<PodKeyStore<int16_t>>& canBeSentIds =
            lbMemory.canBeSentIds;
        ColdHashSet<PodKeyStore<int16_t>>& canBeSentMarkerIds =
            lbMemory.canBeSentMarkerIds;
        // The weak-variable record (D-134). Key =
        // packStatementKey(variableId, validityId). Written at
        // updateWeakVariables' insert; filtered at wipeSubtree by the
        // low-16-bits validity predicate; swapped at the destroyGrid
        // capacity release. Probed by reduceEqClassIds. The dump section
        // derives its byte-identical output by decode + lex-sort.
        ColdHashSet<PodKeyStore<int32_t>>& intWeakVariables =
            lbMemory.intWeakVariables;

        // Dedicated id space for admission/rejected template strings
        // (D-132). Disjoint from nameMap's id space —
        // see TemplateInterner. Resets only in lockstep with the
        // destroyGrid nameMap reset.
        TemplateInterner templateInterner;

        // Dedicated int32 id space for admission/rejected VALUE-field
        // strings (D-1322). See ValueInterner; same
        // destroyGrid lockstep reset.
        ValueInterner valueInterner;

        // Dedicated int32 id space for the origin maps' expression AND
        // validity strings (D-131). Deliberately not
        // the NameMap — the trace dumps the full NameMap id table, and
        // mail-carried dependencies can name child-LB scopes never
        // interned locally. Same destroyGrid lockstep reset; never wiped
        // on scope teardown (exprOriginMap itself survives wipeSubtree, so
        // its id space must too).
        ValueInterner originInterner;

        // Dedicated int32 id space for this LB's outgoing routing mailbox.
        // The table is owned by lbMemory.mailOut and therefore deloads, reloads,
        // and clears in lockstep with the two id-bearing mailOut columns.
        ValueInterner mailOutInterner;

        // Dedicated int32 id space for the rule-registry strings — LMV head
        // templates, premise-chain elements, original implications,
        // remaining-arg names, and the originals chains
        // (D-133). Encodes happen only at the
        // single-threaded install sites; the parallel hashburst only
        // decodes (array-index const refs, I-83-safe). Same destroyGrid
        // lockstep reset.
        ValueInterner ruleInterner;

        // Dedicated id space for the LB-state strings — OR-branching
        // bookkeeping (derived expressions, or-signatures, branch
        // disjuncts) and the expanded-implication registry
        // (D-135). Not the NameMap: mail-absorbed
        // implications may be locally un-interned, and a mint there would
        // shift the dumped id table. All writers single-threaded; same
        // destroyGrid lockstep reset.
        ValueInterner lbStateInterner;

        // Lazy per-id memoization for the equivalence-class path (classifyName /
        // scanSpecialTokens results keyed by NameMap id). Derived state, a pure
        // function of nameMap contents — never wiped or filtered on scope
        // teardown. On the cold deloadable arena (tag base 705), survives
        // discharge; a reference alias into lbMemory. See EqClassNameCaches.
        EqClassNameCaches& eqClassNameCaches = lbMemory.eqClassNameCaches;

        // --- Incubator: contradiction proving ---
        bool primedForContradiction = false;
        // The pre-built theorem published when this primed LB finds its
        // contradiction, as a NameMap id; 0 when unset (a defined state —
        // the discharge publishes nothing then). Decoded at the publish
        // site; content lives on the cold name table.
        int16_t contradictionTheoremId = 0;

        // --- OR bookkeeping: convergence tracking ---
        // (The legacy orAdmissionSet gate container was dropped with
        // D-135 — it had no insert site (D-31), so
        // the always-empty set made its gate constant-false.)
        // Key: packLbStateKey(exprId, sigId) in lbStateInterner space —
        // (derived_expression, or_signature) → a set of branch-disjunct ids
        // kept in DECODED-LEX order. Statified onto the cold-map set form
        // (Batch 2) — a reference alias into lbMemory; build the run with
        // insertSorted + a per-call DecodedIdLess (comparator state is the
        // lbStateInterner), and read it IN RUN ORDER (valueAt), NEVER via
        // coldIntSetAt (which would re-sort by raw int). The disjunct order
        // feeds ordisMerge's origin construction and its per-branch cleanup
        // loop — decoded order, never id order (I-84).
        TypedColdSetMap<LbStatePairKey, int32_t>& orBookkeeping =
            lbMemory.orBookkeeping;
        // Number of disjuncts per OR signature (lbStateInterner id) for
        // the convergence check.
        ColdHashMap<PodKeyStore<int32_t>, int>& orDisjunctCount =
            lbMemory.orDisjunctCount;

        // --- Static hot path: int16_t-based hash memory ---
        NameMap nameMap;

        // --- Statification: the static per-LB aggregate (tier 1) ---
        // Owns the LB's block manager + every statified container; the
        // deload machinery serializes it as one unit. Lazy: holds zero
        // pool blocks until a statified container first grows, so the
        // `Memory empty` teardown temporaries and the pre-config `body`
        // member stay free.
        LbMemory lbMemory{ &staticMemory() };

        // --- Statification: the PERSISTENT per-LB arena (the determinism fix) ---
        // A second arena drawn from the SEPARATE persistent pool
        // (persistentMemory(), static_persistent_block_bytes) that is NEVER
        // deloaded. It backs Memory::intToBeProved (the goal registry, below) so
        // the deactivation survey reads it regardless of the main lbMemory
        // arena's deload state — removing the timing-dependent input that made
        // deactivation non-deterministic. Lazy: holds zero blocks until the first
        // goal insert; reclaimed (releaseAll) only at discharge
        // (dischargeStatementContent).
        //
        // DECLARATION ORDER IS LOAD-BEARING: persistentArena must outlive
        // intToBeProved (members destruct in reverse declaration order, and the
        // map frees its pages into the arena at destruction), so it is declared
        // HERE, above the intToBeProved member. persistentDirty satisfies the
        // TypedColdSetMap ctor but is never consulted — intToBeProved is in no
        // deload stream, so its dirty state is never read.
        DirtyState persistentDirty = DirtyState::Clean;
        LbArena persistentArena{ &persistentMemory() };

        // THE statement registry (int16 rows; string form on demand via
        // decodeExpression) — statified: lives on the LB's static pages
        // inside `lbMemory`. The member below is a pure REFERENCE ALIAS
        // so the registry's many access sites (including the Rule-14
        // sacred hashburst dump) keep their `body.intEncodedStatements`
        // spelling; it is not separate state. The alias makes `Memory`
        // non-assignable as a whole — by design, nothing may
        // whole-assign an LB (teardown swaps individual members).
        PagedVector<IntEncodedExpr>& intEncodedStatements =
            lbMemory.intEncodedStatements;

        // Deload bookkeeping: the file set written by the last
        // deloadStaticContainers call and the chain string it was keyed
        // on. Reload reads THESE, never the manifest (file count is not
        // recoverable from names alone). Heap bookkeeping by design
        // (D-174).
        std::vector<std::string> deloadFiles;
        std::string deloadChain;
        // Injective per-LB file-name handle (replaced the chain hash):
        // assigned ONCE at first dump by ensureDeloadOrdinal, then reused
        // for every re-dump and tail so a base+tail set shares one
        // identity. -1 = unassigned. Heap skeleton, read while cold.
        int64_t deloadOrdinal = -1;

        /// @brief Which on-disk format the LAST dump wrote — the dispatch key
        ///        `reloadFromImage` reads to pick the matching loader.
        ///
        /// @details
        /// `Canonical` = the v3 element stream (discharge / chapter-export
        /// path; the determinism doctrine stays intact there); `Raw` = the v4
        /// arena image (the near-memcpy eviction path). Set by whichever dump
        /// ran last (`dumpStaticContainers` -> `Canonical`,
        /// `dumpStaticContainersRaw` -> `Raw`), so a raw-evicted LB that is
        /// later discharged flips back to `Canonical` and its next reload uses
        /// the v3 loader. Recorded, not peeked — the reload needs no extra
        /// file open. Both loaders still assert their own header version, so a
        /// field/file mismatch is caught (Rule 19).
        enum class DeloadKind : uint8_t { Canonical = 0, Raw = 1 };
        DeloadKind deloadKind = DeloadKind::Canonical;

        // Payload byte count of the LAST raw image this LB wrote (page + byte-
        // bump bytes). 0 until the first raw dump; stale but harmless after a v3
        // dump (the pager only consults it on the raw path). Read via the
        // lastRawImageBytes() accessor.
        int64_t lastRawImageBytes_ = 0;

        /// @brief Payload byte count of the last v4 raw image this LB wrote —
        ///        the working-set pager's victim-ranking input.
        ///
        /// @details
        /// The reload cost model is `lastRawImageBytes() / readBW`, with
        /// `readBW` derived from the aggregate `DeloadStats` raw-load counters:
        /// ranking eviction victims by ACTUAL image cost (rather than block
        /// count alone) lets the policy prefer the cheapest-to-reload victims.
        /// Readable whether the LB is resident or deloaded (a plain scalar in
        /// the never-deloaded shell, not a container — no residency assert). 0
        /// before the first raw dump; a v3 dump leaves it stale (the pager only
        /// consults it for raw-format LBs). Pure telemetry-shaped policy input,
        /// never a determinism-gated value.
        ///
        /// @return Bytes of the last raw image (page + byte-bump); >= 0.
        int64_t lastRawImageBytes() const { return lastRawImageBytes_; }

        // Extent-file placement of this LB's v4 RAW eviction image
        // (D-195 extent datapath). REPLACES the raw
        // path's use of deloadFiles: the LB owns ONE slab in the single
        // preallocated extent file for its active life. rawExtentOffset_ = the
        // slab's byte offset (-1 = no slab yet); rawExtentClassBytes_ = the
        // slab's capacity (the size class), so a stable-size re-dump overwrites
        // in place and only a growth past the class reallocates. The image
        // LENGTH needs no metadata — the reload seeks to rawExtentOffset_,
        // reads the self-describing header, and recovers the geometry, exactly
        // as the named-file raw path did. Maintained by the extent wiring
        // (dumpStaticContainersRaw / discharge-free); plain shell scalars,
        // readable while the LB is cold. Freed to the allocator on discharge
        // (Raw->Canonical) and at batch-start purge.
        int64_t rawExtentOffset_ = -1;
        int32_t rawExtentClassBytes_ = 0;
        // The extent epoch the slab was allocated in. A slab is valid ONLY
        // while this equals staticMemory().extentEpoch(); a mismatch means the
        // slab is from a purged prior batch (offsets recycled from 0) and the
        // next dump must allocate a fresh slab rather than overwrite in place.
        int64_t rawExtentEpoch_ = -1;
        // Image bytes (header + payload) this LB currently occupies in the
        // extent file — the per-LB term of the extentLiveBytes telemetry, kept
        // in sync by a delta on each extent dump and zeroed on slab free.
        int64_t lastExtentLiveBytes_ = 0;
        // Tail-delta bookkeeping (the WAL/threshold pattern): row total
        // of the last FULL dump, rows accumulated across tail files
        // since it, and the number of tail file-sets written (also the
        // next tail's name index). A restructuring mutation, tails
        // exceeding the compaction fraction of the base, or too many
        // tail sets trigger the next full canonical rewrite.
        int64_t deloadBaseRows = 0;
        int64_t deloadTailRows = 0;
        int32_t deloadTailCount = 0;
        // Element count per statified container (tag order), recorded at
        // deload — the ONLY legal metadata about a cold LB's containers
        // (size() asserts residency, I-111).
        // POSITIONAL: built by LbMemory::visitContainers order. Indexing it by a
        // raw ContainerTag value is valid ONLY below the retired 35-37 gap
        // (intToBeProved moved to the persistent pool); a tag >= 38 indexed
        // directly would be off by 3. The one tag indexer below uses tag 0; any
        // new indexer must be position-aware, not tag-value-keyed.
        std::vector<int32_t> deloadedCounts;
        // Set when the LB leaves the active set (the kernel's end-of-
        // iteration sweep) — deactivation is permanent: no path flips
        // isActive back, so a flagged LB never runs again
        // (I-112). Distinguishes "was deactivated"
        // from "was never active": induction zero-condition blocks parked
        // at birth carry isActive == false but NOT this flag — they wake
        // through activateZeroCondition, which asserts against it. The
        // flag is set by dischargeStatementContent, which also empties
        // the statified containers (their content is dead; the post-
        // prove gates probe exact RAM records instead) — ensureLoaded
        // asserts against the flag: nothing ever reloads a discharged LB.
        bool dischargedForever = false;

        // Exact (originalId, validityId) pair set of intEncodedStatements
        // at discharge time — the post-prove equality-node gate's probe
        // target (the registry itself is emptied at discharge). Captured
        // by dischargeStatementContent; exact BY CONSTRUCTION, unlike the
        // intKnownStatements registered bit, which keeps
        // canonicalization-erased rows and would answer as a superset
        // (D-157).
        std::unordered_set<int32_t> dischargedRegistryKeys;

        /// @brief The discharge protocol's memory step: capture the
        ///        exact registry record, empty every dischargeable
        ///        container, and reshuffle — every block except the
        ///        cold-string survivors' returns to the pool with zero
        ///        I/O.
        ///
        /// @details
        /// Runs single-threaded at the kernel barrier, on the iteration
        /// the LB leaves the active set (resident by construction — it
        /// just ran its phases). Order: set `dischargedForever`; copy
        /// every `intEncodedStatements` row's packed
        /// `(originalId, validityId)` key into
        /// `dischargedRegistryKeys` (the equality-node gate's exact
        /// probe target; the local-origin gate probes the existing
        /// `intLocalEncodedStatementsSet` mirror, which survives as-is);
        /// `clearDischargeableContainers` (the cold string tags survive —
        /// `LbMemory::survivesDischarge`); `LbMemory::reshuffle` — the
        /// copying compaction packs the surviving string content onto a
        /// minimal block prefix and returns everything else (see
        /// `LbMemory::reshuffle`). The LB stays RESIDENT; the pressure-lazy
        /// pending dump later writes the near-empty canonical image
        /// (statement tags empty, string tags full), and reload never
        /// happens (I-112).
        ///
        /// @invariant After return: `dischargedForever` set, every
        ///            dischargeable container empty, manager resident.
        ///
        /// @param scr A thread-exclusive scratch arena (never-deloaded pool) for
        ///            the compaction the discharge triggers (see
        ///            `LbArena::compactPages`).
        void dischargeStatementContent(LbArena& scr);

        /// @brief Who may touch this LB's statified storage during the
        ///        steward's asynchronous execution windows — the single
        ///        word of the eviction claim handshake
        ///        (I-122).
        ///
        /// @details
        /// `Idle`: no steward business this iteration. `Planned`: barrier-
        /// selected eviction victim, dump not started. `Busy`: the steward
        /// is mid-operation on this LB. `Dumped`: the steward dumped and
        /// released it; the LB is cold until its phase-1 slot reloads it.
        /// `WorkerOwned`: the LB's phase-1 worker claimed it — the steward
        /// is locked out for the rest of the iteration. Writers: the
        /// kernel entry installs `Planned` (single-threaded); the steward
        /// and the phase-1 handshake move states only through
        /// compare-and-swap; the barrier fold resets terminal states to
        /// `Idle` (single-threaded, post-quiesce).
        enum class StewardClaim : uint8_t {
            Idle = 0,
            Planned = 1,
            Busy = 3,
            Dumped = 4,
            WorkerOwned = 5,
        };

        // The claim word itself. Plain default Idle — CE clones and every
        // LB outside an eviction plan never leave it.
        std::atomic<uint8_t> stewardClaim{
            static_cast<uint8_t>(StewardClaim::Idle) };

        /// @brief Statement-registry element count, readable whether the
        ///        LB is resident or deloaded.
        ///
        /// @details
        /// Resident: the live `size()`. Deloaded: the count recorded by
        /// `deloadStaticContainers` — both branches are defined results
        /// of the same logical question (the registry's content is
        /// unchanged by the SSD round-trip), not a fallback. Exists for
        /// between-iteration reporting that must not reload every LB
        /// just to count.
        ///
        /// @return Number of rows in `intEncodedStatements`.
        int32_t intEncodedStatementsCount() const {
            if (lbMemory.manager.resident())
                return intEncodedStatements.size();
            assert(!deloadedCounts.empty()
                && "deloaded LB without recorded counts");
            return deloadedCounts[static_cast<std::size_t>(
                LbMemory::ContainerTag::IntEncodedStatements)];
        }

        /// @brief Assign this LB's deload ordinal on first sight and record
        ///        its `ordinal -> chain` mapping in the global registry.
        ///
        /// @details
        /// The single-assignment guard (`deloadOrdinal < 0`) makes the
        /// ordinal lifetime-stable: a base dump, its tail dumps, a reload
        /// and a re-dump all share one file-name identity. Idempotent —
        /// both branches are defined results (cache hit / first compute),
        /// not a fallback (Rule 19). Stamped single-threaded at the kernel
        /// barrier (discharge enqueue, eviction-plan build, the pre-pass
        /// before the parallel synchronous-eviction sweep) BEFORE any
        /// worker-self-service or steward dump runs, so both possible
        /// dumpers name the file from the same stored ordinal. Also called
        /// first by `dumpStaticContainers` as the fallback for direct
        /// (test / teardown) callers. Re-registers every call so the
        /// per-batch-cleared registry always carries the current LB.
        ///
        /// @invariant After return: `deloadOrdinal >= 0`, and the global
        ///            registry maps it to `buildLbChainString(*this)`.
        void ensureDeloadOrdinal();

        /// @brief Write-through dump: bring the on-disk file set up to
        ///        date with the in-memory content (skip / tail / full
        ///        per the dirty state) WITHOUT releasing any blocks.
        ///
        /// @details
        /// The kernel calls this for every active LB at iteration end —
        /// the image stays fresh, so a later `releaseStaticBlocks` (the
        /// pressure path) needs no further writing, and a resident LB's
        /// next `ensureLoaded` is a no-op. Resets the dirty state to
        /// `Clean` (RAM == disk holds afterwards). Asserts residency.
        ///
        /// @param directory The deload directory (production: `.deload`).
        void dumpStaticContainers(const std::string& directory);

        /// @brief Return the LB's blocks to the pool and flag it
        ///        deloaded — legal only when the on-disk image is fresh.
        ///
        /// @details
        /// The pressure path: asserts residency, a recorded file set,
        /// and `DirtyState::Clean` (i.e., `dumpStaticContainers` ran
        /// after the last mutation — releasing stale-imaged blocks would
        /// lose content). Releases every statified container's pages,
        /// returns the blocks, `markDeloaded()`.
        void releaseStaticBlocks();

        /// @brief Dump + release in one call — the unconditional-deload
        ///        composition (tests, teardown paths).
        ///
        /// @param directory The deload directory (production: `.deload`).
        void deloadStaticContainers(const std::string& directory);

        /// @brief Write-through v4 RAW dump: bring the on-disk raw image up to
        ///        date with the in-memory content WITHOUT releasing any blocks
        ///        — the near-memcpy eviction dump.
        ///
        /// @details
        /// The eviction twin of `dumpStaticContainers`. Captures the per-
        /// container `deloadedCounts` (the only legal cold metadata) BEFORE the
        /// dump, `ensureDeloadOrdinal`, and — unless the LB is `Clean` with a
        /// fresh raw base (the skip-clean branch, a defined cache-hit result) —
        /// streams the whole live page tier to `lb<ordinal>_raw.bin` via
        /// `lbdeload::dumpLbMemoryRaw`, sets `deloadFiles = { rawName }`,
        /// `deloadKind = Raw`, records `lastRawImageBytes`, and resets the v3
        /// tail bookkeeping (the raw path never writes tails). Leaves
        /// `dirty = Clean`. Asserts residency. The bytes are nondeterministic
        /// (I-103 waived for eviction images) but the restored logical state is
        /// byte-identical.
        ///
        /// @param directory The deload directory (production: `.deload`).
        void dumpStaticContainersRaw(const std::string& directory);

        /// @brief Return the LB's blocks to the pool and flag it deloaded,
        ///        WITHOUT the container-release walk — the raw eviction's
        ///        release step.
        ///
        /// @details
        /// The near-memcpy win: a v3 `releaseStaticBlocks` walks every
        /// container's `release()` (resetting `rootVid_`/`size_`/`numPages_`)
        /// so the image can be rebuilt element-by-element; the raw path SKIPS
        /// that walk entirely — the raw image preserves the vids, so a raw
        /// reload rebinds the same bookkeeping. Just `manager.releaseAll()` +
        /// `markDeloaded()`. Asserts residency, a fresh raw image
        /// (`deloadKind == Raw`, `dirty == Clean`). The container scalars
        /// survive on the deloaded arena until the raw reload (or the teardown
        /// residency branch in `~PagedVector` / `~PagedHashIndex` frees
        /// nothing at destruction; a LIVE `clear()` there still asserts).
        void releaseStaticBlocksRaw();

        /// @brief Release the LB's blocks through the path matching the
        ///        recorded on-disk format — the release-side mirror of
        ///        `reloadFromImage`'s dispatch.
        ///
        /// @details
        /// Dispatches on `deloadKind` exactly as the reload does: `Raw` →
        /// `releaseStaticBlocksRaw` (NO container-release walk — the raw image
        /// preserves the vids, so the container bookkeeping MUST survive for
        /// the raw rebind; a v3 release here would reset
        /// `rootVid_`/`size_`/`numPages_` and a later raw reload would rebind
        /// dead bookkeeping), `Canonical` → the v3 `releaseStaticBlocks`
        /// (container walk, element-by-element rebuild on reload). For callers
        /// that release an LB they did not dump themselves — the chapter
        /// export's per-theorem G-53 release of `g_exportReloadSink` entries is
        /// the production case — and therefore cannot know which format the
        /// last dump wrote. Callers inside a same-format dump+release
        /// composition (`deloadStaticContainers` / `deloadStaticContainersRaw`)
        /// keep calling their own path directly; each path's preconditions stay
        /// asserted unweakened in the path itself.
        void releaseStaticBlocksDispatch();

        /// @brief Raw dump + raw release in one call — the unconditional
        ///        raw-deload composition (tests, the synchronous eviction).
        ///
        /// @param directory The deload directory (production: `.deload`).
        void deloadStaticContainersRaw(const std::string& directory);

        /// @brief Bring a deloaded LB's statified containers back from
        ///        the file set recorded at deload; no-op when resident.
        ///
        /// @details
        /// The resident no-op is a defined contract (the touch points
        /// call unconditionally), not a fallback. On the reload path:
        /// asserts the chain still matches the LB's tree position,
        /// `markResident()`, then `lbdeload::loadLbMemory` rebuilds every
        /// container element-by-element onto a fresh consecutive virtual
        /// index (the approved "straightening").
        ///
        /// @param directory The deload directory the files live in.
        void ensureLoaded(const std::string& directory);

        /// @brief Sanctioned post-prove READ reload — like `ensureLoaded`
        ///        but legal on a discharged LB
        ///        (D-158).
        ///
        /// @details
        /// The chapter export reads origin history in full; the strings
        /// campaign moved its strings into the deloadable cold image, so
        /// a discharged-and-drained LB must come back for its chapter
        /// walk. Does NOT reactivate: `isActive` / `dischargedForever`
        /// untouched; no kernel runs after the export. Defined no-op when
        /// resident.
        ///
        /// @param directory The deload directory the files live in.
        void ensureLoadedForRead(const std::string& directory);

    private:
        /// @brief Shared reload core of the two ensure variants: chain
        ///        check, mark resident, element-by-element rebuild, dirty
        ///        reset.
        ///
        /// @param directory The deload directory the files live in.
        void reloadFromImage(const std::string& directory);

    public:

        // The int mirrors of the local / delta vectors — statified like
        // the registry above; same pure reference aliases into
        // `lbMemory`.
        PagedVector<IntEncodedExpr>& intLocalEncodedStatements =
            lbMemory.intLocalEncodedStatements;
        PagedVector<IntEncodedExpr>& intLocalEncodedStatementsDelta =
            lbMemory.intLocalEncodedStatementsDelta;
        /// @brief Statements absorbed from EXTERNAL mail this burst (int16
        ///        rows).
        ///
        /// @details Companion of `workingMemory` for the non-implication
        ///          side: the facts this LB's external (`mailIn`) traffic
        ///          deposits during the post-hashburst absorption phase.
        ///          Emptied and refilled from fresh external mail every burst
        ///          (NOT a persistent local container). Consumed by the
        ///          static request-generation pipeline (`mandatorySrcInt` /
        ///          pair sources). Statified reference alias into
        ///          `lbMemory` like its siblings.
        PagedVector<IntEncodedExpr>& intExternalStatements =
            lbMemory.intExternalStatements;

        // Packed-key statement registry: packStatementKey(originalId,
        // validityId) -> StatementFlags. Carries TWO memberships via the
        // flag bits — `registered` (add-path registration) and `known`
        // (level-registry admission; the Site F dedup record). Gates test
        // the bit their contract
        // names, never bare presence. Writes go through upsertStatementKey.
        TypedColdMap<StatementKey, StatementFlags>& intKnownStatements =
            lbMemory.intKnownStatements;

        // Packed-key local-origin membership index:
        // packStatementKey(originalId, validityId) per local statement,
        // maintained beside intLocalEncodedStatements at every mutation
        // site. O(1) probe for the D-29 locality gate inside
        // checkLocalEncodedMemoryStatic and the applyEquivalenceClass
        // absorption tests; string-keyed probes go through the non-minting
        // isLocalEncodedStatement.
        ColdHashSet<PodKeyStore<int32_t>>& intLocalEncodedStatementsSet =
            lbMemory.intLocalEncodedStatementsSet;

        // Packed-key per-statement LB-level index:
        // packStatementKey(originalId, validityId) -> a SORTED-UNIQUE set of LB
        // levels at which the statement was deposited. Statified onto the
        // cold-map family's set form (Batch 2) — a reference alias into
        // lbMemory; the run is kept ascending so the hashburst dump prints
        // levels in order after a decode + lex-sort of the keys (Rule 14
        // byte-identity contract). String-keyed probes go through the
        // non-minting lookupStatementLevels (returns the cold key id).
        TypedColdSetMap<StatementKey, int>& intStatementLevelsMap =
            lbMemory.intStatementLevelsMap;

        // Packed-key goal registry: packStatementKey(originalId, validityId)
        // -> a SORTED-UNIQUE set of auxies (induction-discharge bookkeeping
        // indices). On the cold-map family's set form, but an OWNED member bound
        // to the PERSISTENT arena (persistentArena, above) — NOT the deloadable
        // lbMemory arena — so it stays resident for the LB's whole active life
        // and the deactivation survey reads it while the main arena is deloaded
        // (the determinism fix). It is in NO deload stream (the persistentDirty
        // flag it carries is never consulted), and is reclaimed at discharge.
        // Point probes with ids in hand pack directly; string-keyed probes go
        // through the non-minting lookupToBeProved (returns the cold key id);
        // order-sensitive walks go through decodeToBeProvedSorted (lex order on
        // the decoded (original, validityName) pair — never raw id order). The
        // dump section prints a literal `tags={}` per row.
        TypedColdSetMap<StatementKey, int> intToBeProved{
            &persistentArena, &persistentDirty };

        // Int mirrors for static addExprToMemoryBlock early checks. Statified
        // onto the cold-map family (Batch 1); reference aliases into lbMemory.
        ColdHashSet<PodKeyStore<int16_t>>& intValidityNamesToFilter =
            lbMemory.intValidityNamesToFilter;
        ColdHashSet<PodKeyStore<int16_t>>& intAxedVariables =
            lbMemory.intAxedVariables;

        /// @brief Insert an outgoing statement from owned expression strings.
        ///
        /// @details Both strings are minted into this LB's dedicated
        /// `mailOutInterner`; the resulting ids and levels are inserted into the
        /// deloadable mailbox, then `mailOutPending` is raised for the serial
        /// commit sweep.
        ///
        /// @param ev Outgoing expression and validity.
        /// @param levels The statement's ascending level set.
        /// @return Nothing.
        /// @invariant The LB is resident and claimed by the caller.
        /// @see clearMailOut, addMailOutOrigin.
        void insertMailOutStatement(const ExpressionWithValidity& ev,
            const std::set<int>& levels) {
            const int32_t originalId = mailOutInterner.encode(ev.original);
            const int32_t validityId = mailOutInterner.encode(ev.validityName);
            mailOut.insertStatement(originalId, validityId, levels);
            mailOutPending = true;
            mailOutLiveBytes = mailOut.liveBytes();
        }

        /// @brief Insert an outgoing statement from spans and a level run.
        ///
        /// @details This is the heap-free phase-3 twin of the owned-string
        /// overload: it mints the two spans into the private mailbox interner,
        /// deposits the ascending-unique level run, and raises the resident
        /// pending bit.
        ///
        /// @param original Outgoing expression bytes.
        /// @param validityName Outgoing validity bytes.
        /// @param levels Pointer to `levelCount` ascending-unique levels; null
        ///               only when `levelCount` is zero.
        /// @param levelCount Number of levels.
        /// @return Nothing.
        /// @invariant Input spans do not alias `mailOutInterner` storage across a
        ///            mint; the LB is resident and claimed.
        /// @see insertMailOutStatement(const ExpressionWithValidity&, const std::set<int>&).
        void insertMailOutStatement(const StrSpan& original,
            const StrSpan& validityName, const int* levels,
            int32_t levelCount) {
            const int32_t originalId = mailOutInterner.encode(original);
            const int32_t validityId = mailOutInterner.encode(validityName);
            mailOut.insertStatement(originalId, validityId, levels, levelCount);
            mailOutPending = true;
            mailOutLiveBytes = mailOut.liveBytes();
        }

        /// @brief Append an owned outgoing origin line.
        ///
        /// @details The key and every dependency are minted into the same
        /// private mailbox interner as outgoing statements, then the capped
        /// origin fold runs and the resident pending bit is raised.
        ///
        /// @param ev Origin-map key.
        /// @param origin Tagged dependency line.
        /// @param maxOrigins Per-key retained-line cap.
        /// @return Nothing.
        /// @invariant The LB is resident and claimed by the caller.
        /// @see addDeloadableMailOutOrigin, clearMailOut.
        void addMailOutOrigin(const ExpressionWithValidity& ev,
            const OriginLine& origin, int maxOrigins) {
            addDeloadableMailOutOrigin(mailOut, mailOutInterner, ev, origin,
                                       maxOrigins);
            mailOutPending = true;
            mailOutLiveBytes = mailOut.liveBytes();
        }

        /// @brief Append a span-form outgoing origin line.
        ///
        /// @details The heap-free producer door mints the key and dependency
        /// spans into the private mailbox interner, performs the capped fold,
        /// and raises the resident pending bit.
        ///
        /// @param original Origin-key expression bytes.
        /// @param validityName Origin-key validity bytes.
        /// @param tag Origin tag.
        /// @param deps Pointer to `depN` dependency spans.
        /// @param depN Dependency count.
        /// @param maxOrigins Per-key retained-line cap.
        /// @return Nothing.
        /// @invariant Input spans do not alias `mailOutInterner` storage across a
        ///            mint; the LB is resident and claimed.
        /// @see addMailOutOrigin(const ExpressionWithValidity&, const OriginLine&, int).
        void addMailOutOrigin(const StrSpan& original,
            const StrSpan& validityName, OriginTag tag, const OriginDep* deps,
            int depN, int maxOrigins) {
            addDeloadableMailOutOrigin(mailOut, mailOutInterner, original,
                validityName, tag, deps, depN, maxOrigins);
            mailOutPending = true;
            mailOutLiveBytes = mailOut.liveBytes();
        }

        /// @brief Clear a delivered outgoing mailbox and its summary bit.
        ///
        /// @details The id-bearing columns and their dedicated interner reset as
        /// one unit before the shell bit is lowered. A subsequent producer starts
        /// a fresh private id space and raises the bit through a write door.
        ///
        /// @return Nothing.
        /// @invariant The LB is resident and claimed, or is in single-threaded
        ///            construction/teardown before steward ownership begins.
        /// @see insertMailOutStatement, addMailOutOrigin.
        void clearMailOut() {
            mailOut.clear();
            mailOutPending = false;
            mailOutLiveBytes = 0;
        }

        // No explicit ~Memory: the four HashMemory instances now live in lbMemory
        // (declared after its arena), so they destruct before that arena
        // naturally and the former releaseAllCold teardown-order safeguard is
        // retired (D-147).
        Memory()
            : startInt(0),
            startIntRepl(0),
            startIntPi(0),
            recursionHypothesisId(0),
            parentMemory(nullptr),
            level(-1),
            isActive(true),
            isPartOfRecursion(false),
            deltaNumberStatements(0),
            recursionCounter(0),
            contradictionIndex(-1),
			templateInterner(),
			valueInterner(),
			primedForContradiction(false),
			contradictionTheoremId(0),
            nameMap()
        {
            // Bind the interner façades to their cold tables (all members
            // are constructed by ctor-body time; the façades carry only
            // this pointer, so clones and teardown temporaries bind to
            // their OWN aggregates automatically).
            templateInterner.bind(&lbMemory.templateStrings);
            valueInterner.bind(&lbMemory.valueStrings);
            originInterner.bind(&lbMemory.originStrings);
            mailOutInterner.bind(&lbMemory.mailOut.strings_);
            ruleInterner.bind(&lbMemory.ruleStrings);
            lbStateInterner.bind(&lbMemory.lbStateStrings);
            nameMap.bind(&lbMemory.nameStrings, &lbMemory.subStrings,
                         &lbMemory.validityNodes);
        }

        /// @brief Radical subtree wipe after a subproof closure.
        ///
        /// @details
        /// When the prover successfully discharges an implication / subproof
        /// whose rooted scope is the `NameMap` validity id `closedVid`
        /// (decoding to a name typically of the form
        /// `main_boundary_…_boundary_(implicationN[…])` minted via
        /// `NameMap::encodePush`), every piece of per-LB state whose attached
        /// validity equals the closed scope's name or starts with that name
        /// + `"_boundary_"` is physically removed.
        ///
        /// Replaces the pre-existing selective cleanup
        /// (`cleanUpIntegrationPreparation` + `cleanUpIntegrationPreparationCore`)
        /// which only erased the rooted scope itself and its direct
        /// `_var0_*` fresh-binding children, leaving deeper `_orint_…` /
        /// nested-hypothesis state as inert orphans behind the
        /// `intValidityNamesToFilter` ancestor-scan filter.
        ///
        /// Eradicated (per LB):
        /// - `intToBeProved`, `intEncodedStatements`
        /// - `intLocalEncodedStatements` (+ delta) and the parallel
        ///   `intLocalEncodedStatementsSet`
        /// - `intStatementLevelsMap`, `intKnownStatements`
        /// - `equivalenceClassesMap`
        /// - `expandedImplications`, `integrationPrepared`,
        ///   `integrationPreparedMarker`, `weakVariables`
        /// - `mailIn.statements`, `mailOut.statements`,
        ///   `sameIterationInternalMail.statements`
        /// - Per HashMemory (overall / local / delta):
        ///   - `encodedMap` LMVs by `lmv.validityName`
        ///   - Four owner-set maps (`normalizedEncodedKeys`,
        ///     `…Subkeys`, `…SubkeysMinusOne`, `…SubkeysMinusTwo`) by
        ///     owner-set; empty owner-set drops the key. Dropped keys are
        ///     also pruned from `remainingArgsNormalizedEncodedMap`.
        ///   - `admissionMap`, `admissionMapIntegration`,
        ///     `admissionSetIntegration`, `triggersForAdmissionSetIntegration`,
        ///     `rejectedMap`, `rejectedMapIntegration`,
        ///     `admissionStatusMap`, `consumedAdmissionKeys`,
        ///     `revisitInProgress`.
        ///
        /// Preserved on purpose:
        /// - `exprOriginMap` (per Rule 16 / I-44 — chapter export reads it).
        /// - `nameMap`'s `stackOfValidity` / `ancestorsOf` (validity
        ///   registries grow monotonically; never pruned).
        /// - `orDisjunctCount` (keyed by OR signature, no scope dimension).
        /// - `orBookkeeping` (the value-set holds branch-disjunct payload
        ///   strings; the key has no parent scope per the known cross-stack
        ///   same-orSignature gotcha — partial filtering deferred).
        /// - `originals` in HashMemory (no scope tag).
        /// - `integrationStartIntMap` (counter snapshots, no scope tag).
        ///
        /// At the end of the sweep every closed scope's int id is inserted
        /// into `intValidityNamesToFilter` so any in-flight mail referencing
        /// a descendant scope is blocked at the Site H ancestor-scan on
        /// absorb (belt-and-suspenders).
        ///
        /// @param closedVid The `NameMap` validity id of the implication /
        ///                  subproof scope that just closed. Must be a
        ///                  minted id (`1 <= closedVid <= nameMap.nameCount()`)
        ///                  and never `MAIN_ID` (`"main"` is never closed).
        ///                  Membership is the `validityNodes` forest walk
        ///                  (`collectClosedSubtreeIds` — provably equivalent
        ///                  to the retired string-prefix-with-`_boundary_`
        ///                  predicate); only the `expandedImplications`
        ///                  sweep, whose scope half lives in the
        ///                  `lbStateInterner` id space, keeps the text gate
        ///                  as a zero-copy span twin.
        ///
        /// @pre  `closedVid` is a minted, non-main `NameMap` validity id —
        ///       assert-enforced at entry (a `"main"` wipe would be an
        ///       unbounded sweep; callers must not invoke one).
        /// @post All per-LB scope-tagged structures are free of entries
        ///       whose validity matches the closed subtree.
        ///       `intValidityNamesToFilter` contains the closed subtree's
        ///       int ids.
        /// @invariant [I-44](../../docs/agentic_swdd/30_invariants.md#i-44) — `exprOriginMap`
        ///            is not touched.
        void wipeSubtree(int16_t closedVid);

        /// @brief Build a fresh, single-use CE logic block from a facts-only
        ///        template LB.
        ///
        /// @details
        /// The counterexample filter loads its fact base once into a "template"
        /// LB via `loadFactsForCEFiltering`, which adds every fact through the
        /// status-4 path of `addExprToMemoryBlock` — a statement-only fast path
        /// that fills the value-type statement containers and the `nameMap` but
        /// never calls `addToHashMemory`, so the template's `overallHashMemory`
        /// stays empty. Each conjecture then runs on its own
        /// throwaway clone of that template; this routine produces one clone.
        ///
        /// Because the template carries no hash rules, the clone is a plain deep
        /// value-copy of the fact containers and the `nameMap` — all pure value
        /// types (`EncodedExpression`, the all-`int16_t` `IntEncodedExpr`, the
        /// packed-`int32_t` statement keys, and the pointer-free `NameMap`) — with
        /// every hash / mail / equivalence / origin / derived container left at its
        /// `Memory()` default. The owner-set maps and the `remainingArgs` index now
        /// key on owning `NormKey`s (keyArena retired), so no dangling-interior-
        /// pointer hazard of a naive whole-`Memory` copy arises. The clone's
        /// `nameMap` is an independent copy, so the per-conjecture run may mint the
        /// conjecture's new names without disturbing the shared template.
        ///
        /// The returned block is allocated from the LB object store (`store`, the
        /// never-deloaded shell pool — [I-109]) and owned by the
        /// caller, which sets its identity (`parentMemory`, `exprKey`,
        /// `contradictionIndex`) and returns it via `store.destroy()` once the
        /// conjecture is resolved.
        ///
        /// @param store The LB object store the clone's shell is carved from. The
        ///              CE-clone path runs on worker threads, so `LbStore::create`
        ///              is mutex-guarded.
        /// @return An `LbStore`-allocated `Memory*` deep-cloned from `*this`; the
        ///         caller owns it and must release it via `store.destroy()`.
        /// @invariant `*this` carries no hash rules — asserted on
        ///            `overallHashMemory.encodedMap`. A future fact load that
        ///            installs rules must extend this routine to rebuild them into
        ///            the clone rather than value-copying the cold arena-backed
        ///            containers.
        /// @see `loadFactsForCEFiltering` — the status-4 template loader.
        /// @see `filterConjecturesWithCE` — the per-conjecture pool driver that
        ///      clones the template once per conjecture.
        Memory* cloneFactsTemplate(LbStore& store) const;

    };


    /// @brief Best replacement peer for one downprioritized `it_/int_` arg
    ///        across the visible scopes — the shared scan core of the two
    ///        end-of-burst sanitize twins (`sanitizeToBeProved` /
    ///        `sanitizeHashMemory`), heap-free.
    ///
    /// @details
    /// Priority rule (unchanged from the twins' historical inline loops): an
    /// `int_lev_*` peer outranks an `it_*_lev_*` peer; within a tier the
    /// lex-smallest name wins; `repl_*` and plain names never rank (skipped).
    /// The heap `decodeClassesAt` snapshot is replaced by the zero-copy cold
    /// blob walk. Byte-order contract:
    /// 1. Class iteration order: `peekRecordBytes(bucketId, j, …)` for `j`
    ///    ascending is the blob run order — the same order `recordsAt` (and
    ///    hence the retired `decodeClassesAt` snapshot) handed the loop.
    ///    Member order: `memberId(k)` reads the blob's stored member run,
    ///    positionally identical to the deserialized `memberIds` vector. So
    ///    every (scope, class, peer) triple is visited in EXACTLY the
    ///    historical sequence and the running best takes the identical value
    ///    sequence — identical final bytes.
    /// 2. The per-class skip `argId == 0 || !classHasMember(cls, argId)` is
    ///    the documented `std::find` twin, at the same nesting level as the
    ///    historical `continue`.
    /// 3. `compareSpans(peer, best) < 0` is the byte-lex `peer < bestPeer`;
    ///    the empty-`bestPeer` sentinel maps to `len == 0` — including the
    ///    degenerate case where an interned empty-named member wins a round
    ///    (the sentinel then still reads "empty" and the next peer
    ///    unconditionally wins; the twin reproduces this faithfully rather
    ///    than "fixing" it).
    /// 4. `kindOf` writes its memo column (`kindById_`, a `PagedVector` on
    ///    the LB's deloadable arena) — a DIFFERENT container from the
    ///    NameMap byte pool, so held `decodeView` spans stay valid
    ///    ([I-3](../../docs/agentic_swdd/30_invariants.md#i-3)
    ///    different-container rule); that write already happened in the
    ///    historical loop.
    /// 5. No mint anywhere: both `lookup`s are non-minting, the peeks are
    ///    read-only, `decodeView` is read-only.
    /// 6. Straddle peeks accumulate on @p peekArena's byte-bump tier with NO
    ///    rewind, so an outer span is never clobbered by a later peek.
    ///
    /// @param mb              Owning LB — classes map, NameMap, name-kind
    ///                        cache.
    /// @param argId           The arg's NameMap id; 0 (never interned) is in
    ///                        no class, yielding the empty result.
    /// @param scopes          The scope names to search, in visit order (own
    ///                        validity first, then sorted strict ancestors).
    /// @param scopeCount      Number of scopes.
    /// @param peekArena       Byte-bump arena for page-straddling blob peeks
    ///                        (the caller's gen-scratch arena).
    /// @param bestPeerOut     [out] The winning peer's name span; `len == 0`
    ///                        when no ranked peer exists.
    /// @param bestPeerIsIntOut [out] Whether the winner is `int_lev_*` tier.
    /// @invariant Scan-mint-free; `bestPeerOut` aliases NameMap cold bytes —
    ///            valid until the caller's next NameMap mint (the caller
    ///            copies at staging). See `I-141`.
    /// @see EquivalenceClassView, classHasMember, EqClassNameCaches::kindOf,
    ///      sanitizeToBeProved, sanitizeHashMemory.
    inline void bestSanitizePeer(Memory& mb, int16_t argId,
                                 const StrSpan* scopes, int32_t scopeCount,
                                 ScratchArena& peekArena,
                                 StrSpan& bestPeerOut, bool& bestPeerIsIntOut) {
        bestPeerOut = StrSpan();          // len 0 == the bestPeer.empty() sentinel
        bestPeerIsIntOut = false;
        for (int32_t s = 0; s < scopeCount; ++s) {
            const int16_t vId = mb.nameMap.lookup(scopes[s]);
            if (vId == 0) continue;                       // decodeClassesAt's lookup-miss -> {}
            const int32_t bucketId = mb.equivalenceClassesMap.lookup(vId);
            if (bucketId == 0) continue;                  // absent bucket -> {}
            const int32_t rl = mb.equivalenceClassesMap.runLen(bucketId);
            for (int32_t j = 0; j < rl; ++j) {
                int32_t blen = 0;
                const char* bp = mb.equivalenceClassesMap.peekRecordBytes(
                    bucketId, j, blen, peekArena);
                const EquivalenceClassView cls{ bp, blen };
                if (argId == 0 || !classHasMember(cls, argId)) continue;
                for (int32_t k = 0; k < cls.memberCount(); ++k) {
                    const int16_t peerId = cls.memberId(k);
                    if (peerId == argId) continue;
                    const NameKind peerKind =
                        mb.eqClassNameCaches.kindOf(peerId, mb.nameMap);
                    if (peerKind == NameKind::Normal) continue;
                    const bool peerIsInt = (peerKind == NameKind::IntLev);
                    const StrSpan peer = mb.nameMap.decodeView(peerId);
                    bool peerWins = false;
                    if (bestPeerOut.len == 0) {
                        peerWins = true;
                    } else if (peerIsInt && !bestPeerIsIntOut) {
                        peerWins = true;
                    } else if (peerIsInt == bestPeerIsIntOut
                               && compareSpans(peer, bestPeerOut) < 0) {
                        peerWins = true;
                    }
                    if (peerWins) {
                        bestPeerOut = peer;
                        bestPeerIsIntOut = peerIsInt;
                    }
                }
            }
        }
    }

    /// @brief Stage one sanitize substitution pair into a caller stack array,
    ///        deduplicating on the key — the zero-heap twin of the twins'
    ///        `substMap[argName] = bestPeer` writes.
    ///
    /// @details
    /// Linear-scans the staged pairs for an `equalSpans` key hit. On a hit,
    /// asserts the staged value equals @p value and returns — the exact
    /// behavior of `std::map::operator[]` overwriting with an identical
    /// value, because the sanitize scan is PURE: a duplicate argName always
    /// recomputes the same bestPeer. The assert is the Rule-19 tripwire for
    /// that purity claim (a divergent duplicate means the scan read mutated
    /// state mid-row). On a miss, asserts capacity and appends.
    ///
    /// @param pairs     The staging array.
    /// @param pairCount [in/out] Live pair count; incremented on append.
    /// @param key       The arg name to replace (span over stable bytes).
    /// @param value     Its replacement peer (span over stable bytes).
    /// @param cap       Capacity of @p pairs.
    /// @see sortSanitizeSubstPairs — the observable-order finalizer;
    ///      bestSanitizePeer — the (pure) value producer.
    inline void addSanitizeSubstPair(StrReplacement* pairs, int32_t& pairCount,
                                     const StrSpan& key, const StrSpan& value,
                                     int32_t cap) {
        for (int32_t i = 0; i < pairCount; ++i) {
            if (equalSpans(pairs[i].key, key)) {
                assert(equalSpans(pairs[i].value, value)
                    && "duplicate arg resolved to a different peer — scan is not pure");
                return;
            }
        }
        assert(pairCount < cap
            && "addSanitizeSubstPair: pair count exceeds caller capacity");
        pairs[pairCount].key = key;
        pairs[pairCount].value = value;
        ++pairCount;
    }

    /// @brief Sort staged sanitize substitution pairs into ascending key
    ///        order — the `std::map<std::string,std::string>` iteration
    ///        order the twins' observable drains replay.
    ///
    /// @details
    /// The SUBSTITUTION itself is order-free in the pair array (the
    /// `replaceKeysToString` / `replaceKeysScratch` position scan takes the
    /// greedy longest key match at each token boundary; at a fixed position
    /// two DISTINCT keys of equal length cannot both match the same bytes,
    /// so the longest match is unique and the output is a pure function of
    /// the pair SET). The sort exists for the OBSERVABLE drain:
    /// `sanitizeHashMemory`'s history block iterates the pairs to build its
    /// `(=[k,v])` equality1 antecedents, and that sequence historically
    /// iterated a `std::map` in ascending-key order — an order that reaches
    /// `originInterner` id mints and hence the deload bytes. Ascending
    /// `compareSpans` == ascending `std::map<std::string, …>`. Keys are
    /// pairwise distinct after `addSanitizeSubstPair`'s dedup, so the order
    /// is strict (tie-free).
    ///
    /// @param pairs     The staged pairs.
    /// @param pairCount Live pair count.
    /// @see addSanitizeSubstPair; `ce::replaceKeysInString` — the order-free
    ///      substitution contract's origin.
    inline void sortSanitizeSubstPairs(StrReplacement* pairs,
                                       int32_t pairCount) {
        std::sort(pairs, pairs + pairCount,
            [](const StrReplacement& a, const StrReplacement& b) {
                return compareSpans(a.key, b.key) < 0;
            });
    }


    /// @brief One dependency record — used during the static-pipeline mandatory
    /// merge to track which auxiliary index an expression depends on.
    ///
    /// @details
    /// A `DependencyItem` is one entry in `Dependencies::originalAuxyMap`.
    /// `auxies` is the set of auxiliary indices the expression cares about;
    /// `expr` is the raw expression text; `allLevelsInvolved` is `true` when
    /// the dependency spans every LB level (overrides per-level filtering).
    /// Plain value type, no ordering.
    ///
    /// @see [`Dependencies`](#dependencies) — owns these.
    struct DependencyItem {
        std::set<int> auxies;
        std::string expr;
        bool allLevelsInvolved;

        DependencyItem()
            : auxies(),
            expr(),
            allLevelsInvolved(false) {
        }

        DependencyItem(const std::set<int>& auxies_,
            const std::string& expr_,
            bool allLevelsInvolved_)
            : auxies(auxies_),
            expr(expr_),
            allLevelsInvolved(allLevelsInvolved_) {
        }
    };

    /// @brief Bidirectional auxiliary-index ↔ original-expression map plus
    /// per-original induction-variable bookkeeping.
    ///
    /// @details
    /// Used by the static-pipeline mandatory merge and by induction-block
    /// expansion. The two maps are kept synchronized by the producer:
    ///
    /// - `auxyOriginalMap`               — auxiliary index → original index.
    /// - `originalAuxyMap`               — original index → `DependencyItem`
    ///   (carries the auxiliary-index set, the expression text, and the
    ///   all-levels flag).
    /// - `originalInductionVariableMap`  — original index → `(induction_var,
    ///   typing_set)` pair. Populated only for induction-target originals.
    /// - `auxyIndex`, `originalIndex`    — running counters; the producer
    ///   bumps these on every fresh insertion to keep the indices stable.
    ///
    /// @see [`DependencyItem`](#dependencyitem) — value type of `originalAuxyMap`.
    /// @see [I-18](../../docs/agentic_swdd/30_invariants.md#i-18) — induction scheduled on
    ///      a bound variable must first prove `(in[n, N])`. The typing set
    ///      lives in `originalInductionVariableMap`'s second tuple slot.
    struct Dependencies {
        std::map<int, int> auxyOriginalMap;
        std::map<int, DependencyItem> originalAuxyMap;
        std::map<int, std::tuple<std::string, std::string>> originalInductionVariableMap;
        int auxyIndex;
        int originalIndex;

        Dependencies()
            : auxyOriginalMap(),
            originalAuxyMap(),
            originalInductionVariableMap(),
            auxyIndex(0),
            originalIndex(0) {
        }

        Dependencies(const std::map<int, int>& auxyOriginalMap_,
            const std::map<int, DependencyItem>& originalAuxyMap_,
            const std::map<int, std::tuple<std::string, std::string>>& originalInductionVariableMap_,
            int auxyIndex_,
            int originalIndex_)
            : auxyOriginalMap(auxyOriginalMap_),
            originalAuxyMap(originalAuxyMap_),
            originalInductionVariableMap(originalInductionVariableMap_),
            auxyIndex(auxyIndex_),
            originalIndex(originalIndex_) {
        }
    };

    /// @brief One *variant* — a paired pair-of-normalized-keys + value text
    /// + mapping-items, used during equivalence-class application to
    /// dedupe rewrite emissions.
    ///
    /// @details
    /// When `applyEquivalenceClass` rewrites an expression, it produces a
    /// `(normalizedIgnoredKey, normalizedNotIgnoredKey, valueVariant,
    /// mappingItems)` tuple. The first two keys distinguish two normalization
    /// modes (with and without ignoring `u_`-prefixed args); the value text
    /// is the rewritten head; `mappingItems` carries the substitution edges
    /// the rewrite performed.
    ///
    /// `operator<` is lexicographic over all four fields so the variant fits
    /// directly into `std::set<VariantItem>` for dedupe.
    ///
    /// @see [I-30](../../docs/agentic_swdd/30_invariants.md#i-30),
    ///      [I-34](../../docs/agentic_swdd/30_invariants.md#i-34) — invariants on the
    ///      `applyEquivalenceClass` path that emits these.
    struct VariantItem {
        NormalizedKey normalizedIgnoredKey;
        NormalizedKey normalizedNotIgnoredKey;
        std::string valueVariant;
        std::set<std::pair<std::string, std::string>> mappingItems;

        bool operator<(const VariantItem& rhs) const {
            if (!(normalizedIgnoredKey == rhs.normalizedIgnoredKey)) {
                return normalizedIgnoredKey < rhs.normalizedIgnoredKey;
            }
            if (!(normalizedNotIgnoredKey == rhs.normalizedNotIgnoredKey)) {
                return normalizedNotIgnoredKey < rhs.normalizedNotIgnoredKey;
            }
            if (valueVariant != rhs.valueVariant) {
                return valueVariant < rhs.valueVariant;
            }
            return mappingItems < rhs.mappingItems;
        }
    };

    // ContradictionItem moved to filter.hpp (CE-only type).


    /// @brief The process-wide, never-deloaded skeleton string interner — the
    ///        home of the statified LB identity strings (`exprKey` and the
    ///        `SimpleMapStore` routing keys).
    ///
    /// @details
    /// A single `ColdStringTable` (`= ColdHashSet<BytesKeyStore>`) on a
    /// function-local-static `LbArena{ &lbMemory() }` — the never-deloaded
    /// LB-body pool. Analyzer-lifetime and shared across every LB: an LB's
    /// `exprKey` is interned here once (deduped) and stored on the LB as a
    /// 4-byte `exprKeyId`, recovered byte-identically by `Memory::exprKey()`.
    /// Never enumerated for deload (it backs the always-resident LB directory),
    /// so it stays readable while an LB's main arena is cold or the LB is
    /// discharged. NOT reset between batches — it is process-lifetime (each batch
    /// is a fresh process; the interned ids never escape, so accumulation is
    /// bounded). The `SimpleMapStore` routing keys intern here too
    /// (D-151, D-152).
    ///
    /// @return The singleton interner.
    /// @see `Memory::exprKey` / `Memory::setExprKey`, `lbMemory`.
    ColdStringTable& skeletonInterner();

    /// @brief The process-wide, never-deloaded GLOBAL mail string interner — the
    ///        home of the statified mail payload strings (expression text,
    ///        validity-scope names, origin labels) that cross LB boundaries in
    ///        the current retained delivery-history window.
    ///
    /// @details
    /// A single `ColdStringTable` (`= ColdHashSet<BytesKeyStore>`) on a
    /// function-local-static `LbArena{ &mailMemory() }` — the never-deloaded mail
    /// pool that already backs the cross-LB `MailLog`. It is the cross-LB
    /// counterpart of the per-LB `NameMap`: a string interned here gets ONE
    /// analyzer-wide 4-byte `int32_t` id that denotes the same fact at every LB, so
    /// a routing-mail batch travels as ids instead of inlined `std::string` bytes
    /// and a receiver decodes the SAME id the sender minted. The per-LB
    /// `NameMap` ids cannot serve this role — they are meaningful only
    /// inside the one LB whose map minted them, which is why mail historically
    /// re-encoded its payload at every seam
    /// ([I-91](../../docs/agentic_swdd/30_invariants.md#i-91)). The 4-byte width is
    /// deliberate: one table spanning every LB far exceeds the per-LB `NameMap`'s
    /// `int16_t` ceiling (`MAX_NAME_IDS` = 32000).
    ///
    /// Concurrency: minting is SINGLE-THREADED and confined to the cycle-end commit
    /// barrier (plus the load-time broadcast and the post-`pool.join()`
    /// `updateGlobal*` drains — all single-threaded). During the parallel worker
    /// phase the table is FROZEN and only `lookup`/`decode`d (read-safe), exactly
    /// the discipline the `MailLog` blobs already follow. Ids are never observable
    /// — every compare / sort / dump decodes first
    /// ([I-97](../../docs/agentic_swdd/30_invariants.md#i-97)) — so the mint order
    /// is invisible to every proof artifact. Never enumerated for deload. In an
    /// all-active grid, `resetMailInterner` clears the table after phase 3 has
    /// consumed every `mailIn` and `MailLog::retireDeliveredBatches` has removed
    /// every blob carrying its ids; the next commit remints a fresh window. A grid
    /// with any initially dormant LB preserves both the full mail log and this id
    /// space for later catch-up.
    ///
    /// @return The singleton mail interner.
    /// @see `skeletonInterner` — the sibling global interner for LB identity
    ///      strings; `mailMemory`, `MailLog`.
    ColdStringTable& mailInterner();

    /// @brief Clear the global mail interner at the rolling-history retirement
    ///        seam.
    ///
    /// @details
    /// Calls `ColdStringTable::resetToFresh`, invalidating all ids and returning
    /// the table's pages to its `LbArena` for reuse. This bounds retained interner
    /// storage by one delivery window. It is forbidden when a dormant LB may later
    /// decode an older `MailLog` blob.
    ///
    /// @return Nothing.
    /// @invariant No parallel proof phase is running; every routing `mailIn` is
    ///            empty and delivered `MailLog` blobs were retired first.
    /// @see mailInterner, MailLog::retireDeliveredBatches,
    ///      ExpressionAnalyzer::proveKernel.
    void resetMailInterner();

} // namespace gl
