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
namespace gl {
/// @brief Encoded OR-scope category.
enum class OrScopeKind { NotOrScope, Integration, Disintegration };
/// @brief Parse the existing goal-carrying and OR scope payload formats.
/// @details Operations read scope text without minting identifiers or changing storage.
class ScopeSyntax {
public:
static constexpr char kContradictionScopePrefix[] = "contradiction_";

/// @brief Fill `out` with OWNED scratch copies of a scope's strict
    ///        ancestors, ready to be probed across minting sinks.
    ///
    /// @details
    /// `NameMap::strictAncestorSpans` returns `decodeView` spans that alias
    /// the NameMap's cold byte pool — valid only until the next mint into
    /// that NameMap ([I-3](../../docs/agentic_swdd/30_invariants.md#i-3)).
    /// Every ancestor-walking consumer (the Pass-B ancestor-inclusive
    /// admission probe, the ancestor-park release) probes and then MINTS
    /// (`templateInterner`, `nameMap`, `originInterner`), so the spans must
    /// be copied first. This is the one shared copy step: the walk
    /// (`compareSpans` order, tie-free — I-140), then one
    /// `ScratchString::copyFrom` per ancestor onto the caller's string-tier
    /// scratch arena. The caller owns the `ScratchScope` window the copies
    /// live in.
    ///
    /// @param mb           Owning LB (its `nameMap` is read-only here).
    /// @param validityName The scope whose strict ancestors are wanted.
    /// @param out          Caller array of at least
    ///                     `ExecutionParameters::MAX_SCOPE_DEPTH` spans.
    /// @param sArena       The caller's string-tier scratch arena; the
    ///                     copies ride it inside the caller's `ScratchScope`.
    /// @return The ancestor count — 0 for `main` or an unknown name; the walk
    ///         asserts on nesting beyond `MAX_SCOPE_DEPTH`.
    /// @invariant Mints nothing; no `out[i]` aliases the NameMap.
    /// @see `NameMap::strictAncestorSpans` — the walk;
    ///      AdmissionMemory::isAdmittedIncludingAncestors,
    ///      RejectedSourceReplay::releaseAncestorRejected2,
    ///      RejectedSourceReplay::releaseAncestorRejectedIntegration2 — the
    ///      consumers.
static inline int32_t copyStrictAncestorSpans(const Memory& mb,
                                           StrSpan validityName,
                                           StrSpan* out,
                                           ScratchArena& sArena) {
        StrSpan anc[ExecutionParameters::MAX_SCOPE_DEPTH];
        const int32_t ancN = mb.nameMap.strictAncestorSpans(
            validityName, anc, ExecutionParameters::MAX_SCOPE_DEPTH);
        for (int32_t i = 0; i < ancN; ++i) {
            out[i] = StrSpan(ScratchString::copyFrom(
                sArena, anc[i].ptr, anc[i].len));
        }
        return ancN;
    }


/// @brief Strip the `<goal>_subproof_` prefix off an integration-scope
        ///        payload, or return the payload unchanged when it has none.
        ///
        /// @details
        /// Convenience form of @ref splitSubproofPayload for consumers that
        /// need only the bare legacy payload (the OR classifiers, the scoped
        /// goal-closure deposit). The returned span aliases `payload`'s bytes,
        /// so it inherits `payload`'s lifetime.
        ///
        /// @param payload The validity payload (top-of-stack sub-string).
        /// @return The bare payload slice on a goal-carrying payload; `payload`
        ///         itself otherwise.
        /// @see splitSubproofPayload — the splitting form with the goal slice.
static inline StrSpan stripSubproofPrefixView(StrSpan payload) {
            StrSpan goalOut, bareOut;
            if (splitSubproofPayload(payload, goalOut, bareOut)) return bareOut;
            return payload;
        }

        // --- OR scope classifier (single choke point for stack-based OR semantics) ---
        // Parses the top-of-stack payload at `validityId` and reports which OR mode
        // the scope belongs to (Integration vs Disintegration vs NotOrScope).
        //
        // Payload grammar (self-describing — no Memory side-tables):
        //   Disintegration:  "ordis_" + orSignature + "_(" + cleanExpr + ")"
        //   Integration:     "orint_" + cleanSignature + "_(" + head + ")"
        // Both prefixes are 6 ASCII chars. orSignature never contains "_(".
        //
        // On match, writes the orSignature into `orExprOut` and the branch body
        // (with its wrapping parens — e.g. "((in[7,1]))") into `branchBodyOut`.
        // Callers must route ALL OR-payload reads through this helper; no
        // substring matching on OR payload anywhere else.
        //
        // Production callers use classifyOrScopeView; this owning-string form
        // is the byte-twin oracle for classify_or_scope_view_twin.

/// @brief Classify the OR mode encoded by a scope payload.
/// @details Preserves the payload grammar and exact signature/body slices.
/// @param nm Read-only scope-name map.
/// @param validityId Scope identifier to inspect.
/// @param orExprOut Receives the OR signature on a matching scope.
/// @param branchBodyOut Receives the wrapped branch body on a matching scope.
/// @return The existing scope category, including NotOrScope for other payloads.
static inline OrScopeKind classifyOrScope(const NameMap& nm, NameId validityId,
                                           std::string& orExprOut,
                                           std::string& branchBodyOut) {
            if (validityId <= 0
                || static_cast<std::size_t>(validityId)
                       >= static_cast<std::size_t>(nm.stackSize()))
                return OrScopeKind::NotOrScope;
            if (nm.stackEmpty(validityId)) return OrScopeKind::NotOrScope;
            // nm is const -> nothing mints in this body, so the payload span is
            // valid for its full extent; only the two OUT-params materialize a
            // std::string at the caller boundary (I-3). A goal-carrying
            // payload (`<goal>_subproof_orint_...`) classifies by its bare
            // half — the goal prefix is ownership metadata, not OR shape.
            const StrSpan payload = stripSubproofPrefixView(
                nm.decodeSubView(nm.stackBack(validityId)));

            const StrSpan INT_PREFIX("orint_", 6);
            const StrSpan DIS_PREFIX("ordis_", 6);

            OrScopeKind kind;
            int32_t prefixLen;
            if (payload.len >= INT_PREFIX.len
                && equalSpans(StrSpan(payload.ptr, INT_PREFIX.len), INT_PREFIX)) {
                kind = OrScopeKind::Integration;
                prefixLen = INT_PREFIX.len;
            } else if (payload.len >= DIS_PREFIX.len
                && equalSpans(StrSpan(payload.ptr, DIS_PREFIX.len), DIS_PREFIX)) {
                kind = OrScopeKind::Disintegration;
                prefixLen = DIS_PREFIX.len;
            } else {
                return OrScopeKind::NotOrScope;
            }

            int32_t sep = -1;
            for (int32_t i = prefixLen; i + 1 < payload.len; ++i) {
                if (payload.ptr[i] == '_' && payload.ptr[i + 1] == '(') { sep = i; break; }
            }
            if (sep < 0) return OrScopeKind::NotOrScope;  // malformed

            orExprOut = StrSpan(payload.ptr + prefixLen, sep - prefixLen).toStdString();
            branchBodyOut = StrSpan(payload.ptr + sep + 1,
                                    payload.len - (sep + 1)).toStdString();  // "(body)" with wrapping parens
            return kind;
        }

/// @brief Zero-copy `StrSpan` twin of @ref classifyOrScope — classify a
        ///        validity scope as `_orint_` (Integration) / `_ordis_`
        ///        (Disintegration) / neither, returning the OR-signature and the
        ///        branch-body payload as spans instead of owned std::strings.
        ///
        /// @details
        /// Byte-for-byte identical classification and slicing to
        /// @ref classifyOrScope: it reads the same `decodeSubView` payload of the
        /// scope's own boundary sub-id, tests the same `orint_` / `ordis_`
        /// prefixes, finds the same `_(` separator, and produces the same two
        /// payload sub-ranges — but writes them as `StrSpan`s over the interned
        /// bytes rather than materialising a `std::string` (the `.toStdString()`
        /// the owning overload applies). Both out-params are left UNTOUCHED on a
        /// `NotOrScope` return, exactly as the owning overload leaves its
        /// std::strings.
        ///
        /// @warning [I-3](../../docs/agentic_swdd/30_invariants.md#i-3): the out
        ///          spans alias `nm`'s sub-expression-table cold bytes — valid
        ///          while `nm` is resident (a deload frees pages outright).
        ///          The precise mint-hazard rule: appends to the paged cold
        ///          tables never relocate existing bytes, so the spans SURVIVE
        ///          any mint that does not write the SUB-table — and only
        ///          `NameMap::encodePush` (or the scoped-`encode` recursion
        ///          ending in it) writes the sub-table; a FLAT `encode` writes
        ///          only the names table + `validityNodes`, and mints into
        ///          other interners are unrelated. The two sanctioned
        ///          production callers rely on exactly this: `ordisMerge`
        ///          consumes both spans before any NameMap mint, and the
        ///          `_orint_` discharge site passes `orExprOut` across two
        ///          `encode` calls that are a lookup HIT and a provably FLAT
        ///          mint (an OR signature is an `encodePush` payload slice,
        ///          delimiter-free by that door's assert). The DEFAULT for an
        ///          unanalyzed caller remains copy-before-mint / recover from
        ///          an interner id.
        ///
        /// @param nm            The name map owning the validity scope forest.
        /// @param validityId    The validity id to classify.
        /// @param orExprOut     OUT: the OR-signature payload slice (set on
        ///                      Integration / Disintegration only).
        /// @param branchBodyOut OUT: the wrapped `(body)` branch payload slice
        ///                      (set on Integration / Disintegration only).
        /// @return The scope kind; `NotOrScope` leaves both out-params untouched.
        /// @see classifyOrScope — the owning-string overload.
static inline OrScopeKind classifyOrScopeView(const NameMap& nm, NameId validityId,
                                               StrSpan& orExprOut,
                                               StrSpan& branchBodyOut) {
            if (validityId <= 0
                || static_cast<std::size_t>(validityId)
                       >= static_cast<std::size_t>(nm.stackSize()))
                return OrScopeKind::NotOrScope;
            if (nm.stackEmpty(validityId)) return OrScopeKind::NotOrScope;
            // Goal-carrying payloads classify by their bare half (see the
            // owning twin above).
            const StrSpan payload = stripSubproofPrefixView(
                nm.decodeSubView(nm.stackBack(validityId)));

            const StrSpan INT_PREFIX("orint_", 6);
            const StrSpan DIS_PREFIX("ordis_", 6);

            OrScopeKind kind;
            int32_t prefixLen;
            if (payload.len >= INT_PREFIX.len
                && equalSpans(StrSpan(payload.ptr, INT_PREFIX.len), INT_PREFIX)) {
                kind = OrScopeKind::Integration;
                prefixLen = INT_PREFIX.len;
            } else if (payload.len >= DIS_PREFIX.len
                && equalSpans(StrSpan(payload.ptr, DIS_PREFIX.len), DIS_PREFIX)) {
                kind = OrScopeKind::Disintegration;
                prefixLen = DIS_PREFIX.len;
            } else {
                return OrScopeKind::NotOrScope;
            }

            int32_t sep = -1;
            for (int32_t i = prefixLen; i + 1 < payload.len; ++i) {
                if (payload.ptr[i] == '_' && payload.ptr[i + 1] == '(') { sep = i; break; }
            }
            if (sep < 0) return OrScopeKind::NotOrScope;  // malformed

            orExprOut = StrSpan(payload.ptr + prefixLen, sep - prefixLen);
            branchBodyOut = StrSpan(payload.ptr + sep + 1,
                                    payload.len - (sep + 1));  // "(body)" with wrapping parens
            return kind;
        }

/// @brief Split a goal-carrying integration-scope payload
        ///        `<goal>_subproof_<bare>` into its root-goal and bare halves.
        ///
        /// @details
        /// Goal-driven integration scopes minted on MAIN carry the root
        /// `toBeProved` goal in their validity payload so a disproof of that
        /// goal can find and wipe every scope its integration spawned. The
        /// payload grammar is
        ///
        ///     [!] ( ...balanced... ) _subproof_ <bare-payload>
        ///
        /// where the leading balanced-paren group (optionally negated) is the
        /// root goal expression VERBATIM and `<bare-payload>` is the legacy
        /// payload shape (`(implicationN[...])` for a sub-implication subproof,
        /// `orint_<sig>_(<head>)` for an OR-integration branch). The grammar is
        /// injective against every legacy payload: a legacy compact is a single
        /// balanced group with nothing after it, `orint_`/`ordis_`/hypo payloads
        /// do not start with `!`/`(`, and `)_subproof_` cannot occur inside a
        /// well-formed MPL expression (inside an expression a `)` is followed
        /// only by `,`, `)`, `]` or end-of-string).
        ///
        /// Statement-driven integration mints (admission/marker replays, the
        /// load-time prefill pass) carry no root goal and keep the bare legacy
        /// payload — both generations coexist by contract, so a non-matching
        /// payload is a defined miss, not an error.
        ///
        /// @param payload  The validity payload (top-of-stack sub-string).
        /// @param goalOut  OUT: the root goal slice (set only on a match).
        /// @param bareOut  OUT: the bare legacy payload slice (set only on a
        ///                 match; always non-empty on a match).
        /// @return `true` iff `payload` carries a goal prefix; on `false` both
        ///         out-params are untouched.
        /// @see stripSubproofPrefixView — the strip-or-identity convenience
        ///      form; classifyOrScope / classifyOrScopeView — consumers.
static inline bool splitSubproofPayload(StrSpan payload,
                                                StrSpan& goalOut,
                                                StrSpan& bareOut) {
            static const char SEP[] = "_subproof_";
            const int32_t sepLen = static_cast<int32_t>(sizeof(SEP) - 1);
            int32_t i = 0;
            if (i < payload.len && payload.ptr[i] == '!') ++i;
            if (i >= payload.len || payload.ptr[i] != '(') return false;
            int32_t depth = 0;
            int32_t close = -1;
            for (int32_t k = i; k < payload.len; ++k) {
                if (payload.ptr[k] == '(') ++depth;
                else if (payload.ptr[k] == ')') {
                    --depth;
                    if (depth == 0) { close = k; break; }
                }
            }
            if (close < 0) return false;
            const int32_t sepStart = close + 1;
            if (sepStart + sepLen >= payload.len) return false;
            if (std::memcmp(payload.ptr + sepStart, SEP,
                            static_cast<size_t>(sepLen)) != 0) return false;
            goalOut = StrSpan(payload.ptr, sepStart);
            bareOut = StrSpan(payload.ptr + sepStart + sepLen,
                              payload.len - (sepStart + sepLen));
            return true;
        }

static constexpr char kIntegrationTrialScopePrefix[] = "integration_";

/// @brief Whether a scope payload names an integration trial scope.
        ///
        /// @details
        /// A gated integration admission is made first in its own trial scope
        /// `<parent>_boundary_integration_<names>`, `<names>` the witnesses the
        /// admission mints joined by `_` in slot order
        /// (D-481); its payload is
        /// `integration_<names>`. The test is a prefix test on the payload,
        /// never a substring test on a full scope name: the
        /// hypothetical-disintegration sentinel payload
        /// `product_of_hypo_disintegration_of_integration_goal_...` contains
        /// `integration_` in its interior.
        ///
        /// @param payload A scope's own payload (its top-of-stack sub-string).
        /// @return `true` iff @p payload is `integration_` followed by a
        ///         non-empty list of witness names.
        /// @invariant Pure; allocates and mints nothing.
        /// @see isIntegrationTrialScope, atOrBelowIntegrationTrial.
static inline bool isIntegrationTrialPayload(StrSpan payload) {
            const int32_t prefixLen =
                static_cast<int32_t>(sizeof(kIntegrationTrialScopePrefix) - 1);
            return payload.len > prefixLen
                && equalSpans(StrSpan(payload.ptr, prefixLen),
                              StrSpan(kIntegrationTrialScopePrefix, prefixLen));
        }

/// @brief Whether a validity scope is itself an integration trial scope.
        ///
        /// @details
        /// Reads the scope's own payload (`decodeSubView` of its top-of-stack
        /// sub-id) and applies @ref isIntegrationTrialPayload. A root such as
        /// `main` has no payload and is never a trial scope. The id range is
        /// the name map's own contract (`NameMap::stackEmpty`), which also
        /// answers `MAIN_ID` on a map no name was encoded into yet.
        ///
        /// @param nm         The name map owning the validity forest.
        /// @param validityId A minted validity id.
        /// @return `true` iff the scope's own payload names a trial.
        /// @invariant Mints nothing.
        /// @see isIntegrationTrialPayload, atOrBelowIntegrationTrial.
static inline bool isIntegrationTrialScope(const NameMap& nm, NameId validityId) {
            assert(validityId > 0 && "isIntegrationTrialScope: expected a minted validity id");
            if (nm.stackEmpty(validityId)) return false;
            return isIntegrationTrialPayload(nm.decodeSubView(nm.stackBack(validityId)));
        }

/// @brief Whether a validity scope is an integration trial scope or lies
        ///        below one.
        ///
        /// @details
        /// Walks the NameMap parent chain from @p validityId to its root, as
        /// `Memory::phase3IgnoresScope` does. Inside a trial scope and below it
        /// a witness takes part only in rule firings, equality classes and
        /// structural expansion: no witness is minted, no OR case is opened,
        /// no integration is prepared and no goal is registered there
        /// (D-481).
        ///
        /// @param nm         The name map owning the validity forest.
        /// @param validityId A minted validity id.
        /// @return `true` iff the scope or one of its ancestors is a trial scope.
        /// @invariant Mints nothing.
        /// @see isIntegrationTrialScope.
static inline bool atOrBelowIntegrationTrial(const NameMap& nm, NameId validityId) {
            for (NameId scope = validityId; scope != 0; scope = nm.parentOf(scope))
                if (isIntegrationTrialScope(nm, scope)) return true;
            return false;
        }

static constexpr char kUsefulnessScopePayload[] = "is_useful";
// The usefulness scope's full validity name: the payload `is_useful` on a
// weak step block's `main` (D-521).
static constexpr char kUsefulnessScopeName[] = "main_boundary_is_useful";

/// @brief Whether a scope payload names the induction usefulness scope.
        ///
        /// @details
        /// A weak induction step block tests its hypothesis in
        /// `main_boundary_is_useful`, whose payload is exactly `is_useful`
        /// (D-521). The test is an exact
        /// comparison: an induction scope payload `is_<chain>_<names>` shares
        /// the `is_` prefix but never equals it.
        ///
        /// @param payload A scope's own payload (its top-of-stack sub-string).
        /// @return `true` iff @p payload is `is_useful`.
        /// @invariant Pure; allocates and mints nothing.
        /// @see atOrBelowUsefulnessScope, isInductionScopePayload.
static inline bool isUsefulnessPayload(StrSpan payload) {
            return equalSpans(payload, StrSpan(kUsefulnessScopePayload,
                static_cast<int32_t>(sizeof(kUsefulnessScopePayload) - 1)));
        }

/// @brief Whether a validity scope is the induction usefulness scope or lies
        ///        below it.
        ///
        /// @details
        /// `main_boundary_is_useful` sits on a weak step block's `main`, so it
        /// sees `main`'s statements and rules: a rule fires on `main`'s facts
        /// and the planted hypothesis rows alike and deposits at the deeper of
        /// its scopes (I-38), never into `main`. Inside it and below only
        /// firings and equality classes run, as in an integration trial scope
        /// (@ref atOrBelowRestrictedScope), and nothing disintegrates or is
        /// admitted (D-521).
        ///
        /// @param nm         The name map owning the validity forest.
        /// @param validityId A minted validity id.
        /// @return `true` iff the scope or one of its ancestors has the payload
        ///         `is_useful`.
        /// @invariant Mints nothing.
        /// @see isUsefulnessPayload, atOrBelowRestrictedScope.
static inline bool atOrBelowUsefulnessScope(const NameMap& nm, NameId validityId) {
            assert(validityId > 0 && "atOrBelowUsefulnessScope: expected a minted validity id");
            for (NameId scope = validityId; scope != 0; scope = nm.parentOf(scope)) {
                if (nm.stackEmpty(scope)) continue;
                if (isUsefulnessPayload(nm.decodeSubView(nm.stackBack(scope)))) return true;
            }
            return false;
        }

/// @brief Whether a validity scope only fires rules and joins classes: an
        ///        integration trial scope, the usefulness scope, or a scope
        ///        below either.
        ///
        /// @details
        /// The common restriction of the two scope kinds: no witness is
        /// minted, no OR case, subproof or contradiction scope opens, no
        /// integration is prepared, no parked source replays into it and no
        /// goal is registered there (D-481, D-521).
        ///
        /// @param nm         The name map owning the validity forest.
        /// @param validityId A minted validity id.
        /// @return `true` iff @ref atOrBelowIntegrationTrial or
        ///         @ref atOrBelowUsefulnessScope holds.
        /// @invariant Mints nothing.
        /// @see atOrBelowIntegrationTrial, atOrBelowUsefulnessScope.
static inline bool atOrBelowRestrictedScope(const NameMap& nm, NameId validityId) {
            return atOrBelowIntegrationTrial(nm, validityId) || atOrBelowUsefulnessScope(nm, validityId);
        }

/// @brief Whether a validity scope is a kind that may hold a registered goal.
        ///
        /// @details
        /// Goals live only where a scope exports nothing but its own result:
        /// a root such as `main` (no payload), an implication or integration
        /// subproof (a compact payload `(...)`, bare or behind its root goal's
        /// `<goal>_subproof_` prefix) and an OR-integration scope (`orint_`).
        /// OR cases, hypothesis, contradiction, induction, trial and sentinel
        /// scopes hold none. The innermost goal set of an OR case
        /// (D-549) relies on it: a goal at a case scope
        /// would shadow outer goals the case still serves through its cohort's
        /// convergence.
        ///
        /// @param nm         The name map owning the validity forest.
        /// @param validityId A minted validity id.
        /// @return `true` iff the scope's own payload is one of the goal kinds.
        /// @invariant Pure; allocates and mints nothing.
        /// @see stripSubproofPrefixView, classifyOrScopeView.
static inline bool canHoldGoal(const NameMap& nm, NameId validityId) {
            assert(validityId > 0 && "canHoldGoal: expected a minted validity id");
            if (nm.stackEmpty(validityId)) return true;
            const StrSpan bare = stripSubproofPrefixView(nm.decodeSubView(nm.stackBack(validityId)));
            if (bare.len > 0 && bare.ptr[0] == '(') return true;
            StrSpan signature, disjunct;
            return classifyOrScopeView(nm, validityId, signature, disjunct) == OrScopeKind::Integration;
        }

static constexpr char kInductionScopePrefix[] = "is_";

/// @brief Whether a scope payload names an induction scope.
        ///
        /// @details
        /// Every witness admission made through an induction key opens its
        /// own scope `<parent>_boundary_is_<chain>_<names>`
        /// (D-519): `<chain>` is the admitting key's
        /// premise chain with the admitted slot written as a marker, `<names>`
        /// the one or two names the key admitted. The payload is
        /// `is_<chain>_<names>`. The test is a prefix test on the payload,
        /// never a substring test on a full scope name: `ordis_` ends in
        /// `is_` and must not count. The usefulness scope's payload
        /// `is_useful` shares the prefix and is not an induction scope
        /// (@ref isUsefulnessPayload).
        ///
        /// @param payload A scope's own payload (its top-of-stack sub-string).
        /// @return `true` iff @p payload starts with `is_`, continues, and is
        ///         not `is_useful`.
        /// @invariant Pure; allocates and mints nothing.
        /// @see isInductionScope, inductionRootOf, inductionChainOnPath.
static inline bool isInductionScopePayload(StrSpan payload) {
            const int32_t prefixLen =
                static_cast<int32_t>(sizeof(kInductionScopePrefix) - 1);
            return payload.len > prefixLen
                && equalSpans(StrSpan(payload.ptr, prefixLen),
                              StrSpan(kInductionScopePrefix, prefixLen))
                && !isUsefulnessPayload(payload);
        }

/// @brief Whether a validity scope is itself an induction scope.
        ///
        /// @details
        /// Reads the scope's own payload (`decodeSubView` of its top-of-stack
        /// sub-id) and applies @ref isInductionScopePayload. A root such as
        /// `main` has no payload and is never an induction scope.
        ///
        /// @param nm         The name map owning the validity forest.
        /// @param validityId A minted validity id.
        /// @return `true` iff the scope's own payload names an induction scope.
        /// @invariant Mints nothing.
        /// @see isInductionScopePayload, inductionRootOf.
static inline bool isInductionScope(const NameMap& nm, NameId validityId) {
            assert(validityId > 0 && "isInductionScope: expected a minted validity id");
            if (nm.stackEmpty(validityId)) return false;
            return isInductionScopePayload(nm.decodeSubView(nm.stackBack(validityId)));
        }

/// @brief The nearest scope at or above a scope that is not an induction
        ///        scope.
        ///
        /// @details
        /// Induction scopes carry no assumption: what a chain derives in them
        /// is true where the chain started. Their root is the scope reached by
        /// dropping every trailing `_is_` segment — usually the step block's
        /// `main`, or an OR branch or subproof scope the admission was made in.
        /// A goal registered at the root is reached from inside the tree
        /// (D-519).
        ///
        /// @param nm         The name map owning the validity forest.
        /// @param validityId A minted validity id.
        /// @return @p validityId itself when it is not an induction scope,
        ///         otherwise its nearest non-induction ancestor.
        /// @invariant Mints nothing; an induction scope always has a parent.
        /// @see isInductionScope.
static inline NameId inductionRootOf(const NameMap& nm, NameId validityId) {
            NameId scope = validityId;
            while (isInductionScope(nm, scope)) {
                scope = nm.parentOf(scope);
                assert(scope != 0 && "inductionRootOf: an induction scope has a parent");
            }
            return scope;
        }

/// @brief Whether an induction payload was opened by exactly this chain.
        ///
        /// @details
        /// The payload is `is_` + the chain elements concatenated in order +
        /// `_` + the admitted names. The chain matches when its concatenation
        /// follows `is_` byte for byte and is followed by `_`: a chain that is
        /// a strict prefix of another one is followed by `(` or `!` there, so
        /// it never matches the longer one's payload.
        ///
        /// @param payload An induction scope payload.
        /// @param chain   The chain elements with their markers, in order.
        /// @param chainN  Number of elements; positive.
        /// @return `true` iff @p payload names @p chain.
        /// @invariant Pure; allocates and mints nothing.
        /// @see inductionChainOnPath, inductionScopePayloadScratch.
static inline bool inductionPayloadNamesChain(StrSpan payload,
            const StrSpan* chain, int32_t chainN) {
            assert(chainN > 0 && "inductionPayloadNamesChain: an empty chain");
            if (!isInductionScopePayload(payload)) return false;
            int32_t at = static_cast<int32_t>(sizeof(kInductionScopePrefix) - 1);
            for (int32_t i = 0; i < chainN; ++i) {
                if (at + chain[i].len > payload.len
                    || std::memcmp(payload.ptr + at, chain[i].ptr,
                                   static_cast<size_t>(chain[i].len)) != 0)
                    return false;
                at += chain[i].len;
            }
            return at < payload.len && payload.ptr[at] == '_';
        }

/// @brief Whether a scope or one of its ancestors is an induction scope
        ///        opened by this chain.
        ///
        /// @details
        /// No backward readmission (D-519): an
        /// induction admission at a scope is refused when the admitting key's
        /// chain already opened an induction scope on the path from that scope
        /// up to the root. Weak chains therefore only move forward; the strong
        /// step's two demands go further and admit one catch per path between
        /// them (`AdmissionMemory::isAdmittedIntegrationDouble`).
        ///
        /// @param nm         The name map owning the validity forest.
        /// @param validityId The scope an admission is probed at.
        /// @param chain      The admitting key's chain with its markers.
        /// @param chainN     Number of chain elements; positive.
        /// @return `true` iff an induction scope on the path names @p chain.
        /// @invariant Mints nothing.
        /// @see inductionPayloadNamesChain.
static inline bool inductionChainOnPath(const NameMap& nm, NameId validityId,
            const StrSpan* chain, int32_t chainN) {
            assert(validityId > 0 && "inductionChainOnPath: expected a minted validity id");
            for (NameId scope = validityId; scope != 0; scope = nm.parentOf(scope)) {
                if (!isInductionScope(nm, scope)) continue;
                if (inductionPayloadNamesChain(nm.decodeSubView(nm.stackBack(scope)), chain, chainN))
                    return true;
            }
            return false;
        }

/// @brief Whether a scope lies inside an induction scope.
        ///
        /// @details
        /// The cheap gate before any chain comparison: outside every induction
        /// scope no admission can be a backward readmission.
        ///
        /// @param nm         The name map owning the validity forest.
        /// @param validityId A minted validity id.
        /// @return `true` iff the scope or one of its ancestors is an
        ///         induction scope.
        /// @invariant Mints nothing.
        /// @see inductionChainOnPath.
static inline bool atOrBelowInductionScope(const NameMap& nm, NameId validityId) {
            for (NameId scope = validityId; scope != 0; scope = nm.parentOf(scope))
                if (isInductionScope(nm, scope)) return true;
            return false;
        }

/// @brief The scope where the induction hypothesis's eating stands for a
        ///        scope: its deepest induction scope at or above it, else `main`.
        ///
        /// @details
        /// The step block files its hypothesis chain at `main`, and every eat
        /// files the advanced chain in the induction scope it opens. On the
        /// path of a scope, the chain filed at the deepest induction scope is
        /// therefore the current state of the eating; a chain filed above it
        /// is a stale state of an earlier node. A hypothesis chain serves a
        /// scope only from this position (D-519).
        ///
        /// @param nm         The name map owning the validity forest.
        /// @param validityId A minted validity id.
        /// @return The deepest induction scope on the path from @p validityId
        ///         to the root, including @p validityId itself; `MAIN_ID` when
        ///         the path holds none.
        /// @invariant Mints nothing.
        /// @see atOrBelowInductionScope, isInductionScope.
static inline NameId inductionEatingPosition(const NameMap& nm, NameId validityId) {
            assert(validityId > 0 && "inductionEatingPosition: expected a minted validity id");
            for (NameId scope = validityId; scope != 0; scope = nm.parentOf(scope))
                if (isInductionScope(nm, scope)) return scope;
            return NameMap::MAIN_ID;
        }

/// @brief Whether a statement in an induction scope is a goal hit, and the
        ///        root it leaves the tree to.
        ///
        /// @details
        /// Induction scopes carry no assumption, so a statement derived in one
        /// holds at the tree's root. It leaves the tree only when it is an open
        /// goal at the root or above it (a tree grown inside an OR branch hands
        /// a goal of the branch's parent chain to the branch). The statement
        /// door then registers it at the root in the same absorb, so the goal
        /// closes in the burst it is derived (D-519).
        ///
        /// @param mb          The owning logic block (goal registry, name map).
        /// @param originalId  The statement's original id.
        /// @param validityId  The statement's scope; a minted validity id.
        /// @param rootOut     [out] The tree's root (`inductionRootOf`) when
        ///                    the result is `true`; untouched otherwise.
        /// @return True iff @p validityId is an induction scope and the
        ///         statement is an open goal at its root or above.
        /// @invariant Mints nothing.
        /// @see inductionRootOf, StatementMemory::addStatement.
static inline bool inductionGoalExit(const Memory& mb, NameId originalId,
            NameId validityId, NameId& rootOut) {
            if (!isInductionScope(mb.nameMap, validityId)) return false;
            const NameId root = inductionRootOf(mb.nameMap, validityId);
            for (NameId scope = root; scope != 0; scope = mb.nameMap.parentOf(scope)) {
                if (mb.intToBeProved.lookup(packStatementKey(originalId, scope)) == 0) continue;
                rootOut = root;
                return true;
            }
            return false;
        }

/// @brief Build the payload of an induction scope.
        ///
        /// @details
        /// `is_` + the chain elements concatenated in order + `_` + each
        /// admitted name, joined by `_` (D-519). Written
        /// with its exact length on the caller's string-tier scratch arena.
        ///
        /// @param strings The caller's string-tier scratch arena.
        /// @param chain   The admitting key's chain with its markers.
        /// @param chainN  Number of chain elements; positive.
        /// @param names   The names the key admitted, in marker order.
        /// @param nameN   One, or two for a double-marker key.
        /// @return The payload, ready for `NameMap::encodePush`.
        /// @invariant Mints nothing; the payload contains no `_boundary_`.
        /// @see inductionPayloadNamesChain.
static inline ScratchString inductionScopePayloadScratch(ScratchArena& strings,
            const StrSpan* chain, int32_t chainN, const StrSpan* names, int32_t nameN) {
            assert(chainN > 0 && nameN >= 1 && nameN <= 2
                && "inductionScopePayloadScratch: a chain and one or two names");
            const int32_t prefixLen = static_cast<int32_t>(sizeof(kInductionScopePrefix) - 1);
            int32_t length = prefixLen;
            for (int32_t i = 0; i < chainN; ++i) length += chain[i].len;
            for (int32_t i = 0; i < nameN; ++i) {
                assert(!names[i].empty() && "inductionScopePayloadScratch: an empty name");
                length += 1 + names[i].len;
            }
            char* bytes = strings.allocBytes(length);
            std::memcpy(bytes, kInductionScopePrefix, static_cast<size_t>(prefixLen));
            int32_t at = prefixLen;
            for (int32_t i = 0; i < chainN; ++i) {
                std::memcpy(bytes + at, chain[i].ptr, static_cast<size_t>(chain[i].len));
                at += chain[i].len;
            }
            for (int32_t i = 0; i < nameN; ++i) {
                bytes[at++] = '_';
                std::memcpy(bytes + at, names[i].ptr, static_cast<size_t>(names[i].len));
                at += names[i].len;
            }
            assert(at == length);
            assert(!containsSpan(StrSpan(bytes, length), StrSpan("_boundary_", 10))
                && "inductionScopePayloadScratch: a payload never contains a boundary");
            return ScratchString::wrap(strings, bytes, length);
        }

/// @brief The one or two names an induction scope's payload carries after
        ///        its chain.
        ///
        /// @details
        /// The payload is `is_` + the chain elements concatenated + `_` + each
        /// admitted name, joined by `_` (D-519, @ref
        /// inductionScopePayloadScratch). Every chain element is one complete
        /// expression, `(...)` or `!(...)`; the names start after the `_` that
        /// follows the last element. One name is the rest of the payload. Two
        /// names split at the one `_` whose two sides are both names: a whole
        /// witness name (`classifyWitnessName`) or a run of letters and digits
        /// followed by any number of `_copy` suffixes. The split is unique
        /// (asserted).
        ///
        /// @param payload An induction scope payload.
        /// @param nameN   One, or two for a scope of the strong step's demand.
        /// @param names   Receives @p nameN views into @p payload.
        /// @invariant Pure; allocates and mints nothing.
        /// @see inductionScopePayloadScratch, inductionPayloadNamesChain.
static inline void inductionScopeNames(StrSpan payload, int32_t nameN, StrSpan* names) {
            assert(isInductionScopePayload(payload) && (nameN == 1 || nameN == 2)
                && "inductionScopeNames: an induction payload and one or two names");
            int32_t at = static_cast<int32_t>(sizeof(kInductionScopePrefix) - 1);
            while (at < payload.len && payload.ptr[at] != '_') {
                if (payload.ptr[at] == '!') ++at;
                assert(at < payload.len && payload.ptr[at] == '('
                    && "inductionScopeNames: a chain element is an expression");
                int32_t depth = 0;
                do {
                    if (payload.ptr[at] == '(') ++depth;
                    else if (payload.ptr[at] == ')') --depth;
                    ++at;
                } while (depth > 0 && at < payload.len);
                assert(depth == 0 && "inductionScopeNames: an unbalanced chain element");
            }
            assert(at + 1 < payload.len && payload.ptr[at] == '_'
                && "inductionScopeNames: the names follow the chain");
            const StrSpan rest(payload.ptr + at + 1, payload.len - at - 1);
            const auto isName = [](StrSpan token) {
                if (token.len == 0) return false;
                if (classifyWitnessName(token, false).kind != WitnessNameKind::Ordinary) return true;
                int32_t k = 0;
                while (k < token.len && ((token.ptr[k] >= 'a' && token.ptr[k] <= 'z')
                        || (token.ptr[k] >= 'A' && token.ptr[k] <= 'Z') || isDecimalDigitByte(token.ptr[k]))) ++k;
                if (k == 0) return false;
                while (k < token.len) {
                    if (token.len - k < 5 || std::memcmp(token.ptr + k, "_copy", 5) != 0) return false;
                    k += 5;
                }
                return true;
            };
            if (nameN == 1) {
                assert(isName(rest) && "inductionScopeNames: one name follows the chain");
                names[0] = rest;
                return;
            }
            int32_t splits = 0;
            for (int32_t p = 1; p + 1 < rest.len; ++p) {
                if (rest.ptr[p] != '_') continue;
                const StrSpan left(rest.ptr, p), right(rest.ptr + p + 1, rest.len - p - 1);
                if (!isName(left) || !isName(right)) continue;
                names[0] = left;
                names[1] = right;
                ++splits;
            }
            assert(splits == 1 && "inductionScopeNames: two names split at exactly one boundary");
            (void)splits;
        }

/// @brief The strong step hypothesis chain an induction scope opened
        ///        through the block's double-marker demand carries in its name.
        ///
        /// @details
        /// The strong hypothesis opens with its bound premise
        /// `(in3[m,x,u_n,u_+])` (I-303); the block's double-marker demand
        /// writes its earlier open argument as `marker1` and its later one as
        /// `marker2` (`AdmissionMemory::makeDoubleMarkerKeyScratch`). The same
        /// two names are rewritten in every premise of the chain, the other
        /// arguments keep their spelling, and the premises are concatenated in
        /// order (D-519).
        ///
        /// @param strings The caller's string-tier scratch arena.
        /// @param chain   The strong hypothesis premise chain in its
        ///                `u_`-prefixed instruction form, bound premise first.
        /// @param chainN  Number of premises; positive.
        /// @return The marked chain, concatenated.
        /// @invariant Mints nothing. The bound premise has exactly two distinct
        ///            open arguments (asserted).
        /// @see inductionScopePayloadScratch,
        ///      AdmissionMemory::makeDoubleMarkerKeyScratch.
static inline ScratchString inductionDoubleChainScratch(ScratchArena& strings,
            const StrSpan* chain, int32_t chainN) {
            assert(chainN > 0 && "inductionDoubleChainScratch: an empty chain");
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t n = getArgsSpans(chain[0], args, ExecutionParameters::MAX_ARITY);
            int32_t open[2] = { -1, -1 };
            int32_t openN = 0;
            for (int32_t i = 0; i < n; ++i) {
                if (startsWithSpan(args[i], "u_", 2)) continue;
                assert(openN < 2 && "inductionDoubleChainScratch: the bound premise has two open arguments");
                open[openN++] = i;
            }
            assert(openN == 2 && !equalSpans(args[open[0]], args[open[1]])
                && "inductionDoubleChainScratch: the bound premise has two distinct open arguments");
            const StrReplacement pairs[2] = {
                { args[open[0]], StrSpan("marker1", 7) },
                { args[open[1]], StrSpan("marker2", 7) } };
            ScratchString marked[ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS];
            assert(chainN <= ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS
                && "inductionDoubleChainScratch: a chain beyond the admission capacity");
            int32_t length = 0;
            for (int32_t i = 0; i < chainN; ++i) {
                marked[i] = replaceKeysScratch(strings, chain[i], pairs, 2);
                length += StrSpan(marked[i]).len;
            }
            char* bytes = strings.allocBytes(length);
            int32_t at = 0;
            for (int32_t i = 0; i < chainN; ++i) {
                const StrSpan piece(marked[i]);
                std::memcpy(bytes + at, piece.ptr, static_cast<size_t>(piece.len));
                at += piece.len;
            }
            assert(at == length);
            return ScratchString::wrap(strings, bytes, length);
        }

/// @brief Split a concatenated hypothesis chain into its premises.
        ///
        /// @details
        /// A marked chain (@ref inductionDoubleChainScratch) or an induction
        /// scope's chain is its premises written one after the other, each a
        /// complete expression `(...)` or a negated one `!(...)`. The walk
        /// starts a premise at an optional `!` and ends it at the parenthesis
        /// that closes its first one; the views alias @p chain.
        ///
        /// @param chain The concatenated premises; every byte belongs to a
        ///              premise (asserted).
        /// @param out   Receives one view per premise, in chain order.
        /// @param cap   Capacity of @p out; overflow asserts.
        /// @return The number of premises written.
        /// @invariant Pure; allocates and mints nothing.
        /// @see inductionDoubleChainScratch, inductionScopeNames.
static inline int32_t splitChainElements(StrSpan chain, StrSpan* out, int32_t cap) {
            int32_t count = 0;
            int32_t at = 0;
            while (at < chain.len) {
                const int32_t start = at;
                if (chain.ptr[at] == '!') ++at;
                assert(at < chain.len && chain.ptr[at] == '('
                    && "splitChainElements: a premise is an expression");
                int32_t depth = 0;
                do {
                    if (chain.ptr[at] == '(') ++depth;
                    else if (chain.ptr[at] == ')') --depth;
                    ++at;
                } while (depth > 0 && at < chain.len);
                assert(depth == 0 && "splitChainElements: an unbalanced premise");
                assert(count < cap && "splitChainElements: more premises than the caller's capacity");
                out[count++] = StrSpan(chain.ptr + start, at - start);
            }
            return count;
        }

/// @brief The premises of an induction hypothesis that are fully specified,
        ///        written as ground statements.
        ///
        /// @details
        /// The premises are in the hypothesis's instruction form: a fixed
        /// argument (the anchor arguments, the induction value, the other fixed
        /// digits) carries the `u_` prefix, an argument a strong catch admitted
        /// is written `marker1` / `marker2` (@ref inductionDoubleChainScratch),
        /// and every other argument is a variable the hypothesis still binds. A
        /// premise is fully specified when each of its arguments is fixed or a
        /// filled marker; it is then a concrete statement, the "missing
        /// expression" the hypothesis may wait for
        /// (D-575). Its ground text replaces `marker1`
        /// and `marker2` by @p names in marker order and drops every `u_`
        /// prefix, the spelling a statement of the block has. A premise with a
        /// variable is skipped.
        ///
        /// @param strings  The caller's string-tier scratch arena; the texts
        ///                 live in the caller's `ScratchScope`.
        /// @param premises The premises, `(...)` or `!(...)`.
        /// @param premiseN Number of premises.
        /// @param names    The names that fill `marker1` and `marker2`.
        /// @param nameN    0 (no catch: only fixed arguments count) or 2.
        /// @param out      Receives the ground texts in premise order.
        /// @param cap      Capacity of @p out; overflow asserts.
        /// @return The number of ground premises written.
        /// @invariant Mints nothing; a written text contains no `marker` and no
        ///            `u_` argument.
        /// @see splitChainElements, inductionDoubleChainScratch.
static inline int32_t groundHypothesisPremisesScratch(ScratchArena& strings,
            const StrSpan* premises, int32_t premiseN, const StrSpan* names, int32_t nameN,
            ScratchString* out, int32_t cap) {
            assert((nameN == 0 || nameN == 2) && "groundHypothesisPremisesScratch: no catch or a strong catch");
            int32_t count = 0;
            for (int32_t p = 0; p < premiseN; ++p) {
                StrSpan args[ExecutionParameters::MAX_ARITY];
                const int32_t argN = getArgsSpans(premises[p], args, ExecutionParameters::MAX_ARITY);
                StrReplacement pairs[ExecutionParameters::MAX_ARITY + 2];
                int32_t pairN = 0;
                bool ground = true;
                for (int32_t a = 0; a < argN && ground; ++a) {
                    bool listed = false;
                    for (int32_t q = 0; q < pairN && !listed; ++q) listed = equalSpans(pairs[q].key, args[a]);
                    if (listed) continue;
                    if (nameN == 2 && equalSpans(args[a], StrSpan("marker1", 7))) {
                        pairs[pairN++] = StrReplacement{ args[a], names[0] };
                    } else if (nameN == 2 && equalSpans(args[a], StrSpan("marker2", 7))) {
                        pairs[pairN++] = StrReplacement{ args[a], names[1] };
                    } else if (startsWithSpan(args[a], "u_", 2) && args[a].len > 2) {
                        pairs[pairN++] = StrReplacement{ args[a], StrSpan(args[a].ptr + 2, args[a].len - 2) };
                    } else {
                        ground = false;
                    }
                }
                if (!ground) continue;
                assert(count < cap && "groundHypothesisPremisesScratch: more ground premises than the caller's capacity");
                out[count++] = replaceKeysScratch(strings, premises[p], pairs, pairN);
            }
            return count;
        }

static constexpr char kIntScopePrefix[] = "int_scope_";

/// @brief Whether a scope payload names an int scope.
        ///
        /// @details
        /// When a weak induction hypothesis install files identical
        /// integration goals once per premise, the first fill of such a
        /// premise-bound instruction opens its own scope
        /// `<parent>_boundary_int_scope_<premise>_<witness>`
        /// (D-578): `<premise>` is the original
        /// theorem premise the copy is bound to, `<witness>` the name that
        /// filled the marker. The payload is `int_scope_<premise>_<witness>`.
        /// The test is a prefix test on the payload, never a substring test on
        /// a full scope name: integration witness names (`int_<k>_lev_<L>_<n>`)
        /// occur inside other payloads.
        ///
        /// @param payload A scope's own payload (its top-of-stack sub-string).
        /// @return `true` iff @p payload starts with `int_scope_` and continues.
        /// @invariant Pure; allocates and mints nothing.
        /// @see isIntScope, atOrBelowIntScope, intScopePayloadScratch.
static inline bool isIntScopePayload(StrSpan payload) {
            const int32_t prefixLen = static_cast<int32_t>(sizeof(kIntScopePrefix) - 1);
            return payload.len > prefixLen
                && equalSpans(StrSpan(payload.ptr, prefixLen), StrSpan(kIntScopePrefix, prefixLen));
        }

/// @brief Whether a validity scope is itself an int scope.
        ///
        /// @details
        /// Reads the scope's own payload (`decodeSubView` of its top-of-stack
        /// sub-id) and applies @ref isIntScopePayload. A root such as `main`
        /// has no payload and is never an int scope.
        ///
        /// @param nm         The name map owning the validity forest.
        /// @param validityId A minted validity id.
        /// @return `true` iff the scope's own payload names an int scope.
        /// @invariant Mints nothing.
        /// @see isIntScopePayload, atOrBelowIntScope.
static inline bool isIntScope(const NameMap& nm, NameId validityId) {
            assert(validityId > 0 && "isIntScope: expected a minted validity id");
            if (nm.stackEmpty(validityId)) return false;
            return isIntScopePayload(nm.decodeSubView(nm.stackBack(validityId)));
        }

/// @brief Whether a validity scope is an int scope or lies below one.
        ///
        /// @details
        /// Int scopes have one level only
        /// (D-578): a first fill decided at or below an
        /// int scope continues in place and opens no further int scope. The
        /// walk follows the NameMap parent chain to the root.
        ///
        /// @param nm         The name map owning the validity forest.
        /// @param validityId A minted validity id.
        /// @return `true` iff the scope or one of its ancestors is an int scope.
        /// @invariant Mints nothing.
        /// @see isIntScope.
static inline bool atOrBelowIntScope(const NameMap& nm, NameId validityId) {
            assert(validityId > 0 && "atOrBelowIntScope: expected a minted validity id");
            for (NameId scope = validityId; scope != 0; scope = nm.parentOf(scope))
                if (isIntScope(nm, scope)) return true;
            return false;
        }

/// @brief Build the payload of an int scope.
        ///
        /// @details
        /// `int_scope_` + the bound copy's original theorem premise + `_` + the
        /// name that filled the marker at the first fill
        /// (D-578). Written with its exact length on the
        /// caller's string-tier scratch arena.
        ///
        /// @param strings The caller's string-tier scratch arena.
        /// @param premise The original premise, a complete expression.
        /// @param witness The filling name; non-empty.
        /// @return The payload, ready for `NameMap::encodePush`.
        /// @invariant Mints nothing; the payload contains no `_boundary_`
        ///            (asserted).
        /// @see isIntScopePayload.
static inline ScratchString intScopePayloadScratch(ScratchArena& strings,
            StrSpan premise, StrSpan witness) {
            assert(premise.len > 0 && premise.ptr[0] == '(' && !witness.empty()
                && "intScopePayloadScratch: a premise expression and a witness name");
            const int32_t prefixLen = static_cast<int32_t>(sizeof(kIntScopePrefix) - 1);
            const int32_t length = prefixLen + premise.len + 1 + witness.len;
            char* bytes = strings.allocBytes(length);
            std::memcpy(bytes, kIntScopePrefix, static_cast<size_t>(prefixLen));
            std::memcpy(bytes + prefixLen, premise.ptr, static_cast<size_t>(premise.len));
            bytes[prefixLen + premise.len] = '_';
            std::memcpy(bytes + prefixLen + premise.len + 1, witness.ptr, static_cast<size_t>(witness.len));
            assert(!containsSpan(StrSpan(bytes, length), StrSpan("_boundary_", 10))
                && "intScopePayloadScratch: a payload never contains a boundary");
            return ScratchString::wrap(strings, bytes, length);
        }

/// @brief Whether an integration key holds the name in `rec`'s role or the
        ///        name in the induction variable's role.
        ///
        /// @details
        /// The lexical first test of `AdmissionMemory::stepKeyConnected`:
        /// an argument of the key, a `u_` prefix stripped, equal to @p rec or
        /// to @p variable connects the key to the induction. A weak step
        /// block's own integrations pass `rec` and the inducted digit
        /// (@ref keyHoldsInductionVariable); a copy bound to the second twin
        /// premise passes the twin partner's predecessor and the partner
        /// (D-589).
        ///
        /// @param key      The key text, `u_`-prefixed arguments and a marker.
        /// @param rec      The name in `rec`'s role (nonempty).
        /// @param variable The name in the induction variable's role
        ///                 (nonempty).
        /// @return `true` iff an argument of @p key is @p rec or @p variable.
        /// @invariant Pure; reads only its arguments.
        /// @see keyHoldsInductionVariable, AdmissionMemory::stepKeyConnected.
static inline bool keyHoldsInductionRole(StrSpan key, StrSpan rec, StrSpan variable) {
            assert(!rec.empty() && !variable.empty() && "keyHoldsInductionRole: both roles are named");
            StrSpan args[ExecutionParameters::MAX_ARITY];
            const int32_t argN = getArgsSpans(key, args, ExecutionParameters::MAX_ARITY);
            for (int32_t a = 0; a < argN; ++a) {
                const StrSpan argument = startsWithSpan(args[a], "u_", 2)
                    ? StrSpan(args[a].ptr + 2, args[a].len - 2) : args[a];
                if (equalSpans(argument, rec) || equalSpans(argument, variable)) return true;
            }
            return false;
        }

/// @brief Whether an integration key holds `rec` or a step block's
        ///        induction variable.
        ///
        /// @details
        /// A weak step block's recursion hypothesis is
        /// `(in2[rec,<variable>,<successor>])` (`Memory::recursionHypothesisId`,
        /// set before the hypothesis install): `<variable>` is the inducted
        /// digit. A key holding one of the two among its arguments (a `u_`
        /// prefix stripped) is connected to the induction; this is the
        /// lexical first test of `AdmissionMemory::stepKeyConnected`,
        /// which every integration key filed in a weak or strong step block
        /// passes (D-583,
        /// D-584), so that no int scope starts
        /// from a singleton (D-578).
        ///
        /// @param mb  The step block.
        /// @param key The key text, `u_`-prefixed arguments and a marker.
        /// @return `true` iff an argument of @p key is `rec` or the induction
        ///         variable.
        /// @invariant Mints nothing; the block has its recursion hypothesis
        ///            in that shape (asserted).
        /// @see AdmissionMemory::stepKeyConnected.
static inline bool keyHoldsInductionVariable(const Memory& mb, StrSpan key) {
            assert(mb.recursionHypothesisId != 0
                && "keyHoldsInductionVariable: a step block with its recursion hypothesis");
            const StrSpan hypothesis = mb.nameMap.decodeView(mb.recursionHypothesisId);
            StrSpan hypothesisArgs[ExecutionParameters::MAX_ARITY];
            const int32_t hypothesisN = getArgsSpans(hypothesis, hypothesisArgs, ExecutionParameters::MAX_ARITY);
            assert(hypothesisN == 3 && equalSpans(hypothesisArgs[0], StrSpan("rec", 3))
                && "keyHoldsInductionVariable: the recursion hypothesis is (in2[rec,<variable>,<successor>])");
            return keyHoldsInductionRole(key, StrSpan("rec", 3), hypothesisArgs[1]);
        }

/// @brief Whether a statement is an instance of an int scope's integration
        ///        goal.
        ///
        /// @details
        /// The goal is the integration goal of the instruction that opened the
        /// scope, with `marker`, `marker1` or `marker2` at its open arguments
        /// (D-578). A statement is an instance when it
        /// has the same core expression and sign and the same number of
        /// arguments, and equals the goal at every argument that is not a
        /// marker; the markers take any argument. Both texts are compared as
        /// written: the caller passes the goal in its canonical form at the
        /// scope and the statement as registered there.
        ///
        /// @param goal      The scope's integration goal with its markers.
        /// @param statement A registered statement.
        /// @return `true` iff @p statement is an instance of @p goal.
        /// @invariant Pure; allocates and mints nothing.
        /// @see isIntScope.
static inline bool intScopeGoalMatches(StrSpan goal, StrSpan statement) {
            const bool goalNegated = goal.len > 0 && goal.ptr[0] == '!';
            const bool statementNegated = statement.len > 0 && statement.ptr[0] == '!';
            if (goalNegated != statementNegated) return false;
            const StrSpan goalCore = extractExpressionUniversalSpan(goal);
            assert(!goalCore.empty() && "intScopeGoalMatches: the goal is an expression");
            if (!equalSpans(goalCore, extractExpressionUniversalSpan(statement))) return false;
            StrSpan goalArgs[ExecutionParameters::MAX_ARITY];
            StrSpan statementArgs[ExecutionParameters::MAX_ARITY];
            const int32_t goalN = getArgsSpans(goal, goalArgs, ExecutionParameters::MAX_ARITY);
            const int32_t statementN = getArgsSpans(statement, statementArgs, ExecutionParameters::MAX_ARITY);
            if (goalN != statementN) return false;
            for (int32_t a = 0; a < goalN; ++a) {
                if (equalSpans(goalArgs[a], StrSpan("marker", 6))
                    || equalSpans(goalArgs[a], StrSpan("marker1", 7))
                    || equalSpans(goalArgs[a], StrSpan("marker2", 7))) continue;
                if (!equalSpans(goalArgs[a], statementArgs[a])) return false;
            }
            return true;
        }

/// Payload prefix of a hypothetical-disintegration sentinel scope:
/// `disintegrateExprHypothetically` names the scope of one dry run
/// `<scope>_boundary_` + this prefix + `<n>_<goal>`
/// (D-496).
static constexpr char kHypotheticalSentinelPrefix[] =
            "product_of_hypo_disintegration_of_integration_goal_";

/// @brief Whether a validity name is a hypothetical-disintegration
        ///        sentinel scope or lies below one.
        ///
        /// @details
        /// A sentinel payload always follows a boundary token, so the test
        /// looks for `_boundary_` + @ref kHypotheticalSentinelPrefix anywhere
        /// in the full validity name; a descendant of a sentinel carries the
        /// same text. Hypothesis scopes (`_var…_hypo_<goal>`), trial scopes and
        /// every other scope kind do not contain it. Used by the chapter
        /// export's tripwire (`ProofReconstructor::buildStack`), which must
        /// never see a dry-run product.
        ///
        /// @param validityName A full validity name.
        /// @return `true` iff @p validityName names a sentinel scope or a
        ///         scope below one.
        /// @invariant Pure; allocates and mints nothing.
        /// @see kHypotheticalSentinelPrefix.
static inline bool namesHypotheticalSentinel(StrSpan validityName) {
            static constexpr std::string_view marker =
                "_boundary_product_of_hypo_disintegration_of_integration_goal_";
            static_assert(marker.substr(10) == std::string_view(kHypotheticalSentinelPrefix),
                "namesHypotheticalSentinel: marker is the boundary token plus the sentinel prefix");
            return std::string_view(validityName.ptr, static_cast<std::size_t>(validityName.len))
                .find(marker) != std::string_view::npos;
        }

/// @brief The anchor arguments of a logic block: the arguments of the first
        ///        anchor key on its chain to the root.
        ///
        /// @details
        /// Walks the `parentMemory` chain from @p memoryBlock to the root and
        /// returns the argument spans of the first key of the form
        /// `(<anchorName>[...])` — the batch's anchor element, whose arguments
        /// are the constants every theorem of the batch shares. The spans alias
        /// the process-wide never-deloaded `skeletonInterner` pages (frozen
        /// through the parallel phase, D-186), so they survive every NameMap or
        /// per-LB interner mint. A chain without an anchor key, or an empty
        /// anchor name, has no anchor arguments: 0 is a defined answer.
        ///
        /// @param memoryBlock The logic block whose chain is walked.
        /// @param anchorName  The batch anchor's name (may be empty).
        /// @param out         Caller array receiving the spans.
        /// @param cap         Capacity of @p out (anchor arity <= MAX_ARITY).
        /// @return The number of anchor argument spans written.
        /// @invariant Pure; allocates and mints nothing.
        /// @see CaseProcessor::collectAnchorArgs,
        ///      AdmissionMemory::witnessLinkArguments.
static inline int32_t anchorArgumentSpans(const Memory& memoryBlock, StrSpan anchorName,
                                                  StrSpan* out, int32_t cap) {
            if (anchorName.empty()) return 0;
            for (const Memory* p = &memoryBlock; p != nullptr; p = p->parentMemory) {
                const StrSpan k = p->exprKeyView();
                if (k.len < anchorName.len + 2 || k.ptr[0] != '('
                    || std::memcmp(k.ptr + 1, anchorName.ptr,
                                   static_cast<std::size_t>(anchorName.len)) != 0
                    || k.ptr[1 + anchorName.len] != '[')
                    continue;
                return getArgsSpans(k, out, cap);
            }
            return 0;
        }
};
}
