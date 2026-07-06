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

#include "../parameters.hpp"

#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace gl {

    /// @brief Pre-encoded statement for the static request pipeline. All fields
    /// `int16_t`; no heap allocation; fixed size (176 bytes on x64).
    ///
    /// @details
    /// The static request pipeline operates entirely on `int16_t` IDs minted from
    /// the per-LB `NameMap` so the hot path in `generateEncodedRequests*` never
    /// touches strings. Each `IntEncodedExpr` carries the canonicalized form of
    /// one expression: its name id, negation flag, arity, two flags for hypo /
    /// anchor classification, and the four fixed-capacity per-arg arrays
    /// (`argId`, `argUnchangeable`, `argIteration`, `argLevPlus1`, `argFullId`).
    /// Capacity is `ExecutionParameters::MAX_ARITY` per arg array.
    ///
    /// The struct is purely arithmetic — no constructors, no operator overloads,
    /// no virtual table — so it can be `memcpy`'d directly into the typed arena
    /// (`TypedArena<IntEncodedExpr>`). This is what lets the request emitter copy
    /// expressions in O(1) and key on the resulting pointer for dedupe.
    ///
    /// @see `StaticRequestEmitter` (`memory.hpp`) — populates an arena of
    ///      these and tracks dedupe.
    /// @see `encodeExpression` / `decodeExpression` — the canonical
    ///      EncodedExpression ↔ IntEncodedExpr converter pair, defined in
    ///      `memory.hpp`.
    struct IntEncodedExpr {
        int16_t nameId;          // NameMap ID of expression name
        int16_t negation;        // 0 or 1
        int16_t arity;           // number of arguments (capped at MAX_ARITY)
        int16_t maxIteration;    // max iteration number across args (-1 if none)
        int16_t originalId;      // NameMap ID of original string
        int16_t validityId;      // NameMap ID of validityName
        int16_t isHypo;          // 1 if validityName contains "_hypo_", else 0
        int16_t isAnchor;        // 1 if name starts with "Anchor", else 0
        int16_t argId[ExecutionParameters::MAX_ARITY];           // NameMap ID of arg name
        int16_t argUnchangeable[ExecutionParameters::MAX_ARITY]; // 1=unchangeable, 0=changeable
        int16_t argIteration[ExecutionParameters::MAX_ARITY];    // iteration number per arg (-1 if none)
        int16_t argLevPlus1[ExecutionParameters::MAX_ARITY];     // level+1 per arg (0 if none)
        int16_t argFullId[ExecutionParameters::MAX_ARITY];       // NameMap ID of full arg string (e.g. "it_0_lev_0_1")
    };

    /// @brief The closed origin-tag vocabulary of the history maps — opaque
    /// declaration.
    ///
    /// @details
    /// Declared here (with its fixed `uint8_t` underlying type, so it is a
    /// complete type) rather than only in `memory.hpp` so `LbMemory` can name the
    /// cold `TypedColdBlobMap<int64_t, IdOrigin>` member `exprOriginMap` and the
    /// `IdOrigin` alias below — the same "complete value type at the member
    /// declaration" reason `StatementFlags` lives here. The full enumerator list
    /// and the tag↔string tables stay together in `memory.hpp`.
    enum class OriginTag : uint8_t;

    /// @brief One id-form history line: (tag, packed dependency keys). The
    /// dependency vector keeps positional order — it is observable in the dump and
    /// the chapter export.
    using IdOrigin = std::pair<OriginTag, std::vector<int64_t>>;

    /// @brief Id-form origin map: packed (expressionId, validityId) key → history
    /// lines in insertion order. Iteration order is NOT key order — every
    /// order-sensitive walk derives a decoded lex-sorted snapshot instead. Backs
    /// `Memory::exprOriginMap`'s working form (now a cold blob map) and the
    /// `EquivalenceClass::equalityOriginMap` transient decode.
    using IdOriginMap = std::unordered_map<int64_t, std::vector<IdOrigin>>;

    /// @brief Per-statement flags stored as the value in the LB statement
    /// registry `Memory::intKnownStatements`.
    ///
    /// @details
    /// Defined here (beside `IntEncodedExpr`) rather than in `memory.hpp` so
    /// `LbMemory` can hold a `ColdHashMap<PodKeyStore<int32_t>, StatementFlags>`
    /// — the cold registry needs the complete value type at its member
    /// declaration. Four `bool`s, no padding: a valid `SingleValueStore` value
    /// (trivially copyable) with a deterministic deload image.
    ///
    /// `local` — true when the statement was added to the LB's local-encoded
    /// containers on a status 0/1 derivation; false for mail-origin / non-local
    /// arrivals. Preserved for provenance and future flag consumers.
    ///
    /// `fullyDisintegrated` — true when the statement entered `disintegrateExpr2`
    /// and came back fully disintegrated: it has no existence inside (nothing to
    /// witness), or every existence inside it got at least one admitted witness.
    /// `checkForEquivalence` suppresses re-disintegration of an equivalence-class
    /// variant only when the matched variant is flagged `fullyDisintegrated`.
    ///
    /// `registered` / `known` — the two membership bits that let one packed-key
    /// map carry both statement records. `registered` means the statement
    /// passed an add-path registration door (`addStatement` shape dispatch,
    /// `addEquality` / `addNegatedEquality`, the status-4 fact load, anchor
    /// handling, recursion mail, the compressor rule load). `known` means the
    /// statement entered the level registry — the record the Site F dedup
    /// ancestor scans, the contradiction negation scans, and the burst
    /// dependency skip consult. The two sets are deliberately NOT equal:
    /// `addStatement` registers unconditionally but admits to the level
    /// registry only behind its iteration-cap / secondary-variable /
    /// equivalence-filter gates; the equivalence-class commit grants `known`
    /// only; the compressor rule load grants `registered` only. Every gate
    /// must test the bit its contract names, never bare map presence.
    ///
    /// @see ExpressionAnalyzer::checkForEquivalence
    /// @see ExpressionAnalyzer::disintegrateExpr2
    /// @see upsertStatementKey — the OR-only write door for the bits.
    struct StatementFlags {
        bool local = false;
        bool fullyDisintegrated = false;
        bool registered = false;
        bool known = false;
    };

}
