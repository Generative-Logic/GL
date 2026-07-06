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

// The mail value types: ExpressionWithValidity plus the cold-map key/record
// types (OriginLine, IntMailStatementKey, IntMailOrigin) and their Codec<>
// specializations. They live in memory_infra (below memory.hpp) so BOTH mail
// homes can see them: RoutingColdMail (the routing mailbox, in
// routing_cold_mail.hpp) and ColdMail
// (the internal-mail mailbox, in cold_mail.hpp, held by LbMemory). The cold
// containers in cold_mail.hpp cannot reach memory.hpp (that would be a cycle:
// memory.hpp -> lb_memory.hpp -> cold_mail.hpp), so the value types they are
// built on are lifted here. Mail itself stays in memory.hpp -- it is
// materialized only transiently at the absorb / dump boundaries (makeHeapMail),
// so it never needs to be visible to LbMemory.

#include "typed_cold_map.hpp"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace gl {

    /// @brief Pair of `(expression, validityName)` used as a `std::map` / `std::set`
    /// key throughout the prover.
    ///
    /// @details
    /// The validity stack matters for almost every map keyed on an expression — the
    /// same expression text in two different scopes is two distinct facts. This
    /// struct couples the expression text with its scope name so the pair can be
    /// used directly as a key (no manual concatenation, no fragile string
    /// delimiters).
    ///
    /// - `original`     — the expression text (typically already disintegrated and
    ///   placed into pretty form).
    /// - `validityName` — the scope id in canonical string form. `"main"` is the
    ///   root; deeper scopes are produced by `NameMap::encodePush`. Per
    ///   [I-2](../../docs/agentic_swdd/30_invariants.md#i-2), non-`"main"` scopes are minted
    ///   only via `encodePush` so the ancestor hierarchy stays consistent.
    ///
    /// `operator<` is lexicographic over `(original, validityName)`; `operator==`
    /// is the natural equality.
    ///
    /// @see [`HashMemory::admissionMap`](#hashmemory) — keyed on this type.
    /// @see [`Memory::exprOriginMap`](#memory) — keyed on this type for origin
    ///      provenance lookups.
    /// @see [I-25](../../docs/agentic_swdd/30_invariants.md#i-25) — `addStatement` returns
    ///      `ExpressionWithValidity` pairs; cross-scope deposits ride through
    ///      `newStatements` so the receiver scope is preserved end-to-end.
    struct ExpressionWithValidity {
        std::string original;
        std::string validityName;

        // 1. Default Constructor
        ExpressionWithValidity()
            : original(""), validityName("") {
        }

        // 2. Parameterized Constructor
        ExpressionWithValidity(std::string original_, std::string validityName_)
            : original(std::move(original_)), validityName(std::move(validityName_)) {
        }

        // 3. Equality Operator (useful for comparisons)
        bool operator==(const ExpressionWithValidity& other) const {
            return original == other.original && validityName == other.validityName;
        }

        // 4. Less-than Operator (REQUIRED for std::set or std::map keys)
        bool operator<(const ExpressionWithValidity& other) const {
            if (original != other.original) {
                return original < other.original;
            }
            return validityName < other.validityName;
        }
    };

    /// @brief Blob record for a mail `origins` column — one history line
    ///        `(label, antecedents)` in string form.
    ///
    /// @details
    /// The mail boundary keeps origins as inlined strings (no cross-LB
    /// interning — see `Codec<Mail>`), so a mail origin run stores the
    /// same `std::pair<std::string, std::vector<ExpressionWithValidity>>` the
    /// heap `Mail::exprOriginMap` value vector held.
    using OriginLine =
        std::pair<std::string, std::vector<ExpressionWithValidity>>;

    /// @brief Record codec for `OriginLine` — `(label, antecedents)` in the
    ///        same inlined-string form `Codec<Mail>` uses for one origin.
    ///
    /// @details
    /// A blob-record codec (`serialize` -> bytes, `deserialize` -> record) as
    /// `TypedColdBlobMap<EwvKey, OriginLine>` requires. Layout: `int32`-length
    /// `label`, then `int32 depCount` followed by each dependency's `original`
    /// and `validityName` (each `int32`-length-prefixed) — matching the per-
    /// origin bytes of `Codec<Mail>::serialize`. `deserialize` is the exact
    /// inverse, asserting on truncation (Rule 19).
    ///
    /// @see OriginLine, Codec<Mail>.
    template <>
    struct Codec<OriginLine> {
        /// @brief Serialize one origin line to bytes.
        /// @param line The `(label, antecedents)` record.
        /// @return The byte blob; `deserialize` is its exact inverse.
        static std::vector<char> serialize(const OriginLine& line) {
            std::vector<char> out;
            const auto putPod = [&out](int32_t v) {
                const char* p = reinterpret_cast<const char*>(&v);
                out.insert(out.end(), p, p + sizeof(int32_t));
            };
            const auto putStr = [&out, &putPod](const std::string& s) {
                putPod(static_cast<int32_t>(s.size()));
                out.insert(out.end(), s.begin(), s.end());
            };
            putStr(line.first);
            putPod(static_cast<int32_t>(line.second.size()));
            for (const ExpressionWithValidity& dep : line.second) {
                putStr(dep.original);
                putStr(dep.validityName);
            }
            return out;
        }
        /// @brief Reconstruct one origin line from its blob (inverse).
        /// @param data The blob bytes.
        /// @param n    The blob length.
        /// @return The decoded origin line.
        static OriginLine deserialize(const char* data, int32_t n) {
            const char* cur = data;
            const char* const end = data + n;
            const auto getPod = [&cur, end](int32_t& v) {
                assert(cur + sizeof(int32_t) <= end
                    && "Codec<OriginLine>::deserialize: truncated pod");
                std::memcpy(&v, cur, sizeof(int32_t)); cur += sizeof(int32_t);
            };
            const auto getStr = [&cur, end, &getPod]() {
                int32_t len = 0; getPod(len);
                assert(len >= 0 && cur + len <= end
                    && "Codec<OriginLine>::deserialize: string past blob");
                std::string s(cur, cur + len); cur += len;
                return s;
            };
            OriginLine line;
            line.first = getStr();
            int32_t depCount = 0; getPod(depCount);
            assert(depCount >= 0
                && "Codec<OriginLine>::deserialize: bad dep count");
            line.second.reserve(static_cast<size_t>(depCount));
            for (int32_t d = 0; d < depCount; ++d) {
                std::string original = getStr();
                std::string validityName = getStr();
                line.second.emplace_back(std::move(original),
                                         std::move(validityName));
            }
            assert(cur == end
                && "Codec<OriginLine>::deserialize: trailing bytes");
            return line;
        }
    };

    // ===================================================================
    //  Id-form mail value types (mail statification).
    //
    //  The cross-LB routing mailboxes and the per-LB internal-mail channels
    //  store INTERNER IDS, not std::string. An EWV pair (original, validityName)
    //  becomes a pair of 4-byte ids, packed to int64 via packOriginKey for the
    //  EWV-keyed columns (origins / expandedImplications / disintegrationSignals)
    //  -- which reuse the existing Codec<int64_t> and Codec<IdOrigin>. Only the
    //  statements column additionally carries the level set, so only it needs its
    //  own variable-length byte key: IntMailStatementKey. The id-space (the GLOBAL
    //  mailInterner for routing mail, the per-LB NameMap for internal mail) is
    //  chosen by the owning mailbox's ROLE, not baked into these types -- decoding
    //  back to strings happens at each boundary with the right interner.
    // ===================================================================

    /// @brief Id-form of @ref MailStatementKey -- the (originalId, validityId,
    ///        levels) triple as a variable-length byte key.
    ///
    /// @details
    /// The statified twin of the string `MailStatementKey`: `original` /
    /// `validityName` are replaced by their interner ids (`originalId` /
    /// `validityId`, 4-byte -- the GLOBAL `mailInterner` for routing mail, the
    /// per-LB `NameMap` for internal mail, chosen by the owning mailbox). `levels`
    /// is the ascending, duplicate-free level list carried verbatim, so two
    /// members sharing (expression, scope) but differing in levels stay distinct
    /// keys (the multiplicity the absorb's level gate relies on). There is
    /// deliberately NO `operator<`: every observable ordering is decoded-lex via
    /// the owning interner at the boundary, never an accidental id-order sort (the
    /// `IntEwv` / [I-84](../../docs/agentic_swdd/30_invariants.md#i-84) discipline).
    ///
    /// @invariant `levels` is ascending and duplicate-free;
    ///            `Codec<IntMailStatementKey>` preserves it. `originalId` /
    ///            `validityId` belong to ONE interner -- meaningful only paired
    ///            with the mailbox whose role names that interner.
    /// @see MailStatementKey (the string twin), IntEwv, packOriginKey.
    struct IntMailStatementKey {
        int32_t originalId;
        int32_t validityId;
        std::vector<int32_t> levels;

        /// @brief Natural equality over all three fields.
        /// @param other The key to compare against.
        /// @return Whether ids and the level list all match.
        bool operator==(const IntMailStatementKey& other) const {
            return originalId == other.originalId
                && validityId == other.validityId
                && levels == other.levels;
        }
    };

    /// @brief Byte-key codec for @ref IntMailStatementKey -- `int32 originalId`,
    ///        `int32 validityId`, `int32 count`, then `count` `int32` levels.
    ///
    /// @details
    /// The id-form twin of `Codec<MailStatementKey>`: the two length-prefixed
    /// strings are replaced by their 4-byte ids (`putPod`, not `putStr`); the
    /// ascending level list is unchanged. `decode` is the exact inverse, asserting
    /// on truncation (Rule 19). All fields are fixed-width little-endian, so the
    /// byte layout is injective and deterministic.
    ///
    /// @see IntMailStatementKey, Codec<MailStatementKey>, BytesKeyCodecBase.
    template <>
    struct Codec<IntMailStatementKey> : BytesKeyCodecBase {
        /// @brief Encode an `IntMailStatementKey` to its canonical byte layout.
        /// @param k The key.
        /// @return The encoded bytes.
        static std::string encode(const IntMailStatementKey& k) {
            std::string out;
            const auto putPod = [&out](int32_t v) {
                const char* p = reinterpret_cast<const char*>(&v);
                out.append(p, p + sizeof(int32_t));
            };
            putPod(k.originalId);
            putPod(k.validityId);
            putPod(static_cast<int32_t>(k.levels.size()));
            for (const int32_t lv : k.levels) putPod(lv);
            return out;
        }
        /// @brief Decode an `IntMailStatementKey` from its byte span (inverse).
        /// @param s The encoded span.
        /// @return The reconstructed key.
        static IntMailStatementKey decode(StrSpan s) {
            const char* cur = s.ptr;
            const char* const end = s.ptr + s.len;
            const auto getPod = [&cur, end](int32_t& v) {
                assert(cur + sizeof(int32_t) <= end
                    && "Codec<IntMailStatementKey>::decode: truncated pod");
                std::memcpy(&v, cur, sizeof(int32_t)); cur += sizeof(int32_t);
            };
            IntMailStatementKey k;
            getPod(k.originalId);
            getPod(k.validityId);
            int32_t lvCount = 0; getPod(lvCount);
            assert(lvCount >= 0
                && "Codec<IntMailStatementKey>::decode: bad level count");
            k.levels.reserve(static_cast<size_t>(lvCount));
            for (int32_t i = 0; i < lvCount; ++i) {
                int32_t lv = 0; getPod(lv); k.levels.push_back(lv);
            }
            assert(cur == end
                && "Codec<IntMailStatementKey>::decode: trailing bytes");
            return k;
        }
    };

    /// @brief Id-form of one mail origin line -- `(tag, packed dependency keys)`,
    ///        the self-contained mail twin of `IdOrigin`.
    ///
    /// @details
    /// `tag` is the `OriginTag` enumerator stored as a raw `uint8_t` (the enum
    /// itself lives in `memory.hpp`, invisible here -- callers cast
    /// `OriginTag <-> uint8_t`), so no string label is ever stored. `deps` are
    /// packed `(originalId, validityId)` keys (`packOriginKey`) in the owning
    /// mailbox's origin id-space (the per-LB `originInterner` for internal mail,
    /// the global `mailInterner` for routing mail). The shape mirrors `IdOrigin`
    /// (`memory_infra/int_encoded_expr.hpp`) but is defined here so the mail
    /// headers (`cold_mail.hpp` / `routing_cold_mail.hpp`) can name the blob-map
    /// record type without reaching into `memory.hpp` (where `IdOrigin`'s `Codec`
    /// lives). Positional dependency order is preserved -- it is observable in the
    /// dump and chapter export.
    ///
    /// @see IdOrigin, OriginTag, packOriginKey, Codec<IntMailOrigin>.
    struct IntMailOrigin {
        uint8_t tag;
        std::vector<int64_t> deps;

        /// @brief Natural equality over tag and the packed dependency list.
        /// @param o The record to compare against.
        /// @return Whether tag and deps both match.
        bool operator==(const IntMailOrigin& o) const {
            return tag == o.tag && deps == o.deps;
        }
    };

    /// @brief Record codec for @ref IntMailOrigin -- `uint8 tag`, `int32 depCount`,
    ///        then `depCount` packed `int64` dependency keys.
    ///
    /// @details
    /// A blob-record codec (`serialize` -> bytes, `deserialize` -> record) as
    /// `TypedColdBlobMap<int64_t, IntMailOrigin>` requires. The id-form twin of
    /// `Codec<IdOrigin>` but pure POD (no string inlining). `deserialize` is the
    /// exact inverse, asserting on truncation (Rule 19); fixed-width little-endian,
    /// so the layout is injective and deterministic.
    ///
    /// @see IntMailOrigin, Codec<IdOrigin>.
    template <>
    struct Codec<IntMailOrigin> {
        /// @brief Serialize one mail origin line to bytes.
        /// @param r The `(tag, deps)` record.
        /// @return The byte blob; `deserialize` is its exact inverse.
        static std::vector<char> serialize(const IntMailOrigin& r) {
            std::vector<char> out;
            out.push_back(static_cast<char>(r.tag));
            const int32_t n = static_cast<int32_t>(r.deps.size());
            const char* np = reinterpret_cast<const char*>(&n);
            out.insert(out.end(), np, np + sizeof(int32_t));
            for (const int64_t d : r.deps) {
                const char* dp = reinterpret_cast<const char*>(&d);
                out.insert(out.end(), dp, dp + sizeof(int64_t));
            }
            return out;
        }
        /// @brief Reconstruct one mail origin line from its blob (inverse).
        /// @param data The blob bytes.
        /// @param n    The blob length.
        /// @return The decoded record.
        static IntMailOrigin deserialize(const char* data, int32_t n) {
            const char* cur = data;
            const char* const end = data + n;
            assert(cur + 1 <= end
                && "Codec<IntMailOrigin>::deserialize: truncated tag");
            IntMailOrigin r;
            r.tag = static_cast<uint8_t>(*cur); cur += 1;
            int32_t depCount = 0;
            assert(cur + sizeof(int32_t) <= end
                && "Codec<IntMailOrigin>::deserialize: truncated dep count");
            std::memcpy(&depCount, cur, sizeof(int32_t)); cur += sizeof(int32_t);
            assert(depCount >= 0
                && "Codec<IntMailOrigin>::deserialize: bad dep count");
            r.deps.reserve(static_cast<size_t>(depCount));
            for (int32_t i = 0; i < depCount; ++i) {
                int64_t d = 0;
                assert(cur + sizeof(int64_t) <= end
                    && "Codec<IntMailOrigin>::deserialize: dep past blob");
                std::memcpy(&d, cur, sizeof(int64_t)); cur += sizeof(int64_t);
                r.deps.push_back(d);
            }
            assert(cur == end
                && "Codec<IntMailOrigin>::deserialize: trailing bytes");
            return r;
        }
    };

} // namespace gl
