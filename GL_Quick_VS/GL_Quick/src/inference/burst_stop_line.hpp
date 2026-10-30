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
#include "hash_request_generator.hpp"
namespace gl {
    /// @brief Streaming request consumer for the phase-2 hashburst — checks each
    ///        request inline as it is generated, instead of buffering every
    ///        request and checking them all afterwards.
    ///
    /// @details
    /// @brief Bit width of the part-ordinal field in a packed doom line.
    ///
    /// @details A doom line packs a burst early-exit trigger as
    /// `(position << kDoomLineOrdinalBits) | partOrdinal` into one `int64_t`,
    /// where `position` is the triggering part's own submatch counter
    /// (`HashRequestGenerator::growthMatchCount()`) at the trigger and
    /// `partOrdinal` is the part's deterministic invocation ordinal (its
    /// expression-bucket index; `0` for an unsplit part). The packing makes the
    /// plain integer order the lexicographic (position, ordinal) order, so one
    /// atomic CAS-min keeps the global minimum — the deterministic winner.
    /// The width must cover every part ordinal an LB can have
    /// (`HashRequestGenerator::kMaxSplitParts`).
    inline constexpr int kDoomLineOrdinalBits = 14;
    static_assert(HashRequestGenerator::kMaxSplitParts
                      <= (int64_t(1) << kDoomLineOrdinalBits),
        "doom-line ordinal field narrower than the part-count ceiling "
        "(kMaxSplitParts) — widen kDoomLineOrdinalBits deliberately");

    /// @brief Sentinel doom-line value: "no doom trigger fired this burst".
    ///
    /// @details Compares greater than every real packed line, so the first
    /// trigger's CAS-min replaces it, and its decoded position lies above any
    /// reachable submatch count, so a part comparing its own counter against
    /// the sentinel's position never stops.
    inline constexpr int64_t kNoDoomLine =
        std::numeric_limits<int64_t>::max();

    /// @brief Pack a doom trigger's (position, part ordinal) pair into one
    ///        lexicographically-ordered doom-line value.
    ///
    /// @details The integer order of packed values equals the lexicographic
    /// order of (position, ordinal) pairs because the ordinal occupies the low
    /// `kDoomLineOrdinalBits` and the position the high bits. Both fields are
    /// asserted in range; a position large enough to touch the sign bit would
    /// be a runaway counter and asserts too.
    ///
    /// @param position The triggering part's submatch counter at the trigger
    ///        (`>= 0`).
    /// @param ordinal  The part's invocation ordinal
    ///        (`0 <= ordinal < HashRequestGenerator::kMaxSplitParts`).
    /// @return The packed doom-line value.
    /// @see `lowerDoomLine`, `doomLinePosition`, `doomLineOrdinal`.
    inline int64_t packDoomLine(int64_t position, int32_t ordinal) {
        assert(position >= 0 && "packDoomLine: position is a submatch tally");
        assert(position < (int64_t(1) << (62 - kDoomLineOrdinalBits))
            && "packDoomLine: position overflows the packed field");
        assert(ordinal >= 0 && ordinal < HashRequestGenerator::kMaxSplitParts
            && "packDoomLine: ordinal outside the part-count ceiling");
        return (position << kDoomLineOrdinalBits)
            | static_cast<int64_t>(ordinal);
    }

    /// @brief The position half of a packed doom line.
    ///
    /// @details Defined for the sentinel too: `kNoDoomLine`'s position is
    /// above every reachable submatch count, which is exactly the "no line —
    /// never stop" reading the burst gate wants.
    ///
    /// @param line A packed doom-line value (or `kNoDoomLine`).
    /// @return The trigger position (the winning part's submatch counter).
    inline int64_t doomLinePosition(int64_t line) {
        return line >> kDoomLineOrdinalBits;
    }

    /// @brief The part-ordinal half of a packed doom line — the winner.
    ///
    /// @param line A packed doom-line value; must not be the sentinel (a
    ///        winner exists only once a trigger fired).
    /// @return The winning part's invocation ordinal.
    inline int32_t doomLineOrdinal(int64_t line) {
        assert(line != kNoDoomLine
            && "doomLineOrdinal: no winner on the sentinel line");
        return static_cast<int32_t>(
            line & ((int64_t(1) << kDoomLineOrdinalBits) - 1));
    }

    /// @brief Atomically retain the smaller packed stop-line value.
    ///
    /// @details
    /// Implements a monotone minimum for callers exercising explicit stop-line
    /// selection. Relaxed ordering is sufficient for this independent scalar.
    ///
    /// @param line Shared packed value to reduce.
    /// @param packed Candidate produced by packDoomLine.
    /// @return Nothing.
    /// @invariant The stored value never increases.
    /// @see packDoomLine.
    inline void lowerDoomLine(std::atomic<int64_t>& line, int64_t packed) {
        int64_t current = line.load(std::memory_order_relaxed);
        while (packed < current
            && !line.compare_exchange_weak(current, packed,
                std::memory_order_relaxed)) {
        }
    }


}
