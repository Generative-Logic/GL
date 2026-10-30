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
#include "../memory_infra/scratch_arena.hpp"
namespace gl {
/// @brief One accumulated batch NormKey — a `Codec<NormKey>` byte blob on
        ///        the caller's gen-scratch byte-bump tier.
        ///
        /// @details
        /// The install-site permutation loop copies each permutation's normalized
        /// key (`Codec<NormKey>` bytes: `int16 numberExpressions ++ int16 length
        /// ++ length×int16 data`) onto the byte-bump tier and records its
        /// `(offset, len)` here; the whole run is then handed to
        /// @ref insertRemainingArgsNormKeyBatch. The byte tier survives the loop's
        /// self-framing helpers (`appendLmvIdsRecord`), so the blobs stay valid
        /// until the batch call.
        ///
        /// @see @ref insertRemainingArgsNormKeyBatch.
        struct RemArgsBatchBlob {
            /// @brief Byte-bump offset of the serialized NormKey blob.
            ArenaOffset off;
            /// @brief Blob byte length.
            int32_t len;
        };
}
