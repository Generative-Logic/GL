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

#include "cold_hash_map.hpp"

namespace gl {

    /// @brief The per-interner cold string table — now the byte-key
    ///        instantiation of the reusable cold-map family.
    ///
    /// @details
    /// `ColdStringTable` was a hand-rolled string interner (heap find-index +
    /// cold byte pool + location index). It is now exactly the byte-key SET of
    /// the cold-map family: `ColdHashSet<BytesKeyStore>` (`cold_hash_map.hpp`,
    /// D-165). The alias preserves the full public surface
    /// — `intern` / `lookup` / `view` / `decodeString` / `count` / `copyFrom` /
    /// `resetToFresh` / `release` / the canonical dump helpers + the
    /// `LengthsView` / `BytesView` deload facets — so the seven `LbMemory`
    /// interners, `NameMap`, and their tests compile unchanged, and the deload
    /// byte stream stays byte-identical.
    using ColdStringTable = ColdHashSet<BytesKeyStore>;

}
