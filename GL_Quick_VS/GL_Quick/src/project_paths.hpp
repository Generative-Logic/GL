/* Generative Logic : A deterministic reasoning and knowledge generation engine.
 Copyright(C) 2025-2026 Generative Logic UG(haftungsbeschr�nkt)

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

#include <filesystem>

namespace gl {

/// @brief Resolve a project root from a deployed native executable path.
///
/// @details The executable belongs to `<root>/GL_Quick_VS/GL_Quick/` on
/// every supported platform. Lexical normalization removes dot components;
/// the build directory and process working directory are not inputs.
///
/// @param executable Absolute path to the running native executable.
/// @return Absolute root of the executable's deployed project tree.
/// @invariant A relative path or an unexpected executable layout asserts.
/// @see projectRoot
std::filesystem::path projectRootFromExecutable(
    const std::filesystem::path& executable);

/// @brief Return the running executable's project root.
///
/// @details The operating system supplies the executable location once.
/// The cached absolute root is shared by configuration, definitions, theorem
/// pools, simple facts and proof exports, including pre-main path objects.
/// Moving a binary to another project tree moves all of these paths together.
///
/// @return Process-lifetime root for this executable's deployed project tree.
/// @invariant Executable discovery must succeed without truncation or fallback.
/// @see projectRootFromExecutable
const std::filesystem::path& projectRoot();

} // namespace gl
