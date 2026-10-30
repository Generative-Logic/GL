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
#include <map>
#include <string>
namespace gl {
/// @brief One owned syntax node carrying its free argument types.
/// @details Deleting the node recursively deletes its two child subtrees.
struct AnalysisNode {
    std::string value;
    AnalysisNode* left = nullptr;
    AnalysisNode* right = nullptr;
    std::map<std::string, std::string> remainingArgsDefs;
    /// @brief Construct an empty analysis node.
    /// @details Child pointers start null and argument metadata starts empty.
    /// @return Constructed node.
    AnalysisNode();
    /// @brief Destroy this node's owned child subtrees.
    /// @details Each child is deleted exactly once by its parent.
    /// @return None.
    ~AnalysisNode();
};
} // namespace gl
