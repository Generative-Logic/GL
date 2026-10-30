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
#include "../gpu/request_frontier_store.hpp"
#include <filesystem>

namespace gl::tests {
    struct RequestFrontierTestFiles {
        PositionedFile files[2];
        explicit RequestFrontierTestFiles(const char* label) {
            const auto directory = std::filesystem::path(".debug") / label;
            std::filesystem::create_directories(directory);
            for (unsigned i = 0; i < 2; ++i) {
                files[i].open(directory / (i == 0 ? "frontier0.bin" : "frontier1.bin"));
                files[i].preallocate(0);
                files[i].writeAt(0, nullptr, 0);
            }
        }
        gpu::RequestFrontierStorageOptions options(int32_t rows) {
            return {{&files[0], &files[1]}, rows};
        }
    };
}
