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

#include "project_paths.hpp"

#include <cassert>
#include <cstdint>
#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#elif defined(__APPLE__)
#  include <mach-o/dyld.h>
#else
#  include <unistd.h>
#endif

namespace gl {

/// @brief Resolve a project root from a deployed native executable path.
///
/// @details Validate the fixed distribution layout after lexical normalization.
/// Neither source-file locations nor the process working directory select data.
///
/// @param executable Absolute path to the running native executable.
/// @return Absolute root of the executable's deployed project tree.
/// @invariant A relative path or an unexpected executable layout asserts.
/// @see projectRoot
std::filesystem::path projectRootFromExecutable(
    const std::filesystem::path& executable) {
    assert(executable.is_absolute() && "native executable path must be absolute");
    const auto path = executable.lexically_normal();
    assert((path.filename() == "gl_quick.exe" || path.filename() == "gl_quick")
        && "unexpected native executable filename");
    assert(path.parent_path().filename() == "GL_Quick"
        && path.parent_path().parent_path().filename() == "GL_Quick_VS"
        && "native executable must be inside GL_Quick_VS/GL_Quick");
    return path.parent_path().parent_path().parent_path();
}

/// @brief Return the running executable's project root.
///
/// @details Discover the actual executable through the platform API and cache
/// its deployed root. The result is independent of invocation spelling and
/// working directory, and remains valid for the complete process lifetime.
///
/// @return Process-lifetime root for this executable's deployed project tree.
/// @invariant Executable discovery must succeed without truncation or fallback.
/// @see projectRootFromExecutable
const std::filesystem::path& projectRoot() {
    static const std::filesystem::path root = [] {
#ifdef _WIN32
        wchar_t executable[32768];
        const DWORD length = GetModuleFileNameW(nullptr, executable, 32768);
        assert(length > 0 && length < 32768 && "native executable discovery failed");
        return projectRootFromExecutable(std::filesystem::path(executable));
#elif defined(__APPLE__)
        char executable[32768];
        uint32_t capacity = sizeof(executable);
        const int result = _NSGetExecutablePath(executable, &capacity);
        assert(result == 0 && "native executable discovery failed");
        return projectRootFromExecutable(std::filesystem::canonical(executable));
#else
        char executable[32768];
        const ssize_t length = readlink("/proc/self/exe", executable, sizeof(executable));
        assert(length > 0 && length < static_cast<ssize_t>(sizeof(executable))
            && "native executable discovery failed");
        return projectRootFromExecutable(std::filesystem::path(executable, executable + length));
#endif
    }();
    return root;
}

} // namespace gl
