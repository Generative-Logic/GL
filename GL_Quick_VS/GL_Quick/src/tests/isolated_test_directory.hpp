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
#include "test_harness.hpp"
#include "../project_paths.hpp"
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>
#ifdef _WIN32
#include <process.h>
#else
#include <spawn.h>
#include <sys/wait.h>
extern char** environ;
#endif

namespace gl::tests {

// Tests run sequentially. Relative writer paths remain inside this unique
// directory, including when a production assertion terminates the process.
class IsolatedTestDirectory {
    std::filesystem::path previous_ = std::filesystem::current_path();
public:
    std::filesystem::path root;
    explicit IsolatedTestDirectory(const char* label) {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        root = previous_ / ".debug" / (std::string(label) + "_" + std::to_string(stamp));
        ASSERT_TRUE(std::filesystem::create_directories(root));
        std::filesystem::current_path(root);
    }
    ~IsolatedTestDirectory() { std::filesystem::current_path(previous_); }
};

// projectRoot is executable-relative and cached before the test. Exercise an
// absolute-path exporter in a child deployed to its own project tree; an abort
// can leave only test-owned artifacts, never files in the live project.
inline void runInIsolatedProject(const char* testName) {
    namespace fs = std::filesystem;
    IsolatedTestDirectory directory("export_test_project");
    const auto liveBinaryDirectory = projectRoot() / "GL_Quick_VS" / "GL_Quick";
    const auto binaryDirectory = directory.root / "GL_Quick_VS" / "GL_Quick";
    fs::create_directories(binaryDirectory);
#ifdef _WIN32
    const char* executableName = "gl_quick.exe";
#else
    const char* executableName = "gl_quick";
#endif
    const auto executable = binaryDirectory / executableName;
    fs::copy_file(liveBinaryDirectory / executableName, executable);
    for (const auto& entry : fs::directory_iterator(liveBinaryDirectory)) {
        if (entry.path().extension() == ".dll")
            fs::copy_file(entry.path(), binaryDirectory / entry.path().filename());
    }
    fs::create_directories(directory.root / "files");
    fs::copy(projectRoot() / "files" / "config", directory.root / "files" / "config",
        fs::copy_options::recursive);
    fs::copy(projectRoot() / "files" / "definitions", directory.root / "files" / "definitions",
        fs::copy_options::recursive);
    fs::create_directories(directory.root / "files" / "GL_binaries");
    std::vector<std::string> environment;
#ifdef _WIN32
    char** inherited = _environ;
#else
    char** inherited = environ;
#endif
    for (char** entry = inherited; *entry; ++entry) {
        const std::string value(*entry);
        if (value.rfind("GL_UNIT_TEST_ONLY=", 0) != 0
            && value.rfind("GL_UNIT_TEST_WORKER=", 0) != 0
            && value.rfind("GL_ISOLATED_EXPORT_ROOT=", 0) != 0) environment.push_back(value);
    }
    environment.push_back(std::string("GL_UNIT_TEST_ONLY=") + testName);
    environment.push_back("GL_ISOLATED_EXPORT_ROOT=" + directory.root.string());
    std::vector<char*> envp;
    for (auto& value : environment) envp.push_back(value.data());
    envp.push_back(nullptr);
    std::string binary = executable.string();
    std::string flag = "--unit-tests";
    std::string backendFlag = "--phase2-backend";
    std::string backend = includeCudaRouteForChildTests() ? "cuda" : "cpu";
#ifdef _WIN32
    // The C runtime joins the arguments into one command line without quoting
    // them, so the executable's path is quoted: a project folder containing a
    // space would otherwise split it into several arguments.
    std::string quotedBinary = "\"" + binary + "\"";
    char* arguments[] = {quotedBinary.data(), flag.data(), backendFlag.data(), backend.data(), nullptr};
    const auto result = _spawnve(_P_WAIT, binary.c_str(), arguments, envp.data());
    ASSERT_EQ(result, 0);
#else
    char* arguments[] = {binary.data(), flag.data(), backendFlag.data(), backend.data(), nullptr};
    pid_t child;
    ASSERT_EQ(posix_spawn(&child, binary.c_str(), nullptr, nullptr, arguments, envp.data()), 0);
    int status = 0;
    ASSERT_EQ(waitpid(child, &status, 0), child);
    ASSERT_TRUE(WIFEXITED(status));
    ASSERT_EQ(WEXITSTATUS(status), 0);
#endif
}
} // namespace gl::tests
