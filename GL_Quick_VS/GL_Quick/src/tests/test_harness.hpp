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

/// @file
/// @brief Hand-rolled unit-test harness, no external dependencies.
///
/// @details
/// Each translation unit under `src/tests/` defines tests using the
/// `TEST(suite, name) { ... }` macro. Static initializers register the
/// test functions into a global registry; `gl::tests::runAllTests()`
/// then walks the registry sequentially, reporting per-test pass/fail
/// with elapsed milliseconds. The harness aborts on the first failure.
///
/// Entry point for users is `gl_quick.exe --unit-tests`; `main.py`
/// invokes it before any pipeline work and aborts on a non-zero exit.
///
/// Performance budget: <30 s wall-clock for the full suite.
/// Per-test budget ~100 µs–500 ms.
///
/// @see `src/tests/test_harness.cpp` for the registry implementation.

#include <cstdio>
#include <vector>
#include <string>
#include <cstddef>

/// @brief Test-local byte-identical oracle for the heap form of
///        `extractExpressionUniversal`, which has no production counterpart.
///
/// @details
/// Production code reads the core name through the span form
/// `gl::extractExpressionUniversalSpan` exclusively; per
/// D-193 a prover-only heap `ce::` function is not
/// kept in the production tree, so its Rule-18 differential oracle lives here
/// instead. A pure `std::string` slice, no dependency on the production tree.
///
/// @param s Canonical MPL expression text, possibly negated.
/// @return Core name (no negation prefix / parens / brackets); empty when the
///         shape is neither `(name[...])` nor `!(name[...])`.
/// @see gl::extractExpressionUniversalSpan — the production span replacement.
inline std::string extractExpressionUniversalOracle(const std::string& s) {
    std::size_t index = s.find('[');
    if (index != std::string::npos) {
        if (!s.empty() && s[0] == '(') {
            return s.substr(1, index - 1);
        }
        else if (s.size() >= 2 && s[0] == '!' && s[1] == '(') {
            return s.substr(2, index - 2);
        }
    }
    return std::string();
}

namespace gl {
namespace tests {

    /// Function pointer signature every test body satisfies.
    using TestFn = void (*)();

    /// Registry record for one test. The `suite` and `name` strings
    /// are produced by the `TEST(...)` macro's stringify; they live in
    /// static storage so storing raw `const char*` is safe.
    struct TestEntry {
        const char* suite;
        const char* name;
        TestFn      fn;
    };

    /// Insert a test into the registry. Called from a static initializer
    /// emitted by the `TEST(suite, name)` macro at file scope. Order
    /// across translation units is unspecified by C++ but the runner
    /// does not rely on it — every test owns its own setup.
    void registerTest(const char* suite, const char* name, TestFn fn);

    /// Throw an internal failure exception (caught by `runAllTests`).
    /// Reports the failing expression text and source location to
    /// stderr before throwing so the location is visible even if the
    /// catch site swallows the exception. Marked `[[noreturn]]` so the
    /// calling assertion macro is treated as terminating by control
    /// flow analysis.
    [[noreturn]] void fail(const char* file, int line, const char* expr);

    /// Walk the registry, run each test, abort on first failure.
    /// @return 0 if all tests pass, 1 on the first failure (or on any
    /// unexpected exception escaping a test body).
    int runAllTests();

}  // namespace tests
}  // namespace gl

/// Define a test. Expands to a forward declaration, a static
/// registration variable, and the body. The two stringified arguments
/// become the suite and name in the registry record.
///
/// Example:
/// @code
/// TEST(memory, namemap_main_id_one) {
///     gl::NameMap nm;
///     ASSERT_EQ(nm.encode("main"), int16_t(1));
/// }
/// @endcode
#define TEST(suite, name)                                                    \
    static void suite##_##name##_body();                                     \
    static int  suite##_##name##_reg = (::gl::tests::registerTest(           \
        #suite, #name, &suite##_##name##_body), 0);                          \
    static void suite##_##name##_body()

/// True-assertion. Calls `gl::tests::fail` with the stringified
/// expression on miss, terminating the test body.
#define ASSERT_TRUE(x)                                                       \
    do {                                                                     \
        if (!(x))                                                            \
            ::gl::tests::fail(__FILE__, __LINE__, #x);                       \
    } while (0)

#define ASSERT_FALSE(x)                                                      \
    do {                                                                     \
        if ((x))                                                             \
            ::gl::tests::fail(__FILE__, __LINE__, "!(" #x ")");              \
    } while (0)

#define ASSERT_EQ(a, b)                                                      \
    do {                                                                     \
        if (!((a) == (b)))                                                   \
            ::gl::tests::fail(__FILE__, __LINE__, #a " == " #b);             \
    } while (0)

#define ASSERT_NE(a, b)                                                      \
    do {                                                                     \
        if ((a) == (b))                                                      \
            ::gl::tests::fail(__FILE__, __LINE__, #a " != " #b);             \
    } while (0)

#define ASSERT_LT(a, b)                                                      \
    do {                                                                     \
        if (!((a) < (b)))                                                    \
            ::gl::tests::fail(__FILE__, __LINE__, #a " < " #b);              \
    } while (0)

#define ASSERT_GE(a, b)                                                      \
    do {                                                                     \
        if (!((a) >= (b)))                                                   \
            ::gl::tests::fail(__FILE__, __LINE__, #a " >= " #b);             \
    } while (0)

/// Catch-and-verify a specific exception type. Body runs, and the macro
/// passes only if exactly `ex_type` (or derived) was thrown.
#define ASSERT_THROW(stmt, ex_type)                                          \
    do {                                                                     \
        bool _gl_threw = false;                                              \
        try { stmt; }                                                        \
        catch (const ex_type&) { _gl_threw = true; }                         \
        catch (...) {}                                                       \
        if (!_gl_threw)                                                      \
            ::gl::tests::fail(__FILE__, __LINE__,                            \
                              #stmt " did not throw " #ex_type);             \
    } while (0)
