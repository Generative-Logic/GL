# Licensing 📜

Generative Logic is dual-licensed under the AGPLv3 and a Commercial License.

**1. AGPLv3 License (Open-Source)**

For open-source projects, academic research, and personal use, Generative Logic is licensed under the AGPLv3. 
This license requires that any derivative works or applications that use this software must also be made open-source under the same terms. The full license text is available in the [LICENSE](LICENSE) file.

**2. Commercial License**

If you wish to use Generative Logic in a proprietary, closed-source commercial application, a commercial license is required. This license frees you from the obligations of the AGPLv3 and includes enterprise-grade features such as a limited warranty and IP indemnity.

For more details and to request a quote, please visit our commercial licensing page:
**https://generative-logic.com/license**

**Contributing**

All contributions require a signed Contributor License Agreement (CLA).
See `legal/CONTRIBUTOR_LICENSE_AGREEMENT.md` for details.


# Run Mode & How to Run
## Run mode

There is only one run mode (previously called “Full Mode”).

End-to-end runtime (reference): ~6 minutes on a Dell G16 7630 (overall).


## Prerequisites

Python 3.9+ with the `regex` package (`pip install regex`) — required for recursive pattern matching in HTML proof graph rendering

For Windows users: the bundled native executable GL_Quick_VS/GL_Quick/gl_quick.exe

For non-Windows or if the executable is missing: a C++17 toolchain to rebuild the native component (see below)

## Quick start

Run from the repository root (where main.py lives) on Windows / macOS / Linux:
python main.py


What happens:

Python creates conjectures (parallelized).

Python calls the native prover executable
GL_Quick_VS/GL_Quick/gl_quick.exe.

Python renders the proof graph pages.

Outputs:

Generated HTML lives under: files/full_proof_graph/index.html (plus chapter*.html).

## Rebuilding the native executable (if needed)

The native prover is a C++ project compiled with Microsoft Visual Studio on Windows 11.

Windows (Visual Studio 2026)

Open GL_Quick_VS/GL_Quick.sln in Visual Studio.

Build Release x64 (recommended).

The binary will be at GL_Quick_VS/GL_Quick/gl_quick.exe (this is the path the Python code expects).

Linux / macOS (experimental)

Source code is under GL_Quick_VS/GL_Quick/src/.
If the code is portable, you can try a simple build:

cd GL_Quick_VS/GL_Quick
c++ -std=c++17 -O3 src/*.cpp -o gl_quick


If you place the binary somewhere else or name it differently, update the path in run_modes.py (function run_gl_quick()).

## Troubleshooting

FileNotFoundError: GL_Quick_VS/GL_Quick/gl_quick.exe
Rebuild the native executable or ensure the file exists at that path.

No HTML output
Check files/raw_proof_graph/* was generated and that files/full_proof_graph/ is created. Running python main.py regenerates these.

Slow run
Make sure you’re using a Release build of the native binary and modern CPU; performance varies.

## Paths recap

Entry point: main.py

Native executable (Windows): GL_Quick_VS/GL_Quick/gl_quick.exe

Native source: GL_Quick_VS/GL_Quick/src/

Output HTML: files/full_proof_graph/

# 3rd party notices

regex — © Matthew Barnett — Apache-2.0 and CNRI-Python.

**nlohmann/json** — © Niels Lohmann — Licensed under MIT.
https://github.com/nlohmann/json

