# Generative Logic — native build (Linux / macOS).
#
# Use this Makefile to build `gl_quick` from sources under `src/` with
# parallel compilation of translation units. Run from this directory:
#
#   make           # parallel build (one .cpp per core, default -j 1)
#   make -j        # parallel build using ALL available cores (auto-detect)
#   make -j$(nproc)  # explicit core count on Linux
#   make -j$(sysctl -n hw.ncpu)  # explicit core count on macOS
#
# The build is `mimalloc`-optional. To enable mimalloc allocator override:
#   make USE_MIMALLOC=1     # (also pass -DUSE_MIMALLOC and -lmimalloc)
#
# The Windows build uses MSBuild via the Visual Studio solution and
# does NOT use this Makefile -- see `GL_Quick_VS/GL_Quick.sln`.

CXX       ?= c++
CXXSTD    ?= -std=c++17
OPT       ?= -O3
INCLUDES  := -Isrc
WARN      := -Wno-builtin-declaration-mismatch
CXXFLAGS  := $(CXXSTD) $(OPT) $(INCLUDES) $(WARN)
LDFLAGS   :=

# Default ON. Linux users without libmimalloc-dev installed will hit a
# linker error -- this is intentional: mimalloc is required for cross-
# platform proof-graph determinism with the Windows MSVC build (which
# also defines USE_MIMALLOC). To opt out (single-host runs where
# determinism doesn't matter), build with `make USE_MIMALLOC=0 -j...`.
USE_MIMALLOC ?= 1

ifeq ($(USE_MIMALLOC),1)
  CXXFLAGS += -DUSE_MIMALLOC
  LDFLAGS  += -lmimalloc
endif

# `src/*.cpp` is the production set; `src/tests/*.cpp` is the unit-test
# harness (test_harness.cpp + test_<module>.cpp), linked into the same
# `gl_quick` binary so the `--unit-tests` flag in main.cpp can dispatch
# to the registered tests. Both globs feed the same OBJS expansion;
# the patsubst rewrites `src/...` to `build/...` preserving the relative
# subdirectory, and the `build/%.o` recipe handles either depth via the
# `mkdir -p $(@D)` order-only step inside the recipe.
SRCS := $(wildcard src/*.cpp) $(wildcard src/tests/*.cpp)
OBJS := $(patsubst src/%.cpp,build/%.o,$(SRCS))
TARGET := gl_quick

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

build/%.o: src/%.cpp | build
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c $< -o $@

build:
	@mkdir -p build

clean:
	@rm -rf build $(TARGET)
