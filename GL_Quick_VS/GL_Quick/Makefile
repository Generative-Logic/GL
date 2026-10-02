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
# A header change recompiles the units that include it, and a changed build
# configuration (USE_MIMALLOC, USE_CUDA, the compilers or their flags)
# recompiles every unit, so `make clean` is never needed for a correct build.
#
# The Windows build uses MSBuild via the Visual Studio solution and
# does NOT use this Makefile -- see `GL_Quick_VS/GL_Quick.sln`.

CXX       ?= c++
CXXSTD    ?= -std=c++17
OPT       ?= -O3
INCLUDES  := -Isrc
WARN      := -Wno-builtin-declaration-mismatch
CXXFLAGS  := $(CXXSTD) $(OPT) $(INCLUDES) $(WARN)
# -rdynamic puts the executable's own symbols into its dynamic symbol table:
# the heap profile names allocation sites through dladdr, which sees nothing
# else (its unit test resolves a probe function by name).
LDFLAGS   := -rdynamic

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

# CUDA Phase 2 route. Default OFF: a plain `make` builds the processor-only
# binary (no CUDA toolkit needed; selecting the CUDA route on that binary
# asserts at startup). `make USE_CUDA=1` compiles the two `.cu` translation
# units with nvcc for the same device targets as the Visual Studio project,
# defines GL_CUDA for every translation unit, and links the CUDA runtime.
# Needs the CUDA toolkit (13.x; `CUDA_HOME` defaults to /usr/local/cuda) and
# an NVIDIA driver on the run host. `main.py --GPU` selects the route.
USE_CUDA  ?= 0
CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
CUDA_ARCH ?= -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120

ifeq ($(USE_CUDA),1)
  CXXFLAGS += -DGL_CUDA -I$(CUDA_HOME)/include
  LDFLAGS  += -L$(CUDA_HOME)/lib64 -Wl,-rpath,$(CUDA_HOME)/lib64 -lcudart -ldl
  CU_SRCS  := $(wildcard src/gpu/*.cu) $(wildcard src/tests/*.cu)
  CU_OBJS  := $(patsubst src/%.cu,build/%.cu.o,$(CU_SRCS))
  # nvcc gets the same standard, optimisation level, include path and
  # preprocessor definitions as the host compiler; host-only warning flags
  # ride -Xcompiler.
  NVCCFLAGS := $(CXXSTD) $(OPT) -lineinfo $(INCLUDES) $(filter -D%,$(CXXFLAGS)) $(CUDA_ARCH) -Xcompiler "$(WARN)"
  CUDA_CONFIG := $(NVCC) $(NVCCFLAGS)
else
  CU_OBJS  :=
  CUDA_CONFIG :=
endif

# Header dependencies: every compile also writes `build/<unit>.d`, the
# project headers the unit includes (`-MP` adds an empty rule per header, so
# a deleted header does not stop the build). The `-include` at the end makes
# a header change recompile exactly the units that include it.
DEPFLAGS := -MMD -MP

# Build configuration: the compilers and their flags, which USE_MIMALLOC and
# USE_CUDA change, are recorded in `build/config.stamp`. The stamp is
# rewritten only when they differ from the previous build, and every object
# and the link depend on it: switching between the processor and the CUDA
# build, or turning mimalloc on or off, recompiles everything instead of
# linking objects compiled for another configuration, while an unchanged
# configuration leaves every object current.
CONFIG_STAMP := build/config.stamp
BUILD_CONFIG := $(CXX) $(CXXFLAGS) | $(LDFLAGS) | $(CUDA_CONFIG)
$(shell mkdir -p build; [ "$$(cat $(CONFIG_STAMP) 2>/dev/null)" = '$(BUILD_CONFIG)' ] || printf '%s\n' '$(BUILD_CONFIG)' > $(CONFIG_STAMP))

# `src/*.cpp` is the production set; `src/infra/*.cpp` is the diagnostic
# / instrumentation set (hashburst dump, runtime measurement tracker —
# code the prover calls into but that is operationally independent of
# the algorithm); `src/memory_infra/*.cpp` is the statification set
# (static memory hierarchy, paged containers, deload, steward — one
# component per file); `src/gpu/*.cpp` is the portable host-side GPU projection
# support (the sealing unit compiles to nothing without GL_CUDA; the CUDA
# `.cu` translation units join the build only under USE_CUDA=1, see below);
# `src/tests/*.cpp` is the unit-test harness
# (test_harness.cpp + test_<module>.cpp), linked into the same
# `gl_quick` binary so the `--unit-tests` flag in main.cpp can dispatch
# to the registered tests. All five globs feed the same OBJS expansion;
# the patsubst rewrites `src/...` to `build/...` preserving the relative
# subdirectory, and the `build/%.o` recipe handles arbitrary depth via
# the `mkdir -p $(@D)` order-only step inside the recipe.
SRCS := $(wildcard src/*.cpp) $(wildcard src/infra/*.cpp) $(wildcard src/memory_infra/*.cpp) $(wildcard src/gpu/*.cpp) $(wildcard src/tests/*.cpp)
OBJS := $(patsubst src/%.cpp,build/%.o,$(SRCS))
TARGET := gl_quick

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS) $(CU_OBJS) $(CONFIG_STAMP)
	$(CXX) $(OBJS) $(CU_OBJS) -o $@ $(LDFLAGS)

build/%.o: src/%.cpp $(CONFIG_STAMP) | build
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

build/%.cu.o: src/%.cu $(CONFIG_STAMP) | build
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $(DEPFLAGS) -MF $(@:.o=.d) -MT $@ -c $< -o $@

build:
	@mkdir -p build

clean:
	@rm -rf build $(TARGET)

# Header dependencies written by the compiles (none before the first build).
-include $(OBJS:.o=.d) $(CU_OBJS:.o=.d)
