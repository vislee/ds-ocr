# ds_ocr — DeepSeek-OCR Pure C Inference Engine
# Makefile

CC = gcc

# Platform-specific CPU tuning
# Apple Silicon: auto-detect M1/M2/M3/M4 for optimal -mcpu flag
# Linux/x86: use -march=native
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S),Darwin)
  # Detect Apple Silicon chip via sysctl
  APPLE_CHIP := $(shell sysctl -n hw.optional.arm64 2>/dev/null)
  ifeq ($(APPLE_CHIP),1)
    # Check for FEAT_BF16 support (M2+) vs M1
    HAS_BF16 := $(shell sysctl -n hw.optional.arm_FeatBF16 2>/dev/null)
    ifeq ($(HAS_BF16),1)
      MARCH_FLAG = -mcpu=apple-m2
    else
      MARCH_FLAG = -mcpu=apple-m1
    endif
  else
    MARCH_FLAG = -mcpu=apple-m2
  endif
else ifeq ($(UNAME_M),x86_64)
  MARCH_FLAG = -march=native
else ifeq ($(UNAME_M),aarch64)
  MARCH_FLAG = -mcpu=native
else
  MARCH_FLAG =
endif

CFLAGS_BASE = -Wall -Wextra -O3 $(MARCH_FLAG)
LDFLAGS = -lm -lpthread

# Platform detection
UNAME_S := $(shell uname -s)

# Source files (shared between main binary and test binary)
SRCS = ds_ocr.c ds_kernels.c ds_kernels_generic.c ds_kernels_neon.c ds_kernels_avx.c \
       ds_image.c ds_visual_tokenizer.c ds_deep_encoder.c ds_moe_decoder.c \
       ds_tokenizer.c ds_safetensors.c ds_quantize.c ds_dash.c
ifeq ($(UNAME_S),Darwin)
SRCS += ds_pdf.m ds_platform_ocr.m ds_metal.m
METALLIB = ds_metal_shaders.metallib
else
SRCS += ds_pdf.c ds_platform_ocr.c
METALLIB =
endif
OBJS = $(SRCS:.c=.o)
OBJS := $(OBJS:.m=.o)
MAIN = main.c
TARGET = ds_ocr
TEST_TARGET = test_ds_ocr
TEST_SRC = test.c

# Debug build flags
DEBUG_CFLAGS = -Wall -Wextra -g -O0 -DDEBUG -fsanitize=address

.PHONY: all clean debug info help blas test test_debug

# Default: show available targets
all: help

help:
	@echo "ds_ocr — DeepSeek-OCR Pure C Inference - Build Targets"
	@echo ""
	@echo "Choose a backend:"
	@echo "  make blas        - With BLAS + Metal GPU acceleration"
	@echo ""
	@echo "Testing:"
	@echo "  make test        - Build and run all tests (BLAS backend)"
	@echo "  make test_debug  - Build and run tests with AddressSanitizer"
	@echo ""
	@echo "Other targets:"
	@echo "  make debug       - Debug build with AddressSanitizer"
	@echo "  make clean       - Remove build artifacts"
	@echo "  make info        - Show build configuration"
	@echo ""
	@echo "Examples:"
	@echo "  make blas && ./ds_ocr -d model_dir -i document.png"
	@echo "  make test"
	@echo "  ./test_ds_ocr test_kernels   # Run specific test suite"

# =============================================================================
# BLAS backend (Accelerate on macOS, OpenBLAS on Linux)
# =============================================================================
ifeq ($(UNAME_S),Darwin)
# ACCELERATE_NEW_LAPACK maps cblas_* to the ILP64 headers, which are annotated
# API_AVAILABLE(macos(13.3)). Declaring the deployment target explicitly keeps
# -Wunguarded-availability-new quiet without per-call __builtin_available guards.
MACOS_MIN = -mmacosx-version-min=13.3
BLAS_CFLAGS = $(CFLAGS_BASE) $(MACOS_MIN) -DUSE_BLAS -DUSE_APPLE_VISION -DACCELERATE_NEW_LAPACK
BLAS_LDFLAGS = -lm -lpthread -framework Accelerate -framework ApplicationServices -framework Foundation -framework Vision -framework ImageIO -framework Metal -framework CoreGraphics
else
BLAS_CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas
BLAS_LDFLAGS = -lm -lpthread -lopenblas
endif

# Metal shader compilation (macOS only)
# Use runtime source compilation as fallback if metallib build fails.
# The ds_metal.m wrapper will compile from .metal source at runtime if
# precompiled metallib is not available.
ifeq ($(UNAME_S),Darwin)
# Metal shader version: auto-detect macOS for max compatibility
# metal3.1 requires macOS 14+, metal3.0 works on macOS 13+
MACOS_MAJOR := $(shell sw_vers -productVersion 2>/dev/null | cut -d. -f1)
ifeq ($(MACOS_MAJOR),)
  METAL_STD_VAL = metal3.0
else ifeq ($(shell test $(MACOS_MAJOR) -ge 14 && echo yes),yes)
  METAL_STD_VAL = metal3.1
else
  METAL_STD_VAL = metal3.0
endif

ds_metal_shaders.metallib: ds_metal_shaders.metal
	-xcrun -sdk macosx metal -std=$(METAL_STD_VAL) -c $< -o $(@:.metallib=.air) 2>/dev/null && \
	xcrun -sdk macosx metallib $(@:.metallib=.air) -o $@ && \
	rm -f $(@:.metallib=.air) || \
	(echo "Note: Metal shader precompilation skipped (will compile at runtime)" && rm -f $(@:.metallib=.air))
endif

blas: clean
	$(MAKE) $(METALLIB)
	$(MAKE) $(TARGET) CFLAGS="$(BLAS_CFLAGS)" LDFLAGS="$(BLAS_LDFLAGS)"
	@echo ""
	@echo "Built with BLAS + Metal GPU backend"

# =============================================================================
# Test suite
# =============================================================================

# Build test binary with BLAS
test: clean
	$(MAKE) $(TEST_TARGET) CFLAGS="$(BLAS_CFLAGS)" LDFLAGS="$(BLAS_LDFLAGS)"
	@echo ""
	./$(TEST_TARGET)

# Build test binary with debug + ASAN
test_debug: clean
	$(MAKE) $(TEST_TARGET) CFLAGS="$(DEBUG_CFLAGS) $(MACOS_MIN) -DUSE_BLAS -DACCELERATE_NEW_LAPACK" \
		LDFLAGS="-lm -lpthread -framework Accelerate -framework ApplicationServices -fsanitize=address"
	@echo ""
	./$(TEST_TARGET)

# Test binary link rule
$(TEST_TARGET): $(OBJS) test.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test.o: test.c ds_ocr.h ds_kernels.h ds_safetensors.h ds_tokenizer.h ds_image.h
	$(CC) $(CFLAGS) -c -o $@ $<

# INT4 kernel test
test_int4: $(OBJS) test_int4.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test_int4.o: test_int4.c ds_ocr.h ds_kernels.h ds_quantize.h
	$(CC) $(CFLAGS) -c -o $@ $<

# =============================================================================
# Build rules
# =============================================================================
$(TARGET): $(OBJS) main.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c ds_ocr.h ds_kernels.h
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.m ds_platform_ocr.h
	$(CC) $(CFLAGS) -fobjc-arc -c -o $@ $<

ds_metal.o: ds_metal.m ds_metal.h ds_metal_shaders.metal
	$(CC) $(CFLAGS) -fobjc-arc -c -o $@ $<

# Debug build
debug: CFLAGS = $(DEBUG_CFLAGS)
ifeq ($(UNAME_S),Darwin)
debug: LDFLAGS = -lm -lpthread -framework Metal -framework Foundation -framework CoreGraphics -fsanitize=address
else
debug: LDFLAGS += -fsanitize=address
endif
debug:
	@$(MAKE) clean
	@$(MAKE) $(TARGET) CFLAGS="$(CFLAGS)" LDFLAGS="$(LDFLAGS)"

# =============================================================================
# Utilities
# =============================================================================
clean:
	rm -f $(OBJS) main.o test.o $(TARGET) $(TEST_TARGET) ds_metal_shaders.metallib ds_metal_shaders.air

info:
	@echo "Platform: $(UNAME_S)"
	@echo "Compiler: $(CC)"
	@echo ""
ifeq ($(UNAME_S),Darwin)
	@echo "Backend: blas (Apple Accelerate)"
else
	@echo "Backend: blas (OpenBLAS)"
endif

# =============================================================================
# Dependencies
# =============================================================================
ds_ocr.o: ds_ocr.c ds_ocr.h ds_kernels.h ds_safetensors.h ds_image.h ds_visual_tokenizer.h ds_deep_encoder.h ds_moe_decoder.h ds_tokenizer.h
ds_kernels.o: ds_kernels.c ds_kernels.h ds_kernels_impl.h
ds_kernels_generic.o: ds_kernels_generic.c ds_kernels_impl.h
ds_kernels_neon.o: ds_kernels_neon.c ds_kernels_impl.h
ds_kernels_avx.o: ds_kernels_avx.c ds_kernels_impl.h
ds_image.o: ds_image.c ds_image.h
ds_visual_tokenizer.o: ds_visual_tokenizer.c ds_visual_tokenizer.h ds_kernels.h ds_safetensors.h
ds_deep_encoder.o: ds_deep_encoder.c ds_deep_encoder.h ds_kernels.h ds_safetensors.h
ds_moe_decoder.o: ds_moe_decoder.c ds_moe_decoder.h ds_kernels.h ds_safetensors.h ds_metal.h
ds_tokenizer.o: ds_tokenizer.c ds_tokenizer.h
ds_safetensors.o: ds_safetensors.c ds_safetensors.h
main.o: main.c ds_ocr.h ds_kernels.h
ifeq ($(UNAME_S),Darwin)
ds_pdf.o: ds_pdf.m ds_pdf.h
	$(CC) $(CFLAGS) -fobjc-arc -c -o $@ $<
ds_platform_ocr.o: ds_platform_ocr.m ds_platform_ocr.h
	$(CC) $(CFLAGS) -fobjc-arc -c -o $@ $<
else
ds_pdf.o: ds_pdf.c ds_pdf.h
	$(CC) $(CFLAGS) -c -o $@ $<
ds_platform_ocr.o: ds_platform_ocr.c ds_platform_ocr.h
	$(CC) $(CFLAGS) -c -o $@ $<
endif
test.o: test.c ds_ocr.h ds_kernels.h ds_safetensors.h ds_tokenizer.h ds_image.h

# ════════════════════════════════════════════════════════════════════
# 单元测试 + 覆盖率（r3）: make test_unit && make coverage
# test_unit 为 unity 编译（直接 #include ds_dash.c / ds_ocr.c 以触达
# static 函数），其余目标文件正常链接；--coverage 出 gcov/llvm-cov 数据。
# ════════════════════════════════════════════════════════════════════
COV_FLAGS = --coverage
UNIT_SRCS = ds_kernels.c ds_kernels_generic.c ds_kernels_neon.c ds_kernels_avx.c \
            ds_image.c ds_visual_tokenizer.c ds_deep_encoder.c ds_moe_decoder.c \
            ds_tokenizer.c ds_safetensors.c ds_quantize.c

test_unit: test_unit.c ds_dash.c ds_ocr.c $(UNIT_SRCS) ds_metal.m ds_kernels.h ds_ocr.h ds_dash.h
	$(CC) $(CFLAGS) $(COV_FLAGS) -fobjc-arc -c -o ds_metal_ut.o ds_metal.m
	$(CC) $(CFLAGS) $(COV_FLAGS) -c -o test_unit-tu.o test_unit.c
	@set -e; for f in $(UNIT_SRCS); do \
	    $(CC) $(CFLAGS) $(COV_FLAGS) -c -o test_unit-$${f%.c}.o $$f; \
	done
	$(CC) $(COV_FLAGS) -o $@ test_unit-tu.o test_unit-ds_*.o ds_metal_ut.o \
	  -lm -lpthread -framework Accelerate -framework Foundation -framework Metal -framework CoreGraphics

UT_E2E_FILTER = test_e2e_all_models

coverage-unit: test_unit
	@rm -f *.gcda
	UT_SKIP_E2E=1 ./test_unit
	@echo "── 覆盖率（单元测试，无 e2e）──"
	@for g in test_unit-ds_*.gcda; do \
	    [ -f "$$g" ] && xcrun llvm-cov gcov "$$g" 2>/dev/null | grep -E "Lines executed|^File" | paste - - ; \
	done

coverage: test_unit
	@rm -f *.gcda *.gcno            # 清掉旧二进制的覆盖数据（防 corrupt arc tag）
	./test_unit
	@echo "── 覆盖率报告（核心文件，行覆盖）──"
	@for g in test_unit-ds_*.gcda; do \
	    [ -f "$$g" ] && xcrun llvm-cov gcov "$$g" 2>/dev/null | grep -E "Lines executed|^File" | paste - - ; \
	done
