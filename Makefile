# ds_ocr — DeepSeek-OCR Pure C Inference Engine
# Makefile

CC = gcc
# Apple M2+: -mcpu=apple-m2 enables FEAT_BF16 and better scheduling
# For Linux: use -march=native
CFLAGS_BASE = -Wall -Wextra -O3 -mcpu=apple-m2
LDFLAGS = -lm -lpthread

# Platform detection
UNAME_S := $(shell uname -s)

# Source files (shared between main binary and test binary)
SRCS = ds_ocr.c ds_kernels.c ds_kernels_generic.c ds_kernels_neon.c ds_kernels_avx.c \
       ds_image.c ds_visual_tokenizer.c ds_deep_encoder.c ds_moe_decoder.c \
       ds_tokenizer.c ds_safetensors.c
ifeq ($(UNAME_S),Darwin)
SRCS += ds_platform_ocr.m ds_metal.m
METALLIB = ds_metal_shaders.metallib
else
SRCS += ds_platform_ocr.c
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
BLAS_CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_APPLE_VISION -DACCELERATE_NEW_LAPACK
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
ds_metal_shaders.metallib: ds_metal_shaders.metal
	-xcrun -sdk macosx metal -std=metal3.1 -c $< -o $(@:.metallib=.air) 2>/dev/null && \
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
	$(MAKE) $(TEST_TARGET) CFLAGS="$(DEBUG_CFLAGS) -DUSE_BLAS -DACCELERATE_NEW_LAPACK" \
		LDFLAGS="-lm -lpthread -framework Accelerate -framework ApplicationServices -fsanitize=address"
	@echo ""
	./$(TEST_TARGET)

# Test binary link rule
$(TEST_TARGET): $(OBJS) test.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

test.o: test.c ds_ocr.h ds_kernels.h ds_safetensors.h ds_tokenizer.h ds_image.h
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
	$(CC) $(CFLAGS) -fobjc-arc -framework Metal -framework Foundation -c -o $@ $<

# Debug build
debug: CFLAGS = $(DEBUG_CFLAGS)
debug: LDFLAGS += -fsanitize=address
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
ds_platform_ocr.o: ds_platform_ocr.m ds_platform_ocr.h
	$(CC) $(CFLAGS) -fobjc-arc -c -o $@ $<
else
ds_platform_ocr.o: ds_platform_ocr.c ds_platform_ocr.h
	$(CC) $(CFLAGS) -c -o $@ $<
endif
test.o: test.c ds_ocr.h ds_kernels.h ds_safetensors.h ds_tokenizer.h ds_image.h
