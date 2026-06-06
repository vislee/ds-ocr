# ds_ocr — DeepSeek-OCR Pure C Inference Engine
# Makefile

CC = gcc
CFLAGS_BASE = -Wall -Wextra -O3 -march=native
LDFLAGS = -lm -lpthread

# Platform detection
UNAME_S := $(shell uname -s)

# Source files (shared between main binary and test binary)
SRCS = ds_ocr.c ds_kernels.c ds_kernels_generic.c ds_kernels_neon.c ds_kernels_avx.c \
       ds_image.c ds_visual_tokenizer.c ds_deep_encoder.c ds_moe_decoder.c \
       ds_tokenizer.c ds_safetensors.c
OBJS = $(SRCS:.c=.o)
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
	@echo "  make blas        - With BLAS acceleration (Accelerate/OpenBLAS)"
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
BLAS_CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DACCELERATE_NEW_LAPACK
BLAS_LDFLAGS = -lm -lpthread -framework Accelerate -framework ApplicationServices
else
BLAS_CFLAGS = $(CFLAGS_BASE) -DUSE_BLAS -DUSE_OPENBLAS -I/usr/include/openblas
BLAS_LDFLAGS = -lm -lpthread -lopenblas
endif

blas: clean
	$(MAKE) $(TARGET) CFLAGS="$(BLAS_CFLAGS)" LDFLAGS="$(BLAS_LDFLAGS)"
	@echo ""
	@echo "Built with BLAS backend"

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
	rm -f $(OBJS) main.o test.o $(TARGET) $(TEST_TARGET)

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
ds_moe_decoder.o: ds_moe_decoder.c ds_moe_decoder.h ds_kernels.h ds_safetensors.h
ds_tokenizer.o: ds_tokenizer.c ds_tokenizer.h
ds_safetensors.o: ds_safetensors.c ds_safetensors.h
main.o: main.c ds_ocr.h ds_kernels.h
test.o: test.c ds_ocr.h ds_kernels.h ds_safetensors.h ds_tokenizer.h ds_image.h
