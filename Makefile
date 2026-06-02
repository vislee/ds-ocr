# ds_ocr — DeepSeek-OCR Pure C Inference Engine
# Makefile

CC = gcc
CFLAGS_BASE = -Wall -Wextra -O3 -march=native -ffast-math
LDFLAGS = -lm -lpthread

# Platform detection
UNAME_S := $(shell uname -s)

# Source files
SRCS = ds_ocr.c ds_kernels.c ds_kernels_generic.c ds_kernels_neon.c ds_kernels_avx.c \
       ds_image.c ds_visual_tokenizer.c ds_deep_encoder.c ds_moe_decoder.c \
       ds_tokenizer.c ds_safetensors.c
OBJS = $(SRCS:.c=.o)
MAIN = main.c
TARGET = ds_ocr

# Debug build flags
DEBUG_CFLAGS = -Wall -Wextra -g -O0 -DDEBUG -fsanitize=address

.PHONY: all clean debug info help blas test

# Default: show available targets
all: help

help:
	@echo "ds_ocr — DeepSeek-OCR Pure C Inference - Build Targets"
	@echo ""
	@echo "Choose a backend:"
	@echo "  make blas     - With BLAS acceleration (Accelerate/OpenBLAS)"
	@echo ""
	@echo "Other targets:"
	@echo "  make debug    - Debug build with AddressSanitizer"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make info     - Show build configuration"
	@echo ""
	@echo "Example: make blas && ./ds_ocr -d model_dir -i document.png"

# =============================================================================
# Backend: blas (Accelerate on macOS, OpenBLAS on Linux)
# =============================================================================
# BLAS backend
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
	rm -f $(OBJS) main.o $(TARGET)

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
