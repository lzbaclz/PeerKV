# Build the standalone multi-device CUDA benches.
# Defaults target A100 (sm_80); override with ARCH=sm_90 etc. on H100.

NVCC      ?= /usr/local/cuda-12.8/bin/nvcc
ARCH      ?= sm_80
OUT       ?= ../build
CXXSTD    ?= -std=c++14
NVCC_OPTS ?= -O3 -arch=$(ARCH) $(CXXSTD)

ATTN_BIN := $(OUT)/peer_fused_bench
FULL_BIN := $(OUT)/peer_fused_decoder_bench

.PHONY: all clean attn full
all: attn full

$(OUT):
	mkdir -p $(OUT)

attn: $(ATTN_BIN)
$(ATTN_BIN): peer_fused_bench.cu | $(OUT)
	$(NVCC) $(NVCC_OPTS) $< -o $@

full: $(FULL_BIN)
$(FULL_BIN): peer_fused_decoder.cu | $(OUT)
	$(NVCC) $(NVCC_OPTS) $< -lcublas -o $@

clean:
	rm -f $(ATTN_BIN) $(FULL_BIN)
