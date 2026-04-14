CXX := nvcc
CXXFLAGS := -std=c++17 -Iinclude

COMMON_OPS := \
	ops/gpt2.cpp \
	ops/transformer_block.cpp \
	ops/attention.cpp \
	ops/mlp.cpp \
	ops/workspace.cpp \
	ops/config.cpp \
	ops/weights.cpp

GEN_OPS := ops/generate.cpp

COMMON_KERNELS := \
	kernels/add_bias.cu \
	kernels/gemm.cu \
	kernels/layernorm.cu \
	kernels/causal_mask.cu \
	kernels/merge_heads.cu \
	kernels/gelu.cu \
	kernels/embedding.cu \
	kernels/residual_add.cu \
	kernels/softmax.cu \
	kernels/scale.cu \
	kernels/qkv_split_reshape.cu \
	kernels/transpose.cu

BIN_DIR := build

.PHONY: all main bench real_smoke kv_cache_test gpt2_test generate_test test clean

all: main bench

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

main: $(BIN_DIR)
	$(CXX) $(CXXFLAGS) main.cpp \
		$(GEN_OPS) $(COMMON_OPS) $(COMMON_KERNELS) \
		-o $(BIN_DIR)/gpt2_main

bench: $(BIN_DIR)
	$(CXX) $(CXXFLAGS) bench/gpt2_bench.cpp \
		$(COMMON_OPS) $(COMMON_KERNELS) \
		-o $(BIN_DIR)/gpt2_bench

real_smoke: $(BIN_DIR)
	$(CXX) $(CXXFLAGS) tests/real_gpt2_smoke.cpp \
		$(COMMON_OPS) $(COMMON_KERNELS) \
		-o $(BIN_DIR)/real_gpt2_smoke

kv_cache_test: $(BIN_DIR)
	$(CXX) $(CXXFLAGS) tests/kv_cache_test.cpp \
		$(COMMON_OPS) $(COMMON_KERNELS) \
		-o $(BIN_DIR)/kv_cache_test

gpt2_test: $(BIN_DIR)
	$(CXX) $(CXXFLAGS) tests/gpt2_test.cpp \
		$(COMMON_OPS) $(COMMON_KERNELS) \
		-o $(BIN_DIR)/gpt2_test

generate_test: $(BIN_DIR)
	$(CXX) $(CXXFLAGS) tests/generate_test.cpp \
		$(GEN_OPS) $(COMMON_OPS) $(COMMON_KERNELS) \
		-o $(BIN_DIR)/generate_test

test: gpt2_test generate_test kv_cache_test

clean:
	rm -rf $(BIN_DIR)
