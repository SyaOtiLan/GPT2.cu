#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "../include/mlp.h"
#include "reference/cpu_reference.h"

static void check_cuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

int main() {
    const int tokens = 2;
    const int hidden = 4;
    const int hidden4 = 4 * hidden; // 16

    const int x_size = tokens * hidden;
    const int w1_size = hidden * hidden4;
    const int w2_size = hidden4 * hidden;
    const int b1_size = hidden4;
    const int b2_size = hidden;
    const int fc1_size = tokens * hidden4;
    const int fc2_size = tokens * hidden;
    const int out_size = tokens * hidden;

    float h_x[x_size] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        0.5f, -1.0f, 2.0f, -0.5f
    };

    float h_w1[w1_size];
    float h_w2[w2_size];
    float h_b1[b1_size];
    float h_b2[b2_size];

    // 给权重一些小值，避免数过大
    for (int i = 0; i < w1_size; ++i) {
        h_w1[i] = 0.01f * (float)((i % 7) - 3);
    }
    for (int i = 0; i < w2_size; ++i) {
        h_w2[i] = 0.01f * (float)((i % 5) - 2);
    }
    for (int i = 0; i < b1_size; ++i) {
        h_b1[i] = 0.01f * (float)((i % 5) - 2);
    }
    for (int i = 0; i < b2_size; ++i) {
        h_b2[i] = 0.01f * (float)((i % 3) - 1);
    }

    float h_out_ref[out_size];

    mlp_cpu_ref(h_x, h_x, h_out_ref, h_w1, h_b1, h_w2, h_b2, tokens, hidden);

    // device buffers
    float *d_x = nullptr, *d_w1 = nullptr, *d_w2 = nullptr, *d_b1 = nullptr, *d_b2 = nullptr;
    float *d_fc1 = nullptr, *d_gelu = nullptr, *d_fc2 = nullptr, *d_out = nullptr;

    check_cuda(cudaMalloc(&d_x, x_size * sizeof(float)), "cudaMalloc d_x");
    check_cuda(cudaMalloc(&d_w1, w1_size * sizeof(float)), "cudaMalloc d_w1");
    check_cuda(cudaMalloc(&d_w2, w2_size * sizeof(float)), "cudaMalloc d_w2");
    check_cuda(cudaMalloc(&d_b1, b1_size * sizeof(float)), "cudaMalloc d_b1");
    check_cuda(cudaMalloc(&d_b2, b2_size * sizeof(float)), "cudaMalloc d_b2");
    check_cuda(cudaMalloc(&d_fc1, fc1_size * sizeof(float)), "cudaMalloc d_fc1");
    check_cuda(cudaMalloc(&d_gelu, fc1_size * sizeof(float)), "cudaMalloc d_gelu");
    check_cuda(cudaMalloc(&d_fc2, fc2_size * sizeof(float)), "cudaMalloc d_fc2");
    check_cuda(cudaMalloc(&d_out, out_size * sizeof(float)), "cudaMalloc d_out");

    check_cuda(cudaMemcpy(d_x, h_x, x_size * sizeof(float), cudaMemcpyHostToDevice), "memcpy x");
    check_cuda(cudaMemcpy(d_w1, h_w1, w1_size * sizeof(float), cudaMemcpyHostToDevice), "memcpy w1");
    check_cuda(cudaMemcpy(d_w2, h_w2, w2_size * sizeof(float), cudaMemcpyHostToDevice), "memcpy w2");
    check_cuda(cudaMemcpy(d_b1, h_b1, b1_size * sizeof(float), cudaMemcpyHostToDevice), "memcpy b1");
    check_cuda(cudaMemcpy(d_b2, h_b2, b2_size * sizeof(float), cudaMemcpyHostToDevice), "memcpy b2");

    // run mlp
    mlp_forward(
        d_x,
        d_x,
        d_out,
        d_w1,
        d_b1,
        d_w2,
        d_b2,
        tokens,
        hidden,
        d_fc1,
        d_gelu,
        d_fc2
    );

    check_cuda(cudaDeviceSynchronize(), "sync after mlp_forward");

    float h_out_gpu[out_size];
    check_cuda(cudaMemcpy(h_out_gpu, d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost), "memcpy out");

    // compare
    float max_err = max_abs_diff_cpu_ref(h_out_gpu, h_out_ref, out_size);
    const float tol = 1e-4f;

    printf("GPU output:\n");
    for (int i = 0; i < tokens; ++i) {
        for (int j = 0; j < hidden; ++j) {
            printf("%8.5f ", h_out_gpu[i * hidden + j]);
        }
        printf("\n");
    }

    printf("\nCPU ref:\n");
    for (int i = 0; i < tokens; ++i) {
        for (int j = 0; j < hidden; ++j) {
            printf("%8.5f ", h_out_ref[i * hidden + j]);
        }
        printf("\n");
    }

    printf("\nmax_err = %g\n", max_err);

    if (!all_finite_cpu_ref(h_out_gpu, out_size)) {
        fprintf(stderr, "MLP output contains NaN/Inf\n");
        return 1;
    }
    if (max_err > tol) {
        fprintf(stderr, "MLP output mismatch: max_err=%g tol=%g\n", max_err, tol);
        return 1;
    }

    cudaFree(d_x);
    cudaFree(d_w1);
    cudaFree(d_w2);
    cudaFree(d_b1);
    cudaFree(d_b2);
    cudaFree(d_fc1);
    cudaFree(d_gelu);
    cudaFree(d_fc2);
    cudaFree(d_out);

    return 0;
}
