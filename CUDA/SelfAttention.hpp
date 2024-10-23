#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <memory>
#include <vector>
#include <stdexcept>

// CUDA error checking helper
#define CUDA_CHECK(call)                                                                     \
    do                                                                                       \
    {                                                                                        \
        cudaError_t err = call;                                                              \
        if (err != cudaSuccess)                                                              \
        {                                                                                    \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            throw std::runtime_error("CUDA error");                                          \
        }                                                                                    \
    } while (0)

struct KVCache
{
    std::unique_ptr<float[]> key;
    std::unique_ptr<float[]> value;
    size_t seq_len;
};

class SelfAttentionBase
{
protected:
    cublasHandle_t handle_;
    int emb_dim_;
    int d_k_;
    float scale_;

    // Device memory
    float *d_temp_memory_ = nullptr;
    size_t temp_memory_size_ = 0;

public:
    SelfAttentionBase(int emb_dim, int d_k)
        : emb_dim_(emb_dim), d_k_(d_k), scale_(1.0f / std::sqrt(d_k))
    {
        cublasCreate(&handle_);
    }

    virtual ~SelfAttentionBase()
    {
        if (d_temp_memory_)
            cudaFree(d_temp_memory_);
        cublasDestroy(handle_);
    }

    virtual void forward(const float *input, float *output, const bool *mask,
                         int batch_size, int seq_len) = 0;
};

class SelfAttentionL1 : public SelfAttentionBase
{
private:
    // Device memory for weights
    float *d_WQ_, *d_WK_, *d_WV_;

public:
    SelfAttentionL1(int emb_dim, int d_k) : SelfAttentionBase(emb_dim, d_k)
    {
        // Allocate weight matrices
        CUDA_CHECK(cudaMalloc(&d_WQ_, emb_dim * d_k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_WK_, emb_dim * d_k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_WV_, emb_dim * emb_dim * sizeof(float)));
    }

    ~SelfAttentionL1() override
    {
        cudaFree(d_WQ_);
        cudaFree(d_WK_);
        cudaFree(d_WV_);
    }

    void forward(const float *input, float *output, const bool *mask,
                 int batch_size, int seq_len) override;
};

class SelfAttentionL2 : public SelfAttentionBase
{
private:
    float *d_WQ_, *d_WK_, *d_WV_;
    bool use_bias_;

public:
    SelfAttentionL2(int emb_dim, int d_k, bool use_bias = false)
        : SelfAttentionBase(emb_dim, d_k), use_bias_(use_bias)
    {
        CUDA_CHECK(cudaMalloc(&d_WQ_, emb_dim * d_k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_WK_, emb_dim * d_k * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_WV_, emb_dim * emb_dim * sizeof(float)));
    }

    ~SelfAttentionL2() override
    {
        cudaFree(d_WQ_);
        cudaFree(d_WK_);
        cudaFree(d_WV_);
    }

    void forward(const float *input, float *output, const bool *mask,
                 int batch_size, int seq_len) override;
};

class SelfAttentionL3 : public SelfAttentionBase
{
private:
    float *d_WQK_, *d_WV_;

public:
    SelfAttentionL3(int emb_dim, int d_k) : SelfAttentionBase(emb_dim, d_k)
    {
        CUDA_CHECK(cudaMalloc(&d_WQK_, emb_dim * (2 * d_k) * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_WV_, emb_dim * emb_dim * sizeof(float)));
    }

    ~SelfAttentionL3() override
    {
        cudaFree(d_WQK_);
        cudaFree(d_WV_);
    }

    void forward(const float *input, float *output, const bool *mask,
                 int batch_size, int seq_len) override;
};

// CUDA kernel declarations
namespace cuda_kernels
{
    __global__ void apply_attention_mask(float *scores, const bool *mask,
                                         int batch_size, int seq_len);

    __global__ void softmax_kernel(float *scores, int batch_size, int seq_len);

    __global__ void fused_qkv_projection(const float *input, const float *weights,
                                         float *q, float *k, float *v,
                                         int batch_size, int seq_len, int emb_dim, int d_k);
}