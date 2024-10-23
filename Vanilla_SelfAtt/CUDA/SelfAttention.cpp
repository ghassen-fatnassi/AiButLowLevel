#include "SelfAttention.hpp"
#include <cmath>

// CUDA kernel implementations
__global__ void cuda_kernels::apply_attention_mask(
    float *scores,
    const bool *mask,
    int batch_size,
    int seq_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = idx / (seq_len * seq_len);
    int seq_idx = (idx / seq_len) % seq_len;
    int key_idx = idx % seq_len;

    if (idx < batch_size * seq_len * seq_len)
    {
        if (!mask[batch_idx * seq_len + key_idx])
        {
            scores[idx] = -INFINITY;
        }
    }
}

__global__ void cuda_kernels::softmax_kernel(
    float *scores,
    int batch_size,
    int seq_len)
{
    __shared__ float shared_max[32];
    __shared__ float shared_sum[32];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int row_idx = bid / seq_len;
    int col_start = tid;

    // Find max value in row
    float max_val = -INFINITY;
    for (int i = col_start; i < seq_len; i += blockDim.x)
    {
        max_val = max(max_val, scores[row_idx * seq_len + i]);
    }

    shared_max[tid] = max_val;
    __syncthreads();

    // Reduce to find row max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride]);
        }
        __syncthreads();
    }

    float row_max = shared_max[0];

    // Compute exp and sum
    float sum = 0.0f;
    for (int i = col_start; i < seq_len; i += blockDim.x)
    {
        float val = exp(scores[row_idx * seq_len + i] - row_max);
        scores[row_idx * seq_len + i] = val;
        sum += val;
    }

    shared_sum[tid] = sum;
    __syncthreads();

    // Reduce to find total sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared_sum[tid] += shared_sum[tid + stride];
        }
        __syncthreads();
    }

    float row_sum = shared_sum[0];

    // Normalize
    for (int i = col_start; i < seq_len; i += blockDim.x)
    {
        scores[row_idx * seq_len + i] /= row_sum;
    }
}

void SelfAttentionL1::forward(
    const float *input,
    float *output,
    const bool *mask,
    int batch_size,
    int seq_len)
{
    // Allocate temporary device memory if needed
    size_t needed_memory = batch_size * seq_len * (3 * d_k_ + emb_dim_) * sizeof(float);
    if (needed_memory > temp_memory_size_)
    {
        if (d_temp_memory_)
            cudaFree(d_temp_memory_);
        CUDA_CHECK(cudaMalloc(&d_temp_memory_, needed_memory));
        temp_memory_size_ = needed_memory;
    }

    float *d_Q = d_temp_memory_;
    float *d_K = d_Q + batch_size * seq_len * d_k_;
    float *d_V = d_K + batch_size * seq_len * d_k_;
    float *d_scores = d_V + batch_size * seq_len * emb_dim_;

    // Q = input * WQ, K = input * WK, V = input * WV
    const float alpha = 1.0f;
    const float beta = 0.0f;

    for (int b = 0; b < batch_size; ++b)
    {
        cublasSgemm(handle_,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    d_k_, seq_len, emb_dim_,
                    &alpha,
                    d_WQ_, d_k_,
                    input + b * seq_len * emb_dim_, emb_dim_,
                    &beta,
                    d_Q + b * seq_len * d_k_, d_k_);

        cublasSgemm(handle_,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    d_k_, seq_len, emb_dim_,
                    &alpha,
                    d_WK_, d_k_,
                    input + b * seq_len * emb_dim_, emb_dim_,
                    &beta,
                    d_K + b * seq_len * d_k_, d_k_);

        cublasSgemm(handle_,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    emb_dim_, seq_len, emb_dim_,
                    &alpha,
                    d_WV_, emb_dim_,
                    input + b * seq_len * emb_dim_, emb_dim_,
                    &beta,
                    d_V + b * seq_len * emb_dim_, emb_dim_);
    }

    // Compute attention scores
    dim3 block(256);
    dim3 grid((batch_size * seq_len * seq_len + block.x - 1) / block.x);

    cuda_kernels::apply_attention_mask<<<grid, block>>>(
        d_scores, mask, batch_size, seq_len);

    cuda_kernels::softmax_kernel<<<batch_size * seq_len, 256>>>(
        d_scores, batch_size, seq_len);

    // Final matrix multiplication
    for (int b = 0; b < batch_size; ++b)
    {
        cublasSgemm(handle_,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    emb_dim_, seq_len, seq_len,
                    &alpha,
                    d_V + b * seq_len * emb_dim_, emb_dim_,
                    d_scores + b * seq_len * seq_len, seq_len,
                    &beta,
                    output + b * seq_len * emb_dim_, emb_dim_);
    }
}

// i still need to implemente L2 and L3