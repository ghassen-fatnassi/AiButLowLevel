
#include "SelfAttention.hpp"
#include <random>

int main()
{
    try
    {
        // parameters
        const int batch_size = 2;
        const int seq_len = 8;
        const int emb_dim = 64;
        const int d_k = 32;

        SelfAttentionL1 attention(emb_dim, d_k);

        // host memory
        std::vector<float> input(batch_size * seq_len * emb_dim);
        std::vector<float> output(batch_size * seq_len * emb_dim);
        std::vector<bool> mask(batch_size * seq_len, true);

        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        for (auto &val : input)
        {
            val = dist(gen);
        }

        // device memory
        float *d_input, *d_output;
        bool *d_mask;
        CUDA_CHECK(cudaMalloc(&d_input, input.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, output.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_mask, mask.size() * sizeof(bool)));

        // data -> device
        CUDA_CHECK(cudaMemcpy(d_input, input.data(),
                              input.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mask, mask.data(),
                              mask.size() * sizeof(bool), cudaMemcpyHostToDevice));

        // Run forward pass
        attention.forward(d_input, d_output, d_mask, batch_size, seq_len);

        // Copy results back to host
        CUDA_CHECK(cudaMemcpy(output.data(), d_output,
                              output.size() * sizeof(float), cudaMemcpyDeviceToHost));

        // Clean up
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_mask);
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}