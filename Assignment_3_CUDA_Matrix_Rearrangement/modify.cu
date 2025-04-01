#include <vector>
#include <utility>
#include <cstring>
#include <cuda_runtime.h>
#include "modify.cuh"

using namespace std;

__global__ void fill_freqs(int *cuda_mat, int *cuda_freq_array, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
    {
        int element = cuda_mat[i];
        atomicAdd(&cuda_freq_array[element], 1);
    }
}

__global__ void fill_matrix(int *cuda_ans, int *cuda_prefix_sum, int *cuda_freq_array, int size, int ran)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ran)
    {
        int freq = cuda_freq_array[i];
        if (freq > 0)
        {
            int start_idx = cuda_prefix_sum[i];
            int end_idx = min(start_idx + freq, size);
            for (int flat_idx = start_idx; flat_idx < end_idx; flat_idx++)
            {
                cuda_ans[flat_idx] = i;
            }
        }
    }
}

void Process_mat(const vector<vector<int>> &matrix, vector<vector<int>> &ans, int ran)
{
    int m = matrix.size();
    int n = matrix[0].size();
    int size = m * n;
    int block_size = 1024;

    cudaStream_t stream1;
    cudaStreamCreate(&stream1);

    int *cuda_mat, *cuda_freq_array, *cuda_prefix_sum;
    cudaMalloc(&cuda_mat, size * sizeof(int));
    cudaMalloc(&cuda_freq_array, ran * sizeof(int));
    cudaMalloc(&cuda_prefix_sum, ran * sizeof(int));

    cudaMemsetAsync(cuda_freq_array, 0, ran * sizeof(int), stream1);
    for (int i = 0; i < m; i++)
    {
        cudaMemcpyAsync(&cuda_mat[i * n], matrix[i].data(), n * sizeof(int), cudaMemcpyHostToDevice, stream1);
    }

    int num_blocks = (size + block_size - 1) / block_size;
    fill_freqs<<<num_blocks, block_size, 0, stream1>>>(cuda_mat, cuda_freq_array, size);

    cudaStreamSynchronize(stream1);

    int *h_freq_array = new int[ran];
    int *h_prefix_sum = new int[ran];

    cudaMemcpy(h_freq_array, cuda_freq_array, ran * sizeof(int), cudaMemcpyDeviceToHost);

    h_prefix_sum[0] = 0;
    for (int i = 1; i < ran; i++)
    {
        h_prefix_sum[i] = h_prefix_sum[i - 1] + h_freq_array[i - 1];
    }

    cudaMemcpy(cuda_prefix_sum, h_prefix_sum, ran * sizeof(int), cudaMemcpyHostToDevice);

    delete[] h_freq_array;
    delete[] h_prefix_sum;

    int num_blocks_distribute = (ran + block_size - 1) / block_size;
    fill_matrix<<<num_blocks_distribute, block_size, 0, stream1>>>(cuda_mat, cuda_prefix_sum, cuda_freq_array, size, ran);

    cudaStreamSynchronize(stream1);

    for (int i = 0; i < m; i++)
    {
        cudaMemcpyAsync(ans[i].data(), &cuda_mat[i * n], n * sizeof(int), cudaMemcpyDeviceToHost, stream1);
    }
    cudaStreamSynchronize(stream1);

    cudaFree(cuda_mat);
    cudaFree(cuda_freq_array);
    cudaFree(cuda_prefix_sum);
    cudaStreamDestroy(stream1);
}

vector<vector<vector<int>>> modify(vector<vector<vector<int>>> &matrices, vector<int> &range)
{
    int num_matrix = matrices.size();
    vector<vector<vector<int>>> res;
    res.reserve(num_matrix);

    for (int a = 0; a < num_matrix; a++)
    {
        const vector<vector<int>> &matrix = matrices[a];
        int m = matrix.size();
        int n = matrix[0].size();
        vector<vector<int>> ans(m, vector<int>(n, 0));
        int ran = range[a] + 1;

        Process_mat(matrix, ans, ran);

        res.emplace_back(std::move(ans));
    }

    return res;
}
