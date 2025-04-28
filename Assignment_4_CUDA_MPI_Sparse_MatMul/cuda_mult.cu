#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include "template.cuh"

__global__ void multiply_blocks_kernel(u_int64_t *integ_vec, int block_size)
{
    int block_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    int offset = 3 * block_idx * block_size * block_size;

    u_int64_t *mat1 = &integ_vec[offset];
    u_int64_t *mat2 = &integ_vec[offset + block_size * block_size];

    int i = thread_idx / block_size;
    int j = thread_idx % block_size;

    u_int64_t sum = 0;
    for (int k = 0; k < block_size; k++)
    {
        sum += mat1[i * block_size + k] * mat2[k * block_size + j];
    }

    integ_vec[offset + 2 * block_size * block_size + thread_idx] = sum;
}

mat_struct multiply_matrices(mat_struct &mat1, mat_struct &mat2, int block_size)
{
    mat_struct multiplied;

    multiplied.exist = 1;
    multiplied.height = mat1.height;
    multiplied.width = mat2.width;

    vector<int> all_mult_pairs;
    int num_ans_block = 0;

    for (auto it1 = mat1.mat_map.begin(); it1 != mat1.mat_map.end(); ++it1)
    {
        auto &p1 = it1->first;
        int v1 = it1->second;

        for (auto it2 = mat2.mat_map.begin(); it2 != mat2.mat_map.end(); ++it2)
        {
            auto &p2 = it2->first;
            int v2 = it2->second;

            if (p1.second == p2.first)
            {
                all_mult_pairs.push_back(p1.first);
                all_mult_pairs.push_back(p2.second);
                all_mult_pairs.push_back(v1);
                all_mult_pairs.push_back(v2);

                if (multiplied.mat_map.find({p1.first, p2.second}) == multiplied.mat_map.end())
                {
                    multiplied.mat_map[{p1.first, p2.second}] = num_ans_block * block_size * block_size;
                    num_ans_block++;
                }
            }
        }
    }

    multiplied.vec.resize(num_ans_block * block_size * block_size, 0);
    multiplied.num_blocks = num_ans_block;

    int total_pairs = all_mult_pairs.size() / 4;

    size_t free_mem = 0, total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
    size_t element_size = sizeof(u_int64_t);
    size_t num_elements = free_mem / element_size;

    int batch_size = num_elements / (3 * block_size * block_size);
    batch_size = batch_size * (0.8);

    int glob_batch_start = 0;

    for (int batch_start = 0; batch_start < total_pairs; batch_start += batch_size)
    {
        int current_batch_size = min(batch_size, total_pairs - batch_start);

        vector<u_int64_t> integrated_vec;
        integrated_vec.reserve(current_batch_size * 3 * block_size * block_size);

        for (int i = 0; i < current_batch_size; i++)
        {
            auto b1_start = mat1.vec.begin() + (all_mult_pairs[4 * (glob_batch_start + i) + 2]);
            auto b1_end = b1_start + block_size * block_size;
            auto b2_start = mat2.vec.begin() + (all_mult_pairs[4 * (glob_batch_start + i) + 3]);
            auto b2_end = b2_start + block_size * block_size;

            integrated_vec.insert(integrated_vec.end(), make_move_iterator(b1_start), make_move_iterator(b1_end));
            integrated_vec.insert(integrated_vec.end(), make_move_iterator(b2_start), make_move_iterator(b2_end));
            integrated_vec.resize(integrated_vec.size() + block_size * block_size, 0);
        }

        u_int64_t *d_integ_vec;
        cudaMalloc(&d_integ_vec, integrated_vec.size() * sizeof(u_int64_t));
        cudaMemcpy(d_integ_vec, integrated_vec.data(), integrated_vec.size() * sizeof(u_int64_t), cudaMemcpyHostToDevice);

        int threads_per_block = block_size * block_size;
        int blocks_per_grid = current_batch_size;

        multiply_blocks_kernel<<<blocks_per_grid, threads_per_block>>>(d_integ_vec, block_size);

        cudaDeviceSynchronize();

        cudaMemcpy(integrated_vec.data(), d_integ_vec, integrated_vec.size() * sizeof(u_int64_t), cudaMemcpyDeviceToHost);
        cudaFree(d_integ_vec);

        for (int i = 0; i < current_batch_size; i++)
        {
            int idx = (3 * i + 2) * block_size * block_size;
            int row = all_mult_pairs[4 * glob_batch_start + 4 * i];
            int col = all_mult_pairs[4 * glob_batch_start + 4 * i + 1];

            int ans_loc = multiplied.mat_map[{row, col}];

            for (int j = 0; j < block_size * block_size; j++)
            {
                multiplied.vec[ans_loc + j] += integrated_vec[idx + j];
            }
        }

        glob_batch_start += current_batch_size;
    }

    return multiplied;
}