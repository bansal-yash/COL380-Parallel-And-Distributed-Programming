#include <iostream>
#include <vector>
#include <map>
#include <utility>
#include <cstdlib>
#include <ctime>
#include <random>
#include <algorithm>
#include <omp.h>
#include "check.h"

using namespace std;

vector<pair<int, int>> generate_rand_pairs(int m, int b)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dis(0, m - 1);

    vector<pair<int, int>> ans;
    map<pair<int, int>, int> ans_map;

    while (ans.size() != b)
    {
        int x = dis(gen);
        int y = dis(gen);
        pair<int, int> p = {x, y};

        if (ans_map.find(p) == ans_map.end())
        {
            ans.push_back(p);
            ans_map[p] = 1;
        }
    }
    return ans;
}

vector<vector<int>> gen_block(int m)
{
    vector<vector<int>> block(m, vector<int>(m, 0));

    random_device rd;
    mt19937 gen(rd());
    std::uniform_int_distribution<> dis(-2147483640 , 2147483640);
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < m; j++)
        {
            block[i][j] = dis(gen);
        }
    }

    block[0][0] = 1;
    return block;
}

map<pair<int, int>, vector<vector<int>>> generate_matrix(int n, int m, int b)
{
    map<pair<int, int>, vector<vector<int>>> matrix_map;
    int num_total_blocks = n / m;

    vector<pair<int, int>> non_zero_block = generate_rand_pairs(num_total_blocks, b);

    int num_cores = omp_get_num_procs();
    omp_set_num_threads(num_cores);

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 0; i < b; i++)
            {
                #pragma omp task if (black_box())
                {
                    // int thread_id = omp_get_thread_num();
                    // cout << thread_id << endl;
                    vector<vector<int>> block = gen_block(m);

                    #pragma omp critical
                    {
                        matrix_map[non_zero_block[i]] = block;
                    }
                }
            }
            #pragma omp taskwait
        }
    }
    return matrix_map;
}

void remove_5_multiples(map<pair<int, int>, vector<vector<int>>> &blocks, int m)
{
    vector<pair<int, int>> all_block_locations = {};
    for (auto &[a, _] : blocks)
    {
        all_block_locations.push_back(a);
    }
    int b = all_block_locations.size();
    vector<pair<int, int>> to_remove;

    int num_cores = omp_get_num_procs();
    omp_set_num_threads(num_cores);

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 0; i < b; i++)
            {
                #pragma omp task if (black_box())
                {
                    pair<int, int> p = all_block_locations[i];
                    // int thread_id = omp_get_thread_num();
                    // cout << thread_id << endl;

                    bool is_zero = true;
                    for (int i = 0; i < m; i++)
                    {
                        for (int j = 0; j < m; j++)
                        {
                            if (blocks[p][i][j] % 5 == 0)
                            {
                                blocks[p][i][j] = 0;
                            }
                            else
                            {
                                is_zero = false;
                            }
                        }
                    }

                    if (is_zero)
                    {
                        #pragma omp critical
                        {
                            to_remove.push_back(p);
                        }
                    }
                }
            }
            #pragma omp taskwait
        }
    }

    for (pair<int, int> &p : to_remove)
    {
        blocks.erase(p);
    }
}

vector<vector<int>> block_mult(vector<vector<int>> &block1, vector<vector<int>> &block2, int m, vector<int> &row_stats_blocks)
{
    vector<vector<int>> product(m, vector<int>(m, 0));

    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            for (int k = 0; k < m; ++k)
            {
                int prod = block1[i][k] * block2[k][j];
                if (prod != 0)
                {
                    product[i][j] += prod;
                    row_stats_blocks[i]++;
                }
            }
        }
    }
    return product;
}

map<pair<int, int>, vector<vector<int>>> multiply_matrices(map<pair<int, int>, vector<vector<int>>> &blocks_1, map<pair<int, int>, vector<vector<int>>> &blocks_2, int n, int m, int k, vector<float> &row_stats)
{
    map<pair<int, int>, vector<vector<int>>> multiplied = {};
    int num_total_blocks = n / m;

    vector<float> row_statistics(n, 0.0);
    vector<int> P(n, 0);
    vector<int> B(n, 0);

    int num_cores = omp_get_num_procs();
    omp_set_num_threads(num_cores);

    #pragma omp parallel
    {
        #pragma omp single
        {
            vector<pair<int, int>> all_block_1_locations = {};
            for (auto &[a, _] : blocks_1)
            {
                all_block_1_locations.push_back(a);
            }

            vector<pair<int, int>> all_block_2_locations = {};
            for (auto &[a, _] : blocks_2)
            {
                all_block_2_locations.push_back(a);
            }

            for (int i = 0; i < all_block_1_locations.size(); i++)
            {
                for (int j = 0; j < all_block_2_locations.size(); j++)
                {
                    if (all_block_1_locations[i].second == all_block_2_locations[j].first)
                    {
                        #pragma omp task shared(all_block_1_locations, all_block_2_locations, P, multiplied) if (black_box())
                        {
                            pair<int, int> p1 = all_block_1_locations[i];
                            pair<int, int> p2 = all_block_2_locations[j];
                            pair<int, int> p3 = {p1.first, p2.second};

                            // int thread_id = omp_get_thread_num();
                            // cout << thread_id << endl;

                            vector<vector<int>> &block1 = blocks_1[p1];
                            vector<vector<int>> &block2 = blocks_2[p2];
                            vector<int> row_stats_blocks(m, 0);

                            vector<vector<int>> block_multiplied = block_mult(block1, block2, m, row_stats_blocks);

                            bool is_zero = true;
                            for (int i1 = 0; i1 < m; i1++)
                            {
                                for (int i2 = 0; i2 < m; i2++)
                                {
                                    if (block_multiplied[i1][i2] != 0)
                                    {
                                        is_zero = false;
                                        break;
                                    }
                                }
                            }

                            if (!is_zero)
                            {
                                bool done = false;

                                #pragma omp critical
                                {
                                    if (multiplied.find(p3) == multiplied.end())
                                    {
                                        multiplied[p3] = block_multiplied;
                                        done = true;
                                    }
                                }
                                
                                if (! done)
                                {
                                    for (int a = 0; a < m; a++)
                                    {
                                        for (int b = 0; b < m; b++)
                                        {
                                            #pragma omp atomic
                                            multiplied[p3][a][b] += block_multiplied[a][b];
                                        }
                                    }
                                }

                                if (k == 2)
                                {
                                    for (int a = 0; a < m; a++) {
                                        int row_idx = p1.first * m + a;
                                        #pragma omp atomic
                                        P[row_idx] += row_stats_blocks[a];
                                    }                                    
                                }
                            }
                        }
                    }
                }
            }
            #pragma omp taskwait
        }
    }

    if (k == 2)
    {
        for (auto &[a, _] : multiplied)
        {
            for (int i = 0; i < m; i++)
            {
                B[a.first * m + i] += m;
            }
        }

        for (int i = 0; i < n; i++)
        {
            if (B[i] != 0)
            {
                row_statistics[i] = static_cast<float>(P[i]) / B[i];
            }
        }
        row_stats = row_statistics;
    }

    return multiplied;
}

vector<float> matmul(map<pair<int, int>, vector<vector<int>>> &blocks, int n, int m, int k)
{
    remove_5_multiples(blocks, m);

    if (k == 2)
    {
        vector<float> row_stats = {};
        blocks = multiply_matrices(blocks, blocks, n, m, k, row_stats);
        return row_stats;
    }
    else
    {
        int exp = k;
        map<pair<int, int>, vector<vector<int>>> result = blocks;
        map<pair<int, int>, vector<vector<int>>> pre = blocks;
        exp -= 1;
        vector<float> row_stats = {};

        while (exp > 0)
        {
            if (exp % 2 == 1)
            {
                result = multiply_matrices(result, pre, n, m, k, row_stats);
            }
            pre = multiply_matrices(pre, pre, n, m, k, row_stats);
            exp /= 2;
        }
        blocks = result;

        return {};
    }
}
