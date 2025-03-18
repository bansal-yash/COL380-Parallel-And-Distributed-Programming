#include "template.hpp"
#include <algorithm>
#include <vector>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <mpi.h>

using namespace std;

void init_mpi(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
}

void end_mpi()
{
    MPI_Finalize();
}

inline pair<vector<int>, int> get_full_vertex_color_map_gather_broadcast(const map<int, int> &partial_vertex_color, int rank, int size)
{
    vector<int> partial_pairs;
    partial_pairs.reserve(partial_vertex_color.size() * 2);

    for (auto &p : partial_vertex_color)
    {
        partial_pairs.push_back(p.first);
        partial_pairs.push_back(p.second);
    }

    int local_pair_size = partial_pairs.size();
    vector<int> all_pair_sizes(size);

    MPI_Gather(&local_pair_size, 1, MPI_INT, all_pair_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    int total_pair_size = 0;
    vector<int> pair_displacements(size, 0);
    vector<int> global_pairs;

    if (rank == 0)
    {
        for (int i = 0; i < size; i++)
        {
            pair_displacements[i] = total_pair_size;
            total_pair_size += all_pair_sizes[i];
        }
        global_pairs.resize(total_pair_size);
    }

    MPI_Gatherv(partial_pairs.data(), local_pair_size, MPI_INT,
                global_pairs.data(), all_pair_sizes.data(), pair_displacements.data(), MPI_INT,
                0, MPI_COMM_WORLD);

    int num_vertex = total_pair_size / 2;
    MPI_Bcast(&num_vertex, 1, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> full_vertex_color(num_vertex);
    int num_colors = 0;

    if (rank == 0)
    {
        unordered_set<int> all_colors;

        for (int i = 0; i < total_pair_size; i += 2)
        {
            full_vertex_color[global_pairs[i]] = global_pairs[i + 1];
            all_colors.insert(global_pairs[i + 1]);
        }

        num_colors = all_colors.size();
        vector<int> num_to_col(num_colors);
        num_to_col.assign(all_colors.begin(), all_colors.end());
        sort(num_to_col.begin(), num_to_col.end());

        unordered_map<int, int> col_to_num;
        for (int i = 0; i < num_colors; i++)
        {
            col_to_num[num_to_col[i]] = i;
        }

        for (int &col : full_vertex_color)
        {
            col = col_to_num[col];
        }
    }

    MPI_Bcast(&num_colors, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(full_vertex_color.data(), num_vertex, MPI_INT, 0, MPI_COMM_WORLD);

    return make_pair(full_vertex_color, num_colors);
}

inline vector<vector<int>> get_deg_cent(const vector<int> &full_deg, int k, int num_colors)
{
    int num_vertex = full_deg.size() / num_colors;
    vector<vector<int>> top_k_nodes_per_color;
    vector<pair<int, int>> all_nodes(num_vertex);

    for (int j = 0; j < num_colors; j++)
    {
        for (int i = 0; i < num_vertex; i++)
        {
            all_nodes[i] = {full_deg[i * num_colors + j], i};
        }

        partial_sort(all_nodes.begin(), all_nodes.begin() + k, all_nodes.end(),
                     [](const pair<int, int> &a, const pair<int, int> &b)
                     {
                         return (a.first > b.first) || (a.first == b.first && a.second < b.second);
                     });

        vector<int> top_k_nodes;
        for (int i = 0; i < min(k, num_vertex); i++)
        {
            top_k_nodes.push_back(all_nodes[i].second);
        }

        top_k_nodes_per_color.push_back(move(top_k_nodes));
    }

    return top_k_nodes_per_color;
}

vector<vector<int>> degree_cen(vector<pair<int, int>> &partial_edge_list, map<int, int> &partial_vertex_color, int k)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Get full vertex colour map for all the processes
    auto [full_vertex_color, num_colors] = get_full_vertex_color_map_gather_broadcast(partial_vertex_color, rank, size);

    // Get the partial degrees for each process
    vector<int> part_deg(full_vertex_color.size() * num_colors, 0);
    for (const auto &[v1, v2] : partial_edge_list)
    {
        part_deg[(v1 * num_colors) + full_vertex_color[v2]]++;
        part_deg[(v2 * num_colors) + full_vertex_color[v1]]++;
    }

    // Sum up all the partial degrees and return the degree centrality
    if (rank == 0)
    {
        MPI_Reduce(MPI_IN_PLACE, part_deg.data(), part_deg.size(), MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        vector<vector<int>> ans = get_deg_cent(part_deg, k, num_colors);
        return ans;
    }
    else
    {
        MPI_Reduce(part_deg.data(), nullptr, part_deg.size(), MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        return {};
    }
}
