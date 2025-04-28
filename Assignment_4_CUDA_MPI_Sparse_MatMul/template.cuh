#include <utility>
#include <unordered_map>
#include <vector>

using namespace std;

struct pair_hash
{
    size_t operator()(const pair<int, int> &p) const
    {
        return static_cast<long long>(p.first) << 32 | p.second;
    }
};

struct mat_struct
{
    int exist;
    int height;
    int width;
    int num_blocks;
    vector<uint64_t> vec;
    unordered_map<pair<int, int>, int, pair_hash> mat_map;
};

mat_struct multiply_matrices(mat_struct &, mat_struct &, int);
