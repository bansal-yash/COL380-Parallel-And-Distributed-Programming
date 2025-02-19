Yash Bansal (2022CS51133)

Optimizations performed
1. Generate matrix function generates all the matrices in parallel,
2. Each block multiplication is done by a seperate task. Synchronization first performed using critical section.
3. Used atomic operations whenever a data was to be added to the block of the new matrix
4. Calculated all the blocks which are required to be multiplied sequentially before initialising pragma omp. This reduces the time by approx 40%
5. Row statistics updated using atomic operations for each block multiplication
6. Binary exponentiation is used for fast exponentiation of the matrices

Final results:- 

->  For n = 100000 and m = 2, the code scales well upto 16 cores, but the performance remains almost constant after that.
    This is because after 16 cores, the iteration in the map, which is sequential, overtakes the time taken for block multiplications.
    This can be seen from the following perf report which is taken for 32 cores, and b = 4096

    --4.93%--std::map<std::pair<int, int>, std::vector<std::vector<int, std::allocator<int> >, std::allocator<std::vector<int, std::allocator<int> > > >, std::less<std::pair<int, int> >, std::allocator<std::pair<std::pair<int, int> const, std::vector<std::vector<int, std::allocator<int> >, std::allocator<std::vector<int, std::allocator<int> > > > > > >::operator[]
    The above was the % time in map iteration in matmul function

    --2.95%--multiply_matrices
    --2.45%--block_mult

->  For n = 100000 and m = 50, the code scales very well until 40 cores. In this case, the block multiplication times are higher as m is larger.
    Thus, the parallel section of the code is the major bottleneck(and thus the number of cores), rather than the sequential section of map iteration.
    The following perf data is taken for 32 cores and b = 4096

    --3.06%--std::vector<std::vector<int, std::allocator<int> >, std::allocator<std::vector<int, std::allocator<int> > > >::operator[]
    --44.47%--block_mult

