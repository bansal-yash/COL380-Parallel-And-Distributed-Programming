# COL380 – Parallel and Distributed Programming  
**Assignments from the Parallel and Distributed Programming course (COL380) at IIT Delhi, taught by Prof. Subodh Kumar**

This repository contains assignments that explore concepts in parallelism and distributed computing using C++, OpenMP, MPI, and CUDA.

---

## 📁 Assignments

### [Assignment 0: Analysis of Sequential Matrix Multiplication](./Assignment_0_Sequential_Matrix_Mult_Analysis)
- Explored the performance of different loop permutations in matrix multiplication using C++.
- Performance measured using `perf` and GNU Profiler (gprof).

---

### [Assignment 1: Sparse Matrix Multiplication using OpenMP](./Assignment_1_Sparse_MatMul_OpenMP)
- Generation and multiplication of block sparse matrices using OpenMP in C++.
- Computed row statistics for analyzing sparse matrix behavior.

---

### [Assignment 2: Graph Centrality Computation using MPI](./Assignment_2_MPI_Parallel_Graph_Centrality)
- Computed the most influential nodes per color in a colored graph using MPI.
- Implemented parallel processing of graph data and analyzed different MPI communication strategies.

---

### [Assignment 3: Matrix Rearrangement using CUDA](./Assignment_3_CUDA_Matrix_Rearrangement)
- Performed sorted rearrangement of matrices in parallel using CUDA on the GPU.
- Implemented in C++ with CUDA kernels for parallel acceleration.

---

### [Assignment 4: Sparse Matrix Multiplication on GPU with CUDA and MPI](./Assignment_4_CUDA_MPI_Sparse_MatMul)
- Optimized sparse matrix multiplication using a hybrid of CUDA, MPI, and OpenMP.
- Analyzed the scalability of the parallel implementation for varying input sizes.

---

Each assignment focuses on solving computational problems using different models of parallel and distributed computing. The implementations demonstrate practical knowledge in optimizing and analyzing performance on multicore CPUs and GPUs.
