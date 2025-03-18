n_procs ?= 1
n_procs_2x := $(shell echo $$(($(n_procs) * 2)))

compile:
	mpic++ -std=c++17 -o check_$(n_procs) check.cpp template.cpp -O3 -march=native -mtune=native -flto -funroll-loops -ftree-vectorize

run:
	mpirun -np $(n_procs_2x) ./check_$(n_procs) V40_E160_C2 s1_10_$(n_procs).txt 10
	mpirun -np $(n_procs_2x) ./check_$(n_procs) V1000_E20000_C5 s2_10_$(n_procs).txt 10
	mpirun -np $(n_procs_2x) ./check_$(n_procs) V1000_E20000_C5 s2_100_$(n_procs).txt 100
	mpirun -np $(n_procs_2x) ./check_$(n_procs) V10000_E5000000_C50 s3_100_$(n_procs).txt 100
	mpirun -np $(n_procs_2x) ./check_$(n_procs) V10000_E5000000_C50 s3_1000_$(n_procs).txt 1000
	mpirun -np $(n_procs_2x) ./check_$(n_procs) V100000_E2cr_C50 s4_100_$(n_procs).txt 100
	mpirun -np $(n_procs_2x) ./check_$(n_procs) V100000_E2cr_C50 s4_1000_$(n_procs).txt 1000

	rm check_$(n_procs)

clean:
	rm -rf job_?.sh.*