run: compile
	mpirun -np 2 ./a4 medium_test

compile: template.cpp
	nvcc -c -O3 cuda_mult.cu -o cuda_mult.o
	mpic++ -fopenmp -std=c++17 -O3 template.cpp cuda_mult.o -lcudart -o a4

clean:
	rm -rf job.sh.e* job.sh.o* a4 *.o