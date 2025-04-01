check: main.cpp modify.cu
	nvcc main.cpp modify.cu -arch=sm_35 -Xcompiler -fopenmp -o check
	
run: check
	./check

.PHONY: run