CC=nvcc
CFLAGS=-g -O3 -std=c++11 -Xcompiler

#make the program
hw5: hw5.cu ppmFile.c
	nvcc hw5.cu -o hw5 -std=c++11 -Xcompiler -fopenmp

#cleanup function
clean:
	rm hw5

#-gencode arch=compute_20,code=sm_20
