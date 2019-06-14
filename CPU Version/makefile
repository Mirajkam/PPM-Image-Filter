CC=gcc
CFLAGS=-g -Wall -fopenmp -O3 -std=c99
LDFLAGS=

#make the program
hw4: hw4.c ppmFile.c
	$(CC) $(CFLAGS) -o $@ $?

#cleanup function
clean:
	rm hw4

#-gencode arch=compute_20,code=sm_20
