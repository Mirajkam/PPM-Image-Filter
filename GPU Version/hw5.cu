#include "ppmFile.h"
#include "ppmFile.c"
#include <cuda_runtime.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

Image *mean(Image *imgin, int r);

__global__ void row(int n, int r, int width, int height, unsigned char *in, unsigned char *out) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        int y = i / width;
        int x = i % width;
        int left = (x - r) * (r < x);
        int right = (width - 1) * (x + r >= width - 1) + (x + r) * (x + r < width - 1);

        int count = (right - left) + 1;

        int sumr = 0;
        int sumg = 0;
        int sumb = 0;

        for (int cx = left; cx <= right; cx++) {
            int ci = y * width + cx;
            sumr += in[ci*3];
            sumg += in[ci*3+1];
            sumb += in[ci*3+2];
        }
        out[(y * width + x)*3] = sumr / count;
        out[(y * width + x)*3 + 1] = sumg / count;
        out[(y * width + x)*3 + 2] = sumb / count;
    }
  }


__global__ void col(int n, int r, int width, int height, unsigned char *in, unsigned char *out) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        int y = i / width;
        int x = i % width;
        int top = (y - r) * (r < y);
        int bottom = (height - 1) * (y + r >= height - 1) + (y + r) * (y + r < height - 1);

        int count = (bottom - top) + 1;

        int sumr = 0;
        int sumg = 0;
        int sumb = 0;

        for (int cy = top; cy <= bottom; cy++) {
            int ci = cy * width + x;
            sumr += in[ci*3];
            sumg += in[ci*3+1];
            sumb += in[ci*3+2];
        }

        out[(y * width + x)*3] = sumr / count;
        out[(y * width + x)*3 + 1] = sumg / count;
        out[(y * width + x)*3 + 2] = sumb / count;
    }
}

/*int main(int argc, char *argv[]) {
  int r = atoi(argv[1]);
  Image *unfilterImage = ImageRead(argv[2]);
  Image *newIm;
  int imageH = ImageHeightHost(unfilterImage);
  int imageW = ImageWidthHost(unfilterImage);
  newIm = filter(imageW, imageH, r, unfilterImage);
  ImageWrite(newIm, argv[3]);
}*/

int main(int argc, char *argv[]) {
   
    

    const char *input = argv[2];
    const char *output = argv[3];
    int r = atoi(argv[1]);

    
    Image *in = ImageRead(input);

    Image *out = mean(in, r);
    ImageWrite(out, output);

    return 0;
}

Image *mean(Image *imgin, int r) {
    int width = imgin->width;
    int height = imgin->height;
    int size = width * height;
    
    unsigned char *imagein = imgin->data;

    Image *imgout = ImageCreate(width, height);
    unsigned char *imageout = imgout->data;

    unsigned char *gpuin;
    unsigned char *gpuint;
    unsigned char *gpuout;

    cudaMalloc((void **)&gpuin, size * 3* sizeof(unsigned char));
    cudaMalloc((void **)&gpuint, size * 3 *sizeof(unsigned char));
    cudaMalloc((void **)&gpuout, size * 3 *sizeof(unsigned char));

    cudaMemcpy(gpuin, imagein, size *3* sizeof(unsigned char), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;

    row<<<numBlocks, blockSize>>>(size, r, width, height, gpuin, gpuint);
    cudaDeviceSynchronize();
    col<<<numBlocks, blockSize>>>(size, r, width, height, gpuint, gpuout);
    cudaDeviceSynchronize();

    cudaMemcpy(imageout, gpuout, size *3* sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(gpuin);
    cudaFree(gpuint);
    cudaFree(gpuout);

    return imgout;
}


