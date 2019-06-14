#include "ppmFile.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

Image *filter(int imageW, int imageH, int r, Image *image);
Image *sobel(int imageW, int imageH, int r, Image *image);

int main(int argc, char *argv[]) {
  int r = atoi(argv[1]);
  Image *unfilterImage = ImageRead(argv[2]);
  Image *newIm;
  int imageH = ImageHeight(unfilterImage);
  int imageW = ImageWidth(unfilterImage);
  newIm = filter(imageW, imageH, r, unfilterImage);
  ImageWrite(newIm, argv[3]);
}

// TODO: neeed to optimize code by implmenting seperable filter algorithm
Image *sobel(int imageW, int imageH, int r, Image *image) {

  Image *im = ImageCreate(
      imageW, imageH); // first image (contains result of horizontal filter)
  int sobel_h[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1}; // Horizontal
  int sobel_v[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1}; // Vertical
#pragma omp parallel for collapse(2)
  for (int y = 0; y < imageH; y++) {
    for (int x = 0; x < imageW; x++) {
      // getting value by applying the filter
      int count = 0;
      int sumr = 0;
      int sumb = 0;
      int sumg = 0;
      int sumrY = 0;
      int sumbY = 0;
      int sumgY = 0;
      int horLimit = x + r;
      int verLimit = y + r;
      for (int currY = y - r; currY <= verLimit; currY++) {
        for (int currX = x - r; currX <= horLimit; currX++) {
          if (currX < 0 || currX >= imageW || currY >= imageH || currY < 0) {
            count++;
            continue;
          } else {
            sumr =
                sumr + ImageGetPixel(image, currX, currY, 0) * sobel_h[count];
            sumb =
                sumb + ImageGetPixel(image, currX, currY, 1) * sobel_h[count];
            sumg =
                sumg + ImageGetPixel(image, currX, currY, 2) * sobel_h[count];
            sumrY =
                sumrY + ImageGetPixel(image, currX, currY, 0) * sobel_v[count];
            sumbY =
                sumbY + ImageGetPixel(image, currX, currY, 1) * sobel_v[count];
            sumgY =
                sumgY + ImageGetPixel(image, currX, currY, 2) * sobel_v[count];
            count++;
          }
        }
      }
      // getting the magnitude for each color of RGB
      int magr = sqrt(pow(sumr, 2.0) + pow(sumrY, 2.0));
      magr = (magr < 0) ? 0 : magr;
      magr = (magr > 255) ? 255 : magr;

      int magb = sqrt(pow(sumb, 2.0) + pow(sumbY, 2.0));
      magb = (magb < 0) ? 0 : magb;
      magb = (magb > 255) ? 255 : magb;

      int magg = sqrt(pow(sumg, 2.0) + pow(sumgY, 2.0));
      magg = (magg < 0) ? 0 : magg;
      magg = (magg > 255) ? 255 : magg;

      // Converting RGB to grayscale
      int avg = ceil((magr + magb + magg) / 3);

      // applying the filter to the image
      ImageSetPixel(im, x, y, 0, magr);
      ImageSetPixel(im, x, y, 1, magb);
      ImageSetPixel(im, x, y, 2, magg);
    }
  }

  return im;
}

// Function: Filters the image using the box filter alogrithm
// Utilized the seperable filter algotihm as well as a rolling mean algorithm
// to filter
Image *filter(int imageW, int imageH, int r, Image *image) {

  Image *im = ImageCreate(
      imageW, imageH); // first image (contains result of horizontal filter)
  Image *im2 = ImageCreate(
      imageW, imageH); // seond image (contains results of vertical filter)

// PART1: applying the horizontal part of the filter
#pragma omp parallel for
  for (int y = 0; y < imageH; ++y) {
    int sumr = 0; // sum for red channel
    int sumb = 0; // sum for blue channel
    int sumg = 0; // sum for green channel

    // getting the values of pixels in the first segment for each row
    for (int currX = 0; currX <= r; ++currX) {
      sumr += ImageGetPixel(image, currX, y, 0);
      sumg += ImageGetPixel(image, currX, y, 1);
      sumb += ImageGetPixel(image, currX, y, 2);
    }

    // count of pixels in first segement
    int c = (r + 1);

    ImageSetPixel(im, 0, y, 0, sumr / c);
    ImageSetPixel(im, 0, y, 1, sumg / c);
    ImageSetPixel(im, 0, y, 2, sumb / c);

    for (int x = 1; x < imageW; x++) {

      // The x-edges of the screen
      int left = ((x - r < 0) ? 0 : x - r);
      int right = ((x + r >= imageW) ? imageW - 1 : x + r);

      // getting count of pixels to divide by
      int count = (right - left + 1);

      // subtracting the old pixel values if possible
      if (x > r) {
        int loc = x - r - 1;
        sumr -= ImageGetPixel(image, loc, y, 0);
        sumg -= ImageGetPixel(image, loc, y, 1);
        sumb -= ImageGetPixel(image, loc, y, 2);
      }
      // adding the new pixel values if possible
      if (x < imageW - r) {
        int loc = x + r;
        sumr += ImageGetPixel(image, loc, y, 0);
        sumg += ImageGetPixel(image, loc, y, 1);
        sumb += ImageGetPixel(image, loc, y, 2);
      }
      // applying the filters to the pixel
      ImageSetPixel(im, x, y, 0, sumr / count);
      ImageSetPixel(im, x, y, 1, sumg / count);
      ImageSetPixel(im, x, y, 2, sumb / count);
    }
  }

// PART2: applying the vertical part of the filter
#pragma omp parallel for
  for (int x = 0; x < imageW; ++x) {
    int sumr = 0; // sum for red channel
    int sumb = 0; // sum for blue channel
    int sumg = 0; // sum for green channel

    // getting the values of pixels in the first segment for each row
    for (int currY = 0; currY <= r; ++currY) {
      sumr += ImageGetPixel(im, x, currY, 0);
      sumg += ImageGetPixel(im, x, currY, 1);
      sumb += ImageGetPixel(im, x, currY, 2);
    }
    int c = (r + 1);

    // Note: we apply the vertical filter to the horizontal image
    ImageSetPixel(im, x, 0, 0, sumr / c);
    ImageSetPixel(im, x, 0, 1, sumg / c);
    ImageSetPixel(im, x, 0, 2, sumb / c);

    for (int y = 1; y < imageH; ++y) {
      // The y-edges of the screen
      int top = ((y - r < 0) ? 0 : y - r);
      int bottom = ((y + r >= imageH) ? imageH - 1 : y + r);

      // getting count of pixels to divide by
      int count = (bottom - top + 1);

      // subtracting the old pixel values if possible
      if (y > r) {
        int loc = top - 1;
        sumr -= ImageGetPixel(im, x, loc, 0);
        sumg -= ImageGetPixel(im, x, loc, 1);
        sumb -= ImageGetPixel(im, x, loc, 2);
      }
      // adding the new pixel values if possible
      if (y < imageH - r) {
        int loc = bottom;
        sumr += ImageGetPixel(im, x, loc, 0);
        sumg += ImageGetPixel(im, x, loc, 1);
        sumb += ImageGetPixel(im, x, loc, 2);
      }

      // applying the filters to the pixel
      ImageSetPixel(im2, x, y, 0, sumr / count);
      ImageSetPixel(im2, x, y, 1, sumg / count);
      ImageSetPixel(im2, x, y, 2, sumb / count);
    }
  }

  return im2;
}
