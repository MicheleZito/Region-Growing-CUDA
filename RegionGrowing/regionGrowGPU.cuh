#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

void regionGrowingGPU(unsigned char *matrix, unsigned char* source_channel_b, unsigned char* source_channel_g, unsigned char* source_channel_r, unsigned char* out_channel_b, unsigned char* out_channel_g, unsigned char* out_channel_r, int rows, int cols, int step_size, int point_i, int point_j , int soglia);