// aslp-cudamatrix/cu-nnet-mpi-sync.cu

// Copyright 2016 ASLP (author: zhangbinbin)

// Created on 2016-02-24

#include "curand.h"

#ifdef CURAND_CHECK
#undef CURAND_CHECK
#endif

#define CURAND_CHECK(status) { curandAssert(status, __FILE__, __LINE__); }
#include "stdio.h"
inline void curandAssert(curandStatus_t status, const char *file, int line, bool abort=true) {
    if (status != CURAND_STATUS_SUCCESS) {
        printf("curandAssert: error code %d in file: %s line: %d\n", status, file, line);
        if (abort) exit(status);
    }
}

const int BLOCK1D = 512;
//const int BLOCK2D = 32;

inline int divup(int x, int y) { return (x + y - 1) / y; }

/// Average 
template<typename Real>
__global__
static void cuda_average_kernel(Real *dst, const Real *src, int num) {
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int bid = blockIdx.x + blockIdx.y * gridDim.x;
    int step = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    for (int i = tid + bid * blockDim.x * blockDim.y; i < num; i += step) {
        dst[i] = (dst[i] + src[i]) / 2;
    }
}

template<typename Real>
void cuda_average_impl(Real *dst, const Real *src, int num, cudaStream_t &stream) {
    dim3 block(BLOCK1D);
    dim3 grid(divup(num, BLOCK1D));
    cuda_average_kernel<<<grid, block, 0, stream>>>(dst, src, num);
}

void cuda_average(float *dst, const float *src, int num, cudaStream_t &stream) {
    cuda_average_impl(dst, src, num, stream);
}

void cuda_average(double *dst, const double *src, int num, cudaStream_t &stream) {
    cuda_average_impl(dst, src, num, stream);
}

