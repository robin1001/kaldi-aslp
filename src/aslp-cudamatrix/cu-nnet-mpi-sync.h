// aslp-cudamatrix/cu-nnet-mpi-sync.h

// Copyright 2016 ASLP (author: zhangbinbin)

// Created on 2016-02-24

#ifndef ASLP_CUDAMATRIX_CU_NNET_MPI_SYNC_H_
#define ASLP_CUDAMATRIX_CU_NNET_MPI_SYNC_H_

void cuda_average(float *dst, const float *src, int num, cudaStream_t &stream);
void cuda_average(double *dst, const double *src, int num, cudaStream_t &stream);

#endif // ASLP_CUDAMATRIX_CU_NNET_MPI_SYNC_H_
