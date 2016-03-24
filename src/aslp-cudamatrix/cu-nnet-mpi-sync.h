// aslp-aslp-cudamatrix/cu-nnet-mpi-sync.h

// Copyright 2016 ASLP (author: zhangbinbin)

// Created on 2016-02-24

#ifndef KALDI_CUDAMATRIX_CU_NNET_MPI_SYNC_H_
#define KALDI_CUDAMATRIX_CU_NNET_MPI_SYNC_H_

void cuda_average(float *dst, const float *src, int num, cudaStream_t &stream);
void cuda_average(double *dst, const double *src, int num, cudaStream_t &stream);

#endif // KALDI_CUDAMATRIX_CU_NNET_MPI_SYNC_H_
