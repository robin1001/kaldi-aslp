/* Created on 2016-07-27
 * Author: Zhang Binbin
 */

#ifndef ASLP_PARALLEL_BMUF_WORKER_H_
#define ASLP_PARALLEL_BMUF_WORKER_H_

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"

#include "aslp-parallel/mpi-node.h"
#include "aslp-parallel/itf.h"

namespace kaldi {

// BmufWorker: Do BMUF(Block Momentum Update Filter) model update
// Refer 1. "Deep learning with Elastic Averaging SGD" for details
//       2. "SCALABLE TRAINING OF DEEP LEARNING MACHINES BY INCREMENTAL BLOCK 
//           TRAINING WITH INTRA-BLOCK PARALLEL OPTIMIZATION AND BLOCKWISE MODEL-UPDATE FILTERING" 
//       3. https://github.com/Microsoft/CNTK/wiki/Multiple-GPUs-and-machines

// Personally, I think that BMUF comes from synchronous EASGD, which try to minimize the difference 
// of global model and local model, but there are some things different: 
// 1) All workers use the same global model after each synchronization, just like model averaging.
// 2) BMUF use momentum.

class BmufWorker : public IWorker {
public:
    BmufWorker(float learn_rate = 1.0, float momentum = 0.9): 
        learn_rate_(learn_rate), momentum_(momentum) {}
    ~BmufWorker();
    // @params type pair: first is to the gpu data points, 
    //                    second is the size of it
    void InitParam(const std::vector<std::pair<BaseFloat *, int> > &params); 

    // @params: num_worker_samples, new sample frames since last synchronization
    // return false if all worker finished their own data
    bool Synchronize(int num_worker_samples); 

    void Stop();
private:
    // Here we use CuSubVector for that the memory is hold and managed by train model,
    // CuSubVector only share and update this pointer, refer to CuSubVector for details
    std::vector<CuSubVector<BaseFloat> *> gpu_params_;
    std::vector<CuVector<BaseFloat > *> prev_gpu_params_;
    std::vector<CuVector<BaseFloat> *> grad_gpu_params_;
    std::vector<CuVector<BaseFloat> *> prev_grad_gpu_params_;
    std::vector<Vector<BaseFloat> *> grad_cpu_params_;

    float learn_rate_;
    float momentum_; // 
};

//The block momentum and block learning rate are usually automatically set according to the number of workers used, i.e.,
//
//  block_momentum = 1.0 - 1.0/num_of_workers
//  block_learning_rate = 1.0
//Our experience indicates that these settings often yield similar convergence as the standard SGD algorithm up to 64 GPUs, which is the largest experiment we performed. It is also possible to manually specify the these parameters using the following options:
//
//blockMomentumAsTimeConstant specifies the time constant of the low-pass filter in block-level model update. It is calculated as:
//
//blockMomentumAsTimeConstant = -syncPeriod / log(block_momentum)
//# or inversely 
//  block_momentum = exp(-syncPeriod/blockMomentumAsTimeConstant)
//  blockLearningRate specifies the block learning rate.

} // namespace kaldi

#endif
