/* Created on 2017-03-07
 * Author: Zhang Binbin
 */

#ifndef ASLP_PARALLEL_SOD_WORKER_H_
#define ASLP_PARALLEL_SOD_WORKER_H_

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"

#include "aslp-parallel/mpi-node.h"
#include "aslp-parallel/itf.h"

#include "aslp-parallel/optimizer.h"

namespace kaldi {

// Synchronous Optimize the Difference between Global and local model

class SodWorker : public IWorker {
public:
    SodWorker(const OptimizerOption &config): config_(config) {}
    ~SodWorker();
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
    std::vector<Vector<BaseFloat> *> grad_cpu_params_;
    std::vector<Optimizer *> optimizers_;
    const OptimizerOption &config_; 
};

} // namespace kaldi

#endif
