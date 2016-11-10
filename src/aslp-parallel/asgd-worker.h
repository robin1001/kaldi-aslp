/* Created on 2016-09-24
 * Author: Zhang Binbin, Li Wenpeng, He Changqing
 */

#ifndef ASLP_PARALLEL_ASGD_WORKER_H_
#define ASLP_PARALLEL_ASGD_WORKER_H_

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"

#include "aslp-parallel/mpi-node.h"
#include "aslp-parallel/itf.h"

namespace kaldi {

// AsgdWorker:  ASGD Worker
// Refer "ASGD" for details

class AsgdWorker: public IWorker {
public:
    AsgdWorker() {};
    ~AsgdWorker();
    // @params type pair: first is to the gpu data points, 
    //                    second is the size of it
    void InitParam(const std::vector<std::pair<BaseFloat *, int> > &params); 
    
    // Synchronize with server
    bool Synchronize(int num_worker_samples); 
    
    void Stop();
private:
    // Here we use CuSubVector for that the memory is hold and managed by train model,
    // CuSubVector only share and update this pointer, refer to CuSubVector for details
    std::vector<CuSubVector<BaseFloat> *> worker_gpu_params_;
    std::vector<CuVector<BaseFloat> *> prev_worker_gpu_params_;
    std::vector<Vector<BaseFloat> *> worker_cpu_params_;
};

} // namespace kaldi

#endif


