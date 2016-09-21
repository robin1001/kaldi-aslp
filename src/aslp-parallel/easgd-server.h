/* Created on 2016-07-28
 * Author: Zhang Binbin
 */

#ifndef ASLP_PARALLEL_EASGD_SERVER_H_
#define ASLP_PARALLEL_EASGD_SERVER_H_

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"

#include "aslp-parallel/mpi-node.h"
#include "aslp-parallel/itf.h"

namespace kaldi {

// EasgdServer: Asynchronize Elastic Averaging SGD Server
// Refer "Deep learning with Elastic Averaging SGD" for details

class EasgdServer : public IServer {
public:
    EasgdServer(float alpha = 0.5): alpha_(alpha) {}
    ~EasgdServer();
    // @params type pair: first is to the gpu data points, 
    //                    second is the size of it
    void InitParam(const std::vector<std::pair<BaseFloat *, int> > &params); 
    void SetAlpha(float alpha) {
        KALDI_ASSERT(alpha >= 0.0);
        KALDI_ASSERT(alpha <= 1.0);
        alpha_ = alpha;
    }
    // Start serve
    void Run();
    // Update server model with one worker
    void Update(int worker_rank);
private:
    float alpha_;
    // Here we use CuSubVector for that the memory is hold and managed by train model,
    // CuSubVector only share and update this pointer, refer to CuSubVector for details
    std::vector<CuSubVector<BaseFloat> *> server_gpu_params_;
    std::vector<CuVector<BaseFloat> *> worker_gpu_params_;
    std::vector<Vector<BaseFloat> *> server_cpu_params_, worker_cpu_params_;
};


} // namespace kaldi

#endif


