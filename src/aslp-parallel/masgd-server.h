/* Created on 2017-01-12
 * Author: Zhang Binbin
 */

#ifndef ASLP_PARALLEL_MASGD_SERVER_H_
#define ASLP_PARALLEL_MASGD_SERVER_H_

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "cudamatrix/cu-matrix.h"
#include "cudamatrix/cu-vector.h"

#include "aslp-parallel/mpi-node.h"
#include "aslp-parallel/itf.h"

namespace kaldi {

// MasgdServer: Asynchronize SGD Server
// Refer ASGD for details

#define GMASGD 0 // global
#define LMASGD 1 // local
#define MASGD_TYPE LMASGD

class MasgdServer : public IServer {
public:
    MasgdServer(int sync_period = 1000, float momentum = 0.9): 
				  sync_period_(sync_period), momentum_(momentum) {}
    ~MasgdServer();
    // @params type pair: first is to the gpu data points, 
    //                    second is the size of it
    void InitParam(const std::vector<std::pair<BaseFloat *, int> > &params); 
    // Start serve
    void Run();
    // Update server model with one worker
    void Update(int worker_rank, int count);
private:
    int32 sync_period_;
    float momentum_;
	// Here we use CuSubVector for that the memory is hold and managed by train model,
    // CuSubVector only share and update this pointer, refer to CuSubVector for details
    std::vector<CuSubVector<BaseFloat> *> server_gpu_params_;
    std::vector<CuVector<BaseFloat> *> worker_gpu_params_;
    std::vector<Vector<BaseFloat> *> cpu_params_;
#if MASGD_TYPE == GMASGD
    std::vector<CuVector<BaseFloat> * > diffs_;
#elif MASGD_TYPE == LMASGD
    std::vector<std::vector<CuVector<BaseFloat >* > > diffs_;
#else 
    #error "Unknown masgd type"
#endif
};


} // namespace kaldi

#endif


