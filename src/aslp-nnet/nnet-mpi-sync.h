// aslp-nnet/nnt-mpi-sync.h

// Copyright 2016 ASLP (Author: zhangbinbin)
// Created on 2016-02-21


#ifndef ASLP_NNET_NNET_MPI_SYNC_H_
#define ASLP_NNET_NNET_MPI_SYNC_H_

#include "mpi.h"

#include "base/kaldi-common.h"
#include "cudamatrix/cu-common.h"

namespace kaldi {
namespace aslp_nnet {

class NnetMpiSync {
public:
    NnetMpiSync(int sync_period = 1);
    ~NnetMpiSync();
    void Init(const std::vector<std::pair<BaseFloat *, int> > &params);
    void Sync();
    /// Just for test
    void SyncTest();
    void SetDone(bool done);
    /// If peer finish data processing
    bool PeerDone();
private:
    MPI_Request send_req_, recv_req_;
    MPI_Status send_status_, recv_status_;
    int rank_, size_, peer_;
    int sync_period_; // every n minibatch to sync
    int num_params_, total_params_;
    std::vector<std::pair<BaseFloat *, int> > my_params_, peer_params_;
    BaseFloat *send_buf_, *recv_buf_;
    cudaStream_t *streams_;
    int64 num_minibatch_;
    bool is_init_;
    bool done_;
};


} // namespace aslp_nnet
} // namespace kaldi

#endif
