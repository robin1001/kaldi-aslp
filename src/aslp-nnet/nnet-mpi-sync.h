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
    NnetMpiSync();
    ~NnetMpiSync();
    void Init(const std::vector<std::pair<BaseFloat *, int> > &params);
    void Sync();
    void SyncStatus() const;
    /// Just for test
    void SyncTest();
    void SetSelfDone() {
        self_done_ = 1;
    }
    /// If self node finish data processing
    bool SelfDone() const {
        return self_done_ > 0;
    }
    /// If peer finish data processing
    bool PeerDone() const {
        return peer_done_ > 0;
    }
    /// If self and peer node all finish data processing
    bool AllDone() const {
        return ((self_done_ > 0) && (peer_done_ > 0));
    }
    int Rank() const { return rank_; }
private:
    MPI_Request send_req_, recv_req_;
    MPI_Status send_status_, recv_status_;
    MPI_Request send_done_req_, recv_done_req_;
    MPI_Status send_done_status_, recv_done_status_;
    int rank_, size_, peer_;
    int num_params_, total_params_;
    std::vector<std::pair<BaseFloat *, int> > my_params_, peer_params_;
    BaseFloat *send_buf_, *recv_buf_;
    cudaStream_t *streams_;
    bool is_init_;
    int peer_done_;
    int self_done_;
};


} // namespace aslp_nnet
} // namespace kaldi

#endif
