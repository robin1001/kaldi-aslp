// aslp-nnet/nnt-mpi-sync.h

// Copyright 2016 ASLP (Author: zhangbinbin)

/* Created on 2016-02-21
 * Author: zhangbinbin
 */
#ifndef ASLP_NNET_NNET_MPI_SYNC_H_
#define ASLP_NNET_NNET_MPI_SYNC_H_

#include "mpi.h"

#include "base/kaldi-common.h"

namespace kaldi {
namespace aslp_nnet {

class NnetMpiSync {
public:
    NnetMpiSync(int sync_period = 1); 
    ~NnetMpiSync();
    void Sync();
    // Just for test
    void SyncTest();
private:
    MPI_Request send_req_, recv_req_;
    MPI_Status send_status_, recv_status_;
    int rank_, size_, peer_;
    int sync_period_;
};


} // namespace aslp_nnet
} // namespace kaldi

#endif
