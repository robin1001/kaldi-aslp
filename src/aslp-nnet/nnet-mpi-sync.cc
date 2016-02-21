// aslp-nnet/nnt-mpi-sync.cc

// Copyright 2016 ASLP (Author: zhangbinbin)

/* Created on 2016-02-21
 * Author: zhangbinbin
 */

#include "aslp-nnet/nnet-mpi-sync.h"

namespace kaldi {
namespace aslp_nnet {
NnetMpiSync::NnetMpiSync(int sync_period): sync_period_(sync_period) {
    KALDI_ASSERT(sync_period > 0);
    MPI::Init();
    rank_ = MPI::COMM_WORLD.Get_rank();
    size_ = MPI::COMM_WORLD.Get_size();
    // Only support two gpu card
    if (size_ != 2) {
        std::cerr << "num of jobs must be 2\n";
        MPI::COMM_WORLD.Abort(1);
    }
    // Peer, 0->1 1->0
    peer_ = (rank_ + 1) % 2;
}

NnetMpiSync::~NnetMpiSync() {
    MPI::Finalize();
}

// Just for mpi test
void NnetMpiSync::SyncTest() {
    int s = 1, r = 2;
    MPI_Isend(&s, 1, MPI_INT, peer_, 0, MPI_COMM_WORLD, &send_req_);
    MPI_Irecv(&r, 1, MPI_INT, peer_, 0, MPI_COMM_WORLD, &recv_req_);
    MPI_Wait(&send_req_, &send_status_);
    MPI_Wait(&recv_req_, &recv_status_);
    std::cout << "rank " << rank_ << " " << s << " " << r << std::endl;
}

void NnetMpiSync::Sync() {

}

} // namespace aslp_nnet
} // namespace kaldi
