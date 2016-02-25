// aslp-nnet/nnt-mpi-sync.cc

// Copyright 2016 ASLP (Author: zhangbinbin)
// Created on 2016-02-21

#include "aslp-nnet/nnet-mpi-sync.h"
#include "aslp-cudamatrix/cu-nnet-mpi-sync.h"

namespace kaldi {
namespace aslp_nnet {

NnetMpiSync::NnetMpiSync() {
    // Mpi init
    MPI::Init();
    rank_ = MPI::COMM_WORLD.Get_rank();
    size_ = MPI::COMM_WORLD.Get_size();
    if (size_ != 2) {
        // Only support two gpu card
        std::cerr << "num of jobs must be 2\n";
        MPI::COMM_WORLD.Abort(1);
    }
    // Peer, 0->1 1->0
    peer_ = (rank_ + 1) % 2;
    total_params_ = 0;
    is_init_ = false;
    self_done_ = 0;
    peer_done_ = 0;
}

NnetMpiSync::~NnetMpiSync() {
    if (is_init_) {
        for (int i = 0; i < num_params_; i++) {
            CU_SAFE_CALL(cudaFree(peer_params_[i].first));
        }
        //delete [] send_buf_;
        //delete [] recv_buf_;
        cudaFreeHost(send_buf_);
        cudaFreeHost(recv_buf_);
        for (int i = 0; i < num_params_; i++) {
            CU_SAFE_CALL(cudaStreamDestroy(streams_[i]));
        }
    }
    MPI::Finalize();
}

void NnetMpiSync::Init(const std::vector<std::pair<BaseFloat *, int> > &params) {
    num_params_ = params.size();
    total_params_ = 0;
    my_params_ = params;
    peer_params_ = params;
    for (int i = 0; i < num_params_; i++) {
        //KALDI_LOG << my_params_[i].first << " " << my_params_[i].second;
        total_params_ += params[i].second;
    }
    KALDI_LOG << "num params " << num_params_;
    KALDI_LOG << "total params " << total_params_;
    // Allocate gpu buffer
    for (int i = 0; i < num_params_; i++) {
        CU_SAFE_CALL(cudaMalloc((void **)&peer_params_[i].first, 
            sizeof(BaseFloat) * peer_params_[i].second));
    }
    //for (int i = 0; i < num_params_; i++) {
    //    KALDI_LOG << peer_params_[i].first << " " << peer_params_[i].second;
    //}
    // Allocate send and receive buffer
    //send_buf_ = new BaseFloat[total_params_];
    //recv_buf_ = new BaseFloat[total_params_];
    cudaHostAlloc((void **)&send_buf_, 
        total_params_ * sizeof(BaseFloat), cudaHostAllocDefault);
    cudaHostAlloc((void **)&recv_buf_, 
        total_params_ * sizeof(BaseFloat), cudaHostAllocDefault);
    // Allocate cuda streams
    streams_ = new cudaStream_t[num_params_];
    for (int i = 0; i < num_params_; i++) {
        CU_SAFE_CALL(cudaStreamCreate(&streams_[i]));
    }

    is_init_ = true;
}

// Just for mpi test
void NnetMpiSync::SyncTest() {
    int s = 1, r = 2;
    MPI_Isend(&s, 1, MPI_INT, peer_, 2, MPI_COMM_WORLD, &send_req_);
    MPI_Irecv(&r, 1, MPI_INT, peer_, 2, MPI_COMM_WORLD, &recv_req_);
    MPI_Wait(&send_req_, &send_status_);
    MPI_Wait(&recv_req_, &recv_status_);
    std::cout << "rank " << rank_ << " " << s << " " << r 
              << std::endl << std::flush;
}

void NnetMpiSync::Sync() {
    //static int counter = 0;
    //KALDI_LOG << "sync once " << counter++;
    // 1. Copy my_params to send_buf_ 
    int offset = 0;
    for (int i = 0; i < num_params_; i++) {
        // KALDI_LOG << i << " " <<  send_buf_ << " " << offset;
        //CU_SAFE_CALL(cudaMemcpy(send_buf_ + offset, my_params_[i].first, 
        //    my_params_[i].second * sizeof(float), cudaMemcpyDeviceToHost));
        CU_SAFE_CALL(cudaMemcpyAsync(send_buf_ + offset, 
                    my_params_[i].first, my_params_[i].second * sizeof(float), 
                    cudaMemcpyDeviceToHost, streams_[i]));
        offset += my_params_[i].second;
    }
    CU_SAFE_CALL(cudaDeviceSynchronize());
    // 2. Mpi send and recv
    MPI_Isend(&self_done_, 1, MPI_INT, peer_, 0,
            MPI_COMM_WORLD, &send_done_req_);
    MPI_Irecv(&peer_done_, 1, MPI_INT, peer_, 0,
            MPI_COMM_WORLD, &recv_done_req_);
    MPI_Isend(send_buf_, total_params_, MPI_FLOAT, peer_, 1, 
            MPI_COMM_WORLD, &send_req_);
    MPI_Irecv(recv_buf_, total_params_, MPI_FLOAT, peer_, 1,
            MPI_COMM_WORLD, &recv_req_);
    MPI_Wait(&send_done_req_, &send_done_status_);
    MPI_Wait(&recv_done_req_, &recv_done_status_);
    MPI_Wait(&send_req_, &send_status_);
    MPI_Wait(&recv_req_, &recv_status_);
    // 3. Copy recv_buf_ to peer_params_ & Do average
    if (PeerDone()) return;
    if (SelfDone()) {
        // Copy peer's param to self param
        offset = 0;
        for (int i = 0; i < num_params_; i++) {
            CU_SAFE_CALL(cudaMemcpyAsync(my_params_[i].first, 
                recv_buf_ + offset, my_params_[i].second * sizeof(float), 
                cudaMemcpyHostToDevice , streams_[i]));
            offset += my_params_[i].second;
        }
        CU_SAFE_CALL(cudaDeviceSynchronize());
    }
    else {
        // Do average my_params_ = (my_params_ + peer_params_) / 2
        offset = 0;
        for (int i = 0; i < num_params_; i++) {
            CU_SAFE_CALL(cudaMemcpyAsync(peer_params_[i].first, 
                        recv_buf_ + offset, peer_params_[i].second * sizeof(float), 
                        cudaMemcpyHostToDevice , streams_[i]));
            offset += peer_params_[i].second;
            // Do average
            cuda_average(my_params_[i].first, peer_params_[i].first, 
                    my_params_[i].second, streams_[i]);
        }
        CU_SAFE_CALL(cudaDeviceSynchronize());
    }
}

void NnetMpiSync::SyncStatus() const {
    std::cerr << "self" << "\t" << "peer" << "\t" << "all" << "\n"; 
    std::cerr << SelfDone() << "\t" << PeerDone() << "\t" << AllDone() << "\n";
}

} // namespace aslp_nnet
} // namespace kaldi
