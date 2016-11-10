/* Created on 2016-07-28
 * Author: Zhang Binbin
 */

#include "aslp-parallel/easgd-server.h"


namespace kaldi {

void EasgdServer::InitParam(
        const std::vector<std::pair<BaseFloat *, int> > &params) {
    server_gpu_params_.resize(params.size());
    server_cpu_params_.resize(params.size());
    worker_gpu_params_.resize(params.size());
    worker_cpu_params_.resize(params.size());

    for (int i = 0; i < params.size(); i++) {
        server_gpu_params_[i] = new 
            CuSubVector<BaseFloat>(params[i].first, params[i].second); 
        server_cpu_params_[i] = new Vector<BaseFloat>(params[i].second);
        worker_gpu_params_[i] = new CuVector<BaseFloat>(params[i].second);
        worker_cpu_params_[i] = new Vector<BaseFloat>(params[i].second);
    }
}

EasgdServer::~EasgdServer() {
    KALDI_ASSERT(server_gpu_params_.size() == server_cpu_params_.size());
    KALDI_ASSERT(worker_gpu_params_.size() == worker_cpu_params_.size());
    for (int i = 0; i < server_gpu_params_.size(); i++) {
        delete server_gpu_params_[i];
        delete server_cpu_params_[i];
        delete worker_gpu_params_[i];
        delete worker_cpu_params_[i];
    }
}

void EasgdServer::Run() {
    int num_running_workers = NumNodes() - 1;
    MPI_Status status;
    int msg_type, worker_rank;
    while (num_running_workers > 0) {
        // tag 0, msg type 
        MPI_Recv(&msg_type, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, 
            MPI_COMM_WORLD, &status);
        worker_rank = status.MPI_SOURCE;
        KALDI_VLOG(2) << "Worker rank " << worker_rank << " Msg " << msg_type;
        switch (msg_type) {
            case kMsgFinished:
                num_running_workers--;
                KALDI_LOG << "Worker " << worker_rank << " Finished ";
                break;
            case kMsgSynchronize: 
                Update(worker_rank); 
                break;
            default:
                KALDI_WARN << "Unknown mpi msg type " << msg_type;
        }
    }

    KALDI_LOG << "All worker finished";
}

void EasgdServer::Update(int worker_rank) {
    // 1. copy server_gpu_params_ to server_cpu_params_ 
    for (int i = 0; i < server_cpu_params_.size(); i++) {
        server_cpu_params_[i]->CopyFromVec(*server_gpu_params_[i]);
    }
    // 2. send server_cpu_params_ and recv worker_cpu_params_
    MPI_Status status;
    for (int i = 0; i < server_cpu_params_.size(); i++) {
        MPI_Sendrecv(server_cpu_params_[i]->Data(), server_cpu_params_[i]->Dim(), 
                     MPI_FLOAT, worker_rank, i,
                     worker_cpu_params_[i]->Data(), worker_cpu_params_[i]->Dim(),
                     MPI_FLOAT, worker_rank, i,
                     MPI_COMM_WORLD, &status);
    }
    // 3. copy worker_cpu_params_ to worker_gpu_params_
    for (int i = 0; i < worker_gpu_params_.size(); i++) {
        worker_gpu_params_[i]->CopyFromVec(*worker_cpu_params_[i]);
    }   
    // 4. update server gpu model
    for (int i = 0; i < server_gpu_params_.size(); i++) {
        //x_server = x_server + alpha(x_worker - x_server)
        //         = (1 - alpha) * x_server + alpha * x_worker
        server_gpu_params_[i]->AddVec(alpha_, *worker_gpu_params_[i], 1 - alpha_);
    }
}


} // namespace kaldi
