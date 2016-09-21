/* Created on 2016-08-01
 * Author: Zhang Binbin
 */

#include "aslp-parallel/easgd-worker.h"


namespace kaldi {

void EasgdWorker::InitParam(
        const std::vector<std::pair<BaseFloat *, int> > &params) {
    server_gpu_params_.resize(params.size());
    server_cpu_params_.resize(params.size());
    worker_gpu_params_.resize(params.size());
    worker_cpu_params_.resize(params.size());

    for (int i = 0; i < params.size(); i++) {
        worker_gpu_params_[i] = new 
            CuSubVector<BaseFloat>(params[i].first, params[i].second); 
        worker_cpu_params_[i] = new Vector<BaseFloat>(params[i].second);
        server_gpu_params_[i] = new CuVector<BaseFloat>(params[i].second);
        server_cpu_params_[i] = new Vector<BaseFloat>(params[i].second);
    }
}

EasgdWorker::~EasgdWorker() {
    KALDI_ASSERT(server_gpu_params_.size() == server_cpu_params_.size());
    KALDI_ASSERT(worker_gpu_params_.size() == worker_cpu_params_.size());
    for (int i = 0; i < server_gpu_params_.size(); i++) {
        delete server_gpu_params_[i];
        delete server_cpu_params_[i];
        delete worker_gpu_params_[i];
        delete worker_cpu_params_[i];
    }
}

bool EasgdWorker::Synchronize(int num_worker_samples) {
    (void)num_worker_samples;
    int msg_type = kMsgSynchronize;
    // 1. send synchronize signal 
    MPI_Send(&msg_type, 1, MPI_INT, MainNode(), kTagMsg, MPI_COMM_WORLD);
    // 2.1 copy worker_gpu_params_ to worker_cpu_params_
    for (int i = 0; i < worker_gpu_params_.size(); i++) {
        worker_cpu_params_[i]->CopyFromVec(*worker_gpu_params_[i]);
    }
    // 2.2 send woker_cpu_params_ and recv server_cpu_params_
    MPI_Status status;
    for (int i = 0; i < server_cpu_params_.size(); i++) {
        MPI_Sendrecv(worker_cpu_params_[i]->Data(), worker_cpu_params_[i]->Dim(), 
                     MPI_FLOAT, MainNode(), kTagModel,
                     server_cpu_params_[i]->Data(), server_cpu_params_[i]->Dim(),
                     MPI_FLOAT, MainNode(), kTagModel,
                     MPI_COMM_WORLD, &status);
    }

    // 2.3 copy server_gpu_params_ to server_cpu_params_ 
    for (int i = 0; i < server_gpu_params_.size(); i++) {
        server_gpu_params_[i]->CopyFromVec(*server_cpu_params_[i]);
    }
    // 2.4 update worker gpu model
    for (int i = 0; i < worker_gpu_params_.size(); i++) {
        worker_gpu_params_[i]->AddVec(alpha_, *server_gpu_params_[i], 1 - alpha_);
    }
    
    // always return ture
    return true;
}

void EasgdWorker::Stop() {
    // Send Stop signal
    int msg_type = kMsgFinished;
    MPI_Send(&msg_type, 1, MPI_INT, MainNode(), kTagMsg, MPI_COMM_WORLD);
    KALDI_LOG << "Worker " << Rank() << " finished";
}

} // namespace kaldi
