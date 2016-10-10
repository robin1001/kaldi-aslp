/* Created on 2016-09-24
 * Author: Zhang Binbin, Li Wenpeng, He Changqing
 */

#include "aslp-parallel/asgd-worker.h"


namespace kaldi {

void AsgdWorker::InitParam(
        const std::vector<std::pair<BaseFloat *, int> > &params) {
    prev_worker_gpu_params_.resize(params.size());
    worker_gpu_params_.resize(params.size());
    worker_cpu_params_.resize(params.size());

    for (int i = 0; i < params.size(); i++) {
        worker_gpu_params_[i] = new 
            CuSubVector<BaseFloat>(params[i].first, params[i].second); 
        prev_worker_gpu_params_[i] = new CuVector<BaseFloat>(params[i].second);
        prev_worker_gpu_params_[i]->CopyFromVec(*worker_gpu_params_[i]);
        worker_cpu_params_[i] = new Vector<BaseFloat>(params[i].second);
    }
}

AsgdWorker::~AsgdWorker() {
    KALDI_ASSERT(worker_gpu_params_.size() == worker_cpu_params_.size());
    for (int i = 0; i < worker_gpu_params_.size(); i++) {
        delete prev_worker_gpu_params_[i];
        delete worker_gpu_params_[i];
        delete worker_cpu_params_[i];
    }
}

bool AsgdWorker::Synchronize(int num_worker_samples) {
    (void)num_worker_samples;
    int msg_type = kMsgSynchronize;
    // 1. send synchronize signal 
    MPI_Send(&msg_type, 1, MPI_INT, MainNode(), kTagMsg, MPI_COMM_WORLD);
    // 2.1 copy worker_gpu_params_ to worker_cpu_params_
    for (int i = 0; i < worker_gpu_params_.size(); i++) {
        // get accumulated gradient
        worker_gpu_params_[i]->AddVec(-1.0, *prev_worker_gpu_params_[i], 1.0);
        worker_cpu_params_[i]->CopyFromVec(*worker_gpu_params_[i]);
    }

    // 2.2 send accumulated gradient woker_cpu_params_
    for (int i = 0; i < worker_cpu_params_.size(); i++) {
        //KALDI_LOG << "send " << i << " size " << worker_cpu_params_[i]->Dim();
        MPI_Send(worker_cpu_params_[i]->Data(), worker_cpu_params_[i]->Dim(), 
                 MPI_FLOAT, MainNode(), i, MPI_COMM_WORLD);
    }
    // 2.3 recive server_cpu_params_
    MPI_Status status;
    for (int i = 0; i < worker_cpu_params_.size(); i++) {
        //KALDI_LOG << "recv " << i << " size " << worker_cpu_params_[i]->Dim();
        MPI_Recv(worker_cpu_params_[i]->Data(), worker_cpu_params_[i]->Dim(),
                 MPI_FLOAT, MainNode(), kTagModel, MPI_COMM_WORLD, &status);
    }

    // 2.3 copy worker_cpu_params_ to worker_gpu_params_ and prev_worker_gpu_params_ 
    for (int i = 0; i < worker_gpu_params_.size(); i++) {
        worker_gpu_params_[i]->CopyFromVec(*worker_cpu_params_[i]);
        prev_worker_gpu_params_[i]->CopyFromVec(*worker_cpu_params_[i]);
    }
    
    // always return ture
    return true;
}

void AsgdWorker::Stop() {
    // Send Stop signal
    int msg_type = kMsgFinished;
    MPI_Send(&msg_type, 1, MPI_INT, MainNode(), kTagMsg, MPI_COMM_WORLD);
    KALDI_LOG << "Worker " << Rank() << " finished";
}

} // namespace kaldi
