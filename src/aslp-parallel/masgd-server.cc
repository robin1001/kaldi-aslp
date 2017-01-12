/* Created on 2017-01-12
 * Author: Zhang Binbin
 */

#include "aslp-parallel/masgd-server.h"


namespace kaldi {

void MasgdServer::InitParam(
        const std::vector<std::pair<BaseFloat *, int> > &params) {
    server_gpu_params_.resize(params.size());
    worker_gpu_params_.resize(params.size());
    cpu_params_.resize(params.size());
#if MASGD_TYPE == GMASGD
    diffs_.resize(params.size());
#elif MASGD_TYPE == LMASGD
    diffs_.resize(NumNodes() - 1);
    for (int i = 0; i < NumNodes() - 1; i++) {
        diffs_[i].resize(params.size());
    }
#else 
    #error "Unknown masgd type"
#endif
    for (int i = 0; i < params.size(); i++) {
        server_gpu_params_[i] = new 
            CuSubVector<BaseFloat>(params[i].first, params[i].second); 
        worker_gpu_params_[i] = new CuVector<BaseFloat>(params[i].second);
        cpu_params_[i] = new Vector<BaseFloat>(params[i].second);
#if MASGD_TYPE == GMASGD
        diffs_[i] = new CuVector<BaseFloat>(params[i].second);
#elif MASGD_TYPE == LMASGD
        for (int j = 0; j < NumNodes() - 1; j++) {
            diffs_[j][i] = new CuVector<BaseFloat>(params[i].second);
        }
#else 
    #error "Unknown masgd type"
#endif
    }
}

MasgdServer::~MasgdServer() {
    KALDI_ASSERT(server_gpu_params_.size() == cpu_params_.size());
    KALDI_ASSERT(worker_gpu_params_.size() == cpu_params_.size());
    for (int i = 0; i < server_gpu_params_.size(); i++) {
        delete server_gpu_params_[i];
        delete worker_gpu_params_[i];
        delete cpu_params_[i];
#if MASGD_TYPE == GMASGD
        delete diffs_[i];
#elif MASGD_TYPE == LMASGD
        for (int j = 0; j < NumNodes() - 1; j++) {
            delete diffs_[j][i];
        }
#else 
    #error "Unknown masgd type"
#endif
    }
}

void MasgdServer::Run() {
    int num_running_workers = NumNodes() - 1;
    int synchronized_count = 0;
	std::vector<int> waited_worker;
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
				++synchronized_count;
                if (sync_period_ > 0 && synchronized_count >= sync_period_)
					waited_worker.push_back(worker_rank);
				Update(worker_rank, synchronized_count); 
                break;
            default:
                KALDI_WARN << "Unknown mpi msg type " << msg_type;
        }
        if (sync_period_ > 0 && 
                synchronized_count >= sync_period_ && 
                waited_worker.size() == num_running_workers && 
                num_running_workers != 0) {
			for (int j = 0; j < waited_worker.size(); j++) {
				KALDI_LOG << "Worker " << waited_worker[j] << " synchronized!";
				for (int i = 0; i < cpu_params_.size(); i++) {
        		//KALDI_LOG << "send " << i << " size " << cpu_params_[i]->Dim();
        			MPI_Send(cpu_params_[i]->Data(), cpu_params_[i]->Dim(), 
            	    	 	 MPI_FLOAT, waited_worker[j], kTagModel, MPI_COMM_WORLD);
    			}
			}
			synchronized_count = synchronized_count - sync_period_;
			waited_worker.clear();
		}
    }

    KALDI_LOG << "All worker finished";
}

void MasgdServer::Update(int worker_rank, int synchronized_count) {
    MPI_Status status;
    // 1. receive gradient from worker
    for (int i = 0; i < cpu_params_.size(); i++) {
        //KALDI_LOG << "recv " << i << " size " << cpu_params_[i]->Dim();
        MPI_Recv(cpu_params_[i]->Data(), cpu_params_[i]->Dim(),
                 MPI_FLOAT, worker_rank, i, MPI_COMM_WORLD, &status);
        worker_gpu_params_[i]->CopyFromVec(*cpu_params_[i]);
    }
    // 2. update model
    for (int i = 0; i < worker_gpu_params_.size(); i++) {
#if MASGD_TYPE == GMASGD
        diffs_[i]->AddVec(1.0, *worker_gpu_params_[i], momentum_);
        server_gpu_params_[i]->AddVec(1.0, *diffs_[i], 1.0);
#elif MASGD_TYPE == LMASGD
        diffs_[worker_rank-1][i]->AddVec(1.0, *worker_gpu_params_[i], momentum_);
        server_gpu_params_[i]->AddVec(1.0, *diffs_[worker_rank-1][i], 1.0);
#else 
    #error "Unknown masgd type"
#endif
        cpu_params_[i]->CopyFromVec(*server_gpu_params_[i]);
    }
    // 3. send new model to worker
    if (synchronized_count < sync_period_ || sync_period_ <= 0) {
		for (int i = 0; i < cpu_params_.size(); i++) {
        	//KALDI_LOG << "send " << i << " size " << cpu_params_[i]->Dim();
        	MPI_Send(cpu_params_[i]->Data(), cpu_params_[i]->Dim(), 
            	     MPI_FLOAT, worker_rank, kTagModel, MPI_COMM_WORLD);
    	}
	}
}

} // namespace kaldi
