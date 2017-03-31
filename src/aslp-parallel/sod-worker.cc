/* Created on 2017-03-07
 * Author: Zhang Binbin
 */

#include "aslp-parallel/sod-worker.h"

namespace kaldi {

void SodWorker::InitParam(
        const std::vector<std::pair<BaseFloat *, int> > &params) {
    gpu_params_.resize(params.size());
    prev_gpu_params_.resize(params.size());
    grad_gpu_params_.resize(params.size());
    grad_cpu_params_.resize(params.size());
    optimizers_.resize(params.size());
    for (int i = 0; i < params.size(); i++) {
        gpu_params_[i] = new 
            CuSubVector<BaseFloat>(params[i].first, params[i].second); 
        prev_gpu_params_[i] = new CuVector<BaseFloat>(params[i].second);
        prev_gpu_params_[i]->CopyFromVec(*gpu_params_[i]);
        grad_gpu_params_[i] = new CuVector<BaseFloat>(params[i].second);
        grad_cpu_params_[i] = new Vector<BaseFloat>(params[i].second);
        optimizers_[i] = config_.NewInstance(params[i].second);
    }
}

SodWorker::~SodWorker() {
    for (int i = 0; i < gpu_params_.size(); i++) {
        delete gpu_params_[i];
        delete prev_gpu_params_[i];
        delete grad_gpu_params_[i];
        delete grad_cpu_params_[i];
        delete optimizers_[i];
    }
}

bool SodWorker::Synchronize(int num_worker_samples) {
    int num_all_samples = num_worker_samples; 
    AllReduce(&num_all_samples, 1);
    /// All workers finished it's data, return instantly
    if (num_all_samples <= 0) {
        KALDI_LOG << "All worker finished their data";
        return false;
    }
   
    for (int i = 0; i < gpu_params_.size(); i++) {
        // 1. calc grad of wg(t), wg(t-1) - w(t) 
        grad_gpu_params_[i]->CopyFromVec(*prev_gpu_params_[i]);
        grad_gpu_params_[i]->AddVec(-1.0, *gpu_params_[i]);
        grad_cpu_params_[i]->CopyFromVec(*grad_gpu_params_[i]);
        // 2. reduce
        AllReduce(grad_cpu_params_[i]->Data(), grad_cpu_params_[i]->Dim());
        // 3. copy to gpu
        grad_gpu_params_[i]->CopyFromVec(*grad_cpu_params_[i]);
        // 4. using solver to optimize grad_gpu_params_[i]
        optimizers_[i]->Optimize(*grad_gpu_params_[i], gpu_params_[i]);
        // 5. update prev
        prev_gpu_params_[i]->CopyFromVec(*gpu_params_[i]);
    }
    return true;
}

void SodWorker::Stop() {
    // Wait other worker to finish their data, it is called when worker finish 
    // it's own data, then loop to wait others 
    KALDI_LOG << "Worker " << Rank() << "finished, waitting for others";
    while (Synchronize(0)); 
}

}
