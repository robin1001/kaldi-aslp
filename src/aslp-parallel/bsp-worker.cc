/* Created on 2016-07-27
 * Author: Zhang Binbin
 */

#include "aslp-parallel/bsp-worker.h"

namespace kaldi {

void BspWorker::InitParam(
        const std::vector<std::pair<BaseFloat *, int> > &params) {
    gpu_params_.resize(params.size());
    cpu_params_.resize(params.size());
    for (int i = 0; i < params.size(); i++) {
        gpu_params_[i] = new 
            CuSubVector<BaseFloat>(params[i].first, params[i].second); 
        cpu_params_[i] = new Vector<BaseFloat>(params[i].second);
    }
}

BspWorker::~BspWorker() {
    KALDI_ASSERT(gpu_params_.size() == cpu_params_.size());
    for (int i = 0; i < gpu_params_.size(); i++) {
        delete gpu_params_[i];
        delete cpu_params_[i];
    }
}

// Instead of scale then sum, we can also first copy to GPU, 
// then sum and scale on gpu then copy to host,
// sum and scale operation could be implemented in one kernel,
// which my be more efficient
// This implemention is simple and stupid
bool BspWorker::Synchronize(int num_worker_samples) {
    int num_all_samples = num_worker_samples; 
    AllReduce(&num_all_samples, 1);
    // All workers finished it's data, return instantly
    if (num_all_samples <= 0) {
        KALDI_LOG << "All worker finished their data";
        return false;
    }
    // 1. Calc scale
    float factor = float(num_worker_samples) / num_all_samples;
    KALDI_ASSERT(factor >= 0.0 && factor <= 1.0);
    // 2. Do average
    KALDI_ASSERT(gpu_params_.size() == cpu_params_.size());
    for (int i = 0; i < gpu_params_.size(); i++) {
        // scale factor
        gpu_params_[i]->Scale(factor);
        // copy from gpu to host
        cpu_params_[i]->CopyFromVec(*gpu_params_[i]);
        // sum, average 
        AllReduce(cpu_params_[i]->Data(), cpu_params_[i]->Dim());
        // copy to gpu
        gpu_params_[i]->CopyFromVec(*cpu_params_[i]);
    }

    return true;
}

void BspWorker::Stop() {
    // Wait other worker to finish their data, it is called when worker finish 
    // it's own data, then loop to wait others 
    KALDI_LOG << "Worker " << Rank() << "finished, waitting for others";
    while (Synchronize(0)); 
}

} // namespace kaldi
