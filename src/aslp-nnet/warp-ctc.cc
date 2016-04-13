// aslp-nnet/warp-ctc.cc

// Copyright 2016  ASLP (author: Binbin Zhang)
// Created on 2016-04-12

#include "aslp-nnet/warp-ctc.h"
#include "aslp-cudamatrix/cu-math.h"
#include "aslp-cudamatrix/ctc-utils.h"
#include "util/edit-distance.h"

#include <sstream>
#include <iterator>

static inline void throw_on_error(ctcStatus_t status, const char* message) {
    if (status != CTC_STATUS_SUCCESS) {
        throw std::runtime_error(message + (", stat = " + 
                    std::string(ctcGetStatusString(status))));
    }   
}

#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>

static inline void throw_on_error(cudaError_t error, const char* message) {
    if (error) {
        throw thrust::system_error(error, thrust::cuda_category(), message);
    }   
}

namespace kaldi {
namespace aslp_nnet {

void WarpCtc::Eval(const std::vector<int32> &frame_num_utt, const CuMatrixBase<BaseFloat> &net_out,
        std::vector< std::vector<int32> > &labels, CuMatrix<BaseFloat> *diff) {
    KALDI_ASSERT(labels.size() == frame_num_utt.size());
    KALDI_ASSERT(diff != NULL);
    if (use_gpu_) {
        EvalGpu(frame_num_utt, net_out, labels, diff);
    } else {
        EvalCpu(frame_num_utt, net_out, labels, diff);
    }
}

void WarpCtc::EvalGpu(const std::vector<int32> &frame_num_utt, const CuMatrixBase<BaseFloat> &net_out,
        std::vector< std::vector<int32> > &labels, CuMatrix<BaseFloat> *diff) {
    diff->Resize(net_out.NumRows(), net_out.NumCols());
    //KALDI_LOG << net_out.NumCols() << " " << net_out.Stride();
    //KALDI_LOG << net_out.NumRows();

    // Prepare label, feat and their length
    const int minibatch = labels.size();
    std::vector<int> flat_labels;
    std::vector<int> label_lengths;
    for (int i = 0; i < labels.size(); i++) {
        std::vector<int> &l = labels[i];
        flat_labels.insert(flat_labels.end(), l.begin(), l.end());
        label_lengths.push_back(l.size());
    }
    int alphabet_size = net_out.NumCols();
    const std::vector<int> &lengths = frame_num_utt;
    std::vector<float> costs(minibatch);

    // Create ctc compute info
    cudaStream_t stream;
    throw_on_error(cudaStreamCreate(&stream), 
                   "cudaStreamCreate");
    ctcComputeInfo info;
    info.loc = CTC_GPU;
    info.stream = stream;

    // Because kaldi cumatrix align, the NumCols may not equal it's allocated,
    // the real allocated width represented as Stride() in CuMatrix
    // But warp ctc doesn't support this, so here we have to allocated another
    // buf, and copy net_out to it
    float *acts_gpu;
    throw_on_error(cudaMalloc((void **)&acts_gpu, 
                        net_out.NumRows() * net_out.NumCols() * sizeof(float)),
                   "cudaMalloc");
    for (int i = 0; i < net_out.NumRows(); i++) {
        throw_on_error(cudaMemcpyAsync(acts_gpu + i * net_out.NumCols(),
                                       net_out.Data() + i * net_out.Stride(),
                                       net_out.NumCols() * sizeof(float),
                                       cudaMemcpyDeviceToDevice, stream),
                       "cudaMemcpyAsync");
    }

    // Allocate workspace
    size_t gpu_alloc_bytes;
    throw_on_error(get_workspace_size(label_lengths.data(),
                                      lengths.data(),
                                      alphabet_size,
                                      lengths.size(),
                                      info,
                                      &gpu_alloc_bytes),
                   "Error in get_workspace_size");

    char *ctc_gpu_workspace;
    throw_on_error(cudaMalloc((void **)&ctc_gpu_workspace, gpu_alloc_bytes),
                   "cudaMalloc");
    float *grads_gpu;
    throw_on_error(cudaMalloc((void **)&grads_gpu, 
                              diff->NumRows() * diff->NumCols() * sizeof(float)),
                   "cudaMalloc");

    // Compute ctc error
    throw_on_error(compute_ctc_loss(acts_gpu, grads_gpu,
                                    flat_labels.data(), 
                                    label_lengths.data(),
                                    lengths.data(),
                                    alphabet_size,
                                    minibatch,
                                    costs.data(),
                                    ctc_gpu_workspace,
                                    info),
                   "Error: compute_ctc_loss");

    // Copy grads_gpu to diff matrix
    //for (int i = 0; i < diff->NumRows(); i++) {
    //    throw_on_error(cudaMemcpyAsync(diff->Data() + i * diff->Stride(),
    //                                   grads_gpu + i * diff->NumCols(),
    //                                   diff->NumCols() * sizeof(float),
    //                                   cudaMemcpyDeviceToDevice, stream),
    //                   "cudaMemcpyAsync");
    //}
    //Matrix<BaseFloat> cpu_diff(diff->NumRows(), diff->NumCols());
    //diff->CopyToMat(&cpu_diff);

    throw_on_error(cudaStreamSynchronize(stream), 
                   "cudaStreamSynchronize");
    throw_on_error(cudaFree(ctc_gpu_workspace),
                   "cudaFree");
    throw_on_error(cudaFree(acts_gpu),
                   "cudaFree");
    throw_on_error(cudaFree(grads_gpu),
                   "cudaFree");
    throw_on_error(cudaStreamDestroy(stream),
                   "cudaStreamDestroy");
    // Statistic
    for (int i = 0; i < minibatch; i++) {
        obj_ += costs[i];
        obj_progress_ += costs[i];
        frames_progress_ += frame_num_utt[i];
        frames_ += frame_num_utt[i];
        //KALDI_LOG << costs[i];
    }
    sequences_progress_ += minibatch;
    sequences_num_ += minibatch;

    // Progress report
    {
        if (sequences_progress_ >= report_step_) {
            KALDI_LOG << "Progress " << sequences_num_ << " sequences (" << frames_/(100.0 * 3600) << "Hr):"
                << " Obj(log[Pzx]) = " << obj_progress_/sequences_progress_
                << " Obj(frame) = " << obj_progress_/frames_progress_
                << " TokenAcc = " << 100.0*(1.0 - error_num_progress_/ref_num_progress_) << " %";
            // reset
            sequences_progress_ = 0;
            frames_progress_ = 0;
            obj_progress_ = 0.0;
            error_num_progress_ = 0;
            ref_num_progress_ = 0;
        }
    }
}

void WarpCtc::EvalCpu(const std::vector<int32> &frame_num_utt, const CuMatrixBase<BaseFloat> &net_out,
        std::vector< std::vector<int32> > &labels, CuMatrix<BaseFloat> *diff) {
    diff->Resize(net_out.NumRows(), net_out.NumCols());

    // Prepare label, feat and their length
    const int minibatch = labels.size();
    std::vector<int> flat_labels;
    std::vector<int> label_lengths;
    for (int i = 0; i < labels.size(); i++) {
        std::vector<int> &l = labels[i];
        flat_labels.insert(flat_labels.end(), l.begin(), l.end());
        label_lengths.push_back(l.size());
    }
    int alphabet_size = net_out.NumCols();
    const std::vector<int> &lengths = frame_num_utt;
    std::vector<float> costs(minibatch);

    // Create ctc compute info
    ctcComputeInfo info;
    info.loc = CTC_CPU;
    info.num_threads = 1;

    // Allocate workspace
    size_t cpu_alloc_bytes;
    throw_on_error(get_workspace_size(label_lengths.data(),
                                      lengths.data(),
                                      alphabet_size,
                                      lengths.size(),
                                      info,
                                      &cpu_alloc_bytes),
                   "Error in get_workspace_size");

    void* ctc_cpu_workspace = malloc(cpu_alloc_bytes);
    // Compute ctc error
    throw_on_error(compute_ctc_loss(net_out.Data(), diff->Data(),
                                    flat_labels.data(), 
                                    label_lengths.data(),
                                    lengths.data(),
                                    alphabet_size,
                                    minibatch,
                                    costs.data(),
                                    ctc_cpu_workspace,
                                    info),
                   "Error: compute_ctc_loss");

    free(ctc_cpu_workspace);

    // Statistic
    for (int i = 0; i < minibatch; i++) {
        obj_ += costs[i];
        obj_progress_ += costs[i];
        frames_progress_ += frame_num_utt[i];
        frames_ += frame_num_utt[i];
        //KALDI_LOG << costs[i];
    }
    sequences_progress_ += minibatch;
    sequences_num_ += minibatch;

    // Progress report
    {
        if (sequences_progress_ >= report_step_) {
            KALDI_LOG << "Progress " << sequences_num_ << " sequences (" << frames_/(100.0 * 3600) << "Hr):"
                << " Obj(log[Pzx]) = " << obj_progress_/sequences_progress_
                << " Obj(frame) = " << obj_progress_/frames_progress_
                << " TokenAcc = " << 100.0*(1.0 - error_num_progress_/ref_num_progress_) << " %";
            // reset
            sequences_progress_ = 0;
            frames_progress_ = 0;
            obj_progress_ = 0.0;
            error_num_progress_ = 0;
            ref_num_progress_ = 0;
        }
    }
}

void WarpCtc::ErrorRate(const std::vector<int> &frame_num_utt, const CuMatrixBase<BaseFloat> &net_out, std::vector< std::vector<int> > &label) {

    // frame-level labels
    CuArray<int32> maxid(net_out.NumRows());
    net_out.FindRowMaxId(&maxid);

    int32 dim = maxid.Dim();
    std::vector<int32> data(dim);
    maxid.CopyToVec(&data);

    // compute errors sequence by sequence
    int32 num_seq = frame_num_utt.size();
    for (int32 s = 0; s < num_seq; s++) {
        int32 num_frame = frame_num_utt[s];
        std::vector<int32> raw_hyp_seq(num_frame);
        for (int32 f = 0; f < num_frame; f++) {
            raw_hyp_seq[f] = data[f*num_seq + s];
        }    
        int32 i = 1, j = 1;
        while(j < num_frame) {
            if (raw_hyp_seq[j] != raw_hyp_seq[j-1]) {
                raw_hyp_seq[i] = raw_hyp_seq[j];
                i++;
            }
            j++;
        }
        std::vector<int32> hyp_seq(0);
        for (int32 n = 0; n < i; n++) {
            if (raw_hyp_seq[n] != 0) {
                hyp_seq.push_back(raw_hyp_seq[n]);
            }
        }
        int32 err, ins, del, sub;
        err =  LevenshteinEditDistance(label[s], hyp_seq, &ins, &del, &sub);
        error_num_ += err;
        ref_num_ += label[s].size();
        error_num_progress_ += err;
        ref_num_progress_ += label[s].size();
    }
}

std::string WarpCtc::Report() {
    std::ostringstream oss;
    oss << " Obj(log[Pzx]) = " << obj_/sequences_num_
        << " Obj(frame) = " << obj_/frames_ 
        << " TOKEN_ACCURACY >> " << 100.0*(1.0 - error_num_/ref_num_) << " % <<";
    return oss.str(); 
}

} // namespace aslp_nnet
} // namespace kaldi
