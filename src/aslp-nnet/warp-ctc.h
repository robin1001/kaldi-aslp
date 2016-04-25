// aslp-nnet/warp-ctc.h

// Copyright 2016  ASLP (author: Binbin Zhang)
// Created on 2016-04-12

#ifndef ASLP_NNET_WARP_CTC_H_
#define ASLP_NNET_WARP_CTC_H_

#include "base/kaldi-common.h"
#include "util/kaldi-holder.h"
#include "aslp-cudamatrix/cu-matrix.h"
#include "aslp-cudamatrix/cu-vector.h"
#include "aslp-cudamatrix/cu-array.h"

#include "warp-ctc/include/ctc.h"

namespace kaldi {
namespace aslp_nnet {

#define WARP_CTC_SUM_LOSS_CHECK 0
#define WARP_CTC_AVG_LOSS_CHECK 1
#define WARP_CTC_NONE_LOSS_CHECK 2

#define WARP_CTC_GRAD_CHECK 1

class WarpCtc {
public:
    WarpCtc() : frames_(0), sequences_num_(0), ref_num_(0), error_num_(0), 
    frames_progress_(0), ref_num_progress_(0), error_num_progress_(0),
    sequences_progress_(0), obj_progress_(0.0), report_step_(100),
    obj_(0), use_gpu_(true),
    loss_sum_(0), loss_square_sum_(0),
    loss_sum_bak_(0), loss_square_sum_bak_(0), 
    normal_num_(0), stat_period_(100) { }

    ~WarpCtc() { }

    /// CTC training over multiple sequences. The errors are returned to [diff]
    void Eval(const std::vector<std::string> &utt,
              const std::vector<int32> &frame_num_utt, 
              const CuMatrixBase<BaseFloat> &net_out,
              const std::vector< std::vector<int32> > &labels, 
              CuMatrix<BaseFloat> *diff);

    void EvalGpu(const std::vector<std::string> &utt,
              const std::vector<int32> &frame_num_utt, 
              const CuMatrixBase<BaseFloat> &net_out,
              const std::vector< std::vector<int32> > &labels, 
              CuMatrix<BaseFloat> *diff);

    void EvalCpu(const std::vector<std::string> &utt,
              const std::vector<int32> &frame_num_utt, 
              const CuMatrixBase<BaseFloat> &net_out,
              const std::vector< std::vector<int32> > &labels, 
              CuMatrix<BaseFloat> *diff);

    /// Compute token error rate from the softmax-layer activations and the given labels. From the softmax activations,
    /// we get the frame-level labels, by selecting the label with the largest probability at each frame. Then, the frame
    /// -level labels are shrunk by removing the blanks and collasping the repetitions. This gives us the utterance-level
    /// labels, from which we can compute the error rate. The error rate is the Levenshtein distance between the hyp labels
    /// and the given reference label sequence.
    void ErrorRate(const std::vector<int> &frame_num_utt, const CuMatrixBase<BaseFloat> &net_out, std::vector< std::vector<int> > &label);

    /// Set the step of reporting
    void SetReportStep(int32 report_step) { report_step_ = report_step;  }

    /// Generate string with report
    std::string Report();

    float NumErrorTokens() const { return error_num_;}
    int32 NumRefTokens() const { return ref_num_;}
    void SetUseGpu(bool use_gpu) {
        use_gpu_ = use_gpu;
    }
    /// Grad check family function
    void StatOnly(const std::vector<std::string> &utt, 
                  const std::vector<int32> &frame_num_utt, 
                  const std::vector<float> &pzx_host,
                  CuMatrix<BaseFloat> *diff);

    void StatAndLossCheck(const std::vector<std::string> &utt, 
                          const std::vector<int32> &frame_num_utt, 
                          const std::vector<float> &pzx_host,
                          CuMatrix<BaseFloat> *diff);

    void StatAndAverageLossCheck(const std::vector<std::string> &utt, 
                                 const std::vector<int32> &frame_num_utt, 
                                 const std::vector<float> &pzx_host,
                                 CuMatrix<BaseFloat> *diff);

private:
    int32 frames_;                    // total frame number
    int32 sequences_num_; 
    int32 ref_num_;                   // total number of tokens in label sequences
    float error_num_;                 // total number of errors (edit distance between hyp and ref)

    int32 frames_progress_;
    int32 ref_num_progress_;
    float error_num_progress_;

    int32 sequences_progress_;         // registry for the number of sequences
    double obj_progress_;              // registry for the optimization objective

    int32 report_step_;                // report obj and accuracy every so many sequences/utterances

    double obj_;
    bool use_gpu_;
    // For statistic grad
    double loss_sum_, loss_square_sum_;
    double loss_sum_bak_, loss_square_sum_bak_;
    int32 normal_num_, stat_period_;
};

} // namespace aslp_nnet
} // namespace kaldi

#endif
