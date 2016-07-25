// aslp-nnet/nnet-row-convolution.cc

// Copyright 2016 ASLP (Author: Zhang Binbin)
// Created on 2016-07-13


#include "aslp-nnet/nnet-row-convolution.h"


namespace kaldi {
namespace aslp_nnet {

void RowConvolution::InitData(std::istream &is) {
    std::string token;
    while (!is.eof()) {
        ReadToken(is, false, &token);
        if (token == "<FutureContext>")
            ReadBasicType(is, false, &future_ctx_);
        else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
            << " (FutureContext)";
        is >> std::ws;
    }
    KALDI_ASSERT(future_ctx_ > 0);

    // initialize
    Matrix<BaseFloat> mat(input_dim_, future_ctx_ + 1);
    for (int r = 0; r < mat.NumRows(); r++) {
        for (int c = 0; c < mat.NumCols(); c++) {
            // init in gauss
            mat(r, c) = 1.0 * RandGauss();
            // init in uniform
            //mat(r, c) = (RandUniform() - 0.5) * 2 * 1.0;
        }
    }

    w_ = mat;
    w_diff_.Resize(input_dim_, future_ctx_ + 1, kSetZero);
    w_corr_.Resize(input_dim_, future_ctx_ + 1, kSetZero);
    // init buffer
    conv_buf_.Resize(input_dim_, input_dim_, kSetZero);
    conv_diff_buf_.Resize(input_dim_, future_ctx_ + 1, kSetZero);
}

void RowConvolution::ReadData(std::istream &is, bool binary) {
    ExpectToken(is, binary, "<FutureContext>");
    ReadBasicType(is, binary, &future_ctx_);
    w_.Read(is, binary);
    
    w_diff_.Resize(input_dim_, future_ctx_ + 1, kSetZero);
    w_corr_.Resize(input_dim_, future_ctx_ + 1, kSetZero);
    // init buffer
    conv_buf_.Resize(input_dim_, input_dim_, kSetZero);
    conv_diff_buf_.Resize(input_dim_, future_ctx_ + 1, kSetZero);
}

void RowConvolution::WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<FutureContext>");
    WriteBasicType(os, binary, future_ctx_);
    w_.Write(os, binary);
}

int32 RowConvolution::NumParams() const {
    return w_.NumRows() * w_.NumCols();
}

void RowConvolution::GetParams(Vector<BaseFloat> *wei_copy) const {
    int num_params = NumParams();
    wei_copy->Resize(num_params);
    wei_copy->Range(0, num_params).CopyRowsFromMat(w_);
}

void RowConvolution::GetGpuParams(std::vector<std::pair<BaseFloat *, int> > *params) {
    params->clear();
    params->push_back(std::make_pair(w_.Data(), w_.NumRows() * w_.NumCols()));
}

std::string RowConvolution::Info() const {
    return std::string("  ")  +
      "\n  w_ "     + MomentStatistics(w_);
}


std::string RowConvolution::InfoGradient() const {
    return std::string("  ") + 
    "\n w_diff_ " + MomentStatistics(w_diff_) + 
    "\n w_corr_ " + MomentStatistics(w_corr_) + 
    "\n in_diff_buf_ " + MomentStatistics(in_diff_buf_);
}

void RowConvolution::PropagateFnc(const CuMatrixBase<BaseFloat> &in, 
                                  CuMatrixBase<BaseFloat> *out) {
    int32 nstream_ = sequence_lengths_.size();
    KALDI_ASSERT(in.NumRows() % nstream_ == 0);
    int32 T = in.NumRows() / nstream_;
    int32 S = nstream_;    
   
    int32 Ts = T + future_ctx_;
    in_buf_.Resize(Ts * S, in.NumCols(), kSetZero);

    for (int s = 0; s < S; s++) {
        // Copy sequence s to sequence buffer
        for (int t = 0; t < sequence_lengths_[s] + future_ctx_; t++) {
            if (t < sequence_lengths_[s]) {
                in_buf_.Row(s * Ts + t).CopyFromVec(in.Row(t * S + s));
            } else { // just copy the last frame
                in_buf_.Row(s * Ts + t).CopyFromVec(
                    in.Row((sequence_lengths_[s] - 1) * S + s));
            }
        }
        // Do row convolution
        for (int t = 0; t < sequence_lengths_[s]; t++) {
            CuSubMatrix<BaseFloat> yh(in_buf_.RowRange(s * Ts + t, 
                future_ctx_ + 1));
            conv_buf_.AddMatMat(1.0, w_, kNoTrans, yh, kNoTrans, 0.0);
            out->Row(t * S + s).CopyDiagFromMat(conv_buf_);
        }
    }
}

void RowConvolution::BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, 
                                      const CuMatrixBase<BaseFloat> &out,
                                      const CuMatrixBase<BaseFloat> &out_diff, 
                                      CuMatrixBase<BaseFloat> *in_diff) {
    int32 nstream_ = sequence_lengths_.size();
    KALDI_ASSERT(in.NumRows() % nstream_ == 0);
    int32 T = in.NumRows() / nstream_;
    int32 S = nstream_;
   
    int32 Ts = T + future_ctx_;
    in_diff_buf_.Resize(Ts * S, in.NumCols(), kSetZero);
    w_diff_.SetZero();

    // Cacl diff
    for (int s = 0; s < S; s++) {
        for (int t = 0; t < sequence_lengths_[s]; t++) {
            CuSubMatrix<BaseFloat> yh(in_buf_.RowRange(s * Ts + t, 
                future_ctx_ + 1));
            CuSubMatrix<BaseFloat> yh_diff(in_diff_buf_.RowRange(
                s * Ts + t, future_ctx_ + 1));
            // Acc in diff 
            conv_diff_buf_.SetZero();
            conv_diff_buf_.AddMat(1.0, w_);
            conv_diff_buf_.MulRowsVec(out_diff.Row(t * S + s));
            yh_diff.AddMat(1.0, conv_diff_buf_, kTrans);
            // Acc w diff
            conv_diff_buf_.SetZero();
            conv_diff_buf_.AddMat(1.0, yh, kTrans);
            conv_diff_buf_.MulRowsVec(out_diff.Row(t * S + s));
            w_diff_.AddMat(1.0, conv_diff_buf_);
        }
    }
    // Copy to in_diff
    for (int s = 0; s < S; s++) {
        for (int t = 0; t < sequence_lengths_[s]; t++) {
            in_diff->Row(t * S + s).CopyFromVec(
                    in_diff_buf_.Row(s * Ts + t));
        }
    }
}

void RowConvolution::Update(const CuMatrixBase<BaseFloat> &input, 
                            const CuMatrixBase<BaseFloat> &diff) {
    const BaseFloat mmt = opts_.momentum;
    w_corr_.Scale(mmt);
    w_corr_.AddMat(1.0, w_diff_);

    const BaseFloat lr  = opts_.learn_rate;
    w_.AddMat(-lr, w_corr_);
}


} // namespace aslp_nnet
} // namespace kaldi
