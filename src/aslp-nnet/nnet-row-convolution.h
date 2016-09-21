// aslp-nnet/nnet-row-convolution.h

// Copyright 2016 ASLP (Author: Zhang Binbin)
// Created on 2016-07-13

#ifndef ASLP_NNET_NNET_ROW_CONVOLUTION_H_
#define ASLP_NNET_NNET_ROW_CONVOLUTION_H_

#include "aslp-nnet/nnet-component.h"
#include "aslp-nnet/nnet-utils.h"
#include "aslp-cudamatrix/cu-math.h"


namespace kaldi {
namespace aslp_nnet {

// Row Convolution is proposed in Deep Speech 2, it use limited 
// future state information and get comparable result to blstm.
// RowConvolution layer is added upon the last recurrent layer
// provide the current frame with a few future information.

class RowConvolution: public UpdatableComponent {
public:
    RowConvolution(int32 input_dim, int32 output_dim) :
        UpdatableComponent(input_dim, output_dim),
        nstream_(0), future_ctx_(0) { 
        if (input_dim_ != output_dim_) {
            KALDI_ERR << "RowConvolution layer input dim and output dim"
                         "must be equal";
        }
    }

    ~RowConvolution() { }

    Component* Copy() const { return new RowConvolution(*this); }
    ComponentType GetType() const { return kRowConvolution; }

    /// set the utterance length used for parallel training
    void SetSeqLengths(const std::vector<int32> &sequence_lengths) {
        sequence_lengths_ = sequence_lengths;
    }
    void InitData(std::istream &is); 

    void ReadData(std::istream &is, bool binary); 
    void WriteData(std::ostream &os, bool binary) const; 
    int32 NumParams() const; 

    void GetParams(Vector<BaseFloat>* wei_copy) const; 
    void GetGpuParams(std::vector<std::pair<BaseFloat *, int> > *params); 
    std::string Info() const; 
    std::string InfoGradient() const; 

    void PropagateFnc(const CuMatrixBase<BaseFloat> &in, 
                      CuMatrixBase<BaseFloat> *out); 
    void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, 
                          const CuMatrixBase<BaseFloat> &out,
                          const CuMatrixBase<BaseFloat> &out_diff, 
                          CuMatrixBase<BaseFloat> *in_diff); 

    void Update(const CuMatrixBase<BaseFloat> &input, 
                const CuMatrixBase<BaseFloat> &diff); 
private:
    std::vector<int32> sequence_lengths_;
    int32 nstream_;
    int future_ctx_;
    CuMatrix<BaseFloat> w_, w_diff_, w_corr_;
    CuMatrix<BaseFloat> in_buf_, in_diff_buf_; // reorder in 
    CuMatrix<BaseFloat> conv_buf_, conv_diff_buf_;
};



} // namespace aslp_nnet
} // namespace kaldi

#endif
