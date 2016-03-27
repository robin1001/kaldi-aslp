// aslp-nnet/nnet-recurrent-component.h

// Copyright 2016 ASLP (Author: Binbin Zhang)
// Created on 2016-03-23

#ifndef ASLP_NNET_NNET_RECURRENT_COMPONETN_H_
#define ASLP_NNET_NNET_RECURRENT_COMPONETN_H_

#include "aslp-nnet/nnet-component.h"
#include "aslp-nnet/nnet-utils.h"
#include "aslp-cudamatrix/cu-math.h"

/*************************************
 * x: input neuron
 * g: squashing neuron near input
 * i: Input gate
 * f: Forget gate
 * o: Output gate
 * c: memory Cell (CEC)
 * h: squashing neuron near output
 * m: output neuron of Memory block
 * y: output neuron of LSTM
 *************************************/

namespace kaldi {
namespace aslp_nnet {

class Lstm : public UpdatableComponent {
public:
    Lstm(int32 input_dim, int32 output_dim) :
        UpdatableComponent(input_dim, output_dim),
        ncell_(output_dim),
        nstream_(0),
        clip_gradient_(0.0)
        //, dropout_rate_(0.0) 
        { }
    ~Lstm() { }
    Component* Copy() const { return new Lstm(*this); }
    ComponentType GetType() const { return kLstm; }

    static void InitMatParam(CuMatrix<BaseFloat> &m, float scale);
    static void InitVecParam(CuVector<BaseFloat> &v, float scale); 
    void InitData(std::istream &is); 
    void ReadData(std::istream &is, bool binary); 
    void WriteData(std::ostream &os, bool binary) const; 
    int32 NumParams() const; 
    void GetParams(Vector<BaseFloat>* wei_copy) const;
    void GetGpuParams(std::vector<std::pair<BaseFloat *, int> > *params);
    std::string Info() const; 
    std::string InfoGradient() const; 
    void ResetLstmStreams(const std::vector<int32> &stream_reset_flag); 
    // For compatible with whole sentence train(like ctc train or lstm who sentence train
    void SetSeqLengths(const std::vector<int32> &sequence_lengths); 
    void PropagateFnc(const CuMatrixBase<BaseFloat> &in, 
                      CuMatrixBase<BaseFloat> *out); 
    void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, 
                          const CuMatrixBase<BaseFloat> &out,
                          const CuMatrixBase<BaseFloat> &out_diff, 
                          CuMatrixBase<BaseFloat> *in_diff); 
    void Update(const CuMatrixBase<BaseFloat> &input, 
                const CuMatrixBase<BaseFloat> &diff); 

private:
    // dims
    int32 ncell_;
    int32 nstream_;

    CuMatrix<BaseFloat> prev_nnet_state_;

    // gradient-clipping value,
    BaseFloat clip_gradient_;

    // non-recurrent dropout
    //BaseFloat dropout_rate_;
    //CuMatrix<BaseFloat> dropout_mask_;

    // feed-forward connections: from x to [g, i, f, o]
    CuMatrix<BaseFloat> w_gifo_x_;
    CuMatrix<BaseFloat> w_gifo_x_corr_;

    // recurrent projection connections: from r to [g, i, f, o]
    CuMatrix<BaseFloat> w_gifo_r_;
    CuMatrix<BaseFloat> w_gifo_r_corr_;

    // biases of [g, i, f, o]
    CuVector<BaseFloat> bias_;
    CuVector<BaseFloat> bias_corr_;

    // peephole from c to i, f, g
    // peephole connections are block-internal, so we use vector form
    CuVector<BaseFloat> peephole_i_c_;
    CuVector<BaseFloat> peephole_f_c_;
    CuVector<BaseFloat> peephole_o_c_;

    CuVector<BaseFloat> peephole_i_c_corr_;
    CuVector<BaseFloat> peephole_f_c_corr_;
    CuVector<BaseFloat> peephole_o_c_corr_;

    // propagate buffer: output of [g, i, f, o, c, h, m]
    CuMatrix<BaseFloat> propagate_buf_;

    // back-propagate buffer: diff-input of [g, i, f, o, c, h, m]
    CuMatrix<BaseFloat> backpropagate_buf_;
};

class BLstm : public UpdatableComponent {
public:
    BLstm(int32 input_dim, int32 output_dim) :
        UpdatableComponent(input_dim, output_dim),
        ncell_(static_cast<int32>(output_dim/2)),
        nstream_(0),
        clip_gradient_(0.0)
        //, dropout_rate_(0.0)
    { }

    ~BLstm() { }

    Component* Copy() const { return new BLstm(*this); }
    ComponentType GetType() const { return kBLstm; }

    static void InitMatParam(CuMatrix<BaseFloat> &m, float scale);
    static void InitVecParam(CuVector<BaseFloat> &v, float scale); 
    /// set the utterance length used for parallel training
    void SetSeqLengths(const std::vector<int32> &sequence_lengths); 
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
    // dims
    int32 ncell_;   ///< the number of cell blocks
    int32 nstream_;
    std::vector<int32> sequence_lengths_;

    // gradient-clipping value,
    BaseFloat clip_gradient_;

    // non-recurrent dropout
    // BaseFloat dropout_rate_;
    // CuMatrix<BaseFloat> dropout_mask_;

    // feed-forward connections: from x to [g, i, f, o]
    // forward direction
    CuMatrix<BaseFloat> f_w_gifo_x_;
    CuMatrix<BaseFloat> f_w_gifo_x_corr_;
    // backward direction
    CuMatrix<BaseFloat> b_w_gifo_x_;
    CuMatrix<BaseFloat> b_w_gifo_x_corr_;

    // recurrent projection connections: from r to [g, i, f, o]
    // forward direction
    CuMatrix<BaseFloat> f_w_gifo_r_;
    CuMatrix<BaseFloat> f_w_gifo_r_corr_;
    // backward direction
    CuMatrix<BaseFloat> b_w_gifo_r_;
    CuMatrix<BaseFloat> b_w_gifo_r_corr_;

    // biases of [g, i, f, o]
    // forward direction
    CuVector<BaseFloat> f_bias_;
    CuVector<BaseFloat> f_bias_corr_;
    // backward direction
    CuVector<BaseFloat> b_bias_;
    CuVector<BaseFloat> b_bias_corr_;

    // peephole from c to i, f, g
    // peephole connections are block-internal, so we use vector form
    // forward direction
    CuVector<BaseFloat> f_peephole_i_c_;
    CuVector<BaseFloat> f_peephole_f_c_;
    CuVector<BaseFloat> f_peephole_o_c_;
    // backward direction
    CuVector<BaseFloat> b_peephole_i_c_;
    CuVector<BaseFloat> b_peephole_f_c_;
    CuVector<BaseFloat> b_peephole_o_c_;

    // forward direction
    CuVector<BaseFloat> f_peephole_i_c_corr_;
    CuVector<BaseFloat> f_peephole_f_c_corr_;
    CuVector<BaseFloat> f_peephole_o_c_corr_;
    // backward direction
    CuVector<BaseFloat> b_peephole_i_c_corr_;
    CuVector<BaseFloat> b_peephole_f_c_corr_;
    CuVector<BaseFloat> b_peephole_o_c_corr_;

    // propagate buffer: output of [g, i, f, o, c, h, m]
    // forward direction
    CuMatrix<BaseFloat> f_propagate_buf_;
    // backward direction
    CuMatrix<BaseFloat> b_propagate_buf_;


    // back-propagate buffer: diff-input of [g, i, f, o, c, h, m]
    // forward direction
    CuMatrix<BaseFloat> f_backpropagate_buf_;
    // backward direction
    CuMatrix<BaseFloat> b_backpropagate_buf_;

};

} // namespace aslp_nnet
} // namespace kaldi

#endif
