// aslp-nnet/nnet-recurrent-component.h

// Copyright 2016 ASLP (Author: Binbin Zhang)
// Created on 2016-03-23

#include "aslp-nnet/nnet-recurrent-component.h"


namespace kaldi {
namespace aslp_nnet {

void Lstm::InitMatParam(CuMatrix<BaseFloat> &m, float scale) {
    m.SetRandUniform();  // uniform in [0, 1]
    m.Add(-0.5);         // uniform in [-0.5, 0.5]
    m.Scale(2 * scale);  // uniform in [-scale, +scale]
}

void Lstm::InitVecParam(CuVector<BaseFloat> &v, float scale) {
    Vector<BaseFloat> tmp(v.Dim());
    for (int i=0; i < tmp.Dim(); i++) {
        tmp(i) = (RandUniform() - 0.5) * 2 * scale;
    }
    v = tmp;
}

void Lstm::InitData(std::istream &is) {
    // define options
    float param_scale = 0.02;
    // parse config
    std::string token;
    while (!is.eof()) {
        ReadToken(is, false, &token);
        if (token == "<ClipGradient>")
            ReadBasicType(is, false, &clip_gradient_);
        //else if (token == "<DropoutRate>")
        //  ReadBasicType(is, false, &dropout_rate_);
        else if (token == "<ParamScale>")
            ReadBasicType(is, false, &param_scale);
        else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
            << " (ClipGradient|ParamScale)";
        //<< " (CellDim|ClipGradient|DropoutRate|ParamScale)";
        is >> std::ws;
    }

    // init weight and bias (Uniform)
    w_gifo_x_.Resize(4*ncell_, input_dim_, kUndefined);
    w_gifo_r_.Resize(4*ncell_, ncell_, kUndefined);

    InitMatParam(w_gifo_x_, param_scale);
    InitMatParam(w_gifo_r_, param_scale);

    bias_.Resize(4*ncell_, kUndefined);
    peephole_i_c_.Resize(ncell_, kUndefined);
    peephole_f_c_.Resize(ncell_, kUndefined);
    peephole_o_c_.Resize(ncell_, kUndefined);

    InitVecParam(bias_, param_scale);
    InitVecParam(peephole_i_c_, param_scale);
    InitVecParam(peephole_f_c_, param_scale);
    InitVecParam(peephole_o_c_, param_scale);

    // init delta buffers
    w_gifo_x_corr_.Resize(4*ncell_, input_dim_, kSetZero);
    w_gifo_r_corr_.Resize(4*ncell_, ncell_, kSetZero);
    bias_corr_.Resize(4*ncell_, kSetZero);

    peephole_i_c_corr_.Resize(ncell_, kSetZero);
    peephole_f_c_corr_.Resize(ncell_, kSetZero);
    peephole_o_c_corr_.Resize(ncell_, kSetZero);

    KALDI_ASSERT(clip_gradient_ >= 0.0);
}

void Lstm::ReadData(std::istream &is, bool binary) {
    ExpectToken(is, binary, "<ClipGradient>");
    ReadBasicType(is, binary, &clip_gradient_);
    //ExpectToken(is, binary, "<DropoutRate>");
    //ReadBasicType(is, binary, &dropout_rate_);

    w_gifo_x_.Read(is, binary);
    w_gifo_r_.Read(is, binary);
    bias_.Read(is, binary);

    peephole_i_c_.Read(is, binary);
    peephole_f_c_.Read(is, binary);
    peephole_o_c_.Read(is, binary);

    // init delta buffers
    w_gifo_x_corr_.Resize(4*ncell_, input_dim_, kSetZero);
    w_gifo_r_corr_.Resize(4*ncell_, ncell_, kSetZero);
    bias_corr_.Resize(4*ncell_, kSetZero);

    peephole_i_c_corr_.Resize(ncell_, kSetZero);
    peephole_f_c_corr_.Resize(ncell_, kSetZero);
    peephole_o_c_corr_.Resize(ncell_, kSetZero);
}

void Lstm::WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<ClipGradient>");
    WriteBasicType(os, binary, clip_gradient_);
    //WriteToken(os, binary, "<DropoutRate>");
    //WriteBasicType(os, binary, dropout_rate_);

    w_gifo_x_.Write(os, binary);
    w_gifo_r_.Write(os, binary);
    bias_.Write(os, binary);

    peephole_i_c_.Write(os, binary);
    peephole_f_c_.Write(os, binary);
    peephole_o_c_.Write(os, binary);
}

int32 Lstm::NumParams() const {
    return ( w_gifo_x_.NumRows() * w_gifo_x_.NumCols() +
            w_gifo_r_.NumRows() * w_gifo_r_.NumCols() +
            bias_.Dim() +
            peephole_i_c_.Dim() +
            peephole_f_c_.Dim() +
            peephole_o_c_.Dim() );
}

void Lstm::GetParams(Vector<BaseFloat>* wei_copy) const {
    wei_copy->Resize(NumParams());

    int32 offset, len;

    offset = 0;  len = w_gifo_x_.NumRows() * w_gifo_x_.NumCols();
    wei_copy->Range(offset, len).CopyRowsFromMat(w_gifo_x_);

    offset += len; len = w_gifo_r_.NumRows() * w_gifo_r_.NumCols();
    wei_copy->Range(offset, len).CopyRowsFromMat(w_gifo_r_);

    offset += len; len = bias_.Dim();
    wei_copy->Range(offset, len).CopyFromVec(bias_);

    offset += len; len = peephole_i_c_.Dim();
    wei_copy->Range(offset, len).CopyFromVec(peephole_i_c_);

    offset += len; len = peephole_f_c_.Dim();
    wei_copy->Range(offset, len).CopyFromVec(peephole_f_c_);

    offset += len; len = peephole_o_c_.Dim();
    wei_copy->Range(offset, len).CopyFromVec(peephole_o_c_);

    return;
}

void Lstm::GetGpuParams(std::vector<std::pair<BaseFloat *, int> > *params) {
    params->clear();
    params->push_back(std::make_pair(w_gifo_x_.Data(), w_gifo_x_.NumRows() * w_gifo_x_.Stride()));
    params->push_back(std::make_pair(w_gifo_r_.Data(), w_gifo_r_.NumRows() * w_gifo_r_.Stride()));
    params->push_back(std::make_pair(bias_.Data(), bias_.Dim()));
    params->push_back(std::make_pair(peephole_i_c_.Data(), peephole_i_c_.Dim()));
    params->push_back(std::make_pair(peephole_f_c_.Data(), peephole_f_c_.Dim()));
    params->push_back(std::make_pair(peephole_o_c_.Data(), peephole_o_c_.Dim()));
}

std::string Lstm::Info() const {
    return std::string("  ") +
        "\n  w_gifo_x_  "   + MomentStatistics(w_gifo_x_) +
        "\n  w_gifo_r_  "   + MomentStatistics(w_gifo_r_) +
        "\n  bias_  "     + MomentStatistics(bias_) +
        "\n  peephole_i_c_  " + MomentStatistics(peephole_i_c_) +
        "\n  peephole_f_c_  " + MomentStatistics(peephole_f_c_) +
        "\n  peephole_o_c_  " + MomentStatistics(peephole_o_c_); 
}

std::string Lstm::InfoGradient() const {
    // disassemble forward-propagation buffer into different neurons,
    const CuSubMatrix<BaseFloat> YG(propagate_buf_.ColRange(0*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> YI(propagate_buf_.ColRange(1*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> YF(propagate_buf_.ColRange(2*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> YO(propagate_buf_.ColRange(3*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> YC(propagate_buf_.ColRange(4*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> YH(propagate_buf_.ColRange(5*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> YM(propagate_buf_.ColRange(6*ncell_, ncell_));

    // disassemble backpropagate buffer into different neurons,
    const CuSubMatrix<BaseFloat> DG(backpropagate_buf_.ColRange(0*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> DI(backpropagate_buf_.ColRange(1*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> DF(backpropagate_buf_.ColRange(2*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> DO(backpropagate_buf_.ColRange(3*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> DC(backpropagate_buf_.ColRange(4*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> DH(backpropagate_buf_.ColRange(5*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> DM(backpropagate_buf_.ColRange(6*ncell_, ncell_));

    return std::string("  ") +
        "\n  Gradients:" +
        "\n  w_gifo_x_corr_  "   + MomentStatistics(w_gifo_x_corr_) +
        "\n  w_gifo_r_corr_  "   + MomentStatistics(w_gifo_r_corr_) +
        "\n  bias_corr_  "     + MomentStatistics(bias_corr_) +
        "\n  peephole_i_c_corr_  " + MomentStatistics(peephole_i_c_corr_) +
        "\n  peephole_f_c_corr_  " + MomentStatistics(peephole_f_c_corr_) +
        "\n  peephole_o_c_corr_  " + MomentStatistics(peephole_o_c_corr_) +
        "\n  Forward-pass:" +
        "\n  YG  " + MomentStatistics(YG) +
        "\n  YI  " + MomentStatistics(YI) +
        "\n  YF  " + MomentStatistics(YF) +
        "\n  YC  " + MomentStatistics(YC) +
        "\n  YH  " + MomentStatistics(YH) +
        "\n  YO  " + MomentStatistics(YO) +
        "\n  YM  " + MomentStatistics(YM) +
        "\n  Backward-pass:" +
        "\n  DG  " + MomentStatistics(DG) +
        "\n  DI  " + MomentStatistics(DI) +
        "\n  DF  " + MomentStatistics(DF) +
        "\n  DC  " + MomentStatistics(DC) +
        "\n  DH  " + MomentStatistics(DH) +
        "\n  DO  " + MomentStatistics(DO) +
        "\n  DM  " + MomentStatistics(DM);
}

void Lstm::ResetLstmStreams(const std::vector<int32> &stream_reset_flag) {
    // allocate prev_nnet_state_ if not done yet,
    if (nstream_ == 0) {
        // Karel: we just got number of streams! (before the 1st batch comes)
        nstream_ = stream_reset_flag.size();
        prev_nnet_state_.Resize(nstream_, 7*ncell_, kSetZero);
        KALDI_LOG << "Running training with " << nstream_ << " streams.";
    }
    // reset flag: 1 - reset stream network state
    KALDI_ASSERT(prev_nnet_state_.NumRows() == stream_reset_flag.size());
    for (int s = 0; s < stream_reset_flag.size(); s++) {
        if (stream_reset_flag[s] == 1) {
            prev_nnet_state_.Row(s).SetZero();
        }
    }
}

void Lstm::PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
    static bool do_stream_reset = false;
    if (nstream_ == 0) {
        do_stream_reset = true;
        nstream_ = 1; // Karel: we are in nnet-forward, so 1 stream,
        prev_nnet_state_.Resize(nstream_, 7*ncell_, kSetZero);
        KALDI_LOG << "Running nnet-forward with per-utterance LSTM-state reset";
    }
    if (do_stream_reset) prev_nnet_state_.SetZero();
    KALDI_ASSERT(nstream_ > 0);

    KALDI_ASSERT(in.NumRows() % nstream_ == 0);
    int32 T = in.NumRows() / nstream_;
    int32 S = nstream_;

    // 0:forward pass history, [1, T]:current sequence, T+1:dummy
    propagate_buf_.Resize((T+2)*S, 7 * ncell_, kSetZero);
    propagate_buf_.RowRange(0*S,S).CopyFromMat(prev_nnet_state_);

    // disassemble entire neuron activation buffer into different neurons
    CuSubMatrix<BaseFloat> YG(propagate_buf_.ColRange(0*ncell_, ncell_));
    CuSubMatrix<BaseFloat> YI(propagate_buf_.ColRange(1*ncell_, ncell_));
    CuSubMatrix<BaseFloat> YF(propagate_buf_.ColRange(2*ncell_, ncell_));
    CuSubMatrix<BaseFloat> YO(propagate_buf_.ColRange(3*ncell_, ncell_));
    CuSubMatrix<BaseFloat> YC(propagate_buf_.ColRange(4*ncell_, ncell_));
    CuSubMatrix<BaseFloat> YH(propagate_buf_.ColRange(5*ncell_, ncell_));
    CuSubMatrix<BaseFloat> YM(propagate_buf_.ColRange(6*ncell_, ncell_));

    CuSubMatrix<BaseFloat> YGIFO(propagate_buf_.ColRange(0, 4*ncell_));

    // x -> g, i, f, o, not recurrent, do it all in once
    YGIFO.RowRange(1*S,T*S).AddMatMat(1.0, in, kNoTrans, w_gifo_x_, kTrans, 0.0);
    //// LSTM forward dropout
    //// Google paper 2014: Recurrent Neural Network Regularization
    //// by Wojciech Zaremba, Ilya Sutskever, Oriol Vinyals
    //if (dropout_rate_ != 0.0) {
    //  dropout_mask_.Resize(in.NumRows(), 4*ncell_, kUndefined);
    //  dropout_mask_.SetRandUniform();   // [0,1]
    //  dropout_mask_.Add(-dropout_rate_);  // [-dropout_rate, 1-dropout_rate_],
    //  dropout_mask_.ApplyHeaviside();   // -tive -> 0.0, +tive -> 1.0
    //  YGIFO.RowRange(1*S,T*S).MulElements(dropout_mask_);
    //}

    // bias -> g, i, f, o
    YGIFO.RowRange(1*S,T*S).AddVecToRows(1.0, bias_);

    for (int t = 1; t <= T; t++) {
        // multistream buffers for current time-step
        CuSubMatrix<BaseFloat> y_g(YG.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> y_i(YI.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> y_f(YF.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> y_o(YO.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> y_c(YC.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> y_h(YH.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> y_m(YM.RowRange(t*S,S));

        CuSubMatrix<BaseFloat> y_gifo(YGIFO.RowRange(t*S,S));

        // r(t-1) -> g, i, f, o
        y_gifo.AddMatMat(1.0, YM.RowRange((t-1)*S,S), kNoTrans, w_gifo_r_, kTrans,  1.0);

        // c(t-1) -> i(t) via peephole
        y_i.AddMatDiagVec(1.0, YC.RowRange((t-1)*S,S), kNoTrans, peephole_i_c_, 1.0);

        // c(t-1) -> f(t) via peephole
        y_f.AddMatDiagVec(1.0, YC.RowRange((t-1)*S,S), kNoTrans, peephole_f_c_, 1.0);

        // i, f sigmoid squashing
        y_i.Sigmoid(y_i);
        y_f.Sigmoid(y_f);

        // g tanh squashing
        y_g.Tanh(y_g);

        // g -> c
        y_c.AddMatMatElements(1.0, y_g, y_i, 0.0);

        // c(t-1) -> c(t) via forget-gate
        y_c.AddMatMatElements(1.0, YC.RowRange((t-1)*S,S), y_f, 1.0);

        y_c.ApplyFloor(-50);   // optional clipping of cell activation
        y_c.ApplyCeiling(50);  // google paper Interspeech2014: LSTM for LVCSR

        // h tanh squashing
        y_h.Tanh(y_c);

        // c(t) -> o(t) via peephole (non-recurrent) & o squashing
        y_o.AddMatDiagVec(1.0, y_c, kNoTrans, peephole_o_c_, 1.0);

        // o sigmoid squashing
        y_o.Sigmoid(y_o);

        // h -> m via output gate
        y_m.AddMatMatElements(1.0, y_h, y_o, 0.0);
    }

    out->CopyFromMat(YM.RowRange(1*S,T*S));

    // now the last frame state becomes previous network state for next batch
    prev_nnet_state_.CopyFromMat(propagate_buf_.RowRange(T*S,S));
}

void Lstm::BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
        const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
        
    int32 T = in.NumRows() / nstream_;
    int32 S = nstream_;

    // disassemble propagated buffer into neurons
    CuSubMatrix<BaseFloat> YG(propagate_buf_.ColRange(0*ncell_, ncell_));
    CuSubMatrix<BaseFloat> YI(propagate_buf_.ColRange(1*ncell_, ncell_));
    CuSubMatrix<BaseFloat> YF(propagate_buf_.ColRange(2*ncell_, ncell_));
    CuSubMatrix<BaseFloat> YO(propagate_buf_.ColRange(3*ncell_, ncell_));
    CuSubMatrix<BaseFloat> YC(propagate_buf_.ColRange(4*ncell_, ncell_));
    CuSubMatrix<BaseFloat> YH(propagate_buf_.ColRange(5*ncell_, ncell_));
    CuSubMatrix<BaseFloat> YM(propagate_buf_.ColRange(6*ncell_, ncell_));

    // 0:dummy, [1,T] frames, T+1 backward pass history
    backpropagate_buf_.Resize((T+2)*S, 7 * ncell_, kSetZero);

    // disassemble backpropagate buffer into neurons
    CuSubMatrix<BaseFloat> DG(backpropagate_buf_.ColRange(0*ncell_, ncell_));
    CuSubMatrix<BaseFloat> DI(backpropagate_buf_.ColRange(1*ncell_, ncell_));
    CuSubMatrix<BaseFloat> DF(backpropagate_buf_.ColRange(2*ncell_, ncell_));
    CuSubMatrix<BaseFloat> DO(backpropagate_buf_.ColRange(3*ncell_, ncell_));
    CuSubMatrix<BaseFloat> DC(backpropagate_buf_.ColRange(4*ncell_, ncell_));
    CuSubMatrix<BaseFloat> DH(backpropagate_buf_.ColRange(5*ncell_, ncell_));
    CuSubMatrix<BaseFloat> DM(backpropagate_buf_.ColRange(6*ncell_, ncell_));

    CuSubMatrix<BaseFloat> DGIFO(backpropagate_buf_.ColRange(0, 4*ncell_));

    // projection layer to LSTM output is not recurrent, so backprop it all in once
    DM.RowRange(1*S,T*S).CopyFromMat(out_diff);

    for (int t = T; t >= 1; t--) {
        CuSubMatrix<BaseFloat> y_g(YG.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> y_i(YI.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> y_f(YF.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> y_o(YO.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> y_c(YC.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> y_h(YH.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> y_m(YM.RowRange(t*S,S));

        CuSubMatrix<BaseFloat> d_g(DG.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> d_i(DI.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> d_f(DF.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> d_o(DO.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> d_c(DC.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> d_h(DH.RowRange(t*S,S));
        CuSubMatrix<BaseFloat> d_m(DM.RowRange(t*S,S));

        // r
        //   Version 1 (precise gradients):
        //   backprop error from g(t+1), i(t+1), f(t+1), o(t+1) to r(t)
        d_m.AddMatMat(1.0, DGIFO.RowRange((t+1)*S,S), kNoTrans, w_gifo_r_, kNoTrans, 1.0);

        /*
        //   Version 2 (Alex Graves' PhD dissertation):
        //   only backprop g(t+1) to r(t)
        CuSubMatrix<BaseFloat> w_g_r_(w_gifo_r_.RowRange(0, ncell_));
        d_m.AddMatMat(1.0, DG.RowRange((t+1)*S,S), kNoTrans, w_g_r_, kNoTrans, 1.0);
         */

        /*
        //   Version 3 (Felix Gers' PhD dissertation):
        //   truncate gradients of g(t+1), i(t+1), f(t+1), o(t+1) once they leak out memory block
        //   CEC(with forget connection) is the only "error-bridge" through time
         */
        // m -> h via output gate
        d_h.AddMatMatElements(1.0, d_m, y_o, 0.0);
        d_h.DiffTanh(y_h, d_h);

        // o
        d_o.AddMatMatElements(1.0, d_m, y_h, 0.0);
        d_o.DiffSigmoid(y_o, d_o);

        // c
        // 1. diff from h(t)
        // 2. diff from c(t+1) (via forget-gate between CEC)
        // 3. diff from i(t+1) (via peephole)
        // 4. diff from f(t+1) (via peephole)
        // 5. diff from o(t)   (via peephole, not recurrent)
        d_c.AddMat(1.0, d_h);
        d_c.AddMatMatElements(1.0, DC.RowRange((t+1)*S,S), YF.RowRange((t+1)*S,S), 1.0);
        d_c.AddMatDiagVec(1.0, DI.RowRange((t+1)*S,S), kNoTrans, peephole_i_c_, 1.0);
        d_c.AddMatDiagVec(1.0, DF.RowRange((t+1)*S,S), kNoTrans, peephole_f_c_, 1.0);
        d_c.AddMatDiagVec(1.0, d_o                   , kNoTrans, peephole_o_c_, 1.0);

        // f
        d_f.AddMatMatElements(1.0, d_c, YC.RowRange((t-1)*S,S), 0.0);
        d_f.DiffSigmoid(y_f, d_f);

        // i
        d_i.AddMatMatElements(1.0, d_c, y_g, 0.0);
        d_i.DiffSigmoid(y_i, d_i);

        // c -> g via input gate
        d_g.AddMatMatElements(1.0, d_c, y_i, 0.0);
        d_g.DiffTanh(y_g, d_g);
    }

    // g,i,f,o -> x, do it all in once
    in_diff->AddMatMat(1.0, DGIFO.RowRange(1*S,T*S), kNoTrans, w_gifo_x_, kNoTrans, 0.0);

    //// backward pass dropout
    //if (dropout_rate_ != 0.0) {
    //  in_diff->MulElements(dropout_mask_);
    //}

    // calculate delta
    const BaseFloat mmt = opts_.momentum;

    // weight x -> g, i, f, o
    w_gifo_x_corr_.AddMatMat(1.0, DGIFO.RowRange(1*S,T*S), kTrans,
            in                     , kNoTrans, mmt);
    // recurrent weight r -> g, i, f, o
    w_gifo_r_corr_.AddMatMat(1.0, DGIFO.RowRange(1*S,T*S), kTrans,
            YM.RowRange(0*S,T*S)   , kNoTrans, mmt);
    // bias of g, i, f, o
    bias_corr_.AddRowSumMat(1.0, DGIFO.RowRange(1*S,T*S), mmt);

    // recurrent peephole c -> i
    peephole_i_c_corr_.AddDiagMatMat(1.0, DI.RowRange(1*S,T*S), kTrans,
            YC.RowRange(0*S,T*S), kNoTrans, mmt);
    // recurrent peephole c -> f
    peephole_f_c_corr_.AddDiagMatMat(1.0, DF.RowRange(1*S,T*S), kTrans,
            YC.RowRange(0*S,T*S), kNoTrans, mmt);
    // peephole c -> o
    peephole_o_c_corr_.AddDiagMatMat(1.0, DO.RowRange(1*S,T*S), kTrans,
            YC.RowRange(1*S,T*S), kNoTrans, mmt);

    if (clip_gradient_ > 0.0) {
        w_gifo_x_corr_.ApplyFloor(-clip_gradient_);
        w_gifo_x_corr_.ApplyCeiling(clip_gradient_);
        w_gifo_r_corr_.ApplyFloor(-clip_gradient_);
        w_gifo_r_corr_.ApplyCeiling(clip_gradient_);
        bias_corr_.ApplyFloor(-clip_gradient_);
        bias_corr_.ApplyCeiling(clip_gradient_);
        peephole_i_c_corr_.ApplyFloor(-clip_gradient_);
        peephole_i_c_corr_.ApplyCeiling(clip_gradient_);
        peephole_f_c_corr_.ApplyFloor(-clip_gradient_);
        peephole_f_c_corr_.ApplyCeiling(clip_gradient_);
        peephole_o_c_corr_.ApplyFloor(-clip_gradient_);
        peephole_o_c_corr_.ApplyCeiling(clip_gradient_);
    }
}

void Lstm::Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
    const BaseFloat lr  = opts_.learn_rate;

    w_gifo_x_.AddMat(-lr, w_gifo_x_corr_);
    w_gifo_r_.AddMat(-lr, w_gifo_r_corr_);
    bias_.AddVec(-lr, bias_corr_, 1.0);

    peephole_i_c_.AddVec(-lr, peephole_i_c_corr_, 1.0);
    peephole_f_c_.AddVec(-lr, peephole_f_c_corr_, 1.0);
    peephole_o_c_.AddVec(-lr, peephole_o_c_corr_, 1.0);

    //    /*
    //      Here we deal with the famous "vanishing & exploding difficulties" in RNN learning.
    //
    //      *For gradients vanishing*
    //      LSTM architecture introduces linear CEC as the "error bridge" across long time distance
    //      solving vanishing problem.
    //
    //      *For gradients exploding*
    //      LSTM is still vulnerable to gradients explosing in BPTT(with large weight & deep time expension).
    //      To prevent this, we tried L2 regularization, which didn't work well
    //
    //      Our approach is a *modified* version of Max Norm Regularization:
    //      For each nonlinear neuron,
    //      1. fan-in weights & bias model a seperation hyper-plane: W x + b = 0
    //      2. squashing function models a differentiable nonlinear slope around this hyper-plane.
    //
    //      Conventional max norm regularization scale W to keep its L2 norm bounded,
    //      As a modification, we scale down large (W & b) *simultaneously*, this:
    //      1. keeps all fan-in weights small, prevents gradients from exploding during backward-pass.
    //      2. keeps the location of the hyper-plane unchanged, so we don't wipe out already learned knowledge.
    //      3. shrinks the "normal" of the hyper-plane, smooths the nonlinear slope, improves generalization.
    //      4. makes the network *well-conditioned* (weights are constrained in a reasonible range).
    //
    //      We've observed faster convergence and performance gain by doing this.
    //    */
    //
    //    int DEBUG = 0;
    //    BaseFloat max_norm = 1.0;   // weights with large L2 norm may cause exploding in deep BPTT expensions
    //                  // TODO: move this config to opts_
    //    CuMatrix<BaseFloat> L2_gifo_x(w_gifo_x_);
    //    CuMatrix<BaseFloat> L2_gifo_r(w_gifo_r_);
    //    L2_gifo_x.MulElements(w_gifo_x_);
    //    L2_gifo_r.MulElements(w_gifo_r_);
    //
    //    CuVector<BaseFloat> L2_norm_gifo(L2_gifo_x.NumRows());
    //    L2_norm_gifo.AddColSumMat(1.0, L2_gifo_x, 0.0);
    //    L2_norm_gifo.AddColSumMat(1.0, L2_gifo_r, 1.0);
    //    L2_norm_gifo.Range(1*ncell_, ncell_).AddVecVec(1.0, peephole_i_c_, peephole_i_c_, 1.0);
    //    L2_norm_gifo.Range(2*ncell_, ncell_).AddVecVec(1.0, peephole_f_c_, peephole_f_c_, 1.0);
    //    L2_norm_gifo.Range(3*ncell_, ncell_).AddVecVec(1.0, peephole_o_c_, peephole_o_c_, 1.0);
    //    L2_norm_gifo.ApplyPow(0.5);
    //
    //    CuVector<BaseFloat> shrink(L2_norm_gifo);
    //    shrink.Scale(1.0/max_norm);
    //    shrink.ApplyFloor(1.0);
    //    shrink.InvertElements();
    //
    //    w_gifo_x_.MulRowsVec(shrink);
    //    w_gifo_r_.MulRowsVec(shrink);
    //    bias_.MulElements(shrink);
    //
    //    peephole_i_c_.MulElements(shrink.Range(1*ncell_, ncell_));
    //    peephole_f_c_.MulElements(shrink.Range(2*ncell_, ncell_));
    //    peephole_o_c_.MulElements(shrink.Range(3*ncell_, ncell_));
    //
    //    if (DEBUG) {
    //      if (shrink.Min() < 0.95) {   // we dont want too many trivial logs here
    //        std::cerr << "gifo shrinking coefs: " << shrink;
    //      }
    //    }
    //
}

void BLstm::InitMatParam(CuMatrix<BaseFloat> &m, float scale) {
    m.SetRandUniform();  // uniform in [0, 1]
    m.Add(-0.5);         // uniform in [-0.5, 0.5]
    m.Scale(2 * scale);  // uniform in [-scale, +scale]
}

void BLstm::InitVecParam(CuVector<BaseFloat> &v, float scale) {
    Vector<BaseFloat> tmp(v.Dim());
    for (int i=0; i < tmp.Dim(); i++) {
        tmp(i) = (RandUniform() - 0.5) * 2 * scale;
    }
    v = tmp;
}

/// set the utterance length used for parallel training
void BLstm::SetSeqLengths(const std::vector<int32> &sequence_lengths) {
    sequence_lengths_ = sequence_lengths;
}

void BLstm::InitData(std::istream &is) {
    // define options
    float param_scale = 0.02;
    // parse config
    std::string token;
    while (!is.eof()) {
        ReadToken(is, false, &token);
        if (token == "<ClipGradient>")
            ReadBasicType(is, false, &clip_gradient_);
        // else if (token == "<DropoutRate>")
        //  ReadBasicType(is, false, &dropout_rate_);
        else if (token == "<ParamScale>")
            ReadBasicType(is, false, &param_scale);
        else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
            << " (NumStream|ParamScale)";
        //<< " (CellDim|NumStream|DropoutRate|ParamScale)";
        is >> std::ws;
    }

    // init weight and bias (Uniform)
    // forward direction
    f_w_gifo_x_.Resize(4*ncell_, input_dim_, kUndefined);
    f_w_gifo_r_.Resize(4*ncell_, ncell_, kUndefined);

    // init weight and bias (Uniform)
    InitMatParam(f_w_gifo_x_, param_scale);
    InitMatParam(f_w_gifo_r_, param_scale);
    // backward direction
    b_w_gifo_x_.Resize(4*ncell_, input_dim_, kUndefined);
    b_w_gifo_r_.Resize(4*ncell_, ncell_, kUndefined);

    // init weight and bias (Uniform)
    InitMatParam(b_w_gifo_x_, param_scale);
    InitMatParam(b_w_gifo_r_, param_scale);

    // forward direction
    f_bias_.Resize(4*ncell_, kUndefined);
    // backward direction
    b_bias_.Resize(4*ncell_, kUndefined);
    InitVecParam(f_bias_, param_scale);
    InitVecParam(b_bias_, param_scale);

    // init weight and bias (Uniform)
    // forward direction
    f_peephole_i_c_.Resize(ncell_, kUndefined);
    f_peephole_f_c_.Resize(ncell_, kUndefined);
    f_peephole_o_c_.Resize(ncell_, kUndefined);
    // backward direction
    b_peephole_i_c_.Resize(ncell_, kUndefined);
    b_peephole_f_c_.Resize(ncell_, kUndefined);
    b_peephole_o_c_.Resize(ncell_, kUndefined);

    InitVecParam(f_peephole_i_c_, param_scale);
    InitVecParam(f_peephole_f_c_, param_scale);
    InitVecParam(f_peephole_o_c_, param_scale);

    InitVecParam(b_peephole_i_c_, param_scale);
    InitVecParam(b_peephole_f_c_, param_scale);
    InitVecParam(b_peephole_o_c_, param_scale);

    // init delta buffers
    // forward direction
    f_w_gifo_x_corr_.Resize(4*ncell_, input_dim_, kSetZero);
    f_w_gifo_r_corr_.Resize(4*ncell_, ncell_, kSetZero);
    f_bias_corr_.Resize(4*ncell_, kSetZero);

    // backward direction
    b_w_gifo_x_corr_.Resize(4*ncell_, input_dim_, kSetZero);
    b_w_gifo_r_corr_.Resize(4*ncell_, ncell_, kSetZero);
    b_bias_corr_.Resize(4*ncell_, kSetZero);

    // peep hole connect
    // forward direction
    f_peephole_i_c_corr_.Resize(ncell_, kSetZero);
    f_peephole_f_c_corr_.Resize(ncell_, kSetZero);
    f_peephole_o_c_corr_.Resize(ncell_, kSetZero);
    // backward direction
    b_peephole_i_c_corr_.Resize(ncell_, kSetZero);
    b_peephole_f_c_corr_.Resize(ncell_, kSetZero);
    b_peephole_o_c_corr_.Resize(ncell_, kSetZero);

    KALDI_ASSERT(clip_gradient_ >= 0.0);
}


void BLstm::ReadData(std::istream &is, bool binary) {
    ExpectToken(is, binary, "<ClipGradient>");
    ReadBasicType(is, binary, &clip_gradient_);
    // ExpectToken(is, binary, "<DropoutRate>");
    // ReadBasicType(is, binary, &dropout_rate_);

    // reading parameters corresponding to forward direction
    f_w_gifo_x_.Read(is, binary);
    f_w_gifo_r_.Read(is, binary);
    f_bias_.Read(is, binary);

    f_peephole_i_c_.Read(is, binary);
    f_peephole_f_c_.Read(is, binary);
    f_peephole_o_c_.Read(is, binary);

    // init delta buffers
    f_w_gifo_x_corr_.Resize(4*ncell_, input_dim_, kSetZero);
    f_w_gifo_r_corr_.Resize(4*ncell_, ncell_, kSetZero);
    f_bias_corr_.Resize(4*ncell_, kSetZero);

    f_peephole_i_c_corr_.Resize(ncell_, kSetZero);
    f_peephole_f_c_corr_.Resize(ncell_, kSetZero);
    f_peephole_o_c_corr_.Resize(ncell_, kSetZero);

    // reading parameters corresponding to backward direction
    b_w_gifo_x_.Read(is, binary);
    b_w_gifo_r_.Read(is, binary);
    b_bias_.Read(is, binary);

    b_peephole_i_c_.Read(is, binary);
    b_peephole_f_c_.Read(is, binary);
    b_peephole_o_c_.Read(is, binary);

    // init delta buffers
    b_w_gifo_x_corr_.Resize(4*ncell_, input_dim_, kSetZero);
    b_w_gifo_r_corr_.Resize(4*ncell_, ncell_, kSetZero);
    b_bias_corr_.Resize(4*ncell_, kSetZero);

    b_peephole_i_c_corr_.Resize(ncell_, kSetZero);
    b_peephole_f_c_corr_.Resize(ncell_, kSetZero);
    b_peephole_o_c_corr_.Resize(ncell_, kSetZero);
}


void BLstm::WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<ClipGradient>");
    WriteBasicType(os, binary, clip_gradient_);
    // WriteToken(os, binary, "<DropoutRate>");
    // WriteBasicType(os, binary, dropout_rate_);

    // writing parameters corresponding to forward direction
    f_w_gifo_x_.Write(os, binary);
    f_w_gifo_r_.Write(os, binary);
    f_bias_.Write(os, binary);

    f_peephole_i_c_.Write(os, binary);
    f_peephole_f_c_.Write(os, binary);
    f_peephole_o_c_.Write(os, binary);

    // writing parameters corresponding to backward direction
    b_w_gifo_x_.Write(os, binary);
    b_w_gifo_r_.Write(os, binary);
    b_bias_.Write(os, binary);

    b_peephole_i_c_.Write(os, binary);
    b_peephole_f_c_.Write(os, binary);
    b_peephole_o_c_.Write(os, binary);

}


int32 BLstm::NumParams() const {
    return 2*( f_w_gifo_x_.NumRows() * f_w_gifo_x_.NumCols() +
            f_w_gifo_r_.NumRows() * f_w_gifo_r_.NumCols() +
            f_bias_.Dim() +
            f_peephole_i_c_.Dim() +
            f_peephole_f_c_.Dim() +
            f_peephole_o_c_.Dim());
}


void BLstm::GetParams(Vector<BaseFloat>* wei_copy) const {
    wei_copy->Resize(NumParams());
    int32 offset, len;

    // Copying parameters corresponding to forward direction
    offset = 0;  len = f_w_gifo_x_.NumRows() * f_w_gifo_x_.NumCols();
    wei_copy->Range(offset, len).CopyRowsFromMat(f_w_gifo_x_);

    offset += len; len =f_w_gifo_r_.NumRows() * f_w_gifo_r_.NumCols();
    wei_copy->Range(offset, len).CopyRowsFromMat(f_w_gifo_r_);

    offset += len; len = f_bias_.Dim();
    wei_copy->Range(offset, len).CopyFromVec(f_bias_);

    offset += len; len = f_peephole_i_c_.Dim();
    wei_copy->Range(offset, len).CopyFromVec(f_peephole_i_c_);

    offset += len; len = f_peephole_f_c_.Dim();
    wei_copy->Range(offset, len).CopyFromVec(f_peephole_f_c_);

    offset += len; len = f_peephole_o_c_.Dim();
    wei_copy->Range(offset, len).CopyFromVec(f_peephole_o_c_);

    // Copying parameters corresponding to backward direction
    offset += len; len = b_w_gifo_x_.NumRows() * b_w_gifo_x_.NumCols();
    wei_copy->Range(offset, len).CopyRowsFromMat(b_w_gifo_x_);

    offset += len; len = b_w_gifo_r_.NumRows() * b_w_gifo_r_.NumCols();
    wei_copy->Range(offset, len).CopyRowsFromMat(b_w_gifo_r_);

    offset += len; len = b_bias_.Dim();
    wei_copy->Range(offset, len).CopyFromVec(b_bias_);

    offset += len; len = b_peephole_i_c_.Dim();
    wei_copy->Range(offset, len).CopyFromVec(b_peephole_i_c_);

    offset += len; len = b_peephole_f_c_.Dim();
    wei_copy->Range(offset, len).CopyFromVec(b_peephole_f_c_);

    offset += len; len = b_peephole_o_c_.Dim();
    wei_copy->Range(offset, len).CopyFromVec(b_peephole_o_c_);

    return;
}

void BLstm::GetGpuParams(std::vector<std::pair<BaseFloat *, int> > *params) {
    params->clear();
    // Forward params
    params->push_back(std::make_pair(f_w_gifo_x_.Data(), f_w_gifo_x_.NumRows() * f_w_gifo_x_.Stride()));
    params->push_back(std::make_pair(f_w_gifo_r_.Data(), f_w_gifo_r_.NumRows() * f_w_gifo_r_.Stride()));
    params->push_back(std::make_pair(f_bias_.Data(), f_bias_.Dim()));
    params->push_back(std::make_pair(f_peephole_i_c_.Data(), f_peephole_i_c_.Dim()));
    params->push_back(std::make_pair(f_peephole_f_c_.Data(), f_peephole_f_c_.Dim()));
    params->push_back(std::make_pair(f_peephole_o_c_.Data(), f_peephole_o_c_.Dim()));
    // Backward params
    params->push_back(std::make_pair(b_w_gifo_x_.Data(), b_w_gifo_x_.NumRows() * b_w_gifo_x_.Stride()));
    params->push_back(std::make_pair(b_w_gifo_r_.Data(), b_w_gifo_r_.NumRows() * b_w_gifo_r_.Stride()));
    params->push_back(std::make_pair(b_bias_.Data(), b_bias_.Dim()));
    params->push_back(std::make_pair(b_peephole_i_c_.Data(), b_peephole_i_c_.Dim()));
    params->push_back(std::make_pair(b_peephole_f_c_.Data(), b_peephole_f_c_.Dim()));
    params->push_back(std::make_pair(b_peephole_o_c_.Data(), b_peephole_o_c_.Dim()));
}

std::string BLstm::Info() const {
    return std::string("  ")  +
        "\n  Forward Direction weights:" +
        "\n  f_w_gifo_x_  "     + MomentStatistics(f_w_gifo_x_) +
        "\n  f_w_gifo_r_  "     + MomentStatistics(f_w_gifo_r_) +
        "\n  f_bias_  "         + MomentStatistics(f_bias_) +
        "\n  f_peephole_i_c_  " + MomentStatistics(f_peephole_i_c_) +
        "\n  f_peephole_f_c_  " + MomentStatistics(f_peephole_f_c_) +
        "\n  f_peephole_o_c_  " + MomentStatistics(f_peephole_o_c_) +
        "\n  Backward Direction weights:" +
        "\n  b_w_gifo_x_  "     + MomentStatistics(b_w_gifo_x_) +
        "\n  b_w_gifo_r_  "     + MomentStatistics(b_w_gifo_r_) +
        "\n  b_bias_  "         + MomentStatistics(b_bias_) +
        "\n  b_peephole_i_c_  " + MomentStatistics(b_peephole_i_c_) +
        "\n  b_peephole_f_c_  " + MomentStatistics(b_peephole_f_c_) +
        "\n  b_peephole_o_c_  " + MomentStatistics(b_peephole_o_c_); 
}


std::string BLstm::InfoGradient() const {
    // disassembling forward-pass forward-propagation buffer into different neurons,
    const CuSubMatrix<BaseFloat> F_YG(f_propagate_buf_.ColRange(0*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> F_YI(f_propagate_buf_.ColRange(1*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> F_YF(f_propagate_buf_.ColRange(2*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> F_YO(f_propagate_buf_.ColRange(3*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> F_YC(f_propagate_buf_.ColRange(4*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> F_YH(f_propagate_buf_.ColRange(5*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> F_YM(f_propagate_buf_.ColRange(6*ncell_, ncell_));

    // disassembling forward-pass back-propagation buffer into different neurons,
    const CuSubMatrix<BaseFloat> F_DG(f_backpropagate_buf_.ColRange(0*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> F_DI(f_backpropagate_buf_.ColRange(1*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> F_DF(f_backpropagate_buf_.ColRange(2*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> F_DO(f_backpropagate_buf_.ColRange(3*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> F_DC(f_backpropagate_buf_.ColRange(4*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> F_DH(f_backpropagate_buf_.ColRange(5*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> F_DM(f_backpropagate_buf_.ColRange(6*ncell_, ncell_));

    // disassembling backward-pass forward-propagation buffer into different neurons,
    const CuSubMatrix<BaseFloat> B_YG(b_propagate_buf_.ColRange(0*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> B_YI(b_propagate_buf_.ColRange(1*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> B_YF(b_propagate_buf_.ColRange(2*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> B_YO(b_propagate_buf_.ColRange(3*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> B_YC(b_propagate_buf_.ColRange(4*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> B_YH(b_propagate_buf_.ColRange(5*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> B_YM(b_propagate_buf_.ColRange(6*ncell_, ncell_));

    // disassembling backward-pass back-propagation buffer into different neurons,
    const CuSubMatrix<BaseFloat> B_DG(b_backpropagate_buf_.ColRange(0*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> B_DI(b_backpropagate_buf_.ColRange(1*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> B_DF(b_backpropagate_buf_.ColRange(2*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> B_DO(b_backpropagate_buf_.ColRange(3*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> B_DC(b_backpropagate_buf_.ColRange(4*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> B_DH(b_backpropagate_buf_.ColRange(5*ncell_, ncell_));
    const CuSubMatrix<BaseFloat> B_DM(b_backpropagate_buf_.ColRange(6*ncell_, ncell_));

    return std::string("  ") +
        "\n The Gradients:" +
        "\n  Forward Direction:" +
        "\n  f_w_gifo_x_corr_  "     + MomentStatistics(f_w_gifo_x_corr_) +
        "\n  f_w_gifo_r_corr_  "     + MomentStatistics(f_w_gifo_r_corr_) +
        "\n  f_bias_corr_  "         + MomentStatistics(f_bias_corr_) +
        "\n  f_peephole_i_c_corr_  " + MomentStatistics(f_peephole_i_c_corr_) +
        "\n  f_peephole_f_c_corr_  " + MomentStatistics(f_peephole_f_c_corr_) +
        "\n  f_peephole_o_c_corr_  " + MomentStatistics(f_peephole_o_c_corr_) +
        "\n  Backward Direction:" +
        "\n  b_w_gifo_x_corr_  "   + MomentStatistics(b_w_gifo_x_corr_) +
        "\n  b_w_gifo_r_corr_  "   + MomentStatistics(b_w_gifo_r_corr_) +
        "\n  b_bias_corr_  "     + MomentStatistics(b_bias_corr_) +
        "\n  b_peephole_i_c_corr_  " + MomentStatistics(b_peephole_i_c_corr_) +
        "\n  b_peephole_f_c_corr_  " + MomentStatistics(b_peephole_f_c_corr_) +
        "\n  b_peephole_o_c_corr_  " + MomentStatistics(b_peephole_o_c_corr_) +
        "\n The Activations:" +
        "\n  Forward Direction:" +
        "\n  F_YG  " + MomentStatistics(F_YG) +
        "\n  F_YI  " + MomentStatistics(F_YI) +
        "\n  F_YF  " + MomentStatistics(F_YF) +
        "\n  F_YC  " + MomentStatistics(F_YC) +
        "\n  F_YH  " + MomentStatistics(F_YH) +
        "\n  F_YO  " + MomentStatistics(F_YO) +
        "\n  F_YM  " + MomentStatistics(F_YM) +
        "\n  Backward Direction:" +
        "\n  B_YG  " + MomentStatistics(B_YG) +
        "\n  B_YI  " + MomentStatistics(B_YI) +
        "\n  B_YF  " + MomentStatistics(B_YF) +
        "\n  B_YC  " + MomentStatistics(B_YC) +
        "\n  B_YH  " + MomentStatistics(B_YH) +
        "\n  B_YO  " + MomentStatistics(B_YO) +
        "\n  B_YM  " + MomentStatistics(B_YM) +
        "\n The Derivatives:" +
        "\n  Forward Direction:" +
        "\n  F_DG  " + MomentStatistics(F_DG) +
        "\n  F_DI  " + MomentStatistics(F_DI) +
        "\n  F_DF  " + MomentStatistics(F_DF) +
        "\n  F_DC  " + MomentStatistics(F_DC) +
        "\n  F_DH  " + MomentStatistics(F_DH) +
        "\n  F_DO  " + MomentStatistics(F_DO) +
        "\n  F_DM  " + MomentStatistics(F_DM) +
        "\n  Backward Direction:" +
        "\n  B_DG  " + MomentStatistics(B_DG) +
        "\n  B_DI  " + MomentStatistics(B_DI) +
        "\n  B_DF  " + MomentStatistics(B_DF) +
        "\n  B_DC  " + MomentStatistics(B_DC) +
        "\n  B_DH  " + MomentStatistics(B_DH) +
        "\n  B_DO  " + MomentStatistics(B_DO) +
        "\n  B_DM  " + MomentStatistics(B_DM); 
}

void BLstm::PropagateFnc(const CuMatrixBase<BaseFloat> &in, 
                  CuMatrixBase<BaseFloat> *out) {
    int32 nstream_ = sequence_lengths_.size();
    KALDI_ASSERT(in.NumRows() % nstream_ == 0);
    int32 T = in.NumRows() / nstream_;
    int32 S = nstream_;

    // 0:forward pass history, [1, T]:current sequence, T+1:dummy
    // forward direction
    f_propagate_buf_.Resize((T+2)*S, 7 * ncell_, kSetZero);
    // backward direction
    b_propagate_buf_.Resize((T+2)*S, 7 * ncell_, kSetZero);

    // disassembling forward-pass forward-propagation buffer into different neurons,
    CuSubMatrix<BaseFloat> F_YG(f_propagate_buf_.ColRange(0*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_YI(f_propagate_buf_.ColRange(1*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_YF(f_propagate_buf_.ColRange(2*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_YO(f_propagate_buf_.ColRange(3*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_YC(f_propagate_buf_.ColRange(4*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_YH(f_propagate_buf_.ColRange(5*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_YM(f_propagate_buf_.ColRange(6*ncell_, ncell_));

    CuSubMatrix<BaseFloat> F_YGIFO(f_propagate_buf_.ColRange(0, 4*ncell_));

    // disassembling backward-pass forward-propagation buffer into different neurons,
    CuSubMatrix<BaseFloat> B_YG(b_propagate_buf_.ColRange(0*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_YI(b_propagate_buf_.ColRange(1*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_YF(b_propagate_buf_.ColRange(2*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_YO(b_propagate_buf_.ColRange(3*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_YC(b_propagate_buf_.ColRange(4*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_YH(b_propagate_buf_.ColRange(5*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_YM(b_propagate_buf_.ColRange(6*ncell_, ncell_));

    CuSubMatrix<BaseFloat> B_YGIFO(b_propagate_buf_.ColRange(0, 4*ncell_));

    // forward direction
    // x -> g, i, f, o, not recurrent, do it all in once
    F_YGIFO.RowRange(1*S, T*S).AddMatMat(1.0, in, kNoTrans, f_w_gifo_x_, kTrans, 0.0);

    // bias -> g, i, f, o
    F_YGIFO.RowRange(1*S, T*S).AddVecToRows(1.0, f_bias_);

    for (int t = 1; t <= T; t++) {
        // multistream buffers for current time-step
        CuSubMatrix<BaseFloat> y_all(f_propagate_buf_.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_g(F_YG.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_i(F_YI.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_f(F_YF.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_o(F_YO.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_c(F_YC.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_h(F_YH.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_m(F_YM.RowRange(t*S, S));

        CuSubMatrix<BaseFloat> y_gifo(F_YGIFO.RowRange(t*S, S));

        // r(t-1) -> g, i, f, o
        y_gifo.AddMatMat(1.0, F_YM.RowRange((t-1)*S, S), kNoTrans, f_w_gifo_r_, kTrans, 1.0);

        // c(t-1) -> i(t) via peephole
        y_i.AddMatDiagVec(1.0, F_YC.RowRange((t-1)*S, S), kNoTrans, f_peephole_i_c_, 1.0);

        // c(t-1) -> f(t) via peephole
        y_f.AddMatDiagVec(1.0, F_YC.RowRange((t-1)*S, S), kNoTrans, f_peephole_f_c_, 1.0);

        // i, f sigmoid squashing
        y_i.Sigmoid(y_i);
        y_f.Sigmoid(y_f);

        // g tanh squashing
        y_g.Tanh(y_g);

        // g -> c
        y_c.AddMatMatElements(1.0, y_g, y_i, 0.0);

        // c(t-1) -> c(t) via forget-gate
        y_c.AddMatMatElements(1.0, F_YC.RowRange((t-1)*S, S), y_f, 1.0);

        y_c.ApplyFloor(-50);   // optional clipping of cell activation
        y_c.ApplyCeiling(50);  // google paper Interspeech2014: LSTM for LVCSR

        // h tanh squashing
        y_h.Tanh(y_c);

        // c(t) -> o(t) via peephole (non-recurrent) & o squashing
        y_o.AddMatDiagVec(1.0, y_c, kNoTrans, f_peephole_o_c_, 1.0);

        // o sigmoid squashing
        y_o.Sigmoid(y_o);

        // h -> m via output gate
        y_m.AddMatMatElements(1.0, y_h, y_o, 0.0);

        // set zeros
        // for (int s = 0; s < S; s++) {
        //   if (t > sequence_lengths_[s])
        //     y_all.Row(s).SetZero();
        // }
    }

    // backward direction
    B_YGIFO.RowRange(1*S, T*S).AddMatMat(1.0, in, kNoTrans, b_w_gifo_x_, kTrans, 0.0);
    //// LSTM forward dropout
    //// Google paper 2014: Recurrent Neural Network Regularization
    //// by Wojciech Zaremba, Ilya Sutskever, Oriol Vinyals
    // if (dropout_rate_ != 0.0) {
    //  dropout_mask_.Resize(in.NumRows(), 4*ncell_, kUndefined);
    //  dropout_mask_.SetRandUniform();   // [0,1]
    //  dropout_mask_.Add(-dropout_rate_);  // [-dropout_rate, 1-dropout_rate_],
    //  dropout_mask_.ApplyHeaviside();   // -tive -> 0.0, +tive -> 1.0
    //  YGIFO.RowRange(1*S,T*S).MulElements(dropout_mask_);
    // }

    // bias -> g, i, f, o
    B_YGIFO.RowRange(1*S, T*S).AddVecToRows(1.0, b_bias_);

    // backward direction, from T to 1, t--
    for (int t = T; t >= 1; t--) {
        // multistream buffers for current time-step
        CuSubMatrix<BaseFloat> y_all(b_propagate_buf_.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_g(B_YG.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_i(B_YI.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_f(B_YF.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_o(B_YO.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_c(B_YC.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_h(B_YH.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_m(B_YM.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_gifo(B_YGIFO.RowRange(t*S, S));

        // r(t+1) -> g, i, f, o
        y_gifo.AddMatMat(1.0, B_YM.RowRange((t+1)*S, S), kNoTrans, b_w_gifo_r_, kTrans, 1.0);

        // c(t+1) -> i(t) via peephole
        y_i.AddMatDiagVec(1.0, B_YC.RowRange((t+1)*S, S), kNoTrans, b_peephole_i_c_, 1.0);

        // c(t+1) -> f(t) via peephole
        y_f.AddMatDiagVec(1.0, B_YC.RowRange((t+1)*S, S), kNoTrans, b_peephole_f_c_, 1.0);

        // i, f sigmoid squashing
        y_i.Sigmoid(y_i);
        y_f.Sigmoid(y_f);

        // g tanh squashing
        y_g.Tanh(y_g);

        // g -> c
        y_c.AddMatMatElements(1.0, y_g, y_i, 0.0);

        // c(t+1) -> c(t) via forget-gate
        y_c.AddMatMatElements(1.0, B_YC.RowRange((t+1)*S, S), y_f, 1.0);

        y_c.ApplyFloor(-50);   // optional clipping of cell activation
        y_c.ApplyCeiling(50);  // google paper Interspeech2014: LSTM for LVCSR

        // h tanh squashing
        y_h.Tanh(y_c);

        // c(t) -> o(t) via peephole (non-recurrent) & o squashing
        y_o.AddMatDiagVec(1.0, y_c, kNoTrans, b_peephole_o_c_, 1.0);

        // o sigmoid squashing
        y_o.Sigmoid(y_o);

        // h -> m via output gate
        y_m.AddMatMatElements(1.0, y_h, y_o, 0.0);

        for (int s = 0; s < S; s++) {
            if (t > sequence_lengths_[s])
                y_all.Row(s).SetZero();
        }
    }

    CuMatrix<BaseFloat> YM_FB;
    YM_FB.Resize((T+2)*S, 2 * ncell_, kSetZero);
    // forward part
    YM_FB.ColRange(0, ncell_).CopyFromMat(f_propagate_buf_.ColRange(6*ncell_, ncell_));
    // backward part
    YM_FB.ColRange(ncell_, ncell_).CopyFromMat(b_propagate_buf_.ColRange(6*ncell_, ncell_));
    // recurrent projection layer is also feed-forward as BLSTM output
    out->CopyFromMat(YM_FB.RowRange(1*S, T*S));
}

void BLstm::BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, 
                      const CuMatrixBase<BaseFloat> &out,
                      const CuMatrixBase<BaseFloat> &out_diff, 
                      CuMatrixBase<BaseFloat> *in_diff) {
    int DEBUG = 0;
    // the number of sequences to be processed in parallel
    int32 nstream_ = sequence_lengths_.size();
    int32 T = in.NumRows() / nstream_;
    int32 S = nstream_;
    // disassembling forward-pass forward-propagation buffer into different neurons,
    CuSubMatrix<BaseFloat> F_YG(f_propagate_buf_.ColRange(0*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_YI(f_propagate_buf_.ColRange(1*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_YF(f_propagate_buf_.ColRange(2*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_YO(f_propagate_buf_.ColRange(3*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_YC(f_propagate_buf_.ColRange(4*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_YH(f_propagate_buf_.ColRange(5*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_YM(f_propagate_buf_.ColRange(6*ncell_, ncell_));

    // 0:dummy, [1,T] frames, T+1 backward pass history
    f_backpropagate_buf_.Resize((T+2)*S, 7 * ncell_, kSetZero);

    // disassembling forward-pass back-propagation buffer into different neurons,
    CuSubMatrix<BaseFloat> F_DG(f_backpropagate_buf_.ColRange(0*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_DI(f_backpropagate_buf_.ColRange(1*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_DF(f_backpropagate_buf_.ColRange(2*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_DO(f_backpropagate_buf_.ColRange(3*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_DC(f_backpropagate_buf_.ColRange(4*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_DH(f_backpropagate_buf_.ColRange(5*ncell_, ncell_));
    CuSubMatrix<BaseFloat> F_DM(f_backpropagate_buf_.ColRange(6*ncell_, ncell_));

    CuSubMatrix<BaseFloat> F_DGIFO(f_backpropagate_buf_.ColRange(0, 4*ncell_));

    // layer to BLSTM output is not recurrent, so backprop it all in once
    F_DM.RowRange(1*S, T*S).CopyFromMat(out_diff.ColRange(0, ncell_));

    for (int t = T; t >= 1; t--) {
        CuSubMatrix<BaseFloat> y_g(F_YG.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_i(F_YI.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_f(F_YF.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_o(F_YO.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_c(F_YC.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_h(F_YH.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_m(F_YM.RowRange(t*S, S));

        CuSubMatrix<BaseFloat> d_g(F_DG.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_i(F_DI.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_f(F_DF.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_o(F_DO.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_c(F_DC.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_h(F_DH.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_m(F_DM.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_all(f_backpropagate_buf_.RowRange(t*S, S));
        // r
        //   Version 1 (precise gradients):
        //   backprop error from g(t+1), i(t+1), f(t+1), o(t+1) to r(t)
        d_m.AddMatMat(1.0, F_DGIFO.RowRange((t+1)*S, S), kNoTrans, f_w_gifo_r_, kNoTrans, 1.0);

        /*
        //   Version 2 (Alex Graves' PhD dissertation):
        //   only backprop g(t+1) to r(t)
        CuSubMatrix<BaseFloat> w_g_r_(w_gifo_r_.RowRange(0, ncell_));
        d_r.AddMatMat(1.0, DG.RowRange((t+1)*S,S), kNoTrans, w_g_r_, kNoTrans, 1.0);
         */

        /*
        //   Version 3 (Felix Gers' PhD dissertation):
        //   truncate gradients of g(t+1), i(t+1), f(t+1), o(t+1) once they leak out memory block
        //   CEC(with forget connection) is the only "error-bridge" through time
        ;
         */

        // m -> h via output gate
        d_h.AddMatMatElements(1.0, d_m, y_o, 0.0);
        d_h.DiffTanh(y_h, d_h);

        // o
        d_o.AddMatMatElements(1.0, d_m, y_h, 0.0);
        d_o.DiffSigmoid(y_o, d_o);

        // c
        // 1. diff from h(t)
        // 2. diff from c(t+1) (via forget-gate between CEC)
        // 3. diff from i(t+1) (via peephole)
        // 4. diff from f(t+1) (via peephole)
        // 5. diff from o(t)   (via peephole, not recurrent)
        d_c.AddMat(1.0, d_h);
        d_c.AddMatMatElements(1.0, F_DC.RowRange((t+1)*S, S), F_YF.RowRange((t+1)*S, S), 1.0);
        d_c.AddMatDiagVec(1.0, F_DI.RowRange((t+1)*S, S), kNoTrans, f_peephole_i_c_, 1.0);
        d_c.AddMatDiagVec(1.0, F_DF.RowRange((t+1)*S, S), kNoTrans, f_peephole_f_c_, 1.0);
        d_c.AddMatDiagVec(1.0, d_o           , kNoTrans, f_peephole_o_c_, 1.0);

        // f
        d_f.AddMatMatElements(1.0, d_c, F_YC.RowRange((t-1)*S, S), 0.0);
        d_f.DiffSigmoid(y_f, d_f);

        // i
        d_i.AddMatMatElements(1.0, d_c, y_g, 0.0);
        d_i.DiffSigmoid(y_i, d_i);

        // c -> g via input gate
        d_g.AddMatMatElements(1.0, d_c, y_i, 0.0);
        d_g.DiffTanh(y_g, d_g);
    }

    // disassembling backward-pass forward-propagation buffer into different neurons,
    CuSubMatrix<BaseFloat> B_YG(b_propagate_buf_.ColRange(0*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_YI(b_propagate_buf_.ColRange(1*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_YF(b_propagate_buf_.ColRange(2*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_YO(b_propagate_buf_.ColRange(3*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_YC(b_propagate_buf_.ColRange(4*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_YH(b_propagate_buf_.ColRange(5*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_YM(b_propagate_buf_.ColRange(6*ncell_, ncell_));

    // 0:dummy, [1,T] frames, T+1 backward pass history
    b_backpropagate_buf_.Resize((T+2)*S, 7 * ncell_, kSetZero);

    // disassembling backward-pass back-propagation buffer into different neurons,
    CuSubMatrix<BaseFloat> B_DG(b_backpropagate_buf_.ColRange(0*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_DI(b_backpropagate_buf_.ColRange(1*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_DF(b_backpropagate_buf_.ColRange(2*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_DO(b_backpropagate_buf_.ColRange(3*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_DC(b_backpropagate_buf_.ColRange(4*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_DH(b_backpropagate_buf_.ColRange(5*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_DM(b_backpropagate_buf_.ColRange(6*ncell_, ncell_));
    CuSubMatrix<BaseFloat> B_DGIFO(b_backpropagate_buf_.ColRange(0, 4*ncell_));

    // projection layer to BLSTM output is not recurrent, so backprop it all in once
    B_DM.RowRange(1*S, T*S).CopyFromMat(out_diff.ColRange(ncell_, ncell_));

    for (int t = 1; t <= T; t++) {
        CuSubMatrix<BaseFloat> y_g(B_YG.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_i(B_YI.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_f(B_YF.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_o(B_YO.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_c(B_YC.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_h(B_YH.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> y_m(B_YM.RowRange(t*S, S));

        CuSubMatrix<BaseFloat> d_g(B_DG.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_i(B_DI.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_f(B_DF.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_o(B_DO.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_c(B_DC.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_h(B_DH.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_m(B_DM.RowRange(t*S, S));
        CuSubMatrix<BaseFloat> d_all(b_backpropagate_buf_.RowRange(t*S, S));

        // r
        //   Version 1 (precise gradients):
        //   backprop error from g(t-1), i(t-1), f(t-1), o(t-1) to r(t)
        d_m.AddMatMat(1.0, B_DGIFO.RowRange((t-1)*S, S), kNoTrans, b_w_gifo_r_, kNoTrans, 1.0);

        /*
        //   Version 2 (Alex Graves' PhD dissertation):
        //   only backprop g(t+1) to r(t)
        CuSubMatrix<BaseFloat> w_g_r_(w_gifo_r_.RowRange(0, ncell_));
        d_r.AddMatMat(1.0, DG.RowRange((t+1)*S,S), kNoTrans, w_g_r_, kNoTrans, 1.0);
         */

        /*
        //   Version 3 (Felix Gers' PhD dissertation):
        //   truncate gradients of g(t+1), i(t+1), f(t+1), o(t+1) once they leak out memory block
        //   CEC(with forget connection) is the only "error-bridge" through time
         */

        // m -> h via output gate
        d_h.AddMatMatElements(1.0, d_m, y_o, 0.0);
        d_h.DiffTanh(y_h, d_h);

        // o
        d_o.AddMatMatElements(1.0, d_m, y_h, 0.0);
        d_o.DiffSigmoid(y_o, d_o);

        // c
        // 1. diff from h(t)
        // 2. diff from c(t+1) (via forget-gate between CEC)
        // 3. diff from i(t+1) (via peephole)
        // 4. diff from f(t+1) (via peephole)
        // 5. diff from o(t)   (via peephole, not recurrent)
        d_c.AddMat(1.0, d_h);
        d_c.AddMatMatElements(1.0, B_DC.RowRange((t-1)*S, S), B_YF.RowRange((t-1)*S, S), 1.0);
        d_c.AddMatDiagVec(1.0, B_DI.RowRange((t-1)*S, S), kNoTrans, b_peephole_i_c_, 1.0);
        d_c.AddMatDiagVec(1.0, B_DF.RowRange((t-1)*S, S), kNoTrans, b_peephole_f_c_, 1.0);
        d_c.AddMatDiagVec(1.0, d_o                     , kNoTrans, b_peephole_o_c_, 1.0);

        // f
        d_f.AddMatMatElements(1.0, d_c, B_YC.RowRange((t-1)*S, S), 0.0);
        d_f.DiffSigmoid(y_f, d_f);

        // i
        d_i.AddMatMatElements(1.0, d_c, y_g, 0.0);
        d_i.DiffSigmoid(y_i, d_i);

        // c -> g via input gate
        d_g.AddMatMatElements(1.0, d_c, y_i, 0.0);
        d_g.DiffTanh(y_g, d_g);
    }

    // g,i,f,o -> x, do it all in once
    // forward direction difference
    in_diff->AddMatMat(1.0, F_DGIFO.RowRange(1*S, T*S), kNoTrans, f_w_gifo_x_, kNoTrans, 0.0);
    // backward direction difference
    in_diff->AddMatMat(1.0, B_DGIFO.RowRange(1*S, T*S), kNoTrans, b_w_gifo_x_, kNoTrans, 1.0);

    // backward pass dropout
    // if (dropout_rate_ != 0.0) {
    //  in_diff->MulElements(dropout_mask_);
    //}

    // calculate delta
    const BaseFloat mmt = opts_.momentum;

    // forward direction
    // weight x -> g, i, f, o
    f_w_gifo_x_corr_.AddMatMat(1.0, F_DGIFO.RowRange(1*S, T*S), kTrans,
            in,                        kNoTrans, mmt);
    // recurrent weight r -> g, i, f, o
    f_w_gifo_r_corr_.AddMatMat(1.0, F_DGIFO.RowRange(1*S, T*S), kTrans,
            F_YM.RowRange(0*S, T*S),    kNoTrans, mmt);
    // bias of g, i, f, o
    f_bias_corr_.AddRowSumMat(1.0, F_DGIFO.RowRange(1*S, T*S), mmt);

    // recurrent peephole c -> i
    f_peephole_i_c_corr_.AddDiagMatMat(1.0, F_DI.RowRange(1*S, T*S), kTrans,
            F_YC.RowRange(0*S, T*S), kNoTrans, mmt);
    // recurrent peephole c -> f
    f_peephole_f_c_corr_.AddDiagMatMat(1.0, F_DF.RowRange(1*S, T*S), kTrans,
            F_YC.RowRange(0*S, T*S), kNoTrans, mmt);
    // peephole c -> o
    f_peephole_o_c_corr_.AddDiagMatMat(1.0, F_DO.RowRange(1*S, T*S), kTrans,
            F_YC.RowRange(1*S, T*S), kNoTrans, mmt);

    // apply the gradient clipping for forwardpass gradients
    if (clip_gradient_ > 0.0) {
        f_w_gifo_x_corr_.ApplyFloor(-clip_gradient_);
        f_w_gifo_x_corr_.ApplyCeiling(clip_gradient_);
        f_w_gifo_r_corr_.ApplyFloor(-clip_gradient_);
        f_w_gifo_r_corr_.ApplyCeiling(clip_gradient_);
        f_bias_corr_.ApplyFloor(-clip_gradient_);
        f_bias_corr_.ApplyCeiling(clip_gradient_);
        f_peephole_i_c_corr_.ApplyFloor(-clip_gradient_);
        f_peephole_i_c_corr_.ApplyCeiling(clip_gradient_);
        f_peephole_f_c_corr_.ApplyFloor(-clip_gradient_);
        f_peephole_f_c_corr_.ApplyCeiling(clip_gradient_);
        f_peephole_o_c_corr_.ApplyFloor(-clip_gradient_);
        f_peephole_o_c_corr_.ApplyCeiling(clip_gradient_);
    }

    // backward direction backpropagate
    // weight x -> g, i, f, o
    b_w_gifo_x_corr_.AddMatMat(1.0, B_DGIFO.RowRange(1*S, T*S), kTrans, in, kNoTrans, mmt);
    // recurrent weight r -> g, i, f, o
    b_w_gifo_r_corr_.AddMatMat(1.0, B_DGIFO.RowRange(1*S, T*S), kTrans,
            B_YM.RowRange(0*S, T*S)   , kNoTrans, mmt);
    // bias of g, i, f, o
    b_bias_corr_.AddRowSumMat(1.0, B_DGIFO.RowRange(1*S, T*S), mmt);

    // recurrent peephole c -> i, c(t+1) --> i
    b_peephole_i_c_corr_.AddDiagMatMat(1.0, B_DI.RowRange(1*S, T*S), kTrans,
            B_YC.RowRange(2*S, T*S), kNoTrans, mmt);
    // recurrent peephole c -> f, c(t+1) --> f
    b_peephole_f_c_corr_.AddDiagMatMat(1.0, B_DF.RowRange(1*S, T*S), kTrans,
            B_YC.RowRange(2*S, T*S), kNoTrans, mmt);
    // peephole c -> o
    b_peephole_o_c_corr_.AddDiagMatMat(1.0, B_DO.RowRange(1*S, T*S), kTrans,
            B_YC.RowRange(1*S, T*S), kNoTrans, mmt);

    // apply the gradient clipping for backwardpass gradients
    if (clip_gradient_ > 0.0) {
        b_w_gifo_x_corr_.ApplyFloor(-clip_gradient_);
        b_w_gifo_x_corr_.ApplyCeiling(clip_gradient_);
        b_w_gifo_r_corr_.ApplyFloor(-clip_gradient_);
        b_w_gifo_r_corr_.ApplyCeiling(clip_gradient_);
        b_bias_corr_.ApplyFloor(-clip_gradient_);
        b_bias_corr_.ApplyCeiling(clip_gradient_);
        b_peephole_i_c_corr_.ApplyFloor(-clip_gradient_);
        b_peephole_i_c_corr_.ApplyCeiling(clip_gradient_);
        b_peephole_f_c_corr_.ApplyFloor(-clip_gradient_);
        b_peephole_f_c_corr_.ApplyCeiling(clip_gradient_);
        b_peephole_o_c_corr_.ApplyFloor(-clip_gradient_);
        b_peephole_o_c_corr_.ApplyCeiling(clip_gradient_);
    }

    // forward direction
    if (DEBUG) {
        std::cerr << "gradients(with optional momentum): \n";
        std::cerr << "w_gifo_x_corr_ " << f_w_gifo_x_corr_;
        std::cerr << "w_gifo_r_corr_ " << f_w_gifo_r_corr_;
        std::cerr << "bias_corr_ "     << f_bias_corr_;
        std::cerr << "peephole_i_c_corr_ " << f_peephole_i_c_corr_;
        std::cerr << "peephole_f_c_corr_ " << f_peephole_f_c_corr_;
        std::cerr << "peephole_o_c_corr_ " << f_peephole_o_c_corr_;
    }
    // backward direction
    if (DEBUG) {
        std::cerr << "gradients(with optional momentum): \n";
        std::cerr << "w_gifo_x_corr_ " << b_w_gifo_x_corr_;
        std::cerr << "w_gifo_r_corr_ " << b_w_gifo_r_corr_;
        std::cerr << "bias_corr_ "     << b_bias_corr_;
        std::cerr << "peephole_i_c_corr_ " << b_peephole_i_c_corr_;
        std::cerr << "peephole_f_c_corr_ " << b_peephole_f_c_corr_;
        std::cerr << "peephole_o_c_corr_ " << b_peephole_o_c_corr_;
    }
}


void BLstm::Update(const CuMatrixBase<BaseFloat> &input, 
                   const CuMatrixBase<BaseFloat> &diff) {
    const BaseFloat lr  = opts_.learn_rate;
    // forward direction update
    f_w_gifo_x_.AddMat(-lr, f_w_gifo_x_corr_);
    f_w_gifo_r_.AddMat(-lr, f_w_gifo_r_corr_);
    f_bias_.AddVec(-lr, f_bias_corr_, 1.0);

    f_peephole_i_c_.AddVec(-lr, f_peephole_i_c_corr_, 1.0);
    f_peephole_f_c_.AddVec(-lr, f_peephole_f_c_corr_, 1.0);
    f_peephole_o_c_.AddVec(-lr, f_peephole_o_c_corr_, 1.0);

    // backward direction update
    b_w_gifo_x_.AddMat(-lr, b_w_gifo_x_corr_);
    b_w_gifo_r_.AddMat(-lr, b_w_gifo_r_corr_);
    b_bias_.AddVec(-lr, b_bias_corr_, 1.0);

    b_peephole_i_c_.AddVec(-lr, b_peephole_i_c_corr_, 1.0);
    b_peephole_f_c_.AddVec(-lr, b_peephole_f_c_corr_, 1.0);
    b_peephole_o_c_.AddVec(-lr, b_peephole_o_c_corr_, 1.0);

    /* For L2 regularization see "vanishing & exploding difficulties" in nnet-lstm-projected-streams.h */
}



} // namespace aslp_nnet
} // namespace kaldi

