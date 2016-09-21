// nnet/nnet-gru-streams.h
// Copyright 2016 ASLP (Author: liwenpeng hechangqing zhangbinbin)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef ASLP_NNET_NNET_GRU_STREAMS_H_
#define ASLP_NNET_NNET_GRU_STREAMS_H_

#include "aslp-nnet/nnet-component.h"
#include "aslp-nnet/nnet-utils.h"
#include "aslp-cudamatrix/cu-math.h"


/*************************************
 * x: input neuron
 * z: update gate
 * r: reset gate
 * m: candidate activation ~h(t)
 * g: r(t) ** h(t-1)
 * h: squashing neuron near output
 *************************************/

namespace kaldi {
namespace aslp_nnet {

class GruStreams : public UpdatableComponent {
public:
	GruStreams(int32 input_dim, int32 output_dim) :
		UpdatableComponent(input_dim, output_dim),
		nstream_(0),
		clip_gradient_(0.0) //,
		// dropout_rate_(0.0)
	{ }

	~GruStreams()
	{ }

	Component* Copy() const { return new GruStreams(*this); }
	ComponentType GetType() const {return kGruStreams; }

	static void InitMatParam(CuMatrix<BaseFloat> &m, float scale) {
		m.SetRandUniform();  // uniform in [0, 1]
		m.Add(-0.5);         // uniform in [-0.5, 0.5]
		m.Scale(2 * scale);  // uniform in [-scale, +scale]
	}

    static void InitVecParam(CuVector<BaseFloat> &v, float scale) {
		Vector<BaseFloat> tmp(v.Dim());
		for (int i=0; i < tmp.Dim(); i++) {
			tmp(i) = (RandUniform() - 0.5) * 2 * scale;
		}
		v = tmp;
	}

	void InitData(std::istream &is) {
		// define options
		float param_scale = 0.02;
		// parse config
		std::string token;
		while (!is.eof()) {
			ReadToken(is, false, &token);
			if (token == "<ClipGradient>")
				ReadBasicType(is, false, &clip_gradient_);
			//else if (token == "<DropoutRate>")
			//	ReadBasicType(is, false, &dropout_rate_);
			else if (token == "<ParamScale>")
				ReadBasicType(is, false, &param_scale);
			else KALDI_ERR << "Unknown token " << token << ", a type in config?"
						   << "(ClipGradient|ParamScale)";
						   //<< "(CellDim|NumStream|DropoutRate|ParamScale)";
			is >> std::ws;
		}
		
		// init weight (Uniform)
		w_zrm_x_.Resize(3*output_dim_, input_dim_, kUndefined);
		w_zr_h_.Resize(2*output_dim_, output_dim_, kUndefined);
		w_m_g_.Resize(output_dim_, output_dim_, kUndefined);

		InitMatParam(w_zrm_x_, param_scale);
		InitMatParam(w_zr_h_, param_scale);
		InitMatParam(w_m_g_, param_scale);
		
		// init bias (Uniform)
		bias_.Resize(3*output_dim_, kUndefined);
		InitVecParam(bias_, param_scale);
		
		// init delta buffers
		w_zrm_x_corr_.Resize(3*output_dim_, input_dim_, kSetZero);
		w_zr_h_corr_.Resize(2*output_dim_, output_dim_, kSetZero);
		w_m_g_corr_.Resize(output_dim_, output_dim_, kSetZero);
		
		bias_corr_.Resize(3*output_dim_, kSetZero);

		KALDI_ASSERT(clip_gradient_ >= 0.0);
	}

	void ReadData(std::istream &is, bool binary) {
		ExpectToken(is, binary, "<ClipGradient>");
		ReadBasicType(is, binary, &clip_gradient_);
		//ExpectToken(is, binary, "<DropoutRate>");
		//ReadBasicType(is, binary, &dropout_rate_);
		
		w_zrm_x_.Read(is, binary);
		w_zr_h_.Read(is, binary);
		w_m_g_.Read(is, binary);
		bias_.Read(is, binary);

		w_zrm_x_corr_.Resize(3*output_dim_, input_dim_, kSetZero);
		w_zr_h_corr_.Resize(2*output_dim_, output_dim_, kSetZero);
		w_m_g_corr_.Resize(output_dim_, output_dim_, kSetZero);
		bias_corr_.Resize(3*output_dim_, kSetZero);
	}

	void WriteData(std::ostream &os, bool binary) const {
		WriteToken(os, binary, "<ClipGradient>");
		WriteBasicType(os, binary, clip_gradient_);
		//WriteToken(os, binary, "<DropoutRate>");
		//WriteBasicType(os, binary, dropout_rate_);
		w_zrm_x_.Write(os, binary);
		w_zr_h_.Write(os, binary);
		w_m_g_.Write(os, binary);
		bias_.Write(os, binary);
	}

	int32 NumParams() const {
		return ( w_zrm_x_.NumRows() * w_zrm_x_.NumCols() +
				 w_zr_h_.NumRows() * w_zr_h_.NumCols() +
				 w_m_g_.NumRows() * w_m_g_.NumCols() +
				 bias_.Dim() );
	}

	void GetParams(Vector<BaseFloat> *wei_copy) const {
		wei_copy->Resize(NumParams());

		int32 offset, len;
		offset = 0; len = w_zrm_x_.NumRows() * w_zrm_x_.NumCols();
		wei_copy->Range(offset, len).CopyRowsFromMat(w_zrm_x_);

		offset += len; len = w_zr_h_.NumRows() * w_zr_h_.NumCols();
		wei_copy->Range(offset, len).CopyRowsFromMat(w_zr_h_);

		offset += len; len = w_m_g_.NumRows() * w_m_g_.NumCols();
		wei_copy->Range(offset, len).CopyRowsFromMat(w_m_g_);

		offset += len; len = bias_.Dim();
		wei_copy->Range(offset, len).CopyFromVec(bias_);

		return;
	}

	void GetGpuParams(std::vector<std::pair<BaseFloat *, int> > *params) {
		params->clear();
		params->push_back(std::make_pair(w_zrm_x_.Data(), w_zrm_x_.NumRows() * w_zrm_x_.Stride()));
		params->push_back(std::make_pair(w_zr_h_.Data(), w_zr_h_.NumRows() * w_zr_h_.Stride()));
		params->push_back(std::make_pair(w_m_g_.Data(), w_m_g_.NumRows() * w_m_g_.Stride()));
		params->push_back(std::make_pair(bias_.Data(), bias_.Dim()));
	}

	std::string Info() const {
		return std::string("  ") +
			"\n w_zrm_x_ " + MomentStatistics(w_zrm_x_) +
			"\n w_zr_h_ " + MomentStatistics(w_zr_h_) +
			"\n w_m_g_ " + MomentStatistics(w_m_g_) +
			"\n bias_ " + MomentStatistics(bias_);
	}

	std::string InfoGradient() const {
		// disassemble forward-propagation buffer into different neurons,
		const CuSubMatrix<BaseFloat> YZ(propagate_buf_.ColRange(0*output_dim_, output_dim_));
		const CuSubMatrix<BaseFloat> YR(propagate_buf_.ColRange(1*output_dim_, output_dim_));
		const CuSubMatrix<BaseFloat> YM(propagate_buf_.ColRange(2*output_dim_, output_dim_));
		const CuSubMatrix<BaseFloat> YG(propagate_buf_.ColRange(3*output_dim_, output_dim_));
		const CuSubMatrix<BaseFloat> YH(propagate_buf_.ColRange(4*output_dim_, output_dim_));

		// disassemble backward-propagation buffer intp different neurons,
		const CuSubMatrix<BaseFloat> DZ(backpropagate_buf_.ColRange(0*output_dim_, output_dim_));
		const CuSubMatrix<BaseFloat> DR(backpropagate_buf_.ColRange(1*output_dim_, output_dim_));
		const CuSubMatrix<BaseFloat> DM(backpropagate_buf_.ColRange(2*output_dim_, output_dim_));
		const CuSubMatrix<BaseFloat> DG(backpropagate_buf_.ColRange(3*output_dim_, output_dim_));
		const CuSubMatrix<BaseFloat> DH(backpropagate_buf_.ColRange(4*output_dim_, output_dim_));

		return std::string("   ") +
			"\n Gradients:" +
			"\n w_zrm_x_corr_ " + MomentStatistics(w_zrm_x_corr_) +
			"\n w_zr_h_corr_ " + MomentStatistics(w_zr_h_corr_) +
			"\n w_m_g_corr_ " + MomentStatistics(w_m_g_corr_) +
			"\n bias_corr_ " + MomentStatistics(bias_corr_) +
			"\n Forward-pass: " +
			"\n YZ " + MomentStatistics(YZ) +
			"\n YR " + MomentStatistics(YR) +
			"\n YM " + MomentStatistics(YM) +
			"\n YG " + MomentStatistics(YG) +
			"\n YH " + MomentStatistics(YH) +
			"\n Backward-pass: " +
			"\n DZ " + MomentStatistics(DZ) +
			"\n DR " + MomentStatistics(DR) +
			"\n DM " + MomentStatistics(DM) +
			"\n DG " + MomentStatistics(DG) +
			"\n DH " + MomentStatistics(DH);
	}

    // For compatible with whole sentence train(like ctc train or lstm who sentence train
    void SetSeqLengths(const std::vector<int32> &sequence_lengths) {
        nstream_ = sequence_lengths.size();
	    prev_nnet_state_.Resize(nstream_, 5 * output_dim_, kSetZero);
    }

	void ResetLstmStreams(const std::vector<int32> &stream_reset_flag) {
		// allocate prev_nnet_state_ if not done yet,
		if (nstream_ == 0) {
			// Karel: we just got number of streams! (before the 1st batch comes)
			nstream_ = stream_reset_flag.size();
			prev_nnet_state_.Resize(nstream_, 5 * output_dim_, kSetZero);
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

	void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
		int DEBUG = 0;
		
		static bool do_stream_reset = false;
		if (nstream_ == 0) {
			do_stream_reset = true;
			nstream_ = 1; // Karel: we are in nnet-forward, so 1 stream,
			prev_nnet_state_.Resize(nstream_, 5 * output_dim_, kSetZero);
			KALDI_LOG << "Runing nnet-forward with per-utterance GRU-state reset";
		}
		if (do_stream_reset) prev_nnet_state_.SetZero();
		KALDI_ASSERT(nstream_ > 0);

		KALDI_ASSERT(in.NumRows() % nstream_ == 0);
		int32 T = in.NumRows() / nstream_;
		int32 S = nstream_;

		// 0:forward pass history, [1, T]:current sequence, T+1:dummy
		propagate_buf_.Resize((T+2)*S, 5 * output_dim_, kSetZero);
		propagate_buf_.RowRange(0*S,S).CopyFromMat(prev_nnet_state_);

		// disassemble entire neuron activation buffer in different neurons
		CuSubMatrix<BaseFloat> YZ(propagate_buf_.ColRange(0*output_dim_, output_dim_));
		CuSubMatrix<BaseFloat> YR(propagate_buf_.ColRange(1*output_dim_, output_dim_));
		CuSubMatrix<BaseFloat> YM(propagate_buf_.ColRange(2*output_dim_, output_dim_));
		CuSubMatrix<BaseFloat> YG(propagate_buf_.ColRange(3*output_dim_, output_dim_));
		CuSubMatrix<BaseFloat> YH(propagate_buf_.ColRange(4*output_dim_, output_dim_));

		CuSubMatrix<BaseFloat> YZRM(propagate_buf_.ColRange(0, 3*output_dim_));
		CuSubMatrix<BaseFloat> YZR(propagate_buf_.ColRange(0, 2*output_dim_));

		// x->z, r, m, not recurrent, do it all in once
		YZRM.RowRange(1*S, T*S).AddMatMat(1.0, in, kNoTrans, w_zrm_x_, kTrans, 0.0);
		
		// bias -> z, r, m
		YZRM.RowRange(1*S, T*S).AddVecToRows(1.0, bias_);

		for (int t = 1; t <= T; t++) {
			// multistream buffers for current time-step
			CuSubMatrix<BaseFloat> y_z(YZ.RowRange(t*S,S));
			CuSubMatrix<BaseFloat> y_r(YR.RowRange(t*S,S));
			CuSubMatrix<BaseFloat> y_m(YM.RowRange(t*S,S));
			CuSubMatrix<BaseFloat> y_g(YG.RowRange(t*S,S));
			CuSubMatrix<BaseFloat> y_h(YH.RowRange(t*S,S));

			CuSubMatrix<BaseFloat> y_zr(YZR.RowRange(t*S,S));

			// h(t-1) -> z, r
			y_zr.AddMatMat(1.0, YH.RowRange((t-1)*S, S), kNoTrans, w_zr_h_, kTrans, 1.0);

			// z, r sigmoid squashing
			y_zr.Sigmoid(y_zr);

			// r(t) * h(t-1) -> g(t)
			y_g.AddMatMatElements(1.0, y_r, YH.RowRange((t-1)*S, S), 0.0);

			// g(t) -> m(t)
			y_m.AddMatMat(1.0, y_g, kNoTrans, w_m_g_, kTrans, 1.0);

			// m tanh squashing
			y_m.Tanh(y_m);

			// h(t-1) z(t) m(t) -> h(t)
			y_h.AddMat(1.0, YH.RowRange((t-1)*S, S));
			y_h.AddMatMatElements(-1.0, YH.RowRange((t-1)*S, S), y_z, 1.0);
			y_h.AddMatMatElements(1.0, y_z, y_m, 1.0);

			if (DEBUG) {
				std::cerr << "forward-pass frame " << t << "\n";
				std::cerr << "activation of z: " << y_z;
				std::cerr << "activation of r: " << y_r;
				std::cerr << "activation of m: " << y_m;
				std::cerr << "activation of g: " << y_g;
				std::cerr << "activation of h: " << y_h;
			}
		}

		// recurrent projection layer is also feed-forward as GRU output
		out->CopyFromMat(YH.RowRange(1*S, T*S));

		// now the last frame state becomes previous network state for next batch
		prev_nnet_state_.CopyFromMat(propagate_buf_.RowRange(T*S, S));
	}

	void BackpropagateFnc( const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
					const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
		
		int DEBUG = 0;

		int32 T = in.NumRows() / nstream_;
		int32 S = nstream_;

		// disassemble propagated buffer into neurons
		CuSubMatrix<BaseFloat> YZ(propagate_buf_.ColRange(0*output_dim_, output_dim_));
		CuSubMatrix<BaseFloat> YR(propagate_buf_.ColRange(1*output_dim_, output_dim_));
		CuSubMatrix<BaseFloat> YM(propagate_buf_.ColRange(2*output_dim_, output_dim_));
		CuSubMatrix<BaseFloat> YG(propagate_buf_.ColRange(3*output_dim_, output_dim_));
		CuSubMatrix<BaseFloat> YH(propagate_buf_.ColRange(4*output_dim_, output_dim_));

		// 0:dummy, [1,T] frames, T+1 backward pass history
		backpropagate_buf_.Resize((T+2)*S, 5 * output_dim_, kSetZero);

		// disassemble backpropagate buffer into nuerons
		CuSubMatrix<BaseFloat> DZ(backpropagate_buf_.ColRange(0*output_dim_, output_dim_));
		CuSubMatrix<BaseFloat> DR(backpropagate_buf_.ColRange(1*output_dim_, output_dim_));
		CuSubMatrix<BaseFloat> DM(backpropagate_buf_.ColRange(2*output_dim_, output_dim_));
		CuSubMatrix<BaseFloat> DG(backpropagate_buf_.ColRange(3*output_dim_, output_dim_));
		CuSubMatrix<BaseFloat> DH(backpropagate_buf_.ColRange(4*output_dim_, output_dim_));

		CuSubMatrix<BaseFloat> DZRM(backpropagate_buf_.ColRange(0, 3*output_dim_));
		CuSubMatrix<BaseFloat> DZR(backpropagate_buf_.ColRange(0, 2*output_dim_));

		// backprop it all in once
		DH.RowRange(1*S,T*S).CopyFromMat(out_diff);

		for (int t = T; t >= 1; t--) {
			CuSubMatrix<BaseFloat> y_z(YZ.RowRange(t*S, S));
			CuSubMatrix<BaseFloat> y_r(YR.RowRange(t*S, S));
			CuSubMatrix<BaseFloat> y_m(YM.RowRange(t*S, S));
			CuSubMatrix<BaseFloat> y_g(YG.RowRange(t*S, S));
			CuSubMatrix<BaseFloat> y_h(YH.RowRange(t*S, S));

			CuSubMatrix<BaseFloat> d_z(DZ.RowRange(t*S, S));
			CuSubMatrix<BaseFloat> d_r(DR.RowRange(t*S, S));
			CuSubMatrix<BaseFloat> d_m(DM.RowRange(t*S, S));
			CuSubMatrix<BaseFloat> d_g(DG.RowRange(t*S, S));
			CuSubMatrix<BaseFloat> d_h(DH.RowRange(t*S, S));

			// d_h
			d_h.AddMatMat(1.0, DZR.RowRange((t+1)*S, S), kNoTrans, w_zr_h_, kNoTrans, 1.0);
			d_h.AddMat(1.0, DH.RowRange((t+1)*S, S));
			d_h.AddMatMatElements(-1.0, DH.RowRange((t+1)*S, S), YZ.RowRange((t+1)*S, S), 1.0);
			d_h.AddMatMatElements(1.0, DG.RowRange((t+1)*S, S), YR.RowRange((t+1)*S, S), 1.0);

			// m
			d_m.AddMatMatElements(1.0, d_h, y_z, 0.0);
			d_m.DiffTanh(y_m, d_m);

			// g
			d_g.AddMatMat(1.0, d_m, kNoTrans, w_m_g_, kNoTrans, 0.0);

			// r
			d_r.AddMatMatElements(1.0, d_g, YH.RowRange((t-1)*S, S), 0.0);
			d_r.DiffSigmoid(y_r, d_r);

			// z
			d_z.AddMatMatElements(1.0, d_h, y_m, 0.0);
			d_z.AddMatMatElements(-1.0, d_h, YH.RowRange((t-1)*S, S), 1.0);
			d_z.DiffSigmoid(y_z, d_z);

			// debug info
			if (DEBUG) {
				std::cerr << "backward-pass frame " << t << "\n";
				std::cerr << "derivative wrt input z " << d_z;
				std::cerr << "derivative wrt input r " << d_r;
				std::cerr << "derivative wrt input m " << d_m;
				std::cerr << "derivative wrt input g " << d_g;
				std::cerr << "derivative wrt input h " << d_h;
			}
		}

		// z,r,m -> x, do it all in once
		in_diff->AddMatMat(1.0, DZRM.RowRange(1*S, T*S), kNoTrans, w_zrm_x_, kNoTrans, 0.0);
		//// backward pass dropout
		// if (dropout_rate_ != 0.0) {
	 	// 		in_diff->MulElements(dropout_mask_);
		// }

		// calculate delta
		const BaseFloat mmt = opts_.momentum;

		// weight x -> z,r,m
		w_zrm_x_corr_.AddMatMat(1.0, DZRM.RowRange(1*S, T*S), kTrans,
									 in						, kNoTrans, mmt);
		//  bias of z, r, m
		bias_corr_.AddRowSumMat(1.0, DZRM.RowRange(1*S, T*S), mmt);

		// recurrent weight h -> z,r
		w_zr_h_corr_.AddMatMat(1.0, DZR.RowRange(1*S, T*S), kTrans,
									YH.RowRange(0*S, T*S),  kNoTrans, mmt);

		// recurrent weight g -> m
		w_m_g_corr_.AddMatMat(1.0, DM.RowRange(1*S, T*S), kTrans,
								   YG.RowRange(1*S, T*S), kNoTrans, mmt);

		if (clip_gradient_ > 0.0) {
			w_zrm_x_corr_.ApplyFloor(-clip_gradient_);
			w_zrm_x_corr_.ApplyCeiling(clip_gradient_);
			w_zr_h_corr_.ApplyFloor(-clip_gradient_);
			w_zr_h_corr_.ApplyCeiling(clip_gradient_);
			bias_corr_.ApplyFloor(-clip_gradient_);
			bias_corr_.ApplyCeiling(clip_gradient_);
			w_m_g_corr_.ApplyFloor(-clip_gradient_);
			w_m_g_corr_.ApplyCeiling(clip_gradient_);
		}
		
		if (DEBUG) {
			std::cerr << "gradients(with optional momentum): \n";
			std::cerr << "w_zrm_x_corr_ " << w_zrm_x_corr_;
			std::cerr << "w_zr_h_corr_ " << w_zr_h_corr_;
			std::cerr << "w_m_g_corr_ " << w_m_g_corr_;
			std::cerr << "bias_corr_ " << bias_corr_;
		}
	}
	
	void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
		const BaseFloat lr = opts_.learn_rate;

		w_zrm_x_.AddMat(-lr, w_zrm_x_corr_);
		w_zr_h_.AddMat(-lr, w_zr_h_corr_);
		w_m_g_.AddMat(-lr, w_m_g_corr_);
		bias_.AddVec(-lr, bias_corr_, 1.0);
	}

private:
	// dims
	int32 nstream_;

	CuMatrix<BaseFloat> prev_nnet_state_;

	// gradient-clipping value
	BaseFloat clip_gradient_;

	// feed-forward connections: from x to [z, r, m]
	CuMatrix<BaseFloat> w_zrm_x_;

	// recurrent connections: from h, g to [z, r, m]
	CuMatrix<BaseFloat> w_zr_h_;
	CuMatrix<BaseFloat> w_m_g_;

	CuMatrix<BaseFloat> w_zrm_x_corr_;
	CuMatrix<BaseFloat> w_zr_h_corr_;
	CuMatrix<BaseFloat> w_m_g_corr_;
	
	// biases of [z, r, m]
	CuVector<BaseFloat> bias_;
	CuVector<BaseFloat> bias_corr_;

	// propagate buffer: output of [z, r, g, m, h]
	CuMatrix<BaseFloat> propagate_buf_;

	// back-propagate buffer: diff-input of [z, r, g, m, h]
	CuMatrix<BaseFloat> backpropagate_buf_;

};

} // namespace aslp_nnet
} // namespace kaldi
#endif
