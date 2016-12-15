// nnet/nnet-cfsmn-component.h

// Copyright 2016 ASLP (Author: liwenpeng)

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


#ifndef ASLP_NNET_FSMN_COMPONENT_H_
#define ASLP_NNET_FSMN_COMPONENT_H_

#include "aslp-cudamatrix/cu-math.h"

#include "aslp-nnet/nnet-component.h"
#include "aslp-nnet/nnet-utils.h"


namespace kaldi {
namespace aslp_nnet {

class CompactFsmn : public UpdatableComponent {
  
  public:
  	CompactFsmn(int32 dim_in, int32 dim_out)
	  : UpdatableComponent(dim_in, dim_out),
	    max_frames_(3000),
		learn_rate_coef_(1.0),
		clip_gradient_(0.0)
	  { }
	~CompactFsmn()
	{ }

 	Component* Copy() const { return new CompactFsmn(*this); }
	ComponentType GetType() const { return kCompactFsmn; }

	void InitMatParam(Matrix<BaseFloat> &m, float scale) {
		m.SetRandUniform();
		m.Add(-0.5);
		m.Scale(2 * scale);
	}
	
	void InitData(std::istream &is) {
		// define options
		int  past_context = 30, future_context = 30;
		float learn_rate_coef = 1.0, vec_coef_mean = 0.0, vec_coef_range = 1.0;
		// parse config
		std::string token;
		while (!is.eof()) {
			ReadToken(is, false, &token);
			if (token == "<PastContext>") ReadBasicType(is, false, &past_context);
			else if (token == "<FutureContext>") ReadBasicType(is, false, &future_context);
			else if (token == "<LearnRateCoef>") ReadBasicType(is, false, &learn_rate_coef);
			else if (token == "<VecCoefMean>") ReadBasicType(is, false, &vec_coef_mean);
			else if (token == "<VecCoefRange>") ReadBasicType(is, false, &vec_coef_range);
			else if (token == "<ClipGradient>") ReadBasicType(is, false, &clip_gradient_);
			else KALDI_ERR << "Unknown token " << token << ", a type in config?"
						   << " (PastContext|FutureContext|VecCoefMean|VecCoefRange|LearnRateCoef)";
			is >> std::ws; // eat-up whitespace
		}

		// initialize
		int32 num_row = past_context + future_context + 1;
		int32 num_col = input_dim_;
		

		Matrix<BaseFloat> vec_coef_mat(num_row, num_col);
		//float param_scale = 1.0f / num_col;
		float param_scale = 0.5 * sqrt(6.0 / (num_col + num_row));
		InitMatParam(vec_coef_mat, param_scale);
		vec_coef_ = vec_coef_mat;
		reversed_vec_coef_.Resize(vec_coef_.NumRows(), vec_coef_.NumCols());
		vec_coef_corr_.Resize(num_row, num_col);
		past_context_ = past_context;
		future_context_ = future_context;
		learn_rate_coef_ = learn_rate_coef;
		
		//KALDI_ASSERT(clip_gradient_ >= 0.0);
	}

	void ReadData(std::istream &is, bool binary) {
		// context infomation
		ExpectToken(is, binary, "<PastContext>");
		ReadBasicType(is, binary, &past_context_);
		ExpectToken(is, binary, "<FutureContext>");
		ReadBasicType(is, binary, &future_context_);
		// learn-rate scale
		ExpectToken(is, binary, "<LearnRateCoef>");
		ReadBasicType(is, binary, &learn_rate_coef_);
		//ExpectToken(is, binary, "<ClipGradient>");
		//ReadBasicType(is, binary, &clip_gradient_);
	
		// weights	
		vec_coef_.Read(is, binary);
		reversed_vec_coef_.Resize(vec_coef_.NumRows(), vec_coef_.NumCols());	
		vec_coef_corr_.Resize(vec_coef_.NumRows(), vec_coef_.NumCols());	
		KALDI_ASSERT(vec_coef_.NumCols() == input_dim_);
	}
	
	void WriteData(std::ostream &os, bool binary) const {
		WriteToken(os, binary, "<PastContext>");
		WriteBasicType(os, binary, past_context_);
		WriteToken(os, binary, "<FutureContext>");
		WriteBasicType(os, binary, future_context_);
		WriteToken(os, binary, "<LearnRateCoef>");
		WriteBasicType(os, binary, learn_rate_coef_);
		//WriteToken(os, binary, "<ClipGradient>");
		//WriteBasicType(os, binary, clip_gradient_);
		// weights
		vec_coef_.Write(os, binary);
	}
	int32 NumParams() const { 
	  return vec_coef_.NumRows() * vec_coef_.NumCols(); 
	}

	void SetMaxSeqLength(int32 max_len) {
		max_frames_ = max_len;
	}

	void GetParams(Vector<BaseFloat>* wei_copy) const {
		wei_copy->Resize(NumParams());
		wei_copy->CopyRowsFromMat(Matrix<BaseFloat>(vec_coef_));
	}

	void GetGradient(Vector<BaseFloat>* grad_copy) const {
		int32 NumGrads = vec_coef_corr_.NumRows() * vec_coef_corr_.NumCols();
		grad_copy->Resize(NumGrads);
		grad_copy->CopyRowsFromMat(Matrix<BaseFloat>(vec_coef_corr_));
	}
	
	void SetParams(Vector<BaseFloat>& params) {
		KALDI_ASSERT(params.Dim() == NumParams());
		vec_coef_.CopyRowsFromVec(params);
	}

	void GetGpuParams(std::vector<std::pair<BaseFloat *, int> > *params) {
		params->clear();
		params->push_back(std::make_pair(vec_coef_.Data(), vec_coef_.NumRows() * vec_coef_.Stride()));
	}

	std::string Info() {
		return std::string("\n vector_coefficient") + MomentStatistics(vec_coef_);
	}
	
	std::string InfoGradient() const {
		return std::string("\n vector_coefficient_grad") + MomentStatistics(vec_coef_corr_) +
			   ", learn-rate-coef" + ToString(learn_rate_coef_);
	}

	// pad mat with 0
	void PadMat(const CuMatrixBase<BaseFloat> &mat, const int32 left_context,
				const int32 right_context, CuMatrix<BaseFloat> *pad_mat) {
		pad_mat->Resize(left_context + mat.NumRows() + right_context, mat.NumCols(), kSetZero);
		CuSubMatrix<BaseFloat> real_data(pad_mat->RowRange(left_context, mat.NumRows()));
		real_data.CopyFromMat(mat);
	}

	void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
		int32 T = in.NumRows();
		int32 D = in.NumCols();
		int32 C = vec_coef_.NumRows();
		if (max_frames_ * C > aux_mat_.NumRows() || in.NumCols() != aux_mat_.NumCols()) {
		    KALDI_ASSERT(T <= max_frames_);
			aux_mat_.Resize(max_frames_ * C, D, kSetZero);
	    }
		if (max_frames_ + C - 1 > aux_pad_mat_.NumRows() || in.NumCols() != aux_pad_mat_.NumCols()) {
			KALDI_ASSERT(T <= max_frames_);
			aux_pad_mat_.Resize(max_frames_ + C - 1, D, kSetZero);
		}
		KALDI_ASSERT(in.NumCols() == vec_coef_.NumCols());

		CuSubMatrix<BaseFloat> padded_mat(aux_pad_mat_.RowRange(0, T + C - 1));
		padded_mat.SetZero();
		padded_mat.RowRange(past_context_, in.NumRows()).CopyFromMat(in);

		// out
		CuSubMatrix<BaseFloat> tmp_mat(aux_mat_.RowRange(0, T * C));
		tmp_mat.AddConvMatMatElements(1.0, padded_mat, vec_coef_, 0.0);	
		//for (int i = 0; i < T; i++) {
		//	const CuSubMatrix<BaseFloat> padded_mat_chunk(padded_mat.RowRange(i, C));
		//	CuSubMatrix<BaseFloat> tmp_mat_chunk(tmp_mat.RowRange(i * C, C));
		//	tmp_mat_chunk.AddMatMatElements(1.0, padded_mat_chunk, vec_coef_, 0.0);
		//}

		out->CopyFromMat(in);
		out->AddRowSumMat(1.0, tmp_mat, 1.0);
	}

	void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
						  const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
		int32 T = in.NumRows();
		//int32 D = in.NumCols();
		int32 C = vec_coef_.NumRows();
		    
	    KALDI_ASSERT(T <= max_frames_);
		
		KALDI_ASSERT(in.NumCols() == vec_coef_.NumCols());

		CuSubMatrix<BaseFloat> padded_mat(aux_pad_mat_.RowRange(0, T + C - 1));
		padded_mat.SetZero();
		padded_mat.RowRange(past_context_, in.NumRows()).CopyFromMat(in);
	
		// vec_coef_corr_
		CuSubMatrix<BaseFloat> tmp_mat(aux_mat_.RowRange(0, T * C));
		//tmp_mat.Resize(C * T, D, kSetZero);
	
		for (int i = 0; i < C; i++) {
			const CuSubMatrix<BaseFloat> padded_mat_chunk(padded_mat.RowRange(i, T));
			CuSubMatrix<BaseFloat> tmp_mat_chunk(tmp_mat.RowRange(i * T, T));
			tmp_mat_chunk.AddMatMatElements(1.0, padded_mat_chunk, out_diff, 0.0);
		}
		
		vec_coef_corr_.AddRowSumMat(1.0, tmp_mat, 0.0);
		//KALDI_LOG << "vec_coef_corr: " << vec_coef_corr_;
		CuSubMatrix<BaseFloat> padded_diff_mat(aux_pad_mat_.RowRange(0, T + C - 1));
		padded_diff_mat.SetZero();
		padded_diff_mat.RowRange(future_context_, out_diff.NumRows()).CopyFromMat(out_diff);
		//CuMatrix<BaseFloat> padded_diff_mat;
		//PadMat(out_diff, future_context_, past_context_, &padded_diff_mat);
		
		// reversed order of vec_coef_
		// CuMatrix<BaseFloat> reversed_vec_coef(vec_coef_.NumRows(), vec_coef_.NumCols());
		for (int i=0, j=vec_coef_.NumRows()-1; i < vec_coef_.NumRows() && j >= 0; i++,j--) {
			reversed_vec_coef_.Row(j).CopyFromVec(vec_coef_.Row(i));
		}
	
		// in_diff
		CuSubMatrix<BaseFloat> in_diff_tmp_mat(aux_mat_.RowRange(0, T * C));
		//CuMatrix<BaseFloat> in_tmp_mat;
		//in_tmp_mat.Resize(T * C, D, kSetZero);
		
		in_diff_tmp_mat.AddConvMatMatElements(1.0, padded_diff_mat, reversed_vec_coef_, 0.0);
		//for (int i = 0; i < T; i++) {
		//	const CuSubMatrix<BaseFloat> padded_diff_mat_chunk(padded_diff_mat.RowRange(i, C));
		//	CuSubMatrix<BaseFloat> in_diff_tmp_mat_chunk(in_diff_tmp_mat.RowRange(i * C, C));
		//	in_diff_tmp_mat_chunk.AddMatMatElements(1.0, padded_diff_mat_chunk,
		//									   reversed_vec_coef_, 0.0);
		//}
		
		in_diff->CopyFromMat(out_diff);
		in_diff->AddRowSumMat(1.0, in_diff_tmp_mat, 1.0);

		if (clip_gradient_ > 0.0) {
			vec_coef_corr_.ApplyFloor(-clip_gradient_);
			vec_coef_corr_.ApplyCeiling(clip_gradient_);
		}
	}

	void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
		const BaseFloat lr = opts_.learn_rate * learn_rate_coef_;
		//KALDI_LOG << "learn-rate " << lr;
		vec_coef_.AddMat(-lr, vec_coef_corr_);
	}

  private:
  	CuMatrix<BaseFloat> vec_coef_;
	CuMatrix<BaseFloat> vec_coef_corr_;
	CuMatrix<BaseFloat> reversed_vec_coef_;

	CuMatrix<BaseFloat> aux_mat_;	
	CuMatrix<BaseFloat> aux_pad_mat_;
	int32 max_frames_;
	//CuMatrix<BaseFloat> aux_in_mat_;
	//CuMatrix<BaseFloat> aux_vec_mat_;
	
	BaseFloat learn_rate_coef_;
	int32 past_context_;
	int32 future_context_;
	BaseFloat clip_gradient_;
};

}// namespace aslp_nnet
}// namespace kaldi

#endif
