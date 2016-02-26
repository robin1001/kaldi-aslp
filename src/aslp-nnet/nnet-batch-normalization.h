// nnet/nnet-affine-transform.h

// Copyright 2016 ASLP (Author: zhangbinbin)

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


#ifndef ASLP_NNET_NNET_BATCH_NORMALIZATION_H_
#define ASLP_NNET_NNET_BATCH_NORMALIZATION_H_

#include "cudamatrix/cu-math.h"

#include "aslp-nnet/nnet-component.h"
#include "aslp-nnet/nnet-utils.h"

namespace kaldi {
namespace aslp_nnet {

class BatchNormalization: public UpdatableComponent {
public:
    BatchNormalization(int32 dim_in, int32 dim_out) 
        : UpdatableComponent(dim_in, dim_out), 
		var_floor_(0.0000001),
		num_acc_frames_(0),
		acc_cleaned_(false)
    { }
    ~BatchNormalization(){ }

    Component* Copy() const { return new BatchNormalization(*this); }
    ComponentType GetType() const { return kBatchNormalization; }

    void InitData(std::istream &is) {
        num_acc_frames_ = 0;
        scale_.Resize(output_dim_);
        scale_.Set(1.0);
        shift_.Resize(output_dim_);
        shift_.SetZero();
        KALDI_ASSERT(output_dim_ > 0 && input_dim_ > 0);
        acc_means_.Resize(output_dim_, kSetZero);
        acc_vars_.Resize(output_dim_, kSetZero);
    }

    void ReadData(std::istream &is, bool binary) {
        ExpectToken(is, binary, "<NumAccFrames>");
        ReadBasicType(is, binary, &num_acc_frames_);
        acc_means_.Read(is, binary);
        acc_vars_.Read(is, binary);
        shift_.Read(is, binary);
        scale_.Read(is, binary);

        KALDI_ASSERT(acc_means_.Dim() == acc_vars_.Dim());
        KALDI_ASSERT(acc_means_.Dim() == shift_.Dim());
        KALDI_ASSERT(acc_means_.Dim() == scale_.Dim());
        if (num_acc_frames_ <= 0.0)
            return;
        float var_floor = 1e-10;
        Vector<double> acc_mean(acc_means_.Dim()), acc_var(acc_means_.Dim());
        Vector<BaseFloat> mean_vec_host(acc_means_.Dim()), var_vec_host(acc_means_.Dim());
        acc_means_.CopyToVec(&acc_mean);
        acc_vars_.CopyToVec(&acc_var);

        //compute the shift and scale per each dimension
        for (int32 d = 0; d < acc_mean.Dim(); d++) {
            BaseFloat mean = acc_mean(d) / num_acc_frames_;
            BaseFloat var = acc_var(d) / num_acc_frames_ - mean*mean;
            if (var <= var_floor) {
                KALDI_WARN << "Very small variance " << var << " flooring to " << var_floor;
                var = var_floor;
            }
            mean_vec_host(d) = mean;
            var_vec_host(d) = 1.0 / sqrt(var + var_floor_);
        }
        mean_vec_ = mean_vec_host;
        var_vec_ = var_vec_host;
    }

    void WriteData(std::ostream &os, bool binary) const {
        WriteToken(os, binary, "<NumAccFrames>");
        WriteBasicType(os, binary, num_acc_frames_);
        acc_means_.Write(os, binary);
        acc_vars_.Write(os, binary);
        shift_.Write(os, binary);
        scale_.Write(os, binary);
    }

    int32 NumParams() const { 
        return shift_.Dim() + scale_.Dim(); 
    }

    void GetParams(Vector<BaseFloat>* wei_copy) const {
        wei_copy->Resize(NumParams());
        wei_copy->Range(0, shift_.Dim()).CopyFromVec(Vector<BaseFloat>(shift_));
        wei_copy->Range(shift_.Dim(), scale_.Dim()).CopyFromVec(Vector<BaseFloat>(scale_));
    }

    void GetGpuParams(std::vector<std::pair<BaseFloat *, int> > *params) {
        params->clear();
        params->push_back(std::make_pair(shift_.Data(), shift_.Dim()));
        params->push_back(std::make_pair(scale_.Data(), scale_.Dim()));
    }

    std::string Info() const {
	    return std::string("\n  batch_normaliztion");
    }

    void CleanAccs() {
        acc_means_.SetZero();
        acc_vars_.SetZero();
        num_acc_frames_ = 0;
    }

    void FeedforwardFnc(const CuMatrixBase<BaseFloat> &in,
                        CuMatrixBase<BaseFloat> *out) {
        int32 batch_size = in.NumRows();
        if (XsharpO_.NumRows() != batch_size) {
            XsharpO_.Resize(batch_size, output_dim_);
        }
        // \delta <- 1/m \sum (x_i - mu)^2
        XsharpO_.CopyFromMat(in);
        XsharpO_.AddVecToRows(-1.0, mean_vec_, 1.0);
        XsharpO_.MulColsVec(var_vec_);
        out->CopyFromMat(XsharpO_);
        out->MulColsVec(scale_);
        out->AddVecToRows(1.0, shift_, 1.0);
    }

    void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
        if (!acc_cleaned_) {
            acc_cleaned_ = true;
            CleanAccs();
        }

        int32 batch_size = in.NumRows();
        if (XsharpO_.NumRows() != batch_size
                || mean_vec_.Dim() != output_dim_
                || var_vec_.Dim() != output_dim_) {
            XsharpO_.Resize(batch_size, output_dim_);
            bufE_.Resize(batch_size, output_dim_);
            mean_vec_.Resize(output_dim_);
            var_vec_.Resize(output_dim_);
        }

        // \mu <- 1/m \sum x_i
        mean_vec_.AddRowSumMat(1.0 / (batch_size), in, 0.0);

        // \delta <- 1/m \sum (x_i - mu)^2
        XsharpO_.CopyFromMat(in);
        XsharpO_.AddVecToRows(-1.0, mean_vec_, 1.0);
        out->AddMatMatElements(1.0, XsharpO_, XsharpO_, 0.0);
        //out->CopyFromMat(XsharpO_);
        //out->MulElements(XsharpO_);
        var_vec_.AddRowSumMat(1.0 / (batch_size), *out, 0.0);

        /// out <- (x - \mu) / \sqrt(\delta^2) ;
        var_vec_.Add(var_floor_);
        var_vec_.ApplyPow(0.5);
        var_vec_.InvertElements();
        XsharpO_.MulColsVec(var_vec_);
        out->CopyFromMat(XsharpO_);

        /// 
        out->MulColsVec(scale_);
        out->AddVecToRows(1.0, shift_, 1.0);

        /// for statis;
        num_acc_frames_ += batch_size;
        acc_means_.AddRowSumMat(1.0, CuMatrix<double>(in), 1.0);
        bufE_.AddMatMatElements(1.0, in, in, 0.0);
        acc_vars_.AddRowSumMat(1.0, CuMatrix<double>(bufE_), 1.0);
    }

    void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
            const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
        int32 batch_size = in.NumRows();

        if (dmean_vec_.Dim() != output_dim_) {
            dmean_vec_.Resize(output_dim_);
            dvar_vec_.Resize(output_dim_);
            dshift_.Resize(output_dim_);
            dscale_.Resize(output_dim_);
        }

        // calculate the W-gradient (dGamma)
        bufE_.AddMatMatElements(1.0, XsharpO_, out_diff, 0.0);
        dscale_.AddRowSumMat(1.0 /*/ (BaseFloat)batch_size*/, bufE_, opts_.momentum);
        dshift_.AddRowSumMat(1.0 /*/ (BaseFloat)batch_size*/, out_diff, opts_.momentum);

        if (in_diff == NULL) return;

        // calculate the backprop-error
        // 1. delta-xSharp : XsharpO_ = XsharpO_*gamma(W_)
        // d(l)/d(xSharp) = d(l)/d(y) * gamma
        XsharpO_.CopyFromMat(out_diff);
        XsharpO_.MulColsVec(scale_);

        // 2. delta-var
        // var^(-3/2)
        dvar_vec_.CopyFromVec(var_vec_);
        dvar_vec_.ApplyPow(3);
        dvar_vec_.Scale(-0.5);
        // bufE_ = x_ij - m_j
        bufE_.CopyFromMat(in);
        bufE_.AddVecToRows(-1.0, mean_vec_, 1.0);
        // ( x_ij - m_j ) * delta * var(-3/2)
        bufE_.MulElements(XsharpO_);
        bufE_.MulColsVec(dvar_vec_);
        dvar_vec_.AddRowSumMat(1.0, bufE_, 0.0);

        // 3.delta-mean
        // - delta * var
        bufE_.CopyFromMat(XsharpO_);
        bufE_.MulColsVec(var_vec_);
        bufE_.Scale(-1.0);
        dmean_vec_.AddRowSumMat(1.0, bufE_, 0.0);
        // ( x_ij - m_j ) * (2 / bat_size) * dvar_vec_
        bufE_.CopyFromMat(in);
        bufE_.AddVecToRows(-1.0, mean_vec_);
        bufE_.Scale(2.0 / batch_size);
        bufE_.MulColsVec(dvar_vec_);
        dmean_vec_.AddRowSumMat(-1.0, bufE_, 1.0);

        // 4.finally calculate backprop-error
        in_diff->CopyFromMat(XsharpO_);
        in_diff->MulColsVec(var_vec_);
        in_diff->AddMat(1.0, bufE_);
        in_diff->AddVecToRows(1.0 / batch_size, dmean_vec_, 1.0);
    }


    void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
        const BaseFloat lr = opts_.learn_rate;
        scale_.AddVec(-lr, dscale_, 1.0);
        shift_.AddVec(-lr, dshift_, 1.0);
    }
 private:
	CuMatrix<BaseFloat> XsharpO_;
	CuMatrix<BaseFloat> bufE_;
	CuVector<BaseFloat> acc_mean_vec_, acc_var_vec_;
	CuVector<BaseFloat> mean_vec_, dmean_vec_;
	CuVector<BaseFloat> var_vec_, dvar_vec_;
	CuVector<BaseFloat> scale_, dscale_;
	CuVector<BaseFloat> shift_, dshift_;
	BaseFloat var_floor_;

	CuVector<double> acc_means_;
	CuVector<double> acc_vars_;
	double num_acc_frames_;
	bool acc_cleaned_;
};

} // namespace aslp_nnet
} // namespace kaldi

#endif
