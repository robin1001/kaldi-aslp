// aslp-nnet/nnet-io.h

// Copyright 2016 ASLP (Author: zhangbinbin)


// Created on 2016-03-05

#ifndef ASLP_NNET_NNET_IO_H_
#define ASLP_NNET_NNET_IO_H_

#include "cudamatrix/cu-math.h"

#include "aslp-nnet/nnet-component.h"
#include "aslp-nnet/nnet-utils.h"

namespace kaldi {
namespace aslp_nnet {

class InputLayer: public Component {
public:
    InputLayer(int32 dim_in, int32 dim_out):
        Component(dim_in, dim_out) {
        KALDI_ASSERT(dim_in == dim_out);
    }
    ~InputLayer() {}
    Component* Copy() const { return new InputLayer(*this); }
    ComponentType GetType() const { return kInputLayer; }
    void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                            CuMatrixBase<BaseFloat> *out) {
        out->CopyFromMat(in);
    }

    void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
            const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
        in_diff->CopyFromMat(out_diff);
    }
};


class OutputLayer: public Component {
public:
    OutputLayer(int32 dim_in, int32 dim_out):
        Component(dim_in, dim_out) {
        KALDI_ASSERT(dim_in == dim_out);
    }
    ~OutputLayer() {}
    Component* Copy() const { return new OutputLayer(*this); }
    ComponentType GetType() const { return kOutputLayer; }
    void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                            CuMatrixBase<BaseFloat> *out) {
        out->CopyFromMat(in);
    }

    void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
            const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
        in_diff->CopyFromMat(out_diff);
    }
};

class ScaleLayer: public Component {
public:
    ScaleLayer(int32 dim_in, int32 dim_out):
        Component(dim_in, dim_out), scale_(1.0) {
        KALDI_ASSERT(dim_in == dim_out);
    }
    ~ScaleLayer() {}
    Component* Copy() const { return new ScaleLayer(*this); }
    ComponentType GetType() const { return kScaleLayer; }

    void InitData(std::istream &is) {
        ExpectToken(is, false, "<Scale>");
        ReadBasicType(is, false, &scale_);
    }

    void ReadData(std::istream &is, bool binary) {
        ExpectToken(is, binary, "<Scale>");
        ReadBasicType(is, binary, &scale_);
    }

    void WriteData(std::ostream &os, bool binary) const {
        WriteToken(os, binary, "<Scale>");
        WriteBasicType(os, binary, scale_);
    }

    void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                            CuMatrixBase<BaseFloat> *out) {
        out->CopyFromMat(in);
        out->Scale(scale_);
    }

    void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
            const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {
        in_diff->CopyFromMat(out_diff);
        in_diff->Scale(scale_);
    }

    BaseFloat Scale() const {
        return scale_;
    }
private:
    BaseFloat scale_;
};


} // namespace aslp_nnet
} // namespace kaldi

#endif
