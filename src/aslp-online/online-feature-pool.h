// aslp-online/online-feature-pool.h


/* Created on 2016-07-02
 * Author: zhangbinbin
 */
#ifndef ASLP_ONLINE_ONLINE_FEATURE_POOL_H_
#define ASLP_ONLINE_ONLINE_FEATURE_POOL_H_

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "itf/online-feature-itf.h"

namespace kaldi {
namespace aslp_online {

class OnlineFeaturePool : public OnlineFeatureInterface {
public:
    OnlineFeaturePool(int dim): dim_(dim), num_frames_(0),
                                input_finished_(false) { }

    virtual int32 Dim() const { return dim_; }

    virtual int32 NumFramesReady() const { return num_frames_; }

    virtual void GetFrame(int32 frame, VectorBase<BaseFloat> *feat) {
        feat->CopyFromVec(feature_pool_.Row(frame));
    }

    virtual bool IsLastFrame(int32 frame) const {
        return (frame == num_frames_ - 1 && input_finished_);
    }

    void InputFinished() {
        input_finished_ = true;
    }

    void AcceptFeature(const MatrixBase<BaseFloat> &feat) {
        KALDI_ASSERT(feat.NumCols() == dim_);
        if (feat.NumRows() == 0) return;
        int new_num_frames = num_frames_ + feat.NumRows();
        // If the feature pool is not big enough, expand it
        if (new_num_frames > feature_pool_.NumRows()) {
            int32 new_num_rows = std::max<int32>(new_num_frames,
                                                 feature_pool_.NumRows() * 2);
            feature_pool_.Resize(new_num_rows, Dim(), kCopyData);
        }
        feature_pool_.Range(num_frames_, feat.NumRows(), 0, 
                        Dim()).CopyFromMat(feat);
        num_frames_ = new_num_frames;
    }

private:
    int dim_;
    int num_frames_;
    bool input_finished_;
    Matrix<BaseFloat> feature_pool_;
};

} // namespace aslp_online
} // namespace kaldi

#endif
