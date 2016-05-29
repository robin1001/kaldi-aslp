// aslp-nnet/nnet-online-decodable.cc

#include "aslp-nnet/nnet-online-decodable.h"

namespace kaldi {
namespace aslp_nnet {

DecodableNnetOnline::DecodableNnetOnline(
        const Nnet &nnet,
        const CuVector<BaseFloat> &log_priors,
        const TransitionModel &trans_model,
        const DecodableNnetOnlineOptions &opts,
        OnlineFeatureInterface *input_feats):
    nnet_(nnet),
    log_priors_(log_priors),
    trans_model_(trans_model),
    features_(input_feats),
    opts_(opts),
    feat_dim_(input_feats->Dim()),
    num_pdfs_(nnet_.OutputDim()),
    begin_frame_(-1) {
        KALDI_ASSERT(opts_.max_nnet_batch_size > 0);
        KALDI_ASSERT(log_priors_.Dim() == trans_model_.NumPdfs() &&
                "Priors in neural network not set up (or mismatch "
                "with transition model).");
}

BaseFloat DecodableNnetOnline::LogLikelihood(int32 frame, int32 index) {
    ComputeForFrame(frame);
    int32 pdf_id = trans_model_.TransitionIdToPdf(index);
    KALDI_ASSERT(frame >= begin_frame_ &&
            frame < begin_frame_ + scaled_loglikes_.NumRows());
    return scaled_loglikes_(frame - begin_frame_, pdf_id);
}


bool DecodableNnetOnline::IsLastFrame(int32 frame) const {
    return features_->IsLastFrame(frame);
}

int32 DecodableNnetOnline::NumFramesReady() const {
    return features_->NumFramesReady();
}

void DecodableNnetOnline::ComputeForFrame(int32 frame) {
    int32 features_ready = features_->NumFramesReady();
    //bool input_finished = features_->IsLastFrame(features_ready - 1);  
    KALDI_ASSERT(frame >= 0);
    if (frame >= begin_frame_ &&
            frame < begin_frame_ + scaled_loglikes_.NumRows())
        return;
    KALDI_ASSERT(frame < NumFramesReady());

    int32 input_frame_begin = frame;
    int32 max_possible_input_frame_end = features_ready;

    int32 input_frame_end = std::min<int32>(max_possible_input_frame_end,
            input_frame_begin + opts_.max_nnet_batch_size);

    KALDI_ASSERT(input_frame_end > input_frame_begin);
    Matrix<BaseFloat> features(input_frame_end - input_frame_begin,
            feat_dim_);
    for (int32 t = input_frame_begin; t < input_frame_end; t++) {
        SubVector<BaseFloat> row(features, t - input_frame_begin);
        features_->GetFrame(t, &row);
    }
    CuMatrix<BaseFloat> cu_features; 
    // This function is perfect, avoid copy and duplicate matrix using 
    // constructor like CuMatrix<BaseFloat> cu_features(features)
    cu_features.Swap(&features);  // Copy to GPU, if we're using one.

    int32 num_frames_out = input_frame_end - input_frame_begin;

    CuMatrix<BaseFloat> cu_posteriors(num_frames_out, num_pdfs_);
    
    // Feedforward.
    // TODO: add more details about feature
    //nnet_.Feedforward(cu_features, &cu_posteriors);

    cu_posteriors.ApplyFloor(1.0e-20); // Avoid log of zero which leads to NaN.
    cu_posteriors.ApplyLog();

    // subtract log-prior (divide by prior)
    cu_posteriors.AddVecToRows(-1.0, log_priors_);
    // apply probability scale.
    cu_posteriors.Scale(opts_.acoustic_scale);

    // Transfer the scores the CPU for faster access by the decoding process.
    scaled_loglikes_.Resize(0, 0);
    cu_posteriors.Swap(&scaled_loglikes_);

    begin_frame_ = frame;
}

} // namespace aslp_nnet
} // namespace kaldi
