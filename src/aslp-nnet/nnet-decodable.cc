// aslp-nnet/nnet-decodable.cc

#include "aslp-nnet/nnet-decodable.h"

namespace kaldi {
namespace aslp_nnet {

NnetDecodableBase::NnetDecodableBase(
        Nnet *nnet,
        const CuVector<BaseFloat> &log_priors,
        const TransitionModel &trans_model,
        const NnetDecodableOptions &opts):
    nnet_(nnet),
    log_priors_(log_priors),
    trans_model_(trans_model),
    opts_(opts),
    num_pdfs_(nnet_->OutputDim()),
    begin_frame_(-1) {
        KALDI_ASSERT(opts_.max_nnet_batch_size > 0);
        KALDI_ASSERT(log_priors_.Dim() == trans_model_.NumPdfs() &&
                "Priors in neural network not set up (or mismatch "
                "with transition model).");
        if (nnet_->NumOutput() != 1) {
            KALDI_ERR << "Num output must equal 1";
        }
        std::vector<int> flags(1, 1);
        nnet_->ResetLstmStreams(flags);
}

BaseFloat NnetDecodableBase::LogLikelihood(int32 frame, int32 index) {
    ComputeForFrame(frame);
    int32 pdf_id = trans_model_.TransitionIdToPdf(index);
    KALDI_ASSERT(frame >= begin_frame_ &&
            frame < begin_frame_ + scaled_loglikes_.NumRows());
    return scaled_loglikes_(frame - begin_frame_, pdf_id);
}

void NnetDecodableBase::ComputeForFrame(int32 frame) {
    int32 features_ready = NumFramesReady();
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
            FeatDim());
    for (int32 t = input_frame_begin; t < input_frame_end; t++) {
        SubVector<BaseFloat> row(features, t - input_frame_begin);
        GetFrame(t, &row);
    }
    CuMatrix<BaseFloat> cu_features; 
    // This function is perfect, avoid copy and duplicate matrix using 
    // constructor like CuMatrix<BaseFloat> cu_features(features)
    cu_features.Swap(&features);  // Copy to GPU, if we're using one.

    int32 num_frames_out = input_frame_end - input_frame_begin;
    CuMatrix<BaseFloat> cu_posteriors(num_frames_out, num_pdfs_);

    // Feedforward.
    // TODO: add more details about feature
    int skip_width = opts_.skip_width;
    if (skip_width > 1) {
        // Decode copy
        if (opts_.skip_type == "copy") {
            int skip_len = (cu_features.NumRows() - 1) / skip_width + 1;
            CuMatrix<BaseFloat> skip_out, skip_feat(skip_len, cu_features.NumCols()); 
            for (int i = 0; i < skip_len; i++) {
                skip_feat.Row(i).CopyFromVec(cu_features.Row(i * skip_width));
            }
            nnet_->Feedforward(skip_feat, &skip_out);
            for (int i = 0; i < skip_len; i++) {
                for (int j = 0; j < skip_width; j++) {
                    int idx = i * skip_width + j;
                    if (idx < cu_posteriors.NumRows()) {
                        cu_posteriors.Row(idx).CopyFromVec(skip_out.Row(i));
                    }
                }
            }
        }
        // Decode split
        else if (opts_.skip_type == "split") {
            CuMatrix<BaseFloat> skip_out, skip_feat; 
            for (int skip_offset = 0; skip_offset < skip_width; skip_offset++) {
                int skip_len = (cu_features.NumRows() - 1 - skip_offset) / skip_width + 1;
                skip_feat.Resize(skip_len, cu_features.NumCols()); 
                for (int i = 0; i < skip_len; i++) {
                    skip_feat.Row(i).CopyFromVec(cu_features.Row(i * skip_width + skip_offset));
                }
                nnet_->Feedforward(skip_feat, &skip_out);
                for (int i = 0; i < skip_len; i++) {
                    cu_posteriors.Row(i * skip_width + skip_offset).CopyFromVec(skip_out.Row(i));
                }
            }
        }
        else {
            KALDI_ERR << "Unsupported decode type " << opts_.skip_type;
        }
    }
    else {
        nnet_->Feedforward(cu_features, &cu_posteriors);
    }

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
