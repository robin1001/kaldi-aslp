/* Created on 2016-04-27
 * Author: Binbin Zhang
 */

#include "aslp-vad/nnet-vad.h"

namespace kaldi {

bool NnetVad::IsSilence(int frame) const {
    KALDI_ASSERT(sil_score_.size() == vad_result_.size());
    KALDI_ASSERT(frame < sil_score_.size());
    if (sil_score_[frame] > nnet_vad_config_.sil_thresh) return true;
    else return false;
}

// TODO optimize the feedfroward option
// especially reduce the copy operation
void NnetVad::GetScore(const Matrix<BaseFloat> &feat) {
    KALDI_ASSERT(feat.NumCols() == nnet_.InputDim());
    // Set nnet stream for recurrent component
    std::vector<int> frame_num_utt;
    frame_num_utt.push_back(feat.NumRows());
    const_cast<Nnet &>(nnet_).SetSeqLengths(frame_num_utt);
    // Get likelyhood
    CuMatrix<BaseFloat> nnet_out;
    Matrix<BaseFloat> nnet_out_host;
    const_cast<Nnet &>(nnet_).Feedforward(CuMatrix<BaseFloat>(feat), &nnet_out);
    nnet_out_host.Resize(nnet_out.NumRows(), nnet_out.NumCols());
    nnet_out.CopyToMat(&nnet_out_host);
    for (int i = 0; i < nnet_out_host.NumRows(); i++) 
        sil_score_[i] = nnet_out_host(i, 0);
}

bool NnetVad::DoVad(const VectorBase<BaseFloat> &raw_wav, 
                    const Matrix<BaseFloat> &raw_feat,
                    Vector<BaseFloat> *vad_wav) {
    sil_score_.resize(raw_feat.NumRows());
    vad_result_.resize(raw_feat.NumRows());
    // get nnet score
    GetScore(raw_feat);
    // call DoVad jugde every frame
    VadAll();
    //FrameExtractionOptions frame_opts;
    int num_vad_feats = 0;
    for (int i = 0; i < vad_result_.size(); i++) {
        if (vad_result_[i]) num_vad_feats++;
    }
    if (num_vad_feats == 0) return false;
    vad_wav->Resize(num_vad_feats * num_points_per_frame_);
    int index = 0;
    for (int i = 0; i < vad_result_.size(); i++) {
        if (vad_result_[i]) {
            KALDI_ASSERT((i+1) * num_points_per_frame_ < raw_wav.Dim());
            vad_wav->Range(index*num_points_per_frame_, \
                num_points_per_frame_).CopyFromVec( \
                raw_wav.Range(i*num_points_per_frame_, num_points_per_frame_));
            index++;
        }
    }
    KALDI_ASSERT(index == num_vad_feats);
    return true;
}

bool NnetVad::DoVad(const Matrix<BaseFloat> &raw_feat,
                    Matrix<BaseFloat> *vad_feat) {
    sil_score_.resize(raw_feat.NumRows());
    vad_result_.resize(raw_feat.NumRows());
    // get nnet score
    GetScore(raw_feat);
    // call DoVad jugde every frame
    VadAll();
    int num_vad_feats = 0;
    for (int i = 0; i < vad_result_.size(); i++) {
        if (vad_result_[i]) num_vad_feats++;
    }
    if (num_vad_feats == 0) return false;
    vad_feat->Resize(num_vad_feats, raw_feat.NumCols());
    int index = 0;
    for (int i = 0; i < vad_result_.size(); i++) {
        if (vad_result_[i]) {
            vad_feat->Row(index).CopyFromVec(raw_feat.Row(i));
            index++;
        }
    }
    KALDI_ASSERT(index == num_vad_feats);
    return true;
}



} // namespace kaldi
