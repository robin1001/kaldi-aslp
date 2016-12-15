/* Created on 2016-04-27
 * Author: Binbin Zhang
 */

#include "aslp-vad/energy-vad.h"

namespace kaldi {

bool EnergyVad::IsSilence(int frame) const {
    KALDI_ASSERT(frame < sil_scores_.size());
    if (sil_scores_[frame] > energy_vad_config_.sil_thresh) {
        return true;
    } else {
        return false;
    }
}

BaseFloat EnergyVad::FrameEnergy(const VectorBase<BaseFloat> &raw_wav, int32 frame) const {
    BaseFloat energy = 0.0;
    int32 index = frame * num_points_per_frame_;
    int i = 0;
    for (i = 0; i < num_points_per_frame_ && index < raw_wav.Dim(); i++, index++) {
        energy += raw_wav(index) * raw_wav(index);
    }
    KALDI_ASSERT(i > 0);
    return energy / i;
}

std::vector<BaseFloat> EnergyVad::GetScore(
    const VectorBase<BaseFloat> &raw_wav) {
    CalculateEnergy(raw_wav);
    CalculateScore();
    //for (int i = 0; i < sil_scores_.size(); i++) {
    //    std::cout << sil_scores_[i] << std::endl;
    //}
    return sil_scores_;
}

void EnergyVad::CalculateEnergy(const VectorBase<BaseFloat> &raw_wav) {
    int32 num_points = raw_wav.Dim();
    energy_vec_.resize(0);
    for (int i = 0; i * num_points_per_frame_ < num_points; i++) {
        energy_vec_.push_back(FrameEnergy(raw_wav, i));
    }
}

void EnergyVad::CalculateScore() {
    sil_scores_.resize(energy_vec_.size());
    for (int i = 0; i < sil_scores_.size(); i++) {
        //KALDI_ASSERT(energy_vec_[i] < raw_wav_max_value_);
        sil_scores_[i] = 1.0 - energy_vec_[i] / raw_wav_max_value_;
    }
}

bool EnergyVad::DoVad(const VectorBase<BaseFloat> &raw_wav, 
                       Vector<BaseFloat> *vad_wav) {
    GetScore(raw_wav);
    vad_result_.resize(sil_scores_.size());
    VadAll();
    int num_vad_feats = 0;
    for (int i = 0; i < vad_result_.size(); i++) {
        if (vad_result_[i]) num_vad_feats++;
    }
    if (num_vad_feats == 0) return false;
    vad_wav->Resize(num_vad_feats * num_points_per_frame_);
    int index = 0;
    for (int i = 0; i < vad_result_.size(); i++) {
        if (vad_result_[i]) {
            if ((i+1) * num_points_per_frame_ < raw_wav.Dim()) {
                vad_wav->Range(index*num_points_per_frame_, \
                    num_points_per_frame_).CopyFromVec( \
                    raw_wav.Range(i*num_points_per_frame_, num_points_per_frame_));
            }
            index++;
        }
    }
    KALDI_ASSERT(index == num_vad_feats);
    return true;
}

} // namespace kaldi
