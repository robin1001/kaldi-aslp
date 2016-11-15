/* 
 * Created on 2016-11-13
 * Author: Zhang Binbin
 */

#include "aslp-online/online-vad-feature-pipeline.h"


namespace kaldi {
namespace aslp_online {

int OnlineVadFeaturePipeline::GetRawFeature(Matrix<BaseFloat> *feats) {
    KALDI_ASSERT(feats != NULL);
    int num_frames = NumFramesReady() - get_raw_feat_offset_;
    if (num_frames > 0) {
        feats->Resize(num_frames, AdaptedFeature()->Dim());
        for (int32 i = get_raw_feat_offset_; i < NumFramesReady(); i++) {
            SubVector<BaseFloat> row(*feats, i - get_raw_feat_offset_);
            AdaptedFeature()->GetFrame(i, &row);
        }
        get_raw_feat_offset_ = NumFramesReady();
    }
    return num_frames;
}

void OnlineVadFeaturePipeline::AcceptWaveform(
    BaseFloat sampling_rate,
    const VectorBase<BaseFloat> &waveform) {
    OnlineFeaturePipeline::AcceptWaveform(sampling_rate, waveform);
    
    Vad();
}

void OnlineVadFeaturePipeline::Vad() {
    Matrix<BaseFloat> raw_feats;
    this->GetRawFeature(&raw_feats);
    nnet_vad_.DoVad(raw_feats);
    const std::vector<bool> &chunk_result = nnet_vad_.VadResult();
    int prev_size = vad_result_.size();
    vad_result_.insert(vad_result_.end(), 
                       chunk_result.begin(), chunk_result.end());
    int cur = prev_size;
    // Lookback
    while (cur < vad_result_.size()) {
        // sil, go ahead
        while (cur < vad_result_.size() && !vad_result_[cur]) cur++;
        // end with silence, no more voice
        if (cur == vad_result_.size()) break;
        // voice segment start, lookback and assign it to voice(true) 
        int start = std::max(0, cur - lookback_frames_);
        for (int i = start; i < cur; i++) vad_result_[i] = true; 
        // voice, go ahead
        while (cur < vad_result_.size() && vad_result_[cur]) cur++;
    }
    KALDI_ASSERT(cur == vad_result_.size());
    //printf("before lookback ");
    //for (int i = 0; i < chunk_result.size(); i++) {
    //    printf("%d ", static_cast<int>(chunk_result[i]));
    //}
    //printf("\n");
    //printf("after  lookback ");
    //for (int i = prev_size; i < vad_result_.size(); i++) {
    //    printf("%d ", static_cast<int>(vad_result_[i]));
    //}
    //printf("\n");

    endpoint_detected_ = false;
    int num_sil = 0;
    for (int i = prev_size; i < vad_result_.size(); i++) {
        if (!vad_result_[i]) {
            num_sil++;
            num_continuous_silence_++;
            if (num_continuous_silence_ > endpoint_frames_) {
                KALDI_LOG << "endpoint detected";
                endpoint_detected_ = true;
                num_continuous_silence_ = 0;
            }
        } else {
            num_continuous_silence_ = 0;
        }
    }
    KALDI_VLOG(1) << "Vad Total " << chunk_result.size() 
                  << " Silence " << num_sil;
}

int OnlineVadFeaturePipeline::GetVadFeature(int num_frames, 
        Matrix<BaseFloat> *vad_feats) {
    KALDI_ASSERT(vad_feats != NULL);
    int num_valid_frames = num_frames;
    if (input_finished_ || num_frames > NumSpeechFramesReady()) {
        num_valid_frames = NumSpeechFramesReady();
    }

    if (num_valid_frames == 0) return 0;

    vad_feats->Resize(num_valid_frames, AdaptedFeature()->Dim());
    int count = 0;
    while (count < num_valid_frames) {
        if (vad_result_[get_vad_feat_offset_]) {
            SubVector<BaseFloat> row(*vad_feats, count);
            AdaptedFeature()->GetFrame(get_vad_feat_offset_, &row);
            count++;
        }
        get_vad_feat_offset_++;
    }
         
    return num_valid_frames;
}

int OnlineVadFeaturePipeline::NumSpeechFramesReady() const {
    int count = 0; 
    int lookback = input_finished_ ? 0 : lookback_frames_;
    int stop = vad_result_.size() - lookback;
    if (stop <= 0) return 0;
    for (int i = get_vad_feat_offset_; i < stop; i++) {
        if (vad_result_[i]) count++;
    }
    return count;
}








}
}
