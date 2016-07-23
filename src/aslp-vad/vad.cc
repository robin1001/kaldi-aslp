/* Created on 2016-04-27
 * Author: Binbin Zhang
 */

#include "aslp-vad/vad.h"

namespace kaldi {

Vad::Vad(const VadOptions &config): config_(config), state_(kSilence),
    silence_frame_cnt_(0), speech_frame_cnt_(0) {
    nframes_silence_trigger_ = config_.silence_trigger_threshold_ms / 
                                 config_.frame_length_ms;
    nframes_speech_trigger_ = config_.speech_trigger_threshold_ms / 
                                 config_.frame_length_ms;
    nframes_lookback_ = config_.lookback_ms / config_.frame_length_ms;
    num_points_per_frame_ = (config_.frame_length_ms * config_.samp_freq) / 1000;
}

//bool Vad::VadOneFrame(int32 frame) {
//    bool is_silence = IsSilence(frame);
//    switch (state_) {
//        case kSpeech:
//            if (is_silence) {
//                state_ = kSpeech2Silence;
//                silence_frame_cnt_ = 0;
//            }
//            break;
//        case kSpeech2Silence:
//            if (is_silence) {
//                silence_frame_cnt_++;
//                if (silence_frame_cnt_ >= nframes_silence_trigger_) {
//                    state_ = kSilence;
//                }
//            } else {
//                state_ = kSpeech;
//            }
//            break;
//        case kSilence:
//            if (!is_silence) {
//                state_ = kSpeech;
//            }
//            break;
//        default:
//            KALDI_ERR << "Vad::VadOneFrame(): Unknown state";
//    }
//    return state_ != kSilence;
//}

bool Vad::VadOneFrame(int32 frame) {
    bool is_silence = IsSilence(frame);
    switch (state_) {
        case kSpeech:
            if (is_silence) {
                silence_frame_cnt_++;
                if (silence_frame_cnt_ > nframes_silence_trigger_) {
                    state_ = kSilence;
                    speech_frame_cnt_ = 0;
                }
            } else {
                silence_frame_cnt_ = 0;
            }
            break;
        case kSilence:
            if (!is_silence) {
                speech_frame_cnt_++;
                if (speech_frame_cnt_ > nframes_speech_trigger_) {
                    state_ = kSpeech;
                    silence_frame_cnt_ = 0;
                }
            }
            else {
                speech_frame_cnt_ = 0;
            }
            break;
        default:
            KALDI_ERR << "Vad::VadOneFrame(): Unknown state";
    }
    return state_ != kSilence;
}

void Vad::VadAll() {
    for (int i = 0; i < vad_result_.size() ; i++) {
        vad_result_[i] = VadOneFrame(i);
    }
}

void Vad::Lookback() {
    if (nframes_lookback_ > 0) {
        int cur = 0;
        while (cur < vad_result_.size()) {
            // sil, go ahead
            while (cur < vad_result_.size() && !vad_result_[cur]) cur++; 
            // end with silence, no more voice
            if (cur == vad_result_.size()) break;
            // voice start, lookback, assign it to voice(true) 
            int start = std::max(0, cur - nframes_lookback_);
            for (int i = start; i < cur; i++) vad_result_[i] = true; 
            // voice, go ahead
            while (cur < vad_result_.size() && vad_result_[cur]) cur++;
        }
    }
}

}
