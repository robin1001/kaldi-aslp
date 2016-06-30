// aslp-online/online-vad.cc

/* Created on 2016-06-29
 * Author: Binbin Zhang
 */

#include "aslp-online/online-vad.h"

namespace kaldi {
namespace aslp_online {

bool OnlineNnetVad::AcceptFeature(const Matrix<BaseFloat> &feat) {
    sil_score_.resize(feat.NumRows());
    vad_result_.resize(feat.NumRows());
    // Get nnet score
    GetScore(feat);
    // Call DoVad jugde every frame, it change vad_result_
    VadAll();

    // Reset endpoint detect state
    endpoint_detected_ = false;

    // Copy speech frames to buffer
    for (int i = 0; i < vad_result_.size(); i++) {
        if (vad_result_[i]) { 
            // Copy to ping buffer when ping is not full
            // or endpoint not be detected
            if (!endpoint_detected_ && buffer_idx_[cur_buffer_] < buffer_size_) {
                feat_buffer_[cur_buffer_].Row(
                    buffer_idx_[cur_buffer_]).CopyFromVec(feat.Row(i)); 
                buffer_idx_[cur_buffer_]++;
            } 
            // Copy to pong buffer otherwise
            else {
                KALDI_ASSERT(buffer_idx_[PeerBuffer()] < buffer_size_);
                feat_buffer_[PeerBuffer()].Row(
                    buffer_idx_[PeerBuffer()]).CopyFromVec(feat.Row(i));
                buffer_idx_[PeerBuffer()]++;
            }
            num_continuous_silence_ = 0;
        }
        else {
            // silence frame 
            num_continuous_silence_++;
            if (num_continuous_silence_ > num_frames_endpoint_trigger_) {
                KALDI_LOG << "endpoint detected";
                endpoint_detected_ = true;
            }
        }
    }

    // current buffer is full
    if (buffer_idx_[cur_buffer_] == buffer_size_) return true;
    else return false;
}

} // namespace  aslp_online
} // namespace kaldi
