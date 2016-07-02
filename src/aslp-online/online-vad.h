// aslp-online/online-vad.h

/* Created on 2016-06-29
 * Author: Binbin Zhang
 */


#ifndef ASLP_ONLINE_ONLINE_VAD_H_
#define ASLP_ONLINE_ONLINE_VAD_H_

#include <string>
#include <vector>
#include <deque>

#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"
#include "feat/online-feature.h"

#include "aslp-vad/nnet-vad.h"

namespace kaldi {
namespace aslp_online {

struct OnlineNnetVadOptions : public NnetVadOptions {
    // for online endpoint detect
    BaseFloat endpoint_trigger_threshold_ms; // in milliseconds
    OnlineNnetVadOptions(): endpoint_trigger_threshold_ms(1000.0) {}

    void Register(OptionsItf *po) {
        NnetVadOptions::Register(po);
        po->Register("endpoint-trigger-threshold", &endpoint_trigger_threshold_ms,
                    "For online decoding, use long silence to do endpoint detecting"
                    "Length of consecutive silence to regard as endpoint in speech." 
                    "ReadSpeech() will return immediately when endpoint is detected.");
    }
};


class OnlineNnetVad : public NnetVad {
    typedef aslp_nnet::Nnet Nnet;
public:
    // @param chunk_length: accumulate chunk_length seconds feat, and process
    OnlineNnetVad(const Nnet &nnet, 
                  const OnlineNnetVadOptions &online_vad_config, 
                  BaseFloat chunk_length): 
        NnetVad(nnet, online_vad_config), 
        online_vad_config_(online_vad_config),
        num_continuous_silence_(0),
        endpoint_detected_(false) {
        num_frames_endpoint_trigger_ = 
            online_vad_config_.endpoint_trigger_threshold_ms / config_.frame_length_ms;
        buffer_size_ = static_cast<int>(1000 * chunk_length / config_.frame_length_ms);
        feat_buffer_[0].Resize(buffer_size_, nnet.InputDim());
        feat_buffer_[1].Resize(buffer_size_, nnet.InputDim());
        buffer_idx_[0] = 0;
        buffer_idx_[1] = 0;
        cur_buffer_ = 0;
    }
    // Return true if the current feature buffer is full
    bool AcceptFeature(const Matrix<BaseFloat> &feat);
    
    bool EndpointDetected() const {
        return endpoint_detected_;
    }

    int PeerBuffer() {
        return (cur_buffer_ + 1) % 2;
    }

    void SwapBuffer() {
        cur_buffer_ = (cur_buffer_ + 1) % 2;
    }

    // Return num valid frames in the buffer
    //int GetFeature(MatrixBase<BaseFloat> **feat) {
    //    int num_frames = buffer_idx_[cur_buffer_];
    //    KALDI_ASSERT(num_frames < buffer_size_);
    //    *feat = &feat_buffer_[cur_buffer_];
    //    buffer_idx_[cur_buffer_] = 0;
    //    SwapBuffer();
    //    return num_frames;
    //}

    const MatrixBase<BaseFloat> & GetFeature() {
        int num_frames = buffer_idx_[cur_buffer_];
        KALDI_ASSERT(num_frames < buffer_size_);
        MatrixBase<BaseFloat> *feat = &feat_buffer_[cur_buffer_];
        buffer_idx_[cur_buffer_] = 0;
        SwapBuffer();
        return (*feat).RowRange(0, num_frames);
    }

    float AudioReceived() const {
        return (config_.frame_length_ms * num_frames_received_) / 1000.0;
    }

private:
    const OnlineNnetVadOptions &online_vad_config_;
    
    int32 num_frames_endpoint_trigger_; // endpoint detector
    int32 num_continuous_silence_;
    int32 num_frames_received_;
    bool endpoint_detected_;

    Matrix<BaseFloat> feat_buffer_[2];
    int buffer_idx_[2]; // index of current buffer
    int cur_buffer_; //  
    int buffer_size_;
};

} // namespace  aslp_online
} // namespace kaldi

#endif
