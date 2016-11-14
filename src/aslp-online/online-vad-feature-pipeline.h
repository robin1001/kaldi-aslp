/* 
 * Created on 2016-11-13
 * Author: Zhang Binbin
 */

#ifndef ASLP_ONLINE_ONLINE_VAD_FEATURE_PIPELINE_H_
#define ASLP_ONLINE_ONLINE_VAD_FEATURE_PIPELINE_H_

#include "aslp-vad/nnet-vad.h"

#include "aslp-online/wav-provider.h"
#include "aslp-online/online-feature-pipeline.h"

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
                    "Length of consecutive silence to regard as endpoint in speech."); 
    }
};

class OnlineVadFeaturePipeline: public OnlineFeaturePipeline {
public:
    typedef aslp_nnet::Nnet Nnet;
    explicit OnlineVadFeaturePipeline(const Nnet &vad_nnet,
            const OnlineNnetVadOptions &online_vad_cfg, 
            const OnlineFeaturePipelineConfig &cfg):
        OnlineFeaturePipeline(cfg),
        get_raw_feat_offset_(0),
        get_vad_feat_offset_(0),
        endpoint_detected_(false),
        input_finished_(false), 
        num_continuous_silence_(0),
        nnet_vad_(vad_nnet, online_vad_cfg), 
        online_vad_cfg_(online_vad_cfg) {

        endpoint_frames_ = online_vad_cfg_.endpoint_trigger_threshold_ms / 
                           online_vad_cfg_.frame_length_ms;
        lookback_frames_ = online_vad_cfg_.lookback_ms / 
                           online_vad_cfg_.frame_length_ms;
    }

    virtual void AcceptWaveform(BaseFloat sampling_rate,
            const VectorBase<BaseFloat> &waveform);

    virtual void InputFinished() {
        OnlineFeaturePipeline::InputFinished();
        input_finished_ = true;
    }

    int GetVadFeature(int num_frames, 
            Matrix<BaseFloat> *vad_feats); 

    int NumSpeechFramesReady() const;

    bool EndpointDetected() const {
        return endpoint_detected_;
    }

    float AudioReceived() const {
        int num_frames = input_finished_ ? 
            vad_result_.size() : vad_result_.size() - lookback_frames_;
        return (online_vad_cfg_.frame_length_ms * num_frames) / 1000.0;
    }

private:
    int GetRawFeature(Matrix<BaseFloat> *feats);
    void Vad();

    int get_raw_feat_offset_, get_vad_feat_offset_;
    bool endpoint_detected_;
    bool input_finished_;
    int num_continuous_silence_;
    std::vector<bool> vad_result_;
    int lookback_frames_, endpoint_frames_;
    NnetVad nnet_vad_;
    OnlineNnetVadOptions online_vad_cfg_;
};


}
}

#endif

