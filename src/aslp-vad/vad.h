/* Created on 2016-04-27
 * Author: Binbin Zhang
 */

 /* code reference: vad code of our online-asr project contributed by hechangqing 
  * http://gitlab.npu-aslp.org/online-asr/asr-server/blob/master/decoder/asr-online/vad.cc
  */
#ifndef ASLP_VAD_VAD_H_
#define ASLP_VAD_VAD_H_

#include "base/kaldi-common.h"
#include "itf/options-itf.h"

namespace kaldi {

struct VadOptions {
    BaseFloat samp_freq;       // in Hz
    BaseFloat frame_length_ms; // in milliseconds
    BaseFloat silence_trigger_threshold_ms; // in milliseconds
    BaseFloat speech_trigger_threshold_ms; // in milliseconds
    BaseFloat lookback_ms; // in milliseconds

    VadOptions() : samp_freq(16000),
    frame_length_ms(10.0),
    silence_trigger_threshold_ms(150.0), 
    speech_trigger_threshold_ms(30),
    lookback_ms(0) {}

    std::string Print() {
        std::ostringstream ss;
        ss << "\nsamp_freq: " << samp_freq;
        ss << "\nframe_length_ms: " << frame_length_ms;
        ss << "\nsilence_trigger_threshold_ms: " << silence_trigger_threshold_ms;
        ss << "\nspeech_trigger_threshold_ms: " << silence_trigger_threshold_ms;
        ss << "\nlookback_ms: " << silence_trigger_threshold_ms;
        return ss.str();
    }

    void Register(OptionsItf *po) {
        po->Register("sample-frequency", &samp_freq,
                "Waveform data sample frequency (must match)");
        po->Register("frame-length", &frame_length_ms, "Frame length in milliseconds");
        po->Register("silence-trigger-threshold", &silence_trigger_threshold_ms, 
                "Length of consecutive silence to regard as silence segment, "
                "in milliseconds. This parameter is used in Finite State Machine in the code. "
                "It determines the transition of state from state speech2silence to state silence.");
        po->Register("speech-trigger-threshold", &speech_trigger_threshold_ms, 
                "Length of consecutive speech to regard as voice segment, "
                "in milliseconds. This parameter is used in Finite State Machine in the code. ");
        po->Register("lookback", &lookback_ms, 
                "Lookback length for voice start point, in milliseconds");
    }
};

class Vad {
public:
    Vad(const VadOptions &config); 
    virtual bool IsSilence(int frame) const = 0;
    // sil: false, voice: true
    virtual bool VadOneFrame(int frame);
    virtual void VadAll();
    virtual const std::vector<bool>& VadResult() const {
        return vad_result_;
    }
    virtual void Lookback();
    // DoVad input:wav, feat already prepared out:wav
protected:
    enum { kSilence = 0x00,
           kSpeech2Silence = 0x01,
           kSpeech = 0x02
    } state_;
    int silence_frame_cnt_;
    int speech_frame_cnt_;
    int nframes_silence_trigger_;
    int nframes_speech_trigger_;
    int nframes_lookback_;
    int num_points_per_frame_;
    std::vector<bool> vad_result_; // vad result, voice(true), sil(false)
    const VadOptions &config_;
};

} // namespace kaldi


#endif
