// aslp-online/vad.h
// hcq

#ifndef ASLP_ONLINE_VAD_H
#define ASLP_ONLINE_VAD_H

#include <vector>
#include <string>
#include <sstream>
#include <deque>

#include "aslp-online/wav-provider.h"
#include "itf/options-itf.h"

namespace aslp_online {

using namespace kaldi;

struct VadOptions {
  BaseFloat samp_freq;       // in Hz
  BaseFloat frame_length_ms; // in milliseconds
  BaseFloat silence_trigger_threshold_ms; // in milliseconds
  BaseFloat left_context_length_ms; // in milliseconds
  BaseFloat endpoint_trigger_threshold_ms; // in milliseconds
  BaseFloat vad_energy_threshold;

  VadOptions() : samp_freq(16000),
                 frame_length_ms(10.0),
                 silence_trigger_threshold_ms(150.0),
                 left_context_length_ms(100.0),
                 endpoint_trigger_threshold_ms(1000.0),
                 vad_energy_threshold(1500000.0)
  {
  }
  std::string Print() {
    std::ostringstream ss;
    ss << "\nsamp_freq: " << samp_freq;
    ss << "\nframe_length_ms: " << frame_length_ms;
    ss << "\nsilence_trigger_threshold_ms: " << silence_trigger_threshold_ms;
    ss << "\nleft_context_length_ms: " << left_context_length_ms;
    ss << "\nendpoint_trigger_threshold_ms: " << endpoint_trigger_threshold_ms;
    ss << "\nvad_energy_threshold: " << vad_energy_threshold << "\n";
    return ss.str();
  }
  void Register(OptionsItf *po) {
    po->Register("sample-frequency", &samp_freq,
                 "Waveform data sample frequency (must match)");
    po->Register("frame-length", &frame_length_ms, "Frame length in milliseconds");
    po->Register("silence-trigger-threshold", &silence_trigger_threshold_ms, 
                 "Length of consecutive silence to regard as silence segment, "
                 "in milliseconds. This parameter is used in Finite State Machine in the code. It determines the transition of state from state speech2silence to state silence.");
    po->Register("left-context-length", &left_context_length_ms,
                 "Will regard this length of waveform before the voiced segment as voice. We do this because this simple Energy-Based VAD method will cut the begining of the voice especially when the begining phoneme is unvoiced phoneme. In milliseconds");
    po->Register("endpoint-trigger-threshold", &endpoint_trigger_threshold_ms,
                 "Length of consecutive silence to regard as endpoint in speech. ReadSpeech() will return immediately when endpoint is detected.");
    po->Register("vad-energy-threshold", &vad_energy_threshold,
                 "Voice Active Detection energy threshold");
  }
};

class Vad {
 public:

  Vad(const VadOptions &config, WavProvider *audio_provider);
  
  void SetOptions(const VadOptions &config) { config_ = config; }

  ~Vad() { }

  int ReadSpeech(int32 required, std::vector<BaseFloat> *data);
 
  bool SilenceDetected() const { return silence_detected_; }

  double AudioReceived() const { return audio_received_; }
  
  bool Done() const {
    return audio_provider_->Done() && processed_data_.empty();
  }
 
 private:
  
  void ProcessOneBlock();

  bool VadOneFrame(int32 frame);

  void ProcessLeftContext();

  double FrameEnergy(int32 frame) const;

  // Finite State Machince states
  enum { kSilence         = 0x00,
         kSpeech2Silence  = 0x01,
         kSpeech          = 0x02
  } state_;

  int32 silence_frame_cnt_;

  VadOptions config_;
  WavProvider *audio_provider_;

  std::deque<BaseFloat> processed_data_;
  std::deque<bool> is_speech_;

  std::vector<BaseFloat> raw_audio_;

  int32 npoints_per_frame_;
  int32 nframes_silence_trigger_;
  int32 nframes_left_context_;
  int32 npoints_left_context_;

  int32 nframes_endpoint_trigger_;

  bool silence_detected_;

  double audio_received_; // total audio read from audio_provider_ in seconds
};

} // end namespace aslp_online.

#endif // ASLP_ONLINE_VAD_H
