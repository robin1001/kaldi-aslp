// aslp-online/vad.cc
// hcq

#include "aslp-online/vad.h"

namespace kaldi {
namespace aslp_online {

using namespace kaldi;
using std::vector;

Vad::Vad(const VadOptions &config, WavProvider *audio_provider)
  : state_(kSilence), silence_frame_cnt_(0),
    config_(config), audio_provider_(audio_provider),
    silence_detected_(false), audio_received_(0.0)
{
  npoints_per_frame_ = (config_.frame_length_ms * config_.samp_freq) / 1000;
  nframes_silence_trigger_ = config_.silence_trigger_threshold_ms / 
                             config_.frame_length_ms;
  nframes_left_context_ = config_.left_context_length_ms / 
                          config_.frame_length_ms;
  npoints_left_context_ = nframes_left_context_ * npoints_per_frame_;
  nframes_endpoint_trigger_ = config_.endpoint_trigger_threshold_ms / 
                              config_.frame_length_ms;
}

int32 Vad::ReadSpeech(int32 required, vector<BaseFloat> *data)
{
  KALDI_ASSERT(required % npoints_per_frame_ == 0);
  data->clear();
  int silence_frames = 0;
  silence_detected_ = false;
  while (required > 0) {
    if (audio_provider_->Done()) 
      break;
    if (silence_frames >= nframes_endpoint_trigger_) {
      silence_detected_ = true;
      break;
    }
    if (processed_data_.size() <= npoints_left_context_) {
      ProcessOneBlock();
    }
    if (processed_data_.size() > npoints_left_context_) {
      while (is_speech_.size() > nframes_left_context_ && !is_speech_.front()) {
        KALDI_ASSERT(is_speech_.size() * npoints_per_frame_ == 
                     processed_data_.size());
        silence_frames++;
        is_speech_.pop_front();
        processed_data_.erase(processed_data_.begin(), 
                              processed_data_.begin() + npoints_per_frame_);
      }
      if (is_speech_.size() > nframes_left_context_) {
        KALDI_ASSERT(is_speech_.front() == true);
        silence_detected_ = false;
        silence_frames = 0;
        is_speech_.pop_front();
        data->insert(data->end(), processed_data_.begin(),
                     processed_data_.begin() + npoints_per_frame_);
        processed_data_.erase(processed_data_.begin(),
                              processed_data_.begin() + npoints_per_frame_);
        required -= npoints_per_frame_;
        KALDI_ASSERT(is_speech_.size() * npoints_per_frame_ ==
                     processed_data_.size());
      }
    }
  } // while required > 0
  if (audio_provider_->Done()) {
    while (required > 0) {
      if (processed_data_.empty()) 
        break;
      while (!is_speech_.empty() && !is_speech_.front()) {
        is_speech_.pop_front();
        processed_data_.erase(processed_data_.begin(),
                              processed_data_.begin() + npoints_per_frame_);
      }
      if (!is_speech_.empty()) {
        KALDI_ASSERT(is_speech_.front() == true);
        is_speech_.pop_front();
        data->insert(data->end(), processed_data_.begin(),
                     processed_data_.begin() + npoints_per_frame_);
        processed_data_.erase(processed_data_.begin(),
                              processed_data_.begin() + npoints_per_frame_);
        required -= npoints_per_frame_;
        KALDI_ASSERT(is_speech_.size() * npoints_per_frame_ ==
                     processed_data_.size());
      }
    }
  } // if audio_provider_->Done()
  return data->size();
}

void Vad::ProcessOneBlock()
{
  int audio_chunk = npoints_per_frame_ * 50;
  int num = audio_provider_->ReadAudio(audio_chunk, &raw_audio_);
  audio_received_ += num / config_.samp_freq;

  if (num < audio_chunk) {
    raw_audio_.insert(raw_audio_.end(), audio_chunk - num, 0);
  }
  KALDI_ASSERT(raw_audio_.size() == audio_chunk);

  processed_data_.insert(processed_data_.end(), 
                         raw_audio_.begin(), 
                         raw_audio_.end());
  int nframes = audio_chunk / npoints_per_frame_;
  for (int i = 0; i < nframes; i++) {
    is_speech_.push_back(VadOneFrame(i));
  }
  KALDI_ASSERT(is_speech_.size() * npoints_per_frame_ == 
               processed_data_.size());
  ProcessLeftContext();
}

bool Vad::VadOneFrame(int32 frame)
{
  double energy = FrameEnergy(frame);
  switch (state_) {
  case kSpeech:
    if (energy < config_.vad_energy_threshold) {
      state_ = kSpeech2Silence;
      silence_frame_cnt_ = 0;
    }
    break;
  case kSpeech2Silence:
    if (energy < config_.vad_energy_threshold) {
      silence_frame_cnt_++;
      if (silence_frame_cnt_ >= nframes_silence_trigger_) {
        state_ = kSilence;
      }
    } else {
      state_ = kSpeech;
    }
    break;
  case kSilence:
    if (energy >= config_.vad_energy_threshold) {
      state_ = kSpeech;
    }
    break;
  default:
    KALDI_ERR << "Vad::VadOneFrame(): Unknown state";
  }
  return state_ != kSilence;
}

double Vad::FrameEnergy(int32 frame) const
{
  double energy = 0.0;
  int32 index = frame * npoints_per_frame_;
  for (int i = 0; i < npoints_per_frame_; i++, index++) {
    energy += raw_audio_[index] * raw_audio_[index] / npoints_per_frame_;
  }
  return energy;
}

void Vad::ProcessLeftContext()
{
  for (size_t i = 0; i+1 < is_speech_.size(); i++) {
    if (is_speech_.at(i) == false && is_speech_.at(i+1) == true) {
      for (size_t j = 0; j < npoints_left_context_ && i >= j; j++) {
        is_speech_[i-j] = true;
      }
    }
  }
}

} // end of namespace aslp_online
} // namespace kaldi
