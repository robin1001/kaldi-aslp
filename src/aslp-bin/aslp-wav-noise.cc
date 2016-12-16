// featbin/wav-reverberate.cc

// Copyright 2015  Tom Ko
// Copyright 2016  ASLP (Zhang Binbin)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <time.h>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/wave-reader.h"
#include "feat/signal.h"

namespace kaldi {

// Randomly add
void AddVectorsOfUnequalLength(const Vector<BaseFloat> &noise, Vector<BaseFloat> *signal) {
  if (noise.Dim() < signal->Dim()) {
    for (int32 po = 0; po < signal->Dim(); po += noise.Dim()) {
      int32 block_length = noise.Dim();
      if (signal->Dim() - po < block_length) block_length = signal->Dim() - po;
      signal->Range(po, block_length).AddVec(1.0, noise.Range(0, block_length));
    }
  } else {
    // noise length bigger than signal, randomly select start position
    int start_thresh = noise.Dim() - signal->Dim(); 
    int start = rand() % start_thresh;
    signal->AddVec(1.0, noise.Range(start, signal->Dim()));
    KALDI_LOG << "Add noise (" << start << ", " << start + signal->Dim() << ")";
  }
}

void AddNoise(BaseFloat snr_db, Vector<BaseFloat> *noise,
              Vector<BaseFloat> *signal) {
  KALDI_ASSERT(noise->Dim() > 0);
  float input_power = VecVec(*signal, *signal) / signal->Dim();
  float noise_power = VecVec(*noise, *noise) / noise->Dim();
  float scale_factor = sqrt(pow(10, -snr_db / 10) * input_power / noise_power);
  noise->Scale(scale_factor);
  KALDI_VLOG(1) << "Noise signal is being scaled with " << scale_factor
      << " to generate output with SNR " << snr_db << "db\n";
  
  AddVectorsOfUnequalLength(*noise, signal);
}

}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Add noise to the corresponding wav file\n"
        "(specified by corresponding files).\n"
        "Usage:  aslp-wav-onise [options...] <wav-in-rxfilename> "
        "<noise-rxfilename> <wav-out-wxfilename>\n"
        "e.g.\n"
        "aslp-wav-noise input.wav noise.wav output.wav\n";

    ParseOptions po(usage);
    BaseFloat snr_db = 20;
    bool multi_channel_output = false;
    int32 input_channel = 0;
    int32 noise_channel = 0;
    bool normalize_output = true;
    BaseFloat volume = 0;
    int seed = 0;

    po.Register("multi-channel-output", &multi_channel_output,
                "Specifies if the output should be multi-channel or not");
    po.Register("input-wave-channel", &input_channel,
                "Specifies the channel to be used from input as only a "
                "single channel will be used to generate reverberated output");
    po.Register("noise-channel", &noise_channel,
                "Specifies the channel of the noise file, "
                "it will only be used when multi-channel-output is false");
    po.Register("snr-db", &snr_db,
                "Desired SNR(dB) of the output");
    po.Register("normalize-output", &normalize_output,
                "If true, then after reverberating and "
                "possibly adding noise, scale so that the signal "
                "energy is the same as the original input signal.");
    po.Register("volume", &volume,
                "If nonzero, a scaling factor on the signal that is applied "
                "after adding noise. "
                "If you set this option to a nonzero value, it will be as"
                "if you had also specified --normalize-output=false.");
    po.Register("seed", &seed,
                "seed for random");

    if (seed == 0) {
        srand(time(0));
    } else {
        srand(seed);
    }

    po.Read(argc, argv);
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    if (multi_channel_output) {
      if (noise_channel != 0)
        KALDI_WARN << "options for --noise-channel are ignored"
                      "as --multi-channel-output is true.";
    }

    std::string input_wave_file = po.GetArg(1);
    std::string noise_file = po.GetArg(2);
    std::string output_wave_file = po.GetArg(3);

    WaveData input_wave;
    {
      Input ki(input_wave_file);
      input_wave.Read(ki.Stream());
    }

    const Matrix<BaseFloat> &input_matrix = input_wave.Data();
    BaseFloat samp_freq_input = input_wave.SampFreq();
    int32 num_samp_input = input_matrix.NumCols(),  // #samples in the input
          num_input_channel = input_matrix.NumRows();  // #channels in the input
    KALDI_VLOG(1) << "sampling frequency of input: " << samp_freq_input
                  << " #samples: " << num_samp_input
                  << " #channel: " << num_input_channel;
    KALDI_ASSERT(input_channel < num_input_channel);

    Matrix<BaseFloat> noise_matrix;
    WaveData noise_wave;
    {
      Input ki(noise_file);
      noise_wave.Read(ki.Stream());
    }
    noise_matrix = noise_wave.Data();
    BaseFloat samp_freq_noise = noise_wave.SampFreq();
    int32 num_samp_noise = noise_matrix.NumCols(),
          num_noise_channel = noise_matrix.NumRows();
    KALDI_VLOG(1) << "sampling frequency of noise: " << samp_freq_noise
                  << " #samples: " << num_samp_noise
                  << " #channel: " << num_noise_channel;
    KALDI_ASSERT(noise_channel < num_noise_channel);

    int32 num_output_channels = (multi_channel_output ? num_noise_channel: 1);
    Matrix<BaseFloat> out_matrix(num_output_channels, num_samp_input);

    for (int32 output_channel = 0; output_channel < num_output_channels; output_channel++) {
      Vector<BaseFloat> input(num_samp_input);
      input.CopyRowFromMat(input_matrix, input_channel);
      float power_before_noise = VecVec(input, input) / input.Dim();

      Vector<BaseFloat> noise(noise_matrix.NumCols());
      int32 this_noise_channel = (multi_channel_output ? output_channel : noise_channel);
      noise.CopyRowFromMat(noise_matrix, this_noise_channel);

      AddNoise(snr_db, &noise, &input);

      float power_after_noise = VecVec(input, input) / input.Dim();

      if (volume > 0)
        input.Scale(volume);
      else if (normalize_output)
        input.Scale(sqrt(power_before_noise / power_after_noise));

      out_matrix.CopyRowFromVec(input, output_channel);
    }

    WaveData out_wave(samp_freq_input, out_matrix);
    Output ko(output_wave_file, false);
    out_wave.Write(ko.Stream());

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

