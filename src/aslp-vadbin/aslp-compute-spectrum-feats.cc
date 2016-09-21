// aslp-vadbinbin/aslp-compute-vad-feats.cc

/* Created on 2016-07-12
 * Author: Zhang Binbin
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/wave-reader.h"

#include "aslp-vad/feature-spectrum.h"


int main(int argc, char *argv[]) {
    try {
        using namespace kaldi;
        const char *usage =
            "Create serveral feature for vad (harmonicity, clarity, inter spectrum flatness, periodicity).\n"
            "Usage:  aslp-compute-vad-feats [options...] <wav-rspecifier> <feats-wspecifier>\n";

        // construct all the global objects
        ParseOptions po(usage);
        SpectrumFeatOptions vad_feat_opts;
        int32 channel = -1;
        BaseFloat min_duration = 0.0;

        // Register the option struct
        vad_feat_opts.Register(&po);
        // Register the options
        po.Register("channel", &channel, "Channel to extract (-1 -> expect mono, 0 -> left, 1 -> right)");
        po.Register("min-duration", &min_duration, "Minimum duration of segments to process (in seconds).");

        po.Read(argc, argv);

        if (po.NumArgs() != 2) {
            po.PrintUsage();
            exit(1);
        }

        std::string wav_rspecifier = po.GetArg(1);
        std::string output_wspecifier = po.GetArg(2);

        SpectrumFeat vad_feat(vad_feat_opts);

        SequentialTableReader<WaveHolder> reader(wav_rspecifier);
        BaseFloatMatrixWriter kaldi_writer(output_wspecifier);  // typedef to TableWriter<something>.

        int32 num_utts = 0, num_success = 0;
        for (; !reader.Done(); reader.Next()) {
            num_utts++;
            std::string utt = reader.Key();
            const WaveData &wave_data = reader.Value();
            if (wave_data.Duration() < min_duration) {
                KALDI_WARN << "File: " << utt << " is too short ("
                    << wave_data.Duration() << " sec): producing no output.";
                continue;
            }
            int32 num_chan = wave_data.Data().NumRows(), this_chan = channel;
            {  // This block works out the channel (0=left, 1=right...)
                KALDI_ASSERT(num_chan > 0);  // should have been caught in
                // reading code if no channels.
                if (channel == -1) {
                    this_chan = 0;
                    if (num_chan != 1)
                        KALDI_WARN << "Channel not specified but you have data with "
                            << num_chan  << " channels; defaulting to zero";
                } else {
                    if (this_chan >= num_chan) {
                        KALDI_WARN << "File with id " << utt << " has "
                            << num_chan << " channels but you specified channel "
                            << channel << ", producing no output.";
                        continue;
                    }
                }
            }
            if (vad_feat_opts.frame_opts.samp_freq != wave_data.SampFreq())
                KALDI_ERR << "Sample frequency mismatch: you specified "
                    << vad_feat_opts.frame_opts.samp_freq << " but data has "
                    << wave_data.SampFreq() << " (use --sample-frequency "
                    << "option).  Utterance is " << utt;

            SubVector<BaseFloat> waveform(wave_data.Data(), this_chan);
            Matrix<BaseFloat> features;
            try {
                vad_feat.Compute(waveform, &features);
            } catch (...) {
                KALDI_WARN << "Failed to compute features for utterance "
                    << utt;
                continue;
            }
            kaldi_writer.Write(utt, features);
            if (num_utts % 10 == 0) {
                KALDI_VLOG(1) << "Processed " << num_utts << " utterances";
            }
            KALDI_VLOG(2) << "Processed features for key " << utt;
            num_success++;
        }
        KALDI_LOG << " Done " << num_success << " out of " << num_utts
            << " utterances.";
        return (num_success != 0 ? 0 : 1);
    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
    return 0;

}


