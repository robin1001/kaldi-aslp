// aslp-vad/aslp-apply-nn-vad.cc

// Copyright 2016 ASLP (Binbin Zhang) 
// Created on 2016-04-27
#include <fstream>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/wave-reader.h"
#include "aslp-nnet/nnet-nnet.h"

#include "aslp-vad/nnet-vad.h"

int main(int argc, char *argv[]) {
    try {
        using namespace kaldi;
        using namespace kaldi::aslp_nnet;

        const char *usage =
            "apply nnet based vad for every wav file in raw-wav-rspecifier, and"
            "the result vad wav file will be stored vad-wav-rspecifier"
            "namely that the converted filename is already assigned\n"
            "Usage:  aslp-apply-nn-vad [options] <nnet-in> <feat-rspecifier>"
            "<raw-wav-rspecifier> <vad-wav-rspecifier\n"
            "e.g.: aslp-apply-nn-vad nnet.in scp:feat.scp scp:raw_wav.scp scp:vad_wav.scp\n";

        ParseOptions po(usage);
        VadOptions vad_opts;
        vad_opts.Register(&po);
        NnetVadOptions nnet_vad_opts;
        nnet_vad_opts.Register(&po);

        po.Read(argc, argv);

        if (po.NumArgs() != 4) {
            po.PrintUsage();
            exit(1);
        }

        std::string nnet_rxfilename = po.GetArg(1),
            feature_rspecifier = po.GetArg(2),
            raw_wav_rspecifier = po.GetArg(3),
            vad_wav_rspecifier = po.GetArg(4);

        Nnet nnet; 
        {
            bool binary_read;
            Input ki(nnet_rxfilename, &binary_read);
            nnet.Read(ki.Stream(), binary_read);
        }

        NnetVad nnet_vad(nnet, nnet_vad_opts, vad_opts);

        SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
        SequentialTableReader<WaveHolder> raw_wav_reader(raw_wav_rspecifier);
        SequentialTokenReader vad_wav_reader(vad_wav_rspecifier);

        int32 num_done = 0, num_err = 0;
        Vector<BaseFloat> vad_wav;

        for (; !feature_reader.Done(); feature_reader.Next(), 
                                       raw_wav_reader.Next(),
                                       vad_wav_reader.Next()) {
            std::string key = feature_reader.Key();
            KALDI_VLOG(2) << "Processing " << key;
            if (key != raw_wav_reader.Key() || key != vad_wav_reader.Key()) {
                KALDI_ERR << "Feat and wav are not in the same order"
                          << " feat " << key
                          << " wav " << raw_wav_reader.Key()
                          << " vad wav " << vad_wav_reader.Key();
            }

            const Matrix<BaseFloat> &mat = feature_reader.Value();
            const WaveData &wave_data = raw_wav_reader.Value();
            const std::string &save_wav_file = vad_wav_reader.Value();
            // Always select channel 0
            SubVector<BaseFloat> raw_wav(wave_data.Data(), 0);
            bool is_ok = nnet_vad.DoVad(raw_wav, mat, &vad_wav);
            if (!is_ok) {
                KALDI_WARN << key << " have no vad result ";
                continue;
            }
            // Only keep one channel
            Matrix<BaseFloat> save_mat(1, vad_wav.Dim());
            save_mat.Row(0).CopyFromVec(vad_wav);
            WaveData vad_wav_data(vad_opts.samp_freq, save_mat);
            // Write file
            std::ofstream out_stream(save_wav_file, std::ofstream::binary);
            vad_wav_data.Write(out_stream);
            out_stream.close();
            num_done++;
        }

        KALDI_LOG << "Done " << num_done << " files; " << num_err
            << " with errors.";
        return (num_done != 0 ? 0 : 1);

    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}

