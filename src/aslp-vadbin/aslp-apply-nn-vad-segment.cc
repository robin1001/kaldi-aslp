// aslp-vadbin/aslp-apply-nn-vad-segment.cc

// Copyright 2016 ASLP (Binbin Zhang) 
// Created on 2016-07-07
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
            "Split long sentence by long silence using nn based vad for silence detection"
            "Usage:  aslp-apply-nn-vad-segment [options] <nnet-in> <feat-rspecifier>\n"
            "e.g.: aslp-apply-nn-vad-segment nnet.in scp:feat.scp ark:\n";

        ParseOptions po(usage);
        NnetVadOptions nnet_vad_opts;
        nnet_vad_opts.Register(&po);

        int min_length = 50;
        po.Register("min-length", &min_length, "Minimum length of the voice segment");
        int max_length = 1000;
        po.Register("max-length", &max_length, "Maximum length of the voice segment");

        po.Read(argc, argv);

        if (po.NumArgs() != 2) {
            po.PrintUsage();
            exit(1);
        }

        std::string nnet_rxfilename = po.GetArg(1),
            feature_rspecifier = po.GetArg(2);

        Nnet nnet; 
        {
            bool binary_read;
            Input ki(nnet_rxfilename, &binary_read);
            nnet.Read(ki.Stream(), binary_read);
        }

        NnetVad nnet_vad(nnet, nnet_vad_opts);

        SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);

        int32 num_done = 0, num_err = 0;
        Vector<BaseFloat> vad_wav;

        for (; !feature_reader.Done(); feature_reader.Next()) {
            std::string key = feature_reader.Key();
            KALDI_VLOG(2) << "Processing " << key;

            const Matrix<BaseFloat> &mat = feature_reader.Value();
            bool has_voice_frame = nnet_vad.DoVad(mat);
            if (!has_voice_frame) {
                KALDI_WARN << key << " have no vad result ";
                num_err++;
                continue;
            }
            // lookback voice frame
            nnet_vad.Lookback();
            const std::vector<bool> vad_result = nnet_vad.VadResult();
            //std::cout << key;
            int cur = 0;
            while (cur < vad_result.size()) {
                // silence go ahead
                while (cur < vad_result.size() && !vad_result[cur]) cur++;
                int start = cur;
                while (cur < vad_result.size() && cur - start < max_length && 
                    (cur - start < min_length || vad_result[cur])) cur++;
                int end = cur;
                // end of sentence, no more speech
                if (start == end) continue;
                std::cout << " [ " << start << " " << end << " ]\n";
            }
            //std::cout << std::endl;
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

