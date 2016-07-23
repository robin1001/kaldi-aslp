// aslp-vadbin/aslp-apply-nn-vad-frame.cc

// Copyright 2016 ASLP (Binbin Zhang) 
// Created on 2016-07-09

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "aslp-nnet/nnet-nnet.h"

#include "aslp-vad/nnet-vad.h"

int main(int argc, char *argv[]) {
    try {
        using namespace kaldi;
        using namespace kaldi::aslp_nnet;

        const char *usage =
            "Judge every frame using nn based vad model and output the result\n"
            "Usage:  aslp-apply-nn-vad-frame [options] <nnet-in> <feature-rspecifier> <result-wspecifier>\n"
            "e.g.: aslp-apply-nn-vad-frame nnet.in scp:test.scp ark:result.ark\n";

        ParseOptions po(usage);
        NnetVadOptions nnet_vad_opts;
        nnet_vad_opts.Register(&po);

        po.Read(argc, argv);

        if (po.NumArgs() != 3) {
            po.PrintUsage();
            exit(1);
        }

        std::string nnet_rxfilename = po.GetArg(1),
            feature_rspecifier = po.GetArg(2),
            result_wspecifier = po.GetArg(3);

        Nnet nnet; 
        {
            bool binary_read;
            Input ki(nnet_rxfilename, &binary_read);
            nnet.Read(ki.Stream(), binary_read);
        }

        SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
        Int32VectorWriter writer(result_wspecifier);

        int32 num_done = 0, num_err = 0;
        NnetVad nnet_vad(nnet, nnet_vad_opts);

        for (; !feature_reader.Done(); feature_reader.Next()) {
            std::string key = feature_reader.Key();
            KALDI_VLOG(2) << "Processing " << key;
            const Matrix<BaseFloat> &mat = feature_reader.Value();
            const std::vector<float> &sil_score = nnet_vad.Score(mat);
            std::vector<int> result(sil_score.size());
            for (int i = 0; i < result.size(); i++) {
                result[i] = (sil_score[i] > nnet_vad_opts.sil_thresh) ? 0:1; 
            }
            writer.Write(key, result);
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

