// aslp-vad/aslp-eval-nn-vad.cc

// Copyright 2016 ASLP (Binbin Zhang) 
// Created on 2016-04-25

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "aslp-nnet/nnet-nnet.h"

#include "aslp-vad/roc.h"

int main(int argc, char *argv[]) {
    try {
        using namespace kaldi;
        using namespace kaldi::aslp_nnet;

        const char *usage =
            "Eval vad nn model\n"
            "Usage:  aslp-eval-nn-vad [options] <nnet-in> <feature-rspecifier> <ali-rspecifier>\n"
            "e.g.: aslp-eval-nn-vad nnet.in scp:test.scp ark:test.ark\n";

        ParseOptions po(usage);
        RocSetOptions roc_opts;
        roc_opts.Register(&po);

        po.Read(argc, argv);

        if (po.NumArgs() != 3) {
            po.PrintUsage();
            exit(1);
        }

        std::string nnet_rxfilename = po.GetArg(1),
            feature_rspecifier = po.GetArg(2),
            alignments_rspecifier = po.GetArg(3);

        Nnet nnet; 
        {
            bool binary_read;
            Input ki(nnet_rxfilename, &binary_read);
            nnet.Read(ki.Stream(), binary_read);
        }


        double tot_frames = 0.0;

        SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
        RandomAccessInt32VectorReader ali_reader(alignments_rspecifier);
        int32 num_done = 0, num_err = 0;
        CuMatrix<BaseFloat> nnet_out;
        Matrix<BaseFloat> nnet_out_host;
        RocSet roc_set(roc_opts);

        for (; !feature_reader.Done(); feature_reader.Next()) {
            std::string key = feature_reader.Key();
            KALDI_VLOG(2) << "Processing " << key;
            if (!ali_reader.HasKey(key)) {
                KALDI_WARN << "file " << key << " do not have aliment";
                num_err++;
                continue;
            }
            const Matrix<BaseFloat> &mat = feature_reader.Value();
            const std::vector<int32> &ali = ali_reader.Value(key);

            int32 num_frames = mat.NumRows();
            KALDI_ASSERT(mat.NumCols() == nnet.InputDim());
            nnet.Feedforward(CuMatrix<BaseFloat>(mat), &nnet_out);

            nnet_out_host.Resize(nnet_out.NumRows(), nnet_out.NumCols());
            nnet_out.CopyToMat(&nnet_out_host);
            for (int i = 0; i < num_frames; i++) {
                roc_set.AddData(nnet_out_host(i, 0), ali[i]);
            }

            tot_frames += num_frames;
            num_done++;
        }
        KALDI_LOG << "Done " << num_done << " files; " << num_err
            << " with errors.";
        roc_set.Report();
        return (num_done != 0 ? 0 : 1);

    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}

