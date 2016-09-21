// aslp-vadbin/aslp-eval-nn-vad-boundary.cc

// Copyright 2016 ASLP (Binbin Zhang) 
// Created on 2016-07-05

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "aslp-nnet/nnet-nnet.h"

#include "aslp-vad/nnet-vad.h"
#include "aslp-vad/boundary-tool.h"

int main(int argc, char *argv[]) {
    try {
        using namespace kaldi;
        using namespace kaldi::aslp_nnet;

        const char *usage =
            "Eval vad nn model boundary accuracy\n"
            "Usage:  aslp-eval-nn-vad-boundary [options] <nnet-in> <feature-rspecifier> <ali-rspecifier>\n"
            "e.g.: aslp-eval-nn-vad-boundary nnet.in scp:test.scp ark:test.ark\n";

        ParseOptions po(usage);
        NnetVadOptions nnet_vad_opts;
        nnet_vad_opts.Register(&po);

        int context = 10;
        po.Register("context", &context, "context size for evaluate the boundary accuracy");

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

        SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
        RandomAccessInt32VectorReader ali_reader(alignments_rspecifier);
        int32 num_done = 0, num_err = 0;
        CuMatrix<BaseFloat> nnet_out;
        Matrix<BaseFloat> nnet_out_host;
        BoundaryTool boundary_tool(context);

        NnetVad nnet_vad(nnet, nnet_vad_opts);

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
            bool has_voice_frame = nnet_vad.DoVad(mat);
            if (!has_voice_frame) {
                KALDI_WARN << key << " have no vad result ";
                continue;
            }
            // lookback voice frame
            nnet_vad.Lookback();
            const std::vector<bool>& vad_result = nnet_vad.VadResult();
            std::vector<int32> ref_ali(vad_result.size());
            for (int i = 0; i < vad_result.size(); i++) {
                ref_ali[i] = (vad_result[i] ? 1 : 0);
            }
            bool is_ok = boundary_tool.AddData(ali, ref_ali);
            if (!is_ok) num_err++;
            num_done++;
        }
        KALDI_LOG << boundary_tool.Report();
        KALDI_LOG << "Done " << num_done << " files; " << num_err
            << " with errors.";
        return (num_done != 0 ? 0 : 1);

    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}

