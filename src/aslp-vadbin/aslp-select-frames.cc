// aslp-bin/aslp-select-frames.cc

// Copyright    Binbin Zhang
// Created on 2016-04-21

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"
#include "feat/feature-functions.h"


int main(int argc, char *argv[]) {
    try {
        using namespace kaldi;
        using kaldi::int32;

        const char *usage =
            "Select a subset of frames of the input files, based on the select id\n"
            "in vad, we use 0 for unvoiced, 1 for voiced segment\n"
            "Usage: aslp-select-frames [options] <feats-rspecifier> "
            " <ali-rspecifier> <feats-wspecifier>\n"
            "E.g.: aslp-select-frames [options] scp:feats.scp scp:ali.scp ark:-\n";

        ParseOptions po(usage);
        int select_id = 0; // 
        po.Register("select-id", &select_id, "select id for select frames,"
                "voiced(0), unvoiced(1) or other, default 0");
        po.Read(argc, argv);

        if (po.NumArgs() != 3) {
            po.PrintUsage();
            exit(1);
        }

        std::string feat_rspecifier = po.GetArg(1),
            ali_rspecifier = po.GetArg(2),
            feat_wspecifier = po.GetArg(3);

        SequentialBaseFloatMatrixReader feat_reader(feat_rspecifier);
        RandomAccessInt32VectorReader ali_reader(ali_rspecifier);
        BaseFloatMatrixWriter feat_writer(feat_wspecifier);

        int32 num_done = 0, num_err = 0;

        for (;!feat_reader.Done(); feat_reader.Next()) {
            std::string utt = feat_reader.Key();
            const Matrix<BaseFloat> &feat = feat_reader.Value();
            if (feat.NumRows() == 0) {
                KALDI_WARN << "Empty feature matrix for utterance " << utt;
                num_err++;
                continue;
            }
            if (!ali_reader.HasKey(utt)) {
                KALDI_WARN << "No ali input found for utterance " << utt;
                num_err++;
                continue;
            }
            std::vector<int32> alignment = ali_reader.Value(utt);

            if (feat.NumRows() != alignment.size()) {
                KALDI_WARN << "Mismatch in number for frames " << feat.NumRows() 
                    << " for features and ali " << alignment.size()
                    << ", for utterance " << utt;
                num_err++;
                continue;
            }

            int32 dim = 0;
            for (int32 i = 0; i < alignment.size(); i++)
                if (alignment[i] == select_id)
                    dim++;
            if (dim == 0) continue;
            Matrix<BaseFloat> select_feat(dim, feat.NumCols());
            int32 index = 0;
            for (int32 i = 0; i < feat.NumRows(); i++) {
                if (alignment[i] == select_id) {
                    select_feat.Row(index).CopyFromVec(feat.Row(i));
                    index++;
                }
            }
            KALDI_ASSERT(index == dim);
            feat_writer.Write(utt, select_feat);
            num_done++;
        }

        KALDI_LOG << "Done selecting frames; processed "
            << num_done << " utterances, "
            << num_err << " had errors.";
        return (num_done != 0 ? 0 : 1);
    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}


