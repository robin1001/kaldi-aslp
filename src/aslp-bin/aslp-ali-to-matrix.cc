// aslp-bin/aslp-ali-to-matrix.cc

// Copyright 2009-2011  Microsoft Corporation
// Copyright 2016 ASLP (Binbin Zhang)
// Created on 2016-05-03


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"

int main(int argc, char *argv[]) {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    try {
        const char *usage =
            "Converts alignments or id input to 1-hot matrix representation\n"
            "Usage:  aslp-ali-to-matrix [options] <alignments-rspecifier> <matrix-wspecifier>\n"
            "e.g.: \n"
            " aslp-ali-to-matrix ark:1.ali ark:matrix.ark \n";
        ParseOptions po(usage);

        int dict_size = 0;
        po.Register("dict-size", &dict_size, "Size of dict," 
                "which decide the number of columns of ouput matrix");

        po.Read(argc, argv);

        if (po.NumArgs() != 2) {
            po.PrintUsage();
            exit(1);
        }
        KALDI_ASSERT(dict_size > 0);

        std::string alignments_rspecifier = po.GetArg(1),
            matrix_wspecifier = po.GetArg(2);

        SequentialInt32VectorReader reader(alignments_rspecifier);
        BaseFloatMatrixWriter writer(matrix_wspecifier);

        int32 num_done = 0;
        for (; !reader.Done(); reader.Next()) {
            std::string key = reader.Key();
            std::vector<int32> alignment = reader.Value();
            Matrix<BaseFloat> one_hot_mat(alignment.size(), dict_size);
            for (int i = 0; i < alignment.size(); i++) {
                int id = alignment[i];
                if (id < 0 || id >= dict_size) {
                    KALDI_ERR << "ali or 1-hot must greater than 0 or less than "
                              << dict_size 
                              << " but got " << id;
                }
                one_hot_mat(i, id) = 1.0;
            }
            writer.Write(key, one_hot_mat);
            num_done++;
        }
        KALDI_LOG << "Converted " << num_done << " alignments to pdf sequences.";
    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}


