// aslp-bin/aslp-txt-to-matrix.cc

// Copyright 2016 ASLP (Binbin Zhang)
// Created on 2016-05-11


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"

static void WriteTxtMatrix(std::string key, const kaldi::Matrix<float> &mat,
                          std::string out_dir) {
    using namespace kaldi;
    std::string out_file_name = out_dir + "/" + key;
    FILE *fp = fopen(out_file_name.c_str(), "w");
    if (fp == NULL) {
        perror(out_file_name.c_str());
        exit(1);
    }
    for (int i = 0; i < mat.NumRows(); i++) {
        for (int j = 0; j < mat.NumCols(); j++) {
            fprintf(fp, "%f ", mat(i, j));
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

int main(int argc, char *argv[]) {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    try {
        const char *usage =
            "Converts kaldi matrix to txt matrix representation for TTS\n"
            "Usage:  aslp-matrix-to-txt [options] <matrix-wspecifier> <out-dir>\n"
            "e.g.: \n"
            " aslp-matrix-to-txt ark:matrix.ark data/param\n";
        ParseOptions po(usage);

        po.Read(argc, argv);

        if (po.NumArgs() != 2) {
            po.PrintUsage();
            exit(1);
        }

        std::string matrix_rspecifier = po.GetArg(1),
            out_dir = po.GetArg(2);

        SequentialBaseFloatMatrixReader reader(matrix_rspecifier);

        int32 num_done = 0;
        for (; !reader.Done(); reader.Next()) {
            std::string key = reader.Key();
            const Matrix<BaseFloat> &mat = reader.Value();
            WriteTxtMatrix(key, mat, out_dir);
            num_done++;
        }
        KALDI_LOG << "Converted " << num_done << " kaldi matrix to tts txt matrix";
    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}


