// aslp-bin/aslp-txt-to-matrix.cc

// Copyright 2016 ASLP (Binbin Zhang)
// Created on 2016-05-11


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"

static int ReadTxtMatrix(const char *txt_file, kaldi::Matrix<float> *mat) {
    KALDI_ASSERT(mat != NULL);
    using namespace kaldi;
    FILE *fp = fopen(txt_file, "r");
    if (fp == NULL) {
        perror(txt_file);
        exit(1);
    }
    char line[10240] = {0};
    std::vector<std::vector<float> > mat_vec;
    while (fgets(line, 10240, fp)) {
        std::vector<float> vec_vec;
        SplitStringToFloats(line, " ", true, &vec_vec);
        mat_vec.push_back(vec_vec);
    }
    KALDI_ASSERT(mat_vec.size() > 0);
    int num_rows = mat_vec.size();
    int num_cols = mat_vec[0].size();
    //KALDI_LOG << num_rows << " " << num_cols;
    // Check
    for (int i = 0; i < num_rows; i++) {
        if (mat_vec[i].size() != num_cols) {
            KALDI_ERR << txt_file << " line " << i << " number of fileds are not equal";
        }
    }
    // Convert to matrix
    mat->Resize(num_rows, num_cols);
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < num_cols; j++) {
            (*mat)(i, j) = mat_vec[i][j];
        }
    }
    fclose(fp);
    return num_cols;
}

int main(int argc, char *argv[]) {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    try {
        const char *usage =
            "Converts txt matrix to kaldi matrix representation\n"
            "Usage:  aslp-txt-to-matrix [options] <txt-mat-rspecifier> <matrix-wspecifier>\n"
            "e.g.: \n"
            " aslp-txt-to-matrix ark:txt_mat.list ark:matrix.ark \n";
        ParseOptions po(usage);

        po.Read(argc, argv);

        if (po.NumArgs() != 2) {
            po.PrintUsage();
            exit(1);
        }

        std::string txt_rxfilename = po.GetArg(1),
            matrix_wspecifier = po.GetArg(2);

        SequentialTokenReader reader(txt_rxfilename);
        BaseFloatMatrixWriter writer(matrix_wspecifier);

        Matrix<BaseFloat> mat;
        int32 num_done = 0;
        for (; !reader.Done(); reader.Next()) {
            std::string key = reader.Key();
            const std::string file_name = reader.Value();
            ReadTxtMatrix(file_name.c_str(), &mat);
            writer.Write(key, mat);
            num_done++;
        }
        KALDI_LOG << "Converted " << num_done << " txt file to matrix";
    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}


