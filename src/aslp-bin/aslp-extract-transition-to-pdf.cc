// aslp-bin/aslp-extract-transition-to-pdf
// Refer https://github.com/ling0322/pocketkaldi/blob/master/tool/extract_id2pdf.cc

// Copyright 2009-2011 Microsoft Corporation

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#define HAVE_ATLAS 1

#include "hmm/transition-model.h"
#include "fst/fstlib.h"
#include "util/common-utils.h"

int main(int argc, char *argv[]) {
    try {
        using namespace kaldi;
        typedef kaldi::int32 int32;
        using fst::SymbolTable;
        using fst::VectorFst;
        using fst::StdArc;

        const char *usage =
            "Extracts transition_id to pdf-id map from transition model \n"
            "Usage:  extract_id2pdf <transition-model or model file>\n"
            "e.g.: \n"
            " extract_id2pdf final.mdl\n";


        ParseOptions po(usage);
        po.Read(argc, argv);
        if (po.NumArgs() != 1) {
            po.PrintUsage();
            exit(1);
        }
        std::string transition_model_rxfilename = po.GetArg(1);


        TransitionModel trans_model;
        ReadKaldiObject(transition_model_rxfilename, &trans_model);
        int num_transition_ids = trans_model.NumTransitionIds();
        int num_pdfs = trans_model.NumPdfs();
        printf("%d\n", num_pdfs);
        printf("%d\n", num_transition_ids);
        for (int tid = 0; tid <= num_transition_ids; ++tid) {
            int pdf_id = trans_model.TransitionIdToPdf(tid);
            printf("%d %d\n", tid, pdf_id);
        }
    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}

