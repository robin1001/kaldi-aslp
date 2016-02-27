// aslp-bin/aslp-tree-bind-info.cc

// Copyright 2009-2011  Microsoft Corporation, GoVivace Inc.
//                2013  Johns Hopkins University (author: Daniel Povey)
// Copyright 2015  ASLP (author: zhangbinbin)
// Modified on 2015-08-30

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "tree/build-tree-utils.h"
#include "hmm/transition-model.h"
#include "hmm/tree-accu.h"

/** @brief Accumulate tree statistics for decision tree training. The
program reads in a feature archive, and the corresponding alignments,
and generats the sufficient statistics for the decision tree
creation. Context width and central phone position are used to
identify the contexts.Transition model is used as an input to identify
the PDF's and the phones.  */
int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Show triphone bind tree info.\n"
        "Usage:  aslp-tree-bind-info tree num_phones\n"
        " aslp-tree-bind-info exp/tri2/tree 48\n";
    int num_states = 3;
    int num_phones;
    ParseOptions po(usage);
    po.Register("num-states", &num_states,
                "number states in per phone, default: 3");
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string tree_filename = po.GetArg(1);
    ConvertStringToInteger(po.GetArg(2), &num_phones);

    KALDI_LOG << "read tree down " << num_phones;

    ContextDependency ctx_dep;  // the tree.
    ReadKaldiObject(tree_filename, &ctx_dep);
    int pdf_id;
    std::vector<int32> phone_window(3, 0);
    for (int i = 0; i < num_phones; i++) { // middle phone
      phone_window[1] = i;
      for (int j = 0; j < num_phones; j++) { // all left
        phone_window[0] = j;
        for (int k = 0; k < num_phones; k++) { // all right
          phone_window[2] = k;
          for (int n = 0; n < num_states; n++) {
            if (ctx_dep.Compute(phone_window, n, &pdf_id)) {
              fprintf(stderr, "%d-%d+%d %d\t%d\n", j, i, k, n, pdf_id);
            }
            //else {
            //  fprintf(stderr, "%d-%d+%d %d\t%d\n", i, j, k, n, -1);
            //}
          }
        }
      }
    }
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


