// bin/convert-ali.cc

// Copyright 2009-2011  Microsoft Corporation
//                2013  Johns Hopkins University (author: Daniel Povey)

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
#include "gmm/am-diag-gmm.h"
#include "hmm/hmm-topology.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "hmm/tree-accu.h" // for ReadPhoneMap
#include "util/simple-io-funcs.h"


void KwsReadPhoneMap(std::string phone_map_rxfilename,
                  std::vector<int32> *phone_map) {
  using namespace kaldi;
  phone_map->clear();
  // phone map file has format e.g.:
  // 1 1
  // 2 1
  // 3 2
  // 4 2
  std::vector<std::vector<int32> > vec;  // vector of vectors, each with two elements
  // (if file has right format). first is old phone, second is new phone
  if (!ReadIntegerVectorVectorSimple(phone_map_rxfilename, &vec))
    KALDI_ERR << "Error reading phone map from " << phone_map_rxfilename;
  for (size_t i = 0; i < vec.size(); i++) {
    if (vec[i].size() != 2 || vec[i][0]<=0 || vec[i][1]<0 ||
       (vec[i][0]<static_cast<int32>(phone_map->size()) &&
        (*phone_map)[vec[i][0]] != -1))
      KALDI_ERR << "Error reading phone map from "
                 << "(bad line " << i << ")";
    if (vec[i][0]>=static_cast<int32>(phone_map->size()))
      phone_map->resize(vec[i][0]+1, -1);
    KALDI_ASSERT((*phone_map)[vec[i][0]] == -1);
    (*phone_map)[vec[i][0]] = vec[i][1];
  }
  if (phone_map->empty()) {
    KALDI_ERR << "Read empty phone map from " << phone_map_rxfilename;
  }
}

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Convert phone alignments kws phone alignment\n"
        "Usage:  aslp-kws-convert-phone-ali [options] phone_map_rxfilename old-alignments-rspecifier new-alignments-wspecifier\n"
        "e.g.: \n"
        " aslp-kws-convert-phone-ali phone.map ark:old.ali ark:new.ali\n";

    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string phone_map_rxfilename = po.GetArg(1);
    std::string old_alignments_rspecifier = po.GetArg(2);
    std::string new_alignments_wspecifier = po.GetArg(3);

    std::vector<int32> phone_map;
    KwsReadPhoneMap(phone_map_rxfilename, &phone_map);
    
    SequentialInt32VectorReader alignment_reader(old_alignments_rspecifier);
    Int32VectorWriter alignment_writer(new_alignments_wspecifier);

    int num_success = 0, num_fail = 0;

    for (; !alignment_reader.Done(); alignment_reader.Next()) {
      std::string key = alignment_reader.Key();
      const std::vector<int32> &old_alignment = alignment_reader.Value();
      std::vector<int32> new_alignment(old_alignment.size(), 0);
      for (int i = 0; i < old_alignment.size(); i++) {
        KALDI_ASSERT(old_alignment[i] < phone_map.size());
        new_alignment[i] = phone_map[old_alignment[i]];
      }
      alignment_writer.Write(key, new_alignment);
      num_success++;
    }

    KALDI_LOG << "Succeeded converting alignments for " << num_success
              <<" files, failed for " << num_fail;

    if (num_success != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


