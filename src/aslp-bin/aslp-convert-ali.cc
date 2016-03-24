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

bool ConvertAlignmentToMonoState(const kaldi::TransitionModel &old_trans_model,
                      const kaldi::TransitionModel &new_trans_model,
                      const kaldi::ContextDependencyInterface &new_ctx_dep,
                      const std::vector<int32> &old_alignment,
                      const std::vector<int32> *phone_map,
                      std::vector<int32> *new_alignment) {
  using namespace kaldi;
  KALDI_ASSERT(new_alignment != NULL);
  new_alignment->clear();
  std::vector<std::vector<int32> > split;  // split into phones.
  if (!SplitToPhones(old_trans_model, old_alignment, &split))
    return false;
  std::vector<int32> phones(split.size());
  for (size_t i = 0; i < split.size(); i++) {
    KALDI_ASSERT(!split[i].empty());
    phones[i] = old_trans_model.TransitionIdToPhone(split[i][0]);
  }
  if (phone_map != NULL) {  // Map the phone sequence.
    int32 sz = phone_map->size();
    for (size_t i = 0; i < split.size(); i++) {
      if (phones[i] < 0 || phones[i] >= sz || (*phone_map)[phones[i]] == -1)
        KALDI_ERR << "ConvertAlignment: could not map phone " << phones[i];
      phones[i] = (*phone_map)[phones[i]];
    }
  }
  int32 N = new_ctx_dep.ContextWidth(),
      P = new_ctx_dep.CentralPosition();

  // by starting at -N and going to split.size()+N, we're
  // being generous and not bothering to work out the exact
  // array bounds.
  for (int32 win_start = -N;
      win_start < static_cast<int32>(split.size()+N);
      win_start++) {  // start of a context window.
    int32 central_pos = win_start + P;
    if (static_cast<size_t>(central_pos)  < split.size()) {
      // i.e. central_pos>=0 && central_pos<split.size()
      std::vector<int32> phone_window(N, 0);
      for (int32 offset = 0; offset < N; offset++)
        if (static_cast<size_t>(win_start+offset) < split.size())
          phone_window[offset] = phones[win_start+offset];
      int32 central_phone = phone_window[P];
      int32 num_pdf_classes = new_trans_model.GetTopo().NumPdfClasses(central_phone);
      std::vector<int32> state_seq(num_pdf_classes);  // Indexed by pdf-class
      for (int32 pdf_class = 0; pdf_class < num_pdf_classes; pdf_class++) {
        if (!new_ctx_dep.Compute(phone_window, pdf_class, &(state_seq[pdf_class]))) {
          std::ostringstream ss;
          WriteIntegerVector(ss, false, phone_window);
          KALDI_ERR << "tree did not succeed in converting phone window "<<ss.str();
        }
      }
      for (size_t j = 0; j < split[central_pos].size(); j++) {
        int32 old_tid = split[central_pos][j];
        int32 phone = phones[central_pos];
        int32 pdf_class = old_trans_model.TransitionIdToPdfClass(old_tid);
        int32 hmm_state = old_trans_model.TransitionIdToHmmState(old_tid);
        int32 trans_idx = old_trans_model.TransitionIdToTransitionIndex(old_tid);
        //if (static_cast<size_t>(pdf_class) >= state_seq.size())
        //  KALDI_ERR << "ConvertAlignment: error converting alignment, possibly different topologies?";
        int32 new_pdf = state_seq[0/*pdf_class*/];
        // Attention: here is the difference, our hmm only have one state
        // So hmm_state = 0, and trans_idx = 0
        int32 new_trans_state =
            new_trans_model.TripleToTransitionState(phone, 0 /* hmm_state */, new_pdf);
        int32 new_tid =
            new_trans_model.PairToTransitionId(new_trans_state, 0 /* trans_idx */);
        //new_alignment->push_back(new_pdf);
        new_alignment->push_back(new_tid);
      }
    }
  }
  KALDI_ASSERT(new_alignment->size() == old_alignment.size());
  return true;
}


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Convert alignments from one decision-tree/model to another mono state model\n"
        "Usage:  aslp-convert-ali  [options] old-model new-model new-tree old-alignments-rspecifier new-alignments-wspecifier\n"
        "e.g.: \n"
        " aslp-convert-ali old.mdl new.mdl new.tree ark:old.ali ark:new.ali\n";


    std::string phone_map_rxfilename;
    ParseOptions po(usage);
    po.Register("phone-map", &phone_map_rxfilename,
                "File name containing old->new phone mapping (each line is: "
                "old-integer-id new-integer-id)");

    po.Read(argc, argv);

    if (po.NumArgs() != 5) {
      po.PrintUsage();
      exit(1);
    }

    std::string old_model_filename = po.GetArg(1);
    std::string new_model_filename = po.GetArg(2);
    std::string new_tree_filename = po.GetArg(3);
    std::string old_alignments_rspecifier = po.GetArg(4);
    std::string new_alignments_wspecifier = po.GetArg(5);

    std::vector<int32> phone_map;
    if (phone_map_rxfilename != "") {  // read phone map.
      ReadPhoneMap(phone_map_rxfilename,
                   &phone_map);
    }
    
    SequentialInt32VectorReader alignment_reader(old_alignments_rspecifier);
    Int32VectorWriter alignment_writer(new_alignments_wspecifier);

    TransitionModel old_trans_model;
    ReadKaldiObject(old_model_filename, &old_trans_model);

    TransitionModel new_trans_model;
    ReadKaldiObject(new_model_filename, &new_trans_model);

    if (!(old_trans_model.GetTopo() == new_trans_model.GetTopo()))
      KALDI_WARN << "Toplogies of models are not equal: "
                 << "conversion may not be correct or may fail.";
    
    
    ContextDependency new_ctx_dep;  // the tree.
    ReadKaldiObject(new_tree_filename, &new_ctx_dep);

    int num_success = 0, num_fail = 0;

    for (; !alignment_reader.Done(); alignment_reader.Next()) {
      std::string key = alignment_reader.Key();
      const std::vector<int32> &old_alignment = alignment_reader.Value();
      std::vector<int32> new_alignment;
      if (ConvertAlignmentToMonoState(old_trans_model,
                          new_trans_model,
                          new_ctx_dep,
                          old_alignment,
                          (phone_map_rxfilename != "" ? &phone_map : NULL),
                          &new_alignment)) {
        alignment_writer.Write(key, new_alignment);
        num_success++;
      } else {
        KALDI_WARN << "Could not convert alignment for key " << key
                   <<" (possibly truncated alignment?)";
        num_fail++;
      }
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


