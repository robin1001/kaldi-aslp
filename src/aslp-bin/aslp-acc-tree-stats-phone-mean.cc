// aslp-bin/aslp-acc-tree-stats-mean.cc

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
#include "hmm/hmm-utils.h"

namespace kaldi {

static int32 MapPhone(const std::vector<int32> *phone_map,
                      int32 phone) {
  if (phone == 0 || phone_map == NULL) return phone;
  else if (phone < 0 || phone >= phone_map->size()) {
    KALDI_ERR << "Out-of-range phone " << phone << " bad --phone-map option?";
  }
  return (*phone_map)[phone];
}

void AccumulateTreeStatsPhone(const TransitionModel &trans_model,
                         BaseFloat var_floor,
                         int N,  // context window size.
                         int P,  // central position.
                         const std::vector<int32> &ci_phones,
                         const std::vector<int32> &alignment,
                         const Matrix<BaseFloat> &features,
                         const std::vector<int32> *phone_map,
                         std::map<EventType, GaussClusterable*> *stats) {

  KALDI_ASSERT(IsSortedAndUniq(ci_phones));
  std::vector<std::vector<int32> > split_alignment;
  bool ans = SplitToPhones(trans_model, alignment, &split_alignment);
  if (!ans) {
    KALDI_WARN << "AccumulateTreeStats: alignment appears to be bad, not using it";
    return;
  }
  int cur_pos = 0;
  int dim = features.NumCols();
  KALDI_ASSERT(features.NumRows() == static_cast<int32>(alignment.size()));
  for (int i = -N; i < static_cast<int>(split_alignment.size()); i++) {
    // consider window starting at i, only if i+P is within
    // list of phones.
    if (i + P >= 0 && i + P < static_cast<int>(split_alignment.size())) {
      int32 central_phone =
          MapPhone(phone_map,
                   trans_model.TransitionIdToPhone(split_alignment[i+P][0]));
      bool is_ctx_dep = ! std::binary_search(ci_phones.begin(),
                                             ci_phones.end(),
                                             central_phone);
      //make cd phone
      EventType evec;
      for (int j = 0; j < N; j++) {
        int phone;
        if (i + j >= 0 && i + j < static_cast<int>(split_alignment.size()))
          phone =
              MapPhone(phone_map,
                       trans_model.TransitionIdToPhone(split_alignment[i+j][0]));
        else
          phone = 0;  // ContextDependency class uses 0 to mean "out of window";
        // we also set the phone arbitrarily to 0
        if (is_ctx_dep || j == P)
          evec.push_back(std::make_pair(static_cast<EventKeyType>(j), static_cast<EventValueType>(phone)));
      }
      //KALDI_LOG << central_phone << " " << evec.size(); 
      // CI Phone(SIL,SP or SPN may) has more than 3 states
      int num_state = trans_model.GetTopo().NumPdfClasses(central_phone);
      //KALDI_LOG << "num_state " << num_state;

      //binbin concatenating 3 states central phone as feature vector, 39 * 3
      std::vector<Vector<BaseFloat> > sum_stats(num_state, Vector<BaseFloat>(dim));

      std::vector <int> nums(num_state, 0);
      int state = 0;
      //std::cerr << "result assign ";
      for (int j = 0; j < static_cast<int>(split_alignment[i+P].size());j++) {
        if (0 == j) {
          sum_stats[0].AddVec(1.0, features.Row(cur_pos)); 
          nums[0] = 1;
          //std::cerr << 0 << " ";
        } else {
          int32 pdf_class_pre = trans_model.TransitionIdToPdfClass(split_alignment[i+P][j-1]);
          int32 pdf_class = trans_model.TransitionIdToPdfClass(split_alignment[i+P][j]);
          if (pdf_class != pdf_class_pre) state++;
          // In some case, state >= num_state, havn't find the reason
          if (state < num_state) {
            nums[state]++;
            //std::cerr << state << " ";
            sum_stats[state].AddVec(1.0, features.Row(cur_pos)); 
          }
        }
        cur_pos++;
      }
      //std::cerr << "\n";
      for (int k = 0; k < sum_stats.size(); k++) {
        sum_stats[k].Scale(1.0f / nums[k]); 
      }

      Vector<BaseFloat> concate_features(dim * 3);
      for (int j = 0; j < 3; j++) {
        concate_features.Range(j*dim, dim).CopyFromVec(sum_stats[j]);
      }
      //for (int k = 0; k < concate_features.Dim(); k++) {
      //  std::cerr << concate_features(k) << " ";
      //}
      //std::cerr << "\n";

      evec.push_back(std::make_pair(static_cast<EventKeyType>(kPdfClass), static_cast<EventValueType>(0)));
      //evec.push_back(std::make_pair(static_cast<EventKeyType>(kPdfClass), static_cast<EventValueType>(central_phone)));
      std::sort(evec.begin(), evec.end());
      if (stats->count(evec) == 0) {
          (*stats)[evec] = new GaussClusterable(dim * 3, var_floor);
      }
      BaseFloat weight = 1.0;
      (*stats)[evec]->AddStats(concate_features, weight);
    }
  }
  KALDI_ASSERT(cur_pos == static_cast<int>(alignment.size()));
}

}

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
        "Accumulate statistics using mean stats for phonetic-context tree building.\n"
        "Usage:  aslp-acc-tree-stats-phone-mean [options] model-in features-rspecifier alignments-rspecifier [tree-accs-out]\n"
        "e.g.: \n"
        " aslp-acc-tree-stats-phone-mean 1.mdl scp:train.scp ark:1.ali 1.tacc\n";
    ParseOptions po(usage);
    bool binary = true;
    float var_floor = 0.01;
    string ci_phones_str;
    std::string phone_map_rxfilename;
    int N = 3;
    int P = 1;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("var-floor", &var_floor, "Variance floor for tree clustering.");
    po.Register("ci-phones", &ci_phones_str, "Colon-separated list of integer "
                "indices of context-independent phones (after mapping, if "
                "--phone-map option is used).");
    po.Register("context-width", &N, "Context window size.");
    po.Register("central-position", &P, "Central context-window position "
                "(zero-based)");
    po.Register("phone-map", &phone_map_rxfilename,
                "File name containing old->new phone mapping (each line is: "
                "old-integer-id new-integer-id)");
    
    po.Read(argc, argv);

    if (po.NumArgs() < 3 || po.NumArgs() > 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        alignment_rspecifier = po.GetArg(3),
        accs_out_wxfilename = po.GetOptArg(4);

    std::vector<int32> phone_map;
    if (phone_map_rxfilename != "") {  // read phone map.
      ReadPhoneMap(phone_map_rxfilename,
                   &phone_map);
    }
    
    std::vector<int32> ci_phones;
    if (ci_phones_str != "") {
      SplitStringToIntegers(ci_phones_str, ":", false, &ci_phones);
      std::sort(ci_phones.begin(), ci_phones.end());
      if (!IsSortedAndUniq(ci_phones) || ci_phones[0] == 0) {
        KALDI_ERR << "Invalid set of ci_phones: " << ci_phones_str;
      }
    }

    

    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      // There is more in this file but we don't need it.
    }

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessInt32VectorReader alignment_reader(alignment_rspecifier);

    std::map<EventType, GaussClusterable*> tree_stats;

    int num_done = 0, num_no_alignment = 0, num_other_error = 0;

    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string key = feature_reader.Key();
      //KALDI_LOG << "Processing " << key;
      std::cerr << "Processing " << key << "\n";
      if (!alignment_reader.HasKey(key)) {
        num_no_alignment++;
      } else {
        const Matrix<BaseFloat> &mat = feature_reader.Value();
        const std::vector<int32> &alignment = alignment_reader.Value(key);

        if (alignment.size() != mat.NumRows()) {
          KALDI_WARN << "Alignments has wrong size "<< (alignment.size())<<" vs. "<< (mat.NumRows());
          num_other_error++;
          continue;
        }

        ////// This is the important part of this program.  ////////
        AccumulateTreeStatsPhone(trans_model,
                            var_floor,
                            N,
                            P,
                            ci_phones,
                            alignment,
                            mat,
                            (phone_map_rxfilename != "" ? &phone_map : NULL),
                            &tree_stats);
        //AccumulateTreeStats(trans_model,
        //                    var_floor,
        //                    N,
        //                    P,
        //                    ci_phones,
        //                    alignment,
        //                    mat,
        //                    (phone_map_rxfilename != "" ? &phone_map : NULL),
        //                    &tree_stats);

        num_done++;
        if (num_done % 1000 == 0)
          KALDI_LOG << "Processed " << num_done << " utterances.";
      }
    }

    BuildTreeStatsType stats;  // vectorized form.

    for (std::map<EventType, GaussClusterable*>::const_iterator iter = tree_stats.begin();  
        iter != tree_stats.end();
        iter++ ) {
      stats.push_back(std::make_pair(iter->first, iter->second));
    }
    tree_stats.clear();

    {
      Output ko(accs_out_wxfilename, binary);
      WriteBuildTreeStats(ko.Stream(), binary, stats);
    }
    KALDI_LOG << "Accumulated stats for " << num_done << " files, "
              << num_no_alignment << " failed due to no alignment, "
              << num_other_error << " failed for other reasons.";
    KALDI_LOG << "Number of separate stats (context-dependent states) is "
              << stats.size();
    DeleteBuildTreeStats(&stats);
    if (num_done != 0) return 0;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


