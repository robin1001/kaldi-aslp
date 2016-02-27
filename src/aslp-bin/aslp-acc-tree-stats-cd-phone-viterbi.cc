// aslp-bin/aslp-acc-tree-stats-cd-phone-viterbi.cc

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
#include "tree/cluster-utils.h"
#include "tree/clusterable-classes.h"
#include "hmm/transition-model.h"
#include "hmm/tree-accu.h"
#include "hmm/hmm-utils.h"

namespace kaldi {

BaseFloat ClusterDistanceViterbi(const MatrixBase<BaseFloat> &features, // N * dim
                            int32 num_cluster,
                            std::vector<int32> *assignments_out,
                            int32 num_iter) {
  int32 num_frame = features.NumRows();
  int32 dim = features.NumCols();
  KALDI_ASSERT(num_cluster > 0);
  KALDI_ASSERT(num_frame > num_cluster);
  // Init the mean, equal align
  int32 stride = num_frame / num_cluster;
  std::vector<Vector<BaseFloat> > mean_stats(num_cluster, Vector<BaseFloat>(dim));
  for (int i = 0; i < num_cluster - 1; i++) {
    mean_stats[i].AddRowSumMat(1.0 / stride, features.RowRange(i * stride, stride), 0.0);
  }
  int32 last_start = (num_cluster - 1) * stride;
  int32 last_size = num_frame - last_start;
  mean_stats[num_cluster - 1].AddRowSumMat(1.0 / last_size, features.RowRange(last_start, last_size), 0.0);

  // Do viterbi distance align for num_iter epoch
  BaseFloat loss, pre_loss = 0;
  Matrix<BaseFloat> distance(num_frame, num_cluster);
  Matrix<BaseFloat> align(num_frame, num_cluster);
  std::vector<int> alignment(num_frame, 0);
  for (int n = 0; n < num_iter; n++) {
    distance.Set(1e10); // Big enough
    align.SetZero(); 
    // Set Init Distance
    {
      Vector<BaseFloat> feat(features.Row(0));
      feat.AddVec(-1.0, mean_stats[0]);
      BaseFloat d = VecVec(feat, feat);
      distance(0, 0) = d;
    }
    // Forward calculate
    for (int i = 1; i < num_frame; i++) {
      //for (int j = 0; j < num_cluster - 1; j++) {
      for (int j = 0; j < num_cluster; j++) {
        // Viterbi distance 
        if (j - 1 < 0) {
          Vector<BaseFloat> feat(features.Row(i));
          feat.AddVec(-1.0, mean_stats[j]);
          BaseFloat d = VecVec(feat, feat);
          distance(i, j) = distance(i-1, j) + d;
          align(i, j) = j;
        }
        else {
          Vector<BaseFloat> feat0(features.Row(i));
          feat0.AddVec(-1.0, mean_stats[j - 1]);
          BaseFloat d0 = distance(i-1, j-1) + VecVec(feat0, feat0);
          Vector<BaseFloat> feat1(features.Row(i));
          feat1.AddVec(-1.0, mean_stats[j]);
          BaseFloat d1 = distance(i-1, j) + VecVec(feat1, feat1);
          if (d0 < d1) {
            distance(i, j) = d0;
            align(i, j) = j - 1;
          }
          else {
            distance(i, j) = d1;
            align(i, j) = j;
          }
        }
      } // for j 
    } // for i
    // The last frame must be aligned with the last cluster
    //KALDI_LOG << align(num_frame-1, num_cluster-1) << " " << num_cluster;
    //KALDI_ASSERT(static_cast<int>(align(num_frame-1, num_cluster-1)) == num_cluster - 1);
    {
      Vector<BaseFloat> feat(features.Row(num_frame - 1));
      feat.AddVec(-1.0, mean_stats[num_cluster - 1]);
      BaseFloat d = VecVec(feat, feat);
      distance(num_frame-1, num_cluster-1) = distance(num_frame-2, num_cluster-1) + d;
      align(num_frame-1, num_cluster-1) = num_cluster - 1;
    }
    loss = distance(num_frame-1, num_cluster-1);
    // Backtrace best path 
    alignment[num_frame - 1] = static_cast<int>(align(num_frame-1, num_cluster-1)); 
    for (int i = num_frame - 2; i >= 0; i--) {
      alignment[i] = static_cast<int>(align(i, alignment[i+1]));
    }
    //KALDI_LOG << "pre_loss " << pre_loss << " loss " << loss ;
    //for (int i = 0; i < num_frame; i++) 
    //  cerr << alignment[i] << " ";
    //cerr << "\n";

    // Update mean
    if (0 == loss - pre_loss)
      break;
    else {
      std::vector<int> start_pos;
      start_pos.push_back(0);
      for (int i = 1; i < alignment.size(); i++) {
        if (alignment[i] != alignment[i-1]) {
          start_pos.push_back(i);
        }
      }
      //KALDI_LOG << start_pos[0] << " " << start_pos[1] << " " << start_pos[2];
      KALDI_ASSERT(start_pos.size() == num_cluster);
      for (int i = 0; i < num_cluster - 1; i++) {
         int num_points = start_pos[i+1] - start_pos[i];
         mean_stats[i].AddRowSumMat(1.0 / num_points, features.RowRange(start_pos[i], num_points), 0.0);
      }
      last_size = num_frame - start_pos[num_cluster-1];
      mean_stats[num_cluster - 1].AddRowSumMat(1.0 / last_size, features.RowRange(start_pos[num_cluster - 1], last_size), 0.0);
    }
    pre_loss = loss;
  } // for n
  (*assignments_out) = alignment;
  return loss;
}

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

      int num_cluster = 3;
      size_t frame_len = split_alignment[i+P].size();
      Vector<BaseFloat> concate_features(dim * num_cluster);
      
      if (frame_len <= num_cluster) { 
        for (int k = 0; k < frame_len; k++)
          concate_features.Range(k*dim, dim).CopyFromVec(features.Row(cur_pos + k));
        // append the last frame's feature
        for (int k = frame_len; k < num_cluster; k++) 
          concate_features.Range(k*dim, dim).CopyFromVec(features.Row(cur_pos + frame_len - 1));
      }
      else {
        //for (int k = 0; k < frame_len; k++) {
        //  // just for type conversion between Vector and SubVector
        //  Vector<BaseFloat> point(features.Row(cur_pos + k)); 
        //  //for (int n = 0; n < point.Dim(); n++) {
        //  //  std::cerr << point(n) << " ";
        //  //}
        //  //std::cerr << "\n";
        //}
        std::vector<int32> assignments;
        //kcfg.refine_cfg.num_iters = 10; //default 100
        BaseFloat ans = ClusterDistanceViterbi(features.RowRange(cur_pos, frame_len), num_cluster, &assignments, 10);
        (void)ans;

        // show the cluster align result
        //std::cerr << "result assign ";
        //for (int k = 0; k < assignments.size(); k++) {
        //  std::cerr << assignments[k] << " ";
        //}
        //std::cerr << "\n";

        std::vector <int> nums(num_cluster, 0);
        std::vector<Vector<BaseFloat> > sum_stats(num_cluster, Vector<BaseFloat>(dim));
        // sum every cluster and then mean
        for (int k = 0; k < frame_len; k++) {
          int index = assignments[k]; //belong's to i'th cluster
          KALDI_ASSERT(index < num_cluster);
          nums[index] += 1; 
          sum_stats[index].AddVec(1.0, features.Row(cur_pos + k));
        }
        for (int k = 0; k < sum_stats.size(); k++) {
          sum_stats[k].Scale(1.0f / nums[k]); 
        }
        
        //for (int k = 0; k < sum_stats.size(); k++) {
        //  for (int n = 0; n < sum_stats[k].Dim(); n++) {
        //    std::cerr << sum_stats[k](n) << " ";
        //  }
        //  std::cerr << "\n";
        //}

        // concatenating all the cluster's mean as new feature
        for (int k = 0; k < num_cluster; k++) {
          concate_features.Range(k*dim, dim).CopyFromVec(sum_stats[k]);
        }
        
        //for (int k = 0; k < concate_features.Dim(); k++) {
        //  std::cerr << concate_features(k) << " ";
        //}
        //std::cerr << "\n";
      }

      evec.push_back(std::make_pair(static_cast<EventKeyType>(kPdfClass), static_cast<EventValueType>(0)));
      std::sort(evec.begin(), evec.end());

      if (stats->count(evec) == 0) {
          (*stats)[evec] = new GaussClusterable(dim * 3, var_floor);
      }
      BaseFloat weight = 1.0;
      (*stats)[evec]->AddStats(concate_features, weight);
      cur_pos += split_alignment[i+P].size();
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
        "Accumulate cd-phone statistics using viterbi realignment for phonetic-context tree building.\n"
        "Usage:  aslp-acc-tree-stats-cd-phone-viterbi [options] model-in features-rspecifier alignments-rspecifier [tree-accs-out]\n"
        "e.g.: \n"
        " aslp-acc-tree-stats-cd-phone-viterbi 1.mdl scp:train.scp ark:1.ali 1.tacc\n";
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


