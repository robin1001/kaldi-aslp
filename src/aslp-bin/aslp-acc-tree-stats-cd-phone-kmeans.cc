// aslp-bin/aslp-acc-tree-stats-cd-phone-equal.cc

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

BaseFloat ClusterKMeansForCDPhone(const std::vector<Clusterable*> &points,
                            int32 num_clust,
                            std::vector<Clusterable*> *clusters_out,
                            std::vector<int32> *assignments_out,
                            ClusterKMeansOptions &cfg) {
  std::vector<int32> my_assignments;
  int32 num_points = points.size();
  KALDI_ASSERT(clusters_out != NULL);
  KALDI_ASSERT(num_points != 0);
  KALDI_ASSERT(num_clust <= num_points);

  KALDI_ASSERT(clusters_out->empty());  // or we wouldn't know what to do with pointers in there.
  clusters_out->resize(num_clust, (Clusterable*)NULL);
  assignments_out->resize(num_points);
  
  // Assign, Now Sequential assign vs Origin random assign
  {
    int32 stride = num_points / num_clust;
    std::vector<int> start, end;
    for (int i = 0; i < num_clust; i++) {
      start.push_back(i * stride);
      if (i == num_clust - 1) { // the last cluster
          end.push_back(num_points - 1); 
      } else {
          end.push_back((i+1) * stride - 1);
      }
    }
    KALDI_ASSERT(start.size() == end.size());
    for (int i = 0; i < num_clust; i++) {
      KALDI_VLOG(2) << "start " << start[i] << " end " << end[i];
      for (int j = start[i]; j <= end[i]; j++) {
        // assign point i to cluster j.
        if ((*clusters_out)[i] == NULL) (*clusters_out)[i] = points[j]->Copy();
        else (*clusters_out)[i]->Add(*(points[j]));
        (*assignments_out)[j] = i;
      }
    }
  }

  BaseFloat normalizer = SumClusterableNormalizer(*clusters_out);
  BaseFloat ans;
  {  // work out initial value of "ans" (objective function improvement).
    Clusterable *all_stats = SumClusterable(*clusters_out);
    ans = SumClusterableObjf(*clusters_out) - all_stats->Objf();  // improvement just from the random
    // initialization.
    if (ans < -0.01 && ans < -0.01 * fabs(all_stats->Objf())) {  // something bad happend.
      KALDI_WARN << "ClusterKMeans: objective function after random assignment to clusters is worse than in single cluster: "<< (all_stats->Objf()) << " changed by " << ans << ".  Perhaps your stats class has the wrong properties?";
    }
    delete all_stats;
  }

  for (int32 iter = 0;iter < cfg.num_iters;iter++) {
    // Keep refining clusters by reassigning points.
    BaseFloat objf_before;
    if (cfg.verbose) objf_before =SumClusterableObjf(*clusters_out);
    BaseFloat impr = RefineClusters(points, clusters_out, assignments_out, cfg.refine_cfg);
    BaseFloat objf_after;
    if (cfg.verbose) objf_after = SumClusterableObjf(*clusters_out);
    ans += impr;
    if (cfg.verbose)
      KALDI_VLOG(2) << "ClusterKMeans: on iteration "<<(iter)<<", objf before = "<<(objf_before)<<", impr = "<<(impr)<<", objf after = "<<(objf_after)<<", normalized by "<<(normalizer)<<" = "<<(objf_after/normalizer);
    if (impr == 0) break;
  }
  return ans;
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
        std::vector<Clusterable*> points;
        for (int k = 0; k < frame_len; k++) {
          // just for type conversion between Vector and SubVector
          Vector<BaseFloat> point(features.Row(cur_pos + k)); 
          //for (int n = 0; n < point.Dim(); n++) {
          //  std::cerr << point(n) << " ";
          //}
          //std::cerr << "\n";
          points.push_back(new VectorClusterable(point, 1.0));
        }
        std::vector<Clusterable*> clusters;
        std::vector<int32> assignments;
        ClusterKMeansOptions kcfg;
        kcfg.num_iters = 5;
        //kcfg.refine_cfg.num_iters = 10; //default 100
        //BaseFloat ans = ClusterKMeans(points, num_cluster, &clusters, &assignments, kcfg);
        BaseFloat ans = ClusterKMeansForCDPhone(points, num_cluster, &clusters, &assignments, kcfg);
        KALDI_ASSERT(clusters.size() == num_cluster);
        (void)ans;

        // show the cluster align result
        std::cerr << "result assign ";
        for (int k = 0; k < assignments.size(); k++) {
          std::cerr << assignments[k] << " ";
        }
        std::cerr << "\n";

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
        "Accumulate cd-phone statistics using kmeans cluster for phonetic-context tree building.\n"
        "Usage:  aslp-acc-tree-stats-cd-phone-kmeans [options] model-in features-rspecifier alignments-rspecifier [tree-accs-out]\n"
        "e.g.: \n"
        " aslp-acc-tree-stats-cd-phone-kmeans 1.mdl scp:train.scp ark:1.ali 1.tacc\n";
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


