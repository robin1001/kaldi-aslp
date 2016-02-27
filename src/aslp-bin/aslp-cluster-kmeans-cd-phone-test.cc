// aslp-bin/aslp-cluster-kmeans-cd-phone-test

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
      KALDI_LOG << "start " << start[i] << " end " << end[i];
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
      KALDI_LOG << "ClusterKMeans: on iteration "<<(iter)<<", objf before = "<<(objf_before)<<", impr = "<<(impr)<<", objf after = "<<(objf_after)<<", normalized by "<<(normalizer)<<" = "<<(objf_after/normalizer);
    if (impr == 0) break;
  }
  return ans;
}

}

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  int num = 7;
  BaseFloat arr[] = {3,0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0};
  std::vector<Clusterable*> points;
  for (int i = 0; i < num; i++) {
    points.push_back(new ScalarClusterable(arr[i]));
  }
  int32 n_clust = 3;
  std::vector<Clusterable*> clusters;
  std::vector<int32> assignments;
  ClusterKMeansOptions kcfg;
  //KALDI_LOG << "num_iters " << kcfg.num_iters << " num_tries " << kcfg.num_tries;
  kcfg.num_iters = 10;
  //kcfg.num_tries = 10;

  BaseFloat ans = ClusterKMeansForCDPhone(points, n_clust, &clusters, &assignments, kcfg);
  //BaseFloat ans = ClusterKMeans(points, n_clust, &clusters, &assignments, kcfg);
  KALDI_LOG << "loss " << ans; 
  for (int i = 0; i < assignments.size(); i++)
    KALDI_LOG << arr[i] << " " << assignments[i];

  DeletePointers(&clusters);
  DeletePointers(&points);
  return 0;
}


