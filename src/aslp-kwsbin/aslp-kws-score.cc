// nnetbin/nnet-forward.cc

// Copyright 2011-2013  Brno University of Technology (Author: Karel Vesely)
// Copyright 2016  ASLP (Author: zhangbinbin liwenpeng duwei)

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

#include <limits>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"

#include "aslp-nnet/nnet-nnet.h"
#include "aslp-nnet/nnet-loss.h"
#include "aslp-nnet/nnet-pdf-prior.h"

#include "aslp-kws/keyword-spot.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::aslp_nnet;
  using namespace kaldi::kws;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Perform keyword spot through Neural Network.\n"
        "\n"
        "Usage:  aslp-kws-score [options] <model-in> <fsm-in> <feature-rspecifier> <confidence-wspecifier>\n"
        "e.g.: \n"
        " aslp-kws-score nnet ark:features.ark ark,t:confidence.ark\n";

    ParseOptions po(usage);

    std::string use_gpu="no";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        fsm_filename = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        confidence_wspecifier = po.GetArg(4);
        
    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    Nnet nnet;
    nnet.Read(model_filename);

    Fsm fsm;
    fsm.Read(fsm_filename.c_str());
    KeywordSpot keyword_spotter(fsm);

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    BaseFloatVectorWriter confidence_writer(confidence_wspecifier);

    CuMatrix<BaseFloat> feats, nnet_out;
    Matrix<BaseFloat> nnet_out_host;
    Vector<BaseFloat> confidence;


    Timer time;
    double time_now = 0;
    int32 num_done = 0;
    // iterate over all feature files
    for (; !feature_reader.Done(); feature_reader.Next()) {
      // read
      Matrix<BaseFloat> mat = feature_reader.Value();
      std::string utt = feature_reader.Key();
      KALDI_VLOG(2) << "Processing utterance " << num_done+1 
                    << ", " << utt
                    << ", " << mat.NumRows() << "frm";
      if (!KALDI_ISFINITE(mat.Sum())) { // check there's no nan/inf,
        KALDI_ERR << "NaN or inf found in features for " << utt;
      }
      // push it to gpu,
      feats = mat;

      std::vector<int> frame_num_utt;
      frame_num_utt.push_back(feats.NumRows());
      nnet.SetSeqLengths(frame_num_utt);
      // fwd-pass, nnet,
      nnet.Feedforward(feats, &nnet_out);

      if (!KALDI_ISFINITE(nnet_out.Sum())) { // check there's no nan/inf,
        KALDI_ERR << "NaN or inf found in nn-output for " << utt;
      }

      // download from GPU,
      nnet_out_host.Resize(nnet_out.NumRows(), nnet_out.NumCols());
      nnet_out.CopyToMat(&nnet_out_host);
      confidence.Resize(nnet_out.NumRows());

      float keyword_confidence = 0.0;
      keyword_spotter.Reset();
      for (int i = 0; i < nnet_out_host.NumRows(); i++) {
        keyword_spotter.Spot(nnet_out_host.Row(i).Data(), 
            nnet_out_host.NumCols(), &keyword_confidence);
        confidence(i) = keyword_confidence;
      }

      // write,
      if (!KALDI_ISFINITE(nnet_out_host.Sum())) { // check there's no nan/inf,
        KALDI_ERR << "NaN or inf found in final output nn-output for " << utt;
      }
      confidence_writer.Write(feature_reader.Key(), confidence);
      
      if (num_done % 100 == 0) {
        time_now = time.Elapsed();
        KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                      << time_now/60 << " min; processed " << tot_t/time_now
                      << " frames per second.";
      }
      num_done++;
      tot_t += mat.NumRows();
    }
    
    // final message
    KALDI_LOG << "Done " << num_done << " files" 
              << " in " << time.Elapsed()/60 << "min," 
              << " (fps " << tot_t/time.Elapsed() << ")"; 

#if HAVE_CUDA==1
    if (kaldi::g_kaldi_verbose_level >= 1) {
      CuDevice::Instantiate().PrintProfile();
    }
#endif

    if (num_done == 0) return -1;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
