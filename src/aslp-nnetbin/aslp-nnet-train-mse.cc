// nnetbin/nnet-train-frmshuff.cc

// Copyright 2013  Brno University of Technology (Author: Karel Vesely)
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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "aslp-cudamatrix/cu-device.h"

#include "aslp-nnet/nnet-trnopts.h"
#include "aslp-nnet/nnet-nnet.h"
#include "aslp-nnet/nnet-loss.h"
#include "aslp-nnet/nnet-randomizer.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::aslp_nnet;
  typedef kaldi::int32 int32;  
  
  try {
    const char *usage =
        "Perform one iteration of Neural Network training by mini-batch Stochastic Gradient Descent.\n"
        "This version use pdf-posterior as targets, prepared typically by ali-to-post.\n"
        "Usage:  aslp-nnet-train-simple [options] <feature-rspecifier> <targets-rspecifier> <model-in> [<model-out>]\n"
        "e.g.: \n"
        " aslp-nnet-train-simple scp:feature.scp ark:posterior.ark nnet.init nnet.iter1\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts;
    trn_opts.Register(&po);
    NnetDataRandomizerOptions rnd_opts;
    rnd_opts.Register(&po);

    bool binary = true, 
         crossvalidate = false,
         randomize = true;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("cross-validate", &crossvalidate, "Perform cross-validation (don't backpropagate)");
    po.Register("randomize", &randomize, "Perform the frame-level shuffling within the Cache::");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in Nnet format");
    std::string objective_function = "mse";
    po.Register("objective-function", &objective_function, "Objective function : xent|mse");

    int32 length_tolerance = 5;
    po.Register("length-tolerance", &length_tolerance, "Allowed length difference of features/targets (frames)");
    
    std::string frame_weights;
    po.Register("frame-weights", &frame_weights, "Per-frame weights to scale gradients (frame selection/weighting).");

    std::string utt_weights;
    po.Register("utt-weights", &utt_weights, "Per-utterance weights (scalar applied to frame-weights).");

    std::string use_gpu="yes";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");
    
    double dropout_retention = 0.0;
    po.Register("dropout-retention", &dropout_retention, "number between 0..1, saying how many neurons to preserve (0.0 will keep original value");
    int report_period = 60000; // 60000 frames with one report 
    po.Register("report-period", &report_period, "Number of frames for one report log, default(60000)");
    
    po.Read(argc, argv);

    if (po.NumArgs() != 4-(crossvalidate?1:0)) {
      po.PrintUsage();
      exit(1);
    }

    std::string feature_rspecifier = po.GetArg(1),
      targets_rspecifier = po.GetArg(2),
      model_filename = po.GetArg(3);
        
    std::string target_model_filename;
    if (!crossvalidate) {
      target_model_filename = po.GetArg(4);
    }

    using namespace kaldi;
    using namespace kaldi::aslp_nnet;
    typedef kaldi::int32 int32;

    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    Nnet nnet_transf;
    if(feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);
    nnet.SetTrainOptions(trn_opts);

    if (dropout_retention > 0.0) {
      nnet_transf.SetDropoutRetention(dropout_retention);
      nnet.SetDropoutRetention(dropout_retention);
    }
    if (crossvalidate) {
      nnet_transf.SetDropoutRetention(1.0);
      nnet.SetDropoutRetention(1.0);
    }

    kaldi::int64 total_frames = 0;
    kaldi::int64 report_frames = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    //RandomAccessPosteriorReader targets_reader(targets_rspecifier);
    //SequentialPosteriorReader targets_reader(targets_rspecifier);
    SequentialBaseFloatMatrixReader targets_reader(targets_rspecifier);

    RandomAccessBaseFloatVectorReader weights_reader;
    if (frame_weights != "") {
      weights_reader.Open(frame_weights);
    }
    RandomAccessBaseFloatReader utt_weights_reader;
    if (utt_weights != "") {
      utt_weights_reader.Open(utt_weights);
    }

    RandomizerMask randomizer_mask(rnd_opts);
    MatrixRandomizer feature_randomizer(rnd_opts);
    //PosteriorRandomizer targets_randomizer(rnd_opts);
    MatrixRandomizer targets_randomizer(rnd_opts);
    VectorRandomizer weights_randomizer(rnd_opts);

    Mse mse;
    if (objective_function != "mse") {
        KALDI_ERR << "Only support mse training";
    }
    
    CuMatrix<BaseFloat> feats_transf, nnet_out, obj_diff;

    Timer time;
    KALDI_LOG << (crossvalidate?"CROSS-VALIDATION":"TRAINING") << " STARTED";

    int32 num_done = 0, num_no_tgt_mat = 0, num_other_error = 0;
    while (!feature_reader.Done()) {
#if HAVE_CUDA==1
      // check the GPU is not overheated
      CuDevice::Instantiate().CheckGpuHealth();
#endif
      // fill the randomizer
      for ( ; !feature_reader.Done(); feature_reader.Next(), targets_reader.Next()) {
        if (feature_randomizer.IsFull()) break; // suspend, keep utt for next loop
        std::string utt = feature_reader.Key(); 
        if (utt != targets_reader.Key()) {
            KALDI_ERR << "feat and target not in the same order or not exist in target"
                      << "feat key " << utt
                      << "target key " << targets_reader.Key();
        }
        KALDI_VLOG(3) << "Reading " << utt;

        // check we have per-frame weights
        if (frame_weights != "" && !weights_reader.HasKey(utt)) {
          KALDI_WARN << utt << ", missing per-frame weights";
          num_other_error++;
          continue;
        }
        // check we have per-utterance weights
        if (utt_weights != "" && !utt_weights_reader.HasKey(utt)) {
          KALDI_WARN << utt << ", missing per-utterance weight";
          num_other_error++;
          continue;
        }
        // get feature / target pair
        Matrix<BaseFloat> mat = feature_reader.Value();
        Matrix<BaseFloat> targets = targets_reader.Value();
        if (mat.NumRows() != targets.NumRows()) {
            KALDI_WARN << utt << " feat and target are not the same dim"
                       << " feat " << mat.NumRows() 
                       << " target " << targets.NumRows();
            continue;
        }
        // get per-frame weights
        Vector<BaseFloat> weights;
        if (frame_weights != "") {
          weights = weights_reader.Value(utt);
        } else { // all per-frame weights are 1.0
          weights.Resize(mat.NumRows());
          weights.Set(1.0);
        }

        // apply optional feature transform
        nnet_transf.Feedforward(CuMatrix<BaseFloat>(mat), &feats_transf);

        // pass data to randomizers
        KALDI_ASSERT(feats_transf.NumRows() == targets.NumRows());
        feature_randomizer.AddData(feats_transf);
        targets_randomizer.AddData(CuMatrix<BaseFloat>(targets));
        weights_randomizer.AddData(weights);
        num_done++;
      
        // report the speed
        if (num_done % 5000 == 0) {
          double time_now = time.Elapsed();
          KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                        << time_now/60 << " min; processed " << total_frames/time_now
                        << " frames per second.";
        }
      }

      // randomize
      if (!crossvalidate && randomize) {
        const std::vector<int32>& mask = randomizer_mask.Generate(feature_randomizer.NumFrames());
        feature_randomizer.Randomize(mask);
        targets_randomizer.Randomize(mask);
        weights_randomizer.Randomize(mask);
      }

      // train with data from randomizers (using mini-batches)
      for ( ; !feature_randomizer.Done(); feature_randomizer.Next(),
                                          targets_randomizer.Next(),
                                          weights_randomizer.Next()) {
        // get block of feature/target pairs
        const CuMatrixBase<BaseFloat> &nnet_in = feature_randomizer.Value();
        const CuMatrixBase<BaseFloat> &nnet_tgt = targets_randomizer.Value();
        const Vector<BaseFloat>& frm_weights = weights_randomizer.Value();

        // forward pass
        if (!crossvalidate) {
            nnet.Propagate(nnet_in, &nnet_out);
        } else {
            nnet.Feedforward(nnet_in, &nnet_out);
        }
        mse.Eval(frm_weights, nnet_out, nnet_tgt, &obj_diff);
        // backward pass
        if (!crossvalidate) {
          // backpropagate
          nnet.Backpropagate(obj_diff, NULL);
        }

        // 1st minibatch : show what happens in network 
        if (kaldi::g_kaldi_verbose_level >= 1 && total_frames == 0) { // vlog-1
          KALDI_VLOG(1) << "### After " << total_frames << " frames,";
          KALDI_VLOG(1) << nnet.InfoPropagate();
          if (!crossvalidate) {
            KALDI_VLOG(1) << nnet.InfoBackPropagate();
            KALDI_VLOG(1) << nnet.InfoGradient();
          }
        }
        
        // monitor the NN training
        if (kaldi::g_kaldi_verbose_level >= 2) { // vlog-2
          if ((total_frames/25000) != ((total_frames+nnet_in.NumRows())/25000)) { // print every 25k frames
            KALDI_VLOG(2) << "### After " << total_frames << " frames,";
            KALDI_VLOG(2) << nnet.InfoPropagate();
            if (!crossvalidate) {
              KALDI_VLOG(2) << nnet.InfoGradient();
            }
          }
        }
        
        total_frames += nnet_in.NumRows();
        // Add by zhangbinbin, date 2016-01-17
        report_frames += nnet_in.NumRows();
        if (report_frames >= report_period) {
            KALDI_LOG << mse.Report();
            report_frames -= report_period;
        }
      }
    }
    
    // after last minibatch : show what happens in network 
    if (kaldi::g_kaldi_verbose_level >= 1) { // vlog-1
      KALDI_VLOG(1) << "### After " << total_frames << " frames,";
      KALDI_VLOG(1) << nnet.InfoPropagate();
      if (!crossvalidate) {
        KALDI_VLOG(1) << nnet.InfoBackPropagate();
        KALDI_VLOG(1) << nnet.InfoGradient();
      }
    }

    if (!crossvalidate) {
      nnet.Write(target_model_filename, binary);
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_no_tgt_mat
              << " with no tgt_mats, " << num_other_error
              << " with other errors. "
              << "[" << (crossvalidate?"CROSS-VALIDATION":"TRAINING")
              << ", " << (randomize?"RANDOMIZED":"NOT-RANDOMIZED") 
              << ", " << time.Elapsed()/60 << " min, fps" << total_frames/time.Elapsed()
              << "]";  

    KALDI_LOG << mse.Report();

#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
