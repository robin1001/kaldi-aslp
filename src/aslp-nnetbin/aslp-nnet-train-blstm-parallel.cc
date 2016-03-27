// aslp-nnetbin/aslp-nnet-train-blstm-parallel.cc

// Copyright 2015 Chongjia Ni
// Copyright 2016 ASLP (Binbin Zhang)
// Created 2016-03-25

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include "aslp-nnet/nnet-trnopts.h"
#include "aslp-nnet/nnet-nnet.h"
#include "aslp-nnet/nnet-loss.h"
#include "aslp-nnet/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "aslp-cudamatrix/cu-device.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::aslp_nnet;
  typedef kaldi::int32 int32;

  try {
    const char *usage =
        "Perform one iteration of senones training by SGD.\n"
        "The updates are done per-utternace and by processing multiple utterances in parallel.\n"
        "\n"
        "Usage: aslp-nnet-train-blstm-parallel [options] <feature-rspecifier> <labels-rspecifier> <model-in> [<model-out>]\n"
        "e.g.: \n"
        " aslp-nnet-train-blstm-parallel scp:feature.scp ark:labels.ark nnet.init nnet.iter1\n";

    ParseOptions po(usage);
    // training options
    NnetTrainOptions trn_opts;
    trn_opts.Register(&po);

    bool binary = true,
         crossvalidate = false;
    po.Register("binary", &binary, "Write model  in binary mode");
    po.Register("cross-validate", &crossvalidate, "Perform cross-validation (no backpropagation)");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in Nnet format");

    int32 length_tolerance = 5;
    po.Register("length-tolerance", &length_tolerance, "Allowed length difference of features/targets (frames)");

    std::string frame_weights;
    po.Register("frame-weights", &frame_weights, "Per-frame weights to scale gradients (frame selection/weighting).");

    std::string objective_function = "xent";
    po.Register("objective-function", &objective_function, "Objective function : xent|mse");

    int32 num_stream = 4;
    po.Register("num-stream", &num_stream, "Number of sequences processed in parallel");

    double frame_limit = 100000;
    po.Register("frame-limit", &frame_limit, "Max number of frames to be processed");

    std::string use_gpu = "yes";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");

    // Add dummy randomizer options, to make the tool compatible with standard scripts
    NnetDataRandomizerOptions rnd_opts;
    rnd_opts.Register(&po);
    bool randomize = false;
    po.Register("randomize", &randomize, "Dummy option, for compatibility...");
    //
    int report_period = 200; // 200 sentence with one report 
    po.Register("report-period", &report_period, "Number of sentence for one report log, default(200)");
    int drop_len = 0;
    po.Register("drop-len", &drop_len, "if Sentence frame length greater than drop_len,"
            "then drop it, default(0, no drop)");

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

    // Select the GPU
#if HAVE_CUDA == 1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    Nnet nnet;
    nnet.Read(model_filename);
    nnet.SetTrainOptions(trn_opts);

    kaldi::int64 total_frames = 0;

    // Initialize feature ans labels readers
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessPosteriorReader targets_reader(targets_rspecifier);
    
    LossItf *loss = NULL;
    if (objective_function == "xent") {
        loss = new Xent;
    } else if (objective_function == "mse") {
        loss = new Mse;
    } else {
        KALDI_ERR << "Unsupported objective function: " << objective_function;
    }

    CuMatrix<BaseFloat> feats, nnet_out, obj_diff;

    Timer time;
    KALDI_LOG << (crossvalidate?"CROSS-VALIDATION":"TRAINING") << " STARTED";
    // Feature matrix of every utterance
    std::vector< Matrix<BaseFloat> > feats_utt(num_stream);
    // Label vector of every utterance
    std::vector< Posterior > labels_utt(num_stream);

    int32 feat_dim = nnet.InputDim();

    int32 num_done = 0, num_no_tgt_mat = 0, num_other_error = 0;
    int32 num_sentence = 0;

    while (1) {

      std::vector<int32> frame_num_utt;
      int32 sequence_index = 0, max_frame_num = 0;

      for ( ; !feature_reader.Done(); feature_reader.Next()) {
        std::string utt = feature_reader.Key();
        // Check that we have targets
        if (!targets_reader.HasKey(utt)) {
          KALDI_WARN << utt << ", missing targets";
          num_no_tgt_mat++;
          continue;
        }
        // Get feature / target pair
        Matrix<BaseFloat> mat = feature_reader.Value();
        if (drop_len > 0 && mat.NumRows() > drop_len) {
            KALDI_WARN << utt << ", too long, droped";
            feature_reader.Next();
            continue;
        }
        Posterior targets  = targets_reader.Value(utt);
        if (mat.NumRows() != targets.size()) {
            KALDI_WARN << utt << "feat and the target are not the same length, droped";
            continue;
        }

        if (max_frame_num < mat.NumRows()) max_frame_num = mat.NumRows();
        feats_utt[sequence_index] = mat;
        labels_utt[sequence_index] = targets;
        frame_num_utt.push_back(mat.NumRows());
        sequence_index++;
        // If the total number of frames reaches frame_limit, then stop adding more sequences, regardless of whether
        // the number of utterances reaches num_sequence or not.
        if (frame_num_utt.size() == num_stream || frame_num_utt.size() * max_frame_num > frame_limit) {
            feature_reader.Next(); break;
        }
      }
      int32 cur_sequence_num = frame_num_utt.size();

      // Create the final feature matrix. Every utterance is padded to the max length within this group of utterances
      Matrix<BaseFloat> feat_mat_host(cur_sequence_num * max_frame_num, feat_dim, kSetZero);
      Posterior target_host;
      target_host.resize(cur_sequence_num * max_frame_num);

      for (int s = 0; s < cur_sequence_num; s++) {
        Matrix<BaseFloat> mat_tmp = feats_utt[s];
        for (int r = 0; r < frame_num_utt[s]; r++) {
          feat_mat_host.Row(r*cur_sequence_num + s).CopyFromVec(mat_tmp.Row(r));
        }
      }

      for (int s = 0; s < cur_sequence_num; s++) {
        Posterior target_tmp = labels_utt[s];
        for (int r = 0; r < frame_num_utt[s]; r++) {
          target_host[r*cur_sequence_num+s] = target_tmp[r];
        }
      }

      // Set the original lengths of utterances before padding
      nnet.SetSeqLengths(frame_num_utt);

      // Propagation and xent training
      if (!crossvalidate) {
          nnet.Propagate(CuMatrix<BaseFloat>(feat_mat_host), &nnet_out);
      } else {
          nnet.Feedforward(CuMatrix<BaseFloat>(feat_mat_host), &nnet_out);
      }

      loss->Eval(nnet_out, target_host, &obj_diff);

      // Backward pass
      if (!crossvalidate) {
        nnet.Backpropagate(obj_diff, NULL);
      }

      num_done += cur_sequence_num;
      total_frames += feat_mat_host.NumRows();
      num_sentence += cur_sequence_num;
      // Report likelyhood
      if (num_sentence >= report_period) {
          KALDI_LOG << loss->Report();
          num_sentence -= report_period;
      }

      if (feature_reader.Done()) break;  // end loop of while(1)
    }

    if (!crossvalidate) {
      nnet.Write(target_model_filename, binary);
    }

    KALDI_LOG << "Done " << num_done << " files, " << num_no_tgt_mat
              << " with no tgt_mats, " << num_other_error
              << " with other errors. "
              << "[" << (crossvalidate?"CROSS-VALIDATION":"TRAINING")
              << ", " << time.Elapsed()/60 << " min, fps" << total_frames/time.Elapsed()
              << "]";
    
    KALDI_LOG << loss->Report();
    if (loss != NULL) delete loss;

#if HAVE_CUDA == 1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
