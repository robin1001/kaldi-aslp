// aslp-nnetbin/aslp-nnet-train-lstm-streams-skip.cc

// Copyright 2015  Brno University of Technology (Author: Karel Vesely)
//           2014  Jiayu DU (Jerry), Wei Li
// Copyright 2016  ASLP(Author: zhangbinbin)
// Modified on 2016-03-29

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
            "Perform one iteration of LSTM training by Stochastic Gradient Descent.\n"
            "This version use pdf-posterior as targets, prepared typically by ali-to-post.\n"
            "The updates are done per-utterance, shuffling options are dummy for compatibility reason.\n"
            "Attention: one sentence will be devided into N part for skip training\n"
            "\n"
            "Usage: aslp-nnet-train-lstm-streams-skip [options] <feature-rspecifier> <targets-rspecifier> <model-in> [<model-out>]\n"
            "e.g.: \n"
            " aslp-nnet-train-lstm-streams-skip scp:feature.scp ark:posterior.ark nnet.init nnet.iter1\n";

        ParseOptions po(usage);

        NnetTrainOptions trn_opts;
        trn_opts.Register(&po);

        bool binary = true, 
             crossvalidate = false;
        po.Register("binary", &binary, "Write output in binary mode");
        po.Register("cross-validate", &crossvalidate, "Perform cross-validation (don't backpropagate)");

        std::string feature_transform;
        po.Register("feature-transform", &feature_transform, "Feature transform in Nnet format");
        std::string objective_function = "xent";
        po.Register("objective-function", &objective_function, "Objective function : xent|mse");

        /*
           int32 length_tolerance = 5;
           po.Register("length-tolerance", &length_tolerance, "Allowed length difference of features/targets (frames)");

           std::string frame_weights;
           po.Register("frame-weights", &frame_weights, "Per-frame weights to scale gradients (frame selection/weighting).");
         */

        std::string use_gpu="yes";
        po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 

        //<jiayu>
        int32 targets_delay=5;
        po.Register("targets-delay", &targets_delay, "---LSTM--- BPTT targets delay"); 

        int32 batch_size=20;
        po.Register("batch-size", &batch_size, "---LSTM--- BPTT batch size"); 

        int32 num_stream=4;
        po.Register("num-stream", &num_stream, "---LSTM--- BPTT multi-stream training"); 

        int32 dump_interval=0;
        po.Register("dump-interval", &dump_interval, "---LSTM--- num utts between model dumping [ 0 == disabled ]"); 
        //</jiayu>

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
        int skip_width = 1;
        po.Register("skip-width", &skip_width, "num of frame for one skip(default 0, no skip)");

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

        kaldi::int64 total_frames = 0;

        Xent xent;
        // evaluate objective function we've chosen
        if (objective_function != "xent") {
            KALDI_ERR << "Only Support xent objective function, but got" << objective_function;
        }

        Timer time;
        KALDI_LOG << (crossvalidate?"CROSS-VALIDATION":"TRAINING") << " STARTED";

        int32 num_done = 0, num_no_tgt_mat = 0, num_other_error = 0;
        int32 num_sentence = 0;

        //  book-keeping for multi-streams
        std::vector<std::string> keys(num_stream);
        std::vector<Matrix<BaseFloat> > feats(num_stream);
        std::vector<Posterior> targets(num_stream);
        std::vector<int> curt(num_stream, 0);
        std::vector<int> lent(num_stream, 0);
        std::vector<int> new_utt_flags(num_stream, 0);

        // bptt batch buffer
        int32 feat_dim = nnet.InputDim();
        Vector<BaseFloat> frame_mask(batch_size * num_stream, kSetZero);
        Matrix<BaseFloat> feat(batch_size * num_stream, feat_dim, kSetZero);
        Posterior target(batch_size * num_stream);
        CuMatrix<BaseFloat> feat_transf, nnet_out, obj_diff;

        // Use one sentence for skip_width times,from different
        // The effect is the same as take one sentence into N part
        for (int skip_offset = 0; skip_offset < skip_width; skip_offset++) {
            SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
            RandomAccessPosteriorReader target_reader(targets_rspecifier);

            while (1) {
                // loop over all streams, check if any stream reaches the end of its utterance,
                // if any, feed the exhausted stream with a new utterance, update book-keeping infos
                for (int s = 0; s < num_stream; s++) {
                    // this stream still has valid frames
                    if (curt[s] < lent[s]) {
                        new_utt_flags[s] = 0;
                        continue;
                    }
                    // else, this stream exhausted, need new utterance
                    while (!feature_reader.Done()) {
                        const std::string& key = feature_reader.Key();
                        // get the feature matrix,
                        const Matrix<BaseFloat> &mat = feature_reader.Value();
                        if (drop_len > 0 && mat.NumRows() > drop_len) {
                            KALDI_WARN << key << ", too long, droped";
                            feature_reader.Next();
                            continue;
                        }
                        // forward the features through a feature-transform,
                        nnet_transf.Feedforward(CuMatrix<BaseFloat>(mat), &feat_transf);

                        // get the labels,
                        if (!target_reader.HasKey(key)) {
                            KALDI_WARN << key << ", missing targets";
                            num_no_tgt_mat++;
                            feature_reader.Next();
                            continue;
                        }
                        const Posterior& target = target_reader.Value(key);

                        // check that the length matches,
                        if (feat_transf.NumRows() != target.size()) {
                            KALDI_WARN << key << ", length miss-match between feats and targets, skip";
                            num_other_error++;
                            feature_reader.Next();
                            continue;
                        }

                        // Use skip
                        if (skip_width > 1) {
                            int skip_len = (feat_transf.NumRows() - 1 - skip_offset) / skip_width + 1;
                            CuMatrix<BaseFloat> skip_feat(skip_len, feat_transf.NumCols());
                            Posterior skip_target(skip_len);
                            for (int i = 0; i < skip_len; i++) {
                                skip_feat.Row(i).CopyFromVec(feat_transf.Row(i * skip_width + skip_offset));
                                skip_target[i] = target[i * skip_width + skip_offset];
                            }
                            feats[s].Resize(skip_feat.NumRows(), skip_feat.NumCols());
                            skip_feat.CopyToMat(&feats[s]); 
                            targets[s] = skip_target;
                        } else {
                            feats[s].Resize(feat_transf.NumRows(), feat_transf.NumCols());
                            feat_transf.CopyToMat(&feats[s]); 
                            targets[s] = target;
                        }
                        // checks ok, put the data in the buffers,
                        keys[s] = key;
                        curt[s] = 0;
                        lent[s] = feats[s].NumRows();
                        new_utt_flags[s] = 1;  // a new utterance feeded to this stream
                        feature_reader.Next();
                        break;
                    }
                }

                // we are done if all streams are exhausted
                int done = 1;
                for (int s = 0; s < num_stream; s++) {
                    if (curt[s] < lent[s]) done = 0;  // this stream still contains valid data, not exhausted
                }
                if (done) break;

                // fill a multi-stream bptt batch
                // * frame_mask: 0 indicates padded frames, 1 indicates valid frames
                // * target: padded to batch_size
                // * feat: first shifted to achieve targets delay; then padded to batch_size
                for (int t = 0; t < batch_size; t++) {
                    for (int s = 0; s < num_stream; s++) {
                        // frame_mask & targets padding
                        if (curt[s] < lent[s]) {
                            frame_mask(t * num_stream + s) = 1;
                            target[t * num_stream + s] = targets[s][curt[s]];
                        } else {
                            frame_mask(t * num_stream + s) = 0;
                            target[t * num_stream + s] = targets[s][lent[s]-1];
                        }
                        // feat shifting & padding
                        if (curt[s] + targets_delay < lent[s]) {
                            feat.Row(t * num_stream + s).CopyFromVec(feats[s].Row(curt[s]+targets_delay));
                        } else {
                            feat.Row(t * num_stream + s).CopyFromVec(feats[s].Row(lent[s]-1));
                        }

                        curt[s]++;
                    }
                }

                // for streams with new utterance, history states need to be reset
                nnet.ResetLstmStreams(new_utt_flags);

                // forward pass
                if (!crossvalidate) {
                    nnet.Propagate(CuMatrix<BaseFloat>(feat), &nnet_out);
                } else {
                    nnet.Feedforward(CuMatrix<BaseFloat>(feat), &nnet_out);
                }

                xent.Eval(frame_mask, nnet_out, target, &obj_diff);

                // backward pass
                if (!crossvalidate) {
                    nnet.Backpropagate(obj_diff, NULL);
                }

                int frame_progress = frame_mask.Sum();
                total_frames += frame_progress;

                int num_done_progress = 0;
                for (int i =0; i < new_utt_flags.size(); i++) {
                    num_done_progress += new_utt_flags[i];
                }
                num_done += num_done_progress;
                num_sentence += num_done_progress;
                // Report likelyhood
                if (num_sentence >= report_period) {
                    KALDI_LOG << xent.Report();
                    num_sentence -= report_period;
                }

                if (dump_interval > 0) { // disabled by 'dump_interval == 0',
                    if ((num_done-num_done_progress)/dump_interval != (num_done/dump_interval)) {
                        char nnet_name[512];
                        if (!crossvalidate) {
                            sprintf(nnet_name, "%s_utt%d", target_model_filename.c_str(), num_done);
                            nnet.Write(nnet_name, binary);
                        }
                    }
                }

            } // while (1)

        } // for (skip_offset)

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

        KALDI_LOG << xent.Report();

#if HAVE_CUDA==1
        CuDevice::Instantiate().PrintProfile();
#endif

        return 0;
    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}
