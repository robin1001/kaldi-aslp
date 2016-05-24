// aslp-nnetbin/aslp-nnet-train-warp-ctc-streams.cc

// Copyright 2016  ASLP (Author: Binbin Zhang)

#include "aslp-nnet/nnet-trnopts.h"
#include "aslp-nnet/nnet-nnet.h"
#include "aslp-nnet/warp-ctc.h"
#include "aslp-nnet/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "fstext/fstext-lib.h"

int main(int argc, char *argv[]) {
    using namespace kaldi;
    using namespace kaldi::aslp_nnet;
    typedef kaldi::int32 int32;  

    try {
        const char *usage =
            "Perform one iteration of CTC training by SGD.\n"
            "The updates are done per-utterance and by processing multiple utterances in parallel.\n"
            "\n"
            "Usage: aslp-nnet-train-warp-ctc-streams [options] <feature-rspecifier> <labels-rspecifier> <model-in> [<model-out>]\n"
            "e.g.: \n"
            "aslp-nnet-train-warp-ctc-streams scp:feature.scp ark:labels.ark nnet.init nnet.iter1\n";

        ParseOptions po(usage);

        NnetTrainOptions trn_opts;  // training options
        trn_opts.Register(&po); 

        bool binary = true, 
             crossvalidate = false;
        po.Register("binary", &binary, "Write model  in binary mode");
        po.Register("cross-validate", &crossvalidate, "Perform cross-validation (no backpropagation)");

        int32 num_stream = 5;
        po.Register("num-stream", &num_stream, "Number of sequences processed in parallel");

        double frame_limit = 100000;
        po.Register("frame-limit", &frame_limit, "Max number of frames to be processed");

        // Add dummy randomizer options, to make the tool compatible with standard scripts
        NnetDataRandomizerOptions rnd_opts;
        rnd_opts.Register(&po);
        bool randomize = false;
        po.Register("randomize", &randomize, "Dummy option, for compatibility...");
        int32 report_step=100;
        po.Register("report-step", &report_step, "Step (number of sequences) for status reporting");
        int report_period = 200; // 200 sentence with one report 
        po.Register("report-period", &report_period, "Number of sentence for one report log, default(200)");
        int drop_len = 0;
        po.Register("drop-len", &drop_len, "if Sentence frame length greater than drop_len,"
                                           "then drop it, default(0, no drop)");
        int skip_width = 0;
        po.Register("skip-width", &skip_width, "num of frame for one skip(default 0, not use skip)");
    

        std::string use_gpu="yes";
        po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 

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
        //CuDevice::Instantiate().DisableCaching();
#endif

        Nnet net;
        net.Read(model_filename);
        net.SetTrainOptions(trn_opts);
        float norm_lr = trn_opts.learn_rate;

        kaldi::int64 total_frames = 0;

        // Initialize feature and labels readers
        SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
        RandomAccessInt32VectorReader targets_reader(targets_rspecifier);

        // Initialize CTC optimizer
        WarpCtc ctc;
        bool use_gpu_flags = (use_gpu == "yes") ? true:false;
        ctc.SetUseGpu(use_gpu_flags);
        ctc.SetReportStep(report_step);

        CuMatrix<BaseFloat> net_out, obj_diff;

        Timer time;
        KALDI_LOG << (crossvalidate?"CROSS-VALIDATION":"TRAINING") << " STARTED";

        std::vector< Matrix<BaseFloat> > feats_utt(num_stream);  // Feature matrix of every utterance
        std::vector< std::vector<int> > labels_utt(num_stream);  // Label vector of every utterance
        std::vector< std::string> key_utt(num_stream);
        int32 feat_dim = net.InputDim();

        int32 num_done = 0, num_no_tgt_mat = 0, num_other_error = 0;
        int32 num_sentence = 0;

        while (1) {

            std::vector<int> frame_num_utt;
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
                Matrix<BaseFloat> raw_mat = feature_reader.Value();;
                if (drop_len > 0 && raw_mat.NumRows() > drop_len) {
                    KALDI_WARN << utt << ", too long, droped";
                    continue;
                }

                Matrix<BaseFloat> mat; 
                
                if (skip_width > 1) {
                    int skip_len = (raw_mat.NumRows() - 1) / skip_width + 1;
                    mat.Resize(skip_len, raw_mat.NumCols());
                    for (int i = 0; i < skip_len; i++) {
                        mat.Row(i).CopyFromVec(raw_mat.Row(i * skip_width));
                     }
                } else {
                    mat = raw_mat;
                }

                std::vector<int32> targets = targets_reader.Value(utt);

                if (max_frame_num < mat.NumRows()) max_frame_num = mat.NumRows();
                feats_utt[sequence_index] = mat;
                labels_utt[sequence_index] = targets;
                key_utt[sequence_index] = utt;
                frame_num_utt.push_back(mat.NumRows());
                sequence_index++;
                // If the total number of frames reaches frame_limit, then stop adding more sequences, regardless of whether
                // the number of utterances reaches num_stream or not.
                if (frame_num_utt.size() == num_stream || frame_num_utt.size() * max_frame_num > frame_limit) {
                    feature_reader.Next(); break;
                }
            }
            int32 cur_sequence_num = frame_num_utt.size();
            int32 num_valid_frame = 0;

            // Create the final feature matrix. Every utterance is padded to the max length within this group of utterances
            Matrix<BaseFloat> feat_mat_host(cur_sequence_num * max_frame_num, feat_dim, kSetZero);
            for (int s = 0; s < cur_sequence_num; s++) {
                Matrix<BaseFloat> mat_tmp = feats_utt[s];
                for (int r = 0; r < frame_num_utt[s]; r++) {
                    feat_mat_host.Row(r*cur_sequence_num + s).CopyFromVec(mat_tmp.Row(r));
                }
                num_valid_frame += frame_num_utt[s];
            }        
            // Set the original lengths of utterances before padding
            net.SetSeqLengths(frame_num_utt);
            // Normalize learn rate
            trn_opts.learn_rate = norm_lr / num_valid_frame;
            net.SetTrainOptions(trn_opts);

            // Propagation and CTC training
            if (!crossvalidate) {
                net.Propagate(CuMatrix<BaseFloat>(feat_mat_host), &net_out);
            } else {
                net.Feedforward(CuMatrix<BaseFloat>(feat_mat_host), &net_out);
            }
            //net.Propagate(CuMatrix<BaseFloat>(feat_mat_host), &net_out);
            ctc.Eval(key_utt, frame_num_utt, net_out, labels_utt, &obj_diff);

            // Error rates
            ctc.ErrorRate(frame_num_utt, net_out, labels_utt);

            // Backward pass
            if (!crossvalidate) {
                net.Backpropagate(obj_diff, NULL);
            }

            num_done += cur_sequence_num;
            total_frames += feat_mat_host.NumRows();
            num_sentence += cur_sequence_num;
            // Report likelyhood
            if (num_sentence >= report_period) {
                KALDI_LOG << ctc.Report();
                num_sentence -= report_period;
            }

            if (feature_reader.Done()) break; // end loop of while(1)
        }

        // Print statistics of gradients when training finishes 
        if (!crossvalidate) {
            KALDI_LOG << net.InfoGradient();
        }

        if (!crossvalidate) {
            net.Write(target_model_filename, binary);
        }

        KALDI_LOG << "Done " << num_done << " files, " << num_no_tgt_mat
            << " with no targets, " << num_other_error
            << " with other errors. "
            << "[" << (crossvalidate?"CROSS-VALIDATION":"TRAINING")
            << ", " << time.Elapsed()/60 << " min, fps" << total_frames/time.Elapsed()
            << "]";
        KALDI_LOG << ctc.Report();

#if HAVE_CUDA==1
        CuDevice::Instantiate().PrintProfile();
#endif

        return 0;
    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}
