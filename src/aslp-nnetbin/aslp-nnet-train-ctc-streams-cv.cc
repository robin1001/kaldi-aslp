// aslp-nnetbin/aslp-nnet-train-ctc-streams.cc

// Copyright 2016  ASLP (Author: Binbin Zhang)

#include "aslp-nnet/nnet-trnopts.h"
#include "aslp-nnet/nnet-nnet.h"
#include "aslp-nnet/ctc-loss.h"
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
            "Usage: aslp-nnet-train-ctc-streams [options] <feature-rspecifier> <labels-rspecifier> <model-in> [<model-out>]\n"
            "e.g.: \n"
            "aslp-nnet-train-ctc-streams scp:feature.scp ark:labels.ark nnet.init nnet.iter1\n";

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
        //po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 

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

        kaldi::int64 total_frames = 0;

        // Initialize feature and labels readers
        SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
        RandomAccessInt32VectorReader targets_reader(targets_rspecifier);

        // Initialize CTC optimizer
        Ctc ctc;
        ctc.SetReportStep(report_step);

        CuMatrix<BaseFloat> nnet_out, obj_diff;
        CuMatrix<BaseFloat> feats, skip_feat, skip_out;

        Timer time;
        KALDI_LOG << (crossvalidate?"CROSS-VALIDATION":"TRAINING") << " STARTED";

        //int32 feat_dim = net.InputDim();

        int32 num_done = 0, num_no_tgt_mat = 0, num_other_error = 0;
        int32 num_sentence = 0;

        for ( ; !feature_reader.Done(); feature_reader.Next()) {
            std::string utt = feature_reader.Key();
            // Check that we have targets
            if (!targets_reader.HasKey(utt)) {
                KALDI_WARN << utt << ", missing targets";
                num_no_tgt_mat++;
                continue;
            }

            // Get feature / target pair
            Matrix<BaseFloat> mat = feature_reader.Value();;
            std::vector<int32> targets = targets_reader.Value(utt);

            // Push it to gpu
            feats = mat; 

            // Use split skip prediction
            for (int skip_offset = 0; skip_offset < skip_width; skip_offset++) {
                int skip_len = (feats.NumRows() - 1 - skip_offset) / skip_width + 1;
                skip_feat.Resize(skip_len, feats.NumCols()); 
                for (int i = 0; i < skip_len; i++) {
                    skip_feat.Row(i).CopyFromVec(feats.Row(i * skip_width + skip_offset));
                }
                std::vector<int> frame_num_utt;
                frame_num_utt.push_back(skip_feat.NumRows());
                net.SetSeqLengths(frame_num_utt);
                net.Feedforward(skip_feat, &skip_out);
                // Resize nnet out
                if (nnet_out.NumRows() != feats.NumRows() || 
                        nnet_out.NumCols() != skip_out.NumCols()) {
                    nnet_out.Resize(feats.NumRows(), skip_out.NumCols()); 
                }
                for (int i = 0; i < skip_len; i++) {
                    nnet_out.Row(i * skip_width + skip_offset).CopyFromVec(skip_out.Row(i));
                }
            }
            
            std::vector< std::vector<int> > labels_utt;  // Label vector of every utterance
            labels_utt.push_back(targets);
            std::vector <int> frame_num_utt;
            frame_num_utt.push_back(feats.NumRows());
            std::vector <std::string> key_utt;
            key_utt.push_back(utt);
            //ctc.EvalParallel(frame_num_utt, nnet_out, labels_utt, &obj_diff);
            ctc.EvalParallel(key_utt, frame_num_utt, nnet_out, labels_utt, &obj_diff);
            ctc.ErrorRateMSeq(frame_num_utt, nnet_out, labels_utt);

            num_done += 1;
            total_frames += feats.NumRows();
            num_sentence += 1;
            // Report likelyhood
            KALDI_LOG << ctc.Report();
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
