// aslp-nnetbin/aslp-nnet-train-frame-mimo.cc

// Copyright 2016  ASLP (Author: zhangbinbin)

// Created on 2016-03-09

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"

#include "aslp-nnet/nnet-trnopts.h"
#include "aslp-nnet/nnet-nnet.h"
#include "aslp-nnet/nnet-loss.h"
#include "aslp-nnet/data-reader.h"

int main(int argc, char *argv[]) {
    using namespace kaldi;
    using namespace kaldi::aslp_nnet;
    typedef kaldi::int32 int32;  
    try {
        const char *usage =
            "Perform one iteration of Neural Network training by mini-batch Stochastic Gradient Descent.\n"
            "It is same to aslp-nnet-train-frame, but the network has multi input or multi output.\n"
            "Attention: num input feat and target must match the input num and the output num of the nnet\n"
            "This version use pdf-posterior as targets, prepared typically by ali-to-post.\n"
            "Usage:  aslp-nnet-train-frame-mimo [options] <feature-rspecifier_1>...<feature_rspecifier_n> "
            "                   <targets-rspecifier_1>...<targets_rspecifier_n> <model-in> [<model-out>]\n"
            "e.g.: \n"
            " aslp-nnet-train-frame-mimo scp:feature1.scp scp:feature2.scp "
            "                       ark:posterior1.ark ark:posterior2.ark nnet.init nnet.iter1\n";

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

        std::string objective_function = "xent";
        po.Register("objective-function", &objective_function, "Objective function : xent|mse");

        std::string use_gpu="yes";
        po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");

        double dropout_retention = 0.0;
        po.Register("dropout-retention", &dropout_retention, "number between 0..1, saying how many neurons to preserve (0.0 will keep original value");

        int report_period = -1; // 
        po.Register("report-period", &report_period, "Number of frames for one report log, default(-1, no report)");

        po.Read(argc, argv);

        if (po.NumArgs() == 0) {
            po.PrintUsage();
            exit(1);
        }
        int num_args = po.NumArgs();
        std::string model_filename, target_model_filename;
        if (!crossvalidate) {
            model_filename = po.GetArg(num_args - 1);
            target_model_filename = po.GetArg(num_args);
        } else {
            model_filename = po.GetArg(num_args);
        }

        // Select the GPU
#if HAVE_CUDA==1
        CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif
        Nnet nnet;
        nnet.Read(model_filename);
        nnet.SetTrainOptions(trn_opts);
        // Here check params
        int num_input = nnet.NumInput(), num_output = nnet.NumOutput();
        KALDI_LOG << "Nnet num_input " << num_input << " num_output " << num_output;
        int extra = !crossvalidate ? 2:1;
        if (num_args != num_input + num_output + extra) {
            po.PrintUsage();
            exit(1);
        }
        // Get feature and target vector
        std::vector<std::string> features, targets;
        for (int i = 0; i < num_input; i++) 
            features.push_back(po.GetArg(i+1));
        for (int i = 0; i < num_output; i++)
            targets.push_back(po.GetArg(i+num_input+1));

        if (dropout_retention > 0.0) {
            nnet.SetDropoutRetention(dropout_retention);
        }
        if (crossvalidate) {
            nnet.SetDropoutRetention(1.0);
        }

        std::vector<LossItf *> losses(num_output, NULL);
        // Parse loss type
        std::vector<std::string> sub_string;
        SplitStringToVector(objective_function, ":", true, &sub_string);
        if (sub_string.size() != num_output) {
            KALDI_ERR << objective_function 
                      << "obj dim not match the nnet output layers num, need "
                      << num_output << " obj function";
        }
        for (int i = 0; i < losses.size(); i++) {
            if (sub_string[i] == "xent") {
                losses[i] = new Xent;
            } else if (sub_string[i] == "mse") {
                losses[i] = new Mse;
            } else {
                KALDI_ERR << "Unsupported objective function: " << sub_string[i];
            }
        }

        Timer time;
        kaldi::int64 total_frames = 0, report_frames = 0;
        KALDI_LOG << (crossvalidate?"CROSS-VALIDATION":"TRAINING") << " STARTED";

        FrameDataReader reader(features, targets, rnd_opts);

        std::vector<const CuMatrixBase<BaseFloat > *> nnet_in;
        std::vector<const Posterior *> nnet_tgt;
        std::vector<CuMatrix<BaseFloat> *> nnet_out, obj_diff;
        std::vector<const CuMatrixBase<BaseFloat> *> const_obj_diff;
        // Attention: obj_diff and const_obj_diff point to the same mat address
        for (int i = 0; i < num_output; i++) {
            nnet_out.push_back(new CuMatrix<BaseFloat>);
            CuMatrix<BaseFloat> *mat = new CuMatrix<BaseFloat>;
            obj_diff.push_back(mat);
            const_obj_diff.push_back(mat);
        }

        while (!reader.Done()) {
            reader.ReadData(&nnet_in, &nnet_tgt); 
            // Forward pass
            if (!crossvalidate) {
                nnet.Propagate(nnet_in, &nnet_out);
            } else {
                nnet.Feedforward(nnet_in, &nnet_out);
            }
            // Eval loss
            for (int i = 0; i < num_output; i++) {
                losses[i]->Eval(*nnet_out[i], *nnet_tgt[i], obj_diff[i]);
            }
            // Backward pass
            if (!crossvalidate) {
                //nnet.Backpropagate(obj_diff, NULL);
                nnet.Backpropagate(const_obj_diff, NULL);
            }
            total_frames += nnet_in[0]->NumRows();
            report_frames += nnet_in[0]->NumRows();
            // Report
            if (report_period > 0 && report_frames >= report_period) {
                for (int i = 0; i < losses.size(); i++) {
                    KALDI_LOG << "Obj " << "[" << i << "] " << sub_string[i];
                    KALDI_LOG << losses[i]->Report();
                }
                report_frames -= report_period;
            }
        }

        if (!crossvalidate) {
            nnet.Write(target_model_filename, binary);
        }

        KALDI_LOG << "[" << (crossvalidate?"CROSS-VALIDATION":"TRAINING")
            << ", " << (randomize?"RANDOMIZED":"NOT-RANDOMIZED") 
            << ", " << time.Elapsed()/60 << " min, fps" << total_frames/time.Elapsed()
            << "]";  
        for (int i = 0; i < losses.size(); i++) {
            KALDI_LOG << "Obj " << "[" << i << "] " << sub_string[i];
            KALDI_LOG << losses[i]->Report();
        }
        for (int i = 0; i < num_output; i++) {
            delete losses[i];
            delete nnet_out[i];
            delete obj_diff[i];
        }

#if HAVE_CUDA==1
        CuDevice::Instantiate().PrintProfile();
#endif
        return 0;
    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}
