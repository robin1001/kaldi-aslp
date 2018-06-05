// aslp-nnetbin/aslp-nnet-train-frame.cc

// Copyright 2016  ASLP (Author: zhangbinbin)

// Created on 2016-03-09

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "aslp-cudamatrix/cu-device.h"

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
            "It is same to aslp-nnet-train-simple, but use FrameDataReader to read feat and label.\n"
            "This version use pdf-posterior as targets, prepared typically by ali-to-post.\n"
            "Usage:  aslp-nnet-train-frame [options] <feature-rspecifier> <targets-rspecifier> <model-in> [<model-out>]\n"
            "e.g.: \n"
            " aslp-nnet-train-frame scp:feature.scp ark:posterior.ark nnet.init nnet.iter1\n";

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
        int32 gpu_id = -1;
        po.Register("gpu-id", &gpu_id, "selected gpu id, if negative then select automaticly");

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
        if (gpu_id >= 0) {
            CuDevice::Instantiate().SetGpuId(gpu_id);
        } else {
            CuDevice::Instantiate().SelectGpuId(use_gpu);
        }
#endif
        Nnet nnet;
        nnet.Read(model_filename);
        nnet.SetTrainOptions(trn_opts);

        if (dropout_retention > 0.0) {
            nnet.SetDropoutRetention(dropout_retention);
        }
        if (crossvalidate) {
            nnet.SetDropoutRetention(1.0);
        }

        LossItf *loss = NULL;
        if (objective_function == "xent") {
            loss = new Xent;
        } else if (objective_function == "mse") {
            loss = new Mse;
        } else {
            KALDI_ERR << "Unsupported objective function: " << objective_function;
        }

        Timer time;
        kaldi::int64 total_frames = 0, report_frames = 0;
        KALDI_LOG << (crossvalidate?"CROSS-VALIDATION":"TRAINING") << " STARTED";

        FrameDataReader reader(feature_rspecifier, targets_rspecifier, rnd_opts);

        const CuMatrixBase<BaseFloat> *nnet_in;
        CuMatrix<BaseFloat> nnet_out, obj_diff;
        const Posterior *nnet_tgt;

        while (!reader.Done()) {
            bool ok = reader.ReadData(&nnet_in, &nnet_tgt); 
            if (!ok) continue;
            // Forward pass
            if (!crossvalidate) {
                nnet.Propagate(*nnet_in, &nnet_out);
            } else {
                nnet.Feedforward(*nnet_in, &nnet_out);
            }
            // Eval loss
            loss->Eval(nnet_out, *nnet_tgt, &obj_diff);
            // Backward pass
            if (!crossvalidate) {
                nnet.Backpropagate(obj_diff, NULL);
            }
            total_frames += nnet_in->NumRows();
            report_frames += nnet_in->NumRows();
            // Report
            if (report_period > 0 && report_frames >= report_period) {
                KALDI_LOG << loss->Report();
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

        KALDI_LOG << loss->Report();
        if (loss != NULL) delete loss;

#if HAVE_CUDA==1
        CuDevice::Instantiate().PrintProfile();
#endif
        return 0;
    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}
