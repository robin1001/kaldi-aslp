// aslp-nnetbin/aslp-nnet-train-frame-worker.cc

// Copyright 2016  ASLP (Author: zhangbinbin)

// Created on 2016-08-01

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "aslp-cudamatrix/cu-device.h"

#include "aslp-nnet/nnet-trnopts.h"
#include "aslp-nnet/nnet-nnet.h"
#include "aslp-nnet/nnet-loss.h"
#include "aslp-nnet/data-reader.h"

#include "aslp-parallel/itf.h"
#include "aslp-parallel/bsp-worker.h"
#include "aslp-parallel/easgd-worker.h"
#include "aslp-parallel/bmuf-worker.h"


int main(int argc, char *argv[]) {
    using namespace kaldi;
    using namespace kaldi::aslp_nnet;
    typedef kaldi::int32 int32;  
    try {
        const char *usage =
            "Parallel worker of aslp-nnet-train-frame, but don't do cross validation"
            "see aslp-nnet-train-frame for details\n"
            "Usage:  aslp-nnet-train-frame-worker [options] "
            "<feature-rspecifier> <targets-rspecifier> <model-in> [<model-out>]\n"
            "e.g.: \n"
            " aslp-nnet-train-frame-worker scp:feature.scp ark:posterior.ark nnet.init nnet.iter1\n";

        ParseOptions po(usage);

        NnetTrainOptions trn_opts;
        trn_opts.Register(&po);
        NnetDataRandomizerOptions rnd_opts;
        rnd_opts.Register(&po);

        bool binary = true, 
             randomize = true;
        po.Register("binary", &binary, "Write output in binary mode");
        po.Register("randomize", &randomize, "Perform the frame-level shuffling within the Cache::");

        std::string objective_function = "xent";
        po.Register("objective-function", &objective_function, "Objective function : xent|mse");
        std::string use_gpu="yes";
        po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");

        double dropout_retention = 0.0;
        po.Register("dropout-retention", &dropout_retention, "number between 0..1, saying how many neurons to preserve (0.0 will keep original value");
        int report_period = -1; // 
        po.Register("report-period", &report_period, "Number of frames for one report log, default(-1, no report)");

        // for worker
        std::string worker_type = "bsp";
        po.Register("worker-type", &worker_type, "Worker type(bsp | bmuf | easgd)");
        float alpha = 0.5;
        po.Register("alpha", &alpha, "Moving rate alpha for easgd worker");
        float bmuf_momentum = 0.9;
        po.Register("bmuf-momentum", &bmuf_momentum, "momentum for bmuf worker");
        float bmuf_learn_rate = 1.0;
        po.Register("bmuf-learn-rate", &bmuf_learn_rate, "learn rate for bmuf worker");
        int sync_period = 25600;
        po.Register("sync-period", &sync_period, "number frames for every synchronization");
        
        po.Read(argc, argv);

        if (po.NumArgs() != 4) {
            po.PrintUsage();
            exit(1);
        }
        std::string feature_rspecifier = po.GetArg(1),
            targets_rspecifier = po.GetArg(2),
            model_filename = po.GetArg(3),
            target_model_filename = po.GetArg(4);

        //Select the GPU
#if HAVE_CUDA==1
        CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif
        Nnet nnet;
        nnet.Read(model_filename);
        nnet.SetTrainOptions(trn_opts);

        if (dropout_retention > 0.0) {
            nnet.SetDropoutRetention(dropout_retention);
        }

        LossItf *loss = NULL;
        if (objective_function == "xent") {
            loss = new Xent;
        } else if (objective_function == "mse") {
            loss = new Mse;
        } else {
            KALDI_ERR << "Unsupported objective function: " << objective_function;
        }

        // Init Worker
        IWorker *worker = NULL;
        if (worker_type == "bsp") {
            worker = new BspWorker();
        } else if (worker_type == "easgd") {
            worker = new EasgdWorker(alpha);
        } else if (worker_type == "bmuf") {
            worker = new BmufWorker(bmuf_learn_rate, bmuf_momentum);
        } else {
            KALDI_ERR << "Unsupported worker type: " << worker_type;
        }

        std::vector<std::pair<BaseFloat *, int> > params;
        nnet.GetGpuParams(&params);
        worker->InitParam(params);
        KALDI_LOG << "Mpi cluster info total " << worker->NumNodes() 
                  << " worker rank " << worker->Rank();
        
        Timer time;
        kaldi::int64 total_frames = 0, report_frames = 0;
        int num_frames_since_last_sync = 0;
        KALDI_LOG << "TRAINING STARTED";

        FrameDataReader reader(feature_rspecifier, targets_rspecifier, rnd_opts);

        const CuMatrixBase<BaseFloat> *nnet_in;
        CuMatrix<BaseFloat> nnet_out, obj_diff;
        const Posterior *nnet_tgt;


        while (!reader.Done()) {
            reader.ReadData(&nnet_in, &nnet_tgt); 
            // Forward pass
            nnet.Propagate(*nnet_in, &nnet_out);
            // Eval loss
            loss->Eval(nnet_out, *nnet_tgt, &obj_diff);
            // Backward pass
            nnet.Backpropagate(obj_diff, NULL);

            total_frames += nnet_in->NumRows();
            report_frames += nnet_in->NumRows();
            num_frames_since_last_sync += nnet_in->NumRows();
            // Do Synchronize
            if (num_frames_since_last_sync > sync_period) {
                KALDI_LOG << "Worker " << worker->Rank() << " synchronize once";
                worker->Synchronize(num_frames_since_last_sync);
                num_frames_since_last_sync = 0;
            }
            // Report
            if (report_period > 0 && report_frames >= report_period) {
                KALDI_LOG << loss->Report();
                report_frames -= report_period;
            }
        }

        // Stop worker
        worker->Stop();

        if ((worker_type == "bsp" || worker_type == "bmuf") && 
                worker->IsMainNode()) {
            nnet.Write(target_model_filename, binary);
        }

        KALDI_LOG << loss->Report();
        KALDI_LOG << "[" << "TRAINING"
            << ", " << (randomize?"RANDOMIZED":"NOT-RANDOMIZED") 
            << ", " << time.Elapsed()/60 << " min, fps" << total_frames/time.Elapsed()
            << "]";  

        if (loss != NULL) delete loss;
        if (worker != NULL) delete worker;

#if HAVE_CUDA==1
        CuDevice::Instantiate().PrintProfile();
#endif
        return 0;
    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}
