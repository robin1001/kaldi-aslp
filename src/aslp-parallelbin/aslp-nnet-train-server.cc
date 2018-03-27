// aslp-nnetbin/aslp-nnet-train-server.cc

// Copyright 2016  ASLP (Author: Zhang Binbin)

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
#include "aslp-parallel/easgd-server.h"
#include "aslp-parallel/asgd-server.h"
#include "aslp-parallel/masgd-server.h"


int main(int argc, char *argv[]) {
    using namespace kaldi;
    using namespace kaldi::aslp_nnet;
    typedef kaldi::int32 int32;  
    try {
        const char *usage =
            "Parameter server for training, it can adapt all kinds of wokers,"
            "eg framewise, sequential and stream training\n"
            "Usage:  aslp-nnet-train-server [options] <model-in> <model-out>\n"
            "e.g.: \n"
            " aslp-nnet-train-server nnet.init nnet.out\n";

        ParseOptions po(usage);

        bool binary = true;
        po.Register("binary", &binary, "Write output in binary mode");
        std::string use_gpu="yes";
        po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");

        std::string server_type = "easgd";
        po.Register("server-type", &server_type, "Server type(easgd | asgd)");
        float alpha = 0.5;
        po.Register("alpha", &alpha, "Moving rate alpha for easgd server");
        int sync_period = 1000;
		po.Register("sync-period", &sync_period, "Synchronization period for ASGD");
		int gpu_id = -1;
        po.Register("gpu-id", &gpu_id, "selected gpu id, if negative then select automaticly");
        float masgd_momentum = 0.9;
        po.Register("masgd-momentum", &masgd_momentum, "momentum for masgd");
        
        po.Read(argc, argv);

        if (po.NumArgs() != 2) {
            po.PrintUsage();
            exit(1);
        }
        std::string model_filename = po.GetArg(1),
            target_model_filename = po.GetArg(2);

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

        // Init Server
        IServer *server = NULL;
        if (server_type == "easgd") {
            server = new EasgdServer(alpha);
        } else if (server_type == "asgd") {
            server = new AsgdServer(alpha, sync_period);
        } else if (server_type == "masgd") {
            server = new MasgdServer(sync_period, masgd_momentum);
        }
        else {
            KALDI_ERR << "Unsupported server type: " << server_type;
        }
        std::vector<std::pair<BaseFloat *, int> > params;
        nnet.GetGpuParams(&params);
        server->InitParam(params);
        KALDI_LOG << "Mpi cluster info total " << server->NumNodes() 
                  << " server rank " << server->Rank();
        
        // Run loop until all worker finished
        server->Run();

        // Acc stats
        std::vector<double *> acc_params; 
        std::vector<std::pair<double*, int> > data_params;
        nnet.GetAccStats(&acc_params, &data_params);
        server->ReduceAccStat(acc_params, data_params);

        nnet.Write(target_model_filename, binary);
        if (server != NULL) delete server;

#if HAVE_CUDA==1
        CuDevice::Instantiate().PrintProfile();
#endif
        return 0;
    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}
