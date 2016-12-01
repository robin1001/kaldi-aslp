// aslp-nnetbin/aslp-nnet-train-lstm-subsequence-stream.cc
// Copyright 2016  ASLP (Author: liwenpeng zhangbinbin)
 
// Created on 2016-08-24


#include "aslp-nnet/nnet-trnopts.h"
#include "aslp-nnet/nnet-nnet.h"
#include "aslp-nnet/nnet-loss.h"
#include "aslp-nnet/nnet-randomizer.h"
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"
#include "aslp-cudamatrix/cu-device.h"
#include "aslp-nnet/data-reader.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::aslp_nnet;
  typedef kaldi::int32 int32;  
  
  try {
    const char *usage =
        "Perform one iteration of LSTM training by Stochastic Gradient Descent.\n"
        "This version use pdf-posterior as targets, prepared typically by ali-to-post.\n"
        "The updates are done per-utterance, shuffling options are dummy for compatibility reason.\n"
        "\n"
        "Usage: aslp-nnet-train-lstm-streams [options] <feature-rspecifier> <targets-rspecifier> <model-in> [<model-out>]\n"
        "e.g.: \n"
        " aslp-nnet-train-lstm-streams scp:feature.scp ark:posterior.ark nnet.init nnet.iter1\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts;
    trn_opts.Register(&po);
	// Add dummy randomizer options, to make the tool compatible with standard scripts
	NnetDataRandomizerOptions rnd_opts;
    rnd_opts.Register(&po);
    SequenceDataReaderOptions read_opts;
	read_opts.Register(&po);
   
   bool binary = true, 
   		crossvalidate = false;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("cross-validate", &crossvalidate, "Perform cross-validation (don't backpropagate)");

    std::string objective_function = "xent";
    po.Register("objective-function", &objective_function, "Objective function : xent|mse");

    std::string use_gpu="yes";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 
	
	int32 gpu_id = -1;
    po.Register("gpu-id", &gpu_id, "selected gpu id, if negative then select automaticly");
	bool randomize = false;
    po.Register("randomize", &randomize, "Dummy option, for compatibility...");
    int report_period = 200; // 200 sentence with one report 
    po.Register("report-period", &report_period, "Number of sentence for one report log, default(200)");
   	int32 dump_interval=0;
   	po.Register("dump-interval", &dump_interval, "---LSTM--- num utts between model dumping [ 0 == disabled ]");
    
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

    kaldi::int64 total_frames = 0;
	int32 num_done = 0, num_sentence = 0;

    RandomizerMask randomizer_mask(rnd_opts);
    MatrixRandomizer feature_randomizer(rnd_opts);
    PosteriorRandomizer targets_randomizer(rnd_opts);
    VectorRandomizer weights_randomizer(rnd_opts);
	
	LossItf *loss = NULL;
	if (objective_function == "xent") {
		loss = new Xent;
	} else if (objective_function == "mse") {
		loss = new Mse;
	} else {
		KALDI_ERR << "Unsupported objective function: " << objective_function;
	}
    
    Timer time;
    KALDI_LOG << (crossvalidate?"CROSS-VALIDATION":"TRAINING") << " STARTED";
	
	SequenceDataReader reader(feature_rspecifier, targets_rspecifier, read_opts);

    CuMatrix<BaseFloat> nnet_out, obj_diff;
	CuMatrix<BaseFloat> nnet_in;
	Vector<BaseFloat> frame_mask;
	Posterior nnet_tgt;	
	
	while (!reader.Done()) {
        
		reader.ReadData(&nnet_in, &nnet_tgt, &frame_mask);	
		// for streams with new utterance, history states need to be reset
		std::vector<int> new_utt_flags;
		new_utt_flags = reader.GetNewUttFlags();
		nnet.ResetLstmStreams(new_utt_flags);

        // forward pass
        if (!crossvalidate) {
            nnet.Propagate(nnet_in, &nnet_out);
        } else {
            nnet.Feedforward(nnet_in, &nnet_out);
        }
  		//evalute objective function we've chose 
   		loss->Eval(frame_mask, nnet_out, nnet_tgt, &obj_diff);
    
        // backward pass
        if (!crossvalidate) {
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
            KALDI_LOG << loss->Report();
            num_sentence -= report_period;
        }

        
        // monitor the NN training
        if (kaldi::g_kaldi_verbose_level >= 2) { // vlog-2
            if ((total_frames-frame_progress)/25000 != (total_frames/25000)) { // print every 25k frames
                KALDI_VLOG(2) << "### After " << total_frames << " frames,";
                KALDI_VLOG(2) << nnet.InfoPropagate();
                if (!crossvalidate) {
                    KALDI_VLOG(2) << nnet.InfoBackPropagate();
                    KALDI_VLOG(2) << nnet.InfoGradient();
                }
            }
        }

        // report the speed
        if ((num_done-num_done_progress)/1000 != (num_done/1000)) {
            double time_now = time.Elapsed();
            KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                        << time_now/60 << " min; processed " << total_frames/time_now
                        << " frames per second.";
            
#if HAVE_CUDA==1
            // check the GPU is not overheated
            CuDevice::Instantiate().CheckGpuHealth();
#endif
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

    KALDI_LOG << "Done " << num_done << " files, " 
			  << "[" << (crossvalidate?"CROSS-VALIDATION":"TRAINING")
              << ", " << (randomize?"RANDOMIZED":"NOT-RANDOMIZED") 
              << ", " << time.Elapsed()/60 << " min, fps" << total_frames/time.Elapsed()
              << "]";  
	loss->Report();
	if (loss != NULL) delete loss;
/*
 * if (objective_function == "xent") {
      KALDI_LOG << xent.Report();
    } else if (objective_function == "mse") {
      KALDI_LOG << mse.Report();
    } else {
      KALDI_ERR << "Unknown objective function code : " << objective_function;
    }
*
* */
#if HAVE_CUDA==1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
