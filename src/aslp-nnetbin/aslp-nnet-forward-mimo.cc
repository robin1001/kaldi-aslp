// aslp-nnetbin/aslp-nnet-forward-mimo.cc

// Copyright 2016  ASLP (Author: zhangbinbin)
// Created on 2016-03-10

#include <limits>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/timer.h"

#include "aslp-nnet/nnet-nnet.h"
#include "aslp-nnet/nnet-loss.h"
#include "aslp-nnet/nnet-pdf-prior.h"


int main(int argc, char *argv[]) {
    using namespace kaldi;
    using namespace kaldi::aslp_nnet;
    try {
        const char *usage =
            "Perform forward pass through Neural Network.\n"
            "\n"
            "Usage:  aslp-nnet-forward-mimo [options] <model-in> <feature-rspecifier_1>...<feature_rspecifier_n> <feature-wspecifier>\n"
            "e.g.: \n"
            " aslp-nnet-forward-mimo nnet ark:features1.ark ark:features2.ark ark:mlpoutput.ark\n";

        ParseOptions po(usage);

        PdfPriorOptions prior_opts;
        prior_opts.Register(&po);

        std::string feature_transform;
        po.Register("feature-transform", &feature_transform, "Feature transform in front of main network (in nnet format)");

        bool no_softmax = false;
        po.Register("no-softmax", &no_softmax, "No softmax on MLP output (or remove it if found), the pre-softmax activations will be used as log-likelihoods, log-priors will be subtracted");
        bool apply_log = false;
        po.Register("apply-log", &apply_log, "Transform MLP output to logscale");

        std::string use_gpu="no";
        po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 

        using namespace kaldi;
        using namespace kaldi::aslp_nnet;
        typedef kaldi::int32 int32;

        int32 time_shift = 0;
        po.Register("time-shift", &time_shift, "LSTM : repeat last input frame N-times, discrad N initial output frames."); 

        po.Read(argc, argv);

        if (po.NumArgs() < 3) {
            po.PrintUsage();
            exit(1);
        }
        // avoid some bad option combinations,
        if (apply_log && no_softmax) {
            KALDI_ERR << "Cannot use both --apply-log=true --no-softmax=true, use only one of the two!";
        }
        //Select the GPU
#if HAVE_CUDA==1
        CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

        int num_args = po.NumArgs();
        std::string model_filename = po.GetArg(1),
            feature_wspecifier = po.GetArg(num_args);

        Nnet nnet;
        nnet.Read(model_filename);
        // Here check params, parse feature_rspecifiers
        int num_input = nnet.NumInput(), num_output = nnet.NumOutput();
        KALDI_LOG << "Nnet num_input " << num_input << " num_output " << num_output;
        if (num_args != 1 + num_input + 1) {
            po.PrintUsage();
            exit(1);
        }
        std::vector<std::string> feature_rspecifiers;
        for (int i = 0; i < num_input; i++) {
            feature_rspecifiers.push_back(po.GetArg(i+2));
        }
        // we will subtract log-priors later,
        PdfPrior pdf_prior(prior_opts); 
        // disable dropout,
        nnet.SetDropoutRetention(1.0);

        kaldi::int64 tot_t = 0;
        std::vector <SequentialBaseFloatMatrixReader *> feature_readers;
        feature_readers.resize(num_input, NULL);
        for (int i = 0; i < num_input; i++) {
            feature_readers[i] = new SequentialBaseFloatMatrixReader(feature_rspecifiers[i]);
        }
        BaseFloatMatrixWriter feature_writer(feature_wspecifier);

        std::vector<const CuMatrixBase<BaseFloat > *> nnet_in;
        std::vector<CuMatrix<BaseFloat> *> nnet_outs;
        Matrix<BaseFloat> nnet_out_host;
        nnet_in.resize(num_input, NULL);
        nnet_outs.resize(num_output, NULL);
        for (int i = 0; i < num_output; i++) {
            nnet_outs[i] = new CuMatrix<BaseFloat>;
        }

        Timer time;
        double time_now = 0;
        int32 num_done = 0;
        while (true) {
            // Check if done
            if (feature_readers[0]->Done()) {
                for (int i = 1; i < num_input; i++) {
                    KALDI_ASSERT(feature_readers[i]->Done());
                }
                break;
            }
            // Add feature
            std::string utt = feature_readers[0]->Key();
            KALDI_VLOG(2) << "Processing " << utt;
            for (int i = 0; i < num_input; i++) {
                std::string utti = feature_readers[i]->Key();
                if (utti != utt) {
                    KALDI_ERR << "Different key from the features "
                              << utt << " " << utti 
                              << " please check the order of feat scp";
                }
                Matrix<BaseFloat> mat = feature_readers[i]->Value();
                if (!KALDI_ISFINITE(mat.Sum())) { // check there's no nan/inf,
                    KALDI_ERR << "NaN or inf found in features for " << utt;
                }
                // time-shift, copy the last frame of LSTM input N-times,
                if (time_shift > 0) {
                    int32 last_row = mat.NumRows() - 1; // last row,
                    mat.Resize(mat.NumRows() + time_shift, mat.NumCols(), kCopyData);
                    for (int32 r = last_row+1; r<mat.NumRows(); r++) {
                        mat.CopyRowFromVec(mat.Row(last_row), r); // copy last row,
                    }
                }
                CuMatrix<BaseFloat> *cu_mat = new CuMatrix<BaseFloat>;
                *(cu_mat) = mat;
                nnet_in[i] = const_cast<const CuMatrix<BaseFloat> *>(cu_mat);
            }
            // Feedforward
            nnet.Feedforward(nnet_in, &nnet_outs);
            // If multitask, only write the last task out
            CuMatrix<BaseFloat> &nnet_out  = *(nnet_outs[num_output-1]);
            // convert posteriors to log-posteriors,
            if (apply_log) {
                if (!(nnet_out.Min() >= 0.0 && nnet_out.Max() <= 1.0)) {
                    KALDI_WARN << utt << " "
                        << "Applying 'log' to data which don't seem to be probabilities "
                        << "(is there a softmax somwhere?)";
                }
                nnet_out.Add(1e-20); // avoid log(0),
                nnet_out.ApplyLog();
            }
            // subtract log-priors from log-posteriors or pre-softmax,
            if (prior_opts.class_frame_counts != "") {
                if (nnet_out.Min() >= 0.0 && nnet_out.Max() <= 1.0) {
                    KALDI_WARN << utt << " " 
                        << "Subtracting log-prior on 'probability-like' data in range [0..1] " 
                        << "(Did you forget --no-softmax=true or --apply-log=true ?)";
                }
                pdf_prior.SubtractOnLogpost(&nnet_out);
            }

            // download from GPU,
            nnet_out_host.Resize(nnet_out.NumRows(), nnet_out.NumCols());
            nnet_out.CopyToMat(&nnet_out_host);

            // time-shift, remove N first frames of LSTM output,
            if (time_shift > 0) {
                Matrix<BaseFloat> tmp(nnet_out_host);
                nnet_out_host = tmp.RowRange(time_shift, tmp.NumRows() - time_shift);
            }

            // write,
            if (!KALDI_ISFINITE(nnet_out_host.Sum())) { // check there's no nan/inf,
                KALDI_ERR << "NaN or inf found in final output nn-output for " << utt;
            }
            feature_writer.Write(utt, nnet_out_host);

            // progress log
            if (num_done % 100 == 0) {
                time_now = time.Elapsed();
                KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                    << time_now/60 << " min; processed " << tot_t/time_now
                    << " frames per second.";
            }
            num_done++;
            tot_t += nnet_in[0]->NumRows();
            // Delete pointer
            for (int i = 0; i < num_input; i++) {
                feature_readers[i]->Next();
                delete nnet_in[i];
            }
        }

        for (int i = 0; i < num_input; i++) {
            delete feature_readers[i];
        }
        for (int i = 0; i < num_output; i++) {
            delete nnet_outs[i];
        }

        // final message
        KALDI_LOG << "Done " << num_done << " files" 
            << " in " << time.Elapsed()/60 << "min," 
            << " (fps " << tot_t/time.Elapsed() << ")"; 

#if HAVE_CUDA==1
        if (kaldi::g_kaldi_verbose_level >= 1) {
            CuDevice::Instantiate().PrintProfile();
        }
#endif

        if (num_done == 0) return -1;
        return 0;
    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}
