// aslp-nnetbin/aslp-nnet-forward-blstm-lc.cc

// Copyright 2011-2013  Brno University of Technology (Author: Karel Vesely)
// Copyright 2016  ASLP (Author: liwenpeng zhangbinbin duwei)

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
  typedef kaldi::int32 int32;

  try {
    const char *usage =
        "Perform forward pass for Latency Control BLSTM through Neural Network.\n"
        "\n"
        "Usage:  aslp-nnet-forward-blstm-lc [options] <model-in> <feature-rspecifier> <feature-wspecifier>\n"
        "e.g.: \n"
        " aslp-nnet-forward-blstm-lc nnet ark:features.ark ark:mlpoutput.ark\n";

    ParseOptions po(usage);

    PdfPriorOptions prior_opts;
    prior_opts.Register(&po);
	
	int32 chunk_size = 64;
	po.Register("chunk-size", &chunk_size, "---BLSTM--- Latency-controlled BPTT chunk size, must be same with training");
	
	int32 right_splice = 16;
	po.Register("right-splice", &right_splice, "---BLSTM--- Latency-controlled BPTT right context size, must be same with training");

	// ---BLSTM--- Latency-controlled BPTT batch size
	int32 batch_size;
	batch_size = chunk_size + right_splice;

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, "Feature transform in front of main network (in nnet format)");

    bool no_softmax = false;
    po.Register("no-softmax", &no_softmax, "No softmax on MLP output (or remove it if found), the pre-softmax activations will be used as log-likelihoods, log-priors will be subtracted");
    bool apply_log = true;
    po.Register("apply-log", &apply_log, "Transform MLP output to logscale");

    std::string use_gpu="no";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA"); 

    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        feature_wspecifier = po.GetArg(3);
        
    //Select the GPU
#if HAVE_CUDA==1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);
    // optionally remove softmax,
    //Component::ComponentType last_type = nnet.GetComponent(nnet.NumComponents()-1).GetType();
    //if (no_softmax) {
    //  if (last_type == Component::kSoftmax || last_type == Component::kBlockSoftmax) {
    //    KALDI_LOG << "Removing " << Component::TypeToMarker(last_type) << " from the nnet " << model_filename;
    //    nnet.RemoveComponent(nnet.NumComponents()-1);
    //  } else {
    //    KALDI_WARN << "Cannot remove softmax using --no-softmax=true, as the last component is " << Component::TypeToMarker(last_type);
    //  }
    //}

    // avoid some bad option combinations,
    if (apply_log && no_softmax) {
      KALDI_ERR << "Cannot use both --apply-log=true --no-softmax=true, use only one of the two!";
    }

    // we will subtract log-priors later,
    PdfPrior pdf_prior(prior_opts); 

    // disable dropout,
    nnet_transf.SetDropoutRetention(1.0);
    nnet.SetDropoutRetention(1.0);
    // set chunk_size for latency control blstm
    nnet.SetChunkSize(chunk_size);

    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    BaseFloatMatrixWriter feature_writer(feature_wspecifier);

    CuMatrix<BaseFloat> feats, feats_transf, nnet_in, nnet_out, nnet_out_chunk;
    Matrix<BaseFloat> nnet_out_host;
	
	int32 feat_dim = nnet.InputDim();
	int32 out_dim = nnet.OutputDim();

    Timer time;
	std::string utt;
	double time_now = 0;
    int32 num_done = 0;

    // iterate over all feature files
    for (; !feature_reader.Done(); feature_reader.Next()) {
      // read
      Matrix<BaseFloat> mat = feature_reader.Value();
      std::string utt = feature_reader.Key();
      KALDI_VLOG(2) << "Processing utterance " << num_done+1 
                    << ", " << utt
                    << ", " << mat.NumRows() << "frm";
 
      if (!KALDI_ISFINITE(mat.Sum())) { // check there's no nan/inf,
        KALDI_ERR << "NaN or inf found in features for " << utt;
      }

      // push it to gpu,
      feats = mat;

      // fwd-pass, feature transform,
      nnet_transf.Feedforward(feats, &feats_transf);
      if (!KALDI_ISFINITE(feats_transf.Sum())) { // check there's no nan/inf,
        KALDI_ERR << "NaN or inf found in transformed-features for " << utt;
      }

	  // for streams with new utterance, history states need to be reset
      std::vector<int32> reset_flags(1, 1);
      nnet.ResetLstmStreams(reset_flags);

      int num_frames = feats_transf.NumRows();
      int num_chunks = (num_frames - 1) / chunk_size + 1;
      nnet_out.Resize(num_frames, out_dim);
      nnet_in.Resize(batch_size, feat_dim);
      // forward in batch for latency control BLSTM
      for (int i = 0; i < num_chunks; i++) {
        int offset = i * chunk_size;
        int len = offset + batch_size < num_frames ? batch_size : num_frames - offset;
        int copy_len = offset + chunk_size < num_frames ? chunk_size : num_frames - offset;
        KALDI_ASSERT(len <= batch_size);
        //KALDI_LOG << i << " " <<  offset << " " << len;
        nnet_in.RowRange(0, len).CopyFromMat(
            feats_transf.RowRange(offset, len));
		nnet.Feedforward(nnet_in, &nnet_out_chunk);
        nnet_out.RowRange(offset, copy_len).CopyFromMat(
            nnet_out_chunk.RowRange(0, copy_len));
      }

      if (!KALDI_ISFINITE(nnet_out.Sum())) { // check there's no nan/inf,
        KALDI_ERR << "NaN or inf found in nn-output for " << utt;
      }
      
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

      // write,
      if (!KALDI_ISFINITE(nnet_out_host.Sum())) { // check there's no nan/inf,
        KALDI_ERR << "NaN or inf found in final output nn-output for " << utt;
      }
      feature_writer.Write(feature_reader.Key(), nnet_out_host);

      // progress log
      if (num_done % 100 == 0) {
        time_now = time.Elapsed();
        KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                      << time_now/60 << " min; processed " << tot_t/time_now
                      << " frames per second.";
      }
      num_done++;
      tot_t += mat.NumRows();

    }

		// final message
	KALDI_LOG << "Done " <<  num_done << "files"
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
