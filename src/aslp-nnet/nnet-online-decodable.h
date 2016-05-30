// aslp-nnet/nnet-online-decodable.h

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

#ifndef ASLP_ONLINE_NNET_DECODABLE_H_
#define ASLP_ONLINE_NNET_DECODABLE_H_

#include "itf/online-feature-itf.h"
#include "itf/decodable-itf.h"
#include "hmm/transition-model.h"

#include "aslp-nnet/nnet-nnet.h"

namespace kaldi {
namespace aslp_nnet {

struct DecodableNnetOnlineOptions {
    BaseFloat acoustic_scale;
    int skip_width;
    std::string skip_type;
    int32 max_nnet_batch_size;

    DecodableNnetOnlineOptions():
        acoustic_scale(0.1),
        skip_width(0),
        skip_type("copy"),
        max_nnet_batch_size(256) { }

    void Register(OptionsItf *opts) {
        opts->Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
        opts->Register("skip-width", &skip_width, 
                 "num of frame for one skip(default 0, not use skip)");
        opts->Register("skip-type", &skip_type, 
                 "decode type using skip, copy or split");
        opts->Register("max-nnet-batch-size", &max_nnet_batch_size,
                "Maximum batch size we use in neural-network decodable object, "
                "in cases where we are not constrained by currently available "
                "frames (this will rarely make a difference)");

    }
};


class DecodableNnetOnline: public DecodableInterface {
public:
    DecodableNnetOnline(Nnet *nnet,
                        const CuVector<BaseFloat> &log_priors,
                        const TransitionModel &trans_model,
                        const DecodableNnetOnlineOptions &opts,
                        OnlineFeatureInterface *input_feats);

    /// Returns the scaled log likelihood
    virtual BaseFloat LogLikelihood(int32 frame, int32 index);

    virtual bool IsLastFrame(int32 frame) const;

    virtual int32 NumFramesReady() const;  

    /// Indices are one-based!  This is for compatibility with OpenFst.
    virtual int32 NumIndices() const { return trans_model_.NumTransitionIds(); }

private:

    /// If the neural-network outputs for this frame are not cached, it computes
    /// them (and possibly for some succeeding frames)
    void ComputeForFrame(int32 frame);

    Nnet *nnet_;
    const CuVector<BaseFloat> &log_priors_;  // log-priors taken from the model.
    const TransitionModel &trans_model_;
    OnlineFeatureInterface *features_;
    DecodableNnetOnlineOptions opts_;
    int32 feat_dim_;  // dimensionality of the input features.
    int32 num_pdfs_;  // Number of pdfs, equals output-dim of the network (cached
    // here)

    int32 begin_frame_;  // First frame for which scaled_loglikes_ is valid
    // (i.e. the first frame of the batch of frames for
    // which we've computed the output).

    // scaled_loglikes_ contains the neural network pseudo-likelihoods: the log of
    // (prob divided by the prior), scaled by opts.acoustic_scale).  We may
    // compute this using the GPU, but we transfer it back to the system memory
    // when we store it here.  These scores are only kept for a subset of frames,
    // starting at begin_frame_, whose length depends how many frames were ready
    // at the time we called LogLikelihood(), and will never exceed
    // opts_.max_nnet_batch_size.
    Matrix<BaseFloat> scaled_loglikes_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableNnetOnline);
};

} // namespace aslp_nnet
} // namespace kaldi

#endif // ASLP_ONLINE_NNET_DECODABLE_H_
