// aslp-online/online-nnet-decoder.h

// Copyright 2014      Johns Hopkins University (author: Daniel Povey)
// Copyright 2015-2016 ASLP (zhangbinbin hechangqing)

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


#ifndef ASLP_ONLINE_ONLINE_NNET_DECODER_H_
#define ASLP_ONLINE_ONLINE_NNET_DECODER_H_

#include <string>
#include <vector>
#include <deque>

#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"
#include "decoder/lattice-faster-online-decoder.h"
#include "hmm/transition-model.h"
#include "hmm/posterior.h"

#include "aslp-nnet/nnet-decodable.h"
#include "aslp-online/online-endpoint.h"
#include "aslp-online/online-helper.h"
#include "aslp-online/online-feature-pipeline.h"

namespace kaldi {
namespace aslp_online {
/// @addtogroup  onlinedecoding OnlineDecoding
/// @{

// This configuration class contains the configuration classes needed to create
// the class MultiUtteranceNnetDecoder.  The actual command line program
// requires other configs that it creates separately, and which are not included
// here: namely, OnlineFeaturePipelineConfig and OnlineEndpointConfig.
struct OnlineNnetDecodingConfig {
    LatticeFasterDecoderConfig decoder_opts;
    aslp_nnet::NnetDecodableOptions decodable_opts;

    OnlineNnetDecodingConfig() {  decodable_opts.acoustic_scale = 0.1; }

    void Register(OptionsItf *po) {
        decoder_opts.Register(po);
        decodable_opts.Register(po);
    }
};

/**
   You will instantiate this class when you want to decode a single
   utterance using the online-decoding setup for neural nets.
*/
class MultiUtteranceNnetDecoder {
public:
    // Constructor.  The feature_pipeline_ pointer is not owned in this
    // class, it's owned externally.
    MultiUtteranceNnetDecoder(const OnlineNnetDecodingConfig &config,
            const TransitionModel &tmodel,
            aslp_nnet::Nnet *model,
            const CuVector<BaseFloat> &log_prior,
            const fst::Fst<fst::StdArc> &fst,
            OnlineFeatureInterface *feat_interface);

    ~MultiUtteranceNnetDecoder() { }
    /// advance the decoding as far as we can.
    void AdvanceDecoding();

    /// Finalizes the decoding. Cleans up and prunes remaining tokens, so the
    /// GetLattice() call will return faster.  You must not call this before
    /// calling (TerminateDecoding() or InputIsFinished()) and then Wait().
    void FinalizeDecoding();

    int32 NumFramesDecoded() const;

    /// Gets the lattice.  The output lattice has any acoustic scaling in it
    /// (which will typically be desirable in an online-decoding context); if you
    /// want an un-scaled lattice, scale it using ScaleLattice() with the inverse
    /// of the acoustic weight.  "end_of_utterance" will be true if you want the
    /// final-probs to be included.
    void GetLattice(bool end_of_utterance,
            CompactLattice *clat) const;

    /// Outputs an FST corresponding to the single best path through the current
    /// lattice. If "use_final_probs" is true AND we reached the final-state of
    /// the graph then it will include those as final-probs, else it will treat
    /// all final-probs as one.
    void GetBestPath(bool end_of_utterance,
            Lattice *best_path) const;


    /// This function calls EndpointDetected from online-endpoint.h,
    /// with the required arguments.
    bool EndpointDetected(const OnlineEndpointConfig &config);

    /// This function resets the data member feature_interface_ and calls
    /// decoder_.InitDecoding().
    /// The feature_pipeline_ should be freed out of this class if necessary.
    void ResetDecoder(OnlineFeatureInterface *new_feat_interface) {
        KALDI_ASSERT(new_feat_interface != NULL);
        feature_interface_ = new_feat_interface;
        decodable_.ResetFeature(feature_interface_);
        decoder_.InitDecoding();
    }

    /// This function get the partial result by calling the function GetBestPath.
    inline void GetPartialResult(const fst::SymbolTable *word_syms,
            std::string *result) {
        decoder_.GetBestPath(&lat_, false);
        GetLinearSymbolSequence(lat_, static_cast<std::vector<int32> *>(0), 
                &words_,
                static_cast<LatticeWeight *>(0));
        aslp_online::WordsToString(words_, word_syms, "", result);
    }

private:

    OnlineNnetDecodingConfig config_;

    OnlineFeatureInterface *feature_interface_;

    const TransitionModel &tmodel_;

    aslp_nnet::NnetDecodableOnline decodable_;

    LatticeFasterOnlineDecoder decoder_;

    // only used by GetPartialResult()
    Lattice lat_;
    std::vector<int32> words_;

};

  
/// @} End of "addtogroup onlinedecoding"

}  // namespace aslp_online
}  // namespace kaldi



#endif  // ASLP_ONLINE_ONLINE_NNET_DECODING_H_
