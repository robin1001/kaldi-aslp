// aslp-online/online-nnet-decoding.h

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


#ifndef ASLP_ONLINE_ONLINE_NNET_DECODING_H_
#define ASLP_ONLINE_ONLINE_NNET_DECODING_H_

#include <string>
#include <vector>
#include <deque>

#include "matrix/matrix-lib.h"
#include "util/common-utils.h"
#include "base/kaldi-error.h"
#include "nnet2/online-nnet2-decodable.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/online-endpoint.h"
#include "decoder/lattice-faster-online-decoder.h"
#include "hmm/transition-model.h"
#include "hmm/posterior.h"

#include "aslp-online/online-helper.h"

namespace aslp_online {
/// @addtogroup  onlinedecoding OnlineDecoding
/// @{

using namespace kaldi;



// This configuration class contains the configuration classes needed to create
// the class MultiUtteranceNnet2Decoder.  The actual command line program
// requires other configs that it creates separately, and which are not included
// here: namely, OnlineNnet2FeaturePipelineConfig and OnlineEndpointConfig.
struct OnlineNnet2DecodingConfig {
  
  LatticeFasterDecoderConfig decoder_opts;
  nnet2::DecodableNnet2OnlineOptions decodable_opts;
  
  OnlineNnet2DecodingConfig() {  decodable_opts.acoustic_scale = 0.1; }
  
  void Register(OptionsItf *po) {
    decoder_opts.Register(po);
    decodable_opts.Register(po);
  }
};

/**
   You will instantiate this class when you want to decode a single
   utterance using the online-decoding setup for neural nets.
*/
class MultiUtteranceNnet2Decoder {
 public:
  // Constructor.  The feature_pipeline_ pointer is not owned in this
  // class, it's owned externally.
  MultiUtteranceNnet2Decoder(const OnlineNnet2DecodingConfig &config,
                              const TransitionModel &tmodel,
                              const nnet2::AmNnet &model,
                              const fst::Fst<fst::StdArc> &fst,
                              OnlineNnet2FeaturePipeline *feature_pipeline);
  
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

  /// This function resets the data member feature_pipeline_ and calls
  /// decoder_.InitDecoding().
  /// The feature_pipeline_ should be freed out of this class if necessary.
  void ResetDecoder(OnlineNnet2FeaturePipeline *new_feat_pipeline) {
    feature_pipeline_ = new_feat_pipeline;
    decoder_.InitDecoding();
  }

  /// This function get the partial result by calling the function GetBestPath.
  inline void GetPartialResult(const fst::SymbolTable *word_syms,
                        std::string *result) {
    decoder_.GetBestPath(&lat_, false);
    GetLinearSymbolSequence(lat_, static_cast<std::vector<int32> *>(0), 
                            &words_,
                            static_cast<LatticeWeight *>(0));
    aslp_online::WordsToString(words_, word_syms, " ", result);
  }

  ~MultiUtteranceNnet2Decoder() { }
 private:

  OnlineNnet2DecodingConfig config_;

  OnlineNnet2FeaturePipeline *feature_pipeline_;

  const TransitionModel &tmodel_;
  
  nnet2::DecodableNnet2Online decodable_;
  
  LatticeFasterOnlineDecoder decoder_;

  // only used by GetPartialResult()
  Lattice lat_;
  std::vector<int32> words_;

};

  
/// @} End of "addtogroup onlinedecoding"

}  // namespace aslp_online



#endif  // ASLP_ONLINE_ONLINE_NNET_DECODING_H_
