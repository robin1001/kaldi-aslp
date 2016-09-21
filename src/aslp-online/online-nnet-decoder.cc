// aslp-online/online-nnet-decoder.cc

// Copyright    2013-2014  Johns Hopkins University (author: Daniel Povey)
// Copyright    2015-2016  ASLP (zhangbinbin hechangqing)

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

#include "lat/lattice-functions.h"
#include "lat/determinize-lattice-pruned.h"

#include "aslp-online/online-nnet-decoder.h"
#include "aslp-online/online-helper.h"

namespace kaldi {
namespace aslp_online {

MultiUtteranceNnetDecoder::MultiUtteranceNnetDecoder(
        const OnlineNnetDecodingConfig &config,
        const TransitionModel &tmodel,
        aslp_nnet::Nnet *model,
        const CuVector<BaseFloat> &log_prior,
        const fst::Fst<fst::StdArc> &fst,
        OnlineFeatureInterface *feature_interface):
    config_(config),
    feature_interface_(feature_interface),
    tmodel_(tmodel),
    decodable_(model, log_prior, tmodel, config.decodable_opts, feature_interface_),
    decoder_(fst, config.decoder_opts) {
        decoder_.InitDecoding();
}

void MultiUtteranceNnetDecoder::AdvanceDecoding() {
    decoder_.AdvanceDecoding(&decodable_);
}

void MultiUtteranceNnetDecoder::FinalizeDecoding() {
    decoder_.FinalizeDecoding();
}

int32 MultiUtteranceNnetDecoder::NumFramesDecoded() const {
    return decoder_.NumFramesDecoded();
}

void MultiUtteranceNnetDecoder::GetLattice(bool end_of_utterance,
        CompactLattice *clat) const {
    if (NumFramesDecoded() == 0)
        KALDI_ERR << "You cannot get a lattice if you decoded no frames.";
    Lattice raw_lat;
    decoder_.GetRawLattice(&raw_lat, end_of_utterance);

    if (!config_.decoder_opts.determinize_lattice)
        KALDI_ERR << "--determinize-lattice=false option is not supported at the moment";

    BaseFloat lat_beam = config_.decoder_opts.lattice_beam;
    DeterminizeLatticePhonePrunedWrapper(
            tmodel_, &raw_lat, lat_beam, clat, config_.decoder_opts.det_opts);
}

void MultiUtteranceNnetDecoder::GetBestPath(bool end_of_utterance,
        Lattice *best_path) const {
    decoder_.GetBestPath(best_path, end_of_utterance);
}

bool MultiUtteranceNnetDecoder::EndpointDetected(
        const OnlineEndpointConfig &config) {
    KALDI_ERR << "Not implemented";
    //return aslp_online::EndpointDetected(config, tmodel_,
    //        feature_interface_->FrameShiftInSeconds(),
    //        decoder_);  
    return false;
}

//void MultiUtteranceNnetDecoder::GetPartialResult(const fst::SymbolTable *word_syms, 
//                      std::string *result) {
//  decoder_.GetBestPath(&lat_, false);
//  GetLinearSymbolSequence(lat_, static_cast<std::vector<int32> *>(0), 
//                          &words_,
//                          static_cast<LatticeWeight *>(0));
//  aslp_online::WordsToString(words_, word_syms, " ", result);
//}

}  // namespace aslp_online
}  // namespace kaldi

