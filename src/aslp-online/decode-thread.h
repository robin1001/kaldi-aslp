// aslp-online/decode-thread.h

/* Created on 2016-05-30
 * Author: Binbin Zhang
 */

#ifndef ASLP_ONLINE_DECODE_THREAD
#define ASLP_ONLINE_DECODE_THREAD

#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"

#include "aslp-online/online-helper.h"
#include "aslp-online/online-nnet-decoder.h"
#include "aslp-online/wav-provider.h"
#include "aslp-online/tcp-server.h"
#include "aslp-online/vad.h"
#include "aslp-online/punctuation-processor.h"
#include "aslp-online/thread-pool.h"


namespace kaldi {
namespace aslp_online {

// The origin code is contributed by hechangqing, which was implemented in a 
// thread, here we refactor it in thread pool for better abstraction,
// better performance and more readable.
// Here Nnet object is allocated as the resource object, namely allocated
// enough at begin 

class DecodeThread : public Threadable {
public:
    DecodeThread(int client_socket, 
                 int chunk_length,
                 BaseFloat samp_freq,
                 bool do_endpointing,
                 const OnlineFeaturePipelineConfig &feature_info,
                 const OnlineNnetDecodingConfig &nnet_decoding_config,
                 const OnlineEndpointConfig &endpoint_config,
                 const VadOptions &vad_config,
                 const TransitionModel &trans_model,
                 const CuVector<BaseFloat> &log_prior,
                 const fst::Fst<fst::StdArc> &decode_fst,
                 const PunctuationProcessor &punctuation_processor,
                 const fst::SymbolTable *word_syms_table):
            client_socket_(client_socket),
            chunk_length_(chunk_length), 
            samp_freq_(samp_freq), 
            do_endpointing_(do_endpointing), 
            feature_info_(feature_info),
            nnet_decoding_config_(nnet_decoding_config), 
            endpoint_config_(endpoint_config), 
            vad_config_(vad_config), 
            trans_model_(trans_model), 
            log_prior_(log_prior),
            decode_fst_(decode_fst), 
            punctuation_processor_(punctuation_processor),
            word_syms_table_(word_syms_table) {
    }
    // Here resource is a pointer to a Nnet ojbect
    virtual void operator() (void *resource);
private:
    int client_socket_;
    int chunk_length_;
    BaseFloat samp_freq_;
    bool do_endpointing_;
    const OnlineFeaturePipelineConfig &feature_info_;
    const OnlineNnetDecodingConfig &nnet_decoding_config_;
    const OnlineEndpointConfig &endpoint_config_;
    const VadOptions &vad_config_;
    const TransitionModel &trans_model_;
    const CuVector<BaseFloat> &log_prior_;
    const fst::Fst<fst::StdArc> &decode_fst_;
    const PunctuationProcessor &punctuation_processor_;
    const fst::SymbolTable *word_syms_table_;
};

} // namespace aslp_online
} // namespace kaldi


#endif

