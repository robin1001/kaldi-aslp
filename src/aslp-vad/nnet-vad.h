/* Created on 2016-04-27
 * Author: Binbin Zhang
 */

#ifndef ASLP_VAD_NNET_VAD_H_
#define ASLP_VAD_NNET_VAD_H_

#include "base/kaldi-common.h"
#include "itf/options-itf.h"
#include "feat/feature-functions.h"

#include "aslp-nnet/nnet-nnet.h"
#include "aslp-vad/vad.h"

namespace kaldi {

struct NnetVadOptions : public VadOptions {
    float sil_thresh; // 
    NnetVadOptions(): sil_thresh(0.5) {}
    void Register(OptionsItf *opts) {
        VadOptions::Register(opts);
        opts->Register("sil-thresh", &sil_thresh, "if like > sil_thresh, it is classifyed to sil");
    }
};

class NnetVad : public Vad {
public:
    typedef aslp_nnet::Nnet Nnet;
    NnetVad(const Nnet &nnet, const NnetVadOptions &nnet_vad_config): 
            Vad(nnet_vad_config),
            nnet_(nnet), 
            nnet_vad_config_(nnet_vad_config) {} 

    virtual bool IsSilence(int frame) const; 
    void GetScore(const Matrix<BaseFloat> &feat); 
    // DoVad input:wav, feat already prepared out:wav
    // return value: true have vad result
    //               false: no vad result
    bool DoVad(const VectorBase<BaseFloat> &raw_wav, 
               const Matrix<BaseFloat> &raw_feat,
               Vector<BaseFloat> *vad_wav);
    // DoVad input:wav, feat already prepared out:feat
    bool DoVad(const Matrix<BaseFloat> &raw_feat,
               Matrix<BaseFloat> *vad_feat);
protected:
    std::vector<float> sil_score_;
    const Nnet &nnet_;
    const NnetVadOptions &nnet_vad_config_;
};

} // namespace kaldi

#endif
