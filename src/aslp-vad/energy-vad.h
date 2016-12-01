/* Created on 2016-04-27
 * Author: Binbin Zhang
 */

#ifndef ASLP_VAD_ENERGY_VAD_H_
#define ASLP_VAD_ENERGY_VAD_H_

#include "base/kaldi-common.h"
#include "itf/options-itf.h"
#include "matrix/matrix-common.h"
#include "matrix/kaldi-vector.h"

#include "aslp-vad/vad.h"

namespace kaldi {

// TODO energy vad options
struct EnergyVadOptions : public VadOptions {
    float sil_thresh; //
    EnergyVadOptions(): sil_thresh(0.9992) {}
    void Register(OptionsItf *opts) {
        VadOptions::Register(opts);
        opts->Register("sil-thresh", &sil_thresh, "if score > sil_thresh, it is classified to sil");
    }
};

class EnergyVad: public Vad {
public:
    EnergyVad(const EnergyVadOptions &vad_config): Vad(vad_config), 
        energy_vad_config_(vad_config),
        raw_wav_max_value_(1e7) {}

    bool DoVad(const VectorBase<BaseFloat> &raw_wav, 
                       Vector<BaseFloat> *vad_wav);

    virtual bool IsSilence(int frame) const; 

    std::vector<BaseFloat> GetScore(const VectorBase<BaseFloat> &raw_wav);
    void CalculateEnergy(const VectorBase<BaseFloat> &raw_wav);
    void CalculateScore();

private:
    BaseFloat FrameEnergy(const VectorBase<BaseFloat> &raw_wav, int32 frame) const;

    const EnergyVadOptions &energy_vad_config_;
    BaseFloat raw_wav_max_value_;
    std::vector<BaseFloat> energy_vec_;
    std::vector<BaseFloat> sil_scores_;
};

} // namespace kaldi

#endif
