/* Created on 2016-04-27
 * Author: Binbin Zhang
 */

#ifndef ASLP_VAD_ENERGY_VAD_H_
#define ASLP_VAD_ENERGY_VAD_H_

#include "base/kaldi-common.h"
#include "itf/options-itf.h"
#include "matrix/matrix-common.h"

#include "aslp-vad/vad.h"

namespace kaldi {

// TODO energy vad options
struct EnergyVadOptions {
    EnergyVadOptions() {}
    void Register(OptionsItf *opts) {
    }
};

class EnergyVad: public Vad {
    EnergyVad(const EnergyVadOptions &energy_vad_config, 
              const VadOptions &vad_config): Vad(vad_config), 
        energy_vad_config_(energy_vad_config) {}

    void DoVad(const VectorBase<BaseFloat> &raw_wav, 
                       Vector<BaseFloat> *vad_wav);
    virtual inline bool IsSilence(int frame) const; 
private:
    const EnergyVadOptions &energy_vad_config_;
};

} // namespace kaldi

#endif
