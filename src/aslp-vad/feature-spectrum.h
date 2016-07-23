// aslp-vad/feature-spectrum.h

/* Created on 2016-07-12
 * Author: Zhang Binbin
 */

#ifndef ASLP_VAD_FEATURE_SPECTRUM_H_
#define ASLP_VAD_FEATURE_SPECTRUM_H_

#include "feat/feature-functions.h"


namespace kaldi {

struct SpectrumFeatOptions {
    FrameExtractionOptions frame_opts;
    int spectrum_lookback_frames;
    SpectrumFeatOptions(): spectrum_lookback_frames(4) {}

    void Register(OptionsItf *opts) {
        frame_opts.Register(opts);
        opts->Register("spectrum-lookback-frames", &spectrum_lookback_frames,
                       "number of lookbadck frames for long term spectrum flatness calculation");
    }
};

class SpectrumFeat {
public:
    SpectrumFeat(const SpectrumFeatOptions &opts);
    void Compute(const VectorBase<BaseFloat> &wave,
                 Matrix<BaseFloat> *output);
protected:
    SpectrumFeatOptions opts_;
    FeatureWindowFunction feature_window_function_;
    SplitRadixRealFft<BaseFloat> *srfft_;
    KALDI_DISALLOW_COPY_AND_ASSIGN(SpectrumFeat);
};

} // namespace kaldi

#endif


