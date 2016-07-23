// aslp-vad/feature-spectrum.cc

/* Created on 2016-07-12
 * Author: Zhang Binbin
 */

#include "aslp-vad/feature-spectrum.h"


namespace kaldi {

typedef int32 Int32;

static float CalInterSpectrumFlatnessFloat(float* spectrum_power,
                                           Int32 spec_dim) {
    assert(((spec_dim - 1)&spec_dim) == 0);//spec_dim == 2^n
    
    float mean = 0, var = 0;
    for (Int32 i = 0; i < spec_dim; ++i) {
        float t = spectrum_power[i];
        mean += t;
        var += t*t;
    }
    mean /= spec_dim;
    var = var/spec_dim - mean*mean;
    return log2(var + 1E-6);
}

static float CalLongTermSpectrumFlatnessFloat(float* spectrum_powers, //2d array [num_frame * spec_dim]
                                              Int32 num_frame,
                                              Int32 spec_dim) {
    float tot_flatness = 0;
    if (num_frame == 1) return 0;
    for (Int32 s = 0; s < spec_dim; ++s) {
        float mean = 0, var = 0;
        for (Int32 f = 0; f < num_frame; ++f) {
            float t = spectrum_powers[f*spec_dim + s];
            mean += t;
            var += t*t;
        }
        mean /= num_frame;
        var = var/num_frame - mean*mean;
        if (var < 0) var = 0;
        float cur_flatness = log2(var + 1E-6);
        tot_flatness += cur_flatness;
    }
    return tot_flatness/spec_dim;
}

static float CalPeriodicityFloat(float *spectrum_power,
                                 Int32 spec_dim) {
    assert(spec_dim == 256);
    const Int32 low_freq = 2, high_freq = 16;
    const Int32 num_freq_copies = spec_dim / 16;
    float max = 0;
    for (Int32 i = low_freq; i < high_freq; ++i) {
        float sum = 0;
        for (Int32 j = 0; j < num_freq_copies; ++j) {
            sum += log2(spectrum_power[j*num_freq_copies + i] + 1);
        }
        if (max < sum) max = sum;
    }
    return max/num_freq_copies;
}

static void CalHarmonicityAndClarityFloat(float* frame_wave, // apply hamming windows, 1kHz sampling rate
                                          float* hamming_window,
                                          Int32 frame_len,
                                          float* harmonicity,
                                          float* clarity) {
    float corr_max = -1E7, corr_min = 1E7;
    float sum_numerator = 0, sum_denominator = 0;
    for (Int32 i = 0; i < frame_len; ++i) {
        sum_numerator += frame_wave[i]*frame_wave[i];
        sum_denominator += 1 + hamming_window[i]*hamming_window[i];
    }
    float corr_0 = sum_numerator/sum_denominator;


    for (Int32 offset = 2; offset < frame_len/2; ++offset) {
        sum_numerator = sum_denominator = 0;
        for (Int32 idx = 0; idx < frame_len - offset; ++idx) {
            sum_numerator += frame_wave[idx]*frame_wave[offset + idx];
            sum_denominator += 1 + hamming_window[idx]*hamming_window[offset + idx];
        }
        float corr = sum_numerator/sum_denominator;
        if (corr_max < corr) {
            corr_max = corr;
        }
        if (corr_min > corr) {
            corr_min = corr;
        }
    }

    assert(corr_max >= corr_min);
    float tmp = fabs((corr_0 - corr_max)/(corr_0 - corr_min + 1));
    *clarity = 1.0 - sqrt(tmp);

    *harmonicity = log2(corr_max + 1);
}

SpectrumFeat::SpectrumFeat(const SpectrumFeatOptions &opts): opts_(opts),
    feature_window_function_(opts.frame_opts), srfft_(NULL) {

    int32 padded_window_size = opts.frame_opts.PaddedWindowSize();
    if ((padded_window_size & (padded_window_size-1)) == 0)  // Is a power of two...
        srfft_ = new SplitRadixRealFft<BaseFloat>(padded_window_size);
}

void SpectrumFeat::Compute(const VectorBase<BaseFloat> &wave, 
                      Matrix<BaseFloat> *output) {
    KALDI_ASSERT(output != NULL);
    // Get dimensions of output features
    int32 rows_out = NumFrames(wave.Dim(), opts_.frame_opts);
    int32 cols_out = 10;
    if (rows_out == 0) {
        output->Resize(0, 0);
        return;
    }
    // Prepare the output buffer
    output->Resize(rows_out, cols_out);

    // Buffers
    Vector<BaseFloat> window;  // windowed waveform.
    std::vector<BaseFloat> temp_buffer;  // used by srfft.  
    float *long_term_spectrum_buffer = NULL;
    int num_frames_lookback = opts_.spectrum_lookback_frames;
    // Compute all the freames, r is frame index..
    for (int32 r = 0; r < rows_out; r++) {
        // Cut the window, apply window function
        ExtractWindow(wave, r, opts_.frame_opts, feature_window_function_, &window);
        // Calc harmonicity And clarity 
        float harmonicity = 0.0, clarity = 0.0;
        CalHarmonicityAndClarityFloat(window.Data(), feature_window_function_.window.Data(),
            window.Dim(), &harmonicity, &clarity);
        if (srfft_ != NULL)  // Compute FFT using split-radix algorithm.
            srfft_->Compute(window.Data(), true, &temp_buffer);
        else  // An alternative algorithm that works for non-powers-of-two.
            RealFft(&window, true);

        //// Convert the FFT into a power spectrum.
        ComputePowerSpectrum(&window);
        SubVector<BaseFloat> power_spectrum(window, 0, window.Dim()/2 + 1);
        int spectrum_dim = power_spectrum.Dim() - 1;
        float inter_spectrum_flatness = CalInterSpectrumFlatnessFloat(
                power_spectrum.Data(), spectrum_dim);
        float periodicity = CalPeriodicityFloat(power_spectrum.Data(), 
                spectrum_dim);
        if (long_term_spectrum_buffer == NULL) {
            long_term_spectrum_buffer = 
                new float[spectrum_dim * num_frames_lookback];
        }

        memcpy(long_term_spectrum_buffer + (r % num_frames_lookback) * spectrum_dim, 
                power_spectrum.Data(), spectrum_dim * sizeof(float));
        float long_term_spectrum_flatness = CalLongTermSpectrumFlatnessFloat(
                long_term_spectrum_buffer, 
                r < num_frames_lookback ? (r+1) : num_frames_lookback,
                spectrum_dim);
        
        (*output)(r, 0) = harmonicity;
        (*output)(r, 1) = clarity;
        (*output)(r, 2) = inter_spectrum_flatness;
        (*output)(r, 3) = periodicity;
        (*output)(r, 4) = long_term_spectrum_flatness;

        KALDI_ASSERT(spectrum_dim == 256);
        // use 5 sepctrum segment(0~16 16~32 32~64 64~128 128~256) feature
        for (int i = 0; i < spectrum_dim; i++) {
            if (i < 16) (*output)(r, 5) += power_spectrum(i);
            else if (16 <= i && i < 32) (*output)(r, 6) += power_spectrum(i);
            else if (32 <= i && i < 64) (*output)(r, 7) += power_spectrum(i);
            else if (64 <= i && i < 128) (*output)(r, 8) += power_spectrum(i);
            else (*output)(r, 9) += power_spectrum(i);
        }
    }

    if (long_term_spectrum_buffer != NULL) delete [] long_term_spectrum_buffer;
}

} // namespace kaldi

