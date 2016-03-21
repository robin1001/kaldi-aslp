// aslp-nnet/data-reader.h

// Copyright 2016 ASLP (Author: zhangbinbin)

// Created on 2016-03-09

#ifndef ASLP_NNET_DATA_READER_H_
#define ASLP_NNET_DATA_READER_H_

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/kaldi-math.h"
#include "aslp-cudamatrix/cu-matrix.h"
#include "hmm/posterior.h"

#include "aslp-nnet/nnet-trnopts.h"
#include "aslp-nnet/nnet-randomizer.h"

namespace kaldi {
namespace aslp_nnet {

class FrameDataReader {
public:
    FrameDataReader(const std::vector<std::string> &feature_rspecifiers,
                    const std::vector<std::string> &targets_rspecifiers,
                    const NnetDataRandomizerOptions &rand_opts);
    FrameDataReader(const std::string &feature_rspecifier, 
                    const std::string &targets_rspecifier,
                    const NnetDataRandomizerOptions &rand_opts);
    ~FrameDataReader();
    void ReadData(const CuMatrixBase<BaseFloat> **feat, const Posterior **targets); 
    void ReadData(std::vector<const CuMatrixBase<BaseFloat > *> *input, 
                  std::vector<const Posterior *> *output); 
    bool Done();
private:
    void FillRandomizer(); 
    std::vector<SequentialBaseFloatMatrixReader *> feature_readers_;
    std::vector<RandomAccessPosteriorReader *> targets_readers_;
    RandomizerMask randomizer_mask_;
    std::vector<MatrixRandomizer *> feature_randomizers_;
    std::vector<PosteriorRandomizer *> targets_randomizers_;
    int num_input_, num_output_;
    const NnetDataRandomizerOptions &rand_opts_;
    bool read_done_;
};



} // namespace aslp_nnet
} // namespace kaldi

#endif
