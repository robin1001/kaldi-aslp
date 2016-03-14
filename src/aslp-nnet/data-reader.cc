// aslp-nnet/data-reader.cc

// Copyright 2016 ASLP (Author: zhangbinbin)

// Created on 2016-03-09

#include "aslp-nnet/data-reader.h"

namespace kaldi {
namespace aslp_nnet {

FrameDataReader::FrameDataReader(
                    const std::vector<std::string> &feature_rspecifiers,
                    const std::vector<std::string> &targets_rspecifiers,
                    const NnetDataRandomizerOptions &rand_opts): 
        num_input_(feature_rspecifiers.size()), 
        num_output_(targets_rspecifiers.size()),
        rand_opts_(rand_opts), read_done_(false) {
    feature_readers_.resize(num_input_);
    feature_randomizers_.resize(num_input_);
    for (int i = 0; i < num_input_; i++) {
        feature_readers_[i] = new SequentialBaseFloatMatrixReader(feature_rspecifiers[i]);
        feature_randomizers_[i] = new MatrixRandomizer(rand_opts_);
    }
    targets_readers_.resize(num_output_);
    targets_randomizers_.resize(num_output_);
    for (int i = 0; i < num_output_; i++) {
        targets_readers_[i] = new RandomAccessPosteriorReader(targets_rspecifiers[i]);
        targets_randomizers_[i] = new PosteriorRandomizer(rand_opts_);
    }
    randomizer_mask_.Init(rand_opts_);
}

FrameDataReader::FrameDataReader(const std::string &feature_rspecifier, 
                                 const std::string &targets_rspecifier,
                                 const NnetDataRandomizerOptions &rand_opts): rand_opts_(rand_opts) {
    std::vector<std::string> feature_rspecifiers;
    feature_rspecifiers.push_back(feature_rspecifier);
    std::vector<std::string> targets_rspecifiers;
    targets_rspecifiers.push_back(targets_rspecifier);
    new (this) FrameDataReader(feature_rspecifiers, targets_rspecifiers, rand_opts);
}

FrameDataReader::~FrameDataReader() {
    for (int i = 0; i < feature_readers_.size(); i++) {
        delete feature_readers_[i];
    }
    for (int i = 0; i < feature_randomizers_.size(); i++) {
        delete feature_randomizers_[i];
    }
    for (int i = 0; i < targets_readers_.size(); i++) {
        delete targets_readers_[i];
    }
    for (int i = 0; i < targets_randomizers_.size(); i++) {
        delete targets_randomizers_[i];
    }
}

inline bool FrameDataReader::Done() {
    KALDI_ASSERT(feature_randomizers_.size() > 0);
    return (read_done_ && feature_randomizers_[0]->Done());
}

void FrameDataReader::FillRandomizer() {
    KALDI_ASSERT(feature_readers_.size() > 0);
    KALDI_ASSERT(targets_readers_.size() > 0);
    //for (; !feature_readers_[0].Done(); feature_readers_[0].Next()) {
    while (true) {
        if (feature_randomizers_[0]->IsFull()) break;
        if (feature_readers_[0]->Done()) {
            for (int i = 1; i < feature_readers_.size(); i++)
                KALDI_ASSERT(feature_readers_[i]->Done());
            read_done_ = true;
            break;
        }
        std::string utt = feature_readers_[0]->Key();
        KALDI_VLOG(3) << "Reading " << utt;
        // Check all key of feature must be equal
        for (int i = 1; i < feature_readers_.size(); i++) {
            if (utt != feature_readers_[i]->Key()) {
                KALDI_ERR << "all feature not in the same order"
                          << "[0] " << utt
                          << "[" << i << "] " << feature_readers_[i]->Key();
            }
        }
        bool all_have_target = true;
        for (int i = 0; i < targets_readers_.size(); i++) {
            if (!targets_readers_[i]->HasKey(utt)) {
                KALDI_WARN << utt << ", missing targets";
                all_have_target = false;
            }
        }
        // Add to randomizer, check dim
        if (all_have_target) {
            int num_frame = 0;
            for (int i = 0; i < feature_readers_.size(); i++) {
                Matrix<BaseFloat> mat = feature_readers_[i]->Value();
                if (0 == i) num_frame = mat.NumRows();
                else if (mat.NumRows() != num_frame) {
                    KALDI_ERR << "all feature dim not equal";
                }
                feature_randomizers_[i]->AddData(CuMatrix<BaseFloat>(mat));
            }
            for (int i = 0; i < targets_readers_.size(); i++) {
                Posterior targets = targets_readers_[i]->Value(utt);
                if (targets.size() != num_frame) {
                    KALDI_ERR << "feature and target dim must match";
                }
                targets_randomizers_[i]->AddData(targets);
            }
        }
        // Add Iter
        for (int i = 0; i < feature_readers_.size(); i++) {
            feature_readers_[i]->Next();
        }
    }
    // Randomize
    const std::vector<int32>& mask = randomizer_mask_.Generate(feature_randomizers_[0]->NumFrames());
    for (int i = 0; i < feature_randomizers_.size(); i++) {
        feature_randomizers_[i]->Randomize(mask);
    }
    for (int i = 0; i < targets_randomizers_.size(); i++) {
        targets_randomizers_[i]->Randomize(mask);
    }
}

void FrameDataReader::ReadData(std::vector<const CuMatrixBase<BaseFloat> *> *input, 
                               std::vector<const Posterior *> *output) {
    KALDI_ASSERT(input != NULL);
    KALDI_ASSERT(output != NULL);
    input->resize(num_input_);
    output->resize(num_output_);
    if (Done()) {
        KALDI_ERR << "Already read done";
    }
    // Used up? fill it
    if (feature_randomizers_[0]->Done()) {
        FillRandomizer();
    }
    for (int i = 0; i < num_input_; i++) {
        const CuMatrixBase<BaseFloat> &mat = feature_randomizers_[i]->Value();
        (*input)[i] = &mat;
        feature_randomizers_[i]->Next();
    }
    for (int i = 0; i < num_output_; i++) {
        const Posterior &tgt = targets_randomizers_[i]->Value();
        (*output)[i] = &tgt;
        targets_randomizers_[i]->Next();
    }
}


void FrameDataReader::ReadData(const CuMatrixBase<BaseFloat> **feat, const Posterior **targets) {
    KALDI_ASSERT(feat != NULL);
    KALDI_ASSERT(targets != NULL);
    KALDI_ASSERT(num_input_ == 1);
    KALDI_ASSERT(num_output_ == 1);
    if (Done()) {
        KALDI_ERR << "Already read done";
    }
    // Used up? fill it
    if (feature_randomizers_[0]->Done()) {
        FillRandomizer();
    }
    const CuMatrixBase<BaseFloat> &mat = feature_randomizers_[0]->Value();
    *feat = &mat;
    feature_randomizers_[0]->Next();
    const Posterior &tgt = targets_randomizers_[0]->Value();
    *targets = &tgt;
    targets_randomizers_[0]->Next();
}



} // namespace aslp_nnet
} // namespace kaldi

