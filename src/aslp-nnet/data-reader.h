// aslp-nnet/data-reader.h

// Copyright 2016 ASLP (Author: zhangbinbin liwenpeng)

// Created on 2016-03-09

#ifndef ASLP_NNET_DATA_READER_H_
#define ASLP_NNET_DATA_READER_H_

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "base/kaldi-math.h"
#include "aslp-cudamatrix/cu-matrix.h"
#include "aslp-cudamatrix/cu-vector.h"
#include "hmm/posterior.h"
#include "itf/options-itf.h"

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
    bool ReadData(const CuMatrixBase<BaseFloat> **feat, const Posterior **targets); 
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

struct SequenceDataReaderOptions {
    int32 batch_size; // --LSTM-- BPTT batch_size
	int32 num_stream; // --LSTM-- BPTT multistream training
	int32 drop_len;  // If sentence frame length greater than drop_len, then drop it 
    int32 skip_width; // num of frame for one skip
	int32 targets_delay; // --LSTM-- BPTT targets delay
	int32 length_tolerance; // Allowed length difference of features/targets (frames), for the whole utterance training
	double frame_limit; // Max number of frames to be processed for whole utterance training

    SequenceDataReaderOptions(): batch_size(20), num_stream(100), drop_len(0),
								 skip_width(1), targets_delay(5), length_tolerance(5),
								 frame_limit(100000){}
    void Register(OptionsItf *opts) {
		opts->Register("batch-size", &batch_size, "--LSTM-- BPTT batch_size");
		opts->Register("num-stream", &num_stream, "--LSTM-- BPTT multistream training");
        opts->Register("drop-len", &drop_len, "if Sentence frame length greater than drop_len,"
                                              "then drop it, default(0, no drop)");
        opts->Register("skip-width", &skip_width, "num of frame for one skip(default 1, no skip)");
		opts->Register("targets-delay", &targets_delay, "--LSTM-- BPTT targets delay");
		opts->Register("length-tolerance", &length_tolerance, "Allowed length difference of features/targets (frames),"
															  "for the whole utterance training");
		opts->Register("frame-limit", &frame_limit, "Max number of frames to be processed for whole utterance training");
    }
};

class SequenceDataReader {
public:
    SequenceDataReader(const std::string &feature_rspecifier, 
                       const std::string &targets_rspecifier,
                       const SequenceDataReaderOptions &read_opts);
    ~SequenceDataReader();
    void ReadData(CuMatrix<BaseFloat> *feat, Posterior *target, Vector<BaseFloat> *frame_mask);
	bool Done();
	const std::vector<int>& GetNewUttFlags () const {
		return new_utt_flags_;
	}

private:
	void AddNewUtt();
	void FillBatchBuff(CuMatrix<BaseFloat> *feat, Posterior *target, Vector<BaseFloat> *frame_mask);

private:
    SequentialBaseFloatMatrixReader *feature_reader_;
    RandomAccessPosteriorReader *target_reader_;
    const SequenceDataReaderOptions &read_opts_;
	int32 read_done_;
	std::vector<std::string> keys_;
	std::vector<Matrix<BaseFloat> > feats_;
	std::vector<Posterior> targets_;
	std::vector<int> curt_;
	std::vector<int> lent_;
	std::vector<int> new_utt_flags_;
};

} // namespace aslp_nnet
} // namespace kaldi

#endif
