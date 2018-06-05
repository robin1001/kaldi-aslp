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

bool FrameDataReader::Done() {
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


bool FrameDataReader::ReadData(const CuMatrixBase<BaseFloat> **feat, const Posterior **targets) {
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
    // Even read, but still can not fill a batch size
    if (!Done()) {
        const CuMatrixBase<BaseFloat> &mat = feature_randomizers_[0]->Value();
        *feat = &mat;
        feature_randomizers_[0]->Next();
        const Posterior &tgt = targets_randomizers_[0]->Value();
        *targets = &tgt;
        targets_randomizers_[0]->Next();
        return true;
    }
    return false;
}


SequenceDataReader::SequenceDataReader(
							const std::string &feature_rspecifier,
							const std::string &targets_rspecifier,
							const SequenceDataReaderOptions &read_opts):read_opts_(read_opts), read_done_(false){
	feature_reader_ = new SequentialBaseFloatMatrixReader(feature_rspecifier);
	target_reader_ = new RandomAccessPosteriorReader(targets_rspecifier);
	curt_.resize(read_opts_.num_stream, 0);
	lent_.resize(read_opts_.num_stream, 0);
	new_utt_flags_.resize(read_opts_.num_stream, 0);
	keys_.resize(read_opts_.num_stream);
	feats_.resize(read_opts_.num_stream);
	targets_.resize(read_opts_.num_stream);

}

SequenceDataReader::~SequenceDataReader(){}

inline bool SequenceDataReader::Done() {
	return (read_done_ && feature_reader_->Done());
}

void SequenceDataReader::AddNewUtt() {
	
	int32 num_stream = read_opts_.num_stream;
	// loop over all streams, check if any stream reaches the end of its utterance
	// if any, feed the exhhausted steam with a new utterance, update book-keeping infos
	for (int s = 0; s < num_stream; s++) {
		// this stream still has valid frames
		if (curt_[s] < lent_[s]) {
			new_utt_flags_[s] = 0;
			continue;
		}
		// else, this stream exhausted, need new utterance
		while (!feature_reader_->Done()) {
			const std::string& key = feature_reader_->Key();
			// get the feature matrix,
			const Matrix<BaseFloat> &mat = feature_reader_->Value();
			// dorp too long sentence	
			int32 drop_len = read_opts_.drop_len;
			if (drop_len > 0 && mat.NumRows() > drop_len) {
				KALDI_WARN << key << ", too long, droped";
				feature_reader_->Next();
				continue;
			}
			// get the labels,
			if (!target_reader_->HasKey(key)) {
				KALDI_WARN << key << ", missing targets";
				feature_reader_->Next();
				continue;
			}

			const Posterior& target = target_reader_->Value(key);

			// check that the length matches,
			if (mat.NumRows() != target.size()) {
				KALDI_WARN << key << ", length miss-match between feats and targers, skip";
				feature_reader_->Next();
				continue;
			}

			// Use skip
			int32 skip_width =  read_opts_.skip_width;
			CuMatrix<BaseFloat> cu_mat(mat.NumRows(), mat.NumCols());
			cu_mat.CopyFromMat(mat);
			if (skip_width > 1) {
				int skip_len = (cu_mat.NumRows() - 1) / skip_width + 1;
				CuMatrix<BaseFloat> skip_feat(skip_len, cu_mat.NumCols());
				Posterior skip_target(skip_len);
				for (int i = 0; i < skip_len; i++) {
					skip_feat.Row(i).CopyFromVec(cu_mat.Row(i * skip_width));
					skip_target[i] = target[i * skip_width];
				}
				feats_[s].Resize(skip_feat.NumRows(), skip_feat.NumCols());
				skip_feat.CopyToMat(&feats_[s]);
				targets_[s] = skip_target;
			} else {
				feats_[s].Resize(cu_mat.NumRows(), cu_mat.NumCols());
				cu_mat.CopyToMat(&feats_[s]);
				targets_[s] = target;
			}
			// checks ok, put the data in the buffers,
			keys_[s] = key;
			curt_[s] = 0;
			lent_[s] = feats_[s].NumRows();
			new_utt_flags_[s] = 1; // a new utterance feeded to this stream
			feature_reader_->Next();
			break;
		}
	}
}

void SequenceDataReader::FillBatchBuff(CuMatrix<BaseFloat> *feat,
									   Posterior *target, 
									   Vector<BaseFloat> *frame_mask) {
	KALDI_ASSERT(feat != NULL);
	KALDI_ASSERT(target != NULL);
	KALDI_ASSERT(frame_mask != NULL);
	
	int32 num_stream = read_opts_.num_stream;
	int32 batch_size = read_opts_.batch_size;
	int32 targets_delay = read_opts_.targets_delay;
	// we are done if all stream are exhausted
	for (int s = 0; s < num_stream; s++) {
		if (curt_[s] < lent_[s]) {
			read_done_ = false; // this stream still contains vaild data, not exhausted
			break;
		}
		else {
			read_done_ = true;
		}
	}
		
	// fill a multi-stream bptt batch
	// * frame_mask: o indicates padded frames, 1 indicates valid frames
	// * target: padded to batch_size
	// *feat: first shifted to achieve targets delay; then padded to batch_size
	int32 feat_dim = feats_[0].NumCols();
	Matrix<BaseFloat> feats(batch_size * num_stream, feat_dim, kSetZero);
	Posterior &targets = *target;
	targets.resize(batch_size * num_stream);
	Vector<BaseFloat>& mask = *frame_mask;
	mask.Resize(batch_size * num_stream, kSetZero);

	if (!read_done_) {
		for ( int t = 0; t < batch_size; t++) {
			for (int s = 0; s < num_stream; s++) {
				if (curt_[s] < lent_[s]) {
					mask(t * num_stream + s) = 1;
					targets[t * num_stream + s] = targets_[s][curt_[s]];
				} else {
					mask(t * num_stream + s) = 0;
					targets[t * num_stream + s] = targets_[s][lent_[s]-1];
				}
				// feat shifting & padding
				if (curt_[s] + targets_delay < lent_[s]) {
					feats.Row(t * num_stream + s).CopyFromVec(feats_[s].Row(curt_[s]+targets_delay));
				} else {
					feats.Row(t * num_stream + s).CopyFromVec(feats_[s].Row(lent_[s]-1));
				}
				curt_[s]++;
			}
		}
	feat->Resize(batch_size * num_stream, feat_dim, kSetZero);
	feat->CopyFromMat(feats);
	}
}

void SequenceDataReader::ReadData(CuMatrix<BaseFloat> *feat,
								  Posterior *target,
								  Vector<BaseFloat> *frame_mask) {
	KALDI_ASSERT(feat != NULL);
	KALDI_ASSERT(target != NULL);
	KALDI_ASSERT(frame_mask != NULL);

	if (Done())
		KALDI_ERR << "Already read done!";
	else {
		AddNewUtt(); // add new utterance to multi-streams
		// fill batch buffer for bptt
		FillBatchBuff(feat, target, frame_mask);
	}
}

} // namespace aslp_nnet
} // namespace kaldi

