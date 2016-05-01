/* Created on 2016-04-25
 * Author: Binbin Zhang
 */

#ifndef ASLP_VAD_ROC_H_
#define ASLP_VAD_ROC_H_

#include <stdio.h>

#include "base/kaldi-common.h"
#include "itf/options-itf.h"

namespace kaldi {

class Roc {
public:
    Roc(float thresh = 0.0): thresh_(thresh), tp_(0), fp_(0), P_(0), N_(0) { }
    void AddData(float score, int label) {
        if (label == 0) P_++;
        else N_++;
        if (label == 0 && score > thresh_) tp_++;
        if (label == 1 && score > thresh_) fp_++;
    }
    float Thresh() const { return thresh_; }
    std::string Report() {
        KALDI_ASSERT(P_ > 0);
        KALDI_ASSERT(N_ > 0);
        char buffer[1024];
        sprintf(buffer, "Thresh %.6f Accuracy %.6f fp %.6f tp %.6f", 
            thresh_, float(tp_ + (N_ - fp_)) / (P_ + N_), 
            float(fp_) / N_, float(tp_) / P_);
        return std::string(buffer);
    }
private:
    float thresh_;
    int tp_, fp_;
    int P_, N_;
};

struct RocSetOptions {
    float stride;
    RocSetOptions(): stride(0.1) {}
    void Register(OptionsItf *opts) {
        opts->Register("stride", &stride, "stride for roc curve");
    }
};

class RocSet {
public:
    RocSet(const RocSetOptions &config): config_(config) {
        for (float score = 1.0; score >= 0; score -= config_.stride) {
            roc_set_.push_back(new Roc(score));
        }
    } 
    ~RocSet() {
        for (int i = 0; i < roc_set_.size(); i++) {
            delete roc_set_[i];
        }
    }
    void AddData(float score, int label) {
        for (int i = 0; i < roc_set_.size(); i++) {
            KALDI_ASSERT(roc_set_[i] != NULL);
            roc_set_[i]->AddData(score, label);
        }
    }
    void Report() {
        for (int i = 0; i < roc_set_.size(); i++) {
            printf("[ %4d ROC ] %s\n", i, roc_set_[i]->Report().c_str());
        }
    }
private:
    std::vector<Roc *> roc_set_;
    const RocSetOptions &config_;
};

}
#endif
