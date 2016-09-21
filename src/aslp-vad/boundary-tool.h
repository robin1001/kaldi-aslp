/* Created on 2016-07-05
 * Author: Binbin Zhang
 */

#ifndef ASLP_VAD_BOUNDARY_TOOL_H_
#define ASLP_VAD_BOUNDARY_TOOL_H_

#include <stdio.h>

#include "base/kaldi-common.h"
#include "itf/options-itf.h"

namespace kaldi {

class BoundaryTool {
public:
    // @param: context, width for eval the boundary accuracy
    BoundaryTool(int context = 10): context_(context), num_sentence_(0), 
        start_acc_(0.0), end_acc_(0.0) { }
    //
    float Weight(int i) {
        // [0, context_]
        if (0 <= i && i < context_) return 1; 
        // [ -context_, 0]
        else if (-context_ <= i && i < 0) return 0; 
        // [-2*context_, context]
        else if (-2 * context_ <= i && i < -context_) return 1.0;
        else KALDI_ERR << "invalid index " << i;
        return 0.0;
    }
    // label: true label for the data, 0 for silence, 1 for speech
    // ref: hypothesis label
    bool AddData(const std::vector<int> &label, 
                 const std::vector<int> &ref) {
        // Here we assume that there is only one speech segment in the
        // label, that is to say the label is start with silence, then 
        // followed by speech segment and end with silence
        // if not, it will not be added.
        KALDI_ASSERT(label.size() == ref.size());
        int start_boundary = 0, end_boundary = label.size() - 1;
        while (label[start_boundary] == 0) start_boundary++;
        if (start_boundary == 0) {
            KALDI_WARN << "Not start with silence, ignore";
            return false;
        }
        while (label[end_boundary] == 0) end_boundary--;
        if (end_boundary == label.size() - 1) {
            KALDI_WARN << "Not end with silence, ignore";
            return false;
        }
        if (start_boundary >= end_boundary) {
            KALDI_WARN << "start boundary greater than end boundary, ignore";
            return false;
        }
        // Eval start boundary 
        int sb_begin = std::max(start_boundary - 2 * context_, 0);
        //int sb_begin = std::max(start_boundary - context_, 0);
        int sb_end = std::min(start_boundary + context_, end_boundary);
        float num_sb_corr = 0, num_sb_all = 0;
        for (int i = sb_begin; i < sb_end; i++) {
            if (label[i] == ref[i]) {
                num_sb_corr += Weight(i - start_boundary);
            }
            num_sb_all += Weight(i - start_boundary);
        }
        KALDI_ASSERT(num_sb_corr <= num_sb_all);
        start_acc_ += num_sb_corr / num_sb_all;
        // Eval end boundary
        int eb_begin = std::max(end_boundary - context_, start_boundary);
        int eb_end = std::min(end_boundary + 2 * context_, (int)label.size());
        //int eb_end = std::min(end_boundary + context_, (int)label.size());
        float num_eb_corr = 0, num_eb_all = 0;
        for (int i = eb_begin; i < eb_end; i++) {
            if (label[i] == ref[i]) {
                num_eb_corr += Weight(end_boundary - i - 1);
            }
            num_eb_all += Weight(end_boundary - i - 1);
        }
        KALDI_ASSERT(num_eb_corr <= num_eb_all);
        end_acc_ += num_eb_corr / num_eb_all;
        //std::cout << " SB " << start_boundary 
        //          << " EB " << end_boundary
        //          << " SBA " << num_sb_corr / num_sb_all
        //          << " EBA " << num_eb_corr / num_eb_all
        //          << "\n";
        num_sentence_++;
        return true;
    }

    std::string Report() {
        std::stringstream ss;
        ss << "Start Boundary Accuracy (SBA) " << start_acc_ / num_sentence_
           << " End Boundary Acc (EBA) " << end_acc_ / num_sentence_;
        return ss.str();
    }

private:
    int context_;
    int num_sentence_;
    float start_acc_, end_acc_;
};

}
#endif
