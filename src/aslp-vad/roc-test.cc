/* Created on 2016-04-26
 * Author: Binbin Zhang
 */
#include <stdio.h>

#include "aslp-vad/roc.h"


void TestRoc() {
    using namespace kaldi;
    std::vector<int> labels = {0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 
                               1, 0, 1, 1, 1, 0, 1, 0, 1};
    std::vector<float> scores = {0.9, 0.8, 0.7, 0.6, 0.55, 0.54, 0.53, 0.52, 
                                 0.51, 0.5, 0.4, 0.39, 0.38, 0.37, 0.36, 0.35, 
                                 0.34, 0.33, 0.3, 0.1};
     RocSetOptions option;
     option.stride = 0.05;
     RocSet roc_set(option);
     for (int i = 0; i < labels.size(); i++) {
        roc_set.AddData(scores[i], labels[i]);
     }
     roc_set.Report();
}

int main() {
    TestRoc();
    return 0;
}


