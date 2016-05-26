/* Created on 2016-01-25
 *  Author: xukaituo zhangbinbin
 */
#ifndef ASLP_PUNCTUATION_PROCESSOR_H_
#define ASLP_PUNCTUATION_PROCESSOR_H_

#include <string>

#include "crfpp.h"

namespace aslp_online {

class PunctuationProcessor {
public:
    PunctuationProcessor(const char *file_name) {
        char param[1024] = {'\0'};
        sprintf(param, "-m %s", file_name);
        tagger = CRFPP::createTagger(param);
    }
    void Process(const std::string &raw_input, std::string *raw_output) const {
        std::string input;
        ConvertToInput(raw_input, &input);
        const char *output = tagger->parse(input.c_str());
        ConvertToOutput(output, raw_output);
    }
private:
    void ConvertToInput(const std::string &raw_input, std::string *input) const; 
    void ConvertToOutput(const char *output, std::string *raw_output) const; 
    CRFPP::Tagger *tagger;
};

} // namespace aslp_online

#endif
