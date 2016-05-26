/* Created on 2016-01-25
   Author: xukaituo zhangbinbin
*/
#include "punctuation-processor.h"

namespace aslp_online {

void PunctuationProcessor::ConvertToInput(const std::string &raw_input, std::string *input) const {
    // convert utf-8 sentence to single character, then convert to crf input format
    std::string ch;
    for (size_t i = 0, len = 0; i != raw_input.length(); i += len) {
        unsigned char byte = (unsigned)raw_input[i];
        if (byte >= 0xFC) // lenght 6
            len = 6;
        else if (byte >= 0xF8)
            len = 5;
        else if (byte >= 0xF0)
            len = 4;
        else if (byte >= 0xE0)
            len = 3;
        else if (byte >= 0xC0)
            len = 2;
        else
            len = 1;
        ch = raw_input.substr(i, len);
        (*input) += ch;
        (*input) += "\tN\n";

    }
}
void PunctuationProcessor::ConvertToOutput(const char *output, std::string *raw_output) const {
    // convert crf output format to utf-8 character, then to a sentence with punctuation
    std::string ch;
    std::string raw_input = output;
    for (size_t i = 0, len = 0; i < raw_input.length(); i += len+5) {
        unsigned char byte = (unsigned)raw_input[i];
        if (byte >= 0xFC) // lenght 6
            len = 6;
        else if (byte >= 0xF8)
            len = 5;
        else if (byte >= 0xF0)
            len = 4;
        else if (byte >= 0xE0)
            len = 3;
        else if (byte >= 0xC0)
            len = 2;
        else
            len = 1;
        ch = raw_input.substr(i, len);
        (*raw_output) += ch;
        switch(raw_input[i + len + 3]) {
            case 'N':break;
            case 'D':(*raw_output) += "，";break;
            case 'J':(*raw_output) += "。";break;
            case 'G':(*raw_output) += "！";break;
            case 'W':(*raw_output) += "？";break;
        }
    }
}

}
