#ifndef ASLP_ONLINE_ONLINE_HELPER_H
#define ASLP_ONLINE_ONLINE_HELPER_H

#include <string>
#include <sstream>
#include <vector>

#include "fstext/fstext-lib.h"

namespace aslp_online {

inline void WordsToString(const std::vector<kaldi::int32> words,
                   const fst::SymbolTable *word_syms,
                   const std::string delim,
                   std::string *result) {
  std::stringstream ss;
  if (word_syms != NULL) {
    for (size_t i = 0; i < words.size(); i++) {
      std::string s = word_syms->Find(words[i]);
      if (s == "")
        KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
      ss << s << delim;
    }
  }
  if (result != NULL)
    *result = ss.str();
}

} // namespace aslp_online

#endif // ASLP_ONLINE_ONLINE_HELPER_H
