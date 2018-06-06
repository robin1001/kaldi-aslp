// Copyright 2009-2011  Microsoft Corporation
//                2013  Johns Hopkins University (author: Daniel Povey)
//                2018  Binbin Zhang

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <sstream>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/hmm-topology.h"
#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "hmm/tree-accu.h" // for ReadPhoneMap
#include "util/simple-io-funcs.h"

void ReadPhoneMap(std::string phone_map_rxfilename,
                  unordered_map<std::string, int> *phone_map) {
  using namespace kaldi;
  phone_map->clear();
  // phone map file has format e.g.:
  // <eps>
  // sil 1
  // aa 2
  // ...
  // #11 N
  std::ifstream is(phone_map_rxfilename.c_str());
  if (is.fail()) {
      KALDI_ERR << "read file " << phone_map_rxfilename << " error, check!!!";
  }
  std::string line;
  while (std::getline(is, line)) {
    if (line[0] == '<') continue; // <eps>
    if (line[0] == '#') continue; // disambig phones
    std::istringstream ss(line);
    std::string phone;
    int phone_id;
    ss >> phone >> phone_id;
    (*phone_map)[phone] = phone_id;
  }
  if (phone_map->empty()) {
    KALDI_ERR << "Read empty phone map from " << phone_map_rxfilename;
  }
}

void ReadKeywordLexicon(std::string lexicon_rxfilename,
                        std::vector<std::vector<std::string> > *lexicon) {
  using namespace kaldi;
  lexicon->clear();
  std::ifstream is(lexicon_rxfilename.c_str());
  if (is.fail()) {
      KALDI_ERR << "read file " << lexicon_rxfilename << " error, check!!!";
  }
  std::string line;
  while (std::getline(is, line)) {
    std::vector<std::string> str_arr;
    SplitStringToVector(line, " ", true, &str_arr);
    lexicon->push_back(str_arr);
  }
}

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Gen keyword state mapping file\n"
        "Usage:  aslp-kws-gen-state-map [options] phone.txt keyword.lexicon tree_file mdl_file transition_id_mapping_output_file state_list_output_file\n"
        "e.g.: \n"
        " aslp-kws-gen-state-map\n";

    ParseOptions po(usage);

    std::string silence = "sil";
    po.Register("silence", &silence, "silence phone, such as SIL sil");

    po.Read(argc, argv);

    if (po.NumArgs() != 6) {
      po.PrintUsage();
      exit(1);
    }

    std::string phone_txt_rxfilename = po.GetArg(1);
    std::string keyword_lexicon_rxfilename = po.GetArg(2);
    std::string trans_model_filename = po.GetArg(3);
    std::string tree_filename = po.GetArg(4);
    std::string transition_id_mapping_filename = po.GetArg(5);
    std::string state_list_filename = po.GetArg(6);

    unordered_map<std::string, int> phone_mapping;
    ReadPhoneMap(phone_txt_rxfilename, &phone_mapping);
    std::vector<std::vector<std::string> > lexicon;
    ReadKeywordLexicon(keyword_lexicon_rxfilename, &lexicon);

    TransitionModel trans_model;
    ReadKaldiObject(trans_model_filename, &trans_model);
    ContextDependency ctx_dep;  // the tree.
    ReadKaldiObject(tree_filename, &ctx_dep);
    int32 N = ctx_dep.ContextWidth(),
          P = ctx_dep.CentralPosition();
    KALDI_ASSERT(N == 3); 
    KALDI_ASSERT(P == 1);

    std::map<int,int> pdf_mapping; 
    std::map<std::string, int> keyword_state_mapping;
    // 0 for filler state
    keyword_state_mapping["<gbg>"] = 0;
    // 1 for silence state
    keyword_state_mapping["sil"] = 1;
    KALDI_ASSERT(phone_mapping.find(silence) != phone_mapping.end());
    {
      int silence_id = phone_mapping[silence];
      std::vector<int32> phone_window(N, 0);
      phone_window[P] = silence_id;
      int32 num_pdf_classes = trans_model.GetTopo().NumPdfClasses(silence_id);
      int32 pdf = 0;
      for (int32 pdf_class = 0; pdf_class < num_pdf_classes; pdf_class++) {
        if (ctx_dep.Compute(phone_window, pdf_class, &pdf)) {
          pdf_mapping[pdf] = 1;
        } else {
          KALDI_ERR << "tree did not succeed in converting phone window " << silence << " " << pdf_class;
        }
      }
    }

    for (size_t i = 0; i < lexicon.size(); i++) {
      std::string word = lexicon[i][0];
      std::cout << word;
      KALDI_ASSERT(lexicon[i].size() > 3);  // at least 2 phones

      for (size_t j = 1; j < lexicon[i].size(); j++) {
        //std::cout << " " << lexicon[i][j];
        std::string cur_phone = lexicon[i][j];
        KALDI_ASSERT(phone_mapping.find(cur_phone) != phone_mapping.end());
        int32 cur_phone_id = phone_mapping[cur_phone];
        std::vector<int32> phone_window(N, 0);
        phone_window[P] = cur_phone_id;
        std::string prev_phone, next_phone;
        int32 prev_phone_id = 0, next_phone_id = 0;
        std::string context; 
        if (j - 1 > 0) {
          prev_phone = lexicon[i][j-1];
          KALDI_ASSERT(phone_mapping.find(prev_phone) != phone_mapping.end());
          prev_phone_id = phone_mapping[prev_phone];
          phone_window[0] = prev_phone_id;
          context += prev_phone;
        } else {
          int silence_id = phone_mapping[silence];
          phone_window[0] = silence_id;
          context += "sil";
        }
        context += "_";
        context += cur_phone;
        context += "_";
        if (j + 1 < lexicon[i].size()) {
          next_phone = lexicon[i][j+1];
          KALDI_ASSERT(phone_mapping.find(next_phone) != phone_mapping.end());
          next_phone_id = phone_mapping[next_phone];
          phone_window[2] = next_phone_id;
          context += next_phone;
        } else {
          int silence_id = phone_mapping[silence];
          phone_window[2] = silence_id;
          context += "sil";
        }
        int32 num_pdf_classes = trans_model.GetTopo().NumPdfClasses(cur_phone_id);
        for (int32 pdf_class = 0; pdf_class < num_pdf_classes; pdf_class++) {
          std::stringstream ss;
          ss << context << "_s" << pdf_class;
          std::string cd_state = ss.str();
          int32 pdf;
          if (ctx_dep.Compute(phone_window, pdf_class, &pdf)) {
            if (keyword_state_mapping.find(ss.str()) == keyword_state_mapping.end()) {
              int32 cd_state_id = keyword_state_mapping.size();
              keyword_state_mapping[cd_state] = cd_state_id;
              pdf_mapping[pdf] = cd_state_id;
            }
          } else {
            KALDI_ERR << "tree did not succeed in converting phone window " << cd_state;
          }
          // std::cout << " " << cd_state << "_" << keyword_state_mapping[cd_state];
          std::cout << " " << cd_state;
        }
      }
      std::cout << "\n";
    }
    
    // Write transition id mapping file
    {
      std::ofstream os(transition_id_mapping_filename.c_str());
      if (os.fail()) {
          KALDI_ERR << "write file " << transition_id_mapping_filename << " error, check!!!";
      }
      for (int32 i = 1; i <= trans_model.NumTransitionIds(); i++) {
        int32 pdf = trans_model.TransitionIdToPdf(i);
        if (pdf_mapping.find(pdf) == pdf_mapping.end()) {
          pdf_mapping[pdf] = 0;
        }
        // std::cout << i << " " << pdf << " " << pdf_mapping[pdf] << "\n";
        os << i << " " << pdf_mapping[pdf] << "\n";
      }
    }
    
    // Write state list file
    {
      std::ofstream os(state_list_filename.c_str());
      if (os.fail()) {
          KALDI_ERR << "write file " << state_list_filename << " error, check!!!";
      }
      std::vector<std::string> state_list(keyword_state_mapping.size()); 
      for (std::map<std::string, int>::iterator it = keyword_state_mapping.begin();
           it != keyword_state_mapping.end(); it++) {
        state_list[it->second] = it->first;
      }
      os << "<eps>" << " " << 0 << "\n";
      for (int i = 0; i < state_list.size(); i++) {
        // std::cout << state_list[i] << " " << i << "\n";
        os << state_list[i] << " " << i+1 << "\n";
      }
    }

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

