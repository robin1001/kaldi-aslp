// aslpnnetbin/aslp-nnet-generate-graph.cc

// Copyright 2016  ASLP (Author: liwenpeng zhangbinbin)

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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "aslp-nnet/nnet-nnet.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace kaldi::aslp_nnet;
    typedef kaldi::int32 int32;

    const char *usage =
        "Generate dot file about the neural network.\n"
        "Usage:  aslp-nnet-generate-graph [options] <nnet-in> <dot-out>\n"
        "e.g.:\n"
        " aslp-nnet-info 1.nnet 1.dot\n";
    
    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string nnet_rxfilename = po.GetArg(1);
	std::string dot_wxfilename = po.GetArg(2);
    // load the network
    Nnet nnet; 
    {
      bool binary_read;
      Input ki(nnet_rxfilename, &binary_read);
      nnet.Read(ki.Stream(), binary_read);
	  std::ofstream ko(dot_wxfilename.c_str());
	  nnet.WriteDotFile(ko);
    }

    KALDI_LOG << "Generate dot file for " << nnet_rxfilename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}


