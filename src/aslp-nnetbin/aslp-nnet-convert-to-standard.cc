// aslp-nnetbin/aslp-nnet-convert-to-standard.cc

/* Created on 2016-07-04
 * Author: zhangbinbin
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "aslp-nnet/nnet-nnet.h"

int main(int argc, char *argv[]) {
    try {
        using namespace kaldi;
        using namespace kaldi::aslp_nnet;
        typedef kaldi::int32 int32;

        const char *usage =
            "Convert aslp nnet to standard kaldi nnet(simple feedforward nnet)\n"
            " so the id, input id, and offset info will not be wrriten\n"
            "Usage:  aslp-nnet-convert-to-standard [options] <nnet-in> <nnet-out>\n"
            "e.g.:\n"
            " aslp-nnet-convert-to-standard --binary=false nnet.in nnet.out\n";

        SetVerboseLevel(1); // be verbose by default

        ParseOptions po(usage);
        bool binary_write = true;
        po.Register("binary", &binary_write, "Write output in binary mode");

        po.Read(argc, argv);

        if (po.NumArgs() != 2) {
            po.PrintUsage();
            exit(1);
        }

        std::string nnet_in_filename = po.GetArg(1),
            nnet_out_filename = po.GetArg(2);

        // initialize the network
        aslp_nnet::Nnet nnet;
        {
            bool binary_read;
            Input ki(nnet_in_filename, &binary_read);
            nnet.Read(ki.Stream(), binary_read);
        }

        // store the network
        Output ko(nnet_out_filename, binary_write);
        nnet.WriteStandard(ko.Stream(), binary_write);

        KALDI_LOG << "Written model to " << nnet_out_filename;
        return 0;
    } catch(const std::exception &e) {
        std::cerr << e.what() << '\n';
        return -1;
    }
}



