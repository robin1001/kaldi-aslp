// aslp-vadbin/aslp-eval-vad-boundary.cc

// Copyright 2016 ASLP (Binbin Zhang) 
// Created on 2016-07-07

#include "base/kaldi-common.h"
#include "util/common-utils.h"

#include "aslp-vad/boundary-tool.h"

int main(int argc, char *argv[]) {
    try {
        using namespace kaldi;

        const char *usage =
            "Eval vad boundary accuracy\n"
            "Usage:  aslp-eval-vad-boundary [options] <nnet-in> <ali-rspecifier> <ref-ali-rspecifier>\n"
            "e.g.: aslp-eval-nn-vad-boundary ark:label.ali ark:ref.ali\n";

        ParseOptions po(usage);

        int context = 10;
        po.Register("context", &context, "context size for evaluate the boundary accuracy");

        po.Read(argc, argv);

        if (po.NumArgs() != 2) {
            po.PrintUsage();
            exit(1);
        }

        std::string alignments_rspecifier = po.GetArg(1),
            ref_alignments_rspecifier = po.GetArg(2);

        SequentialInt32VectorReader ali_reader(alignments_rspecifier);
        RandomAccessInt32VectorReader ref_ali_reader(ref_alignments_rspecifier);
        int32 num_done = 0, num_err = 0;
        BoundaryTool boundary_tool(context);

        for (; !ali_reader.Done(); ali_reader.Next()) {
            std::string key = ali_reader.Key();
            KALDI_VLOG(2) << "Processing " << key;
            //std::cout << key << " ";
            if (!ref_ali_reader.HasKey(key)) {
                KALDI_WARN << "file " << key << " do not have aliment";
                num_err++;
                continue;
            }
            const std::vector<int32> &ali = ali_reader.Value();
            const std::vector<int32> &ref_ali = ref_ali_reader.Value(key);

            bool is_ok = boundary_tool.AddData(ali, ref_ali);
            if (!is_ok) num_err++;
            num_done++;
        }
        KALDI_LOG << boundary_tool.Report();
        KALDI_LOG << "Done " << num_done << " files; " << num_err
            << " with errors.";
        return (num_done != 0 ? 0 : 1);

    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}

