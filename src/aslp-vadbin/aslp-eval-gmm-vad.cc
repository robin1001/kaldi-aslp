// aslp-vadbin/aslp-eval-gmm-vad.cc

// Copyright 2016 ASLP (Binbin Zhang) 
// Created on 2016-04-25

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "gmm/model-common.h"
#include "gmm/full-gmm.h"
#include "gmm/diag-gmm.h"
#include "gmm/mle-full-gmm.h"


int main(int argc, char *argv[]) {
    try {
        using namespace kaldi;

        const char *usage =
            "Eval vad gmm model\n"
            "Usage:  aslp-eval-gmm-vad [options] <sil-model-in> <voice-model> <feature-rspecifier> <ali-rspecifier>\n"
            "e.g.: aslp-eval-gmm-vad sil.model voice.mdl scp:test.scp ark:test.ark\n";

        ParseOptions po(usage);

        po.Read(argc, argv);

        if (po.NumArgs() != 4) {
            po.PrintUsage();
            exit(1);
        }

        std::string sil_model_filename = po.GetArg(1),
            voice_model_filename = po.GetArg(2),
            feature_rspecifier = po.GetArg(3),
            alignments_rspecifier = po.GetArg(4);

        DiagGmm sil_gmm;
        {
            bool binary_read;
            Input ki(sil_model_filename, &binary_read);
            sil_gmm.Read(ki.Stream(), binary_read);
        }

        DiagGmm voice_gmm;
        {
            bool binary_read;
            Input ki(voice_model_filename, &binary_read);
            voice_gmm.Read(ki.Stream(), binary_read);
        }

        double tot_frames = 0.0;
        int total_sil_frames = 0, total_voice_frames = 0;
        int corr_sil_frames = 0, corr_voice_frames = 0;

        SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
        RandomAccessInt32VectorReader ali_reader(alignments_rspecifier);
        int32 num_done = 0, num_err = 0;

        for (; !feature_reader.Done(); feature_reader.Next()) {
            std::string key = feature_reader.Key();
            if (!ali_reader.HasKey(key)) {
                KALDI_WARN << "file " << key << " do not have aliment";
                num_err++;
                continue;
            }
            const Matrix<BaseFloat> &mat = feature_reader.Value();
            const std::vector<int32> &ali = ali_reader.Value(key);

            int32 num_frames = mat.NumRows();
            Vector<BaseFloat> sil_likes(num_frames), voice_likes(num_frames);
            for (int32 i = 0; i < num_frames; i++) {
                if (ali[i] == 0) total_sil_frames++;
                else total_voice_frames++;
                sil_likes(i) = sil_gmm.LogLikelihood(mat.Row(i));
                voice_likes(i) = voice_gmm.LogLikelihood(mat.Row(i));
                if (ali[i] == 0 && sil_likes(i) > voice_likes(i)) corr_sil_frames++;
                if (ali[i] == 1 && voice_likes(i) > sil_likes(i)) corr_voice_frames++;
            }

            tot_frames += num_frames;
            num_done++;
        }
        KALDI_LOG << "Done " << num_done << " files; " << num_err
            << " with errors.";
        KALDI_LOG << "Total frame " << tot_frames;
        KALDI_LOG << "sil frame " << total_sil_frames << " correct " << corr_sil_frames;
        KALDI_LOG << "voice frame " << total_voice_frames << " correct " << corr_voice_frames;
        KALDI_LOG << "ACCURACY << " << (corr_sil_frames + corr_voice_frames) / tot_frames;
        return (num_done != 0 ? 0 : 1);

    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}

