// aslp-vadbin/aslp-eval-energy-vad.cc

// Copyright 2016 ASLP 
// Created on 2016-11-30

#include "base/kaldi-common.h"
#include "util/common-utils.h"

#include "aslp-vad/energy-vad.h"
#include "aslp-vad/roc.h"

int main(int argc, char *argv[]) {
    try {
        using namespace kaldi;

        const char *usage =
            "Eval energy vad \n"
            "Usage:  aslp-eval-energy-vad [options] <raw-wav-rspecifier> <ali-rspecifier>\n"
            "e.g.: aslp-eval-energy-vad scp:raw_wav.scp ark:test.ark\n";

        ParseOptions po(usage);
        EnergyVadOptions vad_opts;
        vad_opts.Register(&po);
        RocSetOptions roc_opts;
        roc_opts.Register(&po);

        po.Read(argc, argv);

        if (po.NumArgs() != 2) {
            po.PrintUsage();
            exit(1);
        }

        std::string raw_wav_rspecifier = po.GetArg(1),
            alignments_rspecifier = po.GetArg(2);

        double tot_frames = 0.0;

        EnergyVad vad(vad_opts);
        
        SequentialTableReader<WaveHolder> raw_wav_reader(raw_wav_rspecifier);
        RandomAccessInt32VectorReader ali_reader(alignments_rspecifier);
        
        int32 num_done = 0, num_err = 0;
        
        RocSet roc_set(roc_opts);

        for (; !raw_wav_reader.Done(); raw_wav_reader.Next()) {
            std::string key = raw_wav_reader.Key();
            KALDI_VLOG(2) << "Processing " << key;
            if (!ali_reader.HasKey(key)) {
                KALDI_WARN << "file " << key << " do not have aliment";
                num_err++;
                continue;
            }
            const WaveData &wave_data = raw_wav_reader.Value();
            const std::vector<int32> &ali = ali_reader.Value(key);

            SubVector<BaseFloat> raw_wav(wave_data.Data(), 0);
            std::vector<BaseFloat> sil_scores = vad.GetScore(raw_wav);

            int32 num_frames = ali.size();
            for (int i = 0; i < num_frames; i++) {
                roc_set.AddData(sil_scores, ali[i]);
            }

            tot_frames += num_frames;
            num_done++;
        }
        KALDI_LOG << "Done " << num_done << " files; " << num_err
            << " with errors.";
        roc_set.Report();
        return (num_done != 0 ? 0 : 1);

    } catch(const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}

