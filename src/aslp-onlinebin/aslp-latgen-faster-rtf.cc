// aslp-onlinebin/aslp-latgen-faster-rtf.cc

/* Created on 2016-06-16
 * Author: Binbin Zhang
 */

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/decoder-wrappers.h"
#include "decoder/decodable-matrix.h"
#include "base/timer.h"

#include "aslp-nnet/nnet-nnet.h"
#include "aslp-nnet/nnet-pdf-prior.h"
#include "aslp-nnet/nnet-decodable.h"

int main(int argc, char *argv[]) {
    try {
        using namespace kaldi;
        using namespace aslp_nnet;
        typedef kaldi::int32 int32;
        using fst::SymbolTable;
        using fst::VectorFst;
        using fst::StdArc;

        const char *usage =
            "Decode with feature input and generate lattice and ouput rtf info\n"
            "Usage: aslp-latgen-faster-rtf [options] nnet_in trans-model-in fst-in feature-rspecifier"
            " lattice-wspecifier [ words-wspecifier [alignments-wspecifier] ]\n";
        ParseOptions po(usage);
        bool allow_partial = false;
        BaseFloat acoustic_scale = 0.1;
        LatticeFasterDecoderConfig config;

        std::string word_syms_filename;
        config.Register(&po);

        PdfPriorOptions prior_config;
        prior_config.Register(&po);

        NnetDecodableOptions nnet_decoding_config;
        nnet_decoding_config.Register(&po);

        po.Register("word-symbol-table", &word_syms_filename, 
                    "Symbol table for words [for debug output]");
        po.Register("allow-partial", &allow_partial, 
                    "If true, produce output even if end state was not reached.");
        double frames_per_second = 100;
        po.Register("frames-per-second", &frames_per_second, 
                    "for calcuate RTF, one second wav for frames-per-second feat");

        po.Read(argc, argv);

        if (po.NumArgs() < 5 || po.NumArgs() > 7) {
            po.PrintUsage();
            exit(1);
        }

        std::string nnet_rxfilename = po.GetArg(1),
            model_in_filename = po.GetArg(2),
            fst_in_str = po.GetArg(3),
            feature_rspecifier = po.GetArg(4),
            lattice_wspecifier = po.GetArg(5),
            words_wspecifier = po.GetOptArg(6),
            alignment_wspecifier = po.GetOptArg(7);

        // Read decode fst file
        VectorFst<StdArc> *decode_fst = fst::ReadFstKaldi(fst_in_str);
        LatticeFasterDecoder decoder(*decode_fst, config);
        // Prior file for pdf prior
        KALDI_LOG << "Read prior file " << prior_config.class_frame_counts;
        if (prior_config.class_frame_counts == "") {
            KALDI_ERR << "class_frame_counts: prior file must be provided";
        }
        PdfPrior pdf_prior(prior_config); 
        const CuVector<BaseFloat> &log_prior = pdf_prior.LogPrior();
        // Nnet model for acoustic model
        Nnet nnet;
        {
            bool binary;
            Input ki(nnet_rxfilename, &binary);
            nnet.Read(ki.Stream(), binary);
        }
        // Read transition model
        TransitionModel trans_model;
        ReadKaldiObject(model_in_filename, &trans_model);

        bool determinize = config.determinize_lattice;
        CompactLatticeWriter compact_lattice_writer;
        LatticeWriter lattice_writer;
        if (! (determinize ? compact_lattice_writer.Open(lattice_wspecifier)
                    : lattice_writer.Open(lattice_wspecifier)))
            KALDI_ERR << "Could not open table for writing lattices: "
                << lattice_wspecifier;

        Int32VectorWriter words_writer(words_wspecifier);

        Int32VectorWriter alignment_writer(alignment_wspecifier);

        fst::SymbolTable *word_syms = NULL;
        if (word_syms_filename != "") 
            if (!(word_syms = fst::SymbolTable::ReadText(word_syms_filename)))
                KALDI_ERR << "Could not read symbol table from file "
                    << word_syms_filename;

        double tot_like = 0.0;
        kaldi::int64 frame_count = 0;
        int num_success = 0, num_fail = 0;
        double total_wav_time = 0, total_decode_time = 0;

        SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
        for (; !feature_reader.Done(); feature_reader.Next()) {
            std::string utt = feature_reader.Key();
            Matrix<BaseFloat> feat(feature_reader.Value());

            Timer timer;
            NnetDecodable decodable(&nnet, log_prior, trans_model, nnet_decoding_config, feat);

            double like;
            if (DecodeUtteranceLatticeFaster(
                        decoder, decodable, trans_model, word_syms, utt,
                        acoustic_scale, determinize, allow_partial, &alignment_writer,
                        &words_writer, &compact_lattice_writer, &lattice_writer,
                        &like)) {
                tot_like += like;
                frame_count += feat.NumRows();
                num_success++;
                // For calcuate RTF
                double decode_time = timer.Elapsed();
                double wav_time = feat.NumRows() / frames_per_second; 
                KALDI_LOG << utt << " RTF " << decode_time / wav_time;
                total_decode_time += decode_time;
                total_wav_time += wav_time;
            } else {
                num_fail++;
            }
        }

        delete decode_fst; // delete this only after decoder goes out of scope.
        
        KALDI_LOG << "TOTAL RTF " << total_decode_time / total_wav_time;
        KALDI_LOG << "Done " << num_success << " utterances, failed for "
            << num_fail;
        KALDI_LOG << "Overall log-likelihood per frame is " << (tot_like/frame_count) << " over "
            << frame_count<<" frames.";

        delete word_syms;
        if (num_success != 0) return 0;
        else return 1;
    } catch (const std::exception &e) {
        std::cerr << e.what();
        return -1;
    }
}
