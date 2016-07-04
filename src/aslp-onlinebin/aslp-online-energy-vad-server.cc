// aslp-onlinebin/aslp-online-energy-vad-server.cc

/* Decoder Server
 * Created on 2015-09-01
 * Refined on 2016-06
 * Author: hechangqing zhangbinbin
 * TODO: ** end point detect
         ** confidence
         ** LM rescoring 
 */

#include "feat/wave-reader.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"

#include "aslp-nnet/nnet-nnet.h"
#include "aslp-nnet/nnet-pdf-prior.h"
#include "aslp-nnet/nnet-decodable.h"

#include "aslp-online/online-helper.h"
#include "aslp-online/online-nnet-decoder.h"
#include "aslp-online/wav-provider.h"
#include "aslp-online/tcp-server.h"
#include "aslp-online/decode-thread.h"
#include "aslp-online/vad.h"
#include "aslp-online/online-endpoint.h"
#include "aslp-online/punctuation-processor.h"

int main(int argc, char *argv[]) {
    try {
        using namespace kaldi;
        using namespace fst;
        using namespace aslp_online;
        using namespace aslp_nnet;

        typedef kaldi::int32 int32;
        typedef kaldi::int64 int64;

        const char *usage =
            "Online wav decoder server with old enery based vad\n"
            "Usage: aslp-online-energy-vad-server [options] <nnet-in> "
            "<trans-model> <fst-in> <punctuation-crf-model>\n";

        ParseOptions po(usage);

        // Register all config
        OnlineEndpointConfig endpoint_config;
        endpoint_config.Register(&po);
        OnlineFeaturePipelineCommandLineConfig feature_cmd_config;
        feature_cmd_config.Register(&po);
        OnlineNnetDecodingConfig nnet_decoding_config;
        nnet_decoding_config.Register(&po);
        aslp_online::VadOptions vad_config;
        vad_config.Register(&po);
        PdfPriorOptions prior_config;
        prior_config.Register(&po);

        BaseFloat chunk_length_secs = 0.1;
        po.Register("chunk-length", &chunk_length_secs,
                "Length of chunk size in seconds, that we process.  Set to <= 0 "
                "to use all input in one chunk.");
        std::string word_syms_rxfilename;
        po.Register("word-symbol-table", &word_syms_rxfilename,
                "Symbol table for words [for debug output]");
        bool do_endpointing = false;
        po.Register("do-endpointing", &do_endpointing,
                "If true, apply endpoint detection");
        int port = 10000;
        po.Register("port", &port, 
                "decoder server port");
        int num_thread = 10;
        po.Register("num-thread", &num_thread,
                "number of thread in the the thread pool");

        po.Read(argc, argv);
        if (po.NumArgs() != 4) {
            po.PrintUsage();
            return 1;
        }

        std::string nnet_rxfilename = po.GetArg(1),
            trans_model_rxfilename = po.GetArg(2),
            fst_rxfilename = po.GetArg(3),
            punc_model_rxfilename = po.GetArg(4);
        
        OnlineFeaturePipelineConfig feature_config(feature_cmd_config);  
        KALDI_LOG << "VAD CONFIGURATION" << vad_config.Print();

        // Start tcp server here, early stop if bind error ocurred
        // for reading fst graph and other input files are time-consuming
        TcpServer tcp_server;
        tcp_server.Listen(port);

        // Prior file for pdf prior
        KALDI_LOG << "Read prior file " << prior_config.class_frame_counts;
        if (prior_config.class_frame_counts == "") {
            KALDI_ERR << "class_frame_counts: prior file must be provided";
        }
        PdfPrior pdf_prior(prior_config); 
        const CuVector<BaseFloat> &log_prior = pdf_prior.LogPrior();
        
        KALDI_LOG << "Reading nnet file " << nnet_rxfilename;
        // Nnet model for acoustic model
        Nnet nnet;
        {
            bool binary;
            Input ki(nnet_rxfilename, &binary);
            nnet.Read(ki.Stream(), binary);
        }
        // Transition model for transition prob
        KALDI_LOG << "Reading transition file " << trans_model_rxfilename;
        TransitionModel trans_model;
        ReadKaldiObject(trans_model_rxfilename, &trans_model);

        // Punctuation file for punctuation predict
        KALDI_LOG << "Reading crf punctuation file " << punc_model_rxfilename;
        PunctuationProcessor punctuation_processor(punc_model_rxfilename.c_str());
        // Fst for decode graph
        fst::SymbolTable *word_syms = NULL;
        if (word_syms_rxfilename != "") {
            if (!(word_syms = fst::SymbolTable::ReadText(word_syms_rxfilename)))
                KALDI_ERR << "Could not read symbol table from file "
                          << word_syms_rxfilename;
        }
        KALDI_LOG << "Reading fst file " << fst_rxfilename;
        fst::Fst<fst::StdArc> *decode_fst = ReadFstKaldi(fst_rxfilename);

        KALDI_LOG << "Read all param files done!!!";

        BaseFloat samp_freq = 16000;
        int32 chunk_length;
        if (chunk_length_secs > 0) {
            chunk_length = int32(samp_freq * chunk_length_secs);
            if (chunk_length == 0) chunk_length = 1;
        } else {
            chunk_length = std::numeric_limits<int32>::max();
        }

        // Nnet pool for thread pool, allocated enough at begin,
        // Avoid dynamic allocating in the running time
        std::vector<void *> nnet_pool(num_thread, NULL);
        for (int i = 0; i < num_thread; i++) {
            Nnet *new_nnet = new Nnet(nnet);
            nnet_pool[i] = static_cast<void *>(new_nnet);
        }

        // Wait ThreadPool destruct then delete nnet in nnet_pool
        {
            ThreadPool thread_pool(num_thread, &nnet_pool);

            while (true) {
                // Wait for new connection
                int32 client_socket = tcp_server.Accept();

                Threadable *task = new DecodeThread(client_socket, chunk_length, 
                                                   samp_freq, do_endpointing,
                                                   feature_config,
                                                   nnet_decoding_config, 
                                                   endpoint_config,
                                                   vad_config, trans_model,
                                                   log_prior, *decode_fst,
                                                   punctuation_processor,
                                                   word_syms);
                // Add in thread pool
                thread_pool.AddTask(task);
            }
        }

        for (int i = 0; i < num_thread; i++) {
            delete static_cast<Nnet *>(nnet_pool[i]); 
        }
        delete decode_fst;
        delete word_syms; // will delete if non-NULL.
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what();
        return -1;
    }

} // main()

