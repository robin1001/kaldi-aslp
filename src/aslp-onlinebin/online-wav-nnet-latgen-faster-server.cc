// aslp-onlinebin/online-wav-nnet-latgen-faster-server.cc

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)

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

/* Decoder Server
 * Created on 2015-09-01
 * Author: hechangqing zhangbinbin
 * TODO: ** support sample rate 8000(now 16000)
         ** end point detect
         ** confidence
         ** LM rescoring 
 */

#include "feat/wave-reader.h"
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"

#include "aslp-online/online-helper.h"
#include "aslp-online/online-nnet-decoding.h"
#include "aslp-online/wav-provider.h"
#include "aslp-online/tcp-server.h"
#include "aslp-online/vad.h"
#include "aslp-online/punctuation-processor.h"

namespace kaldi {

void GetDiagnosticsAndPrintOutput(const std::string &utt,
                                  const fst::SymbolTable *word_syms,
                                  const CompactLattice &clat,
                                  int64 *tot_num_frames,
                                  double *tot_like,
                                  std::string *result) {
  if (clat.NumStates() == 0) {
    KALDI_WARN << "Empty lattice.";
    return;
  }
  CompactLattice best_path_clat;
  CompactLatticeShortestPath(clat, &best_path_clat);
  
  Lattice best_path_lat;
  ConvertLattice(best_path_clat, &best_path_lat);
  
  double likelihood;
  LatticeWeight weight;
  int32 num_frames;
  std::vector<int32> alignment;
  std::vector<int32> words;
  GetLinearSymbolSequence(best_path_lat, &alignment, &words, &weight);
  num_frames = alignment.size();
  likelihood = -(weight.Value1() + weight.Value2());
  *tot_num_frames += num_frames;
  *tot_like += likelihood;
  KALDI_VLOG(2) << "Likelihood per frame for utterance " << utt << " is "
                << (likelihood / num_frames) << " over " << num_frames
                << " frames.";

  aslp_online::WordsToString(words, word_syms, "", result);  
  KALDI_LOG << utt << ' ' << *result;
}

} // namespace kaldi

namespace kaldi {

using namespace aslp_online;

// This class is used by the decoder thread.
// The data members are used by the decoder thread and the data members 
// are read only. Most of the data members are actually shared by all the
// decoder threads in the process.
struct ReadOnlyComponent {
  int client_socket;
  int chunk_length;
  BaseFloat samp_freq;
  bool do_endpointing;
  const OnlineNnet2FeaturePipelineInfo &feature_info;
  const OnlineNnetDecodingConfig &nnet2_decoding_config;
  const OnlineEndpointConfig &endpoint_config;
  const VadOptions &vad_config;
  const TransitionModel &trans_model;
  const nnet2::AmNnet &nnet;
  const fst::Fst<fst::StdArc> &decode_fst;
  const fst::SymbolTable *word_syms;
  const PunctuationProcessor &punctuation_processor;

  ReadOnlyComponent(int socket, int chunk_len, BaseFloat sample_frequence,
      bool do_endpoint, const OnlineNnet2FeaturePipelineInfo &feat_info,
      const OnlineNnetDecodingConfig &decoding_config,
      const OnlineEndpointConfig &endpoint_cfg,
      const VadOptions &vad_cfg,
      const TransitionModel &transition_mdl,
      const nnet2::AmNnet &net,
      const fst::Fst<fst::StdArc> &fst,
      const fst::SymbolTable *word_syms_tab,
      const PunctuationProcessor &punc_processor) 
    : client_socket(socket), chunk_length(chunk_len),
      samp_freq(sample_frequence), do_endpointing(do_endpoint),
      feature_info(feat_info), nnet2_decoding_config(decoding_config),
      endpoint_config(endpoint_cfg), vad_config(vad_cfg), 
      trans_model(transition_mdl), 
      nnet(net), decode_fst(fst), word_syms(word_syms_tab),
      punctuation_processor(punc_processor) {
  }
};

// The thread function to do speech recognition for an apply from a client.
// The parameter ptr_in is actually a pointer to class ReadOnlyComponent.
// The component (transition model, decoding fst, neural network, 
// configuations etc.) are referenced in class ReadOnlyComponent.
// The client socket are also in the class ReadOnlyComponent.
// The function delete the class ReadOnlyComponent before return.
void *RunDecoder(void *ptr_in) {
  using namespace fst;
  using namespace aslp_online;

 try {
  
  ReadOnlyComponent *ptr_readonly =
    reinterpret_cast<ReadOnlyComponent *>(ptr_in);
  
  int client_socket = ptr_readonly->client_socket;
  int chunk_length = ptr_readonly->chunk_length;
  BaseFloat samp_freq = ptr_readonly->samp_freq;
  bool do_endpointing = ptr_readonly->do_endpointing;
  const OnlineNnet2FeaturePipelineInfo &feature_info = 
    ptr_readonly->feature_info;
  const OnlineNnetDecodingConfig &nnet2_decoding_config =
    ptr_readonly->nnet2_decoding_config;
  const OnlineEndpointConfig &endpoint_config =
    ptr_readonly->endpoint_config;
  const VadOptions &vad_config =
    ptr_readonly->vad_config;
  const TransitionModel &trans_model = ptr_readonly->trans_model;
  const nnet2::AmNnet &nnet = ptr_readonly->nnet;
  const fst::Fst<fst::StdArc> &decode_fst = ptr_readonly->decode_fst;
  const fst::SymbolTable *word_syms = ptr_readonly->word_syms;
  const PunctuationProcessor &punctuation_processor = ptr_readonly->punctuation_processor;
 
  pthread_detach(pthread_self());
  
  double tot_like = 0.0;
  int64 num_frames = 0;
  
  // This object receives raw wave data and sends the recognition results
  // to the client.
  // The client_socket is closed by this object.
  WavProvider wav_provider(client_socket);
      
  OnlineIvectorExtractorAdaptationState adaptation_state(
    feature_info.ivector_extractor_info);
  OnlineNnet2FeaturePipeline *feature_pipeline = 
    new OnlineNnet2FeaturePipeline(feature_info);
  MultiUtteranceNnetDecoder decoder(nnet2_decoding_config,
                                      trans_model,
                                      nnet,
                                      decode_fst,
                                      feature_pipeline);
  
  std::vector<std::pair<int32, BaseFloat> > delta_weights;
  std::vector<BaseFloat> data;
  Vad vad(vad_config, &wav_provider);
  double get_partial_result_progress = 0.0;
  std::string all_result;

  while (true) {
    if (vad.Done()) break;
    int num_read = vad.ReadSpeech(chunk_length, &data);
    std::cerr << "vad.ReadSpeech() read " << num_read << std::endl;
    SubVector<BaseFloat> wave_part(data.data(), num_read);
    feature_pipeline->AcceptWaveform(samp_freq, wave_part);
    
    decoder.AdvanceDecoding();

    if (vad.SilenceDetected()) {
      std::cerr << "VAD::SilenceDetected" << std::endl; 
      std::cerr << "VAD::AudioReceived " << vad.AudioReceived() << std::endl;
    }
    // print partial results
    if (decoder.NumFramesDecoded() > 0 && !vad.SilenceDetected() && vad.AudioReceived() - get_partial_result_progress >= 0.7) {
      get_partial_result_progress = vad.AudioReceived();
      std::string result;
      decoder.GetPartialResult(word_syms, &result);
      if (result != "") {
        wav_provider.WritePartialReslut(result);
      }
      KALDI_VLOG(1) << "Partial: " << result;
    } // if NumFramesDecoded > 0

    if (vad.SilenceDetected() && decoder.NumFramesDecoded() > 0) {
      feature_pipeline->InputFinished();
      decoder.AdvanceDecoding();
      
      std::string result;
      decoder.GetPartialResult(word_syms, &result);
      KALDI_VLOG(1) << "-Final: " << result;
      if (result != "") {
        wav_provider.WriteFinalReslut(result);
        all_result += result;
      }

      delete feature_pipeline;
      feature_pipeline = NULL;
      feature_pipeline = new OnlineNnet2FeaturePipeline(feature_info);
      decoder.ResetDecoder(feature_pipeline);
    }
  }
  feature_pipeline->InputFinished();
  decoder.FinalizeDecoding();
  
  std::string recog_result;
  CompactLattice clat;
  if (decoder.NumFramesDecoded() > 0) {
    bool end_of_utterance = true;
    decoder.GetLattice(end_of_utterance, &clat);
    GetDiagnosticsAndPrintOutput("+Final: ", word_syms, clat,
                                 &num_frames, &tot_like, &recog_result);
  } else {
    KALDI_LOG << "no frames decoded";
  }
  if (recog_result != "") {
    wav_provider.WriteFinalReslut(recog_result);
    all_result += recog_result;
  }
  if (all_result.size() > 0) {
    std::string punc_result;
    punctuation_processor.Process(all_result, &punc_result);
    KALDI_LOG << "Final Punctuation Result: " << punc_result;
    wav_provider.WritePuncResult(punc_result);
  }
  wav_provider.WriteEOS();
  
  // In an application you might avoid updating the adaptation state if
  // you felt the utterance had low confidence.  See lat/confidence.h
  // feature_pipeline->GetAdaptationState(&adaptation_state);
  // we want to output the lattice with un-scaled acoustics.
  BaseFloat inv_acoustic_scale =
      1.0 / nnet2_decoding_config.decodable_opts.acoustic_scale;
  ScaleLattice(AcousticLatticeScale(inv_acoustic_scale), &clat);
  
  delete feature_pipeline;
  delete ptr_readonly;
  return static_cast<void *>(NULL);
 } catch (const std::exception &e) {
  std::cerr << e.what();
  return static_cast<void *>(NULL);
 }
}

} // namespace kaldi

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    using namespace aslp_online;

    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    
    const char *usage =
        "Wav Decoder Server\n"
        "Usage: online2-wav-nnet2-latgen-faster-server [options] <nnet2-in> "
                "<fst-in> <punctuation-crf-model> <lattice-wspecifier>\n";
    
    ParseOptions po(usage);
    
    std::string word_syms_rxfilename;
    
    OnlineEndpointConfig endpoint_config;

    // feature_config includes configuration for the iVector adaptation,
    // as well as the basic features.
    OnlineNnet2FeaturePipelineConfig feature_config;  
    OnlineNnetDecodingConfig nnet2_decoding_config;
    
    VadOptions vad_config;

    BaseFloat chunk_length_secs = 0.1;
    bool do_endpointing = false;
    int port = 10000;
    
    po.Register("chunk-length", &chunk_length_secs,
                "Length of chunk size in seconds, that we process.  Set to <= 0 "
                "to use all input in one chunk.");
    po.Register("word-symbol-table", &word_syms_rxfilename,
                "Symbol table for words [for debug output]");
    po.Register("do-endpointing", &do_endpointing,
                "If true, apply endpoint detection");
    po.Register("port", &port,
                "decoder server port");
    
    feature_config.Register(&po);
    nnet2_decoding_config.Register(&po);
    endpoint_config.Register(&po);
    vad_config.Register(&po);
    
    po.Read(argc, argv);
    
    if (po.NumArgs() != 4) {
      po.PrintUsage();
      return 1;
    }
    
    std::string nnet2_rxfilename = po.GetArg(1),
        fst_rxfilename = po.GetArg(2),
        punc_model_rxfilename = po.GetArg(3),
        clat_wspecifier = po.GetArg(4);
    
    TcpServer tcp_server;
    if (!tcp_server.Listen(port)) {
      return 1;
    }
    
    OnlineNnet2FeaturePipelineInfo feature_info(feature_config);
    
    TransitionModel trans_model;
    nnet2::AmNnet nnet;
    {
      bool binary;
      Input ki(nnet2_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      nnet.Read(ki.Stream(), binary);
    }
    
    fst::Fst<fst::StdArc> *decode_fst = ReadFstKaldi(fst_rxfilename);
    
    fst::SymbolTable *word_syms = NULL;
    if (word_syms_rxfilename != "")
      if (!(word_syms = fst::SymbolTable::ReadText(word_syms_rxfilename)))
        KALDI_ERR << "Could not read symbol table from file "
                  << word_syms_rxfilename;
    // Read CRF punctuation predict model 
    PunctuationProcessor punctuation_processor(punc_model_rxfilename.c_str());
    int32 num_done = 0;
    
    CompactLatticeWriter clat_writer(clat_wspecifier);
    BaseFloat samp_freq = 16000;
    int32 chunk_length;
    if (chunk_length_secs > 0) {
      chunk_length = int32(samp_freq * chunk_length_secs);
      if (chunk_length == 0) chunk_length = 1;
    } else {
      chunk_length = std::numeric_limits<int32>::max();
    }   
    
    pthread_t tid;

    KALDI_LOG << "VAD CONFIGURATION" << vad_config.Print();
    while (true) {
      // wait new connection
      int32 client_socket = -1;
      client_socket = tcp_server.Accept();
      
      ReadOnlyComponent *decoding_component = 
        new ReadOnlyComponent(client_socket, chunk_length, samp_freq,
            do_endpointing, feature_info, nnet2_decoding_config,
            endpoint_config, vad_config, trans_model, nnet, *decode_fst, 
            word_syms, punctuation_processor);
      
      pthread_create(&tid, NULL, RunDecoder, decoding_component);

      std::stringstream ss;
      ss << "audio_counter_" << num_done;
      std::string utt = ss.str();
      KALDI_LOG << "Decoded utterance " << utt;
      num_done++;
    }
    
    delete decode_fst;
    delete word_syms; // will delete if non-NULL.
    return 0;
  } catch(const std::exception& e) {
    std::cerr << e.what();
    return -1;
  }
} // main()

