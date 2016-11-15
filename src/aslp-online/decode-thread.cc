// aslp-online/decode-thread.h

/* Created on 2016-05-30
 * Author: Binbin Zhang
 */
#include "aslp-online/decode-thread.h"

namespace kaldi {
namespace aslp_online {

static void GetDiagnosticsAndPrintOutput(const std::string &utt,
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


void DecodeThread::operator() (void *resource) {
    try {
        aslp_nnet::Nnet *nnet = static_cast<aslp_nnet::Nnet *>(resource);
        double tot_like = 0.0;
        int64 num_frames = 0;
        // This object receives raw wave data and sends the recognition results
        // to the client. The client_socket is closed by this object.
        WavProvider wav_provider(client_socket_);

        OnlineFeaturePipeline *feature_pipeline = 
            new OnlineFeaturePipeline(feature_info_);
        MultiUtteranceNnetDecoder decoder(nnet_decoding_config_,
                trans_model_,
                nnet,
                log_prior_,
                decode_fst_,
                feature_pipeline);

        std::vector<std::pair<int32, BaseFloat> > delta_weights;
        std::vector<BaseFloat> data;
        Vad vad(vad_config_, &wav_provider);
        double get_partial_result_progress = 0.0;
        std::string all_result;

        while (true) {
            if (vad.Done()) break;
            int num_read = vad.ReadSpeech(chunk_length_, &data);
            std::cerr << "vad.ReadSpeech() read " << num_read << std::endl;
            SubVector<BaseFloat> wave_part(data.data(), num_read);
            feature_pipeline->AcceptWaveform(samp_freq_, wave_part);

            decoder.AdvanceDecoding();

            if (vad.SilenceDetected()) {
                std::cerr << "VAD::SilenceDetected" << std::endl; 
                std::cerr << "VAD::AudioReceived " << vad.AudioReceived() << std::endl;
            }
            // print partial results
            if (decoder.NumFramesDecoded() > 0 && 
                    !vad.SilenceDetected() && 
                    vad.AudioReceived() - get_partial_result_progress >= 0.7) {
                get_partial_result_progress = vad.AudioReceived();
                std::string result;
                decoder.GetPartialResult(word_syms_table_, &result);
                if (result != "") {
                    wav_provider.WritePartialReslut(result);
                }
                KALDI_VLOG(1) << "Partial: " << result;
            } // if NumFramesDecoded > 0

            if (vad.SilenceDetected() && decoder.NumFramesDecoded() > 0) {
                feature_pipeline->InputFinished();
                decoder.AdvanceDecoding();

                std::string result;
                decoder.GetPartialResult(word_syms_table_, &result);
                KALDI_VLOG(1) << "-Final: " << result;
                if (result != "") {
                    wav_provider.WriteFinalReslut(result);
                    all_result += result;
                }

                delete feature_pipeline;
                feature_pipeline = NULL;
                feature_pipeline = new OnlineFeaturePipeline(feature_info_);
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
            GetDiagnosticsAndPrintOutput("+Final: ", word_syms_table_, clat,
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
            punctuation_processor_.Process(all_result, &punc_result);
            KALDI_LOG << "All Result: " << all_result;
            KALDI_LOG << "Final Punctuation Result: " << punc_result;
            wav_provider.WritePuncResult(punc_result);
        }
        wav_provider.WriteEOS();

        delete feature_pipeline;
    } catch (const std::exception &e) {
        std::cerr << e.what();
    }

}


static void AddVadFeatureToFeaturePool(int num_frames, 
                OnlineVadFeaturePipeline *vad_pipeline,
                OnlineFeaturePool *feature_pool) {
    KALDI_ASSERT(vad_pipeline != NULL);
    KALDI_ASSERT(feature_pool != NULL);
    Matrix<BaseFloat> vad_feat;
    int num_voice_frames = 
            vad_pipeline->GetVadFeature(num_frames, &vad_feat);
    if (num_voice_frames > 0) {
            feature_pool->AcceptFeature(vad_feat);
    }
    //KALDI_LOG << "Add " << num_voice_frames << " frames to the feature pool";
}

void NnetVadDecodeThread::operator() (void *resource) {
    try {
        NnetVadDecodeThreadResource *nnet_vad_resource = 
            static_cast<NnetVadDecodeThreadResource *>(resource);
        aslp_nnet::Nnet *am_nnet = nnet_vad_resource->am_nnet;
        aslp_nnet::Nnet *vad_nnet = nnet_vad_resource->vad_nnet;
        double tot_like = 0.0;
        int64 num_frames = 0;

        // This object receives raw wave data and sends the recognition results
        // to the client. The client_socket is closed by this object.
        WavProvider wav_provider(client_socket_);
        OnlineVadFeaturePipeline *vad_pipeline = 
            new OnlineVadFeaturePipeline(*vad_nnet, vad_config_, feature_info_);
        OnlineFeaturePool *feature_pool = 
            new OnlineFeaturePool(vad_pipeline->Dim());
        MultiUtteranceNnetDecoder decoder(nnet_decoding_config_,
                trans_model_,
                am_nnet,
                log_prior_,
                decode_fst_,
                feature_pool);

        double get_partial_result_progress = 0.0;
        std::vector<BaseFloat> data;
        std::string all_result;
 
        // Vad on feats then decode speech frames
        /* Here we assume that the feature for vad and the feature 
           for decoder are the same, so the vad out feature is directly used
           by the decoder
           OnlineNnetVad has inner buffers that stores the none silence frames
           when the buffer is full or endpoint detected, it is avaliable in 
           the following code
         */
        while (!wav_provider.Done()) {
            // Read until forward_batch speech frames or endpoint detected
            while (!wav_provider.Done() && 
                    vad_pipeline->NumSpeechFramesReady() < forward_batch_) {
                // Read raw pcm audio
                int num_read = wav_provider.ReadAudio(chunk_length_, &data);
                if (num_read == 0) continue;
                std::cerr << "WavProvider.ReadAudio() read " << num_read << std::endl;
                // Feature extraction 
                SubVector<BaseFloat> wave_part(data.data(), num_read);
                vad_pipeline->AcceptWaveform(samp_freq_, wave_part);
                if (vad_pipeline->EndpointDetected()) break;
            }

            // Get voiced frames to feature pool 
            AddVadFeatureToFeaturePool(forward_batch_, vad_pipeline, feature_pool);

            // Advance decoding
            decoder.AdvanceDecoding();

            int partial_progress = 
                vad_pipeline->AudioReceived() - get_partial_result_progress;
            // Print partial results
            if (decoder.NumFramesDecoded() > 0 && 
                    !vad_pipeline->EndpointDetected() && 
                    partial_progress >= 0.7) {
                get_partial_result_progress = vad_pipeline->AudioReceived();
                std::string result;
                decoder.GetPartialResult(word_syms_table_, &result);
                if (result != "") {
                    wav_provider.WritePartialReslut(result);
                }
                KALDI_VLOG(1) << "Partial: " << result;
            } // if NumFramesDecoded > 0

            if (vad_pipeline->EndpointDetected() && 
                    decoder.NumFramesDecoded() > 0) {
                vad_pipeline->InputFinished();
                AddVadFeatureToFeaturePool(forward_batch_, vad_pipeline, feature_pool);
                feature_pool->InputFinished();
                decoder.AdvanceDecoding();

                std::string result;
                decoder.GetPartialResult(word_syms_table_, &result);
                KALDI_VLOG(1) << "-Final: " << result;
                if (result != "") {
                    wav_provider.WriteFinalReslut(result);
                    all_result += result;
                }
                delete vad_pipeline;
                delete feature_pool;
                vad_pipeline = new OnlineVadFeaturePipeline(*vad_nnet, 
                        vad_config_, feature_info_);
                feature_pool = new OnlineFeaturePool(vad_pipeline->Dim());
                decoder.ResetDecoder(feature_pool);
            }

        } // end while

        vad_pipeline->InputFinished();
        AddVadFeatureToFeaturePool(forward_batch_, vad_pipeline, feature_pool);
        feature_pool->InputFinished();

        decoder.FinalizeDecoding();

        std::string recog_result;
        CompactLattice clat;
        if (decoder.NumFramesDecoded() > 0) {
            bool end_of_utterance = true;
            decoder.GetLattice(end_of_utterance, &clat);
            GetDiagnosticsAndPrintOutput("+Final: ", word_syms_table_, clat,
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
            punctuation_processor_.Process(all_result, &punc_result);
            KALDI_LOG << "All Result: " << all_result;
            KALDI_LOG << "Final Punctuation Result: " << punc_result;
            wav_provider.WritePuncResult(punc_result);
        }
        wav_provider.WriteEOS();

        delete feature_pool;
        delete vad_pipeline;
    } catch (const std::exception &e) {
        std::cerr << e.what();
    }

}

} // namespace aslp_online
} // namespace kaldi
