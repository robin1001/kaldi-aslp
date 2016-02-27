// aslp-bin/aslp-make-ctc-transducer.cc

// Copyright 2009-2011 Microsoft Corporation
// Copyright 2015  ASLP (author: zhangbinbin)
// Modified on 2015-08-30

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

#include "hmm/transition-model.h"
#include "hmm/hmm-utils.h"
#include "tree/context-dep.h"
#include "util/common-utils.h"
#include "fst/fstlib.h"
#include "fstext/table-matcher.h"
#include "fstext/fstext-utils.h"
#include "fstext/context-fst.h"

namespace kaldi {

fst::VectorFst<fst::StdArc> *GetHmmAsFst3(
    std::vector<int32> phone_window,
    const ContextDependencyInterface &ctx_dep,
    const TransitionModel &trans_model,
    const HTransducerConfig &config,    
    HmmCacheType *cache) {
  using namespace fst;

  if (config.reverse) ReverseVector(&phone_window);  // phone_window represents backwards
  // phone sequence.  Make it "forwards" so the ctx_dep object can interpret it
  // right.  will also have to reverse the FST we produce.

  if (static_cast<int32>(phone_window.size()) != ctx_dep.ContextWidth())
    KALDI_ERR <<"Context size mismatch, ilabel-info [from context FST is "
              <<(phone_window.size())<<", context-dependency object "
        "expects "<<(ctx_dep.ContextWidth());

  int P = ctx_dep.CentralPosition();
  int32 phone = phone_window[P];
  if (phone == 0) {  // error.  Error message depends on whether reversed.
    if (config.reverse)
      KALDI_ERR << "phone == 0.  Possibly you are trying to get a reversed "
          "FST with a non-central \"central position\" P (i.e. asymmetric "
          "context), but forgot to initialize the ContextFst object with P "
          "as N-1-P (or it could be a simpler problem)";
    else
      KALDI_ERR << "phone == 0.  Some mismatch happened, or there is "
          "a code error.";
  }

  const HmmTopology &topo = trans_model.GetTopo();
  const HmmTopology::TopologyEntry &entry  = topo.TopologyForPhone(phone);

  // vector of the pdfs, indexed by pdf-class (pdf-classes must start from zero
  // and be contiguous).
  std::vector<int32> pdfs(topo.NumPdfClasses(phone));
  for (int32 pdf_class = 0;
       pdf_class < static_cast<int32>(pdfs.size());
       pdf_class++) {
    if ( ! ctx_dep.Compute(phone_window, pdf_class, &(pdfs[pdf_class])) ) {
      std::ostringstream ctx_ss;
      for (size_t i = 0; i < phone_window.size(); i++)
        ctx_ss << phone_window[i] << ' ';
      KALDI_ERR << "GetHmmAsFst: context-dependency object could not produce "
                << "an answer: pdf-class = " << pdf_class << " ctx-window = "
                << ctx_ss.str() << ".  This probably points "
          "to either a coding error in some graph-building process, "
          "a mismatch of topology with context-dependency object, the "
          "wrong FST being passed on a command-line, or something of "
          " that general nature.";
    }
  }
  std::pair<int32, std::vector<int32> > cache_index(phone, pdfs);
  if (cache != NULL) {
    HmmCacheType::iterator iter = cache->find(cache_index);
    if (iter != cache->end())
      return iter->second;
  }
  
  VectorFst<StdArc> *ans = new VectorFst<StdArc>;

  typedef StdArc Arc;
  typedef Arc::Weight Weight;
  typedef Arc::StateId StateId;
  typedef Arc::Label Label;

  std::vector<StateId> state_ids;
  KALDI_ASSERT(entry.size() == 2);
  //add 2 more states
  for (size_t i = 0; i < entry.size() + 2; i++)
    state_ids.push_back(ans->AddState());
  KALDI_ASSERT(state_ids.size() != 0);  // Or empty topology entry.
  ans->SetStart(state_ids[0]);
  StateId final = state_ids.back();
  ans->SetFinal(final, Weight::One());

  for (int32 hmm_state = 0;
       hmm_state < static_cast<int32>(entry.size());
       hmm_state++) {
    int32 pdf_class = entry[hmm_state].pdf_class, pdf;
    if (pdf_class == kNoPdf) pdf = kNoPdf;  // nonemitting state.
    else {
      KALDI_ASSERT(pdf_class < static_cast<int32>(pdfs.size()));
      pdf = pdfs[pdf_class];
    }
    int32 trans_idx;
    for (trans_idx = 0;
        trans_idx < static_cast<int32>(entry[hmm_state].transitions.size());
        trans_idx++) {
      BaseFloat log_prob;
      Label label;
      int32 dest_state = entry[hmm_state].transitions[trans_idx].first;
      bool is_self_loop = (dest_state == hmm_state);
      if (is_self_loop)
        continue; // We will add self-loops in at a later stage of processing,
      // not in this function.
      if (pdf_class == kNoPdf) {
        // no pdf, hence non-estimated probability.
        // [would not happen with normal topology] .  There is no transition-state
        // involved in this case.
        log_prob = Log(entry[hmm_state].transitions[trans_idx].second);
        label = 0;
      } else {  // normal probability.
        int32 trans_state =
            trans_model.TripleToTransitionState(phone, hmm_state, pdf);
        int32 trans_id =
            trans_model.PairToTransitionId(trans_state, trans_idx);
        log_prob = trans_model.GetTransitionLogProbIgnoringSelfLoops(trans_id);
        // log_prob is a negative number (or zero)...
        label = trans_id;
      }
      // Will add probability-scale later (we may want to push first).
      ans->AddArc(state_ids[hmm_state],
                  Arc(label, label, Weight(-log_prob), state_ids[dest_state]));
      //add two more state
      ans->AddArc(state_ids[dest_state],
                  Arc(label, label, Weight(-log_prob), state_ids[dest_state+1]));
      ans->AddArc(state_ids[dest_state+1],
                  Arc(label, label, Weight(-log_prob), state_ids[dest_state+2]));
    }
  }

  if (config.reverse) {
    VectorFst<StdArc> *tmp = new VectorFst<StdArc>;
    fst::Reverse(*ans, tmp);
    fst::RemoveEpsLocal(tmp);  // this is safe and will not blow up.
    if (config.push_weights)  // Push to make it stochastic again.
      PushInLog<REWEIGHT_TO_INITIAL>(tmp, kPushWeights, config.push_delta);
    delete ans;
    ans = tmp;
  } else {
    fst::RemoveEpsLocal(ans);  // this is safe and will not blow up.
  }

  // Now apply probability scale.
  // We waited till after the possible weight-pushing steps,
  // because weight-pushing needs "real" weights in order to work.
  ApplyProbabilityScale(config.transition_scale, ans);
  if (cache != NULL)
    (*cache)[cache_index] = ans;
  return ans;
}

// The H transducer has a separate outgoing arc for each of the symbols in ilabel_info.
fst::VectorFst<fst::StdArc> *GetHTransducer3 (const std::vector<std::vector<int32> > &ilabel_info,
                                             const ContextDependencyInterface &ctx_dep,
                                             const TransitionModel &trans_model,
                                             const HTransducerConfig &config,
                                             std::vector<int32> *disambig_syms_left) {
  KALDI_ASSERT(ilabel_info.size() >= 1 && ilabel_info[0].size() == 0);  // make sure that eps == eps.
  HmmCacheType cache;
  // "cache" is an optimization that prevents GetHmmAsFst repeating work
  // unnecessarily.
  using namespace fst;
  typedef StdArc Arc;
  typedef Arc::Weight Weight;
  typedef Arc::StateId StateId;
  typedef Arc::Label Label;

  std::vector<const ExpandedFst<Arc>* > fsts(ilabel_info.size(), NULL);
  std::vector<int32> phones = trans_model.GetPhones();

  KALDI_ASSERT(disambig_syms_left != 0);
  disambig_syms_left->clear();

  int32 first_disambig_sym = trans_model.NumTransitionIds() + 1;  // First disambig symbol we can have on the input side.
  int32 next_disambig_sym = first_disambig_sym;

  if (ilabel_info.size() > 0)
    KALDI_ASSERT(ilabel_info[0].size() == 0);  // make sure epsilon is epsilon...

  for (int32 j = 1; j < static_cast<int32>(ilabel_info.size()); j++) {  // zero is eps.
    KALDI_ASSERT(!ilabel_info[j].empty());
    if (ilabel_info[j].size() == 1 &&
       ilabel_info[j][0] <= 0) {  // disambig symbol

      // disambiguation symbol.
      int32 disambig_sym_left = next_disambig_sym++;
      disambig_syms_left->push_back(disambig_sym_left);
      // get acceptor with one path with "disambig_sym" on it.
      VectorFst<Arc> *fst = new VectorFst<Arc>;
      fst->AddState();
      fst->AddState();
      fst->SetStart(0);
      fst->SetFinal(1, Weight::One());
      fst->AddArc(0, Arc(disambig_sym_left, disambig_sym_left, Weight::One(), 1));
      fsts[j] = fst;
    } else {  // Real phone-in-context.
      std::vector<int32> phone_window = ilabel_info[j];

      VectorFst<Arc> *fst = GetHmmAsFst3(phone_window,
                                        ctx_dep,
                                        trans_model,
                                        config,
                                        &cache);
      fsts[j] = fst;
    }
  }

  VectorFst<Arc> *ans = MakeLoopFst(fsts);
  SortAndUniq(&fsts); // remove duplicate pointers, which we will have
  // in general, since we used the cache.
  DeletePointers(&fsts);
  return ans;
}

}


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Make h transducer for 3-state cd-phone from transition-ids to context-dependent phones, \n"
        " without self-loops [use add-self-loops to add them]\n"
        "Usage:   aslp-make-h3-transducer <ilabel-info-file> <tree-file> <transition-gmm/acoustic-model> [<H-fst-out>]\n"
        "e.g.: \n"
        " aslp-make-h3-transducer ilabel_info  1.tree 1.mdl > H.fst\n";
    ParseOptions po(usage);

    HTransducerConfig hcfg;
    std::string disambig_out_filename;
    hcfg.Register(&po);
    po.Register("disambig-syms-out", &disambig_out_filename, "List of disambiguation symbols on input of H [to be output from this program]");

    po.Read(argc, argv);

    if (po.NumArgs() < 3 || po.NumArgs() > 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string ilabel_info_filename = po.GetArg(1);
    std::string tree_filename = po.GetArg(2);
    std::string model_filename = po.GetArg(3);
    std::string fst_out_filename;
    if (po.NumArgs() >= 4) fst_out_filename = po.GetArg(4);
    if (fst_out_filename == "-") fst_out_filename = "";

    std::vector<std::vector<int32> > ilabel_info;
    {
      bool binary_in;
      Input ki(ilabel_info_filename, &binary_in);
      fst::ReadILabelInfo(ki.Stream(), binary_in, &ilabel_info);
    }

    ContextDependency ctx_dep;
    ReadKaldiObject(tree_filename, &ctx_dep);

    TransitionModel trans_model;
    ReadKaldiObject(model_filename, &trans_model);

    std::vector<int32> disambig_syms_out;

    // The work gets done here.
    fst::VectorFst<fst::StdArc> *H = GetHTransducer3 (ilabel_info,
                                                     ctx_dep,
                                                     trans_model,
                                                     hcfg,
                                                     &disambig_syms_out);
#if _MSC_VER
    if (fst_out_filename == "")
      _setmode(_fileno(stdout),  _O_BINARY);
#endif

    if (disambig_out_filename != "") {  // if option specified..
      if (disambig_out_filename == "-")
        disambig_out_filename = "";
      if (! WriteIntegerVectorSimple(disambig_out_filename, disambig_syms_out))
        KALDI_ERR << "Could not write disambiguation symbols to "
                   << (disambig_out_filename == "" ?
                       "standard output" : disambig_out_filename);
    }

    if (! H->Write(fst_out_filename) )
      KALDI_ERR << "make-h-transducer: error writing FST to "
                 << (fst_out_filename == "" ?
                     "standard output" : fst_out_filename);

    delete H;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

