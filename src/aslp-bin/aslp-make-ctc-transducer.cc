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
#include "fstext/fstext-lib.h"

//#include "fstext/context-fst.h"

namespace kaldi {

template<class Arc>
fst::VectorFst <Arc>* MakeCtcLoopFst(const vector<const fst::ExpandedFst<Arc> * > &fsts, int blank_id) {
  using namespace fst;
  typedef typename Arc::Weight Weight;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;

  VectorFst<Arc> *ans = new VectorFst<Arc>;
  StateId blank_state = ans->AddState();  // = 0.
  ans->SetStart(blank_state);
  ans->SetFinal(blank_state, Weight::One());
  
  StateId loop_state = ans->AddState();  // = 0.

  //connect blank_state-->first_state eps:eps
  ans->AddArc(blank_state, Arc(0, 0, Weight::One(), loop_state));
  //connect first_state-->blank_state blank:eps
  //ans->AddArc(loop_state, Arc(blank_id, 0, Weight::One(), blank_state));
  ans->AddArc(blank_state, Arc(blank_id, 0, Weight::One(), blank_state));

  // "cache" is used as an optimization when some of the pointers in "fsts"
  // may have the same value.
  unordered_map<const ExpandedFst<Arc> *, Arc> cache;

  for (Label i = 0; i < static_cast<Label>(fsts.size()); i++) {
    const ExpandedFst<Arc> *fst = fsts[i];
    if (fst == NULL) continue;
    { // optimization with cache: helpful if some members of "fsts" may
      // contain the same pointer value (e.g. in GetHTransducer).
      typename unordered_map<const ExpandedFst<Arc> *, Arc>::iterator
          iter = cache.find(fst);
      if (iter != cache.end()) {
        Arc arc = iter->second;
        arc.olabel = i;
        ans->AddArc(0, arc);
        continue;
      }
    }

    KALDI_ASSERT(fst->Properties(kAcceptor, true) == kAcceptor);  // expect acceptor.

    StateId fst_num_states = fst->NumStates();
    StateId fst_start_state = fst->Start();

    if (fst_start_state == kNoStateId)
      continue;  // empty fst.

    bool share_start_state =
        fst->Properties(kInitialAcyclic, true) == kInitialAcyclic
        && fst->NumArcs(fst_start_state) == 1
        && fst->Final(fst_start_state) == Weight::Zero();

    vector<StateId> state_map(fst_num_states);  // fst state -> ans state
    for (StateId s = 0; s < fst_num_states; s++) {
      if (s == fst_start_state && share_start_state) state_map[s] = loop_state;
      else state_map[s] = ans->AddState();
    }
    if (!share_start_state) {
      Arc arc(0, i, Weight::One(), state_map[fst_start_state]);
      cache[fst] = arc;
      ans->AddArc(0, arc);
    }
    for (StateId s = 0; s < fst_num_states; s++) {
      // Add arcs out of state s.
      for (ArcIterator<ExpandedFst<Arc> > aiter(*fst, s); !aiter.Done(); aiter.Next()) {
        const Arc &arc = aiter.Value();
        Label olabel = (s == fst_start_state && share_start_state ? i : 0);
        Arc newarc(arc.ilabel, olabel, arc.weight, state_map[arc.nextstate]);
        ans->AddArc(state_map[s], newarc);
        if (s == fst_start_state && share_start_state)
          cache[fst] = newarc;
      }
      if (fst->Final(s) != Weight::Zero()) {
        KALDI_ASSERT(!(s == fst_start_state && share_start_state));
        ans->AddArc(state_map[s], Arc(0, 0, fst->Final(s), blank_state));
      }
    }
  }
  return ans;
}

fst::VectorFst<fst::StdArc> *GetHmmAsCtcFst(
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
  for (size_t i = 0; i < entry.size(); i++)
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
      //bool is_self_loop = (dest_state == hmm_state);
      //if (is_self_loop) {
      ////  //continue; // We will add self-loops in at a later stage of processing,
      //   int32 trans_state =
      //      trans_model.TripleToTransitionState(phone, hmm_state, pdf);
      //   int32 trans_id = trans_model.SelfLoopOf(trans_state);
      //   if (trans_id != 0) {
      //  	log_prob = trans_model.GetTransitionLogProb(trans_id);
      //      //ans->AddArc(s, Arc(trans_id, 0, Weight(-log_prob*self_loop_scale), s));
      //      ans->AddArc(state_ids[hmm_state],
      //            Arc(label, label, Weight(-log_prob), state_ids[dest_state]));
      //   }
      //   continue;
      //}
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
        //KALDI_LOG << phone << " " << hmm_state << " " << pdf << " " << trans_id;
        //log_prob = trans_model.GetTransitionLogProbIgnoringSelfLoops(trans_id);
        log_prob = trans_model.GetTransitionLogProb(trans_id);
        // log_prob is a negative number (or zero)...
        label = trans_id;
      }
      // Will add probability-scale later (we may want to push first).
      ans->AddArc(state_ids[hmm_state],
                  Arc(label, label, Weight(-log_prob), state_ids[dest_state]));
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

fst::VectorFst<fst::StdArc> *GetCtcTransducer (const std::vector<std::vector<int32> > &ilabel_info,
                                             const ContextDependencyInterface &ctx_dep,
                                             const TransitionModel &trans_model,
                                             const HTransducerConfig &config,
                                             int blank_transition_id, 
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
      //KALDI_LOG << disambig_sym_left;
      fst->AddArc(0, Arc(disambig_sym_left, disambig_sym_left, Weight::One(), 1));
      fsts[j] = fst;
    } else {  // Real phone-in-context.
      std::vector<int32> phone_window = ilabel_info[j];

      VectorFst<Arc> *fst = GetHmmAsCtcFst(phone_window,
                                        ctx_dep,
                                        trans_model,
                                        config,
                                        &cache);
      fsts[j] = fst;
    }
  }
  VectorFst<Arc> *ans = MakeCtcLoopFst(fsts, blank_transition_id);
  SortAndUniq(&fsts); // remove duplicate pointers, which we will have
  // in general, since we used the cache.
  DeletePointers(&fsts);
  return ans;
}

}

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    typedef kaldi::int32 int32;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;
  	typedef StdArc Arc;
    typedef Arc::Weight Weight;
    typedef Arc::StateId StateId;
    

    const char *usage =
        "Make CTC transducer from transition-ids to context-dependent phones, \n"
        " without self-loops [use add-self-loops to add them]\n"
        "Usage:   aslp-make-ctc-transducer <ilabel-info-file> <tree-file> <transition-gmm/acoustic-model> [<H-fst-out>]\n"
        "e.g.: \n"
        " aslp-make-ctc-transducer ilabel_info  1.tree 1.mdl > H.fst\n";
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
    int blank_phone = trans_model.NumPhones();
    //get blank's transition-id
    int32 trans_state =
        trans_model.TripleToTransitionState(blank_phone, 0, 0);
    int32 trans_id =
        trans_model.PairToTransitionId(trans_state, 0);
    KALDI_ASSERT(trans_model.TransitionIdToPdfClass(trans_id) == 0);
    std::cerr << "blank_phone " << blank_phone << "\n" 
              << "trans_state " << trans_state << "\n" 
              << "trans_id " << trans_id << "\n"
              << "blank pdf " << trans_model.TransitionIdToPdfClass(trans_id) << "\n";
    // The work gets done here.
    fst::VectorFst<fst::StdArc> *H = GetCtcTransducer (ilabel_info,
                                                     ctx_dep,
                                                     trans_model,
                                                     hcfg,
                                                     trans_id,
                                                     &disambig_syms_out);
	//add by zhangbinbin	
    //StateId blank_state = H->AddState();  // = 0.
	//StateId start_state = H->Start();
	//H->SetStart(blank_state);
	//H->SetFinal(blank_state, Weight::One());
    ////connect blank_state-->first_state eps:eps
    //H->AddArc(blank_state, Arc(0, 0, Weight::One(), start_state));
    //get blank's transition-id
    //int32 trans_state =
    //    trans_model.TripleToTransitionState(1, 0, 0);
    //int32 trans_id =
    //    trans_model.PairToTransitionId(trans_state, 0);
    ////connect first_state-->blank_state blank:eps
    //H->AddArc(start_state, Arc(trans_id, 0, Weight::One(), blank_state));

  	//SortAndUniq(&H); // remove duplicate pointers, which we will have
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

