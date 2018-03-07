/* 
 * Created on 2016-11-08
 * Author: Zhang Binbin
 */

#include <string.h>

#include <queue>

#include "fsm.h"

namespace kaldi {
namespace kws {

Fsm::~Fsm() {
    Reset();
}

void Fsm::Reset() {
    start_ = 0;
    for (int32_t i = 0; i < states_.size(); i++) {
        if (states_[i] != NULL) {
            delete states_[i];
        }
    }
    states_.clear();
    final_set_.clear();
}

int32_t Fsm::AddState() {
    int32_t id = states_.size();
    states_.push_back(new State());
    return id;
}

void Fsm::AddArc(int32_t id, const Arc &arc) {
    int32_t max = id > arc.next_state ? id : arc.next_state;
    int32_t old_size = states_.size();
    if (max >= old_size) {
        for (int32_t i = old_size; i < max + 1; i++) {
            states_.push_back(new State());
        }
    }
    CHECK(id < states_.size());
    states_[id]->AddArc(arc);
}

int32_t Fsm::NumArcs() const {
    int32_t count = 0;
    for (int32_t i = 0; i < states_.size(); i++) {
        count += states_[i]->NumArcs();
    }
    return count;
}

int32_t Fsm::NumArcs(int32_t id) const {
    CHECK(id < states_.size());
    return states_[id]->NumArcs();
}

void Fsm::SortArcs() {
    for (int32_t i = 0; i < states_.size(); i++) {
        states_[i]->SortArcs();
    }
}

bool Fsm::IsFinal(int32_t id) const {
    return (final_set_.find(id) != final_set_.end());
}

/* read fsm from text file, eg:
   src  dest    ilabel  weight 
   0    1       1       1.0
   ...
   n
   attention: first line src must be start state,
   only one src in the last line as the final state
*/

void Fsm::ReadTopo(const char *file) {
    CHECK(file != NULL);
    Reset();

    FILE *fp = fopen(file, "r");
    if (!fp) {
        ERROR("file %s not exist", file);
    }

    char buffer[1024];
    bool first_line = true;
    int32_t src, dest, label;
    float weight = 0.0f;

    while(fgets(buffer, 1024, fp)) {
        int32_t num = sscanf(buffer, "%d %d %d %f", &src, &dest, &label, &weight);
        //LOG("read arc %d %d %d", src, dest, label);
        if (num == 3 || num == 4) {
            if (num == 3) weight = 0.0;
            this->AddArc(src, Arc(label, weight, dest));
            if (first_line) {
                SetStart(src);
                first_line = false;
            }
        }
        else if (num == 1) {//final state
            SetFinal(src);
        }
        else {
            LOG("wrong line, expected (src, dest, label, weight) or (final)");
            break;
        }
    }
    fclose(fp);
}


// fsm read write file format 
// * num_states(int) 
// * start_(int) 
// * final_set_(int)
// * every state's arcs info: NumArcs(int) {ilabel(int) weight(float) next_state(int) ...}

void Fsm::Read(const char *file) {
    CHECK(file != NULL);
    Reset();

    FILE *fin = NULL;
    if (strcmp(file, "-") == 0) {
        fin = stdin;
    } else {
        fin = fopen(file, "rb");
    }

    if (!fin) {
        ERROR("file %s not exist", file);
    }

    int32_t num_states, num_finals;
    // read start, state final number
    ReadBasic(fin, &num_states);
    ReadBasic(fin, &start_);
    ReadBasic(fin, &num_finals);

    int32_t state_id = 0;
    // read final set
    for (int32_t i = 0; i < num_finals; i++) {
        fread(&state_id, sizeof(int32_t), 1, fin);
        final_set_.insert(state_id);
    }

    for (int32_t i = 0; i < num_states; i++) {
        states_.push_back(new State());
    }
    // read state info
    for (int32_t i = 0; i < num_states; i++) {
        states_[i]->Read(fin);
    }

    if (fin != stdin) {
        fclose(fin);
    }
}

// sort and Write
void Fsm::Write(const char *file) const {
    CHECK(file != NULL);
    // write
    FILE *fout = NULL;
    if (strcmp(file, "-") == 0) {
        fout = stdout;
    }
    else {
        fout = fopen(file, "wb");
    }

    int32_t num_states = NumStates();
    int32_t num_finals = NumFinals();
    // write start, state & final number
    WriteBasic(fout, num_states);
    WriteBasic(fout, start_);
    WriteBasic(fout, num_finals);

    unordered_set<int32_t>::const_iterator it = final_set_.begin();
    // write final set
    for (; it != final_set_.end(); it++) {
        WriteBasic(fout, *it);
    }
    // write state info
    for (int32_t i = 0; i < num_states; i++) {
        states_[i]->Write(fout);
    }

    if (fout != stdout) {
        fclose(fout);
    }
}

// Show the text format fsm info
void Fsm::Info() const {
    fprintf(stderr, "fsm info table\n");
    // state arc start info
    fprintf(stderr, "num_states:\t%d\n", NumStates());
    fprintf(stderr, "num_arcs:\t%d\n", NumArcs());
    fprintf(stderr, "start id:\t%d\n", start_);
    // final set info
    fprintf(stderr, "final set:\t%d { ", NumFinals());
    unordered_set<int32_t>::const_iterator it = final_set_.begin();
    for (; it != final_set_.end(); it++) {
        fprintf(stderr, "%d ", *it);
    }
    fprintf(stderr, "}\n");

    // state info
    for (int32_t i = 0; i < states_.size(); i++) {
        fprintf(stderr, "state %d arcs %d: { ", i, states_[i]->NumArcs());
        for (int32_t j = 0; j < states_[i]->NumArcs(); j++) {
            fprintf(stderr, "(%d, %f, %d) ", states_[i]->arcs[j].ilabel, 
                                             states_[i]->arcs[j].weight,
                                             states_[i]->arcs[j].next_state);
        }
        fprintf(stderr, "}\n");
    }
}

void Fsm::Dot(SymbolTable *symbol_table) const {
    printf("digraph FSM {\n");
    printf("rankdir = LR;\n");
    // printf("orientation = Landscape;\n");
    printf("node [shape = \"circle\"]\n");
    for (int32_t i = 0; i < states_.size(); i++) {
        if (IsFinal(i)) {
            printf("%d [label = \"%d\" shape = doublecircle ]\n", i, i);
        } else {
            printf("%d [label = \"%d\" ]\n", i, i);
        }
        for (int32_t j = 0; j < states_[i]->NumArcs(); j++) {
            if (symbol_table == NULL) {
                printf("\t %d -> %d [label = \"%d/%f\" ]\n", i, 
                        states_[i]->arcs[j].next_state,
                        states_[i]->arcs[j].ilabel,
                        states_[i]->arcs[j].weight);
            } else {
                printf("\t %d -> %d [label = \"%s/%f\" ]\n", i, 
                        states_[i]->arcs[j].next_state,
                        symbol_table->MapToWord(states_[i]->arcs[j].ilabel).c_str(),
                        states_[i]->arcs[j].weight);
            }
        }
    }
    printf("}\n");
}

//bool Fsm::IsSimpleFsm() const {
//    for (int i = 0; i < states_.size(); i++) {
//        for (const Arc *arc = ArcStart(i); arc != ArcEnd(i); arc++) {
//            if (arc->weight != 0.0) {
//                return false;
//            }
//        }
//    }
//    return true;
//}
//
//void Fsm::EpsilonClosure(const std::unordered_set<int> &in_set, 
//        std::unordered_set<int> *out_set) const {
//    CHECK(out_set != NULL);
//    out_set->clear();
//    std::queue<int> q;
//    for (std::unordered_set<int>::const_iterator it = in_set.begin();
//            it != in_set.end(); it++) {
//        q.push(*it);
//        out_set->insert(*it);
//    }
//    while (!q.empty()) {
//        int id = q.front();
//        q.pop();
//        CHECK(id < states_.size());
//        for (int i = 0; i < states_[id]->NumArcs(); i++) {
//            if (0 == states_[id]->arcs[i].ilabel) {
//                int next_state = states_[id]->arcs[i].next_state;
//                if (out_set->find(next_state) == out_set->end()) {
//                    out_set->insert(next_state);
//                    q.push(next_state);
//                }
//            }
//        }
//    }
//}
//
//bool Fsm::HaveFinal(const std::unordered_set<int> & in_set) const {
//    for (std::unordered_set<int>::const_iterator it = in_set.begin();
//            it != in_set.end(); it++) {
//        if (IsFinal(*it)) return true;
//    }
//    return false;
//}
//
//void Fsm::Determine(Fsm *fsm_out) const {
//    CHECK(fsm_out != NULL);
//    if (!IsSimpleFsm()) {
//        ERROR("determine error: this fsm is not simple fsm"
//              "(all arcs weight must be zero)");
//    }
//    fsm_out->Reset();
//
//    SetTable table;
//    std::unordered_set<int> current_set, tmp_set, next_set;
//    tmp_set.insert(start_);
//    EpsilonClosure(tmp_set, &current_set);
//    int start = fsm_out->AddState();
//    fsm_out->SetStart(start);
//    if (HaveFinal(current_set)) fsm_out->SetFinal(start);
//    SetTableIter iter = table.begin();
//    iter = table.insert(iter, std::make_pair(current_set, start));
//    std::queue<SetTableIter > q;
//    q.push(iter); //second is bool
//    std::unordered_map<int, std::unordered_set<int> > move_table;
//
//    while (!q.empty()) {
//        SetTableIter iter_q = q.front();
//        q.pop();
//        current_set = iter_q->first;
//        int src_state = iter_q->second;
//        move_table.clear();
//        // Move on each label;
//        //std::cerr << "current set { ";
//        for (std::unordered_set<int>::iterator it = current_set.begin(); 
//                it != current_set.end(); it++) {
//            //std::cerr << *it << " ";
//            State *state = states_[*it];
//            for (int i = 0; i < state->NumArcs(); i++) {
//                int label = state->arcs[i].ilabel;
//                int next_state =  state->arcs[i].next_state;
//                if (label == 0) continue;
//                if (move_table.find(label) == move_table.end()) { //new label in map
//                    move_table.insert(std::make_pair(label, 
//                                std::unordered_set<int>()));
//                }
//                move_table[label].insert(next_state);
//            }
//        }
//        //std::cerr << "}\n";
//        // Epsilon closure
//        int dest_state = 0;
//        for (std::unordered_map<int, std::unordered_set<int> >::iterator it = 
//                move_table.begin(); it != move_table.end(); it++) {
//            EpsilonClosure(it->second, &next_set); 
//            if (table.find(next_set) == table.end()) {
//                dest_state = fsm_out->AddState();
//                if (HaveFinal(next_set)) fsm_out->SetFinal(dest_state);
//                iter = table.insert(iter, std::make_pair(next_set, dest_state));
//                q.push(iter);
//            }
//            else {
//                dest_state = table[next_set];
//            }
//            //std::cerr << "\ton label " << it->first << " turn to next set { ";
//            //for (std::unordered_set<int>::iterator ip = next_set.begin(); 
//            //        ip != next_set.end(); ip++) {
//            //    std::cerr << *ip << " ";
//            //}
//            //std::cerr << "}\n";
//            fsm_out->AddArc(src_state, Arc(it->first, 0.0f, dest_state));
//        }
//    }
//}
//
//
//// Delete states which can not reach the final state
//void Fsm::Trim(Fsm *fsm_out) const {
//    CHECK(fsm_out != NULL);
//    fsm_out->Reset();
//    std::unordered_set<int> reach_final_set(final_set_);
//    std::unordered_set<int> none_final_set, prev_none_final_set;
//    for (int i = 0; i < states_.size(); i++) {
//        if (reach_final_set.find(i) == reach_final_set.end()) {
//            none_final_set.insert(i);
//        }
//    }
//    while (true) {
//        prev_none_final_set  = none_final_set;
//        for (std::unordered_set<int>::iterator it = prev_none_final_set.begin(); 
//                it != prev_none_final_set.end(); it++) {
//            State *state = states_[*it];
//            for (int j = 0; j < state->NumArcs(); j++) {
//                int next_state = state->arcs[j].next_state;
//                if (reach_final_set.find(next_state) != reach_final_set.end()) {
//                    reach_final_set.insert(*it); 
//                    none_final_set.erase(*it);
//                }
//            }
//        }
//        if (none_final_set.size() == prev_none_final_set.size())
//            break;
//    }
//    // Start Must in none_final_set
//    CHECK(reach_final_set.find(start_) != reach_final_set.end());
//
//    std::unordered_map<int, int> state_map;
//    int start = fsm_out->AddState();
//    state_map.insert(std::make_pair(start_,start));
//    for (int i = 0; i < states_.size(); i++) {
//        if (i == start_) continue;
//        if (reach_final_set.find(i) != reach_final_set.end()) {
//            int state = fsm_out->AddState(); 
//            if (IsFinal(i)) fsm_out->SetFinal(state);
//            state_map.insert(std::make_pair(i, state));
//        }
//    }
//    for (std::unordered_map<int, int>::iterator it = state_map.begin();
//            it != state_map.end(); it++) {
//        State *state = states_[it->first];
//        for (int j = 0; j < state->NumArcs(); j++) {
//            int label = state->arcs[j].ilabel;
//            float weight = state->arcs[j].weight;
//            int next_state = state->arcs[j].next_state;
//            if (state_map.find(next_state) != state_map.end()) {
//                fsm_out->AddArc(it->second, 
//                        Arc(label, weight, state_map[next_state]));
//            }
//        }
//    }
//}
//
//// First trim, then do minimize
//void Fsm::Minimize(Fsm *fsm_out) const {
//    if (!IsSimpleFsm()) {
//        ERROR("minimize error: this fsm is not simple fsm"
//              "(all arcs weight must be zero)");
//    }
//    CHECK(fsm_out != NULL);
//    Fsm fsm_trim;
//    Trim(&fsm_trim);
//    fsm_trim.MinimizeOnly(fsm_out);
//}
//
//void Fsm::GetLabelSet(const std::unordered_set<int> &state_set, 
//        std::unordered_set<int> *label_set) const {
//    CHECK(label_set != NULL);
//    label_set->clear();
//    for (std::unordered_set<int>::const_iterator it = state_set.begin(); 
//            it != state_set.end(); it++) {
//        State *state = states_[*it];
//        for (int j = 0; j < state->NumArcs(); j++) {
//            if (state->arcs[j].ilabel != 0) {
//                label_set->insert(state->arcs[j].ilabel);
//            }
//        }
//    } 
//}
//
//void Fsm::Move(const std::unordered_set<int> &in_set, int label, 
//        std::unordered_set<int> *out_set) const {
//    CHECK(out_set != NULL);
//    out_set->clear();
//    for (std::unordered_set<int>::const_iterator it = in_set.begin(); 
//            it != in_set.end(); it++) {
//        State *state = states_[*it];
//        for (int j = 0; j < state->NumArcs(); j++) {
//            if (label == state->arcs[j].ilabel) {
//                out_set->insert(state->arcs[j].next_state);
//            }
//        }
//    }
//}
//
//// If set1 is the subset of set0
//bool Fsm::IsSubset(const std::unordered_set<int> &set0, 
//        const std::unordered_set<int> &set1) const {
//    if (set1.size() > set0.size()) return false;
//    for (std::unordered_set<int>::const_iterator it = set1.begin(); 
//            it != set1.end(); it++) {
//        if (set0.find(*it) == set0.end()) {
//            return false;
//        }
//    }
//    return true;
//}
//
//bool Fsm::EqualSet(const std::unordered_set<int> &s1, 
//        const std::unordered_set<int> &s2) const {
//    if (s1.size() != s2.size()) return false;
//    for (std::unordered_set<int>::const_iterator it = s1.begin(); 
//            it != s1.end(); it++) {
//        if (s2.find(*it) == s2.end()) return false;
//    }
//    return true;
//}
//
//std::unordered_set<int> * Fsm::FindSet(const std::unordered_set<int> &s,
//        const std::unordered_set<std::unordered_set<int> *> &set_set) const {
//    for (std::unordered_set<std::unordered_set<int> *>::const_iterator it = 
//            set_set.begin(); it != set_set.end(); it++) {
//        if (EqualSet(s, **it)) return *it;
//    }
//    return NULL;
//}
//
//int Fsm::SplitSetByInput(const std::unordered_set<int> &in_set, int label,
//        std::unordered_map<int, std::unordered_set<int>* > *table, 
//        std::vector<std::unordered_set<int> *> *split,
//        std::unordered_set<std::unordered_set<int> *> *set_table) const {
//
//    CHECK(table != NULL);
//    CHECK(set_table != NULL);
//    if (in_set.size() == 1) return 1;
//
//    std::unordered_map<std::unordered_set<int>*, 
//        std::unordered_set<int> *> p_table;
//    
//    for (std::unordered_set<int>::const_iterator it = in_set.begin(); 
//            it != in_set.end(); it++) {
//        State * state = states_[*it];
//        std::unordered_set<int> *key = NULL;
//        int dest_state = -1; // no next_state on this input
//        for (int k = 0; k < state->NumArcs(); k++) {
//            if (label == state->arcs[k].ilabel) {
//                dest_state = state->arcs[k].next_state;
//                CHECK(table->find(dest_state) != table->end());
//                key = (*table)[dest_state];
//                break;
//            }
//        }
//        if (p_table.find(key) == p_table.end()) {
//            p_table[key] = new std::unordered_set<int>();
//        }
//        p_table[key]->insert(*it);
//    }
//
//    if (p_table.size() == 1) {
//        delete p_table.begin()->second;
//        return 1;
//    }
//
//    split->clear();
//    for (std::unordered_map<std::unordered_set<int>*, 
//            std::unordered_set<int>* >::iterator it = p_table.begin(); 
//            it != p_table.end(); it++) {
//        std::unordered_set<int> *p_set = FindSet(*it->second, *set_table);
//        if (p_set != NULL) {
//            delete it->second;
//        } else {
//            p_set = it->second;
//            set_table->insert(it->second);
//        }
//        split->push_back(p_set);
//        for (std::unordered_set<int>::iterator jt = it->second->begin();
//                jt != it->second->end(); jt++) {
//            (*table)[*jt] = p_set;
//        }
//    }
//
//    return p_table.size();
//}
//
//// Only minimize
//void Fsm::MinimizeOnly(Fsm *fsm_out) const {
//    CHECK(fsm_out != NULL);
//    std::unordered_map<int, std::unordered_set<int>* > table; 
//    std::unordered_set<std::unordered_set<int> *> set_set;
//    std::unordered_set<int> *final_set = new std::unordered_set<int>(),
//                            *none_final_set = new std::unordered_set<int>();
//    // Split into two sets, final and none final states
//    for (int i = 0; i < states_.size(); i++) {
//        if (IsFinal(i)) {
//            final_set->insert(i);
//            table[i] = final_set;
//        } else {
//            none_final_set->insert(i);
//            table[i] = none_final_set;
//        }
//    }
//    set_set.insert(final_set);
//    set_set.insert(none_final_set);
//    
//    std::vector<std::unordered_set<int> *> result;
//    std::queue<std::unordered_set<int> *> q;
//    std::unordered_set<std::unordered_set<int> *> q_table;
//    q.push(final_set);
//    q.push(none_final_set);
//    q_table.insert(final_set);
//    q_table.insert(none_final_set);
//    // split into equivalent set
//    while (!q.empty()) {
//        std::unordered_set<int> *cur_set = q.front(); 
//        q.pop();
//        if (cur_set->size() == 1) {
//            result.push_back(cur_set);
//            continue;
//        }
//        bool can_split = false;
//        std::unordered_set<int> label_set;
//        GetLabelSet(*cur_set, &label_set);
//        for (std::unordered_set<int>::iterator it = label_set.begin();
//                it != label_set.end(); it++) {
//            std::vector<std::unordered_set<int> *> splits;
//            int num_splits = 
//                SplitSetByInput(*cur_set, *it, &table, &splits, &set_set);
//            if (num_splits > 1) {
//                for (int i = 0; i < splits.size(); i++) {
//                    if (q_table.find(splits[i]) == q_table.end()) {
//                        q.push(splits[i]);
//                        q_table.insert(splits[i]);
//                    }
//                }
//                can_split = true;
//                break;
//            }
//        }
//        if (!can_split) {
//            result.push_back(cur_set);
//        }
//    }
//
//    LOG("equivalent set size %lu", result.size());
//    for (int i = 0; i < result.size(); i++) {
//        std::cerr << "{ ";
//        for (std::unordered_set<int>::iterator it = result[i]->begin(); 
//                it != result[i]->end(); it++) {
//            std::cerr << *it << " ";
//        }
//        std::cerr << "}\n";
//    }
//
//    // Fsm out add state, set start, set final state
//    std::unordered_map<std::unordered_set<int> *, int> set_to_id;
//    for (int i = 0; i < result.size(); i++) {
//        if (result[i]->find(start_) != result[i]->end()) {
//            int state_id = fsm_out->AddState(); 
//            fsm_out->SetStart(state_id);
//            set_to_id[result[i]] = state_id;
//            break;
//        }
//    }
//    for (int i = 0; i < result.size(); i++) {
//        if (result[i]->find(start_) != result[i]->end()) {
//            continue;
//        }
//        int state_id = fsm_out->AddState(); 
//        if (HaveFinal(*result[i])) {
//            fsm_out->SetFinal(state_id);
//        }
//        set_to_id[result[i]] = state_id;
//    }
//    // Fsm and arc Information
//    //std::unordered_set<int> out_set;
//    for (int i = 0; i < result.size(); i++) {
//        int src = set_to_id[result[i]];
//        std::unordered_set<int> cur_set = *result[i];
//        std::unordered_set<int> label_set, out_set;
//        GetLabelSet(cur_set, &label_set);
//        for (std::unordered_set<int>::iterator jt = label_set.begin(); 
//                jt != label_set.end(); jt++) {
//            // out_set should be one equivalent set
//            Move(cur_set, *jt, &out_set);
//            CHECK(out_set.size() > 0);
//            std::unordered_set<int> *next = table[*out_set.begin()];
//            CHECK(set_to_id.find(next) != set_to_id.end());
//            int dest = set_to_id[next];
//            fsm_out->AddArc(src, Arc(*jt, 0.0f, dest));
//        }
//    }
//
//    for (std::unordered_set<std::unordered_set<int> *>::iterator it = 
//            set_set.begin(); it != set_set.end(); it++) {
//        delete *it;
//    }
//}
//
//Fsm *FsmConcat(const Fsm &fsm1, const Fsm &fsm2) {
//    int num_states = fsm1.NumStates() + fsm2.NumStates();
//    int offset = fsm1.NumStates();
//    int fsm2_start = offset + fsm2.Start();
//    Fsm *fsm = new Fsm();
//    for (int i = 0; i < num_states; i++) {
//        fsm->AddState(); 
//    }
//    fsm->SetStart(fsm1.Start());
//    
//    // Copy fsm1 to fsm
//    for (int i = 0; i < fsm1.NumStates(); i++) {
//        for (const Arc *arc = fsm1.ArcStart(i); arc != fsm1.ArcEnd(i); arc++) {
//            fsm->AddArc(i, Arc(arc->ilabel, arc->weight, arc->next_state));
//        }
//        // concat fsm1 final to fsm2 start
//        if (fsm1.IsFinal(i)) {
//            fsm->AddArc(i, Arc(0, 0.0f, fsm2_start));
//        }
//    }
//    // Copy fsm2 to fsm
//    for (int i = 0; i < fsm2.NumStates(); i++) {
//        for (const Arc *arc = fsm2.ArcStart(i); arc != fsm2.ArcEnd(i); arc++) {
//            fsm->AddArc(i + offset , 
//                    Arc(arc->ilabel, arc->weight, arc->next_state + offset));
//        }
//        // set fsm2 final as fsm final
//        if (fsm2.IsFinal(i)) {
//            fsm->SetFinal(i + offset);
//        }
//    }
//
//    return fsm;
//}
//
//Fsm *FsmUnion(const Fsm &fsm1, const Fsm &fsm2) {
//    int num_states = 1 + 1 + fsm1.NumStates() + fsm2.NumStates();
//    int fsm1_offset = 1;
//    int fsm2_offset = 1 + fsm1.NumStates();
//    Fsm *fsm = new Fsm();
//    for (int i = 0; i < num_states; i++) {
//        fsm->AddState(); 
//    }
//    fsm->SetStart(0);
//    fsm->SetFinal(num_states - 1);
//    // Copy fsm1 to fsm
//    for (int i = 0; i < fsm1.NumStates(); i++) {
//        for (const Arc *arc = fsm1.ArcStart(i); arc != fsm1.ArcEnd(i); arc++) {
//            fsm->AddArc(i + fsm1_offset, Arc(arc->ilabel, arc->weight, 
//                        arc->next_state + fsm1_offset));
//        }
//        if (fsm1.IsStart(i)) {
//            fsm->AddArc(0, Arc(0, 0.0f, i + fsm1_offset));
//        } 
//        else if (fsm1.IsFinal(i)) {
//            fsm->AddArc(i + fsm1_offset, Arc(0, 0.0f, num_states - 1));
//        }
//    }
//    // Copy fsm2 to fsm
//    for (int i = 0; i < fsm2.NumStates(); i++) {
//        for (const Arc *arc = fsm2.ArcStart(i); arc != fsm2.ArcEnd(i); arc++) {
//            fsm->AddArc(i + fsm2_offset, Arc(arc->ilabel, arc->weight, 
//                        arc->next_state + fsm2_offset));
//        }
//        if (fsm2.IsStart(i)) {
//            fsm->AddArc(0, Arc(0, 0.0f, i + fsm2_offset));
//        } 
//        else if (fsm2.IsFinal(i)) {
//            fsm->AddArc(i + fsm2_offset, Arc(0, 0.0f, num_states - 1));
//        }
//    }
//
//    return fsm;
//}
//
//
//Fsm *FsmClosure(const Fsm &fsm1) {
//    int num_states = 1 + 1 + fsm1.NumStates();
//    int fsm1_offset = 1;
//    Fsm *fsm = new Fsm();
//    for (int i = 0; i < num_states; i++) {
//        fsm->AddState(); 
//    }
//    fsm->SetStart(0);
//    fsm->SetFinal(num_states - 1);
//    fsm->AddArc(0, Arc(0, 0.0f, num_states - 1));
//    // Copy fsm1 to fsm
//    for (int i = 0; i < fsm1.NumStates(); i++) {
//        for (const Arc *arc = fsm1.ArcStart(i); arc != fsm1.ArcEnd(i); arc++) {
//            fsm->AddArc(i + fsm1_offset, Arc(arc->ilabel, arc->weight, 
//                        arc->next_state + fsm1_offset));
//        }
//        if (fsm1.IsStart(i)) {
//            fsm->AddArc(0, Arc(0, 0.0f, i + fsm1_offset));
//        }
//        else if (fsm1.IsFinal(i)) {
//            fsm->AddArc(i + fsm1_offset, Arc(0, 0.0f, 0));
//        }
//    }
//    return fsm;
//}
//

}
}
