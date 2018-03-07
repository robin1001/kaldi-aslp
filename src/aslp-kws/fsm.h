/*
 * Created on 2016-11-08
 * Author: Zhang Binbin
 */

#ifndef FSM_H_
#define FSM_H_

#include <stdio.h>
#include <vector>
#include <iostream>
#include <algorithm>

#include "util/stl-utils.h"
#include "utils.h"
#include "symbol-table.h"

namespace kaldi {
namespace kws {

template<class T> 
inline void ReadBasic(FILE *fin, T *t) {
    CHECK(t != NULL);
    int32_t size = fread(t, sizeof(*t), 1, fin);
    if (size != 1) {
        ERROR("Read failure in ReadBasic, file position is %ld",
              ftell(fin));
    }
}

template<class T>
inline void WriteBasic(FILE *fout, T t) {
    int32_t size = fwrite(&t, sizeof(t), 1, fout);
    if (size != 1) {
        ERROR("Write failure in WriteBasic.");
    }
}

struct Arc {
    Arc() {}
    Arc(int32_t ilabel, float weight, int32_t next_state): 
        ilabel(ilabel), weight(weight), next_state(next_state) {}

    void Read(FILE *fin) {
        ReadBasic(fin, &ilabel);
        ReadBasic(fin, &weight);
        ReadBasic(fin, &next_state);
    }

    void Write(FILE *fout) {
        WriteBasic(fout, ilabel);
        WriteBasic(fout, weight);
        WriteBasic(fout, next_state);
    }

    bool operator< (const Arc &arc) const {
        return ilabel < arc.ilabel;
    }

    int32_t ilabel;
	float weight;
    int32_t next_state;
};

struct State {
    State() {}

    void AddArc(const Arc &arc) {
        arcs.push_back(arc);
    }

    int32_t NumArcs() const {
        return arcs.size();
    }

    void SortArcs() {
        std::sort(arcs.begin(), arcs.end());         
    }

    void Read(FILE *fin) {
        int32_t num_arcs = 0; 
        ReadBasic(fin, &num_arcs);
        CHECK(num_arcs >= 0);
        arcs.resize(num_arcs);
        for (int32_t i = 0; i < num_arcs; i++) {
            arcs[i].Read(fin);
        }
    }

    void Write(FILE *fout) {
        int32_t num_arcs = arcs.size(); 
        WriteBasic(fout, num_arcs);
        for (int32_t i = 0; i < num_arcs; i++) {
            arcs[i].Write(fout);
        }
    }

    /// members
    std::vector<Arc> arcs;
};


// Finite State Machine
class Fsm {
public:
    Fsm(): start_(0) {}
    ~Fsm(); 
    void Reset();

    int32_t Start() const { 
        return start_; 
    }

    int32_t IsStart(int32_t id) const {
        return id == start_;
    }

    void SetStart(int32_t id) {
		CHECK(id < states_.size());
        start_ = id;
    }

    void SetFinal(int32_t id) {
		CHECK(id < states_.size());
        final_set_.insert(id);
    }

    int32_t NumFinals() const {
        return final_set_.size();
    }

    int32_t NumStates() const {
        return states_.size();
    }

    const Arc *ArcStart(int32_t id) const {
		CHECK(id < states_.size());
        return states_[id]->arcs.data();
    }

    const Arc *ArcEnd(int32_t id) const {
		CHECK(id < states_.size());
        return states_[id]->arcs.data() + states_[id]->arcs.size();
    }

    const Arc *ArcSeek(int32_t state_id, int32_t arc_id) const {
		CHECK(state_id < states_.size());
		CHECK(arc_id < states_[state_id]->arcs.size());
        return &states_[state_id]->arcs[arc_id];
    }

    int32_t NumArcs() const;
    int32_t NumArcs(int32_t id) const;
    void SortArcs();
    bool IsFinal(int32_t id) const;
    int32_t AddState();
    void AddArc(int32_t id, const Arc &arc);
    void ReadTopo(const char *file);
	void Read(const char *file); //read fsm from file
	void Write(const char *file) const; // write fsm to file
    void Info() const; 
    // print to dot format
    void Dot(SymbolTable *symbol_table = NULL) const;

    // Determine, simple determine, ignore weight
//    void Determine(Fsm *fsm_out) const;
//    // Trim, delete states which can not reach the final state
//    void Trim(Fsm *fsm_out) const; 
//    // Simple FSM, all arcs weight are 0
//    bool IsSimpleFsm() const;
//    void Minimize(Fsm *fsm_out) const;
//private:
//    // for set hash 
//    class SetIntHash {
//    public:
//        size_t operator()(const std::unordered_set<int>&t) const {
//            size_t sum = 0;
//            for (std::unordered_set<int>::const_iterator it = t.begin();
//                    it != t.end(); it++) {
//                sum += *it * 131;
//            }
//            return sum;
//            //return reinterpret_cast<size_t>(&t);
//        }
//    };
//    // for set equal 
//    class SetIntEqual {
//    public:
//        bool operator() (const std::unordered_set<int> &s1, 
//                         const std::unordered_set<int> &s2) const {
//            if (s1.size() != s2.size()) return false;
//            for (std::unordered_set<int>::const_iterator it = s1.begin(); 
//                    it != s1.end(); it++) {
//                if (s2.find(*it) == s2.end()) return false;
//            }
//            return true;
//        }
//    };
//
//    typedef std::unordered_map<std::unordered_set<int>, int, 
//            SetIntHash, SetIntEqual> SetTable;
//    typedef SetTable::iterator SetTableIter;
//    typedef std::unordered_set<std::unordered_set<int>, 
//            SetIntHash, SetIntEqual> SetSet;
//    typedef SetSet::iterator SetSetIter;
//
//    bool HaveFinal(const std::unordered_set<int> & in_set) const;
//    void EpsilonClosure(const std::unordered_set<int> &in_set, 
//            std::unordered_set<int> *out_set) const; 
//    void MinimizeOnly(Fsm *fsm_out) const;
//    void GetLabelSet(const std::unordered_set<int> &state_set, 
//            std::unordered_set<int> *label_set) const;
//    void Move(const std::unordered_set<int> &in_set, int32_t label, 
//            std::unordered_set<int> *out_set) const; 
//    bool IsSubset(const std::unordered_set<int> &set0, 
//            const std::unordered_set<int> &set1) const; 
//
//    int SplitSetByInput(const std::unordered_set<int> &in_set, int label,
//        std::unordered_map<int, std::unordered_set<int>* > *table, 
//        std::vector<std::unordered_set<int> *> *split,
//        std::unordered_set<std::unordered_set<int> *> *set_table) const; 
//    bool EqualSet(const std::unordered_set<int> &s1, 
//            const std::unordered_set<int> &s2) const; 
//    std::unordered_set<int>* FindSet(const std::unordered_set<int> &s,
//            const std::unordered_set<std::unordered_set<int> *> &set_set) const; 
protected:
    int32_t start_;
    unordered_set<int32_t> final_set_;
    std::vector<State *> states_;
    DISALLOW_COPY_AND_ASSIGN(Fsm);
};


//Fsm *FsmConcat(const Fsm &fsm1, const Fsm &fsm2);
//Fsm *FsmUnion(const Fsm &fsm1, const Fsm &fsm2);
//Fsm *FsmClosure(const Fsm &fsm1);

}
}
#endif
