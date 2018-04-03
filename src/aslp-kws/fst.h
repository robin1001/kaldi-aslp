/*
 * Created on 2016-11-08
 * Author: Zhang Binbin
 */

#ifndef FST_H_
#define FST_H_

#include <stdio.h>
#include <vector>
#include <iostream>
#include <algorithm>

#include "util/stl-utils.h"
#include "utils.h"
#include "symbol-table.h"

namespace kaldi {
namespace kws {

/* Here we keep the weight and olabel to make it a complete WFST structure,
   So it is convenient for further optimizaiton in the future */

struct Arc {
    Arc(): ilabel(0), olabel(0), weight(0.0f), next_state(0) {}
    Arc(int32_t ilabel, int32_t olabel, float weight, int32_t next_state): 
        ilabel(ilabel), olabel(olabel), weight(weight), 
        next_state(next_state) {}

    bool operator< (const Arc &arc) const {
        return ilabel < arc.ilabel;
    }

    void Read(FILE *fin); 
    void Write(FILE *fout) const; 

    int32_t ilabel, olabel;
	float weight;
    int32_t next_state;
};

class Fst {
public:
    Fst(): start_(0) {}
    Fst(const std::string &file) {
        Read(file);
    }
    //~Fst();
    void Reset();
    void Info() const;
    
    int32_t Start() const { 
        return start_; 
    }

    void SetStart(int32_t id) {
        start_ = id;
    }
    
    int32_t NumFinals() const {
        return finals_.size();
    }

    int32_t NumArcs() const {
        return arcs_.size();
    }

    int32_t NumStates() const {
        return arc_offset_.size();
    }
    
    bool IsFinal(int32_t id) const {
        return (finals_.find(id) != finals_.end());
    }

    int32_t NumArcs(int32_t id) const {
        if (id < NumStates() - 1) {
            return arc_offset_[id + 1] - arc_offset_[id];
        } else {
            return arcs_.size() - arc_offset_[id];
        }
    }

    const Arc *ArcStart(int32_t id) const {
        CHECK(id < NumStates());
        return arcs_.data() + arc_offset_[id];
    }

    const Arc *ArcEnd(int32_t id) const {
        CHECK(id < NumStates());
        if (id < NumStates() - 1) {
            return arcs_.data() + arc_offset_[id + 1];
        } else {
            return arcs_.data() + arcs_.size();
        }
    }

    void ReadTopo(const SymbolTable &isymbol_table, 
                  const SymbolTable &osymbol_table, 
                  const std::string &topo_file);
    
    void Read(const std::string &file);
    void Write(const std::string &file) const;
    void Dot(const SymbolTable &isymbol_table, 
             const SymbolTable &osymbol_table) const; 
private:
    int32_t start_;
    std::vector<int32_t> arc_offset_; // arc offset of state
    unordered_map<int32_t, float> finals_;
    std::vector<Arc> arcs_;
    DISALLOW_COPY_AND_ASSIGN(Fst);
};

}
}

#endif
