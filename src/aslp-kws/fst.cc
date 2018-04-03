/* 
 * Created on 2018-04-22
 * Author: Zhang Binbin
 */

#include <string.h>

#include "fst.h"

template<class T> 
static inline void ReadBasic(FILE *fin, T *t) {
    CHECK(t != NULL);
    int32_t size = fread(t, sizeof(*t), 1, fin);
    if (size != 1) {
        ERROR("Read failure in ReadBasic, file position is %ld",
              ftell(fin));
    }
}

template<class T>
static inline void WriteBasic(FILE *fout, T t) {
    int32_t size = fwrite(&t, sizeof(t), 1, fout);
    if (size != 1) {
        ERROR("Write failure in WriteBasic.");
    }
}

namespace kaldi {
namespace kws {

void Arc::Read(FILE *fin) {
    ReadBasic(fin, &ilabel);
    ReadBasic(fin, &weight);
    ReadBasic(fin, &olabel);
    ReadBasic(fin, &next_state);
}

void Arc::Write(FILE *fout) const {
    WriteBasic(fout, ilabel);
    WriteBasic(fout, weight);
    WriteBasic(fout, olabel);
    WriteBasic(fout, next_state);
}

void Fst::Reset() {
    start_ = 0;
    arcs_.clear();
    arc_offset_.clear();
    finals_.clear();
}

void Fst::ReadTopo(const SymbolTable &isymbol_table, 
                   const SymbolTable &osymbol_table, 
                   const std::string &topo_file) {
    Reset();
    FILE *fp = fopen(topo_file.c_str(), "r");
    if (!fp) {
        ERROR("file %s not exist", topo_file.c_str());
    }

    char buffer[1024];
    char ilabel[1024], olabel[1024];
    bool first_line = true;
    int32_t src, dest;
    float weight = 0.0f;
    
    std::vector<std::vector<Arc> > all_arcs;
    while(fgets(buffer, 1024, fp)) {
        int32_t num = sscanf(buffer, "%d %d %s %s %f", &src, &dest, ilabel, olabel, &weight);
        if (num >= 4) {
            if (num == 4) weight = 0;
            if (first_line) {
                first_line = false;
                start_ = src;
            }
            
            Arc arc(isymbol_table.GetId(ilabel), osymbol_table.GetId(olabel), weight, dest);
            if (src >= all_arcs.size()) all_arcs.resize(src + 1);
            if (dest >= all_arcs.size()) all_arcs.resize(dest + 1);
            all_arcs[src].push_back(arc);
        } 
        else if (sscanf(buffer, "%d %f", &src, &weight) == 2) {
            finals_[src] = weight; 
        }
        else {
            ERROR("wrong line, expected (src, dest, ilabel, olabel, weight) " 
                  "or (final, weight) but get %s", buffer);
            break;
        }
    }
    fclose(fp);

    arc_offset_.resize(all_arcs.size());
    int32_t offset = 0;
    for (int i = 0; i < all_arcs.size(); i++) {
        arc_offset_[i] = offset;
        arcs_.insert(arcs_.end(), all_arcs[i].begin(), all_arcs[i].end()); 
        offset += all_arcs[i].size();
    }
}

// Show the text format fsm info
void Fst::Info() const {
    fprintf(stderr, "fst info table\n");
    // state arc start info
    fprintf(stderr, "start id:\t%d\n", start_);
    fprintf(stderr, "num_states:\t%d\n", NumStates());
    fprintf(stderr, "num_arcs:\t%d\n", NumArcs());
    // final set info
    fprintf(stderr, "final states:\t%d { ", NumFinals());
    unordered_map<int32_t, float>::const_iterator it = finals_.begin();
    for (; it != finals_.end(); it++) {
        fprintf(stderr, "(%d, %f) ", it->first, it->second);
    }
    fprintf(stderr, "}\n");

    // state info
    for (int32_t i = 0; i < NumStates(); i++) {
        fprintf(stderr, "state %d arcs %d: { ", i, NumArcs(i));
        for (const Arc *arc = ArcStart(i); arc != ArcEnd(i); arc++) {
            fprintf(stderr, "(%d, %d, %f, %d) ", arc->ilabel, 
                                                 arc->olabel,
                                                 arc->weight,
                                                 arc->next_state);
        }
        fprintf(stderr, "}\n");
    }
}

void Fst::Read(const std::string &file) {
    Reset();

    FILE *fin = fopen(file.c_str(), "rb");
    if (!fin) {
        ERROR("file %s not exist", file.c_str());
    }

    int32_t num_states, num_finals, num_arcs;
    ReadBasic(fin, &start_);
    ReadBasic(fin, &num_states);
    ReadBasic(fin, &num_finals);
    ReadBasic(fin, &num_arcs);

    arc_offset_.resize(num_states);
    for (int i = 0; i < num_states; i++) {
        ReadBasic(fin, &arc_offset_[i]);   
    }

    for (int i = 0; i < num_finals; i++) {
        int32_t state;
        float weight;
        ReadBasic(fin, &state);
        ReadBasic(fin, &weight);
        finals_[state] = weight;
    }

    arcs_.resize(num_arcs);
    for (int i = 0; i < num_arcs; i++) {
        arcs_[i].Read(fin);
    }

    fclose(fin);
}

void Fst::Write(const std::string &file) const {
    FILE *fout = fopen(file.c_str(), "wb");
    if (!fout) {
        ERROR("can not oopen file %s write", file.c_str());
    }

    int32_t num_states = NumStates(), 
            num_finals = NumFinals(), 
            num_arcs = NumArcs();
    WriteBasic(fout, start_);
    WriteBasic(fout, num_states);
    WriteBasic(fout, num_finals);
    WriteBasic(fout, num_arcs);

    for (int i = 0; i < num_states; i++) {
        WriteBasic(fout, arc_offset_[i]);   
    }
    
    unordered_map<int, float>::const_iterator it = finals_.begin();
    for (; it != finals_.end(); it++) {
        WriteBasic(fout, it->first);
        WriteBasic(fout, it->second);
    }

    for (int i = 0; i < num_arcs; i++) {
        arcs_[i].Write(fout);
    }

    fclose(fout);
}

void Fst::Dot(const SymbolTable &isymbol_table, 
              const SymbolTable &osymbol_table) const {
    printf("digraph FSM {\n");
    printf("rankdir = LR;\n");
    // printf("orientation = Landscape;\n");
    printf("node [shape = \"circle\"]\n");
    for (int32_t i = 0; i < NumStates(); i++) {
        if (IsFinal(i)) {
            printf("%d [label = \"%d\" shape = doublecircle ]\n", i, i);
        } else {
            printf("%d [label = \"%d\" ]\n", i, i);
        }
        for (const Arc *arc = ArcStart(i); arc != ArcEnd(i); arc++) {
            printf("\t %d -> %d [label = \"%s:%s/%f\" ]\n", i, 
                arc->next_state,
                isymbol_table.GetSymbol(arc->ilabel).c_str(),
                osymbol_table.GetSymbol(arc->olabel).c_str(),
                arc->weight);
        }
    }
    printf("}\n");
}

}
}
