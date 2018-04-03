/* 
 * Created on 2016-11-11
 * Author: Zhang Binbin
 */

#ifndef SYMBOL_TABLE_H_
#define SYMBOL_TABLE_H_

#include <stdio.h>
#include <string>

#include "utils.h"

namespace kaldi {
namespace kws {

const int kEpsilon = 0;

class SymbolTable {
public:
    SymbolTable(const std::string &symbol_file) {
        ReadSymbolFile(symbol_file);
    }

    ~SymbolTable() {}

    std::string GetSymbol(int32_t id) const {
        CHECK(id < symbol_tabel_.size());
        return symbol_tabel_[id];
    }

    // GetId is used in the construction fst period
    // so here just a lazy/inefficient implemenation
    int32_t GetId(const std::string &symbol) const {
        for (int32_t i = 0; i < symbol_tabel_.size(); i++) {
            if (symbol == symbol_tabel_[i]) return i;
        }
        return 0;
    }

protected:
    void ReadSymbolFile(const std::string &symbol_file) {
        FILE *fp = fopen(symbol_file.c_str(), "r");
        if (!fp) {
            ERROR("%s not exint, please check!!!", symbol_file.c_str());
        }
        char buffer[1024], str[1024];
        int id;
        while (fgets(buffer, 1024, fp)) {
            int num = sscanf(buffer, "%s %d", str, &id);
            if (num != 2) {
                ERROR("each line shoud have 2 fields, symbol & id");
            }
            CHECK(str != NULL);
            CHECK(id >= 0);
            
            std::string symbol = str;
            if (id >= symbol_tabel_.size()) {
                symbol_tabel_.resize(id + 1); 
            }

            symbol_tabel_[id] = symbol;
        }
        fclose(fp);
    }
    
    std::vector<std::string> symbol_tabel_;
    DISALLOW_COPY_AND_ASSIGN(SymbolTable);
};

}
}

#endif
