/* 
 * Created on 2016-11-11
 * Author: Zhang Binbin
 */

#ifndef SYMBOL_TABLE_H_
#define SYMBOL_TABLE_H_

#include <stdio.h>

#include <string>

#include "util/stl-utils.h"
#include "utils.h"

namespace kaldi {
namespace kws {

const int kEpsilon = -1;

class SymbolTable {
public:
    SymbolTable(const char *symbol_file) {
        ReadWordFile(symbol_file);
        word_table_["<eps>"] = kEpsilon;
        id_table_[kEpsilon] = "<eps>";
    }

    ~SymbolTable() {}

    int MapToId(std::string word) const {
        if (!HaveSymbol(word)) {
            ERROR("%s is not in the symbol table", word.c_str());
        }
        return word_table_[word];
    }

    std::string MapToWord(int id) const {
        return id_table_[id];
    }

    bool HaveSymbol(std::string word) const {
        if (word_table_.find(word) != word_table_.end()) {
            return true;
        } else {
            return false;
        }
    }

    int NumSymbols() const {
        return word_table_.size();
    }

protected:
    void ReadWordFile(const char *symbol_file) {
        word_table_.clear();
        id_table_.clear();
        FILE *fp = fopen(symbol_file, "r");
        if (!fp) {
            ERROR("%s not exint, please check!!!", symbol_file);
        }
        char buffer[1024], str[1024];
        int word_id;
        while (fgets(buffer, 1024, fp)) {
            int num = sscanf(buffer, "%s %d", str, &word_id);
            if (num != 2) {
                ERROR("each line shoud have 2 fields, symbol & id");
            }
            CHECK(str != NULL);
            CHECK(word_id >= 0);
            std::string word  = str;
            word_table_[word] = word_id;
            id_table_[word_id] = word;
        }
        fclose(fp);
    }

    mutable unordered_map<std::string, int> word_table_;
    mutable unordered_map<int, std::string> id_table_;
    DISALLOW_COPY_AND_ASSIGN(SymbolTable);
};

}
}

#endif
