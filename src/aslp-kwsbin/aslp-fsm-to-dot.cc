#include <stdio.h>

#include "aslp-kws/parse-option.h"

#include "aslp-kws/symbol-table.h"
#include "aslp-kws/fsm.h"

int main(int argc, char *argv[]) {
    using namespace kaldi::kws;
    const char *usage = "Convert Fsm to dot format\n" 
                        "if in file not provided, read from stdin stream\n"
                        "Usage: fsm-to-dot [fsm_file]\n"
                        "eg: fsm-to-dot in.fsm\n";

    ParseOptions option(usage);
    
    std::string symbol_table_file = "";
    option.Register("symbol-table-file", &symbol_table_file, "");

    option.Read(argc, argv);

    int num_args = option.NumArgs();
    if (num_args != 0 && num_args != 1) {
        option.PrintUsage();
        exit(1);
    }

    Fsm fsm;
    if (num_args == 0) {
        fsm.Read("-");
    }
    else {
        std::string fsm_file = option.GetArg(1);
        fsm.Read(fsm_file.c_str());
    }

    SymbolTable *symbol_table = NULL;
    if (symbol_table_file != "") {
        symbol_table = new SymbolTable(symbol_table_file.c_str()); 
    }

    fsm.Dot(symbol_table);
    return 0;
}

