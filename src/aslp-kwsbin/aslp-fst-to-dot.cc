#include <stdio.h>

#include "base/kaldi-common.h"
#include "util/common-utils.h"

#include "aslp-kws/symbol-table.h"
#include "aslp-kws/fst.h"

int main(int argc, char *argv[]) {
    using namespace kaldi;
    using namespace kaldi::kws;
    const char *usage = "Convert Fst to dot format\n" 
                        "Usage: aslp-fst-to-dot fsm_file\n"
                        "eg: aslp-fst-to-dot in.fsm\n";

    ParseOptions po(usage);
    std::string isymbols = "";
    po.Register("isymbols", &isymbols, "input symbol file"); 
    std::string osymbols = "";
    po.Register("osymbols", &osymbols, "output symbol file"); 

    po.Read(argc, argv);

    if (po.NumArgs() != 1) {
        po.PrintUsage();
        exit(1);
    }
    
    std::string fst_file = po.GetArg(1);

    SymbolTable isymbol_table(isymbols), osymbol_table(osymbols);
    Fst fst(fst_file);
    fst.Dot(isymbol_table, osymbol_table);
    return 0;
}

