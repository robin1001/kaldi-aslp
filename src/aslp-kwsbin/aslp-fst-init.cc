#include <stdio.h>

#include "base/kaldi-common.h"
#include "util/common-utils.h"

#include "aslp-kws/fst.h"

int main(int argc, char *argv[]) {
    using namespace kaldi;
    using namespace kaldi::kws;
    const char *usage = "Init fst from topo file, just like the way openfst compile"
                        "Usage: aslp-fst-init topo_file out_file\n"
                        "eg: aslp-fst-init topo_file out.fst\n";
    
    ParseOptions po(usage);
    std::string isymbols = "";
    po.Register("isymbols", &isymbols, "input symbol file"); 
    std::string osymbols = "";
    po.Register("osymbols", &osymbols, "output symbol file"); 
    
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
        po.PrintUsage();
        exit(1);
    }

    std::string topo_file = po.GetArg(1),
                out_file = po.GetArg(2);
    
    SymbolTable isymbol_table(isymbols), osymbol_table(osymbols);
    
    Fst fst;

    fst.ReadTopo(isymbol_table, osymbol_table, topo_file);
    fst.Write(out_file);
    
    return 0;
}

