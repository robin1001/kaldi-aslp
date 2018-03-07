#include <stdio.h>

#include "aslp-kws/fsm.h"

int main(int argc, char *argv[]) {
    using namespace kaldi::kws;
    const char *usage = "Init fsm from topo file "
                        "if out_file is not provided, copy to the stdout stream\n"
                        "Usage: fsm-init topo_file [out_file]\n"
                        "eg: fsm-copy topo_file out.fsm\n";
    if (argc < 2 && argc > 3) {
        printf("%s", usage);
        return -1;
    }

    Fsm fsm;
    fsm.ReadTopo(argv[1]);
    if (argc == 2) {
        fsm.Write("-");
    }
    else {
        fsm.Write(argv[2]);
    }
    return 0;
}

