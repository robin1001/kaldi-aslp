#include <stdio.h>

#include "aslp-kws/fsm.h"

int main(int argc, char *argv[]) {
    using namespace kaldi::kws;
    const char *usage = "Showing details text information of fsm format file," 
                        "if in file not provided, read from stdin stream\n"
                        "Usage: fsm-info [fsm_file]\n"
                        "eg: fsm-info in.fsm\n";

    if (argc < 1 || argc > 2) {
        printf("%s", usage);
        return -1;
    }

    Fsm fsm;
    if (argc == 1) {
        fsm.Read("-");
    }
    else {
        fsm.Read(argv[1]);
    }

    fsm.Info();
    return 0;
}

