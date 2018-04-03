#include <stdio.h>

#include "aslp-kws/fst.h"

int main(int argc, char *argv[]) {
    using namespace kaldi::kws;
    const char *usage = "Showing details text information of fsm format file\n" 
                        "Usage: aslp-fst-info fst-file]\n"
                        "eg: aslp-fst-info in.fst\n";

    if (argc != 2) {
        printf("%s", usage);
        return -1;
    }

    Fst fst(argv[1]);

    fst.Info();
    return 0;
}

