// Copyright 2016 ASLP (Author: zhangbinbin)

/* Created on 2016-02-21
 * Author: zhangbinbin
 */

#include "aslp-nnet/nnet-mpi-sync.h"


int main(int argc, char *argv[]) {
    using namespace kaldi;
    using namespace kaldi::aslp_nnet;
    NnetMpiSync mpi_sync;
    for (int i = 0; i < 10; i++) {
        mpi_sync.SyncTest();
        //sleep(1);
    }
}
