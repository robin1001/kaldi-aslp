/* Created on 2016-07-27
 * Author: Zhang Binbin
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "mpi-node.h"


int main(int argc, char *argv[]) {
    using namespace kaldi;
    MpiNode mpi_node;
    srand(time(NULL));
    //int epoch = rand() % 5;
    int epoch = mpi_node.Rank() + 3;
    printf("rank %d epoch %d\n", mpi_node.Rank(), epoch);
    for (int i = 0; i < epoch + 1; i++) {
        int n = 1; 
        mpi_node.AllReduce(&n, 1);
        printf("rank %d epoch %d sum %d\n", mpi_node.Rank(), i, n);
    }

    while (true) {
       int sum = 0; 
       mpi_node.AllReduce(&sum, 1);
       printf("rank %d sum %d\n", mpi_node.Rank(), sum);
       if (sum == 0) break;
    }
    mpi_node.Barrier();
}

