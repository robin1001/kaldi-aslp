# Aslp Train Parallelization

Here we implement serveral kinds of train parallelization, in both synchronize
and asynchronize ways. 


## Deprecated
nnet-mpi-sync.h nnet-mpi-sync.cc are deprecated, which implemented 2-gpu parallization.

## Bulk Synchronize Parallel(BSP Model Average)
Refer "Theano-MPI: a Theano-based Distributed Training Framework" for details
We regard BSP as SGD method.

## Elastic Averaging SGD
Refer "Deep learning with Elastic Averaging SGD" for details.
We only use the asynchronize Elastic Averaging SGD in our implementation
for it's simplity, and regard it as ASGD mothod.


# BlockwiseModel-Update Filtering (BMUF)
Refer "SCALABLE TRAINING OF DEEP LEARNING MACHINES BY INCREMENTAL BLOCK TRAINING WITH INTRA-BLOCK PARALLEL OPTIMIZATION AND BLOCKWISE MODEL-UPDATE FILTERING"


