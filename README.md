# ASLP Kaldi Code Trunk

## Feature
* nnet enhancement
    * BatchNormalizaiton
    * Standard Lstm & BLstm, without projection
    * Latency control BLstm
    * CTC in two ways(Warp-CTC vs Eesen)
    * Skip training & decode
    * Graph network(eg multi-input, multi-output, add and splice) 
    * Row Convolution
    * GRU
    * FSMN
* vad 
* online
* parallel(MPI based multi-card speedup)
    * BSP
    * ASGD
    * EASGD
    * BMUF
* kws

## Install
kaldi-aslp compile just like the standard kaldi recipe.

First, compile kaldi dependent tools
if the network is not avaliable, You can download kaldi depending tools from [here](http://wiki.npu-aslp.org/aslpdata/groupsfile/1-ASR/kaldi-aslp/kaldi-tools.tar.gz). 

``` bash
cd tools
make
```

Then, go to src dir, modify aslp.mk, change the compile switch as you like 

```bash
# Optional make warp-ctc
USE_CTC = true
USE_WARP_CTC = false

# Optinal make mpi parallel(aslp-paralle aslp-parallelbin)
USE_MPI = true

# Optinal make online part, Online depends on crf++, add it's lib and include
USE_ONLINE = false

CRF_ROOT = /home/zhangbinbin/zhangbinbin/kaldi-aslp/src/crf
CRF_FLAGS = /home/zhangbinbin/zhangbinbin/kaldi-aslp/src/crf/libcrfpp.a
```
Last, just configure and compile.

```bash
cd src
./configure
make -j 24
```

## How to Use?

To use it, just copy aslp_scripts to your project, add kaldi-aslp bin dir to your path.sh

``` bash
export KALDI_ROOT=`pwd`/../../../
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh 
export PATH=$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin/:$KALDI_ROOT/src/kwsbin:$KALDI_ROOT/src/online2bin/:$KALDI_ROOT/src/ivectorbin/:$KALDI_ROOT/src/lmbin/:$KALDI_ROOT/src/nnet3bin/:$PWD:$KALDI_ROOT/src/aslp-nnetbin/:$KALDI_ROOT/src/aslp-bin/:$KALDI_ROOT/src/aslp-vadbin/:$KALDI_ROOT/src/aslp-parallelbin:$KALDI_ROOT/src/aslp-onlinebin:$PATH
export LC_ALL=C
```
## How to Develop & Contribution
Anybody, any contribution is highly appriciated.

## What's New
* 2017-03-20 SOD parallel surpport, adagrad, adadelta, rmpsprop, adam optimizer
* 2016-10-06 MPI based Parallel support
* 2016-07-10 More VAD support, ROC, boundary accuracy, and Praat format tools
* 2016-06-06 Online support
* 2016-04-30 VAD support(variable vad models, energy based, gmm based, dnn and lstm based)
* 2016-04-15 CTC support. Both Eesen and Warp-CTC(Baidu AI-Lab) is intergrated.
* 2016-03-28 Skip training & decode support.
* 2016-03-20 Phone CD-Phone prepare and training support.
* 2016-03-20 Standard Lstm BLstm support.
* 2016-03-07 Graph nnet support.
* 2016-02-20 MPI based 2-gpu speed up.
* 2016-02-20 Batch Normalizaiton support.
* 2016-01-15 Init trunk(DNN, CNN prepare and training script)
