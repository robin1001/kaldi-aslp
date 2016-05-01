# ASLP Kaldi Code Trunk

## Feature
* nnet enhancement
    * BatchNormalizaiton
    * Standard Lstm & BLstm, without projection
    * Latency control BLstm
    * MPI base 2-gpu speedup
    * CTC in two ways(Warp-CTC vs Eesen)
    * Skip training & decode
    * Graph network(eg multi-input, multi-output, add and splice) 
* vad 

## TODO
* kws
* GRU 
* TTS align script
* MDN for TTS

## Install
Kaldi-aslp rely on the standard kaldi recipe. So you should first install specific kaldi version, 
Download it from [here](http://wiki.npu-aslp.org/aslpdata/groupsfile/1-ASR/kaldi-aslp/kaldi-master.zip), 
You can also Download kaldi depending tools from [here](http://wiki.npu-aslp.org/aslpdata/groupsfile/1-ASR/kaldi-aslp/kaldi-tools.tar.gz). 
And clone this trunk to your workspace, and change to the src dir then modify the aslp.mk, set the KALDI_DIR to your kaldi path.
Finally, just type make. It's very easy to build, right?
``` bash
cd src
vi aslp.mk ## modify KALDI_DIR = your_kaldi_path/src
make -j 12
``` 
To use it in your project, just add kaldi-aslp bin dir to your path.sh
``` bash
export KALDI_ROOT=`pwd`/../../../
export ASLP_ROOT=/home/disk1/zhangbinbin/kaldi-aslp
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh 
export PATH=$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin/:$KALDI_ROOT/src/kwsbin:$KALDI_ROOT/src/online2bin/:$KALDI_ROOT/src/ivectorbin/:$KALDI_ROOT/src/lmbin/:$KALDI_ROOT/src/nnet3bin/:$PWD:$ASLP_ROOT/src/aslp-nnetbin/:$ASLP_ROOT/src/aslp-bin/:$ASLP_ROOT/src/aslp-vad/:$PATH
export LC_ALL=C
```

## How to Develop & Contribution
Anybody, any contribution is highly appriciated.

## What's New
* 2016-04-30 VAD support(variable vad models, energy based, gmm based, dnn and lstm based)
* 2016-04-15 CTC support. Both Eesen and Warp-CTC(Baidu AI-Lab) is intergrated.
* 2016-03-28 Skip training & decode support.
* 2016-03-20 Phone CD-Phone prepare and training support.
* 2016-03-20 Standard Lstm BLstm support.
* 2016-03-07 Graph nnet support.
* 2016-02-20 MPI based 2-gpu speed up.
* 2016-02-20 Batch Normalizaiton support.
* 2016-01-15 Init trunk(DNN, CNN prepare and training script)

