#!/bin/bash

# Copyright 2016  ASLP (Author: zhangbinbin)
# Apache 2.0

# Pretrain nnet layer-wise

# Begin configuration.
learn_rate=0.01
train_tool="aslp-nnet-train-perutt"
minibatch_size=256
randomizer_size=32768
momentum=0
train_tool_opts=
iters_per_epoch=1

# End configuration.

echo "$0 $@"  # Print the command line for logging
[ -f path.sh ] && . ./path.sh; 
. parse_options.sh || exit 1;

set -euo pipefail

# Options
if [ $# != 6 ]; then
   echo "Usage: $0 <feat-train> <label-train> <hidden-layers> <cfsmn-layers> <insert-offset> <exp-dir>"
   echo "Options:"
   echo "--train-tool <nn-train-tool> # different train tool for (nn/lstm/ctc)"
   echo "--learn-rate <nn-learn-rate> # nn learnint rate for training"
   echo "--momentum <nn-momentum> # nn momentum for training"
   echo "--minibatch-size <nn-batch-size> # nn minibatch size for training"
   echo "--randomizer-size <nn-random-size> # nn random size for training"
   echo "--train_tool_opts <nn-train-opts> # other optional param for training"
   echo "--iters-per-epoch <iters-per-epoch> # iters per epoch for nnet pretrain"
   exit 1;
fi

feats_tr=$1
labels_tr=$2
num_hid=$3
num_cfsmn=$4
insert_offset=$5
dir=$6

[ ! -d $dir ] && mkdir $dir
[ ! -d $dir/log ] && mkdir $dir/log
[ ! -d $dir/nnet ] && mkdir $dir/nnet

[ -e $dir/nnet/pretrain.final.nnet ] && echo "'$dir/nnet/pretrain.final.nnet' exists, skipping training" && exit 0

# Check files
for f in $dir/nnet.proto $dir/hidden_dnn.conf $dir/cfsnm.conf; do
    [ ! -f $f ] && echo "$f: no such file" && exit 1;
done

# Delete previous files
#rm -f $dir/nnet/pretrain*

# Pretrain
for ((i=1; i<=$[num_hid+1]; i++)); do
	[ -e $dir/nnet/pretrain.$i.final.nnet ] && echo  pretrain.${i}.final.nnet exists, skipping... && mlp_final=$dir/nnet/pretrain.$i.final.nnet && continue
	mlp_init=$dir/nnet/pretrain.$i.init.nnet
    echo "Train with $i hidden layers, EPOCH: $i" 
    if [ $i -eq 1 ]; then
        aslp-nnet-init $dir/nnet.proto $mlp_init
    else
        [ ! -f $mlp_final ] && echo "$mlp_final: no such file" && exit 1;
        if [ $i -le $[num_cfsmn+1] ]; then
	    aslp-nnet-init --seed=0 $dir/cfsnm.conf - | \
            aslp-nnet-insert --insert_offset=$insert_offset --randomize-next-component=false $mlp_final - $mlp_init
	else
	    aslp-nnet-init --seed=0 $dir/hidden_dnn.conf - | \
	    aslp-nnet-insert --insert_offset=$insert_offset --randomize-next-component=false $mlp_final - $mlp_init
	fi
    fi
    mlp_final=$dir/nnet/pretrain.$i.final.nnet
	mlp_cur=$mlp_init
    for j in `seq $iters_per_epoch`; do
        echo "Train with $i hidden layers iters, ITERATION: $j" `date`
        echo "LOG (Just-for-log-analysis) ProgressLoss[last 0h of Nh]: 0 (Likelyhood) 0 (Xent)"
        mlp_next=$dir/nnet/pretrain.$i.$j.nnet
	    $train_tool $train_tool_opts \
                --momentum=$momentum \
                --learn-rate=$learn_rate \
                "$feats_tr" "$labels_tr" $mlp_cur $mlp_next
        mlp_cur=$mlp_next 
    done
    # Create symbol link make the last iter model as mlp_final model
    ln -s $(basename $mlp_cur) $mlp_final
done

f=$dir/nnet/pretrain.$[$num_hid+1].final.nnet
[ ! -f $f ] && echo "$f: no such file" && exit 1;
ln -s $(basename $f) $dir/nnet/pretrain.final.nnet

echo "Pretrain down!"
