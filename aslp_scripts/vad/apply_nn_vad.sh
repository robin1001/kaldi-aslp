#!/bin/bash

# Created on 2016-04-28
# Author: Binbin Zhang

delta_opts="--delta-order=2" #eg. "--delta-order=2" # (optional) adds 'add-deltas' to input feature pipeline, see opts,
#splice_opts="--left-context=5 --right-context=5" #eg. "--left-context=4 --right-context=4"
splice_opts=
#eg. "--norm-means=true --norm-vars=true" # (optional) adds 'apply-cmvn' to input feature pipeline, see opts,
cmvn_opts="--norm-vars=false --center=true --cmn-window=300" 
feat_tool=compute-mfcc-feats
feat_config=conf/mfcc.conf
cmd=run.pl
nj=12

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh; 
. parse_options.sh || exit 1;

set -euo pipefail

if [ ! $# -eq 3 ];then
    echo "Apply nn vad for wav files"    
    echo "Usage $0 nnet.mdl in_scp out_scp"
    echo "eg: $0 exp/dnn_vad/final.nnet scp/raw_wav.scp scp/vad_wav.scp"
    exit 1;
fi

nnet=$1
raw_wav_scp=$2
vad_wav_scp=$3

# Compute feat
feats="ark:$feat_tool --verbose=2 --config=$feat_config scp:$raw_wav_scp.JOB ark:- |"

if [ ! -z "$delta_opts" ]; then
    feats="$feats add-deltas $delta_opts ark:- ark:- |"
fi

if [ ! -z "$cmvn_opts" ]; then
    feats="$feats apply-cmvn-sliding $cmvn_opts ark:- ark:- |"
fi

if [ ! -z "$splice_opts" ]; then
  feats="$feats splice-feats $splice_opts ark:- ark:- |"
fi

echo "Make dir"
# Make vad wav dir
while read line; do
    vad_file_name=$(echo $line | awk '{print $2}')
    dir=$(dirname $vad_file_name)
    [ ! -d $dir ] && mkdir -p $dir
done < $vad_wav_scp

echo "Split"
raw_wav_split_scps=""
vad_wav_split_scps=""
for n in $(seq $nj); do
    raw_wav_split_scps="$raw_wav_split_scps $raw_wav_scp.$n"
    vad_wav_split_scps="$vad_wav_split_scps $vad_wav_scp.$n"
done
utils/split_scp.pl $raw_wav_scp $raw_wav_split_scps || exit 1;
utils/split_scp.pl $vad_wav_scp $vad_wav_split_scps || exit 1;

echo "Do vad"
$cmd JOB=1:$nj $raw_wav_scp.JOB.log \
    aslp-apply-nn-vad --frame-length=10 \
        --sample-frequency=16000 \
        --sil-thresh=0.261010 \
        --silence-trigger-threshold=80 \
        $nnet "$feats" "scp:$raw_wav_scp.JOB" "ark:$vad_wav_scp.JOB"

