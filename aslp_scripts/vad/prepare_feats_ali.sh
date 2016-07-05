#!/bin/bash

# Created on 2016-04-19
# Author: Binbin Zhang

# Prepare feature & alignment vad training
# Begin configuration.

# feature processing
splice=5            # (default) splice features both-ways along time axis,
delta_opts= #eg. "--delta-order=2" # (optional) adds 'add-deltas' to input feature pipeline, see opts,
splice_opts= #eg. "--left-context=4 --right-context=4"
cmvn_opts= #eg. "--norm-means=true --norm-vars=true" # (optional) adds 'apply-cmvn' to input feature pipeline, see opts,
global_cmvn_file=
sil_id=1 # Index of silence phone
labels=

# data processing, misc.
skip_cuda_check=false

# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh; 
. parse_options.sh || exit 1;

set -euo pipefail

if [ $# != 8 ]; then
   echo "Usage: $0 <data-train> <data-dev> <data-test> <lang-dir> <ali-train> <ali-dev> <ali-test> <exp-dir>"
   echo " e.g.: $0 data/train data/cv data/lang train_tr train_cv test exp/vad"
   echo ""
   echo " Training data : <data-train>,<ali-train> (for optimizing cross-entropy)"
   echo " Held-out data : <data-dev>,<ali-dev> (for learn-rate scheduling, model selection)"
   echo " note.: <ali-train>, <ali-dev>, <ali-test> can point to same directory, or 3 separate directories."
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>   # config containing options"
   echo "  --delta-opts <string>            # add 'add-deltas' to input feature pipeline"
   echo "  --splice <N>                     # splice +/-N frames of input features"
   echo "  --sil-id <int> # index for silence phone, default 1"
   echo ""
   exit 1;
fi

data=$1
data_cv=$2
data_test=$3
lang=$4
alidir=$5
alidir_cv=$6
alidir_test=$7
dir=$8

# Using alidir for supervision (default)
if [ -z "$labels" ]; then 
  for f in $alidir/final.mdl $alidir/ali.1.gz $alidir_cv/ali.1.gz; do
    [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
  done
fi

for f in $data/feats.scp $data_cv/feats.scp $data_test/feats.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

mkdir -p $dir/{log,nnet}

# check if CUDA compiled in and GPU is available,
#if ! $skip_cuda_check; then cuda-gpu-available || exit 1; fi

###### PREPARE ALIGNMENTS ######
echo
echo "# PREPARING ALIGNMENTS"

if [ ! -z "$labels" ]; then
  echo "Using targets '$labels' (by force)"
  labels_tr="$labels"
  labels_cv="$labels"
else
  echo "Using PDF targets from dirs '$alidir' '$alidir_cv'"
  # training targets in posterior format,
  labels_tr_pdf="ark:ali-to-phones --per-frame=true $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- | aslp-ali-to-sil --sil-id=${sil_id} ark:- ark:-|"
  labels_tr_gmm="ark:ali-to-phones --per-frame=true $alidir/final.mdl \\\"ark:gunzip -c $alidir/ali.*.gz |\\\" ark:- | aslp-ali-to-sil --sil-id=${sil_id} ark:- ark:-|"
  labels_tr="ark:ali-to-phones --per-frame=true $alidir/final.mdl \\\"ark:gunzip -c $alidir/ali.*.gz |\\\" ark:- | aslp-ali-to-sil --sil-id=${sil_id} ark:- ark:- | ali-to-post ark:- ark:- |"
  labels_cv_gmm=$labels_tr_gmm
  labels_test_gmm=$labels_tr_gmm
  labels_cv=$labels_tr
  labels_test=$labels_tr_gmm
  # get pdf-counts, used later for decoding/aligning,
  analyze-counts --verbose=1 --binary=false "$labels_tr_pdf" $dir/ali_train_pdf.counts 2>$dir/log/analyze_counts_pdf.log || exit 1
fi

###### PREPARE FEATURES ######
cp $data/feats.scp $dir/train_sorted.scp
cp $data_cv/feats.scp $dir/cv.scp
cp $data_test/feats.scp $dir/test.scp

# shuffle the list,
utils/shuffle_list.pl --srand ${seed:-777} <$dir/train_sorted.scp >$dir/train.scp

###### PREPARE FEATURE PIPELINE ######
# read the features,
feats_tr="ark:copy-feats scp:$dir/train.scp ark:- |"
feats_cv="ark:copy-feats scp:$dir/cv.scp ark:- |"
feats_test="ark:copy-feats scp:$dir/test.scp ark:- |"

if [ ! -z "$cmvn_opts" ]; then
  if [ ! -z $global_cmvn_file ]; then
    echo "# + 'apply-cmvn' with '$cmvn_opts' using statistics : $global_cmvn_file"
    feats_tr="$feats_tr apply-cmvn $cmvn_opts $global_cmvn_file ark:- ark:- |"
    feats_cv="$feats_cv apply-cmvn $cmvn_opts $global_cmvn_file ark:- ark:- |"
    feats_test="$feats_test apply-cmvn $cmvn_opts $global_cmvn_file ark:- ark:- |"
    echo "$global_cmvn_file" > $dir/global_cmvn_opts
  else
    echo "# + 'apply-cmvn-sliding' with '$cmvn_opts'"
    feats_tr="$feats_tr apply-cmvn-sliding $cmvn_opts ark:- ark:- |"
    feats_cv="$feats_cv apply-cmvn-sliding $cmvn_opts ark:- ark:- |"
    feats_test="$feats_test apply-cmvn-sliding $cmvn_opts ark:- ark:- |"
    echo "# + 'apply-cmvn-sliding' with '$cmvn_opts'"
  fi
fi

# optionally add deltas,
if [ ! -z "$delta_opts" ]; then
  feats_tr="$feats_tr add-deltas $delta_opts ark:- ark:- |"
  feats_cv="$feats_cv add-deltas $delta_opts ark:- ark:- |"
  feats_test="$feats_test add-deltas $delta_opts ark:- ark:- |"
  echo "# + 'add-deltas' with '$delta_opts'"
fi

# optionally add splice,
if [ ! -z "$splice_opts" ]; then
  feats_tr="$feats_tr splice-feats $splice_opts ark:- ark:- |"
  feats_cv="$feats_cv splice-feats $splice_opts ark:- ark:- |"
  feats_test="$feats_test splice-feats $splice_opts ark:- ark:- |"
  echo "# + 'splice-feats' with '$splice_opts'"
fi

# Keep track of the config,
[ ! -z "$cmvn_opts" ] && echo "$cmvn_opts" > $dir/cmvn_opts
[ ! -z "$delta_opts" ] && echo "$delta_opts" >$dir/delta_opts
[ ! -z "$splice_opts" ] && echo "$splice_opts" >$dir/splice_opts

# Save in the config file for nn training
train_config=$dir/train.conf
echo "feats_tr=\"$feats_tr\"" > $train_config
echo "feats_cv=\"$feats_cv\"" >> $train_config
echo "feats_test=\"$feats_test\"" >> $train_config
echo "labels_tr_gmm=\"$labels_tr_gmm\"" >> $train_config
echo "labels_cv_gmm=\"$labels_cv_gmm\"" >> $train_config
echo "labels_test_gmm=\"$labels_test_gmm\"" >> $train_config
echo "labels_tr=\"$labels_tr\"" >> $train_config
echo "labels_cv=\"$labels_cv\"" >> $train_config
echo "labels_test=\"$labels_test\"" >> $train_config

exit 0
