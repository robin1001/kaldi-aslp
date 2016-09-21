#!/bin/bash

# Copyright 2016 ASLP (author: zhangbinbin)
# Apache 2.0

# Prepare feature & alignment for dnn training
# Begin configuration.

# feature processing,
splice=5            # (default) splice features both-ways along time axis,
cmvn_opts= #eg. "--norm-means=true --norm-vars=true" # (optional) adds 'apply-cmvn' to input feature pipeline, see opts,
delta_opts= #eg. "--delta-order=2" # (optional) adds 'add-deltas' to input feature pipeline, see opts,
splice_opts= #eg. "--left-context=4 --right-context=4"
global_cmvn_file=

labels=

# data processing, misc.
copy_feats=false # resave the train/cv features into /tmp (disabled by default),
copy_feats_tmproot=/tmp/kaldi.XXXX # sets tmproot for 'copy-feats',
seed=777            # seed value used for data-shuffling, nn-initialization, and training,
skip_cuda_check=false

# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh; 
. parse_options.sh || exit 1;

set -euo pipefail

if [ $# != 6 ]; then
   echo "Usage: $0 <data-train> <data-dev> <lang-dir> <ali-train> <ali-dev> <exp-dir>"
   echo " e.g.: $0 data/train data/cv data/lang exp/mono_ali_train exp/mono_ali_cv exp/mono_nnet"
   echo ""
   echo " Training data : <data-train>,<ali-train> (for optimizing cross-entropy)"
   echo " Held-out data : <data-dev>,<ali-dev> (for learn-rate scheduling, model selection)"
   echo " note.: <ali-train>,<ali-dev> can point to same directory, or 2 separate directories."
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>   # config containing options"
   echo "  --cmvn-opts  <string>            # add 'apply-cmvn' to input feature pipeline"
   echo "  --delta-opts <string>            # add 'add-deltas' to input feature pipeline"
   echo "  --splice <N>                     # splice +/-N frames of input features"
   echo ""
   exit 1;
fi

data=$1
data_cv=$2
lang=$3
alidir=$4
alidir_cv=$5
dir=$6

# Using alidir for supervision (default)
if [ -z "$labels" ]; then 
  silphonelist=`cat $lang/phones/silence.csl` || exit 1;
  for f in $alidir/final.mdl $alidir/ali.1.gz $alidir_cv/ali.1.gz; do
    [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
  done
fi

for f in $data/feats.scp $data_cv/feats.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

mkdir -p $dir/{log,nnet}

# check if CUDA compiled in and GPU is available,
if ! $skip_cuda_check; then cuda-gpu-available || exit 1; fi

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
  labels_tr="ark:ali-to-pdf $alidir/final.mdl \\\"ark:gunzip -c $alidir/ali.*.gz |\\\" ark:- | ali-to-post ark:- ark:- |"
  labels_cv="ark:ali-to-pdf $alidir/final.mdl \\\"ark:gunzip -c $alidir_cv/ali.*.gz |\\\" ark:- | ali-to-post ark:- ark:- |"
  # training targets for analyze-counts,
  labels_tr_pdf="ark:ali-to-pdf $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- |"
  labels_tr_phn="ark:ali-to-phones --per-frame=true $alidir/final.mdl \"ark:gunzip -c $alidir/ali.*.gz |\" ark:- |"

  # get pdf-counts, used later for decoding/aligning,
  analyze-counts --verbose=1 --binary=false "$labels_tr_pdf" $dir/ali_train_pdf.counts 2>$dir/log/analyze_counts_pdf.log || exit 1
  # copy the old transition model, will be needed by decoder,
  copy-transition-model --binary=false $alidir/final.mdl $dir/final.mdl || exit 1
  # copy the tree
  cp $alidir/tree $dir/tree || exit 1

  # make phone counts for analysis,
  [ -e $lang/phones.txt ] && analyze-counts --verbose=1 --symbol-table=$lang/phones.txt "$labels_tr_phn" /dev/null 2>$dir/log/analyze_counts_phones.log || exit 1
fi

###### PREPARE FEATURES ######
echo
echo "# PREPARING FEATURES"
if [ "$copy_feats" == "true" ]; then
  echo "# re-saving features to local disk,"
  tmpdir=$(mktemp -d $copy_feats_tmproot)
  copy-feats scp:$data/feats.scp ark,scp:$tmpdir/train.ark,$dir/train_sorted.scp || exit 1
  copy-feats scp:$data_cv/feats.scp ark,scp:$tmpdir/cv.ark,$dir/cv.scp || exit 1
  trap "echo \"# Removing features tmpdir $tmpdir @ $(hostname)\"; ls $tmpdir; rm -r $tmpdir" EXIT
else
  # or copy the list,
  cp $data/feats.scp $dir/train_sorted.scp
  cp $data_cv/feats.scp $dir/cv.scp
fi
# shuffle the list,
utils/shuffle_list.pl --srand ${seed:-777} <$dir/train_sorted.scp >$dir/train.scp

# create a 10k utt subset for global cmvn estimates,
head -n 10000 $dir/train.scp > $dir/train.scp.10k

# for debugging, add lists with non-local features,
utils/shuffle_list.pl --srand ${seed:-777} <$data/feats.scp >$dir/train.scp_non_local
cp $data_cv/feats.scp $dir/cv.scp_non_local

###### PREPARE FEATURE PIPELINE ######
# read the features,
feats_tr="ark:copy-feats scp:$dir/train.scp ark:- |"
feats_cv="ark:copy-feats scp:$dir/cv.scp ark:- |"

# optionally add per-speaker CMVN,
if [ ! -z "$cmvn_opts" ]; then
  if [ ! -z $global_cmvn_file ]; then
    [ -z $global_cmvn_file ] && echo "Using global cmvn config, but missing global cmvn file"
    echo "# + 'apply-cmvn' with '$cmvn_opts' using statistics : $global_cmvn_file"
    feats_tr="$feats_tr apply-cmvn $cmvn_opts $global_cmvn_file ark:- ark:- |"
    feats_cv="$feats_cv apply-cmvn $cmvn_opts $global_cmvn_file ark:- ark:- |"
    echo "$global_cmvn_file" > $dir/global_cmvn_opts
  else
    echo "# + 'apply-cmvn' with '$cmvn_opts' using statistics : $data/cmvn.scp, $data_cv/cmvn.scp"
    [ ! -r $data/cmvn.scp ] && echo "Missing $data/cmvn.scp" && exit 1;
    [ ! -r $data_cv/cmvn.scp ] && echo "Missing $data_cv/cmvn.scp" && exit 1;
    feats_tr="$feats_tr apply-cmvn $cmvn_opts --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp ark:- ark:- |"
    feats_cv="$feats_cv apply-cmvn $cmvn_opts --utt2spk=ark:$data_cv/utt2spk scp:$data_cv/cmvn.scp ark:- ark:- |"
  fi
else
  echo "# 'apply-cmvn' is not used,"
fi

# optionally add deltas,
if [ ! -z "$delta_opts" ]; then
  feats_tr="$feats_tr add-deltas $delta_opts ark:- ark:- |"
  feats_cv="$feats_cv add-deltas $delta_opts ark:- ark:- |"
  echo "# + 'add-deltas' with '$delta_opts'"
fi

# optionally add splice,
if [ ! -z "$splice_opts" ]; then
  feats_tr="$feats_tr splice-feats $splice_opts ark:- ark:- |"
  feats_cv="$feats_cv splice-feats $splice_opts ark:- ark:- |"
fi

# Keep track of the config,
[ ! -z "$cmvn_opts" ] && echo "$cmvn_opts" >$dir/cmvn_opts 
[ ! -z "$delta_opts" ] && echo "$delta_opts" >$dir/delta_opts
[ ! -z "$splice_opts" ] && echo "$splice_opts" >$dir/splice_opts

# Save in the config file for nn training
train_config=$dir/train.conf
echo "feats_tr=\"$feats_tr\"" > $train_config
echo "feats_cv=\"$feats_cv\"" >> $train_config
echo "labels_tr=\"$labels_tr\"" >> $train_config
echo "labels_cv=\"$labels_cv\"" >> $train_config

exit 0
