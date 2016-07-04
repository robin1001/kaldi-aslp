#!/bin/bash

# Created on 2016-04-19
# Author: Binbin Zhang

stage=4
cmd=run.pl
nj=4
num_cv_utt=2000
num_test_utt=2000
feat_dir=mfcc
gmmdir=exp/tri2b
ali=${gmmdir}_ali
dir=exp/gmm_vad
vad_dir=vad_data

echo "$0 $@"  # Print the command line for logging
[ -f cmd.sh ] && . ./cmd.sh;
[ -f path.sh ] && . ./path.sh; 
. parse_options.sh || exit 1;

set -euo pipefail

# Prepare feat feat_type: mfcc
if [ $stage -le 0 ]; then 
    echo "Extracting feats & Create tr cv set"
    #aslp_scripts/make_feats.sh --feat-type "mfcc" data/train $feat_dir
    # Split tr & cv
    utils/shuffle_list.pl $feat_dir/train/feats.scp > $feat_dir/train/random.scp
    cat $feat_dir/train/random.scp | awk -v ed=$num_test_utt '{if(NR <= ed) print $0}' |\
        sort > $feat_dir/train/test.scp
    cat $feat_dir/train/random.scp | awk -v st=$num_test_utt -v ed=$[$num_test_utt+$num_cv_utt] '{if (NR > st && NR <= ed) print $0}' |\
        sort > $feat_dir/train/cv.scp
    cat $feat_dir/train/random.scp | awk -v st=$[$num_test_utt+$num_cv_utt] '{if (NR > st ) print $0}' |\
        sort > $feat_dir/train/tr.scp
    utils/subset_data_dir.sh --utt-list $feat_dir/train/test.scp \
        $feat_dir/train $feat_dir/test
    utils/subset_data_dir.sh --utt-list $feat_dir/train/tr.scp \
        $feat_dir/train $feat_dir/train_tr
    utils/subset_data_dir.sh --utt-list $feat_dir/train/cv.scp \
        $feat_dir/train $feat_dir/train_cv
fi

# Prepare ali
if [ $stage -le 1 ]; then 
    #--splice_opts "--left-context=5 --right-context=5" \
    aslp_scripts/vad/prepare_feats_ali.sh \
        --cmvn_opts "--norm-vars=false --center=true --cmn-window=300" \
        --delta_opts "--delta-order=2" \
        --sil-id 1 \
        $feat_dir/train_tr $feat_dir/train_cv $feat_dir/test data/lang $ali $ali $ali $dir || exit 1;
fi

# Get feats_tr feats_cv labels_tr labels_cv 
[ ! -f $dir/train.conf ] && \
    echo "$dir/train.conf(config file for nn training): no such file" && exit 1 
source $dir/train.conf

# Generate sil and none sil data for gmm train
if [ $stage -le 2 ]; then
    [ ! -d $vad_dir ] && mkdir -p $vad_dir
    aslp-select-frames --select-id=0 "$feats_tr" "$labels_tr_gmm" "ark:|gzip -c > $vad_dir/sil.ark.gz"
    aslp-select-frames --select-id=1 "$feats_tr" "$labels_tr_gmm" "ark:|gzip -c > $vad_dir/voice.ark.gz"
fi

# Train Speech and Non-Speech GMM
if [ $stage -le 3 ]; then
    aslp_scripts/vad/train_diag_gmm.sh \
        --mdl-prefix "sil" --num-gauss 1024 \
        "ark:gunzip -c $vad_dir/sil.ark.gz |" $dir
    aslp_scripts/vad/train_diag_gmm.sh \
        --mdl-prefix "voice" --num-gauss 1024 \
        "ark:gunzip -c $vad_dir/voice.ark.gz |" $dir
fi

# Test
if [ $stage -le 4 ]; then
    [ ! -e $dir/sil.final.dubm ] && echo "$dir/sil.final.dubm: no such file" && exit 1;
    [ ! -e $dir/voice.final.dubm ] && echo "$dir/voice.final.dubm: no such file" && exit 1;
    aslp-eval-gmm-vad $dir/sil.final.dubm $dir/voice.final.dubm \
        "$feats_test" "$labels_test_gmm"
fi


