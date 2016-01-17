#!/bin/bash

# Created on 2016-01-15
# Author: zhangbinbin

feat_type=fbank #fmllr
gmmdir=exp/tri3 #for fmllr feature
cv_utt_percent=10 # default 10% of total utterances 

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
. ./path.sh ## Source the tools/utils (import the queue.pl)

. utils/parse_options.sh || exit 1;


if [ $# != 2 ]; then
   echo "Usage: $0  <data-dir> <feat-dir>"
   echo " e.g.: $0 data data-fbank"
   echo "Options:"
   echo "  --feat-type (fbank|fmllr) # type of input features"
   echo "  --gmm-dir # gmm dir, for fmllr feature extraction"
   echo "  --cv-utt-percent # percent of utt for cross validation"
   exit 1
fi

data_dir=$1
feat_dir=$2

echo "# feature type : $feat_type"
case $feat_type in
    fbank)
        # Test set
        dir=$feat_dir/test
        utils/copy_data_dir.sh $data_dir/test $dir || exit 1; rm $dir/{cmvn,feats}.scp
        steps/make_fbank.sh --nj 10 --cmd "$train_cmd" \
            $dir $dir/log $dir/data || exit 1;
        steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
        # Training set
        dir=$feat_dir/train
        utils/copy_data_dir.sh $data_dir/train $dir || exit 1; rm $dir/{cmvn,feats}.scp
        steps/make_fbank.sh --nj 10 --cmd "$train_cmd -tc 10" \
        $dir $dir/log $dir/data || exit 1;
        steps/compute_cmvn_stats.sh $dir $dir/log $dir/data || exit 1;
        utils/subset_data_dir_tr_cv.sh --cv-utt-percent $cv_utt_percent $dir ${dir}_tr90 ${dir}_cv10
        ;;
    fmllr)
        # Test
        dir=$feat_dir/test
        steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
            --transform-dir $gmmdir/decode_test \
            $dir $data_dir/test $gmmdir $dir/log $dir/data || exit 1
        # Train
        dir=$feat_dir/train
        steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
            --transform-dir ${gmmdir}_ali \
            $dir $data_dir/train $gmmdir $dir/log $dir/data || exit 1
        utils/subset_data_dir_tr_cv.sh --cv-utt-percent $cv_utt_percent $dir ${dir}_tr90 ${dir}_cv10 || exit 1
        ;;
    *)
        echo "Unknown feature type $feat_type"
        exit 1;
        ;;
esac

echo "Done!!!"
