#!/bin/bash

# Created on 2016-01-15
# Author: zhangbinbin liwenpeng

feat_type=fbank #fmllr
gmmdir=exp/tri3 #for fmllr feature
cv_utt_percent=10 # default 10% of total utterances 
split_cv=false
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
. ./path.sh ## Source the tools/utils (import the queue.pl)

. utils/parse_options.sh || exit 1;


if [ $# != 2 ]; then
   echo "Usage: $0  <data-dir> <feat-dir>"
   echo " e.g.: $0 data data-fbank"
   echo "Options:"
   echo "  --feat-type (fbank|fmllr) # type of input features"
   echo "  --gmm-dir # gmm dir, for fmllr feature extraction"
   echo "  --split-cv # whether or not split cross validation, define is true"
   echo "  --cv-utt-percent # percent of utt for cross validation"
   exit 1
fi

data_dir=$1
feat_dir=$2

echo "# feature type : $feat_type"
case $feat_type in
    fbank)
        # Extracting fbank feats
	dir=$feat_dir/$(basename $data_dir)
        utils/copy_data_dir.sh $data_dir $dir || exit 1; rm $dir/{cmvn,feats}.scp
        steps/make_fbank.sh --nj 10 --cmd "$train_cmd -tc 10" \
        $dir $dir/log $dir/feat || exit 1;
        steps/compute_cmvn_stats.sh $dir $dir/log $dir/feat || exit 1;
        if $split_cv; then
           utils/subset_data_dir_tr_cv.sh --cv-utt-percent $cv_utt_percent $dir \
		${dir}_tr$[100-$cv_utt_percent] ${dir}_cv${cv_utt_percent}
        fi
        ;;
    fmllr)
        # Extracting fmllr feats
	[ -z $gmmdir ] && echo "gmmdir is empty" && exit 1;
	dir=$feat_dir/$(basename $data_dir)
        steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
            --transform-dir ${gmmdir}_ali \
            $dir $data_dir $gmmdir $dir/log $dir/feat || exit 1
        if $split_cv; then
           utils/subset_data_dir_tr_cv.sh --cv-utt-percent $cv_utt_percent $dir \
		${dir}_tr$[100-$cv_utt_percent] ${dir}_cv${cv_utt_percent} || exit 1
        fi
        ;;
    *)
        echo "Unknown feature type $feat_type"
        exit 1;
        ;;
esac

echo "Done!!!"
