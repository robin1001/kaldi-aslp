#!/bin/bash

# Created on 2016-01-19
# Author: zhangbinbin liwenpeng duwei

feat_type=fbank #fmllr
gmmdir=exp/tri3 #for fmllr feature
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
. ./path.sh ## Source the tools/utils (import the queue.pl)

. utils/parse_options.sh || exit 1;


if [ $# != 2 ]; then
   echo "Usage: $0  <data-dir> <feat-dir>"
   echo " e.g.: $0 data data-fbank"
   echo "Options:"
   echo "  --feat-type (fbank|fmllr) # type of input features"
   echo "  --gmm-dir # gmm dir, for fmllr feature extraction"
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
        ;;
    fmllr)
        # Extracting fmllr feats
	    [ -z $gmmdir ] && echo "gmmdir is empty" && exit 1;
	    dir=$feat_dir/$(basename $data_dir)
        steps/nnet/make_fmllr_feats.sh --nj 10 --cmd "$train_cmd" \
            --transform-dir ${gmmdir}_ali \
            $dir $data_dir $gmmdir $dir/log $dir/feat || exit 1
        ;;
    *)
        echo "Unknown feature type $feat_type"
        exit 1;
        ;;
esac

echo "Done!!!"
