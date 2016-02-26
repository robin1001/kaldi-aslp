#!/bin/bash 

# Created on 2016-01-18
# Author: zhangbinbin

# Decode multiple models, not only the final or last model
. cmd.sh
. path.sh

gmmdir=exp/tri2b
feat_dir=data_fbank
dir=exp/dnn_fbank_sync100

for x in $dir/nnet/train.nnet_iter*; do
    echo "Decoding $x"
    iter=$(basename $x | awk -F'_' '{print $2}')
    if [ -d $dir/decode_test3000_$iter ]; then continue; fi

    aslp_scripts/aslp_nnet/decode.sh --nj 2 --num-threads 12 \
        --cmd "$decode_cmd" --acwt 0.0666667 \
        --nnet $x \
        $gmmdir/graph $feat_dir/test $dir/decode_test3000 || exit 1;
    aslp_scripts/score_basic.sh --cmd "$decode_cmd" $feat_dir/test \
        $gmmdir/graph $dir/decode_test3000 || exit 1;
done

exit 0

