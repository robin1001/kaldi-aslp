#!/bin/bash 

# Created on 2016-01-18
# Author: zhangbinbin

# Decode multiple models, not only the final or last model
. cmd.sh
. path.sh

dir=exp/dnn_fmllr_with_pretrain2
for x in $dir/nnet/train.nnet_iter*; do
    echo "Decoding $x"
    iter=$(basename $x | awk -F'_' '{print $2}')
    aslp_scripts/aslp_nnet/decode.sh --nj 20 --cmd run.pl --acwt 0.2 \
        --nnet $x \
        exp/tri3/graph data-fmllr-tri3/test \
        $dir/decode_test_$iter
done

exit 0

