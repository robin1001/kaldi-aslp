#!/bin/bash

# Created on 2016-11-30

stage=1

# test raw wav scp
wav_test="scp:data/test/wav.scp"

dir=exp/energy_vad

echo "$0 $@"  # Print the command line for logging
[ -f cmd.sh ] && . ./cmd.sh;
[ -f path.sh ] && . ./path.sh; 
. parse_options.sh || exit 1;

set -euo pipefail

sil_id=1
alidir=exp/tri2b_ali

labels_test="ark:ali-to-phones --per-frame=true $alidir/final.mdl \\\"ark:gunzip -c $alidir/ali.*.gz |\\\" ark:- | aslp-ali-to-sil --sil-id=${sil_id} ark:- ark:-|"

# Test
if [ $stage -le 1 ]; then
    aslp-eval-energy-vad --stride=0.0001 "$wav_test" "$labels_test" > $dir/log/roc.log
    aslp_scripts/vad/calc_auc.sh --stride 0.001 $dir/log/roc.log | tee $dir/log/auc.log
    aslp_scripts/calc_eer.sh $dir/log/roc.log | tee $dir/log/eer.log
fi

exit 0;

