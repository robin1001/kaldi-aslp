#!/bin/bash

# Created on 2016-04-21
# Author: Binbin Zhang

mdl_prefix=sil
nj=10
cmd=run.pl
num_iters=4
stage=-2
num_gselect=30 # Number of Gaussian-selection indices to use while training
num_frames=500000 # number of frames to keep in memory for initialization
num_iters_init=20
initial_gauss_proportion=0.5 # Start with half the target number of Gaussians
subsample=5 # subsample all features with this periodicity, in the main E-M phase.
cleanup=true
min_gaussian_weight=0.0001
remove_low_count_gaussians=true # set this to false if you need #gauss to stay fixed.
num_threads=12
parallel_opts="-pe smp 32"
num_gauss=1024

echo "$feats_tr"
echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh; 
. parse_options.sh || exit 1;

set -euo pipefail

if [ $# != 2 ]; then
    echo "Usage: $0 <feat> <output-dir>"
    echo " e.g.: $0 scp:feat.scp exp/diag_ubm"
    echo "Options: "
    exit 1;
fi

feats=$1
dir=$2

num_gauss_init=$(perl -e "print int($initial_gauss_proportion * $num_gauss); ");
! [ $num_gauss -gt 0 ] && echo "Bad num-gauss $num_gauss" && exit 1;

if [ $stage -le -2 ]; then
    $cmd $parallel_opts $dir/log/${mdl_prefix}.gmm_init.log \
        gmm-global-init-from-feats --num-threads=$num_threads --num-frames=$num_frames \
         --min-gaussian-weight=$min_gaussian_weight \
         --num-gauss=$num_gauss --num-gauss-init=$num_gauss_init --num-iters=$num_iters_init \
         "$feats" $dir/${mdl_prefix}.0.dubm || exit 1;
fi

# Store Gaussian selection indices on disk-- this speeds up the training passes.
if [ $stage -le -1 ]; then
    echo Getting Gaussian-selection info
    $cmd JOB=1:$nj $dir/log/${mdl_prefix}.gselect.JOB.log \
        gmm-gselect --n=$num_gselect $dir/${mdl_prefix}.0.dubm "$feats" \
        "ark:|gzip -c >$dir/${mdl_prefix}.gselect.JOB.gz" || exit 1;
fi

echo "$0: will train for $num_iters iterations, in parallel over"
echo "$0: $nj machines, parallelized with '$cmd'"

for x in `seq 0 $[$num_iters-1]`; do
    echo "$0: Training pass $x"
    if [ $stage -le $x ]; then
        # Accumulate stats.
        $cmd JOB=1:$nj $dir/log/${mdl_prefix}.acc.$x.JOB.log \
            gmm-global-acc-stats "--gselect=ark:gunzip -c $dir/${mdl_prefix}.gselect.JOB.gz|" \
            $dir/${mdl_prefix}.$x.dubm "$feats" $dir/${mdl_prefix}.$x.JOB.acc || exit 1;
        if [ $x -lt $[$num_iters-1] ]; then # Don't remove low-count Gaussians till last iter,
            opt="--remove-low-count-gaussians=false" # or gselect info won't be valid any more.
        else
            opt="--remove-low-count-gaussians=$remove_low_count_gaussians"
        fi
        $cmd $dir/log/${mdl_prefix}.update.$x.log \
            gmm-global-est $opt --min-gaussian-weight=$min_gaussian_weight $dir/${mdl_prefix}.$x.dubm "gmm-global-sum-accs - $dir/${mdl_prefix}.$x.*.acc|" \
            $dir/${mdl_prefix}.$[$x+1].dubm || exit 1;
        rm $dir/${mdl_prefix}.$x.*.acc $dir/${mdl_prefix}.$x.dubm
    fi
done

rm $dir/${mdl_prefix}.gselect.*.gz
mv $dir/${mdl_prefix}.$num_iters.dubm $dir/${mdl_prefix}.final.dubm || exit 1;

