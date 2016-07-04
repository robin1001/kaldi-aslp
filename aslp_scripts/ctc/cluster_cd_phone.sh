#!/bin/bash

# Created on 2016-03-15
# Author: Binbin Zhang

context_opts=
delta_opts=
cmvn_opts=
norm_vars=false # deprecated.  Prefer --cmvn-opts "--norm-vars=true"
stage=0
cluster_thresh=-1
cmd=run.pl

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

set -e
set -o pipefail
set -u

if [ $# != 6 ]; then
   echo "Usage: $0 <num-leaves> <data-dir> <lang-dir> <model-dir> <alignment-dir> <exp-dir>"
   echo "e.g.: $0 2000 data/train_si84_half data/lang exp/mono_ali exp/tri1"
   echo "main options (for others, see top of script file)"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --config <config-file>                           # config containing options"
   echo "  --stage <stage>                                  # stage to do partial re-run from."
   exit 1;
fi

numleaves=$1
data=$2
lang=$3
mdldir=$4
alidir=$5
dir=$6

for f in $alidir/final.mdl $alidir/ali.1.gz $data/feats.scp $lang/phones.txt; do
  [ ! -f $f ] && echo "train_deltas.sh: no such file $f" && exit 1;
done

ciphonelist=`cat $lang/phones/context_indep.csl` || exit 1;
nj=`cat $alidir/num_jobs` || exit 1;
mkdir -p $dir/log
echo $nj > $dir/num_jobs

sdata=$data/split$nj;
split_data.sh $data $nj || exit 1;

[ $(cat $alidir/cmvn_opts 2>/dev/null | wc -c) -gt 1 ] && [ -z "$cmvn_opts" ] && \
  echo "$0: warning: ignoring CMVN options from source directory $alidir"
$norm_vars && cmvn_opts="--norm-vars=true $cmvn_opts"
echo $cmvn_opts  > $dir/cmvn_opts # keep track of options to CMVN.
[ ! -z $delta_opts ] && echo $delta_opts > $dir/delta_opts

feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |"

#rm $dir/.error 2>/dev/null

if [ $stage -le 0 ]; then
  echo "$0: accumulating tree stats"
  $cmd JOB=1:$nj $dir/log/acc_tree.JOB.log \
    aslp-acc-tree-stats-phone-mean $context_opts \
    --ci-phones=$ciphonelist $alidir/final.mdl "$feats" \
    "ark:gunzip -c $alidir/ali.JOB.gz|" $dir/JOB.treeacc || exit 1;
  sum-tree-stats $dir/treeacc $dir/*.treeacc 2>$dir/log/sum_tree_acc.log || exit 1;
  rm $dir/*.treeacc
fi

if [ $stage -le 1 ]; then
  echo "$0: getting questions for tree-building, via clustering"
  # preparing questions, roots file...
  cluster-phones $context_opts --pdf-class-list=0 $dir/treeacc $lang/phones/sets.int \
    $dir/questions.int 2> $dir/log/questions.log || exit 1;
  cat $lang/phones/extra_questions.int >> $dir/questions.int
  compile-questions $context_opts $lang/topo $dir/questions.int \
    $dir/questions.qst 2>$dir/log/compile_questions.log || exit 1;

  # copy questions
  #cp $mdldir/questions.int $dir/questions.int
  #compile-questions $context_opts $dir/topo $dir/questions.int \
  #  $dir/questions.qst 2>$dir/log/compile_questions.log || exit 1;

  echo "$0: building the tree"
  $cmd $dir/log/build_tree.log \
    build-tree $context_opts --verbose=1 --max-leaves=$numleaves \
    --cluster-thresh=$cluster_thresh $dir/treeacc $lang/phones/roots.int \
    $dir/questions.qst $dir/topo $dir/tree || exit 1;
fi
