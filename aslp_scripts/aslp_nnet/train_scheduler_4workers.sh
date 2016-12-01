#!/bin/bash

# Copyright 2012-2015  Brno University of Technology (author: Karel Vesely)
# Copyright 2016  ASLP (author: zhangbinbin)
# Created on 2016-08-02
# Apache 2.0

# Schedules epochs and controls learning rate during the neural network training

# Begin configuration.

# training options,
learn_rate=0.008
momentum=0
minibatch_size=256
randomizer_size=32768
l1_penalty=0
l2_penalty=0

# data processing,
train_tool="aslp-nnet-train-frame"
train_tool_opts=
server_tool="aslp-nnet-train-server"
server_tool_opts=
worker_tool="aslp-nnet-train-frame-worker"
worker_tool_opts=
train_type="easgd"
gpu_num=4
gpu_id=0

# learn rate scheduling,
max_iters=20
min_iters=0 # keep training, disable weight rejection, start learn-rate halving as usual,
keep_lr_iters=0 # fix learning rate for N initial epochs, disable weight rejection,
start_halving_impr=0.01
end_halving_impr=0.001
halving_factor=0.5
parallel_config_file=

# misc,
verbose=0
 
# End configuration.

echo "$0 $@"  # Print the command line for logging
[ -f path.sh ] && . ./path.sh; 

. parse_options.sh || exit 1;

set -euo pipefail

if [ $# != 6 ]; then
   echo "Usage: $0 <mlp-init> <feats-tr> <feats-cv> <labels-tr> <labels-cv> <exp-dir>"
   echo " e.g.: $0 0.nnet scp:train.scp scp:cv.scp ark:labels_tr.ark ark:labels_cv.ark exp/dnn1"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>  # config containing options"
   exit 1;
fi

mlp_init=$1
feats_tr=$2
feats_cv=$3
labels_tr=$4
labels_cv=$5
dir=$6


[ ! -d $dir ] && mkdir $dir
[ ! -d $dir/log ] && mkdir $dir/log
[ ! -d $dir/nnet ] && mkdir $dir/nnet

# Skip training
[ -e $dir/final.nnet ] && echo "'$dir/final.nnet' exists, skipping training" && exit 0

# train.parallel.conf provide the data parallel conf, eg feats_tr1, feats_tr2, feats_tr3 feats_tr4
# which will be used in mpirun later
[ -z $parallel_config_file ] && echo "Use $dir/train.parallel.conf as parallel_config_file" && parallel_config_file=$dir/train.parallel.conf 
[ ! -e $parallel_config_file ] && echo "$parallel_config_file do not exists, it is essential" && exit 0
source $parallel_config_file

##############################
# start training

# choose mlp to start with,
mlp_best=$mlp_init
mlp_base=${mlp_init##*/}; mlp_base=${mlp_base%.*}

# optionally resume training from the best epoch, using saved learning-rate,
[ -e $dir/.mlp_best ] && mlp_best=$(cat $dir/.mlp_best)
[ -e $dir/.learn_rate ] && learn_rate=$(cat $dir/.learn_rate)

# cross-validation on original network,
if [ -e $dir/.init_cv ]; then
    loss=$(cat $dir/.init_cv)
    loss_type=$(cat $dir/.loss_type)
    echo "CROSSVAL PRERUN AVG.LOSS $(printf "%.4f" $loss) $loss_type"
else
    log=$dir/log/iter00.initial.log; hostname>$log
    $train_tool --gpu-id=$gpu_id --cross-validate=true --randomize=false --verbose=$verbose $train_tool_opts \
      ${feature_transform:+ --feature-transform=$feature_transform} \
      "$feats_cv" "$labels_cv" $mlp_best \
      2>> $log
    loss=$(cat $dir/log/iter00.initial.log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
    loss_type=$(cat $dir/log/iter00.initial.log | grep "AvgLoss:" | tail -n 1 | awk '{ print $5; }')
    echo "CROSSVAL PRERUN AVG.LOSS $(printf "%.4f" $loss) $loss_type"
    echo "$loss" >> $dir/.init_cv
    echo "$loss_type" >> $dir/.loss_type
fi 

# resume lr-halving,
halving=0
[ -e $dir/.halving ] && halving=$(cat $dir/.halving)

# training,
for iter in $(seq -w $max_iters); do
  echo -n `date` " ITERATION $iter: "
  mlp_next=$dir/nnet/${mlp_base}_iter${iter}
  
  # skip iteration (epoch) if already done,
  [ -e $dir/.done_iter$iter ] && echo -n "skipping... " && ls $mlp_next* && continue 
  
  # training,
  log=$dir/log/iter${iter}.tr.log; hostname>$log

  train_tool_opts="$train_tool_opts --randomize=true --verbose=$verbose --learn-rate=$learn_rate --momentum=$momentum --minibatch-size=$minibatch_size"
  train_tool_opts="$train_tool_opts --randomizer-size=$randomizer_size --l1-penalty=$l1_penalty --l2-penalty=$l2_penalty"
  
  # Mpi parallel training
  # with server
  if [[ $train_type == "easgd" || $train_type == "asgd" ]]; then
    mpirun --output-filename $log \
        -n 1 $server_tool --server-type=$train_type $server_tool_opts --gpu-id=$gpu_id $mlp_best $mlp_next : \
        -n 1 $worker_tool --worker-type=$train_type $worker_tool_opts --gpu-id=$[$gpu_id+0] $train_tool_opts "$feats_tr1" "$labels_tr1" $mlp_best $mlp_next : \
        -n 1 $worker_tool --worker-type=$train_type $worker_tool_opts --gpu-id=$[$gpu_id+1] $train_tool_opts "$feats_tr2" "$labels_tr2" $mlp_best $mlp_next : \
        -n 1 $worker_tool --worker-type=$train_type $worker_tool_opts --gpu-id=$[$gpu_id+2] $train_tool_opts "$feats_tr3" "$labels_tr3" $mlp_best $mlp_next : \
        -n 1 $worker_tool --worker-type=$train_type $worker_tool_opts --gpu-id=$[$gpu_id+3] $train_tool_opts "$feats_tr4" "$labels_tr4" $mlp_best $mlp_next \
        2>>$log #&> /dev/null
  else  #without server
    mpirun --output-filename $log \
        -n 1 $worker_tool --worker-type=$train_type $worker_tool_opts --gpu-id=$[$gpu_id+0] $train_tool_opts "$feats_tr1" "$labels_tr1" $mlp_best $mlp_next : \
        -n 1 $worker_tool --worker-type=$train_type $worker_tool_opts --gpu-id=$[$gpu_id+1] $train_tool_opts "$feats_tr2" "$labels_tr2" $mlp_best $mlp_next : \
        -n 1 $worker_tool --worker-type=$train_type $worker_tool_opts --gpu-id=$[$gpu_id+2] $train_tool_opts "$feats_tr3" "$labels_tr3" $mlp_best $mlp_next : \
        -n 1 $worker_tool --worker-type=$train_type $worker_tool_opts --gpu-id=$[$gpu_id+3] $train_tool_opts "$feats_tr4" "$labels_tr4" $mlp_best $mlp_next \
        2>>$log #2>>$log #&> /dev/null
  fi

  tr_loss=$(cat $dir/log/iter${iter}.tr.log.1.1 | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
  echo -n "TRAIN AVG.LOSS $(printf "%.4f" $tr_loss), (lrate$(printf "%.6g" $learn_rate)), "
  # cross-validation,
  log=$dir/log/iter${iter}.cv.log; hostname>$log
  $train_tool --gpu-id=$gpu_id --cross-validate=true --randomize=false --verbose=$verbose $train_tool_opts \
    "$feats_cv" "$labels_cv" $mlp_next \
    2>>$log || exit 1;
  
  loss_new=$(cat $dir/log/iter${iter}.cv.log | grep "AvgLoss:" | tail -n 1 | awk '{ print $4; }')
  echo "CROSSVAL AVG.LOSS $(printf "%.4f" $loss_new)"

  # accept or reject?
  loss_prev=$loss
  if [ 1 == $(bc <<< "$loss_new < $loss") -o $iter -le $keep_lr_iters -o $iter -le $min_iters ]; then
    # accepting: the loss was better, or we had fixed learn-rate, or we had fixed epoch-number,
    loss=$loss_new
    mlp_best=$dir/nnet/${mlp_base}_iter${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $tr_loss)_cv$(printf "%.4f" $loss_new)
    mlp_best=$dir/nnet/${mlp_base}_iter${iter}_learnrate${learn_rate}_cv$(printf "%.4f" $loss_new)
    [ $iter -le $min_iters ] && mlp_best=${mlp_best}_min-iters-$min_iters
    [ $iter -le $keep_lr_iters ] && mlp_best=${mlp_best}_keep-lr-iters-$keep_lr_iters
    mv $mlp_next $mlp_best
    echo "nnet accepted ($(basename $mlp_best))"
    echo $mlp_best > $dir/.mlp_best 
  else
    # rejecting,
    mlp_reject=$dir/nnet/${mlp_base}_iter${iter}_learnrate${learn_rate}_tr$(printf "%.4f" $tr_loss)_cv$(printf "%.4f" $loss_new)_rejected
    mlp_reject=$dir/nnet/${mlp_base}_iter${iter}_learnrate${learn_rate}_cv$(printf "%.4f" $loss_new)_rejected
    mv $mlp_next $mlp_reject
    echo "nnet rejected ($(basename $mlp_reject))"
  fi

  # create .done file, the iteration (epoch) is completed,
  touch $dir/.done_iter$iter
  
  # continue with original learn-rate,
  [ $iter -le $keep_lr_iters ] && continue 

  # stopping criterion,
  rel_impr=$(bc <<< "scale=10; ($loss_prev-$loss)/$loss_prev")
  if [ 1 == $halving -a 1 == $(bc <<< "$rel_impr < $end_halving_impr") ]; then
    if [ $iter -le $min_iters ]; then
      echo we were supposed to finish, but we continue as min_iters : $min_iters
      continue
    fi
    echo finished, too small rel. improvement $rel_impr
    break
  fi

  # start learning-rate fade-out when improvement is low,
  if [ 1 == $(bc <<< "$rel_impr < $start_halving_impr") ]; then
    halving=1
    echo $halving >$dir/.halving
  fi
  
  # reduce the learning-rate,
  if [ 1 == $halving ]; then
    learn_rate=$(awk "BEGIN{print($learn_rate*$halving_factor)}")
    echo $learn_rate >$dir/.learn_rate
  fi
done

# select the best network,
if [ $mlp_best != $mlp_init ]; then 
  mlp_final=${mlp_best}_final_
  ( cd $dir/nnet; ln -s $(basename $mlp_best) $(basename $mlp_final); )
  ( cd $dir; ln -s nnet/$(basename $mlp_final) final.nnet; )
  echo "Succeeded training the Neural Network : $dir/final.nnet"
else
  "Error training neural network..."
  exit 1
fi

