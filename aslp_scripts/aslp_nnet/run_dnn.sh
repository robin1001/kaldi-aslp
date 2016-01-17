#!/bin/bash

# Copyright 2016  ASLP (Author: zhangbinbin)
# Apache 2.0

stage=1
feat_dir=data-fbank
cv_utt_percent=10 # default 10% of total utterances 
gmmdir=exp/tri3
dir=exp/dnn_fbank_pretrain
ali=${gmmdir}_ali
train_dir=$feat_dir/train_tr$[100-$cv_utt_percent]
cv_dir=$feat_dir/train_cv$cv_utt_percent

echo "$0 $@"  # Print the command line for logging
[ -f path.sh ] && . ./path.sh; 
. parse_options.sh || exit 1;

# Options
#if [ $# != 6 ]; then
#    echo "Options:"    
#fi

# Making features, this script will gen corresponding feat and dir
# eg data-fbank/train data-fbank/train_tr90 data-fbank/train_cv10 data-fbank/test
if [ $stage -le 0 ]; then
    aslp_scripts/make_feats.sh \
        --feat-type "fbank" \
        --cv-utt-percent $cv_utt_percent \
        data $feat_dir
fi

# Prepare feature and alignment config file for nn training
# This script will make $dir/train.conf automaticlly
if [ $stage -le 1 ]; then
    aslp_scripts/aslp_nnet/prepare_feats_ali.sh \
        --cmvn_opts "--norm-means=true --norm-vars=true" \
        --splice_opts "--left-context=4 --right-context=4" \
        --delta_opts "--delta-order=2" \
        $train_dir $cv_dir data/lang $ali $ali $dir || exit 1;
fi

# Get feats_tr feats_cv labels_tr labels_cv 
[ ! -f $dir/train.conf ] && \
    echo "$dir/train.conf(config file for nn training): no such file" && exit 1 
source $dir/train.conf

# Prerain nnet(dnn, cnn, lstm)
if [ $stage -le 2 ]; then
    num_feat=$(feat-to-dim "$feats_tr" -) 
    num_tgt=$(hmm-info --print-args=false $ali/final.mdl | grep pdfs | awk '{ print $NF }')
    hid_dim=1024        # number of neurons per layer,
    hid_layers=4        # nr. of hidden layers (before sotfmax or bottleneck),
    echo $num_feat $num_tgt
cat > $dir/hidden.conf <<EOF
<NnetProto>
<AffineTransform> <InputDim> $hid_dim <OutputDim> $hid_dim <BiasMean> -2.000000 <BiasRange> 4.000000 <ParamStddev> 0.1
<Sigmoid> <InputDim> $hid_dim <OutputDim> $hid_dim 
</NnetProto>
EOF

# Init nnet.proto with 3 layers
cat > $dir/nnet.proto <<EOF
<NnetProto>
<AffineTransform> <InputDim> $num_feat <OutputDim> $hid_dim <BiasMean> -2.000000 <BiasRange> 4.000000 <ParamStddev> 0.1
<Sigmoid> <InputDim> $hid_dim <OutputDim> $hid_dim 
<AffineTransform> <InputDim> $hid_dim <OutputDim> $num_tgt <BiasMean> -2.000000 <BiasRange> 4.000000 <ParamStddev> 0.1
<Softmax> <InputDim> $num_tgt <OutputDim> $num_tgt
</NnetProto>
EOF
    aslp_scripts/aslp_nnet/pretrain.sh --train-tool "aslp-nnet-train-simple" \
        --learn-rate 0.008 \
        --minibatch_size 256 \
        --train-tool-opts "--report-period=60000" \
        --iters_per_epoch 2 \
        "$feats_tr" "$labels_tr" $hid_layers $dir
fi

## Test
#if [ $stage -lt 2]; then

#fi
