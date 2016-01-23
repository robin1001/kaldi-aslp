#!/bin/bash

# Copyright 2016  ASLP (Author: zhangbinbin)
# Apache 2.0

stage=0
feat_dir=data_fbank
cv_utt_percent=10 # default 10% of total utterances 
gmmdir=exp/tri3
dir=exp/dnn_fbank
ali=${gmmdir}_ali
num_cv_utt=500

echo "$0 $@"  # Print the command line for logging
[ -f path.sh ] && . ./path.sh; 
. parse_options.sh || exit 1;

set -euo pipefail

# Making features, this script will gen corresponding feat and dir
# eg data-fbank/train data-fbank/train_tr90 data-fbank/train_cv10 data-fbank/test
if [ $stage -le 0 ]; then
    echo "Extracting feats & Create tr cv set"
    aslp_scripts/make_feats.sh --feat-type "fbank" data/train $feat_dir
    # Split tr & cv
    utils/shuffle_list.pl $feat_dir/train/feats.scp > $feat_dir/train/random.scp
    head -n$num_cv_utt $feat_dir/train/random.scp | sort > $feat_dir/train/cv.scp
    total=$(cat $feat_dir/train/random.scp | wc -l)
    left=$[$total-$num_cv_utt]
    tail -n$left $feat_dir/train/random.scp | sort > $feat_dir/train/tr.scp
    utils/subset_data_dir.sh --utt-list $feat_dir/train/tr.scp \
        $feat_dir/train $feat_dir/train_tr
    utils/subset_data_dir.sh --utt-list $feat_dir/train/cv.scp \
        $feat_dir/train $feat_dir/train_cv
	aslp_scripts/make_feats.sh --feat-type "fbank" data/test $feat_dir
fi

exit 0;
# Prepare feature and alignment config file for nn training
# This script will make $dir/train.conf automaticlly
if [ $stage -le 1 ]; then
    echo "Preparing alignment and feats"
        #--delta_opts "--delta-order=2" \
    aslp_scripts/aslp_nnet/prepare_feats_ali.sh \
        --cmvn_opts "--norm-means=true --norm-vars=true" \
        --splice_opts "--left-context=5 --right-context=5" \
        $feat_dir/train_tr $feat_dir/train_cv data/lang $ali $ali $dir || exit 1;
fi

# Get feats_tr feats_cv labels_tr labels_cv 
[ ! -f $dir/train.conf ] && \
    echo "$dir/train.conf(config file for nn training): no such file" && exit 1 
source $dir/train.conf

# Prerain nnet(dnn, cnn, lstm)
if [ $stage -le 2 ]; then
    echo "Pretraining nnet"
    num_feat=$(feat-to-dim "$feats_tr" -) 
    num_tgt=$(hmm-info --print-args=false $ali/final.mdl | grep pdfs | awk '{ print $NF }')
    hid_dim=1024        # number of neurons per layer,
    hid_layers=4        # nr. of hidden layers (before sotfmax or bottleneck),
    # num_patch=1+(40-9)/1=32  out_dim=128*32=4096 max_pool_out_dim=((32-4)/4+1)*128=1024
    num_filters=128
    conv_out_dim=4096
    max_pool_out_dim=1024
    [ ! $hid_dim -eq $max_pool_out_dim ] && 
        echo "hidden dim max_out_dim not equal $hid_dim $max_pool_out_dim" && exit 1
# Init nnet.proto with one cnn layers
cat > $dir/nnet.proto <<EOF
<NnetProto>
<ConvolutionalComponent> <InputDim> $num_feat <OutputDim> $conv_out_dim <PatchDim> 9 <PatchStep> 1 <PatchStride> 40 <BiasMean> -2.000000 <BiasRange> 4.000000 <ParamStddev> 0.1 <MaxNorm> 30
<MaxPoolingComponent> <InputDim> $conv_out_dim <OutputDim> $max_pool_out_dim <PoolSize> 4 <PoolStep> 4 <PoolStride> $num_filters
<Sigmoid> <InputDim> $max_pool_out_dim <OutputDim> $max_pool_out_dim
<AffineTransform> <InputDim> $hid_dim <OutputDim> $num_tgt <BiasMean> -2.000000 <BiasRange> 4.000000 <ParamStddev> 0.1
<Softmax> <InputDim> $num_tgt <OutputDim> $num_tgt
</NnetProto>
EOF

cat > $dir/hidden.conf <<EOF
<NnetProto>
<AffineTransform> <InputDim> $hid_dim <OutputDim> $hid_dim <BiasMean> -2.000000 <BiasRange> 4.000000 <ParamStddev> 0.1
<Sigmoid> <InputDim> $hid_dim <OutputDim> $hid_dim 
</NnetProto>
EOF

    #"$train_cmd" $dir/log/pretrain.log \
    aslp_scripts/aslp_nnet/pretrain.sh --train-tool "aslp-nnet-train-simple" \
        --learn-rate 0.008 \
        --momentum 0.0 \
        --minibatch_size 256 \
        --train-tool-opts "--report-period=60000" \
        --iters_per_epoch 2 \
        "$feats_tr" "$labels_tr" $hid_layers $dir
fi

# Train nnet(dnn, cnn, lstm)
if [ $stage -le 3 ]; then
    echo "Training nnet"
    [ ! -f $dir/nnet/pretrain.final.nnet ] && \
        echo "$dir/nnet/pretrain.final.nnet: no such file" && exit 1
    nnet_init=$dir/nnet/train.nnet.init
    [ -e $nnet_init ] && rm $nnet_init
    ln -s $(basename $dir/nnet/pretrain.final.nnet) $nnet_init
    #"$train_cmd" $dir/log/train.log \
    aslp_scripts/aslp_nnet/train_scheduler.sh --train-tool "aslp-nnet-train-simple" \
        --learn-rate 0.008 \
        --momentum 0.0 \
        --minibatch_size 256 \
        --train-tool-opts "--report-period=60000" \
        $nnet_init "$feats_tr" "$feats_cv" "$labels_tr" "$labels_cv" $dir
fi

# Decoding 
if [ $stage -le 4 ]; then
    aslp_scripts/aslp_nnet/decode.sh --nj 20 --cmd "$decode_cmd" --acwt 0.2 \
        $gmmdir/graph $feat_dir/test $dir/decode_test_iter90 || exit 1;
fi
