#!/bin/bash

# Copyright 2016  ASLP (Author: zhangbinbin)
# Apache 2.0

stage=2
feat_dir=data_fbank
gmmdir=exp/tri2b
dir=exp/state_2blstm
ali=${gmmdir}_ali
num_cv_utt=500

echo "$0 $@"  # Print the command line for logging
[ -f cmd.sh ] && . ./cmd.sh;
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

# Prepare feature and alignment config file for nn training
# This script will make $dir/train.conf automaticlly
if [ $stage -le 1 ]; then
    echo "Preparing alignment and feats"
        #--delta_opts "--delta-order=2" \
        #--splice_opts "--left-context=5 --right-context=5" \
    aslp_scripts/aslp_nnet/prepare_feats_ali.sh \
        --cmvn_opts "--norm-means=true --norm-vars=true" \
        $feat_dir/train_tr $feat_dir/train_cv data/lang $ali $ali $dir || exit 1;
fi

# Get feats_tr feats_cv labels_tr labels_cv 
[ ! -f $dir/train.conf ] && \
    echo "$dir/train.conf(config file for nn training): no such file" && exit 1 
source $dir/train.conf

# Prepare lstm init nnet
if [ $stage -le 2 ]; then
    echo "Pretraining nnet"
    num_feat=$(feat-to-dim "$feats_tr" -) 
    num_tgt=$(hmm-info --print-args=false $ali/final.mdl | grep pdfs | awk '{ print $NF }')
    echo $num_feat $num_tgt

# Init nnet.proto with 2 lstm layers
cat > $dir/nnet.proto <<EOF
<NnetProto>
<BLstm> <InputDim> $num_feat <OutputDim> 512 <ParamScale> 0.010000 <ClipGradient> 5.000000
<BatchNormalization> <InputDim> 512 <OutputDim> 512
<BLstm> <InputDim> 512 <OutputDim> 512 <ParamScale> 0.010000 <ClipGradient> 5.000000
<BatchNormalization> <InputDim> 512 <OutputDim> 512
<BLstm> <InputDim> 512 <OutputDim> 512 <ParamScale> 0.010000 <ClipGradient> 5.000000
<BatchNormalization> <InputDim> 512 <OutputDim> 512
<BLstm> <InputDim> 512 <OutputDim> 512 <ParamScale> 0.010000 <ClipGradient> 5.000000
<BatchNormalization> <InputDim> 512 <OutputDim> 512
<AffineTransform> <InputDim> 512 <OutputDim> $num_tgt <BiasMean> 0.0 <BiasRange> 0.0 <ParamStddev> 0.040000
<Softmax> <InputDim> $num_tgt <OutputDim> $num_tgt 
</NnetProto>
EOF

fi

# Train nnet(dnn, cnn, lstm)
if [ $stage -le 3 ]; then
    echo "Training nnet"
    nnet_init=$dir/nnet/train.nnet.init
    aslp-nnet-init $dir/nnet.proto $nnet_init
    #"$train_cmd" $dir/log/train.log \
    aslp_scripts/aslp_nnet/train_scheduler.sh --train-tool "aslp-nnet-train-blstm-streams" \
        --learn-rate 0.00001 \
        --momentum 0.9 \
        --train-tool-opts "--num-stream=10 --report-period=20 --drop-len=1500" \
        $nnet_init "$feats_tr" "$feats_cv" "$labels_tr" "$labels_cv" $dir
fi

# Decoding 
if [ $stage -le 4 ]; then
    aslp_scripts/aslp_nnet/decode.sh --nj 2 --num-threads 12 \
        --cmd "$decode_cmd" --acwt 0.0666667 \
        --nnet-forward-opts "--no-softmax=false --apply-log=true" \
        $gmmdir/graph $feat_dir/test $dir/decode_test3000 || exit 1;
    aslp_scripts/score_basic.sh --cmd "$decode_cmd" $feat_dir/test \
        $gmmdir/graph $dir/decode_test3000 || exit 1;
fi
