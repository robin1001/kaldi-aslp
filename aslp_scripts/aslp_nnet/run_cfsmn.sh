#!/bin/bash

# Copyright 2016  ASLP (Author: zhangbinbin)
# Apache 2.0

stage=3
feat_dir=data_fbank
gmmdir=exp/tri2b
dir=exp/cnn_4cfsmn
ali=${gmmdir}_ali
num_cv_utt=4000

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
    aslp_scripts/aslp_nnet/prepare_feats_ali.sh \
        --delta_opts "--delta-order=2" \
		--cmvn_opts "--norm-means=true --norm-vars=true" \
        --splice_opts "--left-context=1 --right-context=1" \
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
    cell_dim=1024 # number of neurons per layer,
    recurrent_dim=512
	bias_mean=0
	bias_range=1
	hid_num=1024
	project_num=512
    echo $num_feat $num_tgt

# Init nnet.proto with 2 lstm layers
cat > $dir/nnet.proto <<EOF
<NnetProto>
<AffineTransform> <InputDim> $num_feat <OutputDim> $hid_num <BiasMean> $bias_mean <BiasRange> $bias_range <ParamStddev> 0.1
<ReLU> <InputDim> $hid_num <OutputDim> $hid_num
<AffineTransform> <InputDim> $hid_num <OutputDim> 512 <BiasMean> $bias_mean <BiasRange> $bias_range <ParamStddev> 0.1
<CompactFsmn> <InputDim> 512 <OutputDim> 512 <PastContext> 30 <FutureContext> 30 <LearnRateCoef> 1.0 
<AffineTransform> <InputDim> 512 <OutputDim> $hid_num <BiasMean> $bias_mean <BiasRange> $bias_range <ParamStddev> 0.1
<Sigmoid> <InputDim> $hid_num <OutputDim> $hid_num
<AffineTransform> <InputDim> $hid_num <OutputDim> 512 <BiasMean> $bias_mean <BiasRange> $bias_range <ParamStddev> 0.1
<CompactFsmn> <InputDim> 512 <OutputDim> 512 <PastContext> 30 <FutureContext> 30 <LearnRateCoef> 1.0 
<AffineTransform> <InputDim> 512 <OutputDim> $hid_num <BiasMean> $bias_mean <BiasRange> $bias_range <ParamStddev> 0.1
<ReLU> <InputDim> $hid_num <OutputDim> $hid_num
<AffineTransform> <InputDim> $hid_num <OutputDim> 512 <BiasMean> $bias_mean <BiasRange> $bias_range <ParamStddev> 0.1
<CompactFsmn> <InputDim> 512 <OutputDim> 512 <PastContext> 30 <FutureContext> 30 <LearnRateCoef> 1.0 
<AffineTransform> <InputDim> 512 <OutputDim> $hid_num <BiasMean> $bias_mean <BiasRange> $bias_range <ParamStddev> 0.1
<Sigmoid> <InputDim> $hid_num <OutputDim> $hid_num
<AffineTransform> <InputDim> $hid_num <OutputDim> $hid_num <BiasMean> $bias_mean <BiasRange> $bias_range <ParamStddev> 0.1
<ReLU> <InputDim> $hid_num <OutputDim> $hid_num
<AffineTransform> <InputDim> $hid_num <OutputDim> $hid_num <BiasMean> $bias_mean <BiasRange> $bias_range <ParamStddev> 0.1
<Sigmoid> <InputDim> $hid_num <OutputDim> $hid_num
<LinearTransform> <InputDim> $hid_num <OutputDim> 512 <ParamStddev> 0.1
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
	aslp-nnet-info $nnet_init
	#"$train_cmd" $dir/log/train.log \
	# --momentum 0.9
	aslp_scripts/aslp_nnet/train_scheduler.sh --train-tool "aslp-nnet-train-perutt" \
        --learn-rate 0.04 \
		--max-iters 30 \
		--train-tool-opts "--gpu-id=1 --report-period=360000" \
        $nnet_init "$feats_tr" "$feats_cv" "$labels_tr" "$labels_cv" $dir
fi

# Decoding 
if [ $stage -le 4 ]; then
	for x in test3000; do
    	aslp_scripts/aslp_nnet/decode.sh --nj 8 --num-threads 4 \
        	--cmd "$decode_cmd" --acwt 0.0666667 \
        	--nnet-forward-opts "--no-softmax=false --apply-log=true" \
        	--forward-tool "aslp-nnet-forward" \
        	$gmmdir/graph $feat_dir/${x} $dir/decode_${x} || exit 1;
    	aslp_scripts/score_basic.sh --cmd "$decode_cmd" $feat_dir/${x} \
        	$gmmdir/graph $dir/decode_${x} || exit 1;
	done
fi

