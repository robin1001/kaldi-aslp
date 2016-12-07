#!/bin/bash

# Copyright 2016  ASLP (Author: zhangbinbin)
# Apache 2.0

stage=4
feat_dir=data_fbank
gmmdir=exp/tri2b
skip_width=3
dir=exp/cnn_3lstm_dnn_asgd
ali=${gmmdir}_dnn
num_cv_utt=4000
#graph=graph_000_009_kni_p1e8_3gram
graph=graph

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
    aslp_scripts/aslp_nnet/prepare_feats_ali_parallel.sh \
        --cmvn_opts "--norm-means=true --norm-vars=true" \
        --splice_opts "--left-context=5 --right-context=5" \
        --num-worker 4 \
	    $feat_dir/train_tr $feat_dir/train_cv data/lang $ali $ali $dir || exit 1;
fi

# Get feats_tr feats_cv labels_tr labels_cv 
[ ! -f $dir/train.parallel.conf ] && \
    echo "$dir/train.parallel.conf(config file for nn training): no such file" && exit 1 
source $dir/train.parallel.conf

# Prepare lstm init nnet
if [ $stage -le 2 ]; then
    echo "Pretraining nnet"
    num_feat=$(feat-to-dim "$feats_tr" -) 
    num_tgt=$(hmm-info --print-args=false $ali/final.mdl | grep pdfs | awk '{ print $NF }')
    cell_dim=1024 # number of neurons per layer,
    recurrent_dim=512
    echo $num_feat $num_tgt

# Init nnet.proto with 2 lstm layers
cat > $dir/nnet.proto <<EOF
<NnetProto>
<LstmProjectedStreams> <InputDim> $num_feat <OutputDim> 512 <CellDim> 1024 <ParamScale> 0.010000 <ClipGradient> 5.000000
<LstmProjectedStreams> <InputDim> 512 <OutputDim> 512 <CellDim> 1024 <ParamScale> 0.010000 <ClipGradient> 5.000000
<LstmProjectedStreams> <InputDim> 512 <OutputDim> 512 <CellDim> 1024 <ParamScale> 0.010000 <ClipGradient> 5.000000
<AffineTransform> <InputDim> 512 <OutputDim> $num_tgt <BiasMean> 0.0 <BiasRange> 0.0 <ParamStddev> 0.040000
<Softmax> <InputDim> $num_tgt <OutputDim> $num_tgt
</NnetProto>
EOF

fi

if [ $stage -le 3 ]; then
    # init model
	nnet_init=$dir/nnet/train.nnet.init
	aslp-nnet-init $dir/nnet.proto $nnet_init
	# warm start
	single_init=$dir/nnet/train.single.nnet.final
    
	aslp_scripts/aslp_nnet/train_scheduler.sh --train-tool "aslp-nnet-train-lstm-streams" \
        --learn-rate 0.000032 \
        --momentum 0.9 \
        --max-iters 25 \
		--train-tool-opts "--gpu-id=2 --skip-width=3 --batch-size=20 --num-stream=100  --report-period=1000" \
        $nnet_init "$feats_tr" "$feats_cv" "$labels_tr" "$labels_cv" $dir
	cp $dir/final.nnet $single_init
	rm $dir/final.nnet
fi

# Train nnet(dnn, cnn, lstm)
if [ $stage -le 4 ]; then
    echo "Training nnet"
	single_final=$dir/nnet/train.single.nnet.final
	asgd_init=$dir/nnet/train.asgd.nnet.init
    cp $single_final $asgd_init 
	# --worker-tool-opts "--alpha=0.5 --sync-period=256000" \
	# --server-tool-opts "--alpha=0.5" \
	# --bmuf-learn-rate=1.0 --bmuf-momentum=0.75 
	aslp_scripts/aslp_nnet/train_scheduler_4workers.sh --train-type "asgd" \
        --learn-rate 0.000032 --momentum 0.9 \
		--gpu-num 4 --gpu-id 4 \
		--skip-width 3 \
		--max-iters 40 \
		--worker-tool-opts "--sync-period=10000" \
		--train-tool "aslp-nnet-train-lstm-streams" \
		--worker-tool "aslp-nnet-train-lstm-stream-worker" \
		--train-tool-opts "--skip-width=$skip_width --batch-size=20 --num-stream=100 --report-period=1000" \
        $asgd_init "$feats_tr" "$feats_cv" "$labels_tr" "$labels_cv" $dir
fi

# Decoding 
if [ $stage -le 5 ]; then
    for x in test3000 test3000_noise; do
		aslp_scripts/aslp_nnet/decode.sh --nj 5 --num-threads 3 \
        	--cmd "$decode_cmd" --acwt 0.0666667 \
        	--nnet-forward-opts "--no-softmax=false --apply-log=true --skip-width=$skip_width" \
        	--forward-tool "aslp-nnet-forward-skip" \
        	$gmmdir/$graph $feat_dir/${x} $dir/decode_${x}_${graph} || exit 1;
    	aslp_scripts/score_basic.sh --cmd "$decode_cmd" $feat_dir/${x} \
        	$gmmdir/$graph $dir/decode_${x}_${graph} || exit 1;
	done
fi

