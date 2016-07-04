#!/bin/bash

# Created on 2016-04-25
# Author: Binbin Zhang

stage=4
cmd=run.pl
nj=4
num_cv_utt=2000
num_test_utt=2000
feat_dir=mfcc
gmmdir=exp/tri2b
ali=${gmmdir}_ali
dir=exp/dnn_vad

echo "$0 $@"  # Print the command line for logging
[ -f cmd.sh ] && . ./cmd.sh;
[ -f path.sh ] && . ./path.sh; 
. parse_options.sh || exit 1;

set -euo pipefail

# Prepare feat feat_type: mfcc
if [ $stage -le 0 ]; then 
    echo "Extracting feats & Create tr cv set"
    #aslp_scripts/make_feats.sh --feat-type "mfcc" data/train $feat_dir
    # Split tr & cv
    utils/shuffle_list.pl $feat_dir/train/feats.scp > $feat_dir/train/random.scp
    cat $feat_dir/train/random.scp | awk -v ed=$num_test_utt '{if(NR <= ed) print $0}' |\
        sort > $feat_dir/train/test.scp
    cat $feat_dir/train/random.scp | awk -v st=$num_test_utt -v ed=$[$num_test_utt+$num_cv_utt] '{if (NR > st && NR <= ed) print $0}' |\
        sort > $feat_dir/train/cv.scp
    cat $feat_dir/train/random.scp | awk -v st=$[$num_test_utt+$num_cv_utt] '{if (NR > st ) print $0}' |\
        sort > $feat_dir/train/tr.scp
    utils/subset_data_dir.sh --utt-list $feat_dir/train/test.scp \
        $feat_dir/train $feat_dir/test
    utils/subset_data_dir.sh --utt-list $feat_dir/train/tr.scp \
        $feat_dir/train $feat_dir/train_tr
    utils/subset_data_dir.sh --utt-list $feat_dir/train/cv.scp \
        $feat_dir/train $feat_dir/train_cv
fi

# Prepare ali
if [ $stage -le 1 ]; then 
    aslp_scripts/vad/prepare_feats_ali.sh \
        --cmvn_opts "--norm-vars=false --center=true --cmn-window=300" \
        --delta_opts "--delta-order=2" \
        --splice_opts "--left-context=5 --right-context=5" \
        --sil-id 1 \
        $feat_dir/train_tr $feat_dir/train_cv $feat_dir/test data/lang $ali $ali $ali $dir || exit 1;
fi

# Get feats_tr feats_cv labels_tr labels_cv 
[ ! -f $dir/train.conf ] && \
    echo "$dir/train.conf(config file for nn training): no such file" && exit 1 
source $dir/train.conf

# DNN network proto
if [ $stage -le 2 ]; then
    num_feat=$(feat-to-dim "$feats_tr" -) 
    hid_dim=512

cat > $dir/nnet.proto <<EOF
<NnetProto>
<AffineTransform> <InputDim> $num_feat <OutputDim> 512 <BiasMean> -2.000000 <BiasRange> 4.000000 <ParamStddev> 0.1
<Sigmoid> <InputDim> 512 <OutputDim> 512
<AffineTransform> <InputDim> 512 <OutputDim> 512 <BiasMean> -2.000000 <BiasRange> 4.000000 <ParamStddev> 0.1
<Sigmoid> <InputDim> 512 <OutputDim> 512
<AffineTransform> <InputDim> 512 <OutputDim> 512 <BiasMean> -2.000000 <BiasRange> 4.000000 <ParamStddev> 0.1
<Sigmoid> <InputDim> 512 <OutputDim> 512
<AffineTransform> <InputDim> 512 <OutputDim> 2 <BiasMean> -2.000000 <BiasRange> 4.000000 <ParamStddev> 0.1
<Softmax> <InputDim> 2 <OutputDim> 2
EOF

fi

# Train nnet(dnn, cnn, lstm)
if [ $stage -le 3 ]; then
    echo "Training nnet"
    nnet_init=$dir/nnet/train.nnet.init
    aslp-nnet-init $dir/nnet.proto $nnet_init
    #"$train_cmd" $dir/log/train.log \
    aslp_scripts/aslp_nnet/train_scheduler.sh --train-tool "aslp-nnet-train-frame" \
        --learn-rate 0.001 \
        --momentum 0.9 \
        --minibatch_size 256 \
        --train-tool-opts "--report-period=120000" \
        $nnet_init "$feats_tr" "$feats_cv" "$labels_tr" "$labels_cv" $dir
fi

# Test
if [ $stage -le 4 ]; then
    [ ! -e $dir/final.nnet ] && echo "$dir/final.nnet: no such file" && exit 1;
    aslp-eval-nn-vad --stride=0.001 $dir/final.nnet "$feats_test" "$labels_test"
fi

exit 0;
