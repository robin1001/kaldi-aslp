
# Created on 2016-07-10
# Author: Zhang Binbin


# Gen Textgrid format file for praat analysis, only support one wave file

wav_scp=test.scp

feats_test="ark:copy-feats \"ark:compute-fbank-feats --config=../../data_vad_800h_fbank24_feat/fbank.conf scp:${wav_scp} ark:- |\" ark:- | apply-cmvn --norm-means=true --norm-vars=true ../../data_vad_800h_fbank24_feat/cmvn ark:- ark:- | splice-feats --left-context=5 --right-context=2 ark:- ark:- |"

nn_model=../exp/dnn_fbank24_delta0_cmvn_ctx5+1+2_relu/final.nnet

hr-apply-nn-vad-segment --sil-thresh=0.5 --lookback=200 \
    --silence-trigger-threshold=200 --speech-trigger-threshold=70 \
    --min-length=0 --max-length=1000 \
    $nn_model "$feats_test" > segment.info

python gen_textgrid_according_vad_interval.py segment.info segment.TextGrid

