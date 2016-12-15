
config="--l2-penalty=1e-4 --report-period=120000 --randomize=true --verbose=0 --learn-rate=0.0001 --momentum=0.9 --minibatch-size=256 --randomizer-size=32768"  
feats1='ark:copy-feats scp:parallel/0.scp ark:- | apply-cmvn --norm-means=true --norm-vars=true ../data_vad_800h_fbank24_feat/cmvn ark:- ark:- | splice-feats --left-context=5 --right-context=5 ark:- ark:- |' 
feats2='ark:copy-feats scp:parallel/1.scp ark:- | apply-cmvn --norm-means=true --norm-vars=true ../data_vad_800h_fbank24_feat/cmvn ark:- ark:- | splice-feats --left-context=5 --right-context=5 ark:- ark:- |' 
ali='ark:ali-to-phones --per-frame=true ../data_vad_800h/ali/final.mdl "ark:gunzip -c ../data_vad_800h/ali/ali.cv.gz |" ark:- | aslp-ali-to-sil --sil-id=1 ark:- ark:- | ali-to-post ark:- ark:- |'
nnet_in=parallel/train.nnet.init
nnet_out=parallel/nnet.out

#train_type="single"
train_type="easgd"
#train_type="bsp"
#train_type="bmuf"

# Single
if [ $train_type == "single" ]; then
    aslp-nnet-train-frame $config "$feats1" "$ali" $nnet_in $nnet_out
fi

# BSP
if [ $train_type == "bsp" ]; then
    mpirun --output-filename parallel/bsp.log \
       -n 1 aslp-nnet-train-frame-worker --worker-type=bsp $config "$feats1" "$ali" $nnet_in $nnet_out : \
       -n 1 aslp-nnet-train-frame-worker --worker-type=bsp $config "$feats2" "$ali" $nnet_in $nnet_out : 
fi

# EASGD
if [ $train_type == "easgd" ]; then
    mpirun --output-filename parallel/easgd.log \
       -n 1 aslp-nnet-train-server --alpha=0.5 $nnet_in $nnet_out : \
       -n 1 aslp-nnet-train-frame-worker --worker-type=easgd $config "$feats1" "$ali" $nnet_in $nnet_out : \
       -n 1 aslp-nnet-train-frame-worker --worker-type=easgd $config "$feats2" "$ali" $nnet_in $nnet_out : 
fi

# BMUF
if [ $train_type == "bmuf" ]; then
    mpirun --output-filename parallel/bmuf.log \
       -n 1 aslp-nnet-train-frame-worker --worker-type=bmuf --bmuf-momentum=0.5 $config "$feats1" "$ali" $nnet_in $nnet_out : \
       -n 1 aslp-nnet-train-frame-worker --worker-type=bmuf --bmuf-momentum=0.5 $config "$feats2" "$ali" $nnet_in $nnet_out : 
fi




