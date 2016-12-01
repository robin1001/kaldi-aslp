#!/bin/bash

# Created on 2016-11-30
# Author: HeChangqing

. path.sh

data=data
# raw wav stores in $raw_wav_dir
raw_wav_dir=./raw-wav
# vad wav will be stored in $vad_wav_dir
vad_wav_dir=./vad-wav

mkdir -p $data
mkdir -p $vad_wav_dir
find $raw_wav_dir -iname "*.wav" | sort > $data/flist
cat $data/flist | sed -e 's:.*/\(.*\).wav:\1:g' > $data/uttids
paste $data/uttids $data/flist > $data/raw_wav.scp

cat $data/uttids | awk -v dir=${vad_wav_dir} '{print $1 " " dir "/" $1 ".wav"}' > $data/vad_wav.scp

aslp-apply-energy-vad --sil-thresh=0.9 scp:$data/raw_wav.scp ark:$data/vad_wav.scp
