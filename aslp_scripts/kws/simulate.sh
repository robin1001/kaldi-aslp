#!/bin/bash

if [ $# != 3 ]; then
    echo "Usage: simulate.sh src num_reps dest"
    exit 1;
fi

foreground_snrs="0:1:2:3:4:5:6:7:8:9:10:11:12:13:14:15:16:17:18:19:20"
background_snrs="0:1:2:3:4:5:6:7:8:9:10:11:12:13:14:15:16:17:18:19:20"

src_dir=$1
num_reps=$2
dest_dir=$3

set -e
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

[ ! -f rirs_noises.zip ] && [ ! -d RIRS_NOISES ] && wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
[ ! -d RIRS_NOISES ] && unzip rirs_noises.zip

# Add Reverb and noise
num_noise=$[$num_reps-1]
/usr/bin/python2.7 steps/data/reverberate_data_dir.py \
  --rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list" \
  --rir-set-parameters "0.25, RIRS_NOISES/simulated_rirs/mediumroom/rir_list" \
  --rir-set-parameters "0.25, RIRS_NOISES/simulated_rirs/largeroom/rir_list" \
  --noise-set-parameters RIRS_NOISES/pointsource_noises/noise_list \
  --prefix "simulation_" \
  --foreground-snrs $foreground_snrs \
  --background-snrs $background_snrs \
  --speech-rvb-probability 1 \
  --pointsource-noise-addition-probability 1 \
  --isotropic-noise-addition-probability 1 \
  --num-replications $num_noise \
  --max-noises-per-minute 1 \
  --source-sampling-rate 16000 \
  --random-seed 777 \
  ${src_dir} ${src_dir}/tmp_${num_noise}

mkdir -p ${dest_dir}

cat $src_dir/wav.scp ${src_dir}/tmp_${num_noise}/wav.scp | sort > $dest_dir/wav.scp
cat $src_dir/text ${src_dir}/tmp_${num_noise}/text | sort > $dest_dir/text
cat $src_dir/utt2spk ${src_dir}/tmp_${num_noise}/utt2spk | sort > $dest_dir/utt2spk
cat $src_dir/spk2utt ${src_dir}/tmp_${num_noise}/spk2utt | sort > $dest_dir/spk2utt

utils/validate_data_dir.sh --no-feats --no-text $dest_dir

