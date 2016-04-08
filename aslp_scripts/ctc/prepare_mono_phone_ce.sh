#!/bin/sh

# Created on 2016-03-22
# Author : Binbin Zhang
# Do data prepare for mono phone CTC train

echo "$0 $@"  # Print the command line for logging
[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
    echo "Do data preparation for dnn mono phone CTC training"
    echo "Usage: $0 src_ali_dir dst_ali_dir"
    echo "eg: $0 exp/tri2b_ali exp/align_mono_phone"
    exit -1;
fi

src_ali_dir=$1
dst_ali_dir=$2

[ ! -d $dst_ali_dir ] && mkdir -p $dst_ali_dir;

# Get number of mono phones
num_phone=$(am-info $src_ali_dir/final.mdl | grep phones | awk '{print $NF}')

# Make fake tree
aslp_scripts/ctc/make_fake_tree.sh $num_phone $dst_ali_dir/tree

# Make fake CTC model
aslp_scripts/ctc/make_fake_mdl.sh $num_phone $dst_ali_dir/final.mdl

# Convert alignment to mono phone alignment
for x in $src_ali_dir/ali.*.gz; do
{
    file_name=$(basename $x)
    aslp-convert-ali $src_ali_dir/final.mdl $dst_ali_dir/final.mdl $dst_ali_dir/tree \
        "ark:gunzip -c $src_ali_dir/$file_name |" "ark:|gzip -c > $dst_ali_dir/$file_name"
} &
done
wait

# Make fixlabel decode graph
aslp_scripts/cd_phone/make_h3_graph.sh --mono data/lang_test \
    $dst_ali_dir $dst_ali_dir/graph

