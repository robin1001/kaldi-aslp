#!/bin/sh

# Created on 2016-04-04
# Author : Binbin Zhang
# Do data prepare for cd phone

num_leaves=3000
self_jump=0.5
cmd=run.pl

echo "$0 $@"  # Print the command line for logging
[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
    echo "Cluster cd phone and Do data preparation for cd phone training"
    echo "Usage: $0 src_mdl_dir src_ali_dir dst_ali_dir"
    echo "eg: $0 exp/tri2b exp/tri2b_ali exp/align_cd_phone"
    exit -1;
fi

src_mdl_dir=$1
src_ali_dir=$2
dst_ali_dir=$3

[ ! -d $dst_ali_dir ] && mkdir -p $dst_ali_dir;

# Attention: Default the first phone(SP) is used as fake blank phone

# Get number of mono phones
num_phone=$(am-info $src_ali_dir/final.mdl | grep phones | awk '{print $NF}')

# Make fake topo
aslp_scripts/ctc/make_fake_topo.sh --self-jump $self_jump $num_phone $dst_ali_dir/topo

# Cluster cd-phone
aslp_scripts/ctc/cluster_cd_phone.sh $num_leaves data/train \
    data/lang $src_mdl_dir $src_ali_dir $dst_ali_dir

# Make fake model
gmm-init-model-flat --binary=false $dst_ali_dir/tree $dst_ali_dir/topo $dst_ali_dir/final.mdl

# Convert alignment to mono phone alignment
for x in $src_ali_dir/ali.*.gz; do
{
    file_name=$(basename $x)
    aslp-convert-ali $src_ali_dir/final.mdl $dst_ali_dir/final.mdl $dst_ali_dir/tree \
        "ark:gunzip -c $src_ali_dir/$file_name |" ark:- | \
    aslp-ali-to-pdf $dst_ali_dir/final.mdl ark:- "ark:|gzip -c > $dst_ali_dir/$file_name"
} &
done
wait

# Make decode graph
aslp_scripts/ctc/make_ctc_graph.sh data/lang_test \
    $dst_ali_dir $dst_ali_dir/graph

