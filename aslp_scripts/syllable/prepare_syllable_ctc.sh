#!/bin/sh

# Created on 2016-05-18
# Author : Binbin Zhang
# Do data prepare for syllable ctc train

num_syllable=1310 #number of syllables in total
phone_mdl_file=exp/tri2b/final.mdl
phone_list_file=exp/tri2b/graph/phones.txt
syllable_list_file=data/lang_test/phones.txt
syllable_bind_info_file=syllable/syllable.bind.info2

echo "$0 $@"  # Print the command line for logging
[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
    echo "Do data preparation for syllable CTC training"
    echo "Usage: $0 src_ali_dir dst_ali_dir"
    echo "eg: $0 exp/tri2b_ali exp/syllable_ctc_ali"
    exit -1;
fi

src_ali_dir=$1
dst_ali_dir=$2

[ ! -d $dst_ali_dir ] && mkdir -p $dst_ali_dir;

# Make fake tree
aslp_scripts/ctc/make_fake_tree.sh $num_syllable $dst_ali_dir/tree

# Make fake CTC model
aslp_scripts/ctc/make_fake_mdl.sh $num_syllable $dst_ali_dir/final.mdl

# Convert alignment to mono phone alignment
#for x in $src_ali_dir/ali.*.gz; do
#{
#    file_name=$(basename $x)
#    copy-int-vector "ark:gunzip -c $src_ali_dir/$file_name |" "ark:-" | \
#        ali-to-phones --per-frame=true $phone_mdl_file "ark:-" "ark,t:-" | \
#        aslp_scripts/syllable/ali_to_syllable.py $phone_list_file $syllable_list_file \
#            $syllable_bind_info_file | \
#        aslp-ali-minus-one --unique=true "ark:-" "ark:|gzip -c > $dst_ali_dir/$file_name"
#} &
#done
#wait

# Make CTC decode graph
aslp_scripts/ctc/make_ctc_graph.sh --mono data/lang_test \
    $dst_ali_dir $dst_ali_dir/graph

