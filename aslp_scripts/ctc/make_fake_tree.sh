#!/bin/sh

# Created on 2016-03-04
# Author: Binbin Zhang

if [ $# != 2 ]; then
    echo "Make mono state fake tree for CTC training"
    echo "Usage: $0 num_state out_tree_file"
    echo "eg: $0 215 fake.tree"
    exit -1;
fi

num_state=$1
tree_file=$2

{
echo "ContextDependency 1 0 ToPdf TE 0 $[$num_state+1] ( NULL"
for ((i = 0; i < $num_state; i++)); do
    echo "TE -1 1 ( CE $i )"
done
echo ")"
echo "EndContextDependency"
} > $tree_file




