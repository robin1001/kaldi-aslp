#!/bin/bash

# Author: zhangbinbin
# Created on 2016-02-27

if [ $# != 1 ]; then 
    echo "Analyse mpi 2-card log file"
    echo "Usage: $0 log_dir"
    echo "eg: $0 exp/dnn_fbank_sync10/log"
    exit 1;
fi

dir=$1

for x in $dir/iter*.tr.log.1.0; do
    echo 0
    grep -H Progress $x | awk '{print $7}'
done
