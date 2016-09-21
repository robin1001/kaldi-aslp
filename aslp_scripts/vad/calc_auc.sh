#!/bin/bash

# Created on 2016-04-26
# Author: Binbin Zhang

stride=0.001

. parse_options.sh || exit 1;

if [ ! $# -eq 1 ]; then
    echo "Caculation AUC for vad"
    echo "Usage: $0 roc_result_file"
    exit 1;
fi

log_file=$1

grep "Thresh" $log_file | \
awk -v stride=$stride '{ 
    sum += $NF;
}
END {
    print "AUC", (sum + 1) * stride;
}' 

