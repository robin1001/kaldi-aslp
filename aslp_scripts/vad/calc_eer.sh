#!/bin/bash

# Created on 2016-04-26
# Author: Binbin Zhang

if [ ! $# -eq 1 ]; then
    echo "Caculation EER for vad"
    echo "Usage: $0 roc_result_file"
    exit 1;
fi

log_file=$1

grep "Thresh" $log_file | \
awk '{ 
    tn=1-$NF; 
    fp=$(NF-2); 
    if (tn>fp) diff=tn-fp; 
    else diff=fp-tn; 
    print "EER", (fp+tn)/2, $5, $6, $7, $8, "FP", fp, "TN", tn, "DIFF", diff; 
}' | \
sort -k12 -n | head


