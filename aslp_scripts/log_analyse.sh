#!/bin/bash

# Created on 2016-01-28
# Author: zhangbinbin

sum=121
stride=5

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# != 1 ]; then
    echo "Analyse log file"
    echo "Usage: $0 log_file"
    echo "Options:"
    echo "--sum <int> # Sum of lines of progress log"
    echo "--stride <int> # Stride of each print"
    exit 1;
fi

log_file=$1

# 121: sum of progress log each iter
cat $log_file | \
grep "Progress" | \
awk -v sum=$sum -v stride=$stride '{
#if ((NR - 1) % 120 == 0) print 0;
#if ((NR - 1) % 5 == 0) print $7;
iter = int(1 + (NR - 1) / sum);
if ((NR - 1) % sum == 0) print $7;
else if ((NR - 1 - iter) % stride == 0) print $7; 
}'

