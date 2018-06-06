#!/usr/bin/python

import sys
import re

if len(sys.argv) != 2:
    print('Usage: generate_simulation_ali.py wav.scp')
    print('the clean ali will be read from stdin')
    sys.exit(-1)

clean_ali = {}

for line in sys.stdin:
    arr = line.strip().split()
    clean_ali[arr[0]] = arr[1:]

with open(sys.argv[1]) as fid:
    for line in fid.readlines():
        arr = line.strip().split()
        find = re.search('^simulation_[0-9]+_', arr[0])
        if find:
            pos = find.span()[1]
            clean_key = arr[0][pos:]
            if clean_key in clean_ali:
                print(arr[0] + ' ' + ' '.join(clean_ali[clean_key]) + '\n')



            
