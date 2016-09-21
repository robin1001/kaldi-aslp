#!/usr/bin/env python
import sys

def load_bind_info(info_file):
    table = {}
    with open(info_file) as fid:
        for line in fid.readlines():
            arr = line.strip().split()
            #assert(len(arr) == 2)
            table[arr[0]] = arr[1]
    return table


def bind_lexicon(lexicon_file, table):
    with open(lexicon_file) as fid:
        for line in fid.readlines():
            arr = line.strip().split()
            print arr[0],
            for i in range(1, len(arr)):
                assert(arr[i] in table)
                print table[arr[i]],
            print ''

# sys.argv[1] input not bind lexicon file
if len(sys.argv) != 3:
    print 'Convert syllable lexicon to bind syllable lexicon'
    print 'Usage: %s lexicon_bind_info lexicon_file'
    sys.exit(1)

table = load_bind_info(sys.argv[1])
bind_lexicon(sys.argv[2], table)
