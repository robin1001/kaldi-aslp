#!/usr/bin/env python
import sys

def load_count(count_file):
    table = {}
    with open(count_file) as fid:
        for line in fid.readlines():
            arr = line.strip().split()
            assert(len(arr) == 2)
            table[arr[0]] = int(arr[1])
    return table

def bind_syllable(table, thresh = 50): 
    mapping = {}
    for x in table:
        if table[x] < thresh:
            #find & bind
            syllable = x[:-1]
            max_count, max_syllable  = 0, None
            for i in range(1, 6): 
                syllable_with_tone = syllable + str(i)
                if syllable_with_tone in table and table[syllable_with_tone] > max_count:
                    max_count, max_syllable = table[syllable_with_tone], syllable_with_tone

            if max_syllable != None:
                mapping[x] = max_syllable
            else:    
                print x, "Not bind"
        else:
            mapping[x] = x
    return mapping

def print_table(table):
    for (key,val) in table.items():
        print key, val, (key == val)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print '''Bind low frequency syllable to corresponding no tune high frequency'
                 syllable
              '''
        print 'Usage %s syllable_count_file' % sys.argv[0]
    table = load_count(sys.argv[1])
    bind_table = bind_syllable(table)
    print_table(bind_table)


