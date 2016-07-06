#!/usr/bin/env python
import sys

consonants = set(['b', 'c', 'ch', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n',
                 'p', 'q', 'r', 's', 'sh', 't', 'w', 'x', 'y', 'z', 'zh'])


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'Convert phone lexicon file to syllable lexicon file'
        print 'Usgae %s phone_lexicon syllable_lexicon' % sys.argv[0]
        sys.exit(1)

    syllable_table = {}
    with open(sys.argv[1]) as phone_fid, open(sys.argv[2], 'w') as syllable_fid:
        for line in phone_fid.readlines():
            arr = line.strip().split()
            #print arr[0],
            syllable_fid.write(arr[0])
            i = 1
            while i < len(arr):
                if arr[i] in consonants:
                    assert(i+1 <len(arr))
                    syllable = arr[i] + arr[i+1]
                    syllable_fid.write(' ' + syllable)
                    syllable_table[syllable] = arr[i] + ' ' +  arr[i+1]
                    i += 2
                else:
                    #print arr[i],
                    syllable_fid.write(' ' + arr[i])
                    syllable_table[arr[i]] = arr[i]
                    i += 1
            #print ''
            syllable_fid.write('\n')

    sort_syllable = sorted(syllable_table.iteritems(), key=lambda d:d[0]) 
    for x in sort_syllable:
        assert(len(x) == 2)
        print x[0], x[1]

    
