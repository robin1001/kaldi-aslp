#!/usr/bin/env python

# Created on 2016-05-18
# Author: Binbin Zhang

import sys

consonants = set(['b', 'c', 'ch', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n',
                 'p', 'q', 'r', 's', 'sh', 't', 'w', 'x', 'y', 'z', 'zh'])

def load_phone_table(table_file):
    table = {}
    with open(table_file) as fid:
        for line in fid.readlines():
            arr = line.strip().split()
            # eg. SP 0
            table[arr[1]] = arr[0]
    return table

def load_syllable_table(table_file):
    table = {}
    with open(table_file) as fid:
        for line in fid.readlines():
            arr = line.strip().split()
            table[arr[0]] = arr[1]
    return table

def ali_to_syllable(phone_ali, phone_table, syllable_table, syllable_bind_talbe):
    syllable_ali = []
    cur = 0
    while cur < len(phone_ali):
        start, end = cur, cur
        phone = phone_table[phone_ali[cur]]
        if phone in consonants:
            while cur < len(phone_ali) and phone_table[phone_ali[cur]] == phone:
                cur += 1
            phone2 = phone_table[phone_ali[cur]] 
            while cur < len(phone_ali) and phone_table[phone_ali[cur]] == phone2:
                cur += 1
            syllable = phone + phone2
            end = cur
        else:
            while cur < len(phone_ali) and phone_table[phone_ali[cur]] == phone:
                cur += 1
            syllable = phone
            end = cur
        assert(end > start)
        #print syllable,
        assert(syllable in syllable_bind_talbe)
        bind_syllable = syllable_bind_talbe[syllable]
        assert(bind_syllable in syllable_table)
        bind_syllable_id = syllable_table[bind_syllable]
        syllable_ali.extend([bind_syllable_id]*(end-start))
    assert(len(syllable_ali) == len(phone_ali))
    #print ''
    return syllable_ali


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print '''Convert phone alignment to syllable aliment
                 input from stdin, ouput to stdout'''
        print 'Usage: %s phones.txt syllable.txt syllable_bind_info_file' % sys.argv[0]
        sys.exit(1)
    
    phone_table = load_phone_table(sys.argv[1])
    syllable_table = load_syllable_table(sys.argv[2])
    syllable_bind_talbe = load_syllable_table(sys.argv[3])
    for line in sys.stdin:
        arr = line.strip().split()
        syllable_ali = ali_to_syllable(arr[1:], phone_table, syllable_table, 
                                       syllable_bind_talbe)
        print arr[0], # sentence id
        for x in syllable_ali:
            print x,
        print ''

