#!/usr/bin/python

# Created on 2016-07-10
# Author: Zhang Binbin

import sys

for line in sys.stdin:
    arr = line.strip().split()
    assert((len(arr) - 1) % 4 == 0)
    end_time = int(arr[-2])
    intervals = ((len(arr) - 1) / 4) * 2 - 1;

    print "File type = \"ooTextFile\""
    print "Object class = \"TextGrid\""
    print "xmin = 0"
    print "xmax = %f" % (end_time / 100.0)
    print "tiers? <exists>"
    print "size = 1"
    print "item []:"
    print "\t item [1]:"
    print "\t\tclass = \"IntervalTier\""
    print "\t\tname = \"%s\"" % arr[0]
    print "\t\txmin = 0" 
    print "\t\txmax = %f" % (end_time / 100.0)
    print "\t\tintervals: size = %d " % intervals
    cur = 1
    for i in range(1, len(arr), 4):
        print "\t\tintervals [%d]:" % cur
        st, ed = int(arr[i+1]) / 100.0, int(arr[i+2]) / 100.0
        print "\t\t\txmin = %f" % st
        print "\t\t\txmax = %f" % ed
        print "\t\t\ttext = \"V\""
        cur += 1
        if i < len(arr) - 4:
            print "\t\tintervals [%d]:" % cur
            st, ed = int(arr[i+2]) / 100.0, int(arr[i+5]) / 100.0
            print "\t\t\txmin = %f" % st
            print "\t\t\txmax = %f" % ed
            print "\t\t\ttext = \"N\""
            cur += 1

