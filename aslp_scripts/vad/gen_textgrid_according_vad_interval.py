#!/bin/env python

def GenIntervalInfo(index, xmin, xmax, text):
    return '\n'.join(['\t\tintervals [{}]:'.format(index),
                      '\t\t\txmin = {}'.format(xmin/100.0),
                      '\t\t\txmax = {}'.format(xmax/100.0),
                      '\t\t\ttext = "{}"'.format(text)]) + '\n'

import os.path
import sys
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print 'Usage: gen_textgrid_according_vad_interval.py vad_intervals_file out.TextGrid'
        sys.exit(1)

    vad_intervals = open(sys.argv[1], 'r').read().splitlines()

    text_grid_infos = []
    interval_index = 1
    voice_index = 1
    last_xmax = 0
    for interval in vad_intervals:
        interval = interval.replace('[', '').replace(']', '').replace(',', ' ').split()
        assert(len(interval) == 2 or len(interval) == 3) #or last interval
        xmin, xmax = int(interval[0]), int(interval[1])
        
        if last_xmax >= xmin:
            assert(last_xmax <= 30 + xmin)
            xmin = last_xmax
        if xmin > last_xmax + 20: #200ms
            text_grid_infos.append(GenIntervalInfo(interval_index, last_xmax, xmin, 'N'))
            interval_index += 1

        text = ''
        if voice_index == 1:
            text = '1'
        elif voice_index == len(vad_intervals):
            text = '2'
        else:
            text = 'V'
        text_grid_infos.append(GenIntervalInfo(interval_index, xmin, xmax, text))
        interval_index += 1
        voice_index += 1

        last_xmax = xmax


    output = open(sys.argv[2], 'w')

    output.write('File type = "ooTextFile"\n')
    output.write('Object class = "TextGrid"\n')

    end_interval = vad_intervals[-1].replace('[', '').replace(']', '').replace(',', ' ').split()
    end_time = int(end_interval[1])
    output.write('xmin = 0\n')
    output.write('xmax = ' + str(end_time/100.0 + 100) + '\n')
    output.write('tiers? <exists>\n')
    output.write('size = 1\n')
    output.write('item []:\n')
    output.write('\titem [1]:\n')
    output.write('\t\tclass = "IntervalTier"\n')
    output.write('\t\tname = "{}"\n'.format(os.path.splitext(os.path.basename(sys.argv[2]))[0]))
    output.write('\t\txmin = 0\n')
    output.write('\t\txmax = ' + str(end_time/100.0 + 100) + '\n')
    output.write('\t\tintervals: size = ' + str(interval_index - 1) + '\n')

    output.write(''.join(text_grid_infos))

    output.close()

