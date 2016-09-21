#!/usr/bin/python

# Created on 2016-07-10
# Author: Zhang Binbin

import os
import sys
import wave
from optparse import OptionParser

usage = '''Split long wav file according to the segment info(read from stdin)
        %prog [options]'''

parser = OptionParser(usage)

parser.add_option('--frame-shift', dest='frame_shift', 
                   help='frame shift of the segment info, in seconds. [default: %default]', 
                   default='0.01', type='float');
parser.add_option('--out-dir', dest='out_dir', 
                   help='out dir that stores the splited segments. [default: %default]', 
                   default='split', type='string');

option, args = parser.parse_args()
if len(args) != 0:
    parser.print_help()
    sys.exit(1)

frame_shift = option.frame_shift
out_dir = option.out_dir
if not os.path.exists(out_dir): os.makedirs(out_dir)

# line in wav_path [ start end ] ... format
for line in sys.stdin:
    arr = line.strip().split()
    wav_path = arr[0]
    assert((len(arr) - 1) % 4 == 0)
    if not os.path.exists(wav_path):
        print 'wav file %s does not exit' % wav_path
        continue
    
    wav_file = wave.open(wav_path)
    base_name = os.path.basename(wav_path)
    if base_name.endswith('.wav'): base_name = base_name[:-4]
    wav_data = wav_file.readframes(wav_file.getnframes())
    num_channel = wav_file.getnchannels()
    sample_width = wav_file.getsampwidth()
    frame_rate = wav_file.getframerate()
    cur = 0
    for i in range(1, len(arr), 4):
        start, end = int(arr[i+1]), int(arr[i+2])
        #print start, end
        st = int(start * num_channel * sample_width * frame_rate * frame_shift)
        ed = int(end * num_channel * sample_width * frame_rate * frame_shift)
        assert(ed < len(wav_data))
        seg_file = out_dir + '/' + base_name + ('_%04d' % cur) + '.wav'
        seg_wav = wave.open(seg_file, 'wb')
        seg_wav.setnchannels(num_channel)
        seg_wav.setsampwidth(sample_width)
        seg_wav.setframerate(frame_rate)
        seg_wav.writeframes(wav_data[st:ed])
        seg_wav.close()
        cur += 1
    wav_file.close()
    print '%s split into %d segments ' % (wav_path, cur)


