import sys
import codecs
import argparse

FLAGS = None

def read_roc(roc_file):
    false_alarms = [] 
    false_rejects = []
    with open(roc_file) as fid:
        for line in fid.readlines():
            arr = line.strip().split()
            false_alarms.append(float(arr[0]))
            false_rejects.append(float(arr[1]))
    return false_alarms, false_rejects

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate text graph info')
    parser.add_argument('roc_files', nargs='+', help='tag:roc file list, eg lstm_cell128:test.roc')
    FLAGS = parser.parse_args()

    labels = []
    fa_array = []
    fr_array = []
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w']
    for roc_file in FLAGS.roc_files:
        arr = roc_file.split(':')
        assert(len(arr) == 2)
        labels.append(arr[0])
        false_alarms, false_rejects = read_roc(arr[1])
        fa_array.append(false_alarms)
        fr_array.append(false_rejects)

    try:
        import numpy as np
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')
        plt.xlabel('False Alarm')    
        plt.ylabel('False Reject')
        #plt.grid(True)
        plt.axis([0, 0.1, 0, 0.2])
        for i in range(len(labels)):
            plt.plot(fa_array[i], fr_array[i], ('%s-' % colors[i]), label=labels[i])
        plt.savefig('roc.pdf')
    except:
        print('Error in plot')

