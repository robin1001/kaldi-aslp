import sys
import codecs
import argparse

FLAGS = None

class Roc:
    def __init__(self, thresh):
        # true positive, false positive
        self.false_reject = 0 
        self.false_alarm = 0
        self.num_positive = 0
        self.num_negative = 0
        self.thresh = thresh
    
    def add_data(self, score, label):
        if label == 1: self.num_positive += 1
        else: self.num_negative += 1
        if (label == 1 and score < self.thresh): self.false_reject += 1
        if (label == 0 and score > self.thresh): self.false_alarm += 1

    #thresh, accuracy, true positive, false positive
    def report(self):
        total = self.num_positive + self.num_negative
        return (self.thresh, 
               1 - float(self.false_reject + self.false_alarm) / total,
               float(self.false_reject) / self.num_positive,
               float(self.false_alarm) / self.num_negative)


class RocSet:
    def __init__(self, stride):
        cur = 0.0
        self.roc_set = []
        while cur < 1.0:
            self.roc_set.append(Roc(cur))
            cur += stride

    def add_data(self, score, label):
        for roc in self.roc_set:
            roc.add_data(score, label)
    
    def report(self):
        for roc in self.roc_set:
            thresh, acc, false_reject, false_alarm = roc.report()
            print('thresh %f acc %f false_reject %f false_alarm %f' % (thresh, acc, false_reject, false_alarm))

def read_score(score_file):
    score_dict = {}
    with open(score_file) as fid:
        for line in fid.readlines():
            arr = line.strip().split()
            key = arr[0]
            max_score = float(arr[2])
            for x in arr[2:-1]: # ignore [ 0.1 0.3 ]
                max_score = max(max_score, float(x))
            score_dict[key] = max_score 
    return score_dict

def read_label(label_file):
    label_dict = {}
    with open(label_file) as fid:
        for line in fid.readlines():
            arr = line.strip().split()
            key = arr[0]
            label_dict[key] = int(arr[1])
    return label_dict

def apply_roc(stride, scores, labels):
    roc_set = RocSet(stride)
    for key in scores:
        assert(key in labels)
        roc_set.add_data(scores[key], labels[key])
    roc_set.report()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate text graph info')
    parser.add_argument('--stride', required=False, type=float, default=0.001,
                        help='step size for roc evluation')
    parser.add_argument('score_file', help='confidence score file')
    parser.add_argument('label_file', help='label file')

    FLAGS = parser.parse_args()

    scores = read_score(FLAGS.score_file)
    labels = read_label(FLAGS.label_file)
    apply_roc(FLAGS.stride, scores, labels)
