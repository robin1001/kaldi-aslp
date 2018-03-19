import sys
import codecs
import argparse

FLAGS = None

def read_phone_map_file(file_name):
    phone_map = {}
    with open(file_name) as fid:
        for line in fid.readlines():
            arr = line.strip().split()
            assert(len(arr) == 2)
            phone, pdf_id = arr[0], int(arr[1])
            phone_map[phone] = pdf_id
    assert('sil' in phone_map)
    assert('<gbg>' in phone_map)
    return phone_map

def read_keyword_file(file_name):
    keywords = []
    fid = codecs.open(file_name, 'r', 'utf-8')
    for line in fid.readlines():
        arr = line.strip().split()
        assert(len(arr) >= 2)
        keywords.append(arr[1:])
    fid.close()
    return keywords

# 0 for <garbage>, pdf 0 for filler state
# -1 for <epsilon>
# Format <source state> <dest state> <input id> <weight>
def build_text_graph(keywords, phone_map):
    # start to silence/filler
    print('0 1 %d' % phone_map['sil'])
    print('0 2 %d' % phone_map['<gbg>'])
    # silence to silence/filler
    print('1 1 %d' % phone_map['sil'])
    print('1 2 %d' % phone_map['<gbg>'])
    # filler to silence/filler
    print('2 1 %d' % phone_map['sil'])
    print('2 2 %d' % phone_map['<gbg>'])
    #final_state = 3
    cur_state = 3
    for keyword in keywords:
        # start/silence/filler to keyword start
        print ('0 %d %d' % (cur_state, phone_map[keyword[0]]))
        print ('1 %d %d' % (cur_state, phone_map[keyword[0]]))
        print ('2 %d %d' % (cur_state, phone_map[keyword[0]]))
        for i in range(0, len(keyword)-1):
            print ('%d %d %d' % (cur_state, cur_state, phone_map[keyword[i]]))
            print ('%d %d %d' % (cur_state, cur_state+1, phone_map[keyword[i+1]]))
            cur_state += 1
        print('%d %d %d' % (cur_state, cur_state, phone_map[keyword[-1]]))
        # make it last state of keyword to final state
        print('%d' % cur_state)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate text graph info')
    parser.add_argument('--filler_prob', required=False, type=float, default=1.0,
                        help='start with filler prob, for tuning filler/keyword prob')
    parser.add_argument('phone_map_file', help='phone map file')
    parser.add_argument('keyword_file', help='keyword file')

    FLAGS = parser.parse_args()

    phone_map = read_phone_map_file(FLAGS.phone_map_file)
    #print(phone_map)
    keywords = read_keyword_file(FLAGS.keyword_file)
    build_text_graph(keywords, phone_map)

