KALDI_DIR = ../../../kaldi/src
EXTRA_CXXFLAGS = -I$(KALDI_DIR)

# Optinal make online part, Online depends on crf++, add it's lib and include
USE_ONLINE = true
CRF_ROOT = /home/disk1/zhangbinbin/kaldi-aslp/src/crf
CRF_LIBS = -L/home/disk1/zhangbinbin/kaldi-aslp/src/crf -lcrfpp


