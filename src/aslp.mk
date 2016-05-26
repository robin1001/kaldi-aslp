KALDI_DIR = ../../../kaldi/src
EXTRA_CXXFLAGS = -I$(KALDI_DIR)

# Optinal make online part
# Online depends on crf++, add it's lib and include
USE_ONLINE = true
CRF_ROOT = /home/disk1/zhangbinbin/kaldi-aslp/src/crf
CRF_FLAGS = -I$(CRF_ROOT) $(CRF_ROOT)/libcrfpp.a

