KALDI_DIR = ../../../kaldi/src
EXTRA_CXXFLAGS = -I$(KALDI_DIR)

# Option make warp-ctc
USE_WARP_CTC = true

# Optinal make online part, Online depends on crf++, add it's lib and include
USE_ONLINE = true

CRF_ROOT = /home/disk1/zhangbinbin/kaldi-aslp/src/crf
CRF_FLAGS = /home/disk1/zhangbinbin/kaldi-aslp/src/crf/libcrfpp.a

