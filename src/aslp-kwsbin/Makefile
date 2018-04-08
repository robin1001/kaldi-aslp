
all:

include ../aslp.mk
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)
#EXTRA_CXXFLAGS += --std=c++11

TESTFILES = 

BINFILES = aslp-kws-convert-phone-ali \
           aslp-fst-init \
           aslp-fst-info \
           aslp-fst-to-dot \
           aslp-kws-score

ADDLIBS = ../aslp-kws/aslp-kws.a ../aslp-nnet/aslp-nnet.a ../aslp-cudamatrix/aslp-cudamatrix.a \
          ../feat/kaldi-feat.a \
          ../matrix/kaldi-matrix.a \
          ../util/kaldi-util.a \
          ../base/kaldi-base.a

include ../makefiles/default_rules.mk
