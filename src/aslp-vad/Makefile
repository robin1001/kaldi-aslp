
all:

include ../aslp.mk
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)
#EXTRA_CXXFLAGS += --std=c++11

TESTFILES = roc-test

OBJFILES = vad.o energy-vad.o nnet-vad.o feature-spectrum.o

LIBNAME = aslp-vad

ADDLIBS = ../aslp-nnet/aslp-nnet.a ../aslp-cudamatrix/aslp-cudamatrix.a \
          ../hmm/kaldi-hmm.a \
          ../gmm/kaldi-gmm.a \
          ../feat/kaldi-feat.a \
	  ../tree/kaldi-tree.a \
          ../matrix/kaldi-matrix.a \
          ../util/kaldi-util.a \
          ../base/kaldi-base.a

include ../makefiles/default_rules.mk
