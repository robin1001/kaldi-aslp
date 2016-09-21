
all:

include ../aslp.mk
include $(KALDI_DIR)/kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)
#EXTRA_CXXFLAGS += --std=c++11

TESTFILES = roc-test

OBJFILES = vad.o energy-vad.o nnet-vad.o feature-spectrum.o

LIBNAME = aslp-vad

ADDLIBS = ../aslp-nnet/aslp-nnet.a ../aslp-cudamatrix/aslp-cudamatrix.a \
          $(KALDI_DIR)/hmm/kaldi-hmm.a \
          $(KALDI_DIR)/gmm/kaldi-gmm.a \
          $(KALDI_DIR)/feat/kaldi-feat.a \
	      $(KALDI_DIR)/tree/kaldi-tree.a \
          $(KALDI_DIR)/matrix/kaldi-matrix.a \
          $(KALDI_DIR)/util/kaldi-util.a \
          $(KALDI_DIR)/base/kaldi-base.a

include $(KALDI_DIR)/makefiles/default_rules.mk
