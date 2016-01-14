
all:

KALDI_DIR = /home/disk1/zhangbinbin/kaldi/src

include $(KALDI_DIR)/kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)
EXTRA_CXXFLAGS = -I$(KALDI_DIR)

TESTFILES = nnet-randomizer-test nnet-component-test

OBJFILES = nnet-nnet.o nnet-component.o nnet-loss.o \
           nnet-randomizer.o

LIBNAME = alsp-nnet

ADDLIBS = $(KALDI_DIR)/cudamatrix/kaldi-cudamatrix.a \
          $(KALDI_DIR)/matrix/kaldi-matrix.a \
          $(KALDI_DIR)/base/kaldi-base.a \
          $(KALDI_DIR)/util/kaldi-util.a 

include $(KALDI_DIR)/makefiles/default_rules.mk

