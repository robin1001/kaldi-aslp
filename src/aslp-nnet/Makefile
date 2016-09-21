
all:

include ../aslp.mk
include $(KALDI_DIR)/kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

TESTFILES = nnet-randomizer-test nnet-component-test

OBJFILES = nnet-nnet.o nnet-component.o nnet-loss.o \
           nnet-randomizer.o nnet-pdf-prior.o \
           data-reader.o ctc-loss.o \
           nnet-recurrent-component.o \
           nnet-decodable.o \
           nnet-row-convolution.o

ifeq ($(USE_WARP_CTC), true)
    OBJFILES += warp-ctc.o
endif

LIBNAME = aslp-nnet

ADDLIBS = ../aslp-cudamatrix/aslp-cudamatrix.a \
          $(KALDI_DIR)/matrix/kaldi-matrix.a \
          $(KALDI_DIR)/base/kaldi-base.a \
          $(KALDI_DIR)/util/kaldi-util.a 

ifeq ($(USE_WARP_CTC), true)
    ADDLIBS += ../warp-ctc/build/libwarpctc.a
endif

include $(KALDI_DIR)/makefiles/default_rules.mk

