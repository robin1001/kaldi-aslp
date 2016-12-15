
all:

include ../aslp.mk
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

TESTFILES = nnet-randomizer-test nnet-component-test

OBJFILES = nnet-nnet.o nnet-component.o nnet-loss.o \
           nnet-randomizer.o nnet-pdf-prior.o \
           data-reader.o \
           nnet-recurrent-component.o \
           nnet-decodable.o \
           nnet-row-convolution.o

ifeq ($(USE_CTC), true)
    OBJFILES += ctc-loss.o
endif

ifeq ($(USE_WARP_CTC), true)
    OBJFILES += warp-ctc.o
endif

LIBNAME = aslp-nnet

ADDLIBS = ../aslp-cudamatrix/aslp-cudamatrix.a \
          ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a \
          ../util/kaldi-util.a 

ifeq ($(USE_WARP_CTC), true)
    ADDLIBS += ../warp-ctc/build/libwarpctc.a
endif

include ../makefiles/default_rules.mk

