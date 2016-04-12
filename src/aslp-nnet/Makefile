
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
           warp-ctc.o

LIBNAME = aslp-nnet

ADDLIBS = ../aslp-cudamatrix/aslp-cudamatrix.a ../warp-ctc/warp-ctc.a \
          $(KALDI_DIR)/matrix/kaldi-matrix.a \
          $(KALDI_DIR)/base/kaldi-base.a \
          $(KALDI_DIR)/util/kaldi-util.a 

include $(KALDI_DIR)/makefiles/default_rules.mk

nnet-component.o: nnet-component.h \
                  nnet-activation.h nnet-various.h \
                  nnet-affine-transform.h nnet-linear-transform.h \
                  nnet-convolutional-component.h nnet-max-pooling-component.h \
                  nnet-lstm-projected-streams.h nnet-blstm-projected-streams.h \
                  nnet-batch-normalization.h nnet-io.h

data-reader.o: data-reader.h
warp-ctc.o: warp-ctc.h

