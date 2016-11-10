all:

include ../aslp.mk
include $(KALDI_DIR)/kaldi.mk

CXX = mpicxx

#EXTRA_CXXFLAGS += -I/home/binbinzhang/tools/openmpi/include

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

TESTFILES = reduce-barrier-test

OBJFILES = bsp-worker.o easgd-server.o easgd-worker.o bmuf-worker.o \
		   asgd-worker.o asgd-server.o

BINFILES = 

LIBNAME = aslp-parallel

ADDLIBS = ../aslp-nnet/aslp-nnet.a \
	      ../aslp-cudamatrix/aslp-cudamatrix.a \
          $(KALDI_DIR)/matrix/kaldi-matrix.a \
          $(KALDI_DIR)/base/kaldi-base.a \
          $(KALDI_DIR)/util/kaldi-util.a 

include $(KALDI_DIR)/makefiles/default_rules.mk


