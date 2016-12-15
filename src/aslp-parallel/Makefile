all:

include ../aslp.mk
include ../kaldi.mk

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
          ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a \
          ../util/kaldi-util.a 

include ../makefiles/default_rules.mk


