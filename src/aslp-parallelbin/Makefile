all:

include ../aslp.mk
include ../kaldi.mk

CXX = mpic++

EXTRA_CXXFLAGS += -I/home/binbinzhang/tools/openmpi/include

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

TESTFILES =

BINFILES = aslp-nnet-train-server aslp-nnet-train-frame-worker \
		   aslp-nnet-train-lstm-stream-worker \
		   aslp-nnet-train-lc-blstm-streams-worker

ADDLIBS = ../aslp-parallel/aslp-parallel.a \
	  ../aslp-nnet/aslp-nnet.a \
	  ../aslp-cudamatrix/aslp-cudamatrix.a \
          ../lat/kaldi-lat.a \
	  ../hmm/kaldi-hmm.a \
          ../tree/kaldi-tree.a \
          ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a \
          ../util/kaldi-util.a 

include ../makefiles/default_rules.mk


