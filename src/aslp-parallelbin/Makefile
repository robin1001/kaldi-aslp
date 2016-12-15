all:

include ../aslp.mk
include $(KALDI_DIR)/kaldi.mk

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
          $(KALDI_DIR)/lat/kaldi-lat.a \
		  $(KALDI_DIR)/hmm/kaldi-hmm.a \
          $(KALDI_DIR)/tree/kaldi-tree.a \
          $(KALDI_DIR)/matrix/kaldi-matrix.a \
          $(KALDI_DIR)/base/kaldi-base.a \
          $(KALDI_DIR)/util/kaldi-util.a 

include $(KALDI_DIR)/makefiles/default_rules.mk


