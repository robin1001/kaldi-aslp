
all:

include ../aslp.mk
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)
#EXTRA_CXXFLAGS += --std=c++11

TESTFILES = roc-test

BINFILES = aslp-ali-to-sil aslp-select-frames \
           aslp-eval-gmm-vad aslp-eval-nn-vad aslp-apply-nn-vad \
		   aslp-eval-vad-boundary aslp-eval-nn-vad-boundary \
		   aslp-apply-nn-vad-segment aslp-apply-nn-vad-frame \
		   aslp-compute-spectrum-feats aslp-apply-energy-vad

ADDLIBS = ../aslp-vad/aslp-vad.a ../aslp-nnet/aslp-nnet.a ../aslp-cudamatrix/aslp-cudamatrix.a \
          ../hmm/kaldi-hmm.a \
          ../gmm/kaldi-gmm.a \
          ../feat/kaldi-feat.a \
	  ../tree/kaldi-tree.a \
          ../matrix/kaldi-matrix.a \
          ../util/kaldi-util.a \
          ../base/kaldi-base.a

include ../makefiles/default_rules.mk
