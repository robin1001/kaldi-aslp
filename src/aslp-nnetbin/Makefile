
all:

include ../aslp.mk
include ../kaldi.mk


EXTRA_CXXFLAGS += -Wno-sign-compare -fopenmp
LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)


BINFILES = aslp-nnet-info aslp-nnet-init \
         aslp-nnet-insert \
         aslp-nnet-train-simple \
         aslp-nnet-forward aslp-nnet-forward-skip \
         aslp-nnet-train-lstm-streams aslp-nnet-train-blstm-streams \
         aslp-nnet-train-frame aslp-nnet-train-frame-mimo \
         aslp-nnet-forward-mimo \
         aslp-nnet-train-blstm-parallel \
         aslp-nnet-train-lstm-streams-skip \
         aslp-nnet-copy \
         aslp-nnet-train-mse \
         aslp-nnet-convert-to-standard \
		 aslp-nnet-train-blstm-streams-lc \
		 aslp-nnet-forward-blstm-lc \
         aslp-nnet-train-perutt \
		 aslp-nnet-dot

#        nnet-train-perutt \
#        nnet-train-mmi-sequential \
#        nnet-train-mpe-sequential \
#        nnet-train-lstm-streams nnet-train-blstm-streams \
#        rbm-train-cd1-frmshuff rbm-convert-to-nnet \
#        nnet-forward nnet-copy nnet-info nnet-concat \
#        transf-to-nnet cmvn-to-nnet nnet-initialize \
#        nnet-kl-hmm-acc nnet-kl-hmm-mat-to-component \
#	feat-to-post paste-post train-transitions \
#	cuda-gpu-available

ifeq ($(USE_CTC), true)
    BINFILES += aslp-nnet-train-ctc aslp-nnet-train-ctc-streams
endif

ifeq ($(USE_WARP_CTC), true)
    BINFILES += aslp-nnet-train-warp-ctc-streams
endif

OBJFILES =

TESTFILES =

ADDLIBS = ../aslp-nnet/aslp-nnet.a ../aslp-cudamatrix/aslp-cudamatrix.a \
          ../lat/kaldi-lat.a ../hmm/kaldi-hmm.a \
          ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a \
          ../util/kaldi-util.a ../base/kaldi-base.a 

ifeq ($(USE_WARP_CTC), true)
    ADDLIBS += ../warp-ctc/build/libwarpctc.a
endif

include ../makefiles/default_rules.mk

