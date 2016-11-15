
all:

include ../aslp.mk
include $(KALDI_DIR)/kaldi.mk


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
         aslp-nnet-train-perutt

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
          $(KALDI_DIR)/lat/kaldi-lat.a $(KALDI_DIR)/hmm/kaldi-hmm.a \
          $(KALDI_DIR)/tree/kaldi-tree.a $(KALDI_DIR)/matrix/kaldi-matrix.a \
          $(KALDI_DIR)/util/kaldi-util.a $(KALDI_DIR)/base/kaldi-base.a 

ifeq ($(USE_WARP_CTC), true)
    ADDLIBS += ../warp-ctc/build/libwarpctc.a
endif

include $(KALDI_DIR)/makefiles/default_rules.mk

