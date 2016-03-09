
all:

include ../aslp.mk
include $(KALDI_DIR)/kaldi.mk


EXTRA_CXXFLAGS += -Wno-sign-compare
LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

BINFILES = aslp-nnet-info aslp-nnet-init \
         aslp-nnet-insert \
         aslp-nnet-train-simple \
         aslp-nnet-forward \
         aslp-nnet-train-lstm-streams \
         aslp-nnet-train-frame

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

OBJFILES =

TESTFILES =

ADDLIBS = ../aslp-nnet/aslp-nnet.a $(KALDI_DIR)/cudamatrix/kaldi-cudamatrix.a \
          $(KALDI_DIR)/lat/kaldi-lat.a $(KALDI_DIR)/hmm/kaldi-hmm.a \
          $(KALDI_DIR)/tree/kaldi-tree.a $(KALDI_DIR)/matrix/kaldi-matrix.a \
          $(KALDI_DIR)/util/kaldi-util.a $(KALDI_DIR)/base/kaldi-base.a 

include $(KALDI_DIR)/makefiles/default_rules.mk
