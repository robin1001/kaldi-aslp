
all:

include ../aslp.mk
include $(KALDI_DIR)/kaldi.mk

EXTRA_CXXFLAGS += -I $(CRF_ROOT)

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

BINFILES = aslp-audio-provider-client \
           aslp-online-energy-vad-server \
           aslp-online-nnet-vad-server \
           aslp-latgen-faster-rtf

OBJFILES = 

TESTFILES = audio-provider-test

          
ADDLIBS = ../aslp-online/aslp-online.a ../aslp-vad/aslp-vad.a ../aslp-nnet/aslp-nnet.a \
          $(CRF_FLAGS) \
          $(KALDI_DIR)/decoder/kaldi-decoder.a \
          $(KALDI_DIR)/lat/kaldi-lat.a \
          $(KALDI_DIR)/fstext/kaldi-fstext.a \
          $(KALDI_DIR)/cudamatrix/kaldi-cudamatrix.a \
          $(KALDI_DIR)/feat/kaldi-feat.a \
          $(KALDI_DIR)/transform/kaldi-transform.a \
          $(KALDI_DIR)/thread/kaldi-thread.a $(KALDI_DIR)/hmm/kaldi-hmm.a $(KALDI_DIR)/tree/kaldi-tree.a \
          $(KALDI_DIR)/matrix/kaldi-matrix.a $(KALDI_DIR)/util/kaldi-util.a $(KALDI_DIR)/base/kaldi-base.a

include $(KALDI_DIR)/makefiles/default_rules.mk

