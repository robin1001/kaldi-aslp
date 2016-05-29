
all:

include ../aslp.mk
include $(KALDI_DIR)/kaldi.mk

EXTRA_CXXFLAGS += -I $(CRF_ROOT)
EXTRA_LDLIBS += $(CRF_LIBS)

TESTFILES = 

OBJFILES = online-feature-pipeline.o online-nnet-decoder.o online-endpoint.o \
           wav-provider.o tcp-server.o \
           vad.o punctuation-processor.o

LIBNAME = aslp-online

ADDLIBS = ../aslp-nnet/aslp-nnet.a \
          $(KALDI_DIR)/gmm/kaldi-gmm.a $(KALDI_DIR)/transform/kaldi-transform.a $(KALDI_DIR)/feat/kaldi-feat.a \
          $(KALDI_DIR)/matrix/kaldi-matrix.a $(KALDI_DIR)/util/kaldi-util.a $(KALDI_DIR)/base/kaldi-base.a \
          $(KALDI_DIR)/lat/kaldi-lat.a $(KALDI_DIR)/decoder/kaldi-decoder.a $(KALDI_DIR)/hmm/kaldi-hmm.a \
          $(KALDI_DIR)/thread/kaldi-thread.a $(KALDI_DIR)/ivector/kaldi-ivector.a \
          $(KALDI_DIR)/cudamatrix/kaldi-cudamatrix.a

include $(KALDI_DIR)/makefiles/default_rules.mk


