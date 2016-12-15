
all:

include ../aslp.mk
include ../kaldi.mk

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
          ../decoder/kaldi-decoder.a \
          ../lat/kaldi-lat.a \
          ../fstext/kaldi-fstext.a \
          ../cudamatrix/kaldi-cudamatrix.a \
          ../feat/kaldi-feat.a \
          ../transform/kaldi-transform.a \
          ../thread/kaldi-thread.a ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a \
          ../matrix/kaldi-matrix.a ../util/kaldi-util.a ../base/kaldi-base.a

include ../makefiles/default_rules.mk

