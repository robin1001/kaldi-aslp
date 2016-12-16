
all:

include ../aslp.mk
include ../kaldi.mk

EXTRA_CXXFLAGS += -I $(CRF_ROOT)

TESTFILES = thread-pool-test

OBJFILES = online-feature-pipeline.o online-nnet-decoder.o online-endpoint.o \
           wav-provider.o tcp-server.o \
           vad.o punctuation-processor.o \
           decode-thread.o online-vad-feature-pipeline.o

LIBNAME = aslp-online

ADDLIBS = ../aslp-nnet/aslp-nnet.a \
          ../gmm/kaldi-gmm.a ../transform/kaldi-transform.a ../feat/kaldi-feat.a \
          ../matrix/kaldi-matrix.a ../util/kaldi-util.a ../base/kaldi-base.a \
          ../lat/kaldi-lat.a ../decoder/kaldi-decoder.a ../hmm/kaldi-hmm.a \
          ../thread/kaldi-thread.a ../ivector/kaldi-ivector.a \
          ../cudamatrix/kaldi-cudamatrix.a

include ../makefiles/default_rules.mk


