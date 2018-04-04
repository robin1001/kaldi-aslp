
all:

include ../aslp.mk
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

BINFILES = aslp-acc-tree-stats-phone-median aslp-acc-tree-stats-phone-mean \
           aslp-compile-questions-phone \
	       aslp-acc-tree-stats-cd-phone-equal aslp-cluster-kmeans-cd-phone-test \
		   aslp-acc-tree-stats-cd-phone-kmeans aslp-acc-tree-stats-cd-phone-viterbi \
		   aslp-make-ctc-transducer aslp-make-h3-transducer \
		   aslp-tree-bind-info aslp-convert-ali \
           aslp-acc-tree-stats-phone-mean-per-frame \
           aslp-copy-vector-from-matrix \
           aslp-ali-to-pdf aslp-ali-to-matrix aslp-ali-minus-one \
           aslp-txt-to-matrix aslp-matrix-to-txt \
           aslp-wav-noise \
           aslp-extract-transition-to-pdf

TESTFILES = 

OBJFILES = 

ADDLIBS = ../lm/kaldi-lm.a ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a \
          ../hmm/kaldi-hmm.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../feat/kaldi-feat.a \
	      ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a  ../util/kaldi-util.a \
          ../base/kaldi-base.a  ../thread/kaldi-thread.a

include ../makefiles/default_rules.mk

