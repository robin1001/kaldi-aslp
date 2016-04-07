
all:

include ../aslp.mk
include $(KALDI_DIR)/kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)

BINFILES = aslp-acc-tree-stats-phone-median aslp-acc-tree-stats-phone-mean \
           aslp-compile-questions-phone \
		   aslp-acc-tree-stats-cd-phone-equal aslp-cluster-kmeans-cd-phone-test \
		   aslp-acc-tree-stats-cd-phone-kmeans aslp-acc-tree-stats-cd-phone-viterbi \
		   aslp-make-ctc-transducer aslp-make-h3-transducer \
		   aslp-tree-bind-info aslp-convert-ali \
           aslp-acc-tree-stats-phone-mean-per-frame \
           aslp-copy-vector-from-matrix

TESTFILES = 

OBJFILES = 

ADDLIBS = $(KALDI_DIR)/lm/kaldi-lm.a $(KALDI_DIR)/decoder/kaldi-decoder.a $(KALDI_DIR)/lat/kaldi-lat.a \
          $(KALDI_DIR)/hmm/kaldi-hmm.a $(KALDI_DIR)/transform/kaldi-transform.a $(KALDI_DIR)/gmm/kaldi-gmm.a \
	      $(KALDI_DIR)/tree/kaldi-tree.a $(KALDI_DIR)/matrix/kaldi-matrix.a  $(KALDI_DIR)/util/kaldi-util.a \
          $(KALDI_DIR)/base/kaldi-base.a  $(KALDI_DIR)/thread/kaldi-thread.a

include $(KALDI_DIR)/makefiles/default_rules.mk

