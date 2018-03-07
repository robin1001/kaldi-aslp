
all:

include ../aslp.mk
include ../kaldi.mk

LDFLAGS += $(CUDA_LDFLAGS)
LDLIBS += $(CUDA_LDLIBS)
#EXTRA_CXXFLAGS += --std=c++11

TESTFILES = 

OBJFILES = fsm.o 

LIBNAME = aslp-kws

ADDLIBS = 

include ../makefiles/default_rules.mk
